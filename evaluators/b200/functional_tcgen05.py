"""Tier 1: bit-exact B200 tcgen05 arithmetic for a format, scaling and sparsity choice, on the CPU.

Wraps tcgen05_model (github.com/sf-tensor/tcgen05), a pure-Python model validated against frozen
B200 result words. The model takes raw container words, so this module encodes floats into them
and decodes results, and it uses the model's own ValueErrors as the legality oracle for formats,
scales and sparse metadata instead of re-deriving the rules. Tile shape does not change the
arithmetic (the model's docs), so this checks formats, not M/N/K.

In the evolution loop it answers one question before any GPU time is spent: if a candidate
narrows the operands (FP8, FP4), does B200's arithmetic on the problem's own input distribution
still meet the problem's tolerance? format_check() runs that.
"""
import math
import random
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import tcgen05_model
except ImportError:  # optional at import time; requirements.txt pins it
    tcgen05_model = None

from . import hw

# Spec format name -> tcgen05_model.mma_dot keyword arguments
FORMATS: Dict[str, Dict] = {
    "tf32": dict(a_format="tf32"),
    "bf16": dict(a_format="bf16"),
    "f16": dict(a_format="f16"),
    "e4m3": dict(a_format="e4m3"),
    "e5m2": dict(a_format="e5m2"),
    "e2m1": dict(a_format="e2m1"),
    "mxf8": dict(a_format="e4m3", scaling="ue8m0", kind="mxf8f6f4", scale_vec=1),
    "mxf4": dict(a_format="e2m1", scaling="ue8m0", kind="mxf4", scale_vec=2),
    "nvfp4": dict(a_format="e2m1", scaling="ue4m3", scale_vec=4),
    "s8": dict(a_format="s8", d_type="s32"),
    "u8": dict(a_format="u8", d_type="s32"),
}
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
Number = Union[int, float]


@dataclass
class VerificationResult:
    success: bool
    error_msg: Optional[str] = None
    output_words: Optional[Tuple[int, ...]] = None
    value: Optional[float] = None     # the output word, decoded
    expected: Optional[float] = None  # fp64 dot product of the decoded operands (scales applied)
    rel_err: Optional[float] = None


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def _round_f32_bits(bits: int, drop: int) -> int:
    """Round an f32 bit pattern to nearest-even, keeping all but the low `drop` mantissa bits."""
    lsb = (bits >> drop) & 1
    return ((bits + (1 << (drop - 1)) - 1 + lsb) >> drop) << drop & 0xFFFFFFFF


def _torch_fp8(x: float, fmt: str) -> int:
    import torch
    dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    return int(torch.tensor([x], dtype=torch.float32).to(dtype).view(torch.uint8).item())


def _torch_fp8_value(word: int, fmt: str) -> float:
    import torch
    dtype = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    return float(torch.tensor([word & 0xFF], dtype=torch.uint8).view(dtype).float().item())


def encode(x: float, fmt: str) -> int:
    """A float as the raw container word tcgen05_model expects for element format `fmt`."""
    if fmt == "tf32":
        return _round_f32_bits(_f32_bits(x), 13)
    if fmt == "bf16":  # BF16 travels in the upper half of an f32 container
        return _round_f32_bits(_f32_bits(x), 16)
    if fmt == "f16":
        return struct.unpack("<H", struct.pack("<e", x))[0]
    if fmt in ("e4m3", "e5m2"):
        return _torch_fp8(x, fmt)
    if fmt == "e2m1":
        magnitude = min(range(8), key=lambda i: (abs(E2M1_VALUES[i] - abs(x)), i & 1))
        return magnitude | (0x8 if x < 0 and magnitude else 0)
    if fmt in ("s8", "u8"):
        return int(round(x)) & 0xFF
    raise ValueError(f"no encoder for {fmt!r}")


def decode(word: int, fmt: str) -> float:
    if fmt in ("tf32", "bf16", "f32"):
        return _bits_f32(word)
    if fmt == "f16":
        return struct.unpack("<e", struct.pack("<H", word & 0xFFFF))[0]
    if fmt in ("e4m3", "e5m2"):
        return _torch_fp8_value(word, fmt)
    if fmt == "e2m1":
        return (-1 if word & 0x8 else 1) * E2M1_VALUES[word & 0x7]
    if fmt == "s8":
        return float((word & 0xFF) - 256 if word & 0x80 else word & 0xFF)
    if fmt == "u8":
        return float(word & 0xFF)
    if fmt == "s32":
        word &= 0xFFFFFFFF
        return float(word - (1 << 32) if word & 0x80000000 else word)
    raise ValueError(f"no decoder for {fmt!r}")


def encode_scale(s: float, scaling: str) -> int:
    if scaling == "ue8m0":  # a power of two: 2^(word - 127)
        return max(0, min(254, round(math.log2(s)) + 127))
    return _torch_fp8(s, "e4m3") & 0x7F  # ue4m3: unsigned E4M3


def decode_scale(word: int, scaling: str) -> float:
    if scaling == "ue8m0":
        return 2.0 ** ((word & 0xFF) - 127)
    return _torch_fp8_value(word & 0x7F, "e4m3")


def check_descriptor(a_format: str, b_format: Optional[str] = None, d_type: str = "f32",
                     scaling: Optional[str] = None, scale_vec: Optional[int] = None, kind: Optional[str] = None,
                     sparse: bool = False) -> Optional[str]:
    """None when tcgen05 accepts this instruction descriptor, else the model's reason."""
    if tcgen05_model is None:
        return "tcgen05_model is not installed (pip install -r requirements.txt)"
    try:
        tcgen05_model.make_model(a_format, b_format, d_type=d_type, scaling=scaling, scale_vec=scale_vec,
                                 kind=kind, sparse=sparse)
    except ValueError as e:
        return str(e)
    return None


def verify_dot(a_tile: Sequence[Number], b_tile: Sequence[Number], fmt: str, c: float = 0.0,
               scale_a: Optional[Sequence[Number]] = None, scale_b: Optional[Sequence[Number]] = None,
               sparse_metadata: Optional[Union[int, Sequence[int]]] = None) -> VerificationResult:
    """One output element of D = A·B + C as B200 computes it.

    Operands are floats (encoded here) or raw container words (ints, used as given). Scales, for
    the block-scaled formats, are one per block of hw.SCALE_BLOCK[fmt] elements and default to 1.
    """
    if tcgen05_model is None:
        return VerificationResult(False, "tcgen05_model is not installed (pip install -r requirements.txt)")
    if fmt not in FORMATS:
        return VerificationResult(False, f"unsupported format {fmt!r} (known: {', '.join(FORMATS)})")
    kwargs = dict(FORMATS[fmt])
    elem, scaling, out = kwargs["a_format"], kwargs.get("scaling"), kwargs.get("d_type", "f32")
    if len(a_tile) != len(b_tile):
        return VerificationResult(False, f"A has {len(a_tile)} elements but B has {len(b_tile)}")
    try:
        a_words = [x if isinstance(x, int) else encode(x, elem) for x in a_tile]
        b_words = [x if isinstance(x, int) else encode(x, elem) for x in b_tile]
    except (ValueError, OverflowError) as e:
        return VerificationResult(False, f"cannot encode operands as {elem}: {e}")
    k = len(a_words)
    scales = (None, None)
    if scaling:
        blocks = math.ceil(k / hw.SCALE_BLOCK[fmt])
        scales = tuple([s if isinstance(s, int) else encode_scale(s, scaling) for s in (given or [1.0] * blocks)]
                       for given in (scale_a, scale_b))
    active = range(k)
    if sparse_metadata is not None:
        try:
            active = sorted(tcgen05_model.sparse_active_indices(
                elem, k, sparse_metadata, kind="nvfp4" if scaling == "ue4m3" else kwargs.get("kind")))
        except ValueError as e:
            return VerificationResult(False, f"illegal sparse metadata: {e}")
    c_word = int(c) & 0xFFFFFFFF if out == "s32" else _f32_bits(c)
    try:
        word = tcgen05_model.mma_dot(a_words, b_words, c_word, scale_a=scales[0], scale_b=scales[1],
                                     sparse_metadata=sparse_metadata, **kwargs)
    except ValueError as e:
        return VerificationResult(False, str(e))
    value = decode(word, out)
    expected = c
    for i in active:
        term = decode(a_words[i], elem) * decode(b_words[i], elem)
        if scaling:
            block = i // hw.SCALE_BLOCK[fmt]
            term *= decode_scale(scales[0][block], scaling) * decode_scale(scales[1][block], scaling)
        expected += term
    rel_err = abs(value - expected) / max(abs(expected), 1e-30) if math.isfinite(value) else math.inf
    return VerificationResult(True, None, (word,), value, expected, rel_err)


def format_check(fmt: str, rtol: float, atol: float, k: int = 256, samples: int = 32,
                 seed: int = 0) -> Dict[str, object]:
    """Would operands narrowed to `fmt` still meet a problem's tolerance on B200?

    Draws `samples` dot products of length `k` from a standard normal (the GEMM problems' inputs),
    computes each the way B200 does after quantizing to `fmt`, and compares with the exact fp64 dot
    of the unquantized values under allclose(rtol, atol) semantics. For block-scaled formats each
    block gets the power-of-two scale that fits its largest value.
    """
    if tcgen05_model is None:
        return {"ok": None, "error": "tcgen05_model is not installed"}
    rng = random.Random(seed)
    elem = FORMATS[fmt]["a_format"] if fmt in FORMATS else fmt
    worst, peak, ok = 0.0, 0.0, True
    for _ in range(samples):
        a = [rng.gauss(0.0, 1.0) for _ in range(k)]
        b = [rng.gauss(0.0, 1.0) for _ in range(k)]
        exact = math.fsum(x * y for x, y in zip(a, b))
        scale_a = scale_b = None
        if fmt in hw.SCALE_BLOCK:
            top = max(E2M1_VALUES) if elem == "e2m1" else 448.0
            block = hw.SCALE_BLOCK[fmt]

            def scales(v):
                return [2.0 ** math.ceil(math.log2(max(1e-30, max(abs(x) for x in v[i:i + block])) / top))
                        for i in range(0, k, block)]

            scale_a, scale_b = scales(a), scales(b)
            a = [x / scale_a[i // block] for i, x in enumerate(a)]
            b = [y / scale_b[i // block] for i, y in enumerate(b)]
        result = verify_dot(a, b, fmt, scale_a=scale_a, scale_b=scale_b)
        if not result.success:
            return {"ok": False, "error": result.error_msg}
        err = abs(result.value - exact)
        worst, peak = max(worst, err), max(peak, abs(exact))
        ok = ok and err <= atol + rtol * abs(exact)
    return {"ok": ok, "deviation": worst / max(peak, 1e-30), "max_abs_err": worst}
