"""Turn a compiled Triton candidate into the B200 evaluator's spec.

Tile sizes come from the kernel's constexprs (static_perf reads them from the launch site), and
the stage count, shared memory and TMEM use from Triton's own compile metadata, which is ground
truth for the hard gates. What Triton does not report is read from the source: a persistent
kernel sizes its grid by the SM count, and narrowed operands show up as FP8 casts.
"""
import re
from typing import Any, Dict, Optional, Tuple

# Problems the evaluator understands: GEMMs, with their shape, operand format and the tolerance
# the harness checks them against (harness.py), which the Tier 1 format check reuses
GEMM_PROBLEMS = {"matmul_v2": {"shape": (4096, 4096, 4096), "format": "f16", "rtol": 2e-2, "atol": 2e-1}}
FP8_RE = re.compile(r"float8e4|float8_e4m3|fp8e4|e4m3")
FP8_E5_RE = re.compile(r"float8e5|float8_e5m2|fp8e5|e5m2")
PERSISTENT_RE = re.compile(r"NUM_SMS|num_sms|multi_processor_count|NUM_PROGRAMS")


def operand_format(code: str, default: str) -> str:
    if FP8_E5_RE.search(code):
        return "e5m2"
    if FP8_RE.search(code):
        return "e4m3"
    return default


def spec_from_static(code: str, problem_id: str, static: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The evaluator spec for a compiled candidate, or None when the problem or tiles are unknown."""
    if problem_id not in GEMM_PROBLEMS or not static:
        return None
    shape, default_fmt = GEMM_PROBLEMS[problem_id]["shape"], GEMM_PROBLEMS[problem_id]["format"]
    tile: Tuple[Any, ...] = (static.get("tile_m"), static.get("tile_n"), static.get("tile_k"))
    if not all(isinstance(t, int) and t > 0 for t in tile):
        return None
    tmem = static.get("tmem_columns") or 0
    return {
        "tile_shape": tile,
        "pipeline_stages": max(1, int(static.get("num_stages") or 1)),
        "format": operand_format(code, default_fmt),
        "is_sparse": False,
        "problem_shape": shape,
        "smem_bytes": static.get("shared_bytes") or None,
        "tmem_columns": tmem or None,
        "acc_buffers": 2 if tmem and tmem >= 2 * tile[1] else 1,
        "warp_specialized": bool(static.get("warp_specialized")),
        "tma": bool(static.get("tma")),
        "persistent": bool(PERSISTENT_RE.search(code)),
    }
