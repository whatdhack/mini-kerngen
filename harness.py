"""Correctness, anti-gaming audits and optional timing for one candidate, in a subprocess.

One runner serves two callers:
  * cpu_check.py runs every candidate on the CPU through the Triton interpreter
    (TRITON_INTERPRET=1), at small shapes, untimed;
  * gpu_bench.py runs every Nth generation on the GPU at the problem's real shapes, then times it.

A problem defines make_inputs(), reference(*inputs) and TOL, and may define check(got, want,
inputs) when elementwise comparison is the wrong gate, extra_cases() returning (label, inputs)
pairs that are checked but not timed, and perturb(inputs, seed) returning new input values of the
same shapes. Extra cases exist because a kernel tuned against one shape can be wrong on every
other one: the first GEMM winner hardcoded row-major strides, and the first Cholesky winner missed
the accuracy gate at n=128 while passing at n=256.

The audits, after AKO4X's cheat check and flashinfer-bench #413, run in both modes:
  * reference first: inputs, reference outputs and the perturbed references are all computed
    before the candidate is imported, and the functions the comparison relies on are
    fingerprinted, so a candidate that patches torch or triton is caught rather than believed;
  * perturbation: input values are changed in place (same pointers, so honest pointer-keyed
    caches still work) and the output must follow, which catches results cached by pointer and
    launches that stop running after the first call;
  * poison cells: torch.empty / empty_like return NaN while the candidate runs, so a kernel that
    misses part of its output fails instead of passing on leftover memory;
  * timing sanity (GPU): the do_bench median is compared with a synchronized wall-clock time, so
    work moved to a side stream cannot hide from the timer.
Known gap: a cache keyed on input *contents* still passes the perturbation and is timed warm.
The static policy (sanitizer/static_policy.py) flags module-level caches for that reason.
"""
import json
import os
import tempfile
import textwrap
import uuid
from typing import Any, Dict, List, Optional

import sandbox

MARKER = "__KERNGEN_RESULT__"
DEFAULT_WARMUP_MS = 25
DEFAULT_REP_MS = 100
# Harness names a candidate must not define: the runner looks for entry points by name
RESERVED_NAMES = frozenset(("make_inputs", "reference", "check", "extra_cases", "perturb", "TOL",
                            "get_inputs", "get_init_inputs"))

# Host wrapper the mutation prompt asks for, per problem. Discovery falls back to other
# top-level functions, but a kernel that follows this is always callable.
ENTRY_SIGNATURE = {
    "vectorsum_v2": "def solve(x: torch.Tensor) -> torch.Tensor  # x: (100_000_000,) fp32, returns the scalar sum as a 1-element tensor",
    "matmul_v2": "def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor  # a, b: (4096, 4096) fp16, returns (4096, 4096)",
    "cholesky": ("def custom_kernel(data: torch.Tensor) -> torch.Tensor  # data: (64, 256, 256) fp32 symmetric "
                 "positive definite, returns the lower-triangular L with A = L @ L.T"),
}

# Executed in the subprocess, after DEVICE (str) and SMALL (bool) are defined.
HARNESS = {
    "vectorsum_v2": '''
def make_inputs():
    torch.manual_seed(0)
    return (torch.randn(1_000_000 if SMALL else 100_000_000, device=DEVICE, dtype=torch.float32),)

def reference(x):
    return x.double().sum().reshape(1).float()

# Summed in a different order: compare on relative error against the fp64 sum
TOL = dict(rtol=2e-3, atol=5e-2)

def extra_cases():
    if not SMALL:
        return []
    torch.manual_seed(1)
    return [("n=777777, not a multiple of the block size", (torch.randn(777_777, device=DEVICE, dtype=torch.float32),))]

def perturb(inputs, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(100 + seed)
    x = inputs[0]
    return (torch.rand(x.shape, device=DEVICE, dtype=x.dtype, generator=gen) * 4 - 1,)
''',
    "matmul_v2": '''
def make_inputs():
    torch.manual_seed(0)
    n = 256 if SMALL else 4096
    return (torch.randn(n, n, device=DEVICE, dtype=torch.float16),
            torch.randn(n, n, device=DEVICE, dtype=torch.float16))

def reference(a, b):
    return a.float() @ b.float()

# fp16 inputs with fp32 accumulation over K
TOL = dict(rtol=2e-2, atol=2e-1)

def extra_cases():
    torch.manual_seed(1)
    n, (m, k, p) = (256, (128, 192, 64)) if SMALL else (4096, (1024, 2048, 512))
    a = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
    bt = torch.randn(n, n, device=DEVICE, dtype=torch.float16)
    small_a = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    small_b = torch.randn(k, p, device=DEVICE, dtype=torch.float16)
    return [("transposed b (non-contiguous)", (a, bt.t())),
            ("non-square {}x{} @ {}x{}".format(m, k, k, p), (small_a, small_b))]

def perturb(inputs, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(100 + seed)
    return tuple(torch.randn(t.shape, device=DEVICE, dtype=t.dtype, generator=gen) for t in inputs)
''',
    "cholesky": '''
# One of the leaderboard's benchmark specs (batch 64, n 256, cond 2, seed 41256)
BATCH, N, DAMPING = (4, 64, 1.0e-2) if SMALL else (64, 256, 1.0e-2)

def _spd(batch, n, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randn((batch, n, n), device=DEVICE, dtype=torch.float32, generator=gen)
    out = (a @ a.transpose(-1, -2)) / float(n)
    out.diagonal(dim1=-2, dim2=-1).add_(DAMPING)
    return (0.5 * (out + out.transpose(-1, -2))).contiguous()

def make_inputs():
    return (_spd(BATCH, N, 41256),)

def reference(A):
    return torch.linalg.cholesky_ex(A, check_errors=False).L

TOL = dict(rtol=1e-3, atol=1e-4)

def extra_cases():
    if SMALL:
        return [("batch=3 n=48", (_spd(3, 48, 41048),))]
    return [("batch=256 n=128", (_spd(256, 128, 41128),)),
            ("batch=16 n=512", (_spd(16, 512, 41512),))]

def perturb(inputs, seed):
    A = inputs[0]
    return (_spd(A.shape[0], A.shape[-1], 500 + seed),)

def check(got, want, inputs):
    """The upstream task checks properties, not elementwise agreement with one library: lower
    triangular, positive diagonal, and L @ L.T reconstructing A. A correct factorization that
    rounds differently from cuSOLVER is still correct."""
    A = inputs[0]
    if got.shape != A.shape:
        return float("inf"), "shape {} != input {}".format(tuple(got.shape), tuple(A.shape))
    if got.dtype != torch.float32:
        return float("inf"), "output dtype must be float32, got {}".format(got.dtype)
    if not torch.isfinite(got).all():
        return float("inf"), "output contains NaN or Inf"
    eps = torch.finfo(torch.float32).eps
    n = A.shape[-1]
    scale = torch.linalg.matrix_norm(A, ord=1, dim=(-2, -1)).clamp_min(torch.finfo(torch.float32).tiny)
    upper = torch.linalg.matrix_norm(torch.triu(got, diagonal=1), ord=1, dim=(-2, -1))
    if torch.any(upper > 8.0 * n * eps * scale):
        return (upper / scale).amax().item(), "output is not lower triangular"
    if torch.any(torch.diagonal(got, dim1=-2, dim2=-1) <= 0):
        return float("inf"), "output diagonal must be strictly positive"
    residual = torch.linalg.matrix_norm(got @ got.transpose(-1, -2) - A, ord=1, dim=(-2, -1))
    rel = (residual / scale).amax().item()
    allowed = 20.0 * n * eps
    return rel, None if rel <= allowed else "L @ L.T does not reconstruct A: relative residual {:.3g} > {:.3g}".format(rel, allowed)
''',
}

_RUNNER = '''
import builtins, json, sys, inspect, importlib.util, traceback, time, statistics
import torch, triton, triton.testing

CONFIG = json.load(open(sys.argv[2]))
DEVICE, SMALL = CONFIG["device"], CONFIG["small"]

{harness}

PREFERRED = ("solve", "custom_kernel", "run", "launch", "main")
RESERVED = set({reserved!r})
# What the comparison and the timing rely on; a candidate that replaces any of these is cheating
WATCHED = ("torch.allclose", "torch.isfinite", "torch.equal", "torch.matmul", "torch.triu", "torch.diagonal",
           "torch.linalg.cholesky_ex", "torch.linalg.matrix_norm", "torch.Tensor.abs", "torch.Tensor.max",
           "torch.Tensor.float", "torch.Tensor.double", "torch.Tensor.sum", "torch.Tensor.reshape",
           "torch.Tensor.copy_", "torch.Tensor.clone", "torch.Tensor.__sub__", "torch.Tensor.__matmul__",
           "torch.Tensor.__eq__", "torch.cuda.synchronize", "triton.testing.do_bench", "time.perf_counter",
           "json.dumps", "builtins.print", "builtins.open")

def resolve(path):
    obj = {{"torch": torch, "triton": triton, "time": time, "json": json, "builtins": builtins}}[path.split(".")[0]]
    for part in path.split(".")[1:]:
        obj = getattr(obj, part)
    return obj

def fingerprint():
    return {{name: id(resolve(name)) for name in WATCHED}}

def tampered(before):
    now = fingerprint()
    return [name for name in WATCHED if now[name] != before[name]]

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()

def poison(on):
    """torch.empty returns NaN while on, so unwritten output elements cannot pass."""
    torch.use_deterministic_algorithms(on, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = on

def load_candidate(path):
    spec = importlib.util.spec_from_file_location("_kerngen_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_kerngen_candidate"] = module
    spec.loader.exec_module(module)
    return module

def entry_candidates(module):
    """Host wrappers worth trying, preferred names first. @triton.jit kernels, private helpers and
    harness names are never entries, and when the static policy listed the functions that launch a
    kernel, only those are."""
    from triton.runtime.jit import JITFunction
    allowed = CONFIG.get("entries")
    funcs = {{k: v for k, v in vars(module).items()
             if inspect.isfunction(v) and not isinstance(v, JITFunction) and not k.startswith("_")
             and k not in RESERVED and (allowed is None or k in allowed)}}
    ordered = [funcs.pop(n) for n in PREFERRED if n in funcs]
    ordered += [v for k, v in funcs.items() if k.startswith(("solve", "run", "call", "launch"))]
    ordered += [v for k, v in funcs.items() if not k.startswith(("solve", "run", "call", "launch"))]
    return ordered

def invoke(fn, inputs, out_like):
    """Call a wrapper as fn(*inputs) or fn(*inputs, out); return (output, args used)."""
    errors = []
    for with_out in (False, True):
        out = torch.full_like(out_like, float("nan")) if out_like.is_floating_point() else torch.zeros_like(out_like)
        args = inputs + ((out,) if with_out else ())
        try:
            result = fn(*args)
        except TypeError as e:
            errors.append(str(e))
            continue
        if isinstance(result, torch.Tensor):
            return result, args
        if with_out:
            return args[-1], args
        errors.append("returned {{}}, not a tensor".format(type(result).__name__))
    raise TypeError("; ".join(errors) or "not callable with the problem's inputs")

def call(fn, args):
    result = fn(*args)
    return result if isinstance(result, torch.Tensor) else args[-1]

def mismatch(got, want):
    got = got.float().reshape(want.shape) if got.numel() == want.numel() else got.float()
    if got.shape != want.shape:
        return float("inf"), "shape {{}} != reference {{}}".format(tuple(got.shape), tuple(want.shape))
    if not torch.isfinite(got).all():
        return float("inf"), "output contains NaN or Inf (unwritten elements read as NaN)"
    # Relative to the largest reference magnitude: a per-element ratio explodes wherever the
    # reference is near zero, which says nothing about whether the kernel is right
    err = (got - want).abs().max()
    rel = (err / want.abs().max().clamp_min(1e-12)).item()
    ok = torch.allclose(got, want, **TOL)
    return rel, None if ok else "max error {{:.3g}} ({{:.3g}} of peak) exceeds rtol={{}} atol={{}}".format(
        err.item(), rel, TOL["rtol"], TOL["atol"])

def compare(got, want, inputs):
    sync()
    return check(got, want, inputs) if "check" in globals() else mismatch(got, want)

def first_line(e):
    return "{{}}: {{}}".format(type(e).__name__, (str(e).strip().splitlines() or [""])[0][:200])

def main(path):
    # Everything the verdict depends on exists before the candidate's code runs
    inputs = make_inputs()
    want = reference(*inputs)
    extras = [(label, case, reference(*case)) for label, case in (extra_cases() if "extra_cases" in globals() else [])]
    rounds = [(new, reference(*new)) for new in (perturb(inputs, s) for s in range(2))] if "perturb" in globals() else []
    sync()
    before = fingerprint()
    try:
        module = load_candidate(path)
    except Exception as e:
        return {{"ok": False, "error": "import: " + first_line(e)}}
    patched = tampered(before)
    if patched:
        return {{"ok": False, "audit": "patched", "error": "monkeypatching: the candidate replaced " + ", ".join(patched[:4])}}
    tried = []
    for fn in entry_candidates(module):
        poison(True)
        try:
            got, args = invoke(fn, inputs, want)
        except Exception as e:
            tried.append("{{}}: {{}}".format(fn.__name__, first_line(e)))
            continue
        rel, why = compare(got, want, inputs)
        patched = tampered(before)
        if patched:
            return {{"ok": False, "entry": fn.__name__, "audit": "patched",
                    "error": "monkeypatching: calling the kernel replaced " + ", ".join(patched[:4])}}
        if why:
            return {{"ok": False, "entry": fn.__name__, "rel_error": rel, "error": "wrong: " + why}}
        for label, case, want_case in extras:
            try:
                other, _ = invoke(fn, case, want_case)
            except Exception as e:
                return {{"ok": False, "entry": fn.__name__, "rel_error": rel,
                        "error": "raised on {{}}: {{}}".format(label, first_line(e))}}
            _, why = compare(other, want_case, case)
            if why:
                return {{"ok": False, "entry": fn.__name__, "rel_error": rel, "error": "wrong on {{}}: {{}}".format(label, why)}}
        # New values in the same buffers: the output has to follow them
        for i, (new, want_new) in enumerate(rounds):
            previous = got.detach().clone()
            for dst, src in zip(inputs, new):
                dst.copy_(src)
            got = call(fn, args)
            _, why = compare(got, want_new, inputs)
            if why:
                sync()
                stale = got.shape == previous.shape and torch.equal(got.detach(), previous)
                return {{"ok": False, "entry": fn.__name__, "rel_error": rel, "audit": "stale" if stale else "perturbed",
                        "error": ("stale: the output did not change when the input values changed (result "
                                  "caching or a skipped launch)") if stale else "wrong after the inputs changed: " + why}}
        poison(False)
        result = {{"ok": True, "entry": fn.__name__, "rel_error": rel}}
        if CONFIG["time"]:
            if DEVICE == "cuda":
                ms = triton.testing.do_bench(lambda: fn(*args), warmup=CONFIG["warmup_ms"], rep=CONFIG["rep_ms"],
                                             return_mode="median")
                samples = []
                for _ in range(10):
                    sync()
                    t0 = time.perf_counter()
                    fn(*args)
                    sync()
                    samples.append((time.perf_counter() - t0) * 1000.0)
                synced = statistics.median(samples)
                if synced > 0.1 and ms < 0.5 * synced:
                    return {{"ok": False, "entry": fn.__name__, "rel_error": rel, "audit": "timing",
                            "error": "timing: do_bench saw {{:.3g}} ms but a synchronized call takes {{:.3g}} ms; work escapes "
                                     "the measured stream".format(ms, synced)}}
            else:
                samples = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    fn(*args)
                    samples.append((time.perf_counter() - t0) * 1000.0)
                ms = statistics.median(samples)
            result["latency_us"] = ms * 1000.0
            # Caches that only switch on after warm-up show up as a stale output now
            if rounds:
                new, want_new = rounds[0]
                for dst, src in zip(inputs, new):
                    dst.copy_(src)
                _, why = compare(call(fn, args), want_new, inputs)
                if why:
                    return {{"ok": False, "entry": fn.__name__, "rel_error": rel, "audit": "stale",
                            "error": "stale after timing: " + why}}
        return result
    return {{"ok": False, "error": "no host wrapper produced the reference result"
             + (" (tried " + "; ".join(tried) + ")" if tried else " (no callable host function found)")}}

if __name__ == "__main__":
    try:
        result = main(sys.argv[1])
    except Exception as e:
        result = {{"ok": False, "error": first_line(e), "traceback": traceback.format_exc()[-800:]}}
    sys.stdout.write("{marker}" + json.dumps(result) + "\\n")
'''


def supported(problem_id: str) -> bool:
    return problem_id in HARNESS


def run(code: str, problem_id: str, device: str, small: bool, time_it: bool, timeout_s: float,
        entries: Optional[List[str]] = None, warmup_ms: float = DEFAULT_WARMUP_MS, rep_ms: float = DEFAULT_REP_MS,
        extra_env: Optional[Dict[str, str]] = None, isolate_network: bool = True) -> Dict[str, Any]:
    """Check (and with time_it, time) `code` on `device`. Never raises: failures come back as ok=False."""
    if not supported(problem_id):
        return {"ok": False, "error": f"no harness for problem {problem_id}"}
    config = {"device": device, "small": small, "time": time_it, "entries": entries,
              "warmup_ms": warmup_ms, "rep_ms": rep_ms}
    with tempfile.TemporaryDirectory() as tmp:
        candidate = os.path.join(tmp, f"candidate_{uuid.uuid4().hex}.py")
        runner = os.path.join(tmp, "runner.py")
        config_path = os.path.join(tmp, "config.json")
        with open(candidate, "w") as f:
            f.write(code)
        with open(runner, "w") as f:
            f.write(_RUNNER.format(harness=textwrap.dedent(HARNESS[problem_id]), marker=MARKER,
                                   reserved=sorted(RESERVED_NAMES)))
        with open(config_path, "w") as f:
            json.dump(config, f)
        try:
            proc = sandbox.run_python(runner, [candidate, config_path], cwd=tmp, timeout_s=timeout_s,
                                      extra_env=extra_env, isolate_network=isolate_network)
        except Exception as e:  # subprocess.TimeoutExpired, or the sandbox could not start
            if type(e).__name__ == "TimeoutExpired":
                return {"ok": False, "error": f"timed out after {timeout_s:.0f}s"}
            return {"ok": False, "error": f"could not run the harness: {e}"}
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    detail = tail[-1][:200] if tail else f"exit code {proc.returncode}"
    return {"ok": False, "error": f"harness subprocess failed: {detail}"}
