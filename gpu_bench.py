"""Real-GPU correctness check and timing for generated kernels.

The static score is a proxy: it flags bad kernels but cannot rank two well-tuned ones. This
module runs a candidate on the GPU instead. harness.py builds the problem's inputs, calls the
kernel's host wrapper, checks the result against a torch reference with the same anti-gaming
audits the CPU check uses (reference first, perturbed inputs, poisoned outputs, monkeypatch
detection), and times it with `triton.testing.do_bench`.

Generated code is executed here, unlike triton_check.py, so every run happens in a separate
interpreter with a wall-clock timeout and without API keys in its environment (sandbox.py): a
kernel that hangs the GPU, crashes the driver or loops forever takes down only its own
subprocess. Network isolation is left to the CPU check, because a user namespace can hide the
GPU's device nodes on some systems.

A problem needs a harness entry in harness.py to be benchmarked; `trimul_alphafold3` has none,
because its reference seed is a stub with no defined semantics.
"""
from typing import Any, Dict, List, Optional

import harness
from harness import ENTRY_SIGNATURE  # noqa: F401  the wrappers the mutation prompt asks for

DEFAULT_TIMEOUT_S = 300
DEFAULT_WARMUP_MS = harness.DEFAULT_WARMUP_MS
DEFAULT_REP_MS = harness.DEFAULT_REP_MS


def supported(problem_id: str) -> bool:
    return harness.supported(problem_id)


def bench_variant(code: str, problem_id: str, timeout_s: int = DEFAULT_TIMEOUT_S,
                  warmup_ms: float = DEFAULT_WARMUP_MS, rep_ms: float = DEFAULT_REP_MS,
                  entries: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run `code` on the GPU. Returns {ok, latency_us?, entry?, error?, rel_error?, audit?}.

    Never raises: a crash, hang, wrong answer or failed audit comes back as ok=False with a reason.
    """
    if not supported(problem_id):
        return {"ok": False, "error": f"no GPU harness for problem {problem_id}"}
    result = harness.run(code, problem_id, device="cuda", small=False, time_it=True, timeout_s=timeout_s,
                         entries=entries, warmup_ms=warmup_ms, rep_ms=rep_ms, isolate_network=False)
    if result.get("error", "").startswith("timed out"):
        result["error"] += " on the GPU"
    return result
