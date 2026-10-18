"""Correctness on the CPU for every candidate, through the Triton interpreter.

The static score cannot tell whether a kernel computes the right answer, and the GPU run happens
only every Nth generation, if there is a GPU at all. With TRITON_INTERPRET=1, Triton executes
@triton.jit kernels on the CPU, so every candidate can be run at small shapes on any machine with
the same harness and audits as the GPU run (harness.py), minus the timing. Its relative error is
the candidate's DEVIATION.

The interpreter models neither tensor-core rounding nor races nor time, so a pass here is not a
pass on the GPU. A kernel that uses something the interpreter cannot run is reported as
"unsupported" and goes on unchecked, which is not a failure. A host wrapper that allocates on a
hardcoded "cuda" device is a failure: the prompt asks for allocations on the inputs' device.
"""
from typing import Any, Dict, List, Optional

import harness

DEFAULT_TIMEOUT_S = 120
CUDA_ONLY = ("Torch not compiled with CUDA", "Found no NVIDIA driver", "CUDA driver", "no CUDA GPUs")
INTERPRETER_GAPS = ("interpreter", "NotImplementedError", "inline_asm", "not supported in TRITON_INTERPRET")


def classify(result: Dict[str, Any]) -> str:
    """ok, needs-cuda (the wrapper hardcodes the GPU), unsupported (interpreter gap) or failed."""
    if result.get("ok"):
        return "ok"
    error = result.get("error", "")
    if any(m in error for m in CUDA_ONLY):
        return "needs-cuda"
    if any(m in error for m in INTERPRETER_GAPS):
        return "unsupported"
    return "failed"


def check_variant(code: str, problem_id: str, entries: Optional[List[str]] = None,
                  timeout_s: float = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    """{ok, status, rel_error?, entry?, error?, audit?}. Never raises."""
    if not harness.supported(problem_id):
        return {"ok": False, "status": "unsupported", "error": f"no harness for problem {problem_id}"}
    result = harness.run(code, problem_id, device="cpu", small=True, time_it=False, timeout_s=timeout_s,
                         entries=entries, extra_env={"TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""})
    result["status"] = classify(result)
    return result
