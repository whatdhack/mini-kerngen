"""Map what the checks measured to KernelWiki's symptom vocabulary (docs/PLAN.md §6.5). No LLM.

Sources, most specific first: the B200 evaluator's Petri net (why the tensor core waits), the
compiled kernel's static metrics (spills, registers, occupancy, tensor-core use), and the kind of
problem when nothing more specific is known. Every symptom comes with the measurement behind it,
which goes into the prompt as the reason the page was retrieved.
"""
from typing import Any, Dict, List, Optional, Tuple

# KernelWiki's symptom vocabulary (pattern pages' `symptoms:` fields); each resolves to a pattern page
VOCABULARY = ("compute-bound", "high-memory-throughput", "load-imbalance", "low-compute-utilization", "low-occupancy",
              "low-sm-utilization", "low-tensor-core-utilization", "memory-bound", "pipeline-stalls",
              "register-pressure", "register-spilling", "tail-effect", "wave-quantization")


def symptoms_for(static: Dict[str, Any], evaluator: Optional[Dict[str, Any]], gemm: bool) -> List[Tuple[str, str]]:
    """(symptom, why) pairs for a kernel, most specific first, without duplicates."""
    found: List[Tuple[str, str]] = []
    metrics = (evaluator or {}).get("metrics") or {}
    pipeline = metrics.get("pipeline") or {}
    bottleneck = metrics.get("bottleneck")
    if bottleneck in ("TMA_MEMORY_BOUND", "SMEM_CAPACITY_CONSTRAINED", "EPILOGUE_BOUND"):
        ratio = metrics.get("bubble_ratio", 0.0)
        if metrics.get("stall_cause") == "bandwidth":
            found.append(("memory-bound", f"Petri net: loading a stage takes {pipeline.get('t_load', 0):.0f} cycles "
                                          f"against {pipeline.get('t_mma', 0):.0f} of MMA; tensor core idle {ratio:.0%}"))
        else:
            found.append(("pipeline-stalls", f"Petri net: tensor core idle {ratio:.0%} of the time, "
                                             f"{metrics.get('required_stages')} stages needed to hide load latency"))
    elif bottleneck == "COMPUTE_SATURATED" and (evaluator or {}).get("fitness_score", 1.0) < 0.6:
        found.append(("compute-bound", f"Petri net: {metrics.get('achieved_tflops')} TFLOPS predicted, "
                                       f"{evaluator['fitness_score']:.0%} of peak"))
    if metrics.get("waves", 0) and metrics["waves"] <= 4 and metrics.get("last_wave_fill", 1.0) < 0.5:
        found.append(("wave-quantization", f"{metrics['waves']} waves of tiles, the last {metrics['last_wave_fill']:.0%} full"))
    if static:
        if static.get("spill_bytes", 0) > 0:
            found.append(("register-spilling", f"{static['spill_bytes']} B of registers spill to local memory"))
        if static.get("registers", 0) >= 224:
            found.append(("register-pressure", f"{static['registers']} registers per thread"))
        if static.get("occupancy") is not None and static["occupancy"] < (0.125 if gemm else 0.5):
            found.append(("low-occupancy", f"theoretical occupancy {static['occupancy']:.0%}"))
        if gemm and not (static.get("tcgen05") or static.get("wgmma") or static.get("mma_sync")):
            found.append(("low-tensor-core-utilization", "no tensor-core instructions in the compiled kernel"))
        if not gemm and static.get("wide_load_frac") is not None and static["wide_load_frac"] < 1.0:
            found.append(("memory-bound", f"{static['wide_load_frac']:.0%} of global loads are 16 bytes wide"))
    found.append(("compute-bound", "GEMM-like problem; no more specific symptom measured") if gemm
                 else ("memory-bound", "bandwidth-bound problem; no more specific symptom measured"))
    seen, ordered = set(), []
    for symptom, why in found:
        if symptom not in seen:
            seen.add(symptom)
            ordered.append((symptom, why))
    return ordered
