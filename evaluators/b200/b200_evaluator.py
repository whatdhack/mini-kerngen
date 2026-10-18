"""The B200 evaluator: hard architectural gates, Tier 1 (tcgen05 numerics) and Tier 2 (timed Petri net).

evaluate_candidate() takes the spec dictionary from the design notes and returns the notes' result
shape. The gates use B200's real limits rather than the notes' first draft (docs/PLAN.md §2.3):
227 KB of shared memory per CTA, 512 TMEM columns counting every accumulator buffer, and a K
tile that is a multiple of the instruction's K and of the format's scale block.
"""
import math
from typing import Any, Dict, Optional

from . import functional_tcgen05, hw
from .petrinet_pipeline import PipelineConfig, as_dict, simulate

KB = 1024


class B200Evaluator:
    def __init__(self, target_frequency_ghz: float = hw.CLOCK_GHZ):
        self.clock_ghz = target_frequency_ghz

    def evaluate_candidate(self, candidate_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Score a candidate B200 kernel configuration.

        candidate_spec keys (the first four are required):
            tile_shape (M, N, K), pipeline_stages, format ("bf16", "e4m3", "nvfp4", ...), is_sparse,
            sparse_metadata, sample_a_tile, sample_b_tile (Tier 1 runs when both are given),
            problem_shape (M, N, K; default 4096^3), acc_buffers, persistent, warp_specialized, tma,
            ctas_per_sm, smem_bytes and tmem_columns (measured, e.g. from Triton's compile metadata,
            used instead of the estimates when given).
        """
        m, n, k = candidate_spec["tile_shape"]
        stages = int(candidate_spec["pipeline_stages"])
        fmt = candidate_spec.get("format", "bf16")
        sparse = bool(candidate_spec.get("is_sparse", False))
        acc_buffers = int(candidate_spec.get("acc_buffers", 1))

        def fail(reason: str, **metrics) -> Dict[str, Any]:
            return {"passed_validation": False, "error_reason": reason, "fitness_score": 0.0, "metrics": metrics}

        if fmt not in hw.FLOPS_PER_CLK:
            return fail(f"unsupported format {fmt!r} (known: {', '.join(hw.FLOPS_PER_CLK)})")
        tmem = candidate_spec.get("tmem_columns") or n * acc_buffers + (hw.SCALE_TMEM_COLUMNS if fmt in hw.SCALE_BLOCK else 0)
        if tmem > hw.TMEM_COLUMNS:
            return fail(f"TMEM column limit ({hw.TMEM_COLUMNS}) exceeded: {tmem} columns "
                        f"({n} per accumulator x {acc_buffers} buffer(s))", tmem_columns_used=tmem)
        elem = hw.ELEMENT_BYTES[fmt]
        smem = candidate_spec.get("smem_bytes") or int(stages * (m + n) * k * elem) + stages * 16  # + mbarriers
        if smem > hw.SMEM_PER_CTA:
            return fail(f"Shared memory capacity ({hw.SMEM_PER_CTA // KB} KB per CTA) exceeded: {smem / KB:.0f} KB",
                        smem_bytes_used=smem)
        step = hw.MMA_K[fmt] * (2 if sparse else 1)
        if k % step:
            return fail(f"K tile {k} is not a multiple of the {fmt} MMA K ({step} elements)")
        if fmt in hw.SCALE_BLOCK and k % hw.SCALE_BLOCK[fmt]:
            return fail(f"K tile {k} is not a multiple of the {fmt} scale block ({hw.SCALE_BLOCK[fmt]})")

        tier1 = None
        if candidate_spec.get("sample_a_tile") is not None and candidate_spec.get("sample_b_tile") is not None:
            check = functional_tcgen05.verify_dot(candidate_spec["sample_a_tile"], candidate_spec["sample_b_tile"], fmt,
                                                  sparse_metadata=candidate_spec.get("sparse_metadata") if sparse else None)
            if not check.success:
                return fail(f"tcgen05: {check.error_msg}")
            tier1 = {"value": check.value, "expected": check.expected, "rel_err": check.rel_err}

        pm, pn, pk = candidate_spec.get("problem_shape", (4096, 4096, 4096))
        smem_cap = hw.SMEM_PER_SM // max(smem, 1)
        ctas = max(1, min(int(candidate_spec.get("ctas_per_sm") or smem_cap), smem_cap, 2))
        result = simulate(PipelineConfig(
            tile_m=m, tile_n=n, tile_k=k, stages=stages, problem_m=pm, problem_n=pn, problem_k=pk, fmt=fmt,
            sparse=sparse, warp_specialized=bool(candidate_spec.get("warp_specialized", True)),
            tma=bool(candidate_spec.get("tma", True)), persistent=bool(candidate_spec.get("persistent", False)),
            acc_buffers=acc_buffers, ctas_per_sm=ctas))
        if result.bottleneck == "TMA_MEMORY_BOUND" and stages < result.required_stages:
            stage_bytes = (m + n) * k * elem
            if (stages + 1) * stage_bytes > hw.SMEM_PER_CTA:
                result.bottleneck = "SMEM_CAPACITY_CONSTRAINED"  # the stages that would hide latency do not fit
        peak = hw.peak_tflops(fmt, sparse) * self.clock_ghz / hw.CLOCK_GHZ
        metrics = {
            "achieved_tflops": round(result.achieved_tflops, 1),
            "bubble_ratio": round(result.bubble_ratio, 4),
            "bottleneck": result.bottleneck,
            "smem_bytes_used": smem,
            "tmem_columns_used": tmem,
            "predicted_latency_us": round(result.predicted_latency_us, 1),
            "required_stages": result.required_stages,
            "stall_cause": result.stall_cause,
            "waves": result.waves,
            "last_wave_fill": round(result.last_wave_fill, 3),
            "ctas_per_sm": ctas,
            "pipeline": as_dict(result),
        }
        if tier1:
            metrics["tier1"] = tier1
        return {"passed_validation": True, "error_reason": None,
                "fitness_score": round(min(1.0, result.achieved_tflops / peak), 4), "metrics": metrics}
