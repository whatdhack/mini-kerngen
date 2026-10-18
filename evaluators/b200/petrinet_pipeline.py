"""Tier 2: a timed Petri net of a B200 GEMM mainloop, and the whole-problem latency it implies.

The net follows CUTLASS's SM100 pipelines (media/docs/cpp/pipeline.md): a TMA -> MMA pipeline
over shared-memory stages, and an MMA -> epilogue pipeline over TMEM accumulator buffers.

  Places                                     Transitions (fire as early as their inputs allow)
  P_smem_free    free stages (k tokens)      T_tma       free stage -> loaded stage; the TMA engine
  P_smem_loaded  loaded stages, timestamped              streams one stage at a time, then the GMEM
  P_tensor_core  the tensor core (1 token)               latency passes before the mbarrier completes
  P_acc_free     free TMEM accumulators      T_mma       loaded stage + tensor core (+ an accumulator
                 (1 or 2 tokens)                         on a tile's first K step) -> the stage goes
                                                         back to P_smem_free when the MMA ends
                                                         (tcgen05.commit arrives on its mbarrier)
                                             T_epilogue  after a tile's last MMA: TMEM -> registers ->
                                                         global; frees the accumulator when done

A non-persistent kernel runs one tile per CTA, so the next tile's loads wait for the previous
CTA's epilogue. A persistent kernel keeps loading across tiles, and with two accumulator buffers
the epilogue of one tile overlaps the mainloop of the next.

Times are in SM clock cycles. The bubble ratio is the fraction of the schedule in which the tensor
core sits idle; the bottleneck says what it was waiting for. All memory and latency constants come
from hw.py and are uncalibrated (docs/PLAN.md, Phase 7).
"""
import math
from dataclasses import dataclass
from typing import Dict

from . import hw


@dataclass
class PipelineConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    problem_m: int = 4096
    problem_n: int = 4096
    problem_k: int = 4096
    fmt: str = "bf16"
    sparse: bool = False
    out_bytes: int = 2            # bytes per output element written by the epilogue
    warp_specialized: bool = True  # a dedicated producer issues the loads
    tma: bool = True               # bulk tensor copies rather than per-thread loads
    persistent: bool = False
    acc_buffers: int = 1           # TMEM accumulators per CTA
    ctas_per_sm: int = 1


@dataclass
class PipelineResult:
    total_cycles: float
    bubble_cycles: float
    bubble_ratio: float
    achieved_tflops: float
    predicted_latency_us: float
    bottleneck: str
    stall_cause: str             # "", "latency" (too few stages) or "bandwidth"
    t_load: float                # cycles to stream one stage
    t_mma: float                 # cycles of tensor-core work per stage
    required_stages: int         # stages that hide the load latency: ceil(load / MMA) + 1
    l2_hit: float
    tiles: int
    waves: int
    last_wave_fill: float
    tiles_per_cta: int


def l2_hit_fraction(cfg: PipelineConfig) -> float:
    """Share of tile loads served by L2: each A and B tile is reused by the CTAs that share its
    row or column, so DRAM sees the operands about L2_REFETCH times rather than once per tile."""
    elem = hw.ELEMENT_BYTES[cfg.fmt]
    tiles = math.ceil(cfg.problem_m / cfg.tile_m) * math.ceil(cfg.problem_n / cfg.tile_n)
    loaded = tiles * cfg.problem_k * (cfg.tile_m + cfg.tile_n) * elem
    compulsory = (cfg.problem_m + cfg.problem_n) * cfg.problem_k * elem
    return max(0.0, min(0.98, 1.0 - hw.L2_REFETCH * compulsory / max(loaded, 1.0)))


def simulate(cfg: PipelineConfig) -> PipelineResult:
    elem = hw.ELEMENT_BYTES[cfg.fmt]
    share = max(1, cfg.ctas_per_sm)
    rate = hw.FLOPS_PER_CLK[cfg.fmt] * (2 if cfg.sparse else 1) / share  # resident CTAs share the tensor core
    hit = l2_hit_fraction(cfg)
    bandwidth = min(hw.L2_BYTES_PER_CLK, hw.dram_bytes_per_clk() / max(1.0 - hit, 1e-3)) / share
    t_load = (cfg.tile_m + cfg.tile_n) * cfg.tile_k * elem / bandwidth
    t_mma = 2.0 * cfg.tile_m * cfg.tile_n * cfg.tile_k / rate
    issue = 0.0 if (cfg.warp_specialized and cfg.tma) else hw.ISSUE_OVERHEAD_CLK
    t_epilogue = cfg.tile_m * cfg.tile_n * cfg.out_bytes / (hw.STORE_BYTES_PER_CLK / share) + hw.GMEM_LATENCY_CLK
    steps = math.ceil(cfg.problem_k / cfg.tile_k)

    tiles = math.ceil(cfg.problem_m / cfg.tile_m) * math.ceil(cfg.problem_n / cfg.tile_n)
    slots = hw.SMS * share
    waves = math.ceil(tiles / slots)
    last_wave_fill = (tiles - (waves - 1) * slots) / slots

    stages = max(1, cfg.stages)
    accs = max(1, cfg.acc_buffers if cfg.persistent else 1)
    smem_free = [0.0] * stages   # when each stage's token returns to P_smem_free
    acc_free = [0.0] * accs      # when each accumulator returns to P_acc_free
    tma_engine = 0.0             # the TMA engine streams one stage at a time
    tensor_core = 0.0            # when the tensor core token is free again
    cta_start = 0.0              # non-persistent: a tile's loads start when its CTA starts
    busy = 0.0
    for tile in range(waves):    # each CTA slot works through `waves` tiles
        acc = tile % accs
        for step in range(steps):
            stage = (tile * steps + step) % stages
            load_start = max(tma_engine, smem_free[stage], cta_start)       # T_tma
            tma_engine = load_start + t_load
            loaded = tma_engine + hw.GMEM_LATENCY_CLK
            mma_start = max(loaded, tensor_core, acc_free[acc] if step == 0 else 0.0) + issue  # T_mma
            tensor_core = mma_start + t_mma
            smem_free[stage] = tensor_core
            busy += t_mma
        epilogue_end = tensor_core + t_epilogue                           # T_epilogue
        acc_free[acc] = epilogue_end
        if not cfg.persistent:
            cta_start = epilogue_end  # the next CTA on this SM starts after this one retires
            tensor_core = max(tensor_core, epilogue_end)
    total = max(acc_free)
    bubbles = max(0.0, total - busy)
    ratio = bubbles / total if total else 0.0

    required = math.ceil((t_load + hw.GMEM_LATENCY_CLK) / max(t_mma + issue, 1e-9)) + 1
    if ratio < 0.15:
        bottleneck, cause = "COMPUTE_SATURATED", ""
    elif t_load > t_mma + issue:
        bottleneck, cause = "TMA_MEMORY_BOUND", "bandwidth"
    else:
        bottleneck, cause = "TMA_MEMORY_BOUND", "latency" if stages < required else ""
        if not cause:
            bottleneck = "EPILOGUE_BOUND" if not cfg.persistent or accs < 2 else "COMPUTE_SATURATED"
    flops = 2.0 * cfg.problem_m * cfg.problem_n * cfg.problem_k
    seconds = total / (hw.CLOCK_GHZ * 1e9) + hw.LAUNCH_OVERHEAD_US * 1e-6
    return PipelineResult(total, bubbles, ratio, flops / seconds / 1e12, seconds * 1e6, bottleneck, cause, t_load,
                          t_mma, required, hit, tiles, waves, last_wave_fill, waves)


def as_dict(result: PipelineResult) -> Dict[str, float]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.__dict__.items()}
