"""NVIDIA B200 (sm_100) constants for the analytical evaluator, each with its source.

Tensor-core rates are whole-GPU dense datasheet peaks divided by SM count and clock, so 148 SMs x
rate x clock reproduces the datasheet (tests/test_petrinet_pipeline.py checks this). The memory,
latency and overhead figures are uncalibrated defaults until Phase 7 of docs/PLAN.md fits them to
measurements: they make predictions useful for ranking variants of one kernel against each
other, not as absolute microseconds.
"""
SMS = 148                    # enabled SMs (KernelWiki pattern-tail-effect: "the locally verified B200 exposes 148 SMs")
CLOCK_GHZ = 1.86             # clock implied by 2.25 PFLOPS dense BF16 at 8,192 FLOP/clk/SM
SMEM_PER_SM = 228 * 1024     # sm_100 (same as sm_90)
SMEM_PER_CTA = 227 * 1024    # largest dynamic shared memory one CTA can opt into
TMEM_COLUMNS = 512           # 128 lanes x 512 columns x 32 bit = 256 KB per SM (PTX ISA, tcgen05)
TMEM_LANES = 128

# Dense tensor-core FLOP per clock per SM (2 FLOP per multiply-add); 2:4 sparsity doubles these.
# Datasheet dense peaks: TF32 1.1, BF16/F16 2.25, FP8/FP6/INT8 4.5, block-scaled FP4 9 PFLOPS.
FLOPS_PER_CLK = {"tf32": 4096, "bf16": 8192, "f16": 8192, "e4m3": 16384, "e5m2": 16384, "mxf8": 16384,
                 "e2m1": 16384,  # plain E2M1 runs as kind::f8f6f4, at the FP8 rate
                 "mxf4": 32768, "nvfp4": 32768, "s8": 16384, "u8": 16384}
# Bytes per element in global and shared memory: kind::f8f6f4 keeps E2M1 in 8-bit containers,
# while the block-scaled FP4 kinds pack two values per byte
ELEMENT_BYTES = {"tf32": 4, "bf16": 2, "f16": 2, "e4m3": 1, "e5m2": 1, "mxf8": 1, "e2m1": 1, "mxf4": 0.5,
                 "nvfp4": 0.5, "s8": 1, "u8": 1}
# K covered by one dense tcgen05.mma, in elements: 32 bytes of K per instruction
MMA_K = {"tf32": 8, "bf16": 16, "f16": 16, "e4m3": 32, "e5m2": 32, "mxf8": 32, "e2m1": 32, "s8": 32, "u8": 32,
         "mxf4": 64, "nvfp4": 64}
# Elements of K that share one scale factor in the block-scaled kinds
SCALE_BLOCK = {"mxf8": 32, "mxf4": 32, "nvfp4": 16}
SCALE_TMEM_COLUMNS = 32      # approximate TMEM reserved for SFA/SFB in block-scaled MMAs (uncalibrated)

HBM_BYTES_PER_S = 8.0e12     # datasheet; the STREAM result KernelWiki cites (doc-blackwell-microbenchmarking) is 4.14 TB/s
DRAM_EFFICIENCY = 0.8        # uncalibrated
L2_BYTES_PER_CLK = 128       # per SM, the design notes' TMA figure, used as the L2-hit ceiling (uncalibrated)
L2_REFETCH = 1.5             # DRAM bytes per compulsory byte when the working set does not stay in L2 (uncalibrated)
GMEM_LATENCY_CLK = 800       # load latency, global to shared memory (uncalibrated)
STORE_BYTES_PER_CLK = 64     # epilogue: TMEM -> registers -> global, per SM (uncalibrated)
ISSUE_OVERHEAD_CLK = 32      # per K step when loads are issued by the MMA warps or by threads instead of TMA (uncalibrated)
LAUNCH_OVERHEAD_US = 5.0     # per kernel launch (uncalibrated)


def dram_bytes_per_clk() -> float:
    """One SM's share of HBM bandwidth when every SM streams at once."""
    return HBM_BYTES_PER_S * DRAM_EFFICIENCY / (SMS * CLOCK_GHZ * 1e9)


def peak_tflops(fmt: str, sparse: bool = False) -> float:
    return FLOPS_PER_CLK[fmt] * (2 if sparse else 1) * SMS * CLOCK_GHZ * 1e9 / 1e12
