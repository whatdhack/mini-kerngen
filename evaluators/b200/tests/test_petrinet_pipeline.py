"""Tier 2: the timed Petri net and the evaluator's hard gates (the design notes' test plan, corrected)."""
import pytest

from evaluators.b200 import B200Evaluator, hw
from evaluators.b200.petrinet_pipeline import PipelineConfig, simulate
from evaluators.b200.triton_adapter import spec_from_static

EVALUATOR = B200Evaluator()


def evaluate(**spec):
    return EVALUATOR.evaluate_candidate(dict({"format": "bf16"}, **spec))


@pytest.mark.parametrize("fmt,datasheet_tflops", [("bf16", 2250), ("e4m3", 4500), ("nvfp4", 9000), ("tf32", 1100)])
def test_per_sm_rates_reproduce_the_datasheet(fmt, datasheet_tflops):
    assert abs(hw.peak_tflops(fmt) - datasheet_tflops) / datasheet_tflops < 0.1


def test_one_stage_on_a_memory_bound_shape_is_mostly_bubbles():
    # One row of tiles: B is streamed with no reuse, so loads dominate
    result = simulate(PipelineConfig(128, 128, 64, stages=1, problem_m=128, problem_n=16384, problem_k=4096))
    assert result.bubble_ratio > 0.5 and result.bottleneck == "TMA_MEMORY_BOUND"


def test_more_stages_never_add_bubbles_and_saturate():
    # One CTA per SM throughout: otherwise a stage can cost occupancy (3 stages of 32 KB fit two
    # CTAs per SM, 4 stages fit one), which is a different effect
    ratios = [evaluate(tile_shape=(128, 128, 64), pipeline_stages=s, ctas_per_sm=1)["metrics"]["bubble_ratio"]
              for s in range(2, 7)]
    assert all(later <= earlier for earlier, later in zip(ratios, ratios[1:])), ratios
    assert ratios[-1] < 0.15


def test_zero_bubble_invariant_needs_one_stage_beyond_the_latency_ratio():
    result = evaluate(tile_shape=(128, 256, 64), pipeline_stages=4)["metrics"]
    assert result["required_stages"] == 4 and result["bottleneck"] == "COMPUTE_SATURATED"
    assert evaluate(tile_shape=(128, 256, 64), pipeline_stages=3)["metrics"]["bottleneck"] == "TMA_MEMORY_BOUND"


def test_tmem_column_limit_counts_every_accumulator_buffer():
    assert "TMEM column limit (512) exceeded" in evaluate(tile_shape=(128, 600, 64), pipeline_stages=2)["error_reason"]
    assert not evaluate(tile_shape=(128, 300, 64), pipeline_stages=2, acc_buffers=2)["passed_validation"]
    assert evaluate(tile_shape=(128, 256, 64), pipeline_stages=2, acc_buffers=2)["passed_validation"]


def test_shared_memory_limit_is_227_kb_per_cta():
    # 256 KB fails; the notes' 128 KB limit would have rejected this standard 192 KB config
    assert "Shared memory capacity" in evaluate(tile_shape=(256, 256, 64), pipeline_stages=4)["error_reason"]
    result = evaluate(tile_shape=(128, 256, 64), pipeline_stages=4)
    assert result["passed_validation"] and result["metrics"]["smem_bytes_used"] // 1024 == 192


def test_k_tile_must_be_a_multiple_of_the_instruction_k_and_scale_block():
    assert "MMA K" in evaluate(tile_shape=(128, 128, 48), pipeline_stages=4, format="e4m3")["error_reason"]
    assert evaluate(tile_shape=(128, 128, 64), pipeline_stages=4, format="nvfp4")["passed_validation"]


def test_reproduces_the_order_of_the_kernelwiki_tcgen05_tutorial():
    # KernelWiki (technique-pipeline-stages, pattern-pipeline-stalls), measured on B200:
    # no pipeline 695 < pipelined 940 < warp-specialized 1209 < persistent 1476 TFLOPS.
    # The model is uncalibrated; it has to get the order right, not the numbers.
    steps = [dict(pipeline_stages=1, warp_specialized=False),
             dict(pipeline_stages=4, warp_specialized=False),
             dict(pipeline_stages=4, warp_specialized=True),
             dict(pipeline_stages=4, warp_specialized=True, persistent=True, acc_buffers=2)]
    tflops = [evaluate(tile_shape=(128, 256, 64), **s)["metrics"]["achieved_tflops"] for s in steps]
    assert tflops == sorted(tflops) and len(set(tflops)) == 4, tflops


def test_result_has_the_notes_shape():
    result = evaluate(tile_shape=(128, 128, 64), pipeline_stages=3)
    assert set(result) == {"passed_validation", "error_reason", "fitness_score", "metrics"}
    assert 0.0 < result["fitness_score"] <= 1.0
    for key in ("achieved_tflops", "bubble_ratio", "bottleneck", "smem_bytes_used", "tmem_columns_used"):
        assert key in result["metrics"]


def test_triton_adapter_reads_tiles_and_features():
    static = {"tile_m": 128, "tile_n": 256, "tile_k": 64, "num_stages": 3, "shared_bytes": 150_000,
              "tmem_columns": 512, "warp_specialized": True, "tma": True}
    spec = spec_from_static("NUM_SMS = 148\nx = tl.float8e4nv", "matmul_v2", static)
    assert spec["tile_shape"] == (128, 256, 64) and spec["format"] == "e4m3"
    assert spec["persistent"] and spec["acc_buffers"] == 2 and spec["smem_bytes"] == 150_000
    assert spec_from_static("", "vectorsum_v2", static) is None
    assert spec_from_static("", "matmul_v2", dict(static, tile_k=None)) is None
