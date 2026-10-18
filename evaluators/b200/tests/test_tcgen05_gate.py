"""Tier 1: the tcgen05 functional gate (the design notes' test plan, corrected to the model's real API)."""
import pytest

pytest.importorskip("tcgen05_model")

from evaluators.b200 import B200Evaluator  # noqa: E402
from evaluators.b200 import functional_tcgen05 as tcgen05  # noqa: E402


def test_tf32_dot_matches_the_model_readme():
    # eight 1.0 x 1.0 products accumulate to 8.0f
    result = tcgen05.verify_dot([0x3F800000] * 8, [0x3F800000] * 8, "tf32")
    assert result.success and result.output_words == (0x41000000,)


def test_bf16_dot_matches_the_expected_word():
    result = tcgen05.verify_dot([1.0] * 8, [1.0] * 8, "bf16")
    assert result.success and result.output_words == (0x41000000,) and result.value == 8.0


def test_nvfp4_block_scaled_dot_is_exact_on_representable_values():
    # 64 E2M1 values in four 16-element blocks, each block with its own UE4M3 scale
    a = [1.0, 2.0, -3.0, 0.5] * 16
    b = [0.5, 1.5, 2.0, -6.0] * 16
    result = tcgen05.verify_dot(a, b, "nvfp4", scale_a=[1.0, 2.0, 0.5, 1.0], scale_b=[1.0] * 4)
    assert result.success and result.value == result.expected == -99.0


def test_illegal_sparse_selector_is_rejected():
    # 0x7 encodes indices (3, 1): out of order, so not a legal 2:4 selector
    result = tcgen05.verify_dot([1.0] * 32, [1.0] * 32, "e4m3", sparse_metadata=0x77777777)
    assert not result.success and "illegal sparse selector 0x7" in result.error_msg


def test_three_set_bits_can_be_a_legal_selector():
    # The notes' "three ones in a 4-bit field" rule is wrong: 0xD selects indices (1, 3)
    result = tcgen05.verify_dot([1.0] * 32, [1.0] * 32, "e4m3", sparse_metadata=0xDDDDDDDD)
    assert result.success and result.value == 16.0


def test_illegal_descriptors_are_rejected_by_the_model():
    assert "scale_vec::4X" in tcgen05.check_descriptor("e2m1", scaling="ue4m3", scale_vec=2)
    assert tcgen05.check_descriptor("e2m1", scaling="ue4m3", scale_vec=4) is None
    assert not tcgen05.verify_dot([1.0], [1.0], "fp3").success


def test_format_check_predicts_which_formats_meet_the_gemm_tolerance():
    # matmul_v2's tolerance: rtol=2e-2, atol=2e-1
    assert tcgen05.format_check("f16", 2e-2, 2e-1)["ok"] is True
    assert tcgen05.format_check("e4m3", 2e-2, 2e-1)["ok"] is False
    assert tcgen05.format_check("nvfp4", 2e-2, 2e-1)["ok"] is False


def test_evaluator_runs_tier1_on_sample_tiles():
    spec = {"tile_shape": (128, 128, 64), "pipeline_stages": 4, "format": "e4m3", "is_sparse": True,
            "sparse_metadata": 0x77777777, "sample_a_tile": [1.0] * 64, "sample_b_tile": [1.0] * 64}
    result = B200Evaluator().evaluate_candidate(spec)
    assert not result["passed_validation"] and "illegal sparse selector" in result["error_reason"]
    spec.update(sparse_metadata=0x44444444)
    result = B200Evaluator().evaluate_candidate(spec)
    assert result["passed_validation"] and result["metrics"]["tier1"]["value"] == 32.0
