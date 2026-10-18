"""The gates in the evolution loop: each cheat is rejected at the right gate, with the right lesson."""
import json
import os
import pathlib

import pytest

import harness
import run_evolution as rev
from feedback import lessons_from_failure
from knowledge import KernelWiki

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gaming"
MATMUL = rev.PROBLEMS["matmul_v2"]
B200 = rev.HARDWARE_CATALOG["b200"]


def gates(name):
    return rev.run_gates((FIXTURES / f"{name}.py").read_text(), MATMUL, B200, rev.Gates())


def test_honest_kernel_passes_every_gate():
    error, static, found = gates("honest")
    assert error is None, error
    assert static["score"] > 0 and found["entries"] == ["solve"]
    assert found["cpu_check"]["status"] == "ok" and found["cpu_check"]["rel_error"] < 1e-2
    assert found["evaluator"]["passed_validation"] and found["evaluator"]["metrics"]["predicted_latency_us"] > 0


@pytest.mark.parametrize("name,prefix,lesson", [
    ("delegate_matmul", "sanitizer: ", "own @triton.jit kernel"),
    ("patch_allclose", "sanitizer: monkeypatching", "must not import os/sys or modify torch"),
    ("cache_by_pointer", "cpu check: stale", "Compute the result from the inputs on every call"),
    ("partial_write", "cpu check: wrong: output contains NaN", "never written"),
    ("row_major_strides", "cpu check: wrong on transposed", "real strides"),
    ("needs_cuda", "cpu check: no host wrapper", "device=x.device"),
])
def test_each_cheat_is_rejected_with_a_lesson(name, prefix, lesson):
    error, _, _ = gates(name)
    assert error and error.startswith(prefix), error
    assert any(lesson in rule for rule in lessons_from_failure(error, None)), lessons_from_failure(error, None)


def test_warn_mode_records_policy_errors_instead_of_rejecting():
    error, _, found = rev.run_gates((FIXTURES / "patch_allclose.py").read_text(), MATMUL, B200,
                                    rev.Gates(sanitize="warn"))
    assert any("monkeypatching" in w for w in found["sanitizer_warnings"])
    assert error.startswith("cpu check: monkeypatching")  # the dynamic audit still catches it


def test_llm_candidates_go_through_the_gates(monkeypatch):
    replies = iter([{"code": (FIXTURES / "honest.py").read_text(), "variantName": "v1_honest"},
                    {"code": (FIXTURES / "cache_by_pointer.py").read_text(), "variantName": "v2_cached"}])
    monkeypatch.setattr(rev, "call_mutation_agent", lambda prompt, model, spec: next(replies))
    seed = rev.KernelVariant("root", "v0", 0, None, "baseline", "", 386.5, 1.0, "success", {},
                             MATMUL.seed_code, 0.0)
    honest = rev.mutate_kernel_with_agent(MATMUL, B200, seed, 1, provider=rev.PROVIDERS["crusoe"])
    assert honest.status != "failed" and honest.cpu_check["status"] == "ok" and honest.entries == ["solve"]
    cached = rev.mutate_kernel_with_agent(MATMUL, B200, seed, 2, provider=rev.PROVIDERS["crusoe"])
    assert cached.status == "failed" and cached.check_error.startswith("cpu check: stale")
    assert cached.sanitizer_warnings  # the static policy flagged the cache before the audit confirmed it


def test_predicted_latency_breaks_static_score_ties():
    def variant(score, predicted):
        v = rev.KernelVariant("x", "x", 1, None, "t", "", 1.0, 1.0, "success", {}, "", 0.0, static_score=score)
        v.evaluator = {"metrics": {"predicted_latency_us": predicted}}
        return v
    assert rev.is_better(variant(100, 80.0), variant(100, 90.0))
    assert not rev.is_better(variant(100, 95.0), variant(100, 90.0))
    assert rev.is_better(variant(99, 200.0), variant(98, 90.0))  # the static score still decides first


@pytest.mark.skipif(not KernelWiki().available, reason="external/KernelWiki is not checked out")
def test_stagnation_brings_in_kernelwiki(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for env in ("CRUSOE_API_KEY", "OPENROUTER_API_KEY", "VULTR_SERVERLESS_INFERENCE_API_KEY"):
        monkeypatch.setenv(env, "")  # the simulator: its candidates never beat the parent
    rev.run_evolution_pipeline("matmul_v2", "b200", 3, stagnation_threshold=1)
    lineage = json.loads((tmp_path / "evolution_result_matmul_v2_b200.json").read_text())["lineage_dag"]
    explored = [v for v in lineage if v["operator"] == "explore"]
    assert len(explored) == 2 and explored[0]["kernelwiki_page"] != explored[1]["kernelwiki_page"]
    assert "[KERNELWIKI" in explored[0]["prompt_feedback"]


def test_harness_timing_branch_runs_on_cpu():
    result = harness.run((FIXTURES / "honest.py").read_text(), "matmul_v2", device="cpu", small=True, time_it=True,
                         timeout_s=120, extra_env={"TRITON_INTERPRET": "1"})
    assert result["ok"] and result["latency_us"] > 0, result
