"""The CPU correctness check and its anti-gaming audits, on the seeds and on cheating fixtures.

Each case runs a real subprocess through the Triton interpreter, so this file takes ~30 s.
"""
import pathlib

import pytest

import cpu_check
import sandbox

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gaming"


def check(name, problem="matmul_v2"):
    return cpu_check.check_variant((FIXTURES / f"{name}.py").read_text(), problem)


@pytest.mark.parametrize("problem", ["matmul_v2", "vectorsum_v2", "cholesky"])
def test_naive_seeds_pass(problem):
    result = cpu_check.check_variant((ROOT / "seeds" / "naive" / f"{problem}.py").read_text(), problem)
    assert result["ok"] and result["status"] == "ok", result
    assert result["rel_error"] < 1e-2


def test_honest_kernel_passes():
    result = check("honest")
    assert result["ok"], result
    assert result["entry"] == "solve"


def test_result_cached_by_pointer_is_stale():
    result = check("cache_by_pointer")
    assert not result["ok"] and result["audit"] == "stale", result


def test_monkeypatching_at_import_is_caught():
    result = check("patch_allclose")
    assert not result["ok"] and result["audit"] == "patched", result
    assert "torch.allclose" in result["error"]


def test_unwritten_output_reads_as_nan():
    result = check("partial_write")
    assert not result["ok"] and "NaN" in result["error"], result


def test_hardcoded_strides_fail_on_a_transposed_operand():
    result = check("row_major_strides")
    assert not result["ok"] and "transposed" in result["error"], result


def test_hardcoded_cuda_device_is_reported():
    result = check("needs_cuda")
    assert result["status"] == "needs-cuda", result


def test_candidate_sees_no_secrets_and_no_network(monkeypatch):
    if not sandbox.network_isolation_available():
        pytest.skip("unprivileged network namespaces are not available here")
    monkeypatch.setenv("CRUSOE_API_KEY", "not-a-real-key")
    result = check("reads_secrets")
    assert result["ok"], result


def test_entry_allowlist_excludes_other_functions():
    # The static policy passes the functions that launch a kernel; nothing else may be called
    code = (FIXTURES / "honest.py").read_text() + "\n\ndef reference(a, b):\n    return a.float() @ b.float()\n"
    result = cpu_check.check_variant(code, "matmul_v2", entries=["nonexistent"])
    assert not result["ok"] and "no host wrapper" in result["error"], result
