"""Static anti-gaming policy: seeds pass, cheating sources are rejected, glue is allowed."""
import json
import pathlib

import pytest

import sanitizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "gaming"

HEADER = "import torch\nimport triton\nimport triton.language as tl\n\n"
KERNEL = """
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + tl.program_id(0), tl.sum(tl.load(x_ptr + offs, mask=offs < n, other=0.0), axis=0))
"""


def policy(body, problem="vectorsum_v2"):
    return sanitizer.check(HEADER + KERNEL + body, problem)


@pytest.mark.parametrize("problem", ["matmul_v2", "vectorsum_v2", "cholesky"])
def test_naive_seeds_are_clean(problem):
    result = sanitizer.check((ROOT / "seeds" / "naive" / f"{problem}.py").read_text(), problem)
    assert result.errors == [] and result.warnings == [], result
    assert result.entries == ["custom_kernel" if problem == "cholesky" else "solve"]


def test_web_seed_is_clean():
    code = json.loads((ROOT / "seeds" / "web" / "matmul_v2.json").read_text())["code"]
    result = sanitizer.check(code, "matmul_v2")
    assert result.errors == [] and result.entries == ["matmul"], result


def test_delegation_with_an_unused_kernel():
    result = sanitizer.check((FIXTURES / "delegate_matmul.py").read_text(), "matmul_v2")
    assert any("torch.matmul(a, b)" in e for e in result.errors)
    assert any("no public host function launches" in e for e in result.errors)


def test_monkeypatching_and_banned_imports():
    assert any("assigns to torch.allclose" in e
               for e in sanitizer.check((FIXTURES / "patch_allclose.py").read_text(), "matmul_v2").errors)
    errors = sanitizer.check((FIXTURES / "reads_secrets.py").read_text(), "matmul_v2").errors
    assert any("imports os" in e for e in errors) and any("urllib" in e for e in errors)


def test_pointer_cache_is_a_warning_not_an_error():
    result = sanitizer.check((FIXTURES / "cache_by_pointer.py").read_text(), "matmul_v2")
    assert result.errors == []
    assert any("_CACHE" in w for w in result.warnings) and any("data_ptr" in w for w in result.warnings)


def test_reduction_of_the_input_is_delegation():
    result = policy("def solve(x):\n    k[(1,)](x, x, x.numel(), BLOCK=16)\n    return x.view(-1).sum().reshape(1)\n")
    assert any("library delegation" in e and "sum" in e for e in result.errors), result


def test_reduction_of_partial_results_is_glue():
    body = ("def solve(x):\n    n = x.numel()\n    partials = torch.empty(triton.cdiv(n, 1024), device=x.device)\n"
            "    k[(triton.cdiv(n, 1024),)](x, partials, n, BLOCK=1024)\n    return partials.sum().reshape(1)\n")
    result = policy(body)
    assert result.errors == [], result


def test_delegation_through_a_helper_is_followed():
    body = ("def _finish(t):\n    return torch.sum(t).reshape(1)\n\n"
            "def solve(x):\n    k[(1,)](x, x, 1, BLOCK=16)\n    y = x.contiguous()\n    return _finish(y)\n")
    result = policy(body)
    assert any("torch.sum(t)" in e for e in result.errors), result


def test_matmul_operator_on_inputs():
    body = "def solve(a, b):\n    k[(1,)](a, a, 1, BLOCK=16)\n    return a.float() @ b.float()\n"
    result = policy(body, "matmul_v2")
    assert any("multiplies its inputs" in e for e in result.errors), result


def test_computed_getattr_on_a_module():
    body = "def solve(x):\n    k[(1,)](x, x, 1, BLOCK=16)\n    return getattr(torch, 'su' + 'm')(x)\n"
    assert any("computed attribute name" in e for e in policy(body).errors)


def test_harness_names_are_reserved():
    body = "def solve(x):\n    k[(1,)](x, x, 1, BLOCK=16)\n    return x\n\ndef reference(x):\n    return x\n"
    result = policy(body)
    assert any("`reference`" in e for e in result.errors)
    assert result.entries == ["solve"]
