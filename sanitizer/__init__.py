"""Anti-gaming sanitizer.

static_policy checks a candidate's source before it is compiled; the dynamic audits (reference
first, perturbed inputs, poisoned outputs, monkeypatch fingerprints, timing sanity) run inside the
harness subprocess, see harness.py. TRAPS.md lists the patterns both are meant to catch.
"""
from .static_policy import PolicyResult, check

__all__ = ["PolicyResult", "check"]
