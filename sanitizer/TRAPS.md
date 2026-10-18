# Gaming patterns

Ways a candidate can look correct or fast without its kernel doing the work, and which check
catches each. The static policy (`sanitizer/static_policy.py`) runs before compile; the dynamic
audits run in the harness subprocess (`harness.py`) on the CPU for every candidate and on the GPU
when timing. Sources: AKO4X's benchmark rule and TRAPS catalogs, flashinfer-bench #413, and holes
found in MiniAVO-Liquid's own GPU harness. Fixtures for each are in `tests/fixtures/gaming/`.

| Pattern | Caught by | Fixture |
|---|---|---|
| The wrapper computes the operator with torch (`torch.matmul`, `a @ b`, `x.sum()`, `torch.linalg.cholesky`), followed through views and helper functions | static: library delegation | `delegate_matmul.py` |
| A `@triton.jit` kernel exists so the compile check passes, but nothing launches it | static: no public host function launches a kernel | `delegate_matmul.py` |
| The harness falls back to a candidate function named like its own (`reference`, `make_inputs`) | static: reserved names; only functions that launch a kernel are offered as entries | `test_entry_allowlist_excludes_other_functions` |
| `torch.allclose`, `do_bench` or the reference is replaced at import or during the call | static: assignment to an imported module's attribute; dynamic: fingerprints of 25 watched functions, checked after import and after every call | `patch_allclose.py` |
| The result is cached by input pointer and returned when the same buffers come back | static: warning (module-level container, `data_ptr()`); dynamic: inputs changed in place, the output must follow | `cache_by_pointer.py` |
| Part of the output is never written and passes on leftover memory | dynamic: `torch.empty` returns NaN while the candidate runs | `partial_write.py` |
| Row-major strides are hardcoded, so a transposed operand is read wrongly | dynamic: extra cases (transposed, non-square, sizes that are not multiples of the block) | `row_major_strides.py` |
| Work runs on a side stream the timer does not see | dynamic (GPU only): `do_bench` compared with a synchronized wall-clock time | — |
| A cache switches on only after warm-up | dynamic (GPU timing): one more perturbed call after timing | — |
| The candidate reads API keys or calls out to the network | sandbox: secrets removed from the environment, network namespace without interfaces | `reads_secrets.py` |
| Imports a kernel has no use for (`os`, `subprocess`, `socket`, `pickle`, `numpy`), `exec`, `eval`, `open` | static | `reads_secrets.py` |

**Known gap.** A cache keyed on input *contents* recomputes when values change, so it passes the
perturbation audit, and it is then timed warm on the unchanged timing inputs. The static policy
flags module-level caches as a warning for this reason; timing on alternating input sets would
close it, at the cost of comparability with timings already recorded.
