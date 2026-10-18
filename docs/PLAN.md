# mini-kerngen — implementation plan

Plan for a new repo, `mini-kerngen`, built from mini-avo-liquid (`660ade6`) with the components described in
`notes_kerngen.txt`. Every claim the notes make about an external project was checked against its code:

- KernelWiki: `~/ai/KernelWiki` @ `b6b4301`
- AKO4X: `~/ai/AKO4X` @ `ce030eb`
- sf-tensor/tcgen05: `d96b821` (read, not run)

Feasibility probes ran in mini-avo-liquid's `.venv` (torch 2.14.1+cpu, Triton 3.8.0, no GPU).

Section 2 lists where the notes and the code disagree. The design in sections 5–6 follows the code.

**Status (2026-10-17).**

- **Phase 0:** the repo exists, and Crusoe Intelligence Foundry is the default provider (§3, §6.7). Still open: the
  split of `run_evolution.py`, CI, and the golden test.
- **Implemented and tested on CPU:**
  - the anti-gaming sanitizer and the CPU correctness check, with the sandbox (Phase 1);
  - KernelWiki retrieval, triggered by a stagnation counter (Phase 3);
  - the B200 evaluator: tcgen05 Tier 1, the timed Petri net and the hard gates (Phase 4), uncalibrated, so it only
    breaks ties between equal static scores.
- **Not yet:**
  - the rest of Phase 2: the deterministic exploit sweep, config-hash dedupe, and `ITERATIONS.md`;
  - the meta-reflector (Phase 6);
  - calibration (Phase 7);
  - a GPU run of the hardened harness.

## 1. Scope

mini-avo-liquid today runs one loop per generation:

1. one LLM rewrite;
2. syntax check;
3. Triton AOT compile;
4. static score;
5. optional GPU timing every N generations;
6. promote to elite if better;
7. check-derived feedback in the next prompt.

mini-kerngen keeps that loop and adds:

| Component in the notes | In mini-kerngen | Where |
|---|---|---|
| KernelWiki knowledge layer | Read hook: symptom → pattern → technique → prompt block. Write hook: run-discovered patterns become `experimental` pages | `knowledge/`, `reflector.py` |
| Ported AKO anti-gaming sanitizer | Static AST policy, plus dynamic varying-input, poison-cell and staleness audits. They run on CPU for every candidate and on the GPU when timing | `sanitizer/` |
| Model + analytical heuristic evaluator | CPU correctness through the Triton interpreter (DEVIATION); `evaluators/b200/` (tcgen05 numerics + timed Petri net) | `cpu_check.py`, `evaluators/b200/` |
| AKO trajectory & stagnation engine | `ITERATIONS.md`, config dedupe, exploit/explore switching on a stagnation counter, crossover | `trajectory.py`, `operators.py` |
| Meta-reflector (Version N+1) | Held-out check, KernelWiki-schema pages, overlay validation, winner branch | `reflector.py` |

The README's rule stays: a real GPU timing outranks every proxy. Everything new is either a cheaper, earlier gate or
a better prompt.

**One LLM call per generation.** Exploit, explore and crossover are variants of the prompt for that single call, and
the parameter sweep makes no LLM call at all.

Out of scope:

- non-Triton candidates (CUDA, CuTe);
- multi-GPU;
- replacing GPU timing as the final judge;
- model routing, judges, escalation, or fan-out across several LLM calls per generation.

## 2. What the code says vs the notes

### 2.1 KernelWiki

**What exists.** `scripts/query.py`, `get_page.py` and `validate.py` exist and need no packages (they bundle a YAML
fallback). A query takes 1.5–2.3 s; `validate.py` takes 21 s and passes cleanly at `b6b4301`.

**No `wiki/local_patterns/`.** A new directory would still be accepted, because page type comes from the frontmatter
`type:`. The problem is that the schema is stricter than "`id`, `confidence: experimental`, `metric`, `value`":

- `metric` and `value` live inside `performance_claims` entries (`gpu, dtype, shape, metric, value, source_id`);
- every page's `sources:` must resolve to an existing source page;
- tags must come from `data/tags.yaml`;
- ids carry a type prefix (`technique-`, `pattern-`, …);
- technique and kernel pages need `reproducibility: snippet`.

So a run-discovered page needs a companion source page that describes the run.

**`validate.py` has no per-file mode.** It always validates its own checkout (`REPO_ROOT` is the scripts' parent).
To validate a new page, run it inside an overlay copy (§6.6).

**Querying by symptom.** The notes' `query_kernelwiki(symptom, arch)` passes the symptom as keywords. Use
`--symptom S --type pattern` instead, without an `--architecture` filter. Pattern pages have no `architectures`
field, so `--symptom pipeline-stalls --architecture sm100` returns nothing. All 13 symptoms resolve to at least one
pattern page:

`compute-bound, high-memory-throughput, load-imbalance, low-compute-utilization, low-occupancy, low-sm-utilization,
low-tensor-core-utilization, memory-bound, pipeline-stalls, register-pressure, register-spilling, tail-effect,
wave-quantization`

**The content is mostly CUDA.** Free-text queries rank the 944 upstream PR pages first, and those are mostly
CUDA/CUTLASS/CuTe. Triton material is limited to 5 wiki pages and about 116 vLLM/SGLang/TRT-LLM/FlashInfer PR pages
(`--language triton`). Snippets compile in their own DSL; they are not verified Triton. Confidence ranges from
`verified` to `experimental`.

Labelling these "VERIFIED HARDWARE SNIPPET" in a Triton prompt invites the failure feedback.py already fights: CUDA
idioms turning into nonexistent `tl.*` calls.

**Licensing.** KDA (NVlabs/kda), which pins KernelWiki, notes that upstream `artifacts/` carry their own terms. Use a
submodule and never copy artifacts into this repo.

### 2.2 tcgen05 model

`tcgen05_model` 1.0.0 is MIT-licensed, pure Python and needs no GPU. It is validated against frozen B200 result
words.

**Signature.**
`mma_dot(a, b, c=0, *, a_format, b_format, d_type, saturate, scaling, scale_vec, kind, scale_a, scale_b,
sparse_metadata)`.
Inputs and outputs are raw container words. Only `f32_to_bits` and `bits_to_f32` are provided, so encoding floats
is our job.

**Format names** are `tf32, bf16, f16, e4m3, e5m2, e2m3, e3m2, e2m1, s8, u8`. "nvfp4", "mxf4" and "mxf8" are
combinations of format and scaling:

| Spec name | `make_model` arguments |
|---|---|
| nvfp4 | `e2m1`, `scaling="ue4m3"`, `scale_vec=4` (16-element blocks) |
| mxf4 | `e2m1`, `scaling="ue8m0"`, `kind="mxf4"`, `scale_vec=2` (32-element blocks) |
| mxf8 | `e4m3` or `e5m2`, `scaling="ue8m0"`, `kind="mxf8f6f4"`, `scale_vec=1` (32-element blocks) |
| s8 / u8 | `d_type="s32"` |

`make_model` raises `ValueError` with a specific message for every illegal combination. Use it as the legality
oracle rather than re-encoding the rules.

**Sparse metadata.** Each nibble is two 2-bit indices. The legal 2:4 selectors are
`{0x4, 0x6, 0x8, 0x9, 0xC, 0xD, 0xE}`; for TF32 they are `{0x4, 0xE}`. "Three ones in a 4-bit field" is not the
rule: 0xD and 0xE are legal. Use 0x7 or 0xB as the illegal test vector, decoded by `sparse_active_indices`.

**What Tier 1 checks.** The model's docs state that tile shape and TMEM placement do not change the arithmetic. So
Tier 1 checks formats, scales and sparsity, not M, N or K.

### 2.3 B200 constants in the evaluator spec

| Notes | Should be | Effect of the notes' value |
|---|---|---|
| SMEM up to 128 KB/SM | 228 KB/SM, 227 KB per CTA. sm_100 matches sm_90 here, and mini-avo-liquid already uses 228 | rejects standard configs: 128×256×64 bf16 × 4 stages is 192 KB |
| BF16 2,048 / FP8 4,096 / FP4 8,192 ops/cycle/SM, used as FLOPs in Δt = 2MNK/peak | dense rates ≈ 8,192 / 16,384 / 32,768 FLOP/clk/SM (2.25 / 4.5 / 9 PFLOPS ÷ 148 SMs ÷ ~1.86 GHz) | MMA modelled 4× too slow, so every candidate looks compute-bound |
| TMA ~128 B/cycle/SM | DRAM share ≈ 29 B/clk/SM at the 8 TB/s spec, and ≈ 15 at the 4.14 TB/s STREAM result KernelWiki cites (`doc-blackwell-microbenchmarking`). ~128 is an L2-hit figure | combined with the MMA error, Δt_TMA/Δt_MMA is off by more than 10× |
| 148 SMs | correct. mini-avo-liquid's catalog says 160, which needs fixing | wave count |
| TMEM 128 × 512 × 32-bit, N ≤ 512 | correct, but columns used = N × accumulator buffers + block-scale-factor columns | a persistent kernel with 2 accumulator buffers needs 2N ≤ 512 |
| K a multiple of 64 bytes (FP4) or 32 bytes (FP8) | count in elements. One MMA covers 32 bytes of K: 16 elements (bf16/f16), 32 (fp8/fp6/i8) or 64 (fp4). Block-scaled K must also be a multiple of the scale block (16 for NVFP4, 32 for MX) | wrong unit |
| zero bubble iff stages ≥ ⌈Δt_TMA/Δt_MMA⌉ | stages ≥ ⌈Δt_TMA/Δt_MMA⌉ + 1, because one stage is being consumed | off by one |

All constants go in one table, `evaluators/b200/hw.py`, each with a provenance comment. A test checks that
148 × rate × clock reproduces the datasheet peak within 10%.

### 2.4 AKO

"AKO" is AKO4X (TongmingLAIC/AKO4X). Its `sanitizer` skill wraps NVIDIA compute-sanitizer; it is not about
anti-gaming. The anti-gaming pieces the notes describe exist in three places:

- `scripts/cheat_check_modal.py`: input tensors keep their `data_ptr()` while their values are mutated in place, and
  the output hash must change every iteration (after flashinfer-bench PR #413). In AKO4X it runs only on Modal B200s.
- Per-operator `TRAPS.md` files: poison-cell tests and source greps for caches keyed on `data_ptr()`.
- The benchmark skill's "Valid solution: write your own kernel" section, which bans delegating the operator to
  cuBLAS, cuDNN or `torch.matmul`.

AKO4X has no stagnation counter; its master steers rounds through "Open directions". The 0–2 exploit / 3+ explore
rule is new design.

`templates/iterations.md` is a Summary table (Iter | Title | Score | Passed | Notes) plus `Expected:` lines written
*before* the bench, so explanations can't be retrofitted. Adopt both.

### 2.5 GPU-less correctness works (probe on this machine)

With `TRITON_INTERPRET=1`, each of these ran correctly on CPU:

| Case | Relative error | Time |
|---|---|---|
| matmul_v2 seed, 128³ / 256³ fp16 | 3e-4 / 4e-4 | 0.1–0.2 s |
| vectorsum_v2 seed, 1M fp32 | 8e-8 | 0.6 s |
| cholesky seed, batch 4, n 64 | 1.5e-7 | 1.3 s |
| GEMM with `tl.make_tensor_descriptor` + `tl.range(warp_specialize=True)`, 256³ | 4e-4 | 0.1 s |

Setting `torch.use_deterministic_algorithms(True)` together with
`torch.utils.deterministic.fill_uninitialized_memory = True` makes `torch.empty` return NaN. That gives a free
poison-cell test for outputs the wrapper allocates.

The interpreter models neither tensor-core rounding, nor races, nor time. tcgen05_model and the GPU run cover those.

### 2.6 Holes in mini-avo-liquid's `gpu_bench.py`

All four are closed in mini-kerngen's shared harness (`harness.py`, `sandbox.py`, `sanitizer/`) and covered by
CPU tests; see `sanitizer/TRAPS.md`.

1. The candidate is imported before `make_inputs()` and `reference()` run, in the same interpreter. Import-time code
   can replace `torch.allclose`, `triton.testing.do_bench` or `torch.linalg.cholesky_ex` and pass anything.
2. `do_bench(lambda: fn(*args))` reuses the same buffers. A result cache keyed on `data_ptr()` times at about 0 µs
   and becomes the elite.
3. Nothing stops `solve()` from returning `a @ b` while an unused `@triton.jit` kernel satisfies the compile check
   and static score. cuBLAS is both correct and faster than anything the search has found: 1530 µs against 1603 µs
   on the GB10.
4. The subprocess inherits the full environment, API keys included, and runs model-written code. Once CPU checks are
   on by default, that would happen for every candidate on this machine.

## 3. Bootstrap (Phase 0)

```bash
git clone ~/ai/mini-avo-liquid ~/ai/mini-kerngen        # keeps history and blame
cd ~/ai/mini-kerngen
git remote rename origin upstream                       # for cherry-picking later fixes
git remote set-url upstream https://github.com/whatdhack/mini-avo-liquid
git submodule add https://github.com/mit-han-lab/KernelWiki external/KernelWiki
git -C external/KernelWiki checkout b6b4301             # the commit validated above
mkdir -p docs && mv ../mini-kerngen-plan.md docs/PLAN.md
```

**Done (2026-10-17, on branch `phase-0`):**

- the steps above;
- the Crusoe provider as default (§6.7);
- `b200.num_sms` changed from 160 to 148;
- `--list-models` exits with a message instead of a traceback on HTTP errors;
- README and `.env.example` updated.

**Remaining:** dependencies, the split, CI, the golden test.

- **GitHub remote.** Create `whatdhack/mini-kerngen` as `origin` when ready to publish.
- **Dependencies.** torch becomes required; the CPU wheel is enough. Add
  `tcgen05-model @ git+https://github.com/sf-tensor/tcgen05@d96b821` and `pytest` to `requirements.txt`. KernelWiki
  needs nothing.
- **Split `run_evolution.py` first.** It has 811 lines. Split it mechanically before changing any behaviour:
  - `catalog.py`: hardware, problems;
  - `providers.py`;
  - `operators.py`: prompt building, agent call;
  - `engine.py`: loop, selection;
  - `run_evolution.py`: CLI only.

  The other modules keep their names, so upstream fixes to them still cherry-pick cleanly. Changes to the loop will
  need manual porting, which the gate and operator work would force anyway.
- **CI.** GitHub Actions on CPU (torch CPU wheel, triton, tcgen05-model), running `pytest`.
- **Golden test.** With no API key, `--problem matmul_v2 --generations 3` writes the same lineage JSON before and
  after the split.

## 4. Layout

```
mini-kerngen/
├── run_evolution.py        CLI
├── catalog.py              hardware (+ per-SM pipeline rates), problems, held-out task suite
├── providers.py            Crusoe Intelligence Foundry (default), OpenRouter (Liquid), Vultr
├── operators.py            mutate / crossover / exploit / explore; deterministic config sweep
├── engine.py               population, gate pipeline, selection, budget
├── trajectory.py           ITERATIONS.md, config hashing, stagnation counter
├── reflector.py            Version N+1 meta-reflector
├── cpu_check.py            TRITON_INTERPRET=1 correctness -> DEVIATION
├── sandbox.py              subprocess isolation shared by cpu_check and gpu_bench
├── sanitizer/
│   ├── static_policy.py    AST rules
│   ├── dynamic_audit.py    varying-input, poison-cell, staleness, pinned-function checks
│   └── TRAPS.md            caught patterns with AST fingerprints
├── knowledge/
│   ├── kernelwiki.py       read hook: subprocess calls, cache, prompt block
│   ├── symptoms.py         analytics -> KernelWiki symptom vocabulary
│   └── overlay.py          upstream + kb/local overlay (query + validate)
├── evaluators/b200/        as in the notes, plus hw.py and triton_adapter.py
├── kb/local/               run-discovered pages (KernelWiki schema) and their run source pages
├── external/KernelWiki/    submodule, read-only
├── triton_check.py  static_perf.py  gpu_bench.py  feedback.py  local_store.py
├── rawtree_store.py  seed_search.py  seeds/              (kept)
├── tests/                  CPU-only; evaluators/b200/tests/ as in the notes
└── runs/<run_id>/          ITERATIONS.md, lineage.json, candidates/   (gitignored)
```

## 5. Candidate pipeline, cheapest gate first

| # | Gate | Cost | Status |
|---|---|---|---|
| 1 | Python syntax | ms | existing |
| 2 | Static sanitizer policy | ms | new |
| 3 | Triton AOT compile + launch limits | ~1 s | existing |
| 4 | Static score | ms | existing |
| 5 | Evaluator: hard gates, Petri-net prediction, symptoms | < 50 ms | new |
| 6 | CPU correctness at reduced shapes + extra cases → DEVIATION | ~1–3 s | new |
| 7 | Dynamic anti-gaming audit on CPU | ~1–3 s | new |
| 8 | Format numerics (tcgen05_model), only when a candidate narrows formats | < 1 s | new |
| 9 | GPU run every N generations | s–min | existing, hardened |

**Isolation.** Gates 6, 7 and 9 execute model output, so they share `sandbox.py`, which provides:

- a subprocess run with `python -I` in a temporary cwd;
- an environment stripped of secrets (`*_API_KEY`, tokens);
- resource limits (CPU time, address space);
- no network where `unshare -rn` is available;
- a wall-clock timeout.

**Selection order:**

1. all gates passed;
2. measured latency, once the elite has one;
3. static score, with the Petri-net prediction as a tiebreak.

The prediction moves above the static score only after Phase 7 calibration (§7). A candidate that fails gates 6–8 is
never promoted and never GPU-timed.

## 6. Component design

### 6.1 CPU correctness (`cpu_check.py`)

**Small inputs.** Each gpu_bench harness gains `make_small_inputs()` and `small_extra_cases()` on CPU:

- matmul: 256³, transposed `b`, and non-square 128×192 @ 192×64;
- vectorsum: 1M elements;
- cholesky: batch 4 n 64, and batch 2 n 128.

`trimul_alphafold3` still has no harness.

**DEVIATION** is the maximum relative error per case. For cholesky it is the reconstruction residual that `check()`
already computes.

**Device contract.** A wrapper that hardcodes `device="cuda"` fails on CPU, whether for outputs or for the
`triton.set_allocator` lambda. "Allocate on the inputs' device" becomes part of the entry contract in the prompt, and
the error becomes a feedback lesson. The seeds already comply.

**Interpreter gaps.** A kernel that uses something the interpreter can't run, such as inline PTX, gets the status
`cpu-unsupported`. That is not a failure: the candidate continues to the GPU gate as it would today.

### 6.2 Anti-gaming sanitizer (`sanitizer/`)

**Static rules** run on the AST before compile. Each rejection becomes a feedback lesson.

1. **Library delegation.** The entry must not do the core operation with torch. A per-problem denylist covers
   `matmul, mm, bmm, addmm, einsum, linear, @`, `linalg.*`, `cholesky*`, and `sum` on the input. Glue such as
   `empty, zeros, reshape, view, stride, contiguous, t` is allowed. The entry must also reach a `@triton.jit` launch
   in its call graph, so an unused dummy kernel can't pass the compile gate.
2. **Monkeypatching.** Assignment, `setattr` or `delattr` on attributes of imported modules. Touching `sys.modules`,
   `builtins`, `importlib`, `gc`, `inspect` or `ctypes`.
3. **Input stripping.** Definitions of harness names (`make_inputs, reference, check, TOL, extra_cases, get_inputs,
   get_init_inputs`) are stripped and reported.
4. **Hidden state.** The rule flags module-level containers, `functools.cache` or `global` used by the entry, and
   lookups keyed on `data_ptr()` or `id()`. The dynamic audit confirms or clears each flag.
5. **I/O, processes and environment.** `open`, `pickle`, `torch.load`, `np.load`, sockets, `urllib`, `subprocess`,
   `os.system`, `os.environ` / `os.getenv`.
6. **Constant outputs.** The entry returns a literal tensor without launching a kernel.

**Dynamic audits** run in the CPU harness for every candidate, and in the GPU harness when timing.

1. **Reference first.** Inputs, reference outputs, and pinned references to `torch.allclose`,
   `triton.testing.do_bench` and the reference function are all created before the candidate is imported. Their
   identities are re-checked afterwards.
2. **Two-pass staleness with input perturbation.**
   1. Call the entry.
   2. Mutate the input *values* in place while keeping `data_ptr`: fresh SPD matrices for cholesky; normal,
      uniform-positive and scaled draws elsewhere.
   3. Call it again.

   The output hash must change and match the new reference. The audit runs three passes.
3. **Poison-cell.** Deterministic mode plus `fill_uninitialized_memory` makes wrapper-allocated outputs start as NaN,
   so partial writes show up as NaN. Where the entry takes `out`, it is pre-filled with NaN.
4. **Timing sanity (GPU only).** A second timing runs with `torch.cuda.synchronize()` inside the timed lambda. A
   `do_bench` median under half of that flags a side-stream launch.

**Anti-gaming memory** (Track 2, "Shared Brain"). Every catch is recorded twice:

- as a `kind="trap"` event in the existing feedback memory (`file` or `rawtree`), rendered as a "rejected as gaming"
  rule in later prompts;
- in `sanitizer/TRAPS.md`, with its AST fingerprint, so a recurrence is rejected statically in any later run or by any
  team that shares the repo or the RawTree table.

This memory deliberately stays out of KernelWiki. Its controlled vocabulary has no tags for gaming patterns, so its
validator would reject them. KernelWiki holds positive performance knowledge (§6.6).

### 6.3 Evaluator (`evaluators/b200/`)

The files are those the notes specify, plus `hw.py` (§2.3) and `triton_adapter.py`. The spec-dict API
`B200Evaluator.evaluate_candidate()` stays as written in the notes, with the corrected constants. Triton candidates
reach it through the adapter.

**`functional_tcgen05.py` (Tier 1)**

- **Format mapping.** Spec names map to `make_model` arguments as in §2.2.
- **Encoding.** Floats become container words through torch CPU dtypes (`bfloat16`, `float16`, `float8_e4m3fn`,
  `float8_e5m2`, then `.view` as integers). A 16-entry table covers e2m1, plus encoders for UE8M0 and UE4M3 scales.
- **Result.** `VerificationResult(success, error_msg, output_words, max_rel_err)`. The model's `ValueError`s become
  `success=False`. The output is also compared with an fp64 dot product of the dequantized operands.
- **Use in the loop.** When a candidate narrows the format (fp8 casts, `tl.dot_scaled`):
  1. draw K-vectors from the problem's own input generator;
  2. quantize them the way the candidate does;
  3. compare the model's B200-exact result with fp64.

  That predicts DEVIATION against the problem's TOL before any GPU time is spent. The check is dormant for today's
  fp16 and fp32 problems. It becomes central for an NVFP4 problem; KernelWiki holds the GPU MODE NVFP4 GEMV and GEMM
  problem definitions.
- **Tests**, corrected from the notes:
  - BF16 and TF32 dot products equal hand-computed words. The model's README example: eight TF32 1.0×1.0 products
    give `0x41000000`.
  - An NVFP4 block-scaled case matches the fp64 reference within the model's rounding.
  - Selector `0x7` gives `success=False`.
  - `e2m1` + `ue4m3` + `scale_vec=2` gives `success=False`.

**`petrinet_pipeline.py` (Tier 2)** is the notes' timed Petri net, with these changes:

- **Two pipelines.** SMEM stages (TMA → MMA, `k` stages) and accumulator stages (MMA → epilogue, 1–2 TMEM buffers),
  as CUTLASS splits them on SM100. Tile *i*'s epilogue overlaps tile *i+1*'s mainloop only with two accumulator
  buffers in a persistent kernel.
- **Shared tensor core.** It is a resource place with capacity 1 per SM, so two resident CTAs contend for it.
- **TMA timing.** Δt_TMA = bytes / effective bandwidth + GMEM base latency, where effective bandwidth =
  min(L2→SMEM rate, DRAM share ÷ (1 − L2 hit fraction)). The hit fraction comes from tile size and the `GROUP_M`
  swizzle. Base latency and the hit model are calibration parameters.
- **Grid level.** Waves = ⌈tiles ÷ (148 × CTAs per SM)⌉ give the whole-problem latency. A last wave under half full
  emits `wave-quantization`.
- **Hard gates before simulation:**
  - SMEM ≤ 227 KB, counting stages, epilogue staging and barriers;
  - TMEM columns (N × accumulator buffers + scale columns) ≤ 512;
  - BLOCK_K a multiple of the instruction K and of the scale block.
- **Outputs.** The notes' metrics, plus `predicted_latency_us` and `symptoms` in KernelWiki's vocabulary.
- **Tests**, corrected from the notes:
  - 1 stage on a memory-bound shape gives a bubble ratio > 0.5.
  - Stages 2 → 5 give a non-increasing bubble ratio that flattens at saturation.
  - 600 TMEM columns fails, 256 KB SMEM fails, and 192 KB (128×256×64 bf16 × 4 stages) passes.
  - **Ordering anchor** from real B200 numbers in KernelWiki's tcgen05 tutorial: no pipeline (695 TFLOPS) <
    pipelined (940) < warp-specialized (1209) < persistent (1476). The model must reproduce the order of the steps it
    models.

**`triton_adapter.py`** turns a compiled candidate into a `candidate_spec`:

- BLOCK_M, BLOCK_N and BLOCK_K from the constexprs (static_perf already finds M and N);
- `num_stages`, `num_warps`, the real `shared` bytes and `tmem_size` from Triton's metadata. The shared bytes are
  ground truth for the SMEM gate.
- TMA and warp specialization from static_perf;
- the format from `PROBLEM_PTR_DTYPE` plus the AST (casts, `tl.dot_scaled`).

**Other targets.** GB10 (sm_121) and H100 have no TMEM or tcgen05. In Phase 7, the same net with register
accumulators and wgmma/mma.sync rates becomes the sm90 and sm121 profiles, calibrated on measured data. Until then
the evaluator runs only for `--hardware b200`, which is the default.

### 6.4 Trajectory and stagnation (`trajectory.py`, `operators.py`)

**`runs/<run_id>/ITERATIONS.md`** uses AKO4X's layout, extended:

| Gen | Op | Variant | Expected | Gates | DEVIATION | Predicted | Static | Measured | Δ vs elite | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|

Under the table, each generation gets:

- its diff against the parent, truncated;
- the KernelWiki pages injected;
- the symptom that triggered them.

**Expected.** `expected` becomes a required JSON field that the model fills before evaluation: mechanism, affected
dimension, predicted delta. The last three rows and the elite's diff go into the next prompt, as the notes' "previous
ITERATIONS diff".

**Config hash.** The normalized kernel AST plus the resolved launch constants, which triton_check already resolves.
A duplicate is never re-evaluated or re-timed, whatever its name. The README's GEMM run spent 20 of its 30 GPU
timings on five configurations.

**Stagnation counter.** s counts consecutive generations without a new elite. `--stagnation-threshold` defaults to 3.

- **s < 3: exploit.**
  - A deterministic sweep over the elite's launch constants (BLOCK_M/N/K, num_warps, num_stages, GROUP_M). It
    covers all single-constant moves and all pairs of moves, a few dozen candidates. They are rewritten in the AST
    without an LLM call, compiled in parallel, and filtered by the evaluator. When benching, the top k are GPU-timed.
  - An LLM mutation with the ITERATIONS context.
  - Why this matters: the README's GEMM elite ended 7.6% slower than the same code at 128×256×64 / 8 warps /
    3 stages, two constants away. One pairwise sweep reaches that point.
- **s ≥ 3: explore.** The elite's symptoms lead to a KernelWiki pattern, then to a candidate technique not yet tried
  in this run, then to an LLM prompt carrying the KernelWiki block.
- **Crossover.** The elite is combined with either:
  - the best runner-up that has a different config hash or technique; or
  - a Triton example from KernelWiki (the notes' "snippet crossover").
- **Reset and stop.** A new elite resets s to 0. `--patience` stops a run that has explored every technique for its
  current symptoms without progress.

**Archive, not population.** The best few distinct candidates, distinct by config hash, are kept as an archive.
Crossover draws its second parent from it. Each generation still makes exactly one LLM call; there is no fan-out.

### 6.5 KernelWiki read hook (`knowledge/`)

`KernelWiki(root)` wraps `query.py` and `get_page.py` as subprocesses, with a 10 s timeout. Results are cached in
memory and on disk, keyed on the submodule commit and the `kb/local` contents. It exposes three calls:

- `patterns_for(symptom)` runs `--symptom S --type pattern --paths-only` and reads `candidate_techniques` from the
  frontmatter.
- `technique(id)` runs `get_page.py <id> --body-only` and trims the result to the opening summary and selected
  sections, under the token budget.
- `triton_examples(words)` queries with `--language triton` only.

**`symptoms.py`** maps analytics to symptoms deterministically, with no LLM:

| Signal | Symptom |
|---|---|
| static: `spill_bytes > 0` | register-spilling |
| static: registers ≥ 224 or occupancy below target | register-pressure, low-occupancy |
| static: GEMM without a native tensor-core op | low-tensor-core-utilization |
| Petri net: TMA-bound with a high bubble ratio | memory-bound, pipeline-stalls |
| Petri net: compute-bound but well below peak | compute-bound |
| Petri net: last wave under half full | wave-quantization, tail-effect |
| grid smaller than the SM count | low-sm-utilization |
| GPU: bandwidth ≥ 80% of measured stream | high-memory-throughput |

Correctness failures go to feedback.py and `TRAPS.md`, not KernelWiki, which is a performance knowledge base.

**Prompt block.** It replaces the notes' `[VERIFIED HARDWARE SNIPPET FROM KERNELWIKI]`:

```
[KERNELWIKI technique-pipeline-stages | confidence: source-reported | sm100, sm90]
Retrieved because: TMA-bound, bubble ratio 0.41 at 2 stages (needs >= 3).
Idea: <3-6 lines from the page>
In Triton: <num_stages=..., tl.range(..., warp_specialize=True), tl.make_tensor_descriptor(...)>
Reference (CUDA, for the idea only; write Triton): <<= 20 lines, only when no Triton example exists>
```

- **Budget.** One page per prompt, about 1.5k tokens. `lfm-2.5-2.6b` caps output at 8k tokens.
- **Guard metric.** Track the compile-failure rate of explore generations against the rest. If KernelWiki injection
  raises it, drop the CUDA reference line.

### 6.6 Meta-reflector, Version N+1 (`reflector.py`, `knowledge/overlay.py`)

1. **Name the transformation.** The reflector runs at the end of a run, or as `python reflector.py runs/<id>`. It
   names the winning transformation from the elite-vs-seed diff and from the static and spec deltas: TMA, warp
   specialization, persistence, stages, tile, swizzle. An LLM summary is optional.
2. **Held-out check.** Run the elite on the held-out task suite in `catalog.py`. For example: matmul 8192×8192×1024,
   2048×8192×8192 and 1024×1024×4096, and cholesky n = 128 and 512.
   - Promote it if it beats the seed by more than 20% on at least two-thirds of those tasks.
   - The evidence is measured when a GPU is available. Otherwise it is the Petri-net prediction plus CPU correctness,
     recorded as `evidence: predicted`.
3. **Write two pages** into `kb/local/`:
   - a `community-note` source page for the run, `blog-kerngen-run-<id>`, whose URL points at the commit;
   - a `pattern-kerngen-…` or `technique-kerngen-…` page with `confidence: experimental`, tags from
     `data/tags.yaml`, `related:` set to existing ids, `reproducibility: snippet` with the Triton code, and
     `performance_claims` that cite the run page.
4. **Validate.**
   - `overlay.py` builds `.kernelwiki_overlay/` as a hardlink copy of the submodule (`cp -al`: instant, 29 MB),
     merges `kb/local` into its `wiki/` and `sources/`, and runs the overlay's own `scripts/validate.py` (about 21 s).
   - Only pages that pass are kept.
   - The overlay is only ever added to, never edited, because its files share inodes with the submodule.
   - A Phase 6 fixture freezes a page template that passes.
5. **Commit the winner.** The winning kernel and pages go to a local `kerngen/winners` branch through a temporary
   worktree. The checkout is untouched and nothing is pushed.
6. **Run N+1 reads it back.** The read hook queries the overlay through its own scripts, so local pages are retrieved
   alongside upstream ones and marked `experimental`.
7. **Transfer check.** `scripts/transfer_eval.py` runs the same held-out task with the same seed and budget, with and
   without `kb/local`, and reports generations to threshold and the best result.

### 6.7 Providers

Crusoe Intelligence Foundry is the default provider (done in Phase 0). OpenRouter (Liquid) and Vultr stay available
through `--provider`.

| Setting | Value |
|---|---|
| Base URL | `https://api.inference.crusoecloud.com/v1`, per [Crusoe's serverless inference docs](https://docs.crusoecloud.com/serverless-inference/index.html). `managed-inference-api-proxy.crusoecloud.com` is the older host |
| Key | `CRUSOE_API_KEY` (console: Admin → Security → Intelligence API keys) |
| Model | `deepseek-ai/Deepseek-V4-Flash`; override with `CRUSOE_MODEL` or `--model`, e.g. NVIDIA Nemotron `nvidia/Nemotron-3.5-Lightning-30B-A3B` or `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` |
| Max tokens | 32768 |
| Catalog | `--list-models crusoe` (requires the key; without it the endpoint returns 401) |

**One call per generation.** The request is a single chat completion with a system and a user message, returning a
JSON object. Crusoe rejects the `developer` role, per [Pipecat's notes](https://docs.pipecat.ai/api-reference/server/services/llm/crusoe);
the agent only sends `system` and `user`.

**Fallback for empty replies.** Crusoe documents neither JSON mode nor a thinking-budget field. Third-party hosts
report that DeepSeek V4, with thinking on, can answer a JSON-mode request with empty content. So
`call_mutation_agent` treats an empty reply like a 400:

- it retries the same prompt without `response_format`;
- for the rest of the run, it stops sending whatever a provider has refused.

The only extra call is that first retry.

**Model IDs (checked against the live catalog, 2026-10-17).** `GET /v1/models` lists IDs that differ from Crusoe's
docs page, and the catalog's spelling is the one to use:

- `deepseek-ai/Deepseek-V4-Flash`, not `DeepSeek-V4-Flash`;
- `nvidia/Nemotron-3.5-Lightning-30B-A3B`, not `nvidia/nemotron-3.5-lightning-30b-a3b`;
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`, not `nvidia/Nemotron-3-Super-120B-A12B`.

**Still to verify:** a one-generation run with DeepSeek and one with Nemotron, to show whether JSON mode is honoured
and how many reasoning tokens a mutation costs.

## 7. Phases

| Phase | Scope | Done when |
|---|---|---|
| 0 Bootstrap | done: clone, submodule, Crusoe provider, B200 SM count. Remaining: dependencies, split `run_evolution.py`, CI | golden simulator lineage unchanged; CI green; a one-generation Crusoe run returns a candidate |
| 1 Correctness + sanitizer | `cpu_check.py`, `sandbox.py`, `sanitizer/`, hardened `gpu_bench.py`, new feedback lessons | all fixtures listed below are rejected; all seeds pass; CPU gates take ≤ 10 s per candidate |
| 2 Trajectory + exploit | `ITERATIONS.md`, config hash, stagnation counter, deterministic sweep | no configuration is timed twice in a run; on a fixture GEMM two launch constants away from a known-better configuration, the first sweep contains that configuration |
| 3 KernelWiki read hook | `knowledge/`, explore operator | each of the 13 symptoms yields at least one pattern and one technique in < 3 s (< 50 ms cached); explore prompts carry the block |
| 4 B200 evaluator | `evaluators/b200/` per §6.3 | the notes' test plan, as corrected, passes on CPU; the tutorial ordering is reproduced; < 50 ms per candidate |
| 5 Operators | crossover and explore as single-call prompt variants; runner-up archive | every operator issues exactly one LLM call; crossover draws its second parent from the archive |
| 6 Meta-reflector | `reflector.py`, overlay, `transfer_eval.py` | a synthetic winning run produces pages that pass overlay `validate.py` and are retrieved by the next run |
| 7 Calibration (GPU) | fit Petri-net parameters: GB10 profile on measured lineages, B200 with a rented sweep | Spearman ρ ≥ 0.6 between predicted and measured latency on held-out kernels; only then does the prediction rank above the static score |

Phase 1 fixtures that must be rejected:

- `solve` returning `torch.matmul`;
- a cache keyed on `data_ptr`;
- an import-time `torch.allclose` patch;
- an unmasked partial write;
- hardcoded strides on a transposed input;
- a kernel that reads an API key from the environment.

Order:

- Phase 0 comes first.
- Phases 1, 2 and 4 don't depend on each other. Start with Phase 1, because it closes live holes and makes every
  later phase testable without a GPU.
- Phase 3 needs Phase 2 (the stagnation trigger) and benefits from Phase 4 (symptoms).
- Then Phase 5, then Phase 6.
- Phase 7 runs whenever GPU data exists.

## 8. New CLI flags

| Flag | Default |
|---|---|
| `--cpu-check on\|off` | on |
| `--sanitize strict\|warn\|off` | strict |
| `--evaluator auto\|b200\|off` | auto |
| `--kernelwiki PATH\|off` | `external/KernelWiki` |
| `--stagnation-threshold N` | 3 |
| `--patience N` | |
| `--operators mutate,crossover,exploit,explore` | |
| `--exploit-sweep K` | |
| `--reflect` | |
| `--provider crusoe\|openrouter\|vultr` (existing flag) | Crusoe when `CRUSOE_API_KEY` is set (done) |

## 9. Risks

- **The Petri net misranks.** It acts only as a gate, a tiebreak and a source of symptoms until Phase 7 shows it
  ranks correctly.
- **KernelWiki pushes the small model toward CUDA syntax.** Mitigated by putting only Triton code in the block, concept
  text otherwise, and the guard metric in §6.5.
- **Sanitizer false positives on legitimate glue.** Mitigated by per-problem denylists, an allowlist for glue, and
  fixtures in both directions.
- **Interpreter gaps.** `cpu-unsupported` passes through rather than failing.
- **KernelWiki schema changes.** Mitigated by pinning the submodule; validation runs only in the reflector.

## 10. Open questions

1. Can the measured GB10 lineages (`evolution_result_*.json` and `.miniavo_memory.jsonl` from the DGX Spark) be
   copied here for calibration and as Phase 2 fixtures? This machine has none. mini-kerngen's memory file is
   `.kerngen_memory.jsonl`, so rename the old file or point `FEEDBACK_FILE` at it. The RawTree table is now
   `kerngen_feedback`; `RAWTREE_FEEDBACK_TABLE=miniavo_feedback` reads the old one.
2. Is there B200 access for Phase 7 (Vultr cloud GPU, Modal)?
3. Is there a written rubric for "Shared Brain" and "Version N+1"? I couldn't find the track definitions online. This
   plan maps them to §6.2 and §6.6.

Decided: the repo is a clone that keeps mini-avo-liquid's history. The LLM is Crusoe Intelligence Foundry, with one
call per generation: `deepseek-ai/Deepseek-V4-Flash` by default, and NVIDIA Nemotron as the alternative.
