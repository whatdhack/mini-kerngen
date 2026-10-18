# mini-kerngen

**Evolutionary search for Triton kernels with an LLM: compile-checked without a GPU, timed on one when there is one.**

![mini-kerngen loop, drawn top to bottom. The prompt is built from the parent kernel, the feedback memory and, once the search stalls, a KernelWiki technique page. Each generation one LLM call (DeepSeek or Nemotron on Crusoe) produces a candidate. Candidates then pass down through rows: Python syntax, ahead-of-time Triton compile, the anti-gaming sanitizer, the model + analytical heuristic evaluator, and every Nth generation a real GPU run. The best loops back up to become the next parent, and what the checks found becomes instructions in the next prompt. A final frame shows the stagnation counter selecting a KernelWiki page after three generations without improvement. The parts mini-kerngen adds (sanitizer, evaluator, KernelWiki, stagnation counter) are drawn in purple](assets/demo.gif)

The evolution loop described below runs with [Crusoe Intelligence Foundry](https://docs.crusoecloud.com/serverless-inference/index.html)
as the default model provider. Its default model is `deepseek-ai/Deepseek-V4-Flash`; an
[NVIDIA Nemotron](https://developer.nvidia.com/nemotron) model such as `nvidia/Nemotron-3.5-Lightning-30B-A3B` can be
selected with `--model`. Every candidate passes three components, each drawn in purple in the animation and each
built on published work (see [References](#references)):

- **Anti-Gaming Sanitizer** (`sanitizer/`, `harness.py`). Static and dynamic checks against reward hacking, after
  [AKO4X](https://github.com/TongmingLAIC/AKO4X):
  - its varying-input cheat check, which grew out of
    [flashinfer-bench #413](https://github.com/flashinfer-ai/flashinfer-bench/pull/413);
  - poison-cell tests;
  - its rule that the operator must not be delegated to a library.

  [sanitizer/TRAPS.md](sanitizer/TRAPS.md) lists what is caught, and by which check.
- **Model + Analytical Heuristic Evaluator** (`cpu_check.py`, `evaluators/b200/`). Three checks that need no GPU:
  - correctness on the CPU through the Triton interpreter;
  - B200-exact numerics for narrow formats from [tcgen05-model](https://github.com/sf-tensor/tcgen05);
  - a timed [Petri net](https://doi.org/10.1109/5.24143) of the TMA → `tcgen05.mma` → epilogue pipeline, following
    [CUTLASS's pipeline semantics](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md), which
    predicts stalls and latency.
- **[KernelWiki](https://github.com/mit-han-lab/KernelWiki)** (`knowledge/`). After `--stagnation-threshold`
  generations without a new elite, the elite's measured symptoms select a technique page, and that page goes into
  the generation's one LLM call.

## Status

| Component | State |
|---|---|
| Evolution loop: one LLM call per generation, GPU-less Triton compile, static score, optional GPU timing, feedback memory | Working |
| Crusoe Intelligence Foundry as the default provider: `deepseek-ai/Deepseek-V4-Flash` by default, NVIDIA Nemotron with `--model` | Working; model IDs checked against Crusoe's live catalog, tests against a fake client, no generations run yet |
| Anti-Gaming Sanitizer: static policy, plus dynamic audits in a sandbox with no network or keys | Working; tested on CPU with cheating fixtures. The GPU harness runs the same audits, not yet tested on a GPU |
| Model + Analytical Heuristic Evaluator, part 1: every candidate run on the CPU through the Triton interpreter (DEVIATION) | Working; 2–7 s per candidate |
| Model + Analytical Heuristic Evaluator, part 2: B200 hard gates, tcgen05-model format check, timed Petri net | Working, for `--hardware b200` GEMMs. The Petri net is uncalibrated, so it only breaks ties between equal static scores |
| KernelWiki retrieval and a stagnation counter that triggers it | Working |
| Deterministic sweep of launch constants (exploit), duplicate detection, per-run `ITERATIONS.md` | Planned (Phase 2) |
| A reflector that writes winning patterns back as KernelWiki pages | Planned (Phase 6) |
| Calibrating the Petri net against measured kernels | Planned (Phase 7; needs GPU data) |

[docs/PLAN.md](docs/PLAN.md) has the design and the order of the phases. It also records where the original design
notes disagree with the projects they build on: B200 constants, KernelWiki's page schema, and the tcgen05 model's
API.

## What it does

mini-kerngen runs an evolutionary loop over GPU kernels for
[GPU MODE](https://www.gpumode.com)-style problems (`matmul_v2`, `vectorsum_v2`, `cholesky`,
`trimul_alphafold3`) on target GPUs from A100 to H100, B200 and the GB10:

1. **Seed.** Start from a naive kernel in `seeds/naive/`, or search the web for a better one
   with [Nimble](https://docs.nimbleway.com) (`--web-seed`).
2. **Mutate.** An LLM rewrites the current parent kernel in one call per generation, following an optimization
   strategy and an optional steering prompt. The model is `deepseek-ai/Deepseek-V4-Flash` on Crusoe Intelligence
   Foundry by default. Crusoe also serves NVIDIA Nemotron: `--model nvidia/Nemotron-3.5-Lightning-30B-A3B`, or
   `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` for a larger model. Other providers are `liquid/lfm-2.5-2.6b:free` on
   OpenRouter (`--provider openrouter`) and `glm-5.3` on Vultr Serverless Inference (`--provider vultr`).
   After `--stagnation-threshold` generations without a new elite, a KernelWiki technique page chosen by the
   elite's measured symptoms joins that one prompt.
3. **Check.** Each candidate must parse as Python, pass the anti-gaming policy, compile as Triton
   for the target GPU, pass the B200 evaluator's hard gates (on `--hardware b200`), and produce the
   right result on the CPU through the Triton interpreter, while surviving the dynamic audits.
4. **Score.** Candidates that pass get a static performance score and, for B200 GEMMs, a Petri-net
   latency prediction. Every Nth generation, the candidate is also run on a real GPU
   (`--gpu-bench-every N`): checked against a torch reference with the same audits, then timed.
5. **Select.** A candidate becomes the new parent only if it scores strictly higher than the
   current one, or ties and is predicted to be faster. Once there is a real timing, it must have
   been timed too and have run faster.
6. **Feed back.** What the checks, the static analysis and any real GPU run found becomes
   instructions in the next generation's prompt. These are kept for the current run, or across
   runs in a local JSONL file (`--feedback-memory file`) or RawTree (`--feedback-memory rawtree`).

The loop repeats for `--generations` rounds.

Everything is written to a lineage JSON (every candidate, including failed ones and why they
failed) and a submission script containing the best kernel.

### How kernels are judged

The search needs no GPU. Every candidate is checked against an anti-gaming policy, compiled for
the target architecture and judged by what the compiler produces, then run on the CPU through
the Triton interpreter at small shapes. That catches most of what a model gets wrong, including
wrong answers, and it works on a laptop. It cannot tell you whether a kernel is *fast*: the B200
Petri net predicts that but is uncalibrated. When there is a GPU on the machine,
`--gpu-bench-every N` adds a real timing, which then outranks every proxy.

| Stage | What it catches | How |
|---|---|---|
| **Syntax** | Code that isn't valid Python (e.g. C-style `0.0f`) | `ast.parse` |
| **Sanitizer** | Library delegation (`torch.matmul`, `x.sum()` on the inputs), kernels never launched, monkeypatching, imports and calls a kernel has no use for | `sanitizer/static_policy.py`, before compile (`--sanitize`) |
| **Triton compile** | Made-up `tl.*` functions, type and shape errors, bad `constexpr` use | `triton.compile` ahead of time for sm_80 / sm_89 / sm_90 / sm_100 / sm_121 (`triton_check.py`) |
| **Launch limits** | Kernels whose shared memory or registers can't fit on an SM | Per-SM limits of the target GPU |
| **Static score** (0–100) | How well the compiled kernel uses the hardware | `static_perf.py`, from the PTX, cubin and compile metadata |
| **B200 evaluator** (b200 GEMMs) | More than 227 KB of shared memory or 512 TMEM columns, K tiles the tensor core cannot step through, narrowed formats that miss the tolerance; predicts latency and the bottleneck | `evaluators/b200/`: hard gates, tcgen05-model format check, timed Petri net (`--evaluator`) |
| **CPU check** | Wrong results, unwritten output, hardcoded strides, cached results, tampering with torch | `cpu_check.py`: the Triton interpreter at small shapes, with the dynamic audits in `harness.py` (`--cpu-check`) |
| **GPU run** (optional) | Kernels that pass on the CPU but are wrong, hang or crash on the GPU, and their real speed | `gpu_bench.py`: the same harness at full size, then `do_bench` (`--gpu-bench-every N`) |

The static score combines:

- **Registers and spills** (`cuobjdump` resource usage). Spills score zero. Register *count*
  is scored in a band: credit ramps in from 32, is full from 96 to 224, and falls back to zero
  at the 255 cap, where the compiler is one allocation away from spilling rather than blocking
  well. Without this the occupancy term alone penalizes exactly the register-blocked kernels
  that win — the fastest GEMM measured on the GB10 uses 226 registers at 12% occupancy.
- **Theoretical occupancy**, from registers, shared memory and warps per block
- **Global load width**: 16-byte vector loads and async copies vs. scalar loads
- **Tensor cores**: `tcgen05` (Blackwell), `wgmma` (Hopper) or `mma.sync`
- **No global atomics** on GEMM-like problems: atomics there mean split-K accumulating into C,
  and every split-K variant measured on the GB10 landed in the slow tail
- **TMA** (`cp.async.bulk.tensor`) and **warp specialization** (`tl.range(..., warp_specialize=True)`)
  on Hopper and datacenter Blackwell (sm_90, sm_100) — **not** on sm_121, where measurement showed
  they buy nothing: the fastest 4096³ GEMM found on the GB10 uses neither and beats the best
  TMA + warp-specialized kernel by 7.6%, while the Hopper-style weights scored it 77.5 against
  that kernel's 100
- **Tile reuse** for GEMM-like problems

GEMM-like and memory-bound problems use different weights. TMEM use is reported but not
scored separately, since Triton allocates it automatically with `tcgen05`.

Even with those corrections the score is a weak ranker: across 14 measured GB10 GEMM kernels
spanning 1480–5135 µs, 8 still tie at 100. Treat it as a gate on what deserves a GPU run, not
as an ordering.

Because argument types aren't known without a real launch, the compile guesses them from
parameter names (`*_ptr` → pointer, `M`/`N`/`K` → `i32`) and adds the specializations a real
launch would (16-byte alignment, unit innermost strides).

Tuning constants are not guessed. They are read from the host wrapper's launch —
`kernel[grid](..., BLOCK_SIZE=4096, num_warps=8)`, following names bound to literals in the
wrapper's body or its parameter defaults — and from a `triton.Config` in an `@triton.autotune`
decorator when there is one. The launch site wins, then autotune, then the parameter default,
then a name-based guess. Without this the wrapper is never run, every variant compiles at
`BLOCK_SIZE=128` / 4 warps, and retuned kernels all score the same.

The score is a **proxy**. It reliably flags bad kernels: spills, launch failures, missing tensor
cores, scalar loads, low occupancy. It cannot rank two well-tuned kernels against each other, and
it cannot tell whether a kernel computes the right answer. A 100/100 GEMM candidate with TMA and
warp specialization turned out to be off by 2.4x the peak value when it was actually run; the CPU
check now catches that kind of error before any GPU time is spent. Between candidates with the same
static score, the B200 evaluator's predicted latency decides. To rank on real speed, use
`--gpu-bench-every N`. The model's own latency estimate is shown as "Claimed" and never used for
selection.

### Running candidates on a real GPU

`--gpu-bench-every N` runs generation N, 2N, … on the local GPU, and the seed once at the start:

- **Inputs and reference** come from a per-problem harness in `harness.py`, shared with the CPU
  check. `vectorsum_v2` and `matmul_v2` and `cholesky` have one; `trimul_alphafold3` does not,
  because its seed is a stub with no defined semantics, and the flag is ignored for it.
- **Extra cases**: a harness may list further inputs that are checked but not timed, because a
  kernel tuned against one shape can be wrong on every other one. Both winners so far failed
  here — the GEMM one hardcoded row-major strides and returns garbage for a transposed operand,
  the Cholesky one misses the reconstruction gate at n=128 while passing at n=256.
- **Entry point**: the harness calls the wrapper the mutation prompt asks for — `solve(x)` for
  `vectorsum_v2`, `solve(a, b)` for `matmul_v2`, `custom_kernel(A)` for `cholesky`, matching the
  upstream GPU MODE task — and falls back to the other public functions the sanitizer found to
  launch a `@triton.jit` kernel, trying `fn(*inputs)` and `fn(*inputs, out)`. The first one that
  reproduces the reference result is used.
- **Audits**, the same as on the CPU: references are computed before the candidate is imported, the
  functions the comparison relies on are fingerprinted, fresh outputs start as NaN, and after the
  first correct call the inputs change in place and the output has to follow. Timing is then
  compared with a synchronized wall-clock time, and one more changed input is checked after it.
- **Timing** is the median of `triton.testing.do_bench`, reported as "Measured" alongside the
  speedup over the seed's own measured latency (not the problem table's `baseline_latency_us`,
  which describes a different GPU).
- **Isolation**: generated code is executed, so each run happens in its own interpreter with a
  wall-clock timeout (`--gpu-bench-timeout`, default 300s). A kernel that hangs, crashes the
  driver or loops forever takes its subprocess down and nothing else.
- **Selection**: a candidate the GPU rejected is never promoted. Once the elite has a timing,
  only a timed and faster candidate replaces it, so an off-cycle generation cannot swap a
  measured elite for an unmeasured one.
- **Feedback**: a wrong result or a slower timing becomes an instruction in the next prompt,
  the same way compile errors do.

**Safety:** kernels are model output. The compile path never executes their top-level code —
only trusted imports, function definitions and literal constants are loaded before compiling.
The CPU check and the GPU run do execute them, which is the point, so every run is a separate
interpreter with a wall-clock timeout and no API keys in its environment (`sandbox.py`). The CPU
check also runs without network access.

**Known gaps:** [sanitizer/TRAPS.md](sanitizer/TRAPS.md) lists what is caught, and by which check.
Two things are not covered yet:

- A cache keyed on input *contents* still passes, and is timed warm.
- The audits in the GPU harness are the ones tested on the CPU, but they have not yet run on a GPU.

#### Remembering measurements across runs

A static score can be recomputed from the code at any time; a timing cannot — it only exists
because something was run. With `--feedback-memory file` (or `rawtree`), every measured latency
and every GPU rejection is persisted alongside the lessons and read back by the next run on the
same problem and hardware, so later prompts start with:

```
Real GPU measurements from this and earlier runs (median wall-clock, lower is better; these are
timings, not estimates):
- 1615.9 μs: v0_naive_baseline [baseline]
- 1657.9 μs: v1_two_stage_blockreduce [warp_shuffle]
- 1707.9 μs: v3_multitile_partial_buffer [tile_tuning]
- wrong or unrunnable on the GPU: v2_gridstride_twopass [vectorization] — output contains NaN or Inf
The kernel to beat ran in 1615.9 μs. The strategies listed above have already been measured at
that speed or slower, so a variation of them will not win: change the memory access pattern or
the amount of work per program.
```

The fastest run per kernel name is kept, this run's result supersedes a remembered one of the
same name, and a store never reads back its own run. The file defaults to
`.kerngen_memory.jsonl` next to the script (`FEEDBACK_FILE` overrides it) and is gitignored.

### Baseline

Measured on an NVIDIA GB10 (DGX Spark, sm_121) with `glm-5.3` using MiniAVO. "Best found" is the fastest kernel the
search produced that passed the GPU harness at the time.

| Problem | Seed | Best found | Library reference |
|---|---|---|---|
| `matmul_v2` (4096³ fp16) | 3337 µs / 40 TFLOPS | 1603 µs / 86 TFLOPS (gen 12 of 100) | 1530 µs / 90 TFLOPS (cuBLAS) |
| `cholesky` (batch 64, n 256) | 2229 µs | 825 µs (gen 1) | 671 µs (cuSOLVER) |
| `vectorsum_v2` (100M fp32) | 1616 µs / 247 GB/s | none faster in 10 generations | 1658 µs (`torch.sum`) |

Read those with the caveats:

- **`vectorsum_v2` has no headroom.** The naive seed already reads at 247 GB/s against a 244 GB/s
  measured stream ceiling. Every rewrite adds a launch or a pass and loses. A search that reports
  progress here is reporting noise.
- **Both winners initially passed only because the harness was too narrow.** The GEMM kernel
  hardcodes row-major strides and returns garbage for a transposed operand; the Cholesky kernel
  misses the reconstruction gate at n=128. The `extra_cases()` inputs now catch both, and both are
  rejected by the current harness.
- **The GEMM search converged at generation 12 and then stalled for 88 generations.** 20 of its 30
  timed candidates were the same five launch configurations under different names, and its final
  kernel is 7.6% slower than the same code at `128×256×64 / 8 warps / 3 stages` — two constants
  away, never tried. Phase 2 of the plan, stagnation detection with a deterministic launch-constant
  sweep, targets exactly this.

### Feedback to the next generation

What the checks find is turned into instructions for the next prompt (`feedback.py`):

- **Rejected candidates** become rules. For example, a made-up `tl.*` name gets its Triton
  replacement (`tl.ceil_div` → `tl.cdiv`; no manual shared-memory allocation), C syntax gets
  "write Python, not C", a function defined inside a kernel gets "move helpers to module
  level", and a reply cut off at the token limit gets "return only the kernel".
- **Near misses** are reported: a candidate that compiled but didn't beat the parent is told
  which score components it lost on, or — when both were timed — the two latencies.
- **GPU rejections** become rules of their own: a kernel that passed every static check and then
  returned NaNs, missed the tolerance or timed out says so in the next prompt.
- **The parent's weak spots** from static analysis are listed as opportunities, such as narrow
  loads, no TMA, no warp specialization, spills or small tiles.

Repeated mistakes are merged with a "seen N×" count, and only the most recent few are kept.

**Feedback memory.** With `--feedback-memory local` (the default), lessons last for one run.
With `--feedback-memory file`, they are appended to `.kerngen_memory.jsonl` next to the script
and read back by the next run on the same problem and hardware — measured timings included.
With `--feedback-memory rawtree`, every lesson and candidate outcome is also written as an event
to a [RawTree](https://rawtree.com) table (Tinybird's analytics database for unstructured data; the
table is created on first insert). The next run on the same problem and hardware loads those
lessons back with SQL, so it starts out knowing the mistakes earlier runs made. The same table
can be queried for analysis across runs, for example:

```sql
SELECT status, count() FROM kerngen_feedback
WHERE kind = 'candidate' AND problem = 'matmul_v2' GROUP BY status
```

If RawTree is unreachable or the key is wrong, the run prints one warning and continues with
in-memory feedback.

## How to run

### 1. Environment

Python 3.12 with the latest PyTorch and Triton (PyTorch 2.14 pins Triton 3.8):

```bash
git clone --recurse-submodules https://github.com/whatdhack/mini-kerngen   # KernelWiki is a submodule (external/KernelWiki)
cd mini-kerngen
conda create -n mini-kerngen python=3.12 -y
conda activate mini-kerngen
pip install -r requirements.txt
```

In an existing checkout, `git submodule update --init` fetches KernelWiki.

No GPU or CUDA driver is needed. Triton ships the `ptxas` and `cuobjdump` used for the compile
and static analysis.

If `/tmp` is a small in-memory filesystem (common on WSL), point pip at a disk-backed temp
directory for the large CUDA wheels: `TMPDIR=~/.cache/pip-tmp pip install -r requirements.txt`.

### 2. Keys

```bash
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `CRUSOE_API_KEY` | Mutations via Crusoe Intelligence Foundry, the default provider. Create it in the Crusoe console under Admin → Security → Intelligence API keys. |
| `CRUSOE_MODEL` | Optional Crusoe override (default: `deepseek-ai/Deepseek-V4-Flash`, with `LLM_MAX_TOKENS` defaulting to `32768`), e.g. `nvidia/Nemotron-3.5-Lightning-30B-A3B` for NVIDIA Nemotron. Use IDs exactly as `--list-models crusoe` prints them |
| `OPENROUTER_API_KEY` | Mutations via OpenRouter (Liquid models). |
| `VULTR_SERVERLESS_INFERENCE_API_KEY` | Mutations via Vultr Serverless Inference. `VULTR_INFERENCE_API_KEY` and `VULTR_API_KEY` also work. |
| `NIMBLE_API_KEY` | `--web-seed` only |
| `RAWTREE_API_KEY` (+ optional `RAWTREE_DATABASE`, `RAWTREE_FEEDBACK_TABLE`) | `--feedback-memory rawtree` only; the key needs `read_write` permission |
| `LIQUID_MODEL`, `LIQUID_MAX_TOKENS`, `LIQUID_REASONING_EFFORT` | Optional OpenRouter overrides (defaults: `liquid/lfm-2.5-2.6b:free`, `8192`, `high`) |
| `VULTR_MODEL`, `VULTR_REASONING_EFFORT`, `LLM_MAX_TOKENS` | Optional Vultr overrides (defaults: `glm-5.3`, `low`, `32768`) |
| `LLM_PROVIDER` | Pins the provider (`crusoe`, `openrouter` or `vultr`); same as `--provider` |

With no provider key, every generation uses the deterministic simulator. With several keys set,
the first of Crusoe, OpenRouter and Vultr is used unless `--provider` or `LLM_PROVIDER` says
otherwise. Only the base URL, key and model default differ between them: all are called through
the OpenAI SDK, and the `X-Title` header and `reasoning.effort` body field are sent to OpenRouter
only. `response_format={"type":"json_object"}` is attempted first and retried without it on a
400/422 or an empty reply, then retried again without the thinking-budget field, so providers
that reject either still work. Whatever a provider refused once is not sent again in that run.

The thinking budget is provider-shaped: OpenRouter takes `{"reasoning": {"effort": …}}` and
defaults to `high`; Vultr takes a flat `reasoning_effort` and defaults to `low`, because
`glm-5.3` at the default effort spends all 32k output tokens reasoning about the mutation
prompt and never emits the JSON object. Crusoe documents no thinking-budget field, so none is
sent; DeepSeek V4 reasons before it answers and that reasoning counts toward `max_tokens`, hence
the 32k default.

### 3. Run

```bash
# Memory-bound reduction on H100, 10 generations
python run_evolution.py --problem vectorsum_v2 --hardware h100 --generations 10

# GEMM on B200, starting from a kernel found on the web
python run_evolution.py --problem matmul_v2 --hardware b200 --web-seed --generations 10

# Search the web again instead of using the cached seed in seeds/web/
python run_evolution.py --problem matmul_v2 --web-seed --refresh-seed

# Keep check-derived lessons across runs in RawTree
python run_evolution.py --problem vectorsum_v2 --hardware h100 --feedback-memory rawtree

# Keep lessons and real GPU timings across runs in a local file
python run_evolution.py --problem vectorsum_v2 --hardware gb10 --gpu-bench-every 2 --feedback-memory file

# Steer the mutations, or pick another model
python run_evolution.py --problem matmul_v2 --steer "Use TMA descriptors and warp specialization"
python run_evolution.py --model deepseek-ai/DeepSeek-V4-Pro
python run_evolution.py --model nvidia/Nemotron-3.5-Lightning-30B-A3B     # NVIDIA Nemotron on Crusoe

# Mutate with another provider instead of Crusoe
python run_evolution.py --problem matmul_v2 --provider openrouter --model liquid/lfm-2.5-1.2b-instruct:free
python run_evolution.py --problem matmul_v2 --provider vultr
python run_evolution.py --provider vultr --model qwen3.8-27b

# What can each provider serve? (Crusoe's catalog needs CRUSOE_API_KEY; Vultr's needs no key)
python run_evolution.py --list-models crusoe
python run_evolution.py --list-models vultr

# Time every 3rd generation on the local GB10 (DGX Spark, sm_121)
python run_evolution.py --problem vectorsum_v2 --hardware gb10 --generations 9 --gpu-bench-every 3

# Batched Cholesky, timed every other generation, remembering timings across runs
python run_evolution.py --problem cholesky --hardware gb10 --generations 20 \
    --gpu-bench-every 2 --feedback-memory file
```

`--gpu-bench-every` needs a CUDA GPU on the machine, and `--hardware` should name it so the
compile check targets the same architecture it will run on.

| Flag | Values | Default |
|---|---|---|
| `--problem` | `matmul_v2`, `vectorsum_v2`, `cholesky`, `trimul_alphafold3` | `matmul_v2` |
| `--hardware` | `b200`, `gb10`, `h100`, `a100`, `l4`, `rtx4090` | `b200` |
| `--generations` | number of generations | `5` |
| `--web-seed` / `--refresh-seed` | start from a Nimble-found kernel / ignore the cache | off |
| `--steer` | free-text guidance added to the mutation prompt | none |
| `--provider` | `crusoe`, `openrouter`, `vultr` | whichever API key is set (Crusoe first) |
| `--model` | any model ID the provider serves, e.g. `nvidia/Nemotron-3.5-Lightning-30B-A3B` on Crusoe | provider default: `deepseek-ai/Deepseek-V4-Flash`, `liquid/lfm-2.5-2.6b:free`, `glm-5.3` |
| `--list-models` | `crusoe`, `openrouter`, `vultr` | prints the catalog and exits |
| `--gpu-bench-every` | run generation N, 2N, … on a real GPU (`0` disables); env `GPU_BENCH_EVERY` | `0` |
| `--gpu-bench-timeout` | seconds one GPU benchmark subprocess may take | `300` |
| `--feedback-memory` | `local` (this run only), `file` (JSONL next to the script) or `rawtree` | `local` |
| `--sanitize` | `strict` (policy errors reject), `warn` (recorded only), `off` | `strict` |
| `--cpu-check` | `on`, `off`: run every candidate through the Triton interpreter on the CPU | `on` |
| `--evaluator` | `auto` (B200 gates, format check and Petri net for `--hardware b200` GEMMs), `off` | `auto` |
| `--kernelwiki` | path to a KernelWiki checkout, or `off` | `external/KernelWiki` |
| `--stagnation-threshold` | generations without a new elite before a KernelWiki technique joins the prompt | `3` |

### 4. Read the output

Each generation prints one of three outcomes:

- **`★ NEW ELITE`**: the candidate passed every check and beat the parent.
- **`✓ PASSES CHECKS, NOT BETTER`**: the candidate passed every check but didn't beat the parent.
- **`✗ FAILED`**: the candidate failed a check: syntax, sanitizer, Triton compile, B200 evaluator or
  CPU check. The reason is shown, and becomes a rule in the next prompt.

Under each candidate, one line per check shows what it found: the CPU check's deviation, the B200
model's prediction and bottleneck, sanitizer warnings, and the KernelWiki page when the generation
explored one.

Files written to the current directory:

- `evolution_result_<problem>_<hw>.json`: the full lineage. Each entry has its code, status,
  `check_error`, `static` metrics and `static_score`, `cpu_check`, `evaluator`, `sanitizer_warnings`,
  `operator` (`mutate` or `explore`) and `kernelwiki_page`, and the seed's source URL is included.
- `submission_<problem>_<variant>.py`: the best kernel. The CPU check covers small inputs only, so
  without `--gpu-bench-every` it is **not benchmarked and not checked at full size on a GPU**: run it
  on a real GPU before submitting it anywhere. With it, the lineage also carries
  `measured_latency_us`, `measured_speedup` and `bench_status` per candidate.

### 5. Tests

```bash
pytest     # CPU only, about a minute; no API keys or network needed
```

The tcgen05 tests are skipped when `tcgen05-model` is not installed; the KernelWiki tests need the
submodule (`git submodule update --init`).

## Repository layout

| Path | Purpose |
|---|---|
| `run_evolution.py` | CLI and evolution loop with the gate chain and stagnation counter, provider registry (Crusoe, OpenRouter, Vultr), problem and hardware catalog |
| `sanitizer/` | Static anti-gaming policy, and `TRAPS.md`, the catalog of what is caught |
| `harness.py` | Per-problem inputs and references, and the runner with the dynamic audits, shared by the CPU check and the GPU run |
| `sandbox.py` | Runs model-written code in its own interpreter: timeout, no secrets, no network |
| `cpu_check.py` | Every candidate through the Triton interpreter at small shapes: DEVIATION and audits |
| `evaluators/b200/` | B200 constants, tcgen05 functional check, timed Petri net, evaluator API, Triton adapter |
| `knowledge/` | KernelWiki retrieval and the mapping from measurements to its symptoms |
| `triton_check.py` | Safe loading plus ahead-of-time Triton compile for the target GPU |
| `gpu_bench.py` | Optional real-GPU correctness check and timing, through `harness.py` |
| `static_perf.py` | Static metrics (PTX / cubin / metadata) and the 0–100 score |
| `feedback.py` | Turns check, static-analysis and GPU-run findings into instructions for the next prompt |
| `local_store.py` | Feedback memory in a local JSONL file (`--feedback-memory file`) |
| `rawtree_store.py` | Optional RawTree backend that persists feedback memory across runs |
| `seed_search.py` | Nimble web search for seed kernels; slices out each `@triton.jit` kernel with its helpers |
| `seeds/naive/` | Built-in starter kernels, one `.py` per problem |
| `seeds/web/` | Cached web seeds (`--web-seed`) |
| `docs/PLAN.md` | The plan: what mini-kerngen adds, and in which order |
| `external/KernelWiki/` | [KernelWiki](https://github.com/mit-han-lab/KernelWiki) submodule, pinned |
| `tests/` | CPU-only tests (`pytest`) |
| `assets/make_demo_gif.py` | Regenerates `assets/demo.gif` |

## References

**Origin**

- [MiniAVO-Liquid](https://github.com/whatdhack/mini-avo-liquid): the loop this repo starts from. It began as the
  standalone Python part of `mini-avo-gs`.

**Anti-Gaming Sanitizer**

- [AKO4X](https://github.com/TongmingLAIC/AKO4X) and the
  [AKO tech report](https://tongminglaic.github.io/AKO/assets/ako-tech-report.pdf). Sources of:
  - the varying-input cheat check;
  - the poison-cell tests;
  - the rule against delegating the operator to a library;
  - the `ITERATIONS.md` trajectory format.
- [flashinfer-bench #413](https://github.com/flashinfer-ai/flashinfer-bench/pull/413): random inputs on every timed
  iteration, so a cached output can't pass. AKO4X's cheat check builds on it.

**Model + Analytical Heuristic Evaluator**

- [tcgen05-model](https://github.com/sf-tensor/tcgen05) (San Francisco Tensor Company): bit-accurate Blackwell
  `tcgen05` MMA arithmetic, used for the functional check (Tier 1).
- T. Murata, [Petri nets: Properties, analysis and applications](https://doi.org/10.1109/5.24143), *Proceedings of
  the IEEE* 77(4), 1989: timed Petri nets, for the pipeline model (Tier 2).
- [CUTLASS synchronization primitives](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/pipeline.md): the
  producer/consumer pipeline semantics the Petri net models.
- A. Jarmusch and S. Chandrasekaran,
  [Microbenchmarking NVIDIA's Blackwell Architecture](https://arxiv.org/abs/2512.02189): measured B200 behaviour,
  for calibrating the model.
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/): `tcgen05`, TMEM and `mbarrier`
  semantics.

**Knowledge**

- [KernelWiki](https://github.com/mit-han-lab/KernelWiki) (MIT HAN Lab): a Blackwell and Hopper kernel optimization
  knowledge base, pinned as a submodule.

**Models and inference**

- [Crusoe Intelligence Foundry](https://docs.crusoecloud.com/serverless-inference/index.html): the default
  inference provider, serving `deepseek-ai/Deepseek-V4-Flash` (the default model) and NVIDIA Nemotron.
- [NVIDIA Nemotron](https://developer.nvidia.com/nemotron): open models available on Crusoe as alternative mutation
  models, e.g. `nvidia/Nemotron-3.5-Lightning-30B-A3B` and `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`.

**Problems**

- [GPU MODE](https://www.gpumode.com): the problem definitions.
