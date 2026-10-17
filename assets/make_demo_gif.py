"""Render assets/demo.gif: an animated overview of the mini-kerngen loop, top to bottom.

Illustrative only (no real results). The prompt is built from the parent kernel, the feedback
memory and, once the search stalls, a KernelWiki technique page. Each generation makes one LLM call
for one candidate; five candidates are drawn side by side so each check can be followed down the
page: the Python syntax check, the GPU-less Triton compile, the anti-gaming sanitizer, the model +
analytical heuristic evaluator, and every Nth generation a real GPU run (torch reference, then
timing). The best becomes the next parent, and what the checks found becomes instructions for the
next prompt. What docs/PLAN.md plans but is not built yet is drawn in purple.

    python assets/make_demo_gif.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 900, 810
BG = (250, 250, 247)
INK = (38, 38, 36)
MUTED = (120, 120, 114)
LINE = (205, 205, 198)
ACCENT = (37, 99, 235)
OK = (22, 142, 86)
BAD = (200, 60, 50)
GOLD = (196, 142, 20)
PLANNED = (124, 58, 237)  # in docs/PLAN.md, not built yet
PLANNED_BG = (244, 240, 254)
CARD = (255, 255, 255)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False, mono=False):
    name = "DejaVuSans%s%s.ttf" % ("Mono" if mono else "", "-Bold" if bold else "")
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_H, F_H2, F_B, F_S = font(26, True), font(16, True), font(14, True), font(15), font(12)
F_RULE, F_MONO, F_KEY, F_VAL = font(14), font(12, mono=True), font(17, True), font(15)

AGENT, SYNTAX, COMPILE, SANITIZE, EVALUATE, TIMED = range(6)
# Pipeline rows, top to bottom: (y center, title lines, subtitle lines, planned)
ROWS = [
    (222, ["LLM agent"], ["1 call per generation:", "DeepSeek or Nemotron on Crusoe"], False),
    (304, ["1. Syntax"], ["ast.parse"], False),
    (386, ["2. Compile"], ["Triton, no GPU: sm_80 … sm_121"], False),
    (470, ["3. Anti-Gaming Sanitizer"], ["after AKO4X: varying inputs,", "poison cells, library calls"], True),
    (562, ["4. Model + Analytical", "Heuristic Evaluator"], ["static score today; planned: CPU", "deviation, tcgen05, timed Petri net"], True),
    (660, ["5. GPU run"], ["every Nth generation:", "correctness, then time"], False),
]
BANDS = [180, 263, 345, 428, 516, 611, 708]  # row boundaries
LANE_X = [370, 465, 560, 655, 750]
LOOP_X = 840
PARENT_BOX, MEMORY_BOX, WIKI_BOX = [32, 92, 252, 158], [268, 92, 548, 158], [564, 92, 800, 158]

# Where each candidate fails (None: passes every check). Generation 1's rules fix draft 2's
# syntax error, so in generation 2 it trips the compiler instead.
FAILS_AT = {1: [None, SYNTAX, SANITIZE, EVALUATE, None],
            2: [None, COMPILE, SANITIZE, EVALUATE, None]}
BARS = [0.82, None, None, None, 1.0]            # illustrative static scores (no real numbers)
GPU = ["1874 µs", None, None, None, "1602 µs"]  # illustrative GPU timings
BEST = 4  # the rightmost lane, so the path back to the parent crosses no other card


def arrow(d, start, end, color, width=2):
    d.line([start, end], fill=color, width=width)
    (x0, y0), (x1, y1) = start, end
    length = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d.polygon([(x1, y1), (x1 - 11 * ux + 5 * uy, y1 - 11 * uy - 5 * ux), (x1 - 11 * ux - 5 * uy, y1 - 11 * uy + 5 * ux)],
              fill=color)


def dashed_arrow(d, x, y0, y1, color, dash=5, gap=4):
    y = y0
    while y < y1 - 10:
        d.line([(x, y), (x, min(y + dash, y1 - 10))], fill=color, width=2)
        y += dash + gap
    d.polygon([(x, y1), (x - 5, y1 - 10), (x + 5, y1 - 10)], fill=color)


def box(d, rect, title, lines, color, fill=CARD, width=2, sub_color=INK):
    d.rounded_rectangle(rect, radius=9, fill=fill, outline=color, width=width)
    cx = (rect[0] + rect[2]) // 2
    d.text((cx, rect[1] + 20), title, font=F_H, fill=color, anchor="mm")
    for i, line in enumerate(lines):
        d.text((cx, rect[1] + 42 + 15 * i), line, font=F_S, fill=sub_color, anchor="mm")


def row_label(d, y, titles, subs, planned):
    lines = [(t, F_H2 if len(titles) > 1 else F_H, PLANNED if planned else INK, 18) for t in titles]
    lines += [(s, F_S, PLANNED if planned else MUTED, 15) for s in subs]
    top = y - (sum(h for *_, h in lines) + 3) / 2
    for i, (text, f, color, h) in enumerate(lines):
        d.text((32, top), text, font=f, fill=color)
        top += h + (3 if i == len(titles) - 1 else 0)


def base(gen_label, caption, rows=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((32, 22), "mini-kerngen", font=F_TITLE, fill=INK)
    title_right = d.textbbox((32, 22), "mini-kerngen", font=F_TITLE)[2]
    d.text((title_right + 16, 31), "evolutionary search for Triton kernels", font=F_B, fill=MUTED)
    d.text((W - 32, 31), gen_label, font=F_H, fill=ACCENT, anchor="ra")
    d.text((W - 32, 54), "purple: planned (docs/PLAN.md)", font=F_S, fill=PLANNED, anchor="ra")
    if rows:
        d.line([(32, BANDS[0]), (W - 32, BANDS[0])], fill=LINE, width=1)
        for i, (y, titles, subs, planned) in enumerate(ROWS):
            if planned:
                d.rectangle([32, BANDS[i] + 1, W - 32, BANDS[i + 1] - 1], fill=PLANNED_BG)
            row_label(d, y, titles, subs, planned)
            d.line([(32, BANDS[i + 1]), (W - 32, BANDS[i + 1])], fill=LINE, width=1)
    d.rounded_rectangle([32, 736, W - 32, 788], radius=10, fill=CARD, outline=LINE)
    d.text((W // 2, 762), caption, font=F_B, fill=INK, anchor="mm")
    return img, d


def prompt_inputs(d, gen, parent_label="parent kernel", parent_color=INK, wiki_active=False):
    box(d, PARENT_BOX, parent_label, ["seed: naive or web (Nimble)"], parent_color, sub_color=MUTED)
    box(d, MEMORY_BOX, "Feedback memory",
        ["no rules yet"] if gen == 1 else ["rules and timings from", "earlier generations"], ACCENT)
    if wiki_active:
        box(d, WIKI_BOX, "KernelWiki", ["symptom: pipeline-stalls", "→ technique-pipeline-stages"], PLANNED,
            fill=PLANNED_BG, width=3)
    else:
        box(d, WIKI_BOX, "KernelWiki", ["technique page for the prompt,", "once the search stalls"], PLANNED)


def candidate(d, x, y, idx, state):
    color = {"new": ACCENT, "ok": OK, "bad": BAD, "best": GOLD}[state]
    d.rounded_rectangle([x - 40, y - 19, x + 40, y + 19], radius=7, fill=CARD, outline=color, width=2)
    mark = {"new": "", "ok": " ✓", "bad": " ✗", "best": " ★"}[state]
    d.text((x, y), f"draft {idx + 1}{mark}", font=F_S, fill=color, anchor="mm")


def score_bar(d, x, y, frac, best):
    d.rounded_rectangle([x - 30, y + 25, x + 30, y + 33], radius=4, fill=(230, 230, 224))
    d.rounded_rectangle([x - 30, y + 25, x - 30 + int(60 * frac), y + 33], radius=4, fill=GOLD if best else OK)


def gpu_label(d, x, y, text, color):
    d.text((x, y + 33), text, font=F_S, fill=color, anchor="mm")


def place(d, fails, stage):
    """Each candidate on the row it has reached by `stage`, red where it failed."""
    for i, x in enumerate(LANE_X):
        failed = fails[i] is not None and fails[i] <= stage
        row = min(stage, fails[i] or stage)
        candidate(d, x, ROWS[row][0], i, "bad" if failed else "ok")


def feedback_frame(gen_label, benched):
    """Findings -> feedback memory (+ KernelWiki on a stall) -> instructions in the next prompt."""
    img, d = base(gen_label, "What the checks%s found becomes instructions for the next prompt"
                  % (" and the GPU run" if benched else ""), rows=False)
    if benched:
        sources = [("draft 2 ✗", "compile error", BAD), ("draft 3 ✗", "gaming: cached result", PLANNED),
                   ("draft 4 ✗", "SMEM over 227 KB", PLANNED), ("draft 5 ✓", "timed, kept", OK)]
    else:
        sources = [("draft 2 ✗", "syntax error", BAD), ("draft 3 ✗", "gaming: torch.matmul", PLANNED),
                   ("draft 4 ✗", "deviation on CPU", PLANNED), ("draft 1 ✓", "near miss", OK)]
    for i, (head, why, color) in enumerate(sources):
        x = 32 + 210 * i
        d.rounded_rectangle([x, 96, x + 206, 150], radius=8, fill=CARD, outline=color, width=2)
        d.text((x + 103, 114), head, font=F_H2, fill=color, anchor="mm")
        d.text((x + 103, 135), why, font=F_S, fill=color, anchor="mm")
        arrow(d, (x + 103, 152), (200 + 70 * i, 206), MUTED)
    box(d, [150, 210, 470, 300], "Feedback memory",
        ["local: this run · file: .kerngen_memory.jsonl", "RawTree: across runs · repeats: \"seen N×\""], ACCENT)
    box(d, [500, 210, 820, 300], "KernelWiki",
        ["symptom → pattern → technique page,", "added once the search stalls"], PLANNED)
    arrow(d, (310, 302), (310, 338), ACCENT, 3)
    arrow(d, (660, 302), (660, 338), PLANNED, 3)
    d.rounded_rectangle([32, 340, W - 32, 700], radius=10, fill=CARD, outline=LINE)
    d.text((52, 356), "Next prompt: rules and timings" if benched else "Next prompt: rules", font=F_H, fill=INK)
    rules = ([("Fix the kernel launch (compile error)", INK), ("Don't cache results between calls", PLANNED),
              ("Shared memory over 227 KB per SM", PLANNED), ("The kernel to beat ran in 1602 µs", INK),
              ("These strategies were already timed", INK)] if benched
             else [("Write Python, not C (no 0.0f)", INK), ("Write the kernel; don't call torch.matmul", PLANNED),
                   ("Cover the whole output (CPU deviation)", PLANNED), ("Loads are narrow → larger BLOCK", INK),
                   ("Scored no higher than the parent", INK)])
    rules.append(("On a stall: a KernelWiki technique page", PLANNED))
    for i, (rule, color) in enumerate(rules):
        d.text((56, 398 + 40 * i), "• " + rule, font=F_RULE, fill=color)
    return img


def stall_frame():
    """Stagnation: the elite's symptom selects a KernelWiki page that joins the next prompt."""
    img, d = base("generation 6", "No better kernel for 3 generations: KernelWiki adds a technique page to the prompt",
                  rows=False)
    prompt_inputs(d, 6, wiki_active=True)
    arrow(d, (142, 160), (142, 456), MUTED)
    arrow(d, (408, 160), (408, 236), ACCENT)
    arrow(d, (682, 160), (682, 236), PLANNED, 3)
    d.rounded_rectangle([268, 240, W - 32, 408], radius=10, fill=CARD, outline=PLANNED, width=2)
    d.text((288, 256), "Prompt block (planned)", font=F_H2, fill=PLANNED)
    block = ["[KERNELWIKI technique-pipeline-stages | source-reported | sm100]",
             "Retrieved because: TMA-bound, bubble ratio 0.41 at 2 stages",
             "Idea: keep more tile buffers in flight so loads overlap MMAs",
             "In Triton: num_stages=4, tl.range(..., warp_specialize=True)"]
    for i, line in enumerate(block):
        d.text((288, 290 + 28 * i), line, font=F_MONO, fill=INK)
    arrow(d, (568, 410), (568, 456), PLANNED, 3)
    d.rounded_rectangle([92, 458, W - 92, 516], radius=10, fill=CARD, outline=ACCENT, width=2)
    d.text((W // 2, 487), "LLM agent: one call, told to explore this technique", font=F_H2, fill=ACCENT, anchor="mm")
    d.rounded_rectangle([92, 556, W - 92, 664], radius=10, fill=PLANNED_BG, outline=PLANNED, width=2)
    d.text((112, 568), "Stagnation counter (planned)", font=F_H2, fill=PLANNED)
    d.text((112, 600), "0–2 generations without a new elite → exploit: sweep launch constants, no LLM",
           font=F_RULE, fill=INK)
    d.text((112, 630), "3 or more → explore: one LLM call carrying a KernelWiki technique page", font=F_RULE, fill=INK)
    return img


def frames():
    out = []
    for gen in (1, 2):
        g = f"generation {gen}"
        benched = gen == 2  # --gpu-bench-every 2: only even generations reach the GPU
        fails = FAILS_AT[gen]
        img, d = base(g, "Start from the current parent kernel and what earlier generations learned")
        prompt_inputs(d, gen)
        out.append((img, 1100))

        img, d = base(g, "One LLM call per generation; five candidates are drawn side by side to follow the checks"
                      if gen == 1 else "The prompt now carries the rules and the measured timings from before")
        prompt_inputs(d, gen)
        arrow(d, (142, 160), (142, 188), MUTED)
        arrow(d, (408, 160), (408, 188), ACCENT, 3 if gen > 1 else 2)
        dashed_arrow(d, 682, 160, 188, PLANNED)
        for i, x in enumerate(LANE_X):
            candidate(d, x, ROWS[AGENT][0], i, "new")
        out.append((img, 1500))

        captions = {SYNTAX: "1. Python syntax check drops unparseable code",
                    COMPILE: "2. Ahead-of-time Triton compile drops made-up tl.* APIs and type errors",
                    SANITIZE: "3. Anti-gaming sanitizer (planned): library calls, cached results, monkeypatching",
                    EVALUATE: "4. Evaluator: static score; planned: CPU deviation, tcgen05, Petri-net timing"}
        for stage in (SYNTAX, COMPILE, SANITIZE, EVALUATE):
            img, d = base(g, captions[stage])
            prompt_inputs(d, gen)
            place(d, fails, stage)
            if stage == EVALUATE:
                for i, x in enumerate(LANE_X):
                    if fails[i] is None:
                        score_bar(d, x, ROWS[EVALUATE][0], BARS[i], False)
            out.append((img, 1300))

        final = TIMED if benched else EVALUATE
        if benched:
            img, d = base(g, "5. Every Nth generation the survivors run on a real GPU: correctness first, then time")
            prompt_inputs(d, gen)
            place(d, fails, TIMED)
            for i, x in enumerate(LANE_X):
                if fails[i] is None:
                    gpu_label(d, x, ROWS[TIMED][0], GPU[i], OK)
            out.append((img, 2200))

        img, d = base(g, "Fastest measured time wins and becomes the next parent" if benched
                      else "Highest static score (strictly better than the parent) becomes the new parent")
        prompt_inputs(d, gen, "old parent", MUTED)
        for i, x in enumerate(LANE_X):
            if fails[i] is None:
                y = ROWS[final][0]
                candidate(d, x, y, i, "best" if i == BEST else "ok")
                if benched:
                    gpu_label(d, x, y, GPU[i], GOLD if i == BEST else OK)
                else:
                    score_bar(d, x, y, BARS[i], i == BEST)
            else:
                candidate(d, x, ROWS[fails[i]][0], i, "bad")
        # Back to the parent: right along the elite's row, up the margin, across above the inputs
        x, y = LANE_X[BEST], ROWS[final][0]
        d.line([(x + 40, y), (LOOP_X, y), (LOOP_X, 80), (142, 80), (142, 84)], fill=GOLD, width=3)
        d.polygon([(142, 92), (135, 81), (149, 81)], fill=GOLD)
        d.text((300, 77), "next generation", font=F_S, fill=GOLD, anchor="mb")
        out.append((img, 1800))
        out.append((feedback_frame(g, benched), 2600))

    out.append((stall_frame(), 3400))

    img, d = base("", "No GPU needed to search; one on the machine turns the proxy into a measurement", rows=False)
    lines = [("Problems", "matmul_v2 · vectorsum_v2 · cholesky · trimul_alphafold3", False),
             ("Model", "Deepseek-V4-Flash or NVIDIA Nemotron on Crusoe; one call per generation", False),
             ("Checks", "Python syntax → Triton compile (sm_80 … sm_121) → launch limits", False),
             ("Sanitizer", "after AKO4X: varying inputs, poison cells, library-call ban", True),
             ("Evaluator", "static score today; CPU deviation, tcgen05 numerics, Petri net", True),
             ("KernelWiki", "symptom-matched technique pages once the search stalls", True),
             ("GPU run", "--gpu-bench-every N: torch reference, extra shapes, then do_bench", False),
             ("Feedback", "findings and timings → next prompt; memory: run, file or RawTree", False),
             ("Output", "lineage JSON + submission script for the best kernel", False)]
    for i, (k, v, planned) in enumerate(lines):
        d.text((56, 120 + 62 * i), k, font=F_KEY, fill=PLANNED if planned else ACCENT)
        d.text((214, 121 + 62 * i), v, font=F_VAL, fill=INK)
    out.append((img, 3000))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fr = frames()
    imgs = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f, _ in fr]
    path = os.path.join(here, "demo.gif")
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[ms for _, ms in fr], loop=0, optimize=True)
    print(f"wrote {path} ({len(imgs)} frames, {os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
