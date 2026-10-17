"""Render assets/demo.gif: an animated overview of the mini-kerngen loop.

Illustrative only (no real results): each generation makes one LLM call for one candidate;
five candidates are drawn side by side so each check can be followed. Candidates pass the
Python syntax check and the GPU-less Triton compile, then the anti-gaming sanitizer and the
model + analytical heuristic evaluator, and every Nth generation they run on a real GPU —
checked against a torch reference, then timed. The best becomes the next parent. What the
checks and the GPU run found is turned into instructions (feedback memory: this run, a local
file, or RawTree) for the next generation's prompt, and KernelWiki pages are added once the
search stalls. What docs/PLAN.md plans but is not built yet is drawn in purple.

    python assets/make_demo_gif.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (250, 250, 247)
INK = (38, 38, 36)
MUTED = (120, 120, 114)
LINE = (205, 205, 198)
ACCENT = (37, 99, 235)
OK = (22, 142, 86)
BAD = (200, 60, 50)
GOLD = (196, 142, 20)
PLANNED = (124, 58, 237)  # in docs/PLAN.md, not built yet
CARD = (255, 255, 255)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE, F_H, F_H2, F_B, F_S = font(26, True), font(16, True), font(13, True), font(15), font(12)

# Pipeline columns: (x center, title lines, subtitle lines, planned)
STAGES = [
    (95, ["Parent"], ["seed: naive or", "web (Nimble)"], False),
    (222, ["LLM agent"], ["1 call / generation", "on Crusoe: DeepSeek", "or Nemotron"], False),
    (330, ["1. Syntax"], ["ast.parse"], False),
    (432, ["2. Compile"], ["Triton, no GPU", "sm_80 … sm_121"], False),
    (548, ["3. Anti-Gaming", "Sanitizer"], ["after AKO4X:", "varying inputs,", "library calls"], True),
    (708, ["4. Model + Analytical", "Heuristic Evaluator"], ["static score, CPU", "deviation, tcgen05,", "timed Petri net"], True),
    (860, ["5. GPU run"], ["every Nth generation", "correctness, then time"], False),
]
SYNTAX, COMPILE, SANITIZE, EVALUATE, TIMED = 2, 3, 4, 5, 6
LANES = [200, 250, 300, 350, 400]
# Where each candidate fails (None: passes every check). Generation 1's rules fix draft 2's
# syntax error, so in generation 2 it trips the compiler instead.
FAILS_AT = {1: [None, SYNTAX, SANITIZE, EVALUATE, None],
            2: [None, COMPILE, SANITIZE, EVALUATE, None]}
BARS = [1.0, None, None, None, 0.82]            # illustrative static scores (no real numbers)
GPU = ["1602 µs", None, None, None, "1874 µs"]  # illustrative GPU timings
BEST = 0


def dashed_line(d, x, y0, y1, color, dash=6, gap=5):
    y = y0
    while y < y1:
        d.line([(x, y), (x, min(y + dash, y1))], fill=color, width=1)
        y += dash + gap


def base(gen_label, caption, stages=True):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((32, 22), "mini-kerngen", font=F_TITLE, fill=INK)
    title_right = d.textbbox((32, 22), "mini-kerngen", font=F_TITLE)[2]
    d.text((title_right + 16, 31), "evolutionary search for Triton kernels: compile-checked, GPU-timed",
           font=F_B, fill=MUTED)
    d.text((W - 32, 31), gen_label, font=F_H, fill=ACCENT, anchor="ra")
    d.text((W - 32, 54), "purple: planned (docs/PLAN.md)", font=F_S, fill=PLANNED, anchor="ra")
    for x, titles, sub, planned in (STAGES if stages else []):
        y = 82
        for t in titles:
            d.text((x, y), t, font=F_H2 if len(titles) > 1 else F_H, fill=PLANNED if planned else INK, anchor="ma")
            y += 18
        y += 5
        for s in sub:
            d.text((x, y), s, font=F_S, fill=PLANNED if planned else MUTED, anchor="ma")
            y += 15
        if planned:
            dashed_line(d, x, 176, 425, PLANNED)
        else:
            d.line([(x, 176), (x, 425)], fill=LINE, width=1)
    d.rounded_rectangle([32, 452, W - 32, 508], radius=10, fill=CARD, outline=LINE)
    d.text((W // 2, 480), caption, font=F_B, fill=INK, anchor="mm")
    return img, d


def parent_card(d, label="parent kernel", color=INK):
    d.rounded_rectangle([40, 270, 150, 330], radius=8, fill=CARD, outline=color, width=2)
    d.text((95, 300), label, font=F_S, fill=color, anchor="mm")


def candidate(d, x, y, idx, state):
    color = {"new": ACCENT, "ok": OK, "bad": BAD, "best": GOLD}[state]
    d.rounded_rectangle([x - 40, y - 19, x + 40, y + 19], radius=7, fill=CARD, outline=color, width=2)
    mark = {"new": "", "ok": " ✓", "bad": " ✗", "best": " ★"}[state]
    d.text((x, y), f"draft {idx + 1}{mark}", font=F_S, fill=color, anchor="mm")


def score_bar(d, y, frac, best):
    x0 = 755
    d.rounded_rectangle([x0, y - 6, x0 + 50, y + 6], radius=4, fill=(236, 236, 230))
    d.rounded_rectangle([x0, y - 6, x0 + int(50 * frac), y + 6], radius=4, fill=GOLD if best else OK)


def gpu_label(d, y, text, color):
    d.text((906, y), text, font=F_S, fill=color, anchor="lm")


def arrow(d, start, end, color, width=2):
    d.line([start, end], fill=color, width=width)
    (x0, y0), (x1, y1) = start, end
    length = max(1.0, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    d.polygon([(x1, y1), (x1 - 11 * ux + 5 * uy, y1 - 11 * uy - 5 * ux), (x1 - 11 * ux - 5 * uy, y1 - 11 * uy + 5 * ux)],
              fill=color)


def feedback_frame(gen_label, benched):
    """Findings -> feedback memory (+ KernelWiki on a stall) -> instructions in the next prompt."""
    img, d = base(gen_label, "What the checks%s found becomes instructions for the next prompt"
                  % (" and the GPU run" if benched else ""), stages=False)
    if benched:
        sources = [("draft 2 ✗  compile error", BAD), ("draft 3 ✗  gaming: cached result", PLANNED),
                   ("draft 4 ✗  SMEM over 227 KB", PLANNED), ("draft 1 ✓  timed, kept", OK)]
    else:
        sources = [("draft 2 ✗  syntax error", BAD), ("draft 3 ✗  gaming: torch.matmul", PLANNED),
                   ("draft 4 ✗  deviation on CPU", PLANNED), ("draft 5 ✓  near miss", OK)]
    for i, (label, color) in enumerate(sources):
        y = 130 + 58 * i
        d.rounded_rectangle([40, y - 20, 268, y + 20], radius=7, fill=CARD, outline=color, width=2)
        d.text((154, y), label, font=F_S, fill=color, anchor="mm")
        arrow(d, (270, y), (336, 190 + 22 * i), MUTED)
    d.rounded_rectangle([340, 168, 566, 278], radius=10, fill=CARD, outline=ACCENT, width=2)
    d.text((453, 190), "Feedback memory", font=F_H, fill=ACCENT, anchor="mm")
    d.text((453, 217), "local: this run", font=F_S, fill=INK, anchor="mm")
    d.text((453, 238), "file: .kerngen_memory.jsonl", font=F_S, fill=INK, anchor="mm")
    d.text((453, 259), "RawTree: across runs", font=F_S, fill=INK, anchor="mm")
    d.text((453, 298), "repeats merged: \"seen N×\"", font=F_S, fill=MUTED, anchor="mm")
    arrow(d, (568, 223), (604, 223), ACCENT, 3)
    d.rounded_rectangle([340, 326, 566, 420], radius=10, fill=CARD, outline=PLANNED, width=2)
    d.text((453, 348), "KernelWiki", font=F_H, fill=PLANNED, anchor="mm")
    d.text((453, 374), "symptom → pattern → technique", font=F_S, fill=INK, anchor="mm")
    d.text((453, 395), "page, once the search stalls", font=F_S, fill=INK, anchor="mm")
    arrow(d, (568, 373), (604, 373), PLANNED, 3)
    d.rounded_rectangle([606, 112, 930, 420], radius=10, fill=CARD, outline=LINE)
    d.text((622, 128), "Next prompt: rules and timings" if benched else "Next prompt: rules",
           font=F_H, fill=INK)
    rules = ([("Fix the kernel launch (compile error)", INK), ("Don't cache results between calls", PLANNED),
              ("Shared memory over 227 KB per SM", PLANNED), ("The kernel to beat ran in 1602 µs", INK),
              ("These strategies were already timed", INK)] if benched
             else [("Write Python, not C (no 0.0f)", INK), ("Write the kernel; don't call torch.matmul", PLANNED),
                   ("Cover the whole output (CPU deviation)", PLANNED), ("Loads are narrow → larger BLOCK", INK),
                   ("Scored no higher than the parent", INK)])
    rules.append(("On a stall: a KernelWiki technique page", PLANNED))
    for i, (rule, color) in enumerate(rules):
        d.text((622, 164 + 31 * i), "• " + rule, font=F_S, fill=color)
    return img, d


def frames():
    out = []
    for gen in (1, 2):
        g = f"generation {gen}"
        benched = gen == 2  # --gpu-bench-every 2: only even generations reach the GPU
        fails = FAILS_AT[gen]
        img, d = base(g, "Start from the current parent kernel")
        parent_card(d)
        out.append((img, 900))
        caption = ("One LLM call per generation; five candidates are drawn side by side to follow the checks"
                   if gen == 1 else "The prompt now carries the rules and the measured timings from before")
        img, d = base(g, caption)
        parent_card(d)
        for i, y in enumerate(LANES):
            d.line([(150, 300), (182, y)], fill=MUTED, width=2)
            candidate(d, STAGES[1][0], y, i, "new")
        if gen > 1:
            d.rounded_rectangle([162, 427, 282, 447], radius=10, fill=ACCENT)
            d.text((222, 437), "+ feedback", font=F_S, fill=CARD, anchor="mm")
        out.append((img, 1500))

        captions = {SYNTAX: "1. Python syntax check drops unparseable code",
                    COMPILE: "2. Ahead-of-time Triton compile drops made-up tl.* APIs and type errors",
                    SANITIZE: "3. Anti-gaming sanitizer (planned): library calls, cached results, monkeypatching",
                    EVALUATE: "4. Evaluator: static score; planned: CPU deviation, tcgen05, Petri-net timing"}
        for stage in (SYNTAX, COMPILE, SANITIZE, EVALUATE):
            img, d = base(g, captions[stage])
            parent_card(d)
            for i, y in enumerate(LANES):
                failed = fails[i] is not None and fails[i] <= stage
                candidate(d, STAGES[min(stage, fails[i] or EVALUATE)][0], y, i, "bad" if failed else "ok")
                if stage == EVALUATE and fails[i] is None:
                    score_bar(d, y, BARS[i], False)
            out.append((img, 1300))

        if benched:
            img, d = base(g, "5. Every Nth generation the survivors run on a real GPU: correctness first, then time")
            parent_card(d)
            for i, y in enumerate(LANES):
                if fails[i] is None:
                    candidate(d, STAGES[TIMED][0], y, i, "ok")
                    gpu_label(d, y, GPU[i], OK)
                else:
                    candidate(d, STAGES[fails[i]][0], y, i, "bad")
            out.append((img, 2200))

        img, d = base(g, "Fastest measured time wins and becomes the next parent" if benched
                      else "Highest static score (strictly better than the parent) becomes the new parent")
        parent_card(d, "old parent", MUTED)
        for i, y in enumerate(LANES):
            if fails[i] is None:
                candidate(d, STAGES[TIMED if benched else EVALUATE][0], y, i, "best" if i == BEST else "ok")
                if benched:
                    gpu_label(d, y, GPU[i], GOLD if i == BEST else OK)
                else:
                    score_bar(d, y, BARS[i], i == BEST)
            else:
                candidate(d, STAGES[fails[i]][0], y, i, "bad")
        # Drop just left of the elite's column: straight down would cut through the cards in the
        # lanes below, and the right margin holds the score bars and measured-time labels
        y, x = LANES[BEST], STAGES[TIMED if benched else EVALUATE][0]
        drop = x - 60
        d.line([(x - 40, y), (drop, y), (drop, 440), (95, 440), (95, 330)], fill=GOLD, width=3)
        d.polygon([(95, 330), (88, 342), (102, 342)], fill=GOLD)
        d.text((480, 436), "next generation", font=F_S, fill=GOLD, anchor="mb")
        out.append((img, 1800))
        out.append((feedback_frame(g, benched)[0], 2600))

    img, d = base("", "No GPU needed to search; one on the machine turns the proxy into a measurement",
                  stages=False)
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
        d.text((90, 96 + 37 * i), k, font=F_H, fill=PLANNED if planned else ACCENT)
        d.text((260, 97 + 37 * i), v, font=F_B, fill=INK)
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
