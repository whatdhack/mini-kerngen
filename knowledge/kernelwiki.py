"""Read hook into KernelWiki (mit-han-lab/KernelWiki, pinned as a submodule at external/KernelWiki).

When the search stalls, the elite kernel's symptoms (knowledge/symptoms.py) pick a KernelWiki
pattern page, the pattern names candidate techniques, and the first technique not yet tried in
this run becomes a block in the next prompt. That prompt is still the generation's one LLM call.

The wiki's own scripts run unchanged as subprocesses: `query.py --symptom S --type pattern`
(pattern pages carry no architectures field, so no --architecture filter) and `get_page.py`.
Results are cached for the run; a missing wiki, a timeout or a failing script just means no block.

KernelWiki is CUDA, CuTe and PTX first, and its confidence ranges from verified to experimental.
The block therefore carries the page's idea and confidence, how the technique is expressed in
Triton, and an explicit instruction not to copy CUDA syntax: CUDA idioms turning into
nonexistent tl.* calls is the commonest failure feedback.py deals with.
"""
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "KernelWiki")
SUMMARY_CHARS = 600

# How each technique KernelWiki proposes is expressed in Triton
TRITON_FORM = {
    "technique-pipeline-stages": "raise num_stages at the launch (or tl.range(..., num_stages=N)) so more K tiles are in flight",
    "technique-double-buffering": "num_stages >= 2, so the next K tile loads while the current one is multiplied",
    "technique-warp-specialization": "write the K loop as `for k in tl.range(0, k_tiles, warp_specialize=True):`",
    "technique-persistent-kernels": "launch one program per SM (grid = NUM_SMS) and loop over output tiles with "
                                    "tl.range(pid, num_tiles, NUM_SMS)",
    "technique-tile-scheduling": "visit output tiles in groups of GROUP_SIZE_M rows (Triton's grouped ordering) for L2 reuse",
    "technique-swizzling": "use grouped tile ordering (GROUP_SIZE_M); Triton swizzles shared memory itself, so do not pad by hand",
    "technique-vectorized-loads": "have each program load a large contiguous block so Triton emits 16-byte vector loads",
    "technique-register-budgeting": "lower BLOCK sizes or num_warps until registers stop spilling",
    "technique-epilogue-fusion": "apply elementwise post-processing in registers before tl.store, not in a second kernel",
    "technique-kernel-fusion": "fuse adjacent operations into one @triton.jit kernel instead of a round trip through memory",
    "technique-chunk-parallelism": "split the long axis into chunks handled by separate programs, then combine the partials",
    "technique-cache-policy": "pass eviction_policy='evict_last' (or a cache_modifier) to tl.load for data that is reused",
    "technique-ping-pong-scheduling": "give each program two tiles so one's math overlaps the other's loads (warp_specialize)",
    "hw-tma": "use device-side TMA descriptors: tl.make_tensor_descriptor(...) and desc.load([m, k])",
}
# Techniques a Triton kernel cannot act on (no equivalent, or the compiler already does it): never
# worth the explore generation. hw-tmem / register-to-tmem: tl.dot on Blackwell accumulates in TMEM
# by itself; hw-clc and 2-SM cooperative MMA have no Triton surface.
NOT_ACTIONABLE = frozenset(("hw-tmem", "migration-register-to-tmem", "hw-clc", "hw-2sm-cooperative"))


@dataclass
class Page:
    id: str
    title: str = ""
    confidence: str = ""
    architectures: List[str] = field(default_factory=list)
    candidate_techniques: List[str] = field(default_factory=list)
    summary: str = ""


def _field(frontmatter: str, name: str) -> str:
    match = re.search(rf"^{name}:\s*(.*)$", frontmatter, re.MULTILINE)
    return match.group(1).strip().strip('"').strip("'") if match else ""


def _list(frontmatter: str, name: str) -> List[str]:
    value = _field(frontmatter, name)
    return [v.strip().strip('"').strip("'") for v in value.strip("[]").split(",") if v.strip()] if value else []


def parse_page(text: str) -> Optional[Page]:
    """A page from get_page.py's output: a '# path' line, YAML frontmatter, then the markdown body."""
    match = re.search(r"^---\s*$(.*?)^---\s*$(.*)", text, re.MULTILINE | re.DOTALL)
    if not match:
        return None
    frontmatter, body = match.groups()
    page_id = _field(frontmatter, "id")
    if not page_id:
        return None
    paragraphs, in_code = [], False
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if block.startswith("```"):
            in_code = not in_code if block.count("```") % 2 else in_code
            continue
        if in_code or not block or block.startswith(("#", "|", "-", ">", "<")):
            continue
        paragraphs.append(" ".join(block.split()))
        if sum(len(p) for p in paragraphs) >= SUMMARY_CHARS:
            break
    summary = " ".join(paragraphs)
    if len(summary) > SUMMARY_CHARS:
        summary = summary[:SUMMARY_CHARS].rsplit(" ", 1)[0] + " …"
    return Page(page_id, _field(frontmatter, "title"), _field(frontmatter, "confidence"),
                _list(frontmatter, "architectures"), _list(frontmatter, "candidate_techniques"), summary)


class KernelWiki:
    def __init__(self, root: str = DEFAULT_ROOT, timeout_s: float = 30.0):
        self.root = root
        self.timeout_s = timeout_s
        self._cache: Dict[Tuple[str, ...], str] = {}

    @property
    def available(self) -> bool:
        return os.path.isfile(os.path.join(self.root, "scripts", "query.py"))

    def _run(self, script: str, *args: str) -> str:
        key = (script,) + args
        if key not in self._cache:
            try:
                proc = subprocess.run([sys.executable, os.path.join(self.root, "scripts", script), *args],
                                      cwd=self.root, capture_output=True, text=True, timeout=self.timeout_s)
                self._cache[key] = proc.stdout if proc.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                self._cache[key] = ""
        return self._cache[key]

    def patterns_for(self, symptom: str) -> List[str]:
        """Paths of the pattern pages that list `symptom`, the page named after it first."""
        out = self._run("query.py", "--symptom", symptom, "--type", "pattern", "--paths-only")
        paths = [line.strip() for line in out.splitlines() if line.strip().endswith(".md")]
        return sorted(paths, key=lambda p: os.path.basename(p) != f"{symptom}.md")

    def page(self, page_id_or_path: str) -> Optional[Page]:
        return parse_page(self._run("get_page.py", page_id_or_path))

    def prompt_block(self, symptoms: Iterable[Tuple[str, str]], tried: Set[str]) -> Optional[Tuple[str, str]]:
        """(technique id, prompt block) for the first untried technique the symptoms lead to, or None.

        `symptoms` are (symptom, why) pairs, most important first.
        """
        for symptom, why in symptoms:
            for path in self.patterns_for(symptom):
                pattern = self.page(path)
                if pattern is None:
                    continue
                for technique_id in pattern.candidate_techniques:
                    if technique_id in tried or technique_id in NOT_ACTIONABLE or technique_id not in TRITON_FORM:
                        continue
                    technique = self.page(technique_id)
                    if technique is None:
                        continue
                    return technique_id, render_block(technique, pattern, symptom, why)
        return None


def render_block(technique: Page, pattern: Page, symptom: str, why: str) -> str:
    archs = ", ".join(technique.architectures) or "any"
    lines = [f"[KERNELWIKI {technique.id} | confidence: {technique.confidence or 'unknown'} | {archs}]",
             f"Retrieved because: symptom `{symptom}` ({why}) -> {pattern.id}.",
             f"Technique: {technique.title}. {technique.summary}"]
    form = TRITON_FORM.get(technique.id)
    if form:
        lines.append(f"In Triton: {form}.")
    lines.append("KernelWiki's examples are CUDA, CuTe or PTX: take the idea, write Triton only, and do not copy "
                 "their syntax or invent tl.* functions.")
    return "\n".join(lines)
