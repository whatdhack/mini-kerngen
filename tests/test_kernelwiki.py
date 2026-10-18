"""KernelWiki retrieval (against the pinned submodule) and the symptom mapping that drives it."""
import pytest

from knowledge import KernelWiki, symptoms_for
from knowledge.kernelwiki import NOT_ACTIONABLE, parse_page

WIKI = KernelWiki()
needs_wiki = pytest.mark.skipif(not WIKI.available, reason="external/KernelWiki is not checked out "
                                                            "(git submodule update --init)")

PAGE = """# wiki/techniques/example.md

---
id: technique-example
title: "An Example"
type: technique
architectures: [sm100, sm90]
confidence: experimental
candidate_techniques: [technique-a, technique-b]
---

# An Example

First paragraph of the idea.

```python
code that must not reach the prompt
```

| a | table |

Second paragraph.
"""


def test_parse_page_reads_frontmatter_and_skips_code():
    page = parse_page(PAGE)
    assert page.id == "technique-example" and page.title == "An Example" and page.confidence == "experimental"
    assert page.architectures == ["sm100", "sm90"] and page.candidate_techniques == ["technique-a", "technique-b"]
    assert page.summary == "First paragraph of the idea. Second paragraph."


@needs_wiki
def test_pipeline_stalls_lead_to_pipeline_stages():
    technique, block = WIKI.prompt_block([("pipeline-stalls", "3 stages needed")], tried=set())
    assert technique == "technique-pipeline-stages"
    assert block.startswith("[KERNELWIKI technique-pipeline-stages | confidence: source-reported")
    assert "pattern-pipeline-stalls" in block and "num_stages" in block and "do not copy" in block


@needs_wiki
def test_tried_techniques_rotate_and_unactionable_ones_are_skipped():
    tried = set()
    for _ in range(3):
        technique, _ = WIKI.prompt_block([("register-spilling", "spills"), ("compute-bound", "default")], tried)
        assert technique not in NOT_ACTIONABLE and technique not in tried
        tried.add(technique)


def test_missing_wiki_yields_no_block(tmp_path):
    wiki = KernelWiki(str(tmp_path))
    assert not wiki.available and wiki.prompt_block([("memory-bound", "x")], set()) is None


def test_symptoms_from_the_petri_net_come_first():
    evaluator = {"fitness_score": 0.6, "metrics": {"bottleneck": "TMA_MEMORY_BOUND", "stall_cause": "latency",
                                                   "bubble_ratio": 0.25, "required_stages": 3, "waves": 4,
                                                   "last_wave_fill": 0.46, "pipeline": {"t_load": 384, "t_mma": 512}}}
    static = {"spill_bytes": 64, "registers": 240, "occupancy": 0.1, "tcgen05": 4}
    found = [s for s, _ in symptoms_for(static, evaluator, gemm=True)]
    assert found[:5] == ["pipeline-stalls", "wave-quantization", "register-spilling", "register-pressure",
                         "low-occupancy"]


def test_symptoms_fall_back_to_the_problem_kind():
    assert symptoms_for({}, None, gemm=True) == [("compute-bound", "GEMM-like problem; no more specific symptom measured")]
    found = [s for s, _ in symptoms_for({"wide_load_frac": 0.5, "occupancy": 0.75}, None, gemm=False)]
    assert found == ["memory-bound"]
