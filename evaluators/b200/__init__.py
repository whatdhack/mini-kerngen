"""B200 (sm_100) evaluator: GPU-less gates and predictions for candidate kernels.

  hw.py                  constants, each with its source
  functional_tcgen05.py  Tier 1: bit-exact tcgen05 arithmetic (tcgen05_model) for formats, scales, sparsity
  petrinet_pipeline.py   Tier 2: timed Petri net of the TMA -> tcgen05.mma -> epilogue pipeline
  b200_evaluator.py      the evaluate_candidate() API: hard gates, Tier 1, Tier 2, fitness
  triton_adapter.py      compiled Triton candidate -> evaluator spec
"""
from .b200_evaluator import B200Evaluator
from .triton_adapter import spec_from_static

__all__ = ["B200Evaluator", "spec_from_static"]
