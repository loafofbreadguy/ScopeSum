"""Scope-aligned summary tokens for code LLMs.

The pure-Python/NumPy core (no torch) is:
``scope_tracker``, ``segmenter``, ``sum_insertion``, ``masks``, ``graph_edges``.
Torch/transformers code lives in ``model``, ``train`` and the ``eval_*`` modules
and is imported lazily so the core stays testable on CPU without torch.
"""

__version__ = "0.1.0"
