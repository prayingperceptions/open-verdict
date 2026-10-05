"""open-verdict: batch-review N agent outputs in ONE pass with typed verdicts."""

from __future__ import annotations

from . import backends  # noqa: F401  (kept for explicitness; backends live in registry)
from .backends import heuristic, llm

__version__ = "0.1.0"

Verdict = dict  # {"id", "verdict", "confidence", "reason"}

_BACKENDS = {
    "heuristic": heuristic.review,
    # "laya" and "llm" are imported lazily so a missing optional
    # dependency never breaks importing `verdict` itself.
}


def _available_backends() -> list[str]:
    names = sorted(set(list(_BACKENDS) + ["laya", "llm"]))
    return names


def review(items: list[dict], rubric: str, backend: str = "heuristic") -> list[dict]:
    """Review a batch of agent outputs in a single pass.

    Each item is a dict with at least "id" and "text". Returns a list of
    Verdict dicts: {"id", "verdict", "confidence", "reason"} where verdict
    is one of "pass", "fail", "retry" and confidence is in [0, 1].

    Unknown backend names raise ValueError listing the available backends.
    """
    if backend in _BACKENDS:
        return _BACKENDS[backend](items, rubric)
    if backend == "laya":
        from .backends import laya

        return laya.review(items, rubric)
    if backend == "llm":
        return llm.review(items, rubric)
    raise ValueError(
        f"Unknown backend {backend!r}. Available backends: {', '.join(_available_backends())}"
    )


__all__ = ["review", "Verdict", "__version__"]
