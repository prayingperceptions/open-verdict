"""Laya backend: local typed-decision model via laya-mlx.

laya-mlx is Apple-Silicon-only. If it's not installed/importable, this
backend raises RuntimeError with a clear message — use backend="heuristic"
instead, or install laya-mlx with `pip install laya-mlx`.
"""

from __future__ import annotations


def review(items: list[dict], rubric: str) -> list[dict]:
    try:
        import laya_mlx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The 'laya' backend requires laya-mlx, which is Apple-Silicon-only. "
            "Install it with `pip install laya-mlx` on an Apple Silicon Mac, "
            'or use backend="heuristic" (free, local, no dependencies).'
        ) from exc

    from laya_mlx import LayaBrain  # type: ignore[import-not-found]

    brain = LayaBrain()
    verdicts = []
    for item in items:
        item_id = item.get("id", "unknown")
        text = str(item.get("text") or "")
        # Keep the integration thin: ask the reflex layer for a typed
        # decision on this item, in one forward pass.
        result = brain.decide(
            prompt=f"Rubric: {rubric}\n\nAgent output:\n{text}",
            choices=["pass", "fail", "retry"],
        )
        choice = str(result.choice if hasattr(result, "choice") else result.get("choice", "retry"))
        score = float(result.score if hasattr(result, "score") else result.get("score", 0.5))
        choice = choice.lower().strip()
        if choice not in ("pass", "fail", "retry"):
            choice = "retry"
        confidence = max(0.0, min(1.0, score))
        verdicts.append(
            {
                "id": item_id,
                "verdict": choice,
                "confidence": round(confidence, 2),
                "reason": f"laya-mlx decision: {choice} (score {confidence:.2f})",
            }
        )
    return verdicts
