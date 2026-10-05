"""Heuristic backend: pure-stdlib, single-pass batch review.

No model calls, no API, no network. One pass over the batch:
  1. Rubric rules  — lines starting with "MUST:" / "MUST NOT:" are enforced
     as case-insensitive substring requirements on each item's text.
  2. Sanity checks — empty/whitespace-only text, absurdly short text,
     error markers ("error", "traceback", "failed", "null", "undefined",
     "todo"), and truncated-looking output.
  3. Cross-item checks — identical texts duplicated across the batch
     (a tell of a stuck/copied agent).

Confidence starts at 1.0 and drops per failed check, clamped to [0,1].
Every reason string names exactly what failed.
"""

from __future__ import annotations

import re

# Markers that almost always mean the agent crashed, bailed, or leaked internals.
ERROR_MARKERS = (
    "traceback",
    "exception",
    "undefined",
    "failed",
    "failure",
    "error",
    "null",
    "none type",
    "not implemented",
    "todo",
    "fixme",
    "xxx",
    "placeholder",
    "lorem ipsum",
)

# Markers that suggest the output was cut off mid-stream.
TRUNCATION_MARKERS = (
    "...",
    "[truncated]",
    "(truncated)",
    "…",
    "<cut off>",
    "continued...",
)

# Below this many characters of non-space text, the item barely said anything.
ABSURDLY_SHORT = 20
# Below this, it's questionable but could still be a terse valid answer.
SHORT_THRESHOLD = 50


def _parse_rubric(rubric: str) -> tuple[list[str], list[str]]:
    """Extract MUST: / MUST NOT: terms from a rubric string.

    Lines not starting with MUST:/MUST NOT: are ignored (they're guidance
    for model-based backends, not enforceable here). Terms are matched
    case-insensitively as substrings of the item text.
    """
    must, must_not = [], []
    for line in (rubric or "").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("MUST NOT:"):
            term = stripped.split(":", 1)[1].strip()
            if term:
                must_not.append(term)
        elif stripped.upper().startswith("MUST:"):
            term = stripped.split(":", 1)[1].strip()
            if term:
                must.append(term)
    return must, must_not


def _find_duplicates(items: list[dict]) -> set[str]:
    """Return ids of items whose text repeats an EARLIER item's text.

    The first occurrence is left clean; repeats are the tell of a
    stuck/copied agent.
    """
    seen: dict[str, str] = {}  # normalized text -> first id
    dupes: set[str] = set()
    for item in items:
        text = (item.get("text") or "")
        key = text.strip().lower()
        if not key:
            continue
        item_id = str(item.get("id", ""))
        if key in seen:
            dupes.add(item_id)
        else:
            seen[key] = item_id
    return dupes


def _score_item(
    item: dict,
    must: list[str],
    must_not: list[str],
    is_duplicate: bool,
) -> dict:
    item_id = item.get("id", "unknown")
    text = item.get("text", "")
    if text is None:
        text = ""
    text = str(text)
    lowered = text.lower()
    stripped = text.strip()

    confidence = 1.0
    failures: list[str] = []
    retries: list[str] = []

    # --- 1. Sanity: empty ---
    if not stripped:
        return {
            "id": item_id,
            "verdict": "fail",
            "confidence": 1.0,
            "reason": "empty or whitespace-only text",
        }

    # --- 2. Sanity: absurdly short ---
    if len(stripped) < ABSURDLY_SHORT:
        failures.append(f"absurdly short text ({len(stripped)} chars < {ABSURDLY_SHORT})")
        confidence -= 0.6
    elif len(stripped) < SHORT_THRESHOLD:
        retries.append(f"very short text ({len(stripped)} chars < {SHORT_THRESHOLD})")
        confidence -= 0.2

    # --- 3. Error markers ---
    for marker in ERROR_MARKERS:
        # word-boundary match so "terror" doesn't trip "error"
        if re.search(rf"\b{re.escape(marker)}\b", lowered):
            failures.append(f"contains error marker {marker!r}")
            confidence -= 0.4
            break  # one is enough to condemn

    # --- 4. Truncation markers ---
    for marker in TRUNCATION_MARKERS:
        if marker in text:
            retries.append(f"output looks truncated ({marker!r})")
            confidence -= 0.3
            break

    # --- 5. Rubric MUST: ---
    for term in must:
        if term.lower() not in lowered:
            failures.append(f"violates MUST: {term!r} (term missing from text)")
            confidence -= 0.5

    # --- 6. Rubric MUST NOT: ---
    for term in must_not:
        if re.search(rf"\b{re.escape(term.lower())}\b", lowered):
            failures.append(f"violates MUST NOT: {term!r} (term present in text)")
            confidence -= 0.5

    # --- 7. Duplicates within the batch ---
    if is_duplicate:
        retries.append("text is duplicated elsewhere in the batch (possible stuck agent)")
        confidence -= 0.25

    # --- Decide verdict ---
    if failures:
        verdict = "fail"
        reason = "; ".join(failures)
    elif retries:
        verdict = "retry"
        reason = "; ".join(retries)
    else:
        verdict = "pass"
        reason = "passed all checks"

    confidence = max(0.0, min(1.0, round(confidence, 2)))
    return {"id": item_id, "verdict": verdict, "confidence": confidence, "reason": reason}


def review(items: list[dict], rubric: str) -> list[dict]:
    """Judge the whole batch in ONE pass. Returns a list of Verdict dicts."""
    must, must_not = _parse_rubric(rubric)
    dupes = _find_duplicates(items)
    return [
        _score_item(item, must, must_not, str(item.get("id", "")) in dupes)
        for item in items
    ]
