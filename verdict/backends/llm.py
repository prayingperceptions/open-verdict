"""LLM backend (OPTIONAL): judge via a hosted chat-completions API.

This backend is opt-in. It reads the API key from the OPENAI_API_KEY
environment variable. If the key is missing, review() raises RuntimeError
with instructions — the heuristic backend is the free, local default and
needs nothing.

Implemented with stdlib urllib only. Minimal by design: batch the items
into one prompt, ask for one JSON verdict per item.
"""

from __future__ import annotations

import json
import os
import urllib.request


_ENDPOINT = "https://api.openai.com/v1/chat/completions"
_MODEL = "gpt-4o-mini"


def _prompt(items: list[dict], rubric: str) -> str:
    numbered = "\n\n".join(
        f"### item {i} (id: {item.get('id', 'unknown')})\n{str(item.get('text') or '')[:4000]}"
        for i, item in enumerate(items)
    )
    return (
        "You are a strict judge of AI agent outputs. Rubric:\n"
        f"{rubric}\n\n"
        "Judge EACH item below. Reply with a JSON array, one object per item, "
        "in the same order, with keys: id, verdict (one of pass/fail/retry), "
        'confidence (0.0-1.0), reason (short string naming what passed/failed).\n\n'
        f"{numbered}"
    )


def review(items: list[dict], rubric: str) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "The 'llm' backend is optional and requires an API key. "
            "Set the OPENAI_API_KEY environment variable to enable it, "
            'or use backend="heuristic" — free, local, no key needed.'
        )

    body = json.dumps(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": _prompt(items, rubric)}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode()

    req = urllib.request.Request(
        _ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())

    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    raw = parsed["verdicts"] if isinstance(parsed, dict) and "verdicts" in parsed else parsed

    verdicts = []
    for entry, item in zip(raw, items):
        v = str(entry.get("verdict", "retry")).lower().strip()
        if v not in ("pass", "fail", "retry"):
            v = "retry"
        c = float(entry.get("confidence", 0.5))
        verdicts.append(
            {
                "id": item.get("id", "unknown"),
                "verdict": v,
                "confidence": max(0.0, min(1.0, c)),
                "reason": str(entry.get("reason", "llm judgement")),
            }
        )
    return verdicts
