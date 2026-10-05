# open-verdict

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Jev, but free and local.** Batch-review N agent outputs in ONE pass with typed verdicts + confidence scores — no API, no per-token bill.

## The pain

Agent swarms execute hundreds of results in parallel — then review them **one at a time through a big model**. A serial queue hiding inside a parallel system. The parallelism you paid for in execution gets erased at review time, and hosted decision APIs charge you per token for the privilege (~$0.042/M input tokens, every item, every run).

## The one-pass fix

`review()` takes the whole batch and judges it in **a single pass** — one call, no per-item round trips:

```python
from verdict import review

verdicts = review(items, rubric, backend="heuristic")
# [{"id": ..., "verdict": "pass"|"fail"|"retry", "confidence": 0.85, "reason": ...}, ...]
```

The default backend is pure stdlib: rubric `MUST:`/`MUST NOT:` rules, sanity checks (empty, truncated, error markers), and cross-batch duplicate detection. Confidence starts at 1.0 and drops per failed check.

## Quickstart

```bash
pip install -e .
```

```python
from verdict import review

items = [
    {"id": "a1", "text": "We apologize for the delay; your refund is processed."},
    {"id": "a2", "text": "Traceback (most recent call last): ..."},
]
verdicts = review(items, "MUST: refund\nMUST NOT: policy")
for v in verdicts:
    print(v["verdict"], v["confidence"], "—", v["reason"])
```

## Backends

| Backend     | Cost | Requirements                          | When to use                          |
|-------------|------|---------------------------------------|--------------------------------------|
| `heuristic` | $0   | none (stdlib only) — **default**      | fast triage, CI gates, obvious fails |
| `laya`      | $0   | laya-mlx (Apple Silicon only)         | local neural verdicts on a Mac       |
| `llm`       | per-token | `OPENAI_API_KEY` env var (optional) | ambiguous cases needing a big model  |

Honest framing: hosted decision APIs charge per token for every judgement. open-verdict's default path is free and local — the heuristic backend catches the empty strings, tracebacks, TODOs, missing required terms, and duplicated outputs that make up the bulk of swarm failures, in milliseconds. Escalate only the ambiguous survivors to a model.

## Jaw-drop demo

```bash
python examples/review_300.py
```

Generates 300 fake agent outputs (clean answers, tracebacks, empty strings, truncated output, missing required terms, duplicated texts) and reviews **all 300 in one `review()` call**, printing wall-clock seconds, verdict distribution, and verdicts/sec. Cost: $0.00. API calls: 0.

## License

MIT — see [LICENSE](LICENSE).
