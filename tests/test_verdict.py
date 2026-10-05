"""Tests for open-verdict."""

import pytest

from verdict import __version__, review


def test_version():
    assert __version__ == "0.1.0"


def test_verdict_dict_shape():
    items = [{"id": "a", "text": "A perfectly fine, sufficiently long agent output with no problems at all."}]
    (v,) = review(items, "no rules here")
    assert set(v.keys()) == {"id", "verdict", "confidence", "reason"}
    assert v["verdict"] in {"pass", "fail", "retry"}
    assert v["id"] == "a"


def test_confidence_within_bounds():
    texts = [
        "",
        "   ",
        "x",
        "ok",
        "Traceback (most recent call last): RuntimeError: boom",
        "TODO fix this later",
        "null",
        "A" * 500,
        "We apologize and have issued your refund today.",
    ]
    items = [{"id": f"t{i}", "text": t} for i, t in enumerate(texts)]
    verdicts = review(items, "MUST: refund\nMUST NOT: policy")
    assert len(verdicts) == len(items)
    for v in verdicts:
        assert isinstance(v["confidence"], float)
        assert 0.0 <= v["confidence"] <= 1.0, f"confidence out of bounds: {v}"


def test_unknown_backend_raises_clean_valueerror():
    with pytest.raises(ValueError) as excinfo:
        review([{"id": "a", "text": "hello world, plenty long enough"}], "rubric", backend="nope")
    msg = str(excinfo.value)
    assert "nope" in msg
    for name in ("heuristic", "laya", "llm"):
        assert name in msg


def test_batch_of_300_single_call():
    items = [
        {"id": f"item-{i:03d}", "text": f"Agent output number {i}: a reasonably complete answer."}
        for i in range(300)
    ]
    verdicts = review(items, "MUST: answer")  # one call, no per-item round trips
    assert len(verdicts) == 300
    assert [v["id"] for v in verdicts] == [f"item-{i:03d}" for i in range(300)]


def test_heuristic_catches_must_violation():
    items = [{"id": "m1", "text": "This is a long enough answer but it never says the required word."}]
    (v,) = review(items, "MUST: platypus", backend="heuristic")
    assert v["verdict"] == "fail"
    assert "platypus" in v["reason"]
    assert "MUST" in v["reason"]


def test_heuristic_catches_must_not_violation():
    items = [{"id": "m2", "text": "This answer is long enough but mentions the forbidden policy word explicitly."}]
    (v,) = review(items, "MUST NOT: policy", backend="heuristic")
    assert v["verdict"] == "fail"
    assert "policy" in v["reason"]
    assert "MUST NOT" in v["reason"]


def test_heuristic_catches_empty_text():
    for text in ("", "   \n\t  "):
        (v,) = review([{"id": "e", "text": text}], "anything", backend="heuristic")
        assert v["verdict"] == "fail"
        assert v["confidence"] == 1.0
        assert "empty" in v["reason"]


def test_heuristic_catches_error_markers():
    (v,) = review(
        [{"id": "x", "text": "Traceback (most recent call last): something exploded badly"}],
        "anything",
        backend="heuristic",
    )
    assert v["verdict"] == "fail"
    assert "traceback" in v["reason"]


def test_heuristic_passes_clean_output():
    (v,) = review(
        [{"id": "ok", "text": "We sincerely apologize for the delay; your refund has been fully processed today."}],
        "MUST: refund\nMUST: apologize",
        backend="heuristic",
    )
    assert v["verdict"] == "pass"
    assert v["confidence"] == 1.0


def test_laya_backend_missing_dependency_raises_clear_error():
    try:
        import laya_mlx  # noqa: F401
        pytest.skip("laya-mlx is installed; skipping missing-dependency check")
    except ImportError:
        pass
    from verdict.backends import laya

    with pytest.raises(RuntimeError) as excinfo:
        laya.review([{"id": "a", "text": "hello"}], "rubric")
    msg = str(excinfo.value).lower()
    assert "apple" in msg or "apple-silicon" in msg
    assert "pip install laya-mlx" in str(excinfo.value)
    assert "heuristic" in str(excinfo.value)


def test_llm_backend_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from verdict.backends import llm

    with pytest.raises(RuntimeError) as excinfo:
        llm.review([{"id": "a", "text": "hello"}], "rubric")
    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    assert "heuristic" in msg
