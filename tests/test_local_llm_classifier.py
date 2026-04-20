"""Tests for LocalLLMNewsClassifier.

We cannot require llama-cpp-python / a GGUF model in the test matrix, so
these tests exercise the fallback paths and the build_classifier()
priority logic. A real on-Jetson integration test lives in
`deploy/jetson/scripts/benchmark_llm.sh`.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from src.intelligence.news import NewsItem
from src.intelligence.news_classifier import (
    KeywordClassifier, ClaudeNewsClassifier, build_classifier,
)
from src.intelligence.news_classifier_local import LocalLLMNewsClassifier


def _items():
    return [NewsItem(symbol="X", headline="Company beats earnings big-time",
                     source="t", published_at=datetime.now(tz=timezone.utc))]


def test_local_llm_returns_score_zero_on_empty_items():
    c = LocalLLMNewsClassifier(model_path="/nonexistent/model.gguf")
    score, rationale = c.score([])
    assert score == 0.0


def test_local_llm_falls_back_to_keyword_when_model_missing():
    c = LocalLLMNewsClassifier(model_path="/nonexistent/model.gguf")
    score, rationale = c.score(_items())
    assert isinstance(score, float)
    assert -1 <= score <= 1
    # keyword fallback tags as 'keywords'
    assert "keyword" in rationale.lower() or "keywords" in rationale.lower()


def test_build_classifier_prefers_local_when_path_set(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bogus = tmp_path / "bogus.gguf"
    bogus.write_bytes(b"not a real model")
    monkeypatch.setenv("LLM_MODEL_PATH", str(bogus))
    c = build_classifier()
    # The "model" we wrote is invalid, so the LLM will refuse to load — the
    # factory should gracefully fall back to keyword.
    assert isinstance(c, KeywordClassifier)


def test_build_classifier_falls_back_to_claude_without_path(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    c = build_classifier()
    assert isinstance(c, ClaudeNewsClassifier)


def test_build_classifier_falls_back_to_keyword_without_anything(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = build_classifier()
    assert isinstance(c, KeywordClassifier)
