"""News sentiment classifiers.

- KeywordClassifier: fast, free, weak signal. Default.
- ClaudeNewsClassifier: LLM-based, far better, batched. Requires ANTHROPIC_API_KEY.

Classifiers return (score, rationale) where score ∈ [-1, +1].
"""
from __future__ import annotations

import abc
import json
import os
from typing import List, Tuple

from .news import NewsItem, score_headlines


class NewsClassifier(abc.ABC):
    @abc.abstractmethod
    def score(self, items: List[NewsItem]) -> Tuple[float, str]: ...


class KeywordClassifier(NewsClassifier):
    def score(self, items: List[NewsItem]) -> Tuple[float, str]:
        if not items:
            return 0.0, "no_items"
        s = score_headlines([i.headline for i in items])
        return s, f"keywords n={len(items)}"


class ClaudeNewsClassifier(NewsClassifier):
    """Uses Claude to classify a batch of headlines. One API call per score().

    Cheap and good: sonnet-class model on a batch of 10-20 headlines costs
    fractions of a cent per call. Falls back to KeywordClassifier on error.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_headlines: int = 20):
        self.model = model
        self.max_headlines = max_headlines
        self._client = None
        self._fallback = KeywordClassifier()
        try:
            import anthropic
            key = os.getenv("ANTHROPIC_API_KEY")
            if key:
                self._client = anthropic.Anthropic(api_key=key)
        except Exception:
            self._client = None

    def score(self, items: List[NewsItem]) -> Tuple[float, str]:
        if not items:
            return 0.0, "no_items"
        if self._client is None:
            return self._fallback.score(items)
        heads = [it.headline for it in items[: self.max_headlines]]
        prompt = (
            "You are a financial news sentiment classifier. For the headlines "
            "below (all tagged to one ticker), return JSON with keys:\n"
            '  "score" (number in [-1,1]), "rationale" (<=120 chars)\n'
            "Score strongly negative for downgrades, misses, FDA rejections, "
            "probes, lawsuits, guidance cuts. Strongly positive for beats, "
            "upgrades, approvals, buybacks, raises. Neutral for routine news.\n\n"
            "Headlines:\n" + "\n".join(f"- {h}" for h in heads) +
            "\n\nRespond with JSON only. No prose."
        )
        try:
            msg = self._client.messages.create(
                model=self.model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text if msg.content else "{}"
            text = raw.strip()
            # strip code fences if present
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{"): text.rfind("}") + 1]
            data = json.loads(text)
            score = float(data.get("score", 0.0))
            score = max(-1.0, min(1.0, score))
            rationale = str(data.get("rationale", ""))[:120]
            return score, f"claude: {rationale}"
        except Exception as e:
            return self._fallback.score(items)


def build_classifier() -> NewsClassifier:
    """Pick a classifier based on env. Falls back to keywords."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeNewsClassifier()
    return KeywordClassifier()
