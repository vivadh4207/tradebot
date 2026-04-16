"""News provider + simple sentiment classifier.

StaticNewsProvider: no external calls, for tests.
NewsAPIProvider:     (stub) — wire NewsAPI / Finnhub / Alpaca News.

Sentiment rules are intentionally conservative: negative news on a long
trade candidate is cause to de-weight or skip; positive on a short candidate
is cause to skip.
"""
from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class NewsItem:
    symbol: str
    headline: str
    source: str
    published_at: datetime
    url: str = ""


@dataclass
class NewsSentiment:
    symbol: str
    score: float       # -1..+1
    label: str         # 'negative' | 'neutral' | 'positive'
    rationale: str = ""
    items: List[NewsItem] = field(default_factory=list)

    def actionable_block(self, direction: str, threshold: float = 0.5) -> bool:
        """Does sentiment block a trade in the given direction?"""
        if direction == "bullish" and self.score < -threshold:
            return True
        if direction == "bearish" and self.score > threshold:
            return True
        return False


_POS_WORDS = {"beat", "beats", "record", "surge", "rally", "upgrade", "outperform",
              "tops", "soars", "strong", "raises", "approved", "buyback"}
_NEG_WORDS = {"miss", "missed", "downgrade", "plunge", "drop", "lawsuit", "investigation",
              "probe", "recall", "warns", "cuts", "slashes", "delays", "fraud", "bankruptcy"}


def score_headlines(headlines: List[str]) -> float:
    pos = neg = 0
    for h in headlines:
        hl = h.lower()
        pos += sum(1 for w in _POS_WORDS if w in hl)
        neg += sum(1 for w in _NEG_WORDS if w in hl)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class NewsProvider(abc.ABC):
    @abc.abstractmethod
    def fetch(self, symbol: str) -> List[NewsItem]: ...

    def sentiment(self, symbol: str) -> NewsSentiment:
        items = self.fetch(symbol)
        if not items:
            return NewsSentiment(symbol=symbol, score=0.0, label="neutral",
                                 rationale="no_news")
        score = score_headlines([n.headline for n in items])
        label = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")
        return NewsSentiment(symbol=symbol, score=score, label=label,
                             rationale=f"{len(items)}_items", items=items)


class StaticNewsProvider(NewsProvider):
    def __init__(self, data: Optional[Dict[str, List[NewsItem]]] = None):
        self._data: Dict[str, List[NewsItem]] = data or {}

    def fetch(self, symbol: str) -> List[NewsItem]:
        return list(self._data.get(symbol, []))


class CachedNewsSentiment:
    """Wraps a provider + classifier with a per-symbol TTL cache so we do
    not hit the news API or pay for an LLM call on every 3-minute tick.

    TTL default 300s matches typical news staleness: anything newer than
    5 minutes we treat as 'fresh'; we re-pull after that.
    """

    def __init__(self, provider: NewsProvider, classifier,
                 ttl_seconds: int = 300):
        self._provider = provider
        self._classifier = classifier
        self._ttl = ttl_seconds
        self._cache: Dict[str, tuple] = {}   # symbol → (expires_at, NewsSentiment)

    def sentiment(self, symbol: str) -> "NewsSentiment":
        import time as _t
        now = _t.time()
        hit = self._cache.get(symbol)
        if hit and hit[0] > now:
            return hit[1]
        items = self._provider.fetch(symbol)
        if not items:
            ns = NewsSentiment(symbol=symbol, score=0.0, label="neutral",
                                rationale="no_news")
        else:
            score, rationale = self._classifier.score(items)
            label = ("positive" if score > 0.2
                     else ("negative" if score < -0.2 else "neutral"))
            ns = NewsSentiment(symbol=symbol, score=score, label=label,
                                rationale=rationale, items=items)
        self._cache[symbol] = (now + self._ttl, ns)
        return ns
