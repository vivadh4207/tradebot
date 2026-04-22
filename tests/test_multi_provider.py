"""MultiProvider + provider shim tests. No real HTTP calls — we fake
providers with the MarketDataProvider protocol."""
from __future__ import annotations

import statistics
from typing import List, Optional

import pytest

from src.data.multi_provider import MultiProvider
from src.data.providers.base import (
    ProviderNewsItem, ProviderOptionRow, ProviderQuote,
)


class _Fake:
    def __init__(self, name, *, enabled=True, quote=None,
                 chain=None, news=None, sentiment=None, raise_exc=None):
        self.name = name
        self._enabled = enabled
        self._quote = quote
        self._chain = chain
        self._news = news
        self._sentiment = sentiment
        self._raise = raise_exc

    def is_enabled(self):
        return self._enabled

    def latest_quote(self, symbol):
        if self._raise: raise self._raise
        return self._quote

    def option_chain(self, underlying, expiry=None):
        if self._raise: raise self._raise
        return self._chain

    def news(self, symbol=None, limit=20):
        if self._raise: raise self._raise
        return self._news

    def news_sentiment(self, symbol):
        return self._sentiment


def test_disabled_providers_are_skipped():
    a = _Fake("a", enabled=False)
    b = _Fake("b", enabled=True,
              quote=ProviderQuote("SPY", 100, 101, 100.5, source="b"))
    mp = MultiProvider([a, b])
    q = mp.latest_quote("SPY")
    assert q is not None and q.source == "b"


def test_quote_agreement_returns_one():
    a = _Fake("a", quote=ProviderQuote("SPY", 100, 101, 100.5, source="a"))
    b = _Fake("b", quote=ProviderQuote("SPY", 100.1, 101.1, 100.6, source="b"))
    mp = MultiProvider([a, b], quote_disagreement_pct=0.01)
    q = mp.latest_quote("SPY")
    assert q is not None
    # Either is fine — prices agree within 0.01.
    assert q.source in ("a", "b")


def test_quote_disagreement_picks_median():
    # 3 sources: a=100, b=105, c=110 — median picks b
    a = _Fake("a", quote=ProviderQuote("SPY", 100, 100, 100, source="a"))
    b = _Fake("b", quote=ProviderQuote("SPY", 105, 105, 105, source="b"))
    c = _Fake("c", quote=ProviderQuote("SPY", 110, 110, 110, source="c"))
    mp = MultiProvider([a, b, c], quote_disagreement_pct=0.005)
    q = mp.latest_quote("SPY")
    assert q is not None
    assert q.source == "b"        # median picked
    assert q.mid == 105


def test_none_quotes_all_around_returns_none():
    a = _Fake("a", quote=None)
    b = _Fake("b", quote=None)
    mp = MultiProvider([a, b])
    assert mp.latest_quote("SPY") is None


def test_chain_takes_longest():
    short = [ProviderOptionRow("x", "SPY", 100, "2026-05-02", "call", source="a")]
    long_ = [ProviderOptionRow(f"x{i}", "SPY", 100 + i, "2026-05-02", "call", source="b")
              for i in range(10)]
    a = _Fake("a", chain=short)
    b = _Fake("b", chain=long_)
    mp = MultiProvider([a, b])
    chain = mp.option_chain("SPY")
    assert chain is not None
    assert len(chain) == 10


def test_news_deduped_and_pooled():
    a = _Fake("a", news=[
        ProviderNewsItem(ts="2026-04-20T00:00:00Z", headline="Fed holds rates",
                         source="a"),
        ProviderNewsItem(ts="2026-04-20T00:30:00Z", headline="SPY hits ATH",
                         source="a"),
    ])
    b = _Fake("b", news=[
        ProviderNewsItem(ts="2026-04-20T00:10:00Z", headline="Fed holds rates",
                         source="b"),      # dupe — same headline
        ProviderNewsItem(ts="2026-04-20T01:00:00Z", headline="QQQ volume surge",
                         source="b"),
    ])
    mp = MultiProvider([a, b])
    items = mp.news("SPY", limit=10)
    heads = {x.headline for x in items}
    assert "Fed holds rates" in heads
    assert "SPY hits ATH" in heads
    assert "QQQ volume surge" in heads
    assert len(items) == 3              # dupe collapsed


def test_provider_exception_does_not_break_others():
    a = _Fake("a", raise_exc=RuntimeError("boom"))
    b = _Fake("b", quote=ProviderQuote("SPY", 100, 101, 100.5, source="b"))
    mp = MultiProvider([a, b])
    q = mp.latest_quote("SPY")
    assert q is not None
    assert q.source == "b"


def test_sentiment_mean():
    a = _Fake("a", sentiment=0.6)
    b = _Fake("b", sentiment=-0.2)
    mp = MultiProvider([a, b])
    s = mp.news_sentiment("SPY")
    assert s is not None
    assert abs(s - 0.2) < 1e-6


def test_from_env_builds_without_keys(monkeypatch):
    # Unset all provider keys — from_env should still return a working
    # MultiProvider (empty active list).
    for k in ("POLYGON_API_KEY", "POLYGON_KEY",
              "TRADIER_TOKEN",
              "FINNHUB_KEY", "FINNHUB_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    mp = MultiProvider.from_env()
    assert mp.active_providers() == []
    assert mp.latest_quote("SPY") is None
