"""OptionsResearchAgent — snapshot shape + LLM-output parsing."""
from __future__ import annotations

from typing import List, Optional

from src.data.providers.base import (
    ProviderNewsItem, ProviderOptionRow, ProviderQuote,
)
from src.intelligence.options_research import (
    OptionsResearchAgent, _best_contracts_near_atm, _parse_llm_json, TradeIdea,
)


class _FakeMP:
    """Minimal MultiProvider stub."""
    def __init__(self, *, spot=100.0, chain=None, sentiment=0.1,
                 news=None, quotes=None):
        self._spot = spot
        self._chain = chain or []
        self._sentiment = sentiment
        self._news = news or []
        self._quotes = quotes or []

    def active_providers(self): return ["polygon", "finnhub"]
    def latest_quote(self, sym): return self._quotes[0] if self._quotes else None
    def all_quotes(self, sym):
        return self._quotes or [
            ProviderQuote(sym, self._spot - 0.1, self._spot + 0.1,
                          self._spot, source="polygon"),
        ]
    def option_chain(self, sym, expiry=None): return self._chain
    def news(self, sym=None, limit=20): return self._news[:limit]
    def news_sentiment(self, sym): return self._sentiment


def _mk_chain(spot=100):
    """10 strikes around spot, calls + puts, with OI + IV."""
    rows = []
    for i in range(-5, 6):
        K = spot + i
        for right in ("call", "put"):
            rows.append(ProviderOptionRow(
                symbol=f"SPY2604{i:+03d}{right[0].upper()}",
                underlying="SPY", strike=float(K), expiry="2026-05-02",
                right=right,
                bid=2.0 + abs(i) * 0.1, ask=2.2 + abs(i) * 0.1,
                last=2.1 + abs(i) * 0.1,
                volume=500 + 50 * abs(i),
                open_interest=5000 - 300 * abs(i),
                implied_vol=0.18 + 0.002 * abs(i),
                delta=0.50 - 0.1 * i,
                source="polygon",
            ))
    return rows


def test_best_contracts_near_atm_picks_closest_each_side():
    chain = _mk_chain(spot=100)
    picks = _best_contracts_near_atm(chain, spot=100, n_each_side=3)
    # 3 nearest-ATM calls + 3 nearest-ATM puts
    assert len(picks) == 6
    assert all(abs(p["strike"] - 100) <= 3 for p in picks)


def test_parse_llm_json_handles_preamble():
    raw = 'sure, here you go: {"ideas": [{"symbol":"SPY","direction":"put","strike":710}], "notes": "ok"}'
    parsed = _parse_llm_json(raw)
    assert parsed is not None
    assert parsed["ideas"][0]["strike"] == 710


def test_parse_llm_json_none_on_garbage():
    assert _parse_llm_json("just some prose, no braces") is None
    assert _parse_llm_json("") is None


def test_agent_report_with_no_llm_model_available(monkeypatch):
    """When Ollama isn't reachable, agent returns a report with
    ideas=[] but still populates snapshot fields."""
    import src.intelligence.options_research as mod

    class _DeadClient:
        class cfg:
            timeout_sec = 1.0
        def ping(self): return False
    monkeypatch.setattr(
        "src.intelligence.ollama_client.build_ollama_client",
        lambda: _DeadClient(),
    )
    agent = OptionsResearchAgent(_FakeMP(spot=100, chain=_mk_chain()))
    rep = agent.run(["SPY"])
    assert rep.underlyings == ["SPY"]
    assert rep.spot_by_symbol.get("SPY") is not None
    assert rep.ideas == []


def test_agent_parses_ideas_when_llm_returns_json(monkeypatch):
    """Agent correctly parses JSON response into TradeIdea objects."""
    import src.intelligence.options_research as mod

    raw = (
        '{"ideas":['
        ' {"symbol":"SPY","direction":"put","strike":705,'
        '   "expiry":"2026-05-02","entry":3.0,"profit_target":4.8,'
        '   "stop_loss":1.8,"time_horizon":"1-3d","confidence":"high",'
        '   "rationale":"VIX up 8%, RSI 24, breadth 2.8x decliners"},'
        ' {"symbol":"QQQ","direction":"call","strike":420,'
        '   "expiry":"2026-05-09","entry":4.1,"profit_target":6.5,'
        '   "stop_loss":2.4,"time_horizon":"1-2wk","confidence":"medium",'
        '   "rationale":"Breakout above 20-bar range on volume"}'
        '],"notes":"risk-off tilt with tactical QQQ bounce"}'
    )

    class _Client:
        class cfg:
            timeout_sec = 1.0
        def ping(self): return True
        def generate(self, **kw): return raw
    monkeypatch.setattr(
        "src.intelligence.ollama_client.build_ollama_client",
        lambda: _Client(),
    )
    agent = OptionsResearchAgent(_FakeMP(spot=708, chain=_mk_chain(708)))
    rep = agent.run(["SPY", "QQQ"])
    assert len(rep.ideas) == 2
    assert rep.ideas[0].symbol == "SPY"
    assert rep.ideas[0].strike == 705
    assert rep.ideas[0].direction == "put"
    assert rep.ideas[1].direction == "call"
    assert "risk-off" in rep.notes


def test_agent_to_markdown_empty_ideas():
    import src.intelligence.options_research as mod
    agent = OptionsResearchAgent(_FakeMP(spot=100, chain=_mk_chain()))
    from src.intelligence.options_research import ResearchReport
    rep = ResearchReport(
        ts="2026-04-21T00:00:00Z",
        underlyings=["SPY"],
        spot_by_symbol={"SPY": 708.80},
        quote_sources={"SPY": ["polygon"]},
        n_headlines=5, sentiment_by_symbol={"SPY": 0.12},
        ideas=[], model="", latency_sec=2.0, notes="",
    )
    md = agent.to_markdown(rep)
    assert "Options Research" in md
    assert "SPY=708.80" in md
    assert "no actionable ideas" in md


def test_agent_to_markdown_with_ideas():
    from src.intelligence.options_research import ResearchReport
    rep = ResearchReport(
        ts="2026-04-21T00:00:00Z", underlyings=["SPY"],
        spot_by_symbol={"SPY": 708.80}, quote_sources={"SPY": ["polygon"]},
        n_headlines=5, sentiment_by_symbol={"SPY": -0.3},
        ideas=[
            TradeIdea(symbol="SPY", direction="put", strike=705,
                      expiry="2026-05-02", entry=3.0, profit_target=4.8,
                      stop_loss=1.8, time_horizon="1-3d",
                      confidence="high",
                      rationale="Bearish divergence + breadth deteriorating"),
        ],
        model="llama3.1:70b", latency_sec=45.2, notes="risk-off",
    )
    agent = OptionsResearchAgent(_FakeMP())
    md = agent.to_markdown(rep)
    assert "SPY PUT" in md
    assert "$705" in md
    assert "2026-05-02" in md
    assert "high" in md
    assert "risk-off" in md
    # Must fit comfortably in a Discord message
    assert len(md) < 1800


def test_question_wants_options_context_detection():
    from src.intelligence.llm_chat import question_wants_options_context
    assert question_wants_options_context("what's a good put on SPY?")
    assert question_wants_options_context("any strike recommendations?")
    assert question_wants_options_context("how's sentiment on QQQ")
    assert not question_wants_options_context("hi how are you")
    assert not question_wants_options_context("when does market close")
