"""Smoke test for the A/B LSTM harness — confirms the SimConfig wiring
actually removes the LSTM strategy from the enabled list.
"""
from src.core.config import load_settings
from src.data.market_data import SyntheticDataAdapter
from src.backtest.simulator import BacktestSimulator, SimConfig


def test_disable_signals_filters_lstm(tmp_path):
    s = load_settings()

    data = SyntheticDataAdapter(seed=42)
    on = BacktestSimulator(s.raw, data, SimConfig(disable_signals=()))
    off = BacktestSimulator(s.raw, data, SimConfig(disable_signals=("lstm",)))
    assert not any(st.name == "lstm" for st in off.strategies)
    # The other strategies remain registered either way
    for st in ("momentum", "vwap_reversion", "orb"):
        assert any(s2.name == st for s2 in off.strategies)
        assert any(s2.name == st for s2 in on.strategies)
    # The "on" sim should have >= strategies as the "off" sim
    assert len(on.strategies) >= len(off.strategies)
