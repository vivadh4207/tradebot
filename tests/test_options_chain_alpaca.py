from datetime import date
from src.data.options_chain_alpaca import _parse_occ, AlpacaOptionsChain
from src.data.options_chain import SyntheticOptionsChain
from src.core.types import OptionRight


def test_parse_occ_call():
    r = _parse_occ("SPY240419C00500000", "SPY")
    assert r is not None
    d, right, strike = r
    assert d == date(2024, 4, 19)
    assert right == OptionRight.CALL
    assert strike == 500.0


def test_parse_occ_put_fractional_strike():
    r = _parse_occ("SPY240419P00497500", "SPY")
    assert r is not None
    _, right, strike = r
    assert right == OptionRight.PUT
    assert strike == 497.5


def test_parse_occ_bad_prefix_returns_none():
    assert _parse_occ("QQQ240419C00500000", "SPY") is None


def test_alpaca_chain_falls_back_when_no_client():
    # invalid creds → no client → fallback used
    provider = AlpacaOptionsChain(api_key="bad", api_secret="bad",
                                  fallback=SyntheticOptionsChain())
    provider._client = None     # simulate install/auth failure
    chain = provider.chain("SPY", 500.0, target_dte=1)
    assert len(chain) > 0
    assert all(c.underlying == "SPY" for c in chain)
