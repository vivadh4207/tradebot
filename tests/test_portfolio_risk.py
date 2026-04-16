from datetime import date, timedelta
from src.core.types import Position, OptionRight
from src.risk.portfolio_risk import PortfolioRiskManager


def _short_put(strike: float, entry: float = 1.00, qty: int = -1) -> Position:
    return Position(symbol=f"X_{strike}_P", qty=qty, avg_price=entry,
                     is_option=True, underlying="X", strike=strike,
                     expiry=date.today() + timedelta(days=5),
                     right=OptionRight.PUT, multiplier=100)


def test_stress_returns_nonpositive():
    prm = PortfolioRiskManager()
    positions = [_short_put(95.0), _short_put(90.0)]
    worst = prm.stress(positions, spot=100.0)
    assert worst <= 0.0


def test_greeks_aggregate_runs():
    prm = PortfolioRiskManager()
    g = prm.aggregate_greeks([_short_put(95.0)], spot=100.0)
    assert "dollar_delta" in g and "dollar_gamma" in g


def test_check_ok_small_position():
    prm = PortfolioRiskManager()
    proposed = _short_put(95.0)
    # Defaults in PortfolioRiskManager are per-$100k and strict. Test with
    # 50× that baseline so a single contract is comfortably within limits.
    ok, reason = prm.check(proposed, [], spot=100.0, equity=5_000_000)
    assert ok, reason
