from src.math_tools.sizing import kelly_fraction, vix_regime_multiplier, hybrid_sizing


def test_kelly_zero_when_no_edge():
    # 50% WR, equal win/loss → kelly = 0
    assert kelly_fraction(0.5, 1.0, 1.0) == 0


def test_kelly_positive_with_edge():
    # 60% WR, R:R 1:1 → edge
    k = kelly_fraction(0.60, 1.0, 1.0, fraction=1.0, hard_cap=0.5)
    assert 0.0 < k < 0.5


def test_kelly_capped():
    k = kelly_fraction(0.90, 1.0, 1.0, fraction=1.0, hard_cap=0.05)
    assert k == 0.05


def test_vix_regime_linear():
    assert vix_regime_multiplier(10, 10, 30) == 0.5
    assert vix_regime_multiplier(30, 10, 30) == 1.5
    assert abs(vix_regime_multiplier(20, 10, 30) - 1.0) < 1e-9


def test_hybrid_sizing_non_negative():
    n = hybrid_sizing(
        equity=50_000, max_loss_per_contract=500,
        win_rate_est=0.60, avg_win=1.0, avg_loss=1.0,
        vix_today=18, vix_52w_low=12, vix_52w_high=40,
        vrp_zscore=0.5, max_contracts=20,
    )
    assert n >= 0
    assert n <= 20


def test_hybrid_sizing_zero_when_no_equity():
    n = hybrid_sizing(
        equity=0, max_loss_per_contract=500,
        win_rate_est=0.80, avg_win=1, avg_loss=1,
        vix_today=20, vix_52w_low=10, vix_52w_high=40,
    )
    assert n == 0
