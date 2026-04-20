"""BreadthProbe unit tests."""
from __future__ import annotations

import time
import pytest

from src.intelligence.breadth_probe import (
    BreadthProbe, BreadthProbeConfig, BreadthSnapshot,
)


class _FakeVix:
    def __init__(self, v): self.v = v
    def value(self): return self.v


def _probe(*, spot, open_px, vix=None, cfg=None):
    """Build a probe with fixed spot/open/vix for testing."""
    return BreadthProbe(
        cfg=cfg,
        spot_fetcher=lambda s: spot.get(s),
        open_fetcher=lambda s: open_px.get(s),
        vix_probe=_FakeVix(vix) if vix is not None else None,
    )


# ---------- sub-score math ----------


def test_vix_subscore_low_is_risk_on():
    """VIX at 10 or below → sub-score = +1.0 (maximally risk-on)."""
    p = _probe(spot={}, open_px={}, vix=10)
    s = p.snapshot()
    assert s.components["vix"] == 1.0


def test_vix_subscore_high_is_risk_off():
    """VIX at 40 or above → sub-score = -1.0 (maximally risk-off)."""
    p = _probe(spot={}, open_px={}, vix=45)
    s = p.snapshot()
    assert s.components["vix"] == -1.0


def test_vix_subscore_mid_is_linear():
    """VIX at 25 (midpoint of [10, 40]) → sub-score = 0."""
    p = _probe(spot={}, open_px={}, vix=25)
    s = p.snapshot()
    assert abs(s.components["vix"]) < 0.01


def test_spy_intraday_drop_15pct_saturates_at_minus_one():
    p = _probe(
        spot={"SPY": 570.0},
        open_px={"SPY": 580.0},   # -1.72% drop
    )
    s = p.snapshot()
    assert s.components["spy_intraday"] == -1.0


def test_spy_intraday_rally_1pct_saturates_at_plus_one():
    p = _probe(
        spot={"SPY": 585.8},
        open_px={"SPY": 580.0},   # +1.0% rally
    )
    s = p.snapshot()
    assert s.components["spy_intraday"] == 1.0


def test_spy_intraday_flat_is_neutral():
    p = _probe(
        spot={"SPY": 580.0},
        open_px={"SPY": 580.0},
    )
    s = p.snapshot()
    assert s.components["spy_intraday"] == 0.0


# ---------- pair ratios ----------


def test_iwm_spy_positive_drift_is_risk_on():
    """IWM outperforming SPY = small-caps leading = risk-on."""
    p = _probe(
        spot={"IWM": 205.0, "SPY": 580.0},    # ratio 0.3534
        open_px={"IWM": 204.0, "SPY": 580.0},  # ratio 0.3517
    )
    s = p.snapshot()
    # Ratio drifted up by ~0.5% — should produce a positive iwm_spy subscore
    assert s.components["iwm_spy"] > 0


def test_hyg_tlt_positive_drift_is_risk_on():
    """High-yield outperforming treasuries = credit appetite up = risk-on."""
    p = _probe(
        spot={"HYG": 77.5, "TLT": 92.0},
        open_px={"HYG": 77.0, "TLT": 92.5},   # HYG up, TLT down → big pos drift
    )
    s = p.snapshot()
    assert s.components["hyg_tlt"] > 0


# ---------- composition + fail-safes ----------


def test_composite_averages_sub_scores_weighted():
    """If VIX says -1 (weight 1) and SPY says +1 (weight 1) and others
    are absent, composite = (−1 + 1) / 2 = 0."""
    p = _probe(
        spot={"SPY": 585.8},           # +1% rally → SPY = +1
        open_px={"SPY": 580.0},
        vix=45,                         # VIX 45 → vix = -1
    )
    s = p.snapshot()
    # Other pair-ratio subscores will be 0 (no IWM/HYG data) and are
    # still included, so composite = (vix-1 + spy+1 + 0 + 0) / (1+1+0.5+0.5) = 0 / 3 = 0
    # Wait — iwm_spy and hyg_tlt return None when spot missing, so they
    # ARE excluded. Composite = (vix-1 + spy+1) / (1+1) = 0
    assert abs(s.score) < 0.01


def test_missing_data_returns_neutral_not_crash():
    """All fetchers return None → score 0, not an exception."""
    p = BreadthProbe(
        spot_fetcher=lambda s: None,
        open_fetcher=lambda s: None,
        vix_probe=None,
    )
    s = p.snapshot()
    assert s.score == 0.0
    assert s.components == {}


def test_data_fetcher_exception_is_caught():
    """If a fetcher raises, the probe should not propagate — just skip
    that sub-score."""
    def broken(sym): raise RuntimeError("feed outage")
    p = BreadthProbe(
        spot_fetcher=broken,
        open_fetcher=broken,
        vix_probe=_FakeVix(15),
    )
    s = p.snapshot()
    # VIX subscore should still work
    assert "vix" in s.components
    # The broken fetchers' subscores are absent (not 0)
    assert "spy_intraday" not in s.components


# ---------- caching ----------


def test_snapshot_is_cached_within_ttl():
    calls = [0]
    def spot(sym):
        calls[0] += 1
        return 100.0
    p = BreadthProbe(
        cfg=BreadthProbeConfig(cache_seconds=60.0),
        spot_fetcher=spot,
        open_fetcher=lambda s: 100.0,
        vix_probe=_FakeVix(15),
    )
    _ = p.snapshot()
    calls_after_first = calls[0]
    _ = p.snapshot()
    _ = p.snapshot()
    assert calls[0] == calls_after_first  # no additional fetches


def test_force_refresh_bypasses_cache():
    calls = [0]
    def spot(sym):
        calls[0] += 1
        return 100.0
    p = BreadthProbe(
        cfg=BreadthProbeConfig(cache_seconds=60.0),
        spot_fetcher=spot,
        open_fetcher=lambda s: 100.0,
    )
    _ = p.snapshot()
    n1 = calls[0]
    _ = p.snapshot(force_refresh=True)
    assert calls[0] > n1


# ---------- risk-off + size multiplier ----------


def test_risk_off_flag_fires_below_threshold():
    cfg = BreadthProbeConfig(risk_off_threshold=-0.3)
    p = _probe(
        spot={"SPY": 570.0},
        open_px={"SPY": 580.0},    # -1.7% → SPY sub = -1.0
        vix=35,                     # vix sub = -0.67
        cfg=cfg,
    )
    snap = p.snapshot()
    assert snap.is_risk_off is True
    assert snap.score < -0.3


def test_size_multiplier_tiering():
    """Verify the piecewise tiering from snapshot.size_multiplier()."""
    def m(score):
        return BreadthSnapshot(score=score, is_risk_off=score < -0.3).size_multiplier()
    assert m(+0.5) == 1.0
    assert m(+0.1) == 0.9
    assert m(-0.1) == 0.75
    assert m(-0.5) == 0.50
    assert m(-0.8) == 0.25
