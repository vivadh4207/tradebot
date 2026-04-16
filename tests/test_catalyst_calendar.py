from datetime import date, datetime, timedelta

import pytest

from src.intelligence.econ_calendar import EconomicCalendar
from src.intelligence.catalyst_calendar import (
    CatalystEvent, CatalystCalendar, StaticCatalystProvider,
)


def test_econ_calendar_symbol_blackout():
    cal = EconomicCalendar()
    today = date.today()
    cal.add_symbol_blackout("AAPL", today)
    now = datetime.combine(today, datetime.min.time()).replace(hour=10)
    assert cal.in_blackout(now, symbol="AAPL") is True
    assert cal.in_blackout(now, symbol="NVDA") is False
    assert cal.in_blackout(now) is False   # no symbol → ignores per-symbol


def test_static_provider_yaml(tmp_path):
    p = tmp_path / "catalysts.yaml"
    today = date.today()
    near = (today + timedelta(days=3)).isoformat()
    far = (today + timedelta(days=60)).isoformat()
    p.write_text(
        f"""
fda:
  - symbol: MRNA
    date: {near}
    event: PDUFA decision
earnings:
  - symbol: AAPL
    date: {near}
    timing: amc
    event: Q2 earnings
  - symbol: MSFT
    date: {far}
    timing: bmo
    event: out_of_window
"""
    )
    prov = StaticCatalystProvider(str(p))
    events = prov.fetch(["AAPL", "MRNA", "MSFT", "NVDA"], days=14)
    # MSFT is beyond 14-day horizon; NVDA not in YAML → neither present
    assert {e.symbol for e in events} == {"MRNA", "AAPL"}
    aapl = next(e for e in events if e.symbol == "AAPL")
    assert aapl.event_type == "earnings" and aapl.timing == "amc"


def test_catalyst_calendar_hydrates_econ_calendar(tmp_path):
    today = date.today()
    near = (today + timedelta(days=2)).isoformat()
    p = tmp_path / "catalysts.yaml"
    p.write_text(
        f"""
earnings:
  - symbol: AAPL
    date: {near}
    timing: amc
"""
    )
    cc = CatalystCalendar(providers=[StaticCatalystProvider(str(p))])
    cc.refresh(["AAPL"])
    assert len(cc.events) == 1

    econ = EconomicCalendar()
    n = cc.hydrate_econ_calendar(econ)
    assert n == 1
    earnings_date = cc.events[0].when
    now = datetime.combine(earnings_date, datetime.min.time()).replace(hour=10)
    assert econ.in_blackout(now, symbol="AAPL") is True
    # symbol NOT in the calendar should NOT be blacked out
    assert econ.in_blackout(now, symbol="NVDA") is False


def test_hydrate_clears_prior_blackouts(tmp_path):
    # Day-1 run adds AAPL; Day-2 data no longer contains AAPL → blackout gone.
    p = tmp_path / "catalysts.yaml"
    p.write_text(
        f"""
earnings:
  - symbol: AAPL
    date: {(date.today() + timedelta(days=1)).isoformat()}
"""
    )
    cc = CatalystCalendar(providers=[StaticCatalystProvider(str(p))])
    econ = EconomicCalendar()
    cc.refresh(["AAPL"])
    cc.hydrate_econ_calendar(econ)
    assert "AAPL" in econ.summary()

    # Rewrite YAML with empty sections and refresh
    p.write_text("earnings: []\nfda: []\n")
    cc.refresh(["AAPL"])
    cc.hydrate_econ_calendar(econ)
    assert "AAPL" not in econ.summary()
