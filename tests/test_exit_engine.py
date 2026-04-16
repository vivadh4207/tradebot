from datetime import datetime, date, time, timedelta
from src.core.types import Position, OptionRight
from src.core.clock import ET
from src.exits.exit_engine import ExitEngine, ExitEngineConfig
from src.exits.auto_stops import compute_auto_stops


def _long_option_pos(dte_days: int = 0, entry: float = 1.00) -> Position:
    exp = date.today() + timedelta(days=dte_days)
    p = Position(symbol="SPY240416C00500000", qty=1, avg_price=entry,
                 is_option=True, underlying="SPY", strike=500.0,
                 expiry=exp, right=OptionRight.CALL, multiplier=100,
                 entry_ts=datetime.now().timestamp())
    pt, sl = compute_auto_stops(p, is_short_dte=True)
    p.auto_profit_target = pt
    p.auto_stop_loss = sl
    return p


def test_layer5_global_stop_triggers_on_bad_pnl():
    ee = ExitEngine()
    pos = _long_option_pos(dte_days=0, entry=1.00)
    now = ET.localize(datetime.combine(date.today(), time(11, 0)))
    # 25% loss on long → price 0.75, > 20% stop
    d = ee.decide(pos, 0.75, now, vix=15, spot=498, vwap=499, bars=[])
    assert d.should_close


def test_layer4_profit_target_triggers():
    ee = ExitEngine()
    pos = _long_option_pos(dte_days=0, entry=1.00)
    now = ET.localize(datetime.combine(date.today(), time(11, 0)))
    # 40% profit on long → price 1.40, exceeds 35% short_dte target
    d = ee.decide(pos, 1.40, now, vix=15, spot=500, vwap=500, bars=[])
    assert d.should_close


def test_layer1_0dte_force_close():
    ee = ExitEngine()
    pos = _long_option_pos(dte_days=0, entry=1.00)
    now = ET.localize(datetime.combine(date.today(), time(15, 50)))
    d = ee.decide(pos, 1.05, now, vix=15, spot=500, vwap=500, bars=[])
    assert d.should_close
    assert d.layer in (1, 2, 4, 6)


def test_no_exit_mid_day_small_pnl():
    ee = ExitEngine()
    pos = _long_option_pos(dte_days=3, entry=1.00)
    now = ET.localize(datetime.combine(date.today(), time(11, 0)))
    d = ee.decide(pos, 1.05, now, vix=15, spot=500, vwap=500, bars=[])
    assert not d.should_close
