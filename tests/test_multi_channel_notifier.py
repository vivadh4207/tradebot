"""Ensure multi-channel routing sends each title to the right webhook.

Locks in the operator-requested behavior: entry/exit → TRADES channel,
catalysts → CATALYSTS channel, HALT/watchdog → ALERTS channel,
everything else → default.
"""
from __future__ import annotations

from src.notify.base import MultiChannelNotifier, Notifier


class _CapturingNotifier(Notifier):
    """Records every notify() call instead of sending HTTP."""
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def notify(self, text, *, level="info", title="", meta=None):
        self.calls.append((title, text, level, meta))


def _build(includes=("default", "trades", "catalysts", "alerts",
                     "calibration", "reason")):
    """Return a MultiChannelNotifier with only the named channels present."""
    channels = {name: _CapturingNotifier(name) for name in includes}
    return MultiChannelNotifier(channels), channels


# ---------- routing ----------
def test_entry_goes_to_trades_channel():
    mc, chans = _build()
    mc.notify("BUY 3 × CALL NVDA", title="entry")
    assert len(chans["trades"].calls) == 1
    assert chans["trades"].calls[0][0] == "entry"
    # default stayed empty
    assert not chans["default"].calls


def test_exit_goes_to_trades_channel():
    mc, chans = _build()
    mc.notify("CLOSE 3 × CALL NVDA → +30%", title="exit")
    assert len(chans["trades"].calls) == 1


def test_catalysts_goes_to_catalysts_channel():
    mc, chans = _build()
    mc.notify("12 catalysts", title="catalysts")
    assert len(chans["catalysts"].calls) == 1
    assert not chans["trades"].calls


def test_halt_goes_to_alerts_channel():
    mc, chans = _build()
    mc.notify("Daily loss breach", title="HALT", level="error")
    assert len(chans["alerts"].calls) == 1


def test_watchdog_goes_to_alerts_channel():
    mc, chans = _build()
    mc.notify("tradebot CRASHED rc=1", title="watchdog", level="error")
    assert len(chans["alerts"].calls) == 1


def test_calibration_goes_to_calibration_channel():
    mc, chans = _build()
    mc.notify("ratio=1.67 adjusted 2 constants", title="calibration")
    assert len(chans["calibration"].calls) == 1


def test_reconcile_goes_to_alerts_channel():
    mc, chans = _build()
    mc.notify("Closed 3 zombies on Alpaca", title="reconcile")
    assert len(chans["alerts"].calls) == 1


def test_backtest_report_goes_to_reason_channel():
    """Nightly walkforward posts title=backtest_report — must land in
    the `reason-to-trade` channel, not alerts or default."""
    mc, chans = _build()
    mc.notify("EDGE CONFIRMED — 75% tradable", title="backtest_report")
    assert len(chans["reason"].calls) == 1
    assert not chans["alerts"].calls
    assert not chans["default"].calls


def test_edge_report_goes_to_reason_channel():
    mc, chans = _build()
    mc.notify("IC for momentum = 0.04", title="edge_report")
    assert len(chans["reason"].calls) == 1


def test_regime_shift_goes_to_reason_channel():
    mc, chans = _build()
    mc.notify("regime: range_lowvol → trend_highvol", title="regime_shift")
    assert len(chans["reason"].calls) == 1


def test_backtest_report_falls_back_to_default_when_reason_unset():
    """If DISCORD_WEBHOOK_URL_REASON isn't configured, report must still
    land in the default channel rather than vanishing."""
    mc, chans = _build(includes=("default",))
    mc.notify("EDGE CONFIRMED", title="backtest_report")
    assert len(chans["default"].calls) == 1


def test_extra_operator_defined_channel_routes():
    """Operator adds a new channel like 'political' plus a custom route
    via DISCORD_EXTRA_CHANNEL_ROUTES. MultiChannelNotifier should honor it."""
    from src.notify.base import MultiChannelNotifier
    chans = {
        "default":   _CapturingNotifier("default"),
        "political": _CapturingNotifier("political"),
    }
    mc = MultiChannelNotifier(
        chans,
        extra_routes={"political": {"political_alert", "geopolitical"}},
    )
    mc.notify("Iran drone strike escalation", title="political_alert")
    assert len(chans["political"].calls) == 1
    assert not chans["default"].calls

    # A title NOT in the extra route or builtins falls to default
    mc.notify("random status blip", title="random")
    assert len(chans["default"].calls) == 1


def test_extra_routes_do_not_override_builtin_routes():
    """If an operator accidentally tries to redirect 'entry' to
    their new channel, the builtin 'trades' route still wins."""
    from src.notify.base import MultiChannelNotifier
    chans = {
        "default":   _CapturingNotifier("default"),
        "trades":    _CapturingNotifier("trades"),
        "political": _CapturingNotifier("political"),
    }
    mc = MultiChannelNotifier(
        chans,
        extra_routes={"political": {"entry"}},   # attempted hijack
    )
    mc.notify("BUY 3 × CALL SPY", title="entry")
    assert len(chans["trades"].calls) == 1
    assert not chans["political"].calls


def test_unknown_title_falls_back_to_default():
    mc, chans = _build()
    mc.notify("some random event", title="something_weird")
    assert len(chans["default"].calls) == 1
    assert not chans["trades"].calls


def test_missing_channel_falls_back_to_default():
    """If we're configured without the trades channel, entry/exit routes
    to default instead of disappearing."""
    mc, chans = _build(includes=("default",))  # only default configured
    mc.notify("BUY CALL NVDA", title="entry")
    # entry would normally go to trades — but trades isn't configured,
    # so falls through to default
    assert len(chans["default"].calls) == 1


def test_close_method_closes_all_underlying():
    """Ensure close() propagates to every subchannel."""
    closed = []

    class _Closeable(Notifier):
        def __init__(self, name):
            self.name = name
        def notify(self, text, *, level="info", title="", meta=None):
            pass
        def close(self):
            closed.append(self.name)

    mc = MultiChannelNotifier({
        "default": _Closeable("default"),
        "trades":  _Closeable("trades"),
    })
    mc.close()
    assert set(closed) == {"default", "trades"}
