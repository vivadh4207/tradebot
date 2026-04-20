"""Centralized error → Discord alerts helper.

Every failure surface in the bot funnels through `report_issue()` so
nothing dies silently. The function:

  1. Pushes a one-line title + short traceback to the alerts channel
     (`level="error"` + `title="error"` → MultiChannelNotifier routes
     to DISCORD_WEBHOOK_URL_ALERTS).
  2. Throttles duplicate errors. If the same (scope, fingerprint) hits
     repeatedly, only the first in a `throttle_sec` window is sent.
     The log record still fires every time so journalctl has the full
     history. Throttling prevents a tight-loop crash from spamming
     Discord into rate-limits, which would hide the actual problem.
  3. Never raises. Alerting about an error must never cascade into
     another error.

Use `report_issue()` directly from `except` blocks. Use
`@alert_on_crash("script_name")` to wrap the main() of any script —
an unhandled exception becomes a Discord alert before the process
exits.
"""
from __future__ import annotations

import functools
import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional

from .base import Notifier, NullNotifier, build_notifier


# ---------- shared notifier + throttle map ------------------------------

_lock = threading.Lock()
_notifier: Optional[Notifier] = None
_last_sent: Dict[str, float] = {}
# Fingerprint → (seconds-to-suppress-duplicates, count-since-last-send).
# A small TTL cache; memory footprint is bounded because the set of
# unique error fingerprints in a running bot is small (<50).


def _get_notifier() -> Notifier:
    """Lazy build. Every call is cheap after the first; build_notifier
    reads env once. If env is unset, returns NullNotifier (silent)."""
    global _notifier
    if _notifier is None:
        try:
            _notifier = build_notifier()
        except Exception:
            # Catastrophic: can't even build the notifier. Don't crash —
            # just degrade to null and carry on.
            _notifier = NullNotifier()
    return _notifier


def _fingerprint(scope: str, message: str) -> str:
    """Stable key for throttling. Drops memory addresses, line numbers,
    and trailing dynamic bits (which change per-instance but mean the
    same error). We keep it deliberately coarse so a tight loop sends
    ONE alert rather than N."""
    # Normalize: take the first 200 chars of (scope + message) and strip
    # digits (PIDs, pointers, timestamps shift a lot across repeats).
    raw = f"{scope}::{message[:200]}"
    return "".join(c for c in raw if not c.isdigit())


def report_issue(
    scope: str,
    message: str,
    exc: Optional[BaseException] = None,
    *,
    throttle_sec: float = 300.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Push an error-level message to the Discord alerts channel.

    Args:
      scope:        short identifier of where the error came from, e.g.
                    "fast_loop", "alpaca.submit", "nightly_walkforward".
                    Shown as the Discord embed title.
      message:      one-line description. Avoid multi-line content —
                    use `exc` for tracebacks.
      exc:          optional exception; its traceback is attached as a
                    field. If None and we're inside an `except` block,
                    the caller can pass sys.exc_info()[1].
      throttle_sec: duplicate suppression window. 0 disables throttling.
      extra:        optional dict appended to the embed fields.

    Never raises. Safe to call from any thread.
    """
    try:
        fp = _fingerprint(scope, message)
        now = time.time()
        with _lock:
            last = _last_sent.get(fp, 0.0)
            if throttle_sec > 0 and (now - last) < throttle_sec:
                return
            _last_sent[fp] = now
            # Prune old entries so the dict doesn't grow unboundedly.
            if len(_last_sent) > 256:
                cutoff = now - 24 * 3600
                _last_sent.clear() if len(_last_sent) > 1024 else None
                for k in list(_last_sent.keys()):
                    if _last_sent[k] < cutoff:
                        del _last_sent[k]

        n = _get_notifier()
        meta: Dict[str, Any] = {"Scope": scope}
        if extra:
            for k, v in extra.items():
                meta[str(k)] = str(v)[:300]
        if exc is not None:
            # Last 20 lines of traceback — enough to diagnose, short
            # enough to fit a Discord embed field.
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            tb_tail = "\n".join(tb.strip().splitlines()[-20:])
            meta["Traceback"] = f"```\n{tb_tail[-900:]}\n```"
        meta["Host"] = os.uname().nodename if hasattr(os, "uname") else "?"

        n.notify(
            text=message[:500],
            level="error",
            title="error",
            meta=meta,
        )
    except Exception:
        # Alerting failed. Swallow — we tried. An alert-about-alerts
        # loop would be worse than silent.
        pass


def alert_on_crash(scope: str, rethrow: bool = True) -> Callable:
    """Decorator for script main() functions.

    Any unhandled exception inside the wrapped function posts to the
    alerts channel with the full traceback, then (by default) re-raises
    so the process still exits with a nonzero code. Use rethrow=False
    for cron-scheduled scripts where you want cron to stay happy while
    still seeing the alert in Discord.

    Example:
        @alert_on_crash("nightly_walkforward")
        def main() -> int: ...
    """
    def _decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except SystemExit:
                raise
            except BaseException as e:
                report_issue(
                    scope=scope,
                    message=f"{scope}: unhandled {type(e).__name__}: {e}",
                    exc=e,
                    throttle_sec=0.0,  # crash alerts never throttled
                )
                if rethrow:
                    raise
                return 1
        return _wrapper
    return _decorator


def install_excepthooks(scope_prefix: str = "unhandled") -> None:
    """Register sys.excepthook + threading.excepthook so ANY stray
    exception — including in non-main threads — lands in Discord alerts
    before the process dies. Idempotent: calling twice is safe (second
    call is a no-op; no chained-double-alerting).

    Call once at process start (run_paper.py, watchdog_run.py).
    """
    import sys

    prev_sys = sys.excepthook
    prev_thr = getattr(threading, "excepthook", None)

    # Idempotency guard: if the currently-installed hook was put there
    # by a previous call to this function, do nothing. Prevents chains
    # like wrapper → wrapper → wrapper firing N alerts for one crash
    # when tests (or anything) call install_excepthooks multiple times.
    if getattr(prev_sys, "_installed_by_tradebot", False):
        return

    def _sys_hook(exc_type, exc, tb):
        if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            report_issue(
                scope=f"{scope_prefix}.main",
                message=f"unhandled {exc_type.__name__}: {exc}",
                exc=exc,
                throttle_sec=0.0,
            )
        # Chain to whatever was there (default handler prints to stderr).
        if prev_sys is not None:
            prev_sys(exc_type, exc, tb)

    def _thread_hook(args):
        # threading.ExceptHookArgs has exc_type, exc_value, exc_traceback, thread
        if args.exc_type is SystemExit:
            return
        report_issue(
            scope=f"{scope_prefix}.thread.{getattr(args.thread, 'name', '?')}",
            message=f"thread crash: {args.exc_type.__name__}: {args.exc_value}",
            exc=args.exc_value,
            throttle_sec=0.0,
        )
        if prev_thr is not None:
            prev_thr(args)

    _sys_hook._installed_by_tradebot = True             # type: ignore[attr-defined]
    _thread_hook._installed_by_tradebot = True          # type: ignore[attr-defined]
    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook
