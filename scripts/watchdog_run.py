"""Watchdog launcher — supervises run_paper.py.

Host-agnostic. Runs under macOS launchd
(deploy/launchd/com.tradebot.paper.plist) AND under Linux/Jetson systemd
--user (deploy/systemd/tradebot-watchdog.service). Both supervisors only
need the watchdog itself; the internal logic is identical on either OS.

Responsibilities (the bits launchd / systemd can't do on their own):

  1. Fire a notifier message when the child exits abnormally so the
     operator actually *hears* about a crash.
  2. Write a crash record to `logs/watchdog_events.jsonl` (exit code,
     signal, duration, last N lines of stderr). The dashboard reads this
     to show "last crash".
  3. Detect stale heartbeat — process alive but main loop frozen (a
     deadlock, a stuck network call, a runaway GC). If
     `logs/heartbeat.txt` hasn't been touched in HEARTBEAT_STALE_SEC,
     SIGTERM the child; the outer supervisor's Restart=on-failure /
     KeepAlive=true brings it back.
  4. Exit with the child's return code. We only exit 0 if asked to stop
     cleanly (SIGTERM from `launchctl stop` / `systemctl --user stop`
     / the user).

Why a Python wrapper instead of a shell script:
  - We already have `src.notify.build_notifier()` — reusing it means
    alerts go to the same Discord/Slack channel as calibration alerts.
  - Heartbeat-timeout math is fiddly; easier to get right in Python
     than bash.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
HEARTBEAT_FILE = LOG_DIR / "heartbeat.txt"
EVENTS_FILE = LOG_DIR / "watchdog_events.jsonl"
STDERR_CAPTURE = LOG_DIR / "tradebot.err"

# Tuning. Override via env if ever needed — defaults are conservative
# (stale heartbeat = 5 minutes, which is ~1.6x the default 180s main loop
# tick, so we don't false-positive on a normal slow cycle).
HEARTBEAT_STALE_SEC = float(os.getenv("WATCHDOG_HEARTBEAT_STALE_SEC", "300"))
CHECK_INTERVAL_SEC = float(os.getenv("WATCHDOG_CHECK_INTERVAL_SEC", "30"))
# Launch grace: don't trip stale-heartbeat during startup (imports,
# CockroachDB handshake, LSTM checkpoint load).
STARTUP_GRACE_SEC = float(os.getenv("WATCHDOG_STARTUP_GRACE_SEC", "120"))
CHILD_KILL_TIMEOUT_SEC = 15.0


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _record_event(event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event = dict(event, ts=_iso_now())
    try:
        with EVENTS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except Exception:
        pass  # best-effort


def _tail_stderr(n: int = 20) -> str:
    """Grab the last `n` lines of the captured stderr file if present."""
    if not STDERR_CAPTURE.exists():
        return ""
    try:
        with STDERR_CAPTURE.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 16 * 1024)
            f.seek(size - block)
            chunk = f.read().decode("utf-8", errors="replace")
        return "\n".join(chunk.splitlines()[-n:])
    except Exception:
        return ""


def _heartbeat_age_sec() -> Optional[float]:
    """Seconds since heartbeat was last written; None if missing."""
    try:
        st = HEARTBEAT_FILE.stat()
    except FileNotFoundError:
        return None
    return max(0.0, time.time() - st.st_mtime)


def _notify(text: str, *, level: str = "warn", title: str = "watchdog") -> None:
    """Best-effort notification. Imports lazy — a broken Notifier should
    never prevent the watchdog from doing its job."""
    try:
        from src.notify import build_notifier
        n = build_notifier()
        n.notify(text, level=level, title=title)
        # Flush webhook queue before exiting the wrapper
        try:
            n.close()                                       # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _reset_heartbeat() -> None:
    """Clear any pre-existing heartbeat so the next child has to write its
    own. Otherwise we could mistakenly read a fresh heartbeat from the
    previous incarnation and delay stale detection by one tick."""
    try:
        HEARTBEAT_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _spawn_child() -> subprocess.Popen:
    py = os.environ.get("TRADEBOT_PY", sys.executable)
    script = str(ROOT / "scripts" / "run_paper.py")
    # Inherit stdout/stderr — the plist captures both to the log files.
    return subprocess.Popen(
        [py, script],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )


def _terminate(child: subprocess.Popen) -> None:
    """Send SIGTERM, give the bot up to CHILD_KILL_TIMEOUT_SEC to flush the
    journal and flatten, then SIGKILL."""
    if child.poll() is not None:
        return
    try:
        child.terminate()
    except Exception:
        pass
    deadline = time.time() + CHILD_KILL_TIMEOUT_SEC
    while time.time() < deadline:
        if child.poll() is not None:
            return
        time.sleep(0.25)
    try:
        child.kill()
    except Exception:
        pass


def main() -> int:
    # Global error hooks — any stray exception inside the watchdog
    # (file I/O, signal handling, child spawn) posts to Discord alerts
    # before the process dies.
    try:
        from src.notify.issue_reporter import install_excepthooks
        install_excepthooks(scope_prefix="watchdog")
    except Exception:
        pass  # notifier failed to import; continue — watchdog itself still logs events.

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _reset_heartbeat()

    started = time.time()
    child = _spawn_child()
    _record_event({
        "kind": "start",
        "pid": child.pid,
        "python": os.environ.get("TRADEBOT_PY", sys.executable),
    })

    # Forward SIGTERM/SIGINT to the child so `launchctl stop com.tradebot.paper`
    # (which sends SIGTERM to us) triggers the child's clean shutdown path.
    shutdown_requested = {"flag": False}

    def _forward(signum, _frame):
        shutdown_requested["flag"] = True
        try:
            child.send_signal(signum)
        except Exception:
            pass

    try:
        signal.signal(signal.SIGTERM, _forward)
        signal.signal(signal.SIGINT, _forward)
    except ValueError:
        pass

    # --- main watchdog loop ---
    while True:
        rc = child.poll()
        if rc is not None:
            break
        age = _heartbeat_age_sec()
        alive_for = time.time() - started
        # After startup grace has elapsed, a missing heartbeat (age=None)
        # counts as "infinitely stale" — it means the main loop never
        # started. Outside grace, treat age > stale-threshold as stale.
        is_stale = (alive_for > STARTUP_GRACE_SEC
                     and (age is None or age > HEARTBEAT_STALE_SEC))
        if is_stale:
            _record_event({
                "kind": "heartbeat_stale",
                "age_sec": round(age, 1) if age is not None else None,
                "alive_for_sec": round(alive_for, 1),
                "pid": child.pid,
            })
            age_txt = f"{age:.0f}s" if age is not None else "never-written"
            _notify(
                f"heartbeat stale ({age_txt}) — killing child pid={child.pid} "
                f"so launchd can restart",
                level="error", title="watchdog",
            )
            _terminate(child)
            # Exit nonzero so launchd/systemd treats this as a crash and
            # restarts us (which respawns the child).
            return 97
        time.sleep(CHECK_INTERVAL_SEC)

    rc = child.returncode
    duration = round(time.time() - started, 1)

    if shutdown_requested["flag"]:
        _record_event({
            "kind": "clean_shutdown",
            "exit_code": rc,
            "duration_sec": duration,
        })
        # Return 0 on deliberate shutdown so launchd doesn't restart.
        return 0

    tail = _tail_stderr()
    level = "info" if rc == 0 else "error"
    _record_event({
        "kind": "exit",
        "exit_code": rc,
        "duration_sec": duration,
        "stderr_tail": tail[:2000],
    })
    if rc == 0:
        # Clean exit but not requested by us — still worth a note. launchd
        # with KeepAlive=true will restart regardless.
        _notify(
            f"tradebot exited cleanly after {duration:.0f}s — restarting",
            level="info",
        )
        return 0
    _notify(
        f"tradebot CRASHED rc={rc} after {duration:.0f}s. "
        f"Last stderr: {tail.splitlines()[-1] if tail else '(empty)'}",
        level="error",
    )
    return rc if rc is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
