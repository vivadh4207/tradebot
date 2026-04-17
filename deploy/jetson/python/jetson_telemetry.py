"""Thin wrapper around jetson-stats (jtop) for the notifier.

Pushes a compact snapshot to the existing Notifier every 15 minutes:
GPU util, GPU temp, CPU load avg, RAM used, power draw. Cheap to run on
a separate thread; the bot's main loop is untouched.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class JetsonTelemetry:
    """Start with `JetsonTelemetry(notifier).start()`; call `.stop()` to end."""

    def __init__(self, notifier, interval_sec: int = 900):
        self._notifier = notifier
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            from jtop import jtop
            self._jtop_cls = jtop
        except ImportError:
            self._jtop_cls = None

    def start(self) -> None:
        if self._jtop_cls is None:
            return
        self._thread = threading.Thread(target=self._loop,
                                          name="jetson-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        try:
            with self._jtop_cls() as jetson:
                while not self._stop.is_set():
                    if not jetson.ok():
                        time.sleep(2)
                        continue
                    try:
                        stats = self._snapshot(jetson)
                        self._notifier.notify(stats, title="jetson")
                    except Exception:
                        pass
                    self._stop.wait(self._interval)
        except Exception:
            pass

    @staticmethod
    def _snapshot(jetson) -> str:
        stats = jetson.stats
        cpu = stats.get("CPU1", 0)
        gpu = stats.get("GPU", 0)
        mem = stats.get("RAM", 0)
        tgpu = stats.get("Temp GPU", 0)
        tcpu = stats.get("Temp CPU", 0)
        power = stats.get("Power TOT", 0)
        return (f"gpu {gpu:.0f}% {tgpu:.0f}°C · cpu {cpu:.0f}% {tcpu:.0f}°C · "
                f"ram {mem:.0f}% · power {power:.1f}W")
