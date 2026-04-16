"""Webhook-based notifier (Discord or Slack). Non-blocking, fail-soft.

Runs posts in a background thread so a slow webhook endpoint can never
block the trade loop. A short queue drops oldest messages on overflow.
"""
from __future__ import annotations

import json
import queue
import threading
import urllib.request
import urllib.error
from typing import Optional

from .base import Notifier


_EMOJI = {"info": "[info]", "warn": "[warn]", "error": "[ERROR]"}


class WebhookNotifier(Notifier):
    def __init__(self, url: str, flavor: str = "discord",
                 timeout: float = 4.0, queue_size: int = 64):
        assert flavor in {"discord", "slack"}, "flavor must be discord or slack"
        self._url = url
        self._flavor = flavor
        self._timeout = timeout
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name=f"notifier-{flavor}", daemon=True,
        )
        self._thread.start()

    def notify(self, text: str, *, level: str = "info", title: str = "") -> None:
        tag = _EMOJI.get(level, "[info]")
        label = f"{title}: " if title else ""
        line = f"{tag} {label}{text}"
        try:
            self._q.put_nowait(line)
        except queue.Full:
            # drop oldest, add newest
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(line)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._timeout + 1)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._post(msg)
            except Exception:
                pass  # never bubble up — notifications are best-effort

    def _post(self, msg: str) -> None:
        if self._flavor == "discord":
            payload = {"content": msg[:1900]}   # Discord 2000-char cap
        else:
            payload = {"text": msg[:3900]}      # Slack ~4000-char cap
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self._timeout).read()
