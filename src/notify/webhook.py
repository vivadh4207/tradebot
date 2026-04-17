"""Webhook-based notifier (Discord or Slack). Non-blocking, fail-soft.

Runs posts in a background thread so a slow webhook endpoint can never
block the trade loop. A short queue drops oldest messages on overflow.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
import urllib.error
from typing import Optional

from .base import Notifier

_log = logging.getLogger(__name__)


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
        # First-POST sentinel so we log a single success or failure after
        # startup — operators need to know whether the webhook actually
        # works, but we don't want per-message spam.
        self._first_post_logged = False
        while not self._stop.is_set():
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._post(msg)
                if not self._first_post_logged:
                    _log.info("notifier_post_ok flavor=%s", self._flavor)
                    self._first_post_logged = True
            except urllib.error.HTTPError as e:
                _log.warning("notifier_post_http_error status=%s reason=%s msg=%r",
                              e.code, e.reason, msg[:120])
            except urllib.error.URLError as e:
                _log.warning("notifier_post_network_error err=%s msg=%r",
                              e.reason, msg[:120])
            except Exception as e:                 # noqa: BLE001
                _log.warning("notifier_post_failed err=%s msg=%r",
                              e, msg[:120])

    def _post(self, msg: str) -> None:
        if self._flavor == "discord":
            payload = {"content": msg[:1900]}   # Discord 2000-char cap
        else:
            payload = {"text": msg[:3900]}      # Slack ~4000-char cap
        data = json.dumps(payload).encode("utf-8")
        # Cloudflare in front of Discord rejects requests with the
        # default urllib User-Agent (returns 403). A non-default UA
        # passes. We send a stable identifier so the requests show up
        # in our webhook audit log as recognizable.
        req = urllib.request.Request(
            self._url, data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "tradebot-notifier/1.0 (+https://github.com)",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self._timeout).read()
