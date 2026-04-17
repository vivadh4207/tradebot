"""Notifier interface + null implementation + factory."""
from __future__ import annotations

import abc
import os
from typing import Optional


class Notifier(abc.ABC):
    """Push short status messages to a human channel."""

    @abc.abstractmethod
    def notify(self, text: str, *, level: str = "info", title: str = "") -> None:
        """Send a message. level in {info, warn, error}. Must not raise."""


class NullNotifier(Notifier):
    """No-op. Used when no webhook is configured or during tests."""

    def notify(self, text: str, *, level: str = "info", title: str = "") -> None:
        return


def build_notifier() -> Notifier:
    """Pick a notifier based on env:
    - DISCORD_WEBHOOK_URL present → Discord
    - SLACK_WEBHOOK_URL present   → Slack
    - else                        → NullNotifier

    Defensive cleanup: strip whitespace *and* surrounding quotes from the
    env value. A leading space or quote-wrapped URL in .env silently
    breaks urllib.request, and the WebhookNotifier's error path is
    non-fatal — so the bot happily runs with broken alerts unless we
    catch it here.
    """
    from .webhook import WebhookNotifier

    def _clean(val: str) -> str:
        v = (val or "").strip()
        # Strip a single layer of surrounding quotes if present.
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1].strip()
        return v

    dsc = _clean(os.getenv("DISCORD_WEBHOOK_URL", ""))
    slk = _clean(os.getenv("SLACK_WEBHOOK_URL", ""))
    if dsc:
        return WebhookNotifier(url=dsc, flavor="discord")
    if slk:
        return WebhookNotifier(url=slk, flavor="slack")
    return NullNotifier()
