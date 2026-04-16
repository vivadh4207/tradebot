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
    """
    from .webhook import WebhookNotifier
    dsc = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    slk = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if dsc:
        return WebhookNotifier(url=dsc, flavor="discord")
    if slk:
        return WebhookNotifier(url=slk, flavor="slack")
    return NullNotifier()
