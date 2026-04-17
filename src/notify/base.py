"""Notifier interface + null implementation + factory."""
from __future__ import annotations

import abc
import os
from typing import Any, Dict, Optional


class Notifier(abc.ABC):
    """Push short status messages to a human channel.

    The `meta` kwarg carries structured event data the notifier can use
    to render richer output (Discord embed, Slack block kit, etc.).
    Plain-text callers can ignore it; rich callers pass an ordered
    mapping of field-name → value and the notifier lays them out
    consistently.
    """

    @abc.abstractmethod
    def notify(self, text: str, *, level: str = "info", title: str = "",
                meta: Optional[Dict[str, Any]] = None) -> None:
        """Send a message.

        Args:
          text:  short summary (plain text, used as a fallback / preview)
          level: "info" | "warn" | "error" | "success" — drives color + emoji
          title: optional short title shown in bold at top of the card
          meta:  optional {field_name: value} dict rendered as a field list
        """


class NullNotifier(Notifier):
    """No-op. Used when no webhook is configured or during tests."""

    def notify(self, text: str, *, level: str = "info", title: str = "",
                meta: Optional[Dict[str, Any]] = None) -> None:
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
