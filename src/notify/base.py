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
    """Pick a notifier based on env.

    Single-channel mode (legacy):
      - DISCORD_WEBHOOK_URL present → all events go there
      - SLACK_WEBHOOK_URL present   → all events go there
      - else                        → NullNotifier

    Multi-channel mode (recommended): set additional env vars to route
    by event type. Missing channels fall back to DISCORD_WEBHOOK_URL.
      - DISCORD_WEBHOOK_URL_TRADES      → entry / exit fills
      - DISCORD_WEBHOOK_URL_CATALYSTS   → earnings / FDA events
      - DISCORD_WEBHOOK_URL_ALERTS      → HALT / shutdown / watchdog
      - DISCORD_WEBHOOK_URL_CALIBRATION → slippage calibration drift
      - DISCORD_WEBHOOK_URL             → everything else (startup, EOD)

    Defensive cleanup: strip whitespace *and* surrounding quotes from the
    env value. A leading space or quote-wrapped URL in .env silently
    breaks urllib.request.
    """
    from .webhook import WebhookNotifier

    def _clean(val: str) -> str:
        v = (val or "").strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1].strip()
        return v

    default_dsc = _clean(os.getenv("DISCORD_WEBHOOK_URL", ""))
    slk = _clean(os.getenv("SLACK_WEBHOOK_URL", ""))

    # Check for any per-channel overrides. Each optional; missing →
    # falls back to default (or to Null if no default either).
    per_channel = {
        "trades":      _clean(os.getenv("DISCORD_WEBHOOK_URL_TRADES", "")),
        "catalysts":   _clean(os.getenv("DISCORD_WEBHOOK_URL_CATALYSTS", "")),
        "alerts":      _clean(os.getenv("DISCORD_WEBHOOK_URL_ALERTS", "")),
        "calibration": _clean(os.getenv("DISCORD_WEBHOOK_URL_CALIBRATION", "")),
    }
    any_per_channel = any(per_channel.values())

    if default_dsc and any_per_channel:
        # Multi-channel routing
        channels = {"default": WebhookNotifier(url=default_dsc, flavor="discord")}
        for ch, url in per_channel.items():
            if url:
                channels[ch] = WebhookNotifier(url=url, flavor="discord")
        return MultiChannelNotifier(channels)

    if default_dsc:
        return WebhookNotifier(url=default_dsc, flavor="discord")
    if slk:
        return WebhookNotifier(url=slk, flavor="slack")
    return NullNotifier()


# ------------------------------------------------------------- routing
# Maps event titles to a channel name. Matching is case-insensitive on
# the title keyword. First match wins. Unmatched titles fall back to
# "default" (or to "alerts" when level == "error" — see _pick below).
_TITLE_TO_CHANNEL = {
    "trades":      {"entry", "exit"},
    "catalysts":   {"catalysts", "catalyst"},
    # startup / shutdown / halts / watchdog / reconcile / news blocks /
    # and generic error-level messages all route to the alerts channel.
    # Operator preference: "startup and any errors should go to alerts".
    "alerts":      {"halt", "shutdown", "startup", "watchdog",
                     "news block", "reconcile", "error", "risk"},
    "calibration": {"calibration"},
}


class MultiChannelNotifier(Notifier):
    """Route notify() calls to one of several WebhookNotifiers based on
    the `title` kwarg AND `level`. Falls back to the `default` channel
    for any unrouted non-error title.

    Designed for Discord multi-channel setups where you want entry/exit
    fills in one channel, catalysts in another, and alerts in a third —
    without flooding a single channel."""

    def __init__(self, channels: dict):
        """`channels` = {channel_name: Notifier}. Must include key "default"."""
        assert "default" in channels, "MultiChannelNotifier requires a 'default' channel"
        self._channels = channels

    def _pick(self, title: str, level: str = "info") -> Notifier:
        t = (title or "").lower().strip()
        for chan, keywords in _TITLE_TO_CHANNEL.items():
            if t in keywords and chan in self._channels:
                return self._channels[chan]
        # Any error-level message without a specific channel goes to
        # alerts (if configured). Keeps default channel free of noise.
        if level == "error" and "alerts" in self._channels:
            return self._channels["alerts"]
        return self._channels["default"]

    def notify(self, text: str, *, level: str = "info", title: str = "",
                meta=None) -> None:
        self._pick(title, level).notify(text, level=level, title=title, meta=meta)

    def close(self) -> None:
        """Close every underlying WebhookNotifier's queue."""
        for n in self._channels.values():
            if hasattr(n, "close"):
                try:
                    n.close()
                except Exception:
                    pass
