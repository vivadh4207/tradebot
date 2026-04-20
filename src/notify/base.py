"""Notifier interface + null implementation + factory."""
from __future__ import annotations

import abc
import os
from typing import Any, Dict, Optional, Set


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
    # The named channels below have built-in title routing; additional
    # fully-operator-defined channels can be added with
    # DISCORD_WEBHOOK_URL_X=<url> and a tag=channel mapping in
    # DISCORD_EXTRA_CHANNEL_ROUTES (see _parse_extra_routes below).
    per_channel = {
        "trades":      _clean(os.getenv("DISCORD_WEBHOOK_URL_TRADES", "")),
        "catalysts":   _clean(os.getenv("DISCORD_WEBHOOK_URL_CATALYSTS", "")),
        "alerts":      _clean(os.getenv("DISCORD_WEBHOOK_URL_ALERTS", "")),
        "calibration": _clean(os.getenv("DISCORD_WEBHOOK_URL_CALIBRATION", "")),
        "reason":      _clean(os.getenv("DISCORD_WEBHOOK_URL_REASON", "")),
    }

    # --- operator-defined extra channels ---
    # Every env var named `DISCORD_WEBHOOK_URL_<NAME>` becomes a
    # channel keyed by the lowercased <NAME>. That way the operator
    # can add an arbitrary number of new channels (e.g. "_LLM",
    # "_AUDIT", "_POLITICAL", "_GAMMA", "_HEADS_UP") without touching
    # the code. Routing from title keywords to those channels is
    # controlled by DISCORD_EXTRA_CHANNEL_ROUTES="title1:channel,
    # title2:channel".
    _known = {"TRADES", "CATALYSTS", "ALERTS", "CALIBRATION", "REASON"}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("DISCORD_WEBHOOK_URL_"):
            continue
        tail = env_key[len("DISCORD_WEBHOOK_URL_"):]
        if not tail or tail in _known:
            continue
        url = _clean(env_val)
        if url:
            per_channel[tail.lower()] = url

    any_per_channel = any(per_channel.values())

    if default_dsc and any_per_channel:
        # Multi-channel routing
        channels = {"default": WebhookNotifier(url=default_dsc, flavor="discord")}
        for ch, url in per_channel.items():
            if url:
                channels[ch] = WebhookNotifier(url=url, flavor="discord")
        return MultiChannelNotifier(channels,
                                      extra_routes=_load_extra_routes())

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
    # "reason-to-trade" channel — high-signal, low-frequency posts that
    # explain *why* the bot would (or would not) be trading. Nightly
    # walk-forward edge report, regime-shift notices, ensemble
    # attribution summaries. Should never be spammy.
    "reason":      {"backtest_report", "edge_report", "regime_shift"},
}


def _load_extra_routes() -> dict:
    """Parse DISCORD_EXTRA_CHANNEL_ROUTES='title1:channel1,title2:channel2'
    so operators can point new titles at their new channels without
    touching the built-in `_TITLE_TO_CHANNEL` map.

    Example: DISCORD_EXTRA_CHANNEL_ROUTES='llm_review:llm,political_alert:political'
    paired with DISCORD_WEBHOOK_URL_LLM=... and DISCORD_WEBHOOK_URL_POLITICAL=...
    gives the bot two brand-new output channels.
    """
    raw = os.getenv("DISCORD_EXTRA_CHANNEL_ROUTES", "").strip()
    if not raw:
        return {}
    extras = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        title, channel = pair.split(":", 1)
        title = title.strip().lower()
        channel = channel.strip().lower()
        if not title or not channel:
            continue
        extras.setdefault(channel, set()).add(title)
    return extras


class MultiChannelNotifier(Notifier):
    """Route notify() calls to one of several WebhookNotifiers based on
    the `title` kwarg AND `level`. Falls back to the `default` channel
    for any unrouted non-error title.

    Designed for Discord multi-channel setups where you want entry/exit
    fills in one channel, catalysts in another, and alerts in a third —
    without flooding a single channel."""

    def __init__(self, channels: dict, extra_routes: Optional[dict] = None):
        """`channels` = {channel_name: Notifier}. Must include key "default".

        `extra_routes` = {channel_name: {title_keyword, ...}} — operator-
        supplied additional title → channel mappings stitched on top of
        the built-in _TITLE_TO_CHANNEL map. Loaded from the env var
        DISCORD_EXTRA_CHANNEL_ROUTES when built via `build_notifier`.
        """
        assert "default" in channels, "MultiChannelNotifier requires a 'default' channel"
        self._channels = channels
        self._extra_routes = extra_routes or {}

    def _pick(self, title: str, level: str = "info") -> Notifier:
        t = (title or "").lower().strip()
        # Built-in routes first
        for chan, keywords in _TITLE_TO_CHANNEL.items():
            if t in keywords and chan in self._channels:
                return self._channels[chan]
        # Operator-supplied routes next (via DISCORD_EXTRA_CHANNEL_ROUTES)
        for chan, keywords in self._extra_routes.items():
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
