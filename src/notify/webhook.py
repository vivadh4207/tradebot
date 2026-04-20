"""Webhook-based notifier (Discord or Slack). Non-blocking, fail-soft.

Runs posts in a background thread so a slow webhook endpoint can never
block the trade loop. A short queue drops oldest messages on overflow.

DISCORD formatting:
  Renders every message as an embed — colored strip on the left, title,
  description, and (optional) list of `meta` fields. This makes the
  Discord channel readable at a glance: greens are profits/entries,
  reds are losses/halts, yellows are warnings.

SLACK formatting:
  Falls back to plain-text (Slack's "text" payload). Rich Slack blocks
  would need separate work; not worth the effort until someone asks.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import Notifier

_log = logging.getLogger(__name__)

# structlog writes ANSI color codes even to files; Discord renders
# them as garbage. Strip at the notifier boundary so every downstream
# caller gets clean text whether it knew to strip or not.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(s: Optional[str]) -> str:
    if not s:
        return ""
    return _ANSI_RE.sub("", s)


# Discord embed colors (integer RGB). Chosen to match a dark-mode theme
# so the strip is readable regardless of the user's Discord theme.
_COLOR_BY_LEVEL = {
    "info":    0x5865F2,   # Discord blurple
    "success": 0x2ECC71,   # green
    "warn":    0xF5A623,   # amber
    "error":   0xED4245,   # red
    # Event-specific overrides. Can be overridden per-call via
    # meta={"_color": 0xAABBCC} if the caller wants something custom.
    "entry":   0x57F287,   # bright green
    "exit":    0x5865F2,   # blurple — actual exit color derived from realized P&L
    "halt":    0xED4245,
    "startup": 0x95A5A6,
    "shutdown": 0xE67E22,
    "calibration": 0xF5A623,
    "watchdog": 0xED4245,
}


def _color_for(level: str, title: str, meta: Optional[Dict[str, Any]]) -> int:
    """Pick an embed color. Priority:
      1. explicit meta["_color"]
      2. title-based (entry / exit / halt / etc.)
      3. exit P&L sign (green win, red loss)
      4. level (info/warn/error/success)
    """
    if meta:
        c = meta.get("_color")
        if isinstance(c, int):
            return c
    t = (title or "").lower().strip()
    if t in _COLOR_BY_LEVEL and t not in {"exit"}:
        return _COLOR_BY_LEVEL[t]
    if t == "exit" and meta:
        pnl = meta.get("pnl") or meta.get("realized_pnl") or meta.get("pnl_pct")
        if isinstance(pnl, (int, float)):
            return 0x2ECC71 if pnl > 0 else 0xED4245 if pnl < 0 else 0x95A5A6
    return _COLOR_BY_LEVEL.get(level, 0x5865F2)


def _fmt_field_value(v: Any) -> str:
    """Keep field values short + scannable. Numbers get decimals
    trimmed; long strings get truncated; dicts/lists are json-dumped."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        # trim trailing zeros, keep up to 4 decimals
        return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, (dict, list, tuple)):
        try:
            s = json.dumps(v, default=str, separators=(",", ":"))
        except Exception:
            s = str(v)
        return s if len(s) <= 100 else s[:97] + "…"
    s = str(v)
    return s if len(s) <= 200 else s[:197] + "…"


def _meta_to_fields(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert meta dict to Discord embed fields. Keys prefixed with
    underscore are control keys (_color, _footer, etc.) and skipped."""
    fields: List[Dict[str, Any]] = []
    for k, v in meta.items():
        if k.startswith("_"):
            continue
        # inline=True so three fields pack into a row
        fields.append({
            "name": str(k).replace("_", " "),
            "value": _fmt_field_value(v),
            "inline": True,
        })
        if len(fields) >= 24:   # Discord caps at 25 fields per embed
            break
    return fields


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

    def notify(self, text: str, *, level: str = "info", title: str = "",
                meta: Optional[Dict[str, Any]] = None) -> None:
        """Queue a message for async delivery. Never blocks, never raises."""
        # Strip any ANSI escapes the caller accidentally included (e.g.
        # raw structlog output from a crash-alert tail).
        clean_text = _strip_ansi(text)
        clean_meta: Optional[Dict[str, Any]] = None
        if meta:
            clean_meta = {k: (_strip_ansi(v) if isinstance(v, str) else v)
                          for k, v in meta.items()}
        payload: Tuple[str, str, str, Optional[Dict[str, Any]]] = (
            clean_text, level, title, clean_meta,
        )
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            try:
                self._q.get_nowait()      # drop oldest
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._timeout + 1)

    # --- internal ---
    def _worker(self) -> None:
        self._first_post_logged = False
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._post(*item)
                if not self._first_post_logged:
                    _log.info("notifier_post_ok flavor=%s", self._flavor)
                    self._first_post_logged = True
            except urllib.error.HTTPError as e:
                _log.warning("notifier_post_http_error status=%s reason=%s title=%r",
                              e.code, e.reason, item[2])
            except urllib.error.URLError as e:
                _log.warning("notifier_post_network_error err=%s title=%r",
                              e.reason, item[2])
            except Exception as e:                 # noqa: BLE001
                _log.warning("notifier_post_failed err=%s title=%r", e, item[2])

    def _post(self, text: str, level: str, title: str,
               meta: Optional[Dict[str, Any]]) -> None:
        if self._flavor == "discord":
            payload = self._build_discord(text, level, title, meta)
        else:
            payload = self._build_slack(text, level, title, meta)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=data,
            headers={
                "Content-Type": "application/json",
                # Cloudflare in front of Discord rejects default urllib UAs.
                "User-Agent": "tradebot-notifier/1.0 (+https://github.com)",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self._timeout).read()

    def _build_discord(self, text: str, level: str, title: str,
                        meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a Discord embed payload. Fields in `meta` become
        inline fields on the embed; `text` becomes the description."""
        embed: Dict[str, Any] = {
            "title": title or level.upper(),
            "description": (text or "")[:4000],
            "color": _color_for(level, title, meta),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        if meta:
            fields = _meta_to_fields(meta)
            if fields:
                embed["fields"] = fields
            footer = meta.get("_footer")
            if footer:
                embed["footer"] = {"text": str(footer)[:2000]}
        return {"embeds": [embed]}

    def _build_slack(self, text: str, level: str, title: str,
                      meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Slack fallback: plain text with level-prefix + optional meta
        rendered as `key=value` pairs on a second line."""
        tag = {"info": "[info]", "warn": "[warn]", "error": "[ERROR]",
                "success": "[ok]"}.get(level, "[info]")
        label = f"{title}: " if title else ""
        line = f"{tag} {label}{text}"
        if meta:
            extras = " ".join(
                f"{k}={_fmt_field_value(v)}"
                for k, v in meta.items() if not k.startswith("_")
            )
            if extras:
                line = line + "\n" + extras
        return {"text": line[:3900]}
