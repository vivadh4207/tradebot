"""Discord terminal — control the Jetson bot from a Discord channel.

Two surfaces, same whitelist:

  1. **Text commands** — start messages with `!` in the configured
     channel: `!status`, `!start`, `!stop`, `!restart`, `!logs 100`,
     `!walkforward`, `!audit`, `!positions`, `!doctor`, `!risk-switch`,
     `!reset-paper`, `!wipe`, `!help`.

  2. **Button panel** — run `!panel` once to post a persistent message
     with clickable buttons. Buttons hit the same whitelist. Destructive
     actions (reset-paper, wipe) require a confirm prompt.

## Security model

This is a remote shell over Discord. The posture is defensive:

  - **Channel allowlist.** Bot reads ONLY in DISCORD_TERMINAL_CHANNEL_ID.
    Ignored in DMs and other channels.
  - **User allowlist.** DISCORD_TERMINAL_AUTHORIZED_USERS is a
    comma-separated list of Discord user IDs. Anyone not in that list
    gets an ephemeral "not authorized" response and nothing runs.
  - **Command whitelist.** No arbitrary shell. Every button + text
    command maps to a fixed subcommand of `tradebotctl.sh` (itself
    audited).
  - **Rate limit.** 12 commands per user per minute, tracked in memory.
  - **Audit log.** Every accepted/rejected command is appended to
    logs/discord_terminal.jsonl with user ID, command, result code.
  - **Destructive-action gate.** reset-paper and wipe require the user
    to type the literal word DESTROY within 30 seconds of the request.

Dependencies: `discord.py>=2.3` (async, includes Components / Views).
Install with `pip install discord.py>=2.3`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


_log = logging.getLogger("discord_terminal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# --------------------------------------------------------------- config


def _parse_id_set(env_value: str) -> Set[int]:
    """Parse 'id1,id2,id3' → set of ints. Ignores blanks and bad values."""
    out: Set[int] = set()
    for raw in (env_value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.add(int(raw))
        except ValueError:
            _log.warning("discord_terminal_bad_id value=%s", raw)
    return out


BOT_TOKEN = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()

# Multi-channel: the bot listens in ANY channel whose ID is in this set.
# Supports both the new list variable AND the legacy single-channel one
# so old configs keep working.
CHANNEL_IDS: Set[int] = (
    _parse_id_set(os.getenv("DISCORD_TERMINAL_CHANNEL_IDS") or "")
    | _parse_id_set(os.getenv("DISCORD_TERMINAL_CHANNEL_ID") or "")
)

# Channels whose free-form chat should use the 70B model for detailed
# answers instead of the 8B default. These must ALSO be in CHANNEL_IDS.
CHAT_70B_CHANNEL_IDS: Set[int] = _parse_id_set(
    os.getenv("DISCORD_CHAT_70B_CHANNELS")
    or os.getenv("DISCORD_CHAT_70B_CHANNEL_IDS") or ""
)

# Tag to use when a channel is in the 70B set. Falls back to LLM_AUDITOR_MODEL
# (same 70B that does the nightly audit) to avoid duplicating the model spec.
CHAT_70B_MODEL = (
    os.getenv("LLM_CHAT_70B_MODEL", "").strip()
    or os.getenv("LLM_AUDITOR_MODEL", "").strip()
    or "llama3.1:70b"
)

AUTH_USERS = _parse_id_set(os.getenv("DISCORD_TERMINAL_AUTHORIZED_USERS") or "")
COMMAND_PREFIX    = os.getenv("DISCORD_TERMINAL_PREFIX", "!").strip() or "!"
MAX_OUT_CHARS     = 1850      # keep us under Discord's 2000-char message cap
CMD_TIMEOUT_SEC   = 180.0     # longest single command (70B audit can take 2 min)
# 70B chat can be slow (3-6 tok/sec on Jetson); give it room without
# blocking the Discord heartbeat forever.
CHAT_70B_TIMEOUT_SEC = float(os.getenv("LLM_CHAT_70B_TIMEOUT_SEC", "180"))
CHAT_70B_MAX_TOKENS  = int(os.getenv("LLM_CHAT_70B_MAX_TOKENS", "700"))
RATE_LIMIT_MAX    = 12
RATE_LIMIT_WINDOW = 60.0      # seconds
AUDIT_LOG_PATH    = Path(
    os.getenv("DISCORD_TERMINAL_AUDIT_LOG")
    or str(ROOT / "logs" / "discord_terminal.jsonl")
)


# --------------------------------------------------------------- whitelist


# Maps public command name → (tradebotctl subcommand, needs_destroy_confirm)
# The value tuple's second item gates commands like reset-paper behind a
# typed DESTROY confirmation. Never put anything NOT in this map.
COMMAND_MAP: Dict[str, Tuple[str, bool]] = {
    "status":        ("status",          False),
    "start":         ("start",           False),
    "stop":          ("stop",            False),
    "restart":       ("restart",         False),
    "doctor":        ("doctor",          False),
    "logs":          ("logs-tail",       False),     # handled specially; see below
    "positions":     ("positions-print", False),     # likewise
    "walkforward":   ("walkforward",     False),
    "risk-switch":   ("putcall-oi",      False),
    "audit":         ("strategy-audit",  False),
    "summary":       ("log-summary",     False),     # handled in-process; see _handle_summary
    "ollama-status":  ("ollama-status",  False),
    "ollama-restart": ("ollama-restart", False),
    "ollama-warmup":  ("ollama-warmup",  False),
    "research":       ("options-research", False),   # handled in-process; see _handle_research
    "catalyst":       ("catalyst-dive",   False),    # handled in-process; see _handle_catalyst
    "llm-autotrade":  ("llm-autotrade",   False),    # handled in-process; see _handle_llm_autotrade
    "saves":          ("saves-report",    False),    # handled in-process; see _handle_saves
    "close":          ("manual-close",    False),    # handled in-process; see _handle_close
    "trim":           ("manual-trim",     False),    # handled in-process; see _handle_trim
    "cleanup":        ("broker-cleanup",  False),    # handled in-process; see _handle_cleanup
    "intel":          ("finnhub-intel",   False),    # handled in-process; see _handle_intel
    "calendars":      ("finnhub-calendars", False),  # handled in-process; see _handle_calendars
    "reset-paper":   ("reset-paper",     True),       # requires DESTROY confirm
    "wipe":          ("wipe-journal",    True),       # requires DESTROY confirm
}

HELP_TEXT = (
    f"Command prefix: `{COMMAND_PREFIX}`\n\n"
    "Safe:\n"
    f"  `{COMMAND_PREFIX}status`        — bot running + last tick\n"
    f"  `{COMMAND_PREFIX}positions`    — current open positions\n"
    f"  `{COMMAND_PREFIX}logs [N]`     — last N log lines (default 100)\n"
    f"  `{COMMAND_PREFIX}doctor`        — run readiness check\n"
    f"  `{COMMAND_PREFIX}panel`         — post the button-panel in this channel\n"
    f"  `{COMMAND_PREFIX}ask <question>` — ask the LLM (8B default, 70B in "
    "designated channels)\n"
    "\nLifecycle:\n"
    f"  `{COMMAND_PREFIX}start`  `{COMMAND_PREFIX}stop`  `{COMMAND_PREFIX}restart`\n"
    "\nReports:\n"
    f"  `{COMMAND_PREFIX}walkforward`   — nightly edge report on demand\n"
    f"  `{COMMAND_PREFIX}risk-switch`  — refresh CBOE put/call OI state\n"
    f"  `{COMMAND_PREFIX}audit`         — 70B strategy audit (~30-120s)\n"
    f"  `{COMMAND_PREFIX}summary [N]`   — concise digest of last N min (default 60)\n"
    f"  `{COMMAND_PREFIX}research [SYMS]` — 70B options-research ideas "
    "(default SPY QQQ)\n"
    f"  `{COMMAND_PREFIX}catalyst [SYMS]` — deep catalyst dive across news, "
    "social, earnings (default: SPY QQQ + big tech)\n"
    f"  `{COMMAND_PREFIX}llm-autotrade [on|off|status]` — toggle LLM-originated trades\n"
    f"  `{COMMAND_PREFIX}ollama-status` — Ollama daemon + loaded models\n"
    f"  `{COMMAND_PREFIX}ollama-warmup` — Pre-load 8B + 70B into GPU memory\n"
    f"  `{COMMAND_PREFIX}ollama-restart` — Restart Ollama daemon\n"
    "\nDangerous (require DESTROY confirmation):\n"
    f"  `{COMMAND_PREFIX}reset-paper`  — flatten + wipe journal\n"
    f"  `{COMMAND_PREFIX}wipe`          — wipe journal only\n"
    "\nFree-form chat: in chat-enabled channels, any non-command message "
    "routes to the LLM automatically.\n"
)


# --------------------------------------------------------------- helpers


def _truncate(out: str, limit: int = MAX_OUT_CHARS) -> str:
    """Discord caps messages at 2000 chars; we keep a safety margin and
    tell the user if we had to cut."""
    if len(out) <= limit:
        return out
    cut = out[:limit - 80]
    return cut + f"\n\n… [truncated {len(out) - len(cut)} chars]"


def _audit(record: Dict) -> None:
    """Append one JSONL audit line. Creates the logs dir on first write.
    Swallows IO errors — audit failure must never block the bot."""
    try:
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record["ts"] = datetime.now(tz=timezone.utc).isoformat()
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception:
        pass


class RateLimiter:
    """Per-user sliding-window rate limit."""

    def __init__(self, max_count: int = RATE_LIMIT_MAX,
                 window_sec: float = RATE_LIMIT_WINDOW):
        self.max = max_count
        self.window = window_sec
        self._events: Dict[int, deque] = defaultdict(deque)

    def allow(self, user_id: int) -> bool:
        now = time.time()
        dq = self._events[user_id]
        cutoff = now - self.window
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.max:
            return False
        dq.append(now)
        return True


# --------------------------------------------------------------- runner


class CommandRunner:
    """Wraps auth + whitelist + subprocess exec. Used by both the
    text-command path and the button-panel path."""

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.pending_destructive: Dict[int, Tuple[str, float]] = {}
        # user_id → (command, expires_at)

    def is_authorized(self, user_id: int) -> bool:
        return user_id in AUTH_USERS

    async def run(self, user_id: int, user_name: str,
                  command: str,
                  raw_text: str = "") -> str:
        """Entry point. Returns the message text to post back."""
        if not self.is_authorized(user_id):
            _audit({"kind": "rejected_auth", "user": user_id,
                    "user_name": user_name, "command": command})
            return "Not authorized."

        if not self.rate_limiter.allow(user_id):
            _audit({"kind": "rejected_rate_limit",
                    "user": user_id, "command": command})
            return f"Rate limit: max {RATE_LIMIT_MAX} cmds / "\
                   f"{int(RATE_LIMIT_WINDOW)}s."

        # Destructive-command confirmation flow
        pending = self.pending_destructive.get(user_id)
        if command == "DESTROY" and pending is not None:
            cmd, expires = pending
            if time.time() > expires:
                self.pending_destructive.pop(user_id, None)
                return "Confirmation expired. Re-run the command."
            # fall through to execute the pending destructive command
            command = cmd
            self.pending_destructive.pop(user_id, None)
            return await self._exec(user_id, command, raw_text, destroyed=True)

        if command not in COMMAND_MAP:
            return f"Unknown command `{command}`. Try `!help`."

        _, needs_destroy = COMMAND_MAP[command]
        if needs_destroy and not self._has_valid_destroy_pending(user_id, command):
            # Arm it; wait for a follow-up `!DESTROY`
            self.pending_destructive[user_id] = (command, time.time() + 30.0)
            _audit({"kind": "awaiting_destroy",
                    "user": user_id, "command": command})
            return (f"**DESTRUCTIVE** — `{command}` will change live state. "
                    f"Reply with `{COMMAND_PREFIX}DESTROY` within 30s to "
                    f"confirm, or anything else to cancel.")

        return await self._exec(user_id, command, raw_text)

    def _has_valid_destroy_pending(self, user_id: int, cmd: str) -> bool:
        pending = self.pending_destructive.get(user_id)
        if pending is None:
            return False
        pending_cmd, expires = pending
        return pending_cmd == cmd and time.time() <= expires

    async def _exec(self, user_id: int, command: str,
                     raw_text: str, destroyed: bool = False) -> str:
        subcmd, _ = COMMAND_MAP[command]
        # Special cases: logs N, positions
        extra_args: List[str] = []
        if command == "logs":
            tail_n = 100
            parts = (raw_text or "").split()
            if len(parts) >= 2 and parts[1].isdigit():
                tail_n = max(5, min(2000, int(parts[1])))
            return await self._tail_logs(tail_n)
        if command == "positions":
            return await self._positions_print()
        if command == "summary":
            # Optional minute-window argument: "!summary 30"
            window = 60
            parts = (raw_text or "").split()
            if len(parts) >= 2 and parts[1].isdigit():
                window = max(5, min(1440, int(parts[1])))
            return await self._handle_summary(window)
        if command == "research":
            # "!research" (default SPY+QQQ) or "!research TSLA" etc.
            parts = (raw_text or "").split()
            syms = [s.upper() for s in parts[1:]] if len(parts) >= 2 else ["SPY", "QQQ"]
            return await self._handle_research(syms)
        if command == "catalyst":
            # `!catalyst` or `!catalyst AAPL MSFT NVDA`
            parts = (raw_text or "").split()
            syms = [s.upper() for s in parts[1:]] if len(parts) >= 2 \
                else ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"]
            return await self._handle_catalyst(syms)
        if command == "llm-autotrade":
            parts = (raw_text or "").split()
            sub = parts[1].lower() if len(parts) >= 2 else "status"
            return self._handle_llm_autotrade(sub)
        if command == "close":
            # Manual close via advisory id: "!close <aid>"
            parts = (raw_text or "").split()
            if len(parts) < 2:
                return ("usage: `!close <advisory_id>` or `!close <symbol>` — "
                        "run `!positions` for live list")
            return await self._handle_close(parts[1].strip())
        if command == "trim":
            # Manual half-close: "!trim <aid>"
            parts = (raw_text or "").split()
            if len(parts) < 2:
                return "usage: `!trim <advisory_id>`"
            return await self._handle_trim(parts[1].strip())
        if command == "saves":
            # "!saves" or "!saves 168" (weekly)
            parts = (raw_text or "").split()
            hours = 24
            if len(parts) >= 2 and parts[1].isdigit():
                hours = max(1, min(720, int(parts[1])))
            return await self._handle_saves(hours)
        if command == "cleanup":
            # List orphaned positions across brokers (local vs tradier
            # vs alpaca mismatches) + optional `--close` to wipe them.
            parts = (raw_text or "").split()
            do_close = any(p.lower() in ("--close", "close")
                            for p in parts[1:])
            return await self._handle_cleanup(do_close)
        if command == "intel":
            parts = (raw_text or "").split()
            if len(parts) < 2:
                return "usage: `!intel SYMBOL` (e.g. `!intel AAPL`)"
            return await self._handle_intel(parts[1].upper().strip())
        if command == "calendars":
            return await self._handle_calendars()
        if destroyed and subcmd == "reset-paper":
            # reset_paper.py supports --yes to skip its own interactive prompt
            extra_args = ["--yes"]

        started = time.time()
        rc, stdout, stderr = await self._sh(
            ["/bin/bash", str(ROOT / "scripts" / "tradebotctl.sh"), subcmd,
             *extra_args],
            cwd=str(ROOT),
            timeout=CMD_TIMEOUT_SEC,
        )
        elapsed = time.time() - started

        _audit({
            "kind": "executed",
            "user": user_id,
            "command": command,
            "subcmd": subcmd,
            "rc": rc,
            "elapsed_sec": round(elapsed, 2),
            "destructive": destroyed,
        })

        head = f"`{COMMAND_PREFIX}{command}` → exit {rc} ({elapsed:.1f}s)"
        body = ""
        if stdout:
            body += f"\n```\n{_truncate(stdout)}\n```"
        if stderr and stderr.strip():
            body += f"\n**stderr:**```\n{_truncate(stderr, 400)}\n```"
        return head + body

    def _handle_llm_autotrade(self, sub: str) -> str:
        """Status / kill-switch for LLM-originated trades. Writes a
        file-based sentinel so bot picks it up on next tick — no
        restart required."""
        try:
            from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
            q = LLMAutotradeQueue()
            if sub == "on":
                q.set_killed(False)
                msg = "LLM autotrade kill-switch CLEARED (bot will execute fresh ideas if LLM_AUTOTRADE=1)."
            elif sub in ("off", "kill"):
                q.set_killed(True)
                msg = "LLM autotrade KILLED — no more LLM-originated trades until you run `!llm-autotrade on`."
            else:
                s = q.peek_state()
                import os as _os
                env_enabled = _os.getenv("LLM_AUTOTRADE", "").strip() in ("1", "true", "yes")
                lines = [
                    f"**LLM autotrade status**",
                    f"· env `LLM_AUTOTRADE`: {'ENABLED' if env_enabled else 'DISABLED'}",
                    f"· kill switch: {'ACTIVE' if s['killed'] else 'clear'}",
                    f"· today: {s['daily_count']}/{s['daily_cap']} trades",
                    f"· queue: {s['queue_fresh']} fresh ideas waiting ({s['queue_total']} total)",
                    f"· allowed confidence: {', '.join(s['allowed_confidences'])}",
                    f"· max idea age: {s['max_age_min']} min",
                ]
                msg = "\n".join(lines)
            return msg
        except Exception as e:                          # noqa: BLE001
            return f"llm-autotrade failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_close(self, token: str) -> str:
        """Close a position manually. Token can be an advisory_id (from
        fade advisory) OR a symbol/underlying. Writes a close intent to
        data/manual_close_intents.json which the bot polls and executes
        on its next fast_loop tick.
        """
        try:
            from pathlib import Path as _P
            import json as _json, time as _t
            from src.intelligence.position_advisor import load_advisory
            from src.core.data_paths import data_path
            intent_path = _P(data_path("manual_close_intents.json"))
            intent_path.parent.mkdir(parents=True, exist_ok=True)

            # Resolve token -> intent payload
            payload = None
            adv = load_advisory(token)
            if adv:
                payload = {
                    "symbol": adv.get("symbol"),
                    "kind": "full_close",
                    "advisory_id": token,
                    "source": "discord_manual",
                    "ts": _t.time(),
                }
            else:
                # Treat as symbol (closes ALL positions for that underlying)
                payload = {
                    "symbol": token.upper(),
                    "kind": "full_close",
                    "source": "discord_manual_symbol",
                    "ts": _t.time(),
                }
            try:
                existing = _json.loads(intent_path.read_text() or "[]")
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
            existing.append(payload)
            intent_path.write_text(_json.dumps(existing, indent=2,
                                                 default=str))
            _audit({"kind": "manual_close_requested",
                    "token": token, "payload": payload})
            return (
                f"🛑 **Close intent queued** for `{payload['symbol']}` "
                f"(source: {payload['source']}).\n"
                "Bot will execute on the next fast_loop tick (~1s)."
            )
        except Exception as e:
            return f"close failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_trim(self, token: str) -> str:
        """Close 50% of the position. Same intent file, kind=trim_half."""
        try:
            from pathlib import Path as _P
            import json as _json, time as _t
            from src.intelligence.position_advisor import load_advisory
            from src.core.data_paths import data_path
            intent_path = _P(data_path("manual_close_intents.json"))
            intent_path.parent.mkdir(parents=True, exist_ok=True)
            adv = load_advisory(token)
            symbol = (adv.get("symbol") if adv else token.upper())
            payload = {
                "symbol": symbol, "kind": "trim_half",
                "advisory_id": token if adv else None,
                "source": "discord_manual_trim", "ts": _t.time(),
            }
            try:
                existing = _json.loads(intent_path.read_text() or "[]")
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []
            existing.append(payload)
            intent_path.write_text(_json.dumps(existing, indent=2,
                                                 default=str))
            _audit({"kind": "manual_trim_requested",
                    "token": token, "payload": payload})
            return f"✂️ **Trim intent queued** for `{symbol}` (50% close)."
        except Exception as e:
            return f"trim failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_intel(self, symbol: str) -> str:
        """Fundamentals deep-dive for a single symbol via Finnhub.
        Returns a markdown-formatted report the user can scan quickly.
        """
        try:
            from src.intelligence.finnhub_intelligence import (
                build_finnhub_intelligence,
            )
            fh = build_finnhub_intelligence()
            if fh is None:
                return "❌ FINNHUB_KEY not set in .env"
            loop = asyncio.get_event_loop()
            def _call():
                snap = fh.bundle(symbol)
                return fh.compact_snapshot(snap)
            d = await loop.run_in_executor(None, _call)
            lines = [f"**🔬 Intel · {symbol}**", ""]
            # Analyst
            a = d.get("analyst") or {}
            if a.get("target_mean"):
                lines.append(
                    f"**📊 Analyst:** target ${a.get('target_mean', 0):.2f} "
                    f"(high ${a.get('target_high', 0):.2f} · "
                    f"low ${a.get('target_low', 0):.2f}) · "
                    f"{a.get('n_analysts', '?')} analysts"
                )
            r = d.get("recommendations") or {}
            if r:
                lines.append(
                    f"**📢 Ratings ({r.get('period','')}):** "
                    f"{r.get('strong_buy',0)} Strong Buy · "
                    f"{r.get('buy',0)} Buy · "
                    f"{r.get('hold',0)} Hold · "
                    f"{r.get('sell',0)} Sell · "
                    f"{r.get('strong_sell',0)} Strong Sell"
                )
            # Fundamentals
            f = d.get("fundamentals") or {}
            if f:
                bits = []
                if f.get("pe_ttm"):        bits.append(f"P/E {f['pe_ttm']:.1f}")
                if f.get("market_cap"):    bits.append(f"MCap ${f['market_cap']/1000:.1f}B")
                if f.get("beta"):          bits.append(f"β {f['beta']:.2f}")
                if f.get("profit_margin"): bits.append(f"Margin {f['profit_margin']:.1f}%")
                if f.get("52w_high"):
                    bits.append(f"52w ${f.get('52w_low', 0):.0f}-${f['52w_high']:.0f}")
                if bits:
                    lines.append("**📈 Fundamentals:** " + " · ".join(bits))
            # Insider
            ins = d.get("insider_90d") or {}
            if ins.get("n_tx"):
                net = ins.get("net_shares", 0)
                lines.append(
                    f"**👔 Insider 90d:** {ins.get('buys', 0)} buys · "
                    f"{ins.get('sells', 0)} sells · "
                    f"net {'+' if net >= 0 else ''}{net:,.0f} shares"
                )
            ism = d.get("insider_sentiment") or {}
            if ism.get("mspr") is not None:
                mspr = ism["mspr"]
                icon = "🟢" if mspr > 20 else "🔴" if mspr < -20 else "⚪"
                lines.append(
                    f"**{icon} Insider sentiment (MSPR):** {mspr:.1f} "
                    f"(−100 bearish · +100 bullish)"
                )
            # Top holders
            th = d.get("top_holders") or []
            if th:
                bits = [f"{h.get('name', '?')[:22]} ({h.get('share', 0)*100:.1f}%)"
                         for h in th[:3]]
                lines.append(f"**🏦 Top holders:** " + " · ".join(bits))
            io = d.get("institutional_ownership") or {}
            if io.get("pct_held"):
                lines.append(
                    f"**🏛 Institutional:** {io['pct_held']*100:.1f}% held · "
                    f"{io.get('n_holders', '?')} institutions"
                )
            # Recent filings
            fl = d.get("recent_filings") or []
            if fl:
                lines.append("**📄 Recent filings:**")
                for f in fl[:3]:
                    lines.append(
                        f"  · {f.get('form', '?')} @ {f.get('filed_at', '?')}"
                    )
            # Dividend
            div = d.get("next_dividend") or {}
            if div.get("ex_date"):
                lines.append(
                    f"**💵 Next dividend:** ${div.get('amount', 0):.2f} "
                    f"ex-date {div['ex_date']}"
                )
            # Revenue segments
            seg = d.get("revenue_segments") or []
            if seg:
                bits = [f"{s.get('segment', '?')[:16]} (${s.get('revenue', 0)/1000:.1f}B)"
                         for s in seg[:4]]
                lines.append("**🥧 Revenue:** " + " · ".join(bits))
            if d.get("_errors"):
                lines.append(f"\n_⚠️ some endpoints failed: "
                             f"{', '.join(d['_errors'][:3])}_")
            return "\n".join(lines)[:1900]
        except Exception as e:
            return f"intel failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_calendars(self) -> str:
        """Upcoming IPO + FDA advisory calendars — market-moving events."""
        try:
            from src.intelligence.finnhub_intelligence import (
                build_finnhub_intelligence,
            )
            fh = build_finnhub_intelligence()
            if fh is None:
                return "❌ FINNHUB_KEY not set"
            loop = asyncio.get_event_loop()
            def _call():
                return fh.ipo_calendar() or [], fh.fda_calendar() or []
            ipos, fdas = await loop.run_in_executor(None, _call)
            lines = ["**📅 Upcoming calendars**", ""]
            lines.append(f"**💼 IPOs ({len(ipos)} upcoming):**")
            for i in ipos[:10]:
                lines.append(
                    f"  · {i.get('date', '?')} · "
                    f"**{i.get('symbol', '?')}** · "
                    f"{i.get('name', '?')[:36]} · "
                    f"{i.get('exchange', '?')}"
                )
            lines.append("")
            lines.append(f"**💊 FDA advisory meetings ({len(fdas)} upcoming):**")
            for f in fdas[:10]:
                lines.append(
                    f"  · {f.get('meetingDate', '?')} · "
                    f"**{f.get('committee', '?')[:30]}** · "
                    f"{f.get('productName', '?')[:40]}"
                )
            return "\n".join(lines)[:1900]
        except Exception as e:
            return f"calendars failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_cleanup(self, do_close: bool) -> str:
        """List positions across local + Tradier + Alpaca, flag any
        that exist on one side but not the other (orphans). With
        --close, submit close orders on every orphan via Tradier
        (since Alpaca paper rejects uncovered-options closes)."""
        try:
            import json as _json
            from pathlib import Path as _P
            from src.brokers.tradier_adapter import build_tradier_broker
            from src.core.types import Order, Side
            lines = ["**🧹 Broker cleanup — orphan position scan**", ""]

            # Local state
            snap = ROOT / "logs" / "broker_state.json"
            local_syms = set()
            if snap.exists():
                data = _json.loads(snap.read_text())
                local_syms = {p.get("symbol", "")
                              for p in (data.get("positions") or [])}
                lines.append(f"**Local (paper):** {len(local_syms)} positions")
                for s in sorted(local_syms):
                    lines.append(f"  · {s}")
            else:
                lines.append("**Local:** no snapshot")

            # Tradier
            tradier_syms = set()
            tradier_positions = []
            tb = build_tradier_broker()
            if tb is not None:
                try:
                    tradier_positions = list(tb.positions())
                    tradier_syms = {p.symbol for p in tradier_positions}
                    lines.append("")
                    lines.append(f"**Tradier:** {len(tradier_syms)} positions")
                    for p in tradier_positions:
                        lines.append(
                            f"  · {p.symbol}  qty={p.qty}  "
                            f"avg=${p.avg_price:.2f}"
                        )
                except Exception as e:
                    lines.append(f"**Tradier:** error: {e}")
            else:
                lines.append("**Tradier:** not configured")

            # Orphans
            orphan_tradier = tradier_syms - local_syms
            orphan_local = local_syms - tradier_syms

            lines.append("")
            if orphan_tradier:
                lines.append(f"**🚨 Orphan on Tradier (not in local):** "
                              f"{len(orphan_tradier)}")
                for s in sorted(orphan_tradier):
                    lines.append(f"  · {s}")
            if orphan_local:
                lines.append(f"**⚠️ Orphan local (not on Tradier):** "
                              f"{len(orphan_local)}")
                for s in sorted(orphan_local):
                    lines.append(f"  · {s}")
            if not orphan_tradier and not orphan_local:
                lines.append("✅ All brokers in sync — no orphans.")

            # Optional close
            if do_close and orphan_tradier and tb is not None:
                lines.append("")
                lines.append("**🔨 Closing orphans on Tradier...**")
                for pos in tradier_positions:
                    if pos.symbol not in orphan_tradier:
                        continue
                    side = Side.SELL if pos.qty > 0 else Side.BUY
                    o = Order(
                        symbol=pos.symbol, side=side, qty=abs(pos.qty),
                        is_option=pos.is_option,
                        limit_price=max(0.01, pos.avg_price * 0.50),
                        tif="DAY", tag="discord_cleanup",
                    )
                    try:
                        tb.submit(o)
                        lines.append(f"  ✅ closed {pos.symbol} "
                                      f"qty={pos.qty}")
                    except Exception as e:
                        lines.append(f"  ❌ {pos.symbol} failed: "
                                      f"{str(e)[:120]}")

            return "\n".join(lines)[:1900]
        except Exception as e:
            return f"cleanup failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_saves(self, hours: int) -> str:
        """On-demand saves report. Refreshes re-checks + returns the
        formatted summary without posting to Discord again (the reply
        goes to the channel where the command was issued)."""
        try:
            from src.data.multi_provider import MultiProvider
            from src.intelligence.saves_tracker import (
                recheck_pending, summary,
            )
            from scripts.run_saves_report import _format_discord
            loop = asyncio.get_event_loop()
            def _call():
                mp = MultiProvider.from_env()
                recheck_pending(mp)
                return _format_discord(summary(since_hours=hours))
            return await loop.run_in_executor(None, _call)
        except Exception as e:
            return f"saves report failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_catalyst(self, symbols: List[str]) -> str:
        """Run the catalyst deep-dive aggregator in a worker thread."""
        try:
            from src.data.multi_provider import MultiProvider
            from scripts.run_catalyst_dive import (
                _gather_snapshot, _PROMPT_TEMPLATE, _call_llm,
                _parse_llm_json, _format_discord,
            )
            import time as _t
            mp = MultiProvider.from_env()
            loop = asyncio.get_event_loop()
            def _call():
                snap = _gather_snapshot(mp, symbols)
                prompt = _PROMPT_TEMPLATE.format(
                    snapshot=json.dumps(snap, indent=2, default=str)[:14000],
                )
                t0 = _t.time()
                raw, model = _call_llm(prompt, timeout_sec=300.0,
                                         max_tokens=800)
                parsed = _parse_llm_json(raw)
                if not parsed:
                    return (f"**📰 Catalyst dive** — no structured output "
                            f"(model={model or 'n/a'})")
                return _format_discord(parsed,
                                         model=model or "n/a",
                                         latency=_t.time() - t0)
            out = await loop.run_in_executor(None, _call)
            return out
        except Exception as e:                          # noqa: BLE001
            return f"catalyst failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_research(self, symbols: List[str]) -> str:
        """Run the options research agent in a worker thread (70B is
        slow) and return its formatted markdown."""
        try:
            from src.data.multi_provider import MultiProvider
            from src.intelligence.options_research import OptionsResearchAgent
            mp = MultiProvider.from_env()
            agent = OptionsResearchAgent(mp)
            loop = asyncio.get_event_loop()
            def _call():
                rep = agent.run(symbols)
                return agent.to_markdown(rep)
            out = await loop.run_in_executor(None, _call)
            return out
        except Exception as e:                          # noqa: BLE001
            return f"research failed: {type(e).__name__}: {str(e)[:200]}"

    async def _handle_summary(self, window_minutes: int) -> str:
        """Build an in-session digest of the last N minutes of
        tradebot.out. Runs the parser in-process — no subprocess — so
        users in Discord get an answer in ~1s."""
        try:
            from src.reports.log_digest import build_digest
            log_path = ROOT / "logs" / "tradebot.out"
            d = build_digest(log_path, window_minutes=window_minutes)
            body = d.to_markdown()
            return body
        except Exception as e:
            return f"summary failed: {type(e).__name__}: {str(e)[:200]}"

    async def _tail_logs(self, n: int) -> str:
        """Read last N lines of logs/tradebot.out."""
        log_path = ROOT / "logs" / "tradebot.out"
        if not log_path.exists():
            return "logs/tradebot.out is empty or missing."
        try:
            # Tail without pulling the whole file into memory.
            with log_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                chunk = min(size, 64 * 1024)
                f.seek(size - chunk)
                data = f.read().decode("utf-8", errors="replace")
            lines = data.splitlines()[-n:]
            return f"```\n{_truncate(chr(10).join(lines))}\n```"
        except Exception as e:
            return f"logs read failed: {e}"

    async def _positions_print(self) -> str:
        """Read logs/broker_state.json and format the open positions."""
        snap = ROOT / "logs" / "broker_state.json"
        if not snap.exists():
            return "No broker snapshot yet."
        try:
            data = json.loads(snap.read_text())
        except Exception as e:
            return f"snapshot read failed: {e}"
        positions = data.get("positions", [])
        if not positions:
            return "No open positions."
        lines = ["**Open positions:**"]
        for p in positions:
            lines.append(
                f"  {p.get('symbol','?')} qty={p.get('qty','?')} "
                f"avg=${p.get('avg_price',0):.2f} "
                f"PT=${p.get('auto_profit_target',0):.2f} "
                f"SL=${p.get('auto_stop_loss',0):.2f}"
            )
        lines.append(f"cash=${data.get('cash',0):,.2f}  "
                      f"day_pnl=${data.get('day_pnl',0):+,.2f}")
        return "\n".join(lines)

    async def _sh(self, argv: List[str], *, cwd: str,
                    timeout: float) -> Tuple[int, str, str]:
        """Async subprocess wrapper. Returns (rc, stdout, stderr)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(),
                                                    timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return (124, "", f"timeout after {timeout}s")
            return (
                proc.returncode if proc.returncode is not None else 0,
                (out or b"").decode("utf-8", errors="replace"),
                (err or b"").decode("utf-8", errors="replace"),
            )
        except Exception as e:
            return (1, "", str(e))


# --------------------------------------------------------------- discord


def _build_chat_context():
    """Pull a compact bot-state snapshot from the journal + broker snap.
    Used by both the startup hello and the free-form chat. Fail-open —
    if anything is missing, return a ChatContext with whatever we have.
    """
    from datetime import datetime, timezone
    try:
        from src.intelligence.llm_chat import ChatContext
    except Exception:
        return None
    ctx = ChatContext(now_iso=datetime.now(tz=timezone.utc).isoformat())
    # broker_state.json → positions + day_pnl
    snap_path = ROOT / "logs" / "broker_state.json"
    if snap_path.exists():
        try:
            data = json.loads(snap_path.read_text())
            ctx.open_positions = len(data.get("positions", []) or [])
            ctx.positions_summary = [
                f"{p.get('symbol','?')} qty={p.get('qty','?')}"
                f" avg=${p.get('avg_price',0):.2f}"
                for p in (data.get("positions") or [])[:6]
            ]
            ctx.day_pnl_usd = float(data.get("day_pnl", 0.0))
        except Exception:
            pass
    # settings.yaml → universe + live_trading flag
    try:
        from src.core.config import load_settings
        s = load_settings(ROOT / "config" / "settings.yaml")
        uni = s.raw.get("universe") if hasattr(s, "raw") else None
        if isinstance(uni, list):
            ctx.universe = [str(x) for x in uni][:6]
        elif isinstance(uni, dict):
            ctx.universe = [str(x) for x in (uni.get("symbols") or [])[:6]]
        ctx.live_trading = bool(
            (s.raw.get("execution", {}) or {}).get("live_trading", False)
        )
    except Exception:
        pass
    # latest strategy-audit summary
    try:
        from src.intelligence.strategy_auditor import read_recent_audits
        recent = read_recent_audits(1)
        if recent:
            ctx.last_audit_health = recent[0].get("overall_health")
            ctx.last_audit_summary = recent[0].get("summary")
    except Exception:
        pass

    # Session awareness: tell the LLM whether market is open right now
    # so it frames analysis correctly (overnight vs live).
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        now_utc = _dt.now(tz=_tz.utc)
        # US Eastern = UTC-5 (standard) or UTC-4 (DST). Approximate with
        # UTC-4 -- we're in DST for most of the year, fine for "is open".
        now_et = now_utc - _td(hours=4)
        weekday = now_et.weekday()                   # Mon=0 ... Sun=6
        is_weekday = weekday < 5
        tmin = now_et.hour * 60 + now_et.minute
        open_min = 9 * 60 + 30
        close_min = 16 * 60
        ctx.market_is_open = (is_weekday
                                and open_min <= tmin <= close_min)
        # Hours until next open (for Friday close / weekend questions)
        if not ctx.market_is_open:
            tgt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            if tmin > close_min or not is_weekday:
                while True:
                    tgt = tgt + _td(days=1)
                    if tgt.weekday() < 5:
                        break
            elif tmin < open_min:
                pass                                   # same-day later
            ctx.hours_until_open = round(
                (tgt - now_et).total_seconds() / 3600.0, 1,
            )
    except Exception:
        pass

    # Political-news pulse so chat can reference Fed / WH / Truth / WSB.
    try:
        from src.core.config import load_settings
        from src.intelligence.political_news import build_political_news_provider
        s = load_settings(ROOT / "config" / "settings.yaml")
        pol = build_political_news_provider(s)
        if pol is not None:
            snap = pol.snapshot_for_auditor()
            items = (snap.get("headlines", []) if isinstance(snap, dict)
                     else snap if isinstance(snap, list) else [])
            ctx.political_headlines = [
                {"ts": x.get("ts") or x.get("published", ""),
                  "source": x.get("source", "?"),
                  "headline": (x.get("headline") or x.get("title") or "")[:200],
                  "summary": (x.get("summary") or "")[:200]}
                for x in items[:15]
                if (x.get("headline") or x.get("title"))
            ]
    except Exception as _e:
        _log.info("political_headlines_enrich_failed err=%s", _e)

    # Actively pull LIVE quotes from the multi-provider stack. This is
    # the critical fix: without this the LLM sees an empty spot_prices
    # object and answers "I don't have prices." Cached at the provider
    # layer so repeated chat messages don't hammer APIs.
    try:
        from src.data.multi_provider import MultiProvider
        mp = MultiProvider.from_env()
        if mp.active_providers():
            live_spots: Dict[str, float] = {}
            for sym in (ctx.universe or ["SPY", "QQQ"])[:4]:
                q = mp.latest_quote(sym)
                if q and q.mid > 0:
                    live_spots[sym] = round(float(q.mid), 2)
            if live_spots:
                # Merge with anything we already scraped from the log
                # (live overrides stale).
                merged = dict(ctx.spot_by_symbol or {})
                merged.update(live_spots)
                ctx.spot_by_symbol = merged
            # Also pull sentiment for the universe so "what's the read
            # on SPY?" questions get weighted correctly.
            for sym in (ctx.universe or ["SPY", "QQQ"])[:2]:
                s = mp.news_sentiment(sym)
                if s is not None:
                    ctx.recent_signals.append(
                        f"sentiment[{sym}]={s:+.2f}"
                    )
            # Live VIX — when market is closed the paper-bot tick loop
            # hasn't logged a regime_snapshot yet, so ctx.vix is None.
            # Grab it directly from Yahoo's ^VIX ticker as a fallback.
            if ctx.vix is None:
                try:
                    for p in getattr(mp, "_providers", []):
                        fn = getattr(p, "latest_vix", None)
                        if fn is None:
                            continue
                        v = fn()
                        if v is not None and v > 0:
                            ctx.vix = float(v)
                            break
                except Exception:
                    pass
    except Exception as _e:
        _log.info("live_quote_enrich_failed err=%s", _e)

    # tradebot.out grep for the most recent regime, VIX, breadth, spot
    # prices, and a few recent signal events. Lets the LLM see live
    # state without a database query or locking the journal.
    try:
        import re as _re
        log_path = ROOT / "logs" / "tradebot.out"
        if log_path.exists():
            import os as _os
            size = log_path.stat().st_size
            read_from = max(0, size - 300_000)        # last ~300 KB
            with log_path.open("rb") as f:
                f.seek(read_from)
                if read_from > 0:
                    f.readline()
                tail = f.read().decode("utf-8", errors="replace")

            # Latest regime: scan backwards for 'regime=<word>'
            for m in _re.finditer(r"regime=(\w+)", tail):
                ctx.regime = m.group(1)
            # Latest VIX
            m = None
            for m in _re.finditer(r"\bvix=([0-9.]+)", tail):
                pass
            if m is not None:
                try:
                    ctx.vix = float(m.group(1))
                except Exception:
                    pass
            # Latest breadth score
            for m in _re.finditer(r"breadth_score=(-?[0-9.]+)", tail):
                try:
                    ctx.breadth_score = float(m.group(1))
                except Exception:
                    pass
            # Latest spot price per symbol (last-seen wins)
            spot: Dict[str, float] = {}
            for m in _re.finditer(r"\bsymbol=([A-Z]{1,5})[^\n]*?(?:spot|price|last)=([0-9.]+)", tail):
                try:
                    spot[m.group(1)] = float(m.group(2))
                except Exception:
                    pass
            if spot:
                ctx.spot_by_symbol = spot
            # Recent signals (last 8 ensemble events for context)
            recent_sigs: List[str] = []
            for line in tail.splitlines()[-400:]:
                if "ensemble_skip" in line or "exec_chain_pass" in line:
                    # keep it short; strip ANSI / structlog brackets
                    clean = _re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
                    clean = _re.sub(r"\[[a-z]+\s*\]\s*", "", clean)
                    recent_sigs.append(clean[:160])
            if recent_sigs:
                ctx.recent_signals = recent_sigs[-8:]
    except Exception:
        pass

    return ctx


def _enrich_context_with_options(ctx, question: str) -> None:
    """When the question mentions options / strikes / news, pull a
    compact snapshot of the live chain + top headlines for the
    underlyings the user's universe covers (or SPY by default). Mutates
    ctx in place. Fail-open — any exception leaves ctx untouched.
    """
    try:
        from src.data.multi_provider import MultiProvider
        mp = MultiProvider.from_env()
        if not mp.active_providers():
            return
        # Which symbols? Prefer ones mentioned in the question, else fall
        # back to the universe.
        mentioned: List[str] = []
        for sym in ("SPY", "QQQ", "IWM", "DIA"):
            if sym.lower() in (question or "").lower():
                mentioned.append(sym)
        targets = mentioned or (ctx.universe[:2] if ctx.universe else ["SPY"])
        atm_rows: List[dict] = []
        headlines: List[dict] = []
        for sym in targets[:2]:                              # cap 2 symbols
            chain = mp.option_chain(sym)
            if chain:
                # Use the same near-ATM slicer the research agent uses.
                try:
                    from src.intelligence.options_research import (
                        _best_contracts_near_atm,
                    )
                    spot = ctx.spot_by_symbol.get(sym) or 0.0
                    if spot > 0:
                        atm_rows.extend(
                            _best_contracts_near_atm(chain, spot, n_each_side=3)
                        )
                except Exception:
                    pass
            for n in (mp.news(sym, limit=5) or [])[:5]:
                headlines.append({
                    "ts": n.ts, "symbol": sym,
                    "headline": n.headline,
                    "source": n.source,
                })
        ctx.option_chain_atm = atm_rows[:20]
        ctx.news_headlines = headlines[:10]
    except Exception as e:                                  # noqa: BLE001
        _log.info("options_context_enrich_failed err=%s", e)


def _build_app():
    """Construct the discord client + command router. Lazy-imports
    discord so `import discord_terminal` doesn't require the lib to be
    installed (useful for tests)."""
    import discord
    from discord import app_commands   # noqa: F401

    # LLM chat — built once, reused across messages. Off unless
    # LLM_CHAT_ENABLED=1 in .env.
    try:
        from src.intelligence.llm_chat import build_llm_chat_from_env
        chat = build_llm_chat_from_env()
    except Exception as e:
        _log.warning("llm_chat_init_failed err=%s", e)
        chat = None

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    runner = CommandRunner()

    def _pick_chat_model_for_channel(channel_id: int) -> Optional[str]:
        """70B for designated channels, None (= default 8B) otherwise."""
        if channel_id in CHAT_70B_CHANNEL_IDS:
            return CHAT_70B_MODEL
        return None

    async def _handle_chat_question(message, question: str) -> None:
        """Route a free-form question through LLMChat and post the
        answer. Uses the 70B model if this channel is in the 70B set."""
        if chat is None or not chat.cfg.enabled:
            return
        model_override = _pick_chat_model_for_channel(message.channel.id)
        is_70b = model_override is not None
        answer = ""
        # Long-running 70B calls: show typing so the user sees progress.
        # asyncio.to_thread() is 3.9+; fall back to run_in_executor on 3.8
        # (Jetson's default). Both dispatch the blocking LLM call to a
        # thread pool so the Discord heartbeat keeps ticking.
        try:
            async with message.channel.typing():
                ctx = _build_chat_context()
                # Enrich context with live options chain + news when the
                # question sounds options-related. Saves a few hundred
                # ms on "hi what's up" style questions that don't need it.
                try:
                    from src.intelligence.llm_chat import question_wants_options_context
                    if question_wants_options_context(question):
                        _enrich_context_with_options(ctx, question)
                except Exception as _e:
                    _log.info("chat_context_enrich_failed err=%s", _e)
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None,
                    lambda: chat.answer(
                        question, ctx,
                        user_id=message.author.id,
                        model_override=model_override,
                        max_tokens_override=(CHAT_70B_MAX_TOKENS if is_70b else None),
                    ),
                )
        except Exception as e:
            _log.warning("llm_chat_handler_failed err=%s", e, exc_info=True)
            answer = f"_chat failed — {type(e).__name__}: {str(e)[:200]}_"
        tag = f" · model={model_override}" if model_override else ""
        await message.channel.send(_truncate(answer) + tag)
        _audit({"kind": "chat",
                "user": message.author.id,
                "channel": message.channel.id,
                "model": model_override or chat.cfg.model_name,
                "q_len": len(question), "a_len": len(answer)})

    # ------------------------- button panels -------------------------

    class QuickQuestionsView(discord.ui.View):
        """Six quick-question buttons for #8b-llm / #70b-llm / #tradebot-chat.

        Clicking a button sends a pre-baked question through the SAME
        LLM chat pipeline (including live options chain + news
        enrichment when relevant). The channel's 70B/8B routing rules
        apply — you get the right model per channel."""
        def __init__(self):
            super().__init__(timeout=None)

        async def _ask(self, interaction, question: str):
            # Guard against double-ack — Discord rejects with
            # "Interaction already acknowledged" if we defer twice
            # (can happen when the button is double-clicked).
            if not interaction.response.is_done():
                await interaction.response.defer(thinking=True)
            try:
                if chat is None or not chat.cfg.enabled:
                    await interaction.followup.send(
                        "_chat is disabled — set `LLM_CHAT_ENABLED=1` in .env._",
                        ephemeral=True,
                    )
                    return
                model_override = _pick_chat_model_for_channel(interaction.channel.id)
                is_70b = model_override is not None
                ctx = _build_chat_context()
                try:
                    from src.intelligence.llm_chat import question_wants_options_context
                    if question_wants_options_context(question):
                        _enrich_context_with_options(ctx, question)
                except Exception:
                    pass
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(
                    None,
                    lambda: chat.answer(
                        question, ctx,
                        user_id=interaction.user.id,
                        model_override=model_override,
                        max_tokens_override=(CHAT_70B_MAX_TOKENS if is_70b else None),
                    ),
                )
                tag = f" · model={model_override}" if model_override else ""
                await interaction.followup.send(_truncate(answer) + tag)
            except Exception as e:                      # noqa: BLE001
                await interaction.followup.send(
                    f"_chat failed — {type(e).__name__}: {str(e)[:150]}_",
                    ephemeral=True,
                )

        @discord.ui.button(label="📊 Market read",
                            style=discord.ButtonStyle.primary,
                            custom_id="qq:market_read", row=0)
        async def btn_market_read(self, interaction, _):
            await self._ask(interaction,
                "Give a concise market read right now — regime, VIX, breadth, what SPY and QQQ are doing.")

        @discord.ui.button(label="🔍 Whats SPY doing?",
                            style=discord.ButtonStyle.primary,
                            custom_id="qq:spy", row=0)
        async def btn_spy(self, interaction, _):
            await self._ask(interaction,
                "What is SPY doing right now? Price, trend, key levels, any setup forming?")

        @discord.ui.button(label="🔍 Whats QQQ doing?",
                            style=discord.ButtonStyle.primary,
                            custom_id="qq:qqq", row=0)
        async def btn_qqq(self, interaction, _):
            await self._ask(interaction,
                "What is QQQ doing right now? Price, trend, key levels, any setup forming?")

        @discord.ui.button(label="💡 Top trade idea",
                            style=discord.ButtonStyle.success,
                            custom_id="qq:top_idea", row=1)
        async def btn_top_idea(self, interaction, _):
            await self._ask(interaction,
                "Given current conditions, what is the single best options trade idea right now? Specific strike, expiry, direction, and reason.")

        @discord.ui.button(label="📰 News pulse",
                            style=discord.ButtonStyle.secondary,
                            custom_id="qq:news", row=1)
        async def btn_news(self, interaction, _):
            await self._ask(interaction,
                "What market news in the last 24h is most relevant to SPY / QQQ right now?")

        @discord.ui.button(label="🧠 Bot health",
                            style=discord.ButtonStyle.secondary,
                            custom_id="qq:health", row=1)
        async def btn_health(self, interaction, _):
            await self._ask(interaction,
                "How is the trading bot doing? Positions, recent signals, any issues, last audit.")

    class AutotradePanel(discord.ui.View):
        """Persistent autotrade control panel. One per channel; user
        runs `!autopanel` in the channel they want to control from.

        Buttons adapt to state:
          - Shows 'Enable' when killed, 'Disable' when active
          - Shows current trade count / cap in status button label
          - Research Now button triggers immediate 70B run
          - View Queue button shows pending ideas
        """
        def __init__(self):
            super().__init__(timeout=None)                # persistent

        async def _status_text(self) -> str:
            from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
            import os as _os
            q = LLMAutotradeQueue()
            s = q.peek_state()
            env_on = _os.getenv("LLM_AUTOTRADE", "").strip() in ("1", "true", "yes")
            state_line = (
                "🟢 **ACTIVE** (executing LLM-originated trades)"
                if env_on and not s["killed"] else
                "🔴 **KILLED** (kill switch ON)" if s["killed"] else
                "⚪ **ENV DISABLED** (LLM_AUTOTRADE not set in .env)"
            )
            return (
                f"**LLM Autotrade Panel**\n{state_line}\n"
                f"· today: {s['daily_count']}/{s['daily_cap']} trades · "
                f"queue: {s['queue_fresh']} fresh ideas · "
                f"max age: {s['max_age_min']} min · "
                f"confidence: {', '.join(s['allowed_confidences'])}"
            )

        async def _authorized(self, interaction) -> bool:
            if not runner.is_authorized(interaction.user.id):
                await interaction.response.send_message(
                    "Not authorized.", ephemeral=True)
                _audit({"kind": "autopanel_rejected_auth",
                        "user": interaction.user.id})
                return False
            return True

        @discord.ui.button(label="✅ Enable", style=discord.ButtonStyle.success,
                            custom_id="ta:enable", row=0)
        async def btn_enable(self, interaction, _):
            if not await self._authorized(interaction):
                return
            await interaction.response.defer(thinking=True, ephemeral=True)
            try:
                from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
                LLMAutotradeQueue().set_killed(False)
                _audit({"kind": "autopanel_enable", "user": interaction.user.id})
                await interaction.followup.send(
                    "✅ Kill switch CLEARED. LLM autotrade will execute "
                    "fresh ideas (if `LLM_AUTOTRADE=1` is set in `.env`).",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.followup.send(
                    f"enable failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="⛔ Disable", style=discord.ButtonStyle.danger,
                            custom_id="ta:disable", row=0)
        async def btn_disable(self, interaction, _):
            if not await self._authorized(interaction):
                return
            await interaction.response.defer(thinking=True, ephemeral=True)
            try:
                from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
                LLMAutotradeQueue().set_killed(True)
                _audit({"kind": "autopanel_disable", "user": interaction.user.id})
                await interaction.followup.send(
                    "⛔ KILLED. No more LLM-originated trades until you "
                    "press **Enable** again.", ephemeral=True,
                )
            except Exception as e:
                await interaction.followup.send(
                    f"disable failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="📊 Status", style=discord.ButtonStyle.primary,
                            custom_id="ta:status", row=0)
        async def btn_status(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                text = await self._status_text()
            except Exception as e:
                text = f"status failed: {e}"
            await interaction.response.send_message(text, ephemeral=True)

        @discord.ui.button(label="🔬 Research Now", style=discord.ButtonStyle.primary,
                            custom_id="ta:research", row=1)
        async def btn_research(self, interaction, _):
            if not await self._authorized(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                from src.data.multi_provider import MultiProvider
                from src.intelligence.options_research import OptionsResearchAgent
                mp = MultiProvider.from_env()
                agent = OptionsResearchAgent(mp)
                loop = asyncio.get_event_loop()
                def _call():
                    return agent.to_markdown(agent.run(["SPY", "QQQ"]))
                out = await loop.run_in_executor(None, _call)
                await interaction.followup.send(_truncate(out))
            except Exception as e:
                await interaction.followup.send(
                    f"research failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="📋 View Queue", style=discord.ButtonStyle.secondary,
                            custom_id="ta:queue", row=1)
        async def btn_queue(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
                q = LLMAutotradeQueue()
                q._maybe_refresh()
                ideas = [
                    i for i in q._cached_rows
                    if i.id not in q._consumed and q._is_fresh(i)
                ]
                if not ideas:
                    await interaction.response.send_message(
                        "Queue empty — no fresh ideas. Press 🔬 Research Now.",
                        ephemeral=True,
                    )
                    return
                lines = [f"**{len(ideas)} fresh ideas in queue:**"]
                for i in ideas[:6]:
                    lines.append(
                        f"· {i.symbol} {i.direction.upper()} "
                        f"${i.strike or '?'} {i.expiry or '?'} "
                        f"({i.confidence}) — {i.rationale[:80]}"
                    )
                await interaction.response.send_message(
                    "\n".join(lines), ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"queue view failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="🔄 Refresh Status", style=discord.ButtonStyle.secondary,
                            custom_id="ta:refresh", row=1)
        async def btn_refresh(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                text = await self._status_text()
                # Edit the original panel message so the top line
                # reflects current state without posting a new one.
                await interaction.response.edit_message(content=text, view=self)
            except Exception as e:
                await interaction.response.send_message(
                    f"refresh failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="⬆⬆ Cap +5", style=discord.ButtonStyle.secondary,
                            custom_id="ta:cap_up5", row=2)
        async def btn_cap_up5(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
                q = LLMAutotradeQueue()
                new_cap = q.current_cap() + 5
                applied = q.set_cap_override(new_cap)
                _audit({"kind": "autopanel_cap_up5", "user": interaction.user.id,
                        "new_cap": applied})
                await interaction.response.send_message(
                    f"⬆⬆ Cap raised to **{applied}** trades/day.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"cap up+5 failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="⬇ Reset Cap", style=discord.ButtonStyle.secondary,
                            custom_id="ta:cap_reset", row=2)
        async def btn_cap_reset(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
                q = LLMAutotradeQueue()
                applied = q.set_cap_override(None)
                _audit({"kind": "autopanel_cap_reset", "user": interaction.user.id,
                        "new_cap": applied})
                await interaction.response.send_message(
                    f"⬇ Cap reset to config default: **{applied}** trades/day.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"cap reset failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="♻ Reset Daily Counter",
                            style=discord.ButtonStyle.danger,
                            custom_id="ta:counter_reset", row=2)
        async def btn_counter_reset(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.intelligence.llm_autotrade_queue import LLMAutotradeQueue
                q = LLMAutotradeQueue()
                q.reset_daily_counter()
                _audit({"kind": "autopanel_counter_reset",
                        "user": interaction.user.id})
                await interaction.response.send_message(
                    "♻ Daily trade counter reset to 0.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"counter reset failed: {e}", ephemeral=True,
                )

        # ---- Row 3: 0DTE cap controls ----------------------------

        def _current_zdte_cap(self) -> int:
            try:
                from src.core.runtime_overrides import get_override
                from src.core.settings import load_settings
                s = load_settings()
                return int(get_override(
                    "max_0dte_per_day",
                    s["execution"]["max_0dte_per_day"],
                ))
            except Exception:
                return 20

        @discord.ui.button(label="🎯🎯 0DTE +5",
                            style=discord.ButtonStyle.secondary,
                            custom_id="ta:zdte_up5", row=3)
        async def btn_zdte_up5(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                new_cap = self._current_zdte_cap() + 5
                set_override("max_0dte_per_day", new_cap)
                _audit({"kind": "zdte_cap_up5", "user": interaction.user.id,
                        "new_cap": new_cap})
                await interaction.response.send_message(
                    f"🎯🎯 0DTE cap raised to **{new_cap}** trades/day.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"0DTE +5 failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="🎯 0DTE=20",
                            style=discord.ButtonStyle.primary,
                            custom_id="ta:zdte_set20", row=3)
        async def btn_zdte_set20(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("max_0dte_per_day", 20)
                _audit({"kind": "zdte_cap_set", "user": interaction.user.id,
                        "new_cap": 20})
                await interaction.response.send_message(
                    "🎯 0DTE cap set to **20** trades/day (operator default).",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"0DTE set failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="🔄 0DTE Reset",
                            style=discord.ButtonStyle.secondary,
                            custom_id="ta:zdte_reset", row=3)
        async def btn_zdte_reset(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("max_0dte_per_day", None)
                _audit({"kind": "zdte_cap_reset", "user": interaction.user.id})
                await interaction.response.send_message(
                    "🔄 0DTE cap reset to config default.", ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"0DTE reset failed: {e}", ephemeral=True,
                )

        # ---- Row 4: smart-bid pricing controls -------------------

        @discord.ui.button(label="💰 Smarter Bids",
                            style=discord.ButtonStyle.primary,
                            custom_id="ta:bid_tighten", row=4)
        async def btn_bid_tighten(self, interaction, _):
            """Lower entry_spread_pct — sit closer to bid (pay less,
            fill less often). Good when you want quality over volume."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("entry_spread_pct", 0.10)
                _audit({"kind": "bid_tighten", "user": interaction.user.id})
                await interaction.response.send_message(
                    "💰 Entry now at **bid + 10% of spread** — tighter "
                    "fills, less slippage, may miss some trades.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"bid tighten failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="⚖ Mid Price",
                            style=discord.ButtonStyle.secondary,
                            custom_id="ta:bid_mid", row=4)
        async def btn_bid_mid(self, interaction, _):
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("entry_spread_pct", 0.50)
                _audit({"kind": "bid_mid", "user": interaction.user.id})
                await interaction.response.send_message(
                    "⚖ Entry now at **mid** (bid + 50% spread) — "
                    "balance of fill rate vs cost.", ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"bid mid failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="🚀 Chase",
                            style=discord.ButtonStyle.danger,
                            custom_id="ta:bid_chase", row=4)
        async def btn_bid_chase(self, interaction, _):
            """Pay ask * 0.95 — use only when you MUST get filled
            (closing a position, catalyst breaking). Bleeds spread."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("entry_spread_pct", 0.90)
                _audit({"kind": "bid_chase", "user": interaction.user.id})
                await interaction.response.send_message(
                    "🚀 Entry now at **bid + 90% spread** — aggressive "
                    "taker. Use sparingly.", ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"bid chase failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="💵 Cheap Only",
                            style=discord.ButtonStyle.primary,
                            custom_id="ta:premium_cap_low", row=4)
        async def btn_premium_cap_low(self, interaction, _):
            """Cap max premium at $2/contract — force the bot to pick
            cheaper, farther-OTM strikes. Protects budget, spreads
            risk across more positions."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("max_premium_per_contract_usd", 2.00)
                _audit({"kind": "premium_cap_low",
                        "user": interaction.user.id})
                await interaction.response.send_message(
                    "💵 Max premium per contract → **$2.00**. Bot will "
                    "skip expensive strikes and pick cheaper OTM ones.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"premium cap failed: {e}", ephemeral=True,
                )

        # ---- Row 5: Entry filter tuning (premium + OI + spread) -------

        @discord.ui.button(label="📈 Looser entries",
                            style=discord.ButtonStyle.success,
                            custom_id="ta:entries_loose", row=5)
        async def btn_entries_loose(self, interaction, _):
            """Bumps premium cap ($3→$4), lowers OI floor (100→50),
            widens spread cap (ETF 3%→5%, stock 6%→8%). Use when the
            chain is blocking too many signals as `premium_too_high`
            or `oi_too_low`."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("max_premium_per_contract_usd", 4.00)
                set_override("min_open_interest", 50)
                set_override("min_today_option_volume", 25)
                set_override("max_spread_pct_etf", 0.05)
                set_override("max_spread_pct_stock", 0.08)
                _audit({"kind": "entries_loose",
                        "user": interaction.user.id})
                await interaction.response.send_message(
                    "📈 **Looser entries active**\n"
                    "• Max premium: $3.00 → **$4.00**\n"
                    "• Min OI: 100 → **50**\n"
                    "• Min today volume: 100 → **25**\n"
                    "• Max spread ETF: 3% → **5%**\n"
                    "• Max spread stock: 6% → **8%**\n"
                    "_Takes effect next tick — no restart needed._",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"loosen failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="📊 Balanced",
                            style=discord.ButtonStyle.secondary,
                            custom_id="ta:entries_balanced", row=5)
        async def btn_entries_balanced(self, interaction, _):
            """Reset entry filters to config defaults."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                for k in ("max_premium_per_contract_usd",
                          "min_open_interest",
                          "min_today_option_volume",
                          "max_spread_pct_etf",
                          "max_spread_pct_stock"):
                    set_override(k, None)
                _audit({"kind": "entries_balanced",
                        "user": interaction.user.id})
                await interaction.response.send_message(
                    "📊 **Balanced entries** — all filter overrides cleared, "
                    "config/settings.yaml defaults restored.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"balanced failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="🔒 Tight entries",
                            style=discord.ButtonStyle.danger,
                            custom_id="ta:entries_tight", row=5)
        async def btn_entries_tight(self, interaction, _):
            """Tighten filters — quality over quantity. Only the best
            confluence + most liquid contracts get through."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import set_override
                set_override("max_premium_per_contract_usd", 2.00)
                set_override("min_open_interest", 200)
                set_override("min_today_option_volume", 200)
                set_override("max_spread_pct_etf", 0.02)
                set_override("max_spread_pct_stock", 0.05)
                _audit({"kind": "entries_tight",
                        "user": interaction.user.id})
                await interaction.response.send_message(
                    "🔒 **Tight entries active** — only liquid, cheap, "
                    "high-confluence trades.\n"
                    "• Max premium: **$2.00**\n"
                    "• Min OI: **200**\n"
                    "• Min today volume: **200**\n"
                    "• Max spread ETF: **2%**  ·  stock: **5%**",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"tighten failed: {e}", ephemeral=True,
                )

        # ---- Row 6: Quantity per entry ------------------------------

        @discord.ui.button(label="📦 Qty = 1",
                            style=discord.ButtonStyle.secondary,
                            custom_id="ta:qty_1", row=6)
        async def btn_qty_1(self, interaction, _):
            """Default — 1 contract per entry (Kelly-independent)."""
            if not await self._authorized(interaction):
                return
            await self._set_qty(interaction, 1, 3, "Qty=1")

        @discord.ui.button(label="📦📦 Qty = 2",
                            style=discord.ButtonStyle.primary,
                            custom_id="ta:qty_2", row=6)
        async def btn_qty_2(self, interaction, _):
            """2 contracts per entry — 2× upside, 2× size."""
            if not await self._authorized(interaction):
                return
            await self._set_qty(interaction, 2, 5, "Qty=2")

        @discord.ui.button(label="📦📦📦 Qty = 3",
                            style=discord.ButtonStyle.primary,
                            custom_id="ta:qty_3", row=6)
        async def btn_qty_3(self, interaction, _):
            """3 contracts per entry — aggressive."""
            if not await self._authorized(interaction):
                return
            await self._set_qty(interaction, 3, 6, "Qty=3")

        @discord.ui.button(label="📦×5 Qty = 5",
                            style=discord.ButtonStyle.danger,
                            custom_id="ta:qty_5", row=6)
        async def btn_qty_5(self, interaction, _):
            """5 contracts per entry — very aggressive, enable scale-out path."""
            if not await self._authorized(interaction):
                return
            await self._set_qty(interaction, 5, 8, "Qty=5")

        async def _set_qty(self, interaction, default_q: int,
                            max_strong: int, label: str):
            try:
                from src.core.runtime_overrides import set_override
                set_override("default_qty_per_entry", default_q)
                set_override("max_qty_if_strong", max_strong)
                _audit({"kind": "qty_override",
                        "user": interaction.user.id,
                        "default": default_q, "max_strong": max_strong})
                await interaction.response.send_message(
                    f"📦 **{label}** — each entry now buys **{default_q} "
                    f"contracts** by default (up to **{max_strong}** on "
                    "strong momentum: delta 0.40-0.55 + 5-bar move ≥0.5% + "
                    "volume ≥2× baseline).\n\n"
                    "⚠️ Reminder — your other safety rails still apply:\n"
                    "• `max_contracts_0dte: 5` and `max_contracts_multiday: 10`\n"
                    "• `kelly_hard_cap_pct: 5%` of equity per position\n"
                    "• `max_risk_per_trade_pct: 1%`\n"
                    "• `max_premium_per_contract_usd` (currently "
                    "from overrides or $3.00 default)\n"
                    "So `Qty=5` × `$4` premium × 100 mult = **$2,000 "
                    "exposure per entry**. Stay aware.",
                    ephemeral=True,
                )
            except Exception as e:
                await interaction.response.send_message(
                    f"qty set failed: {e}", ephemeral=True,
                )

        @discord.ui.button(label="🔍 Show overrides",
                            style=discord.ButtonStyle.secondary,
                            custom_id="ta:show_overrides", row=5)
        async def btn_show_overrides(self, interaction, _):
            """Show all active runtime overrides."""
            if not await self._authorized(interaction):
                return
            try:
                from src.core.runtime_overrides import all_overrides
                d = all_overrides()
                if not d:
                    msg = ("_No active overrides — using "
                           "`config/settings.yaml` defaults._")
                else:
                    lines = ["**Active runtime overrides:**"]
                    for k, v in sorted(d.items()):
                        lines.append(f"  · `{k}` = **{v}**")
                    msg = "\n".join(lines)
                await interaction.response.send_message(msg, ephemeral=True)
            except Exception as e:
                await interaction.response.send_message(
                    f"show failed: {e}", ephemeral=True,
                )

    class ControlPanel(discord.ui.View):
        """Persistent button panel. `custom_id`s survive bot restart
        when we call client.add_view(ControlPanel()) on_ready."""

        def __init__(self):
            super().__init__(timeout=None)   # persistent

        async def _button(self, interaction, command: str):
            if not runner.is_authorized(interaction.user.id):
                await interaction.response.send_message(
                    "Not authorized.", ephemeral=True)
                _audit({"kind": "button_rejected_auth",
                        "user": interaction.user.id,
                        "command": command})
                return
            # Ack within 3s to keep Discord happy; then run async.
            await interaction.response.defer(thinking=True)
            msg = await runner.run(
                interaction.user.id,
                str(interaction.user),
                command,
            )
            await interaction.followup.send(msg)

        @discord.ui.button(label="Status", style=discord.ButtonStyle.primary,
                            custom_id="tb:status")
        async def btn_status(self, interaction, _):
            await self._button(interaction, "status")

        @discord.ui.button(label="Positions", style=discord.ButtonStyle.primary,
                            custom_id="tb:positions")
        async def btn_positions(self, interaction, _):
            await self._button(interaction, "positions")

        @discord.ui.button(label="Logs", style=discord.ButtonStyle.secondary,
                            custom_id="tb:logs")
        async def btn_logs(self, interaction, _):
            await self._button(interaction, "logs")

        @discord.ui.button(label="Doctor", style=discord.ButtonStyle.secondary,
                            custom_id="tb:doctor")
        async def btn_doctor(self, interaction, _):
            await self._button(interaction, "doctor")

        @discord.ui.button(label="Start", style=discord.ButtonStyle.success,
                            custom_id="tb:start", row=1)
        async def btn_start(self, interaction, _):
            await self._button(interaction, "start")

        @discord.ui.button(label="Restart", style=discord.ButtonStyle.success,
                            custom_id="tb:restart", row=1)
        async def btn_restart(self, interaction, _):
            await self._button(interaction, "restart")

        @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger,
                            custom_id="tb:stop", row=1)
        async def btn_stop(self, interaction, _):
            await self._button(interaction, "stop")

        @discord.ui.button(label="Walkforward", style=discord.ButtonStyle.secondary,
                            custom_id="tb:walkforward", row=2)
        async def btn_walkforward(self, interaction, _):
            await self._button(interaction, "walkforward")

        @discord.ui.button(label="Risk Switch", style=discord.ButtonStyle.secondary,
                            custom_id="tb:risk-switch", row=2)
        async def btn_risk(self, interaction, _):
            await self._button(interaction, "risk-switch")

        @discord.ui.button(label="70B Audit", style=discord.ButtonStyle.secondary,
                            custom_id="tb:audit", row=2)
        async def btn_audit(self, interaction, _):
            await self._button(interaction, "audit")

        @discord.ui.button(label="Ollama Status", style=discord.ButtonStyle.secondary,
                            custom_id="tb:ollama-status", row=3)
        async def btn_ollama_status(self, interaction, _):
            await self._button(interaction, "ollama-status")

        @discord.ui.button(label="Warm LLMs", style=discord.ButtonStyle.secondary,
                            custom_id="tb:ollama-warmup", row=3)
        async def btn_ollama_warmup(self, interaction, _):
            await self._button(interaction, "ollama-warmup")

        @discord.ui.button(label="Restart Ollama", style=discord.ButtonStyle.danger,
                            custom_id="tb:ollama-restart", row=3)
        async def btn_ollama_restart(self, interaction, _):
            await self._button(interaction, "ollama-restart")

        @discord.ui.button(label="Summary 60m", style=discord.ButtonStyle.primary,
                            custom_id="tb:summary", row=3)
        async def btn_summary(self, interaction, _):
            await self._button(interaction, "summary")

        @discord.ui.button(label="Research", style=discord.ButtonStyle.primary,
                            custom_id="tb:research", row=3)
        async def btn_research(self, interaction, _):
            await self._button(interaction, "research")

    # ------------------------- events -------------------------

    # on_ready fires on EVERY gateway connection, including reconnects.
    # On flaky networks that means the startup hello gets posted every
    # time Discord's gateway drops and reconnects — very spammy.
    # Gate the hello behind a once-per-process flag.
    _hello_state = {"sent": False}

    @client.event
    async def on_ready():
        _log.info("discord_terminal_ready user=%s id=%s",
                   client.user, getattr(client.user, "id", "?"))
        _log.info("listening in channels=%s authorized_users=%s chat_70b=%s",
                   sorted(CHANNEL_IDS), sorted(AUTH_USERS),
                   sorted(CHAT_70B_CHANNEL_IDS))
        # Re-register the persistent view so old panel buttons keep working
        # across every channel the user has !panel-ed in. Safe to call
        # on every reconnect.
        client.add_view(ControlPanel())
        client.add_view(AutotradePanel())
        client.add_view(QuickQuestionsView())

        if _hello_state["sent"]:
            _log.info("discord_terminal_hello_skipped reason=already_sent_this_process")
            return

        # Channel-aware hellos. Instead of posting the same one-liner
        # to every configured channel, we inspect each channel's name
        # and post the panel + quick-question view that fits its
        # purpose. A control-panel-named channel gets AutotradePanel +
        # ControlPanel. An llm-named channel gets QuickQuestionsView.
        # Other channels just get a minimal banner.
        ctx = _build_chat_context()
        base_hello = chat.hello(ctx) if (chat is not None and ctx is not None) \
            else "**tradebot online**"
        sent_any = False
        for cid in sorted(CHANNEL_IDS):
            try:
                ch = client.get_channel(cid) or await client.fetch_channel(cid)
                if ch is None:
                    _log.warning("discord_terminal_channel_missing id=%s", cid)
                    continue
                name = (ch.name or "").lower() if hasattr(ch, "name") else ""
                is_70b_chat = cid in CHAT_70B_CHANNEL_IDS
                chat_suffix = ""
                if chat is not None and chat.cfg.enabled:
                    if is_70b_chat:
                        chat_suffix = (f" · chat=ON ({CHAT_70B_MODEL}, 70B "
                                        "for detailed answers)")
                    else:
                        chat_suffix = f" · chat=ON ({chat.cfg.model_name})"
                else:
                    chat_suffix = " · chat=OFF"
                header = base_hello + chat_suffix

                # Route to an appropriate greeting based on channel
                # name. Fall through to a minimal hello for unknown
                # channels.
                posted = False

                # Control / admin channels get the button panels.
                if any(k in name for k in ("control-panel", "sagarbot",
                                             "dashboard-control")):
                    await ch.send(header + "\n\n**Control panel** — click any button:",
                                   view=ControlPanel())
                    await ch.send("**LLM autotrade panel** — Enable/Disable/Research:",
                                   view=AutotradePanel())
                    posted = True

                # Chat / LLM channels get QuickQuestionsView.
                elif any(k in name for k in ("8b-llm", "70b-llm", "tradebot-chat",
                                               "tradebot-market")):
                    prompt = (
                        f"{header}\n\n"
                        "**Ask me anything** — free-form message or click a quick button below.\n"
                        "Examples: `what looks good on SPY?`, `why did we skip that entry?`, "
                        "`summarize today's signals`"
                    )
                    await ch.send(prompt, view=QuickQuestionsView())
                    posted = True

                # Terminal-style channels: banner + `!help` prompt.
                elif any(k in name for k in ("terminal-access", "terminal")):
                    await ch.send(
                        header + "\n\n"
                        "**Command terminal.** Type `!help` to see commands. "
                        "Run `!autopanel` here to post an LLM-autotrade control panel."
                    )
                    posted = True

                # Reason-for-entry / market-analysis channels — quiet
                # banner since traffic is automated.
                elif any(k in name for k in ("reasonforentry", "tradebot-reason")):
                    await ch.send(
                        header + " · *This channel receives automated signal "
                        "rationales. Use `!summary` for a quick digest.*"
                    )
                    posted = True

                # Catalyst / alerts / default — minimal one-liner.
                if not posted:
                    await ch.send(header + " — ask me anything")

                sent_any = True
            except Exception as e:
                _log.warning("discord_terminal_hello_failed cid=%s err=%s",
                              cid, e)
        if sent_any:
            _hello_state["sent"] = True

    @client.event
    async def on_message(message):
        # Ignore self
        if message.author.id == (client.user.id if client.user else 0):
            return
        # Channel allowlist — ignore DMs and any unlisted channel
        if not CHANNEL_IDS:
            return
        if message.channel.id not in CHANNEL_IDS:
            return
        text = (message.content or "").strip()
        if not text:
            return

        # Non-command: route to LLM chat if enabled, else silently ignore
        if not text.startswith(COMMAND_PREFIX):
            if chat is not None and chat.cfg.enabled:
                await _handle_chat_question(message, text)
            return

        # Extract the command word
        bare = text[len(COMMAND_PREFIX):].strip()
        parts = bare.split()
        if not parts:
            return
        command = parts[0].lower()

        if command == "help":
            await message.channel.send(HELP_TEXT)
            return
        if command == "panel":
            if not runner.is_authorized(message.author.id):
                await message.channel.send("Not authorized.")
                return
            await message.channel.send(
                "**tradebot control panel**",
                view=ControlPanel(),
            )
            return
        if command == "autopanel":
            # Post the LLM autotrade control panel. Operator runs this
            # once in the channel they want to control autotrade from
            # (typically #control-panel or #sagarbot). Persistent —
            # survives bot restart.
            if not runner.is_authorized(message.author.id):
                await message.channel.send("Not authorized.")
                return
            view = AutotradePanel()
            try:
                status_text = await view._status_text()
            except Exception:
                status_text = "**LLM Autotrade Panel**"
            await message.channel.send(status_text, view=view)
            return
        if command == "ask":
            # Explicit LLM chat: !ask <question>
            question = bare[len("ask"):].strip()
            if not question:
                await message.channel.send(
                    f"Usage: `{COMMAND_PREFIX}ask <question>`"
                )
                return
            await _handle_chat_question(message, question)
            return
        if command == "destroy":
            # confirm a pending destructive action
            msg = await runner.run(message.author.id,
                                     str(message.author),
                                     "DESTROY", raw_text=text)
            await message.channel.send(_truncate(msg))
            return

        msg = await runner.run(message.author.id,
                                 str(message.author),
                                 command, raw_text=text)
        await message.channel.send(_truncate(msg))

    return client


def main() -> int:
    if not BOT_TOKEN:
        print("DISCORD_BOT_TOKEN not set in .env — aborting.", file=sys.stderr)
        return 2
    if not CHANNEL_IDS:
        print("No channels configured. Set DISCORD_TERMINAL_CHANNEL_IDS "
              "(comma-separated) or DISCORD_TERMINAL_CHANNEL_ID in .env.",
              file=sys.stderr)
        return 2
    if not AUTH_USERS:
        print("DISCORD_TERMINAL_AUTHORIZED_USERS empty — aborting (would "
              "leave the bot open to anyone in the channel).", file=sys.stderr)
        return 2
    client = _build_app()
    client.run(BOT_TOKEN, log_handler=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
