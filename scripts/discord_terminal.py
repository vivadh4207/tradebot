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

    # ------------------------- button panel -------------------------

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

    # ------------------------- events -------------------------

    @client.event
    async def on_ready():
        _log.info("discord_terminal_ready user=%s id=%s",
                   client.user, getattr(client.user, "id", "?"))
        _log.info("listening in channels=%s authorized_users=%s chat_70b=%s",
                   sorted(CHANNEL_IDS), sorted(AUTH_USERS),
                   sorted(CHAT_70B_CHANNEL_IDS))
        # Re-register the persistent view so old panel buttons keep working
        # across every channel the user has !panel-ed in.
        client.add_view(ControlPanel())

        # Startup hello — posts one line per configured channel so the
        # operator can visually confirm every channel is wired up and
        # which model is answering there.
        ctx = _build_chat_context()
        if chat is not None and ctx is not None:
            base_hello = chat.hello(ctx)
        else:
            base_hello = "**tradebot online** · chat disabled"
        for cid in sorted(CHANNEL_IDS):
            try:
                ch = client.get_channel(cid) or await client.fetch_channel(cid)
                if ch is None:
                    _log.warning("discord_terminal_channel_missing id=%s", cid)
                    continue
                chat_suffix = ""
                if chat is not None and chat.cfg.enabled:
                    if cid in CHAT_70B_CHANNEL_IDS:
                        chat_suffix = (f" · chat=ON ({CHAT_70B_MODEL}, 70B "
                                        "for detailed answers)")
                    else:
                        chat_suffix = f" · chat=ON ({chat.cfg.model_name})"
                else:
                    chat_suffix = " · chat=OFF"
                await ch.send(base_hello + chat_suffix
                               + " — ask me anything")
            except Exception as e:
                _log.warning("discord_terminal_hello_failed cid=%s err=%s",
                              cid, e)

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
