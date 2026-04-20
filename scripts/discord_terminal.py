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

AUTH_USERS = _parse_id_set(os.getenv("DISCORD_TERMINAL_AUTHORIZED_USERS") or "")
COMMAND_PREFIX    = os.getenv("DISCORD_TERMINAL_PREFIX", "!").strip() or "!"
MAX_OUT_CHARS     = 1850      # keep us under Discord's 2000-char message cap
CMD_TIMEOUT_SEC   = 180.0     # longest single command (70B audit can take 2 min)
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
    "\nLifecycle:\n"
    f"  `{COMMAND_PREFIX}start`  `{COMMAND_PREFIX}stop`  `{COMMAND_PREFIX}restart`\n"
    "\nReports:\n"
    f"  `{COMMAND_PREFIX}walkforward`   — nightly edge report on demand\n"
    f"  `{COMMAND_PREFIX}risk-switch`  — refresh CBOE put/call OI state\n"
    f"  `{COMMAND_PREFIX}audit`         — 70B strategy audit (~30-120s)\n"
    "\nDangerous (require DESTROY confirmation):\n"
    f"  `{COMMAND_PREFIX}reset-paper`  — flatten + wipe journal\n"
    f"  `{COMMAND_PREFIX}wipe`          — wipe journal only\n"
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


def _build_app():
    """Construct the discord client + command router. Lazy-imports
    discord so `import discord_terminal` doesn't require the lib to be
    installed (useful for tests)."""
    import discord
    from discord import app_commands   # noqa: F401

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    runner = CommandRunner()

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
        _log.info("listening in channels=%s authorized_users=%s",
                   sorted(CHANNEL_IDS), sorted(AUTH_USERS))
        # Re-register the persistent view so old panel buttons keep working
        # across every channel the user has !panel-ed in.
        client.add_view(ControlPanel())

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
        if not text.startswith(COMMAND_PREFIX):
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
