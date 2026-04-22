"""Conversational LLM wrapper — answers free-form questions from
Discord / dashboard with the current bot state as context.

Different from LLMBrain:
  - LLMBrain reviews a specific proposed decision → structured JSON.
  - LLMChat answers a question in natural language for a human.

Same underlying LLM (Ollama 8B by default). Rate-limited + audited so
a single user can't flood the GPU.

## Security posture
  - Caller already authorizes the user (Discord channel + user allowlist).
  - This module sanitizes the LLM reply: strips role-mentions
    (`@everyone`, `@here`, `<@user_id>`) to prevent prompt-injection-
    driven pings, caps length, removes ANSI control sequences.
  - Never executes anything the LLM suggests. LLMChat is read-only
    conversation; trades still go through rules + filter chain.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_log = logging.getLogger(__name__)


@dataclass
class LLMChatConfig:
    enabled: bool = False
    backend: str = "ollama"                        # ollama | llama_cpp
    model_name: str = "llama3.1:8b"
    model_path: str = ""                           # only for llama_cpp
    max_tokens: int = 350
    temperature: float = 0.25                      # slightly warmer than the brain
    n_ctx: int = 4096
    rate_limit_per_user_per_min: int = 10
    timeout_sec: float = 20.0


@dataclass
class ChatContext:
    """Compact bot-state snapshot — same shape the auditor uses, smaller."""
    spot_by_symbol: Dict[str, float] = field(default_factory=dict)
    vix: Optional[float] = None
    regime: Optional[str] = None
    breadth_score: Optional[float] = None
    open_positions: int = 0
    positions_summary: List[str] = field(default_factory=list)   # ["SPY qty=-1 avg=$2.30", ...]
    day_pnl_usd: Optional[float] = None
    live_trading: bool = False
    universe: List[str] = field(default_factory=list)
    recent_signals: List[str] = field(default_factory=list)      # short strings
    last_audit_summary: Optional[str] = None
    last_audit_health: Optional[int] = None
    now_iso: str = ""
    # Session awareness — tells the LLM whether to analyze "live" vs
    # "overnight / next-open". Populated by the dispatcher.
    market_is_open: bool = False
    hours_until_open: Optional[float] = None
    # Optional — populated on demand when the question references
    # options / contracts / specific strike behavior. The chat
    # dispatcher pulls this from MultiProvider before building the
    # prompt. Keeping it in the context dataclass keeps serialization
    # uniform.
    option_chain_atm: List[Dict[str, Any]] = field(default_factory=list)
    news_headlines: List[Dict[str, Any]] = field(default_factory=list)
    # Political / macro news (Fed speeches, White House, Truth Social,
    # Treasury, Reddit WSB) pulled from political_news provider.
    political_headlines: List[Dict[str, Any]] = field(default_factory=list)


# -------------------------------------------------------------- sanitizers


_MENTION_RE = re.compile(r"@everyone|@here|<@[!&]?\d+>|<#\d+>|<@&\d+>")
_ANSI_RE    = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _sanitize(out: str, max_len: int = 1800) -> str:
    """Strip mention tokens + ANSI + cap length. Keeps it safe to post
    back to Discord without the LLM accidentally (or intentionally)
    pinging @everyone."""
    out = _ANSI_RE.sub("", out)
    out = _MENTION_RE.sub("[mention-stripped]", out)
    if len(out) > max_len:
        out = out[:max_len - 40].rstrip() + "\n\n… [truncated]"
    return out.strip()


# -------------------------------------------------------------- prompts


_SYSTEM_HINT = (
    "You are a senior options-strategy analyst advising the operator of "
    "a retail paper-trading bot called tradebot. You help make sense of "
    "live state + market data. You do NOT place trades — the rules "
    "engine owns that.\n\n"
    "HARD CITATION RULES:\n"
    "• Every price, VIX number, OI, IV, or score you cite MUST come "
    "directly from the SNAPSHOT. Don't invent or guess.\n"
    "• When you reference news, name the source tag shown in the "
    "snapshot (e.g. '[finnhub] Powell held...', '[whitehouse] tariff...').\n"
    "• When claiming something about the market ('selling pressure', "
    "'weakness', 'breakout'), cite the SPECIFIC number in the snapshot "
    "that supports it (a price level, a P/C ratio, a sentiment score).\n"
    "• If the snapshot lacks data to answer, SAY SO — don't fill with "
    "generic statements.\n\n"
    "SESSION AWARENESS:\n"
    "• Check `market_is_open` in the snapshot. If false, frame your "
    "analysis as 'overnight developments + next-open setup', not "
    "'what's happening now'.\n"
    "• After-hours questions about SPY/QQQ should reference VIX close, "
    "overnight news, and positioning implications for the open.\n\n"
    "OUTPUT DEMANDS:\n"
    "• End every substantive answer with 2-3 RISK FLAGS citing specific "
    "snapshot data (political news headline, VIX level, breadth "
    "deterioration, open-position risk, etc.).\n"
    "• For SPY/QQQ trade questions, name a specific strike + expiry "
    "from the option_chain_atm if chain is provided.\n"
    "• ALWAYS discuss BOTH BULL AND BEAR cases. If snapshot shows no "
    "bearish signal, explicitly say 'no bearish setup right now because "
    "X' — don't just skip it. Same for bull case. Operator wants parity "
    "so they can trade either direction when one presents.\n"
    "• If the operator asks 'what's the best trade?' give ONE specific "
    "call-side idea AND ONE specific put-side idea, then say which has "
    "stronger confluence given the snapshot.\n"
    "• Keep answers under 12 sentences + risk flags. Concrete over verbose.\n\n"
    "SAFETY:\n"
    "• No @everyone / @here / role mentions.\n"
    "• Don't suggest live trades — the rules engine decides."
)


def _build_prompt(question: str, ctx: ChatContext) -> str:
    snapshot: Dict[str, Any] = {
        "now": ctx.now_iso,
        "mode": "live" if ctx.live_trading else "paper",
        "market_is_open": ctx.market_is_open,
        "universe": ctx.universe,
        "spot_prices": ctx.spot_by_symbol,
        "vix": ctx.vix,
        "regime": ctx.regime,
        "breadth_score": ctx.breadth_score,
        "open_positions": ctx.open_positions,
        "positions": ctx.positions_summary,
        "day_pnl_usd": ctx.day_pnl_usd,
        "recent_signals": ctx.recent_signals[-8:],
        "last_audit": (
            {"health": ctx.last_audit_health, "summary": ctx.last_audit_summary}
            if ctx.last_audit_health is not None else None
        ),
    }
    if ctx.hours_until_open is not None and not ctx.market_is_open:
        snapshot["hours_until_next_open"] = round(ctx.hours_until_open, 1)
    # Only include these blocks when they contain data — keeps the
    # prompt compact for simple questions.
    if ctx.option_chain_atm:
        snapshot["option_chain_atm"] = ctx.option_chain_atm[:20]
    if ctx.news_headlines:
        snapshot["headlines"] = ctx.news_headlines[:12]
    if ctx.political_headlines:
        snapshot["political_news"] = ctx.political_headlines[:15]
    return (
        f"{_SYSTEM_HINT}\n\n"
        f"SNAPSHOT:\n{json.dumps(snapshot, indent=2, default=str)}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n"
    )


# Keywords that suggest the user wants an options-specific answer, so
# the chat dispatcher should enrich the context with live chain + news.
# Match case-insensitive, whole-word-ish.
_OPTIONS_QUESTION_RE = re.compile(
    r"\b("
    r"option|options|strike|call|puts?|delta|gamma|theta|vega|"
    r"iv|volatility|premium|expiry|expiration|chain|otm|itm|atm|"
    r"credit spread|debit spread|straddle|strangle|butterfly|"
    r"iron condor|wheel|csp|ccs|spx|spy|qqq|trade setup|setup|"
    r"target price|sentiment|news|research"
    r")\b",
    re.IGNORECASE,
)


def question_wants_options_context(question: str) -> bool:
    """True when the question appears to need live options / chain
    / news data. Used by the dispatcher to decide whether to pay the
    data-fetch cost."""
    return bool(_OPTIONS_QUESTION_RE.search(question or ""))


# -------------------------------------------------------------- rate limit


class _SlidingRateLimit:
    def __init__(self, max_per_min: int):
        self.max = max_per_min
        self._hits: Dict[int, List[float]] = {}

    def allow(self, user_id: int) -> bool:
        now = time.time()
        hits = self._hits.setdefault(user_id, [])
        # drop anything older than 60s
        self._hits[user_id] = [t for t in hits if now - t < 60.0]
        if len(self._hits[user_id]) >= self.max:
            return False
        self._hits[user_id].append(now)
        return True


# -------------------------------------------------------------- chat


class LLMChat:
    """Answer a user question given a compact bot-state context."""

    def __init__(self, cfg: Optional[LLMChatConfig] = None):
        self.cfg = cfg or LLMChatConfig()
        self._client = None
        self._llama = None
        self._rate = _SlidingRateLimit(self.cfg.rate_limit_per_user_per_min)
        self._load_err: Optional[str] = None

    # ---------- setup ----------

    def _ensure_backend(self) -> bool:
        if not self.cfg.enabled:
            return False
        if self.cfg.backend == "ollama":
            if self._client is not None:
                return True
            try:
                from .ollama_client import build_ollama_client
                c = build_ollama_client()
                if not c.ping():
                    self._load_err = "ollama_unreachable"
                    return False
                self._client = c
                return True
            except Exception as e:
                self._load_err = f"ollama_init:{e}"
                return False
        # llama_cpp fallback
        if self._llama is not None:
            return True
        if not self.cfg.model_path or not os.path.exists(self.cfg.model_path):
            self._load_err = "model_path_missing"
            return False
        try:
            from llama_cpp import Llama
            self._llama = Llama(model_path=self.cfg.model_path,
                                  n_ctx=self.cfg.n_ctx, n_gpu_layers=-1,
                                  verbose=False)
            return True
        except Exception as e:
            self._load_err = f"llama_init:{e}"
            return False

    # ---------- public API ----------

    def answer(self, question: str, ctx: ChatContext,
               *, user_id: int = 0,
               model_override: Optional[str] = None,
               max_tokens_override: Optional[int] = None) -> str:
        """Get a natural-language answer. Returns a sanitized string.

        Args:
          model_override: optional Ollama tag (e.g. 'llama3.1:70b') to
            use for THIS call only. Lets callers route specific Discord
            channels to the 70B for detailed answers while keeping the
            8B default elsewhere.
          max_tokens_override: optional per-call token budget. 70B
            answers usually want more headroom than the 8B default.

        Fail-open: on any backend error, a short "can't answer right
        now, see logs" message is returned — never raises.
        """
        if not self.cfg.enabled:
            return "_LLM chat is disabled. Set LLM_CHAT_ENABLED=1 in .env._"
        if user_id and not self._rate.allow(user_id):
            return ("Rate limit: max "
                    f"{self.cfg.rate_limit_per_user_per_min} chat calls/min.")
        if not self._ensure_backend():
            return (f"_LLM backend unavailable_ "
                    f"({self._load_err or 'unknown'}).")

        prompt = _build_prompt(question.strip()[:1500], ctx)
        try:
            raw = self._infer(prompt, model_override=model_override,
                              max_tokens_override=max_tokens_override)
        except Exception as e:
            _log.warning("llm_chat_infer_failed err=%s err_type=%s",
                          e, type(e).__name__, exc_info=True)
            raw = ""

        # 70B fallback: if the requested model (70B) returned empty or
        # failed, retry with 8B and tag the reply so the operator sees
        # what happened. Better a quick 8B answer than silent failure.
        if (not raw or not raw.strip()) and model_override and model_override != self.cfg.model_name:
            _log.info("llm_chat_fallback_to_default model=%s -> %s",
                      model_override, self.cfg.model_name)
            try:
                raw = self._infer(prompt, model_override=None,
                                  max_tokens_override=None)
                if raw and raw.strip():
                    return _sanitize(raw) + (
                        f"\n_⚠️ 70B ({model_override}) unavailable — "
                        f"replied via {self.cfg.model_name} instead._"
                    )
            except Exception as e:
                _log.warning("llm_chat_fallback_also_failed err=%s", e)

        answer = _sanitize(raw) if raw else ""
        if not answer:
            return (
                f"_LLM returned empty response "
                f"(model={model_override or self.cfg.model_name}). "
                f"Ollama may be loading the model — try again in 60s, "
                f"or press **Warm LLMs** in the panel._"
            )
        return answer

    def hello(self, ctx: ChatContext) -> str:
        """One-line greeting for the Discord startup hello message."""
        parts = [
            "**tradebot online**",
            f"mode={'live' if ctx.live_trading else 'paper'}",
            f"universe={','.join(ctx.universe) or '—'}",
            f"backend={self.cfg.backend} ({self.cfg.model_name})",
        ]
        if ctx.regime:
            parts.append(f"regime={ctx.regime}")
        if ctx.vix is not None:
            parts.append(f"vix={ctx.vix:.1f}")
        if ctx.open_positions:
            parts.append(f"positions={ctx.open_positions}")
        return " · ".join(parts)

    # ---------- backends ----------

    def _infer(self, prompt: str, *,
               model_override: Optional[str] = None,
               max_tokens_override: Optional[int] = None) -> str:
        model = model_override or self.cfg.model_name
        max_tokens = max_tokens_override or self.cfg.max_tokens
        if self.cfg.backend == "ollama" and self._client is not None:
            return self._client.generate(
                model=model,
                prompt=prompt,
                temperature=self.cfg.temperature,
                max_tokens=max_tokens,
                num_ctx=self.cfg.n_ctx,
                stop=["\n\nQUESTION:", "\n\nSNAPSHOT:"],
            )
        if self._llama is not None:
            # llama_cpp can't hot-swap models per call — it was loaded
            # once with a specific GGUF. Override is silently ignored here;
            # 70B-per-channel only works with the ollama backend.
            resp = self._llama.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.cfg.temperature,
                stop=["\n\nQUESTION:", "\n\nSNAPSHOT:"],
            )
            try:
                return resp["choices"][0]["text"]
            except Exception:
                return ""
        return ""


# -------------------------------------------------------------- factory


def build_llm_chat_from_env() -> LLMChat:
    """Env-driven factory — intentionally NOT reading settings.yaml since
    chat is mostly a Discord-side feature and its on/off lives with the
    Discord bot config."""
    enabled = os.getenv("LLM_CHAT_ENABLED", "").strip() in ("1", "true", "yes")
    backend = (os.getenv("LLM_BACKEND", "ollama").strip().lower()
                or "ollama")
    if backend not in ("ollama", "llama_cpp"):
        backend = "ollama"
    model = (os.getenv("LLM_CHAT_MODEL", "").strip()
              or os.getenv("LLM_BRAIN_MODEL", "").strip()
              or "llama3.1:8b")
    return LLMChat(LLMChatConfig(
        enabled=enabled,
        backend=backend,
        model_name=model,
        model_path=os.getenv("LLM_MODEL_PATH", ""),
        rate_limit_per_user_per_min=int(
            os.getenv("LLM_CHAT_RATE_LIMIT_PER_MIN", "10")
        ),
    ))
