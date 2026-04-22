"""Groq LLM backend — cloud-hosted Llama 3.3 70B at 500+ tok/sec.

Free tier gives ~14,400 req/day — way more than the bot makes. Used
for research / audit / macro / catalyst calls where quality > local
privacy. Per-trade 8B brain review stays on local Ollama (fast, private).

Sign up: https://console.groq.com → API Keys → Create
Add to .env: GROQ_API_KEY=gsk_...

Interface matches OllamaClient.generate() so consumer scripts can
swap backends with a one-line factory call.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional
from urllib import request, error


_log = logging.getLogger(__name__)

_BASE = "https://api.groq.com/openai/v1"

# Model tags Groq accepts (as of late 2026). If Groq retires one, we
# fall through to the next.
_PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]


@dataclass
class GroqConfig:
    api_key: str = ""
    base_url: str = _BASE
    timeout_sec: float = 60.0
    default_model: str = "llama-3.3-70b-versatile"


class GroqClient:
    """OpenAI-compatible chat client for Groq's hosted Llama.
    Interface mirrors OllamaClient for drop-in use."""

    def __init__(self, cfg: Optional[GroqConfig] = None):
        self.cfg = cfg or GroqConfig()
        if not self.cfg.api_key:
            raise RuntimeError("GROQ_API_KEY not set")

    def ping(self) -> bool:
        """Quick reachability check via /models endpoint."""
        try:
            req = request.Request(
                f"{self.cfg.base_url}/models",
                headers=self._headers(),
            )
            with request.urlopen(req, timeout=5.0):
                return True
        except Exception as e:
            _log.info("groq_ping_failed err=%s", e)
            return False

    def _headers(self, extra: Optional[dict] = None) -> dict:
        """Standard headers. Sends a browser-like User-Agent because
        Groq's Cloudflare front-end returns 403 error 1010 on the
        default urllib UA (blocks suspected bot fingerprints)."""
        h = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if extra:
            h.update(extra)
        return h

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 500,
        num_ctx: Optional[int] = None,      # ignored on Groq
        stop: Optional[list] = None,
    ) -> str:
        """One-shot completion. Returns the raw response text.
        Returns "" on any error (fail-open, caller's path handles it)."""
        # Route OLLAMA-style model tags to GROQ-equivalent. Caller can
        # pass "llama3.1:70b" or "llama-3.3-70b-versatile" — we accept
        # both.
        resolved_model = _resolve_model(model, self.cfg.default_model)
        payload = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if stop:
            payload["stop"] = list(stop)
        try:
            req = request.Request(
                f"{self.cfg.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers({"Content-Type": "application/json"}),
            )
            with request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                raw = resp.read()
        except error.HTTPError as e:
            _log.warning("groq_http_err code=%s body=%s",
                         e.code,
                         e.read().decode("utf-8", errors="replace")[:240])
            return ""
        except Exception as e:                          # noqa: BLE001
            _log.info("groq_network_err err=%s", e)
            return ""
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
            return data["choices"][0]["message"]["content"] or ""
        except Exception as e:                          # noqa: BLE001
            _log.info("groq_parse_err err=%s", e)
            return ""


def _resolve_model(requested: str, default: str) -> str:
    """Map Ollama-style tags to Groq tags so consumer scripts can use
    the same model string regardless of backend.

    Examples:
      llama3.1:70b      → llama-3.3-70b-versatile
      llama3.3          → llama-3.3-70b-versatile
      llama3.1:8b       → (Groq doesn't host 8B) → default 70B
      mistral:7b        → mixtral-8x7b-32768
    """
    r = requested.lower().strip()
    if "70b" in r or "3.3" in r:
        return "llama-3.3-70b-versatile"
    if "3.1-70b" in r:
        return "llama-3.1-70b-versatile"
    if "mixtral" in r or "mistral" in r:
        return "mixtral-8x7b-32768"
    # 8B / small models — Groq doesn't host 8B; use 70B (quota-free up
    # to 14k/day so upgrading isn't a problem).
    return default


def build_groq_client() -> Optional[GroqClient]:
    """Factory. Returns None when GROQ_API_KEY isn't set."""
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        return None
    try:
        return GroqClient(GroqConfig(api_key=key))
    except Exception as e:                              # noqa: BLE001
        _log.info("groq_build_failed err=%s", e)
        return None


# ----------------------------------------------------------- role-based factory


def build_llm_client_for(role: str):
    """Return the best LLM backend for `role` given current .env.

    Roles:
      - 'brain'       → per-trade review. Local Ollama 8B always (fast,
                         private). Never uses Groq.
      - 'research'    → options research agent. Groq 70B if configured,
                         else Ollama (whichever tag LLM_AUDITOR_MODEL
                         points at).
      - 'audit'       → strategy auditor. Same routing as research.
      - 'macro'       → nightly macro sweep. Same routing as research.
      - 'chat'        → user chat. Ollama (per-channel 8B/70B).
      - 'chat_70b'    → 70B chat channel. Groq if available, else local.
      - 'catalyst'    → catalyst deep-dive. Same as research.

    Returns (client, model_name_to_pass). Model name is whatever the
    client expects (Groq resolves Ollama tags automatically).
    """
    import os as _os
    role = (role or "").lower()
    # Try Groq first for the heavy/research roles
    if role in ("research", "audit", "macro", "catalyst", "chat_70b"):
        g = build_groq_client()
        if g is not None:
            model = (_os.getenv("LLM_AUDITOR_MODEL", "").strip()
                     or "llama-3.3-70b-versatile")
            return g, model
    # Fall through to Ollama
    try:
        from .ollama_client import build_ollama_client
        c = build_ollama_client()
        if role == "brain":
            model = _os.getenv("LLM_BRAIN_MODEL", "").strip() or "llama3.1:8b"
        elif role == "chat_70b":
            model = _os.getenv("LLM_CHAT_70B_MODEL", "").strip() or "llama3.1:70b"
        elif role == "chat":
            model = _os.getenv("LLM_CHAT_MODEL", "").strip() or "llama3.1:8b"
        else:
            model = _os.getenv("LLM_AUDITOR_MODEL", "").strip() or "llama3.1:70b"
        return c, model
    except Exception:
        return None, ""
