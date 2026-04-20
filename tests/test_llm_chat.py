"""LLMChat tests — context assembly, sanitization, fail-open behavior."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest

from src.intelligence.llm_chat import (
    LLMChat, LLMChatConfig, ChatContext,
    _sanitize, _build_prompt, build_llm_chat_from_env,
)


# ---------- sanitizer ----------


def test_sanitize_strips_everyone_mention():
    assert "[mention-stripped]" in _sanitize("ping @everyone now")
    assert "@everyone" not in _sanitize("ping @everyone now")


def test_sanitize_strips_user_and_role_mentions():
    out = _sanitize("hey <@12345> and <@&67890> and <#54321>")
    assert "<@12345>" not in out
    assert "<@&67890>" not in out
    assert "<#54321>" not in out


def test_sanitize_strips_ansi_escapes():
    raw = "\x1b[31mred text\x1b[0m ok"
    out = _sanitize(raw)
    assert "\x1b" not in out
    assert "red text" in out


def test_sanitize_truncates_long_output():
    out = _sanitize("A" * 3000, max_len=500)
    assert len(out) <= 500 + 80      # include truncation marker slack
    assert "truncated" in out


# ---------- prompt builder ----------


def test_prompt_includes_snapshot_and_question():
    ctx = ChatContext(
        spot_by_symbol={"SPY": 580.1, "QQQ": 641.0},
        vix=16.2, regime="trend_lowvol", open_positions=2,
        universe=["SPY", "QQQ"], live_trading=False,
        now_iso="2026-04-20T12:00:00Z",
    )
    p = _build_prompt("how are we doing today?", ctx)
    assert "580.1" in p
    assert "trend_lowvol" in p
    assert "how are we doing today?" in p
    assert "QUESTION:" in p and "SNAPSHOT:" in p


def test_prompt_injects_system_safety_rules():
    ctx = ChatContext()
    p = _build_prompt("hi", ctx)
    # Anti-mention rule must be in the prompt
    assert "@everyone" in p or "mention" in p
    # No-trade rule
    assert "do not" in p.lower() or "sanity-check" in p.lower()


# ---------- rate limit + fail-open ----------


def test_disabled_chat_returns_message_without_calling_backend():
    chat = LLMChat(LLMChatConfig(enabled=False))
    out = chat.answer("hi?", ChatContext(), user_id=1)
    assert "disabled" in out.lower()


def test_rate_limit_blocks_after_threshold():
    chat = LLMChat(LLMChatConfig(enabled=True, backend="ollama",
                                    rate_limit_per_user_per_min=2))
    # Mock backend so the answer path succeeds
    chat._client = MagicMock()
    chat._client.ping = lambda: True
    chat._client.generate = lambda **kw: "stub answer"
    # 3rd hit should block
    chat.answer("q1", ChatContext(), user_id=42)
    chat.answer("q2", ChatContext(), user_id=42)
    blocked = chat.answer("q3", ChatContext(), user_id=42)
    assert "Rate limit" in blocked


def test_answer_fails_open_when_ollama_unreachable(monkeypatch):
    chat = LLMChat(LLMChatConfig(enabled=True, backend="ollama"))

    class _BrokenClient:
        class cfg: base_url = "http://127.0.0.1:11434"
        def ping(self): return False
    monkeypatch.setattr(
        "src.intelligence.ollama_client.build_ollama_client",
        lambda: _BrokenClient(),
    )
    out = chat.answer("hi?", ChatContext(), user_id=1)
    assert "unavailable" in out.lower() or "ollama_unreachable" in out.lower()


def test_answer_calls_ollama_and_sanitizes(monkeypatch):
    chat = LLMChat(LLMChatConfig(enabled=True, backend="ollama",
                                    rate_limit_per_user_per_min=100))

    class _Client:
        class cfg: base_url = "http://127.0.0.1:11434"
        def ping(self): return True
        def generate(self, **kw):
            return "sure! ping @everyone with <@12345> for details"
    monkeypatch.setattr(
        "src.intelligence.ollama_client.build_ollama_client",
        lambda: _Client(),
    )
    out = chat.answer("what's up?", ChatContext(), user_id=1)
    assert "@everyone" not in out
    assert "<@12345>" not in out
    assert "sure!" in out


def test_hello_includes_mode_and_model():
    chat = LLMChat(LLMChatConfig(enabled=True, backend="ollama",
                                    model_name="llama3.1:8b"))
    ctx = ChatContext(universe=["SPY", "QQQ"], live_trading=False,
                        regime="trend_lowvol", vix=16.0, open_positions=1)
    msg = chat.hello(ctx)
    assert "online" in msg.lower()
    assert "paper" in msg
    assert "llama3.1:8b" in msg
    assert "trend_lowvol" in msg


# ---------- factory ----------


def test_factory_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_CHAT_ENABLED", "1")
    monkeypatch.setenv("LLM_CHAT_MODEL", "my-custom-tag")
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    chat = build_llm_chat_from_env()
    assert chat.cfg.enabled is True
    assert chat.cfg.model_name == "my-custom-tag"
    assert chat.cfg.backend == "ollama"


def test_factory_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LLM_CHAT_ENABLED", raising=False)
    chat = build_llm_chat_from_env()
    assert chat.cfg.enabled is False
