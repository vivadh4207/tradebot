"""Thin HTTP client for Ollama's local API.

Ollama exposes http://localhost:11434/api/generate (POST, JSON). We use
that instead of loading GGUF files through llama-cpp-python when the
operator has Ollama installed. Two big wins:

  1. No per-model symlink dance — Ollama manages the blob store.
     `ollama pull llama3.1:70b` and the bot uses it by tag.
  2. No Jetson-specific llama-cpp-python CUDA rebuild. Ollama ships
     its own CUDA-accelerated runtime.

Stdlib urllib only — zero new dependencies.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib import request, error


_log = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    # 70B first-call cold-load is 60-90s; after that generation at
    # 5-10 tok/sec means 300 tokens = 30-60s. Keep generous headroom
    # so chat doesn't empty-return when the model is swapping in.
    timeout_sec: float = 300.0


class OllamaClient:
    """Stateless HTTP client. `generate()` is synchronous + non-streaming
    so integration with the existing LLMBrain/StrategyAuditor is a drop-in
    replacement for `llama_cpp.Llama.create_completion`."""

    def __init__(self, cfg: Optional[OllamaConfig] = None):
        self.cfg = cfg or OllamaConfig()

    def ping(self) -> bool:
        """Is Ollama actually reachable? Used at startup so we fail-open
        fast if the daemon isn't running (instead of on every call)."""
        try:
            req = request.Request(self.cfg.base_url.rstrip("/") + "/api/tags")
            with request.urlopen(req, timeout=3.0):
                return True
        except Exception as e:
            _log.info("ollama_ping_failed url=%s err=%s",
                       self.cfg.base_url, e)
            return False

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 200,
        num_ctx: Optional[int] = None,
        stop: Optional[list] = None,
    ) -> str:
        """One-shot completion. Returns the raw string response.

        Returns "" on any error (network timeout, 500 from Ollama,
        unexpected payload). Caller's fail-open path takes it from there.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,                # single-shot, simpler parsing
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        if num_ctx is not None:
            payload["options"]["num_ctx"] = int(num_ctx)
        if stop:
            payload["options"]["stop"] = list(stop)

        try:
            req = request.Request(
                self.cfg.base_url.rstrip("/") + "/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                raw = resp.read()
        except error.URLError as e:
            _log.warning("ollama_generate_network_err model=%s err=%s", model, e)
            return ""
        except Exception as e:
            _log.warning("ollama_generate_err model=%s err=%s", model, e)
            return ""

        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return ""
        # Standard response shape: {"response": "...", "done": true, ...}
        return str(data.get("response", ""))


def build_ollama_client() -> OllamaClient:
    """Factory reading env overrides. No new settings.yaml plumbing
    needed — the caller supplies model names from its own config."""
    base = (os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").strip()
    timeout = float(os.getenv("OLLAMA_TIMEOUT_SEC", "300.0"))
    return OllamaClient(OllamaConfig(base_url=base, timeout_sec=timeout))
