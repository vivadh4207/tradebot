"""OllamaClient + backend-switch tests.

No real Ollama daemon is required — we monkey-patch urllib to avoid
network. These tests check shape, error handling, fail-open behavior.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from src.intelligence.ollama_client import (
    OllamaClient, OllamaConfig, build_ollama_client,
)


def _fake_urlopen(body: dict, *, status: int = 200):
    """Helper: fabricate a urllib response that returns `body` as JSON."""
    class _R:
        def __init__(self, data):
            self._data = data
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    return _R(json.dumps(body).encode("utf-8"))


def test_ping_ok_when_server_responds():
    client = OllamaClient(OllamaConfig(base_url="http://localhost:11434"))
    with patch("src.intelligence.ollama_client.request.urlopen",
                return_value=_fake_urlopen({"models": []})):
        assert client.ping() is True


def test_ping_fails_graceful_when_daemon_down():
    from urllib import error
    client = OllamaClient(OllamaConfig(base_url="http://localhost:11434"))
    with patch("src.intelligence.ollama_client.request.urlopen",
                side_effect=error.URLError("connection refused")):
        assert client.ping() is False


def test_generate_returns_response_field():
    client = OllamaClient(OllamaConfig())
    fake = _fake_urlopen({"response": "hello world", "done": True})
    with patch("src.intelligence.ollama_client.request.urlopen",
                return_value=fake):
        out = client.generate(model="llama3.1:8b", prompt="hi")
    assert out == "hello world"


def test_generate_fails_graceful_on_network_error():
    from urllib import error
    client = OllamaClient(OllamaConfig())
    with patch("src.intelligence.ollama_client.request.urlopen",
                side_effect=error.URLError("connection refused")):
        out = client.generate(model="llama3.1:8b", prompt="hi")
    assert out == ""   # fail-open


def test_generate_fails_graceful_on_malformed_json():
    client = OllamaClient(OllamaConfig())
    class _Broken:
        def read(self):
            return b"not-json-at-all"
        def __enter__(self): return self
        def __exit__(self, *a): return False
    with patch("src.intelligence.ollama_client.request.urlopen",
                return_value=_Broken()):
        out = client.generate(model="llama3.1:8b", prompt="hi")
    assert out == ""


def test_build_ollama_client_reads_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-remote:11434")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SEC", "45")
    c = build_ollama_client()
    assert c.cfg.base_url == "http://my-remote:11434"
    assert c.cfg.timeout_sec == 45.0


def test_generate_payload_includes_options():
    """Verify the request body carries temperature, num_predict, etc.
    Rather than mock urllib directly, we inspect the bytes passed to
    urllib.request.Request via a side-effect capture."""
    client = OllamaClient(OllamaConfig())
    captured = {}

    def capture_request(url, data=None, headers=None):
        captured["url"] = url
        captured["data"] = data
        return MagicMock()

    fake = _fake_urlopen({"response": "x", "done": True})
    with patch("src.intelligence.ollama_client.request.Request",
                side_effect=capture_request):
        with patch("src.intelligence.ollama_client.request.urlopen",
                    return_value=fake):
            client.generate(
                model="llama3.1:8b", prompt="hello",
                temperature=0.2, max_tokens=123, num_ctx=2048,
                stop=["\n\n"],
            )
    body = json.loads(captured["data"])
    assert body["model"] == "llama3.1:8b"
    assert body["prompt"] == "hello"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.2
    assert body["options"]["num_predict"] == 123
    assert body["options"]["num_ctx"] == 2048
    assert body["options"]["stop"] == ["\n\n"]
