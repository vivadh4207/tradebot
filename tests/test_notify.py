import os
import time
import threading
from unittest.mock import patch

from src.notify.base import NullNotifier, build_notifier
from src.notify.webhook import WebhookNotifier


def test_null_notifier_silent():
    n = NullNotifier()
    # must not raise
    n.notify("hello")
    n.notify("err", level="error", title="x")


def test_build_notifier_without_env_returns_null(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    n = build_notifier()
    assert isinstance(n, NullNotifier)


def test_build_notifier_discord_when_env_set(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/fake")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    n = build_notifier()
    assert isinstance(n, WebhookNotifier)
    assert n._flavor == "discord"
    n.close()


def test_webhook_payload_shape(monkeypatch):
    posted = {}

    class FakeResp:
        def read(self):
            return b""

    def fake_urlopen(req, timeout):
        posted["url"] = req.full_url
        posted["data"] = req.data
        return FakeResp()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        n = WebhookNotifier(url="https://hook.test/abc", flavor="discord")
        n.notify("hello world", title="entry")
        time.sleep(0.2)
        n.close()
    assert posted["url"] == "https://hook.test/abc"
    assert b"hello world" in posted["data"]
    assert b"entry" in posted["data"]


def test_webhook_drops_under_overflow():
    # queue_size=2 so 10 messages don't all make it in; verifies no raise
    n = WebhookNotifier(url="https://hook.test/abc", flavor="slack", queue_size=2)
    try:
        for i in range(10):
            n.notify(f"m{i}")
        # should not have raised even though queue was full
    finally:
        n.close()
