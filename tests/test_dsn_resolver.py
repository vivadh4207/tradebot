import pytest
from src.storage.journal import resolve_cockroach_dsn


def test_dsn_env_used_as_is():
    env = {"COCKROACH_DSN": "postgresql://u:p@h:26257/d?sslmode=verify-full"}
    assert resolve_cockroach_dsn(env=env) == env["COCKROACH_DSN"]


def test_dsn_assembled_from_parts():
    env = {
        "COCKROACH_DSN": "",
        "COCKROACH_HOST": "free-tier.cockroachlabs.cloud",
        "COCKROACH_USER": "me",
        "COCKROACH_PASSWORD": "s3cret",
        "COCKROACH_DATABASE": "tradebot",
    }
    dsn = resolve_cockroach_dsn(env=env)
    assert dsn.startswith("postgresql://me:s3cret@free-tier.cockroachlabs.cloud:26257/tradebot?")
    assert "sslmode=verify-full" in dsn


def test_dsn_url_escapes_password_special_chars():
    env = {
        "COCKROACH_DSN": "",
        "COCKROACH_HOST": "h",
        "COCKROACH_USER": "u",
        "COCKROACH_PASSWORD": "a/b@c:d",    # chars that MUST be percent-encoded
        "COCKROACH_DATABASE": "tradebot",
    }
    dsn = resolve_cockroach_dsn(env=env)
    # raw special chars would corrupt the URL; escaped chars must appear
    assert "a/b@c:d" not in dsn
    assert "a%2Fb%40c%3Ad" in dsn


def test_dsn_missing_raises_with_guidance():
    env = {"COCKROACH_DSN": "", "COCKROACH_HOST": "", "COCKROACH_USER": "",
           "COCKROACH_PASSWORD": ""}
    with pytest.raises(RuntimeError) as excinfo:
        resolve_cockroach_dsn(env=env)
    assert "CockroachDB is not configured" in str(excinfo.value)


def test_dsn_placeholder_values_rejected():
    env = {"COCKROACH_DSN": "",
           "COCKROACH_HOST": "<your-cluster-host.cockroachlabs.cloud>",
           "COCKROACH_USER": "u", "COCKROACH_PASSWORD": "p"}
    with pytest.raises(RuntimeError) as excinfo:
        resolve_cockroach_dsn(env=env)
    assert "placeholder" in str(excinfo.value).lower()


def test_dsn_cluster_option_appended():
    env = {
        "COCKROACH_DSN": "",
        "COCKROACH_HOST": "h", "COCKROACH_USER": "u",
        "COCKROACH_PASSWORD": "p", "COCKROACH_DATABASE": "tradebot",
        "COCKROACH_CLUSTER": "happy-horse-42",
    }
    dsn = resolve_cockroach_dsn(env=env)
    assert "options=" in dsn and "happy-horse-42" in dsn
