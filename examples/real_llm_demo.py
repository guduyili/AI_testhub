import os

from AI_testhub.real_config import load_real_model_config, parse_bool_env


def test_parse_bool_env_accepts_common_truthy_and_falsey_values(monkeypatch):
    monkeypatch.setenv("BROWSER_HEADLESS", "false")
    assert parse_bool_env("BROWSER_HEADLESS", default=True) is False

    monkeypatch.setenv("BROWSER_HEADLESS", "1")
    assert parse_bool_env("BROWSER_HEADLESS", default=False) is True

    monkeypatch.delenv("BROWSER_HEADLESS")
    assert parse_bool_env("BROWSER_HEADLESS", default=True) is True


def test_load_real_model_config_reads_openai_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_PROVIDER", "openai-compatible")

    config = load_real_model_config()

    assert config.name == "real-env"
    assert config.model_type == "openai-compatible"
    assert config.model_name == "gpt-4o-mini"
    assert config.api_key == "test-key"
    assert config.base_url == "https://api.example.test/v1"


def test_load_real_model_config_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        load_real_model_config()
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected missing OPENAI_API_KEY error")
