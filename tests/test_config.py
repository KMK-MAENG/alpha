import pytest

from alpha import config


def test_get_env_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("ALPHA_TEST_KEY", "test-value")
    assert config.get_env("ALPHA_TEST_KEY") == "test-value"


def test_get_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ALPHA_TEST_MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPHA_TEST_MISSING_KEY"):
        config.get_env("ALPHA_TEST_MISSING_KEY")
