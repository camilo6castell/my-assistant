"""Tests for model capabilities registry (src/config/models/__init__.py)."""

import pytest

from src.config.models import (
    build_kwargs,
    get_context_window,
    get_default_think,
    get_supports,
)


class TestGetSupports:
    def test_returns_frozenset(self) -> None:
        result = get_supports("ollama", "qwen3.5:4b")
        assert isinstance(result, frozenset)

    def test_includes_extra(self) -> None:
        result = get_supports("ollama", "qwen3.5:4b")
        assert "extra" in result

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_supports("nonexistent", "model")


class TestGetContextWindow:
    def test_returns_int_or_none(self) -> None:
        result = get_context_window("ollama", "qwen3.5:4b")
        assert result is None or isinstance(result, int)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_context_window("nonexistent", "model")


class TestGetDefaultThink:
    def test_returns_bool_or_none(self) -> None:
        result = get_default_think("ollama", "qwen3.5:4b")
        assert result is None or isinstance(result, bool)

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_default_think("nonexistent", "model")


class TestBuildKwargs:
    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            build_kwargs("nonexistent", "model", [])
