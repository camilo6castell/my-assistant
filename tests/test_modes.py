"""Tests for ChatMode enum (src/cli/modes.py)."""

from src.cli.modes import ChatMode


class TestChatMode:
    def test_values(self) -> None:
        assert ChatMode.SOFT.value == "SOFT"
        assert ChatMode.HARD.value == "HARD"

    def test_is_str(self) -> None:
        assert isinstance(ChatMode.SOFT, str)
        assert ChatMode.SOFT + "_modified" == "SOFT_modified"

    def test_from_value(self) -> None:
        assert ChatMode("SOFT") is ChatMode.SOFT
        assert ChatMode("HARD") is ChatMode.HARD

    def test_invalid_value(self) -> None:
        try:
            ChatMode("INVALID")
            raise AssertionError("Should have raised")
        except ValueError:
            pass
