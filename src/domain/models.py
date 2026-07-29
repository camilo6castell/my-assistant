"""
Shared domain models used across CLI, API, graph, and LLM layers.

TurnMemory: a single conversation turn (frozen, append-only history).
ChatMode:   SOFT/HARD retrieval mode (StrEnum for CLI ergonomics).
LLMRole:    identifies a pipeline point that calls an LLM.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TurnMemory(BaseModel):
    """
    A single turn in the conversation history.

    frozen=True because turns are immutable once recorded:
    history is an append-only list in ChatSession.
    """

    model_config = ConfigDict(frozen=True)

    user: str
    assistant: str


class ChatMode(StrEnum):
    SOFT = "SOFT"
    HARD = "HARD"


class LLMRole(StrEnum):
    """A pipeline point that needs an LLM."""

    GENERATE = "generate"
    WEB_SUPPLEMENT = "web_supplement"
