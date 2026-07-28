# src/chat/types.py

"""
Shared types for the chat module.

Defined in their own module so that session.py, generate.py, and
builder.py can import them without creating circular dependencies.
"""

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
