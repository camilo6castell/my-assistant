"""
Tipos compartidos del módulo chat.
Definidos aquí para que session.py, search.py, generate.py y builder.py
los importen sin crear dependencias circulares.
"""

from typing import TypedDict


class TurnMemory(TypedDict):
    """Un turno del historial de conversación."""

    user: str
    assistant: str
