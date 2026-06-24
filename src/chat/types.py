# src/chat/types.py

"""
Tipos compartidos del módulo chat.

Definidos en un módulo propio para que session.py, generate.py y
builder.py los importen sin crear dependencias circulares.
"""

from pydantic import BaseModel, ConfigDict


class TurnMemory(BaseModel):
    """
    Un turno del historial de conversación.

    frozen=True porque los turnos son inmutables una vez registrados:
    el historial es una lista append-only en ChatSession.
    """

    model_config = ConfigDict(frozen=True)

    user: str
    assistant: str
