"""
Estado compartido del grafo LangGraph.

RAGState es el único objeto que viaja entre nodos. LangGraph lo pasa
como argumento a cada nodo y aplica el dict de retorno como un merge
parcial — solo los campos devueltos se actualizan, el resto se conserva.

Campos:
  question       pregunta original del usuario
  mode           ChatMode.SOFT | ChatMode.HARD
  collections    colecciones FAISS cargadas en memoria
  chat_memory    historial de turnos (ventana deslizante en generate.py)
  results        chunks recuperados por retrieve_node
  confidence     score promedio de los resultados
  reformulated   True si ya se reformuló la query en esta ejecución
                 (evita loops infinitos en el grafo)
  answer         respuesta final del LLM

NOTA: este módulo NO usa `from __future__ import annotations`.
LangGraph llama get_type_hints(RAGState) en runtime para inspeccionar
los campos del estado. Con annotations postponed, todos los tipos se
convierten en strings lazy y get_type_hints() falla al resolver
LoadedCollection si está bajo TYPE_CHECKING (NameError en runtime).

La solución es importar LoadedCollection directamente —sin guard—
para que exista en el namespace del módulo cuando LangGraph lo evalúa.
"""

from typing import TypedDict

from src.chat.types import TurnMemory
from src.context.manager import LoadedCollection
from src.context.models import SearchResult


class RAGState(TypedDict):
    question: str
    mode: str
    collections: list[LoadedCollection]
    chat_memory: list[TurnMemory]
    results: list[SearchResult]
    confidence: float
    reformulated: bool
    answer: str
