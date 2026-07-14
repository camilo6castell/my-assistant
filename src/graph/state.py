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
  review_passed    True si review_node aprobó la respuesta (o no se revisó)
  review_feedback  motivo del rechazo, usado para regenerar con corrección
  review_attempts  cuántas veces se regeneró tras un rechazo del reviewer
                   (evita loops infinitos: ver MAX_REVIEW_ATTEMPTS)
  max_tokens/think_mode/extra
                   overrides de generación por-request (ver
                   GenerationOptions en src/api/schemas/chat.py); None
                   en todos = comportamiento actual sin cambios. La
                   temperatura NO vive acá -- es una propiedad fija de
                   cada modelo (src/config/models/<backend>.py), nunca
                   un override por-request.

                   No hay top_k_initial/top_k_final ni max_turns acá:
                   dejaron de ser overrides por-request (ver
                   GenerationOptions en src/api/schemas/chat.py) --
                   retrieve_node/generate_node los resuelven
                   directamente contra settings, sin pasar por el
                   estado del grafo. Si en algún momento hace falta
                   reexponerlos, existieron acá antes y se sacaron
                   deliberadamente -- ver el historial de este archivo
                   antes de reinventar la rueda.
  attachments      archivos adjuntos ad-hoc (ver
                   src/context/attachments.py), como lista de
                   (filename, content). Solo generate_node los inyecta
                   en el prompt final (ver inject_attachments() en
                   src/prompts/builder.py) -- retrieve_node y
                   reformulate_node usan `question` sin adjuntos, para
                   no ensuciar el embedding de búsqueda ni la
                   reformulación con contenido de archivo.

NOTA: este módulo NO usa `from __future__ import annotations`.
LangGraph llama get_type_hints(RAGState) en runtime para inspeccionar
los campos del estado. Con annotations postponed, todos los tipos se
convierten en strings lazy y get_type_hints() falla al resolver
LoadedCollection si está bajo TYPE_CHECKING (NameError en runtime).

La solución es importar LoadedCollection directamente —sin guard—
para que exista en el namespace del módulo cuando LangGraph lo evalúa.
"""

from typing import Any, TypedDict

from src.cli.types import TurnMemory
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
    review_passed: bool
    review_feedback: str
    review_attempts: int
    # Overrides de generación por-request (ver GenerationOptions en
    # src/api/schemas/chat.py). None = usar el default de settings/.env.
    # Solo generate_node/correct_node los leen -- reformulate_node y
    # review_node siempre usan la temperatura por defecto, son tareas
    # internas de una sola pasada, no la respuesta final al usuario.
    max_tokens: int | None
    think_mode: bool | None
    # Passthrough genérico sin validar (ver GenerationOptions.extra) --
    # dict[str, Any] es la única excepción deliberada al tipado estricto
    # del resto del proyecto: por definición puede contener cualquier
    # parámetro propio de un provider que el backend no modela.
    extra: dict[str, Any] | None
    attachments: list[tuple[str, str]]


class RAGStateUpdate(TypedDict, total=False):
    """
    Actualización parcial de RAGState.

    Cada nodo del grafo (src/graph/nodes.py) devuelve solo el subconjunto
    de campos que modifica -- LangGraph aplica el resto como merge parcial
    sobre el estado existente (ver docstring del módulo). `total=False`
    modela justamente eso: todos los campos son opcionales en el dict de
    retorno, pero cada uno que sí esté presente queda tipado igual que en
    RAGState, en vez de perder precisión con `dict[str, object]`.
    """

    question: str
    mode: str
    collections: list[LoadedCollection]
    chat_memory: list[TurnMemory]
    results: list[SearchResult]
    confidence: float
    reformulated: bool
    answer: str
    review_passed: bool
    review_feedback: str
    review_attempts: int
    max_tokens: int | None
    think_mode: bool | None
    extra: dict[str, Any] | None
    attachments: list[tuple[str, str]]
