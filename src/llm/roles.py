"""
Roles de LLM: cada punto del pipeline que llama a un modelo se identifica
con uno de estos roles. El provider que atiende cada rol se configura de
forma independiente (ver Settings.provider_for en src/config/settings.py)
-- antes reformular, revisar, y evaluar el complemento web compartían el
mismo "reformulate_provider" sin ninguna razón salvo que nunca se
separaron; ahora cada rol tiene su propia env var, así se puede, por
ejemplo, correr la respuesta final en un modelo local potente y las
tareas cortas de apoyo en un modelo rápido en la nube, sin acoplar unas
con otras.

Módulo separado, sin importar nada del resto del proyecto (ni siquiera
settings), específicamente para poder importarse tanto desde
src/config/settings.py como desde los call sites (src/llm/generate.py,
src/graph/nodes.py, src/api/routers/chat.py, src/chat/interface.py) sin
riesgo de import circular -- settings.py es de las cosas más
tempranamente importadas del proyecto.
"""

from enum import Enum


class LLMRole(str, Enum):
    """Un punto del pipeline que necesita un LLM."""

    # Genera la respuesta final de RAG -- generate_node/correct_node en
    # el grafo, el pipeline lineal de /query, y el chat directo del CLI.
    GENERATE = "generate"
    # Reescribe la pregunta del usuario para mejorar el retrieval cuando
    # la confidence inicial es baja -- reformulate_node.
    REFORMULATE = "reformulate"
    # Evalúa si la respuesta generada pasa control de calidad (grounding,
    # atribución de fuentes) -- review_node.
    REVIEW = "review"
    # Decide si una búsqueda web le agrega algo genuinamente nuevo a una
    # respuesta ya generada desde contexto local -- ver
    # _supplement_with_web en src/api/routers/chat.py.
    WEB_SUPPLEMENT = "web_supplement"
