"""
Estimación de tokens para el guard de contexto (src/llm/context_guard.py).

No hay un tokenizador exacto disponible para todos los backends (Qwen
vía FastFlowLM, DeepSeek vía Ollama, Gemini) sin traer 3 librerías
distintas. tiktoken (tokenizador de OpenAI, cl100k_base) da una
aproximación razonable para texto en inglés/español -- suele
sobre-contar un poco frente a tokenizadores tipo SentencePiece
(Qwen/Gemini), lo cual es el sesgo correcto para un guard PREVENTIVO
(mejor sobreestimar y avisar de más que subestimar y dejar pasar un
request que el modelo va a rechazar o cortar).

Si tiktoken no está instalado, fallback a una heurística de
caracteres/N (aproximación gruesa estándar para texto en inglés; para
español con más tildes/palabras largas puede subestimar un poco -- por
eso HEURISTIC_CHARS_PER_TOKEN queda en 3.5, no 4, para compensar).
"""

from __future__ import annotations

from src.utils.logger import logger

HEURISTIC_CHARS_PER_TOKEN = 3.5

try:
    import tiktoken

    _encoder: tiktoken.Encoding | None = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover -- tiktoken no instalado o falla al cargar
    _encoder = None
    logger.warning(
        "[tokens] tiktoken no disponible -- usando heurística de caracteres "
        "para estimar tokens (menos precisa, ver docstring del módulo)."
    )


def estimate_tokens(text: str) -> int:
    """
    Estima la cantidad de tokens de `text`. No es exacto para ningún
    backend en particular -- ver docstring del módulo para el porqué
    ese margen de error es aceptable (y deliberadamente conservador)
    para un guard preventivo de contexto.
    """
    if not text:
        return 0
    if _encoder is not None:
        return len(_encoder.encode(text, disallowed_special=()))
    return int(len(text) / HEURISTIC_CHARS_PER_TOKEN)
