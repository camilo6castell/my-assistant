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

import tiktoken

from src.utils.logger import logger

HEURISTIC_CHARS_PER_TOKEN = 3.5

# Anotación sin asignar: le da a mypy el tipo de _encoder de entrada
# (Encoding | None) sin forzar una asignación inicial redundante --
# ambas ramas del try/except de abajo asignan un valor compatible con
# ese tipo. Sin esto, mypy infiere el tipo a partir de la primera
# asignación (tiktoken.get_encoding(...) -> Encoding) y se queja al
# asignar None en el except (error: typeddict/assignment).
_encoder: tiktoken.Encoding | None

try:
    # tiktoken es dependencia dura (ver pyproject.toml), así que el
    # import en sí no falla -- lo que puede fallar es get_encoding():
    # la primera vez que corre, descarga el archivo de encoding desde
    # una URL externa (no vendorizado), lo cual falla sin acceso de red
    # a ese host puntual. Por eso el try/except envuelve la llamada, no
    # el import.
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover -- típicamente sin red hacia el host de descarga
    _encoder = None
    logger.warning(
        "[tokens] tiktoken no pudo cargar su encoding (sin red hacia el host "
        "de descarga) -- usando heurística de caracteres para estimar tokens "
        "(menos precisa, ver docstring del módulo)."
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
