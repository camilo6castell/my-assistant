"""
Token estimation for the context guard (src/llm/context_guard.py).

There is no exact tokenizer available for all backends (Qwen via
FastFlowLM, DeepSeek via Ollama, Gemini) without bringing in 3 different
libraries. tiktoken (OpenAI tokenizer, cl100k_base) provides a reasonable
approximation for English/Spanish text -- it tends to over-count slightly
compared to SentencePiece-style tokenizers (Qwen/Gemini), which is the
correct bias for a PREVENTIVE guard (better to overestimate and warn than
to underestimate and let through a request the model will reject or crop).

If tiktoken is not installed, falls back to a characters/N heuristic
(coarse standard approximation for English text; for Spanish with more
accents/long words it may underestimate slightly -- that is why
HEURISTIC_CHARS_PER_TOKEN is 3.5, not 4, to compensate).
"""

from __future__ import annotations

import tiktoken

from src.utils.logger import logger

HEURISTIC_CHARS_PER_TOKEN = 3.5

# Annotation without assignment: gives mypy the type of _encoder upfront
# (Encoding | None) without forcing a redundant initial assignment --
# both branches of the try/except below assign a value compatible with
# that type. Without this, mypy infers the type from the first
# assignment (tiktoken.get_encoding(...) -> Encoding) and complains when
# assigning None in the except (error: typeddict/assignment).
_encoder: tiktoken.Encoding | None

try:
    # tiktoken is a hard dependency (see pyproject.toml), so the import
    # itself does not fail -- what can fail is get_encoding(): the first
    # time it runs, it downloads the encoding file from an external URL
    # (not vendored), which fails without network access to that specific
    # host. That is why the try/except wraps the call, not the import.
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover -- typically no network to download host
    _encoder = None
    logger.warning(
        "[tokens] tiktoken could not load its encoding (no network to download "
        "host) -- using character heuristic to estimate tokens "
        "(less accurate, see module docstring)."
    )


def estimate_tokens(text: str) -> int:
    """
    Estimates the number of tokens in `text`. Not exact for any
    particular backend -- see module docstring for why that margin
    of error is acceptable (and deliberately conservative) for a
    preventive context guard.
    """
    if not text:
        return 0
    if _encoder is not None:
        return len(_encoder.encode(text, disallowed_special=()))
    return int(len(text) / HEURISTIC_CHARS_PER_TOKEN)
