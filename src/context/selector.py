"""
Resolves context patterns against the list of available collections.

Supported patterns per token:
  sociology           → all collections under sociology/
  sociology/*         → same (explicit alias)
  sociology/debord    → exact collection

match_contexts accepts multiple space-separated tokens,
resolves each one and returns the sorted union without duplicates.
"""


def _match_single(pattern: str, available: list[str]) -> list[str]:
    """Resolves a single token against the list of collections."""
    pattern = pattern.strip()

    if not pattern:
        return []

    # "sociology/*"  →  explicit namespace
    if pattern.endswith("/*"):
        namespace = pattern[:-2]
        return [ctx for ctx in available if ctx.startswith(namespace + "/")]

    # "sociology/debord"  →  exact match
    if "/" in pattern:
        return [pattern] if pattern in available else []

    # "sociology"  →  implicit namespace
    prefix = pattern + "/"
    matches = [ctx for ctx in available if ctx.startswith(prefix)]

    if not matches and pattern in available:
        return [pattern]

    return matches


def match_namespace(
    pattern: str,
    available_contexts: list[str],
) -> list[str]:
    """
    Backward compatibility with the previous contract: resolves a single pattern.
    Used by ContextManager.
    """
    return sorted(_match_single(pattern, available_contexts))


def match_contexts(
    raw: str,
    available_contexts: list[str],
) -> list[str]:
    """
    Resolves multiple space-separated tokens.

    Examples:
        "sociology"                               → sociology/*
        "sociology react"                         → sociology/* + react/*
        "sociology/debord react"                  → exact collection + react/*
        "sociology/debord react/hooks psychology" → mix of exact and namespace

    Returns the sorted union without duplicates.
    """
    tokens = raw.strip().split()

    if not tokens:
        return []

    seen: set[str] = set()
    result: list[str] = []

    for token in tokens:
        for ctx in _match_single(token, available_contexts):
            if ctx not in seen:
                seen.add(ctx)
                result.append(ctx)

    return sorted(result)
