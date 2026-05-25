"""
Resuelve patrones de contexto contra la lista de colecciones disponibles.

Patrones soportados por token:
  sociologia          → todas las colecciones bajo sociologia/
  sociologia/*        → ídem (alias explícito)
  sociologia/debord   → colección exacta

match_contexts acepta múltiples tokens separados por espacio,
resuelve cada uno y devuelve la unión ordenada sin duplicados.
"""


def _match_single(pattern: str, available: list[str]) -> list[str]:
    """Resuelve un único token contra la lista de colecciones."""
    pattern = pattern.strip()

    if not pattern:
        return []

    # "sociologia/*"  →  namespace explícito
    if pattern.endswith("/*"):
        namespace = pattern[:-2]
        return [ctx for ctx in available if ctx.startswith(namespace + "/")]

    # "sociologia/debord"  →  match exacto
    if "/" in pattern:
        return [pattern] if pattern in available else []

    # "sociologia"  →  namespace implícito
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
    Compatibilidad con el contrato anterior: resuelve un único patrón.
    Usado por ContextManager.
    """
    return sorted(_match_single(pattern, available_contexts))


def match_contexts(
    raw: str,
    available_contexts: list[str],
) -> list[str]:
    """
    Resuelve múltiples tokens separados por espacio.

    Ejemplos:
        "sociologia"                               → sociologia/*
        "sociologia react"                         → sociologia/* + react/*
        "sociologia/debord react"                  → colección exacta + react/*
        "sociologia/debord react/hooks psicologia" → mezcla de exactos y namespaces

    Retorna la unión ordenada sin duplicados.
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
