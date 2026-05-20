"""
src/context/selector.py

Resuelve patrones de contexto contra la lista de colecciones disponibles.

Patrones soportados por token:
  sociologia          → todas las colecciones bajo sociologia/
  sociologia/*        → ídem (alias explícito)
  sociologia/debord   → colección exacta

match_contexts acepta múltiples tokens separados por espacio,
resuelve cada uno y devuelve la unión ordenada sin duplicados.
"""

from typing import List


def _match_single(pattern: str, available: List[str]) -> List[str]:
    """Resuelve un único token contra la lista de colecciones."""

    pattern = pattern.strip()

    if not pattern:
        return []

    # "sociologia/*"  →  namespace explícito
    if pattern.endswith("/*"):
        namespace: str = pattern[:-2]
        return [ctx for ctx in available if ctx.startswith(namespace + "/")]

    # "sociologia/debord"  →  match exacto
    if "/" in pattern:
        return [pattern] if pattern in available else []

    # "sociologia"  →  namespace implícito (azúcar sintáctico)
    prefix: str = pattern + "/"
    matches: List[str] = [ctx for ctx in available if ctx.startswith(prefix)]

    # Si no hay matches de namespace, intenta match exacto de todas formas
    # (por si alguien tiene una colección raíz sin subdirectorio)
    if not matches and pattern in available:
        return [pattern]

    return matches


def match_namespace(
    pattern: str,
    available_contexts: List[str],
) -> List[str]:
    """
    Compatibilidad con el contrato anterior: resuelve un único patrón.
    Sigue funcionando igual que antes para el ContextManager.
    """
    return sorted(_match_single(pattern, available_contexts))


def match_contexts(
    raw: str,
    available_contexts: List[str],
) -> List[str]:
    """
    Resuelve múltiples tokens separados por espacio.

    Ejemplos:
        "sociologia"                              → sociologia/*
        "sociologia react"                        → sociologia/* + react/*
        "sociologia/debord react"                 → colección exacta + react/*
        "sociologia/debord react/hooks psicologia" → mezcla de exactos y namespaces

    Retorna la unión ordenada sin duplicados.
    """
    tokens: List[str] = raw.strip().split()

    if not tokens:
        return []

    seen: set[str] = set()
    result: List[str] = []

    for token in tokens:
        for ctx in _match_single(token, available_contexts):
            if ctx not in seen:
                seen.add(ctx)
                result.append(ctx)

    return sorted(result)
