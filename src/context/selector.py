from typing import List


def match_namespace(
    pattern: str,
    available_contexts: List[str],
) -> List[str]:

    pattern = pattern.strip()

    if not pattern:
        return []

    # sociologia/*
    if pattern.endswith("/*"):

        namespace = pattern[:-2]

        return sorted(
            [ctx for ctx in available_contexts if ctx.startswith(namespace + "/")]
        )

    # exact match
    if pattern in available_contexts:
        return [pattern]

    return []
