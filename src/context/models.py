"""
Domain models for the RAG system.

They use BaseModel with frozen=True because they are read-only objects:
  - Created during retrieval and consumed in the prompt.
  - Must not be mutated after creation.
  - frozen=True enables hashing, making them usable in sets and as dict keys.
"""

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    """
    Result of a semantic search.

    Combines the retrieved text fragment with its source metadata
    and the cosine similarity score with the query.
    """

    model_config = ConfigDict(frozen=True)

    score: float
    text: str
    source: str
    page: int
    collection: str
    chunk_index: int
