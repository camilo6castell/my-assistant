from dataclasses import dataclass


@dataclass(slots=True)
class ContextSource:
    source_id: str
    source_type: str
    source_name: str


@dataclass(slots=True)
class SearchResult:
    score: float
    text: str
    source: str
    page: int
    collection: str
    chunk_index: int
