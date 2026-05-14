from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextSource:
    source_id: str
    source_type: str
    source_name: str


@dataclass
class SearchResult:
    score: float
    text: str
    source: str
    page: int
    collection: str
    chunk_index: int
