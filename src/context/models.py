# src/context/models.py

"""
Modelos de dominio del sistema RAG.

Usan BaseModel con frozen=True porque son objetos de solo lectura:
  - Se crean durante la búsqueda y se consumen en el prompt.
  - No deben mutarse después de su creación.
  - frozen=True habilita hashing, lo que los hace usables en sets y como dict keys.
"""

from pydantic import BaseModel, ConfigDict


class ContextSource(BaseModel):
    """Referencia a una fuente de datos indexada."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_type: str
    source_name: str


class SearchResult(BaseModel):
    """
    Resultado de una búsqueda semántica.

    Combina el fragmento de texto recuperado con su metadata de origen
    y el score de similitud coseno con la query.
    """

    model_config = ConfigDict(frozen=True)

    score: float
    text: str
    source: str
    page: int
    collection: str
    chunk_index: int
