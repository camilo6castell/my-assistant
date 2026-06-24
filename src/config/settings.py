"""
Configuración centralizada del sistema usando pydantic-settings.

Ventajas sobre os.getenv manual:
  - Validación de tipos en el arranque (falla rápido si el env está mal)
  - Constraints en campos (gt=0, ge=0, le=2.0)
  - Validación cruzada entre campos (@field_validator)
  - Lectura automática de .env sin llamar load_dotenv()
  - Un solo objeto `settings` como fuente de verdad

Los exports al final del módulo mantienen compatibilidad con todos los
archivos que ya importan directamente (ej: from src.config.settings import CHUNK_SIZE).
"""

from pathlib import Path

from pydantic import Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración del sistema RAG.

    Pydantic-settings lee las variables de entorno (y el archivo .env)
    de forma automática. El nombre del campo en snake_case se mapea
    al env var en UPPER_CASE (ej: chunk_size -> CHUNK_SIZE).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Root
    ai_home: Path = Field(default=Path("/srv/ai"))

    # Paths
    data_path: Path = Field(default=Path("/srv/ai/data"))
    vector_store_path: Path = Field(default=Path("/srv/ai/vector_stores"))
    log_path: Path = Field(default=Path("/srv/ai/logs"))
    hf_home: Path = Field(default=Path("/srv/ai/hf"))

    # Embeddings
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")

    # Chunking
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info: ValidationInfo) -> int:
        """
        Garantiza que el overlap no sea >= al tamaño del chunk.
        Sin esta validación, chunk_text() produciría un loop infinito.
        """
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError(
                f"chunk_overlap ({v}) debe ser menor que "
                f"chunk_size ({info.data['chunk_size']})"
            )
        return v

    # Retrieval
    hard_top_k_initial: int = Field(default=15, gt=0)
    hard_top_k_final: int = Field(default=5, gt=0)
    soft_top_k_initial: int = Field(default=25, gt=0)
    soft_top_k_final: int = Field(default=7, gt=0)
    max_turns: int = Field(default=4, gt=0)
    default_soft_mode: bool = Field(default=False)

    # LLM
    llm_provider: str = Field(default="fastflowlm")
    llm_base_url: str = Field(default="http://127.0.0.1:52625/v1")
    llm_api_key: str = Field(default="flm")
    llm_model: str = Field(default="qwen3-it:4b")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout: int = Field(default=600, gt=0)

    # Web ingest
    max_pages: int = Field(default=50, gt=0)
    delay: float = Field(default=1.0, ge=0.0)


# Instancia singleton — se valida al importar el módulo.
settings = Settings()

# ======================================================
# EXPORTS DE COMPATIBILIDAD
# Permiten que los archivos existentes sigan importando sin cambios:
#   from src.config.settings import CHUNK_SIZE
# ======================================================

# AI_HOME: Path = settings.ai_home
# HF_HOME: Path = settings.hf_home

# DEFAULT_SOFT_MODE: bool = settings.default_soft_mode

# LLM_PROVIDER: str = settings.llm_provider
