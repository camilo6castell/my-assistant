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

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración del sistema RAG.

    Pydantic-settings lee las variables de entorno (y el archivo .env)
    de forma automática. El nombre del campo en snake_case se mapea
    al env var en UPPER_CASE (ej: chunk_size -> CHUNK_SIZE).
    """

    model_config = SettingsConfigDict(
        # env_file=".env",
        env_file=(".env", ".env.providers"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Root
    ai_home: Path = Field(default=Path("/srv/ai"))

    # Paths: Los paths derivados son propiedades calculadas
    @property
    def data_path(self) -> Path:
        return self.ai_home / "data"

    @property
    def vector_store_path(self) -> Path:
        return self.ai_home / "vector_stores"

    @property
    def log_path(self) -> Path:
        return self.ai_home / "logs"

    # ======================================================
    # EMBEDDING BACKEND
    # ======================================================
    # EMBEDDING_BACKEND controla qué runtime genera los embeddings:
    #
    #   sentence_transformers  → carga el modelo localmente en Python
    #                            (CPU o GPU según disponibilidad de torch).
    #                            Sin servidor externo. Default.
    #
    #   ollama                 → petición HTTP a Ollama (GPU Vulkan).
    #                            Requiere: EMBEDDING_BASE_URL, EMBEDDING_MODEL.
    #
    #   fastflowlm             → petición HTTP a FastFlowLM (NPU).
    #                            Requiere: EMBEDDING_BASE_URL, EMBEDDING_MODEL.
    #                            Usa el endpoint OpenAI-compatible /v1/embeddings.
    #
    # IMPORTANTE: cambiar backend o modelo invalida los índices existentes.
    # Los vectores de backends distintos viven en rutas separadas:
    #   /srv/ai/vector_stores/<backend>/<model_safe>/<category>/<collection>/
    # donde model_safe reemplaza '/' por '_' para evitar subdirectorios.

    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_backend: str = Field(default="sentence_transformers")
    embedding_base_url: str = Field(default="http://127.0.0.1:11434")
    embedding_api_key: str = Field(default="ollama")

    @property
    def embedding_model_safe(self) -> str:
        """Nombre del modelo sanitizado para usar como directorio."""
        return self.embedding_model.replace("/", "_").replace(":", "_")

    @property
    def vector_store_path_for_backend(self) -> Path:
        """
        Ruta base que incluye backend y modelo:
        /srv/ai/vector_stores/<backend>/<model_safe>/

        Los índices de backends/modelos distintos son incompatibles
        (viven en espacios vectoriales distintos), por eso se aíslan
        en subdirectorios separados en vez de mezclarlos.
        Las carpetas se crean automáticamente al primer uso en get_collection_paths().
        """
        return (
            self.ai_home
            / "vector_stores"
            / self.embedding_backend
            / self.embedding_model_safe
        )

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

    # Web ingest
    max_pages: int = Field(default=50, gt=0)
    delay: float = Field(default=1.0, ge=0.0)

    # Retrieval
    hard_top_k_initial: int = Field(default=15, gt=0)
    hard_top_k_final: int = Field(default=5, gt=0)
    soft_top_k_initial: int = Field(default=25, gt=0)
    soft_top_k_final: int = Field(default=7, gt=0)

    max_turns: int = Field(default=4, gt=0)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout: int = Field(default=600, gt=0)

    # ======================================================
    # MULTI-PROVIDER LLM
    # ======================================================
    # Arquitectura "provider-per-node": cada nodo del grafo puede usar un
    # proveedor distinto (local, gemini, y los que se agreguen después)
    # sin tocar graph.py ni nodes.py. Ver src/llm/providers.py.
    #
    # local  -> modelo en runtime local (FastFlowLM/Ollama, OpenAI-compatible)
    # gemini -> Gemini vía endpoint OpenAI-compatible de Google
    local_base_url: str = Field(default="")
    local_api_key: str = Field(default="")
    local_model: str = Field(default="")
    # "openai_compat" | "ollama_native" -- qué implementación de
    # LLMClient usar (ver src/llm/backends/).
    local_client: str = Field(default="openai_compat")
    # Qué archivo de src/config/models/ mirar para las capacidades reales
    # de local_model (temperature, max_tokens, think mode y cómo
    # activarlo). El "supports" que antes vivía acá como string plana
    # ahora se DERIVA de esa estructura tipada -- no puede desincronizarse.
    local_capabilities: str = Field(default="fastflowlm")

    gemini_base_url: str = Field(default="")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="")
    gemini_client: str = Field(default="openai_compat")
    gemini_capabilities: str = Field(default="gemini")

    # Qué proveedor usa cada nodo del grafo. Configurable en .env,
    # sin tocar código — esto es lo que hace la arquitectura extensible.
    reformulate_provider: str = Field(default="gemini")
    generate_provider: str = Field(default="local")

    # Agent — umbral de foco temático para el grafo LangGraph.
    # Con 1 colección mide spread de chunk_index (menor = match).
    # Con N colecciones mide source dominance (mayor = match).
    # Sobreescribible en .env: CONFIDENCE_LIMIT=0.20
    confidence_limit: float = Field(default=0.79, ge=0.0, le=1.0)

    # API
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, gt=0)

    # ======================================================
    # WEB SEARCH (Tavily)
    # ======================================================
    # Búsqueda web como fuente de retrieval complementaria/alternativa a
    # las colecciones locales (ver QueryRequest.web_search en
    # src/api/schemas/chat.py y src/retrieval/web_search.py).
    #
    # web_search_enabled es un kill-switch global e independiente de si
    # se manda web_search=True en el request: permite desactivar la
    # feature entera en un ambiente (ej. sin salida a internet, o para
    # no generar costo en la API de Tavily) sin tocar el frontend ni el
    # código -- el router devuelve 400 si se pide web_search=True con
    # esto en False.
    web_search_enabled: bool = Field(default=False)
    tavily_api_key: str = Field(default="")
    web_search_timeout: int = Field(default=15, gt=0)
    web_search_max_results: int = Field(default=5, gt=0)


# Instancia singleton — se valida al importar el módulo.
settings = Settings()
