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

from src.nlp.llm.roles import LLMRole


def _split_backend_model(raw: str, var_name: str) -> tuple[str, str]:
    """
    Parsea el formato "backend,modelo" que usan EMBEDDER y LLM_ROL_* en
    .env.providers (ej. "flm,qwen3.5:9b" -> ("flm", "qwen3.5:9b")).

    El split es solo en la PRIMERA coma: el nombre del modelo puede traer
    ':' legítimamente (tags de Ollama/FastFlowLM, ej. "qwen3.5:9b"), así
    que nunca se parte por eso.
    """
    backend, sep, model = raw.partition(",")
    backend, model = backend.strip(), model.strip()
    if not sep or not backend or not model:
        raise ValueError(
            f"{var_name} debe tener el formato 'backend,modelo' "
            f"(ej. 'ollama,bge-m3'). Valor actual: {raw!r}"
        )
    return backend, model


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
    # BACKENDS -- URLs de runtime (.env.providers)
    # ======================================================
    # Un backend es un runtime concreto (FastFlowLM, Ollama, Gemini).
    # Acá solo vive CÓMO conectarse a cada uno (URL + credencial); QUÉ
    # modelo de ese backend usa cada rol se decide más abajo, en
    # EMBEDDER / LLM_ROL_* -- nunca en esta sección.
    #
    # flm/ollama no requieren API key real (runtimes locales); el campo
    # existe solo porque el SDK de OpenAI exige un string no vacío -- ver
    # Settings.embedding_api_key y providers._backend_api_key().
    llm_flm_url: str = Field(default="")
    llm_ollama_url: str = Field(default="")
    llm_gemini_url: str = Field(default="")
    gemini_api_key: str = Field(default="")

    embedder_flm_url: str = Field(default="")
    embedder_ollama_url: str = Field(default="")

    # ======================================================
    # EMBEDDER -- qué backend + modelo generan los embeddings
    # ======================================================
    # Formato "backend,modelo" (ver _split_backend_model). Backends
    # válidos: sentence_transformers (en proceso, sin URL) | ollama |
    # flm. Agregar un modelo nuevo a un backend existente es una entrada
    # en src/config/models/<backend>.py -- nunca una variable de entorno
    # nueva.
    #
    # IMPORTANTE: cambiar backend o modelo invalida los índices
    # existentes. Los índices se guardan en rutas separadas por
    # backend/modelo:
    #   /srv/ai/vector_stores/<backend>/<model_safe>/<categoria>/<coleccion>/
    # donde model_safe reemplaza '/' y ':' por '_'.
    embedder: str = Field(default="ollama,bge-m3")

    @field_validator(
        "embedder",
        "llm_rol_generate",
        "llm_rol_reformulate",
        "llm_rol_review",
        "llm_rol_supplement",
        mode="after",
    )
    @classmethod
    def _validate_backend_model_format(cls, v: str, info: ValidationInfo) -> str:
        """Falla rápido en el arranque si EMBEDDER/LLM_ROL_* no vienen 'backend,modelo'."""
        if info.field_name is not None:
            _split_backend_model(v, info.field_name.upper())
        return v

    @property
    def embedder_backend(self) -> str:
        return _split_backend_model(self.embedder, "EMBEDDER")[0]

    @property
    def embedder_model(self) -> str:
        return _split_backend_model(self.embedder, "EMBEDDER")[1]

    # Alias retrocompatibles -- ingest/core.py, cli/commands.py,
    # context/*.py y nlp/embedders/encoder.py ya conocían estos nombres
    # desde antes de la simplificación de .env.providers; se mantienen
    # como la interfaz pública de "qué backend/modelo generan embeddings"
    # para no tener que tocar esos call sites.
    @property
    def embedding_backend(self) -> str:
        return self.embedder_backend

    @property
    def embedding_model(self) -> str:
        return self.embedder_model

    @property
    def embedding_model_safe(self) -> str:
        """Nombre del modelo sanitizado para usar como directorio."""
        return self.embedding_model.replace("/", "_").replace(":", "_")

    @property
    def embedding_base_url(self) -> str:
        """
        URL HTTP del backend de embeddings activo.

        Solo tiene sentido para backends HTTP (ollama/flm) -- llamar esto
        con EMBEDDER=sentence_transformers,... es un error del caller
        (ese backend corre en proceso, sin URL) y se señaliza como tal.
        """
        urls = {"flm": self.embedder_flm_url, "ollama": self.embedder_ollama_url}
        backend = self.embedding_backend
        try:
            return urls[backend]
        except KeyError:
            raise ValueError(
                f"El backend de embeddings {backend!r} no usa URL HTTP "
                f"(¿EMBEDDER=sentence_transformers,...? ese backend no tiene base_url)."
            ) from None

    @property
    def embedding_api_key(self) -> str:
        # Ni Ollama ni FastFlowLM validan esta key -- el SDK de OpenAI
        # simplemente exige un string no vacío para construirse.
        return "not-needed"

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
        return self.ai_home / "vector_stores" / self.embedding_backend / self.embedding_model_safe

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
                f"chunk_overlap ({v}) debe ser menor que chunk_size ({info.data['chunk_size']})"
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
    llm_timeout: int = Field(default=600, gt=0)
    # Diagnóstico opt-in: vuelca a ./debug_last_llm_request.json el
    # request EXACTO (mensajes + extra_fields aplanados) que se le manda
    # al provider en cada llamada -- ver _dump_request_for_debug en
    # src/llm/generate.py. Pensado para reproducir con curl un fallo que
    # depende del tamaño/contenido real del prompt (ej. RAG con muchos
    # chunks) sin reconstruirlo a mano. Default False: nunca escribe
    # archivos en uso normal.
    llm_debug_dump: bool = Field(default=False)

    # ======================================================
    # LLM POR ROL -- qué backend + modelo atiende cada rol
    # ======================================================
    # Cada punto del pipeline que llama a un LLM se identifica con un rol
    # (ver LLMRole en src/nlp/llm/roles.py). Acá se decide, por
    # separado y en formato "backend,modelo" (ver _split_backend_model),
    # qué backend + modelo lo atiende -- este es el ÚNICO lugar donde se
    # decide "qué modelo hace qué rol". No hay una capa de indirección
    # extra tipo PROVIDER_GENERATE=local: el rol especifica su backend y
    # modelo directamente, y ambos ejes (backend -> URL/cliente,
    # modelo -> capacidades) se resuelven en src/nlp/llm/providers.py.
    #
    # Agregar un modelo nuevo a un backend existente = una entrada en
    # src/config/models/<backend>.py, nunca una variable de entorno
    # nueva. Agregar un backend nuevo (ej. Claude, OpenAI) = una entrada
    # en los registros de src/nlp/llm/providers.py + un archivo en
    # src/config/models/ + (si hace falta un cliente nuevo) uno en
    # src/nlp/llm/backends/ -- nunca hace falta tocar graph.py, nodes.py
    # ni generate.py.
    llm_rol_generate: str = Field(default="")
    llm_rol_reformulate: str = Field(default="")
    llm_rol_review: str = Field(default="")
    # Nombrada distinto a LLMRole.WEB_SUPPLEMENT a propósito: en
    # .env.providers el rol se llama "SUPPLEMENT" (más corto), el nombre
    # interno completo ("web_supplement") solo vive en el enum. El mapeo
    # entre ambos está en role_spec() más abajo -- único lugar que lo
    # conoce.
    llm_rol_supplement: str = Field(default="")

    def role_spec(self, role: LLMRole) -> tuple[str, str]:
        """
        Devuelve (backend, modelo) configurado para `role`, ej.
        role_spec(LLMRole.GENERATE) -> ("flm", "qwen3.5:9b").

        Único punto de lookup rol -> (backend, modelo). Los call sites
        (src/nlp/llm/providers.py, src/nlp/llm/context_guard.py) nunca
        leen llm_rol_* directamente.
        """
        raw_by_role: dict[LLMRole, tuple[str, str]] = {
            LLMRole.GENERATE: ("LLM_ROL_GENERATE", self.llm_rol_generate),
            LLMRole.REFORMULATE: ("LLM_ROL_REFORMULATE", self.llm_rol_reformulate),
            LLMRole.REVIEW: ("LLM_ROL_REVIEW", self.llm_rol_review),
            LLMRole.WEB_SUPPLEMENT: ("LLM_ROL_SUPPLEMENT", self.llm_rol_supplement),
        }
        var_name, raw = raw_by_role[role]
        return _split_backend_model(raw, var_name)

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
