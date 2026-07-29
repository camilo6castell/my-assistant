"""
Centralized system configuration using pydantic-settings.

Advantages over manual os.getenv:
  - Type validation at startup (fails fast if env is wrong)
  - Field constraints (gt=0, ge=0, le=2.0)
  - Cross-field validation (@field_validator)
  - Automatic .env reading without calling load_dotenv()
  - Single `settings` object as the source of truth

The exports at the bottom of the module maintain compatibility with all
files that already import directly (e.g.: from src.config.settings import CHUNK_SIZE).
"""

from pathlib import Path

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.models import LLMRole


def _split_backend_model(raw: str, var_name: str) -> tuple[str, str]:
    """
    Parses the "backend,model" format used by EMBEDDER and LLM_ROL_* in
    .env.providers (e.g. "flm,qwen3.5:9b" -> ("flm", "qwen3.5:9b")).

    The split is only on the FIRST comma: the model name can legitimately
    contain ':' (Ollama/FastFlowLM tags, e.g. "qwen3.5:9b"), so we
    never split on that.
    """
    backend, sep, model = raw.partition(",")
    backend, model = backend.strip(), model.strip()
    if not sep or not backend or not model:
        raise ValueError(
            f"{var_name} must have the format 'backend,model' "
            f"(e.g. 'ollama,bge-m3'). Current value: {raw!r}"
        )
    return backend, model


class Settings(BaseSettings):
    """
    RAG system configuration.

    Pydantic-settings reads environment variables (and the .env file)
    automatically. The snake_case field name maps to the UPPER_CASE
    env var (e.g: chunk_size -> CHUNK_SIZE).
    """

    model_config = SettingsConfigDict(
        # env_file=".env",
        env_file=(".env", ".env.providers"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Root
    ai_home: Path = Field(default_factory=lambda: Path.home() / "Documents" / "my-assistant")

    # Paths: Derived paths are computed properties
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
    # BACKENDS -- Runtime URLs (.env.providers)
    # ======================================================
    # A backend is a concrete runtime (FastFlowLM, Ollama, Gemini).
    # Here only HOW to connect to each one lives (URL + credential); WHICH
    # model from that backend each role uses is decided below, in
    # EMBEDDER / LLM_ROL_* -- never in this section.
    #
    # flm/ollama don't require a real API key (local runtimes); the field
    # exists only because the OpenAI SDK requires a non-empty string -- see
    # Settings.embedding_api_key and providers._backend_api_key().
    llm_flm_url: str = Field(default="")
    llm_ollama_url: str = Field(default="")
    llm_gemini_url: str = Field(default="")
    gemini_api_key: str = Field(default="")

    embedder_flm_url: str = Field(default="")
    embedder_ollama_url: str = Field(default="")

    # ======================================================
    # EMBEDDER -- which backend + model generates embeddings
    # ======================================================
    # Format "backend,model" (see _split_backend_model). Valid backends:
    # sentence_transformers (in-process, no URL) | ollama | flm. Adding
    # a new model to an existing backend is an entry in
    # src/config/models/<backend>.py -- never a new environment variable.
    #
    # IMPORTANT: changing backend or model invalidates existing indexes.
    # Indexes are stored in paths separated by backend/model:
    #   /srv/ai/vector_stores/<backend>/<model_safe>/<category>/<collection>/
    # where model_safe replaces '/' and ':' with '_'.
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
        """Fails fast at startup if EMBEDDER/LLM_ROL_* are not in 'backend,model' format."""
        if info.field_name is not None:
            _split_backend_model(v, info.field_name.upper())
        return v

    @property
    def embedder_backend(self) -> str:
        return _split_backend_model(self.embedder, "EMBEDDER")[0]

    @property
    def embedder_model(self) -> str:
        return _split_backend_model(self.embedder, "EMBEDDER")[1]

    # Backward-compatible aliases -- ingest/core.py, cli/commands.py,
    # context/*.py and nlp/embedders/encoder.py already knew these names
    # before the .env.providers simplification; they are kept as the
    # public interface for "which backend/model generates embeddings"
    # so those call sites don't need to be changed.
    @property
    def embedding_backend(self) -> str:
        return self.embedder_backend

    @property
    def embedding_model(self) -> str:
        return self.embedder_model

    @property
    def embedding_model_safe(self) -> str:
        """Sanitized model name for use as a directory name."""
        return self.embedding_model.replace("/", "_").replace(":", "_")

    @property
    def embedding_base_url(self) -> str:
        """
        HTTP URL of the active embeddings backend.

        Only makes sense for HTTP backends (ollama/flm) -- calling this
        with EMBEDDER=sentence_transformers,... is a caller error
        (that backend runs in-process, no URL) and is flagged as such.
        """
        urls = {"flm": self.embedder_flm_url, "ollama": self.embedder_ollama_url}
        backend = self.embedding_backend
        try:
            return urls[backend]
        except KeyError:
            raise ValueError(
                f"Embedding backend {backend!r} does not use an HTTP URL "
                f"(is EMBEDDER=sentence_transformers,...? That backend has no base_url)."
            ) from None

    @property
    def embedding_api_key(self) -> str:
        # Neither Ollama nor FastFlowLM validate this key -- the OpenAI SDK
        # simply requires a non-empty string to construct itself.
        return "not-needed"

    @property
    def vector_store_path_for_backend(self) -> Path:
        """
        Base path including backend and model:
        /srv/ai/vector_stores/<backend>/<model_safe>/

        Indexes from different backends/models are incompatible (they live
        in different vector spaces), which is why they are isolated in
        separate subdirectories instead of being mixed together.
        Folders are created automatically on first use in get_collection_paths().
        """
        return self.ai_home / "vector_stores" / self.embedding_backend / self.embedding_model_safe

    # Chunking
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info: ValidationInfo) -> int:
        """
        Ensures that overlap is not >= chunk size.
        Without this validation, chunk_text() would produce an infinite loop.
        """
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({info.data['chunk_size']})"
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
    # Opt-in diagnostics: dumps the EXACT request (messages + flattened
    # extra_fields) sent to the provider on each call to
    # ./debug_last_llm_request.json -- see _dump_request_for_debug in
    # src/llm/generate.py. Intended for reproducing with curl a failure
    # that depends on the actual prompt size/content (e.g. RAG with many
    # chunks) without rebuilding it by hand. Default False: never writes
    # files during normal use.
    llm_debug_dump: bool = Field(default=False)

    # ======================================================
    # LLM BY ROLE -- which backend + model handles each role
    # ======================================================
    # Each pipeline step that calls an LLM is identified by a role
    # (see LLMRole in src/nlp/llm/roles.py). Here we decide, separately
    # and in "backend,model" format (see _split_backend_model), which
    # backend + model handles it -- this is the ONLY place where we
    # decide "which model handles which role". There is no extra layer
    # of indirection like PROVIDER_GENERATE=local: the role specifies
    # its backend and model directly, and both axes (backend -> URL/client,
    # model -> capabilities) are resolved in src/nlp/llm/providers.py.
    #
    # Adding a new model to an existing backend = an entry in
    # src/config/models/<backend>.py, never a new environment variable.
    # Adding a new backend (e.g. Claude, OpenAI) = an entry in the
    # registry of src/nlp/llm/providers.py + a file in
    # src/config/models/ + (if a new client is needed) one in
    # src/nlp/llm/backends/ -- graph.py, nodes.py or generate.py
    # never need to be touched.
    llm_rol_generate: str = Field(default="")
    # Named differently from LLMRole.WEB_SUPPLEMENT on purpose: in
    # .env.providers the role is called "SUPPLEMENT" (shorter), the full
    # internal name ("web_supplement") only lives in the enum. The mapping
    # between both is in role_spec() below -- the only place that knows it.
    llm_rol_supplement: str = Field(default="")

    def role_spec(self, role: LLMRole) -> tuple[str, str]:
        """
        Returns the (backend, model) configured for `role`, e.g.
        role_spec(LLMRole.GENERATE) -> ("flm", "qwen3.5:9b").

        Single lookup point for role -> (backend, model). The call sites
        (src/nlp/llm/providers.py, src/nlp/llm/context_guard.py) never
        read llm_rol_* directly.
        """
        raw_by_role: dict[LLMRole, tuple[str, str]] = {
            LLMRole.GENERATE: ("LLM_ROL_GENERATE", self.llm_rol_generate),
            LLMRole.WEB_SUPPLEMENT: ("LLM_ROL_SUPPLEMENT", self.llm_rol_supplement),
        }
        var_name, raw = raw_by_role[role]
        return _split_backend_model(raw, var_name)

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Settings":
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        return self

    # Context guard -- token estimation safety margins
    context_guard_safety_margin: float = Field(default=0.10, ge=0.0, le=0.5)
    context_guard_output_reserve: int = Field(default=1024, gt=0)

    # Embeddings
    embedding_batch_size: int = Field(default=64, gt=0)

    # Demo endpoint (POST /api/v1/demo/query)
    demo_max_concurrency: int = Field(default=1, ge=1)
    demo_max_tokens: int | None = Field(default=None)

    # API
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000, gt=0)

    # MCP Server
    mcp_host: str = Field(default="127.0.0.1")
    mcp_port: int = Field(default=8100, gt=0)
    mcp_bearer_token: str = Field(default="")
    mcp_allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])

    # ======================================================
    # WEB SEARCH (Tavily)
    # ======================================================
    # Web search as a complementary/alternative retrieval source to
    # local collections (see QueryRequest.web_search in
    # src/api/schemas/chat.py and src/retrieval/web_search.py).
    #
    # web_search_enabled is a global kill-switch independent of whether
    # web_search=True is sent in the request: it allows disabling the
    # entire feature in an environment (e.g. without internet access, or
    # to avoid generating Tavily API costs) without touching the frontend
    # or the code -- the router returns 400 if web_search=True is
    # requested with this set to False.
    web_search_enabled: bool = Field(default=False)
    tavily_api_key: str = Field(default="")
    web_search_timeout: int = Field(default=15, gt=0)
    web_search_max_results: int = Field(default=5, gt=0)
    web_search_depth: str = Field(default="basic")
    web_search_include_answer: bool = Field(default=False)


# Singleton instance -- validated on module import.
settings = Settings()
