import logging

FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
)

logger: logging.Logger = logging.getLogger("rag")

# ======================================================
# NOISE REDUCTION
# ======================================================

logging.getLogger("httpx").setLevel(logging.WARNING)

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.getLogger("transformers").setLevel(logging.WARNING)

logging.getLogger("pypdf").setLevel(logging.ERROR)

logging.getLogger("urllib3").setLevel(logging.WARNING)
