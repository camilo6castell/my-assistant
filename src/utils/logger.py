import logging

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s " "[%(levelname)s] " "%(name)s: " "%(message)s"),
)

logger = logging.getLogger("rag")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logging.getLogger("pypdf").setLevel(logging.ERROR)
