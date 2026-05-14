from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ======================================================
# PATHS
# ======================================================

AI_HOME = Path(
    os.getenv(
        "AI_HOME",
        "/srv/ai",
    )
)

BASE_VECTOR_PATH = Path(
    os.getenv(
        "VECTOR_STORE_PATH",
        f"{AI_HOME}/vector_stores",
    )
)

DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        f"{AI_HOME}/data",
    )
)

LOG_PATH = AI_HOME / "logs"

HF_HOME = Path(os.environ["HF_HOME"])


# ======================================================
# EMBEDDINGS
# ======================================================

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-small-en-v1.5",
)


# ======================================================
# CHUNKING
# ======================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        500,
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        100,
    )
)


# ======================================================
# RETRIEVAL
# ======================================================

BASE_TOP_K_INITIAL = int(
    os.getenv(
        "BASE_TOP_K_INITIAL",
        15,
    )
)

BASE_TOP_K_FINAL = int(
    os.getenv(
        "BASE_TOP_K_FINAL",
        5,
    )
)

MAX_TURNS = int(
    os.getenv(
        "MAX_TURNS",
        4,
    )
)

INTERPRETATIVE_MODE = (
    os.getenv(
        "INTERPRETATIVE_MODE",
        "False",
    ).lower()
    == "true"
)


# ======================================================
# LLM
# ======================================================

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "fastflowlm",
)

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://127.0.0.1:52625/v1",
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    "flm",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3-it:4b",
)


# ======================================================
# WEB CRAWLER
# ======================================================

MAX_PAGES = int(
    os.getenv(
        "MAX_PAGES",
        50,
    )
)

DELAY = float(
    os.getenv(
        "DELAY",
        1,
    )
)
