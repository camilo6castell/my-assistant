# src/config/settings.py

import os

from pathlib import Path

from dotenv import load_dotenv

from src.utils.tools import env_bool

load_dotenv()

# ======================================================
# ROOT
# ======================================================

AI_HOME = Path(
    os.getenv(
        "AI_HOME",
        "/srv/ai",
    )
)

# ======================================================
# PATHS
# ======================================================

DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        f"{AI_HOME}/data",
    )
)

BASE_VECTOR_PATH = Path(
    os.getenv(
        "VECTOR_STORE_PATH",
        f"{AI_HOME}/vector_stores",
    )
)

LOG_PATH = Path(
    os.getenv(
        "LOG_PATH",
        f"{AI_HOME}/logs",
    )
)

HF_HOME = Path(
    os.getenv(
        "HF_HOME",
        f"{AI_HOME}/hf",
    )
)

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

INTERPRETATIVE_TOP_K_INITIAL = int(
    os.getenv(
        "INTERPRETATIVE_TOP_K_INITIAL",
        25,
    )
)

INTERPRETATIVE_TOP_K_FINAL = int(
    os.getenv(
        "INTERPRETATIVE_TOP_K_FINAL",
        7,
    )
)

MAX_TURNS = int(
    os.getenv(
        "MAX_TURNS",
        4,
    )
)

DEFAULT_INTERPRETATIVE_MODE = env_bool(
    "DEFAULT_INTERPRETATIVE_MODE",
    False,
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

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        0.2,
    )
)

# Timeout generoso para modelos locales cuantizados.
# Un 8B en modo interpretativo puede tardar varios minutos
# con prompts densos. Ajustable via env: LLM_TIMEOUT=300
LLM_TIMEOUT = int(
    os.getenv(
        "LLM_TIMEOUT",
        600,
    )
)

# ======================================================
# WEB INGEST
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
