from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

AI_HOME = Path(os.environ["AI_HOME"])

HF_HOME = Path(os.environ["HF_HOME"])

BASE_VECTOR_PATH = Path(os.environ["VECTOR_STORE_PATH"])

DATA_PATH = AI_HOME / "data"

LOG_PATH = AI_HOME / "logs"
