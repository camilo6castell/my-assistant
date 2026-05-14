from openai import OpenAI

from src.config.settings import (
    LLM_BASE_URL,
    LLM_API_KEY,
)

from src.utils.logger import logger


def create_client():

    logger.info(f"Inicializando cliente LLM " f"| base_url={LLM_BASE_URL}")

    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )


client = create_client()
