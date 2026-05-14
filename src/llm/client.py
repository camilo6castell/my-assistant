from openai import OpenAI

from src.utils.env import (
    LLM_BASE_URL,
    LLM_API_KEY,
)

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)
