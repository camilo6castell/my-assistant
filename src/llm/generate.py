from openai import OpenAIError

from src.llm.client import client

from src.utils.env import LLM_MODEL

from src.utils.logger import logger


def ask_llm(prompt: str) -> str:

    try:

        logger.info(f"Consultando LLM: {LLM_MODEL}")

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            timeout=120,
        )

        content = response.choices[0].message.content

        if not content:
            logger.warning("LLM devolvió respuesta vacía.")
            return "El modelo no devolvió respuesta."

        return content

    except OpenAIError as e:

        logger.exception("Error consultando LLM")

        return f"Error consultando modelo: {e}"
