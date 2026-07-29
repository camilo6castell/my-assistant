from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from src.api.schemas.demo import DemoQueryRequest
from src.api.services.query import resolve_collections
from src.config.models import build_kwargs
from src.config.settings import settings
from src.domain.models import LLMRole
from src.nlp.llm.generate import build_messages
from src.nlp.llm.providers import get_provider
from src.prompts.builder import build_prompt
from src.retrieval.search import format_context_chunks, search
from src.utils.logger import logger

router = APIRouter(tags=["demo"])

_semaphore = asyncio.Semaphore(settings.demo_max_concurrency)


@router.post("/demo/query")
async def demo_query(request: DemoQueryRequest) -> StreamingResponse:
    try:
        collections = await asyncio.to_thread(resolve_collections, request.collections)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if not collections:
        raise HTTPException(
            status_code=422,
            detail="No collections specified. At least one collection is required.",
        )

    if _semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing another request. Please try again.",
        )

    await _semaphore.acquire()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            results, confidence = await asyncio.to_thread(
                search, request.question, request.mode, collections
            )

            if not results:
                yield "data: [DONE]\n\n"
                return

            context_chunks = format_context_chunks(results)
            prompt = build_prompt(context_chunks, request.question, request.mode)
            messages = build_messages(prompt=prompt, chat_memory=[])

            config = get_provider(LLMRole.GENERATE.value)
            kwargs = build_kwargs(config.capabilities, config.model, messages)
            kwargs["stream"] = True
            if settings.demo_max_tokens is not None:
                kwargs["max_tokens"] = settings.demo_max_tokens

            async_client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
            )

            try:
                stream = await async_client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {content}\n\n"
            except Exception as exc:
                logger.exception("[demo] LLM streaming error")
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            finally:
                await async_client.close()

        except Exception:
            logger.exception("[demo] Unexpected error")
            yield f"data: {json.dumps({'error': 'Internal error'})}\n\n"
        finally:
            _semaphore.release()

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
