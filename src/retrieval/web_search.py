"""
Web search via Tavily, used as a complementary or alternative retrieval source
to local collections (see QueryRequest.web_search in src/api/schemas/chat.py
and its usage in src/api/routers/chat.py).

Why Tavily instead of hitting a search engine directly: scraping
Google/Bing violates their ToS and is fragile (HTML changes without notice).
Tavily is designed specifically for RAG/LLMs -- it returns clean snippets
with URLs via a simple HTTP API (same pattern already used for external HTTP
with `requests`, see src/ingest/).

This module NEVER lets a failure here take down the full request: every
exception (timeout, network, invalid JSON, HTTP 4xx/5xx) is caught, logged,
and returned as a WebSearchOutcome with empty results -- the router decides
what to do with that (see chat.py). The only distinction that IS propagated
is "quota exhausted" (status=QUOTA_EXCEEDED) vs.
"any other failure" (status=ERROR): they are the same "no results" for the
generation pipeline, but require different UX (see GenerationSection.tsx --
a transient network failure should not disable the "Web" button, but
exhausting the Tavily free tier should warn the user instead of failing
silently every time).
"""

from __future__ import annotations

from enum import StrEnum

import requests
from pydantic import BaseModel

from src.config.settings import settings
from src.utils.logger import logger

TAVILY_ENDPOINT = "https://api.tavily.com/search"

# HTTP codes documented by Tavily for account limits (see official SDK
# tavily-python: both map to ForbiddenError).
#   432 = Plan Limit Exceeded (plan credits exhausted, e.g. the
#         free tier of 1000/month).
#   433 = Pay-As-You-Go Limit Exceeded (spending cap configured on the
#         account, for accounts with PAYGO enabled).
# Source: https://docs.tavily.com (research endpoint) and
# https://github.com/tavily-ai/tavily-n8n-node (error table).
TAVILY_QUOTA_EXCEEDED_STATUS_CODES = frozenset({432, 433})


class WebSearchResult(BaseModel):
    title: str
    url: str
    content: str  # snippet already summarized by Tavily, not raw HTML


class WebSearchStatus(StrEnum):
    OK = "ok"
    # Tavily account credits exhausted (free tier or other plan) --
    # unlike a transient failure: retrying until the month/plan changes
    # makes no sense (see settings.web_search_quota_exceeded in the frontend).
    QUOTA_EXCEEDED = "quota_exceeded"
    # Any other failure: missing API key, feature disabled, timeout,
    # network error, invalid JSON, or any non-2xx HTTP code.
    ERROR = "error"


class WebSearchOutcome(BaseModel):
    results: list[WebSearchResult]
    status: WebSearchStatus


def search_web(query: str, max_results: int | None = None) -> WebSearchOutcome:
    """
    Returns up to max_results web snippets for `query`, along with a status
    distinguishing "no results due to quota exhausted" from any other reason
    (see WebSearchStatus). Never raises an exception.
    """
    if not settings.web_search_enabled:
        logger.warning("[web_search] WEB_SEARCH_ENABLED=false -- search skipped.")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    if not settings.tavily_api_key:
        logger.warning("[web_search] TAVILY_API_KEY not configured -- search skipped.")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    effective_max_results = (
        max_results if max_results is not None else settings.web_search_max_results
    )

    try:
        response = requests.post(
            TAVILY_ENDPOINT,
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "search_depth": settings.web_search_depth,
                "max_results": effective_max_results,
                "include_answer": settings.web_search_include_answer,
            },
            timeout=settings.web_search_timeout,
        )
    except requests.RequestException as e:
        logger.warning(f"[web_search] Tavily request failed: {e}")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    if response.status_code in TAVILY_QUOTA_EXCEEDED_STATUS_CODES:
        logger.warning(
            f"[web_search] Tavily quota exceeded (HTTP {response.status_code}) -- "
            "check your plan at https://app.tavily.com."
        )
        return WebSearchOutcome(results=[], status=WebSearchStatus.QUOTA_EXCEEDED)

    if not response.ok:
        logger.warning(
            f"[web_search] Tavily returned HTTP {response.status_code}: {response.text[:200]}"
        )
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    try:
        payload = response.json()
    except ValueError:
        logger.warning("[web_search] Tavily response is not valid JSON.")
        return WebSearchOutcome(results=[], status=WebSearchStatus.ERROR)

    results: list[WebSearchResult] = []
    for item in payload.get("results", []):
        title = item.get("title")
        url = item.get("url")
        content = item.get("content")
        if not (title and url and content):
            continue
        results.append(WebSearchResult(title=title, url=url, content=content))

    logger.info(f"[web_search] Tavily returned {len(results)} result(s) for: {query!r}")
    return WebSearchOutcome(results=results, status=WebSearchStatus.OK)
