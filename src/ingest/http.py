"""
Shared HTTP content extraction for web ingest modules.

Provides extract_main_content() used by both web_ingest.py (single URL)
and web_crawler.py (BFS crawl). Uses readability + BeautifulSoup to
extract the main text content from an HTML page.
"""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from readability import Document

from src.utils.logger import logger


def extract_main_content(url: str) -> str | None:
    """Download a URL and extract its main text content."""
    try:
        logger.info(f"Downloading content: {url}")
        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        doc: Document = Document(response.text)
        soup: BeautifulSoup = BeautifulSoup(doc.summary(), "html.parser")
        text: str = soup.get_text(separator="\n")

        logger.info(f"Content extracted successfully: {url}")
        return text

    except requests.RequestException as e:
        logger.error(f"HTTP error on {url}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return None
