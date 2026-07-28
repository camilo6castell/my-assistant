# python -m src.ingest.web_crawler <category> <collection> <start_url>

import sys
import time
from urllib.parse import ParseResult, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

from src.config.settings import settings
from src.ingest.core import (
    build_metadata,
    chunk_text,
    encode_chunks,
)
from src.ingest.http import extract_main_content
from src.storage.faiss_store import (
    ChunkMetadata,
    RawCollection,
    load_collection,
    save_collection,
)
from src.utils.logger import logger


def get_links(url: str, domain: str) -> set[str]:
    try:
        logger.info(f"Extracting links from: {url}")

        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
        links: set[str] = set()

        for a in soup.find_all("a", href=True):
            raw_href = a["href"]
            if not isinstance(raw_href, str):
                continue

            parsed: ParseResult = urlparse(urljoin(url, raw_href))

            if parsed.netloc == domain:
                clean: str = parsed.scheme + "://" + parsed.netloc + parsed.path
                links.add(clean)

        logger.info(f"Links found: {len(links)}")
        return links

    except requests.RequestException as e:
        logger.error(f"HTTP error fetching links from {url}: {e}")
        return set()

    except Exception as e:
        logger.error(f"Error extracting links from {url}: {e}")
        return set()


def main() -> None:

    print(sys.argv)

    category: str = sys.argv[1]
    collection_name: str = sys.argv[2]
    start_url: str = sys.argv[3]

    collection: str = f"{category}/{collection_name}"

    logger.info(f"Starting crawler: {collection}")

    collection_data: RawCollection = load_collection(collection)
    domain: str = urlparse(start_url).netloc

    visited: set[str] = set()
    to_visit: list[str] = [start_url]
    new_chunks: list[str] = []
    new_metadata: list[ChunkMetadata] = []

    while to_visit and len(visited) < settings.max_pages:
        url: str = to_visit.pop(0)

        if url in visited:
            continue

        visited.add(url)
        logger.info(f"Crawling: {url}")

        text: str | None = extract_main_content(url)

        if not text:
            logger.warning(f"Could not extract text from {url}")
            continue

        chunks: list[str] = chunk_text(text)
        logger.info(f"Chunks generated ({len(chunks)}) for {url}")

        for i, chunk in enumerate(chunks):
            new_chunks.append(chunk)
            new_metadata.append(
                build_metadata(
                    source=url,
                    source_type="url",
                    page=1,
                    chunk=chunk,
                    chunk_index=i,
                    collection=collection,
                )
            )

        for link in get_links(url, domain):
            if link not in visited:
                to_visit.append(link)

        logger.info(f"Pending URLs: {len(to_visit)}")
        time.sleep(settings.delay)

    if not new_chunks:
        logger.warning("No new pages found.")
        print("No new pages.")
        return

    logger.info(f"Generating embeddings for {len(new_chunks)} chunks")

    new_embeddings: np.ndarray = encode_chunks(new_chunks)
    save_collection(collection_data, new_embeddings, new_metadata)

    logger.info(f"Crawler finished | chunks={len(new_chunks)}")
    print(f"Indexed {len(new_chunks)} chunks into '{collection}'.")


if __name__ == "__main__":
    main()
