# python -m src.ingest.web_crawler react https://sitio.com

import sys
import time
from urllib.parse import ParseResult, urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from readability import Document

from src.config.settings import DELAY, MAX_PAGES
from src.ingest.core import (
    ChunkMetadata,
    RawCollection,
    build_metadata,
    chunk_text,
    encode_chunks,
    load_collection,
    save_collection,
)
from src.utils.logger import logger


def extract_main_content(url: str) -> str | None:
    try:
        logger.info(f"Descargando contenido: {url}")

        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        doc: Document = Document(response.text)
        soup: BeautifulSoup = BeautifulSoup(doc.summary(), "html.parser")
        text: str = soup.get_text(separator="\n")

        logger.info(f"Contenido extraído correctamente: {url}")
        return text

    except requests.RequestException as e:
        logger.error(f"Error HTTP en {url}: {e}")
        return None

    except Exception as e:
        logger.error(f"Error extrayendo contenido de {url}: {e}")
        return None


def get_links(url: str, domain: str) -> set[str]:
    try:
        logger.info(f"Extrayendo links desde: {url}")

        response: requests.Response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup: BeautifulSoup = BeautifulSoup(response.text, "html.parser")
        links: set[str] = set()

        for a in soup.find_all("a", href=True):
            href: str = urljoin(url, a["href"])
            # FIX bonus: ParseResult en lugar de Any
            parsed: ParseResult = urlparse(href)

            if parsed.netloc == domain:
                clean: str = parsed.scheme + "://" + parsed.netloc + parsed.path
                links.add(clean)

        logger.info(f"Links encontrados: {len(links)}")
        return links

    except requests.RequestException as e:
        logger.error(f"Error HTTP obteniendo links de {url}: {e}")
        return set()

    except Exception as e:
        logger.error(f"Error extrayendo links de {url}: {e}")
        return set()


def main() -> None:
    if len(sys.argv) < 3:
        logger.error("Argumentos insuficientes.")
        print("Uso: python -m src.ingest.web_crawler <collection> <URL_BASE>")
        return

    collection: str = sys.argv[1]
    start_url: str = sys.argv[2]

    logger.info(f"Iniciando crawler | collection={collection}")

    collection_data: RawCollection = load_collection(collection)
    domain: str = urlparse(start_url).netloc

    visited: set[str] = set()
    to_visit: list[str] = [start_url]
    new_chunks: list[str] = []
    new_metadata: list[ChunkMetadata] = []

    while to_visit and len(visited) < MAX_PAGES:
        url: str = to_visit.pop(0)

        if url in visited:
            continue

        visited.add(url)
        logger.info(f"Crawling: {url}")

        text: str | None = extract_main_content(url)

        if not text:
            logger.warning(f"No se pudo extraer texto de {url}")
            continue

        chunks: list[str] = chunk_text(text)
        logger.info(f"Chunks generados ({len(chunks)}) para {url}")

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

        links: set[str] = get_links(url, domain)

        for link in links:
            if link not in visited:
                to_visit.append(link)

        logger.info(f"URLs pendientes: {len(to_visit)}")
        time.sleep(DELAY)

    if not new_chunks:
        logger.warning("No se encontraron nuevas páginas.")
        print("No nuevas páginas.")
        return

    logger.info(f"Generando embeddings para {len(new_chunks)} chunks")

    new_embeddings: np.ndarray = encode_chunks(new_chunks)

    save_collection(collection_data, new_embeddings, new_metadata)

    logger.info(f"Crawler finalizado | chunks={len(new_chunks)}")
    print(f"Se indexaron {len(new_chunks)} chunks en '{collection}'.")


if __name__ == "__main__":
    main()
