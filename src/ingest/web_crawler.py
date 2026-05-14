# python -m src.ingest.web_crawler react https://sitio.com

import sys
import time
import requests

from urllib.parse import (
    urljoin,
    urlparse,
)

from bs4 import BeautifulSoup
from readability import Document

from src.utils.logger import logger

from src.config.settings import (
    MAX_PAGES,
    DELAY,
)

from src.ingest.core import (
    chunk_text,
    load_collection,
    encode_chunks,
    save_collection,
    build_metadata,
)


def extract_main_content(url: str):

    try:

        logger.info(f"Descargando contenido: {url}")

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        doc = Document(response.text)

        soup = BeautifulSoup(
            doc.summary(),
            "html.parser",
        )

        text = soup.get_text(separator="\n")

        logger.info(f"Contenido extraído correctamente: " f"{url}")

        return text

    except requests.RequestException as e:

        logger.error(f"Error HTTP en {url}: {e}")

        return None

    except Exception as e:

        logger.error(f"Error extrayendo contenido " f"de {url}: {e}")

        return None


def get_links(
    url: str,
    domain: str,
):

    try:

        logger.info(f"Extrayendo links desde: {url}")

        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        links = set()

        for a in soup.find_all(
            "a",
            href=True,
        ):

            href = urljoin(
                url,
                a["href"],
            )

            parsed = urlparse(href)

            if parsed.netloc == domain:

                clean = parsed.scheme + "://" + parsed.netloc + parsed.path

                links.add(clean)

        logger.info(f"Links encontrados: " f"{len(links)}")

        return links

    except requests.RequestException as e:

        logger.error(f"Error HTTP obteniendo links " f"de {url}: {e}")

        return set()

    except Exception as e:

        logger.error(f"Error extrayendo links " f"de {url}: {e}")

        return set()


def main():

    if len(sys.argv) < 3:

        logger.error("Argumentos insuficientes.")

        print("Uso: python -m " "src.ingest.web_crawler " "<collection> <URL_BASE>")

        return

    collection = sys.argv[1]

    start_url = sys.argv[2]

    logger.info(f"Iniciando crawler | " f"collection={collection}")

    collection_data = load_collection(collection)

    domain = urlparse(start_url).netloc

    visited = set()

    to_visit = [start_url]

    new_chunks = []

    new_metadata = []

    while to_visit and len(visited) < MAX_PAGES:

        url = to_visit.pop(0)

        if url in visited:
            continue

        visited.add(url)

        logger.info(f"Crawling: {url}")

        text = extract_main_content(url)

        if not text:

            logger.warning(f"No se pudo extraer texto " f"de {url}")

            continue

        chunks = chunk_text(text)

        logger.info(f"Chunks generados " f"({len(chunks)}) " f"para {url}")

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

        links = get_links(
            url,
            domain,
        )

        for link in links:

            if link not in visited:
                to_visit.append(link)

        logger.info(f"URLs pendientes: " f"{len(to_visit)}")

        time.sleep(DELAY)

    if not new_chunks:

        logger.warning("No se encontraron " "nuevas páginas.")

        print("No nuevas páginas.")

        return

    logger.info(f"Generando embeddings " f"para {len(new_chunks)} chunks")

    new_embeddings = encode_chunks(new_chunks)

    save_collection(
        collection_data,
        new_embeddings,
        new_metadata,
    )

    logger.info(f"Crawler finalizado | " f"chunks={len(new_chunks)}")

    print(f"Se indexaron " f"{len(new_chunks)} chunks " f"en '{collection}'.")


if __name__ == "__main__":
    main()
