"""
scripts/ingest.py – Scrape Debales AI website and build a FAISS vector index.

Usage:
    python scripts/ingest.py

The script:
1. Crawls the Debales AI website (BFS up to MAX_PAGES pages).
2. Extracts clean text from each page.
3. Chunks the text.
4. Embeds chunks with OpenAI and saves a FAISS index to data/faiss_index/.
"""

from __future__ import annotations

import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
START_URL   = "https://debales.ai"
MAX_PAGES   = 60          # crawl budget
DELAY       = 0.5         # seconds between requests
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 100
INDEX_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DebalesInternBot/1.0; "
        "+https://debales.ai)"
    )
}

SKIP_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                   ".css", ".js", ".xml", ".zip"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def is_scrapable(url: str) -> bool:
    path = urlparse(url).path.lower()
    return not any(path.endswith(ext) for ext in SKIP_EXTENSIONS)


def extract_text(soup: BeautifulSoup, url: str) -> str:
    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "iframe", "svg"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else ""
    body  = soup.get_text(separator="\n", strip=True)
    return f"URL: {url}\nTitle: {title}\n\n{body}"


# ── Crawler ───────────────────────────────────────────────────────────────────
def crawl() -> list[Document]:
    visited: set[str] = set()
    queue:   deque[str] = deque([START_URL])
    docs:    list[Document] = []

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()
        if url in visited or not is_scrapable(url):
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  ✗ {url}  ({exc})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        text = extract_text(soup, url)
        if len(text.strip()) > 200:
            docs.append(Document(page_content=text, metadata={"source": url}))
            print(f"  ✓ {url}  ({len(text)} chars)")

        # Enqueue internal links
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0].split("?")[0]
            if same_domain(href, START_URL) and href not in visited:
                queue.append(href)

        time.sleep(DELAY)

    print(f"\nCrawled {len(visited)} pages, collected {len(docs)} documents.")
    return docs


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Debales AI – Ingestion Pipeline ===\n")

    print("Step 1: Crawling website …")
    docs = crawl()

    print("\nStep 2: Chunking …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created.")

    print("\nStep 3: Embedding & building FAISS index …")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vs.save_local(INDEX_PATH)
    print(f"  Index saved to {INDEX_PATH}")

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
