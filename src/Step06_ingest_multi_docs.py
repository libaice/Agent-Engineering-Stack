import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from Step01_pdf_chunk_demo import load_pdf_by_page, chunk_text


EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

DOCS_DIR = Path("data/docs")
STORAGE_DIR = Path("storage")

DOCUMENTS_PATH = STORAGE_DIR / "documents.json"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
INDEX_PATH = STORAGE_DIR / "faiss.index"
MANIFEST_PATH = STORAGE_DIR / "manifest.json"

CHUNK_SIZE = 800
OVERLAP = 120


def discover_pdf_documents(docs_dir: Path) -> List[Path]:
    return sorted(docs_dir.glob("*.pdf"))

def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

        


def main():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = discover_pdf_documents(DOCS_DIR)


if __name__ == "__main__":
    main()
