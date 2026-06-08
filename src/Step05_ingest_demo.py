import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from Step01_pdf_chunk_demo import chunk_pdf


EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


DATA_PATH = "data/project.pdf"
STORAGE_DIR = Path("storage")
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
INDEX_PATH = STORAGE_DIR / "faiss.index"


def save_chunks(chunks: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {path}")



def build_faiss_index(chunks: List[Dict[str, Any]], model_name: str):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts,normalize_embeddings=True, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def main():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and chunking PDF...")
    chunks = chunk_pdf(DATA_PATH)
    print(f"Total chunks: {len(chunks)}")

    print("Saving chunks...")
    save_chunks(chunks, CHUNKS_PATH)

    print("Building FAISS index...")
    index = build_faiss_index(chunks, EMBEDDING_MODEL_NAME)

    print("Saving FAISS index...")
    faiss.write_index(index, str(INDEX_PATH))

    print("Done.")
    print(f"Chunks saved to: {CHUNKS_PATH}")
    print(f"Index saved to: {INDEX_PATH}")

    

if __name__ == "__main__":
    main()