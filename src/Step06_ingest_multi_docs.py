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



def load_and_chunk_pdf(file_path: Path, document_id: str) -> List[Dict[str, Any]]:
    pages = load_pdf_by_page(str(file_path))
    chunks = []
    for page in pages:
        text_chunks = chunk_text(page["text"], chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        for chunk_index, text in enumerate(text_chunks):
            chunks.append({
                "document_id": document_id,
                "source": page["source"],
                "file_path": str(file_path),
                "file_type": "pdf",
                "page": page["page"],
                "chunk_id": f"{document_id}:p{page['page']}:c{chunk_index}",
                "text": text,
            })
    return chunks


def make_document_id(i: int) -> str:
    return f"doc_{i:04d}"

def discover_pdf_documents(docs_dir: Path) -> List[Path]:
    return sorted(docs_dir.glob("*.pdf"))

def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

        


def build_faiss_index(chunks: List[Dict[str, Any]], model_name: str):
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = discover_pdf_documents(DOCS_DIR)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DOCS_DIR}")

    documents = []
    all_chunks = []

    for i, pdf_path in enumerate(pdf_files, start=1):
        document_id = make_document_id(i)
        print(f"Ingesting {pdf_path.name} as {document_id}...")
        documents.append({
            "document_id": document_id,
            "source": pdf_path.name,
            "file_path": str(pdf_path),
            "file_type": "pdf",
        })

        chunks = load_and_chunk_pdf(
            file_path=pdf_path,
            document_id=document_id,
        )
        all_chunks.extend(chunks)

    print(f"Saving documents.json ({len(documents)} docs)...")
    save_json(documents, DOCUMENTS_PATH)

    print(f"Saving chunks.json ({len(all_chunks)} chunks)...")
    save_json(all_chunks, CHUNKS_PATH)

    print("Building FAISS index...")
    index = build_faiss_index(all_chunks, EMBEDDING_MODEL_NAME)

    print("Saving FAISS index...")
    faiss.write_index(index, str(INDEX_PATH))

    print("Saving manifest...")
    manifest = {
        "ingest_time": datetime.now().isoformat(),
        "num_docs": len(documents),
        "num_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL_NAME
    }
    save_json(manifest, MANIFEST_PATH)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
