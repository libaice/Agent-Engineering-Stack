from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.Step01_pdf_chunk_demo import chunk_pdf


class SimpleVectorStore:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]):
        self.documents = chunks
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        )
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"Built vector index with {len(chunks)} chunks")
        print(f"Embedding dimension: {dim}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built yet")

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx]
            results.append(
                {
                    "score": float(score),
                    "source": doc["source"],
                    "page": doc.get("page"),
                    "sheet_name": doc.get("sheet_name"),
                    "row_index": doc.get("row_index"),
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                }
            )

        return results


if __name__ == "__main__":
    file_path = "data/project.pdf"
    chunks = chunk_pdf(file_path)

    store = SimpleVectorStore()

    # 1. build index
    store.build(chunks)

    # 2. search
    while True:
        query = input("Query: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break

        results = store.search(query, top_k=5)
        print("\nTop results:")
        print("=" * 80)

        for i, result in enumerate(results, start=1):
            print(f"[{i}] score={result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Chunk: {result['chunk_id']}")
            print(result["text"][:800])
            print("-" * 80)
