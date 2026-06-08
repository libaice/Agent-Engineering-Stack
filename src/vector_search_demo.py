from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_chunk_demo import chunk_pdf


class SimpleVectorStore:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents: List[Dict[str,Any]] = []

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


if __name__ == "__main__":
    file_path = "data/project.pdf"
    chunks = chunk_pdf(file_path)

    store = SimpleVectorStore()

    # 1/ build index
    store.build(chunks)
