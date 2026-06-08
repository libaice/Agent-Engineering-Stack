import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from Step03_rag_answer_demo import answer_with_llm


EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

STORAGE_DIR = Path("storage")
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
INDEX_PATH = STORAGE_DIR / "faiss.index"


class PersistentVectorStore:
    def __init__(
        self,
        chunks_path: str | Path,
        index_path: str | Path,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        self.model = SentenceTransformer(embedding_model_name)
        self.documents = self._load_chunks()
        self.index = faiss.read_index(str(self.index_path))

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                f"Index size mismatch: FAISS has {self.index.ntotal} vectors, "
                f"but chunks.json has {len(self.documents)} chunks."
            )

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx]
            # results.append(
            #     {
            #         "score": float(score),
            #         "source": doc["source"],
            #         "page": doc["page"],
            #         "chunk_id": doc["chunk_id"],
            #         "text": doc["text"],
            #     }
            # )
            results.append(
                {
                    "score": float(score),
                    "document_id": doc["document_id"],
                    "source": doc["source"],
                    "file_path": doc["file_path"],
                    "file_type": doc["file_type"],
                    "page": doc["page"],
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                }
            )

        return results


def main():
    store = PersistentVectorStore(CHUNKS_PATH, INDEX_PATH)

    print(f"Loaded {len(store.documents)} chunks ")
    print("Ready.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        retrieved = store.search(question, top_k=5)
        print("\nRetrieved evidence:")
        print("=" * 80)

        for i, item in enumerate(retrieved, start=1):
            print(f"[{i}] score={item['score']:.4f}")
            print(
                f"source={item['source']} page={item['page']} chunk_id={item['chunk_id']}"
            )
            print(item["text"][:500])
            print("-" * 80)

        answer = answer_with_llm(question, retrieved)
        print("\nAnswer:")
        print("=" * 80)
        print(answer)


if __name__ == "__main__":
    main()
