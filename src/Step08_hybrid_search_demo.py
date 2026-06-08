import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

STORAGE_DIR = Path("storage")
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
INDEX_PATH = STORAGE_DIR / "faiss.index"


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_zh(text: str) -> List[str]:
    return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]

class HybridSearchStore:
    def __init__(
        self,
        chunks_path: str | Path = CHUNKS_PATH,
        index_path: str | Path = INDEX_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        # 1. get chunked data
        self.chunks_path = Path(chunks_path)
        self.index_path = Path(index_path)
        self.documents = load_chunks(self.chunks_path)

        # 2. embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.faiss_index = faiss.read_index(str(self.index_path))

        if self.faiss_index.ntotal != len(self.documents):
            raise ValueError(
                f"FAISS index size mismatch: "
                f"{self.faiss_index.ntotal} vectors vs {len(self.documents)} chunks"
            )

        # 3. bm25
        self.tokenized_corpus = [
            tokenize_zh(doc["text"]) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    
    def vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        pass


    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        pass
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        pass



def print_results(results: List[Dict[str, Any]]) -> None:
    print("\nHybrid Search Results")
    print("=" * 80)
    for i, r in enumerate(results, start=1):
        location = ""
        if r.get("file_type") == "pdf":
            location = f"page={r.get('page')}"
        elif r.get("file_type") in {"xlsx", "xls"}:
            location = f"sheet={r.get('sheet_name')} row={r.get('row_index')}"
        
        print(f"[{i}] hybrid_score={r['score']:.4f}")
        print(f"vector_score={r['vector_score']:.4f} bm25_score={r['bm25_score']:.4f}")
        print(f"matched_by={r['matched_by']}")
        print(
            f"document_id={r.get('document_id')} "
            f"source={r.get('source')} "
            f"{location} "
            f"chunk_id={r.get('chunk_id')}"
        )
        print(r["text"][:800])
        print("-" * 80)

def main():
    store = HybridSearchStore()
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break

        results = store.hybrid_search(
            query=query,
            top_k=5,
            candidate_k=10,
            vector_weight=0.5,
            bm25_weight=0.5,
        )
        print_results(results)

if __name__ == "__main__":
    main()
