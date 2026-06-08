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
        self.tokenized_corpus = [tokenize_zh(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def vector_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.array(query_embedding).astype("float32")

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            doc = self.documents[idx]
            results.append(
                {
                    "idx": idx,
                    "score": float(score),
                    "method": "vector",
                    "document": doc,
                }
            )
        return results

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokenized_query = tokenize_zh(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            score = float(scores[idx])
            doc = self.documents[idx]
            results.append(
                {
                    "idx": idx,
                    "score": score,
                    "method": "bm25",
                    "document": doc,
                }
            )
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:

        vector_results = self.vector_search(query, top_k=candidate_k)
        bm25_results = self.bm25_search(query, top_k=candidate_k)

        vector_scores = normalize_scores([r["score"] for r in vector_results])
        bm25_scores = normalize_scores([r["score"] for r in bm25_results])

        merged: Dict[int, Dict[str, Any]] = {}

        for result, norm_score in zip(vector_results, vector_scores):
            idx = result["idx"]
            if idx not in merged:
                merged[idx] = {
                    "idx": idx,
                    "document": result["document"],
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "matched_by": set(),
                }
            merged[idx]["vector_score"] = norm_score
            merged[idx]["matched_by"].add("vector")

        for result, norm_score in zip(bm25_results, bm25_scores):
            idx = result["idx"]

            if idx not in merged:
                merged[idx] = {
                    "idx": idx,
                    "document": result["document"],
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                    "matched_by": set(),
                }

            merged[idx]["bm25_score"] = norm_score
            merged[idx]["matched_by"].add("bm25")

        final_results = []

        for item in merged.values():
            hybrid_score = (
                vector_weight * item["vector_score"] + bm25_weight * item["bm25_score"]
            )
            doc = item["document"]

            final_results.append(
                {
                    "score": hybrid_score,
                    "vector_score": item["vector_score"],
                    "bm25_score": item["bm25_score"],
                    "matched_by": sorted(list(item["matched_by"])),
                    "document_id": doc.get("document_id"),
                    "source": doc.get("source"),
                    "file_type": doc.get("file_type"),
                    "page": doc.get("page"),
                    "sheet_name": doc.get("sheet_name"),
                    "row_index": doc.get("row_index"),
                    "chunk_id": doc.get("chunk_id"),
                    "text": doc.get("text"),
                }
            )

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]


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


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


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
