from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from rag.Step08_hybrid_search_demo import HybridSearchStore, print_results
from rag.Step03_rag_answer_demo import answer_with_llm

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pairs = [[query, candidate["text"]] for candidate in candidates]
        scores = self.model.predict(pairs)

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


def print_reranked_results(results: List[Dict[str, Any]]) -> None:
    print("\nReranked Results")
    print("=" * 80)

    for i, r in enumerate(results, start=1):
        location = ""
        if r.get("file_type") == "pdf":
            location = f"page={r.get('page')}"
        elif r.get("file_type") in {"xlsx", "xls"}:
            location = f"sheet={r.get('sheet_name')} row={r.get('row_index')}"

        print(f"[{i}] rerank_score={r['rerank_score']:.4f}")
        print(
            f"hybrid_score={r.get('score', 0):.4f} "
            f"vector_score={r.get('vector_score', 0):.4f} "
            f"bm25_score={r.get('bm25_score', 0):.4f}"
        )
        print(f"matched_by={r.get('matched_by')}")
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
    reranker = Reranker()

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break
        candidates = store.hybrid_search(
            query=query,
            top_k=20,
            candidate_k=20,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

        # 1/ before rerank
        # print("\nBefore Rerank: Hybrid Candidates")
        # print_results(candidates[:5])

        reranked = reranker.rerank(query, candidates, top_k=5)

        # 2/ after rerank
        # print_reranked_results(reranked)

        # 3/ answer with llm
        answer = answer_with_llm(query, reranked)
        print(f"\nFinal Answer: {answer}")

        

if __name__ == "__main__":
    main()
