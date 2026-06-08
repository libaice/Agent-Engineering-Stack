from typing import List, Dict, Any

from rag.Step13_query_rewrite_demo import rewrite_query
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker, print_reranked_results
from rag.Step12_structured_answer_demo import answer_with_structured_output


def dedup_by_chunk_id(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in results:
        chunk_id = item.get("chunk_id")

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        deduped.append(item)

    return deduped


def multi_query_search(
    store: HybridSearchStore,
    search_queries: List[str],
    per_query_k: int = 10,
) -> List[Dict[str, Any]]:
    all_results = []
    for query in search_queries:
        results = store.hybrid_search(
            query=query,
            top_k=per_query_k,
            candidate_k=per_query_k,
            vector_weight=0.5,
            bm25_weight=0.5,
        )
        for r in results:
            item = dict(r)
            item["generated_query"] = query
            all_results.append(item)

    return dedup_by_chunk_id(all_results)


def main():
    store = HybridSearchStore()
    reranker = Reranker()

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        # 1. query rewrite
        rewrite = rewrite_query(question)
        print("\nQuery Rewrite")
        print("=" * 80)
        print(rewrite.model_dump_json(indent=2))

        # 2. hybrid search
        candidates = multi_query_search(
            store=store,
            search_queries=rewrite.search_queries,
            per_query_k=10,
        )

        reranked = reranker.rerank(
            query=rewrite.rewritten_query,
            candidates=candidates,
            top_k=5,
        )

        print_reranked_results(reranked)

        structured_answer = answer_with_structured_output(
            question=question,
            retrieved_chunks=reranked,
        )
        print("\nStructured Answer")
        print("=" * 80)
        print(structured_answer.model_dump_json(indent=2))

        print("\nUser-facing Answer")
        print("=" * 80)
        print(structured_answer.answer)



if __name__ == "__main__":
    main()
