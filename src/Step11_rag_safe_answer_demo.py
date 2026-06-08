from Step08_hybrid_search_demo import HybridSearchStore
from Step09_rerank_demo import Reranker, print_reranked_results
from Step10_evidence_check_demo import check_answerability
from Step03_rag_answer_demo import answer_with_llm


UNKNOWN_PHRASE = "根据现有资料无法确定。"


def main():
    print("===hybrid search demo===")
    store = HybridSearchStore()
    reranker = Reranker()

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        candidates = store.hybrid_search(
            query=question,
            top_k=20,
            candidate_k=20,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

        reranked = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=5,
        )

        print_reranked_results(reranked)

        check = check_answerability(question, reranked)
        print("\nAnswerability Check")
        print("=" * 80)
        print(check)

        if not check.get("answerable", False):
            print("\nAnswer")
            print("=" * 80)
            print(UNKNOWN_PHRASE)
            print(f"原因：{check.get('reason')}")
            print(f"缺少信息：{check.get('missing_information')}")
            continue

        answer = answer_with_llm(question, reranked)

        print("\nAnswer")
        print("=" * 80)
        print(answer)


if __name__ == "__main__":
    main()
