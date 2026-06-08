import yaml
from pathlib import Path
import time
from typing import List, Dict, Any

from Step01_pdf_chunk_demo import chunk_pdf
from Step02_vector_search_demo import SimpleVectorStore
from Step03_rag_answer_demo import answer_with_llm



UNKNOWN_PHRASE = "根据现有资料无法确定"

def load_eval_cases(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_retrieval_hit(
    retrieved: List[Dict[str, Any]],
    expected_pages: List[int],
) -> bool:
    if not expected_pages:
        return True

    retrieved_pages = {item["page"] for item in retrieved}
    return any(page in retrieved_pages for page in expected_pages)




def check_answer_contains(
    answer: str,
    expected_terms: List[str],
) -> bool:
    if not expected_terms:
        return True

    return all(term in answer for term in expected_terms)


def check_should_not_answer(answer: str) -> bool:
    return UNKNOWN_PHRASE in answer


def check_citation_exists(answer: str) -> bool:
    return "[" in answer


def evaluate_case(case: Dict[str, Any], store: SimpleVectorStore, top_k: int = 5) -> Dict[str, Any]:
    question = case["question"]
    expected_terms = case.get("expected_answer_contains", [])
    expected_pages = case.get("expected_pages", [])
    should_answer = case.get("should_answer", True)

    start = time.time()
    retrieved = store.search(question, top_k=top_k)
    answer = answer_with_llm(question, retrieved)

    latency = time.time() - start

    retrieval_hit = check_retrieval_hit(retrieved, expected_pages)
    citation_exists = check_citation_exists(answer)

    if should_answer:
        answer_contains = check_answer_contains(answer, expected_terms)
        refused_correctly = True
        passed = retrieval_hit and answer_contains and citation_exists
    else:
        answer_contains = True
        refused_correctly = check_should_not_answer(answer)
        passed = refused_correctly

    return {
        "id": case["id"],
        "question": question,
        "should_answer": should_answer,
        "passed": passed,
        "retrieval_hit": retrieval_hit,
        "answer_contains": answer_contains,
        "citation_exists": citation_exists,
        "refused_correctly": refused_correctly,
        "latency_sec": round(latency, 2),
        "answer": answer,
        "retrieved_pages": [item["page"] for item in retrieved],
        "top_chunk_ids": [item["chunk_id"] for item in retrieved],
    }




def print_report(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print("\nEval Summary")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {passed / total:.2%}")

    print("\nCase Details")
    print("=" * 80)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"

        print(f"\n[{status}] {r['id']}")
        print(f"Question: {r['question']}")
        print(f"Should answer: {r['should_answer']}")
        print(f"Retrieval hit: {r['retrieval_hit']}")
        print(f"Answer contains expected terms: {r['answer_contains']}")
        print(f"Citation exists: {r['citation_exists']}")
        print(f"Refused correctly: {r['refused_correctly']}")
        print(f"Latency: {r['latency_sec']}s")
        print(f"Retrieved pages: {r['retrieved_pages']}")
        print(f"Top chunk ids: {r['top_chunk_ids']}")
        print("Answer:")
        print(r["answer"])
        print("-" * 80)


        # token_usage
        # cost
        # groundedness_score
        # faithfulness_score
        # answer_relevance
        # context_precision
        # context_recall

def main():

    file_path = "data/project.pdf"
    eval_path = "evals/cases.yaml"

    print("Loading eval cases...")
    cases = load_eval_cases(eval_path)

    print("Loading and chunking PDF...")
    chunks = chunk_pdf(file_path)

    print("Building vector index...")
    store = SimpleVectorStore()
    store.build(chunks)

    results = []
    for case in cases:
        print(f"\nRunning {case['id']}...")
        result = evaluate_case(case, store, top_k=5)
        results.append(result)

    print_report(results)



  



if __name__ == "__main__":
    main()