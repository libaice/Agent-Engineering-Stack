import json
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.Graph04_langgraph_memory_rag_demo import build_graph, make_turn_input


EVAL_CASES_PATH = Path("evals/agent_cases.yaml")
EVAL_RESULTS_PATH = Path("evals/agent_eval_results.jsonl")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def contains_all(text: str | None, terms: List[str]) -> bool:
    if not terms:
        return True

    text = text or ""
    return all(term in text for term in terms)


def build_input_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    question = case["question"]

    messages = []

    for msg in case.get("conversation", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    if not messages or messages[-1].content != question:
        messages.append(HumanMessage(content=question))

    base_input = make_turn_input(question)
    base_input["messages"] = messages

    return base_input


def check_retrieval_pages(
    evidence: List[Dict[str, Any]],
    expected_pages: List[int],
) -> bool:
    if not expected_pages:
        return True

    retrieved_pages = {
        item.get("page") for item in evidence if item.get("page") is not None
    }

    return any(page in retrieved_pages for page in expected_pages)


def check_citations(
    citations: List[int],
    citation_required: bool,
) -> bool:
    if citation_required:
        return bool(citations)

    return True


def write_result(record: Dict[str, Any]) -> None:
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EVAL_RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_case(app, case: Dict[str, Any]) -> Dict[str, Any]:
    thread_id = f"eval-{case['id']}"

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "eval_case_id": case["id"],
            "tags": case.get("tags", []),
        },
        "tags": ["agent-eval"] + case.get("tags", []),
    }

    start = time.perf_counter()

    result = app.invoke(
        build_input_from_case(case),
        config=config,
    )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    should_answer = case.get("should_answer", True)

    standalone_ok = contains_all(
        result.get("standalone_question"),
        case.get("expected_standalone_question_contains", []),
    )

    retrieval_hit = check_retrieval_pages(
        result.get("evidence", []),
        case.get("expected_evidence_pages", []),
    )

    answer_contains = contains_all(
        result.get("answer"),
        case.get("expected_answer_contains", []),
    )

    citation_ok = check_citations(
        result.get("citations", []),
        case.get("citation_required", True),
    )

    answerability_ok = (
        result.get("answerable") is True
        if should_answer
        else result.get("answerable") is False
    )

    no_errors = len(result.get("errors", [])) == 0

    passed = all(
        [
            standalone_ok,
            retrieval_hit,
            answer_contains,
            citation_ok,
            answerability_ok,
            no_errors,
        ]
    )

    return {
        "id": case["id"],
        "passed": passed,
        "latency_ms": latency_ms,
        "checks": {
            "standalone_ok": standalone_ok,
            "retrieval_hit": retrieval_hit,
            "answer_contains": answer_contains,
            "citation_ok": citation_ok,
            "answerability_ok": answerability_ok,
            "no_errors": no_errors,
        },
        "expected": {
            "should_answer": should_answer,
            "expected_answer_contains": case.get("expected_answer_contains", []),
            "expected_standalone_question_contains": case.get(
                "expected_standalone_question_contains", []
            ),
            "expected_evidence_pages": case.get("expected_evidence_pages", []),
            "citation_required": case.get("citation_required", True),
        },
        "actual": {
            "standalone_question": result.get("standalone_question"),
            "answerable": result.get("answerable"),
            "answer": result.get("answer"),
            "citations": result.get("citations"),
            "confidence": result.get("confidence"),
            "status": result.get("status"),
            "retry_count": result.get("retry_count"),
            "errors": result.get("errors", []),
            "evidence_summary": [
                {
                    "chunk_id": item.get("chunk_id"),
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "sheet_name": item.get("sheet_name"),
                    "row_index": item.get("row_index"),
                    "rerank_score": item.get("rerank_score"),
                    "text_preview": (item.get("text") or "")[:200],
                }
                for item in result.get("evidence", [])
            ],
        },
        "tags": case.get("tags", []),
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    print("\nAgent Eval Summary")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {passed / total:.2%}")

    print("\nFailures")
    print("=" * 80)

    for r in results:
        if r["passed"]:
            continue

        print(f"\n[FAIL] {r['id']}")
        print("Checks:", r["checks"])
        print("Question standalone:", r["actual"]["standalone_question"])
        print("Answerable:", r["actual"]["answerable"])
        print("Answer:", r["actual"]["answer"])
        print("Evidence:")
        for ev in r["actual"]["evidence_summary"]:
            print(ev)
        print("-" * 80)


def main():
    cases = load_cases(EVAL_CASES_PATH)
    app = build_graph()
    results = []

    if EVAL_RESULTS_PATH.exists():
        EVAL_RESULTS_PATH.unlink()

    for case in cases:
        print(f"Running {case['id']}...")
        record = evaluate_case(app, case)
        results.append(record)
        write_result(record)

    print_summary(results)


if __name__ == "__main__":
    main()
