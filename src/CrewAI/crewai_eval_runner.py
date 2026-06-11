import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml

# Add the 'src' directory to sys.path so imports are resolved relative to 'src'
# (consistent with setuptools src-layout configuration and IDE/linter path settings)
_src_path = str(Path(__file__).resolve().parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from CrewAI.crewai_structured_critic_demo import run_crewai_pipeline

CASES_PATH = Path("evals/crewai_cases.yaml")
RESULTS_PATH = Path("evals/crewai_eval_results.jsonl")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_memo_text(memo) -> str:
    parts = [
        memo.project,
        memo.summary,
        memo.reasoning,
        memo.recommendation,
        " ".join(memo.next_actions),
    ]

    for fact in memo.key_facts:
        parts.append(fact.claim)

    for risk in memo.risks:
        parts.append(risk.risk)
        if risk.mitigation:
            parts.append(risk.mitigation)

    return "\n".join(parts)


def contains_all(text: str, terms: List[str]) -> bool:
    return all(term in text for term in terms)


def contains_any_invalid(text: str, forbidden_terms: List[str]) -> bool:
    return any(term in text for term in forbidden_terms)


def collect_used_citation_ids(memo) -> set[int]:
    ids = set()

    for fact in memo.key_facts:
        ids.update(fact.citations)

    for risk in memo.risks:
        ids.update(risk.citations)

    return ids


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    pipeline_result = run_crewai_pipeline(
        question=case["question"],
        project=case.get("project", "PMI Agent"),
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    evidence_summary = pipeline_result["evidence_summary"]
    memo = pipeline_result["memo"]
    review = pipeline_result["review"]
    deterministic_eval = pipeline_result["eval_result"]

    memo_text = collect_memo_text(memo)

    answer_contains_ok = contains_all(
        memo_text,
        case.get("expected_answer_contains", []),
    )

    forbidden_ok = not contains_any_invalid(
        memo_text,
        case.get("forbidden_answer_contains", []),
    )

    recommendation_ok = memo.recommendation in case.get(
        "expected_recommendations",
        ["continue", "pause", "reject", "clarify"],
    )
    used_citation_ids = collect_used_citation_ids(memo)

    required_citations_ok = set(case.get("required_citation_ids", [])).issubset(
        used_citation_ids
    )

    critic_ok = review.verdict in {"pass", "needs_revision"}

    deterministic_ok = deterministic_eval["deterministic_passed"]

    passed = all(
        [
            answer_contains_ok,
            forbidden_ok,
            recommendation_ok,
            required_citations_ok,
            critic_ok,
            deterministic_ok,
        ]
    )

    return {
        "id": case["id"],
        "pipeline": "crewai_structured_critic",
        "passed": passed,
        "latency_ms": latency_ms,
        "checks": {
            "answer_contains_ok": answer_contains_ok,
            "forbidden_ok": forbidden_ok,
            "recommendation_ok": recommendation_ok,
            "required_citations_ok": required_citations_ok,
            "critic_ok": critic_ok,
            "deterministic_ok": deterministic_ok,
        },
        "expected": {
            "expected_answer_contains": case.get("expected_answer_contains", []),
            "forbidden_answer_contains": case.get("forbidden_answer_contains", []),
            "expected_recommendations": case.get("expected_recommendations", []),
            "required_citation_ids": case.get("required_citation_ids", []),
        },
        "actual": {
            "recommendation": memo.recommendation,
            "used_citation_ids": sorted(used_citation_ids),
            "critic_verdict": review.verdict,
            "memo": memo.model_dump(),
            "review": review.model_dump(),
            "evidence_summary": evidence_summary.model_dump(),
            "deterministic_eval": deterministic_eval,
        },
        "tags": case.get("tags", []),
    }


def write_result(record: Dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(results: List[Dict[str, Any]]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))

    print("\nCrewAI Eval Summary")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {passed / total:.2%}" if total else "Pass rate: N/A")

    print("\nFailures")
    print("=" * 80)

    for r in results:
        if r.get("passed"):
            continue

        print(f"\n[FAIL] {r.get('id')}")
        print("Checks:", r.get("checks"))
        print("Recommendation:", r.get("actual", {}).get("recommendation"))
        print("Used citations:", r.get("actual", {}).get("used_citation_ids"))
        print("Critic verdict:", r.get("actual", {}).get("critic_verdict"))
        print("Deterministic eval:", r.get("actual", {}).get("deterministic_eval"))
        print("-" * 80)


def main():
    cases = load_cases(CASES_PATH)

    results = []

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    for case in cases:
        print(f"Running {case['id']}...")

        try:
            record = evaluate_case(case)
        except Exception as e:
            record = {
                "id": case["id"],
                "pipeline": "crewai_structured_critic",
                "passed": False,
                "error": str(e),
                "tags": case.get("tags", []),
            }

        results.append(record)
        write_result(record)

    print_summary(results)


if __name__ == "__main__":
    main()
