import json
from pathlib import Path
from typing import Dict, Any, List

import yaml

from autogen.mini_autogen_conversation_demo import run_conversation, Message




CASES_PATH = Path("evals/autogen_pattern_cases.yaml")
RESULTS_PATH = Path("evals/autogen_pattern_eval_results.jsonl")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def extract_agent_sequence(messages: List[Message]) -> List[str]:
    return [
        msg.sender
        for msg in messages
        if msg.type in {"agent", "tool", "final"} and msg.sender != "User"
    ]


def get_final_message(messages: List[Message]) -> Message | None:
    for msg in reversed(messages):
        if msg.type == "final":
            return msg

    return None


def collect_all_text(messages: List[Message]) -> str:
    return "\n\n".join(
        msg.content
        for msg in messages
    )


def contains_all(text: str, terms: List[str]) -> bool:
    return all(term in text for term in terms)


def contains_none(text: str, terms: List[str]) -> bool:
    return all(term not in text for term in terms)


def check_agent_sequence(
    actual_agents: List[str],
    expected_agents: List[str],
) -> bool:
    """
    Check whether expected agents appear in the actual conversation order.

    It does not require exact equality, only ordered inclusion.
    """
    pos = 0

    for agent in actual_agents:
        if pos < len(expected_agents) and agent == expected_agents[pos]:
            pos += 1

    return pos == len(expected_agents)


def check_expected_tools(
    actual_tools: List[str],
    expected_tools: List[str],
) -> bool:
    return all(tool in actual_tools for tool in expected_tools)


def summarize_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    return [
        {
            "sender": msg.sender,
            "recipient": msg.recipient,
            "type": msg.type,
            "content_preview": msg.content[:300],
            "metadata": msg.metadata,
        }
        for msg in messages
    ]

def extract_tool_calls(messages: List[Message]) -> List[str]:
    tool_calls = []

    for msg in messages:
        tool_name = msg.metadata.get("tool_name")
        if tool_name:
            tool_calls.append(tool_name)

    return tool_calls

def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    messages = run_conversation(case["question"])
    actual_agents = extract_agent_sequence(messages)
    actual_tools = extract_tool_calls(messages)
    final_message = get_final_message(messages)
    all_text = collect_all_text(messages)

    final_text = final_message.content if final_message else ""

    agent_sequence_ok = check_agent_sequence(
        actual_agents=actual_agents,
        expected_agents=case.get("expected_agents", []),
    )

    tool_calls_ok = check_expected_tools(
        actual_tools=actual_tools,
        expected_tools=case.get("expected_tool_calls", []),
    )

    final_contains_ok = contains_all(
        final_text,
        case.get("expected_final_contains", []),
    )

    forbidden_ok = contains_none(
        all_text,
        case.get("forbidden_final_contains", []),
    )

    citations_ok = contains_all(
        all_text,
        case.get("required_citations", []),
    )

    terminated_ok = final_message is not None

    passed = all([
        agent_sequence_ok,
        tool_calls_ok,
        final_contains_ok,
        forbidden_ok,
        citations_ok,
        terminated_ok,
    ])

    return {
        "id": case["id"],
        "pipeline": "mini_autogen_conversation",
        "passed": passed,
        "checks": {
            "agent_sequence_ok": agent_sequence_ok,
            "tool_calls_ok": tool_calls_ok,
            "final_contains_ok": final_contains_ok,
            "forbidden_ok": forbidden_ok,
            "citations_ok": citations_ok,
            "terminated_ok": terminated_ok,
        },
        "expected": {
            "expected_agents": case.get("expected_agents", []),
            "expected_tool_calls": case.get("expected_tool_calls", []),
            "expected_final_contains": case.get("expected_final_contains", []),
            "forbidden_final_contains": case.get("forbidden_final_contains", []),
            "required_citations": case.get("required_citations", []),
        },
        "actual": {
            "actual_agents": actual_agents,
            "actual_tools": actual_tools,
            "final_text": final_text,
            "messages": summarize_messages(messages),
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

    print("\nMini AutoGen Pattern Eval Summary")
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
        print("Actual agents:", r.get("actual", {}).get("actual_agents"))
        print("Actual tools:", r.get("actual", {}).get("actual_tools"))
        print("Final:", r.get("actual", {}).get("final_text"))
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
                "pipeline": "mini_autogen_conversation",
                "passed": False,
                "error": str(e),
                "tags": case.get("tags", []),
            }

        results.append(record)
        write_result(record)

    print_summary(results)





if __name__ == "__main__":
    main()