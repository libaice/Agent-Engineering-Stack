from __future__ import annotations

import difflib
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


StepType = Literal[
    "task_start",
    "tool_call",
    "observation",
    "file_write",
    "diff",
    "test_result",
    "policy_denied",
    "task_end",
]


@dataclass
class TrajectoryStep:
    step_id: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    observation: Optional[str] = None


@dataclass
class CodeTask:
    task_id: str
    description: str
    bug_type: str

@dataclass
class HarnessResult:
    task_id: str
    description: str
    success: bool
    reward: float
    final_test_output: str
    diff: str
    trajectory: List[TrajectoryStep]


def run_eval() -> List[HarnessResult]:
    tasks = [
        CodeTask(
            task_id="bugfix_add_001",
            description="Fix add(a, b), which incorrectly subtracts b from a.",
            bug_type="add_bug",
        ),
        CodeTask(
            task_id="bugfix_divide_001",
            description="Fix divide(a, b), which incorrectly multiplies instead of divides.",
            bug_type="divide_bug",
        ),
    ]


def print_summary(results: List[HarnessResult]) -> None:
    total = len(results)
    passed = sum(1 for result in results if result.success)

    print("\nCode Agent Harness Eval Summary")
    print("=" * 80)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed / total:.2%}" if total else "Success rate: N/A")

    for result in results:
        print("\n" + "-" * 80)
        print(f"Task: {result.task_id}")
        print(f"Success: {result.success}")
        print(f"Reward: {result.reward}")
        print("\nDiff:")
        print(result.diff)


def main() -> None:
    results = run_eval()

    print_summary(results)

    Path("outputs/results.json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )



if __name__ == "__main__":
    main()
