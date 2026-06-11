from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


StepType = Literal[
    "task_start",
    "tool_call",
    "observation",
    "file_write",
    "test_result",
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
class HarnessResult:
    task_id: str
    workspace: str
    success: bool
    reward: float
    trajectory: List[TrajectoryStep]
    final_test_output: str


class CodeAgentHarness:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name) / "sample_repo"
        self.trajectory: List[TrajectoryStep] = []
        self._step_id = 0

    def close(self) -> None:
        self.temp_dir.cleanup()

    def _record(
        self,
        step_type: StepType,
        content: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        observation: Optional[str] = None,
    ) -> None:
        self._step_id += 1

        self.trajectory.append(
            TrajectoryStep(
                step_id=self._step_id,
                step_type=step_type,
                content=content,
                tool_name=tool_name,
                tool_args=tool_args or {},
                observation=observation,
            )
        )

    def setup_sample_repo(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)

        calculator_py = self.workspace / "calculator.py"
        test_py = self.workspace / "test_calculator.py"

        calculator_py.write_text(
            """
def add(a, b):
    return a - b


def multiply(a, b):
    return a * b
""".strip()
            + "\n",
            encoding="utf-8",
        )

        test_py.write_text(
            """
from calculator import add, multiply


def test_add():
    assert add(2, 3) == 5


def test_multiply():
    assert multiply(2, 3) == 6
""".strip()
            + "\n",
            encoding="utf-8",
        )

        self._record(
            step_type="task_start",
            content="Created sample repo with a failing add() implementation.",
        )

    def read_file(self, relative_path: str) -> str:
        path = self.workspace / relative_path

        self._record(
            step_type="tool_call",
            content=f"Reading file: {relative_path}",
            tool_name="read_file",
            tool_args={"path": relative_path},
        )

        if not path.exists():
            observation = f"File not found: {relative_path}"
            self._record(
                step_type="observation",
                content=observation,
                observation=observation,
            )
            return observation

        content = path.read_text(encoding="utf-8")

        self._record(
            step_type="observation",
            content=f"Read {relative_path}",
            observation=content,
        )

        return content

    def write_file(self, relative_path: str, content: str) -> None:
        path = self.workspace / relative_path

        self._record(
            step_type="tool_call",
            content=f"Writing file: {relative_path}",
            tool_name="write_file",
            tool_args={"path": relative_path, "content_preview": content[:120]},
        )

        path.write_text(content, encoding="utf-8")

        self._record(
            step_type="file_write",
            content=f"Wrote file: {relative_path}",
            observation=content,
        )

    def run_tests(self, timeout_seconds: int = 10) -> str:
        self._record(
            step_type="tool_call",
            content="Running pytest",
            tool_name="run_tests",
            tool_args={"timeout_seconds": timeout_seconds},
        )

        try:
            completed = subprocess.run(
                ["python", "-m", "pytest", "-q"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            output = (
                f"returncode={completed.returncode}\n\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            )

        except subprocess.TimeoutExpired:
            output = "TEST_TIMEOUT"

        self._record(
            step_type="test_result",
            content="Pytest finished",
            observation=output,
        )

        return output

    def get_file_snapshot(self) -> Dict[str, str]:
        snapshot = {}

        for path in self.workspace.glob("*.py"):
            snapshot[path.name] = path.read_text(encoding="utf-8")

        return snapshot

    @staticmethod
    def tests_passed(test_output: str) -> bool:
        return "returncode=0" in test_output

    @staticmethod
    def calculate_reward(test_output: str) -> float:
        return 1.0 if CodeAgentHarness.tests_passed(test_output) else 0.0

    def save_trajectory(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = [asdict(step) for step in self.trajectory]

        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )



def run_manual_bugfix_agent() -> HarnessResult:
    harness = CodeAgentHarness(task_id="bugfix_add_001")

    try:
        harness.setup_sample_repo()
        before = harness.run_tests()

        source = harness.read_file("calculator.py")

        fixed_source = source.replace("return a - b", "return a + b")

        harness.write_file("calculator.py", fixed_source)

        after = harness.run_tests()

        success = harness.tests_passed(after)
        reward = harness.calculate_reward(after)

        harness._record(
            step_type="task_end",
            content=f"Task completed. success={success}, reward={reward}",
        )

        result = HarnessResult(
            task_id=harness.task_id,
            workspace=str(harness.workspace),
            success=success,
            reward=reward,
            trajectory=harness.trajectory,
            final_test_output=after,
        )

        harness.save_trajectory(Path("outputs/trajectory_bugfix_add_001.json"))

        return result

    finally:
        # harness.cleanup()
        pass


def main():
    result = run_manual_bugfix_agent()

    print("\nHarness Result")
    print("=" * 80)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    print("\nTrajectory saved to:")
    print("outputs/trajectory_bugfix_add_001.json")


if __name__ == "__main__":
    main()
