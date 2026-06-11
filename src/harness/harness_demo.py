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



def run_manual_bugfix_agent() -> HarnessResult:
    harness = CodeAgentHarness(task_id="bugfix_add_001")

    pass

def main():
    result = run_manual_bugfix_agent()
    print("\nHarness Result")

    print("=" * 80)
    pass


if __name__ == "__main__":
    main()
