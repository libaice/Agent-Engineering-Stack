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
class HarnessResult:
    task_id: str
    description: str
    success: bool
    reward: float
    final_test_output: str
    diff: str
    trajectory: List[TrajectoryStep]




def run_eval() -> List[HarnessResult]:
    pass



def main() -> None:
    results = run_eval()


if __name__ == "__main__":
    main()