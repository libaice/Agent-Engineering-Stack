import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any


LOG_PATH = Path("logs/runs.jsonl")


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_run_log(
    thread_id: str,
    final_state: Dict[str, Any],
    log_path: Path = LOG_PATH,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": f"run_{uuid4().hex[:8]}",
        "thread_id": thread_id,
        "question": final_state.get("question"),
        "standalone_question": final_state.get("standalone_question"),
        "answer": final_state.get("answer"),
        "answerable": final_state.get("answerable"),
        "citations": final_state.get("citations"),
        "confidence": final_state.get("confidence"),
        "status": final_state.get("status"),
        "retry_count": final_state.get("retry_count"),
        "steps": final_state.get("steps", []),
        "errors": final_state.get("errors", []),
        "created_at": now(),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
