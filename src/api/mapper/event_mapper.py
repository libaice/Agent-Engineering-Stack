from datetime import datetime
from typing import Any, Dict


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"







def sse(event: Dict[str, Any]) -> str:
    import json
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"