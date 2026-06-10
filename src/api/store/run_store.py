from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"



class InMemoryRunStore:
    def __init__(self):
        self.runs: Dict[str, Dict[str, Any]] = {}


