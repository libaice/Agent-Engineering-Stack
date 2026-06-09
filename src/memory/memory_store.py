import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4


MEMORY_PATH = Path("storage/memory.json")


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class JsonMemoryStore:
    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save([])

    def _load(self) -> List[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, memories: List[Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)

    def list_memories(self) -> List[Dict[str, Any]]:
        return self._load()

    def add_memory(
        self,
        memory_type: str,
        name: str,
        content: str,
        aliases: Optional[List[str]] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pass

    def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass



