import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4


import jieba

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

        memories = self._load()

        memory = {
            "memory_id": f"mem_{uuid4().hex[:8]}",
            "type": memory_type,
            "name": name,
            "aliases": aliases or [],
            "content": content,
            "importance": importance,
            "metadata": metadata or {},
            "created_at": now(),
            "updated_at": now(),
        }
        memories.append(memory)
        self._save(memories)

        return memory

        pass

    def format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return "no long term memory。"
        blocks = []

        for m in memories:
            blocks.append(
                f"- [{m['memory_id']}] type={m['type']} name={m['name']}\n"
                f"  aliases={m.get('aliases', [])}\n"
                f"  content={m['content']}"
            )
        return "\n".join(blocks)

    def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        memories = self._load()
        query_lower = query.lower()

        # Tokenize query using jieba for robust Chinese matching
        query_tokens = [t.lower() for t in jieba.cut(query_lower) if t.strip()]

        scored = []

        for memory in memories:
            text = " ".join(
                [
                    memory.get("name", ""),
                    " ".join(memory.get("aliases", [])),
                    memory.get("content", ""),
                    json.dumps(memory.get("metadata", {}), ensure_ascii=False),
                ]
            ).lower()

            score = 0

            # Match individual tokens
            for token in query_tokens:
                if token in text:
                    score += 1

            # Bonus for exact full match
            if query_lower in text:
                score += 3

            if score > 0:
                scored.append((score + memory.get("importance", 0), memory))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [m for _, m in scored[:top_k]]
