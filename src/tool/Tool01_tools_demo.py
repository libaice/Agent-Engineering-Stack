from typing import Dict, Any, Callable, List

from rag.Step13_query_rewrite_demo import rewrite_query

from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output


class ToolContext:
    def __init__(self):
        self.store = HybridSearchStore()
        self.reranker = Reranker()


def retrieve_documents_tool(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    context: ToolContext | None = None,
) -> Dict[str, Any]:
    if context is None:
        context = ToolContext()
    rewrite = rewrite_query(query)
    candidates = multi_query_search(
        store=context.store,
        search_queries=rewrite.search_queries,
        per_query_k=10,
    )
    reranked = context.reranker.rerank(
        query=rewrite.rewritten_query,
        candidates=candidates,
        top_k=top_k,
    )

    return {
        "tool": "retrieve_documents",
        "query": query,
        "rewrite": rewrite.model_dump(),
        "results": reranked,
    }


def answer_from_documents_tool(
    question: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    answer = answer_with_structured_output(
        question=question,
        retrieved_chunks=evidence,
    )
    return {
        "tool": "answer_from_documents",
        "answer": answer.model_dump(),
    }


class ToolRegistry:
    def __init__(self, context: ToolContext):
        self.context = context
        self.tools: Dict[str, Callable[..., Dict[str, Any]]] = {
            "retrieve_documents": self._retrieve_documents,
            "answer_from_documents": self._answer_from_documents,
        }

    def _retrieve_documents(self, **kwargs) -> Dict[str, Any]:
        return retrieve_documents_tool(
            context=self.context,
            **kwargs,
        )

    def _answer_from_documents(self, **kwargs) -> Dict[str, Any]:
        return answer_from_documents_tool(**kwargs)

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        return self.tools[tool_name](**arguments)


if __name__ == "__main__":
    context = ToolContext()
    registry = ToolRegistry(context)
    result = registry.call(
        "retrieve_documents",
        {
            "query": "这个项目的Retrieval Eval 设计",
            "top_k": 5,
            "candidate_k": 20,
        },
    )
    print(result.keys())
    print(result["rewrite"])
    print(result["results"][0]["text"][:500])
