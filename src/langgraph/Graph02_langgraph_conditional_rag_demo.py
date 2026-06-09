from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from datetime import datetime
import operator

from langgraph.graph import StateGraph, START, END
from rag.Step13_query_rewrite_demo import rewrite_query
from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output


class RAGState(TypedDict):
    # input
    question: str

    # query understanding
    rewritten_query: Optional[str]
    search_queries: List[str]
    intent: Optional[str]
    needs_context: bool
    missing_context: Optional[str]

    # retrieval
    candidates: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]

    # answer
    answerable: Optional[bool]
    answer: Optional[str]
    citations: List[int]
    confidence: Optional[float]
    missing_information: Optional[str]

    # tracing
    steps: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    # control
    status: str
    retry_count: int


store = HybridSearchStore()
reranker = Reranker()


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def rewrite_node(state: RAGState) -> Dict[str, Any]:
    try:
        rewrite = rewrite_query(state["question"])
        return {
            "rewritten_query": rewrite.rewritten_query,
            "search_queries": rewrite.search_queries,
            "intent": rewrite.intent,
            "needs_context": rewrite.needs_context,
            "missing_context": rewrite.missing_context,
            "status": "rewritten",
            "steps": [
                {
                    "node": "rewrite",
                    "status": "success",
                    "time": now(),
                    "data": rewrite.model_dump(),
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"rewrite_node failed: {e}"],
            "steps": [
                {
                    "node": "rewrite",
                    "status": "error",
                    "time": now(),
                    "error": str(e),
                }
            ],
        }


def clarify_node(state: RAGState) -> Dict[str, Any]:
    """
    Stop and ask user for missing context.
    In a real app, this would return a clarification question to the UI.
    """
    missing_context = state.get("missing_context") or "需要更多上下文。"

    answer = f"我需要先确认一下：{missing_context}"

    return {
        "answerable": False,
        "answer": answer,
        "citations": [],
        "confidence": 0.0,
        "missing_information": missing_context,
        "status": "need_clarification",
        "steps": [
            {
                "node": "clarify",
                "status": "need_clarification",
                "time": now(),
                "data": {
                    "missing_context": missing_context,
                },
            }
        ],
    }


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    try:
        candidates = multi_query_search(
            store=store,
            search_queries=state["search_queries"],
            per_query_k=10,
        )

        evidence = reranker.rerank(
            query=state["rewritten_query"] or state["question"],
            candidates=candidates,
            top_k=5,
        )

        return {
            "candidates": candidates,
            "evidence": evidence,
            "status": "retrieved",
            "steps": [
                {
                    "node": "retrieve",
                    "status": "success",
                    "time": now(),
                    "data": {
                        "num_candidates": len(candidates),
                        "num_evidence": len(evidence),
                        "top_chunk_ids": [item.get("chunk_id") for item in evidence],
                    },
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"retrieve_node failed: {e}"],
            "steps": [
                {
                    "node": "retrieve",
                    "status": "error",
                    "time": now(),
                    "error": str(e),
                }
            ],
        }


def retry_retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Retry retrieval once with a broader query.
    This is a simple version. Later we can call a smarter rewrite-again node.
    """
    try:
        retry_count = state["retry_count"] + 1

        broadened_queries = list(state["search_queries"])

        if state.get("rewritten_query"):
            broadened_queries.append(state["rewritten_query"])

        broadened_queries.append(state["question"])

        # simple dedup
        broadened_queries = list(dict.fromkeys(broadened_queries))

        candidates = multi_query_search(
            store=store,
            search_queries=broadened_queries,
            per_query_k=15,
        )

        evidence = reranker.rerank(
            query=state["rewritten_query"] or state["question"],
            candidates=candidates,
            top_k=5,
        )

        return {
            "retry_count": retry_count,
            "candidates": candidates,
            "evidence": evidence,
            "status": "retried_retrieve",
            "steps": [
                {
                    "node": "retry_retrieve",
                    "status": "success",
                    "time": now(),
                    "data": {
                        "retry_count": retry_count,
                        "queries": broadened_queries,
                        "num_candidates": len(candidates),
                        "num_evidence": len(evidence),
                    },
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"retry_retrieve_node failed: {e}"],
            "steps": [
                {
                    "node": "retry_retrieve",
                    "status": "error",
                    "time": now(),
                    "error": str(e),
                }
            ],
        }


def answer_node(state: RAGState) -> Dict[str, Any]:
    try:
        if not state["evidence"]:
            return {
                "answerable": False,
                "answer": "根据现有资料无法确定。",
                "citations": [],
                "confidence": 0.0,
                "missing_information": "没有检索到可用证据。",
                "status": "answered",
                "steps": [
                    {
                        "node": "answer",
                        "status": "no_evidence",
                        "time": now(),
                    }
                ],
            }

        result = answer_with_structured_output(
            question=state["question"],
            retrieved_chunks=state["evidence"],
        )

        return {
            "answerable": result.answerable,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "missing_information": result.missing_information,
            "status": "answered",
            "steps": [
                {
                    "node": "answer",
                    "status": "success",
                    "time": now(),
                    "data": result.model_dump(),
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"answer_node failed: {e}"],
            "steps": [
                {
                    "node": "answer",
                    "status": "error",
                    "time": now(),
                    "error": str(e),
                }
            ],
        }


def route_after_rewrite(state: RAGState) -> Literal["clarify", "retrieve", "end"]:
    """
    Decide next node after rewrite.
    """
    if state.get("status") == "failed":
        return "end"

    if state.get("needs_context"):
        return "clarify"

    return "retrieve"


def route_after_answer(state: RAGState) -> Literal["retry", "end"]:
    """
    Decide next node after answer.
    """
    if state.get("status") == "failed":
        return "end"

    if state.get("answerable") is True:
        return "end"

    if state.get("retry_count", 0) < 1:
        return "retry"

    return "end"


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retry_retrieve", retry_retrieve_node)
    graph.add_node("answer", answer_node)

    # 1. start -> rewrite user input
    graph.add_edge(START, "rewrite")

    # 2.check promopt
    graph.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "clarify": "clarify",
            "retrieve": "retrieve",
            "end": END,
        },
    )

    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "answer")

    # 3. check answer 
    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "retry": "retry_retrieve",
            "end": END,
        },
    )

    graph.add_edge("retry_retrieve", "answer")

    return graph.compile()


def initial_state(question: str) -> RAGState:
    return {
        "question": question,
        "rewritten_query": None,
        "search_queries": [],
        "intent": None,
        "needs_context": False,
        "missing_context": None,
        "candidates": [],
        "evidence": [],
        "answerable": None,
        "answer": None,
        "citations": [],
        "confidence": None,
        "missing_information": None,
        "steps": [],
        "errors": [],
        "status": "started",
        "retry_count": 0,
    }


def main():
    app = build_graph()
    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"q", "quit", "exit"}:
            break

        result = app.invoke(initial_state(question))
        
        print("\nFinal Answer")
        print("=" * 80)
        print(result.get("answer"))

        print("\nAnswer Metadata")
        print("=" * 80)
        print({
            "answerable": result.get("answerable"),
            "citations": result.get("citations"),
            "confidence": result.get("confidence"),
            "missing_information": result.get("missing_information"),
            "status": result.get("status"),
            "retry_count": result.get("retry_count"),
        })

        print("\nSteps")
        print("=" * 80)
        for step in result.get("steps", []):
            print(step)

        if result.get("errors"):
            print("\nErrors")
            print("=" * 80)
            for err in result["errors"]:
                print(err)


if __name__ == "__main__":
    main()
