import operator
import os
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, START, END

from rag.Step13_query_rewrite_demo import rewrite_query
from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def rewrite_node(state: RAGState) -> Dict[str, Any]:
    pass


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    pass


def answer_node(state: RAGState) -> Dict[str, Any]:
    pass


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

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
        print(
            {
                "answerable": result.get("answerable"),
                "citations": result.get("citations"),
                "confidence": result.get("confidence"),
                "missing_information": result.get("missing_information"),
                "status": result.get("status"),
            }
        )

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
