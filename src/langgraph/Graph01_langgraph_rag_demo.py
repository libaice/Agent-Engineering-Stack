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

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


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
    print(f"\n[{now()}] === 开始执行 rewrite_node ===")
    print(f"原始问题: {state['question']}")
    try:
        rewrite = rewrite_query(state["question"])
        print(f"[{now()}] --- rewrite_node 执行成功 ---")
        print(f"改写后的问题: {rewrite.rewritten_query}")
        print(f"生成子查询数: {len(rewrite.search_queries)} {rewrite.search_queries}")
        print(f"用户意图: {rewrite.intent}")
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
        print(f"[{now()}] --- rewrite_node 发生错误: {e} ---")
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


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    print(f"\n[{now()}] === 开始执行 retrieve_node ===")
    print(f"检索查询词: {state['search_queries']}")
    try:
        candidates = multi_query_search(
            store=store,
            search_queries=state["search_queries"],
            per_query_k=10,
        )
        print(f"[{now()}] 召回候选块数量: {len(candidates)}")
        print("开始使用 Cross-Encoder 进行重排 (耗时较长)...")
        evidence = reranker.rerank(
            query=state["rewritten_query"] or state["question"],
            candidates=candidates,
            top_k=5,
        )
        print(f"[{now()}] --- retrieve_node 执行成功 ---")
        print(f"重排后保留的证据数: {len(evidence)}")
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
        print(f"[{now()}] --- retrieve_node 发生错误: {e} ---")
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


def answer_node(state: RAGState) -> Dict[str, Any]:
    print(f"\n[{now()}] === 开始执行 answer_node ===")
    try:
        if not state["evidence"]:
            print(f"[{now()}] 无证据，直接拒绝回答。")
            return {
                "answerable": False,
                "answer": "根据现有资料无法确定。",
                "citations": [],
                "confidence": 0.0,
                "missing_information": "没有检索到可用证据。",
                "status": "done",
                "steps": [
                    {
                        "node": "answer",
                        "status": "no_evidence",
                        "time": now(),
                    }
                ],
            }

        print(f"证据数量: {len(state['evidence'])}，开始调用 LLM 生成回答...")
        result = answer_with_structured_output(
            question=state["question"],
            retrieved_chunks=state["evidence"],
        )
        print(f"[{now()}] --- answer_node 执行成功 ---")
        print(f"可回答性 (answerable): {result.answerable}")
        print(f"置信度 (confidence): {result.confidence}")
        return {
            "answerable": result.answerable,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "missing_information": result.missing_information,
            "status": "done",
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
        print(f"[{now()}] --- answer_node 发生错误: {e} ---")
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
