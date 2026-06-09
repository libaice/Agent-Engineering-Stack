from langgraph.constants import END
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.Graph04_langgraph_memory_rag_demo import MemoryRAGState
from rag.Step13_query_rewrite_demo import rewrite_query


from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output

from langgraph.checkpoint.memory import InMemorySaver

from langgraph.Graph02_langgraph_conditional_rag_demo import (
    RAGState,
    clarify_node,
    retrieve_node,
    retry_retrieve_node,
    route_after_rewrite,
    route_after_answer,
    initial_state,
)


from langgraph.Graph04_langgraph_memory_rag_demo import (
    load_memory_node,
    contextualize_question_node,
    rewrite_node,
    retrieve_node,
    answer_node,
    extract_memory_node,
    save_memory_node,
)


def human_review_memory_node(state: MemoryRAGState) -> Dict[str, Any]:
    candidate = state.get("memory_candidate")

    if not candidate or not candidate.get("should_save"):
        return {
            "memory_approval": {
                "action": "skip",
                "reason": "No memory candidate to review.",
            },
            "steps": [
                {
                    "node": "human_review_memory",
                    "status": "skipped",
                    "data": "No memory candidate.",
                }
            ],
        }

    human_decision = interrupt(
        {
            "type": "memory_review",
            "message": "是否保存这条长期记忆？",
            "candidate": candidate,
            "allowed_actions": ["approve", "reject", "edit"],
            "edit_schema": {
                "memory_type": "string",
                "name": "string",
                "content": "string",
                "aliases": ["string"],
                "importance": "float",
            },
        }
    )

    return {
        "memory_approval": human_decision,
        "steps": [
            {
                "node": "human_review_memory",
                "status": "reviewed",
                "data": human_decision,
            }
        ],
    }


def route_after_memory_review(
    state: MemoryRAGState,
) -> Literal["save_memory", "end"]:
    approval = state.get("memory_approval") or {}
    action = approval.get("action")

    if action in {"approve", "edit", "reject", "skip"}:
        return "save_memory"

    return "end"


def build_graph():
    graph = StateGraph(MemoryRAGState)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("contextualize_question", contextualize_question_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("extract_memory", extract_memory_node)
    graph.add_node("human_review_memory", human_review_memory_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "contextualize_question")
    graph.add_edge("contextualize_question", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "extract_memory")
    graph.add_edge("extract_memory", "human_review_memory")

    graph.add_conditional_edges(
        "human_review_memory",
        route_after_memory_review,
        {
            "save_memory": "save_memory",
            "end": END,
        },
    )

    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=InMemorySaver())


def main():
    app = build_graph()

    thread_id = input("Thread ID: ").strip() or "hitl-memory-thread"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\nUser: ").strip()

        if question.lower() in {"q", "quit", "exit"}:
            break

        result = app.invoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
            },
            config=config,
        )

        if "__interrupt__" in result:
            print("\nGraph interrupted.")
            print("=" * 80)
            print(result["__interrupt__"])

            action = input("\nApprove memory? [approve/reject]: ").strip()

            if action == "approve":
                resume_value = {"action": "approve"}
            else:
                reason = input("Reject reason: ").strip()
                resume_value = {
                    "action": "reject",
                    "reason": reason or "User rejected memory save.",
                }

            result = app.invoke(
                Command(resume=resume_value),
                config=config,
            )

        print("\nAssistant:")
        print(result.get("answer"))

        print("\nSteps:")
        for step in result.get("steps", [])[-8:]:
            print(step)


if __name__ == "__main__":
    main()
