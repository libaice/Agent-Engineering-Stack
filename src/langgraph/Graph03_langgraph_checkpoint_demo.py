# CheckPointer

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from rag.Step13_query_rewrite_demo import rewrite_query
from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output


from langgraph.Graph02_langgraph_conditional_rag_demo import (
    RAGState,
    rewrite_node,
    clarify_node,
    retrieve_node,
    retry_retrieve_node,
    answer_node,
    route_after_rewrite,
    route_after_answer,
    initial_state
)


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retry_retrieve", retry_retrieve_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "rewrite")

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

    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "retry": "retry_retrieve",
            "end": END,
        },
    )

    graph.add_edge("retry_retrieve", "answer")

    # add checkpointer for persistance
    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)


def main():
    app = build_graph()
    thread_id = input("Thread ID: ").strip() or "rag-thread-001"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        result = app.invoke(initial_state(question), config=config)

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

        print("\nLatest State Snapshot")
        print("=" * 80)
        latest_state = app.get_state(config)
        print(latest_state)




if __name__ == "__main__":
    main()
