
# CheckPointer

from langgraph.checkpoint.memory import InMemorySaver



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

    checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)

def main():
    # app = build_graph()
    pass


if __name__ == "__main__":
    main()


