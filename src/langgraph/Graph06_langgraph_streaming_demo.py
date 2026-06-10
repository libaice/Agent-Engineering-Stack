from langgraph.Graph04_langgraph_memory_rag_demo import build_graph, make_turn_input


def main():
    app = build_graph()
    thread_id = input("Thread ID: ").strip() or "stream-demo-thread"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\nUser: ").strip()

        if question.lower() in {"q", "quit", "exit"}:
            break

        print("\nStreaming updates")
        print("=" * 80)

        final_state = None

        for update in app.stream(
            make_turn_input(question),
            config=config,
            stream_mode="updates",
            # stream_mode="values",
        ):
            print(update)

        final_state = app.get_state(config).values

        print("\nFinal Answer")
        print("=" * 80)
        print(final_state.get("answer"))

        print("\nStatus")
        print("=" * 80)
        print(
            {
                "status": final_state.get("status"),
                "answerable": final_state.get("answerable"),
                "confidence": final_state.get("confidence"),
            }
        )


def update_to_ui_event(update: dict) -> dict:
    node_name = list(update.keys())[0]
    payload = update[node_name]

    if node_name == "rewrite":
        return {
            "type": "node_completed",
            "node": "rewrite",
            "title": "Query rewrite completed",
            "data": {
                "rewritten_query": payload.get("rewritten_query"),
                "search_queries": payload.get("search_queries"),
            },
        }

    if node_name == "retrieve":
        return {
            "type": "node_completed",
            "node": "retrieve",
            "title": "Retrieved evidence",
            "data": {
                "num_candidates": len(payload.get("candidates", [])),
                "num_evidence": len(payload.get("evidence", [])),
            },
        }

    if node_name == "answer":
        return {
            "type": "node_completed",
            "node": "answer",
            "title": "Answer generated",
            "data": {
                "answer": payload.get("answer"),
                "citations": payload.get("citations"),
            },
        }

    return {
        "type": "node_completed",
        "node": node_name,
        "data": payload,
    }


if __name__ == "__main__":
    main()
