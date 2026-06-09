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
            stream_mode="values",
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


if __name__ == "__main__":
    main()
