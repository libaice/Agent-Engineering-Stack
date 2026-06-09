import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from langgraph.Graph04_langgraph_memory_rag_demo import build_graph, make_turn_input

def test_run():
    app = build_graph()
    config = {"configurable": {"thread_id": "test-memory-session"}}

    # Turn 1
    question = "PMI项目是啥"
    print(f"\nUser: {question}")
    result = app.invoke(make_turn_input(question), config=config)

    print("\nAssistant:")
    print(result.get("answer"))

    print("\nStandalone Question:")
    print(result.get("standalone_question"))

    print("\nLong Term Memories retrieved:")
    for m in result.get("long_term_memories", []):
        print(f"- {m['name']}: {m['content']}")

if __name__ == "__main__":
    test_run()
