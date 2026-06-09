
from typing import Dict, Any
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage
from langgraph.Graph04_langgraph_memory_rag_demo import MemoryRAGState



def human_review_memory_node(state: MemoryRAGState) -> Dict[str, Any]:
    pass



def build_graph():
    pass

def main():
    app = build_graph()

    thread_id = input("Thread ID: ").strip() or "hitl-memory-thread"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

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
                    "reason": reason or "User rejected memory save."
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