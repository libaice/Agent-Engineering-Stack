from tool.Tool01_tools_demo import ToolContext, ToolRegistry
from tool.Tool02_tool_router_demo import decide_tool


def run_document_qa_agent(question: str) -> dict:
    context = ToolContext()
    registry = ToolRegistry(context)

    steps = []

    # 1. Router选择工具
    decision = decide_tool(question)
    steps.append(
        {
            "type": "tool_decision",
            "decision": decision.model_dump(),
        }
    )

    if decision.tool_name != "retrieve_documents":
        raise ValueError(
            f"Expected retrieve_documents as first tool, got {decision.tool_name}"
        )

    retrieve_result = registry.call(
        decision.tool_name,
        decision.arguments,
    )

    steps.append({
        "type": "tool_result",
        "tool_name": "retrieve_documents",
        "result_summary": {
            "num_results": len(retrieve_result["results"]),
            "rewrite": retrieve_result["rewrite"],
        },
    })

    # Step 2: answer from retrieved evidence
    answer_result = registry.call(
        "answer_from_documents",
        {
            "question": question,
            "evidence": retrieve_result["results"],
        },
    )

    steps.append({
        "type": "tool_result",
        "tool_name": "answer_from_documents",
        "result": answer_result["answer"],
    })

    return {
        "question": question,
        "steps": steps,
        "final_answer": answer_result["answer"],
        "evidence": retrieve_result["results"],
    }

    pass


def main():
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        result = run_document_qa_agent(question)
        print("\nAgent Steps")
        print("=" * 80)
        for step in result["steps"]:
            print(step)

        print("\nFinal Answer")
        print("=" * 80)
        print(result["final_answer"])


if __name__ == "__main__":
    main()
