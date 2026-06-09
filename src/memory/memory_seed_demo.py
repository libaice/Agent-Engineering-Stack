from memory.memory_store import JsonMemoryStore



def main():
    store = JsonMemoryStore()

    store.add_memory(
        memory_type="project",
        name="PMI Agent",
        aliases=[
            "Prediction Market Intelligence Agent",
            "Polymarket Agent",
            "预测市场 Agent",
            "PMI"
        ],
        content=(
            "用户正在构建 Prediction Market Intelligence Agent。"
            "定位是 Bloomberg + Perplexity for prediction markets，"
            "用于预测市场信息聚合、证据检索、结构化 memo、source verification 和 eval。"
        ),
        importance=0.95,
        metadata={
            "tech_stack": ["FastAPI", "LangGraph", "RAG", "SSE", "Next.js"],
            "repo": "OrderBookTrade/Prediction-Market-Intelligence-Agent",
        }
    )

    store.add_memory(
        memory_type="career_goal",
        name="AI Agent / FDE 转型",
        aliases=["Agent Engineer", "FDE", "AI Agent 工程师"],
        content=(
            "用户正在从 Crypto Engineer 转向 AI Agent Engineer / FDE。"
            "当前学习重点是 Production RAG、LangGraph、Tool Calling、Eval Harness、Observability。"
        ),
        importance=0.9,
        metadata={
            "target_roles": ["AI Agent Engineer", "FDE", "AI Native Full-stack"]
        }
    )

    print("Seeded memories.")




if __name__ == "__main__":
    main()