from mcp.server.fastmcp import FastMCP

mcp = FastMCP("knowledge-base-server")


@mcp.tool()
def search_project_documents(query: str, top_k: int = 5) -> str:
    """
    Search PMI Agent project documents for grounded evidence.

    Use this tool for questions about project pricing, payment terms,
    delivery timeline, implementation scope, risks, and evidence.
    """
    # TODO: replace with real LlamaIndex / hybrid retriever
    return """
[1] source=proposal.pdf page=3
PMI Agent 项目报价为 8500 美元。付款方式为 50% 预付款，50% 交付后支付。

[2] source=delivery_plan.pdf page=2
PMI Agent 的交付周期为两周。交付内容包括 FastAPI 后端、SSE workflow streaming、前端三栏 UI、RAG 检索和 eval harness。

[3] source=risk_report.pdf page=5
主要风险包括：数据源质量不稳定、预测市场流动性不足、LLM 检索结果可能存在噪声、引用证据需要校验。
"""


if __name__ == "__main__":
    mcp.run()
