from typing import Any, Dict, List, TypedDict

from crewai.tools import tool

class FakeMCPToolClient:
    """
    Fake MCP client adapter for local development.

    Later replace this with a real MCP client implementation that connects to:
    - stdio MCP server
    - Streamable HTTP MCP server
    - SSE MCP server
    """

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "search_project_documents":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            return self._search_project_documents(query=query, top_k=top_k)

        raise ValueError(f"Unknown MCP tool: {name}")

    def _search_project_documents(self, query: str, top_k: int = 5) -> str:
        return f"""
[1] source=proposal.pdf page=3
PMI Agent 项目报价为 8500 美元。付款方式为 50% 预付款，50% 交付后支付。

[2] source=delivery_plan.pdf page=2
PMI Agent 的交付周期为两周。交付内容包括 FastAPI 后端、SSE workflow streaming、前端三栏 UI、RAG 检索和 eval harness。

[3] source=risk_report.pdf page=5
主要风险包括：数据源质量不稳定、预测市场流动性不足、LLM 检索结果可能存在噪声、引用证据需要校验。

Query used: {query}
Top k: {top_k}
"""

tool_client = FakeMCPToolClient()



class AgentState(TypedDict):
    question: str
    rewritten_query: str
    evidence: str
    answer: str
    tool_trace: List[Dict[str, Any]]
    errors: List[str]

    
def retrieve_via_mcp_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_query") or state["question"]

    try:
        result = tool_client.call_tool(
            name="search_project_documents",
            arguments={
                "query": query,
                "top_k": 5,
            },
        )

        trace = {
            "tool_name": "search_project_documents",
            "arguments": {
                "query": query,
                "top_k": 5,
            },
            "status": "success",
            "result_preview": result[:300],
        }

        return {
            **state,
            "evidence": result,
            "tool_trace": state.get("tool_trace", []) + [trace],
        }

    except Exception as e:
        trace = {
            "tool_name": "search_project_documents",
            "arguments": {
                "query": query,
                "top_k": 5,
            },
            "status": "error",
            "error": str(e),
        }

        return {
            **state,
            "errors": state.get("errors", []) + [str(e)],
            "tool_trace": state.get("tool_trace", []) + [trace],
        }


@tool("search_project_documents")
def search_project_documents(query: str) -> str:
    """
    Search PMI Agent project documents through the MCP knowledge server.

    Use this tool for questions about project pricing, payment terms,
    delivery timeline, implementation scope, risks, and grounded evidence.
    """
    return tool_client.call_tool(
        name="search_project_documents",
        arguments={
            "query": query,
            "top_k": 5,
        },
    )