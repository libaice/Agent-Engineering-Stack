import json
import os
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from tool.Tool01_tools_demo import ToolContext, ToolRegistry


load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


class ToolCallDecision(BaseModel):
    tool_name: str = Field(description="The tool to call.")
    arguments: Dict[str, Any] = Field(description="Arguments for the selected tool.")
    reason: str = Field(description="Why this tool is selected.")


AVAILABLE_TOOLS = [
    {
        "name": "retrieve_documents",
        "description": "Search relevant chunks from PDF/Excel document knowledge base. Use this for factual questions about documents, contracts, project proposals, risks, pricing, timelines, status, and evidence lookup.",
        "arguments": {
            "query": "string: the search query",
            "top_k": "integer: number of evidence chunks to return",
            "candidate_k": "integer: number of recall candidates before rerank",
        },
    },
    {
        "name": "answer_from_documents",
        "description": "Generate a structured answer from retrieved evidence. Use this only after retrieve_documents has returned evidence.",
        "arguments": {
            "question": "string: original user question",
            "evidence": "list: retrieved evidence chunks",
        },
    },
]


def build_tool_router_prompt(question: str) -> str:
    return f"""
你是一个 Agent 的工具路由器。

你的任务是根据用户问题，选择下一步应该调用的工具。

当前可用工具：
{json.dumps(AVAILABLE_TOOLS, ensure_ascii=False, indent=2)}

规则：
1. 如果用户问的是文档里的事实、报价、风险、周期、条款、状态，选择 retrieve_documents。
2. 如果用户已经提供了 evidence，并要求基于 evidence 回答，才选择 answer_from_documents。
3. 不要直接回答用户问题，只输出工具调用决策。
4. 必须输出严格 JSON，不要输出 markdown。

用户问题：
{question}

输出 JSON 格式：
{{
  "tool_name": "retrieve_documents",
  "arguments": {{
    "query": "string",
    "top_k": 5,
    "candidate_k": 20
  }},
  "reason": "string"
}}
""".strip()


def decide_tool(question: str, model: str = "deepseek-v4-pro"):
    prompt = build_tool_router_prompt(question)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You select tools for an agent and output valid JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from tool router: {raw}") from e

    try:
        return ToolCallDecision.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Tool decision validation failed: {data}") from e


def main():
    context = ToolContext()
    registry = ToolRegistry(context)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        decision = decide_tool(question)

        print("\nTool Decision")
        print("=" * 80)
        print(decision.model_dump_json(indent=2))

        tool_result = registry.call(
            decision.tool_name,
            decision.arguments,
        )

        if decision.tool_name == "retrieve_documents":
            print(f"Retrieved {len(tool_result['results'])} chunks.")
            for i, item in enumerate(tool_result["results"], start=1):
                print(f"\n[{i}] {item.get('source')} {item.get('chunk_id')}")
                print(item["text"][:500])
        else:
            print(tool_result)

        print("\nTool Result")
        print("=" * 80)


if __name__ == "__main__":
    main()
