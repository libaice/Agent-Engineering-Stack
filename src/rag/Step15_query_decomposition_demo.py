import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


class SubQuestion(BaseModel):
    id: str = Field(description="Sub-question id, e.g. sq_1.")
    question: str = Field(description="A standalone sub-question.")
    intent: str = Field(description="Intent of the sub-question.")
    priority: int = Field(description="Priority order, starting from 1.")


class QueryDecompositionResult(BaseModel):
    original_question: str
    is_complex: bool
    reason: str
    sub_questions: List[SubQuestion]


def build_decomposition_prompt(question: str) -> str:
    return f"""
你是一个 RAG / Agent 系统里的 query decomposition 模块。

你的任务是判断用户问题是否需要拆解。
如果问题包含多个信息需求、多个实体、多个维度，应该拆成多个独立子问题。
如果问题很简单，就保留为一个子问题。

要求：
1. 不要回答问题，只做拆解。
2. 每个 sub-question 必须是独立、清晰、可检索的问题。
3. 子问题不能改变用户原始意图。
4. 子问题数量控制在 1 到 6 个。
5. 如果用户问题涉及“报价、风险、周期、交付、负责人、状态、下一步”等多个维度，要拆开。
6. 必须输出严格 JSON，不要输出 markdown。

用户问题：
{question}

输出 JSON 格式：
{{
  "original_question": "string",
  "is_complex": true,
  "reason": "为什么需要或不需要拆解",
  "sub_questions": [
    {{
      "id": "sq_1",
      "question": "string",
      "intent": "string",
      "priority": 1
    }}
  ]
}}
""".strip()


def decompose_query(
    question: str,
    model: str = "deepseek-v4-pro",
) -> QueryDecompositionResult:
    prompt = build_decomposition_prompt(question)
    print(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You decompose complex user questions into retrieval-friendly sub-questions.",
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
        raise ValueError(f"Invalid JSON from decomposition: {raw}") from e

    try:
        return QueryDecompositionResult.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Decomposition schema validation failed: {data}") from e


if __name__ == "__main__":
    while True:
        q = input("\nQuestion: ").strip()

        if q.lower() in {"q", "quit", "exit"}:
            break

        result = decompose_query(q)
        print(result.model_dump_json(indent=2))
