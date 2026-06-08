import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


class QueryRewriteResult(BaseModel):
    original_question: str = Field(description="The original user question.")
    rewritten_query: str = Field(description="A clearer standalone search query.")
    search_queries: List[str] = Field(
        description="Multiple search queries for retrieval."
    )
    intent: str = Field(
        description="The user's likely intent, e.g. pricing_lookup, risk_lookup, summary, table_calculation."
    )
    needs_context: bool = Field(
        description="Whether the question depends on previous context or selected document scope."
    )
    missing_context: Optional[str] = Field(
        default=None, description="What context is missing if needs_context is true."
    )


def build_query_rewrite_prompt(question: str) -> str:
    return f"""
你是一个 RAG 检索 query 改写器。

你的任务是把用户原始问题改写成更适合文档检索的 query。

要求：
1. 保留用户原始意图，不要改变问题含义。
2. 如果用户问题口语化，要改写成清晰、独立、可检索的问题。
3. 如果问题包含“这个、那个、它、他们、这里”等指代，要判断是否需要上下文。
4. 生成 3 到 5 个 search_queries，用于 BM25 + Vector 检索。
5. search_queries 可以包含同义词、专业术语、中英文表达。
6. 不要回答问题，只做 query rewrite。
7. 必须输出严格 JSON，不要输出 markdown。

用户问题：
{question}

输出 JSON 格式：
{{
  "original_question": "string",
  "rewritten_query": "string",
  "search_queries": ["string"],
  "intent": "string",
  "needs_context": false,
  "missing_context": null
}}
""".strip()


def rewrite_query(
    question: str,
    model: str = "deepseek-v4-pro",
) -> QueryRewriteResult:

    prompt = build_query_rewrite_prompt(question)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You rewrite user questions into retrieval-optimized search queries.",
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
        raise ValueError(f"Invalid JSON from query rewrite: {raw}") from e

    try:
        return QueryRewriteResult.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Query rewrite schema validation failed: {data}") from e



def main():
    while True:
        q = input("\nQuestion: ").strip()

        if q.lower() in {"q", "quit", "exit"}:
            break

        # 1. get rewrite resilts list 
        result = rewrite_query(q)
        print(result.model_dump_json(indent=2))

  



if __name__ == "__main__":
    main()
