import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from Step03_rag_answer_demo import format_evidence


load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


# refer openai guardrail https://developers.openai.com/api/docs/guides/agents/guardrails-approvals#choose-the-right-control
def build_answerability_prompt(question: str, evidence: str) -> str:
    return f"""
你是一个严格的 RAG 证据审查器。

你的任务不是回答问题，而是判断给定 Evidence 是否足够支持回答 Question。

判断标准：
1. 如果 Evidence 中有直接信息可以回答问题，answerable = true。
2. 如果 Evidence 只是主题相关，但没有直接答案，answerable = false。
3. 如果 Evidence 需要猜测、常识补充、外部知识，answerable = false。
4. 如果 Evidence 互相冲突，但可以指出冲突，answerable = true，并说明冲突。
5. supporting_evidence_ids 只能使用 Evidence 中出现的编号，例如 [1], [2]。
6. 必须输出严格 JSON，不要输出 markdown。

Evidence:
{evidence}

Question:
{question}

请输出 JSON，格式如下：
{{
  "answerable": true,
  "reason": "为什么证据足够或不足",
  "supporting_evidence_ids": [1, 2],
  "missing_information": "如果不能回答，缺少什么信息"
}}
""".strip()


def check_answerability(
    question: str, retrieved_chunks: List[Dict[str, Any]], model: str = "deepseek-v4-pro"
) -> Dict[str, Any]:
    evidence = format_evidence(retrieved_chunks)
    prompt = build_answerability_prompt(question, evidence)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evidence sufficiency checker for RAG systems."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return {
            "answerable": False,
            "reason": "Answerability checker returned invalid JSON.",
            "supporting_evidence_ids": [],
            "missing_information": "无法解析证据审查结果。"
        }

    return result



