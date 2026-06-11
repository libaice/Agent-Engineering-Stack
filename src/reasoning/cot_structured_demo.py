from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field


class StructuredReasoningAnswer(BaseModel):
    reasoning_summary: str = Field(
        description="Short summary of the reasoning approach. Do not include hidden chain-of-thought."
    )
    answerable: bool
    answer: str
    citations: List[int] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)


EVIDENCE = """
[1] source=proposal.pdf page=3
PMI Agent 项目报价为 8500 美元。付款方式为 50% 预付款，50% 交付后支付。

[2] source=delivery_plan.pdf page=2
PMI Agent 的交付周期为两周。交付内容包括 FastAPI 后端、SSE workflow streaming、前端三栏 UI、RAG 检索和 eval harness。

[3] source=risk_report.pdf page=5
主要风险包括：数据源质量不稳定、预测市场流动性不足、LLM 检索结果可能存在噪声、引用证据需要校验。
"""


def run_cot_structured(question: str) -> StructuredReasoningAnswer:
    pass


def main():
    questions = [
        "PMI Agent 的报价和风险是什么？",
        "PMI Agent 的交付周期是什么？",
        "PMI Agent 的创始人是谁？",
    ]

    for q in questions:
        result = run_cot_structured(q)
        print("\nQuestion:", q)
        print(result.model_dump_json(indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
