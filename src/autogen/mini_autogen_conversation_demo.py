from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class Message(BaseModel):
    sender: str
    recipient: str
    type: Literal["user", "agent", "tool", "final"]
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseMiniAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def respond(self, messages: List[Message]) -> Message:
        raise NotImplementedError


def retrieve_project_documents(query: str) -> str:
    return """
[1] source=proposal.pdf page=3
PMI Agent 项目报价为 8500 美元。付款方式为 50% 预付款，50% 交付后支付。

[2] source=delivery_plan.pdf page=2
PMI Agent 的交付周期为两周。交付内容包括 FastAPI 后端、SSE workflow streaming、前端三栏 UI、RAG 检索和 eval harness。

[3] source=risk_report.pdf page=5
主要风险包括：数据源质量不稳定、预测市场流动性不足、LLM 检索结果可能存在噪声、引用证据需要校验。
"""


class PlannerAgent(BaseMiniAgent):
    def respond(self, messages: List[Message]) -> Message:
        user_message = next(m for m in messages if m.type == "user")

        plan = (
            "Plan:\n"
            "1. Ask ResearchAgent to retrieve evidence about pricing, payment, delivery, and risks.\n"
            "2. Ask AnalystAgent to produce an opportunity memo based on evidence.\n"
            "3. Ask CriticAgent to verify citations and unsupported claims."
        )

        return Message(
            sender=self.name,
            recipient="ResearchAgent",
            type="agent",
            content=plan,
            metadata={
                "next_action": "retrieve_evidence",
                "query": user_message.content,
            },
        )

class ResearchAgent(BaseMiniAgent):
    def respond(self, messages: List[Message]) -> Message:
        planner_message = messages[-1]
        query = planner_message.metadata.get("query", planner_message.content)

        evidence = retrieve_project_documents(query)

        return Message(
            sender=self.name,
            recipient="AnalystAgent",
            type="tool",
            content=evidence,
            metadata={
                "tool_name": "retrieve_project_documents",
                "query": query,
            },
        )

class AnalystAgent(BaseMiniAgent):
    def respond(self, messages: List[Message]) -> Message:
        evidence_message = messages[-1]
        evidence = evidence_message.content

        memo = (
            "Opportunity Memo:\n"
            "- Pricing: PMI Agent 项目报价为 8500 美元。 [1]\n"
            "- Payment: 50% 预付款，50% 交付后支付。 [1]\n"
            "- Delivery: 交付周期为两周。 [2]\n"
            "- Scope: 包括 FastAPI 后端、SSE workflow streaming、前端三栏 UI、RAG 检索和 eval harness。 [2]\n"
            "- Risks: 数据源质量不稳定、预测市场流动性不足、LLM 检索结果可能存在噪声、引用证据需要校验。 [3]\n"
            "- Recommendation: continue, but clarify data source scope and acceptance criteria before starting.\n\n"
            "Evidence used:\n"
            f"{evidence}"
        )

        return Message(
            sender=self.name,
            recipient="CriticAgent",
            type="agent",
            content=memo,
            metadata={
                "recommendation": "continue",
            },
        )


class CriticAgent(BaseMiniAgent):
    def respond(self, messages: List[Message]) -> Message:
        memo_message = messages[-1]
        memo = memo_message.content

        issues = []

        if "[1]" not in memo:
            issues.append("Missing citation [1] for pricing/payment.")
        if "[2]" not in memo:
            issues.append("Missing citation [2] for delivery/scope.")
        if "[3]" not in memo:
            issues.append("Missing citation [3] for risks.")

        if "创始人" in memo:
            issues.append("Unsupported founder claim.")

        if issues:
            content = "Critic verdict: needs_revision\n" + "\n".join(issues)
            verdict = "needs_revision"
        else:
            content = (
                "Critic verdict: pass\n"
                "The memo is grounded in retrieved evidence and preserves citations."
            )
            verdict = "pass"

        return Message(
            sender=self.name,
            recipient="User",
            type="final",
            content=content,
            metadata={
                "verdict": verdict,
                "issues": issues,
            },
        )


def run_conversation(user_question: str) -> List[Message]:
    messages = [
        Message(
            sender="User",
            recipient="PlannerAgent",
            type="user",
            content=user_question,
        )
    ]

    planner = PlannerAgent("PlannerAgent")
    researcher = ResearchAgent("ResearchAgent")
    analyst = AnalystAgent("AnalystAgent")
    critic = CriticAgent("CriticAgent")

    for agent in [planner, researcher, analyst, critic]:
        msg = agent.respond(messages)
        messages.append(msg)

    return messages




def main():
    messages = run_conversation(
        "分析 PMI Agent 的报价、付款、交付周期、范围和风险，判断是否值得推进。"
    )

    print("\nConversation Trace")
    print("=" * 80)

    for i, msg in enumerate(messages, start=1):
        print(f"\n[{i}] {msg.sender} -> {msg.recipient} ({msg.type})")
        print(msg.content)
        print("metadata:", msg.metadata)
        print("-" * 80)


if __name__ == "__main__":
    main()
