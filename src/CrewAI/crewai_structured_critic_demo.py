import os
from dotenv import load_dotenv
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.llms.providers.openai.completion import OpenAICompletion

# Monkey-patch supports_function_calling to return False for DeepSeek models
# so CrewAI's output converter falls back to JSON/Pydantic text parser
# instead of using OpenAI's beta.chat.completions.parse which fails on DeepSeek.
_original_supports_function_calling = OpenAICompletion.supports_function_calling
def _custom_supports_function_calling(self) -> bool:
    if "deepseek" in getattr(self, "model", "").lower():
        return False
    return _original_supports_function_calling(self)
OpenAICompletion.supports_function_calling = _custom_supports_function_calling

load_dotenv()


class EvidenceItem(BaseModel):
    citation_id: int = Field(description="Evidence id, e.g. 1, 2, 3.")
    source: str = Field(description="Source file name.")
    location: Optional[str] = Field(
        default=None, description="Page, sheet/row, or other location."
    )
    claim: str = Field(description="A factual claim supported by this evidence.")


class EvidenceSummary(BaseModel):
    project: str
    evidence: List[EvidenceItem]
    missing_information: List[str] = Field(default_factory=list)


class CitedClaim(BaseModel):
    claim: str
    citations: List[int]
    confidence: float = Field(ge=0, le=1)


class RiskItem(BaseModel):
    risk: str
    severity: Literal["low", "medium", "high"]
    citations: List[int]
    mitigation: Optional[str] = None


class OpportunityMemo(BaseModel):
    project: str
    summary: str
    key_facts: List[CitedClaim]
    risks: List[RiskItem]
    recommendation: Literal["continue", "pause", "reject", "clarify"]
    reasoning: str
    next_actions: List[str]


class UnsupportedClaim(BaseModel):
    claim: str
    reason: str
    suggested_fix: Optional[str] = None


class CriticReview(BaseModel):
    verdict: Literal["pass", "needs_revision", "fail"]
    unsupported_claims: List[UnsupportedClaim] = Field(default_factory=list)
    missing_citations: List[str] = Field(default_factory=list)
    overconfident_statements: List[str] = Field(default_factory=list)
    required_clarifications: List[str] = Field(default_factory=list)
    final_notes: str


@tool("retrieve_project_documents")
def retrieve_project_documents(query: str) -> str:
    """
    Retrieve grounded project evidence from the PMI Agent document index.
    Use this tool for questions about pricing, payment terms, delivery timeline,
    project scope, implementation details, risks, and evidence.
    """
    return """
[1] source=proposal.pdf page=3 chunk_id=doc_proposal:chunk_0
PMI Agent 项目报价为 8500 美元。付款方式为 50% 预付款，50% 交付后支付。商业范围包括 RAG 后端、workflow UI 和 eval harness。

[2] source=delivery_plan.pdf page=2 chunk_id=doc_delivery:chunk_0
PMI Agent 的交付周期为两周。交付内容包括 FastAPI 后端、SSE workflow streaming、前端三栏 UI、RAG 检索、LangGraph workflow 和 eval harness。

[3] source=risk_report.pdf page=5 chunk_id=doc_risk:chunk_0
PMI Agent 的主要风险包括：数据源质量不稳定、预测市场流动性不足、LLM 检索结果可能存在噪声、引用证据需要校验、RAG answerability 需要严格控制。
"""


def build_agents():
    llm = LLM(
        model="deepseek-v4-pro",
        provider="openai",
        base_url="https://api.deepseek.com",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    researcher = Agent(
        role="Project Researcher",
        goal=(
            "Collect grounded evidence about the PMI Agent project. "
            "You must use retrieve_project_documents for factual claims."
        ),
        backstory=(
            "You are a careful project researcher. "
            "You only report facts supported by retrieved evidence. "
            "You preserve citation ids exactly as [1], [2], [3]."
        ),
        tools=[retrieve_project_documents],
        verbose=True,
        llm=llm,
    )

    analyst = Agent(
        role="Business Analyst",
        goal=(
            "Create a structured opportunity memo based only on the research evidence."
        ),
        backstory=(
            "You are a pragmatic FDE-style analyst. "
            "You care about customer value, revenue, delivery risk, and concrete next actions. "
            "You must cite every factual claim."
        ),
        verbose=True,
        llm=llm,
    )

    critic = Agent(
        role="Evidence Critic",
        goal=(
            "Review the opportunity memo against the evidence summary. "
            "Detect unsupported claims, missing citations, and overconfident conclusions."
        ),
        backstory=(
            "You are strict about evidence quality. "
            "You prefer 'needs_revision' over passing weak work. "
            "You do not allow unsupported recommendations."
        ),
        verbose=True,
        llm=llm,
    )

    return researcher, analyst, critic


def build_tasks(researcher, analyst, critic):
    research_task = Task(
        description=(
            "Research the PMI Agent project using retrieve_project_documents.\n\n"
            "Find evidence about:\n"
            "1. Pricing\n"
            "2. Payment terms\n"
            "3. Delivery timeline\n"
            "4. Scope / deliverables\n"
            "5. Main risks\n\n"
            "Return structured evidence items. Each EvidenceItem must include:\n"
            "- citation_id matching the retrieved evidence id\n"
            "- source\n"
            "- location\n"
            "- claim supported by that evidence\n\n"
            "Do not include unsupported claims."
        ),
        expected_output="A structured EvidenceSummary object.",
        agent=researcher,
        output_pydantic=EvidenceSummary,
    )

    analysis_task = Task(
        description=(
            "Based only on the EvidenceSummary, write a structured OpportunityMemo.\n\n"
            "Rules:\n"
            "1. Every key fact must include citations.\n"
            "2. Every risk must include citations.\n"
            "3. Do not introduce facts that are not in the EvidenceSummary.\n"
            "4. If important information is missing, set recommendation='clarify'.\n"
            "5. Recommendation must be one of: continue, pause, reject, clarify.\n"
        ),
        expected_output="A structured OpportunityMemo object with cited claims and risks.",
        agent=analyst,
        context=[research_task],
        output_pydantic=OpportunityMemo,
    )

    critique_task = Task(
        description=(
            "Review the OpportunityMemo against the EvidenceSummary.\n\n"
            "Check:\n"
            "1. Are all important factual claims supported by citations?\n"
            "2. Are citation ids valid?\n"
            "3. Are there unsupported claims?\n"
            "4. Are there missing citations?\n"
            "5. Is the recommendation too confident given the evidence?\n\n"
            "Return a CriticReview.\n"
            "Use verdict='pass' only if the memo is well-grounded.\n"
            "Use verdict='needs_revision' if there are fixable issues.\n"
            "Use verdict='fail' if the memo contains serious unsupported claims."
        ),
        expected_output="A structured CriticReview object.",
        agent=critic,
        context=[research_task, analysis_task],
        output_pydantic=CriticReview,
    )

    return research_task, analysis_task, critique_task


def build_crew():
    researcher, analyst, critic = build_agents()
    research_task, analysis_task, critique_task = build_tasks(
        researcher,
        analyst,
        critic,
    )

    crew = Crew(
        agents=[researcher, analyst, critic],
        tasks=[research_task, analysis_task, critique_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def evaluate_critic_result(
    evidence_summary: EvidenceSummary,
    memo: OpportunityMemo,
    review: CriticReview,
) -> dict:
    valid_citation_ids = {item.citation_id for item in evidence_summary.evidence}

    errors = []

    for fact in memo.key_facts:
        if not fact.citations:
            errors.append(f"Key fact missing citations: {fact.claim}")

        for cid in fact.citations:
            if cid not in valid_citation_ids:
                errors.append(f"Invalid citation {cid} in fact: {fact.claim}")

    for risk in memo.risks:
        if not risk.citations:
            errors.append(f"Risk missing citations: {risk.risk}")

        for cid in risk.citations:
            if cid not in valid_citation_ids:
                errors.append(f"Invalid citation {cid} in risk: {risk.risk}")

    if memo.recommendation == "continue":
        has_risk = len(memo.risks) > 0
        if not has_risk:
            errors.append("Recommendation is continue but risks are empty.")

    critic_passed = review.verdict == "pass"
    deterministic_passed = len(errors) == 0

    return {
        "deterministic_passed": deterministic_passed,
        "critic_verdict": review.verdict,
        "critic_passed": critic_passed,
        "errors": errors,
        "unsupported_claims": [
            claim.model_dump() for claim in review.unsupported_claims
        ],
        "missing_citations": review.missing_citations,
        "overconfident_statements": review.overconfident_statements,
        "required_clarifications": review.required_clarifications,
    }


def main():
    crew = build_crew()

    result = crew.kickoff(
        inputs={
            "project": "PMI Agent",
            "question": (
                "分析 PMI Agent 这个项目的报价、付款、交付周期、范围和风险，"
                "判断是否值得继续推进。"
            ),
        }
    )

    print("\nCrew Result")
    print("=" * 80)
    print(result)

    print("\nRaw")
    print("=" * 80)
    print(result.raw)

    print("\nPydantic")
    print("=" * 80)
    print(result.pydantic)

    print("\nTasks Output")
    print("=" * 80)
    for i, task_output in enumerate(result.tasks_output, start=1):
        print(f"\nTask {i}")
        print("raw:", task_output.raw)
        print("pydantic:", task_output.pydantic)
        print("-" * 80)

    tasks = result.tasks_output
    evidence_summary = tasks[0].pydantic
    memo = tasks[1].pydantic
    review = tasks[2].pydantic

    eval_result = evaluate_critic_result(
        evidence_summary=evidence_summary,
        memo=memo,
        review=review,
    )

    print("\nCritic Eval")
    print("=" * 80)
    print(eval_result)



def run_crewai_pipeline(question: str, project: str = "PMI Agent") -> dict:
    crew = build_crew()

    result = crew.kickoff(
        inputs={
            "project": project,
            "question": question,
        }
    )

    tasks = result.tasks_output

    evidence_summary = tasks[0].pydantic
    memo = tasks[1].pydantic
    review = tasks[2].pydantic

    eval_result = evaluate_critic_result(
        evidence_summary=evidence_summary,
        memo=memo,
        review=review,
    )

    return {
        "question": question,
        "project": project,
        "crew_result": result,
        "evidence_summary": evidence_summary,
        "memo": memo,
        "review": review,
        "eval_result": eval_result,
    }


if __name__ == "__main__":
    main()
