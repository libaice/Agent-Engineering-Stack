from typing import List
import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import LLMMetadata, MessageRole
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

_retriever = None


class CustomOpenAI(OpenAI):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


def build_documents() -> List[Document]:
    return [
        Document(
            text=(
                "项目名称：PMI Agent。\n"
                "项目报价为 8500 美元。\n"
                "付款方式为 50% 预付款，50% 交付后支付。\n"
                "商业范围包括 RAG 后端、workflow UI 和 eval harness。"
            ),
            metadata={
                "document_id": "doc_proposal",
                "source": "proposal.pdf",
                "page": 3,
                "project": "PMI Agent",
                "doc_type": "proposal",
            },
        ),
        Document(
            text=(
                "PMI Agent 的交付周期为两周。\n"
                "交付内容包括 FastAPI 后端、SSE workflow streaming、"
                "前端三栏 UI、RAG 检索、LangGraph workflow 和 eval harness。"
            ),
            metadata={
                "document_id": "doc_delivery",
                "source": "delivery_plan.pdf",
                "page": 2,
                "project": "PMI Agent",
                "doc_type": "delivery",
            },
        ),
        Document(
            text=(
                "PMI Agent 的主要风险包括：数据源质量不稳定、"
                "预测市场流动性不足、LLM 检索结果可能存在噪声、"
                "引用证据需要校验、RAG answerability 需要严格控制。"
            ),
            metadata={
                "document_id": "doc_risk",
                "source": "risk_report.pdf",
                "page": 5,
                "project": "PMI Agent",
                "doc_type": "risk",
            },
        ),
    ]


def build_retriever():
    Settings.llm = CustomOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com",
        temperature=0,
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    splitter = SentenceSplitter(
        chunk_size=180,
        chunk_overlap=30,
    )

    documents = build_documents()
    nodes = splitter.get_nodes_from_documents(documents)

    for i, node in enumerate(nodes):
        node.metadata["chunk_id"] = f"{node.metadata.get('document_id')}:chunk_{i}"

    index = VectorStoreIndex(nodes)
    return index.as_retriever(
        similarity_top_k=5,
    )


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


@tool("retrieve_project_documents")
def retrieve_project_documents(query: str) -> str:
    """Retrieve relevant project documents based on a search query."""
    retriever = get_retriever()
    results = retriever.retrieve(query)
    if not results:
        return "No relevant project documents found."

    blocks = []

    for i, node_with_score in enumerate(results, start=1):
        node = node_with_score.node
        meta = node.metadata
        location = ""

        if meta.get("page") is not None:
            location = f"page={meta.get('page')}"
        elif meta.get("sheet_name"):
            location = f"sheet={meta.get('sheet_name')} row={meta.get('row_index')}"

        blocks.append(
            f"[{i}] source={meta.get('source')} {location} "
            f"chunk_id={meta.get('chunk_id')} score={node_with_score.score}\n"
            f"{node.get_content()}"
        )
    return "\n\n".join(blocks)


def build_crew():
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
            "You must use the retrieve_project_documents tool for factual claims."
        ),
        backstory=(
            "You are a careful project researcher. "
            "You do not guess. You only report facts supported by retrieved evidence."
        ),
        tools=[retrieve_project_documents],
        verbose=True,
        llm=llm,
    )

    analyst = Agent(
        role="Business Analyst",
        goal=(
            "Analyze whether the PMI Agent project is worth pursuing based on "
            "pricing, delivery scope, delivery timeline, and risks."
        ),
        backstory=(
            "You are a pragmatic FDE-style analyst. "
            "You care about customer value, revenue, delivery risk, and concrete next actions."
        ),
        verbose=True,
        llm=llm,
    )

    critic = Agent(
        role="Evidence Critic",
        goal=(
            "Review the memo and check whether important claims are grounded in evidence. "
            "Flag unsupported claims, missing citations, and overconfident conclusions."
        ),
        backstory=(
            "You are strict about evidence quality and hallucination control. "
            "You prefer saying 'unknown' over inventing facts."
        ),
        verbose=True,
        llm=llm,
    )

    research_task = Task(
        description=(
            "Research the PMI Agent project using the retrieve_project_documents tool.\n\n"
            "Find evidence about:\n"
            "1. Pricing\n"
            "2. Payment terms\n"
            "3. Delivery timeline\n"
            "4. Scope / deliverables\n"
            "5. Main risks\n\n"
            "Return concise bullet points with citations like [1], [2], [3]. "
            "Do not include claims that are not supported by retrieved evidence."
        ),
        expected_output=(
            "A concise evidence summary with citations for pricing, payment, delivery, scope, and risks."
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "Based on the research summary, write an opportunity analysis memo.\n\n"
            "Analyze:\n"
            "- Revenue potential\n"
            "- Scope clarity\n"
            "- Delivery complexity\n"
            "- Risk level\n"
            "- Whether this project is worth pursuing\n"
            "- Suggested next action\n\n"
            "Do not invent facts beyond the research. Keep citations where relevant."
        ),
        expected_output=(
            "A structured opportunity memo with recommendation, reasoning, risks, and next action."
        ),
        agent=analyst,
        context=[research_task],
    )

    critique_task = Task(
        description=(
            "Review the opportunity memo for evidence quality.\n\n"
            "Check:\n"
            "1. Are pricing, delivery, and risk claims supported by citations?\n"
            "2. Are there unsupported assumptions?\n"
            "3. Is the recommendation too confident?\n"
            "4. What information should be clarified before acting?\n\n"
            "Return the final reviewed memo."
        ),
        expected_output=(
            "A final reviewed memo with evidence critique and a calibrated recommendation."
        ),
        agent=critic,
        context=[research_task, analysis_task],
    )

    crew = Crew(
        agents=[researcher, analyst, critic],
        tasks=[research_task, analysis_task, critique_task],
        process=Process.sequential,
        verbose=True,
    )
    return crew


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

    print("\nFinal Result")
    print("=" * 80)
    print(result)
    pass


if __name__ == "__main__":
    main()
