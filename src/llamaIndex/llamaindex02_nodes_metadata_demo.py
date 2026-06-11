from llama_index.core.schema import TextNode
from pathlib import Path
from typing import List
import os

from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.llms import LLMMetadata, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


load_dotenv()


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
    docs = [
        Document(
            text=(
                "项目名称：PMI Agent。\n"
                "项目报价为 8500 美元。\n"
                "交付周期为两周。\n"
                "付款方式为 50% 预付款，50% 交付后支付。"
            ),
            metadata={
                "document_id": "doc_pmi_proposal",
                "source": "proposal.pdf",
                "page": 3,
                "file_type": "pdf",
                "project": "PMI Agent",
                "acl_org_id": "org_orderbook",
                "allowed_user_ids": ["user_bruce"],
            },
        ),
        Document(
            text=(
                "主要风险包括：数据源质量不稳定、预测市场流动性不足、"
                "LLM 检索结果可能存在噪声、引用证据需要校验。"
            ),
            metadata={
                "document_id": "doc_pmi_risk",
                "source": "risk_report.pdf",
                "page": 5,
                "file_type": "pdf",
                "project": "PMI Agent",
                "acl_org_id": "org_orderbook",
                "allowed_user_ids": ["user_bruce"],
            },
        ),
    ]
    return docs


def build_nodes(documents: List[Document]):
    splitter = SentenceSplitter(
        chunk_size=120,
        chunk_overlap=20,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    for i, node in enumerate(nodes):
        node.metadata["chunk_id"] = f"{node.metadata.get('document_id')}:chunk_{i}"

    return nodes


def build_excel_nodes():
    rows = [
        {
            "project": "PMI Agent",
            "item": "Backend Development",
            "amount_usd": 3500,
            "sheet_name": "Budget",
            "row_index": 2,
        },
        {
            "project": "PMI Agent",
            "item": "Frontend Development",
            "amount_usd": 3000,
            "sheet_name": "Budget",
            "row_index": 3,
        },
        {
            "project": "PMI Agent",
            "item": "Evaluation Harness",
            "amount_usd": 2000,
            "sheet_name": "Budget",
            "row_index": 4,
        },
    ]

    nodes = []

    for i, row in enumerate(rows):
        text = (
            f"Project: {row['project']}\n"
            f"Item: {row['item']}\n"
            f"Amount USD: {row['amount_usd']}"
        )

        node = TextNode(
            text=text,
            metadata={
                "document_id": "doc_budget_excel",
                "source": "budget.xlsx",
                "file_type": "excel",
                "sheet_name": row["sheet_name"],
                "row_index": row["row_index"],
                "project": row["project"],
                "chunk_id": f"doc_budget_excel:row_{row['row_index']}",
            },
        )

        nodes.append(node)

    return nodes


    


def main():
    Settings.llm = CustomOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com",
        temperature=0,
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    # 1. pdf -> documents
    documents = build_documents()

    # 2. documents -> nodes
    pdf_nodes = build_nodes(documents)

    excel_nodes = build_excel_nodes()

    all_nodes = pdf_nodes + excel_nodes

    print("\nNodes")
    print("=" * 80)

    for node in all_nodes:
        print("node_id:", node.node_id)
        print("metadata:", node.metadata)
        print("text:", node.get_content())
        print("-" * 80)

    index = VectorStoreIndex(all_nodes)

    query_engine = index.as_query_engine(
        similarity_top_k=3,
    )

    response = query_engine.query("这个项目报价是多少？")

    print("\nAnswer")
    print("=" * 80)
    print(response)

    print("\nSource Nodes")
    print("=" * 80)


    for i, source_node in enumerate(response.source_nodes, start=1):
        # node = source_node.node


        node.metadata["chunk_id"] = f"{node.metadata.get('document_id')}:chunk_{i}"

        node.excluded_embed_metadata_keys = [
            "allowed_user_ids",
            "acl_org_id",
        ]

        node.excluded_llm_metadata_keys = [
            "allowed_user_ids",
            "acl_org_id",
        ]

        print(f"[{i}] score={source_node.score}")
        print("metadata:", node.metadata)
        print("text:", node.get_content())
        print("-" * 80)






if __name__ == "__main__":
    main()
