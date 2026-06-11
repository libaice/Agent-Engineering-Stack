from typing import List, Optional
from dotenv import load_dotenv
import os

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

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
    pass


def build_nodes(documents: List[Document]):
    pass


def make_filters(
    user_id: str,
    project: Optional[str] = None,
    document_id: Optional[str] = None,
    file_type: Optional[str] = None,
) -> MetadataFilter:

    pass


def print_response(response):
    print("\nAnswer")
    print("=" * 80)
    print(response)

    print("\nSource Nodes")
    print("=" * 80)

    for i, source_node in enumerate(response.source_nodes, start=1):
        node = source_node.node

        print(f"[{i}] score={source_node.score}")
        print("metadata:", node.metadata)
        print("text:", node.get_content()[:500])
        print("-" * 80)


def main():
    Settings.llm = CustomOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com",
        temperature=0,
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    documents = build_documents()
    nodes = build_nodes(documents)

    index = VectorStoreIndex(nodes)

    print("\n=== Case 1: No project filter, only user filter ===")

    filters = make_filters(user_id="user_bruce")

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        filters=filters,
    )

    response = query_engine.query("项目报价是多少？")
    print_response(response)

    


if __name__ == "__main__":
    main()
