import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    Document,
)
from llama_index.core.readers.base import BaseReader
from llama_index.core.llms import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
DATA_DIR = Path("data")


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


class PyMuPDFReader(BaseReader):
    def load_data(self, file_path: Path, extra_info: dict = None) -> list[Document]:
        import fitz
        doc = fitz.open(str(file_path))
        documents = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            metadata = extra_info or {}
            metadata["page"] = page_num + 1
            documents.append(Document(text=text, metadata=metadata))
        return documents


class PandasExcelReader(BaseReader):
    def load_data(self, file_path: Path, extra_info: dict = None) -> list[Document]:
        import pandas as pd
        xls = pd.ExcelFile(str(file_path))
        documents = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            text = df.to_string(index=False)
            metadata = extra_info or {}
            metadata["sheet_name"] = sheet_name
            documents.append(Document(text=text, metadata=metadata))
        return documents


def build_query_engine():
    Settings.llm = CustomOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com",
        temperature=0,
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        recursive=True,
        file_extractor={
            ".pdf": PyMuPDFReader(),
            ".xlsx": PandasExcelReader(),
        }
    ).load_data()

    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine(
        similarity_top_k=5,
    )

    return query_engine


def main():
    query_engine = build_query_engine()
    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"q", "quit", "exit"}:
            break

        response = query_engine.query(question)

        print("\nAnswer")
        print("=" * 80)
        print(response)

        print("\nSource Nodes")
        print("=" * 80)

        for i, node in enumerate(response.source_nodes, start=1):
            print(f"[{i}] score={node.score}")
            print(node.node.metadata)
            print(node.node.get_content()[:500])
            print("-" * 80)


if __name__ == "__main__":
    main()
