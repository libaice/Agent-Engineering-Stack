from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_chunk_demo import chunk_pdf


class SimpleVectorStore:
    def _init__(self, embedding_model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents: List[Dict[str:Any]] = []


    



if __name__ == "__main__":
    file_path = "data/project.pdf"
    chunks = chunk_pdf(file_path)

    store = SimpleVectorStore()
