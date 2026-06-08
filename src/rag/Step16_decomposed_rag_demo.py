import json
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rag.Step15_query_decomposition_demo import decompose_query, SubQuestion
from rag.Step13_query_rewrite_demo import rewrite_query
from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output, RAGAnswer


load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


def answer_sub_question(
    sub_question: SubQuestion,
    store: HybridSearchStore,
    reranker: Reranker,
) -> Dict[str, Any]:
    rewrite = rewrite_query(sub_question.question)
    candidates = multi_query_search(
        store=store,
        search_queries=rewrite.search_queries,
        per_query_k=10,
    )

    reranked = reranker.rerank(
        query=rewrite.rewritten_query,
        candidates=candidates,
        top_k=5,
    )

    answer = answer_with_structured_output(
        question=sub_question.question,
        retrieved_chunks=reranked,
    )

    return {
        "sub_question_id": sub_question.id,
        "sub_question": sub_question.question,
        "intent": sub_question.intent,
        "priority": sub_question.priority,
        "rewrite": rewrite.model_dump(),
        "answer": answer.model_dump(),
        "evidence": [
            {
                "source": item.get("source"),
                "page": item.get("page"),
                "sheet_name": item.get("sheet_name"),
                "row_index": item.get("row_index"),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text"),
            }
            for item in reranked
        ],
    }


def main():
    pass


if __name__ == "__main__":
    main()