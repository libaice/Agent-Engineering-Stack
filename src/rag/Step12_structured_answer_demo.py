import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from rag.Step03_rag_answer_demo import format_evidence

from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker, print_reranked_results

load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com")


class RAGAnswer(BaseModel):
    answerable: bool = Field(
        description="Whether the provided evidence is sufficient to answer the question."
    )
    answer: str = Field(
        description="Final answer to the user. If not answerable, use the standard unknown phrase."
    )
    citations: List[int] = Field(
        description="List of evidence IDs that support the answer, e.g. [1, 2]."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0 to 1 based only on the provided evidence.",
    )
    missing_information: Optional[str] = Field(
        default=None,
        description="What information is missing if the question cannot be answered.",
    )


UNKNOWN_PHRASE = "根据现有资料无法确定。"


def build_structured_answer_prompt(question: str, evidence: str) -> str:
    return f"""
你是一个严谨的 RAG 文档问答助手。

你只能根据 Evidence 回答问题。

强制规则：
1. 不允许使用 Evidence 之外的知识。
2. 如果 Evidence 不足以回答问题，answerable 必须为 false。
3. 如果 answerable=false，answer 必须是：“{UNKNOWN_PHRASE}”
4. 如果 answerable=true，answer 必须简洁直接，并且每个事实性结论都必须被 citations 支持。
5. citations 只能使用 Evidence 中出现的编号，例如 1、2、3。
6. confidence 必须基于证据充分程度，而不是主观猜测。
7. 必须输出严格 JSON，不要输出 markdown，不要输出额外解释。

输出 JSON 格式：
{{
  "answerable": true,
  "answer": "string",
  "citations": [1],
  "confidence": 0.0,
  "missing_information": null
}}

Evidence:
{evidence}

Question:
{question}
""".strip()


def validate_citations(
    citations: List[int],
    num_evidence: int,
) -> bool:
    return all(1 <= cid <= num_evidence for cid in citations)


def answer_with_structured_output(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = "deepseek-v4-pro",
) -> RAGAnswer:
    evidence = format_evidence(retrieved_chunks)
    prompt = build_structured_answer_prompt(question, evidence)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict RAG assistant that returns valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {raw}") from e

    try:
        parsed = RAGAnswer.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"LLM output failed schema validation: {data}") from e

    if not validate_citations(parsed.citations, len(retrieved_chunks)):
        raise ValueError(
            f"Invalid citations: {parsed.citations}. "
            f"Only 1 to {len(retrieved_chunks)} are allowed."
        )

    if not parsed.answerable:
        if parsed.answer.strip() != UNKNOWN_PHRASE:
            raise ValueError(f"When answerable=false, answer must be: {UNKNOWN_PHRASE}")

        if parsed.citations:
            raise ValueError("When answerable=false, citations must be empty.")

    if parsed.answerable:
        if not parsed.citations:
            raise ValueError("When answerable=true, citations cannot be empty.")

    return parsed


def main():
    store = HybridSearchStore()
    reranker = Reranker()

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        candidates = store.hybrid_search(
            query=question,
            top_k=20,
            candidate_k=20,
            vector_weight=0.5,
            bm25_weight=0.5,
        )

        reranked = reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=5,
        )
        print_reranked_results(reranked)

        structured_answer = answer_with_structured_output(
            question=question,
            retrieved_chunks=reranked,
        )

        print("\nStructured Answer")
        print("=" * 80)
        print(structured_answer.model_dump_json(indent=2))

        print("\nUser-facing Answer")
        print("=" * 80)
        print(structured_answer.answer)

        if structured_answer.citations:
            print(f"\nCitations: {structured_answer.citations}")




if __name__ == "__main__":
    main()
