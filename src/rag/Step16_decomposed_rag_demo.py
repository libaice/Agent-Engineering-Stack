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


class FinalSynthesis(BaseModel):
    answer: str = Field(description="Final synthesized answer to original question.")
    answered_sub_questions: List[str] = Field(
        description="Sub-question ids that were answered."
    )
    unanswered_sub_questions: List[str] = Field(
        description="Sub-question ids that could not be answered."
    )
    citations: List[str] = Field(
        description="Citation references such as source/page/chunk_id."
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


def build_synthesis_prompt(
    original_question: str,
    sub_results: List[Dict[str, Any]],
) -> str:
    return f"""
你是一个严谨的 RAG 综合回答器。

你会收到原始问题，以及多个子问题的结构化回答。
你的任务是综合这些子答案，回答用户的原始问题。

要求：
1. 只能使用 sub_results 中的信息。
2. 不要补充外部知识。
3. 如果某个子问题无法回答，要在最终答案中说明“该项根据现有资料无法确定”。
4. 保留清晰结构，适合用户阅读。
5. citations 使用 source/page/chunk_id 或 sheet/row 信息。
6. 输出严格 JSON，不要输出 markdown。

原始问题：
{original_question}

子问题结果：
{json.dumps(sub_results, ensure_ascii=False, indent=2)}

输出 JSON 格式：
{{
  "answer": "string",
  "answered_sub_questions": ["sq_1"],
  "unanswered_sub_questions": [],
  "citations": ["source.pdf page 3 chunk_id xxx"]
}}
""".strip()


def synthesize_final_answer(
    original_question: str,
    sub_results: List[Dict[str, Any]],
    model: str = "deepseek-v4-pro",
) -> FinalSynthesis:
    prompt = build_synthesis_prompt(original_question, sub_results)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You synthesize final answers from structured RAG sub-results.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    return FinalSynthesis.model_validate(data)


def main():
    store = HybridSearchStore()
    reranker = Reranker()

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"q", "quit", "exit"}:
            break

        decomposition = decompose_query(question)
        print("\nQuery Decomposition")
        print("=" * 80)
        print(decomposition.model_dump_json(indent=2))

        sub_results = []
        for sub_question in decomposition.sub_questions:
            print(f"\nRunning {sub_question.id}: {sub_question.question}")
            result = answer_sub_question(
                sub_question=sub_question,
                store=store,
                reranker=reranker,
            )
            sub_results.append(result)

            print("\nSub-question Answer")
            print("=" * 80)
            print(json.dumps(result["answer"], ensure_ascii=False, indent=2))

        final = synthesize_final_answer(
            original_question=question,
            sub_results=sub_results,
        )

        print("\nFinal Answer")
        print("=" * 80)
        print(final.model_dump_json(indent=2))

        print("\nUser-facing Answer")
        print("=" * 80)
        print(final.answer)


if __name__ == "__main__":
    main()
