import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from pdf_chunk_demo import chunk_pdf
from vector_search_demo import SimpleVectorStore

load_dotenv()


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


# response = client.chat.completions.create(
#     model="deepseek-v4-pro",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello, who are you , and what is your training time ?"},
#     ],
#     stream=False,
#     reasoning_effort="high",
#     extra_body={"thinking": {"type": "enabled"}}
# )


def format_evidence(results: List[Dict[str, Any]]) -> str:
    """
    Convert retrieved chunks into evidence blocks.
    """
    evidence_blocks = []

    for i, item in enumerate(results, start=1):
        block = f"""[{i}]
source: {item["source"]}
page: {item["page"]}
chunk_id: {item["chunk_id"]}
text:
{item["text"]}
"""
        evidence_blocks.append(block)

    return "\n".join(evidence_blocks)


def build_prompt(question: str, evidence: str) -> str:
    return f"""
You are a careful document QA assistant.

You must answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. If the evidence is insufficient, say: "根据现有资料无法确定。"
3. Cite evidence using bracket IDs like [1], [2].
4. Every factual claim should be supported by at least one citation.
5. Keep the answer concise and directly address the question.

Evidence:
{evidence}

Question:
{question}

Answer in Chinese:
""".strip()

def answer_with_llm(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    evidence = format_evidence(retrieved_chunks)
    prompt = build_prompt(question, evidence)

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": "You are a rigorous retrieval-augmented generation assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


def main():
    file_path = "data/project.pdf"
    print("Loading and chuck PDF ... ")
    chunks = chunk_pdf(file_path)
    print(f"Split into {len(chunks)} chunks")

    print("Building vector index...")

    store = SimpleVectorStore()
    store.build(chunks)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        print("\nRetrieving evidence...")
        retrieved = store.search(question, top_k=5)

        print("\nRetrieved evidence:")
        print("=" * 80)

        for i, item in enumerate(retrieved, start=1):
            print(f"[{i}] score={item['score']:.4f}")
            print(
                f"source={item['source']} page={item['page']} chunk_id={item['chunk_id']}"
            )
            print(item["text"][:500])
            print("-" * 80)

        print("\nGenerating answer...")
        answer = answer_with_llm(question, retrieved)

        print("\nAnswer:")
        print("=" * 80)
        print(answer)


if __name__ == "__main__":
    main()
