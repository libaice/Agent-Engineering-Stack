from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import json
import os
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


from rag.Step13_query_rewrite_demo import rewrite_query
from rag.Step14_query_rewrite_rag_demo import multi_query_search
from rag.Step08_hybrid_search_demo import HybridSearchStore
from rag.Step09_rerank_demo import Reranker
from rag.Step12_structured_answer_demo import answer_with_structured_output

from langgraph.Graph02_langgraph_conditional_rag_demo import (
    RAGState,
    clarify_node,
    retrieve_node,
    retry_retrieve_node,
    route_after_rewrite,
    route_after_answer,
    initial_state,
)

from memory.memory_store import JsonMemoryStore

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)


class MemoryRAGState(TypedDict):
    # conversation memory 新加的，
    messages: Annotated[List[AnyMessage], add_messages]

    # latest user input
    question: str
    standalone_question: Optional[str]

    # long term memory
    long_term_memories: List[Dict[str, Any]]

    # query understanding
    rewritten_query: Optional[str]
    search_queries: List[str]
    intent: Optional[str]
    needs_context: bool
    missing_context: Optional[str]

    # retrieval
    candidates: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]

    # answer
    answerable: Optional[bool]
    answer: Optional[str]
    citations: List[int]
    confidence: Optional[float]
    missing_information: Optional[str]

    # tracing
    steps: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    # control
    status: str
    retry_count: int


class ContextualizedQuestion(BaseModel):
    standalone_question: str = Field(
        description="A standalone question that resolves pronouns and references using conversation history."
    )
    needs_context: bool = Field(
        description="Whether the question still lacks necessary context."
    )
    missing_context: str | None = Field(
        default=None, description="What context is still missing."
    )


memory_store = JsonMemoryStore()


def load_memory_node(state: MemoryRAGState) -> Dict[str, Any]:
    try:
        query = state["question"]

        memories = memory_store.search_memories(
            query=query,
            top_k=5,
        )

        return {
            "long_term_memories": memories,
            "steps": [
                {
                    "node": "load_memory",
                    "status": "success",
                    "data": {
                        "num_memories": len(memories),
                        "memory_ids": [m["memory_id"] for m in memories],
                        "memory_names": [m["name"] for m in memories],
                    },
                }
            ],
        }

    except Exception as e:
        return {
            "errors": [f"load_memory_node failed: {e}"],
            "steps": [
                {
                    "node": "load_memory",
                    "status": "error",
                    "error": str(e),
                }
            ],
        }


def messages_to_text(messages) -> str:
    lines = []
    for msg in messages[-8:]:
        role = msg.type
        content = msg.content
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def contextualize_question_node(state: MemoryRAGState) -> Dict[str, Any]:
    try:
        history_text = messages_to_text(state["messages"])
        question = state["question"]
        prompt = f"""
你是一个多轮对话中的问题改写器。

你的任务是根据 conversation history，把当前用户问题改写成一个独立、清晰、可检索的问题。

要求：
1. 如果当前问题包含“这个、那个、它、那、上面、刚才”等指代，请结合历史补全。
2. 不要回答问题，只改写问题。
3. 如果历史仍不足以确定指代，needs_context=true。
4. 输出严格 JSON。

Conversation history:
{history_text}

Current user question:
{question}

输出 JSON：
{{
  "standalone_question": "string",
  "needs_context": false,
  "missing_context": null
}}
""".strip()
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite multi-turn user questions into standalone retrieval questions.",
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
        result = ContextualizedQuestion.model_validate(data)

        return {
            "standalone_question": result.standalone_question,
            "needs_context": result.needs_context,
            "missing_context": result.missing_context,
            "status": "contextualized",
            "steps": [
                {
                    "node": "contextualize_question",
                    "status": "success",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "data": result.model_dump(),
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"contextualize_question_node failed: {e}"],
            "steps": [
                {
                    "node": "contextualize_question",
                    "status": "error",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "error": str(e),
                }
            ],
        }


def rewrite_node(state: MemoryRAGState) -> Dict[str, Any]:
    try:
        query = state.get("standalone_question") or state["question"]
        rewrite = rewrite_query(query)

        return {
            "rewritten_query": rewrite.rewritten_query,
            "search_queries": rewrite.search_queries,
            "intent": rewrite.intent,
            "needs_context": state.get("needs_context") or rewrite.needs_context,
            "missing_context": state.get("missing_context") or rewrite.missing_context,
            "status": "rewritten",
            "steps": [
                {
                    "node": "rewrite",
                    "status": "success",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "data": rewrite.model_dump(),
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"rewrite_node failed: {e}"],
            "steps": [
                {
                    "node": "rewrite",
                    "status": "error",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "error": str(e),
                }
            ],
        }


def answer_node(state: MemoryRAGState) -> Dict[str, Any]:
    try:
        if not state["evidence"]:
            answer = "根据现有资料无法确定。"

            return {
                "answerable": False,
                "answer": answer,
                "citations": [],
                "confidence": 0.0,
                "missing_information": "没有检索到可用证据。",
                "messages": [AIMessage(content=answer)],
                "status": "answered",
                "steps": [
                    {
                        "node": "answer",
                        "status": "no_evidence",
                        "time": datetime.utcnow().isoformat() + "Z",
                    }
                ],
            }

        result = answer_with_structured_output(
            question=state["standalone_question"] or state["question"],
            retrieved_chunks=state["evidence"],
        )

        return {
            "answerable": result.answerable,
            "answer": result.answer,
            "citations": result.citations,
            "confidence": result.confidence,
            "missing_information": result.missing_information,
            "messages": [AIMessage(content=result.answer)],
            "status": "answered",
            "steps": [
                {
                    "node": "answer",
                    "status": "success",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "data": result.model_dump(),
                }
            ],
        }

    except Exception as e:
        return {
            "status": "failed",
            "errors": [f"answer_node failed: {e}"],
            "steps": [
                {
                    "node": "answer",
                    "status": "error",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "error": str(e),
                }
            ],
        }


def make_turn_input(question: str) -> Dict[str, Any]:
    return {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "standalone_question": None,
        "rewritten_query": None,
        "search_queries": [],
        "intent": None,
        "needs_context": False,
        "missing_context": None,
        "candidates": [],
        "evidence": [],
        "answerable": None,
        "answer": None,
        "citations": [],
        "confidence": None,
        "missing_information": None,
        "steps": [],
        "errors": [],
        "status": "started",
        "retry_count": 0,
    }


def build_graph():
    graph = StateGraph(MemoryRAGState)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("contextualize_question", contextualize_question_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("retry_retrieve", retry_retrieve_node)
    graph.add_node("answer", answer_node)

    # 1. Start from here , Get the content from the memory.
    graph.add_edge(START, "contextualize_question")

    # 2. After enough content, then rewrite the user's prompt.
    graph.add_edge("contextualize_question", "rewrite")

    # 3. After rewrite and syntax check, use the prompt to clarify/ retrieve
    graph.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "clarify": "clarify",
            "retrieve": "retrieve",
            "end": END,
        },
    )

    # 4. If context is not enough, go to clarify node to ask user.
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "answer")

    graph.add_conditional_edges(
        "answer",
        route_after_answer,
        {
            "retry": "retry_retrieve",
            "end": END,
        },
    )

    graph.add_edge("retry_retrieve", "answer")

    return graph.compile(checkpointer=InMemorySaver())


def main():
    app = build_graph()
    thread_id = input("Thread ID: ").strip() or "memory-demo-thread"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\nUser: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break

        result = app.invoke(
            make_turn_input(question),
            config=config,
        )

        print("\nAssistant:")
        print(result.get("answer"))

        print("\nStandalone Question:")
        print(result.get("standalone_question"))

        print("\nMessages:")
        for msg in result.get("messages", [])[-6:]:
            print(msg.type, ":", msg.content)


if __name__ == "__main__":
    main()
