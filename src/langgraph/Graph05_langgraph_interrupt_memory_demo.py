import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Annotated, Literal
import operator

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.wrappers import wrap_openai

from memory.memory_store import JsonMemoryStore
from langgraph.Graph04_langgraph_memory_rag_demo import (
    MemoryRAGState,
    load_memory_node,
    contextualize_question_node,
    rewrite_node,
    retrieve_node,
    answer_node,
)

load_dotenv()
client = wrap_openai(
    OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
)

memory_store = JsonMemoryStore()


class HITLMemoryRAGState(TypedDict):
    # conversation memory
    messages: Annotated[List[AnyMessage], operator.add]

    # latest user input
    question: str
    standalone_question: Optional[str]

    # long term memory
    long_term_memories: List[Dict[str, Any]]
    memory_candidate: Optional[Dict[str, Any]]
    memory_approval: Optional[Dict[str, Any]]

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


def extract_memory_node(state: HITLMemoryRAGState) -> Dict[str, Any]:
    try:
        messages = state["messages"]
        if len(messages) < 2:
            return {"memory_candidate": {"should_save": False}}

        user_msg = messages[-2].content
        ai_msg = messages[-1].content

        prompt = f"""
你是一个长期记忆提取器。

分析下面的对话，判断用户是否透露了需要被存入长期数据库的个人信息、背景、技术栈或偏好。
只提取具有长期价值的信息，过滤掉随意的闲聊。例如：如果用户说“我正在用 LangGraph 做开发”，这说明用户的技术栈有 LangGraph，应该提取。

用户问题：
{user_msg}

助手回答：
{ai_msg}

输出 JSON：
{{
  "should_save": true,
  "type": "project" / "preference" / "goal",
  "name": "例如：LangGraph 技术栈",
  "content": "用户正在使用 LangGraph 进行开发。",
  "aliases": ["LangGraph"]
}}
如果不需要记录，输出：
{{
  "should_save": false,
  "type": null,
  "name": null,
  "content": null,
  "aliases": []
}}
""".strip()

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)
        return {
            "memory_candidate": data,
            "steps": [
                {
                    "node": "extract_memory",
                    "status": "success",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "data": data,
                }
            ],
        }
    except Exception as e:
        return {
            "errors": [f"extract_memory_node failed: {e}"],
            "steps": [
                {
                    "node": "extract_memory",
                    "status": "error",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "error": str(e),
                }
            ],
        }


def human_review_memory_node(state: HITLMemoryRAGState) -> Dict[str, Any]:
    candidate = state.get("memory_candidate")

    if not candidate or not candidate.get("should_save"):
        return {
            "memory_approval": {
                "action": "skip",
                "reason": "No memory candidate to review.",
            },
            "steps": [
                {
                    "node": "human_review_memory",
                    "status": "skipped",
                    "data": "No memory candidate.",
                }
            ],
        }

    human_decision = interrupt(
        {
            "type": "memory_review",
            "message": "是否保存这条长期记忆？",
            "candidate": candidate,
            "allowed_actions": ["approve", "reject", "edit"],
        }
    )

    return {
        "memory_approval": human_decision,
        "steps": [
            {
                "node": "human_review_memory",
                "status": "reviewed",
                "data": human_decision,
            }
        ],
    }


def save_memory_node(state: HITLMemoryRAGState) -> Dict[str, Any]:
    try:
        approval = state.get("memory_approval") or {}
        action = approval.get("action")
        candidate = state.get("memory_candidate")

        if action == "approve" and candidate and candidate.get("should_save"):
            memory_store.add_memory(
                memory_type=candidate["type"],
                name=candidate["name"],
                content=candidate["content"],
                aliases=candidate.get("aliases"),
                importance=0.8,
            )
            return {
                "steps": [
                    {
                        "node": "save_memory",
                        "status": "success",
                        "time": datetime.utcnow().isoformat() + "Z",
                        "data": f"Saved memory: {candidate['name']}",
                    }
                ]
            }

        return {
            "steps": [
                {
                    "node": "save_memory",
                    "status": "skipped",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "data": f"Action is {action}.",
                }
            ]
        }
    except Exception as e:
        return {
            "errors": [f"save_memory_node failed: {e}"],
            "steps": [
                {
                    "node": "save_memory",
                    "status": "error",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "error": str(e),
                }
            ],
        }


def route_after_memory_review(
    state: HITLMemoryRAGState,
) -> Literal["save_memory", "end"]:
    approval = state.get("memory_approval") or {}
    action = approval.get("action")

    if action == "approve":
        return "save_memory"

    return "end"


def build_graph():
    graph = StateGraph(HITLMemoryRAGState)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("contextualize_question", contextualize_question_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("extract_memory", extract_memory_node)
    graph.add_node("human_review_memory", human_review_memory_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "contextualize_question")
    graph.add_edge("contextualize_question", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "extract_memory")
    graph.add_edge("extract_memory", "human_review_memory")

    graph.add_conditional_edges(
        "human_review_memory",
        route_after_memory_review,
        {
            "save_memory": "save_memory",
            "end": END,
        },
    )

    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=InMemorySaver())


def main():
    app = build_graph()

    thread_id = input("Thread ID: ").strip() or "hitl-memory-thread"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        question = input("\nUser: ").strip()

        if question.lower() in {"q", "quit", "exit"}:
            break

        result = app.invoke(
            {
                "question": question,
                "messages": [HumanMessage(content=question)],
            },
            config=config,
        )

        if "__interrupt__" in result:
            print("\nGraph interrupted (Reviewing memory candidate)...")
            print("=" * 80)
            candidate = result.get("memory_candidate")
            print(f"Candidate to save:\n- Name: {candidate.get('name')}\n- Type: {candidate.get('type')}\n- Content: {candidate.get('content')}")

            action = input("\nApprove memory? [approve/reject]: ").strip().lower()

            if action == "approve":
                resume_value = {"action": "approve"}
            else:
                resume_value = {
                    "action": "reject",
                    "reason": "User rejected memory save.",
                }

            result = app.invoke(
                Command(resume=resume_value),
                config=config,
            )

        print("\nAssistant:")
        print(result.get("answer"))

        print("\nSteps:")
        for step in result.get("steps", [])[-8:]:
            print(step)


if __name__ == "__main__":
    main()
