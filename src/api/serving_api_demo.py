from dotenv import load_dotenv


import json
from uuid import uuid4
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import uvicorn

from langfuse.langchain import CallbackHandler
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from langgraph.Graph04_langgraph_memory_rag_demo import build_graph, make_turn_input
from langgraph.Graph06_langgraph_streaming_demo import update_to_ui_event
from langgraph.prebuilt import InMemoryRunStore


load_dotenv()


api = FastAPI(title="RAG Agent API")

graph_app = build_graph()

run_store = InMemoryRunStore()


class CreateRunRequest(BaseModel):
    thread_id: Optional[str] = None
    question: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ResumeRunRequest(BaseModel):
    action: Literal["approve", "reject", "edit"]
    reason: Optional[str] = None
    edited_memory: Optional[Dict[str, Any]] = None


def make_thread_id() -> str:
    return f"thread_{uuid4().hex[:8]}"


def make_config(thread_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": metadata or {},
        "tags": ["rag-agent-api"],
    }



def summarize_response(run: Dict[str, Any]) -> Dict[str, Any]:
    final_state = run.get("final_state") or {}

    return {
        "run_id": run["run_id"],
        "thread_id": run["thread_id"],
        "status": run["status"],
        "answer": final_state.get("answer"),
        "answerable": final_state.get("answerable"),
        "citations": final_state.get("citations", []),
        "confidence": final_state.get("confidence"),
        "interrupt": run.get("interrupt"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
    }

class StreamRequest(BaseModel):
    question: str = Field(..., description="用户提问的内容")
    thread_id: str = Field("default-thread", description="会话ID，用于保持对话上下文")



# health
@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/runs")
def create_run(req: CreateRunRequest):
    pass


# @api.post("/runs/stream")
# def create_run_stream(req: CreateRunRequest):

@api.post("/runs/stream")
def stream_run(payload: StreamRequest):
    question = payload.question
    thread_id = payload.thread_id

    # 3. 创建 CallbackHandler
    langfuse_handler = CallbackHandler()

    # 4. 配置运行参数，通过 callbacks 传入，使用 run_name 作为 Trace 名称，同时将 thread_id 绑定为 langfuse_session_id
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [langfuse_handler],
        "run_name": "RAG-Agent-Stream-Serving",
        "metadata": {
            "langfuse_session_id": thread_id,  # 关联多轮对话 Session
        },
    }

    def event_generator():
        # 1. 逐步迭代 graph 节点的运行状态并流式输出
        for update in graph_app.stream(
            make_turn_input(question),
            config=config,
            stream_mode="updates",
        ):
            event = update_to_ui_event(update)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # 2. 运行完成后获取最终状态
        final_state = graph_app.get_state(config).values

        final_payload = {
            "type": "final",
            "data": {
                "answer": final_state.get("answer"),
                "answerable": final_state.get("answerable"),
                "citations": final_state.get("citations"),
                "confidence": final_state.get("confidence"),
            },
        }
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@api.get("/runs/{run_id}")
def get_run(run_id: str):
    pass


@api.get("/runs/{run_id}/events")
def get_run_events(run_id: str):
    pass

@api.post("/runs/{run_id}/resume")
def resume_run(run_id: str, req: ResumeRunRequest):
    pass



@api.get("/threads/{thread_id}/state")
def get_thread_state(thread_id: str):
    pass


if __name__ == "__main__":
    uvicorn.run("api.serving_api_demo:api", host="127.0.0.1", port=8000, reload=True)
