from dotenv import load_dotenv
load_dotenv()

import json
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import uvicorn

from langfuse.langchain import CallbackHandler
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from langgraph.Graph04_langgraph_memory_rag_demo import build_graph, make_turn_input
from langgraph.Graph06_langgraph_streaming_demo import update_to_ui_event

api = FastAPI(title="RAG Agent API")

class StreamRequest(BaseModel):
    question: str = Field(..., description="用户提问的内容")
    thread_id: str = Field("default-thread", description="会话ID，用于保持对话上下文")

# 初始化 Graph 实例
graph_app = build_graph()

# health
@api.get("/health")
def health():
    return {"status": "ok"}


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
        }
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
            'type': 'final',
            'data': {
                'answer': final_state.get('answer'),
                'answerable': final_state.get('answerable'),
                'citations': final_state.get('citations'),
                'confidence': final_state.get('confidence'),
            }
        }
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run("api.serving_api_demo:api", host="127.0.0.1", port=8000, reload=True)
