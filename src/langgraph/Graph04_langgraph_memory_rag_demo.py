import json
import os
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


class MemoryRAGState(TypedDict):
    # conversation memory 新加的，
    messages: Annotated[List[AnyMessage], add_messages]

    # latest user input
    question: str
    standalone_question: Optional[str]

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
        default=None,
        description="What context is still missing."
    )
