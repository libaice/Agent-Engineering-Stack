from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator


class RAGAgentState(TypedDict):
    # original input
    question: str

    # query understanding
    intent: Optional[str]
    rewritten_query: Optional[str]
    search_queries: List[str]
    needs_context: bool
    missing_context: Optional[str]

    # retrieval
    candidates: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]

    # tool / execution trace
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    steps: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]

    # answer
    answerable: Optional[bool]
    answer: Optional[str]
    citations: List[int]
    confidence: Optional[float]
    missing_information: Optional[str]

    # control
    retry_count: int
    status: str