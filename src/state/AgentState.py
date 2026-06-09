from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime

from tool.Tool01_tools_demo import ToolContext

from tool.Tool02_tool_router_demo import decide_tool


class AgentState(TypedDict):
    question: str
    steps: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    answer: Optional[Dict[str, Any]]
    errors: List[str]



def main():
    print("hello,State ")



if __name__ == "__main__":
    main()