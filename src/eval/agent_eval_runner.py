import json
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.Graph04_langgraph_memory_rag_demo import build_graph, make_turn_input


EVAL_CASES_PATH = Path("evals/agent_cases.yaml")
EVAL_RESULTS_PATH = Path("evals/agent_eval_results.jsonl")

def main():
    print("hello,world")

if __name__ == "__main__":
    main()