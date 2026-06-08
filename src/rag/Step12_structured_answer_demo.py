import json
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from rag.Step03_rag_answer_demo import format_evidence


load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"))