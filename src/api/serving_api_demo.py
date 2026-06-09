from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


api = FastAPI(title="RAG Agent API")

# health
def health():
    return {"status": "ok"}

