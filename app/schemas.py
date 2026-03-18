from typing import Any
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    prompt: str

class IngestTextRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class IngestTextResponse(BaseModel):
    doc_id: str
    chunks_added: int
    total_chunks: int


class ChatRagRequest(BaseModel):
    question: str
    top_k: int = 3
    debug: bool = False


class SourceOut(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any]
    snippet: str
    text: str | None = None

class ChatRagResponse(BaseModel):
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    retrieved_count: int = 0