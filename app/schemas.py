from typing import Any
from pydantic import BaseModel, Field
from app.config import settings


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
    doc_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=settings.doc_id_max_length,
        pattern=r"^[A-Za-z0-9_.-]+$",
        examples=[None]
    )

class SourceOut(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any]
    snippet: str
    citation: str
    text: str | None = None

class ChatRagResponse(BaseModel):
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    retrieved_count: int = 0

class GetDocsResponse(BaseModel):
    doc_id: str
    chunk_count: int

class MessageResponse(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str