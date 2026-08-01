from typing import Any, Literal
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

class CompareDocumentRef(BaseModel):
    doc_id: str
    label: str | None = None

class CompareDocumentsAgentRequest(BaseModel):
    question: str
    documents: list[CompareDocumentRef]
    criteria: list[str] | None = None
    top_k_per_document: int = Field(default=5, ge=1, le=settings.max_top_k)
    debug: bool = False


class ComparisonEvidence(BaseModel):
    citation: str
    doc_id: str
    label: str | None = None
    chunk_index: int
    snippet: str
    score: float | None = None


class ComparisonItem(BaseModel):
    criterion: str
    document_a: str
    document_b: str
    winner: Literal["document_a", "document_b", "tie", "unclear"]
    reasoning: str
    evidence: list[ComparisonEvidence] = Field(default_factory=list)

class CompareDocumentsAgentResponse(BaseModel):
    summary: str
    recommendation: str
    winner: Literal["document_a", "document_b", "tie", "unclear"]
    comparison: list[ComparisonItem] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    sources: list[ComparisonEvidence] = Field(default_factory=list)
    debug_info: dict[str, Any] | None = None
