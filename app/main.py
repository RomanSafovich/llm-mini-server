from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile
)

from app.file_extractor import extract_upload_file
from app.schemas import (
    Prompt,
    IngestTextRequest,
    IngestTextResponse,
    ChatRagRequest,
    ChatRagResponse,
    GetDocsResponse,
    MessageResponse,
    ChatResponse
)

import uvicorn
from app.config import settings
from app.store.milvus_store import store
from app.embeddings.embedder import embedder
from app.llm import generate_text
from app.ingest import run_ingest
from app.rag import run_chat_rag
from app.logger import logger
from app.models import llm_manager
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up services...")
    llm_manager.load_model()
    embedder.load_embedder(model_name=settings.embedder_model_name)
    store.load_milvus_store()
    yield
    logger.info("Shutting down services...")

app = FastAPI(lifespan=lifespan)

@app.get("/documents", response_model=list[GetDocsResponse])
def get_docs():
    return store.list_docs()

@app.delete("/documents/{doc_id}", response_model=MessageResponse)
def delete_document(doc_id: str):
    doc_id = doc_id.strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id must not be blank")

    if not store.delete_doc(doc_id):
        raise HTTPException(404, detail=f"Document {doc_id} not found")

    return {"message": f'document {doc_id} deleted successfully'}

@app.post("/documents/clear", response_model=MessageResponse)
def clear_documents():
    store.clear()
    return {"message": "documents have been cleared successfully"}

@app.post("/ingest_text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest):
    return run_ingest(req, store=store, embedder=embedder)

@app.post("/ingest_file", response_model=IngestTextResponse)
async def ingest_file(doc_id: str = Form(...), file: UploadFile = File(...)):
    extracted = await extract_upload_file(file)
    req = IngestTextRequest(
        doc_id=doc_id,
        text=extracted.text,
        metadata=extracted.metadata
    )
    return run_ingest(req, store=store, embedder=embedder)

@app.post("/chat_rag", response_model=ChatRagResponse)
def chat_rag(req: ChatRagRequest):
    return run_chat_rag(req, store=store, embedder=embedder, model=llm_manager.model, tokenizer=llm_manager.tokenizer)


@app.post("/chat", response_model=ChatResponse)
def chat_llm(prompt: Prompt):
    return {
        "answer": generate_text(prompt.prompt, tokenizer=llm_manager.tokenizer, model=llm_manager.model)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
