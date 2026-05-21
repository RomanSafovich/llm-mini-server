from fastapi import FastAPI, HTTPException
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import torch
import uvicorn
from app.config import settings
from app.store.milvus_store import MilvusVectorStore
from app.embeddings.embedder import Embedder
from app.llm import generate_text
from app.ingest import run_ingest
from app.rag import run_chat_rag
from app.logger import logger

app = FastAPI()


store = MilvusVectorStore()

embedder = Embedder(model_name=settings.embedder_model_name)

logger.info(f"Loading model {settings.llm_model_name}... this may take a minute ⏳")
tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)

# GPU if available; otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# using float32 for CPU; using float16 for GPU
# dtype = torch.bfloat16 if device == "cuda" else torch.float32

if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )


# model.to(device)
logger.info(f"Model loaded successfully (CUDA: {torch.cuda.is_available()}) ✅")
model.eval()


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


@app.post("/chat_rag", response_model=ChatRagResponse)
def chat_rag(req: ChatRagRequest):
    return run_chat_rag(req, store=store, embedder=embedder, model=model, tokenizer=tokenizer)


@app.post("/chat", response_model=ChatResponse)
def chat_llm(prompt: Prompt):
    return {
        "answer": generate_text(prompt.prompt, tokenizer=tokenizer, model=model)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
