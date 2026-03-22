from fastapi import FastAPI, HTTPException
from app.schemas import (
    Prompt, 
    IngestTextRequest, 
    IngestTextResponse, 
    ChatRagRequest, 
    SourceOut, 
    ChatRagResponse
)
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer
)
import torch
import uvicorn
from app import config
from app.store.vector_store import InMemoryVectorStore
from app.embeddings.embedder import Embedder
from app.llm import generate_text
from app.ingest import chunk_text, run_ingest
from app.rag import run_chat_rag

app = FastAPI()


store = InMemoryVectorStore()
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
# Load model and tokenizer at startup

print("Loading model... this may take a minute ⏳")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# GPU if available; otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# using float32 for CPU; using float16 for GPU
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,  # helpful even without accelerate
)

model.to(device)
print("Model loaded ✅")
print("Model loaded ✅ (cuda? ", torch.cuda.is_available(), ")")
model.eval()


@app.post("/ingest_text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest):
    return run_ingest(req, store=store, embedder=embedder)
    

@app.post("/chat_rag", response_model=ChatRagResponse)
def chat_rag(req: ChatRagRequest):
    return run_chat_rag(req, store=store, embedder=embedder, model=model, tokenizer=tokenizer)


@app.post("/chat")
def chat_llm(prompt: Prompt):
    return {
        "answer": generate_text(prompt.prompt, tokenizer=tokenizer, model=model)
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)