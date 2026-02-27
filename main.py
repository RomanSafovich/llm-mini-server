from fastapi import FastAPI, HTTPException

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel, Field
import requests
import uvicorn
from typing import Any
from vector_store import InMemoryVectorStore
from embeddings import Embedder

app = FastAPI()


class Prompt(BaseModel):
    prompt: str

store = InMemoryVectorStore()
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
# Load model and tokenizer once at startup
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading model... this may take a minute ⏳")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPU if available; otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# For CPU use float32; for GPU you can use float16
dtype = torch.float16 if device == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,  # helpful even without accelerate
)
model.to(device)
print("Model loaded ✅")

print("Model loaded ✅ (cuda? ", torch.cuda.is_available(), ")")

model.eval()


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


class ChatRagResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_count: int = 0

def chunk_text(text: str, chunk_size: int=800, overlap: int=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    
    chunks = []
    if not text:
        return chunks
    start = 0
    while start < len(text):
        end = min(len(text), start+chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        

    return chunks


@app.post("/ingest_text", response_model=IngestTextResponse)
def ingest_text(req: IngestTextRequest):

    doc_id = req.doc_id.strip()

    if not req.doc_id or not doc_id:
        raise HTTPException(status_code=400, detail="doc_id must not be blank")

    if doc_id.lower() == "string":
        raise HTTPException(status_code=400, detail="doc_id must be a meaningful value, not default placeholder 'string'")

    if len(doc_id) < 3 or len(doc_id) > 80:
        raise HTTPException(status_code=400, detail="doc_id length must be between 3 and 80 characters")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be blank")

    chunks = chunk_text(req.text)
    if not chunks:
        return IngestTextResponse(
            doc_id=doc_id,
            chunks_added=0,
            total_chunks=store.count()
        )

    vectors = embedder.encode_many(chunks)

    if len(vectors) != len(chunks):
        raise HTTPException(status_code=500, detail="embedding count does not match chunk count")

    items = []

    for i, chunk in enumerate(chunks):
        item = {
            "id": f"{doc_id}_{i}",
            "text": chunk,
            "embedding": vectors[i],
            "metadata": {
                **req.metadata,
                "doc_id": doc_id,
                "chunk_index": i
            }

        }
        items.append(item)

    store.add_many(items)

    return IngestTextResponse(
        doc_id=doc_id,
        chunks_added=len(chunks),
        total_chunks=store.count()
    )

def generate_text(prompt_str):
    try:
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            new_tokens = output_tokens[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_rag", response_model=ChatRagResponse)
def chat_rag(req: ChatRagRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="question must not be blank")


    query_vector = embedder.encode_one(question)
    hits = store.search(query_vector, req.top_k)
    retrieved_count = len(hits)

    if retrieved_count == 0:
        return  ChatRagResponse(
            answer="No relevant context found. Please ingest documents first.",
            sources=[],
            retrieved_count=0
        )

    concat_text = ""
    for i, hit in enumerate(hits):
        concat_text += f"SOURCE {i+1}\n{hit['text']}\n\n" 

    augmented_prompt = f"Instruction:\nAnswer using only the context below. If the answer is not in the context, say you don’t know.\n" \
                    f"Context:\n{concat_text}\n" \
                    f"Question:\n{question}\n" \
                    f"Answer:\n"
                    
    ans = generate_text(augmented_prompt)
    return ChatRagResponse(
        answer=ans,
        sources=hits,
        retrieved_count=retrieved_count
    )


@app.post("/chat")
def chat_llm(prompt: Prompt):
    return {
        "answer": generate_text(prompt.prompt)
        }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)