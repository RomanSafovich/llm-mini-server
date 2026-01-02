from fastapi import FastAPI, HTTPException

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pydantic import BaseModel
import requests
import uvicorn

app = FastAPI()


class Prompt(BaseModel):
    prompt: str


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


@app.post("/chat")
def chat_llm(prompt: Prompt):
    try:
        inputs = tokenizer(prompt.prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)