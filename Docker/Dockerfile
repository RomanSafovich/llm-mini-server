FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/mistral

# HuggingFace cache directory
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    PYTHONUNBUFFERED=1

# Install CUDA-enabled PyTorch + transformers stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121" \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir sentencepiece fastapi uvicorn[standard] transformers accelerate sentence-transformers

# ✅ Pre-download Mistral into /models
# RUN python3 - << 'EOF'
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# model = "mistralai/Mistral-7B-Instruct-v0.2"

# print(">>> Downloading tokenizer...")
# AutoTokenizer.from_pretrained(model)

# print(">>> Downloading model...")
# AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)

# print(">>> Mistral download complete!")
# EOF

# Copy your FastAPI app
COPY main.py /workspace/mistral/main.py

EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host","0.0.0.0","--port","8000"]