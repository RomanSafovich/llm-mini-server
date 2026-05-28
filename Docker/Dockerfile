FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/llm-mini-server

COPY requirements.txt .


# HuggingFace cache directory
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    PYTHONUNBUFFERED=1


# Install CUDA-enabled PyTorch + transformers stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" "torchaudio==2.4.1+cu121" \
        --index-url https://download.pytorch.org/whl/cu121


RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/llm-mini-server

EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]