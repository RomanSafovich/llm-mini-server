# LLM Mini Server

A local **RAG (Retrieval-Augmented Generation)** backend built with **FastAPI**, **Milvus**, **sentence-transformers**, and a local LLM.

The project supports document ingestion, semantic search, and source-grounded question answering over locally indexed documents.

## Overview

LLM Mini Server provides:

* plain local LLM chat
* text ingestion into a vector store
* RAG-based question answering over indexed documents
* persistent vector search with Milvus
* document lifecycle management

## Architecture

```mermaid
graph TD
       User["Client"] --> API["FastAPI API"]

       API --> Chat["/chat"]
       Chat --> LLM["Local LLM"]

       API --> Ingest["/ingest_text"]
       Ingest --> Embed["Chunk + Embed"]
       Embed --> Store["Milvus"]

       API --> Rag["/chat_rag"]
       Rag --> Retrieve["Retrieve Context"]
       Retrieve --> Store
       Store --> Context["Retrieved Context"]
       Context --> LLM

       API --> Docs["Document Endpoints"]
       Docs --> Store
```

## Features

### Local LLM Chat

The `/chat` endpoint sends a prompt directly to the local language model.

This mode is used for general, non-document-based responses.

### RAG Chat

The `/chat_rag` endpoint performs document-aware question answering over indexed content.

The RAG workflow includes:

1. embedding the user question
2. retrieving relevant chunks from Milvus
3. filtering duplicate or low-quality context
4. building a context window from retrieved chunks
5. generating a grounded answer with source metadata

### Text Ingestion

The `/ingest_text` endpoint accepts raw text, splits it into chunks, embeds each chunk, and stores the result in Milvus.

The ingestion pipeline includes:

* input validation
* text chunking with overlap
* embedding generation
* vector store upsert
* metadata tracking by document ID and chunk index

### Persistent Vector Store

The project uses **Milvus** as the persistent vector database for storing and searching embedded document chunks.

Stored chunk data includes:

* chunk ID
* chunk text
* embedding vector
* document ID
* chunk index

### Document Lifecycle Management

The API includes endpoints for managing indexed documents:

* list indexed documents
* delete a document by `doc_id`
* clear all stored documents

## API Endpoints

| Method   | Endpoint              | Description                          |
| -------- | --------------------- | ------------------------------------ |
| `POST`   | `/chat`               | Plain local LLM response             |
| `POST`   | `/chat_rag`           | RAG response using indexed documents |
| `POST`   | `/ingest_text`        | Ingest text into the vector store    |
| `GET`    | `/documents`          | List indexed documents               |
| `DELETE` | `/documents/{doc_id}` | Delete a document by ID              |
| `POST`   | `/documents/clear`    | Clear all indexed documents          |

## Tech Stack

* **Backend:** FastAPI, Pydantic, Uvicorn, Python
* **LLM / Embeddings:** Hugging Face Transformers, sentence-transformers, local instruction-tuned LLM
* **Vector database:** Milvus, Attu
* **Infrastructure:** Docker, Docker Compose
* **Testing:** pytest, FastAPI TestClient, unittest / mocking

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/RomanSafovich/llm-mini-server.git
cd llm-mini-server
```

### 2. Build the API image

```bash
docker build -f Docker/Dockerfile -t llm-mini-server .
```

### 3. Start the stack and follow API logs

```bash
docker-compose -f Docker/docker-compose.yml up -d && docker logs -f llm-api
```

### 4. Stop the stack

```bash
# Press Ctrl+C to exit the Docker logs stream first
docker-compose -f Docker/docker-compose.yml down
```

## Example Usage

### Ingest text

```http
POST /ingest_text
```

Example request:

```json
{
  "doc_id": "rag_notes",
  "text": "Milvus is used as the persistent vector store for semantic retrieval..."
}
```

### Ask a RAG question

```http
POST /chat_rag
```

Example request:

```json
{
  "question": "What vector database does this project use?",
  "top_k": 5,
  "debug": true
}
```

### Ask a plain chat question

```http
POST /chat
```

Example request:

```json
{
  "prompt": "Explain what cosine similarity means."
}
```

## Running Tests

Run the test suite with:

```bash
pytest
```

## Project Goals

LLM Mini Server is built to run local RAG workflows without relying on external LLM APIs.

The project is designed to:

* keep the API small and easy to understand
* support local document ingestion and retrieval
* use a persistent vector database instead of in-memory storage
* separate plain chat, RAG chat, and future agent workflows
* stay modular enough to extend over time

## Planned Improvements

Future improvements may include:

* file and PDF ingestion
* query rewriting, hybrid search, and reranking
* improved source formatting
* conversation memory
* simple UI for chat and document management

## Future Agent Direction

A future `/research_agent` endpoint is planned for multi-step document research.

The current endpoint design is:

* `/chat` — plain local LLM response
* `/chat_rag` — single retrieval + grounded answer
* `/research_agent` — multi-step retrieval and reasoning over documents

This keeps the existing API simple while leaving room for a controlled tool-using agent workflow later.

## License

This project is licensed under the Apache 2.0 License.
