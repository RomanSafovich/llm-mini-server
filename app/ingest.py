from fastapi import HTTPException
from app.schemas import IngestTextResponse, IngestTextRequest

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


def run_ingest(req: IngestTextRequest, store, embedder) -> IngestTextResponse:
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