from app.schemas import ChatRagRequest, ChatRagResponse, SourceOut
from app.llm import generate_text
from app.logger import logger
from app.config import settings
import numpy as np
from fastapi import HTTPException

def run_chat_rag(req: ChatRagRequest, store, embedder, model, tokenizer) -> ChatRagResponse:
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="question must not be blank")


    effective_top_k = min(req.top_k, settings.max_top_k)
    hits = retrieve_unique_hits(question, effective_top_k=effective_top_k, store=store, embedder=embedder)
    retrieved_count = len(hits)


    if retrieved_count == 0:
        return  ChatRagResponse(
            answer="No relevant context found. Please ingest documents first.",
            sources=[],
            retrieved_count=0
        )

    top1_score = hits[0]["score"]
    top2_score = hits[1]["score"] if retrieved_count > 1 else 0
    margin = top1_score - top2_score
    top_k_scores = [h["score"] for h in hits]
    logger.info(f"top1_score={top1_score}, top2_score={top2_score}, margin={margin},top_k_scores={top_k_scores}")

    if top1_score < settings.score_threshold or margin < settings.margin_threshold:
        logger.info("MODE: fallback")
        ans = generate_text(question, tokenizer=tokenizer, model=model)
        return ChatRagResponse (
            answer=ans,
            sources=[],
            retrieved_count=0
        )
    else:
        logger.info("MODE: rag")
        concat_text, used_hits, used_chunks, used_chars = build_context(hits)

        logger.info(f"CTX: effective_top_k={effective_top_k}, used_chunks={used_chunks}, used_chars={used_chars}")

        sources_out = build_sources_out(used_hits, req.debug)

        augmented_prompt = build_augmented_prompt(concat_text, question, n_sources=len(used_hits))
                        
        ans = generate_text(augmented_prompt, tokenizer=tokenizer, model=model)
        return ChatRagResponse(
            answer=ans,
            sources=sources_out,
            retrieved_count=len(used_hits)
        )


def retrieve_unique_hits(question, effective_top_k, store, embedder):
    query_embedding = embedder.encode_one(question)
    hits = store.search(query_embedding, effective_top_k)
    unique_hits = []
    for hit in hits:
        sim = 0
        is_duplicate = False
        for unique_hit in unique_hits:
            sim = np.dot(hit["embedding"], unique_hit["embedding"])
            if sim > settings.near_duplicate_cosine_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_hits.append(hit)

    return unique_hits 


def build_context(hits):
    used_hits = []
    concat_text = ""
    used_chars = 0
    used_chunks = 0
    for i, hit in enumerate(hits):
        snippet = hit["text"][:settings.max_chunk_snippet_chars]
        block = f"SOURCE {i+1}\n{snippet}\n\n"
        if len(block) + used_chars > settings.max_context_chars:
            break
        used_chars += len(block)
        used_chunks += 1
        concat_text += block
        used_hits.append(hit)

    return concat_text, used_hits, used_chunks, used_chars



def build_sources_out(used_hits, debug):
    sources_out = []
    for hit in used_hits:
        snippet = hit["text"][:settings.source_snippet_chars]
        text = hit["text"] if debug else None
        sources_out.append(
            SourceOut(
                id = hit["id"],
                score = hit["score"],
                metadata = hit["metadata"],
                snippet = snippet,
                text = text
            )
        )
    return sources_out



def build_augmented_prompt(concat_text, question, n_sources):
    return "Instruction:\n" \
            "Answer using only the context below.\n" \
            "The context is divided into blocks labeled SOURCE 1, SOURCE 2, etc. When you use a fact, cite it like [SOURCE 1] or [SOURCE 2].\n" \
            f"Only cite sources that exist in the context: SOURCE 1 to SOURCE {n_sources}. Do not invent other source numbers.\n" \
            "If the context does not contain the information needed to answer the question, reply exactly: \"I don’t know based on the provided context.\"\n" \
            "Do not guess or add external knowledge.\n" \
            f"Context:\n{concat_text}\n" \
            f"Question:\n{question}\n" \
            f"Answer:\n"