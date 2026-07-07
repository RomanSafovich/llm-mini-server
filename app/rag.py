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
    hits = retrieve_unique_hits(question, effective_top_k=effective_top_k, store=store, embedder=embedder, doc_id=req.doc_id)
    retrieved_count = len(hits)


    if retrieved_count == 0:
        logger.info("MODE: no_retrieval_hits")
        return ChatRagResponse(
            answer="No relevant context found. Please ingest documents first.",
            sources=[],
            retrieved_count=0
        )

    top1_score = hits[0]["score"]
    top2_score = hits[1]["score"] if retrieved_count > 1 else 0
    margin = top1_score - top2_score
    top_k_scores = [h["score"] for h in hits]
    logger.info(f"top1_score={top1_score}, top2_score={top2_score}, margin={margin},top_k_scores={top_k_scores}")

    if top1_score < settings.score_threshold:
        logger.info("MODE: low_confidence")
        return ChatRagResponse(
            answer="No sufficiently relevant information was found in the indexed documents.",
            sources=[],
            retrieved_count=0
        )
    else:
        logger.info("MODE: rag")
        concat_text, used_hits, used_chunks, used_chars = build_context(hits)

        logger.info(f"CTX: effective_top_k={effective_top_k}, used_chunks={used_chunks}, used_chars={used_chars}")

        sources_out = build_sources_out(used_hits, req.debug)

        augmented_prompt = build_augmented_prompt(concat_text, question)
                        
        ans = generate_text(augmented_prompt, tokenizer=tokenizer, model=model)
        return ChatRagResponse(
            answer=ans,
            sources=sources_out,
            retrieved_count=len(used_hits)
        )


def retrieve_unique_hits(question, effective_top_k, store, embedder, doc_id):
    filter_expr = None
    if doc_id:
        filter_expr = f'doc_id == "{doc_id}"'

    query_embedding = embedder.encode_one(question)
    hits = store.search(query_embedding, effective_top_k, filters=filter_expr)
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


def build_citation(hit) -> str:
    return f'{hit["metadata"]["doc_id"]}:chunk_{hit["metadata"]["chunk_index"]}'


def build_context(hits):
    used_hits = []
    concat_text = ""
    used_chars = 0
    used_chunks = 0
    for hit in hits:
        snippet = hit["text"][:settings.max_chunk_snippet_chars]
        block = f"{build_citation(hit)}\n{snippet}\n\n"
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
                citation = build_citation(hit),
                text = text
            )
        )
    return sources_out



def build_augmented_prompt(concat_text, question):
    return "Instruction:\n" \
            "Answer using only the context below.\n" \
            "Each context block begins with its exact citation label.\n" \
            "When you use a fact, cite that label in square brackets, " \
            "for example [doc_1:chunk_0].\n" \
            "For multiple sources, cite each label separately, for example " \
            "[doc_1:chunk_0] [doc_1:chunk_5]. " \
            "Never place multiple citation labels inside one pair of brackets.\n" \
            "Use only citation labels that appear in the context. " \
            "Do not invent or modify citation labels.\n" \
            "If the context does not contain the information needed to answer " \
            "the question, reply exactly: " \
            "\"I don't know based on the provided context.\"\n" \
            "Do not guess or add external knowledge.\n" \
            f"Context:\n{concat_text}\n" \
            f"Question:\n{question}\n" \
            "Answer:\n"
