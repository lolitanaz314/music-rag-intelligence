from __future__ import annotations

from fastapi import APIRouter, HTTPException
from src.api.schemas import (
    SearchRequest, SearchResponse, ChunkHit,
    AnswerRequest, AnswerResponse, Source
)
from src.retrieval.retriever_v2 import RetrieverV2
from src.generation.answer_with_citations import AnswerGenerator

retriever = RetrieverV2(reranker_kind="local")
router = APIRouter()

# Load once at startup (important for performance)
generator = AnswerGenerator()


def get_retriever():
    global _retriever
    if _retriever is None:
        try:
            _retriever = RetrieverV2(reranker_kind="cohere")
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Retriever not ready: {e}. Build the FAISS index first."
            )
    return _retriever


@router.get("/health")
def health():
    return {"ok": True}

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    out = retriever.search(req.query, top_k=req.top_k, top_n=req.top_n, rewrite=req.rewrite)

    hits = [
        ChunkHit(
            row_id=h.row_id,
            score=h.score,
            doc_id=h.doc_id,
            source_file=h.source_file,
            chunk_index=h.chunk_index,
            chunk_id=h.chunk_id,
            text=h.text,
        )
        for h in out["hits"]
    ]

    return SearchResponse(query=req.query, final_query=out["final_query"], hits=hits)


@router.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    out = retriever.search(req.query, top_k=req.top_k, top_n=req.top_n, rewrite=req.rewrite)

    hits_for_gen = [
        {"doc_id": h.doc_id, "chunk_index": h.chunk_index, "chunk_id": h.chunk_id, "text": h.text}
        for h in out["hits"]
    ]

    ans = generator.generate(
        user_query=req.query,
        used_query=out["final_query"],
        top_hits=hits_for_gen,
        max_sources=req.max_sources,
    )

    sources = [
        Source(cite_id=s.cite_id, doc_id=s.doc_id, chunk_index=s.chunk_index, chunk_id=s.chunk_id)
        for s in ans.sources
    ]

    return AnswerResponse(query=req.query, final_query=ans.used_query, answer=ans.answer, sources=sources)