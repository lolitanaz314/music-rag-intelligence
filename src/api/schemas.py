from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str
    top_n: int = 50
    top_k: int = 5
    rewrite: bool = True


class ChunkHit(BaseModel):
    row_id: int
    score: float
    doc_id: str
    source_file: str
    chunk_index: int
    chunk_id: str
    text: str


class SearchResponse(BaseModel):
    query: str
    final_query: str
    hits: List[ChunkHit]


class AnswerRequest(BaseModel):
    query: str
    top_n: int = 50
    top_k: int = 5
    rewrite: bool = True
    max_sources: int = 6


class Source(BaseModel):
    cite_id: str
    doc_id: str
    chunk_index: int
    chunk_id: str


class AnswerResponse(BaseModel):
    query: str
    final_query: str
    answer: str
    sources: List[Source]