from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


@dataclass
class RerankResult:
    indices: List[int]        # indices into original candidates
    scores: List[float]       # reranker scores aligned to indices


class Reranker:
    def rerank(self, query: str, docs: List[str], top_k: int) -> RerankResult:
        raise NotImplementedError


class CohereReranker(Reranker):
    def __init__(self, model: str = "rerank-english-v3.0") -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY not set (.env).")
        import cohere
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, query: str, docs: List[str], top_k: int) -> RerankResult:
        resp = self.client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )
        # Cohere returns items with index + relevance_score
        indices = [r.index for r in resp.results]
        scores = [float(r.relevance_score) for r in resp.results]
        return RerankResult(indices=indices, scores=scores)


class LocalCrossEncoderReranker(Reranker):
    """
    Local reranker using SentenceTransformers CrossEncoder.
    Good default model: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast).
    If you want heavier: 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    """
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model)

    def rerank(self, query: str, docs: List[str], top_k: int) -> RerankResult:
        pairs = [(query, d) for d in docs]
        scores = self.ce.predict(pairs)  # array-like
        # sort by score desc
        import numpy as np
        idx = np.argsort(-scores)[: min(top_k, len(docs))]
        return RerankResult(indices=[int(i) for i in idx], scores=[float(scores[i]) for i in idx])


def make_reranker(kind: Literal["cohere", "local"]) -> Reranker:
    if kind == "cohere":
        return CohereReranker()
    if kind == "local":
        return LocalCrossEncoderReranker()
    raise ValueError("kind must be one of: cohere, local")
