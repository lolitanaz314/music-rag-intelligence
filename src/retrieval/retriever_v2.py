from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from .query_rewrite import QueryRewriter
from .rerank import make_reranker

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


@dataclass
class RetrievalHit:
    row_id: int
    score: float               # rerank score if reranked, else faiss score
    doc_id: str
    source_file: str
    chunk_index: int
    chunk_id: str
    text: str


class RetrieverV2:
    def __init__(
        self,
        index_path: str = "vectorstore/faiss_index.bin",
        chunks_jsonl_path: str = "data/merged/all_chunks.jsonl",
        embedding_model: str = "text-embedding-3-large",
        rewrite_model: str = "gpt-4.1-mini",
        reranker_kind: str = "cohere",  # "local" or "cohere"
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env.")
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

        self.index_path = Path(index_path)
        self.chunks_jsonl_path = Path(chunks_jsonl_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.chunks_jsonl_path.exists():
            raise FileNotFoundError(f"Chunks JSONL not found: {self.chunks_jsonl_path}")

        self.index = faiss.read_index(str(self.index_path))

        self.rewriter = QueryRewriter(model=rewrite_model)
        self.reranker: Reranker = make_reranker(reranker_kind)

    def _embed_query(self, q: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embedding_model, input=[q])
        v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(v)  # must match cosine/IP index build
        return v

    def _fetch_by_row_ids(self, row_ids: set[int]) -> Dict[int, Dict]:
        out = {}
        with self.chunks_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid = int(obj["row_id"])
                if rid in row_ids:
                    out[rid] = obj
                    if len(out) == len(row_ids):
                        break
        return out

    def retrieve(self, query: str, top_n: int = 50, rewrite: bool = True) -> Tuple[str, List[RetrievalHit]]:
        """
        Returns (final_query_used, hits) from FAISS only (no rerank).
        """
        final_query = query
        if rewrite:
            rr = self.rewriter.rewrite(query)
            final_query = rr.rewritten

        qv = self._embed_query(final_query)
        D, I = self.index.search(qv, top_n)

        row_ids = [int(x) for x in I[0].tolist()]
        scores = [float(s) for s in D[0].tolist()]

        objs = self._fetch_by_row_ids(set(row_ids))

        hits: List[RetrievalHit] = []
        for rid, sc in zip(row_ids, scores):
            obj = objs.get(rid)
            if not obj:
                continue
            hits.append(
                RetrievalHit(
                    row_id=rid,
                    score=sc,
                    doc_id=obj["doc_id"],
                    source_file=obj["source_file"],
                    chunk_index=int(obj["chunk_index"]),
                    chunk_id=obj["chunk_id"],
                    text=obj["text"],
                )
            )
        return final_query, hits

    def search(self, query: str, top_k: int = 5, top_n: int = 50, rewrite: bool = True) -> Dict:
        """
        Full Day-4 pipeline: rewrite -> retrieve top_n -> rerank -> return top_k.
        """
        final_query, candidates = self.retrieve(query, top_n=top_n, rewrite=rewrite)
        if not candidates:
            return {"query": query, "final_query": final_query, "hits": []}

        docs = [c.text for c in candidates]
        rr = self.reranker.rerank(final_query, docs, top_k=top_k)

        reranked = []
        for idx, score in zip(rr.indices, rr.scores):
            c = candidates[idx]
            reranked.append(
                RetrievalHit(
                    row_id=c.row_id,
                    score=score,  # rerank score now
                    doc_id=c.doc_id,
                    source_file=c.source_file,
                    chunk_index=c.chunk_index,
                    chunk_id=c.chunk_id,
                    text=c.text,
                )
            )

        return {
            "query": query,
            "final_query": final_query,
            "hits": reranked,
        }


if __name__ == "__main__":
    r = RetrieverV2(reranker_kind="cohere")

    query = "How do mechanical royalties differ in the US vs EU?"

    print("\n==============================")
    print("USER QUERY:")
    print(query)
    print("==============================")

    # -------- BEFORE: FAISS ONLY --------
    final_query, faiss_hits = r.retrieve(
        query,
        top_n=5,        # top 5 directly from FAISS
        rewrite=True
    )

    print("\n--- TOP 5 BEFORE RERANK (FAISS ONLY) ---")
    print("Rewritten query:", final_query)

    for i, h in enumerate(faiss_hits, start=1):
        print(f"\n#{i} FAISS score={h.score:.4f} doc={h.doc_id} chunk={h.chunk_index}")
        print(h.text[:400].replace("\n", " ") + "...")

    # -------- AFTER: FAISS + RERANK --------
    out = r.search(
        query,
        top_n=50,   # retrieve more candidates
        top_k=5,    # rerank down to 5
        rewrite=True
    )

    print("\n--- TOP 5 AFTER RERANK ---")
    print("Rewritten query:", out["final_query"])

    for i, h in enumerate(out["hits"], start=1):
        print(f"\n#{i} RERANK score={h.score:.4f} doc={h.doc_id} chunk={h.chunk_index}")
        print(h.text[:400].replace("\n", " ") + "...")