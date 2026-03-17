from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Load .env from repo root robustly
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


@dataclass
class RetrievalHit:
    row_id: int
    score: float
    doc_id: str
    source_file: str
    chunk_index: int
    chunk_id: str
    text: str


class Retriever:
    """
    Simple FAISS retriever over merged corpus.
    Assumes:
      - FAISS index built from normalized embeddings (cosine via IP)
      - data/merged/all_chunks.jsonl contains row_id aligned with embeddings row order
    """

    def __init__(
        self,
        index_path: str = "vectorstore/faiss_index.bin",
        chunks_jsonl_path: str = "data/merged/all_chunks.jsonl",
        embedding_model: str = "text-embedding-3-large",
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Put it in .env at repo root.")

        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

        self.index_path = index_path
        self.chunks_jsonl_path = Path(chunks_jsonl_path)

        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not self.chunks_jsonl_path.exists():
            raise FileNotFoundError(f"Chunks JSONL not found: {self.chunks_jsonl_path}")

        self.index = faiss.read_index(index_path)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embedding_model, input=[query])
        v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        # Must match build step using cosine(IP) on normalized vectors
        faiss.normalize_L2(v)
        return v

    def _fetch_by_row_ids(self, row_ids: set[int]) -> Dict[int, Dict[str, Any]]:
        """
        Stream scan JSONL to fetch matching row_ids. Fine for now; later we’ll store text in SQLite for speed.
        """
        hits: Dict[int, Dict[str, Any]] = {}
        with self.chunks_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid = int(obj["row_id"])
                if rid in row_ids:
                    hits[rid] = obj
                    if len(hits) == len(row_ids):
                        break
        return hits

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        qv = self._embed_query(query)
        D, I = self.index.search(qv, top_k)

        row_ids = [int(x) for x in I[0].tolist()]
        scores = [float(s) for s in D[0].tolist()]

        objs = self._fetch_by_row_ids(set(row_ids))

        results: List[RetrievalHit] = []
        for rid, score in zip(row_ids, scores):
            obj = objs.get(rid)
            if not obj:
                continue
            results.append(
                RetrievalHit(
                    row_id=rid,
                    score=score,
                    doc_id=obj["doc_id"],
                    source_file=obj["source_file"],
                    chunk_index=int(obj["chunk_index"]),
                    chunk_id=obj["chunk_id"],
                    text=obj["text"],
                )
            )
        return results


if __name__ == "__main__":
    r = Retriever()
    q = "How do mechanical royalties differ in the US vs EU?"
    hits = r.search(q, top_k=5)

    print("\n=== TOP HITS ===")
    for i, h in enumerate(hits, start=1):
        print(f"\n#{i} score={h.score:.4f} doc={h.doc_id} chunk={h.chunk_index}")
        print(h.text[:700].replace("\n", " ") + ("..." if len(h.text) > 700 else ""))
