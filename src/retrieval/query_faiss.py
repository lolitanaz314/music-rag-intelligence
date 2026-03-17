import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import faiss
import os

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def get_texts_by_row_ids(chunks_jsonl_path: Path, row_ids: set[int]) -> dict[int, dict]:
    # Simple streaming lookup (fine for now)
    out = {}
    with chunks_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rid = int(obj["row_id"])
            if rid in row_ids:
                out[rid] = obj
                if len(out) == len(row_ids):
                    break
    return out


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed_one(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        v = np.array(resp.data[0].embedding, dtype=np.float32)
        return v


def main(query: str, k: int, index_path: str, chunks_jsonl: str, provider: str):
    index = faiss.read_index(index_path)

    if provider != "openai":
        raise ValueError("For now, query script supports provider=openai (matches your embedding choice).")

    emb = OpenAIEmbedder().embed_one(query)
    faiss.normalize_L2(emb.reshape(1, -1))
    D, I = index.search(emb.reshape(1, -1), k)

    row_ids = set(int(x) for x in I[0].tolist())
    texts = get_texts_by_row_ids(Path(chunks_jsonl), row_ids)

    print("\n=== TOP RESULTS ===")
    for rank, rid in enumerate(I[0].tolist(), start=1):
        obj = texts.get(int(rid))
        if not obj:
            continue
        print(f"\n#{rank}  score={D[0][rank-1]:.4f}  doc={obj['doc_id']}  chunk_index={obj['chunk_index']}")
        print(obj["text"][:600].replace("\n", " ") + ("..." if len(obj["text"]) > 600 else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index", default="data/vectorstore/faiss.index")
    p.add_argument("--chunks_jsonl", default="data/merged/all_chunks.jsonl")
    p.add_argument("--provider", default="openai")
    args = p.parse_args()

    main(args.query, args.k, args.index, args.chunks_jsonl, args.provider)
