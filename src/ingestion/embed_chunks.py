import argparse
import json
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm
from sqlalchemy import create_engine, text as sql
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")



# --------- Embedders ---------

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 128):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            # preserve order
            batch_vecs = [d.embedding for d in resp.data]
            vectors.extend(batch_vecs)
        return np.asarray(vectors, dtype=np.float32)

class BGEEmbedder:
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

# --------- SQLite metadata ---------

def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(sql("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            source_file TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            char_len INTEGER NOT NULL,
            chunk_path TEXT NOT NULL,
            embedding_path TEXT NOT NULL
        );
        """))
        conn.execute(sql("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);"))
        conn.execute(sql("CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_chunk_id ON chunks(chunk_id);"))

def upsert_chunk_rows(db_path: Path, rows: List[Dict]) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        for r in rows:
            conn.execute(sql("""
                INSERT OR IGNORE INTO chunks
                (doc_id, source_file, chunk_index, chunk_id, char_len, chunk_path, embedding_path)
                VALUES
                (:doc_id, :source_file, :chunk_index, :chunk_id, :char_len, :chunk_path, :embedding_path)
            """), r)

# --------- Processing ---------

def read_chunks_jsonl(chunk_file: Path) -> List[Dict]:
    chunks = []
    with chunk_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
    return chunks


def main(
    chunks_dir: str,
    embeddings_dir: str,
    db_path: str,
    provider: str,
    openai_model: str,
    bge_model: str,
) -> None:
    chunks_path = Path(chunks_dir)
    emb_path = Path(embeddings_dir)
    emb_path.mkdir(parents=True, exist_ok=True)

    db_file = Path(db_path)
    init_db(db_file)

    chunk_files = sorted([p for p in chunks_path.glob("*.jsonl") if p.is_file()])
    if not chunk_files:
        print(f"[WARN] No chunk files found in {chunks_path}")
        return

    # Choose embedder
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set. export OPENAI_API_KEY=...")
        embedder = OpenAIEmbedder(model=openai_model)
    elif provider == "bge":
        embedder = BGEEmbedder(model=bge_model)
    else:
        raise ValueError("provider must be one of: openai, bge")

    print(f"[INFO] Embedding {len(chunk_files)} chunk files using provider={provider}")
    for cf in tqdm(chunk_files, desc="Embedding docs"):
        chunks = read_chunks_jsonl(cf)
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        vectors = embedder.embed(texts)  # shape: (n_chunks, dim)

        # Save embeddings per doc (one .npy per source doc)
        out_npy = emb_path / (cf.stem + ".npy")
        np.save(out_npy, vectors)

        # Write metadata rows into sqlite
        rows = []
        for c in chunks:
            rows.append({
                "doc_id": c["doc_id"],
                "source_file": c["source_file"],
                "chunk_index": int(c["chunk_index"]),
                "chunk_id": c["chunk_id"],
                "char_len": int(c["char_len"]),
                "chunk_path": str(cf),
                "embedding_path": str(out_npy),
            })
        upsert_chunk_rows(db_file, rows)

    print("[DONE] Embeddings + metadata complete.")
    print(f"Embeddings: {emb_path}")
    print(f"Metadata DB: {db_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunk JSONL files and write embeddings + SQLite metadata.")
    parser.add_argument("--chunks_dir", required=True, help="Directory containing chunk JSONL files (e.g., data/chunks)")
    parser.add_argument("--embeddings_dir", required=True, help="Directory to write .npy embeddings (e.g., data/embeddings)")
    parser.add_argument("--db_path", default="data/metadata/chunks.sqlite", help="SQLite DB path for chunk metadata")
    parser.add_argument("--provider", choices=["openai", "bge"], default="openai", help="Embedding provider")
    parser.add_argument("--openai_model", default="text-embedding-3-large", help="OpenAI embedding model")
    parser.add_argument("--bge_model", default="BAAI/bge-large-en-v1.5", help="BGE model name")
    args = parser.parse_args()

    main(
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        db_path=args.db_path,
        provider=args.provider,
        openai_model=args.openai_model,
        bge_model=args.bge_model,
    )