import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from sqlalchemy import create_engine, text as sql

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_rows(db_path: Path, doc_id: str | None = None) -> List[Dict]:
    engine = create_engine(f"sqlite:///{db_path}")
    q = """
      SELECT doc_id, domain, chunk_id, chunk_index, embedding_path, embedding_model, embedding_dim
      FROM chunks
    """
    params = {}
    if doc_id:
        q += " WHERE doc_id = :doc_id"
        params["doc_id"] = doc_id
    q += " ORDER BY doc_id, chunk_index"

    with engine.begin() as conn:
        rows = conn.execute(sql(q), params).mappings().all()
    return [dict(r) for r in rows]

def build_index(vectors: np.ndarray) -> faiss.Index:
    # cosine similarity via inner-product on L2-normalized vectors
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vectors)
    return idx

def main(db_path: str, out_dir: str, doc_id: str | None = None):
    db = Path(db_path)
    out = Path(out_dir)
    ensure_dir(out)

    rows = load_rows(db, doc_id=doc_id)
    if not rows:
        print("[WARN] No rows found in chunks.sqlite.")
        return

    # group: domain -> list of (chunk_id, embedding_path, chunk_index)
    by_domain: Dict[str, List[Dict]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    for domain, items in by_domain.items():
        domain_dir = out / domain
        ensure_dir(domain_dir)

        # Load all vectors in correct order:
        # embeddings are stored per doc as .npy; we need per-chunk vectors.
        vectors_list = []
        id_map: List[str] = []

        # cache per doc embeddings
        cache: Dict[str, np.ndarray] = {}

        for r in items:
            d = r["doc_id"]
            ep = Path(r["embedding_path"])
            if d not in cache:
                cache[d] = np.load(ep)  # shape: (n_chunks, dim)
            vec = cache[d][int(r["chunk_index"])]
            vectors_list.append(vec)
            id_map.append(r["chunk_id"])

        X = np.vstack(vectors_list).astype("float32")
        index = build_index(X)

        faiss.write_index(index, str(domain_dir / "index.faiss"))
        with (domain_dir / "id_map.json").open("w", encoding="utf-8") as f:
            json.dump(id_map, f)

        print(f"[OK] Wrote {domain}: {len(id_map)} vectors -> {domain_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-domain FAISS indexes from chunks.sqlite + .npy embeddings.")
    parser.add_argument("--db_path", default="data/metadata/chunks.sqlite", help="Path to chunks.sqlite")
    parser.add_argument("--out_dir", default="vectorstore", help="Output vectorstore directory")
    parser.add_argument("--doc_id", default=None, help="Optional: rebuild using only one doc_id")
    args = parser.parse_args()
    main(args.db_path, args.out_dir, args.doc_id)