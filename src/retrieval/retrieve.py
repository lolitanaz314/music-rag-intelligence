import os, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sqlalchemy import create_engine, text as sql

from src.config.paths import VECTORSTORE_DIR, METADATA_DB

def embed_query_openai(query: str, model: str = "text-embedding-3-large") -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=[query])
    return np.array(resp.data[0].embedding, dtype="float32")

def load_index(domain: str) -> Tuple[faiss.Index | None, List[str]]:
    ddir = VECTORSTORE / domain
    ip = ddir / "index.faiss"
    mp = ddir / "id_map.json"
    if not ip.exists() or not mp.exists():
        return None, []
    idx = faiss.read_index(str(ip))
    id_map = json.loads(mp.read_text(encoding="utf-8"))
    return idx, id_map

def search_domain(qvec: np.ndarray, domain: str, top_k: int) -> List[Tuple[str, float]]:
    idx, id_map = load_index(domain)
    if idx is None:
        return []
    q = qvec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = idx.search(q, top_k)
    hits = []
    for score, i in zip(D[0].tolist(), I[0].tolist()):
        if i == -1:
            continue
        hits.append((id_map[i], float(score)))
    return hits

def fetch_chunk_texts(chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids:
        return {}

    engine = create_engine(f"sqlite:///{CHUNKS_DB}")
    qmarks = ",".join("?" for _ in chunk_ids)

    with engine.begin() as conn:
        rows = conn.execute(sql(f"""
          SELECT chunk_id, chunk_path
          FROM chunks
          WHERE chunk_id IN ({qmarks})
        """), chunk_ids).fetchall()

    cid_to_path = {cid: Path(p) for cid, p in rows}

    by_path: Dict[Path, List[str]] = {}
    for cid, p in cid_to_path.items():
        by_path.setdefault(p, []).append(cid)

    out: Dict[str, str] = {}
    for p, cids in by_path.items():
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid in cids:
                    out[cid] = obj.get("text", "")
    return out

def retrieve(query: str, domain: str = "music_royalties", top_k: int = 6) -> List[Dict[str, Any]]:
    qvec = embed_query_openai(query)
    hits = search_domain(qvec, domain, top_k)
    texts = fetch_chunk_texts([cid for cid, _ in hits])
    return [{"chunk_id": cid, "score": score, "text": texts.get(cid, "")} for cid, score in hits]

def retrieve_all(query: str, top_k_each: int = 4) -> List[Dict[str, Any]]:
    a = retrieve(query, "music_royalties", top_k_each)
    b = retrieve(query, "other", top_k_each)
    merged = a + b
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged
