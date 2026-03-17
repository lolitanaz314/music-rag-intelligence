import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import faiss
from sqlalchemy import create_engine, text as sql

from src.config.paths import VECTORSTORE_DIR, METADATA_DB

def load_index(domain: str) -> tuple[faiss.Index, List[str]]:
    ddir = VECTORSTORE_DIR / domain
    ip = ddir / "index.faiss"
    mp = ddir / "id_map.json"
    if not ip.exists() or not mp.exists():
        return None, []
    idx = faiss.read_index(str(ip))
    id_map = json.loads(mp.read_text(encoding="utf-8"))
    return idx, id_map

def fetch_chunk_texts(chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids:
        return {}
    engine = create_engine(f"sqlite:///{METADATA_DB}")
    qmarks = ",".join("?" for _ in chunk_ids)
    # We don't store chunk text in chunks.sqlite; it's in chunk JSONLs.
    # So we read from chunk_path JSONL quickly per chunk_id.
    # For speed later: add a separate docstore. For now, simple lookup.
    with engine.begin() as conn:
        rows = conn.execute(sql(f"""
          SELECT chunk_id, chunk_path
          FROM chunks
          WHERE chunk_id IN ({qmarks})
        """), chunk_ids).fetchall()

    # Map chunk_id -> chunk_path
    cid_to_path = {cid: Path(p) for cid, p in rows}

    # Read files and extract chunk texts
    out = {}
    # group by chunk_path to avoid re-reading file many times
    by_path: Dict[Path, List[str]] = {}
    for cid, p in cid_to_path.items():
        by_path.setdefault(p, []).append(cid)

    for p, cids in by_path.items():
        if not p.exists():
            continue
        # chunk jsonl lines contain chunk_id and text
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid in cids:
                    out[cid] = obj.get("text", "")
    return out

def search_domain(query_vec: np.ndarray, domain: str, top_k: int = 6) -> List[Tuple[str, float]]:
    idx, id_map = load_index(domain)
    if idx is None:
        return []
    q = query_vec.astype("float32").reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = idx.search(q, top_k)
    hits = []
    for score, i in zip(D[0].tolist(), I[0].tolist()):
        if i == -1:
            continue
        hits.append((id_map[i], float(score)))
    return hits