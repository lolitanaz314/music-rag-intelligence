import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
import sqlite3


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def init_merged_db(db_path: Path) -> None:
    ensure_dir(db_path.parent)
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        row_id INTEGER PRIMARY KEY,
        doc_id TEXT NOT NULL,
        source_file TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_id TEXT NOT NULL,
        char_len INTEGER NOT NULL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id);")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_chunk_id ON chunks(chunk_id);")
    con.commit()
    con.close()


def insert_rows(db_path: Path, rows: List[Tuple]) -> None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.executemany(
        "INSERT OR IGNORE INTO chunks(row_id, doc_id, source_file, chunk_index, chunk_id, char_len) VALUES(?,?,?,?,?,?)",
        rows
    )
    con.commit()
    con.close()


def main(chunks_dir: str, embeddings_dir: str, out_dir: str) -> None:
    chunks_dir = Path(chunks_dir)
    embeddings_dir = Path(embeddings_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    merged_chunks_path = out_dir / "all_chunks.jsonl"
    merged_embeddings_path = out_dir / "all_embeddings.npy"
    merged_db_path = out_dir / "all_chunks.sqlite"

    init_merged_db(merged_db_path)

    chunk_files = sorted(glob.glob(str(chunks_dir / "*.jsonl")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk JSONL files found in {chunks_dir}")

    all_embeddings = []
    row_id = 0

    # Write merged chunks as we go (streaming)
    with merged_chunks_path.open("w", encoding="utf-8") as out_f:
        for cf in tqdm(chunk_files, desc="Merging"):
            doc_stem = Path(cf).stem
            emb_file = embeddings_dir / f"{doc_stem}.npy"
            if not emb_file.exists():
                print(f"[WARN] Skipping {doc_stem} — embeddings not found")
                continue

            chunks = read_jsonl(cf)
            vecs = np.load(str(emb_file))

            if len(chunks) != vecs.shape[0]:
                raise ValueError(
                    f"Chunk/embedding count mismatch for {doc_stem}: "
                    f"{len(chunks)} chunks vs {vecs.shape[0]} embeddings"
                )

            # Append embeddings
            all_embeddings.append(vecs.astype(np.float32, copy=False))

            # Insert metadata rows + write merged JSONL
            rows_for_db = []
            for i, ch in enumerate(chunks):
                # Force a stable global row id aligned to embeddings row order
                ch_out = {
                    "row_id": row_id,
                    "doc_id": ch["doc_id"],
                    "source_file": ch["source_file"],
                    "chunk_index": int(ch["chunk_index"]),
                    "chunk_id": ch["chunk_id"],
                    "char_len": int(ch["char_len"]),
                    "text": ch["text"],
                }
                out_f.write(json.dumps(ch_out, ensure_ascii=False) + "\n")

                rows_for_db.append((
                    row_id,
                    ch_out["doc_id"],
                    ch_out["source_file"],
                    ch_out["chunk_index"],
                    ch_out["chunk_id"],
                    ch_out["char_len"],
                ))

                row_id += 1

            insert_rows(merged_db_path, rows_for_db)

    # Concatenate and save one big matrix
    merged = np.vstack(all_embeddings).astype(np.float32, copy=False)
    np.save(str(merged_embeddings_path), merged)

    print("\n[DONE] Merge complete.")
    print("Merged chunks JSONL:", merged_chunks_path)
    print("Merged embeddings NPY:", merged_embeddings_path)
    print("Merged metadata SQLite:", merged_db_path)
    print("Embeddings shape:", merged.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-doc chunk JSONL + embeddings into one global corpus.")
    parser.add_argument("--chunks_dir", default="data/chunks", help="Directory with per-doc chunk JSONL files")
    parser.add_argument("--embeddings_dir", default="data/embeddings", help="Directory with per-doc .npy embedding files")
    parser.add_argument("--out_dir", default="data/merged", help="Output directory for merged artifacts")
    args = parser.parse_args()

    main(args.chunks_dir, args.embeddings_dir, args.out_dir)