import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def stable_id(text: str) -> str:
    """Stable hash for chunk IDs."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24]

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)

def main(input_dir: str, output_dir: str, chunk_size: int, chunk_overlap: int, doc_id: str | None) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([p for p in in_dir.glob("*.txt") if p.is_file()])
    if doc_id:
        txt_files = [p for p in txt_files if p.stem == doc_id]

    if not txt_files:
        print(f"[WARN] No .txt files found in {in_dir} (doc_id={doc_id})")
        return

    print(f"[INFO] Chunking {len(txt_files)} docs from {in_dir} -> {out_dir}")
    for fp in tqdm(txt_files, desc="Chunking"):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        out_path = out_dir / (fp.stem + ".jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                ch = ch.strip()
                if not ch:
                    continue
                obj: Dict = {
                    "doc_id": fp.stem,
                    "source_file": fp.name,
                    "chunk_index": i,
                    "chunk_id": stable_id(f"{fp.stem}:{i}:{ch[:200]}"),
                    "text": ch,
                    "char_len": len(ch),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("[DONE] Chunking complete.")
    print(f"Chunks written to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk cleaned text documents into JSONL.")
    parser.add_argument("--input", required=True, help="Input directory of cleaned .txt files (e.g., data/text_clean)")
    parser.add_argument("--output", required=True, help="Output directory for chunk JSONL files (e.g., data/chunks)")
    parser.add_argument("--chunk_size", type=int, default=1500, help="Chunk size in characters (default: 1500)")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters (default: 200)")
    parser.add_argument("--doc_id", default=None, help="Only process a single doc_id (filename stem)")
    args = parser.parse_args()

    main(args.input, args.output, args.chunk_size, args.overlap, args.doc_id)