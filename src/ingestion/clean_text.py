import argparse
from pathlib import Path

def clean(s: str) -> str:
    # minimal cleaning; replace with your current logic if you have it
    s = (s or "").replace("\r\n", "\n").strip()
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s

def main(input_dir: str, output_dir: str, doc_id: str | None):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.txt") if p.is_file()])
    if doc_id:
        files = [p for p in files if p.stem == doc_id]

    if not files:
        print(f"[WARN] No txt files found in {in_dir} (doc_id={doc_id})")
        return

    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        out = clean(raw)
        (out_dir / fp.name).write_text(out, encoding="utf-8", errors="ignore")

    print(f"[DONE] Cleaned {len(files)} files -> {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--doc_id", default=None)
    args = parser.parse_args()
    main(args.input, args.output, args.doc_id)
