from __future__ import annotations

import csv
import re
from pathlib import Path

from src.extraction.schema import CONTRACT_FEATURES
from src.extraction.normalize import normalize_text, to_bool, to_float, to_int


RAW_DIR = Path("data/parsed_text")
OUTPUT_PATH = Path("data/processed/contract_features.csv")


def load_documents(raw_dir: Path) -> list[dict]:
    docs = []

    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            docs.append({"doc_id": path.name, "text": text})

    return docs


def extract_royalty_rate(text: str):
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if match:
        return to_float(match.group(1) + "%")
    return None


def extract_advance_amount(text: str):
    match = re.search(r"\$ ?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return to_float(match.group(0))
    return None


def extract_360_clause(text: str):
    lowered = text.lower()
    if "360" in lowered or "touring rights" in lowered or "merchandising rights" in lowered:
        return True
    return None


def extract_master_ownership(text: str):
    lowered = text.lower()
    if "artist retains ownership" in lowered or "artist shall own the masters" in lowered:
        return "artist"
    if "label shall own the masters" in lowered or "label owns the masters" in lowered:
        return "label"
    return None


def extract_territory(text: str):
    lowered = text.lower()
    if "worldwide" in lowered:
        return "worldwide"
    if "united states" in lowered or "u.s." in lowered or "us only" in lowered:
        return "united states"
    if "europe" in lowered or "eu" in lowered:
        return "europe"
    return None


def extract_row(doc: dict) -> dict:
    text = doc["text"]

    row = {field: None for field in CONTRACT_FEATURES}
    row["doc_id"] = doc["doc_id"]
    row["territory"] = normalize_text(extract_territory(text))
    row["royalty_rate"] = extract_royalty_rate(text)
    row["advance_amount"] = extract_advance_amount(text)
    row["has_360_clause"] = extract_360_clause(text)
    row["master_ownership"] = normalize_text(extract_master_ownership(text))

    return row


def save_rows(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONTRACT_FEATURES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    docs = load_documents(RAW_DIR)
    rows = [extract_row(doc) for doc in docs]
    save_rows(rows, OUTPUT_PATH)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()