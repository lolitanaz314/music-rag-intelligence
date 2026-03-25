from pathlib import Path

import fitz  # PyMuPDF


PDF_DIR = Path("data/raw_pdfs")
TEXT_DIR = Path("data/parsed_text")


def pdf_to_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return "\n".join(pages)


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        text = pdf_to_text(pdf_path)
        out_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()