import os
from pdfminer.high_level import extract_text  # pyright: ignore[reportMissingImports]
import argparse

def convert_pdf_to_text(pdf_path, output_dir):
    text = extract_text(pdf_path)
    base = os.path.basename(pdf_path).replace(".pdf", ".txt")
    output_path = os.path.join(output_dir, base)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def main(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            convert_pdf_to_text(
                os.path.join(input_dir, filename), 
                output_dir
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)