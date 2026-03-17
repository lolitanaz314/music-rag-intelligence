import time
from pathlib import Path

import streamlit as st

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document
except Exception:
    Document = None

from src.config.paths import TEXT_DIR, MANIFEST_PATH
from src.policies.domain_labeler import label_domain
from src.storage.manifest import append_manifest


def extract_pdf(file) -> str:
    if PdfReader is None:
        st.error("PDF support not installed. Run: python -m pip install pypdf")
        return ""
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def extract_docx(file) -> str:
    if Document is None:
        st.error("DOCX support not installed. Run: python -m pip install python-docx")
        return ""
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


st.set_page_config(page_title="Ingest", layout="wide")
st.title("Ingest")

title = st.text_input("Title", value="")
tags = st.text_input("Tags (comma-separated)", value="")

uploaded_file = st.file_uploader(
    "Upload a document to ingest (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
)

# Upload-only ingestion
text = ""
source_type = "upload"
source_ref = None

if uploaded_file:
    source_ref = uploaded_file.name

    if uploaded_file.type == "application/pdf":
        text = extract_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_docx(uploaded_file)

    if not text.strip():
        st.warning("⚠️ No extractable text found in this file (it may be scanned/image-only).")
    else:
        with st.expander("Preview extracted text", expanded=False):
            st.text_area(
                "Extracted text (read-only)",
                value=text[:15000],
                height=220,
                disabled=True,
            )
else:
    st.info("Upload a document to continue.")

domain_guess = label_domain(text) if text.strip() else None
if domain_guess:
    st.write(
        f"Suggested domain: **{domain_guess.domain}** (score={domain_guess.score:.2f}) — {domain_guess.reason}"
    )

domain = st.selectbox(
    "Domain (override if needed)",
    ["music_royalties", "other"],
    index=0 if (domain_guess and domain_guess.domain == "music_royalties") else 1,
)

TEXT = Path(TEXT_DIR)
MANIFEST = Path(MANIFEST_PATH)

# Require an upload AND successful text extraction
save_disabled = (uploaded_file is None) or (not text.strip())

if st.button("Save document", type="primary", disabled=save_disabled):
    doc_id = f"doc_{int(time.time() * 1000)}"

    TEXT.mkdir(parents=True, exist_ok=True)
    out_txt = TEXT / f"{doc_id}.txt"
    out_txt.write_text(text, encoding="utf-8", errors="ignore")

    append_manifest(
        MANIFEST,
        {
            "doc_id": doc_id,
            "domain": domain,
            "title": title.strip() or doc_id,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "source_type": source_type,          # always "upload"
            "source_ref": source_ref or str(out_txt),  # filename if available
        },
    )

    st.success(f"Saved {doc_id}. Next: run the pipeline for this doc_id: `{doc_id}`")
    st.code(
        f"""
python -m src.ingestion.clean_text --input data/text --output data/text_clean --doc_id {doc_id}
python -m src.ingestion.chunk_documents --input data/text_clean --output data/chunks --doc_id {doc_id}
python -m src.ingestion.embed_chunks --chunks_dir data/chunks --embeddings_dir data/embeddings --db_path data/metadata/chunks.sqlite --manifest_path data/metadata/doc_manifest.jsonl --doc_id {doc_id}
python -m src.vectorstore.build_faiss --db_path data/metadata/chunks.sqlite --out_dir vectorstore
""".strip()
    )