import re
import sys
import time
from pathlib import Path

import streamlit as st

try:
    from pypdf import PdfReader
    PDF_IMPORT_ERROR = None
except Exception as e:
    PdfReader = None
    PDF_IMPORT_ERROR = str(e)

try:
    from docx import Document
    DOCX_IMPORT_ERROR = None
except Exception as e:
    Document = None
    DOCX_IMPORT_ERROR = str(e)

from src.config.paths import TEXT_DIR, MANIFEST_PATH
from src.policies.domain_labeler import label_domain
from src.storage.manifest import append_manifest


# ----------------------------
# Config
# ----------------------------
MIN_DOMAIN_SCORE = 0.60

MUSIC_TERMS = [
    "music",
    "artist",
    "recording artist",
    "songwriter",
    "composer",
    "publisher",
    "publishing",
    "record label",
    "label",
    "master recording",
    "sound recording",
    "composition",
    "track",
    "album",
    "single",
    "catalog",
    "music industry",
    "performing rights organization",
    "bmi",
    "ascap",
    "sesac",
    "harry fox",
    "cue sheet",
]

ROYALTY_TERMS = [
    "royalty",
    "royalties",
    "mechanical royalty",
    "performance royalty",
    "neighboring rights",
    "publishing income",
    "writer share",
    "publisher share",
    "royalty rate",
    "royalty statement",
    "statement period",
    "recoupment",
    "advance",
    "net receipts",
    "gross receipts",
    "deductions",
    "income",
    "synchronization fee",
    "sync fee",
]

CONTRACT_TERMS = [
    "agreement",
    "contract",
    "license",
    "licensing",
    "grant of rights",
    "rights",
    "copyright",
    "ownership",
    "term",
    "territory",
    "exclusivity",
    "assignment",
    "warranty",
    "indemnification",
    "breach",
    "termination",
    "work made for hire",
    "delivery requirements",
    "distribution agreement",
    "artist agreement",
    "publishing agreement",
    "producer agreement",
    "recording agreement",
    "music publishing agreement",
]

# ----------------------------
# Extraction helpers
# ----------------------------
def extract_pdf(file) -> str:
    if PdfReader is None:
        st.error(f"PDF support not installed. Import error: {PDF_IMPORT_ERROR}")
        return ""
    try:
        reader = PdfReader(file)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""


def extract_docx(file) -> str:
    if Document is None:
        st.error(f"DOCX support not installed. Import error: {DOCX_IMPORT_ERROR}")
        return ""
    try:
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"Failed to read DOCX: {e}")
        return ""


def extract_txt(file) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Failed to read TXT: {e}")
        return ""

# ----------------------------
# Filtering helpers
# ----------------------------
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def phrase_present(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
    return re.search(pattern, text) is not None


def get_matches(text: str, keywords: list[str]) -> list[str]:
    normalized = normalize_text(text)
    return [kw for kw in keywords if phrase_present(normalized, kw)]


def split_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks = []
    start = 0
    n = len(normalized)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(normalized[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def get_domain_evidence(text: str) -> dict:
    music_matches = get_matches(text, MUSIC_TERMS)
    royalty_matches = get_matches(text, ROYALTY_TERMS)
    contract_matches = get_matches(text, CONTRACT_TERMS)

    chunks = split_into_chunks(text)
    chunk_evidence = []

    for idx, chunk in enumerate(chunks):
        chunk_music = get_matches(chunk, MUSIC_TERMS)
        chunk_royalty = get_matches(chunk, ROYALTY_TERMS)
        chunk_contract = get_matches(chunk, CONTRACT_TERMS)

        total = len(chunk_music) + len(chunk_royalty) + len(chunk_contract)

        chunk_evidence.append(
            {
                "chunk_index": idx,
                "music_matches": chunk_music,
                "royalty_matches": chunk_royalty,
                "contract_matches": chunk_contract,
                "total_matches": total,
                "is_in_domain_chunk": (
                    len(chunk_music) >= 1
                    and len(chunk_royalty) >= 1
                    and len(chunk_contract) >= 1
                    and total >= 4
                ),
            }
        )

    in_domain_chunks = [c for c in chunk_evidence if c["is_in_domain_chunk"]]

    return {
        "music_matches": music_matches,
        "royalty_matches": royalty_matches,
        "contract_matches": contract_matches,
        "chunk_evidence": chunk_evidence,
        "in_domain_chunk_count": len(in_domain_chunks),
        "total_chunk_count": len(chunk_evidence),
    }


def is_music_royalty_contract_doc(evidence: dict) -> tuple[bool, str]:
    music_count = len(evidence["music_matches"])
    royalty_count = len(evidence["royalty_matches"])
    contract_count = len(evidence["contract_matches"])
    total_count = music_count + royalty_count + contract_count

    in_domain_chunk_count = evidence.get("in_domain_chunk_count", 0)
    total_chunk_count = evidence.get("total_chunk_count", 0)

    # Strong document-level evidence
    if music_count < 2:
        return False, f"Rejected: only {music_count} music-related matches found."

    if royalty_count < 2:
        return False, f"Rejected: only {royalty_count} royalty-related matches found."

    if contract_count < 2:
        return False, f"Rejected: only {contract_count} contract-related matches found."

    if total_count < 7:
        return False, f"Rejected: only {total_count} total in-domain matches found."

    # Coverage rule: must be about the document overall, not one tiny section
    if in_domain_chunk_count < 2:
        return False, (
            "Rejected: document has too little in-domain coverage "
            f"({in_domain_chunk_count} matching chunk(s))."
        )

    # Optional extra guard for longer docs
    if total_chunk_count >= 4:
        coverage_ratio = in_domain_chunk_count / total_chunk_count
        if coverage_ratio < 0.30:
            return False, (
                "Rejected: in-domain content is too sparse across the document "
                f"({in_domain_chunk_count}/{total_chunk_count} chunks)."
            )

    return True, ""


def evaluate_document_gate(text: str):
    """
    Returns:
        reject_doc: bool
        rejection_reason: str
        domain_guess: any
        evidence: dict
    """
    if not text.strip():
        return True, "No extractable text found.", None, {}

    evidence = get_domain_evidence(text)
    in_domain, reason = is_music_royalty_contract_doc(evidence)

    domain_guess = label_domain(text)

    if not in_domain:
        return True, reason, domain_guess, evidence

    if domain_guess is None:
        return True, "Could not determine document domain.", None, evidence

    if domain_guess.domain != "music_royalties" and domain_guess.score >= MIN_DOMAIN_SCORE:
        return True, (
            f"Rejected: classified as `{domain_guess.domain}`, "
            f"not `music_royalties`."
        ), domain_guess, evidence

    return False, "", domain_guess, evidence

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Ingest", layout="wide")
st.title("Ingest")

with st.expander("Runtime debug", expanded=False):
    st.caption(f"Python executable: {sys.executable}")
    if Document is None:
        st.caption(f"DOCX import failed: {DOCX_IMPORT_ERROR}")
    if PdfReader is None:
        st.caption(f"PDF import failed: {PDF_IMPORT_ERROR}")

title = st.text_input("Title", value="")
tags = st.text_input("Tags (comma-separated)", value="")

uploaded_file = st.file_uploader(
    "Upload a document to ingest (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
)

text = ""
source_type = "upload"
source_ref = None
extraction_failed = False

reject_doc = False
rejection_reason = ""
domain_guess = None
evidence = {"music_matches": [], "royalty_matches": [], "contract_matches": []}

if uploaded_file:
    source_ref = uploaded_file.name
    suffix = Path(uploaded_file.name).suffix.lower()

    if uploaded_file.type == "application/pdf" or suffix == ".pdf":
        text = extract_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain" or suffix == ".txt":
        text = extract_txt(uploaded_file)
    elif (
        uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or suffix == ".docx"
    ):
        text = extract_docx(uploaded_file)
    else:
        extraction_failed = True
        st.error(f"Unsupported file type: {uploaded_file.type} ({suffix})")

    if not extraction_failed:
        reject_doc, rejection_reason, domain_guess, evidence = evaluate_document_gate(text)

        if text.strip():
            with st.expander("Preview extracted text", expanded=False):
                st.text_area(
                    "Extracted text (read-only)",
                    value=text[:15000],
                    height=220,
                    disabled=True,
                    key="preview_extracted_text",
                )

        if domain_guess:
            st.write(
                f"Suggested domain: **{domain_guess.domain}** "
                f"(score={domain_guess.score:.2f}) — {domain_guess.reason}"
            )

        music_matches = evidence.get("music_matches", [])
        royalty_matches = evidence.get("royalty_matches", [])
        contract_matches = evidence.get("contract_matches", [])

        st.write(f"Music matches: **{len(music_matches)}**")
        st.write(f"Royalty matches: **{len(royalty_matches)}**")
        st.write(f"Contract matches: **{len(contract_matches)}**")

        with st.expander("Matched keywords", expanded=False):
            st.write("**Music:**", music_matches if music_matches else [])
            st.write("**Royalty:**", royalty_matches if royalty_matches else [])
            st.write("**Contract:**", contract_matches if contract_matches else [])

        if reject_doc:
            st.error(
                f"❌ Document rejected.\n\nReason: {rejection_reason}\n\n"
                f"This file will not be ingested unless you override."
            )
        else:
            st.success("✅ Document passed ingestion filter.")
else:
    st.info("Upload a document to continue.")

override_reject = False
if uploaded_file and reject_doc:
    override_reject = st.checkbox(
        "Force ingest anyway (override filter)",
        value=False,
        key="force_ingest_override",
    )

effective_reject = reject_doc and not override_reject

default_domain_index = 0 if (domain_guess and domain_guess.domain == "music_royalties") else 1
domain = st.selectbox(
    "Domain (override if needed)",
    ["music_royalties", "other"],
    index=default_domain_index,
)

TEXT = Path(TEXT_DIR)
MANIFEST = Path(MANIFEST_PATH)

save_disabled = (
    (uploaded_file is None)
    or extraction_failed
    or (not text.strip())
    or effective_reject
)

if st.button("Save document", type="primary", disabled=save_disabled):
    doc_id = f"doc_{int(time.time() * 1000)}"

    TEXT.mkdir(parents=True, exist_ok=True)
    out_txt = TEXT / f"{doc_id}.txt"
    out_txt.write_text(text, encoding="utf-8", errors="ignore")

    music_matches = evidence.get("music_matches", [])
    royalty_matches = evidence.get("royalty_matches", [])
    contract_matches = evidence.get("contract_matches", [])

    append_manifest(
        MANIFEST,
        {
            "doc_id": doc_id,
            "domain": domain,
            "title": title.strip() or doc_id,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "source_type": source_type,
            "source_ref": source_ref or str(out_txt),
            "domain_score": float(domain_guess.score) if domain_guess else None,
            "domain_reason": domain_guess.reason if domain_guess else None,
            "music_match_count": len(music_matches),
            "royalty_match_count": len(royalty_matches),
            "contract_match_count": len(contract_matches),
            "music_matches": music_matches,
            "royalty_matches": royalty_matches,
            "contract_matches": contract_matches,
            "was_rejected_by_gate": reject_doc,
            "override_used": override_reject,
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