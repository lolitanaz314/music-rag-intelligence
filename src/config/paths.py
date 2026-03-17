from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
CHUNKS_DIR = DATA / "chunks"
EMBEDDINGS_DIR = DATA / "embeddings"
METADATA_DIR = DATA / "metadata"
METADATA_DB = METADATA_DIR / "chunks.sqlite"
MANIFEST_PATH = METADATA_DIR / "doc_manifest.jsonl"

TEXT_DIR = DATA / "text"
TEXT_CLEAN_DIR = DATA / "text_clean"
RAW_PDFS_DIR = DATA / "raw_pdfs"

VECTORSTORE_DIR = ROOT / "vectorstore"
VS_ROYALTIES = VECTORSTORE_DIR / "music_royalties"
VS_OTHER = VECTORSTORE_DIR / "other"