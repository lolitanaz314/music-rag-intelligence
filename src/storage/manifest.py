import json, time
from pathlib import Path
from typing import Dict, Any, Optional

def load_manifest_map(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns: doc_id -> manifest_record
    Latest record wins if duplicated.
    """
    m: Dict[str, Dict[str, Any]] = {}
    if not manifest_path.exists():
        return m
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id:
                m[doc_id] = obj
    return m

def get_doc_domain(manifest_path: Path, doc_id: str) -> str:
    m = load_manifest_map(manifest_path)
    return (m.get(doc_id, {}).get("domain") or "other")

def append_manifest(manifest_path: Path, record: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("created_at", int(time.time()))
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")