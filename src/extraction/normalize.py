from typing import Any, Optional


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    raw = str(value).strip().lower()
    if not raw:
        return None

    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()

    try:
        num = float(cleaned)
        if "%" in raw or "percent" in raw:
            return num / 100.0
        return num
    except ValueError:
        return None


def to_int(value: Any) -> Optional[int]:
    f = to_float(value)
    if f is None:
        return None
    return int(f)