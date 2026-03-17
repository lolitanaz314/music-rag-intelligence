from dataclasses import dataclass

ROYALTY_KEYWORDS = {
    "royalty", "royalties", "publishing", "publisher", "songwriter",
    "mechanical", "performance", "neighboring", "sync", "license", "licensing",
    "master", "composition", "splits", "pro", "ascap", "bmi", "sesac",
    "mlc", "harry fox", "statement", "recoup", "audit", "territory",
    "isrc", "iswc", "dsp", "spotify", "apple music", "youtube", "tiktok",
    "advance", "agreement", "contract", "catalog", "collection society"
}

@dataclass
class DomainResult:
    domain: str             # "music_royalties" | "other"
    score: float            # 0..1
    reason: str

def label_domain(text: str) -> DomainResult:
    t = (text or "").lower()
    if not t.strip():
        return DomainResult(domain="other", score=0.0, reason="Empty text")

    hits = 0
    for kw in ROYALTY_KEYWORDS:
        if kw in t:
            hits += 1

    # simple scoring
    score = min(1.0, hits / 8.0)  # 8+ hits => strong
    if score >= 0.35:
        return DomainResult("music_royalties", score, f"Matched {hits} royalties keywords")
    return DomainResult("other", score, f"Only {hits} royalties keywords matched")