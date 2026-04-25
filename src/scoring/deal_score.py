def score_contract(features: dict) -> dict:
    score = 100
    reasons = []

    royalty_rate = features.get("royalty_rate")
    if royalty_rate is not None and royalty_rate < 0.15:
        score -= 20
        reasons.append("Low royalty rate")

    if features.get("recoupable") is True:
        score -= 10
        reasons.append("Advance is recoupable")

    term_years = features.get("term_years")
    if term_years is not None and term_years > 5:
        score -= 15
        reasons.append("Long contract term")

    ownership = features.get("ownership") or ""
    if "label" in ownership.lower() and "master" in ownership.lower():
        score -= 20
        reasons.append("Label appears to control masters")

    if features.get("audit_rights") is False:
        score -= 10
        reasons.append("No clear audit rights")

    if features.get("exclusivity") is True:
        score -= 10
        reasons.append("Exclusivity restriction")

    score = max(score, 0)

    if score >= 75:
        risk_level = "Low"
    elif score >= 50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "artist_friendliness_score": score,
        "risk_level": risk_level,
        "score_reasons": reasons,
    }