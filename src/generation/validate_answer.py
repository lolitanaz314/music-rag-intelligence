import re
from typing import List, Dict

def extract_citations(answer: str) -> List[str]:
    return re.findall(r"\[(S\d+)\]", answer)

def validate(answer: str, provided_sources: List[str]) -> Dict[str, object]:
    cited = extract_citations(answer)
    cited_set = set(cited)
    provided_set = set(provided_sources)

    errors = []

    if not cited:
        errors.append("No citations found (must include at least one [S#]).")

    unknown = cited_set - provided_set
    if unknown:
        errors.append(f"Cited unknown sources: {sorted(unknown)}")

    unused = provided_set - cited_set
    if unused:
        errors.append(f"Provided but unused sources (should be removed): {sorted(unused)}")

    if re.search(r"\[S\d+\s*,\s*S\d+\]", answer):
        errors.append("Combined citation format found. Use [S1][S2], not [S1, S2].")

    return {
        "ok": len(errors) == 0,
        "cited": cited,
        "errors": errors,
    }


if __name__ == "__main__":
    # ⬇️ Paste your full answer text between the triple quotes
    answer = """
PASTE YOUR ANSWER HERE
"""

    provided_sources = ["S1", "S2", "S3", "S4", "S5"]

    report = validate(answer, provided_sources)

    print("OK:", report["ok"])
    print("CITED:", report["cited"])
    print("ERRORS:")
    for e in report["errors"]:
        print("-", e)