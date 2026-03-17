import re
from dataclasses import dataclass

@dataclass
class GroundingCheck:
    ok: bool
    reason: str

def has_citations(text: str) -> GroundingCheck:
    # expects citations like [S1]
    if re.search(r"\[S\d+\]", text):
        return GroundingCheck(True, "Citations present.")
    return GroundingCheck(False, "No inline citations like [S1] found.")

def minimal_length(text: str, min_chars: int = 80) -> GroundingCheck:
    if len(text.strip()) >= min_chars:
        return GroundingCheck(True, "Answer length ok.")
    return GroundingCheck(False, "Answer too short to be meaningful.")