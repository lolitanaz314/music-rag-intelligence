from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from repo root
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    notes: str


class QueryRewriter:
    """
    Rewrites user queries into retrieval-optimized queries.
    Guarantees plain-text output (no JSON, no code fences).
    """

    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def rewrite(self, user_query: str, context: Optional[str] = None) -> RewriteResult:
        system = (
            "You rewrite user questions into short, retrieval-optimized search queries.\n"
            "Rules:\n"
            "- Preserve intent.\n"
            "- Add domain-specific keywords likely to appear in contracts or legal docs.\n"
            "- Remove conversational language.\n"
            "- Output JSON with keys: rewritten, notes.\n"
            "- Do NOT wrap output in markdown or code fences.\n"
        )

        user = f"User query: {user_query}\n"
        if context:
            user += f"Context: {context}\n"

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        text = resp.output_text.strip()

        # ---- HARDENED PARSING ----

        # Remove ```json ``` or ``` ``` if model ignores instructions
        m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if m:
            text = m.group(1).strip()

        rewritten = user_query
        notes = ""

        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                rewritten = obj.get("rewritten", user_query).strip()
                notes = obj.get("notes", "").strip()
            except Exception:
                rewritten = user_query
        else:
            # Fallback: treat entire output as rewritten query
            rewritten = text.strip() or user_query

        # Safety guard
        if not rewritten:
            rewritten = user_query

        return RewriteResult(
            original=user_query,
            rewritten=rewritten,
            notes=notes,
        )


if __name__ == "__main__":
    qr = QueryRewriter()
    q = "How do mechanical royalties differ in the US vs EU?"
    out = qr.rewrite(q)
    print("ORIGINAL:", out.original)
    print("REWRITTEN:", out.rewritten)
    print("NOTES:", out.notes)