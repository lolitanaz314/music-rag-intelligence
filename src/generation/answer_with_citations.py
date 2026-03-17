from __future__ import annotations
from src.generation.grounding_checks import has_citations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

import re
def cited_source_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[(S\d+)\]", answer))

@dataclass
class Source:
    cite_id: str          # e.g. S1
    doc_id: str
    chunk_index: int
    chunk_id: str
    text: str


@dataclass
class AnswerResult:
    answer: str
    sources: List[Source]
    used_query: str


class AnswerGenerator:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in .env.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, user_query: str, used_query: str, top_hits: List[dict], max_sources: int = 6) -> AnswerResult:
        # Build sources list (limit)
        sources: List[Source] = []
        for i, h in enumerate(top_hits[:max_sources], start=1):
            sources.append(
                Source(
                    cite_id=f"S{i}",
                    doc_id=h["doc_id"],
                    chunk_index=int(h["chunk_index"]),
                    chunk_id=h["chunk_id"],
                    text=h["text"],
                )
            )

        # Evidence block fed to LLM
        evidence_lines = []
        for s in sources:
            evidence_lines.append(f"[{s.cite_id}] doc={s.doc_id} chunk={s.chunk_index} id={s.chunk_id}\n{s.text}")
        evidence = "\n\n".join(evidence_lines)

        system = (
            "You answer questions using ONLY the provided sources.\n"
            "Rules:\n"
            "- Every factual claim must have an inline citation like [S1].\n"
            "- Do NOT cite multiple sources in one bracket; cite like [S1][S4].\n"
            "- If sources lack key details (e.g., explicit rates), say 'Not found in sources.'\n"
            "- Include a final section 'Evidence excerpts' with 1–3 short quotes (<=25 words each) from the sources.\n"
            "- Do NOT add external knowledge.\n"
        )

        prompt = (
            f"User question: {user_query}\n"
            f"Retrieval query used: {used_query}\n\n"
            f"SOURCES:\n{evidence}\n\n"
            "Write the best answer you can, citing sources."
        )

        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )

        ans_text = resp.output_text.strip()
        ans_text = re.sub(r"\[\s*(S\d+)\s*\]", r"[\1]", ans_text)

        from src.generation.grounding_checks import has_citations
        check = has_citations(ans_text)
        if not check.ok:
            ans_text = (
                "I don’t have enough grounded evidence in the provided documents to answer confidently. "
                "Try asking a narrower question or ingest more relevant documents."
            )

        # Auto-trim sources to only cited ones to prevent citation-audit fail.
        cited_ids = set(re.findall(r"\[(S\d+)\]", ans_text))
        sources = [s for s in sources if s.cite_id in cited_ids]

        return AnswerResult(answer=ans_text, sources=sources, used_query=used_query)

