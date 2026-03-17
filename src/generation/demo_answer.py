from __future__ import annotations

from src.retrieval.retriever_v2 import RetrieverV2
from src.generation.answer_with_citations import AnswerGenerator

if __name__ == "__main__":
    r = RetrieverV2(reranker_kind="cohere")  # or local if stable for you
    gen = AnswerGenerator()

    q = "What is the statutory mechanical royalty rate per Spotify stream in the US in 2025?"
    out = r.search(q, top_k=5, top_n=50, rewrite=True)

    # Convert RetrievalHit objects to dicts expected by generator
    hits = []
    for h in out["hits"]:
        hits.append({
            "doc_id": h.doc_id,
            "chunk_index": h.chunk_index,
            "chunk_id": h.chunk_id,
            "text": h.text,
        })

    ans = gen.generate(user_query=q, used_query=out["final_query"], top_hits=hits)

    print("\nQUESTION:", q)
    print("RETRIEVAL QUERY:", ans.used_query)
    print("\nANSWER:\n", ans.answer)
    print("\nSOURCES:")
    for s in ans.sources:
        print(f"- [{s.cite_id}] {s.doc_id} (chunk {s.chunk_index}) {s.chunk_id}")