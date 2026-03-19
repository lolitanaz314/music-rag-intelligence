import json
import requests
import streamlit as st
import os

st.set_page_config(page_title="Music RAG Intelligence Engine", layout="wide")
st.title("Music Industry RAG Intelligence Engine")
st.caption("Ask questions over your ingested contracts / royalty docs. Answers are grounded with citations.")

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
API_BASE = st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE).rstrip("/")

if st.sidebar.button("Reset API URL"):
    st.session_state.api_base = DEFAULT_API_BASE
    st.rerun()

with st.sidebar:
    st.subheader("Retrieval settings")
    top_n = st.slider("Top-N candidates (FAISS)", 10, 200, 50, step=10)
    top_k = st.slider("Top-K after rerank", 1, 15, 5)
    rewrite = st.checkbox("Query rewrite", value=True)
    max_sources = st.slider("Max sources to pass to generator", 1, 10, 6)
    st.divider()
    st.subheader("Tips")
    st.write("- Try: **360 deal clause comparison**, **mechanical royalties US vs EU**, **sub-publishing terms**")
    st.write("- If the corpus lacks evidence, the system should say so.")

query = st.text_input(
    "Question",
    value="How do mechanical royalties differ in the US vs EU?",
    placeholder="Ask a question about royalties, contracts, licensing, A&R, etc.",
)

colA, colB = st.columns([1, 1])
with colA:
    ask = st.button("Ask", type="primary")
with colB:
    st.write("")

def call_answer_api(q: str):
    url = f"{API_BASE.rstrip('/')}/answer"
    payload = {
        "query": q,
        "top_n": top_n,
        "top_k": top_k,
        "rewrite": rewrite,
        "max_sources": max_sources,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

if ask:
    if not query.strip():
        st.warning("Type a question first.")
        st.stop()

    with st.spinner("Retrieving + reranking + generating grounded answer..."):
        try:
            out = call_answer_api(query.strip())
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(out.get("answer", ""))

    with st.expander("Retrieval details"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top-N", top_n)
        m2.metric("Top-K", top_k)
        m3.metric("Rewrite", "ON" if rewrite else "OFF")
        m4.metric("Max sources", max_sources)

        st.markdown("**Rewritten query**")
        st.code(out.get("final_query", ""), language="text")

    st.subheader("Sources")
    sources = out.get("sources", [])
    if not sources:
        st.info("No sources returned.")
    else:
        for s in sources:
            cite = s.get("cite_id", "")
            doc = s.get("doc_id", "")
            chunk_index = s.get("chunk_index", "")
            chunk_id = s.get("chunk_id", "")
            st.markdown(f"**[{cite}]** `{doc}` — chunk `{chunk_index}` — id `{chunk_id}`")

    with st.expander("Raw JSON (debug)"):
        st.code(json.dumps(out, indent=2), language="json")

st.divider()
st.caption("UI: Streamlit client → FastAPI /answer → RAG pipeline (rewrite → retrieve → rerank → grounded generation).")
import json
import requests
import streamlit as st

st.set_page_config(page_title="Music RAG Intelligence Engine", layout="wide")
st.title("Music Industry RAG Intelligence Engine")
st.caption("Ask questions over your ingested contracts / royalty docs. Answers are grounded with citations.")

import os
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
API_BASE = st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE).rstrip("/")


with st.sidebar:
    st.subheader("Retrieval settings")
    top_n = st.slider("Top-N candidates (FAISS)", 10, 200, 50, step=10)
    top_k = st.slider("Top-K after rerank", 1, 15, 5)
    rewrite = st.checkbox("Query rewrite", value=True)
    max_sources = st.slider("Max sources to pass to generator", 1, 10, 6)
    st.divider()
    st.subheader("Tips")
    st.write("- Try: **360 deal clause comparison**, **mechanical royalties US vs EU**, **sub-publishing terms**")
    st.write("- If the corpus lacks evidence, the system should say so.")

query = st.text_input(
    "Question",
    value="How do mechanical royalties differ in the US vs EU?",
    placeholder="Ask a question about royalties, contracts, licensing, A&R, etc.",
)

colA, colB = st.columns([1, 1])
with colA:
    ask = st.button("Ask", type="primary")
with colB:
    st.write("")

def call_answer_api(q: str):
    url = f"{API_BASE.rstrip('/')}/answer"
    payload = {
        "query": q,
        "top_n": top_n,
        "top_k": top_k,
        "rewrite": rewrite,
        "max_sources": max_sources,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

if ask:
    if not query.strip():
        st.warning("Type a question first.")
        st.stop()

    with st.spinner("Retrieving + reranking + generating grounded answer..."):
        try:
            out = call_answer_api(query.strip())
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(out.get("answer", ""))

    with st.expander("Retrieval details"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Top-N", top_n)
        m2.metric("Top-K", top_k)
        m3.metric("Rewrite", "ON" if rewrite else "OFF")
        m4.metric("Max sources", max_sources)

        st.markdown("**Rewritten query**")
        st.code(out.get("final_query", ""), language="text")

    st.subheader("Sources")
    sources = out.get("sources", [])
    if not sources:
        st.info("No sources returned.")
    else:
        for s in sources:
            cite = s.get("cite_id", "")
            doc = s.get("doc_id", "")
            chunk_index = s.get("chunk_index", "")
            chunk_id = s.get("chunk_id", "")
            st.markdown(f"**[{cite}]** `{doc}` — chunk `{chunk_index}` — id `{chunk_id}`")

    with st.expander("Raw JSON (debug)"):
        st.code(json.dumps(out, indent=2), language="json")

st.divider()
st.caption("UI: Streamlit client → FastAPI /answer → RAG pipeline (rewrite → retrieve → rerank → grounded generation).")