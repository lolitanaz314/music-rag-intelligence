🚀 Deployment & Access

This project is designed as a two-tier system:

Streamlit UI  →  FastAPI backend  →  RAG pipeline (FAISS + LLMs)

Local Execution (Current Setup)

The application is currently run locally:

- FastAPI backend serves /search and /answer endpoints
- Streamlit UI provides an interactive front-end
- Both services communicate over localhost
- This setup is intentional:
- avoids exposing API keys publicly
- prevents uncontrolled third-party usage
- keeps inference costs bounded
- mirrors internal tooling setups commonly used in industry

Running Locally
# Start backend
uvicorn src.api.main:app --reload --port 8000

# Start UI
python -m streamlit run ui/app.py

🌍 Public Deployment (Optional)

The system is fully deployable as a public service, but is not hosted publicly by default.

A production deployment would require:

- Hosting the FastAPI backend (e.g. Render, Fly.io, Railway, AWS)
- Hosting the Streamlit UI (e.g. Streamlit Community Cloud)
- Securely managing API keys (OPENAI_API_KEY, COHERE_API_KEY)
- Adding authentication and rate limiting to control usage and cost

This separation of concerns reflects real-world RAG system architectures, where UI and inference services are deployed independently.

🔐 Security & Cost Considerations

This system makes live LLM API calls for:

- embeddings
- query rewriting
- reranking
- answer generation

For that reason:

- API keys are never committed to the repository
- public deployment is intentionally disabled
- inference is gated behind local execution

📸 Demo Evidence

The repository includes:

- screenshots of the Streamlit UI
- example JSON responses from the /answer endpoint
- before/after retrieval vs reranking comparisons
- These artifacts demonstrate full system functionality without requiring public hosting.

🧠 Why This Matters

Many production RAG systems are not publicly exposed due to:

- proprietary document corpora
- cost constraints
- legal and security requirements
- This project mirrors that reality while still demonstrating:

- retrieval optimization
- grounded generation
- citation auditing
- API + UI integration

(Optional short line you can add near the top)

Note: This project is run locally by design. Public deployment is possible but intentionally omitted to control API usage and costs.