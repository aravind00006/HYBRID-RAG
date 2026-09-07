"""
Streamlit frontend for the HYBRID-RAG system. Single responsibility: UI only.
Pure HTTP client to the FastAPI backend — no pipeline logic here.
"""

import os

import requests
import streamlit as st

from logger import get_logger

logger = get_logger(__name__)

#  Config 
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")

REQUEST_TIMEOUT_QUERY  = 30  # seconds — LLM generation can take a moment
REQUEST_TIMEOUT_UPLOAD = 60  # seconds — embedding a full 10-K takes longer
REQUEST_TIMEOUT_HEALTH = 3   # seconds — fast liveness check


#  Page config ─

st.set_page_config(
    page_title="HYBRID-RAG",
    page_icon="🔍",
    layout="wide",
)


#  Helpers ─

def check_health() -> dict:
    """Ping the /health endpoint and return the status dict."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=REQUEST_TIMEOUT_HEALTH)
        return r.json() if r.status_code == 200 else {}
    except requests.exceptions.ConnectionError:
        logger.warning("Health check failed — API not reachable at %s.", API_BASE)
        return {}


def ask_question(question: str, top_k: int) -> dict:
    """
    Send a question to the /query endpoint and return the response dict.
    """
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": top_k},
            timeout=REQUEST_TIMEOUT_QUERY,
        )
        return r.json()
    except requests.exceptions.Timeout:
        logger.error("Query timed out after %ds.", REQUEST_TIMEOUT_QUERY)
        return {"detail": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        logger.error("Query failed — API not reachable.")
        return {"detail": "API is not reachable. Make sure the backend is running."}
# Sidebar 

with st.sidebar:
    st.title("🔍 HYBRID-RAG")
    st.caption("Hybrid BM25 + Semantic search over SEC 10-K filings.")

    st.divider()

    # Health check
    health = check_health()
    if health.get("status") == "ok":
        st.success("API connected", icon="✅")
        st.caption(f"Model: `{health.get('model', '—')}`")
    else:
        st.error("API not reachable", icon="🔴")
        st.caption("Start the backend: `uvicorn api.main:app --reload`")

    st.divider()

    # Active document
    st.subheader("Active Document")
    active_doc = st.session_state.get("active_doc", "aapl-10k-2024.pdf (default)")
    st.info(f"📄 {active_doc}")

    st.divider()

    # PDF upload
    st.subheader("Upload Your PDF")
    st.caption("Upload any 10-K to switch documents. Replaces the active document.")

    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("Load PDF", use_container_width=True, type="primary"):
            with st.spinner(f"Processing '{uploaded.name}'..."):
                result = upload_pdf(uploaded)

            if "chunks_created" in result:
                st.success(f"Loaded — {result['chunks_created']} chunks created.")
                st.session_state["active_doc"] = uploaded.name
                st.session_state["messages"]   = []  # Clear chat on new document.
                st.rerun()
            else:
                error = result.get("detail", "Upload failed.")
                st.error(error)
                logger.warning("Upload failed for '%s': %s", uploaded.name, error)

    if st.button("Reset to Default PDF", use_container_width=True):
        st.session_state["active_doc"] = "aapl-10k-2024.pdf (default)"
        st.session_state["messages"]   = []
        st.rerun()

    st.divider()

    # Retrieval settings
    st.subheader("Settings")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values retrieve more context but use more tokens.",
    )