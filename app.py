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


#  Helpers 

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


def upload_pdf(file) -> dict:
    """
    Upload a PDF to the /upload endpoint and return the response dict.

    """
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=REQUEST_TIMEOUT_UPLOAD,
        )
        return r.json()
    except requests.exceptions.Timeout:
        logger.error("Upload timed out after %ds.", REQUEST_TIMEOUT_UPLOAD)
        return {"detail": "Upload timed out. Try a smaller PDF."}
    except requests.exceptions.ConnectionError:
        logger.error("Upload failed — API not reachable.")
        return {"detail": "API is not reachable. Make sure the backend is running."}


#  Sidebar 

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


#  Main chat area 

st.title("Ask a Question")
st.caption(
    "Answers are grounded in the active document using Hybrid BM25 + Semantic search. "
    "Every claim is cited with a source filename and page number."
)

# Initialise chat history.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render existing messages.
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — Page {s['page']}")
                    st.caption(s["preview"])

# Chat input.
if question := st.chat_input("e.g. What was Apple's revenue in fiscal year 2024?"):

    if not health:
        st.error("Backend is not reachable. Start the API first.")
        st.stop()

    # Render user message immediately.
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call backend and render answer.
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            response = ask_question(question, top_k)

        if "answer" in response:
            st.markdown(response["answer"])

            with st.expander("Sources"):
                for s in response.get("sources", []):
                    st.markdown(f"**{s['source']}** — Page {s['page']}")
                    st.caption(s["preview"])

            st.caption(
                f"Model: `{response['model']}` · "
                f"Tokens: `{response['tokens_used']}` · "
                f"Chunks retrieved: `{top_k}`"
            )

            st.session_state["messages"].append({
                "role":    "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
            })

        else:
            error_msg = response.get("detail", "Something went wrong.")
            st.error(error_msg)
            logger.error("Query error for question '%s': %s", question[:60], error_msg)