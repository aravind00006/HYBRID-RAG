"""
Streamlit frontend for the HYBRID-RAG system. Single responsibility: UI only.
Streamlit Community Cloud version — calls the pipeline directly instead of
calling FastAPI over HTTP. 
"""

from pathlib import Path

import streamlit as st

from config import get_settings
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, load_store
from ingestion.loader import load_pdf
from logger import get_logger
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid import hybrid_retrieve
from retrieval.semantic_retriever import SemanticRetriever
from generation.generator import generate_answer

logger = get_logger(__name__)


# Pipeline startup — runs ONCE on server boot, cached forever.

@st.cache_resource(show_spinner="Loading pipeline... this takes ~30s on first run.")
def load_pipeline() -> tuple:
    """
    Load the default PDF and build both retrievers.
    """
    settings = get_settings()
    logger.info("Pipeline startup — loading default document.")

    pages  = load_pdf(settings.default_pdf_path)
    chunks = chunk_documents(pages)

    chroma_path = settings.chroma_path
    if Path(chroma_path).exists() and any(Path(chroma_path).iterdir()):
        logger.info("ChromaDB found at '%s' — loading from disk.", chroma_path)
        store = load_store()
    else:
        logger.info("ChromaDB not found — embedding %d chunks.", len(chunks))
        store = embed_and_store(chunks)

    bm25 = BM25Retriever(chunks)

    logger.info("Pipeline ready — %d chunks, retrievers built.", len(chunks))
    return chunks, store, bm25


def run_query(question: str, top_k: int) -> dict:
    """
    Run the full RAG pipeline for a question.

    """
    chunks, store, bm25 = load_pipeline()

    # If the user uploaded a new document this session, use that instead.
    if st.session_state.get("session_chunks") is not None:
        chunks = st.session_state["session_chunks"]
        store  = st.session_state["session_store"]
        bm25   = st.session_state["session_bm25"]

    try:
        semantic = SemanticRetriever(store)
        results  = hybrid_retrieve(
            query    = question,
            bm25     = bm25,
            semantic = semantic,
            top_k    = top_k,
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return {"detail": f"Retrieval failed: {exc}"}

    try:
        response = generate_answer(question, results)
    except Exception as exc:
        logger.error("Generation failed: %s", exc)
        return {"detail": f"Generation failed: {exc}"}

    logger.info(
        "Query complete — tokens: %d, sources: %d.",
        response.tokens_used,
        len(response.sources),
    )

    return {
        "answer":      response.answer,
        "sources":     response.sources,
        "model":       response.model,
        "tokens_used": response.tokens_used,
    }


def process_upload(uploaded_file) -> dict:
    """
    Embed an uploaded PDF and store the retrievers in session_state.
    """
    # Path traversal fix — same as routes.py used Path(file.filename).name
    safe_name = Path(uploaded_file.name).name

    if not safe_name.endswith(".pdf"):
        return {"detail": "Only PDF files are supported."}

    logger.info("Upload received: '%s'.", safe_name)

    try:
        # Write the uploaded bytes to a temp file so load_pdf() gets a path.
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        pages  = load_pdf(tmp_path)
        chunks = chunk_documents(pages)
        store  = embed_and_store(
            chunks,
            chroma_path=f"./data/chroma_uploads/{safe_name}",
        )
        bm25   = BM25Retriever(chunks)

    except Exception as exc:
        logger.error("Upload processing failed for '%s': %s", safe_name, exc)
        return {"detail": f"Processing failed: {exc}"}
    finally:
        tmp_path.unlink(missing_ok=True)  # Always clean up the temp file.

    st.session_state["session_chunks"] = chunks
    st.session_state["session_store"]  = store
    st.session_state["session_bm25"]   = bm25

    logger.info(
        "Upload complete — '%s', %d chunks embedded, BM25 rebuilt.",
        safe_name,
        len(chunks),
    )

    return {"chunks_created": len(chunks)}


# Page config — must be the first Streamlit call in the script.


st.set_page_config(
    page_title="HYBRID-RAG",
    page_icon="🔍",
    layout="wide",
)


# Sidebar — identical layout to the original app.py.


with st.sidebar:
    st.title("🔍 HYBRID-RAG")
    st.caption("Hybrid BM25 + Semantic search over SEC 10-K filings.")

    st.divider()

    # Pipeline status — replaces the HTTP health check.
    try:
        settings = get_settings()
        load_pipeline()  # No-op if already cached.
        st.success("Pipeline ready", icon="✅")
        st.caption(f"Model: `{settings.llm_model}`")
    except Exception as e:
        st.error("Pipeline failed to load", icon="🔴")
        st.caption(str(e))

    st.divider()

    # Active document display.
    st.subheader("Active Document")
    active_doc = st.session_state.get("active_doc", "aapl-10k-2024.pdf (default)")
    st.info(f"📄 {active_doc}")

    st.divider()

    # PDF upload — same UI as before.
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
                result = process_upload(uploaded)

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
        # Clear session overrides so run_query() falls back to cached pipeline.
        st.session_state.pop("session_chunks", None)
        st.session_state.pop("session_store",  None)
        st.session_state.pop("session_bm25",   None)
        st.session_state["active_doc"] = "aapl-10k-2024.pdf (default)"
        st.session_state["messages"]   = []
        st.rerun()

    st.divider()

    # Retrieval settings — identical slider.
    st.subheader("Settings")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values retrieve more context but use more tokens.",
    )


# Main chat area — identical to the original app.py.

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

    # Render user message immediately.
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Run pipeline and render answer.
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            response = run_query(question, top_k)

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