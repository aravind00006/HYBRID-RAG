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
