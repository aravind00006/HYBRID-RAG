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


