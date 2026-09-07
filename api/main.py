"""
api/main.py — FastAPI application factory and lifespan manager.
Single responsibility: app wiring only.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rank_bm25 import BM25Okapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routes import router
from config import get_settings
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, load_store
from ingestion.loader import load_pdf
from logger import get_logger

logger = get_logger(__name__)

# Rate limiter 
# get_remote_address extracts the client IP from the request for per-IP limits.
# The limiter instance is imported by routes.py to decorate individual endpoints.
limiter = Limiter(key_func=get_remote_address)

