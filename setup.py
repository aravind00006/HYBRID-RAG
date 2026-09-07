"""
Editable install configuration for HYBRID-RAG.
"""

from setuptools import setup, find_packages

setup(
    name="hybrid-rag",
    version="1.0.0",
    description="Production-grade Hybrid RAG system over SEC 10-K filings.",
    author="Batman",
    python_requires=">=3.11",
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "data",
            "data.*",
        ]
    ),
    install_requires=[
        # Core framework
        "fastapi==0.115.0",
        "uvicorn==0.30.6",
        "streamlit==1.39.0",
        "pydantic==2.9.2",
        "pydantic-settings==2.5.2",
        # Rate limiting
        "slowapi==0.1.9",
        # LangChain
        "langchain==0.3.1",
        "langchain-core==0.3.6",
        "langchain-community==0.3.1",
        "langchain-openai==0.2.1",
        "langchain-chroma==0.1.4",
        "langchain-text-splitters==0.3.0",
        # Vector store
        "chromadb==0.5.11",
        # PDF loading
        "pypdf==5.0.1",
        # Retrieval
        "rank-bm25==0.2.2",
        # LLM and embeddings
        "openai==1.51.0",
        # Evaluation
        "ragas==0.2.6",
        "datasets==3.0.1",
        # Observability
        "langsmith==0.1.129",
        # HTTP client
        "requests==2.32.3",
        # Utilities
        "python-multipart==0.0.12",
        "python-dotenv==1.0.1",
    ],
    extras_require={
        # Install with: pip install -e ".[dev]"
        "dev": [
            "pytest==8.3.3",
            "pytest-asyncio==0.24.0",
            "httpx==0.27.2",
        ],
    },
)