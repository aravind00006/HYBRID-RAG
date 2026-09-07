"""
Central settings loaded from .env. Single source of truth for all configuration.

"""

from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    #  API Keys 
    openai_api_key: str = Field(..., description="OpenAI API key (required).")
    langchain_api_key: str = Field("", description="LangSmith API key (optional).")

    #  LangSmith 
    langchain_tracing_v2: bool = Field(default=True)
    langchain_project: str = Field(default="hybrid-rag")

    #  LLM 
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.0)

    #  Embeddings 
    embedding_model: str = Field(default="text-embedding-3-small")

    #  ChromaDB 
    chroma_path: str = Field(default="./data/chroma")
    collection_name: str = Field(default="hybrid_rag")

    #  Chunking 
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=100)

    #  Retrieval 
    top_k: int = Field(default=5)

    #  Default PDF 
    default_pdf_path: str = Field(
        default="data/raw/aapl-10k-2024.pdf",
        description="Path to the default PDF loaded on startup.",
    )

    #  Rate Limiting 
    rate_limit_query: str = Field(
        default="100/minute",
        description="Max requests per minute for the /query endpoint.",
    )
    rate_limit_upload: str = Field(
        default="10/minute",
        description="Max requests per minute for the /upload endpoint.",
    )

    #  Validators 
    @field_validator("llm_temperature")
    @classmethod
    def valid_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be 0.0–2.0, got {v}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 500)
        if v >= size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({size})."
            )
        return v

    @field_validator("top_k")
    @classmethod
    def valid_top_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError(f"top_k must be between 1 and 20, got {v}.")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    """
    return Settings()