"""
Pydantic models for API request and response contracts.
Single responsibility: data shape validation only.
"""

from pydantic import BaseModel, Field


#  Requests 

class QueryRequest(BaseModel):
    """POST /api/v1/query request body."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to answer from the loaded document.",
        examples=["What was Apple's total revenue in fiscal year 2024?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Must be between 1 and 20.",
    )

# Responses 

class SourceReference(BaseModel):
    """A single cited source chunk returned in a query response."""

    source:  str = Field(description="Source document filename.")
    page:    int = Field(description="1-indexed page number in the PDF.")
    preview: str = Field(description="First 150 characters of the chunk text.")


class QueryResponse(BaseModel):
    """POST /api/v1/query response body."""

    answer:      str                   = Field(description="Grounded answer from the LLM.")
    sources:     list[SourceReference] = Field(description="Source chunks cited in the answer.")
    model:       str                   = Field(description="LLM model used to generate the answer.")
    tokens_used: int                   = Field(description="Total tokens consumed (prompt + completion).")
    question:    str                   = Field(description="Original question echoed back.")


class HealthResponse(BaseModel):
    """GET /api/v1/health response body."""

    status:          str  = Field(description="'ok' if the service is healthy.")
    retriever_ready: bool = Field(description="True if retrievers are loaded in app.state.")
    model:           str  = Field(description="Configured LLM model name.")


class UploadResponse(BaseModel):
    """POST /api/v1/upload response body."""

    filename:       str = Field(description="Sanitised name of the uploaded PDF.")
    chunks_created: int = Field(description="Number of chunks embedded and stored.")
    message:        str = Field(description="Human-readable status message.")


class ErrorResponse(BaseModel):
    """Standard error response body for all handled exceptions."""

    detail: str = Field(description="Human-readable error description.")