"""
State schema and data models for the QA Subagent (Feature 2).
"""

from enum import Enum
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class RouteDecision(str, Enum):
    DIRECT = "direct"
    WEB_SEARCH = "web_search"
    RAG = "rag"


class IntentClassification(BaseModel):
    """Structured output for QA intent router."""
    route: RouteDecision = Field(
        description="Select 'direct' for general programming concepts, 'web_search' for real-time/version info, 'rag' for repository-specific code."
    )
    reasoning: str = Field(description="Explanation for routing choice.")


class QAState(TypedDict):
    """State schema for the 3-Way QA agent graph."""
    query: str
    route: Optional[RouteDecision]
    needs_web_search: bool
    max_results: int
    search_results: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    error: Optional[str]
