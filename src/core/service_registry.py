"""
Unified Shared Services Layer (Phase 10 ServiceRegistry).
Consolidates LLM client(s), Tavily search client, GitHub MCP / REST client,
Vector store client, and checkpointer into a single injectable dependency.
"""

import os
from typing import Optional, List, Dict, Any
from langgraph.checkpoint.memory import MemorySaver

from src.core.config import config
from src.core.llm_manager import get_default_llm, MultiKeyGeminiLLM, ResilientBedrockLLM
from src.github_mcp_client import GitHubMCPClient


class VectorStoreClient:
    """Wraps Chroma vector store for code snippet retrieval."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or config.chroma_db_dir
        self._vector_store = None
        self._embeddings = None

    def _get_store(self):
        if self._vector_store is None:
            try:
                from langchain_chroma import Chroma
                from langchain_mistralai import MistralAIEmbeddings
                self._embeddings = MistralAIEmbeddings(
                    model="mistral-embed",
                    api_key=config.mistral_api_key
                )
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self._embeddings,
                )
            except Exception as e:
                print(f"⚠️ [VectorStoreClient] Chroma initialization warning: {e}")
                self._vector_store = False
        return self._vector_store

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant code chunks."""
        store = self._get_store()
        if not store:
            return []
        try:
            docs_and_scores = store.similarity_search_with_relevance_scores(query, k=k)
            results = []
            for doc, score in docs_and_scores:
                results.append({
                    "content": doc.page_content,
                    "source_file": doc.metadata.get("source", "unknown"),
                    "score": round(float(score), 4),
                })
            return results
        except Exception as e:
            print(f"⚠️ [VectorStoreClient] Retrieval exception: {e}")
            return []


class TavilySearchClient:
    """Wrapper for Tavily Web Search."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.tavily_api_key
        self._tool = None

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        if not self.api_key:
            print("⚠️ [TavilySearchClient] TAVILY_API_KEY is not set.")
            return []
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            if self._tool is None:
                self._tool = TavilySearchResults(
                    tavily_api_key=self.api_key,
                    max_results=max_results
                )
            raw = self._tool.invoke({"query": query})
            results = []
            if isinstance(raw, list):
                for item in raw:
                    if isinstance(item, dict):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "content": item.get("content", "")
                        })
            return results
        except Exception as e:
            print(f"⚠️ [TavilySearchClient] Search failed: {e}")
            return []


class ServiceRegistry:
    """
    Phase 10 Service Registry.
    Single point of injection for all shared services across subgraphs.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        tavily: Optional[TavilySearchClient] = None,
        github: Optional[GitHubMCPClient] = None,
        vector_store: Optional[VectorStoreClient] = None,
        checkpointer: Optional[MemorySaver] = None
    ):
        self.llm = llm or get_default_llm()
        self.tavily = tavily or TavilySearchClient()
        self.github = github or GitHubMCPClient()
        self.vector_store = vector_store or VectorStoreClient()
        self.checkpointer = checkpointer or MemorySaver()


# Global shared registry
services = ServiceRegistry()
