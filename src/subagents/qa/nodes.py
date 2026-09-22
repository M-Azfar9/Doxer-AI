"""
Execution nodes for QA Subagent.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from src.core.service_registry import ServiceRegistry, services as default_services
from src.core.structured_output import StructuredOutputNode
from src.subagents.qa.state import QAState, RouteDecision, IntentClassification


ROUTER_SYSTEM_PROMPT = """You are the intent-classification component of an AI technical engineering assistant.
Your task is to classify the user's question into EXACTLY ONE of three routes:

1. 'direct': General programming concepts, standard language syntax, CS theory, data structures, algorithms.
   Examples: 'What is dependency injection?', 'Explain list vs tuple in Python', 'How does quicksort work?'

2. 'web_search': Current versions, release dates, breaking news, new library announcements, external framework setup.
   Examples: 'What is the latest stable version of LangGraph?', 'Next.js 15 release notes', 'Current price of GPU compute'

3. 'rag': Queries asking about the local codebase, repository implementation details, specific files/classes/functions.
   Examples: 'Show me the StateGraph builder in this repo', 'How does the IntentRouter handle retries in our code?', 'Find the GitHub client'
"""


class QANodes:
    """Encapsulates all node logic for the QA subgraph."""

    def __init__(self, services: Optional[ServiceRegistry] = None):
        self.services = services or default_services
        self.router = StructuredOutputNode(
            llm=self.services.llm,
            schema=IntentClassification,
            system_prompt=ROUTER_SYSTEM_PROMPT
        )

    def classify_intent(self, state: QAState) -> Dict[str, Any]:
        """Node 1: Classifies the question into direct, web_search, or rag."""
        query = state["query"]
        try:
            classification: IntentClassification = self.router.invoke(user_prompt=query)
            route = classification.route
        except Exception as e:
            print(f"⚠️ [QA classify_intent] Router fallback to DIRECT: {e}")
            route = RouteDecision.DIRECT

        return {
            "route": route,
            "needs_web_search": (route == RouteDecision.WEB_SEARCH),
        }

    def answer_direct(self, state: QAState) -> Dict[str, Any]:
        """Node 2A: Direct answer without external tools."""
        query = state["query"]
        messages = [
            SystemMessage(content="You are an expert software engineer. Provide a concise, technically precise answer."),
            HumanMessage(content=query)
        ]
        response = self.services.llm.invoke(messages)
        answer = getattr(response, "content", str(response))
        return {"answer": answer}

    def web_search(self, state: QAState) -> Dict[str, Any]:
        """Node 2B: Fetch real-time web search results."""
        query = state["query"]
        max_results = state.get("max_results", 5)
        results = self.services.tavily.search(query, max_results=max_results)
        return {"search_results": results}

    def synthesize_web(self, state: QAState) -> Dict[str, Any]:
        """Node 2B-2: Synthesizes web search results."""
        query = state["query"]
        results = state.get("search_results", [])
        if not results:
            return {"answer": "I attempted to search the web for current information, but no relevant results were found."}

        context_str = "\n\n".join([
            f"Source [{r.get('url', 'N/A')}]:\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')}"
            for r in results
        ])
        messages = [
            SystemMessage(content="You are an expert research engineer. Synthesize the provided web search results to answer the question. Cite source URLs inline or as footnotes."),
            HumanMessage(content=f"Question: {query}\n\nWeb Search Evidence:\n{context_str}")
        ]
        response = self.services.llm.invoke(messages)
        answer = getattr(response, "content", str(response))
        return {"answer": answer}

    def retrieve_code(self, state: QAState) -> Dict[str, Any]:
        """Node 2C: Retrieve relevant code snippets from Chroma DB."""
        query = state["query"]
        chunks = self.services.vector_store.retrieve(query, k=4)
        return {"retrieved_chunks": chunks}

    def synthesize_code(self, state: QAState) -> Dict[str, Any]:
        """Node 2C-2: Synthesizes an answer grounded in retrieved code."""
        query = state["query"]
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            # Fallback if no code found in Chroma
            messages = [
                SystemMessage(content="You are an expert codebase assistant. No direct code snippets were found in the index; answer based on general engineering knowledge with a caveat."),
                HumanMessage(content=query)
            ]
            response = self.services.llm.invoke(messages)
            answer = getattr(response, "content", str(response))
            return {"answer": f"{answer}\n\n*(Note: No exact code matches found in local index)*"}

        evidence_str = "\n\n".join([
            f"File: {c.get('source_file', 'unknown')} (score: {c.get('score', 0)}):\n```\n{c.get('content', '')}\n```"
            for c in chunks
        ])
        messages = [
            SystemMessage(content="You are an expert code architect. Explain the codebase details answering the user question based strictly on the provided file snippets. Reference file names where relevant."),
            HumanMessage(content=f"Question: {query}\n\nRetrieved Code Chunks:\n{evidence_str}")
        ]
        response = self.services.llm.invoke(messages)
        answer = getattr(response, "content", str(response))
        return {"answer": answer}
