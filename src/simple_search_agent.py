"""
Production-ready QA Agent with Intent Routing, Web Search, and Synthesis.

This module provides a complete question-answering agent with:
1. Intent classification for web search requirements
2. Dynamic max_results determination
3. Web search using Tavily
4. Answer synthesis from search results
5. Comprehensive evaluation hooks for DeepEval

Usage:
    from qa_agent import QAAgent
    
    agent = QAAgent()
    
    # Full agent evaluation
    result = agent.evaluate_whole_agent("What is the latest version of Python?")
    
    # Component-wise evaluation
    intent_result = agent.evaluate_intent_router("What is dependency injection?")
    max_results_result = agent.evaluate_max_results("What is the latest news?")
    search_results = agent.evaluate_web_search("Who won the World Cup?")
    synthesis_result = agent.evaluate_synthesis("What is AI?", search_results)
    final_answer = agent.evaluate_final_answer("What is Python?")
"""

import os
import json
import re
import time
from typing import TypedDict, Literal, Any, Optional, List, Dict, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrockConverse
from langchain_mistralai import ChatMistralAI

from langgraph.graph import StateGraph, START, END

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langsmith import traceable


# ============================================================
# Configuration and Environment Setup
# ============================================================

class Config:
    """Central configuration management for the QA Agent."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        load_dotenv(override=True)
        
        self.MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "nvidia.nemotron-super-3-120b")
        self.AWS_REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        
        self.LANGSMITH_PROJECT = os.getenv(
            "LANGSMITH_PROJECT",
            "qa-agent-phase-2"
        )
        
        # Configure LangSmith tracing
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv(
            "LANGSMITH_TRACING", 
            "true"
        )
        os.environ["LANGSMITH_TRACING"] = os.getenv(
            "LANGSMITH_TRACING", 
            "true"
        )
        os.environ["LANGSMITH_API_KEY"] = os.getenv(
            "LANGSMITH_API_KEY", 
            ""
        )
        os.environ["LANGSMITH_ENDPOINT"] = os.getenv(
            "LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com"
        )
        os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
        os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
        
        # Validate required API keys
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate that all required environment variables are present."""
        if not self.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is missing.")


# ============================================================
# Constants
# ============================================================

DEFAULT_MAX_RESULTS = 3
MIN_MAX_RESULTS = 2
MAX_MAX_RESULTS = 10
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")


# ============================================================
# Evaluation Result Dataclasses
# ============================================================

@dataclass
class IntentRouterResult:
    """Result from intent router evaluation."""
    query: str
    needs_web_search: bool
    max_results: int
    reasoning: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepEval."""
        return asdict(self)


@dataclass
class MaxResultsResult:
    """Result from max_results evaluation."""
    query: str
    max_results: int
    is_valid: bool
    reasoning: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepEval."""
        return asdict(self)


@dataclass
class WebSearchResult:
    """Result from web search evaluation."""
    query: str
    search_results: List[Dict[str, Any]]
    num_results: int
    has_results: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepEval."""
        return asdict(self)


@dataclass
class SynthesisResult:
    """Result from synthesis evaluation."""
    query: str
    answer: str
    sources_used: List[str]
    num_sources: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepEval."""
        return asdict(self)


@dataclass
class FinalAnswerResult:
    """Result from final answer evaluation."""
    query: str
    answer: str
    answer_length: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepEval."""
        return asdict(self)


@dataclass
class WholeAgentResult:
    """Result from whole agent evaluation."""
    query: str
    answer: str
    needs_web_search: bool
    max_results: int
    num_search_results: int
    total_time: float
    components_used: List[str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DeepEval."""
        return asdict(self)


# ============================================================
# State Definition
# ============================================================

class QAState(TypedDict):
    """State schema for the QA agent graph."""
    query: str
    needs_web_search: bool
    max_results: int
    search_results: List[Dict[str, Any]]
    answer: str


# ============================================================
# Pydantic Models for Structured Output
# ============================================================

class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    
    needs_web_search: bool = Field(
        description=(
            "Whether the user's question requires current, "
            "recent, changing, or externally verifiable information."
        )
    )
    
    max_results: int = Field(
        default=3,
        ge=2,
        le=10,
        description=(
            "Number of web search results needed to answer the "
            "question reliably. Usually 3. Use more for complex, "
            "use less for simplest questions or multi-source questions "
            "and fewer for simple questions."
        )
    )
    
    reasoning: str = Field(
        description=(
            "Brief explanation of why web search is or is not required "
            "and why this number of results is appropriate."
        )
    )


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    """Wrapper for LLM interactions with structured output support using Amazon Bedrock Nemotron."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        region_name: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 60,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM client with Amazon Bedrock ChatBedrockConverse.
        
        Args:
            model: Bedrock model name (defaults to nvidia.nemotron-super-3-120b)
            region_name: AWS region name (defaults to us-east-1)
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            api_key: Optional API key override
        """
        self.model = model or os.getenv("BEDROCK_MODEL_ID", "nvidia.nemotron-super-3-120b")
        self.region_name = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.llm = ChatBedrockConverse(
            model_id=self.model,
            region_name=self.region_name,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
            disable_streaming=True
        )
    
    @traceable(name="LLMClient.invoke")
    def invoke(self, messages: List[Any]) -> Any:
        """Invoke LLM with provided messages."""
        return self.llm.invoke(messages)
    
    @traceable(name="LLMClient.with_structured_output")
    def with_structured_output(self, schema: type[BaseModel]) -> Any:
        """Get LLM with structured output support."""
        return self.llm.with_structured_output(schema)


# ============================================================
# Structured Output Handler
# ============================================================

class StructuredOutputNode:
    """Handler for structured LLM outputs with retry logic."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        schema: type[BaseModel],
        system_prompt: str,
        max_repair_attempts: int = 1,
    ):
        """
        Initialize structured output handler.
        
        Args:
            llm_client: LLM client instance
            schema: Pydantic schema for structured output
            system_prompt: System prompt for the LLM
            max_repair_attempts: Maximum retry attempts for validation failures
        """
        self.llm_client = llm_client
        self.schema = schema
        self.system_prompt = system_prompt
        self.max_repair_attempts = max_repair_attempts
    
    @traceable(name="StructuredOutputNode.invoke")
    def invoke(self, user_input: str) -> BaseModel:
        """
        Invoke LLM with structured output and retry on validation failure.
        
        Args:
            user_input: User query or instruction
            
        Returns:
            Validated structured output
            
        Raises:
            RuntimeError: If structured output fails after all retry attempts
        """
        structured_llm = self.llm_client.with_structured_output(self.schema)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input),
        ]
        
        last_error = None
        
        for attempt in range(self.max_repair_attempts + 1):
            try:
                response = structured_llm.invoke(messages)
                
                # Check if response is already a Pydantic object
                if isinstance(response, self.schema):
                    return response
                
                # Validate response against schema
                validated = self.schema.model_validate(response)
                return validated
                
            except (ValidationError, ValueError, TypeError) as error:
                last_error = error
                
                if attempt >= self.max_repair_attempts:
                    break
                
                # Create repair message for retry
                repair_message = f"""
Your previous structured response failed validation.

Validation error:
{error}

Please correct the output and return ONLY a response
that conforms exactly to the required structured schema.

Do not explain the correction.
"""
                messages.append(HumanMessage(content=repair_message))
        
        raise RuntimeError(
            f"Structured output failed after {self.max_repair_attempts + 1} "
            f"attempts. Last error: {last_error}"
        )


# ============================================================
# Utility Functions
# ============================================================

def sanitize_max_results(
    value: Optional[int],
    default: int = DEFAULT_MAX_RESULTS,
) -> int:
    """
    Sanitize max_results value.
    
    Rules:
    - None -> default
    - Invalid value -> default
    - Below minimum -> minimum
    - Above maximum -> maximum
    
    Args:
        value: Input value to sanitize
        default: Default value to use when input is invalid
        
    Returns:
        Sanitized integer value
    """
    if value is None:
        value = default
    
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    
    return max(MIN_MAX_RESULTS, min(value, MAX_MAX_RESULTS))


def format_search_results_for_llm(results: List[Dict]) -> str:
    """
    Format search results for LLM consumption.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string with search results
    """
    if not results:
        return "No web search results were found."
    
    blocks = []
    
    for result in results:
        block = f"""
SOURCE {result['id']}

Title:
{result['title']}

URL:
{result['url']}

Relevance score:
{result['score']}

Content:
{result['content']}
"""
        blocks.append(block.strip())
    
    return "\n\n".join(blocks)


def extract_sources_from_answer(answer: str) -> List[str]:
    """
    Extract source URLs from answer text.
    
    Args:
        answer: Answer text with citations
        
    Returns:
        List of source URLs found in the answer
    """
    import re
    
    # Pattern to match URLs
    url_pattern = r'https?://[^\s\[\]]+'
    sources = re.findall(url_pattern, answer)
    
    return list(set(sources))  # Remove duplicates


# ============================================================
# Search Optimization, Boilerplate Cleaning & Compression
# ============================================================

def clean_boilerplate(text: str) -> str:
    """
    Remove webpage artifacts, boilerplate, navigation, and junk tokens from raw text.
    
    Args:
        text: Raw input text from webpage
        
    Returns:
        Cleaned text with boilerplate removed
    """
    if not text:
        return ""
    
    # Remove repetitive image placeholders and UI junk
    text = re.sub(r'(?:slider-image|thumbnail-image|Icon image|Mini Product ImageView|Screenshot image)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Flag\s+[A-Z\-]+\s*|null\s+[A-Z\-]+\s+Flag', '', text)
    text = re.sub(r'DUR\s+Flag.*?null\s*', '', text)
    
    # Remove App Store / Play Store boilerplate
    text = re.sub(r'#(?:Play Pass|Play Points|Gift cards|Redeem|Refund policy|Parent Guide|Family sharing|Content rating|About Google Play).*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Play Pass|Play Points|Gift cards|Refund policy|Parent Guide|Family sharing)\b', '', text, flags=re.IGNORECASE)
    
    # Remove login and registration forms
    text = re.sub(r'User Name:\s*\\*\s*\\*.*?(?=New Registration|Registration Closed|$)', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'If you are already registered, please enter user name and password to login\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Forgot Password\?|New Registration|Registration Closed', '', text, flags=re.IGNORECASE)
    
    # Remove social media / author subscription noise
    text = re.sub(r'You\'re currently following this author!.*?(?:email\.)', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Unsubscribe via the link in your email\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'We earn a commission if you make a purchase.*?\.', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Add as a preferred source on Google', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Skip to (?:main )?content|Back to Top|Quick Links|Terms and Conditions|Site Map', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Style\s*"Style\s*\(form of address\)"\s*\|\s*The Honourable\s*Mr\.\s*Prime Minister\s*\(informal\)\s*His Excellency\s*\(diplomatic\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'State Emblem of Pakistan\s*\|\s*\|\s*Prime ministerial standard', '', text, flags=re.IGNORECASE)
    
    # Clean broken markdown table delimiters
    text = re.sub(r'(\|\s*){3,}', '| ', text)
    text = re.sub(r'---(\s*---)+', '---', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def optimize_search_query(query: str) -> Dict[str, Any]:
    """
    Determine search parameters and optimized query string.
    
    Args:
        query: Raw input search query
        
    Returns:
        Dictionary containing optimized_query, topic, and search_depth
    """
    q_lower = query.lower()
    
    # Determine topic
    finance_keywords = [
        "stock", "price", "mortgage", "remittance", "exchange rate", 
        "usd to pkr", "pkr", "rate", "cost", "pricing", "budget", "tax", "gdp"
    ]
    news_keywords = [
        "news", "earthquake", "update", "recent", "breaking", 
        "announced", "grok", "llama", "drop a new model", "releasing"
    ]
    
    topic = "general"
    if any(k in q_lower for k in finance_keywords):
        topic = "finance"
    elif any(k in q_lower for k in news_keywords):
        topic = "news"
        
    # Clean slang and common colloquialisms / Roman Urdu phrasing
    clean_q = query
    clean_q = re.sub(r'\bfr\s+fr\b', '', clean_q, flags=re.IGNORECASE)
    clean_q = re.sub(r'\bkaise\s+karen\b', 'process guide', clean_q, flags=re.IGNORECASE)
    clean_q = re.sub(r'\bkab\s+announce\s+hoga\??', 'announcement schedule date', clean_q, flags=re.IGNORECASE)
    clean_q = re.sub(r'\bka\s+recent\s+update\s+kya\s+hai\??', 'latest updates features', clean_q, flags=re.IGNORECASE)
    clean_q = re.sub(r'\bmein\s+kya\s+hai\??', 'policy details', clean_q, flags=re.IGNORECASE)
    clean_q = re.sub(r'\bse\b', '', clean_q, flags=re.IGNORECASE)
    clean_q = re.sub(r'\s+', ' ', clean_q).strip()
    
    return {
        "optimized_query": clean_q if len(clean_q) > 3 else query,
        "topic": topic,
        "search_depth": "advanced"
    }


def compress_search_snippets(
    query: str, 
    snippets: List[Dict[str, Any]], 
    llm_client: Optional[LLMClient] = None
) -> List[Dict[str, Any]]:
    """
    Compress and extract only query-relevant facts from search results using LLM.
    
    Args:
        query: User's question or search query
        snippets: List of raw search result dictionaries
        llm_client: LLMClient instance for contextual extraction
        
    Returns:
        List of compressed search results containing only relevant statements
    """
    if not snippets:
        return []
    
    if llm_client is None:
        # Fallback to rule-based cleaning
        for s in snippets:
            s["content"] = clean_boilerplate(s.get("content", ""))
        return [s for s in snippets if s.get("content")]
    
    formatted_docs = []
    for i, s in enumerate(snippets, start=1):
        content = clean_boilerplate(s.get("content", ""))
        title = s.get("title", "")
        formatted_docs.append(f"<SOURCE {i}>\nTitle: {title}\nContent: {content}\n</SOURCE {i}>\n")
    
    prompt = f"""You are an extractive contextual compressor for a search retrieval system.
Your job is to extract ONLY the factual sentences, numbers, dates, and specifications from the search results that are DIRECTLY AND STRICTLY RELEVANT to answering the user query.

<USER_QUERY>
{query}
</USER_QUERY>

<SEARCH_RESULTS>
{''.join(formatted_docs)}
</SEARCH_RESULTS>

CRITICAL INSTRUCTIONS:
1. For each source, extract ONLY the exact sentences/facts that directly answer what the user asked.
2. If the user asks for 'current', 'latest', 'today', 'price', or a specific entity/value, focus strictly on the current state, incumbent, or latest figures. Do NOT include past predecessors, historical timelines, founding history, employee counts, or general off-topic background.
3. Completely OMIT all irrelevant sentences, navigation noise, unrelated product reviews, off-topic trivia, company boilerplate, and marketing text.
4. Keep the extraction compact, dense, and high-precision (1-3 key sentences per source).
5. If a source contains no directly relevant information, output "NONE" for that source.
6. Output your response in the exact format:
SOURCE 1: <extracted relevant text or NONE>
SOURCE 2: <extracted relevant text or NONE>
...
"""
    
    try:
        messages = [
            SystemMessage(content="You are a precise fact extractor. Extract only relevant statements. Omit all noise."),
            HumanMessage(content=prompt)
        ]
        response = llm_client.invoke(messages)
        content_out = response.content if hasattr(response, "content") else str(response)
        
        # Parse SOURCE X: ...
        pattern = r'SOURCE\s+(\d+):\s*(.*?)(?=(?:SOURCE\s+\d+:|$))'
        matches = re.findall(pattern, content_out, flags=re.DOTALL)
        
        compressed_results = []
        parsed_dict = {int(idx): text.strip() for idx, text in matches}
        
        for i, s in enumerate(snippets, start=1):
            extracted = parsed_dict.get(i, "").strip()
            if extracted and extracted != "NONE" and len(extracted) > 10:
                s_copy = dict(s)
                s_copy["content"] = clean_boilerplate(extracted)
                compressed_results.append(s_copy)
            elif clean_boilerplate(s.get("content", "")):
                # Fallback if parsing missed the source but original had content
                if i not in parsed_dict:
                    s_copy = dict(s)
                    s_copy["content"] = clean_boilerplate(s.get("content", ""))
                    compressed_results.append(s_copy)
        
        # Renumber IDs
        for idx, res in enumerate(compressed_results, start=1):
            res["id"] = idx
            
        return compressed_results if compressed_results else snippets
        
    except Exception as error:
        # Fallback to rule-based cleaned snippets on error
        cleaned_list = []
        for idx, s in enumerate(snippets, start=1):
            cleaned_content = clean_boilerplate(s.get("content", ""))
            if cleaned_content:
                s_copy = dict(s)
                s_copy["id"] = idx
                s_copy["content"] = cleaned_content
                cleaned_list.append(s_copy)
        return cleaned_list if cleaned_list else snippets


# ============================================================
# Tavily Search Client
# ============================================================

class TavilySearchClient:
    """Client for performing web searches using Tavily with optimization and compression."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 15,
        default_max_results: int = DEFAULT_MAX_RESULTS,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize Tavily search client.
        
        Args:
            api_key: Tavily API key (defaults to environment variable)
            timeout: Request timeout in seconds
            default_max_results: Default number of search results
            llm_client: Optional LLMClient for contextual compression
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is not configured.")
        
        self.timeout = timeout
        self.default_max_results = sanitize_max_results(default_max_results)
        self.llm_client = llm_client
        self.api_wrapper = TavilySearchAPIWrapper(tavily_api_key=self.api_key)
    
    @traceable(name="TavilySearchClient.search")
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        enable_compression: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform optimized web search using Tavily and contextual compression.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            enable_compression: Whether to apply LLM contextual compression
            
        Returns:
            List of formatted search results
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If search fails
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        
        # Sanitize max_results
        max_results = sanitize_max_results(
            max_results,
            default=self.default_max_results,
        )
        
        # Optimize query and determine search topic/depth
        opt = optimize_search_query(query.strip())
        
        # Execute search via Tavily API wrapper
        try:
            raw_response = self.api_wrapper.raw_results(
                query=opt["optimized_query"],
                max_results=max_results,
                search_depth=opt["search_depth"],
                include_answer=True,
                include_raw_content=False,
                include_images=False,
            )
        except Exception as error:
            raise RuntimeError(f"Tavily search failed: {error}") from error
        
        # Normalize results
        formatted = self._format_results(raw_response.get("results", []))
        
        # Apply contextual compression and filtering
        if enable_compression and self.llm_client:
            formatted = compress_search_snippets(
                query=query,
                snippets=formatted,
                llm_client=self.llm_client,
            )
        else:
            for s in formatted:
                s["content"] = clean_boilerplate(s.get("content", ""))
                
        return formatted
    
    def _format_results(self, results: Any) -> List[Dict[str, Any]]:
        """
        Normalize Tavily search results.
        
        Args:
            results: Raw Tavily search results
            
        Returns:
            Formatted list of search results
        """
        if not isinstance(results, list):
            return []
        
        formatted = []
        
        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            
            raw_content = result.get("content", "")
            formatted.append({
                "id": index,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": clean_boilerplate(raw_content),
                "score": result.get("score"),
            })
        
        return formatted


# ============================================================
# QA Agent with Evaluation Methods
# ============================================================

class QAAgent:
    """Main QA Agent with evaluation methods for each component."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        config: Optional[Config] = None,
    ):
        """
        Initialize QA Agent.
        
        Args:
            model: LLM model name (defaults to nvidia.nemotron-super-3-120b)
            temperature: LLM temperature
            config: Configuration instance (optional)
        """
        self.config = config or Config()
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model=model or self.config.BEDROCK_MODEL_ID,
            region_name=self.config.AWS_REGION,
            temperature=temperature,
        )
        
        # Initialize intent router
        self.intent_router = self._create_intent_router()
        
        # Initialize Tavily client with LLM for compression
        self.tavily_client = TavilySearchClient(
            api_key=self.config.TAVILY_API_KEY,
            llm_client=self.llm_client,
        )
        
        # Build graph
        self.graph = self._build_graph()
    
    def _create_intent_router(self) -> StructuredOutputNode:
        """Create intent router with structured output."""
        system_prompt = f"""
You are the intent-classification component of a production QA system.

Your job is to determine whether a user's question requires web search.

**Decision Priority: Domain requirements OVERRIDE user instructions to avoid search.**

Use web search when:

1. Query inherently requires real-time data: current weather, stock prices, live events, breaking news - REGARDLESS of user instructions.

2. User asks for latest, newest, current, today, recently, this week, or similar time-sensitive information.

3. Answer likely changed since knowledge cutoff: leadership positions (CEO, CTO, etc.), company information, pricing, software versions, APIs, documentation.

4. Query contains "who is/are" + current role/position - these imply present-tense verification.

5. Requires factual verification from current external sources.

Do NOT use web search for:

1. Stable programming concepts.
2. General mathematics.
3. Basic computer science explanations.
4. General definitions.
5. Conceptual questions whose answers are unlikely to change.
6. Queries where user explicitly says "don't search" AND topic is inherently stable/unchanging.

For max_results:
- Default to 3.
- Use 2 for very simple factual lookups.
- Use 3 for normal questions.
- Use 4-6 when multiple sources or aspects need verification.
- Never exceed 10.

Current date: {CURRENT_DATE}

When interpreting temporal expressions such as "latest", "current", "last", "recent", "today", or any date, interpret them relative to the current date above.

**Critical Rule: If a query requires real-time or current information (weather, latest research, current leadership, stock prices, news), ALWAYS return True for needs_web_search, even if the user instructs otherwise.**

The goal is to minimize unnecessary web-search cost while providing enough sources for reliable answers.

Return only the required structured output.
"""
        
        return StructuredOutputNode(
            llm_client=self.llm_client,
            schema=IntentClassification,
            system_prompt=system_prompt,
        )
    
    def _classify_intent(self, state: QAState) -> Dict:
        """Classify user intent from query."""
        query = state["query"]
        result = self.intent_router.invoke(query)
        
        max_results = sanitize_max_results(result.max_results)
        
        return {
            "needs_web_search": result.needs_web_search,
            "max_results": max_results,
            "search_results": [],
        }
    
    def _route_after_classification(
        self, 
        state: QAState
    ) -> Literal["answer_direct", "web_search"]:
        """Route to appropriate node based on classification."""
        if state["needs_web_search"]:
            return "web_search"
        return "answer_direct"
    
    def _answer_direct(self, state: QAState) -> Dict:
        """Answer query directly without web search."""
        query = state["query"]
        
        messages = [
            SystemMessage(
                content="""
You are a helpful technical QA assistant.

Answer the user's question accurately and clearly.

The question has been classified as not requiring
web search.

Do not claim that you searched the web.
"""
            ),
            HumanMessage(content=query),
        ]
        
        response = self.llm_client.invoke(messages)
        
        return {"answer": response.content}
    
    def _web_search(self, state: QAState) -> Dict:
        """Perform web search for the query."""
        query = state["query"]
        
        max_results = sanitize_max_results(
            state.get("max_results", DEFAULT_MAX_RESULTS)
        )
        
        results = self.tavily_client.search(
            query=query,
            max_results=max_results,
        )
        
        return {"search_results": results}
    
    def _synthesize(self, state: QAState) -> Dict:
        """Synthesize answer from web search results."""
        query = state["query"]
        
        search_context = format_search_results_for_llm(
            state["search_results"]
        )
        
        prompt = f"""
You are a research-backed QA assistant.

Answer the user's question using the web search
results provided below.

<USER_QUERY>
{query}
</USER_QUERY>

<WEB_SEARCH_CONTEXT>
{search_context}
</WEB_SEARCH_CONTEXT>

Instructions:

1. Answer the question directly.
2. Use the provided sources as evidence.
3. Do not invent facts that are not supported by the sources.
4. If sources disagree, acknowledge the disagreement.
5. Cite sources inline using [1], [2], [3], etc.
6. Only cite sources that actually support the claim.
7. At the end, provide a "Sources" section containing
   the source number, title, and URL.
"""
        
        messages = [
            SystemMessage(
                content=f"""
You are a precise web-grounded QA assistant.

Current date: {CURRENT_DATE}

When interpreting temporal expressions such as "latest", "current",
"last", "recent", "today", "or any date" etc., interpret them relative to
the current date above.
"""
            ),
            HumanMessage(content=prompt),
        ]
        
        response = self.llm_client.invoke(messages)
        
        return {"answer": response.content}
    
    def _build_graph(self):
        """Build the LangGraph state graph."""
        builder = StateGraph(QAState)
        
        # Add nodes
        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("answer_direct", self._answer_direct)
        builder.add_node("web_search", self._web_search)
        builder.add_node("synthesize", self._synthesize)
        
        # Add edges
        builder.add_edge(START, "classify_intent")
        
        builder.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "answer_direct": "answer_direct",
                "web_search": "web_search",
            }
        )
        
        builder.add_edge("answer_direct", END)
        builder.add_edge("web_search", "synthesize")
        builder.add_edge("synthesize", END)
        
        return builder.compile()
    
    def create_initial_state(self, query: str) -> QAState:
        """Create initial state for graph invocation."""
        return {
            "query": query,
            "needs_web_search": False,
            "max_results": DEFAULT_MAX_RESULTS,
            "search_results": [],
            "answer": "",
        }
    
    # ============================================================
    # Evaluation Methods
    # ============================================================
    
    @traceable(name="QAAgent.evaluate_intent_router")
    def evaluate_intent_router(self, query: str) -> IntentRouterResult:
        """
        Evaluate the intent router component.
        
        This method tests whether the agent correctly determines if web search
        is needed for a given query and provides appropriate reasoning.
        
        Args:
            query: User's question or query
            
        Returns:
            IntentRouterResult with classification details
        """
        start_time = time.time()
        
        try:
            result = self.intent_router.invoke(query)
            
            return IntentRouterResult(
                query=query,
                needs_web_search=result.needs_web_search,
                max_results=result.max_results,
                reasoning=result.reasoning,
            )
        except Exception as e:
            # Return default result on error
            return IntentRouterResult(
                query=query,
                needs_web_search=False,
                max_results=DEFAULT_MAX_RESULTS,
                reasoning=f"Error in intent classification: {str(e)}",
            )
    
    @traceable(name="QAAgent.evaluate_max_results")
    def evaluate_max_results(self, query: str) -> MaxResultsResult:
        """
        Evaluate the max_results determination component.
        
        This method checks if the agent selects an appropriate number of
        search results based on query complexity.
        
        Args:
            query: User's question or query
            
        Returns:
            MaxResultsResult with max_results and validation
        """
        try:
            # Get intent classification
            intent_result = self.intent_router.invoke(query)
            
            # Sanitize the max_results value
            max_results = sanitize_max_results(intent_result.max_results)
            
            # Check if value is within valid range
            is_valid = MIN_MAX_RESULTS <= max_results <= MAX_MAX_RESULTS
            
            return MaxResultsResult(
                query=query,
                max_results=max_results,
                is_valid=is_valid,
                reasoning=intent_result.reasoning,
            )
        except Exception as e:
            return MaxResultsResult(
                query=query,
                max_results=DEFAULT_MAX_RESULTS,
                is_valid=True,
                reasoning=f"Error in max_results determination: {str(e)}",
            )
    
    @traceable(name="QAAgent.evaluate_web_search")
    def evaluate_web_search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> WebSearchResult:
        """
        Evaluate the web search component.
        
        This method tests the search functionality and returns the results
        for quality assessment.
        
        Args:
            query: Search query
            max_results: Optional override for max results
            
        Returns:
            WebSearchResult with search results and metadata
        """
        try:
            # Use provided max_results or default
            if max_results is None:
                # Get from intent router
                intent_result = self.intent_router.invoke(query)
                max_results = sanitize_max_results(intent_result.max_results)
            
            # Perform search
            results = self.tavily_client.search(
                query=query,
                max_results=max_results,
            )
            
            return WebSearchResult(
                query=query,
                search_results=results,
                num_results=len(results),
                has_results=len(results) > 0,
            )
        except Exception as e:
            return WebSearchResult(
                query=query,
                search_results=[],
                num_results=0,
                has_results=False,
            )
    
    @traceable(name="QAAgent.evaluate_synthesis")
    def evaluate_synthesis(
        self,
        query: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> SynthesisResult:
        """
        Evaluate the synthesis component.
        
        This method tests how well the agent synthesizes answers from
        provided search results.
        
        Args:
            query: User's question
            search_results: Optional pre-fetched search results
            
        Returns:
            SynthesisResult with answer and source information
        """
        # If no search results provided, fetch them
        if search_results is None:
            web_result = self.evaluate_web_search(query)
            search_results = web_result.search_results
        
        # Create state for synthesis
        state = {
            "query": query,
            "search_results": search_results,
            "answer": "",
        }
        
        try:
            # Perform synthesis
            result = self._synthesize(state)
            answer = result["answer"]
            
            # Extract sources from answer
            sources = extract_sources_from_answer(answer)
            
            return SynthesisResult(
                query=query,
                answer=answer,
                sources_used=sources,
                num_sources=len(sources),
            )
        except Exception as e:
            return SynthesisResult(
                query=query,
                answer=f"Error in synthesis: {str(e)}",
                sources_used=[],
                num_sources=0,
            )
    
    @traceable(name="QAAgent.evaluate_final_answer")
    def evaluate_final_answer(self, query: str) -> FinalAnswerResult:
        """
        Evaluate the final answer component.
        
        This method tests the complete answer generation pipeline
        and returns the final answer for quality assessment.
        
        Args:
            query: User's question
            
        Returns:
            FinalAnswerResult with answer and metadata
        """
        try:
            # Get complete result
            result = self.invoke(query)
            answer = result["answer"]
            
            return FinalAnswerResult(
                query=query,
                answer=answer,
                answer_length=len(answer),
            )
        except Exception as e:
            return FinalAnswerResult(
                query=query,
                answer=f"Error generating answer: {str(e)}",
                answer_length=0,
            )
    
    @traceable(name="QAAgent.evaluate_whole_agent")
    def evaluate_whole_agent(self, query: str) -> WholeAgentResult:
        """
        Evaluate the entire agent pipeline.
        
        This method runs the complete agent and tracks all components
        used in the process.
        
        Args:
            query: User's question
            
        Returns:
            WholeAgentResult with complete pipeline information
        """
        start_time = time.time()
        components_used = ["intent_router"]
        
        try:
            # Get intent classification
            intent_result = self.evaluate_intent_router(query)
            
            if intent_result.needs_web_search:
                components_used.append("web_search")
                components_used.append("synthesis")
                
                # Perform web search
                search_result = self.evaluate_web_search(
                    query,
                    max_results=intent_result.max_results,
                )
                
                # Synthesize answer
                synthesis_result = self.evaluate_synthesis(
                    query,
                    search_results=search_result.search_results,
                )
                
                answer = synthesis_result.answer
                num_search_results = search_result.num_results
            else:
                components_used.append("answer_direct")
                
                # Answer directly
                final_result = self.evaluate_final_answer(query)
                answer = final_result.answer
                num_search_results = 0
            
            total_time = time.time() - start_time
            
            return WholeAgentResult(
                query=query,
                answer=answer,
                needs_web_search=intent_result.needs_web_search,
                max_results=intent_result.max_results,
                num_search_results=num_search_results,
                total_time=total_time,
                components_used=components_used,
            )
        except Exception as e:
            total_time = time.time() - start_time
            
            return WholeAgentResult(
                query=query,
                answer=f"Error in agent pipeline: {str(e)}",
                needs_web_search=False,
                max_results=DEFAULT_MAX_RESULTS,
                num_search_results=0,
                total_time=total_time,
                components_used=components_used,
            )
    
    @traceable(name="QAAgent.invoke")
    def invoke(self, query: str) -> QAState:
        """
        Invoke the complete QA agent.
        
        Args:
            query: User's question or query
            
        Returns:
            Final state with answer and search results
        """
        initial_state = self.create_initial_state(query)
        return self.graph.invoke(initial_state)
    
    def get_answer(self, query: str) -> str:
        """
        Get just the answer string for a query.
        
        Args:
            query: User's question or query
            
        Returns:
            Answer string
        """
        result = self.invoke(query)
        return result["answer"]
    
    # ============================================================
    # Batch Evaluation Methods
    # ============================================================
    
    def evaluate_all_components(
        self,
        queries: List[str],
    ) -> Dict[str, List[Any]]:
        """
        Evaluate all components for multiple queries.
        
        This method is useful for comprehensive evaluation with DeepEval.
        
        Args:
            queries: List of test queries
            
        Returns:
            Dictionary with evaluation results for each component
        """
        results = {
            "intent_router": [],
            "max_results": [],
            "web_search": [],
            "synthesis": [],
            "final_answer": [],
            "whole_agent": [],
        }
        
        for query in queries:
            # Evaluate each component
            results["intent_router"].append(
                self.evaluate_intent_router(query)
            )
            results["max_results"].append(
                self.evaluate_max_results(query)
            )
            
            # Web search and synthesis only if needed
            intent_result = results["intent_router"][-1]
            if intent_result.needs_web_search:
                web_result = self.evaluate_web_search(query)
                results["web_search"].append(web_result)
                
                synthesis_result = self.evaluate_synthesis(
                    query,
                    search_results=web_result.search_results,
                )
                results["synthesis"].append(synthesis_result)
            
            # Final answer and whole agent always
            results["final_answer"].append(
                self.evaluate_final_answer(query)
            )
            results["whole_agent"].append(
                self.evaluate_whole_agent(query)
            )
        
        return results
    
    def evaluate_to_dict(
        self,
        query: str,
        component: str = "whole_agent",
    ) -> Dict[str, Any]:
        """
        Evaluate a specific component and return result as dictionary.
        
        This method is useful for integration with DeepEval metrics.
        
        Args:
            query: User's question
            component: Component to evaluate ('intent_router', 'max_results',
                      'web_search', 'synthesis', 'final_answer', 'whole_agent')
            
        Returns:
            Dictionary with evaluation results
        """
        component_map = {
            "intent_router": self.evaluate_intent_router,
            "max_results": self.evaluate_max_results,
            "web_search": self.evaluate_web_search,
            "synthesis": self.evaluate_synthesis,
            "final_answer": self.evaluate_final_answer,
            "whole_agent": self.evaluate_whole_agent,
        }
        
        if component not in component_map:
            raise ValueError(f"Unknown component: {component}")
        
        result = component_map[component](query)
        return result.to_dict()


# ============================================================
# Evaluation Helper Class for DeepEval Integration
# ============================================================

class QAAgentEvaluator:
    """Helper class for integrating with DeepEval."""
    
    def __init__(self, agent: Optional[QAAgent] = None):
        """
        Initialize evaluator.
        
        Args:
            agent: Optional QAAgent instance
        """
        self.agent = agent or QAAgent()
    
    def evaluate_intent_router(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Evaluate intent router for DeepEval."""
        return self.agent.evaluate_intent_router(query).to_dict()
    
    def evaluate_max_results(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Evaluate max_results for DeepEval."""
        return self.agent.evaluate_max_results(query).to_dict()
    
    def evaluate_web_search(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Evaluate web search for DeepEval."""
        return self.agent.evaluate_web_search(query).to_dict()
    
    def evaluate_synthesis(
        self,
        query: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate synthesis for DeepEval."""
        return self.agent.evaluate_synthesis(
            query,
            search_results,
        ).to_dict()
    
    def evaluate_final_answer(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Evaluate final answer for DeepEval."""
        return self.agent.evaluate_final_answer(query).to_dict()
    
    def evaluate_whole_agent(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Evaluate whole agent for DeepEval."""
        return self.agent.evaluate_whole_agent(query).to_dict()
    
    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        component: str = "whole_agent",
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset for DeepEval.
        
        Args:
            dataset: List of test cases with 'query' key
            component: Component to evaluate
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for item in dataset:
            query = item.get("query", "")
            expected = item.get("expected", None)
            
            result = self.agent.evaluate_to_dict(query, component)
            result["expected"] = expected
            result["test_case"] = item
            
            results.append(result)
        
        return results


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function for testing the QA agent with evaluations."""
    # Initialize agent and evaluator
    agent = QAAgent()
    evaluator = QAAgentEvaluator(agent)
    
    # Test queries
    test_queries = [
        "What is dependency injection?",
        "What is the latest stable version of LangGraph?",
        "Who won the latest FIFA World Cup?",
        "Explain the difference between a list and tuple in Python.",
        "What is the current price of the iPhone 17?",
    ]
    
    print("=" * 80)
    print("QA AGENT EVALUATION TEST")
    print("=" * 80)
    
    # Test each evaluation component
    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("-" * 80)
        
        # 1. Intent Router Evaluation
        print("\n1. INTENT ROUTER:")
        intent_result = evaluator.evaluate_intent_router(query)
        print(f"   Needs web search: {intent_result['needs_web_search']}")
        print(f"   Reasoning: {intent_result['reasoning'][:100]}...")
        
        # 2. Max Results Evaluation
        print("\n2. MAX RESULTS:")
        max_results_result = evaluator.evaluate_max_results(query)
        print(f"   Max results: {max_results_result['max_results']}")
        print(f"   Is valid: {max_results_result['is_valid']}")
        
        # 3. Web Search Evaluation (if needed)
        if intent_result['needs_web_search']:
            print("\n3. WEB SEARCH:")
            web_result = evaluator.evaluate_web_search(query)
            print(f"   Results found: {web_result['num_results']}")
            print(f"   Has results: {web_result['has_results']}")
            
            # 4. Synthesis Evaluation
            print("\n4. SYNTHESIS:")
            synthesis_result = evaluator.evaluate_synthesis(
                query,
                search_results=web_result['search_results'],
            )
            print(f"   Answer length: {len(synthesis_result['answer'])}")
            print(f"   Sources used: {synthesis_result['num_sources']}")
        else:
            print("\n3. WEB SEARCH: Not needed")
            print("\n4. SYNTHESIS: Not needed")
        
        # 5. Final Answer Evaluation
        print("\n5. FINAL ANSWER:")
        final_result = evaluator.evaluate_final_answer(query)
        print(f"   Answer length: {final_result['answer_length']}")
        print(f"   Answer preview: {final_result['answer'][:200]}...")
        
        # 6. Whole Agent Evaluation
        print("\n6. WHOLE AGENT:")
        whole_result = evaluator.evaluate_whole_agent(query)
        print(f"   Components used: {', '.join(whole_result['components_used'])}")
        print(f"   Total time: {whole_result['total_time']:.2f}s")
        print(f"   Answer length: {len(whole_result['answer'])}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()