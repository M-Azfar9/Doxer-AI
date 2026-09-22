"""
Resilient LLM Manager with Multi-Key Failover and Provider Fallbacks.
Supports automatic key rotation across multiple Google Gemini API keys on 429,
as well as Mistral and OpenRouter fallbacks.
"""

import time
import threading
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

from src.core.config import config


class RotatingStructuredRunner:
    """Wraps .with_structured_output() with automatic key rotation on 429/rate limits."""

    def __init__(self, parent: "MultiKeyGeminiLLM", schema, kwargs: Dict[str, Any]):
        self.parent = parent
        self.schema = schema
        self.kwargs = kwargs

    def invoke(self, messages: Any, **invoke_kwargs) -> Any:
        attempts = 0
        max_attempts = len(self.parent.keys) * 2

        while attempts < max_attempts:
            client = self.parent.get_active_client()
            try:
                structured_client = client.with_structured_output(self.schema, **self.kwargs)
                return structured_client.invoke(messages, **invoke_kwargs)
            except Exception as exc:
                attempts += 1
                if self.parent._is_rate_limit_or_recoverable(exc) and attempts < max_attempts:
                    self.parent.rotate_to_next_key(error_desc=exc.__class__.__name__)
                    time.sleep(1.0)
                else:
                    raise exc
        raise RuntimeError(f"All Gemini API keys exhausted or rate limited on structured output.")

    async def ainvoke(self, messages: Any, **invoke_kwargs) -> Any:
        attempts = 0
        max_attempts = len(self.parent.keys) * 2

        while attempts < max_attempts:
            client = self.parent.get_active_client()
            try:
                structured_client = client.with_structured_output(self.schema, **self.kwargs)
                return await structured_client.ainvoke(messages, **invoke_kwargs)
            except Exception as exc:
                attempts += 1
                if self.parent._is_rate_limit_or_recoverable(exc) and attempts < max_attempts:
                    self.parent.rotate_to_next_key(error_desc=exc.__class__.__name__)
                    time.sleep(1.0)
                else:
                    raise exc
        raise RuntimeError(f"All Gemini API keys exhausted or rate limited on structured output.")


class MultiKeyGeminiLLM:
    """
    Thread-safe LangChain LLM wrapper that rotates Google API keys
    automatically when hitting 429 rate limits, quotas, or transient disconnects.
    """

    def __init__(
        self,
        keys: List[str],
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        timeout: float = 120.0
    ):
        if not keys:
            raise ValueError("MultiKeyGeminiLLM requires at least one API key.")
        self.keys = keys
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.current_idx = 0
        self.lock = threading.Lock()
        self._clients: List[Optional[ChatGoogleGenerativeAI]] = [None] * len(keys)

    def _get_client_at(self, idx: int) -> ChatGoogleGenerativeAI:
        if self._clients[idx] is None:
            self._clients[idx] = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.keys[idx],
                temperature=self.temperature,
                request_timeout=self.timeout
            )
        return self._clients[idx]

    def get_active_client(self) -> ChatGoogleGenerativeAI:
        with self.lock:
            return self._get_client_at(self.current_idx)

    def rotate_to_next_key(self, error_desc: str = "") -> ChatGoogleGenerativeAI:
        with self.lock:
            old_idx = self.current_idx
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            print(f"🔄 [Key Failover] Gemini Key #{old_idx + 1} ({error_desc}) -> Switching to Key #{self.current_idx + 1}/{len(self.keys)}")
            return self._get_client_at(self.current_idx)

    def _is_rate_limit_or_recoverable(self, exc: Exception) -> bool:
        err_str = str(exc).lower()
        err_cls = exc.__class__.__name__.lower()
        triggers = [
            "429", "resource_exhausted", "quota", "rate limit", "ratelimit",
            "remoteprotocolerror", "disconnected", "403", "permission_denied",
            "404", "not_found", "no longer available"
        ]
        return any(t in err_str or t in err_cls for t in triggers)

    def invoke(self, messages: Any, **kwargs) -> Any:
        attempts = 0
        max_attempts = len(self.keys) * 2

        while attempts < max_attempts:
            client = self.get_active_client()
            try:
                return client.invoke(messages, **kwargs)
            except Exception as exc:
                attempts += 1
                if self._is_rate_limit_or_recoverable(exc) and attempts < max_attempts:
                    self.rotate_to_next_key(error_desc=exc.__class__.__name__)
                    time.sleep(1.0)
                else:
                    raise exc
        raise RuntimeError(f"All {len(self.keys)} Gemini API keys exhausted or rate limited.")

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        attempts = 0
        max_attempts = len(self.keys) * 2

        while attempts < max_attempts:
            client = self.get_active_client()
            try:
                return await client.ainvoke(messages, **kwargs)
            except Exception as exc:
                attempts += 1
                if self._is_rate_limit_or_recoverable(exc) and attempts < max_attempts:
                    self.rotate_to_next_key(error_desc=exc.__class__.__name__)
                    time.sleep(1.0)
                else:
                    raise exc
        raise RuntimeError(f"All {len(self.keys)} Gemini API keys exhausted or rate limited.")

    def with_structured_output(self, schema: Any, **kwargs) -> RotatingStructuredRunner:
        return RotatingStructuredRunner(self, schema, kwargs)


def get_default_llm(temperature: float = 0.0, model: Optional[str] = None) -> Any:
    """
    Factory function returning the best available resilient LLM instance:
    1. MultiKeyGeminiLLM if Google API keys are present.
    2. ChatMistralAI if Mistral key is present.
    3. ChatOpenRouter as fallback.
    """
    if config.google_api_keys:
        return MultiKeyGeminiLLM(
            keys=config.google_api_keys,
            model=model or config.default_gemini_model,
            temperature=temperature
        )
    elif config.mistral_api_key:
        return ChatMistralAI(
            model=model or config.default_mistral_model,
            api_key=config.mistral_api_key,
            temperature=temperature
        )
    elif config.openrouter_api_key:
        try:
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                model=model or config.default_openrouter_model,
                api_key=config.openrouter_api_key,
                temperature=temperature
            )
        except ImportError:
            pass
    raise ValueError("No valid LLM API keys found in configuration (Google, Mistral, or OpenRouter).")
