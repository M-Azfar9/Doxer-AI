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
from langchain_aws import ChatBedrockConverse
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
                    time.sleep(min(3.0, 0.5 * attempts))
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
                    time.sleep(min(3.0, 0.5 * attempts))
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
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.0,
        timeout: float = 120.0
    ):
        if not keys:
            raise ValueError("MultiKeyGeminiLLM requires at least one API key.")
        self.keys = list(keys)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.current_idx = 0
        self.lock = threading.Lock()
        self._disabled_keys = set()
        self._clients: List[Optional[ChatGoogleGenerativeAI]] = [None] * len(keys)

    def _get_client_at(self, idx: int) -> ChatGoogleGenerativeAI:
        if self._clients[idx] is None:
            self._clients[idx] = ChatGoogleGenerativeAI(
                model=self.model,
                api_key=self.keys[idx],
                temperature=self.temperature,
                timeout=self.timeout,
                max_retries=3
            )
        return self._clients[idx]

    def get_active_client(self) -> ChatGoogleGenerativeAI:
        with self.lock:
            # Advance past any permanently disabled keys
            attempts = 0
            while self.current_idx in self._disabled_keys and attempts < len(self.keys):
                self.current_idx = (self.current_idx + 1) % len(self.keys)
                attempts += 1
            return self._get_client_at(self.current_idx)

    def rotate_to_next_key(self, error_desc: str = "") -> ChatGoogleGenerativeAI:
        with self.lock:
            old_idx = self.current_idx
            # If 403 or permission denied, permanently disable this key
            if "permission" in error_desc.lower() or "403" in error_desc:
                self._disabled_keys.add(old_idx)

            # Move to next non-disabled key
            attempts = 0
            while attempts < len(self.keys):
                self.current_idx = (self.current_idx + 1) % len(self.keys)
                attempts += 1
                if self.current_idx not in self._disabled_keys:
                    break

            print(f"🔄 [Key Failover] Gemini Key #{old_idx + 1} ({error_desc}) -> Switching to Key #{self.current_idx + 1}/{len(self.keys)}", flush=True)
            return self._get_client_at(self.current_idx)

    def _is_rate_limit_or_recoverable(self, exc: Exception) -> bool:
        err_str = str(exc).lower()
        err_cls = exc.__class__.__name__.lower()
        triggers = [
            "429", "resource_exhausted", "quota", "rate limit", "ratelimit",
            "503", "unavailable", "high demand", "spikes in demand", "servererror",
            "504", "deadline", "deadline_exceeded",
            "remoteprotocolerror", "disconnected", "403", "permission_denied",
            "404", "not_found", "no longer available", "timeout", "readerror", "connecterror"
        ]
        return any(t in err_str or t in err_cls for t in triggers)

    def invoke(self, messages: Any, **kwargs) -> Any:
        attempts = 0
        max_attempts = len(self.keys) * 3

        while attempts < max_attempts:
            client = self.get_active_client()
            try:
                return client.invoke(messages, **kwargs)
            except Exception as exc:
                attempts += 1
                if self._is_rate_limit_or_recoverable(exc) and attempts < max_attempts:
                    self.rotate_to_next_key(error_desc=exc.__class__.__name__)
                    # Exponential backoff with jitter on 503 / 429
                    sleep_time = min(5.0, 1.0 + (0.5 * attempts))
                    time.sleep(sleep_time)
                else:
                    raise exc
        raise RuntimeError(f"All {len(self.keys)} Gemini API keys exhausted or rate limited.")

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        attempts = 0
        max_attempts = len(self.keys) * 3

        while attempts < max_attempts:
            client = self.get_active_client()
            try:
                return await client.ainvoke(messages, **kwargs)
            except Exception as exc:
                attempts += 1
                if self._is_rate_limit_or_recoverable(exc) and attempts < max_attempts:
                    self.rotate_to_next_key(error_desc=exc.__class__.__name__)
                    sleep_time = min(5.0, 1.0 + (0.5 * attempts))
                    time.sleep(sleep_time)
                else:
                    raise exc
        raise RuntimeError(f"All {len(self.keys)} Gemini API keys exhausted or rate limited.")

    def with_structured_output(self, schema: Any, **kwargs) -> RotatingStructuredRunner:
        return RotatingStructuredRunner(self, schema, kwargs)



class BedrockStructuredRunner:
    """Wraps ChatBedrockConverse.with_structured_output() with fallback to secondary LLM on failure."""

    def __init__(self, parent: "ResilientBedrockLLM", schema: Any, kwargs: Dict[str, Any]):
        self.parent = parent
        self.schema = schema
        self.kwargs = kwargs

    def invoke(self, messages: Any, **invoke_kwargs) -> Any:
        try:
            structured_client = self.parent.client.with_structured_output(self.schema, **self.kwargs)
            return structured_client.invoke(messages, **invoke_kwargs)
        except Exception as exc:
            if self.parent.fallback_llm and self.parent._is_recoverable_error(exc):
                print(f"🔄 [Bedrock Failover] Structured call ({exc.__class__.__name__}: {exc}) -> Failing over to secondary LLM...", flush=True)
                return self.parent.fallback_llm.with_structured_output(self.schema, **self.kwargs).invoke(messages, **invoke_kwargs)
            raise exc

    async def ainvoke(self, messages: Any, **invoke_kwargs) -> Any:
        try:
            structured_client = self.parent.client.with_structured_output(self.schema, **self.kwargs)
            return await structured_client.ainvoke(messages, **invoke_kwargs)
        except Exception as exc:
            if self.parent.fallback_llm and self.parent._is_recoverable_error(exc):
                print(f"🔄 [Bedrock Failover] Structured ainvoke ({exc.__class__.__name__}: {exc}) -> Failing over to secondary LLM...", flush=True)
                fallback = self.parent.fallback_llm.with_structured_output(self.schema, **self.kwargs)
                if hasattr(fallback, "ainvoke"):
                    return await fallback.ainvoke(messages, **invoke_kwargs)
                return fallback.invoke(messages, **invoke_kwargs)
            raise exc


class ResilientBedrockLLM:
    """
    Primary LangChain LLM wrapper for Amazon Bedrock (NVIDIA Nemotron Super 120B).
    Provides automatic fallback to secondary MultiKeyGeminiLLM or ChatMistralAI on throttles or outages.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        region_name: Optional[str] = None,
        temperature: float = 0.0,
        fallback_llm: Optional[Any] = None,
        **kwargs
    ):
        self.model = model or config.default_bedrock_model
        self.region_name = region_name or config.bedrock_region
        self.temperature = temperature
        self.fallback_llm = fallback_llm

        client_kwargs: Dict[str, Any] = {
            "model_id": self.model,
            "region_name": self.region_name,
            "temperature": self.temperature,
            "disable_streaming": True,
            **kwargs
        }
        if config.bedrock_profile:
            client_kwargs["credentials_profile_name"] = config.bedrock_profile

        self.client = ChatBedrockConverse(**client_kwargs)

    def _is_recoverable_error(self, exc: Exception) -> bool:
        err_str = str(exc).lower()
        err_cls = exc.__class__.__name__.lower()
        triggers = [
            "throttling", "throttlingexception", "too many tokens", "rate limit", "ratelimit",
            "429", "503", "500", "serviceunavailable", "timeout", "connectionerror",
            "endpointconnectionerror", "connecttimeouterror", "readtimeouterror",
            "validationexception", "modelnotfound", "accessdenied"
        ]
        return any(t in err_str or t in err_cls for t in triggers)

    def invoke(self, messages: Any, **kwargs) -> Any:
        try:
            return self.client.invoke(messages, **kwargs)
        except Exception as exc:
            if self.fallback_llm and self._is_recoverable_error(exc):
                print(f"🔄 [Bedrock Failover] ({exc.__class__.__name__}: {exc}) -> Switching to secondary fallback LLM...", flush=True)
                return self.fallback_llm.invoke(messages, **kwargs)
            raise exc

    async def ainvoke(self, messages: Any, **kwargs) -> Any:
        try:
            return await self.client.ainvoke(messages, **kwargs)
        except Exception as exc:
            if self.fallback_llm and self._is_recoverable_error(exc):
                print(f"🔄 [Bedrock Failover] ({exc.__class__.__name__}: {exc}) -> Switching to secondary fallback LLM...", flush=True)
                if hasattr(self.fallback_llm, "ainvoke"):
                    return await self.fallback_llm.ainvoke(messages, **kwargs)
                return self.fallback_llm.invoke(messages, **kwargs)
            raise exc

    def with_structured_output(self, schema: Any, **kwargs) -> BedrockStructuredRunner:
        return BedrockStructuredRunner(self, schema, kwargs)


def get_secondary_fallback_llm(temperature: float = 0.0) -> Optional[Any]:
    """Helper returning secondary fallback LLM instance if configured."""
    if config.google_api_keys:
        return MultiKeyGeminiLLM(
            keys=config.google_api_keys,
            model=config.default_gemini_model,
            temperature=temperature
        )
    elif config.mistral_api_key:
        return ChatMistralAI(
            model=config.default_mistral_model,
            api_key=config.mistral_api_key,
            temperature=temperature
        )
    elif config.openrouter_api_key:
        try:
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                model=config.default_openrouter_model,
                api_key=config.openrouter_api_key,
                temperature=temperature
            )
        except ImportError:
            pass
    return None


def get_default_llm(temperature: float = 0.0, model: Optional[str] = None) -> Any:
    """
    Factory function returning the best available resilient LLM instance:
    1. If specific non-Bedrock model requested (e.g. Gemini / Mistral), route directly.
    2. ResilientBedrockLLM (NVIDIA Nemotron Super 120B on Bedrock) as primary with secondary fallback.
    3. MultiKeyGeminiLLM if Bedrock fails to initialize and Google keys exist.
    4. ChatMistralAI if Mistral key is present.
    5. ChatOpenRouter as fallback.
    """
    if model and "gemini" in model.lower() and config.google_api_keys:
        return MultiKeyGeminiLLM(
            keys=config.google_api_keys,
            model=model,
            temperature=temperature
        )
    if model and "mistral" in model.lower() and config.mistral_api_key:
        return ChatMistralAI(
            model=model,
            api_key=config.mistral_api_key,
            temperature=temperature
        )

    fallback = get_secondary_fallback_llm(temperature=temperature)
    try:
        return ResilientBedrockLLM(
            model=model or config.default_bedrock_model,
            temperature=temperature,
            fallback_llm=fallback
        )
    except Exception as exc:
        print(f"⚠️ [LLMManager] Bedrock initialization error ({exc}). Falling back to secondary...", flush=True)
        if fallback:
            return fallback
        raise exc

