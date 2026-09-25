"""
Global Judge Model Configuration for DevDocs AI / Sprinter Evaluation Framework.
Hierarchy:
  - Primary Model: Gemini 'gemini-3.5-flash-lite' (via project MultiKeyGeminiLLM 8-key rotating pool)
  - Secondary Model (Fallback): OpenRouter 'nvidia/nemotron-3.5-lightning:free'
"""

import os
import sys
import time
import random
import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict, Type, Union

from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure UTF-8 console output across all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from deepeval.models import DeepEvalBaseLLM
from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.core.config import config
from src.core.llm_manager import MultiKeyGeminiLLM

logger = logging.getLogger("LLMAsJudge")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ResilientNemotronJudge(DeepEvalBaseLLM):
    """
    Resilient Judge Model for DeepEval and G-Eval evaluations.
    Primary: Amazon Bedrock 'nvidia.nemotron-super-3-120b' (via ChatBedrockConverse).
    Secondary / Fallback: Google Gemini (MultiKeyGeminiLLM) or OpenRouter Nemotron.
    """

    def __init__(
        self,
        primary_model: str = "nvidia.nemotron-super-3-120b",
        secondary_model: str = "gemini-3.5-flash-lite",
        region_name: str = "us-east-1",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 60.0
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.region_name = region_name or config.bedrock_region
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
        self.using_secondary = False
        
        # 1. Primary: Amazon Bedrock Nemotron Client
        try:
            client_kwargs = {
                "model_id": self.primary_model,
                "region_name": self.region_name,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "disable_streaming": True
            }
            if config.bedrock_profile:
                client_kwargs["credentials_profile_name"] = config.bedrock_profile
            self.primary_client = ChatBedrockConverse(**client_kwargs)
        except Exception as exc:
            logger.warning(f"Failed to initialize primary Bedrock client ({exc}). Defaulting to secondary.")
            self.primary_client = None
            self.using_secondary = True

        # 2. Secondary Fallback Client: MultiKey Gemini or OpenRouter
        if config.google_api_keys:
            self.secondary_client = MultiKeyGeminiLLM(
                keys=config.google_api_keys,
                model=self.secondary_model,
                temperature=self.temperature,
                timeout=self.timeout
            )
        elif self.openrouter_api_key:
            self.secondary_client = ChatOpenAI(
                model="nvidia/nemotron-3.5-lightning:free",
                openai_api_key=self.openrouter_api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=self.temperature,
                request_timeout=self.timeout,
                max_retries=1
            )
        else:
            self.secondary_client = None

        super().__init__(model=self.primary_model)

    def load_model(self) -> Any:
        """DeepEval model loader hook."""
        if self.using_secondary or self.primary_client is None:
            return self.secondary_client
        return self.primary_client

    def get_model_name(self) -> str:
        """Returns active model identifier."""
        if self.using_secondary:
            return f"secondary:{self.secondary_model}"
        return self.primary_model

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Any:
        """
        Executes text generation.
        Tries Primary (Bedrock Nemotron) first, then Secondary fallback.
        """
        if schema is not None:
            return self.generate_with_schema(prompt, schema=schema, **kwargs)

        if not self.using_secondary and self.primary_client is not None:
            try:
                res = self.primary_client.invoke([HumanMessage(content=prompt)])
                return str(res.content)
            except Exception as exc:
                logger.warning(
                    f"Primary Bedrock judge ({self.primary_model}) encountered error ({exc.__class__.__name__}: {exc}). "
                    f"Failing over to secondary judge ({self.secondary_model})..."
                )
                self.using_secondary = True

        # Secondary
        if self.secondary_client is not None:
            try:
                res = self.secondary_client.invoke([HumanMessage(content=prompt)])
                return str(res.content)
            except Exception as exc:
                logger.error(f"Secondary fallback judge also failed: {exc}")
                raise exc

        raise RuntimeError("All judge models (Primary Bedrock and Secondary fallback) failed.")

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None, **kwargs) -> str:
        """Async implementation of generate."""
        return self.generate(prompt, schema=schema, **kwargs)

    def generate_with_schema(self, prompt: str, schema: Type[BaseModel], **kwargs) -> Any:
        """
        Structured generation for DeepEval G-Eval metrics (e.g. ReasonScore).
        Returns a schema instance directly for DeepEval's extract_schema.
        """
        import re

        if not self.using_secondary and self.primary_client is not None:
            try:
                res = self.primary_client.with_structured_output(schema).invoke([HumanMessage(content=prompt)])
                if res is not None and isinstance(res, schema):
                    return res
                # Fallback to direct invoke and parse JSON if structured output returned None
                raw_res = self.primary_client.invoke([HumanMessage(content=prompt)])
                text = str(raw_res.content)
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    return schema(**data)
            except Exception as exc:
                logger.warning(
                    f"Primary Bedrock judge structured call failed ({exc.__class__.__name__}: {exc}). "
                    f"Failing over to secondary judge ({self.secondary_model})..."
                )
                self.using_secondary = True

        # Secondary
        if self.secondary_client is not None:
            try:
                if hasattr(self.secondary_client, "with_structured_output"):
                    res = self.secondary_client.with_structured_output(schema).invoke([HumanMessage(content=prompt)])
                    if res is not None and isinstance(res, schema):
                        return res
                raw_text = self.generate(
                    f"{prompt}\nStrictly respond with a valid JSON object matching schema: {schema.model_json_schema()}"
                )
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    return schema(**data)
            except Exception:
                pass

        raise RuntimeError("All judge models (Primary Bedrock and Secondary fallback) failed.")

    async def a_generate_with_schema(self, prompt: str, schema: Type[BaseModel], **kwargs) -> Any:
        """Async implementation of generate_with_schema."""
        return self.generate_with_schema(prompt, schema, **kwargs)


def get_judge_model(
    primary_model: str = "nvidia.nemotron-super-3-120b",
    secondary_model: str = "gemini-3.5-flash-lite"
) -> ResilientNemotronJudge:
    """Singleton/Factory for the global evaluation judge model."""
    return ResilientNemotronJudge(primary_model=primary_model, secondary_model=secondary_model)


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing ResilientNemotronJudge (Primary: Bedrock Nemotron, Secondary: Gemini)")
    print("=" * 60)
    judge = get_judge_model()
    test_prompt = "Return a JSON object with 'status': 'operational' and 'model': 'gemini-3.5-flash-lite'."
    print(f"Target Model: {judge.get_model_name()}")
    response = judge.generate(test_prompt)
    print(f"Judge Response:\n{response}")
    print(f"Active Model after call: {judge.get_model_name()}")
    print("=" * 60)
