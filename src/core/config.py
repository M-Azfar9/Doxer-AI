"""
Configuration and Environment Management for DevDocs AI / Sprinter.
Loads API keys, endpoints, and runtime defaults.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv, dotenv_values

# Ensure UTF-8 console output across all platforms (prevents Windows CP1252 charmap errors)
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


class Config:
    """Central configuration class."""

    def __init__(self, env_path: Optional[str] = None):
        if env_path is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent
            env_path = str(workspace_root / ".env")
        
        self.env_path = env_path
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path, override=True)
            self._env_vals = dotenv_values(env_path)
        else:
            self._env_vals = {}

        # Google Gemini API keys (Key failover rotation)
        raw_keys: List[str] = []
        for i in range(1, 9):
            k = (self._env_vals.get(f"GOOGLE_API_KEY_{i}") or os.getenv(f"GOOGLE_API_KEY_{i}") or "").strip().strip('"').strip("'")
            if k and k not in raw_keys:
                raw_keys.append(k)
        
        base_key = (self._env_vals.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
        if base_key and base_key not in raw_keys:
            raw_keys.append(base_key)

        self.google_api_keys: List[str] = raw_keys

        # Other providers
        self.mistral_api_key: str = (self._env_vals.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY") or "").strip().strip('"').strip("'")
        self.openrouter_api_key: str = (self._env_vals.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip().strip('"').strip("'")
        self.tavily_api_key: str = (self._env_vals.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY") or "").strip().strip('"').strip("'")
        self.github_access_token: str = (self._env_vals.get("GITHUB_ACCESS_TOKEN") or os.getenv("GITHUB_ACCESS_TOKEN") or "").strip().strip('"').strip("'")
        self.huggingface_token: str = (self._env_vals.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip().strip('"').strip("'")

        # LangSmith
        self.langsmith_tracing: bool = (os.getenv("LANGSMITH_TRACING", "true").lower() == "true")
        self.langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "doxer_ai")
        self.langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self.langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")

        # Amazon Bedrock
        self.bedrock_model_id: str = (self._env_vals.get("BEDROCK_MODEL_ID") or os.getenv("BEDROCK_MODEL_ID") or "nvidia.nemotron-super-3-120b").strip().strip('"').strip("'")
        self.bedrock_region: str = (self._env_vals.get("AWS_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1").strip().strip('"').strip("'")
        self.bedrock_profile: Optional[str] = (self._env_vals.get("AWS_PROFILE") or os.getenv("AWS_PROFILE") or None)
        if self.bedrock_profile:
            self.bedrock_profile = self.bedrock_profile.strip().strip('"').strip("'")

        # Default paths
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self.chroma_db_dir = str(self.workspace_root / "chroma_code_db")
        self.artifacts_dir = str(self.workspace_root / "artifacts")

        # Default Model Names
        self.default_bedrock_model = self.bedrock_model_id
        self.default_gemini_model = "gemini-3.5-flash-lite"
        self.default_mistral_model = "mistral-small-latest"
        self.default_openrouter_model = "nvidia/nemotron-3.5-lightning:free"


# Global singleton config
config = Config()
