"""
GitHub MCP Client Module for DevDocs AI (Phase 5).

Connects to GitHub's Model Context Protocol (MCP) server (@modelcontextprotocol/server-github)
via langchain-mcp-adapters with an automatic, resilient GitHub REST API fallback to ensure
zero-hang execution across all platforms (Windows/Linux/macOS).
"""

import os
import sys
import json
import asyncio
import urllib.request
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient

DEFAULT_IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv",
    "env", ".env", "dist", "build", ".idea", ".vscode", ".egg-info", "target",
    "bin", "obj", ".mypy_cache", ".ruff_cache", "coverage", ".next", ".nuxt"
}

DEFAULT_IGNORED_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".mp4", ".mp3",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".pyd", ".pkl", ".joblib", ".bin", ".weights", ".h5", ".onnx",
    ".pdf", ".docx", ".xlsx", ".csv", ".tsv", ".sqlite", ".db", ".lock"
}

MAX_FILE_SIZE_BYTES = 80 * 1024  # 80 KB limit per file fetch


def parse_github_url(url: str) -> Tuple[str, str, Optional[str]]:
    """Extract owner, repo, and optional branch from GitHub URL."""
    cleaned = url.strip()
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) >= 2:
            return parts[0], parts[1], None
        raise ValueError(f"Invalid GitHub identifier: '{url}'. Expected 'owner/repo'.")

    parsed = urlparse(cleaned)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: '{url}'. Missing owner and repository.")

    owner, repo = path_parts[0], path_parts[1]
    branch = path_parts[3] if len(path_parts) >= 4 and path_parts[2] == "tree" else None
    return owner, repo, branch


class GitHubMCPClient:
    """
    Resilient GitHub Client:
    - Primary: GitHub MCP Server (@modelcontextprotocol/server-github) via langchain-mcp-adapters.
    - Fallback: Direct GitHub REST API (prevents hangs if npx or stdio is delayed).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        ignored_dirs: Optional[set[str]] = None,
        ignored_exts: Optional[set[str]] = None,
        max_file_size: int = MAX_FILE_SIZE_BYTES
    ):
        raw_token = token or os.getenv("GITHUB_ACCESS_TOKEN") or os.getenv("GITHUB_TOKEN") or ""
        self.token = raw_token.strip().strip('"').strip("'")
        self.ignored_dirs = ignored_dirs or DEFAULT_IGNORED_DIRS
        self.ignored_exts = ignored_exts or DEFAULT_IGNORED_EXTS
        self.max_file_size = max_file_size
        self._mcp_client: Optional[MultiServerMCPClient] = None
        self._tools_cache: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    def _get_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "DevDocs-AI-Agent", "Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    async def _fetch_rest(self, owner: str, repo: str, path: str = "") -> Any:
        """Direct GitHub REST API call with instant timeout."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        req = urllib.request.Request(url, headers=self._get_headers())
        loop = asyncio.get_event_loop()

        def _call():
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        return await loop.run_in_executor(None, _call)

    async def get_tools(self) -> Dict[str, Any]:
        """Initialize MCP client with full os.environ so PATH, SYSTEMROOT, and APPDATA are preserved."""
        if self._tools_cache is None:
            async with self._lock:
                if self._tools_cache is None:
                    full_env = dict(os.environ)
                    if self.token:
                        full_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.token

                    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
                    self._mcp_client = MultiServerMCPClient({
                        "github": {
                            "transport": "stdio",
                            "command": npx_cmd,
                            "args": ["-y", "@modelcontextprotocol/server-github"],
                            "env": full_env
                        }
                    })
                    tools = await asyncio.wait_for(self._mcp_client.get_tools(), timeout=12.0)
                    self._tools_cache = {t.name: t for t in tools}
        return self._tools_cache

    async def fetch_file_or_dir(self, owner: str, repo: str, path: str = "") -> Any:
        """Fetch file metadata or directory list using MCP with automatic REST fallback."""
        try:
            tools = await self.get_tools()
            tool = tools.get("get_file_contents")
            if tool:
                res = await asyncio.wait_for(
                    tool.ainvoke({"owner": owner, "repo": repo, "path": path}),
                    timeout=8.0
                )
                text = res[0]["text"] if isinstance(res, list) and res and "text" in res[0] else str(res)
                return json.loads(text)
        except Exception:
            pass  # Fall through to direct REST API

        return await self._fetch_rest(owner, repo, path)

    async def fetch_repo_tree(
        self,
        owner: str,
        repo: str,
        branch: Optional[str] = None,
        max_depth: int = 2,
        max_files: int = 60
    ) -> List[str]:
        """Recursively list repo files filtering non-code and lock files."""
        collected: List[str] = []

        async def _traverse(path: str = "", depth: int = 0):
            if depth > max_depth or len(collected) >= max_files:
                return

            try:
                entries = await self.fetch_file_or_dir(owner, repo, path)
            except Exception:
                return

            if not isinstance(entries, list):
                return

            tasks = []
            for entry in entries:
                if len(collected) >= max_files:
                    break
                t = entry.get("type")
                p = entry.get("path", "")
                name = entry.get("name", "")
                if t == "file":
                    ext = os.path.splitext(name)[1].lower()
                    if ext not in self.ignored_exts and name not in self.ignored_dirs:
                        collected.append(p)
                elif t == "dir":
                    if name not in self.ignored_dirs and not name.startswith("."):
                        tasks.append(_traverse(p, depth + 1))

            if tasks:
                await asyncio.gather(*tasks)

        await _traverse("", depth=0)
        return sorted(collected)

    async def fetch_file_content(
        self,
        owner: str,
        repo: str,
        file_path: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch raw file text content."""
        try:
            data = await self.fetch_file_or_dir(owner, repo, file_path)
            content = data.get("content", "") if isinstance(data, dict) else str(data)
            truncated = len(content) > self.max_file_size
            if truncated:
                content = content[:self.max_file_size] + "\n\n... [Truncated due to size limit] ..."
            return {"file_path": file_path, "content": content, "truncated": truncated}
        except Exception as e:
            return {"file_path": file_path, "content": f"[Error reading file: {e}]", "truncated": False}
