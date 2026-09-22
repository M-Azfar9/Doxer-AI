"""
Strict Sandboxing & Path Boundary Security for DevDocs AI CLI.
Ensures that all file operations and code reading activities are strictly restricted
to the user-approved repository and local directory boundary.
"""

import os
from pathlib import Path
from typing import Optional, List, Union


class SandboxSecurityError(PermissionError):
    """Raised when an operation attempts to access a path outside the allowed sandbox boundary."""
    pass


class SandboxManager:
    """
    Enforces strict path containment rules.
    Prevents path traversal, symlink escapes, and unauthorized directory inspection.
    """

    def __init__(self, allowed_directory: Union[str, Path]):
        self.allowed_directory = Path(allowed_directory).resolve()
        if not self.allowed_directory.exists():
            raise FileNotFoundError(f"Allowed directory does not exist: {self.allowed_directory}")
        if not self.allowed_directory.is_dir():
            raise NotADirectoryError(f"Allowed path is not a directory: {self.allowed_directory}")

    @property
    def base_path(self) -> Path:
        return self.allowed_directory

    def is_safe_path(self, target_path: Union[str, Path]) -> bool:
        """
        Returns True if target_path resolves strictly within the allowed directory boundary.
        """
        try:
            self.validate_path(target_path)
            return True
        except SandboxSecurityError:
            return False

    def validate_path(self, target_path: Union[str, Path]) -> Path:
        """
        Resolves target_path and verifies it resides inside the sandbox root.
        Raises SandboxSecurityError if the path escapes the sandbox.
        """
        path_obj = Path(target_path)
        if not path_obj.is_absolute():
            resolved = (self.allowed_directory / path_obj).resolve()
        else:
            resolved = path_obj.resolve()

        # Handle Windows drive letters & path canonicalization
        try:
            common = Path(os.path.commonpath([str(self.allowed_directory), str(resolved)])).resolve()
            if common != self.allowed_directory:
                raise SandboxSecurityError(
                    f"Access Denied: Path '{target_path}' resolves to '{resolved}', "
                    f"which is outside the allowed sandbox directory '{self.allowed_directory}'."
                )
        except ValueError as exc:
            # Raised by os.path.commonpath when paths are on different drives on Windows (e.g. C: vs D:)
            raise SandboxSecurityError(
                f"Access Denied: Path '{target_path}' is on a different drive than the allowed sandbox '{self.allowed_directory}'."
            ) from exc

        return resolved

    def safe_read_text(self, file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Safely reads a text file within the sandbox."""
        validated = self.validate_path(file_path)
        if not validated.exists():
            raise FileNotFoundError(f"File not found: {validated}")
        if not validated.is_file():
            raise IsADirectoryError(f"Path is not a regular file: {validated}")
        return validated.read_text(encoding=encoding, errors="replace")

    def safe_list_files(self, sub_dir: Optional[Union[str, Path]] = None, max_files: int = 100) -> List[Path]:
        """Safely lists files within a subdirectory of the sandbox."""
        target_dir = self.validate_path(sub_dir) if sub_dir else self.allowed_directory
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Target is not a directory: {target_dir}")

        files: List[Path] = []
        for root, dirs, filenames in os.walk(target_dir):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "venv", ".venv")]
            for f in filenames:
                if not f.startswith("."):
                    files.append(Path(root) / f)
                    if len(files) >= max_files:
                        return files
        return files

    def update_allowed_directory(self, new_dir: Union[str, Path]):
        """Updates the allowed sandbox directory after validation."""
        new_path = Path(new_dir).resolve()
        if not new_path.exists():
            raise FileNotFoundError(f"New directory does not exist: {new_path}")
        if not new_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {new_path}")
        self.allowed_directory = new_path
