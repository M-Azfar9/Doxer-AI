"""
DocGen Subagent Package (Feature 1).
"""

from src.subagents.docgen.state import DocGenState, DocIntentPlan, SpecialistScope
from src.subagents.docgen.v1_baseline import DocGenV1BaselinePipeline, build_local_repo_map
from src.subagents.docgen.graph import compile_docgen_subgraph

__all__ = [
    "DocGenState",
    "DocIntentPlan",
    "SpecialistScope",
    "DocGenV1BaselinePipeline",
    "build_local_repo_map",
    "compile_docgen_subgraph"
]
