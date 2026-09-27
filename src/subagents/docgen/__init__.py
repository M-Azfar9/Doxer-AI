"""
DocGen Subagent Package (Feature 1).
Supports Phase 3 Architecture D closed-loop pipeline and Phase 1 baseline.
"""

from src.subagents.docgen.state import (
    DocGenState,
    DocGenPhase3State,
    DocIntentPlan,
    SpecialistScope,
    SufficiencyCheck,
    GroundingReport,
    APIEndpointSymbol,
    MergedEvidenceContext
)
from src.subagents.docgen.v1_baseline import DocGenV1BaselinePipeline, build_local_repo_map
from src.subagents.docgen.v3_architecture_d import DocGenArchitectureDPipeline, ASTSymbolParser
from src.subagents.docgen.graph import compile_docgen_subgraph

__all__ = [
    "DocGenState",
    "DocGenPhase3State",
    "DocIntentPlan",
    "SpecialistScope",
    "SufficiencyCheck",
    "GroundingReport",
    "APIEndpointSymbol",
    "MergedEvidenceContext",
    "DocGenV1BaselinePipeline",
    "DocGenArchitectureDPipeline",
    "ASTSymbolParser",
    "build_local_repo_map",
    "compile_docgen_subgraph"
]
