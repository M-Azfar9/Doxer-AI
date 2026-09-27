import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.getcwd())

from src.subagents.docgen import compile_docgen_subgraph, DocGenState
from src.core.service_registry import services

print("--- Testing DocGen Architecture D (v3) End-to-End ---")
graph = compile_docgen_subgraph(services=services, version="v3")

test_input: DocGenState = {
    "user_query": "Document the parse_github_url function in src/github_mcp_client.py including its signature and error handling",
    "local_path": os.getcwd(),
    "repo_url": None,
    "repo_map": "",
    "detected_tech_stack": [],
    "intent_plan": None,
    "specialist_outputs": [],
    "draft_markdown": "",
    "citations": [],
    "error": None
}

t0 = time.time()
result = graph.invoke(test_input)
duration = round(time.time() - t0, 2)

print("\n" + "=" * 60)
print(f"DocGen v3 Completed in {duration}s")
print("=" * 60)
plan = result.get("intent_plan")
if plan:
    print(f"Selected Archetype: {plan.doc_type}")
    print(f"Planning Rationale: {plan.planning_rationale[:120]}...")

report = result.get("grounding_report")
if report:
    print(f"Grounding Audit: {'✅ GROUNDED' if report.is_grounded else '⚠️ UNGROUNDED'} (Confidence: {report.confidence_score})")

final_doc = result.get("final_doc_markdown") or result.get("draft_markdown", "")
print(f"Final Document Length: {len(final_doc.split())} words")
print("\n--- Document Snippet (First 500 chars) ---")
print(final_doc[:500])
print("\n--- End of Snippet ---")
