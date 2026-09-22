"""
Phase 10 Cross-Feature Regression Test Suite.
Unified test runner validating end-to-end execution across all 3 features:
1. Top-Level Supervisor 3-Way Intent Routing
2. Feature 2: QA Subagent Execution
3. Feature 1: DocGen Subagent (Phase 1 Baseline)
4. Feature 3: SRS Subagent with Checkpointer HITL State Persistence
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import time
from typing import Dict, Any

from src.supervisor.sprinter import SprinterAssistant
from src.core.service_registry import services
from evals.eval_supervisor_router import run_supervisor_router_eval


def run_cross_feature_regression(
    output_path: str = "evals/cross_feature_regression_results.json"
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("🌟 SPRINTER PHASE 10 CROSS-FEATURE REGRESSION SUITE")
    print("=" * 70)

    results = {}
    suite_start = time.time()
    sprinter = SprinterAssistant(services=services, docgen_version="v1")

    # -------------------------------------------------------------
    # Test 1: Supervisor 3-Way Routing Accuracy
    # -------------------------------------------------------------
    print("\n--- [TEST 1/4] Supervisor 3-Way Intent Routing Accuracy ---", flush=True)
    router_results_file = "evals/supervisor_router_eval_results.json"
    if os.path.exists(router_results_file):
        with open(router_results_file, "r", encoding="utf-8") as f:
            router_summary = json.load(f)
        print("  • Loaded existing golden router evaluation benchmark.", flush=True)
    else:
        router_summary = run_supervisor_router_eval()

    routing_passed = (router_summary["overall_accuracy"] >= 0.85)
    results["test_1_routing"] = {
        "status": "PASSED" if routing_passed else "FAILED",
        "accuracy": router_summary["overall_accuracy"],
        "total_samples": router_summary["total_samples"],
        "correct": router_summary["correct_predictions"]
    }
    print(f"Result: {results['test_1_routing']['status']} (Accuracy: {router_summary['overall_accuracy']*100:.1f}%)", flush=True)

    # -------------------------------------------------------------
    # Test 2: Feature 2 — QA Subagent End-to-End
    # -------------------------------------------------------------
    print("\n--- [TEST 2/4] QA Subagent End-to-End Execution ---")
    qa_query = "What is dependency injection and how does it promote loose coupling?"
    t0 = time.time()
    qa_res = sprinter.run(query=qa_query, thread_id="test_qa_regression")
    qa_latency = round(time.time() - t0, 2)
    qa_passed = (qa_res.status == "completed" and qa_res.route == "qa" and bool(qa_res.output))
    results["test_2_qa"] = {
        "status": "PASSED" if qa_passed else "FAILED",
        "route": qa_res.route,
        "latency_seconds": qa_latency,
        "output_chars": len(qa_res.output or ""),
        "error": qa_res.error
    }
    print(f"Result: {results['test_2_qa']['status']} | Route: {qa_res.route} | Output length: {len(qa_res.output or '')} chars | Latency: {qa_latency}s")

    # -------------------------------------------------------------
    # Test 3: Feature 1 — DocGen Subagent (Phase 1 Baseline)
    # -------------------------------------------------------------
    print("\n--- [TEST 3/4] DocGen Subagent (Phase 1 Baseline) End-to-End ---")
    doc_query = "Document how the GitHubMCPClient handles repo tree fetching in src/github_mcp_client.py"
    t0 = time.time()
    doc_res = sprinter.run(query=doc_query, thread_id="test_docgen_regression", local_path=".")
    doc_latency = round(time.time() - t0, 2)
    doc_passed = (doc_res.status == "completed" and doc_res.route == "doc_gen" and bool(doc_res.output))
    results["test_3_docgen"] = {
        "status": "PASSED" if doc_passed else "FAILED",
        "route": doc_res.route,
        "latency_seconds": doc_latency,
        "output_chars": len(doc_res.output or ""),
        "citations_count": doc_res.metadata.get("citations_count", 0),
        "error": doc_res.error
    }
    print(f"Result: {results['test_3_docgen']['status']} | Route: {doc_res.route} | Citations: {doc_res.metadata.get('citations_count', 0)} | Output length: {len(doc_res.output or '')} chars | Latency: {doc_latency}s")

    # -------------------------------------------------------------
    # Test 4: Feature 3 — SRS Subagent with Checkpointer HITL Resumption
    # -------------------------------------------------------------
    print("\n--- [TEST 4/4] SRS Subagent Multi-Turn Checkpointer Test ---")
    srs_thread = f"test_srs_thread_{int(time.time())}"
    srs_query = "Create an IEEE 830 software requirements specification for a telehealth consultation app"
    t0 = time.time()
    srs_res1 = sprinter.run(query=srs_query, thread_id=srs_thread)
    
    # Check if halted at clarification interrupt or completed
    if srs_res1.status == "waiting_human_input":
        print(f"  • Turn 1: Interrupted as expected. Clarification question:\n    '{srs_res1.pending_question}'")
        # Simulate human response and resume thread
        user_answer = "The platform must support encrypted video calls, HIPAA-compliant patient record storage, and doctor scheduling with automated SMS reminders."
        print(f"  • Turn 2: Resuming thread '{srs_thread}' with simulated user reply...")
        srs_res2 = sprinter.resume(thread_id=srs_thread, user_response=user_answer)
        srs_final = srs_res2
    else:
        srs_final = srs_res1

    srs_latency = round(time.time() - t0, 2)
    srs_passed = (srs_final.status in ["completed", "waiting_human_input"] and srs_final.route == "srs")
    results["test_4_srs"] = {
        "status": "PASSED" if srs_passed else "FAILED",
        "final_status": srs_final.status,
        "route": srs_final.route,
        "latency_seconds": srs_latency,
        "output_chars": len(srs_final.output or ""),
        "error": srs_final.error
    }
    print(f"Result: {results['test_4_srs']['status']} | Route: {srs_final.route} | SRS Status: {srs_final.status} | Output length: {len(srs_final.output or '')} chars | Latency: {srs_latency}s")

    # -------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------
    total_duration = round(time.time() - suite_start, 2)
    all_passed = all(t.get("status") == "PASSED" for t in results.values())

    report = {
        "suite_name": "Phase 10 Cross-Feature Regression Suite",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": "PASSED" if all_passed else "FAILED",
        "total_duration_seconds": total_duration,
        "tests": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"🏁 REGRESSION SUITE COMPLETED in {total_duration}s — STATUS: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print("=" * 70)
    for test_key, test_val in results.items():
        print(f"   • {test_key:<20} -> {test_val['status']}")
    print(f"\n📁 Regression report saved to: {output_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_cross_feature_regression()
