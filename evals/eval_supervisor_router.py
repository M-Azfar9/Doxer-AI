"""
Evaluation script for Phase 7 & 10 Supervisor 3-Way Intent Router.
Evaluates classification accuracy across QA, DocGen, and SRS on a golden dataset.
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
from typing import Dict, Any, List
from collections import defaultdict

from src.supervisor.router import SupervisorRouter
from src.core.service_registry import services


def run_supervisor_router_eval(
    dataset_path: str = "golden_datasets/supervisor_routing_golden.json",
    output_path: str = "evals/supervisor_router_eval_results.json"
) -> Dict[str, Any]:
    print("=" * 65)
    print("🚀 Running Supervisor 3-Way Intent Routing Evaluation")
    print("=" * 65)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    router = SupervisorRouter(services=services)
    total = len(samples)
    correct = 0

    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    detailed_results = []

    start_time = time.time()

    for idx, item in enumerate(samples, 1):
        query = item["query"]
        expected = item["expected_route"]
        cat = item.get("category", "unknown")

        class_stats[expected]["support"] += 1

        t0 = time.time()
        decision = router.route(query)
        latency = round(time.time() - t0, 3)

        predicted = decision.route
        is_correct = (predicted == expected)

        if is_correct:
            correct += 1
            class_stats[expected]["tp"] += 1
            mark = "✓"
        else:
            class_stats[expected]["fn"] += 1
            class_stats[predicted]["fp"] += 1
            mark = "✗"

        print(f"[{idx:02d}/{total:02d}] {mark} Expected: {expected:<8} | Predicted: {predicted:<8} | Latency: {latency}s")
        if not is_correct:
            print(f"     Query: {query[:70]}...")
            print(f"     Reason: {decision.reasoning[:90]}...")

        detailed_results.append({
            "id": item.get("id"),
            "query": query,
            "expected_route": expected,
            "predicted_route": predicted,
            "category": cat,
            "is_correct": is_correct,
            "reasoning": decision.reasoning,
            "latency_s": latency
        })

    total_latency = round(time.time() - start_time, 2)
    overall_accuracy = round(correct / total, 4) if total > 0 else 0.0

    # Calculate per-class metrics
    per_class_metrics = {}
    for cls, stat in class_stats.items():
        tp = stat["tp"]
        fp = stat["fp"]
        fn = stat["fn"]
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0
        per_class_metrics[cls] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": stat["support"]
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": overall_accuracy,
        "total_latency_seconds": total_latency,
        "avg_latency_seconds": round(total_latency / total, 3) if total > 0 else 0.0,
        "per_class_metrics": per_class_metrics,
        "results": detailed_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"📊 SUMMARY: Overall Accuracy: {overall_accuracy * 100:.1f}% ({correct}/{total})")
    for cls, m in per_class_metrics.items():
        print(f"   • {cls.upper():<8} -> Precision: {m['precision']*100:.1f}%, Recall: {m['recall']*100:.1f}%, F1: {m['f1_score']*100:.1f}%")
    print(f"📁 Detailed report saved to: {output_path}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    run_supervisor_router_eval()
