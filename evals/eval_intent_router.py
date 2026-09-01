"""
Intent Router Evaluation Script
Evaluates the intent router component using DeepEval with G-Eval metric.
Uses Mistral Medium 3.5 as the LLM judge.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the agent evaluator
from src.simple_search_agent import QAAgentEvaluator

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import GEval
try:
    from deepeval.test_case import LLMTestCase, SingleTurnParams as EvaluationParams
except ImportError:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams as EvaluationParams
from deepeval.dataset import EvaluationDataset

# Configure Mistral as the LLM judge
from deepeval.models.base_model import DeepEvalBaseLLM
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral


class MistralJudge(DeepEvalBaseLLM):
    """Custom Mistral LLM wrapper for DeepEval"""
    
    def __init__(self, model_name: str = "mistral-medium-latest"):
        self.model_name = model_name
        super().__init__(model=model_name)
    
    def load_model(self):
        """Load Mistral client"""
        return Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    def generate(self, prompt: str) -> str:
        """Generate response from Mistral"""
        try:
            chat_response = self.model.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def a_generate(self, prompt: str) -> str:
        """Async generate method"""
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.model_name


def load_golden_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load golden dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset.get('goldens', [])


def create_intent_router_test_cases(
    golden_data: List[Dict[str, Any]], 
    evaluator: Optional[QAAgentEvaluator] = None
) -> List[LLMTestCase]:
    """Create DeepEval test cases from golden dataset"""
    if evaluator is None:
        evaluator = QAAgentEvaluator()
        
    test_cases = []
    
    for golden in golden_data:
        query = golden['input']
        expected_route = golden['expected_route']
        
        # Get actual result from intent router
        try:
            actual_result = evaluator.evaluate_intent_router(query)
            
            # Extract the classification
            actual_needs_web_search = actual_result.get('needs_web_search', False)
            actual_reasoning = actual_result.get('reasoning', '')
            
            # Create LLMTestCase
            test_case = LLMTestCase(
                input=query,
                actual_output=f"needs_web_search: {actual_needs_web_search}, reasoning: {actual_reasoning}",
                expected_output=f"needs_web_search: {expected_route}",
                context=[
                    f"Category: {golden.get('category', 'unknown')}",
                    f"Difficulty: {golden.get('difficulty', 'unknown')}",
                    f"Tags: {', '.join(golden.get('tags', []))}",
                    f"ID: {golden['id']}"
                ],
                additional_metadata={
                    'golden_id': golden['id'],
                    'expected_route': expected_route,
                    'actual_route': actual_needs_web_search,
                    'category': golden.get('category', 'unknown'),
                    'difficulty': golden.get('difficulty', 'unknown'),
                    'tags': golden.get('tags', [])
                }
            )
            test_cases.append(test_case)
            
        except Exception as e:
            print(f"Error processing golden {golden.get('id', 'unknown')}: {str(e)}")
            continue
    
    return test_cases


def create_route_accuracy_metric(model_name: str = "mistral-medium-latest"):
    """Create G-Eval metric for Route Accuracy evaluation"""
    
    evaluation_criteria = """
    You are evaluating the accuracy of an intent router system that determines whether a user query requires web search or can be answered with direct knowledge.

    Evaluation Criteria:
    1. Route Classification Accuracy: Does the system correctly classify whether web search is needed?
    2. Reasoning Quality: Is the reasoning provided logical, relevant, and aligned with the classification?
    3. Policy Adherence: Does the classification follow these routing policies?
       - Use web search (True) for: current events, real-time data, recent information, changing facts, external verification, live data needs
       - Direct answer (False) for: stable knowledge, reasoning tasks, explanations, transformations, writing tasks, static facts
    4. Context Appropriateness: Does the classification consider the nature of the query appropriately?

    Scoring Rubric:
    - Score 1.0: Perfect classification with excellent reasoning that clearly explains why web search is or isn't needed
    - Score 0.8: Correct classification with good reasoning that mostly explains the decision well
    - Score 0.6: Correct classification but reasoning is weak, unclear, or partially misaligned
    - Score 0.4: Incorrect classification but reasoning shows some understanding of routing principles
    - Score 0.2: Incorrect classification with poor reasoning that contradicts routing policies
    - Score 0.0: Completely wrong classification with no valid reasoning or understanding
    """
    
    metric = GEval(
        name="Route Accuracy",
        criteria=evaluation_criteria,
        evaluation_params=[
            EvaluationParams.ACTUAL_OUTPUT,
            EvaluationParams.EXPECTED_OUTPUT,
            EvaluationParams.CONTEXT
        ],
        threshold=0.7,
        model=MistralJudge(model_name=model_name),
        verbose_mode=True
        
    )
    
    return metric


def main():
    """Main evaluation function"""
    
    # Configuration
    DATASET_PATH = str(PROJECT_ROOT / "golden_datasets" / "intent_router.json")
    OUTPUT_DIR = str(PROJECT_ROOT / "evals")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for Mistral API key
    if not os.getenv("MISTRAL_API_KEY"):
        print("⚠️  Warning: MISTRAL_API_KEY not found in environment variables")
        print("Please set it using: set MISTRAL_API_KEY=your_api_key_here")
        return
    
    print("🚀 Starting Intent Router Evaluation")
    print("=" * 50)
    
    # Load golden dataset
    print("📂 Loading golden dataset...")
    golden_data = load_golden_dataset(DATASET_PATH)
    print(f"   Loaded {len(golden_data)} golden test cases")
    
    # Create evaluator and test cases
    print("🔄 Initializing agent evaluator and creating DeepEval test cases...")
    evaluator = QAAgentEvaluator()
    test_cases = create_intent_router_test_cases(golden_data, evaluator=evaluator)
    print(f"   Created {len(test_cases)} test cases")
    
    # Create evaluation dataset (optional container for test cases)
    eval_dataset = EvaluationDataset()
    eval_dataset.test_cases = test_cases
    
    # Create metric
    print("📊 Setting up G-Eval metric with Mistral judge...")
    route_accuracy_metric = create_route_accuracy_metric()
    
    # Run evaluation
    print("🏃 Running evaluation...")
    results = evaluate(
        test_cases=test_cases,
        metrics=[route_accuracy_metric]
    )
    
    # Save results
    print("💾 Saving evaluation results...")
    results_file = os.path.join(OUTPUT_DIR, "intent_router_eval_results.json")
    
    metrics_summary = {}
    test_results_summary = []
    total_cases = len(results.test_results) if hasattr(results, 'test_results') else len(test_cases)
    
    if hasattr(results, 'test_results'):
        for i, tr in enumerate(results.test_results):
            case_metrics = {}
            if tr.metrics_data:
                for m in tr.metrics_data:
                    case_metrics[m.name] = {
                        "score": m.score,
                        "success": m.success,
                        "threshold": m.threshold,
                        "reason": m.reason,
                        "error": m.error,
                    }
                    if m.name not in metrics_summary:
                        metrics_summary[m.name] = {
                            "total_score": 0.0,
                            "count": 0,
                            "success_count": 0,
                            "threshold": m.threshold
                        }
                    if m.score is not None:
                        metrics_summary[m.name]["total_score"] += m.score
                        metrics_summary[m.name]["count"] += 1
                    if m.success:
                        metrics_summary[m.name]["success_count"] += 1
            
            test_results_summary.append({
                "test_case_id": i + 1,
                "input": tr.input,
                "actual_output": tr.actual_output,
                "expected_output": tr.expected_output,
                "success": tr.success,
                "metadata": tr.metadata,
                "metrics_scores": case_metrics
            })
    
    overall_metrics = {}
    for m_name, m_stats in metrics_summary.items():
        count = m_stats["count"]
        overall_metrics[m_name] = {
            "average_score": round((m_stats["total_score"] / count), 4) if count > 0 else 0.0,
            "success_rate": round((m_stats["success_count"] / count), 4) if count > 0 else 0.0,
            "total_evaluated": count,
            "threshold": m_stats["threshold"]
        }
    
    results_summary = {
        "total_test_cases": total_cases,
        "metrics": overall_metrics,
        "test_results": test_results_summary
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"   Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    
    for metric_name, stats in overall_metrics.items():
        print(f"Metric: {metric_name}")
        print(f"  Average Score: {stats['average_score']}")
        print(f"  Success Rate:  {stats['success_rate'] * 100:.1f}%")
        print(f"  Threshold:     {stats['threshold']}")
    
    print("\n✅ Evaluation completed!")
    
    return results


if __name__ == "__main__":
    main()