"""
Web Search Contextual Relevancy Evaluation Script
Evaluates the web search component using DeepEval with ContextualRelevancyMetric.
Uses Mistral Medium 3.5 (mistral-medium-3-5) as the LLM judge.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure DeepEval task and attempt timeout overrides (seconds)
# Provides ample buffer for large web search content and LLM-as-judge calculations
DEFAULT_TASK_TIMEOUT = "600"       # 10 minutes total per test case
DEFAULT_ATTEMPT_TIMEOUT = "180"    # 3 minutes per LLM call attempt
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = os.getenv(
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", DEFAULT_TASK_TIMEOUT
)
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = os.getenv(
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", DEFAULT_ATTEMPT_TIMEOUT
)

# Load environment variables
load_dotenv(override=True)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the agent and evaluator
from src.simple_search_agent import (
    QAAgent,
    QAAgentEvaluator,
    WebSearchResult,
    format_search_results_for_llm,
)

import asyncio
import time
import random

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig

# Configure Mistral as the LLM judge
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral


class AsyncRateLimiter:
    """Async rate limiter to enforce minimum time interval between Mistral API calls."""
    
    def __init__(self, min_interval_seconds: float = 1.0):
        self.min_interval = min_interval_seconds
        self.last_call = 0.0
        self._lock = asyncio.Lock()
    
    async def wait(self):
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


class MistralJudge(DeepEvalBaseLLM):
    """
    Custom Mistral LLM wrapper for DeepEval LLM-as-judge evaluation.
    Supports structured output parsing, json mode, rate limiting, and jittered exponential backoff.
    """
    
    def __init__(
        self,
        model_name: str = "mistral-medium-3-5",
        max_retries: int = 8,
        timeout_ms: int = 60000,
        min_request_interval: float = 1.2,
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_ms = timeout_ms
        self._rate_limiter = AsyncRateLimiter(min_interval_seconds=min_request_interval)
        self._sync_last_call = 0.0
        super().__init__(model=model_name)
    
    def load_model(self):
        """Load Mistral client with configured timeout."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not configured in environment variables.")
        return Mistral(api_key=api_key, timeout_ms=self.timeout_ms)
    
    def _sync_rate_limit(self, interval: float = 1.2):
        """Enforce rate limiting on synchronous calls."""
        now = time.time()
        elapsed = now - self._sync_last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._sync_last_call = time.time()
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Any:
        """Generate synchronous response from Mistral with rate limiting and retry."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self._sync_rate_limit()
                if schema is not None:
                    response = self.model.chat.parse(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=schema,
                        temperature=0.01,
                    )
                    return response.choices[0].message.parsed
                else:
                    is_json_prompt = "json" in prompt.lower() or "schema" in prompt.lower()
                    response = self.model.chat.complete(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.01,
                        response_format={"type": "json_object"} if is_json_prompt else None,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                last_error = e
                is_rate_limit = "429" in str(e) or "rate_limited" in str(e).lower() or "rate limit" in str(e).lower()
                if is_rate_limit:
                    sleep_time = min(60.0, (2 ** attempt) * 3 + random.uniform(1.0, 3.0))
                    print(f"   [RATE LIMIT] 429 on sync call, retrying in {sleep_time:.1f}s (attempt {attempt+1}/{self.max_retries})...")
                else:
                    sleep_time = min(30.0, (2 ** attempt) + random.uniform(0.5, 1.5))
                time.sleep(sleep_time)
        
        raise RuntimeError(f"Mistral generation failed after {self.max_retries} attempts: {last_error}")
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Any:
        """Generate asynchronous response from Mistral with rate limiting and retry."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                await self._rate_limiter.wait()
                if schema is not None:
                    response = await self.model.chat.parse_async(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=schema,
                        temperature=0.01,
                    )
                    return response.choices[0].message.parsed
                else:
                    is_json_prompt = "json" in prompt.lower() or "schema" in prompt.lower()
                    response = await self.model.chat.complete_async(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.01,
                        response_format={"type": "json_object"} if is_json_prompt else None,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                last_error = e
                is_rate_limit = "429" in str(e) or "rate_limited" in str(e).lower() or "rate limit" in str(e).lower()
                if is_rate_limit:
                    sleep_time = min(60.0, (2 ** attempt) * 3 + random.uniform(1.0, 3.0))
                    print(f"   [RATE LIMIT] 429 on async call, retrying in {sleep_time:.1f}s (attempt {attempt+1}/{self.max_retries})...")
                else:
                    sleep_time = min(30.0, (2 ** attempt) + random.uniform(0.5, 1.5))
                await asyncio.sleep(sleep_time)
        
        raise RuntimeError(f"Mistral async generation failed after {self.max_retries} attempts: {last_error}")
    
    def generate_with_schema(self, *args, schema=None, **kwargs):
        """Handle synchronous structured schema generation."""
        return self.generate(*args, schema=schema, **kwargs)
    
    async def a_generate_with_schema(self, *args, schema=None, **kwargs):
        """Handle asynchronous structured schema generation."""
        return await self.a_generate(*args, schema=schema, **kwargs)
    
    def get_model_name(self) -> str:
        return self.model_name


def load_golden_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load golden dataset from JSON file.
    
    Args:
        file_path: Path to the JSON dataset file
        
    Returns:
        List of golden test case dictionaries
    """
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset.get("goldens", [])


def clean_snippet(text: str, max_chars: int = 2000) -> str:
    """
    Clean and optimize search snippet for Contextual Relevancy evaluation.
    Removes extraneous whitespace and bounds snippet size to avoid LLM timeouts.
    """
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "..."
    return cleaned


def create_web_search_test_cases(
    golden_data: List[Dict[str, Any]],
    agent: Optional[QAAgent] = None,
) -> List[LLMTestCase]:
    """
    Pass all test cases to evaluate_web_search function to get search results,
    then create DeepEval LLMTestCase instances with optimized retrieval contexts.
    
    Args:
        golden_data: List of golden test cases from dataset
        agent: Optional QAAgent instance (creates a new one if not provided)
        
    Returns:
        List of DeepEval LLMTestCase objects
    """
    if agent is None:
        agent = QAAgent()
        
    test_cases = []
    total = len(golden_data)
    
    print(f"[*] Running evaluate_web_search for {total} test queries...")
    
    for idx, golden in enumerate(golden_data, start=1):
        golden_id = golden.get("id", f"case_{idx}")
        query = golden.get("input", "")
        max_results = golden.get("max_results", None)
        target_context = golden.get("target_context", [])
        
        print(f"   [{idx}/{total}] Processing {golden_id}: '{query}' (max_results={max_results})...")
        
        try:
            # Execute web search evaluation using the agent's evaluate_web_search function
            web_search_result: WebSearchResult = agent.evaluate_web_search(
                query=query,
                max_results=max_results,
            )
            
            search_results = web_search_result.search_results
            
            # Extract and optimize retrieved contexts (clean text snippets)
            retrieval_context = [
                clean_snippet(res.get("content", ""))
                for res in search_results
                if res.get("content") and clean_snippet(res.get("content", ""))
            ]
            
            # Fallback if no content found
            if not retrieval_context:
                retrieval_context = ["No search results were retrieved for this query."]
            
            # Format actual output representation
            actual_output = format_search_results_for_llm(search_results)
            
            # Expected output summary from target context
            expected_output = "\n\n".join(target_context) if target_context else "Relevant search results."
            
            # Create DeepEval LLMTestCase
            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
                context=target_context,
                additional_metadata={
                    "golden_id": golden_id,
                    "max_results_requested": max_results,
                    "num_results_retrieved": web_search_result.num_results,
                    "has_results": web_search_result.has_results,
                    "search_results": search_results,
                },
            )
            test_cases.append(test_case)
            
        except Exception as e:
            print(f"[ERROR] Error processing test case {golden_id} ('{query}'): {str(e)}")
            continue
            
    return test_cases


def create_contextual_relevancy_metric(
    model_name: str = "mistral-medium-3-5",
    threshold: float = 0.7,
) -> ContextualRelevancyMetric:
    """
    Create DeepEval ContextualRelevancyMetric configured with Mistral judge.
    
    Args:
        model_name: Mistral model name to use as judge
        threshold: Score threshold for passing evaluation (0.0 to 1.0)
        
    Returns:
        Configured ContextualRelevancyMetric instance
    """
    mistral_judge = MistralJudge(
        model_name=model_name,
        max_retries=8,
        timeout_ms=60000,
        min_request_interval=1.2,
    )
    
    # async_mode=False evaluates contexts sequentially per test case
    # to avoid concurrent request spikes hitting Mistral rate limits
    metric = ContextualRelevancyMetric(
        threshold=threshold,
        model=mistral_judge,
        include_reason=True,
        async_mode=False,
        verbose_mode=True,
    )
    
    return metric


def main():
    """Main function to run web search evaluation."""
    # File paths
    DATASET_PATH = str(PROJECT_ROOT / "golden_datasets" / "web_search_golden.json")
    OUTPUT_DIR = str(PROJECT_ROOT / "evals")
    RESULTS_FILE = os.path.join(OUTPUT_DIR, "web_search_eval_results.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Validate API keys
    if not os.getenv("MISTRAL_API_KEY"):
        print("[WARNING] MISTRAL_API_KEY not found in environment variables.")
        print("Please ensure it is set in your .env file or environment.")
        return
        
    if not os.getenv("TAVILY_API_KEY"):
        print("[WARNING] TAVILY_API_KEY not found in environment variables.")
        print("Please ensure it is set in your .env file or environment.")
        return
    
    print("============================================================")
    print("Starting Web Search Evaluation (Contextual Relevancy)")
    print("============================================================")
    print(f"Dataset path:    {DATASET_PATH}")
    print(f"LLM Judge:       Mistral (mistral-medium-3-5)")
    print(f"Metric:          ContextualRelevancyMetric")
    print(f"Task Timeout:    {os.getenv('DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE')}s")
    print(f"Attempt Timeout: {os.getenv('DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE')}s")
    print("============================================================")
    
    # 1. Load Golden Dataset
    print("\n[1/5] Loading golden dataset...")
    golden_data = load_golden_dataset(DATASET_PATH)
    print(f"      Loaded {len(golden_data)} golden test cases from dataset.")
    
    # 2. Execute Web Search and Create Test Cases
    print("\n[2/5] Running web search and building DeepEval test cases...")
    agent = QAAgent()
    test_cases = create_web_search_test_cases(golden_data, agent=agent)
    print(f"      Successfully created {len(test_cases)} LLM test cases.")
    
    if not test_cases:
        print("[ERROR] No test cases were created. Exiting.")
        return
        
    # 3. Create Evaluation Dataset container
    eval_dataset = EvaluationDataset()
    eval_dataset.test_cases = test_cases
    
    # 4. Create Metric with Mistral Judge
    print("\n[3/5] Initializing ContextualRelevancyMetric with Mistral judge...")
    contextual_relevancy_metric = create_contextual_relevancy_metric(
        model_name="mistral-medium-3-5",
        threshold=0.7,
    )
    
    # 5. Run Evaluation with controlled rate limits
    print("\n[4/5] Running DeepEval evaluation...")
    results = evaluate(
        test_cases=test_cases,
        metrics=[contextual_relevancy_metric],
        async_config=AsyncConfig(
            run_async=True,
            throttle_value=1.5,
            max_concurrent=2,
        ),
        error_config=ErrorConfig(
            ignore_errors=True,
        ),
    )
    
    # 6. Process and Save Results
    print("\n[5/5] Processing and saving evaluation results...")
    metrics_summary = {}
    test_results_summary = []
    total_cases = len(results.test_results) if hasattr(results, "test_results") else len(test_cases)
    
    if hasattr(results, "test_results"):
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
                        "evaluation_model": "mistral-medium-3-5",
                    }
                    if m.name not in metrics_summary:
                        metrics_summary[m.name] = {
                            "total_score": 0.0,
                            "count": 0,
                            "success_count": 0,
                            "threshold": m.threshold,
                        }
                    if m.score is not None:
                        metrics_summary[m.name]["total_score"] += m.score
                        metrics_summary[m.name]["count"] += 1
                    if m.success:
                        metrics_summary[m.name]["success_count"] += 1
            
            test_results_summary.append({
                "test_case_id": i + 1,
                "golden_id": tr.metadata.get("golden_id") if tr.metadata else None,
                "input": tr.input,
                "num_retrieved": tr.metadata.get("num_results_retrieved") if tr.metadata else None,
                "success": tr.success,
                "metrics_scores": case_metrics,
                "retrieval_context_preview": [ctx[:150] + "..." for ctx in (tr.retrieval_context or [])],
            })
            
    overall_metrics = {}
    for m_name, m_stats in metrics_summary.items():
        count = m_stats["count"]
        overall_metrics[m_name] = {
            "average_score": round((m_stats["total_score"] / count), 4) if count > 0 else 0.0,
            "success_rate": round((m_stats["success_count"] / count), 4) if count > 0 else 0.0,
            "total_evaluated": count,
            "threshold": m_stats["threshold"],
            "judge_model": "mistral-medium-3-5",
        }
        
    results_summary = {
        "dataset_name": "web_search_contextual_relevancy_v1",
        "judge_model": "mistral-medium-3-5",
        "total_test_cases": total_cases,
        "metrics": overall_metrics,
        "test_results": test_results_summary,
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
    print(f"      Results successfully saved to: {RESULTS_FILE}")
    
    # 7. Print Evaluation Summary Report
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY REPORT")
    print("============================================================")
    for metric_name, stats in overall_metrics.items():
        print(f"Metric:        {metric_name}")
        print(f"Judge Model:   {stats['judge_model']}")
        print(f"Total Cases:   {stats['total_evaluated']}")
        print(f"Average Score: {stats['average_score']}")
        print(f"Success Rate:  {stats['success_rate'] * 100:.1f}%")
        print(f"Threshold:     {stats['threshold']}")
    print("============================================================")
    print("[SUCCESS] Web search evaluation completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
