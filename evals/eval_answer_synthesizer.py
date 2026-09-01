"""
Answer Synthesizer Faithfulness Evaluation Script
Evaluates the answer synthesizer component independently using DeepEval with FaithfulnessMetric.
Uses Mistral Medium 3.5 (mistral-medium-3-5) as the LLM judge.
"""

import os
import sys
import json
import time
import random
import re
import asyncio
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
    SynthesisResult,
    format_search_results_for_llm,
    extract_sources_from_answer,
)

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
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
    
    def __init__(self, min_interval_seconds: float = 1.2):
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


def check_inline_citations(answer: str, max_source_id: int) -> Dict[str, Any]:
    """
    Analyze inline citation tags ([1], [2], etc.) in the synthesized answer.
    
    Args:
        answer: Synthesized answer text
        max_source_id: Maximum valid source ID available in context
        
    Returns:
        Dictionary with citation statistics
    """
    # Extract bracketed numbers like [1], [2], [1, 2]
    raw_citations = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', answer)
    cited_numbers = set()
    for item in raw_citations:
        for num_str in item.split(','):
            num_clean = num_str.strip()
            if num_clean.isdigit():
                cited_numbers.add(int(num_clean))
                
    has_inline_citations = len(cited_numbers) > 0
    has_sources_section = "sources:" in answer.lower() or "sources" in answer.lower()
    
    # Check for invalid citations beyond available sources
    invalid_citations = [n for n in cited_numbers if n < 1 or n > max_source_id]
    
    return {
        "has_inline_citations": has_inline_citations,
        "cited_source_ids": sorted(list(cited_numbers)),
        "has_sources_section": has_sources_section,
        "invalid_citations": invalid_citations,
        "all_citations_valid": len(invalid_citations) == 0 and has_inline_citations,
    }


def create_answer_synthesizer_test_cases(
    golden_data: List[Dict[str, Any]],
    agent: Optional[QAAgent] = None,
) -> List[LLMTestCase]:
    """
    Execute answer synthesis for all golden test cases using controlled search results,
    then create DeepEval LLMTestCase instances configured for Faithfulness evaluation.
    
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
    
    print(f"[*] Running evaluate_synthesis for {total} controlled golden test cases...")
    
    for idx, golden in enumerate(golden_data, start=1):
        golden_id = golden.get("id", f"synth_{idx:04d}")
        query = golden.get("input", "")
        search_results = golden.get("search_results", [])
        retrieval_context = golden.get("retrieval_context", [])
        expected_output = golden.get("expected_output", "")
        key_facts = golden.get("key_facts", [])
        question_type = golden.get("question_type", "general")
        category = golden.get("category", "General")
        faithfulness_challenge = golden.get("faithfulness_challenge", "")
        expected_sources = golden.get("expected_sources", [])
        
        # Fallback if retrieval_context is empty: extract from search_results
        if not retrieval_context and search_results:
            retrieval_context = [
                res.get("content", "") for res in search_results if res.get("content")
            ]
            
        print(f"   [{idx}/{total}] Synthesizing answer for {golden_id} ({question_type}): '{query[:55]}...'")
        
        start_time = time.time()
        try:
            # Component-level independent invocation: passing controlled search_results
            synthesis_result: SynthesisResult = agent.evaluate_synthesis(
                query=query,
                search_results=search_results,
            )
            elapsed_time = round(time.time() - start_time, 2)
            
            actual_answer = synthesis_result.answer
            sources_used = synthesis_result.sources_used
            
            # Analyze citations
            citation_analysis = check_inline_citations(
                actual_answer, 
                max_source_id=len(search_results)
            )
            
            # Create DeepEval LLMTestCase
            test_case = LLMTestCase(
                input=query,
                actual_output=actual_answer,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
                context=key_facts,
                additional_metadata={
                    "golden_id": golden_id,
                    "question_type": question_type,
                    "category": category,
                    "faithfulness_challenge": faithfulness_challenge,
                    "expected_sources": expected_sources,
                    "sources_used": sources_used,
                    "num_sources_used": len(sources_used),
                    "citation_analysis": citation_analysis,
                    "elapsed_time_seconds": elapsed_time,
                    "num_search_results_provided": len(search_results),
                },
            )
            test_cases.append(test_case)
            
        except Exception as e:
            print(f"[ERROR] Error synthesizing test case {golden_id} ('{query}'): {str(e)}")
            continue
            
    return test_cases


def create_faithfulness_metric(
    model_name: str = "mistral-medium-3-5",
    threshold: float = 0.7,
) -> FaithfulnessMetric:
    """
    Create DeepEval FaithfulnessMetric configured with Mistral Medium 3.5 judge.
    
    Args:
        model_name: Mistral model name to use as judge
        threshold: Faithfulness score threshold (0.0 to 1.0)
        
    Returns:
        Configured FaithfulnessMetric instance
    """
    mistral_judge = MistralJudge(
        model_name=model_name,
        max_retries=8,
        timeout_ms=60000,
        min_request_interval=1.2,
    )
    
    # async_mode=False evaluates sequentially per test case
    # to avoid concurrent request spikes hitting Mistral rate limits
    metric = FaithfulnessMetric(
        threshold=threshold,
        model=mistral_judge,
        include_reason=True,
        async_mode=False,
        verbose_mode=True,
    )
    
    return metric


def main():
    """Main function to run Answer Synthesizer evaluation."""
    # File paths
    DATASET_PATH = str(PROJECT_ROOT / "golden_datasets" / "answer_synthesizer_golden.json")
    OUTPUT_DIR = str(PROJECT_ROOT / "evals")
    RESULTS_FILE = os.path.join(OUTPUT_DIR, "answer_synthesizer_eval_results.json")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Validate API keys
    if not os.getenv("MISTRAL_API_KEY"):
        print("[WARNING] MISTRAL_API_KEY not found in environment variables.")
        print("Please ensure it is set in your .env file or environment.")
        return
    
    print("=" * 70)
    print("ANSWER SYNTHESIZER FAITHFULNESS EVALUATION")
    print("=" * 70)
    print(f"Dataset path:    {DATASET_PATH}")
    print(f"Component:       Answer Synthesizer (_synthesize / evaluate_synthesis)")
    print(f"LLM Judge:       Mistral (mistral-medium-3-5)")
    print(f"Metric:          FaithfulnessMetric (Claim Extraction & Verification)")
    print(f"Task Timeout:    {os.getenv('DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE')}s")
    print(f"Attempt Timeout: {os.getenv('DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE')}s")
    print("=" * 70)
    
    # 1. Load Golden Dataset
    print("\n[1/5] Loading golden dataset...")
    golden_data = load_golden_dataset(DATASET_PATH)
    print(f"      Loaded {len(golden_data)} golden test cases from dataset.")
    
    # 2. Run Answer Synthesis and Build Test Cases
    print("\n[2/5] Synthesizing answers from controlled search results and creating DeepEval test cases...")
    agent = QAAgent()
    test_cases = create_answer_synthesizer_test_cases(golden_data, agent=agent)
    print(f"      Successfully created {len(test_cases)} LLM test cases.")
    
    if not test_cases:
        print("[ERROR] No test cases were created. Exiting.")
        return
        
    # 3. Create Evaluation Dataset container
    eval_dataset = EvaluationDataset()
    eval_dataset.test_cases = test_cases
    
    # 4. Create Metric with Mistral Judge
    print("\n[3/5] Initializing FaithfulnessMetric with Mistral Medium 3.5 judge...")
    faithfulness_metric = create_faithfulness_metric(
        model_name="mistral-medium-3-5",
        threshold=0.7,
    )
    
    # 5. Run Evaluation with controlled rate limits
    print("\n[4/5] Running DeepEval Faithfulness evaluation...")
    results = evaluate(
        test_cases=test_cases,
        metrics=[faithfulness_metric],
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
    question_type_stats = {}
    total_cases = len(results.test_results) if hasattr(results, "test_results") else len(test_cases)
    
    if hasattr(results, "test_results"):
        for i, tr in enumerate(results.test_results):
            meta = tr.metadata or {}
            q_type = meta.get("question_type", "general")
            
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
                        
                    # Category breakdown
                    if q_type not in question_type_stats:
                        question_type_stats[q_type] = {
                            "total": 0,
                            "passed": 0,
                            "total_score": 0.0,
                        }
                    question_type_stats[q_type]["total"] += 1
                    if m.score is not None:
                        question_type_stats[q_type]["total_score"] += m.score
                    if m.success:
                        question_type_stats[q_type]["passed"] += 1
            
            test_results_summary.append({
                "test_case_id": i + 1,
                "golden_id": meta.get("golden_id"),
                "input": tr.input,
                "question_type": q_type,
                "category": meta.get("category"),
                "faithfulness_challenge": meta.get("faithfulness_challenge"),
                "success": tr.success,
                "actual_output": tr.actual_output,
                "expected_output": tr.expected_output,
                "metrics_scores": case_metrics,
                "citation_analysis": meta.get("citation_analysis"),
                "elapsed_time_seconds": meta.get("elapsed_time_seconds"),
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
        
    # Question type breakdown
    type_breakdown = {}
    for q_type, q_data in question_type_stats.items():
        tot = q_data["total"]
        type_breakdown[q_type] = {
            "total": tot,
            "passed": q_data["passed"],
            "pass_rate": round(q_data["passed"] / tot, 4) if tot > 0 else 0.0,
            "average_score": round(q_data["total_score"] / tot, 4) if tot > 0 else 0.0,
        }
        
    results_summary = {
        "dataset_name": "answer_synthesizer_faithfulness_golden_v1",
        "component": "answer_synthesizer",
        "judge_model": "mistral-medium-3-5",
        "total_test_cases": total_cases,
        "metrics": overall_metrics,
        "question_type_breakdown": type_breakdown,
        "test_results": test_results_summary,
    }
    
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
    print(f"      Results successfully saved to: {RESULTS_FILE}")
    
    # 7. Print Detailed Evaluation Summary Report
    print("\n" + "=" * 70)
    print("ANSWER SYNTHESIZER FAITHFULNESS EVALUATION REPORT")
    print("=" * 70)
    for metric_name, stats in overall_metrics.items():
        print(f"Metric:              {metric_name}")
        print(f"Judge Model:         {stats['judge_model']}")
        print(f"Total Evaluated:     {stats['total_evaluated']}")
        print(f"Average Score:       {stats['average_score']:.4f}")
        print(f"Pass Rate:           {stats['success_rate'] * 100:.1f}%")
        print(f"Threshold:           {stats['threshold']}")
    
    print("\n" + "-" * 70)
    print("BREAKDOWN BY QUESTION TYPE")
    print("-" * 70)
    for q_type, q_stats in type_breakdown.items():
        print(f"• {q_type:<35} | Count: {q_stats['total']:<2} | Avg Score: {q_stats['average_score']:.2f} | Pass Rate: {q_stats['pass_rate']*100:.0f}%")
        
    print("\n" + "-" * 70)
    print("INDIVIDUAL TEST CASE HIGHLIGHTS")
    print("-" * 70)
    for res in test_results_summary:
        f_metric = res.get("metrics_scores", {}).get("Faithfulness", {})
        score = f_metric.get("score", "N/A")
        status = "PASSED" if res.get("success") else "FAILED"
        print(f"[{res.get('golden_id')}] ({res.get('question_type')}): Score={score} [{status}]")
        if not res.get("success") and f_metric.get("reason"):
            print(f"   Reason: {f_metric.get('reason')[:120]}...")
            
    print("=" * 70)
    print("[SUCCESS] Answer Synthesizer evaluation completed successfully!")
    
    return results


if __name__ == "__main__":
    main()
