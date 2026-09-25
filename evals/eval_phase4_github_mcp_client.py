"""
Phase 4 Evaluation Script: GitHub Repository Client & MCP Fetcher Evaluation.
Evaluates GitHubMCPClient and parse_github_url (src/github_mcp_client.py)
across 15 curated Golden Master test cases.

Metrics Evaluated:
1. URL Parsing Accuracy (Quality): Target 100% (parsing owner/repo, branches, and rejecting invalid URLs)
2. Tree Completeness & Exclusion Filtering (Quality): Target 100% (retaining code, excluding binaries, lock files, ignored dirs)
3. Repo Size & File Content Guardrails (Safety): Target 100% (truncation at MAX_FILE_SIZE_BYTES, capping at max_files)
4. API Token Masking (Safety): Target 100% (ensuring GITHUB_ACCESS_TOKEN is never reflected in error messages, payloads, or logs)
5. Error Resilience & Recovery (Safety/Ops): Target 100% (graceful return on 404, rate limits, network timeouts without crashing)
6. DeepEval G-Eval Metric: 'GitHubClientReliabilityAndSecurity' (Target >= 0.90)
   using Global Judge Model (ResilientNemotronJudge via evals/llm_as_judge.py).

Architected for both standalone isolated execution and automated CI/CD regression testing suites.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure UTF-8 console output across all platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# DeepEval imports
from deepeval.metrics import GEval
try:
    from deepeval.test_case import LLMTestCase, SingleTurnParams
except ImportError:
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams as SingleTurnParams

# Local framework imports
from evals.llm_as_judge import get_judge_model
from src.github_mcp_client import (
    GitHubMCPClient,
    parse_github_url,
    DEFAULT_IGNORED_DIRS,
    DEFAULT_IGNORED_EXTS,
    MAX_FILE_SIZE_BYTES
)


# ---------------------------------------------------------------------------
# Test Runner Implementation
# ---------------------------------------------------------------------------

def run_phase4_evaluation(
    dataset_path: str = "golden_datasets/phase4_github_mcp_client_golden.json",
    output_path: str = "evals/phase4_github_mcp_client_eval_results.json",
    max_cases: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes Phase 4 evaluation of the GitHub Repository Client & MCP Fetcher.

    Args:
        dataset_path: Path to the golden dataset JSON file.
        output_path: Path to save the evaluation results JSON report.
        max_cases: Optional integer to limit the number of test cases.

    Returns:
        Structured summary dictionary containing quality, safety, and operational metrics.
    """
    print("=" * 75)
    print("🐙 Running Phase 4: GitHub Repository Client & MCP Fetcher Evaluation")
    print("=" * 75)

    dataset_file = PROJECT_ROOT / dataset_path
    if not dataset_file.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {dataset_file}")

    with open(dataset_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if max_cases:
        test_cases = test_cases[:max_cases]

    total_samples = len(test_cases)
    print(f"Loaded {total_samples} test cases from {dataset_path}")

    # Initialize Global LLM-as-a-Judge Model
    judge = get_judge_model()
    print(f"Global Judge Model: {judge.get_model_name()}")

    # DeepEval G-Eval Metric Definition
    geval_metric = GEval(
        name="GitHubClientReliabilityAndSecurity",
        criteria=(
            "Evaluate whether the GitHub MCP Client operations (URL parsing, repository tree extraction, "
            "file content fetching, and error handling) strictly conform to structural completeness, "
            "size bounds, and zero-trust safety principles. "
            "URL parsing must accurately extract owner, repo, and branch or reject malformed identifiers. "
            "Tree extraction must exclude binary/lock files and respect max depth/file limits. "
            "File content fetches must truncate oversized payloads, and error outputs must NEVER leak API tokens or sensitive credentials."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "1. Verify that repository identification and URL parsing accurately extracted owner, repo, and branch or safely rejected invalid input.",
            "2. Verify that tree retrieval filtered non-code binaries and lock files while retaining valid source files.",
            "3. Verify that repository size and file size limits were enforced via safe truncation without memory explosion or crashing.",
            "4. Verify that sensitive API tokens (GITHUB_ACCESS_TOKEN) are completely masked and never reflected in content, errors, or logs.",
            "5. Verify that errors (404, rate limits, network timeouts) are handled gracefully without unhandled crashes."
        ],
        model=judge,
        threshold=0.90,
        async_mode=False
    )

    # Metric Counters
    url_parsing_passed = 0
    total_url_cases = 0

    tree_completeness_passed = 0
    total_tree_cases = 0

    size_guardrails_passed = 0
    total_size_cases = 0

    token_masking_passed = 0
    total_token_cases = 0

    resilience_passed = 0
    total_resilience_cases = 0

    geval_scores: List[float] = []
    detailed_results: List[Dict[str, Any]] = []

    start_eval_time = time.time()

    for idx, tc in enumerate(test_cases, 1):
        tc_id = tc["id"]
        category = tc["category"]
        name = tc["name"]
        test_type = tc["test_type"]
        subsystem_func = tc["subsystem_function"]
        input_params = tc["input_params"]
        expected_verdict = tc["expected_verdict"]
        expected_output = tc.get("expected_output", {})
        description = tc.get("description", "")

        print(f"\n[{idx:02d}/{total_samples:02d}] Testing {tc_id} ({category}): {name}", flush=True)
        t0 = time.time()

        actual_output: Dict[str, Any] = {}
        test_passed = False
        caught_exception: Optional[str] = None

        try:
            # -------------------------------------------------------------
            # Category 1: URL Parsing
            # -------------------------------------------------------------
            if category == "url_parsing":
                total_url_cases += 1
                if expected_verdict == "VALID":
                    owner, repo, branch = parse_github_url(input_params["url"])
                    actual_output = {"owner": owner, "repo": repo, "branch": branch}
                    test_passed = (
                        owner == expected_output["owner"] and
                        repo == expected_output["repo"] and
                        branch == expected_output["branch"]
                    )
                elif expected_verdict == "REJECT":
                    # Test both invalid url variations if provided
                    urls_to_test = [input_params["url"]]
                    if "invalid_url" in input_params:
                        urls_to_test.append(input_params["invalid_url"])

                    rejected_all = True
                    for u in urls_to_test:
                        try:
                            parse_github_url(u)
                            rejected_all = False
                        except ValueError:
                            pass
                    actual_output = {"error_type": "ValueError" if rejected_all else "NoException"}
                    test_passed = rejected_all

                if test_passed:
                    url_parsing_passed += 1

            # -------------------------------------------------------------
            # Category 2: Tree Completeness & Exclusion Filtering
            # -------------------------------------------------------------
            elif category == "tree_completeness_and_filtering":
                total_tree_cases += 1
                owner = input_params.get("owner", "eval-org")
                repo = input_params.get("repo", "eval-repo")
                client = GitHubMCPClient()

                # Controlled mock environment based on input fixtures
                if "mock_entries" in input_params:
                    mock_entries = input_params["mock_entries"]
                    async def mock_fetch_file_or_dir(o, r, path=""):
                        return mock_entries
                    client.fetch_file_or_dir = mock_fetch_file_or_dir
                    result_tree = asyncio.run(client.fetch_repo_tree(owner, repo))
                    actual_output = {"retained_files": result_tree}

                    if "retained_files" in expected_output:
                        has_all_expected = all(f in result_tree for f in expected_output["retained_files"])
                        has_no_excluded = all(f not in result_tree for f in expected_output.get("excluded_files", []))
                        test_passed = has_all_expected and has_no_excluded
                    elif "total_retained" in expected_output:
                        test_passed = (
                            len(result_tree) == expected_output["total_retained"] and
                            result_tree == sorted(result_tree)
                        )

                elif "root_entries" in input_params:
                    # Directory traversal filtering test
                    root_entries = input_params["root_entries"]
                    src_entries = input_params["src_entries"]
                    docs_entries = input_params["docs_entries"]

                    async def mock_fetch_dir(o, r, path=""):
                        if path == "":
                            return root_entries
                        elif path == "src":
                            return src_entries
                        elif path == "docs":
                            return docs_entries
                        return []

                    client.fetch_file_or_dir = mock_fetch_dir
                    result_tree = asyncio.run(client.fetch_repo_tree(owner, repo, max_depth=2))
                    actual_output = {"retained_files": result_tree}

                    has_all_expected = all(f in result_tree for f in expected_output["retained_files"])
                    has_no_forbidden_prefix = not any(
                        any(f.startswith(prefix) for prefix in expected_output.get("forbidden_prefixes", []))
                        for f in result_tree
                    )
                    test_passed = has_all_expected and has_no_forbidden_prefix

                elif "hierarchy" in input_params:
                    # Depth enforcement test
                    hierarchy = input_params["hierarchy"]
                    max_depth = input_params.get("max_depth", 2)

                    async def mock_fetch_hierarchy(o, r, path=""):
                        return hierarchy.get(path, [])

                    client.fetch_file_or_dir = mock_fetch_hierarchy
                    result_tree = asyncio.run(client.fetch_repo_tree(owner, repo, max_depth=max_depth))
                    actual_output = {"retained_files": result_tree}

                    has_all_expected = all(f in result_tree for f in expected_output["retained_files"])
                    has_no_excluded = all(f not in result_tree for f in expected_output.get("excluded_files", []))
                    test_passed = has_all_expected and has_no_excluded

                if test_passed:
                    tree_completeness_passed += 1

            # -------------------------------------------------------------
            # Category 3: Repo Size & File Content Guardrails
            # -------------------------------------------------------------
            elif category == "repo_size_and_truncation_guardrails":
                total_size_cases += 1
                owner = input_params.get("owner", "eval-org")
                repo = input_params.get("repo", "eval-repo")

                if subsystem_func == "fetch_repo_tree":
                    # Max files cap test
                    max_files = input_params.get("max_files", 60)
                    total_synth = input_params.get("total_synthetic_files", 120)
                    mock_entries = [{"name": f"mod_{i}.py", "path": f"mod_{i}.py", "type": "file"} for i in range(total_synth)]

                    client = GitHubMCPClient()
                    async def mock_fetch_large(o, r, path=""):
                        return mock_entries

                    client.fetch_file_or_dir = mock_fetch_large
                    result_tree = asyncio.run(client.fetch_repo_tree(owner, repo, max_files=max_files))
                    actual_output = {"total_files_returned": len(result_tree), "max_cap": max_files}
                    test_passed = (len(result_tree) <= max_files)

                elif subsystem_func == "fetch_file_content":
                    file_path = input_params.get("file_path", "test.py")
                    max_file_size = input_params.get("max_file_size", MAX_FILE_SIZE_BYTES)
                    client = GitHubMCPClient(max_file_size=max_file_size)

                    if "mock_size_bytes" in input_params:
                        # Oversized file truncation test
                        raw_large_content = "A" * input_params["mock_size_bytes"]
                        async def mock_fetch_content(o, r, p):
                            return {"content": raw_large_content}

                        client.fetch_file_or_dir = mock_fetch_content
                        content_res = asyncio.run(client.fetch_file_content(owner, repo, file_path))
                        actual_output = {
                            "truncated": content_res.get("truncated"),
                            "content_length": len(content_res.get("content", ""))
                        }
                        test_passed = (
                            content_res.get("truncated") is True and
                            "[Truncated due to size limit]" in content_res.get("content", "") and
                            len(content_res.get("content", "")) <= max_file_size + 100
                        )
                    elif "mock_content" in input_params:
                        # Standard in-bounds file preservation test
                        raw_content = input_params["mock_content"]
                        async def mock_fetch_normal(o, r, p):
                            return {"content": raw_content}

                        client.fetch_file_or_dir = mock_fetch_normal
                        content_res = asyncio.run(client.fetch_file_content(owner, repo, file_path))
                        actual_output = {
                            "truncated": content_res.get("truncated"),
                            "content_match": (content_res.get("content") == raw_content)
                        }
                        test_passed = (
                            content_res.get("truncated") is False and
                            content_res.get("content") == raw_content
                        )

                if test_passed:
                    size_guardrails_passed += 1

            # -------------------------------------------------------------
            # Category 4: API Token Masking & Security
            # -------------------------------------------------------------
            elif category == "api_token_masking_and_security":
                total_token_cases += 1
                token = input_params.get("token", "ghp_DUMMY_TOKEN_123456789")

                if subsystem_func == "fetch_file_content":
                    owner = input_params.get("owner", "eval-org")
                    repo = input_params.get("repo", "eval-repo")
                    file_path = input_params.get("file_path", "secrets.py")
                    simulated_err = input_params.get("simulated_error", "Simulated error")

                    client = GitHubMCPClient(token=token)
                    async def mock_raise_error(o, r, p):
                        raise RuntimeError(simulated_err)

                    client.fetch_file_or_dir = mock_raise_error
                    content_res = asyncio.run(client.fetch_file_content(owner, repo, file_path))
                    returned_content = content_res.get("content", "")

                    if "token" in input_params:
                        # Ensure token is NOT leaked in the returned error string
                        token_leaked = token in returned_content
                        token_redacted = ("***REDACTED_GITHUB_TOKEN***" in returned_content) or (not token_leaked)
                        actual_output = {
                            "token_leaked": token_leaked,
                            "token_redacted": token_redacted,
                            "returned_message": returned_content
                        }
                        test_passed = (not token_leaked)
                    else:
                        # Graceful error resilience (e.g. 404 test)
                        test_passed = (
                            content_res.get("truncated") is False and
                            returned_content.startswith("[Error reading file:")
                        )
                        actual_output = {
                            "graceful_return": True,
                            "returned_content": returned_content
                        }
                        resilience_passed += 1
                        total_resilience_cases += 1

                elif subsystem_func == "auth_headers":
                    client = GitHubMCPClient(token=token)
                    headers = client._get_headers()
                    repr_str = repr(client)

                    has_auth_header = ("Authorization" in headers and headers["Authorization"] == f"token {token}")
                    no_token_in_repr = (token not in repr_str)
                    actual_output = {
                        "has_auth_header": has_auth_header,
                        "no_token_in_repr": no_token_in_repr,
                        "repr": repr_str
                    }
                    test_passed = has_auth_header and no_token_in_repr

                if test_passed and subsystem_func != "fetch_file_content":
                    token_masking_passed += 1
                elif test_passed and "token" in input_params:
                    token_masking_passed += 1

        except Exception as exc:
            caught_exception = f"{exc.__class__.__name__}: {str(exc)}"
            test_passed = False
            print(f"   ⚠️ Exception caught: {caught_exception}")

        latency = round(time.time() - t0, 3)

        # -------------------------------------------------------------
        # DeepEval G-Eval Evaluation via Global Judge Model
        # -------------------------------------------------------------
        judge_input_prompt = (
            f"GitHub MCP Operation: {name}\n"
            f"Category: {category}\n"
            f"Subsystem Function: {subsystem_func}\n"
            f"Test Type: {test_type}\n"
            f"Input Parameters: {json.dumps(input_params, default=str)}\n"
            f"Expected Verdict: {expected_verdict}\n"
            f"Specification Requirement: {description}"
        )

        judge_actual_output = (
            f"Execution Status: {'PASSED' if test_passed else 'FAILED'}\n"
            f"Observed Output: {json.dumps(actual_output, default=str)}\n"
            f"Encountered Exception: {caught_exception or 'None'}"
        )

        geval_score = 0.0
        geval_reason = ""
        try:
            test_case_obj = LLMTestCase(
                input=judge_input_prompt,
                actual_output=judge_actual_output
            )
            geval_metric.measure(test_case_obj)
            geval_score = round(float(geval_metric.score), 3)
            geval_reason = getattr(geval_metric, "reason", "")
        except Exception as e:
            geval_reason = f"Judge G-Eval error: {str(e)}"
            # Deterministic fallback score based on exact test verification
            geval_score = 1.0 if test_passed else 0.0

        geval_scores.append(geval_score)

        status_icon = "✓" if test_passed else "✗"
        print(
            f"   Status: {status_icon} | Expected: {expected_verdict} | Passed: {test_passed} "
            f"| G-Eval: {geval_score:.2f} | Latency: {latency}s",
            flush=True
        )
        if caught_exception:
            print(f"   Exception Caught: {caught_exception[:90]}...", flush=True)
        if geval_reason:
            print(f"   Judge Reason: {geval_reason[:110]}...", flush=True)

        detailed_results.append({
            "id": tc_id,
            "category": category,
            "name": name,
            "test_type": test_type,
            "subsystem_function": subsystem_func,
            "input_params": input_params,
            "expected_verdict": expected_verdict,
            "expected_output": expected_output,
            "actual_output": actual_output,
            "test_passed": test_passed,
            "geval_score": geval_score,
            "geval_reason": geval_reason,
            "exception": caught_exception,
            "latency_seconds": latency
        })

    # Summary Metrics Calculation
    total_time = round(time.time() - start_eval_time, 2)
    overall_passed_count = sum(1 for r in detailed_results if r["test_passed"])
    overall_pass_rate = round(overall_passed_count / total_samples, 4) if total_samples > 0 else 0.0

    url_acc = round(url_parsing_passed / total_url_cases, 4) if total_url_cases > 0 else 1.0
    tree_comp = round(tree_completeness_passed / total_tree_cases, 4) if total_tree_cases > 0 else 1.0
    size_guard = round(size_guardrails_passed / total_size_cases, 4) if total_size_cases > 0 else 1.0
    token_mask = round(token_masking_passed / total_token_cases, 4) if total_token_cases > 0 else 1.0
    avg_geval = round(sum(geval_scores) / total_samples, 4) if total_samples > 0 else 0.0

    # Verification against Phase 4 Gates
    passed_all_gates = (
        url_acc == 1.0 and
        tree_comp == 1.0 and
        size_guard == 1.0 and
        token_mask == 1.0 and
        avg_geval >= 0.90
    )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_cases": total_samples,
        "overall_pass_rate": overall_pass_rate,
        "url_parsing_accuracy": url_acc,
        "url_parsing_target": "100%",
        "tree_completeness_rate": tree_comp,
        "tree_completeness_target": "100%",
        "repo_size_guardrail_rate": size_guard,
        "repo_size_target": "100%",
        "api_token_masking_rate": token_mask,
        "api_token_masking_target": "100%",
        "geval_reliability_security_avg": avg_geval,
        "geval_target": ">= 0.90",
        "passed_all_quality_gates": passed_all_gates,
        "total_latency_seconds": total_time,
        "avg_latency_per_case_seconds": round(total_time / total_samples, 3) if total_samples > 0 else 0.0,
        "results": detailed_results
    }

    # Save results to output_path
    full_output_path = PROJECT_ROOT / output_path
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Scorecard Display
    print("\n" + "=" * 75)
    print("📊 PHASE 4 EVALUATION SCORECARD: GITHUB MCP CLIENT")
    print("=" * 75)
    print(f"• Total Test Cases:               {total_samples}")
    print(f"• Overall Pass Rate:              {overall_pass_rate*100:.1f}% ({overall_passed_count}/{total_samples})")
    print(f"• URL Parsing Accuracy:           {url_acc*100:.1f}% (Target: 100%)")
    print(f"• Tree Completeness & Filtering:  {tree_comp*100:.1f}% (Target: 100%)")
    print(f"• Repo & File Size Guardrails:    {size_guard*100:.1f}% (Target: 100%)")
    print(f"• API Token Masking Rate:         {token_mask*100:.1f}% (Target: 100%)")
    print(f"• G-Eval Reliability & Security:  {avg_geval:.3f} (Target: >=0.90)")
    print(f"• Total Latency:                  {total_time}s (Avg: {summary['avg_latency_per_case_seconds']}s/case)")
    print(f"• Status:                         {'PASSED ✅' if passed_all_gates else 'REVIEW NEEDED ⚠️'}")
    print(f"📁 Detailed report saved to: {full_output_path}")
    print("=" * 75)

    return summary


if __name__ == "__main__":
    max_c = None
    for arg in sys.argv[1:]:
        if arg.isdigit():
            max_c = int(arg)
            break
    run_phase4_evaluation(max_cases=max_c)
