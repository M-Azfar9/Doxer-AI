Doxer AI is a multi-agent documentation assistant built for software engineers. Ask it to write a setup guide, explain part of a codebase, answer a technical question, or generate a complete SRS — it decides whether to pull live info from the web, analyze a GitHub repo file-by-file, retrieve relevant code via RAG, or run a full requirements-gathering conversation before producing a structured doc with diagrams. Built with LangGraph for multi-agent orchestration and LangSmith for tracing and evaluation.


CHALLENGES
1) What should we need to use Rag or analyze every file for document generation or for normal questions?? so the answer is hybrid approach. 

2) What 



# EVALUATION METRICS

==================================================
📊 EVALUATION SUMMARY OF INTENT ROUTER COMPONENT
==================================================

*BASE_LINE*

Metric: Route Accuracy [GEval]
  Average Score: 0.956
  Success Rate:  94.0%
  Threshold:     0.7

*AFTER_IMPROVEMENTS*
Metric: Route Accuracy [GEval]
  Average Score: 0.984
  Success Rate:  98.0%
  Threshold:     0.7

# WEB SEARCH COMPONENT

Metric:        Contextual Relevancy
Judge Model:   mistral-medium-3-5
Total Cases:   30
Average Score: 0.6388
Success Rate:  53.3%
Threshold:     0.7

*AFTER IMPROVEMENTS*
Improves query conversion, tavily parameters from basic search to advanced search, after fetching content Removes webpage artifacts, boilerplate, navigation, and junk tokens, and then Compress and extract only query-relevant facts from each snippet using LLM. and then pass to the sythesizer


Metric:        Contextual Relevancy
Judge Model:   mistral-medium-3-5
Total Cases:   30
Average Score: 0.9267
Success Rate:  96.7%
Threshold:     0.7


# ANSWER SYNTHESIZER OF WEB SEARCH COMPONENT
Aggregate Metrics

Metric          Average Score    Pass Rate                         Total
Faithfulness    0.98             100.00% (passed=20, failed=0)     20

======================================================================
ANSWER SYNTHESIZER FAITHFULNESS EVALUATION REPORT
======================================================================
Metric:              Faithfulness
Judge Model:         mistral-medium-3-5
Total Evaluated:     20
Average Score:       0.9838
Pass Rate:           100.0%
Threshold:           0.7


We use jitter technique as well, because if api fails then we try after some random amount of time to request again.

Jitter is simply adding a random amount of variation to timing. In your code, it's the random extra delay added to retry attempts.

A Circuit Breaker is a resilience pattern that prevents cascading failures by catching repeated errors from a component (like LLM structured output validation) and returning a safe fallback value instead of letting exceptions crash the entire system. In agentic systems, it's essential because agents run multi-step workflows where one failing node shouldn't halt the entire graph—if an LLM returns unparseable JSON or an API times out, the circuit breaker "trips" and provides a graceful default (e.g., RouteDecision.DIRECT), allowing the agent to continue processing other tasks. This ensures high availability, self-healing behavior, and better user experience—users get partial results or safe defaults instead of complete failures. Simply put: circuit breakers transform catastrophic crashes into manageable degradations, keeping agents resilient even when individual components fail.




# REMAINING FEATURES
Actuall AST Needs to be implemented in GIT RAG, We can also use Reranker, 



