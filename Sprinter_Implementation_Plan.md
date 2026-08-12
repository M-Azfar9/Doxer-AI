# Sprinter — Phased Implementation Plan
### An agentic documentation assistant for software engineers (LangGraph + LangSmith)

This plan is written to be built **incrementally**. Each phase is a working, demoable system on its own — you never build "the whole graph" in one sitting. Every phase adds exactly one new axis of complexity: a new node type, a new routing decision, a new sub-agent, or a new evaluation layer.

---

## 0. End-State Architecture (what you're building toward)

Three features map to three distinct workflows, unified under one **Supervisor graph**:

```
                        ┌────────────────────┐
                        │   Supervisor /      │
        user query ───▶ │   Intent Router     │
                        └─────────┬───────────┘
                 ┌────────────────┼────────────────┐
                 ▼                ▼                ▼
        ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
        │  QA Subgraph     │ │ DocGen        │ │ SRS Subgraph     │
        │  (Feature 2)     │ │ Subgraph      │ │ (Feature 3)      │
        │                  │ │ (Feature 1)   │ │                  │
        └──────────────────┘ └──────────────┘ └─────────────────┘
                 │                │                    │
                 └────────────────┴────────────────────┘
                                  ▼
                     Shared Services Layer
        (Tavily client, GitHub client, Vector store,
         Structured-output validator, Diagram validator,
         Doc assembler, Checkpointer/persistence)
```

Why sub-agents (not one flat graph): the three features have almost nothing in common at the node level — QA is a 1–3 hop lookup, DocGen is a research-and-synthesize pipeline, SRS is a stateful multi-turn conversation with fan-out worker agents. Forcing them into one graph produces a bloated state schema and unreadable conditional edges. **Use LangGraph's subgraph-as-node pattern**: each subgraph is compiled independently and invoked as a node from the Supervisor, communicating through a shared top-level state slice.

Keep this section as your north star. Everything below is how you walk toward it without ever facing this whole diagram at once.

---

## Phase 1 — Bare-bones single-node QA agent (no tools)

**Goal:** Get LangGraph, LangSmith tracing, and your project skeleton working before any "agentic" behavior exists.

**Architecture:** One node. `START → answer_node → END`. No routing, no tools, no sub-agents.

**State (TypedDict):**
```python
class QAState(TypedDict):
    query: str
    answer: str
```

**Helper classes:** None yet — just an `LLMClient` thin wrapper (model name, temperature, retry-on-429) so every later phase reuses it instead of calling the SDK raw.

**Structured output:** Not needed. Plain text response.

**Evaluation:** None yet — just confirm every run appears in LangSmith with correct input/output logging. This is your tracing smoke test, not a real eval.

**Definition of done:** You can ask a plain question ("what is dependency injection") and get a streamed answer, visible as a trace in LangSmith.

---

## Phase 2 — Add tool-routing (web search) via structured output

**Goal:** Introduce the first real "agentic" decision: should this query hit the web or not.

**Architecture:**
```
START → classify_intent → (conditional) → [answer_direct | web_search → synthesize] → END
```
Still one subgraph, no sub-agents. This *is* the first slice of your Feature 2 (general QA).

**State (extend):**
```python
class QAState(TypedDict):
    query: str
    needs_web_search: bool
    search_results: list[dict]   # Tavily results
    answer: str
```

**Helper classes:**
- `TavilyClient` — wraps search call, timeout, result truncation/formatting into a citation-friendly block.
- `StructuredOutputNode` (generic wrapper) — takes a Pydantic schema + prompt, calls `.with_structured_output()`, and on validation failure retries once with the error message appended ("repair loop"). Build this now; you'll reuse it in every phase from here on.

**Structured output — yes, first real use:**
```python
class IntentClassification(BaseModel):
    needs_web_search: bool
    reasoning: str   # forces the model to justify itself, improves accuracy
```

**Evaluation — first golden dataset:**
Create `datasets/qa_tool_routing.jsonl`, ~20-30 hand-labeled rows:
```json
{"id": "qa-001", "query": "what is the latest stable version of LangGraph", "expected_needs_web_search": true}
{"id": "qa-002", "query": "explain the difference between a list and a tuple in python", "expected_needs_web_search": false}
```
Metric: routing accuracy (exact match on boolean). Run manually via a small script for now — no LangSmith Experiments integration yet, that comes in Phase 11.

**Definition of done:** Router correctly separates "needs current info" from "general knowledge" questions on your golden set at ≥90% accuracy.

---

## Phase 3 — Add RAG for code-related queries (3-way router)

**Goal:** Handle "fetch related code against user query" — the RAG half of Feature 2.

**Architecture:**
```
START → classify_intent (3-way: none | web_search | rag) → route → [answer_direct | web_search→synthesize | retrieve→synthesize] → END
```

**State (extend):**
```python
class RouteDecision(str, Enum):
    NONE = "none"
    WEB_SEARCH = "web_search"
    RAG = "rag"

class QAState(TypedDict):
    query: str
    route: RouteDecision
    search_results: list[dict]
    retrieved_chunks: list[dict]   # {content, source_file, score}
    answer: str
```

**Helper classes:**
- `CodeIngestionPipeline` — chunking strategy for code (AST-aware or function/class-level chunking, not naive fixed-size — this matters a lot for code retrieval quality), embedding, upsert into vector store.
- `VectorStoreClient` — wraps Chroma/pgvector: `upsert(chunks)`, `query(text, k)`, metadata filtering by file path/language.
- Extend `StructuredOutputNode` usage: `IntentClassification` becomes a 3-way enum instead of boolean.

**Structured output:** Yes — `RouteDecision` enum classification, same repair-loop pattern as Phase 2.

**Evaluation — add retrieval metrics:**
Extend the golden dataset with a `rag` category, and — critically — annotate *which chunks are actually relevant* per query so you can compute:
- **Recall@k** — did the relevant chunk appear in top-k retrieved
- **MRR** (mean reciprocal rank)

This is the first time you're evaluating a *component* (retriever) separately from the *end-to-end* answer — a distinction worth keeping permanently: component-level evals catch retrieval bugs that end-to-end LLM-judge evals mask.

**Definition of done:** 3-way router accuracy ≥85%, retrieval recall@5 ≥80% on your annotated set.

---

## Phase 4 — Harden the QA workflow (error handling, retries, fallbacks)

**Goal:** Make Feature 2 production-grade before moving to Feature 1. This is a "no new capability, all robustness" phase — resist the urge to skip it.

**Architecture:** Same graph shape, but every external-call node gets a fallback edge:
```
web_search → (on failure) → fallback_direct_answer (with caveat: "I couldn't reach live search, answering from training knowledge")
retrieve → (on empty results) → fallback_direct_answer
classify_intent → (on repair-loop exhaustion) → default to NONE, log for review
```

**State (extend):** add `error_log: list[str]` and `retry_count: int` for observability.

**Helper classes:**
- `RetryPolicy` — exponential backoff wrapper around Tavily/vector-store/LLM calls, distinguishing retryable (timeout, 429) from non-retryable (bad request) errors.
- `NodeMetadata` — a small decorator/context manager that attaches `node_name`, `latency_ms`, `success` to every LangSmith run for later cost/latency dashboards.

**Structured output:** No new schemas — but the repair loop from Phase 2 now has a hard cap (e.g. 2 retries) with graceful degradation instead of infinite loop or crash.

**Evaluation:** Add a **robustness eval set** — deliberately malformed/adversarial inputs (empty query, query in another language, prompt-injection-style "ignore previous instructions" test) and assert the graph doesn't crash and doesn't leak system prompt.

**Definition of done:** Killing your network mid-run (simulate Tavily/vector store failure) degrades gracefully instead of throwing.

---

## Phase 5 — GitHub repo analyzer (standalone subgraph, not yet wired to doc gen)

**Goal:** Build the hardest new primitive in Feature 1 in isolation: given a repo URL and a query, decide *which files matter* and summarize them.

**Architecture (new subgraph, tested independently):**
```
START → fetch_repo_tree → select_relevant_files (structured output) → 
  Send() fan-out per file → summarize_file (map) → 
  reduce_summaries → END
```
Use LangGraph's `Send` API here — this is your first genuine fan-out/map-reduce pattern, needed because repos can have far more files than fit in one context window.

**State:**
```python
class RepoAnalysisState(TypedDict):
    repo_url: str
    user_query: str
    file_tree: list[str]                 # flat list of paths from GitHub API
    selected_files: list[FileSelection]  # structured output
    file_summaries: Annotated[list[dict], operator.add]  # reducer for parallel writes
    combined_summary: str
```

**Helper classes:**
- `GitHubRepoClient` — fetch tree via GitHub REST/GraphQL API (respect rate limits, use conditional requests/ETags), fetch individual file content, handle binary/large-file skip logic.
- `FileChunker` — for files too large for one context window, chunk before summarizing (reuse logic conceptually from Phase 3's ingestion pipeline, but no embedding needed here — this is direct summarization, not retrieval).

**Structured output — new schema:**
```python
class FileSelection(BaseModel):
    files_to_analyze: list[str]
    reasoning: str

class FileSummary(BaseModel):
    file_path: str
    summary: str
    relevance_to_query: Literal["high", "medium", "low"]
```
The file-selection step is one of your highest-leverage structured outputs — a bad selection here poisons everything downstream, so this is a good candidate for a stricter validator (e.g. reject if `files_to_analyze` references a path not in `file_tree`, force re-ask).

**Evaluation:** New golden dataset `datasets/repo_file_selection.jsonl` — pick 3-5 real public repos, hand-label "for this query, these are the files a competent engineer would read." Metric: precision/recall on file selection, not just "did it run."

**Definition of done:** Given a repo URL + "how does auth work here," the subgraph selects a sensible file subset and produces per-file summaries — tested standalone via a script, not yet inside the Supervisor.

---

## Phase 6 — DocGen subgraph: combine web + repo analysis + synthesis

**Goal:** Wire Phase 2's web search and Phase 5's repo analyzer together into the actual Feature 1 workflow, with a planning node deciding what's needed.

**Architecture:**
```
START → plan (structured output: needs_web, needs_repo, repo_url?) → 
  (parallel branches, conditionally skipped) 
    web_search_branch ─┐
    repo_analysis_branch ─┤→ join → synthesize_doc → END
```

**State:**
```python
class DocGenState(TypedDict):
    query: str
    needs_web: bool
    needs_repo: bool
    repo_url: str | None
    web_results: list[dict]
    repo_summary: str
    doc_sections: list[DocSection]   # structured
    final_doc_markdown: str
    citations: list[Citation]
```

**Helper classes:**
- `DocAssembler` — takes structured `doc_sections` + citations, renders final Markdown (headers, code blocks, footnote-style citation list). Keep this separate from the LLM generation step — deterministic assembly, not another LLM call, reduces hallucinated formatting.
- `CitationTracker` — every claim in the synthesized doc should trace back to either a web result URL or a repo file path; this class just enforces the bookkeeping.

**Structured output:**
```python
class DocPlan(BaseModel):
    needs_web: bool
    needs_repo: bool
    repo_url: str | None
    reasoning: str

class DocSection(BaseModel):
    heading: str
    content: str
    source_type: Literal["web", "repo", "general_knowledge"]
    source_refs: list[str]
```

**Evaluation:** LLM-as-judge rubric evaluator (your first one) scoring: completeness (did it address the query), grounding (are claims traceable to a source_ref, penalize unsupported claims), structure quality. Keep the rubric as a fixed prompt template you reuse — consistency across runs matters more than a clever one-off prompt.

**Definition of done:** "Document how to set up LangGraph with LangSmith observability" produces a doc that correctly triggers `needs_web=true`, pulls current setup steps, and cites sources — no repo needed for this example, so also test a repo-required example separately.

---

## Phase 7 — Promote to the Supervisor: unify QA + DocGen under one router

**Goal:** This is the architectural pivot point — you now have two independently-working subgraphs; wire them together as the real multi-agent system begins.

**Architecture:**
```
START → supervisor_classify (structured output: qa | doc_gen | srs*) → 
  route to compiled subgraph as a node → END
```
*(`srs` route added in Phase 8; for now it's a 2-way route.)*

This is where the subgraph-as-node pattern from Section 0 actually gets implemented: `QAState` and `DocGenState` become independent, and the Supervisor state only carries what's needed to route + the final output.

**State:**
```python
class SupervisorState(TypedDict):
    query: str
    conversation_history: list[dict]
    route: Literal["qa", "doc_gen", "srs"]
    final_output: str
    thread_id: str
```

**Helper classes:**
- `Checkpointer` setup — now is the right time to add persistence (Postgres or SQLite checkpointer), because Phase 8's HITL requirements-gathering loop *requires* a checkpointer to pause/resume across turns. Wire it in now against the simpler 2-route graph so you debug persistence issues before adding conversational complexity.

**Structured output:**
```python
class RouteDecision(BaseModel):
    route: Literal["qa", "doc_gen", "srs"]
    reasoning: str
```

**Evaluation:** Supervisor-level routing accuracy dataset — this is a *different* golden set than Phase 2/3's tool routing (that was inside QA; this is feature-level routing across all three).

**Definition of done:** A single entry point correctly dispatches to QA vs DocGen, conversation state persists across a process restart (proves the checkpointer works) before you build anything that depends on it.

---

## Phase 8 — SRS subagent, part 1: requirements-gathering loop (HITL)

**Goal:** The hardest UX pattern in the whole project — an agent that asks clarifying questions until it's satisfied, not on a fixed turn count.

**Architecture:**
```
START → ask_clarifying_question → interrupt() [pause for human] → 
  process_answer → check_completeness (structured output) → 
    (loop back to ask_clarifying_question) OR (proceed to Phase 9's outline stage)
```
Use LangGraph's `interrupt()` / `Command(resume=...)` pattern, not a manual while-loop — this is exactly what it's for, and it's what makes the checkpointer from Phase 7 mandatory here.

**State:**
```python
class SRSState(TypedDict):
    conversation: Annotated[list[dict], operator.add]
    gathered_requirements: RequirementsSnapshot   # structured, updated each turn
    completeness: CompletenessCheck
```

**Helper classes:**
- `RequirementsTracker` — maintains the running structured snapshot of what's been gathered (functional requirements, non-functional requirements, stakeholders, constraints), diffed each turn rather than re-extracted from scratch, which is more token-efficient and stabler.

**Structured output — the core mechanism here:**
```python
class CompletenessCheck(BaseModel):
    is_complete: bool
    missing_areas: list[str]
    next_question: str | None

class RequirementsSnapshot(BaseModel):
    functional_requirements: list[str]
    non_functional_requirements: list[str]
    stakeholders: list[str]
    open_questions: list[str]
```
Cap the loop (e.g. max 8 turns) with a forced "proceed with what we have + flag gaps" exit — an infinite clarification loop is a real failure mode worth testing for explicitly.

**Evaluation:** This is genuinely hard to eval automatically. Build a **simulated user** — a second LLM call playing "the software engineer" from a fixed persona/requirements brief, so you can run the clarification loop end-to-end without a human in the loop for every test. Score: turns-to-completion, and recall of the persona's ground-truth requirements against `gathered_requirements` at the end.

**Definition of done:** Given a vague initial ask ("I need an SRS for a delivery app"), the agent asks non-redundant clarifying questions and correctly stops once (simulated) requirements are saturated.

---

## Phase 9 — SRS subagent, part 2: outline → parallel worker agents → diagrams → assembly

**Goal:** Turn gathered requirements into the actual document, using fan-out worker agents — your second use of the `Send()` map pattern, now with generative (not just summarizing) workers.

**Architecture:**
```
[from Phase 8] → plan_outline (structured) → 
  Send() fan-out per section → write_section (worker, parallel) → 
  reduce → generate_diagrams (structured, per diagram needed) → 
    validate_diagram_syntax → (retry on invalid) → 
  assemble_doc → END
```

**State:**
```python
class SRSState(TypedDict):
    # ...from Phase 8
    outline: DocOutline
    section_drafts: Annotated[list[SectionDraft], operator.add]
    diagram_specs: list[DiagramSpec]
    validated_diagrams: list[ValidatedDiagram]
    final_srs_markdown: str
```

**Helper classes:**
- `DiagramValidator` — actually compiles the generated Mermaid/PlantUML (via mermaid-cli or a PlantUML jar/server) rather than trusting the LLM's syntax; on failure, feeds the compiler error back to the LLM for a repair attempt (same repair-loop pattern as your structured-output validator, applied to diagram code instead of JSON).
- `DocAssembler` (extended from Phase 6) — now also handles image placeholder insertion: `![Architecture Diagram](diagrams/architecture.png)` with the actual rendered diagram file saved alongside.

**Structured output:**
```python
class DocOutline(BaseModel):
    sections: list[OutlineSection]  # heading, purpose, needs_diagram: bool, diagram_type

class DiagramSpec(BaseModel):
    diagram_type: Literal["sequence", "flowchart", "class", "component"]
    format: Literal["mermaid", "plantuml"]
    code: str
    caption: str
```

**Evaluation:**
- Diagram validity rate (% that compile without repair, % that compile after ≤1 repair).
- Section-level LLM-judge rubric (same pattern as Phase 6's DocGen judge — reuse the evaluator, don't reinvent it).
- End-to-end: does the assembled SRS cover every item in `gathered_requirements` from Phase 8 (a traceability check you can do programmatically, not just via LLM judge).

**Definition of done:** A full SRS document generates with correctly-compiling diagrams and section content traceable to the gathered requirements.

---

## Phase 10 — Wire SRS into the Supervisor; unify the shared services layer

**Goal:** All three features now exist; this phase is integration, not new capability.

**Architecture:** Supervisor's `route` becomes the full 3-way (`qa | doc_gen | srs`) from Section 0's diagram. Each subgraph is compiled once and imported as a node — no logic duplication.

**Refactor pass (do this explicitly, don't skip it):**
- Collapse the three separate `TavilyClient`/`GitHubRepoClient`/`VectorStoreClient` instantiations into a single `ServiceRegistry` injected into all subgraphs, so credentials/config live in one place.
- Audit that every structured-output node uses the same `StructuredOutputNode` repair-loop wrapper from Phase 2 — by now you'll have 6-7 schemas; consistency here is what makes the system maintainable.

**Evaluation:** First **cross-feature regression suite** — run all golden datasets (routing, RAG, file-selection, doc-gen judge, SRS traceability, diagram validity) in one script, produce one report. This is the point where "golden dataset" stops meaning "one JSONL file" and starts meaning "a directory per feature, one runner."

**Definition of done:** One entry point (`sprinter.run(query, thread_id)`) correctly handles all three feature types, and one command runs the full eval suite.

---

## Phase 11 — Formal evaluation pipeline (LangSmith Datasets + Experiments)

**Goal:** Move from "scripts that print accuracy" to a real, versioned evaluation pipeline you can run as a regression gate.

**Golden dataset structure (standardize across all features now):**
```json
{
  "id": "srs-014",
  "category": "srs_traceability",
  "input": {
    "query": "...",
    "conversation_context": [ /* prior turns, for multi-turn cases */ ]
  },
  "expected": {
    "route": "srs",
    "requirements_ground_truth": ["...", "..."],
    "rubric_notes": "must include a data retention section given persona mentions user data"
  },
  "metadata": { "difficulty": "medium", "added_date": "2026-08-01", "source": "manual" }
}
```
Upload each category as a LangSmith Dataset; write one **custom evaluator function per metric** (routing accuracy, retrieval recall@k, LLM-judge rubric score, diagram compile rate, requirements traceability) and register them as LangSmith evaluators so every run against a dataset produces a comparable Experiment.

**Helper classes:**
- `EvalRunner` — CLI entry point: `python -m sprinter.eval --category doc_gen --compare-to <previous_experiment_id>`, so you can diff scores across prompt/graph changes.

**Structured output:** No new agent schemas, but your **evaluator outputs** should themselves be structured (`JudgeScore(score: int, justification: str)`) so results are analyzable, not free text.

**Definition of done:** Changing a prompt and re-running the suite shows a clear before/after score diff per category, not just pass/fail.

---

## Phase 12 — Production hardening

**Goal:** The last phase — nothing new architecturally, everything about reliability and deployability.

- **API layer:** FastAPI wrapper exposing `/chat` (streaming, SSE) and `/threads/{id}` for resuming interrupted SRS sessions.
- **Observability:** full LangSmith tracing tags per subgraph/node, cost-per-request and latency dashboards (built on the `NodeMetadata` from Phase 4).
- **Guardrails:** input length limits, per-user rate limiting, token/cost budgets per request with hard cutoffs, prompt-injection screening on any content pulled from GitHub files or web search before it reaches synthesis nodes (treat repo/web content as untrusted input, same principle as the Phase 4 adversarial eval set).
- **Persistence:** move the Phase 7 checkpointer to a real Postgres instance; add TTL/cleanup for stale SRS sessions.
- **Caching:** semantic cache in front of Tavily calls and repo-tree fetches (same repo analyzed twice in an hour shouldn't re-fetch).
- **Continuous eval growth:** log low-confidence or human-corrected outputs from production back into the golden datasets from Phase 11 — this closes the loop and is the difference between a static eval suite and one that actually improves over time.

**Definition of done:** Deployed behind an API, degrades gracefully under load/failure, and every prompt change goes through the Phase 11 regression suite before shipping.

---

## Suggested build order recap

| Phase | New complexity axis | Sub-agents introduced |
|---|---|---|
| 1 | Skeleton + tracing | none |
| 2 | Conditional routing + structured output | none |
| 3 | RAG retrieval | none |
| 4 | Error handling / robustness | none |
| 5 | Map-reduce fan-out (`Send`) | repo analyzer (standalone) |
| 6 | Multi-source synthesis | DocGen subgraph |
| 7 | Hierarchical orchestration + persistence | Supervisor |
| 8 | Human-in-the-loop (`interrupt`) | requirements-gathering agent |
| 9 | Parallel generative workers | outline/worker/diagram agents |
| 10 | Integration & shared services | — (refactor) |
| 11 | Formal eval pipeline | — |
| 12 | Production hardening | — |

This ordering front-loads the two hardest *mechanisms* (map-reduce fan-out in Phase 5, HITL interrupts in Phase 8) each in isolation, before they ever have to coexist with the rest of the system — that's deliberate, since debugging a new LangGraph pattern is much harder once it's tangled up with routing logic from three other features.
