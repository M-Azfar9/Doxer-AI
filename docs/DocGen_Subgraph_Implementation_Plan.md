# DocGen Subgraph Agent — Phased Implementation Plan & Deep Architectural Specification
### Architecture D: Hierarchical Supervisor with Bounded, Sufficiency-Gated Subgraphs + Grounding Critique

---

## 1. Executive Overview & System Objective

The **DocGen Subgraph Agent** is an autonomous, production-grade technical documentation generator designed for software engineering systems. It takes developer queries (e.g., *"Document how authentication and session management work in this repo"*, *"Generate an API reference for our order processing service"*, or *"Create a developer quickstart guide for integrating the payment webhook"*) and outputs rigorous, fully grounded Markdown documentation.

To prevent the classic failure modes of autonomous doc-generation systems—**blind directory wandering**, **unbounded loops/token explosion**, and **hallucinated API signatures or claims**—this system is built upon **Architecture D: Hierarchical Supervisor with Bounded, Sufficiency-Gated Subgraphs + Grounding Critique**.

```
                           ┌─────────────────────────┐
                           │   repo_map_generator    │
                           │  (Deterministic Seed)   │
                           └────────────┬────────────┘
                                        │ repo_map / symbols
                                        ▼
                           ┌─────────────────────────┐
                           │    doc_intent_router    │
                           │   (Structured Intent)   │
                           └────────────┬────────────┘
                                        │ LangGraph Send()
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
      ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
      │  GitHub Specialist │ │ FileSystem (Local) │ │   Web Research     │
      │     Subgraph       │ │ Specialist Subgraph│ │ Specialist Subgraph│
      │  ┌──────────────┐  │ │  ┌──────────────┐  │ │  ┌──────────────┐  │
      │  │  gh_explore  │  │ │  │  fs_explore  │  │ │  │  web_explore │  │
      │  └──────┬───────┘  │ │  └──────┬───────┘  │ │  │ (Tavily/Exa) │  │
      │         ▼          │ │         ▼          │ │  └──────┬───────┘  │
      │  ┌──────────────┐  │ │  ┌──────────────┐  │ │         ▼          │
      │  │sufficiency   │  │ │  │sufficiency   │  │ │  ┌──────────────┐  │
      │  │    gate      │  │ │  │    gate      │  │ │  │sufficiency   │  │
      │  └──────┬───────┘  │ │  └──────┬───────┘  │ │  │    gate      │  │
      │         ▼          │ │         ▼          │ │  └──────┬───────┘  │
      │  ┌──────────────┐  │ │  ┌──────────────┐  │ │         ▼          │
      │  │evidence_pack │  │ │  │evidence_pack │  │ │  ┌──────────────┐  │
      │  └──────────────┘  │ │  └──────────────┘  │ │  │evidence_pack │  │
      │                    │ │ (AST Signatures)   │ │  └──────────────┘  │
      └─────────┬──────────┘ └─────────┬──────────┘ └─────────┬──────────┘
                │                      │                      │
                │ GitHubEvidence       │ FSEvidence           │ WebEvidence
                └──────────────────────┼──────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │    evidence_aggregator    │
                         │(Merged Synthesis Context) │
                         └─────────────┬─────────────┘
                                       │ merged_evidence
                                       ▼
                         ┌───────────────────────────┐
                         │   template_aware_docgen   │
                         │ (Arch | API Ref | Guide)  │
                         └─────────────┬─────────────┘
                                       │ draft_markdown
                                       ▼
                         ┌───────────────────────────┐
                         │      grounding_critic     │
                         │  (Audit against Evidence) │
                         └──────┬─────────────▲──────┘
                     Grounded?  │             │ Refined Evidence
                                │ No          │
                                ▼             │
                         ┌────────────────────┴──────┐
                         │ targeted_refinement_node  │
                         │ (Max 1–2 Targeted Hops)   │
                         └───────────────────────────┘
                                │ Yes (or max refinements hit)
                                ▼
                         ┌───────────────────────────┐
                         │    final_doc_emitter      │
                         │  (Clean Cleaned Markdown) │
                         └───────────────────────────┘
```

---

## 2. The 3-Phase Implementation Strategy

To build this system reliably without encountering massive graph debugging overhead, we construct it in **3 progressive phases in isolation**, before mounting it into the Phase 7 Top-Level Supervisor alongside the QA Agent.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Baseline Linear DocGen Subgraph                                     │
│ • Single compiled graph with direct tool routing (GitHub, FS, Tavily, Exa)  │
│ • Flat state, single-pass exploration, single template generation           │
│ • Objective: Validate all 4 MCP connections and baseline document synthesis │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Progressive Enhancement
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Isolated Specialist Subgraphs with Bounded Sufficiency Gates       │
│ • Decouple into 3 isolated subgraphs via LangGraph Send()                   │
│ • Add state hop counter + structured SufficiencyCheck gate per specialist   │
│ • Return typed EvidencePackages into an Aggregator node                     │
│ • Objective: Achieve bounded, adaptive multi-hop exploration per source     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Progressive Hardening
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Hardened Architecture D (Template-Aware + Grounding Critic)        │
│ • Deterministic AST symbol extractor for FileSystem specialist              │
│ • Multi-template prompt specialization (Architecture vs API vs Tutorial)   │
│ • Grounding critique node with 1–2 hop targeted refinement loop             │
│ • Package as a plug-and-play Subgraph Node for the Phase 7 Supervisor        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Component-by-Component Architectural Breakdown

Every node in Architecture D has a **single, strictly bounded responsibility**, deterministic inputs, and typed outputs. Below is the complete, node-by-node specification.

---

### Component 1: `repo_map_generator` (Deterministic Seed Node)

- **What it does:**
  Executes a non-agentic, deterministic pre-pass on the target codebase (either remote GitHub repository via the GitHub client or local repository via the FileSystem client). Instead of reading entire files, it builds a lightweight structural skeleton: directory layout (depth-limited to 3 levels) and top-level file names. This is fed directly to the router and specialists to eliminate the costly "blind wandering" pattern where an agent spends its first 2–3 hops guessing directory names.
- **Input State:**
  ```python
  class SeedInput(TypedDict):
      repo_url: str | None
      local_path: str | None
      user_query: str
  ```
- **Output State Update:**
  ```python
  class SeedOutput(TypedDict):
      repo_map: str  # Formatted ASCII tree or compact JSON path hierarchy
      detected_tech_stack: list[str]  # e.g., ["Python", "FastAPI", "Docker"]
  ```

---

### Component 2: `doc_intent_router` (Structured Classification & Scope Planner)

- **What it does:**
  Takes the user query and the structural `repo_map`, and makes an explicit, structured routing decision. It determines:
  1. Which specialists must be engaged (`needs_github`, `needs_filesystem`, `needs_web_research`).
  2. The precise **scope instruction** for each specialist (e.g., *"Focus exclusively on the `/auth` directory, JWT verification middleware, and user models; ignore frontend assets"*).
  3. The target documentation template (`doc_type`: `architecture_explainer`, `api_reference`, or `tutorial_quickstart`).
- **Input State:**
  ```python
  class RouterInputState(TypedDict):
      user_query: str
      repo_map: str
      detected_tech_stack: list[str]
  ```
- **Structured LLM Output Schema (`DocIntentPlan`):**
  ```python
  class SpecialistScope(BaseModel):
      enabled: bool = Field(description="Whether this specialist should be invoked")
      focus_scope: str = Field(description="Strict scope guidance for the specialist")
      target_paths_or_topics: list[str] = Field(default_factory=list)

  class DocIntentPlan(BaseModel):
      doc_type: Literal["architecture_explainer", "api_reference", "tutorial_quickstart"]
      needs_github: SpecialistScope
      needs_filesystem: SpecialistScope
      needs_web: SpecialistScope
      planning_rationale: str
  ```
- **Output State Update:**
  ```python
  class RouterOutputState(TypedDict):
      intent_plan: DocIntentPlan
      doc_type: str
  ```

---

### Component 3: `fan_out_dispatcher` (LangGraph `Send` Fan-Out)

- **What it does:**
  A conditional routing edge using LangGraph's `Send()` API. Inspects `intent_plan` and spawns parallel executions to the selected specialist subgraphs. If a specialist is disabled, no tasks are queued for it.
- **Input State:**
  Reads `intent_plan`, `user_query`, `repo_map`, `repo_url`, and `local_path`.
- **Output:**
  Returns a list of `Send` primitives:
  ```python
  def fan_out_dispatcher(state: DocGenState) -> list[Send]:
      sends = []
      plan = state["intent_plan"]
      if plan.needs_github.enabled:
          sends.append(Send("github_specialist_subgraph", {
              "repo_url": state["repo_url"],
              "scope": plan.needs_github.focus_scope,
              "target_paths": plan.needs_github.target_paths_or_topics,
              "repo_map": state["repo_map"],
              "hop_count": 0
          }))
      if plan.needs_filesystem.enabled:
          sends.append(Send("filesystem_specialist_subgraph", {
              "local_path": state["local_path"],
              "scope": plan.needs_filesystem.focus_scope,
              "target_paths": plan.needs_filesystem.target_paths_or_topics,
              "repo_map": state["repo_map"],
              "hop_count": 0
          }))
      if plan.needs_web.enabled:
          sends.append(Send("web_research_specialist_subgraph", {
              "user_query": state["user_query"],
              "scope": plan.needs_web.focus_scope,
              "search_topics": plan.needs_web.target_paths_or_topics,
              "hop_count": 0
          }))
      return sends
  ```

---

### Component 4: Specialist Subgraphs (Bounded ReAct with Sufficiency Gating)

Each specialist runs an internal loop capped by a strict counter (`MAX_HOPS = 3`).

```
              ┌───────────────┐
              │  explore_step │ ◄─────────────────────┐
              └───────┬───────┘                       │
                      │ updates raw findings          │
                      ▼                               │ Insufficient &
              ┌───────────────┐                       │ hop_count < MAX_HOPS
              │  sufficiency  ├───────────────────────┘
              │     gate      │
              └───────┬───────┘
                      │ Sufficient OR hop_count >= MAX_HOPS
                      ▼
              ┌───────────────┐
              │ evidence_pack │
              └───────────────┘
```

#### 4.1. GitHub Specialist Subgraph
- **Tools**: GitHub MCP Server (`get_file_contents`, `list_directory`, `search_repositories`, `get_commit_history`) via `GitHubMCPClient`.
- **Nodes**:
  1. `gh_explore_step`: Invokes tools based on the scoped instruction. Increments `hop_count += 1`.
  2. `gh_sufficiency_gate`: Structured-output LLM check evaluating whether the collected files and commit details sufficiently cover the scope.
     ```python
     class SufficiencyCheck(BaseModel):
         is_sufficient: bool
         confidence_score: float = Field(ge=0.0, le=1.0)
         missing_information: str | None
         next_tool_guidance: str | None
     ```
  3. `gh_evidence_packager`: Compresses raw exploration into a clean, structured package.
- **Output Schema (`GitHubEvidencePackage`):**
  ```python
  class GitHubEvidencePackage(BaseModel):
      files_examined: list[str]
      key_abstractions: list[dict]  # [{"name": "AuthManager", "role": "token issuance", "path": "src/auth.py"}]
      relevant_commits: list[str]
      architectural_notes: list[str]
  ```

#### 4.2. FileSystem Specialist Subgraph (with AST Symbol Extraction)
- **Tools**:
  - FileSystem MCP Server (`read_file`, `list_directory`).
  - **AST / Tree-Sitter Symbol Extractor Tool** (`extract_symbols`): Directly parses Python/JS files to extract function signatures, class hierarchies, route decorators (`@router.get`, `@app.post`), and docstrings without dumping raw file text into context.
- **Nodes**:
  1. `fs_explore_step`: Reads directories or calls `extract_symbols` on key candidate files. Increments `hop_count += 1`.
  2. `fs_sufficiency_gate`: Evaluates whether endpoints and models are sufficiently discovered.
  3. `fs_evidence_packager`: Formats the extracted symbol definitions.
- **Output Schema (`FSEvidencePackage`):**
  ```python
  class APIEndpointSymbol(BaseModel):
      method: str  # GET, POST, etc.
      path: str    # /api/v1/auth/login
      handler_name: str
      decorators: list[str]
      docstring: str | None
      signature: str

  class FSEvidencePackage(BaseModel):
      scanned_paths: list[str]
      endpoints: list[APIEndpointSymbol]
      models: list[dict]  # Pydantic/SQLAlchemy models with field names and types
      entry_points: list[str]
  ```

#### 4.3. Web Research Specialist Subgraph (Tavily MCP + Exa MCP)
- **Tools**:
  - **Tavily MCP**: Quick factual verification, official version checks, changelogs.
  - **Exa MCP**: Deep semantic crawl, neural content search, retrieving real-world developer tutorials and production code examples.
- **Nodes**:
  1. `web_explore_step`: Issues Tavily for fast facts and Exa for deep code context. Increments `hop_count += 1`.
  2. `web_sufficiency_gate`: Checks if official documentation, version compatibility, and code examples have been acquired.
  3. `web_evidence_packager`: Compiles web evidence with exact source citations.
- **Output Schema (`WebEvidencePackage`):**
  ```python
  class WebEvidencePackage(BaseModel):
      claims_supported: list[str]
      sources: list[dict]  # [{"title": "FastAPI Docs", "url": "https://fastapi.tiangolo.com/..."}]
      verified_code_examples: list[str]
      compatibility_notes: list[str]
  ```

---

### Component 5: `evidence_aggregator` (Synthesis Context Normalizer)

- **What it does:**
  Merges the outputs from the three parallel specialists into a unified, compact, and strongly typed `MergedEvidenceContext`. It performs deduplication, aligns local FileSystem AST endpoints with remote GitHub commits, and pairs claims with verified URLs. It ensures the downstream generator receives **zero raw tool logs**, strictly bounded token payload, and high semantic density.
- **Input State:**
  ```python
  class AggregatorInput(TypedDict):
      github_evidence: GitHubEvidencePackage | None
      fs_evidence: FSEvidencePackage | None
      web_evidence: WebEvidencePackage | None
  ```
- **Output State Update:**
  ```python
  class MergedEvidenceContext(BaseModel):
      abstractions: list[dict]
      endpoints_and_interfaces: list[APIEndpointSymbol]
      data_models: list[dict]
      external_references: list[dict]
      code_snippets: list[dict]
      evidence_summary: str
  ```

---

### Component 6: `template_aware_docgen` (Specialized Documentation Synthesizer)

- **What it does:**
  Synthesizes the technical documentation from the `MergedEvidenceContext`. Crucially, it selects a tailored prompt template based on `doc_type` rather than using a generic catch-all prompt:
  - **`architecture_explainer`**: Focuses on component topology, data flow, sequence of operations, state management, and design patterns.
  - **`api_reference`**: Focuses on endpoints, request/response schemas, error status codes, headers, and code examples.
  - **`tutorial_quickstart`**: Focuses on prerequisites, step-by-step installation, environment variables, minimal working examples, and verification steps.
  Enforces inline citation tags linked to evidence items (e.g., `[^fs-auth_middleware:24]`, `[^gh-commit-a8f1]`, `[^web-fastapi-docs]`).
- **Input State:**
  ```python
  class DocGenInput(TypedDict):
      doc_type: str
      user_query: str
      merged_evidence: MergedEvidenceContext
  ```
- **Output State Update:**
  ```python
  class DocGenOutput(TypedDict):
      draft_markdown: str
      draft_sections: list[dict]
  ```

---

### Component 7: `grounding_critic` (Rigorous Verification & Hallucination Auditor)

- **What it does:**
  Acts as an impartial verification judge before any documentation is emitted. It parses every factual assertion, function name, API route, and parameter in `draft_markdown` and cross-examines it against the `MergedEvidenceContext`.
  If an assertion cannot be found in the evidence packages (e.g., hallucinated endpoint `@router.delete("/users/{id}/purge")` that does not exist in the AST symbol list):
  - Flags the claim as `unsupported`.
  - Determines if the draft passes grounding (`is_grounded`).
  - Generates targeted refinement queries for the specific missing gap.
- **Input State:**
  ```python
  class CriticInput(TypedDict):
      draft_markdown: str
      merged_evidence: MergedEvidenceContext
      refinement_count: int
  ```
- **Structured LLM Output Schema (`GroundingReport`):**
  ```python
  class GroundingViolation(BaseModel):
      claim: str
      location: str  # Section or line
      issue: Literal["hallucination", "incorrect_signature", "unsupported_assertion", "missing_reference"]
      correction_instruction: str

  class GroundingReport(BaseModel):
      is_grounded: bool
      confidence_score: float
      violations: list[GroundingViolation]
      targeted_refinement_queries: list[str]  # e.g., ["Verify if /users/{id}/purge exists in controllers/"]
  ```
- **Conditional Routing:**
  - If `is_grounded == True` OR `refinement_count >= MAX_REFINEMENTS (2)`: Transition to `final_doc_emitter`.
  - If `is_grounded == False` AND `refinement_count < MAX_REFINEMENTS`: Transition to `targeted_refinement_node`.

---

### Component 8: `targeted_refinement_node` (Bounded 1–2 Hop Gap-Filling)

- **What it does:**
  Executes a narrow, laser-targeted investigation solely for the items flagged in `violations`. It does NOT re-run the entire pipeline. It calls the specific tool relevant to the query (e.g., checks one specific file via `read_file` or asks Tavily/Exa for a specific version clarification), updates the `merged_evidence`, increments `refinement_count += 1`, and routes back to `template_aware_docgen` for section patch-up.
- **Input State:**
  ```python
  class RefinementInput(TypedDict):
      violations: list[GroundingViolation]
      targeted_refinement_queries: list[str]
      merged_evidence: MergedEvidenceContext
      refinement_count: int
  ```
- **Output State Update:**
  ```python
  class RefinementOutput(TypedDict):
      merged_evidence: MergedEvidenceContext  # Enriched with the targeted gap-filling findings
      refinement_count: int  # Incremented
  ```

---

### Component 9: `final_doc_emitter` (Formatter & Exporter)

- **What it does:**
  Deterministic formatting node. Assembles the final Markdown:
  - Generates Table of Contents.
  - Cleans citation keys into standardized markdown footnotes.
  - Generates Mermaid diagram blocks if architectural patterns are documented.
  - Adds generation metadata (timestamp, model used, sources consulted, grounding confidence score).
- **Input State:**
  Reads `draft_markdown`, `grounding_report`, and `merged_evidence`.
- **Output State Update:**
  ```python
  class FinalDocOutput(TypedDict):
      final_doc_markdown: str
      citations_manifest: list[dict]
      grounding_score: float
  ```

---

## 4. The 3 Implementation Phases in Deep Detail

---

### Phase 1: Baseline Linear DocGen Subgraph

**Goal:** Stand up the basic end-to-end DocGen pipeline in isolation. Verify all 4 MCP connections (GitHub, FileSystem, Tavily, Exa) and ensure that a user query generates a coherent technical document in a single execution pass without complex loops.

#### Graph Architecture
```
START → repo_map_generator → doc_intent_router → direct_tool_fetch → baseline_docgen → END
```

#### What Gets Built:
1. **MCP Connectors**:
   - Wrap GitHub MCP (`@modelcontextprotocol/server-github`) with local fallback.
   - Wrap FileSystem MCP (`@modelcontextprotocol/server-filesystem`).
   - Wrap Tavily search MCP/API for quick web queries.
   - Wrap Exa MCP (`exa-mcp-server`) for semantic deep search.
2. **Simple Router**: Direct LLM structured classification checking whether remote repo, local files, or web are required.
3. **Single Retrieval Pass**: Sequential tool execution fetching top-level file trees, key file contents, and search snippets.
4. **Baseline Synthesis**: A single LLM prompt generating standard Markdown documentation with basic citations.

#### Definition of Done (Phase 1):
- A standalone script `test_docgen_phase1.py` can be executed against a real GitHub repo or local path.
- The pipeline connects to all 4 MCP servers without hanging.
- Produces a well-structured 500+ word Markdown document with citations.
- Traced cleanly in LangSmith under project `DevDocs-AI`.

---

### Phase 2: Specialist Subgraphs with Bounded Sufficiency Gates & Fan-Out

**Goal:** Decouple monolithic tool calling into 3 isolated specialist subgraphs running in parallel via LangGraph `Send()`. Introduce bounded exploration loops with structured sufficiency gates and strict hop limits.

#### Graph Architecture
```
START → repo_map_generator → doc_intent_router (with scopes)
      → Send() Fan-Out:
          ├─► GitHub Specialist Subgraph (bounded ReAct + Sufficiency Gate)
          ├─► FileSystem Specialist Subgraph (bounded ReAct + Sufficiency Gate)
          └─► Web Research Specialist Subgraph (bounded ReAct + Sufficiency Gate)
      → Join in evidence_aggregator → baseline_docgen → END
```

#### What Gets Built:
1. **Specialist Subgraph Definitions**:
   - `github_specialist_subgraph`: State with `hop_count`, `max_hops=3`, `files_accumulated`.
   - `filesystem_specialist_subgraph`: State with `hop_count`, `max_hops=3`, `local_entries`.
   - `web_research_specialist_subgraph`: Dual-tier search coordinating Tavily and Exa.
2. **Sufficiency Gate Pattern**:
   - A dedicated `StructuredOutputNode(SufficiencyCheck)` called after each tool round.
   - Decides: `is_sufficient=True` (exit early to packager) or `is_sufficient=False` (loop back for another hop, unless `hop_count >= 3`).
3. **Structured Evidence Packages**:
   - `GitHubEvidencePackage`, `FSEvidencePackage`, `WebEvidencePackage`.
4. **`evidence_aggregator` Node**:
   - Merges evidence packages into `MergedEvidenceContext`.

#### Definition of Done (Phase 2):
- Specialist subgraphs run concurrently in LangGraph traces via `Send()`.
- If sufficiency is reached on hop 1, the specialist stops early (saves tokens).
- If sufficiency fails, it stops strictly at hop 3 (no runaway loops).
- Aggregator successfully creates a structured, compact evidence state.

---

### Phase 3: Hardened Architecture D (Template Specialization + Grounding Critique)

**Goal:** Fully harden the agent into Architecture D. Add the AST symbol extractor for high-efficiency code analysis, template-specialized doc generation, and the Grounding Critic with a targeted 1–2 hop refinement loop.

#### Graph Architecture
```
START → repo_map_generator → doc_intent_router (with doc_type)
      → Parallel Specialist Subgraphs (with AST tool in FS)
      → evidence_aggregator
      → template_aware_docgen (Architecture vs API vs Tutorial)
      ▼
      grounding_critic ◄─────────────────────────┐
      │                                          │
      ├─► (if grounded or max refinements) ──► final_doc_emitter → END
      │
      └─► (if ungrounded & refinement < 2) ──► targeted_refinement_node ─┘
```

#### What Gets Built:
1. **AST Symbol Extractor Tool**:
   - A Python AST / tree-sitter utility integrated into the FileSystem specialist's toolkit.
   - Extracts `@router.get`, `@app.post`, function parameters, return types, class definitions, and docstrings.
2. **Template-Aware Prompts**:
   - 3 distinct system prompts: `PROMPT_ARCHITECTURE_EXPLAINER`, `PROMPT_API_REFERENCE`, `PROMPT_TUTORIAL_QUICKSTART`.
3. **`grounding_critic` Node**:
   - Evaluates draft claims against `MergedEvidenceContext`.
   - Flags hallucinations and produces `GroundingReport`.
4. **`targeted_refinement_node`**:
   - Issues 1–2 precision queries to patch flagged gaps.
5. **`final_doc_emitter`**:
   - Emits finalized Markdown documentation with verified citations.

#### Definition of Done (Phase 3):
- Complete Architecture D running autonomously in isolation.
- Simulated hallucination test (deliberately seeding an unsupported claim) triggers the Grounding Critic, runs targeted refinement, and corrects or eliminates the false claim.
- Token consumption per run is predictable due to AST filtering and bounded loops.
- Generates high-fidelity Architecture, API, and Tutorial documents.

---

## 5. Connecting with Phase 7 Supervisor (Top-Level Multi-Agent Graph)

Once Phase 3 is completed, the entire DocGen graph compiles into a **single self-contained subgraph node** (`docgen_subgraph_node`).

In Phase 7 of `Sprinter_Implementation_Plan.md`, the Supervisor routes between the **QA Agent** and the **DocGen Subgraph Agent**:

```
                         ┌────────────────────┐
                         │   Supervisor /     │
         user query ───▶ │   Intent Router    │
                         └─────────┬──────────┘
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
         ┌────────────────┐                ┌──────────────────┐
         │    QA Agent    │                │  DocGen Subgraph │
         │   (Feature 2)  │                │  (Architecture D)│
         └────────────────┘                └──────────────────┘
```

### Supervisor State Interface Contract
```python
class SupervisorState(TypedDict):
    query: str
    target_repo_url: str | None
    target_local_path: str | None
    route: Literal["qa", "doc_gen"]
    final_output: str
    citations: list[dict]
```

When `route == "doc_gen"`, the supervisor directly calls:
```python
# Compiled DocGen Subgraph as a node
builder.add_node("doc_gen", compiled_docgen_subgraph)
```

The DocGen subgraph handles its own internal routing, specialist fan-out, sufficiency loops, aggregation, template generation, critique, and refinement, returning clean Markdown directly into `SupervisorState["final_output"]`.

---

## 6. Directory Structure for Implementation

```
d:\Projects\DevDocs AI\
├── src\
│   ├── docgen\
│   │   ├── __init__.py
│   │   ├── state.py                  # TypedDicts and Pydantic schemas (all evidence contracts)
│   │   ├── mcp_tools\
│   │   │   ├── github_tool.py        # GitHub MCP wrapper
│   │   │   ├── filesystem_tool.py    # FileSystem MCP wrapper
│   │   │   ├── ast_symbol_parser.py  # AST/Tree-sitter signature extractor
│   │   │   ├── tavily_tool.py        # Tavily search MCP/API wrapper
│   │   │   └── exa_tool.py           # Exa deep research MCP wrapper
│   │   ├── specialists\
│   │   │   ├── github_specialist.py  # Subgraph with sufficiency gate
│   │   │   ├── fs_specialist.py      # Subgraph with AST + sufficiency gate
│   │   │   └── web_specialist.py     # Subgraph with dual search + sufficiency gate
│   │   ├── nodes\
│   │   │   ├── repo_map.py           # Pre-pass ASCII/JSON repo layout
│   │   │   ├── router.py             # Structured intent & scope planner
│   │   │   ├── aggregator.py         # Merges evidence packages
│   │   │   ├── docgen.py             # Template-aware generator
│   │   │   ├── critic.py             # Grounding critique auditor
│   │   │   ├── refinement.py         # Targeted 1-2 hop query executor
│   │   │   └── emitter.py            # Final formatting & footnotes
│   │   └── graph.py                  # Assembles and compiles the full DocGen subgraph
├── tests\
│   ├── test_docgen_phase1.py         # Test linear baseline
│   ├── test_docgen_phase2.py         # Test specialist fan-out & sufficiency gates
│   └── test_docgen_phase3.py         # Test critique, AST extraction & full Arch D
├── DocGen_Subgraph_Implementation_Plan.md
└── Sprinter_Implementation_Plan.md
```
