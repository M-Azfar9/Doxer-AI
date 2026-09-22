# SRS-Generation Subagent — Phased Implementation Plan & Deep Architectural Specification
### Architecture B: Two-Phase Composed Subgraphs (Clarification Subgraph + Generation Subgraph)

---

## 1. Executive Overview & System Objective

The **SRS-Generation Subagent** is a specialized, autonomous technical agent within the **Sprinter** AI documentation assistant ecosystem. Its primary mission is twofold:
1. **Intelligent Requirements Clarification:** Engage a software engineer in an active, stateful, multi-turn dialogue to extract, disambiguate, and structure ambiguous software requirements into a formal requirements model.
2. **Autonomous SRS Generation:** Transform those structured requirements into a publication-ready, IEEE 830-compliant Software Requirements Specification (SRS) document, complete with parallel-drafted functional sections and compiler-validated architectural diagrams (Mermaid / PlantUML).

### The Core Problem: Why Monolithic / Flat Graphs Fail for SRS
Traditional monolithic agent graphs that attempt requirements gathering and document generation in a single flat state graph suffer from four catastrophic failure modes:
1. **The LangGraph Interrupt Replay Token Burn:** When an agent pauses for human input via `interrupt()`, LangGraph replays the interrupted node from the beginning upon resumption. If question generation and human interruption share the same node, every answer provided by the human causes the agent to re-run the expensive question-generation LLM call, burning redundant tokens and risking prompt drift.
2. **$O(N^2)$ Context Explosion during Clarification:** Naive conversational agents pass the entire chat history on every turn. As the conversation progresses, token costs grow quadratically.
3. **Diagram Syntax Fragility:** LLMs frequently generate invalid Mermaid or PlantUML syntax. Without isolated compiler-in-the-loop validation and bounded repair cycles, broken diagram code leaks directly into the output document.
4. **State Schema Pollution:** Conflating conversational state (`conversation_log`, `turn_count`, `completeness`) with document drafting state (`outline`, `section_drafts`, `diagram_specs`, `validated_diagrams`) produces an unmaintainable, tightly coupled system where changing a prompt in the interview loop can inadvertently corrupt document assembly.

### The Solution: Architecture B (Two-Phase Composed Subgraphs)
To eliminate these issues, the SRS Subagent is architected as **Two-Phase Composed Subgraphs**. 
- The parent graph contains exactly two operational nodes: `run_clarification` and `run_generation`.
- `run_clarification` executes a self-contained compiled graph (`clarification_graph`) that owns conversation and requirements extraction.
- `run_generation` executes a separate self-contained compiled graph (`generation_graph`) that owns document drafting, parallel fan-out, diagram compilation, and assembly.
- The boundary crossing between the two subgraphs is strictly guarded: only the validated `RequirementsModel` and a list of `unresolved_gaps` cross from clarification to generation.

---

## 2. End-State Architecture Specification

```
====================================================================================================
                                      PARENT SRS GRAPH
====================================================================================================

      START
        │
        ▼
 ┌──────────────────────┐
 │  run_clarification   │  ◄── Invokes compiled clarification_graph (owns HITL & requirements)
 └──────────┬───────────┘
            │
            ▼
    [route_parent] ─────────── (status == "failed" / unrecoverable) ──────────► END (Failed)
            │
            ▼ (status == "completed" or gaps_carried_forward)
 ┌──────────────────────┐
 │    run_generation    │  ◄── Invokes compiled generation_graph (owns sections, diagrams, assembly)
 └──────────┬───────────┘
            │
            ▼
           END (Completed SRS Document + Rendered Diagrams)


====================================================================================================
                        PHASE 1 SUBGRAPH: CLARIFICATION GRAPH (Stateful HITL)
====================================================================================================

      START
        │
        ▼
 ┌──────────────────────────────┐
 │ analyze_initial_requirements │  ◄── Ingests initial prompt & context; seeds RequirementsModel
 └──────────────┬───────────────┘
                │
                ▼
      ┌───────────────────┐
  ┌──►│ check_completeness│  ◄── Structured CompletenessCheck (missing_areas, next_question)
  │   └─────────┬─────────┘
  │             │
  │     [route_completeness]
  │       ├── (is_complete == True) ──────────────► END (Requirements Saturated)
  │       ├── (turn_count >= max_turns) ──────────► END (Cap Reached; Copy gaps to unresolved_gaps)
  │       │
  │       └── (Needs Clarification & turns remain)
  │             │
  │             ▼
  │      ┌──────────────┐
  │      │ ask_question │  ◄── PURE INTERRUPT: answer = interrupt(next_question); NO LLM CALL!
  │      └──────┬───────┘
  │             │ human response received via Command(resume=answer)
  │             ▼
  │      ┌───────────────────┐
  └──────┤update_requirements│  ◄── Delta-update: Merges (current RequirementsModel + latest answer)
         └───────────────────┘      Keeps token usage strictly FLAT per turn!


====================================================================================================
                    PHASE 2 SUBGRAPH: GENERATION GRAPH (Parallel Workers + Compiler)
====================================================================================================

      START (Receives frozen RequirementsModel + unresolved_gaps)
        │
        ▼
 ┌──────────────────────┐
 │   generate_outline   │  ◄── Generates IEEE 830 section outline (resilient to thin requirements)
 └──────────┬───────────┘
            │
            │ LangGraph Send() Fan-Out (one task per section)
            ▼
 ┌──────────────────────┐
 │   generate_section   │  ◄── Parallel generative workers (frozen snapshot of requirements)
 └──────────┬───────────┘
            │
            │ LangGraph Reducer (aggregates section_drafts)
            ▼
 ┌──────────────────────┐
 │  determine_diagrams  │  ◄── Plans needed architectural diagrams (Architecture, Sequence, ERD)
 └──────────┬───────────┘
            │
            │ LangGraph Send() Fan-Out (one task per diagram)
            ▼
 ┌──────────────────────┐
 │   generate_diagram   │ ◄───────────────────────────────────────────────┐
 └──────────┬───────────┘                                                 │ (Compiler Error
            │                                                             │  Feedback Loop)
            ▼                                                             │
 ┌──────────────────────┐                                                 │
 │   validate_diagram   │ ──► Subprocess CLI (mmdc / plantuml)            │
 └──────────┬───────────┘                                                 │
            │                                                             │
     [route_diagram] ── (invalid syntax AND repair_attempts < MAX) ───────┘
            │
            ├── (valid syntax) ──────────────────────────┐
            │                                            │
            └── (invalid syntax AND repairs exhausted) ──┤ (Marks as invalid; generates review note)
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ assemble_document │ ◄── Deterministic Markdown
                                               └─────────┬─────────┘     Assembler & SVG embedder
                                                         │
                                                         ▼
                                                        END
```

---

## 3. The 3-Phase Implementation Strategy

To build this advanced system reliably, we decouple the conversational human-in-the-loop complexity from the multi-agent generative pipeline. We construct and evaluate the subagent across **three distinct phases**:

| Phase | Milestone Name | Focus / Complexity Axis | Deliverable |
|---|---|---|---|
| **Phase 1** | **Standalone Clarification Subgraph** | Multi-turn HITL, `interrupt()` mechanics, delta updates, completeness check, turn guard, simulated user eval. | Compiled `clarification_graph` tested standalone against simulated developer personas. |
| **Phase 2** | **Standalone Generation Subgraph** | Hierarchical outline, parallel `Send()` workers, subprocess diagram compilation CLI, bounded repair loops, document assembly. | Compiled `generation_graph` tested standalone against 10 canned `RequirementsModel` fixtures. |
| **Phase 3** | **Merged Two-Phase Subagent & Supervisor Integration** | Parent graph orchestration, boundary contract, unified checkpointer inheritance, `SRSSubagentResponse` API. | Complete SRS Subagent ready to plug into the Phase 10 Supervisor alongside QA and DocGen. |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Standalone Clarification Subgraph (HITL Requirements Gathering)    │
│ • Implement RequirementsModel & CompletenessCheck schemas                   │
│ • Build analyze_initial_requirements & check_completeness                   │
│ • Build zero-LLM ask_question node using LangGraph interrupt()              │
│ • Implement flat-cost update_requirements delta-merging                     │
│ • Add code-level max_turns guard with unresolved_gaps propagation           │
│ • Validate with a Simulated Engineer Persona (eval harness)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Verified Requirements Extraction
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Standalone Generation Subgraph (Parallel Drafting & Diagrams)      │
│ • Implement DocOutline, SectionDraft, and DiagramSpec schemas                │
│ • Build generate_outline with thin-input resiliency                         │
│ • Dispatch parallel generate_section workers via LangGraph Send()           │
│ • Build determine_diagrams and parallel generate_diagram workers            │
│ • Implement validate_diagram with subprocess CLI (mmdc & plantuml)          │
│ • Implement bounded compiler-error repair loop (max 3 attempts)             │
│ • Build deterministic assemble_document with fallback notice injection      │
│ • Validate with canned RequirementsModel fixtures (complete and thin)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Verified Document & Diagram Pipeline
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Merged Two-Phase Subagent & Top-Level Integration                  │
│ • Build Parent SRS Graph linking run_clarification & run_generation        │
│ • Enforce strict boundary passing: {RequirementsModel, unresolved_gaps}     │
│ • Configure root Checkpointer (SqliteSaver/PostgresSaver) for child graphs  │
│ • Implement SRSSubagentResponse interface contract                          │
│ • Wire into the Top-Level Supervisor alongside QA and DocGen Agents         │
│ • Comprehensive end-to-end regression & stress testing                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep Component-by-Component Architectural Breakdown

Every single node, conditional router, schema, and edge in Architecture B has a strictly bounded responsibility. Below is the exhaustive specification for each component.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPONENT DIRECTORY INDEX                                 │
├──────────────────────────────────┬─────────────────────────────────────────────────────┤
│ 4.1. Schemas & Models            │ RequirementsModel, CompletenessCheck, DocOutline,   │
│                                  │ SectionDraft, DiagramSpec, ValidatedDiagram         │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ 4.2. Clarification Subgraph      │ • analyze_initial_requirements                      │
│                                  │ • check_completeness                                │
│                                  │ • route_after_completeness [Conditional Edge]       │
│                                  │ • ask_question (Interrupt Node)                     │
│                                  │ • update_requirements                               │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ 4.3. Generation Subgraph         │ • generate_outline                                  │
│                                  │ • fan_out_sections [Conditional Send Edge]          │
│                                  │ • generate_section (Parallel Worker)                │
│                                  │ • determine_diagrams                                │
│                                  │ • fan_out_diagrams [Conditional Send Edge]           │
│                                  │ • generate_diagram (Parallel Worker)                │
│                                  │ • validate_diagram (Subprocess CLI)                 │
│                                  │ • route_diagram_repair [Conditional Edge]           │
│                                  │ • assemble_document                                 │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ 4.4. Parent Graph Components     │ • run_clarification                                 │
│                                  │ • route_parent_transition [Conditional Edge]        │
│                                  │ • run_generation                                    │
│                                  │ • Root Checkpointer & HITL Bubbling                 │
└──────────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

### 4.1. Core Data Models & Typed State Schemas

#### 1. `RequirementsModel` (The Structured Requirements Representation)
```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str = Field(description="Name of the user role or external system")
    description: str = Field(description="Role responsibilities and access permissions")

class FunctionalRequirement(BaseModel):
    req_id: str = Field(description="Identifier e.g. FR-001")
    title: str = Field(description="Short title")
    description: str = Field(description="Detailed requirement description")
    priority: str = Field(description="Must Have | Should Have | Could Have")
    acceptance_criteria: list[str] = Field(default_factory=list)

class NonFunctionalRequirement(BaseModel):
    category: str = Field(description="Performance | Security | Scalability | Reliability | Usability")
    description: str = Field(description="Specific quantifiable constraint")
    metric: str | None = Field(default=None, description="Measurable target e.g. p99 < 200ms")

class RequirementsModel(BaseModel):
    project_title: str = Field(default="Software System")
    project_scope: str = Field(description="High-level product vision and operational scope")
    target_users_and_actors: list[Actor] = Field(default_factory=list)
    functional_requirements: list[FunctionalRequirement] = Field(default_factory=list)
    non_functional_requirements: list[NonFunctionalRequirement] = Field(default_factory=list)
    system_constraints: list[str] = Field(default_factory=list, description="Tech stack, regulatory, or environment rules")
    assumptions_and_dependencies: list[str] = Field(default_factory=list)
```

#### 2. `CompletenessCheck` (Structured Output Schema)
```python
class CompletenessCheck(BaseModel):
    reasoning: str = Field(description="Step-by-step audit of current requirements against standard IEEE 830 categories")
    missing_areas: list[str] = Field(
        description="Explicit list of ambiguous or missing dimensions (e.g., 'Authentication mechanism unspecified', 'Export format undefined'). MUST be populated before deciding is_complete."
    )
    is_complete: bool = Field(description="True ONLY if no critical functional or architectural ambiguities remain")
    next_question: str | None = Field(
        default=None, 
        description="A targeted, clear question focusing on the highest-priority missing area. None if is_complete is True."
    )
```

#### 3. `ClarificationState` (Clarification Subgraph State)
```python
import operator
from typing import Annotated, TypedDict

class ClarificationState(TypedDict):
    user_prompt: str
    project_context: str | None
    conversation_log: Annotated[list[dict], operator.add]
    turn_count: int
    max_turns: int
    requirements: RequirementsModel
    completeness: CompletenessCheck | None
    unresolved_gaps: list[str]
```

#### 4. Document Generation Models
```python
class OutlineSection(BaseModel):
    section_id: str = Field(description="Hierarchical section ID e.g. '1.0', '2.1'")
    title: str = Field(description="Section heading title")
    purpose_and_guidance: str = Field(description="Instructions for the generative worker drafting this section")
    subsections: list["OutlineSection"] = Field(default_factory=list)

class DocOutline(BaseModel):
    document_title: str
    target_audience: str
    sections: list[OutlineSection]

class SectionDraft(BaseModel):
    section_id: str
    title: str
    content_markdown: str

class DiagramSpec(BaseModel):
    diagram_id: str = Field(description="Unique diagram identifier e.g. 'diag-001'")
    title: str = Field(description="Diagram title")
    diagram_type: str = Field(description="architecture | sequence | erd | state | flowchart")
    syntax_format: str = Field(description="mermaid | plantuml")
    source_code: str = Field(description="Raw Mermaid or PlantUML code string")
    target_section_id: str = Field(description="Which outline section this diagram belongs to")
    repair_attempts: int = Field(default=0)
    last_error: str | None = Field(default=None)

class ValidatedDiagram(BaseModel):
    diagram_id: str
    title: str
    syntax_format: str
    source_code: str
    target_section_id: str
    is_valid: bool
    svg_file_path: str | None = Field(default=None)
    rendered_svg_content: str | None = Field(default=None)
    repair_attempts: int
    review_warning: str | None = Field(default=None)
```

#### 5. `GenerationState` (Generation Subgraph State)
```python
class GenerationState(TypedDict):
    requirements: RequirementsModel
    unresolved_gaps: list[str]
    outline: DocOutline | None
    section_drafts: Annotated[list[SectionDraft], operator.add]
    diagram_specs: list[DiagramSpec]
    validated_diagrams: Annotated[list[ValidatedDiagram], operator.add]
    final_document: str | None
    diagram_manifest: list[dict]
```

---

### 4.2. Phase 1: Clarification Subgraph Components

#### Node 1: `analyze_initial_requirements`
- **What it does:**
  Performs the initial requirements bootstrap. It takes the developer's raw opening request (e.g. *"Build an automated payment and invoice reconciliation service for our SaaS"*) along with optional project context (e.g. existing repo architecture or database specs), and extracts the first structured `RequirementsModel`. It seeds default values for unknown fields and initializes the conversation log.
- **Input State:**
  ```python
  class InitialInput(TypedDict):
      user_prompt: str
      project_context: str | None
      max_turns: int
  ```
- **Output State Update:**
  ```python
  {
      "requirements": seeded_requirements_model,
      "conversation_log": [{"role": "user", "content": state["user_prompt"]}],
      "turn_count": 0,
      "unresolved_gaps": []
  }
  ```
- **Implementation Detail:**
  Uses `.with_structured_output(RequirementsModel)`. Instructs the model: *"Extract all explicit and strongly implied requirements. Do not invent details; leave ambiguous areas unpopulated so the completeness auditor can probe them."*

---

#### Node 2: `check_completeness`
- **What it does:**
  Audits the current `RequirementsModel` against the IEEE 830 standard requirements completeness criteria. 
  **Crucial Architectural Safeguard:** The prompt strictly forces the LLM to enumerate `missing_areas` *before* outputting the boolean `is_complete`. LLMs asked a simple boolean question suffer high false-positive rates (declaring requirements "complete" prematurely). Forcing structured itemization of missing dimensions dramatically enhances clarification quality.
  Furthermore, the `next_question` is generated directly inside this node and stored in `state["completeness"]`.
- **Input State:**
  Reads `state["requirements"]`, `state["turn_count"]`, `state["max_turns"]`.
- **Output State Update:**
  ```python
  {
      "completeness": completeness_check_instance  # CompletenessCheck
  }
  ```
- **Why `next_question` lives here, NOT in `ask_question`:**
  In LangGraph, when execution resumes from an `interrupt()`, the node containing the interrupt re-executes from its very first line. If question generation sat inside `ask_question`, resuming from an interrupt would needlessly re-invoke the LLM that generated the question, burning API tokens and introducing non-deterministic latency. Placing question generation in `check_completeness` guarantees that `ask_question` is a zero-LLM, instantaneous operation upon resume.

---

#### Conditional Edge: `route_after_completeness`
- **What it does:**
  A deterministic, code-level routing guard. It governs the lifecycle of the clarification loop:
  1. If `is_complete == True`: routes to `END` (successful clarification).
  2. If `turn_count >= max_turns`: routes to `END` (forced completion due to turn exhaustion). Before exiting, it copies `completeness.missing_areas` directly into `unresolved_gaps`.
  3. If not complete and turns remain: routes to `ask_question`.
- **Logic:**
  ```python
  def route_after_completeness(state: ClarificationState) -> str:
      completeness = state.get("completeness")
      turn_count = state.get("turn_count", 0)
      max_turns = state.get("max_turns", 5)
      
      if completeness and completeness.is_complete:
          return END
      
      if turn_count >= max_turns:
          # Turn cap reached; carry missing areas forward to unresolved_gaps
          if completeness and completeness.missing_areas:
              state["unresolved_gaps"] = list(completeness.missing_areas)
          return END
          
      return "ask_question"
  ```

---

#### Node 3: `ask_question` (Human-in-the-Loop Interrupt Node)
- **What it does:**
  Pauses the execution graph and yields control back to the caller using LangGraph's `interrupt()` primitive. It presents the `next_question` to the human developer. When the human responds via `Command(resume=answer)`, the node receives the response and records it.
- **Input State:**
  Reads `state["completeness"].next_question`.
- **Output State Update:**
  ```python
  def ask_question(state: ClarificationState) -> dict:
      question = state["completeness"].next_question
      # Execution halts here until external resume
      human_answer = interrupt(question)
      
      return {
          "conversation_log": [{"role": "human", "content": human_answer}],
          "turn_count": state["turn_count"] + 1
      }
  ```
- **Key Invariant:** Zero LLM calls occur in this node. It is purely an I/O synchronization point.

---

#### Node 4: `update_requirements` (Delta-Update Node)
- **What it does:**
  Takes the existing `RequirementsModel` and merges *only the latest human answer* (`state["conversation_log"][-1]`). It does **not** re-parse the full conversation history.
- **Input State:**
  `state["requirements"]` and `latest_answer = state["conversation_log"][-1]`.
- **Output State Update:**
  ```python
  {
      "requirements": updated_requirements_model
  }
  ```
- **Token Efficiency Rationale:**
  Passing the entire conversation history on every turn results in $O(N^2)$ token growth. Passing only `(current_model, latest_answer)` keeps the context size virtually constant across all turns ($O(1)$ token cost per turn).

---

### 4.3. Phase 2: Generation Subgraph Components

#### Node 1: `generate_outline`
- **What it does:**
  Transforms the incoming `RequirementsModel` and `unresolved_gaps` into a comprehensive, hierarchical IEEE 830-compliant document outline (`DocOutline`).
  **Resiliency Invariant:** Treats thin requirements + non-empty `unresolved_gaps` as a completely normal, expected input (e.g. when clarification was forced to terminate by the turn cap). It generates appropriate placeholder sections and explicit gap-analysis sections rather than failing or crashing.
- **Input State:**
  `GenerationState` (`requirements`, `unresolved_gaps`).
- **Output State Update:**
  ```python
  {
      "outline": doc_outline_instance  # DocOutline
  }
  ```

---

#### Conditional Edge: `fan_out_sections` (LangGraph `Send` Fan-Out)
- **What it does:**
  Inspects `state["outline"].sections`. For every leaf section, it dispatches an independent, parallel execution of `generate_section` via LangGraph's `Send()` API.
- **Implementation:**
  ```python
  from langgraph.types import Send

  def fan_out_sections(state: GenerationState) -> list[Send]:
      sends = []
      for sec in state["outline"].sections:
          sends.append(Send("generate_section", {
              "section_id": sec.section_id,
              "title": sec.title,
              "purpose": sec.purpose_and_guidance,
              "requirements": state["requirements"],
              "unresolved_gaps": state["unresolved_gaps"],
              "outline_summary": [s.title for s in state["outline"].sections]
          }))
      return sends
  ```

---

#### Node 2: `generate_section` (Parallel Generative Worker)
- **What it does:**
  Drafts the complete, detailed Markdown content for a single assigned section. 
  - Each worker receives a **frozen snapshot** of the incoming `RequirementsModel` and `unresolved_gaps`.
  - Each worker receives the list of all section titles in the document for cross-referencing purposes.
  - **Isolation Boundary:** No worker can mutate shared state or observe drafts being concurrently generated by other workers.
- **Input State:**
  Worker-specific payload dispatched by `Send()`.
- **Output State Update:**
  ```python
  {
      "section_drafts": [SectionDraft(
          section_id=worker_input["section_id"],
          title=worker_input["title"],
          content_markdown=drafted_markdown
      )]
  }
  ```
  *(Aggregated into `state["section_drafts"]` using `operator.add` reducer).*

---

#### Node 3: `determine_diagrams`
- **What it does:**
  Acts as the visual architectural planner. Once all section drafts have been aggregated, this node scans the `RequirementsModel` and drafted sections to decide which diagrams must be included in the SRS. Standard types include:
  - System Context / High-Level Architecture (C4 or Component diagram)
  - Core Business Workflow (Sequence diagram)
  - Domain Data Model (Entity-Relationship diagram or Class diagram)
  - State Transitions (State diagram for lifecycles like order/auth states)
- **Input State:**
  `GenerationState` (`requirements`, `section_drafts`).
- **Output State Update:**
  ```python
  {
      "diagram_specs": list_of_diagram_specs  # list[DiagramSpec]
  }
  ```

---

#### Conditional Edge: `fan_out_diagrams` (LangGraph `Send` Fan-Out)
- **What it does:**
  Iterates over `state["diagram_specs"]` and dispatches each diagram to `generate_diagram` in parallel using `Send("generate_diagram", spec)`.
- **Implementation:**
  ```python
  def fan_out_diagrams(state: GenerationState) -> list[Send]:
      return [Send("generate_diagram", {"diagram_spec": spec}) for spec in state["diagram_specs"]]
  ```

---

#### Node 4: `generate_diagram` (Diagram Code Synthesizer & Repair Node)
- **What it does:**
  Generates clean, syntactically strict Mermaid or PlantUML source code for an assigned `DiagramSpec`.
  - **First Pass:** Prompts the LLM with the diagram goal, type, and system entities.
  - **Repair Pass:** If `diagram_spec.last_error` is populated, the node enters **repair mode**. It provides the LLM with the failing code and the exact compiler error output, instructing it to fix syntax errors (e.g. unescaped brackets, invalid node shapes, illegal PlantUML skinparams).
- **Input State:**
  Worker payload containing `DiagramSpec`.
- **Output State Update:**
  Returns the updated `DiagramSpec` with generated `source_code`.

---

#### Node 5: `validate_diagram` (Subprocess Compiler Execution)
- **What it does:**
  Validates the diagram code by executing real command-line compilers in an isolated subprocess.
  - **Mermaid:** Writes code to a temporary `.mmd` file and executes:
    `mmdc -i temp.mmd -o temp.svg`
  - **PlantUML:** Writes code to a temporary `.puml` file and executes:
    `plantuml -tsvg temp.puml`
  - Inspects the exit code and stderr output.
  - If successful, reads the generated SVG and creates a `ValidatedDiagram(is_valid=True, rendered_svg_content=...)`.
  - If failed, records the compiler error string, increments `repair_attempts += 1`, and creates `ValidatedDiagram(is_valid=False, last_error=stderr)`.
- **Input State:**
  `DiagramSpec`.
- **Output State Update:**
  `ValidatedDiagram` instance.

---

#### Conditional Edge: `route_diagram_repair`
- **What it does:**
  Evaluates the result of `validate_diagram`:
  1. If `is_valid == True`: Settled. Diagram proceeds to document assembly.
  2. If `is_valid == False` and `repair_attempts < MAX_REPAIRS` (e.g. 3): Routes back to `generate_diagram` for a targeted repair cycle.
  3. If `is_valid == False` and `repair_attempts >= MAX_REPAIRS`: Hard cap reached. The diagram is permanently marked invalid with `review_warning="Compiler syntax error could not be resolved automatically"`. It proceeds to assembly so that the document is never blocked.
- **Logic:**
  ```python
  def route_diagram_repair(diag: ValidatedDiagram) -> str:
      if diag.is_valid:
          return "assemble_document"
      if diag.repair_attempts < 3:
          return "generate_diagram"
      return "assemble_document"
  ```

---

#### Node 6: `assemble_document` (Deterministic Document Assembler)
- **What it does:**
  Takes the ordered `DocOutline`, all completed `section_drafts`, and all `validated_diagrams`, and deterministically compiles the final Software Requirements Specification in Markdown format.
  - **Deterministic Stitching:** Strictly non-agentic assembly (no LLM call) prevents hallucinated text alterations during stitching.
  - **Diagram Embedding:**
    - For valid diagrams: Injects the SVG file reference or inline collapsible diagram block with raw source.
    - For invalid diagrams: Injects an explicit callout block:
      ```markdown
      > ⚠️ **Diagram Placeholder [Architecture Overview]:** Diagram generation encountered syntax compilation errors after 3 repair attempts. Manual review required. Raw source retained in Appendix.
      ```
  - **Unresolved Gaps Appendix:** If `unresolved_gaps` is non-empty, appends Section `Appendix X: Unresolved Requirements & Ambiguity Register` listing all items that require developer clarification before system implementation.
  - Generates Table of Contents, metadata header, and traceability cross-references.
- **Input State:**
  `GenerationState`.
- **Output State Update:**
  ```python
  {
      "final_document": assembled_srs_markdown,
      "diagram_manifest": [
          {
              "diagram_id": d.diagram_id,
              "title": d.title,
              "status": "rendered" if d.is_valid else "failed_placeholder",
              "path": d.svg_file_path
          }
          for d in state["validated_diagrams"]
      ]
  }
  ```

---

### 4.4. Parent Graph Components & Boundaries

#### `ParentSRSState`
```python
class ParentSRSState(TypedDict):
    request_id: str
    thread_id: str
    user_prompt: str
    project_context: str | None
    max_turns: int
    requirements: RequirementsModel | None
    unresolved_gaps: list[str]
    final_document: str | None
    diagram_manifest: list[dict]
    status: str  # waiting_human_input | processing | completed | failed
    pending_question: str | None
    error: str | None
```

#### Node 1: `run_clarification`
- **What it does:**
  Invokes the compiled `clarification_graph`. 
  - Passes `{user_prompt, project_context, max_turns}`.
  - Propagates LangGraph interrupts directly to the parent runner.
  - Upon completion of `clarification_graph`, extracts `requirements` and `unresolved_gaps` and sets them in `ParentSRSState`.
- **Boundary Invariant:**
  Does **not** leak `conversation_log`, `turn_count`, or internal `completeness` reasoning into the parent state.

#### Conditional Edge: `route_parent_transition`
- Inspects whether `clarification_graph` finished cleanly.
- If requirements extraction succeeded (even with `unresolved_gaps`), routes to `run_generation`.
- If an unhandled exception occurred, routes to `END` with `status="failed"`.

#### Node 2: `run_generation`
- **What it does:**
  Invokes the compiled `generation_graph`.
  - Passes `{requirements, unresolved_gaps}`.
  - Receives back `{final_document, diagram_manifest}`.
  - Updates `ParentSRSState["final_document"]`, `ParentSRSState["diagram_manifest"]`, and marks `status="completed"`.
- **Boundary Invariant:**
  `generation_graph` never imports or accesses `interrupt()`, `conversation_log`, or turn counters.

#### Root Checkpointer Inheritance Mechanics
- **The Golden Rule:** The persistent checkpointer (`SqliteSaver` or `PostgresSaver`) is attached **strictly and exclusively to the parent graph** during parent compilation:
  ```python
  # CORRECT: Child graphs are compiled WITHOUT their own checkpointer
  compiled_clarification = clarification_builder.compile()
  compiled_generation = generation_builder.compile()

  # Attached strictly at parent level:
  parent_graph = parent_builder.compile(checkpointer=sqlite_checkpointer)
  ```
- **Why this is critical:** When child subgraphs are compiled without an explicit checkpointer, LangGraph automatically propagates the parent's checkpointer down to child nodes. Any `interrupt()` raised inside `clarification_graph` transparently pauses the parent graph and persists state under `ParentSRSState.thread_id`. Compiling a child subgraph with its own checkpointer creates separate, isolated persistence sessions and causes silent state loss on resume.

---

## 5. Detailed Breakdown of the 3 Implementation Phases

---

### Phase 1: Standalone Clarification Subgraph

#### Objective
Build, test, and evaluate the interactive requirements-gathering loop in complete isolation from document generation.

#### Components Implemented:
1. `src/srs/schemas/requirements.py`: `RequirementsModel`, `Actor`, `FunctionalRequirement`, `NonFunctionalRequirement`.
2. `src/srs/schemas/clarification.py`: `CompletenessCheck`, `ClarificationState`.
3. `src/srs/clarification/nodes.py`:
   - `analyze_initial_requirements`
   - `check_completeness`
   - `ask_question` (using `interrupt()`)
   - `update_requirements` (delta merging)
4. `src/srs/clarification/graph.py`: Assembles and compiles `clarification_graph`.

#### Testing & Evaluation Harness: The Simulated Developer Persona
Because manual human testing of conversational agents is slow and unrepeatable, Phase 1 includes an automated evaluation harness using a **Simulated Developer Persona**:
- A secondary LLM agent is provisioned with a secret "Ground Truth Requirements Brief" (e.g. a hidden specification for an IoT telemetry platform with MQTT ingestion, Redis caching, TimescaleDB storage, and JWT auth).
- The Simulated Developer responds to the agent's questions according to its brief.
- We evaluate:
  1. **Turn Efficiency:** Does the clarification agent saturate requirements within 3–5 turns?
  2. **Non-Redundancy:** Does the agent avoid re-asking questions already answered?
  3. **Recall of Ground Truth:** How many hidden requirements from the persona brief were successfully captured in the resulting `RequirementsModel`?
  4. **Turn Cap Enforcement:** When `max_turns` is reached, does the graph cleanly exit and copy remaining missing areas into `unresolved_gaps`?

#### Definition of Done (Phase 1):
- `clarification_graph` executes standalone with zero runtime crashes.
- Resumes seamlessly via `Command(resume="...")` without re-running question generation.
- Achieves $\ge 85\%$ recall on a golden dataset of 10 simulated developer briefs.
- Average token cost per turn remains flat between turn 1 and turn 5.

---

### Phase 2: Standalone Generation Subgraph

#### Objective
Build, test, and evaluate the parallel drafting, diagram compilation, and document assembly pipeline using pre-canned `RequirementsModel` fixtures.

#### Components Implemented:
1. `src/srs/schemas/generation.py`: `DocOutline`, `SectionDraft`, `DiagramSpec`, `ValidatedDiagram`, `GenerationState`.
2. `src/srs/generation/nodes/outline.py`: `generate_outline`.
3. `src/srs/generation/nodes/section_worker.py`: `generate_section` (dispatched via `Send`).
4. `src/srs/generation/nodes/diagram_planner.py`: `determine_diagrams`.
5. `src/srs/generation/nodes/diagram_worker.py`: `generate_diagram` (supports initial pass & compiler error repair).
6. `src/srs/generation/nodes/diagram_validator.py`: `validate_diagram` with subprocess CLI execution (`mmdc` and `plantuml`).
7. `src/srs/generation/nodes/assembler.py`: `assemble_document`.
8. `src/srs/generation/graph.py`: Assembles and compiles `generation_graph`.

#### Local Subprocess Tooling Setup:
- Install `@mermaid-js/mermaid-cli` (`npm install -g @mermaid-js/mermaid-cli`) for `mmdc`.
- Provide `plantuml.jar` and Graphviz `dot` executable for PlantUML rendering.
- Encapsulate execution inside `SubprocessCompilerRunner` with a 15-second execution timeout to prevent zombie processes.

#### Testing & Evaluation Harness:
Create a suite of 10 diverse, pre-canned `RequirementsModel` fixtures:
- 5 Complete specifications (e-commerce checkout, OAuth2 provider, distributed task queue, multi-tenant CRM, video streaming transcoding pipeline).
- 3 "Thin" specifications with intentional gaps (e.g. only 1 functional requirement and 4 items in `unresolved_gaps`).
- 2 Malformed/edge-case specifications (syntax-sensitive entity names with special characters).

#### Metrics Evaluated:
- **Parallel Fan-out:** Validated execution of all section workers concurrently.
- **Diagram Compilation Pass Rate:**
  - Zero-shot pass rate (percentage of diagrams compiling on first try).
  - Post-repair pass rate (percentage of diagrams compiling after $\le 2$ repair cycles, target $\ge 95\%$).
- **Assembly Robustness:** In thin specifications, verifies that unresolved gaps are clearly rendered in the final document without pipeline failure.

#### Definition of Done (Phase 2):
- All 10 fixtures generate complete, coherent Markdown documents.
- Diagrams compile to valid SVGs or gracefully degrade to clear review notices without hanging.
- Clean separation: `generation_graph` has zero imports of `interrupt` or conversational logging.

---

### Phase 3: Merged Two-Phase Subagent & Supervisor Integration

#### Objective
Compose the two independent subgraphs under the Parent SRS Graph, connect persistent SQLite/Postgres checkpointing, and expose the typed `SRSSubagentResponse` interface for the Top-Level Multi-Agent Supervisor.

#### Components Implemented:
1. `src/srs/parent/state.py`: `ParentSRSState`.
2. `src/srs/parent/graph.py`: Parent graph definition linking `run_clarification` and `run_generation`.
3. `src/srs/parent/checkpointer.py`: Root persistence configuration (`SqliteSaver`).
4. `src/srs/interface.py`: Subagent wrapper exposing `run(request_id, thread_id, ...)` and `resume(thread_id, answer)`.
5. `src/srs/supervisor_node.py`: Subgraph-as-node wrapper for the Phase 10 Top-Level Supervisor.

#### Definition of Done (Phase 3):
- Complete end-to-end execution: Initial prompt $\to$ multi-turn clarification loop $\to$ document generation $\to$ final Markdown SRS.
- Interrupted states safely persist across process restarts.
- Passes full integration tests alongside the QA Agent and DocGen Subgraph Agent.

---

## 6. Supervisor-to-Subagent Interface Contract

The Top-Level Supervisor (Phase 10 of `Sprinter_Implementation_Plan.md`) interacts with the SRS Subagent through a clean, typed Pydantic contract. The Supervisor never inspects raw LangGraph internal states or child subgraph nodes.

### Interface Schema
```python
from typing import Literal, Optional
from pydantic import BaseModel

class SRSSubagentResponse(BaseModel):
    status: Literal["waiting_human_input", "processing", "completed", "failed"]
    thread_id: str
    pending_question: Optional[str] = None
    final_document: Optional[str] = None
    diagram_manifest: Optional[list[dict]] = None
    unresolved_gaps: Optional[list[str]] = None
    error: Optional[str] = None
```

### Runtime Status Resolution Logic
When the Supervisor invokes or resumes the SRS Subagent, it derives the status by checking `graph.get_state(config)`:

```python
def invoke_srs_subagent(
    graph,
    thread_id: str,
    user_prompt: str | None = None,
    resume_answer: str | None = None
) -> SRSSubagentResponse:
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        if resume_answer is not None:
            # Resuming from a human-in-the-loop interrupt
            from langgraph.types import Command
            graph.invoke(Command(resume=resume_answer), config=config)
        else:
            # Initial invocation
            graph.invoke({
                "user_prompt": user_prompt,
                "max_turns": 5,
                "project_context": None
            }, config=config)
            
        # Inspect state after execution
        state = graph.get_state(config)
        
        # If 'next' is non-empty, the graph is paused at an interrupt()
        if state.next:
            pending_question = None
            if "completeness" in state.values and state.values["completeness"]:
                pending_question = state.values["completeness"].next_question
                
            return SRSSubagentResponse(
                status="waiting_human_input",
                thread_id=thread_id,
                pending_question=pending_question
            )
            
        # If 'next' is empty and final_document exists, execution completed
        if state.values.get("final_document"):
            return SRSSubagentResponse(
                status="completed",
                thread_id=thread_id,
                final_document=state.values["final_document"],
                diagram_manifest=state.values.get("diagram_manifest", []),
                unresolved_gaps=state.values.get("unresolved_gaps", [])
            )
            
        return SRSSubagentResponse(status="processing", thread_id=thread_id)
        
    except Exception as exc:
        return SRSSubagentResponse(
            status="failed",
            thread_id=thread_id,
            error=str(exc)
        )
```

---

## 7. Connecting with Phase 10 Supervisor (Multi-Agent Unified Architecture)

In Phase 10 of `Sprinter_Implementation_Plan.md`, all three specialized agents are united under the hierarchical Supervisor:

```
                               ┌──────────────────────────┐
                               │       Supervisor /       │
               User Query ───► │      Intent Router       │
                               └────────────┬─────────────┘
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
           ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
           │     QA Agent     │   │  DocGen Subgraph │   │   SRS Subagent   │
           │   (Feature 2)    │   │   (Feature 1)    │   │   (Feature 3)    │
           │  Fast RAG/Search │   │  Architecture D  │   │  Architecture B  │
           └──────────────────┘   └──────────────────┘   └──────────────────┘
```

When user intent is classified as `srs`, the Supervisor dispatches the session to the compiled SRS Subagent with a unique `thread_id`. If `status == "waiting_human_input"`, the Supervisor yields the `pending_question` back to the user interface, pausing the thread until the developer replies.

---

## 8. Directory Structure for Implementation

```
d:\Projects\DevDocs AI\
├── src\
│   ├── srs\
│   │   ├── __init__.py
│   │   ├── interface.py                # SRSSubagentResponse & high-level runner
│   │   ├── supervisor_node.py          # Wrapper for Phase 10 Top-Level Supervisor
│   │   ├── schemas\
│   │   │   ├── __init__.py
│   │   │   ├── requirements.py         # RequirementsModel, FunctionalRequirement, Actor
│   │   │   ├── clarification.py        # CompletenessCheck, ClarificationState
│   │   │   └── generation.py           # DocOutline, SectionDraft, DiagramSpec, ValidatedDiagram
│   │   ├── clarification\
│   │   │   ├── __init__.py
│   │   │   ├── nodes.py                # analyze, check_completeness, ask_question, update
│   │   │   ├── edges.py                # route_after_completeness logic
│   │   │   └── graph.py                # clarification_graph definition
│   │   ├── generation\
│   │   │   ├── __init__.py
│   │   │   ├── nodes\
│   │   │   │   ├── __init__.py
│   │   │   │   ├── outline.py          # generate_outline node
│   │   │   │   ├── section_worker.py   # parallel generate_section worker
│   │   │   │   ├── diagram_planner.py  # determine_diagrams node
│   │   │   │   ├── diagram_worker.py   # generate_diagram worker & repair prompt
│   │   │   │   ├── diagram_validator.py# validate_diagram subprocess runner
│   │   │   │   └── assembler.py        # assemble_document deterministic builder
│   │   │   ├── edges.py                # fan_out_sections, fan_out_diagrams, route_repair
│   │   │   └── graph.py                # generation_graph definition
│   │   ├── parent\
│   │   │   ├── __init__.py
│   │   │   ├── state.py                # ParentSRSState
│   │   │   ├── nodes.py                # run_clarification & run_generation wrappers
│   │   │   ├── edges.py                # route_parent_transition
│   │   │   └── graph.py                # Parent composed StateGraph
│   │   └── utils\
│   │       ├── __init__.py
│   │       ├── subprocess_runner.py    # Safe CLI runner for mmdc & plantuml
│   │       └── simulated_user.py       # Simulated developer persona for Phase 1 evals
├── tests\
│   ├── test_srs_phase1_clarification.py# Test standalone clarification & HITL
│   ├── test_srs_phase2_generation.py   # Test standalone generation & diagram validation
│   └── test_srs_phase3_composition.py  # Test full Parent graph & checkpointing
├── fixtures\
│   └── srs_requirements_fixtures.py    # 10 canned RequirementsModel test fixtures
├── DocGen_Subgraph_Implementation_Plan.md
├── Sprinter_Implementation_Plan.md
└── SRS_Subgraph_Implementation_Plan.md
```

---

## 9. Failure Modes, Edge Cases, and Mitigations Matrix

| Failure Mode / Edge Case | Risk Level | Root Cause | Architecture B Mitigation Strategy |
|---|---|---|---|
| **Interrupt Replay Token Explosion** | Critical | LangGraph resumes execution by re-running the node calling `interrupt()`. | **Node Splitting:** Question generation lives in `check_completeness`. The `ask_question` node only invokes `interrupt()`. On resume, zero LLM calls occur. |
| **Quadratic Token Cost Growth** | High | Conversational chat history re-sent on every turn. | **Delta Model Updating:** `update_requirements` merges only the current `RequirementsModel` + latest human reply. Token cost per turn remains strictly flat ($O(1)$). |
| **Premature Clarification Exit** | High | LLM answering a boolean "is this complete" tends to say Yes prematurely. | **Forced Enumeration:** `CompletenessCheck` forces the model to itemize `missing_areas` in structured JSON before deciding `is_complete`. |
| **Infinite Clarification Loop** | High | User answers ambiguously; LLM continually asks further questions. | **Code-Level Max Turns Guard:** `route_after_completeness` forces an exit if `turn_count >= max_turns`, copying missing areas into `unresolved_gaps`. |
| **Diagram Syntax Compilation Failure** | Critical | LLMs produce invalid Mermaid/PlantUML syntax with broken brackets or links. | **Subprocess CLI Validation + Repair Loop:** Real `mmdc` and `plantuml` CLI validation. Exact compiler errors are fed back into repair prompts. Bounded at 3 attempts; falls back to an explicit review callout instead of crashing. |
| **Thin Input Downstream Crash** | Medium | Generation assumes all sections have rich requirements data. | **Resilient Outline Planning:** `generate_outline` treats thin requirements + unresolved gaps as a standard case, drafting appropriate exploratory and gap-register sections. |
| **State Loss on Resume** | Critical | Compiling child subgraphs with their own checkpointers breaks parent persistence. | **Single Root Checkpointer:** Child subgraphs are compiled without checkpointers. They inherit the parent's `SqliteSaver` via thread_id scoping. |
| **Subprocess Execution Hang** | Medium | `mmdc` or `plantuml` process hangs indefinitely on invalid inputs. | **Hard Subprocess Timeout:** `subprocess.run(..., timeout=15)` forcefully kills stalled CLI rendering jobs and treats them as compiler failures. |

---

## 10. Summary: Execution Roadmap for the Engineer

1. **Step 1:** Create `src/srs/schemas/` models (`RequirementsModel`, `CompletenessCheck`, `DocOutline`, `DiagramSpec`, etc.).
2. **Step 2 (Phase 1):** Build `src/srs/clarification/` graph. Verify with `test_srs_phase1_clarification.py` using the Simulated Developer persona. Ensure `interrupt()` resumes without re-generating questions.
3. **Step 3 (Phase 2):** Build `src/srs/generation/` graph. Set up local `mmdc` and `plantuml` subprocess runners. Run `test_srs_phase2_generation.py` against the 10 canned `RequirementsModel` fixtures. Verify bounded diagram repair and clean assembly.
4. **Step 4 (Phase 3):** Build `src/srs/parent/` graph. Compile parent with `SqliteSaver`. Verify checkpoint inheritance and export `SRSSubagentResponse`. Run full integration tests.
5. **Step 5 (Phase 10 of Sprinter):** Mount the compiled SRS Subagent into the top-level Supervisor alongside the QA Agent and DocGen Subagent.
