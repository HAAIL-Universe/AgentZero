---
topic: Conversation Turn Decomposition
status: ready_for_implementation
priority: high
estimated_complexity: large
researched_at: 2026-03-18T20:30:00Z
---

# Conversation Turn Decomposition

## Problem Statement

`agent_zero_server.py:_run_conversation_turn()` is a 1,464-line async method spanning lines
1258-2721. It handles 23 distinct phases from user input to response delivery, contains
200+ await calls, 230+ WebSocket transmissions, 15+ nested try/except blocks, and 7 major
conditional branches. This is the single largest maintainability liability in the Agent Zero
codebase -- any change risks unintended side effects, the method is untestable in isolation,
and understanding any single phase requires reading all 1,464 lines for context.

## Current State in Agent Zero

**File:** `agent_zero/agent_zero_server.py`, lines 1258-2721

**Method signature:** 10 parameters, returns `str`

**23 Identified Phases (with line ranges):**

| # | Phase | Lines | Duration | Key Responsibility |
|---|-------|-------|----------|-------------------|
| 1 | User Input & Persistence | 1258-1293 | 36 | Name extraction, DB persist, session append |
| 2 | Routing & Status | 1295-1308 | 14 | Route turn, emit status |
| 3 | Intervention Resolution | 1310-1340 | 31 | Resolve pending interventions |
| 4 | Retrieval Packet | 1342-1384 | 43 | Build retrieval, hydrate from DB |
| 5 | Tool Selection & Guardrails | 1388-1416 | 29 | Select tools, evaluate guardrails |
| 6 | Self-Introspection Notes | 1418-1433 | 16 | Log self-introspection route notes |
| 7 | Reasoning Run Init | 1435-1448 | 14 | Create reasoning run DB record |
| 8 | Reasoning Run Logging | 1450-1563 | 114 | Log all pre-gen decisions to reasoning run |
| 9 | Guardrail Early Exits | 1565-1639 | 75 | Handle clarify/refuse/escalate (early return) |
| 10 | Tool Execution | 1641-1699 | 59 | Execute pre-selected tools |
| 11 | Cognitive Pipeline | 1701-2127 | 427 | Full strategic reasoning (11 sub-phases) |
| 12 | System Prompt Building | 2134-2156 | 23 | Build prompt, compress context |
| 13 | Model Loading | 2158-2189 | 32 | Load model if needed |
| 14 | Fallback Echo Mode | 2191-2228 | 38 | Echo response if model fails (early return) |
| 15 | Response Gen Init | 2230-2243 | 14 | Initialize streaming state |
| 16 | Native Tool Calling Loop | 2245-2352 | 108 | vLLM function calling rounds |
| 17 | Streaming Generation | 2354-2458 | 105 | Stream tokens to user |
| 18 | Question Generation | 2460-2480 | 21 | Append curiosity question |
| 19 | Response Sanitization | 2482-2519 | 38 | Strip tags, output guardrails |
| 20 | Quality Gates | 2520-2604 | 85 | Speaker quality, stage classification |
| 21 | Final Emission | 2606-2630 | 25 | Send done, persist message |
| 22 | Post-Response Logging | 2632-2715 | 84 | Memory notes, intervention logging |
| 23 | Cleanup | 2717-2720 | 4 | Extract update requests, return |

**Shared mutable state crossing phase boundaries:**
- `session_messages` (list, mutated in-place across phases 1, 21)
- `shadow` (dict, read/written across phases 4, 11, 18, 22)
- `tool_results` (list, built incrementally in phases 10, 16)
- `generation_context` (list, mutated in phases 15, 16, 17)
- `full_response` (str, accumulated in phase 17, modified in 18, 19)
- `reasoning_packet` (dict, set in phase 11, used in 12, 20, 22)
- `scenario_packet` (dict, set in phase 11, used in 18, 20)

**Critical path:** route -> retrieval -> guardrails -> (cognitive pipeline) -> system prompt
-> streaming -> sanitization -> logging

## Industry Standard / Research Findings

### Split Phase (Fowler, 2018)

Martin Fowler's *Refactoring: Improving the Design of Existing Code* (2nd ed., 2018)
defines the **Split Phase** refactoring: when code deals with two different concerns in
sequence, split into phases connected by an intermediate data structure. Each phase has
clear inputs and outputs.

Source: [Catalog of Refactorings](https://refactoring.com/catalog/)

Fowler specifically notes that refactorings were described for single-process software and
that concurrent/async code would benefit from adapted patterns.

Source: [Changes for the 2nd Edition of Refactoring](https://martinfowler.com/articles/refactoring-2nd-changes.html)

### Extract Method Visitor (Johal, 2026)

The Extract Method Visitor pattern combines Extract Method with the Visitor pattern for
automated code decomposition. A case study on 300-line ETL functions showed:
- Cyclomatic complexity: 22 -> 8 (64% reduction)
- Build time: 35% improvement
- Test coverage: 22% increase

Source: [Refactor Guru Patterns: Python Extract Method Visitor](https://johal.in/refactor-guru-patterns-python-extract-method-visitor-2026/)

### Async Pipeline Pattern

Modern Python async frameworks (FastAPI, Starlette) use the **middleware pipeline** pattern
where each processing stage is an independent async callable receiving a context object and
optionally delegating to the next stage. This enables:
- Independent testing of each stage
- Conditional stage execution via route-based dispatch
- Error isolation per stage

Source: [FastAPI for Microservices: High-Performance Python API Design Patterns](https://talent500.com/blog/fastapi-microservices-python-api-design-patterns-2025/)

### Data Pipeline Design Patterns

Start Data Engineering (2025) documents the **functional pipeline** pattern for data
processing: each stage is a pure function that receives an immutable context and returns
a new context with its contributions added. This prevents shared mutable state bugs.

Source: [Data Pipeline Design Patterns](https://www.startdataengineering.com/post/code-patterns/)

### Python Async Pipeline Library

The `async-pipeline` PyPI package (2024) provides a framework for building async processing
pipelines with typed stages, error handling, and retry logic per stage.

Source: [async-pipeline on PyPI](https://pypi.org/project/async-pipeline/)

### The "Class Too Large" Refactoring (Fowler, 2022)

Fowler's article on oversized classes recommends identifying **natural clusters** of
functionality, extracting them into collaborator objects, and having the original class
delegate to them. The key insight: the original class becomes an **orchestrator** that
coordinates smaller, focused units.

Source: [Refactoring: This class is too large](https://martinfowler.com/articles/class-too-large.html)

## Proposed Implementation

### Architecture: TurnContext + Phase Functions

Replace the monolithic method with a **TurnContext** dataclass and **phase functions**.

**Design principles:**
1. Each phase is an independent async function taking `TurnContext` and returning `TurnContext`
2. `TurnContext` is a dataclass holding all state that flows between phases (replaces shared locals)
3. The orchestrator (`_run_conversation_turn`) becomes a thin pipeline that calls phases in sequence
4. Early returns become sentinel values in `TurnContext` (e.g., `ctx.early_response`)
5. Phase 11 (cognitive pipeline) gets its own sub-pipeline in a separate module

### Step 1: Define TurnContext

Create `agent_zero/turn_context.py`:

```python
"""Context object flowing through conversation turn phases."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from fastapi import WebSocket


@dataclass
class TurnContext:
    """Immutable-ish context flowing through turn phases."""
    # --- Inputs (set once, never modified) ---
    websocket: WebSocket
    user_content: str
    inference: object
    session_messages: list
    message_index: int
    user: Optional[dict] = None
    session_id: Optional[str] = None
    shadow: Optional[dict] = None
    user_metadata: Optional[dict] = None
    assistant_metadata: Optional[dict] = None
    extra_system_prompt: Optional[str] = None

    # --- Phase outputs (set by phases, read by later phases) ---
    route_result: dict = field(default_factory=dict)
    is_casual: bool = False
    resolved_interventions: list = field(default_factory=list)
    intervention_summary: dict = field(default_factory=dict)
    user_model: dict = field(default_factory=dict)
    retrieval_packet: dict = field(default_factory=dict)
    memory_notes: list = field(default_factory=list)
    tool_specs: list = field(default_factory=list)
    guardrail_decision: Optional[dict] = None
    filtered_tool_calls: list = field(default_factory=list)
    context_bundle: dict = field(default_factory=dict)
    reasoning_run_id: Optional[str] = None
    reasoning_packet: dict = field(default_factory=dict)
    scenario_packet: dict = field(default_factory=dict)
    tool_results: list = field(default_factory=list)
    system_prompt: str = ""
    context: list = field(default_factory=list)
    full_response: str = ""
    final_response: str = ""

    # --- Control flow ---
    early_response: Optional[str] = None  # Set to short-circuit pipeline
```

### Step 2: Extract Phase Functions

Create `agent_zero/turn_phases.py` with one async function per phase cluster:

```python
"""Conversation turn phase functions.

Each function takes a TurnContext and modifies it in place.
If a phase sets ctx.early_response, the pipeline short-circuits.
"""

async def phase_persist_input(ctx: TurnContext) -> None:
    """Phase 1: Persist user message, extract name preference."""
    ...  # Lines 1272-1293

async def phase_route(ctx: TurnContext) -> None:
    """Phase 2: Route the turn and send thinking status."""
    ...  # Lines 1295-1308

async def phase_resolve_interventions(ctx: TurnContext) -> None:
    """Phase 3: Resolve pending interventions (skip on casual)."""
    ...  # Lines 1310-1340

async def phase_build_retrieval(ctx: TurnContext) -> None:
    """Phase 4: Build and hydrate retrieval packet."""
    ...  # Lines 1342-1386

async def phase_select_tools_and_guardrails(ctx: TurnContext) -> None:
    """Phase 5: Select tools, evaluate guardrails, build context bundle."""
    ...  # Lines 1388-1416

async def phase_reasoning_run_init(ctx: TurnContext) -> None:
    """Phase 7: Create reasoning run DB record."""
    ...  # Lines 1435-1448

async def phase_log_pre_generation(ctx: TurnContext) -> None:
    """Phase 8: Log routing/retrieval/guardrail decisions to reasoning run."""
    ...  # Lines 1450-1563

async def phase_guardrail_early_exit(ctx: TurnContext) -> None:
    """Phase 9: Handle guardrail clarify/refuse/escalate. Sets early_response."""
    ...  # Lines 1565-1639

async def phase_execute_tools(ctx: TurnContext) -> None:
    """Phase 10: Execute pre-selected tools."""
    ...  # Lines 1641-1699

async def phase_cognitive_pipeline(ctx: TurnContext) -> None:
    """Phase 11: Full strategic reasoning cognitive pipeline."""
    ...  # Lines 1701-2127 (delegate to turn_cognitive.py)

async def phase_build_system_prompt(ctx: TurnContext) -> None:
    """Phase 12: Build system prompt with context compression."""
    ...  # Lines 2134-2156

async def phase_load_model(ctx: TurnContext) -> None:
    """Phase 13-14: Load model, handle fallback echo mode."""
    ...  # Lines 2158-2228 (sets early_response if model fails)

async def phase_generate_response(ctx: TurnContext) -> None:
    """Phase 15-17: Initialize, tool calling loop, streaming generation."""
    ...  # Lines 2230-2458

async def phase_post_generation(ctx: TurnContext) -> None:
    """Phase 18-19: Question generation, response sanitization."""
    ...  # Lines 2460-2519

async def phase_quality_gates(ctx: TurnContext) -> None:
    """Phase 20: Speaker quality, stage classification, capability checks."""
    ...  # Lines 2520-2604

async def phase_emit_and_persist(ctx: TurnContext) -> None:
    """Phase 21: Send final done, persist assistant message."""
    ...  # Lines 2606-2630

async def phase_post_response_logging(ctx: TurnContext) -> None:
    """Phase 22-23: Memory notes, intervention logging, cleanup."""
    ...  # Lines 2632-2720
```

### Step 3: Extract Cognitive Pipeline

Create `agent_zero/turn_cognitive.py` for Phase 11's 427 lines (the largest single phase):

```python
"""Strategic reasoning cognitive pipeline.

Extracted from _run_conversation_turn Phase 11 (lines 1701-2127).
Only executes for route == "strategic_reasoning".
"""

async def run_strategic_reasoning(ctx: TurnContext) -> None:
    """Execute the full cognitive pipeline: extract -> plan -> reason -> decide."""
    await _extract_state(ctx)           # 11b
    await _generate_paths(ctx)           # 11c
    await _frame_scenario(ctx)           # 11d
    await _simulate_and_critique(ctx)    # 11e
    await _score_and_decide(ctx)         # 11f
    await _run_cognitive_agents(ctx)     # 11g
    await _capture_episode(ctx)          # 11h
    await _emit_hub_resolution(ctx)      # 11i
    await _persist_reasoning(ctx)        # 11j
```

### Step 4: Refactor Orchestrator

The original `_run_conversation_turn` becomes a thin pipeline:

```python
async def _run_conversation_turn(
    websocket, user_content, inference, session_messages, message_index,
    user=None, session_id=None, shadow=None,
    user_metadata=None, assistant_metadata=None, extra_system_prompt=None,
) -> str:
    """Process one user turn -- thin orchestrator delegating to phase functions."""
    ctx = TurnContext(
        websocket=websocket, user_content=user_content, inference=inference,
        session_messages=session_messages, message_index=message_index,
        user=user, session_id=session_id, shadow=shadow,
        user_metadata=user_metadata, assistant_metadata=assistant_metadata,
        extra_system_prompt=extra_system_prompt,
    )

    phases = [
        phase_persist_input,
        phase_route,
        phase_resolve_interventions,
        phase_build_retrieval,
        phase_select_tools_and_guardrails,
        phase_reasoning_run_init,
        phase_log_pre_generation,
        phase_guardrail_early_exit,
        phase_execute_tools,
        phase_cognitive_pipeline,
        phase_build_system_prompt,
        phase_load_model,
        phase_generate_response,
        phase_post_generation,
        phase_quality_gates,
        phase_emit_and_persist,
        phase_post_response_logging,
    ]

    for phase in phases:
        await phase(ctx)
        if ctx.early_response is not None:
            return ctx.early_response

    return ctx.final_response
```

### Migration Strategy

**CRITICAL: This must be done incrementally to maintain the zero-bug streak.**

1. **Phase A (low risk):** Create `turn_context.py` with `TurnContext` dataclass. No behavior change.
2. **Phase B (low risk):** Create `turn_phases.py` and `turn_cognitive.py` as empty files with stubs.
3. **Phase C (per-phase, 17 iterations):** Extract one phase at a time:
   - Copy the phase's code into the phase function
   - Replace local variable references with `ctx.` attribute access
   - Call the phase function from `_run_conversation_turn` in the exact same position
   - Run tests after each extraction
   - Commit after each green test run
4. **Phase D:** Once all phases are extracted, refactor `_run_conversation_turn` to the
   thin pipeline form
5. **Phase E:** Extract Phase 11 sub-phases into `turn_cognitive.py`

Each phase extraction is a single-phase refactoring that can be independently tested and
rolled back.

### Changes Summary

| File | Change |
|------|--------|
| `agent_zero/turn_context.py` | **NEW** -- TurnContext dataclass (~50 lines) |
| `agent_zero/turn_phases.py` | **NEW** -- 17 async phase functions (~900 lines total) |
| `agent_zero/turn_cognitive.py` | **NEW** -- 9 cognitive sub-phase functions (~400 lines) |
| `agent_zero/agent_zero_server.py` | `_run_conversation_turn` reduces from 1464 lines to ~40 lines |
| `agent_zero/test_turn_phases.py` | **NEW** -- Unit tests for individual phases (~300 lines) |

**Net effect:** ~1464 lines in one function -> ~40 line orchestrator + ~1300 lines across
17 focused functions in 2 modules. Total code volume stays similar but testability and
readability dramatically improve.

## Test Specifications

### Unit Tests for Phase Functions

```python
class TestPhasePersistInput(unittest.IsolatedAsyncioTestCase):
    async def test_appends_user_message_to_session(self):
        """User message should be appended to session_messages."""
        ctx = make_test_ctx(user_content="hello", session_messages=[])
        await phase_persist_input(ctx)
        assert len(ctx.session_messages) == 1
        assert ctx.session_messages[0]["role"] == "user"

    async def test_caps_session_messages_at_200(self):
        """Session messages should be capped at 200."""
        ctx = make_test_ctx(session_messages=[{"role": "user", "content": f"msg{i}"} for i in range(250)])
        await phase_persist_input(ctx)
        assert len(ctx.session_messages) <= 201  # 200 existing + 1 new


class TestPhaseRoute(unittest.IsolatedAsyncioTestCase):
    async def test_sets_is_casual_for_greeting(self):
        """Greeting messages should route as casual."""
        ctx = make_test_ctx(user_content="hi there")
        await phase_route(ctx)
        assert ctx.is_casual is True
        assert ctx.route_result["route"] == "casual_chat"


class TestPhaseGuardrailEarlyExit(unittest.IsolatedAsyncioTestCase):
    async def test_refuse_sets_early_response(self):
        """Refuse guardrail should set early_response."""
        ctx = make_test_ctx()
        ctx.guardrail_decision = {"action": "refuse", "message": "Cannot help with that."}
        await phase_guardrail_early_exit(ctx)
        assert ctx.early_response is not None
        assert "Cannot help" in ctx.early_response

    async def test_allow_does_not_set_early_response(self):
        """Allow guardrail should not set early_response."""
        ctx = make_test_ctx()
        ctx.guardrail_decision = {"action": "allow"}
        await phase_guardrail_early_exit(ctx)
        assert ctx.early_response is None


class TestPhasePostGeneration(unittest.IsolatedAsyncioTestCase):
    async def test_sanitizes_think_tags(self):
        """Response should have <think> tags stripped."""
        ctx = make_test_ctx()
        ctx.full_response = "<think>internal</think>Hello user!"
        await phase_post_generation(ctx)
        assert "<think>" not in ctx.final_response
        assert "Hello user!" in ctx.final_response


class TestPipelineOrchestration(unittest.IsolatedAsyncioTestCase):
    async def test_early_exit_stops_pipeline(self):
        """Pipeline should stop at first phase that sets early_response."""
        ctx = make_test_ctx()
        # Simulate guardrail refuse
        ctx.guardrail_decision = {"action": "refuse", "message": "Blocked."}
        # Run pipeline
        result = await _run_conversation_turn_pipeline(ctx)
        assert result == "Blocked."
        # Phases after guardrail should NOT have run
        assert ctx.full_response == ""  # response generation never ran

    async def test_full_pipeline_returns_final_response(self):
        """Full pipeline with no early exit should return final_response."""
        ctx = make_test_ctx()
        ctx.guardrail_decision = {"action": "allow"}
        result = await _run_conversation_turn_pipeline(ctx)
        assert isinstance(result, str)
        assert len(result) > 0
```

### Integration Tests

```python
class TestTurnContextDataFlow(unittest.IsolatedAsyncioTestCase):
    async def test_route_result_flows_to_retrieval(self):
        """Route result from phase 2 should be available in phase 4."""
        ctx = make_test_ctx(user_content="I want to improve my sleep habits")
        await phase_route(ctx)
        assert ctx.route_result["route"] != ""
        await phase_build_retrieval(ctx)
        assert ctx.retrieval_packet["route"] == ctx.route_result["route"]

    async def test_tool_results_flow_to_generation(self):
        """Tool results from phase 10 should be available in phase 15-17."""
        ctx = make_test_ctx()
        ctx.filtered_tool_calls = [{"name": "test_tool", "args": {}}]
        await phase_execute_tools(ctx)
        assert len(ctx.tool_results) > 0
```

### Regression Test

```python
class TestDecompositionRegression(unittest.IsolatedAsyncioTestCase):
    async def test_equivalent_output_casual(self):
        """Decomposed pipeline should produce identical output to monolithic for casual."""
        # Use a mock websocket, mock inference, known shadow
        # Compare: old _run_conversation_turn vs new pipeline
        # Output and websocket messages should match exactly

    async def test_equivalent_output_strategic(self):
        """Decomposed pipeline should produce identical output for strategic reasoning."""
        # Same as above but for strategic_reasoning route
```

## Estimated Impact

- **Testability:** Each phase can be unit-tested independently with mock TurnContext.
  Current state: zero phase-level tests (only end-to-end via WebSocket)
- **Readability:** Understanding a single phase requires reading ~30-100 lines instead of
  1,464 lines. New developer onboarding time for this code path reduced by ~80%
- **Maintainability:** Changes to one phase (e.g., adding a new guardrail action) only
  touch one function. No risk of accidentally breaking unrelated phases
- **Debugging:** Stack traces point to specific phase functions instead of a generic
  1,464-line method
- **Parallelization potential:** Phases 3 and 4 (interventions and retrieval) could be
  run concurrently with `asyncio.gather()` since they have no mutual dependencies --
  saving ~200-500ms per turn on DB-heavy requests
- **Cognitive pipeline isolation:** Phase 11 (427 lines) moves to its own module,
  allowing the cognitive reasoning architecture to evolve independently of HTTP handling
