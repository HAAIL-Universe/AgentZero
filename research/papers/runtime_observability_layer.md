---
topic: Runtime Observability Layer
status: implemented
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# Runtime Observability Layer

## Problem Statement

Agent Zero has no structured observability. All diagnostic output goes to `print()` on stdout.
There are no metrics for DB query latency, agent execution time, end-to-end turn duration,
consolidation timing, error rates, or model inference performance. The only timing data
collected is tool execution duration (agent_zero_server.py:329).

OpenTelemetry's AI Agent Observability initiative (2025) established that "by establishing
standardized conventions, AI agent frameworks can report standardized metrics, traces, and
logs, making it easier to integrate observability solutions and compare performance across
frameworks." Agent Zero currently has none of this.

The "wide events" paradigm (Honeycomb, Greptime 2025) proposes replacing scattered
metrics/logs/traces with high-dimensional structured events as the single source of truth --
an approach that naturally fits agent scenarios where "a typical agent execution event
contains dozens or even hundreds of fields."

## Current State in Agent Zero

### What IS Collected
| Metric | Location | How |
|--------|----------|-----|
| Tool duration | agent_zero_server.py:329 | `time.monotonic()` diff, stored in result dict |
| Tool activity | agent_zero_server.py:394-420 | `_emit_tool_activity_summary()` WebSocket event |
| Consolidation event count | agent_zero_server.py:1879-1885 | Rule/episode count delta |
| Reasoning stages | Via `append_stage_message()` calls | Stage name + payload to DB |
| Agent outputs | `_emit_reasoning_step()` callback | Thought + confidence to WebSocket |
| Runtime events | `_emit_runtime_event()` (354-378) | Stage, event type, meta dict |

### What IS NOT Collected
1. **Database query latency** -- `await fetch_one()` has no timing
2. **Agent execution time** -- no `_duration_ms` on agent outputs
3. **End-to-end turn latency** -- no root timestamp on `_run_conversation_turn`
4. **Error rates** -- exceptions logged to stdout only, not structured
5. **Memory consolidation timing** -- `run_consolidation()` timing unknown
6. **Episode capture timing** -- `append_episode()` duration not tracked
7. **Intervention logging latency** -- `log_intervention()` timing unknown
8. **Shadow save latency** -- `save_shadow()` timing unknown
9. **Guardrail evaluation time** -- `evaluate_turn_guardrails()` unmetered
10. **Retrieval latency** -- `build_retrieval_packet()` timing unknown
11. **Model inference token usage** -- not tracked per-turn
12. **Connection pool utilization** -- not monitored

### Logging Pattern
All logging is bare `print()`:
```python
print(f"[Agent Zero] Tool execution error: {error_type}: {exc}")
print(f"[Agent Zero] Consolidation: {len(new_rules)} new rules")
```
No structured format, no severity levels, no correlation IDs.

## Industry Standard / Research Findings

### 1. OpenTelemetry GenAI Semantic Conventions (v1.37+, 2025)
OpenTelemetry has stabilized semantic conventions for generative AI systems covering
three signal types:

**Metrics** (standard names):
- `gen_ai.client.token.usage` (Histogram) -- input/output tokens per operation
- `gen_ai.client.operation.duration` (Histogram, seconds) -- end-to-end latency
- `gen_ai.server.request.duration` (Histogram) -- model server latency
- `gen_ai.server.time_per_output_token` (Histogram) -- decode performance
- `gen_ai.server.time_to_first_token` (Histogram) -- prefill + queue latency

**Span Attributes** (for agent traces):
- `gen_ai.operation.name` -- operation type
- `gen_ai.agent.id`, `gen_ai.agent.name` -- agent identification
- `gen_ai.request.model` -- model name
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` -- token counts
- `gen_ai.response.finish_reasons` -- completion status

**Agent-Specific Spans**:
- `create_agent {name}` -- agent instantiation
- `invoke_agent {name}` -- agent execution with full lifecycle tracking

Citation: OpenTelemetry, "Semantic conventions for generative AI systems," 2025.
https://opentelemetry.io/docs/specs/semconv/gen-ai/

Citation: OpenTelemetry, "AI Agent Observability - Evolving Standards and Best Practices," 2025.
https://opentelemetry.io/blog/2025/ai-agent-observability/

### 2. Wide Events Paradigm (Honeycomb, Greptime 2025)
The observability community is converging on "wide events" -- single structured events
with hundreds of fields -- as the replacement for separate metrics/logs/traces.

Key insight from Greptime (2025): "Agents generate massive amounts of semi-structured data.
A typical agent execution event contains dozens or even hundreds of fields. Each tool call
can return a completely different structure." Wide events handle this naturally by writing
high-dimensional raw events and deriving metrics on demand.

For Agent Zero, this means a single `TurnEvent` dict that captures everything about a turn
(timing, agent outputs, tool calls, errors, outcomes) and can be queried for any metric.

Citation: Greptime, "Agent Observability: Can the Old Playbook Handle the New Game?" 2025.
https://www.greptime.com/blogs/2025-12-11-agent-observability

Citation: Honeycomb, "Structured Events Are the Basis of Observability."
https://www.honeycomb.io/blog/structured-events-basis-observability

### 3. Observability 2.0 Principles (Majors, 2025)
Charity Majors (Honeycomb) distinguishes Observability 1.0 (three separate pillars with
many sources of truth) from Observability 2.0 (one source of truth with wide structured
events from which you derive all other data types).

For a system like Agent Zero that doesn't yet have ANY observability infrastructure,
starting with wide events is cheaper and more powerful than building separate
metrics/logs/traces pipelines.

Citation: Majors, "Observability 2.0," charity.wtf, 2025. https://charity.wtf/tag/observability-2-0/

### 4. Datadog GenAI Observability (2025)
Datadog's LLM Observability natively supports OTel GenAI semantic conventions, showing
industry adoption of the standard. Key tracked dimensions:
- Token usage by model and operation
- Latency distributions (P50/P95/P99)
- Error rates by error type
- Agent execution traces with tool call breakdown

Citation: Datadog, "LLM Observability natively supports OpenTelemetry GenAI Semantic Conventions," 2025.
https://www.datadoghq.com/blog/llm-otel-semantic-convention/

## Proposed Implementation

### File: `agent_zero/observability.py` (NEW -- ~150 lines)

The design follows the wide-event pattern: a single `TurnEvent` accumulates all timing
and metadata during a turn, then is emitted as a structured JSON line at turn end.

```python
"""Lightweight observability layer using wide events."""

import time
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("agent_zero.obs")

@dataclass
class TurnEvent:
    """Wide event accumulator for a single conversation turn."""
    turn_id: str = ""
    user_id: str = ""
    session_id: str = ""
    started_at: float = 0.0

    # Timing (all in milliseconds)
    total_duration_ms: float = 0.0
    model_inference_ms: float = 0.0
    db_total_ms: float = 0.0
    agent_durations_ms: dict = field(default_factory=dict)  # {agent_id: ms}
    tool_durations_ms: dict = field(default_factory=dict)   # {tool_name: ms}
    retrieval_ms: float = 0.0
    consolidation_ms: float = 0.0
    guardrail_ms: float = 0.0

    # Counts
    agent_count: int = 0
    tool_call_count: int = 0
    db_query_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    # Errors
    errors: list = field(default_factory=list)  # [{type, message, operation, category}]
    error_count: int = 0

    # Outcomes
    intervention_logged: bool = False
    episode_captured: bool = False
    consolidation_triggered: bool = False
    consolidation_rules_added: int = 0

    def record_error(self, exc: Exception, operation: str, category: str = "unknown"):
        self.errors.append({
            "type": type(exc).__name__,
            "message": str(exc)[:200],
            "operation": operation,
            "category": category,
        })
        self.error_count += 1

    def finalize(self):
        self.total_duration_ms = (time.monotonic() - self.started_at) * 1000
        return self

    def to_dict(self) -> dict:
        """Serialize to flat dict for JSON output."""
        d = {
            "turn_id": self.turn_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "total_ms": round(self.total_duration_ms, 1),
            "model_ms": round(self.model_inference_ms, 1),
            "db_ms": round(self.db_total_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "consolidation_ms": round(self.consolidation_ms, 1),
            "guardrail_ms": round(self.guardrail_ms, 1),
            "agents": self.agent_durations_ms,
            "tools": self.tool_durations_ms,
            "agent_count": self.agent_count,
            "tool_calls": self.tool_call_count,
            "db_queries": self.db_query_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error_count": self.error_count,
            "errors": self.errors if self.errors else None,
            "intervention": self.intervention_logged,
            "episode": self.episode_captured,
            "consolidation": self.consolidation_triggered,
            "rules_added": self.consolidation_rules_added,
        }
        return {k: v for k, v in d.items() if v is not None}


# Context variable for current turn event
_current_event: TurnEvent | None = None

def current_event() -> TurnEvent | None:
    return _current_event

@asynccontextmanager
async def observe_turn(turn_id: str, user_id: str, session_id: str):
    """Context manager that creates, populates, and emits a turn event."""
    global _current_event
    event = TurnEvent(
        turn_id=turn_id, user_id=user_id, session_id=session_id,
        started_at=time.monotonic()
    )
    _current_event = event
    try:
        yield event
    finally:
        event.finalize()
        _current_event = None
        _emit(event)

def _emit(event: TurnEvent):
    """Emit turn event as structured JSON log line."""
    logger.info(json.dumps(event.to_dict(), default=str))


@asynccontextmanager
async def timed(event: TurnEvent, attr: str, key: str = None):
    """Time an async block and record in the turn event."""
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        if key and isinstance(getattr(event, attr, None), dict):
            getattr(event, attr)[key] = round(elapsed_ms, 1)
        else:
            setattr(event, attr, round(elapsed_ms, 1))


def record_db_query(event: TurnEvent, duration_ms: float):
    """Accumulate DB query timing."""
    event.db_total_ms += duration_ms
    event.db_query_count += 1
```

### Integration Points in agent_zero_server.py

#### 1. Wrap `_run_conversation_turn()` with `observe_turn()`
```python
# At the top of _run_conversation_turn():
async with observe_turn(turn_id, user_id, session_id) as event:
    # ... existing turn logic ...
```

#### 2. Time agent execution (cognitive_runtime.py)
```python
# In _execute_agent_async():
from agent_zero.observability import current_event, timed

event = current_event()
if event:
    async with timed(event, "agent_durations_ms", key=agent_id):
        output = await loop.run_in_executor(...)
    event.agent_count += 1
```

#### 3. Time model inference
```python
# Around get_inference_async() call:
if event:
    async with timed(event, "model_inference_ms"):
        inference = await get_inference_async()
```

#### 4. Time DB operations
```python
# In database.py fetch_one/fetch_all/execute wrappers:
t0 = time.monotonic()
result = await pool.fetch(...)
duration_ms = (time.monotonic() - t0) * 1000
event = current_event()
if event:
    record_db_query(event, duration_ms)
```

#### 5. Record errors (replace print-based logging)
```python
# BEFORE:
except Exception as e:
    print(f"[Agent Zero] Error: {e}")

# AFTER:
except Exception as e:
    event = current_event()
    if event:
        event.record_error(e, "append_stage_message", classify_error(e).value)
    logger.warning("stage_message_failed", exc_info=True)
```

#### 6. Record token usage
```python
# After model response:
if event and hasattr(response, 'usage'):
    event.input_tokens = response.usage.prompt_tokens
    event.output_tokens = response.usage.completion_tokens
```

### Structured Logging Setup

Replace all `print()` calls with Python `logging`:
```python
# In agent_zero_server.py startup:
import logging
logging.basicConfig(
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    level=logging.INFO,
)
```

This gives structured JSON logs compatible with any log aggregator (CloudWatch, Grafana Loki,
etc.) without adding external dependencies.

### WebSocket Metrics Endpoint (optional)

Expose a `/metrics` endpoint that returns aggregated metrics from recent turn events:
```python
@app.get("/metrics")
async def metrics():
    return {
        "turns_processed": _turn_count,
        "avg_turn_ms": _avg_turn_ms,
        "avg_model_ms": _avg_model_ms,
        "error_rate": _error_rate,
        "circuit_breaker_states": {
            name: cb.state for name, cb in _circuits.items()
        },
    }
```

## Test Specifications

### test_observability.py (~14 tests)

```python
# TurnEvent tests
def test_turn_event_records_timing():
    """TurnEvent records total_duration_ms on finalize()."""

def test_turn_event_records_agent_durations():
    """Agent durations stored as {agent_id: ms} dict."""

def test_turn_event_records_errors():
    """record_error() appends structured error with type, message, operation."""

def test_turn_event_to_dict_omits_none():
    """to_dict() excludes None values for compact JSON."""

def test_turn_event_accumulates_db_queries():
    """record_db_query() accumulates total and increments count."""

# observe_turn context manager tests
@pytest.mark.asyncio
async def test_observe_turn_sets_current_event():
    """current_event() returns event inside observe_turn block."""

@pytest.mark.asyncio
async def test_observe_turn_clears_on_exit():
    """current_event() returns None after observe_turn exits."""

@pytest.mark.asyncio
async def test_observe_turn_emits_on_exit():
    """Turn event is emitted as JSON log on context exit."""

@pytest.mark.asyncio
async def test_observe_turn_emits_on_exception():
    """Turn event is emitted even when exception occurs."""

# timed context manager tests
@pytest.mark.asyncio
async def test_timed_records_scalar():
    """timed() sets scalar attribute on event."""

@pytest.mark.asyncio
async def test_timed_records_dict_key():
    """timed() sets key in dict attribute on event."""

# Integration tests
def test_structured_log_format():
    """Emitted log line is valid JSON with required fields."""

def test_error_recording_with_classification():
    """record_error includes error category from classify_error."""

def test_token_usage_tracked():
    """input_tokens and output_tokens populated from model response."""
```

## Estimated Impact

- **Visibility**: Every turn produces a structured event with 20+ dimensions of timing,
  counting, and error data -- replacing scattered `print()` calls
- **Performance debugging**: Agent execution times, DB latency, model inference duration
  all measurable for the first time
- **Cost tracking**: Token usage per turn enables cost-per-conversation estimates
- **Error diagnosis**: Structured error classification replaces bare exception messages
- **Production readiness**: JSON structured logging compatible with any log aggregator
- **Standards alignment**: Follows OpenTelemetry GenAI semantic conventions for metric names
  and span attributes, enabling future OTel integration
- **Zero external dependencies**: Pure Python stdlib (logging, json, dataclasses, time)

## Citations

1. OpenTelemetry, "Semantic conventions for generative AI systems," 2025. https://opentelemetry.io/docs/specs/semconv/gen-ai/
2. OpenTelemetry, "Semantic conventions for generative AI metrics," 2025. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
3. OpenTelemetry, "AI Agent Observability - Evolving Standards and Best Practices," 2025. https://opentelemetry.io/blog/2025/ai-agent-observability/
4. OpenTelemetry, "Semantic Conventions for GenAI agent and framework spans," 2025. https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
5. Greptime, "Agent Observability: Can the Old Playbook Handle the New Game?" 2025. https://www.greptime.com/blogs/2025-12-11-agent-observability
6. Honeycomb, "Structured Events Are the Basis of Observability." https://www.honeycomb.io/blog/structured-events-basis-observability
7. Datadog, "LLM Observability natively supports OpenTelemetry GenAI Semantic Conventions," 2025. https://www.datadoghq.com/blog/llm-otel-semantic-convention/
8. Majors, "Observability 2.0," charity.wtf. https://charity.wtf/tag/observability-2-0/
