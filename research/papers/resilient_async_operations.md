---
topic: Resilient Async Operations
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# Resilient Async Operations: Circuit Breakers, Retry, and Error Classification

## Problem Statement

Agent Zero's async operations (database calls, model inference, tool execution) have no resilience
layer. Failures are caught with broad `except Exception` blocks and either silently swallowed
or logged to stdout. There are no timeouts on most async operations, no retry logic for transient
failures, no circuit breakers to prevent cascading failure, and no structured error classification.

A systematic review of 26 microservice resilience studies (Falahah et al., 2025) found that
"naive backoff without jitter causes retry storms" (P99 = 2600ms, 17% error rate) while
"jittered retries with budgets" reduced P99 to 1400ms and errors to 6%. The same review
identified that "over-tight circuit-breaker thresholds reduce throughput" -- configuration
must be data-driven, not arbitrary.

## Current State in Agent Zero

### Silent Exception Swallowing (agent_zero_server.py)
Over 25 `except Exception` blocks catch and swallow errors:
- Line 347: `except Exception: pass` -- failed stage message logging
- Line 390: `except Exception: return str(data)[:max_chars]` -- JSON serialization
- Lines 1054-1057: intervention summary fetch failure -> empty default
- Lines 1358-1359: retrieval hydration failure -> fall back to empty
- Lines 1538-1576: reasoning stage persistence failures -> continue
- Line 2088: context compression failure -> `except Exception: pass`
- Lines 2606-2621: intervention/episode shadow update failures -> continue

### No Timeout Protection
- Database operations: `await fetch_one/fetch_all/execute()` -- no timeout wrapper
- Agent execution via executor: `cognitive_runtime.py:1234-1236` -- no timeout
- Model inference: `get_inference_async()` -- no timeout
- Tool execution: `agent_zero_server.py:322` -- no timeout
- Consolidation: `agent_zero_server.py:1875` -- synchronous, no timeout

### No Retry Logic
No transient failure retry exists anywhere. A single NeonDB connection timeout kills the
entire operation with no recovery attempt.

### No Error Classification
All exceptions flatten to a single `print()` line. No distinction between:
- Transient (network timeout, connection pool exhaustion) -- should retry
- Permanent (bad SQL, schema mismatch) -- should fail fast
- Resource (memory, thread pool exhaustion) -- should backpressure
- External (model provider down, API rate limit) -- should circuit break

## Industry Standard / Research Findings

### 1. Circuit Breaker Pattern (Nygard 2007, refined by Netflix Hystrix)
The standard three-state machine (closed -> open -> half-open) prevents cascading failure.
Falahah et al. (2025) systematic review of 412 papers identified circuit breaking as Theme T1,
noting that "configuration sprawl (inconsistent thresholds)" is the primary anti-pattern.

**Recommended parameters** (from the constraint-aware decision matrix):
- Failure threshold: 5 consecutive failures or >50% error rate in 60s window
- Open duration: 30s (with exponential increase on repeated trips)
- Half-open probe: 1 request, success resets to closed
- Per-dependency isolation (not global)

Citation: Falahah et al., "Resilient Microservices: A Systematic Review of Recovery Patterns,
Strategies, and Evaluation Frameworks," arXiv:2512.16959, Dec 2025.
URL: https://arxiv.org/abs/2512.16959

### 2. Retry with Jitter and Budgets
The same review found that combining bounded retries with circuit breakers achieved optimal
results: P99 = 1100ms, 3% error rate. Key parameters:
- Exponential backoff: `wait = min(base * 2^attempt, max_wait)`
- Full jitter: `wait = random(0, wait)` (AWS recommendation)
- Retry budget: max 3 attempts for DB, 2 for model inference, 1 for tools
- Budget cap: no more than 10% of total requests should be retries

The Tenacity library (Python) provides production-ready implementation with async support:
`@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=1, max=10))`

Citation: Tenacity documentation, https://tenacity.readthedocs.io/
Citation: AWS Architecture Blog, "Exponential Backoff and Jitter"

### 3. Error Classification Taxonomy
The systematic review (Table I) identifies 8 failure categories. For Agent Zero, the relevant ones:

| Category | Agent Zero Examples | Strategy |
|----------|-----------------|----------|
| Network/Timeout | NeonDB connection timeout, vLLM unreachable | Retry with jitter |
| Dependency Outage | NeonDB fully down, vLLM OOM | Circuit break |
| Resource Exhaustion | Connection pool exhausted, thread starvation | Backpressure (reject) |
| Config/Deploy | Bad migration, schema mismatch | Fail fast, alert |
| External Provider | vLLM rate limit, API throttle | Circuit break + queue |

### 4. Bulkhead Pattern
Falahah et al. found isolation reduces blast radius but costs ~15% utilization. For Agent Zero:
- Separate DB connection pools for critical (turn processing) vs non-critical (logging/metrics)
- Separate executor pools for agent execution vs tool execution

Citation: Falahah et al., arXiv:2512.16959, Theme T5 (Bulkheads)

### 5. Resilience Maturity Model
The review proposes 5 maturity levels (RML-1 to RML-5). Agent Zero is currently at RML-1
(manual restarts, minimal monitoring). Target: RML-3 (standardized circuit breakers,
bulkheads, DLQs) within one implementation cycle.

## Proposed Implementation

### File: `agent_zero/resilience.py` (NEW -- ~200 lines)

#### 1. Error Classifier
```python
class ErrorCategory(Enum):
    TRANSIENT = "transient"       # retry-safe
    PERMANENT = "permanent"       # fail fast
    RESOURCE = "resource"         # backpressure
    EXTERNAL = "external"         # circuit break
    UNKNOWN = "unknown"           # default to transient

def classify_error(exc: Exception) -> ErrorCategory:
    """Classify exception by retry strategy."""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, (asyncpg.TooManyConnectionsError,)):
        return ErrorCategory.RESOURCE
    if isinstance(exc, (asyncpg.PostgresSyntaxError, asyncpg.UndefinedTableError)):
        return ErrorCategory.PERMANENT
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return ErrorCategory.EXTERNAL
    return ErrorCategory.UNKNOWN
```

#### 2. Circuit Breaker (lightweight, no external dependency)
```python
class CircuitBreaker:
    """Per-dependency circuit breaker with three states."""
    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0, half_open_max: int = 1):
        self.name = name
        self.state = "closed"  # closed | open | half_open
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.half_open_max = half_open_max
        self.half_open_count = 0

    async def call(self, coro_fn, *args, **kwargs):
        if self.state == "open":
            if time.monotonic() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                self.half_open_count = 0
            else:
                raise CircuitOpenError(self.name)
        if self.state == "half_open" and self.half_open_count >= self.half_open_max:
            raise CircuitOpenError(self.name)
        try:
            result = await coro_fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self, exc):
        cat = classify_error(exc)
        if cat == ErrorCategory.PERMANENT:
            return  # don't count permanent errors
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.state == "half_open":
            self.state = "open"
            self.recovery_timeout = min(self.recovery_timeout * 2, 300)
        elif self.failure_count >= self.failure_threshold:
            self.state = "open"
```

#### 3. Resilient Wrapper
```python
async def resilient_call(coro_fn, *args,
                         circuit: CircuitBreaker = None,
                         max_retries: int = 3,
                         timeout_s: float = 10.0,
                         fallback=None,
                         operation_name: str = "unknown",
                         **kwargs):
    """Wrap an async call with timeout, retry, circuit breaker, and error classification."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            if circuit:
                result = await asyncio.wait_for(
                    circuit.call(coro_fn, *args, **kwargs),
                    timeout=timeout_s
                )
            else:
                result = await asyncio.wait_for(
                    coro_fn(*args, **kwargs),
                    timeout=timeout_s
                )
            return result
        except CircuitOpenError:
            if fallback is not None:
                return fallback() if callable(fallback) else fallback
            raise
        except Exception as exc:
            last_exc = exc
            cat = classify_error(exc)
            if cat == ErrorCategory.PERMANENT:
                raise  # no retry
            if cat == ErrorCategory.RESOURCE:
                raise  # backpressure, don't retry
            if attempt < max_retries:
                wait = min(1.0 * (2 ** attempt), 10.0)
                jitter = random.uniform(0, wait)
                await asyncio.sleep(jitter)
            else:
                if fallback is not None:
                    return fallback() if callable(fallback) else fallback
                raise
    raise last_exc
```

#### 4. Pre-configured Circuit Breakers (module-level singletons)
```python
# One circuit breaker per external dependency
db_circuit = CircuitBreaker("neondb", failure_threshold=5, recovery_timeout=30)
model_circuit = CircuitBreaker("vllm", failure_threshold=3, recovery_timeout=60)
```

### Integration Points in agent_zero_server.py

#### DB Operations (replace bare awaits)
```python
# BEFORE (line ~531):
result = await fetch_one("SELECT ...", user_id)

# AFTER:
result = await resilient_call(
    fetch_one, "SELECT ...", user_id,
    circuit=db_circuit, max_retries=2, timeout_s=5.0,
    operation_name="fetch_user"
)
```

#### Non-Critical DB Writes (use fallback instead of silent swallow)
```python
# BEFORE (line ~1538):
try:
    await append_stage_message(...)
except Exception:
    pass

# AFTER:
await resilient_call(
    append_stage_message, ...,
    circuit=db_circuit, max_retries=1, timeout_s=3.0,
    fallback=None,  # graceful skip
    operation_name="append_stage_message"
)
```

#### Agent Execution Timeout (cognitive_runtime.py:1234)
```python
# BEFORE:
output = await loop.run_in_executor(None, lambda: execute_cognitive_agent(...))

# AFTER:
output = await asyncio.wait_for(
    loop.run_in_executor(None, lambda: execute_cognitive_agent(...)),
    timeout=30.0  # 30s per agent
)
```

### Rollout Strategy
1. Add `resilience.py` with CircuitBreaker, classify_error, resilient_call
2. Wrap critical-path DB calls first (user fetch, shadow load/save)
3. Wrap model inference calls with model_circuit
4. Replace `except Exception: pass` blocks with resilient_call + fallback
5. Add agent execution timeout (30s per agent)
6. Add structured error logging (JSON to stdout, not bare print)

## Test Specifications

### test_resilience.py (~15 tests)

```python
# Circuit breaker tests
def test_circuit_breaker_stays_closed_on_success():
    """CB stays closed after successful calls."""

def test_circuit_breaker_opens_after_threshold():
    """CB opens after N consecutive transient failures."""

def test_circuit_breaker_rejects_when_open():
    """CB raises CircuitOpenError when open."""

def test_circuit_breaker_half_open_recovery():
    """CB transitions open -> half_open -> closed on success."""

def test_circuit_breaker_half_open_failure_reopens():
    """CB transitions half_open -> open on failure, doubles recovery_timeout."""

def test_circuit_breaker_ignores_permanent_errors():
    """Permanent errors don't increment failure count."""

# Error classification tests
def test_classify_timeout_as_transient():
    """asyncio.TimeoutError -> TRANSIENT."""

def test_classify_connection_error_as_transient():
    """ConnectionError -> TRANSIENT."""

def test_classify_syntax_error_as_permanent():
    """PostgresSyntaxError -> PERMANENT."""

def test_classify_pool_exhaustion_as_resource():
    """TooManyConnectionsError -> RESOURCE."""

# Resilient call tests
def test_resilient_call_retries_on_transient():
    """Retries up to max_retries on transient errors."""

def test_resilient_call_no_retry_on_permanent():
    """Fails immediately on permanent errors."""

def test_resilient_call_uses_fallback():
    """Returns fallback value when all retries exhausted."""

def test_resilient_call_timeout():
    """Raises TimeoutError when operation exceeds timeout_s."""

def test_resilient_call_with_circuit_breaker():
    """Circuit breaker integration: opens after threshold failures."""
```

## Estimated Impact

- **Reliability**: Transient DB/model failures recover automatically instead of silently failing
- **Visibility**: Structured error classification replaces 25+ `except Exception: pass` blocks
- **Cascading failure prevention**: Circuit breakers stop retrying downed dependencies
- **User experience**: Failed non-critical operations degrade gracefully instead of corrupting state
- **Maturity**: Moves Agent Zero from RML-1 to RML-3 on the Resilience Maturity Model

## Citations

1. Falahah et al., "Resilient Microservices: A Systematic Review of Recovery Patterns, Strategies, and Evaluation Frameworks," arXiv:2512.16959, Dec 2025. https://arxiv.org/abs/2512.16959
2. Tenacity: Retrying library for Python, https://tenacity.readthedocs.io/
3. Nygard, M., "Release It! Design and Deploy Production-Ready Software," 2007 (circuit breaker pattern origin)
4. AWS Architecture Blog, "Exponential Backoff and Jitter," https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
5. PyBreaker: Python Circuit Breaker, https://github.com/danielfm/pybreaker
6. IJRAI, "Circuit Breaker Pattern in Modern Distributed Systems: Implementation, Monitoring, and Best Practices," 2025. https://ijrai.org/index.php/ijrai/article/view/433
