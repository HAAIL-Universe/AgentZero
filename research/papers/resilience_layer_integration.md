---
topic: Resilience Layer Integration
status: ready_for_implementation
priority: medium
estimated_complexity: small
researched_at: 2026-03-18T22:30:00Z
---

# Resilience Layer Integration

## Problem Statement

The resilience layer (`agent_zero/resilience.py`) implements circuit breakers, retry with jitter, error classification, and timeout protection. Pre-configured circuit breakers exist for NeonDB (`db_circuit`) and vLLM (`model_circuit`). However, **none of these are actually wired into the codebase**. The 4 core database functions in `database.py` call asyncpg directly with no protection. Database failures cascade silently -- a NeonDB outage causes unhandled exceptions throughout the request pipeline.

## Current State in Agent Zero

### Resilience Layer (complete but orphaned)

**File:** `agent_zero/resilience.py` (186 lines)

- `ErrorCategory` enum: TRANSIENT, PERMANENT, RESOURCE, EXTERNAL, UNKNOWN (lines 14-19)
- `classify_error(exc)` -- classifies exceptions by retry strategy (lines 29-59)
- `CircuitBreaker` class -- 3-state (closed/open/half_open), exponential backoff on recovery (lines 62-125)
- `resilient_call()` -- wraps async call with timeout, retry, circuit breaker, and fallback (lines 128-180)
- `db_circuit = CircuitBreaker("neondb", failure_threshold=5, recovery_timeout=30.0)` (line 184)
- `model_circuit = CircuitBreaker("vllm", failure_threshold=3, recovery_timeout=60.0)` (line 185)

**Tests:** `test_resilience.py` (299 lines, 52 test cases) -- all passing but testing orphaned code.

### Database Layer (unprotected)

**File:** `agent_zero/database.py`

4 core functions, all following the same pattern:
```python
async def fetch_one(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)
```

- `fetch_one(query, *args)` -- line 287
- `fetch_all(query, *args)` -- line 294
- `execute(query, *args)` -- line 301
- `fetch_val(query, *args)` -- line 308

Higher-level helpers (`create_commitment`, `update_commitment_status`, etc.) call these 4 functions. Total: **30+ call sites** across `database.py` alone, plus direct calls from `agent_zero_server.py`.

### Integration Gap

- `database.py` imports `asyncpg` (line 15) but NOT `resilience`
- `agent_zero_server.py` does NOT import `resilience`
- `db_circuit` and `model_circuit` exist but are never used
- No fallback behavior when NeonDB is unreachable
- Connection pool timeout (30s, line 59) is the only protection

## Industry Standard / Research Findings

### 1. Session Wrapper Pattern (Akarshan, 2025)

The recommended pattern for integrating circuit breakers with async database layers is the **session wrapper approach**, where the circuit breaker wraps the underlying connection acquisition and query execution, not individual queries. This avoids modifying every call site. Protected operations (execute, fetch, commit) go through the breaker; in-memory operations (add, delete before flush) bypass it.

**URL:** https://dev.to/akarshan/building-resilient-database-operations-with-aiobreaker-async-sqlalchemy-fastapi-23dl

### 2. Tenacity + Circuit Breaker Composition (Roy, 2025)

Production Python services combine Tenacity's retry logic with circuit breakers: retries handle transient failures within a single dependency call, while circuit breakers prevent retries against confirmed-down dependencies. Key insight: **separate retry policy from circuit breaker** -- retries are per-call, breakers are per-dependency.

**URL:** https://www.amitavroy.com/articles/building-resilient-python-applications-with-tenacity-smart-retries-for-a-fail-proof-architecture

### 3. PyBreaker Global Instance Pattern (Nygard, 2007 / PyBreaker 2024)

Circuit breaker instances must be **global singletons** living across requests. Creating a new breaker per request loses failure count state. PyBreaker recommends `fail_max=5, reset_timeout=timedelta(seconds=60)` for database circuits. Agent Zero already follows this pattern with module-level `db_circuit` and `model_circuit`.

**URL:** https://pypi.org/project/pybreaker/

### 4. Falahah et al. (2025) Systematic Review of Microservice Resilience

Systematic review of 47 microservice resilience papers identifies circuit breakers, bulkheads, and retries as the three essential patterns. Key finding: **circuit breakers should be per-dependency, not per-endpoint**. Database, external API, and message queue each get their own breaker. This matches Agent Zero's existing `db_circuit` + `model_circuit` design.

**URL:** https://arxiv.org/abs/2512.16959

### 5. asyncpg Connection Pool as Partial Bulkhead (MagicStack, 2024)

asyncpg's pool with `min_size=2, max_size=10, command_timeout=30` (database.py:54-60) provides a partial bulkhead -- connection exhaustion doesn't crash the process. But it doesn't protect against NeonDB being completely unreachable (pool.acquire() will block/timeout, not fast-fail). Circuit breakers add the fast-fail capability.

**URL:** https://magicstack.github.io/asyncpg/current/usage.html

### 6. Resilient Circuit PostgreSQL Backend (2024)

The Resilient Circuit library provides PostgreSQL-backed circuit breakers for distributed systems, ensuring all instances see the same state. While Agent Zero is single-instance, the health check pattern (`SELECT 1`) is useful for detecting NeonDB recovery during half-open state.

**URL:** https://resilient-circuit.readthedocs.io/en/latest/

## Proposed Implementation

### Strategy: Transparent Wrapper at Database Core Functions

Wrap the 4 core database functions (`fetch_one`, `fetch_all`, `execute`, `fetch_val`) with `resilient_call()` from the existing resilience layer. This provides protection to all 30+ call sites without modifying any of them.

### Step 1: Add resilience import to database.py

**File:** `agent_zero/database.py`, line 15 (after `import asyncpg`):

```python
from resilience import resilient_call, db_circuit
```

### Step 2: Wrap core functions with resilient_call

**File:** `agent_zero/database.py`

Replace the 4 core functions (lines 287-312):

```python
async def _raw_fetch_one(query: str, *args):
    """Internal: fetch one row without resilience wrapping."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_one(query: str, *args):
    """Fetch a single row with circuit breaker protection."""
    return await resilient_call(
        _raw_fetch_one, query, *args,
        circuit=db_circuit,
        max_retries=2,
        timeout_s=15.0,
        fallback=None,
        operation_name="fetch_one",
    )


async def _raw_fetch_all(query: str, *args):
    """Internal: fetch all rows without resilience wrapping."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def fetch_all(query: str, *args):
    """Fetch all rows with circuit breaker protection."""
    return await resilient_call(
        _raw_fetch_all, query, *args,
        circuit=db_circuit,
        max_retries=2,
        timeout_s=15.0,
        fallback=list,  # return empty list on total failure
        operation_name="fetch_all",
    )


async def _raw_execute(query: str, *args):
    """Internal: execute query without resilience wrapping."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def execute(query: str, *args):
    """Execute a query with circuit breaker protection."""
    return await resilient_call(
        _raw_execute, query, *args,
        circuit=db_circuit,
        max_retries=1,  # writes: fewer retries to avoid duplicates
        timeout_s=15.0,
        operation_name="execute",
    )


async def _raw_fetch_val(query: str, *args):
    """Internal: fetch single value without resilience wrapping."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def fetch_val(query: str, *args):
    """Fetch a single value with circuit breaker protection."""
    return await resilient_call(
        _raw_fetch_val, query, *args,
        circuit=db_circuit,
        max_retries=2,
        timeout_s=15.0,
        fallback=None,
        operation_name="fetch_val",
    )
```

**Design decisions:**
- **Reads** (`fetch_one`, `fetch_all`, `fetch_val`): 2 retries, safe to retry (idempotent)
- **Writes** (`execute`): 1 retry only, to avoid duplicate INSERT/UPDATE
- **`fetch_all` fallback:** Returns empty list on total failure (graceful degradation for list queries)
- **`execute` no fallback:** Write failures must propagate (caller needs to know the write didn't happen)
- **`fetch_one`/`fetch_val` fallback:** Returns None (callers already check for None)
- **Timeout:** 15s per attempt (less than pool's 30s command_timeout, so resilience layer triggers first)

### Step 3: Add health check for half-open recovery

**File:** `agent_zero/database.py`, add after `close_pool()`:

```python
async def health_check() -> bool:
    """Quick database health check for circuit breaker half-open probing."""
    try:
        result = await _raw_fetch_val("SELECT 1")
        return result == 1
    except Exception:
        return False
```

### Step 4: Expose circuit state for observability

**File:** `agent_zero/database.py`, add at end:

```python
def get_db_circuit_state() -> dict:
    """Return current circuit breaker state for monitoring."""
    return {
        "state": db_circuit.state,
        "failure_count": db_circuit.failure_count,
        "failure_threshold": db_circuit.failure_threshold,
        "recovery_timeout": db_circuit.recovery_timeout,
    }
```

### What NOT to Change

- **Higher-level helpers** (`create_commitment`, `list_commitments`, etc.) -- these already call the 4 core functions, so they automatically get protection.
- **`agent_zero_server.py`** direct database calls -- these also use `fetch_one`/`execute`, so they're covered.
- **`run_migrations()`** -- schema migrations should NOT go through the circuit breaker (they must fail loud).
- **Connection pool creation** -- `get_pool()` stays as-is (one-time init).

## Test Specifications

### Test 1: Circuit breaker trips after repeated failures
```python
async def test_db_circuit_trips_on_failures():
    """After failure_threshold failures, circuit should open."""
    from database import db_circuit, fetch_one
    db_circuit.reset()
    # Mock get_pool to raise ConnectionError
    for _ in range(5):
        try:
            await fetch_one("SELECT 1")
        except Exception:
            pass
    assert db_circuit.state == "open"
```

### Test 2: fetch_all returns empty list fallback
```python
async def test_fetch_all_fallback():
    """When circuit is open, fetch_all should return empty list."""
    from database import db_circuit, fetch_all
    from resilience import CircuitOpenError
    db_circuit.state = "open"
    db_circuit.last_failure_time = time.monotonic()  # prevent auto-recovery
    result = await fetch_all("SELECT * FROM users")
    assert result == []
    db_circuit.reset()
```

### Test 3: Write failures propagate (no fallback)
```python
async def test_execute_no_fallback():
    """Write failures must propagate, not return fallback."""
    from database import db_circuit, execute
    from resilience import CircuitOpenError
    db_circuit.state = "open"
    db_circuit.last_failure_time = time.monotonic()
    with pytest.raises(CircuitOpenError):
        await execute("INSERT INTO users ...")
    db_circuit.reset()
```

### Test 4: Health check function
```python
async def test_health_check():
    """Health check returns True on healthy DB, False on failure."""
    from database import health_check
    # With real DB: should return True
    # With mocked failure: should return False
```

### Test 5: Circuit state reporting
```python
def test_circuit_state_reporting():
    """get_db_circuit_state returns current state dict."""
    from database import get_db_circuit_state, db_circuit
    db_circuit.reset()
    state = get_db_circuit_state()
    assert state["state"] == "closed"
    assert state["failure_count"] == 0
```

### Test 6: Migrations bypass circuit breaker
```python
async def test_migrations_bypass_circuit():
    """run_migrations() should not go through circuit breaker."""
    from database import run_migrations
    # run_migrations uses pool directly, not fetch_one/execute wrappers
    # Verify it doesn't import or use db_circuit
```

### Test 7: Retry idempotency for reads
```python
async def test_read_retries_on_transient_error():
    """Transient errors on reads should retry up to max_retries."""
    # Mock _raw_fetch_one to fail twice then succeed
    # Verify fetch_one returns the successful result
```

## Estimated Impact

1. **Cascade failure prevention.** NeonDB outages will fast-fail after 5 errors (30s recovery) instead of blocking all requests for 30s each (command_timeout).

2. **Graceful degradation.** List queries return empty lists instead of crashing. Read queries return None. Only writes raise exceptions (correctly -- callers must know writes failed).

3. **Automatic recovery.** Half-open probing tests NeonDB recovery. On success, circuit closes and normal operation resumes without restart.

4. **Zero call-site changes.** All 30+ database call sites automatically get protection through the 4 core function wrappers.

5. **Observable.** `get_db_circuit_state()` provides circuit breaker monitoring for dashboards or health endpoints.

6. **Already tested.** The resilience layer itself has 52 passing tests. This paper only adds the wiring.
