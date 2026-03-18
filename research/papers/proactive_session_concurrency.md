---
topic: Proactive Session Concurrency Safety
status: ready_for_implementation
priority: high
estimated_complexity: small
researched_at: 2026-03-18T14:00:00Z
---

# Proactive Session Concurrency Safety

## Problem Statement

The proactive evaluation loop in agent_zero_server.py accesses a shared `_proactive_sessions` dictionary from multiple concurrent coroutines without any synchronization. The loop catches all exceptions with bare `except Exception: pass`, silently masking WebSocket errors, database failures, and logic bugs. Closed WebSocket connections are never checked before sending, causing RuntimeErrors that are silently swallowed.

## Current State in Agent Zero

### Shared Dictionary Without Lock

**File:** `agent_zero_server.py`

- **Line 196-236**: `_proactive_sessions` is a plain dict accessed by three functions:
  - `_proactive_evaluation_loop()` (line 204) -- iterates dict values in a loop
  - `_register_proactive_session()` (line 226) -- writes to dict
  - `_unregister_proactive_session()` (line 236) -- deletes from dict
  - `_run_conversation_turn()` (line 3418) -- reads/writes to dict

- **Line 204**: The evaluation loop iterates `list(_proactive_sessions.items())` which creates a snapshot, but between snapshot and `ws.send_text()` (line 216), the session could be unregistered and the WebSocket closed.

- **Line 3417-3418**: Check-then-act pattern is non-atomic:
  ```python
  if session_id and session_id in _proactive_sessions:
      _proactive_sessions[session_id]["last_user_msg_at"] = ...
  ```
  Between the `in` check and the dict access, another coroutine at an await point could delete the entry.

### Silent Exception Swallowing

- **Line 217-218**:
  ```python
  except Exception:
      pass  # Don't crash the loop
  ```
  This masks:
  - WebSocket closed-connection RuntimeErrors
  - Database connection failures from `evaluate_proactive_triggers()`
  - JSON serialization errors
  - KeyError from deleted sessions
  - Any logic bugs in the proactive evaluation path

  No logging, no metrics, no error recovery. If `evaluate_proactive_triggers()` consistently fails for a user, they silently receive no proactive messages forever.

### Missing WebSocket State Check

- **Line 216**: `await ws.send_text(...)` is called on a stored WebSocket reference without checking if the connection is still open. Starlette WebSocket objects have a `client_state` attribute that should be checked before sending.

### No Session Cleanup on Failure

If sending fails (WebSocket closed), the session remains in `_proactive_sessions`. The loop will attempt to send again on the next cycle (60 seconds), fail again silently, and repeat indefinitely. No mechanism removes dead sessions.

## Industry Standard / Research Findings

### Asyncio Shared State Safety (2025)

Kovalchuk (2025) demonstrates that Python dictionaries are NOT async-safe despite the GIL, because coroutines can interleave at any `await` point. The recommended pattern is `asyncio.Lock()` around all shared-state mutations:

```python
lock = asyncio.Lock()
async def safe_access():
    async with lock:
        if key in shared_dict:
            shared_dict[key]["field"] = value
```

**Source:** https://medium.com/@goldengrisha/is-pythons-dictionary-async-safe-why-you-can-still-get-race-conditions-in-async-code-c786412af567

### Best Practices for Async Concurrency (Aleinikov, 2025)

Aleinikov's 2025 guide on avoiding race conditions in Python identifies three key patterns: (1) use `asyncio.Lock` for shared state, (2) never assume dict operations are atomic across await boundaries, (3) implement high-concurrency testing to catch intermittent failures.

**Source:** https://medium.com/pythoneers/avoiding-race-conditions-in-python-in-2025-best-practices-for-async-and-threads-4e006579a622

### Python asyncio Synchronization Primitives (CPython docs, 3.14)

The official Python docs provide `asyncio.Lock`, `asyncio.Event`, `asyncio.Condition` as the correct primitives for protecting shared state in async code. The Lock is reentrant-safe and works correctly with `async with`.

**Source:** https://docs.python.org/3/library/asyncio-sync.html

### Coroutine Safety (Super Fast Python, 2025)

Brownlee's comprehensive guide defines "coroutine-safe" as code that can be executed concurrently with multiple coroutines without race conditions. Shared mutable state (dicts, lists, sets) requires explicit protection via asyncio synchronization primitives.

**Source:** https://superfastpython.com/asyncio-coroutine-safe/

### Inngest Lost Updates Analysis (2025)

Inngest's analysis of Python asyncio primitives shows that "check-then-act" patterns (like `if key in dict: dict[key] = ...`) are the most common source of lost updates in async Python. The fix is either an `asyncio.Lock` or restructuring to use atomic operations.

**Source:** https://www.inngest.com/blog/no-lost-updates-python-asyncio

## Proposed Implementation

### Step 1: Add asyncio.Lock for _proactive_sessions

**File:** `agent_zero_server.py`, near line 196

```python
# BEFORE:
_proactive_sessions: dict[str, dict] = {}

# AFTER:
_proactive_sessions: dict[str, dict] = {}
_proactive_sessions_lock = asyncio.Lock()
```

### Step 2: Protect all _proactive_sessions access with the lock

**Register (line 226):**
```python
async def _register_proactive_session(session_id: str, ws, shadow, config):
    async with _proactive_sessions_lock:
        _proactive_sessions[session_id] = {
            "ws": ws, "shadow": shadow, "config": config,
            "last_user_msg_at": datetime.now(timezone.utc),
        }
```

**Unregister (line 236):**
```python
async def _unregister_proactive_session(session_id: str):
    async with _proactive_sessions_lock:
        _proactive_sessions.pop(session_id, None)
```

**Conversation turn update (line 3417-3418):**
```python
async with _proactive_sessions_lock:
    if session_id and session_id in _proactive_sessions:
        _proactive_sessions[session_id]["last_user_msg_at"] = datetime.now(timezone.utc)
```

**Evaluation loop (line 204):**
```python
async with _proactive_sessions_lock:
    snapshot = list(_proactive_sessions.items())
```

### Step 3: Add WebSocket state check before send

**Line 216, inside the loop:**
```python
from starlette.websockets import WebSocketState

for session_id, info in snapshot:
    ws = info["ws"]
    # Check WebSocket is still connected
    if ws.client_state != WebSocketState.CONNECTED:
        async with _proactive_sessions_lock:
            _proactive_sessions.pop(session_id, None)
        continue
    try:
        result = await evaluate_proactive_triggers(info["shadow"], info["config"], ...)
        if result:
            await ws.send_text(json.dumps({"type": "proactive", "content": result}))
    except Exception as exc:
        logger.warning("Proactive evaluation failed for session %s: %s", session_id, exc)
        # Remove dead sessions
        async with _proactive_sessions_lock:
            _proactive_sessions.pop(session_id, None)
```

### Step 4: Replace bare except-pass with structured logging

**Line 217-218:**
```python
# BEFORE:
except Exception:
    pass

# AFTER:
except Exception as exc:
    logger.warning(
        "Proactive eval failed for session %s: %s",
        session_id, type(exc).__name__,
    )
    # If WebSocket error, remove the dead session
    if isinstance(exc, (RuntimeError, ConnectionError, WebSocketDisconnect)):
        async with _proactive_sessions_lock:
            _proactive_sessions.pop(session_id, None)
```

## Test Specifications

```python
import asyncio
import pytest

@pytest.mark.asyncio
async def test_proactive_register_unregister_thread_safe():
    """Concurrent register/unregister does not raise or lose entries."""
    lock = asyncio.Lock()
    sessions = {}

    async def register(sid):
        async with lock:
            sessions[sid] = {"ws": None}

    async def unregister(sid):
        async with lock:
            sessions.pop(sid, None)

    # Register 100 sessions, unregister 50 concurrently
    tasks = [register(f"s{i}") for i in range(100)]
    tasks += [unregister(f"s{i}") for i in range(0, 100, 2)]
    await asyncio.gather(*tasks)
    assert len(sessions) == 50

@pytest.mark.asyncio
async def test_proactive_loop_skips_closed_websocket():
    """Loop does not attempt send on closed WebSocket."""
    # Mock WebSocket with client_state = DISCONNECTED
    # Verify send_text is NOT called
    # Verify session is removed from dict

@pytest.mark.asyncio
async def test_proactive_loop_logs_on_failure():
    """Loop logs warning instead of silently swallowing errors."""
    # Mock evaluate_proactive_triggers to raise RuntimeError
    # Verify logger.warning was called with session ID

@pytest.mark.asyncio
async def test_check_then_act_atomic():
    """Dict read + write in conversation turn is atomic."""
    # Simulate concurrent update and delete
    # Verify no KeyError raised

@pytest.mark.asyncio
async def test_dead_session_removed_after_send_failure():
    """Session is removed from dict after WebSocket send fails."""
    # Mock ws.send_text to raise RuntimeError
    # Verify session removed from _proactive_sessions
```

## Estimated Impact

- **Reliability:** Eliminates silent WebSocket send failures that cause proactive messages to silently stop for affected users.
- **Debuggability:** Structured logging replaces `pass` -- failures now appear in logs for diagnosis.
- **Data integrity:** Lock prevents dict mutation race conditions between coroutines.
- **Resource cleanup:** Dead sessions are removed instead of accumulating in memory.
- **Scope:** Small change (~30 lines modified), high confidence, no backward compatibility concerns.
