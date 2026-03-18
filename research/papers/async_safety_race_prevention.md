---
topic: Async Safety and Race Condition Prevention
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T13:00:00Z
---

# Async Safety and Race Condition Prevention

## Problem Statement

Agent Zero's agent_zero_server.py uses Python asyncio for concurrent WebSocket handling but has
no explicit synchronization primitives protecting shared mutable state. While asyncio's
single-threaded event loop provides implicit safety for purely awaited operations, several
patterns in the codebase create race condition windows:

1. `run_in_executor()` calls (8 instances) run code in a thread pool, breaking asyncio's
   single-thread guarantee
2. Shadow dict is mutated from multiple await points within `_run_conversation_turn()`
   (1463 lines long), and any intermediate await can interleave with consolidation or
   other coroutines
3. `session_messages` list is appended from within the conversation turn while compression
   runs concurrently
4. The inference singleton (`agent_zero_inference.py:429`) uses `asyncio.Lock` correctly for
   model access, but shadow and session state have no equivalent protection

No race conditions have been observed in production yet (single-user system), but the
architecture will fail under concurrent users or if any operation is moved to `run_in_executor`.

## Current State in Agent Zero

### Protected state (good):
- `agent_zero_inference.py:431`: `_inference_lock = asyncio.Lock()` guards model singleton
- `agent_zero_inference.py:69`: `threading.Lock()` guards model loading in thread context
- `agent_zero_telegram/agent_zero_bridge.py:38`: `asyncio.Lock()` for bridge operations

### Unprotected state (risk):
- **Shadow dict**: Created in `_open_ws_session()`, mutated across 13+ locations in
  `agent_zero_server.py` (lines 1935-2703), read/written by consolidation loop (lines 1935-1939),
  episode logging (lines 2679-2703), and finalization (line 3108)
- **session_messages list**: Created per-connection (line 3059), but mutated at lines
  1285, 1587, 1625, 2224, 2626, 2817 -- any `await` between read and append could interleave
- **run_in_executor calls** (lines 626, 1173, 2168, 2255, 2363, 3181, 4381): These run in
  a thread pool. If they access any shared state (e.g., inference.model), they need thread-safe
  protection

### Current architecture:
```
websocket_chat()
  -> session_messages = []       # per-connection, local
  -> shadow = await _open_ws()   # per-connection, local
  -> while True:
       await _run_conversation_turn(...)  # 1463 lines, many awaits
         -> shadow["episodes"].append(...)  # mutation
         -> session_messages.append(...)    # mutation
         -> await consolidation(shadow)     # may mutate shadow
         -> await run_in_executor(...)      # thread-pool escape
```

## Industry Standard / Research Findings

### 1. Asyncio Race Conditions Are Real (Even Single-Threaded)

Inngest's 2025 analysis [1] demonstrated that asyncio's `Condition.wait_for()` loses
intermediate state transitions when transitions happen between await points. The recommended
pattern is per-consumer queues that buffer all state changes, but for Agent Zero's simpler
case, `asyncio.Lock` per session is sufficient.

### 2. FastAPI Global State Anti-Pattern

DataSciOcean's 2025 analysis [2] identifies the core pattern: "Two coroutines might
simultaneously read a value, modify it, and write it back, producing a lost update." Even in
asyncio, a read-modify-write across an await boundary is non-atomic:

```python
# UNSAFE: read-await-write pattern
episodes = shadow.get("episodes", [])
await some_async_operation()  # another coroutine can modify shadow here
shadow["episodes"] = episodes + [new_episode]
```

### 3. Per-Connection State Isolation (Already Mostly Done)

The websockets documentation [3] recommends creating per-connection state objects rather than
sharing global state. Agent Zero already does this -- `session_messages` and `shadow` are created
per WebSocket connection. The risk is that *within* a connection, the long-running
`_run_conversation_turn()` has many await points where a disconnect handler or background
task could access the same state.

### 4. The Golden Rule: Externalize State or Lock It

Aleinikov (2025) [4] recommends: "Use database transactions when modifying persisted data;
use asyncio.Lock when working with in-memory state within a single process." For Agent Zero,
shadow is in-memory state that should use asyncio.Lock.

### 5. Thread Safety for run_in_executor

FastAPI documentation [5] is explicit: sync endpoints run in a threadpool (finite limit).
When using `run_in_executor()`, any shared state accessed inside the executor function must
use `threading.Lock`, not `asyncio.Lock`. Agent Zero's inference module correctly uses both
(threading.Lock for thread context, asyncio.Lock for async context).

## Proposed Implementation

### Phase 1: Add Per-Session Lock

Add an `asyncio.Lock` per WebSocket session to protect shadow and session_messages:

```python
# agent_zero_server.py, in websocket_chat() and websocket_voice()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: Optional[str] = Query(None)):
    await websocket.accept()
    user, session_id, shadow = await _open_ws_session(token)
    inference = await get_inference_async()
    session_messages = []
    session_lock = asyncio.Lock()  # NEW: per-session lock
    message_index = 0

    try:
        # ... existing setup ...
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            user_content = msg.get("content", "").strip()
            if not user_content:
                continue

            async with session_lock:  # NEW: lock for entire turn
                message_index += 1
                await _run_conversation_turn(
                    websocket, user_content, inference,
                    session_messages, message_index,
                    user=user, session_id=session_id,
                    shadow=shadow, ...
                )
    except WebSocketDisconnect:
        pass
    finally:
        async with session_lock:  # NEW: lock for finalization
            await _finalize_ws_session(user, shadow, session_messages, session_id)
```

**Why per-turn locking is sufficient:** Within a single WebSocket connection, messages arrive
sequentially (the `while True` loop awaits one message at a time). The lock prevents
finalization from racing with an in-progress turn. For multiple concurrent connections,
each has its own lock -- no contention.

### Phase 2: Protect Shadow Mutations in _run_conversation_turn

Add a `SessionState` wrapper that makes mutations explicit:

```python
# agent_zero_server.py (new helper)

class SessionState:
    """Thread-safe wrapper for per-session mutable state."""

    __slots__ = ("shadow", "messages", "_lock")

    def __init__(self, shadow: dict, messages: list):
        self.shadow = shadow
        self.messages = messages
        self._lock = asyncio.Lock()

    async def append_message(self, role: str, content: str):
        async with self._lock:
            self.messages.append({"role": role, "content": content})

    async def append_episode(self, episode: dict):
        async with self._lock:
            self.shadow.setdefault("episodes", []).append(episode)

    async def get_shadow_snapshot(self) -> dict:
        """Return a shallow copy for read-only operations."""
        async with self._lock:
            return dict(self.shadow)

    async def run_consolidation(self, consolidation_fn):
        """Run consolidation under lock to prevent concurrent mutation."""
        async with self._lock:
            return consolidation_fn(self.shadow)
```

### Phase 3: Audit run_in_executor Calls

Review each `run_in_executor` call to verify it does not access shared mutable state:

| Line | Function | State Accessed | Safe? |
|------|----------|---------------|-------|
| 626 | Deepgram transcription | audio bytes (immutable) | Yes |
| 1173 | Load shadow from DB | DB connection (pooled) | Yes |
| 2168 | Load model | guarded by threading.Lock | Yes |
| 2255 | Generate response | model (read-only inference) | Yes |
| 2363 | Stream generation | model (read-only inference) | Yes |
| 3181 | Deepgram transcription | audio bytes (immutable) | Yes |
| 4381 | Load model | guarded by threading.Lock | Yes |

**Current assessment:** All `run_in_executor` calls appear safe because they operate on
immutable inputs or are already protected. Document this in a comment block.

### Phase 4: Add Defensive Assertions (Debug Mode)

```python
# agent_zero_server.py

import os
_DEBUG_CONCURRENCY = os.environ.get("AGENT_ZERO_DEBUG_CONCURRENCY", "").lower() == "true"

def _assert_session_lock_held(lock: asyncio.Lock):
    """Debug assertion that session lock is held."""
    if _DEBUG_CONCURRENCY and not lock.locked():
        raise RuntimeError("Session state modified without holding session lock")
```

Call this before every shadow/session_messages mutation in debug mode.

## Test Specifications

### Test: SessionState is async-safe
```python
@pytest.mark.asyncio
async def test_session_state_concurrent_append():
    """Concurrent appends should not lose messages."""
    state = SessionState(shadow={}, messages=[])

    async def append_n(n):
        for i in range(n):
            await state.append_message("user", f"msg-{n}-{i}")

    await asyncio.gather(append_n(100), append_n(100))
    assert len(state.messages) == 200
```

### Test: Consolidation runs under lock
```python
@pytest.mark.asyncio
async def test_consolidation_under_lock():
    """Consolidation should run atomically with respect to episode appends."""
    shadow = {"episodes": []}
    state = SessionState(shadow=shadow, messages=[])

    consolidated = []

    async def consolidate():
        async def fn(s):
            consolidated.append(len(s["episodes"]))
            return s
        await state.run_consolidation(fn)

    await state.append_episode({"id": "1"})
    await consolidate()
    assert consolidated == [1]  # saw exactly 1 episode
```

### Test: Per-session lock prevents turn/finalization race
```python
@pytest.mark.asyncio
async def test_session_lock_prevents_race():
    """Finalization should not run while a turn is in progress."""
    lock = asyncio.Lock()
    order = []

    async def mock_turn():
        async with lock:
            order.append("turn_start")
            await asyncio.sleep(0.01)
            order.append("turn_end")

    async def mock_finalize():
        async with lock:
            order.append("finalize")

    # Start turn, then try to finalize while turn is running
    task1 = asyncio.create_task(mock_turn())
    await asyncio.sleep(0.001)  # let turn start
    task2 = asyncio.create_task(mock_finalize())
    await asyncio.gather(task1, task2)

    assert order == ["turn_start", "turn_end", "finalize"]
```

### Test: run_in_executor does not access shared state
```python
def test_executor_calls_are_stateless():
    """Verify run_in_executor functions don't close over mutable session state."""
    # This is a static analysis test -- grep for run_in_executor and verify
    # the callables don't reference shadow, session_messages, or other session state.
    import ast, inspect
    # ... parse agent_zero_server.py AST, find run_in_executor calls,
    # verify closure variables don't include session-scoped names
```

## Estimated Impact

1. **Correctness under concurrency:** Prevents potential lost updates to shadow and
   session_messages if the system is ever deployed for multiple concurrent users.

2. **Future-proofing:** As Agent Zero scales beyond single-user, per-session locking ensures
   safe operation without requiring a full rewrite.

3. **Minimal performance impact:** `asyncio.Lock` has negligible overhead (~1 microsecond)
   and no contention in single-user mode. Per-session locks never contend with other sessions.

4. **Debug tooling:** Concurrency assertions in debug mode will catch regressions early.

5. **No behavioral change:** The system currently works correctly because there's only one
   user. This change preserves behavior while adding safety guarantees.

## Citations

1. Inngest (2025). "What Python's asyncio primitives get wrong about shared state." https://www.inngest.com/blog/no-lost-updates-python-asyncio
2. DataSciOcean (2025). "The Concurrency Trap in FastAPI: From Race Conditions to Deadlocks." https://datasciocean.com/en/other/fastapi-race-condition/
3. websockets 16.0 documentation. "Server (asyncio)." https://websockets.readthedocs.io/en/stable/reference/asyncio/server.html
4. Aleinikov (2025). "Avoiding Race Conditions in Python in 2025." https://medium.com/pythoneers/avoiding-race-conditions-in-python-in-2025-best-practices-for-async-and-threads-4e006579a622
5. FastAPI documentation. "Concurrency and async/await." https://fastapi.tiangolo.com/async/
6. Johal (2025). "Preventing Race Conditions in Async Python Code." https://johal.in/preventing-race-conditions-in-async-python-code/
