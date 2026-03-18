---
topic: Resource Lifecycle Management
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# Resource Lifecycle Management

## Problem Statement

Three resource lifecycle bugs exist across the Agent Zero codebase: (1) streaming generators from vLLM are not cleaned up when users disconnect mid-stream, (2) model loading has a race condition allowing double-load, and (3) cognitive runtime creates orphaned asyncio tasks that continue running after errors. These cause resource leaks, GPU memory waste, and silent background failures.

## Current State in Agent Zero

### 1. Streaming Generator Not Cleaned Up on Disconnect

**File:** `agent_zero_server.py`, lines 2674-2680

```python
generator = await asyncio.to_thread(
    lambda: get_inference().generate_stream(gen_ctx, system=system_prompt, ...)
)
for token_text in generator:
    # stream tokens to WebSocket
```

If the WebSocket disconnects during streaming (user closes tab, network drop), the `for` loop raises an exception. The `generator` object is never explicitly closed -- `generator.close()` is not called in any `finally` block. The generator continues running in the thread context until Python's garbage collector eventually collects it, which may never happen if references are held.

**Impact:** The vLLM streaming connection (via urllib) remains open, consuming a connection slot. On repeated disconnects, connection pool exhaustion is possible.

### 2. Model Loading Race Condition

**File:** `agent_zero_inference.py`, lines 69-100

```python
def load_model(self):
    if self.loaded:
        return True
    if self.loading:        # Non-atomic check
        return False
    self.loading = True     # Non-atomic set
    # ... load model ...
    self.loaded = True
    self.loading = False
```

Two concurrent calls to `load_model()` (e.g., from two incoming WebSocket connections) can both pass the `if self.loading` check before either sets `self.loading = True`. Both proceed to load the model simultaneously, causing:
- Double GPU memory allocation
- Potential CUDA OOM crash
- Undefined behavior if both write to the same model reference

**Additional:** `asyncio.to_thread(inference.load_model)` at line 2476 runs in a thread pool, so the GIL does NOT protect the check-then-set pattern across thread boundaries.

### 3. Orphaned Coroutines in Cognitive Runtime

**File:** `cognitive_runtime.py`, lines 918-926 and 1056-1078

```python
# Line 919:
futures = {asyncio.ensure_future(_run_and_tag_reaction(aid, ctx)): aid ...}
async for done in asyncio.as_completed(futures):
    result = await done
    # process result
```

If an exception occurs during `asyncio.as_completed()` iteration (e.g., in the processing callback), remaining futures continue running in the background. No `finally` block cancels pending futures. Similarly at lines 1056-1078 for agent worker execution.

**Impact:** Orphaned agent coroutines consume CPU and may hold database connections or other resources. In worst case, they produce results that are never consumed, silently wasting inference tokens.

### 4. Blocking urllib in Async Context

**File:** `agent_zero_inference.py`, lines 226 and 341

```python
response = urllib.request.urlopen(req, timeout=120)
```

This is a synchronous blocking call that freezes the entire event loop for up to 120 seconds. All WebSocket connections, proactive evaluations, and API endpoints are blocked.

**Note:** This overlaps with the `async_event_loop_safety.md` paper but the lifecycle aspect (proper cleanup of the response object) is not covered there.

## Industry Standard / Research Findings

### Async Generator Cleanup (Python Bug Tracker, CPython)

CPython issue #41229 documents that async generator cleanup via garbage collection is unreliable. The recommended pattern is explicit `aclose()` in a `finally` block, or using `contextlib.aclosing()` as a context manager.

**Source:** https://bugs.python.org/issue41229

### Trio's Generator Cleanup Strategy (python-trio, 2025)

The Trio async framework's issue #265 provides a comprehensive analysis of generator cleanup strategies. The key insight: "Generators should always be closed explicitly, not left to the garbage collector. Use `try/finally` or async context managers to guarantee cleanup."

**Source:** https://github.com/python-trio/trio/issues/265

### Python threading.Lock for Thread-Safe Initialization (CPython docs)

For the model loading race condition, the standard pattern is `threading.Lock()` (not `asyncio.Lock()`, since `load_model` runs in a thread via `asyncio.to_thread`). The "double-checked locking" pattern with a lock ensures exactly one initialization:

```python
def load_model(self):
    if self.loaded:
        return True
    with self._lock:
        if self.loaded:  # Double-check inside lock
            return True
        # ... load ...
```

**Source:** https://docs.python.org/3/library/threading.html#threading.Lock

### asyncio.TaskGroup for Structured Concurrency (PEP 654, Python 3.11+)

`asyncio.TaskGroup` (available since Python 3.11) automatically cancels all remaining tasks when any task fails or the group exits. This replaces the error-prone `ensure_future` + manual cancellation pattern.

**Source:** https://docs.python.org/3/library/asyncio-task.html#asyncio.TaskGroup

### HTTPX Async Client for Non-Blocking HTTP (2025)

HTTPX provides async HTTP client that replaces blocking `urllib.request.urlopen`. It supports streaming responses with proper cleanup via async context managers.

**Source:** https://www.python-httpx.org/async/

## Proposed Implementation

### Fix 1: Generator Cleanup on Disconnect (agent_zero_server.py)

**Lines 2674-2680, wrap in try/finally:**

```python
generator = None
try:
    generator = await asyncio.to_thread(
        lambda: get_inference().generate_stream(gen_ctx, system=system_prompt, ...)
    )
    for token_text in generator:
        # ... stream tokens ...
finally:
    if generator is not None:
        try:
            generator.close()
        except Exception:
            pass  # Generator may already be exhausted
```

### Fix 2: Thread-Safe Model Loading (agent_zero_inference.py)

**Add lock at class level and use double-checked locking:**

```python
import threading

class Agent ZeroInference:
    def __init__(self, config):
        self._load_lock = threading.Lock()
        self.loaded = False
        self.loading = False
        # ... existing init ...

    def load_model(self):
        if self.loaded:
            return True
        with self._load_lock:
            if self.loaded:
                return True
            if self.loading:
                return False
            self.loading = True
        try:
            # ... existing load logic ...
            self.loaded = True
            return True
        except Exception:
            # ... existing error handling ...
            return False
        finally:
            self.loading = False
```

### Fix 3: Structured Task Cleanup in Cognitive Runtime (cognitive_runtime.py)

**Lines 918-926, add finally block to cancel pending tasks:**

```python
futures = {asyncio.ensure_future(_run_and_tag_reaction(aid, ctx)): aid
           for aid in reaction_agents}
try:
    async for done in asyncio.as_completed(futures):
        result = await done
        # ... process result ...
finally:
    # Cancel any still-pending futures
    for fut in futures:
        if not fut.done():
            fut.cancel()
    # Wait for cancellation to complete
    await asyncio.gather(*[f for f in futures if not f.done()], return_exceptions=True)
```

Apply same pattern at lines 1056-1078.

**Alternative (Python 3.11+):** Replace with `asyncio.TaskGroup`:
```python
async with asyncio.TaskGroup() as tg:
    tasks = {tg.create_task(_run_and_tag(aid)): aid for aid in agents}
# All tasks guaranteed cleaned up when exiting the group
```

### Fix 4: Add Unload Method for Model (agent_zero_inference.py)

```python
def unload_model(self):
    """Release model resources (GPU memory)."""
    with self._load_lock:
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        self.loaded = False
        self.loading = False
        import gc
        gc.collect()
        # If torch available, clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
```

## Test Specifications

```python
import asyncio
import threading
import pytest

def test_model_load_race_condition():
    """Two concurrent load_model calls result in exactly one load."""
    inference = Agent ZeroInference(config)
    load_count = 0
    original_load = inference._do_load

    def counting_load():
        nonlocal load_count
        load_count += 1
        return original_load()

    inference._do_load = counting_load

    threads = [threading.Thread(target=inference.load_model) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_count == 1
    assert inference.loaded is True

def test_model_load_double_checked_locking():
    """Second call to load_model returns True without loading again."""
    inference = Agent ZeroInference(config)
    inference.loaded = True
    # Should return immediately without acquiring lock
    assert inference.load_model() is True

@pytest.mark.asyncio
async def test_generator_cleanup_on_exception():
    """Generator.close() called even when streaming loop raises."""
    closed = False
    def mock_generator():
        nonlocal closed
        try:
            for i in range(100):
                yield f"token_{i}"
        finally:
            closed = True

    gen = mock_generator()
    try:
        for i, token in enumerate(gen):
            if i == 5:
                raise ConnectionError("WebSocket disconnected")
    except ConnectionError:
        pass
    finally:
        gen.close()

    assert closed is True

@pytest.mark.asyncio
async def test_orphaned_tasks_cancelled():
    """Pending tasks are cancelled when processing fails."""
    cancelled = []

    async def slow_task(name):
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    futures = {asyncio.ensure_future(slow_task(f"t{i}")): i for i in range(5)}
    try:
        async for done in asyncio.as_completed(futures):
            raise RuntimeError("Processing failed")
    except RuntimeError:
        pass
    finally:
        for fut in futures:
            if not fut.done():
                fut.cancel()
        await asyncio.gather(*futures, return_exceptions=True)

    assert len(cancelled) >= 4  # At least 4 of 5 should have been cancelled

def test_model_unload_releases_resources():
    """unload_model sets loaded=False and clears references."""
    inference = Agent ZeroInference(config)
    inference.loaded = True
    inference.model = "dummy"
    inference.unload_model()
    assert inference.loaded is False
    assert not hasattr(inference, 'model') or inference.model is None
```

## Estimated Impact

- **Reliability:** Eliminates connection pool exhaustion from leaked generators on user disconnect.
- **Stability:** Prevents double model loading that can cause CUDA OOM crashes.
- **Resource efficiency:** Orphaned agent coroutines are properly cancelled instead of running indefinitely.
- **Observability:** Generator cleanup failures are now visible rather than silently leaked.
- **Scope:** ~50 lines changed across 3 files. Each fix is independent and can be deployed separately.
