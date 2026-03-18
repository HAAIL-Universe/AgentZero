---
topic: Streaming Generation Timeout and Recovery
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T22:30:00Z
---

# Streaming Generation Timeout and Recovery

## Problem Statement

Agent Zero's LLM streaming generation has three paths where a hung model causes the WebSocket client to block indefinitely:

1. **vLLM streaming loop** -- The 120-second timeout on `urlopen()` (agent_zero_inference.py:226) is a connection-level timeout, not a per-token timeout. If vLLM sends even 1 byte every 119 seconds, the timeout never fires. A model stuck in a long decode can keep the connection alive while producing no useful tokens for minutes.

2. **Local model `thread.join()`** -- The local Phi-3 streaming path (agent_zero_inference.py:422) calls `thread.join()` with **no timeout**. If `model.generate()` hangs (GPU OOM, CUDA stall, infinite loop), the join blocks forever, freezing the WebSocket handler.

3. **No fallback response** -- When streaming fails, the error handler (agent_zero_server.py:2687-2689) sends `[Generation error: ...]` as a raw token. There is no graceful degradation: no canned response, no retry, no user-friendly error message.

These are not theoretical risks. vLLM Issue #17385 (April 2025) and #17972 (May 2025) document the exact failure mode: vLLM hangs on successive batches or becomes unresponsive after initial requests.

## Current State in Agent Zero

### vLLM Streaming Generator (agent_zero_inference.py:192-241)
```python
def _vllm_stream(self, conversation, max_tokens=2048, ...):
    # ...
    with urllib.request.urlopen(req, timeout=120) as resp:  # Line 226
        for raw_line in resp:       # No per-line timeout
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue  # Silent skip -- no logging
```

**Issues:**
- `timeout=120` is socket-level, not per-token
- Malformed JSON chunks silently skipped with no logging
- No total generation time cap

### Local Model Streaming (agent_zero_inference.py:382-422)
```python
thread = threading.Thread(target=self._generate_in_thread, args=(generation_kwargs,))
thread.start()
for text in streamer:
    yield text
thread.join()  # Line 422 -- NO TIMEOUT
```

**Issues:**
- `thread.join()` blocks indefinitely if model hangs
- `TextIteratorStreamer` queue has no size limit (unbounded memory)

### WebSocket Streaming Handler (agent_zero_server.py:2604-2696)
```python
generator = await loop.run_in_executor(
    None, lambda: get_inference().generate_stream(
        gen_ctx, system=system_prompt, enable_thinking=_enable_thinking
    )
)

for token_text in generator:  # Sync iteration in executor result
    # ... thinking phase + streaming logic ...
```

**Issues:**
- `run_in_executor` wraps sync generator -- no async timeout possible on individual tokens
- Exception handler sends raw error text to client (agent_zero_server.py:2687-2689)

### Error Handler (agent_zero_server.py:2687-2689)
```python
except Exception as e:
    full_response = f"[Generation error: {e}]"
    await websocket.send_text(json.dumps({"type": "token", "content": full_response}))
```

**Issues:**
- Exposes internal error details to client (information leakage)
- No retry, no fallback, no user-friendly message

## Industry Standard / Research Findings

### 1. Per-Token Timeout Pattern (Pankrashov 2025)

The standard approach for streaming timeout is **per-chunk timeout** rather than whole-stream timeout. Pankrashov (2025) describes a class-based async generator wrapper that applies `asyncio.wait_for()` to each `__anext__()` call individually. This catches both connection-level hangs and token-level stalls.

**Pattern:**
```python
class TimeoutAsyncIterator:
    def __init__(self, aiterable, timeout):
        self._ait = aiterable.__aiter__()
        self._timeout = timeout

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await asyncio.wait_for(self._ait.__anext__(), self._timeout)
        except asyncio.TimeoutError:
            raise StopAsyncIteration
```

For synchronous generators in thread executors, the equivalent is wrapping the executor call with `asyncio.wait_for()` on each iteration, or using a queue-based approach with timeout on `queue.get()`.

**Source:** https://medium.com/@dmitry8912/implementing-timeouts-in-pythons-asynchronous-generators-f7cbaa6dc1e9

### 2. LLM Fallback Chains (ZenML 2025 + LiteLLM)

Production LLM deployments use multi-tier fallback chains:

1. **Primary**: Full model generation with streaming
2. **Secondary**: Cached response for similar queries (if available)
3. **Tertiary**: Template-based graceful degradation message

ZenML's 2025 survey of 1,200 production deployments found that the most mature implementations use circuit breakers with hard limits that automatically stop agents when thresholds are exceeded, then gracefully hand off.

LiteLLM's production best practices recommend configuring request timeouts (e.g., 600s for long generations) alongside per-token streaming timeouts to detect stalled models quickly.

**Sources:**
- ZenML LLMOps Database: https://www.zenml.io/llmops-database/implementing-llm-fallback-mechanisms-for-production-incident-response-system
- LiteLLM Production: https://docs.litellm.ai/docs/proxy/prod
- ZenML 1200 Deployments: https://www.zenml.io/blog/what-1200-production-deployments-reveal-about-llmops-in-2025

### 3. vLLM Known Hanging Issues (2025)

Multiple vLLM GitHub issues document the exact failure mode Agent Zero is exposed to:

- **Issue #17385** (April 2025): AsyncLLM hangs on second batch of requests. Model processes first batch, then stops responding.
- **Issue #17972** (May 2025): vLLM server hangs after initial requests, subsequent requests timeout.
- **Issue #23582** (2025): Server timeout due to multiprocessing communication error, EngineDeadError raised during async output.

These are upstream bugs that Agent Zero cannot fix but must defend against with timeouts and fallbacks.

**Sources:**
- vLLM #17385: https://github.com/vllm-project/vllm/issues/17385
- vLLM #17972: https://github.com/vllm-project/vllm/issues/17972
- vLLM #23582: https://github.com/vllm-project/vllm/issues/23582

### 4. asyncio.timeout() (Python 3.11+ Standard Library)

Python 3.11 introduced `asyncio.timeout()` as the preferred timeout context manager, replacing the older `asyncio.wait_for()` pattern for block-level timeouts. For per-iteration timeouts on async generators, the pattern is to reset the timeout deadline on each successful iteration.

**Source:** https://docs.python.org/3/library/asyncio-task.html

### 5. Graceful Degradation for AI Companions (Boral 2025)

Boral (2025) documents strategies for reducing LLM inference latency in production: when primary inference stalls, the system should degrade to simpler responses rather than showing errors. For a behavioral AI companion like Agent Zero, this means maintaining conversational continuity even when the model is unavailable.

**Source:** https://medium.com/@sumanta.boral/strategies-for-reducing-llm-inference-latency-and-making-tradeoffs-lessons-from-building-9434a98e91bc

### 6. Error Handling in Production LLM Applications (MarkAICode 2025)

Production LLM error handling should categorize errors (transient vs permanent), apply circuit breakers per provider, and never expose raw exception details to users. User-facing messages should be helpful and maintain trust.

**Source:** https://markaicode.com/llm-error-handling-production-guide/

## Proposed Implementation

### Phase 1: Per-Token Timeout for vLLM Streaming (agent_zero_inference.py)

**Replace the vLLM streaming loop with a queue-based approach that supports per-token timeout:**

```python
import queue
import threading

# Configuration constant (add to config.py)
STREAM_TOKEN_TIMEOUT = 30  # seconds between tokens before considering model hung
STREAM_TOTAL_TIMEOUT = 300  # seconds total generation time

def _vllm_stream(self, conversation, max_tokens=2048, temperature=0.7,
                 top_p=0.9, system=None, enable_thinking=True):
    """Stream tokens from vLLM with per-token timeout."""
    import urllib.request
    import time

    messages = [{"role": "system", "content": system or SYSTEM_PROMPT}]
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    payload_dict = {
        "model": VLLM_MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if not enable_thinking:
        payload_dict["chat_template_kwargs"] = {"enable_thinking": False}
    payload = json.dumps(payload_dict).encode("utf-8")

    url = VLLM_API_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    token_queue = queue.Queue(maxsize=256)  # Bounded queue for backpressure
    error_holder = [None]  # Mutable to capture thread exceptions
    start_time = time.monotonic()

    def _read_stream():
        try:
            with urllib.request.urlopen(req, timeout=STREAM_TOKEN_TIMEOUT) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            token_queue.put(text, timeout=10)
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        _log.warning("vllm_chunk_parse_error", error=str(e))
                        continue
        except Exception as e:
            error_holder[0] = e
        finally:
            token_queue.put(None)  # Sentinel

    reader = threading.Thread(target=_read_stream, daemon=True)
    reader.start()

    while True:
        # Check total timeout
        elapsed = time.monotonic() - start_time
        if elapsed > STREAM_TOTAL_TIMEOUT:
            _log.warning("vllm_total_timeout", elapsed=elapsed)
            break

        try:
            token = token_queue.get(timeout=STREAM_TOKEN_TIMEOUT)
        except queue.Empty:
            _log.warning("vllm_token_timeout",
                         seconds=STREAM_TOKEN_TIMEOUT,
                         msg="No token received within timeout")
            break

        if token is None:  # Sentinel
            break
        yield token

    if error_holder[0]:
        _log.error("vllm_stream_error", error=str(error_holder[0]))
```

**Add to config.py:**
```python
stream_token_timeout: int = Field(
    default=30, ge=5, le=120,
    description="Max seconds to wait between streaming tokens"
)
stream_total_timeout: int = Field(
    default=300, ge=30, le=600,
    description="Max total seconds for streaming generation"
)
```

### Phase 2: Local Model Thread Timeout (agent_zero_inference.py:422)

**Replace unbounded `thread.join()` with timeout:**

```python
thread = threading.Thread(target=self._generate_in_thread, args=(generation_kwargs,))
thread.start()

for text in streamer:
    yield text

thread.join(timeout=STREAM_TOTAL_TIMEOUT)
if thread.is_alive():
    _log.error("local_model_thread_hung",
               msg="Generation thread did not complete within timeout")
    # Thread will be cleaned up when process exits (daemon=True)
    # Or on next model reload
```

Also make the thread a daemon thread:
```python
thread = threading.Thread(target=self._generate_in_thread,
                          args=(generation_kwargs,), daemon=True)
```

### Phase 3: Graceful Fallback Response (agent_zero_server.py:2687-2689)

**Replace raw error text with user-friendly fallback:**

```python
except Exception as e:
    _log.error("streaming_generation_failed",
               error=str(e), error_type=type(e).__name__)

    # Classify error for appropriate response
    if isinstance(e, (TimeoutError, queue.Empty, asyncio.TimeoutError)):
        fallback_msg = ("I'm sorry, I'm taking longer than usual to think. "
                        "Could you try again? If this keeps happening, "
                        "it might help to simplify your question.")
    elif "connection" in str(e).lower() or "url" in str(e).lower():
        fallback_msg = ("I'm having trouble connecting to my thinking engine "
                        "right now. Please try again in a moment.")
    else:
        fallback_msg = ("I encountered an unexpected issue while responding. "
                        "Please try again.")

    full_response = fallback_msg
    await websocket.send_text(json.dumps({
        "type": "token",
        "content": full_response
    }))
```

### Phase 4: Log Malformed vLLM Chunks (agent_zero_inference.py:234-241)

**Replace silent `continue` with structured logging:**

Already addressed in Phase 1 above -- the `except` block now logs with `_log.warning("vllm_chunk_parse_error", ...)`.

### Phase 5: Generation Timeout Metric (optional, for observability)

**Track timeout occurrences for monitoring:**

```python
# In the streaming handler, after fallback:
if isinstance(e, (TimeoutError, queue.Empty)):
    # Increment timeout counter for observability
    _log.info("generation_timeout_metric",
              timeout_type="token" if isinstance(e, queue.Empty) else "total",
              elapsed=time.monotonic() - gen_start_time)
```

## Test Specifications

### Test 1: Per-Token Timeout Detection
```python
def test_vllm_stream_token_timeout(monkeypatch):
    """Stream should stop if no token received within timeout."""
    import queue

    # Mock vLLM to send one token then stall
    def mock_urlopen(req, timeout=None):
        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self):
                yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n"
                import time; time.sleep(999)  # Stall forever
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    monkeypatch.setattr("agent_zero_inference.STREAM_TOKEN_TIMEOUT", 1)

    inference = Agent ZeroInference()
    tokens = list(inference._vllm_stream([{"role": "user", "content": "test"}]))
    assert tokens == ["hello"]  # Got first token, then timed out gracefully
```

### Test 2: Total Generation Timeout
```python
def test_vllm_stream_total_timeout(monkeypatch):
    """Stream should stop after total timeout even if tokens keep arriving."""
    def mock_urlopen(req, timeout=None):
        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self):
                import time
                for i in range(1000):
                    yield f'data: {{"choices": [{{"delta": {{"content": "tok{i}"}}}}]}}\n'.encode()
                    time.sleep(0.1)
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    monkeypatch.setattr("agent_zero_inference.STREAM_TOTAL_TIMEOUT", 2)

    inference = Agent ZeroInference()
    tokens = list(inference._vllm_stream([{"role": "user", "content": "test"}]))
    assert len(tokens) < 1000  # Should have been cut short
    assert len(tokens) > 0  # Should have gotten some tokens
```

### Test 3: Local Model Thread Timeout
```python
def test_local_model_thread_timeout(monkeypatch):
    """Local model thread.join should not block forever."""
    import threading

    original_join = threading.Thread.join
    join_timeout_used = []

    def tracking_join(self, timeout=None):
        join_timeout_used.append(timeout)
        original_join(self, timeout=0.01)

    monkeypatch.setattr(threading.Thread, "join", tracking_join)
    # ... run local model streaming ...
    # Verify timeout was passed to thread.join
    assert any(t is not None and t > 0 for t in join_timeout_used)
```

### Test 4: Graceful Fallback Messages
```python
@pytest.mark.asyncio
async def test_timeout_fallback_message():
    """Timeout errors should produce user-friendly messages."""
    import queue
    e = queue.Empty()
    msg = _get_fallback_message(e)
    assert "sorry" in msg.lower() or "try again" in msg.lower()
    assert "[" not in msg  # No raw error brackets
    assert "Exception" not in msg  # No exception class names

@pytest.mark.asyncio
async def test_connection_fallback_message():
    """Connection errors should mention connectivity."""
    e = ConnectionError("Connection refused")
    msg = _get_fallback_message(e)
    assert "connect" in msg.lower() or "trouble" in msg.lower()

@pytest.mark.asyncio
async def test_generic_fallback_message():
    """Unknown errors should not leak details."""
    e = RuntimeError("Internal state corruption at 0xDEAD")
    msg = _get_fallback_message(e)
    assert "0xDEAD" not in msg
    assert "corruption" not in msg
```

### Test 5: Bounded Queue Backpressure
```python
def test_vllm_stream_queue_bounded():
    """Token queue should have a max size to prevent unbounded memory."""
    # Verify queue is created with maxsize parameter
    # When queue is full, producer should block (not grow unbounded)
    q = queue.Queue(maxsize=256)
    for i in range(256):
        q.put(f"token{i}")
    assert q.full()
    # Verify put with timeout raises when full
    with pytest.raises(queue.Full):
        q.put("overflow", timeout=0.01)
```

### Test 6: Malformed Chunk Logging
```python
def test_malformed_chunks_logged(monkeypatch, caplog):
    """Malformed SSE chunks should be logged, not silently dropped."""
    def mock_urlopen(req, timeout=None):
        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def __iter__(self):
                yield b"data: {invalid json}\n"
                yield b"data: [DONE]\n"
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    inference = Agent ZeroInference()
    tokens = list(inference._vllm_stream([{"role": "user", "content": "test"}]))
    assert tokens == []  # No valid tokens
    assert "vllm_chunk_parse_error" in caplog.text
```

## Estimated Impact

| Area | Before | After |
|------|--------|-------|
| **vLLM hang** | WebSocket blocks forever | 30s per-token timeout, graceful fallback |
| **Local model hang** | `thread.join()` blocks forever | Timeout + daemon thread cleanup |
| **User experience** | `[Generation error: ...]` raw text | Friendly fallback messages |
| **Memory safety** | Unbounded streamer queue | 256-item bounded queue |
| **Observability** | Silent chunk parse failures | Structured warning logs |
| **Total time cap** | None (could run 24h+) | 300s configurable maximum |
| **Error leakage** | Internal exception details sent to client | Generic user-facing messages |

**Backward compatibility:** All changes are internal. No API changes, no frontend changes needed. The WebSocket message format (`{"type": "token", "content": "..."}`) is unchanged -- only the content of error tokens changes from raw exceptions to friendly messages.

**Configuration:** Two new config fields (`stream_token_timeout`, `stream_total_timeout`) with sensible defaults. No env var changes required for existing deployments.
