---
topic: Async Event Loop Safety
status: ready_for_implementation
priority: high
estimated_complexity: small
researched_at: 2026-03-18T22:00:00Z
---

# Async Event Loop Safety

## Problem Statement

Agent Zero's async codebase contains 15 event-loop-blocking antipatterns across 4 modules: 9 uses of the deprecated `asyncio.get_event_loop()` (emits DeprecationWarning in Python 3.10+, RuntimeError in some Python 3.12+ configurations), 2 blocking `subprocess.run()` calls that freeze the event loop for up to 10 seconds, and 4 synchronous `urllib.request.urlopen()` calls that block for up to 120 seconds each. These block ALL concurrent WebSocket connections during execution -- a single slow Deepgram or vLLM API call stalls every user's session.

## Current State in Agent Zero

### 1. Deprecated `asyncio.get_event_loop()` (11 sites)

**agent_zero_server.py:**
| Line | Context |
|------|---------|
| 723 | `loop = asyncio.get_event_loop()` -- audio transcription path |
| 1413 | `await asyncio.get_event_loop().run_in_executor(None, inference.load_model)` -- model loading |
| 2412 | `await asyncio.get_event_loop().run_in_executor(None, inference.load_model)` -- model startup |
| 2498 | `loop = asyncio.get_event_loop()` -- tool execution |
| 2612 | `loop = asyncio.get_event_loop()` -- generation context |
| 3447 | `loop = asyncio.get_event_loop()` -- WebSocket audio transcription |
| 4653 | `loop = asyncio.get_event_loop()` -- model preload |
| 4658 | `deadline = asyncio.get_event_loop().time() + timeout_seconds` -- timeout calc |
| 4659 | `while asyncio.get_event_loop().time() < deadline:` -- polling loop |

**cognitive_runtime.py:**
| Line | Context |
|------|---------|
| 856 | `loop = asyncio.get_event_loop()` -- agent reaction mode |
| 1290 | `loop = asyncio.get_event_loop()` -- agent execution wrapper |

Lines 4658-4659 are worst: they call `get_event_loop()` repeatedly in a tight polling loop.

### 2. Blocking `subprocess.run()` in async context (2 sites)

**tool_runtime.py:**
| Line | Context |
|------|---------|
| 549 | `subprocess.run(["py", "-3.12", str(_MQ_SCRIPT), "send", ...])` -- MQ send |
| 591 | `subprocess.run(["py", "-3.12", str(_MQ_SCRIPT), "inbox", "agent_zero"])` -- MQ inbox |

Both have `timeout=10`. During that 10 seconds, the entire event loop is frozen.

### 3. Blocking synchronous HTTP calls (3 sites)

**voice.py:**
| Line | Context |
|------|---------|
| 90 | `urllib.request.urlopen(req, timeout=60)` -- Deepgram transcription |

**agent_zero_inference.py:**
| Line | Context |
|------|---------|
| 226 | `urllib.request.urlopen(req, timeout=120)` -- vLLM streaming |
| 341 | `urllib.request.urlopen(req, timeout=120)` -- vLLM non-streaming |

The inference calls are 120-second blocking windows. If called from an async context without executor wrapping, every WebSocket connection stalls.

### 4. Python version

System runs **Python 3.12.3**, where `get_event_loop()` deprecation is active.

## Industry Standard / Research Findings

### asyncio.get_event_loop() deprecation

Python 3.10 deprecated `get_event_loop()` when called without a running loop. Python 3.12 raises `DeprecationWarning` actively. The Python docs state: "Application developers should typically use `asyncio.run()` and should rarely need to reference the loop object." The replacement is `asyncio.get_running_loop()` inside coroutines.

- **Source:** [Python 3.14 Event Loop docs](https://docs.python.org/3/library/asyncio-eventloop.html)
- **Source:** [CPython issue #93453 -- finish deprecation](https://github.com/python/cpython/issues/93453)
- **Source:** [Changes to async event loops in Python 3.10](https://blog.teclado.com/changes-to-async-event-loops-in-python-3-10/)

### Blocking call detection

Python's asyncio debug mode logs "slow callbacks" exceeding 100ms (`loop.slow_callback_duration`). Production tools like **aiocop** (< 0.05% overhead, sys.audit hooks) and **aiodebug** provide continuous monitoring without debug mode.

- **Source:** [Python asyncio dev docs](https://docs.python.org/3/library/asyncio-dev.html)
- **Source:** [aiocop -- non-intrusive asyncio monitoring](https://github.com/Feverup/aiocop)
- **Source:** [DZone -- Python async/sync advanced blocking detection](https://dzone.com/articles/python-asyncsync-advanced-blocking-detection-and-b)

### asyncio.to_thread() vs run_in_executor()

`asyncio.to_thread()` (Python 3.9+) is the modern replacement for `loop.run_in_executor(None, func)`. It doesn't require obtaining the loop reference and is simpler to use. For subprocess calls, `asyncio.create_subprocess_exec()` with `await proc.communicate()` is the correct non-blocking pattern.

- **Source:** [Python 3.14 subprocess docs](https://docs.python.org/3/library/asyncio-subprocess.html)
- **Source:** [Super Fast Python -- asyncio create_subprocess_exec](https://superfastpython.com/asyncio-create_subprocess_exec/)
- **Source:** [Codilime -- running blocking functions in event loop](https://codilime.com/blog/how-fit-triangles-into-squares-run-blocking-functions-event-loop/)

## Proposed Implementation

### Phase 1: Replace deprecated get_event_loop() (11 sites)

In all 11 locations, replace:
```python
# OLD
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_fn)

# NEW
result = await asyncio.to_thread(blocking_fn)
```

For the deadline polling loop (agent_zero_server.py:4658-4659):
```python
# OLD
deadline = asyncio.get_event_loop().time() + timeout_seconds
while asyncio.get_event_loop().time() < deadline:

# NEW
loop = asyncio.get_running_loop()
deadline = loop.time() + timeout_seconds
while loop.time() < deadline:
```

For simple loop references used only for `.time()` or callbacks:
```python
# OLD
loop = asyncio.get_event_loop()
# NEW
loop = asyncio.get_running_loop()
```

### Phase 2: Fix blocking subprocess calls (2 sites in tool_runtime.py)

Replace `subprocess.run()` with `asyncio.create_subprocess_exec()`:

```python
# OLD (tool_runtime.py:549)
result = subprocess.run(["py", "-3.12", str(_MQ_SCRIPT), "send", ...],
                        capture_output=True, text=True, timeout=10)

# NEW
async def _send_mq_async(...):
    proc = await asyncio.create_subprocess_exec(
        "py", "-3.12", str(_MQ_SCRIPT), "send", ...,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("MQ subprocess timed out")
        return None
    return stdout.decode() if proc.returncode == 0 else None
```

If the calling function is NOT async (tool_runtime functions may be sync), wrap with `asyncio.to_thread()`:
```python
# If the function must remain sync for backward compatibility
result = subprocess.run(..., timeout=10)  # Keep as-is
```

Check whether `_analysis_request` and `_analysis_results` are called from async context. If yes, convert to async subprocess. If no (called only from sync code), leave as-is -- subprocess.run is fine in sync contexts.

### Phase 3: Fix blocking HTTP calls (3 sites)

**voice.py:90** -- Deepgram transcription:
If `transcribe_audio()` is called from an async handler (likely via `run_in_executor`), verify the wrapping. If called directly from async code, convert:
```python
# Option A: wrap in to_thread (minimal change)
audio_text = await asyncio.to_thread(transcribe_audio, audio_bytes)

# Option B: convert function to use aiohttp (more work but better)
async def transcribe_audio_async(audio_bytes):
    async with aiohttp.ClientSession() as session:
        async with session.post(DEEPGRAM_URL, data=audio_bytes,
                               headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            result = await resp.json()
            ...
```

**agent_zero_inference.py:226, 341** -- vLLM API calls:
These are likely already wrapped in `run_in_executor` at the call site (agent_zero_server.py:1413 does `await loop.run_in_executor(None, inference.load_model)`). Verify the wrapping is consistent for `generate()` and `generate_with_tools()` too. If any path calls these without executor wrapping, add `asyncio.to_thread()` at the call site.

### Phase 4: Enable slow callback detection (optional, recommended)

Add to server startup:
```python
if config.debug_mode:
    loop = asyncio.get_running_loop()
    loop.set_debug(True)
    loop.slow_callback_duration = 0.1  # 100ms threshold
```

### Files to modify:
- `agent_zero_server.py` -- 9 replacements (lines 723, 1413, 2412, 2498, 2612, 3447, 4653, 4658, 4659)
- `cognitive_runtime.py` -- 2 replacements (lines 856, 1290)
- `tool_runtime.py` -- 2 replacements (lines 549, 591) -- only if called from async
- `voice.py` -- 1 wrapping verification (line 90)
- `agent_zero_inference.py` -- 2 wrapping verifications (lines 226, 341)

## Test Specifications

### Test 1: get_running_loop replacement
```python
def test_no_deprecated_get_event_loop():
    """Grep agent_zero_server.py and cognitive_runtime.py for get_event_loop.
    Should find zero occurrences outside of comments."""
    import ast, inspect
    # Parse each file's AST, check no call to asyncio.get_event_loop
```

### Test 2: subprocess non-blocking (if converted)
```python
@pytest.mark.asyncio
async def test_mq_send_does_not_block_loop():
    """Verify MQ send completes without blocking event loop > 100ms."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    # Mock subprocess to sleep 0.5s
    # Verify other coroutines can run during that time
```

### Test 3: HTTP call wrapping
```python
def test_voice_transcription_not_blocking():
    """Verify transcribe_audio is called via to_thread or executor."""
    # Check call site in agent_zero_server.py websocket_voice handler
    # Verify it's wrapped in asyncio.to_thread or run_in_executor
```

### Test 4: Slow callback detection (debug mode)
```python
@pytest.mark.asyncio
async def test_slow_callback_detection_enabled_in_debug():
    """When debug_mode=True, loop.slow_callback_duration should be set."""
    # Start app with debug_mode=True
    # Verify loop.get_debug() is True
    # Verify loop.slow_callback_duration <= 0.1
```

### Test 5: Code audit (static analysis)
```python
def test_no_urllib_in_async_functions():
    """Verify no urllib.request.urlopen calls exist in async functions."""
    # Parse AST of voice.py, agent_zero_inference.py
    # For each async def, verify no call to urllib.request.urlopen
```

## Estimated Impact

- **Reliability:** Eliminates Python 3.12 DeprecationWarnings (9 sites) and prevents future RuntimeError when Python drops get_event_loop() entirely
- **Performance:** Unblocks event loop during subprocess calls (up to 10s) and HTTP calls (up to 120s), allowing concurrent WebSocket connections to proceed
- **User experience:** During model loading or Deepgram transcription, other users' chat sessions will no longer freeze
- **Monitoring:** Slow callback detection catches future blocking regressions automatically
