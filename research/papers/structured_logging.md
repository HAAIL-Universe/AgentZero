---
topic: Structured Logging Replacement
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T23:00:00Z
---

# Structured Logging Replacement

## Problem Statement

Agent Zero uses 15+ `print()` statements in `agent_zero_server.py` for operational logging and 14 `except: pass` blocks that silently swallow errors. This creates three problems:

1. **No severity levels.** `print("[Agent Zero] Shadow save failed")` and `print("[DB] Schema migrations complete")` look identical. Operators can't filter errors from info.
2. **No structured data.** Plain text requires regex to extract user_id, latency, or error type. Log aggregation tools (Datadog, Loki, CloudWatch) can't query unstructured print output.
3. **Silent failures.** 14 `except: pass` blocks hide errors that could indicate systematic problems (NeonDB intermittent failures, model timeouts, WebSocket drops).

## Current State in Agent Zero

### Print Statements in agent_zero_server.py

**File:** `agent_zero/agent_zero_server.py`

Key print statements (15+ instances):
- Line 118: `print(f"[Agent Zero] Starting...")`
- Line 121: `print(f"[DB] Pool ready")`
- Line 123: `print(f"[DB] Migrations done")`
- Line 1213: `print(f"[Agent Zero] Shadow load failed: {e}")`
- Line 1268: `print(f"[Agent Zero] Shadow save failed: {e}")`
- Line 1317: `print(f"[Agent Zero] Tool execution: {tool_name}")`
- Line 1360: `print(f"[Agent Zero] Cognitive deliberation: {route}")`
- Line 1881: `print(f"[Agent Zero] Intervention outcome sync failed for {_iid}: {e}")`
- Line 2534: `print(f"[Agent Zero] WebSocket connected: {user_id}")`

### Silent Exception Handlers

14 instances of `except Exception: pass` or `except: pass`:
- Lines 348, 1847, 2104, 2983: Failed event/artifact operations
- Lines 3049, 3054, 3173, 3178: WebSocket disconnect handling
- `behavioral_insights.py` line 489: Insight generation failure
- `cognitive_agents.py` line 438: JSON decode failure

### Existing Logging Usage

- `consolidator.py` line 29: `_quality_logger = logging.getLogger("agent_zero.quality_gate")` -- already uses Python logging for quality gate warnings
- This proves the pattern works in the codebase; it just hasn't been applied broadly

## Industry Standard / Research Findings

### 1. Structlog for Production Python (Schlawack, 2013-2025)

structlog has been used in production at every scale since 2013. It provides async-native logging via `ainfo()`, `aerror()` etc., contextvars integration for request-scoped context binding, and JSON rendering for machine-parseable output. Key processors: `TimeStamper`, `StackInfoRenderer`, `format_exc_info`.

**URL:** https://pypi.org/project/structlog/

### 2. Python stdlib logging with JSON Formatter (Thakur, 2026)

For minimal-dependency environments, Python's built-in `logging` module with a custom JSON formatter achieves 80% of structlog's benefit. A 20-line `JsonFormatter` class produces structured JSON output without any pip install. This is the recommended approach when adding dependencies is a concern.

**URL:** https://medium.com/@dhruvshirar/structured-logging-in-python-a-practical-guide-for-production-systems-9659f461fa93

### 3. Contextvars for Request Tracing (Python 3.7+)

Python's `contextvars` module (PEP 567) provides async-safe context for binding user_id, session_id, and request_id to all log entries within a request scope. This eliminates the need to pass identifiers through every function call.

**URL:** https://docs.python.org/3/library/contextvars.html

### 4. FastAPI Structured Logging Integration (OneUptime, 2026)

Production FastAPI applications use middleware to inject correlation IDs into the logging context. Every log entry within a request automatically includes the request_id, user_id, and latency. This enables tracing a single user interaction across all system components.

**URL:** https://oneuptime.com/blog/post/2026-02-02-fastapi-structured-logging/view

### 5. SigNoz Structlog Guide (2025)

Recommended processor pipeline: `TimeStamper(fmt="ISO")` -> `add_log_level` -> `StackInfoRenderer()` -> `format_exc_info` -> `JSONRenderer()`. Bind session context at middleware level; use `logger.bind(user_id=..., session_id=...)` for per-request context.

**URL:** https://signoz.io/guides/structlog/

### 6. Async Logging Best Practices (UptimeRobot, 2025)

In async applications, avoid blocking logging calls in the event loop. Queue-based handlers (`logging.handlers.QueueHandler`) offload I/O from the event loop. For structlog, the async methods (`ainfo()`, `aerror()`) handle this automatically.

**URL:** https://uptimerobot.com/knowledge-hub/logging/python-logging-explained/

## Proposed Implementation

### Strategy: Python stdlib logging with JSON formatter (zero new dependencies)

Use Python's built-in `logging` module with a custom JSON formatter. This avoids adding structlog as a dependency while providing structured, severity-leveled, queryable logs. The quality_gate logger in consolidator.py already proves this pattern works.

### Step 1: Create logging configuration module

**New file:** `agent_zero/logging_config.py`

```python
"""Structured JSON logging for Agent Zero Agent Zero.

Replaces print() statements with severity-leveled, structured JSON logs.
Uses Python stdlib logging -- no external dependencies.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Request-scoped context
_request_user_id: ContextVar[str] = ContextVar("user_id", default="")
_request_session_id: ContextVar[str] = ContextVar("session_id", default="")


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Add request context if available
        uid = _request_user_id.get("")
        sid = _request_session_id.get("")
        if uid:
            entry["user_id"] = uid
        if sid:
            entry["session_id"] = sid
        # Add extra fields
        if hasattr(record, "data") and record.data:
            entry["data"] = record.data
        # Add exception info
        if record.exc_info and record.exc_info[0]:
            entry["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO"):
    """Configure structured JSON logging for all Agent Zero modules."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger("agent_zero")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False

    return root


def bind_request_context(user_id: str = "", session_id: str = ""):
    """Bind user/session context for the current async task."""
    if user_id:
        _request_user_id.set(user_id)
    if session_id:
        _request_session_id.set(session_id)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the agent_zero hierarchy."""
    return logging.getLogger(f"agent_zero.{name}")
```

### Step 2: Replace print statements in agent_zero_server.py

**File:** `agent_zero/agent_zero_server.py`

Add at top (after imports):
```python
from logging_config import setup_logging, get_logger, bind_request_context

_log = get_logger("server")
```

Replace print statements:
```python
# Before:
print(f"[Agent Zero] Starting...")
# After:
_log.info("Server starting")

# Before:
print(f"[DB] Pool ready")
# After:
_log.info("Database pool ready")

# Before:
print(f"[Agent Zero] Shadow load failed: {e}")
# After:
_log.error("Shadow load failed", exc_info=True)

# Before:
print(f"[Agent Zero] Tool execution: {tool_name}")
# After:
_log.info("Tool execution", extra={"data": {"tool": tool_name}})

# Before:
print(f"[Agent Zero] Intervention outcome sync failed for {_iid}: {e}")
# After:
_log.warning("Intervention outcome sync failed", extra={"data": {"intervention_id": _iid}}, exc_info=True)
```

### Step 3: Replace silent except+pass with logged warnings

**File:** `agent_zero/agent_zero_server.py`

```python
# Before (line 348):
except Exception:
    pass

# After:
except Exception:
    _log.debug("Stage message append failed", exc_info=True)

# Before (line 1847):
except Exception:
    pass

# After:
except Exception:
    _log.debug("Reasoning event emit failed", exc_info=True)
```

Use `debug` level for non-critical failures (event append, artifact storage). Use `warning` for failures that affect data consistency (outcome sync, shadow save). Use `error` for failures that affect user experience.

### Step 4: Add request context binding in WebSocket handler

**File:** `agent_zero/agent_zero_server.py`

In the WebSocket message handler, after user authentication:
```python
bind_request_context(user_id=str(user["user_id"]), session_id=session_id)
```

This ensures all subsequent log entries within that request include user_id and session_id automatically via contextvars.

### Step 5: Add loggers to other modules

**File:** `agent_zero/database.py`:
```python
from logging_config import get_logger
_log = get_logger("database")
# Replace: print("[DB] Schema migrations complete")
# With: _log.info("Schema migrations complete")
```

**File:** `agent_zero/cognitive_agents.py` (line 438):
```python
from logging_config import get_logger
_log = get_logger("agents")
# Replace: except: pass (JSON decode)
# With: except Exception: _log.debug("Agent JSON decode failed", exc_info=True)
```

### Step 6: Initialize logging at startup

**File:** `agent_zero/agent_zero_server.py`, in the startup function:
```python
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
```

### What NOT to Change

- **consolidator.py** `_quality_logger` -- already uses Python logging correctly; keep as-is
- **Test files** -- print statements in tests are fine (test output, not production logging)
- **Log rotation** -- RunPod containers use stdout → container log aggregation; no file rotation needed

## Test Specifications

### Test 1: JSON formatter produces valid JSON
```python
def test_json_formatter_output():
    """Log entries should be valid JSON with required fields."""
    import json
    from logging_config import JsonFormatter
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
    output = formatter.format(record)
    parsed = json.loads(output)
    assert parsed["level"] == "INFO"
    assert parsed["msg"] == "hello"
    assert "ts" in parsed
```

### Test 2: Context binding propagates
```python
def test_context_binding():
    """Bound user_id should appear in log output."""
    from logging_config import bind_request_context, JsonFormatter
    bind_request_context(user_id="u123", session_id="s456")
    formatter = JsonFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
    output = json.loads(formatter.format(record))
    assert output["user_id"] == "u123"
    assert output["session_id"] == "s456"
```

### Test 3: Exception info included
```python
def test_exception_logging():
    """Exception info should be structured in the log entry."""
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()
    record = logging.LogRecord("test", logging.ERROR, "", 0, "fail", (), exc_info)
    output = json.loads(JsonFormatter().format(record))
    assert output["error"]["type"] == "ValueError"
    assert "test error" in output["error"]["message"]
```

### Test 4: get_logger returns namespaced logger
```python
def test_logger_naming():
    """Loggers should be namespaced under agent_zero."""
    from logging_config import get_logger
    log = get_logger("server")
    assert log.name == "agent_zero.server"
```

### Test 5: Silent handlers now log at debug level
```python
def test_previously_silent_handlers_log():
    """Former except+pass blocks should now produce debug-level entries."""
    # Verify by checking that debug handler captures output
    # when a known silent exception path is triggered
```

## Estimated Impact

1. **Severity filtering.** Operators can filter `level=ERROR` to see only failures, `level=WARNING` for data consistency issues, `level=INFO` for operational events.

2. **Structured queries.** Log aggregators can query by user_id, session_id, tool_name, error_type without regex. Example: "show all errors for user X in the last hour."

3. **Silent failure detection.** 14 formerly invisible exception paths now produce debug-level entries. Setting `LOG_LEVEL=DEBUG` reveals hidden failures.

4. **Request tracing.** Contextvars bind user_id and session_id to every log entry within a request, enabling end-to-end request tracing.

5. **Zero new dependencies.** Uses Python's built-in `logging` module + `contextvars`. No pip install needed.

6. **Backward compatible.** No behavioral changes -- just visibility improvements. All print output becomes structured JSON to stdout.
