---
topic: Error Message Information Leakage Prevention
status: implemented
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T00:00:00Z
---

# Error Message Information Leakage Prevention

## Problem Statement

agent_zero_server.py sends raw exception strings (`str(e)`) to clients via WebSocket and HTTP in **9 locations**. These expose internal details: database connection strings, file paths, GPU/CUDA errors, model locations, SQL query failures, and library version info. This violates OWASP A10:2025 (Mishandling of Exceptional Conditions) and A02:2025 (Security Misconfiguration), specifically CWE-209 (Error Message Containing Sensitive Information) and CWE-550 (Server Generated Error Message Containing Sensitive Information).

## Current State in Agent Zero

**File:** `agent_zero/agent_zero_server.py` (~3,900 lines)

### Critical Exposures (WebSocket -- direct to clients):

| Line | Context | Code Pattern | What Leaks |
|------|---------|-------------|------------|
| 2505 | Tool call error | `f"[Tool call error: {e}]"` sent via WebSocket | DB queries, file paths, API errors |
| 2691 | LLM generation error | `f"[Generation error: {e}]"` sent via WebSocket | CUDA/GPU errors, model paths, vLLM internals |
| 3358 | Text WS catch-all | `{"type": "error", "content": str(e)}` | Any unhandled exception from entire text flow |
| 3445 | Voice transcription | `{"type": "error", "content": str(exc)}` | Deepgram API errors, rate limits |
| 3489 | Voice WS catch-all | `{"type": "error", "content": str(e)}` | Any unhandled exception from voice flow |

### HTTP Exposures:

| Line | Endpoint | Code Pattern | What Leaks |
|------|----------|-------------|------------|
| 728 | POST /transcribe | `HTTPException(422, detail=str(exc))` | Deepgram API errors |
| 1356 | GET /health | `{"status": "down", "error": str(e)}` | DB connection strings, host/port |
| 1380 | GET /health | `{"status": "down", "error": str(e)}` | Model paths, GPU device info |
| 941 | DELETE /users/me/all-data | `f"skipped: {e}"` | Table names, SQL errors |

### Existing Partial Sanitization:

**tool_runtime.py line 112** has a partial pattern:
```python
error_msg = str(exc)[:200]  # cap length
return _tool_error(tool_name, f"Tool execution failed ({error_type}: {error_msg})")
```
This caps length but still leaks the first 200 chars of raw exception text.

**auth.py line 66**: `HTTPException(401, detail=f"Invalid token: {e}")` -- leaks JWT library error messages.

### Attack Surface:

An attacker can learn:
- **Infrastructure**: DB host/port from connection errors, GPU model from CUDA errors
- **File paths**: Model directories, cache locations, installation paths
- **Capabilities**: Which libraries/tools are installed from exception class names
- **API details**: Deepgram rate limit info reveals usage patterns

## Industry Standard / Research Findings

### 1. OWASP A10:2025 -- Mishandling of Exceptional Conditions (NEW)

This is a brand-new OWASP Top 10 category for 2025, reflecting the severity of error handling failures. It covers 24 CWEs including:
- **CWE-209**: Generation of Error Message Containing Sensitive Information
- **CWE-550**: Server Generated Error Message Containing Sensitive Information
- **CWE-636**: Not Failing Securely (Failing Open)

Prevention requirements:
1. Catch every possible system error at the place where it occurs
2. Fail securely -- roll back transactions on error
3. Use centralized, consistent error handling organization-wide
4. Never expose technical details in client responses
5. Conduct threat modeling and code reviews for error paths

Source: https://owasp.org/Top10/2025/A10_2025-Mishandling_of_Exceptional_Conditions/

### 2. RFC 9457 -- Problem Details for HTTP APIs

The IETF standard (supersedes RFC 7807) defines a structured error response format:

```json
{
  "type": "https://agent_zero.app/errors/generation-failed",
  "title": "Generation Failed",
  "status": 500,
  "detail": "The response could not be generated. Please try again.",
  "instance": "/ws/text/session-123"
}
```

Five standard fields: `type` (URI identifying problem type), `title` (short summary), `status` (HTTP code), `detail` (human-readable, must NOT contain sensitive info), `instance` (URI for specific occurrence).

Source: https://www.rfc-editor.org/rfc/rfc9457.html

### 3. OWASP Error Handling Cheat Sheet

Key principles:
- **Show users**: Generic, non-descriptive messages only. Example: `"An error occurred, please retry."`
- **Log server-side**: Full exception object, stack trace, request context, user ID, timestamp
- **Separate error channels**: Client-facing messages are independent from server logs
- Include `X-ERROR: true` header to signal error responses to client applications

Source: https://cheatsheetseries.owasp.org/cheatsheets/Error_Handling_Cheat_Sheet.html

### 4. OWASP A02:2025 -- Security Misconfiguration

Surged from #5 to #2 in 2025, now affecting 3% of tested apps. Explicitly lists "error handling that reveals stack traces or overly informative error messages" as a misconfiguration. Prevention: "Define and enforce all API response payload schemas including error responses."

Source: https://owasp.org/Top10/2025/A02_2025-Security_Misconfiguration/

### 5. Enterprise Python Error Handling Patterns

Augment Code's enterprise guide recommends:
1. **Exception taxonomy**: Domain-specific hierarchies (`Agent ZeroError -> GenerationError, ToolError, TranscriptionError`)
2. **Middleware error translation**: Catch infrastructure errors at boundaries, translate to domain exceptions
3. **Structured logging**: Python 3.12+ `add_note()` enriches exceptions with context

Source: https://www.augmentcode.com/guides/python-error-handling-10-enterprise-grade-tactics

### 6. PortSwigger Information Disclosure Research

Information disclosure via error messages is a top web security vulnerability. Verbose error messages help attackers understand application internals and craft targeted exploits.

Source: https://portswigger.net/web-security/information-disclosure

## Proposed Implementation

### Step 1: Create Error Sanitization Module (agent_zero/error_responses.py)

New file (~60 lines):

```python
"""Centralized error response sanitization for Agent Zero.

All client-facing error messages go through this module.
Server-side logging uses the structured logger directly.
"""
import logging
import uuid

logger = logging.getLogger("agent_zero.errors")


# Error codes map to generic client messages
_CLIENT_MESSAGES = {
    "generation_error": "I'm having trouble generating a response. Please try again.",
    "tool_error": "A tool encountered an issue. Please try a different approach.",
    "transcription_error": "Voice transcription failed. Please try speaking again.",
    "connection_error": "A connection issue occurred. Please try again shortly.",
    "auth_error": "Authentication failed. Please log in again.",
    "internal_error": "An unexpected error occurred. Please try again.",
    "data_error": "There was an issue processing your data.",
}


def safe_error(error_code: str, exc: Exception, *, context: str = "") -> dict:
    """Return a client-safe error dict and log the full exception server-side.

    Args:
        error_code: Key into _CLIENT_MESSAGES (e.g., "generation_error")
        exc: The actual exception (logged, never sent to client)
        context: Additional context for server-side log

    Returns:
        dict with "type": "error" and "content": generic message
    """
    incident_id = uuid.uuid4().hex[:12]
    client_msg = _CLIENT_MESSAGES.get(error_code, _CLIENT_MESSAGES["internal_error"])

    # Full details logged server-side only
    logger.error(
        "Client error [%s] %s: %s | context=%s",
        incident_id,
        error_code,
        exc,
        context,
        exc_info=True,
    )

    return {
        "type": "error",
        "content": client_msg,
        "incident_id": incident_id,
    }


def safe_http_error(error_code: str, exc: Exception, status: int = 500, context: str = "") -> dict:
    """Return RFC 9457-style problem details for HTTP responses."""
    incident_id = uuid.uuid4().hex[:12]
    client_msg = _CLIENT_MESSAGES.get(error_code, _CLIENT_MESSAGES["internal_error"])

    logger.error(
        "HTTP error [%s] %s: %s | context=%s",
        incident_id,
        error_code,
        exc,
        context,
        exc_info=True,
    )

    return {
        "type": f"errors/{error_code}",
        "title": client_msg,
        "status": status,
        "instance": incident_id,
    }
```

### Step 2: Replace WebSocket Error Exposures (agent_zero_server.py)

**Line 2505** (tool call error):
```python
# Before:
full_response = f"[Tool call error: {e}]"
await websocket.send_text(json.dumps({"type": "token", "content": full_response}))

# After:
from agent_zero.error_responses import safe_error
err = safe_error("tool_error", e, context=f"tool={tool_name}")
await websocket.send_text(json.dumps(err))
```

**Line 2691** (generation error):
```python
# Before:
full_response = f"[Generation error: {e}]"
await websocket.send_text(json.dumps({"type": "token", "content": full_response}))

# After:
err = safe_error("generation_error", e, context="streaming_generation")
await websocket.send_text(json.dumps(err))
```

**Line 3358** (text WS catch-all):
```python
# Before:
await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))

# After:
err = safe_error("internal_error", e, context="ws_text_handler")
await websocket.send_text(json.dumps(err))
```

**Line 3445** (voice transcription):
```python
# Before:
await websocket.send_text(json.dumps({"type": "error", "content": str(exc)}))

# After:
err = safe_error("transcription_error", exc, context="deepgram_transcription")
await websocket.send_text(json.dumps(err))
```

**Line 3489** (voice WS catch-all):
```python
# Before:
await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))

# After:
err = safe_error("internal_error", e, context="ws_voice_handler")
await websocket.send_text(json.dumps(err))
```

### Step 3: Replace HTTP Endpoint Exposures (agent_zero_server.py)

**Line 728** (POST /transcribe):
```python
# Before:
raise HTTPException(status_code=422, detail=str(exc))

# After:
from agent_zero.error_responses import safe_http_error
err = safe_http_error("transcription_error", exc, status=422, context="transcribe_endpoint")
raise HTTPException(status_code=422, detail=err["title"])
```

**Lines 1356, 1380** (GET /health):
```python
# Before:
checks["database"] = {"status": "down", "error": str(e)}
checks["model"] = {"status": "down", "error": str(e)}

# After:
checks["database"] = {"status": "down"}  # No error details in response
logger.error("Health check database failure: %s", e, exc_info=True)
checks["model"] = {"status": "down"}
logger.error("Health check model failure: %s", e, exc_info=True)
```

**Line 941** (DELETE /users/me/all-data):
```python
# Before:
deleted_counts[table] = f"skipped: {e}"

# After:
deleted_counts[table] = "skipped"
logger.error("User data clear failed for table %s: %s", table, e, exc_info=True)
```

### Step 4: Fix auth.py (line 66)

```python
# Before:
raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

# After:
logger.warning("Invalid JWT token: %s", e)
raise HTTPException(status_code=401, detail="Invalid or expired token")
```

### Step 5: Update Frontend Error Display (agent_zero.html)

The frontend likely displays error content directly. Update the error message handler to show generic messages and optionally include the incident_id for support:

```javascript
// In WebSocket message handler, when type === "error":
if (data.type === "error") {
    displayError(data.content);  // Already generic from server
    if (data.incident_id) {
        console.debug("Error incident:", data.incident_id);  // For debugging
    }
}
```

## Test Specifications

```python
def test_safe_error_returns_generic_message():
    """safe_error() returns client-safe message, not the raw exception."""
    err = safe_error("generation_error", ValueError("CUDA OOM at /dev/nvidia0"))
    assert "CUDA" not in err["content"]
    assert "nvidia" not in err["content"]
    assert err["content"] == "I'm having trouble generating a response. Please try again."
    assert "incident_id" in err

def test_safe_error_logs_full_exception(caplog):
    """safe_error() logs the full exception details server-side."""
    with caplog.at_level(logging.ERROR):
        safe_error("tool_error", RuntimeError("SQL: SELECT * FROM users WHERE..."))
    assert "SQL: SELECT * FROM users" in caplog.text

def test_safe_error_unknown_code_returns_internal():
    """Unknown error codes fall back to internal_error message."""
    err = safe_error("unknown_code", Exception("test"))
    assert err["content"] == "An unexpected error occurred. Please try again."

def test_safe_http_error_rfc9457_format():
    """safe_http_error() returns RFC 9457-style problem details."""
    err = safe_http_error("auth_error", Exception("test"), status=401)
    assert "type" in err
    assert "title" in err
    assert "status" in err
    assert err["status"] == 401
    assert "instance" in err

def test_health_endpoint_no_error_details():
    """GET /health returns status 'down' without error message details."""
    # Mock database failure
    # Assert: response contains {"status": "down"} without "error" key

def test_websocket_tool_error_sanitized():
    """WebSocket tool error sends generic message, not raw exception."""
    # Trigger tool call that raises exception
    # Assert: WebSocket message content does not contain exception text

def test_websocket_generation_error_sanitized():
    """WebSocket generation error sends generic message."""
    # Trigger generation failure
    # Assert: message content is generic

def test_incident_id_unique():
    """Each error generates a unique incident ID for correlation."""
    err1 = safe_error("internal_error", Exception("a"))
    err2 = safe_error("internal_error", Exception("b"))
    assert err1["incident_id"] != err2["incident_id"]

def test_auth_error_no_token_details():
    """JWT auth errors do not reveal token structure."""
    # Send invalid JWT to protected endpoint
    # Assert: response detail is "Invalid or expired token", not raw jwt error
```

## Estimated Impact

- **Security**: Eliminates 9 information leakage points. Prevents infrastructure enumeration, path disclosure, and capability mapping attacks.
- **OWASP compliance**: Addresses A10:2025 (Mishandling of Exceptional Conditions) and A02:2025 (Security Misconfiguration).
- **Debugging**: Incident IDs enable correlating client-reported errors with server-side logs without exposing internals.
- **User experience**: Generic error messages are friendlier than raw Python tracebacks appearing in chat.
- **Operational**: Health endpoint no longer leaks DB connection details or model paths to unauthenticated callers.
