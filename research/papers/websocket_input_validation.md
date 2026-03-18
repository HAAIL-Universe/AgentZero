---
topic: WebSocket Input Validation and Message Security
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T20:00:00Z
---

# WebSocket Input Validation and Message Security

## Problem Statement

Agent Zero's two WebSocket endpoints (`/ws/chat` and `/ws/voice`) accept raw JSON text messages with zero schema validation, no message size limits, and no audio payload size caps. Malformed JSON crashes connections. Oversized audio base64 payloads can cause memory exhaustion (OOM). There is no rate limiting on WebSocket messages. This violates OWASP API8:2023 (Security Misconfiguration) and the OWASP WebSocket Security Cheat Sheet, which mandates: "Validate message structure and content using JSON schemas and allow-lists. Set reasonable size limits (typically 64KB or less)."

## Current State in Agent Zero

### /ws/chat (agent_zero_server.py:3162-3228)

```python
# Line 3187-3188: Raw JSON parse, no validation
data = await websocket.receive_text()
msg = json.loads(data)              # Crashes on malformed JSON
user_content = msg.get("content", "").strip()
```

**Issues:**
1. `json.loads(data)` has no try/except -- malformed JSON propagates to the outer `except Exception` which sends `str(e)` to client (information leakage, line 3222)
2. No validation that `msg` is a dict (could be a list, string, number)
3. No validation that `content` is a string (could be a nested object)
4. No message size check before `json.loads` -- a 1GB string will be parsed before any check
5. No message rate limiting -- client can flood the server

### /ws/voice (agent_zero_server.py:3231-3340+)

```python
# Line 3255: Same raw JSON parse
payload = json.loads(await websocket.receive_text())

# Lines 3295-3302: Audio with no size validation
audio_base64 = payload.get("audio_base64", "")
mime_type = payload.get("mime_type", "audio/webm")
# ... no size check ...
audio_bytes = decode_audio_base64(audio_base64)  # Decodes arbitrarily large payloads
```

**Issues:**
1. Same malformed JSON crash risk as /ws/chat
2. `audio_base64` has no size limit -- attacker can send 500MB base64 string
3. `decode_audio_base64()` allocates ~75% of base64 size in memory (e.g., 375MB for 500MB base64)
4. `mime_type` is not validated against an allowlist -- accepts any string
5. `type` field is not validated against known values (only "ping", "text", "audio" are handled)
6. No rate limiting on audio submissions

### Error Handling (agent_zero_server.py:3220-3224)

```python
except Exception as e:
    try:
        await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
    except Exception:
        pass
```

Sends raw exception strings to the client, potentially leaking internal paths, database errors, or stack traces.

### Starlette/FastAPI WebSocket Layer

Starlette does not impose a `max_size` on WebSocket text messages by default (unlike the `websockets` library which defaults to 1MB). FastAPI inherits this -- there is no built-in message size limit. The application must enforce limits manually.

## Industry Standard / Research Findings

### OWASP WebSocket Security Cheat Sheet
**Source:** https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html

Key mandates:
- **Input validation**: "Validate message structure and content using JSON schemas and allow-lists"
- **Size limits**: "Set reasonable size limits (typically 64KB or less)"
- **Rate limiting**: "Implement rate limiting to prevent message flooding -- 100 messages per minute is a common starting point"
- **Binary validation**: "For binary data, verify file types using magic numbers rather than trusting content-type headers"
- **Error handling**: "Log full errors server-side, send generic messages to client"

### OWASP API4:2023 -- Unrestricted Resource Consumption
**Source:** https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/

The classification states APIs must "limit certain kinds of interactions or requests" to avoid DoS. Unbounded audio upload directly violates this.

### OWASP API8:2023 -- Security Misconfiguration
**Source:** https://owasp.org/API-Security/editions/2023/en/0xa8-security-misconfiguration/

Mandates: "Error messages revealing stack traces or sensitive details" are a vulnerability. Agent Zero's `str(e)` pattern violates this directly.

### Pydantic Discriminated Union Pattern for WebSocket Messages
**Source:** https://docs.pydantic.dev/latest/concepts/unions/

Industry best practice for multi-type WebSocket APIs: define a discriminated union of Pydantic models keyed on a `type` field. This provides:
- Automatic JSON schema validation on every message
- Type-safe message handling with clear error messages
- Validation of nested fields (string length, allowed values, numeric ranges)
- Performance: discriminated unions avoid checking all union members

### Chanx Framework -- WebSocket Message Routing
**Source:** https://chanx.readthedocs.io/en/latest/

Demonstrates production pattern: Literal `action` field for automatic routing with Pydantic validation and AsyncAPI documentation.

### Python websockets Library -- max_size
**Source:** https://github.com/python-websockets/websockets/issues/866

The `websockets` library defaults to 1MB max message size. Starlette/ASGI does not expose this setting, so application-level validation is required before processing.

## Proposed Implementation

### 1. Pydantic Message Models (new file: `agent_zero/ws_messages.py`)

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Annotated, Union

# --- Inbound Messages ---

class ChatMessage(BaseModel):
    """Text chat message from /ws/chat."""
    type: Literal["chat"] = "chat"
    content: str = Field(..., min_length=1, max_length=32_000)

class TextMessage(BaseModel):
    """Text message from /ws/voice."""
    type: Literal["text"]
    content: str = Field(..., min_length=1, max_length=32_000)

class AudioMessage(BaseModel):
    """Audio blob from /ws/voice."""
    type: Literal["audio"]
    audio_base64: str = Field(..., min_length=1, max_length=25_000_000)  # ~18MB decoded
    mime_type: str = Field(default="audio/webm")

    @field_validator("mime_type")
    @classmethod
    def validate_mime(cls, v: str) -> str:
        allowed = {"audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg"}
        if v not in allowed:
            raise ValueError(f"mime_type must be one of {allowed}")
        return v

class PingMessage(BaseModel):
    """Keep-alive ping from /ws/voice."""
    type: Literal["ping"]

# Discriminated union for /ws/voice
VoiceInbound = Annotated[
    Union[TextMessage, AudioMessage, PingMessage],
    Field(discriminator="type"),
]

# --- Constants ---
MAX_RAW_MESSAGE_BYTES = 26_000_000   # 26MB (covers 25MB base64 + JSON overhead)
MAX_TEXT_MESSAGE_BYTES = 64_000       # 64KB for text-only messages
WS_RATE_LIMIT_PER_MINUTE = 60        # Messages per minute per connection
WS_AUDIO_RATE_LIMIT_PER_MINUTE = 10  # Audio messages per minute
```

### 2. Validation Wrapper Function (in `agent_zero/ws_messages.py`)

```python
import json
import time
from collections import deque
from pydantic import ValidationError

class MessageRateLimiter:
    """Sliding-window rate limiter for WebSocket messages."""

    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self._timestamps: deque = deque()

    def check(self) -> bool:
        """Return True if message is allowed, False if rate-limited."""
        now = time.monotonic()
        # Evict timestamps older than 60 seconds
        while self._timestamps and self._timestamps[0] < now - 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_minute:
            return False
        self._timestamps.append(now)
        return True


def validate_ws_message(raw: str, model_class, max_bytes: int = MAX_TEXT_MESSAGE_BYTES):
    """Validate a raw WebSocket message string.

    Returns (parsed_model, None) on success or (None, error_string) on failure.
    """
    # 1. Size check BEFORE parsing
    if len(raw.encode("utf-8", errors="replace")) > max_bytes:
        return None, "Message too large"

    # 2. JSON parse with error handling
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, "Invalid JSON"

    # 3. Type check
    if not isinstance(data, dict):
        return None, "Expected JSON object"

    # 4. Pydantic validation
    try:
        msg = model_class.model_validate(data)
        return msg, None
    except ValidationError as e:
        # Return first error only -- don't leak full schema details
        first = e.errors()[0]
        return None, f"Validation error: {first.get('msg', 'invalid')}"
```

### 3. Update /ws/chat Handler (agent_zero_server.py:3186-3191)

**Replace:**
```python
data = await websocket.receive_text()
msg = json.loads(data)
user_content = msg.get("content", "").strip()
if not user_content:
    continue
```

**With:**
```python
data = await websocket.receive_text()
parsed, err = validate_ws_message(data, ChatMessage, MAX_TEXT_MESSAGE_BYTES)
if err:
    await websocket.send_text(json.dumps({"type": "error", "content": err}))
    continue
if not rate_limiter.check():
    await websocket.send_text(json.dumps({"type": "error", "content": "Rate limit exceeded"}))
    continue
user_content = parsed.content.strip()
if not user_content:
    continue
```

Initialize `rate_limiter = MessageRateLimiter(WS_RATE_LIMIT_PER_MINUTE)` after line 3168.

### 4. Update /ws/voice Handler (agent_zero_server.py:3254-3302)

**Replace:**
```python
payload = json.loads(await websocket.receive_text())
```

**With:**
```python
raw = await websocket.receive_text()
parsed, err = validate_ws_message(raw, VoiceInbound, MAX_RAW_MESSAGE_BYTES)
if err:
    await websocket.send_text(json.dumps({"type": "error", "content": err}))
    continue
if not rate_limiter.check():
    await websocket.send_text(json.dumps({"type": "error", "content": "Rate limit exceeded"}))
    continue
```

Then use `isinstance(parsed, PingMessage)`, `isinstance(parsed, TextMessage)`, `isinstance(parsed, AudioMessage)` for dispatch instead of string key checks.

Initialize both `rate_limiter = MessageRateLimiter(WS_RATE_LIMIT_PER_MINUTE)` and `audio_rate_limiter = MessageRateLimiter(WS_AUDIO_RATE_LIMIT_PER_MINUTE)` after line 3238.

### 5. Fix Error Information Leakage (agent_zero_server.py:3220-3224)

**Replace:**
```python
except Exception as e:
    try:
        await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
```

**With:**
```python
except Exception:
    _log.exception("WebSocket /ws/chat error")
    try:
        await websocket.send_text(json.dumps({"type": "error", "content": "Internal error"}))
```

Apply the same pattern to the /ws/voice error handler.

### 6. Configuration Integration

Add to `agent_zero/config.py` Agent ZeroConfig:
```python
ws_max_text_message_bytes: int = 64_000
ws_max_audio_message_bytes: int = 26_000_000
ws_rate_limit_per_minute: int = 60
ws_audio_rate_limit_per_minute: int = 10
ws_allowed_audio_mimes: list = ["audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg"]
```

## Test Specifications

### test_ws_messages.py

1. **test_valid_chat_message** -- `{"content": "hello"}` parses to ChatMessage with content="hello"
2. **test_chat_message_missing_content** -- `{}` returns validation error
3. **test_chat_message_empty_content** -- `{"content": ""}` returns validation error (min_length=1)
4. **test_chat_message_too_long** -- content with 33,000 chars returns validation error
5. **test_valid_audio_message** -- `{"type": "audio", "audio_base64": "SGVsbG8=", "mime_type": "audio/webm"}` parses correctly
6. **test_audio_invalid_mime** -- `{"type": "audio", "audio_base64": "x", "mime_type": "video/mp4"}` returns validation error
7. **test_audio_oversized** -- audio_base64 with 26M chars returns validation error
8. **test_valid_ping** -- `{"type": "ping"}` parses to PingMessage
9. **test_discriminated_dispatch** -- VoiceInbound correctly dispatches "text", "audio", "ping"
10. **test_unknown_type** -- `{"type": "unknown"}` returns validation error
11. **test_malformed_json** -- `"not{json"` returns "Invalid JSON"
12. **test_json_array** -- `[1,2,3]` returns "Expected JSON object"
13. **test_raw_size_limit** -- 65KB message returns "Message too large"
14. **test_rate_limiter_allows** -- 60 messages in 60s all pass
15. **test_rate_limiter_blocks** -- 61st message in 60s is blocked
16. **test_rate_limiter_window_slides** -- after 60s, new messages are allowed
17. **test_audio_rate_limiter** -- 11th audio message in 60s is blocked
18. **test_error_leakage_prevented** -- exception handler sends "Internal error" not str(e)
19. **test_valid_text_message** -- `{"type": "text", "content": "hi"}` parses to TextMessage
20. **test_chat_message_type_default** -- ChatMessage defaults type to "chat"

## Estimated Impact

- **Security**: Eliminates OOM attack vector via oversized audio (critical)
- **Security**: Eliminates information leakage via exception strings (medium)
- **Reliability**: Prevents malformed JSON from crashing connections (high)
- **Performance**: Rate limiting prevents message flooding DoS (high)
- **Code quality**: Pydantic models make message contract explicit and self-documenting
- **Maintainability**: New message types are added by extending the discriminated union

## References

1. OWASP WebSocket Security Cheat Sheet -- https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html
2. OWASP API4:2023 Unrestricted Resource Consumption -- https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/
3. OWASP API8:2023 Security Misconfiguration -- https://owasp.org/API-Security/editions/2023/en/0xa8-security-misconfiguration/
4. Pydantic Discriminated Unions -- https://docs.pydantic.dev/latest/concepts/unions/
5. Chanx WebSocket Framework (discriminated routing) -- https://chanx.readthedocs.io/en/latest/
6. Python websockets max_size discussion -- https://github.com/python-websockets/websockets/issues/866
