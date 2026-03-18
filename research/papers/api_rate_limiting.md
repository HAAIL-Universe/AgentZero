---
topic: API Rate Limiting and Abuse Prevention
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T14:00:00Z
---

# API Rate Limiting and Abuse Prevention

## Problem Statement

Agent Zero exposes 37 HTTP endpoints and 2 WebSocket endpoints with **zero rate limiting**. Three publicly accessible endpoints -- `/auth/register` (line 515), `/auth/login` (line 522), and `/api/load-model` (line 1173) -- accept unlimited requests from any IP address. This violates OWASP API Security Top 10 item API4:2023 "Unrestricted Resource Consumption" and OWASP LLM Top 10 item LLM10:2025 "Unbounded Consumption".

**Specific risks:**
- Registration flood: unlimited account creation (no CAPTCHA, no rate limit)
- Brute force login: unlimited password guessing attempts
- Model loading DoS: `/api/load-model` triggers expensive GPU inference initialization
- Inference abuse: authenticated users can submit unlimited conversation turns via WebSocket, each triggering multi-agent LLM inference (Qwen3-235B)
- Cost explosion: each inference call consumes GPU compute on RunPod; unbounded requests = unbounded cost

## Current State in Agent Zero

### Unauthenticated endpoints (no protection):
- `POST /auth/register` (agent_zero_server.py:515) -- creates DB user, no limit
- `POST /auth/login` (agent_zero_server.py:522) -- JWT generation, no limit
- `GET /api/status` (agent_zero_server.py:1139) -- read-only, low risk
- `POST /api/load-model` (agent_zero_server.py:1173) -- expensive GPU operation, no limit

### Authenticated endpoints (JWT required but no rate limit):
- 33 REST endpoints (agent_zero_server.py:529-1131) all use `Depends(get_current_user)` but accept unlimited requests
- `WS /ws/chat` (agent_zero_server.py:3066) -- each message triggers full cognitive pipeline
- `WS /ws/voice` (agent_zero_server.py:3127) -- each audio blob triggers transcription + inference

### Existing infrastructure:
- `resilience.py` has CircuitBreaker but it protects **outbound** calls (DB, external APIs), not inbound requests
- `config.py` (Agent ZeroConfig) provides centralized configuration -- rate limit params should live here
- FastAPI middleware stack already has CORSMiddleware (agent_zero_server.py:157)

## Industry Standard / Research Findings

### OWASP API4:2023 -- Unrestricted Resource Consumption
OWASP recommends: "Implement a limit on how often a client can interact with the API within a defined timeframe" with endpoint-specific policies where "some API endpoints might require stricter policies" (OWASP, 2023).
**URL:** https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/

### OWASP LLM10:2025 -- Unbounded Consumption
For LLM applications specifically: "Apply rate limiting and user quotas to restrict the number of requests a single source entity can make in a given time period. Dynamic resource management: Monitor and manage resource allocation dynamically to prevent any single user or request from consuming excessive resources" (OWASP GenAI, 2025).
**URL:** https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/

### Token Bucket Algorithm
Token bucket is the standard for API rate limiting. Tokens are added at a fixed rate; each request consumes one token. Burst capacity equals bucket size. "Token bucket feels API-friendly because occasional spikes are okay as long as the average stays in range" (Talamantes, 2025).
**URL:** https://blog.compliiant.io/api-defense-with-rate-limiting-using-fastapi-and-token-buckets-0f5206fc5029

### SlowAPI for FastAPI
SlowAPI is the de-facto rate limiting middleware for Starlette/FastAPI, adapted from flask-limiter. Supports in-memory and Redis backends, per-IP and per-user identification, async-compatible with sub-ms overhead (Laurent S., 2024).
**URL:** https://github.com/laurentS/slowapi

### FastAPI Advanced Rate Limiter
Production-ready library with 6 algorithms (Token Bucket, Leaky Bucket, Fixed Window, Sliding Window, Sliding Window Log, Queue-based) and both in-memory and Redis backends (PyPI, 2025).
**URL:** https://pypi.org/project/fastapi-advanced-rate-limiter/

### Tiered Rate Limiting for AI APIs
Best practice for AI inference: separate limits for auth endpoints (strict, per-IP), CRUD endpoints (moderate, per-user), and inference endpoints (aggressive, per-user with cost weighting). "Hybrid approaches rule: Token bucket for intra-window bursts, sliding window for inter-window fairness" (Johal, 2025).
**URL:** https://johal.in/api-rate-limiting-in-python-fastapi-middleware-for-scalable-endpoints-2025/

## Proposed Implementation

### New file: `agent_zero/rate_limiter.py`

A zero-dependency in-memory rate limiter using token bucket algorithm. No external libraries needed -- Agent Zero is single-instance on RunPod, so in-memory is appropriate. The implementation should be async-safe using `asyncio.Lock`.

```python
"""In-memory token bucket rate limiter for Agent Zero API endpoints.

No external dependencies. Single-instance deployment on RunPod means
in-memory storage is appropriate (no Redis needed).
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import config


@dataclass
class TokenBucket:
    """Token bucket for a single key (IP or user_id)."""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimiter:
    """Thread-safe rate limiter with per-key token buckets.

    Supports tiered limits: auth (per-IP), api (per-user), inference (per-user).
    Automatically evicts stale buckets to prevent memory leak.
    """

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300  # 5 minutes

    async def check(self, key: str, capacity: int, refill_rate: float) -> bool:
        """Check if request is allowed for the given key.

        Args:
            key: identifier (e.g., "auth:192.168.1.1" or "api:user-uuid")
            capacity: max burst size
            refill_rate: tokens per second

        Returns:
            True if request is allowed, False if rate limited.
        """
        async with self._lock:
            self._maybe_cleanup()
            if key not in self._buckets:
                bucket = TokenBucket(capacity=capacity, refill_rate=refill_rate)
                bucket.tokens = capacity  # start full
                self._buckets[key] = bucket
            return self._buckets[key].consume()

    def _maybe_cleanup(self):
        """Evict buckets not used in 10 minutes to prevent memory growth."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        cutoff = now - 600
        stale = [k for k, b in self._buckets.items() if b.last_refill < cutoff]
        for k in stale:
            del self._buckets[k]


# Singleton
limiter = RateLimiter()
```

### New config fields in `agent_zero/config.py` (add to Agent ZeroConfig class)

```python
# --- Rate Limiting ---
rate_limit_auth_capacity: int = Field(
    default=10, ge=1, le=100,
    description="Max burst requests for auth endpoints (per IP)"
)
rate_limit_auth_refill: float = Field(
    default=0.1, ge=0.01, le=10.0,
    description="Auth endpoint token refill rate (tokens/sec, 0.1 = 6/min)"
)
rate_limit_api_capacity: int = Field(
    default=60, ge=10, le=1000,
    description="Max burst requests for authenticated API endpoints (per user)"
)
rate_limit_api_refill: float = Field(
    default=2.0, ge=0.1, le=100.0,
    description="API endpoint token refill rate (tokens/sec)"
)
rate_limit_inference_capacity: int = Field(
    default=5, ge=1, le=50,
    description="Max burst requests for inference/WebSocket turns (per user)"
)
rate_limit_inference_refill: float = Field(
    default=0.2, ge=0.01, le=5.0,
    description="Inference token refill rate (tokens/sec, 0.2 = 12/min)"
)
rate_limit_model_load_capacity: int = Field(
    default=2, ge=1, le=10,
    description="Max burst requests for model loading (per IP)"
)
rate_limit_model_load_refill: float = Field(
    default=0.017, ge=0.001, le=1.0,
    description="Model load token refill rate (0.017 = ~1/min)"
)
```

### Integration into `agent_zero/agent_zero_server.py`

**Step 1:** Import at top (after resilience import, ~line 121):
```python
from rate_limiter import limiter
```

**Step 2:** Add rate limit helper function (after imports, ~line 122):
```python
async def _check_rate_limit_ip(request, tier: str = "auth") -> None:
    """Raise 429 if IP exceeds rate limit for the given tier."""
    ip = request.client.host if request.client else "unknown"
    capacity = getattr(_cfg, f"rate_limit_{tier}_capacity")
    refill = getattr(_cfg, f"rate_limit_{tier}_refill")
    key = f"{tier}:{ip}"
    if not await limiter.check(key, capacity, refill):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")


async def _check_rate_limit_user(user_id: str, tier: str = "api") -> None:
    """Raise 429 if user exceeds rate limit for the given tier."""
    capacity = getattr(_cfg, f"rate_limit_{tier}_capacity")
    refill = getattr(_cfg, f"rate_limit_{tier}_refill")
    key = f"{tier}:{user_id}"
    if not await limiter.check(key, capacity, refill):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")
```

**Step 3:** Apply to unauthenticated endpoints:

```python
# agent_zero_server.py:515-519
@app.post("/auth/register")
async def api_register(req: AuthRequest, request: Request):
    await _check_rate_limit_ip(request, "auth")
    result = await register_user(req.email, req.password, req.display_name)
    return JSONResponse(result)

# agent_zero_server.py:522-526
@app.post("/auth/login")
async def api_login(req: AuthRequest, request: Request):
    await _check_rate_limit_ip(request, "auth")
    result = await login_user(req.email, req.password)
    return JSONResponse(result)

# agent_zero_server.py:1173-1186
@app.post("/api/load-model")
async def load_model(request: Request):
    await _check_rate_limit_ip(request, "model_load")
    # ... existing code
```

**Step 4:** Add `Request` import (already imported via FastAPI) and apply inference rate limiting in WebSocket handlers:

```python
# agent_zero_server.py:3087 (inside websocket_chat while loop, before json.loads)
# Add after line 3088:
user_id = user["user_id"] if user else (websocket.client.host if websocket.client else "anon")
if not await limiter.check(
    f"inference:{user_id}",
    _cfg.rate_limit_inference_capacity,
    _cfg.rate_limit_inference_refill,
):
    await websocket.send_text(json.dumps({
        "type": "error",
        "content": "Rate limit exceeded. Please wait before sending another message."
    }))
    continue
```

Apply the same pattern in `websocket_voice` after audio decoding (line ~3180).

**Step 5:** Add `Request` to the function signatures that need it. FastAPI auto-injects `Request` when it appears as a parameter.

### Rate limit tiers summary

| Tier | Scope | Capacity | Refill | Effective Rate | Rationale |
|------|-------|----------|--------|---------------|-----------|
| auth | per-IP | 10 | 0.1/s | 6/min sustained | Brute force prevention (OWASP) |
| api | per-user | 60 | 2.0/s | 120/min sustained | Normal CRUD, generous burst |
| inference | per-user | 5 | 0.2/s | 12/min sustained | Each turn costs GPU compute |
| model_load | per-IP | 2 | 0.017/s | ~1/min sustained | Expensive GPU initialization |

## Test Specifications

### File: `agent_zero/test_rate_limiter.py`

```
test_token_bucket_basic -- Create bucket(capacity=3, refill=1.0). Consume 3 times -> all True. 4th -> False.
test_token_bucket_refill -- Consume all tokens, wait 1.1s, consume again -> True (refilled).
test_rate_limiter_different_keys -- Same capacity, different keys. Each gets independent bucket.
test_rate_limiter_same_key -- Exhaust key "a", verify "a" blocked while "b" still allowed.
test_rate_limiter_cleanup -- Add bucket, set last_refill to 700s ago, trigger cleanup, verify evicted.
test_rate_limiter_no_cleanup_recent -- Add bucket used 100s ago, trigger cleanup, verify NOT evicted.
test_auth_rate_limit_blocks_after_burst -- Mock Request with client.host. Call _check_rate_limit_ip 11 times (capacity=10). First 10 pass, 11th raises HTTPException(429).
test_inference_rate_limit_blocks -- Call _check_rate_limit_user("user1", "inference") 6 times (capacity=5). First 5 pass, 6th raises HTTPException(429).
test_different_tiers_independent -- Rate limit user on "auth" tier, verify "api" tier still works for same identifier.
test_config_integration -- Verify Agent ZeroConfig loads rate_limit_* fields with correct defaults and constraints.
test_websocket_rate_limit_message -- Simulate rate-limited WebSocket turn. Verify error JSON message sent instead of processing.
test_model_load_rate_limit -- POST /api/load-model 3 times rapidly. Third should return 429.
```

## Estimated Impact

- **Security:** Eliminates brute force login, registration flooding, and inference DoS attacks
- **Cost:** Prevents unbounded GPU compute consumption on RunPod
- **Compliance:** Satisfies OWASP API4:2023 and LLM10:2025 requirements
- **User experience:** Generous limits (12 turns/min for inference) won't affect normal usage
- **Performance:** Token bucket check is O(1) with sub-microsecond latency; asyncio.Lock contention is negligible at expected load
- **Zero dependencies:** No new pip packages required
