---
topic: JWT Security Hardening
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T22:00:00Z
---

# JWT Security Hardening

## Problem Statement

Agent Zero's JWT authentication has four critical security gaps:

1. **Weak default secret** -- The signing secret defaults to a hardcoded string `"agent_zero-dev-secret-change-in-prod"` (auth.py:18). If deployed without setting `AGENT_ZERO_JWT_SECRET`, any attacker who reads the source can forge valid tokens for any user. Per Auth0's research, HS256 secrets shorter than 256 bits can be brute-forced using hashcat in hours on consumer hardware.

2. **Token in WebSocket query parameters** -- JWT tokens are passed as `?token=<jwt>` in WebSocket URLs (agent_zero.html:2129, agent_zero_server.py:3173-3177). Query parameters are logged in browser history, server access logs, proxy logs, and CDN logs. OWASP and VideoSDK's 2025 WebSocket security guide both classify this as a token leakage vector.

3. **No server-side token revocation** -- Logout only clears `localStorage` on the frontend (agent_zero.html:2049). The JWT remains valid until expiry (24h default). A stolen token cannot be invalidated.

4. **No token refresh mechanism** -- Access tokens last 24 hours with no refresh rotation. Best practice (RFC 6819, OWASP) is short-lived access tokens (15-30 min) with rotating refresh tokens.

## Current State in Agent Zero

### Token Creation (auth.py:41-51)
```python
def create_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id, "email": email, "aud": "agent_zero",
        "iat": now, "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
```

### Secret Configuration (auth.py:18-19)
```python
JWT_SECRET = os.environ.get("AGENT_ZERO_JWT_SECRET", "agent_zero-dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
```

### Token Validation (auth.py:54-66)
- Verifies signature, expiry, audience="agent_zero"
- Raises HTTPException(401) on failure
- No `jti` claim, no revocation check

### WebSocket Auth (agent_zero_server.py:3173-3177, 3242-3246)
```python
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: Optional[str] = Query(None)):
    await websocket.accept()
    user, session_id, shadow = await _open_ws_session(token)
```
Token passed as query parameter. Connection accepted before authentication.

### Frontend Token Storage (agent_zero.html:1940, 2014-2015)
```javascript
let authToken = localStorage.getItem('agent_zero_token');
localStorage.setItem('agent_zero_token', authToken);
```

### WebSocket Connection (agent_zero.html:2129)
```javascript
const tokenParam = authToken ? '?token=' + encodeURIComponent(authToken) : '';
ws = new WebSocket(`${protocol}//${location.host}${route}${tokenParam}`);
```

## Industry Standard / Research Findings

### 1. Secret Key Strength (RFC 7518 Section 3.2 + OWASP)

RFC 7518 mandates that HMAC keys MUST be at least as long as the hash output -- 256 bits (32 bytes) for HS256. OWASP's JWT Cheat Sheet goes further: "at least 64 characters, generated using a secure source of randomness."

Auth0's research demonstrates that HS256 with weak secrets can be brute-forced using hashcat or John the Ripper. A 20-character dictionary-based secret can be cracked in minutes on modern GPUs.

**Recommendation**: Fail-fast on startup if `AGENT_ZERO_JWT_SECRET` is not set or is the default. Generate with `secrets.token_hex(32)` (64 hex chars = 256 bits).

**Sources:**
- RFC 7518 Section 3.2: https://tools.ietf.org/html/rfc7518#section-3.2
- OWASP JWT Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html
- Auth0 HS256 Brute Force: https://auth0.com/blog/brute-forcing-hs256-is-possible-the-importance-of-using-strong-keys-to-sign-jwts/

### 2. WebSocket First-Message Authentication (VideoSDK 2025 + WebSocket.org)

VideoSDK's 2025 WebSocket Authentication guide explicitly states: "Passing tokens via URL query parameters is not recommended, as URLs can be logged and leaked." The recommended pattern is **in-band first-message authentication**: accept the connection, then require an `{"type": "auth", "token": "..."}` message before processing any other messages. This keeps tokens out of URLs while supporting token renewal during long-lived connections.

**Sources:**
- VideoSDK WebSocket Auth 2025: https://www.videosdk.live/developer-hub/websocket/websocket-authentication
- WebSocket.org Security Guide: https://websocket.org/guides/security/
- Socket.IO JWT Guide: https://socket.io/how-to/use-with-jwt

### 3. Token Revocation (OWASP + SuperTokens 2025)

OWASP recommends maintaining a server-side denylist storing SHA-256 digests of revoked tokens, checked on every request. For Python applications without Redis, an in-memory set with TTL-based cleanup is sufficient for single-instance deployments. SuperTokens' 2025 guide recommends storing `jti` claims with TTL matching remaining token lifetime.

**Sources:**
- OWASP JWT Revocation: https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html
- SuperTokens JWT Blacklist: https://supertokens.com/blog/revoking-access-with-a-jwt-blacklist
- OneUpTime JWT Revocation 2026: https://oneuptime.com/blog/post/2026-02-02-jwt-revocation/view

### 4. Short-Lived Access + Refresh Rotation (RFC 6819 + Choudhary 2025)

Best practice is short-lived access tokens (15-30 min) paired with longer-lived refresh tokens (7-14 days) that rotate on every use. When a refresh token is used, the old one is invalidated and a new one is issued. This limits the damage window of a stolen access token while maintaining session continuity.

**Sources:**
- JWT Refresh Rotation Pattern: https://choudharycodes.medium.com/securing-your-web-applications-with-jwt-authentication-and-refresh-token-rotation-63a9aa1a4b12
- FastAPI JWT Refresh: https://medium.com/@jagan_reddy/jwt-in-fastapi-the-secure-way-refresh-tokens-explained-f7d2d17b1d17
- CodeSignal Refresh Rotation: https://codesignal.com/learn/courses/preventing-refresh-token-abuse-in-your-python-rest-api/lessons/refresh-token-rotation

## Proposed Implementation

### Phase 1: Fail-Fast Secret Validation (auth.py)

**File: `agent_zero/auth.py`, lines 18-22**

Replace:
```python
JWT_SECRET = os.environ.get("AGENT_ZERO_JWT_SECRET", "agent_zero-dev-secret-change-in-prod")
```

With:
```python
_DEFAULT_SECRET = "agent_zero-dev-secret-change-in-prod"

def _get_jwt_secret() -> str:
    """Get JWT secret with fail-fast validation."""
    secret = os.environ.get("AGENT_ZERO_JWT_SECRET", _DEFAULT_SECRET)
    if secret == _DEFAULT_SECRET:
        import sys
        env = os.environ.get("AGENT_ZERO_ENV", "development")
        if env == "production":
            print("FATAL: AGENT_ZERO_JWT_SECRET must be set in production. "
                  "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"",
                  file=sys.stderr)
            sys.exit(1)
        else:
            import logging
            logging.getLogger("agent_zero.auth").warning(
                "Using default JWT secret -- NOT SAFE FOR PRODUCTION. "
                "Set AGENT_ZERO_JWT_SECRET environment variable."
            )
    if len(secret) < 32:
        import sys
        print(f"FATAL: AGENT_ZERO_JWT_SECRET must be at least 32 characters (got {len(secret)}). "
              "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"",
              file=sys.stderr)
        sys.exit(1)
    return secret

JWT_SECRET = _get_jwt_secret()
```

**Add to config.py** (new field):
```python
env: str = Field(
    default="development",
    description="Deployment environment: development, staging, production"
)
```
Environment variable: `AGENT_ZERO_ENV`

### Phase 2: Add JTI Claim + In-Memory Revocation (auth.py)

**Add `jti` to token creation (auth.py:41-51):**
```python
import uuid as _uuid

def create_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "jti": str(_uuid.uuid4()),
        "sub": user_id,
        "email": email,
        "aud": "agent_zero",
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
```

**Add revocation store (auth.py, new section):**
```python
import threading
import time

class TokenDenyList:
    """In-memory JWT denylist with automatic TTL cleanup."""

    def __init__(self):
        self._denied: dict[str, float] = {}  # jti -> expiry_timestamp
        self._lock = threading.Lock()

    def revoke(self, jti: str, exp_timestamp: float) -> None:
        """Add a token JTI to the denylist."""
        with self._lock:
            self._denied[jti] = exp_timestamp

    def is_revoked(self, jti: str) -> bool:
        """Check if a token JTI is revoked."""
        with self._lock:
            return jti in self._denied

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        with self._lock:
            expired = [jti for jti, exp in self._denied.items() if exp < now]
            for jti in expired:
                del self._denied[jti]
            return len(expired)

_denylist = TokenDenyList()
```

**Add revocation check to decode_token (auth.py:54-66):**
```python
def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token, JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience="agent_zero",
        )
        jti = payload.get("jti")
        if jti and _denylist.is_revoked(jti):
            raise HTTPException(status_code=401, detail="Token has been revoked")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
```

**Add logout endpoint (agent_zero_server.py):**
```python
@app.post("/auth/logout")
async def api_logout(user: dict = Depends(get_current_user),
                     credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Revoke the current JWT token."""
    payload = jwt.decode(credentials.credentials, JWT_SECRET,
                         algorithms=[JWT_ALGORITHM], audience="agent_zero")
    jti = payload.get("jti")
    exp = payload.get("exp", 0)
    if jti:
        _denylist.revoke(jti, exp)
    return {"status": "logged_out"}
```

**Add periodic cleanup (agent_zero_server.py startup):**
```python
import asyncio

async def _denylist_cleanup_loop():
    """Clean expired entries from token denylist every 10 minutes."""
    while True:
        await asyncio.sleep(600)
        removed = _denylist.cleanup()
        if removed:
            _log.info("denylist_cleanup", removed=removed)

@app.on_event("startup")
async def start_denylist_cleanup():
    asyncio.create_task(_denylist_cleanup_loop())
```

### Phase 3: First-Message WebSocket Authentication (agent_zero_server.py + agent_zero.html)

**Server-side (agent_zero_server.py:3173-3177, 3242-3246):**

Replace query-parameter auth with first-message auth:
```python
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for text chat."""
    await websocket.accept()
    # Wait for auth message (5 second timeout)
    user, session_id, shadow = await _ws_first_message_auth(websocket)
    # ... rest of handler
```

**New helper function:**
```python
async def _ws_first_message_auth(
    websocket: WebSocket,
    timeout: float = 5.0
) -> tuple[Optional[dict], Optional[str], Optional[dict]]:
    """Authenticate WebSocket via first message.

    Expects: {"type": "auth", "token": "<jwt>"}
    Sends back: {"type": "auth_ok"} or {"type": "auth_error", "detail": "..."}
    """
    user = None
    session_id = None
    shadow = None

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
        msg = json.loads(raw)

        if msg.get("type") != "auth":
            await websocket.send_json({"type": "auth_error", "detail": "First message must be auth"})
            return user, session_id, shadow

        token = msg.get("token")
        if not token:
            await websocket.send_json({"type": "auth_error", "detail": "Missing token"})
            return user, session_id, shadow

        user, session_id, shadow = await _open_ws_session(token)
        if user:
            await websocket.send_json({"type": "auth_ok", "session_id": session_id})
        else:
            await websocket.send_json({"type": "auth_error", "detail": "Invalid token"})

    except asyncio.TimeoutError:
        _log.warning("ws_auth_timeout", detail="No auth message within timeout")
    except (json.JSONDecodeError, Exception) as e:
        _log.warning("ws_auth_failed", error=str(e))

    return user, session_id, shadow
```

**Frontend (agent_zero.html:2129):**

Replace:
```javascript
const tokenParam = authToken ? '?token=' + encodeURIComponent(authToken) : '';
ws = new WebSocket(`${protocol}//${location.host}${route}${tokenParam}`);
```

With:
```javascript
ws = new WebSocket(`${protocol}//${location.host}${route}`);
ws.onopen = function() {
    if (authToken) {
        ws.send(JSON.stringify({type: "auth", token: authToken}));
    }
};
```

Add auth response handler early in `ws.onmessage`:
```javascript
// Inside ws.onmessage, before other handling:
if (data.type === "auth_ok") {
    console.log("WebSocket authenticated, session:", data.session_id);
    return;
}
if (data.type === "auth_error") {
    console.warn("WebSocket auth failed:", data.detail);
    // Optionally redirect to login
    return;
}
```

### Phase 4: Refresh Token Support (auth.py + agent_zero_server.py + agent_zero.html)

This is a larger change that can be implemented in a follow-up. The pattern:

1. **Short-lived access tokens** (30 min instead of 24h)
2. **Refresh tokens** stored in `httpOnly` cookie (7-day lifetime)
3. **Rotation**: each refresh issues new access + new refresh, old refresh invalidated
4. **Database table**: `refresh_tokens(id, user_id, token_hash, expires_at, revoked_at)`

This phase is estimated as a separate medium-complexity task and should be tracked separately.

## Test Specifications

### Test 1: Fail-Fast on Default Secret in Production
```python
def test_fail_fast_default_secret_production(monkeypatch):
    """Server should refuse to start with default secret in production."""
    monkeypatch.setenv("AGENT_ZERO_ENV", "production")
    monkeypatch.delenv("AGENT_ZERO_JWT_SECRET", raising=False)
    with pytest.raises(SystemExit):
        _get_jwt_secret()

def test_warning_default_secret_development(monkeypatch, caplog):
    """Development mode should warn but not crash."""
    monkeypatch.setenv("AGENT_ZERO_ENV", "development")
    monkeypatch.delenv("AGENT_ZERO_JWT_SECRET", raising=False)
    secret = _get_jwt_secret()
    assert secret == "agent_zero-dev-secret-change-in-prod"
    assert "NOT SAFE FOR PRODUCTION" in caplog.text

def test_reject_short_secret(monkeypatch):
    """Secrets under 32 chars should be rejected."""
    monkeypatch.setenv("AGENT_ZERO_JWT_SECRET", "tooshort")
    with pytest.raises(SystemExit):
        _get_jwt_secret()

def test_accept_strong_secret(monkeypatch):
    """64-char hex secret should be accepted."""
    monkeypatch.setenv("AGENT_ZERO_JWT_SECRET", "a" * 64)
    secret = _get_jwt_secret()
    assert len(secret) == 64
```

### Test 2: JTI Claim in Tokens
```python
def test_token_contains_jti():
    """Created tokens must include a jti claim."""
    token = create_token("user-123", "test@example.com")
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience="agent_zero")
    assert "jti" in payload
    assert len(payload["jti"]) == 36  # UUID format

def test_each_token_has_unique_jti():
    """Each token creation should produce a unique jti."""
    t1 = create_token("user-123", "test@example.com")
    t2 = create_token("user-123", "test@example.com")
    p1 = jwt.decode(t1, JWT_SECRET, algorithms=["HS256"], audience="agent_zero")
    p2 = jwt.decode(t2, JWT_SECRET, algorithms=["HS256"], audience="agent_zero")
    assert p1["jti"] != p2["jti"]
```

### Test 3: Token Revocation
```python
def test_denylist_revoke_and_check():
    """Revoked JTI should be detected."""
    dl = TokenDenyList()
    dl.revoke("jti-abc", time.time() + 3600)
    assert dl.is_revoked("jti-abc")
    assert not dl.is_revoked("jti-xyz")

def test_denylist_cleanup_removes_expired():
    """Cleanup should remove entries whose exp < now."""
    dl = TokenDenyList()
    dl.revoke("old", time.time() - 10)
    dl.revoke("valid", time.time() + 3600)
    removed = dl.cleanup()
    assert removed == 1
    assert not dl.is_revoked("old")
    assert dl.is_revoked("valid")

def test_decode_rejects_revoked_token():
    """decode_token should reject tokens whose jti is in the denylist."""
    token = create_token("user-123", "test@example.com")
    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience="agent_zero")
    _denylist.revoke(payload["jti"], payload["exp"])
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401
    assert "revoked" in exc.value.detail.lower()
```

### Test 4: Logout Endpoint
```python
@pytest.mark.asyncio
async def test_logout_revokes_token(client, auth_headers):
    """POST /auth/logout should revoke the current token."""
    resp = await client.post("/auth/logout", headers=auth_headers)
    assert resp.status_code == 200
    # Subsequent requests with same token should fail
    resp2 = await client.get("/auth/me", headers=auth_headers)
    assert resp2.status_code == 401
```

### Test 5: WebSocket First-Message Auth
```python
@pytest.mark.asyncio
async def test_ws_first_message_auth_success(client):
    """WebSocket should authenticate via first message."""
    token = create_token("user-123", "test@example.com")
    async with client.websocket_connect("/ws/chat") as ws:
        await ws.send_json({"type": "auth", "token": token})
        resp = await ws.receive_json()
        assert resp["type"] == "auth_ok"
        assert "session_id" in resp

@pytest.mark.asyncio
async def test_ws_auth_no_query_param(client):
    """WebSocket should NOT accept token as query parameter."""
    token = create_token("user-123", "test@example.com")
    async with client.websocket_connect(f"/ws/chat?token={token}") as ws:
        # Should not auto-authenticate from query param
        # Must send auth message
        await ws.send_json({"type": "auth", "token": token})
        resp = await ws.receive_json()
        assert resp["type"] == "auth_ok"

@pytest.mark.asyncio
async def test_ws_auth_timeout(client):
    """WebSocket should handle auth timeout gracefully."""
    async with client.websocket_connect("/ws/chat") as ws:
        # Don't send auth, wait for timeout
        # Connection should still work but user=None
        await asyncio.sleep(6)
        # Send a message -- should work as anonymous
        await ws.send_json({"type": "message", "text": "hello"})

@pytest.mark.asyncio
async def test_ws_auth_invalid_token(client):
    """WebSocket should reject invalid tokens."""
    async with client.websocket_connect("/ws/chat") as ws:
        await ws.send_json({"type": "auth", "token": "invalid.jwt.token"})
        resp = await ws.receive_json()
        assert resp["type"] == "auth_error"
```

### Test 6: Algorithm Enforcement
```python
def test_reject_none_algorithm():
    """Tokens signed with 'none' algorithm must be rejected."""
    payload = {"sub": "user-123", "email": "test@example.com", "aud": "agent_zero",
               "iat": datetime.now(timezone.utc),
               "exp": datetime.now(timezone.utc) + timedelta(hours=1)}
    # Craft a token with algorithm=none (attack vector)
    import base64, json as _json
    header = base64.urlsafe_b64encode(_json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    body = base64.urlsafe_b64encode(_json.dumps(payload, default=str).encode()).rstrip(b"=")
    forged = header.decode() + "." + body.decode() + "."
    with pytest.raises(HTTPException):
        decode_token(forged)
```

## Estimated Impact

| Area | Before | After |
|------|--------|-------|
| **Secret safety** | Hardcoded default usable in production | Fail-fast crash in production, warning in dev |
| **Token leakage** | JWT in WebSocket URLs, logged by proxies | First-message auth, tokens never in URLs |
| **Logout** | Frontend-only, token valid 24h | Server-side revocation via denylist |
| **Attack surface** | Stolen token usable until expiry | Can be revoked immediately on logout |
| **OWASP compliance** | Violates API2:2023 (Broken Auth) | Aligned with OWASP JWT Cheat Sheet |
| **Key strength** | 35-char dictionary string | Enforced 32+ char minimum, 256-bit recommended |

**OWASP violations addressed:**
- API2:2023 Broken Authentication (weak secret, no revocation)
- API8:2023 Security Misconfiguration (default secret in production)

**Backward compatibility:** Phase 1-3 require coordinated frontend+backend deployment. The WebSocket auth change (Phase 3) is a breaking change for existing clients -- the frontend must be updated simultaneously. Phase 4 (refresh tokens) can be done incrementally.
