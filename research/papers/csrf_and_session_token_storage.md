---
topic: CSRF and Session Token Storage
status: ready_for_implementation
priority: high
estimated_complexity: large
researched_at: 2026-03-18T00:00:00Z
---

# CSRF and Session Token Storage

## Problem Statement

Agent Zero stores JWT tokens in browser localStorage (XSS-accessible), passes tokens in WebSocket URL query parameters (logged in proxies and browser history), has zero CSRF protection on state-changing endpoints, and uses over-permissive CORS configuration. Combined with the 60+ innerHTML injections documented in the frontend_xss_and_csp paper, this creates a complete token theft chain: XSS -> localStorage read -> account takeover.

## Current State in Agent Zero

### 1. localStorage Token Storage

**File: `agent_zero/templates/agent_zero.html`**
- **Line 1940**: `let authToken = localStorage.getItem('agent_zero_token');`
- **Line 2015**: `localStorage.setItem('agent_zero_token', authToken);`
- **Line 2049**: `localStorage.removeItem('agent_zero_token');`

Any JavaScript executing on the page (including via XSS) can call `localStorage.getItem('agent_zero_token')` and exfiltrate the JWT.

### 2. WebSocket Token in URL Query Parameters

**File: `agent_zero/templates/agent_zero.html`**
- **Line 2129**: `const tokenParam = authToken ? '?token=' + encodeURIComponent(authToken) : '';`
- **Line 2131**: `ws = new WebSocket(\`${protocol}//${location.host}${route}${tokenParam}\`);`

**File: `agent_zero/agent_zero_server.py`**
- **Line 3372**: `async def websocket_chat(websocket: WebSocket, token: Optional[str] = Query(None)):`
- **Line 3461**: `async def websocket_voice(websocket: WebSocket, token: Optional[str] = Query(None)):`
- **Lines 1466-1509**: `_open_ws_session(token)` extracts and decodes the token

**Exposure vectors**:
- Browser history contains the full URL with token
- Web server access logs contain the full URL
- Reverse proxy logs (nginx, Caddy) contain the full URL
- HTTP Referer headers leak the URL to third parties

### 3. CORS Configuration

**File: `agent_zero/agent_zero_server.py` (lines 178-192)**

```python
_ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get(
    "AGENT_ZERO_CORS_ORIGINS",
    "http://localhost:8888,http://localhost:3000,http://127.0.0.1:8888,http://127.0.0.1:3000",
).split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],        # Over-permissive
    allow_headers=["*"],        # Over-permissive
)
```

- `allow_credentials=True` + `allow_headers=["*"]` allows credentialed cross-origin requests with any headers
- `allow_methods=["*"]` allows all HTTP methods (DELETE, PUT, PATCH) from cross-origin

### 4. Zero CSRF Protection

- No CSRF middleware installed
- No CSRF token generation or validation anywhere in the codebase
- All state-changing endpoints (POST, PUT, DELETE) are unprotected:
  - `/auth/register` (line 621)
  - `/auth/login` (line 635)
  - `/api/onboarding` (line 685)
  - `/user/goals` POST/DELETE (lines 863, 923)
  - `/user/commitments` POST/PUT/DELETE (lines 1021, 1119, 1135)
  - `/api/user/clear-data` DELETE (documented in silent_exception_data_loss paper)

### 5. No Cookie-Based Auth

- No `Set-Cookie` headers anywhere in the codebase
- No `httpOnly`, `Secure`, or `SameSite` attributes used
- All auth is via Bearer token in Authorization header (HTTP) or query parameter (WebSocket)

## Industry Standard / Research Findings

### OWASP CSRF Prevention Cheat Sheet (2025)

OWASP recommends the Synchronizer Token Pattern for CSRF protection: generate a unique token per session, include it in state-changing requests, validate server-side. For SPAs using JWT, the recommended approach is: (1) store JWT in httpOnly cookie, (2) add CSRF token as a separate non-httpOnly cookie or header, (3) validate both on the server.

**Source**: [OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html) -- OWASP Foundation, 2025.

### OWASP WebSocket Security Cheat Sheet (2025)

OWASP explicitly warns against passing tokens in WebSocket URL query parameters: "Tokens will appear in access logs and should be redacted." The recommended pattern is **first-message authentication**: connect the WebSocket without credentials, then send the token as the first message over the established encrypted channel.

**Source**: [OWASP WebSocket Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/WebSocket_Security_Cheat_Sheet.html) -- OWASP Foundation, 2025.

### JWT Storage: localStorage vs httpOnly Cookies (2025)

The Descope Developer Guide (2025) documents the consensus: "Storing JWTs in localStorage prevents pure CSRF attacks but leaves applications vulnerable to far more common XSS attacks, which can steal tokens. The safest approach for most applications is to store JWTs in httpOnly, Secure cookies with SameSite attributes." This mitigates both XSS (httpOnly prevents JS access) and CSRF (SameSite prevents cross-origin cookie sending).

**Source**: [The Developer's Guide to JWT Storage](https://www.descope.com/blog/post/developer-guide-jwt-storage) -- Descope, 2025.

### WebSocket First-Message Authentication Pattern (VideoSDK, 2025)

VideoSDK (2025) documents the first-message auth pattern: "The client first authenticates via a standard HTTP login API and receives a session token. The client then establishes a WebSocket connection. Once the connection is open, the client immediately sends an authentication message containing the token. The server validates the token before processing any subsequent messages."

**Source**: [WebSocket Authentication: Securing Real-Time Connections in 2025](https://www.videosdk.live/developer-hub/websocket/websocket-authentication) -- VideoSDK, 2025.

### JWT Vulnerabilities and Mitigation (Red Sentry, 2026)

Red Sentry (2026) documents common JWT vulnerabilities including token leakage via logs, referer headers, and browser history when tokens are in URLs. Recommended: "Rotate tokens in long-lived connections to prevent hijacked sessions from persisting."

**Source**: [JWT Vulnerabilities List: 2026 Security Risks & Mitigation Guide](https://redsentry.com/resources/blog/jwt-vulnerabilities-list-2026-security-risks-mitigation-guide) -- Red Sentry, 2026.

### SameSite Cookie Attribute (OWASP HTML5 Security, 2025)

OWASP recommends `SameSite=Strict` for session cookies in same-site applications, or `SameSite=Lax` if cross-site navigation is needed. Combined with `Secure` (HTTPS-only) and `httpOnly` (no JS access), this is the defense-in-depth standard.

**Source**: [OWASP HTML5 Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/HTML5_Security_Cheat_Sheet.html) -- OWASP Foundation, 2025.

## Proposed Implementation

### Phase 1: Migrate to httpOnly Cookie Storage (HIGH PRIORITY)

#### Server-side (auth.py + agent_zero_server.py)

**auth.py -- Set cookie on login:**
```python
from starlette.responses import JSONResponse

async def login(request):
    # ... existing validation ...
    token = create_token(user)
    response = JSONResponse({"user": user_data})
    response.set_cookie(
        key="agent_zero_token",
        value=token,
        httponly=True,
        secure=True,          # HTTPS only (always true on RunPod)
        samesite="strict",    # No cross-site sending
        max_age=43200,        # 12 hours (matches JWT_EXPIRY_HOURS)
        path="/",
    )
    return response
```

**auth.py -- Extract token from cookie OR header:**
```python
async def get_current_user(request):
    """Extract JWT from httpOnly cookie (preferred) or Authorization header (fallback)."""
    token = request.cookies.get("agent_zero_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(401, "Not authenticated")
    return decode_token(token)
```

**auth.py -- Clear cookie on logout:**
```python
async def logout(request):
    response = JSONResponse({"status": "ok"})
    response.delete_cookie("agent_zero_token", path="/")
    return response
```

#### Client-side (agent_zero.html)

Remove all `localStorage.getItem/setItem/removeItem('agent_zero_token')` calls. The browser automatically sends the httpOnly cookie with every request to the same origin. Update fetch calls to include `credentials: 'include'`:

```javascript
// Replace all fetch calls that set Authorization header:
// BEFORE:
// headers: { 'Authorization': 'Bearer ' + authToken }
// AFTER:
fetch(url, { credentials: 'include' });
// Cookie is sent automatically
```

### Phase 2: WebSocket First-Message Authentication

Replace query-parameter token passing with first-message auth:

#### Server-side (agent_zero_server.py)

```python
async def websocket_chat(websocket: WebSocket):
    """WebSocket chat endpoint with first-message auth."""
    await websocket.accept()

    # Wait for auth message (5 second timeout)
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        msg = json.loads(raw)
        if msg.get("type") != "auth":
            await websocket.close(4001, "First message must be auth")
            return
        token = msg.get("token")
        # Also check cookie as fallback
        if not token:
            token = websocket.cookies.get("agent_zero_token")
        user = await _open_ws_session(token)
    except asyncio.TimeoutError:
        await websocket.close(4002, "Auth timeout")
        return
    except Exception:
        await websocket.close(4003, "Auth failed")
        return

    # ... proceed with authenticated session ...
```

#### Client-side (agent_zero.html)

```javascript
// BEFORE:
// const tokenParam = authToken ? '?token=' + encodeURIComponent(authToken) : '';
// ws = new WebSocket(`${protocol}//${location.host}${route}${tokenParam}`);

// AFTER:
ws = new WebSocket(`${protocol}//${location.host}${route}`);
ws.onopen = function() {
    // Send auth as first message (cookie handles REST, this handles WS)
    ws.send(JSON.stringify({ type: 'auth' }));
    // Server reads cookie from the upgrade request
};
```

Note: WebSocket upgrade requests DO include cookies, so with httpOnly cookies the server can authenticate from `websocket.cookies` directly. The first-message pattern is a defense-in-depth layer.

### Phase 3: CSRF Protection with Double-Submit Cookie

For state-changing REST endpoints, add a CSRF token using the double-submit cookie pattern:

#### Server-side (new file: agent_zero/csrf.py)

```python
import secrets
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

CSRF_COOKIE_NAME = "agent_zero_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Skip safe methods
        if request.method in SAFE_METHODS:
            response = await call_next(request)
            # Set CSRF cookie if not present
            if CSRF_COOKIE_NAME not in request.cookies:
                csrf_token = secrets.token_urlsafe(32)
                response.set_cookie(
                    CSRF_COOKIE_NAME, csrf_token,
                    httponly=False,  # JS must read this
                    secure=True, samesite="strict", path="/",
                )
            return response

        # Validate CSRF on state-changing methods
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        header_token = request.headers.get(CSRF_HEADER_NAME)
        if not cookie_token or not header_token or cookie_token != header_token:
            return Response("CSRF validation failed", status_code=403)

        return await call_next(request)
```

#### Client-side (agent_zero.html)

```javascript
function getCsrfToken() {
    const match = document.cookie.match(/agent_zero_csrf=([^;]+)/);
    return match ? match[1] : '';
}

// Add to all fetch calls for POST/PUT/DELETE:
fetch(url, {
    method: 'POST',
    credentials: 'include',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': getCsrfToken(),
    },
    body: JSON.stringify(data),
});
```

### Phase 4: Tighten CORS Configuration

```python
# agent_zero_server.py -- replace lines 186-192
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-CSRF-Token"],
)
```

## Test Specifications

### test_csrf_and_token_storage.py

1. **test_login_sets_httponly_cookie** -- POST /auth/login with valid credentials. Verify response has `Set-Cookie: agent_zero_token=...; HttpOnly; Secure; SameSite=Strict`.

2. **test_login_no_token_in_body** -- POST /auth/login. Verify response body does NOT contain the JWT string (it's in the cookie only).

3. **test_authenticated_request_via_cookie** -- Set agent_zero_token cookie, call protected endpoint without Authorization header. Verify 200 response.

4. **test_logout_clears_cookie** -- POST /auth/logout. Verify `Set-Cookie` with `max-age=0` or `expires` in the past.

5. **test_csrf_token_set_on_get** -- GET any page. Verify `agent_zero_csrf` cookie is set (non-httpOnly).

6. **test_csrf_required_on_post** -- POST to state-changing endpoint without X-CSRF-Token header. Verify 403 response.

7. **test_csrf_valid_on_post** -- POST with matching CSRF cookie and header. Verify request succeeds.

8. **test_csrf_mismatch_rejected** -- POST with mismatched CSRF cookie and header. Verify 403.

9. **test_websocket_no_query_token** -- Connect WebSocket without query parameter. Send auth message. Verify connection accepted.

10. **test_websocket_first_message_auth** -- Connect WebSocket, send `{"type": "auth"}` as first message. Verify server reads cookie and authenticates.

11. **test_websocket_no_auth_timeout** -- Connect WebSocket, send no message for 5s. Verify connection closed with 4002.

12. **test_websocket_wrong_first_message** -- Connect WebSocket, send non-auth message. Verify connection closed with 4001.

13. **test_cors_explicit_methods** -- Send OPTIONS preflight with DELETE method. Verify allowed. Send with PATCH. Verify denied (not in allow_methods list).

14. **test_cors_explicit_headers** -- Send preflight with `X-CSRF-Token` header. Verify allowed. Send with `X-Custom` header. Verify denied.

15. **test_no_localstorage_in_frontend** -- Grep agent_zero.html for `localStorage.getItem('agent_zero_token')`. Verify zero matches (all removed).

## Estimated Impact

- **Security (critical)**: Eliminates the XSS -> token theft -> account takeover chain by making JWT inaccessible to JavaScript
- **Security (high)**: Removes token from WebSocket URLs, preventing log/history/referer leakage
- **Security (medium)**: CSRF protection prevents cross-site state-changing attacks
- **Security (medium)**: Tightened CORS reduces cross-origin attack surface
- **Complexity**: Large -- touches auth.py, agent_zero_server.py, agent_zero.html, adds csrf.py. Must be tested carefully with existing auth flows.
- **Backward compatibility**: The Authorization header fallback ensures API clients still work during migration
- **Dependencies**: Should be implemented AFTER frontend_xss_and_csp paper (CSP headers provide defense-in-depth even before token storage migration)
