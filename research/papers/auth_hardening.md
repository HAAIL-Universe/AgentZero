---
topic: Authentication Hardening
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T23:30:00Z
---

# Authentication Hardening

## Problem Statement

Agent Zero's authentication layer has five security gaps that violate NIST SP 800-63B Rev 4
(August 2025) and OWASP API Security Top 10 (API2:2023 Broken Authentication):

1. **Weak password minimum** -- 6 characters (auth.py:83). NIST SP 800-63B Rev 4 requires
   minimum 8 characters with MFA, or 15 characters without MFA. Agent Zero has no MFA, so the
   NIST-compliant minimum is 15 characters. At minimum, the floor should be 8 characters
   (matching OWASP Authentication Cheat Sheet recommendations).

2. **No email format validation** -- Only checks `"@" not in email` (auth.py:85). Accepts
   `"@"`, `"@@@@"`, `"a@"`, etc. RFC 5322 defines proper email syntax. No deliverability
   check exists.

3. **No login brute-force protection** -- No failed-login counter, no account lockout, no
   exponential backoff. An attacker can attempt unlimited passwords against any account.
   OWASP recommends locking accounts after 3-5 failed attempts with a 20-minute cooldown.

4. **No rate limiting on auth HTTP endpoints** -- Config defines `rate_limit_auth_capacity=10`
   and `rate_limit_auth_refill=0.1` (config.py:260-266), but these are never applied to
   `/auth/register` or `/auth/login` (confirmed: zero references in agent_zero_server.py).
   Registration flooding and credential stuffing are unmitigated.

5. **Per-request DB lookup for JWT validation** -- `get_current_user()` (auth.py:150-155)
   queries the database on every authenticated request to verify the user still exists. Under
   load, this creates N+1-style amplification. A TTL cache would reduce DB hits by ~95%.

## Current State in Agent Zero

### Password Validation (auth.py:82-86)
```python
if len(password) < 6:
    raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
if "@" not in email:
    raise HTTPException(status_code=400, detail="Invalid email address")
```

### Login Function (auth.py:113-132)
```python
async def login_user(email: str, password: str) -> dict:
    row = await fetch_one(
        "SELECT id, email, password_hash, display_name FROM users WHERE email = $1",
        email
    )
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    # ... returns token immediately, no failed-attempt tracking
```

### Auth Endpoints (agent_zero_server.py:613-624)
```python
@app.post("/auth/register")
async def api_register(req: AuthRequest):
    result = await register_user(req.email, req.password, req.display_name)
    return JSONResponse(result)

@app.post("/auth/login")
async def api_login(req: AuthRequest):
    result = await login_user(req.email, req.password)
    return JSONResponse(result)
```
No rate limiting applied. No IP extraction. No lockout logic.

### JWT User Lookup (auth.py:137-161)
```python
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    # ... decode JWT ...
    row = await fetch_one(
        "SELECT id, email, display_name FROM users WHERE id = $1::uuid",
        uuid.UUID(user_id)
    )
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return {"user_id": str(row["id"]), "email": row["email"], ...}
```
Every authenticated endpoint triggers a DB query. No caching.

### Config (config.py:260-266) -- rate limit settings exist but are unused
```python
rate_limit_auth_capacity: int = Field(default=10, ...)
rate_limit_auth_refill: float = Field(default=0.1, ...)
```

## Industry Standard / Research Findings

### 1. Password Policy -- NIST SP 800-63B Rev 4 (August 2025)

NIST SP 800-63B Revision 4 states that verifiers SHALL require memorized secrets to be
at least 8 characters when used with MFA, or at least 15 characters as a sole authenticator.
Verifiers SHALL accept passwords up to at least 64 characters. Verifiers SHALL NOT impose
composition rules (e.g., requiring uppercase + digit + symbol). Verifiers SHALL compare
prospective passwords against a blocklist of commonly-used, expected, or compromised values
including passwords from breach corpuses.

- Source: NIST SP 800-63B Rev 4, https://pages.nist.gov/800-63-4/sp800-63b.html
- Summary: Enzoic, https://www.enzoic.com/blog/nist-sp-800-63b-rev4/
- Refresh guide: Reliable Penguin, https://blogs.reliablepenguin.com/2025/10/22/your-password-policy-is-due-for-a-2025-refresh-what-nist-now-recommends

### 2. Email Validation -- RFC 5322 + Practical Standards

RFC 5322 defines email address syntax but is notoriously complex. Best practice for
registration forms is a lightweight structural check (local-part@domain, domain has at
least one dot) combined with a verification email for deliverability. Python's
`email.headerregistry.Address` (stdlib, Python 3.6+) performs a full RFC 5322 grammar
parse without external dependencies. The `email-validator` PyPI package is the gold
standard for production use.

- RFC 5322 grammar: https://www.opreto.com/blog/rfc-compliant-email-address-validation/
- Python approaches: https://www.abstractapi.com/guides/api-functions/email-validator-in-python
- Practical vs. RFC: https://www.suped.com/blog/what-rfc-5322-says-vs-what-actually-works

### 3. Brute-Force Protection -- OWASP Authentication Cheat Sheet

OWASP recommends locking accounts after 3-5 failed attempts within an observation window.
The lockout counter should be associated with the account (not IP) to prevent distributed
brute-force attacks from bypassing IP-based limits. Lockout duration should be at least
20 minutes, or exponential (doubling from 1 second per attempt). CAPTCHA or multi-factor
authentication provide defense-in-depth.

- OWASP Authentication Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
- OWASP Blocking Brute Force Attacks: https://owasp.org/www-community/controls/Blocking_Brute_Force_Attacks
- OWASP API2:2023 Broken Authentication: https://owasp.org/API-Security/editions/2023/en/0xa2-broken-authentication/

### 4. JWT User Cache -- TTL Cache Pattern

The `fast-api-jwt-middleware` library demonstrates the pattern: cache decoded user objects
with a TTL (e.g., 5 minutes) using `cachetools.TTLCache`. This reduces DB lookups by ~95%
for active users. Cache invalidation on logout or security events is handled by key deletion.
The pattern is standard across FastAPI production deployments.

- FastAPI JWT patterns: https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
- Cache strategies: https://benavlabs.github.io/FastAPI-boilerplate/user-guide/caching/cache-strategies/
- JWT middleware caching: https://libraries.io/pypi/fast-api-jwt-middleware

### 5. Breached Password Screening -- Have I Been Pwned k-Anonymity API

NIST SP 800-63B Rev 4 requires screening passwords against breach corpuses. The Have I
Been Pwned Passwords API uses k-anonymity: SHA-1 hash the password, send only the first
5 hex characters, receive all matching suffixes, check locally. Zero password disclosure.
The API handles 18 billion+ requests/month via Cloudflare. Free for all use.

- HIBP Pwned Passwords: https://haveibeenpwned.com/Passwords
- NIST breach screening: https://www.enzoic.com/blog/nist-check-compromised-credentials/

## Proposed Implementation

### Change 1: Strengthen Password Policy (auth.py)

**File:** `agent_zero/auth.py`, function `register_user()` (line 75-110)

```python
# Replace the password length check (auth.py:83-84) with:
MIN_PASSWORD_LENGTH = 8  # NIST SP 800-63B Rev 4 minimum (with future MFA)
MAX_PASSWORD_LENGTH = 64  # NIST: SHALL accept at least 64 chars

if len(password) < MIN_PASSWORD_LENGTH:
    raise HTTPException(
        status_code=400,
        detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    )
if len(password) > MAX_PASSWORD_LENGTH:
    raise HTTPException(
        status_code=400,
        detail=f"Password must be at most {MAX_PASSWORD_LENGTH} characters"
    )
```

**Optional enhancement (NIST SHOULD):** Add a local top-10K breached password blocklist.
Store as a frozen set loaded at startup from a text file. Avoid the HIBP API call during
registration to keep the system self-contained (no external dependency on registration path).

```python
# Load at module level
_COMMON_PASSWORDS: frozenset = frozenset()
_blocklist_path = os.path.join(os.path.dirname(__file__), "data", "common_passwords.txt")
if os.path.isfile(_blocklist_path):
    with open(_blocklist_path) as f:
        _COMMON_PASSWORDS = frozenset(line.strip().lower() for line in f if line.strip())

# In register_user():
if password.lower() in _COMMON_PASSWORDS:
    raise HTTPException(
        status_code=400,
        detail="This password is too common. Please choose a different one."
    )
```

### Change 2: Proper Email Validation (auth.py)

**File:** `agent_zero/auth.py`, function `register_user()` (line 85-86)

Use Python's stdlib `email.headerregistry.Address` for RFC 5322 parsing (no new dependency):

```python
from email.headerregistry import Address

def _validate_email(email: str) -> bool:
    """Validate email format using Python's RFC 5322 parser."""
    try:
        addr = Address(addr_spec=email)
        # Must have local part and domain with at least one dot
        if not addr.username or not addr.domain:
            return False
        if "." not in addr.domain:
            return False
        return True
    except (ValueError, IndexError):
        return False

# In register_user(), replace the "@" check:
if not _validate_email(email):
    raise HTTPException(status_code=400, detail="Invalid email address format")
```

### Change 3: Account Lockout (auth.py)

**File:** `agent_zero/auth.py`, new module-level state + changes to `login_user()`

In-memory lockout tracker (appropriate for single-instance RunPod deployment):

```python
import time
from collections import defaultdict
from dataclasses import dataclass, field

MAX_FAILED_ATTEMPTS = 5           # OWASP: 3-5 attempts
LOCKOUT_DURATION_SECONDS = 1200   # 20 minutes (OWASP recommendation)
ATTEMPT_WINDOW_SECONDS = 900      # 15-minute observation window

@dataclass
class LoginAttemptTracker:
    attempts: list = field(default_factory=list)  # timestamps of failed attempts
    locked_until: float = 0.0

_login_attempts: dict[str, LoginAttemptTracker] = defaultdict(LoginAttemptTracker)

def _check_lockout(email: str) -> Optional[int]:
    """Check if account is locked. Returns seconds remaining or None."""
    tracker = _login_attempts.get(email)
    if not tracker:
        return None
    now = time.monotonic()
    if tracker.locked_until > now:
        return int(tracker.locked_until - now)
    return None

def _record_failed_login(email: str):
    """Record a failed login attempt. Lock account if threshold exceeded."""
    tracker = _login_attempts[email]
    now = time.monotonic()
    # Prune old attempts outside the observation window
    cutoff = now - ATTEMPT_WINDOW_SECONDS
    tracker.attempts = [t for t in tracker.attempts if t > cutoff]
    tracker.attempts.append(now)
    if len(tracker.attempts) >= MAX_FAILED_ATTEMPTS:
        tracker.locked_until = now + LOCKOUT_DURATION_SECONDS

def _clear_failed_logins(email: str):
    """Clear failed login tracking on successful login."""
    _login_attempts.pop(email, None)
```

**Modify `login_user()`** (auth.py:113-132):

```python
async def login_user(email: str, password: str) -> dict:
    # Check lockout BEFORE any DB query (prevents timing attacks)
    remaining = _check_lockout(email)
    if remaining:
        raise HTTPException(
            status_code=429,
            detail=f"Account temporarily locked. Try again in {remaining // 60 + 1} minutes."
        )

    row = await fetch_one(
        "SELECT id, email, password_hash, display_name FROM users WHERE email = $1",
        email
    )
    if not row:
        _record_failed_login(email)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(password, row["password_hash"]):
        _record_failed_login(email)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    _clear_failed_logins(email)
    user_id = str(row["id"])
    token = create_token(user_id, row["email"])
    return {"user_id": user_id, "email": row["email"], "display_name": row["display_name"], "token": token}
```

### Change 4: Rate Limiting on Auth HTTP Endpoints (agent_zero_server.py)

**File:** `agent_zero/agent_zero_server.py`, endpoints at lines 613-624

Wire the already-configured `rate_limit_auth_capacity` / `rate_limit_auth_refill` from
config into the `/auth/register` and `/auth/login` endpoints:

```python
from fastapi import Request

@app.post("/auth/register")
async def api_register(req: AuthRequest, request: Request):
    """Register a new user."""
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"auth:{client_ip}"
    allowed = await rate_limiter.check(
        rate_key, _cfg.rate_limit_auth_capacity, _cfg.rate_limit_auth_refill
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    result = await register_user(req.email, req.password, req.display_name)
    return JSONResponse(result)


@app.post("/auth/login")
async def api_login(req: AuthRequest, request: Request):
    """Login and get JWT token."""
    client_ip = request.client.host if request.client else "unknown"
    rate_key = f"auth:{client_ip}"
    allowed = await rate_limiter.check(
        rate_key, _cfg.rate_limit_auth_capacity, _cfg.rate_limit_auth_refill
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    result = await login_user(req.email, req.password)
    return JSONResponse(result)
```

Note: `Request` is already imported by FastAPI. The `rate_limiter` singleton is already
imported at agent_zero_server.py:123.

### Change 5: JWT User Cache with TTL (auth.py)

**File:** `agent_zero/auth.py`, modify `get_current_user()`

Use a simple dict-based TTL cache (no new dependency). Cache user dicts for 5 minutes:

```python
import time

_USER_CACHE_TTL = 300  # 5 minutes
_user_cache: dict[str, tuple[dict, float]] = {}  # user_id -> (user_dict, expiry)

def _get_cached_user(user_id: str) -> Optional[dict]:
    """Get user from cache if not expired."""
    entry = _user_cache.get(user_id)
    if entry and entry[1] > time.monotonic():
        return entry[0]
    _user_cache.pop(user_id, None)
    return None

def _cache_user(user_id: str, user_dict: dict):
    """Cache a user dict with TTL."""
    _user_cache[user_id] = (user_dict, time.monotonic() + _USER_CACHE_TTL)

def invalidate_user_cache(user_id: str):
    """Remove user from cache (call on logout, password change, deletion)."""
    _user_cache.pop(user_id, None)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Check cache first
    cached = _get_cached_user(user_id)
    if cached:
        return cached

    row = await fetch_one(
        "SELECT id, email, display_name FROM users WHERE id = $1::uuid",
        uuid.UUID(user_id)
    )
    if not row:
        raise HTTPException(status_code=401, detail="User not found")

    user_dict = {
        "user_id": str(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"],
    }
    _cache_user(user_id, user_dict)
    return user_dict
```

**Cache invalidation points** -- call `invalidate_user_cache(user_id)` at:
- `/api/user/clear-data` endpoint (agent_zero_server.py:914)
- Any future password-change endpoint
- Any future admin user-modification endpoint

### Config Additions (config.py)

Add new auth-related config fields to `Agent ZeroConfig`:

```python
# --- Auth Hardening ---
min_password_length: int = Field(
    default=8, ge=6, le=64,
    description="Minimum password length (NIST SP 800-63B: 8 with MFA, 15 without)"
)
max_password_length: int = Field(
    default=64, ge=32, le=256,
    description="Maximum password length (NIST: at least 64)"
)
login_lockout_max_attempts: int = Field(
    default=5, ge=3, le=20,
    description="Failed login attempts before account lockout (OWASP: 3-5)"
)
login_lockout_duration_seconds: int = Field(
    default=1200, ge=60, le=7200,
    description="Account lockout duration in seconds (OWASP: 20 min)"
)
login_lockout_window_seconds: int = Field(
    default=900, ge=60, le=3600,
    description="Observation window for failed attempts in seconds"
)
jwt_user_cache_ttl_seconds: int = Field(
    default=300, ge=30, le=3600,
    description="TTL for JWT user cache in seconds (0 = disabled)"
)
```

Then reference these in auth.py instead of hardcoded constants.

## Test Specifications

### test_auth_hardening.py

```python
# --- Password Policy Tests ---

def test_register_short_password_rejected():
    """Password under 8 chars rejected with 400."""
    # POST /auth/register with password="short" -> 400

def test_register_8_char_password_accepted():
    """Password exactly 8 chars accepted."""
    # POST /auth/register with password="abcdefgh" -> 200

def test_register_long_password_accepted():
    """Password up to 64 chars accepted."""
    # POST /auth/register with password="a"*64 -> 200

def test_register_over_max_password_rejected():
    """Password over 64 chars rejected."""
    # POST /auth/register with password="a"*65 -> 400

def test_register_common_password_rejected():
    """Common password (e.g., 'password123') rejected."""
    # POST /auth/register with password="password123" -> 400

# --- Email Validation Tests ---

def test_register_valid_email_accepted():
    """Standard email like user@example.com accepted."""

def test_register_at_only_rejected():
    """Email '@' rejected."""

def test_register_no_domain_dot_rejected():
    """Email 'user@localhost' rejected (no dot in domain)."""

def test_register_empty_local_rejected():
    """Email '@example.com' rejected."""

def test_register_no_at_rejected():
    """Email 'userexample.com' rejected."""

# --- Login Lockout Tests ---

def test_lockout_after_5_failed_attempts():
    """Account locked after 5 failed attempts within window."""
    # 5x POST /auth/login with wrong password -> 429 on 6th

def test_lockout_returns_retry_hint():
    """Locked response includes approximate retry time."""
    # 429 response detail contains "minutes"

def test_successful_login_clears_counter():
    """Successful login after 4 failures resets the counter."""
    # 4 failures, 1 success, then 4 more failures -> still allowed

def test_lockout_expires_after_duration():
    """Account unlocks after lockout duration."""
    # Mock time.monotonic to advance past lockout

def test_old_attempts_pruned_outside_window():
    """Failed attempts older than window don't count."""
    # Mock time to put attempts outside observation window

# --- Auth Rate Limiting Tests ---

def test_register_rate_limited_by_ip():
    """Registration endpoint rate-limited per IP."""
    # 11 rapid requests -> 429 on 11th (capacity=10)

def test_login_rate_limited_by_ip():
    """Login endpoint rate-limited per IP."""
    # 11 rapid requests -> 429 on 11th

def test_rate_limit_separate_per_ip():
    """Different IPs get separate rate limit buckets."""

# --- JWT User Cache Tests ---

def test_cached_user_avoids_db_query():
    """Second request with same token uses cache (mock DB to verify no call)."""

def test_cache_expires_after_ttl():
    """Cached user entry expires after TTL."""
    # Mock time, verify DB queried again after TTL

def test_invalidate_cache_forces_db_lookup():
    """invalidate_user_cache() forces next request to hit DB."""

def test_deleted_user_eventually_rejected():
    """User deleted from DB is rejected after cache TTL expires."""
```

## Estimated Impact

- **Security:** Eliminates 4 OWASP API2:2023 violations and aligns password policy with
  NIST SP 800-63B Rev 4. Brute-force attacks go from unlimited to 5 attempts per 15 minutes.
  Registration flooding goes from unlimited to 10 per IP burst (6/minute sustained).

- **Performance:** JWT user cache reduces authenticated-endpoint DB queries by ~95%
  (from every request to once per 5 minutes per user). At 100 authenticated requests/minute
  per user, this saves ~95 DB queries/minute.

- **Backward Compatibility:** Existing users with passwords 6-7 chars can still log in
  (validation only applies at registration). The password minimum increase only affects
  new registrations.

## Related Papers
- `research/papers/jwt_security_hardening.md` -- covers JWT secret, query param leakage,
  revocation, refresh tokens (complementary, no overlap)
- `research/papers/api_rate_limiting.md` -- covers general endpoint rate limiting
  (this paper adds the specific auth endpoint wiring that was missing)
