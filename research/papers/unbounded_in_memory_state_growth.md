---
topic: Unbounded In-Memory State Growth
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T00:00:00Z
---

# Unbounded In-Memory State Growth

## Problem Statement

Multiple modules in Agent Zero maintain dictionaries and caches that grow without eviction, cleanup, or size bounds. In a long-running production server, these structures accumulate entries indefinitely, leading to gradual memory consumption growth and eventually OOM conditions. The most critical is `_login_attempts` in auth.py, which can be grown deliberately by an attacker.

## Current State in Agent Zero

### 1. auth.py: `_login_attempts` (line 91)

```python
_login_attempts: dict[str, LoginAttemptTracker] = defaultdict(LoginAttemptTracker)
```

- **Growth mechanism**: Line 107 -- `tracker = _login_attempts[email]` creates a new entry for every unique email that attempts login
- **Partial cleanup**: Line 110 -- attempts within the tracker are culled by time window (900s default), but **the email key itself persists forever**
- **Cleanup on success only**: Line 118 -- `_clear_failed_logins()` removes entries only on successful login
- **Attack vector**: An attacker sending failed logins for millions of unique emails will grow this dict unboundedly. Each entry contains a list of timestamps.
- **No background reaper**: No periodic task cleans stale entries

### 2. auth.py: `_user_cache` (line 123)

```python
_user_cache: dict[str, tuple[dict, float]] = {}
```

- **Growth mechanism**: Line 137 -- entries added when users authenticate (user_id -> (user_dict, expiry_time))
- **Lazy expiry only**: Line 131 -- expired entries removed only when that specific key is accessed again
- **No background reaper**: If a user_id is never accessed again, its expired entry remains indefinitely
- **Risk**: Long-running servers with many unique users accumulate stale entries over weeks

### 3. tool_runtime.py: `_A2_INBOX_CACHE` (line 520)

```python
_A2_INBOX_CACHE: list[dict] = []
```

- **Status**: Declared but never used (dead code). No append/clear logic found.
- **Risk**: Low (currently no impact), but maintenance hazard if code is added later without bounds

### 4. context_manager.py: `_calibration_ratio` (line 33)

```python
_calibration_ratio: float = 1.0
```

- **Thread safety**: Lines 74-80 -- `calibrate_from_usage()` uses `global _calibration_ratio` and updates via exponential moving average without any lock
- **Risk**: Not a growth issue, but concurrent updates from multiple coroutines can cause incorrect calibration. Since asyncio is single-threaded, this is only a risk if `run_in_executor()` is used (which it is -- 8 calls documented in async_safety_race_prevention paper)

## Industry Standard / Research Findings

### BetterUp FastAPI Memory Leak Case Study (2025)

BetterUp engineers documented chasing a memory leak in their async FastAPI service. They found that unbounded dict-based caches were a primary contributor to RSS growth. Their fix included: (1) bounded caches with maxsize, (2) TTL-based eviction, and (3) periodic background cleanup tasks.

**Source**: [Chasing a Memory Leak in our Async FastAPI Service](https://build.betterup.com/chasing-a-memory-leak-in-our-async-fastapi-service-how-jemalloc-fixed-our-rss-creep/) -- BetterUp Engineering, 2025.

### cachetools: Bounded Memoizing Collections (v7.0, 2025)

The `cachetools` library provides `TTLCache(maxsize, ttl)` -- a bounded dict with both size limit and per-item time-to-live. Items are evicted by LRU when the cache is full, and expired items are cleaned up on the next mutating operation. `cachetools` is the standard Python library for bounded caching.

**Source**: [cachetools Documentation](https://cachetools.readthedocs.io/en/stable/) -- cachetools v7.0.5.

### Python Memory Leak Debugging Guide (2025)

OneUptime (2025) documents that unbounded caches and growing dicts are the #1 cause of memory leaks in long-running Python services. Recommended mitigations: (1) use `tracemalloc` to detect growing object counts, (2) implement maxsize bounds on all caches, (3) add background cleanup coroutines for TTL-based eviction.

**Source**: [How to Handle Memory Leaks in Python](https://oneuptime.com/blog/post/2025-01-06-python-memory-leak-debugging/view) -- OneUptime, January 2025.

### async-lru: Async-Aware Bounded Cache (2025)

The `async-lru` library provides `@alru_cache(maxsize=N, ttl=seconds)` for asyncio functions. It ensures multiple concurrent calls to the same key result in a single computation, with automatic TTL-based expiry.

**Source**: [async-lru on PyPI](https://pypi.org/project/async-lru/) -- aio-libs, 2025.

### Python asyncio Background Tasks Pattern

The standard pattern for periodic cleanup in asyncio is `asyncio.create_task()` with a while loop and `asyncio.sleep()`. This is preferable to checking on every access (lazy cleanup) because it bounds worst-case memory between cleanup intervals.

**Source**: [Python asyncio Documentation - Tasks](https://docs.python.org/3/library/asyncio-task.html) -- Python 3.12 stdlib.

## Proposed Implementation

### Step 1: Bound `_login_attempts` with TTL eviction (auth.py)

Replace the unbounded `defaultdict` with a bounded dict and add a background reaper:

```python
# auth.py -- replace lines 91+

import time
import asyncio

_MAX_LOGIN_ATTEMPT_ENTRIES = 10000  # Max unique emails tracked
_LOGIN_ATTEMPT_TTL = 1800  # 30 minutes -- remove entries older than this

_login_attempts: dict[str, LoginAttemptTracker] = {}

def _get_login_tracker(email: str) -> LoginAttemptTracker:
    """Get or create a login tracker, evicting oldest if at capacity."""
    if email not in _login_attempts:
        if len(_login_attempts) >= _MAX_LOGIN_ATTEMPT_ENTRIES:
            # Evict oldest entry (by last attempt time)
            oldest_key = min(_login_attempts, key=lambda k: _login_attempts[k].last_attempt)
            del _login_attempts[oldest_key]
        _login_attempts[email] = LoginAttemptTracker()
    return _login_attempts[email]

async def _reap_stale_login_attempts():
    """Background task: remove login attempt entries older than TTL."""
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        now = time.monotonic()
        stale = [k for k, v in _login_attempts.items()
                 if now - v.last_attempt > _LOGIN_ATTEMPT_TTL]
        for k in stale:
            del _login_attempts[k]
```

Add `last_attempt` field to `LoginAttemptTracker` (or derive from max of attempts list).

### Step 2: Add background reaper for `_user_cache` (auth.py)

```python
# auth.py -- add after _user_cache definition

_MAX_USER_CACHE_SIZE = 1000

async def _reap_expired_user_cache():
    """Background task: remove expired user cache entries."""
    while True:
        await asyncio.sleep(600)  # Run every 10 minutes
        now = time.time()
        expired = [k for k, (_, exp) in _user_cache.items() if now > exp]
        for k in expired:
            del _user_cache[k]
        # If still over limit after TTL eviction, remove oldest
        if len(_user_cache) > _MAX_USER_CACHE_SIZE:
            sorted_keys = sorted(_user_cache, key=lambda k: _user_cache[k][1])
            for k in sorted_keys[:len(_user_cache) - _MAX_USER_CACHE_SIZE]:
                del _user_cache[k]
```

### Step 3: Remove dead code `_A2_INBOX_CACHE` (tool_runtime.py)

Delete the unused `_A2_INBOX_CACHE` declaration at line 520. It's dead code with no references.

### Step 4: Add lock for `_calibration_ratio` (context_manager.py)

```python
# context_manager.py -- add lock for thread safety
import asyncio

_calibration_lock = asyncio.Lock()

async def calibrate_from_usage(actual_tokens: int, estimated_tokens: int):
    """Update calibration ratio with lock for concurrent safety."""
    async with _calibration_lock:
        global _calibration_ratio
        if estimated_tokens > 0:
            actual_ratio = actual_tokens / estimated_tokens
            _calibration_ratio = 0.9 * _calibration_ratio + 0.1 * actual_ratio
```

### Step 5: Start background reapers on server startup (agent_zero_server.py)

In the server startup event, launch the background cleanup tasks:

```python
# agent_zero_server.py -- in startup event handler
from auth import _reap_stale_login_attempts, _reap_expired_user_cache

@app.on_event("startup")
async def startup():
    # ... existing startup code ...
    asyncio.create_task(_reap_stale_login_attempts())
    asyncio.create_task(_reap_expired_user_cache())
```

## Test Specifications

### test_unbounded_state_growth.py

1. **test_login_attempts_bounded** -- Add `_MAX_LOGIN_ATTEMPT_ENTRIES + 100` unique emails to `_login_attempts`. Verify size never exceeds `_MAX_LOGIN_ATTEMPT_ENTRIES`.

2. **test_login_attempts_evicts_oldest** -- Add entries with known timestamps. When capacity is reached, verify the oldest entry is evicted.

3. **test_login_reaper_removes_stale** -- Add entries with `last_attempt` older than TTL. Run `_reap_stale_login_attempts()` once. Verify stale entries removed, fresh entries retained.

4. **test_login_reaper_preserves_fresh** -- Add entries with recent `last_attempt`. Run reaper. Verify all entries remain.

5. **test_user_cache_reaper_removes_expired** -- Add entries with past expiry times. Run `_reap_expired_user_cache()` once. Verify expired entries removed.

6. **test_user_cache_reaper_preserves_valid** -- Add entries with future expiry times. Run reaper. Verify all entries remain.

7. **test_user_cache_size_capped** -- Add more than `_MAX_USER_CACHE_SIZE` entries (all valid). Run reaper. Verify oldest evicted down to cap.

8. **test_calibration_ratio_lock** -- Run `calibrate_from_usage()` concurrently from multiple tasks. Verify `_calibration_ratio` converges to a reasonable value (no corruption).

9. **test_a2_inbox_cache_removed** -- Verify `_A2_INBOX_CACHE` is no longer defined in tool_runtime.py (dead code removed).

10. **test_login_attempt_attack_simulation** -- Simulate 50,000 failed logins from unique emails. Verify memory usage stays bounded (dict size <= max).

## Estimated Impact

- **Security**: Mitigates denial-of-service via login attempt flooding (currently unbounded memory growth vector)
- **Reliability**: Prevents gradual memory consumption growth in long-running production deployments (days/weeks)
- **Correctness**: Lock on `_calibration_ratio` prevents subtle token estimation errors under concurrent load
- **Maintenance**: Removing dead `_A2_INBOX_CACHE` code reduces confusion
- **Affected modules**: auth.py, tool_runtime.py, context_manager.py, agent_zero_server.py (startup)
