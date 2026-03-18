---
topic: Silent Exception Data Loss and Destructive Operation Guards
status: implemented
priority: high
estimated_complexity: small
researched_at: 2026-03-18T22:30:00Z
---

# Silent Exception Data Loss and Destructive Operation Guards

## Problem Statement

Agent Zero has two data integrity risks: (1) bare `except Exception: pass` blocks silently discard database write failures for user commitments, causing data loss with no trace; (2) the `DELETE /api/user/clear-data` endpoint permanently destroys all user data with a single unauthenticated-beyond-JWT request, no re-authentication, no confirmation, and no recovery window. Together, these mean user data can be silently lost (commitments) or irreversibly destroyed (clear-data) without safeguards.

## Current State in Agent Zero

### Silent Exception Swallowing

**behavioural_shadow.py:439-440** -- `_update_commitment_tracking`:
```python
try:
    await create_commitment(
        user_id=user_id,
        title=commitment_text,
        due_hint=_extract_due_hint(content),
        created_from="conversation",
    )
except Exception:
    pass  # Shadow still tracks it even if DB write fails
```
If `create_commitment` raises (network timeout, constraint violation, connection pool exhausted), the commitment is silently discarded. The user said "I commit to X", Agent Zero acknowledged it, but the DB never recorded it. No log entry, no metric, no retry.

**behavioural_shadow.py:488-489** -- `_sync_commitment_status_to_db`:
```python
try:
    rows = await list_commitments(user_id, status="active")
    for row in rows:
        if _same_commitment(commitment_text, row.get("title", "")):
            await db_update_commitment_status(user_id, str(row["id"]), new_status)
            return
except Exception:
    pass
```
Complete silent failure. A status update (e.g., marking a commitment as completed) silently does nothing if DB is temporarily unavailable. The shadow state diverges from DB state permanently.

**Additional silent-swallow sites** (lower severity, but should be audited):
| File | Lines | Context |
|------|-------|---------|
| agent_zero_server.py | 210, 419, 462, 1170, 1175, 2105, 2117, 2169, 2391, 3294, 3359, 3490 | Various handlers |
| proactive_evaluator.py | 105, 109, 141 | DB fetch fallbacks |
| agent_zero_telegram/app.py | 217 | Check-in loop (no logging) |

### Destructive Operation Without Guards

**agent_zero_server.py:909** -- `DELETE /api/user/clear-data`:
```python
@app.delete("/api/user/clear-data")
async def clear_user_data(user: dict = Depends(get_current_user)):
    user_id = uuid.UUID(user["user_id"])
    # Deletes across 14 tables in sequence
    # No re-authentication required
    # No confirmation step
    # No soft-delete or recovery window
    # Logged only via print() on line 946
```
A stolen JWT (e.g., from browser history, proxy logs, or the query-parameter exposure noted in jwt_security_hardening.md) allows permanent, irrecoverable deletion of all user data. There is no "undo" and no grace period.

## Industry Standard / Research Findings

### Silent Exception Antipattern

The Python community consensus, codified in PEP 20 ("Errors should never pass silently"), is that bare `except: pass` is one of the most dangerous antipatterns. In async contexts, it's even worse: exceptions can be garbage-collected without any visible output, and the event loop may complete parent functions while child tasks silently fail.

- **Source:** [index.dev -- How to Avoid Silent Failures in Python](https://www.index.dev/blog/avoid-silent-failures-python)
- **Source:** [charlax/antipatterns -- Error Handling Antipatterns](https://github.com/charlax/antipatterns/blob/master/error-handling-antipatterns.md)
- **Source:** [Pybites -- Python Errors Should Not Pass Silently](https://pybit.es/articles/python-errors-should-not-pass-silently/)

Production best practice: log at WARNING level minimum, emit a metric counter, and optionally retry with backoff for transient failures. Never silently discard a user-data write.

### Destructive API Endpoint Protection

OWASP REST Security Cheat Sheet recommends: "For particularly sensitive destructive operations, require re-authentication or step-up authentication." The standard pattern is a soft-delete with a configurable grace period (24-72 hours) before permanent erasure, plus an audit log entry.

- **Source:** [OWASP REST Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html)
- **Source:** [Azure API Management Soft-Delete](https://learn.microsoft.com/en-us/azure/api-management/soft-delete)
- **Source:** [Abstract API -- DELETE Method Best Practices](https://www.abstractapi.com/guides/api-glossary/delete-method-in-apis)

GitHub, Google, and Stripe all require password re-entry or 2FA for account deletion. Azure uses a 48-hour soft-delete window with purge protection.

## Proposed Implementation

### Part 1: Fix silent exception swallowing (behavioural_shadow.py)

**behavioural_shadow.py:439-440:**
```python
# BEFORE
except Exception:
    pass  # Shadow still tracks it even if DB write fails

# AFTER
except Exception as e:
    logger.warning(
        "commitment_create_failed",
        user_id=str(user_id),
        commitment=commitment_text[:100],
        error=str(e),
    )
```

**behavioural_shadow.py:488-489:**
```python
# BEFORE
except Exception:
    pass

# AFTER
except Exception as e:
    logger.warning(
        "commitment_status_sync_failed",
        user_id=str(user_id),
        commitment=commitment_text[:100],
        target_status=new_status,
        error=str(e),
    )
```

Add at top of file:
```python
from logging_config import get_logger
logger = get_logger("behavioural_shadow")
```

### Part 2: Audit remaining silent-swallow sites

For each `except Exception: pass` in agent_zero_server.py and other modules:
1. If it guards a non-critical fallback (e.g., analytics, metrics): add `logger.debug()`
2. If it guards a data write (commitments, episodes, outcomes): add `logger.warning()`
3. If it guards a user-facing operation: add `logger.error()` and consider a user-visible fallback message

Priority order for the agent_zero_server.py sites:
- Lines 2105, 2117, 2169 (likely data writes) -- HIGH
- Lines 3294, 3359, 3490 (WebSocket handlers) -- MEDIUM
- Lines 210, 419, 462, 1170, 1175, 2391 (startup/config) -- LOW

### Part 3: Destructive operation guard on clear_user_data

**Step 1: Require password re-confirmation**

Add a request body to the DELETE endpoint:
```python
class ClearDataRequest(BaseModel):
    password: str

@app.delete("/api/user/clear-data")
async def clear_user_data(
    body: ClearDataRequest,
    user: dict = Depends(get_current_user)
):
    # Verify password before proceeding
    user_record = await get_user_by_id(user["user_id"])
    if not verify_password(body.password, user_record["password_hash"]):
        raise HTTPException(status_code=403, detail="Password verification failed")
    ...
```

**Step 2: Soft-delete with 48-hour recovery window**

Instead of immediately deleting rows, mark them:
```python
# Add column: deleted_at TIMESTAMP NULL DEFAULT NULL
# DELETE becomes UPDATE:
await db.execute(
    "UPDATE commitments SET deleted_at = NOW() WHERE user_id = $1 AND deleted_at IS NULL",
    user_id
)
# ... repeat for other tables

# Add a cleanup cron job (or daily task) that purges rows where deleted_at < NOW() - INTERVAL '48 hours'
```

**Step 3: Structured audit log**
```python
logger.warning(
    "user_data_clear_requested",
    user_id=str(user_id),
    tables_affected=14,
    recovery_window_hours=48,
)
```

**Step 4: Rate limit**
Add to rate_limiter.py tier config:
```python
"destructive": TokenBucketConfig(rate=1/86400, capacity=1)  # 1 per day
```

### Files to modify:
- `behavioural_shadow.py` -- lines 439-440, 488-489 (add logging)
- `agent_zero_server.py` -- line 909 (add password re-auth, soft-delete, audit log)
- `database.py` -- add `deleted_at` column to schema migration, add purge function
- `auth.py` -- expose `verify_password` as a public function if not already

## Test Specifications

### Test 1: Commitment create failure is logged
```python
@pytest.mark.asyncio
async def test_commitment_create_failure_logged(caplog):
    """When create_commitment raises, a warning is logged with user_id and commitment text."""
    with patch("behavioural_shadow.create_commitment", side_effect=Exception("DB timeout")):
        await _update_commitment_tracking(user_id="test-user", content="I will exercise daily")
    assert "commitment_create_failed" in caplog.text
    assert "test-user" in caplog.text
```

### Test 2: Commitment sync failure is logged
```python
@pytest.mark.asyncio
async def test_commitment_sync_failure_logged(caplog):
    """When list_commitments raises, a warning is logged."""
    with patch("behavioural_shadow.list_commitments", side_effect=Exception("connection reset")):
        await _sync_commitment_status_to_db("test-user", "exercise", "completed")
    assert "commitment_status_sync_failed" in caplog.text
```

### Test 3: clear_user_data requires password
```python
@pytest.mark.asyncio
async def test_clear_data_requires_password():
    """DELETE /api/user/clear-data without password body returns 422."""
    response = await client.delete("/api/user/clear-data", headers=auth_headers)
    assert response.status_code == 422
```

### Test 4: clear_user_data rejects wrong password
```python
@pytest.mark.asyncio
async def test_clear_data_wrong_password():
    """DELETE /api/user/clear-data with wrong password returns 403."""
    response = await client.delete(
        "/api/user/clear-data",
        headers=auth_headers,
        json={"password": "wrong"}
    )
    assert response.status_code == 403
```

### Test 5: Soft delete marks but doesn't purge
```python
@pytest.mark.asyncio
async def test_clear_data_soft_deletes():
    """After clear-data, rows have deleted_at set but still exist in DB."""
    # Create test data
    # Call clear-data with correct password
    # Verify rows still exist with deleted_at != NULL
    # Verify API no longer returns them (filtered by deleted_at IS NULL)
```

### Test 6: No bare except-pass in data write paths
```python
def test_no_bare_except_pass_on_data_writes():
    """Static analysis: no 'except Exception: pass' in behavioural_shadow.py."""
    import ast
    with open("agent_zero/behavioural_shadow.py") as f:
        tree = ast.parse(f.read())
    # Walk AST, find ExceptHandler nodes with Pass body
    # Fail if any exist in functions containing 'create_', 'update_', 'delete_'
```

## Estimated Impact

- **Data integrity:** Commitment write failures become visible via structured logs instead of silently lost (affects every user who makes commitments during DB hiccups)
- **Security:** Account deletion requires password re-entry, preventing stolen-JWT wipe attacks
- **Recoverability:** 48-hour soft-delete window allows users to recover from accidental deletion
- **Auditability:** All destructive operations produce structured log entries for compliance
- **Debugging:** When commitment state diverges between shadow and DB, the log trail shows exactly when and why
