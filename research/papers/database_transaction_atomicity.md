---
topic: Database Transaction Atomicity
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T00:00:00Z
---

# Database Transaction Atomicity

## Problem Statement

Agent Zero's database layer (database.py) uses asyncpg for PostgreSQL access but performs **zero explicit transactions**. Multi-statement operations execute as separate auto-committed statements, creating data inconsistency risks. A failure between the first and second INSERT in `create_commitment()` leaves a commitment without an audit event. A concurrent streak update in `_update_streak()` creates a classic read-modify-write race condition.

## Current State in Agent Zero

**File: `agent_zero/database.py`**

### Non-Atomic Multi-Statement Operations

1. **`create_commitment()` (lines 410-424)** -- Two separate INSERT statements:
   - Line 410-418: `INSERT INTO commitments ... RETURNING id` via `fetch_one()`
   - Line 420-424: `INSERT INTO commitment_events (commitment_id, ...) VALUES (...)` via `execute()`
   - **Risk**: If second INSERT fails, commitment exists without "created" event. Audit trail broken.

2. **`update_commitment_status()` (lines 464-484)** -- Three separate operations:
   - Line 464-468: `UPDATE commitments SET status = $1`
   - Line 480: `log_commitment_event()` (separate INSERT)
   - Line 483: Conditional `_update_streak()` (separate SELECT + UPDATE)
   - **Risk**: Status can be "completed" without corresponding event logged. Streak update is a third non-atomic step.

3. **`_update_streak()` (lines 489-510)** -- Read-modify-write race:
   - Line 489-492: `SELECT cadence, streak_current, streak_best FROM commitments WHERE id = $1`
   - Lines 493-505: Compute new streak values in Python
   - Line 506-510: `UPDATE commitments SET streak_current = $1, streak_best = $2`
   - **Risk**: If two coroutines update the same commitment concurrently, the second overwrites the first's changes (lost update anomaly).

### Connection Management (lines 293-382)

All query helpers (`execute()`, `fetch_one()`, `fetch_all()`) acquire a **separate connection per call** via `pool.acquire()`. This means multi-statement operations each get a different connection with auto-commit -- there is no shared connection or transaction context.

### Transaction Usage

**Zero transaction blocks exist in the entire file.** Searched for `transaction()`, `BEGIN`, `COMMIT`, `SAVEPOINT` -- none found.

## Industry Standard / Research Findings

### asyncpg Transaction API (Official Documentation)

asyncpg provides `Connection.transaction()` as an async context manager that automatically starts, commits, and rolls back transactions. Nested calls create savepoints automatically.

```python
async with conn.transaction():
    await conn.execute(...)
    await conn.execute(...)
    # Both succeed or both fail
```

**Source**: [asyncpg API Reference](https://magicstack.github.io/asyncpg/current/api/index.html) -- MagicStack/asyncpg official documentation.

### Async Database Access Patterns for AI Systems (2026)

Dasroot (2026) documents patterns for async database access in AI-powered services. Key recommendation: "Every multi-statement write path must be wrapped in an explicit transaction, even in asyncpg where auto-commit is the default. The cost of a transaction wrapper is negligible compared to the cost of data inconsistency."

**Source**: [Async Database Access Patterns for AI Systems](https://dasroot.net/posts/2026/03/async-database-access-patterns-ai-systems/) -- dasroot.net, March 2026.

### Unit of Work Pattern for Clean Database Transactions (2025)

DentedLogic (2025) describes the Unit of Work pattern: "Database work needs to be atomic (all-or-nothing). The Unit of Work pattern provides automatic cleanup and separation between business logic and persistence mechanics." The pattern wraps related operations in a single transaction boundary.

**Source**: [Stop Writing try/except Hell: Clean Database Transactions with SQLAlchemy](https://dev.to/dentedlogic/stop-writing-tryexcept-hell-clean-database-transactions-with-sqlalchemy-with-the-unit-of-work-hjk) -- DEV Community, 2025.

### PostgreSQL Advisory Locks for Read-Modify-Write

The `_update_streak()` pattern (SELECT then UPDATE) is a textbook lost-update race condition. PostgreSQL provides `SELECT ... FOR UPDATE` to lock the row during the transaction, preventing concurrent modifications. This is the standard solution for read-modify-write cycles.

**Source**: [PostgreSQL Documentation - Explicit Locking](https://www.postgresql.org/docs/current/explicit-locking.html) -- PostgreSQL official docs.

### asyncpg executemany() Atomicity

asyncpg's `executemany()` is documented as atomic -- all executions succeed or none do. This should be used for batch inserts where applicable.

**Source**: [asyncpg Usage Documentation](https://magicstack.github.io/asyncpg/current/usage.html) -- MagicStack/asyncpg.

## Proposed Implementation

### Step 1: Add transaction helper to database.py

Add a new helper function that exposes the connection with a transaction:

```python
# database.py -- new helper (after existing helpers, ~line 385)
from contextlib import asynccontextmanager

@asynccontextmanager
async def transaction():
    """Yield a connection inside an explicit transaction.

    Usage:
        async with transaction() as conn:
            await conn.execute(...)
            await conn.execute(...)
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            yield conn
```

### Step 2: Wrap create_commitment() in transaction

```python
async def create_commitment(user_id, title, description, cadence, weight, due_hint, parent_id, created_from):
    uid = str(user_id)
    pid = str(parent_id) if parent_id else None
    async with transaction() as conn:
        row = await conn.fetchrow(
            """INSERT INTO commitments
                   (user_id, title, description, cadence, weight, due_hint, parent_id, created_from)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
               RETURNING id, title, description, cadence, weight, status, due_hint,
                         parent_id, streak_current, streak_best, created_from,
                         created_at, updated_at""",
            uid, title, description, cadence, weight, due_hint, pid, created_from,
        )
        await conn.execute(
            """INSERT INTO commitment_events (commitment_id, user_id, event_type, payload)
               VALUES ($1, $2, 'created', $3::jsonb)""",
            row["id"], uid, json.dumps({"title": title, "cadence": cadence}),
        )
    return dict(row)
```

### Step 3: Wrap update_commitment_status() in transaction

```python
async def update_commitment_status(user_id, commitment_id, new_status):
    uid = str(user_id)
    cid = str(commitment_id)
    async with transaction() as conn:
        result = await conn.execute(
            """UPDATE commitments SET status = $1, updated_at = now()
               WHERE id = $2 AND user_id = $3""",
            new_status, cid, uid,
        )
        if result == "UPDATE 0":
            return None
        event_type = {"active": "reactivated", "completed": "completed",
                      "missed": "missed", "paused": "paused"}.get(new_status, new_status)
        await conn.execute(
            """INSERT INTO commitment_events (commitment_id, user_id, event_type)
               VALUES ($1, $2, $3)""",
            cid, uid, event_type,
        )
        if new_status in ("completed", "missed"):
            await _update_streak_atomic(conn, uid, cid, new_status)
    # Fetch updated row after transaction
    return await fetch_one("SELECT * FROM commitments WHERE id = $1", cid)
```

### Step 4: Fix _update_streak() with SELECT FOR UPDATE

```python
async def _update_streak_atomic(conn, user_id, commitment_id, status):
    """Update streak within an existing transaction using row-level lock."""
    row = await conn.fetchrow(
        "SELECT cadence, streak_current, streak_best FROM commitments WHERE id = $1 FOR UPDATE",
        commitment_id,
    )
    if not row:
        return
    current = row["streak_current"]
    best = row["streak_best"]
    if status == "completed":
        current += 1
        best = max(best, current)
    else:
        current = 0
    await conn.execute(
        """UPDATE commitments SET streak_current = $1, streak_best = $2, updated_at = now()
           WHERE id = $3""",
        current, best, commitment_id,
    )
```

### Step 5: Update internal callers

Any function that currently calls `_update_streak()` standalone must be updated to use the transactional version. The old `_update_streak()` can be kept as a wrapper that creates its own transaction for backward compatibility:

```python
async def _update_streak(user_id, commitment_id, status):
    """Standalone streak update (creates own transaction)."""
    async with transaction() as conn:
        await _update_streak_atomic(conn, user_id, commitment_id, status)
```

## Test Specifications

### test_database_transaction_atomicity.py

1. **test_create_commitment_atomic** -- Mock the second INSERT to raise. Verify no commitment row exists (transaction rolled back).

2. **test_update_status_atomic** -- Mock event INSERT to raise after status UPDATE. Verify status unchanged (rolled back).

3. **test_streak_update_for_update_lock** -- Simulate concurrent streak updates. Run two `_update_streak_atomic()` calls for the same commitment in overlapping transactions. Verify final streak is correct (no lost updates).

4. **test_transaction_helper_commits** -- Use `transaction()` context manager, INSERT a row, exit cleanly. Verify row exists.

5. **test_transaction_helper_rollback** -- Use `transaction()` context manager, INSERT a row, raise exception. Verify row does NOT exist.

6. **test_create_commitment_event_created** -- Create commitment via `create_commitment()`. Verify both commitment row and event row exist.

7. **test_update_status_event_logged** -- Update commitment status. Verify event row matches new status.

8. **test_streak_completed_increments** -- Complete a commitment. Verify `streak_current` incremented and `streak_best` updated if applicable.

9. **test_streak_missed_resets** -- Miss a commitment. Verify `streak_current` reset to 0, `streak_best` unchanged.

10. **test_nested_transaction_savepoint** -- Verify nested `async with conn.transaction()` creates a savepoint that can roll back independently.

## Estimated Impact

- **Data integrity**: Eliminates the possibility of orphaned commitments without events, or status updates without audit trails
- **Concurrency safety**: `SELECT FOR UPDATE` prevents lost-update anomaly on streak counters
- **Production reliability**: Prevents subtle data corruption that accumulates over time in long-running deployments
- **Affected users**: All users creating/updating commitments (core workflow)
- **Risk**: Low -- asyncpg transactions are well-tested; changes are additive (new helper + wrapping existing code)
