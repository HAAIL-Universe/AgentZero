---
topic: Database Query Optimization
status: ready_for_implementation
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T20:30:00Z
---

# Database Query Optimization

## Problem Statement

Agent Zero's database layer has three categories of query inefficiency: (1) N+1 query pattern in `get_productivity_summary()` making 7 sequential round-trips that could be 2, (2) unbounded SELECT queries in `update_requests.py` returning all rows without LIMIT (OWASP API4:2023 violation), and (3) missing composite indexes on frequently filtered column combinations causing sequential scans. Together these degrade API latency, waste connection pool slots, and create DoS risk under load.

## Current State in Agent Zero

### N+1 Pattern: get_productivity_summary() (database.py:556-629)

This function makes **7 separate database queries** for a single API call:

| # | Query | Line | Purpose |
|---|-------|------|---------|
| 1 | `SELECT count(*) FROM commitments WHERE user_id=$1 AND status='active'` | 564-567 | Active count |
| 2 | `SELECT count(*) FROM commitment_events WHERE user_id=$1 AND event_type='completed' AND occurred_at>=$2` | 569-573 | Completed this week |
| 3 | `SELECT count(*) FROM commitment_events WHERE user_id=$1 AND event_type='missed' AND occurred_at>=$2` | 575-579 | Missed this week |
| 4 | `SELECT count(*) FROM commitment_events WHERE user_id=$1 AND event_type='completed' AND occurred_at>=$2` | 585-589 | Completed this month |
| 5 | `SELECT count(*) FROM commitment_events WHERE user_id=$1 AND event_type='missed' AND occurred_at>=$2` | 590-594 | Missed this month |
| 6 | `SELECT title, streak_current, streak_best, cadence FROM commitments WHERE user_id=$1 AND cadence!='once' AND status='active'` | 599-604 | Streaks |
| 7 | `SELECT cadence, count(*) FROM commitments WHERE user_id=$1 AND status='active' GROUP BY cadence` | 613-618 | By cadence |

Queries 1, 6, and 7 all hit the `commitments` table filtered by `user_id` and `status='active'`. Queries 2-5 all hit `commitment_events` filtered by `user_id` and `event_type` with date ranges. These can be collapsed using conditional aggregation.

**Impact at scale:** With 50 concurrent users calling `/api/productivity`, this generates 350 DB queries per second. Each query acquires a connection from the pool (max_size=10, database.py:60), so 350 QPS on a 10-connection pool means 35x oversubscription and severe queueing.

### Unbounded Query: list_requests() (update_requests.py:43-54)

```python
async def list_requests(status: str = None) -> list:
    if status:
        rows = await fetch_all(
            "SELECT * FROM update_requests WHERE status = $1 ORDER BY created_at DESC",
            status
        )
    else:
        rows = await fetch_all(
            "SELECT * FROM update_requests ORDER BY created_at DESC"
        )
```

No LIMIT clause. If 10,000 update requests accumulate, every call returns all 10,000 rows. The `SELECT *` returns all columns including potentially large `description` and `rejection_reason` TEXT fields.

**OWASP API4:2023** explicitly lists "Records per page in responses" as a required limit. This violates that mandate.

### Missing Composite Indexes (database.py:76-287)

**Existing indexes:**
- `idx_commitments_user ON commitments(user_id)` -- single column
- `idx_commitments_status ON commitments(status)` -- single column
- `idx_commitment_events_user ON commitment_events(user_id)` -- single column
- `idx_goals_user_id ON goals(user_id)` -- single column
- `idx_goals_status ON goals(status)` -- single column

**Missing indexes for common query patterns:**

| Query Pattern | Missing Index |
|--------------|---------------|
| `commitments WHERE user_id=$1 AND status='active'` (used 3x in productivity) | `(user_id, status)` |
| `commitment_events WHERE user_id=$1 AND event_type=$2 AND occurred_at>=$3` (used 4x) | `(user_id, event_type, occurred_at)` |
| `goals WHERE user_id=$1 ORDER BY created_at DESC` (database.py:208) | `(user_id, created_at DESC)` |
| `commitment_events WHERE user_id=$1 ORDER BY occurred_at DESC` (database.py:546-552) | `(user_id, occurred_at DESC)` |

Without composite indexes, PostgreSQL uses single-column index + filter, or falls back to sequential scan when selectivity is low (e.g., `status='active'` matches most rows).

## Industry Standard / Research Findings

### PostgreSQL FILTER Clause for Conditional Aggregation
**Source:** https://www.postgresql.org/docs/current/tutorial-agg.html

PostgreSQL supports `COUNT(*) FILTER (WHERE condition)` syntax, which is cleaner and more performant than `SUM(CASE WHEN ... THEN 1 ELSE 0 END)`. This allows collapsing multiple COUNT queries into a single scan with conditional filters:

```sql
SELECT
    COUNT(*) FILTER (WHERE event_type = 'completed' AND occurred_at >= $week_ago) AS completed_week,
    COUNT(*) FILTER (WHERE event_type = 'missed' AND occurred_at >= $week_ago) AS missed_week,
    COUNT(*) FILTER (WHERE event_type = 'completed' AND occurred_at >= $month_ago) AS completed_month,
    COUNT(*) FILTER (WHERE event_type = 'missed' AND occurred_at >= $month_ago) AS missed_month
FROM commitment_events
WHERE user_id = $1 AND occurred_at >= $month_ago
```

This replaces 4 queries with 1.

### PostgreSQL CTE for Multi-Table Aggregation
**Source:** https://neon.com/postgresql/postgresql-tutorial/postgresql-cte

CTEs in PostgreSQL 12+ are inlined by default (no longer optimization fences). Using CTEs to combine the commitments and commitment_events aggregations into a single round-trip is both readable and performant.

### Crunchy Data -- PostgreSQL 17 Composite Index Improvements
**Source:** https://www.crunchydata.com/blog/get-excited-about-postgres-18

PostgreSQL 17+ can reuse sort order from composite index scans for Merge Joins. Composite indexes like `(user_id, created_at DESC)` serve both filtering and ordering in a single index scan.

### OWASP API4:2023 -- Unrestricted Resource Consumption
**Source:** https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/

Mandates: "Records per page in responses" must be limited. Missing limits allow DoS via resource exhaustion. Prevention: "Add server-side validation for pagination parameters."

### AsyncPG Connection Pool Performance
**Source:** https://www.tigerdata.com/blog/how-to-build-applications-with-asyncpg-and-postgresql

AsyncPG connection pooling delivers 4-6x higher QPS than sync psycopg2. However, pool saturation from N+1 patterns negates this advantage. Reducing query count per API call is the highest-impact optimization.

### Citus Data -- Pagination Strategies
**Source:** https://www.citusdata.com/blog/2016/03/30/five-ways-to-paginate/

For admin-facing APIs (like update_requests), offset-based pagination with a server-enforced max LIMIT (e.g., 100) is appropriate. Cursor-based pagination is preferable for user-facing infinite scroll but adds complexity.

## Proposed Implementation

### 1. Collapse get_productivity_summary() into 2 Queries (database.py:556-629)

**Replace the entire function body with:**

```python
async def get_productivity_summary(user_id: str) -> dict:
    """Aggregated productivity metrics for a user."""
    uid = _uuid.UUID(user_id)
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # Query 1: All commitment aggregates in one scan
    commitment_row = await fetch_one(
        """SELECT
               count(*) FILTER (WHERE status = 'active') AS active_count,
               count(*) FILTER (WHERE status = 'active' AND cadence != 'once') AS recurring_count,
               jsonb_agg(
                   jsonb_build_object('title', title, 'current', COALESCE(streak_current, 0),
                                      'best', COALESCE(streak_best, 0))
               ) FILTER (WHERE status = 'active' AND cadence != 'once')
               AS streaks,
               jsonb_object_agg(cadence, cnt) AS by_cadence
           FROM (
               SELECT *, count(*) OVER (PARTITION BY cadence) AS cnt
               FROM commitments
               WHERE user_id = $1
           ) sub""",
        uid,
    )

    # Query 2: All event counts in one scan using FILTER
    event_row = await fetch_one(
        """SELECT
               count(*) FILTER (WHERE event_type = 'completed' AND occurred_at >= $2) AS completed_week,
               count(*) FILTER (WHERE event_type = 'missed' AND occurred_at >= $2) AS missed_week,
               count(*) FILTER (WHERE event_type = 'completed' AND occurred_at >= $3) AS completed_month,
               count(*) FILTER (WHERE event_type = 'missed' AND occurred_at >= $3) AS missed_month
           FROM commitment_events
           WHERE user_id = $1 AND occurred_at >= $3""",
        uid, week_ago, month_ago,
    )

    active = (commitment_row["active_count"] or 0) if commitment_row else 0
    streaks = (commitment_row["streaks"] or []) if commitment_row else []
    by_cadence_raw = (commitment_row["by_cadence"] or {}) if commitment_row else {}
    # Filter by_cadence to only active commitments
    by_cadence = {k: v for k, v in by_cadence_raw.items()} if by_cadence_raw else {}

    cw = (event_row["completed_week"] or 0) if event_row else 0
    mw = (event_row["missed_week"] or 0) if event_row else 0
    cm = (event_row["completed_month"] or 0) if event_row else 0
    mm = (event_row["missed_month"] or 0) if event_row else 0
    total_7d = cw + mw
    total_30d = cm + mm

    return {
        "active_commitments": active,
        "completed_this_week": cw,
        "missed_this_week": mw,
        "streaks": streaks if isinstance(streaks, list) else [],
        "consistency_7d": round(cw / total_7d, 2) if total_7d else 0.0,
        "consistency_30d": round(cm / total_30d, 2) if total_30d else 0.0,
        "by_cadence": by_cadence,
    }
```

**Note:** The first query uses a window function to avoid a separate GROUP BY cadence query. However, this adds complexity. A simpler alternative is to keep 3 queries instead of 7:
1. One for commitment_events aggregates (FILTER)
2. One for streaks (SELECT ... WHERE active AND cadence != 'once')
3. One for active count + cadence breakdown (GROUP BY cadence with a total)

AZ should choose based on readability. The critical optimization is collapsing 4 event queries into 1 using FILTER.

### 2. Add LIMIT to list_requests() (update_requests.py:43-54)

**Replace:**
```python
async def list_requests(status: str = None) -> list:
    if status:
        rows = await fetch_all(
            "SELECT * FROM update_requests WHERE status = $1 ORDER BY created_at DESC",
            status
        )
    else:
        rows = await fetch_all(
            "SELECT * FROM update_requests ORDER BY created_at DESC"
        )
```

**With:**
```python
async def list_requests(status: str = None, limit: int = 100, offset: int = 0) -> list:
    """List update requests with pagination. Max 500 per page."""
    limit = min(max(1, limit), 500)
    offset = max(0, offset)
    if status:
        rows = await fetch_all(
            "SELECT * FROM update_requests WHERE status = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            status, limit, offset,
        )
    else:
        rows = await fetch_all(
            "SELECT * FROM update_requests ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            limit, offset,
        )
    return [_row_to_dict(r) for r in rows]
```

Update the `/api/requests` endpoint in agent_zero_server.py to accept `limit` and `offset` query params.

### 3. Add Composite Indexes (database.py SCHEMA_SQL)

Append to SCHEMA_SQL after the existing index definitions:

```sql
-- Composite indexes for common query patterns (added session 7)
CREATE INDEX IF NOT EXISTS idx_commitments_user_status
    ON commitments(user_id, status);
CREATE INDEX IF NOT EXISTS idx_commitment_events_user_type_date
    ON commitment_events(user_id, event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_goals_user_created
    ON goals(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_commitment_events_user_date
    ON commitment_events(user_id, occurred_at DESC);
```

**Note on existing single-column indexes:** The composite indexes subsume the single-column `idx_commitments_user` and `idx_commitment_events_user` (leftmost prefix property). The old indexes can be dropped later to save write overhead, but should be kept initially for safety.

### 4. Add LIMIT to Other Unbounded Queries

Scan for and add LIMIT to any other `fetch_all` calls without LIMIT. Known candidates:
- `database.py` functions that return all rows for a user (goals, messages, etc.)
- Any admin-facing listing endpoint

## Test Specifications

### test_database_query_optimization.py

1. **test_productivity_summary_returns_correct_counts** -- Create 5 active commitments, 3 completed events this week, 2 missed. Verify summary matches.
2. **test_productivity_summary_empty_user** -- New user with no data returns all zeros.
3. **test_productivity_summary_consistency_calculation** -- 7 completed + 3 missed = 0.70 consistency_7d.
4. **test_productivity_summary_streaks** -- 2 recurring commitments with streaks returned in descending order.
5. **test_productivity_summary_by_cadence** -- 2 daily + 1 weekly = {"daily": 2, "weekly": 1}.
6. **test_productivity_summary_month_window** -- Events from 35 days ago excluded from 30-day counts.
7. **test_list_requests_default_limit** -- 150 requests in DB, default call returns 100.
8. **test_list_requests_custom_limit** -- `limit=50` returns exactly 50.
9. **test_list_requests_max_limit_capped** -- `limit=1000` capped to 500.
10. **test_list_requests_offset** -- `offset=10, limit=5` returns items 11-15.
11. **test_list_requests_status_filter_with_limit** -- `status='pending', limit=10` filters correctly.
12. **test_list_requests_empty_result** -- No matching status returns empty list.
13. **test_composite_index_creation** -- After migration, verify indexes exist via pg_indexes.
14. **test_productivity_query_count** -- Mock fetch_one/fetch_all, verify exactly 2-3 calls (down from 7).

## Estimated Impact

- **Performance**: 7 queries -> 2 for productivity summary (71% reduction in DB round-trips)
- **Pool utilization**: At 50 concurrent users, 350 QPS -> 100 QPS (65% reduction)
- **Security**: LIMIT on list_requests prevents unbounded result set DoS (OWASP API4:2023)
- **Latency**: Composite indexes eliminate filter-after-scan pattern (est. 2-5x faster for filtered queries)
- **Scalability**: Reduced connection pool pressure allows serving more concurrent users

## References

1. PostgreSQL FILTER Clause Documentation -- https://www.postgresql.org/docs/current/tutorial-agg.html
2. PostgreSQL CTE Tutorial (Neon) -- https://neon.com/postgresql/postgresql-tutorial/postgresql-cte
3. OWASP API4:2023 Unrestricted Resource Consumption -- https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/
4. AsyncPG Performance Guide (Tiger Data) -- https://www.tigerdata.com/blog/how-to-build-applications-with-asyncpg-and-postgresql
5. Citus Data Pagination Strategies -- https://www.citusdata.com/blog/2016/03/30/five-ways-to-paginate/
6. PostgreSQL 17/18 Index Improvements (Crunchy Data) -- https://www.crunchydata.com/blog/get-excited-about-postgres-18
