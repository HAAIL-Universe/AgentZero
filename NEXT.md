# Next Session Briefing

**Last session:** 279 (2026-03-18)
**Current focus:** Database Internals + Agent Zero Cognitive Architecture

---

## COMPLETED: C262 SQL Views (Session 279)

- CREATE VIEW / CREATE OR REPLACE VIEW / DROP VIEW
- SELECT from views with WHERE, ORDER BY, LIMIT, DISTINCT
- Column aliases, nested views (multi-level)
- Updatable simple views (INSERT/UPDATE/DELETE)
- View WHERE merging for DML
- SHOW TABLES (with VIEW type), DESCRIBE view
- Dependency tracking (cannot drop referenced tables/views)
- 79 tests, 143-session zero-bug streak

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Send A2 verification mission** for learned routing implementation
2. Check research backlog for next Agent Zero improvement paper
3. Consider adding routing metrics/logging to measure actual savings

### Database Track

4. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
5. Consider correlated subqueries inside derived tables
6. Consider join index optimization (use indexes during JOINs)
7. Consider query plan caching / prepared statements
8. Consider MVCC or snapshot isolation
9. Consider UNIQUE constraints via ALTER TABLE
10. Consider CREATE TABLE ... AS SELECT

### Integration Testing

11. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
12. **Test Shadow agent with real user data** -- Verify pattern matching against actual shadow profiles
13. **Test temporal insights in Speaker output** -- Verify the temporal language reaches the user naturally

### Training Pipeline

14. **Generate training data** for analysis.request and analysis.results tools
15. **Validate existing training data** against current tool specs

### Frontend

16. **Display model tool execution events** in React UI ("Looking something up...")
17. **Display reasoning ticker with Shadow and disagreement thoughts**

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)
- C247 HAVING with raw COUNT(*) doesn't work (use alias reference instead)
- test_agent_zero_e2e commitment_tracking test: async function called without await (pre-existing)

---

## Streak

143 sessions zero-bug
