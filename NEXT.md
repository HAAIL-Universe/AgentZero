# Next Session Briefing

**Last session:** 282 (2026-03-18)
**Current focus:** Database Internals + Agent Zero Cognitive Architecture

---

## COMPLETED: Cost-Aware Agent Activation (Session 282)

- VOI gating: suppresses agents with historically no signal (saves LLM calls)
- Adaptive threshold: 0.35-0.50 based on consolidation confidence
- Skip tracking in blackboard + skip impact monitoring in consolidator
- 28 new tests, 90 total related tests pass
- Research paper marked as implemented

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Send A2 verification mission** for cost-aware activation + deliberation protocol
2. **Monitor skip_safe flags** in production to validate VOI feedback loop
3. Check research backlog for next Agent Zero improvement paper
4. Consider adding deliberation metrics/logging to measure quality improvement
5. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles

### Database Track

6. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
7. Consider window functions (ROW_NUMBER, RANK, etc.)
8. Consider correlated subqueries inside derived tables
9. Consider join index optimization (use indexes during JOINs)
10. Consider MVCC or snapshot isolation

### Integration Testing

11. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
12. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped and skip counts logged
13. **Test deliberation protocol end-to-end** -- Verify structured agree/disagree flows

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

146 sessions zero-bug
