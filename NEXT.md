# Next Session Briefing

**Last session:** 282 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture + Database Internals

---

## COMPLETED: Cost-Aware Agent Activation + Speaker Quality Gates (Session 282)

### Cost-Aware Agent Activation
- VOI gating: suppresses agents with historically no signal (saves LLM calls)
- Adaptive threshold: 0.35-0.50 based on consolidation confidence
- Skip tracking in blackboard + skip impact monitoring in consolidator
- 28 tests, A2 verification sent (8943e150)

### Speaker Quality Gates
- evaluate_speaker_quality() in guardrails.py -- deterministic quality check (<1ms)
- Anti-patterns, item count, MITI R:Q ratio, style mismatch, emotional register
- Speaker context now includes intervention_effectiveness from consolidated rules
- Speaker prompt updated with Learned Effectiveness section
- 22 tests, A2 verification sent (d78b755b)

**112 total agent_zero tests pass, 146-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Check A2 replies** for cost-aware activation (8943e150) and speaker quality gates (d78b755b)
2. **Monitor skip_safe flags** in production to validate VOI feedback loop
3. **Monitor quality gate flags** to measure anti-pattern frequency
4. Check research backlog for next paper (tool activity log UI is next unresearched high-priority)
5. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
6. Consider adding quality gate metrics to consolidation learning loop

### Database Track

7. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
8. Consider window functions (ROW_NUMBER, RANK, etc.)
9. Consider correlated subqueries inside derived tables
10. Consider join index optimization (use indexes during JOINs)

### Integration Testing

11. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
12. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
13. **Test quality gates end-to-end** -- Verify flags appear in reasoning events

### Frontend

14. **Display model tool execution events** in React UI ("Looking something up...")
15. **Display reasoning ticker with Shadow and disagreement thoughts**

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
