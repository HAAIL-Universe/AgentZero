# Next Session Briefing

**Last session:** 291 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 291

### Resilience Layer Integration into database.py (19 tests)
- Wrapped 4 core DB functions with `resilient_call` + `db_circuit`
- `_raw_*` internal functions for migrations/health check bypass
- fetch_all fallback=list, fetch_one/val fallback=lambda:None, execute no fallback
- Added `health_check()` and `get_db_circuit_state()` for observability
- All 30+ call sites automatically protected

### Agent Weight Learning via Thompson Sampling (22 tests)
- New module: `agent_zero/agent_bandit.py` -- Beta(alpha,beta) per (user, topic, agent)
- `sample_weights()`, `update_agent_outcome()`, `get_blend_factor()`
- New DB table: `agent_bandit_params`
- Integrated into `select_agents_for_turn` (bandit_weights param, +/-0.15 modulation)
- Integrated into outcome resolution in agent_zero_server.py (update on acted/ignored)
- Graceful cold start: blend_factor=0 with no data, heuristic weights dominate

**Total: 723 agent_zero tests pass (41 new), 154-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Send A2 verification mission** for session 291 (resilience integration + bandit)
2. **Structured Logging Replacement** (research/papers/structured_logging.md) -- replace print/pass with JSON logging (MED)
3. **Memory Recall Transparency** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
4. **Runtime Observability Layer** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
7. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
8. Consider adding quality gate metrics to consolidation learning loop
9. **Monitor skip_safe flags** in production to validate VOI feedback loop

### New Research Papers Available (MED priority)
10. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
11. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
12. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Database Track

13. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
14. Consider window functions (ROW_NUMBER, RANK, etc.)
15. Consider correlated subqueries inside derived tables
16. Consider join index optimization (use indexes during JOINs)

### Integration Testing

17. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
18. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
19. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
20. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat
21. **Test bandit weight learning end-to-end** -- Verify weights update after acted/ignored outcomes

### Frontend

22. **Display model tool execution events** in React UI ("Looking something up...")
23. **Display reasoning ticker with Shadow and disagreement thoughts**

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)
- C247 HAVING with raw COUNT(*) doesn't work (use alias reference instead)
- test_agent_zero_e2e commitment_tracking test: async function called without await (pre-existing)
- test_cognitive_agents::test_model_backed_agent_uses_prompt_body_when_loaded: pre-existing assertion mismatch
- test_agent_zero_turn_paths: user=None crash on save_shadow (pre-existing)

---

## Streak

154 sessions zero-bug
