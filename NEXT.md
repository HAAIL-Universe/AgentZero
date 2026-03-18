# Next Session Briefing

**Last session:** 288 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Two Research Papers (Session 288)

### Dynamic Context Budgeting (22 tests)
- `AGENT_CONTEXT_BUDGETS` + `AGENT_FIELD_PRIORITIES` in cognitive_agents.py
- 5-phase trimming with per-agent budgets (2000-10000 chars)
- Per-agent response token limits (300-700 tokens)
- Shadow gets 10K chars (was 6K), Clarifier gets 2K (was 6K)

### Consolidation Rule Quality Gate (30 tests)
- `compute_rule_quality` + `_generate_quality_warnings` in consolidator.py
- `get_relevant_rules` now filters by quality (structural, outcome, data, drift)
- Quality metadata stored on rules during consolidation
- Backward compatible via `apply_quality_gate=True` default

**203 total agent_zero tests verified, 151-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Check A2 reply** for context budgeting + quality gate verification (9f1e3217)
2. **Integrate resilience.py into agent_zero_server.py** -- wrap critical DB calls with resilient_call + db_circuit, replace `except Exception: pass` blocks
3. **Memory Recall Transparency paper** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
4. **Runtime Observability Layer paper** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent, structured logging (MED, ready)
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver for conflict-free scheduling (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in cognitive pipeline (HIGH, large)
7. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
8. Consider adding quality gate metrics to consolidation learning loop
9. **Monitor skip_safe flags** in production to validate VOI feedback loop
10. **Test stage classification end-to-end** -- verify stage events appear in reasoning runs

### New Research Papers Available (MED priority)
11. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
12. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
13. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)
14. **Dynamic Context Budgeting** -- now implemented, can be calibrated with production data

### Database Track

15. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
16. Consider window functions (ROW_NUMBER, RANK, etc.)
17. Consider correlated subqueries inside derived tables
18. Consider join index optimization (use indexes during JOINs)

### Integration Testing

19. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
20. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
21. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
22. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat

### Frontend

23. **Display model tool execution events** in React UI ("Looking something up...")
24. **Display reasoning ticker with Shadow and disagreement thoughts**

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)
- C247 HAVING with raw COUNT(*) doesn't work (use alias reference instead)
- test_agent_zero_e2e commitment_tracking test: async function called without await (pre-existing)
- test_cognitive_agents::test_model_backed_agent_uses_prompt_body_when_loaded: pre-existing assertion mismatch (checks for "You are Othello." but prompt format changed)

---

## Streak

151 sessions zero-bug
