# Next Session Briefing

**Last session:** 293 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 293

### Async Safety and Race Condition Prevention (19 tests)
- New `agent_zero/session_state.py` -- SessionState with asyncio.Lock-protected mutations
- Per-session lock in websocket_chat and websocket_voice (turn + finalization)
- Debug assertions via AGENT_ZERO_DEBUG_CONCURRENCY env var
- No behavioral change -- purely defensive for multi-user readiness

### Consolidation Threshold Validation (24 tests)
- 3 new Agent ZeroConfig fields: temporal_half_life_days, quality_gate_threshold, outcome_decay_margin
- Replaced 3 hardcoded constants with config-sourced values
- Added _compute_clustering_quality (silhouette telemetry)
- Added THRESHOLD_RATIONALE research documentation dict
- All 42 existing consolidator tests continue to pass

**Total: 43 new tests, 156-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Proactive Conversation Starters** (research/papers/proactive_conversation_starters.md) -- SSE, JITAI triggers (MED, new)
2. **Memory Recall Transparency** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
3. **Runtime Observability Layer** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent
4. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
5. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
6. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
7. **Monitor skip_safe flags** in production to validate VOI feedback loop
8. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility

### New Research Papers Available (MED priority)
9. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
10. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
11. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Database Track

12. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
13. Consider window functions (ROW_NUMBER, RANK, etc.)
14. Consider correlated subqueries inside derived tables
15. Consider join index optimization (use indexes during JOINs)

### Integration Testing

16. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
17. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
18. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
19. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat
20. **Test bandit weight learning end-to-end** -- Verify weights update after acted/ignored outcomes

### Frontend

21. **Display model tool execution events** in React UI ("Looking something up...")
22. **Display reasoning ticker with Shadow and disagreement thoughts**

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

156 sessions zero-bug
