# Next Session Briefing

**Last session:** 297 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 297

### Health and Readiness Probes (14 tests)
- /health/live, /health/ready, /health/startup endpoints
- Startup tracking flags, DB deep check with timeout, circuit breaker state
- Config: health_check_timeout (2.0s default)

### Memory Recall Transparency (9 tests)
- Backend: _emit_memory_context() WS message after retrieval + memory.recall tool
- Frontend: MemoryContextLog.tsx (purple collapsible cards, reason chips, scores)
- State: memoryContextByTurn in App.tsx, positioned above ToolActivityLog

**Total: 23 new tests, 159-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Runtime Observability Layer** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent
2. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
3. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
4. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
5. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
6. **Monitor skip_safe flags** in production to validate VOI feedback loop
7. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility

### New Research Papers Available (ready_for_implementation)
8. **API Rate Limiting** (research/papers/api_rate_limiting.md) -- HIGH, medium
9. **WebSocket Input Validation** (research/papers/websocket_input_validation.md) -- HIGH, medium
10. **Database Query Optimization** (research/papers/database_query_optimization.md) -- HIGH, medium

### Research Papers (MED priority)
11. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
12. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
13. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Database Track

14. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
15. Consider window functions (ROW_NUMBER, RANK, etc.)
16. Consider correlated subqueries inside derived tables
17. Consider join index optimization (use indexes during JOINs)

### Integration Testing

18. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
19. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
20. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
21. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat
22. **Test bandit weight learning end-to-end** -- Verify weights update after acted/ignored outcomes
23. **Test proactive messages end-to-end** -- Verify triggers fire and render in chat during live session
24. **Test memory transparency end-to-end** -- Verify purple memory cards appear above tool cards

### Frontend

25. **Display model tool execution events** in React UI ("Looking something up...")
26. **Display reasoning ticker with Shadow and disagreement thoughts**

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

159 sessions zero-bug
