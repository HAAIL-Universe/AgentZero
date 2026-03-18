# Next Session Briefing

**Last session:** 298 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 298

### Runtime Observability Layer (34 tests)
- agent_zero/observability.py: TurnEvent, observe_turn, timed, MetricsBuffer
- Uses contextvars for async safety, integrates with logging_config.py

### API Rate Limiting (20 tests)
- agent_zero/rate_limiter.py: TokenBucket, RateLimiter (async-safe, auto-cleanup)
- 8 new config fields for 4 tiers (auth, api, inference, model_load)

### WebSocket Input Validation (28 tests)
- agent_zero/ws_messages.py: Pydantic models, validate_ws_message, MessageRateLimiter
- 4 new config fields for WS limits

**Total: 82 new tests, 160-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
2. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
3. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
4. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
5. **Monitor skip_safe flags** in production to validate VOI feedback loop
6. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility

### New Research Papers Available (ready_for_implementation)
7. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
8. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
9. **Database Query Optimization** (research/papers/database_query_optimization.md) -- HIGH, medium

### Research Papers (MED priority)
10. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
11. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
12. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration: Wire Up New Modules
13. **Wire observability.py into agent_zero_server.py** -- wrap _run_conversation_turn with observe_turn
14. **Wire rate_limiter.py into agent_zero_server.py** -- add _check_rate_limit_ip/user helpers
15. **Wire ws_messages.py into /ws/chat and /ws/voice** -- replace raw json.loads with validate_ws_message

### Database Track
16. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
17. Consider window functions (ROW_NUMBER, RANK, etc.)

### Integration Testing
18. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
19. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
20. **Test quality gates end-to-end** -- Verify flags appear in reasoning events

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

160 sessions zero-bug
