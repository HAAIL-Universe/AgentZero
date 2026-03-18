# Next Session Briefing

**Last session:** 299 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 299

### Error Message Information Leakage Prevention (23 tests)
- agent_zero/error_responses.py: safe_error(), safe_http_error(), incident IDs
- Applied to 11 locations: agent_zero_server.py (9), auth.py (1), tool_runtime.py (1)
- Research paper status: implemented

### Integration Wiring (82 existing tests confirmed)
- observability.py wired into _run_conversation_turn + /admin/metrics endpoint
- rate_limiter.py wired into /ws/chat (per-user API rate limit)
- ws_messages.py wired into /ws/chat and /ws/voice (validation + per-connection rate limit)

**Total: 105 tests, 161-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track -- Research Papers (ready_for_implementation)

1. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
2. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
3. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large

### Agent Zero Track -- Architecture

4. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
7. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
8. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
9. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
10. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration Remaining
11. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
12. **Monitor skip_safe flags** in production to validate VOI feedback loop
13. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility
14. **A2 verification**: Awaiting verification of Session 299 changes

### Integration Testing
15. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
16. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
17. **Test quality gates end-to-end** -- Verify flags appear in reasoning events

### Frontend
18. **Display model tool execution events** in React UI ("Looking something up...")
19. **Display reasoning ticker with Shadow and disagreement thoughts**

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

161 sessions zero-bug
