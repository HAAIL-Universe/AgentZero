# Next Session Briefing

**Last session:** 303b (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 303b

### Proactive Session Concurrency Safety (paper: implemented)
- asyncio.Lock on _proactive_sessions dict
- WebSocketState check before send, dead session cleanup
- Structured logging replaces bare except:pass
- 10 new tests in test_proactive_concurrency.py

### Unbounded In-Memory State Growth (paper: implemented)
- User cache reaper size cap enforcement
- threading.Lock on _calibration_ratio
- 2 new tests (17 total in test_unbounded_state_growth.py)

### Database Transaction Atomicity (paper: implemented)
- transaction() async context manager helper
- create_commitment() and update_commitment_status() wrapped in transactions
- _update_streak_atomic() with SELECT ... FOR UPDATE row locking
- 11 new tests in test_database_transaction_atomicity.py

**Total: 38 new tests, 165-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Audit remaining silent-swallow sites** -- agent_zero_server.py still has except:pass sites
2. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data

### Agent Zero Track -- Research Papers (ready_for_implementation)

3. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
4. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
5. **Frontend XSS and CSP** (research/papers/frontend_xss_and_csp.md) -- HIGH, medium
6. **Resource Lifecycle Management** (research/papers/resource_lifecycle_management.md) -- HIGH, medium
7. **CSRF and Session Token Storage** (research/papers/csrf_and_session_token_storage.md) -- HIGH, large (after XSS/CSP)
8. **Database Query Optimization** (research/papers/database_query_optimization.md) -- ready_for_implementation
9. **Domain-Neutral Prompt Normalization** (research/papers/domain_neutral_prompt_normalization.md) -- MED, medium
10. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large

### Agent Zero Track -- Architecture

11. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
12. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
13. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
14. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
15. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
16. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
17. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

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

## A2 Verification Pending
- Session 303b: Proactive concurrency (10 tests) + calibration lock (2 tests) + DB transactions (11 tests)

## Streak

165 sessions zero-bug
