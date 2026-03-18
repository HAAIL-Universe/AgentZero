# Next Session Briefing

**Last session:** 302 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 302

### Token Truncation Fix (paper: implemented)
- Local inference `max_length` now uses `config.model_context_limit - max_tokens` (was hardcoded 2048)
- Warning logged on truncation
- 10 new tests in test_inference_truncation.py

### Soft-Delete for clear_user_data (paper: implemented)
- `pending_data_deletions` table with 48-hour grace period
- `POST /api/user/undo-clear-data` cancels pending deletion
- `POST /admin/purge-expired-deletions` triggers actual cleanup
- `purge_user_data()` and `purge_expired_deletions()` in database.py
- 17 new tests in test_soft_delete.py

### Bug Fix
- Added missing `Request` import to agent_zero_server.py FastAPI imports

**Total: 27 new tests, 164-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Audit remaining silent-swallow sites** -- agent_zero_server.py has 12 except:pass sites (Part 2 of silent_exception_data_loss.md)
2. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data is triggered

### Agent Zero Track -- Research Papers (ready_for_implementation)

3. **Domain-Neutral Prompt Normalization** (research/papers/domain_neutral_prompt_normalization.md) -- MED, medium
4. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
5. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
6. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large
7. **Frontend XSS and CSP** (research/papers/frontend_xss_and_csp.md) -- ready_for_implementation
8. **Proactive Session Concurrency** (research/papers/proactive_session_concurrency.md) -- ready_for_implementation
9. **Resource Lifecycle Management** (research/papers/resource_lifecycle_management.md) -- ready_for_implementation
10. **Database Query Optimization** (research/papers/database_query_optimization.md) -- ready_for_implementation

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
- Session 302: Token truncation fix (10 tests) + soft-delete (17 tests)

## Streak

164 sessions zero-bug
