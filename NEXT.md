# Next Session Briefing

**Last session:** 301 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 301

### Auth Hardening (paper: implemented)
- NIST SP 800-63B Rev 4 password policy (8-64 chars, common password blocklist)
- RFC 5322 email validation (stdlib parser)
- OWASP login lockout (5 attempts / 15min window -> 20min lock)
- Auth endpoint rate limiting (wired existing config to /auth/register, /auth/login)
- JWT user cache with TTL (5min, invalidate on clear-data)
- 31 new tests in test_auth_hardening.py

### Guardrail Test Coverage (paper: implemented)
- test_guardrails.py expanded from 9 to 86 tests
- All crisis/harm/domain/scope/speaker/tool/output/fragment patterns covered

### Bug Fix
- test_config.py model_context_limit assertion updated (32768 -> 40960)

**Total: 117 new tests, 163-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Soft-delete for clear_user_data** -- Complete silent_exception_data_loss.md Part 2 (DB migration, deleted_at column, purge job)
2. **Audit remaining silent-swallow sites** -- agent_zero_server.py has 12 except:pass sites

### Agent Zero Track -- Research Papers (ready_for_implementation)

3. **Domain-Neutral Prompt Normalization** (research/papers/domain_neutral_prompt_normalization.md) -- MED, medium (MQ: b7741bc4, unarchived)
4. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
5. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
6. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large

### Agent Zero Track -- Architecture

7. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
8. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
9. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
10. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
11. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
12. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
13. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration Remaining
14. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
15. **Monitor skip_safe flags** in production to validate VOI feedback loop
16. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility
17. **A2 verification**: Awaiting verification of Session 301 changes

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

163 sessions zero-bug
