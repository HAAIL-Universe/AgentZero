# Next Session Briefing

**Last session:** 306 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 306

### XSS Hardening + Security Headers (paper: frontend_xss_and_csp.md)
- DOMPurify added to React renderMarkdown() -- defense-in-depth for dangerouslySetInnerHTML
- Security headers middleware: X-Content-Type-Options, X-Frame-Options, Referrer-Policy, CSP
- CORS tightened: allow_headers wildcard -> [Authorization, Content-Type]
- Legacy agent_zero.html: all innerHTML injections escaped, all empty catch blocks fixed
- 17 tests, all passing
- tsc clean, Vite build clean

**Zero-bug streak: 168 sessions**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Frontend: Component tests** -- ErrorBoundary.test.tsx, AppShell.test.tsx (paper Phase 6)
2. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data

### Agent Zero Track -- Research Papers (ready_for_implementation)

5. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
6. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
7. ~~**Frontend XSS and CSP**~~ -- DONE (Session 306, 17 tests)
8. **Resource Lifecycle Management** (research/papers/resource_lifecycle_management.md) -- HIGH, medium
9. **CSRF and Session Token Storage** (research/papers/csrf_and_session_token_storage.md) -- HIGH, large (after XSS/CSP)
10. **Database Query Optimization** (research/papers/database_query_optimization.md) -- ready_for_implementation
11. **Domain-Neutral Prompt Normalization** (research/papers/domain_neutral_prompt_normalization.md) -- MED, medium
12. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large
13. **React Frontend Error Resilience** (remaining phases) -- MED (Phases 1-3 done)
14. **Predictive Scenario Engine Reliability** (research/papers/predictive_engine_reliability.md) -- MED, medium (new)

### Agent Zero Track -- Architecture

15. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
16. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
17. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
18. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
19. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
20. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
21. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration Testing
22. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
23. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
24. **Test quality gates end-to-end** -- Verify flags appear in reasoning events

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

## A2 Verification Pending
- Session 304: ErrorBoundary + typed props (tsc clean), silent-swallow fixes
- Session 306: XSS hardening + security headers (17 tests)

## Streak

168 sessions zero-bug
