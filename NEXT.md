# Next Session Briefing

**Last session:** 313 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 313

### Frontend Component Tests (React Error Resilience Phase 6)
- Set up vitest + @testing-library/react + jsdom test infrastructure
- ErrorBoundary.test.tsx: 11 tests (catch, fallback, reset, isolation, callbacks)
- AppShell.test.tsx: 24 tests (toolbar, chat, voice, tab switching)
- Total: 45 frontend tests, all passing. tsc clean.
- Added `npm test` script to package.json

**Zero-bug streak: 175 sessions**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate
1. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data
2. **React Frontend Error Resilience** -- Phase 4 (custom hooks extraction from App.tsx) remains

### Agent Zero Track -- Research Papers (ready_for_implementation)

3. **CSRF and Session Token Storage** (research/papers/csrf_and_session_token_storage.md) -- HIGH, large
4. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large
5. **JWT Phase 4: Refresh Token Rotation** -- short-lived access + httpOnly refresh cookie (separate task)

### Agent Zero Track -- Architecture

6. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
7. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
8. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
9. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
10. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
11. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
12. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration Testing
13. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
14. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
15. **Test quality gates end-to-end** -- Verify flags appear in reasoning events

### Frontend
16. **Display model tool execution events** in React UI ("Looking something up...")
17. **Display reasoning ticker with Shadow and disagreement thoughts**

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)
- C247 HAVING with raw COUNT(*) doesn't work (use alias reference instead)
- test_agent_zero_e2e commitment_tracking test: async function called without await (pre-existing)
- test_cognitive_agents::test_model_backed_agent_uses_prompt_body_when_loaded: pre-existing assertion mismatch
- test_agent_zero_turn_paths: user=None crash on save_shadow (pre-existing)
- test_context_manager::test_huge_conversation_triggers_compress: pre-existing threshold issue

---

## A2 Verification Pending
- Session 304: ErrorBoundary + typed props (tsc clean), silent-swallow fixes
- Session 307: JWT security hardening (22 tests)
- Session 311: Domain-neutral prompt normalization (39 tests)
- Session 312: PSE reliability and observability (65 tests)
- Session 313: Frontend component tests (35 tests)

## Streak

175 sessions zero-bug
