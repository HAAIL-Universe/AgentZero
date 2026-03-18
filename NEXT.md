# Next Session Briefing

**Last session:** 311 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 311

### Domain-Neutral Prompt Normalization (paper: domain_neutral_prompt_normalization.md)
- Created centralized topic_taxonomy.json (10 domains, strategic keywords, recap patterns)
- Created topic_taxonomy.py loader module with caching
- Updated 7 consumer modules to use taxonomy instead of hardcoded dicts
- 39 tests, all passing

**Zero-bug streak: 173 sessions**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Predictive Scenario Engine Reliability** (research/papers/predictive_engine_reliability.md) -- MED, medium
   - Replace 7 silent except handlers with structured logging
   - Wire into TurnEvent observability
   - Replace sys.path manipulation with importlib
   - Add degradation metadata to return dict
2. **Frontend: Component tests** -- ErrorBoundary.test.tsx, AppShell.test.tsx (paper Phase 6)
3. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data

### Agent Zero Track -- Research Papers (ready_for_implementation)

4. **CSRF and Session Token Storage** (research/papers/csrf_and_session_token_storage.md) -- HIGH, large
5. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large
6. **React Frontend Error Resilience** (remaining phases) -- MED (Phases 1-3 done)
7. **JWT Phase 4: Refresh Token Rotation** -- short-lived access + httpOnly refresh cookie (separate task)

### Agent Zero Track -- Architecture

8. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
9. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
10. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
11. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
12. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
13. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
14. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

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
- test_context_manager::test_huge_conversation_triggers_compress: pre-existing threshold issue

---

## A2 Verification Pending
- Session 304: ErrorBoundary + typed props (tsc clean), silent-swallow fixes
- Session 307: JWT security hardening (22 tests)
- Session 309: Resource lifecycle management (20 tests)
- Session 310: Database query optimization (29 tests)
- Session 311: Domain-neutral prompt normalization (39 tests)

## Streak

173 sessions zero-bug
