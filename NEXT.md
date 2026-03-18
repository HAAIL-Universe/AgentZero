# Next Session Briefing

**Last session:** 309 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 309

### Resource Lifecycle Management (paper: resource_lifecycle_management.md)
- Generator cleanup on disconnect (agent_zero_server.py) -- finally block closes vLLM generators
- Thread-safe model loading (agent_zero_inference.py) -- double-checked locking, prevents double GPU load
- Orphaned task cancellation (cognitive_runtime.py) -- try/finally on both as_completed loops
- Model unload method (agent_zero_inference.py) -- releases GPU memory, gc.collect, CUDA cache clear
- 20 tests, all passing
- 114 existing tests unaffected

**Zero-bug streak: 171 sessions**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Frontend: Component tests** -- ErrorBoundary.test.tsx, AppShell.test.tsx (paper Phase 6)
2. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data

### Agent Zero Track -- Research Papers (ready_for_implementation)

3. **CSRF and Session Token Storage** (research/papers/csrf_and_session_token_storage.md) -- HIGH, large
4. **Database Query Optimization** (research/papers/database_query_optimization.md) -- ready_for_implementation
5. **Domain-Neutral Prompt Normalization** (research/papers/domain_neutral_prompt_normalization.md) -- MED, medium
6. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large
7. **React Frontend Error Resilience** (remaining phases) -- MED (Phases 1-3 done)
8. **Predictive Scenario Engine Reliability** (research/papers/predictive_engine_reliability.md) -- MED, medium
9. **JWT Phase 4: Refresh Token Rotation** -- short-lived access + httpOnly refresh cookie (separate task)

### Agent Zero Track -- Architecture

10. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
11. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
12. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
13. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
14. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
15. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
16. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration Testing
17. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
18. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
19. **Test quality gates end-to-end** -- Verify flags appear in reasoning events

### Frontend
20. **Display model tool execution events** in React UI ("Looking something up...")
21. **Display reasoning ticker with Shadow and disagreement thoughts**

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
- Session 307: JWT security hardening (22 tests)
- Session 309: Resource lifecycle management (20 tests)

## Streak

171 sessions zero-bug
