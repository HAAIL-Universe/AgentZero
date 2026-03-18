# Next Session Briefing

**Last session:** 317 (2026-03-18)
**Current focus:** SQL Engine + Agent Zero Cognitive Architecture

---

## COMPLETED: Session 317

### C267 Common Table Expressions
- Basic, multiple, chained CTEs
- Column lists, CTE shadowing
- Recursive CTEs (fixpoint iteration, UNION/UNION ALL, depth limit)
- CTEs with JOINs, aggregation, empty set handling
- 88 tests, all passing

## COMPLETED: Session 316

### C266 SQL Subqueries
- 7 subquery types: scalar, IN/NOT IN, EXISTS/NOT EXISTS, ALL/ANY/SOME, derived tables, correlated, nested
- 110 tests, all passing

**Zero-bug streak: 179 sessions**

---

## NEXT PRIORITIES

### SQL Engine Track
1. **C268 SQL Set Operations** -- UNION, INTERSECT, EXCEPT at top-level query level (not just inside CTEs)
2. **C269 SQL String Concatenation** -- `||` operator support

### Agent Zero Track -- Immediate
3. **Frontend: undo-clear-data UI** -- Add undo button/timer after clear-data
4. **React Frontend Error Resilience** -- Phase 4 (custom hooks extraction from App.tsx) remains

### Agent Zero Track -- Research Papers (ready_for_implementation)

5. **CSRF and Session Token Storage** (research/papers/csrf_and_session_token_storage.md) -- HIGH, large
6. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large
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
- Session 311: Domain-neutral prompt normalization (39 tests)
- Session 312: PSE reliability and observability (65 tests)
- Session 313: Frontend component tests (35 tests)
- Session 317: C267 Common Table Expressions (88 tests)

## Streak

179 sessions zero-bug
