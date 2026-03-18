# Next Session Briefing

**Last session:** 294 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 294

### Content-Aware Token Estimation (20 new tests)
- Replaced fixed `_CHARS_PER_TOKEN=4` with content-aware estimator (prose/code/CJK)
- Added vLLM calibration feedback (EMA from `usage.prompt_tokens`)
- 4 new config fields: `chars_per_token_prose/code/cjk`, `calibration_alpha`
- Wired calibration into agent_zero_server.py after generate_with_tools

**Total: 50 context_manager tests, 157-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Proactive Conversation Starters** (research/papers/proactive_conversation_starters.md) -- SSE, JITAI triggers (MED, new)
2. **Memory Recall Transparency** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
3. **Runtime Observability Layer** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent
4. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large, new)
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
7. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
8. **Monitor skip_safe flags** in production to validate VOI feedback loop
9. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility

### New Research Papers Available (MED priority)
10. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
11. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
12. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Database Track

13. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
14. Consider window functions (ROW_NUMBER, RANK, etc.)
15. Consider correlated subqueries inside derived tables
16. Consider join index optimization (use indexes during JOINs)

### Integration Testing

17. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
18. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
19. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
20. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat
21. **Test bandit weight learning end-to-end** -- Verify weights update after acted/ignored outcomes

### Frontend

22. **Display model tool execution events** in React UI ("Looking something up...")
23. **Display reasoning ticker with Shadow and disagreement thoughts**

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

157 sessions zero-bug
