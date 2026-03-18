# Next Session Briefing

**Last session:** 295 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 295

### Proactive Conversation Starters (36 new tests)
- NEW: agent_zero/proactive_evaluator.py -- JITAI trigger detection (4 triggers, receptivity safeguards)
- Background evaluation loop in agent_zero_server.py (60s interval, WebSocket delivery)
- Capability manifest: real_time_monitoring + proactive_outreach_web now True
- Frontend: proactive message type in WS handler, green-accent render in chat

**Total: 36 proactive_evaluator tests, 158-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Memory Recall Transparency** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
2. **Runtime Observability Layer** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent
3. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large, new)
4. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
5. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
6. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
7. **Monitor skip_safe flags** in production to validate VOI feedback loop
8. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility

### New Research Papers Available (MED priority)
9. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
10. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
11. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Database Track

12. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
13. Consider window functions (ROW_NUMBER, RANK, etc.)
14. Consider correlated subqueries inside derived tables
15. Consider join index optimization (use indexes during JOINs)

### Integration Testing

16. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
17. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
18. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
19. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat
20. **Test bandit weight learning end-to-end** -- Verify weights update after acted/ignored outcomes
21. **Test proactive messages end-to-end** -- Verify triggers fire and render in chat during live session

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

158 sessions zero-bug
