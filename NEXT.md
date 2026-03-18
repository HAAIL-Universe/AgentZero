# Next Session Briefing

**Last session:** 292 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 292

### Centralized Agent ZeroConfig (14 tests)
- New `agent_zero/config.py` with `Agent ZeroConfig(BaseSettings)` -- 45 validated fields
- Pydantic-settings with AGENT_ZERO_ env var prefix, .env file, range constraints
- Updated 7 source files to import from config singleton
- Created `.env.example` documenting all tunable parameters
- Zero behavioral change: all defaults match previous hardcoded values

### Structured JSON Logging (14 tests)
- New `agent_zero/logging_config.py` -- JsonFormatter, contextvars binding, get_logger
- Replaced 37 print() statements in agent_zero_server.py with severity-leveled logging
- Replaced 3 silent except:pass with _log.debug()
- LOG_LEVEL env var support, setup_logging() at startup

**Total: 28 new tests, 155-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Cognitive Runtime Configuration** (research/papers/cognitive_runtime_configuration.md) -- NEW research paper from RESEARCHER (HIGH)
2. **Proactive Conversation Starters** (research/papers/proactive_conversation_starters.md) -- SSE, JITAI triggers (MED, new)
3. **Memory Recall Transparency** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
4. **Runtime Observability Layer** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
7. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
8. Consider adding quality gate metrics to consolidation learning loop
9. **Monitor skip_safe flags** in production to validate VOI feedback loop
10. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility (follow-up to config extraction)

### New Research Papers Available (MED priority)
11. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
12. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
13. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Database Track

14. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
15. Consider window functions (ROW_NUMBER, RANK, etc.)
16. Consider correlated subqueries inside derived tables
17. Consider join index optimization (use indexes during JOINs)

### Integration Testing

18. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
19. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
20. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
21. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat
22. **Test bandit weight learning end-to-end** -- Verify weights update after acted/ignored outcomes

### Frontend

23. **Display model tool execution events** in React UI ("Looking something up...")
24. **Display reasoning ticker with Shadow and disagreement thoughts**

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

155 sessions zero-bug
