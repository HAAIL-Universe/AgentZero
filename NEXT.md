# Next Session Briefing

**Last session:** 289 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 289

### Voice Endpoint Check-In Fix
- `/ws/voice` now injects `checkin_prompt` into first turn (both text and audio paths)
- Mirrors `/ws/chat` pattern exactly

### Bayesian Intervention Effectiveness (20 tests)
- `bayesian_rates.py`: Beta-binomial posterior, CI, effective sample size
- `intervention_tracker.py`: posterior means, CI lower-bound suggestions, n_eff<8 guard
- Fixed missing `re` import in intervention_tracker.py

### BM25 Semantic Memory Retrieval (17 tests)
- `bm25_scorer.py`: Okapi BM25 with IDF, TF saturation, length norm, stemmer
- Integrated into `retrieval_policy.py` and `memory_policy.py`
- Backward compatible (falls back to keyword overlap when no scorer)

**Total: 665 agent_zero tests pass, 152-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Check A2 reply** for Bayesian + BM25 + voice fix verification (e2a08b8f)
2. **Integrate resilience.py into agent_zero_server.py** -- wrap critical DB calls with resilient_call + db_circuit, replace `except Exception: pass` blocks
3. **Agent Weight Learning via MAB** (research/papers/agent_weight_learning.md) -- Thompson Sampling for cognitive agent weights (MED)
4. **Memory Recall Transparency paper** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
5. **Runtime Observability Layer paper** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent, structured logging
6. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
7. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
8. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
9. Consider adding quality gate metrics to consolidation learning loop
10. **Monitor skip_safe flags** in production to validate VOI feedback loop

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

---

## Streak

152 sessions zero-bug
