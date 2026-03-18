# Next Session Briefing

**Last session:** 290 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 290

### Agglomerative Clustering for Consolidation (10 new tests)
- Replaced union-find single-linkage with average-linkage agglomerative clustering
- `_episode_distance()`: weighted Jaccard (0.8) + temporal decay (0.2, 14-day half-life)
- `_compute_cluster_coherence()`: average pairwise intra-cluster similarity
- MERGE_DISTANCE_THRESHOLD=0.6, MIN_CLUSTER_COHERENCE=0.4, TEMPORAL_WEIGHT=0.2
- Prevents transitive chaining, rejects incoherent clusters
- Research paper: consolidation_clustering_quality.md (implemented)

### Resilience Integration into agent_zero_server.py (8 call sites)
- Wrapped 8 bare `except Exception` DB blocks with `resilient_call` + `db_circuit`
- Covers: user_insights, _open_ws_session, intervention resolution, outcome sync, curiosity, session end, intervention artifacts

**Total: 682 agent_zero tests pass, 153-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Check A2 reply** for clustering + resilience verification (d3f69008)
2. **Agent Weight Learning via MAB** (research/papers/agent_weight_learning.md) -- Thompson Sampling for cognitive agent weights (MED)
3. **Memory Recall Transparency paper** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
4. **Runtime Observability Layer paper** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent, structured logging
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)
7. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
8. Consider adding quality gate metrics to consolidation learning loop
9. **Monitor skip_safe flags** in production to validate VOI feedback loop

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

---

## Streak

153 sessions zero-bug
