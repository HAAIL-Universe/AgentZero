# Research Backlog

Prioritised list of research topics for the Research Agent.
Mark with [x] and date when researched. Papers go in `research/papers/`.

## High Priority

- [x] Learned agent routing -- use consolidation episode data to select which agents activate per topic instead of keyword-based complexity scoring. Reference: NeurIPS 2025 dynamic orchestration paper. (Researched 2026-03-18, paper: research/papers/learned_agent_routing.md)
- [x] Explicit agree/disagree deliberation protocol -- require agents to justify positions and revise only on high-confidence rationale. Reference: ICLR 2025 DMAD, arxiv 2511.07784. (Researched 2026-03-18, paper: research/papers/explicit_deliberation_protocol.md)
- [x] Cost-aware agent activation -- skip agents when consolidation data shows they historically add no signal for this topic. Penalise unnecessary agent calls. (Researched 2026-03-18, updated 2026-03-18 for all-hybrid agents, paper: research/papers/cost_aware_agent_activation.md)
- [x] Quality gates on Speaker output -- verify response style matches learned intervention effectiveness from consolidation rules before sending to user. (Researched 2026-03-18, paper: research/papers/speaker_quality_gates.md)
- [x] Tool activity collapsible log in chat thread -- show tool calls as an expandable activity section between user message and response, collapsible after completion. (Researched 2026-03-18, updated 2026-03-18, paper: research/papers/tool_activity_collapsible_log.md)

## Medium Priority

- [x] Connect AZ's constraint solver (C094) for commitment scheduling -- optimise commitment cadence/timing given user constraints. (Researched 2026-03-18, paper: research/papers/constraint_commitment_scheduling.md)
- [x] Connect AZ's logic programming (C095) for transparent reasoning -- show inference proof chains to user. (Researched 2026-03-18, paper: research/papers/logic_transparent_reasoning.md)
- [x] Adaptive voice personality per topic and stage-of-change -- Speaker adjusts tone based on consolidated data about what works. (Researched 2026-03-18, paper: research/papers/adaptive_voice_personality.md)
- [x] Session-start check-in from pending commitments -- proactive greeting with commitment status. (Researched 2026-03-18, paper: research/papers/session_start_checkin.md)
- [x] Chip persistence across page refreshes -- store reasoning chips in sessionStorage so they survive F5. (Researched 2026-03-18, paper: research/papers/chip_persistence.md)
- [x] Memory recall transparency -- show what memories were retrieved and how they influenced the response. (Researched 2026-03-18, paper: research/papers/memory_recall_transparency.md)

- [x] Resilient async operations -- circuit breakers, timeouts, and error classification for DB writes and external service calls that currently fail silently. (Researched 2026-03-18, paper: research/papers/resilient_async_operations.md)
- [x] Episode-intervention outcome synchronization -- bidirectional outcome resolution between episode_store and intervention_tracker to prevent divergence. (Researched 2026-03-18, paper: research/papers/episode_intervention_sync.md)
- [x] Runtime observability layer -- structured metrics for agent execution times, consolidation performance, rule application rates, and DB query latency. (Researched 2026-03-18, paper: research/papers/runtime_observability_layer.md)

## Medium Priority (New -- from gap analysis 2026-03-18)

- [x] Dynamic context budgeting for multi-agent pipeline -- agents get different token budgets based on role and input size. (Researched 2026-03-18, paper: research/papers/dynamic_context_budgeting.md, 6 citations)
- [x] Consolidation rule quality gate -- validate rule quality before injection into agent context. (Researched 2026-03-18, paper: research/papers/consolidation_rule_quality_gate.md, 6 citations)
- [x] Topic-aware memory decay rates -- exponential decay lambda per topic category. (Researched 2026-03-18, paper: research/papers/topic_aware_decay_rates.md, 6 citations)
- [x] Outcome pattern confidence scoring -- MISC-inspired commitment strength + hedge detection. (Researched 2026-03-18, paper: research/papers/outcome_pattern_confidence.md, 6 citations)
- [x] External outcome resolution API -- webhook + REST endpoints for external outcome reporting. (Researched 2026-03-18, paper: research/papers/external_outcome_resolution_api.md, 6 citations)

## High Priority (New -- from gap analysis 2026-03-18 session 2)

- [x] Semantic memory retrieval ranking -- replace keyword overlap in retrieval_policy.py with BM25/learned ranking. (Researched 2026-03-18, paper: research/papers/semantic_memory_retrieval.md, 6 citations)
- [x] Bayesian intervention effectiveness estimation -- replace raw acted_rate ratios in intervention_tracker.py with beta-binomial posterior for small-sample robustness. (Researched 2026-03-18, paper: research/papers/bayesian_intervention_effectiveness.md, 6 citations)
- [x] Agent weight learning via multi-armed bandits -- replace hardcoded agent weights (cognitive_runtime.py:578-584) with Thompson Sampling that updates from consolidation outcomes. (Researched 2026-03-18, paper: research/papers/agent_weight_learning.md, 6 citations)
- [x] Consolidation clustering quality -- replace O(n^2) union-find topic overlap (consolidator.py:81-100) with density-based or hierarchical clustering for higher-quality rule extraction. (Researched 2026-03-18, paper: research/papers/consolidation_clustering_quality.md, 6 citations)

## Medium Priority (New -- from gap analysis 2026-03-18 session 3)

- [x] Resilience layer integration -- resilience.py circuit breakers exist but are not wired into database.py or agent_zero_server.py. Research production circuit breaker integration patterns (Hystrix, Polly, Tenacity). (Researched 2026-03-18, paper: research/papers/resilience_layer_integration.md, 6 citations)
- [x] Structured logging replacement -- 15+ print() statements in agent_zero_server.py and 14 silent except+pass handlers. Research structured async logging for production Python services (structlog, loguru patterns). (Researched 2026-03-18, paper: research/papers/structured_logging.md, 6 citations)
- [x] Cognitive runtime configuration extraction -- 10+ hardcoded thresholds in cognitive_runtime.py (DELIBERATION_THRESHOLD, agent weight multipliers, tension coefficients). Research dynamic configuration management for AI pipelines. (Researched 2026-03-18, paper: research/papers/cognitive_runtime_configuration.md, 6 citations)

## Low Priority

- [ ] PWA optimisation for mobile use -- service worker, manifest, offline capability.
- [ ] Cross-user learning -- schema is forward-compatible but currently per-user only. Research privacy-preserving aggregation.
- [ ] Telegram bot activation and testing -- verify existing check-in engine works end-to-end.
- [x] Proactive conversation starters -- Agent Zero initiates based on shadow patterns without waiting for user. (Researched 2026-03-18, paper: research/papers/proactive_conversation_starters.md, 6 citations)
- [ ] Consolidation threshold validation -- all constants (MIN_UNCONSOLIDATED_EPISODES=5, MAX_HOURS=1, etc.) need research justification. Gap: consolidator.py:29-38.
- [ ] Deterministic vs LLM-generated insights -- when can template-based consolidation insights match semantic extraction quality?
