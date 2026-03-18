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

- [ ] Dynamic context budgeting for multi-agent pipeline -- agents get different token budgets based on role and input size. Current: flat 6000 char cap for all. Gap: cognitive_agents.py:101-135.
- [ ] Consolidation rule quality gate -- validate rule quality before injection into agent context. Detect when rules produce poor outcomes. Gap: cognitive_runtime.py:229-260.
- [ ] Topic-aware memory decay rates -- exponential decay lambda=0.005/hr is uniform. Career decisions should decay slower than fitness check-ins. Gap: episode_store.py:40,206-211.
- [ ] Outcome pattern confidence scoring -- "i did it" vs "i tried it" should have different outcome confidence. Gap: outcome_patterns.py treats all matches identically.
- [ ] External outcome resolution API -- episodes only resolve from chat input. Need webhook/API for external systems (calendar, habit tracker). Gap: episode_store.py global.

## Low Priority

- [ ] PWA optimisation for mobile use -- service worker, manifest, offline capability.
- [ ] Cross-user learning -- schema is forward-compatible but currently per-user only. Research privacy-preserving aggregation.
- [ ] Telegram bot activation and testing -- verify existing check-in engine works end-to-end.
- [ ] Proactive conversation starters -- Agent Zero initiates based on shadow patterns without waiting for user.
- [ ] Consolidation threshold validation -- all constants (MIN_UNCONSOLIDATED_EPISODES=5, MAX_HOURS=1, etc.) need research justification. Gap: consolidator.py:29-38.
- [ ] Deterministic vs LLM-generated insights -- when can template-based consolidation insights match semantic extraction quality?
