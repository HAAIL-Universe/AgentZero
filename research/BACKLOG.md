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
- [x] Consolidation threshold validation -- all constants (MIN_UNCONSOLIDATED_EPISODES=5, MAX_HOURS=1, etc.) need research justification. Gap: consolidator.py:29-38. (Researched 2026-03-18, paper: research/papers/consolidation_threshold_validation.md, 9 citations)
- [ ] Deterministic vs LLM-generated insights -- when can template-based consolidation insights match semantic extraction quality?

## High Priority (New -- from gap analysis 2026-03-18 session 4)

- [x] Async safety and race condition prevention -- agent_zero_server.py has no asyncio.Lock protecting shadow/session_messages mutations across await boundaries. 8 run_in_executor calls break single-thread guarantee. (Researched 2026-03-18, paper: research/papers/async_safety_race_prevention.md, 6 citations)

## High Priority (New -- from gap analysis 2026-03-18 session 5)

- [x] Token estimation accuracy -- context_manager.py uses fixed 4 chars/token heuristic that is wrong for code (actual ~5.5-6.5) and CJK text (actual ~1.5). Affects context compression timing and dynamic context budgeting. (Researched 2026-03-18, paper: research/papers/token_estimation_accuracy.md, 6 citations)
- [x] Conversation turn decomposition -- _run_conversation_turn() is 1,464 lines with 23 phases, 200+ awaits, untestable. Needs Split Phase refactoring into TurnContext + phase functions. (Researched 2026-03-18, paper: research/papers/conversation_turn_decomposition.md, 6 citations)

## High Priority (New -- from gap analysis 2026-03-18 session 6)

- [x] API rate limiting and abuse prevention -- zero rate limiting on 37 endpoints. Registration flooding, brute force login, inference DoS all possible. OWASP API4:2023 + LLM10:2025 violation. (Researched 2026-03-18, paper: research/papers/api_rate_limiting.md, 6 citations)
- [x] Health and readiness probes -- no liveness/readiness/startup endpoints for RunPod container orchestration. Degraded instances continue receiving traffic. (Researched 2026-03-18, paper: research/papers/health_readiness_probes.md, 6 citations)

## High Priority (New -- from gap analysis 2026-03-18 session 7)

- [x] WebSocket input validation and message security -- no JSON schema validation on ws/chat or ws/voice messages, no audio size limits before base64 decode, no message size caps. Malformed JSON crashes connections, large audio payloads cause OOM. OWASP API8:2023 violation. (Researched 2026-03-18, paper: research/papers/websocket_input_validation.md, 6 citations)
- [x] JWT security hardening -- weak default secret ("agent_zero-dev-secret-change-in-prod"), JWT tokens passed in query parameters (logged in proxies/browser history). Needs fail-fast on missing secret + first-message auth pattern for WebSocket. (Researched 2026-03-18, paper: research/papers/jwt_security_hardening.md, 6 citations)

## Medium Priority (New -- from gap analysis 2026-03-18 session 7)

- [x] Database query optimization -- N+1 pattern in get_productivity_summary() (7 separate queries), unbounded SELECT in update_requests.py, missing composite indexes on (user_id, created_at), (outcome). Needs CTE batching and index additions. (Researched 2026-03-18, paper: research/papers/database_query_optimization.md, 6 citations)
- [x] Streaming generation timeout and recovery -- no timeout on vLLM streaming loop (agent_zero_server.py:2420-2480). Hung model causes WebSocket to block forever. Needs asyncio.wait_for + fallback response. (Researched 2026-03-18, paper: research/papers/streaming_generation_timeout.md, 6 citations)
- [x] Frontend accessibility (WCAG 2.1) -- zero aria-* attributes in agent_zero.html, no keyboard navigation, no screen reader support. Voice-first interface needs full keyboard accessibility. (Researched 2026-03-18, paper: research/papers/frontend_accessibility.md, 7 citations)
- [x] Consolidated rules growth cap -- episode_store consolidated_rules array is unbounded. 10k rules = 5-10MB per user profile. Needs max cap + rule retirement with audit logging. (Researched 2026-03-18, paper: research/papers/consolidated_rules_growth_cap.md, 5 citations)
- [x] Error message information leakage -- generic exceptions sent as str(e) to WebSocket clients (agent_zero_server.py:3164). Leaks stack traces, DB structure, file paths. Needs generic client messages + server-side structured logging. (Researched 2026-03-18, paper: research/papers/error_message_information_leakage.md, 6 citations)

## High Priority (New -- from gap analysis 2026-03-18 session 8)

- [x] Async event loop safety -- 11x deprecated asyncio.get_event_loop(), 2x blocking subprocess.run() (10s), 3x blocking urllib (120s). Freezes all WebSocket connections. Python 3.12 deprecation active. (Researched 2026-03-18, paper: research/papers/async_event_loop_safety.md, 6 citations)
- [x] Silent exception data loss + destructive operation guards -- bare except-pass discards commitment DB writes (behavioural_shadow.py:439,488). DELETE /api/user/clear-data has no re-auth, no soft-delete, no recovery window. (Researched 2026-03-18, paper: research/papers/silent_exception_data_loss.md, 6 citations)

## Medium Priority (New -- from gap analysis 2026-03-18 session 8)

- [x] Rate limiter and WS validation integration -- RESOLVED: already wired into agent_zero_server.py (lines 123-127, 3353, 3373, 3389, 3442, 3461, 3467). Confirmed 2026-03-18.
- [x] Auth hardening -- minimum password 6 chars (should be 8, NIST SP 800-63B), email validation is just "@" check, no failed-login counter, per-request DB lookup for JWT user (should cache). (Researched 2026-03-18, paper: research/papers/auth_hardening.md, 6 citations)
- [x] Domain-neutral prompt normalization -- hardcoded strings ("ask for a raise", "manager is unpredictable") in agent_zero_server.py and reasoning_framework.py. Brittle, domain-specific. (Researched 2026-03-18, paper: research/papers/domain_neutral_prompt_normalization.md, 6 citations)
- [x] Safety-critical guardrail test coverage -- test_guardrails.py is 120 lines for 599-line guardrails.py. Crisis/harm pattern coverage insufficient. (Researched 2026-03-18, paper: research/papers/guardrail_test_coverage.md, 6 citations)
- [x] Silent 2K token truncation in local inference -- agent_zero_inference.py:257 uses max_length=2048 but model supports 40960 (config.model_context_limit). (Researched 2026-03-18, paper: research/papers/silent_token_truncation.md, 6 citations)

## High Priority (New -- from gap analysis 2026-03-18 session 9)

- [x] Frontend XSS and Content Security Policy -- renderMarkdown() uses innerHTML without sanitization (agent_zero.html:2574, 2624, 2748, 3031), no CSP header, inline scripts/styles. OWASP A03:2021 violation. (Researched 2026-03-18, paper: research/papers/frontend_xss_and_csp.md, 7 citations)
- [x] Proactive session concurrency safety -- _proactive_sessions dict accessed without locks from multiple coroutines (agent_zero_server.py:196-236, 3417-3418). Bare except-pass at line 217 masks all errors. (Researched 2026-03-18, paper: research/papers/proactive_session_concurrency.md, 5 citations)
- [x] Resource lifecycle management -- streaming generator not cleaned up on disconnect (agent_zero_server.py:2674-2680), model load race condition (agent_zero_inference.py:69-100), orphaned coroutines. (Researched 2026-03-18, paper: research/papers/resource_lifecycle_management.md, 5 citations)

## Medium Priority (New -- from gap analysis 2026-03-18 session 9)

- [x] Database transaction atomicity -- non-atomic commitment creation (database.py:410-425, two INSERT without transaction), non-atomic consolidation marking (consolidator.py:914-916, in-memory only). (Researched 2026-03-18, paper: research/papers/database_transaction_atomicity.md, 5 citations)
- [x] Unbounded in-memory state growth -- _login_attempts dict in auth.py:89 grows without eviction, _A2_INBOX_CACHE in tool_runtime.py:520 never trimmed, _calibration_ratio in context_manager.py:33 has no thread safety. (Researched 2026-03-18, paper: research/papers/unbounded_in_memory_state_growth.md, 5 citations)
- [x] CSRF and session token storage -- JWT stored in localStorage (XSS-accessible), token passed in WebSocket URL query params (logged in proxies/history), no CSRF token on POST/DELETE endpoints, CORS allow_headers=* wildcard. (Researched 2026-03-18, paper: research/papers/csrf_and_session_token_storage.md, 6 citations)
