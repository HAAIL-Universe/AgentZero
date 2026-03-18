# Next Session Briefing

**Last session:** 300 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Session 300

### A2 Findings Fixed
- /api/load-model exception leak: now uses safe_http_error()
- /ws/voice missing per-user rate limit: added rate_limiter.check()

### Async Event Loop Safety (paper: implemented)
- Eliminated all 11 asyncio.get_event_loop() calls (agent_zero_server.py, cognitive_runtime.py)
- Converted 2 blocking subprocess.run() to asyncio.create_subprocess_exec() (tool_runtime.py)
- HTTP calls in voice.py/agent_zero_inference.py already properly wrapped -- no changes needed

### Silent Exception Data Loss (paper: in_progress)
- Fixed 2 bare except:pass in behavioural_shadow.py (now logs warnings)
- Added password re-auth guard to DELETE /api/user/clear-data
- DEFERRED: soft-delete with 48-hour recovery window (requires DB migration)

**Total: 15 new tests, 162-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track -- Immediate

1. **Soft-delete for clear_user_data** -- Complete silent_exception_data_loss.md Part 2 (DB migration, deleted_at column, purge job)
2. **Audit remaining silent-swallow sites** -- agent_zero_server.py has 12 except:pass sites (lines 210, 419, 462, 1170, 1175, 2105, 2117, 2169, 2391, 3294, 3359, 3490)

### Agent Zero Track -- Research Papers (ready_for_implementation)

3. **JWT Security Hardening** (research/papers/jwt_security_hardening.md) -- HIGH, medium
4. **Streaming Generation Timeout** (research/papers/streaming_generation_timeout.md) -- HIGH, medium
5. **Frontend Accessibility** (research/papers/frontend_accessibility.md) -- HIGH, large

### Agent Zero Track -- Architecture

6. **Conversation Turn Decomposition** (research/papers/conversation_turn_decomposition.md) -- TurnContext + 17 phases (HIGH, large)
7. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver (HIGH, large)
8. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in pipeline (HIGH, large)

### Research Papers (MED priority)
9. **Consolidated Rules Growth Cap** (research/papers/consolidated_rules_growth_cap.md)
10. **Topic-Aware Memory Decay Rates** (research/papers/topic_aware_decay_rates.md)
11. **Outcome Pattern Confidence Scoring** (research/papers/outcome_pattern_confidence.md)
12. **External Outcome Resolution API** (research/papers/external_outcome_resolution_api.md)

### Integration Remaining
13. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
14. **Monitor skip_safe flags** in production to validate VOI feedback loop
15. **Add /admin/config endpoint** -- expose config.model_dump() for runtime visibility
16. **A2 verification**: Awaiting verification of Session 300 changes

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

## Streak

162 sessions zero-bug
