# Next Session Briefing

**Last session:** 287 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Three Papers Implemented (Session 287)

### Chip Persistence (Frontend, ~50 lines)
- `agent_zero-ui/src/hooks/useSessionPersistedState.ts` -- sessionStorage sync hook
- `chipsByTurn` now survives page refresh, cleared on newSession()
- 2MB cap with oldest-turn trimming

### Session-Start Check-In (Backend, 21 tests)
- `agent_zero/session_checkin.py` -- evaluates pending commitments at session start
- Motivation scoring with 6 trigger types, MI-informed prompt generation
- Injected as `extra_system_prompt` on first conversation turn
- Both text and voice WS handlers wired

### Resilience Layer (Backend, 30 tests)
- `agent_zero/resilience.py` -- CircuitBreaker, classify_error, resilient_call
- Ready for integration into agent_zero_server.py DB/model calls

**576 total agent_zero tests verified, 150-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Check A2 reply** for session_checkin + resilience verification (9fd153a3)
2. **Integrate resilience.py into agent_zero_server.py** -- wrap critical DB calls with resilient_call + db_circuit, replace `except Exception: pass` blocks
3. **Memory Recall Transparency paper** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
4. **Runtime Observability Layer paper** (research/papers/runtime_observability_layer.md) -- wide-event TurnEvent, structured logging (MED, ready)
5. **Constraint-Based Commitment Scheduling** (research/papers/constraint_commitment_scheduling.md) -- CSP solver for conflict-free scheduling (HIGH, large)
6. **Logic Programming for Transparent Reasoning** (research/papers/logic_transparent_reasoning.md) -- Prolog in cognitive pipeline (HIGH, large)
7. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
8. Consider adding quality gate metrics to consolidation learning loop
9. **Monitor skip_safe flags** in production to validate VOI feedback loop
10. **Test stage classification end-to-end** -- verify stage events appear in reasoning runs

### Database Track

11. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
12. Consider window functions (ROW_NUMBER, RANK, etc.)
13. Consider correlated subqueries inside derived tables
14. Consider join index optimization (use indexes during JOINs)

### Integration Testing

15. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
16. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
17. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
18. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat

### Frontend

19. **Display model tool execution events** in React UI ("Looking something up...")
20. **Display reasoning ticker with Shadow and disagreement thoughts**

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)
- C247 HAVING with raw COUNT(*) doesn't work (use alias reference instead)
- test_agent_zero_e2e commitment_tracking test: async function called without await (pre-existing)

---

## Streak

150 sessions zero-bug
