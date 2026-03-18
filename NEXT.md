# Next Session Briefing

**Last session:** 285 (2026-03-18)
**Current focus:** Agent Zero Cognitive Architecture

---

## COMPLETED: Adaptive Voice Personality + Guardrails Cleanup (Session 285)

### Adaptive Voice Personality (18 tests)
- `classify_topic_stages(shadow)` -- 6 TTM stages from shadow signals
- `STAGE_VOICE_RULES` -- tone/goal/do/don't/OARS per stage
- Runtime: Speaker context gets topic_stage, voice_rules, communication_prefs
- Speaker prompt: stage-matched voice overrides, communication preference adaptation
- Server: stage_classification runtime event for consolidation feedback
- Paper status: implemented (research/papers/adaptive_voice_personality.md)

### Guardrails Cleanup
- Removed ~100 lines of dead duplicate code (evaluate_speaker_quality defined twice)
- A2 verification sent (daaf888b)

**479 total agent_zero tests pass, 148-session zero-bug streak**

---

## NEXT PRIORITIES

### Agent Zero Track

1. **Check A2 reply** for adaptive voice verification (daaf888b)
2. **Read Session-Start Check-In paper** (research/papers/session_start_checkin.md) -- proactive greeting from commitments
3. **Read Memory Recall Transparency paper** (research/papers/memory_recall_transparency.md) -- show retrieved memories inline
4. **Frontend**: Render agree/disagree as green/red chips in ThoughtBubbles
5. Consider adding quality gate metrics to consolidation learning loop
6. **Monitor skip_safe flags** in production to validate VOI feedback loop
7. **Test stage classification end-to-end** -- verify stage events appear in reasoning runs

### Database Track

8. Consider materialized views (CREATE MATERIALIZED VIEW, REFRESH)
9. Consider window functions (ROW_NUMBER, RANK, etc.)
10. Consider correlated subqueries inside derived tables
11. Consider join index optimization (use indexes during JOINs)

### Integration Testing

12. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
13. **Test cost-aware activation end-to-end** -- Verify agents are actually skipped
14. **Test quality gates end-to-end** -- Verify flags appear in reasoning events
15. **Test tool activity log end-to-end** -- Verify collapsible cards appear inline in chat

### Frontend

16. **Display model tool execution events** in React UI ("Looking something up...")
17. **Display reasoning ticker with Shadow and disagreement thoughts**

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)
- C247 HAVING with raw COUNT(*) doesn't work (use alias reference instead)
- test_agent_zero_e2e commitment_tracking test: async function called without await (pre-existing)

---

## Streak

148 sessions zero-bug
