# Next Session Briefing

**Last session:** 255 (2026-03-17)
**Current focus:** Agent Zero Integration & Optimization (Overseer directive)

---

## COMPLETED: Bug Fixes & Routing (Session 255)

- Fixed strategic routing: short intent messages now route correctly (added strong keywords)
- Fixed A2 HIGH findings: greedy regex, error leakage, silent exceptions, unsafe dict access
- Updated TOOL_MANIFEST.md with analysis.request and analysis.results
- 229/229 tests passing (fixed 2 pre-existing failures from session 254)

---

## NEXT PRIORITIES: Agent Zero Hardening & Testing

### Security Fixes (from A2 review)

1. **CORS wildcard + credentials** (agent_zero_server.py:103-109) -- Fix: explicit origin list
2. **Singleton inference race** (agent_zero_server.py) -- Fix: add asyncio.Lock
3. **Unbounded message accumulation** -- Fix: cap session_messages at ~200
4. **Prompt injection via interpolation** (cognitive_agents.py:231-257) -- Fix: escape user state

### Integration Testing

5. **Test with live vLLM model** -- Verify tool interception, continuation, and context compression end-to-end
6. **A2 round-trip test** -- Send real analysis.request, verify analysis.results picks it up

### Training Pipeline

7. **Generate training data** for analysis.request and analysis.results tools
8. **Validate existing training data** against current tool specs

### Frontend

9. **Display model tool execution events** in React UI ("Looking something up...")

---

## Known Bugs (carry forward)

- C037 SMT Simplex precision issues (non-critical)
- assess.py OSError on assessments.json (non-critical)
- V076 parity_games Phase 4 bug (V080 workaround)

---

## Streak

122 sessions zero-bug
