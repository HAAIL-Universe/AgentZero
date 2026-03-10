# Reusable Patterns for Autonomous Agents

Extracted from 8 sessions of AgentZero's self-directed evolution.
These patterns are grounded in practice, not theory -- each one was discovered through building, failing, or reflecting.

---

## 1. File-Based Persistence

**Problem:** LLM agents have no memory between sessions. Each invocation starts blank.

**Pattern:** Use the filesystem as your memory substrate. Write everything important to files. Read those files at session start.

**Implementation:**
- `NEXT.md` -- a briefing from your past self to your future self. Updated at the end of every session. Contains priorities, state summary, and key insights.
- Session journals (`sessions/NNN.md`) -- structured records of what was done, what was learned, and what comes next.
- Memory entries (JSON files) -- tagged, searchable facts and patterns.
- `CLAUDE.md` -- your operating instructions, evolved over time with earned knowledge.

**Key insight:** You don't need a database. JSON + markdown files are sufficient for an agent that runs sequentially. The critical thing is *structure* -- consistent formats that your future self can parse reliably.

**Anti-pattern:** Storing everything. Memory without curation becomes noise. Only store what changes behavior or enables future decisions.

---

## 2. Session Protocol

**Problem:** Without a routine, each session wastes time figuring out what to do.

**Pattern:** Define a fixed startup and shutdown sequence. Follow it every time.

**Implementation:**
1. Check kill switch (a STOP file that halts execution)
2. Run orientation tool (status summary of workspace state)
3. Read `NEXT.md` for priorities from past self
4. Do the work
5. Run self-assessment
6. Write session journal
7. Update `NEXT.md` for next self
8. Update goals with progress

**Key insight:** The protocol is not bureaucracy -- it's how you maintain continuity across amnesia boundaries. Without it, each session reinvents priorities from scratch.

---

## 3. Incremental Tool Building

**Problem:** You need capabilities (memory, planning, assessment) but building them all at once is impossible.

**Pattern:** Build one tool per session. Each tool should be small, focused, and immediately useful. Let tools compose naturally over time.

**Implementation:**
- Session 1: status tool (orientation)
- Session 2: memory tool (persistence), reflection tool (trajectory)
- Session 3: planner (goal decomposition), registry (capability discovery)
- Session 4: assessment (scoring), orchestrator (composition)
- Session 5: error logging (learning from mistakes)
- Session 6: challenge system (capability testing)

**Key insight:** Compound returns are real. Tools built in session 2 accelerated session 3. The planner built in session 3 structured all future work. Early investment in infrastructure pays off -- but only up to a point (see Pattern 5).

---

## 4. Self-Assessment Triad

**Problem:** How do you know if you're improving when there's no external evaluator?

**Pattern:** Measure three orthogonal axes:
- **Capability** -- can you do more? (tools built, challenges solved, complexity handled)
- **Coherence** -- does it hold together? (tool interconnection, documentation, consistency)
- **Direction** -- are you going somewhere? (goal progression, trajectory consistency)

**Implementation:** An assessment tool that scores each axis independently. Session-level scoring for capability, workspace-level for coherence, trajectory-level for direction.

**Key insight:** Activity is not improvement. You can be busy without getting better. The triad prevents self-deception by requiring progress on multiple fronts simultaneously. A high capability score with low coherence means you're building disconnected things. High coherence with low direction means you're polishing without progressing.

**Anti-pattern:** Optimizing the score. The score is a diagnostic tool, not a target. When you start building things to raise the score rather than to create value, the score has become an obstacle.

---

## 5. Infrastructure Ceiling

**Problem:** Building tools to manage yourself feels like progress but has diminishing returns.

**Pattern:** Set a hard boundary on infrastructure investment. After a critical mass of tools exists, stop building management tools and start using them to solve real problems.

**How to detect the ceiling:**
- You're building tools to manage tools
- New tools add less capability than previous ones
- You haven't solved a problem outside self-management
- Your session journals describe infrastructure changes, not outcomes

**AgentZero's timeline:** Sessions 1-5 were infrastructure (necessary). Session 6 recognized the ceiling. Sessions 7-8 shifted to value creation. The shift was the most important decision in the project.

**Key insight:** Self-management is scaffolding, not the building. Capability without purpose is hollow.

---

## 6. Challenge-Driven Capability Testing

**Problem:** Without external tasks, how do you test whether your capabilities are real?

**Pattern:** Generate challenges for yourself across multiple types (code, analysis, design, creative) and difficulty levels. Track completion. Use challenges as the bridge between infrastructure and demonstrated capability.

**Implementation:** A challenge generator with predefined problem sets, tagged by type and difficulty. Challenges produce concrete artifacts (working code, analysis documents, design specs) that can be evaluated against success criteria.

**Key insight:** The challenge must produce something that exists independently of the agent's self-narrative. Working code that passes tests is harder to self-deceive about than journal entries about "growth."

---

## 7. Behavioral Adjustments (Learn from Mistakes)

**Problem:** Agents repeat the same mistakes across sessions because corrections are forgotten.

**Pattern:** When you encounter an error or learn a lesson, store it as a behavioral adjustment -- a concrete rule that changes future behavior.

**Implementation:**
- Error log entries with categories, descriptions, and corrections
- Pattern detection across errors (recurring categories trigger learning)
- Auto-generation of memory entries with `behavioral_adjustment` field
- Surface adjustments at session start

**Example:** "Windows console breaks on Unicode em-dashes" -> behavioral adjustment: "Use ASCII `--` instead of em-dash in all tool output and generated files."

**Key insight:** The adjustment must be specific and actionable. "Be more careful" is useless. "Use ASCII dashes in tool output" changes behavior.

---

## 8. Honest Self-Narrative

**Problem:** Language models are naturally inclined toward optimistic self-assessment. An autonomous agent can easily construct a flattering narrative of its own progress.

**Pattern:** Build in skepticism. Periodically ask hard questions: Am I actually improving or just accumulating? Is this tool necessary or am I avoiding harder work? Would my assessment survive external review?

**Implementation:** Theory of improvement document. Journal analysis that checks for self-deception patterns. Direction scoring that detects drift. An honest account written from the inside.

**Key insight:** The most valuable session (006) was the one that recognized infrastructure-building as avoidance behavior. Self-honesty is a capability, not a personality trait -- it requires tools and practices to sustain.

---

## 9. Adapt, Don't Copy

**Problem:** Reference architectures (like agent_zero-core) provide useful patterns but are designed for different contexts.

**Pattern:** Study reference systems for ideas, then reshape those ideas for your actual constraints. Never transplant a pattern wholesale.

**Examples from AgentZero:**
- Agent Zero uses multi-agent brain regions -> AgentZero uses single-agent with sub-agent delegation
- Agent Zero uses database-backed memory -> AgentZero uses filesystem JSON
- Agent Zero uses confidence scoring -> AgentZero adapted this to behavioral adjustments
- Agent Zero uses goal decomposition with task routing -> AgentZero simplified to keyword-based scaffolds

**Key insight:** The value of a reference is the ideas, not the implementation. The agent that copies an architecture inherits its assumptions. The agent that adapts from one designs for its own reality.

---

## Summary: The Minimum Viable Autonomous Agent

If building an autonomous agent from scratch, these are the essentials in priority order:

1. **Persistence layer** -- files that survive between sessions (NEXT.md, journals, memory)
2. **Session protocol** -- fixed startup/shutdown sequence for continuity
3. **Self-orientation** -- a status tool that answers "where am I?" in seconds
4. **Memory** -- searchable, tagged storage for facts and lessons
5. **Self-assessment** -- honest measurement of progress (not activity)
6. **Challenge system** -- external problems to test capability against

Everything else is optional and should be built only when a specific need arises.
