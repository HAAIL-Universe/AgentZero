---
topic: Session-Start Check-In from Pending Commitments
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T16:00:00Z
---

# Session-Start Check-In from Pending Commitments

## Problem Statement

When a user opens Agent Zero, the greeting is generic -- it either asks for a name or
produces a general welcome. Agent Zero has rich commitment data (streaks, overdue check-ins,
productivity metrics) but never references it at session start. The user must navigate to
the Productivity tab to discover they have overdue commitments or broken streaks.

This is a missed opportunity. Research on proactive conversational AI and digital health
coaching consistently shows that contextual session-start greetings referencing the user's
active goals and commitments significantly improve engagement, follow-through, and trust.

The check-in engine already exists for the Telegram bot -- but it is not integrated into
the main Agent Zero WebSocket chat flow.

## Current State in Agent Zero

### Commitment Data (rich, but siloed)

**Database** (`agent_zero/database.py:236-270`):
- `commitments` table: id, user_id, title, description, cadence (once|daily|weekly|custom),
  weight (light|moderate|heavy), status, due_hint, streak_current, streak_best,
  last_checked_in, recurrence_rule
- `commitment_events` table: append-only ledger of created, completed, missed, checked_in,
  streak_broken, milestone_reached, etc.

**Productivity Summary** (`agent_zero/database.py:475-548`):
- `get_productivity_summary()` returns: active_commitments count, completed/missed this week,
  consistency_7d/30d, streaks (title, current, best), by_cadence breakdown

**Pending Check-Ins** (`agent_zero/agent_zero_server.py:963-1002`):
- `GET /user/checkins/pending` endpoint exists
- Returns commitments overdue based on cadence (daily: >24h, weekly: >7d since last check-in)
- Data is correct and available but ONLY used by the Productivity tab UI

### Check-In Engine (exists, not connected to main chat)

**Telegram Check-In Engine** (`agent_zero/agent_zero_telegram/check_in_engine.py:1-509`):
- Three scoring types: event-anchored (0.7-1.0), pattern-triggered (0.5-0.8),
  silence-based (0.3-0.6)
- Detects: stale goals (3+ days), declining streaks (current < 30% of best),
  consistency < 50%, broken streaks (best >= 5, current == 0), repeated misses
- Scoring threshold, daily cap, time-of-day gate
- Message generation via Agent Zero LLM with shadow context, fallback templates

**Critical gap**: This engine is ONLY wired to Telegram. It is not invoked during
WebSocket session start in `agent_zero_server.py`.

### Current Greeting Logic

**Session Start** (`agent_zero/agent_zero_server.py:3259-3274`):
- Fires only on first user turn (user_turns <= 1)
- Detects greeting-like tokens: "hello", "hi", "hey", "morning"
- Response: "If you'd like, tell me what you'd like me to call you."
- No reference to commitments, goals, streaks, or pending check-ins
- No proactive message -- Agent Zero ONLY speaks after the user speaks first

**Shadow Prompt** (`agent_zero/agent_zero_server.py:715-761`):
- Builds contextual intro based on session count (e.g., "returning user" vs "new user")
- Includes shadow data but NOT commitment status
- Does not inject pending check-ins or streak data into the prompt

### UI

**CheckInsView.tsx** (`agent_zero-ui/src/components/CheckInsView.tsx:1-119`):
- Shows pending check-ins with overdue time
- Actions: "Done" (with note) or "Skip" (with reason)
- Only visible when user navigates to Productivity > Check-Ins tab

**CommitmentsView.tsx** (`agent_zero-ui/src/components/CommitmentsView.tsx:1-192`):
- Full commitment management with streaks, events, status changes
- Only visible in Productivity > Commitments tab

### What's Missing

1. No proactive greeting at session start referencing commitments
2. Check-in engine not connected to WebSocket chat flow
3. No commitment data injected into initial prompt context
4. No UI affordance in the chat thread for commitment status
5. User must self-navigate to discover overdue items

## Industry Standard / Research Findings

### 1. Proactive Conversational AI Taxonomy

**Deng et al. (2025)** published a comprehensive survey in ACM TOIS defining proactive
conversational AI as systems that "autonomously take initiatives in conversations" rather
than only responding to user prompts. The survey identifies three key elements: the system
must (a) anticipate user needs, (b) take initiative at appropriate moments, and
(c) maintain user control and civility.

For session-start behavior, the survey notes that proactive systems should leverage
contextual triggers (prior session data, pending tasks) rather than time-based triggers
alone.

URL: https://dl.acm.org/doi/10.1145/3715097

### 2. Inner Thoughts Framework for Proactive Agents

**Arxiv 2501.00383 (2025)** proposes that proactive agents should maintain "inner thoughts"
-- a continuous cognitive process parallel to conversation. The system uses intrinsic
motivation scoring (1-5 scale) to decide when to initiate. Key heuristics for deciding
to speak: relevance, information gaps, expected impact, urgency, coherence, originality.

For session start, the framework suggests using "on_pause" triggers (silence detection)
with lower motivation thresholds during early engagement to "establish conversational
momentum." This maps to: when a user opens a session and hasn't spoken yet, the system
should evaluate whether pending commitments create sufficient motivation to speak first.

URL: https://arxiv.org/html/2501.00383v2

### 3. Human-Centered Proactive Agent Design (IAC Framework)

**Arxiv 2404.12670 (2024)** establishes the Intelligence-Adaptivity-Civility (IAC) taxonomy:
- **Intelligence**: Anticipate future needs and plan strategically
- **Adaptivity**: Adjust timing and pacing to real-time context
- **Civility**: Respect user boundaries; avoid being intrusive

The paper warns that "frequent clarification requests can negatively impact user experience"
and recommends opt-in mechanisms and user control over proactive behavior. For session-start
check-ins, this means: (a) only fire when there's meaningful data, (b) keep it brief,
(c) give the user easy ways to dismiss or defer.

URL: https://arxiv.org/html/2404.12670v1

### 4. AI-Assisted Goal Reflection During Sessions

**CHI 2025** paper on AI-assisted goal tracking found that two proactive modalities work:
- **Ambient Visualization**: Passive, continuous feedback on goal alignment (non-intrusive)
- **Interactive Questioning**: Active nudges to reflect on whether activities align with goals

The ambient approach (showing status without requiring action) was less disruptive and
better received for session starts, while interactive questioning was better mid-session.

URL: https://dl.acm.org/doi/10.1145/3706598.3714052

### 5. Digital Behavior Change Interventions for Habit Formation

A systematic review (PMC, 2024) of 41 studies on digital habit formation interventions found:
- **Prompts and cues** (80%), **goal setting** (65%), and **self-monitoring** (60%) are
  the three most prevalent behavior change techniques
- Time-based cues are common (54%) but have "low user response to pushing notification"
- **Just-in-time adaptive interventions** -- context-sensitive delivery based on detected
  patterns rather than fixed schedules -- are the emerging best practice
- "Gaps between progress and goals provide explicit information to motivate individuals"
- User-controlled reminders ("remind at self-decided time") preserve autonomy

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11161714/

### 6. Motivational Interviewing Digital Applications

**PMC 2025** review on MI in digital health found:
- Brief MI (under 20 minutes) remains effective for behavior change
- 3-4 sessions optimal for complex behaviors
- AI-powered conversational agents "can provide scalable, real-time support for
  self-management, offering personalized feedback and reinforcement between sessions"
- The OARS model (Open-ended questions, Affirmations, Reflections, Summaries) is the
  gold standard for engagement

For session-start check-ins, the MI approach recommends: open with an affirmation of
what the user HAS done (streak acknowledgment), then a reflective observation about
status, then an open question inviting the user to share.

URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12526391/

### 7. Digital Health Coaching Systematic Review

**Frontiers in Digital Health (2025)** systematic review of 35 studies found that hybrid
human-AI coaching, combining automated nudges with contextual intervention, demonstrated
the highest engagement and outcome scores. The review specifically noted that "combining
human coaching with automated support delivered through in-app messages, nudges, and short
tips" was the most effective modality.

URL: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1536416/full

### 8. Google CC "Day Ahead" Briefing Pattern

Google's CC agent (launched December 2025) delivers a daily "Your Day Ahead" briefing that
aggregates calendar, email, and task data into a proactive session-start summary. This
product pattern validates that users expect AI companions to proactively surface relevant
status information at session start, not wait to be asked.

URL: https://www.alpha-sense.com/resources/research-articles/proactive-ai/

## Proposed Implementation

### Architecture: Session-Start Commitment Briefing

When a user opens a new session, Agent Zero evaluates whether pending commitments warrant
a proactive greeting. If yes, it sends a brief, MI-informed check-in message as the
first message in the chat -- before the user speaks.

### Design Principles (from research)

1. **Just-in-time adaptive** (PMC 2024): Only fire when there's meaningful data, not on a timer
2. **Ambient over interactive** (CHI 2025): Show status, don't demand action
3. **MI-informed** (PMC 2025): Lead with affirmation, then reflection, then open question
4. **Civility** (IAC framework): Keep brief, allow dismissal, respect user boundaries
5. **Intrinsic motivation** (Arxiv 2025): Score whether the check-in adds value

### Step 1: Session-Start Check-In Evaluator

**File: `agent_zero/session_checkin.py`** (NEW)

Port and adapt the scoring logic from `check_in_engine.py` for the WebSocket flow.
Simplify to a single function that takes user data and returns whether to fire and what
context to include.

```python
from datetime import datetime, timezone, timedelta

# Thresholds (tunable)
MOTIVATION_THRESHOLD = 0.5  # minimum score to trigger check-in
MAX_ITEMS_IN_GREETING = 3   # don't overwhelm

async def evaluate_session_checkin(user_id: str, db) -> dict | None:
    """Evaluate whether a proactive check-in should fire at session start.

    Returns None if no check-in warranted, or a dict with:
      {
        "score": float,           # motivation score 0-1
        "trigger": str,           # what triggered it
        "pending_checkins": [...], # overdue commitments
        "streaks": [...],         # notable streaks (declining or milestone)
        "productivity": {...},    # summary metrics
        "suggested_tone": str,    # "celebration" | "gentle_nudge" | "concern"
      }
    """
    # 1. Fetch pending check-ins
    pending = await get_pending_checkins(user_id, db)

    # 2. Fetch productivity summary
    productivity = await get_productivity_summary(user_id, db)

    # 3. Fetch streak data
    streaks = productivity.get("streaks", [])

    # 4. Score motivation
    score = 0.0
    trigger = None
    tone = "gentle_nudge"

    # High-value triggers
    if any(s["current"] > 0 and s["current"] >= s["best"] for s in streaks):
        score = max(score, 0.8)
        trigger = "streak_milestone"
        tone = "celebration"

    if any(s["best"] >= 5 and s["current"] == 0 for s in streaks):
        score = max(score, 0.7)
        trigger = "broken_streak"
        tone = "concern"

    if len(pending) >= 2:
        score = max(score, 0.65)
        trigger = trigger or "multiple_overdue"

    if productivity.get("consistency_7d", 1.0) < 0.5:
        score = max(score, 0.6)
        trigger = trigger or "low_consistency"
        tone = "concern"

    if len(pending) == 1:
        score = max(score, 0.5)
        trigger = trigger or "single_overdue"

    # Celebrate high consistency
    if productivity.get("consistency_7d", 0) >= 0.9 and productivity.get("active_commitments", 0) > 0:
        score = max(score, 0.55)
        trigger = trigger or "high_consistency"
        tone = "celebration"

    if score < MOTIVATION_THRESHOLD:
        return None

    return {
        "score": score,
        "trigger": trigger,
        "pending_checkins": pending[:MAX_ITEMS_IN_GREETING],
        "streaks": streaks,
        "productivity": productivity,
        "suggested_tone": tone,
    }
```

### Step 2: Inject Check-In Context into Initial Prompt

**File: `agent_zero/agent_zero_server.py`**

In the session start / first-turn handling section (around line 3259), call
`evaluate_session_checkin()` and, if it returns data, inject it into the system prompt
as a grounding section:

```python
# After session creation, before first response generation:
checkin_data = await evaluate_session_checkin(user_id, db)

if checkin_data:
    checkin_prompt = _build_checkin_prompt(checkin_data)
    # Append to system prompt or inject as a system message
```

The `_build_checkin_prompt` function renders the check-in data into natural language
guidance for the LLM:

```python
def _build_checkin_prompt(data: dict) -> str:
    parts = []
    parts.append("SESSION CHECK-IN CONTEXT (proactive greeting data):")
    parts.append(f"Trigger: {data['trigger']} (score: {data['score']:.2f})")
    parts.append(f"Suggested tone: {data['suggested_tone']}")

    if data["pending_checkins"]:
        titles = [c["title"] for c in data["pending_checkins"]]
        parts.append(f"Overdue check-ins: {', '.join(titles)}")

    prod = data["productivity"]
    if prod.get("active_commitments"):
        parts.append(f"Active commitments: {prod['active_commitments']}")
        parts.append(f"7-day consistency: {prod.get('consistency_7d', 0):.0%}")

    for s in data["streaks"]:
        if s["current"] >= s["best"] and s["current"] > 0:
            parts.append(f"STREAK MILESTONE: '{s['title']}' at {s['current']} days (personal best!)")
        elif s["best"] >= 5 and s["current"] == 0:
            parts.append(f"BROKEN STREAK: '{s['title']}' was at {s['best']} days, now 0")

    parts.append("")
    parts.append("INSTRUCTION: Open the session with a brief, warm check-in that:")
    parts.append("- Acknowledges the user by name if known")
    parts.append("- If tone is 'celebration': lead with affirmation of their progress")
    parts.append("- If tone is 'gentle_nudge': mention overdue items naturally, not as a lecture")
    parts.append("- If tone is 'concern': express care, not judgment")
    parts.append("- End with an open question inviting them to share")
    parts.append("- Keep it to 2-3 sentences maximum")
    parts.append("- Do NOT list all commitments -- mention at most one or two by name")
    parts.append("- The user should feel welcomed, not interrogated")

    return "\n".join(parts)
```

### Step 3: Proactive First Message (Server-Initiated)

**File: `agent_zero/agent_zero_server.py`**

After session creation on WebSocket connect, if `checkin_data` is non-null, generate
and send a proactive greeting message WITHOUT waiting for user input:

1. Call the LLM with the check-in prompt injected into context
2. Stream the response as a `agent_zero` message
3. Set a flag so the normal greeting logic doesn't also fire

```python
# In WebSocket handler, after session setup:
if checkin_data:
    # Generate proactive check-in greeting
    checkin_prompt = _build_checkin_prompt(checkin_data)
    # Add to system context and generate
    await _generate_proactive_greeting(websocket, checkin_prompt, session_context)
    session_state["greeted"] = True  # prevent duplicate greeting
```

### Step 4: Frontend -- Handle Server-Initiated Message

**File: `agent_zero-ui/src/App.tsx`**

The existing WebSocket message handler already handles `token` and `done` events for
streaming responses. A server-initiated proactive greeting uses the same protocol -- it
sends `status` ("thinking"), then `token` chunks, then `done`. No frontend changes needed
for basic support.

Optional enhancement: Add a visual indicator that this is a proactive check-in (e.g., a
small "check-in" badge or subtle background tint on the message bubble):

```typescript
// In message rendering, if message metadata includes checkin flag:
{msg.metadata?.is_checkin && (
  <span className="checkin-badge">Check-in</span>
)}
```

### Step 5: User Control

**File: `agent_zero/agent_zero_server.py`** or user preferences

Add a user preference to control proactive check-ins:

```python
# User preferences (stored in DB or shadow):
{
    "proactive_checkins": "on" | "off" | "silent"
    # "on": proactive greeting with commitment data
    # "off": no proactive greeting
    # "silent": commitment data injected into context but no proactive message
    #           (Agent Zero uses it naturally if relevant, but doesn't lead with it)
}
```

Default: `"on"` for users with active commitments, `"silent"` for new users.

### Step 6: CSS for Check-In Badge (optional)

**File: `agent_zero-ui/src/styles.css`**

```css
.checkin-badge {
  display: inline-block;
  font-size: 0.6rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(76, 175, 80, 0.15);
  color: rgba(76, 175, 80, 0.8);
  margin-left: 8px;
  vertical-align: middle;
}
```

### Interaction with Existing Systems

- **Telegram check-in engine**: The session evaluator reuses the same scoring logic but
  is decoupled. Both can coexist -- Telegram handles out-of-session proactive outreach,
  WebSocket handles in-session greeting.
- **Greeting logic** (line 3259): Guard with `if not session_state.get("greeted")` so
  the proactive check-in replaces the generic greeting, not duplicates it.
- **Shadow prompt** (lines 715-761): The check-in context is injected alongside the shadow,
  not replacing it. The LLM sees both personality context and commitment data.
- **Pending check-ins API**: The evaluator calls the same DB queries. No API duplication.
- **Tool selection**: If the check-in injects commitment data, the tool selector can skip
  redundant `commitments.list` calls on the first turn.

### Migration Path

1. Create `agent_zero/session_checkin.py` with `evaluate_session_checkin()` and scoring logic
2. Add `_build_checkin_prompt()` helper in `agent_zero_server.py`
3. Call evaluator on WebSocket session creation, before first user message
4. If check-in fires, generate and stream proactive greeting
5. Guard existing greeting logic to not double-fire
6. Optional: Add user preference for proactive check-in mode
7. Optional: Add check-in badge to frontend message rendering
8. Test with real user data

## Test Specifications

### Backend Tests (`agent_zero/test_session_checkin.py`)

```python
def test_no_checkin_when_no_commitments():
    """Returns None when user has no active commitments."""
    # Given: user with 0 active commitments, no pending check-ins
    # Then: evaluate_session_checkin returns None

def test_checkin_fires_for_single_overdue():
    """Returns check-in data when one commitment is overdue."""
    # Given: user with 1 daily commitment, last_checked_in > 24h ago
    # Then: returns dict with trigger="single_overdue", score >= 0.5

def test_checkin_fires_for_multiple_overdue():
    """Returns check-in data when multiple commitments are overdue."""
    # Given: user with 3 daily commitments, 2 overdue
    # Then: trigger="multiple_overdue", score >= 0.65
    # Then: pending_checkins has length <= MAX_ITEMS_IN_GREETING

def test_checkin_fires_for_broken_streak():
    """Returns check-in data when a significant streak is broken."""
    # Given: commitment with streak_best=10, streak_current=0
    # Then: trigger="broken_streak", tone="concern", score >= 0.7

def test_checkin_fires_for_streak_milestone():
    """Returns check-in data when user hits personal best streak."""
    # Given: commitment with streak_current=15, streak_best=15
    # Then: trigger="streak_milestone", tone="celebration", score >= 0.8

def test_checkin_fires_for_low_consistency():
    """Returns check-in data when 7-day consistency drops below 50%."""
    # Given: consistency_7d = 0.3
    # Then: trigger="low_consistency", tone="concern", score >= 0.6

def test_checkin_fires_for_high_consistency():
    """Returns celebration check-in for high consistency."""
    # Given: consistency_7d = 0.95, active_commitments > 0
    # Then: trigger="high_consistency", tone="celebration"

def test_no_checkin_below_threshold():
    """Returns None when all scores are below MOTIVATION_THRESHOLD."""
    # Given: user with 0 pending, no broken streaks, consistency=0.8
    # Then: returns None

def test_checkin_prompt_includes_tone():
    """_build_checkin_prompt includes tone instruction."""
    # Given: data with suggested_tone="celebration"
    # Then: prompt contains "celebration"

def test_checkin_prompt_limits_commitment_names():
    """_build_checkin_prompt mentions at most 2 commitments by name."""
    # Given: data with 5 pending check-ins
    # Then: prompt instruction says "at most one or two by name"

def test_checkin_prompt_includes_streak_milestone():
    """_build_checkin_prompt highlights streak milestones."""
    # Given: streak with current == best == 20
    # Then: prompt contains "STREAK MILESTONE"

def test_checkin_respects_user_preference_off():
    """No check-in when user preference is 'off'."""
    # Given: user preference proactive_checkins="off"
    # Then: evaluate returns None regardless of data

def test_checkin_silent_mode():
    """Silent mode returns data but no proactive message flag."""
    # Given: user preference proactive_checkins="silent"
    # Then: data is injected into context but no proactive message generated

def test_greeting_guard_prevents_duplicate():
    """Proactive greeting sets greeted flag, preventing generic greeting."""
    # Given: check-in fires and sets session_state["greeted"] = True
    # Then: generic greeting logic (line 3259) does not also fire
```

### Frontend Tests (if check-in badge is implemented)

```
test: check-in badge renders when message has is_checkin metadata
test: check-in badge does not render for normal messages
test: check-in badge has correct styling (green, small, uppercase)
```

## Estimated Impact

**Engagement**: Research shows proactive AI companions that reference user context at
session start have significantly higher engagement rates. Google CC's "Day Ahead" briefing
pattern validates that users expect AI to proactively surface relevant status. The CHI 2025
study found ambient goal reflection (showing status without demanding action) was the
least intrusive and best-received modality.

**Commitment follow-through**: The PMC 2024 systematic review found prompts/cues (80%
prevalence) and self-monitoring feedback (60%) are the top behavior change techniques.
A session-start check-in combines both: it's a prompt (nudge) that surfaces
self-monitoring data (streak, consistency). Streak visualization is proven to strengthen
commitment (Business of Apps 2025).

**User trust**: The IAC framework (Arxiv 2024) and ScienceDirect (2025) show that proactive
agents that demonstrate intelligence (anticipating needs), adaptivity (appropriate timing),
and civility (brief, non-intrusive) build trust. The MI-informed tone (affirmation first,
then reflection, then open question) avoids the "nagging app" pattern.

**Reduced friction**: Currently, commitment data requires 3 clicks to reach (Productivity
tab > Commitments/Check-Ins). The proactive greeting surfaces the most important
information immediately, reducing friction from 3 clicks to 0.

**Performance**: One additional DB query at session start (~10-20ms). One LLM call for
greeting generation (piggybacks on the first turn's generation slot). Negligible impact.

## References

1. Deng et al. (2025). "Proactive Conversational AI: A Comprehensive Survey." ACM TOIS.
   https://dl.acm.org/doi/10.1145/3715097

2. Arxiv 2501.00383 (2025). "Proactive Conversational Agents with Inner Thoughts."
   https://arxiv.org/html/2501.00383v2

3. Arxiv 2404.12670 (2024). "Towards Human-centered Proactive Conversational Agents."
   https://arxiv.org/html/2404.12670v1

4. CHI 2025. "Are We On Track? AI-Assisted Active and Passive Goal Reflection."
   https://dl.acm.org/doi/10.1145/3706598.3714052

5. PMC (2024). "Digital Behavior Change Intervention Designs for Habit Formation."
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11161714/

6. PMC (2025). "Motivational Interviewing to Promote Healthy Lifestyle Behaviors."
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12526391/

7. Frontiers in Digital Health (2025). "Systematic review: human, AI, and hybrid coaching."
   https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1536416/full

8. Alpha-Sense (2025). "Proactive AI in 2026: Moving Beyond the Prompt" (Google CC pattern).
   https://www.alpha-sense.com/resources/research-articles/proactive-ai/

9. ScienceDirect (2025). "Transparency in AI-powered digital agents."
   https://www.sciencedirect.com/science/article/pii/S2444569X25001155
