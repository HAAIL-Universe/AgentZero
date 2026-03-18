---
topic: Proactive Conversation Starters
status: implemented
priority: low
estimated_complexity: medium
researched_at: 2026-03-18T22:30:00Z
---

# Proactive Conversation Starters

## Problem Statement

Agent Zero is purely reactive on the web interface: it only speaks when spoken to. The session check-in (session_checkin.py) fires exactly once at session start, and only if the user has overdue commitments or streak changes. Between sessions, the only proactive channel is Telegram (which requires separate bot setup). The web UI has no mechanism for Agent Zero to initiate a message mid-session or between sessions.

This is a significant gap. Research on proactive conversational AI (Deng et al., 2025) shows that systems equipped with inner-thought-driven proactive behavior significantly outperform reactive-only systems on engagement, perceived intelligence, and initiative. JITAI (Just-In-Time Adaptive Intervention) research (Nahum-Shani et al., 2018) establishes that interventions timed to vulnerability/opportunity/receptivity states are dramatically more effective than fixed-schedule or purely reactive approaches.

The capability manifest already marks `proactive_outreach_web` as `available: False` and `real_time_monitoring` as `available: False` (capability_manifest.py:61-66). This paper provides the implementation plan to change both to `True`.

## Current State in Agent Zero

### What exists:
1. **Session check-in** (`session_checkin.py`): evaluates at session start whether to greet with commitment status. Fires once per session. Uses motivation scoring (streak milestones, broken streaks, overdue items, consistency).
2. **Telegram check-in engine** (`agent_zero_telegram/app.py:244`): scheduled proactive messages via Telegram with `apscheduler`. Supports daily reminders, weekly analysis.
3. **Capability manifest** (`capability_manifest.py:60-66`): `real_time_monitoring: False`, `proactive_outreach_web: False`.
4. **Behavioral insights** (`behavioral_insights.py`): topic classification, stage-of-change, regime inference -- all the data needed to decide WHEN to intervene.
5. **Shadow profile** (`behavioural_shadow.py`): tracks commitment follow-through rates, patterns, predictions -- the "vulnerability" and "opportunity" signals.

### What's missing:
1. **SSE endpoint** for server-to-client push on the web UI
2. **Proactive evaluation loop** that runs server-side on a timer or event trigger
3. **Trigger taxonomy** defining what events warrant proactive messages
4. **Receptivity model** to avoid over-messaging and reactance
5. **Frontend listener** that renders proactive messages in the chat thread

## Industry Standard / Research Findings

### 1. Proactive Conversational AI Survey (Deng et al., 2025)
Comprehensive ACM TOIS survey identifying proactive conversational AI as systems that "lead conversations towards achieving pre-defined targets or fulfilling specific goals." The survey categorizes proactive behaviors into: target-guided dialogue, non-collaborative dialogue, enriched dialogue, and initiative-taking. The key finding: proactivity requires (a) trigger detection, (b) thought formation, (c) timing evaluation, and (d) participation decision.
URL: https://dl.acm.org/doi/10.1145/3715097

### 2. Inner Thoughts Framework (Deng et al., CHI 2025)
A five-stage framework for proactive conversational agents: (1) Trigger -- detect events that warrant proactive behavior, (2) Retrieval -- gather relevant context, (3) Thought Formation -- compose the proactive message internally, (4) Evaluation -- assess whether now is the right time (receptivity), (5) Participation -- deliver or suppress. User study showed significant improvements in turn appropriateness (+15%), perceived engagement (+22%), and initiative (+31%) over reactive baselines.
URL: https://dl.acm.org/doi/10.1145/3706598.3713760

### 3. JITAI Design Framework (Nahum-Shani et al., 2018)
Defines five components for adaptive interventions: decision points, intervention options, tailoring variables, decision rules. Distinguishes between **vulnerability states** (susceptibility to negative outcomes), **opportunity states** (susceptibility to positive behavior change), and **receptivity** (willingness to receive support). Key insight: timing should be event-based (state change triggers), not clock-based (fixed schedules).
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5364076/

### 4. JITAI with Trigger Detection and Generative Chatbot (Bosschaerts et al., 2025)
Architecture integrating trigger detection to determine optimal intervention moments with LLM-generated personalized support. Combines passive sensing (behavior signals) with active assessment (self-report) to detect vulnerability/opportunity windows. Uses prompt engineering on the LLM to generate contextually appropriate proactive messages.
URL: https://journals.sagepub.com/doi/10.1177/20552076251381747

### 5. Server-Sent Events (SSE) Standard (MDN, W3C)
Unidirectional server-to-client push over HTTP. The `EventSource` API is supported in all modern browsers, provides automatic reconnection, and is the standard pattern for proactive chatbot messages (used by ChatGPT for streaming). Simpler than WebSockets for server-initiated messages. Compatible with FastAPI's `StreamingResponse`.
URL: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events

### 6. Digital Health Coaching Engagement (Frontiers, 2025)
Systematic review of human, AI, and hybrid health coaching finding that proactive interventions increase engagement 2-3x over reactive-only systems, but must be personalized and well-timed to avoid "chatbot fatigue." Poorly timed proactive messages cause psychological reactance and decreased engagement.
URL: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1536416/full

## Proposed Implementation

### Architecture Overview

```
[Browser]                    [Server]
    |                            |
    |--- GET /api/events ------->|  SSE connection
    |<--- event: proactive ------|
    |<--- event: proactive ------|  (periodic evaluation)
    |                            |
    |  EventSource auto-reconnect|
```

### 1. SSE Endpoint: `agent_zero_server.py`

Add a new SSE endpoint that maintains a persistent connection per session:

```python
# In agent_zero_server.py

from starlette.responses import StreamingResponse
import asyncio
import json

# Per-session event queues
_proactive_queues: dict[str, asyncio.Queue] = {}

@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events endpoint for proactive messages."""
    session_id = _extract_session_id(request)
    if not session_id:
        return JSONResponse({"error": "No session"}, status_code=401)

    queue = asyncio.Queue()
    _proactive_queues[session_id] = queue

    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: proactive\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            _proactive_queues.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

### 2. Proactive Evaluator: `agent_zero/proactive_evaluator.py` (NEW)

The core logic that decides WHEN and WHAT to send proactively. Implements the JITAI decision framework with Agent Zero-specific triggers.

```python
"""Proactive conversation starter evaluator.

Implements JITAI-style trigger detection for web UI proactive messages.
Uses shadow profile and behavioral insights to detect vulnerability,
opportunity, and receptivity states.
"""

from datetime import datetime, timezone, timedelta
from behavioral_insights import classify_topic_stages, infer_behavioral_regime

# --- Trigger Taxonomy ---
# Each trigger has: name, detector function, priority, cooldown

TRIGGERS = [
    # Opportunity triggers (positive moments to capitalize on)
    ("streak_milestone",    60 * 60 * 4),   # 4hr cooldown
    ("momentum_building",   60 * 60 * 8),   # 8hr cooldown
    ("commitment_due_soon", 60 * 60 * 2),   # 2hr cooldown

    # Vulnerability triggers (moments needing support)
    ("pattern_risk",        60 * 60 * 6),   # 6hr cooldown
    ("long_absence",        60 * 60 * 24),  # 24hr cooldown

    # Engagement triggers (conversation enrichment)
    ("idle_in_session",     60 * 5),        # 5min cooldown
    ("insight_available",   60 * 60 * 12),  # 12hr cooldown
]

MAX_PROACTIVE_PER_SESSION = 3  # hard cap per session
MIN_SECONDS_BETWEEN = 300      # 5 min minimum gap


async def evaluate_proactive_triggers(
    user_id: str,
    session_id: str,
    shadow_profile: dict,
    db_module,
    sent_count: int = 0,
    last_sent_at: datetime | None = None,
) -> dict | None:
    """Evaluate whether a proactive message should fire.

    Returns None if no trigger, or a dict with:
      trigger, message_hint, tone, priority, data
    """
    if sent_count >= MAX_PROACTIVE_PER_SESSION:
        return None

    now = datetime.now(timezone.utc)
    if last_sent_at and (now - last_sent_at).total_seconds() < MIN_SECONDS_BETWEEN:
        return None

    # Receptivity check: don't interrupt active typing
    # (frontend sends typing indicators; server tracks last_user_message_at)

    # Check triggers in priority order
    result = None

    # 1. Streak milestone approaching
    commitments = await db_module.list_commitments(user_id, status="active")
    productivity = await db_module.get_productivity_summary(user_id)
    streaks = productivity.get("streaks", [])

    for s in streaks:
        if s["current"] > 0 and s["current"] == s["best"]:
            result = {
                "trigger": "streak_milestone",
                "message_hint": f"You're at a personal best on '{s['title']}' -- {s['current']} days!",
                "tone": "celebration",
                "priority": 0.8,
            }
            break

    # 2. Commitment due soon (within 2 hours)
    if not result:
        for c in commitments:
            deadline = c.get("target_date")
            if deadline:
                if isinstance(deadline, str):
                    try:
                        deadline = datetime.fromisoformat(deadline)
                    except Exception:
                        continue
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                hours_left = (deadline - now).total_seconds() / 3600
                if 0 < hours_left <= 2:
                    result = {
                        "trigger": "commitment_due_soon",
                        "message_hint": f"'{c['title']}' is due in {hours_left:.0f} hours",
                        "tone": "gentle_nudge",
                        "priority": 0.7,
                    }
                    break

    # 3. Pattern risk: shadow predicts low follow-through
    if not result and shadow_profile:
        regime = infer_behavioral_regime(shadow_profile)
        if regime == "intent_without_followthrough":
            active_items = [c for c in commitments if c.get("cadence") != "once"]
            if active_items:
                result = {
                    "trigger": "pattern_risk",
                    "message_hint": "Check in on active commitments given follow-through pattern",
                    "tone": "curious",
                    "priority": 0.6,
                }

    # 4. Long absence: user hasn't been active in >48 hours
    # (checked via last_active timestamp in session)

    # 5. Idle in session: user connected but no message for 5+ minutes
    # (tracked via SSE connection age vs last message timestamp)

    # 6. New insight available from consolidation
    # (checked via consolidation rule timestamps)

    return result
```

### 3. Background Evaluation Loop

Add a periodic task in `agent_zero_server.py` that evaluates triggers for connected sessions:

```python
# In agent_zero_server.py, add to startup

async def _proactive_evaluation_loop():
    """Periodically evaluate proactive triggers for connected sessions."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        for session_id, queue in list(_proactive_queues.items()):
            try:
                # Look up user_id and shadow_profile for this session
                # ... (from session store)
                result = await evaluate_proactive_triggers(
                    user_id=user_id,
                    session_id=session_id,
                    shadow_profile=shadow_profile,
                    db_module=database,
                    sent_count=_proactive_sent_counts.get(session_id, 0),
                    last_sent_at=_proactive_last_sent.get(session_id),
                )
                if result:
                    await queue.put(result)
                    _proactive_sent_counts[session_id] = _proactive_sent_counts.get(session_id, 0) + 1
                    _proactive_last_sent[session_id] = datetime.now(timezone.utc)
            except Exception:
                pass  # Don't crash evaluation loop
```

### 4. Frontend SSE Listener

Add to the chat HTML/JS (in the existing `agent_zero_server.py` HTML template or separate JS):

```javascript
// Proactive message listener
const evtSource = new EventSource('/api/events');

evtSource.addEventListener('proactive', function(event) {
    const data = JSON.parse(event.data);
    // Render as a system message in the chat thread
    appendProactiveMessage(data);
});

function appendProactiveMessage(data) {
    const chatContainer = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message proactive-message';
    msgDiv.innerHTML = `
        <div class="proactive-indicator">Agent Zero noticed something</div>
        <div class="proactive-content">${data.message_hint}</div>
    `;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
```

### 5. Update Capability Manifest

```python
# In capability_manifest.py, change:
"real_time_monitoring": {
    "available": True,
    "via": "sse",
    "note": "Server-Sent Events provide real-time proactive messages during active sessions.",
},
"proactive_outreach_web": {
    "available": True,
    "via": "sse",
    "note": "Proactive messages delivered via SSE to web UI during active sessions.",
},
```

### Files to Modify

1. **NEW: `agent_zero/proactive_evaluator.py`** -- trigger taxonomy, JITAI evaluation logic (~120 lines)
2. **MODIFY: `agent_zero/agent_zero_server.py`** -- add `/api/events` SSE endpoint, background evaluation loop, proactive queue management (~80 lines)
3. **MODIFY: `agent_zero/capability_manifest.py`** -- update 2 capability entries to `available: True`
4. **MODIFY: chat HTML/JS in `agent_zero_server.py`** -- add EventSource listener and proactive message rendering (~30 lines JS, ~20 lines CSS)

### Receptivity Safeguards (Anti-Reactance)

Per the digital health coaching literature (Frontiers, 2025) and JITAI framework:

1. **Hard cap**: MAX_PROACTIVE_PER_SESSION = 3 (configurable via Agent ZeroConfig if configuration paper is implemented first)
2. **Minimum gap**: 5 minutes between proactive messages
3. **Cooldown per trigger type**: Different cooldowns prevent repetitive nudging
4. **No interruption**: Don't send proactive messages while user is actively typing (track via typing indicators)
5. **Suppress on engagement**: If user is actively messaging (message in last 60 seconds), defer proactive messages
6. **Regime-aware**: For users in "guarded_avoidant" regime, reduce proactive frequency by 50%
7. **User control**: Add a "Pause proactive messages" toggle in the UI

### Backward Compatibility

- SSE endpoint is additive (new route, no existing routes modified)
- Frontend listener is additive (new JS, no existing behavior changed)
- If SSE connection is not established (old client), system behaves exactly as before
- Session check-in (session_checkin.py) continues to function independently
- Telegram proactive messages continue independently

## Test Specifications

### test_proactive_evaluator.py

```python
# Test 1: No trigger when no commitments
def test_no_trigger_without_commitments():
    """evaluate_proactive_triggers returns None when user has no commitments."""

# Test 2: Streak milestone trigger fires
def test_streak_milestone_trigger():
    """When user has current == best streak, trigger fires with celebration tone."""

# Test 3: Commitment due soon trigger fires
def test_commitment_due_soon():
    """When commitment deadline is within 2 hours, trigger fires with gentle_nudge tone."""

# Test 4: Pattern risk trigger fires for intent_without_followthrough
def test_pattern_risk_trigger():
    """When regime is intent_without_followthrough and active commitments exist, trigger fires."""

# Test 5: Max proactive per session enforced
def test_max_proactive_cap():
    """Returns None when sent_count >= MAX_PROACTIVE_PER_SESSION."""

# Test 6: Minimum gap enforced
def test_minimum_gap_enforcement():
    """Returns None when last_sent_at is within MIN_SECONDS_BETWEEN."""

# Test 7: Priority ordering
def test_trigger_priority_ordering():
    """Streak milestone (0.8) takes precedence over commitment_due_soon (0.7)."""

# Test 8: No false triggers on empty shadow profile
def test_empty_shadow_profile():
    """Returns None or non-vulnerability trigger when shadow_profile is empty."""

# Test 9: SSE endpoint returns correct content type
async def test_sse_endpoint_content_type():
    """GET /api/events returns text/event-stream."""

# Test 10: SSE keepalive sent on timeout
async def test_sse_keepalive():
    """SSE connection sends keepalive comment every 30 seconds."""

# Test 11: Queue cleanup on disconnect
async def test_queue_cleanup_on_disconnect():
    """Session queue is removed from _proactive_queues when client disconnects."""

# Test 12: Proactive message renders in chat
def test_proactive_message_render():
    """Frontend appendProactiveMessage creates DOM element with correct classes."""
```

## Estimated Impact

- **User engagement**: Proactive systems show +22% perceived engagement and +31% initiative ratings (Deng et al., CHI 2025)
- **Behavioral outcomes**: JITAI-timed interventions are 2-3x more effective than fixed-schedule (Nahum-Shani et al., 2018)
- **Capability coverage**: Moves 2 capabilities from False to True in the manifest (proactive_outreach_web, real_time_monitoring)
- **Differentiation**: Most behavioral AI chatbots are purely reactive; proactive initiation is a competitive differentiator
- **Risk mitigation**: Receptivity safeguards (caps, cooldowns, regime-awareness) prevent chatbot fatigue and psychological reactance

## Citations

1. Deng, Y. et al. (2025). "Proactive Conversational AI: A Comprehensive Survey." ACM Transactions on Information Systems. https://dl.acm.org/doi/10.1145/3715097
2. Deng, Y. et al. (2025). "Proactive Conversational Agents with Inner Thoughts." CHI 2025. https://dl.acm.org/doi/10.1145/3706598.3713760
3. Nahum-Shani, I. et al. (2018). "Just-in-Time Adaptive Interventions (JITAIs) in Mobile Health." Annals of Behavioral Medicine, 52(6). https://pmc.ncbi.nlm.nih.gov/articles/PMC5364076/
4. Bosschaerts, K. et al. (2025). "Designing a JITAI with Trigger Detection and a Generative Chatbot." Digital Health. https://journals.sagepub.com/doi/10.1177/20552076251381747
5. MDN Web Docs (2025). "Using Server-Sent Events." https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
6. Frontiers (2025). "Systematic Review: Human, AI, and Hybrid Health Coaching in Digital Health Interventions." https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1536416/full
