---
topic: External Outcome Resolution API
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T16:00:00Z
---

# External Outcome Resolution API

## Problem Statement

Agent Zero episodes can only be resolved through chat input -- the user must explicitly say "I did it" or similar in conversation for outcomes to be recorded. This creates a fundamental gap:

1. **Many outcomes happen outside the chat**: A user commits to "go for a run" but uses Garmin/Strava to track it. The episode never resolves unless they return to chat and report.
2. **Quantified self data is siloed**: Wearables, habit trackers (Habitify, Streaks, Apple Health), calendars, and productivity tools all capture outcome data that Agent Zero cannot access.
3. **Resolution delay**: If the user doesn't chat for days, pending episodes accumulate unresolved. External resolution could happen in real-time.
4. **No integration surface**: There is no API endpoint for external systems to report outcomes. The only input path is the WebSocket chat handler.

## Current State in Agent Zero

### Outcome Resolution Flow -- episode_store.py:224-309

```python
def resolve_episode_outcomes(shadow: dict, user_content: str, intervention_resolutions: list) -> dict:
```

- Called at start of each turn in `agent_zero_server.py:1871`
- Takes `user_content: str` -- the user's chat message
- Uses `classify_outcome(user_content)` to detect acted/ignored/pushed_back
- Also takes `intervention_resolutions: list` from `intervention_tracker.py`
- No external input mechanism

### API Surface -- agent_zero_server.py

Existing endpoints (lines 505-1161):
- Auth: `/auth/register`, `/auth/login`, `/auth/me`
- Sessions: `/api/sessions/*`
- Commitments: `/user/commitments/*` (CRUD + check-in)
- Goals: `/user/goals/*`
- Shadow: `/api/shadow` (read-only)
- No endpoint for external outcome reporting

### Commitment Check-In -- agent_zero_server.py:978

```python
@app.post("/user/commitments/{commitment_id}/check-in")
```

This endpoint records check-in events for commitments but does NOT resolve episodes. Commitments and episodes are separate systems with only one-way sync (episode -> intervention, not external -> episode).

## Industry Standard / Research Findings

### 1. Terra API: Unified Wearable Webhook Integration (2025)

Terra provides a single API that connects 200+ wearable devices (Garmin, Fitbit, Apple Health, Oura, WHOOP) with webhooks pushing updated data into applications. Key design: external systems POST event payloads to a registered webhook URL, with payload normalization across providers. This is the standard pattern for health/fitness data integration.

**URL**: https://tryterra.co/

### 2. Open Wearables / Apple Health MCP Server (2026)

The Apple Health MCP Server evolved into Open Wearables, a self-hosted platform unifying wearable health data from Apple Health, Garmin, Polar, Suunto, and WHOOP. The architecture uses webhooks and event buses to sync state -- "workout completed" events update CRM stages and trigger follow-ups.

**URL**: https://www.themomentum.ai/blog/open-wearables-0-3-android-google-health-connect-samsung-health-railway

### 3. SMART on FHIR Behavioral Health Integration (2025)

SMART on FHIR applications for behavioral health support Provider APIs, Patient Access APIs, and FHIR Bulk data access via webhooks. The standard enables third-party applications to report outcomes, session data, and goal progress through standardized API contracts. Pal.Tech implemented this for behavioral health provider networks.

**URL**: https://www.pal.tech/case_study/developed-smart-on-fhir-application-for-behavioral-health-provider-network/

### 4. Hookdeck: Resilient Webhook Infrastructure (2025)

Hookdeck provides production-grade webhook reliability: event buffering, retry logic, idempotency, and failure replay. Key insight for Agent Zero: external outcome webhooks must be idempotent (receiving the same event twice should not double-count) and resilient (failed processing should be retried).

**URL**: https://hookdeck.com

### 5. Garmin Activity API Webhook Pattern (2025)

Garmin's Health API pushes activity completion data via webhooks with OAuth authentication. The payload includes activity type, duration, completion timestamp, and metrics. This is the canonical pattern for external outcome resolution.

**URL**: https://developer.garmin.com/gc-developer-program/activity-api/

### 6. Habitify Webhook Integration (2025)

Habitify connects habit completion events to external systems via webhooks, enabling real-time habit tracking data flow. The pattern: habit completed -> webhook fires -> external system processes outcome.

**URL**: https://integrately.com/integrations/habitify/webhook-api

## Proposed Implementation

### Design: REST API Endpoint + Webhook Receiver

Add two new endpoints to `agent_zero_server.py`:
1. **`POST /api/outcomes/resolve`** -- authenticated endpoint for direct outcome reporting
2. **`POST /api/webhooks/outcome`** -- webhook receiver for external systems (API key auth)

### Step 1: Define Outcome Resolution Models

Add to `agent_zero_server.py` (near other Pydantic models):

```python
class ExternalOutcomeRequest(BaseModel):
    """External outcome resolution request."""
    episode_id: str | None = None           # Direct episode ID if known
    commitment_title: str | None = None     # Match by commitment title
    topic_signals: list[str] | None = None  # Match by topic overlap
    outcome: str                            # "acted" | "ignored" | "pushed_back"
    confidence: float = 0.85                # How certain the external source is
    source: str = "external"                # e.g., "garmin", "habitify", "calendar"
    timestamp: str | None = None            # ISO timestamp of the outcome event
    metadata: dict | None = None            # Source-specific data (steps, duration, etc.)


class WebhookOutcomePayload(BaseModel):
    """Webhook payload from external integrations."""
    event_type: str                         # e.g., "habit.completed", "activity.finished"
    source: str                             # Integration name
    user_id: str | None = None              # External user ID (mapped via settings)
    data: dict                              # Source-specific payload
    timestamp: str | None = None
    idempotency_key: str | None = None      # Prevent duplicate processing
```

### Step 2: Add Episode Matching Logic

Add to `episode_store.py`:

```python
def resolve_episode_from_external(
    shadow: dict,
    *,
    episode_id: str | None = None,
    commitment_title: str | None = None,
    topic_signals: list[str] | None = None,
    outcome: str,
    confidence: float = 0.85,
    source: str = "external",
    timestamp: str | None = None,
) -> dict | None:
    """Resolve a pending episode outcome from an external source.

    Matching priority:
    1. Direct episode_id match
    2. Commitment title match (fuzzy, against episode intervention titles)
    3. Topic signal overlap (highest overlap among pending episodes)

    Returns the resolved episode dict, or None if no match found.
    """
    episodes = shadow.get("episodes", [])
    pending = [
        ep for ep in episodes
        if ep.get("outcome", {}).get("user_followed_up") is None
    ]
    if not pending:
        return None

    target = None

    # Priority 1: Direct episode ID
    if episode_id:
        for ep in pending:
            if ep.get("episode_id") == episode_id:
                target = ep
                break

    # Priority 2: Commitment title match
    if not target and commitment_title:
        title_lower = commitment_title.lower()
        for ep in pending:
            ep_title = (ep.get("intervention", {}) or {}).get("title", "").lower()
            if ep_title and (title_lower in ep_title or ep_title in title_lower):
                target = ep
                break

    # Priority 3: Topic signal overlap
    if not target and topic_signals:
        topic_set = set(topic_signals)
        best_overlap = 0
        for ep in pending:
            ep_topics = set(ep.get("topic_signals", []))
            overlap = len(ep_topics & topic_set)
            if overlap > best_overlap:
                best_overlap = overlap
                target = ep

    if not target:
        return None

    # Resolve the episode
    ep_outcome = target.setdefault("outcome", {})
    ep_outcome["user_followed_up"] = outcome == "acted" and confidence >= 0.5
    ep_outcome["intervention_outcome"] = outcome
    ep_outcome["outcome_confidence"] = confidence
    ep_outcome["resolution_source"] = source
    ep_outcome["resolved_at"] = timestamp or datetime.now(timezone.utc).isoformat()

    return target
```

### Step 3: Add API Endpoints

Add to `agent_zero_server.py`:

```python
# Idempotency tracking (in-memory, per-session)
_processed_webhook_keys: set[str] = set()
_WEBHOOK_KEY_CAP = 10000

@app.post("/api/outcomes/resolve")
async def resolve_outcome_external(
    req: ExternalOutcomeRequest,
    user=Depends(get_current_user),
):
    """Resolve an episode outcome from an external source (authenticated)."""
    if req.outcome not in ("acted", "ignored", "pushed_back"):
        raise HTTPException(400, "outcome must be acted, ignored, or pushed_back")
    if not (0.0 <= req.confidence <= 1.0):
        raise HTTPException(400, "confidence must be between 0 and 1")

    shadow = await get_shadow(user["user_id"])
    if not shadow:
        raise HTTPException(404, "No shadow profile found")

    resolved = resolve_episode_from_external(
        shadow,
        episode_id=req.episode_id,
        commitment_title=req.commitment_title,
        topic_signals=req.topic_signals,
        outcome=req.outcome,
        confidence=req.confidence,
        source=req.source,
        timestamp=req.timestamp,
    )

    if not resolved:
        return JSONResponse({"status": "no_match", "message": "No pending episode matched"}, status_code=404)

    await save_shadow(user["user_id"], shadow)
    return {
        "status": "resolved",
        "episode_id": resolved.get("episode_id"),
        "outcome": req.outcome,
        "confidence": req.confidence,
        "source": req.source,
    }


@app.post("/api/webhooks/outcome")
async def webhook_outcome(
    payload: WebhookOutcomePayload,
    api_key: str = Query(..., alias="key"),
):
    """Webhook receiver for external outcome integrations.

    Authenticated via API key (stored in user settings).
    Idempotent: duplicate idempotency_keys are silently accepted.
    """
    # Validate API key (look up user by webhook key)
    user_id = await _lookup_user_by_webhook_key(api_key)
    if not user_id:
        raise HTTPException(401, "Invalid webhook API key")

    # Idempotency check
    if payload.idempotency_key:
        if payload.idempotency_key in _processed_webhook_keys:
            return {"status": "duplicate", "message": "Already processed"}
        _processed_webhook_keys.add(payload.idempotency_key)
        if len(_processed_webhook_keys) > _WEBHOOK_KEY_CAP:
            # Evict oldest (set doesn't preserve order, but cap prevents unbounded growth)
            _processed_webhook_keys.clear()

    # Map external event to outcome
    outcome_map = _map_webhook_to_outcome(payload)
    if not outcome_map:
        return {"status": "unmapped", "message": f"Unknown event_type: {payload.event_type}"}

    shadow = await get_shadow(user_id)
    if not shadow:
        return JSONResponse({"status": "no_shadow"}, status_code=404)

    resolved = resolve_episode_from_external(
        shadow,
        topic_signals=outcome_map.get("topic_signals"),
        commitment_title=outcome_map.get("commitment_title"),
        outcome=outcome_map["outcome"],
        confidence=outcome_map.get("confidence", 0.80),
        source=payload.source,
        timestamp=payload.timestamp,
    )

    if resolved:
        await save_shadow(user_id, shadow)

    return {
        "status": "resolved" if resolved else "no_match",
        "episode_id": resolved.get("episode_id") if resolved else None,
    }


def _map_webhook_to_outcome(payload: WebhookOutcomePayload) -> dict | None:
    """Map external webhook events to Agent Zero outcome format."""
    event = payload.event_type.lower()
    data = payload.data or {}

    # Common patterns across habit trackers and fitness apps
    if event in ("habit.completed", "activity.completed", "workout.finished",
                 "task.done", "goal.achieved"):
        return {
            "outcome": "acted",
            "confidence": 0.90,
            "topic_signals": _infer_topics_from_webhook(data),
            "commitment_title": data.get("title") or data.get("name"),
        }
    if event in ("habit.skipped", "activity.skipped", "task.skipped"):
        return {
            "outcome": "ignored",
            "confidence": 0.75,
            "topic_signals": _infer_topics_from_webhook(data),
            "commitment_title": data.get("title") or data.get("name"),
        }
    if event in ("habit.deleted", "goal.abandoned"):
        return {
            "outcome": "pushed_back",
            "confidence": 0.80,
            "topic_signals": _infer_topics_from_webhook(data),
            "commitment_title": data.get("title") or data.get("name"),
        }
    return None


def _infer_topics_from_webhook(data: dict) -> list[str]:
    """Infer topic signals from webhook payload data."""
    text_parts = []
    for key in ("title", "name", "category", "description", "activity_type"):
        if key in data and data[key]:
            text_parts.append(str(data[key]))
    if not text_parts:
        return []
    from outcome_patterns import extract_topics
    return extract_topics(" ".join(text_parts))


async def _lookup_user_by_webhook_key(api_key: str) -> str | None:
    """Look up user_id from webhook API key.

    Webhook keys are stored in user_settings table (key: 'webhook_api_key').
    """
    pool = get_pool()
    if not pool:
        return None
    row = await fetch_one(
        "SELECT user_id FROM user_settings WHERE key = 'webhook_api_key' AND value = $1",
        api_key,
    )
    return row["user_id"] if row else None
```

### Step 4: Add Webhook Key Management

Add endpoint for users to generate/view their webhook API key:

```python
@app.post("/api/settings/webhook-key")
async def generate_webhook_key(user=Depends(get_current_user)):
    """Generate or regenerate a webhook API key for external integrations."""
    import secrets
    key = secrets.token_urlsafe(32)
    await execute(
        """INSERT INTO user_settings (user_id, key, value)
           VALUES ($1, 'webhook_api_key', $2)
           ON CONFLICT (user_id, key) DO UPDATE SET value = $2""",
        user["user_id"], key,
    )
    return {"webhook_api_key": key}


@app.get("/api/settings/webhook-key")
async def get_webhook_key(user=Depends(get_current_user)):
    """Get the current webhook API key."""
    val = await fetch_val(
        "SELECT value FROM user_settings WHERE user_id = $1 AND key = 'webhook_api_key'",
        user["user_id"],
    )
    return {"webhook_api_key": val}
```

## Test Specifications

### test_external_outcome_api.py

```python
# Test 1: Direct episode_id resolution
def test_resolve_by_episode_id():
    """External resolution by episode_id should set outcome correctly."""
    shadow = _shadow_with_pending_episode("ep-123", ["career"])
    result = resolve_episode_from_external(
        shadow, episode_id="ep-123", outcome="acted", confidence=0.90, source="garmin",
    )
    assert result is not None
    assert result["outcome"]["user_followed_up"] is True
    assert result["outcome"]["resolution_source"] == "garmin"
    assert result["outcome"]["outcome_confidence"] == 0.90

# Test 2: Commitment title matching
def test_resolve_by_commitment_title():
    """Match pending episode by commitment title substring."""
    shadow = _shadow_with_intervention("ep-456", "Go for a morning run", ["fitness"])
    result = resolve_episode_from_external(
        shadow, commitment_title="morning run", outcome="acted", confidence=0.85,
    )
    assert result is not None
    assert result["episode_id"] == "ep-456"

# Test 3: Topic signal matching
def test_resolve_by_topic_signals():
    """Match pending episode by topic overlap when no ID or title."""
    shadow = _shadow_with_pending_episode("ep-789", ["career", "learning"])
    result = resolve_episode_from_external(
        shadow, topic_signals=["career"], outcome="acted", confidence=0.80,
    )
    assert result is not None

# Test 4: No match returns None
def test_no_match_returns_none():
    """When no pending episode matches, return None."""
    shadow = _shadow_with_pending_episode("ep-100", ["fitness"])
    result = resolve_episode_from_external(
        shadow, topic_signals=["career"], outcome="acted",
    )
    assert result is None

# Test 5: Already-resolved episodes are skipped
def test_skip_resolved_episodes():
    """Episodes with existing outcomes should not be re-resolved."""
    shadow = {"episodes": [{
        "episode_id": "ep-200", "topic_signals": ["career"],
        "outcome": {"user_followed_up": True},
    }]}
    result = resolve_episode_from_external(
        shadow, episode_id="ep-200", outcome="ignored",
    )
    assert result is None

# Test 6: Low confidence "acted" does not set user_followed_up
def test_low_confidence_no_follow_up():
    """outcome='acted' with confidence < 0.5 should not set user_followed_up=True."""
    shadow = _shadow_with_pending_episode("ep-300", ["fitness"])
    result = resolve_episode_from_external(
        shadow, episode_id="ep-300", outcome="acted", confidence=0.3,
    )
    assert result is not None
    assert result["outcome"]["user_followed_up"] is False

# Test 7: Webhook event mapping
def test_webhook_event_mapping():
    """Standard webhook events should map to outcomes."""
    from agent_zero_server import _map_webhook_to_outcome, WebhookOutcomePayload
    payload = WebhookOutcomePayload(
        event_type="habit.completed", source="habitify",
        data={"title": "Morning Run", "category": "fitness"},
    )
    result = _map_webhook_to_outcome(payload)
    assert result is not None
    assert result["outcome"] == "acted"
    assert result["confidence"] >= 0.85

# Test 8: Unknown webhook event returns None
def test_unknown_webhook_event():
    """Unrecognized event types should return None (unmapped)."""
    from agent_zero_server import _map_webhook_to_outcome, WebhookOutcomePayload
    payload = WebhookOutcomePayload(
        event_type="foo.bar", source="unknown", data={},
    )
    assert _map_webhook_to_outcome(payload) is None

# Test 9: Idempotency key prevents double processing
def test_idempotency():
    """Same idempotency_key should be accepted once, duplicate silently ignored."""
    # This would be an integration test against the running server

# Test 10: Topic inference from webhook data
def test_topic_inference_from_webhook():
    """Webhook data with fitness keywords should infer fitness topic."""
    from agent_zero_server import _infer_topics_from_webhook
    topics = _infer_topics_from_webhook({"title": "Morning Run", "activity_type": "running"})
    assert "health" in topics or "fitness" in topics  # depends on TOPIC_KEYWORDS
```

## Estimated Impact

1. **Closes the outcome gap**: Users who track habits externally (Garmin, Habitify, Apple Health, Strava) can now have outcomes automatically resolved. This is the single biggest source of unresolved episodes -- most behavioral outcomes happen outside the chat.

2. **Real-time resolution**: Instead of waiting for the user to return to chat and say "I did it," external systems push outcomes as they happen. This means consolidation operates on fresher, more complete data.

3. **Enables quantified self integration**: The webhook endpoint is the foundation for connecting Agent Zero to the broader quantified self ecosystem. Calendar events (Google Calendar webhook -> "meeting attended"), fitness trackers (Garmin -> "ran 5k"), and habit apps (Habitify -> "meditated") can all feed outcomes.

4. **Backward compatible**: All existing resolution logic is untouched. The new `resolve_episode_from_external` function supplements, not replaces, the chat-based `resolve_episode_outcomes`. Episodes can be resolved by either path.

5. **Security**: The authenticated endpoint uses existing JWT auth. The webhook endpoint uses per-user API keys stored in the database. Idempotency keys prevent double-processing.
