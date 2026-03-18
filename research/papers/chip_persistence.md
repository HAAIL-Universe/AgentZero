---
topic: Reasoning Chip Persistence Across Page Refreshes
status: implemented
priority: medium
estimated_complexity: small
researched_at: 2026-03-18T16:00:00Z
---

# Reasoning Chip Persistence Across Page Refreshes

## Problem Statement

Reasoning chips (agent thought bubbles) are lost on page refresh. Users who refresh mid-conversation lose all visual reasoning traces -- both live chips from the current turn and completed chips from prior turns. The only recovery path is navigating to the "trace" tab, which fetches historical reasoning runs from the backend but does not restore individual chips into the chat thread.

This breaks the "persistent context UI" pattern identified as a key 2025 AI interface design trend: important reasoning context should remain available throughout a conversation.

## Current State in Agent Zero

### State Management (`agent_zero-ui/src/App.tsx:121-125`)
```typescript
const [reasoningSteps, setReasoningSteps] = useState<ReasoningStep[]>([]);
const [chipsByTurn, setChipsByTurn] = useState<Record<number, ReasoningStep[]>>({});
```
Both are React state only -- volatile, lost on any re-render triggered by refresh.

### Chip Lifecycle
1. **Receive**: WebSocket `type: "reasoning_step"` -> appended to `reasoningSteps` (`App.tsx:784`)
2. **Display**: `ThoughtBubbles.tsx` (live) and `AgentChipGrid.tsx` (grid)
3. **Archive**: On `type: "done"`, chips move to `chipsByTurn[turnIndex]` (`App.tsx:835-843`)
4. **Display archived**: `AppShell.tsx:131` renders `turnChips` from `chipsByTurn[messageIndex]`
5. **Refresh**: Everything gone

### Existing Browser Storage
- `agent_zero_token`, `agent_zero_session_token`, `agent_zero_email` -- auth tokens
- `agent_zero_active_tab` -- UI tab state
- `agent_zero_onboarding_dismissed` -- onboarding flag
- **No chip/reasoning storage**

### Backend Storage
- `reasoning_runs` table: stores run metadata (route, trigger, context)
- `reasoning_stage_messages` table: stores inter-stage artifacts
- Individual chips are NOT persisted to backend -- they are fire-and-forget WebSocket events

## Industry Standard / Research Findings

### 1. Persistent Context UI Pattern

The Persistent Context UI pattern keeps important information available throughout a conversation. Smashing Magazine's 2025 AI interface design patterns guide identifies this as essential for AI applications where reasoning context informs user decisions.

Key principles:
- Keep persistent elements minimal -- only essential information
- Make contextual information collapsible on mobile
- Use clear visual hierarchy to distinguish context from conversation flow

**Citation**: Friedman, V. (2025). "Design Patterns For AI Interfaces." *Smashing Magazine*. https://www.smashingmagazine.com/2025/07/design-patterns-ai-interfaces/

### 2. sessionStorage Sync Pattern for React

The established pattern for React state persistence uses sessionStorage as a backing store synced via useEffect. The source of truth remains in React state, but sessionStorage provides recovery on refresh.

Best practice from Darren Lester (2024): "Keep the source of truth in React state and sync this to sessionStorage so if a new component mounts it can pull in the value from sessionStorage to initialise its own state."

Key implementation detail: sessionStorage/localStorage can only store strings -- objects must be serialized via `JSON.stringify()`/`JSON.parse()`.

**Citation**: Lester, D. (2024). "Syncing React State and Session Storage." https://www.darrenlester.com/blog/syncing-react-state-and-session-storage

### 3. Progressive Disclosure for Reasoning Traces

AI chat interfaces in 2025 emphasize progressive disclosure: reasoning details start collapsed and expand on user interaction. This applies to persisted chips -- they should restore as collapsed summaries, not as full expanded panels that would overwhelm the refreshed page.

**Citation**: "Innovative Chat UI Design Trends 2025." *MultitaskAI*. https://multitaskai.com/blog/chat-ui-design/

### 4. Agentic UX: Transparency Through Persistence

UX Magazine's 2025 analysis of agentic AI design patterns found that transparency features (reasoning traces, source attribution, confidence indicators) **increase retention** of AI features. Users who can see reasoning traces are more likely to continue using the system. Losing these traces on refresh directly undermines this effect.

**Citation**: "Secrets of Agentic UX: Emerging Design Patterns for Human Interaction with AI Agents." (2025). *UX Magazine*. https://uxmag.com/articles/secrets-of-agentic-ux-emerging-design-patterns-for-human-interaction-with-ai-agents

## Proposed Implementation

### Approach: Hybrid (sessionStorage + Backend)

**sessionStorage** for fast refresh recovery within the same browser tab.
**Backend** for cross-tab and cross-session chip access.

### Step 1: sessionStorage Sync Hook

**Create**: `agent_zero-ui/src/hooks/useSessionPersistedState.ts`

```typescript
import { useState, useEffect } from 'react';

export function useSessionPersistedState<T>(
    key: string,
    initialValue: T
): [T, React.Dispatch<React.SetStateAction<T>>] {
    const [state, setState] = useState<T>(() => {
        try {
            const stored = sessionStorage.getItem(key);
            return stored ? JSON.parse(stored) : initialValue;
        } catch {
            return initialValue;
        }
    });

    useEffect(() => {
        try {
            sessionStorage.setItem(key, JSON.stringify(state));
        } catch {
            // sessionStorage full or unavailable -- degrade gracefully
        }
    }, [key, state]);

    return [state, setState];
}
```

### Step 2: Replace useState with Persisted Hook

**Modify**: `agent_zero-ui/src/App.tsx`

```typescript
// BEFORE:
const [chipsByTurn, setChipsByTurn] = useState<Record<number, ReasoningStep[]>>({});

// AFTER:
const [chipsByTurn, setChipsByTurn] = useSessionPersistedState<Record<number, ReasoningStep[]>>(
    'agent_zero_chips_by_turn',
    {}
);
```

Live `reasoningSteps` do NOT need persistence -- they are transient by nature (streaming in during current turn). Only `chipsByTurn` (completed turns) needs to survive refresh.

### Step 3: Size Management

sessionStorage has a ~5MB limit. Each chip is ~500 bytes JSON. A conversation with 50 turns * 7 agents = 350 chips = ~175KB -- well within limits. But add a safety cap:

```typescript
// In the sync effect:
useEffect(() => {
    try {
        const serialized = JSON.stringify(state);
        if (serialized.length > 2_000_000) { // 2MB cap
            // Trim oldest turns to fit
            const entries = Object.entries(state as Record<number, unknown>);
            const trimmed = Object.fromEntries(entries.slice(-30)); // keep last 30 turns
            sessionStorage.setItem(key, JSON.stringify(trimmed));
        } else {
            sessionStorage.setItem(key, serialized);
        }
    } catch { /* degrade gracefully */ }
}, [key, state]);
```

### Step 4: Backend Chip Persistence (Optional Enhancement)

**Modify**: `agent_zero/agent_zero_server.py` in `_emit_reasoning_step()`

After broadcasting via WebSocket, also persist each chip to the existing `reasoning_stage_messages` table:

```python
async def _emit_reasoning_step(step, run_id=None):
    # Existing: emit via WebSocket
    await websocket.send_json({"type": "reasoning_step", "content": step})

    # NEW: Persist to DB if run_id available
    if run_id:
        await append_stage_message(
            run_id=run_id,
            stage=step.get("stage", "worker"),
            role="chip",
            content=step
        )
```

**Modify**: `/api/reasoning` endpoint to return chips:

```python
@app.get("/api/reasoning/{run_id}/chips")
async def get_reasoning_chips(run_id: str, user: dict = Depends(get_current_user)):
    """Return persisted chips for a reasoning run."""
    messages = await fetch_stage_messages(run_id)
    chips = [m["content"] for m in messages if m.get("role") == "chip"]
    return JSONResponse({"chips": chips})
```

### Step 5: Hydrate on Reconnect

**Modify**: `agent_zero-ui/src/App.tsx`

When WebSocket reconnects after a refresh, hydrate any missing turn chips from backend:

```typescript
// On WebSocket reconnect:
useEffect(() => {
    if (connected && messages.length > 0 && Object.keys(chipsByTurn).length === 0) {
        // Chips lost (fresh refresh) -- attempt backend hydration
        fetch('/api/reasoning/recent?limit=50', { headers: authHeaders })
            .then(res => res.json())
            .then(data => {
                if (data.chips_by_turn) {
                    setChipsByTurn(data.chips_by_turn);
                }
            })
            .catch(() => { /* degrade gracefully */ });
    }
}, [connected]);
```

## Test Specifications

### Frontend Tests: `agent_zero-ui/src/hooks/useSessionPersistedState.test.ts`

```typescript
test('initializes from sessionStorage if available')
test('falls back to initialValue if sessionStorage empty')
test('syncs state changes to sessionStorage')
test('handles JSON parse errors gracefully')
test('handles sessionStorage full gracefully')
test('trims oldest entries when size exceeds 2MB cap')
```

### Integration Tests

```typescript
test('chipsByTurn survives simulated page refresh')
test('live reasoningSteps are cleared on refresh (by design)')
test('archived chips from 30+ turns are trimmed to last 30')
test('backend chip persistence writes to reasoning_stage_messages')
test('backend chip hydration restores chips on reconnect')
```

### Expected Test Count: ~10 tests

## Estimated Impact

1. **Immediate UX improvement**: Refreshing the page no longer destroys reasoning context. Users can refresh without losing their conversation's reasoning traces.

2. **Trust preservation**: Research shows transparency features increase AI feature retention (UX Magazine 2025). Losing reasoning traces on refresh undermines this trust -- persistence fixes it.

3. **Minimal code change**: The core fix is replacing one `useState` with a `useSessionPersistedState` hook -- a ~30-line change plus a ~20-line hook. Backend persistence is an optional enhancement.

4. **Low risk**: sessionStorage is same-origin, same-tab scoped -- no cross-tab interference. Falls back gracefully if storage is unavailable.

## Files to Create/Modify

| Action | File | Changes |
|--------|------|---------|
| CREATE | `agent_zero-ui/src/hooks/useSessionPersistedState.ts` | Generic hook (~30 lines) |
| MODIFY | `agent_zero-ui/src/App.tsx:121-125` | Replace useState with persisted hook for chipsByTurn |
| MODIFY | `agent_zero/agent_zero_server.py:1831-1846` | Optional: persist chips to DB |
| MODIFY | `agent_zero/agent_zero_server.py` | Optional: add /api/reasoning/{id}/chips endpoint |
| CREATE | `agent_zero-ui/src/hooks/useSessionPersistedState.test.ts` | ~10 tests |
