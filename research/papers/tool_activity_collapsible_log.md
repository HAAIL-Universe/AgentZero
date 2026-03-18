---
topic: Tool Activity Collapsible Log in Chat Thread
status: implemented
priority: high
estimated_complexity: medium
researched_at: 2026-03-18T15:30:00Z
---

# Tool Activity Collapsible Log in Chat Thread

## Problem Statement

When Agent Zero executes tools during a conversation turn (memory recall, goal lookup, commitment
checks, workspace reads, analysis requests), the user has no persistent visibility into what
happened. Tool activity currently appears as ephemeral gray preprocess lines during generation,
then vanishes completely when the response arrives. The user cannot:

1. See which tools ran after the fact (post-response)
2. Inspect what arguments were passed to each tool
3. View tool results or errors
4. Understand how tool outputs influenced the response
5. Build trust through transparency of agent actions

This is a significant gap. Research consistently shows that progressive disclosure of agent
actions is the primary mechanism for building user trust in agentic AI systems.

## Current State in Agent Zero

### Backend (tool execution + events)

**Tool Registry** (`agent_zero/tool_registry.py`):
- 15 executable tools defined (memory.recall, sessions.recent, goals.list, commitments.list,
  commitments.update, self.shadow, self.insights, self.requests, self.capabilities,
  self.runtime_notes, self.trace_deltas, workspace.list_dir, workspace.read_file,
  analysis.request, analysis.results)
- Each tool spec includes name, description, input_schema, output_schema, availability

**Tool Runtime** (`agent_zero/tool_runtime.py`):
- `execute_tool()` (line 58): executes tools and returns standardized result dict:
  `{tool_name, success, summary, data, error?}`
- `select_tools_for_turn()` (line 119): deterministic tool selection based on route + phrases
- `build_tool_result_prompt_fragment()` (line 186): renders results into prompt context

**Agent Zero Server** (`agent_zero/agent_zero_server.py`):
- Lines 1571-1623: Pre-turn tool execution loop emits two WebSocket event types per tool:
  1. `runtime` event with `event: "tool_call"` (stage: tool_call, meta: {tool_name, args})
  2. `runtime` event with `event: "tool_result"` (stage: tool_result, meta: {tool_name, success})
- Lines 2155-2225: Model-initiated tool calls (vLLM native function calling) emit:
  1. `reasoning_step` with `agent_id: "tool_activity"`, `stage: "tool_call"`
  2. `reasoning_step` with `agent_id: "tool_activity"`, `stage: "tool_result"`
- `_execute_model_tool_calls()` (line 298): handles model-initiated tools, max 3 rounds
- `_emit_runtime_event()` (line 347): sends runtime events to WebSocket

**Key data available per tool call (already computed, just not persisted inline):**
- Tool name and arguments (from selection or model request)
- Execution result: success/failure, summary string, full data dict
- Timestamps (ISO 8601)
- Whether it was pre-turn (deterministic) or model-initiated (native function calling)

### Frontend (current display)

**App.tsx** (lines 771-773, 823-831):
- `reasoning_step` messages collected into `reasoningSteps` state (max 12, rolling window)
- On response completion, steps with `agent_id` in `PRE_PROCESS` set are filtered OUT
- `PRE_PROCESS` = `{'router', 'state_extractor', 'scorer', 'agent_zero_thinking', 'tool_activity'}`
- Tool steps are discarded, never persisted to `chipsByTurn`

**AppShell.tsx** (lines 172-190):
- Preprocess indicators rendered as gray lines at bottom of chat during generation
- Format: `[dot] TOOL [thought text]` in monospace, 0.72rem, rgba(255,255,255,0.45)
- Disappears entirely when response completes (lines 823-831 clear the array)

**styles.css** (lines 1421-1456):
- `.preprocess-indicators`: flex column, 4px gap
- `.preprocess-line`: gray, small, monospace
- No hover, click, or expand interactions

**Runtime tab** (AppShell.tsx lines 553-579):
- Full runtime events including tool details are visible ONLY in the Runtime tab
- Rendered as stack cards with up to 6 meta key-value pairs
- User must actively switch tabs to see this -- not inline with chat

**Result: Tool activity is fire-and-forget in the chat UI. No persistence, no inspection,
no history. The data exists in state but is only surfaced in a separate tab.**

## Industry Standard / Research Findings

### 1. Progressive Disclosure for Agent Transparency (Core Pattern)

The principle of progressive disclosure -- gradually revealing information as users need it --
is the established standard for displaying agent tool activity in AI interfaces.

**Ramos & Honra (2025)** recommend a three-layer architecture for agent information:
- **Layer 1 (Index):** Lightweight metadata visible by default (tool name, status icon)
- **Layer 2 (Details):** Full content loaded on demand (arguments, result summary)
- **Layer 3 (Deep Dive):** Supporting materials for power users (raw data, timing)

They explicitly recommend "keeping progressive disclosure to 2-3 layers to avoid user
frustration" and maintaining "explicit control points so users understand when additional
context is activated." They note that Claude Code's Skills implementation demonstrates
this pattern: "Scripts are executed but not read into context; their outputs consume tokens
instead. Only results return to the main thread."

URL: https://www.honra.io/articles/progressive-disclosure-for-ai-agents

### 2. User Trust Research

**ScienceDirect (2025)** found that "the relationship between information disclosure and
trust in digital agents is positive and significant." However, "overly detailed or poorly
timed information can distract users" -- validating the collapsed-by-default approach.

URL: https://www.sciencedirect.com/science/article/pii/S2444569X25001155

**ScienceDirect (2025)** in a clinical AI study found that ~66% of participants preferred
detailed disclosures about AI system actions, while most remaining participants wanted
"detail-on-demand" (collapsible) formats. Almost no participants wanted hidden actions.
Progressive disclosure "helps users manage information load and better understand AI
recommendations while fostering greater trust."

URL: https://www.sciencedirect.com/science/article/abs/pii/S107158192500148X

**Springer et al. (2020)** in ACM TIIS found that users want transparency information at
specific moments: during decision-making, after unexpected outcomes, and when building
initial trust. The timing matters as much as content.

URL: https://dl.acm.org/doi/10.1145/3374218

### 3. Vercel AI SDK Tool Component (Production Standard)

The Vercel AI SDK Elements library (2025-2026) provides a production-ready `<Tool>` component
that has become the de facto standard for displaying tool invocations in React AI chat apps.

**Design:**
- Compositional: `<Tool>` > `<ToolHeader>` + `<ToolContent>` > `<ToolInput>` + `<ToolOutput>`
- Four visual states: `input-streaming`, `input-available`, `output-available`, `output-error`
- Collapsed by default; auto-expands on completion or error
- Status badges with contextual icons (Pending, Running, Completed, Error)
- JSON syntax highlighting for parameters
- Accessible keyboard navigation and screen reader support

```tsx
<Tool defaultOpen={false}>
  <ToolHeader type="tool-memory_recall" state={tool.state} />
  <ToolContent>
    <ToolInput input={tool.input} />
    <ToolOutput output={result} errorText={tool.errorText} />
  </ToolContent>
</Tool>
```

URL: https://elements.ai-sdk.dev/components/tool

### 4. Shinychat Collapsible Tool Cards

Posit's shinychat framework (2025) implements a similar collapsible card pattern:
- Tool calls displayed as collapsed cards inline in the chat thread
- Expanding shows arguments and results
- Custom titles, icons per tool
- Optional `_intent` argument shows WHY the LLM invoked the tool
- `show_request = FALSE` to hide call details when output is self-explanatory
- HTML/Markdown/plain text rendering for results
- Error states captured and displayed in the card

URL: https://shiny.posit.co/blog/posts/shinychat-tool-ui/

### 5. Minimal Explanation Packet (Agentic AI Transparency)

An Engineering Archive survey (2025) on "Transparency in Agentic AI" proposes the
Minimal Explanation Packet (MEP) as a standardized artifact for agent audit trails. MEP
bundles: agent identity, tool invocations with inputs/outputs, decision rationale, and
outcome verification. The survey emphasizes that "plans, tool I/O, memory writes/retrievals,
and coordination signals" are the critical agent-specific artifacts that must be exposed.

URL: https://engrxiv.org/preprint/view/6451/version/8400

### 6. Google PAIR Guidebook

Google's People + AI Research guidebook recommends:
- "Focus on sharing the information users need to make decisions and move forward"
- "Don't attempt to explain everything that's happening in the system"
- Use progressive disclosure with "clear, concise, and contextually relevant" explanations

URL: https://pair.withgoogle.com/guidebook/

### 7. Agent Management Interface Patterns (Wroblewski 2025)

Luke Wroblewski identifies six patterns for agent activity display (kanban, dashboards,
inboxes, task lists, calendars). For inline chat contexts, the inbox/task-list pattern is
most applicable: chronological streams with unread indicators and completion states. He
emphasizes leveraging familiar UI patterns while extending them for agent-specific needs.

URL: https://lukew.com/ff/entry.asp?2106=

### 8. ChatGPT's Inline Activity Indicators

ChatGPT (2025-2026) shows tool activity as collapsible inline sections within the response:
- "Searching the web..." with spinner, then collapses to "Searched 3 sources" with expandable list
- "Reading document..." with progress, then collapses to show extracted summary
- Pattern: animated verb during execution, collapsed past-tense summary after completion

URL: https://intuitionlabs.ai/articles/conversational-ai-ui-comparison-2025

## Proposed Implementation

### Architecture: Three-Layer Progressive Disclosure

**Layer 1 -- Collapsed Summary Bar (always visible after response)**
A single-line collapsible bar positioned between user message and Agent Zero response.
Shows: tool count, names, overall status. Collapsed by default.

```
[green dot] 3 tools used  [v]
```

**Layer 2 -- Expanded Tool Cards (on click)**
Each tool gets a card showing name, status badge, summary, duration, and arg chips.

```
[v] 3 tools used

  [check] memory.recall                              320ms
  Retrieved 4 memory notes matching "exercise routine"
  [query: "exercise routine"] [limit: 4]

  [check] goals.list                                 180ms
  Found 3 active goals

  [x] commitments.list                               510ms
  Error: session expired
```

**Layer 3 -- Tool Detail (on "Show data" click within a card)**
Full result data as formatted JSON, scrollable, max 200px height.

### Step 1: Backend -- Emit tool_activity summary message

**File: `agent_zero/agent_zero_server.py`**

After the tool execution loop (around line 1615 for pre-turn tools, and after line 2225
for model-initiated tools), emit a new `tool_activity` WebSocket message bundling all
tool calls for the turn.

1. In the pre-turn tool execution loop, capture args and duration on each result:

```python
import time

# Inside the for-loop over filtered_tool_calls (line ~1571):
t0 = time.monotonic()
tool_result = await execute_tool(tool_name, tool_args, user=user, session_id=session_id)
tool_result["_args"] = tool_args
tool_result["_duration_ms"] = round((time.monotonic() - t0) * 1000)
```

2. After the loop completes, emit the summary:

```python
async def _emit_tool_activity_summary(websocket, tool_results, turn_index):
    if not tool_results:
        return
    tools = []
    for r in tool_results:
        tools.append({
            "tool_name": r["tool_name"],
            "success": r.get("success", False),
            "summary": r.get("summary", ""),
            "args": r.get("_args", {}),
            "duration_ms": r.get("_duration_ms", 0),
            "data_preview": _truncate_data(r.get("data", {}), 500),
        })
    msg = {
        "type": "tool_activity",
        "content": {
            "turn_index": turn_index,
            "tools": tools,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    }
    await websocket.send_text(json.dumps(msg))
```

3. Add `_truncate_data` helper:

```python
def _truncate_data(data, max_chars=500):
    """Serialize data dict, truncating to max_chars."""
    if not data:
        return ""
    try:
        s = json.dumps(data, default=str, ensure_ascii=False)
        if len(s) > max_chars:
            return s[:max_chars] + "..."
        return s
    except Exception:
        return str(data)[:max_chars]
```

4. Same pattern for model-initiated tool calls: collect results in `_execute_model_tool_calls`
   and emit summary after each round completes.

### Step 2: Frontend types

**File: `agent_zero-ui/src/types.ts`**

Add:

```typescript
export interface ToolActivityItem {
  tool_name: string
  success: boolean
  summary: string
  args: Record<string, unknown>
  duration_ms: number
  data_preview: string
}

export interface ToolActivity {
  turn_index: number
  tools: ToolActivityItem[]
  ts: string
}
```

### Step 3: Frontend state management

**File: `agent_zero-ui/src/App.tsx`**

1. Add state:
```typescript
const [toolActivityByTurn, setToolActivityByTurn] = useState<Record<number, ToolActivity>>({})
```

2. In WebSocket message handler, add:
```typescript
if (type === 'tool_activity') {
  const activity = payload.content as ToolActivity
  setToolActivityByTurn(prev => ({
    ...prev,
    [activity.turn_index]: activity
  }))
  return
}
```

3. Pass `toolActivityByTurn` to AppShell as prop.

### Step 4: ToolActivityLog component

**File: `agent_zero-ui/src/components/ToolActivityLog.tsx`** (NEW)

```typescript
import { useState } from 'react'
import type { ToolActivity, ToolActivityItem } from '../types'

interface Props {
  activity: ToolActivity
}

export function ToolActivityLog({ activity }: Props) {
  const [expanded, setExpanded] = useState(false)
  const { tools } = activity
  if (tools.length === 0) return null

  const allSuccess = tools.every(t => t.success)
  const totalMs = tools.reduce((sum, t) => sum + t.duration_ms, 0)

  return (
    <div className="tool-activity-log">
      {/* Layer 1: Summary bar */}
      <button
        className={`tool-activity-bar ${allSuccess ? '' : 'has-error'}`}
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className={`tool-status-dot ${allSuccess ? 'success' : 'partial'}`} />
        <span className="tool-activity-count">
          {tools.length} tool{tools.length !== 1 ? 's' : ''} used
        </span>
        {totalMs > 0 && (
          <span className="tool-activity-timing">{totalMs}ms</span>
        )}
        <span className={`tool-activity-chevron ${expanded ? 'open' : ''}`}>
          &#9662;
        </span>
      </button>

      {/* Layer 2: Tool cards */}
      {expanded && (
        <div className="tool-activity-cards">
          {tools.map((tool, i) => (
            <ToolCallCard key={`${tool.tool_name}-${i}`} tool={tool} />
          ))}
        </div>
      )}
    </div>
  )
}

function ToolCallCard({ tool }: { tool: ToolActivityItem }) {
  const [showData, setShowData] = useState(false)
  const argEntries = Object.entries(tool.args)

  return (
    <div className={`tool-card ${tool.success ? 'success' : 'error'}`}>
      <div className="tool-card-header">
        <span className={`tool-status-icon ${tool.success ? 'success' : 'error'}`}>
          {tool.success ? '\u2713' : '\u2717'}
        </span>
        <span className="tool-card-name">{tool.tool_name}</span>
        {tool.duration_ms > 0 && (
          <span className="tool-card-duration">{tool.duration_ms}ms</span>
        )}
      </div>
      <div className="tool-card-summary">{tool.summary}</div>

      {argEntries.length > 0 && (
        <div className="tool-card-args">
          {argEntries.map(([k, v]) => (
            <span key={k} className="tool-arg-chip">
              {k}: {typeof v === 'string' ? v : JSON.stringify(v)}
            </span>
          ))}
        </div>
      )}

      {/* Layer 3: Raw data toggle */}
      {tool.data_preview && tool.data_preview.length > 2 && (
        <>
          <button
            className="tool-data-toggle"
            onClick={() => setShowData(!showData)}
          >
            {showData ? 'Hide data' : 'Show data'}
          </button>
          {showData && (
            <pre className="tool-data-raw">{tool.data_preview}</pre>
          )}
        </>
      )}
    </div>
  )
}
```

### Step 5: Insert in AppShell

**File: `agent_zero-ui/src/components/AppShell.tsx`**

In the message rendering section, for each Agent Zero message, insert the ToolActivityLog
between the AgentChipGrid and the response text:

```tsx
{toolActivityByTurn[turnIndex] && (
  <ToolActivityLog activity={toolActivityByTurn[turnIndex]} />
)}
```

Position: after user message div, after agent chips, before assistant message content.

### Step 6: CSS styles

**File: `agent_zero-ui/src/styles.css`**

```css
/* ---- Tool Activity Log ---- */
.tool-activity-log {
  margin: 4px 0;
  font-family: var(--font-mono);
}

.tool-activity-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.72rem;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.tool-activity-bar:hover {
  background: rgba(255, 255, 255, 0.07);
  border-color: rgba(255, 255, 255, 0.15);
  color: rgba(255, 255, 255, 0.7);
}
.tool-activity-bar.has-error {
  border-color: rgba(255, 107, 107, 0.3);
}

.tool-status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.tool-status-dot.success { background: #4caf50; }
.tool-status-dot.partial { background: #ff9800; }

.tool-activity-count { font-weight: 600; white-space: nowrap; }
.tool-activity-timing {
  color: rgba(255, 255, 255, 0.35);
  font-size: 0.65rem;
}

.tool-activity-chevron {
  margin-left: auto;
  transition: transform 0.15s ease;
  font-size: 0.7rem;
}
.tool-activity-chevron.open { transform: rotate(180deg); }

.tool-activity-cards {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px 0 2px 0;
}

.tool-card {
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 6px;
  border-left: 3px solid #4caf50;
}
.tool-card.error { border-left-color: #f44336; }

.tool-card-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.72rem;
}
.tool-status-icon.success { color: #4caf50; }
.tool-status-icon.error { color: #f44336; }
.tool-card-name {
  font-weight: 600;
  color: rgba(255, 255, 255, 0.7);
}
.tool-card-duration {
  margin-left: auto;
  color: rgba(255, 255, 255, 0.35);
  font-size: 0.65rem;
}
.tool-card-summary {
  font-size: 0.72rem;
  color: rgba(255, 255, 255, 0.45);
  margin-top: 2px;
  font-family: var(--font-serif);
}

.tool-card-args {
  display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;
}
.tool-arg-chip {
  font-size: 0.68rem;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.05);
  color: rgba(255, 255, 255, 0.4);
}

.tool-data-toggle {
  background: none; border: none;
  color: rgba(108, 99, 255, 0.8);
  font-size: 0.68rem;
  cursor: pointer;
  padding: 2px 0; margin-top: 4px;
}
.tool-data-toggle:hover { text-decoration: underline; }

.tool-data-raw {
  font-size: 0.65rem;
  background: rgba(0, 0, 0, 0.25);
  padding: 6px 8px;
  border-radius: 4px;
  overflow-x: auto;
  max-height: 200px;
  margin-top: 4px;
  color: rgba(255, 255, 255, 0.5);
  white-space: pre-wrap;
  word-break: break-word;
}
```

### Interaction with Existing Systems

- **Preprocess indicators**: Keep as-is during generation (live "Calling..." feedback).
  The `ToolActivityLog` replaces them as the persistent record after response completes.
- **Agent chip grid**: No changes. Tool activity sits between chips and response text.
- **Runtime tab**: No changes. Runtime events continue flowing for the Runtime tab.
- **reasoning_step filtering**: `tool_activity` remains in `PRE_PROCESS` set.
  The new `tool_activity` message type is a separate WebSocket channel.
- **Backward compatibility**: No existing behavior changes. This is purely additive.

### Migration Path

1. Backend: Add `_truncate_data()` and `_emit_tool_activity_summary()` helpers
2. Backend: Capture args + duration on result dicts in tool execution loops
3. Backend: Call `_emit_tool_activity_summary()` after pre-turn and model-initiated loops
4. Frontend: Add `ToolActivityItem` and `ToolActivity` interfaces to types.ts
5. Frontend: Add `toolActivityByTurn` state and WebSocket handler in App.tsx
6. Frontend: Create `ToolActivityLog.tsx` component
7. Frontend: Insert component in AppShell message rendering
8. Frontend: Add CSS styles
9. Test end-to-end with real tool calls

## Test Specifications

### Backend Tests (`agent_zero/test_tool_activity_log.py`)

```python
def test_tool_activity_summary_message_format():
    """_emit_tool_activity_summary produces correct JSON structure."""
    # Verify: type == "tool_activity"
    # content has: turn_index (int), tools (list), ts (ISO string)
    # Each tool: tool_name, success, summary, args, duration_ms, data_preview

def test_tool_activity_includes_all_executed_tools():
    """Summary includes every tool that ran in the turn, in execution order."""
    # Given: 3 tools selected and executed (memory.recall, goals.list, commitments.list)
    # Then: summary.tools has length 3 with correct names in order

def test_tool_activity_captures_args():
    """Each tool entry includes the arguments passed to execute_tool."""
    # Given: memory.recall called with {query: "exercise", limit: 4}
    # Then: tool entry args == {query: "exercise", limit: 4}

def test_tool_activity_captures_duration():
    """Each tool entry includes execution duration in milliseconds."""
    # Given: tool takes ~200ms (mocked)
    # Then: duration_ms is between 150 and 300

def test_tool_activity_captures_errors():
    """Failed tools have success=False and error info in summary."""
    # Given: a tool that raises an exception
    # Then: entry has success=False, summary contains error text

def test_tool_activity_data_preview_truncated():
    """data_preview is truncated to 500 chars max."""
    # Given: tool returns a data dict > 500 chars when serialized
    # Then: data_preview ends with "..."

def test_tool_activity_not_emitted_when_no_tools():
    """No tool_activity message sent when zero tools execute."""
    # Given: a turn with no tool calls
    # Verify: no tool_activity message sent via WebSocket

def test_model_initiated_tools_included():
    """Tools called via vLLM native function calling appear in summary."""
    # Given: model requests memory.recall via tool_calls
    # Then: tool_activity includes it with correct args and result

def test_truncate_data_empty():
    """_truncate_data handles empty dict."""
    # Given: data = {}
    # Then: returns ""

def test_truncate_data_small():
    """_truncate_data returns full JSON for small data."""
    # Given: data = {"key": "value"}
    # Then: returns '{"key": "value"}'

def test_truncate_data_large():
    """_truncate_data truncates large data."""
    # Given: data with values > 500 chars total
    # Then: result length <= 503 (500 + "...")
```

### Frontend Tests (vitest + react-testing-library)

```
test: ToolActivityLog renders nothing when tools array is empty
test: ToolActivityLog renders summary bar with correct tool count (singular/plural)
test: ToolActivityLog shows green dot when all tools succeed
test: ToolActivityLog shows orange dot when any tool fails
test: ToolActivityLog shows total duration in summary bar
test: clicking summary bar toggles expanded state
test: collapsed state does not render tool cards
test: expanded state renders one ToolCallCard per tool
test: ToolCallCard shows check icon for success, x for error
test: ToolCallCard shows tool name and duration
test: ToolCallCard shows summary text
test: ToolCallCard renders arg chips when args present
test: ToolCallCard hides arg chips when args empty
test: "Show data" button appears when data_preview is non-empty
test: clicking "Show data" reveals formatted JSON
test: clicking "Hide data" hides the JSON
test: error card has red left border
test: tool_activity WebSocket message updates toolActivityByTurn state
test: ToolActivityLog appears in correct position between chips and response
```

## Estimated Impact

**User trust**: Progressive disclosure of tool activity is the #1 cited mechanism for
building trust in agentic AI (ScienceDirect 2025, PAIR guidebook). Users who can inspect
what tools ran and what they returned will have significantly higher confidence in responses.
~66% of users prefer seeing detailed disclosures about AI system actions.

**Debugging**: When Agent Zero gives an unexpected response, the user can expand the tool log
to see if wrong memories were recalled, goals were stale, or a tool errored silently.
This replaces the current workflow of switching to the Runtime tab.

**Transparency compliance**: The Minimal Explanation Packet framework (EngrXiv 2025)
identifies tool I/O as a mandatory transparency artifact for agentic AI. This satisfies
that requirement inline in the chat thread.

**Cognitive load**: The three-layer design ensures casual users see only a subtle 1-line
summary bar (minimal distraction) while power users can drill into full detail. Matches
PAIR guidebook: "share what users need to move forward" without explaining everything.

**Performance**: Negligible. One additional WebSocket message per tool-using turn (~200 bytes).
Frontend renders a simple collapsible div. No additional API calls or state complexity.

## References

1. Honra (2025). "Why AI Agents Need Progressive Disclosure, Not More Data."
   https://www.honra.io/articles/progressive-disclosure-for-ai-agents

2. ScienceDirect (2025). "The key role of design and transparency in enhancing trust
   in AI-powered digital agents."
   https://www.sciencedirect.com/science/article/pii/S2444569X25001155

3. ScienceDirect (2025). "Operationalizing selective transparency using progressive
   disclosure in artificial intelligence clinical diagnosis systems."
   https://www.sciencedirect.com/science/article/abs/pii/S107158192500148X

4. Springer et al. (2020). "Progressive Disclosure: When, Why, and How Do Users Want
   Algorithmic Transparency Information?" ACM TIIS 10(4).
   https://dl.acm.org/doi/10.1145/3374218

5. Vercel AI SDK Elements (2025-2026). Tool Component.
   https://elements.ai-sdk.dev/components/tool

6. Posit shinychat (2025). "Tool Calling UI in shinychat."
   https://shiny.posit.co/blog/posts/shinychat-tool-ui/

7. EngrXiv (2025). "Transparency in Agentic AI: A Survey of Interpretability,
   Explainability, and Governance."
   https://engrxiv.org/preprint/view/6451/version/8400

8. Google PAIR (2025). People + AI Guidebook.
   https://pair.withgoogle.com/guidebook/

9. Wroblewski, L. (2025). "Agent Management Interface Patterns."
   https://lukew.com/ff/entry.asp?2106=

10. IntuitionLabs (2025). "Comparing Conversational AI Tool User Interfaces 2025."
    https://intuitionlabs.ai/articles/conversational-ai-ui-comparison-2025
