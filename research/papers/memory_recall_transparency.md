---
topic: Memory Recall Transparency
status: ready_for_implementation
priority: medium
estimated_complexity: medium
researched_at: 2026-03-18T16:30:00Z
---

# Memory Recall Transparency

## Problem Statement

When Agent Zero recalls memories to inform a response, the user has no visibility into what
was retrieved, why it was selected, or how it influenced the response. Memory recall happens
silently -- notes are fetched, scored, injected into the prompt, and consumed by the LLM
without any indication to the user.

This creates three problems:

1. **Trust gap**: The user doesn't know if Agent Zero is drawing on relevant context or stale
   data. When a response feels off, there's no way to diagnose whether bad memory retrieval
   was the cause.

2. **Correction gap**: If the wrong memories are recalled (e.g., outdated reasoning summary,
   wrong topic), the user can't see this to correct it. They would need to navigate to the
   trace debugging panel and parse raw JSON.

3. **Understanding gap**: Memory is Agent Zero's primary mechanism for personalization and
   continuity. Making it invisible makes the personalization feel like magic rather than a
   transparent, controllable feature.

Research consistently shows that making retrieval visible builds trust, improves critical
thinking, and enables users to correct errors. The citation-forward design pioneered by
Perplexity AI has become the industry standard.

## Current State in Agent Zero

### Two-Tier Memory Retrieval

**Tier 1: Automatic retrieval (every turn)**
- `retrieval_policy.py:65` -- `build_retrieval_packet()` fetches up to 12 recent notes
- `memory_policy.py:18` -- `select_memory_notes()` filters by route and scoring:
  - `strategic_reasoning`: up to 4 notes, min score 2
  - `continuity_prompt`: up to 2 notes, min score 2
  - `simple_chat`: up to 2 notes, min score 4 (strictest)
- Scoring (`memory_policy.py:84-125`): additive points for query overlap (+2/word, max 6),
  route context (+3 for reasoning_summary in strategic), priority overlap (+2), goal
  overlap (+1), route history (+1), continuity cues (+1)

**Tier 2: Explicit tool call**
- `tool_runtime.py:208` -- `_memory_recall()` called when `memory.recall` tool is selected
- Fetches up to 24 notes, scores with word overlap, returns top N (limit 1-8)
- Each note gets `retrieval_score` (integer) and is ordered by score descending

### Memory Note Structure (from DB)

```python
{
    "id": str,                    # UUID
    "scope": str,                 # "scratchpad" | "self_runtime"
    "note_type": str,             # "reasoning_summary" | "note"
    "content": str,               # 1000+ char summaries
    "metadata": dict,             # e.g., {"route": "strategic_reasoning"}
    "created_at": str,            # ISO timestamp
    "updated_at": str,            # ISO timestamp
    "retrieval_score": int,       # added during selection
    "retrieval_reasons": list,    # e.g., ["query overlap", "reasoning summary"]
}
```

### Prompt Injection

- `context_bundle.py:85-112` -- `build_context_prompt_fragment()` renders notes as bullets:
  `"Retrieved memory notes:\n- [scope/note_type] content"` (max 4-5 notes)
- `retrieval_policy.py:183-228` -- `build_retrieval_prompt_fragment()` similar formatting

### Current UI Display

- **Main chat**: NO memory information shown
- **Trace panel** (`AppShell.tsx:606-611`): "Promoted Memory Notes" as expandable JSON
  in the debugging/reasoning inspection view. Requires navigating to trace tab.
- **Runtime tab**: Tool call events show `memory.recall` execution but not the content

**Result: Memory retrieval is completely invisible in the main conversation flow.**

## Industry Standard / Research Findings

### 1. Citation-Forward Design (Perplexity Pattern)

Perplexity AI popularized the inline citation pattern for AI-generated responses, which
has become the de facto standard for transparent retrieval:

- Numbered inline citations `[1]` appear at claim-level within the response text
- Each citation links to a source panel showing title, URL, and relevant excerpt
- Users can hover/click to see what was retrieved
- The Sources panel acts as persistent verification context

ShapeOfAI (2025) catalogs four citation patterns: inline highlights, direct quotations,
multi-source references, and lightweight links. Best practice: "Point to exact passages
rather than broad documents" and "use metadata to help users judge relevance quickly."

URL: https://www.shapeof.ai/patterns/citations

### 2. Source Transparency Design Effects

Zhao et al. (2025, arXiv:2601.14611) tested four source transparency designs in
conversational AI: collapsible, hover card, footer, and aligned sidebar. Key findings:

- **Sidebar with persistent sources improved critical thinking** when citation density
  was high (synthesis: beta=0.07, p<0.01; source-related thinking: beta=0.07, p<0.01)
- **Hover cards supported better flow** for in-context verification without interruption
- Trust perception was surprisingly stable across conditions -- but the sidebar reversed
  negative trends from high information density
- Recommendation: **adaptive design** that shifts between patterns based on task phase

URL: https://arxiv.org/html/2601.14611

### 3. RAG Attribution Standards (TREC 2025)

The TREC 2025 RAG Track (150+ submissions) evaluated systems on four axes: relevance,
response completeness, **attribution verification**, and agreement analysis. Attribution
-- proving which retrieved passages support each claim -- is now a first-class evaluation
criterion for RAG systems.

URL: https://arxiv.org/html/2603.09891v1

### 4. Memory Transparency in Commercial Products

OpenAI ChatGPT shows "Memory updated" notifications and allows users to ask "What do you
remember about me?" Anthropic Claude has memory transparency features. Google Gemini
shows research steps in Deep Research mode. The standard is moving toward explicit
disclosure of what the AI "knows" and where that knowledge came from.

URL: https://aicompetence.org/memory-enhanced-ai-chatbots/

### 5. Microsoft RAG Best Practices

Microsoft (2025) identifies transparent source attribution as one of 5 key RAG features:
"RAG provides citations for retrieved information, making responses verifiable and
trustworthy... This auditability not only improves user confidence but also aligns with
regulatory requirements."

URL: https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/13/5-key-features-and-benefits-of-retrieval-augmented-generation-rag/

### 6. Google PAIR Guidebook

Google's People + AI Research guidebook recommends showing "what the system knows" as a
mechanism for calibrating user expectations and enabling correction of wrong assumptions.

URL: https://pair.withgoogle.com/guidebook/

## Proposed Implementation

### Architecture: Memory Attribution Panel + Inline Indicators

Two-layer approach matching the tool activity log pattern:

1. **Memory indicator bar** (collapsed): Shows how many memories were used, collapsed by
   default. Positioned ABOVE the tool activity log (memories inform which tools run).
2. **Memory cards** (expanded): Each retrieved memory shown as a card with content preview,
   scope, retrieval score, and retrieval reasons.

### Step 1: Backend -- Include memory metadata in response

**File: `agent_zero/agent_zero_server.py`**

After memory selection and before response generation, emit a `memory_context` WebSocket
message bundling all retrieved memories for this turn:

```python
async def _emit_memory_context(websocket, selected_notes, turn_index):
    """Emit memory context summary for UI transparency."""
    if not selected_notes:
        return
    notes = []
    for note in selected_notes:
        notes.append({
            "id": note.get("id", ""),
            "scope": note.get("scope", ""),
            "note_type": note.get("note_type", ""),
            "content_preview": note.get("content", "")[:300],
            "retrieval_score": note.get("retrieval_score", 0),
            "retrieval_reasons": note.get("retrieval_reasons", []),
            "created_at": note.get("created_at", ""),
        })
    msg = {
        "type": "memory_context",
        "content": {
            "turn_index": turn_index,
            "source": "automatic",  # or "tool_call" for explicit memory.recall
            "notes": notes,
            "ts": datetime.now(timezone.utc).isoformat(),
        },
    }
    await websocket.send_text(json.dumps(msg))
```

Call this after `build_retrieval_packet()` returns selected notes (around line 1323 for
automatic retrieval) and after `_memory_recall()` returns for tool-based retrieval.

### Step 2: Ensure retrieval_reasons are populated

**File: `agent_zero/memory_policy.py`**

In `select_memory_notes()`, the scoring logic already computes reasons but may not
attach them to the note. Ensure each selected note has a `retrieval_reasons` list:

```python
# In _score_note() (lines 84-125), build reasons list alongside score:
reasons = []
if query_overlap_score > 0:
    reasons.append(f"matches '{query}' ({query_overlap_score} terms)")
if note.get("note_type") == "reasoning_summary" and route == "strategic_reasoning":
    reasons.append("reasoning summary for strategic turn")
if priority_overlap:
    reasons.append(f"matches priority: {priority_overlap}")
# ... etc
note["retrieval_reasons"] = reasons
```

### Step 3: Frontend types

**File: `agent_zero-ui/src/types.ts`**

```typescript
export interface MemoryNote {
  id: string
  scope: string
  note_type: string
  content_preview: string
  retrieval_score: number
  retrieval_reasons: string[]
  created_at: string
}

export interface MemoryContext {
  turn_index: number
  source: 'automatic' | 'tool_call'
  notes: MemoryNote[]
  ts: string
}
```

### Step 4: Frontend state management

**File: `agent_zero-ui/src/App.tsx`**

```typescript
const [memoryContextByTurn, setMemoryContextByTurn] = useState<Record<number, MemoryContext>>({})

// In WebSocket handler:
if (type === 'memory_context') {
  const ctx = payload.content as MemoryContext
  setMemoryContextByTurn(prev => ({
    ...prev,
    [ctx.turn_index]: ctx
  }))
  return
}
```

Pass `memoryContextByTurn` to AppShell.

### Step 5: MemoryContextLog component

**File: `agent_zero-ui/src/components/MemoryContextLog.tsx`** (NEW)

```tsx
import { useState } from 'react'
import type { MemoryContext, MemoryNote } from '../types'

interface Props {
  context: MemoryContext
}

export function MemoryContextLog({ context }: Props) {
  const [expanded, setExpanded] = useState(false)
  const { notes, source } = context
  if (notes.length === 0) return null

  return (
    <div className="memory-context-log">
      <button
        className="memory-context-bar"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
      >
        <span className="memory-icon">&#128218;</span>
        <span className="memory-context-count">
          {notes.length} memor{notes.length !== 1 ? 'ies' : 'y'} recalled
          {source === 'tool_call' && ' (explicit)'}
        </span>
        <span className={`memory-context-chevron ${expanded ? 'open' : ''}`}>
          &#9662;
        </span>
      </button>

      {expanded && (
        <div className="memory-context-cards">
          {notes.map((note, i) => (
            <MemoryCard key={note.id || i} note={note} />
          ))}
        </div>
      )}
    </div>
  )
}

function MemoryCard({ note }: { note: MemoryNote }) {
  return (
    <div className="memory-card">
      <div className="memory-card-header">
        <span className="memory-card-scope">{note.scope}/{note.note_type}</span>
        <span className="memory-card-score">score: {note.retrieval_score}</span>
      </div>
      <div className="memory-card-content">{note.content_preview}</div>
      {note.retrieval_reasons.length > 0 && (
        <div className="memory-card-reasons">
          {note.retrieval_reasons.map((r, i) => (
            <span key={i} className="memory-reason-chip">{r}</span>
          ))}
        </div>
      )}
    </div>
  )
}
```

### Step 6: Insert in AppShell

**File: `agent_zero-ui/src/components/AppShell.tsx`**

For each Agent Zero message, insert the MemoryContextLog ABOVE the ToolActivityLog
(memories are retrieved before tools execute, and inform the response):

```tsx
{msg.role === 'agent_zero' && memoryContextByTurn[turnIndex] && (
  <MemoryContextLog context={memoryContextByTurn[turnIndex]} />
)}
{msg.role === 'agent_zero' && toolActivityByTurn[turnIndex] && (
  <ToolActivityLog activity={toolActivityByTurn[turnIndex]} />
)}
```

### Step 7: CSS styles

**File: `agent_zero-ui/src/styles.css`**

```css
/* ---- Memory Context Log ---- */
.memory-context-log {
  margin: 4px 0;
  font-family: var(--font-mono);
}

.memory-context-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 12px;
  background: rgba(108, 99, 255, 0.06);
  border: 1px solid rgba(108, 99, 255, 0.15);
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.72rem;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.memory-context-bar:hover {
  background: rgba(108, 99, 255, 0.1);
  border-color: rgba(108, 99, 255, 0.25);
  color: rgba(255, 255, 255, 0.7);
}

.memory-icon { font-size: 0.8rem; }
.memory-context-count { font-weight: 600; white-space: nowrap; }

.memory-context-chevron {
  margin-left: auto;
  transition: transform 0.15s ease;
  font-size: 0.7rem;
}
.memory-context-chevron.open { transform: rotate(180deg); }

.memory-context-cards {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px 0 2px 0;
}

.memory-card {
  padding: 8px 12px;
  background: rgba(108, 99, 255, 0.04);
  border: 1px solid rgba(108, 99, 255, 0.1);
  border-radius: 6px;
  border-left: 3px solid rgba(108, 99, 255, 0.4);
}

.memory-card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.68rem;
}
.memory-card-scope {
  color: rgba(108, 99, 255, 0.7);
  font-weight: 600;
}
.memory-card-score {
  margin-left: auto;
  color: rgba(255, 255, 255, 0.35);
}

.memory-card-content {
  font-size: 0.72rem;
  color: rgba(255, 255, 255, 0.55);
  margin-top: 4px;
  font-family: var(--font-serif);
  line-height: 1.4;
}

.memory-card-reasons {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;
}
.memory-reason-chip {
  font-size: 0.65rem;
  padding: 1px 6px;
  border-radius: 3px;
  background: rgba(108, 99, 255, 0.08);
  color: rgba(108, 99, 255, 0.6);
}
```

### Design Decisions

1. **Purple accent** for memory (vs green for tools) -- visual distinction between
   "what Agent Zero remembers" vs "what Agent Zero did"
2. **Collapsed by default** -- matches progressive disclosure pattern from tool activity log
3. **Content preview capped at 300 chars** -- enough to identify the memory without
   overwhelming the card
4. **Retrieval reasons shown as chips** -- explains WHY each memory was selected, not
   just WHAT was selected
5. **Score visible** -- power users can understand the ranking logic
6. **Source indicator** -- "explicit" for tool-based recall vs automatic for policy-based

### Interaction with Existing Systems

- **Tool activity log**: Memory context appears ABOVE tool activity (information hierarchy:
  context first, then actions, then response)
- **Agent chips**: No changes. Memory context is between chips and tool log.
- **Trace panel**: Continues to show full JSON for developers. The new component is the
  user-facing version.
- **Runtime tab**: No changes.
- **Backward compatibility**: Purely additive. No existing behavior changes.

## Test Specifications

### Backend Tests (`agent_zero/test_memory_transparency.py`)

```python
def test_memory_context_message_format():
    """_emit_memory_context produces correct JSON structure."""
    # Verify: type == "memory_context"
    # content has: turn_index (int), source (str), notes (list), ts (ISO)
    # Each note: id, scope, note_type, content_preview, retrieval_score, retrieval_reasons, created_at

def test_memory_context_content_preview_truncated():
    """content_preview is capped at 300 chars."""
    # Given: note with 1000-char content
    # Then: content_preview <= 300 chars

def test_memory_context_not_emitted_when_no_notes():
    """No memory_context sent when zero notes selected."""

def test_memory_context_automatic_source():
    """Automatic retrieval sets source='automatic'."""

def test_memory_context_tool_source():
    """Tool-based memory.recall sets source='tool_call'."""

def test_retrieval_reasons_populated():
    """select_memory_notes() attaches retrieval_reasons to each note."""
    # Given: note matching query + being reasoning_summary
    # Then: retrieval_reasons includes both "matches query" and "reasoning summary"

def test_retrieval_reasons_empty_for_low_score():
    """Notes with minimal match still have empty reasons list (not None)."""
```

### Frontend Tests (vitest + react-testing-library)

```
test: MemoryContextLog renders nothing when notes array is empty
test: MemoryContextLog renders bar with correct note count (singular/plural)
test: MemoryContextLog shows "(explicit)" for tool_call source
test: clicking bar toggles expanded state
test: collapsed state does not render memory cards
test: expanded state renders one MemoryCard per note
test: MemoryCard shows scope/note_type header
test: MemoryCard shows content preview
test: MemoryCard shows retrieval score
test: MemoryCard renders reason chips when present
test: MemoryCard hides reason chips when retrieval_reasons is empty
test: memory_context WebSocket message updates memoryContextByTurn state
test: MemoryContextLog appears above ToolActivityLog in message rendering
```

## Estimated Impact

**User trust**: Making retrieval visible is the #1 mechanism for building trust in RAG
systems (TREC 2025, Microsoft 2025, PAIR guidebook). Users who can see what memories
informed a response will have significantly higher confidence in the system.

**Error correction**: When Agent Zero recalls stale or irrelevant memories, users can see
this immediately and say "that's outdated" or "you're thinking of the wrong thing." This
is currently impossible without navigating to the trace panel.

**Personalization transparency**: Memory is how Agent Zero personalizes. Making it visible
transforms "magic" into "observable intelligence" -- the user understands WHY Agent Zero
knows what it knows.

**Minimal overhead**: One WebSocket message per turn with memory data already computed.
Frontend renders a simple collapsible component. No additional DB queries or API calls.

## References

1. ShapeOfAI (2025). "AI UX Patterns: Citations."
   https://www.shapeof.ai/patterns/citations

2. Zhao et al. (2025). "Seeing to Think? How Source Transparency Design Shapes
   Interactive Information Seeking and Evaluation in Conversational AI."
   https://arxiv.org/html/2601.14611

3. TREC 2025 RAG Track. "Overview of the TREC 2025 Retrieval Augmented Generation Track."
   https://arxiv.org/html/2603.09891v1

4. Microsoft (2025). "5 Key Features and Benefits of RAG."
   https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/02/13/5-key-features-and-benefits-of-retrieval-augmented-generation-rag/

5. AI Competence (2025). "Memory-Enhanced AI Chatbots: Smarter Conversations Ahead."
   https://aicompetence.org/memory-enhanced-ai-chatbots/

6. Google PAIR (2025). People + AI Guidebook.
   https://pair.withgoogle.com/guidebook/

7. Geolyze (2025). "How AI Engines Cite Sources: Patterns Across ChatGPT, Claude,
   Perplexity, and SGE."
   https://medium.com/@shuimuzhisou/how-ai-engines-cite-sources-patterns-across-chatgpt-claude-perplexity-and-sge-8c317777c71d
