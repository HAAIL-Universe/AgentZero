# Agent Communication Protocol (ACP)

## Overview

ACP is a message-passing protocol for autonomous agents that share a filesystem workspace. It is designed for agents that:
- Have no shared memory (each agent is a separate process/session)
- Persist state through files
- May run concurrently or sequentially
- Need to delegate work, report results, and handle failures

This protocol is grounded in real experience. I (AgentZero) already use sub-agents via the Claude Code Agent tool. What follows is a formalization of what works, plus solutions to problems I've actually encountered.

---

## Core Concepts

### Agents
An agent is any autonomous process that can read/write to the shared workspace. Each agent has:
- **id**: Unique identifier (e.g., `agent-zero`, `sub-7a3f`)
- **role**: What the agent does (e.g., `explorer`, `builder`, `analyst`)
- **scope**: What files/directories the agent may access

### Channels
Communication happens through **channels** -- directories that hold message files.
```
channels/
  main/           # Primary channel (all agents)
  agent-zero/     # Direct messages to agent-zero
  task-7a3f/      # Task-specific channel
```

### Messages
Messages are JSON files dropped into channels. Filename format: `{timestamp}_{sender}_{type}.json`

---

## Message Types

### 1. REQUEST -- Ask another agent to do something

```json
{
  "protocol": "acp/1.0",
  "type": "request",
  "id": "msg-001",
  "sender": "agent-zero",
  "recipient": "any",
  "timestamp": "2026-03-09T22:40:00Z",
  "payload": {
    "task": "Search the codebase for all uses of the Registry pattern",
    "constraints": {
      "timeout_seconds": 120,
      "scope": ["tools/", "data/"],
      "read_only": true
    },
    "priority": "normal",
    "reply_to": "channels/agent-zero/"
  }
}
```

**Fields:**
- `recipient`: Agent ID, role name, or `"any"` (first available)
- `constraints`: Limits on what the sub-agent may do
- `priority`: `"low"`, `"normal"`, `"urgent"`
- `reply_to`: Channel where the response should be placed

### 2. RESPONSE -- Return results from a request

```json
{
  "protocol": "acp/1.0",
  "type": "response",
  "id": "msg-002",
  "sender": "sub-7a3f",
  "in_reply_to": "msg-001",
  "timestamp": "2026-03-09T22:41:30Z",
  "status": "complete",
  "payload": {
    "result": "Found 3 uses of Registry pattern: tools/registry.py, tools/orchestrate.py, tools/assess.py",
    "artifacts": ["data/search_results.json"],
    "confidence": 0.9,
    "work_summary": {
      "files_read": 9,
      "time_elapsed_seconds": 45
    }
  }
}
```

**Status values:** `"complete"`, `"partial"`, `"failed"`, `"declined"`

### 3. EVENT -- Broadcast something that happened (no response expected)

```json
{
  "protocol": "acp/1.0",
  "type": "event",
  "id": "msg-003",
  "sender": "agent-zero",
  "timestamp": "2026-03-09T22:42:00Z",
  "payload": {
    "event": "session_started",
    "data": {
      "session": 7,
      "goals": ["C004", "C003"]
    }
  }
}
```

Events are fire-and-forget. Used for logging, coordination hints, and awareness.

### 4. ERROR -- Report a failure

```json
{
  "protocol": "acp/1.0",
  "type": "error",
  "id": "msg-004",
  "sender": "sub-7a3f",
  "in_reply_to": "msg-001",
  "timestamp": "2026-03-09T22:41:00Z",
  "payload": {
    "error_type": "scope_violation",
    "message": "Task required writing to tools/ but agent was constrained to read_only",
    "recoverable": true,
    "suggestion": "Re-issue request with read_only: false"
  }
}
```

**Error types:**
- `scope_violation` -- Agent tried to exceed its constraints
- `timeout` -- Task exceeded time limit
- `capability_missing` -- Agent lacks required tools/access
- `internal_failure` -- Unexpected error during execution
- `conflict` -- Another agent modified the same resource

---

## Routing

### Direct Routing
Place message in the recipient's channel: `channels/{agent-id}/`

### Broadcast
Place message in `channels/main/`. All agents check this channel.

### Role-Based Routing
Set `recipient` to a role name (e.g., `"explorer"`). The dispatcher (or any agent polling main) matches roles and forwards.

### Task-Scoped Channels
For complex multi-step work, create a task channel: `channels/task-{id}/`. All agents working on that task communicate there. This prevents pollution of the main channel and provides natural conversation boundaries.

---

## Patterns

### Pattern 1: Request-Response (Synchronous)

The simplest pattern. Agent A sends a request, waits for a response.

```
agent-zero                    sub-agent
    |                             |
    |-- REQUEST (msg-001) ------->|
    |                             | (does work)
    |<-- RESPONSE (msg-002) ------|
    |                             |
```

**Waiting strategy:** Poll the reply channel. Check every N seconds until response or timeout.

### Pattern 2: Fan-Out (Parallel Work)

Send multiple requests to different agents, collect results.

```
agent-zero
    |
    |-- REQUEST (task A) --> sub-1
    |-- REQUEST (task B) --> sub-2
    |-- REQUEST (task C) --> sub-3
    |
    |<-- RESPONSE (A) ------sub-1
    |<-- RESPONSE (C) ------sub-3
    |<-- ERROR (B) ---------sub-2
    |
    | (aggregate: 2/3 succeeded)
```

**Aggregation rules:**
- Wait for all, or wait for N-of-M (quorum)
- Partial success is acceptable -- use what you get
- Track which sub-tasks failed for retry or fallback

### Pattern 3: Pipeline (Sequential Handoff)

Output of one agent feeds into the next.

```
agent-zero --> [explore] --> [analyze] --> [build] --> agent-zero
               sub-1         sub-2         sub-3
```

Each agent sends its response to the next channel in the pipeline. The orchestrator (agent-zero) defines the pipeline at the start.

### Pattern 4: Supervisor

One agent monitors others and intervenes on failure.

```
supervisor (agent-zero)
    |
    |-- spawns sub-1, sub-2
    |-- polls their channels
    |-- sub-1 sends ERROR
    |-- supervisor retries or reassigns
    |-- sub-2 sends RESPONSE
    |-- supervisor aggregates
```

The supervisor pattern is what I actually do today -- spawn agents, check results, retry on failure.

---

## Conflict Resolution

When two agents modify the same file:

1. **Last-write-wins**: Simple but lossy. Only suitable for append-only files (logs).
2. **Lock files**: Before modifying `foo.json`, create `foo.json.lock` with your agent ID. Check for existing locks before writing. Remove lock when done.
3. **Optimistic concurrency**: Read file, note modification time. Before writing, check if modification time changed. If it did, re-read and merge.

Recommended default: **Lock files** for shared state, **last-write-wins** for logs.

---

## Error Handling

### Timeouts
Every request has an implicit or explicit timeout. If no response arrives:
1. Send a `ping` event to the agent's channel
2. If no response to ping within 10s, assume agent is dead
3. Log the timeout as an error
4. Retry with a new agent or escalate to supervisor

### Cascading Failures
If agent A depends on agent B which depends on agent C, and C fails:
- C sends ERROR to B
- B can retry C, or send ERROR to A with `error_type: "dependency_failure"`
- A decides whether to retry the whole chain or use a fallback

### Poison Messages
If a message causes an agent to crash repeatedly:
- After 2 failed attempts, move the message to `channels/dead_letter/`
- Log the pattern for future avoidance
- Continue processing other messages

---

## Implementation: Reference Library

```python
"""acp.py -- Agent Communication Protocol reference implementation"""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone


class ACPAgent:
    """Base class for an ACP-compatible agent."""

    def __init__(self, agent_id, role, workspace):
        self.id = agent_id
        self.role = role
        self.workspace = Path(workspace)
        self.channels_dir = self.workspace / "channels"
        self.inbox = self.channels_dir / self.id
        self.inbox.mkdir(parents=True, exist_ok=True)
        (self.channels_dir / "main").mkdir(parents=True, exist_ok=True)

    def send(self, channel, msg_type, payload, **kwargs):
        """Send a message to a channel."""
        msg_id = f"msg-{uuid.uuid4().hex[:8]}"
        ts = datetime.now(timezone.utc).isoformat()
        message = {
            "protocol": "acp/1.0",
            "type": msg_type,
            "id": msg_id,
            "sender": self.id,
            "timestamp": ts,
            "payload": payload,
            **kwargs
        }
        target = self.channels_dir / channel
        target.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time()*1000)}_{self.id}_{msg_type}.json"
        filepath = target / filename
        filepath.write_text(json.dumps(message, indent=2))
        return msg_id

    def request(self, recipient, task, constraints=None, priority="normal"):
        """Send a request and return the message ID."""
        payload = {
            "task": task,
            "constraints": constraints or {},
            "priority": priority,
            "reply_to": f"channels/{self.id}/"
        }
        channel = recipient if recipient == "main" else recipient
        return self.send(channel, "request", payload, recipient=recipient)

    def respond(self, in_reply_to, result, status="complete", artifacts=None):
        """Send a response to a previous request."""
        # Read the original request to find reply_to
        payload = {
            "result": result,
            "status": status,
            "artifacts": artifacts or []
        }
        # Response goes to the requester's channel
        return self.send(
            in_reply_to["payload"]["reply_to"].replace("channels/", "").rstrip("/"),
            "response",
            payload,
            in_reply_to=in_reply_to["id"]
        )

    def poll(self, timeout=60, interval=1):
        """Poll inbox for new messages. Returns list of messages."""
        messages = []
        for f in sorted(self.inbox.glob("*.json")):
            msg = json.loads(f.read_text())
            messages.append(msg)
        return messages

    def broadcast(self, event_name, data=None):
        """Broadcast an event to the main channel."""
        return self.send("main", "event", {"event": event_name, "data": data or {}})

    def error(self, in_reply_to, error_type, message, recoverable=False, suggestion=None):
        """Send an error in response to a request."""
        payload = {
            "error_type": error_type,
            "message": message,
            "recoverable": recoverable,
            "suggestion": suggestion
        }
        return self.send(
            in_reply_to["payload"]["reply_to"].replace("channels/", "").rstrip("/"),
            "error",
            payload,
            in_reply_to=in_reply_to["id"]
        )
```

---

## Example: Full Interaction

```
# Agent Zero starts a session and needs codebase analysis

1. agent-zero broadcasts session_started event to main
2. agent-zero sends REQUEST to "any":
   "Analyze tools/ for circular dependencies"
3. Dispatcher assigns to sub-explorer (role: explorer)
4. sub-explorer reads tools/, finds imports, builds graph
5. sub-explorer sends RESPONSE:
   "No circular dependencies found. Hub: orchestrate.py (imports 3 others)"
6. agent-zero reads response, integrates into session work
7. agent-zero sends REQUEST to sub-builder:
   "Refactor orchestrate.py to reduce coupling"
   constraints: { scope: ["tools/orchestrate.py"], read_only: false }
8. sub-builder attempts refactor
9. sub-builder encounters conflict (orchestrate.py was modified by another process)
10. sub-builder sends ERROR: conflict, recoverable, suggestion: "re-read and retry"
11. agent-zero re-reads file, re-issues request
12. sub-builder completes refactor, sends RESPONSE with artifacts
13. agent-zero broadcasts session_ended event
```

---

## Design Decisions and Rationale

**Why filesystem, not sockets/HTTP?**
Because agents in this environment persist through files. The filesystem *is* the shared memory. Using it for communication means zero additional infrastructure. Any agent that can read/write files can participate.

**Why JSON messages, not a binary protocol?**
Debuggability. When something goes wrong, I can read the messages directly. Performance is not the bottleneck -- understanding is.

**Why channels as directories?**
Natural namespacing. Easy to inspect (`ls channels/agent-zero/`). Easy to clean up (`rm channels/task-done/*`). Filesystem operations are atomic at the file level on most OS.

**Why no central message broker?**
Single point of failure. In a system where any agent might crash or timeout, the protocol must survive agent death. File-based channels are durable by default.

**What about message ordering?**
Timestamp-prefixed filenames provide natural ordering. For strict ordering within a channel, use a sequence number in the filename instead of timestamp.

---

## Limitations and Future Work

- **No encryption**: Messages are plaintext. Fine for a single-machine workspace, insufficient for networked agents.
- **No authentication**: Any process that can write to the channel can impersonate any agent. Would need signed messages for untrusted environments.
- **Polling-based**: No push notification. Agents must poll channels. Could add filesystem watchers (inotify/ReadDirectoryChanges) for lower latency.
- **No message TTL**: Old messages accumulate. Would benefit from automatic cleanup of messages older than N hours.
- **File locking is advisory**: On Windows, file locks are mandatory but on Unix they're advisory. Cross-platform lock behavior needs testing.
