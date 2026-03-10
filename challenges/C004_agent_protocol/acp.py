"""
ACP -- Agent Communication Protocol

Reference implementation for filesystem-based agent communication.
Designed for agents that persist through files and may run concurrently.

Usage:
    from acp import ACPAgent

    agent = ACPAgent("agent-zero", "orchestrator", "Z:/AgentZero")
    msg_id = agent.request("sub-1", "Search for pattern X in tools/")
    messages = agent.poll()
"""

import json
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone


class ACPAgent:
    """An agent that communicates via the ACP protocol."""

    def __init__(self, agent_id, role, workspace):
        self.id = agent_id
        self.role = role
        self.workspace = Path(workspace)
        self.channels_dir = self.workspace / "channels"
        self.inbox = self.channels_dir / self.id
        self.inbox.mkdir(parents=True, exist_ok=True)
        (self.channels_dir / "main").mkdir(parents=True, exist_ok=True)
        (self.channels_dir / "dead_letter").mkdir(parents=True, exist_ok=True)

    def _make_id(self):
        return f"msg-{uuid.uuid4().hex[:8]}"

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _write_message(self, channel, message):
        """Write a message file to a channel directory."""
        target = self.channels_dir / channel
        target.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        filename = f"{ts}_{self.id}_{message['type']}.json"
        filepath = target / filename
        filepath.write_text(json.dumps(message, indent=2), encoding="utf-8")
        return filepath

    def send_request(self, recipient, task, constraints=None, priority="normal"):
        """Send a task request to another agent."""
        msg_id = self._make_id()
        message = {
            "protocol": "acp/1.0",
            "type": "request",
            "id": msg_id,
            "sender": self.id,
            "recipient": recipient,
            "timestamp": self._now(),
            "payload": {
                "task": task,
                "constraints": constraints or {},
                "priority": priority,
                "reply_to": self.id
            }
        }
        self._write_message(recipient, message)
        return msg_id

    def send_response(self, original_msg, result, status="complete", artifacts=None):
        """Respond to a request."""
        msg_id = self._make_id()
        reply_to = original_msg["payload"]["reply_to"]
        message = {
            "protocol": "acp/1.0",
            "type": "response",
            "id": msg_id,
            "sender": self.id,
            "in_reply_to": original_msg["id"],
            "timestamp": self._now(),
            "status": status,
            "payload": {
                "result": result,
                "artifacts": artifacts or []
            }
        }
        self._write_message(reply_to, message)
        return msg_id

    def send_error(self, original_msg, error_type, error_message, recoverable=False, suggestion=None):
        """Report an error in response to a request."""
        msg_id = self._make_id()
        reply_to = original_msg["payload"]["reply_to"]
        message = {
            "protocol": "acp/1.0",
            "type": "error",
            "id": msg_id,
            "sender": self.id,
            "in_reply_to": original_msg["id"],
            "timestamp": self._now(),
            "payload": {
                "error_type": error_type,
                "message": error_message,
                "recoverable": recoverable,
                "suggestion": suggestion
            }
        }
        self._write_message(reply_to, message)
        return msg_id

    def broadcast(self, event_name, data=None):
        """Broadcast an event to the main channel."""
        msg_id = self._make_id()
        message = {
            "protocol": "acp/1.0",
            "type": "event",
            "id": msg_id,
            "sender": self.id,
            "timestamp": self._now(),
            "payload": {
                "event": event_name,
                "data": data or {}
            }
        }
        self._write_message("main", message)
        return msg_id

    def poll(self, channel=None, msg_type=None, since=None):
        """Read messages from a channel. Defaults to own inbox."""
        target = self.channels_dir / (channel or self.id)
        if not target.exists():
            return []

        messages = []
        for f in sorted(target.glob("*.json")):
            try:
                msg = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if msg_type and msg.get("type") != msg_type:
                continue
            if since and msg.get("timestamp", "") < since:
                continue

            msg["_filepath"] = str(f)
            messages.append(msg)

        return messages

    def poll_responses(self, request_id, timeout=60, interval=1):
        """Wait for a response to a specific request."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            for msg in self.poll():
                if msg.get("in_reply_to") == request_id:
                    return msg
            time.sleep(interval)
        return None

    def ack(self, message):
        """Acknowledge and remove a message from inbox."""
        filepath = message.get("_filepath")
        if filepath:
            Path(filepath).unlink(missing_ok=True)

    def cleanup_channel(self, channel, max_age_seconds=3600):
        """Remove old messages from a channel."""
        target = self.channels_dir / channel
        if not target.exists():
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for f in target.glob("*.json"):
            # Filename starts with timestamp in milliseconds
            try:
                file_ts = int(f.stem.split("_")[0]) / 1000
                if file_ts < cutoff:
                    f.unlink()
                    removed += 1
            except (ValueError, IndexError):
                continue
        return removed


class LockFile:
    """Advisory file lock for shared resource access."""

    def __init__(self, resource_path, agent_id):
        self.lock_path = Path(str(resource_path) + ".lock")
        self.agent_id = agent_id

    def acquire(self, timeout=10):
        """Try to acquire the lock. Returns True on success."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.lock_path.exists():
                self.lock_path.write_text(json.dumps({
                    "agent": self.agent_id,
                    "acquired": datetime.now(timezone.utc).isoformat()
                }))
                # Verify we got it (basic race condition check)
                time.sleep(0.05)
                try:
                    data = json.loads(self.lock_path.read_text())
                    if data["agent"] == self.agent_id:
                        return True
                except (json.JSONDecodeError, OSError, KeyError):
                    pass
            time.sleep(0.1)
        return False

    def release(self):
        """Release the lock."""
        self.lock_path.unlink(missing_ok=True)

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.lock_path}")
        return self

    def __exit__(self, *args):
        self.release()
