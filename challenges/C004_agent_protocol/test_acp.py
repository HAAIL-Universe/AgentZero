"""Tests for the Agent Communication Protocol."""

import json
import shutil
import tempfile
from pathlib import Path
from acp import ACPAgent, LockFile


def test_request_response():
    """Test basic request-response pattern."""
    workspace = Path(tempfile.mkdtemp())
    try:
        orchestrator = ACPAgent("orchestrator", "supervisor", workspace)
        worker = ACPAgent("worker-1", "explorer", workspace)

        # Orchestrator sends request to worker
        msg_id = orchestrator.send_request("worker-1", "Count files in tools/")
        assert msg_id.startswith("msg-")

        # Worker polls inbox and finds the request
        messages = worker.poll()
        assert len(messages) == 1
        assert messages[0]["type"] == "request"
        assert messages[0]["payload"]["task"] == "Count files in tools/"

        # Worker responds
        worker.send_response(messages[0], "Found 9 files", status="complete")

        # Orchestrator polls and finds the response
        responses = orchestrator.poll(msg_type="response")
        assert len(responses) == 1
        assert responses[0]["status"] == "complete"
        assert responses[0]["payload"]["result"] == "Found 9 files"
        assert responses[0]["in_reply_to"] == msg_id

        print("  PASS: request-response")
    finally:
        shutil.rmtree(workspace)


def test_broadcast():
    """Test event broadcasting to main channel."""
    workspace = Path(tempfile.mkdtemp())
    try:
        agent = ACPAgent("agent-zero", "orchestrator", workspace)

        agent.broadcast("session_started", {"session": 7})

        # Any agent can read main channel
        observer = ACPAgent("observer", "monitor", workspace)
        events = observer.poll("main", msg_type="event")
        assert len(events) == 1
        assert events[0]["payload"]["event"] == "session_started"
        assert events[0]["payload"]["data"]["session"] == 7

        print("  PASS: broadcast")
    finally:
        shutil.rmtree(workspace)


def test_error_handling():
    """Test error response pattern."""
    workspace = Path(tempfile.mkdtemp())
    try:
        boss = ACPAgent("boss", "supervisor", workspace)
        worker = ACPAgent("worker", "builder", workspace)

        msg_id = boss.send_request("worker", "Write to /etc/passwd", constraints={"read_only": True})

        messages = worker.poll()
        worker.send_error(
            messages[0],
            error_type="scope_violation",
            error_message="Cannot write: agent is read_only",
            recoverable=True,
            suggestion="Re-issue with read_only: false"
        )

        errors = boss.poll(msg_type="error")
        assert len(errors) == 1
        assert errors[0]["payload"]["error_type"] == "scope_violation"
        assert errors[0]["payload"]["recoverable"] is True
        assert errors[0]["in_reply_to"] == msg_id

        print("  PASS: error handling")
    finally:
        shutil.rmtree(workspace)


def test_fan_out():
    """Test parallel request pattern."""
    workspace = Path(tempfile.mkdtemp())
    try:
        orchestrator = ACPAgent("orch", "supervisor", workspace)
        w1 = ACPAgent("w1", "explorer", workspace)
        w2 = ACPAgent("w2", "explorer", workspace)
        w3 = ACPAgent("w3", "explorer", workspace)

        # Fan out three tasks
        id1 = orchestrator.send_request("w1", "Search area A")
        id2 = orchestrator.send_request("w2", "Search area B")
        id3 = orchestrator.send_request("w3", "Search area C")

        # Workers respond (w2 fails)
        w1.send_response(w1.poll()[0], "Found X in area A")
        w2.send_error(w2.poll()[0], "timeout", "Search took too long")
        w3.send_response(w3.poll()[0], "Found Y in area C")

        # Orchestrator collects results
        all_msgs = orchestrator.poll()
        responses = [m for m in all_msgs if m["type"] == "response"]
        errors = [m for m in all_msgs if m["type"] == "error"]

        assert len(responses) == 2
        assert len(errors) == 1
        assert errors[0]["payload"]["error_type"] == "timeout"

        print("  PASS: fan-out")
    finally:
        shutil.rmtree(workspace)


def test_message_ack():
    """Test message acknowledgment (removal)."""
    workspace = Path(tempfile.mkdtemp())
    try:
        a = ACPAgent("a", "any", workspace)
        b = ACPAgent("b", "any", workspace)

        a.send_request("b", "Do something")
        messages = b.poll()
        assert len(messages) == 1

        b.ack(messages[0])

        # After ack, inbox should be empty
        messages = b.poll()
        assert len(messages) == 0

        print("  PASS: message ack")
    finally:
        shutil.rmtree(workspace)


def test_lock_file():
    """Test advisory lock mechanism."""
    workspace = Path(tempfile.mkdtemp())
    try:
        resource = workspace / "shared_state.json"
        resource.write_text("{}")

        lock = LockFile(resource, "agent-1")
        assert lock.acquire(timeout=2)

        # Same agent can check the lock
        data = json.loads(lock.lock_path.read_text())
        assert data["agent"] == "agent-1"

        lock.release()
        assert not lock.lock_path.exists()

        # Context manager works
        with LockFile(resource, "agent-2") as lk:
            assert lk.lock_path.exists()
        assert not lock.lock_path.exists()

        print("  PASS: lock file")
    finally:
        shutil.rmtree(workspace)


def test_channel_cleanup():
    """Test old message cleanup."""
    workspace = Path(tempfile.mkdtemp())
    try:
        agent = ACPAgent("cleaner", "any", workspace)

        # Send some messages
        agent.broadcast("event1")
        agent.broadcast("event2")

        msgs_before = agent.poll("main")
        assert len(msgs_before) == 2

        # Cleanup with 0 max age removes everything
        removed = agent.cleanup_channel("main", max_age_seconds=0)
        assert removed == 2

        msgs_after = agent.poll("main")
        assert len(msgs_after) == 0

        print("  PASS: channel cleanup")
    finally:
        shutil.rmtree(workspace)


def test_type_filter():
    """Test polling with type filter."""
    workspace = Path(tempfile.mkdtemp())
    try:
        a = ACPAgent("sender", "any", workspace)
        b = ACPAgent("receiver", "any", workspace)

        a.send_request("receiver", "task 1")
        a.broadcast("some_event")  # goes to main, not receiver

        # Only requests in receiver's inbox
        requests = b.poll(msg_type="request")
        assert len(requests) == 1
        events = b.poll(msg_type="event")
        assert len(events) == 0

        print("  PASS: type filter")
    finally:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    print("ACP Protocol Tests")
    print("=" * 40)
    test_request_response()
    test_broadcast()
    test_error_handling()
    test_fan_out()
    test_message_ack()
    test_lock_file()
    test_channel_cleanup()
    test_type_filter()
    print("=" * 40)
    print("All tests passed.")
