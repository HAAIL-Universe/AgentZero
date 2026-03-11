"""
Tests for C209: Distributed Lock Service
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from distributed_lock import (
    LockStateMachine, LockServiceCluster, LockClient, LockServiceStats,
    LockMode, LockState, SessionState, WatchEvent, FencingToken,
    LockEntry, WaitEntry, Session, Notification, WatchRegistration,
)


# ===========================================================================
# Helper
# ===========================================================================

def make_cluster(n=3):
    """Create and initialize a lock service cluster."""
    ids = [f"node{i}" for i in range(1, n + 1)]
    cluster = LockServiceCluster(ids)
    cluster.wait_for_leader()
    return cluster


def make_client(cluster, client_id=None, ttl=30.0, now=0):
    """Create a client with an active session."""
    client = LockClient(cluster, client_id=client_id)
    client.create_session(ttl=ttl, now=now)
    return client


# ===========================================================================
# 1. State Machine Unit Tests
# ===========================================================================

class TestLockStateMachine:
    def test_create(self):
        sm = LockStateMachine()
        assert sm.locks == {}
        assert sm.sessions == {}
        assert sm.fencing_counter == 0

    def test_apply_noop(self):
        sm = LockStateMachine()
        result = sm.apply(None, 1)
        assert result["ok"]

    def test_unknown_op(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "nonexistent"}, 1)
        assert not result["ok"]
        assert "unknown" in result["error"]

    def test_create_session(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        assert result["ok"]
        assert result["session_id"] == "s1"
        assert "s1" in sm.sessions

    def test_create_session_duplicate(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        result = sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 2)
        assert result["ok"]
        assert result.get("already_exists")

    def test_heartbeat(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        result = sm.apply({"op": "heartbeat", "session_id": "s1", "now": 5}, 2)
        assert result["ok"]
        assert result["expires_at"] == 15

    def test_heartbeat_invalid_session(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "heartbeat", "session_id": "nonexistent", "now": 0}, 1)
        assert not result["ok"]

    def test_destroy_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        result = sm.apply({"op": "destroy_session", "session_id": "s1", "now": 0}, 2)
        assert result["ok"]
        assert sm.sessions["s1"].state == SessionState.EXPIRED

    def test_destroy_nonexistent(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "destroy_session", "session_id": "nope"}, 1)
        assert not result["ok"]

    def test_acquire_exclusive(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        result = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        assert result["ok"]
        assert "fencing_token" in result
        assert result["fencing_token"] == 1

    def test_acquire_already_held(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        assert result["ok"]
        assert result.get("already_held")

    def test_acquire_contention_queues(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        assert result["ok"]
        assert result.get("queued")
        assert result["position"] == 1

    def test_acquire_invalid_session(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "acquire", "resource": "r1", "session_id": "nope",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 1)
        assert not result["ok"]

    def test_release(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 3)
        assert result["ok"]
        assert "r1" not in sm.locks

    def test_release_not_held(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        result = sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 2)
        assert not result["ok"]

    def test_release_wrong_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "release", "resource": "r1", "session_id": "s2", "now": 0}, 4)
        assert not result["ok"]

    def test_try_acquire_success(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        result = sm.apply({"op": "try_acquire", "resource": "r1", "session_id": "s1",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        assert result["ok"]
        assert result["acquired"]

    def test_try_acquire_fail(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "try_acquire", "resource": "r1", "session_id": "s2",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        assert result["ok"]
        assert not result["acquired"]

    def test_try_acquire_already_held(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "try_acquire", "resource": "r1", "session_id": "s1",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        assert result["ok"]
        assert result["acquired"]
        assert result.get("already_held")


# ===========================================================================
# 2. Shared Lock Tests
# ===========================================================================

class TestSharedLocks:
    def test_shared_multiple_holders(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        r1 = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                       "mode": "shared", "ttl": 30, "now": 0}, 3)
        r2 = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                       "mode": "shared", "ttl": 30, "now": 0}, 4)
        assert r1["ok"] and r2["ok"]
        assert isinstance(sm.locks["r1"], list)
        assert len(sm.locks["r1"]) == 2

    def test_exclusive_blocked_by_shared(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "try_acquire", "resource": "r1", "session_id": "s2",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        assert not result["acquired"]

    def test_shared_blocked_by_exclusive(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "try_acquire", "resource": "r1", "session_id": "s2",
                           "mode": "shared", "ttl": 30, "now": 0}, 4)
        assert not result["acquired"]

    def test_release_one_shared(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "shared", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 5)
        assert isinstance(sm.locks["r1"], list)
        assert len(sm.locks["r1"]) == 1
        assert sm.locks["r1"][0].owner == "s2"

    def test_release_all_shared(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 3)
        assert "r1" not in sm.locks

    def test_shared_already_held(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                           "mode": "shared", "ttl": 30, "now": 0}, 3)
        assert result["ok"]
        assert result.get("already_held")


# ===========================================================================
# 3. Fencing Token Tests
# ===========================================================================

class TestFencingTokens:
    def test_monotonic(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)

        r1 = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                       "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        t1 = r1["fencing_token"]

        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 4)

        r2 = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                       "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        t2 = r2["fencing_token"]

        assert t2 > t1

    def test_different_resources_independent(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        r1 = sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                       "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        r2 = sm.apply({"op": "acquire", "resource": "r2", "session_id": "s1",
                       "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        # Both get tokens, monotonically increasing globally
        assert r2["fencing_token"] > r1["fencing_token"]

    def test_fencing_token_comparison(self):
        t1 = FencingToken(1, "r1", "s1", 1)
        t2 = FencingToken(2, "r1", "s2", 1)
        assert t2 > t1
        assert t2 >= t1
        assert not (t1 > t2)

    def test_fencing_token_not_comparable_to_other(self):
        t1 = FencingToken(1, "r1", "s1", 1)
        assert t1.__gt__(42) is NotImplemented


# ===========================================================================
# 4. Wait Queue / Promotion Tests
# ===========================================================================

class TestWaitQueues:
    def test_promote_on_release(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        assert "r1" in sm.wait_queues

        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 5)
        # s2 should now hold the lock
        assert "r1" in sm.locks
        assert sm.locks["r1"].owner == "s2"
        # Wait queue should be empty
        assert "r1" not in sm.wait_queues or len(sm.wait_queues.get("r1", [])) == 0

    def test_promote_shared_waiters(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "create_session", "session_id": "s3", "ttl": 30, "now": 0}, 3)

        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "shared", "ttl": 30, "now": 0}, 5)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s3",
                  "mode": "shared", "ttl": 30, "now": 0}, 6)

        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 7)
        # Both shared waiters should be promoted
        assert isinstance(sm.locks["r1"], list)
        assert len(sm.locks["r1"]) == 2

    def test_no_double_queue(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        assert len(sm.wait_queues["r1"]) == 1

    def test_promote_skips_expired_sessions(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "create_session", "session_id": "s3", "ttl": 30, "now": 0}, 3)

        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s3",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 6)

        # Expire s2
        sm.apply({"op": "destroy_session", "session_id": "s2", "now": 0}, 7)

        # Release s1 -- should skip s2, promote s3
        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 8)
        assert sm.locks["r1"].owner == "s3"


# ===========================================================================
# 5. Expiration Tests
# ===========================================================================

class TestExpiration:
    def test_lock_expires(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 10, "now": 0}, 2)
        result = sm.apply({"op": "expire_check", "now": 10}, 3)
        assert "r1" in result["expired_locks"]
        assert "r1" not in sm.locks

    def test_session_expires(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 100, "now": 0}, 2)
        result = sm.apply({"op": "expire_check", "now": 10}, 3)
        assert "s1" in result["expired_sessions"]
        assert "r1" not in sm.locks

    def test_heartbeat_extends_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        sm.apply({"op": "heartbeat", "session_id": "s1", "now": 8}, 2)
        result = sm.apply({"op": "expire_check", "now": 10}, 3)
        assert "s1" not in result["expired_sessions"]

    def test_expired_lock_promotes_waiter(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 100, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 5, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "expire_check", "now": 5}, 5)
        assert sm.locks["r1"].owner == "s2"

    def test_renew_extends_ttl(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 5, "now": 0}, 2)
        result = sm.apply({"op": "renew", "resource": "r1", "session_id": "s1",
                           "ttl": 20, "now": 4}, 3)
        assert result["ok"]
        assert result["expires_at"] == 24
        # Should not expire at time 5
        result2 = sm.apply({"op": "expire_check", "now": 5}, 4)
        assert "r1" not in result2["expired_locks"]

    def test_renew_not_held(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        result = sm.apply({"op": "renew", "resource": "r1", "session_id": "s1",
                           "ttl": 20, "now": 0}, 2)
        assert not result["ok"]

    def test_renew_wrong_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 100, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "renew", "resource": "r1", "session_id": "s2",
                           "ttl": 20, "now": 0}, 4)
        assert not result["ok"]

    def test_renew_shared_lock(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 5, "now": 0}, 2)
        result = sm.apply({"op": "renew", "resource": "r1", "session_id": "s1",
                           "ttl": 20, "now": 4}, 3)
        assert result["ok"]


# ===========================================================================
# 6. Watch / Notification Tests
# ===========================================================================

class TestWatches:
    def test_watch_acquire(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "events": ["acquired"]}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        notifications = sm.drain_notifications()
        assert len(notifications) >= 1
        assert any(n.event == WatchEvent.ACQUIRED for n in notifications)

    def test_watch_release(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "events": ["released"]}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.drain_notifications()  # clear acquire notification
        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 4)
        notifications = sm.drain_notifications()
        assert any(n.event == WatchEvent.RELEASED for n in notifications)

    def test_watch_expired(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "events": ["expired"]}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 5, "now": 0}, 3)
        sm.drain_notifications()
        sm.apply({"op": "expire_check", "now": 5}, 4)
        notifications = sm.drain_notifications()
        assert any(n.event == WatchEvent.EXPIRED for n in notifications)

    def test_unwatch(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1"}, 2)
        sm.apply({"op": "unwatch", "resource": "r1", "session_id": "s1"}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        notifications = sm.drain_notifications()
        assert len(notifications) == 0

    def test_watch_update(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "events": ["acquired"]}, 2)
        result = sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                           "events": ["released"]}, 3)
        assert result.get("updated")

    def test_watch_queued_event(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "events": ["queued"]}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.drain_notifications()
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        notifications = sm.drain_notifications()
        assert any(n.event == WatchEvent.QUEUED for n in notifications)


# ===========================================================================
# 7. Lock Group Tests
# ===========================================================================

class TestLockGroups:
    def test_acquire_group_success(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        result = sm.apply({"op": "acquire_group", "resources": ["r1", "r2", "r3"],
                           "session_id": "s1", "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        assert result["ok"]
        assert result["acquired"]
        assert len(result["fencing_tokens"]) == 3

    def test_acquire_group_partial_fail(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "acquire_group", "resources": ["r1", "r2", "r3"],
                           "session_id": "s2", "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        assert not result["acquired"]
        # r1 should NOT have been acquired (all-or-nothing)
        assert "r1" not in sm.locks or (
            not isinstance(sm.locks.get("r1"), list) and sm.locks.get("r1") and
            sm.locks["r1"].owner != "s2"
        ) or "r1" not in sm.locks

    def test_release_group(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire_group", "resources": ["r1", "r2"],
                  "session_id": "s1", "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "release_group", "resources": ["r1", "r2"],
                           "session_id": "s1", "now": 0}, 3)
        assert result["ok"]
        assert len(result["released"]) == 2

    def test_acquire_group_with_already_held(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "acquire_group", "resources": ["r1", "r2"],
                           "session_id": "s1", "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        assert result["ok"]
        assert result["acquired"]

    def test_acquire_group_invalid_session(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "acquire_group", "resources": ["r1"],
                           "session_id": "nope", "mode": "exclusive", "ttl": 30, "now": 0}, 1)
        assert not result["ok"]


# ===========================================================================
# 8. Deadlock Detection Tests
# ===========================================================================

class TestDeadlockDetection:
    def test_no_deadlock(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "deadlock_check"}, 3)
        assert result["ok"]
        assert len(result["deadlocks"]) == 0

    def test_simple_deadlock(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)

        # s1 holds r1, waits for r2
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        # s2 holds r2, waits for r1
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)

        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 6)

        result = sm.apply({"op": "deadlock_check"}, 7)
        assert len(result["deadlocks"]) > 0

    def test_wait_for_graph(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        result = sm.apply({"op": "deadlock_check"}, 5)
        assert "s2" in result["wait_for_graph"]
        assert "s1" in result["wait_for_graph"]["s2"]


# ===========================================================================
# 9. Info Operations Tests
# ===========================================================================

class TestInfoOperations:
    def test_get_lock_info_free(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "get_lock_info", "resource": "r1"}, 1)
        assert result["ok"]
        assert result["state"] == "free"

    def test_get_lock_info_exclusive(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "get_lock_info", "resource": "r1"}, 3)
        assert result["state"] == "held"
        assert result["mode"] == "exclusive"
        assert result["owner"] == "s1"

    def test_get_lock_info_shared(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "shared", "ttl": 30, "now": 0}, 4)
        result = sm.apply({"op": "get_lock_info", "resource": "r1"}, 5)
        assert result["state"] == "held"
        assert result["mode"] == "shared"
        assert len(result["holders"]) == 2

    def test_get_session_info(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "get_session_info", "session_id": "s1"}, 3)
        assert result["ok"]
        assert result["state"] == "active"
        assert "r1" in result["held_locks"]

    def test_get_session_info_not_found(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "get_session_info", "session_id": "nope"}, 1)
        assert not result["ok"]


# ===========================================================================
# 10. Snapshot / Restore Tests
# ===========================================================================

class TestSnapshotRestore:
    def test_snapshot_and_restore(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)

        snapshot = sm.snapshot()

        sm2 = LockStateMachine()
        sm2.restore(snapshot)

        assert "s1" in sm2.sessions
        assert "r1" in sm2.locks
        assert sm2.locks["r1"].owner == "s1"
        assert sm2.fencing_counter == sm.fencing_counter

    def test_snapshot_shared_locks(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "shared", "ttl": 30, "now": 0}, 4)

        snapshot = sm.snapshot()
        sm2 = LockStateMachine()
        sm2.restore(snapshot)

        assert isinstance(sm2.locks["r1"], list)
        assert len(sm2.locks["r1"]) == 2

    def test_snapshot_wait_queues(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)

        snapshot = sm.snapshot()
        sm2 = LockStateMachine()
        sm2.restore(snapshot)

        assert "r1" in sm2.wait_queues
        assert len(sm2.wait_queues["r1"]) == 1


# ===========================================================================
# 11. Cluster Integration Tests
# ===========================================================================

class TestClusterIntegration:
    def test_cluster_creation(self):
        cluster = make_cluster(3)
        leader = cluster.get_leader()
        assert leader is not None

    def test_client_session(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        assert client.session_id is not None

    def test_client_acquire_release(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        result = client.acquire("resource1")
        assert result["ok"]
        assert "fencing_token" in result

        result = client.release("resource1")
        assert result["ok"]

    def test_client_try_acquire(self):
        cluster = make_cluster(3)
        c1 = make_client(cluster, "c1")
        c2 = make_client(cluster, "c2")

        r1 = c1.acquire("resource1")
        assert r1["ok"]

        r2 = c2.try_acquire("resource1")
        assert r2["ok"]
        assert not r2["acquired"]

    def test_client_contention(self):
        cluster = make_cluster(3)
        c1 = make_client(cluster, "c1")
        c2 = make_client(cluster, "c2")

        c1.acquire("resource1")
        r2 = c2.acquire("resource1")
        assert r2.get("queued")

        c1.release("resource1")
        # After release, the queued client should get the lock
        # Check via lock info
        info = c1.get_lock_info("resource1")
        assert info["state"] == "held"
        assert info["owner"] == c2.session_id

    def test_client_heartbeat(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1", ttl=10, now=0)
        result = client.heartbeat(now=8)
        assert result["ok"]

    def test_client_destroy_session(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        client.acquire("resource1")
        result = client.destroy_session()
        assert result["ok"]
        assert "resource1" in result.get("released_locks", [])

    def test_client_lock_group(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        result = client.acquire_group(["r1", "r2", "r3"])
        assert result["ok"]
        assert result["acquired"]

        result = client.release_group(["r1", "r2", "r3"])
        assert result["ok"]

    def test_client_watch(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        result = client.watch("resource1")
        assert result["ok"]

    def test_client_expire_check(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1", ttl=10, now=0)
        client.acquire("resource1", ttl=5, now=0)
        result = client.expire_check(now=5)
        assert result["ok"]
        assert "resource1" in result["expired_locks"]

    def test_client_deadlock_check(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        result = client.deadlock_check()
        assert result["ok"]
        assert len(result["deadlocks"]) == 0

    def test_client_get_session_info(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        result = client.get_session_info()
        assert result["ok"]
        assert result["state"] == "active"

    def test_client_no_session_errors(self):
        cluster = make_cluster(3)
        client = LockClient(cluster, "c1")
        # No session created
        assert not client.acquire("r1")["ok"]
        assert not client.release("r1")["ok"]
        assert not client.try_acquire("r1")["ok"]
        assert not client.heartbeat()["ok"]
        assert not client.destroy_session()["ok"]
        assert not client.renew("r1")["ok"]
        assert not client.acquire_group(["r1"])["ok"]
        assert not client.release_group(["r1"])["ok"]
        assert not client.watch("r1")["ok"]
        assert not client.unwatch("r1")["ok"]

    def test_client_renew(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        client.acquire("resource1", ttl=5, now=0)
        result = client.renew("resource1", ttl=20, now=4)
        assert result["ok"]
        assert result["expires_at"] == 24


# ===========================================================================
# 12. Leader Election / Partition Tests
# ===========================================================================

class TestLeaderPartition:
    def test_partition_and_heal(self):
        cluster = make_cluster(5)
        client = make_client(cluster, "c1")
        client.acquire("resource1")

        old_leader = cluster.get_leader()
        assert old_leader is not None

        # Partition off the leader
        majority = [n for n in cluster.node_ids if n != old_leader]
        cluster.partition([old_leader], majority)

        # After enough ticks, old leader steps down (loses heartbeat responses)
        # and majority elects new leader
        for _ in range(5000):
            cluster.tick()
            cluster.deliver_all()

        # Heal and wait for stable state
        cluster.heal_partition()
        for _ in range(3000):
            cluster.tick()
            cluster.deliver_all()

        leader = cluster.get_leader()
        assert leader is not None

    def test_heal_partition_replicas_converge(self):
        cluster = make_cluster(3)
        client = make_client(cluster, "c1")
        client.acquire("resource1")

        # Tick to ensure replication
        for _ in range(200):
            cluster.tick()
            cluster.deliver_all()

        # Leader's state machine should have the lock
        sm = cluster.get_state_machine()
        assert sm is not None
        assert "resource1" in sm.locks


# ===========================================================================
# 13. Stats Tests
# ===========================================================================

class TestStats:
    def test_basic_stats(self):
        cluster = make_cluster(3)
        stats = LockServiceStats(cluster)
        client = make_client(cluster, "c1")
        client.acquire("resource1")
        client.acquire("resource2")

        s = stats.get_stats()
        assert s["active_sessions"] == 1
        assert s["total_locks"] == 2
        assert s["exclusive_locks"] == 2
        assert s["shared_locks"] == 0

    def test_shared_stats(self):
        cluster = make_cluster(3)
        stats = LockServiceStats(cluster)
        c1 = make_client(cluster, "c1")
        c2 = make_client(cluster, "c2")
        c1.acquire("resource1", mode="shared")
        c2.acquire("resource1", mode="shared")

        s = stats.get_stats()
        assert s["shared_locks"] == 2
        assert s["exclusive_locks"] == 0

    def test_waiter_stats(self):
        cluster = make_cluster(3)
        stats = LockServiceStats(cluster)
        c1 = make_client(cluster, "c1")
        c2 = make_client(cluster, "c2")
        c1.acquire("resource1")
        c2.acquire("resource1")

        s = stats.get_stats()
        assert s["total_waiters"] == 1


# ===========================================================================
# 14. Session Lifecycle Tests
# ===========================================================================

class TestSessionLifecycle:
    def test_session_tracks_locks(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        assert "r1" in sm.sessions["s1"].held_locks
        assert "r2" in sm.sessions["s1"].held_locks

    def test_release_updates_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 3)
        assert "r1" not in sm.sessions["s1"].held_locks

    def test_destroy_releases_all(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "destroy_session", "session_id": "s1", "now": 0}, 4)
        assert "r1" in result["released_locks"]
        assert "r2" in result["released_locks"]
        assert "r1" not in sm.locks
        assert "r2" not in sm.locks

    def test_session_expiry_releases_locks(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 100, "now": 0}, 2)
        sm.apply({"op": "expire_check", "now": 10}, 3)
        assert "r1" not in sm.locks

    def test_session_metadata(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0,
                  "metadata": {"app": "myservice"}}, 1)
        assert sm.sessions["s1"].metadata["app"] == "myservice"

    def test_recreate_expired_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 10, "now": 0}, 1)
        sm.apply({"op": "expire_check", "now": 10}, 2)
        # Session expired, can recreate
        result = sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 10}, 3)
        assert result["ok"]
        assert sm.sessions["s1"].state == SessionState.ACTIVE


# ===========================================================================
# 15. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_acquire_after_session_destroy_removes_from_queue(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        # Destroy s2 while it's waiting
        sm.apply({"op": "destroy_session", "session_id": "s2", "now": 0}, 5)
        assert "r1" not in sm.wait_queues or len(sm.wait_queues.get("r1", [])) == 0

    def test_multiple_resources_independent(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        assert sm.locks["r1"].owner == "s1"
        assert sm.locks["r2"].owner == "s2"

    def test_shared_lock_expire_partial(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 100, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 100, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 5, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "shared", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "expire_check", "now": 5}, 5)
        # s1's lock expired, s2's remains
        assert isinstance(sm.locks["r1"], list)
        assert len(sm.locks["r1"]) == 1
        assert sm.locks["r1"][0].owner == "s2"

    def test_fencing_token_monotonic_across_resources(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        tokens = []
        for i in range(5):
            r = sm.apply({"op": "acquire", "resource": f"r{i}", "session_id": "s1",
                          "mode": "exclusive", "ttl": 30, "now": 0}, i + 2)
            tokens.append(r["fencing_token"])
        # Tokens should be strictly increasing
        for i in range(1, len(tokens)):
            assert tokens[i] > tokens[i - 1]

    def test_release_shared_not_held_by_session(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "shared", "ttl": 30, "now": 0}, 3)
        result = sm.apply({"op": "release", "resource": "r1", "session_id": "s2", "now": 0}, 4)
        assert not result["ok"]

    def test_release_invalid_session(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "release", "resource": "r1", "session_id": "bad", "now": 0}, 1)
        assert not result["ok"]

    def test_try_acquire_invalid_session(self):
        sm = LockStateMachine()
        result = sm.apply({"op": "try_acquire", "resource": "r1", "session_id": "bad",
                           "mode": "exclusive", "ttl": 30, "now": 0}, 1)
        assert not result["ok"]

    def test_lock_entry_properties(self):
        token = FencingToken(1, "r1", "s1", 1)
        entry = LockEntry("r1", LockMode.EXCLUSIVE, "s1", token, 10.0, 5.0)
        assert entry.expires_at == 15.0
        assert not entry.is_expired(14.9)
        assert entry.is_expired(15.0)

    def test_session_properties(self):
        sess = Session("s1", 0, 10, 5.0)
        assert sess.expires_at == 15.0
        assert not sess.is_expired(14.9)
        assert sess.is_expired(15.0)

    def test_unwatch_nonexistent(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        result = sm.apply({"op": "unwatch", "resource": "r1", "session_id": "s1"}, 2)
        assert result["ok"]

    def test_unwatch_by_callback_id(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "callback_id": "cb1"}, 2)
        sm.apply({"op": "watch", "resource": "r1", "session_id": "s1",
                  "callback_id": "cb2"}, 3)
        sm.apply({"op": "unwatch", "resource": "r1", "session_id": "s1",
                  "callback_id": "cb1"}, 4)
        assert len(sm.watches["r1"]) == 1
        assert sm.watches["r1"][0].callback_id == "cb2"


# ===========================================================================
# 16. Lock Service Node Tests
# ===========================================================================

class TestLockServiceNode:
    def test_node_status(self):
        from distributed_lock import LockServiceNode
        node = LockServiceNode("n1", ["n2", "n3"])
        status = node.status()
        assert "lock_count" in status
        assert "session_count" in status
        assert status["lock_count"] == 0

    def test_node_is_leader(self):
        from distributed_lock import LockServiceNode
        node = LockServiceNode("n1", ["n2", "n3"])
        # Initially follower
        assert not node.is_leader


# ===========================================================================
# 17. Complex Scenarios
# ===========================================================================

class TestComplexScenarios:
    def test_three_way_deadlock_detection(self):
        sm = LockStateMachine()
        for i in range(1, 4):
            sm.apply({"op": "create_session", "session_id": f"s{i}", "ttl": 30, "now": 0}, i)

        # s1 holds r1, s2 holds r2, s3 holds r3
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        sm.apply({"op": "acquire", "resource": "r3", "session_id": "s3",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 6)

        # s1 wants r2, s2 wants r3, s3 wants r1 -> cycle
        sm.apply({"op": "acquire", "resource": "r2", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 7)
        sm.apply({"op": "acquire", "resource": "r3", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 8)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s3",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 9)

        result = sm.apply({"op": "deadlock_check"}, 10)
        assert len(result["deadlocks"]) > 0

    def test_cascade_promotion(self):
        """When a lock is released, promote first exclusive waiter, blocking subsequent ones."""
        sm = LockStateMachine()
        for i in range(1, 5):
            sm.apply({"op": "create_session", "session_id": f"s{i}", "ttl": 30, "now": 0}, i)

        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        # s2, s3, s4 all want exclusive
        for i in range(2, 5):
            sm.apply({"op": "acquire", "resource": "r1", "session_id": f"s{i}",
                      "mode": "exclusive", "ttl": 30, "now": 0}, 5 + i)

        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 10)
        # s2 gets promoted, s3 and s4 still waiting
        assert sm.locks["r1"].owner == "s2"
        assert len(sm.wait_queues.get("r1", [])) == 2

    def test_mixed_shared_exclusive_queue(self):
        """Shared waiters after exclusive should not jump ahead."""
        sm = LockStateMachine()
        for i in range(1, 5):
            sm.apply({"op": "create_session", "session_id": f"s{i}", "ttl": 30, "now": 0}, i)

        # s1 holds exclusive
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 5)
        # s2 wants exclusive, s3 and s4 want shared
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 6)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s3",
                  "mode": "shared", "ttl": 30, "now": 0}, 7)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s4",
                  "mode": "shared", "ttl": 30, "now": 0}, 8)

        # Release s1 -- s2 (exclusive) should get it first
        sm.apply({"op": "release", "resource": "r1", "session_id": "s1", "now": 0}, 9)
        assert not isinstance(sm.locks["r1"], list)
        assert sm.locks["r1"].owner == "s2"

    def test_lock_metadata(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0,
                  "metadata": {"purpose": "migration"}}, 2)
        assert sm.locks["r1"].metadata["purpose"] == "migration"

    def test_many_sessions_many_locks(self):
        sm = LockStateMachine()
        n = 20
        for i in range(n):
            sm.apply({"op": "create_session", "session_id": f"s{i}", "ttl": 30, "now": 0}, i + 1)
        for i in range(n):
            sm.apply({"op": "acquire", "resource": f"r{i}", "session_id": f"s{i}",
                      "mode": "exclusive", "ttl": 30, "now": 0}, n + i + 1)
        assert len(sm.locks) == n
        assert sm.fencing_counter == n

    def test_session_destroy_promotes_waiters(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "create_session", "session_id": "s2", "ttl": 30, "now": 0}, 2)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 3)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s2",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 4)
        sm.apply({"op": "destroy_session", "session_id": "s1", "now": 0}, 5)
        # s2 should now hold the lock
        assert sm.locks["r1"].owner == "s2"

    def test_release_group_partial_errors(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        result = sm.apply({"op": "release_group", "resources": ["r1", "r2"],
                           "session_id": "s1", "now": 0}, 3)
        assert "r1" in result["released"]
        assert len(result["errors"]) == 1

    def test_fencing_tokens_survive_snapshot(self):
        sm = LockStateMachine()
        sm.apply({"op": "create_session", "session_id": "s1", "ttl": 30, "now": 0}, 1)
        sm.apply({"op": "acquire", "resource": "r1", "session_id": "s1",
                  "mode": "exclusive", "ttl": 30, "now": 0}, 2)
        token_before = sm.locks["r1"].fencing_token.value

        snap = sm.snapshot()
        sm2 = LockStateMachine()
        sm2.restore(snap)

        assert sm2.locks["r1"].fencing_token.value == token_before
        assert sm2.fencing_counter == sm.fencing_counter


# ===========================================================================
# 18. Data Type Tests
# ===========================================================================

class TestDataTypes:
    def test_lock_mode_enum(self):
        assert LockMode.EXCLUSIVE.value == "exclusive"
        assert LockMode.SHARED.value == "shared"

    def test_lock_state_enum(self):
        assert LockState.FREE.value == "free"
        assert LockState.HELD.value == "held"

    def test_session_state_enum(self):
        assert SessionState.ACTIVE.value == "active"
        assert SessionState.EXPIRED.value == "expired"

    def test_watch_event_enum(self):
        assert WatchEvent.ACQUIRED.value == "acquired"
        assert WatchEvent.RELEASED.value == "released"
        assert WatchEvent.EXPIRED.value == "expired"
        assert WatchEvent.QUEUED.value == "queued"

    def test_notification_dataclass(self):
        n = Notification("r1", WatchEvent.ACQUIRED, {"owner": "s1"}, 1.0)
        assert n.resource == "r1"
        assert n.event == WatchEvent.ACQUIRED

    def test_wait_entry_dataclass(self):
        w = WaitEntry("r1", LockMode.EXCLUSIVE, "s1", 0.0, 30.0)
        assert w.resource == "r1"
        assert w.session_id == "s1"

    def test_watch_registration_dataclass(self):
        w = WatchRegistration("r1", "s1", [WatchEvent.ACQUIRED], "cb1")
        assert w.resource == "r1"
        assert w.callback_id == "cb1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
