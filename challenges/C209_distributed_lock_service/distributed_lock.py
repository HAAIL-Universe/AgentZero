"""
C209: Distributed Lock Service
Composing C201 (Raft Consensus) + C206 (Distributed KV Store)

A Chubby/ZooKeeper-inspired distributed lock service with:
- Exclusive and shared (read/write) locks
- Fencing tokens for safe resource access
- Session-based leases with TTL and heartbeats
- Lock queuing (fair ordering)
- Distributed deadlock detection (wait-for graph cycle detection)
- Lock groups (atomic multi-resource locking)
- Watch/notification system for lock state changes
- Leader-based lock management via Raft
"""

import sys
import os
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C201_raft_consensus'))
from raft import RaftNode, RaftCluster, RaftLog, LogEntry


# ---------------------------------------------------------------------------
# Enums & Data Types
# ---------------------------------------------------------------------------

class LockMode(Enum):
    EXCLUSIVE = "exclusive"
    SHARED = "shared"


class LockState(Enum):
    FREE = "free"
    HELD = "held"
    WAITING = "waiting"


class SessionState(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"


class WatchEvent(Enum):
    ACQUIRED = "acquired"
    RELEASED = "released"
    EXPIRED = "expired"
    QUEUED = "queued"


@dataclass
class FencingToken:
    """Monotonically increasing token for safe resource access."""
    value: int
    resource: str
    owner: str
    epoch: int  # Raft term when issued

    def __gt__(self, other):
        if not isinstance(other, FencingToken):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other):
        if not isinstance(other, FencingToken):
            return NotImplemented
        return self.value >= other.value


@dataclass
class LockEntry:
    """A single lock on a resource."""
    resource: str
    mode: LockMode
    owner: str  # session_id
    fencing_token: FencingToken
    acquired_at: float
    ttl: float  # seconds
    metadata: dict = field(default_factory=dict)

    @property
    def expires_at(self):
        return self.acquired_at + self.ttl

    def is_expired(self, now):
        return now >= self.expires_at


@dataclass
class WaitEntry:
    """A queued lock request."""
    resource: str
    mode: LockMode
    session_id: str
    requested_at: float
    ttl: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    """Client session with lease-based TTL."""
    session_id: str
    created_at: float
    last_heartbeat: float
    ttl: float  # seconds
    state: SessionState = SessionState.ACTIVE
    held_locks: list = field(default_factory=list)  # resource names
    metadata: dict = field(default_factory=dict)

    @property
    def expires_at(self):
        return self.last_heartbeat + self.ttl

    def is_expired(self, now):
        return now >= self.expires_at


@dataclass
class WatchRegistration:
    """Watch on a resource for state changes."""
    resource: str
    session_id: str
    events: list  # list of WatchEvent types to watch for
    callback_id: str  # identifier for the callback


@dataclass
class Notification:
    """Event notification for watches."""
    resource: str
    event: WatchEvent
    details: dict
    timestamp: float


# ---------------------------------------------------------------------------
# Lock State Machine (for Raft)
# ---------------------------------------------------------------------------

class LockStateMachine:
    """Raft state machine managing distributed lock state."""

    def __init__(self):
        self.locks = {}          # resource -> LockEntry or list[LockEntry] for shared
        self.wait_queues = {}    # resource -> [WaitEntry]
        self.sessions = {}       # session_id -> Session
        self.watches = {}        # resource -> [WatchRegistration]
        self.notifications = []  # pending notifications
        self.fencing_counter = 0
        self.current_term = 0

    def apply(self, command, index):
        """Apply a command from the Raft log."""
        if command is None:
            return {"ok": True}

        op = command.get("op")
        if op == "create_session":
            return self._create_session(command)
        elif op == "heartbeat":
            return self._heartbeat(command)
        elif op == "destroy_session":
            return self._destroy_session(command)
        elif op == "acquire":
            return self._acquire(command)
        elif op == "release":
            return self._release(command)
        elif op == "try_acquire":
            return self._try_acquire(command)
        elif op == "renew":
            return self._renew(command)
        elif op == "expire_check":
            return self._expire_check(command)
        elif op == "watch":
            return self._watch(command)
        elif op == "unwatch":
            return self._unwatch(command)
        elif op == "acquire_group":
            return self._acquire_group(command)
        elif op == "release_group":
            return self._release_group(command)
        elif op == "get_lock_info":
            return self._get_lock_info(command)
        elif op == "get_session_info":
            return self._get_session_info(command)
        elif op == "deadlock_check":
            return self._deadlock_check(command)
        else:
            return {"ok": False, "error": f"unknown op: {op}"}

    def snapshot(self):
        """Snapshot state for Raft compaction."""
        return {
            "locks": {r: self._serialize_lock(l) for r, l in self.locks.items()},
            "wait_queues": {r: [self._serialize_wait(w) for w in q] for r, q in self.wait_queues.items()},
            "sessions": {s: self._serialize_session(sess) for s, sess in self.sessions.items()},
            "fencing_counter": self.fencing_counter,
            "current_term": self.current_term,
        }

    def restore(self, data):
        """Restore from snapshot."""
        self.fencing_counter = data["fencing_counter"]
        self.current_term = data["current_term"]
        self.locks = {}
        for r, l in data["locks"].items():
            self.locks[r] = self._deserialize_lock(l)
        self.wait_queues = {}
        for r, q in data["wait_queues"].items():
            self.wait_queues[r] = [self._deserialize_wait(w) for w in q]
        self.sessions = {}
        for s, sess in data["sessions"].items():
            self.sessions[s] = self._deserialize_session(sess)

    # -- Session operations --

    def _create_session(self, cmd):
        sid = cmd["session_id"]
        if sid in self.sessions and self.sessions[sid].state == SessionState.ACTIVE:
            return {"ok": True, "session_id": sid, "already_exists": True}
        now = cmd.get("now", 0)
        ttl = cmd.get("ttl", 30.0)
        self.sessions[sid] = Session(
            session_id=sid,
            created_at=now,
            last_heartbeat=now,
            ttl=ttl,
            metadata=cmd.get("metadata", {}),
        )
        return {"ok": True, "session_id": sid}

    def _heartbeat(self, cmd):
        sid = cmd["session_id"]
        sess = self.sessions.get(sid)
        if not sess or sess.state != SessionState.ACTIVE:
            return {"ok": False, "error": "session not found or expired"}
        now = cmd.get("now", 0)
        sess.last_heartbeat = now
        return {"ok": True, "session_id": sid, "expires_at": sess.expires_at}

    def _destroy_session(self, cmd):
        sid = cmd["session_id"]
        sess = self.sessions.get(sid)
        if not sess:
            return {"ok": False, "error": "session not found"}
        # Release all locks held by this session
        released = self._release_session_locks(sid, cmd.get("now", 0))
        sess.state = SessionState.EXPIRED
        return {"ok": True, "session_id": sid, "released_locks": released}

    def _release_session_locks(self, session_id, now):
        """Release all locks held by a session."""
        released = []
        resources_to_check = list(self.locks.keys())
        for resource in resources_to_check:
            lock = self.locks[resource]
            if isinstance(lock, list):
                # Shared locks
                remaining = [l for l in lock if l.owner != session_id]
                removed = [l for l in lock if l.owner == session_id]
                if removed:
                    released.extend([r.resource for r in removed])
                    if remaining:
                        self.locks[resource] = remaining
                    else:
                        del self.locks[resource]
                        self._notify(resource, WatchEvent.RELEASED, {"session_id": session_id}, now)
                        self._promote_waiters(resource, now)
            else:
                if lock.owner == session_id:
                    released.append(resource)
                    del self.locks[resource]
                    self._notify(resource, WatchEvent.RELEASED, {"session_id": session_id}, now)
                    self._promote_waiters(resource, now)

        # Remove from wait queues too
        for resource in list(self.wait_queues.keys()):
            self.wait_queues[resource] = [
                w for w in self.wait_queues[resource] if w.session_id != session_id
            ]
            if not self.wait_queues[resource]:
                del self.wait_queues[resource]

        # Update session
        if session_id in self.sessions:
            self.sessions[session_id].held_locks = []

        return released

    # -- Lock operations --

    def _acquire(self, cmd):
        resource = cmd["resource"]
        session_id = cmd["session_id"]
        mode = LockMode(cmd.get("mode", "exclusive"))
        now = cmd.get("now", 0)
        ttl = cmd.get("ttl", 30.0)
        metadata = cmd.get("metadata", {})

        # Validate session
        sess = self.sessions.get(session_id)
        if not sess or sess.state != SessionState.ACTIVE:
            return {"ok": False, "error": "invalid session"}

        # Check if already held by this session
        existing = self.locks.get(resource)
        if existing is not None:
            if isinstance(existing, list):
                for l in existing:
                    if l.owner == session_id:
                        return {"ok": True, "already_held": True,
                                "fencing_token": l.fencing_token.value}
            elif existing.owner == session_id:
                return {"ok": True, "already_held": True,
                        "fencing_token": existing.fencing_token.value}

        # Try to acquire
        acquired = self._try_lock(resource, mode, session_id, now, ttl, metadata)
        if acquired:
            token = acquired.fencing_token
            return {"ok": True, "fencing_token": token.value,
                    "resource": resource, "mode": mode.value}

        # Queue the request
        wait = WaitEntry(resource, mode, session_id, now, ttl, metadata)
        if resource not in self.wait_queues:
            self.wait_queues[resource] = []
        # Don't double-queue
        if not any(w.session_id == session_id for w in self.wait_queues[resource]):
            self.wait_queues[resource].append(wait)
            self._notify(resource, WatchEvent.QUEUED, {"session_id": session_id}, now)
        return {"ok": True, "queued": True, "resource": resource,
                "position": len(self.wait_queues[resource])}

    def _try_acquire(self, cmd):
        """Non-blocking lock attempt -- no queuing."""
        resource = cmd["resource"]
        session_id = cmd["session_id"]
        mode = LockMode(cmd.get("mode", "exclusive"))
        now = cmd.get("now", 0)
        ttl = cmd.get("ttl", 30.0)
        metadata = cmd.get("metadata", {})

        sess = self.sessions.get(session_id)
        if not sess or sess.state != SessionState.ACTIVE:
            return {"ok": False, "error": "invalid session"}

        # Check if already held
        existing = self.locks.get(resource)
        if existing is not None:
            if isinstance(existing, list):
                for l in existing:
                    if l.owner == session_id:
                        return {"ok": True, "acquired": True, "already_held": True,
                                "fencing_token": l.fencing_token.value}
            elif existing.owner == session_id:
                return {"ok": True, "acquired": True, "already_held": True,
                        "fencing_token": existing.fencing_token.value}

        acquired = self._try_lock(resource, mode, session_id, now, ttl, metadata)
        if acquired:
            return {"ok": True, "acquired": True,
                    "fencing_token": acquired.fencing_token.value}
        return {"ok": True, "acquired": False, "resource": resource}

    def _try_lock(self, resource, mode, session_id, now, ttl, metadata):
        """Attempt to acquire a lock, return LockEntry or None."""
        existing = self.locks.get(resource)

        if existing is None:
            # Free -- acquire
            entry = self._make_lock_entry(resource, mode, session_id, now, ttl, metadata)
            if mode == LockMode.SHARED:
                self.locks[resource] = [entry]
            else:
                self.locks[resource] = entry
            self._track_session_lock(session_id, resource)
            self._notify(resource, WatchEvent.ACQUIRED, {
                "session_id": session_id, "mode": mode.value,
                "fencing_token": entry.fencing_token.value
            }, now)
            return entry

        if isinstance(existing, list):
            # Currently shared locks
            if mode == LockMode.SHARED:
                # Can add another shared lock
                entry = self._make_lock_entry(resource, mode, session_id, now, ttl, metadata)
                existing.append(entry)
                self._track_session_lock(session_id, resource)
                self._notify(resource, WatchEvent.ACQUIRED, {
                    "session_id": session_id, "mode": mode.value,
                    "fencing_token": entry.fencing_token.value
                }, now)
                return entry
            else:
                # Want exclusive but shared locks exist
                return None
        else:
            # Currently exclusive lock held by someone else
            return None

    def _make_lock_entry(self, resource, mode, session_id, now, ttl, metadata):
        self.fencing_counter += 1
        token = FencingToken(
            value=self.fencing_counter,
            resource=resource,
            owner=session_id,
            epoch=self.current_term,
        )
        return LockEntry(
            resource=resource,
            mode=mode,
            owner=session_id,
            fencing_token=token,
            acquired_at=now,
            ttl=ttl,
            metadata=metadata,
        )

    def _track_session_lock(self, session_id, resource):
        sess = self.sessions.get(session_id)
        if sess and resource not in sess.held_locks:
            sess.held_locks.append(resource)

    def _release(self, cmd):
        resource = cmd["resource"]
        session_id = cmd["session_id"]
        now = cmd.get("now", 0)

        sess = self.sessions.get(session_id)
        if not sess or sess.state != SessionState.ACTIVE:
            return {"ok": False, "error": "invalid session"}

        existing = self.locks.get(resource)
        if existing is None:
            return {"ok": False, "error": "lock not held"}

        if isinstance(existing, list):
            found = [l for l in existing if l.owner == session_id]
            if not found:
                return {"ok": False, "error": "lock not held by session"}
            remaining = [l for l in existing if l.owner != session_id]
            if remaining:
                self.locks[resource] = remaining
            else:
                del self.locks[resource]
                self._notify(resource, WatchEvent.RELEASED, {"session_id": session_id}, now)
                self._promote_waiters(resource, now)
        else:
            if existing.owner != session_id:
                return {"ok": False, "error": "lock not held by session"}
            del self.locks[resource]
            self._notify(resource, WatchEvent.RELEASED, {"session_id": session_id}, now)
            self._promote_waiters(resource, now)

        if resource in sess.held_locks:
            sess.held_locks.remove(resource)

        return {"ok": True, "resource": resource}

    def _renew(self, cmd):
        """Renew a lock's TTL."""
        resource = cmd["resource"]
        session_id = cmd["session_id"]
        now = cmd.get("now", 0)
        new_ttl = cmd.get("ttl", 30.0)

        existing = self.locks.get(resource)
        if existing is None:
            return {"ok": False, "error": "lock not held"}

        if isinstance(existing, list):
            for l in existing:
                if l.owner == session_id:
                    l.acquired_at = now
                    l.ttl = new_ttl
                    return {"ok": True, "expires_at": l.expires_at}
            return {"ok": False, "error": "lock not held by session"}
        else:
            if existing.owner != session_id:
                return {"ok": False, "error": "lock not held by session"}
            existing.acquired_at = now
            existing.ttl = new_ttl
            return {"ok": True, "expires_at": existing.expires_at}

    def _promote_waiters(self, resource, now):
        """Try to promote queued waiters after a lock release."""
        if resource not in self.wait_queues or not self.wait_queues[resource]:
            return

        promoted = []
        remaining = []
        for waiter in self.wait_queues[resource]:
            # Check if waiter's session is still active
            sess = self.sessions.get(waiter.session_id)
            if not sess or sess.state != SessionState.ACTIVE:
                continue

            result = self._try_lock(
                resource, waiter.mode, waiter.session_id, now, waiter.ttl, waiter.metadata
            )
            if result:
                promoted.append(waiter.session_id)
                # If we promoted an exclusive lock, stop
                if waiter.mode == LockMode.EXCLUSIVE:
                    remaining = self.wait_queues[resource][self.wait_queues[resource].index(waiter) + 1:]
                    break
            else:
                remaining.append(waiter)

        if remaining:
            self.wait_queues[resource] = remaining
        elif resource in self.wait_queues:
            del self.wait_queues[resource]

    # -- Expiration --

    def _expire_check(self, cmd):
        """Check and expire locks and sessions."""
        now = cmd.get("now", 0)
        expired_locks = []
        expired_sessions = []

        # Expire locks
        for resource in list(self.locks.keys()):
            lock = self.locks[resource]
            if isinstance(lock, list):
                remaining = []
                for l in lock:
                    if l.is_expired(now):
                        expired_locks.append(resource)
                        self._untrack_session_lock(l.owner, resource)
                    else:
                        remaining.append(l)
                if remaining:
                    self.locks[resource] = remaining
                else:
                    del self.locks[resource]
                    self._notify(resource, WatchEvent.EXPIRED, {}, now)
                    self._promote_waiters(resource, now)
            else:
                if lock.is_expired(now):
                    expired_locks.append(resource)
                    self._untrack_session_lock(lock.owner, resource)
                    del self.locks[resource]
                    self._notify(resource, WatchEvent.EXPIRED, {}, now)
                    self._promote_waiters(resource, now)

        # Expire sessions
        for sid, sess in list(self.sessions.items()):
            if sess.state == SessionState.ACTIVE and sess.is_expired(now):
                released = self._release_session_locks(sid, now)
                sess.state = SessionState.EXPIRED
                expired_sessions.append(sid)

        return {"ok": True, "expired_locks": expired_locks,
                "expired_sessions": expired_sessions}

    def _untrack_session_lock(self, session_id, resource):
        sess = self.sessions.get(session_id)
        if sess and resource in sess.held_locks:
            sess.held_locks.remove(resource)

    # -- Watch operations --

    def _watch(self, cmd):
        resource = cmd["resource"]
        session_id = cmd["session_id"]
        events = [WatchEvent(e) for e in cmd.get("events", ["acquired", "released", "expired"])]
        callback_id = cmd.get("callback_id", f"watch_{session_id}_{resource}")

        if resource not in self.watches:
            self.watches[resource] = []

        # Don't duplicate
        for w in self.watches[resource]:
            if w.session_id == session_id and w.callback_id == callback_id:
                w.events = events
                return {"ok": True, "updated": True}

        self.watches[resource].append(WatchRegistration(
            resource=resource, session_id=session_id,
            events=events, callback_id=callback_id
        ))
        return {"ok": True, "callback_id": callback_id}

    def _unwatch(self, cmd):
        resource = cmd["resource"]
        session_id = cmd["session_id"]
        callback_id = cmd.get("callback_id")

        if resource not in self.watches:
            return {"ok": True}

        if callback_id:
            self.watches[resource] = [
                w for w in self.watches[resource]
                if not (w.session_id == session_id and w.callback_id == callback_id)
            ]
        else:
            self.watches[resource] = [
                w for w in self.watches[resource]
                if w.session_id != session_id
            ]

        if not self.watches[resource]:
            del self.watches[resource]
        return {"ok": True}

    def _notify(self, resource, event, details, timestamp):
        """Create notifications for watchers."""
        if resource not in self.watches:
            return
        for w in self.watches[resource]:
            if event in w.events:
                self.notifications.append(Notification(
                    resource=resource, event=event,
                    details=details, timestamp=timestamp
                ))

    def drain_notifications(self):
        """Drain and return pending notifications."""
        notifications = self.notifications[:]
        self.notifications = []
        return notifications

    # -- Lock groups (atomic multi-lock) --

    def _acquire_group(self, cmd):
        """Atomically acquire locks on multiple resources."""
        resources = cmd["resources"]
        session_id = cmd["session_id"]
        mode = LockMode(cmd.get("mode", "exclusive"))
        now = cmd.get("now", 0)
        ttl = cmd.get("ttl", 30.0)

        sess = self.sessions.get(session_id)
        if not sess or sess.state != SessionState.ACTIVE:
            return {"ok": False, "error": "invalid session"}

        # Sort resources to prevent deadlocks (canonical ordering)
        sorted_resources = sorted(resources)

        # Check all can be acquired (all-or-nothing)
        can_acquire = True
        for resource in sorted_resources:
            existing = self.locks.get(resource)
            if existing is not None:
                if isinstance(existing, list):
                    if any(l.owner == session_id for l in existing):
                        continue  # already held
                    if mode == LockMode.EXCLUSIVE:
                        can_acquire = False
                        break
                elif existing.owner == session_id:
                    continue  # already held
                else:
                    can_acquire = False
                    break

        if not can_acquire:
            return {"ok": True, "acquired": False,
                    "blocked_by": resource}

        # Acquire all
        tokens = {}
        for resource in sorted_resources:
            existing = self.locks.get(resource)
            if existing is not None:
                if isinstance(existing, list):
                    held = [l for l in existing if l.owner == session_id]
                    if held:
                        tokens[resource] = held[0].fencing_token.value
                        continue
                elif existing.owner == session_id:
                    tokens[resource] = existing.fencing_token.value
                    continue

            entry = self._try_lock(resource, mode, session_id, now, ttl, {})
            if entry:
                tokens[resource] = entry.fencing_token.value

        return {"ok": True, "acquired": True, "fencing_tokens": tokens}

    def _release_group(self, cmd):
        """Release locks on multiple resources."""
        resources = cmd["resources"]
        session_id = cmd["session_id"]
        now = cmd.get("now", 0)

        released = []
        errors = []
        for resource in resources:
            result = self._release({"resource": resource, "session_id": session_id, "now": now})
            if result.get("ok"):
                released.append(resource)
            else:
                errors.append({"resource": resource, "error": result.get("error")})

        return {"ok": True, "released": released, "errors": errors}

    # -- Info operations --

    def _get_lock_info(self, cmd):
        resource = cmd["resource"]
        existing = self.locks.get(resource)
        if existing is None:
            waiters = self.wait_queues.get(resource, [])
            return {"ok": True, "state": "free", "resource": resource,
                    "waiters": len(waiters)}

        if isinstance(existing, list):
            holders = [{"session_id": l.owner, "mode": l.mode.value,
                        "fencing_token": l.fencing_token.value,
                        "expires_at": l.expires_at} for l in existing]
            return {"ok": True, "state": "held", "mode": "shared",
                    "holders": holders, "resource": resource,
                    "waiters": len(self.wait_queues.get(resource, []))}
        else:
            return {"ok": True, "state": "held", "mode": "exclusive",
                    "owner": existing.owner,
                    "fencing_token": existing.fencing_token.value,
                    "expires_at": existing.expires_at,
                    "resource": resource,
                    "waiters": len(self.wait_queues.get(resource, []))}

    def _get_session_info(self, cmd):
        sid = cmd["session_id"]
        sess = self.sessions.get(sid)
        if not sess:
            return {"ok": False, "error": "session not found"}
        return {
            "ok": True,
            "session_id": sid,
            "state": sess.state.value,
            "created_at": sess.created_at,
            "last_heartbeat": sess.last_heartbeat,
            "expires_at": sess.expires_at,
            "ttl": sess.ttl,
            "held_locks": sess.held_locks[:],
            "metadata": sess.metadata,
        }

    # -- Deadlock detection --

    def _deadlock_check(self, cmd):
        """Detect deadlocks via wait-for graph cycle detection."""
        # Build wait-for graph: session -> set of sessions it waits for
        wait_for = {}

        for resource, waiters in self.wait_queues.items():
            # Who holds this lock?
            holders = set()
            lock = self.locks.get(resource)
            if lock is None:
                continue
            if isinstance(lock, list):
                for l in lock:
                    holders.add(l.owner)
            else:
                holders.add(lock.owner)

            # Each waiter waits for the holders
            for waiter in waiters:
                if waiter.session_id not in wait_for:
                    wait_for[waiter.session_id] = set()
                wait_for[waiter.session_id].update(holders)

        # DFS cycle detection
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node):
            if node in path_set:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            if node in visited:
                return
            visited.add(node)
            path.append(node)
            path_set.add(node)
            for neighbor in wait_for.get(node, []):
                dfs(neighbor)
            path.pop()
            path_set.remove(node)

        for node in wait_for:
            if node not in visited:
                dfs(node)

        return {"ok": True, "deadlocks": cycles, "wait_for_graph": {
            k: list(v) for k, v in wait_for.items()
        }}

    # -- Serialization helpers --

    def _serialize_lock(self, lock):
        if isinstance(lock, list):
            return [self._serialize_single_lock(l) for l in lock]
        return self._serialize_single_lock(lock)

    def _serialize_single_lock(self, lock):
        return {
            "resource": lock.resource, "mode": lock.mode.value,
            "owner": lock.owner,
            "fencing_token": {"value": lock.fencing_token.value,
                              "resource": lock.fencing_token.resource,
                              "owner": lock.fencing_token.owner,
                              "epoch": lock.fencing_token.epoch},
            "acquired_at": lock.acquired_at, "ttl": lock.ttl,
            "metadata": lock.metadata,
        }

    def _deserialize_lock(self, data):
        if isinstance(data, list):
            return [self._deserialize_single_lock(d) for d in data]
        return self._deserialize_single_lock(data)

    def _deserialize_single_lock(self, data):
        ft = data["fencing_token"]
        token = FencingToken(ft["value"], ft["resource"], ft["owner"], ft["epoch"])
        return LockEntry(
            resource=data["resource"], mode=LockMode(data["mode"]),
            owner=data["owner"], fencing_token=token,
            acquired_at=data["acquired_at"], ttl=data["ttl"],
            metadata=data.get("metadata", {}),
        )

    def _serialize_wait(self, wait):
        return {
            "resource": wait.resource, "mode": wait.mode.value,
            "session_id": wait.session_id, "requested_at": wait.requested_at,
            "ttl": wait.ttl, "metadata": wait.metadata,
        }

    def _deserialize_wait(self, data):
        return WaitEntry(
            resource=data["resource"], mode=LockMode(data["mode"]),
            session_id=data["session_id"], requested_at=data["requested_at"],
            ttl=data["ttl"], metadata=data.get("metadata", {}),
        )

    def _serialize_session(self, sess):
        return {
            "session_id": sess.session_id, "created_at": sess.created_at,
            "last_heartbeat": sess.last_heartbeat, "ttl": sess.ttl,
            "state": sess.state.value, "held_locks": sess.held_locks[:],
            "metadata": sess.metadata,
        }

    def _deserialize_session(self, data):
        return Session(
            session_id=data["session_id"], created_at=data["created_at"],
            last_heartbeat=data["last_heartbeat"], ttl=data["ttl"],
            state=SessionState(data["state"]),
            held_locks=data.get("held_locks", []),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Lock Service Cluster
# ---------------------------------------------------------------------------

class LockServiceNode:
    """A node in the distributed lock service."""

    def __init__(self, node_id, peers, election_timeout_range=(150, 300), heartbeat_interval=50):
        self.node_id = node_id
        self.state_machine = LockStateMachine()
        self.raft = RaftNode(
            node_id=node_id,
            peers=peers,
            state_machine=self.state_machine,
            election_timeout_range=election_timeout_range,
            heartbeat_interval=heartbeat_interval,
        )

    @property
    def is_leader(self):
        return self.raft.role == "leader"

    def status(self):
        s = self.raft.status()
        s["lock_count"] = len(self.state_machine.locks)
        s["session_count"] = len([
            s2 for s2 in self.state_machine.sessions.values()
            if s2.state == SessionState.ACTIVE
        ])
        s["wait_queue_count"] = sum(len(q) for q in self.state_machine.wait_queues.values())
        return s


class LockServiceCluster:
    """Distributed Lock Service built on Raft consensus."""

    def __init__(self, node_ids, election_timeout_range=(150, 300), heartbeat_interval=50):
        self.node_ids = list(node_ids)
        self.nodes = {}
        self.raft_cluster = RaftCluster(
            node_ids,
            election_timeout_range=election_timeout_range,
            heartbeat_interval=heartbeat_interval,
        )

        # Replace state machines with LockStateMachine
        for nid in node_ids:
            sm = LockStateMachine()
            self.raft_cluster.nodes[nid].state_machine = sm
            self.nodes[nid] = self.raft_cluster.nodes[nid]

    def tick(self, ms=1):
        self.raft_cluster.tick(ms)

    def tick_until(self, predicate, max_ticks=2000, ms_per_tick=1):
        return self.raft_cluster.tick_until(predicate, max_ticks, ms_per_tick)

    def deliver_all(self):
        self.raft_cluster.deliver_all()

    def wait_for_leader(self, max_ticks=2000):
        return self.raft_cluster.wait_for_leader(max_ticks)

    def get_leader(self):
        leader_node = self.raft_cluster.get_leader()
        if leader_node is None:
            return None
        return leader_node.id

    def get_state_machine(self, node_id=None):
        """Get state machine from leader or specific node."""
        if node_id is None:
            leader = self.get_leader()
            if leader is None:
                return None
            node_id = leader
        return self.nodes[node_id].state_machine

    def submit(self, command, request_id=None):
        """Submit command to leader."""
        leader = self.get_leader()
        if leader is None:
            return {"ok": False, "error": "no leader"}
        return self.raft_cluster.submit(command, request_id=request_id, node_id=leader)

    def submit_and_commit(self, command, request_id=None, max_ticks=2000):
        """Submit command and wait for commit."""
        return self.raft_cluster.submit_and_commit(command, request_id=request_id, max_ticks=max_ticks)

    def partition(self, group_a, group_b):
        self.raft_cluster.partition(group_a, group_b)

    def heal_partition(self):
        self.raft_cluster.heal_partition()


# ---------------------------------------------------------------------------
# Lock Client
# ---------------------------------------------------------------------------

class LockClient:
    """Client interface for the distributed lock service."""

    def __init__(self, cluster, client_id=None):
        self.cluster = cluster
        self.client_id = client_id or f"client_{id(self)}"
        self.session_id = None
        self._request_counter = 0

    def _next_request_id(self):
        self._request_counter += 1
        return f"{self.client_id}_req_{self._request_counter}"

    def create_session(self, ttl=30.0, now=0, metadata=None):
        """Create a new session."""
        self.session_id = f"session_{self.client_id}_{self._request_counter}"
        result = self.cluster.submit_and_commit({
            "op": "create_session",
            "session_id": self.session_id,
            "ttl": ttl,
            "now": now,
            "metadata": metadata or {},
        }, request_id=self._next_request_id())
        return result

    def heartbeat(self, now=0):
        """Send session heartbeat."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "heartbeat",
            "session_id": self.session_id,
            "now": now,
        }, request_id=self._next_request_id())

    def destroy_session(self, now=0):
        """Destroy session and release all locks."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        result = self.cluster.submit_and_commit({
            "op": "destroy_session",
            "session_id": self.session_id,
            "now": now,
        }, request_id=self._next_request_id())
        self.session_id = None
        return result

    def acquire(self, resource, mode="exclusive", ttl=30.0, now=0, metadata=None):
        """Acquire a lock (will queue if not immediately available)."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "acquire",
            "resource": resource,
            "session_id": self.session_id,
            "mode": mode,
            "ttl": ttl,
            "now": now,
            "metadata": metadata or {},
        }, request_id=self._next_request_id())

    def try_acquire(self, resource, mode="exclusive", ttl=30.0, now=0, metadata=None):
        """Try to acquire lock without queuing."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "try_acquire",
            "resource": resource,
            "session_id": self.session_id,
            "mode": mode,
            "ttl": ttl,
            "now": now,
            "metadata": metadata or {},
        }, request_id=self._next_request_id())

    def release(self, resource, now=0):
        """Release a lock."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "release",
            "resource": resource,
            "session_id": self.session_id,
            "now": now,
        }, request_id=self._next_request_id())

    def renew(self, resource, ttl=30.0, now=0):
        """Renew a lock's TTL."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "renew",
            "resource": resource,
            "session_id": self.session_id,
            "ttl": ttl,
            "now": now,
        }, request_id=self._next_request_id())

    def acquire_group(self, resources, mode="exclusive", ttl=30.0, now=0):
        """Atomically acquire multiple locks."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "acquire_group",
            "resources": resources,
            "session_id": self.session_id,
            "mode": mode,
            "ttl": ttl,
            "now": now,
        }, request_id=self._next_request_id())

    def release_group(self, resources, now=0):
        """Release multiple locks."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "release_group",
            "resources": resources,
            "session_id": self.session_id,
            "now": now,
        }, request_id=self._next_request_id())

    def watch(self, resource, events=None, callback_id=None):
        """Watch a resource for state changes."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "watch",
            "resource": resource,
            "session_id": self.session_id,
            "events": events or ["acquired", "released", "expired"],
            "callback_id": callback_id or f"watch_{self.session_id}_{resource}",
        }, request_id=self._next_request_id())

    def unwatch(self, resource, callback_id=None):
        """Stop watching a resource."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "unwatch",
            "resource": resource,
            "session_id": self.session_id,
            "callback_id": callback_id,
        }, request_id=self._next_request_id())

    def get_lock_info(self, resource):
        """Get info about a lock."""
        return self.cluster.submit_and_commit({
            "op": "get_lock_info",
            "resource": resource,
        }, request_id=self._next_request_id())

    def get_session_info(self):
        """Get info about current session."""
        if not self.session_id:
            return {"ok": False, "error": "no session"}
        return self.cluster.submit_and_commit({
            "op": "get_session_info",
            "session_id": self.session_id,
        }, request_id=self._next_request_id())

    def expire_check(self, now=0):
        """Trigger expiration check."""
        return self.cluster.submit_and_commit({
            "op": "expire_check",
            "now": now,
        }, request_id=self._next_request_id())

    def deadlock_check(self):
        """Check for deadlocks."""
        return self.cluster.submit_and_commit({
            "op": "deadlock_check",
        }, request_id=self._next_request_id())


# ---------------------------------------------------------------------------
# Lock Service Stats
# ---------------------------------------------------------------------------

class LockServiceStats:
    """Statistics for the lock service."""

    def __init__(self, cluster):
        self.cluster = cluster

    def get_stats(self):
        sm = self.cluster.get_state_machine()
        if sm is None:
            return {"error": "no leader"}

        active_sessions = sum(
            1 for s in sm.sessions.values() if s.state == SessionState.ACTIVE
        )
        total_locks = 0
        exclusive_locks = 0
        shared_locks = 0
        for lock in sm.locks.values():
            if isinstance(lock, list):
                total_locks += len(lock)
                shared_locks += len(lock)
            else:
                total_locks += 1
                exclusive_locks += 1

        total_waiters = sum(len(q) for q in sm.wait_queues.values())
        watched_resources = len(sm.watches)

        return {
            "active_sessions": active_sessions,
            "total_sessions": len(sm.sessions),
            "total_locks": total_locks,
            "exclusive_locks": exclusive_locks,
            "shared_locks": shared_locks,
            "total_waiters": total_waiters,
            "watched_resources": watched_resources,
            "fencing_counter": sm.fencing_counter,
            "leader": self.cluster.get_leader(),
        }
