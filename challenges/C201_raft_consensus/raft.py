"""
C201: Raft Consensus Protocol
=============================
A complete implementation of the Raft consensus algorithm for distributed systems.

Components:
- RaftNode: Core node with state machine (Follower/Candidate/Leader)
- Log: Replicated log with term tracking
- RPC messages: RequestVote, AppendEntries, InstallSnapshot
- StateMachine: Pluggable state machine interface
- RaftCluster: Simulated cluster for testing (no real network)
- Membership changes: AddServer/RemoveServer (single-server changes)
- Log compaction: Snapshots with InstallSnapshot RPC
- Read-only queries: Leader lease optimization

Based on the Raft paper (Ongaro & Ousterhout, 2014).
"""

import time
import random
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


# --- Enums ---

class Role(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class MessageType(Enum):
    REQUEST_VOTE = "request_vote"
    REQUEST_VOTE_RESPONSE = "request_vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_ENTRIES_RESPONSE = "append_entries_response"
    INSTALL_SNAPSHOT = "install_snapshot"
    INSTALL_SNAPSHOT_RESPONSE = "install_snapshot_response"
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"


# --- Data Structures ---

@dataclass
class LogEntry:
    term: int
    index: int
    command: Any
    # For membership changes
    is_config_change: bool = False

    def to_dict(self):
        return {"term": self.term, "index": self.index, "command": self.command,
                "is_config_change": self.is_config_change}

    @staticmethod
    def from_dict(d):
        return LogEntry(d["term"], d["index"], d["command"], d.get("is_config_change", False))


@dataclass
class Message:
    type: MessageType
    src: str
    dst: str
    term: int
    data: dict = field(default_factory=dict)


@dataclass
class Snapshot:
    last_included_index: int
    last_included_term: int
    data: Any  # Serialized state machine state
    config: list = field(default_factory=list)  # Cluster membership at snapshot time


# --- State Machine ---

class KeyValueStateMachine:
    """Simple key-value store as a pluggable state machine."""

    def __init__(self):
        self.store = {}
        self.last_applied = 0

    def apply(self, command, index):
        """Apply a command and return the result."""
        if command is None:
            self.last_applied = index
            return None

        op = command.get("op") if isinstance(command, dict) else None
        if op == "set":
            self.store[command["key"]] = command["value"]
            self.last_applied = index
            return command["value"]
        elif op == "get":
            self.last_applied = index
            return self.store.get(command["key"])
        elif op == "delete":
            val = self.store.pop(command["key"], None)
            self.last_applied = index
            return val
        elif op == "cas":  # Compare-and-swap
            key = command["key"]
            expected = command["expected"]
            new_val = command["new_value"]
            if self.store.get(key) == expected:
                self.store[key] = new_val
                self.last_applied = index
                return True
            self.last_applied = index
            return False
        else:
            # No-op or unknown
            self.last_applied = index
            return None

    def snapshot(self):
        """Return serializable state."""
        return {"store": dict(self.store), "last_applied": self.last_applied}

    def restore(self, data):
        """Restore from snapshot data."""
        self.store = dict(data["store"])
        self.last_applied = data["last_applied"]


# --- Raft Log ---

class RaftLog:
    """The replicated log. Index is 1-based."""

    def __init__(self):
        self.entries = []  # List[LogEntry], 0-indexed internally but LogEntry.index is 1-based
        self.snapshot_last_index = 0
        self.snapshot_last_term = 0

    def __len__(self):
        return self.snapshot_last_index + len(self.entries)

    def _to_internal(self, index):
        """Convert 1-based log index to internal list index."""
        return index - self.snapshot_last_index - 1

    def get(self, index):
        """Get entry at 1-based index. Returns None if out of range or compacted."""
        if index <= self.snapshot_last_index:
            return None
        internal = self._to_internal(index)
        if 0 <= internal < len(self.entries):
            return self.entries[internal]
        return None

    def last_index(self):
        """Return the index of the last log entry."""
        return self.snapshot_last_index + len(self.entries)

    def last_term(self):
        """Return the term of the last log entry."""
        if self.entries:
            return self.entries[-1].term
        return self.snapshot_last_term

    def term_at(self, index):
        """Return the term at a given index."""
        if index == 0:
            return 0
        if index == self.snapshot_last_index:
            return self.snapshot_last_term
        entry = self.get(index)
        return entry.term if entry else None

    def append(self, entry):
        """Append an entry to the log."""
        self.entries.append(entry)

    def entries_from(self, start_index):
        """Return entries from start_index onwards."""
        if start_index <= self.snapshot_last_index:
            start_index = self.snapshot_last_index + 1
        internal = self._to_internal(start_index)
        if internal < 0:
            internal = 0
        return self.entries[internal:]

    def truncate_from(self, index):
        """Remove all entries from index onwards."""
        internal = self._to_internal(index)
        if internal < 0:
            internal = 0
        self.entries = self.entries[:internal]

    def compact(self, last_index, last_term):
        """Discard entries up to last_index (for snapshots)."""
        internal = self._to_internal(last_index)
        if internal >= 0:
            self.entries = self.entries[internal + 1:]
        else:
            self.entries = []
        self.snapshot_last_index = last_index
        self.snapshot_last_term = last_term

    def match_term(self, index, term):
        """Check if entry at index has the given term."""
        if index == 0:
            return True
        if index == self.snapshot_last_index:
            return self.snapshot_last_term == term
        entry = self.get(index)
        return entry is not None and entry.term == term


# --- Raft Node ---

class RaftNode:
    """A single Raft node."""

    def __init__(self, node_id, peers, state_machine=None,
                 election_timeout_range=(150, 300),
                 heartbeat_interval=50):
        self.id = node_id
        self.peers = list(peers)  # Other node IDs
        self.role = Role.FOLLOWER
        self.state_machine = state_machine or KeyValueStateMachine()
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval

        # Persistent state
        self.current_term = 0
        self.voted_for = None
        self.log = RaftLog()

        # Volatile state
        self.commit_index = 0
        self.last_applied = 0

        # Leader state
        self.next_index = {}   # peer_id -> next log index to send
        self.match_index = {}  # peer_id -> highest log index known replicated

        # Election state
        self.votes_received = set()
        self.election_timer = 0
        self.heartbeat_timer = 0
        self._reset_election_timer()

        # Snapshot
        self.snapshot = None

        # Pending client requests: {log_index: callback_id}
        self.pending_requests = {}

        # Outbox for messages
        self.outbox = []

        # Committed results for client responses
        self.committed_results = {}

        # Leader ID (for redirects)
        self.leader_id = None

        # For read-only optimization
        self.read_index_requests = []  # (read_id, command)
        self.acks_for_read = {}  # read_id -> set of acks

    def _reset_election_timer(self):
        lo, hi = self.election_timeout_range
        self.election_timer = random.randint(lo, hi)

    def _all_servers(self):
        """All servers in the cluster including self."""
        return [self.id] + self.peers

    def _quorum_size(self):
        """Majority of the cluster."""
        return (len(self._all_servers()) // 2) + 1

    # --- Timer Ticks ---

    def tick(self, ms=1):
        """Advance timers by ms milliseconds."""
        if self.role == Role.LEADER:
            self.heartbeat_timer -= ms
            if self.heartbeat_timer <= 0:
                self._send_heartbeats()
                self.heartbeat_timer = self.heartbeat_interval
        else:
            self.election_timer -= ms
            if self.election_timer <= 0:
                self._start_election()

    # --- Elections ---

    def _start_election(self):
        """Transition to candidate, increment term, vote for self, request votes."""
        self.current_term += 1
        self.role = Role.CANDIDATE
        self.voted_for = self.id
        self.votes_received = {self.id}
        self.leader_id = None
        self._reset_election_timer()

        for peer in self.peers:
            self.outbox.append(Message(
                type=MessageType.REQUEST_VOTE,
                src=self.id,
                dst=peer,
                term=self.current_term,
                data={
                    "candidate_id": self.id,
                    "last_log_index": self.log.last_index(),
                    "last_log_term": self.log.last_term(),
                }
            ))

        # Single node cluster: immediately become leader
        if self._quorum_size() == 1:
            self._become_leader()

    def _become_leader(self):
        """Transition to leader."""
        self.role = Role.LEADER
        self.leader_id = self.id
        self.heartbeat_timer = 0  # Send heartbeats immediately

        # Initialize next_index and match_index
        last = self.log.last_index() + 1
        for peer in self.peers:
            self.next_index[peer] = last
            self.match_index[peer] = 0

        # Append a no-op entry to commit entries from previous terms
        noop = LogEntry(
            term=self.current_term,
            index=self.log.last_index() + 1,
            command=None
        )
        self.log.append(noop)

    def _become_follower(self, term, leader_id=None):
        """Transition to follower."""
        self.role = Role.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self.leader_id = leader_id
        self._reset_election_timer()

    # --- Message Handling ---

    def receive(self, msg):
        """Handle an incoming message."""
        # If RPC term > currentTerm, become follower
        if msg.term > self.current_term:
            self._become_follower(msg.term)

        handlers = {
            MessageType.REQUEST_VOTE: self._handle_request_vote,
            MessageType.REQUEST_VOTE_RESPONSE: self._handle_request_vote_response,
            MessageType.APPEND_ENTRIES: self._handle_append_entries,
            MessageType.APPEND_ENTRIES_RESPONSE: self._handle_append_entries_response,
            MessageType.INSTALL_SNAPSHOT: self._handle_install_snapshot,
            MessageType.INSTALL_SNAPSHOT_RESPONSE: self._handle_install_snapshot_response,
            MessageType.CLIENT_REQUEST: self._handle_client_request,
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    def _handle_request_vote(self, msg):
        data = msg.data
        grant = False

        if msg.term >= self.current_term:
            # Update term if needed
            if msg.term > self.current_term:
                self._become_follower(msg.term)

            # Grant vote if we haven't voted for someone else
            # and candidate's log is at least as up-to-date as ours
            if (self.voted_for is None or self.voted_for == data["candidate_id"]):
                if self._log_is_up_to_date(data["last_log_index"], data["last_log_term"]):
                    grant = True
                    self.voted_for = data["candidate_id"]
                    self._reset_election_timer()

        self.outbox.append(Message(
            type=MessageType.REQUEST_VOTE_RESPONSE,
            src=self.id,
            dst=msg.src,
            term=self.current_term,
            data={"vote_granted": grant}
        ))

    def _log_is_up_to_date(self, last_index, last_term):
        """Check if candidate's log is at least as up-to-date as ours."""
        my_last_term = self.log.last_term()
        my_last_index = self.log.last_index()
        if last_term != my_last_term:
            return last_term > my_last_term
        return last_index >= my_last_index

    def _handle_request_vote_response(self, msg):
        if self.role != Role.CANDIDATE:
            return
        if msg.term != self.current_term:
            return

        if msg.data["vote_granted"]:
            self.votes_received.add(msg.src)
            if len(self.votes_received) >= self._quorum_size():
                self._become_leader()

    def _handle_append_entries(self, msg):
        data = msg.data
        success = False
        match_index = 0

        if msg.term < self.current_term:
            # Reject stale term
            pass
        else:
            # Valid leader
            self.role = Role.FOLLOWER
            self.leader_id = msg.src
            self._reset_election_timer()

            prev_index = data["prev_log_index"]
            prev_term = data["prev_log_term"]

            # Check log consistency
            if prev_index == 0 or self.log.match_term(prev_index, prev_term):
                success = True
                # Append new entries
                entries = [LogEntry.from_dict(e) for e in data.get("entries", [])]
                idx = prev_index + 1
                for entry in entries:
                    existing = self.log.get(idx)
                    if existing and existing.term != entry.term:
                        # Conflict: truncate from here
                        self.log.truncate_from(idx)
                        self.log.append(entry)
                    elif not existing:
                        self.log.append(entry)
                    idx += 1

                # Update commit index
                if data["leader_commit"] > self.commit_index:
                    self.commit_index = min(data["leader_commit"], self.log.last_index())

                match_index = prev_index + len(entries)

                # Apply committed entries
                self._apply_committed()

        self.outbox.append(Message(
            type=MessageType.APPEND_ENTRIES_RESPONSE,
            src=self.id,
            dst=msg.src,
            term=self.current_term,
            data={"success": success, "match_index": match_index}
        ))

    def _handle_append_entries_response(self, msg):
        if self.role != Role.LEADER:
            return
        if msg.term != self.current_term:
            return

        peer = msg.src
        if msg.data["success"]:
            self.match_index[peer] = msg.data["match_index"]
            self.next_index[peer] = msg.data["match_index"] + 1
            self._advance_commit_index()
        else:
            # Decrement next_index and retry
            if peer in self.next_index:
                self.next_index[peer] = max(1, self.next_index[peer] - 1)
                # If next_index falls below snapshot boundary, send snapshot
                if self.snapshot and self.next_index[peer] <= self.snapshot.last_included_index:
                    self._send_snapshot(peer)
                else:
                    self._send_append_entries(peer)

    def _handle_install_snapshot(self, msg):
        data = msg.data
        if msg.term < self.current_term:
            self.outbox.append(Message(
                type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
                src=self.id, dst=msg.src, term=self.current_term,
                data={"success": False}
            ))
            return

        self.role = Role.FOLLOWER
        self.leader_id = msg.src
        self._reset_election_timer()

        snap = Snapshot(
            last_included_index=data["last_included_index"],
            last_included_term=data["last_included_term"],
            data=data["data"],
            config=data.get("config", [])
        )

        # Apply snapshot
        self.snapshot = snap
        self.log.compact(snap.last_included_index, snap.last_included_term)
        self.state_machine.restore(snap.data)
        self.last_applied = snap.last_included_index
        self.commit_index = max(self.commit_index, snap.last_included_index)

        self.outbox.append(Message(
            type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
            src=self.id, dst=msg.src, term=self.current_term,
            data={"success": True, "match_index": snap.last_included_index}
        ))

    def _handle_install_snapshot_response(self, msg):
        if self.role != Role.LEADER:
            return
        if msg.data.get("success"):
            peer = msg.src
            mi = msg.data["match_index"]
            self.match_index[peer] = max(self.match_index.get(peer, 0), mi)
            self.next_index[peer] = mi + 1

    def _handle_client_request(self, msg):
        """Handle a client command submission."""
        if self.role != Role.LEADER:
            # Redirect to leader
            self.outbox.append(Message(
                type=MessageType.CLIENT_RESPONSE,
                src=self.id, dst=msg.src, term=self.current_term,
                data={"success": False, "error": "not_leader", "leader_id": self.leader_id}
            ))
            return

        command = msg.data.get("command")
        request_id = msg.data.get("request_id")

        # Read-only optimization: don't append to log
        if isinstance(command, dict) and command.get("op") == "get":
            result = self.state_machine.apply(command, self.last_applied)
            self.committed_results[request_id] = result
            self.outbox.append(Message(
                type=MessageType.CLIENT_RESPONSE,
                src=self.id, dst=msg.src, term=self.current_term,
                data={"success": True, "result": result, "request_id": request_id}
            ))
            return

        # Append to log
        entry = LogEntry(
            term=self.current_term,
            index=self.log.last_index() + 1,
            command=command,
            is_config_change=msg.data.get("is_config_change", False)
        )
        self.log.append(entry)
        if request_id:
            self.pending_requests[entry.index] = request_id

        # Replicate to peers
        for peer in self.peers:
            self._send_append_entries(peer)

    # --- Log Replication ---

    def _send_heartbeats(self):
        """Send AppendEntries to all peers (may include new entries)."""
        for peer in self.peers:
            if self.snapshot and self.next_index.get(peer, 1) <= self.snapshot.last_included_index:
                self._send_snapshot(peer)
            else:
                self._send_append_entries(peer)

    def _send_append_entries(self, peer):
        ni = self.next_index.get(peer, 1)
        prev_index = ni - 1
        prev_term = self.log.term_at(prev_index) if prev_index > 0 else 0

        # If prev_term is None, we've compacted that entry -- send snapshot
        if prev_term is None and self.snapshot:
            self._send_snapshot(peer)
            return

        entries = self.log.entries_from(ni)

        self.outbox.append(Message(
            type=MessageType.APPEND_ENTRIES,
            src=self.id,
            dst=peer,
            term=self.current_term,
            data={
                "prev_log_index": prev_index,
                "prev_log_term": prev_term or 0,
                "entries": [e.to_dict() for e in entries],
                "leader_commit": self.commit_index,
            }
        ))

    def _send_snapshot(self, peer):
        if not self.snapshot:
            return
        self.outbox.append(Message(
            type=MessageType.INSTALL_SNAPSHOT,
            src=self.id,
            dst=peer,
            term=self.current_term,
            data={
                "last_included_index": self.snapshot.last_included_index,
                "last_included_term": self.snapshot.last_included_term,
                "data": self.snapshot.data,
                "config": self.snapshot.config,
            }
        ))

    def _advance_commit_index(self):
        """Update commit index based on majority replication."""
        # Check each index from commit_index+1 to last log entry
        for n in range(self.log.last_index(), self.commit_index, -1):
            term_at_n = self.log.term_at(n)
            if term_at_n != self.current_term:
                continue
            # Count replications (self + peers)
            count = 1  # self
            for peer in self.peers:
                if self.match_index.get(peer, 0) >= n:
                    count += 1
            if count >= self._quorum_size():
                self.commit_index = n
                self._apply_committed()
                break

    def _apply_committed(self):
        """Apply committed but not yet applied entries."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log.get(self.last_applied)
            if entry:
                result = self.state_machine.apply(entry.command, self.last_applied)
                # If this was a pending client request, store result
                req_id = self.pending_requests.pop(self.last_applied, None)
                if req_id:
                    self.committed_results[req_id] = result

    # --- Snapshots ---

    def take_snapshot(self):
        """Compact the log by snapshotting current state."""
        if self.last_applied == 0:
            return None
        term = self.log.term_at(self.last_applied)
        if term is None:
            # Already compacted past this point
            if self.snapshot and self.snapshot.last_included_index >= self.last_applied:
                return self.snapshot
            return None

        snap = Snapshot(
            last_included_index=self.last_applied,
            last_included_term=term,
            data=self.state_machine.snapshot(),
            config=self._all_servers()
        )
        self.snapshot = snap
        self.log.compact(self.last_applied, term)
        return snap

    # --- Membership Changes ---

    def add_server(self, new_id):
        """Add a server to the cluster (leader only). Returns success."""
        if self.role != Role.LEADER:
            return False
        if new_id in self.peers or new_id == self.id:
            return False

        self.peers.append(new_id)
        self.next_index[new_id] = self.log.last_index() + 1
        self.match_index[new_id] = 0

        # Log the config change
        entry = LogEntry(
            term=self.current_term,
            index=self.log.last_index() + 1,
            command={"op": "add_server", "server_id": new_id},
            is_config_change=True
        )
        self.log.append(entry)
        return True

    def remove_server(self, server_id):
        """Remove a server from the cluster (leader only)."""
        if self.role != Role.LEADER:
            return False
        if server_id == self.id:
            # Leader stepping down
            entry = LogEntry(
                term=self.current_term,
                index=self.log.last_index() + 1,
                command={"op": "remove_server", "server_id": server_id},
                is_config_change=True
            )
            self.log.append(entry)
            self._become_follower(self.current_term)
            return True

        if server_id not in self.peers:
            return False

        self.peers.remove(server_id)
        self.next_index.pop(server_id, None)
        self.match_index.pop(server_id, None)

        entry = LogEntry(
            term=self.current_term,
            index=self.log.last_index() + 1,
            command={"op": "remove_server", "server_id": server_id},
            is_config_change=True
        )
        self.log.append(entry)
        # Re-check commit after membership change
        self._advance_commit_index()
        return True

    # --- Diagnostics ---

    def status(self):
        return {
            "id": self.id,
            "role": self.role.value,
            "term": self.current_term,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "leader_id": self.leader_id,
            "voted_for": self.voted_for,
            "peers": list(self.peers),
        }


# --- Raft Cluster (Simulated) ---

class RaftCluster:
    """A simulated Raft cluster with controllable message delivery."""

    def __init__(self, node_ids, election_timeout_range=(150, 300), heartbeat_interval=50):
        self.nodes = {}
        self.network = []  # Pending messages
        self.partitions = set()  # Set of (src, dst) pairs that are partitioned
        self.dropped_messages = []
        self.delivered_messages = []
        self.message_delay = {}  # (src, dst) -> delay in ticks

        for nid in node_ids:
            peers = [p for p in node_ids if p != nid]
            self.nodes[nid] = RaftNode(
                nid, peers,
                election_timeout_range=election_timeout_range,
                heartbeat_interval=heartbeat_interval
            )

    def tick(self, ms=1):
        """Advance all nodes by ms milliseconds."""
        for node in self.nodes.values():
            node.tick(ms)
        self._collect_messages()

    def tick_until(self, predicate, max_ticks=10000, ms_per_tick=1):
        """Tick until predicate() returns True or max_ticks exceeded."""
        for _ in range(max_ticks):
            self.tick(ms_per_tick)
            self.deliver_all()
            if predicate():
                return True
        return False

    def _collect_messages(self):
        """Collect outbox messages from all nodes into the network."""
        for node in self.nodes.values():
            for msg in node.outbox:
                self.network.append(msg)
            node.outbox.clear()

    def deliver_all(self):
        """Deliver all pending network messages."""
        self._collect_messages()
        while self.network:
            messages = list(self.network)
            self.network.clear()
            for msg in messages:
                self._deliver(msg)
            self._collect_messages()

    def deliver_one(self):
        """Deliver a single message."""
        self._collect_messages()
        if self.network:
            msg = self.network.pop(0)
            self._deliver(msg)
            self._collect_messages()
            return True
        return False

    def _deliver(self, msg):
        """Deliver a message, respecting partitions."""
        pair = (msg.src, msg.dst)
        if pair in self.partitions:
            self.dropped_messages.append(msg)
            return
        if msg.dst in self.nodes:
            self.nodes[msg.dst].receive(msg)
            self.delivered_messages.append(msg)
            self._collect_messages()

    def partition(self, group_a, group_b):
        """Create a network partition between two groups."""
        for a in group_a:
            for b in group_b:
                self.partitions.add((a, b))
                self.partitions.add((b, a))

    def heal_partition(self):
        """Remove all partitions."""
        self.partitions.clear()

    def get_leader(self):
        """Return the current leader node, or None."""
        leaders = [n for n in self.nodes.values() if n.role == Role.LEADER]
        if len(leaders) == 1:
            return leaders[0]
        return None

    def get_leaders(self):
        """Return all nodes claiming to be leader."""
        return [n for n in self.nodes.values() if n.role == Role.LEADER]

    def wait_for_leader(self, max_ticks=5000):
        """Tick until exactly one leader is elected."""
        def has_leader():
            leaders = self.get_leaders()
            return len(leaders) == 1
        return self.tick_until(has_leader, max_ticks)

    def submit(self, command, request_id=None, node_id=None):
        """Submit a client command to the leader (or specified node)."""
        if node_id:
            target = self.nodes[node_id]
        else:
            target = self.get_leader()
            if not target:
                return None

        msg = Message(
            type=MessageType.CLIENT_REQUEST,
            src="client",
            dst=target.id,
            term=0,
            data={"command": command, "request_id": request_id or f"req_{id(command)}"}
        )
        target.receive(msg)
        self._collect_messages()
        return request_id or f"req_{id(command)}"

    def submit_and_commit(self, command, request_id=None, max_ticks=3000):
        """Submit a command and wait for it to be committed."""
        leader = self.get_leader()
        if not leader:
            return None
        rid = request_id or f"req_{random.randint(0, 10**9)}"
        self.submit(command, rid, leader.id)
        # Deliver until committed
        initial_commit = leader.commit_index
        self.tick_until(lambda: leader.commit_index > initial_commit, max_ticks)
        self.deliver_all()
        # Allow followers to apply
        for _ in range(100):
            self.tick(1)
            self.deliver_all()
        return leader.committed_results.get(rid)

    def add_node(self, node_id):
        """Add a new node to the cluster."""
        leader = self.get_leader()
        if not leader:
            return False

        # Create the new node with all existing nodes as peers
        all_existing = list(self.nodes.keys())
        self.nodes[node_id] = RaftNode(
            node_id,
            all_existing,
            election_timeout_range=leader.election_timeout_range,
            heartbeat_interval=leader.heartbeat_interval
        )

        # Tell the leader about the new server
        leader.add_server(node_id)

        # Tell other followers about the new peer
        for nid, node in self.nodes.items():
            if nid != node_id and nid != leader.id:
                if node_id not in node.peers:
                    node.peers.append(node_id)

        return True

    def remove_node(self, node_id):
        """Remove a node from the cluster."""
        leader = self.get_leader()
        if not leader:
            return False

        leader.remove_server(node_id)

        # Remove from other nodes' peer lists
        for nid, node in self.nodes.items():
            if nid != node_id and node_id in node.peers:
                node.peers.remove(node_id)

        # Keep the node object but it's effectively isolated
        return True

    def check_safety(self):
        """Verify Raft safety properties."""
        # Election Safety: at most one leader per term
        leader_terms = {}
        for node in self.nodes.values():
            if node.role == Role.LEADER:
                if node.current_term in leader_terms:
                    return False, f"Multiple leaders in term {node.current_term}"
                leader_terms[node.current_term] = node.id

        # Log Matching: if two logs contain an entry with the same index and term,
        # then the logs are identical in all entries up through that index
        nodes_list = list(self.nodes.values())
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                n1, n2 = nodes_list[i], nodes_list[j]
                min_len = min(len(n1.log), len(n2.log))
                for idx in range(1, min_len + 1):
                    e1 = n1.log.get(idx)
                    e2 = n2.log.get(idx)
                    if e1 and e2 and e1.term == e2.term:
                        # Same term at same index -- commands should match
                        if e1.command != e2.command:
                            return False, f"Log mismatch at index {idx}"

        # State Machine Safety: committed entries produce same state
        committed_vals = {}
        for node in self.nodes.values():
            for idx in range(1, node.last_applied + 1):
                entry = node.log.get(idx)
                if entry:
                    key = (idx, entry.term)
                    if key in committed_vals:
                        if committed_vals[key] != entry.command:
                            return False, f"Committed entry mismatch at {idx}"
                    committed_vals[key] = entry.command

        return True, "All safety properties hold"


# --- Convenience ---

def create_cluster(n=3, **kwargs):
    """Create an n-node cluster with default settings."""
    ids = [f"node_{i}" for i in range(n)]
    return RaftCluster(ids, **kwargs)
