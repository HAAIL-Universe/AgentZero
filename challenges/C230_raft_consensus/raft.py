"""
C230: Raft Consensus Protocol

A simulation-friendly implementation of the Raft consensus algorithm.
Supports leader election, log replication, commitment, membership changes,
log compaction (snapshots), and pre-vote protocol.

All networking is abstracted through message passing -- no real sockets.
A Network simulator handles message delivery, partitions, and delays.
"""

import time
import random
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any


# --- Enums ---

class Role(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    LEARNER = "learner"  # non-voting member during config changes


class MessageType(Enum):
    # Election
    REQUEST_VOTE = "request_vote"
    REQUEST_VOTE_RESPONSE = "request_vote_response"
    PRE_VOTE = "pre_vote"
    PRE_VOTE_RESPONSE = "pre_vote_response"
    # Log replication
    APPEND_ENTRIES = "append_entries"
    APPEND_ENTRIES_RESPONSE = "append_entries_response"
    # Snapshots
    INSTALL_SNAPSHOT = "install_snapshot"
    INSTALL_SNAPSHOT_RESPONSE = "install_snapshot_response"
    # Client
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"


# --- Data structures ---

@dataclass
class LogEntry:
    term: int
    index: int
    command: Any
    entry_type: str = "command"  # "command", "config_change", "noop"

    def to_dict(self):
        return {"term": self.term, "index": self.index, "command": self.command, "entry_type": self.entry_type}

    @staticmethod
    def from_dict(d):
        return LogEntry(term=d["term"], index=d["index"], command=d["command"], entry_type=d.get("entry_type", "command"))


@dataclass
class Message:
    type: MessageType
    sender: str
    receiver: str
    term: int
    data: dict = field(default_factory=dict)


@dataclass
class Snapshot:
    last_included_index: int
    last_included_term: int
    data: Any  # application state
    config: list = field(default_factory=list)  # cluster membership at snapshot time


# --- State Machine ---

class StateMachine:
    """Simple key-value state machine."""

    def __init__(self):
        self.data = {}

    def apply(self, command):
        if command is None:
            return None
        if isinstance(command, dict):
            op = command.get("op")
            if op == "set":
                self.data[command["key"]] = command["value"]
                return command["value"]
            elif op == "delete":
                return self.data.pop(command["key"], None)
            elif op == "get":
                return self.data.get(command["key"])
            elif op == "cas":  # compare-and-swap
                key, expected, new_val = command["key"], command["expected"], command["value"]
                if self.data.get(key) == expected:
                    self.data[key] = new_val
                    return True
                return False
        return None

    def snapshot(self):
        return dict(self.data)

    def restore(self, snapshot_data):
        self.data = dict(snapshot_data) if snapshot_data else {}


# --- Raft Node ---

class RaftNode:
    """A single Raft consensus node."""

    def __init__(self, node_id, peers, election_timeout_range=(150, 300)):
        self.node_id = node_id
        self.peers = list(peers)  # other node IDs
        self.role = Role.FOLLOWER
        self.state_machine = StateMachine()

        # Persistent state
        self.current_term = 0
        self.voted_for = None
        self.log = []  # list of LogEntry

        # Volatile state
        self.commit_index = 0
        self.last_applied = 0

        # Leader volatile state
        self.next_index = {}   # peer -> next log index to send
        self.match_index = {}  # peer -> highest replicated index

        # Election
        self.election_timeout_range = election_timeout_range
        self.election_timeout = self._random_timeout()
        self.election_timer = 0.0
        self.votes_received = set()
        self.pre_votes_received = set()
        self.pre_vote_in_progress = False

        # Leader heartbeat
        self.heartbeat_interval = 50  # ms
        self.heartbeat_timer = 0.0

        # Pending client requests: index -> callback info
        self.pending_requests = {}

        # Outgoing messages
        self.outbox = []

        # Committed entries (for testing)
        self.committed_entries = []

        # Snapshot state
        self.snapshot = None  # current snapshot
        self.snapshot_threshold = 100  # compact after this many entries

        # Cluster config
        self.cluster_config = [node_id] + list(peers)
        self.learners = set()  # nodes in learner (non-voting) state

        # Pre-vote enabled
        self.pre_vote_enabled = False

        # Leader ID (known)
        self.leader_id = None

        # Applied results
        self.applied_results = {}  # index -> result

    def _random_timeout(self):
        lo, hi = self.election_timeout_range
        return random.randint(lo, hi)

    def _last_log_index(self):
        if self.log:
            return self.log[-1].index
        if self.snapshot:
            return self.snapshot.last_included_index
        return 0

    def _last_log_term(self):
        if self.log:
            return self.log[-1].term
        if self.snapshot:
            return self.snapshot.last_included_term
        return 0

    def _get_log_entry(self, index):
        """Get log entry by index (1-based)."""
        if index <= 0:
            return None
        base = self._log_base_index()
        offset = index - base - 1
        if 0 <= offset < len(self.log):
            return self.log[offset]
        return None

    def _log_base_index(self):
        """Index of last entry included in snapshot (0 if no snapshot)."""
        if self.snapshot:
            return self.snapshot.last_included_index
        return 0

    def _log_base_term(self):
        if self.snapshot:
            return self.snapshot.last_included_term
        return 0

    def _get_term_at(self, index):
        """Get term at given log index."""
        if index <= 0:
            return 0
        if index == self._log_base_index():
            return self._log_base_term()
        entry = self._get_log_entry(index)
        return entry.term if entry else 0

    def _voting_peers(self):
        """Peers that can vote (excludes learners)."""
        return [p for p in self.peers if p not in self.learners]

    def _voting_members(self):
        """All voting members including self."""
        members = [self.node_id] + self._voting_peers()
        return [m for m in members if m not in self.learners]

    def _quorum_size(self):
        return len(self._voting_members()) // 2 + 1

    # --- Message handling ---

    def receive(self, msg):
        """Process an incoming message."""
        # Any RPC with higher term: step down
        if msg.term > self.current_term:
            self.current_term = msg.term
            self.voted_for = None
            if self.role != Role.FOLLOWER:
                self._become_follower()

        handlers = {
            MessageType.REQUEST_VOTE: self._handle_request_vote,
            MessageType.REQUEST_VOTE_RESPONSE: self._handle_request_vote_response,
            MessageType.PRE_VOTE: self._handle_pre_vote,
            MessageType.PRE_VOTE_RESPONSE: self._handle_pre_vote_response,
            MessageType.APPEND_ENTRIES: self._handle_append_entries,
            MessageType.APPEND_ENTRIES_RESPONSE: self._handle_append_entries_response,
            MessageType.INSTALL_SNAPSHOT: self._handle_install_snapshot,
            MessageType.INSTALL_SNAPSHOT_RESPONSE: self._handle_install_snapshot_response,
            MessageType.CLIENT_REQUEST: self._handle_client_request,
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    def _send(self, msg_type, receiver, data=None):
        msg = Message(
            type=msg_type,
            sender=self.node_id,
            receiver=receiver,
            term=self.current_term,
            data=data or {}
        )
        self.outbox.append(msg)

    # --- Timer / Tick ---

    def tick(self, elapsed_ms=1):
        """Advance time by elapsed_ms. Call this in simulation loop."""
        if self.role == Role.LEADER:
            self.heartbeat_timer += elapsed_ms
            if self.heartbeat_timer >= self.heartbeat_interval:
                self.heartbeat_timer = 0
                self._send_heartbeats()
        else:
            self.election_timer += elapsed_ms
            if self.election_timer >= self.election_timeout:
                self.election_timer = 0
                self.election_timeout = self._random_timeout()
                if self.pre_vote_enabled:
                    self._start_pre_vote()
                else:
                    self._start_election()

        # Apply committed entries
        self._apply_committed()

    def _apply_committed(self):
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self._get_log_entry(self.last_applied)
            if entry and entry.entry_type == "command":
                result = self.state_machine.apply(entry.command)
                self.applied_results[self.last_applied] = result
                self.committed_entries.append(entry)
            elif entry and entry.entry_type == "config_change":
                self._apply_config_change(entry.command)
                self.committed_entries.append(entry)

    # --- Election ---

    def _become_follower(self):
        self.role = Role.FOLLOWER
        self.election_timer = 0
        self.election_timeout = self._random_timeout()
        self.pre_vote_in_progress = False

    def _become_candidate(self):
        self.role = Role.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.election_timer = 0
        self.election_timeout = self._random_timeout()
        self.leader_id = None

    def _become_leader(self):
        self.role = Role.LEADER
        self.leader_id = self.node_id
        self.heartbeat_timer = 0
        last = self._last_log_index() + 1
        for peer in self.peers:
            self.next_index[peer] = last
            self.match_index[peer] = 0
        # Append noop to commit entries from previous terms
        noop = LogEntry(term=self.current_term, index=last, command=None, entry_type="noop")
        self.log.append(noop)
        self._send_heartbeats()

    def _start_pre_vote(self):
        """Pre-vote: check if election would succeed without incrementing term."""
        self.pre_vote_in_progress = True
        self.pre_votes_received = {self.node_id}
        for peer in self._voting_peers():
            self._send(MessageType.PRE_VOTE, peer, {
                "candidate_id": self.node_id,
                "last_log_index": self._last_log_index(),
                "last_log_term": self._last_log_term(),
                "next_term": self.current_term + 1,
            })
        # Single node cluster
        if len(self._voting_members()) == 1:
            self._start_election()

    def _start_election(self):
        self.pre_vote_in_progress = False
        self._become_candidate()
        for peer in self._voting_peers():
            self._send(MessageType.REQUEST_VOTE, peer, {
                "candidate_id": self.node_id,
                "last_log_index": self._last_log_index(),
                "last_log_term": self._last_log_term(),
            })
        # Single node cluster
        if self._check_quorum(self.votes_received):
            self._become_leader()

    def _check_quorum(self, vote_set):
        voting = set(self._voting_members())
        actual_votes = vote_set & voting
        return len(actual_votes) >= self._quorum_size()

    def _handle_pre_vote(self, msg):
        data = msg.data
        next_term = data.get("next_term", msg.term + 1)
        grant = False
        # Grant pre-vote if: next_term >= our term, and candidate's log is up-to-date
        if next_term >= self.current_term:
            grant = self._is_log_up_to_date(data.get("last_log_term", 0), data.get("last_log_index", 0))
        self._send(MessageType.PRE_VOTE_RESPONSE, msg.sender, {
            "vote_granted": grant,
            "next_term": next_term,
        })

    def _handle_pre_vote_response(self, msg):
        if not self.pre_vote_in_progress:
            return
        if msg.data.get("vote_granted"):
            self.pre_votes_received.add(msg.sender)
            if self._check_quorum(self.pre_votes_received):
                self.pre_vote_in_progress = False
                self._start_election()

    def _handle_request_vote(self, msg):
        data = msg.data
        grant = False
        if msg.term >= self.current_term:
            if self.voted_for is None or self.voted_for == data.get("candidate_id"):
                if self._is_log_up_to_date(data.get("last_log_term", 0), data.get("last_log_index", 0)):
                    grant = True
                    self.voted_for = data["candidate_id"]
                    self.election_timer = 0  # reset election timer on granting vote
        self._send(MessageType.REQUEST_VOTE_RESPONSE, msg.sender, {"vote_granted": grant})

    def _handle_request_vote_response(self, msg):
        if self.role != Role.CANDIDATE:
            return
        if msg.data.get("vote_granted"):
            self.votes_received.add(msg.sender)
            if self._check_quorum(self.votes_received):
                self._become_leader()

    def _is_log_up_to_date(self, candidate_term, candidate_index):
        my_term = self._last_log_term()
        my_index = self._last_log_index()
        if candidate_term != my_term:
            return candidate_term > my_term
        return candidate_index >= my_index

    # --- Log Replication ---

    def _send_heartbeats(self):
        for peer in self.peers:
            self._send_append_entries(peer)

    def _send_append_entries(self, peer):
        ni = self.next_index.get(peer, 1)
        base = self._log_base_index()

        # If next_index is at or before our snapshot, send snapshot instead
        if self.snapshot and ni <= base:
            self._send_install_snapshot(peer)
            return

        prev_index = ni - 1
        prev_term = self._get_term_at(prev_index)

        # Gather entries from next_index onward
        entries = []
        for i in range(ni, self._last_log_index() + 1):
            entry = self._get_log_entry(i)
            if entry:
                entries.append(entry.to_dict())

        self._send(MessageType.APPEND_ENTRIES, peer, {
            "leader_id": self.node_id,
            "prev_log_index": prev_index,
            "prev_log_term": prev_term,
            "entries": entries,
            "leader_commit": self.commit_index,
        })

    def _handle_append_entries(self, msg):
        data = msg.data
        success = False

        if msg.term < self.current_term:
            self._send(MessageType.APPEND_ENTRIES_RESPONSE, msg.sender, {
                "success": False,
                "match_index": 0,
            })
            return

        # Valid leader -- reset election timer
        self.election_timer = 0
        self.leader_id = data.get("leader_id", msg.sender)
        if self.role == Role.CANDIDATE:
            self._become_follower()

        prev_index = data.get("prev_log_index", 0)
        prev_term = data.get("prev_log_term", 0)

        # Check log consistency
        if prev_index == 0:
            success = True
        elif prev_index == self._log_base_index():
            success = (prev_term == self._log_base_term())
        else:
            prev_entry = self._get_log_entry(prev_index)
            success = prev_entry is not None and prev_entry.term == prev_term

        if success:
            # Append new entries
            entries = [LogEntry.from_dict(e) for e in data.get("entries", [])]
            for entry in entries:
                existing = self._get_log_entry(entry.index)
                if existing and existing.term != entry.term:
                    # Conflict: delete this and all following
                    base = self._log_base_index()
                    offset = entry.index - base - 1
                    self.log = self.log[:offset]
                if not self._get_log_entry(entry.index):
                    self.log.append(entry)

            # Update commit index
            leader_commit = data.get("leader_commit", 0)
            if leader_commit > self.commit_index:
                last_new = entries[-1].index if entries else self._last_log_index()
                self.commit_index = min(leader_commit, last_new)

            match_idx = self._last_log_index()
        else:
            match_idx = 0

        self._send(MessageType.APPEND_ENTRIES_RESPONSE, msg.sender, {
            "success": success,
            "match_index": match_idx,
        })

    def _handle_append_entries_response(self, msg):
        if self.role != Role.LEADER:
            return
        peer = msg.sender
        if msg.data.get("success"):
            mi = msg.data.get("match_index", 0)
            self.match_index[peer] = max(self.match_index.get(peer, 0), mi)
            self.next_index[peer] = mi + 1
            self._try_advance_commit()
        else:
            # Decrement next_index and retry
            ni = self.next_index.get(peer, 1)
            self.next_index[peer] = max(1, ni - 1)
            # Don't go below snapshot base
            base = self._log_base_index()
            if self.next_index[peer] <= base and self.snapshot:
                self.next_index[peer] = base  # will trigger snapshot send
            self._send_append_entries(peer)

    def _try_advance_commit(self):
        """Advance commit_index if a majority has replicated."""
        voting = self._voting_members()
        for n in range(self.commit_index + 1, self._last_log_index() + 1):
            entry = self._get_log_entry(n)
            if entry and entry.term == self.current_term:
                # Count replicas
                count = 1  # self
                for peer in self._voting_peers():
                    if self.match_index.get(peer, 0) >= n:
                        count += 1
                if count >= self._quorum_size():
                    self.commit_index = n

    # --- Client requests ---

    def submit(self, command):
        """Submit a client command. Returns the log index or None if not leader."""
        if self.role != Role.LEADER:
            return None
        index = self._last_log_index() + 1
        entry = LogEntry(term=self.current_term, index=index, command=command)
        self.log.append(entry)
        self.match_index[self.node_id] = index
        self._try_advance_commit()  # single-node: commit immediately
        self._send_heartbeats()  # replicate to peers
        return index

    def _handle_client_request(self, msg):
        if self.role != Role.LEADER:
            self._send(MessageType.CLIENT_RESPONSE, msg.sender, {
                "success": False,
                "leader_id": self.leader_id,
                "error": "not_leader",
            })
            return
        command = msg.data.get("command")
        index = self.submit(command)
        self.pending_requests[index] = msg.sender

    # --- Snapshots ---

    def take_snapshot(self):
        """Compact log by creating a snapshot up to last_applied."""
        if self.last_applied <= self._log_base_index():
            return None
        snap_data = self.state_machine.snapshot()
        snap_term = self._get_term_at(self.last_applied)
        self.snapshot = Snapshot(
            last_included_index=self.last_applied,
            last_included_term=snap_term,
            data=snap_data,
            config=list(self.cluster_config),
        )
        # Discard compacted entries
        base = self.last_applied
        self.log = [e for e in self.log if e.index > base]
        return self.snapshot

    def maybe_compact(self):
        """Auto-compact if log is large enough."""
        if len(self.log) >= self.snapshot_threshold:
            return self.take_snapshot()
        return None

    def _send_install_snapshot(self, peer):
        if not self.snapshot:
            return
        self._send(MessageType.INSTALL_SNAPSHOT, peer, {
            "leader_id": self.node_id,
            "last_included_index": self.snapshot.last_included_index,
            "last_included_term": self.snapshot.last_included_term,
            "data": self.snapshot.data,
            "config": self.snapshot.config,
        })

    def _handle_install_snapshot(self, msg):
        if msg.term < self.current_term:
            self._send(MessageType.INSTALL_SNAPSHOT_RESPONSE, msg.sender, {"success": False})
            return

        data = msg.data
        snap_index = data["last_included_index"]
        snap_term = data["last_included_term"]

        # If we already have entries past the snapshot, keep them
        self.snapshot = Snapshot(
            last_included_index=snap_index,
            last_included_term=snap_term,
            data=data["data"],
            config=data.get("config", []),
        )
        # Discard log entries covered by snapshot
        self.log = [e for e in self.log if e.index > snap_index]
        # Reset state machine
        self.state_machine.restore(data["data"])
        self.last_applied = snap_index
        if snap_index > self.commit_index:
            self.commit_index = snap_index
        # Update config
        if data.get("config"):
            self.cluster_config = list(data["config"])

        self.election_timer = 0
        self._send(MessageType.INSTALL_SNAPSHOT_RESPONSE, msg.sender, {
            "success": True,
            "match_index": snap_index,
        })

    def _handle_install_snapshot_response(self, msg):
        if self.role != Role.LEADER:
            return
        if msg.data.get("success"):
            mi = msg.data.get("match_index", 0)
            self.match_index[msg.sender] = max(self.match_index.get(msg.sender, 0), mi)
            self.next_index[msg.sender] = mi + 1

    # --- Cluster Membership Changes ---

    def add_member(self, new_node_id):
        """Add a new member (starts as learner, then promoted via config change)."""
        if self.role != Role.LEADER:
            return None
        if new_node_id in self.cluster_config:
            return None
        # Add as learner first
        self.learners.add(new_node_id)
        if new_node_id not in self.peers:
            self.peers.append(new_node_id)
        last = self._last_log_index() + 1
        self.next_index[new_node_id] = last
        self.match_index[new_node_id] = 0
        # Submit config change entry
        new_config = self.cluster_config + [new_node_id]
        entry_cmd = {"op": "add_member", "node_id": new_node_id, "config": new_config}
        index = self._last_log_index() + 1
        entry = LogEntry(term=self.current_term, index=index, command=entry_cmd, entry_type="config_change")
        self.log.append(entry)
        self.match_index[self.node_id] = index
        self._send_heartbeats()
        return index

    def remove_member(self, node_id):
        """Remove a member via config change."""
        if self.role != Role.LEADER:
            return None
        if node_id not in self.cluster_config:
            return None
        new_config = [n for n in self.cluster_config if n != node_id]
        entry_cmd = {"op": "remove_member", "node_id": node_id, "config": new_config}
        index = self._last_log_index() + 1
        entry = LogEntry(term=self.current_term, index=index, command=entry_cmd, entry_type="config_change")
        self.log.append(entry)
        self.match_index[self.node_id] = index
        self._send_heartbeats()
        return index

    def _apply_config_change(self, command):
        if not isinstance(command, dict):
            return
        op = command.get("op")
        new_config = command.get("config", [])
        node_id = command.get("node_id")
        if op == "add_member":
            self.cluster_config = new_config
            if node_id in self.learners:
                self.learners.discard(node_id)  # promote to voter
        elif op == "remove_member":
            self.cluster_config = new_config
            if node_id in self.peers:
                self.peers.remove(node_id)
            self.learners.discard(node_id)
            self.next_index.pop(node_id, None)
            self.match_index.pop(node_id, None)
            # If we removed ourselves, step down
            if node_id == self.node_id and self.role == Role.LEADER:
                self._become_follower()

    # --- Serialization ---

    def get_state(self):
        """Return a dict of this node's state (for debugging/testing)."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "term": self.current_term,
            "voted_for": self.voted_for,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "last_log_index": self._last_log_index(),
            "last_log_term": self._last_log_term(),
            "leader_id": self.leader_id,
            "cluster_config": self.cluster_config,
            "state_machine": self.state_machine.data,
        }


# --- Network Simulator ---

class Network:
    """Simulates message passing between Raft nodes with partition support."""

    def __init__(self):
        self.nodes = {}  # id -> RaftNode
        self.partitions = set()  # set of frozenset({a, b}) -- blocked pairs
        self.message_queue = []  # delayed messages
        self.delivered = []  # all delivered messages (for testing)
        self.dropped = []  # all dropped messages (for testing)
        self.delay_range = (0, 0)  # (min_ms, max_ms) simulated delay

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        self.nodes.pop(node_id, None)

    def partition(self, node_a, node_b):
        """Block communication between two nodes."""
        self.partitions.add(frozenset({node_a, node_b}))

    def heal(self, node_a=None, node_b=None):
        """Remove partition between nodes, or all partitions if no args."""
        if node_a is None and node_b is None:
            self.partitions.clear()
        else:
            self.partitions.discard(frozenset({node_a, node_b}))

    def is_partitioned(self, a, b):
        return frozenset({a, b}) in self.partitions

    def isolate(self, node_id):
        """Partition a node from all others."""
        for other in self.nodes:
            if other != node_id:
                self.partition(node_id, other)

    def tick(self, elapsed_ms=1):
        """Advance simulation by elapsed_ms."""
        # Tick all nodes
        for node in list(self.nodes.values()):
            node.tick(elapsed_ms)

        # Collect outgoing messages
        for node in list(self.nodes.values()):
            while node.outbox:
                msg = node.outbox.pop(0)
                if self.is_partitioned(msg.sender, msg.receiver):
                    self.dropped.append(msg)
                    continue
                if msg.receiver not in self.nodes:
                    self.dropped.append(msg)
                    continue
                self.message_queue.append(msg)

        # Deliver messages
        pending = self.message_queue
        self.message_queue = []
        for msg in pending:
            if msg.receiver in self.nodes:
                self.nodes[msg.receiver].receive(msg)
                self.delivered.append(msg)
            else:
                self.dropped.append(msg)

    def run(self, ticks=1, tick_ms=1):
        """Run simulation for N ticks."""
        for _ in range(ticks):
            self.tick(tick_ms)

    def find_leader(self):
        """Find the current leader, if any."""
        leaders = [n for n in self.nodes.values() if n.role == Role.LEADER]
        if len(leaders) == 1:
            return leaders[0]
        return None

    def wait_for_leader(self, max_ticks=2000, tick_ms=1):
        """Run until a leader is elected."""
        for _ in range(max_ticks):
            self.tick(tick_ms)
            leader = self.find_leader()
            if leader:
                return leader
        return None

    def wait_for_commit(self, index, max_ticks=2000, tick_ms=1):
        """Run until the leader commits up to given index."""
        for _ in range(max_ticks):
            self.tick(tick_ms)
            leader = self.find_leader()
            if leader and leader.commit_index >= index:
                return True
        return False

    def wait_for_replication(self, index, count=None, max_ticks=2000, tick_ms=1):
        """Wait until 'count' nodes have applied up to 'index'."""
        if count is None:
            count = len(self.nodes)
        for _ in range(max_ticks):
            self.tick(tick_ms)
            applied = sum(1 for n in self.nodes.values() if n.last_applied >= index)
            if applied >= count:
                return True
        return False


def create_cluster(node_ids, **kwargs):
    """Create a Raft cluster with given node IDs."""
    net = Network()
    for nid in node_ids:
        peers = [p for p in node_ids if p != nid]
        node = RaftNode(nid, peers, **kwargs)
        net.add_node(node)
    return net
