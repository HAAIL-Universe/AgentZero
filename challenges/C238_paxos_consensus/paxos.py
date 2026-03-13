"""
C238: Paxos Consensus Protocol

Implements:
- Basic Paxos (single-decree): Proposer, Acceptor, Learner roles
- Multi-Paxos: replicated state machine with stable leader optimization
- Distinguished Proposer (leader) to reduce conflicts
- Log compaction via snapshots
- Membership changes
- Network simulator with partition support

All networking is abstracted through message passing -- no real sockets.
"""

import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


# --- Enums ---

class MessageType(Enum):
    # Basic Paxos Phase 1
    PREPARE = "prepare"
    PROMISE = "promise"
    # Basic Paxos Phase 2
    ACCEPT = "accept"
    ACCEPTED = "accepted"
    NACK = "nack"
    # Learner notification
    LEARN = "learn"
    # Multi-Paxos
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"
    # Leader election
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    # Snapshots
    INSTALL_SNAPSHOT = "install_snapshot"
    INSTALL_SNAPSHOT_RESPONSE = "install_snapshot_response"


class NodeRole(Enum):
    PROPOSER = "proposer"
    ACCEPTOR = "acceptor"
    LEARNER = "learner"


# --- Data Structures ---

@dataclass
class ProposalID:
    """Proposal number: (sequence_number, node_id) for total ordering."""
    number: int
    node_id: str

    def __lt__(self, other):
        if other is None:
            return False
        return (self.number, self.node_id) < (other.number, other.node_id)

    def __le__(self, other):
        if other is None:
            return False
        return (self.number, self.node_id) <= (other.number, other.node_id)

    def __gt__(self, other):
        if other is None:
            return True
        return (self.number, self.node_id) > (other.number, other.node_id)

    def __ge__(self, other):
        if other is None:
            return True
        return (self.number, self.node_id) >= (other.number, other.node_id)

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, ProposalID):
            return NotImplemented
        return (self.number, self.node_id) == (other.number, other.node_id)

    def __hash__(self):
        return hash((self.number, self.node_id))

    def to_dict(self):
        return {"number": self.number, "node_id": self.node_id}

    @staticmethod
    def from_dict(d):
        if d is None:
            return None
        return ProposalID(number=d["number"], node_id=d["node_id"])


@dataclass
class Message:
    type: MessageType
    sender: str
    receiver: str
    data: dict = field(default_factory=dict)


@dataclass
class LogEntry:
    slot: int
    proposal_id: Optional[ProposalID]
    value: Any
    entry_type: str = "command"  # "command", "config_change", "noop"

    def to_dict(self):
        return {
            "slot": self.slot,
            "proposal_id": self.proposal_id.to_dict() if self.proposal_id else None,
            "value": self.value,
            "entry_type": self.entry_type,
        }

    @staticmethod
    def from_dict(d):
        return LogEntry(
            slot=d["slot"],
            proposal_id=ProposalID.from_dict(d.get("proposal_id")),
            value=d["value"],
            entry_type=d.get("entry_type", "command"),
        )


@dataclass
class Snapshot:
    last_included_slot: int
    data: Any
    config: list = field(default_factory=list)


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
            elif op == "cas":
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


# --- Acceptor ---

class Acceptor:
    """
    Paxos Acceptor: maintains promised and accepted state per slot.

    For each slot tracks:
    - highest_promised: highest proposal ID promised
    - accepted_proposal: highest proposal ID accepted
    - accepted_value: value accepted with that proposal
    """

    def __init__(self, node_id):
        self.node_id = node_id
        # Per-slot state: slot -> {promised, accepted_id, accepted_value}
        self.slots = {}
        self.outbox = []

    def _get_slot(self, slot):
        if slot not in self.slots:
            self.slots[slot] = {
                "promised": None,
                "accepted_id": None,
                "accepted_value": None,
            }
        return self.slots[slot]

    def receive_prepare(self, msg):
        """Handle Phase 1a: Prepare request."""
        slot = msg.data["slot"]
        proposal_id = ProposalID.from_dict(msg.data["proposal_id"])
        state = self._get_slot(slot)

        if state["promised"] is None or proposal_id >= state["promised"]:
            # Promise: won't accept anything less than this proposal
            state["promised"] = proposal_id
            self.outbox.append(Message(
                type=MessageType.PROMISE,
                sender=self.node_id,
                receiver=msg.sender,
                data={
                    "slot": slot,
                    "proposal_id": proposal_id.to_dict(),
                    "accepted_id": state["accepted_id"].to_dict() if state["accepted_id"] else None,
                    "accepted_value": state["accepted_value"],
                }
            ))
        else:
            # NACK: already promised higher
            self.outbox.append(Message(
                type=MessageType.NACK,
                sender=self.node_id,
                receiver=msg.sender,
                data={
                    "slot": slot,
                    "proposal_id": proposal_id.to_dict(),
                    "promised": state["promised"].to_dict(),
                }
            ))

    def receive_accept(self, msg):
        """Handle Phase 2a: Accept request."""
        slot = msg.data["slot"]
        proposal_id = ProposalID.from_dict(msg.data["proposal_id"])
        value = msg.data["value"]
        state = self._get_slot(slot)

        if state["promised"] is None or proposal_id >= state["promised"]:
            # Accept the proposal
            state["promised"] = proposal_id
            state["accepted_id"] = proposal_id
            state["accepted_value"] = value
            self.outbox.append(Message(
                type=MessageType.ACCEPTED,
                sender=self.node_id,
                receiver=msg.sender,
                data={
                    "slot": slot,
                    "proposal_id": proposal_id.to_dict(),
                    "value": value,
                }
            ))
        else:
            # NACK: already promised higher
            self.outbox.append(Message(
                type=MessageType.NACK,
                sender=self.node_id,
                receiver=msg.sender,
                data={
                    "slot": slot,
                    "proposal_id": proposal_id.to_dict(),
                    "promised": state["promised"].to_dict(),
                }
            ))

    def get_accepted(self, slot):
        """Return (proposal_id, value) for a slot, or (None, None)."""
        state = self.slots.get(slot)
        if state:
            return state["accepted_id"], state["accepted_value"]
        return None, None

    def compact(self, up_to_slot):
        """Remove slot state for slots <= up_to_slot (after snapshot)."""
        to_remove = [s for s in self.slots if s <= up_to_slot]
        for s in to_remove:
            del self.slots[s]


# --- Learner ---

class Learner:
    """
    Paxos Learner: tracks accepted values to detect when consensus is reached.
    """

    def __init__(self, node_id, quorum_size):
        self.node_id = node_id
        self.quorum_size = quorum_size
        # slot -> {proposal_id -> set of acceptors}
        self.accepted = {}
        # slot -> decided value
        self.decided = {}

    def receive_accepted(self, slot, proposal_id, value, acceptor_id):
        """Record that an acceptor accepted a value for a slot."""
        if slot in self.decided:
            return self.decided[slot]

        if slot not in self.accepted:
            self.accepted[slot] = {}

        pid_key = (proposal_id.number, proposal_id.node_id)
        if pid_key not in self.accepted[slot]:
            self.accepted[slot][pid_key] = {"value": value, "acceptors": set()}
        self.accepted[slot][pid_key]["acceptors"].add(acceptor_id)

        # Check quorum
        if len(self.accepted[slot][pid_key]["acceptors"]) >= self.quorum_size:
            self.decided[slot] = value
            # Clean up
            del self.accepted[slot]
            return value

        return None

    def is_decided(self, slot):
        return slot in self.decided

    def get_decided(self, slot):
        return self.decided.get(slot)

    def compact(self, up_to_slot):
        """Remove state for slots <= up_to_slot."""
        for s in list(self.accepted.keys()):
            if s <= up_to_slot:
                del self.accepted[s]


# --- Proposer (Basic Paxos) ---

class Proposer:
    """
    Paxos Proposer: drives consensus for individual slots.
    """

    def __init__(self, node_id, acceptor_ids, quorum_size):
        self.node_id = node_id
        self.acceptor_ids = list(acceptor_ids)
        self.quorum_size = quorum_size
        self.proposal_seq = 0
        self.outbox = []

        # Per-slot proposal state
        # slot -> {proposal_id, value, promises: list, phase}
        self.proposals = {}

    def _next_proposal_id(self):
        self.proposal_seq += 1
        return ProposalID(number=self.proposal_seq, node_id=self.node_id)

    def propose(self, slot, value):
        """Start Phase 1: send Prepare to all acceptors."""
        pid = self._next_proposal_id()
        self.proposals[slot] = {
            "proposal_id": pid,
            "value": value,
            "promises": [],
            "nacks": 0,
            "accepted_count": 0,
            "phase": 1,
        }
        for aid in self.acceptor_ids:
            self.outbox.append(Message(
                type=MessageType.PREPARE,
                sender=self.node_id,
                receiver=aid,
                data={"slot": slot, "proposal_id": pid.to_dict()},
            ))
        return pid

    def receive_promise(self, msg):
        """Handle Phase 1b: Promise from acceptor."""
        slot = msg.data["slot"]
        if slot not in self.proposals:
            return None
        state = self.proposals[slot]
        if state["phase"] != 1:
            return None

        pid = ProposalID.from_dict(msg.data["proposal_id"])
        if pid != state["proposal_id"]:
            return None  # stale promise

        state["promises"].append({
            "acceptor": msg.sender,
            "accepted_id": ProposalID.from_dict(msg.data.get("accepted_id")),
            "accepted_value": msg.data.get("accepted_value"),
        })

        if len(state["promises"]) >= self.quorum_size:
            # Phase 1 complete: move to Phase 2
            # Use highest-numbered accepted value if any
            highest = None
            for p in state["promises"]:
                if p["accepted_id"] is not None:
                    if highest is None or p["accepted_id"] > highest["accepted_id"]:
                        highest = p
            if highest is not None and highest["accepted_value"] is not None:
                state["value"] = highest["accepted_value"]

            state["phase"] = 2
            # Send Accept to all acceptors
            for aid in self.acceptor_ids:
                self.outbox.append(Message(
                    type=MessageType.ACCEPT,
                    sender=self.node_id,
                    receiver=aid,
                    data={
                        "slot": slot,
                        "proposal_id": state["proposal_id"].to_dict(),
                        "value": state["value"],
                    },
                ))
            return "phase2_started"
        return None

    def receive_accepted(self, msg):
        """Handle Phase 2b: Accepted from acceptor."""
        slot = msg.data["slot"]
        if slot not in self.proposals:
            return None
        state = self.proposals[slot]
        if state["phase"] != 2:
            return None

        pid = ProposalID.from_dict(msg.data["proposal_id"])
        if pid != state["proposal_id"]:
            return None

        state["accepted_count"] += 1
        if state["accepted_count"] >= self.quorum_size:
            state["phase"] = 3  # decided
            return state["value"]
        return None

    def receive_nack(self, msg):
        """Handle NACK: another proposer has higher proposal."""
        slot = msg.data["slot"]
        if slot not in self.proposals:
            return
        state = self.proposals[slot]
        state["nacks"] += 1
        promised = ProposalID.from_dict(msg.data.get("promised"))
        if promised and promised.number >= self.proposal_seq:
            self.proposal_seq = promised.number  # catch up

    def is_decided(self, slot):
        state = self.proposals.get(slot)
        return state is not None and state["phase"] == 3

    def get_decided_value(self, slot):
        state = self.proposals.get(slot)
        if state and state["phase"] == 3:
            return state["value"]
        return None


# --- Multi-Paxos Node ---

class MultiPaxosNode:
    """
    A combined Proposer/Acceptor/Learner node for Multi-Paxos.

    Supports:
    - Stable leader optimization (skip Phase 1 when leader is stable)
    - Log-based replicated state machine
    - Heartbeat-based leader detection
    - Snapshot/compaction
    - Membership changes
    """

    def __init__(self, node_id, peer_ids, election_timeout_range=(150, 300)):
        self.node_id = node_id
        self.peer_ids = list(peer_ids)
        self.all_ids = [node_id] + self.peer_ids

        # Roles
        self.acceptor = Acceptor(node_id)
        quorum = len(self.all_ids) // 2 + 1
        self.learner = Learner(node_id, quorum)
        self.proposer = Proposer(node_id, self.all_ids, quorum)

        # State machine
        self.state_machine = StateMachine()
        self.last_applied = 0
        self.applied_results = {}

        # Leader state
        self.leader_id = None
        self.is_leader = False
        self.leader_lease_timer = 0.0
        self.leader_lease_timeout = 300  # ms
        self.heartbeat_interval = 50  # ms
        self.heartbeat_timer = 0.0

        # Election
        self.election_timeout_range = election_timeout_range
        self.election_timeout = self._random_timeout()
        self.election_timer = 0.0

        # Multi-Paxos slot tracking
        self.next_slot = 1  # next slot to propose into
        self.first_unchosen = 1  # first slot not yet decided

        # Stable leader optimization: if we are the stable leader, skip Phase 1
        self.leader_prepare_done = {}  # slot -> True if Phase 1 done for this leader era
        self.leader_era = 0  # incremented each time we become leader

        # Pending client requests
        self.pending_requests = {}  # slot -> client info

        # Outbox
        self.outbox = []

        # Snapshot
        self.snapshot = None
        self.snapshot_threshold = 100

        # Committed entries (for testing)
        self.committed_entries = []

        # Cluster config
        self.cluster_config = list(self.all_ids)

        # Heartbeat tracking
        self.heartbeat_acks = {}  # peer -> last ack time
        self.sim_time = 0.0

    def _random_timeout(self):
        lo, hi = self.election_timeout_range
        return random.randint(lo, hi)

    def _quorum_size(self):
        return len(self.cluster_config) // 2 + 1

    # --- Tick ---

    def tick(self, elapsed_ms=1):
        """Advance time."""
        self.sim_time += elapsed_ms

        if self.is_leader:
            self.heartbeat_timer += elapsed_ms
            if self.heartbeat_timer >= self.heartbeat_interval:
                self.heartbeat_timer = 0
                self._send_heartbeats()
        else:
            self.election_timer += elapsed_ms
            if self.election_timer >= self.election_timeout:
                self.election_timer = 0
                self.election_timeout = self._random_timeout()
                self._try_become_leader()

        # Apply decided entries
        self._apply_decided()

    def _apply_decided(self):
        """Apply all consecutively decided slots."""
        while True:
            slot = self.last_applied + 1
            value = self.learner.get_decided(slot)
            if value is None:
                break
            if isinstance(value, dict) and value.get("__type") == "config_change":
                self._apply_config_change(value)
            elif isinstance(value, dict) and value.get("__type") == "noop":
                pass  # noop, just advance
            else:
                result = self.state_machine.apply(value)
                self.applied_results[slot] = result
            self.committed_entries.append(LogEntry(slot=slot, proposal_id=None, value=value))
            self.last_applied = slot
            self.first_unchosen = slot + 1

    # --- Leader Election via Paxos ---

    def _try_become_leader(self):
        """Attempt to become leader by running Paxos on a leadership slot."""
        self._become_leader()

    def _become_leader(self):
        """Declare self as leader. In Multi-Paxos, this means we start
        driving consensus for new slots."""
        self.is_leader = True
        self.leader_id = self.node_id
        self.leader_era += 1
        self.heartbeat_timer = 0
        self.leader_prepare_done = {}

        # Update proposer's acceptor list
        self.proposer.acceptor_ids = list(self.cluster_config)
        self.proposer.quorum_size = self._quorum_size()
        self.learner.quorum_size = self._quorum_size()

        # Ensure next_slot is past all known decided/proposed slots
        max_known = max(self.last_applied, self.first_unchosen - 1)
        for s in self.proposer.proposals:
            max_known = max(max_known, s)
        self.next_slot = max(self.next_slot, max_known + 1)

        # Propose a noop to establish leadership (fills any gaps)
        self._propose_noop()
        self._send_heartbeats()

    def _step_down(self):
        """Revert to follower."""
        self.is_leader = False
        self.election_timer = 0
        self.election_timeout = self._random_timeout()

    def _propose_noop(self):
        """Propose a no-op to establish leadership for the current slot."""
        slot = self.next_slot
        noop = {"__type": "noop"}
        self._run_paxos(slot, noop)
        self.next_slot = slot + 1

    # --- Heartbeats ---

    def _send_heartbeats(self):
        for peer in self.peer_ids:
            if peer in self.cluster_config:
                self.outbox.append(Message(
                    type=MessageType.HEARTBEAT,
                    sender=self.node_id,
                    receiver=peer,
                    data={
                        "leader_id": self.node_id,
                        "first_unchosen": self.first_unchosen,
                        "last_applied": self.last_applied,
                    },
                ))

    def _handle_heartbeat(self, msg):
        leader = msg.data.get("leader_id")
        if leader:
            self.leader_id = leader
            if self.is_leader and leader != self.node_id:
                # Another leader exists -- step down if their proposal is higher
                # Simple: step down
                self._step_down()
            self.election_timer = 0
            # Send ack
            self.outbox.append(Message(
                type=MessageType.HEARTBEAT_ACK,
                sender=self.node_id,
                receiver=msg.sender,
                data={"last_applied": self.last_applied},
            ))

    def _handle_heartbeat_ack(self, msg):
        self.heartbeat_acks[msg.sender] = self.sim_time

    # --- Paxos Protocol ---

    def _run_paxos(self, slot, value):
        """Run full Paxos (Phase 1 + Phase 2) for a slot."""
        self.proposer.propose(slot, value)
        self._flush_proposer_outbox()

    def _run_paxos_phase2_only(self, slot, value):
        """Skip Phase 1 (leader optimization). Send Accept directly."""
        pid = self.proposer._next_proposal_id()
        self.proposer.proposals[slot] = {
            "proposal_id": pid,
            "value": value,
            "promises": [],
            "nacks": 0,
            "accepted_count": 0,
            "phase": 2,
        }
        for aid in self.proposer.acceptor_ids:
            self.outbox.append(Message(
                type=MessageType.ACCEPT,
                sender=self.node_id,
                receiver=aid,
                data={
                    "slot": slot,
                    "proposal_id": pid.to_dict(),
                    "value": value,
                },
            ))

    def _flush_proposer_outbox(self):
        """Move proposer outbox messages to node outbox."""
        while self.proposer.outbox:
            self.outbox.append(self.proposer.outbox.pop(0))

    def _flush_acceptor_outbox(self):
        """Move acceptor outbox messages to node outbox."""
        while self.acceptor.outbox:
            self.outbox.append(self.acceptor.outbox.pop(0))

    # --- Message Handling ---

    def receive(self, msg):
        """Process an incoming message."""
        handlers = {
            MessageType.PREPARE: self._handle_prepare,
            MessageType.PROMISE: self._handle_promise,
            MessageType.ACCEPT: self._handle_accept,
            MessageType.ACCEPTED: self._handle_accepted,
            MessageType.NACK: self._handle_nack,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.HEARTBEAT_ACK: self._handle_heartbeat_ack,
            MessageType.CLIENT_REQUEST: self._handle_client_request,
            MessageType.INSTALL_SNAPSHOT: self._handle_install_snapshot,
            MessageType.INSTALL_SNAPSHOT_RESPONSE: self._handle_install_snapshot_response,
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    def _handle_prepare(self, msg):
        """Acceptor role: handle Prepare."""
        self.acceptor.receive_prepare(msg)
        self._flush_acceptor_outbox()

    def _handle_promise(self, msg):
        """Proposer role: handle Promise."""
        result = self.proposer.receive_promise(msg)
        self._flush_proposer_outbox()
        if result == "phase2_started":
            slot = msg.data["slot"]
            self.leader_prepare_done[slot] = True

    def _handle_accept(self, msg):
        """Acceptor role: handle Accept request."""
        self.acceptor.receive_accept(msg)
        self._flush_acceptor_outbox()
        # Also forward ACCEPTED to learner for ourselves
        slot = msg.data["slot"]
        pid = ProposalID.from_dict(msg.data["proposal_id"])
        value = msg.data["value"]
        a_id, a_val = self.acceptor.get_accepted(slot)
        if a_id == pid:
            self.learner.receive_accepted(slot, pid, value, self.node_id)

    def _handle_accepted(self, msg):
        """Proposer+Learner: handle Accepted response."""
        slot = msg.data["slot"]
        pid = ProposalID.from_dict(msg.data["proposal_id"])
        value = msg.data["value"]

        # Proposer tracking
        self.proposer.receive_accepted(msg)

        # Learner tracking
        decided = self.learner.receive_accepted(slot, pid, value, msg.sender)
        if decided is not None:
            # Broadcast LEARN to all peers
            for peer in self.peer_ids:
                self.outbox.append(Message(
                    type=MessageType.LEARN,
                    sender=self.node_id,
                    receiver=peer,
                    data={"slot": slot, "value": value},
                ))

    def _handle_nack(self, msg):
        """Proposer: handle NACK."""
        self.proposer.receive_nack(msg)
        slot = msg.data["slot"]
        # If we're the leader and got NACKed, we may need to retry
        if self.is_leader and slot in self.proposer.proposals:
            state = self.proposer.proposals[slot]
            if state["nacks"] >= self._quorum_size():
                # Too many NACKs -- another proposer is competing. Retry.
                value = state["value"]
                del self.proposer.proposals[slot]
                self._run_paxos(slot, value)

    # --- Client Interface ---

    def submit(self, command):
        """Submit a client command. Returns slot number or None."""
        if not self.is_leader:
            return None
        slot = self.next_slot
        self.next_slot += 1
        self.pending_requests[slot] = command
        self._run_paxos(slot, command)
        return slot

    def submit_phase2(self, command):
        """Submit using Phase 2 only (stable leader optimization)."""
        if not self.is_leader:
            return None
        slot = self.next_slot
        self.next_slot += 1
        self.pending_requests[slot] = command
        self._run_paxos_phase2_only(slot, command)
        return slot

    def _handle_client_request(self, msg):
        if not self.is_leader:
            self.outbox.append(Message(
                type=MessageType.CLIENT_RESPONSE,
                sender=self.node_id,
                receiver=msg.sender,
                data={"success": False, "leader_id": self.leader_id, "error": "not_leader"},
            ))
            return
        command = msg.data.get("command")
        self.submit(command)

    # --- Snapshot ---

    def take_snapshot(self):
        """Compact decided slots into a snapshot."""
        if self.last_applied <= 0:
            return None
        snap_data = self.state_machine.snapshot()
        self.snapshot = Snapshot(
            last_included_slot=self.last_applied,
            data=snap_data,
            config=list(self.cluster_config),
        )
        # Compact sub-components
        self.acceptor.compact(self.last_applied)
        self.learner.compact(self.last_applied)
        return self.snapshot

    def maybe_compact(self):
        """Auto-compact if decided entries are large enough."""
        if len(self.committed_entries) >= self.snapshot_threshold:
            return self.take_snapshot()
        return None

    def _send_snapshot(self, peer):
        if not self.snapshot:
            return
        self.outbox.append(Message(
            type=MessageType.INSTALL_SNAPSHOT,
            sender=self.node_id,
            receiver=peer,
            data={
                "last_included_slot": self.snapshot.last_included_slot,
                "data": self.snapshot.data,
                "config": self.snapshot.config,
            },
        ))

    def _handle_install_snapshot(self, msg):
        data = msg.data
        snap_slot = data["last_included_slot"]
        if snap_slot <= self.last_applied:
            # We're already past this snapshot
            self.outbox.append(Message(
                type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
                sender=self.node_id,
                receiver=msg.sender,
                data={"success": True, "last_applied": self.last_applied},
            ))
            return

        self.snapshot = Snapshot(
            last_included_slot=snap_slot,
            data=data["data"],
            config=data.get("config", []),
        )
        self.state_machine.restore(data["data"])
        self.last_applied = snap_slot
        self.first_unchosen = max(self.first_unchosen, snap_slot + 1)
        if data.get("config"):
            self.cluster_config = list(data["config"])

        # Mark slots as decided in learner
        self.acceptor.compact(snap_slot)
        self.learner.compact(snap_slot)

        self.outbox.append(Message(
            type=MessageType.INSTALL_SNAPSHOT_RESPONSE,
            sender=self.node_id,
            receiver=msg.sender,
            data={"success": True, "last_applied": snap_slot},
        ))

    def _handle_install_snapshot_response(self, msg):
        pass  # Acknowledged

    # --- Membership Changes ---

    def add_member(self, new_node_id):
        """Add a member via config change command."""
        if not self.is_leader:
            return None
        if new_node_id in self.cluster_config:
            return None
        new_config = self.cluster_config + [new_node_id]
        cmd = {"__type": "config_change", "op": "add_member", "node_id": new_node_id, "config": new_config}
        return self.submit(cmd)

    def remove_member(self, node_id):
        """Remove a member via config change command."""
        if not self.is_leader:
            return None
        if node_id not in self.cluster_config:
            return None
        new_config = [n for n in self.cluster_config if n != node_id]
        cmd = {"__type": "config_change", "op": "remove_member", "node_id": node_id, "config": new_config}
        return self.submit(cmd)

    def _apply_config_change(self, command):
        if not isinstance(command, dict):
            return
        new_config = command.get("config", [])
        self.cluster_config = new_config
        # Update quorum sizes
        q = len(self.cluster_config) // 2 + 1
        self.proposer.quorum_size = q
        self.learner.quorum_size = q
        self.proposer.acceptor_ids = list(self.cluster_config)

        node_id = command.get("node_id")
        op = command.get("op")
        if op == "add_member" and node_id not in self.peer_ids and node_id != self.node_id:
            self.peer_ids.append(node_id)
        elif op == "remove_member":
            if node_id in self.peer_ids:
                self.peer_ids.remove(node_id)
            if node_id == self.node_id and self.is_leader:
                self._step_down()

    # --- State ---

    def get_state(self):
        return {
            "node_id": self.node_id,
            "is_leader": self.is_leader,
            "leader_id": self.leader_id,
            "last_applied": self.last_applied,
            "next_slot": self.next_slot,
            "first_unchosen": self.first_unchosen,
            "cluster_config": self.cluster_config,
            "state_machine": self.state_machine.data,
        }


# --- Network Simulator ---

class Network:
    """Simulates message passing between Paxos nodes with partition support."""

    def __init__(self):
        self.nodes = {}  # id -> MultiPaxosNode
        self.partitions = set()  # frozenset pairs
        self.message_queue = []
        self.delivered = []
        self.dropped = []

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node_id):
        self.nodes.pop(node_id, None)

    def partition(self, node_a, node_b):
        self.partitions.add(frozenset({node_a, node_b}))

    def heal(self, node_a=None, node_b=None):
        if node_a is None and node_b is None:
            self.partitions.clear()
        else:
            self.partitions.discard(frozenset({node_a, node_b}))

    def is_partitioned(self, a, b):
        return frozenset({a, b}) in self.partitions

    def isolate(self, node_id):
        for other in self.nodes:
            if other != node_id:
                self.partition(node_id, other)

    def tick(self, elapsed_ms=1):
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
        for _ in range(ticks):
            self.tick(tick_ms)

    def find_leader(self):
        leaders = [n for n in self.nodes.values() if n.is_leader]
        if len(leaders) == 1:
            return leaders[0]
        return None

    def wait_for_leader(self, max_ticks=2000, tick_ms=1):
        for _ in range(max_ticks):
            self.tick(tick_ms)
            leader = self.find_leader()
            if leader:
                return leader
        return None

    def wait_for_decision(self, slot, max_ticks=2000, tick_ms=1):
        """Wait until at least one node has decided a value for the slot."""
        for _ in range(max_ticks):
            self.tick(tick_ms)
            for node in self.nodes.values():
                if node.learner.is_decided(slot):
                    return node.learner.get_decided(slot)
        return None

    def wait_for_apply(self, slot, count=None, max_ticks=2000, tick_ms=1):
        """Wait until count nodes have applied up to slot."""
        if count is None:
            count = len(self.nodes)
        for _ in range(max_ticks):
            self.tick(tick_ms)
            applied = sum(1 for n in self.nodes.values() if n.last_applied >= slot)
            if applied >= count:
                return True
        return False

    def wait_for_replication(self, slot, count=None, max_ticks=2000, tick_ms=1):
        """Alias for wait_for_apply."""
        return self.wait_for_apply(slot, count, max_ticks, tick_ms)


def create_cluster(node_ids, **kwargs):
    """Create a Multi-Paxos cluster."""
    net = Network()
    for nid in node_ids:
        peers = [p for p in node_ids if p != nid]
        node = MultiPaxosNode(nid, peers, **kwargs)
        net.add_node(node)
    return net


# --- Basic Paxos (single-decree) convenience ---

class BasicPaxos:
    """
    Single-decree Paxos for one consensus decision.
    Uses Proposer/Acceptor/Learner directly.
    """

    def __init__(self, node_ids):
        self.node_ids = list(node_ids)
        self.quorum_size = len(node_ids) // 2 + 1
        self.acceptors = {nid: Acceptor(nid) for nid in node_ids}
        self.learners = {nid: Learner(nid, self.quorum_size) for nid in node_ids}
        self.proposers = {}
        self.decided_value = None
        self.message_log = []

    def create_proposer(self, node_id):
        """Create a proposer on a specific node."""
        p = Proposer(node_id, self.node_ids, self.quorum_size)
        self.proposers[node_id] = p
        return p

    def run_round(self, proposer_id, slot, value):
        """Run a complete Paxos round synchronously. Returns decided value."""
        proposer = self.proposers.get(proposer_id)
        if not proposer:
            proposer = self.create_proposer(proposer_id)

        # Phase 1: Prepare
        proposer.propose(slot, value)
        messages = list(proposer.outbox)
        proposer.outbox.clear()
        self.message_log.extend(messages)

        # Deliver Prepares to acceptors
        responses = []
        for msg in messages:
            if msg.type == MessageType.PREPARE:
                acc = self.acceptors[msg.receiver]
                acc.receive_prepare(msg)
                responses.extend(acc.outbox)
                acc.outbox.clear()

        self.message_log.extend(responses)

        # Deliver Promises to proposer
        for msg in responses:
            if msg.type == MessageType.PROMISE:
                proposer.receive_promise(msg)

        phase2_msgs = list(proposer.outbox)
        proposer.outbox.clear()
        self.message_log.extend(phase2_msgs)

        # Phase 2: Accept
        accept_responses = []
        for msg in phase2_msgs:
            if msg.type == MessageType.ACCEPT:
                acc = self.acceptors[msg.receiver]
                acc.receive_accept(msg)
                accept_responses.extend(acc.outbox)
                acc.outbox.clear()

        self.message_log.extend(accept_responses)

        # Deliver Accepted to proposer and learners
        for msg in accept_responses:
            if msg.type == MessageType.ACCEPTED:
                proposer.receive_accepted(msg)
                pid = ProposalID.from_dict(msg.data["proposal_id"])
                val = msg.data["value"]
                for learner in self.learners.values():
                    decided = learner.receive_accepted(slot, pid, val, msg.sender)
                    if decided is not None:
                        self.decided_value = decided

        return self.decided_value

    def get_decided(self, slot=1):
        """Get decided value from any learner."""
        for learner in self.learners.values():
            if learner.is_decided(slot):
                return learner.get_decided(slot)
        return None
