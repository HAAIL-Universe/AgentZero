"""
C208: Paxos / Multi-Paxos Consensus Protocol
=============================================
Complete implementation of Lamport's Paxos consensus algorithm.

Components:
- SingleDecreePaxos: Basic Paxos for agreeing on a single value
  - Proposer: Generates proposals with unique ballot numbers
  - Acceptor: Accepts/rejects proposals based on promises
  - Learner: Learns decided values from acceptor majorities
- MultiPaxos: Extends single-decree to ordered log of commands
  - Stable leader optimization (skip Phase 1 after election)
  - Log slots with independent Paxos instances
  - Commit index tracking and state machine application
- PaxosCluster: Simulated cluster for testing (no real network)
- Distinguished proposer (leader election via ballots)
- Noop gap-filling for log holes
- Acceptor persistence (durable state)
- Comparison with Raft (same problem, different approach)

Based on "Paxos Made Simple" (Lamport, 2001) and
"Paxos Made Moderately Complex" (van Renesse & Altinbuken, 2015).
"""

import time
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


# --- Enums ---

class MessageType(Enum):
    # Phase 1
    PREPARE = "prepare"
    PROMISE = "promise"
    # Phase 2
    ACCEPT = "accept"
    ACCEPTED = "accepted"
    # Learning
    DECIDE = "decide"
    # Multi-Paxos
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"
    # Leader
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"


class ProposalStatus(Enum):
    PENDING = "pending"
    PROMISED = "promised"
    ACCEPTED = "accepted"
    DECIDED = "decided"
    REJECTED = "rejected"


# --- Data Structures ---

@dataclass
class BallotNumber:
    """Unique, totally-ordered ballot number = (round, proposer_id)."""
    round: int
    proposer_id: str

    def __lt__(self, other):
        if not isinstance(other, BallotNumber):
            return NotImplemented
        return (self.round, self.proposer_id) < (other.round, other.proposer_id)

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if not isinstance(other, BallotNumber):
            return NotImplemented
        return (self.round, self.proposer_id) > (other.round, other.proposer_id)

    def __ge__(self, other):
        return self == other or self > other

    def __eq__(self, other):
        if not isinstance(other, BallotNumber):
            return NotImplemented
        return (self.round, self.proposer_id) == (other.round, other.proposer_id)

    def __hash__(self):
        return hash((self.round, self.proposer_id))

    def to_dict(self):
        return {"round": self.round, "proposer_id": self.proposer_id}

    @staticmethod
    def from_dict(d):
        return BallotNumber(d["round"], d["proposer_id"])


@dataclass
class Proposal:
    """A proposal in Paxos: ballot + value."""
    ballot: BallotNumber
    value: Any
    slot: int = 0  # For Multi-Paxos

    def to_dict(self):
        return {"ballot": self.ballot.to_dict(), "value": self.value, "slot": self.slot}

    @staticmethod
    def from_dict(d):
        return Proposal(BallotNumber.from_dict(d["ballot"]), d["value"], d.get("slot", 0))


@dataclass
class Message:
    type: MessageType
    src: str
    dst: str
    ballot: Optional[BallotNumber] = None
    data: dict = field(default_factory=dict)


# --- Acceptor ---

class AcceptorState:
    """Persistent state for a Paxos acceptor."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        # Highest ballot promised (will not accept lower)
        self.promised: Optional[BallotNumber] = None
        # Highest ballot accepted with its value
        self.accepted_ballot: Optional[BallotNumber] = None
        self.accepted_value: Any = None
        # For Multi-Paxos: per-slot state
        self.slots: dict[int, dict] = {}  # slot -> {promised, accepted_ballot, accepted_value}

    def handle_prepare(self, ballot: BallotNumber, slot: int = 0) -> dict:
        """Phase 1b: respond to Prepare with Promise or reject."""
        state = self._get_slot(slot)
        if state["promised"] is not None and ballot < state["promised"]:
            return {"ok": False, "promised": state["promised"]}

        state["promised"] = ballot
        self._set_slot(slot, state)

        return {
            "ok": True,
            "accepted_ballot": state["accepted_ballot"],
            "accepted_value": state["accepted_value"],
        }

    def handle_accept(self, ballot: BallotNumber, value: Any, slot: int = 0) -> dict:
        """Phase 2b: accept if ballot >= promised."""
        state = self._get_slot(slot)
        if state["promised"] is not None and ballot < state["promised"]:
            return {"ok": False, "promised": state["promised"]}

        state["promised"] = ballot
        state["accepted_ballot"] = ballot
        state["accepted_value"] = value
        self._set_slot(slot, state)

        return {"ok": True}

    def _get_slot(self, slot: int) -> dict:
        if slot == 0:
            return {
                "promised": self.promised,
                "accepted_ballot": self.accepted_ballot,
                "accepted_value": self.accepted_value,
            }
        if slot not in self.slots:
            self.slots[slot] = {
                "promised": None,
                "accepted_ballot": None,
                "accepted_value": None,
            }
        return self.slots[slot]

    def _set_slot(self, slot: int, state: dict):
        if slot == 0:
            self.promised = state["promised"]
            self.accepted_ballot = state["accepted_ballot"]
            self.accepted_value = state["accepted_value"]
        else:
            self.slots[slot] = state

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "promised": self.promised.to_dict() if self.promised else None,
            "accepted_ballot": self.accepted_ballot.to_dict() if self.accepted_ballot else None,
            "accepted_value": self.accepted_value,
            "slots": {
                k: {
                    "promised": v["promised"].to_dict() if v["promised"] else None,
                    "accepted_ballot": v["accepted_ballot"].to_dict() if v["accepted_ballot"] else None,
                    "accepted_value": v["accepted_value"],
                }
                for k, v in self.slots.items()
            },
        }


# --- Single-Decree Paxos ---

class SingleDecreePaxos:
    """
    Basic Paxos: agree on a single value among a group of nodes.

    Roles: proposer, acceptor, learner (all co-located in each node).
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.all_nodes = [node_id] + list(peers)
        self.quorum_size = len(self.all_nodes) // 2 + 1

        # Proposer state
        self.current_round = 0
        self.proposal_value: Any = None
        self.proposal_status = ProposalStatus.PENDING
        self.promises_received: dict[str, dict] = {}

        # Acceptor state
        self.acceptor = AcceptorState(node_id)

        # Learner state
        self.decided_value: Any = None
        self._decided = False
        self.accepted_notifications: dict[str, dict] = {}  # node_id -> {ballot, value}

        # Message outbox
        self.outbox: list[Message] = []

    @property
    def is_decided(self) -> bool:
        return self._decided

    def propose(self, value: Any) -> BallotNumber:
        """Phase 1a: send Prepare to all acceptors."""
        self.current_round += 1
        ballot = BallotNumber(self.current_round, self.node_id)
        self.proposal_value = value
        self.proposal_status = ProposalStatus.PENDING
        self.promises_received = {}

        for node in self.all_nodes:
            self.outbox.append(Message(
                type=MessageType.PREPARE,
                src=self.node_id,
                dst=node,
                ballot=ballot,
            ))

        return ballot

    def receive(self, msg: Message):
        """Process an incoming message."""
        handlers = {
            MessageType.PREPARE: self._handle_prepare,
            MessageType.PROMISE: self._handle_promise,
            MessageType.ACCEPT: self._handle_accept,
            MessageType.ACCEPTED: self._handle_accepted,
            MessageType.DECIDE: self._handle_decide,
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    def _handle_prepare(self, msg: Message):
        """Acceptor: respond to Prepare."""
        result = self.acceptor.handle_prepare(msg.ballot)
        data = dict(result)
        if data.get("accepted_ballot"):
            data["accepted_ballot"] = data["accepted_ballot"].to_dict()
        if data.get("promised"):
            data["promised"] = data["promised"].to_dict()
        self.outbox.append(Message(
            type=MessageType.PROMISE,
            src=self.node_id,
            dst=msg.src,
            ballot=msg.ballot,
            data=data,
        ))

    def _handle_promise(self, msg: Message):
        """Proposer: collect promises, then send Accept."""
        if not msg.data.get("ok"):
            return

        ballot = msg.ballot
        expected = BallotNumber(self.current_round, self.node_id)
        if ballot != expected:
            return

        self.promises_received[msg.src] = msg.data

        if len(self.promises_received) >= self.quorum_size:
            if self.proposal_status in (ProposalStatus.PROMISED, ProposalStatus.ACCEPTED):
                return
            self.proposal_status = ProposalStatus.PROMISED

            # Must use highest-numbered accepted value (Paxos constraint)
            value = self._pick_value()

            # Phase 2a: send Accept to all
            for node in self.all_nodes:
                self.outbox.append(Message(
                    type=MessageType.ACCEPT,
                    src=self.node_id,
                    dst=node,
                    ballot=ballot,
                    data={"value": value},
                ))

    def _pick_value(self) -> Any:
        """Pick value: use highest previously accepted, or own proposal."""
        highest_ballot = None
        highest_value = None
        found = False

        for data in self.promises_received.values():
            ab = data.get("accepted_ballot")
            if ab is not None:
                if isinstance(ab, dict):
                    ab = BallotNumber.from_dict(ab)
                if highest_ballot is None or ab > highest_ballot:
                    highest_ballot = ab
                    highest_value = data["accepted_value"]
                    found = True

        if found:
            return highest_value
        return self.proposal_value

    def _handle_accept(self, msg: Message):
        """Acceptor: accept or reject."""
        result = self.acceptor.handle_accept(msg.ballot, msg.data["value"])
        data = dict(result)
        data["value"] = msg.data["value"]
        if data.get("promised"):
            data["promised"] = data["promised"].to_dict()
        self.outbox.append(Message(
            type=MessageType.ACCEPTED,
            src=self.node_id,
            dst=msg.src,
            ballot=msg.ballot,
            data=data,
        ))

    def _handle_accepted(self, msg: Message):
        """Learner: collect Accepted, decide on quorum."""
        if not msg.data.get("ok"):
            return

        self.accepted_notifications[msg.src] = {
            "ballot": msg.ballot,
            "value": msg.data["value"],
        }

        # Count acceptances for this ballot+value
        count = sum(
            1 for info in self.accepted_notifications.values()
            if info["ballot"] == msg.ballot and info["value"] == msg.data["value"]
        )

        if count >= self.quorum_size and not self.is_decided:
            self.decided_value = msg.data["value"]
            self._decided = True
            self.proposal_status = ProposalStatus.DECIDED

            # Broadcast decision
            for node in self.all_nodes:
                self.outbox.append(Message(
                    type=MessageType.DECIDE,
                    src=self.node_id,
                    dst=node,
                    ballot=msg.ballot,
                    data={"value": msg.data["value"]},
                ))

    def _handle_decide(self, msg: Message):
        """Learn a decided value."""
        if not self.is_decided:
            self.decided_value = msg.data["value"]
            self._decided = True
            self.proposal_status = ProposalStatus.DECIDED


# --- Multi-Paxos ---

class MultiPaxosNode:
    """
    Multi-Paxos: agree on an ordered sequence of values (log).

    Optimizations over basic Paxos:
    - Stable leader skips Phase 1 for subsequent slots
    - Heartbeats for leader election
    - Noop gap-filling
    - Batch commit notifications
    """

    def __init__(self, node_id: str, peers: list[str]):
        self.node_id = node_id
        self.peers = peers
        self.all_nodes = [node_id] + list(peers)
        self.quorum_size = len(self.all_nodes) // 2 + 1

        # Leader state
        self.is_leader = False
        self.leader_id: Optional[str] = None
        self.leader_ballot: Optional[BallotNumber] = None
        self.current_round = 0

        # Per-slot acceptor state
        self.acceptor = AcceptorState(node_id)

        # Log: slot -> decided value
        self.log: dict[int, Any] = {}
        self.next_slot = 1
        self.commit_index = 0

        # State machine
        self.state_machine: dict[str, Any] = {}
        self.last_applied = 0

        # Phase 1 tracking (leader election)
        self.phase1_promises: dict[str, dict] = {}
        self.phase1_complete = False

        # Phase 2 tracking per slot
        self.slot_accepts: dict[int, dict[str, bool]] = {}  # slot -> {node_id -> accepted}
        self.slot_values: dict[int, Any] = {}  # slot -> proposed value
        self.slot_ballots: dict[int, BallotNumber] = {}  # slot -> ballot used

        # Heartbeat tracking
        self.last_heartbeat_time = 0.0
        self.heartbeat_responses: dict[str, float] = {}

        # Client pending requests
        self.pending_requests: list[dict] = []  # [{client_id, command, slot}]

        # Message outbox
        self.outbox: list[Message] = []

    @property
    def is_decided(self) -> bool:
        """Check if at least one slot has been decided."""
        return len(self.log) > 0

    def get_decided_values(self) -> list:
        """Return decided values in slot order."""
        if not self.log:
            return []
        max_slot = max(self.log.keys())
        return [self.log.get(i) for i in range(1, max_slot + 1)]

    # --- Leader Election ---

    def start_election(self) -> BallotNumber:
        """Become a candidate: run Phase 1 across all slots."""
        self.current_round += 1
        ballot = BallotNumber(self.current_round, self.node_id)
        self.leader_ballot = ballot
        self.phase1_promises = {}
        self.phase1_complete = False
        self.is_leader = False

        # Send Prepare to all (including self)
        for node in self.all_nodes:
            self.outbox.append(Message(
                type=MessageType.PREPARE,
                src=self.node_id,
                dst=node,
                ballot=ballot,
                data={"type": "election"},
            ))

        return ballot

    def submit_command(self, command: Any, client_id: str = "client") -> Optional[int]:
        """Submit a client command (leader only). Returns slot or None."""
        if not self.is_leader:
            return None

        slot = self.next_slot
        self.next_slot += 1

        self.pending_requests.append({
            "client_id": client_id,
            "command": command,
            "slot": slot,
        })

        self._propose_slot(slot, command)
        return slot

    def _propose_slot(self, slot: int, value: Any):
        """Phase 2a for a specific slot (leader skips Phase 1 if stable)."""
        ballot = self.leader_ballot
        self.slot_values[slot] = value
        self.slot_ballots[slot] = ballot
        self.slot_accepts[slot] = {}

        for node in self.all_nodes:
            self.outbox.append(Message(
                type=MessageType.ACCEPT,
                src=self.node_id,
                dst=node,
                ballot=ballot,
                data={"value": value, "slot": slot},
            ))

    def send_heartbeat(self):
        """Leader heartbeat to maintain authority."""
        if not self.is_leader:
            return
        self.last_heartbeat_time = time.time()
        for peer in self.peers:
            self.outbox.append(Message(
                type=MessageType.HEARTBEAT,
                src=self.node_id,
                dst=peer,
                ballot=self.leader_ballot,
                data={"commit_index": self.commit_index},
            ))

    # --- Message Handling ---

    def receive(self, msg: Message):
        """Process an incoming message."""
        handlers = {
            MessageType.PREPARE: self._handle_prepare,
            MessageType.PROMISE: self._handle_promise,
            MessageType.ACCEPT: self._handle_accept,
            MessageType.ACCEPTED: self._handle_accepted,
            MessageType.DECIDE: self._handle_decide,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.HEARTBEAT_RESPONSE: self._handle_heartbeat_response,
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    def _handle_prepare(self, msg: Message):
        """Acceptor: handle Phase 1 Prepare for election."""
        # Check against highest round we've seen
        ballot = msg.ballot
        # Collect all slot state to send back
        slot_state = {}
        all_ok = True

        # Check global promise first
        if self.acceptor.promised is not None and ballot < self.acceptor.promised:
            all_ok = False

        if all_ok:
            self.acceptor.promised = ballot
            # If this node was leader, step down
            if self.is_leader and self.leader_id == self.node_id and (
                self.leader_ballot is None or ballot > self.leader_ballot
            ):
                self.is_leader = False

            self.leader_id = msg.src
            self.leader_ballot = ballot

            # Gather accepted values from all slots
            for slot_num, slot_data in self.acceptor.slots.items():
                if slot_data["accepted_value"] is not None:
                    ab = slot_data["accepted_ballot"]
                    slot_state[slot_num] = {
                        "accepted_ballot": ab.to_dict() if ab else None,
                        "accepted_value": slot_data["accepted_value"],
                    }
            # Also check slot 0 (single-decree state)
            if self.acceptor.accepted_value is not None:
                ab = self.acceptor.accepted_ballot
                slot_state[0] = {
                    "accepted_ballot": ab.to_dict() if ab else None,
                    "accepted_value": self.acceptor.accepted_value,
                }

        self.outbox.append(Message(
            type=MessageType.PROMISE,
            src=self.node_id,
            dst=msg.src,
            ballot=ballot,
            data={
                "ok": all_ok,
                "slot_state": slot_state,
                "next_slot": self.next_slot,
            },
        ))

    def _handle_promise(self, msg: Message):
        """Proposer: collect Phase 1 promises for election."""
        if not msg.data.get("ok"):
            return

        ballot = msg.ballot
        if self.leader_ballot is None or ballot != self.leader_ballot:
            return

        self.phase1_promises[msg.src] = msg.data

        if len(self.phase1_promises) >= self.quorum_size and not self.phase1_complete:
            self.phase1_complete = True
            self.is_leader = True
            self.leader_id = self.node_id

            # Determine next_slot from max seen
            max_next = self.next_slot
            for data in self.phase1_promises.values():
                ns = data.get("next_slot", 1)
                if ns > max_next:
                    max_next = ns

            # Re-propose any previously accepted values (gap filling)
            accepted_slots: dict[int, tuple] = {}
            for data in self.phase1_promises.values():
                for slot_str, sdata in data.get("slot_state", {}).items():
                    slot_num = int(slot_str)
                    if slot_num == 0:
                        continue
                    ab = sdata.get("accepted_ballot")
                    if ab is not None:
                        if isinstance(ab, dict):
                            ab = BallotNumber.from_dict(ab)
                        if slot_num not in accepted_slots or ab > accepted_slots[slot_num][0]:
                            accepted_slots[slot_num] = (ab, sdata["accepted_value"])

            # Fill gaps with noops, re-propose accepted values
            for slot_num in range(1, max_next):
                if slot_num in self.log:
                    continue  # Already decided
                if slot_num in accepted_slots:
                    self._propose_slot(slot_num, accepted_slots[slot_num][1])
                else:
                    self._propose_slot(slot_num, {"type": "noop"})

            self.next_slot = max_next

    def _handle_accept(self, msg: Message):
        """Acceptor: handle Phase 2 Accept."""
        slot = msg.data.get("slot", 0)
        result = self.acceptor.handle_accept(msg.ballot, msg.data["value"], slot)
        data = dict(result)
        data["value"] = msg.data["value"]
        data["slot"] = slot
        if data.get("promised"):
            data["promised"] = data["promised"].to_dict()
        self.outbox.append(Message(
            type=MessageType.ACCEPTED,
            src=self.node_id,
            dst=msg.src,
            ballot=msg.ballot,
            data=data,
        ))

    def _handle_accepted(self, msg: Message):
        """Leader/Learner: collect Accepted, decide on quorum."""
        if not msg.data.get("ok"):
            return

        slot = msg.data.get("slot", 0)
        if slot == 0:
            return

        if slot not in self.slot_accepts:
            self.slot_accepts[slot] = {}
        self.slot_accepts[slot][msg.src] = True

        if len(self.slot_accepts[slot]) >= self.quorum_size and slot not in self.log:
            value = msg.data["value"]
            self.log[slot] = value

            # Broadcast decision
            for node in self.all_nodes:
                self.outbox.append(Message(
                    type=MessageType.DECIDE,
                    src=self.node_id,
                    dst=node,
                    ballot=msg.ballot,
                    data={"value": value, "slot": slot},
                ))

            # Advance commit index
            self._advance_commit_index()

    def _handle_decide(self, msg: Message):
        """Learn a decided value for a slot."""
        slot = msg.data.get("slot", 0)
        if slot == 0:
            return
        if slot not in self.log:
            self.log[slot] = msg.data["value"]
            self._advance_commit_index()

    def _handle_heartbeat(self, msg: Message):
        """Follower: acknowledge leader heartbeat."""
        if self.leader_ballot is None or msg.ballot >= self.leader_ballot:
            self.leader_id = msg.src
            self.leader_ballot = msg.ballot
            self.is_leader = False

            # Update commit index from leader
            leader_commit = msg.data.get("commit_index", 0)
            if leader_commit > self.commit_index:
                self.commit_index = leader_commit
                self._apply_committed()

        self.outbox.append(Message(
            type=MessageType.HEARTBEAT_RESPONSE,
            src=self.node_id,
            dst=msg.src,
            ballot=msg.ballot,
        ))

    def _handle_heartbeat_response(self, msg: Message):
        """Leader: track follower liveness."""
        self.heartbeat_responses[msg.src] = time.time()

    def _advance_commit_index(self):
        """Advance commit index through consecutive decided slots."""
        while self.commit_index + 1 in self.log:
            self.commit_index += 1
        self._apply_committed()

    def _apply_committed(self):
        """Apply committed log entries to state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log.get(self.last_applied)
            if entry is not None:
                self._apply_command(entry)

    def _apply_command(self, command: Any):
        """Apply a command to the key-value state machine."""
        if isinstance(command, dict):
            if command.get("type") == "noop":
                return
            if command.get("type") == "set":
                self.state_machine[command["key"]] = command["value"]
            elif command.get("type") == "delete":
                self.state_machine.pop(command.get("key"), None)


# --- Paxos Cluster (Simulation) ---

class PaxosCluster:
    """
    Simulated Paxos cluster for testing.

    Delivers messages between nodes with optional network partitions
    and message drops.
    """

    def __init__(self, node_ids: list[str], multi: bool = False):
        self.node_ids = node_ids
        self.multi = multi

        if multi:
            self.nodes: dict[str, Any] = {}
            for nid in node_ids:
                peers = [p for p in node_ids if p != nid]
                self.nodes[nid] = MultiPaxosNode(nid, peers)
        else:
            self.nodes = {}
            for nid in node_ids:
                peers = [p for p in node_ids if p != nid]
                self.nodes[nid] = SingleDecreePaxos(nid, peers)

        # Network simulation
        self.partitioned: set[tuple[str, str]] = set()
        self.dropped_messages = 0
        self.delivered_messages = 0
        self.message_log: list[dict] = []

    def partition(self, group1: list[str], group2: list[str]):
        """Create a network partition between two groups."""
        for a in group1:
            for b in group2:
                self.partitioned.add((a, b))
                self.partitioned.add((b, a))

    def heal_partition(self):
        """Remove all partitions."""
        self.partitioned.clear()

    def is_partitioned(self, src: str, dst: str) -> bool:
        return (src, dst) in self.partitioned

    def deliver_messages(self, max_rounds: int = 100) -> int:
        """Deliver all pending messages until quiescence or max_rounds."""
        total = 0
        for _ in range(max_rounds):
            messages = []
            for node in self.nodes.values():
                while node.outbox:
                    messages.append(node.outbox.pop(0))

            if not messages:
                break

            for msg in messages:
                if self.is_partitioned(msg.src, msg.dst):
                    self.dropped_messages += 1
                    continue
                dst_node = self.nodes.get(msg.dst)
                if dst_node:
                    dst_node.receive(msg)
                    self.delivered_messages += 1
                    total += 1
                    self.message_log.append({
                        "type": msg.type.value,
                        "src": msg.src,
                        "dst": msg.dst,
                    })

        return total

    def run_election(self, candidate: str) -> bool:
        """Run leader election for Multi-Paxos."""
        if not self.multi:
            return False
        node = self.nodes[candidate]
        node.start_election()
        self.deliver_messages()
        return node.is_leader

    def propose_single(self, proposer: str, value: Any) -> bool:
        """Run a single-decree proposal to completion."""
        if self.multi:
            return False
        node = self.nodes[proposer]
        node.propose(value)
        self.deliver_messages()
        return node.is_decided

    def submit(self, leader: str, command: Any, client_id: str = "client") -> Optional[int]:
        """Submit a command via the Multi-Paxos leader."""
        if not self.multi:
            return None
        node = self.nodes[leader]
        slot = node.submit_command(command, client_id)
        self.deliver_messages()
        return slot

    def get_decided_value(self, node_id: str) -> Any:
        """Get single-decree decided value."""
        return self.nodes[node_id].decided_value

    def get_log(self, node_id: str) -> dict:
        """Get Multi-Paxos log."""
        return self.nodes[node_id].log

    def get_state_machine(self, node_id: str) -> dict:
        """Get Multi-Paxos state machine."""
        return self.nodes[node_id].state_machine

    def get_leader(self) -> Optional[str]:
        """Find the current leader."""
        for nid, node in self.nodes.items():
            if hasattr(node, 'is_leader') and node.is_leader:
                return nid
        return None

    def all_decided_same(self) -> bool:
        """Check if all non-partitioned nodes agree on decided values."""
        values = set()
        for node in self.nodes.values():
            if isinstance(node, SingleDecreePaxos):
                if node.is_decided:
                    v = node.decided_value
                    values.add(json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        return len(values) <= 1

    def all_logs_consistent(self) -> bool:
        """Check Multi-Paxos log consistency (all decided slots agree)."""
        if not self.multi:
            return True
        all_slots = set()
        for node in self.nodes.values():
            all_slots.update(node.log.keys())

        for slot in all_slots:
            values = set()
            for node in self.nodes.values():
                if slot in node.log:
                    v = node.log[slot]
                    values.add(json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            if len(values) > 1:
                return False
        return True


# --- Flexible Paxos ---

class FlexiblePaxos:
    """
    Flexible Paxos: Phase 1 and Phase 2 quorums don't need to be majorities,
    they just need to intersect (Q1 + Q2 > N).

    This allows trading Phase 1 availability for Phase 2 performance
    (e.g., Phase 2 quorum of 1 with Phase 1 quorum of N).
    """

    def __init__(self, node_id: str, peers: list[str],
                 phase1_quorum: Optional[int] = None,
                 phase2_quorum: Optional[int] = None):
        self.node_id = node_id
        self.peers = peers
        self.all_nodes = [node_id] + list(peers)
        n = len(self.all_nodes)

        # Flexible quorums: Q1 + Q2 > N
        self.phase1_quorum = phase1_quorum or (n // 2 + 1)
        self.phase2_quorum = phase2_quorum or (n // 2 + 1)

        if self.phase1_quorum + self.phase2_quorum <= n:
            raise ValueError(
                f"Quorums must intersect: Q1({self.phase1_quorum}) + Q2({self.phase2_quorum}) "
                f"must be > N({n})"
            )

        self.acceptor = AcceptorState(node_id)
        self.current_round = 0
        self.promises_received: dict[str, dict] = {}
        self.accepts_received: dict[str, bool] = {}
        self.proposal_value: Any = None
        self.decided_value: Any = None
        self._decided = False
        self.outbox: list[Message] = []

    @property
    def is_decided(self) -> bool:
        return self._decided

    def propose(self, value: Any) -> BallotNumber:
        self.current_round += 1
        ballot = BallotNumber(self.current_round, self.node_id)
        self.proposal_value = value
        self.promises_received = {}
        self.accepts_received = {}

        for node in self.all_nodes:
            self.outbox.append(Message(
                type=MessageType.PREPARE,
                src=self.node_id,
                dst=node,
                ballot=ballot,
            ))
        return ballot

    def receive(self, msg: Message):
        handlers = {
            MessageType.PREPARE: self._handle_prepare,
            MessageType.PROMISE: self._handle_promise,
            MessageType.ACCEPT: self._handle_accept,
            MessageType.ACCEPTED: self._handle_accepted,
            MessageType.DECIDE: self._handle_decide,
        }
        handler = handlers.get(msg.type)
        if handler:
            handler(msg)

    def _handle_prepare(self, msg: Message):
        result = self.acceptor.handle_prepare(msg.ballot)
        data = dict(result)
        if data.get("accepted_ballot"):
            data["accepted_ballot"] = data["accepted_ballot"].to_dict()
        if data.get("promised"):
            data["promised"] = data["promised"].to_dict()
        self.outbox.append(Message(
            type=MessageType.PROMISE,
            src=self.node_id,
            dst=msg.src,
            ballot=msg.ballot,
            data=data,
        ))

    def _handle_promise(self, msg: Message):
        if not msg.data.get("ok"):
            return
        ballot = msg.ballot
        expected = BallotNumber(self.current_round, self.node_id)
        if ballot != expected:
            return

        self.promises_received[msg.src] = msg.data

        if len(self.promises_received) >= self.phase1_quorum:
            # Pick value
            highest_ballot = None
            highest_value = None
            for data in self.promises_received.values():
                ab = data.get("accepted_ballot")
                if ab is not None:
                    if isinstance(ab, dict):
                        ab = BallotNumber.from_dict(ab)
                    if highest_ballot is None or ab > highest_ballot:
                        highest_ballot = ab
                        highest_value = data["accepted_value"]

            value = highest_value if highest_value is not None else self.proposal_value

            for node in self.all_nodes:
                self.outbox.append(Message(
                    type=MessageType.ACCEPT,
                    src=self.node_id,
                    dst=node,
                    ballot=ballot,
                    data={"value": value},
                ))

    def _handle_accept(self, msg: Message):
        result = self.acceptor.handle_accept(msg.ballot, msg.data["value"])
        data = dict(result)
        data["value"] = msg.data["value"]
        if data.get("promised"):
            data["promised"] = data["promised"].to_dict()
        self.outbox.append(Message(
            type=MessageType.ACCEPTED,
            src=self.node_id,
            dst=msg.src,
            ballot=msg.ballot,
            data=data,
        ))

    def _handle_accepted(self, msg: Message):
        if not msg.data.get("ok"):
            return
        self.accepts_received[msg.src] = True

        if len(self.accepts_received) >= self.phase2_quorum and not self.is_decided:
            self.decided_value = msg.data["value"]
            self._decided = True
            for node in self.all_nodes:
                self.outbox.append(Message(
                    type=MessageType.DECIDE,
                    src=self.node_id,
                    dst=node,
                    ballot=msg.ballot,
                    data={"value": msg.data["value"]},
                ))

    def _handle_decide(self, msg: Message):
        if not self.is_decided:
            self.decided_value = msg.data["value"]
            self._decided = True


# --- CatchUp Protocol ---

class CatchUpProtocol:
    """
    Catch-up protocol for nodes that fall behind.
    A lagging node can request missing log entries from the leader.
    """

    def __init__(self, node: MultiPaxosNode):
        self.node = node
        self.pending_catchup = False

    def needs_catchup(self, leader_commit: int) -> bool:
        """Check if this node is behind the leader."""
        return leader_commit > self.node.commit_index

    def request_catchup(self, leader_id: str, from_slot: int):
        """Request missing entries from leader."""
        self.pending_catchup = True
        self.node.outbox.append(Message(
            type=MessageType.CLIENT_REQUEST,
            src=self.node.node_id,
            dst=leader_id,
            data={"type": "catchup", "from_slot": from_slot},
        ))

    def handle_catchup_response(self, entries: dict[int, Any]):
        """Apply received entries."""
        for slot, value in entries.items():
            if slot not in self.node.log:
                self.node.log[slot] = value
        self.node._advance_commit_index()
        self.pending_catchup = False

    @staticmethod
    def serve_catchup(leader: MultiPaxosNode, from_slot: int) -> dict[int, Any]:
        """Leader serves missing entries to a follower."""
        entries = {}
        for slot in range(from_slot, leader.commit_index + 1):
            if slot in leader.log:
                entries[slot] = leader.log[slot]
        return entries


# --- Paxos Statistics ---

class PaxosStats:
    """Track Paxos protocol statistics."""

    def __init__(self):
        self.proposals_started = 0
        self.proposals_decided = 0
        self.proposals_rejected = 0
        self.messages_sent = 0
        self.messages_dropped = 0
        self.elections_held = 0
        self.leader_changes = 0
        self.noops_proposed = 0
        self.catchups_requested = 0
        self.rounds_to_decide: list[int] = []

    def record_proposal(self, decided: bool, rounds: int = 1):
        self.proposals_started += 1
        if decided:
            self.proposals_decided += 1
            self.rounds_to_decide.append(rounds)
        else:
            self.proposals_rejected += 1

    def record_election(self, success: bool):
        self.elections_held += 1
        if success:
            self.leader_changes += 1

    @property
    def average_rounds(self) -> float:
        if not self.rounds_to_decide:
            return 0.0
        return sum(self.rounds_to_decide) / len(self.rounds_to_decide)

    @property
    def success_rate(self) -> float:
        if self.proposals_started == 0:
            return 0.0
        return self.proposals_decided / self.proposals_started

    def summary(self) -> dict:
        return {
            "proposals": self.proposals_started,
            "decided": self.proposals_decided,
            "rejected": self.proposals_rejected,
            "success_rate": round(self.success_rate, 3),
            "avg_rounds": round(self.average_rounds, 2),
            "elections": self.elections_held,
            "leader_changes": self.leader_changes,
        }
