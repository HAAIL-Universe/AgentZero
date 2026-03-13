"""
C239: Practical Byzantine Fault Tolerance (PBFT)

A complete implementation of the PBFT consensus protocol for Byzantine fault
tolerance. Handles up to f faulty nodes in a system of 3f+1 total nodes.

Components:
- PBFTMessage: typed protocol messages with signatures
- PBFTLog: per-node message log with quorum tracking
- PBFTNode: full node implementation (primary/backup roles)
- PBFTNetwork: simulated network for testing
- ViewChangeManager: view change protocol for primary failure
- CheckpointManager: garbage collection via stable checkpoints
- ClientRequestHandler: client request ordering and reply dedup
"""

import hashlib
import json
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class MessageType(Enum):
    REQUEST = auto()
    PRE_PREPARE = auto()
    PREPARE = auto()
    COMMIT = auto()
    REPLY = auto()
    VIEW_CHANGE = auto()
    NEW_VIEW = auto()
    CHECKPOINT = auto()


@dataclass
class PBFTMessage:
    """A PBFT protocol message."""
    msg_type: MessageType
    view: int
    sequence: int
    sender: int
    digest: str = ""
    payload: Any = None
    timestamp: float = 0.0
    client_id: str = ""
    signature: str = ""  # simplified: hash-based

    def compute_signature(self, node_id: int) -> str:
        """Compute a simplified signature (hash of content + node secret)."""
        content = f"{self.msg_type.name}:{self.view}:{self.sequence}:{self.digest}:{node_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def sign(self, node_id: int):
        """Sign this message."""
        self.signature = self.compute_signature(node_id)

    def verify_signature(self, node_id: int) -> bool:
        """Verify the message signature."""
        expected = self.compute_signature(node_id)
        return self.signature == expected

    def digest_of(self) -> str:
        """Compute digest of the payload."""
        if self.payload is None:
            return ""
        content = json.dumps(self.payload, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


def compute_digest(payload: Any) -> str:
    """Compute digest of arbitrary payload."""
    content = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# PBFT Log
# ---------------------------------------------------------------------------

class PBFTLog:
    """Per-node message log tracking protocol progress."""

    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3  # max faulty nodes

        # Message storage: (view, sequence) -> list of messages
        self.pre_prepares: dict[tuple[int, int], PBFTMessage] = {}
        self.prepares: dict[tuple[int, int], list[PBFTMessage]] = {}
        self.commits: dict[tuple[int, int], list[PBFTMessage]] = {}

        # State tracking
        self.prepared: set[tuple[int, int]] = set()  # (view, seq) that reached prepared
        self.committed_local: set[tuple[int, int]] = set()  # (view, seq) committed locally
        self.executed: set[int] = set()  # sequences that have been executed

        # Low/high water marks for garbage collection
        self.low_water_mark = 0
        self.high_water_mark = 0  # set by checkpoint window

    def add_pre_prepare(self, msg: PBFTMessage) -> bool:
        """Add a pre-prepare message. Returns False if conflict exists."""
        key = (msg.view, msg.sequence)
        if key in self.pre_prepares:
            existing = self.pre_prepares[key]
            return existing.digest == msg.digest
        if msg.sequence <= self.low_water_mark:
            return False
        self.pre_prepares[key] = msg
        return True

    def add_prepare(self, msg: PBFTMessage) -> bool:
        """Add a prepare message. Returns True if new."""
        key = (msg.view, msg.sequence)
        if key not in self.prepares:
            self.prepares[key] = []
        # Don't add duplicates from same sender
        for existing in self.prepares[key]:
            if existing.sender == msg.sender:
                return False
        self.prepares[key].append(msg)
        return True

    def add_commit(self, msg: PBFTMessage) -> bool:
        """Add a commit message. Returns True if new."""
        key = (msg.view, msg.sequence)
        if key not in self.commits:
            self.commits[key] = []
        for existing in self.commits[key]:
            if existing.sender == msg.sender:
                return False
        self.commits[key].append(msg)
        return True

    def is_prepared(self, view: int, sequence: int, digest: str) -> bool:
        """Check if (view, sequence) has reached 'prepared' state.
        Requires: matching pre-prepare + 2f prepares with same digest."""
        key = (view, sequence)
        if key in self.prepared:
            return True

        pp = self.pre_prepares.get(key)
        if pp is None or pp.digest != digest:
            return False

        matching = sum(
            1 for p in self.prepares.get(key, [])
            if p.digest == digest
        )
        # Need 2f prepares (from different replicas, not counting pre-prepare sender)
        if matching >= 2 * self.f:
            self.prepared.add(key)
            return True
        return False

    def is_committed_local(self, view: int, sequence: int, digest: str) -> bool:
        """Check if (view, sequence) is committed-local.
        Requires: prepared + 2f+1 commits with same digest."""
        key = (view, sequence)
        if key in self.committed_local:
            return True

        if not self.is_prepared(view, sequence, digest):
            return False

        matching = sum(
            1 for c in self.commits.get(key, [])
            if c.digest == digest
        )
        if matching >= 2 * self.f + 1:
            self.committed_local.add(key)
            return True
        return False

    def get_prepare_certificate(self, view: int, sequence: int) -> Optional[list]:
        """Get the prepare certificate for (view, sequence) if it exists."""
        key = (view, sequence)
        pp = self.pre_prepares.get(key)
        if pp is None:
            return None
        prepares = self.prepares.get(key, [])
        if len(prepares) < 2 * self.f:
            return None
        return [pp] + prepares[:2 * self.f]

    def garbage_collect(self, new_low_water_mark: int):
        """Remove messages below the new low water mark."""
        self.low_water_mark = new_low_water_mark
        to_remove = []
        for key in self.pre_prepares:
            if key[1] <= new_low_water_mark:
                to_remove.append(key)
        for key in to_remove:
            self.pre_prepares.pop(key, None)
            self.prepares.pop(key, None)
            self.commits.pop(key, None)
            self.prepared.discard(key)
            self.committed_local.discard(key)


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages periodic checkpoints for garbage collection."""

    def __init__(self, node_id: int, total_nodes: int, checkpoint_interval: int = 100):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3
        self.checkpoint_interval = checkpoint_interval

        # checkpoint_seq -> {node_id: digest}
        self.checkpoint_proofs: dict[int, dict[int, str]] = {}
        self.stable_checkpoint: int = 0
        self.stable_digest: str = ""

    def should_checkpoint(self, sequence: int) -> bool:
        """Check if we should create a checkpoint at this sequence."""
        return sequence > 0 and sequence % self.checkpoint_interval == 0

    def create_checkpoint(self, sequence: int, state_digest: str) -> PBFTMessage:
        """Create a checkpoint message."""
        msg = PBFTMessage(
            msg_type=MessageType.CHECKPOINT,
            view=0,
            sequence=sequence,
            sender=self.node_id,
            digest=state_digest,
        )
        msg.sign(self.node_id)
        self.add_checkpoint(sequence, self.node_id, state_digest)
        return msg

    def add_checkpoint(self, sequence: int, node_id: int, digest: str) -> bool:
        """Add a checkpoint proof. Returns True if checkpoint becomes stable."""
        if sequence not in self.checkpoint_proofs:
            self.checkpoint_proofs[sequence] = {}
        self.checkpoint_proofs[sequence][node_id] = digest

        # Check if we have 2f+1 matching checkpoints
        proofs = self.checkpoint_proofs[sequence]
        digest_counts: dict[str, int] = {}
        for d in proofs.values():
            digest_counts[d] = digest_counts.get(d, 0) + 1

        for d, count in digest_counts.items():
            if count >= 2 * self.f + 1:
                if sequence > self.stable_checkpoint:
                    self.stable_checkpoint = sequence
                    self.stable_digest = d
                    # Clean old checkpoint proofs
                    old_seqs = [s for s in self.checkpoint_proofs if s < sequence]
                    for s in old_seqs:
                        del self.checkpoint_proofs[s]
                    return True
        return False

    def get_stable_checkpoint(self) -> tuple[int, str]:
        """Return (sequence, digest) of the last stable checkpoint."""
        return (self.stable_checkpoint, self.stable_digest)


# ---------------------------------------------------------------------------
# View Change Manager
# ---------------------------------------------------------------------------

class ViewChangeManager:
    """Manages view change protocol for primary failure recovery."""

    def __init__(self, node_id: int, total_nodes: int):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3
        self.current_view = 0

        # View change tracking
        self.view_change_msgs: dict[int, list[PBFTMessage]] = {}
        self.view_change_triggered: set[int] = set()
        self.new_view_sent: set[int] = set()
        self.view_change_timer_active = False
        self.view_change_timeout = 5.0  # seconds

    def primary_for_view(self, view: int) -> int:
        """Compute primary node for a given view."""
        return view % self.total_nodes

    def is_primary(self, view: int, node_id: int) -> bool:
        """Check if node is primary for given view."""
        return self.primary_for_view(view) == node_id

    def create_view_change(self, new_view: int, stable_checkpoint: int,
                           checkpoint_proofs: dict, prepare_certs: list) -> PBFTMessage:
        """Create a VIEW-CHANGE message."""
        msg = PBFTMessage(
            msg_type=MessageType.VIEW_CHANGE,
            view=new_view,
            sequence=stable_checkpoint,
            sender=self.node_id,
            payload={
                "checkpoint_proofs": checkpoint_proofs,
                "prepare_certificates": prepare_certs,
            },
        )
        msg.sign(self.node_id)
        return msg

    def add_view_change(self, msg: PBFTMessage) -> bool:
        """Add a view-change message. Returns True if we have enough for new-view."""
        new_view = msg.view
        if new_view not in self.view_change_msgs:
            self.view_change_msgs[new_view] = []

        # Don't add duplicates
        for existing in self.view_change_msgs[new_view]:
            if existing.sender == msg.sender:
                return False
        self.view_change_msgs[new_view].append(msg)

        # Need 2f+1 view-change messages
        return len(self.view_change_msgs[new_view]) >= 2 * self.f + 1

    def create_new_view(self, new_view: int) -> Optional[PBFTMessage]:
        """Create a NEW-VIEW message (only by new primary)."""
        if not self.is_primary(new_view, self.node_id):
            return None
        if new_view in self.new_view_sent:
            return None

        vc_msgs = self.view_change_msgs.get(new_view, [])
        if len(vc_msgs) < 2 * self.f + 1:
            return None

        # Compute min-s (lowest stable checkpoint) and max-s (highest prepared seq)
        min_s = min(m.sequence for m in vc_msgs)
        max_s = min_s  # default if no prepare certs

        all_certs = []
        for m in vc_msgs:
            if m.payload and "prepare_certificates" in m.payload:
                for cert in m.payload["prepare_certificates"]:
                    if isinstance(cert, dict):
                        seq = cert.get("sequence", 0)
                        if seq > max_s:
                            max_s = seq
                        all_certs.append(cert)

        msg = PBFTMessage(
            msg_type=MessageType.NEW_VIEW,
            view=new_view,
            sequence=0,
            sender=self.node_id,
            payload={
                "view_change_messages": [
                    {"sender": m.sender, "sequence": m.sequence} for m in vc_msgs
                ],
                "min_s": min_s,
                "max_s": max_s,
                "prepare_certificates": all_certs,
            },
        )
        msg.sign(self.node_id)
        self.new_view_sent.add(new_view)
        return msg

    def process_new_view(self, msg: PBFTMessage) -> bool:
        """Process a NEW-VIEW message. Returns True if valid."""
        new_view = msg.view
        if not self.is_primary(new_view, msg.sender):
            return False
        self.current_view = new_view
        self.view_change_timer_active = False
        return True


# ---------------------------------------------------------------------------
# State Machine (application layer)
# ---------------------------------------------------------------------------

class StateMachine:
    """Simple key-value state machine for PBFT to replicate."""

    def __init__(self):
        self.state: dict[str, Any] = {}
        self.history: list[tuple[int, Any, Any]] = []  # (seq, operation, result)

    def execute(self, sequence: int, operation: Any) -> Any:
        """Execute an operation and return the result."""
        if operation is None:
            return None

        op_type = operation.get("type", "noop")

        if op_type == "set":
            key = operation["key"]
            value = operation["value"]
            self.state[key] = value
            result = value
        elif op_type == "get":
            key = operation["key"]
            result = self.state.get(key)
        elif op_type == "delete":
            key = operation["key"]
            result = self.state.pop(key, None)
        elif op_type == "cas":  # compare-and-swap
            key = operation["key"]
            expected = operation["expected"]
            new_value = operation["new_value"]
            current = self.state.get(key)
            if current == expected:
                self.state[key] = new_value
                result = True
            else:
                result = False
        elif op_type == "noop":
            result = None
        else:
            result = None

        self.history.append((sequence, operation, result))
        return result

    def get_digest(self) -> str:
        """Get a digest of the current state."""
        content = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def snapshot(self) -> dict:
        """Return a snapshot of the current state."""
        return dict(self.state)

    def restore(self, state: dict):
        """Restore from a snapshot."""
        self.state = dict(state)


# ---------------------------------------------------------------------------
# PBFT Node
# ---------------------------------------------------------------------------

class PBFTNode:
    """A single PBFT replica node."""

    def __init__(self, node_id: int, total_nodes: int, checkpoint_interval: int = 100):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3

        self.view = 0
        self.sequence = 0  # next sequence to assign (primary only)
        self.last_executed = 0

        self.log = PBFTLog(node_id, total_nodes)
        self.checkpoint_mgr = CheckpointManager(node_id, total_nodes, checkpoint_interval)
        self.view_change_mgr = ViewChangeManager(node_id, total_nodes)
        self.state_machine = StateMachine()

        # Request tracking
        self.pending_requests: dict[str, PBFTMessage] = {}  # digest -> request msg
        self.client_replies: dict[str, Any] = {}  # client_id:timestamp -> reply
        self.request_digests: dict[tuple[int, int], str] = {}  # (view,seq) -> digest

        # Network callback (set by PBFTNetwork)
        self._send_callback = None

        # Byzantine behavior flags (for testing)
        self.byzantine = False
        self.byzantine_behavior: str = ""  # "silent", "wrong_digest", "double_vote"
        self.active = True  # can be set to False to simulate crash

        # Execution waiters: seq -> list of request msgs waiting
        self._request_by_seq: dict[int, PBFTMessage] = {}
        self._commit_sent: set[tuple[int, int]] = set()  # (view, seq) for idempotent commit

    @property
    def is_primary(self) -> bool:
        return self.view_change_mgr.is_primary(self.view, self.node_id)

    @property
    def primary_id(self) -> int:
        return self.view_change_mgr.primary_for_view(self.view)

    def send(self, target: Optional[int], msg: PBFTMessage):
        """Send a message to a specific node (None = broadcast)."""
        if self._send_callback:
            self._send_callback(self.node_id, target, msg)

    def broadcast(self, msg: PBFTMessage):
        """Broadcast to all nodes."""
        self.send(None, msg)

    # ---- Client request handling ----

    def handle_request(self, client_id: str, timestamp: float, operation: Any) -> Optional[PBFTMessage]:
        """Handle a client request. Only the primary assigns sequence numbers."""
        if not self.active:
            return None

        # Check for duplicate
        reply_key = f"{client_id}:{timestamp}"
        if reply_key in self.client_replies:
            return self.client_replies[reply_key]

        digest = compute_digest({"client_id": client_id, "timestamp": timestamp, "operation": operation})

        request_msg = PBFTMessage(
            msg_type=MessageType.REQUEST,
            view=self.view,
            sequence=0,
            sender=self.node_id,
            digest=digest,
            payload=operation,
            timestamp=timestamp,
            client_id=client_id,
        )

        if self.is_primary:
            self._assign_sequence(request_msg)
        else:
            # Forward to primary
            self.pending_requests[digest] = request_msg
            self.send(self.primary_id, request_msg)

        return None

    def _assign_sequence(self, request_msg: PBFTMessage):
        """Primary assigns a sequence number and sends pre-prepare."""
        self.sequence += 1
        seq = self.sequence

        self.pending_requests[request_msg.digest] = request_msg
        self._request_by_seq[seq] = request_msg

        # Create and send PRE-PREPARE
        pp_msg = PBFTMessage(
            msg_type=MessageType.PRE_PREPARE,
            view=self.view,
            sequence=seq,
            sender=self.node_id,
            digest=request_msg.digest,
            payload=request_msg.payload,
            timestamp=request_msg.timestamp,
            client_id=request_msg.client_id,
        )
        pp_msg.sign(self.node_id)

        if self.byzantine and self.byzantine_behavior == "wrong_digest":
            pp_msg.digest = "fake_digest_" + str(seq)

        self.log.add_pre_prepare(pp_msg)
        self.request_digests[(self.view, seq)] = request_msg.digest
        self.broadcast(pp_msg)

    # ---- Protocol message handlers ----

    def receive(self, msg: PBFTMessage):
        """Main message dispatch."""
        if not self.active:
            return

        if self.byzantine and self.byzantine_behavior == "silent":
            return

        handlers = {
            MessageType.REQUEST: self._on_request,
            MessageType.PRE_PREPARE: self._on_pre_prepare,
            MessageType.PREPARE: self._on_prepare,
            MessageType.COMMIT: self._on_commit,
            MessageType.CHECKPOINT: self._on_checkpoint,
            MessageType.VIEW_CHANGE: self._on_view_change,
            MessageType.NEW_VIEW: self._on_new_view,
        }
        handler = handlers.get(msg.msg_type)
        if handler:
            handler(msg)

    def _on_request(self, msg: PBFTMessage):
        """Handle forwarded client request (primary only)."""
        if self.is_primary:
            self.pending_requests[msg.digest] = msg
            self._assign_sequence(msg)

    def _on_pre_prepare(self, msg: PBFTMessage):
        """Handle PRE-PREPARE from primary."""
        # Verify it's from the current primary
        if msg.sender != self.primary_id:
            return
        if msg.view != self.view:
            return

        # Verify no conflicting pre-prepare
        if not self.log.add_pre_prepare(msg):
            return

        # Store the request info
        self.pending_requests[msg.digest] = msg
        self._request_by_seq[msg.sequence] = msg
        self.request_digests[(msg.view, msg.sequence)] = msg.digest

        # Send PREPARE
        prepare_msg = PBFTMessage(
            msg_type=MessageType.PREPARE,
            view=self.view,
            sequence=msg.sequence,
            sender=self.node_id,
            digest=msg.digest,
        )
        prepare_msg.sign(self.node_id)

        if self.byzantine and self.byzantine_behavior == "double_vote":
            bad_msg = PBFTMessage(
                msg_type=MessageType.PREPARE,
                view=self.view,
                sequence=msg.sequence,
                sender=self.node_id,
                digest="evil_digest",
            )
            bad_msg.sign(self.node_id)
            self.broadcast(bad_msg)

        self.log.add_prepare(prepare_msg)
        self.broadcast(prepare_msg)

        # Check progress -- we may already have enough prepares from cascaded delivery
        self._check_prepared(msg.view, msg.sequence, msg.digest)

    def _on_prepare(self, msg: PBFTMessage):
        """Handle PREPARE from a replica."""
        if msg.view != self.view:
            return

        self.log.add_prepare(msg)
        digest = self.request_digests.get((msg.view, msg.sequence), msg.digest)
        self._check_prepared(msg.view, msg.sequence, digest)

    def _check_prepared(self, view: int, sequence: int, digest: str):
        """Check if prepared and send COMMIT if so (idempotent)."""
        key = (view, sequence)
        if key in self._commit_sent:
            return
        if self.log.is_prepared(view, sequence, digest):
            self._commit_sent.add(key)
            commit_msg = PBFTMessage(
                msg_type=MessageType.COMMIT,
                view=self.view,
                sequence=sequence,
                sender=self.node_id,
                digest=digest,
            )
            commit_msg.sign(self.node_id)
            self.log.add_commit(commit_msg)
            self.broadcast(commit_msg)
            # Re-check committed -- we may already have enough commits from earlier delivery
            if self.log.is_committed_local(view, sequence, digest):
                self._try_execute()

    def _on_commit(self, msg: PBFTMessage):
        """Handle COMMIT from a replica."""
        if msg.view != self.view:
            return

        self.log.add_commit(msg)
        digest = self.request_digests.get((msg.view, msg.sequence), msg.digest)
        if self.log.is_committed_local(msg.view, msg.sequence, digest):
            self._try_execute()

    def _try_execute(self):
        """Execute committed requests in order."""
        committed_seqs = {s for (v, s) in self.log.committed_local if v == self.view}
        while self.last_executed + 1 in committed_seqs:
            next_seq = self.last_executed + 1
            req = self._request_by_seq.get(next_seq)
            if req is None:
                break

            operation = req.payload
            result = self.state_machine.execute(next_seq, operation)
            self.last_executed = next_seq

            reply = PBFTMessage(
                msg_type=MessageType.REPLY,
                view=self.view,
                sequence=next_seq,
                sender=self.node_id,
                payload=result,
                timestamp=req.timestamp,
                client_id=req.client_id,
            )
            reply.sign(self.node_id)

            reply_key = f"{req.client_id}:{req.timestamp}"
            self.client_replies[reply_key] = reply
            self.send(None, reply)

            if self.checkpoint_mgr.should_checkpoint(next_seq):
                self._do_checkpoint(next_seq)

    def _do_checkpoint(self, sequence: int):
        """Create and broadcast a checkpoint."""
        state_digest = self.state_machine.get_digest()
        cp_msg = self.checkpoint_mgr.create_checkpoint(sequence, state_digest)
        self.broadcast(cp_msg)
        # Own checkpoint may have made it stable -- sync GC
        stable_seq = self.checkpoint_mgr.stable_checkpoint
        if stable_seq > self.log.low_water_mark:
            self.log.garbage_collect(stable_seq)

    def _on_checkpoint(self, msg: PBFTMessage):
        """Handle CHECKPOINT from a replica."""
        self.checkpoint_mgr.add_checkpoint(
            msg.sequence, msg.sender, msg.digest
        )
        # Always sync GC with stable checkpoint
        stable_seq = self.checkpoint_mgr.stable_checkpoint
        if stable_seq > self.log.low_water_mark:
            self.log.garbage_collect(stable_seq)

    # ---- View change ----

    def start_view_change(self, new_view: int):
        """Initiate a view change."""
        if new_view <= self.view:
            return

        # Gather prepare certificates
        prepare_certs = []
        for (v, s), pp in self.log.pre_prepares.items():
            if (v, s) in self.log.prepared:
                prepare_certs.append({
                    "view": v,
                    "sequence": s,
                    "digest": pp.digest,
                })

        stable_seq, stable_digest = self.checkpoint_mgr.get_stable_checkpoint()
        vc_msg = self.view_change_mgr.create_view_change(
            new_view, stable_seq,
            {"sequence": stable_seq, "digest": stable_digest},
            prepare_certs,
        )
        self.view_change_mgr.view_change_triggered.add(new_view)
        self.broadcast(vc_msg)
        self.view_change_mgr.add_view_change(vc_msg)

    def _on_view_change(self, msg: PBFTMessage):
        """Handle VIEW-CHANGE message."""
        new_view = msg.view
        if new_view <= self.view:
            return

        enough = self.view_change_mgr.add_view_change(msg)
        if enough and self.view_change_mgr.is_primary(new_view, self.node_id):
            nv_msg = self.view_change_mgr.create_new_view(new_view)
            if nv_msg:
                self.view = new_view
                self.view_change_mgr.current_view = new_view
                self.sequence = max(self.sequence, self.last_executed)
                self._commit_sent.clear()
                self.broadcast(nv_msg)

    def _on_new_view(self, msg: PBFTMessage):
        """Handle NEW-VIEW message."""
        if self.view_change_mgr.process_new_view(msg):
            self.view = msg.view
            # Sync sequence counter so new primary continues from last executed
            self.sequence = max(self.sequence, self.last_executed)
            self._commit_sent.clear()


# ---------------------------------------------------------------------------
# PBFT Network (simulated)
# ---------------------------------------------------------------------------

class PBFTNetwork:
    """Simulated network connecting PBFT nodes."""

    def __init__(self, num_nodes: int, checkpoint_interval: int = 100):
        self.num_nodes = num_nodes
        self.f = (num_nodes - 1) // 3
        self.nodes: list[PBFTNode] = []
        self.message_log: list[tuple[int, Optional[int], PBFTMessage]] = []
        self.client_replies: dict[str, list[PBFTMessage]] = {}
        self.dropped_messages: list[PBFTMessage] = []

        # Network partition simulation
        self.partitions: dict[int, set[int]] = {}  # node -> set of unreachable nodes

        # Create nodes
        for i in range(num_nodes):
            node = PBFTNode(i, num_nodes, checkpoint_interval)
            node._send_callback = self._deliver
            self.nodes.append(node)

    def _deliver(self, sender: int, target: Optional[int], msg: PBFTMessage):
        """Deliver a message from sender to target (None = broadcast)."""
        self.message_log.append((sender, target, msg))

        if msg.msg_type == MessageType.REPLY:
            key = f"{msg.client_id}:{msg.timestamp}"
            if key not in self.client_replies:
                self.client_replies[key] = []
            self.client_replies[key].append(msg)

        if target is not None:
            # Unicast
            if self._is_partitioned(sender, target):
                self.dropped_messages.append(msg)
                return
            if target < len(self.nodes) and self.nodes[target].active:
                self.nodes[target].receive(msg)
        else:
            # Broadcast
            for i, node in enumerate(self.nodes):
                if i == sender:
                    continue
                if self._is_partitioned(sender, i):
                    self.dropped_messages.append(msg)
                    continue
                if node.active:
                    node.receive(msg)

    def _is_partitioned(self, a: int, b: int) -> bool:
        """Check if two nodes are partitioned."""
        if a in self.partitions and b in self.partitions[a]:
            return True
        if b in self.partitions and a in self.partitions[b]:
            return True
        return False

    def partition(self, node_a: int, node_b: int):
        """Create a network partition between two nodes."""
        if node_a not in self.partitions:
            self.partitions[node_a] = set()
        self.partitions[node_a].add(node_b)

    def heal_partition(self, node_a: int, node_b: int):
        """Remove a network partition."""
        if node_a in self.partitions:
            self.partitions[node_a].discard(node_b)
        if node_b in self.partitions:
            self.partitions[node_b].discard(node_a)

    def heal_all(self):
        """Remove all partitions."""
        self.partitions.clear()

    def submit_request(self, client_id: str, operation: Any, timestamp: float = None) -> str:
        """Submit a client request to the primary."""
        if timestamp is None:
            timestamp = time.time()
        primary = self.get_primary()
        primary.handle_request(client_id, timestamp, operation)
        return f"{client_id}:{timestamp}"

    def get_primary(self) -> PBFTNode:
        """Get the current primary node."""
        for node in self.nodes:
            if node.is_primary and node.active:
                return node
        # If primary is down, return first active node
        for node in self.nodes:
            if node.active:
                return node
        raise RuntimeError("No active nodes")

    def get_client_reply(self, request_key: str) -> Optional[Any]:
        """Get the consensus reply for a client request (f+1 matching)."""
        replies = self.client_replies.get(request_key, [])
        if not replies:
            return None

        # Need f+1 matching replies
        result_counts: dict = {}
        for r in replies:
            key = json.dumps(r.payload, sort_keys=True, default=str)
            result_counts[key] = result_counts.get(key, 0) + 1
            if result_counts[key] >= self.f + 1:
                return r.payload
        return None

    def check_safety(self) -> bool:
        """Verify safety: all active nodes have the same state for executed sequences."""
        active_nodes = [n for n in self.nodes if n.active and not n.byzantine]
        if len(active_nodes) < 2:
            return True

        ref = active_nodes[0]
        for node in active_nodes[1:]:
            min_executed = min(ref.last_executed, node.last_executed)
            for seq in range(1, min_executed + 1):
                ref_hist = [h for h in ref.state_machine.history if h[0] == seq]
                node_hist = [h for h in node.state_machine.history if h[0] == seq]
                if ref_hist and node_hist:
                    if ref_hist[0][2] != node_hist[0][2]:
                        return False
        return True

    def check_liveness(self, expected_executed: int) -> bool:
        """Check that at least f+1 correct nodes have executed expected number of requests."""
        correct_nodes = [n for n in self.nodes if n.active and not n.byzantine]
        count = sum(1 for n in correct_nodes if n.last_executed >= expected_executed)
        return count >= self.f + 1

    def trigger_view_change(self, new_view: int):
        """Trigger view change on all active correct nodes."""
        for node in self.nodes:
            if node.active and not node.byzantine:
                node.start_view_change(new_view)

    def get_states(self) -> dict[int, dict]:
        """Get state of all nodes."""
        return {
            n.node_id: {
                "view": n.view,
                "last_executed": n.last_executed,
                "is_primary": n.is_primary,
                "active": n.active,
                "byzantine": n.byzantine,
                "state": n.state_machine.snapshot(),
            }
            for n in self.nodes
        }


# ---------------------------------------------------------------------------
# Batch Request Handler
# ---------------------------------------------------------------------------

class BatchRequestHandler:
    """Handles batching of client requests for throughput."""

    def __init__(self, network: PBFTNetwork, batch_size: int = 10):
        self.network = network
        self.batch_size = batch_size
        self.pending: list[tuple[str, float, Any]] = []
        self.batch_count = 0

    def add_request(self, client_id: str, operation: Any, timestamp: float = None):
        """Add a request to the pending batch."""
        if timestamp is None:
            timestamp = time.time()
        self.pending.append((client_id, timestamp, operation))

        if len(self.pending) >= self.batch_size:
            return self.flush()
        return []

    def flush(self) -> list[str]:
        """Flush all pending requests."""
        keys = []
        for client_id, timestamp, operation in self.pending:
            key = self.network.submit_request(client_id, operation, timestamp)
            keys.append(key)
        self.batch_count += 1
        self.pending.clear()
        return keys


# ---------------------------------------------------------------------------
# Byzantine Fault Detector
# ---------------------------------------------------------------------------

class ByzantineDetector:
    """Detects potential Byzantine behavior from message patterns."""

    def __init__(self, total_nodes: int):
        self.total_nodes = total_nodes
        self.f = (total_nodes - 1) // 3
        self.suspicions: dict[int, list[str]] = {}  # node_id -> list of reasons

    def check_double_prepare(self, prepares: dict[tuple[int, int], list[PBFTMessage]]) -> list[int]:
        """Detect nodes that sent conflicting prepares for the same (view, seq)."""
        suspects = []
        for key, msgs in prepares.items():
            sender_digests: dict[int, set[str]] = {}
            for m in msgs:
                if m.sender not in sender_digests:
                    sender_digests[m.sender] = set()
                sender_digests[m.sender].add(m.digest)

            for sender, digests in sender_digests.items():
                if len(digests) > 1:
                    suspects.append(sender)
                    reason = f"Double prepare at (view={key[0]}, seq={key[1]}): {digests}"
                    if sender not in self.suspicions:
                        self.suspicions[sender] = []
                    self.suspicions[sender].append(reason)
        return suspects

    def check_primary_equivocation(self, pre_prepares: dict[tuple[int, int], PBFTMessage],
                                    all_pre_prepares: list[PBFTMessage]) -> bool:
        """Check if primary sent different pre-prepares for the same (view, seq)."""
        seen: dict[tuple[int, int], str] = {}
        for msg in all_pre_prepares:
            key = (msg.view, msg.sequence)
            if key in seen:
                if seen[key] != msg.digest:
                    sender = msg.sender
                    if sender not in self.suspicions:
                        self.suspicions[sender] = []
                    self.suspicions[sender].append(
                        f"Primary equivocation at (view={key[0]}, seq={key[1]})"
                    )
                    return True
            else:
                seen[key] = msg.digest
        return False

    def get_suspects(self) -> dict[int, list[str]]:
        """Return all suspected Byzantine nodes and reasons."""
        return dict(self.suspicions)

    def is_suspected(self, node_id: int) -> bool:
        """Check if a node is suspected of Byzantine behavior."""
        return node_id in self.suspicions and len(self.suspicions[node_id]) > 0
