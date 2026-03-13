"""
C237: Distributed Transactions -- Two-Phase Commit (2PC) and Three-Phase Commit (3PC)

Implements distributed transaction protocols for coordinating atomic commits
across multiple participants (nodes/databases).

Components:
- TransactionLog: Write-ahead log for crash recovery
- Participant: Node that holds data and votes on transactions
- TwoPhaseCommitCoordinator: Classic 2PC protocol
- ThreePhaseCommitCoordinator: Non-blocking 3PC protocol
- TransactionManager: High-level API with recovery
"""

import time
import enum
import threading
from collections import defaultdict


# --- Transaction States ---

class TxState(enum.Enum):
    """Transaction states for coordinator log."""
    INITIATED = "INITIATED"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"       # All voted YES (2PC decision point)
    PRE_COMMITTED = "PRE_COMMITTED"  # 3PC pre-commit phase
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"


class Vote(enum.Enum):
    """Participant vote."""
    YES = "YES"
    NO = "NO"
    TIMEOUT = "TIMEOUT"


class ParticipantState(enum.Enum):
    """Participant-side transaction state."""
    INIT = "INIT"
    READY = "READY"         # Voted YES, waiting for decision
    PRE_COMMITTED = "PRE_COMMITTED"  # 3PC: acknowledged pre-commit
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


# --- Transaction Log (WAL) ---

class TransactionLog:
    """Write-ahead log for crash recovery.

    Records state transitions so that after a crash, the coordinator
    or participant can determine what to do with in-flight transactions.
    """

    def __init__(self):
        self._entries = []  # [(tx_id, state, timestamp, metadata)]
        self._lock = threading.Lock()

    def append(self, tx_id, state, metadata=None):
        """Append a log entry."""
        with self._lock:
            self._entries.append((tx_id, state, time.time(), metadata or {}))

    def get_entries(self, tx_id=None):
        """Get log entries, optionally filtered by tx_id."""
        with self._lock:
            if tx_id is None:
                return list(self._entries)
            return [e for e in self._entries if e[0] == tx_id]

    def get_latest_state(self, tx_id):
        """Get the most recent state for a transaction."""
        entries = self.get_entries(tx_id)
        if not entries:
            return None
        return entries[-1][1]

    def get_all_tx_ids(self):
        """Get all unique transaction IDs."""
        with self._lock:
            return list(set(e[0] for e in self._entries))

    def clear(self):
        """Clear all entries (for testing)."""
        with self._lock:
            self._entries.clear()


# --- Participant ---

class Participant:
    """A node that participates in distributed transactions.

    Holds key-value data and supports prepare/commit/abort protocol.
    """

    def __init__(self, name, fail_on_prepare=False, slow_prepare=0,
                 fail_on_commit=False, fail_on_pre_commit=False):
        self.name = name
        self.data = {}
        self._pending = {}  # tx_id -> {key: value} staging area
        self._states = {}   # tx_id -> ParticipantState
        self._locks = {}    # key -> tx_id (key-level locking)
        self._log = TransactionLog()
        self._lock = threading.Lock()

        # Fault injection for testing
        self._fail_on_prepare = fail_on_prepare
        self._slow_prepare = slow_prepare
        self._fail_on_commit = fail_on_commit
        self._fail_on_pre_commit = fail_on_pre_commit

    def get_state(self, tx_id):
        """Get participant state for a transaction."""
        return self._states.get(tx_id)

    def prepare(self, tx_id, operations):
        """Phase 1: Prepare to commit.

        Validates operations, acquires locks, stages changes.
        Returns Vote.YES if ready, Vote.NO if cannot commit.

        operations: list of (op, key, value) where op is 'set' or 'delete'
        """
        if self._fail_on_prepare:
            self._log.append(tx_id, ParticipantState.ABORTED, {"reason": "injected failure"})
            self._states[tx_id] = ParticipantState.ABORTED
            return Vote.NO

        if self._slow_prepare > 0:
            time.sleep(self._slow_prepare)

        with self._lock:
            # Check for lock conflicts
            for op, key, *rest in operations:
                if key in self._locks and self._locks[key] != tx_id:
                    self._log.append(tx_id, ParticipantState.ABORTED,
                                     {"reason": f"lock conflict on {key}"})
                    self._states[tx_id] = ParticipantState.ABORTED
                    return Vote.NO

            # Acquire locks and stage changes
            staging = {}
            for op, key, *rest in operations:
                self._locks[key] = tx_id
                if op == "set":
                    staging[key] = ("set", rest[0] if rest else None)
                elif op == "delete":
                    staging[key] = ("delete", None)

            self._pending[tx_id] = staging
            self._states[tx_id] = ParticipantState.READY
            self._log.append(tx_id, ParticipantState.READY)
            return Vote.YES

    def pre_commit(self, tx_id):
        """3PC Phase 2: Acknowledge pre-commit.

        Only used in 3PC. Confirms that participant is still ready.
        """
        if self._fail_on_pre_commit:
            return False

        with self._lock:
            state = self._states.get(tx_id)
            if state != ParticipantState.READY:
                return False
            self._states[tx_id] = ParticipantState.PRE_COMMITTED
            self._log.append(tx_id, ParticipantState.PRE_COMMITTED)
            return True

    def commit(self, tx_id):
        """Phase 2/3: Apply staged changes."""
        if self._fail_on_commit:
            return False

        with self._lock:
            staging = self._pending.get(tx_id)
            if staging is None:
                # Already committed or unknown -- idempotent
                return True

            # Apply changes
            for key, (op, value) in staging.items():
                if op == "set":
                    self.data[key] = value
                elif op == "delete":
                    self.data.pop(key, None)

            # Release locks and clean up
            self._release_locks(tx_id)
            del self._pending[tx_id]
            self._states[tx_id] = ParticipantState.COMMITTED
            self._log.append(tx_id, ParticipantState.COMMITTED)
            return True

    def abort(self, tx_id):
        """Abort: Discard staged changes and release locks."""
        with self._lock:
            self._pending.pop(tx_id, None)
            self._release_locks(tx_id)
            self._states[tx_id] = ParticipantState.ABORTED
            self._log.append(tx_id, ParticipantState.ABORTED)
            return True

    def _release_locks(self, tx_id):
        """Release all locks held by a transaction."""
        keys_to_release = [k for k, v in self._locks.items() if v == tx_id]
        for k in keys_to_release:
            del self._locks[k]

    def get(self, key, default=None):
        """Read committed data."""
        return self.data.get(key, default)


# --- Two-Phase Commit Coordinator ---

class TwoPhaseCommitCoordinator:
    """Classic Two-Phase Commit (2PC) coordinator.

    Phase 1 (Prepare): Ask all participants to prepare
    Phase 2 (Commit/Abort): If all vote YES, commit; otherwise abort

    Properties:
    - Atomic: all-or-nothing across participants
    - Blocking: if coordinator crashes after prepare, participants block
    - Safe: never commits unless all voted YES
    """

    def __init__(self, participants, timeout=5.0):
        self.participants = {p.name: p for p in participants}
        self.timeout = timeout
        self._log = TransactionLog()
        self._tx_counter = 0
        self._tx_operations = {}  # tx_id -> {participant_name: operations}
        self._tx_results = {}     # tx_id -> TxState
        self._lock = threading.Lock()

    def new_transaction_id(self):
        """Generate a unique transaction ID."""
        with self._lock:
            self._tx_counter += 1
            return f"tx_{self._tx_counter:04d}"

    def begin(self):
        """Begin a new transaction. Returns tx_id."""
        tx_id = self.new_transaction_id()
        self._tx_operations[tx_id] = {}
        self._log.append(tx_id, TxState.INITIATED)
        self._tx_results[tx_id] = TxState.INITIATED
        return tx_id

    def add_operation(self, tx_id, participant_name, operations):
        """Add operations for a participant in this transaction.

        operations: list of (op, key, value) tuples
        """
        if tx_id not in self._tx_operations:
            raise ValueError(f"Unknown transaction: {tx_id}")
        if participant_name not in self.participants:
            raise ValueError(f"Unknown participant: {participant_name}")
        if participant_name not in self._tx_operations[tx_id]:
            self._tx_operations[tx_id][participant_name] = []
        self._tx_operations[tx_id][participant_name].extend(operations)

    def execute(self, tx_id):
        """Execute 2PC protocol for the transaction.

        Returns (success: bool, state: TxState)
        """
        ops = self._tx_operations.get(tx_id, {})
        if not ops:
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED

        # Phase 1: Prepare
        self._log.append(tx_id, TxState.PREPARING)
        self._tx_results[tx_id] = TxState.PREPARING

        votes = {}
        for pname, operations in ops.items():
            participant = self.participants[pname]
            try:
                vote = participant.prepare(tx_id, operations)
                votes[pname] = vote
            except Exception:
                votes[pname] = Vote.TIMEOUT

        all_yes = all(v == Vote.YES for v in votes.values())

        if all_yes:
            # Decision: COMMIT
            self._log.append(tx_id, TxState.PREPARED)
            self._log.append(tx_id, TxState.COMMITTING)
            self._tx_results[tx_id] = TxState.COMMITTING

            # Phase 2: Commit
            commit_results = {}
            for pname in ops:
                participant = self.participants[pname]
                try:
                    result = participant.commit(tx_id)
                    commit_results[pname] = result
                except Exception:
                    commit_results[pname] = False

            if all(commit_results.values()):
                self._log.append(tx_id, TxState.COMMITTED)
                self._tx_results[tx_id] = TxState.COMMITTED
                return True, TxState.COMMITTED
            else:
                # Partial commit failure -- log it but decision was commit
                # In real systems, would retry indefinitely
                self._log.append(tx_id, TxState.COMMITTED,
                                 {"partial_failures": [k for k, v in commit_results.items() if not v]})
                self._tx_results[tx_id] = TxState.COMMITTED
                return True, TxState.COMMITTED
        else:
            # Decision: ABORT
            self._log.append(tx_id, TxState.ABORTING)
            self._tx_results[tx_id] = TxState.ABORTING

            # Phase 2: Abort
            for pname in ops:
                participant = self.participants[pname]
                try:
                    participant.abort(tx_id)
                except Exception:
                    pass  # Best effort abort

            self._log.append(tx_id, TxState.ABORTED,
                             {"no_votes": [k for k, v in votes.items() if v != Vote.YES]})
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED

    def get_tx_state(self, tx_id):
        """Get the current state of a transaction."""
        return self._tx_results.get(tx_id)

    def get_log(self):
        """Get the transaction log."""
        return self._log


# --- Three-Phase Commit Coordinator ---

class ThreePhaseCommitCoordinator:
    """Three-Phase Commit (3PC) coordinator.

    Phase 1 (Prepare/CanCommit): Ask participants to vote
    Phase 2 (Pre-Commit): Tell participants to prepare for commit (can still abort)
    Phase 3 (DoCommit): Final commit

    Properties:
    - Non-blocking: participants can make progress even if coordinator fails
    - After pre-commit, participants can commit autonomously on timeout
    - More messages but better availability than 2PC
    """

    def __init__(self, participants, timeout=5.0):
        self.participants = {p.name: p for p in participants}
        self.timeout = timeout
        self._log = TransactionLog()
        self._tx_counter = 0
        self._tx_operations = {}
        self._tx_results = {}
        self._lock = threading.Lock()

    def new_transaction_id(self):
        with self._lock:
            self._tx_counter += 1
            return f"tx3_{self._tx_counter:04d}"

    def begin(self):
        tx_id = self.new_transaction_id()
        self._tx_operations[tx_id] = {}
        self._log.append(tx_id, TxState.INITIATED)
        self._tx_results[tx_id] = TxState.INITIATED
        return tx_id

    def add_operation(self, tx_id, participant_name, operations):
        if tx_id not in self._tx_operations:
            raise ValueError(f"Unknown transaction: {tx_id}")
        if participant_name not in self.participants:
            raise ValueError(f"Unknown participant: {participant_name}")
        if participant_name not in self._tx_operations[tx_id]:
            self._tx_operations[tx_id][participant_name] = []
        self._tx_operations[tx_id][participant_name].extend(operations)

    def execute(self, tx_id):
        """Execute 3PC protocol.

        Returns (success: bool, state: TxState)
        """
        ops = self._tx_operations.get(tx_id, {})
        if not ops:
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED

        # Phase 1: CanCommit (Prepare)
        self._log.append(tx_id, TxState.PREPARING)
        self._tx_results[tx_id] = TxState.PREPARING

        votes = {}
        for pname, operations in ops.items():
            participant = self.participants[pname]
            try:
                vote = participant.prepare(tx_id, operations)
                votes[pname] = vote
            except Exception:
                votes[pname] = Vote.TIMEOUT

        all_yes = all(v == Vote.YES for v in votes.values())

        if not all_yes:
            # Abort immediately
            self._log.append(tx_id, TxState.ABORTING)
            for pname in ops:
                try:
                    self.participants[pname].abort(tx_id)
                except Exception:
                    pass
            self._log.append(tx_id, TxState.ABORTED,
                             {"no_votes": [k for k, v in votes.items() if v != Vote.YES]})
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED

        # Phase 2: Pre-Commit
        self._log.append(tx_id, TxState.PRE_COMMITTED)
        self._tx_results[tx_id] = TxState.PRE_COMMITTED

        pre_commit_acks = {}
        for pname in ops:
            participant = self.participants[pname]
            try:
                ack = participant.pre_commit(tx_id)
                pre_commit_acks[pname] = ack
            except Exception:
                pre_commit_acks[pname] = False

        all_acked = all(pre_commit_acks.values())

        if not all_acked:
            # Pre-commit failed -- abort
            self._log.append(tx_id, TxState.ABORTING)
            for pname in ops:
                try:
                    self.participants[pname].abort(tx_id)
                except Exception:
                    pass
            self._log.append(tx_id, TxState.ABORTED,
                             {"pre_commit_failures": [k for k, v in pre_commit_acks.items() if not v]})
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED

        # Phase 3: DoCommit
        self._log.append(tx_id, TxState.COMMITTING)
        self._tx_results[tx_id] = TxState.COMMITTING

        for pname in ops:
            participant = self.participants[pname]
            try:
                participant.commit(tx_id)
            except Exception:
                pass  # After pre-commit, participants can commit autonomously

        self._log.append(tx_id, TxState.COMMITTED)
        self._tx_results[tx_id] = TxState.COMMITTED
        return True, TxState.COMMITTED

    def get_tx_state(self, tx_id):
        return self._tx_results.get(tx_id)

    def get_log(self):
        return self._log


# --- Transaction Manager ---

class TransactionManager:
    """High-level transaction manager with recovery support.

    Provides a simple API for distributed transactions across participants.
    Supports both 2PC and 3PC protocols.
    """

    def __init__(self, participants, protocol="2pc", timeout=5.0):
        self.participants = {p.name: p for p in participants}
        self.protocol = protocol
        self.timeout = timeout
        self._log = TransactionLog()
        self._history = {}       # tx_id -> {state, operations, protocol, start_time, end_time}
        self._active_txs = {}    # tx_id -> coordinator
        self._tx_counter = 0
        self._lock = threading.Lock()

    def _new_tx_id(self):
        with self._lock:
            self._tx_counter += 1
            return f"txm_{self._tx_counter:04d}"

    def _make_coordinator(self, participants=None):
        """Create a coordinator for the configured protocol."""
        parts = list((participants or self.participants).values())
        if self.protocol == "3pc":
            return ThreePhaseCommitCoordinator(parts, self.timeout)
        return TwoPhaseCommitCoordinator(parts, self.timeout)

    def execute_transaction(self, operations_map):
        """Execute a distributed transaction.

        operations_map: {participant_name: [(op, key, value), ...]}
        Returns (success, tx_id, state)
        """
        tx_id = self._new_tx_id()
        self._log.append(tx_id, TxState.INITIATED)
        self._history[tx_id] = {
            "state": TxState.INITIATED,
            "operations": operations_map,
            "protocol": self.protocol,
            "start_time": time.time(),
            "end_time": None,
        }

        # Validate participants
        for pname in operations_map:
            if pname not in self.participants:
                self._history[tx_id]["state"] = TxState.ABORTED
                self._history[tx_id]["end_time"] = time.time()
                return False, tx_id, TxState.ABORTED

        coordinator = self._make_coordinator()
        coord_tx_id = coordinator.begin()

        for pname, ops in operations_map.items():
            coordinator.add_operation(coord_tx_id, pname, ops)

        success, state = coordinator.execute(coord_tx_id)

        self._history[tx_id]["state"] = state
        self._history[tx_id]["end_time"] = time.time()
        self._log.append(tx_id, state)

        return success, tx_id, state

    def get_transaction_info(self, tx_id):
        """Get info about a transaction."""
        return self._history.get(tx_id)

    def get_all_transactions(self):
        """Get all transaction IDs and their states."""
        return {tx_id: info["state"] for tx_id, info in self._history.items()}

    def recover(self):
        """Recovery protocol: examine log and complete pending transactions.

        For 2PC (presumed abort):
        - If log shows COMMITTING/COMMITTED -> re-commit
        - If log shows anything else -> abort

        For 3PC:
        - If log shows PRE_COMMITTED or COMMITTING -> re-commit
        - Otherwise -> abort

        Returns list of (tx_id, action_taken)
        """
        actions = []
        for tx_id, info in list(self._history.items()):
            state = info["state"]
            if state in (TxState.COMMITTED, TxState.ABORTED):
                continue  # Already completed

            if self.protocol == "2pc":
                if state in (TxState.COMMITTING, TxState.COMMITTED):
                    # Decision was commit -- re-commit
                    for pname, ops in info["operations"].items():
                        if pname in self.participants:
                            self.participants[pname].commit(tx_id)
                    info["state"] = TxState.COMMITTED
                    self._log.append(tx_id, TxState.COMMITTED)
                    actions.append((tx_id, "recommitted"))
                else:
                    # Presumed abort
                    for pname in info["operations"]:
                        if pname in self.participants:
                            self.participants[pname].abort(tx_id)
                    info["state"] = TxState.ABORTED
                    self._log.append(tx_id, TxState.ABORTED)
                    actions.append((tx_id, "aborted"))
            else:
                # 3PC recovery
                if state in (TxState.PRE_COMMITTED, TxState.COMMITTING):
                    for pname, ops in info["operations"].items():
                        if pname in self.participants:
                            self.participants[pname].commit(tx_id)
                    info["state"] = TxState.COMMITTED
                    self._log.append(tx_id, TxState.COMMITTED)
                    actions.append((tx_id, "recommitted"))
                else:
                    for pname in info["operations"]:
                        if pname in self.participants:
                            self.participants[pname].abort(tx_id)
                    info["state"] = TxState.ABORTED
                    self._log.append(tx_id, TxState.ABORTED)
                    actions.append((tx_id, "aborted"))

        return actions

    def get_log(self):
        return self._log


# --- Distributed Key-Value Store ---

class DistributedKVStore:
    """A distributed key-value store using distributed transactions.

    Provides a high-level API for multi-node atomic operations.
    Keys are sharded across participants by hash.
    """

    def __init__(self, node_names, protocol="2pc"):
        self.nodes = {name: Participant(name) for name in node_names}
        self.node_list = list(node_names)
        self.manager = TransactionManager(
            list(self.nodes.values()), protocol=protocol
        )

    def _shard(self, key):
        """Determine which node owns a key."""
        h = hash(key) % len(self.node_list)
        return self.node_list[h]

    def put(self, key, value):
        """Put a key-value pair (single-key transaction)."""
        node = self._shard(key)
        success, tx_id, state = self.manager.execute_transaction({
            node: [("set", key, value)]
        })
        return success

    def get(self, key):
        """Get a value by key."""
        node = self._shard(key)
        return self.nodes[node].get(key)

    def delete(self, key):
        """Delete a key."""
        node = self._shard(key)
        success, tx_id, state = self.manager.execute_transaction({
            node: [("delete", key)]
        })
        return success

    def multi_put(self, items):
        """Atomically put multiple key-value pairs (may span nodes).

        items: dict of {key: value}
        Returns success boolean.
        """
        # Group by shard
        ops_map = defaultdict(list)
        for key, value in items.items():
            node = self._shard(key)
            ops_map[node].append(("set", key, value))

        success, tx_id, state = self.manager.execute_transaction(dict(ops_map))
        return success

    def multi_delete(self, keys):
        """Atomically delete multiple keys."""
        ops_map = defaultdict(list)
        for key in keys:
            node = self._shard(key)
            ops_map[node].append(("delete", key))

        success, tx_id, state = self.manager.execute_transaction(dict(ops_map))
        return success

    def transfer(self, from_key, to_key, amount):
        """Atomic transfer: subtract from one key, add to another.

        Both keys must hold numeric values.
        """
        from_node = self._shard(from_key)
        to_node = self._shard(to_key)

        from_val = self.nodes[from_node].get(from_key, 0)
        to_val = self.nodes[to_node].get(to_key, 0)

        if from_val < amount:
            return False  # Insufficient funds

        ops_map = {}
        if from_node == to_node:
            ops_map[from_node] = [
                ("set", from_key, from_val - amount),
                ("set", to_key, to_val + amount),
            ]
        else:
            ops_map[from_node] = [("set", from_key, from_val - amount)]
            ops_map[to_node] = [("set", to_key, to_val + amount)]

        success, tx_id, state = self.manager.execute_transaction(ops_map)
        return success


# --- Saga Coordinator (Compensating Transactions) ---

class SagaStep:
    """A single step in a saga with its compensating action."""

    def __init__(self, name, action, compensation):
        """
        name: step identifier
        action: callable() -> bool (returns True on success)
        compensation: callable() -> bool (undo the action)
        """
        self.name = name
        self.action = action
        self.compensation = compensation


class SagaCoordinator:
    """Saga pattern for long-lived distributed transactions.

    Unlike 2PC/3PC which hold locks, sagas execute steps sequentially
    and compensate (undo) on failure. Better for long-running workflows.

    Properties:
    - No distributed locks held during execution
    - Eventually consistent (not immediately atomic)
    - Compensations run in reverse order on failure
    """

    def __init__(self):
        self._log = TransactionLog()
        self._saga_counter = 0
        self._history = {}
        self._lock = threading.Lock()

    def execute(self, saga_id, steps):
        """Execute a saga (list of SagaSteps).

        Returns (success, completed_steps, compensated_steps)
        """
        if saga_id is None:
            with self._lock:
                self._saga_counter += 1
                saga_id = f"saga_{self._saga_counter:04d}"

        self._log.append(saga_id, TxState.INITIATED)
        completed = []
        compensated = []

        for step in steps:
            self._log.append(saga_id, TxState.PREPARING,
                             {"step": step.name, "phase": "executing"})
            try:
                result = step.action()
                if result:
                    completed.append(step.name)
                    self._log.append(saga_id, TxState.PREPARED,
                                     {"step": step.name, "phase": "completed"})
                else:
                    # Step failed -- compensate
                    self._log.append(saga_id, TxState.ABORTING,
                                     {"step": step.name, "phase": "failed"})
                    compensated = self._compensate(saga_id, steps, completed)
                    self._log.append(saga_id, TxState.ABORTED)
                    self._history[saga_id] = {
                        "state": TxState.ABORTED,
                        "completed": completed,
                        "compensated": compensated,
                        "failed_at": step.name,
                    }
                    return False, completed, compensated
            except Exception as e:
                self._log.append(saga_id, TxState.ABORTING,
                                 {"step": step.name, "phase": "exception", "error": str(e)})
                compensated = self._compensate(saga_id, steps, completed)
                self._log.append(saga_id, TxState.ABORTED)
                self._history[saga_id] = {
                    "state": TxState.ABORTED,
                    "completed": completed,
                    "compensated": compensated,
                    "failed_at": step.name,
                }
                return False, completed, compensated

        self._log.append(saga_id, TxState.COMMITTED)
        self._history[saga_id] = {
            "state": TxState.COMMITTED,
            "completed": completed,
            "compensated": [],
        }
        return True, completed, []

    def _compensate(self, saga_id, steps, completed):
        """Run compensations in reverse order for completed steps."""
        compensated = []
        step_map = {s.name: s for s in steps}

        for step_name in reversed(completed):
            step = step_map[step_name]
            self._log.append(saga_id, TxState.ABORTING,
                             {"step": step_name, "phase": "compensating"})
            try:
                step.compensation()
                compensated.append(step_name)
            except Exception:
                # Compensation failure -- log but continue
                compensated.append(f"{step_name}(failed)")

        return compensated

    def get_saga_info(self, saga_id):
        return self._history.get(saga_id)

    def get_log(self):
        return self._log


# --- Timeout Coordinator ---

class TimeoutCoordinator(TwoPhaseCommitCoordinator):
    """2PC coordinator with configurable timeouts per participant.

    Demonstrates timeout handling: if a participant doesn't respond
    within the timeout, treat as Vote.TIMEOUT (which triggers abort).
    """

    def __init__(self, participants, timeout=5.0):
        super().__init__(participants, timeout)

    def execute_with_timeout(self, tx_id, per_participant_timeout=None):
        """Execute 2PC with per-participant timeouts using threads.

        per_participant_timeout: {participant_name: timeout_seconds}
        """
        ops = self._tx_operations.get(tx_id, {})
        if not ops:
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED

        self._log.append(tx_id, TxState.PREPARING)
        self._tx_results[tx_id] = TxState.PREPARING

        # Phase 1: Parallel prepare with timeouts
        votes = {}
        threads = []
        results_lock = threading.Lock()

        def prepare_participant(pname, operations, timeout_val):
            try:
                vote = self.participants[pname].prepare(tx_id, operations)
                with results_lock:
                    votes[pname] = vote
            except Exception:
                with results_lock:
                    votes[pname] = Vote.TIMEOUT

        for pname, operations in ops.items():
            t_val = (per_participant_timeout or {}).get(pname, self.timeout)
            t = threading.Thread(target=prepare_participant,
                                 args=(pname, operations, t_val))
            threads.append((t, t_val))
            t.start()

        for t, t_val in threads:
            t.join(timeout=t_val)

        # Check for timed-out threads
        for pname in ops:
            if pname not in votes:
                votes[pname] = Vote.TIMEOUT

        all_yes = all(v == Vote.YES for v in votes.values())

        if all_yes:
            self._log.append(tx_id, TxState.PREPARED)
            self._log.append(tx_id, TxState.COMMITTING)
            self._tx_results[tx_id] = TxState.COMMITTING

            for pname in ops:
                try:
                    self.participants[pname].commit(tx_id)
                except Exception:
                    pass

            self._log.append(tx_id, TxState.COMMITTED)
            self._tx_results[tx_id] = TxState.COMMITTED
            return True, TxState.COMMITTED
        else:
            self._log.append(tx_id, TxState.ABORTING)
            for pname in ops:
                try:
                    self.participants[pname].abort(tx_id)
                except Exception:
                    pass
            self._log.append(tx_id, TxState.ABORTED)
            self._tx_results[tx_id] = TxState.ABORTED
            return False, TxState.ABORTED
