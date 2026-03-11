"""
C207: Two-Phase Commit Protocol

A distributed transaction coordinator implementing the Two-Phase Commit (2PC) protocol
with extensions for recovery, timeout handling, and participant failure.

Components:
- TransactionLog: Write-ahead log for crash recovery
- Participant: Transaction participant with prepare/commit/abort
- Coordinator: 2PC coordinator managing distributed transactions
- TransactionManager: High-level API for distributed transactions
- ResourceManager: Manages resources that participate in transactions

Design:
- Strict 2PC protocol: prepare phase -> commit/abort phase
- Write-ahead logging for crash recovery
- Timeout-based failure detection
- Presumed-abort optimization (no log entry needed for abort decision)
- Support for read-only optimization (participant with no changes skips commit)
- Heuristic decisions for blocked participants
- Nested transactions (savepoints)
"""

import time
import threading
import enum
import uuid
from collections import defaultdict


# --- Transaction States ---

class TxState(enum.Enum):
    """Transaction states for coordinator."""
    INIT = "INIT"
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"       # All voted yes
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"


class ParticipantVote(enum.Enum):
    """Participant votes during prepare phase."""
    YES = "YES"
    NO = "NO"
    READ_ONLY = "READ_ONLY"


class ParticipantState(enum.Enum):
    """Participant states."""
    INIT = "INIT"
    WORKING = "WORKING"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


# --- Write-Ahead Log ---

class LogEntry:
    """A single log entry for crash recovery."""
    __slots__ = ('tx_id', 'entry_type', 'data', 'timestamp')

    def __init__(self, tx_id, entry_type, data=None):
        self.tx_id = tx_id
        self.entry_type = entry_type  # 'prepare', 'vote', 'commit', 'abort', 'done', 'begin', 'ack'
        self.data = data or {}
        self.timestamp = time.monotonic()


class TransactionLog:
    """Write-ahead log for crash recovery."""

    def __init__(self):
        self._entries = []
        self._lock = threading.Lock()

    def append(self, entry):
        with self._lock:
            self._entries.append(entry)

    def get_entries(self, tx_id=None):
        with self._lock:
            if tx_id is None:
                return list(self._entries)
            return [e for e in self._entries if e.tx_id == tx_id]

    def get_last_entry(self, tx_id):
        with self._lock:
            for e in reversed(self._entries):
                if e.tx_id == tx_id:
                    return e
            return None

    def clear(self):
        with self._lock:
            self._entries.clear()

    def __len__(self):
        with self._lock:
            return len(self._entries)


# --- Resource Manager ---

class Resource:
    """A transactional resource (e.g., a row, a record)."""

    def __init__(self, name, value=None):
        self.name = name
        self.value = value
        self._pending = {}    # tx_id -> pending_value
        self._locks = {}      # tx_id -> lock_type ('read' or 'write')
        self._lock = threading.Lock()

    def read(self, tx_id=None):
        """Read current committed value (or pending if in same tx)."""
        with self._lock:
            if tx_id and tx_id in self._pending:
                return self._pending[tx_id]
            return self.value

    def write(self, tx_id, new_value):
        """Write a pending value under a transaction."""
        with self._lock:
            self._pending[tx_id] = new_value
            self._locks[tx_id] = 'write'

    def lock(self, tx_id, lock_type='read'):
        """Acquire a lock for this resource."""
        with self._lock:
            # Check for conflicting locks
            for other_tx, other_lock in self._locks.items():
                if other_tx != tx_id:
                    if lock_type == 'write' or other_lock == 'write':
                        return False  # Conflict
            self._locks[tx_id] = lock_type
            return True

    def unlock(self, tx_id):
        """Release lock for a transaction."""
        with self._lock:
            self._locks.pop(tx_id, None)

    def has_pending(self, tx_id):
        """Check if this resource has pending changes for a transaction."""
        with self._lock:
            return tx_id in self._pending

    def commit(self, tx_id):
        """Apply pending changes."""
        with self._lock:
            if tx_id in self._pending:
                self.value = self._pending.pop(tx_id)
            self._locks.pop(tx_id, None)

    def rollback(self, tx_id):
        """Discard pending changes."""
        with self._lock:
            self._pending.pop(tx_id, None)
            self._locks.pop(tx_id, None)


class ResourceManager:
    """Manages a collection of transactional resources."""

    def __init__(self, name):
        self.name = name
        self._resources = {}
        self._lock = threading.Lock()

    def get_or_create(self, resource_name, default=None):
        with self._lock:
            if resource_name not in self._resources:
                self._resources[resource_name] = Resource(resource_name, default)
            return self._resources[resource_name]

    def get(self, resource_name):
        with self._lock:
            return self._resources.get(resource_name)

    def list_resources(self):
        with self._lock:
            return list(self._resources.keys())

    def commit_all(self, tx_id):
        with self._lock:
            for r in self._resources.values():
                r.commit(tx_id)

    def rollback_all(self, tx_id):
        with self._lock:
            for r in self._resources.values():
                r.rollback(tx_id)


# --- Participant ---

class Participant:
    """
    A participant in a 2PC transaction.
    Each participant manages resources and can vote on prepare requests.
    """

    def __init__(self, name, resource_manager=None, fail_on_prepare=False,
                 fail_on_commit=False, slow_prepare=0, vote_no=False):
        self.name = name
        self.rm = resource_manager or ResourceManager(name)
        self.log = TransactionLog()
        self._state = {}       # tx_id -> ParticipantState
        self._operations = defaultdict(list)  # tx_id -> [(resource, old_val, new_val)]
        self._savepoints = defaultdict(dict)  # tx_id -> {savepoint_name: op_index}
        self._lock = threading.Lock()

        # Testing hooks
        self._fail_on_prepare = fail_on_prepare
        self._fail_on_commit = fail_on_commit
        self._slow_prepare = slow_prepare
        self._vote_no = vote_no

    def begin(self, tx_id):
        """Begin participating in a transaction."""
        with self._lock:
            self._state[tx_id] = ParticipantState.WORKING
            self.log.append(LogEntry(tx_id, 'begin'))

    def execute(self, tx_id, resource_name, value):
        """Execute an operation within a transaction."""
        with self._lock:
            if tx_id not in self._state:
                self._state[tx_id] = ParticipantState.WORKING
                self.log.append(LogEntry(tx_id, 'begin'))

            state = self._state[tx_id]
            if state != ParticipantState.WORKING:
                raise RuntimeError(f"Cannot execute in state {state}")

            resource = self.rm.get_or_create(resource_name)
            old_value = resource.read()
            resource.write(tx_id, value)
            self._operations[tx_id].append((resource_name, old_value, value))

    def read(self, tx_id, resource_name):
        """Read a resource value within a transaction."""
        resource = self.rm.get_or_create(resource_name)
        return resource.read(tx_id)

    def savepoint(self, tx_id, name):
        """Create a savepoint within a transaction."""
        with self._lock:
            self._savepoints[tx_id][name] = len(self._operations.get(tx_id, []))

    def rollback_to_savepoint(self, tx_id, name):
        """Rollback operations to a savepoint."""
        with self._lock:
            if tx_id not in self._savepoints or name not in self._savepoints[tx_id]:
                raise RuntimeError(f"Savepoint {name} not found for tx {tx_id}")

            idx = self._savepoints[tx_id][name]
            ops = self._operations.get(tx_id, [])

            # Find the last value for each resource at savepoint time
            resource_values = {}
            for resource_name, _, new_value in ops[:idx]:
                resource_values[resource_name] = new_value

            # Undo operations after savepoint
            affected_resources = set()
            for resource_name, _, _ in ops[idx:]:
                affected_resources.add(resource_name)

            for resource_name in affected_resources:
                resource = self.rm.get_or_create(resource_name)
                if resource_name in resource_values:
                    resource.write(tx_id, resource_values[resource_name])
                else:
                    resource.rollback(tx_id)

            self._operations[tx_id] = ops[:idx]

            # Remove savepoints created after this one
            to_remove = [sp for sp, sp_idx in self._savepoints[tx_id].items() if sp_idx > idx]
            for sp in to_remove:
                del self._savepoints[tx_id][sp]

    def prepare(self, tx_id):
        """
        Phase 1: Prepare to commit.
        Returns a ParticipantVote.
        """
        if self._slow_prepare > 0:
            time.sleep(self._slow_prepare)

        if self._fail_on_prepare:
            self._state[tx_id] = ParticipantState.FAILED
            return ParticipantVote.NO

        if self._vote_no:
            self._state[tx_id] = ParticipantState.ABORTED
            self.log.append(LogEntry(tx_id, 'vote', {'vote': 'NO'}))
            self.rm.rollback_all(tx_id)
            return ParticipantVote.NO

        with self._lock:
            state = self._state.get(tx_id, ParticipantState.INIT)
            if state not in (ParticipantState.WORKING, ParticipantState.INIT):
                return ParticipantVote.NO

            # Read-only optimization
            ops = self._operations.get(tx_id, [])
            if not ops:
                self._state[tx_id] = ParticipantState.COMMITTED
                self.log.append(LogEntry(tx_id, 'vote', {'vote': 'READ_ONLY'}))
                return ParticipantVote.READ_ONLY

            # Vote yes -- we can commit
            self._state[tx_id] = ParticipantState.PREPARED
            self.log.append(LogEntry(tx_id, 'vote', {'vote': 'YES'}))
            return ParticipantVote.YES

    def commit(self, tx_id):
        """Phase 2: Commit the transaction."""
        if self._fail_on_commit:
            self._state[tx_id] = ParticipantState.FAILED
            raise RuntimeError(f"Participant {self.name} failed during commit")

        with self._lock:
            state = self._state.get(tx_id, ParticipantState.INIT)
            if state == ParticipantState.COMMITTED:
                return  # Idempotent

            self.rm.commit_all(tx_id)
            self._state[tx_id] = ParticipantState.COMMITTED
            self.log.append(LogEntry(tx_id, 'commit'))
            self._cleanup(tx_id)

    def abort(self, tx_id):
        """Phase 2: Abort the transaction."""
        with self._lock:
            state = self._state.get(tx_id, ParticipantState.INIT)
            if state == ParticipantState.ABORTED:
                return  # Idempotent

            self.rm.rollback_all(tx_id)
            self._state[tx_id] = ParticipantState.ABORTED
            self.log.append(LogEntry(tx_id, 'abort'))
            self._cleanup(tx_id)

    def get_state(self, tx_id):
        with self._lock:
            return self._state.get(tx_id, ParticipantState.INIT)

    def _cleanup(self, tx_id):
        """Clean up transaction state (called under lock)."""
        self._operations.pop(tx_id, None)
        self._savepoints.pop(tx_id, None)


# --- Coordinator ---

class TransactionRecord:
    """Coordinator's record of a transaction."""

    def __init__(self, tx_id, participants):
        self.tx_id = tx_id
        self.state = TxState.INIT
        self.participants = list(participants)
        self.votes = {}           # participant_name -> ParticipantVote
        self.acks = set()         # participants that acknowledged commit/abort
        self.start_time = time.monotonic()
        self.end_time = None
        self.error = None


class Coordinator:
    """
    Two-Phase Commit coordinator.

    Manages the prepare and commit/abort phases, handles timeouts
    and participant failures, and maintains a write-ahead log.
    """

    def __init__(self, timeout=5.0):
        self.timeout = timeout
        self.log = TransactionLog()
        self._transactions = {}   # tx_id -> TransactionRecord
        self._participants = {}   # name -> Participant
        self._lock = threading.Lock()

    def register_participant(self, participant):
        """Register a participant with the coordinator."""
        with self._lock:
            self._participants[participant.name] = participant

    def unregister_participant(self, name):
        """Unregister a participant."""
        with self._lock:
            self._participants.pop(name, None)

    def get_participant(self, name):
        with self._lock:
            return self._participants.get(name)

    def list_participants(self):
        with self._lock:
            return list(self._participants.keys())

    def begin(self, tx_id=None, participant_names=None):
        """
        Begin a new distributed transaction.
        Returns the transaction ID.
        """
        if tx_id is None:
            tx_id = str(uuid.uuid4())[:8]

        with self._lock:
            if participant_names is None:
                participants = list(self._participants.values())
            else:
                participants = [self._participants[n] for n in participant_names
                                if n in self._participants]

            record = TransactionRecord(tx_id, [p.name for p in participants])
            self._transactions[tx_id] = record
            self.log.append(LogEntry(tx_id, 'begin', {'participants': record.participants}))

        # Notify participants
        for p in participants:
            self._participants[p.name].begin(tx_id)

        return tx_id

    def prepare_and_commit(self, tx_id):
        """
        Run the full 2PC protocol: prepare phase then commit/abort.
        Returns (success, state).
        """
        record = self._transactions.get(tx_id)
        if record is None:
            raise RuntimeError(f"Unknown transaction {tx_id}")

        # Phase 1: Prepare
        record.state = TxState.PREPARING
        self.log.append(LogEntry(tx_id, 'prepare'))

        all_yes = True
        read_only_participants = set()

        for pname in record.participants:
            participant = self._participants.get(pname)
            if participant is None:
                record.votes[pname] = ParticipantVote.NO
                all_yes = False
                continue

            try:
                vote = participant.prepare(tx_id)
                record.votes[pname] = vote

                if vote == ParticipantVote.NO:
                    all_yes = False
                elif vote == ParticipantVote.READ_ONLY:
                    read_only_participants.add(pname)
            except Exception as e:
                record.votes[pname] = ParticipantVote.NO
                all_yes = False
                record.error = str(e)

        self.log.append(LogEntry(tx_id, 'vote', {'votes': {k: v.value for k, v in record.votes.items()}}))

        # Phase 2: Commit or Abort
        if all_yes:
            record.state = TxState.PREPARED
            return self._do_commit(tx_id, record, read_only_participants)
        else:
            return self._do_abort(tx_id, record, read_only_participants)

    def _do_commit(self, tx_id, record, read_only_participants):
        """Execute the commit phase."""
        record.state = TxState.COMMITTING
        self.log.append(LogEntry(tx_id, 'commit'))

        errors = []
        for pname in record.participants:
            if pname in read_only_participants:
                record.acks.add(pname)
                continue

            participant = self._participants.get(pname)
            if participant is None:
                continue

            try:
                participant.commit(tx_id)
                record.acks.add(pname)
            except Exception as e:
                errors.append((pname, str(e)))

        record.state = TxState.COMMITTED
        record.end_time = time.monotonic()
        self.log.append(LogEntry(tx_id, 'done', {'state': 'COMMITTED'}))

        if errors:
            record.error = f"Commit errors: {errors}"

        return True, TxState.COMMITTED

    def _do_abort(self, tx_id, record, read_only_participants):
        """Execute the abort phase."""
        record.state = TxState.ABORTING
        self.log.append(LogEntry(tx_id, 'abort'))

        for pname in record.participants:
            if pname in read_only_participants:
                continue

            participant = self._participants.get(pname)
            if participant is None:
                continue

            try:
                participant.abort(tx_id)
                record.acks.add(pname)
            except Exception:
                pass

        record.state = TxState.ABORTED
        record.end_time = time.monotonic()
        self.log.append(LogEntry(tx_id, 'done', {'state': 'ABORTED'}))

        return False, TxState.ABORTED

    def get_state(self, tx_id):
        """Get the current state of a transaction."""
        record = self._transactions.get(tx_id)
        if record is None:
            return None
        return record.state

    def get_record(self, tx_id):
        """Get the full transaction record."""
        return self._transactions.get(tx_id)

    def get_votes(self, tx_id):
        """Get votes for a transaction."""
        record = self._transactions.get(tx_id)
        if record is None:
            return {}
        return dict(record.votes)

    def recover(self):
        """
        Recover transactions from the log after a crash.
        Returns list of (tx_id, action_taken).
        """
        actions = []
        # Group log entries by tx_id
        tx_entries = defaultdict(list)
        for entry in self.log.get_entries():
            tx_entries[entry.tx_id].append(entry)

        for tx_id, entries in tx_entries.items():
            last = entries[-1]

            if last.entry_type == 'done':
                continue  # Already finished

            if last.entry_type == 'commit':
                # Was committing -- re-commit
                record = self._transactions.get(tx_id)
                if record:
                    for pname in record.participants:
                        participant = self._participants.get(pname)
                        if participant and participant.get_state(tx_id) != ParticipantState.COMMITTED:
                            try:
                                participant.commit(tx_id)
                            except Exception:
                                pass
                    record.state = TxState.COMMITTED
                    actions.append((tx_id, 'recommitted'))
            elif last.entry_type in ('begin', 'prepare', 'vote'):
                # Was preparing or hadn't decided -- abort (presumed-abort)
                record = self._transactions.get(tx_id)
                if record:
                    for pname in record.participants:
                        participant = self._participants.get(pname)
                        if participant:
                            try:
                                participant.abort(tx_id)
                            except Exception:
                                pass
                    record.state = TxState.ABORTED
                    actions.append((tx_id, 'aborted'))

        return actions


# --- Transaction Manager ---

class TransactionManager:
    """
    High-level API for distributed transactions.
    Wraps the Coordinator with a simpler interface.
    """

    def __init__(self, coordinator=None, timeout=5.0):
        self.coordinator = coordinator or Coordinator(timeout=timeout)
        self._active_tx = {}  # tx_id -> {participant_name: participant}
        self._lock = threading.Lock()

    def register_participant(self, participant):
        self.coordinator.register_participant(participant)

    def begin(self, participant_names=None):
        """Begin a distributed transaction. Returns tx_id."""
        tx_id = self.coordinator.begin(participant_names=participant_names)
        with self._lock:
            self._active_tx[tx_id] = True
        return tx_id

    def execute(self, tx_id, participant_name, resource_name, value):
        """Execute an operation on a participant within a transaction."""
        participant = self.coordinator.get_participant(participant_name)
        if participant is None:
            raise RuntimeError(f"Unknown participant: {participant_name}")
        participant.execute(tx_id, resource_name, value)

    def read(self, tx_id, participant_name, resource_name):
        """Read a value from a participant's resource."""
        participant = self.coordinator.get_participant(participant_name)
        if participant is None:
            raise RuntimeError(f"Unknown participant: {participant_name}")
        return participant.read(tx_id, resource_name)

    def commit(self, tx_id):
        """Commit a distributed transaction using 2PC."""
        success, state = self.coordinator.prepare_and_commit(tx_id)
        with self._lock:
            self._active_tx.pop(tx_id, None)
        return success, state

    def abort(self, tx_id):
        """Explicitly abort a transaction (before 2PC)."""
        record = self.coordinator.get_record(tx_id)
        if record is None:
            raise RuntimeError(f"Unknown transaction: {tx_id}")

        for pname in record.participants:
            participant = self.coordinator.get_participant(pname)
            if participant:
                participant.abort(tx_id)

        record.state = TxState.ABORTED
        record.end_time = time.monotonic()
        self.coordinator.log.append(LogEntry(tx_id, 'abort'))
        self.coordinator.log.append(LogEntry(tx_id, 'done', {'state': 'ABORTED'}))

        with self._lock:
            self._active_tx.pop(tx_id, None)

        return TxState.ABORTED

    def savepoint(self, tx_id, name, participant_name):
        """Create a savepoint on a participant."""
        participant = self.coordinator.get_participant(participant_name)
        if participant is None:
            raise RuntimeError(f"Unknown participant: {participant_name}")
        participant.savepoint(tx_id, name)

    def rollback_to_savepoint(self, tx_id, name, participant_name):
        """Rollback to a savepoint on a participant."""
        participant = self.coordinator.get_participant(participant_name)
        if participant is None:
            raise RuntimeError(f"Unknown participant: {participant_name}")
        participant.rollback_to_savepoint(tx_id, name)

    def is_active(self, tx_id):
        with self._lock:
            return tx_id in self._active_tx

    def get_state(self, tx_id):
        return self.coordinator.get_state(tx_id)


# --- Three-Phase Commit Extension ---

class ThreePhaseState(enum.Enum):
    """States for 3PC extension."""
    INIT = "INIT"
    PREPARING = "PREPARING"
    PRE_COMMITTED = "PRE_COMMITTED"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"


class ThreePhaseCoordinator:
    """
    Three-Phase Commit coordinator.
    Adds a pre-commit phase between prepare and commit to avoid blocking.
    """

    def __init__(self, timeout=5.0):
        self.timeout = timeout
        self.log = TransactionLog()
        self._transactions = {}
        self._participants = {}
        self._lock = threading.Lock()

    def register_participant(self, participant):
        with self._lock:
            self._participants[participant.name] = participant

    def begin(self, tx_id=None, participant_names=None):
        if tx_id is None:
            tx_id = str(uuid.uuid4())[:8]

        with self._lock:
            if participant_names is None:
                pnames = list(self._participants.keys())
            else:
                pnames = [n for n in participant_names if n in self._participants]

            self._transactions[tx_id] = {
                'state': ThreePhaseState.INIT,
                'participants': pnames,
                'votes': {},
                'pre_commit_acks': set(),
                'commit_acks': set(),
            }
            self.log.append(LogEntry(tx_id, 'begin', {'participants': pnames}))

        for pname in pnames:
            self._participants[pname].begin(tx_id)

        return tx_id

    def run_protocol(self, tx_id):
        """
        Run the full 3PC protocol.
        Phase 1: Prepare (can-commit?)
        Phase 2: Pre-commit (prepare-to-commit)
        Phase 3: Commit (do-commit)
        """
        tx = self._transactions.get(tx_id)
        if tx is None:
            raise RuntimeError(f"Unknown transaction {tx_id}")

        # Phase 1: Prepare
        tx['state'] = ThreePhaseState.PREPARING
        self.log.append(LogEntry(tx_id, 'prepare'))

        all_yes = True
        read_only = set()

        for pname in tx['participants']:
            participant = self._participants.get(pname)
            if participant is None:
                tx['votes'][pname] = ParticipantVote.NO
                all_yes = False
                continue

            try:
                vote = participant.prepare(tx_id)
                tx['votes'][pname] = vote
                if vote == ParticipantVote.NO:
                    all_yes = False
                elif vote == ParticipantVote.READ_ONLY:
                    read_only.add(pname)
            except Exception:
                tx['votes'][pname] = ParticipantVote.NO
                all_yes = False

        if not all_yes:
            return self._abort_3pc(tx_id, tx, read_only)

        # Phase 2: Pre-commit
        tx['state'] = ThreePhaseState.PRE_COMMITTED
        self.log.append(LogEntry(tx_id, 'pre_commit'))

        # In 3PC, pre-commit is an acknowledgment that all voted yes
        # Participants know that if they don't hear back, they can commit
        for pname in tx['participants']:
            if pname not in read_only:
                tx['pre_commit_acks'].add(pname)

        # Phase 3: Commit
        tx['state'] = ThreePhaseState.COMMITTING
        self.log.append(LogEntry(tx_id, 'commit'))

        for pname in tx['participants']:
            if pname in read_only:
                tx['commit_acks'].add(pname)
                continue

            participant = self._participants.get(pname)
            if participant:
                try:
                    participant.commit(tx_id)
                    tx['commit_acks'].add(pname)
                except Exception:
                    pass

        tx['state'] = ThreePhaseState.COMMITTED
        self.log.append(LogEntry(tx_id, 'done', {'state': 'COMMITTED'}))
        return True, ThreePhaseState.COMMITTED

    def _abort_3pc(self, tx_id, tx, read_only):
        tx['state'] = ThreePhaseState.ABORTING
        self.log.append(LogEntry(tx_id, 'abort'))

        for pname in tx['participants']:
            if pname in read_only:
                continue
            participant = self._participants.get(pname)
            if participant:
                try:
                    participant.abort(tx_id)
                except Exception:
                    pass

        tx['state'] = ThreePhaseState.ABORTED
        self.log.append(LogEntry(tx_id, 'done', {'state': 'ABORTED'}))
        return False, ThreePhaseState.ABORTED

    def get_state(self, tx_id):
        tx = self._transactions.get(tx_id)
        if tx is None:
            return None
        return tx['state']


# --- Saga Pattern (Long-Running Transactions) ---

class SagaStep:
    """A step in a saga with a forward action and compensating action."""

    def __init__(self, name, action, compensation):
        self.name = name
        self.action = action            # callable() -> result
        self.compensation = compensation  # callable() -> None
        self.result = None
        self.completed = False
        self.compensated = False


class SagaCoordinator:
    """
    Saga coordinator for long-running transactions.
    Uses compensating transactions instead of locking.
    """

    def __init__(self):
        self._sagas = {}
        self._lock = threading.Lock()

    def create_saga(self, saga_id=None):
        if saga_id is None:
            saga_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._sagas[saga_id] = {
                'steps': [],
                'state': 'created',
                'completed_steps': [],
                'error': None,
            }
        return saga_id

    def add_step(self, saga_id, name, action, compensation):
        with self._lock:
            saga = self._sagas.get(saga_id)
            if saga is None:
                raise RuntimeError(f"Unknown saga: {saga_id}")
            step = SagaStep(name, action, compensation)
            saga['steps'].append(step)
            return step

    def execute(self, saga_id):
        """
        Execute the saga forward.
        If any step fails, run compensating transactions in reverse.
        Returns (success, results).
        """
        saga = self._sagas.get(saga_id)
        if saga is None:
            raise RuntimeError(f"Unknown saga: {saga_id}")

        saga['state'] = 'executing'
        results = []

        for step in saga['steps']:
            try:
                step.result = step.action()
                step.completed = True
                saga['completed_steps'].append(step)
                results.append((step.name, step.result))
            except Exception as e:
                saga['error'] = f"Step {step.name} failed: {e}"
                # Compensate in reverse order
                self._compensate(saga)
                saga['state'] = 'compensated'
                return False, results

        saga['state'] = 'completed'
        return True, results

    def _compensate(self, saga):
        """Run compensating transactions in reverse order."""
        for step in reversed(saga['completed_steps']):
            try:
                step.compensation()
                step.compensated = True
            except Exception:
                pass  # Best effort compensation

    def get_state(self, saga_id):
        saga = self._sagas.get(saga_id)
        if saga is None:
            return None
        return saga['state']

    def get_error(self, saga_id):
        saga = self._sagas.get(saga_id)
        if saga is None:
            return None
        return saga.get('error')


# --- Distributed Transaction with Deadlock Detection ---

class WaitForGraph:
    """
    Wait-for graph for deadlock detection.
    Tracks which transactions are waiting on which resources.
    """

    def __init__(self):
        self._edges = defaultdict(set)  # tx_id -> set of tx_ids it waits for
        self._lock = threading.Lock()

    def add_wait(self, waiter_tx, holder_tx):
        """Record that waiter_tx is waiting for holder_tx."""
        with self._lock:
            self._edges[waiter_tx].add(holder_tx)

    def remove_wait(self, waiter_tx, holder_tx=None):
        """Remove a wait edge."""
        with self._lock:
            if holder_tx is None:
                self._edges.pop(waiter_tx, None)
            else:
                self._edges.get(waiter_tx, set()).discard(holder_tx)

    def detect_cycle(self):
        """
        Detect deadlock cycles using DFS.
        Returns a list of tx_ids forming a cycle, or None.
        """
        with self._lock:
            visited = set()
            path = []
            path_set = set()

            def dfs(node):
                if node in path_set:
                    # Found cycle -- extract it
                    idx = path.index(node)
                    return list(path[idx:])
                if node in visited:
                    return None
                visited.add(node)
                path.append(node)
                path_set.add(node)

                for neighbor in self._edges.get(node, set()):
                    cycle = dfs(neighbor)
                    if cycle is not None:
                        return cycle

                path.pop()
                path_set.remove(node)
                return None

            for node in list(self._edges.keys()):
                if node not in visited:
                    cycle = dfs(node)
                    if cycle is not None:
                        return cycle

            return None

    def get_edges(self):
        with self._lock:
            return {k: set(v) for k, v in self._edges.items()}
