"""
C227: Event Sourcing / CQRS
Composes C224 (Distributed Log) + C212 (Transaction Manager)

Event Sourcing: Store all state changes as immutable events. Rebuild state by replaying.
CQRS: Separate command (write) and query (read) models.

Components:
  1. Event / EventMetadata - immutable domain events with metadata
  2. EventStore - append-only event storage using C224's distributed log
  3. Aggregate - domain entity that applies events to rebuild state
  4. AggregateRepository - load/save aggregates via event replay + snapshots
  5. Command / CommandBus - write-side command dispatch
  6. Projection / ProjectionManager - read-side materialized views
  7. EventBus - publish/subscribe for event distribution
  8. SnapshotStore - periodic aggregate snapshots for fast replay
  9. Saga - multi-step process coordination across aggregates
  10. EventSourcingSystem - unified CQRS system orchestrator
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C224_distributed_log'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C212_transaction_manager'))

from distributed_log import (
    Broker, MessageQueue, Producer, Consumer, Record,
    ProducerRecord, Topic, Partition, ConsumerGroup, OffsetManager,
    DeadLetterQueue, DeliverySemantics, AckLevel, RetentionPolicy
)
from transaction_manager import (
    TransactionalDatabase, TransactionManager, Transaction,
    IsolationLevel, TxState, MVCCTable, WAL, WALRecord,
    WALRecordType, LockManager, Snapshot as TxSnapshot,
    TransactionError, DeadlockError, SerializationError
)

import time
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum, auto


# ---------------------------------------------------------------------------
# 1. Event / EventMetadata
# ---------------------------------------------------------------------------

@dataclass
class EventMetadata:
    """Metadata attached to every event."""
    event_id: str = ""
    timestamp: float = 0.0
    aggregate_id: str = ""
    aggregate_type: str = ""
    version: int = 0
    correlation_id: str = ""
    causation_id: str = ""
    user_id: str = ""
    schema_version: int = 1

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class Event:
    """Immutable domain event."""
    event_type: str
    data: dict = field(default_factory=dict)
    metadata: EventMetadata = field(default_factory=EventMetadata)

    def to_dict(self) -> dict:
        return {
            'event_type': self.event_type,
            'data': self.data,
            'metadata': {
                'event_id': self.metadata.event_id,
                'timestamp': self.metadata.timestamp,
                'aggregate_id': self.metadata.aggregate_id,
                'aggregate_type': self.metadata.aggregate_type,
                'version': self.metadata.version,
                'correlation_id': self.metadata.correlation_id,
                'causation_id': self.metadata.causation_id,
                'user_id': self.metadata.user_id,
                'schema_version': self.metadata.schema_version,
            }
        }

    @staticmethod
    def from_dict(d: dict) -> 'Event':
        md = d.get('metadata', {})
        return Event(
            event_type=d['event_type'],
            data=d.get('data', {}),
            metadata=EventMetadata(
                event_id=md.get('event_id', ''),
                timestamp=md.get('timestamp', 0.0),
                aggregate_id=md.get('aggregate_id', ''),
                aggregate_type=md.get('aggregate_type', ''),
                version=md.get('version', 0),
                correlation_id=md.get('correlation_id', ''),
                causation_id=md.get('causation_id', ''),
                user_id=md.get('user_id', ''),
                schema_version=md.get('schema_version', 1),
            )
        )


# ---------------------------------------------------------------------------
# 2. EventStore - append-only event storage using C224
# ---------------------------------------------------------------------------

class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""
    pass


class AggregateNotFoundError(Exception):
    """Raised when aggregate has no events."""
    pass


class EventStore:
    """
    Append-only event store backed by C224's distributed log.
    Each aggregate type gets its own topic. Events are keyed by aggregate_id.
    Supports optimistic concurrency via expected_version.
    """

    def __init__(self, broker: Optional[Broker] = None):
        self.broker = broker or Broker()
        self.mq = MessageQueue(self.broker)
        self._topics: dict[str, Topic] = {}
        # Track aggregate versions: (aggregate_type, aggregate_id) -> version
        self._versions: dict[tuple[str, str], int] = {}
        # Global event log for cross-aggregate queries
        self._all_events: list[Event] = []
        # Subscribers
        self._subscribers: list[Callable[[Event], None]] = []

    def _ensure_topic(self, aggregate_type: str) -> str:
        """Ensure a topic exists for an aggregate type."""
        topic_name = f"events.{aggregate_type}"
        if topic_name not in self._topics:
            self._topics[topic_name] = self.mq.create_topic(
                topic_name, partitions=1
            )
        return topic_name

    def append(self, events: list[Event], aggregate_type: str,
               aggregate_id: str, expected_version: int = -1) -> int:
        """
        Append events for an aggregate. Returns the new version.
        If expected_version >= 0, performs optimistic concurrency check.
        """
        key = (aggregate_type, aggregate_id)
        current_version = self._versions.get(key, 0)

        if expected_version >= 0 and current_version != expected_version:
            raise ConcurrencyError(
                f"Expected version {expected_version} but got {current_version} "
                f"for {aggregate_type}/{aggregate_id}"
            )

        topic_name = self._ensure_topic(aggregate_type)

        for i, event in enumerate(events):
            current_version += 1
            event.metadata.aggregate_id = aggregate_id
            event.metadata.aggregate_type = aggregate_type
            event.metadata.version = current_version

            self.mq.publish(
                topic_name,
                value=json.dumps(event.to_dict()),
                key=aggregate_id,
                headers={'version': str(current_version)}
            )
            self._all_events.append(event)

            # Notify subscribers
            for sub in self._subscribers:
                sub(event)

        self._versions[key] = current_version
        return current_version

    def load_events(self, aggregate_type: str, aggregate_id: str,
                    from_version: int = 0) -> list[Event]:
        """Load all events for an aggregate, optionally from a version."""
        key = (aggregate_type, aggregate_id)
        if key not in self._versions:
            return []

        return [
            e for e in self._all_events
            if e.metadata.aggregate_type == aggregate_type
            and e.metadata.aggregate_id == aggregate_id
            and e.metadata.version > from_version
        ]

    def get_version(self, aggregate_type: str, aggregate_id: str) -> int:
        """Get current version of an aggregate."""
        return self._versions.get((aggregate_type, aggregate_id), 0)

    def get_all_events(self, from_position: int = 0,
                       max_count: int = 1000) -> list[Event]:
        """Get events from the global stream."""
        return self._all_events[from_position:from_position + max_count]

    def get_events_by_type(self, event_type: str) -> list[Event]:
        """Get all events of a specific type across all aggregates."""
        return [e for e in self._all_events if e.event_type == event_type]

    def subscribe(self, callback: Callable[[Event], None]):
        """Subscribe to all new events."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]):
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def count(self) -> int:
        """Total number of events."""
        return len(self._all_events)

    def stream_count(self, aggregate_type: str, aggregate_id: str) -> int:
        """Number of events for a specific aggregate."""
        return self.get_version(aggregate_type, aggregate_id)


# ---------------------------------------------------------------------------
# 3. Aggregate - domain entity rebuilt from events
# ---------------------------------------------------------------------------

class Aggregate:
    """
    Base class for event-sourced aggregates.
    Subclasses implement apply_<event_type> methods.
    """

    def __init__(self, aggregate_id: str = ""):
        self.aggregate_id = aggregate_id or str(uuid.uuid4())
        self.version = 0
        self._uncommitted_events: list[Event] = []
        self._is_deleted = False

    @property
    def aggregate_type(self) -> str:
        return self.__class__.__name__

    @property
    def uncommitted_events(self) -> list[Event]:
        return list(self._uncommitted_events)

    def clear_uncommitted_events(self):
        self._uncommitted_events.clear()

    def raise_event(self, event_type: str, data: dict = None,
                    correlation_id: str = "", causation_id: str = "",
                    user_id: str = ""):
        """Create and apply a new event."""
        event = Event(
            event_type=event_type,
            data=data or {},
            metadata=EventMetadata(
                correlation_id=correlation_id,
                causation_id=causation_id,
                user_id=user_id,
            )
        )
        self._apply_event(event, is_new=True)

    def _apply_event(self, event: Event, is_new: bool = False):
        """Apply an event to this aggregate's state."""
        handler_name = f"apply_{event.event_type}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
        # Always bump version
        self.version += 1
        if is_new:
            self._uncommitted_events.append(event)

    def load_from_events(self, events: list[Event]):
        """Rebuild aggregate state from a list of events."""
        for event in events:
            self._apply_event(event, is_new=False)

    def load_from_snapshot(self, snapshot_data: dict, version: int):
        """Restore aggregate state from a snapshot."""
        self._restore_from_snapshot(snapshot_data)
        self.version = version

    def take_snapshot(self) -> dict:
        """Capture current state as a snapshot. Override in subclasses."""
        return self._get_snapshot_data()

    def _get_snapshot_data(self) -> dict:
        """Default snapshot: all non-private, non-method attributes."""
        data = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and k not in ('aggregate_id', 'version'):
                data[k] = v
        return data

    def _restore_from_snapshot(self, data: dict):
        """Restore state from snapshot data. Override for custom behavior."""
        for k, v in data.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# 4. SnapshotStore - periodic aggregate snapshots
# ---------------------------------------------------------------------------

@dataclass
class SnapshotEntry:
    """A stored aggregate snapshot."""
    aggregate_type: str
    aggregate_id: str
    version: int
    data: dict
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SnapshotStore:
    """Stores periodic snapshots to speed up aggregate loading."""

    def __init__(self):
        self._snapshots: dict[tuple[str, str], SnapshotEntry] = {}

    def save(self, aggregate: Aggregate):
        """Save a snapshot of an aggregate."""
        key = (aggregate.aggregate_type, aggregate.aggregate_id)
        self._snapshots[key] = SnapshotEntry(
            aggregate_type=aggregate.aggregate_type,
            aggregate_id=aggregate.aggregate_id,
            version=aggregate.version,
            data=aggregate.take_snapshot(),
        )

    def load(self, aggregate_type: str, aggregate_id: str) -> Optional[SnapshotEntry]:
        """Load the latest snapshot for an aggregate."""
        return self._snapshots.get((aggregate_type, aggregate_id))

    def delete(self, aggregate_type: str, aggregate_id: str) -> bool:
        """Delete a snapshot."""
        key = (aggregate_type, aggregate_id)
        if key in self._snapshots:
            del self._snapshots[key]
            return True
        return False

    def has_snapshot(self, aggregate_type: str, aggregate_id: str) -> bool:
        return (aggregate_type, aggregate_id) in self._snapshots

    def count(self) -> int:
        return len(self._snapshots)


# ---------------------------------------------------------------------------
# 5. AggregateRepository - load/save aggregates
# ---------------------------------------------------------------------------

class AggregateRepository:
    """
    Repository for loading and saving event-sourced aggregates.
    Uses EventStore for events and SnapshotStore for snapshots.
    """

    def __init__(self, event_store: EventStore,
                 snapshot_store: Optional[SnapshotStore] = None,
                 snapshot_interval: int = 50):
        self.event_store = event_store
        self.snapshot_store = snapshot_store or SnapshotStore()
        self.snapshot_interval = snapshot_interval

    def load(self, aggregate_class: type, aggregate_id: str) -> Aggregate:
        """
        Load an aggregate by replaying events.
        Uses snapshot if available to reduce replay.
        """
        aggregate = aggregate_class(aggregate_id)
        aggregate_type = aggregate.aggregate_type
        from_version = 0

        # Try to load from snapshot first
        snap = self.snapshot_store.load(aggregate_type, aggregate_id)
        if snap:
            aggregate.load_from_snapshot(snap.data, snap.version)
            from_version = snap.version

        # Replay events since snapshot
        events = self.event_store.load_events(
            aggregate_type, aggregate_id, from_version=from_version
        )

        if not events and from_version == 0:
            raise AggregateNotFoundError(
                f"{aggregate_type}/{aggregate_id} not found"
            )

        aggregate.load_from_events(events)
        return aggregate

    def save(self, aggregate: Aggregate, expected_version: int = -1) -> int:
        """
        Save uncommitted events from an aggregate.
        Returns the new version.
        """
        events = aggregate.uncommitted_events
        if not events:
            return aggregate.version

        if expected_version < 0:
            # Auto-calculate from aggregate version minus uncommitted
            expected_version = aggregate.version - len(events)

        new_version = self.event_store.append(
            events,
            aggregate.aggregate_type,
            aggregate.aggregate_id,
            expected_version=expected_version,
        )

        aggregate.clear_uncommitted_events()

        # Auto-snapshot if interval reached
        if (self.snapshot_interval > 0 and
                new_version % self.snapshot_interval == 0):
            self.snapshot_store.save(aggregate)

        return new_version

    def exists(self, aggregate_class: type, aggregate_id: str) -> bool:
        """Check if an aggregate exists."""
        aggregate = aggregate_class(aggregate_id)
        return self.event_store.get_version(
            aggregate.aggregate_type, aggregate_id
        ) > 0


# ---------------------------------------------------------------------------
# 6. Command / CommandBus - write-side dispatch
# ---------------------------------------------------------------------------

@dataclass
class Command:
    """A command requesting a state change."""
    command_type: str
    data: dict = field(default_factory=dict)
    aggregate_id: str = ""
    correlation_id: str = ""
    user_id: str = ""
    command_id: str = ""

    def __post_init__(self):
        if not self.command_id:
            self.command_id = str(uuid.uuid4())


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    aggregate_id: str = ""
    version: int = 0
    error: str = ""
    events: list = field(default_factory=list)


class CommandBus:
    """
    Dispatches commands to registered handlers.
    Handlers return CommandResult.
    """

    def __init__(self):
        self._handlers: dict[str, Callable[[Command], CommandResult]] = {}
        self._middleware: list[Callable] = []

    def register(self, command_type: str,
                 handler: Callable[[Command], CommandResult]):
        """Register a handler for a command type."""
        self._handlers[command_type] = handler

    def add_middleware(self, middleware: Callable):
        """Add middleware that wraps command execution."""
        self._middleware.append(middleware)

    def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command to its handler."""
        handler = self._handlers.get(command.command_type)
        if not handler:
            return CommandResult(
                success=False,
                error=f"No handler for command type: {command.command_type}"
            )

        # Build middleware chain
        def execute(cmd):
            return handler(cmd)

        chain = execute
        for mw in reversed(self._middleware):
            prev = chain
            def make_wrapper(m, p):
                def wrapper(cmd):
                    return m(cmd, p)
                return wrapper
            chain = make_wrapper(mw, prev)

        try:
            return chain(command)
        except ConcurrencyError as e:
            return CommandResult(success=False, error=str(e))
        except Exception as e:
            return CommandResult(success=False, error=str(e))

    def has_handler(self, command_type: str) -> bool:
        return command_type in self._handlers


# ---------------------------------------------------------------------------
# 7. Projection / ProjectionManager - read-side views
# ---------------------------------------------------------------------------

class ProjectionState(Enum):
    """State of a projection."""
    STOPPED = auto()
    RUNNING = auto()
    REBUILDING = auto()
    ERROR = auto()


class Projection:
    """
    Base class for read-side projections.
    Subclasses implement handle_<event_type> methods.
    Maintains a materialized view from events.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.position: int = 0  # Last processed global position
        self.state = ProjectionState.STOPPED
        self._error: str = ""

    def handle_event(self, event: Event, position: int):
        """Route event to type-specific handler."""
        handler_name = f"handle_{event.event_type}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(event)
        self.position = position

    def reset(self):
        """Reset projection state for rebuilding."""
        self.position = 0
        self.state = ProjectionState.STOPPED
        self._error = ""
        self._on_reset()

    def _on_reset(self):
        """Override to clear projection-specific state."""
        pass


class ProjectionManager:
    """
    Manages projections: catches them up from event store,
    handles rebuilding, and processes new events.
    """

    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self._projections: dict[str, Projection] = {}

    def register(self, projection: Projection):
        """Register a projection."""
        self._projections[projection.name] = projection
        projection.state = ProjectionState.STOPPED

    def unregister(self, name: str) -> bool:
        """Unregister a projection."""
        if name in self._projections:
            del self._projections[name]
            return True
        return False

    def get(self, name: str) -> Optional[Projection]:
        """Get a projection by name."""
        return self._projections.get(name)

    def start(self, name: str = ""):
        """Start one or all projections (catch up from event store)."""
        targets = ([self._projections[name]] if name
                   else list(self._projections.values()))
        for proj in targets:
            proj.state = ProjectionState.RUNNING
            self._catch_up(proj)

    def start_all(self):
        """Start all registered projections."""
        self.start()

    def rebuild(self, name: str):
        """Rebuild a projection from scratch."""
        proj = self._projections.get(name)
        if not proj:
            return
        proj.state = ProjectionState.REBUILDING
        proj.reset()
        self._catch_up(proj)
        proj.state = ProjectionState.RUNNING

    def _catch_up(self, projection: Projection):
        """Process events from projection's position to current."""
        events = self.event_store.get_all_events(
            from_position=projection.position
        )
        for i, event in enumerate(events):
            try:
                position = projection.position + i + 1
                projection.handle_event(event, position)
            except Exception as e:
                projection.state = ProjectionState.ERROR
                projection._error = str(e)
                return

    def process_event(self, event: Event):
        """Process a new event through all running projections."""
        for proj in self._projections.values():
            if proj.state == ProjectionState.RUNNING:
                try:
                    proj.handle_event(event, proj.position + 1)
                except Exception as e:
                    proj.state = ProjectionState.ERROR
                    proj._error = str(e)

    def status(self) -> dict:
        """Get status of all projections."""
        return {
            name: {
                'state': proj.state.name,
                'position': proj.position,
                'error': proj._error,
            }
            for name, proj in self._projections.items()
        }

    @property
    def projection_names(self) -> list[str]:
        return list(self._projections.keys())


# ---------------------------------------------------------------------------
# 8. EventBus - publish/subscribe for event distribution
# ---------------------------------------------------------------------------

class EventBus:
    """
    In-process event bus for distributing events to handlers.
    Supports typed subscriptions and wildcard (*) subscriptions.
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable[[Event], None]]] = {}
        self._wildcard_handlers: list[Callable[[Event], None]] = []
        self._dead_letter: list[tuple[Event, str]] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Subscribe to a specific event type."""
        if event_type == '*':
            self._wildcard_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """Unsubscribe from an event type."""
        if event_type == '*':
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        elif event_type in self._handlers:
            handlers = self._handlers[event_type]
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._wildcard_handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._dead_letter.append((event, str(e)))

    def publish_all(self, events: list[Event]):
        """Publish multiple events."""
        for event in events:
            self.publish(event)

    @property
    def dead_letter_count(self) -> int:
        return len(self._dead_letter)

    def get_dead_letters(self) -> list[tuple[Event, str]]:
        return list(self._dead_letter)

    def clear_dead_letters(self):
        self._dead_letter.clear()

    def handler_count(self, event_type: str = "") -> int:
        if event_type:
            return len(self._handlers.get(event_type, []))
        total = sum(len(h) for h in self._handlers.values())
        return total + len(self._wildcard_handlers)


# ---------------------------------------------------------------------------
# 9. Saga - multi-step process coordination
# ---------------------------------------------------------------------------

class SagaState(Enum):
    """State of a saga instance."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()
    FAILED = auto()


@dataclass
class SagaStep:
    """A step in a saga with action and compensation."""
    name: str
    action: Callable[['SagaContext'], None]
    compensation: Optional[Callable[['SagaContext'], None]] = None
    completed: bool = False


@dataclass
class SagaContext:
    """Context passed through saga steps."""
    saga_id: str
    data: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    commands: list = field(default_factory=list)

    def emit_event(self, event_type: str, data: dict = None):
        self.events.append(Event(event_type=event_type, data=data or {}))

    def emit_command(self, command_type: str, data: dict = None,
                     aggregate_id: str = ""):
        self.commands.append(Command(
            command_type=command_type, data=data or {},
            aggregate_id=aggregate_id
        ))


class Saga:
    """
    Orchestrates multi-step processes across aggregates.
    Supports compensation (rollback) on failure.
    """

    def __init__(self, saga_id: str = "", name: str = ""):
        self.saga_id = saga_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.state = SagaState.NOT_STARTED
        self.steps: list[SagaStep] = []
        self.context = SagaContext(saga_id=self.saga_id)
        self._current_step = 0
        self._error: str = ""

    def add_step(self, name: str,
                 action: Callable[[SagaContext], None],
                 compensation: Callable[[SagaContext], None] = None):
        """Add a step to the saga."""
        self.steps.append(SagaStep(
            name=name, action=action, compensation=compensation
        ))

    def execute(self) -> bool:
        """Execute the saga. Returns True if all steps succeed."""
        self.state = SagaState.IN_PROGRESS
        self._current_step = 0

        for i, step in enumerate(self.steps):
            self._current_step = i
            try:
                step.action(self.context)
                step.completed = True
            except Exception as e:
                self._error = f"Step '{step.name}' failed: {e}"
                self._compensate()
                return False

        self.state = SagaState.COMPLETED
        return True

    def _compensate(self):
        """Run compensation for completed steps in reverse."""
        self.state = SagaState.COMPENSATING
        # Compensate in reverse order, only completed steps
        for i in range(self._current_step - 1, -1, -1):
            step = self.steps[i]
            if step.completed and step.compensation:
                try:
                    step.compensation(self.context)
                except Exception:
                    self.state = SagaState.FAILED
                    return
        self.state = SagaState.COMPENSATED

    @property
    def error(self) -> str:
        return self._error

    @property
    def completed_steps(self) -> list[str]:
        return [s.name for s in self.steps if s.completed]

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.completed) / len(self.steps)


class SagaManager:
    """Manages saga instances."""

    def __init__(self, command_bus: Optional[CommandBus] = None,
                 event_bus: Optional[EventBus] = None):
        self.command_bus = command_bus
        self.event_bus = event_bus
        self._sagas: dict[str, Saga] = {}

    def register(self, saga: Saga):
        """Register a saga instance."""
        self._sagas[saga.saga_id] = saga

    def execute(self, saga: Saga) -> bool:
        """Execute a saga and dispatch resulting commands/events."""
        self.register(saga)
        success = saga.execute()

        # Dispatch commands from saga context
        if self.command_bus:
            for cmd in saga.context.commands:
                self.command_bus.dispatch(cmd)

        # Publish events from saga context
        if self.event_bus:
            for event in saga.context.events:
                self.event_bus.publish(event)

        return success

    def get(self, saga_id: str) -> Optional[Saga]:
        return self._sagas.get(saga_id)

    def status(self) -> dict:
        return {
            sid: {
                'name': s.name,
                'state': s.state.name,
                'progress': s.progress,
                'error': s.error,
            }
            for sid, s in self._sagas.items()
        }


# ---------------------------------------------------------------------------
# 10. EventSourcingSystem - unified CQRS orchestrator
# ---------------------------------------------------------------------------

class EventSourcingSystem:
    """
    Unified CQRS system that wires together all components:
    - EventStore for event persistence
    - AggregateRepository for aggregate lifecycle
    - CommandBus for write-side dispatch
    - ProjectionManager for read-side views
    - EventBus for event distribution
    - SnapshotStore for aggregate snapshots
    - SagaManager for process coordination
    """

    def __init__(self, broker: Optional[Broker] = None,
                 snapshot_interval: int = 50):
        self.event_store = EventStore(broker)
        self.snapshot_store = SnapshotStore()
        self.repository = AggregateRepository(
            self.event_store, self.snapshot_store,
            snapshot_interval=snapshot_interval
        )
        self.command_bus = CommandBus()
        self.event_bus = EventBus()
        self.projection_manager = ProjectionManager(self.event_store)
        self.saga_manager = SagaManager(self.command_bus, self.event_bus)

        # Wire event store to event bus and projection manager
        self.event_store.subscribe(self._on_event)

    def _on_event(self, event: Event):
        """Route events from store to bus and projections."""
        self.event_bus.publish(event)
        self.projection_manager.process_event(event)

    def register_command(self, command_type: str,
                         handler: Callable[[Command], CommandResult]):
        """Register a command handler."""
        self.command_bus.register(command_type, handler)

    def register_projection(self, projection: Projection):
        """Register and start a projection."""
        self.projection_manager.register(projection)

    def dispatch(self, command: Command) -> CommandResult:
        """Dispatch a command."""
        return self.command_bus.dispatch(command)

    def load_aggregate(self, aggregate_class: type,
                       aggregate_id: str) -> Aggregate:
        """Load an aggregate from the repository."""
        return self.repository.load(aggregate_class, aggregate_id)

    def save_aggregate(self, aggregate: Aggregate) -> int:
        """Save an aggregate's uncommitted events."""
        return self.repository.save(aggregate)

    def execute_saga(self, saga: Saga) -> bool:
        """Execute a saga."""
        return self.saga_manager.execute(saga)

    def rebuild_projection(self, name: str):
        """Rebuild a projection from scratch."""
        self.projection_manager.rebuild(name)

    def start_projections(self):
        """Start all projections."""
        self.projection_manager.start_all()

    def stats(self) -> dict:
        """System statistics."""
        return {
            'event_count': self.event_store.count(),
            'snapshot_count': self.snapshot_store.count(),
            'projections': self.projection_manager.status(),
            'sagas': self.saga_manager.status(),
        }
