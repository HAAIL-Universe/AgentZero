"""
Tests for C227: Event Sourcing / CQRS
Composes C224 (Distributed Log) + C212 (Transaction Manager)
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from event_sourcing import (
    Event, EventMetadata, EventStore, Aggregate, AggregateRepository,
    SnapshotStore, SnapshotEntry, Command, CommandResult, CommandBus,
    Projection, ProjectionManager, ProjectionState, EventBus,
    Saga, SagaState, SagaContext, SagaStep, SagaManager,
    EventSourcingSystem, ConcurrencyError, AggregateNotFoundError,
)


# ============================================================
# Test Aggregates (domain models for testing)
# ============================================================

class BankAccount(Aggregate):
    """Test aggregate: simple bank account."""

    def __init__(self, aggregate_id=""):
        super().__init__(aggregate_id)
        self.balance = 0
        self.owner = ""
        self.is_open = False
        self.transaction_log = []

    def open(self, owner: str, initial_deposit: int = 0):
        if self.is_open:
            raise ValueError("Account already open")
        self.raise_event("AccountOpened", {
            'owner': owner, 'initial_deposit': initial_deposit
        })

    def deposit(self, amount: int):
        if not self.is_open:
            raise ValueError("Account not open")
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.raise_event("MoneyDeposited", {'amount': amount})

    def withdraw(self, amount: int):
        if not self.is_open:
            raise ValueError("Account not open")
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.raise_event("MoneyWithdrawn", {'amount': amount})

    def close(self):
        if not self.is_open:
            raise ValueError("Account not open")
        if self.balance != 0:
            raise ValueError("Balance must be zero to close")
        self.raise_event("AccountClosed", {})

    # Event handlers
    def apply_AccountOpened(self, event):
        self.owner = event.data['owner']
        self.balance = event.data.get('initial_deposit', 0)
        self.is_open = True

    def apply_MoneyDeposited(self, event):
        self.balance += event.data['amount']
        self.transaction_log.append(('deposit', event.data['amount']))

    def apply_MoneyWithdrawn(self, event):
        self.balance -= event.data['amount']
        self.transaction_log.append(('withdraw', event.data['amount']))

    def apply_AccountClosed(self, event):
        self.is_open = False

    def _get_snapshot_data(self):
        return {
            'balance': self.balance,
            'owner': self.owner,
            'is_open': self.is_open,
            'transaction_log': list(self.transaction_log),
        }

    def _restore_from_snapshot(self, data):
        self.balance = data['balance']
        self.owner = data['owner']
        self.is_open = data['is_open']
        self.transaction_log = list(data.get('transaction_log', []))


class ShoppingCart(Aggregate):
    """Test aggregate: shopping cart."""

    def __init__(self, aggregate_id=""):
        super().__init__(aggregate_id)
        self.items = {}  # item_id -> {name, qty, price}
        self.checked_out = False

    def add_item(self, item_id: str, name: str, qty: int, price: float):
        self.raise_event("ItemAdded", {
            'item_id': item_id, 'name': name, 'qty': qty, 'price': price
        })

    def remove_item(self, item_id: str):
        if item_id not in self.items:
            raise ValueError(f"Item {item_id} not in cart")
        self.raise_event("ItemRemoved", {'item_id': item_id})

    def checkout(self):
        if not self.items:
            raise ValueError("Cart is empty")
        if self.checked_out:
            raise ValueError("Already checked out")
        total = sum(i['qty'] * i['price'] for i in self.items.values())
        self.raise_event("CartCheckedOut", {'total': total})

    def apply_ItemAdded(self, event):
        d = event.data
        item_id = d['item_id']
        if item_id in self.items:
            self.items[item_id]['qty'] += d['qty']
        else:
            self.items[item_id] = {
                'name': d['name'], 'qty': d['qty'], 'price': d['price']
            }

    def apply_ItemRemoved(self, event):
        del self.items[event.data['item_id']]

    def apply_CartCheckedOut(self, event):
        self.checked_out = True

    @property
    def total(self):
        return sum(i['qty'] * i['price'] for i in self.items.values())


# ============================================================
# Test Projections
# ============================================================

class AccountBalanceProjection(Projection):
    """Read model: account balances."""

    def __init__(self):
        super().__init__("AccountBalances")
        self.balances = {}  # aggregate_id -> balance
        self.account_owners = {}  # aggregate_id -> owner

    def handle_AccountOpened(self, event):
        aid = event.metadata.aggregate_id
        self.balances[aid] = event.data.get('initial_deposit', 0)
        self.account_owners[aid] = event.data['owner']

    def handle_MoneyDeposited(self, event):
        aid = event.metadata.aggregate_id
        self.balances[aid] = self.balances.get(aid, 0) + event.data['amount']

    def handle_MoneyWithdrawn(self, event):
        aid = event.metadata.aggregate_id
        self.balances[aid] = self.balances.get(aid, 0) - event.data['amount']

    def handle_AccountClosed(self, event):
        aid = event.metadata.aggregate_id
        if aid in self.balances:
            del self.balances[aid]
        if aid in self.account_owners:
            del self.account_owners[aid]

    def _on_reset(self):
        self.balances.clear()
        self.account_owners.clear()


class EventCountProjection(Projection):
    """Read model: count events by type."""

    def __init__(self):
        super().__init__("EventCounts")
        self.counts = {}

    def handle_event(self, event, position):
        et = event.event_type
        self.counts[et] = self.counts.get(et, 0) + 1
        self.position = position

    def _on_reset(self):
        self.counts.clear()


class TotalDepositsProjection(Projection):
    """Read model: total deposits across all accounts."""

    def __init__(self):
        super().__init__("TotalDeposits")
        self.total = 0
        self.count = 0

    def handle_MoneyDeposited(self, event):
        self.total += event.data['amount']
        self.count += 1

    def _on_reset(self):
        self.total = 0
        self.count = 0


# ============================================================
# 1. Event / EventMetadata Tests
# ============================================================

class TestEvent:
    def test_create_event(self):
        e = Event("UserCreated", {"name": "Alice"})
        assert e.event_type == "UserCreated"
        assert e.data == {"name": "Alice"}
        assert e.metadata.event_id != ""
        assert e.metadata.timestamp > 0

    def test_event_metadata_defaults(self):
        m = EventMetadata()
        assert m.event_id != ""
        assert m.timestamp > 0
        assert m.schema_version == 1
        assert m.aggregate_id == ""

    def test_event_metadata_custom(self):
        m = EventMetadata(
            aggregate_id="acc-1", aggregate_type="Account",
            version=5, correlation_id="corr-1", user_id="user-1"
        )
        assert m.aggregate_id == "acc-1"
        assert m.version == 5
        assert m.user_id == "user-1"

    def test_event_serialization(self):
        e = Event("OrderPlaced", {"item": "book", "qty": 2},
                  EventMetadata(aggregate_id="ord-1", version=1))
        d = e.to_dict()
        assert d['event_type'] == "OrderPlaced"
        assert d['data']['item'] == "book"
        assert d['metadata']['aggregate_id'] == "ord-1"

    def test_event_deserialization(self):
        d = {
            'event_type': 'ItemShipped',
            'data': {'tracking': 'TRK123'},
            'metadata': {
                'aggregate_id': 'ord-2',
                'version': 3,
                'schema_version': 2,
            }
        }
        e = Event.from_dict(d)
        assert e.event_type == "ItemShipped"
        assert e.data['tracking'] == "TRK123"
        assert e.metadata.version == 3
        assert e.metadata.schema_version == 2

    def test_event_roundtrip(self):
        original = Event("Test", {"x": 42},
                         EventMetadata(aggregate_id="a1", version=7,
                                       correlation_id="c1"))
        restored = Event.from_dict(original.to_dict())
        assert restored.event_type == original.event_type
        assert restored.data == original.data
        assert restored.metadata.aggregate_id == "a1"
        assert restored.metadata.version == 7
        assert restored.metadata.correlation_id == "c1"

    def test_event_empty_data(self):
        e = Event("Ping")
        assert e.data == {}
        d = e.to_dict()
        assert d['data'] == {}

    def test_unique_event_ids(self):
        e1 = Event("A")
        e2 = Event("A")
        assert e1.metadata.event_id != e2.metadata.event_id


# ============================================================
# 2. EventStore Tests
# ============================================================

class TestEventStore:
    def test_create_event_store(self):
        store = EventStore()
        assert store.count() == 0

    def test_append_events(self):
        store = EventStore()
        events = [Event("Created", {"name": "test"})]
        v = store.append(events, "Widget", "w-1")
        assert v == 1
        assert store.count() == 1

    def test_append_multiple_events(self):
        store = EventStore()
        events = [
            Event("Created", {"name": "test"}),
            Event("Updated", {"name": "test2"}),
        ]
        v = store.append(events, "Widget", "w-1")
        assert v == 2
        assert store.count() == 2

    def test_load_events(self):
        store = EventStore()
        store.append([Event("Created", {"x": 1})], "Thing", "t-1")
        store.append([Event("Updated", {"x": 2})], "Thing", "t-1")
        events = store.load_events("Thing", "t-1")
        assert len(events) == 2
        assert events[0].event_type == "Created"
        assert events[1].event_type == "Updated"

    def test_load_events_from_version(self):
        store = EventStore()
        store.append([Event("A"), Event("B"), Event("C")], "X", "x-1")
        events = store.load_events("X", "x-1", from_version=2)
        assert len(events) == 1
        assert events[0].event_type == "C"

    def test_load_events_nonexistent(self):
        store = EventStore()
        events = store.load_events("Ghost", "g-1")
        assert events == []

    def test_optimistic_concurrency_success(self):
        store = EventStore()
        store.append([Event("Created")], "W", "w-1", expected_version=0)
        v = store.append([Event("Updated")], "W", "w-1", expected_version=1)
        assert v == 2

    def test_optimistic_concurrency_failure(self):
        store = EventStore()
        store.append([Event("Created")], "W", "w-1")
        with pytest.raises(ConcurrencyError):
            store.append([Event("Updated")], "W", "w-1", expected_version=0)

    def test_get_version(self):
        store = EventStore()
        assert store.get_version("X", "x-1") == 0
        store.append([Event("A"), Event("B")], "X", "x-1")
        assert store.get_version("X", "x-1") == 2

    def test_get_all_events(self):
        store = EventStore()
        store.append([Event("A")], "X", "x-1")
        store.append([Event("B")], "Y", "y-1")
        all_events = store.get_all_events()
        assert len(all_events) == 2

    def test_get_all_events_pagination(self):
        store = EventStore()
        for i in range(10):
            store.append([Event(f"E{i}")], "X", "x-1")
        page = store.get_all_events(from_position=5, max_count=3)
        assert len(page) == 3
        assert page[0].event_type == "E5"

    def test_get_events_by_type(self):
        store = EventStore()
        store.append([Event("Created"), Event("Updated")], "X", "x-1")
        store.append([Event("Created")], "X", "x-2")
        created = store.get_events_by_type("Created")
        assert len(created) == 2

    def test_subscribe(self):
        store = EventStore()
        received = []
        store.subscribe(lambda e: received.append(e))
        store.append([Event("Test")], "X", "x-1")
        assert len(received) == 1
        assert received[0].event_type == "Test"

    def test_unsubscribe(self):
        store = EventStore()
        received = []
        cb = lambda e: received.append(e)
        store.subscribe(cb)
        store.append([Event("A")], "X", "x-1")
        store.unsubscribe(cb)
        store.append([Event("B")], "X", "x-1")
        assert len(received) == 1

    def test_stream_count(self):
        store = EventStore()
        store.append([Event("A"), Event("B")], "X", "x-1")
        store.append([Event("C")], "X", "x-2")
        assert store.stream_count("X", "x-1") == 2
        assert store.stream_count("X", "x-2") == 1

    def test_multiple_aggregate_types(self):
        store = EventStore()
        store.append([Event("A")], "TypeA", "a-1")
        store.append([Event("B")], "TypeB", "b-1")
        assert store.load_events("TypeA", "a-1")[0].event_type == "A"
        assert store.load_events("TypeB", "b-1")[0].event_type == "B"

    def test_event_metadata_set_on_append(self):
        store = EventStore()
        store.append([Event("Test")], "Widget", "w-1")
        events = store.load_events("Widget", "w-1")
        assert events[0].metadata.aggregate_id == "w-1"
        assert events[0].metadata.aggregate_type == "Widget"
        assert events[0].metadata.version == 1


# ============================================================
# 3. Aggregate Tests
# ============================================================

class TestAggregate:
    def test_create_aggregate(self):
        acc = BankAccount("acc-1")
        assert acc.aggregate_id == "acc-1"
        assert acc.version == 0
        assert acc.balance == 0

    def test_aggregate_type(self):
        acc = BankAccount()
        assert acc.aggregate_type == "BankAccount"

    def test_raise_event(self):
        acc = BankAccount("acc-1")
        acc.open("Alice", 100)
        assert acc.is_open is True
        assert acc.owner == "Alice"
        assert acc.balance == 100
        assert acc.version == 1
        assert len(acc.uncommitted_events) == 1

    def test_multiple_events(self):
        acc = BankAccount("acc-1")
        acc.open("Bob", 0)
        acc.deposit(500)
        acc.withdraw(200)
        assert acc.balance == 300
        assert acc.version == 3
        assert len(acc.uncommitted_events) == 3

    def test_clear_uncommitted(self):
        acc = BankAccount("acc-1")
        acc.open("Eve")
        assert len(acc.uncommitted_events) == 1
        acc.clear_uncommitted_events()
        assert len(acc.uncommitted_events) == 0
        # State is preserved
        assert acc.is_open is True

    def test_load_from_events(self):
        # Create events manually
        events = [
            Event("AccountOpened", {"owner": "Charlie", "initial_deposit": 50},
                  EventMetadata(version=1)),
            Event("MoneyDeposited", {"amount": 100}, EventMetadata(version=2)),
        ]
        acc = BankAccount("acc-1")
        acc.load_from_events(events)
        assert acc.owner == "Charlie"
        assert acc.balance == 150
        assert acc.version == 2
        assert len(acc.uncommitted_events) == 0

    def test_domain_validation(self):
        acc = BankAccount("acc-1")
        with pytest.raises(ValueError, match="not open"):
            acc.deposit(100)

    def test_insufficient_funds(self):
        acc = BankAccount("acc-1")
        acc.open("Dave", 50)
        with pytest.raises(ValueError, match="Insufficient"):
            acc.withdraw(100)

    def test_snapshot(self):
        acc = BankAccount("acc-1")
        acc.open("Eve", 100)
        acc.deposit(50)
        snap = acc.take_snapshot()
        assert snap['balance'] == 150
        assert snap['owner'] == "Eve"
        assert snap['is_open'] is True

    def test_restore_from_snapshot(self):
        acc = BankAccount("acc-1")
        acc.load_from_snapshot({
            'balance': 500, 'owner': 'Frank', 'is_open': True,
            'transaction_log': [('deposit', 500)]
        }, version=5)
        assert acc.balance == 500
        assert acc.owner == "Frank"
        assert acc.version == 5

    def test_snapshot_then_replay(self):
        # Snapshot at version 2, then replay events 3+
        acc = BankAccount("acc-1")
        acc.load_from_snapshot({
            'balance': 200, 'owner': 'Grace', 'is_open': True,
            'transaction_log': []
        }, version=2)
        events = [
            Event("MoneyDeposited", {"amount": 50}, EventMetadata(version=3)),
            Event("MoneyWithdrawn", {"amount": 30}, EventMetadata(version=4)),
        ]
        acc.load_from_events(events)
        assert acc.balance == 220
        assert acc.version == 4

    def test_shopping_cart_aggregate(self):
        cart = ShoppingCart("cart-1")
        cart.add_item("sku-1", "Widget", 2, 9.99)
        cart.add_item("sku-2", "Gadget", 1, 19.99)
        assert len(cart.items) == 2
        assert cart.total == 2 * 9.99 + 19.99

    def test_shopping_cart_remove(self):
        cart = ShoppingCart("cart-1")
        cart.add_item("sku-1", "Widget", 1, 10.0)
        cart.remove_item("sku-1")
        assert len(cart.items) == 0

    def test_shopping_cart_checkout(self):
        cart = ShoppingCart("cart-1")
        cart.add_item("sku-1", "Widget", 1, 25.0)
        cart.checkout()
        assert cart.checked_out is True

    def test_shopping_cart_empty_checkout(self):
        cart = ShoppingCart("cart-1")
        with pytest.raises(ValueError, match="empty"):
            cart.checkout()

    def test_auto_generated_id(self):
        acc = BankAccount()
        assert acc.aggregate_id != ""


# ============================================================
# 4. SnapshotStore Tests
# ============================================================

class TestSnapshotStore:
    def test_save_and_load(self):
        store = SnapshotStore()
        acc = BankAccount("acc-1")
        acc.open("Alice", 100)
        store.save(acc)
        snap = store.load("BankAccount", "acc-1")
        assert snap is not None
        assert snap.version == 1
        assert snap.data['balance'] == 100

    def test_load_nonexistent(self):
        store = SnapshotStore()
        assert store.load("X", "x-1") is None

    def test_overwrite_snapshot(self):
        store = SnapshotStore()
        acc = BankAccount("acc-1")
        acc.open("Alice", 100)
        store.save(acc)
        acc.deposit(50)
        store.save(acc)
        snap = store.load("BankAccount", "acc-1")
        assert snap.version == 2
        assert snap.data['balance'] == 150

    def test_delete_snapshot(self):
        store = SnapshotStore()
        acc = BankAccount("acc-1")
        acc.open("Alice")
        store.save(acc)
        assert store.delete("BankAccount", "acc-1") is True
        assert store.load("BankAccount", "acc-1") is None

    def test_delete_nonexistent(self):
        store = SnapshotStore()
        assert store.delete("X", "x-1") is False

    def test_has_snapshot(self):
        store = SnapshotStore()
        assert store.has_snapshot("X", "x-1") is False
        acc = BankAccount("acc-1")
        acc.open("Bob")
        store.save(acc)
        assert store.has_snapshot("BankAccount", "acc-1") is True

    def test_count(self):
        store = SnapshotStore()
        assert store.count() == 0
        acc = BankAccount("acc-1")
        acc.open("A")
        store.save(acc)
        assert store.count() == 1


# ============================================================
# 5. AggregateRepository Tests
# ============================================================

class TestAggregateRepository:
    def setup_method(self):
        self.event_store = EventStore()
        self.snapshot_store = SnapshotStore()
        self.repo = AggregateRepository(
            self.event_store, self.snapshot_store, snapshot_interval=5
        )

    def test_save_and_load(self):
        acc = BankAccount("acc-1")
        acc.open("Alice", 100)
        self.repo.save(acc)

        loaded = self.repo.load(BankAccount, "acc-1")
        assert loaded.owner == "Alice"
        assert loaded.balance == 100
        assert loaded.version == 1

    def test_save_multiple_operations(self):
        acc = BankAccount("acc-1")
        acc.open("Bob", 0)
        acc.deposit(500)
        acc.withdraw(200)
        self.repo.save(acc)

        loaded = self.repo.load(BankAccount, "acc-1")
        assert loaded.balance == 300
        assert loaded.version == 3

    def test_load_nonexistent(self):
        with pytest.raises(AggregateNotFoundError):
            self.repo.load(BankAccount, "ghost")

    def test_incremental_save(self):
        acc = BankAccount("acc-1")
        acc.open("Charlie", 0)
        self.repo.save(acc)

        loaded = self.repo.load(BankAccount, "acc-1")
        loaded.deposit(100)
        self.repo.save(loaded)

        reloaded = self.repo.load(BankAccount, "acc-1")
        assert reloaded.balance == 100
        assert reloaded.version == 2

    def test_auto_snapshot(self):
        acc = BankAccount("acc-1")
        acc.open("Dave", 0)
        for i in range(4):
            acc.deposit(10)
        # 5 events total (1 open + 4 deposits) -> triggers snapshot at v5
        self.repo.save(acc)
        assert self.snapshot_store.has_snapshot("BankAccount", "acc-1")

    def test_load_from_snapshot_plus_events(self):
        acc = BankAccount("acc-1")
        acc.open("Eve", 0)
        for i in range(4):
            acc.deposit(10)
        self.repo.save(acc)  # v5, snapshot taken

        loaded = self.repo.load(BankAccount, "acc-1")
        loaded.deposit(100)
        self.repo.save(loaded)  # v6, no snapshot

        # Load again -- uses snapshot at v5 + replay event at v6
        reloaded = self.repo.load(BankAccount, "acc-1")
        assert reloaded.balance == 140  # 4*10 + 100
        assert reloaded.version == 6

    def test_exists(self):
        assert self.repo.exists(BankAccount, "acc-1") is False
        acc = BankAccount("acc-1")
        acc.open("Frank")
        self.repo.save(acc)
        assert self.repo.exists(BankAccount, "acc-1") is True

    def test_concurrency_conflict(self):
        acc = BankAccount("acc-1")
        acc.open("Grace")
        self.repo.save(acc)

        # Two "users" load the same aggregate
        user1 = self.repo.load(BankAccount, "acc-1")
        user2 = self.repo.load(BankAccount, "acc-1")

        user1.deposit(100)
        self.repo.save(user1)

        user2.deposit(200)
        with pytest.raises(ConcurrencyError):
            self.repo.save(user2)

    def test_save_no_events(self):
        acc = BankAccount("acc-1")
        acc.open("Hank")
        self.repo.save(acc)
        loaded = self.repo.load(BankAccount, "acc-1")
        # No changes -> save returns current version
        v = self.repo.save(loaded)
        assert v == 1


# ============================================================
# 6. Command / CommandBus Tests
# ============================================================

class TestCommand:
    def test_create_command(self):
        cmd = Command("CreateAccount", {"owner": "Alice"}, aggregate_id="acc-1")
        assert cmd.command_type == "CreateAccount"
        assert cmd.data['owner'] == "Alice"
        assert cmd.command_id != ""

    def test_command_id_auto_generated(self):
        c1 = Command("A")
        c2 = Command("A")
        assert c1.command_id != c2.command_id


class TestCommandBus:
    def setup_method(self):
        self.bus = CommandBus()
        self.event_store = EventStore()
        self.repo = AggregateRepository(self.event_store)

    def test_register_and_dispatch(self):
        def handle_create(cmd):
            acc = BankAccount(cmd.aggregate_id)
            acc.open(cmd.data['owner'], cmd.data.get('deposit', 0))
            self.repo.save(acc)
            return CommandResult(success=True, aggregate_id=acc.aggregate_id,
                                version=acc.version)

        self.bus.register("CreateAccount", handle_create)
        result = self.bus.dispatch(Command(
            "CreateAccount", {"owner": "Alice", "deposit": 100},
            aggregate_id="acc-1"
        ))
        assert result.success is True
        assert result.aggregate_id == "acc-1"

    def test_dispatch_unknown_command(self):
        result = self.bus.dispatch(Command("Unknown"))
        assert result.success is False
        assert "No handler" in result.error

    def test_has_handler(self):
        assert self.bus.has_handler("X") is False
        self.bus.register("X", lambda cmd: CommandResult(success=True))
        assert self.bus.has_handler("X") is True

    def test_handler_exception(self):
        def bad_handler(cmd):
            raise RuntimeError("boom")

        self.bus.register("Boom", bad_handler)
        result = self.bus.dispatch(Command("Boom"))
        assert result.success is False
        assert "boom" in result.error

    def test_middleware(self):
        log = []

        def logging_middleware(cmd, next_handler):
            log.append(f"before:{cmd.command_type}")
            result = next_handler(cmd)
            log.append(f"after:{cmd.command_type}")
            return result

        self.bus.add_middleware(logging_middleware)
        self.bus.register("Test", lambda cmd: CommandResult(success=True))
        self.bus.dispatch(Command("Test"))
        assert log == ["before:Test", "after:Test"]

    def test_multiple_middleware(self):
        order = []

        def mw1(cmd, next_handler):
            order.append("mw1-before")
            r = next_handler(cmd)
            order.append("mw1-after")
            return r

        def mw2(cmd, next_handler):
            order.append("mw2-before")
            r = next_handler(cmd)
            order.append("mw2-after")
            return r

        self.bus.add_middleware(mw1)
        self.bus.add_middleware(mw2)
        self.bus.register("T", lambda cmd: CommandResult(success=True))
        self.bus.dispatch(Command("T"))
        # Middleware wraps in order: mw1 wraps mw2 wraps handler
        assert order == ["mw1-before", "mw2-before", "mw2-after", "mw1-after"]

    def test_concurrency_error_in_handler(self):
        def conflict_handler(cmd):
            raise ConcurrencyError("conflict!")

        self.bus.register("Conflict", conflict_handler)
        result = self.bus.dispatch(Command("Conflict"))
        assert result.success is False
        assert "conflict" in result.error


# ============================================================
# 7. Projection / ProjectionManager Tests
# ============================================================

class TestProjection:
    def test_projection_handles_events(self):
        proj = AccountBalanceProjection()
        e1 = Event("AccountOpened", {"owner": "Alice", "initial_deposit": 100},
                    EventMetadata(aggregate_id="a1"))
        proj.handle_event(e1, 1)
        assert proj.balances["a1"] == 100
        assert proj.position == 1

    def test_projection_ignores_unknown_events(self):
        proj = AccountBalanceProjection()
        e = Event("UnknownEvent", {}, EventMetadata(aggregate_id="x"))
        proj.handle_event(e, 1)  # Should not raise
        assert proj.position == 1

    def test_projection_reset(self):
        proj = AccountBalanceProjection()
        e = Event("AccountOpened", {"owner": "Bob", "initial_deposit": 50},
                  EventMetadata(aggregate_id="b1"))
        proj.handle_event(e, 1)
        proj.reset()
        assert proj.balances == {}
        assert proj.position == 0

    def test_event_count_projection(self):
        proj = EventCountProjection()
        proj.handle_event(Event("A"), 1)
        proj.handle_event(Event("B"), 2)
        proj.handle_event(Event("A"), 3)
        assert proj.counts == {"A": 2, "B": 1}


class TestProjectionManager:
    def setup_method(self):
        self.store = EventStore()
        self.mgr = ProjectionManager(self.store)

    def test_register_and_start(self):
        proj = AccountBalanceProjection()
        self.mgr.register(proj)
        assert proj.state == ProjectionState.STOPPED

        # Add some events
        self.store.append([
            Event("AccountOpened", {"owner": "Alice", "initial_deposit": 100}),
        ], "BankAccount", "a1")

        self.mgr.start("AccountBalances")
        assert proj.state == ProjectionState.RUNNING
        assert proj.balances["a1"] == 100

    def test_start_all(self):
        p1 = AccountBalanceProjection()
        p2 = EventCountProjection()
        self.mgr.register(p1)
        self.mgr.register(p2)

        self.store.append([Event("AccountOpened", {"owner": "X", "initial_deposit": 0})],
                          "BankAccount", "x1")

        self.mgr.start_all()
        assert p1.state == ProjectionState.RUNNING
        assert p2.state == ProjectionState.RUNNING

    def test_rebuild(self):
        proj = AccountBalanceProjection()
        self.mgr.register(proj)

        self.store.append([
            Event("AccountOpened", {"owner": "A", "initial_deposit": 50}),
        ], "BankAccount", "a1")

        self.mgr.start("AccountBalances")
        assert proj.balances["a1"] == 50

        # Add more events
        self.store.append([
            Event("MoneyDeposited", {"amount": 100}),
        ], "BankAccount", "a1")

        # Rebuild catches up
        self.mgr.rebuild("AccountBalances")
        assert proj.balances["a1"] == 150

    def test_process_event_live(self):
        proj = EventCountProjection()
        self.mgr.register(proj)
        proj.state = ProjectionState.RUNNING

        e = Event("TestEvent", {}, EventMetadata(aggregate_id="x"))
        self.mgr.process_event(e)
        assert proj.counts.get("TestEvent") == 1

    def test_process_event_skips_stopped(self):
        proj = EventCountProjection()
        self.mgr.register(proj)
        # Stopped by default
        self.mgr.process_event(Event("X"))
        assert proj.counts == {}

    def test_error_handling_in_projection(self):
        class BrokenProjection(Projection):
            def __init__(self):
                super().__init__("Broken")
            def handle_event(self, event, position):
                raise RuntimeError("broken!")

        proj = BrokenProjection()
        self.mgr.register(proj)
        proj.state = ProjectionState.RUNNING
        self.mgr.process_event(Event("X"))
        assert proj.state == ProjectionState.ERROR

    def test_status(self):
        p1 = AccountBalanceProjection()
        self.mgr.register(p1)
        status = self.mgr.status()
        assert "AccountBalances" in status
        assert status["AccountBalances"]["state"] == "STOPPED"

    def test_unregister(self):
        proj = AccountBalanceProjection()
        self.mgr.register(proj)
        assert self.mgr.unregister("AccountBalances") is True
        assert self.mgr.get("AccountBalances") is None

    def test_projection_names(self):
        self.mgr.register(AccountBalanceProjection())
        self.mgr.register(EventCountProjection())
        assert set(self.mgr.projection_names) == {"AccountBalances", "EventCounts"}


# ============================================================
# 8. EventBus Tests
# ============================================================

class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("OrderPlaced", lambda e: received.append(e))
        bus.publish(Event("OrderPlaced", {"item": "book"}))
        assert len(received) == 1
        assert received[0].data['item'] == "book"

    def test_type_filtering(self):
        bus = EventBus()
        received = []
        bus.subscribe("TypeA", lambda e: received.append(e))
        bus.publish(Event("TypeA"))
        bus.publish(Event("TypeB"))
        assert len(received) == 1

    def test_wildcard_subscription(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish(Event("A"))
        bus.publish(Event("B"))
        assert len(received) == 2

    def test_multiple_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe("X", lambda e: r1.append(e))
        bus.subscribe("X", lambda e: r2.append(e))
        bus.publish(Event("X"))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe("X", handler)
        bus.publish(Event("X"))
        bus.unsubscribe("X", handler)
        bus.publish(Event("X"))
        assert len(received) == 1

    def test_unsubscribe_wildcard(self):
        bus = EventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe("*", handler)
        bus.unsubscribe("*", handler)
        bus.publish(Event("A"))
        assert len(received) == 0

    def test_publish_all(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish_all([Event("A"), Event("B"), Event("C")])
        assert len(received) == 3

    def test_dead_letter_on_error(self):
        bus = EventBus()
        def bad_handler(e):
            raise RuntimeError("fail")
        bus.subscribe("X", bad_handler)
        bus.publish(Event("X"))
        assert bus.dead_letter_count == 1
        dl = bus.get_dead_letters()
        assert "fail" in dl[0][1]

    def test_clear_dead_letters(self):
        bus = EventBus()
        bus.subscribe("X", lambda e: 1/0)
        bus.publish(Event("X"))
        bus.clear_dead_letters()
        assert bus.dead_letter_count == 0

    def test_handler_count(self):
        bus = EventBus()
        bus.subscribe("A", lambda e: None)
        bus.subscribe("A", lambda e: None)
        bus.subscribe("B", lambda e: None)
        assert bus.handler_count("A") == 2
        assert bus.handler_count("B") == 1

    def test_handler_count_total(self):
        bus = EventBus()
        bus.subscribe("A", lambda e: None)
        bus.subscribe("*", lambda e: None)
        assert bus.handler_count() == 2


# ============================================================
# 9. Saga Tests
# ============================================================

class TestSaga:
    def test_create_saga(self):
        saga = Saga(name="TestSaga")
        assert saga.state == SagaState.NOT_STARTED
        assert saga.name == "TestSaga"

    def test_execute_success(self):
        saga = Saga()
        log = []
        saga.add_step("step1", lambda ctx: log.append("s1"))
        saga.add_step("step2", lambda ctx: log.append("s2"))
        assert saga.execute() is True
        assert saga.state == SagaState.COMPLETED
        assert log == ["s1", "s2"]

    def test_execute_failure_compensates(self):
        saga = Saga()
        log = []
        saga.add_step("step1",
                       lambda ctx: log.append("do1"),
                       lambda ctx: log.append("undo1"))
        saga.add_step("step2",
                       lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")),
                       lambda ctx: log.append("undo2"))
        assert saga.execute() is False
        assert saga.state == SagaState.COMPENSATED
        assert "do1" in log
        assert "undo1" in log  # step1 was completed, so compensated
        assert "undo2" not in log  # step2 failed, not compensated

    def test_progress(self):
        saga = Saga()
        saga.add_step("s1", lambda ctx: None)
        saga.add_step("s2", lambda ctx: None)
        assert saga.progress == 0.0
        saga.execute()
        assert saga.progress == 1.0

    def test_context_data_sharing(self):
        saga = Saga()
        saga.context.data['initial'] = 42

        def step1(ctx):
            ctx.data['result'] = ctx.data['initial'] * 2

        def step2(ctx):
            ctx.data['final'] = ctx.data['result'] + 1

        saga.add_step("s1", step1)
        saga.add_step("s2", step2)
        saga.execute()
        assert saga.context.data['final'] == 85

    def test_saga_emits_events(self):
        saga = Saga()
        def step(ctx):
            ctx.emit_event("OrderCreated", {"order_id": "o1"})
        saga.add_step("create", step)
        saga.execute()
        assert len(saga.context.events) == 1
        assert saga.context.events[0].event_type == "OrderCreated"

    def test_saga_emits_commands(self):
        saga = Saga()
        def step(ctx):
            ctx.emit_command("DebitAccount", {"amount": 100}, aggregate_id="acc-1")
        saga.add_step("debit", step)
        saga.execute()
        assert len(saga.context.commands) == 1

    def test_error_message(self):
        saga = Saga()
        saga.add_step("bad", lambda ctx: (_ for _ in ()).throw(ValueError("oops")))
        saga.execute()
        assert "oops" in saga.error

    def test_completed_steps(self):
        saga = Saga()
        saga.add_step("a", lambda ctx: None)
        saga.add_step("b", lambda ctx: None)
        saga.add_step("c", lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")))
        saga.execute()
        assert saga.completed_steps == ["a", "b"]

    def test_compensation_failure(self):
        saga = Saga()
        saga.add_step("s1", lambda ctx: None,
                       lambda ctx: (_ for _ in ()).throw(RuntimeError("comp fail")))
        saga.add_step("s2", lambda ctx: (_ for _ in ()).throw(RuntimeError("fail")))
        saga.execute()
        assert saga.state == SagaState.FAILED

    def test_empty_saga(self):
        saga = Saga()
        assert saga.execute() is True
        assert saga.state == SagaState.COMPLETED


class TestSagaManager:
    def test_execute_saga(self):
        mgr = SagaManager()
        saga = Saga()
        saga.add_step("s1", lambda ctx: None)
        assert mgr.execute(saga) is True
        assert mgr.get(saga.saga_id) is not None

    def test_status(self):
        mgr = SagaManager()
        saga = Saga(name="OrderSaga")
        saga.add_step("s1", lambda ctx: None)
        mgr.execute(saga)
        status = mgr.status()
        assert saga.saga_id in status
        assert status[saga.saga_id]['state'] == "COMPLETED"

    def test_dispatches_commands(self):
        dispatched = []
        bus = CommandBus()
        bus.register("DoThing", lambda cmd: (
            dispatched.append(cmd.command_type),
            CommandResult(success=True)
        )[1])

        mgr = SagaManager(command_bus=bus)
        saga = Saga()
        saga.add_step("s1", lambda ctx: ctx.emit_command("DoThing", {}))
        mgr.execute(saga)
        assert "DoThing" in dispatched

    def test_publishes_events(self):
        published = []
        bus = EventBus()
        bus.subscribe("*", lambda e: published.append(e.event_type))

        mgr = SagaManager(event_bus=bus)
        saga = Saga()
        saga.add_step("s1", lambda ctx: ctx.emit_event("Done", {}))
        mgr.execute(saga)
        assert "Done" in published


# ============================================================
# 10. EventSourcingSystem (Integration) Tests
# ============================================================

class TestEventSourcingSystem:
    def setup_method(self):
        self.system = EventSourcingSystem(snapshot_interval=10)

    def test_create_system(self):
        assert self.system.event_store is not None
        assert self.system.command_bus is not None
        assert self.system.projection_manager is not None

    def test_aggregate_lifecycle(self):
        acc = BankAccount("acc-1")
        acc.open("Alice", 100)
        acc.deposit(50)
        self.system.save_aggregate(acc)

        loaded = self.system.load_aggregate(BankAccount, "acc-1")
        assert loaded.balance == 150
        assert loaded.version == 2

    def test_command_dispatch_integration(self):
        def handle_create(cmd):
            acc = BankAccount(cmd.aggregate_id)
            acc.open(cmd.data['owner'], cmd.data.get('deposit', 0))
            self.system.save_aggregate(acc)
            return CommandResult(success=True, aggregate_id=acc.aggregate_id)

        self.system.register_command("CreateAccount", handle_create)
        result = self.system.dispatch(Command(
            "CreateAccount", {"owner": "Bob", "deposit": 200},
            aggregate_id="acc-1"
        ))
        assert result.success is True
        acc = self.system.load_aggregate(BankAccount, "acc-1")
        assert acc.balance == 200

    def test_projection_receives_events(self):
        proj = AccountBalanceProjection()
        self.system.register_projection(proj)
        proj.state = ProjectionState.RUNNING

        acc = BankAccount("acc-1")
        acc.open("Charlie", 300)
        self.system.save_aggregate(acc)

        assert proj.balances.get("acc-1") == 300

    def test_projection_rebuild(self):
        acc = BankAccount("acc-1")
        acc.open("Dave", 100)
        acc.deposit(50)
        self.system.save_aggregate(acc)

        proj = AccountBalanceProjection()
        self.system.register_projection(proj)
        # Not started yet -- doesn't have data
        assert proj.balances == {}

        self.system.projection_manager.start("AccountBalances")
        assert proj.balances["acc-1"] == 150

    def test_event_bus_integration(self):
        received = []
        self.system.event_bus.subscribe("AccountOpened",
                                        lambda e: received.append(e))

        acc = BankAccount("acc-1")
        acc.open("Eve")
        self.system.save_aggregate(acc)
        assert len(received) == 1

    def test_full_cqrs_flow(self):
        """Full CQRS: command -> aggregate -> events -> projection."""
        # Register projection
        proj = AccountBalanceProjection()
        self.system.register_projection(proj)
        proj.state = ProjectionState.RUNNING

        # Register command handler
        def handle_create(cmd):
            acc = BankAccount(cmd.aggregate_id)
            acc.open(cmd.data['owner'], cmd.data.get('deposit', 0))
            self.system.save_aggregate(acc)
            return CommandResult(success=True, aggregate_id=acc.aggregate_id)

        def handle_deposit(cmd):
            acc = self.system.load_aggregate(BankAccount, cmd.aggregate_id)
            acc.deposit(cmd.data['amount'])
            self.system.save_aggregate(acc)
            return CommandResult(success=True)

        self.system.register_command("CreateAccount", handle_create)
        self.system.register_command("Deposit", handle_deposit)

        # Write side: dispatch commands
        self.system.dispatch(Command("CreateAccount",
                                     {"owner": "Frank", "deposit": 100},
                                     aggregate_id="acc-1"))
        self.system.dispatch(Command("Deposit",
                                     {"amount": 50},
                                     aggregate_id="acc-1"))

        # Read side: query projection
        assert proj.balances["acc-1"] == 150
        assert proj.account_owners["acc-1"] == "Frank"

    def test_multiple_aggregates(self):
        proj = TotalDepositsProjection()
        self.system.register_projection(proj)
        proj.state = ProjectionState.RUNNING

        for i in range(3):
            acc = BankAccount(f"acc-{i}")
            acc.open(f"User{i}", 0)
            acc.deposit(100)
            self.system.save_aggregate(acc)

        assert proj.total == 300
        assert proj.count == 3

    def test_saga_integration(self):
        # Setup command handlers
        accounts_created = []

        def handle_create(cmd):
            acc = BankAccount(cmd.aggregate_id)
            acc.open(cmd.data['owner'])
            self.system.save_aggregate(acc)
            accounts_created.append(cmd.aggregate_id)
            return CommandResult(success=True)

        self.system.register_command("CreateAccount", handle_create)

        # Create saga that creates two accounts
        saga = Saga(name="CreateTwoAccounts")
        saga.add_step("create_first",
                       lambda ctx: ctx.emit_command("CreateAccount",
                                                    {"owner": "A"},
                                                    aggregate_id="acc-1"))
        saga.add_step("create_second",
                       lambda ctx: ctx.emit_command("CreateAccount",
                                                    {"owner": "B"},
                                                    aggregate_id="acc-2"))
        self.system.execute_saga(saga)
        assert len(accounts_created) == 2

    def test_stats(self):
        acc = BankAccount("acc-1")
        acc.open("Stat")
        self.system.save_aggregate(acc)

        stats = self.system.stats()
        assert stats['event_count'] == 1
        assert stats['snapshot_count'] == 0

    def test_snapshot_through_system(self):
        system = EventSourcingSystem(snapshot_interval=3)
        acc = BankAccount("acc-1")
        acc.open("Snap", 0)
        acc.deposit(10)
        acc.deposit(20)
        system.save_aggregate(acc)  # 3 events -> snapshot

        assert system.snapshot_store.has_snapshot("BankAccount", "acc-1")

    def test_cross_aggregate_event_stream(self):
        acc = BankAccount("acc-1")
        acc.open("X")
        self.system.save_aggregate(acc)

        cart = ShoppingCart("cart-1")
        cart.add_item("sku-1", "W", 1, 10.0)
        self.system.save_aggregate(cart)

        all_events = self.system.event_store.get_all_events()
        assert len(all_events) == 2
        types = {e.event_type for e in all_events}
        assert "AccountOpened" in types
        assert "ItemAdded" in types

    def test_event_count_projection_integration(self):
        proj = EventCountProjection()
        self.system.register_projection(proj)
        proj.state = ProjectionState.RUNNING

        acc = BankAccount("acc-1")
        acc.open("Y", 0)
        acc.deposit(10)
        acc.deposit(20)
        self.system.save_aggregate(acc)

        assert proj.counts.get("AccountOpened") == 1
        assert proj.counts.get("MoneyDeposited") == 2

    def test_projection_rebuild_after_new_events(self):
        acc = BankAccount("acc-1")
        acc.open("Z", 100)
        self.system.save_aggregate(acc)

        proj = AccountBalanceProjection()
        self.system.register_projection(proj)
        self.system.projection_manager.start("AccountBalances")
        assert proj.balances["acc-1"] == 100

        # New events
        loaded = self.system.load_aggregate(BankAccount, "acc-1")
        loaded.deposit(50)
        self.system.save_aggregate(loaded)
        # Live projection catches this
        assert proj.balances["acc-1"] == 150

        # Rebuild from scratch
        self.system.rebuild_projection("AccountBalances")
        assert proj.balances["acc-1"] == 150

    def test_account_close_lifecycle(self):
        proj = AccountBalanceProjection()
        self.system.register_projection(proj)
        proj.state = ProjectionState.RUNNING

        acc = BankAccount("acc-1")
        acc.open("Closer", 100)
        self.system.save_aggregate(acc)
        assert proj.balances["acc-1"] == 100

        loaded = self.system.load_aggregate(BankAccount, "acc-1")
        loaded.withdraw(100)
        loaded.close()
        self.system.save_aggregate(loaded)

        assert "acc-1" not in proj.balances

    def test_shopping_cart_through_system(self):
        cart = ShoppingCart("cart-1")
        cart.add_item("s1", "Book", 2, 15.0)
        cart.add_item("s2", "Pen", 5, 2.0)
        self.system.save_aggregate(cart)

        loaded = self.system.load_aggregate(ShoppingCart, "cart-1")
        assert loaded.total == 40.0
        assert len(loaded.items) == 2

        loaded.remove_item("s2")
        loaded.checkout()
        self.system.save_aggregate(loaded)

        reloaded = self.system.load_aggregate(ShoppingCart, "cart-1")
        assert reloaded.checked_out is True
        assert len(reloaded.items) == 1

    def test_start_projections(self):
        p1 = AccountBalanceProjection()
        p2 = EventCountProjection()
        self.system.register_projection(p1)
        self.system.register_projection(p2)
        self.system.start_projections()
        assert p1.state == ProjectionState.RUNNING
        assert p2.state == ProjectionState.RUNNING


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_event_list_append(self):
        store = EventStore()
        v = store.append([], "X", "x-1")
        assert v == 0

    def test_event_from_dict_missing_fields(self):
        e = Event.from_dict({'event_type': 'Minimal'})
        assert e.event_type == "Minimal"
        assert e.data == {}
        assert e.metadata.version == 0

    def test_aggregate_no_handler(self):
        """Events without handlers are applied (version bumps) but state unchanged."""
        acc = BankAccount("acc-1")
        events = [Event("UnknownEvent", {"x": 1})]
        acc.load_from_events(events)
        assert acc.version == 1
        assert acc.balance == 0  # No handler, no state change

    def test_saga_with_no_compensation(self):
        saga = Saga()
        saga.add_step("s1", lambda ctx: None)  # No compensation
        saga.add_step("s2", lambda ctx: (_ for _ in ()).throw(RuntimeError()))
        saga.execute()
        # Step1 has no compensation, so compensation phase succeeds trivially
        assert saga.state == SagaState.COMPENSATED

    def test_multiple_event_stores_isolation(self):
        store1 = EventStore()
        store2 = EventStore()
        store1.append([Event("A")], "X", "x-1")
        assert store2.count() == 0

    def test_projection_error_does_not_affect_others(self):
        store = EventStore()
        mgr = ProjectionManager(store)

        good_proj = EventCountProjection()
        bad_proj = type('Bad', (Projection,), {
            '__init__': lambda self: Projection.__init__(self, "Bad"),
            'handle_event': lambda self, e, p: (_ for _ in ()).throw(RuntimeError()),
        })()

        mgr.register(good_proj)
        mgr.register(bad_proj)
        good_proj.state = ProjectionState.RUNNING
        bad_proj.state = ProjectionState.RUNNING

        mgr.process_event(Event("X"))
        assert good_proj.counts.get("X") == 1
        assert bad_proj.state == ProjectionState.ERROR

    def test_large_event_stream(self):
        store = EventStore()
        for i in range(100):
            store.append([Event(f"E{i}", {"i": i})], "Bulk", "b-1")
        assert store.count() == 100
        assert store.get_version("Bulk", "b-1") == 100
        events = store.load_events("Bulk", "b-1", from_version=95)
        assert len(events) == 5

    def test_concurrent_saga_instances(self):
        mgr = SagaManager()
        sagas = []
        for i in range(5):
            s = Saga(name=f"Saga{i}")
            s.add_step("step", lambda ctx, i=i: ctx.data.update({"id": i}))
            mgr.execute(s)
            sagas.append(s)
        assert all(s.state == SagaState.COMPLETED for s in sagas)
        assert len(mgr.status()) == 5

    def test_event_correlation(self):
        """Events carry correlation/causation IDs for tracing."""
        acc = BankAccount("acc-1")
        acc.raise_event("AccountOpened", {"owner": "Trace"},
                        correlation_id="req-123", causation_id="cmd-456")
        event = acc.uncommitted_events[0]
        assert event.metadata.correlation_id == "req-123"
        assert event.metadata.causation_id == "cmd-456"

    def test_system_with_custom_broker(self):
        from distributed_log import Broker
        broker = Broker("custom-broker")
        system = EventSourcingSystem(broker=broker)
        acc = BankAccount("acc-1")
        acc.open("Custom")
        system.save_aggregate(acc)
        loaded = system.load_aggregate(BankAccount, "acc-1")
        assert loaded.owner == "Custom"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
