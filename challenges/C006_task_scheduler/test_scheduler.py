"""Tests for the task scheduler."""

import unittest
from datetime import datetime, timedelta
from scheduler import Scheduler, TaskStatus


class FakeClock:
    """Controllable clock for testing."""
    def __init__(self, start: datetime = None):
        self.time = start or datetime(2026, 3, 9, 12, 0, 0)

    def __call__(self) -> datetime:
        return self.time

    def advance(self, **kwargs):
        self.time += timedelta(**kwargs)


class TestBasicOperations(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.s = Scheduler(clock=self.clock)

    def test_add_single_task(self):
        task = self.s.add("t1", "Task one", priority=5)
        assert task.id == "t1"
        assert task.name == "Task one"
        assert task.priority == 5
        assert task.status == TaskStatus.PENDING

    def test_add_duplicate_pending_raises(self):
        self.s.add("t1", "First", priority=1)
        with self.assertRaises(ValueError):
            self.s.add("t1", "Duplicate", priority=2)

    def test_add_after_cancel_ok(self):
        self.s.add("t1", "First", priority=1)
        self.s.cancel("t1")
        task = self.s.add("t1", "Second", priority=2)
        assert task.name == "Second"

    def test_peek_returns_highest_priority(self):
        self.s.add("low", "Low", priority=10)
        self.s.add("high", "High", priority=1)
        self.s.add("mid", "Mid", priority=5)
        assert self.s.peek().id == "high"

    def test_peek_empty_returns_none(self):
        assert self.s.peek() is None

    def test_execute_next_returns_highest_priority(self):
        self.s.add("low", "Low", priority=10)
        self.s.add("high", "High", priority=1)
        task = self.s.execute_next()
        assert task.id == "high"
        assert task.status == TaskStatus.EXECUTED

    def test_execute_next_empty_returns_none(self):
        assert self.s.execute_next() is None

    def test_execute_order(self):
        self.s.add("c", "C", priority=3)
        self.s.add("a", "A", priority=1)
        self.s.add("b", "B", priority=2)
        order = []
        while True:
            t = self.s.execute_next()
            if t is None:
                break
            order.append(t.id)
        assert order == ["a", "b", "c"]

    def test_pending_count(self):
        self.s.add("t1", "One", priority=1)
        self.s.add("t2", "Two", priority=2)
        assert self.s.pending_count() == 2
        self.s.execute_next()
        assert self.s.pending_count() == 1


class TestCancel(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.s = Scheduler(clock=self.clock)

    def test_cancel_removes_from_scheduling(self):
        self.s.add("t1", "One", priority=1)
        self.s.add("t2", "Two", priority=2)
        self.s.cancel("t1")
        task = self.s.execute_next()
        assert task.id == "t2"

    def test_cancel_nonexistent_raises(self):
        with self.assertRaises(KeyError):
            self.s.cancel("nope")

    def test_cancel_already_executed_raises(self):
        self.s.add("t1", "One", priority=1)
        self.s.execute_next()
        with self.assertRaises(ValueError):
            self.s.cancel("t1")

    def test_cancel_already_cancelled_raises(self):
        self.s.add("t1", "One", priority=1)
        self.s.cancel("t1")
        with self.assertRaises(ValueError):
            self.s.cancel("t1")

    def test_cancel_updates_pending_count(self):
        self.s.add("t1", "One", priority=1)
        self.s.add("t2", "Two", priority=2)
        self.s.cancel("t1")
        assert self.s.pending_count() == 1


class TestDeadlines(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.s = Scheduler(clock=self.clock)

    def test_overdue_task_promoted(self):
        soon = self.clock.time + timedelta(minutes=5)
        later = self.clock.time + timedelta(hours=1)

        self.s.add("high", "High priority", priority=1, deadline=later)
        self.s.add("low", "Low priority but urgent", priority=10, deadline=soon)

        # Before deadline, high priority wins
        assert self.s.peek().id == "high"

        # Advance past the low-priority task's deadline
        self.clock.advance(minutes=10)

        # Now the overdue task should be promoted above everything
        assert self.s.peek().id == "low"

    def test_no_deadline_not_overdue(self):
        self.s.add("t1", "No deadline", priority=5)
        task = self.s.get("t1")
        assert not task.is_overdue(self.clock.time)

    def test_future_deadline_not_overdue(self):
        future = self.clock.time + timedelta(hours=1)
        self.s.add("t1", "Future", priority=5, deadline=future)
        task = self.s.get("t1")
        assert not task.is_overdue(self.clock.time)

    def test_past_deadline_is_overdue(self):
        past = self.clock.time - timedelta(hours=1)
        self.s.add("t1", "Past", priority=5, deadline=past)
        task = self.s.get("t1")
        assert task.is_overdue(self.clock.time)

    def test_multiple_overdue_sorted_by_priority(self):
        soon = self.clock.time + timedelta(minutes=1)
        self.s.add("a", "A", priority=5, deadline=soon)
        self.s.add("b", "B", priority=1, deadline=soon)
        self.s.add("c", "C", priority=10)  # no deadline

        self.clock.advance(minutes=5)

        # Both a and b are overdue; b has lower priority number
        first = self.s.execute_next()
        assert first.id == "b"
        second = self.s.execute_next()
        assert second.id == "a"
        third = self.s.execute_next()
        assert third.id == "c"


class TestActions(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.s = Scheduler(clock=self.clock)

    def test_action_called_on_execute(self):
        results = []
        self.s.add("t1", "With action", priority=1, action=lambda: results.append("ran"))
        self.s.execute_next()
        assert results == ["ran"]

    def test_no_action_executes_fine(self):
        self.s.add("t1", "No action", priority=1)
        task = self.s.execute_next()
        assert task.status == TaskStatus.EXECUTED


class TestPendingTasks(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.s = Scheduler(clock=self.clock)

    def test_pending_tasks_sorted(self):
        self.s.add("c", "C", priority=3)
        self.s.add("a", "A", priority=1)
        self.s.add("b", "B", priority=2)
        ids = [t.id for t in self.s.pending_tasks()]
        assert ids == ["a", "b", "c"]

    def test_pending_excludes_cancelled(self):
        self.s.add("t1", "One", priority=1)
        self.s.add("t2", "Two", priority=2)
        self.s.cancel("t1")
        ids = [t.id for t in self.s.pending_tasks()]
        assert ids == ["t2"]

    def test_pending_excludes_executed(self):
        self.s.add("t1", "One", priority=1)
        self.s.add("t2", "Two", priority=2)
        self.s.execute_next()
        ids = [t.id for t in self.s.pending_tasks()]
        assert ids == ["t2"]


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.s = Scheduler(clock=self.clock)

    def test_same_priority_fifo(self):
        """Tasks with same priority execute in insertion order."""
        self.s.add("first", "First", priority=1)
        self.clock.advance(seconds=1)
        self.s.add("second", "Second", priority=1)
        self.clock.advance(seconds=1)
        self.s.add("third", "Third", priority=1)

        order = []
        while True:
            t = self.s.execute_next()
            if t is None:
                break
            order.append(t.id)
        assert order == ["first", "second", "third"]

    def test_get_task(self):
        self.s.add("t1", "One", priority=1)
        task = self.s.get("t1")
        assert task.name == "One"

    def test_get_nonexistent_raises(self):
        with self.assertRaises(KeyError):
            self.s.get("nope")

    def test_all_tasks(self):
        self.s.add("t1", "One", priority=1)
        self.s.add("t2", "Two", priority=2)
        assert len(self.s.all_tasks()) == 2

    def test_large_batch(self):
        """Schedule 100 tasks, verify correct ordering."""
        for i in range(100):
            self.s.add(f"t{i:03d}", f"Task {i}", priority=99 - i)
        # Highest priority (0) should be task 99
        first = self.s.execute_next()
        assert first.id == "t099"
        assert self.s.pending_count() == 99


if __name__ == "__main__":
    unittest.main()
