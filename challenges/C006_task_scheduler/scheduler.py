"""
Task Scheduler with Priorities and Deadlines

A heap-based scheduler that manages tasks with:
- Integer priorities (lower number = higher priority)
- Optional deadlines (datetime)
- Add, cancel, peek, and execute-next operations

Deadline policy: overdue tasks are promoted to effective priority -1
(always scheduled before non-overdue tasks at any priority level).
"""

import heapq
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    CANCELLED = "cancelled"
    EXECUTED = "executed"


@dataclass(order=False)
class Task:
    id: str
    name: str
    priority: int
    deadline: Optional[datetime] = None
    action: Optional[Callable] = None
    status: TaskStatus = field(default=TaskStatus.PENDING)
    created_at: datetime = field(default_factory=datetime.now)

    def is_overdue(self, now: Optional[datetime] = None) -> bool:
        if self.deadline is None:
            return False
        now = now or datetime.now()
        return now > self.deadline

    def effective_priority(self, now: Optional[datetime] = None) -> tuple:
        """Returns a sort key: (overdue_flag, priority, created_at).

        Overdue tasks sort first (0 before 1), then by priority (lower first),
        then by creation time (earlier first).
        """
        overdue = 0 if self.is_overdue(now) else 1
        return (overdue, self.priority, self.created_at)


class Scheduler:
    def __init__(self, clock: Optional[Callable[[], datetime]] = None):
        """Create a scheduler.

        Args:
            clock: Optional callable returning current datetime.
                   Defaults to datetime.now(). Inject for testing.
        """
        self._heap: list[tuple] = []
        self._tasks: dict[str, Task] = {}
        self._counter = 0  # tie-breaker for heap stability
        self._clock = clock or datetime.now

    @property
    def now(self) -> datetime:
        return self._clock()

    def add(self, task_id: str, name: str, priority: int,
            deadline: Optional[datetime] = None,
            action: Optional[Callable] = None) -> Task:
        """Add a task to the scheduler.

        Raises ValueError if task_id already exists and is still pending.
        """
        if task_id in self._tasks and self._tasks[task_id].status == TaskStatus.PENDING:
            raise ValueError(f"Task '{task_id}' already exists and is pending")

        task = Task(
            id=task_id,
            name=name,
            priority=priority,
            deadline=deadline,
            action=action,
            created_at=self.now,
        )
        self._tasks[task_id] = task
        self._counter += 1
        sort_key = task.effective_priority(self.now)
        heapq.heappush(self._heap, (sort_key, self._counter, task_id))
        return task

    def cancel(self, task_id: str) -> Task:
        """Cancel a pending task. Raises KeyError if not found, ValueError if not pending."""
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        task = self._tasks[task_id]
        if task.status != TaskStatus.PENDING:
            raise ValueError(f"Task '{task_id}' is {task.status.value}, cannot cancel")
        task.status = TaskStatus.CANCELLED
        return task

    def peek(self) -> Optional[Task]:
        """Return the highest-priority pending task without removing it.

        Re-evaluates effective priorities (deadlines may have changed status).
        Returns None if no pending tasks exist.
        """
        self._rebuild_heap()
        while self._heap:
            _, _, task_id = self._heap[0]
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                return task
            heapq.heappop(self._heap)
        return None

    def execute_next(self) -> Optional[Task]:
        """Execute the highest-priority pending task.

        Calls task.action() if provided. Marks task as executed.
        Returns the executed task, or None if queue is empty.
        """
        self._rebuild_heap()
        while self._heap:
            _, _, task_id = heapq.heappop(self._heap)
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                if task.action:
                    task.action()
                task.status = TaskStatus.EXECUTED
                return task
        return None

    def get(self, task_id: str) -> Task:
        """Get a task by ID. Raises KeyError if not found."""
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        return self._tasks[task_id]

    def pending_count(self) -> int:
        """Number of tasks still pending."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    def all_tasks(self) -> list[Task]:
        """Return all tasks in insertion order."""
        return list(self._tasks.values())

    def pending_tasks(self) -> list[Task]:
        """Return pending tasks sorted by effective priority."""
        now = self.now
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        pending.sort(key=lambda t: t.effective_priority(now))
        return pending

    def _rebuild_heap(self):
        """Rebuild the heap with current effective priorities.

        This ensures deadline changes are reflected in ordering.
        Only includes pending tasks.
        """
        now = self.now
        new_heap = []
        seen = set()
        for task in self._tasks.values():
            if task.status == TaskStatus.PENDING and task.id not in seen:
                seen.add(task.id)
                self._counter += 1
                sort_key = task.effective_priority(now)
                new_heap.append((sort_key, self._counter, task.id))
        heapq.heapify(new_heap)
        self._heap = new_heap


if __name__ == "__main__":
    # Demo
    s = Scheduler()
    s.add("t1", "Low priority", priority=10)
    s.add("t2", "High priority", priority=1)
    s.add("t3", "Medium priority", priority=5)

    print("Pending tasks:")
    for t in s.pending_tasks():
        print(f"  [{t.priority}] {t.name}")

    print(f"\nPeek: {s.peek().name}")

    executed = s.execute_next()
    print(f"Executed: {executed.name}")

    print(f"Remaining: {s.pending_count()}")
