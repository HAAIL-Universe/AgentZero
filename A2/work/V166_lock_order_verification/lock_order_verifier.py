"""
V166: Lock Order Verification

Verifies whether a program's lock acquisition order is consistent
(no potential deadlocks) by building a lock-order graph and detecting cycles.

Supports:
- Lock acquire/release event traces per transaction
- Lock order graph construction (acquire A then B => edge A->B)
- Cycle detection (inconsistent ordering = potential deadlock)
- Reporting conflicting lock pairs and transactions involved
- Hierarchical locks (parent lock implicitly locks children)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class EventType(Enum):
    ACQUIRE = auto()
    RELEASE = auto()


@dataclass(frozen=True)
class LockEvent:
    """A single lock acquire or release event."""
    event_type: EventType
    lock_name: str
    timestamp: int = 0  # optional ordering within a transaction


@dataclass
class LockTrace:
    """A sequence of lock events for a single transaction."""
    transaction_id: str
    events: list[LockEvent] = field(default_factory=list)

    def add_acquire(self, lock_name: str, timestamp: int = 0) -> "LockTrace":
        self.events.append(LockEvent(EventType.ACQUIRE, lock_name, timestamp))
        return self

    def add_release(self, lock_name: str, timestamp: int = 0) -> "LockTrace":
        self.events.append(LockEvent(EventType.RELEASE, lock_name, timestamp))
        return self


@dataclass(frozen=True)
class OrderEdge:
    """A directed edge in the lock order graph: from_lock acquired before to_lock."""
    from_lock: str
    to_lock: str
    transaction_id: str


@dataclass
class Cycle:
    """A cycle found in the lock order graph."""
    locks: list[str]  # lock names forming the cycle (first == last)
    edges: list[OrderEdge]  # the edges forming the cycle

    @property
    def lock_pairs(self) -> list[tuple[str, str]]:
        """Return conflicting lock pairs in the cycle."""
        pairs = []
        for i in range(len(self.locks) - 1):
            pairs.append((self.locks[i], self.locks[i + 1]))
        return pairs

    @property
    def transactions(self) -> set[str]:
        """Return all transaction IDs involved in the cycle."""
        return {e.transaction_id for e in self.edges}


@dataclass
class LockHierarchy:
    """Defines a hierarchy of locks (tree structure).
    If a parent lock is held, all children are implicitly locked.
    Children cannot be acquired independently while parent is held.
    """
    parent_map: dict[str, Optional[str]] = field(default_factory=dict)

    def add_root(self, lock_name: str) -> "LockHierarchy":
        self.parent_map[lock_name] = None
        return self

    def add_child(self, child: str, parent: str) -> "LockHierarchy":
        self.parent_map[child] = parent
        return self

    def get_parent(self, lock_name: str) -> Optional[str]:
        return self.parent_map.get(lock_name)

    def get_ancestors(self, lock_name: str) -> list[str]:
        """Return all ancestors from immediate parent to root."""
        ancestors = []
        current = lock_name
        visited = set()
        while current in self.parent_map and self.parent_map[current] is not None:
            parent = self.parent_map[current]
            if parent in visited:
                break
            visited.add(parent)
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_children(self, lock_name: str) -> list[str]:
        """Return immediate children of a lock."""
        return [k for k, v in self.parent_map.items() if v == lock_name]

    def get_all_descendants(self, lock_name: str) -> list[str]:
        """Return all descendants of a lock (BFS)."""
        descendants = []
        queue = self.get_children(lock_name)
        visited = set()
        while queue:
            child = queue.pop(0)
            if child in visited:
                continue
            visited.add(child)
            descendants.append(child)
            queue.extend(self.get_children(child))
        return descendants

    def is_ancestor_of(self, ancestor: str, descendant: str) -> bool:
        """Check if ancestor is an ancestor of descendant."""
        return ancestor in self.get_ancestors(descendant)

    def get_root(self, lock_name: str) -> str:
        """Get root of the hierarchy containing lock_name."""
        current = lock_name
        visited = set()
        while current in self.parent_map and self.parent_map[current] is not None:
            if current in visited:
                return current
            visited.add(current)
            current = self.parent_map[current]
        return current

    def contains(self, lock_name: str) -> bool:
        return lock_name in self.parent_map


@dataclass
class VerificationResult:
    """Result of lock order verification."""
    consistent: bool
    cycles: list[Cycle] = field(default_factory=list)
    order_graph: dict[str, set[str]] = field(default_factory=dict)
    edge_info: dict[tuple[str, str], list[OrderEdge]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    hierarchy_violations: list[str] = field(default_factory=list)

    @property
    def num_cycles(self) -> int:
        return len(self.cycles)

    @property
    def all_conflicting_pairs(self) -> set[tuple[str, str]]:
        pairs = set()
        for c in self.cycles:
            for p in c.lock_pairs:
                pairs.add(p)
        return pairs

    @property
    def all_involved_transactions(self) -> set[str]:
        txns = set()
        for c in self.cycles:
            txns.update(c.transactions)
        return txns


class LockOrderVerifier:
    """Verifies lock ordering consistency across transactions."""

    def __init__(self, hierarchy: Optional[LockHierarchy] = None):
        self.hierarchy = hierarchy
        # order_graph[A] = {B, C} means A was acquired before B and C
        self.order_graph: dict[str, set[str]] = {}
        # edge_info[(A,B)] = [OrderEdge(...), ...] for provenance
        self.edge_info: dict[tuple[str, str], list[OrderEdge]] = {}
        self.traces: list[LockTrace] = []
        self.warnings: list[str] = []
        self.hierarchy_violations: list[str] = []

    def add_trace(self, trace: LockTrace) -> None:
        """Add a transaction's lock trace."""
        self.traces.append(trace)

    def add_traces(self, traces: list[LockTrace]) -> None:
        for t in traces:
            self.add_trace(t)

    def _expand_lock_with_hierarchy(self, lock_name: str) -> list[str]:
        """Expand a lock to include its descendants if hierarchy is defined."""
        if self.hierarchy is None:
            return [lock_name]
        locks = [lock_name]
        locks.extend(self.hierarchy.get_all_descendants(lock_name))
        return locks

    def _check_hierarchy_violation(self, lock_name: str, held_locks: set[str],
                                    tx_id: str) -> None:
        """Check for hierarchy violations when acquiring a lock."""
        if self.hierarchy is None:
            return
        if not self.hierarchy.contains(lock_name):
            return

        # Violation: acquiring parent when child is already held
        descendants = self.hierarchy.get_all_descendants(lock_name)
        for d in descendants:
            if d in held_locks:
                msg = (f"Hierarchy violation in {tx_id}: acquiring parent '{lock_name}' "
                       f"while child '{d}' is already held")
                self.hierarchy_violations.append(msg)

        # Violation: acquiring child when a non-ancestor holds it
        # Actually: check if acquiring a lock whose ancestor is already held
        # (this is fine -- parent before child is correct order)
        # But acquiring a child without the parent is suspicious in strict mode
        # We just note implicit coverage

    def _build_graph_from_trace(self, trace: LockTrace) -> None:
        """Extract lock ordering edges from a single trace."""
        held_locks: list[str] = []  # ordered list of currently held locks
        held_set: set[str] = set()

        for event in trace.events:
            if event.event_type == EventType.ACQUIRE:
                lock = event.lock_name

                # Check hierarchy violations
                self._check_hierarchy_violation(lock, held_set, trace.transaction_id)

                # Get the effective locks (including implicit children from hierarchy)
                effective_locks = self._expand_lock_with_hierarchy(lock)

                # For every currently held lock, add edge: held -> newly acquired
                for held in held_locks:
                    # Add edges from held lock to all effective locks
                    for eff_lock in effective_locks:
                        if held == eff_lock:
                            continue
                        self._add_edge(held, eff_lock, trace.transaction_id)

                    # Also add edges from implicit ancestors/descendants of held
                    if self.hierarchy is not None:
                        held_effective = self._expand_lock_with_hierarchy(held)
                        for he in held_effective:
                            for el in effective_locks:
                                if he == el:
                                    continue
                                self._add_edge(he, el, trace.transaction_id)

                held_locks.append(lock)
                held_set.add(lock)

            elif event.event_type == EventType.RELEASE:
                lock = event.lock_name
                if lock in held_set:
                    held_locks.remove(lock)
                    held_set.remove(lock)
                else:
                    self.warnings.append(
                        f"Release of unheld lock '{lock}' in {trace.transaction_id}")

    def _add_edge(self, from_lock: str, to_lock: str, tx_id: str) -> None:
        """Add a directed edge to the lock order graph."""
        if from_lock not in self.order_graph:
            self.order_graph[from_lock] = set()
        self.order_graph[from_lock].add(to_lock)

        key = (from_lock, to_lock)
        edge = OrderEdge(from_lock, to_lock, tx_id)
        if key not in self.edge_info:
            self.edge_info[key] = []
        # Avoid duplicate edges from same transaction
        if not any(e.transaction_id == tx_id for e in self.edge_info[key]):
            self.edge_info[key].append(edge)

    def _find_all_cycles(self) -> list[Cycle]:
        """Find all elementary cycles in the lock order graph using Johnson's algorithm."""
        # Collect all nodes
        all_nodes = set()
        for n in self.order_graph:
            all_nodes.add(n)
            for m in self.order_graph[n]:
                all_nodes.add(m)

        nodes = sorted(all_nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Build adjacency list with indices
        adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
        for n in self.order_graph:
            for m in self.order_graph[n]:
                adj[node_to_idx[n]].append(node_to_idx[m])

        cycles = []
        # Johnson's elementary cycle detection
        self._johnson_cycles(adj, nodes, cycles)
        return cycles

    def _johnson_cycles(self, adj: dict[int, list[int]], nodes: list[str],
                         result: list[Cycle]) -> None:
        """Johnson's algorithm for finding all elementary cycles."""
        n = len(nodes)
        if n == 0:
            return

        # Simple DFS-based cycle detection
        # For each node as starting point, find cycles
        visited_cycles: set[tuple[str, ...]] = set()

        for start in range(n):
            # DFS from start, looking for paths back to start
            path = [start]
            visited = {start}
            self._dfs_cycles(adj, nodes, start, start, path, visited,
                           visited_cycles, result)

    def _dfs_cycles(self, adj: dict[int, list[int]], nodes: list[str],
                     start: int, current: int, path: list[int],
                     visited: set[int], visited_cycles: set[tuple[str, ...]],
                     result: list[Cycle]) -> None:
        """DFS to find cycles starting and ending at 'start'."""
        for neighbor in adj.get(current, []):
            if neighbor == start and len(path) > 1:
                # Found a cycle
                cycle_nodes = [nodes[i] for i in path] + [nodes[start]]
                # Normalize: start from the smallest element to avoid duplicates
                min_idx = 0
                for i in range(1, len(cycle_nodes) - 1):
                    if cycle_nodes[i] < cycle_nodes[min_idx]:
                        min_idx = i
                normalized = tuple(cycle_nodes[min_idx:len(cycle_nodes)-1])
                if normalized not in visited_cycles:
                    visited_cycles.add(normalized)
                    # Build edges for this cycle
                    edges = []
                    for i in range(len(cycle_nodes) - 1):
                        a, b = cycle_nodes[i], cycle_nodes[i + 1]
                        key = (a, b)
                        if key in self.edge_info and self.edge_info[key]:
                            edges.append(self.edge_info[key][0])
                        else:
                            edges.append(OrderEdge(a, b, "unknown"))
                    result.append(Cycle(locks=cycle_nodes, edges=edges))
            elif neighbor not in visited and neighbor > start:
                # Only explore nodes > start to avoid finding same cycle multiple times
                visited.add(neighbor)
                path.append(neighbor)
                self._dfs_cycles(adj, nodes, start, neighbor, path, visited,
                               visited_cycles, result)
                path.pop()
                visited.remove(neighbor)

    def verify(self) -> VerificationResult:
        """Run verification on all added traces."""
        self.order_graph.clear()
        self.edge_info.clear()
        self.warnings.clear()
        self.hierarchy_violations.clear()

        for trace in self.traces:
            self._build_graph_from_trace(trace)

        cycles = self._find_all_cycles()

        return VerificationResult(
            consistent=len(cycles) == 0 and len(self.hierarchy_violations) == 0,
            cycles=cycles,
            order_graph={k: set(v) for k, v in self.order_graph.items()},
            edge_info=dict(self.edge_info),
            warnings=list(self.warnings),
            hierarchy_violations=list(self.hierarchy_violations),
        )

    def verify_traces(self, traces: list[LockTrace]) -> VerificationResult:
        """Convenience: add traces and verify in one call."""
        self.traces.clear()
        self.add_traces(traces)
        return self.verify()

    def check_new_trace(self, trace: LockTrace) -> VerificationResult:
        """Check if adding a new trace would create inconsistency.
        Does not permanently add the trace."""
        old_traces = list(self.traces)
        old_graph = {k: set(v) for k, v in self.order_graph.items()}
        old_info = dict(self.edge_info)
        old_warnings = list(self.warnings)
        old_violations = list(self.hierarchy_violations)

        self.add_trace(trace)
        result = self.verify()

        # Restore
        self.traces = old_traces
        self.order_graph = old_graph
        self.edge_info = old_info
        self.warnings = old_warnings
        self.hierarchy_violations = old_violations

        return result


def make_trace(tx_id: str, *lock_ops: tuple[str, str]) -> LockTrace:
    """Helper: create a trace from (op, lock) tuples.
    op is 'A' for acquire, 'R' for release.
    Example: make_trace("tx1", ("A", "X"), ("A", "Y"), ("R", "Y"), ("R", "X"))
    """
    trace = LockTrace(transaction_id=tx_id)
    for i, (op, lock) in enumerate(lock_ops):
        if op.upper() == 'A':
            trace.add_acquire(lock, timestamp=i)
        elif op.upper() == 'R':
            trace.add_release(lock, timestamp=i)
    return trace


def make_acquire_sequence(tx_id: str, *locks: str) -> LockTrace:
    """Helper: create a trace that acquires locks in order then releases in reverse.
    Example: make_acquire_sequence("tx1", "A", "B", "C")
    => acquire A, acquire B, acquire C, release C, release B, release A
    """
    trace = LockTrace(transaction_id=tx_id)
    t = 0
    for lock in locks:
        trace.add_acquire(lock, timestamp=t)
        t += 1
    for lock in reversed(locks):
        trace.add_release(lock, timestamp=t)
        t += 1
    return trace
