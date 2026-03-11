"""
C219: Query Planner -- Cost-based query planner with lock-aware optimization.

Composes:
- C210 Query Optimizer (SQL parsing, logical/physical plans, cost estimation)
- C216 Lock Manager (hierarchical locking, deadlock detection)

Adds beyond C210:
- Parameterized queries with plan caching
- Lock strategy planning (row vs page vs table locking)
- Lock cost integration into plan costing
- Multi-statement transaction planning
- Adaptive re-planning based on statistics changes
- Plan cache with LRU eviction and schema invalidation
- Concurrency-aware cost model
"""

import sys, os, math, time, hashlib, re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C210_query_optimizer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C216_lock_manager'))

from query_optimizer import (
    Catalog, TableDef, ColumnStats, IndexDef, CostParams, CostEstimator,
    QueryOptimizer, parse_sql, explain,
    LogicalOp, LogicalScan, LogicalFilter, LogicalProject, LogicalJoin,
    LogicalAggregate, LogicalSort, LogicalLimit, LogicalDistinct,
    PhysicalOp, SeqScan, IndexScan, HashJoin, MergeJoin, NestedLoopJoin,
    PhysicalFilter, PhysicalProject, PhysicalSort, HashAggregate,
    SortAggregate, PhysicalLimit, PhysicalDistinct,
    SelectStmt, ColumnRef, Literal, BinExpr, FuncCall, AliasedExpr,
    TableRef, JoinClause, OrderByItem
)

from lock_manager import (
    LockManager, LockMode, ResourceId, ResourceLevel, LockResult,
    LockManagerAnalyzer, TwoPhaseLockHelper, MultiGranularityLocker,
    DeadlockError, LockTimeoutError, LockEscalationError,
    make_db, make_table, make_page, make_row
)


# ---------------------------------------------------------------------------
# Lock Strategy
# ---------------------------------------------------------------------------

class LockGranularity(Enum):
    """Granularity at which locks are acquired during plan execution."""
    ROW = auto()
    PAGE = auto()
    TABLE = auto()


class LockStrategy(Enum):
    """How locks are acquired during query execution."""
    NO_LOCK = auto()        # Read-only, snapshot isolation
    ROW_SHARED = auto()     # S locks on individual rows
    ROW_EXCLUSIVE = auto()  # X locks on individual rows
    PAGE_SHARED = auto()    # S locks on pages
    PAGE_EXCLUSIVE = auto() # X locks on pages
    TABLE_SHARED = auto()   # S lock on whole table
    TABLE_EXCLUSIVE = auto()# X lock on whole table


@dataclass
class LockPlan:
    """Describes the locking strategy for a plan node."""
    strategy: LockStrategy = LockStrategy.NO_LOCK
    granularity: LockGranularity = LockGranularity.ROW
    estimated_locks: int = 0
    escalation_likely: bool = False
    lock_cost: float = 0.0
    tables: list = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"strategy={self.strategy.name}"]
        if self.estimated_locks > 0:
            parts.append(f"locks={self.estimated_locks}")
        if self.escalation_likely:
            parts.append("escalation_likely")
        if self.lock_cost > 0:
            parts.append(f"lock_cost={self.lock_cost:.2f}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Annotated Plan (physical plan + lock plan + metadata)
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedPlan:
    """A physical plan annotated with lock strategy and metadata."""
    physical: PhysicalOp
    lock_plan: LockPlan
    total_cost: float = 0.0
    estimated_rows: float = 0.0
    tables_accessed: list = field(default_factory=list)
    indexes_used: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    plan_id: str = ""

    def explain(self, show_locks: bool = True) -> str:
        lines = []
        lines.append(f"Plan ID: {self.plan_id}")
        lines.append(f"Total Cost: {self.total_cost:.2f}")
        lines.append(f"Estimated Rows: {self.estimated_rows:.0f}")
        if self.tables_accessed:
            lines.append(f"Tables: {', '.join(self.tables_accessed)}")
        if self.indexes_used:
            lines.append(f"Indexes: {', '.join(self.indexes_used)}")
        if show_locks:
            lines.append(f"Lock Strategy: {self.lock_plan.summary()}")
        if self.warnings:
            lines.append(f"Warnings: {'; '.join(self.warnings)}")
        lines.append("")
        lines.append(explain(self.physical))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan Cache
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A cached query plan."""
    plan: AnnotatedPlan
    sql_template: str
    created_at: float
    hit_count: int = 0
    last_hit: float = 0.0
    schema_version: int = 0


class PlanCache:
    """LRU plan cache with schema version invalidation."""

    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._schema_version = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, cache_key: str) -> Optional[AnnotatedPlan]:
        entry = self._cache.get(cache_key)
        if entry is None:
            self._misses += 1
            return None
        if entry.schema_version != self._schema_version:
            del self._cache[cache_key]
            self._misses += 1
            return None
        entry.hit_count += 1
        entry.last_hit = time.time()
        self._cache.move_to_end(cache_key)
        self._hits += 1
        return entry.plan

    def put(self, cache_key: str, plan: AnnotatedPlan, sql_template: str):
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            self._cache[cache_key] = CacheEntry(
                plan=plan, sql_template=sql_template,
                created_at=time.time(), schema_version=self._schema_version
            )
            return
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
            self._evictions += 1
        self._cache[cache_key] = CacheEntry(
            plan=plan, sql_template=sql_template,
            created_at=time.time(), schema_version=self._schema_version
        )

    def invalidate(self):
        self._schema_version += 1

    def invalidate_table(self, table_name: str):
        to_remove = []
        for key, entry in self._cache.items():
            if table_name in entry.sql_template:
                to_remove.append(key)
        for key in to_remove:
            del self._cache[key]

    def clear(self):
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': self._hits / max(1, self._hits + self._misses),
            'schema_version': self._schema_version,
        }


# ---------------------------------------------------------------------------
# Query Parameterization
# ---------------------------------------------------------------------------

_PARAM_PLACEHOLDER = '$'

def parameterize_sql(sql: str) -> tuple:
    """Extract literal values from SQL, replace with $N placeholders.
    Returns (template, params_dict).
    """
    params = {}
    param_idx = [0]
    result = []
    i = 0
    in_string = False
    string_char = None
    current_string = []

    while i < len(sql):
        ch = sql[i]

        if in_string:
            if ch == string_char:
                # End of string
                val = ''.join(current_string)
                param_idx[0] += 1
                key = f"${param_idx[0]}"
                params[key] = val
                result.append(key)
                in_string = False
                i += 1
                continue
            current_string.append(ch)
            i += 1
            continue

        if ch in ("'", '"'):
            in_string = True
            string_char = ch
            current_string = []
            i += 1
            continue

        # Check for numbers (int or float)
        if ch.isdigit() or (ch == '-' and i + 1 < len(sql) and sql[i + 1].isdigit()):
            # Only parameterize if preceded by operator/space/paren
            if i == 0 or sql[i - 1] in (' ', '=', '<', '>', '!', '(', ',', '+', '-', '*', '/'):
                num_start = i
                if ch == '-':
                    i += 1
                while i < len(sql) and (sql[i].isdigit() or sql[i] == '.'):
                    i += 1
                num_str = sql[num_start:i]
                # Don't parameterize if it looks like part of identifier
                if i >= len(sql) or not sql[i].isalpha():
                    try:
                        val = int(num_str) if '.' not in num_str else float(num_str)
                        param_idx[0] += 1
                        key = f"${param_idx[0]}"
                        params[key] = val
                        result.append(key)
                        continue
                    except ValueError:
                        pass
                result.append(num_str)
                continue

        result.append(ch)
        i += 1

    return ''.join(result), params


def sql_cache_key(sql_template: str) -> str:
    """Generate a cache key from a parameterized SQL template."""
    normalized = ' '.join(sql_template.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Lock Cost Model
# ---------------------------------------------------------------------------

@dataclass
class LockCostParams:
    """Parameters for estimating lock overhead."""
    lock_acquire_cost: float = 0.05      # Cost per lock acquire
    lock_release_cost: float = 0.02      # Cost per lock release
    intention_lock_cost: float = 0.03    # Cost per intention lock
    escalation_cost: float = 1.0         # Fixed cost for escalation
    contention_factor: float = 0.1       # Extra cost per concurrent tx
    deadlock_check_cost: float = 0.5     # Cost of deadlock detection


class LockCostEstimator:
    """Estimates the cost of lock operations for a plan."""

    def __init__(self, params: LockCostParams = None,
                 escalation_threshold: int = 100,
                 concurrent_txs: int = 1):
        self.params = params or LockCostParams()
        self.escalation_threshold = escalation_threshold
        self.concurrent_txs = concurrent_txs

    def estimate_lock_cost(self, strategy: LockStrategy,
                           estimated_rows: float,
                           is_write: bool = False) -> LockPlan:
        """Estimate lock cost for a given strategy and row count."""
        plan = LockPlan(strategy=strategy)

        if strategy == LockStrategy.NO_LOCK:
            return plan

        if strategy in (LockStrategy.TABLE_SHARED, LockStrategy.TABLE_EXCLUSIVE):
            plan.granularity = LockGranularity.TABLE
            plan.estimated_locks = 2  # intention on DB + table lock
            plan.lock_cost = (
                self.params.intention_lock_cost +
                self.params.lock_acquire_cost +
                self.params.lock_release_cost * 2
            )
        elif strategy in (LockStrategy.PAGE_SHARED, LockStrategy.PAGE_EXCLUSIVE):
            plan.granularity = LockGranularity.PAGE
            est_pages = max(1, int(estimated_rows / 100))
            plan.estimated_locks = est_pages + 2  # intention on DB + table + page locks
            plan.lock_cost = (
                self.params.intention_lock_cost * 2 +
                plan.estimated_locks * (self.params.lock_acquire_cost +
                                        self.params.lock_release_cost)
            )
        else:  # ROW level
            plan.granularity = LockGranularity.ROW
            plan.estimated_locks = int(estimated_rows) + 2  # intention + row locks
            plan.lock_cost = (
                self.params.intention_lock_cost * 2 +
                plan.estimated_locks * (self.params.lock_acquire_cost +
                                        self.params.lock_release_cost)
            )
            # Check escalation
            if plan.estimated_locks > self.escalation_threshold:
                plan.escalation_likely = True
                plan.lock_cost += self.params.escalation_cost

        # Contention factor
        if self.concurrent_txs > 1:
            plan.lock_cost *= (1 + self.params.contention_factor *
                               (self.concurrent_txs - 1))

        return plan

    def choose_strategy(self, estimated_rows: float,
                        total_rows: float,
                        is_write: bool = False,
                        has_index: bool = False) -> LockStrategy:
        """Choose optimal lock strategy based on access pattern."""
        if total_rows == 0:
            return LockStrategy.NO_LOCK

        selectivity = estimated_rows / max(1, total_rows)

        if is_write:
            if selectivity > 0.3:
                return LockStrategy.TABLE_EXCLUSIVE
            elif estimated_rows > self.escalation_threshold:
                return LockStrategy.TABLE_EXCLUSIVE
            elif has_index and estimated_rows <= 10:
                return LockStrategy.ROW_EXCLUSIVE
            elif estimated_rows > 50:
                return LockStrategy.PAGE_EXCLUSIVE
            else:
                return LockStrategy.ROW_EXCLUSIVE
        else:
            if selectivity > 0.5:
                return LockStrategy.TABLE_SHARED
            elif estimated_rows > self.escalation_threshold * 2:
                return LockStrategy.TABLE_SHARED
            elif has_index and estimated_rows <= 20:
                return LockStrategy.ROW_SHARED
            elif estimated_rows > 100:
                return LockStrategy.PAGE_SHARED
            else:
                return LockStrategy.ROW_SHARED


# ---------------------------------------------------------------------------
# Statistics Tracker (for adaptive re-planning)
# ---------------------------------------------------------------------------

@dataclass
class TableStats:
    """Runtime statistics for a table."""
    actual_row_count: int = 0
    actual_scans: int = 0
    actual_index_lookups: int = 0
    last_analyzed: float = 0.0
    stale: bool = False


class StatsTracker:
    """Tracks runtime statistics and detects staleness."""

    def __init__(self, staleness_threshold: float = 0.2):
        self._stats: dict[str, TableStats] = {}
        self.staleness_threshold = staleness_threshold

    def record_scan(self, table: str, actual_rows: int):
        if table not in self._stats:
            self._stats[table] = TableStats()
        s = self._stats[table]
        s.actual_scans += 1
        s.actual_row_count = actual_rows
        s.last_analyzed = time.time()

    def record_index_lookup(self, table: str, actual_rows: int):
        if table not in self._stats:
            self._stats[table] = TableStats()
        s = self._stats[table]
        s.actual_index_lookups += 1
        s.actual_row_count = actual_rows
        s.last_analyzed = time.time()

    def check_staleness(self, catalog: Catalog) -> list:
        """Check if catalog statistics diverge from actual observations."""
        stale_tables = []
        for table_name, stats in self._stats.items():
            if stats.actual_row_count == 0:
                continue
            table_def = catalog.get_table(table_name)
            if table_def is None:
                continue
            ratio = abs(table_def.row_count - stats.actual_row_count) / max(1, table_def.row_count)
            if ratio > self.staleness_threshold:
                stats.stale = True
                stale_tables.append((table_name, table_def.row_count, stats.actual_row_count))
        return stale_tables

    def get_stats(self, table: str) -> Optional[TableStats]:
        return self._stats.get(table)

    def all_stats(self) -> dict:
        return {k: v for k, v in self._stats.items()}


# ---------------------------------------------------------------------------
# Plan Analyzer
# ---------------------------------------------------------------------------

def collect_tables(plan: PhysicalOp) -> list:
    """Collect all tables accessed by a physical plan."""
    tables = []
    if isinstance(plan, (SeqScan, IndexScan)):
        tables.append(plan.table)
    for child in plan.children():
        tables.extend(collect_tables(child))
    return list(dict.fromkeys(tables))  # unique, preserve order


def collect_indexes(plan: PhysicalOp) -> list:
    """Collect all indexes used by a physical plan."""
    indexes = []
    if isinstance(plan, IndexScan) and plan.index:
        indexes.append(plan.index)
    for child in plan.children():
        indexes.extend(collect_indexes(child))
    return list(dict.fromkeys(indexes))


def count_joins(plan: PhysicalOp) -> int:
    """Count joins in a plan."""
    count = 0
    if isinstance(plan, (HashJoin, MergeJoin, NestedLoopJoin)):
        count = 1
    for child in plan.children():
        count += count_joins(child)
    return count


def plan_depth(plan: PhysicalOp) -> int:
    """Calculate plan tree depth."""
    children = plan.children()
    if not children:
        return 1
    return 1 + max(plan_depth(c) for c in children)


def has_seq_scan(plan: PhysicalOp) -> bool:
    """Check if plan contains any sequential scans."""
    if isinstance(plan, SeqScan):
        return True
    return any(has_seq_scan(c) for c in plan.children())


def has_sort(plan: PhysicalOp) -> bool:
    """Check if plan contains any sorts."""
    if isinstance(plan, PhysicalSort):
        return True
    return any(has_sort(c) for c in plan.children())


# ---------------------------------------------------------------------------
# Transaction Plan
# ---------------------------------------------------------------------------

@dataclass
class StatementPlan:
    """Plan for a single SQL statement in a transaction."""
    sql: str
    annotated_plan: AnnotatedPlan
    statement_type: str = "SELECT"  # SELECT, UPDATE, DELETE, INSERT
    lock_order: list = field(default_factory=list)  # Ordered resources to lock


@dataclass
class TransactionPlan:
    """Plan for an entire transaction (multiple statements)."""
    tx_id: int
    statements: list = field(default_factory=list)  # list[StatementPlan]
    lock_order: list = field(default_factory=list)  # Global lock order
    total_cost: float = 0.0
    deadlock_risk: str = "low"  # low, medium, high
    warnings: list = field(default_factory=list)

    def add_statement(self, stmt_plan: StatementPlan):
        self.statements.append(stmt_plan)
        self.total_cost += stmt_plan.annotated_plan.total_cost
        # Merge lock orders
        for res in stmt_plan.lock_order:
            if res not in self.lock_order:
                self.lock_order.append(res)

    def explain(self) -> str:
        lines = [f"Transaction Plan (tx={self.tx_id})"]
        lines.append(f"Total Cost: {self.total_cost:.2f}")
        lines.append(f"Deadlock Risk: {self.deadlock_risk}")
        lines.append(f"Statements: {len(self.statements)}")
        if self.lock_order:
            lines.append(f"Lock Order: {' -> '.join(self.lock_order)}")
        if self.warnings:
            lines.append(f"Warnings: {'; '.join(self.warnings)}")
        lines.append("")
        for i, stmt in enumerate(self.statements):
            lines.append(f"--- Statement {i + 1}: {stmt.statement_type} ---")
            lines.append(stmt.annotated_plan.explain())
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query Planner (main entry point)
# ---------------------------------------------------------------------------

class QueryPlanner:
    """Cost-based query planner with lock-aware optimization.

    Extends C210 QueryOptimizer with:
    - Lock strategy planning integrated into cost model
    - Plan caching with parameterized queries
    - Multi-statement transaction planning
    - Adaptive re-planning based on statistics drift
    - Concurrency-aware cost estimation
    """

    def __init__(self, catalog: Catalog,
                 cost_params: CostParams = None,
                 lock_cost_params: LockCostParams = None,
                 escalation_threshold: int = 100,
                 concurrent_txs: int = 1,
                 cache_size: int = 256,
                 enable_cache: bool = True):
        self.catalog = catalog
        self.optimizer = QueryOptimizer(catalog, cost_params)
        self.lock_estimator = LockCostEstimator(
            params=lock_cost_params,
            escalation_threshold=escalation_threshold,
            concurrent_txs=concurrent_txs
        )
        self.cache = PlanCache(max_size=cache_size) if enable_cache else None
        self.stats_tracker = StatsTracker()
        self._plan_counter = 0
        self._concurrent_txs = concurrent_txs
        self._escalation_threshold = escalation_threshold

    def plan(self, sql: str, is_write: bool = False,
             use_cache: bool = True, params: dict = None) -> AnnotatedPlan:
        """Plan a single SQL query.

        Args:
            sql: SQL query string
            is_write: Whether this is a write operation (UPDATE/DELETE/INSERT)
            use_cache: Whether to use/populate the plan cache
            params: Parameter values for parameterized queries

        Returns:
            AnnotatedPlan with physical plan and lock strategy
        """
        # Parameterize and check cache
        template, extracted_params = parameterize_sql(sql)
        if params:
            extracted_params.update(params)
        cache_key = sql_cache_key(template + ("_W" if is_write else "_R"))

        if use_cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Generate physical plan via C210 optimizer
        physical = self.optimizer.optimize(sql)

        # Collect plan metadata
        tables = collect_tables(physical)
        indexes = collect_indexes(physical)

        # Compute lock strategy for each table
        lock_plan = self._plan_locks(physical, tables, is_write)

        # Compute total cost (optimizer cost + lock cost)
        optimizer_cost = physical.estimated_cost if hasattr(physical, 'estimated_cost') else 0
        total_cost = optimizer_cost + lock_plan.lock_cost

        # Generate warnings
        warnings = self._generate_warnings(physical, lock_plan, tables)

        # Build annotated plan
        self._plan_counter += 1
        annotated = AnnotatedPlan(
            physical=physical,
            lock_plan=lock_plan,
            total_cost=total_cost,
            estimated_rows=physical.estimated_rows if hasattr(physical, 'estimated_rows') else 0,
            tables_accessed=tables,
            indexes_used=indexes,
            warnings=warnings,
            plan_id=f"P{self._plan_counter:04d}"
        )

        # Cache the plan
        if use_cache and self.cache:
            self.cache.put(cache_key, annotated, template)

        return annotated

    def plan_transaction(self, tx_id: int,
                         statements: list) -> TransactionPlan:
        """Plan a multi-statement transaction.

        Args:
            tx_id: Transaction ID
            statements: List of (sql, is_write) tuples

        Returns:
            TransactionPlan with ordered lock acquisition
        """
        tx_plan = TransactionPlan(tx_id=tx_id)

        all_tables = []
        for sql, is_write in statements:
            annotated = self.plan(sql, is_write=is_write)
            stmt_type = "UPDATE" if is_write else "SELECT"
            if sql.strip().upper().startswith("DELETE"):
                stmt_type = "DELETE"
            elif sql.strip().upper().startswith("INSERT"):
                stmt_type = "INSERT"
            elif sql.strip().upper().startswith("UPDATE"):
                stmt_type = "UPDATE"

            lock_order = sorted(annotated.tables_accessed)
            stmt_plan = StatementPlan(
                sql=sql,
                annotated_plan=annotated,
                statement_type=stmt_type,
                lock_order=lock_order
            )
            tx_plan.add_statement(stmt_plan)
            all_tables.extend(annotated.tables_accessed)

        # Global lock order: alphabetical (canonical order prevents deadlocks)
        tx_plan.lock_order = sorted(set(all_tables))

        # Assess deadlock risk
        tx_plan.deadlock_risk = self._assess_deadlock_risk(tx_plan)

        # Generate transaction-level warnings
        if tx_plan.deadlock_risk == "high":
            tx_plan.warnings.append("High deadlock risk: consider reordering statements")
        if len(tx_plan.statements) > 5:
            tx_plan.warnings.append("Long transaction: consider breaking into smaller units")

        return tx_plan

    def explain(self, sql: str, is_write: bool = False,
                show_locks: bool = True) -> str:
        """Return EXPLAIN output for a query."""
        annotated = self.plan(sql, is_write=is_write, use_cache=False)
        return annotated.explain(show_locks=show_locks)

    def explain_analyze(self, sql: str, actual_rows: dict = None,
                        is_write: bool = False) -> str:
        """EXPLAIN ANALYZE: show plan with actual vs estimated comparison."""
        annotated = self.plan(sql, is_write=is_write, use_cache=False)
        lines = [annotated.explain(show_locks=True)]

        if actual_rows:
            lines.append("\n--- Actual vs Estimated ---")
            for table, actual in actual_rows.items():
                table_def = self.catalog.get_table(table)
                if table_def:
                    estimated = table_def.row_count
                    ratio = actual / max(1, estimated)
                    status = "OK" if 0.5 <= ratio <= 2.0 else "DRIFT"
                    lines.append(
                        f"  {table}: estimated={estimated}, actual={actual}, "
                        f"ratio={ratio:.2f} [{status}]"
                    )
                    # Record in stats tracker
                    self.stats_tracker.record_scan(table, actual)

        return "\n".join(lines)

    def check_replan(self) -> list:
        """Check if any plans need re-optimization due to statistics drift."""
        stale = self.stats_tracker.check_staleness(self.catalog)
        if stale and self.cache:
            for table_name, _, _ in stale:
                self.cache.invalidate_table(table_name)
        return stale

    def update_statistics(self, table_name: str, row_count: int,
                          column_stats: dict = None):
        """Update catalog statistics for a table (triggers replan check)."""
        table_def = self.catalog.get_table(table_name)
        if table_def is None:
            return
        table_def.row_count = row_count
        if column_stats:
            for col_name, stats in column_stats.items():
                col = table_def.get_column(col_name)
                if col:
                    for attr, val in stats.items():
                        if hasattr(col, attr):
                            setattr(col, attr, val)
        if self.cache:
            self.cache.invalidate_table(table_name)

    def add_index(self, index: IndexDef):
        """Add an index and invalidate relevant cached plans."""
        self.catalog.add_index(index)
        table_def = self.catalog.get_table(index.table)
        if table_def:
            table_def.indexes.append(index)
        if self.cache:
            self.cache.invalidate_table(index.table)

    def drop_index(self, index_name: str, table_name: str):
        """Remove an index and invalidate cached plans."""
        table_def = self.catalog.get_table(table_name)
        if table_def:
            table_def.indexes = [i for i in table_def.indexes if i.name != index_name]
        if table_name in self.catalog.indexes:
            del self.catalog.indexes[index_name]
        if self.cache:
            self.cache.invalidate_table(table_name)

    def set_concurrency(self, concurrent_txs: int):
        """Update concurrency level for cost estimation."""
        self._concurrent_txs = concurrent_txs
        self.lock_estimator.concurrent_txs = concurrent_txs
        if self.cache:
            self.cache.invalidate()

    def cache_stats(self) -> dict:
        """Return plan cache statistics."""
        if self.cache:
            return self.cache.stats()
        return {'enabled': False}

    # -------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------

    def _plan_locks(self, physical: PhysicalOp,
                    tables: list, is_write: bool) -> LockPlan:
        """Plan lock strategy for a physical plan."""
        total_lock_cost = 0.0
        total_locks = 0
        escalation_likely = False
        strategies = []

        for table_name in tables:
            table_def = self.catalog.get_table(table_name)
            if table_def is None:
                continue

            # Estimate rows accessed for this table
            rows = self._estimate_rows_for_table(physical, table_name)
            has_idx = any(isinstance(n, IndexScan) and n.table == table_name
                         for n in self._collect_nodes(physical))

            strategy = self.lock_estimator.choose_strategy(
                estimated_rows=rows,
                total_rows=table_def.row_count,
                is_write=is_write,
                has_index=has_idx
            )
            lp = self.lock_estimator.estimate_lock_cost(
                strategy, rows, is_write
            )
            total_lock_cost += lp.lock_cost
            total_locks += lp.estimated_locks
            if lp.escalation_likely:
                escalation_likely = True
            strategies.append((table_name, strategy))

        # Pick the dominant strategy
        if not strategies:
            return LockPlan(strategy=LockStrategy.NO_LOCK)

        dominant = max(strategies, key=lambda x: x[1].value)
        return LockPlan(
            strategy=dominant[1],
            granularity=self._strategy_to_granularity(dominant[1]),
            estimated_locks=total_locks,
            escalation_likely=escalation_likely,
            lock_cost=total_lock_cost,
            tables=[t for t, _ in strategies]
        )

    def _estimate_rows_for_table(self, plan: PhysicalOp,
                                 table_name: str) -> float:
        """Estimate how many rows of a specific table are accessed."""
        if isinstance(plan, SeqScan) and plan.table == table_name:
            return plan.estimated_rows if hasattr(plan, 'estimated_rows') else 0
        if isinstance(plan, IndexScan) and plan.table == table_name:
            return plan.estimated_rows if hasattr(plan, 'estimated_rows') else 0

        total = 0
        for child in plan.children():
            total += self._estimate_rows_for_table(child, table_name)
        return total

    def _collect_nodes(self, plan: PhysicalOp) -> list:
        """Collect all nodes in a plan tree."""
        nodes = [plan]
        for child in plan.children():
            nodes.extend(self._collect_nodes(child))
        return nodes

    def _strategy_to_granularity(self, strategy: LockStrategy) -> LockGranularity:
        if strategy in (LockStrategy.TABLE_SHARED, LockStrategy.TABLE_EXCLUSIVE):
            return LockGranularity.TABLE
        if strategy in (LockStrategy.PAGE_SHARED, LockStrategy.PAGE_EXCLUSIVE):
            return LockGranularity.PAGE
        return LockGranularity.ROW

    def _generate_warnings(self, physical: PhysicalOp,
                           lock_plan: LockPlan, tables: list) -> list:
        """Generate plan warnings."""
        warnings = []
        if has_seq_scan(physical) and any(
            self.catalog.get_table(t) and self.catalog.get_table(t).row_count > 10000
            for t in tables
        ):
            warnings.append("Sequential scan on large table -- consider adding an index")
        if lock_plan.escalation_likely:
            warnings.append("Lock escalation likely -- consider table-level lock")
        if count_joins(physical) > 3:
            warnings.append("Complex multi-join -- verify join order")
        if plan_depth(physical) > 8:
            warnings.append("Deep plan tree -- may indicate suboptimal structure")
        return warnings

    def _assess_deadlock_risk(self, tx_plan: TransactionPlan) -> str:
        """Assess deadlock risk for a transaction plan."""
        # Multiple write statements accessing overlapping tables = risk
        write_stmts = [s for s in tx_plan.statements if s.statement_type != "SELECT"]
        if len(write_stmts) <= 1:
            return "low"

        # Check for overlapping table access across write statements
        table_sets = [set(s.annotated_plan.tables_accessed) for s in write_stmts]
        overlaps = 0
        for i in range(len(table_sets)):
            for j in range(i + 1, len(table_sets)):
                if table_sets[i] & table_sets[j]:
                    overlaps += 1

        if overlaps == 0:
            return "low"
        elif overlaps <= 2:
            return "medium"
        else:
            return "high"


# ---------------------------------------------------------------------------
# Plan Comparator (compare alternative plans)
# ---------------------------------------------------------------------------

class PlanComparator:
    """Compare multiple plans for the same query."""

    def __init__(self, planner: QueryPlanner):
        self.planner = planner

    def compare_strategies(self, sql: str,
                           strategies: list = None) -> list:
        """Generate plans with different lock strategies and compare costs."""
        if strategies is None:
            strategies = [False, True]  # read vs write

        results = []
        for is_write in strategies:
            plan = self.planner.plan(sql, is_write=is_write, use_cache=False)
            results.append({
                'is_write': is_write,
                'total_cost': plan.total_cost,
                'lock_strategy': plan.lock_plan.strategy.name,
                'lock_cost': plan.lock_plan.lock_cost,
                'estimated_rows': plan.estimated_rows,
                'plan_id': plan.plan_id,
            })
        return results

    def compare_with_without_index(self, sql: str,
                                   index: IndexDef) -> dict:
        """Compare plan cost with and without an index."""
        # Plan without index
        plan_without = self.planner.plan(sql, use_cache=False)
        cost_without = plan_without.total_cost

        # Add index temporarily
        self.planner.add_index(index)
        plan_with = self.planner.plan(sql, use_cache=False)
        cost_with = plan_with.total_cost

        # Remove index
        self.planner.drop_index(index.name, index.table)

        return {
            'without_index': {
                'cost': cost_without,
                'plan': explain(plan_without.physical),
            },
            'with_index': {
                'cost': cost_with,
                'index': index.name,
                'plan': explain(plan_with.physical),
            },
            'improvement': (cost_without - cost_with) / max(0.01, cost_without),
        }


# ---------------------------------------------------------------------------
# Index Advisor
# ---------------------------------------------------------------------------

class IndexAdvisor:
    """Recommends indexes based on query workload."""

    def __init__(self, planner: QueryPlanner):
        self.planner = planner
        self._workload: list = []

    def record_query(self, sql: str, frequency: int = 1):
        """Record a query in the workload."""
        self._workload.append((sql, frequency))

    def recommend(self, max_recommendations: int = 5) -> list:
        """Recommend indexes based on recorded workload."""
        recommendations = []
        filter_columns = {}  # table -> {column -> frequency}

        for sql, freq in self._workload:
            try:
                ast = parse_sql(sql)
                # Extract filter columns from WHERE clause
                if ast.where:
                    cols = self._extract_filter_columns(ast.where)
                    for table, col in cols:
                        if table not in filter_columns:
                            filter_columns[table] = {}
                        filter_columns[table][col] = (
                            filter_columns[table].get(col, 0) + freq
                        )
                # Extract join columns
                for join in ast.joins:
                    if join.condition:
                        cols = self._extract_filter_columns(join.condition)
                        for table, col in cols:
                            if table not in filter_columns:
                                filter_columns[table] = {}
                            filter_columns[table][col] = (
                                filter_columns[table].get(col, 0) + freq * 2
                            )
            except Exception:
                continue

        # Generate recommendations
        for table, cols in filter_columns.items():
            table_def = self.planner.catalog.get_table(table)
            if table_def is None:
                continue

            existing_idx_cols = set()
            for idx in table_def.indexes:
                existing_idx_cols.update(idx.columns)

            sorted_cols = sorted(cols.items(), key=lambda x: -x[1])
            for col, freq in sorted_cols:
                if col in existing_idx_cols:
                    continue
                recommendations.append({
                    'table': table,
                    'column': col,
                    'frequency': freq,
                    'index_name': f"idx_{table}_{col}",
                    'reason': f"Column '{col}' used in {freq} filter/join operations",
                })

        recommendations.sort(key=lambda x: -x['frequency'])
        return recommendations[:max_recommendations]

    def _extract_filter_columns(self, expr) -> list:
        """Extract (table, column) pairs from a filter expression."""
        results = []
        if isinstance(expr, BinExpr):
            if expr.op in ('=', '!=', '<', '>', '<=', '>=', 'LIKE'):
                if isinstance(expr.left, ColumnRef):
                    table = expr.left.table or self._resolve_column_table(expr.left.column)
                    results.append((table, expr.left.column))
                if isinstance(expr.right, ColumnRef):
                    table = expr.right.table or self._resolve_column_table(expr.right.column)
                    results.append((table, expr.right.column))
            elif expr.op in ('AND', 'OR'):
                results.extend(self._extract_filter_columns(expr.left))
                results.extend(self._extract_filter_columns(expr.right))
        return results

    def _resolve_column_table(self, column: str) -> str:
        """Resolve an unqualified column name to its table."""
        catalog = self.planner.catalog
        for table_name in catalog.tables:
            table_def = catalog.get_table(table_name)
            if table_def and table_def.get_column(column):
                return table_name
        return ''


# ---------------------------------------------------------------------------
# Query Planner Analyzer (diagnostic reports)
# ---------------------------------------------------------------------------

class QueryPlannerAnalyzer:
    """Diagnostic reports for the query planner."""

    def __init__(self, planner: QueryPlanner):
        self.planner = planner

    def cache_report(self) -> dict:
        """Report on plan cache effectiveness."""
        stats = self.planner.cache_stats()
        report = {
            'cache_stats': stats,
            'recommendation': None,
        }
        if stats.get('hit_rate', 0) < 0.3 and stats.get('misses', 0) > 10:
            report['recommendation'] = "Low cache hit rate -- consider parameterizing queries"
        elif stats.get('evictions', 0) > stats.get('size', 0):
            report['recommendation'] = "High eviction rate -- consider increasing cache size"
        return report

    def staleness_report(self) -> dict:
        """Report on statistics staleness."""
        stale = self.planner.check_replan()
        return {
            'stale_tables': [
                {'table': t, 'catalog_rows': c, 'actual_rows': a,
                 'drift': abs(c - a) / max(1, c)}
                for t, c, a in stale
            ],
            'action_needed': len(stale) > 0,
        }

    def workload_report(self, queries: list) -> dict:
        """Analyze a workload (list of SQL strings)."""
        plans = []
        total_cost = 0
        tables_used = {}
        indexes_used = {}
        scan_types = {'seq_scan': 0, 'index_scan': 0}

        for sql in queries:
            try:
                plan = self.planner.plan(sql, use_cache=False)
                plans.append(plan)
                total_cost += plan.total_cost

                for t in plan.tables_accessed:
                    tables_used[t] = tables_used.get(t, 0) + 1
                for idx in plan.indexes_used:
                    indexes_used[idx] = indexes_used.get(idx, 0) + 1

                if has_seq_scan(plan.physical):
                    scan_types['seq_scan'] += 1
                else:
                    scan_types['index_scan'] += 1
            except Exception:
                continue

        return {
            'query_count': len(queries),
            'total_cost': total_cost,
            'avg_cost': total_cost / max(1, len(queries)),
            'tables_used': tables_used,
            'indexes_used': indexes_used,
            'scan_types': scan_types,
            'seq_scan_pct': scan_types['seq_scan'] / max(1, len(queries)),
        }


# ---------------------------------------------------------------------------
# Prepared Statement
# ---------------------------------------------------------------------------

class PreparedStatement:
    """A prepared (parameterized) SQL statement with cached plan."""

    def __init__(self, planner: QueryPlanner, sql: str,
                 is_write: bool = False):
        self.planner = planner
        self.original_sql = sql
        self.is_write = is_write
        self.template, self.param_slots = parameterize_sql(sql)
        self._plan: Optional[AnnotatedPlan] = None
        self._schema_version = planner.cache._schema_version if planner.cache else 0
        self.execution_count = 0

    def plan(self, params: dict = None) -> AnnotatedPlan:
        """Get execution plan, using cached plan if available."""
        current_version = (self.planner.cache._schema_version
                           if self.planner.cache else 0)
        if self._plan is None or self._schema_version != current_version:
            self._plan = self.planner.plan(
                self.original_sql, is_write=self.is_write, use_cache=False
            )
            self._schema_version = current_version
        self.execution_count += 1
        return self._plan

    def explain(self) -> str:
        plan = self.plan()
        return plan.explain()

    @property
    def param_count(self) -> int:
        return len(self.param_slots)


# ---------------------------------------------------------------------------
# Lock Executor (bridges planner -> lock manager)
# ---------------------------------------------------------------------------

class LockExecutor:
    """Executes lock plans using the C216 LockManager."""

    def __init__(self, lock_manager: LockManager, db: str = "default"):
        self.lock_manager = lock_manager
        self.db = db

    def acquire_locks(self, tx_id: int,
                      annotated_plan: AnnotatedPlan) -> list:
        """Acquire locks according to the plan's lock strategy.

        Returns list of (resource_key, lock_mode, result) tuples.
        """
        results = []
        lp = annotated_plan.lock_plan

        if lp.strategy == LockStrategy.NO_LOCK:
            return results

        # Sort tables for consistent lock ordering
        tables = sorted(lp.tables)

        for table_name in tables:
            if lp.strategy == LockStrategy.TABLE_SHARED:
                resource = make_table(self.db, table_name)
                result = self.lock_manager.acquire(
                    tx_id, resource, LockMode.S
                )
                results.append((resource.key, LockMode.S, result))

            elif lp.strategy == LockStrategy.TABLE_EXCLUSIVE:
                resource = make_table(self.db, table_name)
                result = self.lock_manager.acquire(
                    tx_id, resource, LockMode.X
                )
                results.append((resource.key, LockMode.X, result))

            elif lp.strategy in (LockStrategy.ROW_SHARED,
                                 LockStrategy.PAGE_SHARED):
                # Acquire IS on table
                resource = make_table(self.db, table_name)
                result = self.lock_manager.acquire(
                    tx_id, resource, LockMode.IS
                )
                results.append((resource.key, LockMode.IS, result))

            elif lp.strategy in (LockStrategy.ROW_EXCLUSIVE,
                                 LockStrategy.PAGE_EXCLUSIVE):
                # Acquire IX on table
                resource = make_table(self.db, table_name)
                result = self.lock_manager.acquire(
                    tx_id, resource, LockMode.IX
                )
                results.append((resource.key, LockMode.IX, result))

        return results

    def release_locks(self, tx_id: int) -> int:
        """Release all locks held by a transaction."""
        return self.lock_manager.release_all(tx_id)

    def execute_transaction(self, tx_plan: TransactionPlan) -> list:
        """Execute all lock acquisitions for a transaction plan.

        Acquires locks in the global lock order to prevent deadlocks.
        Returns list of (resource_key, mode, result) tuples.
        """
        results = []
        for stmt in tx_plan.statements:
            stmt_results = self.acquire_locks(
                tx_plan.tx_id, stmt.annotated_plan
            )
            results.extend(stmt_results)
        return results
