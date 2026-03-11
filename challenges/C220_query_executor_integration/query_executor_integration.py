"""C220: Query Executor Integration

Full SQL pipeline: parse -> plan -> lock -> execute -> release.
Composes C210 (Query Optimizer) + C211 (Query Execution) + C216 (Lock Manager) + C219 (Query Planner).

Provides:
- IntegratedQueryEngine: unified SQL execution with lock-aware planning
- TransactionContext: explicit BEGIN/COMMIT/ROLLBACK with 2PL
- Auto-commit mode for single statements
- Statistics feedback from actual execution
- EXPLAIN ANALYZE with real row counts
- DDL/DML support through unified interface
"""

import sys
import os
import time
import threading
import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C210_query_optimizer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C211_query_execution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C216_lock_manager'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C219_query_planner'))

from query_optimizer import (
    QueryOptimizer, Catalog, TableDef, ColumnStats, IndexDef, CostParams,
    parse_sql, SelectStmt, LogicalOp, PhysicalOp,
    SeqScan, IndexScan, PhysicalFilter, PhysicalProject,
    HashJoin, MergeJoin, NestedLoopJoin, PhysicalSort,
    HashAggregate, SortAggregate, PhysicalLimit, PhysicalDistinct
)
from query_execution import (
    QueryEngine, Database, Table, eval_expr,
    PlanExecutor, ExecutionContext
)
from lock_manager import (
    LockManager, LockMode, LockResult, ResourceId, ResourceLevel,
    make_db, make_table, make_row, make_page,
    TwoPhaseLockHelper, MultiGranularityLocker,
    WaitForGraph, LockManagerAnalyzer
)
from query_planner import (
    QueryPlanner, AnnotatedPlan, LockStrategy, LockGranularity,
    LockCostEstimator, LockCostParams, LockPlan,
    PlanCache, StatsTracker, TransactionPlan, StatementPlan,
    parameterize_sql, sql_cache_key,
    PlanComparator, IndexAdvisor, QueryPlannerAnalyzer
)


class IsolationLevel(Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = auto()
    READ_COMMITTED = auto()
    REPEATABLE_READ = auto()
    SERIALIZABLE = auto()


class TxState(Enum):
    """Transaction lifecycle states."""
    ACTIVE = auto()
    COMMITTED = auto()
    ABORTED = auto()
    PREPARING = auto()


class StatementType(Enum):
    """SQL statement classification."""
    SELECT = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    CREATE_TABLE = auto()
    DROP_TABLE = auto()
    CREATE_INDEX = auto()
    DROP_INDEX = auto()
    BEGIN = auto()
    COMMIT = auto()
    ROLLBACK = auto()
    EXPLAIN = auto()
    EXPLAIN_ANALYZE = auto()


@dataclass
class ExecutionResult:
    """Result of executing a SQL statement."""
    rows: list = field(default_factory=list)
    affected_rows: int = 0
    columns: list = field(default_factory=list)
    execution_time_ms: float = 0.0
    plan_used: Optional[AnnotatedPlan] = None
    locks_acquired: int = 0
    locks_released: int = 0
    from_cache: bool = False
    statement_type: Optional[StatementType] = None
    message: str = ""

    def __repr__(self):
        if self.rows:
            return f"ExecutionResult({len(self.rows)} rows, {self.execution_time_ms:.1f}ms)"
        return f"ExecutionResult({self.message or f'{self.affected_rows} affected'}, {self.execution_time_ms:.1f}ms)"


@dataclass
class UndoEntry:
    """Undo log entry for rollback support."""
    table_name: str
    operation: str  # 'insert', 'update', 'delete'
    row_data: dict  # original row for delete/update, inserted row for insert
    row_id: Optional[int] = None


class TransactionContext:
    """Manages a single transaction's state, locks, and undo log."""

    _next_tx_id = 1
    _tx_lock = threading.Lock()

    def __init__(self, engine, isolation=IsolationLevel.READ_COMMITTED, tx_id=None):
        if tx_id is not None:
            self.tx_id = tx_id
        else:
            with TransactionContext._tx_lock:
                self.tx_id = TransactionContext._next_tx_id
                TransactionContext._next_tx_id += 1
        self.engine = engine
        self.isolation = isolation
        self.state = TxState.ACTIVE
        self.undo_log: list[UndoEntry] = []
        self.locks_held: list[tuple] = []  # (resource, mode)
        self.tables_modified: set = set()
        self.start_time = time.time()
        self.statement_count = 0
        self.rows_read = 0
        self.rows_written = 0

    @property
    def is_active(self):
        return self.state == TxState.ACTIVE

    def record_undo(self, entry: UndoEntry):
        """Record an undo entry for rollback."""
        self.undo_log.append(entry)

    def record_lock(self, resource, mode):
        """Track a lock acquired by this transaction."""
        self.locks_held.append((resource, mode))

    def apply_undo(self, db: Database):
        """Apply undo log in reverse to rollback changes."""
        for entry in reversed(self.undo_log):
            table = db.get_table(entry.table_name)
            if table is None:
                continue
            if entry.operation == 'insert':
                # Remove the inserted row
                table.rows = [r for r in table.rows if r != entry.row_data]
                # Rebuild indexes
                for idx in table.indexes.values():
                    idx['data'].clear()
                    for row in table.rows:
                        key = tuple(row.get(c) for c in idx['columns'])
                        if len(idx['columns']) == 1:
                            key = key[0]
                        idx['data'].setdefault(key, []).append(row)
            elif entry.operation == 'delete':
                # Re-insert the deleted row
                table.rows.append(entry.row_data)
                for idx in table.indexes.values():
                    key = tuple(entry.row_data.get(c) for c in idx['columns'])
                    if len(idx['columns']) == 1:
                        key = key[0]
                    idx['data'].setdefault(key, []).append(entry.row_data)
            elif entry.operation == 'update':
                # Restore original row data
                for i, row in enumerate(table.rows):
                    if id(row) == entry.row_id or row == entry.row_data.get('_new'):
                        for k, v in entry.row_data.get('_old', {}).items():
                            row[k] = v
                        break
        self.undo_log.clear()


class StatementClassifier:
    """Classifies SQL statements by type."""

    _patterns = [
        (r'^\s*SELECT\b', StatementType.SELECT),
        (r'^\s*INSERT\b', StatementType.INSERT),
        (r'^\s*UPDATE\b', StatementType.UPDATE),
        (r'^\s*DELETE\b', StatementType.DELETE),
        (r'^\s*CREATE\s+TABLE\b', StatementType.CREATE_TABLE),
        (r'^\s*DROP\s+TABLE\b', StatementType.DROP_TABLE),
        (r'^\s*CREATE\s+(UNIQUE\s+)?INDEX\b', StatementType.CREATE_INDEX),
        (r'^\s*DROP\s+INDEX\b', StatementType.DROP_INDEX),
        (r'^\s*BEGIN\b', StatementType.BEGIN),
        (r'^\s*COMMIT\b', StatementType.COMMIT),
        (r'^\s*ROLLBACK\b', StatementType.ROLLBACK),
        (r'^\s*EXPLAIN\s+ANALYZE\b', StatementType.EXPLAIN_ANALYZE),
        (r'^\s*EXPLAIN\b', StatementType.EXPLAIN),
    ]

    @staticmethod
    def classify(sql: str) -> StatementType:
        upper = sql.strip().upper()
        for pattern, stype in StatementClassifier._patterns:
            if re.match(pattern, upper):
                return stype
        return StatementType.SELECT  # default


class LockAcquirer:
    """Acquires appropriate locks based on the query plan's lock strategy."""

    def __init__(self, lock_manager: LockManager, db_name: str = "default"):
        self.lock_manager = lock_manager
        self.db_name = db_name

    def acquire_for_plan(self, tx_id: int, plan: AnnotatedPlan,
                         timeout: float = 5.0) -> tuple[LockResult, int]:
        """Acquire locks based on the annotated plan's lock strategy.
        Returns (result, lock_count)."""
        lock_plan = plan.lock_plan
        tables = plan.tables_accessed
        count = 0

        if lock_plan.strategy == LockStrategy.NO_LOCK:
            return LockResult.GRANTED, 0

        for table in tables:
            resource, mode = self._strategy_to_lock(lock_plan.strategy, table)
            result = self.lock_manager.acquire(tx_id, resource, mode, timeout=timeout)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return result, count
            count += 1

        return LockResult.GRANTED, count

    def acquire_table_lock(self, tx_id: int, table: str, mode: LockMode,
                           timeout: float = 5.0) -> LockResult:
        """Acquire a specific table-level lock."""
        resource = make_table(self.db_name, table)
        return self.lock_manager.acquire(tx_id, resource, mode, timeout=timeout)

    def acquire_row_locks(self, tx_id: int, table: str, rows: list,
                          mode: LockMode, timeout: float = 5.0) -> tuple[LockResult, int]:
        """Acquire row-level locks for specific rows."""
        count = 0
        for i, row in enumerate(rows):
            resource = make_row(self.db_name, table, 0, i)
            result = self.lock_manager.acquire(tx_id, resource, mode, timeout=timeout)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return result, count
            count += 1
        return LockResult.GRANTED, count

    def release_all(self, tx_id: int) -> int:
        """Release all locks held by a transaction."""
        return self.lock_manager.release_all(tx_id)

    def _strategy_to_lock(self, strategy: LockStrategy, table: str):
        """Convert a lock strategy to a resource + mode pair."""
        mapping = {
            LockStrategy.ROW_SHARED: (make_table(self.db_name, table), LockMode.IS),
            LockStrategy.ROW_EXCLUSIVE: (make_table(self.db_name, table), LockMode.IX),
            LockStrategy.PAGE_SHARED: (make_table(self.db_name, table), LockMode.IS),
            LockStrategy.PAGE_EXCLUSIVE: (make_table(self.db_name, table), LockMode.IX),
            LockStrategy.TABLE_SHARED: (make_table(self.db_name, table), LockMode.S),
            LockStrategy.TABLE_EXCLUSIVE: (make_table(self.db_name, table), LockMode.X),
        }
        return mapping.get(strategy, (make_table(self.db_name, table), LockMode.S))


class DMLExecutor:
    """Executes INSERT, UPDATE, DELETE statements with lock integration."""

    def __init__(self, db: Database, lock_acquirer: LockAcquirer):
        self.db = db
        self.lock_acquirer = lock_acquirer

    def execute_insert(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute an INSERT statement."""
        start = time.time()
        # Parse: INSERT INTO table (cols) VALUES (vals), ...
        match = re.match(
            r'\s*INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s+(.+)',
            sql.strip(), re.IGNORECASE | re.DOTALL
        )
        if not match:
            # Try without column list
            match2 = re.match(
                r'\s*INSERT\s+INTO\s+(\w+)\s+VALUES\s+(.+)',
                sql.strip(), re.IGNORECASE | re.DOTALL
            )
            if not match2:
                return ExecutionResult(message="Invalid INSERT syntax")
            table_name = match2.group(1)
            columns = None
            values_str = match2.group(2)
        else:
            table_name = match.group(1)
            columns = [c.strip() for c in match.group(2).split(',')]
            values_str = match.group(3)

        table = self.db.get_table(table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{table_name}' not found")

        if columns is None:
            columns = table.columns

        # Acquire lock
        locks = 0
        if tx:
            result = self.lock_acquirer.acquire_table_lock(
                tx.tx_id, table_name, LockMode.IX)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(message=f"Lock denied: {result.name}")
            locks += 1

        # Parse value tuples
        rows_inserted = 0
        for val_match in re.finditer(r'\(([^)]+)\)', values_str):
            vals = self._parse_values(val_match.group(1))
            row = dict(zip(columns, vals))
            if tx:
                tx.record_undo(UndoEntry(table_name, 'insert', row.copy()))
                tx.tables_modified.add(table_name)
                tx.rows_written += 1
            table.insert(row)
            rows_inserted += 1

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            affected_rows=rows_inserted,
            locks_acquired=locks,
            execution_time_ms=elapsed,
            statement_type=StatementType.INSERT,
            message=f"Inserted {rows_inserted} row(s)"
        )

    def execute_update(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute an UPDATE statement."""
        start = time.time()
        match = re.match(
            r'\s*UPDATE\s+(\w+)\s+SET\s+(.+?)(?:\s+WHERE\s+(.+))?\s*$',
            sql.strip(), re.IGNORECASE | re.DOTALL
        )
        if not match:
            return ExecutionResult(message="Invalid UPDATE syntax")

        table_name = match.group(1)
        set_clause = match.group(2)
        where_clause = match.group(3)

        table = self.db.get_table(table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{table_name}' not found")

        # Acquire lock
        locks = 0
        if tx:
            result = self.lock_acquirer.acquire_table_lock(
                tx.tx_id, table_name, LockMode.IX)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(message=f"Lock denied: {result.name}")
            locks += 1

        # Parse SET assignments
        assignments = self._parse_set_clause(set_clause)

        # Apply updates
        updated = 0
        for row in list(table.rows):
            if where_clause and not self._eval_where(where_clause, row):
                continue
            if tx:
                old_vals = {k: row.get(k) for k in assignments}
                new_vals = {k: row.get(k) for k in assignments}
                for col, val in assignments.items():
                    new_vals[col] = val
                tx.record_undo(UndoEntry(
                    table_name, 'update',
                    {'_old': old_vals, '_new': new_vals},
                    row_id=id(row)
                ))
                tx.tables_modified.add(table_name)
                tx.rows_written += 1
            for col, val in assignments.items():
                row[col] = val
            updated += 1

        # Rebuild indexes after update
        if updated > 0:
            self._rebuild_indexes(table)

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            affected_rows=updated,
            locks_acquired=locks,
            execution_time_ms=elapsed,
            statement_type=StatementType.UPDATE,
            message=f"Updated {updated} row(s)"
        )

    def execute_delete(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute a DELETE statement."""
        start = time.time()
        match = re.match(
            r'\s*DELETE\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?\s*$',
            sql.strip(), re.IGNORECASE | re.DOTALL
        )
        if not match:
            return ExecutionResult(message="Invalid DELETE syntax")

        table_name = match.group(1)
        where_clause = match.group(2)

        table = self.db.get_table(table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{table_name}' not found")

        # Acquire lock
        locks = 0
        if tx:
            result = self.lock_acquirer.acquire_table_lock(
                tx.tx_id, table_name, LockMode.X)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(message=f"Lock denied: {result.name}")
            locks += 1

        # Find and remove matching rows
        deleted = 0
        remaining = []
        for row in table.rows:
            if where_clause and not self._eval_where(where_clause, row):
                remaining.append(row)
            else:
                if tx:
                    tx.record_undo(UndoEntry(table_name, 'delete', row.copy()))
                    tx.tables_modified.add(table_name)
                    tx.rows_written += 1
                deleted += 1

        table.rows = remaining
        if deleted > 0:
            self._rebuild_indexes(table)

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            affected_rows=deleted,
            locks_acquired=locks,
            execution_time_ms=elapsed,
            statement_type=StatementType.DELETE,
            message=f"Deleted {deleted} row(s)"
        )

    def _parse_values(self, val_str: str) -> list:
        """Parse comma-separated values from a VALUES clause."""
        values = []
        for v in self._split_values(val_str):
            v = v.strip()
            if v.upper() == 'NULL':
                values.append(None)
            elif v.startswith("'") and v.endswith("'"):
                values.append(v[1:-1])
            elif '.' in v:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)
            else:
                try:
                    values.append(int(v))
                except ValueError:
                    values.append(v)
        return values

    def _split_values(self, s: str) -> list:
        """Split values respecting quoted strings."""
        parts = []
        current = []
        in_quote = False
        for ch in s:
            if ch == "'" and not in_quote:
                in_quote = True
                current.append(ch)
            elif ch == "'" and in_quote:
                in_quote = False
                current.append(ch)
            elif ch == ',' and not in_quote:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _parse_set_clause(self, clause: str) -> dict:
        """Parse SET col=val, col=val assignments."""
        assignments = {}
        for part in self._split_values(clause):
            if '=' in part:
                col, val = part.split('=', 1)
                col = col.strip()
                val = val.strip()
                if val.upper() == 'NULL':
                    assignments[col] = None
                elif val.startswith("'") and val.endswith("'"):
                    assignments[col] = val[1:-1]
                elif '.' in val:
                    try:
                        assignments[col] = float(val)
                    except ValueError:
                        assignments[col] = val
                else:
                    try:
                        assignments[col] = int(val)
                    except ValueError:
                        assignments[col] = val
        return assignments

    def _eval_where(self, where_clause: str, row: dict) -> bool:
        """Simple WHERE clause evaluation for DML."""
        # Handle AND
        if ' AND ' in where_clause.upper():
            parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
            return all(self._eval_where(p, row) for p in parts)
        # Handle OR
        if ' OR ' in where_clause.upper():
            parts = re.split(r'\s+OR\s+', where_clause, flags=re.IGNORECASE)
            return any(self._eval_where(p, row) for p in parts)

        # Handle comparison operators
        for op in ['>=', '<=', '!=', '<>', '=', '>', '<']:
            if op in where_clause:
                col, val = where_clause.split(op, 1)
                col = col.strip()
                val = val.strip()
                if val.startswith("'") and val.endswith("'"):
                    val = val[1:-1]
                elif val.upper() == 'NULL':
                    val = None
                elif '.' in val:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        pass

                row_val = row.get(col)

                if op == '=' or op == '==':
                    return row_val == val
                elif op == '!=' or op == '<>':
                    return row_val != val
                elif op == '>':
                    return row_val is not None and val is not None and row_val > val
                elif op == '<':
                    return row_val is not None and val is not None and row_val < val
                elif op == '>=':
                    return row_val is not None and val is not None and row_val >= val
                elif op == '<=':
                    return row_val is not None and val is not None and row_val <= val

        # IS NULL / IS NOT NULL
        is_null = re.match(r'(\w+)\s+IS\s+NULL', where_clause, re.IGNORECASE)
        if is_null:
            return row.get(is_null.group(1)) is None
        is_not_null = re.match(r'(\w+)\s+IS\s+NOT\s+NULL', where_clause, re.IGNORECASE)
        if is_not_null:
            return row.get(is_not_null.group(1)) is not None

        return True  # unknown clause, pass through

    def _rebuild_indexes(self, table: Table):
        """Rebuild all indexes on a table after modification."""
        for idx in table.indexes.values():
            idx['data'].clear()
            for row in table.rows:
                key = tuple(row.get(c) for c in idx['columns'])
                if len(idx['columns']) == 1:
                    key = key[0]
                idx['data'].setdefault(key, []).append(row)


class DDLExecutor:
    """Executes CREATE TABLE, DROP TABLE, CREATE INDEX, DROP INDEX."""

    def __init__(self, db: Database, catalog: Catalog, planner: QueryPlanner,
                 lock_acquirer: LockAcquirer):
        self.db = db
        self.catalog = catalog
        self.planner = planner
        self.lock_acquirer = lock_acquirer

    def execute_create_table(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute CREATE TABLE."""
        start = time.time()
        match = re.match(
            r'\s*CREATE\s+TABLE\s+(\w+)\s*\((.+)\)\s*$',
            sql.strip(), re.IGNORECASE | re.DOTALL
        )
        if not match:
            return ExecutionResult(message="Invalid CREATE TABLE syntax")

        table_name = match.group(1)
        cols_str = match.group(2)

        # Parse columns
        columns = []
        primary_key = None
        for col_def in self._split_column_defs(cols_str):
            col_def = col_def.strip()
            if col_def.upper().startswith('PRIMARY KEY'):
                pk_match = re.match(r'PRIMARY\s+KEY\s*\((\w+)\)', col_def, re.IGNORECASE)
                if pk_match:
                    primary_key = pk_match.group(1)
                continue
            parts = col_def.split()
            if len(parts) >= 1:
                col_name = parts[0]
                columns.append(col_name)
                if len(parts) > 1 and 'PRIMARY' in col_def.upper():
                    primary_key = col_name

        if not columns:
            return ExecutionResult(message="No columns defined")

        # Acquire schema lock
        locks = 0
        if tx:
            result = self.lock_acquirer.acquire_table_lock(
                tx.tx_id, table_name, LockMode.X)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(message=f"Lock denied: {result.name}")
            locks += 1

        # Create table in database
        table = self.db.create_table(table_name, columns, primary_key)

        # Update catalog
        col_stats = [ColumnStats(c, 0, 0) for c in columns]
        table_def = TableDef(table_name, col_stats, 0)
        self.catalog.add_table(table_def)

        # Invalidate plan cache
        self.planner.cache.invalidate()

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            locks_acquired=locks,
            execution_time_ms=elapsed,
            statement_type=StatementType.CREATE_TABLE,
            message=f"Table '{table_name}' created"
        )

    def execute_drop_table(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute DROP TABLE."""
        start = time.time()
        match = re.match(
            r'\s*DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)\s*$',
            sql.strip(), re.IGNORECASE
        )
        if not match:
            return ExecutionResult(message="Invalid DROP TABLE syntax")

        table_name = match.group(1)
        if_exists = 'IF EXISTS' in sql.upper()

        table = self.db.get_table(table_name)
        if table is None:
            if if_exists:
                return ExecutionResult(
                    statement_type=StatementType.DROP_TABLE,
                    message=f"Table '{table_name}' does not exist (IF EXISTS)")
            return ExecutionResult(message=f"Table '{table_name}' not found")

        # Acquire lock
        locks = 0
        if tx:
            result = self.lock_acquirer.acquire_table_lock(
                tx.tx_id, table_name, LockMode.X)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(message=f"Lock denied: {result.name}")
            locks += 1

        self.db.drop_table(table_name)

        # Remove from catalog
        if hasattr(self.catalog, 'tables'):
            self.catalog.tables.pop(table_name, None)
        elif hasattr(self.catalog, '_tables'):
            self.catalog._tables.pop(table_name, None)

        # Invalidate plan cache for this table
        self.planner.cache.invalidate_table(table_name)

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            locks_acquired=locks,
            execution_time_ms=elapsed,
            statement_type=StatementType.DROP_TABLE,
            message=f"Table '{table_name}' dropped"
        )

    def execute_create_index(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute CREATE INDEX."""
        start = time.time()
        match = re.match(
            r'\s*CREATE\s+(UNIQUE\s+)?INDEX\s+(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)\s*$',
            sql.strip(), re.IGNORECASE
        )
        if not match:
            return ExecutionResult(message="Invalid CREATE INDEX syntax")

        unique = match.group(1) is not None
        index_name = match.group(2)
        table_name = match.group(3)
        columns = [c.strip() for c in match.group(4).split(',')]

        table = self.db.get_table(table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{table_name}' not found")

        # Acquire lock
        locks = 0
        if tx:
            result = self.lock_acquirer.acquire_table_lock(
                tx.tx_id, table_name, LockMode.S)
            if result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(message=f"Lock denied: {result.name}")
            locks += 1

        # Create index on table
        table.create_index(index_name, columns, unique)

        # Update catalog
        index_def = IndexDef(index_name, table_name, columns, unique)
        self.catalog.add_index(index_def)
        self.planner.add_index(index_def)

        # Invalidate cache for this table
        self.planner.cache.invalidate_table(table_name)

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            locks_acquired=locks,
            execution_time_ms=elapsed,
            statement_type=StatementType.CREATE_INDEX,
            message=f"Index '{index_name}' created on {table_name}({', '.join(columns)})"
        )

    def execute_drop_index(self, sql: str, tx: Optional[TransactionContext] = None) -> ExecutionResult:
        """Execute DROP INDEX."""
        start = time.time()
        match = re.match(
            r'\s*DROP\s+INDEX\s+(\w+)\s+ON\s+(\w+)\s*$',
            sql.strip(), re.IGNORECASE
        )
        if not match:
            return ExecutionResult(message="Invalid DROP INDEX syntax")

        index_name = match.group(1)
        table_name = match.group(2)

        table = self.db.get_table(table_name)
        if table is None:
            return ExecutionResult(message=f"Table '{table_name}' not found")

        # Remove index
        if index_name in table.indexes:
            del table.indexes[index_name]

        self.planner.drop_index(index_name, table_name)
        self.planner.cache.invalidate_table(table_name)

        elapsed = (time.time() - start) * 1000
        return ExecutionResult(
            locks_acquired=0,
            execution_time_ms=elapsed,
            statement_type=StatementType.DROP_INDEX,
            message=f"Index '{index_name}' dropped"
        )

    def _split_column_defs(self, s: str) -> list:
        """Split column definitions respecting parentheses."""
        parts = []
        current = []
        depth = 0
        for ch in s:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts


class IntegratedQueryEngine:
    """Full SQL pipeline: parse -> plan -> lock -> execute -> release.

    Unified interface composing:
    - C210 QueryOptimizer (parsing + planning)
    - C211 QueryEngine (execution)
    - C216 LockManager (concurrency control)
    - C219 QueryPlanner (lock-aware planning + caching)
    """

    def __init__(self, db_name: str = "default",
                 isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
                 cost_params: Optional[CostParams] = None,
                 lock_cost_params: Optional[LockCostParams] = None,
                 escalation_threshold: int = 100,
                 concurrent_txs: int = 10,
                 cache_size: int = 256,
                 deadlock_detection: bool = True,
                 lock_timeout: float = 5.0):
        self.db_name = db_name
        self.default_isolation = isolation
        self.lock_timeout = lock_timeout

        # Core components
        self.db = Database()
        self.catalog = Catalog()
        self.lock_manager = LockManager(
            escalation_threshold=escalation_threshold,
            default_timeout=lock_timeout,
            deadlock_detection=deadlock_detection
        )

        # Query engine (C211) -- operates on the database
        self.query_engine = QueryEngine(self.db)

        # Query planner (C219) -- produces lock-aware plans
        self.planner = QueryPlanner(
            self.catalog,
            cost_params=cost_params,
            lock_cost_params=lock_cost_params,
            escalation_threshold=escalation_threshold,
            concurrent_txs=concurrent_txs,
            cache_size=cache_size
        )

        # Lock acquirer -- bridges planner output to lock manager
        self.lock_acquirer = LockAcquirer(self.lock_manager, db_name)

        # DML executor
        self.dml = DMLExecutor(self.db, self.lock_acquirer)

        # DDL executor
        self.ddl = DDLExecutor(self.db, self.catalog, self.planner, self.lock_acquirer)

        # Active transactions
        self._transactions: dict[int, TransactionContext] = {}
        self._tx_lock = threading.Lock()

        # Auto-commit transaction counter
        self._auto_tx_counter = 1000000

        # Statistics
        self.stats = EngineStats()

    # ---- Public API ----

    def execute(self, sql: str, tx_id: Optional[int] = None,
                params: Optional[dict] = None) -> ExecutionResult:
        """Execute a SQL statement. Main entry point.

        If tx_id is provided, executes within that transaction.
        Otherwise, uses auto-commit mode.
        """
        start = time.time()
        stmt_type = StatementClassifier.classify(sql)

        # Handle transaction control
        if stmt_type == StatementType.BEGIN:
            return self._handle_begin()
        elif stmt_type == StatementType.COMMIT:
            return self._handle_commit(tx_id)
        elif stmt_type == StatementType.ROLLBACK:
            return self._handle_rollback(tx_id)
        elif stmt_type == StatementType.EXPLAIN:
            return self._handle_explain(sql, show_locks=True)
        elif stmt_type == StatementType.EXPLAIN_ANALYZE:
            return self._handle_explain_analyze(sql, tx_id)

        # Get or create transaction
        tx = None
        auto_commit = False
        if tx_id is not None:
            tx = self._transactions.get(tx_id)
            if tx is None:
                return ExecutionResult(message=f"Transaction {tx_id} not found")
            if not tx.is_active:
                return ExecutionResult(message=f"Transaction {tx_id} is {tx.state.name}")
        else:
            # Auto-commit: create ephemeral transaction
            tx = self._create_auto_tx()
            auto_commit = True

        try:
            result = self._execute_in_tx(sql, stmt_type, tx, params)
            result.execution_time_ms = (time.time() - start) * 1000

            if auto_commit:
                self.lock_acquirer.release_all(tx.tx_id)
                self.stats.commits += 1

            self.stats.statements_executed += 1
            return result

        except Exception as e:
            if auto_commit:
                tx.apply_undo(self.db)
                self.lock_acquirer.release_all(tx.tx_id)
                self.stats.rollbacks += 1
            raise

    def begin(self, isolation: Optional[IsolationLevel] = None) -> int:
        """Begin a new transaction. Returns tx_id."""
        iso = isolation or self.default_isolation
        tx = TransactionContext(self, iso)
        with self._tx_lock:
            self._transactions[tx.tx_id] = tx
        return tx.tx_id

    def commit(self, tx_id: int) -> ExecutionResult:
        """Commit a transaction."""
        return self._handle_commit(tx_id)

    def rollback(self, tx_id: int) -> ExecutionResult:
        """Rollback a transaction."""
        return self._handle_rollback(tx_id)

    def query(self, sql: str, tx_id: Optional[int] = None) -> list[dict]:
        """Convenience: execute SELECT and return list of dicts."""
        result = self.execute(sql, tx_id)
        return result.rows

    def create_table(self, name: str, columns: list, primary_key: Optional[str] = None) -> ExecutionResult:
        """Create a table via API (not SQL)."""
        pk = f" PRIMARY KEY" if primary_key else ""
        col_defs = []
        for c in columns:
            if c == primary_key:
                col_defs.append(f"{c} INT PRIMARY KEY")
            else:
                col_defs.append(f"{c} TEXT")
        sql = f"CREATE TABLE {name} ({', '.join(col_defs)})"
        return self.execute(sql)

    def insert(self, table_name: str, values: dict) -> ExecutionResult:
        """Insert a row via API."""
        cols = ', '.join(values.keys())
        vals = ', '.join(self._format_val(v) for v in values.values())
        sql = f"INSERT INTO {table_name} ({cols}) VALUES ({vals})"
        return self.execute(sql)

    def insert_many(self, table_name: str, rows: list[dict]) -> ExecutionResult:
        """Insert multiple rows via API."""
        if not rows:
            return ExecutionResult(message="No rows to insert")
        cols = list(rows[0].keys())
        col_str = ', '.join(cols)
        val_strs = []
        for row in rows:
            vals = ', '.join(self._format_val(row.get(c)) for c in cols)
            val_strs.append(f"({vals})")
        sql = f"INSERT INTO {table_name} ({col_str}) VALUES {', '.join(val_strs)}"
        return self.execute(sql)

    def create_index(self, table_name: str, index_name: str,
                     columns: list, unique: bool = False) -> ExecutionResult:
        """Create an index via API."""
        u = "UNIQUE " if unique else ""
        sql = f"CREATE {u}INDEX {index_name} ON {table_name} ({', '.join(columns)})"
        return self.execute(sql)

    def explain(self, sql: str, show_locks: bool = True) -> str:
        """Get EXPLAIN output for a query."""
        result = self._handle_explain(sql, show_locks)
        return result.message

    def explain_analyze(self, sql: str, tx_id: Optional[int] = None) -> str:
        """Get EXPLAIN ANALYZE output with actual execution stats."""
        result = self._handle_explain_analyze(sql, tx_id)
        return result.message

    def get_transaction(self, tx_id: int) -> Optional[TransactionContext]:
        """Get a transaction context."""
        return self._transactions.get(tx_id)

    def active_transactions(self) -> list[int]:
        """List active transaction IDs."""
        return [tid for tid, tx in self._transactions.items() if tx.is_active]

    def sync_catalog(self):
        """Synchronize catalog with actual database state."""
        built = self.db.build_catalog()
        # Merge built catalog into our catalog
        if hasattr(built, 'tables'):
            for name, tdef in built.tables.items():
                self.catalog.add_table(tdef)
        elif hasattr(built, '_tables'):
            for name, tdef in built._tables.items():
                self.catalog.add_table(tdef)

    def update_statistics(self, table_name: str):
        """Update catalog statistics for a table from actual data."""
        table = self.db.get_table(table_name)
        if table is None:
            return
        row_count = len(table)
        col_stats = []
        for col in table.columns:
            values = [r.get(col) for r in table.rows if r.get(col) is not None]
            distinct = len(set(values)) if values else 0
            nulls = sum(1 for r in table.rows if r.get(col) is None)
            min_val = min(values) if values else None
            max_val = max(values) if values else None
            col_stats.append(ColumnStats(col, distinct, nulls, min_val, max_val))

        table_def = TableDef(table_name, col_stats, row_count)
        self.catalog.add_table(table_def)
        # Planner expects dict {col_name: {attr: val}}
        col_stats_dict = {}
        for cs in col_stats:
            col_stats_dict[cs.name] = {
                'distinct_count': cs.distinct_count,
                'null_count': cs.null_count,
                'min_value': cs.min_value,
                'max_value': cs.max_value,
            }
        self.planner.update_statistics(table_name, row_count, col_stats_dict)

    # ---- Internal Methods ----

    def _execute_in_tx(self, sql: str, stmt_type: StatementType,
                       tx: TransactionContext, params: Optional[dict] = None) -> ExecutionResult:
        """Execute a statement within a transaction context."""
        tx.statement_count += 1

        if stmt_type == StatementType.SELECT:
            return self._execute_select(sql, tx, params)
        elif stmt_type == StatementType.INSERT:
            return self.dml.execute_insert(sql, tx)
        elif stmt_type == StatementType.UPDATE:
            return self.dml.execute_update(sql, tx)
        elif stmt_type == StatementType.DELETE:
            return self.dml.execute_delete(sql, tx)
        elif stmt_type == StatementType.CREATE_TABLE:
            return self.ddl.execute_create_table(sql, tx)
        elif stmt_type == StatementType.DROP_TABLE:
            return self.ddl.execute_drop_table(sql, tx)
        elif stmt_type == StatementType.CREATE_INDEX:
            return self.ddl.execute_create_index(sql, tx)
        elif stmt_type == StatementType.DROP_INDEX:
            return self.ddl.execute_drop_index(sql, tx)
        else:
            return ExecutionResult(message=f"Unsupported statement type: {stmt_type}")

    def _execute_select(self, sql: str, tx: TransactionContext,
                        params: Optional[dict] = None) -> ExecutionResult:
        """Execute a SELECT using the full pipeline: plan -> lock -> execute."""
        # Step 1: Plan (with caching)
        try:
            annotated = self.planner.plan(sql, is_write=False, params=params)
            from_cache = True  # planner handles caching internally
        except Exception:
            # Fallback: direct execution without planning
            annotated = None
            from_cache = False

        # Step 2: Acquire locks
        locks = 0
        if annotated:
            lock_result, lock_count = self.lock_acquirer.acquire_for_plan(
                tx.tx_id, annotated, timeout=self.lock_timeout)
            if lock_result not in (LockResult.GRANTED, LockResult.UPGRADED):
                return ExecutionResult(
                    message=f"Lock acquisition failed: {lock_result.name}",
                    plan_used=annotated
                )
            locks = lock_count

        # Step 3: Execute
        try:
            rows = self.query_engine.execute_raw(sql)
            tx.rows_read += len(rows)
        except Exception as e:
            return ExecutionResult(message=f"Execution error: {e}", plan_used=annotated)

        # Step 4: Update statistics
        if annotated and annotated.tables_accessed:
            for table in annotated.tables_accessed:
                self.planner.stats_tracker.record_scan(table, len(rows))

        self.stats.rows_returned += len(rows)

        return ExecutionResult(
            rows=rows,
            columns=list(rows[0].keys()) if rows else [],
            plan_used=annotated,
            locks_acquired=locks,
            from_cache=from_cache,
            statement_type=StatementType.SELECT
        )

    def _handle_begin(self) -> ExecutionResult:
        """Handle BEGIN statement."""
        tx_id = self.begin()
        return ExecutionResult(
            statement_type=StatementType.BEGIN,
            message=f"Transaction {tx_id} started"
        )

    def _handle_commit(self, tx_id: Optional[int] = None) -> ExecutionResult:
        """Handle COMMIT statement."""
        if tx_id is None:
            # Find the most recent active transaction
            active = self.active_transactions()
            if not active:
                return ExecutionResult(message="No active transaction")
            tx_id = active[-1]

        tx = self._transactions.get(tx_id)
        if tx is None:
            return ExecutionResult(message=f"Transaction {tx_id} not found")
        if not tx.is_active:
            return ExecutionResult(message=f"Transaction {tx_id} already {tx.state.name}")

        tx.state = TxState.COMMITTED
        released = self.lock_acquirer.release_all(tx.tx_id)
        self.stats.commits += 1

        return ExecutionResult(
            locks_released=released,
            statement_type=StatementType.COMMIT,
            message=f"Transaction {tx_id} committed ({released} locks released)"
        )

    def _handle_rollback(self, tx_id: Optional[int] = None) -> ExecutionResult:
        """Handle ROLLBACK statement."""
        if tx_id is None:
            active = self.active_transactions()
            if not active:
                return ExecutionResult(message="No active transaction")
            tx_id = active[-1]

        tx = self._transactions.get(tx_id)
        if tx is None:
            return ExecutionResult(message=f"Transaction {tx_id} not found")
        if not tx.is_active:
            return ExecutionResult(message=f"Transaction {tx_id} already {tx.state.name}")

        # Apply undo log
        tx.apply_undo(self.db)
        tx.state = TxState.ABORTED
        released = self.lock_acquirer.release_all(tx.tx_id)
        self.stats.rollbacks += 1

        return ExecutionResult(
            locks_released=released,
            statement_type=StatementType.ROLLBACK,
            message=f"Transaction {tx_id} rolled back ({len(tx.undo_log)} operations undone, {released} locks released)"
        )

    def _handle_explain(self, sql: str, show_locks: bool = True) -> ExecutionResult:
        """Handle EXPLAIN statement."""
        # Strip EXPLAIN prefix
        inner_sql = re.sub(r'^\s*EXPLAIN\s+', '', sql, count=1, flags=re.IGNORECASE)
        try:
            explanation = self.planner.explain(inner_sql, show_locks=show_locks)
        except Exception as e:
            explanation = f"Cannot explain: {e}"
        return ExecutionResult(
            statement_type=StatementType.EXPLAIN,
            message=explanation
        )

    def _handle_explain_analyze(self, sql: str, tx_id: Optional[int] = None) -> ExecutionResult:
        """Handle EXPLAIN ANALYZE -- plan + execute + compare."""
        inner_sql = re.sub(r'^\s*EXPLAIN\s+ANALYZE\s+', '', sql, count=1, flags=re.IGNORECASE)

        # Plan
        try:
            annotated = self.planner.plan(inner_sql, is_write=False)
        except Exception as e:
            return ExecutionResult(
                statement_type=StatementType.EXPLAIN_ANALYZE,
                message=f"Planning failed: {e}"
            )

        # Execute (auto-commit)
        start = time.time()
        try:
            rows = self.query_engine.execute_raw(inner_sql)
            actual_rows = len(rows)
        except Exception as e:
            return ExecutionResult(
                statement_type=StatementType.EXPLAIN_ANALYZE,
                message=f"Execution failed: {e}"
            )
        exec_time = (time.time() - start) * 1000

        # Build analysis
        estimated = annotated.estimated_rows
        accuracy = 0
        if estimated > 0:
            accuracy = min(actual_rows, estimated) / max(actual_rows, estimated) * 100 if max(actual_rows, estimated) > 0 else 100

        lines = [
            annotated.explain(show_locks=True),
            f"\n--- Actual Execution ---",
            f"Actual rows: {actual_rows} (estimated: {estimated})",
            f"Estimation accuracy: {accuracy:.1f}%",
            f"Execution time: {exec_time:.2f}ms",
        ]
        if abs(actual_rows - estimated) > estimated * 0.5 and estimated > 0:
            lines.append(f"WARNING: Row estimate off by {abs(actual_rows - estimated)} rows -- consider ANALYZE")

        return ExecutionResult(
            rows=rows,
            statement_type=StatementType.EXPLAIN_ANALYZE,
            message='\n'.join(lines),
            execution_time_ms=exec_time
        )

    def _create_auto_tx(self) -> TransactionContext:
        """Create an ephemeral auto-commit transaction."""
        with self._tx_lock:
            self._auto_tx_counter += 1
            tx = TransactionContext(self, self.default_isolation, tx_id=self._auto_tx_counter)
        return tx

    def _format_val(self, v) -> str:
        """Format a Python value for SQL insertion."""
        if v is None:
            return 'NULL'
        elif isinstance(v, str):
            return f"'{v}'"
        else:
            return str(v)


@dataclass
class EngineStats:
    """Aggregate statistics for the integrated engine."""
    statements_executed: int = 0
    commits: int = 0
    rollbacks: int = 0
    rows_returned: int = 0
    deadlocks_detected: int = 0

    def summary(self) -> dict:
        return {
            'statements': self.statements_executed,
            'commits': self.commits,
            'rollbacks': self.rollbacks,
            'rows_returned': self.rows_returned,
            'deadlocks': self.deadlocks_detected,
        }


class IntegratedEngineAnalyzer:
    """Diagnostic and analysis tools for the integrated engine."""

    def __init__(self, engine: IntegratedQueryEngine):
        self.engine = engine

    def health_report(self) -> dict:
        """Overall engine health report."""
        lock_stats = self.engine.lock_manager.stats.summary()
        cache_stats = self.engine.planner.cache.stats()
        engine_stats = self.engine.stats.summary()

        return {
            'engine': engine_stats,
            'locks': lock_stats,
            'plan_cache': cache_stats,
            'active_transactions': len(self.engine.active_transactions()),
            'tables': len([t for t in self._table_names()]),
        }

    def lock_report(self) -> dict:
        """Detailed lock status report."""
        analyzer = LockManagerAnalyzer(self.engine.lock_manager)
        return analyzer.contention_report()

    def cache_report(self) -> dict:
        """Plan cache effectiveness report."""
        return self.engine.planner.cache.stats()

    def transaction_report(self, tx_id: int) -> dict:
        """Report on a specific transaction."""
        tx = self.engine.get_transaction(tx_id)
        if tx is None:
            return {'error': f'Transaction {tx_id} not found'}
        return {
            'tx_id': tx.tx_id,
            'state': tx.state.name,
            'isolation': tx.isolation.name,
            'statements': tx.statement_count,
            'rows_read': tx.rows_read,
            'rows_written': tx.rows_written,
            'undo_entries': len(tx.undo_log),
            'tables_modified': list(tx.tables_modified),
            'duration_s': time.time() - tx.start_time,
        }

    def workload_analysis(self, queries: list[str]) -> dict:
        """Analyze a workload of queries."""
        results = []
        total_cost = 0
        for sql in queries:
            try:
                plan = self.engine.planner.plan(sql)
                results.append({
                    'sql': sql[:80],
                    'cost': plan.total_cost,
                    'estimated_rows': plan.estimated_rows,
                    'tables': plan.tables_accessed,
                    'indexes': plan.indexes_used,
                    'lock_strategy': plan.lock_plan.strategy.name,
                })
                total_cost += plan.total_cost
            except Exception as e:
                results.append({'sql': sql[:80], 'error': str(e)})

        return {
            'query_count': len(queries),
            'total_estimated_cost': total_cost,
            'queries': results,
        }

    def recommend_indexes(self, queries: list[str], max_recs: int = 5) -> list:
        """Recommend indexes for a workload."""
        advisor = IndexAdvisor(self.engine.planner)
        for sql in queries:
            advisor.record_query(sql)
        return advisor.recommend(max_recs)

    def _table_names(self) -> list:
        """Get all table names."""
        if hasattr(self.engine.db, 'tables'):
            return list(self.engine.db.tables.keys())
        return []


class PipelineExecutor:
    """Executes multi-statement pipelines (scripts) with transaction support."""

    def __init__(self, engine: IntegratedQueryEngine):
        self.engine = engine

    def execute_script(self, script: str, tx_id: Optional[int] = None) -> list[ExecutionResult]:
        """Execute a multi-statement SQL script.
        Statements are separated by semicolons."""
        results = []
        statements = self._split_statements(script)

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            result = self.engine.execute(stmt, tx_id)
            results.append(result)

            # Track transaction changes from BEGIN
            if result.statement_type == StatementType.BEGIN:
                # Extract tx_id from message
                match = re.search(r'Transaction (\d+)', result.message)
                if match:
                    tx_id = int(match.group(1))
            elif result.statement_type in (StatementType.COMMIT, StatementType.ROLLBACK):
                tx_id = None

        return results

    def execute_transaction(self, statements: list[str],
                            isolation: Optional[IsolationLevel] = None) -> list[ExecutionResult]:
        """Execute statements in an explicit transaction."""
        tx_id = self.engine.begin(isolation)
        results = []

        try:
            for sql in statements:
                result = self.engine.execute(sql, tx_id)
                results.append(result)

                if "error" in result.message.lower() or "denied" in result.message.lower():
                    # Abort on error
                    self.engine.rollback(tx_id)
                    results.append(ExecutionResult(
                        statement_type=StatementType.ROLLBACK,
                        message=f"Transaction {tx_id} auto-rolled back due to error"
                    ))
                    return results

            commit_result = self.engine.commit(tx_id)
            results.append(commit_result)
        except Exception:
            self.engine.rollback(tx_id)
            results.append(ExecutionResult(
                statement_type=StatementType.ROLLBACK,
                message=f"Transaction {tx_id} rolled back due to exception"
            ))
            raise

        return results

    def _split_statements(self, script: str) -> list:
        """Split script into statements by semicolons, respecting strings."""
        stmts = []
        current = []
        in_quote = False
        for ch in script:
            if ch == "'" and not in_quote:
                in_quote = True
                current.append(ch)
            elif ch == "'" and in_quote:
                in_quote = False
                current.append(ch)
            elif ch == ';' and not in_quote:
                stmts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            s = ''.join(current).strip()
            if s:
                stmts.append(s)
        return stmts


class ConcurrentExecutionManager:
    """Manages concurrent query execution with deadlock handling."""

    def __init__(self, engine: IntegratedQueryEngine, max_retries: int = 3):
        self.engine = engine
        self.max_retries = max_retries

    def execute_concurrent(self, tx_id: int, sql: str) -> ExecutionResult:
        """Execute with automatic deadlock retry."""
        for attempt in range(self.max_retries):
            result = self.engine.execute(sql, tx_id)
            if 'DEADLOCK' not in result.message:
                return result
            # Deadlock detected -- rollback and retry
            self.engine.rollback(tx_id)
            self.engine.stats.deadlocks_detected += 1
            tx_id = self.engine.begin()
            time.sleep(0.01 * (attempt + 1))  # backoff

        return ExecutionResult(message=f"Failed after {self.max_retries} deadlock retries")

    def execute_batch(self, queries: list[tuple[int, str]]) -> list[ExecutionResult]:
        """Execute a batch of (tx_id, sql) pairs."""
        return [self.execute_concurrent(tid, sql) for tid, sql in queries]
