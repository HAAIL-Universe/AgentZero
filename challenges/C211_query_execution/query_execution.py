"""
C211: Query Execution Engine

A Volcano/iterator-model query execution engine that actually runs SQL queries
against in-memory tables. Composes C210 (Query Optimizer) for plan generation.

Components:
- Table: In-memory row storage with column schema
- Database: Table management (CREATE, INSERT, DROP)
- Volcano operators: SeqScanExec, IndexScanExec, FilterExec, ProjectExec,
  HashJoinExec, MergeJoinExec, NestedLoopJoinExec, SortExec,
  HashAggregateExec, LimitExec, DistinctExec
- Expression evaluator: evaluates SQL expressions against row data
- QueryEngine: end-to-end SQL parsing, optimization, and execution

No external dependencies. Pure Python.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator
import math
import re
import sys
import os

# Import C210 optimizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C210_query_optimizer'))
from query_optimizer import (
    Lexer, Parser, parse_sql, SelectStmt,
    ColumnRef, Literal, BinExpr, UnaryExpr, FuncCall, StarExpr,
    InExpr, BetweenExpr, ExistsExpr, SubqueryExpr, CaseExpr,
    AliasedExpr, TableRef, JoinClause, OrderByItem,
    LogicalOp, LogicalScan, LogicalFilter, LogicalProject, LogicalJoin,
    LogicalAggregate, LogicalSort, LogicalLimit, LogicalDistinct,
    PhysicalOp, SeqScan, IndexScan, HashJoin, MergeJoin, NestedLoopJoin,
    PhysicalFilter, PhysicalProject, PhysicalSort, HashAggregate,
    SortAggregate, PhysicalLimit, PhysicalDistinct,
    Catalog, TableDef, ColumnStats, IndexDef, CostParams,
    QueryOptimizer, LogicalPlanner, PlanTransformer, PhysicalPlanner,
    TokenType, Token,
)


# ============================================================
# Row representation
# ============================================================

# A Row is a dict mapping qualified column names to values
# e.g. {"users.id": 1, "users.name": "Alice"}
Row = dict[str, Any]


# ============================================================
# In-Memory Table
# ============================================================

class Table:
    """In-memory table with column schema and row storage."""

    def __init__(self, name: str, columns: list[str], primary_key: str = None):
        self.name = name
        self.columns = columns
        self.rows: list[dict[str, Any]] = []  # list of {col_name: value}
        self.primary_key = primary_key
        self.indexes: dict[str, dict] = {}  # index_name -> {value -> [row_indices]}

    def insert(self, values: dict[str, Any] | list):
        """Insert a row. Accepts dict or list (positional)."""
        if isinstance(values, (list, tuple)):
            if len(values) != len(self.columns):
                raise ValueError(f"Expected {len(self.columns)} values, got {len(values)}")
            row = dict(zip(self.columns, values))
        else:
            row = dict(values)
        row_idx = len(self.rows)
        self.rows.append(row)
        # Update indexes
        for idx_name, idx_data in self.indexes.items():
            idx_cols = idx_data['columns']
            key = tuple(row.get(c) for c in idx_cols)
            if len(idx_cols) == 1:
                key = key[0]
            idx_data['entries'].setdefault(key, []).append(row_idx)

    def insert_many(self, rows: list):
        """Insert multiple rows."""
        for row in rows:
            self.insert(row)

    def create_index(self, name: str, columns: list[str], unique: bool = False):
        """Create an index on the specified columns."""
        entries: dict = {}
        for i, row in enumerate(self.rows):
            key = tuple(row.get(c) for c in columns)
            if len(columns) == 1:
                key = key[0]
            entries.setdefault(key, []).append(i)
        self.indexes[name] = {
            'columns': columns,
            'unique': unique,
            'entries': entries,
        }

    def scan(self) -> Iterator[dict[str, Any]]:
        """Full table scan."""
        for row in self.rows:
            yield dict(row)

    def index_lookup(self, index_name: str, value) -> Iterator[dict[str, Any]]:
        """Lookup rows by index value."""
        idx = self.indexes.get(index_name)
        if not idx:
            return
        indices = idx['entries'].get(value, [])
        for i in indices:
            yield dict(self.rows[i])

    def index_range(self, index_name: str, low=None, high=None) -> Iterator[dict[str, Any]]:
        """Range scan on an index. Returns rows where low <= key <= high."""
        idx = self.indexes.get(index_name)
        if not idx:
            return
        for key, indices in sorted(idx['entries'].items()):
            if low is not None and key < low:
                continue
            if high is not None and key > high:
                continue
            for i in indices:
                yield dict(self.rows[i])

    def __len__(self):
        return len(self.rows)


# ============================================================
# Database
# ============================================================

class Database:
    """In-memory database managing tables."""

    def __init__(self):
        self.tables: dict[str, Table] = {}

    def create_table(self, name: str, columns: list[str], primary_key: str = None) -> Table:
        table = Table(name, columns, primary_key)
        self.tables[name] = table
        return table

    def drop_table(self, name: str):
        self.tables.pop(name, None)

    def get_table(self, name: str) -> Optional[Table]:
        return self.tables.get(name)

    def insert(self, table_name: str, values):
        table = self.tables.get(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found")
        table.insert(values)

    def build_catalog(self) -> Catalog:
        """Build a C210 Catalog from the actual table data for the optimizer."""
        catalog = Catalog()
        for name, table in self.tables.items():
            col_stats = []
            for col_name in table.columns:
                values = [r.get(col_name) for r in table.rows if r.get(col_name) is not None]
                distinct = len(set(values)) if values else 1
                null_count = sum(1 for r in table.rows if r.get(col_name) is None)
                min_v = min(values) if values else None
                max_v = max(values) if values else None
                col_stats.append(ColumnStats(
                    name=col_name,
                    distinct_count=max(1, distinct),
                    null_count=null_count,
                    min_value=min_v,
                    max_value=max_v,
                ))
            indexes = []
            for idx_name, idx_data in table.indexes.items():
                indexes.append(IndexDef(
                    name=idx_name,
                    table=name,
                    columns=idx_data['columns'],
                    unique=idx_data.get('unique', False),
                ))
            tdef = TableDef(
                name=name,
                columns=col_stats,
                row_count=max(1, len(table.rows)),
                indexes=indexes,
            )
            catalog.add_table(tdef)
        return catalog


# ============================================================
# Expression Evaluator
# ============================================================

def _like_to_regex(pattern: str) -> str:
    """Convert SQL LIKE pattern to regex."""
    result = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == '%':
            result.append('.*')
        elif ch == '_':
            result.append('.')
        elif ch in r'\.^$+?{}[]|()':
            result.append('\\' + ch)
        else:
            result.append(ch)
        i += 1
    return '^' + ''.join(result) + '$'


def eval_expr(expr, row: Row, context: 'ExecutionContext' = None) -> Any:
    """Evaluate a SQL expression against a row.

    Row keys can be qualified (table.col) or unqualified (col).
    We try exact match first, then unqualified lookup.
    """
    if expr is None:
        return None

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, ColumnRef):
        # Try qualified name first
        if expr.table:
            key = f"{expr.table}.{expr.column}"
            if key in row:
                return row[key]
        # Try unqualified
        if expr.column in row:
            return row[expr.column]
        # Try finding by column suffix
        for k, v in row.items():
            if k.endswith('.' + expr.column):
                return v
        return None

    if isinstance(expr, BinExpr):
        if expr.op == 'AND':
            left = eval_expr(expr.left, row, context)
            if not left:
                return False
            return bool(eval_expr(expr.right, row, context))
        if expr.op == 'OR':
            left = eval_expr(expr.left, row, context)
            if left:
                return True
            return bool(eval_expr(expr.right, row, context))

        left = eval_expr(expr.left, row, context)
        right = eval_expr(expr.right, row, context)

        if expr.op == '=':
            return left == right
        if expr.op in ('!=', '<>'):
            return left != right
        if expr.op == '<':
            return _safe_compare(left, right, '<')
        if expr.op == '>':
            return _safe_compare(left, right, '>')
        if expr.op == '<=':
            return _safe_compare(left, right, '<=')
        if expr.op == '>=':
            return _safe_compare(left, right, '>=')
        if expr.op == '+':
            if left is None or right is None:
                return None
            return left + right
        if expr.op == '-':
            if left is None or right is None:
                return None
            return left - right
        if expr.op == '*':
            if left is None or right is None:
                return None
            return left * right
        if expr.op == '/':
            if left is None or right is None or right == 0:
                return None
            return left / right
        if expr.op == '%':
            if left is None or right is None or right == 0:
                return None
            return left % right
        if expr.op == 'LIKE':
            if left is None or right is None:
                return False
            return bool(re.match(_like_to_regex(str(right)), str(left), re.IGNORECASE))
        if expr.op == '||':
            # String concatenation
            return str(left or '') + str(right or '')

    if isinstance(expr, UnaryExpr):
        if expr.op == 'NOT':
            return not eval_expr(expr.operand, row, context)
        if expr.op == '-':
            val = eval_expr(expr.operand, row, context)
            return -val if val is not None else None
        if expr.op == 'IS NULL':
            return eval_expr(expr.operand, row, context) is None
        if expr.op == 'IS NOT NULL':
            return eval_expr(expr.operand, row, context) is not None

    if isinstance(expr, FuncCall):
        # Aggregate functions handled at aggregate operator level
        # Here we handle scalar functions
        args = [eval_expr(a, row, context) for a in expr.args]
        name = expr.name.upper()
        if name == 'COALESCE':
            for a in args:
                if a is not None:
                    return a
            return None
        if name == 'NULLIF':
            return None if len(args) >= 2 and args[0] == args[1] else args[0]
        if name == 'ABS':
            return abs(args[0]) if args and args[0] is not None else None
        if name == 'UPPER':
            return str(args[0]).upper() if args and args[0] is not None else None
        if name == 'LOWER':
            return str(args[0]).lower() if args and args[0] is not None else None
        if name == 'LENGTH':
            return len(str(args[0])) if args and args[0] is not None else None
        if name == 'SUBSTR' or name == 'SUBSTRING':
            if len(args) >= 2 and args[0] is not None:
                s = str(args[0])
                start = int(args[1]) - 1  # SQL is 1-indexed
                length = int(args[2]) if len(args) >= 3 else len(s) - start
                return s[start:start + length]
            return None
        if name == 'TRIM':
            return str(args[0]).strip() if args and args[0] is not None else None
        if name == 'ROUND':
            if args and args[0] is not None:
                decimals = int(args[1]) if len(args) >= 2 else 0
                return round(args[0], decimals)
            return None
        if name == 'CAST':
            return args[0] if args else None
        # Aggregate functions used as scalar (e.g. in HAVING after grouping)
        if name in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX'):
            # If we encounter these outside aggregation context, evaluate from row
            agg_key = str(expr)
            if agg_key in row:
                return row[agg_key]
            return args[0] if args else None
        return None

    if isinstance(expr, InExpr):
        val = eval_expr(expr.expr, row, context)
        values = [eval_expr(v, row, context) for v in expr.values]
        result = val in values
        return not result if expr.negated else result

    if isinstance(expr, BetweenExpr):
        val = eval_expr(expr.expr, row, context)
        low = eval_expr(expr.low, row, context)
        high = eval_expr(expr.high, row, context)
        if val is None or low is None or high is None:
            return False
        result = low <= val <= high
        return not result if expr.negated else result

    if isinstance(expr, CaseExpr):
        if expr.operand:
            operand = eval_expr(expr.operand, row, context)
            for cond, result in expr.whens:
                if eval_expr(cond, row, context) == operand:
                    return eval_expr(result, row, context)
        else:
            for cond, result in expr.whens:
                if eval_expr(cond, row, context):
                    return eval_expr(result, row, context)
        if expr.else_result is not None:
            return eval_expr(expr.else_result, row, context)
        return None

    if isinstance(expr, StarExpr):
        return row  # Return entire row

    if isinstance(expr, AliasedExpr):
        return eval_expr(expr.expr, row, context)

    if isinstance(expr, SubqueryExpr):
        if context:
            return context.execute_subquery(expr.query, row)
        return None

    if isinstance(expr, ExistsExpr):
        if context:
            result = context.execute_subquery_exists(expr.subquery, row)
            return not result if expr.negated else result
        return False

    # If it's a string, treat as column name
    if isinstance(expr, str):
        if expr in row:
            return row[expr]
        for k, v in row.items():
            if k.endswith('.' + expr):
                return v
        return None

    return None


def _safe_compare(left, right, op: str) -> bool:
    """Compare with NULL handling."""
    if left is None or right is None:
        return False
    try:
        if op == '<':
            return left < right
        if op == '>':
            return left > right
        if op == '<=':
            return left <= right
        if op == '>=':
            return left >= right
    except TypeError:
        return False
    return False


# ============================================================
# Execution Context (for subqueries)
# ============================================================

class ExecutionContext:
    """Holds state for query execution, including subquery support."""

    def __init__(self, engine: 'QueryEngine'):
        self.engine = engine

    def execute_subquery(self, stmt: SelectStmt, outer_row: Row = None) -> Any:
        """Execute a subquery and return the scalar result."""
        rows = self.engine.execute_select(stmt, outer_row)
        if not rows:
            return None
        first = rows[0]
        # Return first column of first row
        for v in first.values():
            return v
        return None

    def execute_subquery_exists(self, stmt: SelectStmt, outer_row: Row = None) -> bool:
        """Execute a subquery and return whether any rows exist."""
        rows = self.engine.execute_select(stmt, outer_row)
        return len(rows) > 0


# ============================================================
# Volcano Execution Operators
# ============================================================

class ExecOperator:
    """Base class for Volcano-model execution operators."""

    def open(self):
        """Initialize the operator."""
        pass

    def next(self) -> Optional[Row]:
        """Return next row or None if exhausted."""
        return None

    def close(self):
        """Clean up resources."""
        pass

    def __iter__(self):
        self.open()
        while True:
            row = self.next()
            if row is None:
                break
            yield row
        self.close()

    def collect(self) -> list[Row]:
        """Materialize all rows."""
        return list(self)


class SeqScanExec(ExecOperator):
    """Full table scan."""

    def __init__(self, table: Table, alias: str = None, filter_expr=None, context=None):
        self.table = table
        self.alias = alias or table.name
        self.filter_expr = filter_expr
        self.context = context
        self._iter = None

    def open(self):
        self._iter = self.table.scan()

    def next(self) -> Optional[Row]:
        while True:
            try:
                raw = next(self._iter)
            except StopIteration:
                return None
            # Qualify column names with alias
            row = {}
            for col, val in raw.items():
                row[f"{self.alias}.{col}"] = val
                row[col] = val  # Also keep unqualified
            if self.filter_expr:
                if not eval_expr(self.filter_expr, row, self.context):
                    continue
            return row

    def close(self):
        self._iter = None


class IndexScanExec(ExecOperator):
    """Index-based lookup scan."""

    def __init__(self, table: Table, index_name: str, alias: str = None,
                 lookup_values=None, scan_type='eq', range_low=None, range_high=None,
                 filter_expr=None, context=None):
        self.table = table
        self.index_name = index_name
        self.alias = alias or table.name
        self.lookup_values = lookup_values
        self.scan_type = scan_type
        self.range_low = range_low
        self.range_high = range_high
        self.filter_expr = filter_expr
        self.context = context
        self._iter = None

    def open(self):
        if self.scan_type == 'eq' and self.lookup_values is not None:
            val = self.lookup_values[0] if isinstance(self.lookup_values, list) else self.lookup_values
            self._iter = self.table.index_lookup(self.index_name, val)
        else:
            self._iter = self.table.index_range(self.index_name, self.range_low, self.range_high)

    def next(self) -> Optional[Row]:
        while True:
            try:
                raw = next(self._iter)
            except StopIteration:
                return None
            row = {}
            for col, val in raw.items():
                row[f"{self.alias}.{col}"] = val
                row[col] = val
            if self.filter_expr:
                if not eval_expr(self.filter_expr, row, self.context):
                    continue
            return row


class FilterExec(ExecOperator):
    """Filter rows by condition."""

    def __init__(self, input_op: ExecOperator, condition, context=None):
        self.input = input_op
        self.condition = condition
        self.context = context

    def open(self):
        self.input.open()

    def next(self) -> Optional[Row]:
        while True:
            row = self.input.next()
            if row is None:
                return None
            if eval_expr(self.condition, row, self.context):
                return row

    def close(self):
        self.input.close()


class ProjectExec(ExecOperator):
    """Project (select) specific expressions."""

    def __init__(self, input_op: ExecOperator, expressions: list, context=None):
        """expressions is a list of (expr, alias) tuples."""
        self.input = input_op
        self.expressions = expressions
        self.context = context

    def open(self):
        self.input.open()

    def next(self) -> Optional[Row]:
        row = self.input.next()
        if row is None:
            return None
        result = {}
        for expr, alias in self.expressions:
            if isinstance(expr, StarExpr):
                # Expand star - include all columns
                if expr.table:
                    for k, v in row.items():
                        if k.startswith(expr.table + '.'):
                            result[k] = v
                            col_name = k.split('.', 1)[1]
                            result[col_name] = v
                else:
                    result.update(row)
                continue
            val = eval_expr(expr, row, self.context)
            if alias:
                result[alias] = val
            else:
                # Derive name from expression
                name = _expr_name(expr)
                result[name] = val
            # Also keep qualified name if it's a column ref
            if isinstance(expr, ColumnRef) and expr.table:
                result[f"{expr.table}.{expr.column}"] = val
                result[expr.column] = val
            elif isinstance(expr, ColumnRef):
                result[expr.column] = val
                # Also check if row had a qualified version
                for k in row:
                    if k.endswith('.' + expr.column):
                        result[k] = val
        return result

    def close(self):
        self.input.close()


class HashJoinExec(ExecOperator):
    """Hash join: builds hash table on right (build side), probes with left."""

    def __init__(self, left: ExecOperator, right: ExecOperator,
                 condition, join_type='INNER', context=None):
        self.left = left
        self.right = right
        self.condition = condition
        self.join_type = join_type
        self.context = context
        self._hash_table: dict = {}
        self._left_iter = None
        self._current_matches = None
        self._current_left = None
        self._right_matched: set = None
        self._left_exhausted = False
        self._build_expr = None  # expression to hash on build (right) side
        self._probe_expr = None  # expression to hash on probe (left) side

    def open(self):
        # Build phase: materialize right side into hash table
        self.right.open()
        self._hash_table = {}
        self._right_matched = set()

        right_rows = []
        while True:
            row = self.right.next()
            if row is None:
                break
            right_rows.append(row)
        self.right.close()

        # Determine which side of condition matches which input
        self._detect_key_mapping(right_rows)

        for idx, row in enumerate(right_rows):
            key = eval_expr(self._build_expr, row, self.context)
            self._hash_table.setdefault(key, []).append((idx, row))

        # Probe phase starts
        self.left.open()
        self._current_matches = iter([])
        self._current_left = None
        self._left_exhausted = False

    def _detect_key_mapping(self, right_rows: list):
        """Detect which side of the = condition refers to build vs probe side."""
        if not isinstance(self.condition, BinExpr) or self.condition.op != '=':
            self._build_expr = self.condition
            self._probe_expr = self.condition
            return
        lhs = self.condition.left
        rhs = self.condition.right
        # Try evaluating lhs on a right row -- if it resolves, lhs is the build key
        if right_rows:
            sample = right_rows[0]
            lval = eval_expr(lhs, sample, self.context)
            rval = eval_expr(rhs, sample, self.context)
            # The expr that produces a non-None value from the right side is the build key
            if lval is not None and rval is None:
                self._build_expr = lhs
                self._probe_expr = rhs
                return
            if rval is not None and lval is None:
                self._build_expr = rhs
                self._probe_expr = lhs
                return
        # Fallback: assume condition left = probe (left input), right = build (right input)
        self._probe_expr = lhs
        self._build_expr = rhs

    def next(self) -> Optional[Row]:
        while True:
            # Try to get next match from current probe
            try:
                idx, right_row = next(self._current_matches)
                combined = {**self._current_left, **right_row}
                self._right_matched.add(idx)
                return combined
            except StopIteration:
                pass

            # LEFT JOIN: emit unmatched left row
            if self._current_left is not None and self.join_type == 'LEFT':
                if not self._had_match:
                    null_row = self._make_null_right()
                    combined = {**self._current_left, **null_row}
                    self._current_left = None
                    return combined

            if self._left_exhausted:
                # RIGHT JOIN: emit unmatched right rows
                if self.join_type == 'RIGHT':
                    return self._next_unmatched_right()
                return None

            # Get next left row
            left_row = self.left.next()
            if left_row is None:
                self._left_exhausted = True
                self._current_left = None
                if self.join_type == 'RIGHT':
                    self._unmatched_right_iter = self._iter_unmatched_right()
                    return self._next_unmatched_right()
                return None

            self._current_left = left_row
            key = eval_expr(self._probe_expr, left_row, self.context)
            matches = self._hash_table.get(key, [])
            self._had_match = len(matches) > 0
            self._current_matches = iter(matches)

    def _make_null_right(self) -> Row:
        """Create a null row for unmatched left join."""
        # Get column names from any right row
        for rows in self._hash_table.values():
            if rows:
                return {k: None for k in rows[0][1]}
        return {}

    def _iter_unmatched_right(self):
        for key, rows in self._hash_table.items():
            for idx, row in rows:
                if idx not in self._right_matched:
                    yield row

    def _next_unmatched_right(self) -> Optional[Row]:
        try:
            right_row = next(self._unmatched_right_iter)
            null_left = {}
            if self._current_left:
                null_left = {k: None for k in self._current_left}
            return {**null_left, **right_row}
        except (StopIteration, AttributeError):
            return None

    def close(self):
        self.left.close()
        self._hash_table.clear()


class NestedLoopJoinExec(ExecOperator):
    """Nested loop join: for each left row, scan all right rows."""

    def __init__(self, left: ExecOperator, right: ExecOperator,
                 condition, join_type='INNER', context=None):
        self.left = left
        self.right = right
        self.condition = condition
        self.join_type = join_type
        self.context = context
        self._right_rows: list = []
        self._left_row = None
        self._right_idx = 0
        self._had_match = False

    def open(self):
        # Materialize right side
        self.right.open()
        self._right_rows = []
        while True:
            row = self.right.next()
            if row is None:
                break
            self._right_rows.append(row)
        self.right.close()
        self.left.open()
        self._left_row = None
        self._right_idx = 0
        self._had_match = False

    def next(self) -> Optional[Row]:
        while True:
            if self._left_row is None:
                self._left_row = self.left.next()
                if self._left_row is None:
                    return None
                self._right_idx = 0
                self._had_match = False

            while self._right_idx < len(self._right_rows):
                right_row = self._right_rows[self._right_idx]
                self._right_idx += 1
                combined = {**self._left_row, **right_row}
                if self.condition is None or eval_expr(self.condition, combined, self.context):
                    self._had_match = True
                    return combined

            # Exhausted right side for current left row
            if self.join_type == 'LEFT' and not self._had_match:
                null_right = {k: None for k in self._right_rows[0]} if self._right_rows else {}
                combined = {**self._left_row, **null_right}
                self._left_row = None
                return combined

            self._left_row = None

    def close(self):
        self.left.close()
        self._right_rows.clear()


class MergeJoinExec(ExecOperator):
    """Sort-merge join: both sides must be sorted on join key."""

    def __init__(self, left: ExecOperator, right: ExecOperator,
                 condition, join_type='INNER', context=None):
        self.left = left
        self.right = right
        self.condition = condition
        self.join_type = join_type
        self.context = context
        self._left_rows: list = []
        self._right_rows: list = []
        self._result_iter = None

    def open(self):
        # Materialize and sort both sides
        self.left.open()
        self._left_rows = []
        while True:
            row = self.left.next()
            if row is None:
                break
            self._left_rows.append(row)
        self.left.close()

        self.right.open()
        self._right_rows = []
        while True:
            row = self.right.next()
            if row is None:
                break
            self._right_rows.append(row)
        self.right.close()

        left_col, right_col = self._get_join_columns()
        if left_col and right_col:
            self._left_rows.sort(key=lambda r: eval_expr(left_col, r, self.context) or '')
            self._right_rows.sort(key=lambda r: eval_expr(right_col, r, self.context) or '')

        self._result_iter = self._merge()

    def _get_join_columns(self):
        if isinstance(self.condition, BinExpr) and self.condition.op == '=':
            return self.condition.left, self.condition.right
        return None, None

    def _merge(self):
        left_col, right_col = self._get_join_columns()
        i, j = 0, 0
        right_matched = set()

        while i < len(self._left_rows) and j < len(self._right_rows):
            lval = eval_expr(left_col, self._left_rows[i], self.context) if left_col else None
            rval = eval_expr(right_col, self._right_rows[j], self.context) if right_col else None

            if lval == rval:
                # Find all matching right rows
                match_start = j
                while j < len(self._right_rows):
                    rv = eval_expr(right_col, self._right_rows[j], self.context) if right_col else None
                    if rv != lval:
                        break
                    j += 1
                # Emit cross product of matching groups
                for ri in range(match_start, j):
                    right_matched.add(ri)
                    yield {**self._left_rows[i], **self._right_rows[ri]}
                i += 1
                j = match_start  # Reset j for next left row with same key
                # Actually, advance j if next left is different
                if i < len(self._left_rows):
                    next_lval = eval_expr(left_col, self._left_rows[i], self.context) if left_col else None
                    if next_lval == lval:
                        continue
                    j = match_start
                    while j < len(self._right_rows):
                        rv = eval_expr(right_col, self._right_rows[j], self.context) if right_col else None
                        if rv != lval:
                            break
                        j += 1
            elif lval is not None and rval is not None and lval < rval:
                if self.join_type == 'LEFT':
                    null_right = {k: None for k in self._right_rows[0]} if self._right_rows else {}
                    yield {**self._left_rows[i], **null_right}
                i += 1
            else:
                j += 1

        # LEFT JOIN: remaining left rows
        if self.join_type == 'LEFT':
            null_right = {k: None for k in self._right_rows[0]} if self._right_rows else {}
            while i < len(self._left_rows):
                yield {**self._left_rows[i], **null_right}
                i += 1

    def next(self) -> Optional[Row]:
        try:
            return next(self._result_iter)
        except StopIteration:
            return None

    def close(self):
        self._left_rows.clear()
        self._right_rows.clear()


class SortExec(ExecOperator):
    """Sort rows by order-by expressions."""

    def __init__(self, input_op: ExecOperator, order_by: list, context=None):
        """order_by is a list of (expr, direction) tuples or OrderByItem objects."""
        self.input = input_op
        self.order_by = order_by
        self.context = context
        self._rows: list = []
        self._idx = 0

    def open(self):
        self.input.open()
        self._rows = []
        while True:
            row = self.input.next()
            if row is None:
                break
            self._rows.append(row)
        self.input.close()
        # Sort
        self._rows.sort(key=lambda r: self._sort_key(r))
        self._idx = 0

    def _sort_key(self, row: Row):
        keys = []
        for item in self.order_by:
            if isinstance(item, OrderByItem):
                val = eval_expr(item.expr, row, self.context)
                desc = item.direction == 'DESC'
            elif isinstance(item, tuple):
                expr, direction = item
                val = eval_expr(expr, row, self.context)
                desc = direction == 'DESC'
            else:
                val = eval_expr(item, row, self.context)
                desc = False
            keys.append(_SortKey(val, desc))
        return keys

    def next(self) -> Optional[Row]:
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def close(self):
        self._rows.clear()


class _SortKey:
    """Comparison wrapper for sorting with NULL handling and direction."""

    def __init__(self, value, desc=False):
        self.value = value
        self.desc = desc

    def __lt__(self, other):
        if self.value is None and other.value is None:
            return False
        if self.value is None:
            return self.desc  # NULLs last in ASC (False=not less), first in DESC (True=less)
        if other.value is None:
            return not self.desc  # opposite
        try:
            result = self.value < other.value
        except TypeError:
            result = str(self.value) < str(other.value)
        return not result if self.desc else result

    def __eq__(self, other):
        return self.value == other.value

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class HashAggregateExec(ExecOperator):
    """Hash-based aggregation."""

    def __init__(self, input_op: ExecOperator, group_by: list,
                 aggregates: list, context=None):
        """
        group_by: list of expressions
        aggregates: list of (FuncCall, alias) tuples
        """
        self.input = input_op
        self.group_by = group_by
        self.aggregates = aggregates
        self.context = context
        self._groups: dict = {}
        self._result_iter = None

    def open(self):
        self.input.open()
        self._groups = {}

        while True:
            row = self.input.next()
            if row is None:
                break
            # Compute group key
            key = tuple(eval_expr(g, row, self.context) for g in self.group_by)
            if key not in self._groups:
                self._groups[key] = {
                    'group_row': row,
                    'accumulators': [self._init_acc(agg) for agg, _ in self.aggregates],
                }
            group = self._groups[key]
            for i, (agg, _) in enumerate(self.aggregates):
                self._update_acc(group['accumulators'][i], agg, row)

        self.input.close()

        # Handle case with no groups but aggregates (e.g., COUNT(*) with no rows)
        if not self._groups and not self.group_by:
            self._groups[()] = {
                'group_row': {},
                'accumulators': [self._init_acc(agg) for agg, _ in self.aggregates],
            }

        self._result_iter = iter(self._groups.items())

    def _init_acc(self, agg):
        name = agg.name.upper() if isinstance(agg, FuncCall) else str(agg).upper()
        return {'func': name, 'value': None, 'count': 0, 'seen': set()}

    def _update_acc(self, acc, agg, row):
        if isinstance(agg, FuncCall):
            name = agg.name.upper()
            if name == 'COUNT':
                if agg.args and not isinstance(agg.args[0], StarExpr):
                    val = eval_expr(agg.args[0], row, self.context)
                    if val is not None:
                        if agg.distinct:
                            if val not in acc['seen']:
                                acc['seen'].add(val)
                                acc['count'] += 1
                        else:
                            acc['count'] += 1
                else:
                    acc['count'] += 1
                acc['value'] = acc['count']
                return

            val = eval_expr(agg.args[0], row, self.context) if agg.args else None
            if val is None:
                return

            if agg.distinct:
                if val in acc['seen']:
                    return
                acc['seen'].add(val)

            acc['count'] += 1

            if name == 'SUM':
                acc['value'] = (acc['value'] or 0) + val
            elif name == 'AVG':
                acc['value'] = ((acc['value'] or 0) * (acc['count'] - 1) + val) / acc['count']
            elif name == 'MIN':
                acc['value'] = val if acc['value'] is None else min(acc['value'], val)
            elif name == 'MAX':
                acc['value'] = val if acc['value'] is None else max(acc['value'], val)

    def next(self) -> Optional[Row]:
        try:
            key, group = next(self._result_iter)
        except StopIteration:
            return None

        row = {}
        # Add group-by values
        for i, g in enumerate(self.group_by):
            name = _expr_name(g)
            val = key[i]
            row[name] = val
            if isinstance(g, ColumnRef):
                if g.table:
                    row[f"{g.table}.{g.column}"] = val
                row[g.column] = val

        # Add aggregate values
        for i, (agg, alias) in enumerate(self.aggregates):
            val = group['accumulators'][i]['value']
            if val is None and group['accumulators'][i]['func'] == 'COUNT':
                val = 0
            agg_name = alias or str(agg)
            row[agg_name] = val

        # Also carry forward group row columns for HAVING evaluation
        for k, v in group['group_row'].items():
            if k not in row:
                row[k] = v

        return row

    def close(self):
        self._groups.clear()


class LimitExec(ExecOperator):
    """Limit + offset."""

    def __init__(self, input_op: ExecOperator, limit: int, offset: int = 0):
        self.input = input_op
        self.limit = limit
        self.offset = offset
        self._count = 0
        self._skipped = 0

    def open(self):
        self.input.open()
        self._count = 0
        self._skipped = 0

    def next(self) -> Optional[Row]:
        while self._skipped < self.offset:
            row = self.input.next()
            if row is None:
                return None
            self._skipped += 1

        if self._count >= self.limit:
            return None
        row = self.input.next()
        if row is None:
            return None
        self._count += 1
        return row

    def close(self):
        self.input.close()


class DistinctExec(ExecOperator):
    """Remove duplicate rows."""

    def __init__(self, input_op: ExecOperator):
        self.input = input_op
        self._seen: set = set()

    def open(self):
        self.input.open()
        self._seen = set()

    def next(self) -> Optional[Row]:
        while True:
            row = self.input.next()
            if row is None:
                return None
            # Create hashable key from row values
            key = tuple(sorted((k, v) for k, v in row.items() if '.' not in k))
            if key not in self._seen:
                self._seen.add(key)
                return row

    def close(self):
        self.input.close()
        self._seen.clear()


# ============================================================
# Utility functions
# ============================================================

def _expr_name(expr) -> str:
    """Derive a display name from an expression."""
    if isinstance(expr, ColumnRef):
        if expr.table:
            return f"{expr.table}.{expr.column}"
        return expr.column
    if isinstance(expr, FuncCall):
        return str(expr)
    if isinstance(expr, AliasedExpr):
        if expr.alias:
            return expr.alias
        return _expr_name(expr.expr)
    return str(expr)


def _resolve_column(row: Row, name: str) -> Any:
    """Resolve a column name from a row, trying qualified and unqualified."""
    if name in row:
        return row[name]
    for k, v in row.items():
        if k.endswith('.' + name):
            return v
    return None


# ============================================================
# Physical Plan -> Execution Plan Converter
# ============================================================

class PlanExecutor:
    """Converts C210 physical plan operators to executable Volcano operators."""

    def __init__(self, db: Database, context: ExecutionContext = None):
        self.db = db
        self.context = context

    def build(self, plan: PhysicalOp) -> ExecOperator:
        """Convert a physical plan to an executable operator tree."""
        if isinstance(plan, SeqScan):
            table = self.db.get_table(plan.table)
            if table is None:
                raise ValueError(f"Table '{plan.table}' not found")
            return SeqScanExec(table, plan.alias or plan.table, plan.filter, self.context)

        if isinstance(plan, IndexScan):
            table = self.db.get_table(plan.table)
            if table is None:
                raise ValueError(f"Table '{plan.table}' not found")
            return IndexScanExec(
                table, plan.index, plan.alias or plan.table,
                plan.lookup_values, plan.scan_type,
                plan.range_low, plan.range_high,
                plan.filter, self.context,
            )

        if isinstance(plan, HashJoin):
            left = self.build(plan.left)
            right = self.build(plan.right)
            return HashJoinExec(left, right, plan.condition, plan.join_type, self.context)

        if isinstance(plan, MergeJoin):
            left = self.build(plan.left)
            right = self.build(plan.right)
            return MergeJoinExec(left, right, plan.condition, plan.join_type, self.context)

        if isinstance(plan, NestedLoopJoin):
            left = self.build(plan.left)
            right = self.build(plan.right)
            return NestedLoopJoinExec(left, right, plan.condition, plan.join_type, self.context)

        if isinstance(plan, PhysicalFilter):
            input_op = self.build(plan.input)
            return FilterExec(input_op, plan.condition, self.context)

        if isinstance(plan, PhysicalProject):
            input_op = self.build(plan.input)
            return ProjectExec(input_op, plan.expressions, self.context)

        if isinstance(plan, PhysicalSort):
            input_op = self.build(plan.input)
            return SortExec(input_op, plan.order_by, self.context)

        if isinstance(plan, (HashAggregate, SortAggregate)):
            input_op = self.build(plan.input)
            return HashAggregateExec(input_op, plan.group_by, plan.aggregates, self.context)

        if isinstance(plan, PhysicalLimit):
            input_op = self.build(plan.input)
            return LimitExec(input_op, plan.limit, plan.offset)

        if isinstance(plan, PhysicalDistinct):
            input_op = self.build(plan.input)
            return DistinctExec(input_op)

        raise ValueError(f"Unknown physical plan node: {type(plan).__name__}")


# ============================================================
# Query Engine
# ============================================================

class QueryEngine:
    """End-to-end query execution engine.

    Parses SQL, optimizes with C210, and executes against in-memory tables.
    """

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self._context = ExecutionContext(self)

    def execute(self, sql: str) -> list[Row]:
        """Execute a SQL query and return result rows."""
        sql = sql.strip()

        # Handle DDL
        upper = sql.upper().lstrip()
        if upper.startswith('CREATE TABLE'):
            return self._exec_create_table(sql)
        if upper.startswith('INSERT'):
            return self._exec_insert(sql)
        if upper.startswith('DROP TABLE'):
            return self._exec_drop_table(sql)

        # SELECT query
        ast = parse_sql(sql)
        return self.execute_select(ast)

    def execute_select(self, ast: SelectStmt, outer_row: Row = None) -> list[Row]:
        """Execute a parsed SELECT statement."""
        catalog = self.db.build_catalog()
        optimizer = QueryOptimizer(catalog)
        logical = optimizer.to_logical(ast)
        optimized = optimizer.transform(logical)
        physical = optimizer.to_physical(optimized)

        executor = PlanExecutor(self.db, self._context)
        op = executor.build(physical)

        rows = op.collect()
        return rows

    def execute_raw(self, sql: str) -> list[dict[str, Any]]:
        """Execute and return clean results (unqualified column names only)."""
        rows = self.execute(sql)
        result = []
        for row in rows:
            clean = {}
            for k, v in row.items():
                if '.' not in k:
                    clean[k] = v
            # If all keys are qualified, extract column parts
            if not clean:
                for k, v in row.items():
                    col = k.split('.')[-1]
                    if col not in clean:
                        clean[col] = v
            result.append(clean)
        return result

    def explain(self, sql: str) -> str:
        """Get EXPLAIN output for a query."""
        catalog = self.db.build_catalog()
        optimizer = QueryOptimizer(catalog)
        return optimizer.explain(sql)

    # --------------------------------------------------------
    # DDL / DML handlers
    # --------------------------------------------------------

    def _exec_create_table(self, sql: str) -> list[Row]:
        """Parse and execute CREATE TABLE statement."""
        # Simple regex-based parser for CREATE TABLE name (col1 type, col2 type, ...)
        m = re.match(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\)\s*;?\s*$',
            sql, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            raise ValueError(f"Invalid CREATE TABLE: {sql}")
        name = m.group(1)
        cols_str = m.group(2)
        columns = []
        pk = None
        for part in self._split_ddl_columns(cols_str):
            part = part.strip()
            if not part:
                continue
            upper_part = part.upper()
            if upper_part.startswith('PRIMARY KEY'):
                # PRIMARY KEY (col)
                pk_match = re.search(r'\((\w+)\)', part)
                if pk_match:
                    pk = pk_match.group(1)
                continue
            # col_name TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT ...]
            tokens = part.split()
            col_name = tokens[0]
            columns.append(col_name)
            if 'PRIMARY' in upper_part and 'KEY' in upper_part:
                pk = col_name
        self.db.create_table(name, columns, pk)
        return [{'result': 'OK'}]

    def _split_ddl_columns(self, cols_str: str) -> list[str]:
        """Split DDL column definitions, handling nested parens."""
        parts = []
        depth = 0
        current = []
        for ch in cols_str:
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

    def _exec_insert(self, sql: str) -> list[Row]:
        """Parse and execute INSERT statement."""
        m = re.match(
            r'INSERT\s+INTO\s+(\w+)\s*(?:\(([^)]*)\))?\s*VALUES\s*(.*?)\s*;?\s*$',
            sql, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            raise ValueError(f"Invalid INSERT: {sql}")
        table_name = m.group(1)
        col_names = [c.strip() for c in m.group(2).split(',')] if m.group(2) else None
        values_str = m.group(3)

        table = self.db.get_table(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found")

        # Parse value tuples: (v1, v2), (v3, v4), ...
        count = 0
        for vm in re.finditer(r'\(([^)]*)\)', values_str):
            vals_raw = self._parse_values(vm.group(1))
            if col_names:
                row = dict(zip(col_names, vals_raw))
            else:
                row = vals_raw
            table.insert(row)
            count += 1

        return [{'inserted': count}]

    def _parse_values(self, s: str) -> list:
        """Parse comma-separated values from an INSERT VALUES clause."""
        values = []
        parts = self._split_values(s)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            upper = p.upper()
            if upper == 'NULL':
                values.append(None)
            elif upper == 'TRUE':
                values.append(True)
            elif upper == 'FALSE':
                values.append(False)
            elif (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
                values.append(p[1:-1])
            else:
                try:
                    if '.' in p:
                        values.append(float(p))
                    else:
                        values.append(int(p))
                except ValueError:
                    values.append(p)
        return values

    def _split_values(self, s: str) -> list[str]:
        """Split values string on commas, respecting quoted strings."""
        parts = []
        current = []
        in_quote = None
        for ch in s:
            if ch in ("'", '"') and in_quote is None:
                in_quote = ch
                current.append(ch)
            elif ch == in_quote:
                in_quote = None
                current.append(ch)
            elif ch == ',' and in_quote is None:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _exec_drop_table(self, sql: str) -> list[Row]:
        """Parse and execute DROP TABLE."""
        m = re.match(
            r'DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)\s*;?\s*$',
            sql, re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"Invalid DROP TABLE: {sql}")
        self.db.drop_table(m.group(1))
        return [{'result': 'OK'}]

    # --------------------------------------------------------
    # Convenience methods
    # --------------------------------------------------------

    def create_table(self, name: str, columns: list[str], primary_key: str = None) -> Table:
        """Create a table directly (API, no SQL parsing)."""
        return self.db.create_table(name, columns, primary_key)

    def insert(self, table_name: str, values):
        """Insert a row directly."""
        self.db.insert(table_name, values)

    def insert_many(self, table_name: str, rows: list):
        """Insert multiple rows."""
        table = self.db.get_table(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found")
        table.insert_many(rows)

    def create_index(self, table_name: str, index_name: str, columns: list[str], unique=False):
        """Create an index on a table."""
        table = self.db.get_table(table_name)
        if table is None:
            raise ValueError(f"Table '{table_name}' not found")
        table.create_index(index_name, columns, unique)

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL query and return clean results."""
        return self.execute_raw(sql)
