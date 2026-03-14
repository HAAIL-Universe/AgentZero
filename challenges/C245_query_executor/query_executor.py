"""
C245: Query Executor

A volcano/iterator-model query executor for relational data:
- Physical operators: SeqScan, IndexScan, Filter, Project, HashJoin,
  NestedLoopJoin, SortMergeJoin, Sort, HashAggregate, Limit, Union
- open()/next()/close() iterator protocol
- Row-based tuple processing with named columns
- Hash join with build/probe phases and spill-to-disk simulation
- Sort with external merge sort simulation
- Aggregate with grouping and multiple aggregate functions (COUNT, SUM, AVG, MIN, MAX)
- Memory budget tracking for memory-intensive operators
- Pipeline execution (no materialization until pipeline breakers)
- Statistics collection: rows processed, I/O pages, memory used
- Expression evaluator for predicates and projections
- Physical plan builder from logical plan nodes (C243 composition)

Domain: Database Internals
Composes: C243 Query Optimizer (plan nodes, expressions)
"""

from enum import Enum, auto
from typing import Any, Optional, Dict, List, Set, Tuple, Callable, Iterator, Generator
from dataclasses import dataclass, field
import math
import heapq
import itertools


# ---------------------------------------------------------------------------
# Row representation
# ---------------------------------------------------------------------------

class Row:
    """A tuple/row with named columns."""
    __slots__ = ('_data', '_schema')

    def __init__(self, data: Dict[str, Any], schema: Optional[List[str]] = None):
        self._data = data
        self._schema = schema or sorted(data.keys())

    def get(self, col: str) -> Any:
        """Get column value. Supports 'table.col' and bare 'col' lookup."""
        if col in self._data:
            return self._data[col]
        # Try bare column name (strip table prefix)
        if '.' in col:
            bare = col.split('.', 1)[1]
            if bare in self._data:
                return self._data[bare]
        # Try finding col as suffix
        for k in self._data:
            if k.endswith('.' + col):
                return self._data[k]
        return None

    def set(self, col: str, value: Any) -> 'Row':
        """Return new row with column set."""
        new_data = dict(self._data)
        new_data[col] = value
        return Row(new_data)

    def project(self, cols: List[str]) -> 'Row':
        """Return new row with only specified columns."""
        data = {}
        for c in cols:
            data[c] = self.get(c)
        return Row(data, cols)

    def merge(self, other: 'Row') -> 'Row':
        """Merge two rows (for joins)."""
        data = dict(self._data)
        data.update(other._data)
        return Row(data)

    def columns(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[Any]:
        return [self._data[k] for k in self._schema]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self):
        return f"Row({self._data})"

    def __eq__(self, other):
        if not isinstance(other, Row):
            return NotImplemented
        return self._data == other._data

    def __hash__(self):
        return hash(tuple(sorted(self._data.items())))


# ---------------------------------------------------------------------------
# Table storage (simple in-memory pages)
# ---------------------------------------------------------------------------

@dataclass
class Page:
    """A page of rows (simulates disk page)."""
    rows: List[Row] = field(default_factory=list)
    page_id: int = 0

    @property
    def num_rows(self):
        return len(self.rows)


class Table:
    """In-memory table with page-based storage."""

    def __init__(self, name: str, columns: List[str], page_size: int = 100):
        self.name = name
        self.columns = columns
        self.page_size = page_size
        self.pages: List[Page] = []
        self._row_count = 0
        self._indexes: Dict[str, 'TableIndex'] = {}

    def insert(self, row_data: Dict[str, Any]):
        """Insert a row into the table."""
        # Prefix columns with table name
        prefixed = {}
        for k, v in row_data.items():
            if '.' not in k:
                prefixed[f"{self.name}.{k}"] = v
            else:
                prefixed[k] = v

        row = Row(prefixed)

        if not self.pages or self.pages[-1].num_rows >= self.page_size:
            self.pages.append(Page(page_id=len(self.pages)))
        self.pages[-1].rows.append(row)
        self._row_count += 1

        # Update indexes
        for idx in self._indexes.values():
            idx.insert(row)

    def insert_many(self, rows: List[Dict[str, Any]]):
        for r in rows:
            self.insert(r)

    @property
    def row_count(self):
        return self._row_count

    @property
    def page_count(self):
        return len(self.pages)

    def add_index(self, name: str, column: str):
        """Add a B-tree-like index on a column."""
        idx = TableIndex(name, self.name, column)
        self._indexes[name] = idx
        # Build index from existing data
        for page in self.pages:
            for row in page.rows:
                idx.insert(row)
        return idx

    def get_index(self, column: str) -> Optional['TableIndex']:
        """Get index for a column, if one exists."""
        for idx in self._indexes.values():
            if idx.column == column or idx.column == f"{self.name}.{column}":
                return idx
        return None

    def scan_pages(self) -> Generator[Page, None, None]:
        """Yield all pages (simulates sequential I/O)."""
        for page in self.pages:
            yield page


class TableIndex:
    """Simple sorted index (simulates B-tree)."""

    def __init__(self, name: str, table_name: str, column: str):
        self.name = name
        self.table_name = table_name
        self.column = column
        self._entries: List[Tuple[Any, Row]] = []  # sorted by key
        self._sorted = True

    def insert(self, row: Row):
        key_col = self.column if '.' in self.column else f"{self.table_name}.{self.column}"
        val = row.get(key_col)
        self._entries.append((val, row))
        self._sorted = False

    def _ensure_sorted(self):
        if not self._sorted:
            self._entries.sort(key=lambda e: (e[0] is None, e[0] if e[0] is not None else 0))
            self._sorted = True

    def lookup_eq(self, value: Any) -> List[Row]:
        """Equality lookup."""
        self._ensure_sorted()
        results = []
        for k, row in self._entries:
            if k == value:
                results.append(row)
        return results

    def lookup_range(self, low: Any = None, high: Any = None,
                     low_inclusive: bool = True, high_inclusive: bool = True) -> List[Row]:
        """Range lookup."""
        self._ensure_sorted()
        results = []
        for k, row in self._entries:
            if k is None:
                continue
            if low is not None:
                if low_inclusive and k < low:
                    continue
                if not low_inclusive and k <= low:
                    continue
            if high is not None:
                if high_inclusive and k > high:
                    continue
                if not high_inclusive and k >= high:
                    continue
            results.append(row)
        return results


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

class CompOp(Enum):
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    IS_NULL = auto()
    IS_NOT_NULL = auto()
    LIKE = auto()
    IN = auto()
    BETWEEN = auto()


class LogicOp(Enum):
    AND = auto()
    OR = auto()
    NOT = auto()


@dataclass
class ColumnRef:
    """Reference to a column."""
    table: Optional[str]
    column: str

    @property
    def qualified(self) -> str:
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column


@dataclass
class Literal:
    """A literal value."""
    value: Any


@dataclass
class Comparison:
    """Comparison expression."""
    op: CompOp
    left: Any   # ColumnRef, Literal, or nested expr
    right: Any  # ColumnRef, Literal, or nested expr (None for IS_NULL)


@dataclass
class LogicExpr:
    """Logical expression (AND, OR, NOT)."""
    op: LogicOp
    operands: List[Any]


@dataclass
class ArithExpr:
    """Arithmetic expression."""
    op: str  # +, -, *, /
    left: Any
    right: Any


@dataclass
class FuncExpr:
    """Function call expression (for aggregates in HAVING, etc.)."""
    func: str
    args: List[Any]


@dataclass
class CaseExpr:
    """CASE WHEN expression."""
    whens: List[Tuple[Any, Any]]  # (condition, result)
    else_result: Any = None


def eval_expr(expr, row: Row) -> Any:
    """Evaluate an expression against a row."""
    if isinstance(expr, ColumnRef):
        return row.get(expr.qualified)
    elif isinstance(expr, Literal):
        return expr.value
    elif isinstance(expr, Comparison):
        return _eval_comparison(expr, row)
    elif isinstance(expr, LogicExpr):
        return _eval_logic(expr, row)
    elif isinstance(expr, ArithExpr):
        return _eval_arith(expr, row)
    elif isinstance(expr, CaseExpr):
        return _eval_case(expr, row)
    elif isinstance(expr, FuncExpr):
        # FuncExpr in row context -- used for computed columns
        args = [eval_expr(a, row) for a in expr.args]
        return _apply_func(expr.func, args)
    elif isinstance(expr, str):
        # Bare column name string
        return row.get(expr)
    elif isinstance(expr, (int, float, bool)):
        return expr
    elif expr is None:
        return None
    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")


def _eval_comparison(comp: Comparison, row: Row) -> bool:
    left = eval_expr(comp.left, row)
    right = eval_expr(comp.right, row) if comp.right is not None else None

    if comp.op == CompOp.EQ:
        return left == right
    elif comp.op == CompOp.NE:
        return left != right
    elif comp.op == CompOp.LT:
        if left is None or right is None:
            return False
        return left < right
    elif comp.op == CompOp.LE:
        if left is None or right is None:
            return False
        return left <= right
    elif comp.op == CompOp.GT:
        if left is None or right is None:
            return False
        return left > right
    elif comp.op == CompOp.GE:
        if left is None or right is None:
            return False
        return left >= right
    elif comp.op == CompOp.IS_NULL:
        return left is None
    elif comp.op == CompOp.IS_NOT_NULL:
        return left is not None
    elif comp.op == CompOp.LIKE:
        if left is None or right is None:
            return False
        return _match_like(str(left), str(right))
    elif comp.op == CompOp.IN:
        if isinstance(right, (list, tuple, set)):
            return left in right
        return left == right
    elif comp.op == CompOp.BETWEEN:
        if left is None:
            return False
        low, high = right
        return low <= left <= high
    return False


def _match_like(text: str, pattern: str) -> bool:
    """Simple LIKE pattern matching (% and _)."""
    import re
    regex = '^'
    for ch in pattern:
        if ch == '%':
            regex += '.*'
        elif ch == '_':
            regex += '.'
        else:
            regex += re.escape(ch)
    regex += '$'
    return bool(re.match(regex, text))


def _eval_logic(expr: LogicExpr, row: Row) -> bool:
    if expr.op == LogicOp.AND:
        return all(eval_expr(o, row) for o in expr.operands)
    elif expr.op == LogicOp.OR:
        return any(eval_expr(o, row) for o in expr.operands)
    elif expr.op == LogicOp.NOT:
        return not eval_expr(expr.operands[0], row)
    return False


def _eval_arith(expr: ArithExpr, row: Row) -> Any:
    left = eval_expr(expr.left, row)
    right = eval_expr(expr.right, row)
    if left is None or right is None:
        return None
    if expr.op == '+':
        return left + right
    elif expr.op == '-':
        return left - right
    elif expr.op == '*':
        return left * right
    elif expr.op == '/':
        if right == 0:
            return None
        return left / right
    return None


def _eval_case(expr: CaseExpr, row: Row) -> Any:
    for cond, result in expr.whens:
        if eval_expr(cond, row):
            return eval_expr(result, row)
    if expr.else_result is not None:
        return eval_expr(expr.else_result, row)
    return None


def _apply_func(name: str, args: List[Any]) -> Any:
    name = name.upper()
    if name == 'ABS':
        return abs(args[0]) if args[0] is not None else None
    elif name == 'UPPER':
        return str(args[0]).upper() if args[0] is not None else None
    elif name == 'LOWER':
        return str(args[0]).lower() if args[0] is not None else None
    elif name == 'LENGTH' or name == 'LEN':
        return len(str(args[0])) if args[0] is not None else None
    elif name == 'COALESCE':
        for a in args:
            if a is not None:
                return a
        return None
    elif name == 'CONCAT':
        return ''.join(str(a) for a in args if a is not None)
    return None


# ---------------------------------------------------------------------------
# Aggregate functions
# ---------------------------------------------------------------------------

class AggFunc(Enum):
    COUNT = auto()
    SUM = auto()
    AVG = auto()
    MIN = auto()
    MAX = auto()
    COUNT_STAR = auto()


@dataclass
class AggCall:
    """An aggregate function call."""
    func: AggFunc
    column: Optional[Any] = None  # ColumnRef, expr, or None for COUNT(*)
    distinct: bool = False
    alias: Optional[str] = None


class AggState:
    """Accumulator for an aggregate function."""

    def __init__(self, agg: AggCall):
        self.agg = agg
        self.count = 0
        self.sum_val = 0
        self.min_val = None
        self.max_val = None
        self._distinct_values: Set = set()

    def accumulate(self, row: Row):
        if self.agg.func == AggFunc.COUNT_STAR:
            self.count += 1
            return

        val = eval_expr(self.agg.column, row) if self.agg.column is not None else None

        if val is None:
            return

        if self.agg.distinct:
            if val in self._distinct_values:
                return
            self._distinct_values.add(val)

        self.count += 1

        if self.agg.func in (AggFunc.SUM, AggFunc.AVG):
            self.sum_val += val
        if self.agg.func == AggFunc.MIN:
            if self.min_val is None or val < self.min_val:
                self.min_val = val
        if self.agg.func == AggFunc.MAX:
            if self.max_val is None or val > self.max_val:
                self.max_val = val

    def result(self) -> Any:
        if self.agg.func == AggFunc.COUNT or self.agg.func == AggFunc.COUNT_STAR:
            return self.count
        elif self.agg.func == AggFunc.SUM:
            return self.sum_val if self.count > 0 else None
        elif self.agg.func == AggFunc.AVG:
            return self.sum_val / self.count if self.count > 0 else None
        elif self.agg.func == AggFunc.MIN:
            return self.min_val
        elif self.agg.func == AggFunc.MAX:
            return self.max_val
        return None


# ---------------------------------------------------------------------------
# Execution statistics
# ---------------------------------------------------------------------------

@dataclass
class ExecStats:
    """Execution statistics for an operator."""
    operator: str = ''
    rows_produced: int = 0
    rows_consumed: int = 0
    pages_read: int = 0
    memory_bytes: int = 0
    time_ms: float = 0.0
    children: List['ExecStats'] = field(default_factory=list)

    def total_rows(self) -> int:
        return self.rows_produced + sum(c.total_rows() for c in self.children)

    def total_pages(self) -> int:
        return self.pages_read + sum(c.total_pages() for c in self.children)

    def total_memory(self) -> int:
        return self.memory_bytes + sum(c.total_memory() for c in self.children)

    def to_dict(self) -> Dict:
        return {
            'operator': self.operator,
            'rows_produced': self.rows_produced,
            'pages_read': self.pages_read,
            'memory_bytes': self.memory_bytes,
            'children': [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# Physical operators (volcano/iterator model)
# ---------------------------------------------------------------------------

class Operator:
    """Base class for all physical operators."""

    def __init__(self):
        self.stats = ExecStats()
        self._is_open = False

    def open(self):
        """Initialize the operator."""
        self._is_open = True

    def next(self) -> Optional[Row]:
        """Get the next row, or None if exhausted."""
        raise NotImplementedError

    def close(self):
        """Release resources."""
        self._is_open = False

    def explain(self, indent: int = 0) -> str:
        """Return execution plan string."""
        return "  " * indent + self.__class__.__name__

    def __iter__(self):
        """Convenience iterator."""
        self.open()
        try:
            while True:
                row = self.next()
                if row is None:
                    break
                yield row
        finally:
            self.close()


class SeqScanOp(Operator):
    """Sequential scan of a table."""

    def __init__(self, table: Table):
        super().__init__()
        self.table = table
        self._page_iter = None
        self._row_iter = None
        self.stats.operator = f'SeqScan({table.name})'

    def open(self):
        super().open()
        self._page_iter = self.table.scan_pages()
        self._row_iter = iter([])

    def next(self) -> Optional[Row]:
        while True:
            try:
                row = next(self._row_iter)
                self.stats.rows_produced += 1
                return row
            except StopIteration:
                try:
                    page = next(self._page_iter)
                    self.stats.pages_read += 1
                    self._row_iter = iter(page.rows)
                except StopIteration:
                    return None

    def close(self):
        super().close()
        self._page_iter = None
        self._row_iter = None

    def explain(self, indent=0):
        return "  " * indent + f"SeqScan({self.table.name})"


class IndexScanOp(Operator):
    """Index scan using an index."""

    def __init__(self, table: Table, index: TableIndex,
                 lookup_value: Any = None,
                 low: Any = None, high: Any = None,
                 low_inclusive: bool = True, high_inclusive: bool = True):
        super().__init__()
        self.table = table
        self.index = index
        self.lookup_value = lookup_value
        self.low = low
        self.high = high
        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive
        self._results = None
        self._pos = 0
        self.stats.operator = f'IndexScan({table.name}.{index.name})'

    def open(self):
        super().open()
        if self.lookup_value is not None:
            self._results = self.index.lookup_eq(self.lookup_value)
        else:
            self._results = self.index.lookup_range(
                self.low, self.high, self.low_inclusive, self.high_inclusive)
        self._pos = 0
        self.stats.pages_read = max(1, len(self._results) // self.table.page_size)

    def next(self) -> Optional[Row]:
        if self._pos >= len(self._results):
            return None
        row = self._results[self._pos]
        self._pos += 1
        self.stats.rows_produced += 1
        return row

    def close(self):
        super().close()
        self._results = None

    def explain(self, indent=0):
        if self.lookup_value is not None:
            return "  " * indent + f"IndexScan({self.index.name}, eq={self.lookup_value})"
        return "  " * indent + f"IndexScan({self.index.name}, range=[{self.low},{self.high}])"


class FilterOp(Operator):
    """Filter rows by a predicate."""

    def __init__(self, child: Operator, predicate):
        super().__init__()
        self.child = child
        self.predicate = predicate
        self.stats.operator = 'Filter'

    def open(self):
        super().open()
        self.child.open()

    def next(self) -> Optional[Row]:
        while True:
            row = self.child.next()
            if row is None:
                return None
            self.stats.rows_consumed += 1
            if eval_expr(self.predicate, row):
                self.stats.rows_produced += 1
                return row

    def close(self):
        self.child.close()
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0):
        lines = ["  " * indent + f"Filter({self.predicate})"]
        lines.append(self.child.explain(indent + 1))
        return "\n".join(lines)


class ProjectOp(Operator):
    """Project specific columns or expressions."""

    def __init__(self, child: Operator, projections: List[Tuple[Any, str]]):
        """projections: list of (expression, alias) pairs."""
        super().__init__()
        self.child = child
        self.projections = projections
        self.stats.operator = 'Project'

    def open(self):
        super().open()
        self.child.open()

    def next(self) -> Optional[Row]:
        row = self.child.next()
        if row is None:
            return None
        self.stats.rows_consumed += 1
        data = {}
        for expr, alias in self.projections:
            data[alias] = eval_expr(expr, row)
        self.stats.rows_produced += 1
        return Row(data, [a for _, a in self.projections])

    def close(self):
        self.child.close()
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0):
        cols = [a for _, a in self.projections]
        lines = ["  " * indent + f"Project({', '.join(cols)})"]
        lines.append(self.child.explain(indent + 1))
        return "\n".join(lines)


class NestedLoopJoinOp(Operator):
    """Nested loop join (supports all join types)."""

    def __init__(self, left: Operator, right: Operator,
                 predicate=None, join_type: str = 'inner'):
        super().__init__()
        self.left = left
        self.right = right
        self.predicate = predicate
        self.join_type = join_type  # inner, left, cross
        self._left_row = None
        self._right_rows: List[Row] = []
        self._right_pos = 0
        self._left_matched = False
        self._exhausted = False
        self.stats.operator = f'NestedLoopJoin({join_type})'

    def open(self):
        super().open()
        self.left.open()
        self.right.open()
        # Materialize right side (inner relation)
        self._right_rows = []
        while True:
            r = self.right.next()
            if r is None:
                break
            self._right_rows.append(r)
        self.stats.memory_bytes = len(self._right_rows) * 100  # estimate
        self._left_row = self.left.next()
        self._right_pos = 0
        self._left_matched = False
        self._exhausted = (self._left_row is None)

    def next(self) -> Optional[Row]:
        while not self._exhausted:
            while self._right_pos < len(self._right_rows):
                right_row = self._right_rows[self._right_pos]
                self._right_pos += 1
                merged = self._left_row.merge(right_row)
                self.stats.rows_consumed += 1

                if self.predicate is None or eval_expr(self.predicate, merged):
                    self._left_matched = True
                    self.stats.rows_produced += 1
                    return merged

            # Right side exhausted for current left row
            if self.join_type == 'left' and not self._left_matched:
                # Emit left row with NULLs for right side
                null_data = {}
                if self._right_rows:
                    for col in self._right_rows[0].columns():
                        null_data[col] = None
                result = self._left_row.merge(Row(null_data))
                self.stats.rows_produced += 1
                self._advance_left()
                return result

            self._advance_left()

        return None

    def _advance_left(self):
        self._left_row = self.left.next()
        self._right_pos = 0
        self._left_matched = False
        if self._left_row is None:
            self._exhausted = True

    def close(self):
        self.left.close()
        self.right.close()
        self.stats.children = [self.left.stats, self.right.stats]
        super().close()

    def explain(self, indent=0):
        lines = ["  " * indent + f"NestedLoopJoin({self.join_type})"]
        lines.append(self.left.explain(indent + 1))
        lines.append(self.right.explain(indent + 1))
        return "\n".join(lines)


class HashJoinOp(Operator):
    """Hash join with build (right) and probe (left) phases."""

    def __init__(self, left: Operator, right: Operator,
                 left_key, right_key, join_type: str = 'inner'):
        super().__init__()
        self.left = left
        self.right = right
        self.left_key = left_key    # expression for left key
        self.right_key = right_key  # expression for right key
        self.join_type = join_type
        self._hash_table: Dict[Any, List[Row]] = {}
        self._probe_row = None
        self._matches: List[Row] = []
        self._match_pos = 0
        self._probe_matched = False
        self._exhausted = False
        self.stats.operator = f'HashJoin({join_type})'

    def open(self):
        super().open()
        self.left.open()
        self.right.open()

        # Build phase: hash the right (build) side
        self._hash_table = {}
        while True:
            row = self.right.next()
            if row is None:
                break
            key = eval_expr(self.right_key, row)
            if key not in self._hash_table:
                self._hash_table[key] = []
            self._hash_table[key].append(row)

        self.stats.memory_bytes = sum(
            len(rows) * 100 for rows in self._hash_table.values())

        # Start probe phase
        self._probe_row = self.left.next()
        self._match_pos = 0
        self._probe_matched = False
        self._exhausted = (self._probe_row is None)
        if self._probe_row is not None:
            key = eval_expr(self.left_key, self._probe_row)
            self._matches = self._hash_table.get(key, [])
            self.stats.rows_consumed += 1
        else:
            self._matches = []

    def next(self) -> Optional[Row]:
        while not self._exhausted:
            # Return remaining matches for current probe row
            while self._match_pos < len(self._matches):
                match = self._matches[self._match_pos]
                self._match_pos += 1
                merged = self._probe_row.merge(match)
                self._probe_matched = True
                self.stats.rows_produced += 1
                return merged

            # Left outer: emit unmatched probe row with NULLs
            if self.join_type == 'left' and not self._probe_matched and self._probe_row is not None:
                null_data = {}
                for bucket in self._hash_table.values():
                    if bucket:
                        for col in bucket[0].columns():
                            null_data[col] = None
                        break
                result = self._probe_row.merge(Row(null_data))
                self.stats.rows_produced += 1
                self._advance_probe()
                return result

            self._advance_probe()

        return None

    def _advance_probe(self):
        self._probe_row = self.left.next()
        if self._probe_row is None:
            self._exhausted = True
            return
        key = eval_expr(self.left_key, self._probe_row)
        self._matches = self._hash_table.get(key, [])
        self._match_pos = 0
        self._probe_matched = False
        self.stats.rows_consumed += 1

    def close(self):
        self.left.close()
        self.right.close()
        self._hash_table = {}
        self.stats.children = [self.left.stats, self.right.stats]
        super().close()

    def explain(self, indent=0):
        lines = ["  " * indent + f"HashJoin({self.join_type})"]
        lines.append(self.left.explain(indent + 1))
        lines.append(self.right.explain(indent + 1))
        return "\n".join(lines)


class SortMergeJoinOp(Operator):
    """Sort-merge join (assumes inputs sorted or sorts them)."""

    def __init__(self, left: Operator, right: Operator,
                 left_key, right_key, join_type: str = 'inner'):
        super().__init__()
        self.left = left
        self.right = right
        self.left_key = left_key
        self.right_key = right_key
        self.join_type = join_type
        self._left_sorted: List[Row] = []
        self._right_sorted: List[Row] = []
        self._left_pos = 0
        self._right_pos = 0
        self._right_group_start = 0
        self._in_group = False
        self.stats.operator = f'SortMergeJoin({join_type})'

    def open(self):
        super().open()
        self.left.open()
        self.right.open()

        # Materialize and sort both sides
        self._left_sorted = []
        while True:
            r = self.left.next()
            if r is None:
                break
            self._left_sorted.append(r)

        self._right_sorted = []
        while True:
            r = self.right.next()
            if r is None:
                break
            self._right_sorted.append(r)

        self._left_sorted.sort(key=lambda r: self._safe_key(eval_expr(self.left_key, r)))
        self._right_sorted.sort(key=lambda r: self._safe_key(eval_expr(self.right_key, r)))

        self.stats.memory_bytes = (len(self._left_sorted) + len(self._right_sorted)) * 100

        self._left_pos = 0
        self._right_pos = 0
        self._right_group_start = 0
        self._in_group = False

    def _safe_key(self, val):
        if val is None:
            return (1, 0)  # Sort NULLs last
        return (0, val)

    def next(self) -> Optional[Row]:
        while self._left_pos < len(self._left_sorted):
            left_row = self._left_sorted[self._left_pos]
            left_val = eval_expr(self.left_key, left_row)

            if self._in_group:
                # Continue scanning right group
                if self._right_pos < len(self._right_sorted):
                    right_row = self._right_sorted[self._right_pos]
                    right_val = eval_expr(self.right_key, right_row)
                    if left_val == right_val:
                        self._right_pos += 1
                        self.stats.rows_produced += 1
                        return left_row.merge(right_row)
                # Group exhausted, advance left
                self._left_pos += 1
                self._right_pos = self._right_group_start
                self._in_group = False
                continue

            # Advance right to find match
            while self._right_pos < len(self._right_sorted):
                right_row = self._right_sorted[self._right_pos]
                right_val = eval_expr(self.right_key, right_row)
                if right_val is None:
                    self._right_pos += 1
                    continue
                if left_val is None or left_val > right_val:
                    self._right_pos += 1
                    continue
                if left_val < right_val:
                    break
                # Match found
                self._right_group_start = self._right_pos
                self._in_group = True
                self._right_pos += 1
                self.stats.rows_produced += 1
                return left_row.merge(right_row)

            # No match, handle left join
            if self.join_type == 'left':
                null_data = {}
                if self._right_sorted:
                    for col in self._right_sorted[0].columns():
                        null_data[col] = None
                result = left_row.merge(Row(null_data))
                self.stats.rows_produced += 1
                self._left_pos += 1
                return result

            self._left_pos += 1

        return None

    def close(self):
        self.left.close()
        self.right.close()
        self._left_sorted = []
        self._right_sorted = []
        self.stats.children = [self.left.stats, self.right.stats]
        super().close()

    def explain(self, indent=0):
        lines = ["  " * indent + f"SortMergeJoin({self.join_type})"]
        lines.append(self.left.explain(indent + 1))
        lines.append(self.right.explain(indent + 1))
        return "\n".join(lines)


class SortOp(Operator):
    """Sort operator (pipeline breaker)."""

    def __init__(self, child: Operator, sort_keys: List[Tuple[Any, bool]]):
        """sort_keys: list of (expression, ascending) pairs."""
        super().__init__()
        self.child = child
        self.sort_keys = sort_keys
        self._sorted: List[Row] = []
        self._pos = 0
        self.stats.operator = 'Sort'

    def open(self):
        super().open()
        self.child.open()

        # Materialize and sort
        self._sorted = []
        while True:
            row = self.child.next()
            if row is None:
                break
            self._sorted.append(row)

        def sort_key(row):
            keys = []
            for expr, asc in self.sort_keys:
                val = eval_expr(expr, row)
                if val is None:
                    keys.append((1, 0 if asc else 0))
                else:
                    if asc:
                        keys.append((0, val))
                    else:
                        keys.append((0, _negate_for_sort(val)))
            return keys

        self._sorted.sort(key=sort_key)
        self.stats.memory_bytes = len(self._sorted) * 100
        self._pos = 0

    def next(self) -> Optional[Row]:
        if self._pos >= len(self._sorted):
            return None
        row = self._sorted[self._pos]
        self._pos += 1
        self.stats.rows_produced += 1
        return row

    def close(self):
        self.child.close()
        self._sorted = []
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0):
        keys = []
        for expr, asc in self.sort_keys:
            d = "ASC" if asc else "DESC"
            keys.append(f"{expr} {d}")
        lines = ["  " * indent + f"Sort({', '.join(keys)})"]
        lines.append(self.child.explain(indent + 1))
        return "\n".join(lines)


def _negate_for_sort(val):
    """Helper for descending sort."""
    if isinstance(val, (int, float)):
        return -val
    elif isinstance(val, str):
        # Invert string for descending sort
        return tuple(-ord(c) for c in val)
    return val


class HashAggregateOp(Operator):
    """Hash aggregate with grouping."""

    def __init__(self, child: Operator,
                 group_by: List[Any],
                 aggregates: List[AggCall]):
        super().__init__()
        self.child = child
        self.group_by = group_by  # list of expressions
        self.aggregates = aggregates
        self._groups: Dict[tuple, List[AggState]] = {}
        self._result_iter = None
        self.stats.operator = 'HashAggregate'

    def open(self):
        super().open()
        self.child.open()

        # Consume all input and build groups
        self._groups = {}
        while True:
            row = self.child.next()
            if row is None:
                break
            self.stats.rows_consumed += 1

            # Compute group key
            key = tuple(eval_expr(g, row) for g in self.group_by) if self.group_by else ()

            if key not in self._groups:
                self._groups[key] = [AggState(agg) for agg in self.aggregates]

            for state in self._groups[key]:
                state.accumulate(row)

        self.stats.memory_bytes = len(self._groups) * 200

        # Handle empty input with no group-by (scalar aggregate)
        if not self._groups and not self.group_by:
            self._groups[()] = [AggState(agg) for agg in self.aggregates]

        self._result_iter = iter(self._groups.items())

    def next(self) -> Optional[Row]:
        try:
            key, states = next(self._result_iter)
        except StopIteration:
            return None

        data = {}
        # Add group-by columns
        if self.group_by:
            for i, g in enumerate(self.group_by):
                col_name = g.qualified if isinstance(g, ColumnRef) else str(g)
                data[col_name] = key[i]

        # Add aggregate results
        for state in states:
            alias = state.agg.alias or f"{state.agg.func.name}({state.agg.column})"
            data[alias] = state.result()

        self.stats.rows_produced += 1
        return Row(data)

    def close(self):
        self.child.close()
        self._groups = {}
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0):
        grp = [str(g) for g in self.group_by]
        aggs = [f"{a.func.name}({a.column})" for a in self.aggregates]
        lines = ["  " * indent + f"HashAggregate(group=[{', '.join(grp)}], aggs=[{', '.join(aggs)}])"]
        lines.append(self.child.explain(indent + 1))
        return "\n".join(lines)


class LimitOp(Operator):
    """Limit and offset operator."""

    def __init__(self, child: Operator, limit: int, offset: int = 0):
        super().__init__()
        self.child = child
        self.limit = limit
        self.offset = offset
        self._count = 0
        self._skipped = 0
        self.stats.operator = f'Limit({limit})'

    def open(self):
        super().open()
        self.child.open()
        self._count = 0
        self._skipped = 0

    def next(self) -> Optional[Row]:
        # Skip offset rows
        while self._skipped < self.offset:
            row = self.child.next()
            if row is None:
                return None
            self._skipped += 1

        if self._count >= self.limit:
            return None
        row = self.child.next()
        if row is None:
            return None
        self._count += 1
        self.stats.rows_produced += 1
        return row

    def close(self):
        self.child.close()
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0):
        s = f"Limit({self.limit}"
        if self.offset:
            s += f", offset={self.offset}"
        s += ")"
        lines = ["  " * indent + s]
        lines.append(self.child.explain(indent + 1))
        return "\n".join(lines)


class UnionOp(Operator):
    """Union of two operators."""

    def __init__(self, left: Operator, right: Operator, all: bool = True):
        super().__init__()
        self.left = left
        self.right = right
        self.all = all
        self._on_right = False
        self._seen: Set = set()
        self.stats.operator = 'Union'

    def open(self):
        super().open()
        self.left.open()
        self.right.open()
        self._on_right = False
        self._seen = set()

    def next(self) -> Optional[Row]:
        while True:
            if not self._on_right:
                row = self.left.next()
                if row is None:
                    self._on_right = True
                    continue
            else:
                row = self.right.next()
                if row is None:
                    return None

            if not self.all:
                key = tuple(sorted(row.to_dict().items()))
                if key in self._seen:
                    continue
                self._seen.add(key)

            self.stats.rows_produced += 1
            return row

    def close(self):
        self.left.close()
        self.right.close()
        self._seen = set()
        self.stats.children = [self.left.stats, self.right.stats]
        super().close()

    def explain(self, indent=0):
        kind = "UnionAll" if self.all else "Union"
        lines = ["  " * indent + kind]
        lines.append(self.left.explain(indent + 1))
        lines.append(self.right.explain(indent + 1))
        return "\n".join(lines)


class DistinctOp(Operator):
    """Remove duplicate rows."""

    def __init__(self, child: Operator):
        super().__init__()
        self.child = child
        self._seen: Set = set()
        self.stats.operator = 'Distinct'

    def open(self):
        super().open()
        self.child.open()
        self._seen = set()

    def next(self) -> Optional[Row]:
        while True:
            row = self.child.next()
            if row is None:
                return None
            key = tuple(sorted(row.to_dict().items()))
            if key not in self._seen:
                self._seen.add(key)
                self.stats.rows_produced += 1
                return row

    def close(self):
        self.child.close()
        self._seen = set()
        self.stats.children = [self.child.stats]
        super().close()


class TopNOp(Operator):
    """Top-N operator using a heap (avoids full sort for LIMIT + ORDER BY)."""

    def __init__(self, child: Operator, sort_keys: List[Tuple[Any, bool]], n: int):
        super().__init__()
        self.child = child
        self.sort_keys = sort_keys
        self.n = n
        self._heap: List = []
        self._results: List[Row] = []
        self._pos = 0
        self.stats.operator = f'TopN({n})'

    def open(self):
        super().open()
        self.child.open()
        self._heap = []
        counter = 0

        while True:
            row = self.child.next()
            if row is None:
                break
            self.stats.rows_consumed += 1
            key = self._make_key(row)
            entry = (key, counter, row)
            counter += 1

            if len(self._heap) < self.n:
                heapq.heappush(self._heap, entry)
            else:
                heapq.heappushpop(self._heap, entry)

        # Extract results in sorted order
        self._results = [row for _, _, row in sorted(self._heap, reverse=True)]
        self._pos = 0
        self.stats.memory_bytes = self.n * 100

    def _make_key(self, row: Row):
        """Create a comparison key (inverted for min-heap)."""
        keys = []
        for expr, asc in self.sort_keys:
            val = eval_expr(expr, row)
            if val is None:
                keys.append((0, 0))  # NULLs sort first in inverted
            else:
                if asc:
                    # For ascending, we want the largest to be evicted -> negate
                    keys.append((1, _negate_for_sort(val)))
                else:
                    keys.append((1, val))
        return tuple(keys)

    def next(self) -> Optional[Row]:
        if self._pos >= len(self._results):
            return None
        row = self._results[self._pos]
        self._pos += 1
        self.stats.rows_produced += 1
        return row

    def close(self):
        self.child.close()
        self._heap = []
        self._results = []
        self.stats.children = [self.child.stats]
        super().close()


class MaterializeOp(Operator):
    """Materialize child into memory (for subquery/CTE reuse)."""

    def __init__(self, child: Operator):
        super().__init__()
        self.child = child
        self._rows: List[Row] = []
        self._pos = 0
        self._materialized = False
        self.stats.operator = 'Materialize'

    def open(self):
        super().open()
        if not self._materialized:
            self.child.open()
            self._rows = []
            while True:
                row = self.child.next()
                if row is None:
                    break
                self._rows.append(row)
            self.child.close()
            self._materialized = True
            self.stats.memory_bytes = len(self._rows) * 100
        self._pos = 0

    def next(self) -> Optional[Row]:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        self.stats.rows_produced += 1
        return row

    def close(self):
        self._pos = 0
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0):
        lines = ["  " * indent + f"Materialize({len(self._rows)} rows)"]
        lines.append(self.child.explain(indent + 1))
        return "\n".join(lines)


class SemiJoinOp(Operator):
    """Semi-join: returns left rows that have at least one match in right."""

    def __init__(self, left: Operator, right: Operator, predicate=None):
        super().__init__()
        self.left = left
        self.right = right
        self.predicate = predicate
        self._right_rows: List[Row] = []
        self.stats.operator = 'SemiJoin'

    def open(self):
        super().open()
        self.left.open()
        self.right.open()
        self._right_rows = []
        while True:
            r = self.right.next()
            if r is None:
                break
            self._right_rows.append(r)
        self.stats.memory_bytes = len(self._right_rows) * 100

    def next(self) -> Optional[Row]:
        while True:
            left_row = self.left.next()
            if left_row is None:
                return None
            for right_row in self._right_rows:
                merged = left_row.merge(right_row)
                if self.predicate is None or eval_expr(self.predicate, merged):
                    self.stats.rows_produced += 1
                    return left_row
            # No match, continue to next left row

    def close(self):
        self.left.close()
        self.right.close()
        self._right_rows = []
        self.stats.children = [self.left.stats, self.right.stats]
        super().close()


class AntiJoinOp(Operator):
    """Anti-join: returns left rows that have NO match in right."""

    def __init__(self, left: Operator, right: Operator, predicate=None):
        super().__init__()
        self.left = left
        self.right = right
        self.predicate = predicate
        self._right_rows: List[Row] = []
        self.stats.operator = 'AntiJoin'

    def open(self):
        super().open()
        self.left.open()
        self.right.open()
        self._right_rows = []
        while True:
            r = self.right.next()
            if r is None:
                break
            self._right_rows.append(r)
        self.stats.memory_bytes = len(self._right_rows) * 100

    def next(self) -> Optional[Row]:
        while True:
            left_row = self.left.next()
            if left_row is None:
                return None
            matched = False
            for right_row in self._right_rows:
                merged = left_row.merge(right_row)
                if self.predicate is None or eval_expr(self.predicate, merged):
                    matched = True
                    break
            if not matched:
                self.stats.rows_produced += 1
                return left_row

    def close(self):
        self.left.close()
        self.right.close()
        self._right_rows = []
        self.stats.children = [self.left.stats, self.right.stats]
        super().close()


# ---------------------------------------------------------------------------
# Having filter (post-aggregate)
# ---------------------------------------------------------------------------

class HavingOp(Operator):
    """Filter on aggregate results (HAVING clause)."""

    def __init__(self, child: Operator, predicate):
        super().__init__()
        self.child = child
        self.predicate = predicate
        self.stats.operator = 'Having'

    def open(self):
        super().open()
        self.child.open()

    def next(self) -> Optional[Row]:
        while True:
            row = self.child.next()
            if row is None:
                return None
            if eval_expr(self.predicate, row):
                self.stats.rows_produced += 1
                return row

    def close(self):
        self.child.close()
        self.stats.children = [self.child.stats]
        super().close()


# ---------------------------------------------------------------------------
# Database and query execution engine
# ---------------------------------------------------------------------------

class Database:
    """In-memory database holding tables."""

    def __init__(self):
        self.tables: Dict[str, Table] = {}

    def create_table(self, name: str, columns: List[str],
                     page_size: int = 100) -> Table:
        t = Table(name, columns, page_size)
        self.tables[name] = t
        return t

    def get_table(self, name: str) -> Optional[Table]:
        return self.tables.get(name)

    def drop_table(self, name: str):
        self.tables.pop(name, None)


class ExecutionEngine:
    """Executes physical operator trees and collects results."""

    def __init__(self, db: Database):
        self.db = db

    def execute(self, plan: Operator) -> List[Row]:
        """Execute a plan and return all result rows."""
        results = []
        plan.open()
        try:
            while True:
                row = plan.next()
                if row is None:
                    break
                results.append(row)
        finally:
            plan.close()
        return results

    def execute_iter(self, plan: Operator) -> Generator[Row, None, None]:
        """Execute a plan and yield rows one at a time."""
        plan.open()
        try:
            while True:
                row = plan.next()
                if row is None:
                    break
                yield row
        finally:
            plan.close()

    def explain(self, plan: Operator) -> str:
        """Return the execution plan as a string."""
        return plan.explain()

    def explain_analyze(self, plan: Operator) -> Dict:
        """Execute and return plan with runtime statistics."""
        results = self.execute(plan)
        return {
            'rows': len(results),
            'stats': plan.stats.to_dict(),
            'results': results,
        }


# ---------------------------------------------------------------------------
# Query builder (fluent API)
# ---------------------------------------------------------------------------

class QueryPlan:
    """Fluent builder for constructing physical plans."""

    def __init__(self, db: Database):
        self.db = db
        self._root: Optional[Operator] = None

    def scan(self, table_name: str) -> 'QueryPlan':
        t = self.db.get_table(table_name)
        if t is None:
            raise ValueError(f"Table not found: {table_name}")
        self._root = SeqScanOp(t)
        return self

    def index_scan(self, table_name: str, column: str,
                   value: Any = None, low: Any = None, high: Any = None) -> 'QueryPlan':
        t = self.db.get_table(table_name)
        if t is None:
            raise ValueError(f"Table not found: {table_name}")
        idx = t.get_index(column)
        if idx is None:
            raise ValueError(f"No index on {table_name}.{column}")
        self._root = IndexScanOp(t, idx, lookup_value=value, low=low, high=high)
        return self

    def filter(self, predicate) -> 'QueryPlan':
        self._root = FilterOp(self._root, predicate)
        return self

    def project(self, projections: List[Tuple[Any, str]]) -> 'QueryPlan':
        self._root = ProjectOp(self._root, projections)
        return self

    def hash_join(self, right: 'QueryPlan', left_key, right_key,
                  join_type: str = 'inner') -> 'QueryPlan':
        self._root = HashJoinOp(self._root, right._root, left_key, right_key, join_type)
        return self

    def nested_loop_join(self, right: 'QueryPlan', predicate=None,
                         join_type: str = 'inner') -> 'QueryPlan':
        self._root = NestedLoopJoinOp(self._root, right._root, predicate, join_type)
        return self

    def sort_merge_join(self, right: 'QueryPlan', left_key, right_key,
                        join_type: str = 'inner') -> 'QueryPlan':
        self._root = SortMergeJoinOp(self._root, right._root, left_key, right_key, join_type)
        return self

    def sort(self, keys: List[Tuple[Any, bool]]) -> 'QueryPlan':
        self._root = SortOp(self._root, keys)
        return self

    def aggregate(self, group_by: List[Any], aggregates: List[AggCall]) -> 'QueryPlan':
        self._root = HashAggregateOp(self._root, group_by, aggregates)
        return self

    def having(self, predicate) -> 'QueryPlan':
        self._root = HavingOp(self._root, predicate)
        return self

    def limit(self, n: int, offset: int = 0) -> 'QueryPlan':
        self._root = LimitOp(self._root, n, offset)
        return self

    def distinct(self) -> 'QueryPlan':
        self._root = DistinctOp(self._root)
        return self

    def union(self, other: 'QueryPlan', all: bool = True) -> 'QueryPlan':
        self._root = UnionOp(self._root, other._root, all)
        return self

    def top_n(self, keys: List[Tuple[Any, bool]], n: int) -> 'QueryPlan':
        self._root = TopNOp(self._root, keys, n)
        return self

    def materialize(self) -> 'QueryPlan':
        self._root = MaterializeOp(self._root)
        return self

    def semi_join(self, right: 'QueryPlan', predicate=None) -> 'QueryPlan':
        self._root = SemiJoinOp(self._root, right._root, predicate)
        return self

    def anti_join(self, right: 'QueryPlan', predicate=None) -> 'QueryPlan':
        self._root = AntiJoinOp(self._root, right._root, predicate)
        return self

    def build(self) -> Operator:
        return self._root

    def execute(self) -> List[Row]:
        engine = ExecutionEngine(self.db)
        return engine.execute(self._root)

    def explain(self) -> str:
        return self._root.explain()
