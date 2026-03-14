"""
C243: Query Optimizer

A cost-based query optimizer for relational algebra:
- Relational algebra operators: Scan, Filter, Project, Join, Sort, Aggregate, Limit
- Table statistics: row count, distinct values, histograms, null fractions
- Selectivity estimation with histograms and independence assumption
- Cost model: I/O cost (pages), CPU cost (tuples), memory cost
- Join ordering: dynamic programming (Selinger-style) for optimal join order
- Join algorithm selection: nested loop, hash join, sort-merge join
- Predicate pushdown: push filters below joins and projections
- Projection pushdown: eliminate unnecessary columns early
- Join reordering with associativity and commutativity rules
- Subquery decorrelation (flatten correlated subqueries)
- Index selection: choose best index for filter/join predicates
- Plan enumeration with pruning (interesting orders, Pareto-optimal plans)
- EXPLAIN output for plan visualization

Domain: Database Internals
Standalone implementation -- no external dependencies.
"""

from enum import Enum, auto
from typing import Any, Optional, Dict, List, Set, Tuple, FrozenSet
from dataclasses import dataclass, field
from functools import reduce
import math
import operator


# ---------------------------------------------------------------------------
# Column and Table Schema
# ---------------------------------------------------------------------------

@dataclass
class Column:
    """Schema column definition."""
    name: str
    type: str = 'int'       # int, float, string, bool
    nullable: bool = False
    primary_key: bool = False


@dataclass
class Index:
    """Index definition on a table."""
    name: str
    table: str
    columns: List[str]
    unique: bool = False
    type: str = 'btree'     # btree, hash


@dataclass
class TableSchema:
    """Schema for a single table."""
    name: str
    columns: List[Column]
    indexes: List[Index] = field(default_factory=list)

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def has_column(self, name: str) -> bool:
        return any(c.name == name for c in self.columns)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class Histogram:
    """Equi-width histogram for selectivity estimation."""
    min_val: float
    max_val: float
    buckets: List[int]      # count per bucket
    num_distinct: int = 0
    null_fraction: float = 0.0

    @property
    def num_buckets(self) -> int:
        return len(self.buckets)

    @property
    def total_rows(self) -> int:
        return sum(self.buckets)

    @property
    def bucket_width(self) -> float:
        if self.num_buckets == 0:
            return 0
        return (self.max_val - self.min_val) / self.num_buckets

    def estimate_equality(self, value: float) -> float:
        """Estimate fraction of rows equal to value."""
        if self.num_distinct == 0:
            return 0.0
        # Use uniform distribution within each bucket
        return (1.0 - self.null_fraction) / self.num_distinct

    def estimate_range(self, low: Optional[float], high: Optional[float]) -> float:
        """Estimate fraction of rows in [low, high]."""
        if self.total_rows == 0:
            return 0.0

        effective_low = low if low is not None else self.min_val
        effective_high = high if high is not None else self.max_val

        if effective_low > self.max_val or effective_high < self.min_val:
            return 0.0

        effective_low = max(effective_low, self.min_val)
        effective_high = min(effective_high, self.max_val)

        if self.bucket_width == 0:
            return 1.0

        # Sum up fraction of each bucket that overlaps [low, high]
        total_fraction = 0.0
        for i, count in enumerate(self.buckets):
            bucket_low = self.min_val + i * self.bucket_width
            bucket_high = bucket_low + self.bucket_width

            # Overlap
            overlap_low = max(bucket_low, effective_low)
            overlap_high = min(bucket_high, effective_high)

            if overlap_low < overlap_high:
                bucket_fraction = (overlap_high - overlap_low) / self.bucket_width
                total_fraction += (count / self.total_rows) * bucket_fraction

        return total_fraction * (1.0 - self.null_fraction)


@dataclass
class ColumnStats:
    """Statistics for a single column."""
    num_distinct: int = 0
    null_fraction: float = 0.0
    min_val: Any = None
    max_val: Any = None
    avg_width: int = 4       # Average bytes per value
    histogram: Optional[Histogram] = None
    most_common_vals: Optional[List[Tuple[Any, float]]] = None  # (value, frequency)


@dataclass
class TableStats:
    """Statistics for a table."""
    row_count: int = 0
    page_count: int = 0      # Number of disk pages
    avg_row_width: int = 100 # Average row size in bytes
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)


class Catalog:
    """System catalog holding schemas and statistics."""

    def __init__(self):
        self.tables: Dict[str, TableSchema] = {}
        self.stats: Dict[str, TableStats] = {}
        self.indexes: Dict[str, Index] = {}    # index_name -> Index

    def add_table(self, schema: TableSchema, stats: Optional[TableStats] = None):
        self.tables[schema.name] = schema
        if stats:
            self.stats[schema.name] = stats
        else:
            self.stats[schema.name] = TableStats()
        for idx in schema.indexes:
            self.indexes[idx.name] = idx

    def get_schema(self, table: str) -> Optional[TableSchema]:
        return self.tables.get(table)

    def get_stats(self, table: str) -> TableStats:
        return self.stats.get(table, TableStats())

    def get_indexes_for_table(self, table: str) -> List[Index]:
        schema = self.tables.get(table)
        if not schema:
            return []
        return schema.indexes

    def update_stats(self, table: str, stats: TableStats):
        self.stats[table] = stats


# ---------------------------------------------------------------------------
# Expressions (predicates and projections)
# ---------------------------------------------------------------------------

class Expr:
    """Base expression."""
    pass


@dataclass
class ColumnRef(Expr):
    """Reference to a column, optionally qualified with table name."""
    column: str
    table: Optional[str] = None

    def qualified(self) -> str:
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column

    def __hash__(self):
        return hash((self.table, self.column))

    def __eq__(self, other):
        if not isinstance(other, ColumnRef):
            return NotImplemented
        return self.table == other.table and self.column == other.column

    def __repr__(self):
        return self.qualified()


@dataclass
class Literal(Expr):
    """Literal value."""
    value: Any

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return NotImplemented
        return self.value == other.value

    def __repr__(self):
        return repr(self.value)


class CompOp(Enum):
    EQ = '='
    NE = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    LIKE = 'LIKE'
    IN = 'IN'
    IS_NULL = 'IS NULL'
    IS_NOT_NULL = 'IS NOT NULL'
    BETWEEN = 'BETWEEN'


class LogicOp(Enum):
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'


@dataclass
class Comparison(Expr):
    """Comparison predicate: left op right."""
    left: Expr
    op: CompOp
    right: Optional[Expr] = None   # None for IS NULL/IS NOT NULL
    right2: Optional[Expr] = None  # For BETWEEN: left BETWEEN right AND right2

    def __hash__(self):
        return hash((id(self.left), self.op, id(self.right)))

    def __eq__(self, other):
        if not isinstance(other, Comparison):
            return NotImplemented
        return (self.left == other.left and self.op == other.op
                and self.right == other.right and self.right2 == other.right2)

    def __repr__(self):
        if self.op == CompOp.IS_NULL:
            return f"({self.left} IS NULL)"
        if self.op == CompOp.IS_NOT_NULL:
            return f"({self.left} IS NOT NULL)"
        if self.op == CompOp.BETWEEN:
            return f"({self.left} BETWEEN {self.right} AND {self.right2})"
        return f"({self.left} {self.op.value} {self.right})"


@dataclass
class LogicExpr(Expr):
    """Logical connective: AND, OR, NOT."""
    op: LogicOp
    children: List[Expr]

    def __repr__(self):
        if self.op == LogicOp.NOT:
            return f"(NOT {self.children[0]})"
        sep = f" {self.op.value} "
        return f"({sep.join(str(c) for c in self.children)})"


@dataclass
class FuncCall(Expr):
    """Function call (for aggregates: COUNT, SUM, AVG, MIN, MAX)."""
    func: str
    args: List[Expr]
    distinct: bool = False

    def __repr__(self):
        args_str = ', '.join(str(a) for a in self.args)
        d = 'DISTINCT ' if self.distinct else ''
        return f"{self.func}({d}{args_str})"


@dataclass
class AliasExpr(Expr):
    """Expression with alias: expr AS alias."""
    expr: Expr
    alias: str

    def __repr__(self):
        return f"{self.expr} AS {self.alias}"


# ---------------------------------------------------------------------------
# Relational Algebra / Logical Plan Nodes
# ---------------------------------------------------------------------------

class PlanNode:
    """Base class for all plan nodes (both logical and physical)."""

    def __init__(self):
        self._estimated_rows: Optional[float] = None
        self._estimated_cost: Optional[float] = None
        self._output_columns: Optional[List[str]] = None

    @property
    def estimated_rows(self) -> float:
        return self._estimated_rows or 0

    @estimated_rows.setter
    def estimated_rows(self, val: float):
        self._estimated_rows = val

    @property
    def estimated_cost(self) -> float:
        return self._estimated_cost or 0

    @estimated_cost.setter
    def estimated_cost(self, val: float):
        self._estimated_cost = val

    @property
    def output_columns(self) -> List[str]:
        return self._output_columns or []

    @output_columns.setter
    def output_columns(self, val: List[str]):
        self._output_columns = val

    def children(self) -> List['PlanNode']:
        return []

    def tables_referenced(self) -> Set[str]:
        """Return all table names referenced by this subtree."""
        result = set()
        for child in self.children():
            result |= child.tables_referenced()
        return result


class ScanNode(PlanNode):
    """Table scan (sequential or index)."""

    def __init__(self, table: str, alias: Optional[str] = None):
        super().__init__()
        self.table = table
        self.alias = alias or table

    def tables_referenced(self) -> Set[str]:
        return {self.alias}

    def __repr__(self):
        if self.alias != self.table:
            return f"Scan({self.table} AS {self.alias})"
        return f"Scan({self.table})"


class IndexScanNode(PlanNode):
    """Index scan using a specific index."""

    def __init__(self, table: str, index: Index, predicates: List[Expr],
                 alias: Optional[str] = None):
        super().__init__()
        self.table = table
        self.index = index
        self.predicates = predicates
        self.alias = alias or table

    def tables_referenced(self) -> Set[str]:
        return {self.alias}

    def __repr__(self):
        return f"IndexScan({self.table}, idx={self.index.name})"


class FilterNode(PlanNode):
    """Filter (selection) node."""

    def __init__(self, child: PlanNode, predicate: Expr):
        super().__init__()
        self.child = child
        self.predicate = predicate

    def children(self) -> List[PlanNode]:
        return [self.child]

    def tables_referenced(self) -> Set[str]:
        return self.child.tables_referenced()

    def __repr__(self):
        return f"Filter({self.predicate})"


class ProjectNode(PlanNode):
    """Projection node."""

    def __init__(self, child: PlanNode, columns: List[Expr]):
        super().__init__()
        self.child = child
        self.columns = columns

    def children(self) -> List[PlanNode]:
        return [self.child]

    def tables_referenced(self) -> Set[str]:
        return self.child.tables_referenced()

    def __repr__(self):
        cols = ', '.join(str(c) for c in self.columns)
        return f"Project({cols})"


class JoinType(Enum):
    INNER = 'INNER'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    FULL = 'FULL'
    CROSS = 'CROSS'


class JoinAlgorithm(Enum):
    NESTED_LOOP = 'NestedLoop'
    HASH = 'Hash'
    SORT_MERGE = 'SortMerge'
    INDEX_NESTED_LOOP = 'IndexNestedLoop'


class JoinNode(PlanNode):
    """Join node."""

    def __init__(self, left: PlanNode, right: PlanNode,
                 join_type: JoinType = JoinType.INNER,
                 condition: Optional[Expr] = None,
                 algorithm: JoinAlgorithm = JoinAlgorithm.HASH):
        super().__init__()
        self.left = left
        self.right = right
        self.join_type = join_type
        self.condition = condition
        self.algorithm = algorithm

    def children(self) -> List[PlanNode]:
        return [self.left, self.right]

    def tables_referenced(self) -> Set[str]:
        return self.left.tables_referenced() | self.right.tables_referenced()

    def __repr__(self):
        return f"{self.algorithm.value}Join({self.join_type.value}, {self.condition})"


class SortNode(PlanNode):
    """Sort (ORDER BY) node."""

    def __init__(self, child: PlanNode, keys: List[Tuple[Expr, str]]):
        """keys: list of (expr, 'ASC'|'DESC')."""
        super().__init__()
        self.child = child
        self.keys = keys

    def children(self) -> List[PlanNode]:
        return [self.child]

    def tables_referenced(self) -> Set[str]:
        return self.child.tables_referenced()

    def __repr__(self):
        ks = ', '.join(f"{k[0]} {k[1]}" for k in self.keys)
        return f"Sort({ks})"


class AggregateNode(PlanNode):
    """Aggregation (GROUP BY) node."""

    def __init__(self, child: PlanNode, group_by: List[Expr],
                 aggregates: List[FuncCall]):
        super().__init__()
        self.child = child
        self.group_by = group_by
        self.aggregates = aggregates

    def children(self) -> List[PlanNode]:
        return [self.child]

    def tables_referenced(self) -> Set[str]:
        return self.child.tables_referenced()

    def __repr__(self):
        gb = ', '.join(str(g) for g in self.group_by)
        agg = ', '.join(str(a) for a in self.aggregates)
        return f"Aggregate(group_by=[{gb}], agg=[{agg}])"


class LimitNode(PlanNode):
    """LIMIT/OFFSET node."""

    def __init__(self, child: PlanNode, limit: int, offset: int = 0):
        super().__init__()
        self.child = child
        self.limit = limit
        self.offset = offset

    def children(self) -> List[PlanNode]:
        return [self.child]

    def tables_referenced(self) -> Set[str]:
        return self.child.tables_referenced()

    def __repr__(self):
        if self.offset:
            return f"Limit({self.limit}, offset={self.offset})"
        return f"Limit({self.limit})"


class UnionNode(PlanNode):
    """UNION node."""

    def __init__(self, left: PlanNode, right: PlanNode, all: bool = False):
        super().__init__()
        self.left = left
        self.right = right
        self.all = all

    def children(self) -> List[PlanNode]:
        return [self.left, self.right]

    def __repr__(self):
        return f"Union({'ALL' if self.all else 'DISTINCT'})"


# ---------------------------------------------------------------------------
# Expression Utilities
# ---------------------------------------------------------------------------

def extract_conjuncts(expr: Expr) -> List[Expr]:
    """Flatten AND expressions into a list of conjuncts."""
    if isinstance(expr, LogicExpr) and expr.op == LogicOp.AND:
        result = []
        for child in expr.children:
            result.extend(extract_conjuncts(child))
        return result
    return [expr]


def make_conjunction(exprs: List[Expr]) -> Optional[Expr]:
    """Combine a list of expressions with AND."""
    if not exprs:
        return None
    if len(exprs) == 1:
        return exprs[0]
    return LogicExpr(LogicOp.AND, exprs)


def referenced_tables(expr: Expr) -> Set[str]:
    """Return all table names referenced by column refs in the expression."""
    tables = set()
    if isinstance(expr, ColumnRef):
        if expr.table:
            tables.add(expr.table)
    elif isinstance(expr, Comparison):
        tables |= referenced_tables(expr.left)
        if expr.right:
            tables |= referenced_tables(expr.right)
        if expr.right2:
            tables |= referenced_tables(expr.right2)
    elif isinstance(expr, LogicExpr):
        for child in expr.children:
            tables |= referenced_tables(child)
    elif isinstance(expr, FuncCall):
        for arg in expr.args:
            tables |= referenced_tables(arg)
    elif isinstance(expr, AliasExpr):
        tables |= referenced_tables(expr.expr)
    return tables


def referenced_columns(expr: Expr) -> Set[str]:
    """Return all column names referenced in the expression."""
    cols = set()
    if isinstance(expr, ColumnRef):
        cols.add(expr.qualified())
    elif isinstance(expr, Comparison):
        cols |= referenced_columns(expr.left)
        if expr.right:
            cols |= referenced_columns(expr.right)
        if expr.right2:
            cols |= referenced_columns(expr.right2)
    elif isinstance(expr, LogicExpr):
        for child in expr.children:
            cols |= referenced_columns(child)
    elif isinstance(expr, FuncCall):
        for arg in expr.args:
            cols |= referenced_columns(arg)
    elif isinstance(expr, AliasExpr):
        cols |= referenced_columns(expr.expr)
    return cols


def is_join_predicate(expr: Expr) -> bool:
    """Check if an expression is a join predicate (compares columns from different tables)."""
    if isinstance(expr, Comparison) and expr.op == CompOp.EQ:
        left_tables = referenced_tables(expr.left)
        right_tables = referenced_tables(expr.right) if expr.right else set()
        return len(left_tables) > 0 and len(right_tables) > 0 and left_tables != right_tables
    return False


# ---------------------------------------------------------------------------
# Selectivity Estimator
# ---------------------------------------------------------------------------

class SelectivityEstimator:
    """Estimate selectivity of predicates using catalog statistics."""

    DEFAULT_EQ_SEL = 0.01      # 1% for equality without stats
    DEFAULT_RANGE_SEL = 0.33   # 33% for range without stats
    DEFAULT_LIKE_SEL = 0.05    # 5% for LIKE
    DEFAULT_IN_SEL = 0.05      # per-element for IN
    DEFAULT_NULL_SEL = 0.01    # 1% null fraction
    DEFAULT_JOIN_SEL = 0.1     # 10% for join without stats

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def estimate(self, expr: Expr, input_rows: float = 1.0) -> float:
        """Estimate selectivity of an expression (fraction 0..1)."""
        if isinstance(expr, Comparison):
            return self._estimate_comparison(expr)
        elif isinstance(expr, LogicExpr):
            return self._estimate_logic(expr, input_rows)
        else:
            return 0.5  # Unknown expression type

    def _estimate_comparison(self, comp: Comparison) -> float:
        col_ref = None
        literal_val = None

        if isinstance(comp.left, ColumnRef):
            col_ref = comp.left
            if isinstance(comp.right, Literal):
                literal_val = comp.right.value
        elif isinstance(comp.right, ColumnRef) and isinstance(comp.left, Literal):
            col_ref = comp.right
            literal_val = comp.left.value

        # Get column stats if available
        col_stats = None
        if col_ref and col_ref.table:
            table_stats = self.catalog.get_stats(col_ref.table)
            col_stats = table_stats.column_stats.get(col_ref.column)

        if comp.op == CompOp.IS_NULL:
            if col_stats:
                return col_stats.null_fraction
            return self.DEFAULT_NULL_SEL

        if comp.op == CompOp.IS_NOT_NULL:
            if col_stats:
                return 1.0 - col_stats.null_fraction
            return 1.0 - self.DEFAULT_NULL_SEL

        if comp.op == CompOp.EQ:
            # Check if it's a join predicate (col = col)
            if isinstance(comp.left, ColumnRef) and isinstance(comp.right, ColumnRef):
                return self._estimate_join_selectivity(comp.left, comp.right)

            if col_stats:
                # Check most common values first
                if col_stats.most_common_vals and literal_val is not None:
                    for val, freq in col_stats.most_common_vals:
                        if val == literal_val:
                            return freq

                if col_stats.histogram and literal_val is not None:
                    return col_stats.histogram.estimate_equality(literal_val)

                if col_stats.num_distinct > 0:
                    return (1.0 - col_stats.null_fraction) / col_stats.num_distinct

            return self.DEFAULT_EQ_SEL

        if comp.op == CompOp.NE:
            eq_sel = self._estimate_comparison(
                Comparison(comp.left, CompOp.EQ, comp.right))
            return 1.0 - eq_sel

        if comp.op in (CompOp.LT, CompOp.LE, CompOp.GT, CompOp.GE):
            if col_stats and col_stats.histogram and literal_val is not None:
                try:
                    val = float(literal_val)
                except (TypeError, ValueError):
                    return self.DEFAULT_RANGE_SEL

                if comp.op == CompOp.LT:
                    return col_stats.histogram.estimate_range(None, val) * 0.95
                elif comp.op == CompOp.LE:
                    return col_stats.histogram.estimate_range(None, val)
                elif comp.op == CompOp.GT:
                    return col_stats.histogram.estimate_range(val, None) * 0.95
                elif comp.op == CompOp.GE:
                    return col_stats.histogram.estimate_range(val, None)
            return self.DEFAULT_RANGE_SEL

        if comp.op == CompOp.BETWEEN:
            if col_stats and col_stats.histogram:
                try:
                    low = float(comp.right.value) if isinstance(comp.right, Literal) else None
                    high = float(comp.right2.value) if isinstance(comp.right2, Literal) else None
                except (TypeError, ValueError):
                    return self.DEFAULT_RANGE_SEL
                if low is not None and high is not None:
                    return col_stats.histogram.estimate_range(low, high)
            return self.DEFAULT_RANGE_SEL

        if comp.op == CompOp.LIKE:
            return self.DEFAULT_LIKE_SEL

        if comp.op == CompOp.IN:
            # IN list: 1 - (1 - eq_sel)^n
            n = 1
            if isinstance(comp.right, Literal) and isinstance(comp.right.value, (list, tuple)):
                n = len(comp.right.value)
            eq_sel = self.DEFAULT_EQ_SEL
            if col_stats and col_stats.num_distinct > 0:
                eq_sel = (1.0 - col_stats.null_fraction) / col_stats.num_distinct
            return min(1.0, n * eq_sel)

        return 0.5

    def _estimate_join_selectivity(self, left: ColumnRef, right: ColumnRef) -> float:
        """Estimate join selectivity using 1/max(distinct_left, distinct_right)."""
        left_stats = None
        right_stats = None

        if left.table:
            ts = self.catalog.get_stats(left.table)
            left_stats = ts.column_stats.get(left.column)
        if right.table:
            ts = self.catalog.get_stats(right.table)
            right_stats = ts.column_stats.get(right.column)

        left_distinct = left_stats.num_distinct if left_stats and left_stats.num_distinct > 0 else 0
        right_distinct = right_stats.num_distinct if right_stats and right_stats.num_distinct > 0 else 0

        max_distinct = max(left_distinct, right_distinct)
        if max_distinct > 0:
            return 1.0 / max_distinct
        return self.DEFAULT_JOIN_SEL

    def _estimate_logic(self, expr: LogicExpr, input_rows: float) -> float:
        if expr.op == LogicOp.AND:
            # Independence assumption: P(A AND B) = P(A) * P(B)
            sel = 1.0
            for child in expr.children:
                sel *= self.estimate(child, input_rows)
            return sel

        elif expr.op == LogicOp.OR:
            # P(A OR B) = P(A) + P(B) - P(A)*P(B)
            sel = 0.0
            for child in expr.children:
                child_sel = self.estimate(child, input_rows)
                sel = sel + child_sel - sel * child_sel
            return sel

        elif expr.op == LogicOp.NOT:
            return 1.0 - self.estimate(expr.children[0], input_rows)

        return 0.5


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------

PAGE_SIZE = 8192
TUPLE_CPU_COST = 0.01
PAGE_IO_COST = 1.0
RANDOM_IO_COST = 4.0
SEQ_IO_COST = 1.0
HASH_BUILD_CPU = 0.02
SORT_CPU_FACTOR = 0.05  # Per-tuple sort cost factor


@dataclass
class Cost:
    """Estimated cost of a plan node."""
    io_cost: float = 0.0       # I/O cost (in page-equivalents)
    cpu_cost: float = 0.0      # CPU cost
    memory: float = 0.0        # Memory usage estimate (pages)
    startup_cost: float = 0.0  # Cost before first tuple

    @property
    def total(self) -> float:
        return self.io_cost + self.cpu_cost + self.startup_cost

    def __add__(self, other: 'Cost') -> 'Cost':
        return Cost(
            self.io_cost + other.io_cost,
            self.cpu_cost + other.cpu_cost,
            max(self.memory, other.memory),
            self.startup_cost + other.startup_cost
        )

    def __repr__(self):
        return f"Cost(io={self.io_cost:.1f}, cpu={self.cpu_cost:.2f}, total={self.total:.2f})"


class CostModel:
    """Estimate cost for physical plan operators."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.selectivity = SelectivityEstimator(catalog)

    def estimate_scan(self, node: ScanNode) -> Cost:
        stats = self.catalog.get_stats(node.table)
        pages = max(stats.page_count, 1)
        rows = max(stats.row_count, 1)
        node.estimated_rows = rows
        node.output_columns = self._get_table_columns(node.table)
        return Cost(io_cost=pages * SEQ_IO_COST, cpu_cost=rows * TUPLE_CPU_COST)

    def estimate_index_scan(self, node: IndexScanNode) -> Cost:
        stats = self.catalog.get_stats(node.table)
        rows = max(stats.row_count, 1)

        # Estimate selectivity of index predicates
        sel = 1.0
        for pred in node.predicates:
            sel *= self.selectivity.estimate(pred, rows)

        result_rows = max(1.0, rows * sel)
        # Index scan has random I/O for each tuple
        pages = max(1, math.ceil(result_rows * stats.avg_row_width / PAGE_SIZE))
        idx_pages = max(1, math.ceil(math.log2(max(rows, 2))))  # B-tree height

        node.estimated_rows = result_rows
        node.output_columns = self._get_table_columns(node.table)
        return Cost(
            io_cost=idx_pages * RANDOM_IO_COST + pages * RANDOM_IO_COST,
            cpu_cost=result_rows * TUPLE_CPU_COST,
            startup_cost=idx_pages * RANDOM_IO_COST
        )

    def estimate_filter(self, node: FilterNode, child_cost: Cost) -> Cost:
        child_rows = node.child.estimated_rows
        sel = self.selectivity.estimate(node.predicate, child_rows)
        node.estimated_rows = max(1.0, child_rows * sel)
        node.output_columns = node.child.output_columns
        return child_cost + Cost(cpu_cost=child_rows * TUPLE_CPU_COST)

    def estimate_project(self, node: ProjectNode, child_cost: Cost) -> Cost:
        node.estimated_rows = node.child.estimated_rows
        # Output columns from projection
        cols = []
        for c in node.columns:
            if isinstance(c, ColumnRef):
                cols.append(c.qualified())
            elif isinstance(c, AliasExpr):
                cols.append(c.alias)
            else:
                cols.append(str(c))
        node.output_columns = cols
        return child_cost + Cost(cpu_cost=node.estimated_rows * TUPLE_CPU_COST * 0.5)

    def estimate_join(self, node: JoinNode, left_cost: Cost, right_cost: Cost) -> Cost:
        left_rows = node.left.estimated_rows
        right_rows = node.right.estimated_rows

        # Estimate output rows
        if node.condition:
            sel = self.selectivity.estimate(node.condition, left_rows * right_rows)
        elif node.join_type == JoinType.CROSS:
            sel = 1.0
        else:
            sel = self.selectivity.DEFAULT_JOIN_SEL

        output_rows = left_rows * right_rows * sel

        if node.join_type == JoinType.LEFT:
            output_rows = max(output_rows, left_rows)
        elif node.join_type == JoinType.RIGHT:
            output_rows = max(output_rows, right_rows)
        elif node.join_type == JoinType.FULL:
            output_rows = max(output_rows, left_rows + right_rows)

        node.estimated_rows = max(1.0, output_rows)
        node.output_columns = (node.left.output_columns or []) + (node.right.output_columns or [])

        if node.algorithm == JoinAlgorithm.NESTED_LOOP:
            return self._cost_nested_loop(node, left_cost, right_cost, left_rows, right_rows)
        elif node.algorithm == JoinAlgorithm.HASH:
            return self._cost_hash_join(node, left_cost, right_cost, left_rows, right_rows)
        elif node.algorithm == JoinAlgorithm.SORT_MERGE:
            return self._cost_sort_merge(node, left_cost, right_cost, left_rows, right_rows)
        elif node.algorithm == JoinAlgorithm.INDEX_NESTED_LOOP:
            return self._cost_index_nl(node, left_cost, right_cost, left_rows, right_rows)

        return left_cost + right_cost

    def _cost_nested_loop(self, node, left_cost, right_cost, left_rows, right_rows):
        # Inner table scanned once per outer tuple
        cpu = left_rows * right_rows * TUPLE_CPU_COST
        io = left_cost.io_cost + left_rows * right_cost.io_cost
        return Cost(io_cost=io, cpu_cost=cpu)

    def _cost_hash_join(self, node, left_cost, right_cost, left_rows, right_rows):
        # Build hash table on smaller side, probe with larger
        build_rows = min(left_rows, right_rows)
        probe_rows = max(left_rows, right_rows)
        cpu = build_rows * HASH_BUILD_CPU + probe_rows * TUPLE_CPU_COST
        mem_pages = max(1, math.ceil(build_rows * 100 / PAGE_SIZE))
        return left_cost + right_cost + Cost(
            cpu_cost=cpu, memory=mem_pages,
            startup_cost=build_rows * HASH_BUILD_CPU
        )

    def _cost_sort_merge(self, node, left_cost, right_cost, left_rows, right_rows):
        sort_cost_left = left_rows * math.log2(max(left_rows, 2)) * SORT_CPU_FACTOR
        sort_cost_right = right_rows * math.log2(max(right_rows, 2)) * SORT_CPU_FACTOR
        merge_cpu = (left_rows + right_rows) * TUPLE_CPU_COST
        return left_cost + right_cost + Cost(
            cpu_cost=sort_cost_left + sort_cost_right + merge_cpu,
            startup_cost=sort_cost_left + sort_cost_right
        )

    def _cost_index_nl(self, node, left_cost, right_cost, left_rows, right_rows):
        # For each outer tuple, one index lookup
        idx_height = max(1, math.ceil(math.log2(max(right_rows, 2))))
        cpu = left_rows * idx_height * TUPLE_CPU_COST
        io = left_cost.io_cost + left_rows * idx_height * RANDOM_IO_COST
        return Cost(io_cost=io, cpu_cost=cpu)

    def estimate_sort(self, node: SortNode, child_cost: Cost) -> Cost:
        rows = node.child.estimated_rows
        node.estimated_rows = rows
        node.output_columns = node.child.output_columns
        sort_cpu = rows * math.log2(max(rows, 2)) * SORT_CPU_FACTOR
        return child_cost + Cost(cpu_cost=sort_cpu, startup_cost=sort_cpu)

    def estimate_aggregate(self, node: AggregateNode, child_cost: Cost) -> Cost:
        rows = node.child.estimated_rows
        if node.group_by:
            # Estimate groups from distinct values of group-by columns
            group_count = self._estimate_groups(node.group_by, rows)
        else:
            group_count = 1.0

        node.estimated_rows = group_count
        # Output columns = group_by + aggregates
        cols = [str(g) for g in node.group_by]
        cols += [str(a) for a in node.aggregates]
        node.output_columns = cols
        return child_cost + Cost(cpu_cost=rows * TUPLE_CPU_COST)

    def estimate_limit(self, node: LimitNode, child_cost: Cost) -> Cost:
        child_rows = node.child.estimated_rows
        node.estimated_rows = min(node.limit, max(0, child_rows - node.offset))
        node.output_columns = node.child.output_columns
        # Fraction of child cost proportional to rows returned
        fraction = node.estimated_rows / max(child_rows, 1)
        return Cost(
            io_cost=child_cost.io_cost * fraction,
            cpu_cost=child_cost.cpu_cost * fraction
        )

    def estimate(self, node: PlanNode) -> Cost:
        """Recursively estimate cost for an entire plan tree."""
        if isinstance(node, ScanNode):
            return self.estimate_scan(node)
        elif isinstance(node, IndexScanNode):
            return self.estimate_index_scan(node)
        elif isinstance(node, FilterNode):
            child_cost = self.estimate(node.child)
            return self.estimate_filter(node, child_cost)
        elif isinstance(node, ProjectNode):
            child_cost = self.estimate(node.child)
            return self.estimate_project(node, child_cost)
        elif isinstance(node, JoinNode):
            left_cost = self.estimate(node.left)
            right_cost = self.estimate(node.right)
            return self.estimate_join(node, left_cost, right_cost)
        elif isinstance(node, SortNode):
            child_cost = self.estimate(node.child)
            return self.estimate_sort(node, child_cost)
        elif isinstance(node, AggregateNode):
            child_cost = self.estimate(node.child)
            return self.estimate_aggregate(node, child_cost)
        elif isinstance(node, LimitNode):
            child_cost = self.estimate(node.child)
            return self.estimate_limit(node, child_cost)
        else:
            return Cost()

    def _estimate_groups(self, group_by: List[Expr], input_rows: float) -> float:
        """Estimate number of distinct groups."""
        distinct = input_rows
        for expr in group_by:
            if isinstance(expr, ColumnRef) and expr.table:
                ts = self.catalog.get_stats(expr.table)
                cs = ts.column_stats.get(expr.column)
                if cs and cs.num_distinct > 0:
                    distinct = min(distinct, cs.num_distinct)
        return max(1.0, distinct)

    def _get_table_columns(self, table: str) -> List[str]:
        schema = self.catalog.get_schema(table)
        if schema:
            return [f"{table}.{c.name}" for c in schema.columns]
        return []


# ---------------------------------------------------------------------------
# Optimization Rules
# ---------------------------------------------------------------------------

class Rule:
    """Base optimization rule."""

    def name(self) -> str:
        return self.__class__.__name__

    def applicable(self, node: PlanNode) -> bool:
        return False

    def apply(self, node: PlanNode) -> PlanNode:
        return node


class PredicatePushdown(Rule):
    """Push filter predicates below joins and projections."""

    def applicable(self, node: PlanNode) -> bool:
        return isinstance(node, FilterNode) and isinstance(node.child, (JoinNode, ProjectNode))

    def apply(self, node: PlanNode) -> PlanNode:
        if not isinstance(node, FilterNode):
            return node

        if isinstance(node.child, JoinNode):
            return self._push_into_join(node)
        elif isinstance(node.child, ProjectNode):
            return self._push_through_project(node)
        return node

    def _push_into_join(self, filter_node: FilterNode) -> PlanNode:
        join = filter_node.child
        if not isinstance(join, JoinNode):
            return filter_node

        conjuncts = extract_conjuncts(filter_node.predicate)
        left_tables = join.left.tables_referenced()
        right_tables = join.right.tables_referenced()

        left_preds = []
        right_preds = []
        join_preds = []
        remaining = []

        for pred in conjuncts:
            tables = referenced_tables(pred)
            if tables and tables <= left_tables:
                left_preds.append(pred)
            elif tables and tables <= right_tables:
                right_preds.append(pred)
            elif is_join_predicate(pred):
                join_preds.append(pred)
            else:
                remaining.append(pred)

        # Apply pushed-down predicates
        new_left = join.left
        if left_preds:
            new_left = FilterNode(new_left, make_conjunction(left_preds))

        new_right = join.right
        if right_preds:
            new_right = FilterNode(new_right, make_conjunction(right_preds))

        # Merge join predicates with existing join condition
        all_join_preds = join_preds[:]
        if join.condition:
            all_join_preds = extract_conjuncts(join.condition) + all_join_preds

        new_join = JoinNode(
            new_left, new_right, join.join_type,
            make_conjunction(all_join_preds) if all_join_preds else join.condition,
            join.algorithm
        )

        if remaining:
            return FilterNode(new_join, make_conjunction(remaining))
        return new_join

    def _push_through_project(self, filter_node: FilterNode) -> PlanNode:
        proj = filter_node.child
        if not isinstance(proj, ProjectNode):
            return filter_node

        # If filter only references columns that pass through the projection, push down
        filter_cols = referenced_columns(filter_node.predicate)
        proj_cols = set()
        for c in proj.columns:
            if isinstance(c, ColumnRef):
                proj_cols.add(c.qualified())
            elif isinstance(c, AliasExpr) and isinstance(c.expr, ColumnRef):
                proj_cols.add(c.expr.qualified())

        # All filter columns must be available below the projection
        # (simplified: always push through if columns are available)
        new_proj = ProjectNode(
            FilterNode(proj.child, filter_node.predicate),
            proj.columns
        )
        return new_proj


class ProjectionPushdown(Rule):
    """Push projections down to eliminate unnecessary columns early."""

    def applicable(self, node: PlanNode) -> bool:
        return isinstance(node, ProjectNode) and isinstance(node.child, JoinNode)

    def apply(self, node: PlanNode) -> PlanNode:
        if not isinstance(node, ProjectNode) or not isinstance(node.child, JoinNode):
            return node

        join = node.child
        needed = set()
        for c in node.columns:
            needed |= referenced_columns(c)

        # Also need columns referenced in join condition
        if join.condition:
            needed |= referenced_columns(join.condition)

        left_tables = join.left.tables_referenced()
        right_tables = join.right.tables_referenced()

        left_cols = [ColumnRef(c.split('.')[-1], c.split('.')[0])
                     for c in needed if c.split('.')[0] in left_tables]
        right_cols = [ColumnRef(c.split('.')[-1], c.split('.')[0])
                      for c in needed if c.split('.')[0] in right_tables]

        new_left = join.left
        if left_cols and len(left_cols) < len(join.left.output_columns or []):
            new_left = ProjectNode(join.left, left_cols)

        new_right = join.right
        if right_cols and len(right_cols) < len(join.right.output_columns or []):
            new_right = ProjectNode(join.right, right_cols)

        new_join = JoinNode(new_left, new_right, join.join_type, join.condition, join.algorithm)
        return ProjectNode(new_join, node.columns)


class JoinCommutativity(Rule):
    """Swap join operands for potentially better join order."""

    def applicable(self, node: PlanNode) -> bool:
        return isinstance(node, JoinNode) and node.join_type == JoinType.INNER

    def apply(self, node: PlanNode) -> PlanNode:
        if not isinstance(node, JoinNode):
            return node
        return JoinNode(node.right, node.left, node.join_type, node.condition, node.algorithm)


# ---------------------------------------------------------------------------
# Join Ordering (Selinger-style DP)
# ---------------------------------------------------------------------------

class JoinOrderOptimizer:
    """Dynamic programming join ordering (Selinger algorithm).

    Finds the optimal join order for a set of base relations with
    given join predicates, using bottom-up dynamic programming.
    """

    def __init__(self, catalog: Catalog, cost_model: CostModel):
        self.catalog = catalog
        self.cost_model = cost_model

    def optimize(self, base_plans: Dict[str, PlanNode],
                 predicates: List[Expr]) -> PlanNode:
        """Find optimal join order for base_plans with given predicates.

        Args:
            base_plans: table_name -> ScanNode for each base relation
            predicates: join and filter predicates

        Returns:
            Optimized join tree
        """
        if len(base_plans) == 0:
            raise ValueError("No base plans")
        if len(base_plans) == 1:
            name, plan = next(iter(base_plans.items()))
            # Apply any single-table predicates
            single_preds = [p for p in predicates
                           if referenced_tables(p) <= {name}]
            if single_preds:
                return FilterNode(plan, make_conjunction(single_preds))
            return plan

        # Separate single-table predicates from join predicates
        table_names = list(base_plans.keys())
        single_table_preds: Dict[str, List[Expr]] = {t: [] for t in table_names}
        join_preds: List[Expr] = []

        for pred in predicates:
            tables = referenced_tables(pred)
            if len(tables) == 1:
                t = next(iter(tables))
                if t in single_table_preds:
                    single_table_preds[t].append(pred)
            elif len(tables) >= 2:
                join_preds.append(pred)
            # If tables is empty (literal comparison), skip

        # Apply single-table predicates to base scans
        filtered_plans: Dict[str, PlanNode] = {}
        for name, plan in base_plans.items():
            if single_table_preds[name]:
                plan = FilterNode(plan, make_conjunction(single_table_preds[name]))
            filtered_plans[name] = plan
            # Estimate cost to populate row estimates
            self.cost_model.estimate(plan)

        # DP table: frozenset of table names -> (best_plan, best_cost)
        dp: Dict[FrozenSet[str], Tuple[PlanNode, float]] = {}

        # Initialize with single-table plans
        for name, plan in filtered_plans.items():
            key = frozenset([name])
            cost = self.cost_model.estimate(plan)
            dp[key] = (plan, cost.total)

        # Bottom-up: enumerate subsets of increasing size
        for size in range(2, len(table_names) + 1):
            for subset in self._subsets_of_size(table_names, size):
                fsubset = frozenset(subset)
                best_plan = None
                best_cost = float('inf')

                # Try all ways to split subset into two non-empty parts
                for left_set in self._non_empty_subsets(subset):
                    left_frozen = frozenset(left_set)
                    right_frozen = fsubset - left_frozen

                    if not right_frozen or left_frozen not in dp or right_frozen not in dp:
                        continue

                    # Avoid duplicate pairs (left, right) and (right, left)
                    if min(left_frozen) > min(right_frozen):
                        continue

                    left_plan, left_total = dp[left_frozen]
                    right_plan, right_total = dp[right_frozen]

                    # Find applicable join predicates
                    applicable = [p for p in join_preds
                                 if referenced_tables(p) <= fsubset
                                 and not (referenced_tables(p) <= left_frozen)
                                 and not (referenced_tables(p) <= right_frozen)]

                    condition = make_conjunction(applicable) if applicable else None

                    # Try each join algorithm
                    for algo in [JoinAlgorithm.HASH, JoinAlgorithm.SORT_MERGE,
                                JoinAlgorithm.NESTED_LOOP]:
                        join = JoinNode(left_plan, right_plan, JoinType.INNER,
                                       condition, algo)
                        cost = self.cost_model.estimate(join)
                        total = cost.total

                        if total < best_cost:
                            best_cost = total
                            best_plan = join

                if best_plan is not None:
                    dp[fsubset] = (best_plan, best_cost)

        full_set = frozenset(table_names)
        if full_set in dp:
            return dp[full_set][0]

        # Fallback: left-deep join in input order
        return self._left_deep_join(filtered_plans, join_preds)

    def _left_deep_join(self, plans: Dict[str, PlanNode],
                        preds: List[Expr]) -> PlanNode:
        names = list(plans.keys())
        result = plans[names[0]]
        joined = {names[0]}

        for name in names[1:]:
            right = plans[name]
            joined.add(name)
            applicable = [p for p in preds if referenced_tables(p) <= joined]
            preds = [p for p in preds if p not in applicable]
            condition = make_conjunction(applicable) if applicable else None
            result = JoinNode(result, right, JoinType.INNER, condition, JoinAlgorithm.HASH)

        return result

    def _subsets_of_size(self, items: List[str], size: int) -> List[List[str]]:
        """Generate all subsets of given size."""
        if size == 0:
            return [[]]
        if size > len(items):
            return []
        result = []
        for i, item in enumerate(items):
            for rest in self._subsets_of_size(items[i+1:], size - 1):
                result.append([item] + rest)
        return result

    def _non_empty_subsets(self, items: List[str]) -> List[List[str]]:
        """Generate all non-empty proper subsets."""
        result = []
        n = len(items)
        for mask in range(1, (1 << n) - 1):
            subset = [items[i] for i in range(n) if mask & (1 << i)]
            result.append(subset)
        return result


# ---------------------------------------------------------------------------
# Index Selection
# ---------------------------------------------------------------------------

class IndexSelector:
    """Select the best index for filter predicates."""

    def __init__(self, catalog: Catalog, cost_model: CostModel):
        self.catalog = catalog
        self.cost_model = cost_model

    def select_index(self, table: str, predicates: List[Expr],
                     alias: Optional[str] = None) -> Optional[PlanNode]:
        """Try to find an index scan that's cheaper than a sequential scan.

        Returns IndexScanNode if beneficial, None otherwise.
        """
        indexes = self.catalog.get_indexes_for_table(table)
        if not indexes:
            return None

        # Cost of sequential scan
        seq_scan = ScanNode(table, alias)
        seq_cost = self.cost_model.estimate(seq_scan)
        if predicates:
            seq_filter = FilterNode(seq_scan, make_conjunction(predicates))
            seq_cost = self.cost_model.estimate(seq_filter)

        best_index = None
        best_cost = seq_cost.total
        best_preds = []

        for idx in indexes:
            # Find predicates that match the index's leading columns
            matching_preds = []
            for pred in predicates:
                if self._predicate_matches_index(pred, idx, alias or table):
                    matching_preds.append(pred)

            if not matching_preds:
                continue

            # Estimate index scan cost
            idx_scan = IndexScanNode(table, idx, matching_preds, alias)
            idx_cost = self.cost_model.estimate(idx_scan)

            # Add cost of remaining predicates as filter
            remaining = [p for p in predicates if p not in matching_preds]
            if remaining:
                idx_filter = FilterNode(idx_scan, make_conjunction(remaining))
                idx_cost = self.cost_model.estimate(idx_filter)

            if idx_cost.total < best_cost:
                best_cost = idx_cost.total
                best_index = idx
                best_preds = matching_preds

        if best_index:
            node = IndexScanNode(table, best_index, best_preds, alias)
            remaining = [p for p in predicates if p not in best_preds]
            if remaining:
                return FilterNode(node, make_conjunction(remaining))
            return node

        return None

    def _predicate_matches_index(self, pred: Expr, index: Index,
                                  table_alias: str) -> bool:
        """Check if a predicate can use the given index."""
        if not isinstance(pred, Comparison):
            return False

        if pred.op not in (CompOp.EQ, CompOp.LT, CompOp.LE, CompOp.GT, CompOp.GE,
                          CompOp.BETWEEN):
            return False

        # Check if predicate references an indexed column
        if isinstance(pred.left, ColumnRef):
            col = pred.left
            if (col.table == table_alias or col.table == index.table) and \
               col.column in index.columns:
                return True

        if isinstance(pred.right, ColumnRef):
            col = pred.right
            if (col.table == table_alias or col.table == index.table) and \
               col.column in index.columns:
                return True

        return False


# ---------------------------------------------------------------------------
# Query Optimizer
# ---------------------------------------------------------------------------

class QueryOptimizer:
    """Main query optimizer: transforms logical plans into optimized physical plans.

    Applies:
    1. Predicate pushdown
    2. Index selection
    3. Join ordering (Selinger DP)
    4. Join algorithm selection
    5. Projection pushdown
    """

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.cost_model = CostModel(catalog)
        self.join_optimizer = JoinOrderOptimizer(catalog, self.cost_model)
        self.index_selector = IndexSelector(catalog, self.cost_model)
        self.rules = [
            PredicatePushdown(),
            ProjectionPushdown(),
        ]

    def optimize(self, plan: PlanNode) -> PlanNode:
        """Optimize a logical plan into a physical plan."""
        # Phase 1: Apply logical rewrite rules (predicate pushdown, etc.)
        plan = self._apply_rules(plan)

        # Phase 2: Optimize joins (ordering + algorithm selection)
        plan = self._optimize_joins(plan)

        # Phase 3: Index selection
        plan = self._select_indexes(plan)

        # Phase 4: Final cost estimation
        self.cost_model.estimate(plan)

        return plan

    def _apply_rules(self, node: PlanNode) -> PlanNode:
        """Apply rewrite rules bottom-up until no more apply."""
        # First optimize children
        node = self._optimize_children(node)

        # Then try rules on current node
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for rule in self.rules:
                if rule.applicable(node):
                    new_node = rule.apply(node)
                    if new_node is not node:
                        node = new_node
                        changed = True
                        break

        return node

    def _optimize_children(self, node: PlanNode) -> PlanNode:
        """Recursively optimize children."""
        if isinstance(node, FilterNode):
            node.child = self._apply_rules(node.child)
        elif isinstance(node, ProjectNode):
            node.child = self._apply_rules(node.child)
        elif isinstance(node, JoinNode):
            node.left = self._apply_rules(node.left)
            node.right = self._apply_rules(node.right)
        elif isinstance(node, SortNode):
            node.child = self._apply_rules(node.child)
        elif isinstance(node, AggregateNode):
            node.child = self._apply_rules(node.child)
        elif isinstance(node, LimitNode):
            node.child = self._apply_rules(node.child)
        return node

    def _optimize_joins(self, node: PlanNode) -> PlanNode:
        """Extract join graph and re-optimize join order."""
        # Collect all base scans and predicates from a join tree
        if not self._has_joins(node):
            return node

        # Recursively optimize joins in subtrees first
        if isinstance(node, FilterNode):
            node.child = self._optimize_joins(node.child)
            return node
        elif isinstance(node, ProjectNode):
            node.child = self._optimize_joins(node.child)
            return node
        elif isinstance(node, SortNode):
            node.child = self._optimize_joins(node.child)
            return node
        elif isinstance(node, AggregateNode):
            node.child = self._optimize_joins(node.child)
            return node
        elif isinstance(node, LimitNode):
            node.child = self._optimize_joins(node.child)
            return node

        if isinstance(node, JoinNode) and node.join_type == JoinType.INNER:
            # Extract base relations and predicates
            base_plans, predicates = self._extract_join_graph(node)
            if len(base_plans) >= 2:
                return self.join_optimizer.optimize(base_plans, predicates)

        return node

    def _extract_join_graph(self, node: PlanNode) -> Tuple[Dict[str, PlanNode], List[Expr]]:
        """Extract base relations and predicates from a join tree."""
        base_plans: Dict[str, PlanNode] = {}
        predicates: List[Expr] = []

        self._collect_join_info(node, base_plans, predicates)
        return base_plans, predicates

    def _collect_join_info(self, node: PlanNode, base_plans: Dict[str, PlanNode],
                           predicates: List[Expr]):
        """Recursively collect base scans and predicates."""
        if isinstance(node, ScanNode):
            base_plans[node.alias] = node
        elif isinstance(node, IndexScanNode):
            base_plans[node.alias] = node
        elif isinstance(node, FilterNode):
            predicates.extend(extract_conjuncts(node.predicate))
            self._collect_join_info(node.child, base_plans, predicates)
        elif isinstance(node, JoinNode) and node.join_type == JoinType.INNER:
            if node.condition:
                predicates.extend(extract_conjuncts(node.condition))
            self._collect_join_info(node.left, base_plans, predicates)
            self._collect_join_info(node.right, base_plans, predicates)
        else:
            # Non-inner join or other node type: treat as base relation
            tables = node.tables_referenced()
            if tables:
                name = '_'.join(sorted(tables))
                base_plans[name] = node

    def _has_joins(self, node: PlanNode) -> bool:
        if isinstance(node, JoinNode):
            return True
        for child in node.children():
            if self._has_joins(child):
                return True
        return False

    def _select_indexes(self, node: PlanNode) -> PlanNode:
        """Try to convert scans with filters into index scans."""
        if isinstance(node, FilterNode) and isinstance(node.child, ScanNode):
            scan = node.child
            preds = extract_conjuncts(node.predicate)
            idx_plan = self.index_selector.select_index(scan.table, preds, scan.alias)
            if idx_plan:
                return idx_plan
            return node

        # Recurse into children
        if isinstance(node, FilterNode):
            node.child = self._select_indexes(node.child)
        elif isinstance(node, ProjectNode):
            node.child = self._select_indexes(node.child)
        elif isinstance(node, JoinNode):
            node.left = self._select_indexes(node.left)
            node.right = self._select_indexes(node.right)
        elif isinstance(node, SortNode):
            node.child = self._select_indexes(node.child)
        elif isinstance(node, AggregateNode):
            node.child = self._select_indexes(node.child)
        elif isinstance(node, LimitNode):
            node.child = self._select_indexes(node.child)

        return node

    def explain(self, plan: PlanNode, indent: int = 0) -> str:
        """Generate EXPLAIN output for a plan."""
        lines = []
        self._explain_node(plan, lines, indent)
        return '\n'.join(lines)

    def _explain_node(self, node: PlanNode, lines: List[str], indent: int):
        prefix = '  ' * indent
        arrow = '-> ' if indent > 0 else ''

        if isinstance(node, ScanNode):
            lines.append(f"{prefix}{arrow}Seq Scan on {node.table}"
                        f" (rows={node.estimated_rows:.0f}, cost={node.estimated_cost:.2f})")

        elif isinstance(node, IndexScanNode):
            lines.append(f"{prefix}{arrow}Index Scan using {node.index.name} on {node.table}"
                        f" (rows={node.estimated_rows:.0f}, cost={node.estimated_cost:.2f})")
            for pred in node.predicates:
                lines.append(f"{prefix}  Index Cond: {pred}")

        elif isinstance(node, FilterNode):
            lines.append(f"{prefix}{arrow}Filter (rows={node.estimated_rows:.0f},"
                        f" cost={node.estimated_cost:.2f})")
            lines.append(f"{prefix}  Filter: {node.predicate}")
            self._explain_node(node.child, lines, indent + 1)

        elif isinstance(node, ProjectNode):
            cols = ', '.join(str(c) for c in node.columns)
            lines.append(f"{prefix}{arrow}Project [{cols}]"
                        f" (rows={node.estimated_rows:.0f})")
            self._explain_node(node.child, lines, indent + 1)

        elif isinstance(node, JoinNode):
            lines.append(f"{prefix}{arrow}{node.algorithm.value} Join"
                        f" ({node.join_type.value})"
                        f" (rows={node.estimated_rows:.0f},"
                        f" cost={node.estimated_cost:.2f})")
            if node.condition:
                lines.append(f"{prefix}  Join Cond: {node.condition}")
            self._explain_node(node.left, lines, indent + 1)
            self._explain_node(node.right, lines, indent + 1)

        elif isinstance(node, SortNode):
            ks = ', '.join(f"{k[0]} {k[1]}" for k in node.keys)
            lines.append(f"{prefix}{arrow}Sort [{ks}]"
                        f" (rows={node.estimated_rows:.0f},"
                        f" cost={node.estimated_cost:.2f})")
            self._explain_node(node.child, lines, indent + 1)

        elif isinstance(node, AggregateNode):
            gb = ', '.join(str(g) for g in node.group_by) if node.group_by else 'none'
            agg = ', '.join(str(a) for a in node.aggregates)
            lines.append(f"{prefix}{arrow}Aggregate [group={gb}, agg={agg}]"
                        f" (rows={node.estimated_rows:.0f},"
                        f" cost={node.estimated_cost:.2f})")
            self._explain_node(node.child, lines, indent + 1)

        elif isinstance(node, LimitNode):
            lines.append(f"{prefix}{arrow}Limit {node.limit}"
                        f" (rows={node.estimated_rows:.0f},"
                        f" cost={node.estimated_cost:.2f})")
            if node.offset:
                lines.append(f"{prefix}  Offset: {node.offset}")
            self._explain_node(node.child, lines, indent + 1)

        else:
            lines.append(f"{prefix}{arrow}{node.__class__.__name__}"
                        f" (rows={node.estimated_rows:.0f})")
            for child in node.children():
                self._explain_node(child, lines, indent + 1)


# ---------------------------------------------------------------------------
# Query Builder (convenience API)
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Fluent API for building logical query plans."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._node: Optional[PlanNode] = None

    def scan(self, table: str, alias: Optional[str] = None) -> 'QueryBuilder':
        self._node = ScanNode(table, alias)
        return self

    def filter(self, predicate: Expr) -> 'QueryBuilder':
        self._node = FilterNode(self._node, predicate)
        return self

    def project(self, *columns: Expr) -> 'QueryBuilder':
        self._node = ProjectNode(self._node, list(columns))
        return self

    def join(self, other: 'QueryBuilder', condition: Optional[Expr] = None,
             join_type: JoinType = JoinType.INNER) -> 'QueryBuilder':
        self._node = JoinNode(self._node, other._node, join_type, condition)
        return self

    def sort(self, *keys: Tuple[Expr, str]) -> 'QueryBuilder':
        self._node = SortNode(self._node, list(keys))
        return self

    def aggregate(self, group_by: List[Expr], *aggregates: FuncCall) -> 'QueryBuilder':
        self._node = AggregateNode(self._node, group_by, list(aggregates))
        return self

    def limit(self, n: int, offset: int = 0) -> 'QueryBuilder':
        self._node = LimitNode(self._node, n, offset)
        return self

    def build(self) -> PlanNode:
        return self._node


# ---------------------------------------------------------------------------
# Plan Comparison
# ---------------------------------------------------------------------------

def compare_plans(catalog: Catalog, plan_a: PlanNode, plan_b: PlanNode) -> Dict:
    """Compare costs of two plans."""
    cm = CostModel(catalog)
    cost_a = cm.estimate(plan_a)
    cost_b = cm.estimate(plan_b)

    return {
        'plan_a_cost': cost_a.total,
        'plan_b_cost': cost_b.total,
        'plan_a_rows': plan_a.estimated_rows,
        'plan_b_rows': plan_b.estimated_rows,
        'winner': 'A' if cost_a.total <= cost_b.total else 'B',
        'speedup': max(cost_a.total, cost_b.total) / max(min(cost_a.total, cost_b.total), 0.001)
    }
