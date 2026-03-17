"""
C259: Cost-Based Query Planner
Composes C258 (B-Tree Indexes) + C245 (Query Executor)

Adds cost-based optimization to the SQL database:
- TableStatistics: row count, distinct values, min/max, histograms
- CostModel: I/O + CPU cost estimation for access paths
- CostBasedPlanner: enumerate candidate plans, pick cheapest
- ANALYZE command: collect/refresh statistics
- Join ordering: dynamic programming for optimal join order
- EXPLAIN ANALYZE: shows cost estimates alongside plan
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from collections import defaultdict

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C258_btree_indexes')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C116_bplus_tree')))

from btree_indexes import (
    IndexedDB, IndexManager, IndexInfo, IndexScanDecision, QueryOptimizer
)
from mini_database import (
    MiniDB, ResultSet, StorageEngine, QueryCompiler, Catalog, TableSchema,
    ColumnDef, CatalogError, CompileError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlAggCall, SqlStar, SelectExpr,
    TokenType, Lexer, Parser, parse_sql
)
from query_executor import (
    Database as QEDatabase, Table as QETable, Row, ExecutionEngine,
    Operator, SeqScanOp, FilterOp, ProjectOp, SortOp,
    HashAggregateOp, HavingOp, LimitOp, DistinctOp,
    HashJoinOp, NestedLoopJoinOp, SortMergeJoinOp,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr,
    CompOp, LogicOp, AggFunc, AggCall, IndexScanOp, TableIndex,
    UnionOp, SemiJoinOp, AntiJoinOp, TopNOp
)


# =============================================================================
# Table Statistics
# =============================================================================

@dataclass
class ColumnStats:
    """Statistics for a single column."""
    distinct_count: int = 0
    null_count: int = 0
    min_value: Any = None
    max_value: Any = None
    avg_width: float = 8.0  # estimated bytes
    histogram: List[Any] = field(default_factory=list)  # equi-depth boundaries
    most_common_values: List[Tuple[Any, float]] = field(default_factory=list)  # (value, frequency)

    @property
    def has_stats(self) -> bool:
        return self.distinct_count > 0 or self.null_count > 0


@dataclass
class TableStats:
    """Statistics for a table."""
    table_name: str
    row_count: int = 0
    page_count: int = 0  # estimated pages (row_count / rows_per_page)
    columns: Dict[str, ColumnStats] = field(default_factory=dict)

    ROWS_PER_PAGE = 100  # assumed rows per disk page

    def update_page_count(self):
        self.page_count = max(1, math.ceil(self.row_count / self.ROWS_PER_PAGE))

    def selectivity_eq(self, column: str, value: Any) -> float:
        """Estimate selectivity for col = value."""
        cs = self.columns.get(column)
        if not cs or cs.distinct_count == 0:
            return 0.1  # default
        # Check MCV list first
        for val, freq in cs.most_common_values:
            if val == value:
                return freq
        # Uniform assumption over non-MCV values
        return 1.0 / max(cs.distinct_count, 1)

    def selectivity_range(self, column: str, low: Any = None, high: Any = None,
                          low_inclusive: bool = True, high_inclusive: bool = True) -> float:
        """Estimate selectivity for range condition."""
        cs = self.columns.get(column)
        if not cs or cs.min_value is None or cs.max_value is None:
            return 0.33  # default
        try:
            col_range = float(cs.max_value) - float(cs.min_value)
            if col_range <= 0:
                return 1.0

            if low is not None and high is not None:
                sel = (float(high) - float(low)) / col_range
            elif low is not None:
                sel = (float(cs.max_value) - float(low)) / col_range
            elif high is not None:
                sel = (float(high) - float(cs.min_value)) / col_range
            else:
                return 1.0

            return max(0.0, min(1.0, sel))
        except (TypeError, ValueError):
            return 0.33

    def selectivity_comparison(self, column: str, op: str, value: Any) -> float:
        """Estimate selectivity for a comparison operator."""
        if op in ('=', '=='):
            return self.selectivity_eq(column, value)
        elif op == '<':
            return self.selectivity_range(column, high=value, high_inclusive=False)
        elif op == '<=':
            return self.selectivity_range(column, high=value, high_inclusive=True)
        elif op == '>':
            return self.selectivity_range(column, low=value, low_inclusive=False)
        elif op == '>=':
            return self.selectivity_range(column, low=value, low_inclusive=True)
        elif op in ('!=', '<>'):
            return 1.0 - self.selectivity_eq(column, value)
        return 0.33  # unknown operator

    def selectivity_and(self, sel_a: float, sel_b: float) -> float:
        """Combined selectivity for AND (assumes independence)."""
        return sel_a * sel_b

    def selectivity_or(self, sel_a: float, sel_b: float) -> float:
        """Combined selectivity for OR (assumes independence)."""
        return sel_a + sel_b - sel_a * sel_b

    def estimate_rows(self, selectivity: float) -> float:
        """Estimated rows after applying selectivity."""
        return max(1.0, self.row_count * selectivity)


class StatisticsCollector:
    """Collects table and column statistics from data."""

    HISTOGRAM_BUCKETS = 10
    MCV_COUNT = 10  # most common values to track

    def __init__(self):
        self.stats: Dict[str, TableStats] = {}

    def analyze_table(self, table_name: str, rows: List[Tuple[int, dict]],
                      column_names: List[str]) -> TableStats:
        """Collect statistics for a table."""
        ts = TableStats(table_name=table_name, row_count=len(rows))
        ts.update_page_count()

        for col in column_names:
            cs = self._analyze_column(col, rows)
            ts.columns[col] = cs

        self.stats[table_name] = ts
        return ts

    def _analyze_column(self, column: str, rows: List[Tuple[int, dict]]) -> ColumnStats:
        """Collect statistics for a single column."""
        cs = ColumnStats()
        values = []
        value_counts = defaultdict(int)

        for _rowid, row_data in rows:
            v = row_data.get(column)
            if v is None:
                cs.null_count += 1
            else:
                values.append(v)
                value_counts[v] += 1

        if not values:
            return cs

        cs.distinct_count = len(value_counts)

        # Min/max (handle mixed types gracefully)
        try:
            sorted_vals = sorted(values)
            cs.min_value = sorted_vals[0]
            cs.max_value = sorted_vals[-1]
        except TypeError:
            # Mixed types -- use string comparison
            str_vals = sorted(str(v) for v in values)
            cs.min_value = values[0]
            cs.max_value = values[-1]

        # Average width
        total_width = sum(len(str(v)) for v in values)
        cs.avg_width = total_width / len(values) if values else 8.0

        # Most common values
        total = len(rows)
        mcv = sorted(value_counts.items(), key=lambda x: -x[1])[:self.MCV_COUNT]
        cs.most_common_values = [(v, count / total) for v, count in mcv]

        # Equi-depth histogram
        if len(sorted_vals) > self.HISTOGRAM_BUCKETS:
            try:
                sorted_vals_clean = sorted(values)
                step = len(sorted_vals_clean) // self.HISTOGRAM_BUCKETS
                cs.histogram = [sorted_vals_clean[i * step] for i in range(1, self.HISTOGRAM_BUCKETS)]
            except TypeError:
                pass

        return cs

    def get_stats(self, table_name: str) -> Optional[TableStats]:
        return self.stats.get(table_name)

    def has_stats(self, table_name: str) -> bool:
        return table_name in self.stats

    def invalidate(self, table_name: str):
        """Remove stats for a table (data changed significantly)."""
        self.stats.pop(table_name, None)

    def all_stats(self) -> Dict[str, TableStats]:
        return dict(self.stats)


# =============================================================================
# Cost Model
# =============================================================================

@dataclass
class CostEstimate:
    """Cost estimate for a plan node."""
    startup_cost: float = 0.0   # cost before first row
    total_cost: float = 0.0     # cost to produce all rows
    output_rows: float = 1.0    # estimated output cardinality
    output_width: float = 8.0   # estimated avg row width in bytes

    @property
    def per_row_cost(self) -> float:
        if self.output_rows <= 0:
            return 0.0
        return self.total_cost / self.output_rows

    def __repr__(self):
        return (f"Cost(startup={self.startup_cost:.2f}, total={self.total_cost:.2f}, "
                f"rows={self.output_rows:.0f}, width={self.output_width:.1f})")


class CostModel:
    """Estimates costs for different physical operators."""

    # Cost constants (relative units)
    SEQ_PAGE_COST = 1.0         # sequential I/O per page
    RANDOM_PAGE_COST = 4.0      # random I/O per page
    CPU_TUPLE_COST = 0.01       # CPU cost per row
    CPU_INDEX_TUPLE_COST = 0.005  # CPU cost per index entry
    CPU_OPERATOR_COST = 0.0025  # CPU cost per operator evaluation
    HASH_BUILD_COST = 0.02      # cost to hash a tuple
    SORT_COST_FACTOR = 2.0      # cost multiplier for sort (n log n)

    def __init__(self, stats_collector: StatisticsCollector):
        self.stats = stats_collector

    def seq_scan_cost(self, table_name: str, filter_selectivity: float = 1.0) -> CostEstimate:
        """Cost of a full sequential scan."""
        ts = self.stats.get_stats(table_name)
        if not ts:
            return CostEstimate(total_cost=100.0, output_rows=100.0)

        io_cost = ts.page_count * self.SEQ_PAGE_COST
        cpu_cost = ts.row_count * self.CPU_TUPLE_COST
        if filter_selectivity < 1.0:
            cpu_cost += ts.row_count * self.CPU_OPERATOR_COST  # filter eval

        output_rows = max(1.0, ts.row_count * filter_selectivity)
        return CostEstimate(
            startup_cost=0.0,
            total_cost=io_cost + cpu_cost,
            output_rows=output_rows,
        )

    def index_scan_cost(self, table_name: str, index_info: IndexInfo,
                        selectivity: float) -> CostEstimate:
        """Cost of an index scan (B+ tree lookup + row fetches)."""
        ts = self.stats.get_stats(table_name)
        if not ts:
            return CostEstimate(total_cost=10.0, output_rows=10.0)

        output_rows = max(1.0, ts.row_count * selectivity)

        # Index traversal cost (B+ tree height)
        height = max(1, int(math.log(max(1, ts.row_count), 32)))  # order=32
        index_io = height * self.RANDOM_PAGE_COST
        index_cpu = height * self.CPU_INDEX_TUPLE_COST

        # Row fetch cost (random I/O for each matching row)
        # Clustered: sequential; unclustered: random
        fetch_pages = min(output_rows / TableStats.ROWS_PER_PAGE, ts.page_count)
        row_io = fetch_pages * self.RANDOM_PAGE_COST
        row_cpu = output_rows * self.CPU_TUPLE_COST

        return CostEstimate(
            startup_cost=index_io + index_cpu,
            total_cost=index_io + index_cpu + row_io + row_cpu,
            output_rows=output_rows,
        )

    def nested_loop_join_cost(self, outer: CostEstimate, inner: CostEstimate,
                              selectivity: float = 1.0) -> CostEstimate:
        """Cost of nested loop join."""
        startup = outer.startup_cost
        total = outer.total_cost + outer.output_rows * inner.total_cost
        cpu = outer.output_rows * inner.output_rows * self.CPU_OPERATOR_COST
        output_rows = max(1.0, outer.output_rows * inner.output_rows * selectivity)
        return CostEstimate(startup_cost=startup, total_cost=total + cpu, output_rows=output_rows)

    def hash_join_cost(self, outer: CostEstimate, inner: CostEstimate,
                       selectivity: float = 1.0) -> CostEstimate:
        """Cost of hash join (build on inner, probe with outer)."""
        build_cost = inner.total_cost + inner.output_rows * self.HASH_BUILD_COST
        probe_cost = outer.total_cost + outer.output_rows * self.CPU_INDEX_TUPLE_COST
        output_rows = max(1.0, outer.output_rows * inner.output_rows * selectivity)
        return CostEstimate(
            startup_cost=inner.total_cost + inner.output_rows * self.HASH_BUILD_COST,
            total_cost=build_cost + probe_cost,
            output_rows=output_rows,
        )

    def sort_merge_join_cost(self, left: CostEstimate, right: CostEstimate,
                             selectivity: float = 1.0) -> CostEstimate:
        """Cost of sort-merge join."""
        sort_left = self._sort_cost(left)
        sort_right = self._sort_cost(right)
        merge_cost = (left.output_rows + right.output_rows) * self.CPU_TUPLE_COST
        output_rows = max(1.0, left.output_rows * right.output_rows * selectivity)
        total = sort_left + sort_right + merge_cost
        return CostEstimate(
            startup_cost=sort_left + sort_right,
            total_cost=total,
            output_rows=output_rows,
        )

    def sort_cost(self, input_est: CostEstimate) -> CostEstimate:
        """Cost of sorting."""
        sort = self._sort_cost(input_est)
        return CostEstimate(
            startup_cost=input_est.total_cost + sort,
            total_cost=input_est.total_cost + sort,
            output_rows=input_est.output_rows,
            output_width=input_est.output_width,
        )

    def aggregate_cost(self, input_est: CostEstimate, num_groups: float) -> CostEstimate:
        """Cost of hash aggregation."""
        hash_cost = input_est.output_rows * self.HASH_BUILD_COST
        return CostEstimate(
            startup_cost=input_est.total_cost + hash_cost,
            total_cost=input_est.total_cost + hash_cost,
            output_rows=max(1.0, num_groups),
        )

    def _sort_cost(self, est: CostEstimate) -> float:
        n = max(1.0, est.output_rows)
        return n * math.log2(max(2, n)) * self.CPU_OPERATOR_COST * self.SORT_COST_FACTOR


# =============================================================================
# Plan Candidates
# =============================================================================

@dataclass
class PlanCandidate:
    """A candidate physical plan with cost estimate."""
    plan_type: str              # 'seq_scan', 'index_scan', 'hash_join', etc.
    cost: CostEstimate
    details: Dict[str, Any] = field(default_factory=dict)
    children: List['PlanCandidate'] = field(default_factory=list)

    def explain_tree(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}{self.plan_type} (cost={self.cost.total_cost:.2f}, rows={self.cost.output_rows:.0f})"]
        for k, v in self.details.items():
            lines.append(f"{prefix}  {k}: {v}")
        for child in self.children:
            lines.append(child.explain_tree(indent + 1))
        return "\n".join(lines)


# =============================================================================
# Cost-Based Planner
# =============================================================================

class CostBasedPlanner:
    """Generates candidate plans and picks the cheapest."""

    def __init__(self, index_manager: IndexManager, cost_model: CostModel,
                 stats_collector: StatisticsCollector):
        self.index_manager = index_manager
        self.cost_model = cost_model
        self.stats = stats_collector

    def plan_scan(self, table_name: str, where_node=None) -> PlanCandidate:
        """Choose best access path for a single table scan."""
        candidates = []

        # Option 1: Sequential scan
        selectivity = self._estimate_where_selectivity(table_name, where_node)
        seq_cost = self.cost_model.seq_scan_cost(table_name, selectivity)
        candidates.append(PlanCandidate(
            plan_type='seq_scan',
            cost=seq_cost,
            details={'table': table_name, 'filter_selectivity': round(selectivity, 4)},
        ))

        # Option 2+: Index scans (one per applicable index)
        if where_node is not None:
            idx_plans = self._enumerate_index_plans(table_name, where_node)
            candidates.extend(idx_plans)

        # Pick cheapest
        candidates.sort(key=lambda c: c.cost.total_cost)
        return candidates[0]

    def plan_join(self, left_table: str, right_table: str,
                  join_condition=None, join_type: str = 'inner') -> PlanCandidate:
        """Choose best join strategy between two tables."""
        left_stats = self.stats.get_stats(left_table)
        right_stats = self.stats.get_stats(right_table)

        # Estimate join selectivity
        join_sel = self._estimate_join_selectivity(
            left_table, right_table, join_condition
        )

        # Get base scan costs
        left_scan = self.cost_model.seq_scan_cost(left_table)
        right_scan = self.cost_model.seq_scan_cost(right_table)

        candidates = []

        # Nested loop: smaller table as outer
        if left_scan.output_rows <= right_scan.output_rows:
            nl_cost = self.cost_model.nested_loop_join_cost(left_scan, right_scan, join_sel)
            candidates.append(PlanCandidate(
                plan_type='nested_loop_join',
                cost=nl_cost,
                details={'outer': left_table, 'inner': right_table,
                         'join_type': join_type, 'selectivity': round(join_sel, 4)},
            ))
        else:
            nl_cost = self.cost_model.nested_loop_join_cost(right_scan, left_scan, join_sel)
            candidates.append(PlanCandidate(
                plan_type='nested_loop_join',
                cost=nl_cost,
                details={'outer': right_table, 'inner': left_table,
                         'join_type': join_type, 'selectivity': round(join_sel, 4)},
            ))

        # Hash join: build on smaller table
        if left_scan.output_rows <= right_scan.output_rows:
            hj_cost = self.cost_model.hash_join_cost(right_scan, left_scan, join_sel)
            candidates.append(PlanCandidate(
                plan_type='hash_join',
                cost=hj_cost,
                details={'build': left_table, 'probe': right_table,
                         'join_type': join_type},
            ))
        else:
            hj_cost = self.cost_model.hash_join_cost(left_scan, right_scan, join_sel)
            candidates.append(PlanCandidate(
                plan_type='hash_join',
                cost=hj_cost,
                details={'build': right_table, 'probe': left_table,
                         'join_type': join_type},
            ))

        # Sort-merge join
        sm_cost = self.cost_model.sort_merge_join_cost(left_scan, right_scan, join_sel)
        candidates.append(PlanCandidate(
            plan_type='sort_merge_join',
            cost=sm_cost,
            details={'left': left_table, 'right': right_table,
                     'join_type': join_type},
        ))

        candidates.sort(key=lambda c: c.cost.total_cost)
        return candidates[0]

    def plan_multi_join(self, tables: List[str], join_conditions: List[dict]) -> PlanCandidate:
        """Optimal join ordering using dynamic programming (for up to ~8 tables).

        join_conditions: list of {'left_table', 'right_table', 'condition'}
        """
        n = len(tables)
        if n == 0:
            raise ValueError("No tables to join")
        if n == 1:
            return self.plan_scan(tables[0])

        # Build adjacency of join conditions
        cond_map = {}
        for jc in join_conditions:
            lt = jc['left_table']
            rt = jc['right_table']
            key = tuple(sorted([lt, rt]))
            cond_map[key] = jc.get('condition')

        # DP over subsets (bitmask)
        # dp[mask] = best PlanCandidate for joining the subset of tables
        dp: Dict[int, PlanCandidate] = {}

        # Base cases: single tables
        for i, t in enumerate(tables):
            mask = 1 << i
            scan = self.plan_scan(t)
            dp[mask] = scan

        # Build up larger subsets
        full_mask = (1 << n) - 1
        for size in range(2, n + 1):
            for mask in range(1, full_mask + 1):
                if bin(mask).count('1') != size:
                    continue

                best = None
                # Try all ways to split mask into two non-empty subsets
                sub = (mask - 1) & mask
                while sub > 0:
                    complement = mask ^ sub
                    if sub < complement:  # avoid duplicates
                        if sub in dp and complement in dp:
                            left_plan = dp[sub]
                            right_plan = dp[complement]

                            # Check if there's a join condition between the two subsets
                            join_sel = self._cross_subset_selectivity(
                                tables, sub, complement, cond_map
                            )

                            # Try join strategies
                            for join_type, cost_fn in [
                                ('hash_join', self.cost_model.hash_join_cost),
                                ('nested_loop_join', self.cost_model.nested_loop_join_cost),
                            ]:
                                if left_plan.cost.output_rows <= right_plan.cost.output_rows:
                                    jcost = cost_fn(right_plan.cost, left_plan.cost, join_sel)
                                else:
                                    jcost = cost_fn(left_plan.cost, right_plan.cost, join_sel)

                                candidate = PlanCandidate(
                                    plan_type=join_type,
                                    cost=jcost,
                                    details={'left_mask': sub, 'right_mask': complement,
                                             'selectivity': round(join_sel, 4)},
                                    children=[left_plan, right_plan],
                                )

                                if best is None or jcost.total_cost < best.cost.total_cost:
                                    best = candidate

                    sub = (sub - 1) & mask

                if best is not None:
                    dp[mask] = best

        return dp.get(full_mask, self.plan_scan(tables[0]))

    def _enumerate_index_plans(self, table_name: str, where_node) -> List[PlanCandidate]:
        """Enumerate all index scan options for a WHERE clause."""
        plans = []
        indexes = self.index_manager.get_indexes_for_table(table_name)

        for idx in indexes:
            sel = self._index_selectivity(table_name, idx, where_node)
            if sel is not None:
                cost = self.cost_model.index_scan_cost(table_name, idx, sel)
                plans.append(PlanCandidate(
                    plan_type='index_scan',
                    cost=cost,
                    details={'table': table_name, 'index': idx.name,
                             'columns': idx.columns, 'selectivity': round(sel, 4)},
                ))

        return plans

    def _index_selectivity(self, table_name: str, idx: IndexInfo,
                           where_node) -> Optional[float]:
        """Check if where_node can use this index, return selectivity or None."""
        ts = self.stats.get_stats(table_name)
        if not ts:
            return None

        col = idx.columns[0]  # primary index column

        if isinstance(where_node, SqlComparison):
            col_name, val = self._extract_col_literal(where_node)
            if col_name == col:
                return ts.selectivity_comparison(col_name, where_node.op, val)

        if isinstance(where_node, SqlLogic) and where_node.op == 'and':
            # Check if any operand uses this index
            for operand in where_node.operands:
                sel = self._index_selectivity(table_name, idx, operand)
                if sel is not None:
                    return sel

        return None

    def _estimate_where_selectivity(self, table_name: str, where_node) -> float:
        """Estimate overall selectivity of a WHERE clause."""
        if where_node is None:
            return 1.0

        ts = self.stats.get_stats(table_name)
        if not ts:
            return 0.33  # default when no stats

        if isinstance(where_node, SqlComparison):
            col_name, val = self._extract_col_literal(where_node)
            if col_name and val is not None:
                return ts.selectivity_comparison(col_name, where_node.op, val)
            return 0.33

        if isinstance(where_node, SqlLogic):
            if where_node.op == 'and' and len(where_node.operands) >= 2:
                sel = 1.0
                for op in where_node.operands:
                    sel = ts.selectivity_and(sel, self._estimate_where_selectivity(table_name, op))
                return sel
            if where_node.op == 'or' and len(where_node.operands) >= 2:
                sel = 0.0
                for op in where_node.operands:
                    sub_sel = self._estimate_where_selectivity(table_name, op)
                    sel = ts.selectivity_or(sel, sub_sel)
                return sel

        return 0.33

    def _estimate_join_selectivity(self, left_table: str, right_table: str,
                                    condition) -> float:
        """Estimate selectivity for a join condition."""
        if condition is None:
            return 1.0  # cross join

        # For equi-join on col: selectivity = 1/max(ndistinct_left, ndistinct_right)
        if isinstance(condition, SqlComparison) and condition.op in ('=', '=='):
            left_col = self._get_column_from_ref(condition.left)
            right_col = self._get_column_from_ref(condition.right)

            if left_col and right_col:
                left_ts = self.stats.get_stats(left_table)
                right_ts = self.stats.get_stats(right_table)
                if left_ts and right_ts:
                    left_nd = left_ts.columns.get(left_col, ColumnStats()).distinct_count
                    right_nd = right_ts.columns.get(right_col, ColumnStats()).distinct_count
                    max_nd = max(left_nd, right_nd, 1)
                    return 1.0 / max_nd

        return 0.1  # default for complex conditions

    def _cross_subset_selectivity(self, tables: List[str], left_mask: int,
                                   right_mask: int, cond_map: dict) -> float:
        """Estimate selectivity between two table subsets."""
        n = len(tables)
        sel = 1.0
        has_condition = False

        for i in range(n):
            if not (left_mask & (1 << i)):
                continue
            for j in range(n):
                if not (right_mask & (1 << j)):
                    continue
                key = tuple(sorted([tables[i], tables[j]]))
                if key in cond_map:
                    has_condition = True
                    js = self._estimate_join_selectivity(
                        tables[i], tables[j], cond_map[key]
                    )
                    sel *= js

        return sel if has_condition else 1.0  # cross join if no condition

    def _extract_col_literal(self, node: SqlComparison) -> Tuple[Optional[str], Any]:
        """Extract (column_name, literal_value) from comparison."""
        col = self._get_column_from_ref(node.left)
        val = self._get_literal_value(node.right)
        if col and val is not None:
            return col, val
        col = self._get_column_from_ref(node.right)
        val = self._get_literal_value(node.left)
        if col and val is not None:
            return col, val
        return None, None

    def _get_column_from_ref(self, node) -> Optional[str]:
        if isinstance(node, SqlColumnRef):
            return node.column
        return None

    def _get_literal_value(self, node) -> Any:
        if isinstance(node, SqlLiteral):
            return node.value
        return None


# =============================================================================
# Cost-Based Indexed Database
# =============================================================================

class CostBasedDB(IndexedDB):
    """IndexedDB extended with cost-based query planning."""

    def __init__(self, pool_size: int = 64):
        super().__init__(pool_size=pool_size)
        self.stats_collector = StatisticsCollector()
        self.cost_model = CostModel(self.stats_collector)
        self.planner = CostBasedPlanner(
            self.index_manager, self.cost_model, self.stats_collector
        )
        self._auto_analyze = True  # auto-collect stats on first query
        self._plan_cache: Dict[str, PlanCandidate] = {}
        self._last_plan: Optional[PlanCandidate] = None

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with ANALYZE and cost-based planning."""
        sql_stripped = sql.strip()

        # Handle ANALYZE
        upper = sql_stripped.upper()
        if upper.startswith('ANALYZE'):
            return self._exec_analyze(sql_stripped)

        # Handle EXPLAIN ANALYZE
        if upper.startswith('EXPLAIN ANALYZE'):
            return self._exec_explain_analyze(sql_stripped)

        return super().execute(sql)

    def _exec_analyze(self, sql: str) -> ResultSet:
        """ANALYZE [table_name] -- collect statistics."""
        parts = sql.split()
        if len(parts) >= 2:
            table_name = parts[1].rstrip(';')
            return self._analyze_table(table_name)
        else:
            # Analyze all tables
            results = []
            for tname in list(self.storage.catalog.tables.keys()):
                r = self._analyze_table(tname)
                results.append(r.message)
            return ResultSet(columns=[], rows=[],
                             message=f"ANALYZE {len(results)} tables")

    def _analyze_table(self, table_name: str) -> ResultSet:
        """Collect stats for a single table."""
        schema = self.storage.catalog.get_table(table_name)
        txn_id = self._get_txn()
        try:
            rows = list(self.storage.scan_table(txn_id, table_name))
            ts = self.stats_collector.analyze_table(
                table_name, rows, schema.column_names()
            )
            self._auto_commit(txn_id)
            return ResultSet(
                columns=[], rows=[],
                message=f"ANALYZE {table_name}: {ts.row_count} rows, {len(ts.columns)} columns"
            )
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _ensure_stats(self, table_name: str):
        """Auto-analyze if no stats exist yet."""
        if self._auto_analyze and not self.stats_collector.has_stats(table_name):
            self._analyze_table(table_name)

    def _exec_select(self, stmt: SelectStmt) -> ResultSet:
        """Select with cost-based optimization."""
        table_name = stmt.from_table.table_name if stmt.from_table else None

        if table_name:
            self._ensure_stats(table_name)

        # Also ensure stats for joined tables
        if stmt.joins:
            for j in stmt.joins:
                self._ensure_stats(j.table.table_name)

        # Use cost-based planner to decide access path
        if table_name and not stmt.joins:
            plan = self.planner.plan_scan(table_name, stmt.where)
            self._last_plan = plan

            if plan.plan_type == 'index_scan':
                # Use index scan via parent
                idx_name = plan.details.get('index')
                idx_info = self.index_manager.get_index(idx_name)
                if idx_info:
                    decision = self._build_index_decision(idx_info, stmt.where)
                    if decision.use_index:
                        self._index_scan_count += 1
                        txn_id = self._get_txn()
                        try:
                            result = self._exec_select_with_index(stmt, txn_id, decision)
                            self._auto_commit(txn_id)
                            return result
                        except Exception:
                            self._auto_abort(txn_id)
                            raise

            # Fall through to seq scan
            self._seq_scan_count += 1
            txn_id = self._get_txn()
            try:
                result = self._exec_select_seq(stmt, txn_id)
                self._auto_commit(txn_id)
                return result
            except Exception:
                self._auto_abort(txn_id)
                raise

        elif table_name and stmt.joins:
            # Multi-table: use cost-based join planning
            tables = [table_name]
            join_conditions = []
            for j in stmt.joins:
                tables.append(j.table.table_name)
                if j.condition:
                    join_conditions.append({
                        'left_table': table_name,
                        'right_table': j.table.table_name,
                        'condition': j.condition,
                    })

            plan = self.planner.plan_multi_join(tables, join_conditions)
            self._last_plan = plan

            # Execute using parent's path (join ordering is advisory for now)
            txn_id = self._get_txn()
            try:
                result = self._exec_select_seq(stmt, txn_id)
                self._auto_commit(txn_id)
                return result
            except Exception:
                self._auto_abort(txn_id)
                raise

        else:
            # No FROM clause
            txn_id = self._get_txn()
            try:
                result = self._exec_select_seq(stmt, txn_id)
                self._auto_commit(txn_id)
                return result
            except Exception:
                self._auto_abort(txn_id)
                raise

    def _build_index_decision(self, idx_info: IndexInfo,
                               where_node) -> IndexScanDecision:
        """Build an IndexScanDecision from cost-based plan selection."""
        # Delegate to the rule-based optimizer for the actual decision details
        decision = self.optimizer.analyze_where(idx_info.table_name, where_node)
        if decision.use_index and decision.index_info and decision.index_info.name == idx_info.name:
            return decision
        # If the cost-based planner chose this index but the rule optimizer chose different,
        # force-select the cost-planner's choice
        decision2 = self._try_index_decision(idx_info, where_node)
        return decision2

    def _try_index_decision(self, idx_info: IndexInfo,
                            where_node) -> IndexScanDecision:
        """Try to build an IndexScanDecision for a specific index."""
        col = idx_info.columns[0]

        if isinstance(where_node, SqlComparison):
            return self._try_comparison_for_index(idx_info, col, where_node)

        if isinstance(where_node, SqlLogic) and where_node.op == 'and':
            remaining = []
            for operand in where_node.operands:
                dec = self._try_comparison_for_index(idx_info, col, operand)
                if dec.use_index:
                    # Remaining filter is the other operands
                    others = [op for op in where_node.operands if op is not operand]
                    if len(others) == 1:
                        dec.remaining_filter = others[0]
                    elif len(others) > 1:
                        dec.remaining_filter = SqlLogic(op='and', operands=others)
                    return dec
                remaining.append(operand)

        return IndexScanDecision()

    def _try_comparison_for_index(self, idx_info: IndexInfo, col: str,
                                   node) -> IndexScanDecision:
        """Try to match a comparison node to an index."""
        if not isinstance(node, SqlComparison):
            return IndexScanDecision()

        col_name = None
        val = None

        if isinstance(node.left, SqlColumnRef) and node.left.column == col:
            col_name = col
            val = node.right.value if isinstance(node.right, SqlLiteral) else None
        elif isinstance(node.right, SqlColumnRef) and node.right.column == col:
            col_name = col
            val = node.left.value if isinstance(node.left, SqlLiteral) else None

        if col_name is None or val is None:
            return IndexScanDecision()

        if node.op in ('=', '=='):
            return IndexScanDecision(
                use_index=True, index_info=idx_info,
                scan_type='eq', lookup_value=val
            )
        elif node.op in ('<', '<='):
            return IndexScanDecision(
                use_index=True, index_info=idx_info,
                scan_type='range', high=val, high_inclusive=(node.op == '<=')
            )
        elif node.op in ('>', '>='):
            return IndexScanDecision(
                use_index=True, index_info=idx_info,
                scan_type='range', low=val, low_inclusive=(node.op == '>=')
            )

        return IndexScanDecision()

    def _exec_explain(self, stmt) -> ResultSet:
        """Enhanced EXPLAIN that shows cost estimates."""
        inner = stmt.stmt if hasattr(stmt, 'stmt') else stmt
        if not isinstance(inner, SelectStmt):
            return ResultSet(columns=['plan'], rows=[['EXPLAIN not supported']])

        table_name = inner.from_table.table_name if inner.from_table else None

        if table_name:
            self._ensure_stats(table_name)
            if inner.joins:
                for j in inner.joins:
                    self._ensure_stats(j.table.table_name)

        # Get cost-based plan
        lines = []
        if table_name and not inner.joins:
            plan = self.planner.plan_scan(table_name, inner.where)
            self._last_plan = plan
            lines.append(f"Plan: {plan.plan_type}")
            lines.append(f"  Estimated cost: {plan.cost.total_cost:.2f}")
            lines.append(f"  Estimated rows: {plan.cost.output_rows:.0f}")
            for k, v in plan.details.items():
                lines.append(f"  {k}: {v}")

            # Show alternatives
            selectivity = self.planner._estimate_where_selectivity(table_name, inner.where)
            seq_cost = self.cost_model.seq_scan_cost(table_name, selectivity)
            idx_plans = self.planner._enumerate_index_plans(table_name, inner.where) if inner.where else []

            if idx_plans or plan.plan_type == 'seq_scan':
                lines.append("")
                lines.append("Alternatives considered:")
                lines.append(f"  seq_scan: cost={seq_cost.total_cost:.2f}, rows={seq_cost.output_rows:.0f}")
                for ip in idx_plans:
                    lines.append(
                        f"  index_scan({ip.details.get('index', '?')}): "
                        f"cost={ip.cost.total_cost:.2f}, rows={ip.cost.output_rows:.0f}"
                    )

        elif table_name and inner.joins:
            tables = [table_name] + [j.table.table_name for j in inner.joins]
            join_conditions = []
            for j in inner.joins:
                if j.condition:
                    join_conditions.append({
                        'left_table': table_name,
                        'right_table': j.table.table_name,
                        'condition': j.condition,
                    })
            plan = self.planner.plan_multi_join(tables, join_conditions)
            self._last_plan = plan
            lines.append(plan.explain_tree())

        else:
            lines.append("No tables to plan")

        # Also show traditional plan
        lines.append("")
        parent_result = super()._exec_explain(stmt)
        for row in parent_result.rows:
            lines.append(row[0] if row else '')

        return ResultSet(columns=['plan'],
                         rows=[[line] for line in lines if line.strip()])

    def _exec_explain_analyze(self, sql: str) -> ResultSet:
        """EXPLAIN ANALYZE: run the query and show actual vs estimated rows."""
        # Extract the inner SQL
        inner_sql = sql.strip()
        for prefix in ['EXPLAIN ANALYZE ', 'explain analyze ']:
            if inner_sql.upper().startswith('EXPLAIN ANALYZE'):
                inner_sql = inner_sql[len('EXPLAIN ANALYZE'):].strip()
                break

        # First get the plan
        table_name = None
        try:
            stmt = parse_sql(inner_sql)
            if isinstance(stmt, SelectStmt):
                select_stmt = stmt
            elif isinstance(stmt, list) and stmt and isinstance(stmt[0], SelectStmt):
                select_stmt = stmt[0]
            else:
                select_stmt = None

            if select_stmt:
                table_name = select_stmt.from_table.table_name if select_stmt.from_table else None
                if table_name:
                    self._ensure_stats(table_name)
                plan = self.planner.plan_scan(table_name, select_stmt.where) if table_name else None
            else:
                plan = None
        except Exception:
            plan = None

        # Execute the query
        result = self.execute(inner_sql)
        actual_rows = len(result.rows)

        lines = []
        if plan:
            lines.append(f"Plan: {plan.plan_type}")
            lines.append(f"  Estimated cost: {plan.cost.total_cost:.2f}")
            lines.append(f"  Estimated rows: {plan.cost.output_rows:.0f}")
            lines.append(f"  Actual rows: {actual_rows}")
            accuracy = 0.0
            if plan.cost.output_rows > 0:
                accuracy = min(actual_rows, plan.cost.output_rows) / max(actual_rows, plan.cost.output_rows) * 100
            lines.append(f"  Estimation accuracy: {accuracy:.1f}%")
            for k, v in plan.details.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"Actual rows: {actual_rows}")

        return ResultSet(columns=['plan'],
                         rows=[[line] for line in lines if line.strip()])

    def _exec_insert(self, stmt: InsertStmt) -> ResultSet:
        """Insert with stats invalidation."""
        result = super()._exec_insert(stmt)
        # Invalidate stats for the affected table
        self.stats_collector.invalidate(stmt.table_name)
        return result

    def _exec_update(self, stmt: UpdateStmt) -> ResultSet:
        """Update with stats invalidation."""
        result = super()._exec_update(stmt)
        self.stats_collector.invalidate(stmt.table_name)
        return result

    def _exec_delete(self, stmt: DeleteStmt) -> ResultSet:
        """Delete with stats invalidation."""
        result = super()._exec_delete(stmt)
        self.stats_collector.invalidate(stmt.table_name)
        return result

    def get_table_stats(self, table_name: str) -> Optional[TableStats]:
        """Get collected statistics for a table."""
        return self.stats_collector.get_stats(table_name)

    def get_last_plan(self) -> Optional[PlanCandidate]:
        """Get the plan chosen for the last SELECT."""
        return self._last_plan

    def get_cost_breakdown(self, table_name: str) -> Dict[str, Any]:
        """Get detailed cost breakdown for a table."""
        ts = self.stats_collector.get_stats(table_name)
        if not ts:
            return {'error': f'No stats for {table_name}. Run ANALYZE first.'}

        result = {
            'table': table_name,
            'row_count': ts.row_count,
            'page_count': ts.page_count,
            'seq_scan_cost': self.cost_model.seq_scan_cost(table_name).total_cost,
            'columns': {},
        }

        for col_name, cs in ts.columns.items():
            result['columns'][col_name] = {
                'distinct': cs.distinct_count,
                'nulls': cs.null_count,
                'min': cs.min_value,
                'max': cs.max_value,
                'avg_width': round(cs.avg_width, 1),
                'mcv': cs.most_common_values[:5],
            }

        return result
