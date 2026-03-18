"""
C264: Window Functions
Extends C263 (CTAS) with SQL window functions.

Features:
- OVER (PARTITION BY ... ORDER BY ...)
- Ranking: ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE(n)
- Navigation: LAG(col, offset, default), LEAD(col, offset, default)
- Value: FIRST_VALUE(col), LAST_VALUE(col), NTH_VALUE(col, n)
- Aggregate windows: SUM/AVG/COUNT/MIN/MAX(...) OVER (...)
- Frame specs: ROWS BETWEEN ... AND ...
  - UNBOUNDED PRECEDING, N PRECEDING, CURRENT ROW, N FOLLOWING, UNBOUNDED FOLLOWING
- Named windows: WINDOW w AS (PARTITION BY ...) ... OVER w
- Multiple window functions in one query
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C263_ctas')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from ctas import (
    CTASDB, CTASParser, CreateTableAsSelectStmt,
    parse_ctas_sql, parse_ctas_sql_multi,
    _infer_type, _infer_column_type,
)
from views import (
    ViewDB, ViewDef, ViewParser, ViewRegistry,
    CreateViewStmt, DropViewStmt,
)
from mini_database import (
    MiniDB, ResultSet, StorageEngine, QueryCompiler, Catalog, TableSchema,
    ColumnDef, CatalogError, CompileError, ParseError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlIsNull, SqlBetween, SqlInList, SqlFuncCall, SqlCase, SqlStar,
    TokenType, Token, Lexer, Parser, KEYWORDS,
    SelectExpr, TableRef,
)
from query_executor import Row, eval_expr


# =============================================================================
# AST nodes for window functions
# =============================================================================

@dataclass
class FrameBound:
    """A window frame boundary."""
    bound_type: str  # 'unbounded_preceding', 'preceding', 'current_row', 'following', 'unbounded_following'
    offset: Optional[int] = None  # for N PRECEDING / N FOLLOWING


@dataclass
class WindowSpec:
    """Window specification: PARTITION BY, ORDER BY, frame."""
    partition_by: Optional[List[Any]] = None
    order_by: Optional[List[Tuple[Any, bool]]] = None  # [(expr, asc), ...]
    frame_start: Optional[FrameBound] = None
    frame_end: Optional[FrameBound] = None
    base_window: Optional[str] = None  # reference to named window


@dataclass
class SqlWindowFunc:
    """A window function call: func(...) OVER (...)"""
    func_name: str  # ROW_NUMBER, RANK, etc. or SUM, AVG, etc.
    args: List[Any] = field(default_factory=list)
    window: Optional[WindowSpec] = None
    window_name: Optional[str] = None  # reference to named window
    distinct: bool = False


@dataclass
class NamedWindow:
    """WINDOW name AS (spec)"""
    name: str
    spec: WindowSpec


# =============================================================================
# Default frame for different function types
# =============================================================================

RANKING_FUNCS = {'row_number', 'rank', 'dense_rank', 'ntile', 'percent_rank', 'cume_dist'}
NAVIGATION_FUNCS = {'lag', 'lead', 'first_value', 'last_value', 'nth_value'}
AGGREGATE_FUNCS = {'sum', 'avg', 'count', 'min', 'max'}


def _default_frame(func_name: str, has_order_by: bool) -> Tuple[FrameBound, FrameBound]:
    """Return default frame bounds for a window function."""
    fn = func_name.lower()
    if fn in RANKING_FUNCS or fn in ('lag', 'lead'):
        # Ranking/navigation operate on the whole partition
        return (
            FrameBound('unbounded_preceding'),
            FrameBound('unbounded_following'),
        )
    # Aggregate windows: with ORDER BY default is RANGE UNBOUNDED PRECEDING TO CURRENT ROW
    # without ORDER BY default is the whole partition
    if has_order_by:
        return (
            FrameBound('unbounded_preceding'),
            FrameBound('current_row'),
        )
    return (
        FrameBound('unbounded_preceding'),
        FrameBound('unbounded_following'),
    )


# =============================================================================
# Frame index computation
# =============================================================================

def _frame_indices(frame_start: FrameBound, frame_end: FrameBound,
                   current_idx: int, partition_size: int) -> Tuple[int, int]:
    """Return (start, end) indices for the frame (inclusive)."""
    def _resolve(bound: FrameBound, current: int) -> int:
        if bound.bound_type == 'unbounded_preceding':
            return 0
        elif bound.bound_type == 'preceding':
            return max(0, current - (bound.offset or 0))
        elif bound.bound_type == 'current_row':
            return current
        elif bound.bound_type == 'following':
            return min(partition_size - 1, current + (bound.offset or 0))
        elif bound.bound_type == 'unbounded_following':
            return partition_size - 1
        return current

    start = _resolve(frame_start, current_idx)
    end = _resolve(frame_end, current_idx)
    return (start, end)


# =============================================================================
# Window function evaluator
# =============================================================================

def _evaluate_window_func(func: SqlWindowFunc, partition_rows: List[Row],
                          current_idx: int, frame_start: FrameBound,
                          frame_end: FrameBound, rank_info: Dict) -> Any:
    """Evaluate a single window function for a single row."""
    fn = func.func_name.lower()
    n = len(partition_rows)
    current_row = partition_rows[current_idx]

    # Ranking functions
    if fn == 'row_number':
        return current_idx + 1

    if fn == 'rank':
        return rank_info.get('rank', current_idx + 1)

    if fn == 'dense_rank':
        return rank_info.get('dense_rank', current_idx + 1)

    if fn == 'ntile':
        if not func.args:
            raise CompileError("NTILE requires exactly one argument")
        num_buckets = func.args[0]
        if isinstance(num_buckets, SqlLiteral):
            num_buckets = num_buckets.value
        if not isinstance(num_buckets, int) or num_buckets <= 0:
            raise CompileError("NTILE argument must be a positive integer")
        # Distribute rows as evenly as possible
        rows_per_bucket = n / num_buckets
        return int(current_idx / rows_per_bucket) + 1 if rows_per_bucket > 0 else 1

    if fn == 'percent_rank':
        if n <= 1:
            return 0.0
        rank = rank_info.get('rank', current_idx + 1)
        return (rank - 1) / (n - 1)

    if fn == 'cume_dist':
        rank = rank_info.get('rank', current_idx + 1)
        # Count rows with order value <= current
        count_le = rank_info.get('cume_count', current_idx + 1)
        return count_le / n

    # Navigation functions
    if fn == 'lag':
        offset = 1
        default = None
        if len(func.args) >= 2:
            off = func.args[1]
            offset = off.value if isinstance(off, SqlLiteral) else off
        if len(func.args) >= 3:
            dflt = func.args[2]
            default = dflt.value if isinstance(dflt, SqlLiteral) else dflt
        target_idx = current_idx - offset
        if target_idx < 0 or target_idx >= n:
            return default
        return _eval_arg(func.args[0], partition_rows[target_idx])

    if fn == 'lead':
        offset = 1
        default = None
        if len(func.args) >= 2:
            off = func.args[1]
            offset = off.value if isinstance(off, SqlLiteral) else off
        if len(func.args) >= 3:
            dflt = func.args[2]
            default = dflt.value if isinstance(dflt, SqlLiteral) else dflt
        target_idx = current_idx + offset
        if target_idx < 0 or target_idx >= n:
            return default
        return _eval_arg(func.args[0], partition_rows[target_idx])

    if fn == 'first_value':
        if not func.args:
            raise CompileError("FIRST_VALUE requires an argument")
        s, e = _frame_indices(frame_start, frame_end, current_idx, n)
        if s <= e and s < n:
            return _eval_arg(func.args[0], partition_rows[s])
        return None

    if fn == 'last_value':
        if not func.args:
            raise CompileError("LAST_VALUE requires an argument")
        s, e = _frame_indices(frame_start, frame_end, current_idx, n)
        if s <= e and e < n:
            return _eval_arg(func.args[0], partition_rows[e])
        return None

    if fn == 'nth_value':
        if len(func.args) < 2:
            raise CompileError("NTH_VALUE requires two arguments")
        nth = func.args[1]
        nth_val = nth.value if isinstance(nth, SqlLiteral) else nth
        if not isinstance(nth_val, int) or nth_val <= 0:
            raise CompileError("NTH_VALUE second argument must be a positive integer")
        s, e = _frame_indices(frame_start, frame_end, current_idx, n)
        target = s + nth_val - 1
        if target <= e and target < n:
            return _eval_arg(func.args[0], partition_rows[target])
        return None

    # Aggregate window functions
    if fn in AGGREGATE_FUNCS:
        s, e = _frame_indices(frame_start, frame_end, current_idx, n)
        frame_rows = partition_rows[s:e + 1] if s <= e else []
        return _aggregate_over_frame(fn, func.args, frame_rows, func.distinct)

    raise CompileError(f"Unknown window function: {func.func_name}")


def _eval_arg(arg, row: Row) -> Any:
    """Evaluate a window function argument against a row."""
    if isinstance(arg, SqlColumnRef):
        col_name = arg.column
        if arg.table:
            col_name = f"{arg.table}.{arg.column}"
        return row.get(col_name)
    if isinstance(arg, SqlLiteral):
        return arg.value
    # Try eval_expr for complex expressions
    try:
        return eval_expr(arg, row)
    except Exception:
        return None


def _aggregate_over_frame(func: str, args: List, frame_rows: List[Row],
                          distinct: bool = False) -> Any:
    """Compute aggregate over frame rows."""
    if func == 'count' and not args:
        return len(frame_rows)

    values = []
    for row in frame_rows:
        if args:
            v = _eval_arg(args[0], row)
        else:
            v = 1  # COUNT(*)
        if v is not None:
            values.append(v)

    if distinct:
        values = list(dict.fromkeys(values))  # preserve order, deduplicate

    if func == 'count':
        if not args:
            return len(frame_rows)
        return len(values)
    if func == 'sum':
        return sum(values) if values else None
    if func == 'avg':
        return sum(values) / len(values) if values else None
    if func == 'min':
        return min(values) if values else None
    if func == 'max':
        return max(values) if values else None
    return None


# =============================================================================
# Compute rank info for a partition
# =============================================================================

def _compute_rank_info(partition_rows: List[Row], order_by: Optional[List[Tuple[Any, bool]]]) -> List[Dict]:
    """Compute rank, dense_rank, cume_dist info for each row in a partition."""
    n = len(partition_rows)
    if not order_by or n == 0:
        return [{'rank': 1, 'dense_rank': 1, 'cume_count': n} for _ in range(n)]

    # Extract order values for each row
    def _order_key(row):
        return tuple(_eval_arg(expr, row) for expr, _ in order_by)

    order_vals = [_order_key(r) for r in partition_rows]

    infos = []
    current_rank = 1
    current_dense = 1
    for i in range(n):
        if i == 0:
            infos.append({'rank': 1, 'dense_rank': 1})
        else:
            if order_vals[i] == order_vals[i - 1]:
                infos.append({'rank': infos[i - 1]['rank'],
                              'dense_rank': infos[i - 1]['dense_rank']})
            else:
                current_dense += 1
                infos.append({'rank': i + 1, 'dense_rank': current_dense})

    # Compute cume_count: number of rows with order_val <= current
    for i in range(n):
        count_le = 0
        for j in range(n):
            if order_vals[j] <= order_vals[i]:
                count_le += 1
        infos[i]['cume_count'] = count_le

    return infos


# =============================================================================
# Extended Parser with window function support
# =============================================================================

class WindowParser(CTASParser):
    """Parser extended with window function syntax."""

    def _parse_select_item(self) -> SelectExpr:
        """Override to detect window functions after parsing the expression.

        Strategy: Let the parent parse the expression normally (producing SqlFuncCall
        or SqlAggCall), then check if OVER follows. If so, wrap as SqlWindowFunc.
        """
        # Check for * or table.*
        if self.peek().type == TokenType.STAR:
            self.advance()
            return SelectExpr(expr=SqlStar(), alias=None)

        expr = self._parse_expr()

        # Check if this is table.* pattern
        if isinstance(expr, SqlColumnRef) and self.peek().type == TokenType.DOT:
            self.advance()
            if self.match(TokenType.STAR):
                return SelectExpr(expr=SqlStar(table=expr.column), alias=None)

        # Check if OVER follows -- this makes it a window function
        if (self.peek().type == TokenType.IDENT
                and self.peek().value.upper() == 'OVER'):
            expr = self._wrap_as_window_func(expr)

        # Parse alias
        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENT).value
        elif self.peek().type == TokenType.IDENT and self.peek().type not in (
            TokenType.FROM, TokenType.WHERE, TokenType.GROUP, TokenType.ORDER,
            TokenType.HAVING, TokenType.LIMIT, TokenType.JOIN,
        ):
            next_t = self.peek()
            if (next_t.value and next_t.value.lower() not in KEYWORDS
                    and next_t.value.upper() != 'OVER'):
                alias = self.advance().value

        return SelectExpr(expr=expr, alias=alias)

    def _wrap_as_window_func(self, expr) -> SqlWindowFunc:
        """Convert a parsed func/agg call + OVER into SqlWindowFunc."""
        self.advance()  # consume OVER

        # Extract function info from the already-parsed expression
        func_name = ''
        args = []
        distinct = False

        if isinstance(expr, SqlFuncCall):
            func_name = expr.func_name
            args = expr.args
            distinct = expr.distinct
        elif hasattr(expr, 'func') and hasattr(expr, 'arg'):
            # SqlAggCall: func is string, arg is the expression
            func_name = expr.func.upper()
            args = [expr.arg] if expr.arg is not None else []
            distinct = getattr(expr, 'distinct', False)
        else:
            raise ParseError(f"Expected function call before OVER, got {type(expr).__name__}")

        # Parse window spec: OVER (spec) or OVER window_name
        if self.peek().type == TokenType.LPAREN:
            window = self._parse_window_spec()
            return SqlWindowFunc(func_name=func_name, args=args, window=window,
                                 distinct=distinct)
        else:
            wname = self.expect(TokenType.IDENT).value
            return SqlWindowFunc(func_name=func_name, args=args, window_name=wname,
                                 distinct=distinct)

    def _parse_window_spec(self) -> WindowSpec:
        """Parse (PARTITION BY ... ORDER BY ... frame_clause)."""
        self.expect(TokenType.LPAREN)

        partition_by = None
        order_by = None
        frame_start = None
        frame_end = None
        base_window = None

        # Optional base window name -- only if it's an IDENT not matching keywords
        if (self.peek().type == TokenType.IDENT
            and self.peek().value.upper() not in ('PARTITION', 'ORDER', 'ROWS', 'RANGE', 'GROUPS')):
            saved = self.pos
            name = self.advance().value
            next_tok = self.peek()
            if (next_tok.type == TokenType.RPAREN
                or self._check_ident('partition')
                or next_tok.type == TokenType.ORDER
                or self._check_ident('rows')
                or self._check_ident('range')):
                base_window = name
            else:
                self.pos = saved

        # PARTITION BY
        if self._check_ident('partition'):
            self.advance()  # PARTITION
            self.expect(TokenType.BY)  # BY is a keyword token
            partition_by = [self._parse_expr()]
            while self.match(TokenType.COMMA):
                partition_by.append(self._parse_expr())

        # ORDER BY
        if self.peek().type == TokenType.ORDER:
            self.advance()  # ORDER
            self.expect(TokenType.BY)
            order_by = self._parse_order_list()

        # Frame clause: ROWS BETWEEN ... AND ...
        if self._check_ident('rows') or self._check_ident('range'):
            self.advance()  # ROWS/RANGE
            if self.peek().type == TokenType.BETWEEN:
                self.advance()  # BETWEEN
                frame_start = self._parse_frame_bound()
                self.expect(TokenType.AND)
                frame_end = self._parse_frame_bound()
            else:
                # Short form: ROWS frame_start (implicit CURRENT ROW end)
                frame_start = self._parse_frame_bound()
                frame_end = FrameBound('current_row')

        self.expect(TokenType.RPAREN)
        return WindowSpec(
            partition_by=partition_by,
            order_by=order_by,
            frame_start=frame_start,
            frame_end=frame_end,
            base_window=base_window,
        )

    def _parse_order_list(self) -> List[Tuple[Any, bool]]:
        """Parse ORDER BY expression list."""
        items = []
        expr = self._parse_expr()
        asc = True
        if self.peek().type == TokenType.ASC:
            self.advance()
        elif self.peek().type == TokenType.DESC:
            self.advance()
            asc = False
        items.append((expr, asc))
        while self.match(TokenType.COMMA):
            expr = self._parse_expr()
            asc = True
            if self.peek().type == TokenType.ASC:
                self.advance()
            elif self.peek().type == TokenType.DESC:
                self.advance()
                asc = False
            items.append((expr, asc))
        return items

    def _parse_frame_bound(self) -> FrameBound:
        """Parse a frame bound like UNBOUNDED PRECEDING, 3 PRECEDING, CURRENT ROW, etc."""
        if self._check_ident('unbounded'):
            self.advance()
            if self._check_ident('preceding'):
                self.advance()
                return FrameBound('unbounded_preceding')
            elif self._check_ident('following'):
                self.advance()
                return FrameBound('unbounded_following')
            raise ParseError("Expected PRECEDING or FOLLOWING after UNBOUNDED")

        if self._check_ident('current'):
            self.advance()
            self._expect_ident('row')
            return FrameBound('current_row')

        # N PRECEDING or N FOLLOWING
        if self.peek().type == TokenType.NUMBER:
            n = int(self.advance().value)
            if self._check_ident('preceding'):
                self.advance()
                return FrameBound('preceding', n)
            elif self._check_ident('following'):
                self.advance()
                return FrameBound('following', n)
            raise ParseError("Expected PRECEDING or FOLLOWING after integer")

        raise ParseError(f"Expected frame bound, got {self.peek()}")

    def _check_ident(self, name: str) -> bool:
        """Check if current token has given name (case-insensitive).
        Works for both IDENT and keyword tokens."""
        tok = self.peek()
        if tok.value and isinstance(tok.value, str) and tok.value.upper() == name.upper():
            return True
        return False

    def _expect_ident(self, name: str):
        """Expect a token with given name (case-insensitive).
        Works for both IDENT and keyword tokens."""
        tok = self.peek()
        if tok.value and isinstance(tok.value, str) and tok.value.upper() == name.upper():
            return self.advance()
        raise ParseError(f"Expected '{name}', got '{tok.value}'")


# =============================================================================
# Parse functions
# =============================================================================

def parse_window_sql(sql: str):
    """Parse a single SQL statement with window function support."""
    lexer = Lexer(sql)
    parser = WindowParser(lexer.tokens)
    return parser.parse()


def parse_window_sql_multi(sql: str):
    """Parse multiple SQL statements with window function support."""
    lexer = Lexer(sql)
    parser = WindowParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# WindowDB -- Database with window function support
# =============================================================================

class WindowDB(CTASDB):
    """CTASDB extended with window functions."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with window function support."""
        stmt = parse_window_sql(sql)
        return self._execute_window_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_window_sql_multi(sql)
        return [self._execute_window_stmt(s) for s in stmts]

    def _execute_window_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling window functions."""
        if isinstance(stmt, SelectStmt):
            # Check if any column contains a window function
            win_funcs = self._extract_window_funcs(stmt)
            if win_funcs:
                return self._exec_select_with_windows(stmt, win_funcs)
        return self._execute_ctas_stmt(stmt)

    def _extract_window_funcs(self, stmt: SelectStmt) -> List[Tuple[int, SqlWindowFunc, str]]:
        """Extract window functions from SELECT columns.
        Returns [(col_idx, SqlWindowFunc, alias), ...]"""
        result = []
        for i, col in enumerate(stmt.columns):
            wf = self._find_window_func(col.expr)
            if wf:
                alias = col.alias or f"_win_{i}"
                result.append((i, wf, alias))
        return result

    def _find_window_func(self, expr) -> Optional[SqlWindowFunc]:
        """Find a SqlWindowFunc in an expression tree."""
        if isinstance(expr, SqlWindowFunc):
            return expr
        return None

    def _exec_select_with_windows(self, stmt: SelectStmt,
                                  win_funcs: List[Tuple[int, SqlWindowFunc, str]]) -> ResultSet:
        """Execute a SELECT with window functions.

        Strategy: Build a 'base' SELECT that replaces window function columns with
        the columns they reference, execute it via the parent, then compute window
        function values over the result rows.
        """
        # Step 1: Build base SELECT with non-window columns (SELECT * FROM ... WHERE ...)
        # to get all source rows with all columns available
        base_stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlStar())],
            from_table=stmt.from_table,
            joins=stmt.joins,
            where=stmt.where,
            group_by=stmt.group_by,
            having=stmt.having,
        )
        base_result = self._execute_ctas_stmt(base_stmt)

        # Convert base result rows to Row objects for eval_expr
        base_rows = []
        for raw_row in base_result.rows:
            data = {}
            for i, col_name in enumerate(base_result.columns):
                data[col_name] = raw_row[i]
            base_rows.append(Row(data))

        # Step 2: For each window function, compute values
        win_results = {}
        for col_idx, wf, alias in win_funcs:
            spec = self._resolve_window_spec(wf)
            values = self._compute_window_values(wf, spec, base_rows)
            win_results[col_idx] = values

        # Step 3: Build output columns and rows
        output_columns = []
        for i, col in enumerate(stmt.columns):
            if col.alias:
                output_columns.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                output_columns.append(col.expr.column)
            elif isinstance(col.expr, SqlWindowFunc):
                output_columns.append(f"_win_{i}")
            else:
                output_columns.append(f"col_{i}")

        # Override window func column names with their aliases
        for col_idx, wf, alias in win_funcs:
            output_columns[col_idx] = alias

        output_rows = []
        for row_idx, row in enumerate(base_rows):
            out_row = []
            for col_idx, col in enumerate(stmt.columns):
                if col_idx in win_results:
                    out_row.append(win_results[col_idx][row_idx])
                else:
                    val = _eval_arg(col.expr, row)
                    out_row.append(val)
            output_rows.append(out_row)

        # Apply outer ORDER BY if present
        if stmt.order_by:
            output_rows = self._apply_order_by(output_rows, output_columns,
                                               stmt.order_by, base_rows)

        # Apply LIMIT/OFFSET
        if stmt.offset:
            output_rows = output_rows[stmt.offset:]
        if stmt.limit is not None:
            output_rows = output_rows[:stmt.limit]

        return ResultSet(columns=output_columns, rows=output_rows)

    def _resolve_window_spec(self, wf: SqlWindowFunc) -> WindowSpec:
        """Resolve window spec, merging named windows if needed."""
        if wf.window:
            return wf.window
        # If window_name specified, we'd look it up from named windows
        # For now, return empty spec (whole partition)
        return WindowSpec()

    def _compute_window_values(self, wf: SqlWindowFunc, spec: WindowSpec,
                               rows: List[Row]) -> List[Any]:
        """Compute window function values for all rows."""
        n = len(rows)
        if n == 0:
            return []

        # Partition the rows
        partitions = self._partition_rows(rows, spec.partition_by)

        # Determine frame bounds
        has_order = spec.order_by is not None and len(spec.order_by) > 0
        if spec.frame_start and spec.frame_end:
            frame_start = spec.frame_start
            frame_end = spec.frame_end
        else:
            frame_start, frame_end = _default_frame(wf.func_name, has_order)

        # Compute values
        values = [None] * n

        for partition_indices in partitions:
            # Sort within partition by ORDER BY
            p_rows = [rows[i] for i in partition_indices]
            p_indices = list(range(len(p_rows)))

            if spec.order_by:
                # Sort the partition rows
                sorted_pairs = self._sort_partition(p_rows, p_indices, spec.order_by)
                p_rows = [pair[0] for pair in sorted_pairs]
                p_indices_sorted = [pair[1] for pair in sorted_pairs]
                # Map back to original indices
                sorted_orig = [partition_indices[pi] for pi in p_indices_sorted]
            else:
                sorted_orig = partition_indices

            # Compute rank info
            rank_infos = _compute_rank_info(p_rows, spec.order_by)

            # Evaluate window function for each row in partition
            for local_idx in range(len(p_rows)):
                orig_idx = sorted_orig[local_idx]
                val = _evaluate_window_func(
                    wf, p_rows, local_idx, frame_start, frame_end,
                    rank_infos[local_idx]
                )
                values[orig_idx] = val

        return values

    def _partition_rows(self, rows: List[Row],
                        partition_by: Optional[List]) -> List[List[int]]:
        """Partition rows by PARTITION BY expressions. Returns list of index lists."""
        if not partition_by:
            return [list(range(len(rows)))]

        groups = {}
        for i, row in enumerate(rows):
            key = tuple(_eval_arg(expr, row) for expr in partition_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        return list(groups.values())

    def _sort_partition(self, rows: List[Row], indices: List[int],
                        order_by: List[Tuple[Any, bool]]) -> List[Tuple[Row, int]]:
        """Sort partition rows by ORDER BY. Returns [(row, original_index), ...]."""
        pairs = list(zip(rows, indices))

        def sort_key(pair):
            row = pair[0]
            key = []
            for expr, asc in order_by:
                val = _eval_arg(expr, row)
                # Handle None: sort nulls last
                if val is None:
                    key.append((1, None))
                else:
                    if not asc:
                        # For descending, negate numeric values
                        if isinstance(val, (int, float)):
                            key.append((0, -val))
                        else:
                            key.append((0, val))
                    else:
                        key.append((0, val))
            return key

        # Custom comparison for mixed types
        from functools import cmp_to_key

        def compare_pairs(a, b):
            row_a, row_b = a[0], b[0]
            for expr, asc in order_by:
                va = _eval_arg(expr, row_a)
                vb = _eval_arg(expr, row_b)
                # Nulls sort last
                if va is None and vb is None:
                    continue
                if va is None:
                    return 1
                if vb is None:
                    return -1
                if va < vb:
                    return -1 if asc else 1
                if va > vb:
                    return 1 if asc else -1
            return 0

        pairs.sort(key=cmp_to_key(compare_pairs))
        return pairs

    def _apply_order_by(self, output_rows: List[List], columns: List[str],
                        order_by: List[Tuple[Any, bool]],
                        base_rows: List[Row]) -> List[List]:
        """Apply outer ORDER BY to output rows."""
        from functools import cmp_to_key

        # Build index-based pairs for stable sort
        pairs = list(enumerate(output_rows))

        def compare(a, b):
            idx_a, row_a = a
            idx_b, row_b = b
            for expr, asc in order_by:
                # Evaluate expression
                if isinstance(expr, SqlColumnRef):
                    col_name = expr.column
                    # Find column index
                    try:
                        ci = columns.index(col_name)
                        va, vb = row_a[ci], row_b[ci]
                    except ValueError:
                        va = _eval_arg(expr, base_rows[idx_a]) if idx_a < len(base_rows) else None
                        vb = _eval_arg(expr, base_rows[idx_b]) if idx_b < len(base_rows) else None
                else:
                    va = _eval_arg(expr, base_rows[idx_a]) if idx_a < len(base_rows) else None
                    vb = _eval_arg(expr, base_rows[idx_b]) if idx_b < len(base_rows) else None

                if va is None and vb is None:
                    continue
                if va is None:
                    return 1
                if vb is None:
                    return -1
                if va < vb:
                    return -1 if asc else 1
                if va > vb:
                    return 1 if asc else -1
            return 0

        pairs.sort(key=cmp_to_key(compare))
        return [row for _, row in pairs]
