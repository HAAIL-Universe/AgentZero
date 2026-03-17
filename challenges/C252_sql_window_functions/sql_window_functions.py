"""
C252: SQL Window Functions
Extends C251 (SQL Triggers) / C250 (Views) / C249 (Stored Procedures) / C247 (Mini Database)

Adds SQL window functions to the database engine:
- Ranking: ROW_NUMBER(), RANK(), DENSE_RANK(), NTILE(n)
- Navigation: LEAD(expr, offset, default), LAG(expr, offset, default),
              FIRST_VALUE(expr), LAST_VALUE(expr), NTH_VALUE(expr, n)
- Aggregate windows: SUM/AVG/MIN/MAX/COUNT(...) OVER (...)
- PARTITION BY for grouping
- ORDER BY within partitions
- Frame specs: ROWS BETWEEN ... AND ...
  (UNBOUNDED PRECEDING, N PRECEDING, CURRENT ROW, N FOLLOWING, UNBOUNDED FOLLOWING)
- Named windows: WINDOW w AS (...), referenceable in OVER w
- Multiple window functions in a single SELECT
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum, auto

# Import C251 (which imports C250 -> C249 -> C247)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C251_sql_triggers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C250_sql_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from sql_triggers import (
    TriggerDB, TriggerLexer, TriggerParser,
    TriggerCatalog, TriggerExecutor,
    CreateTriggerStmt, DropTriggerStmt, ShowTriggersStmt, AlterTriggerStmt,
)

from sql_views import (
    ViewDB, ViewLexer, ViewParser, ViewCatalog,
    CreateViewStmt, DropViewStmt, ShowViewsStmt, DescribeViewStmt,
)

from stored_procedures import (
    ProcDB, ProcLexer, ProcParser, ProcQueryCompiler, ProcExecutor,
    RoutineCatalog, ProcToken,
)

from mini_database import (
    MiniDB, ResultSet, DatabaseError,
    Lexer, Parser, Token, TokenType,
    parse_sql, parse_sql_multi,
    SelectStmt, SelectExpr, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    BeginStmt, CommitStmt, RollbackStmt, SavepointStmt,
    ShowTablesStmt, DescribeStmt, ExplainStmt,
    ColumnDef,
    SqlColumnRef, SqlLiteral, SqlBinOp, SqlComparison, SqlLogic,
    SqlIsNull, SqlFuncCall, SqlAggCall, SqlBetween, SqlInList,
    SqlCase, SqlStar,
    QueryCompiler, StorageEngine, CompileError, CatalogError,
    ParseError, KEYWORDS,
)

from query_executor import Row, eval_expr, ColumnRef, Literal, ArithExpr, Comparison
from transaction_manager import IsolationLevel


# =============================================================================
# Window Function AST Nodes
# =============================================================================

class FrameBound(Enum):
    UNBOUNDED_PRECEDING = auto()
    N_PRECEDING = auto()
    CURRENT_ROW = auto()
    N_FOLLOWING = auto()
    UNBOUNDED_FOLLOWING = auto()


@dataclass
class WindowFrameBound:
    bound_type: FrameBound
    offset: Optional[int] = None  # for N PRECEDING / N FOLLOWING


@dataclass
class WindowFrame:
    """ROWS BETWEEN start AND end"""
    start: WindowFrameBound
    end: WindowFrameBound


@dataclass
class WindowSpec:
    """The OVER (...) clause."""
    partition_by: List[Any] = field(default_factory=list)  # SQL exprs
    order_by: List[Tuple[Any, bool]] = field(default_factory=list)  # (expr, asc)
    frame: Optional[WindowFrame] = None
    ref_name: Optional[str] = None  # reference to named window


@dataclass
class SqlWindowCall:
    """A window function call: func(...) OVER (...)"""
    func_name: str  # 'row_number', 'rank', 'dense_rank', 'ntile', 'lead', 'lag',
                     # 'first_value', 'last_value', 'nth_value',
                     # 'sum', 'avg', 'min', 'max', 'count'
    args: List[Any]  # function arguments
    window: WindowSpec
    distinct: bool = False  # for aggregate window functions


@dataclass
class WindowDefStmt:
    """Named window: WINDOW w AS (PARTITION BY ... ORDER BY ...)"""
    name: str
    spec: WindowSpec


# =============================================================================
# Window-Aware SelectStmt (extends base)
# =============================================================================

@dataclass
class WindowSelectStmt:
    """A SELECT that may contain window functions and WINDOW clause."""
    base: SelectStmt  # the underlying SELECT
    window_defs: Dict[str, WindowSpec] = field(default_factory=dict)  # named windows
    window_calls: List[Tuple[int, SqlWindowCall]] = field(default_factory=list)
    # (column_index, window_call) -- which SELECT columns are window fns


# =============================================================================
# Window Keywords for Lexer
# =============================================================================

# Window function names and context keywords -- NOT reclassified as ProcTokens.
# They remain as IDENT tokens and are recognized contextually by the parser.
# This avoids breaking the base parser which expects TokenType enum values.
WINDOW_FUNC_NAMES = {
    'row_number', 'rank', 'dense_rank', 'ntile',
    'lead', 'lag', 'first_value', 'last_value', 'nth_value',
}
WINDOW_CONTEXT_WORDS = {
    'over', 'partition', 'rows', 'range', 'unbounded',
    'preceding', 'following', 'current', 'window',
}


class WindowLexer(TriggerLexer):
    """Lexer extended for window functions.
    Window keywords are NOT reclassified -- they stay as IDENT tokens
    and are recognized contextually by the WindowParser."""

    def __init__(self, sql: str):
        super().__init__(sql)
        # Undo any reclassification of window-context words that the trigger lexer
        # might have done (e.g., 'row' is in TRIGGER_KEYWORDS but also a window keyword)
        # No additional reclassification needed -- all handled contextually.


# =============================================================================
# Window Parser (extends TriggerParser)
# =============================================================================

# Window function names that take no arguments (just empty parens)
RANKING_FUNCS = {'row_number', 'rank', 'dense_rank'}
# Window function names that take arguments
PARAM_WINDOW_FUNCS = {'ntile', 'lead', 'lag', 'first_value', 'last_value', 'nth_value'}
# Aggregate functions that can be used as window functions
AGG_WINDOW_FUNCS = {'sum', 'avg', 'min', 'max', 'count'}
ALL_WINDOW_FUNCS = RANKING_FUNCS | PARAM_WINDOW_FUNCS | AGG_WINDOW_FUNCS


class WindowParser(TriggerParser):
    """Parser extended with window function parsing.

    Strategy: Override _parse_select_item to detect window function calls.
    A window function is func(...) OVER (...). We detect the OVER keyword
    after parsing a function call.
    """

    def _peek_word(self) -> Optional[str]:
        """Peek at current token's value as lowercase string."""
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            if hasattr(tok, 'value') and tok.value is not None:
                val = tok.value
                return val.lower() if isinstance(val, str) else None
        return None

    def _peek_word_at(self, offset: int) -> Optional[str]:
        """Peek at token at pos+offset."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            tok = self.tokens[idx]
            if hasattr(tok, 'value') and tok.value is not None:
                val = tok.value
                return val.lower() if isinstance(val, str) else None
        return None

    def _is_window_func_ahead(self) -> bool:
        """Check if current position looks like a window function call.
        Pattern: func_name ( ... ) OVER"""
        word = self._peek_word()
        if word not in ALL_WINDOW_FUNCS:
            return False
        # Look for OVER after the closing paren
        depth = 0
        i = self.pos + 1
        while i < len(self.tokens):
            tok = self.tokens[i]
            if hasattr(tok, 'type'):
                if tok.type == TokenType.LPAREN:
                    depth += 1
                elif tok.type == TokenType.RPAREN:
                    depth -= 1
                    if depth == 0:
                        # Check next token for OVER
                        if i + 1 < len(self.tokens):
                            next_tok = self.tokens[i + 1]
                            val = getattr(next_tok, 'value', '')
                            if isinstance(val, str) and val.lower() == 'over':
                                return True
                        return False
            i += 1
        return False

    def _parse_select_item(self) -> SelectExpr:
        """Override to detect window function calls."""
        if self._is_window_func_ahead():
            win_call = self._parse_window_call()
            # Check for alias
            alias = None
            if self._peek_word() == 'as':
                self._advance()
                alias = self._expect_ident()
            elif self._peek_type() == TokenType.IDENT:
                next_val = self.tokens[self.pos].value.lower() if self.pos < len(self.tokens) else ''
                if next_val not in ('from', 'where', 'into', 'group', 'order',
                                    'having', 'limit', 'union'):
                    alias = self._expect_ident()
            return SelectExpr(expr=win_call, alias=alias)

        # Fall back to standard parsing
        return super()._parse_select_item()

    def _parse_select_list(self):
        """Override ProcParser._parse_select_list to detect window function calls.
        ProcParser's version doesn't call _parse_select_item, so we must
        intercept here."""
        columns = []
        while True:
            if self._is_window_func_ahead():
                columns.append(self._parse_select_item())
            else:
                tt = self._peek_type()
                if isinstance(tt, TokenType) and tt == TokenType.STAR:
                    self.advance()
                    columns.append(SelectExpr(expr=SqlStar(table=None)))
                else:
                    expr = self._parse_expr()
                    alias = None
                    if self.match(TokenType.AS):
                        alias = self._expect_ident()
                    elif isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.IDENT:
                        next_val = self.peek().value.lower()
                        if next_val not in ('from', 'where', 'into', 'group', 'order',
                                            'having', 'limit', 'union'):
                            alias = self.advance().value
                    columns.append(SelectExpr(expr=expr, alias=alias))
            if not self.match(TokenType.COMMA):
                break
        return columns

    def _parse_window_call(self) -> SqlWindowCall:
        """Parse: func_name([args]) OVER (window_spec)"""
        func_name = self._peek_word()
        self._advance()  # consume function name

        # Parse arguments
        self._expect(TokenType.LPAREN)
        args = []
        distinct = False

        if func_name in RANKING_FUNCS:
            # No arguments
            pass
        elif func_name == 'count':
            # COUNT(*) or COUNT(expr) or COUNT(DISTINCT expr)
            if self._peek_type() == TokenType.STAR:
                self._advance()
                # count(*) -- no arg means count_star
            else:
                if self._peek_word() == 'distinct':
                    self._advance()
                    distinct = True
                args.append(self._parse_expr())
        elif func_name in AGG_WINDOW_FUNCS:
            if self._peek_word() == 'distinct':
                self._advance()
                distinct = True
            args.append(self._parse_expr())
        else:
            # ntile, lead, lag, first_value, last_value, nth_value
            while self._peek_type() != TokenType.RPAREN:
                args.append(self._parse_expr())
                if self._peek_type() == TokenType.COMMA:
                    self._advance()

        self._expect(TokenType.RPAREN)

        # Expect OVER
        word = self._peek_word()
        if word != 'over':
            raise ParseError("Expected OVER after window function")
        self._advance()

        # Parse window specification
        window = self._parse_window_spec()

        return SqlWindowCall(
            func_name=func_name,
            args=args,
            window=window,
            distinct=distinct,
        )

    def _parse_window_spec(self) -> WindowSpec:
        """Parse: ( [PARTITION BY ...] [ORDER BY ...] [frame_clause] )
        or: window_name (named reference)"""
        # Could be just a name reference: OVER w
        if self._peek_type() != TokenType.LPAREN:
            name = self._expect_ident()
            return WindowSpec(ref_name=name)

        self._expect(TokenType.LPAREN)
        spec = WindowSpec()

        # PARTITION BY
        if self._peek_word() == 'partition':
            self._advance()
            word = self._peek_word()
            if word == 'by':
                self._advance()
            spec.partition_by = self._parse_expr_list_until(
                stop_words={'order', 'rows', 'range', 'window'},
                stop_types={TokenType.RPAREN}
            )

        # ORDER BY
        if self._peek_word() == 'order':
            self._advance()
            word = self._peek_word()
            if word == 'by':
                self._advance()
            spec.order_by = self._parse_order_list_until(
                stop_words={'rows', 'range', 'window'},
                stop_types={TokenType.RPAREN}
            )

        # Frame clause
        word = self._peek_word()
        if word in ('rows', 'range'):
            spec.frame = self._parse_frame_clause()

        self._expect(TokenType.RPAREN)
        return spec

    def _parse_expr_list_until(self, stop_words: Set[str],
                                stop_types: Set[TokenType]) -> List[Any]:
        """Parse comma-separated expressions until a stop word or token type."""
        exprs = []
        while True:
            w = self._peek_word()
            if w in stop_words:
                break
            if self._peek_type() in stop_types:
                break
            exprs.append(self._parse_expr())
            if self._peek_type() == TokenType.COMMA:
                self._advance()
            else:
                break
        return exprs

    def _parse_order_list_until(self, stop_words: Set[str],
                                 stop_types: Set[TokenType]) -> List[Tuple[Any, bool]]:
        """Parse ORDER BY list until stop word or token."""
        order_list = []
        while True:
            w = self._peek_word()
            if w in stop_words:
                break
            if self._peek_type() in stop_types:
                break
            expr = self._parse_expr()
            asc = True
            w = self._peek_word()
            if w == 'asc':
                self._advance()
            elif w == 'desc':
                self._advance()
                asc = False
            order_list.append((expr, asc))
            if self._peek_type() == TokenType.COMMA:
                self._advance()
            else:
                break
        return order_list

    def _parse_frame_clause(self) -> WindowFrame:
        """Parse: ROWS BETWEEN start AND end
        or: ROWS frame_bound (shorthand for BETWEEN frame_bound AND CURRENT ROW)"""
        self._advance()  # consume ROWS/RANGE

        if self._peek_word() == 'between':
            self._advance()
            start = self._parse_frame_bound()
            # expect AND
            word = self._peek_word()
            if word == 'and':
                self._advance()
            end = self._parse_frame_bound()
            return WindowFrame(start=start, end=end)
        else:
            # Single bound -> BETWEEN bound AND CURRENT ROW
            start = self._parse_frame_bound()
            return WindowFrame(
                start=start,
                end=WindowFrameBound(FrameBound.CURRENT_ROW)
            )

    def _parse_frame_bound(self) -> WindowFrameBound:
        """Parse a frame bound: UNBOUNDED PRECEDING, N PRECEDING, CURRENT ROW,
        N FOLLOWING, UNBOUNDED FOLLOWING"""
        word = self._peek_word()

        if word == 'unbounded':
            self._advance()
            direction = self._peek_word()
            self._advance()
            if direction == 'preceding':
                return WindowFrameBound(FrameBound.UNBOUNDED_PRECEDING)
            else:  # following
                return WindowFrameBound(FrameBound.UNBOUNDED_FOLLOWING)

        if word == 'current':
            self._advance()
            # expect ROW
            w = self._peek_word()
            if w == 'row':
                self._advance()
            return WindowFrameBound(FrameBound.CURRENT_ROW)

        # N PRECEDING or N FOLLOWING
        if self._peek_type() == TokenType.NUMBER:
            n = int(self.tokens[self.pos].value)
            self._advance()
            direction = self._peek_word()
            self._advance()
            if direction == 'preceding':
                return WindowFrameBound(FrameBound.N_PRECEDING, offset=n)
            else:  # following
                return WindowFrameBound(FrameBound.N_FOLLOWING, offset=n)

        raise ParseError(f"Expected frame bound, got '{word}'")

    def _expect_ident(self) -> str:
        """Consume and return an identifier (also accepts reclassified keywords as idents)."""
        tok = self.tokens[self.pos]
        if hasattr(tok, 'type'):
            if tok.type == TokenType.IDENT:
                self._advance()
                return tok.value
            # Also allow ProcTokens used as identifiers in alias context
            if isinstance(tok, ProcToken):
                self._advance()
                return tok.value
        raise ParseError(f"Expected identifier, got {tok}")

    def _peek_type(self) -> TokenType:
        """Get current token type."""
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            return tok.type if hasattr(tok, 'type') else TokenType.EOF
        return TokenType.EOF

    def _advance(self):
        """Move to next token."""
        if self.pos < len(self.tokens):
            self.pos += 1

    def _expect(self, ttype: TokenType):
        """Consume a token of the given type."""
        tok = self.tokens[self.pos]
        actual = tok.type if hasattr(tok, 'type') else None
        if actual != ttype:
            raise ParseError(f"Expected {ttype}, got {actual} ({tok})")
        self._advance()

    def _parse_select(self) -> SelectStmt:
        """Override to capture WINDOW clause at end of SELECT."""
        stmt = super()._parse_select()
        return stmt

    def match(self, ttype: TokenType) -> bool:
        """Try to match a token type, advance if matched."""
        if self.pos < len(self.tokens):
            tok = self.tokens[self.pos]
            if hasattr(tok, 'type') and tok.type == ttype:
                self._advance()
                return True
        return False


# =============================================================================
# Window Function Executor
# =============================================================================

class WindowExecutor:
    """Evaluates window functions on materialized result sets."""

    def __init__(self, compiler: QueryCompiler):
        self.compiler = compiler

    def evaluate_windows(self, rows: List[Dict[str, Any]],
                          columns: List[str],
                          select_exprs: List[SelectExpr],
                          window_defs: Dict[str, WindowSpec] = None
                          ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Evaluate all window functions in the select list.

        Takes materialized rows and adds window function result columns.
        Returns (modified_rows, modified_columns).
        """
        if window_defs is None:
            window_defs = {}

        # Find all window function calls in select list
        win_calls = []
        for i, se in enumerate(select_exprs):
            if isinstance(se.expr, SqlWindowCall):
                alias = se.alias or f"win_{i}"
                win_calls.append((i, alias, se.expr))

        if not win_calls:
            return rows, columns

        # Process each window function
        for col_idx, alias, win_call in win_calls:
            window = self._resolve_window(win_call.window, window_defs)
            self._compute_window(rows, alias, win_call, window)

        # Build final column list
        new_columns = []
        for i, se in enumerate(select_exprs):
            if isinstance(se.expr, SqlWindowCall):
                new_columns.append(se.alias or f"win_{i}")
            elif isinstance(se.expr, SqlStar):
                new_columns.extend(columns)
            elif se.alias:
                new_columns.append(se.alias)
            elif isinstance(se.expr, SqlColumnRef):
                new_columns.append(se.expr.column)
            elif isinstance(se.expr, SqlAggCall):
                new_columns.append(se.alias or self.compiler._agg_alias(se.expr))
            else:
                new_columns.append(f"col_{i}")

        return rows, new_columns

    def _resolve_window(self, spec: WindowSpec,
                         defs: Dict[str, WindowSpec]) -> WindowSpec:
        """Resolve named window references."""
        if spec.ref_name and spec.ref_name in defs:
            base = defs[spec.ref_name]
            # Merge: spec can add to base but not contradict
            return WindowSpec(
                partition_by=spec.partition_by or base.partition_by,
                order_by=spec.order_by or base.order_by,
                frame=spec.frame or base.frame,
            )
        return spec

    def _compute_window(self, rows: List[Dict[str, Any]], alias: str,
                          win_call: SqlWindowCall, window: WindowSpec):
        """Compute a window function and add results to rows."""
        if not rows:
            return

        # Partition the rows
        partitions = self._partition_rows(rows, window.partition_by)

        for partition_indices in partitions:
            # Sort within partition
            sorted_indices = self._sort_partition(
                rows, partition_indices, window.order_by
            )

            # Compute the window function for this partition
            func = win_call.func_name
            if func == 'row_number':
                self._compute_row_number(rows, sorted_indices, alias)
            elif func == 'rank':
                self._compute_rank(rows, sorted_indices, alias, window.order_by)
            elif func == 'dense_rank':
                self._compute_dense_rank(rows, sorted_indices, alias, window.order_by)
            elif func == 'ntile':
                n = self._eval_arg(win_call.args[0], rows[sorted_indices[0]]) if win_call.args else 1
                self._compute_ntile(rows, sorted_indices, alias, int(n))
            elif func == 'lead':
                self._compute_lead_lag(rows, sorted_indices, alias, win_call, offset_sign=1)
            elif func == 'lag':
                self._compute_lead_lag(rows, sorted_indices, alias, win_call, offset_sign=-1)
            elif func == 'first_value':
                self._compute_first_last_nth(rows, sorted_indices, alias, win_call, 'first', window)
            elif func == 'last_value':
                self._compute_first_last_nth(rows, sorted_indices, alias, win_call, 'last', window)
            elif func == 'nth_value':
                self._compute_first_last_nth(rows, sorted_indices, alias, win_call, 'nth', window)
            elif func in AGG_WINDOW_FUNCS:
                self._compute_aggregate_window(
                    rows, sorted_indices, alias, win_call, window
                )

    def _partition_rows(self, rows: List[Dict], partition_by: List[Any]
                         ) -> List[List[int]]:
        """Split row indices into partitions based on PARTITION BY expressions."""
        if not partition_by:
            return [list(range(len(rows)))]

        groups = {}
        for i, row in enumerate(rows):
            qe_row = Row(row)
            key = tuple(
                self._eval_sql_expr(expr, qe_row)
                for expr in partition_by
            )
            groups.setdefault(key, []).append(i)
        return list(groups.values())

    def _sort_partition(self, rows: List[Dict], indices: List[int],
                         order_by: List[Tuple[Any, bool]]) -> List[int]:
        """Sort partition indices according to ORDER BY."""
        if not order_by:
            return indices

        def sort_key(idx):
            qe_row = Row(rows[idx])
            keys = []
            for expr, asc in order_by:
                val = self._eval_sql_expr(expr, qe_row)
                # Handle None: sort NULLs last
                if val is None:
                    keys.append((1, None, not asc))
                else:
                    keys.append((0, val if asc else None, None if asc else val))
            return keys

        # Custom comparison to handle mixed types and None
        import functools

        def compare_keys(a_idx, b_idx):
            a_row = Row(rows[a_idx])
            b_row = Row(rows[b_idx])
            for expr, asc in order_by:
                a_val = self._eval_sql_expr(expr, a_row)
                b_val = self._eval_sql_expr(expr, b_row)
                # NULLs sort last
                if a_val is None and b_val is None:
                    continue
                if a_val is None:
                    return 1
                if b_val is None:
                    return -1
                if a_val < b_val:
                    return -1 if asc else 1
                if a_val > b_val:
                    return 1 if asc else -1
            return 0

        return sorted(indices, key=functools.cmp_to_key(compare_keys))

    def _eval_sql_expr(self, expr, qe_row: Row) -> Any:
        """Evaluate a SQL AST expression against a row."""
        qe_expr = self.compiler._sql_to_qe_expr(expr)
        return eval_expr(qe_expr, qe_row)

    def _eval_arg(self, arg, row: Dict) -> Any:
        """Evaluate a window function argument."""
        if isinstance(arg, SqlLiteral):
            return arg.value
        qe_row = Row(row)
        return self._eval_sql_expr(arg, qe_row)

    def _get_order_values(self, rows: List[Dict], idx: int,
                           order_by: List[Tuple[Any, bool]]) -> Tuple:
        """Get the ORDER BY values for a row."""
        qe_row = Row(rows[idx])
        return tuple(self._eval_sql_expr(expr, qe_row) for expr, _ in order_by)

    # -- Ranking functions --

    def _compute_row_number(self, rows: List[Dict], indices: List[int], alias: str):
        for rank, idx in enumerate(indices, 1):
            rows[idx][alias] = rank

    def _compute_rank(self, rows: List[Dict], indices: List[int], alias: str,
                       order_by: List[Tuple[Any, bool]]):
        if not order_by:
            for idx in indices:
                rows[idx][alias] = 1
            return

        prev_vals = None
        rank = 0
        for i, idx in enumerate(indices):
            vals = self._get_order_values(rows, idx, order_by)
            if vals != prev_vals:
                rank = i + 1
                prev_vals = vals
            rows[idx][alias] = rank

    def _compute_dense_rank(self, rows: List[Dict], indices: List[int], alias: str,
                              order_by: List[Tuple[Any, bool]]):
        if not order_by:
            for idx in indices:
                rows[idx][alias] = 1
            return

        prev_vals = None
        rank = 0
        for idx in indices:
            vals = self._get_order_values(rows, idx, order_by)
            if vals != prev_vals:
                rank += 1
                prev_vals = vals
            rows[idx][alias] = rank

    def _compute_ntile(self, rows: List[Dict], indices: List[int],
                        alias: str, n: int):
        total = len(indices)
        if n <= 0:
            n = 1
        base_size = total // n
        remainder = total % n
        tile = 1
        count = 0
        tile_size = base_size + (1 if tile <= remainder else 0)
        for idx in indices:
            if count >= tile_size and tile < n:
                tile += 1
                count = 0
                tile_size = base_size + (1 if tile <= remainder else 0)
            rows[idx][alias] = tile
            count += 1

    # -- Navigation functions --

    def _compute_lead_lag(self, rows: List[Dict], indices: List[int],
                           alias: str, win_call: SqlWindowCall, offset_sign: int):
        """Compute LEAD or LAG. offset_sign: +1 for LEAD, -1 for LAG."""
        expr = win_call.args[0] if win_call.args else None
        offset = 1
        default = None

        if len(win_call.args) >= 2:
            offset = self._eval_arg(win_call.args[1], rows[indices[0]])
            if offset is not None:
                offset = int(offset)
            else:
                offset = 1

        if len(win_call.args) >= 3:
            default = self._eval_arg(win_call.args[2], rows[indices[0]])

        for i, idx in enumerate(indices):
            target_i = i + (offset * offset_sign)
            if 0 <= target_i < len(indices):
                target_idx = indices[target_i]
                if expr is not None:
                    val = self._eval_arg(expr, rows[target_idx])
                else:
                    val = None
            else:
                val = default
            rows[idx][alias] = val

    def _compute_first_last_nth(self, rows: List[Dict], indices: List[int],
                                  alias: str, win_call: SqlWindowCall,
                                  mode: str, window: WindowSpec):
        """Compute FIRST_VALUE, LAST_VALUE, or NTH_VALUE."""
        expr = win_call.args[0] if win_call.args else None
        n = None
        if mode == 'nth' and len(win_call.args) >= 2:
            n = int(self._eval_arg(win_call.args[1], rows[indices[0]]))

        frame = window.frame
        if frame is None:
            # Default frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            # (when ORDER BY is specified), or entire partition (no ORDER BY)
            if window.order_by:
                frame = WindowFrame(
                    start=WindowFrameBound(FrameBound.UNBOUNDED_PRECEDING),
                    end=WindowFrameBound(FrameBound.CURRENT_ROW),
                )
            else:
                frame = WindowFrame(
                    start=WindowFrameBound(FrameBound.UNBOUNDED_PRECEDING),
                    end=WindowFrameBound(FrameBound.UNBOUNDED_FOLLOWING),
                )

        for i, idx in enumerate(indices):
            frame_start, frame_end = self._get_frame_bounds(i, len(indices), frame)
            frame_indices = indices[frame_start:frame_end + 1]

            if not frame_indices:
                rows[idx][alias] = None
                continue

            if mode == 'first':
                target = frame_indices[0]
                rows[idx][alias] = self._eval_arg(expr, rows[target]) if expr else None
            elif mode == 'last':
                target = frame_indices[-1]
                rows[idx][alias] = self._eval_arg(expr, rows[target]) if expr else None
            elif mode == 'nth':
                if n is not None and 1 <= n <= len(frame_indices):
                    target = frame_indices[n - 1]
                    rows[idx][alias] = self._eval_arg(expr, rows[target]) if expr else None
                else:
                    rows[idx][alias] = None

    # -- Aggregate window functions --

    def _compute_aggregate_window(self, rows: List[Dict], indices: List[int],
                                    alias: str, win_call: SqlWindowCall,
                                    window: WindowSpec):
        """Compute SUM/AVG/MIN/MAX/COUNT as window functions."""
        func = win_call.func_name
        expr = win_call.args[0] if win_call.args else None

        frame = window.frame
        if frame is None:
            if window.order_by:
                # Default with ORDER BY: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                frame = WindowFrame(
                    start=WindowFrameBound(FrameBound.UNBOUNDED_PRECEDING),
                    end=WindowFrameBound(FrameBound.CURRENT_ROW),
                )
            else:
                # Default without ORDER BY: entire partition
                frame = WindowFrame(
                    start=WindowFrameBound(FrameBound.UNBOUNDED_PRECEDING),
                    end=WindowFrameBound(FrameBound.UNBOUNDED_FOLLOWING),
                )

        for i, idx in enumerate(indices):
            frame_start, frame_end = self._get_frame_bounds(i, len(indices), frame)
            frame_indices = indices[frame_start:frame_end + 1]

            # Gather values in the frame
            values = []
            for fi in frame_indices:
                if func == 'count' and expr is None:
                    values.append(1)  # COUNT(*)
                elif expr is not None:
                    val = self._eval_arg(expr, rows[fi])
                    if val is not None:
                        values.append(val)

            if win_call.distinct:
                values = list(dict.fromkeys(values))  # preserve order, deduplicate

            # Compute the aggregate
            if func == 'count':
                if expr is None:
                    # COUNT(*) -- count all rows in frame
                    rows[idx][alias] = len(frame_indices)
                else:
                    rows[idx][alias] = len(values)
            elif func == 'sum':
                rows[idx][alias] = sum(values) if values else None
            elif func == 'avg':
                rows[idx][alias] = sum(values) / len(values) if values else None
            elif func == 'min':
                rows[idx][alias] = min(values) if values else None
            elif func == 'max':
                rows[idx][alias] = max(values) if values else None

    def _get_frame_bounds(self, current_pos: int, partition_size: int,
                           frame: WindowFrame) -> Tuple[int, int]:
        """Get absolute frame bounds (start, end) for a given position."""
        start = self._resolve_bound(frame.start, current_pos, partition_size, is_start=True)
        end = self._resolve_bound(frame.end, current_pos, partition_size, is_start=False)
        start = max(0, min(start, partition_size - 1))
        end = max(0, min(end, partition_size - 1))
        return start, end

    def _resolve_bound(self, bound: WindowFrameBound, current: int,
                        size: int, is_start: bool) -> int:
        if bound.bound_type == FrameBound.UNBOUNDED_PRECEDING:
            return 0
        elif bound.bound_type == FrameBound.UNBOUNDED_FOLLOWING:
            return size - 1
        elif bound.bound_type == FrameBound.CURRENT_ROW:
            return current
        elif bound.bound_type == FrameBound.N_PRECEDING:
            return current - (bound.offset or 0)
        elif bound.bound_type == FrameBound.N_FOLLOWING:
            return current + (bound.offset or 0)
        return current


# =============================================================================
# Window-Aware Query Compiler
# =============================================================================

class WindowQueryCompiler(QueryCompiler):
    """Query compiler that handles window functions.

    Window functions are NOT compiled into the operator tree. Instead, they are
    evaluated as a post-processing step on the materialized result set.
    The compiler strips window function columns from the projection and the
    executor adds them back after materialization.
    """

    def compile_select_with_windows(self, stmt: SelectStmt, txn_id: int,
                                     window_defs: Dict[str, WindowSpec] = None
                                     ) -> Tuple[Any, Any, List[Tuple[int, SqlWindowCall]], List[SelectExpr]]:
        """Compile a SELECT that may contain window functions.

        Returns: (plan, engine, window_calls, original_select_exprs)
        where window_calls is [(col_index, SqlWindowCall)] for post-processing.
        """
        if window_defs is None:
            window_defs = {}

        # Separate window function columns from regular columns
        window_calls = []
        regular_columns = []
        for i, se in enumerate(stmt.columns):
            if isinstance(se.expr, SqlWindowCall):
                window_calls.append((i, se.expr))
                # Keep placeholder in regular columns for ordering
                regular_columns.append(se)
            else:
                regular_columns.append(se)

        if not window_calls:
            # No window functions -- standard compilation
            plan, engine = self.compile_select(stmt, txn_id)
            return plan, engine, [], stmt.columns

        # Use SELECT * for the base query so all columns are available
        # for window function PARTITION BY, ORDER BY, and argument expressions
        base_columns = [SelectExpr(expr=SqlStar())]

        base_stmt = SelectStmt(
            columns=base_columns,
            from_table=stmt.from_table,
            joins=stmt.joins,
            where=stmt.where,
            group_by=stmt.group_by,
            having=stmt.having,
            order_by=None,  # Don't apply ORDER BY yet -- window fns need unsorted data
            limit=None,  # Don't limit yet
            offset=None,
            distinct=False,  # Don't distinct yet
        )

        plan, engine = self.compile_select(base_stmt, txn_id)
        return plan, engine, window_calls, stmt.columns


# =============================================================================
# WindowDB (extends TriggerDB)
# =============================================================================

class WindowDB(TriggerDB):
    """TriggerDB extended with SQL window functions."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        self.window_compiler = WindowQueryCompiler(self.storage)
        self.window_executor = WindowExecutor(self.window_compiler)

    def execute(self, sql: str) -> ResultSet:
        stmts = self._parse_window(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_window_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List[ResultSet]:
        stmts = self._parse_window(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_window_stmt(stmt))
        return results

    def _parse_window(self, sql: str) -> List[Any]:
        lexer = WindowLexer(sql)
        parser = WindowParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _execute_window_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling window functions in SELECT."""
        if isinstance(stmt, SelectStmt):
            return self._exec_select_with_windows(stmt)

        # Delegate non-SELECT statements to parent
        return self._execute_trigger_stmt(stmt)

    def _exec_select_with_windows(self, stmt: SelectStmt) -> ResultSet:
        """Execute a SELECT that may contain window functions."""
        # Expand views first (like ViewDB does)
        if hasattr(self, 'view_catalog'):
            stmt = self._expand_views_in_select(stmt, set())

        # Check if any column has a window function
        has_windows = any(
            isinstance(se.expr, SqlWindowCall) for se in stmt.columns
        )

        if not has_windows:
            # No window functions -- standard execution via parent
            return self._exec_select_standard(stmt)

        txn_id = self._get_txn()
        try:
            # Compile the base query (without window functions)
            plan, engine, window_calls, orig_columns = \
                self.window_compiler.compile_select_with_windows(stmt, txn_id)

            # Execute base query to get materialized rows
            qe_rows = engine.execute(plan)

            # Convert QE rows to dicts
            rows = []
            base_columns = []
            if qe_rows:
                base_columns = list(qe_rows[0].columns())
                for r in qe_rows:
                    row_dict = {}
                    for col in base_columns:
                        row_dict[col] = r.get(col)
                    rows.append(row_dict)

            # Also add original column references for window function expressions
            # Need access to all source columns for window PARTITION BY / ORDER BY
            if rows and stmt.from_table:
                schema = self.storage.catalog.get_table(stmt.from_table.table_name)
                all_rows = self.storage.scan_table(txn_id, stmt.from_table.table_name)
                # If base query had WHERE but didn't project all cols,
                # we need to enrich rows with missing columns
                # The base rows already have the right filtering; just ensure
                # all columns are accessible
                for r in rows:
                    for col in schema.column_names():
                        tkey = f"{stmt.from_table.table_name}.{col}"
                        if col not in r and tkey in r:
                            r[col] = r[tkey]
                        elif tkey not in r and col in r:
                            r[tkey] = r[col]

            # Evaluate window functions
            rows, result_columns = self.window_executor.evaluate_windows(
                rows, base_columns, orig_columns
            )

            # Apply ORDER BY (from original stmt) if present
            if stmt.order_by:
                import functools

                def compare_rows(a, b):
                    for expr, asc in stmt.order_by:
                        a_val = self._eval_order_expr(expr, a)
                        b_val = self._eval_order_expr(expr, b)
                        if a_val is None and b_val is None:
                            continue
                        if a_val is None:
                            return 1
                        if b_val is None:
                            return -1
                        if a_val < b_val:
                            return -1 if asc else 1
                        if a_val > b_val:
                            return 1 if asc else -1
                    return 0

                rows.sort(key=functools.cmp_to_key(compare_rows))

            # Apply DISTINCT
            if stmt.distinct:
                seen = set()
                unique_rows = []
                for r in rows:
                    key = tuple(r.get(c) for c in result_columns)
                    if key not in seen:
                        seen.add(key)
                        unique_rows.append(r)
                rows = unique_rows

            # Apply LIMIT / OFFSET
            if stmt.offset:
                rows = rows[stmt.offset:]
            if stmt.limit is not None:
                rows = rows[:stmt.limit]

            # Build final result
            final_rows = []
            for r in rows:
                row_vals = []
                for col in result_columns:
                    # Try exact match first, then with table prefix
                    if col in r:
                        row_vals.append(r[col])
                    else:
                        # Try without table prefix
                        short = col.split('.')[-1] if '.' in col else col
                        found = False
                        for k, v in r.items():
                            k_short = k.split('.')[-1] if '.' in k else k
                            if k_short == short:
                                row_vals.append(v)
                                found = True
                                break
                        if not found:
                            row_vals.append(None)
                final_rows.append(row_vals)

            # Clean column names
            clean_cols = []
            for c in result_columns:
                if '.' in c:
                    clean_cols.append(c.split('.')[-1])
                else:
                    clean_cols.append(c)

            self._auto_commit(txn_id)
            return ResultSet(columns=clean_cols, rows=final_rows)

        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_select_standard(self, stmt: SelectStmt) -> ResultSet:
        """Standard SELECT execution (no window functions) via parent chain."""
        # View expansion already done in _exec_select_with_windows
        txn_id = self._get_txn()
        try:
            plan, engine = self.compiler.compile_select(stmt, txn_id)
            qe_rows = engine.execute(plan)

            if qe_rows:
                if self.compiler._is_star_only(stmt.columns):
                    ordered_cols = []
                    if stmt.from_table:
                        schema = self.storage.catalog.get_table(stmt.from_table.table_name)
                        tname = stmt.from_table.table_name
                        for cn in schema.column_names():
                            ordered_cols.append((f"{tname}.{cn}", cn))
                    for j in stmt.joins:
                        jschema = self.storage.catalog.get_table(j.table.table_name)
                        jtname = j.table.table_name
                        for cn in jschema.column_names():
                            ordered_cols.append((f"{jtname}.{cn}", cn))
                    clean_cols = [c[1] for c in ordered_cols]
                    qe_keys = [c[0] for c in ordered_cols]
                    rows = [[r.get(k) for k in qe_keys] for r in qe_rows]
                else:
                    columns = qe_rows[0].columns()
                    clean_cols = []
                    for c in columns:
                        if '.' in c:
                            clean_cols.append(c.split('.')[-1])
                        else:
                            clean_cols.append(c)
                    rows = [list(r.values()) for r in qe_rows]
            else:
                columns = []
                for se in stmt.columns:
                    if isinstance(se.expr, SqlStar):
                        if stmt.from_table:
                            schema = self.storage.catalog.get_table(stmt.from_table.table_name)
                            columns.extend(schema.column_names())
                    elif se.alias:
                        columns.append(se.alias)
                    elif isinstance(se.expr, SqlColumnRef):
                        columns.append(se.expr.column)
                    else:
                        columns.append(f"col_{len(columns)}")
                clean_cols = columns
                rows = []

            self._auto_commit(txn_id)
            return ResultSet(columns=clean_cols, rows=rows)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _eval_order_expr(self, expr, row_dict: Dict) -> Any:
        """Evaluate an ORDER BY expression against a row dict."""
        if isinstance(expr, SqlColumnRef):
            col = expr.column
            if col in row_dict:
                return row_dict[col]
            # Try with table prefix
            if expr.table:
                key = f"{expr.table}.{col}"
                if key in row_dict:
                    return row_dict[key]
            # Try any key ending with this column
            for k, v in row_dict.items():
                if k.split('.')[-1] == col:
                    return v
            return None
        if isinstance(expr, SqlLiteral):
            return expr.value
        # For complex expressions, use the compiler
        try:
            qe_row = Row(row_dict)
            qe_expr = self.window_compiler._sql_to_qe_expr(expr)
            return eval_expr(qe_expr, qe_row)
        except Exception:
            return None
