"""
C255: SQL Subqueries
Extends C254 (Set Operations) / C253 (CTEs) / ... / C247 (Mini Database)

Adds SQL subquery support:
- Scalar subqueries: SELECT (SELECT MAX(x) FROM t) in expressions
- IN subqueries: WHERE x IN (SELECT y FROM t2)
- NOT IN subqueries: WHERE x NOT IN (SELECT y FROM t2)
- EXISTS subqueries: WHERE EXISTS (SELECT 1 FROM t2 WHERE ...)
- NOT EXISTS subqueries: WHERE NOT EXISTS (SELECT ...)
- Correlated subqueries: subqueries referencing outer query columns
- Comparison subqueries: WHERE x > (SELECT AVG(y) FROM t2)
- Subqueries in SELECT list, WHERE, HAVING clauses
- Nested subqueries: subqueries inside subqueries
- Subqueries with JOINs, aggregates, CTEs, set operations
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import chain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C254_set_operations'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C253_common_table_expressions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C252_sql_window_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C251_sql_triggers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C250_sql_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from set_operations import (
    SetOpDB, SetOpLexer, SetOpParser, SetOpStmt, SET_OP_WORDS,
)

from mini_database import (
    ResultSet, DatabaseError, ParseError,
    Token, TokenType, KEYWORDS,
    SelectStmt, SelectExpr, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    SqlColumnRef, SqlLiteral, SqlBinOp, SqlComparison, SqlLogic,
    SqlIsNull, SqlFuncCall, SqlAggCall, SqlBetween, SqlInList,
    SqlCase, SqlStar,
)

from common_table_expressions import CTESelectStmt, CTEDef, UnionStmt
from transaction_manager import IsolationLevel


# =============================================================================
# Subquery AST Nodes
# =============================================================================

@dataclass
class SqlSubquery:
    """A subquery (SELECT ...) used as an expression.
    When used as a scalar subquery, must return exactly one row and one column.
    """
    stmt: Any  # SelectStmt | CTESelectStmt | SetOpStmt


@dataclass
class SqlExistsSubquery:
    """EXISTS (SELECT ...) -- returns true if subquery produces any rows."""
    stmt: Any  # SelectStmt | CTESelectStmt | SetOpStmt


@dataclass
class SqlInSubquery:
    """expr IN (SELECT ...) -- true if expr matches any row from subquery."""
    expr: Any       # SQL expression on left side
    subquery: Any   # SelectStmt | CTESelectStmt | SetOpStmt
    negated: bool = False  # True for NOT IN


# =============================================================================
# Subquery Lexer
# =============================================================================

class SubqueryLexer(SetOpLexer):
    """Lexer for subqueries. No changes needed -- SELECT and parens already lexed."""
    def __init__(self, sql: str):
        super().__init__(sql)


# =============================================================================
# Subquery Parser (extends SetOpParser)
# =============================================================================

class SubqueryParser(SetOpParser):
    """Parser extended with subquery support.

    Subqueries can appear:
    1. In expressions: (SELECT ...) as a scalar subquery
    2. After IN: x IN (SELECT ...)
    3. After EXISTS: EXISTS (SELECT ...)
    4. After comparison ops: x > (SELECT ...)
    """

    def __init__(self, tokens):
        super().__init__(tokens)

    def _is_subquery_start(self) -> bool:
        """Check if the next tokens start a subquery: ( SELECT ... or ( WITH ..."""
        if self._peek_type() != TokenType.LPAREN:
            return False
        # Look ahead past ( to see if SELECT or WITH follows
        saved = self.pos
        self.pos += 1  # skip (
        word = self._peek_word()
        self.pos = saved
        return word in ('select', 'with')

    def _parse_subquery_stmt(self) -> Any:
        """Parse the inner statement of a subquery (SELECT or WITH...SELECT or set ops)."""
        self.expect(TokenType.LPAREN)
        # Parse inner statement (could be CTE or plain SELECT)
        inner = super()._parse_statement()
        # Check for set operations after the inner select
        if isinstance(inner, (SelectStmt, CTESelectStmt)):
            inner = self._parse_set_operations(inner)
            if isinstance(inner, SetOpStmt) and isinstance(self._get_original_stmt(inner), CTESelectStmt):
                pass  # Already restructured by parent
        self.expect(TokenType.RPAREN)
        return inner

    def _get_original_stmt(self, setop):
        """Get the leftmost leaf of a set operation tree."""
        node = setop
        while isinstance(node.left, SetOpStmt):
            node = node.left
        return node.left

    # Override _parse_primary to handle (SELECT ...) and EXISTS (SELECT ...)
    def _parse_primary(self):
        """Extended to handle subquery expressions."""
        # EXISTS (SELECT ...)
        if self._peek_type() == TokenType.EXISTS:
            if self._is_exists_subquery():
                self._advance()  # consume EXISTS
                stmt = self._parse_subquery_stmt()
                return SqlExistsSubquery(stmt=stmt)

        # ( SELECT ... ) -- scalar subquery
        if self._is_subquery_start():
            stmt = self._parse_subquery_stmt()
            return SqlSubquery(stmt=stmt)

        return super()._parse_primary()

    def _is_exists_subquery(self) -> bool:
        """Check if EXISTS is followed by (SELECT ...)."""
        if self._peek_type() != TokenType.EXISTS:
            return False
        saved = self.pos
        self.pos += 1  # skip EXISTS
        result = self._is_subquery_start()
        self.pos = saved
        return result

    # Override _parse_comparison to handle IN (SELECT ...) and NOT IN (SELECT ...)
    def _parse_comparison(self):
        """Extended to handle IN subqueries and comparison with subqueries."""
        left = self._parse_addition()

        # NOT IN / NOT BETWEEN / NOT LIKE
        if self._peek_word() == 'not':
            saved = self.pos
            self._advance()  # consume NOT
            word2 = self._peek_word()
            if word2 == 'in':
                self._advance()  # consume IN
                if self._is_subquery_start():
                    stmt = self._parse_subquery_stmt()
                    return SqlInSubquery(expr=left, subquery=stmt, negated=True)
                else:
                    # Regular NOT IN (val1, val2, ...)
                    self.expect(TokenType.LPAREN)
                    vals = [self._parse_expr()]
                    while self.match(TokenType.COMMA):
                        vals.append(self._parse_expr())
                    self.expect(TokenType.RPAREN)
                    return SqlLogic(op='not', operands=[SqlInList(expr=left, values=vals)])
            elif word2 == 'between':
                low = self._parse_addition()
                if self._peek_word() != 'and':
                    raise ParseError("Expected AND in BETWEEN")
                self._advance()  # consume AND
                high = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlBetween(expr=left, low=low, high=high)])
            elif word2 == 'like':
                right = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlComparison(op='like', left=left, right=right)])
            elif word2 == 'exists':
                # NOT EXISTS handled at expression level, not here
                self.pos = saved
                return left
            else:
                self.pos = saved
                return left

        # IN (SELECT ...) or IN (val1, val2, ...)
        if self._peek_word() == 'in' or self._peek_type() == TokenType.IN:
            self._advance()  # consume IN
            if self._is_subquery_start():
                stmt = self._parse_subquery_stmt()
                return SqlInSubquery(expr=left, subquery=stmt)
            else:
                self.expect(TokenType.LPAREN)
                vals = [self._parse_expr()]
                while self.match(TokenType.COMMA):
                    vals.append(self._parse_expr())
                self.expect(TokenType.RPAREN)
                return SqlInList(expr=left, values=vals)

        # BETWEEN
        if self._peek_word() == 'between':
            self._advance()
            low = self._parse_addition()
            if self._peek_word() != 'and':
                raise ParseError("Expected AND in BETWEEN")
            self._advance()
            high = self._parse_addition()
            return SqlBetween(expr=left, low=low, high=high)

        # IS NULL / IS NOT NULL
        if self._peek_word() == 'is':
            self._advance()
            if self._peek_word() == 'not':
                self._advance()
                if self._peek_word() == 'null':
                    self._advance()
                    return SqlIsNull(expr=left, negated=True)
            elif self._peek_word() == 'null':
                self._advance()
                return SqlIsNull(expr=left, negated=False)

        # LIKE
        if self._peek_word() == 'like':
            self._advance()
            right = self._parse_addition()
            return SqlComparison(op='like', left=left, right=right)

        # Standard comparison operators: =, !=, <, <=, >, >=
        # Right side could be a subquery
        op_map = {
            TokenType.EQ: '=', TokenType.NE: '!=',
            TokenType.LT: '<', TokenType.LE: '<=',
            TokenType.GT: '>', TokenType.GE: '>=',
        }
        tt = self._peek_type()
        if tt in op_map:
            op = op_map[tt]
            self._advance()
            # Check for subquery on right side
            if self._is_subquery_start():
                stmt = self._parse_subquery_stmt()
                right = SqlSubquery(stmt=stmt)
            else:
                right = self._parse_addition()
            return SqlComparison(op=op, left=left, right=right)

        return left

    def _parse_not(self):
        """Override to handle NOT EXISTS."""
        if self._peek_word() == 'not':
            saved = self.pos
            self._advance()  # consume NOT
            if self._peek_type() == TokenType.EXISTS and self._is_exists_subquery():
                self._advance()  # consume EXISTS
                stmt = self._parse_subquery_stmt()
                return SqlLogic(op='not', operands=[SqlExistsSubquery(stmt=stmt)])
            self.pos = saved
        return super()._parse_not()


# =============================================================================
# Subquery DB (extends SetOpDB)
# =============================================================================

class SubqueryDB(SetOpDB):
    """SetOpDB extended with subquery support.

    Subqueries are evaluated by running them as mini-SELECTs against this DB.
    Correlated subqueries receive outer row context for column resolution.
    """

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        self._outer_row_stack = []  # Stack of outer row contexts for correlated subqueries

    def execute(self, sql: str) -> ResultSet:
        stmts = self._parse_subquery(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_subquery_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List[ResultSet]:
        stmts = self._parse_subquery(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_subquery_stmt(stmt))
        return results

    def _parse_subquery(self, sql: str) -> List[Any]:
        lexer = SubqueryLexer(sql)
        parser = SubqueryParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _execute_subquery_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling subqueries within."""
        if isinstance(stmt, SetOpStmt):
            return self._exec_set_operation(stmt)
        if isinstance(stmt, CTESelectStmt):
            if isinstance(stmt.main_stmt, SetOpStmt):
                return self._exec_cte_with_set_ops(stmt)
            return self._exec_with_ctes(stmt)
        if isinstance(stmt, SelectStmt):
            return self._exec_select_with_cte_context(stmt)
        return self._execute_trigger_stmt(stmt)

    # =========================================================================
    # Subquery Expression Evaluation
    # =========================================================================

    def _eval_subquery_expr(self, node, outer_row=None) -> Any:
        """Evaluate a SQL expression that may contain subqueries.

        outer_row: dict mapping column names to values from outer query
        Returns: Python value
        """
        if isinstance(node, SqlSubquery):
            return self._eval_scalar_subquery(node.stmt, outer_row)

        if isinstance(node, SqlExistsSubquery):
            return self._eval_exists_subquery(node.stmt, outer_row)

        if isinstance(node, SqlInSubquery):
            left_val = self._eval_subquery_expr(node.expr, outer_row)
            result = self._eval_in_subquery(left_val, node.subquery, outer_row)
            if node.negated:
                return not result
            return result

        if isinstance(node, SqlLiteral):
            return node.value

        if isinstance(node, SqlColumnRef):
            if outer_row is not None:
                # Try to resolve from outer row
                col = node.column
                tbl = node.table
                if tbl:
                    key = f"{tbl}.{col}"
                    if key in outer_row:
                        return outer_row[key]
                if col in outer_row:
                    return outer_row[col]
                # Try with table prefix
                for k, v in outer_row.items():
                    if '.' in k and k.split('.', 1)[1] == col:
                        return v
            return None  # unresolved column ref

        if isinstance(node, SqlBinOp):
            left = self._eval_subquery_expr(node.left, outer_row)
            right = self._eval_subquery_expr(node.right, outer_row)
            if node.op == '+':
                return (left or 0) + (right or 0)
            if node.op == '-':
                return (left or 0) - (right or 0)
            if node.op == '*':
                return (left or 0) * (right or 0)
            if node.op == '/':
                r = right or 0
                if r == 0:
                    return None
                return (left or 0) / r
            return None

        if isinstance(node, SqlComparison):
            left = self._eval_subquery_expr(node.left, outer_row)
            right = self._eval_subquery_expr(node.right, outer_row)
            if left is None or right is None:
                return None  # NULL comparison
            ops = {
                '=': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '<=': lambda a, b: a <= b,
                '>': lambda a, b: a > b,
                '>=': lambda a, b: a >= b,
                'like': lambda a, b: self._like_match(str(a), str(b)),
            }
            return ops.get(node.op, lambda a, b: None)(left, right)

        if isinstance(node, SqlLogic):
            if node.op == 'not':
                val = self._eval_subquery_expr(node.operands[0], outer_row)
                if val is None:
                    return None
                return not val
            if node.op == 'and':
                result = True
                for operand in node.operands:
                    val = self._eval_subquery_expr(operand, outer_row)
                    if val is None:
                        result = None
                    elif not val:
                        return False
                return result
            if node.op == 'or':
                result = False
                for operand in node.operands:
                    val = self._eval_subquery_expr(operand, outer_row)
                    if val is None:
                        result = None
                    elif val:
                        return True
                return result

        if isinstance(node, SqlIsNull):
            val = self._eval_subquery_expr(node.expr, outer_row)
            if node.negated:
                return val is not None
            return val is None

        if isinstance(node, SqlBetween):
            val = self._eval_subquery_expr(node.expr, outer_row)
            low = self._eval_subquery_expr(node.low, outer_row)
            high = self._eval_subquery_expr(node.high, outer_row)
            if val is None or low is None or high is None:
                return None
            return low <= val <= high

        if isinstance(node, SqlInList):
            val = self._eval_subquery_expr(node.expr, outer_row)
            if val is None:
                return None
            for item in node.values:
                item_val = self._eval_subquery_expr(item, outer_row)
                if item_val == val:
                    return True
            return False

        if isinstance(node, SqlAggCall):
            # Aggregate in subquery context -- should already be resolved
            return None

        if isinstance(node, SqlFuncCall):
            # Function calls -- evaluate args
            args = [self._eval_subquery_expr(a, outer_row) for a in node.args]
            func = node.func_name.lower()
            if func == 'coalesce':
                for a in args:
                    if a is not None:
                        return a
                return None
            if func == 'abs':
                return abs(args[0]) if args[0] is not None else None
            if func == 'upper':
                return str(args[0]).upper() if args[0] is not None else None
            if func == 'lower':
                return str(args[0]).lower() if args[0] is not None else None
            if func == 'length':
                return len(str(args[0])) if args[0] is not None else None
            return None

        if isinstance(node, SqlCase):
            for cond, result in node.whens:
                cond_val = self._eval_subquery_expr(cond, outer_row)
                if cond_val:
                    return self._eval_subquery_expr(result, outer_row)
            if node.else_result:
                return self._eval_subquery_expr(node.else_result, outer_row)
            return None

        return None

    def _like_match(self, text: str, pattern: str) -> bool:
        """Simple LIKE pattern matching (% = any, _ = single char)."""
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
        return bool(re.match(regex, text, re.IGNORECASE))

    def _eval_scalar_subquery(self, stmt, outer_row=None) -> Any:
        """Execute a subquery and return its scalar value (one row, one column)."""
        result = self._run_subquery(stmt, outer_row)
        if len(result.rows) == 0:
            return None
        if len(result.rows) > 1:
            raise DatabaseError("Scalar subquery returned more than one row")
        if len(result.columns) > 1:
            raise DatabaseError("Scalar subquery returned more than one column")
        return result.rows[0][0]

    def _eval_exists_subquery(self, stmt, outer_row=None) -> bool:
        """Execute a subquery and return True if it produces any rows."""
        result = self._run_subquery(stmt, outer_row)
        return len(result.rows) > 0

    def _eval_in_subquery(self, left_val, stmt, outer_row=None) -> bool:
        """Execute a subquery and check if left_val is in the result."""
        if left_val is None:
            return False  # NULL IN (...) is false/unknown
        result = self._run_subquery(stmt, outer_row)
        if len(result.columns) != 1:
            raise DatabaseError("IN subquery must return exactly one column")
        for row in result.rows:
            if row[0] == left_val:
                return True
        return False

    def _run_subquery(self, stmt, outer_row=None) -> ResultSet:
        """Execute a subquery statement, optionally with outer row context.

        For correlated subqueries (outer_row != None), we use our own row-by-row
        evaluator so that outer column references resolve correctly.
        """
        if outer_row is not None:
            self._outer_row_stack.append(outer_row)
        try:
            if isinstance(stmt, SetOpStmt):
                return self._exec_set_operation(stmt)
            if isinstance(stmt, CTESelectStmt):
                if isinstance(stmt.main_stmt, SetOpStmt):
                    return self._exec_cte_with_set_ops(stmt)
                return self._exec_with_ctes(stmt)
            if isinstance(stmt, SelectStmt):
                # For correlated subqueries, use row-by-row evaluator
                # so outer column references resolve correctly
                if outer_row is not None and self._has_outer_refs(stmt, outer_row):
                    return self._exec_correlated_subquery(stmt, outer_row)
                return self._exec_select_with_cte_context(stmt)
            return self._execute_subquery_stmt(stmt)
        finally:
            if outer_row is not None:
                self._outer_row_stack.pop()

    def _has_outer_refs(self, stmt: SelectStmt, outer_row: Dict) -> bool:
        """Check if a subquery references columns from the outer query."""
        # Check WHERE clause for references to outer table aliases
        if stmt.where and self._expr_refs_outer(stmt.where, outer_row):
            return True
        if stmt.having and self._expr_refs_outer(stmt.having, outer_row):
            return True
        for col in stmt.columns:
            if self._expr_refs_outer(col.expr, outer_row):
                return True
        return False

    def _expr_refs_outer(self, expr, outer_row: Dict) -> bool:
        """Check if an expression references columns available in outer_row."""
        if isinstance(expr, SqlColumnRef):
            if expr.table:
                key = f"{expr.table}.{expr.column}"
                return key in outer_row
            return False  # Unqualified refs might be ambiguous
        if isinstance(expr, SqlComparison):
            return self._expr_refs_outer(expr.left, outer_row) or self._expr_refs_outer(expr.right, outer_row)
        if isinstance(expr, SqlLogic):
            return any(self._expr_refs_outer(o, outer_row) for o in expr.operands)
        if isinstance(expr, SqlBinOp):
            return self._expr_refs_outer(expr.left, outer_row) or self._expr_refs_outer(expr.right, outer_row)
        if isinstance(expr, (SqlSubquery, SqlExistsSubquery, SqlInSubquery)):
            return True  # Nested subqueries always go through correlated path
        return False

    def _exec_correlated_subquery(self, stmt: SelectStmt, outer_row: Dict) -> ResultSet:
        """Execute a correlated subquery using row-by-row evaluation with outer context."""
        # Get source rows from inner FROM table
        source_rows = self._get_source_rows(stmt)

        # Enrich each source row with outer_row context for correlation
        enriched = []
        for row in source_rows:
            combined = dict(outer_row)
            combined.update(row)
            enriched.append(combined)

        # Filter with WHERE
        if stmt.where:
            filtered = []
            for row in enriched:
                val = self._eval_subquery_expr(stmt.where, row)
                if val:
                    filtered.append(row)
            enriched = filtered

        # GROUP BY
        if stmt.group_by:
            groups = self._do_groupby(enriched, stmt.group_by, stmt)
            if stmt.having:
                groups = {k: v for k, v in groups.items()
                          if self._eval_having_subquery(stmt.having, k, v, stmt)}
            result_rows, result_cols = self._project_grouped(groups, stmt)
        else:
            has_aggs = any(self._expr_has_agg(col.expr) for col in stmt.columns)
            if has_aggs:
                groups = {(): enriched}
                result_rows, result_cols = self._project_grouped(groups, stmt)
            else:
                result_rows, result_cols = self._project_ungrouped(enriched, stmt)

        # DISTINCT
        if stmt.distinct:
            seen = set()
            deduped = []
            for row in result_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    deduped.append(row)
            result_rows = deduped

        # ORDER BY
        if stmt.order_by:
            result_rows = self._sort_rows(result_rows, result_cols, stmt.order_by)

        # LIMIT/OFFSET
        if stmt.offset:
            result_rows = result_rows[stmt.offset:]
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_cols, rows=result_rows)

    # =========================================================================
    # Override SELECT execution to handle subqueries in expressions
    # =========================================================================

    def _exec_select_standard(self, stmt: SelectStmt) -> ResultSet:
        """Override: intercept subquery-containing SELECTs before the compiler."""
        if self._stmt_has_subqueries(stmt):
            return self._exec_select_with_subqueries(stmt)
        return super()._exec_select_standard(stmt)

    def _exec_select_from_cte(self, stmt: SelectStmt) -> ResultSet:
        """Override: handle subqueries in CTE-sourced SELECTs."""
        if self._stmt_has_subqueries(stmt):
            return self._exec_select_with_subqueries(stmt)
        return super()._exec_select_from_cte(stmt)

    def _eval_cte_expr(self, expr, row, alias) -> Any:
        """Override: handle subqueries in CTE WHERE expressions."""
        if self._expr_has_subquery(expr):
            # Build context from CTE row
            ctx = dict(row)
            for k, v in row.items():
                ctx[f"{alias}.{k}"] = v
            return self._eval_subquery_expr(expr, ctx)
        return super()._eval_cte_expr(expr, row, alias)

    def _stmt_has_subqueries(self, stmt) -> bool:
        """Check if a SELECT statement contains any subquery expressions."""
        # Check WHERE
        if stmt.where and self._expr_has_subquery(stmt.where):
            return True
        # Check HAVING
        if stmt.having and self._expr_has_subquery(stmt.having):
            return True
        # Check SELECT list
        for col in stmt.columns:
            if self._expr_has_subquery(col.expr):
                return True
        return False

    def _expr_has_subquery(self, expr) -> bool:
        """Recursively check if an expression contains subqueries."""
        if isinstance(expr, (SqlSubquery, SqlExistsSubquery, SqlInSubquery)):
            return True
        if isinstance(expr, SqlComparison):
            return self._expr_has_subquery(expr.left) or self._expr_has_subquery(expr.right)
        if isinstance(expr, SqlLogic):
            return any(self._expr_has_subquery(o) for o in expr.operands)
        if isinstance(expr, SqlBinOp):
            return self._expr_has_subquery(expr.left) or self._expr_has_subquery(expr.right)
        if isinstance(expr, SqlIsNull):
            return self._expr_has_subquery(expr.expr)
        if isinstance(expr, SqlBetween):
            return (self._expr_has_subquery(expr.expr) or
                    self._expr_has_subquery(expr.low) or
                    self._expr_has_subquery(expr.high))
        if isinstance(expr, SqlInList):
            return (self._expr_has_subquery(expr.expr) or
                    any(self._expr_has_subquery(v) for v in expr.values))
        if isinstance(expr, SqlFuncCall):
            return any(self._expr_has_subquery(a) for a in expr.args)
        if isinstance(expr, SqlCase):
            for cond, result in expr.whens:
                if self._expr_has_subquery(cond) or self._expr_has_subquery(result):
                    return True
            if expr.else_result and self._expr_has_subquery(expr.else_result):
                return True
        if isinstance(expr, SelectExpr):
            return self._expr_has_subquery(expr.expr)
        return False

    def _exec_select_with_subqueries(self, stmt: SelectStmt) -> ResultSet:
        """Execute a SELECT with subquery expressions (row-by-row evaluation)."""
        return self._exec_select_full_subquery(stmt)

    def _agg_display_name(self, agg: SqlAggCall) -> str:
        """Generate display name for aggregate."""
        if isinstance(agg.arg, SqlStar):
            return f"{agg.func}(*)"
        if isinstance(agg.arg, SqlColumnRef):
            return f"{agg.func}({agg.arg.column})"
        return f"{agg.func}(expr)"

    def _exec_select_full_subquery(self, stmt: SelectStmt) -> ResultSet:
        """Execute SELECT with subqueries in WHERE/HAVING (row-by-row evaluation)."""
        # Get all rows from the source tables
        source_rows = self._get_source_rows(stmt)

        # Filter with WHERE (subquery-aware)
        if stmt.where:
            filtered = []
            for row in source_rows:
                val = self._eval_subquery_expr(stmt.where, row)
                if val:
                    filtered.append(row)
            source_rows = filtered

        # GROUP BY
        if stmt.group_by:
            groups = self._do_groupby(source_rows, stmt.group_by, stmt)
            # HAVING with subquery support
            if stmt.having:
                groups = {k: v for k, v in groups.items()
                          if self._eval_having_subquery(stmt.having, k, v, stmt)}
            # Project grouped results
            result_rows, result_cols = self._project_grouped(groups, stmt)
        else:
            # Check if there are aggregates without GROUP BY
            has_aggs = any(self._expr_has_agg(col.expr) for col in stmt.columns)
            if has_aggs:
                groups = {(): source_rows}
                if stmt.having:
                    groups = {k: v for k, v in groups.items()
                              if self._eval_having_subquery(stmt.having, k, v, stmt)}
                result_rows, result_cols = self._project_grouped(groups, stmt)
            else:
                # No grouping -- project each row
                result_rows, result_cols = self._project_ungrouped(source_rows, stmt)

        # DISTINCT
        if stmt.distinct:
            seen = set()
            deduped = []
            for row in result_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    deduped.append(row)
            result_rows = deduped

        # ORDER BY
        if stmt.order_by:
            result_rows = self._sort_rows(result_rows, result_cols, stmt.order_by)

        # OFFSET
        if stmt.offset:
            result_rows = result_rows[stmt.offset:]

        # LIMIT
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_cols, rows=result_rows)

    def _get_source_rows(self, stmt: SelectStmt) -> List[Dict[str, Any]]:
        """Get all rows from FROM table and JOINs as dicts."""
        rows = []

        if stmt.from_table is None:
            # No FROM -- single row with no columns (for SELECT (SELECT ...))
            return [{}]

        table_name = stmt.from_table.table_name
        alias = stmt.from_table.alias or table_name

        # Check CTE tables first
        if hasattr(self, '_cte_tables') and table_name.lower() in self._cte_tables:
            cte_rows = self._cte_tables[table_name.lower()]
            cte_cols = self._cte_columns.get(table_name.lower(), [])
            for cte_row in cte_rows:
                row = {}
                for col_name in cte_cols:
                    val = cte_row.get(col_name)
                    row[col_name] = val
                    row[f"{alias}.{col_name}"] = val
                rows.append(row)
        else:
            # Regular table scan
            txn_id = self._get_txn()
            try:
                schema = self.storage.catalog.get_table(table_name)
                col_names = schema.column_names()
                all_rows = self.storage.scan_table(txn_id, table_name)
                for rowid, row_data in all_rows:
                    row = {}
                    for cn in col_names:
                        val = row_data.get(cn)
                        row[cn] = val
                        row[f"{alias}.{cn}"] = val
                        row[f"{table_name}.{cn}"] = val
                    rows.append(row)
                self._auto_commit(txn_id)
            except Exception:
                self._auto_abort(txn_id)
                raise

        # Process JOINs
        for join in stmt.joins:
            rows = self._do_join(rows, join, stmt)

        return rows

    def _do_join(self, left_rows: List[Dict], join: JoinClause,
                 stmt: SelectStmt) -> List[Dict]:
        """Execute a JOIN, returning combined rows."""
        right_table = join.table.table_name
        right_alias = join.table.alias or right_table

        # Get right-side rows
        right_rows = []
        if hasattr(self, '_cte_tables') and right_table.lower() in self._cte_tables:
            cte_rows = self._cte_tables[right_table.lower()]
            cte_cols = self._cte_columns.get(right_table.lower(), [])
            for cte_row in cte_rows:
                row = {}
                for col_name in cte_cols:
                    val = cte_row.get(col_name)
                    row[col_name] = val
                    row[f"{right_alias}.{col_name}"] = val
                    row[f"{right_table}.{col_name}"] = val
                right_rows.append(row)
        else:
            txn_id = self._get_txn()
            try:
                schema = self.storage.catalog.get_table(right_table)
                col_names = schema.column_names()
                all_rows = self.storage.scan_table(txn_id, right_table)
                for rowid, row_data in all_rows:
                    row = {}
                    for cn in col_names:
                        val = row_data.get(cn)
                        row[cn] = val
                        row[f"{right_alias}.{cn}"] = val
                        row[f"{right_table}.{cn}"] = val
                    right_rows.append(row)
                self._auto_commit(txn_id)
            except Exception:
                self._auto_abort(txn_id)
                raise

        # Nested loop join with subquery-aware condition evaluation
        result = []
        join_type = join.join_type.upper() if join.join_type else 'INNER'

        for lrow in left_rows:
            matched = False
            for rrow in right_rows:
                combined = {**lrow, **rrow}
                if join.condition:
                    cond_val = self._eval_subquery_expr(join.condition, combined)
                    if cond_val:
                        result.append(combined)
                        matched = True
                else:
                    result.append(combined)
                    matched = True

            if not matched and join_type in ('LEFT', 'LEFT OUTER'):
                # Add left row with NULLs for right side
                combined = dict(lrow)
                result.append(combined)

        return result

    def _do_groupby(self, rows, group_by, stmt) -> Dict[tuple, List[Dict]]:
        """Group rows by GROUP BY expressions."""
        groups = {}
        for row in rows:
            key = tuple(self._eval_subquery_expr(expr, row) for expr in group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
        return groups

    def _eval_having_subquery(self, having, group_key, group_rows, stmt) -> bool:
        """Evaluate HAVING clause with subquery and aggregate support."""
        # Build a context row with group key and aggregates
        ctx = {}
        if stmt.group_by:
            for i, expr in enumerate(stmt.group_by):
                if isinstance(expr, SqlColumnRef):
                    ctx[expr.column] = group_key[i]
                    if expr.table:
                        ctx[f"{expr.table}.{expr.column}"] = group_key[i]

        # Evaluate HAVING expression with aggregate resolution
        return self._eval_having_expr(having, ctx, group_rows)

    def _eval_having_expr(self, expr, ctx, group_rows) -> Any:
        """Evaluate a HAVING expression that may contain aggregates and subqueries."""
        if isinstance(expr, SqlAggCall):
            return self._compute_agg(expr, group_rows)

        if isinstance(expr, SqlSubquery):
            return self._eval_scalar_subquery(expr.stmt, ctx)

        if isinstance(expr, SqlExistsSubquery):
            return self._eval_exists_subquery(expr.stmt, ctx)

        if isinstance(expr, SqlInSubquery):
            left_val = self._eval_having_expr(expr.expr, ctx, group_rows)
            result = self._eval_in_subquery(left_val, expr.subquery, ctx)
            return not result if expr.negated else result

        if isinstance(expr, SqlComparison):
            left = self._eval_having_expr(expr.left, ctx, group_rows)
            right = self._eval_having_expr(expr.right, ctx, group_rows)
            if left is None or right is None:
                return None
            ops = {
                '=': lambda a, b: a == b, '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
                '>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
            }
            return ops.get(expr.op, lambda a, b: None)(left, right)

        if isinstance(expr, SqlLogic):
            if expr.op == 'not':
                val = self._eval_having_expr(expr.operands[0], ctx, group_rows)
                return not val if val is not None else None
            if expr.op == 'and':
                return all(self._eval_having_expr(o, ctx, group_rows) for o in expr.operands)
            if expr.op == 'or':
                return any(self._eval_having_expr(o, ctx, group_rows) for o in expr.operands)

        if isinstance(expr, SqlColumnRef):
            col = expr.column
            if col in ctx:
                return ctx[col]
            if expr.table and f"{expr.table}.{col}" in ctx:
                return ctx[f"{expr.table}.{col}"]
            return None

        if isinstance(expr, SqlLiteral):
            return expr.value

        if isinstance(expr, SqlBinOp):
            left = self._eval_having_expr(expr.left, ctx, group_rows)
            right = self._eval_having_expr(expr.right, ctx, group_rows)
            ops = {'+': lambda a, b: (a or 0) + (b or 0),
                   '-': lambda a, b: (a or 0) - (b or 0),
                   '*': lambda a, b: (a or 0) * (b or 0),
                   '/': lambda a, b: (a or 0) / (b or 1)}
            return ops.get(expr.op, lambda a, b: None)(left, right)

        return self._eval_subquery_expr(expr, ctx)

    def _compute_agg(self, agg: SqlAggCall, rows: List[Dict]) -> Any:
        """Compute an aggregate function over a group of rows."""
        func = agg.func.lower()
        if func == 'count':
            if agg.arg is None or isinstance(agg.arg, SqlStar):
                return len(rows)
            vals = [self._resolve_col(agg.arg, r) for r in rows]
            vals = [v for v in vals if v is not None]
            if agg.distinct:
                vals = list(set(vals))
            return len(vals)
        elif func == 'sum':
            vals = [self._resolve_col(agg.arg, r) for r in rows]
            vals = [v for v in vals if v is not None]
            if agg.distinct:
                vals = list(set(vals))
            return sum(vals) if vals else 0
        elif func == 'avg':
            vals = [self._resolve_col(agg.arg, r) for r in rows]
            vals = [v for v in vals if v is not None]
            if agg.distinct:
                vals = list(set(vals))
            return sum(vals) / len(vals) if vals else None
        elif func == 'min':
            vals = [self._resolve_col(agg.arg, r) for r in rows]
            vals = [v for v in vals if v is not None]
            return min(vals) if vals else None
        elif func == 'max':
            vals = [self._resolve_col(agg.arg, r) for r in rows]
            vals = [v for v in vals if v is not None]
            return max(vals) if vals else None
        return None

    def _resolve_col(self, expr, row: Dict) -> Any:
        """Resolve a column expression against a row dict."""
        if isinstance(expr, SqlColumnRef):
            col = expr.column
            tbl = expr.table
            if tbl:
                key = f"{tbl}.{col}"
                if key in row:
                    return row[key]
            if col in row:
                return row[col]
            # Try all table-qualified variants
            for k, v in row.items():
                if '.' in k and k.split('.', 1)[1] == col:
                    return v
            return None
        if isinstance(expr, SqlLiteral):
            return expr.value
        return self._eval_subquery_expr(expr, row)

    def _project_grouped(self, groups, stmt) -> Tuple[List[List], List[str]]:
        """Project grouped results into result rows with column names."""
        result_rows = []
        result_cols = []

        # Determine column names
        for i, col in enumerate(stmt.columns):
            if col.alias:
                result_cols.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                result_cols.append(col.expr.column)
            elif isinstance(col.expr, SqlAggCall):
                result_cols.append(self._agg_display_name(col.expr))
            elif isinstance(col.expr, SqlSubquery):
                result_cols.append(f"col_{i}")
            else:
                result_cols.append(f"col_{i}")

        for group_key, group_rows in groups.items():
            row_vals = []
            # Build context from group key
            ctx = {}
            if stmt.group_by:
                for idx, gexpr in enumerate(stmt.group_by):
                    if isinstance(gexpr, SqlColumnRef):
                        ctx[gexpr.column] = group_key[idx]
                        if gexpr.table:
                            ctx[f"{gexpr.table}.{gexpr.column}"] = group_key[idx]
            # Also add first row's data for non-group columns
            if group_rows:
                for k, v in group_rows[0].items():
                    if k not in ctx:
                        ctx[k] = v

            for col in stmt.columns:
                val = self._eval_select_expr(col.expr, ctx, group_rows)
                row_vals.append(val)
            result_rows.append(row_vals)

        return result_rows, result_cols

    def _project_ungrouped(self, rows, stmt) -> Tuple[List[List], List[str]]:
        """Project ungrouped results."""
        result_cols = []
        is_star = False

        for i, col in enumerate(stmt.columns):
            if isinstance(col.expr, SqlStar):
                is_star = True
                # Get column names from first row or schema
                if rows:
                    # Use non-qualified column names
                    seen = set()
                    for k in rows[0]:
                        if '.' not in k and k not in seen:
                            result_cols.append(k)
                            seen.add(k)
                elif stmt.from_table:
                    schema = self.storage.catalog.get_table(stmt.from_table.table_name)
                    result_cols = list(schema.column_names())
            elif col.alias:
                result_cols.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                result_cols.append(col.expr.column)
            elif isinstance(col.expr, SqlSubquery):
                result_cols.append(f"col_{i}")
            else:
                result_cols.append(f"col_{i}")

        result_rows = []
        for row in rows:
            if is_star:
                vals = [row.get(c) for c in result_cols]
            else:
                vals = []
                for col in stmt.columns:
                    vals.append(self._eval_select_expr(col.expr, row, [row]))
            result_rows.append(vals)

        return result_rows, result_cols

    def _eval_select_expr(self, expr, ctx: Dict, group_rows: List[Dict]) -> Any:
        """Evaluate a SELECT expression that may contain aggregates or subqueries."""
        if isinstance(expr, SqlAggCall):
            return self._compute_agg(expr, group_rows)
        if isinstance(expr, SqlSubquery):
            return self._eval_scalar_subquery(expr.stmt, ctx)
        if isinstance(expr, SqlExistsSubquery):
            return self._eval_exists_subquery(expr.stmt, ctx)
        if isinstance(expr, SqlBinOp):
            left = self._eval_select_expr(expr.left, ctx, group_rows)
            right = self._eval_select_expr(expr.right, ctx, group_rows)
            ops = {'+': lambda a, b: (a or 0) + (b or 0),
                   '-': lambda a, b: (a or 0) - (b or 0),
                   '*': lambda a, b: (a or 0) * (b or 0),
                   '/': lambda a, b: (a or 0) / (b or 1)}
            return ops.get(expr.op, lambda a, b: None)(left, right)
        if isinstance(expr, SqlStar):
            return None  # Handled by is_star path
        if isinstance(expr, SqlFuncCall):
            args = [self._eval_select_expr(a, ctx, group_rows) for a in expr.args]
            return self._eval_func(expr.func_name, args)
        if isinstance(expr, SqlCase):
            for cond, result in expr.whens:
                cond_val = self._eval_select_expr(cond, ctx, group_rows)
                if cond_val:
                    return self._eval_select_expr(result, ctx, group_rows)
            if expr.else_result:
                return self._eval_select_expr(expr.else_result, ctx, group_rows)
            return None
        return self._eval_subquery_expr(expr, ctx)

    def _eval_func(self, name: str, args: List) -> Any:
        """Evaluate a scalar function."""
        func = name.lower()
        if func == 'coalesce':
            for a in args:
                if a is not None:
                    return a
            return None
        if func == 'abs':
            return abs(args[0]) if args and args[0] is not None else None
        if func == 'upper':
            return str(args[0]).upper() if args and args[0] is not None else None
        if func == 'lower':
            return str(args[0]).lower() if args and args[0] is not None else None
        if func == 'length':
            return len(str(args[0])) if args and args[0] is not None else None
        return None

    def _sort_rows(self, rows, columns, order_by):
        """Sort result rows by ORDER BY clause."""
        import functools

        def resolve(expr, row_vals):
            if isinstance(expr, SqlColumnRef):
                col = expr.column
                if col in columns:
                    return row_vals[columns.index(col)]
                for i, c in enumerate(columns):
                    if c.lower() == col.lower():
                        return row_vals[i]
                return None
            if isinstance(expr, SqlLiteral):
                if isinstance(expr.value, (int, float)):
                    idx = int(expr.value) - 1
                    if 0 <= idx < len(columns):
                        return row_vals[idx]
                return expr.value
            return None

        def compare(a, b):
            for expr, asc in order_by:
                a_val = resolve(expr, a)
                b_val = resolve(expr, b)
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

        return sorted(rows, key=functools.cmp_to_key(compare))

    def _expr_has_agg(self, expr) -> bool:
        """Check if expression contains an aggregate function."""
        if isinstance(expr, SqlAggCall):
            return True
        if isinstance(expr, SqlBinOp):
            return self._expr_has_agg(expr.left) or self._expr_has_agg(expr.right)
        if isinstance(expr, SqlFuncCall):
            return any(self._expr_has_agg(a) for a in expr.args)
        if isinstance(expr, SqlCase):
            for c, r in expr.whens:
                if self._expr_has_agg(c) or self._expr_has_agg(r):
                    return True
            if expr.else_result and self._expr_has_agg(expr.else_result):
                return True
        return False

    # =========================================================================
    # Override UPDATE/DELETE for subquery support in WHERE
    # =========================================================================

    def _exec_update(self, stmt: UpdateStmt) -> ResultSet:
        """Override to support subqueries in UPDATE WHERE clause."""
        if stmt.where and self._expr_has_subquery(stmt.where):
            return self._exec_update_with_subqueries(stmt)
        return super()._exec_update(stmt)

    def _exec_update_with_subqueries(self, stmt: UpdateStmt) -> ResultSet:
        """Execute UPDATE with subquery-aware WHERE evaluation."""
        txn_id = self._get_txn()
        try:
            schema = self.storage.catalog.get_table(stmt.table_name)
            col_names = schema.column_names()
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0

            for rowid, row_data in all_rows:
                # Build row context
                ctx = {}
                for cn in col_names:
                    ctx[cn] = row_data.get(cn)
                    ctx[f"{stmt.table_name}.{cn}"] = row_data.get(cn)

                if self._eval_subquery_expr(stmt.where, ctx):
                    new_data = dict(row_data)
                    for col_name, val_expr in stmt.assignments:
                        new_data[col_name] = self._eval_subquery_expr(val_expr, ctx)
                    self.storage.update_row(txn_id, stmt.table_name, rowid, new_data)
                    count += 1

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"Updated {count} row(s)")
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_delete(self, stmt: DeleteStmt) -> ResultSet:
        """Override to support subqueries in DELETE WHERE clause."""
        if stmt.where and self._expr_has_subquery(stmt.where):
            return self._exec_delete_with_subqueries(stmt)
        return super()._exec_delete(stmt)

    def _exec_delete_with_subqueries(self, stmt: DeleteStmt) -> ResultSet:
        """Execute DELETE with subquery-aware WHERE evaluation."""
        txn_id = self._get_txn()
        try:
            schema = self.storage.catalog.get_table(stmt.table_name)
            col_names = schema.column_names()
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0

            to_delete = []
            for rowid, row_data in all_rows:
                ctx = {}
                for cn in col_names:
                    ctx[cn] = row_data.get(cn)
                    ctx[f"{stmt.table_name}.{cn}"] = row_data.get(cn)

                if self._eval_subquery_expr(stmt.where, ctx):
                    to_delete.append(rowid)
                    count += 1

            for rowid in to_delete:
                self.storage.delete_row(txn_id, stmt.table_name, rowid)

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"Deleted {count} row(s)")
        except Exception:
            self._auto_abort(txn_id)
            raise
