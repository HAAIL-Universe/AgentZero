"""
C266: SQL Subqueries
Extends C265 (Built-in Functions) with comprehensive subquery support.

Subquery types:
- Scalar subqueries: (SELECT MAX(price) FROM items)
- Subqueries in WHERE/HAVING: WHERE x IN (SELECT ...), WHERE x > (SELECT ...)
- EXISTS/NOT EXISTS: WHERE EXISTS (SELECT ...)
- Subqueries in FROM (derived tables): FROM (SELECT ...) AS alias
- Correlated subqueries: WHERE x > (SELECT AVG(y) FROM t2 WHERE t2.id = t1.id)
- ALL/ANY/SOME: WHERE x > ALL (SELECT ...)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C265_builtin_functions')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C264_window_functions')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C263_ctas')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from builtin_functions import (
    BuiltinDB, BuiltinParser, parse_builtin_sql, parse_builtin_sql_multi,
    builtin_eval_expr, _builtin_apply, _UNKNOWN_FUNC, _do_cast,
    BUILTIN_FUNC_NAMES, SqlCast, _KEYWORD_FUNC_TOKENS,
)
from window_functions import (
    WindowDB, WindowParser, parse_window_sql,
    SqlWindowFunc, WindowSpec, FrameBound, NamedWindow,
)
from mini_database import (
    ResultSet, ParseError, CompileError,
    SqlFuncCall, SqlLiteral, SqlColumnRef, SqlCase, SqlStar,
    SqlBinOp, SqlComparison, SqlLogic, SqlIsNull, SqlBetween, SqlInList,
    SqlAggCall,
    TokenType, Token, Lexer, Parser, KEYWORDS,
    SelectExpr, SelectStmt, TableRef, JoinClause,
)
from query_executor import Row, eval_expr


# =============================================================================
# Subquery AST Nodes
# =============================================================================

@dataclass
class SqlSubquery:
    """A subquery expression: (SELECT ...)"""
    query: SelectStmt


@dataclass
class SqlExists:
    """EXISTS (SELECT ...)"""
    query: SelectStmt
    negated: bool = False


@dataclass
class SqlInSubquery:
    """expr IN (SELECT ...) or expr NOT IN (SELECT ...)"""
    expr: Any
    query: SelectStmt
    negated: bool = False


@dataclass
class SqlQuantifiedComparison:
    """expr op ALL/ANY/SOME (SELECT ...)"""
    op: str  # '=', '!=', '<', '>', '<=', '>='
    quantifier: str  # 'all', 'any', 'some'
    expr: Any
    query: SelectStmt


@dataclass
class SubqueryTableRef:
    """FROM (SELECT ...) AS alias -- derived table"""
    query: SelectStmt
    alias: str


# =============================================================================
# Subquery Parser
# =============================================================================

class SubqueryParser(BuiltinParser):
    """Parser extended with subquery support."""

    def _parse_primary(self):
        """Override to handle subqueries and EXISTS in expressions."""
        tok = self.peek()

        # EXISTS (SELECT ...)
        if tok.type == TokenType.EXISTS:
            return self._parse_exists(negated=False)

        # Parenthesized expression or subquery
        if tok.type == TokenType.LPAREN:
            # Lookahead: is this (SELECT ...)?
            if self._is_subquery_ahead():
                self.advance()  # consume (
                subquery = self._parse_select()
                self.expect(TokenType.RPAREN)
                return SqlSubquery(query=subquery)

        return super()._parse_primary()

    def _is_subquery_ahead(self) -> bool:
        """Check if ( is followed by SELECT keyword."""
        if self.pos + 1 >= len(self.tokens):
            return False
        next_tok = self.tokens[self.pos + 1]
        return next_tok.type == TokenType.SELECT

    def _parse_exists(self, negated=False) -> SqlExists:
        """Parse EXISTS (SELECT ...)."""
        self.advance()  # consume EXISTS
        self.expect(TokenType.LPAREN)
        subquery = self._parse_select()
        self.expect(TokenType.RPAREN)
        return SqlExists(query=subquery, negated=negated)

    def _parse_not(self):
        """Override to handle NOT EXISTS."""
        if self.peek().type == TokenType.NOT:
            # Lookahead for NOT EXISTS
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.EXISTS:
                self.advance()  # consume NOT
                return self._parse_exists(negated=True)
        return super()._parse_not()

    def _parse_comparison(self):
        """Override to handle IN (subquery), NOT IN (subquery), and ALL/ANY/SOME."""
        left = self._parse_addition()
        tok = self.peek()

        # IS [NOT] NULL -- must check before NOT IN
        if tok.type == TokenType.IS:
            self.advance()
            negated = bool(self.match(TokenType.NOT))
            self.expect(TokenType.NULL)
            return SqlIsNull(expr=left, negated=negated)

        # NOT IN (subquery) / NOT IN (list) / NOT LIKE / NOT BETWEEN
        if tok.type == TokenType.NOT:
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.IN:
                self.advance()  # consume NOT
                self.advance()  # consume IN
                self.expect(TokenType.LPAREN)
                if self.peek().type == TokenType.SELECT:
                    subquery = self._parse_select()
                    self.expect(TokenType.RPAREN)
                    return SqlInSubquery(expr=left, query=subquery, negated=True)
                else:
                    vals = [self._parse_expr()]
                    while self.match(TokenType.COMMA):
                        vals.append(self._parse_expr())
                    self.expect(TokenType.RPAREN)
                    return SqlLogic(op='not', operands=[SqlInList(expr=left, values=vals)])
            elif self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.LIKE:
                self.advance()  # NOT
                self.advance()  # LIKE
                pattern = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlComparison(op='like', left=left, right=pattern)])
            elif self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.BETWEEN:
                self.advance()  # NOT
                self.advance()  # BETWEEN
                low = self._parse_addition()
                self.expect(TokenType.AND)
                high = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlBetween(expr=left, low=low, high=high)])

        # IN (subquery) / IN (list)
        if tok.type == TokenType.IN:
            self.advance()  # consume IN
            self.expect(TokenType.LPAREN)
            if self.peek().type == TokenType.SELECT:
                subquery = self._parse_select()
                self.expect(TokenType.RPAREN)
                return SqlInSubquery(expr=left, query=subquery)
            else:
                vals = [self._parse_expr()]
                while self.match(TokenType.COMMA):
                    vals.append(self._parse_expr())
                self.expect(TokenType.RPAREN)
                return SqlInList(expr=left, values=vals)

        # BETWEEN
        if tok.type == TokenType.BETWEEN:
            self.advance()
            low = self._parse_addition()
            self.expect(TokenType.AND)
            high = self._parse_addition()
            return SqlBetween(expr=left, low=low, high=high)

        # LIKE
        if tok.type == TokenType.LIKE:
            self.advance()
            pattern = self._parse_addition()
            return SqlComparison(op='like', left=left, right=pattern)

        # Comparison operators: check for ALL/ANY/SOME quantifier
        op_map = {
            TokenType.EQ: '=', TokenType.NE: '!=',
            TokenType.LT: '<', TokenType.LE: '<=',
            TokenType.GT: '>', TokenType.GE: '>=',
        }
        if tok.type in op_map:
            # Lookahead for ALL/ANY/SOME
            if self.pos + 1 < len(self.tokens):
                next_tok = self.tokens[self.pos + 1]
                if (next_tok.type == TokenType.ALL or
                        (next_tok.type == TokenType.IDENT and
                         next_tok.value.upper() in ('ANY', 'SOME'))):
                    op = op_map[tok.type]
                    self.advance()  # consume op
                    quantifier = self.advance().value.upper()  # consume ALL/ANY/SOME
                    self.expect(TokenType.LPAREN)
                    subquery = self._parse_select()
                    self.expect(TokenType.RPAREN)
                    return SqlQuantifiedComparison(
                        op=op, quantifier=quantifier, expr=left, query=subquery
                    )
            # Regular comparison
            op = op_map[tok.type]
            self.advance()
            right = self._parse_addition()
            return SqlComparison(op=op, left=left, right=right)

        return left

    def _parse_table_ref(self) -> TableRef:
        """Override to handle subqueries in FROM clause."""
        tok = self.peek()
        if tok.type == TokenType.LPAREN:
            # Derived table: (SELECT ...) AS alias
            self.advance()  # consume (
            subquery = self._parse_select()
            self.expect(TokenType.RPAREN)
            # Alias is required for derived tables
            alias = None
            if self.match(TokenType.AS):
                alias = self.advance().value
            elif self.peek().type == TokenType.IDENT:
                alias = self.advance().value
            if alias is None:
                raise ParseError("Derived table subquery requires an alias")
            return SubqueryTableRef(query=subquery, alias=alias)
        return super()._parse_table_ref()


# =============================================================================
# Parse functions
# =============================================================================

def parse_subquery_sql(sql: str):
    """Parse a single SQL statement with subquery support."""
    lexer = Lexer(sql)
    parser = SubqueryParser(lexer.tokens)
    return parser.parse()


def parse_subquery_sql_multi(sql: str):
    """Parse multiple SQL statements with subquery support."""
    lexer = Lexer(sql)
    parser = SubqueryParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# Subquery evaluator
# =============================================================================

def subquery_eval_expr(expr, row: Row, db: 'SubqueryDB') -> Any:
    """Extended eval_expr that handles subquery nodes.

    Passes db reference to enable subquery execution.
    """
    if isinstance(expr, SqlSubquery):
        # Scalar subquery -- execute and return single value
        result = db._exec_subquery(expr.query, row)
        if not result.rows:
            return None
        if len(result.rows) > 1:
            raise CompileError("Scalar subquery returned more than one row")
        return result.rows[0][0]

    if isinstance(expr, SqlExists):
        result = db._exec_subquery(expr.query, row)
        exists = len(result.rows) > 0
        return not exists if expr.negated else exists

    if isinstance(expr, SqlInSubquery):
        val = subquery_eval_expr(expr.expr, row, db)
        result = db._exec_subquery(expr.query, row)
        values = {r[0] for r in result.rows}
        found = val in values
        return not found if expr.negated else found

    if isinstance(expr, SqlQuantifiedComparison):
        val = subquery_eval_expr(expr.expr, row, db)
        result = db._exec_subquery(expr.query, row)
        subvals = [r[0] for r in result.rows]

        if not subvals:
            # ALL with empty set is True, ANY with empty set is False
            return expr.quantifier == 'ALL'

        op = expr.op
        cmp_fn = _get_cmp_fn(op)

        if expr.quantifier == 'ALL':
            return all(cmp_fn(val, sv) for sv in subvals)
        else:  # ANY or SOME
            return any(cmp_fn(val, sv) for sv in subvals)

    # For SqlCast
    if isinstance(expr, SqlCast):
        val = subquery_eval_expr(expr.expr, row, db)
        return _do_cast(val, expr.type_name)

    # For SqlFuncCall with builtins
    if isinstance(expr, SqlFuncCall):
        args = [subquery_eval_expr(a, row, db) for a in expr.args]
        result = _builtin_apply(expr.func_name, args)
        if result is not _UNKNOWN_FUNC:
            return result

    # Delegate non-subquery expressions to builtin_eval_expr
    # But first check for nested subqueries in compound expressions
    if isinstance(expr, SqlBinOp):
        left = subquery_eval_expr(expr.left, row, db)
        right = subquery_eval_expr(expr.right, row, db)
        if left is None or right is None:
            return None
        op = expr.op
        if op == '+':
            return left + right
        if op == '-':
            return left - right
        if op == '*':
            return left * right
        if op == '/':
            return left / right if right != 0 else None
        if op == '%':
            return left % right if right != 0 else None
        if op == '||':
            return str(left) + str(right)
        return None

    if isinstance(expr, SqlComparison):
        left = subquery_eval_expr(expr.left, row, db)
        right = subquery_eval_expr(expr.right, row, db) if expr.right is not None else None
        op = expr.op
        if op == '=':
            return left == right
        if op in ('!=', '<>'):
            return left != right
        if op == '<':
            return left < right if left is not None and right is not None else False
        if op == '>':
            return left > right if left is not None and right is not None else False
        if op == '<=':
            return left <= right if left is not None and right is not None else False
        if op == '>=':
            return left >= right if left is not None and right is not None else False
        if op == 'like':
            if left is None or right is None:
                return False
            import re
            pattern = str(right).replace('%', '.*').replace('_', '.')
            return bool(re.fullmatch(pattern, str(left), re.IGNORECASE))
        return False

    if isinstance(expr, SqlLogic):
        op = expr.op.lower()
        if op == 'and':
            return all(subquery_eval_expr(o, row, db) for o in expr.operands)
        if op == 'or':
            return any(subquery_eval_expr(o, row, db) for o in expr.operands)
        if op == 'not':
            return not subquery_eval_expr(expr.operands[0], row, db)
        return False

    if isinstance(expr, SqlIsNull):
        val = subquery_eval_expr(expr.expr, row, db)
        result = val is None
        return not result if expr.negated else result

    if isinstance(expr, SqlBetween):
        val = subquery_eval_expr(expr.expr, row, db)
        low = subquery_eval_expr(expr.low, row, db)
        high = subquery_eval_expr(expr.high, row, db)
        if val is None or low is None or high is None:
            return False
        return low <= val <= high

    if isinstance(expr, SqlInList):
        val = subquery_eval_expr(expr.expr, row, db)
        vals = [subquery_eval_expr(v, row, db) for v in expr.values]
        return val in vals

    if isinstance(expr, SqlCase):
        for cond, val in expr.whens:
            if subquery_eval_expr(cond, row, db):
                return subquery_eval_expr(val, row, db)
        if expr.else_result is not None:
            return subquery_eval_expr(expr.else_result, row, db)
        return None

    # Fall through to builtin_eval_expr for simple nodes
    return builtin_eval_expr(expr, row)


def _get_cmp_fn(op):
    """Return a comparison function for the given operator."""
    if op == '=':
        return lambda a, b: a == b
    if op in ('!=', '<>'):
        return lambda a, b: a != b
    if op == '<':
        return lambda a, b: a is not None and b is not None and a < b
    if op == '>':
        return lambda a, b: a is not None and b is not None and a > b
    if op == '<=':
        return lambda a, b: a is not None and b is not None and a <= b
    if op == '>=':
        return lambda a, b: a is not None and b is not None and a >= b
    return lambda a, b: False


# =============================================================================
# SubqueryDB -- Database with subquery support
# =============================================================================

class SubqueryDB(BuiltinDB):
    """BuiltinDB extended with subquery support."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with subquery support."""
        stmt = parse_subquery_sql(sql)
        return self._execute_subquery_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_subquery_sql_multi(sql)
        return [self._execute_subquery_stmt(s) for s in stmts]

    def _execute_subquery_stmt(self, stmt) -> ResultSet:
        """Execute a statement with subquery support."""
        if isinstance(stmt, SelectStmt):
            if self._has_subquery(stmt):
                return self._exec_select_with_subqueries(stmt)
        # Fall through to builtin/window/base execution
        return self._execute_builtin_stmt(stmt)

    def _has_subquery(self, stmt: SelectStmt) -> bool:
        """Check if the statement has any subquery nodes."""
        for col in stmt.columns:
            if self._expr_has_subquery(col.expr):
                return True
        if stmt.where and self._expr_has_subquery(stmt.where):
            return True
        if stmt.having and self._expr_has_subquery(stmt.having):
            return True
        if stmt.order_by:
            for expr, _ in stmt.order_by:
                if self._expr_has_subquery(expr):
                    return True
        if isinstance(stmt.from_table, SubqueryTableRef):
            return True
        for join in stmt.joins:
            if isinstance(join.table, SubqueryTableRef):
                return True
        return False

    def _expr_has_subquery(self, expr) -> bool:
        """Recursively check if an expression contains subquery nodes."""
        if isinstance(expr, (SqlSubquery, SqlExists, SqlInSubquery, SqlQuantifiedComparison)):
            return True
        if isinstance(expr, SqlBinOp):
            return self._expr_has_subquery(expr.left) or self._expr_has_subquery(expr.right)
        if isinstance(expr, SqlComparison):
            return self._expr_has_subquery(expr.left) or (
                expr.right is not None and self._expr_has_subquery(expr.right))
        if isinstance(expr, SqlLogic):
            return any(self._expr_has_subquery(o) for o in expr.operands)
        if isinstance(expr, SqlCase):
            for cond, val in expr.whens:
                if self._expr_has_subquery(cond) or self._expr_has_subquery(val):
                    return True
            if expr.else_result and self._expr_has_subquery(expr.else_result):
                return True
        if isinstance(expr, SqlFuncCall):
            return any(self._expr_has_subquery(a) for a in expr.args)
        if isinstance(expr, SqlIsNull):
            return self._expr_has_subquery(expr.expr)
        if isinstance(expr, SqlBetween):
            return (self._expr_has_subquery(expr.expr) or
                    self._expr_has_subquery(expr.low) or
                    self._expr_has_subquery(expr.high))
        if isinstance(expr, SqlInList):
            return (self._expr_has_subquery(expr.expr) or
                    any(self._expr_has_subquery(v) for v in expr.values))
        if isinstance(expr, SqlCast):
            return self._expr_has_subquery(expr.expr)
        return False

    def _exec_select_with_subqueries(self, stmt: SelectStmt) -> ResultSet:
        """Execute SELECT with subquery evaluation."""
        # Step 1: Resolve FROM clause (may be a derived table)
        if isinstance(stmt.from_table, SubqueryTableRef):
            return self._exec_derived_table_query(stmt)

        # Step 2: Handle JOINs with derived tables
        has_derived_join = any(isinstance(j.table, SubqueryTableRef) for j in stmt.joins)
        if has_derived_join:
            return self._exec_derived_join_query(stmt)

        # Step 3: Get base rows
        if stmt.from_table is None:
            # Expression-only query with subqueries (SELECT (SELECT ...))
            return self._exec_expr_with_subqueries(stmt)

        # Get all rows from the base table
        base_stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlStar())],
            from_table=stmt.from_table,
            joins=stmt.joins,
        )
        base_result = self._execute_builtin_stmt(base_stmt)

        rows = []
        for raw_row in base_result.rows:
            data = {}
            for i, col_name in enumerate(base_result.columns):
                data[col_name] = raw_row[i]
            rows.append(Row(data))

        # Step 4: Apply WHERE with subquery evaluation
        if stmt.where:
            rows = [r for r in rows if subquery_eval_expr(stmt.where, r, self)]

        # Step 5: Handle GROUP BY
        if stmt.group_by:
            return self._exec_grouped_with_subqueries(stmt, rows)

        # Step 6: Evaluate column expressions
        output_columns = []
        for i, col in enumerate(stmt.columns):
            if col.alias:
                output_columns.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                output_columns.append(col.expr.column)
            elif isinstance(col.expr, SqlStar):
                output_columns.append(f"col_{i}")
            else:
                output_columns.append(f"col_{i}")

        output_rows = []
        for row in rows:
            out_row = []
            for col in stmt.columns:
                if isinstance(col.expr, SqlStar):
                    for v in row.data.values():
                        out_row.append(v)
                else:
                    out_row.append(subquery_eval_expr(col.expr, row, self))
            output_rows.append(out_row)

        # Handle star column names
        if any(isinstance(c.expr, SqlStar) for c in stmt.columns):
            star_cols = list(rows[0].data.keys()) if rows else []
            output_columns = []
            for col in stmt.columns:
                if isinstance(col.expr, SqlStar):
                    output_columns.extend(star_cols)
                elif col.alias:
                    output_columns.append(col.alias)
                elif isinstance(col.expr, SqlColumnRef):
                    output_columns.append(col.expr.column)
                else:
                    output_columns.append(f"col_{len(output_columns)}")

        # Step 7: ORDER BY
        if stmt.order_by:
            output_rows = self._sort_with_subqueries(output_rows, output_columns, stmt.order_by, rows)

        # Step 8: DISTINCT
        if stmt.distinct:
            seen = set()
            unique = []
            for row in output_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            output_rows = unique

        # Step 9: LIMIT/OFFSET
        if stmt.offset:
            output_rows = output_rows[stmt.offset:]
        if stmt.limit is not None:
            output_rows = output_rows[:stmt.limit]

        return ResultSet(columns=output_columns, rows=output_rows)

    def _exec_subquery(self, query: SelectStmt, outer_row: Row = None) -> ResultSet:
        """Execute a subquery, optionally with outer row context for correlation."""
        # For correlated subqueries, we need to inject outer row values
        # into column references that don't match subquery tables
        if outer_row is not None and outer_row.data:
            return self._exec_correlated_subquery(query, outer_row)
        return self._execute_subquery_stmt(query)

    def _exec_correlated_subquery(self, query: SelectStmt, outer_row: Row) -> ResultSet:
        """Execute a correlated subquery with outer row context.

        Strategy: rewrite column refs that match outer row columns to literals.
        """
        # Get inner table columns
        inner_cols = set()
        if query.from_table and not isinstance(query.from_table, SubqueryTableRef):
            table_name = query.from_table.table_name
            if table_name in self.tables:
                inner_cols = set(self.tables[table_name]['columns'])
            # Also add aliased table refs
            if query.from_table.alias:
                inner_cols.update(f"{query.from_table.alias}.{c}" for c in self.tables.get(table_name, {}).get('columns', []))

        # Rewrite WHERE clause to substitute outer references
        rewritten_where = self._rewrite_correlated_expr(query.where, outer_row, inner_cols, query) if query.where else None

        # Rewrite columns
        rewritten_cols = []
        for col in query.columns:
            new_expr = self._rewrite_correlated_expr(col.expr, outer_row, inner_cols, query)
            rewritten_cols.append(SelectExpr(expr=new_expr, alias=col.alias))

        # Rewrite HAVING
        rewritten_having = self._rewrite_correlated_expr(query.having, outer_row, inner_cols, query) if query.having else None

        new_query = SelectStmt(
            columns=rewritten_cols,
            from_table=query.from_table,
            joins=query.joins,
            where=rewritten_where,
            group_by=query.group_by,
            having=rewritten_having,
            order_by=query.order_by,
            limit=query.limit,
            offset=query.offset,
            distinct=query.distinct,
        )
        return self._execute_subquery_stmt(new_query)

    def _rewrite_correlated_expr(self, expr, outer_row: Row, inner_cols: set, query: SelectStmt):
        """Rewrite expression to substitute outer row references with literals."""
        if expr is None:
            return None

        if isinstance(expr, SqlColumnRef):
            # If column has a table qualifier that matches an outer table
            if expr.table:
                qualified = f"{expr.table}.{expr.column}"
                # Check if this references the inner table
                if query.from_table and not isinstance(query.from_table, SubqueryTableRef):
                    inner_table = query.from_table.table_name
                    inner_alias = query.from_table.alias
                    if expr.table == inner_table or expr.table == inner_alias:
                        return expr  # Inner table reference, keep as-is

                # Check joins
                for join in query.joins:
                    if not isinstance(join.table, SubqueryTableRef):
                        if expr.table == join.table.table_name or expr.table == join.table.alias:
                            return expr  # Join table reference

                # Must be an outer reference
                if qualified in outer_row.data:
                    return SqlLiteral(outer_row.data[qualified])
                if expr.column in outer_row.data:
                    return SqlLiteral(outer_row.data[expr.column])
            else:
                # Unqualified column -- outer if not in inner table
                if expr.column not in inner_cols:
                    if expr.column in outer_row.data:
                        return SqlLiteral(outer_row.data[expr.column])
            return expr

        if isinstance(expr, SqlBinOp):
            return SqlBinOp(
                op=expr.op,
                left=self._rewrite_correlated_expr(expr.left, outer_row, inner_cols, query),
                right=self._rewrite_correlated_expr(expr.right, outer_row, inner_cols, query),
            )

        if isinstance(expr, SqlComparison):
            return SqlComparison(
                op=expr.op,
                left=self._rewrite_correlated_expr(expr.left, outer_row, inner_cols, query),
                right=self._rewrite_correlated_expr(expr.right, outer_row, inner_cols, query) if expr.right else None,
            )

        if isinstance(expr, SqlLogic):
            return SqlLogic(
                op=expr.op,
                operands=[self._rewrite_correlated_expr(o, outer_row, inner_cols, query) for o in expr.operands],
            )

        if isinstance(expr, SqlIsNull):
            return SqlIsNull(
                expr=self._rewrite_correlated_expr(expr.expr, outer_row, inner_cols, query),
                negated=expr.negated,
            )

        if isinstance(expr, SqlBetween):
            return SqlBetween(
                expr=self._rewrite_correlated_expr(expr.expr, outer_row, inner_cols, query),
                low=self._rewrite_correlated_expr(expr.low, outer_row, inner_cols, query),
                high=self._rewrite_correlated_expr(expr.high, outer_row, inner_cols, query),
            )

        if isinstance(expr, SqlInList):
            return SqlInList(
                expr=self._rewrite_correlated_expr(expr.expr, outer_row, inner_cols, query),
                values=[self._rewrite_correlated_expr(v, outer_row, inner_cols, query) for v in expr.values],
            )

        if isinstance(expr, SqlCase):
            return SqlCase(
                whens=[(self._rewrite_correlated_expr(c, outer_row, inner_cols, query),
                         self._rewrite_correlated_expr(v, outer_row, inner_cols, query))
                        for c, v in expr.whens],
                else_result=self._rewrite_correlated_expr(expr.else_result, outer_row, inner_cols, query),
            )

        if isinstance(expr, SqlFuncCall):
            return SqlFuncCall(
                func_name=expr.func_name,
                args=[self._rewrite_correlated_expr(a, outer_row, inner_cols, query) for a in expr.args],
                distinct=expr.distinct,
            )

        if isinstance(expr, SqlAggCall):
            return SqlAggCall(
                func=expr.func,
                arg=self._rewrite_correlated_expr(expr.arg, outer_row, inner_cols, query) if expr.arg else None,
                distinct=expr.distinct,
                alias=expr.alias,
            )

        if isinstance(expr, SqlSubquery):
            # Nested subquery -- pass through (it will get its own correlation context)
            return expr

        if isinstance(expr, (SqlExists, SqlInSubquery, SqlQuantifiedComparison)):
            return expr

        if isinstance(expr, SqlCast):
            return SqlCast(
                expr=self._rewrite_correlated_expr(expr.expr, outer_row, inner_cols, query),
                type_name=expr.type_name,
            )

        return expr

    def _exec_derived_table_query(self, stmt: SelectStmt) -> ResultSet:
        """Execute a query with a derived table (subquery in FROM)."""
        derived = stmt.from_table
        # Execute the subquery to get the derived table
        sub_result = self._execute_subquery_stmt(derived.query)

        # Create virtual rows with alias-qualified column names
        alias = derived.alias
        rows = []
        for raw_row in sub_result.rows:
            data = {}
            for i, col_name in enumerate(sub_result.columns):
                data[col_name] = raw_row[i]
                data[f"{alias}.{col_name}"] = raw_row[i]
            rows.append(Row(data))

        # Handle JOINs with the derived table
        if stmt.joins:
            rows = self._apply_joins_on_rows(rows, stmt.joins)

        # Apply WHERE
        if stmt.where:
            rows = [r for r in rows if subquery_eval_expr(stmt.where, r, self)]

        # GROUP BY
        if stmt.group_by:
            return self._exec_grouped_on_rows(stmt, rows)

        # Evaluate columns
        output_columns, output_rows = self._eval_columns_on_rows(stmt.columns, rows)

        # ORDER BY
        if stmt.order_by:
            output_rows = self._sort_with_subqueries(output_rows, output_columns, stmt.order_by, rows)

        # DISTINCT
        if stmt.distinct:
            seen = set()
            unique = []
            for row in output_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            output_rows = unique

        # LIMIT/OFFSET
        if stmt.offset:
            output_rows = output_rows[stmt.offset:]
        if stmt.limit is not None:
            output_rows = output_rows[:stmt.limit]

        return ResultSet(columns=output_columns, rows=output_rows)

    def _exec_derived_join_query(self, stmt: SelectStmt) -> ResultSet:
        """Execute a query with derived tables in JOINs."""
        # Get base table rows
        if isinstance(stmt.from_table, SubqueryTableRef):
            return self._exec_derived_table_query(stmt)

        base_stmt = SelectStmt(
            columns=[SelectExpr(expr=SqlStar())],
            from_table=stmt.from_table,
        )
        base_result = self._execute_builtin_stmt(base_stmt)
        rows = []
        for raw_row in base_result.rows:
            data = {}
            for i, col_name in enumerate(base_result.columns):
                data[col_name] = raw_row[i]
                if stmt.from_table.alias:
                    data[f"{stmt.from_table.alias}.{col_name}"] = raw_row[i]
                data[f"{stmt.from_table.table_name}.{col_name}"] = raw_row[i]
            rows.append(Row(data))

        # Apply joins
        rows = self._apply_joins_on_rows(rows, stmt.joins)

        # WHERE
        if stmt.where:
            rows = [r for r in rows if subquery_eval_expr(stmt.where, r, self)]

        # GROUP BY
        if stmt.group_by:
            return self._exec_grouped_on_rows(stmt, rows)

        # Evaluate columns
        output_columns, output_rows = self._eval_columns_on_rows(stmt.columns, rows)

        # ORDER BY
        if stmt.order_by:
            output_rows = self._sort_with_subqueries(output_rows, output_columns, stmt.order_by, rows)

        # DISTINCT, LIMIT, OFFSET
        if stmt.distinct:
            seen = set()
            unique = []
            for row in output_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            output_rows = unique
        if stmt.offset:
            output_rows = output_rows[stmt.offset:]
        if stmt.limit is not None:
            output_rows = output_rows[:stmt.limit]

        return ResultSet(columns=output_columns, rows=output_rows)

    def _apply_joins_on_rows(self, left_rows: List[Row], joins: List[JoinClause]) -> List[Row]:
        """Apply JOIN operations on Row objects."""
        result = left_rows
        for join in joins:
            if isinstance(join.table, SubqueryTableRef):
                # Derived table join
                sub_result = self._execute_subquery_stmt(join.table.query)
                alias = join.table.alias
                right_rows = []
                for raw_row in sub_result.rows:
                    data = {}
                    for i, col_name in enumerate(sub_result.columns):
                        data[col_name] = raw_row[i]
                        data[f"{alias}.{col_name}"] = raw_row[i]
                    right_rows.append(Row(data))
            else:
                # Regular table join
                table_name = join.table.table_name
                table_alias = join.table.alias or table_name
                if table_name not in self.tables:
                    raise CompileError(f"Table '{table_name}' does not exist")
                table_data = self.tables[table_name]
                right_rows = []
                for raw_row in table_data['rows']:
                    data = {}
                    for i, col_name in enumerate(table_data['columns']):
                        data[col_name] = raw_row[i]
                        data[f"{table_alias}.{col_name}"] = raw_row[i]
                        data[f"{table_name}.{col_name}"] = raw_row[i]
                    right_rows.append(Row(data))

            join_type = join.join_type.lower()
            new_result = []

            if join_type == 'cross':
                for lr in result:
                    for rr in right_rows:
                        merged = Row({**lr.data, **rr.data})
                        new_result.append(merged)
            elif join_type in ('inner', 'join'):
                for lr in result:
                    for rr in right_rows:
                        merged = Row({**lr.data, **rr.data})
                        if join.condition is None or subquery_eval_expr(join.condition, merged, self):
                            new_result.append(merged)
            elif join_type == 'left':
                for lr in result:
                    matched = False
                    for rr in right_rows:
                        merged = Row({**lr.data, **rr.data})
                        if join.condition is None or subquery_eval_expr(join.condition, merged, self):
                            new_result.append(merged)
                            matched = True
                    if not matched:
                        # Add null columns from right side
                        null_data = {k: None for k in (right_rows[0].data if right_rows else {})}
                        merged = Row({**lr.data, **null_data})
                        new_result.append(merged)
            else:
                # Default to inner join
                for lr in result:
                    for rr in right_rows:
                        merged = Row({**lr.data, **rr.data})
                        if join.condition is None or subquery_eval_expr(join.condition, merged, self):
                            new_result.append(merged)

            result = new_result

        return result

    def _eval_columns_on_rows(self, columns, rows: List[Row]) -> Tuple[List[str], List[list]]:
        """Evaluate column expressions against Row objects."""
        output_columns = []
        has_star = False
        for i, col in enumerate(columns):
            if isinstance(col.expr, SqlStar):
                has_star = True
                output_columns.append(f"col_{i}")
            elif col.alias:
                output_columns.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                output_columns.append(col.expr.column)
            else:
                output_columns.append(f"col_{i}")

        output_rows = []
        for row in rows:
            out_row = []
            for col in columns:
                if isinstance(col.expr, SqlStar):
                    for v in row.data.values():
                        # Skip alias-qualified duplicates
                        out_row.append(v)
                else:
                    out_row.append(subquery_eval_expr(col.expr, row, self))
            output_rows.append(out_row)

        # Fix star column names
        if has_star and rows:
            # Get non-duplicate columns (skip alias.col entries)
            star_cols = []
            for k in rows[0].data.keys():
                if '.' not in k:
                    star_cols.append(k)

            output_columns = []
            for col in columns:
                if isinstance(col.expr, SqlStar):
                    output_columns.extend(star_cols)
                elif col.alias:
                    output_columns.append(col.alias)
                elif isinstance(col.expr, SqlColumnRef):
                    output_columns.append(col.expr.column)
                else:
                    output_columns.append(f"col_{len(output_columns)}")

            # Rebuild rows to only include non-duplicate values
            output_rows = []
            for row in rows:
                out_row = []
                for col in columns:
                    if isinstance(col.expr, SqlStar):
                        for k in star_cols:
                            out_row.append(row.data.get(k))
                    else:
                        out_row.append(subquery_eval_expr(col.expr, row, self))
                output_rows.append(out_row)

        return output_columns, output_rows

    def _exec_grouped_on_rows(self, stmt: SelectStmt, rows: List[Row]) -> ResultSet:
        """Execute GROUP BY on pre-fetched rows with subquery support."""
        # Group rows
        groups = {}
        for row in rows:
            key_parts = []
            for gb_expr in stmt.group_by:
                key_parts.append(subquery_eval_expr(gb_expr, row, self))
            key = tuple(key_parts)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        # Evaluate each group
        output_columns = []
        for i, col in enumerate(stmt.columns):
            if col.alias:
                output_columns.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                output_columns.append(col.expr.column)
            elif isinstance(col.expr, SqlAggCall):
                output_columns.append(f"{col.expr.func}_{i}")
            else:
                output_columns.append(f"col_{i}")

        output_rows = []
        for key, group_rows in groups.items():
            representative = group_rows[0]
            out_row = []
            for col in stmt.columns:
                if isinstance(col.expr, SqlAggCall):
                    out_row.append(self._eval_agg_on_rows(col.expr, group_rows))
                else:
                    out_row.append(subquery_eval_expr(col.expr, representative, self))
            output_rows.append(out_row)

        # Apply HAVING
        if stmt.having:
            filtered = []
            for i, (key, group_rows) in enumerate(groups.items()):
                # Build a row with aggregated + group values for HAVING eval
                having_row = Row({})
                for j, col in enumerate(stmt.columns):
                    col_name = output_columns[j]
                    having_row.data[col_name] = output_rows[i][j]
                # Also add representative data
                having_row.data.update(group_rows[0].data)
                if subquery_eval_expr(stmt.having, having_row, self):
                    filtered.append(output_rows[i])
            output_rows = filtered

        # ORDER BY
        if stmt.order_by:
            output_rows = self._sort_by_columns(output_rows, output_columns, stmt.order_by)

        # LIMIT/OFFSET
        if stmt.offset:
            output_rows = output_rows[stmt.offset:]
        if stmt.limit is not None:
            output_rows = output_rows[:stmt.limit]

        return ResultSet(columns=output_columns, rows=output_rows)

    def _eval_agg_on_rows(self, agg: SqlAggCall, rows: List[Row]) -> Any:
        """Evaluate an aggregate function on a group of rows."""
        func = agg.func.lower()

        if func == 'count':
            if agg.arg is None:
                return len(rows)
            vals = [subquery_eval_expr(agg.arg, r, self) for r in rows]
            vals = [v for v in vals if v is not None]
            if agg.distinct:
                vals = list(set(vals))
            return len(vals)

        vals = [subquery_eval_expr(agg.arg, r, self) for r in rows]
        vals = [v for v in vals if v is not None]
        if agg.distinct:
            vals = list(set(vals))

        if func == 'sum':
            return sum(vals) if vals else 0
        if func == 'avg':
            return sum(vals) / len(vals) if vals else None
        if func == 'min':
            return min(vals) if vals else None
        if func == 'max':
            return max(vals) if vals else None
        return None

    def _exec_expr_with_subqueries(self, stmt: SelectStmt) -> ResultSet:
        """Execute expression-only SELECT with subqueries (no FROM)."""
        output_columns = []
        output_row = []
        dummy_row = Row({})

        for i, col in enumerate(stmt.columns):
            val = subquery_eval_expr(col.expr, dummy_row, self)
            output_row.append(val)
            if col.alias:
                output_columns.append(col.alias)
            else:
                output_columns.append(f"col_{i}")

        return ResultSet(columns=output_columns, rows=[output_row])

    def _sort_with_subqueries(self, output_rows, columns, order_by, base_rows):
        """Sort output rows by ORDER BY with subquery eval support."""
        from functools import cmp_to_key

        pairs = list(enumerate(output_rows))

        def compare(a, b):
            idx_a, row_a = a
            idx_b, row_b = b
            for expr, asc in order_by:
                if isinstance(expr, SqlColumnRef):
                    col_name = expr.column
                    try:
                        ci = columns.index(col_name)
                        va, vb = row_a[ci], row_b[ci]
                    except ValueError:
                        va = subquery_eval_expr(expr, base_rows[idx_a], self) if idx_a < len(base_rows) else None
                        vb = subquery_eval_expr(expr, base_rows[idx_b], self) if idx_b < len(base_rows) else None
                else:
                    va = subquery_eval_expr(expr, base_rows[idx_a], self) if idx_a < len(base_rows) else None
                    vb = subquery_eval_expr(expr, base_rows[idx_b], self) if idx_b < len(base_rows) else None

                if va is None and vb is None:
                    continue
                if va is None:
                    return 1 if asc else -1
                if vb is None:
                    return -1 if asc else 1
                if va < vb:
                    return -1 if asc else 1
                if va > vb:
                    return 1 if asc else -1
            return 0

        pairs.sort(key=cmp_to_key(compare))
        return [row for _, row in pairs]

    def _sort_by_columns(self, output_rows, columns, order_by):
        """Sort output rows using column name lookup."""
        from functools import cmp_to_key

        pairs = list(enumerate(output_rows))

        def compare(a, b):
            idx_a, row_a = a
            idx_b, row_b = b
            for expr, asc in order_by:
                if isinstance(expr, SqlColumnRef):
                    col_name = expr.column
                    try:
                        ci = columns.index(col_name)
                        va, vb = row_a[ci], row_b[ci]
                    except ValueError:
                        va = vb = None
                else:
                    va = vb = None

                if va is None and vb is None:
                    continue
                if va is None:
                    return 1 if asc else -1
                if vb is None:
                    return -1 if asc else 1
                if va < vb:
                    return -1 if asc else 1
                if va > vb:
                    return 1 if asc else -1
            return 0

        pairs.sort(key=cmp_to_key(compare))
        return [row for _, row in pairs]
