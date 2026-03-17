"""
C248: SQL Subqueries
Extends C247 (Mini Database Engine)

Adds full subquery support to the SQL database engine:
- Scalar subqueries: SELECT (SELECT MAX(x) FROM t)
- IN subqueries: WHERE x IN (SELECT y FROM t2)
- NOT IN subqueries: WHERE x NOT IN (SELECT y FROM t2)
- EXISTS / NOT EXISTS: WHERE EXISTS (SELECT 1 FROM t2 WHERE ...)
- Correlated subqueries: WHERE x > (SELECT AVG(y) FROM t2 WHERE t2.id = t1.id)
- Derived tables: FROM (SELECT ...) AS alias
- Subqueries in SELECT list
- Subqueries with ANY/ALL: WHERE x > ANY (SELECT y FROM t2)
- Nested subqueries (subquery inside subquery)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import C247
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from mini_database import (
    MiniDB, ResultSet, DatabaseError,
    Lexer, Parser, Token, TokenType,
    parse_sql, parse_sql_multi,
    # AST nodes
    SelectStmt, SelectExpr, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    BeginStmt, CommitStmt, RollbackStmt, SavepointStmt,
    ShowTablesStmt, DescribeStmt, ExplainStmt,
    ColumnDef,
    # SQL expression AST
    SqlColumnRef, SqlLiteral, SqlBinOp, SqlComparison, SqlLogic,
    SqlIsNull, SqlFuncCall, SqlAggCall, SqlBetween, SqlInList,
    SqlCase, SqlStar,
    # Compiler / storage
    QueryCompiler, StorageEngine, CompileError, CatalogError,
    ParseError, KEYWORDS,
)

from query_executor import (
    Database as QEDatabase, Table as QETable, Row, ExecutionEngine,
    Operator, SeqScanOp, FilterOp, ProjectOp, SortOp,
    HashAggregateOp, HavingOp, LimitOp, DistinctOp,
    HashJoinOp, NestedLoopJoinOp,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr,
    CompOp, LogicOp, AggFunc, AggCall,
    eval_expr,
)

from transaction_manager import IsolationLevel

_original_eval_expr = eval_expr


# =============================================================================
# New AST Nodes for Subqueries
# =============================================================================

@dataclass
class SqlSubquery:
    """A subquery: (SELECT ...)"""
    select: SelectStmt


@dataclass
class SqlInSubquery:
    """expr IN (SELECT ...) or expr NOT IN (SELECT ...)"""
    expr: Any
    subquery: SqlSubquery
    negated: bool = False


@dataclass
class SqlExistsExpr:
    """EXISTS (SELECT ...) or NOT EXISTS (SELECT ...)"""
    subquery: SqlSubquery
    negated: bool = False


@dataclass
class SqlAnyAll:
    """expr op ANY/ALL (SELECT ...)"""
    expr: Any
    op: str          # '=', '!=', '<', '<=', '>', '>='
    quantifier: str  # 'any' or 'all'
    subquery: SqlSubquery


@dataclass
class DerivedTable:
    """FROM (SELECT ...) AS alias"""
    subquery: SqlSubquery
    alias: str


# =============================================================================
# Extended Parser with Subquery Support
# =============================================================================

class SubqueryParser(Parser):
    """Extends C247 Parser to handle subqueries."""

    def _is_any_all(self) -> bool:
        """Check if current token is ANY or ALL (handles ALL being a keyword)."""
        t = self.peek()
        if t.type == TokenType.ALL:
            return True
        if t.type == TokenType.IDENT and t.value.lower() == 'any':
            return True
        return False

    def _consume_any_all(self) -> str:
        """Consume and return 'any' or 'all'."""
        t = self.peek()
        if t.type == TokenType.ALL:
            self.advance()
            return 'all'
        if t.type == TokenType.IDENT and t.value.lower() == 'any':
            self.advance()
            return 'any'
        raise ParseError(f"Expected ANY or ALL, got {t.type.name}")

    def _parse_primary(self) -> Any:
        t = self.peek()

        # EXISTS (SELECT ...)
        if t.type == TokenType.EXISTS:
            self.advance()
            self.expect(TokenType.LPAREN)
            sub_select = self._parse_select()
            self.expect(TokenType.RPAREN)
            return SqlExistsExpr(subquery=SqlSubquery(select=sub_select), negated=False)

        # Parenthesized expression or subquery
        if t.type == TokenType.LPAREN:
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.SELECT:
                self.advance()  # consume '('
                sub_select = self._parse_select()
                self.expect(TokenType.RPAREN)
                return SqlSubquery(select=sub_select)
            self.advance()
            expr = self._parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        return super()._parse_primary()

    def _parse_not(self) -> Any:
        if self.match(TokenType.NOT):
            if self.peek().type == TokenType.EXISTS:
                self.advance()
                self.expect(TokenType.LPAREN)
                sub_select = self._parse_select()
                self.expect(TokenType.RPAREN)
                return SqlExistsExpr(subquery=SqlSubquery(select=sub_select), negated=True)
            expr = self._parse_not()
            return SqlLogic(op='not', operands=[expr])
        return self._parse_comparison()

    def _parse_comparison(self) -> Any:
        left = self._parse_addition()

        # IS [NOT] NULL
        if self.peek().type == TokenType.IS:
            self.advance()
            negated = bool(self.match(TokenType.NOT))
            self.expect(TokenType.NULL)
            return SqlIsNull(expr=left, negated=negated)

        # NOT IN / NOT LIKE / NOT BETWEEN
        if self.peek().type == TokenType.NOT:
            self.advance()
            if self.peek().type == TokenType.IN:
                self.advance()
                self.expect(TokenType.LPAREN)
                if self.peek().type == TokenType.SELECT:
                    sub_select = self._parse_select()
                    self.expect(TokenType.RPAREN)
                    return SqlInSubquery(expr=left,
                                        subquery=SqlSubquery(select=sub_select),
                                        negated=True)
                vals = [self._parse_expr()]
                while self.match(TokenType.COMMA):
                    vals.append(self._parse_expr())
                self.expect(TokenType.RPAREN)
                return SqlLogic(op='not', operands=[SqlInList(expr=left, values=vals)])
            elif self.peek().type == TokenType.LIKE:
                self.advance()
                pattern = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlComparison(op='like', left=left, right=pattern)])
            elif self.peek().type == TokenType.BETWEEN:
                self.advance()
                low = self._parse_addition()
                self.expect(TokenType.AND)
                high = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlBetween(expr=left, low=low, high=high)])
            else:
                self.pos -= 1

        # IN (subquery or value list)
        if self.match(TokenType.IN):
            self.expect(TokenType.LPAREN)
            if self.peek().type == TokenType.SELECT:
                sub_select = self._parse_select()
                self.expect(TokenType.RPAREN)
                return SqlInSubquery(expr=left,
                                    subquery=SqlSubquery(select=sub_select),
                                    negated=False)
            vals = [self._parse_expr()]
            while self.match(TokenType.COMMA):
                vals.append(self._parse_expr())
            self.expect(TokenType.RPAREN)
            return SqlInList(expr=left, values=vals)

        # BETWEEN
        if self.match(TokenType.BETWEEN):
            low = self._parse_addition()
            self.expect(TokenType.AND)
            high = self._parse_addition()
            return SqlBetween(expr=left, low=low, high=high)

        # LIKE
        if self.match(TokenType.LIKE):
            pattern = self._parse_addition()
            return SqlComparison(op='like', left=left, right=pattern)

        # Standard comparisons (with ANY/ALL support)
        op_map = {
            TokenType.EQ: '=', TokenType.NE: '!=', TokenType.LT: '<',
            TokenType.LE: '<=', TokenType.GT: '>', TokenType.GE: '>=',
        }
        for tt, op_str in op_map.items():
            if self.match(tt):
                if self._is_any_all():
                    quantifier = self._consume_any_all()
                    self.expect(TokenType.LPAREN)
                    sub_select = self._parse_select()
                    self.expect(TokenType.RPAREN)
                    return SqlAnyAll(expr=left, op=op_str, quantifier=quantifier,
                                    subquery=SqlSubquery(select=sub_select))
                right = self._parse_addition()
                return SqlComparison(op=op_str, left=left, right=right)

        return left

    def _parse_table_ref(self) -> Any:
        """Extended to handle derived tables: (SELECT ...) AS alias"""
        if self.peek().type == TokenType.LPAREN:
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.SELECT:
                self.advance()  # consume '('
                sub_select = self._parse_select()
                self.expect(TokenType.RPAREN)
                alias = None
                if self.match(TokenType.AS):
                    alias = self.expect(TokenType.IDENT).value
                elif self.peek().type == TokenType.IDENT and self.peek().value.lower() not in KEYWORDS:
                    alias = self.advance().value
                if alias is None:
                    raise ParseError("Derived table requires an alias")
                return DerivedTable(subquery=SqlSubquery(select=sub_select), alias=alias)
        return super()._parse_table_ref()


def parse_sql_subquery(sql: str) -> Any:
    lexer = Lexer(sql)
    parser = SubqueryParser(lexer.tokens)
    return parser.parse()


def parse_sql_subquery_multi(sql: str) -> List[Any]:
    lexer = Lexer(sql)
    parser = SubqueryParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# Subquery Expression Wrappers
# =============================================================================

class _SubqueryExprBase:
    """Base for all subquery expression wrappers."""
    pass


class _ScalarSubqueryExpr(_SubqueryExprBase):
    def __init__(self, subquery: SqlSubquery, db: 'SubqueryDB'):
        self.subquery = subquery
        self.db = db


class _InSubqueryExpr(_SubqueryExprBase):
    def __init__(self, qe_expr, subquery: SqlSubquery, negated: bool, db: 'SubqueryDB'):
        self.qe_expr = qe_expr
        self.subquery = subquery
        self.negated = negated
        self.db = db


class _ExistsSubqueryExpr(_SubqueryExprBase):
    def __init__(self, subquery: SqlSubquery, negated: bool, db: 'SubqueryDB'):
        self.subquery = subquery
        self.negated = negated
        self.db = db


class _AnyAllSubqueryExpr(_SubqueryExprBase):
    def __init__(self, qe_expr, op: str, quantifier: str,
                 subquery: SqlSubquery, db: 'SubqueryDB'):
        self.qe_expr = qe_expr
        self.op = op
        self.quantifier = quantifier
        self.subquery = subquery
        self.db = db


class _LogicSubqueryExpr(_SubqueryExprBase):
    """AND/OR/NOT combining subquery and non-subquery predicates."""
    def __init__(self, op: str, operands: list):
        self.op = op
        self.operands = operands


class _ComparisonWithSubquery(_SubqueryExprBase):
    """Comparison where one side is a subquery."""
    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


def _compare(left, op, right):
    if left is None or right is None:
        return False
    if op == '=': return left == right
    if op == '!=': return left != right
    if op == '<': return left < right
    if op == '<=': return left <= right
    if op == '>': return left > right
    if op == '>=': return left >= right
    return False


def _exec_subquery_correlated(db, subquery: SqlSubquery, outer_row: Row):
    """Execute a subquery with outer row values injected for correlated references."""
    return db._exec_select_correlated(subquery.select, outer_row)


def subquery_eval(expr, row):
    """Evaluate a subquery-aware expression against a row."""
    if isinstance(expr, _ScalarSubqueryExpr):
        result = _exec_subquery_correlated(expr.db, expr.subquery, row)
        if len(result.rows) == 0:
            return None
        if len(result.rows) > 1:
            raise DatabaseError("Scalar subquery returned more than one row")
        return result.rows[0][0]

    if isinstance(expr, _InSubqueryExpr):
        left_val = subquery_eval(expr.qe_expr, row)
        result = _exec_subquery_correlated(expr.db, expr.subquery, row)
        values = {r[0] for r in result.rows}
        found = left_val in values
        return not found if expr.negated else found

    if isinstance(expr, _ExistsSubqueryExpr):
        result = _exec_subquery_correlated(expr.db, expr.subquery, row)
        exists = len(result.rows) > 0
        return not exists if expr.negated else exists

    if isinstance(expr, _AnyAllSubqueryExpr):
        left_val = subquery_eval(expr.qe_expr, row)
        result = _exec_subquery_correlated(expr.db, expr.subquery, row)
        values = [r[0] for r in result.rows]
        if not values:
            return expr.quantifier == 'all'
        comps = [_compare(left_val, expr.op, v) for v in values]
        return any(comps) if expr.quantifier == 'any' else all(comps)

    if isinstance(expr, _LogicSubqueryExpr):
        if expr.op == 'not':
            return not subquery_eval(expr.operands[0], row)
        if expr.op == 'and':
            return all(subquery_eval(o, row) for o in expr.operands)
        if expr.op == 'or':
            return any(subquery_eval(o, row) for o in expr.operands)

    if isinstance(expr, _ComparisonWithSubquery):
        left_val = subquery_eval(expr.left, row)
        right_val = subquery_eval(expr.right, row)
        return _compare(left_val, expr.op, right_val)

    # Delegate to C245 eval_expr
    return _original_eval_expr(expr, row)


# =============================================================================
# Custom Operators
# =============================================================================

class SubqueryFilterOp(Operator):
    """Filter using subquery-aware evaluation."""

    def __init__(self, child: Operator, predicate, db: 'SubqueryDB'):
        super().__init__()
        self.child = child
        self.predicate = predicate
        self.db = db

    def open(self):
        super().open()
        self.child.open()

    def next(self) -> Optional[Row]:
        while True:
            row = self.child.next()
            if row is None:
                return None
            self.stats.rows_consumed += 1
            if subquery_eval(self.predicate, row):
                self.stats.rows_produced += 1
                return row

    def close(self):
        self.child.close()
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0) -> str:
        prefix = "  " * indent
        return f"{prefix}SubqueryFilter\n{self.child.explain(indent + 1)}"


class SubqueryProjectOp(Operator):
    """Project with subquery-aware evaluation in SELECT list."""

    def __init__(self, child: Operator, projections: List[Tuple[Any, str]], db: 'SubqueryDB'):
        super().__init__()
        self.child = child
        self.projections = projections
        self.db = db

    def open(self):
        super().open()
        self.child.open()

    def next(self) -> Optional[Row]:
        row = self.child.next()
        if row is None:
            return None
        self.stats.rows_consumed += 1
        result = {}
        schema = []
        for expr, alias in self.projections:
            result[alias] = subquery_eval(expr, row)
            schema.append(alias)
        self.stats.rows_produced += 1
        return Row(result, schema=schema)

    def close(self):
        self.child.close()
        self.stats.children = [self.child.stats]
        super().close()

    def explain(self, indent=0) -> str:
        prefix = "  " * indent
        cols = ", ".join(alias for _, alias in self.projections)
        return f"{prefix}SubqueryProject: [{cols}]\n{self.child.explain(indent + 1)}"


# =============================================================================
# Extended Query Compiler
# =============================================================================

class SubqueryCompiler(QueryCompiler):
    """Extends QueryCompiler to handle subquery expressions."""

    def __init__(self, storage: StorageEngine, db: 'SubqueryDB'):
        super().__init__(storage)
        self.db = db

    def _has_aggregates(self, columns) -> bool:
        """Override to recursively check for aggregates inside expressions."""
        for col in columns:
            if self._expr_has_aggregate(col.expr):
                return True
        return False

    def _expr_has_aggregate(self, node) -> bool:
        """Recursively check for aggregate calls in an expression."""
        if isinstance(node, SqlAggCall):
            return True
        if isinstance(node, SqlCase):
            for cond, result in node.whens:
                if self._expr_has_aggregate(cond) or self._expr_has_aggregate(result):
                    return True
            if node.else_result and self._expr_has_aggregate(node.else_result):
                return True
        if isinstance(node, SqlComparison):
            return self._expr_has_aggregate(node.left) or self._expr_has_aggregate(node.right)
        if isinstance(node, SqlBinOp):
            return self._expr_has_aggregate(node.left) or self._expr_has_aggregate(node.right)
        if isinstance(node, SqlLogic):
            return any(self._expr_has_aggregate(o) for o in node.operands)
        if isinstance(node, SqlFuncCall):
            return any(self._expr_has_aggregate(a) for a in node.args)
        return False

    def _extract_aggregates(self, columns):
        """Override to recursively extract aggregates from expressions."""
        aggs = []
        for col in columns:
            self._collect_aggregates(col.expr, aggs, col.alias)
        return aggs

    def _collect_aggregates(self, node, aggs, alias=None):
        """Recursively collect aggregates from an expression tree."""
        if isinstance(node, SqlAggCall):
            func_map = {
                'count': AggFunc.COUNT_STAR if node.arg is None else AggFunc.COUNT,
                'sum': AggFunc.SUM, 'avg': AggFunc.AVG,
                'min': AggFunc.MIN, 'max': AggFunc.MAX,
            }
            func = func_map.get(node.func, AggFunc.COUNT)
            arg_expr = None
            if node.arg is not None:
                arg_expr = super()._sql_to_qe_expr(node.arg)
            agg_alias = alias or self._agg_alias(node)
            aggs.append(AggCall(func=func, column=arg_expr,
                               distinct=node.distinct, alias=agg_alias))
            return
        if isinstance(node, SqlCase):
            for cond, result in node.whens:
                self._collect_aggregates(cond, aggs)
                self._collect_aggregates(result, aggs)
            if node.else_result:
                self._collect_aggregates(node.else_result, aggs)
        if isinstance(node, SqlComparison):
            self._collect_aggregates(node.left, aggs)
            self._collect_aggregates(node.right, aggs)
        if isinstance(node, SqlBinOp):
            self._collect_aggregates(node.left, aggs)
            self._collect_aggregates(node.right, aggs)
        if isinstance(node, SqlLogic):
            for o in node.operands:
                self._collect_aggregates(o, aggs)

    def _sql_to_qe_expr(self, node) -> Any:
        """Convert SQL AST, including subquery nodes, to evaluation expressions."""

        if isinstance(node, SqlSubquery):
            return _ScalarSubqueryExpr(node, self.db)

        if isinstance(node, SqlInSubquery):
            qe_expr = self._sql_to_qe_expr(node.expr)
            return _InSubqueryExpr(qe_expr, node.subquery, node.negated, self.db)

        if isinstance(node, SqlExistsExpr):
            return _ExistsSubqueryExpr(node.subquery, node.negated, self.db)

        if isinstance(node, SqlAnyAll):
            qe_expr = self._sql_to_qe_expr(node.expr)
            return _AnyAllSubqueryExpr(qe_expr, node.op, node.quantifier,
                                       node.subquery, self.db)

        # For Logic/Comparison that might contain subqueries, wrap them
        if isinstance(node, SqlLogic):
            operands = [self._sql_to_qe_expr(o) for o in node.operands]
            if any(isinstance(o, _SubqueryExprBase) for o in operands):
                return _LogicSubqueryExpr(node.op, operands)
            # No subqueries, use normal C245 path
            if node.op == 'not':
                return LogicExpr(LogicOp.NOT, operands)
            op_map = {'and': LogicOp.AND, 'or': LogicOp.OR}
            return LogicExpr(op_map[node.op], operands)

        if isinstance(node, SqlComparison):
            left = self._sql_to_qe_expr(node.left)
            right = self._sql_to_qe_expr(node.right)
            if isinstance(left, _SubqueryExprBase) or isinstance(right, _SubqueryExprBase):
                return _ComparisonWithSubquery(node.op, left, right)
            op_map = {
                '=': CompOp.EQ, '!=': CompOp.NE, '<': CompOp.LT,
                '<=': CompOp.LE, '>': CompOp.GT, '>=': CompOp.GE,
                'like': CompOp.LIKE,
            }
            return Comparison(op_map[node.op], left, right)

        # Everything else delegates to parent
        return super()._sql_to_qe_expr(node)

    def _build_plan(self, stmt: SelectStmt, qe_db: QEDatabase,
                    table_aliases: Dict[str, str]) -> Operator:
        """Build plan, using SubqueryFilterOp/SubqueryProjectOp when needed."""

        # Start with FROM clause
        if stmt.from_table is None:
            from query_executor import MaterializeOp
            dummy_table = qe_db.create_table('__dual', ['__dummy'])
            dummy_table.insert({'__dummy': 1})
            plan = SeqScanOp(dummy_table)
        else:
            tname = stmt.from_table.table_name
            plan = SeqScanOp(qe_db.get_table(tname))

        # JOINs (reuse parent logic)
        for j in stmt.joins:
            right_table = qe_db.get_table(j.table.table_name)
            right_scan = SeqScanOp(right_table)
            if j.join_type == 'cross':
                plan = NestedLoopJoinOp(plan, right_scan, predicate=None, join_type='cross')
            elif j.condition is not None:
                qe_pred = self._sql_to_qe_expr(j.condition)
                if j.join_type == 'inner':
                    if self._is_equijoin(j.condition):
                        plan = HashJoinOp(
                            plan, right_scan,
                            left_key=self._extract_join_key(j.condition, 'left'),
                            right_key=self._extract_join_key(j.condition, 'right'),
                            join_type='inner')
                    else:
                        plan = NestedLoopJoinOp(plan, right_scan, predicate=qe_pred, join_type='inner')
                elif j.join_type == 'left':
                    if self._is_equijoin(j.condition):
                        plan = HashJoinOp(
                            plan, right_scan,
                            left_key=self._extract_join_key(j.condition, 'left'),
                            right_key=self._extract_join_key(j.condition, 'right'),
                            join_type='left')
                    else:
                        plan = NestedLoopJoinOp(plan, right_scan, predicate=qe_pred, join_type='left')
            else:
                plan = NestedLoopJoinOp(plan, right_scan, predicate=None, join_type=j.join_type)

        # WHERE -- use SubqueryFilterOp if subqueries present
        if stmt.where is not None:
            qe_pred = self._sql_to_qe_expr(stmt.where)
            if isinstance(qe_pred, _SubqueryExprBase):
                plan = SubqueryFilterOp(plan, qe_pred, self.db)
            else:
                plan = FilterOp(plan, qe_pred)

        # GROUP BY + aggregates
        has_aggs = self._has_aggregates(stmt.columns)
        if stmt.group_by or has_aggs:
            group_exprs = []
            if stmt.group_by:
                group_exprs = [self._sql_to_qe_expr(g) for g in stmt.group_by]
            agg_calls = self._extract_aggregates(stmt.columns)
            plan = HashAggregateOp(plan, group_exprs, agg_calls)
            if stmt.having:
                having_pred = self._sql_to_qe_expr(stmt.having)
                plan = HavingOp(plan, having_pred)

        # ORDER BY
        if stmt.order_by:
            sort_keys = [(self._sql_to_qe_expr(expr), asc) for expr, asc in stmt.order_by]
            plan = SortOp(plan, sort_keys)

        # SELECT (projection) -- use SubqueryProjectOp if needed
        if not self._is_star_only(stmt.columns):
            projections = self._build_projections(stmt.columns, stmt.group_by, has_aggs)
            if projections:
                has_subquery_proj = any(isinstance(e, _SubqueryExprBase) for e, _ in projections)
                if has_subquery_proj:
                    plan = SubqueryProjectOp(plan, projections, self.db)
                else:
                    plan = ProjectOp(plan, projections)

        # DISTINCT
        if stmt.distinct:
            plan = DistinctOp(plan)

        # LIMIT / OFFSET
        if stmt.limit is not None:
            plan = LimitOp(plan, stmt.limit, stmt.offset or 0)

        return plan

    def compile_select(self, stmt: SelectStmt, txn_id: int):
        """Extended compile_select that handles derived tables."""
        qe_db = QEDatabase()
        table_aliases = {}
        tables_needed = set()
        derived_tables = {}

        from_ref = stmt.from_table
        if from_ref is not None:
            if isinstance(from_ref, DerivedTable):
                result = self.db._exec_select_with_txn(from_ref.subquery.select, txn_id=txn_id)
                alias = from_ref.alias
                derived_tables[alias] = (result, result.columns)
                table_aliases[alias] = alias
            else:
                tables_needed.add(from_ref.table_name)
                if from_ref.alias:
                    table_aliases[from_ref.alias] = from_ref.table_name

        for j in stmt.joins:
            if isinstance(j.table, DerivedTable):
                result = self.db._exec_select_with_txn(j.table.subquery.select, txn_id=txn_id)
                alias = j.table.alias
                derived_tables[alias] = (result, result.columns)
                table_aliases[alias] = alias
            else:
                tables_needed.add(j.table.table_name)
                if j.table.alias:
                    table_aliases[j.table.alias] = j.table.table_name

        # Build alias->real_name map for loading
        alias_to_real = {}
        if isinstance(stmt.from_table, TableRef) and stmt.from_table and stmt.from_table.alias:
            alias_to_real[stmt.from_table.alias] = stmt.from_table.table_name
        for j in stmt.joins:
            if isinstance(j.table, TableRef) and j.table.alias:
                alias_to_real[j.table.alias] = j.table.table_name

        # Load real tables. If a table has an alias, register under alias name
        # AND create the table ref to use the alias.
        loaded_names = set()
        for tname in tables_needed:
            schema = self.storage.catalog.get_table(tname)
            # Check if this table has an alias
            qe_name = tname
            for alias, real in alias_to_real.items():
                if real == tname and alias not in loaded_names:
                    qe_name = alias
                    break
            if qe_name in loaded_names:
                continue
            loaded_names.add(qe_name)
            qe_table = qe_db.create_table(qe_name, schema.column_names())
            rows = self.storage.scan_table(txn_id, tname)
            for _rid, row_data in rows:
                qe_table.insert(row_data)
            # Also register under real name if different (for queries that reference both)
            if qe_name != tname and tname not in loaded_names:
                loaded_names.add(tname)
                qe_table2 = qe_db.create_table(tname, schema.column_names())
                for _rid, row_data in self.storage.scan_table(txn_id, tname):
                    qe_table2.insert(row_data)

        # Load derived tables
        for alias, (result, columns) in derived_tables.items():
            qe_table = qe_db.create_table(alias, columns)
            for row in result.rows:
                row_dict = dict(zip(columns, row))
                qe_table.insert(row_dict)

        engine = ExecutionEngine(qe_db)

        # Remap derived tables in stmt to plain TableRef
        effective_stmt = self._remap_stmt(stmt)
        plan = self._build_plan(effective_stmt, qe_db, table_aliases)
        return plan, engine

    def _remap_stmt(self, stmt: SelectStmt) -> SelectStmt:
        """Replace DerivedTable refs and aliased table refs with plain refs using effective names."""
        from_table = stmt.from_table
        needs_remap = False

        if isinstance(from_table, DerivedTable):
            from_table = TableRef(table_name=from_table.alias, alias=None)
            needs_remap = True
        elif isinstance(from_table, TableRef) and from_table and from_table.alias:
            # Remap to use alias as table name for plan building
            from_table = TableRef(table_name=from_table.alias, alias=None)
            needs_remap = True

        joins = []
        for j in stmt.joins:
            if isinstance(j.table, DerivedTable):
                joins.append(JoinClause(
                    join_type=j.join_type,
                    table=TableRef(table_name=j.table.alias, alias=None),
                    condition=j.condition))
                needs_remap = True
            elif isinstance(j.table, TableRef) and j.table.alias:
                joins.append(JoinClause(
                    join_type=j.join_type,
                    table=TableRef(table_name=j.table.alias, alias=None),
                    condition=j.condition))
                needs_remap = True
            else:
                joins.append(j)

        if not needs_remap:
            return stmt

        return SelectStmt(
            columns=stmt.columns, from_table=from_table, joins=joins,
            where=stmt.where, group_by=stmt.group_by, having=stmt.having,
            order_by=stmt.order_by, limit=stmt.limit, offset=stmt.offset,
            distinct=stmt.distinct)


# =============================================================================
# SubqueryDB
# =============================================================================

def _substitute_outer_refs(node, inner_tables: set, outer_data: dict):
    """Recursively substitute outer row column references with literal values.

    A column ref like `departments.id` is an outer reference if 'departments'
    is not in inner_tables. We replace it with SqlLiteral(outer_data[key]).
    """
    if node is None:
        return None

    if isinstance(node, SqlColumnRef):
        if node.table and node.table not in inner_tables:
            # Outer reference -- look up in outer_data
            # Try both "table.column" key and just "column" key
            key = f"{node.table}.{node.column}"
            if key in outer_data:
                return SqlLiteral(outer_data[key])
            if node.column in outer_data:
                return SqlLiteral(outer_data[node.column])
            # Not found in outer data -- leave as-is (might be a real error)
            return node
        return node

    if isinstance(node, SqlComparison):
        return SqlComparison(
            op=node.op,
            left=_substitute_outer_refs(node.left, inner_tables, outer_data),
            right=_substitute_outer_refs(node.right, inner_tables, outer_data))

    if isinstance(node, SqlLogic):
        return SqlLogic(
            op=node.op,
            operands=[_substitute_outer_refs(o, inner_tables, outer_data) for o in node.operands])

    if isinstance(node, SqlBinOp):
        return SqlBinOp(
            op=node.op,
            left=_substitute_outer_refs(node.left, inner_tables, outer_data),
            right=_substitute_outer_refs(node.right, inner_tables, outer_data))

    if isinstance(node, SqlIsNull):
        return SqlIsNull(
            expr=_substitute_outer_refs(node.expr, inner_tables, outer_data),
            negated=node.negated)

    if isinstance(node, SqlBetween):
        return SqlBetween(
            expr=_substitute_outer_refs(node.expr, inner_tables, outer_data),
            low=_substitute_outer_refs(node.low, inner_tables, outer_data),
            high=_substitute_outer_refs(node.high, inner_tables, outer_data))

    if isinstance(node, SqlInList):
        return SqlInList(
            expr=_substitute_outer_refs(node.expr, inner_tables, outer_data),
            values=[_substitute_outer_refs(v, inner_tables, outer_data) for v in node.values])

    if isinstance(node, SqlInSubquery):
        return SqlInSubquery(
            expr=_substitute_outer_refs(node.expr, inner_tables, outer_data),
            subquery=node.subquery,  # don't recurse into subqueries -- they handle their own correlation
            negated=node.negated)

    if isinstance(node, SqlExistsExpr):
        return node  # EXISTS subqueries handle their own correlation

    if isinstance(node, SqlAnyAll):
        return SqlAnyAll(
            expr=_substitute_outer_refs(node.expr, inner_tables, outer_data),
            op=node.op,
            quantifier=node.quantifier,
            subquery=node.subquery)

    if isinstance(node, SqlSubquery):
        return node  # Subqueries handle their own correlation

    if isinstance(node, SqlFuncCall):
        return SqlFuncCall(
            func_name=node.func_name,
            args=[_substitute_outer_refs(a, inner_tables, outer_data) for a in node.args],
            distinct=node.distinct)

    if isinstance(node, SqlAggCall):
        return node  # Aggregates reference inner query

    if isinstance(node, SqlCase):
        whens = [(_substitute_outer_refs(c, inner_tables, outer_data),
                  _substitute_outer_refs(r, inner_tables, outer_data))
                 for c, r in node.whens]
        else_r = _substitute_outer_refs(node.else_result, inner_tables, outer_data)
        return SqlCase(whens=whens, else_result=else_r)

    # Literals, stars, etc. -- return as-is
    return node


def _expr_has_subquery(node) -> bool:
    """Recursively check if an expression contains subqueries."""
    if isinstance(node, (SqlSubquery, SqlInSubquery, SqlExistsExpr, SqlAnyAll)):
        return True
    if isinstance(node, SqlLogic):
        return any(_expr_has_subquery(o) for o in node.operands)
    if isinstance(node, SqlComparison):
        return _expr_has_subquery(node.left) or _expr_has_subquery(node.right)
    if isinstance(node, SqlBinOp):
        return _expr_has_subquery(node.left) or _expr_has_subquery(node.right)
    return False


class SubqueryDB(MiniDB):
    """MiniDB with full SQL subquery support."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        self.compiler = SubqueryCompiler(self.storage, self)
        self._current_txn_id: Optional[int] = None

    def execute(self, sql: str) -> ResultSet:
        stmt = parse_sql_subquery(sql)
        return self._execute_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        stmts = parse_sql_subquery_multi(sql)
        return [self._execute_stmt(stmt) for stmt in stmts]

    def _execute_stmt(self, stmt) -> ResultSet:
        if isinstance(stmt, SelectStmt):
            return self._exec_select(stmt)
        return super()._execute_stmt(stmt)

    def _exec_select_with_txn(self, stmt: SelectStmt, txn_id: int = None) -> ResultSet:
        """Execute a SELECT using a specific or current txn_id. Used by subqueries."""
        if txn_id is None:
            txn_id = self._current_txn_id
        if txn_id is None:
            txn_id = self._get_txn()
            auto = self._active_txn is None
        else:
            auto = False

        try:
            plan, engine = self.compiler.compile_select(stmt, txn_id)
            qe_rows = engine.execute(plan)
            result = self._rows_to_result(stmt, qe_rows)
            if auto:
                self.storage.commit(txn_id)
            return result
        except Exception:
            if auto:
                try:
                    self.storage.abort(txn_id)
                except Exception:
                    pass
            raise

    def _exec_select_correlated(self, stmt: SelectStmt, outer_row: Row) -> ResultSet:
        """Execute a subquery with access to outer row values for correlated references.

        Substitutes unresolvable table-qualified column references in the WHERE clause
        with literal values from the outer row.
        """
        txn_id = self._current_txn_id
        if txn_id is None:
            txn_id = self._get_txn()
            auto = self._active_txn is None
        else:
            auto = False

        try:
            # Determine which table names the inner query owns
            # Key rule: if a table has an alias, only the alias is the inner name.
            # The original table name becomes available for outer correlation.
            inner_tables = set()
            if stmt.from_table:
                if isinstance(stmt.from_table, DerivedTable):
                    inner_tables.add(stmt.from_table.alias)
                else:
                    if stmt.from_table.alias:
                        inner_tables.add(stmt.from_table.alias)
                    else:
                        inner_tables.add(stmt.from_table.table_name)
            for j in stmt.joins:
                if isinstance(j.table, DerivedTable):
                    inner_tables.add(j.table.alias)
                else:
                    if j.table.alias:
                        inner_tables.add(j.table.alias)
                    else:
                        inner_tables.add(j.table.table_name)

            # Substitute outer row values in the WHERE clause
            outer_data = dict(outer_row._data) if hasattr(outer_row, '_data') else {}
            substituted_where = _substitute_outer_refs(stmt.where, inner_tables, outer_data) if stmt.where else None

            # Build a new stmt with substituted WHERE
            corr_stmt = SelectStmt(
                columns=stmt.columns,
                from_table=stmt.from_table,
                joins=stmt.joins,
                where=substituted_where,
                group_by=stmt.group_by,
                having=stmt.having,
                order_by=stmt.order_by,
                limit=stmt.limit,
                offset=stmt.offset,
                distinct=stmt.distinct,
            )

            plan, engine = self.compiler.compile_select(corr_stmt, txn_id)
            qe_rows = engine.execute(plan)
            result = self._rows_to_result(corr_stmt, qe_rows)
            if auto:
                self.storage.commit(txn_id)
            return result
        except Exception:
            if auto:
                try:
                    self.storage.abort(txn_id)
                except Exception:
                    pass
            raise

    def _exec_select(self, stmt: SelectStmt) -> ResultSet:
        txn_id = self._get_txn()
        auto = self._active_txn is None

        # Save txn_id so subqueries can find it
        prev_txn = self._current_txn_id
        self._current_txn_id = txn_id

        try:
            plan, engine = self.compiler.compile_select(stmt, txn_id)
            qe_rows = engine.execute(plan)
            result = self._rows_to_result(stmt, qe_rows)
            if auto:
                self.storage.commit(txn_id)
            return result
        except Exception:
            if auto:
                try:
                    self.storage.abort(txn_id)
                except Exception:
                    pass
            raise
        finally:
            self._current_txn_id = prev_txn

    def _rows_to_result(self, stmt: SelectStmt, qe_rows) -> ResultSet:
        """Convert QE rows to ResultSet."""
        if qe_rows:
            if self.compiler._is_star_only(stmt.columns):
                ordered_cols = []
                from_ref = stmt.from_table
                if from_ref is not None:
                    if isinstance(from_ref, DerivedTable):
                        tname = from_ref.alias
                    else:
                        tname = from_ref.table_name
                    if self.storage.catalog.has_table(tname):
                        schema = self.storage.catalog.get_table(tname)
                        for cn in schema.column_names():
                            ordered_cols.append((f"{tname}.{cn}", cn))
                    else:
                        for k in qe_rows[0].columns():
                            cn = k.split('.')[-1] if '.' in k else k
                            ordered_cols.append((k, cn))
                for j in stmt.joins:
                    if isinstance(j.table, DerivedTable):
                        jtname = j.table.alias
                    else:
                        jtname = j.table.table_name
                    if self.storage.catalog.has_table(jtname):
                        jschema = self.storage.catalog.get_table(jtname)
                        for cn in jschema.column_names():
                            ordered_cols.append((f"{jtname}.{cn}", cn))
                    else:
                        for k in qe_rows[0].columns():
                            if k.startswith(f"{jtname}."):
                                cn = k.split('.')[-1]
                                ordered_cols.append((k, cn))

                if ordered_cols:
                    clean_cols = [c[1] for c in ordered_cols]
                    qe_keys = [c[0] for c in ordered_cols]
                    rows = [[r.get(k) for k in qe_keys] for r in qe_rows]
                else:
                    columns = qe_rows[0].columns()
                    clean_cols = [c.split('.')[-1] if '.' in c else c for c in columns]
                    rows = [list(r.values()) for r in qe_rows]
            else:
                columns = qe_rows[0].columns()
                clean_cols = [c.split('.')[-1] if '.' in c else c for c in columns]
                rows = [list(r.values()) for r in qe_rows]
        else:
            clean_cols = self._derive_column_names(stmt)
            rows = []

        return ResultSet(columns=clean_cols, rows=rows)

    def _derive_column_names(self, stmt: SelectStmt) -> List[str]:
        columns = []
        for se in stmt.columns:
            if isinstance(se.expr, SqlStar):
                from_ref = stmt.from_table
                if from_ref is not None:
                    if isinstance(from_ref, DerivedTable):
                        columns.append('*')
                    elif self.storage.catalog.has_table(from_ref.table_name):
                        schema = self.storage.catalog.get_table(from_ref.table_name)
                        columns.extend(schema.column_names())
            elif se.alias:
                columns.append(se.alias)
            elif isinstance(se.expr, SqlColumnRef):
                columns.append(se.expr.column)
            elif isinstance(se.expr, SqlAggCall):
                columns.append(self.compiler._agg_alias(se.expr))
            else:
                columns.append(f"col_{len(columns)}")
        return columns

    # -- DML with subquery support --

    def _exec_update(self, stmt: UpdateStmt) -> ResultSet:
        if stmt.where is not None and _expr_has_subquery(stmt.where):
            return self._exec_update_with_subquery(stmt)
        return super()._exec_update(stmt)

    def _exec_update_with_subquery(self, stmt: UpdateStmt) -> ResultSet:
        txn_id = self._get_txn()
        prev_txn = self._current_txn_id
        self._current_txn_id = txn_id
        try:
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0
            pred = self.compiler._sql_to_qe_expr(stmt.where)
            for rowid, row_data in all_rows:
                qe_row = Row(row_data)
                if subquery_eval(pred, qe_row):
                    updates = {}
                    for col, val_expr in stmt.assignments:
                        if isinstance(val_expr, SqlLiteral):
                            updates[col] = val_expr.value
                        else:
                            qe_expr = self.compiler._sql_to_qe_expr(val_expr)
                            updates[col] = _original_eval_expr(qe_expr, qe_row)
                    self.storage.update_row(txn_id, stmt.table_name, rowid, updates)
                    count += 1
            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"UPDATE {count}", rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise
        finally:
            self._current_txn_id = prev_txn

    def _exec_delete(self, stmt: DeleteStmt) -> ResultSet:
        if stmt.where is not None and _expr_has_subquery(stmt.where):
            return self._exec_delete_with_subquery(stmt)
        return super()._exec_delete(stmt)

    def _exec_delete_with_subquery(self, stmt: DeleteStmt) -> ResultSet:
        txn_id = self._get_txn()
        prev_txn = self._current_txn_id
        self._current_txn_id = txn_id
        try:
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0
            pred = self.compiler._sql_to_qe_expr(stmt.where)
            for rowid, row_data in all_rows:
                qe_row = Row(row_data)
                if subquery_eval(pred, qe_row):
                    self.storage.delete_row(txn_id, stmt.table_name, rowid)
                    count += 1
            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"DELETE {count}", rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise
        finally:
            self._current_txn_id = prev_txn
