"""
C256: Derived Tables (Subqueries in FROM clause)
Extends C255 (Subqueries)

Adds derived table support:
- FROM (SELECT ...) AS alias -- subquery as table source
- Derived tables in JOINs: JOIN (SELECT ...) AS alias ON ...
- Derived tables with aggregation, WHERE, GROUP BY
- Derived tables composed with CTEs, set operations, subqueries
- Nested derived tables: FROM (SELECT * FROM (SELECT ...) AS inner) AS outer
- Multiple derived tables via JOINs
- Column aliasing through derived table alias
- Derived tables with correlated subqueries
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import chain
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C255_subqueries'))
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

from subqueries import (
    SubqueryDB, SubqueryLexer, SubqueryParser,
    SqlSubquery, SqlExistsSubquery, SqlInSubquery,
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
from set_operations import SetOpStmt, SET_OP_WORDS
from transaction_manager import IsolationLevel


# =============================================================================
# Derived Table AST Node
# =============================================================================

@dataclass
class DerivedTable:
    """A subquery used as a table source in FROM or JOIN.
    FROM (SELECT ...) AS alias
    """
    stmt: Any       # SelectStmt | CTESelectStmt | SetOpStmt
    alias: str      # Required alias for derived tables


# =============================================================================
# Lexer
# =============================================================================

class DerivedTableLexer(SubqueryLexer):
    """Lexer for derived tables -- no new tokens needed."""
    def __init__(self, sql: str):
        super().__init__(sql)


# =============================================================================
# Parser
# =============================================================================

class DerivedTableParser(SubqueryParser):
    """Parser that recognizes derived tables in FROM and JOIN positions."""

    def __init__(self, tokens: List[Token]):
        super().__init__(tokens)

    def _parse_table_ref(self) -> Any:
        """Override to detect (SELECT ...) AS alias as derived table."""
        if self._is_derived_table_start():
            return self._parse_derived_table()
        return super()._parse_table_ref()

    def _is_derived_table_start(self) -> bool:
        """Check if current position is a derived table: ( SELECT ... )"""
        if self.peek().type != TokenType.LPAREN:
            return False
        # Look ahead past ( to see if SELECT or WITH follows
        pos = self.pos + 1
        # Skip nested parens
        while pos < len(self.tokens) and self.tokens[pos].type == TokenType.LPAREN:
            pos += 1
        if pos < len(self.tokens):
            val = self.tokens[pos].value
            if val and val.upper() in ('SELECT', 'WITH'):
                return True
        return False

    def _parse_derived_table(self) -> DerivedTable:
        """Parse (SELECT ...) AS alias."""
        self.expect(TokenType.LPAREN)

        # Parse inner statement (could be SELECT, CTE, or set operation)
        inner_stmt = self._parse_inner_select()

        self.expect(TokenType.RPAREN)

        # Alias is required for derived tables
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENT).value
        elif self.peek().type == TokenType.IDENT and self.peek().value.lower() not in KEYWORDS:
            alias = self.advance().value
        else:
            raise ParseError("Derived table requires an alias (AS name)")

        return DerivedTable(stmt=inner_stmt, alias=alias)

    def _parse_inner_select(self) -> Any:
        """Parse a SELECT inside parens that might be a CTE or set op."""
        word = self._peek_word()

        # WITH ... SELECT (CTE)
        if word == 'with':
            stmt = self._parse_cte_statement()
            # Check for set operations after CTE
            pw = self._peek_word()
            if pw in ('union', 'intersect', 'except'):
                return self._parse_set_tail(stmt)
            return stmt

        # Regular SELECT
        stmt = self._parse_select()
        # Check for set operations
        pw = self._peek_word()
        if pw in ('union', 'intersect', 'except'):
            return self._parse_set_tail(stmt)
        return stmt

    def _parse_set_tail(self, left: Any) -> SetOpStmt:
        """Parse UNION/INTERSECT/EXCEPT after an initial SELECT."""
        op = self.advance().value.upper()
        all_flag = False
        if self.peek().value and self.peek().value.upper() == 'ALL':
            self.advance()
            all_flag = True
        right = self._parse_select()
        return SetOpStmt(op=op, left=left, right=right, all=all_flag)


# =============================================================================
# Database Engine
# =============================================================================

class DerivedTableDB(SubqueryDB):
    """Database with derived table support."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)

    # -----------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------
    # Execute entry point
    # -----------------------------------------------------------------

    def _parse_derived_multi(self, sql: str) -> List[Any]:
        """Parse SQL into multiple statements using DerivedTableParser."""
        lexer = DerivedTableLexer(sql)
        parser = DerivedTableParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with derived table support."""
        stmts = self._parse_derived_multi(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_subquery_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute SQL and return all results."""
        stmts = self._parse_derived_multi(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_subquery_stmt(stmt))
        return results

    # -----------------------------------------------------------------
    # Source Row Gathering -- fully overridden for derived tables
    # -----------------------------------------------------------------

    def _get_source_rows(self, stmt: SelectStmt) -> List[Dict[str, Any]]:
        """Get source rows, handling derived tables in FROM."""
        if stmt.from_table is None:
            return [{}]

        if isinstance(stmt.from_table, DerivedTable):
            rows = self._resolve_derived_table(stmt.from_table)
        else:
            rows = self._get_regular_source_rows(stmt)

        # Process JOINs
        for join in stmt.joins:
            rows = self._do_join(rows, join, stmt)

        return rows

    def _get_regular_source_rows(self, stmt: SelectStmt) -> List[Dict[str, Any]]:
        """Get source rows from a regular table (not derived), without JOINs."""
        rows = []
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

        return rows

    def _resolve_derived_table(self, dt: DerivedTable) -> List[Dict[str, Any]]:
        """Execute a derived table subquery and return rows as dicts."""
        result = self._run_subquery(dt.stmt, outer_row=None)
        columns = result.columns
        rows_out = []

        for row_tuple in result.rows:
            row = {}
            for i, col_name in enumerate(columns):
                val = row_tuple[i] if i < len(row_tuple) else None
                row[col_name] = val
                row[f"{dt.alias}.{col_name}"] = val
            rows_out.append(row)

        return rows_out

    # -----------------------------------------------------------------
    # JOIN -- fully overridden for derived table support
    # -----------------------------------------------------------------

    def _do_join(self, left_rows: List[Dict], join: JoinClause,
                 stmt: SelectStmt) -> List[Dict]:
        """Execute a JOIN, handling derived tables on the right side."""
        if isinstance(join.table, DerivedTable):
            right_rows = self._resolve_derived_table(join.table)
        else:
            right_table = join.table.table_name
            right_alias = join.table.alias or right_table
            right_rows = self._get_table_rows(right_table, right_alias)

        # Nested loop join
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
                combined = dict(lrow)
                result.append(combined)

        return result

    def _get_table_rows(self, table_name: str, alias: str) -> List[Dict]:
        """Get rows from a regular table or CTE."""
        rows = []
        if hasattr(self, '_cte_tables') and table_name.lower() in self._cte_tables:
            cte_rows = self._cte_tables[table_name.lower()]
            cte_cols = self._cte_columns.get(table_name.lower(), [])
            for cte_row in cte_rows:
                row = {}
                for col_name in cte_cols:
                    val = cte_row.get(col_name)
                    row[col_name] = val
                    row[f"{alias}.{col_name}"] = val
                    row[f"{table_name}.{col_name}"] = val
                rows.append(row)
        else:
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
        return rows

    # -----------------------------------------------------------------
    # Override select routing to ensure derived tables use full path
    # -----------------------------------------------------------------

    def _exec_select_with_cte_context(self, stmt) -> ResultSet:
        """Override to handle derived tables before CTE routing."""
        if self._has_derived_tables(stmt):
            return self._exec_select_full_subquery(stmt)
        return super()._exec_select_with_cte_context(stmt)

    def _exec_select_standard(self, stmt: SelectStmt) -> ResultSet:
        """Route SELECT through derived-aware path if needed."""
        if self._has_derived_tables(stmt):
            return self._exec_select_full_subquery(stmt)
        return super()._exec_select_standard(stmt)

    def _exec_select_from_cte(self, stmt: SelectStmt) -> ResultSet:
        """Route CTE SELECT through derived-aware path if needed."""
        if self._has_derived_tables(stmt):
            return self._exec_select_full_subquery(stmt)
        return super()._exec_select_from_cte(stmt)

    def _has_derived_tables(self, stmt) -> bool:
        """Check if statement uses derived tables."""
        if not isinstance(stmt, SelectStmt):
            return False
        if isinstance(getattr(stmt, 'from_table', None), DerivedTable):
            return True
        for join in getattr(stmt, 'joins', []):
            if isinstance(getattr(join, 'table', None), DerivedTable):
                return True
        return False
