"""
C263: CREATE TABLE AS SELECT (CTAS)
Extends C262 (Views) with CREATE TABLE ... AS SELECT.

Features:
- CREATE TABLE name AS SELECT ... (creates table from query results)
- CREATE TABLE IF NOT EXISTS name AS SELECT ...
- Column types inferred from data (INT, FLOAT, TEXT, BOOL)
- Works with views as source
- Works with JOINs, WHERE, GROUP BY, ORDER BY, LIMIT
- Works with expressions and aliases in SELECT
- Explicit column list: CREATE TABLE name (col1, col2) AS SELECT ...
- Column count validation when explicit columns provided
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C261_foreign_keys')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from views import (
    ViewDB, ViewDef, ViewParser, ViewRegistry,
    CreateViewStmt, DropViewStmt,
    parse_view_sql, parse_view_sql_multi,
)
from mini_database import (
    MiniDB, ResultSet, StorageEngine, QueryCompiler, Catalog, TableSchema,
    ColumnDef, CatalogError, CompileError, ParseError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlIsNull, SqlBetween, SqlInList, SqlFuncCall, SqlCase,
    TokenType, Token, Lexer, Parser, KEYWORDS,
    SelectExpr, TableRef,
)
from query_executor import Row, eval_expr


# =============================================================================
# AST node for CTAS
# =============================================================================

@dataclass
class CreateTableAsSelectStmt:
    """CREATE TABLE name [(col1, col2, ...)] AS SELECT ..."""
    table_name: str
    select_stmt: SelectStmt
    column_names: Optional[List[str]] = None  # explicit column names
    if_not_exists: bool = False


# =============================================================================
# Extended Parser with CTAS support
# =============================================================================

class CTASParser(ViewParser):
    """Parser extended with CREATE TABLE ... AS SELECT support."""

    def _parse_create(self):
        """Override to handle CTAS in addition to views and regular tables."""
        self.expect(TokenType.CREATE)

        # CREATE INDEX
        if self.peek().type == TokenType.INDEX:
            return self._parse_create_index()

        # CREATE OR REPLACE VIEW
        or_replace = False
        if self.peek().type == TokenType.OR:
            self.advance()  # OR
            self._expect_ident('replace')
            or_replace = True

        # CREATE [OR REPLACE] VIEW
        if self._check_ident('view'):
            return self._parse_create_view(or_replace)

        if or_replace:
            raise ParseError("Expected VIEW after CREATE OR REPLACE")

        # CREATE TABLE ... possibly AS SELECT
        return self._parse_create_table_or_ctas()

    def _parse_create_table_or_ctas(self):
        """Parse CREATE TABLE, detecting AS SELECT for CTAS."""
        self.expect(TokenType.TABLE)
        if_not_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True

        table_name = self.expect(TokenType.IDENT).value

        # Check for AS SELECT (no column defs)
        if self.peek().type == TokenType.AS:
            self.advance()  # AS
            select_stmt = self._parse_select()
            return CreateTableAsSelectStmt(
                table_name=table_name,
                select_stmt=select_stmt,
                column_names=None,
                if_not_exists=if_not_exists,
            )

        self.expect(TokenType.LPAREN)

        # Could be column definitions or column name list for CTAS
        # Peek ahead to determine: if we see just names followed by ) AS SELECT,
        # it's CTAS with explicit column names.
        # If we see name TYPE ..., it's a regular CREATE TABLE.
        saved_pos = self.pos

        # Try to parse as column name list (IDENT, IDENT, ...) AS SELECT
        col_names = self._try_parse_column_name_list()
        if col_names is not None:
            return CreateTableAsSelectStmt(
                table_name=table_name,
                select_stmt=col_names['select_stmt'],
                column_names=col_names['columns'],
                if_not_exists=if_not_exists,
            )

        # Restore and parse as normal CREATE TABLE
        self.pos = saved_pos
        col_defs = self._parse_column_defs()
        self.expect(TokenType.RPAREN)

        # Check for trailing AS SELECT (CREATE TABLE t (col1 INT, col2 TEXT) AS SELECT ...)
        # This is not standard SQL but some databases support it -- we don't.
        return CreateTableStmt(
            table_name=table_name,
            columns=col_defs,
            if_not_exists=if_not_exists,
        )

    def _try_parse_column_name_list(self):
        """Try to parse (col1, col2, ...) AS SELECT ... for CTAS.
        Returns None if this doesn't look like a CTAS column list."""
        saved = self.pos
        try:
            columns = [self.expect(TokenType.IDENT).value]
            while self.match(TokenType.COMMA):
                columns.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.RPAREN)
            self.expect(TokenType.AS)
            select_stmt = self._parse_select()
            return {'columns': columns, 'select_stmt': select_stmt}
        except (ParseError, Exception):
            self.pos = saved
            return None


# =============================================================================
# Parse functions
# =============================================================================

def parse_ctas_sql(sql: str):
    """Parse a single SQL statement with CTAS + View + FK + CHECK support."""
    lexer = Lexer(sql)
    parser = CTASParser(lexer.tokens)
    return parser.parse()


def parse_ctas_sql_multi(sql: str):
    """Parse multiple SQL statements with CTAS support."""
    lexer = Lexer(sql)
    parser = CTASParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# Type inference
# =============================================================================

def _infer_type(value) -> str:
    """Infer SQL type from a Python value."""
    if value is None:
        return 'TEXT'
    if isinstance(value, bool):
        return 'BOOL'
    if isinstance(value, int):
        return 'INT'
    if isinstance(value, float):
        return 'FLOAT'
    return 'TEXT'


def _infer_column_type(values: List) -> str:
    """Infer column type from a list of values. Non-null values take precedence."""
    types = set()
    for v in values:
        if v is not None:
            types.add(_infer_type(v))
    if not types:
        return 'TEXT'
    # Priority: FLOAT > INT > BOOL > TEXT
    if 'FLOAT' in types:
        return 'FLOAT'
    if 'INT' in types:
        return 'INT'
    if 'BOOL' in types:
        return 'BOOL'
    return 'TEXT'


# =============================================================================
# CTASDB -- Database with CTAS + View support
# =============================================================================

class CTASDB(ViewDB):
    """ViewDB extended with CREATE TABLE AS SELECT."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with CTAS + View support."""
        stmt = parse_ctas_sql(sql)
        return self._execute_ctas_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_ctas_sql_multi(sql)
        return [self._execute_ctas_stmt(s) for s in stmts]

    def _execute_ctas_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling CTAS-specific types."""
        if isinstance(stmt, CreateTableAsSelectStmt):
            return self._exec_ctas(stmt)
        else:
            return self._execute_view_stmt(stmt)

    def _exec_ctas(self, stmt: CreateTableAsSelectStmt) -> ResultSet:
        """Execute CREATE TABLE ... AS SELECT."""
        table_name = stmt.table_name

        # Check if table already exists
        try:
            self.storage.catalog.get_table(table_name)
            if stmt.if_not_exists:
                return ResultSet(columns=['result'], rows=[['Table already exists']])
            raise CatalogError(f"Table '{table_name}' already exists")
        except CatalogError as e:
            if "already exists" in str(e):
                if stmt.if_not_exists:
                    return ResultSet(columns=['result'], rows=[['Table already exists']])
                raise
            pass  # table doesn't exist, good

        # Check for view name conflict
        if self._views.exists(table_name):
            raise CatalogError(
                f"View '{table_name}' already exists; cannot create table with same name"
            )

        # Execute the SELECT query (with view expansion)
        result = self._exec_select_with_views(stmt.select_stmt)

        # Determine column names
        if stmt.column_names:
            if len(stmt.column_names) != len(result.columns):
                raise CatalogError(
                    f"CTAS column list has {len(stmt.column_names)} entries "
                    f"but SELECT produces {len(result.columns)} columns"
                )
            col_names = stmt.column_names
        else:
            col_names = list(result.columns)

        # Infer types from data
        col_types = []
        for i in range(len(col_names)):
            values = [row[i] for row in result.rows if i < len(row)]
            col_types.append(_infer_column_type(values))

        # Create table schema
        col_defs = []
        for cname, col_type in zip(col_names, col_types):
            col_defs.append(ColumnDef(name=cname, col_type=col_type))

        self.storage.catalog.create_table(table_name, col_defs)

        # Insert all rows
        if result.rows:
            txn_id = self.storage.txn_manager.begin()
            try:
                for row in result.rows:
                    row_data = {}
                    for i, col in enumerate(col_names):
                        row_data[col] = row[i] if i < len(row) else None
                    self.storage.insert_row(txn_id, table_name, row_data)
                self.storage.txn_manager.commit(txn_id)
            except Exception:
                self.storage.txn_manager.abort(txn_id)
                raise

        row_count = len(result.rows)
        return ResultSet(
            columns=['result'],
            rows=[[f"Table created with {row_count} rows"]]
        )
