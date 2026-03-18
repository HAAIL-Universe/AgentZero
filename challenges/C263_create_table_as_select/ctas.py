"""
C263: CREATE TABLE ... AS SELECT (CTAS)
Extends C262 (Views) with CTAS support.

Features:
- CREATE TABLE name AS SELECT ... (create + populate in one statement)
- CREATE TABLE IF NOT EXISTS name AS SELECT ...
- Column types inferred from SELECT result values
- Explicit column names: CREATE TABLE name (col1, col2) AS SELECT ...
- Empty result sets: creates table with column names, no rows
- Computed columns, aliases, expressions, aggregates
- Subquery-based CTAS
- Works with existing views, joins, GROUP BY, etc.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C262_views')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from views import (
    ViewDB, ViewParser, parse_view_sql, parse_view_sql_multi,
    ViewDef, ViewRegistry, CreateViewStmt, DropViewStmt,
)
from mini_database import (
    ResultSet, ColumnDef, CreateTableStmt,
    SelectStmt, TokenType, Token, Lexer, ParseError, DatabaseError,
    CatalogError,
)


# =============================================================================
# AST Node
# =============================================================================

@dataclass
class CreateTableAsSelectStmt:
    """CREATE TABLE [IF NOT EXISTS] name [(col1, col2)] AS SELECT ..."""
    table_name: str
    select_stmt: SelectStmt
    if_not_exists: bool = False
    column_names: List[str] = field(default_factory=list)


# =============================================================================
# Extended Parser
# =============================================================================

class CTASParser(ViewParser):
    """Parser extended with CREATE TABLE ... AS SELECT support."""

    def _parse_create_table_standard(self):
        """Override to detect AS keyword after table name for CTAS."""
        self.expect(TokenType.TABLE)
        if_not_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True
        table_name = self.expect(TokenType.IDENT).value

        # Check for CTAS: either AS directly, or (col_list) AS
        if self.peek().type == TokenType.LPAREN:
            # Could be column defs OR column name list for CTAS
            # Peek ahead: if next-next after LPAREN is IDENT followed by COMMA or RPAREN
            # (no type keyword), it's a CTAS column list
            save_pos = self.pos
            self.advance()  # consume LPAREN

            # Try to detect: is this (name, name, ...) AS or (name TYPE, ...)?
            first_ident = self.peek()
            if first_ident.type == TokenType.IDENT:
                after_first = self._lookahead(1)
                if after_first and after_first.type in (TokenType.COMMA, TokenType.RPAREN):
                    # CTAS column name list: (col1, col2, ...)
                    col_names = [self.expect(TokenType.IDENT).value]
                    while self.match(TokenType.COMMA):
                        col_names.append(self.expect(TokenType.IDENT).value)
                    self.expect(TokenType.RPAREN)
                    self.expect(TokenType.AS)
                    select_stmt = self._parse_select()
                    return CreateTableAsSelectStmt(
                        table_name=table_name,
                        select_stmt=select_stmt,
                        if_not_exists=if_not_exists,
                        column_names=col_names,
                    )

            # Not CTAS -- backtrack and parse as normal column defs
            self.pos = save_pos
            self.advance()  # re-consume LPAREN
            col_defs = self._parse_column_defs()
            self.expect(TokenType.RPAREN)
            return CreateTableStmt(
                table_name=table_name,
                columns=col_defs,
                if_not_exists=if_not_exists,
            )

        # No LPAREN -- check for AS (CTAS without explicit columns)
        if self.peek().type == TokenType.AS:
            self.advance()  # consume AS
            select_stmt = self._parse_select()
            return CreateTableAsSelectStmt(
                table_name=table_name,
                select_stmt=select_stmt,
                if_not_exists=if_not_exists,
            )

        # Should not reach here for valid SQL
        raise ParseError(f"Expected '(' or AS after table name '{table_name}'")

    def _lookahead(self, offset: int):
        """Look ahead by offset tokens from current position without consuming."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None


# =============================================================================
# Parse functions
# =============================================================================

def parse_ctas_sql(sql: str):
    """Parse a single SQL statement with CTAS + view support."""
    lexer = Lexer(sql)
    parser = CTASParser(lexer.tokens)
    return parser.parse()


def parse_ctas_sql_multi(sql: str):
    """Parse multiple SQL statements with CTAS + view support."""
    lexer = Lexer(sql)
    parser = CTASParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# Type Inference
# =============================================================================

def _infer_column_type(values: list) -> str:
    """Infer SQL column type from a list of Python values."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "TEXT"  # default for all-NULL columns

    # Check types in priority order
    has_float = False
    has_int = False
    has_bool = False
    has_str = False
    for v in non_null:
        if isinstance(v, bool):
            has_bool = True
        elif isinstance(v, float):
            has_float = True
        elif isinstance(v, int):
            has_int = True
        elif isinstance(v, str):
            has_str = True

    # Any string means TEXT (mixed types default to TEXT)
    if has_str:
        return "TEXT"
    if has_float:
        return "FLOAT"
    if has_int and not has_bool:
        return "INTEGER"
    if has_bool and not has_int and not has_float:
        return "BOOLEAN"
    return "TEXT"


# =============================================================================
# CTASDB -- Database with CTAS support
# =============================================================================

class CTASDB(ViewDB):
    """ViewDB extended with CREATE TABLE ... AS SELECT."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with CTAS support."""
        stmt = parse_ctas_sql(sql)
        return self._execute_ctas_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements with CTAS support."""
        stmts = parse_ctas_sql_multi(sql)
        return [self._execute_ctas_stmt(s) for s in stmts]

    def _execute_ctas_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling CTAS type."""
        if isinstance(stmt, CreateTableAsSelectStmt):
            return self._exec_ctas(stmt)
        return self._execute_view_stmt(stmt)

    def _exec_ctas(self, stmt: CreateTableAsSelectStmt) -> ResultSet:
        """Execute CREATE TABLE ... AS SELECT."""
        # Check if table already exists
        try:
            self.storage.catalog.get_table(stmt.table_name)
            if stmt.if_not_exists:
                return ResultSet(
                    columns=[], rows=[],
                    message=f"Table {stmt.table_name} already exists (skipped)",
                )
            raise DatabaseError(f"Table '{stmt.table_name}' already exists")
        except CatalogError:
            pass  # Table doesn't exist, good

        # Also check views
        if self._views.exists(stmt.table_name):
            if stmt.if_not_exists:
                return ResultSet(
                    columns=[], rows=[],
                    message=f"View {stmt.table_name} already exists (skipped)",
                )
            raise DatabaseError(f"Name '{stmt.table_name}' is already used by a view")

        # Execute the SELECT to get data
        select_result = self._execute_view_stmt(stmt.select_stmt)

        # Determine column names
        if stmt.column_names:
            if len(stmt.column_names) != len(select_result.columns):
                raise DatabaseError(
                    f"CTAS column count mismatch: "
                    f"{len(stmt.column_names)} names given, "
                    f"SELECT produces {len(select_result.columns)} columns"
                )
            col_names = stmt.column_names
        else:
            col_names = select_result.columns

        # Infer column types from result data
        col_types = []
        for i in range(len(col_names)):
            values = [row[i] for row in select_result.rows] if select_result.rows else []
            col_types.append(_infer_column_type(values))

        # Build column definitions
        col_defs = [
            ColumnDef(name=name, col_type=ctype)
            for name, ctype in zip(col_names, col_types)
        ]

        # Create the table
        create_stmt = CreateTableStmt(
            table_name=stmt.table_name,
            columns=col_defs,
            if_not_exists=False,
        )
        self._execute_stmt(create_stmt)

        # Insert the data
        if select_result.rows:
            txn_id = self._get_txn()
            try:
                for row in select_result.rows:
                    row_data = {col_names[i]: row[i] for i in range(len(col_names))}
                    self.storage.insert_row(txn_id, stmt.table_name, row_data)
                self._auto_commit(txn_id)
            except Exception:
                self._auto_abort(txn_id)
                raise

        return ResultSet(
            columns=[], rows=[],
            message=f"SELECT {len(select_result.rows)} INTO {stmt.table_name}",
            rows_affected=len(select_result.rows),
        )
