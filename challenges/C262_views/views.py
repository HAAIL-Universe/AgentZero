"""
C262: SQL Views
Extends C247 (Mini Database) with view support.

Features:
- CREATE VIEW name AS SELECT ...
- CREATE OR REPLACE VIEW
- DROP VIEW [IF EXISTS] name
- SELECT from views (transparently expands to underlying query)
- Views with column aliases: CREATE VIEW v (a, b) AS SELECT x, y FROM t
- Nested views (view referencing another view)
- SHOW TABLES includes views (marked as VIEW)
- DESCRIBE works on views (shows view columns)
- Updatable simple views (INSERT/UPDATE/DELETE on single-table, no-aggregate views)
- View dependency tracking (cannot drop table referenced by view)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from mini_database import (
    MiniDB, ResultSet, StorageEngine, QueryCompiler, Catalog, TableSchema,
    ColumnDef, CatalogError, CompileError, ParseError, DatabaseError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, ShowTablesStmt, DescribeStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlIsNull, SqlBetween, SqlInList, SqlFuncCall, SqlCase,
    SqlAggCall, SqlStar, SelectExpr,
    TokenType, Token, Lexer, Parser, parse_sql, KEYWORDS,
    TableRef,
)
from query_executor import (
    Row, ExecutionEngine, eval_expr,
)


# =============================================================================
# View Data Model
# =============================================================================

@dataclass
class ViewDef:
    """Definition of a SQL view."""
    name: str
    select_sql: str             # original SQL of the SELECT
    select_stmt: Any            # parsed SelectStmt AST
    column_aliases: List[str]   # explicit column names (empty = derive from SELECT)

    @property
    def is_updatable(self) -> bool:
        """A view is updatable if it selects from a single table with no
        aggregates, DISTINCT, GROUP BY, HAVING, UNION, or subqueries."""
        stmt = self.select_stmt
        if not isinstance(stmt, SelectStmt):
            return False
        if stmt.distinct:
            return False
        if stmt.group_by:
            return False
        if stmt.having:
            return False
        if stmt.joins:
            return False
        if not stmt.from_table:
            return False
        # Check for aggregates in SELECT list
        for se in stmt.columns:
            if _has_aggregate(se.expr):
                return False
        return True

    @property
    def base_table(self) -> Optional[str]:
        """For updatable views, returns the underlying table name."""
        if not self.is_updatable:
            return None
        return self.select_stmt.from_table.table_name

    def get_column_names(self) -> List[str]:
        """Get the column names this view exposes."""
        if self.column_aliases:
            return list(self.column_aliases)
        # Derive from SELECT list
        names = []
        for se in self.select_stmt.columns:
            if se.alias:
                names.append(se.alias)
            elif isinstance(se.expr, SqlColumnRef):
                names.append(se.expr.column)
            elif isinstance(se.expr, SqlStar):
                names.append('*')  # will be expanded at query time
            elif isinstance(se.expr, SqlAggCall):
                names.append(f"{se.expr.func}_{len(names)}")
            else:
                names.append(f"col_{len(names)}")
        return names


def _has_aggregate(expr) -> bool:
    """Check if an expression contains aggregate functions."""
    if isinstance(expr, SqlAggCall):
        return True
    if isinstance(expr, SqlBinOp):
        return _has_aggregate(expr.left) or _has_aggregate(expr.right)
    if isinstance(expr, SqlFuncCall):
        return any(_has_aggregate(a) for a in expr.args)
    if isinstance(expr, SqlCase):
        for cond, result in expr.whens:
            if _has_aggregate(cond) or _has_aggregate(result):
                return True
        if expr.else_expr and _has_aggregate(expr.else_expr):
            return True
    return False


# =============================================================================
# AST Nodes for View Statements
# =============================================================================

@dataclass
class CreateViewStmt:
    view_name: str
    select_stmt: Any            # SelectStmt
    select_sql: str             # original SQL text
    column_aliases: List[str]   # optional column name list
    or_replace: bool = False

@dataclass
class DropViewStmt:
    view_name: str
    if_exists: bool = False


# =============================================================================
# View Registry
# =============================================================================

class ViewRegistry:
    """Stores and manages view definitions."""

    def __init__(self):
        self._views: Dict[str, ViewDef] = {}

    def create(self, name: str, select_sql: str, select_stmt: Any,
               column_aliases: List[str], or_replace: bool = False):
        """Create or replace a view."""
        if name in self._views and not or_replace:
            raise CatalogError(f"View '{name}' already exists")
        self._views[name] = ViewDef(
            name=name,
            select_sql=select_sql,
            select_stmt=select_stmt,
            column_aliases=column_aliases,
        )

    def drop(self, name: str, if_exists: bool = False):
        """Drop a view."""
        if name not in self._views:
            if if_exists:
                return
            raise CatalogError(f"View '{name}' does not exist")
        del self._views[name]

    def get(self, name: str) -> Optional[ViewDef]:
        return self._views.get(name)

    def exists(self, name: str) -> bool:
        return name in self._views

    def list_views(self) -> List[str]:
        return sorted(self._views.keys())

    def get_dependents(self, table_name: str) -> List[str]:
        """Get views that depend on the given table."""
        dependents = []
        for vname, vdef in self._views.items():
            if self._references_table(vdef.select_stmt, table_name):
                dependents.append(vname)
        return dependents

    def _references_table(self, stmt, table_name: str) -> bool:
        """Check if a SELECT statement references a specific table."""
        if not isinstance(stmt, SelectStmt):
            return False
        if stmt.from_table and stmt.from_table.table_name == table_name:
            return True
        for j in stmt.joins:
            if j.table.table_name == table_name:
                return True
        return False


# =============================================================================
# Extended Parser with VIEW support
# =============================================================================

class ViewParser(Parser):
    """Parser extended with CREATE/DROP VIEW support."""

    VIEW_KEYWORDS = {
        'view': 'VIEW',
        'replace': 'REPLACE',
    }

    def __init__(self, tokens):
        remapped = []
        for t in tokens:
            if t.type == TokenType.IDENT and t.value.lower() in self.VIEW_KEYWORDS:
                remapped.append(Token(TokenType.IDENT, t.value.lower(), t.pos))
            else:
                remapped.append(t)
        super().__init__(remapped)

    def _check_ident(self, value: str) -> bool:
        tok = self.peek()
        return tok.type == TokenType.IDENT and tok.value.lower() == value

    def _expect_ident(self, value: str):
        tok = self.advance()
        if tok.type != TokenType.IDENT or tok.value.lower() != value:
            raise ParseError(f"Expected '{value}', got '{tok.value}'")
        return tok

    def _parse_statement(self):
        """Override to handle DROP VIEW."""
        t = self.peek()
        if t.type == TokenType.DROP:
            return self._parse_drop()
        return super()._parse_statement()

    def _parse_drop(self):
        """Parse DROP TABLE or DROP VIEW."""
        self.expect(TokenType.DROP)
        if self._check_ident('view'):
            return self._parse_drop_view()
        # Fall through to DROP TABLE
        return self._parse_drop_table_body()

    def _parse_drop_table_body(self):
        """Parse DROP TABLE after DROP already consumed."""
        self.expect(TokenType.TABLE)
        if_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.EXISTS)
            if_exists = True
        name = self.expect(TokenType.IDENT).value
        return DropTableStmt(table_name=name, if_exists=if_exists)

    def _parse_drop_view(self):
        """Parse DROP VIEW [IF EXISTS] name."""
        self._expect_ident('view')
        if_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.EXISTS)
            if_exists = True
        name = self.expect(TokenType.IDENT).value
        return DropViewStmt(view_name=name, if_exists=if_exists)

    def _parse_create(self):
        """Override to handle CREATE [OR REPLACE] VIEW."""
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

        # CREATE VIEW / CREATE OR REPLACE VIEW
        if self._check_ident('view'):
            return self._parse_create_view(or_replace)

        if or_replace:
            raise ParseError("Expected VIEW after CREATE OR REPLACE")

        # CREATE TABLE (default)
        return self._parse_create_table_standard()

    def _parse_create_table_standard(self):
        """Parse standard CREATE TABLE (CREATE already consumed)."""
        self.expect(TokenType.TABLE)
        if_not_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True
        table_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        col_defs = self._parse_column_defs()
        self.expect(TokenType.RPAREN)
        return CreateTableStmt(
            table_name=table_name,
            columns=col_defs,
            if_not_exists=if_not_exists
        )

    def _parse_create_view(self, or_replace: bool) -> CreateViewStmt:
        """Parse CREATE [OR REPLACE] VIEW name [(cols)] AS SELECT ..."""
        self._expect_ident('view')
        view_name = self.expect(TokenType.IDENT).value

        # Optional column aliases: (col1, col2, ...)
        column_aliases = []
        if self.peek().type == TokenType.LPAREN:
            self.advance()
            column_aliases.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                column_aliases.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.RPAREN)

        # AS keyword
        self.expect(TokenType.AS)

        # Save position for SQL extraction
        start_pos = self.pos

        # Parse the SELECT statement
        select_stmt = self._parse_select()

        # Extract the SQL text from tokens
        select_sql = self._tokens_to_sql(start_pos, self.pos)

        return CreateViewStmt(
            view_name=view_name,
            select_stmt=select_stmt,
            select_sql=select_sql,
            column_aliases=column_aliases,
            or_replace=or_replace,
        )

    def _tokens_to_sql(self, start: int, end: int) -> str:
        """Reconstruct approximate SQL from token range."""
        parts = []
        for i in range(start, min(end, len(self.tokens))):
            tok = self.tokens[i]
            if tok.type == TokenType.EOF:
                break
            if tok.type == TokenType.STRING:
                parts.append(f"'{tok.value}'")
            else:
                parts.append(str(tok.value))
        return ' '.join(parts)


# =============================================================================
# Parse functions
# =============================================================================

def parse_view_sql(sql: str):
    """Parse a single SQL statement with view support."""
    lexer = Lexer(sql)
    parser = ViewParser(lexer.tokens)
    return parser.parse()

def parse_view_sql_multi(sql: str):
    """Parse multiple SQL statements with view support."""
    lexer = Lexer(sql)
    parser = ViewParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# ViewDB -- Database with view support
# =============================================================================

class ViewDB(MiniDB):
    """MiniDB extended with SQL views."""

    def __init__(self):
        super().__init__()
        self._views = ViewRegistry()

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with view support."""
        stmt = parse_view_sql(sql)
        return self._execute_view_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements with view support."""
        stmts = parse_view_sql_multi(sql)
        return [self._execute_view_stmt(s) for s in stmts]

    def _execute_view_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling view-specific types."""
        if isinstance(stmt, CreateViewStmt):
            return self._exec_create_view(stmt)
        elif isinstance(stmt, DropViewStmt):
            return self._exec_drop_view(stmt)
        elif isinstance(stmt, SelectStmt):
            return self._exec_select_with_views(stmt)
        elif isinstance(stmt, InsertStmt):
            return self._exec_insert_into_view(stmt)
        elif isinstance(stmt, UpdateStmt):
            return self._exec_update_view(stmt)
        elif isinstance(stmt, DeleteStmt):
            return self._exec_delete_from_view(stmt)
        elif isinstance(stmt, DropTableStmt):
            return self._exec_drop_table_with_views(stmt)
        elif isinstance(stmt, ShowTablesStmt):
            return self._exec_show_tables_with_views()
        elif isinstance(stmt, DescribeStmt):
            return self._exec_describe_with_views(stmt)
        else:
            return self._execute_stmt(stmt)

    # -- CREATE VIEW --

    def _exec_create_view(self, stmt: CreateViewStmt) -> ResultSet:
        """Create a view."""
        # Validate: referenced tables/views must exist
        self._validate_view_references(stmt.select_stmt)

        # If column aliases provided, count must match SELECT columns
        if stmt.column_aliases:
            # Try to determine column count from SELECT
            select_cols = self._count_select_columns(stmt.select_stmt)
            if select_cols is not None and len(stmt.column_aliases) != select_cols:
                raise CatalogError(
                    f"VIEW column list has {len(stmt.column_aliases)} entries "
                    f"but SELECT has {select_cols} columns"
                )

        # Check for name conflict with tables
        try:
            self.storage.catalog.get_table(stmt.view_name)
            raise CatalogError(
                f"Cannot create view '{stmt.view_name}': "
                f"a table with that name already exists"
            )
        except CatalogError as e:
            if "already exists" in str(e):
                raise
            pass  # table doesn't exist, good

        self._views.create(
            name=stmt.view_name,
            select_sql=stmt.select_sql,
            select_stmt=stmt.select_stmt,
            column_aliases=stmt.column_aliases,
            or_replace=stmt.or_replace,
        )
        return ResultSet(columns=['result'], rows=[['View created']])

    def _validate_view_references(self, stmt):
        """Validate that all tables/views referenced in SELECT exist."""
        if not isinstance(stmt, SelectStmt):
            return
        if stmt.from_table:
            name = stmt.from_table.table_name
            if not self._views.exists(name):
                try:
                    self.storage.catalog.get_table(name)
                except CatalogError:
                    raise CatalogError(
                        f"Table or view '{name}' does not exist"
                    )
        for j in stmt.joins:
            name = j.table.table_name
            if not self._views.exists(name):
                try:
                    self.storage.catalog.get_table(name)
                except CatalogError:
                    raise CatalogError(
                        f"Table or view '{name}' does not exist"
                    )

    def _count_select_columns(self, stmt) -> Optional[int]:
        """Count columns in a SELECT statement (None if ambiguous due to *)."""
        if not isinstance(stmt, SelectStmt):
            return None
        count = 0
        for se in stmt.columns:
            if isinstance(se.expr, SqlStar):
                return None  # can't count without schema lookup
            count += 1
        return count

    # -- DROP VIEW --

    def _exec_drop_view(self, stmt: DropViewStmt) -> ResultSet:
        """Drop a view."""
        # Check if other views depend on this
        if self._views.exists(stmt.view_name):
            for vname in self._views.list_views():
                if vname == stmt.view_name:
                    continue
                vdef = self._views.get(vname)
                if vdef and self._view_references(vdef.select_stmt, stmt.view_name):
                    raise CatalogError(
                        f"Cannot drop view '{stmt.view_name}': "
                        f"view '{vname}' depends on it"
                    )

        self._views.drop(stmt.view_name, if_exists=stmt.if_exists)
        return ResultSet(columns=['result'], rows=[['View dropped']])

    def _view_references(self, stmt, name: str) -> bool:
        """Check if a SELECT references a given table/view name."""
        if not isinstance(stmt, SelectStmt):
            return False
        if stmt.from_table and stmt.from_table.table_name == name:
            return True
        for j in stmt.joins:
            if j.table.table_name == name:
                return True
        return False

    # -- SELECT from views --

    def _exec_select_with_views(self, stmt: SelectStmt) -> ResultSet:
        """Execute SELECT, expanding view references."""
        result = self._expand_view_refs(stmt)
        # _expand_view_refs returns ResultSet if view was materialized,
        # or SelectStmt if no view was involved
        if isinstance(result, ResultSet):
            return result
        return self._execute_stmt(result)

    def _expand_view_refs(self, stmt: SelectStmt) -> SelectStmt:
        """If FROM clause references a view, rewrite the query to use the view's SELECT."""
        if not isinstance(stmt, SelectStmt):
            return stmt

        if stmt.from_table:
            view_def = self._views.get(stmt.from_table.table_name)
            if view_def:
                return self._rewrite_select_from_view(stmt, view_def)

        # Check joins for view references
        has_view_join = False
        for j in stmt.joins:
            if self._views.exists(j.table.table_name):
                has_view_join = True
                break

        if has_view_join:
            return self._rewrite_join_views(stmt)

        return stmt

    def _rewrite_select_from_view(self, stmt: SelectStmt, view_def: ViewDef) -> SelectStmt:
        """Rewrite a SELECT that reads from a view.

        Strategy: Execute the view's SELECT first, store results in a temp concept,
        then wrap with the outer query's WHERE/ORDER BY/LIMIT.

        Simpler approach: merge the view's WHERE with the outer WHERE, and map columns.
        But this gets complex with aggregates, etc.

        Simplest correct approach: execute the view query, get results,
        then filter/project/sort the results according to the outer query.
        We do this by running the view query to get materialized rows,
        then running the outer query against those rows.
        """
        # Materialize view results (use _execute_view_stmt for nested view support)
        view_result = self._execute_view_stmt(view_def.select_stmt)

        # Get view column names
        view_cols = view_def.get_column_names()
        if view_cols == ['*'] or '*' in view_cols:
            view_cols = view_result.columns

        # Apply column aliases if specified
        if view_def.column_aliases:
            actual_cols = view_def.column_aliases
        else:
            actual_cols = view_cols

        # Ensure column count matches
        if len(actual_cols) < len(view_result.columns):
            actual_cols = actual_cols + view_result.columns[len(actual_cols):]
        elif len(actual_cols) > len(view_result.columns):
            actual_cols = actual_cols[:len(view_result.columns)]

        # Build materialized rows as dicts
        view_rows = []
        for row_vals in view_result.rows:
            row_dict = {}
            for i, col in enumerate(actual_cols):
                if i < len(row_vals):
                    row_dict[col] = row_vals[i]
                    # Also add table-qualified name for the view
                    view_name = view_def.name
                    alias = stmt.from_table.alias or view_name
                    row_dict[f"{alias}.{col}"] = row_vals[i]
            view_rows.append(row_dict)

        # Now apply outer query operations on materialized rows
        return self._apply_outer_query(stmt, view_rows, actual_cols, view_def.name)

    def _apply_outer_query(self, stmt: SelectStmt, rows: List[Dict],
                           view_cols: List[str], view_name: str) -> ResultSet:
        """Apply WHERE, projection, ORDER BY, LIMIT to materialized view rows."""
        alias = view_name
        if stmt.from_table and stmt.from_table.alias:
            alias = stmt.from_table.alias

        # WHERE filtering
        filtered = rows
        if stmt.where:
            filtered = []
            for row_dict in rows:
                qe_row = Row(row_dict)
                pred = self.compiler._sql_to_qe_expr(stmt.where)
                if eval_expr(pred, qe_row):
                    filtered.append(row_dict)

        # ORDER BY
        if stmt.order_by:
            filtered = self._sort_rows(filtered, stmt.order_by, alias)

        # Projection
        if self.compiler._is_star_only(stmt.columns):
            result_cols = list(view_cols)
            result_rows = []
            for row_dict in filtered:
                vals = [row_dict.get(c) for c in view_cols]
                result_rows.append(vals)
        else:
            result_cols = []
            result_rows = []
            for row_dict in filtered:
                vals = []
                for se in stmt.columns:
                    if se.alias:
                        col_name = se.alias
                    elif isinstance(se.expr, SqlColumnRef):
                        col_name = se.expr.column
                    else:
                        col_name = f"col_{len(result_cols)}"

                    if col_name not in result_cols:
                        result_cols.append(col_name)

                    qe_row = Row(row_dict)
                    qe_expr = self.compiler._sql_to_qe_expr(se.expr)
                    val = eval_expr(qe_expr, qe_row)
                    vals.append(val)
                result_rows.append(vals)

        # DISTINCT
        if stmt.distinct:
            seen = set()
            unique_rows = []
            for r in result_rows:
                key = tuple(r)
                if key not in seen:
                    seen.add(key)
                    unique_rows.append(r)
            result_rows = unique_rows

        # LIMIT / OFFSET
        if stmt.offset:
            result_rows = result_rows[stmt.offset:]
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_cols, rows=result_rows)

    def _sort_rows(self, rows: List[Dict], order_by: List, alias: str) -> List[Dict]:
        """Sort materialized rows by ORDER BY clause.
        order_by is List[Tuple[expr, asc_bool]] from the parser."""
        import functools

        def compare(a, b):
            for expr, asc in order_by:
                if isinstance(expr, SqlColumnRef):
                    col_name = expr.column
                elif isinstance(expr, str):
                    col_name = expr
                else:
                    # Try to evaluate expression
                    qe_expr = self.compiler._sql_to_qe_expr(expr)
                    va = eval_expr(qe_expr, Row(a))
                    vb = eval_expr(qe_expr, Row(b))
                    if va == vb:
                        continue
                    if va is None:
                        return 1 if asc else -1
                    if vb is None:
                        return -1 if asc else 1
                    if va < vb:
                        return -1 if asc else 1
                    if va > vb:
                        return 1 if asc else -1
                    continue

                va = a.get(col_name, a.get(f"{alias}.{col_name}"))
                vb = b.get(col_name, b.get(f"{alias}.{col_name}"))
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

        return sorted(rows, key=functools.cmp_to_key(compare))

    def _rewrite_join_views(self, stmt: SelectStmt) -> SelectStmt:
        """For joins involving views, materialize view results first.
        This is complex -- for now, raise an error for joins with views."""
        raise CatalogError(
            "JOIN with views is not yet supported. "
            "Use a subquery instead: SELECT * FROM (SELECT ...) AS v"
        )

    # -- INSERT into updatable view --

    def _exec_insert_into_view(self, stmt: InsertStmt) -> ResultSet:
        """INSERT into a view (if updatable)."""
        view_def = self._views.get(stmt.table_name)
        if view_def is None:
            return self._execute_stmt(stmt)

        if not view_def.is_updatable:
            raise CatalogError(
                f"View '{stmt.table_name}' is not updatable "
                f"(requires single table, no aggregates/DISTINCT/GROUP BY/JOIN)"
            )

        # Rewrite INSERT to target the base table
        base_table = view_def.base_table
        new_stmt = InsertStmt(
            table_name=base_table,
            columns=self._map_view_cols_to_base(stmt.columns, view_def) if stmt.columns else stmt.columns,
            values_list=stmt.values_list,
        )
        return self._execute_stmt(new_stmt)

    # -- UPDATE through updatable view --

    def _exec_update_view(self, stmt: UpdateStmt) -> ResultSet:
        """UPDATE through a view (if updatable)."""
        view_def = self._views.get(stmt.table_name)
        if view_def is None:
            return self._execute_stmt(stmt)

        if not view_def.is_updatable:
            raise CatalogError(
                f"View '{stmt.table_name}' is not updatable"
            )

        base_table = view_def.base_table

        # Map column names and merge WHERE clauses
        mapped_assignments = []
        for col, expr in stmt.assignments:
            mapped_col = self._map_view_col(col, view_def)
            mapped_assignments.append((mapped_col, expr))

        # Merge view's WHERE with UPDATE's WHERE
        merged_where = self._merge_where(view_def.select_stmt.where, stmt.where)

        new_stmt = UpdateStmt(
            table_name=base_table,
            assignments=mapped_assignments,
            where=merged_where,
        )
        return self._execute_stmt(new_stmt)

    # -- DELETE from updatable view --

    def _exec_delete_from_view(self, stmt: DeleteStmt) -> ResultSet:
        """DELETE from a view (if updatable)."""
        view_def = self._views.get(stmt.table_name)
        if view_def is None:
            return self._execute_stmt(stmt)

        if not view_def.is_updatable:
            raise CatalogError(
                f"View '{stmt.table_name}' is not updatable"
            )

        base_table = view_def.base_table
        merged_where = self._merge_where(view_def.select_stmt.where, stmt.where)

        new_stmt = DeleteStmt(
            table_name=base_table,
            where=merged_where,
        )
        return self._execute_stmt(new_stmt)

    # -- DROP TABLE with view dependency check --

    def _exec_drop_table_with_views(self, stmt: DropTableStmt) -> ResultSet:
        """DROP TABLE -- check if any views depend on it."""
        dependents = self._views.get_dependents(stmt.table_name)
        if dependents:
            raise CatalogError(
                f"Cannot drop table '{stmt.table_name}': "
                f"view(s) {', '.join(repr(v) for v in dependents)} depend on it"
            )
        return self._execute_stmt(stmt)

    # -- SHOW TABLES with views --

    def _exec_show_tables_with_views(self) -> ResultSet:
        """SHOW TABLES including views."""
        tables = self.storage.catalog.list_tables()
        views = self._views.list_views()
        rows = []
        for t in sorted(tables):
            rows.append([t, 'TABLE'])
        for v in sorted(views):
            rows.append([v, 'VIEW'])
        return ResultSet(columns=['name', 'type'], rows=rows)

    # -- DESCRIBE with views --

    def _exec_describe_with_views(self, stmt: DescribeStmt) -> ResultSet:
        """DESCRIBE a view -- show columns."""
        view_def = self._views.get(stmt.table_name)
        if view_def is None:
            return self._execute_stmt(stmt)  # regular table DESCRIBE

        # Get column names
        col_names = view_def.get_column_names()
        if '*' in col_names:
            # Expand * by running the view query
            result = self._execute_stmt(view_def.select_stmt)
            col_names = result.columns

        if view_def.column_aliases and len(view_def.column_aliases) == len(col_names):
            col_names = view_def.column_aliases

        rows = []
        for name in col_names:
            rows.append([name, 'ANY', '', 'YES'])
        return ResultSet(
            columns=['column', 'type', 'constraints', 'nullable'],
            rows=rows
        )

    # -- Column mapping helpers --

    def _map_view_cols_to_base(self, cols: List[str], view_def: ViewDef) -> List[str]:
        """Map view column names back to base table column names."""
        if not view_def.column_aliases:
            return cols  # no mapping needed
        # Build alias -> base name mapping
        base_names = []
        for se in view_def.select_stmt.columns:
            if isinstance(se.expr, SqlColumnRef):
                base_names.append(se.expr.column)
            elif se.alias:
                base_names.append(se.alias)
            else:
                base_names.append(None)

        result = []
        for col in cols:
            mapped = col
            for i, alias in enumerate(view_def.column_aliases):
                if alias == col and i < len(base_names) and base_names[i]:
                    mapped = base_names[i]
                    break
            result.append(mapped)
        return result

    def _map_view_col(self, col: str, view_def: ViewDef) -> str:
        """Map a single view column name to base table column name."""
        result = self._map_view_cols_to_base([col], view_def)
        return result[0]

    def _merge_where(self, view_where, outer_where):
        """Merge view's WHERE clause with outer WHERE clause using AND."""
        if view_where is None:
            return outer_where
        if outer_where is None:
            return view_where
        return SqlLogic(op='and', operands=[view_where, outer_where])

    # -- Introspection --

    def get_view(self, name: str) -> Optional[ViewDef]:
        """Get a view definition."""
        return self._views.get(name)

    def list_views(self) -> List[str]:
        """List all view names."""
        return self._views.list_views()

    def is_view(self, name: str) -> bool:
        """Check if a name is a view."""
        return self._views.exists(name)
