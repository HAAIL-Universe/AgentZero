"""
C253: Common Table Expressions (CTEs)
Extends C252 (SQL Window Functions) / C251 (Triggers) / C250 (Views) / C249 (Stored Procedures) / C247 (Mini Database)

Adds WITH ... AS common table expressions to the database engine:
- Non-recursive CTEs: WITH name AS (SELECT ...) SELECT ...
- Multiple CTEs: WITH a AS (...), b AS (...) SELECT ...
- Recursive CTEs: WITH RECURSIVE name AS (base UNION ALL recursive) SELECT ...
- CTEs referencing earlier CTEs in the WITH list
- CTEs with column aliases: WITH name(col1, col2) AS (...)
- CTEs in INSERT, UPDATE, DELETE contexts
- Nested CTEs (CTE query body can itself contain CTEs)
- Recursive CTE cycle detection and depth limiting
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import C252 (which imports C251 -> C250 -> C249 -> C247)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C252_sql_window_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C251_sql_triggers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C250_sql_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from sql_window_functions import (
    WindowDB, WindowLexer, WindowParser, WindowQueryCompiler,
    WindowExecutor, SqlWindowCall, WindowSpec,
)

from sql_triggers import (
    TriggerDB, TriggerLexer, TriggerParser,
)

from sql_views import (
    ViewDB, ViewLexer, ViewParser, ViewCatalog,
    CreateViewStmt, DropViewStmt,
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
# CTE AST Nodes
# =============================================================================

@dataclass
class CTEDef:
    """A single CTE definition: name [(col1, col2, ...)] AS (SELECT ...)"""
    name: str
    body: SelectStmt  # the CTE query body
    column_aliases: Optional[List[str]] = None  # explicit column names
    recursive: bool = False


@dataclass
class CTESelectStmt:
    """WITH [RECURSIVE] name AS (...) [, ...] SELECT ..."""
    ctes: List[CTEDef]
    main_stmt: Any  # SelectStmt, InsertStmt, UpdateStmt, DeleteStmt
    recursive: bool = False


@dataclass
class UnionStmt:
    """base UNION [ALL] recursive -- used inside recursive CTEs"""
    left: SelectStmt
    right: SelectStmt
    union_all: bool = True  # UNION ALL (not deduped) vs UNION (deduped)


# =============================================================================
# CTE Lexer
# =============================================================================

class CTELexer(WindowLexer):
    """Lexer extended for CTEs. 'WITH' and 'RECURSIVE' stay as IDENT tokens
    and are recognized contextually by the parser."""

    def __init__(self, sql: str):
        super().__init__(sql)


# =============================================================================
# CTE Parser (extends WindowParser)
# =============================================================================

# Max recursion depth for recursive CTEs
MAX_RECURSIVE_DEPTH = 1000


class CTEParser(WindowParser):
    """Parser extended with CTE support.

    Recognizes:
      WITH [RECURSIVE] name [(col1, col2)] AS (SELECT ...)
        [, name2 AS (...)]
      SELECT ... / INSERT ... / UPDATE ... / DELETE ...
    """

    def __init__(self, tokens):
        super().__init__(tokens)
        self._in_cte_body = False

    def _parse_statement(self):
        """Override to detect WITH keyword at statement start."""
        word = self._peek_word()
        if word == 'with' and not self._in_cte_body:
            return self._parse_cte_statement()
        return super()._parse_statement()

    def _parse_cte_statement(self) -> CTESelectStmt:
        """Parse WITH [RECURSIVE] ... AS (...) [, ...] main_stmt"""
        self._advance()  # consume WITH

        recursive = False
        if self._peek_word() == 'recursive':
            recursive = True
            self._advance()  # consume RECURSIVE

        ctes = []
        while True:
            cte = self._parse_single_cte(recursive)
            ctes.append(cte)
            if not self.match(TokenType.COMMA):
                break

        # Parse the main statement (SELECT, INSERT, UPDATE, DELETE)
        main_stmt = super()._parse_statement()

        return CTESelectStmt(ctes=ctes, main_stmt=main_stmt, recursive=recursive)

    def _parse_single_cte(self, recursive: bool) -> CTEDef:
        """Parse: name [(col1, col2, ...)] AS (SELECT ...)"""
        name = self._expect_ident()

        # Optional column aliases
        column_aliases = None
        if self._peek_type() == TokenType.LPAREN:
            # Check if this is column aliases or the AS keyword ahead
            # Column aliases: name(col1, col2) AS (...)
            # We peek ahead to see if after the parens there's AS
            saved_pos = self.pos
            self._advance()  # consume (
            # Try to parse identifiers
            aliases = []
            try:
                aliases.append(self._expect_ident())
                while self.match(TokenType.COMMA):
                    aliases.append(self._expect_ident())
                if self._peek_type() == TokenType.RPAREN:
                    self._advance()  # consume )
                    column_aliases = aliases
                else:
                    # Not column aliases, restore position
                    self.pos = saved_pos
            except (ParseError, IndexError):
                self.pos = saved_pos

        # Expect AS
        if self._peek_word() != 'as':
            raise ParseError(f"Expected AS after CTE name '{name}'")
        self._advance()  # consume AS

        # Expect ( SELECT ... )
        if self._peek_type() != TokenType.LPAREN:
            raise ParseError("Expected '(' after AS in CTE definition")
        self._advance()  # consume (

        # Parse the CTE body - could be SELECT or SELECT UNION [ALL] SELECT
        body = self._parse_cte_body()

        if self._peek_type() != TokenType.RPAREN:
            raise ParseError("Expected ')' after CTE body")
        self._advance()  # consume )

        return CTEDef(
            name=name,
            body=body,
            column_aliases=column_aliases,
            recursive=recursive,
        )

    def _parse_cte_body(self) -> Any:
        """Parse CTE body, handling UNION [ALL] for recursive CTEs."""
        left = self._parse_inner_select()

        # Check for UNION
        if self._peek_word() == 'union':
            self._advance()  # consume UNION
            union_all = False
            if self._peek_word() == 'all':
                union_all = True
                self._advance()  # consume ALL
            right = self._parse_inner_select()
            return UnionStmt(left=left, right=right, union_all=union_all)

        return left

    def _parse_inner_select(self) -> SelectStmt:
        """Parse a SELECT inside CTE body, avoiding re-entering CTE detection."""
        word = self._peek_word()
        if word != 'select':
            raise ParseError(f"Expected SELECT in CTE body, got '{word}'")
        # Set flag to prevent _parse_statement from detecting WITH as CTE
        old_flag = self._in_cte_body
        self._in_cte_body = True
        try:
            return super()._parse_statement()
        finally:
            self._in_cte_body = old_flag


# =============================================================================
# CTE DB (extends WindowDB)
# =============================================================================

class CTEDB(WindowDB):
    """WindowDB extended with Common Table Expression support."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        # Temporary CTE data available during query execution
        self._cte_tables: Dict[str, List[Dict[str, Any]]] = {}
        self._cte_columns: Dict[str, List[str]] = {}

    def execute(self, sql: str) -> ResultSet:
        stmts = self._parse_cte(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_cte_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List[ResultSet]:
        stmts = self._parse_cte(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_cte_stmt(stmt))
        return results

    def _parse_cte(self, sql: str) -> List[Any]:
        lexer = CTELexer(sql)
        parser = CTEParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _execute_cte_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling CTEs."""
        if isinstance(stmt, CTESelectStmt):
            return self._exec_with_ctes(stmt)

        # Delegate non-CTE statements through the window/trigger/view chain
        if isinstance(stmt, SelectStmt):
            return self._exec_select_with_windows(stmt)

        return self._execute_trigger_stmt(stmt)

    # =========================================================================
    # CTE Execution
    # =========================================================================

    def _exec_with_ctes(self, cte_stmt: CTESelectStmt) -> ResultSet:
        """Execute a WITH ... statement by materializing CTEs then running main query."""
        # Save any existing CTE context (for nested CTEs)
        saved_tables = dict(self._cte_tables)
        saved_columns = dict(self._cte_columns)

        try:
            # Materialize each CTE in order
            for cte_def in cte_stmt.ctes:
                self._materialize_cte(cte_def)

            # Execute the main statement with CTE tables available
            main = cte_stmt.main_stmt
            if isinstance(main, SelectStmt):
                return self._exec_select_with_cte_context(main)
            elif isinstance(main, InsertStmt):
                return self._exec_insert_with_cte_context(main)
            elif isinstance(main, UpdateStmt):
                return self._exec_update_with_cte_context(main)
            elif isinstance(main, DeleteStmt):
                return self._exec_delete_with_cte_context(main)
            else:
                return self._execute_cte_stmt(main)
        finally:
            # Restore previous CTE context
            self._cte_tables = saved_tables
            self._cte_columns = saved_columns

    def _materialize_cte(self, cte_def: CTEDef):
        """Materialize a single CTE into _cte_tables."""
        name = cte_def.name.lower()

        if isinstance(cte_def.body, UnionStmt):
            # Recursive CTE: base UNION [ALL] recursive
            self._materialize_recursive_cte(cte_def, name)
        else:
            # Non-recursive CTE: just execute the SELECT
            result = self._exec_cte_select(cte_def.body)
            columns = list(result.columns)
            if cte_def.column_aliases:
                if len(cte_def.column_aliases) != len(columns):
                    raise DatabaseError(
                        f"CTE '{name}' has {len(cte_def.column_aliases)} column aliases "
                        f"but query returns {len(columns)} columns"
                    )
                columns = list(cte_def.column_aliases)

            # Store as list of dicts
            rows = []
            for row_vals in result.rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row_vals[i] if i < len(row_vals) else None
                rows.append(row_dict)

            self._cte_tables[name] = rows
            self._cte_columns[name] = columns

    def _materialize_recursive_cte(self, cte_def: CTEDef, name: str):
        """Materialize a recursive CTE using iterative fixpoint."""
        union = cte_def.body
        assert isinstance(union, UnionStmt)

        # Step 1: Execute base case
        base_result = self._exec_cte_select(union.left)
        columns = list(base_result.columns)
        if cte_def.column_aliases:
            if len(cte_def.column_aliases) != len(columns):
                raise DatabaseError(
                    f"CTE '{name}' has {len(cte_def.column_aliases)} column aliases "
                    f"but base query returns {len(columns)} columns"
                )
            columns = list(cte_def.column_aliases)

        # Convert to dicts
        all_rows = []
        working_rows = []
        for row_vals in base_result.rows:
            row_dict = {}
            for i, col in enumerate(columns):
                row_dict[col] = row_vals[i] if i < len(row_vals) else None
            all_rows.append(row_dict)
            working_rows.append(row_dict)

        # Step 2: Iterate recursive part until fixpoint or depth limit
        seen = set()
        if not union.union_all:
            for r in all_rows:
                seen.add(tuple(r.get(c) for c in columns))

        depth = 0
        while working_rows and depth < MAX_RECURSIVE_DEPTH:
            depth += 1

            # Set the CTE table to current working set (for self-reference)
            self._cte_tables[name] = working_rows
            self._cte_columns[name] = columns

            # Execute recursive part
            rec_result = self._exec_cte_select(union.right)

            new_rows = []
            for row_vals in rec_result.rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row_vals[i] if i < len(row_vals) else None

                if not union.union_all:
                    key = tuple(row_dict.get(c) for c in columns)
                    if key in seen:
                        continue
                    seen.add(key)

                new_rows.append(row_dict)

            if not new_rows:
                break  # fixpoint reached

            all_rows.extend(new_rows)
            working_rows = new_rows

        if depth >= MAX_RECURSIVE_DEPTH:
            raise DatabaseError(
                f"Recursive CTE '{name}' exceeded maximum depth of {MAX_RECURSIVE_DEPTH}"
            )

        # Store final result
        self._cte_tables[name] = all_rows
        self._cte_columns[name] = columns

    def _exec_cte_select(self, stmt: SelectStmt) -> ResultSet:
        """Execute a SELECT that may reference CTE tables."""
        return self._exec_select_with_cte_context(stmt)

    def _exec_select_with_cte_context(self, stmt: SelectStmt) -> ResultSet:
        """Execute a SELECT, substituting CTE tables where referenced."""
        # Check if the FROM clause references a CTE
        if stmt.from_table and stmt.from_table.table_name.lower() in self._cte_tables:
            return self._exec_select_from_cte(stmt)

        # Check JOINs for CTE references
        has_cte_join = any(
            j.table.table_name.lower() in self._cte_tables
            for j in stmt.joins
        )
        if has_cte_join:
            return self._exec_select_with_cte_joins(stmt)

        # No CTE references -- standard execution
        return self._exec_select_with_windows(stmt)

    def _exec_select_from_cte(self, stmt: SelectStmt) -> ResultSet:
        """Execute a SELECT FROM cte_name."""
        cte_name = stmt.from_table.table_name.lower()
        cte_alias = stmt.from_table.alias or cte_name
        cte_rows = self._cte_tables[cte_name]
        cte_cols = self._cte_columns[cte_name]

        # Handle JOINs with other tables or CTEs
        if stmt.joins:
            return self._exec_cte_with_joins(stmt, cte_rows, cte_cols, cte_alias)

        # Filter with WHERE
        filtered = cte_rows
        if stmt.where:
            filtered = [r for r in filtered if self._eval_cte_expr(stmt.where, r, cte_alias)]

        # Handle GROUP BY
        if stmt.group_by:
            return self._exec_cte_group_by(stmt, filtered, cte_cols, cte_alias)

        # Handle aggregates without GROUP BY
        has_agg = any(
            isinstance(se.expr, SqlAggCall) or
            (isinstance(se.expr, SqlWindowCall))
            for se in stmt.columns
        )
        if has_agg and not stmt.group_by:
            return self._exec_cte_aggregate(stmt, filtered, cte_alias)

        # Project columns
        result_columns, result_rows = self._project_cte_rows(
            stmt.columns, filtered, cte_cols, cte_alias
        )

        # ORDER BY
        if stmt.order_by:
            result_rows_dicts = []
            for row_vals in result_rows:
                d = {}
                for i, c in enumerate(result_columns):
                    d[c] = row_vals[i]
                result_rows_dicts.append(d)
            result_rows_dicts = self._sort_cte_rows(
                result_rows_dicts, stmt.order_by, cte_alias
            )
            result_rows = [[d.get(c) for c in result_columns] for d in result_rows_dicts]

        # DISTINCT
        if stmt.distinct:
            seen = set()
            unique = []
            for r in result_rows:
                key = tuple(r)
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            result_rows = unique

        # OFFSET / LIMIT
        if stmt.offset:
            result_rows = result_rows[stmt.offset:]
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_columns, rows=result_rows)

    def _exec_cte_with_joins(self, stmt, cte_rows, cte_cols, cte_alias):
        """Execute a SELECT from CTE with JOINs."""
        # Start with CTE rows
        combined_rows = []
        for r in cte_rows:
            row_with_prefix = {}
            for k, v in r.items():
                row_with_prefix[k] = v
                row_with_prefix[f"{cte_alias}.{k}"] = v
            combined_rows.append(row_with_prefix)

        # Process each JOIN
        for join in stmt.joins:
            join_table = join.table.table_name.lower()
            join_alias = join.table.alias or join_table

            if join_table in self._cte_tables:
                # Joining with another CTE
                join_rows = self._cte_tables[join_table]
                join_cols = self._cte_columns[join_table]
            else:
                # Joining with a real table
                txn_id = self._get_txn()
                try:
                    raw_rows = self.storage.scan_table(txn_id, join_table)
                    schema = self.storage.catalog.get_table(join_table)
                    join_cols = schema.column_names()
                    join_rows = []
                    for raw in raw_rows:
                        row_data = raw[1] if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict) else raw
                        if isinstance(row_data, dict):
                            join_rows.append(row_data)
                        else:
                            d = {}
                            for i, col in enumerate(join_cols):
                                d[col] = row_data[i] if i < len(row_data) else None
                            join_rows.append(d)
                    self._auto_commit(txn_id)
                except Exception:
                    self._auto_abort(txn_id)
                    raise

            new_combined = []
            for left_row in combined_rows:
                for right_row in join_rows:
                    merged = dict(left_row)
                    for k, v in right_row.items():
                        # Use aliased keys to avoid overwrites
                        merged[f"{join_alias}.{k}"] = v
                        if k not in merged:
                            merged[k] = v
                    # Check join condition
                    if join.condition is None or self._eval_cte_expr(join.condition, merged, None):
                        new_combined.append(merged)

            combined_rows = new_combined

        # WHERE
        if stmt.where:
            combined_rows = [r for r in combined_rows if self._eval_cte_expr(stmt.where, r, None)]

        # GROUP BY
        if stmt.group_by:
            return self._exec_cte_group_by(stmt, combined_rows, None, None)

        # Project
        result_columns, result_rows = self._project_cte_rows(
            stmt.columns, combined_rows, None, None
        )

        # ORDER BY
        if stmt.order_by:
            dicts = []
            for rv in result_rows:
                d = {}
                for i, c in enumerate(result_columns):
                    d[c] = rv[i]
                dicts.append(d)
            dicts = self._sort_cte_rows(dicts, stmt.order_by, None)
            result_rows = [[d.get(c) for c in result_columns] for d in dicts]

        # DISTINCT
        if stmt.distinct:
            seen = set()
            unique = []
            for r in result_rows:
                key = tuple(r)
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            result_rows = unique

        if stmt.offset:
            result_rows = result_rows[stmt.offset:]
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_columns, rows=result_rows)

    def _exec_select_with_cte_joins(self, stmt: SelectStmt) -> ResultSet:
        """Execute SELECT from real table with CTE in JOINs."""
        # Execute the main FROM table normally, then join with CTE data
        # For simplicity, we materialize the main table first, then join
        txn_id = self._get_txn()
        try:
            main_table = stmt.from_table.table_name
            main_alias = stmt.from_table.alias or main_table
            schema = self.storage.catalog.get_table(main_table)
            main_cols = schema.column_names()
            raw_rows = self.storage.scan_table(txn_id, main_table)

            combined_rows = []
            for raw in raw_rows:
                row_data = raw[1] if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict) else raw
                d = {}
                if isinstance(row_data, dict):
                    for col in main_cols:
                        d[col] = row_data.get(col)
                        d[f"{main_alias}.{col}"] = row_data.get(col)
                else:
                    for i, col in enumerate(main_cols):
                        d[col] = row_data[i] if i < len(row_data) else None
                        d[f"{main_alias}.{col}"] = row_data[i] if i < len(row_data) else None
                combined_rows.append(d)

            # Process JOINs
            for join in stmt.joins:
                join_table = join.table.table_name.lower()
                join_alias = join.table.alias or join_table

                if join_table in self._cte_tables:
                    join_rows = self._cte_tables[join_table]
                    join_cols = self._cte_columns[join_table]
                else:
                    j_raw = self.storage.scan_table(txn_id, join_table)
                    j_schema = self.storage.catalog.get_table(join_table)
                    join_cols = j_schema.column_names()
                    join_rows = []
                    for raw in j_raw:
                        row_data = raw[1] if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict) else raw
                        if isinstance(row_data, dict):
                            join_rows.append(row_data)
                        else:
                            d = {}
                            for i, col in enumerate(join_cols):
                                d[col] = row_data[i] if i < len(row_data) else None
                            join_rows.append(d)

                new_combined = []
                for left_row in combined_rows:
                    for right_row in join_rows:
                        merged = dict(left_row)
                        for k, v in right_row.items():
                            merged[f"{join_alias}.{k}"] = v
                            if k not in merged:
                                merged[k] = v
                        if join.condition is None or self._eval_cte_expr(join.condition, merged, None):
                            new_combined.append(merged)
                combined_rows = new_combined

            # WHERE
            if stmt.where:
                combined_rows = [r for r in combined_rows if self._eval_cte_expr(stmt.where, r, None)]

            # GROUP BY
            if stmt.group_by:
                self._auto_commit(txn_id)
                return self._exec_cte_group_by(stmt, combined_rows, None, None)

            # Project
            result_columns, result_rows = self._project_cte_rows(
                stmt.columns, combined_rows, None, None
            )

            # ORDER BY
            if stmt.order_by:
                dicts = []
                for rv in result_rows:
                    d = {}
                    for i, c in enumerate(result_columns):
                        d[c] = rv[i]
                    dicts.append(d)
                dicts = self._sort_cte_rows(dicts, stmt.order_by, None)
                result_rows = [[d.get(c) for c in result_columns] for d in dicts]

            if stmt.distinct:
                seen = set()
                unique = []
                for r in result_rows:
                    key = tuple(r)
                    if key not in seen:
                        seen.add(key)
                        unique.append(r)
                result_rows = unique

            if stmt.offset:
                result_rows = result_rows[stmt.offset:]
            if stmt.limit is not None:
                result_rows = result_rows[:stmt.limit]

            self._auto_commit(txn_id)
            return ResultSet(columns=result_columns, rows=result_rows)
        except Exception:
            self._auto_abort(txn_id)
            raise

    # =========================================================================
    # CTE Expression Evaluation
    # =========================================================================

    def _eval_cte_expr(self, expr, row: Dict, default_table: Optional[str]) -> Any:
        """Evaluate a SQL expression against a CTE row dict."""
        if isinstance(expr, SqlLiteral):
            return expr.value

        if isinstance(expr, SqlColumnRef):
            col = expr.column
            tbl = expr.table
            # Try table.col
            if tbl:
                key = f"{tbl}.{col}"
                if key in row:
                    return row[key]
            # Try plain col
            if col in row:
                return row[col]
            # Try with default table
            if default_table:
                key = f"{default_table}.{col}"
                if key in row:
                    return row[key]
            # Try any prefix
            for k, v in row.items():
                if k.split('.')[-1] == col:
                    return v
            return None

        if isinstance(expr, SqlComparison):
            left = self._eval_cte_expr(expr.left, row, default_table)
            right = self._eval_cte_expr(expr.right, row, default_table)
            op = expr.op
            if left is None or right is None:
                if op == '=':
                    return left is None and right is None
                if op in ('!=', '<>'):
                    return not (left is None and right is None)
                return False
            if op == '=':
                return left == right
            if op in ('!=', '<>'):
                return left != right
            if op == '<':
                return left < right
            if op == '<=':
                return left <= right
            if op == '>':
                return left > right
            if op == '>=':
                return left >= right
            return False

        if isinstance(expr, SqlLogic):
            op = expr.op.upper() if isinstance(expr.op, str) else expr.op
            if op == 'AND':
                return all(self._eval_cte_expr(o, row, default_table) for o in expr.operands)
            if op == 'OR':
                return any(self._eval_cte_expr(o, row, default_table) for o in expr.operands)
            if op == 'NOT':
                return not self._eval_cte_expr(expr.operands[0], row, default_table)
            return False

        if isinstance(expr, SqlBinOp):
            left = self._eval_cte_expr(expr.left, row, default_table)
            right = self._eval_cte_expr(expr.right, row, default_table)
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
                if right == 0:
                    return None
                return left / right
            if op == '%':
                return left % right
            return None

        if isinstance(expr, SqlIsNull):
            val = self._eval_cte_expr(expr.expr, row, default_table)
            result = val is None
            return not result if expr.negated else result

        if isinstance(expr, SqlFuncCall):
            args = [self._eval_cte_expr(a, row, default_table) for a in expr.args]
            return self._eval_cte_func(expr.func_name, args)

        if isinstance(expr, SqlBetween):
            val = self._eval_cte_expr(expr.expr, row, default_table)
            low = self._eval_cte_expr(expr.low, row, default_table)
            high = self._eval_cte_expr(expr.high, row, default_table)
            if val is None or low is None or high is None:
                return False
            result = low <= val <= high
            return not result if expr.negated else result

        if isinstance(expr, SqlInList):
            val = self._eval_cte_expr(expr.expr, row, default_table)
            values = [self._eval_cte_expr(v, row, default_table) for v in expr.values]
            result = val in values
            return not result if expr.negated else result

        if isinstance(expr, SqlCase):
            if expr.operand:
                op_val = self._eval_cte_expr(expr.operand, row, default_table)
                for when_expr, then_expr in expr.whens:
                    when_val = self._eval_cte_expr(when_expr, row, default_table)
                    if op_val == when_val:
                        return self._eval_cte_expr(then_expr, row, default_table)
            else:
                for when_expr, then_expr in expr.whens:
                    if self._eval_cte_expr(when_expr, row, default_table):
                        return self._eval_cte_expr(then_expr, row, default_table)
            if expr.else_expr:
                return self._eval_cte_expr(expr.else_expr, row, default_table)
            return None

        if isinstance(expr, SqlStar):
            return None

        # Unknown expression type
        return None

    def _eval_cte_func(self, name: str, args: List) -> Any:
        """Evaluate a scalar function in CTE context."""
        name = name.lower()
        if name == 'abs':
            return abs(args[0]) if args[0] is not None else None
        if name == 'upper':
            return str(args[0]).upper() if args[0] is not None else None
        if name == 'lower':
            return str(args[0]).lower() if args[0] is not None else None
        if name == 'length' or name == 'len':
            return len(str(args[0])) if args[0] is not None else None
        if name == 'coalesce':
            for a in args:
                if a is not None:
                    return a
            return None
        if name == 'nullif':
            if len(args) >= 2 and args[0] == args[1]:
                return None
            return args[0] if args else None
        if name == 'ifnull' or name == 'isnull':
            return args[0] if args[0] is not None else (args[1] if len(args) > 1 else None)
        if name in ('cast', 'typeof'):
            return args[0] if args else None
        if name == 'concat':
            return ''.join(str(a) for a in args if a is not None)
        if name == 'substr' or name == 'substring':
            if len(args) >= 2:
                s = str(args[0]) if args[0] is not None else ''
                start = int(args[1]) - 1  # SQL is 1-based
                length = int(args[2]) if len(args) > 2 else len(s)
                return s[start:start + length]
            return None
        if name == 'replace':
            if len(args) >= 3 and args[0] is not None:
                return str(args[0]).replace(str(args[1]), str(args[2]))
            return None
        if name == 'trim':
            return str(args[0]).strip() if args and args[0] is not None else None
        if name == 'round':
            if args and args[0] is not None:
                decimals = int(args[1]) if len(args) > 1 else 0
                return round(args[0], decimals)
            return None
        if name == 'max':
            vals = [a for a in args if a is not None]
            return max(vals) if vals else None
        if name == 'min':
            vals = [a for a in args if a is not None]
            return min(vals) if vals else None
        return None

    # =========================================================================
    # CTE Projection
    # =========================================================================

    def _project_cte_rows(self, columns: List[SelectExpr], rows: List[Dict],
                           cte_cols: Optional[List[str]], default_table: Optional[str]
                           ) -> Tuple[List[str], List[List]]:
        """Project CTE rows to match SELECT columns."""
        result_columns = []
        for se in columns:
            if isinstance(se.expr, SqlStar):
                # Expand * -- use keys from first row or cte_cols
                if cte_cols:
                    result_columns.extend(cte_cols)
                elif rows:
                    # Use keys without table prefix, deduplicated
                    seen = set()
                    for k in rows[0].keys():
                        short = k.split('.')[-1] if '.' in k else k
                        if short not in seen:
                            seen.add(short)
                            result_columns.append(short)
            elif se.alias:
                result_columns.append(se.alias)
            elif isinstance(se.expr, SqlColumnRef):
                result_columns.append(se.expr.column)
            elif isinstance(se.expr, SqlAggCall):
                result_columns.append(f"{se.expr.func}_{len(result_columns)}")
            else:
                result_columns.append(f"col_{len(result_columns)}")

        result_rows = []
        for row in rows:
            vals = []
            col_idx = 0
            for se in columns:
                if isinstance(se.expr, SqlStar):
                    if cte_cols:
                        for c in cte_cols:
                            vals.append(row.get(c))
                            col_idx += 1
                    elif rows:
                        seen = set()
                        for k in rows[0].keys():
                            short = k.split('.')[-1] if '.' in k else k
                            if short not in seen:
                                seen.add(short)
                                vals.append(row.get(k, row.get(short)))
                                col_idx += 1
                else:
                    val = self._eval_cte_expr(se.expr, row, default_table)
                    vals.append(val)
                    col_idx += 1
            result_rows.append(vals)

        return result_columns, result_rows

    # =========================================================================
    # CTE Aggregation
    # =========================================================================

    def _exec_cte_group_by(self, stmt: SelectStmt, rows: List[Dict],
                            cte_cols: Optional[List[str]], default_table: Optional[str]
                            ) -> ResultSet:
        """Execute GROUP BY on CTE rows."""
        # Group rows by GROUP BY expressions
        groups: Dict[tuple, List[Dict]] = {}
        for row in rows:
            key_parts = []
            for gb_expr in stmt.group_by:
                val = self._eval_cte_expr(gb_expr, row, default_table)
                key_parts.append(val)
            key = tuple(key_parts)
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        # Evaluate each group
        result_columns = []
        result_rows = []

        # Build column names from SELECT list
        for se in stmt.columns:
            if se.alias:
                result_columns.append(se.alias)
            elif isinstance(se.expr, SqlColumnRef):
                result_columns.append(se.expr.column)
            elif isinstance(se.expr, SqlAggCall):
                result_columns.append(f"{se.expr.func}_{len(result_columns)}")
            else:
                result_columns.append(f"col_{len(result_columns)}")

        for key, group_rows in groups.items():
            row_vals = []
            for se in stmt.columns:
                if isinstance(se.expr, SqlAggCall):
                    val = self._eval_cte_agg(se.expr, group_rows, default_table)
                    row_vals.append(val)
                else:
                    # Use the first row's value for non-aggregate columns
                    val = self._eval_cte_expr(se.expr, group_rows[0], default_table)
                    row_vals.append(val)
            result_rows.append(row_vals)

        # HAVING
        if stmt.having:
            filtered = []
            for i, (key, group_rows) in enumerate(groups.items()):
                # Build a dict with both column values and aggregate results
                having_row = dict(group_rows[0])
                for j, se in enumerate(stmt.columns):
                    if se.alias:
                        having_row[se.alias] = result_rows[i][j]
                if self._eval_cte_having(stmt.having, having_row, group_rows, default_table):
                    filtered.append(result_rows[i])
            result_rows = filtered

        # ORDER BY
        if stmt.order_by:
            dicts = []
            for rv in result_rows:
                d = {}
                for i, c in enumerate(result_columns):
                    d[c] = rv[i]
                dicts.append(d)
            dicts = self._sort_cte_rows(dicts, stmt.order_by, default_table)
            result_rows = [[d.get(c) for c in result_columns] for d in dicts]

        if stmt.offset:
            result_rows = result_rows[stmt.offset:]
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_columns, rows=result_rows)

    def _exec_cte_aggregate(self, stmt: SelectStmt, rows: List[Dict],
                             default_table: Optional[str]) -> ResultSet:
        """Execute aggregate functions without GROUP BY (whole-table aggregate)."""
        result_columns = []
        row_vals = []
        for se in stmt.columns:
            if se.alias:
                result_columns.append(se.alias)
            elif isinstance(se.expr, SqlAggCall):
                result_columns.append(f"{se.expr.func}_{len(result_columns)}")
            elif isinstance(se.expr, SqlColumnRef):
                result_columns.append(se.expr.column)
            else:
                result_columns.append(f"col_{len(result_columns)}")

            if isinstance(se.expr, SqlAggCall):
                val = self._eval_cte_agg(se.expr, rows, default_table)
                row_vals.append(val)
            else:
                val = self._eval_cte_expr(se.expr, rows[0] if rows else {}, default_table)
                row_vals.append(val)

        return ResultSet(columns=result_columns, rows=[row_vals] if rows or any(
            isinstance(se.expr, SqlAggCall) for se in stmt.columns
        ) else [])

    def _eval_cte_agg(self, agg: SqlAggCall, rows: List[Dict],
                       default_table: Optional[str]) -> Any:
        """Evaluate an aggregate function on a group of CTE rows."""
        func = agg.func.upper()

        if func == 'COUNT':
            if agg.arg is None or isinstance(agg.arg, SqlStar):
                return len(rows)
            if agg.distinct:
                vals = set()
                for r in rows:
                    v = self._eval_cte_expr(agg.arg, r, default_table)
                    if v is not None:
                        vals.add(v)
                return len(vals)
            count = 0
            for r in rows:
                v = self._eval_cte_expr(agg.arg, r, default_table)
                if v is not None:
                    count += 1
            return count

        values = []
        for r in rows:
            v = self._eval_cte_expr(agg.arg, r, default_table)
            if v is not None:
                values.append(v)

        if agg.distinct:
            values = list(set(values))

        if not values:
            return 0 if func == 'COUNT' else None

        if func == 'SUM':
            return sum(values)
        if func == 'AVG':
            return sum(values) / len(values)
        if func == 'MIN':
            return min(values)
        if func == 'MAX':
            return max(values)

        return None

    def _eval_cte_having(self, expr, row: Dict, group_rows: List[Dict],
                          default_table: Optional[str]) -> bool:
        """Evaluate a HAVING expression, handling aggregate references."""
        if isinstance(expr, SqlAggCall):
            return self._eval_cte_agg(expr, group_rows, default_table)
        if isinstance(expr, SqlComparison):
            left = self._eval_cte_having(expr.left, row, group_rows, default_table)
            right = self._eval_cte_having(expr.right, row, group_rows, default_table)
            op = expr.op
            if left is None or right is None:
                return False
            if op == '=':
                return left == right
            if op in ('!=', '<>'):
                return left != right
            if op == '<':
                return left < right
            if op == '<=':
                return left <= right
            if op == '>':
                return left > right
            if op == '>=':
                return left >= right
        if isinstance(expr, SqlLogic):
            op = expr.op.upper() if isinstance(expr.op, str) else expr.op
            if op == 'AND':
                return all(self._eval_cte_having(o, row, group_rows, default_table) for o in expr.operands)
            if op == 'OR':
                return any(self._eval_cte_having(o, row, group_rows, default_table) for o in expr.operands)
        if isinstance(expr, SqlColumnRef):
            return self._eval_cte_expr(expr, row, default_table)
        if isinstance(expr, SqlLiteral):
            return expr.value
        return self._eval_cte_expr(expr, row, default_table)

    # =========================================================================
    # CTE Sorting
    # =========================================================================

    def _sort_cte_rows(self, rows: List[Dict], order_by: List[Tuple],
                        default_table: Optional[str]) -> List[Dict]:
        """Sort CTE row dicts by ORDER BY expressions."""
        import functools

        def compare(a, b):
            for expr, asc in order_by:
                a_val = self._eval_cte_expr(expr, a, default_table)
                b_val = self._eval_cte_expr(expr, b, default_table)
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

    # =========================================================================
    # CTE in INSERT/UPDATE/DELETE
    # =========================================================================

    def _exec_insert_with_cte_context(self, stmt: InsertStmt) -> ResultSet:
        """Execute INSERT that may reference CTEs (INSERT INTO ... SELECT FROM cte)."""
        # If values contain a SelectStmt referencing a CTE, we handle it
        # Otherwise delegate to parent
        if hasattr(stmt, 'select') and stmt.select:
            # INSERT INTO ... SELECT ...
            result = self._exec_select_with_cte_context(stmt.select)
            # Insert the result rows
            txn_id = self._get_txn()
            try:
                for row_vals in result.rows:
                    self.storage.insert_row(txn_id, stmt.table_name, row_vals)
                self._auto_commit(txn_id)
                return ResultSet(columns=[], rows=[],
                                 message=f"Inserted {len(result.rows)} rows")
            except Exception:
                self._auto_abort(txn_id)
                raise
        return self._execute_trigger_stmt(stmt)

    def _exec_update_with_cte_context(self, stmt: UpdateStmt) -> ResultSet:
        """Execute UPDATE that uses CTEs in WHERE clause."""
        return self._execute_trigger_stmt(stmt)

    def _exec_delete_with_cte_context(self, stmt: DeleteStmt) -> ResultSet:
        """Execute DELETE that uses CTEs in WHERE clause."""
        return self._execute_trigger_stmt(stmt)
