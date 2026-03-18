"""
C267: Common Table Expressions (CTEs)
Extends C266 (Subqueries) with WITH clause support.

CTE types:
- Basic CTEs: WITH cte AS (SELECT ...) SELECT ... FROM cte
- Multiple CTEs: WITH a AS (...), b AS (...) SELECT ...
- CTEs with column lists: WITH cte(x, y) AS (SELECT a, b ...) SELECT ...
- CTEs referencing earlier CTEs (chained)
- Recursive CTEs: WITH RECURSIVE cte AS (base UNION ALL recursive) SELECT ...
- Recursive CTEs with depth limits (safety)
- CTEs used in INSERT, UPDATE, DELETE
- CTEs with subqueries, aggregation, joins
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C266_subqueries')))
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

from subqueries import (
    SubqueryDB, SubqueryParser, parse_subquery_sql, parse_subquery_sql_multi,
    subquery_eval_expr, SubqueryTableRef,
    SqlSubquery, SqlExists, SqlInSubquery, SqlQuantifiedComparison,
)
from mini_database import (
    ResultSet, ParseError, CompileError,
    SqlFuncCall, SqlLiteral, SqlColumnRef, SqlCase, SqlStar,
    SqlBinOp, SqlComparison, SqlLogic, SqlIsNull, SqlBetween, SqlInList,
    SqlAggCall,
    TokenType, Token, Lexer, Parser, KEYWORDS,
    SelectExpr, SelectStmt, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
)
from query_executor import Row, eval_expr

# Max recursion depth for recursive CTEs (safety limit)
MAX_RECURSION_DEPTH = 1000


# =============================================================================
# CTE AST Nodes
# =============================================================================

@dataclass
class CTEDefinition:
    """A single CTE definition: name [(col_list)] AS (query)"""
    name: str
    query: SelectStmt
    column_list: Optional[List[str]] = None


@dataclass
class CTEStatement:
    """WITH [RECURSIVE] cte1 AS (...), cte2 AS (...) main_query"""
    ctes: List[CTEDefinition]
    main_query: Any  # SelectStmt, InsertStmt, UpdateStmt, or DeleteStmt
    recursive: bool = False


# =============================================================================
# CTE Parser
# =============================================================================

class CTEParser(SubqueryParser):
    """Parser extended with CTE (WITH clause) support."""

    def parse(self):
        """Override to handle WITH at top level."""
        tok = self.peek()
        if tok.type == TokenType.IDENT and tok.value.upper() == 'WITH':
            return self._parse_with()
        return super().parse()

    def parse_multi(self):
        """Parse multiple statements, supporting WITH."""
        stmts = []
        while self.peek().type != TokenType.EOF:
            self.match(TokenType.SEMICOLON)
            if self.peek().type == TokenType.EOF:
                break
            tok = self.peek()
            if tok.type == TokenType.IDENT and tok.value.upper() == 'WITH':
                stmts.append(self._parse_with())
            else:
                stmts.append(self._parse_statement())
            self.match(TokenType.SEMICOLON)
        return stmts

    def _parse_with(self) -> CTEStatement:
        """Parse WITH [RECURSIVE] cte_def [, cte_def ...] main_query"""
        self.advance()  # consume WITH

        # Check for RECURSIVE
        recursive = False
        tok = self.peek()
        if tok.type == TokenType.IDENT and tok.value.upper() == 'RECURSIVE':
            recursive = True
            self.advance()

        # Parse CTE definitions
        ctes = [self._parse_cte_def()]
        while self.match(TokenType.COMMA):
            ctes.append(self._parse_cte_def())

        # Parse main query (SELECT, INSERT, UPDATE, DELETE)
        tok = self.peek()
        if tok.type == TokenType.SELECT:
            main_query = self._parse_select()
        elif tok.type == TokenType.INSERT:
            main_query = self._parse_insert()
        elif tok.type == TokenType.UPDATE:
            main_query = self._parse_update()
        elif tok.type == TokenType.DELETE:
            main_query = self._parse_delete()
        else:
            raise ParseError(f"Expected SELECT/INSERT/UPDATE/DELETE after WITH, got {tok.value!r}")

        return CTEStatement(ctes=ctes, main_query=main_query, recursive=recursive)

    def _parse_cte_def(self) -> CTEDefinition:
        """Parse: name [(col_list)] AS (query)"""
        # CTE name
        name_tok = self.advance()
        if name_tok.type != TokenType.IDENT:
            raise ParseError(f"Expected CTE name, got {name_tok.value!r}")
        name = name_tok.value

        # Optional column list
        column_list = None
        if self.peek().type == TokenType.LPAREN:
            # Lookahead: is this (col_list) or is it the AS (query)?
            # Column list comes BEFORE AS, query comes AFTER
            # So if next after ( is not SELECT, it's a column list
            if not self._is_subquery_ahead():
                self.advance()  # consume (
                column_list = []
                col_tok = self.advance()
                column_list.append(col_tok.value)
                while self.match(TokenType.COMMA):
                    col_tok = self.advance()
                    column_list.append(col_tok.value)
                self.expect(TokenType.RPAREN)

        # AS keyword
        if not self.match(TokenType.AS):
            raise ParseError(f"Expected AS in CTE definition for '{name}'")

        # (query)
        self.expect(TokenType.LPAREN)
        query = self._parse_select_or_union()
        self.expect(TokenType.RPAREN)

        return CTEDefinition(name=name, query=query, column_list=column_list)

    def _parse_select_or_union(self):
        """Parse SELECT possibly followed by UNION [ALL]."""
        left = self._parse_select()

        while self.peek().type == TokenType.UNION:
            self.advance()  # consume UNION
            all_ = bool(self.match(TokenType.ALL))
            right = self._parse_select()
            left = UnionQuery(left=left, right=right, all_=all_)

        return left


@dataclass
class UnionQuery:
    """UNION [ALL] of two queries."""
    left: Any  # SelectStmt or UnionQuery
    right: Any  # SelectStmt
    all_: bool = False  # UNION ALL vs UNION (distinct)


# =============================================================================
# Parse functions
# =============================================================================

def parse_cte_sql(sql: str):
    """Parse a single SQL statement with CTE support."""
    lexer = Lexer(sql)
    parser = CTEParser(lexer.tokens)
    return parser.parse()


def parse_cte_sql_multi(sql: str):
    """Parse multiple SQL statements with CTE support."""
    lexer = Lexer(sql)
    parser = CTEParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# CTEDB -- Database with CTE support
# =============================================================================

class CTEDB(SubqueryDB):
    """SubqueryDB extended with Common Table Expression support."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with CTE support."""
        stmt = parse_cte_sql(sql)
        return self._execute_cte_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_cte_sql_multi(sql)
        return [self._execute_cte_stmt(s) for s in stmts]

    def _execute_cte_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling CTEs."""
        if isinstance(stmt, CTEStatement):
            return self._exec_with_ctes(stmt)
        # Fall through to subquery execution
        return self._execute_subquery_stmt(stmt)

    def _exec_with_ctes(self, cte_stmt: CTEStatement) -> ResultSet:
        """Execute a WITH statement by materializing CTEs as virtual tables."""
        # Materialize each CTE in order
        cte_results = {}  # name -> ResultSet

        for cte_def in cte_stmt.ctes:
            if cte_stmt.recursive and self._is_recursive_cte(cte_def, cte_def.name):
                result = self._exec_recursive_cte(cte_def, cte_results)
            else:
                result = self._exec_cte_query(cte_def.query, cte_results)

            # Apply column renaming if column list specified
            if cte_def.column_list:
                if len(cte_def.column_list) != len(result.columns):
                    raise CompileError(
                        f"CTE '{cte_def.name}' has {len(cte_def.column_list)} column names "
                        f"but query produces {len(result.columns)} columns"
                    )
                result = ResultSet(columns=list(cte_def.column_list), rows=result.rows)

            cte_results[cte_def.name] = result

        # Execute main query with CTEs available
        return self._exec_main_with_ctes(cte_stmt.main_query, cte_results)

    def _is_recursive_cte(self, cte_def: CTEDefinition, name: str) -> bool:
        """Check if a CTE references itself (is actually recursive)."""
        return self._query_references_table(cte_def.query, name)

    def _query_references_table(self, query, table_name: str) -> bool:
        """Check if a query references a given table name."""
        if isinstance(query, UnionQuery):
            return (self._query_references_table(query.left, table_name) or
                    self._query_references_table(query.right, table_name))
        if isinstance(query, SelectStmt):
            if query.from_table:
                if isinstance(query.from_table, SubqueryTableRef):
                    if self._query_references_table(query.from_table.query, table_name):
                        return True
                elif query.from_table.table_name.lower() == table_name.lower():
                    return True
            for join in query.joins:
                if isinstance(join.table, SubqueryTableRef):
                    if self._query_references_table(join.table.query, table_name):
                        return True
                elif join.table.table_name.lower() == table_name.lower():
                    return True
        return False

    def _exec_recursive_cte(self, cte_def: CTEDefinition, cte_results: dict) -> ResultSet:
        """Execute a recursive CTE using iterative fixpoint.

        Recursive CTEs must have the form:
          base_query UNION [ALL] recursive_query
        where recursive_query references the CTE itself.
        """
        query = cte_def.query
        if not isinstance(query, UnionQuery):
            raise CompileError(f"Recursive CTE '{cte_def.name}' must use UNION or UNION ALL")

        name = cte_def.name
        union_all = query.all_

        # Execute base case (must not reference CTE itself)
        base_result = self._exec_cte_query(query.left, cte_results)
        columns = list(base_result.columns)

        # Apply column renaming for recursion
        if cte_def.column_list:
            if len(cte_def.column_list) != len(columns):
                raise CompileError(
                    f"CTE '{name}' has {len(cte_def.column_list)} column names "
                    f"but query produces {len(columns)} columns"
                )
            columns = list(cte_def.column_list)

        all_rows = list(base_result.rows)
        working_rows = list(base_result.rows)
        iteration = 0

        while working_rows and iteration < MAX_RECURSION_DEPTH:
            iteration += 1
            # Make current CTE result available for recursive query
            current_result = ResultSet(columns=columns, rows=working_rows)
            recursive_ctes = dict(cte_results)
            recursive_ctes[name] = current_result

            # Execute recursive part
            new_result = self._exec_cte_query(query.right, recursive_ctes)
            new_rows = new_result.rows

            if not new_rows:
                break

            if not union_all:
                # UNION: deduplicate against all_rows
                existing = {tuple(r) for r in all_rows}
                new_rows = [r for r in new_rows if tuple(r) not in existing]
                if not new_rows:
                    break

            all_rows.extend(new_rows)
            working_rows = new_rows

        if iteration >= MAX_RECURSION_DEPTH:
            raise CompileError(
                f"Recursive CTE '{name}' exceeded maximum depth of {MAX_RECURSION_DEPTH}"
            )

        return ResultSet(columns=columns, rows=all_rows)

    def _exec_cte_query(self, query, cte_results: dict) -> ResultSet:
        """Execute a query with CTE virtual tables available."""
        if isinstance(query, UnionQuery):
            return self._exec_union_query(query, cte_results)

        if isinstance(query, SelectStmt):
            # Check if FROM references a CTE
            if query.from_table and not isinstance(query.from_table, SubqueryTableRef):
                table_name = query.from_table.table_name
                if table_name in cte_results:
                    return self._exec_select_from_cte(query, cte_results)

            # Check joins for CTE references
            for join in query.joins:
                if not isinstance(join.table, SubqueryTableRef):
                    if join.table.table_name in cte_results:
                        return self._exec_select_with_cte_joins(query, cte_results)

            # No CTE references -- delegate to parent
            return self._execute_subquery_stmt(query)

        return self._execute_subquery_stmt(query)

    def _exec_union_query(self, union: UnionQuery, cte_results: dict) -> ResultSet:
        """Execute a UNION query."""
        left_result = self._exec_cte_query(union.left, cte_results)
        right_result = self._exec_cte_query(union.right, cte_results)

        columns = list(left_result.columns)
        all_rows = list(left_result.rows) + list(right_result.rows)

        if not union.all_:
            # UNION: deduplicate
            seen = set()
            unique = []
            for row in all_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    unique.append(row)
            all_rows = unique

        return ResultSet(columns=columns, rows=all_rows)

    def _exec_select_from_cte(self, stmt: SelectStmt, cte_results: dict) -> ResultSet:
        """Execute a SELECT whose FROM is a CTE."""
        table_name = stmt.from_table.table_name
        alias = stmt.from_table.alias or table_name
        cte_result = cte_results[table_name]

        # Build rows from CTE result
        rows = []
        for raw_row in cte_result.rows:
            data = {}
            for i, col_name in enumerate(cte_result.columns):
                data[col_name] = raw_row[i]
                data[f"{alias}.{col_name}"] = raw_row[i]
                if alias != table_name:
                    data[f"{table_name}.{col_name}"] = raw_row[i]
            rows.append(Row(data))

        # Apply JOINs
        if stmt.joins:
            rows = self._apply_cte_joins(rows, stmt.joins, cte_results)

        # WHERE
        if stmt.where:
            rows = [r for r in rows if subquery_eval_expr(stmt.where, r, self)]

        # GROUP BY
        if stmt.group_by:
            return self._exec_grouped_on_rows(stmt, rows)

        # Check for implicit aggregation
        has_agg = any(isinstance(col.expr, SqlAggCall) for col in stmt.columns)
        if has_agg:
            if not rows:
                # Implicit aggregation over empty set: COUNT(*)=0, SUM=0, AVG/MIN/MAX=NULL
                return self._exec_empty_agg(stmt.columns)
            return self._exec_grouped_on_rows(
                SelectStmt(
                    columns=stmt.columns,
                    group_by=[],
                    having=stmt.having,
                    order_by=stmt.order_by,
                    limit=stmt.limit,
                    offset=stmt.offset,
                ), rows)

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

    def _exec_empty_agg(self, columns) -> ResultSet:
        """Handle implicit aggregation over empty set (e.g., COUNT(*) FROM empty_cte)."""
        output_columns = []
        output_row = []
        for i, col in enumerate(columns):
            if col.alias:
                output_columns.append(col.alias)
            elif isinstance(col.expr, SqlAggCall):
                output_columns.append(f"{col.expr.func}_{i}")
            elif isinstance(col.expr, SqlColumnRef):
                output_columns.append(col.expr.column)
            else:
                output_columns.append(f"col_{i}")

            if isinstance(col.expr, SqlAggCall):
                func = col.expr.func.lower()
                if func == 'count':
                    output_row.append(0)
                elif func == 'sum':
                    output_row.append(0)
                else:
                    output_row.append(None)
            else:
                output_row.append(None)
        return ResultSet(columns=output_columns, rows=[output_row])

    def _exec_select_with_cte_joins(self, stmt: SelectStmt, cte_results: dict) -> ResultSet:
        """Execute a SELECT with CTE references in JOINs."""
        # Get base rows (could be regular table or CTE)
        if stmt.from_table and not isinstance(stmt.from_table, SubqueryTableRef):
            table_name = stmt.from_table.table_name
            if table_name in cte_results:
                # FROM is also a CTE
                cte_result = cte_results[table_name]
                alias = stmt.from_table.alias or table_name
                rows = []
                for raw_row in cte_result.rows:
                    data = {}
                    for i, col_name in enumerate(cte_result.columns):
                        data[col_name] = raw_row[i]
                        data[f"{alias}.{col_name}"] = raw_row[i]
                    rows.append(Row(data))
            else:
                # Regular table
                base_stmt = SelectStmt(
                    columns=[SelectExpr(expr=SqlStar())],
                    from_table=stmt.from_table,
                )
                base_result = self._execute_subquery_stmt(base_stmt)
                alias = stmt.from_table.alias or stmt.from_table.table_name
                rows = []
                for raw_row in base_result.rows:
                    data = {}
                    for i, col_name in enumerate(base_result.columns):
                        data[col_name] = raw_row[i]
                        data[f"{alias}.{col_name}"] = raw_row[i]
                        data[f"{stmt.from_table.table_name}.{col_name}"] = raw_row[i]
                    rows.append(Row(data))
        else:
            rows = [Row({})]

        # Apply joins (some may reference CTEs)
        rows = self._apply_cte_joins(rows, stmt.joins, cte_results)

        # WHERE
        if stmt.where:
            rows = [r for r in rows if subquery_eval_expr(stmt.where, r, self)]

        # GROUP BY
        if stmt.group_by:
            return self._exec_grouped_on_rows(stmt, rows)

        has_agg = any(isinstance(col.expr, SqlAggCall) for col in stmt.columns)
        if has_agg:
            return self._exec_grouped_on_rows(
                SelectStmt(
                    columns=stmt.columns,
                    group_by=[],
                    having=stmt.having,
                    order_by=stmt.order_by,
                    limit=stmt.limit,
                    offset=stmt.offset,
                ), rows)

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

    def _apply_cte_joins(self, left_rows: List[Row], joins: List[JoinClause],
                         cte_results: dict) -> List[Row]:
        """Apply JOINs, resolving CTE references."""
        result = left_rows
        for join in joins:
            if isinstance(join.table, SubqueryTableRef):
                # Derived table join -- delegate to parent
                sub_result = self._execute_subquery_stmt(join.table.query)
                alias = join.table.alias
                right_rows = []
                for raw_row in sub_result.rows:
                    data = {}
                    for i, col_name in enumerate(sub_result.columns):
                        data[col_name] = raw_row[i]
                        data[f"{alias}.{col_name}"] = raw_row[i]
                    right_rows.append(Row(data))
            elif join.table.table_name in cte_results:
                # CTE reference
                cte_result = cte_results[join.table.table_name]
                alias = join.table.alias or join.table.table_name
                right_rows = []
                for raw_row in cte_result.rows:
                    data = {}
                    for i, col_name in enumerate(cte_result.columns):
                        data[col_name] = raw_row[i]
                        data[f"{alias}.{col_name}"] = raw_row[i]
                    right_rows.append(Row(data))
            else:
                # Regular table
                table_name = join.table.table_name
                table_alias = join.table.alias or table_name
                join_result = self._execute_subquery_stmt(SelectStmt(
                    columns=[SelectExpr(expr=SqlStar())],
                    from_table=join.table,
                ))
                right_rows = []
                for raw_row in join_result.rows:
                    data = {}
                    for i, col_name in enumerate(join_result.columns):
                        data[col_name] = raw_row[i]
                        data[f"{table_alias}.{col_name}"] = raw_row[i]
                        data[f"{table_name}.{col_name}"] = raw_row[i]
                    right_rows.append(Row(data))

            join_type = join.join_type.lower()
            new_result = []

            if join_type == 'cross':
                for lr in result:
                    for rr in right_rows:
                        new_result.append(Row({**lr._data, **rr._data}))
            elif join_type in ('inner', 'join'):
                for lr in result:
                    for rr in right_rows:
                        merged = Row({**lr._data, **rr._data})
                        if join.condition is None or subquery_eval_expr(join.condition, merged, self):
                            new_result.append(merged)
            elif join_type == 'left':
                for lr in result:
                    matched = False
                    for rr in right_rows:
                        merged = Row({**lr._data, **rr._data})
                        if join.condition is None or subquery_eval_expr(join.condition, merged, self):
                            new_result.append(merged)
                            matched = True
                    if not matched:
                        null_data = {k: None for k in (right_rows[0]._data if right_rows else {})}
                        new_result.append(Row({**lr._data, **null_data}))
            else:
                for lr in result:
                    for rr in right_rows:
                        merged = Row({**lr._data, **rr._data})
                        if join.condition is None or subquery_eval_expr(join.condition, merged, self):
                            new_result.append(merged)

            result = new_result
        return result

    def _exec_main_with_ctes(self, main_query, cte_results: dict) -> ResultSet:
        """Execute the main query with CTEs available."""
        if isinstance(main_query, SelectStmt):
            return self._exec_cte_query(main_query, cte_results)

        if isinstance(main_query, InsertStmt):
            return self._exec_insert_with_ctes(main_query, cte_results)

        if isinstance(main_query, UpdateStmt):
            return self._exec_update_with_ctes(main_query, cte_results)

        if isinstance(main_query, DeleteStmt):
            return self._exec_delete_with_ctes(main_query, cte_results)

        return self._execute_subquery_stmt(main_query)

    def _exec_insert_with_ctes(self, stmt: InsertStmt, cte_results: dict) -> ResultSet:
        """INSERT INTO table SELECT ... FROM cte"""
        # If the insert has a subquery source, execute it with CTEs
        if hasattr(stmt, 'source_query') and stmt.source_query:
            result = self._exec_cte_query(stmt.source_query, cte_results)
            # Insert each row
            count = 0
            for row in result.rows:
                values = list(row)
                if stmt.columns:
                    self.storage.insert(stmt.table_name, dict(zip(stmt.columns, values)))
                else:
                    table = self.storage.catalog.get_table(stmt.table_name)
                    col_names = table.column_names()
                    self.storage.insert(stmt.table_name, dict(zip(col_names, values)))
                count += 1
            return ResultSet(columns=['rows_affected'], rows=[[count]], rows_affected=count)
        return self._execute_subquery_stmt(stmt)

    def _exec_update_with_ctes(self, stmt: UpdateStmt, cte_results: dict) -> ResultSet:
        """UPDATE with CTE -- just execute UPDATE normally after materializing CTEs."""
        # CTEs are used via subqueries in SET or WHERE
        return self._execute_subquery_stmt(stmt)

    def _exec_delete_with_ctes(self, stmt: DeleteStmt, cte_results: dict) -> ResultSet:
        """DELETE with CTE -- execute normally after materializing CTEs."""
        return self._execute_subquery_stmt(stmt)
