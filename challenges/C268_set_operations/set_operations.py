"""
C268: SQL Set Operations
Extends C267 (Common Table Expressions) with top-level set operations.

Set operations:
- UNION [ALL]: combine rows from two queries (existing in C267, now at top level too)
- INTERSECT [ALL]: rows present in both queries
- EXCEPT [ALL]: rows in left but not in right
- Chained set operations: SELECT ... UNION ... INTERSECT ... EXCEPT ...
- Set operations with ORDER BY, LIMIT, OFFSET on final result
- Set operations in CTEs, subqueries, and derived tables
- Column count validation across operands
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple
from collections import Counter

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C267_common_table_expressions')))
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

from cte import (
    CTEDB, CTEParser, CTEStatement, CTEDefinition, UnionQuery,
    parse_cte_sql, parse_cte_sql_multi,
    MAX_RECURSION_DEPTH,
)
from subqueries import (
    SubqueryDB, SubqueryParser, SubqueryTableRef,
    subquery_eval_expr,
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


# =============================================================================
# Set Operation AST Node
# =============================================================================

@dataclass
class SetOperation:
    """A set operation (UNION, INTERSECT, EXCEPT) between two queries.

    Left-associative chaining: A UNION B INTERSECT C becomes
    SetOperation('intersect', SetOperation('union', A, B), C)
    """
    op: str  # 'union', 'intersect', 'except'
    left: Any  # SelectStmt, SetOperation, or UnionQuery
    right: Any  # SelectStmt
    all_: bool = False  # ALL variant (preserves duplicates)


# =============================================================================
# Set Operations Parser
# =============================================================================

class SetOpParser(CTEParser):
    """Parser extended with INTERSECT and EXCEPT support at all levels."""

    def parse(self):
        """Override to handle top-level set operations."""
        tok = self.peek()
        if tok.type == TokenType.IDENT and tok.value.upper() == 'WITH':
            return self._parse_with()
        if tok.type == TokenType.SELECT:
            return self._parse_select_or_set_ops()
        # Non-SELECT statements (CREATE, INSERT, UPDATE, DELETE, etc.)
        return self._parse_statement()

    def parse_multi(self):
        """Parse multiple statements with set operation support."""
        stmts = []
        while self.peek().type != TokenType.EOF:
            self.match(TokenType.SEMICOLON)
            if self.peek().type == TokenType.EOF:
                break
            tok = self.peek()
            if tok.type == TokenType.IDENT and tok.value.upper() == 'WITH':
                stmts.append(self._parse_with())
            elif tok.type == TokenType.SELECT:
                stmts.append(self._parse_select_or_set_ops())
            else:
                stmts.append(self._parse_statement())
            self.match(TokenType.SEMICOLON)
        return stmts

    def _parse_with(self) -> CTEStatement:
        """Parse WITH, allowing set operations in CTE body and main query."""
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

        # Parse main query -- may be a set operation
        tok = self.peek()
        if tok.type == TokenType.SELECT:
            main_query = self._parse_select_or_set_ops()
        elif tok.type == TokenType.INSERT:
            main_query = self._parse_insert()
        elif tok.type == TokenType.UPDATE:
            main_query = self._parse_update()
        elif tok.type == TokenType.DELETE:
            main_query = self._parse_delete()
        else:
            raise ParseError(f"Expected SELECT/INSERT/UPDATE/DELETE after WITH, got {tok.value!r}")

        return CTEStatement(ctes=ctes, main_query=main_query, recursive=recursive)

    def _parse_select_or_union(self):
        """Override: Parse SELECT possibly followed by any set operation.
        Used inside CTE definitions (parenthesized context).
        ORDER BY/LIMIT/OFFSET don't need hoisting here since CTE body is parenthesized."""
        return self._parse_set_ops_chain()

    def _parse_select_or_set_ops(self):
        """Parse SELECT possibly followed by set operations at top level.

        Handles trailing ORDER BY, LIMIT, OFFSET that apply to the whole result.
        In SQL, ORDER BY/LIMIT/OFFSET after a set operation apply to the combined result,
        not the last individual SELECT. Since _parse_select greedily consumes these,
        we strip them from the rightmost SELECT and hoist them to the set operation level.
        """
        result = self._parse_set_ops_chain()

        if isinstance(result, SetOperation):
            # Strip ORDER BY/LIMIT/OFFSET from rightmost SELECT and hoist
            order_by, limit, offset = self._strip_ordering_from_rightmost(result)
            if order_by or limit is not None or offset is not None:
                result = SetOpWithClauses(
                    set_op=result,
                    order_by=order_by,
                    limit=limit,
                    offset=offset,
                )

        return result

    def _strip_ordering_from_rightmost(self, node):
        """Strip ORDER BY/LIMIT/OFFSET from the rightmost SelectStmt in a set operation tree."""
        order_by = None
        limit = None
        offset = None

        # Find the rightmost SelectStmt
        target = node
        while isinstance(target, SetOperation):
            target = target.right

        if isinstance(target, SelectStmt):
            if target.order_by or target.limit is not None or target.offset is not None:
                order_by = target.order_by
                limit = target.limit
                offset = target.offset
                # Mutate the SelectStmt to remove ordering
                target.order_by = None
                target.limit = None
                target.offset = None

        return order_by, limit, offset

    def _parse_set_ops_chain(self):
        """Parse a chain of set operations (left-associative).

        Grammar: select_stmt ((UNION|INTERSECT|EXCEPT) [ALL] select_stmt)*
        """
        left = self._parse_select()

        while self._is_set_op_keyword():
            op, all_ = self._consume_set_op()
            right = self._parse_select()
            left = SetOperation(op=op, left=left, right=right, all_=all_)

        return left

    def _is_set_op_keyword(self) -> bool:
        """Check if current token is a set operation keyword."""
        tok = self.peek()
        return tok.type in (TokenType.UNION, TokenType.INTERSECT, TokenType.EXCEPT)

    def _consume_set_op(self) -> Tuple[str, bool]:
        """Consume a set operation keyword and optional ALL. Returns (op_name, is_all)."""
        tok = self.advance()
        if tok.type == TokenType.UNION:
            op = 'union'
        elif tok.type == TokenType.INTERSECT:
            op = 'intersect'
        elif tok.type == TokenType.EXCEPT:
            op = 'except'
        else:
            raise ParseError(f"Expected UNION/INTERSECT/EXCEPT, got {tok.value!r}")

        all_ = bool(self.match(TokenType.ALL))
        return op, all_

    def _parse_order_by_clause(self):
        """Parse ORDER BY clause (reuse parent logic but return the list)."""
        self.advance()  # ORDER
        self.expect(TokenType.BY)
        order_items = []
        while True:
            expr = self._parse_expr()
            asc = True
            if self.match(TokenType.ASC):
                asc = True
            elif self.match(TokenType.DESC):
                asc = False
            order_items.append((expr, asc))
            if not self.match(TokenType.COMMA):
                break
        return order_items


@dataclass
class SetOpWithClauses:
    """A set operation with trailing ORDER BY / LIMIT / OFFSET."""
    set_op: Any  # SetOperation or UnionQuery
    order_by: Optional[List[Tuple[Any, bool]]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


# =============================================================================
# Parse functions
# =============================================================================

def parse_set_op_sql(sql: str):
    """Parse a single SQL statement with set operation support."""
    lexer = Lexer(sql)
    parser = SetOpParser(lexer.tokens)
    return parser.parse()


def parse_set_op_sql_multi(sql: str):
    """Parse multiple SQL statements with set operation support."""
    lexer = Lexer(sql)
    parser = SetOpParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# SetOpDB -- Database with set operation support
# =============================================================================

class SetOpDB(CTEDB):
    """CTEDB extended with INTERSECT and EXCEPT support."""

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with set operation support."""
        stmt = parse_set_op_sql(sql)
        return self._execute_set_op_stmt(stmt)

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_set_op_sql_multi(sql)
        return [self._execute_set_op_stmt(s) for s in stmts]

    def _execute_set_op_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling set operations."""
        if isinstance(stmt, SetOpWithClauses):
            return self._exec_set_op_with_clauses(stmt)
        if isinstance(stmt, SetOperation):
            return self._exec_set_operation(stmt, {})
        if isinstance(stmt, CTEStatement):
            return self._exec_with_ctes_set_ops(stmt)
        # UnionQuery from C267 still works
        if isinstance(stmt, UnionQuery):
            return self._exec_union_query(stmt, {})
        # Fall through to CTE/subquery execution
        return self._execute_cte_stmt(stmt)

    def _exec_with_ctes_set_ops(self, cte_stmt: CTEStatement) -> ResultSet:
        """Execute a WITH statement, supporting set operations in main query."""
        # Materialize each CTE
        cte_results = {}

        for cte_def in cte_stmt.ctes:
            if cte_stmt.recursive and self._is_recursive_cte_set_op(cte_def, cte_def.name):
                result = self._exec_recursive_cte_set_op(cte_def, cte_results)
            else:
                result = self._exec_cte_query_set_op(cte_def.query, cte_results)

            # Apply column renaming
            if cte_def.column_list:
                if len(cte_def.column_list) != len(result.columns):
                    raise CompileError(
                        f"CTE '{cte_def.name}' has {len(cte_def.column_list)} column names "
                        f"but query produces {len(result.columns)} columns"
                    )
                result = ResultSet(columns=list(cte_def.column_list), rows=result.rows)

            cte_results[cte_def.name] = result

        # Execute main query with CTEs available
        return self._exec_main_with_ctes_set_ops(cte_stmt.main_query, cte_results)

    def _is_recursive_cte_set_op(self, cte_def: CTEDefinition, name: str) -> bool:
        """Check if a CTE references itself, supporting SetOperation nodes."""
        return self._query_references_table_set_op(cte_def.query, name)

    def _query_references_table_set_op(self, query, table_name: str) -> bool:
        """Check if a query references a given table name (set-op aware)."""
        if isinstance(query, SetOperation):
            return (self._query_references_table_set_op(query.left, table_name) or
                    self._query_references_table_set_op(query.right, table_name))
        if isinstance(query, SetOpWithClauses):
            return self._query_references_table_set_op(query.set_op, table_name)
        # Delegate to parent for UnionQuery and SelectStmt
        return self._query_references_table(query, table_name)

    def _exec_recursive_cte_set_op(self, cte_def: CTEDefinition, cte_results: dict) -> ResultSet:
        """Execute a recursive CTE that may use SetOperation instead of UnionQuery."""
        query = cte_def.query
        name = cte_def.name

        # Recursive CTEs require UNION form
        if isinstance(query, SetOperation) and query.op == 'union':
            union_all = query.all_
            base_query = query.left
            recursive_query = query.right
        elif isinstance(query, UnionQuery):
            # Legacy support
            return self._exec_recursive_cte(cte_def, cte_results)
        else:
            raise CompileError(f"Recursive CTE '{name}' must use UNION or UNION ALL")

        # Execute base case
        base_result = self._exec_cte_query_set_op(base_query, cte_results)
        columns = list(base_result.columns)

        # Apply column renaming
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
            current_result = ResultSet(columns=columns, rows=working_rows)
            recursive_ctes = dict(cte_results)
            recursive_ctes[name] = current_result

            new_result = self._exec_cte_query_set_op(recursive_query, recursive_ctes)
            new_rows = new_result.rows

            if not new_rows:
                break

            if not union_all:
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

    def _exec_cte_query_set_op(self, query, cte_results: dict) -> ResultSet:
        """Execute a query with CTE virtual tables, supporting set operations."""
        if isinstance(query, SetOperation):
            return self._exec_set_operation(query, cte_results)
        if isinstance(query, SetOpWithClauses):
            return self._exec_set_op_with_clauses_cte(query, cte_results)
        # Delegate to parent for UnionQuery and SelectStmt
        return self._exec_cte_query(query, cte_results)

    def _exec_main_with_ctes_set_ops(self, main_query, cte_results: dict) -> ResultSet:
        """Execute the main query after CTEs, supporting set operations."""
        if isinstance(main_query, SetOperation):
            return self._exec_set_operation(main_query, cte_results)
        if isinstance(main_query, SetOpWithClauses):
            return self._exec_set_op_with_clauses_cte(main_query, cte_results)
        # Delegate to parent
        return self._exec_main_with_ctes(main_query, cte_results)

    def _exec_set_operation(self, op: SetOperation, cte_results: dict) -> ResultSet:
        """Execute a set operation (UNION, INTERSECT, EXCEPT)."""
        left_result = self._exec_cte_query_set_op(op.left, cte_results)
        right_result = self._exec_cte_query_set_op(op.right, cte_results)

        # Validate column count
        if len(left_result.columns) != len(right_result.columns):
            raise CompileError(
                f"Set operation requires equal column counts: "
                f"left has {len(left_result.columns)}, right has {len(right_result.columns)}"
            )

        columns = list(left_result.columns)
        left_rows = left_result.rows
        right_rows = right_result.rows

        if op.op == 'union':
            return self._exec_union(columns, left_rows, right_rows, op.all_)
        elif op.op == 'intersect':
            return self._exec_intersect(columns, left_rows, right_rows, op.all_)
        elif op.op == 'except':
            return self._exec_except(columns, left_rows, right_rows, op.all_)
        else:
            raise CompileError(f"Unknown set operation: {op.op}")

    def _exec_union(self, columns, left_rows, right_rows, all_: bool) -> ResultSet:
        """UNION [ALL]: combine rows."""
        all_rows = list(left_rows) + list(right_rows)
        if not all_:
            all_rows = self._deduplicate(all_rows)
        return ResultSet(columns=columns, rows=all_rows)

    def _exec_intersect(self, columns, left_rows, right_rows, all_: bool) -> ResultSet:
        """INTERSECT [ALL]: rows present in both."""
        if all_:
            # INTERSECT ALL: min(count_left, count_right) for each distinct row
            left_counts = Counter(tuple(r) for r in left_rows)
            right_counts = Counter(tuple(r) for r in right_rows)
            result = []
            for row_key, left_count in left_counts.items():
                right_count = right_counts.get(row_key, 0)
                count = min(left_count, right_count)
                for _ in range(count):
                    result.append(list(row_key))
            return ResultSet(columns=columns, rows=result)
        else:
            # INTERSECT: unique rows present in both
            left_set = {tuple(r) for r in left_rows}
            right_set = {tuple(r) for r in right_rows}
            common = left_set & right_set
            # Preserve order from left side
            seen = set()
            result = []
            for r in left_rows:
                key = tuple(r)
                if key in common and key not in seen:
                    seen.add(key)
                    result.append(r)
            return ResultSet(columns=columns, rows=result)

    def _exec_except(self, columns, left_rows, right_rows, all_: bool) -> ResultSet:
        """EXCEPT [ALL]: rows in left but not in right."""
        if all_:
            # EXCEPT ALL: subtract right counts from left counts
            right_counts = Counter(tuple(r) for r in right_rows)
            result = []
            for r in left_rows:
                key = tuple(r)
                if right_counts.get(key, 0) > 0:
                    right_counts[key] -= 1
                else:
                    result.append(r)
            return ResultSet(columns=columns, rows=result)
        else:
            # EXCEPT: unique rows in left that don't appear in right
            right_set = {tuple(r) for r in right_rows}
            seen = set()
            result = []
            for r in left_rows:
                key = tuple(r)
                if key not in right_set and key not in seen:
                    seen.add(key)
                    result.append(r)
            return ResultSet(columns=columns, rows=result)

    def _deduplicate(self, rows):
        """Remove duplicate rows, preserving order."""
        seen = set()
        result = []
        for r in rows:
            key = tuple(r)
            if key not in seen:
                seen.add(key)
                result.append(r)
        return result

    def _exec_set_op_with_clauses(self, stmt: SetOpWithClauses) -> ResultSet:
        """Execute a set operation with trailing ORDER BY / LIMIT / OFFSET."""
        return self._exec_set_op_with_clauses_cte(stmt, {})

    def _exec_set_op_with_clauses_cte(self, stmt: SetOpWithClauses, cte_results: dict) -> ResultSet:
        """Execute SetOpWithClauses with CTE context."""
        # Execute the set operation
        if isinstance(stmt.set_op, SetOperation):
            result = self._exec_set_operation(stmt.set_op, cte_results)
        elif isinstance(stmt.set_op, UnionQuery):
            result = self._exec_union_query(stmt.set_op, cte_results)
        else:
            result = self._exec_cte_query_set_op(stmt.set_op, cte_results)

        rows = list(result.rows)
        columns = result.columns

        # ORDER BY
        if stmt.order_by:
            rows = self._sort_set_op_rows(rows, columns, stmt.order_by)

        # OFFSET
        if stmt.offset:
            rows = rows[stmt.offset:]

        # LIMIT
        if stmt.limit is not None:
            rows = rows[:stmt.limit]

        return ResultSet(columns=columns, rows=rows)

    def _sort_set_op_rows(self, rows, columns, order_by):
        """Sort set operation result rows by ORDER BY expressions."""
        # Build Row objects for expression evaluation
        row_objects = []
        for raw_row in rows:
            data = {}
            for i, col_name in enumerate(columns):
                data[col_name] = raw_row[i]
            row_objects.append((raw_row, Row(data)))

        def sort_key(item):
            raw_row, row_obj = item
            key = []
            for expr, asc in order_by:
                if isinstance(expr, SqlColumnRef):
                    # Could be column name or column position
                    col = expr.column
                    if col in row_obj._data:
                        val = row_obj._data[col]
                    else:
                        # Try case-insensitive match
                        val = None
                        for k, v in row_obj._data.items():
                            if k.lower() == col.lower():
                                val = v
                                break
                elif isinstance(expr, SqlLiteral) and isinstance(expr.value, (int, float)):
                    # Positional reference (1-based)
                    idx = int(expr.value) - 1
                    if 0 <= idx < len(raw_row):
                        val = raw_row[idx]
                    else:
                        val = None
                else:
                    try:
                        val = subquery_eval_expr(expr, row_obj, self)
                    except Exception:
                        val = None

                # Handle None in sorting
                if val is None:
                    key.append((1, ''))  # NULLs sort last
                elif isinstance(val, str):
                    key.append((0, val) if asc else (0, val))
                else:
                    key.append((0, val) if asc else (0, val))
            return key

        # Sort with ASC/DESC handling
        def full_sort_key(item):
            raw_row, row_obj = item
            keys = []
            for expr, asc in order_by:
                if isinstance(expr, SqlColumnRef):
                    col = expr.column
                    val = row_obj._data.get(col)
                    if val is None:
                        for k, v in row_obj._data.items():
                            if k.lower() == col.lower():
                                val = v
                                break
                elif isinstance(expr, SqlLiteral) and isinstance(expr.value, (int, float)):
                    idx = int(expr.value) - 1
                    val = raw_row[idx] if 0 <= idx < len(raw_row) else None
                else:
                    try:
                        val = subquery_eval_expr(expr, row_obj, self)
                    except Exception:
                        val = None

                if val is None:
                    keys.append((1, 0, ''))
                elif isinstance(val, (int, float)):
                    keys.append((0, val if asc else -val, ''))
                elif isinstance(val, str):
                    keys.append((0, 0, val))
                else:
                    keys.append((0, 0, str(val)))
            return keys

        # Simple approach: sort using comparison
        decorated = [(full_sort_key(item), item) for item in row_objects]

        # Handle DESC by reversing individual sort fields
        # We need a custom approach for mixed ASC/DESC
        from functools import cmp_to_key

        def compare_rows(a, b):
            _, (raw_a, row_a) = a
            _, (raw_b, row_b) = b
            for expr, asc in order_by:
                val_a = self._eval_order_expr(expr, raw_a, row_a, columns)
                val_b = self._eval_order_expr(expr, raw_b, row_b, columns)

                # Both None
                if val_a is None and val_b is None:
                    continue
                if val_a is None:
                    return 1  # NULLs last
                if val_b is None:
                    return -1

                if val_a < val_b:
                    return -1 if asc else 1
                if val_a > val_b:
                    return 1 if asc else -1
            return 0

        decorated.sort(key=cmp_to_key(compare_rows))
        return [item[1][0] for item in decorated]

    def _eval_order_expr(self, expr, raw_row, row_obj, columns):
        """Evaluate an ORDER BY expression for set operation sorting."""
        if isinstance(expr, SqlColumnRef):
            col = expr.column
            val = row_obj._data.get(col)
            if val is None:
                for k, v in row_obj._data.items():
                    if k.lower() == col.lower():
                        return v
            return val
        elif isinstance(expr, SqlLiteral) and isinstance(expr.value, (int, float)):
            idx = int(expr.value) - 1
            return raw_row[idx] if 0 <= idx < len(raw_row) else None
        else:
            try:
                return subquery_eval_expr(expr, row_obj, self)
            except Exception:
                return None
