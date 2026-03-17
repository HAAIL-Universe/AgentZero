"""
C254: UNION / INTERSECT / EXCEPT Set Operations
Extends C253 (Common Table Expressions) / C252 (Window Functions) / ... / C247 (Mini Database)

Adds SQL set operations to the database engine:
- UNION: combine result sets, removing duplicates
- UNION ALL: combine result sets, keeping duplicates
- INTERSECT: rows appearing in both result sets
- INTERSECT ALL: intersection preserving duplicate counts
- EXCEPT: rows in left but not in right
- EXCEPT ALL: difference preserving duplicate counts
- Chained operations: SELECT ... UNION ... INTERSECT ... EXCEPT ...
- ORDER BY / LIMIT on final result
- Column count validation across operands
- Works with CTEs, window functions, aggregates, JOINs
- Parenthesized subqueries for precedence control
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import C253 (which imports C252 -> C251 -> ... -> C247)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C253_common_table_expressions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C252_sql_window_functions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C251_sql_triggers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C250_sql_views'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C249_stored_procedures'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from common_table_expressions import (
    CTEDB, CTELexer, CTEParser, CTESelectStmt, CTEDef, UnionStmt,
    MAX_RECURSIVE_DEPTH,
)

from mini_database import (
    MiniDB, ResultSet, DatabaseError,
    Lexer, Parser, Token, TokenType,
    SelectStmt, SelectExpr, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    SqlColumnRef, SqlLiteral, SqlBinOp, SqlComparison, SqlLogic,
    SqlIsNull, SqlFuncCall, SqlAggCall, SqlBetween, SqlInList,
    SqlCase, SqlStar,
    ParseError, KEYWORDS,
)

from transaction_manager import IsolationLevel


# =============================================================================
# Set Operation AST Node
# =============================================================================

@dataclass
class SetOpStmt:
    """A set operation combining two SELECT results.

    Supports: UNION, UNION ALL, INTERSECT, INTERSECT ALL, EXCEPT, EXCEPT ALL
    Left and right can be SelectStmt, CTESelectStmt, or nested SetOpStmt.
    """
    op: str           # 'UNION', 'INTERSECT', 'EXCEPT'
    all: bool         # True for ALL variant (keep duplicates)
    left: Any         # SelectStmt | CTESelectStmt | SetOpStmt
    right: Any        # SelectStmt | CTESelectStmt | SetOpStmt
    order_by: Optional[List[Tuple]] = None   # applied to final result
    limit: Optional[int] = None
    offset: Optional[int] = None


# =============================================================================
# Set Operations Lexer
# =============================================================================

class SetOpLexer(CTELexer):
    """Lexer extended for set operations.
    UNION, INTERSECT, EXCEPT, ALL stay as IDENT tokens
    and are recognized contextually by the parser."""

    def __init__(self, sql: str):
        super().__init__(sql)


# =============================================================================
# Set Operations Parser (extends CTEParser)
# =============================================================================

SET_OP_WORDS = {'union', 'intersect', 'except'}


class SetOpParser(CTEParser):
    """Parser extended with UNION/INTERSECT/EXCEPT support.

    After parsing a SELECT (or CTE), checks for set operation keywords
    and chains them left-associatively.

    INTERSECT binds tighter than UNION/EXCEPT (standard SQL precedence).
    """

    def __init__(self, tokens):
        super().__init__(tokens)

    def _is_set_op_word(self, value):
        """Check if a token value is a set operation keyword."""
        return value and value.lower() in SET_OP_WORDS

    def _parse_table_ref(self):
        """Override to prevent set-op keywords from being consumed as table aliases."""
        name = self.expect(TokenType.IDENT).value
        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENT).value
        elif (self.peek().type == TokenType.IDENT and
              self.peek().value.lower() not in KEYWORDS and
              not self._is_set_op_word(self.peek().value)):
            alias = self.advance().value
        return TableRef(table_name=name, alias=alias)

    def _parse_select_list(self):
        """Override to prevent set-op keywords from being consumed as column aliases."""
        columns = []
        while True:
            if self._is_window_func_ahead():
                columns.append(self._parse_select_item_setop())
            else:
                tt = self._peek_type()
                if isinstance(tt, TokenType) and tt == TokenType.STAR:
                    self.advance()
                    columns.append(SelectExpr(expr=SqlStar(table=None)))
                else:
                    expr = self._parse_expr()
                    alias = None
                    if self.match(TokenType.AS):
                        alias = self._expect_ident()
                    elif isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.IDENT:
                        next_val = self.peek().value.lower()
                        if (next_val not in ('from', 'where', 'into', 'group', 'order',
                                             'having', 'limit', 'union') and
                                not self._is_set_op_word(next_val)):
                            alias = self.advance().value
                    columns.append(SelectExpr(expr=expr, alias=alias))
            if not self.match(TokenType.COMMA):
                break
        return columns

    def _parse_select_item_setop(self) -> SelectExpr:
        """Window function select item with set-op keyword protection."""
        win_call = self._parse_window_call()
        alias = None
        if self._peek_word() == 'as':
            self._advance()
            alias = self._expect_ident()
        elif self._peek_type() == TokenType.IDENT:
            next_val = self.tokens[self.pos].value.lower() if self.pos < len(self.tokens) else ''
            if (next_val not in ('from', 'where', 'into', 'group', 'order',
                                 'having', 'limit', 'union') and
                    not self._is_set_op_word(next_val)):
                alias = self._expect_ident()
        return SelectExpr(expr=win_call, alias=alias)

    def _parse_statement(self):
        """Override to detect set operations after SELECT."""
        # Handle parenthesized subqueries at top level: (SELECT ... UNION SELECT ...) INTERSECT ...
        if self._peek_type() == TokenType.LPAREN:
            saved_pos = self.pos
            try:
                stmt = self._parse_set_operand()
                # Check for set operation after the parenthesized subquery
                if self._peek_word() in SET_OP_WORDS:
                    return self._parse_set_operations(stmt)
                return stmt
            except (ParseError, IndexError):
                self.pos = saved_pos

        stmt = super()._parse_statement()

        # Check for set operation keywords after a SELECT/CTE statement
        if isinstance(stmt, (SelectStmt, CTESelectStmt)):
            result = self._parse_set_operations(stmt)
            # If a CTE's main_stmt was used as the left of a set operation,
            # restructure so the set operation is inside the CTE scope
            if isinstance(result, SetOpStmt) and isinstance(stmt, CTESelectStmt):
                cte_copy = CTESelectStmt(
                    ctes=stmt.ctes,
                    main_stmt=result,
                    recursive=stmt.recursive,
                )
                # Replace the left operand with just the original main_stmt
                self._replace_leftmost(result, stmt.main_stmt)
                return cte_copy
            stmt = result

        return stmt

    def _replace_leftmost(self, setop, new_left):
        """Replace the leftmost leaf of a SetOpStmt tree."""
        node = setop
        while isinstance(node.left, SetOpStmt):
            node = node.left
        node.left = new_left

    def _parse_set_operations(self, left):
        """Parse chained set operations with SQL-standard precedence.

        INTERSECT binds tighter than UNION/EXCEPT.
        We implement this with a two-level grammar:
          set_expr = intersect_expr ((UNION|EXCEPT) [ALL] intersect_expr)*
          intersect_expr = primary (INTERSECT [ALL] primary)*
        """
        # First, parse any INTERSECT operations (higher precedence)
        left = self._parse_intersect_chain(left)

        # Then parse UNION/EXCEPT at lower precedence
        while True:
            word = self._peek_word()
            if word in ('union', 'except'):
                op = word.upper()
                self._advance()  # consume UNION/EXCEPT
                all_flag = False
                if self._peek_word() == 'all':
                    all_flag = True
                    self._advance()  # consume ALL
                right = self._parse_set_operand()
                right = self._parse_intersect_chain(right)
                left = SetOpStmt(op=op, all=all_flag, left=left, right=right)
            else:
                break

        # Steal ORDER BY / LIMIT / OFFSET from rightmost operand
        # (parent parser greedily consumes these into the last SELECT)
        if isinstance(left, SetOpStmt):
            self._steal_trailing_clauses(left)
            # Also check for unconsumed ORDER BY/LIMIT/OFFSET in token stream
            # (happens when right operand was parenthesized or a SetOpStmt)
            if self._peek_word() == 'order':
                left.order_by = self._parse_order_by_clause()
            if self._peek_word() == 'limit':
                self._advance()
                tok = self.tokens[self.pos]
                self._advance()
                left.limit = int(tok.value)
            if self._peek_word() == 'offset':
                self._advance()
                tok = self.tokens[self.pos]
                self._advance()
                left.offset = int(tok.value)

        return left

    def _steal_trailing_clauses(self, setop: SetOpStmt):
        """Move ORDER BY/LIMIT/OFFSET from rightmost SELECT to set operation."""
        # Find the rightmost leaf
        node = setop
        while isinstance(node.right, SetOpStmt):
            node = node.right
        right = node.right
        if isinstance(right, SelectStmt):
            if right.order_by:
                setop.order_by = right.order_by
                right.order_by = None
            if right.limit is not None:
                setop.limit = right.limit
                right.limit = None
            if right.offset is not None:
                setop.offset = right.offset
                right.offset = None

    def _parse_intersect_chain(self, left):
        """Parse INTERSECT [ALL] chains (higher precedence than UNION/EXCEPT)."""
        while True:
            word = self._peek_word()
            if word == 'intersect':
                self._advance()  # consume INTERSECT
                all_flag = False
                if self._peek_word() == 'all':
                    all_flag = True
                    self._advance()  # consume ALL
                right = self._parse_set_operand()
                left = SetOpStmt(op='INTERSECT', all=all_flag, left=left, right=right)
            else:
                break
        return left

    def _parse_set_operand(self):
        """Parse a single operand for a set operation.

        Can be:
        - A parenthesized set expression: (SELECT ... UNION SELECT ...)
        - A plain SELECT statement
        - A CTE statement (WITH ... SELECT ...)
        """
        if self._peek_type() == TokenType.LPAREN:
            # Check if this is a parenthesized subquery for precedence
            saved_pos = self.pos
            self._advance()  # consume (
            try:
                inner = super()._parse_statement()
                inner = self._parse_set_operations(inner)
                if self._peek_type() == TokenType.RPAREN:
                    self._advance()  # consume )
                    return inner
                else:
                    # Not a valid parenthesized subquery, restore
                    self.pos = saved_pos
            except (ParseError, IndexError):
                self.pos = saved_pos

        # Parse a regular SELECT or CTE statement
        return super()._parse_statement()

    def _parse_order_by_clause(self):
        """Parse ORDER BY clause for set operations."""
        if self._peek_word() != 'order':
            return None
        self._advance()  # consume ORDER
        if self._peek_word() != 'by':
            raise ParseError("Expected BY after ORDER")
        self._advance()  # consume BY

        order_items = []
        while True:
            expr = self._parse_expr()
            asc = True
            word = self._peek_word()
            if word == 'asc':
                self._advance()
            elif word == 'desc':
                asc = False
                self._advance()
            order_items.append((expr, asc))
            if not self.match(TokenType.COMMA):
                break
        return order_items

    def _parse_cte_body(self):
        """Override: CTE body can now contain set operations too."""
        left = self._parse_inner_select()

        # Check for set operations inside CTE body
        word = self._peek_word()
        if word == 'union':
            # For recursive CTEs, we still need to produce UnionStmt
            # Check if we're in a recursive CTE context
            self._advance()  # consume UNION
            union_all = False
            if self._peek_word() == 'all':
                union_all = True
                self._advance()  # consume ALL
            right = self._parse_inner_select()

            # Check for further chained set ops after the right side
            next_word = self._peek_word()
            if next_word in ('union', 'intersect', 'except'):
                # Chained set ops in CTE body -- wrap as SetOpStmt
                first = SetOpStmt(
                    op='UNION', all=union_all,
                    left=left, right=right
                )
                return self._parse_cte_body_chain(first)

            return UnionStmt(left=left, right=right, union_all=union_all)
        elif word in ('intersect', 'except'):
            op = word.upper()
            self._advance()
            all_flag = False
            if self._peek_word() == 'all':
                all_flag = True
                self._advance()
            right = self._parse_inner_select()
            first = SetOpStmt(op=op, all=all_flag, left=left, right=right)
            next_word = self._peek_word()
            if next_word in ('union', 'intersect', 'except'):
                return self._parse_cte_body_chain(first)
            return first

        return left

    def _parse_cte_body_chain(self, left):
        """Parse remaining set operations inside a CTE body."""
        while True:
            word = self._peek_word()
            if word in ('union', 'intersect', 'except'):
                op = word.upper()
                self._advance()
                all_flag = False
                if self._peek_word() == 'all':
                    all_flag = True
                    self._advance()
                right = self._parse_inner_select()
                left = SetOpStmt(op=op, all=all_flag, left=left, right=right)
            else:
                break
        return left


# =============================================================================
# Set Operations DB (extends CTEDB)
# =============================================================================

class SetOpDB(CTEDB):
    """CTEDB extended with UNION/INTERSECT/EXCEPT set operations."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)

    def execute(self, sql: str) -> ResultSet:
        stmts = self._parse_setop(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_setop_stmt(stmt))
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List[ResultSet]:
        stmts = self._parse_setop(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_setop_stmt(stmt))
        return results

    def _parse_setop(self, sql: str) -> List[Any]:
        lexer = SetOpLexer(sql)
        parser = SetOpParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _exec_select_with_cte_context(self, stmt) -> ResultSet:
        """Override to handle SetOpStmt in CTE context."""
        if isinstance(stmt, SetOpStmt):
            return self._exec_set_operation(stmt)
        return super()._exec_select_with_cte_context(stmt)

    def _execute_setop_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling set operations."""
        if isinstance(stmt, SetOpStmt):
            return self._exec_set_operation(stmt)
        if isinstance(stmt, CTESelectStmt):
            # CTE main_stmt might be a SetOpStmt
            if isinstance(stmt.main_stmt, SetOpStmt):
                return self._exec_cte_with_set_ops(stmt)
            return self._exec_with_ctes(stmt)
        if isinstance(stmt, SelectStmt):
            return self._exec_select_with_cte_context(stmt)
        return self._execute_trigger_stmt(stmt)

    # =========================================================================
    # Set Operation Execution
    # =========================================================================

    def _exec_set_operation(self, stmt: SetOpStmt) -> ResultSet:
        """Execute a UNION/INTERSECT/EXCEPT set operation."""
        # Execute left and right sides
        left_result = self._exec_setop_operand(stmt.left)
        right_result = self._exec_setop_operand(stmt.right)

        # Validate column counts match
        if len(left_result.columns) != len(right_result.columns):
            raise DatabaseError(
                f"Set operation requires equal column counts: "
                f"left has {len(left_result.columns)}, right has {len(right_result.columns)}"
            )

        # Use left side's column names for the result
        columns = list(left_result.columns)

        # Apply the set operation
        if stmt.op == 'UNION':
            result_rows = self._union(left_result.rows, right_result.rows, stmt.all)
        elif stmt.op == 'INTERSECT':
            result_rows = self._intersect(left_result.rows, right_result.rows, stmt.all)
        elif stmt.op == 'EXCEPT':
            result_rows = self._except(left_result.rows, right_result.rows, stmt.all)
        else:
            raise DatabaseError(f"Unknown set operation: {stmt.op}")

        # ORDER BY on final result
        if stmt.order_by:
            result_rows = self._sort_set_result(result_rows, columns, stmt.order_by)

        # OFFSET
        if stmt.offset:
            result_rows = result_rows[stmt.offset:]

        # LIMIT
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=columns, rows=result_rows)

    def _exec_setop_operand(self, operand) -> ResultSet:
        """Execute one operand of a set operation."""
        if isinstance(operand, SetOpStmt):
            return self._exec_set_operation(operand)
        if isinstance(operand, CTESelectStmt):
            if isinstance(operand.main_stmt, SetOpStmt):
                return self._exec_cte_with_set_ops(operand)
            return self._exec_with_ctes(operand)
        if isinstance(operand, SelectStmt):
            return self._exec_select_with_cte_context(operand)
        return self._execute_setop_stmt(operand)

    def _exec_cte_select(self, stmt) -> ResultSet:
        """Override to handle SetOpStmt in CTE context."""
        if isinstance(stmt, SetOpStmt):
            return self._exec_set_operation(stmt)
        return self._exec_select_with_cte_context(stmt)

    def _materialize_cte(self, cte_def):
        """Override to handle SetOpStmt CTE bodies."""
        if isinstance(cte_def.body, SetOpStmt):
            self._materialize_cte_setop(cte_def)
        else:
            super()._materialize_cte(cte_def)

    def _exec_cte_with_set_ops(self, cte_stmt: CTESelectStmt) -> ResultSet:
        """Execute CTE where main_stmt is a SetOpStmt."""
        saved_tables = dict(self._cte_tables)
        saved_columns = dict(self._cte_columns)
        try:
            for cte_def in cte_stmt.ctes:
                self._materialize_cte_setop(cte_def)
            return self._exec_set_operation(cte_stmt.main_stmt)
        finally:
            self._cte_tables = saved_tables
            self._cte_columns = saved_columns

    def _materialize_cte_setop(self, cte_def: CTEDef):
        """Materialize CTE, handling SetOpStmt bodies."""
        if isinstance(cte_def.body, SetOpStmt):
            # Non-recursive CTE with set op body
            result = self._exec_set_operation(cte_def.body)
            name = cte_def.name.lower()
            columns = list(result.columns)
            if cte_def.column_aliases:
                if len(cte_def.column_aliases) != len(columns):
                    raise DatabaseError(
                        f"CTE '{name}' has {len(cte_def.column_aliases)} column aliases "
                        f"but query returns {len(columns)} columns"
                    )
                columns = list(cte_def.column_aliases)
            rows = []
            for row_vals in result.rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = row_vals[i] if i < len(row_vals) else None
                rows.append(row_dict)
            self._cte_tables[name] = rows
            self._cte_columns[name] = columns
        else:
            # Delegate to parent (handles UnionStmt for recursive, plain SelectStmt)
            self._materialize_cte(cte_def)

    # =========================================================================
    # Set Operation Logic
    # =========================================================================

    def _row_key(self, row):
        """Convert a row to a hashable key for deduplication."""
        return tuple(row)

    def _union(self, left_rows, right_rows, all_flag):
        """UNION: combine rows. Without ALL, remove duplicates."""
        if all_flag:
            return list(left_rows) + list(right_rows)
        # UNION without ALL: deduplicate
        seen = set()
        result = []
        for row in left_rows:
            key = self._row_key(row)
            if key not in seen:
                seen.add(key)
                result.append(row)
        for row in right_rows:
            key = self._row_key(row)
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    def _intersect(self, left_rows, right_rows, all_flag):
        """INTERSECT: rows appearing in both sets."""
        if all_flag:
            # INTERSECT ALL: for each row, min of counts from left and right
            from collections import Counter
            left_counts = Counter(self._row_key(r) for r in left_rows)
            right_counts = Counter(self._row_key(r) for r in right_rows)
            result = []
            # Preserve order from left side
            used = Counter()
            for row in left_rows:
                key = self._row_key(row)
                if used[key] < min(left_counts[key], right_counts.get(key, 0)):
                    result.append(row)
                    used[key] += 1
            return result
        else:
            # INTERSECT: distinct intersection
            right_keys = set(self._row_key(r) for r in right_rows)
            seen = set()
            result = []
            for row in left_rows:
                key = self._row_key(row)
                if key in right_keys and key not in seen:
                    seen.add(key)
                    result.append(row)
            return result

    def _except(self, left_rows, right_rows, all_flag):
        """EXCEPT: rows in left but not in right."""
        if all_flag:
            # EXCEPT ALL: for each row, left_count - right_count (non-negative)
            from collections import Counter
            right_counts = Counter(self._row_key(r) for r in right_rows)
            remaining = dict(right_counts)
            result = []
            for row in left_rows:
                key = self._row_key(row)
                if remaining.get(key, 0) > 0:
                    remaining[key] -= 1
                else:
                    result.append(row)
            return result
        else:
            # EXCEPT: distinct difference
            right_keys = set(self._row_key(r) for r in right_rows)
            seen = set()
            result = []
            for row in left_rows:
                key = self._row_key(row)
                if key not in right_keys and key not in seen:
                    seen.add(key)
                    result.append(row)
            return result

    # =========================================================================
    # Set Operation Sorting
    # =========================================================================

    def _sort_set_result(self, rows, columns, order_by):
        """Sort set operation result by ORDER BY clause."""
        import functools

        def resolve_expr(expr, row_vals):
            """Resolve an expression against a row (list of values with column names)."""
            if isinstance(expr, SqlColumnRef):
                col = expr.column
                if col in columns:
                    return row_vals[columns.index(col)]
                # Try case-insensitive
                for i, c in enumerate(columns):
                    if c.lower() == col.lower():
                        return row_vals[i]
                return None
            if isinstance(expr, SqlLiteral):
                # ORDER BY 1, 2 -- ordinal position
                if isinstance(expr.value, (int, float)):
                    idx = int(expr.value) - 1  # 1-based
                    if 0 <= idx < len(columns):
                        return row_vals[idx]
                return expr.value
            return None

        def compare(a, b):
            for expr, asc in order_by:
                a_val = resolve_expr(expr, a)
                b_val = resolve_expr(expr, b)
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
