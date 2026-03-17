"""
C257: Table Aliases Fix
Extends C256 (Derived Tables)

Fixes the long-standing bug where table aliases in JOINs don't resolve column values.
Before: FROM t1 x JOIN t2 y ON x.a = y.b -> NULLs (alias not resolved)
After:  FROM t1 x JOIN t2 y ON x.a = y.b -> correct values

Root cause: C247's QueryCompiler._sql_to_qe_expr passes alias names to ColumnRef
without resolving them to real table names. The QE engine stores rows under real
table names, so alias-qualified lookups fail.

Fix strategy: Detect queries with table aliases and route them through the
subquery evaluation path (_exec_select_full_subquery) which already resolves
aliases correctly via _get_regular_source_rows (stores rows under both
alias.col and table_name.col keys).

Also adds alias support to:
- SELECT * output column ordering (uses alias-qualified names)
- SELECT alias.col projections
- WHERE, GROUP BY, HAVING, ORDER BY with alias-qualified refs
- Multi-table JOINs with mixed aliases
- Self-joins using aliases
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import chain -- use abspath to avoid nested relative path resolution issues
_challenges = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _dep in (
    'C256_derived_tables', 'C255_subqueries', 'C254_set_operations',
    'C253_common_table_expressions', 'C252_sql_window_functions',
    'C251_sql_triggers', 'C250_sql_views', 'C249_stored_procedures',
    'C247_mini_database', 'C246_transaction_manager',
    'C245_query_executor', 'C244_buffer_pool',
    'C242_lock_manager', 'C241_wal', 'C240_mvcc',
):
    _p = os.path.join(_challenges, _dep)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from derived_tables import (
    DerivedTableDB, DerivedTableLexer, DerivedTableParser,
    DerivedTable,
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
# Database Engine with Table Alias Fix
# =============================================================================

class TableAliasDB(DerivedTableDB):
    """Database with proper table alias resolution in all query paths."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)

    # -----------------------------------------------------------------
    # Parsing -- reuse DerivedTable parser
    # -----------------------------------------------------------------

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with table alias support."""
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
    # Alias detection
    # -----------------------------------------------------------------

    def _has_table_aliases(self, stmt) -> bool:
        """Check if statement uses table aliases."""
        if not isinstance(stmt, SelectStmt):
            return False
        if stmt.from_table and not isinstance(stmt.from_table, DerivedTable):
            if stmt.from_table.alias:
                return True
        for join in getattr(stmt, 'joins', []):
            if not isinstance(join.table, DerivedTable) and join.table.alias:
                return True
        return False

    # -----------------------------------------------------------------
    # Override SELECT routing to intercept aliased queries
    # -----------------------------------------------------------------

    def _exec_select_with_cte_context(self, stmt) -> ResultSet:
        """Override: route aliased queries through full subquery path."""
        if self._has_table_aliases(stmt):
            return self._exec_select_full_subquery(stmt)
        return super()._exec_select_with_cte_context(stmt)

    def _exec_select_standard(self, stmt: SelectStmt) -> ResultSet:
        """Override: route aliased queries through full subquery path."""
        if self._has_table_aliases(stmt):
            return self._exec_select_full_subquery(stmt)
        return super()._exec_select_standard(stmt)

    def _exec_select_from_cte(self, stmt: SelectStmt) -> ResultSet:
        """Override: route aliased queries through full subquery path."""
        if self._has_table_aliases(stmt):
            return self._exec_select_full_subquery(stmt)
        return super()._exec_select_from_cte(stmt)

    # -----------------------------------------------------------------
    # Override _get_source_rows to ensure alias keys exist
    # -----------------------------------------------------------------

    def _get_source_rows(self, stmt: SelectStmt) -> List[Dict[str, Any]]:
        """Get source rows with alias-qualified column keys."""
        if stmt.from_table is None:
            return [{}]

        if isinstance(stmt.from_table, DerivedTable):
            rows = self._resolve_derived_table(stmt.from_table)
        else:
            rows = self._get_regular_source_rows_aliased(stmt)

        # Process JOINs
        for join in stmt.joins:
            rows = self._do_join_aliased(rows, join, stmt)

        return rows

    def _get_regular_source_rows_aliased(self, stmt: SelectStmt) -> List[Dict[str, Any]]:
        """Get source rows from a regular table with full alias support."""
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
                    if alias != table_name:
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
                        if alias != table_name:
                            row[f"{table_name}.{cn}"] = val
                    rows.append(row)
                self._auto_commit(txn_id)
            except Exception:
                self._auto_abort(txn_id)
                raise

        return rows

    def _do_join_aliased(self, left_rows: List[Dict], join: JoinClause,
                         stmt: SelectStmt) -> List[Dict]:
        """Execute a JOIN with full alias support."""
        if isinstance(join.table, DerivedTable):
            right_rows = self._resolve_derived_table(join.table)
        else:
            right_table = join.table.table_name
            right_alias = join.table.alias or right_table
            right_rows = self._get_table_rows_aliased(right_table, right_alias)

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
                # For LEFT JOIN with no match, add NULLs for right side
                combined = dict(lrow)
                if isinstance(join.table, DerivedTable):
                    pass  # derived table columns not available
                else:
                    right_table = join.table.table_name
                    right_alias = join.table.alias or right_table
                    try:
                        schema = self.storage.catalog.get_table(right_table)
                        for cn in schema.column_names():
                            combined[cn] = None
                            combined[f"{right_alias}.{cn}"] = None
                            if right_alias != right_table:
                                combined[f"{right_table}.{cn}"] = None
                    except Exception:
                        pass
                result.append(combined)

        return result

    def _get_table_rows_aliased(self, table_name: str, alias: str) -> List[Dict]:
        """Get rows from a regular table or CTE with full alias support."""
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
                    if alias != table_name:
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
                        if alias != table_name:
                            row[f"{table_name}.{cn}"] = val
                    rows.append(row)
                self._auto_commit(txn_id)
            except Exception:
                self._auto_abort(txn_id)
                raise
        return rows

    # -----------------------------------------------------------------
    # Fix 1: SELECT * with aliases -- use qualified names for ordering
    # -----------------------------------------------------------------

    def _project_ungrouped(self, rows, stmt) -> Tuple[List[List], List[str]]:
        """Override: handle SELECT * with aliased multi-table joins properly."""
        result_cols = []
        col_keys = []  # Keys to use for extracting values from row dicts
        is_star = False

        for i, col in enumerate(stmt.columns):
            if isinstance(col.expr, SqlStar):
                is_star = True
                # Build ordered column list from schema using aliases
                if stmt.from_table and not isinstance(stmt.from_table, DerivedTable):
                    table_name = stmt.from_table.table_name
                    alias = stmt.from_table.alias or table_name
                    try:
                        schema = self.storage.catalog.get_table(table_name)
                        for cn in schema.column_names():
                            result_cols.append(cn)
                            col_keys.append(f"{alias}.{cn}")
                    except Exception:
                        pass
                elif isinstance(stmt.from_table, DerivedTable):
                    # Derived table -- use row keys
                    if rows:
                        seen = set()
                        for k in rows[0]:
                            if '.' not in k and k not in seen:
                                result_cols.append(k)
                                col_keys.append(k)
                                seen.add(k)

                for join in stmt.joins:
                    if isinstance(join.table, DerivedTable):
                        if rows:
                            dt_alias = join.table.alias
                            for k in rows[0]:
                                if k.startswith(f"{dt_alias}."):
                                    bare = k.split('.', 1)[1]
                                    if bare not in result_cols:
                                        result_cols.append(bare)
                                        col_keys.append(k)
                    else:
                        jt_name = join.table.table_name
                        jt_alias = join.table.alias or jt_name
                        try:
                            jschema = self.storage.catalog.get_table(jt_name)
                            for cn in jschema.column_names():
                                result_cols.append(cn)
                                col_keys.append(f"{jt_alias}.{cn}")
                        except Exception:
                            pass

                if not col_keys and rows:
                    # Fallback
                    seen = set()
                    for k in rows[0]:
                        if '.' not in k and k not in seen:
                            result_cols.append(k)
                            col_keys.append(k)
                            seen.add(k)
            elif col.alias:
                result_cols.append(col.alias)
            elif isinstance(col.expr, SqlColumnRef):
                result_cols.append(col.expr.column)
            else:
                result_cols.append(f"col_{i}")

        result_rows = []
        for row in rows:
            if is_star:
                vals = [row.get(k) for k in col_keys]
            else:
                vals = []
                for col in stmt.columns:
                    vals.append(self._eval_select_expr(col.expr, row, [row]))
            result_rows.append(vals)

        return result_rows, result_cols

    # -----------------------------------------------------------------
    # Fix 2: Override _exec_select_full_subquery for pre-projection sorting
    # -----------------------------------------------------------------

    def _exec_select_full_subquery(self, stmt: SelectStmt) -> ResultSet:
        """Override: sort using source row dicts (with alias-qualified keys)
        before projection to handle ambiguous column names like d.name vs e.name."""
        source_rows = self._get_source_rows(stmt)

        # Filter with WHERE
        if stmt.where:
            filtered = []
            for row in source_rows:
                val = self._eval_subquery_expr(stmt.where, row)
                if val:
                    filtered.append(row)
            source_rows = filtered

        # GROUP BY
        if stmt.group_by:
            groups = self._do_groupby(source_rows, stmt.group_by, stmt)
            if stmt.having:
                groups = {k: v for k, v in groups.items()
                          if self._eval_having_subquery(stmt.having, k, v, stmt)}
            result_rows, result_cols = self._project_grouped(groups, stmt)

            # For grouped results, sort after projection (no ambiguity)
            if stmt.order_by:
                result_rows = self._sort_rows_basic(result_rows, result_cols, stmt.order_by)
        else:
            has_aggs = any(self._expr_has_agg(col.expr) for col in stmt.columns)
            if has_aggs:
                groups = {(): source_rows}
                if stmt.having:
                    groups = {k: v for k, v in groups.items()
                              if self._eval_having_subquery(stmt.having, k, v, stmt)}
                result_rows, result_cols = self._project_grouped(groups, stmt)
                if stmt.order_by:
                    result_rows = self._sort_rows_basic(result_rows, result_cols, stmt.order_by)
            else:
                # Sort source rows BEFORE projection using dict keys
                if stmt.order_by:
                    source_rows = self._sort_source_rows(source_rows, stmt.order_by)
                result_rows, result_cols = self._project_ungrouped(source_rows, stmt)

        # DISTINCT
        if stmt.distinct:
            seen = set()
            deduped = []
            for row in result_rows:
                key = tuple(row)
                if key not in seen:
                    seen.add(key)
                    deduped.append(row)
            result_rows = deduped

        # OFFSET
        if stmt.offset:
            result_rows = result_rows[stmt.offset:]

        # LIMIT
        if stmt.limit is not None:
            result_rows = result_rows[:stmt.limit]

        return ResultSet(columns=result_cols, rows=result_rows)

    def _sort_source_rows(self, rows: List[Dict], order_by) -> List[Dict]:
        """Sort source row dicts using alias-qualified column keys."""
        import functools

        def resolve(expr, row_dict):
            if isinstance(expr, SqlColumnRef):
                col = expr.column
                tbl = expr.table
                if tbl:
                    key = f"{tbl}.{col}"
                    if key in row_dict:
                        return row_dict[key]
                if col in row_dict:
                    return row_dict[col]
                # Fallback: find any key ending with .col
                for k, v in row_dict.items():
                    if '.' in k and k.split('.', 1)[1] == col:
                        return v
                return None
            if isinstance(expr, SqlLiteral):
                return expr.value
            return None

        def compare(a, b):
            for expr, asc in order_by:
                va = resolve(expr, a)
                vb = resolve(expr, b)
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

    def _sort_rows_basic(self, rows, columns, order_by):
        """Sort projected result rows (basic column name matching)."""
        import functools

        def resolve(expr, row_vals):
            if isinstance(expr, SqlColumnRef):
                col = expr.column
                if col in columns:
                    return row_vals[columns.index(col)]
                for i, c in enumerate(columns):
                    if c.lower() == col.lower():
                        return row_vals[i]
                return None
            if isinstance(expr, SqlLiteral):
                if isinstance(expr.value, (int, float)):
                    idx = int(expr.value) - 1
                    if 0 <= idx < len(columns):
                        return row_vals[idx]
                return expr.value
            return None

        def compare(a, b):
            for expr, asc in order_by:
                va = resolve(expr, a)
                vb = resolve(expr, b)
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

    # -----------------------------------------------------------------
    # Fix 3: HAVING with aggregate alias resolution
    # -----------------------------------------------------------------

    def _eval_having_subquery(self, having, group_key, group_rows, stmt) -> bool:
        """Override: resolve aggregate aliases (e.g. HAVING cnt >= 2)."""
        # Build context with group key values
        ctx = {}
        if stmt.group_by:
            for i, expr in enumerate(stmt.group_by):
                if isinstance(expr, SqlColumnRef):
                    ctx[expr.column] = group_key[i]
                    if expr.table:
                        ctx[f"{expr.table}.{expr.column}"] = group_key[i]

        # Also add aggregate aliases to context
        for col in stmt.columns:
            if isinstance(col.expr, SqlAggCall) and col.alias:
                ctx[col.alias] = self._compute_agg(col.expr, group_rows)

        return self._eval_having_expr(having, ctx, group_rows)
