"""
C258: B-Tree Indexed Database
Composes C116 (B+ Tree) + C247 (Mini Database)

Adds real B+ tree indexes to the SQL database engine:
- IndexManager: maintains B+ tree indexes on INSERT/UPDATE/DELETE
- QueryOptimizer: detects indexed WHERE conditions, uses index scans
- Composite indexes (multi-column)
- UNIQUE indexes with enforcement
- DROP INDEX support
- EXPLAIN shows index usage vs seq scan
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C116_bplus_tree')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))

from bplus_tree import BPlusTreeMap, BPlusTree, BulkLoader
from mini_database import (
    MiniDB, ResultSet, StorageEngine, QueryCompiler, Catalog, TableSchema,
    ColumnDef, CatalogError, CompileError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlAggCall, SqlStar, SelectExpr,
    TokenType, Lexer, Parser, parse_sql
)
from query_executor import (
    Database as QEDatabase, Table as QETable, Row, ExecutionEngine,
    Operator, SeqScanOp, FilterOp, ProjectOp, SortOp,
    HashAggregateOp, HavingOp, LimitOp, DistinctOp,
    HashJoinOp, NestedLoopJoinOp, SortMergeJoinOp,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr,
    CompOp, LogicOp, AggFunc, AggCall, IndexScanOp, TableIndex,
    UnionOp, SemiJoinOp, AntiJoinOp, TopNOp
)


# =============================================================================
# Index Manager -- maintains real B+ tree indexes
# =============================================================================

@dataclass
class IndexInfo:
    """Metadata for a B+ tree index."""
    name: str
    table_name: str
    columns: List[str]  # single or composite
    unique: bool = False
    tree: BPlusTreeMap = field(default_factory=lambda: BPlusTreeMap(order=32))

    @property
    def is_composite(self) -> bool:
        return len(self.columns) > 1


class IndexManager:
    """Manages B+ tree indexes for all tables."""

    def __init__(self):
        self.indexes: Dict[str, IndexInfo] = {}  # index_name -> IndexInfo
        self._table_indexes: Dict[str, List[str]] = {}  # table_name -> [index_names]

    def create_index(self, name: str, table_name: str, columns: List[str],
                     unique: bool = False) -> IndexInfo:
        """Create a new B+ tree index."""
        if name in self.indexes:
            raise CatalogError(f"Index '{name}' already exists")

        info = IndexInfo(
            name=name,
            table_name=table_name,
            columns=columns,
            unique=unique,
            tree=BPlusTreeMap(order=32),
        )
        self.indexes[name] = info

        if table_name not in self._table_indexes:
            self._table_indexes[table_name] = []
        self._table_indexes[table_name].append(name)

        return info

    def drop_index(self, name: str):
        """Remove an index."""
        if name not in self.indexes:
            raise CatalogError(f"Index '{name}' does not exist")
        info = self.indexes[name]
        self._table_indexes[info.table_name].remove(name)
        del self.indexes[name]

    def get_index(self, name: str) -> Optional[IndexInfo]:
        return self.indexes.get(name)

    def get_indexes_for_table(self, table_name: str) -> List[IndexInfo]:
        """All indexes on a given table."""
        names = self._table_indexes.get(table_name, [])
        return [self.indexes[n] for n in names]

    def find_index_for_column(self, table_name: str, column: str) -> Optional[IndexInfo]:
        """Find an index whose first column matches (covers equality/range on that col)."""
        for idx in self.get_indexes_for_table(table_name):
            if idx.columns[0] == column:
                return idx
        return None

    def find_index_for_columns(self, table_name: str, columns: List[str]) -> Optional[IndexInfo]:
        """Find a composite index that exactly matches the given columns."""
        for idx in self.get_indexes_for_table(table_name):
            if idx.columns == columns:
                return idx
        return None

    def _make_key(self, info: IndexInfo, row: Dict[str, Any]) -> Any:
        """Extract index key from a row dict."""
        if info.is_composite:
            return tuple(row.get(c) for c in info.columns)
        return row.get(info.columns[0])

    def build_index(self, info: IndexInfo, rows: List[Tuple[int, Dict[str, Any]]]):
        """Populate an index from existing table data."""
        info.tree = BPlusTreeMap(order=32)
        for rowid, row_data in rows:
            key = self._make_key(info, row_data)
            if key is None:
                continue  # skip NULL keys
            existing = info.tree.get(key)
            if existing is not None:
                if info.unique:
                    raise CatalogError(
                        f"UNIQUE index '{info.name}' violation: duplicate key {key!r}")
                existing.append(rowid)
            else:
                info.tree[key] = [rowid]

    def on_insert(self, table_name: str, rowid: int, row_data: Dict[str, Any]):
        """Update all indexes after an INSERT."""
        for idx in self.get_indexes_for_table(table_name):
            key = self._make_key(idx, row_data)
            if key is None:
                continue
            existing = idx.tree.get(key)
            if existing is not None:
                if idx.unique:
                    raise CatalogError(
                        f"UNIQUE index '{idx.name}' violation: duplicate key {key!r}")
                existing.append(rowid)
            else:
                idx.tree[key] = [rowid]

    def on_delete(self, table_name: str, rowid: int, row_data: Dict[str, Any]):
        """Update all indexes after a DELETE."""
        for idx in self.get_indexes_for_table(table_name):
            key = self._make_key(idx, row_data)
            if key is None:
                continue
            existing = idx.tree.get(key)
            if existing is not None:
                if rowid in existing:
                    existing.remove(rowid)
                if not existing:
                    del idx.tree[key]

    def on_update(self, table_name: str, rowid: int,
                  old_data: Dict[str, Any], new_data: Dict[str, Any]):
        """Update all indexes after an UPDATE."""
        for idx in self.get_indexes_for_table(table_name):
            old_key = self._make_key(idx, old_data)
            new_key = self._make_key(idx, new_data)

            if old_key == new_key:
                continue  # indexed columns didn't change

            # Remove old entry
            if old_key is not None:
                existing = idx.tree.get(old_key)
                if existing is not None:
                    if rowid in existing:
                        existing.remove(rowid)
                    if not existing:
                        del idx.tree[old_key]

            # Add new entry
            if new_key is not None:
                existing = idx.tree.get(new_key)
                if existing is not None:
                    if idx.unique:
                        raise CatalogError(
                            f"UNIQUE index '{idx.name}' violation: duplicate key {new_key!r}")
                    existing.append(rowid)
                else:
                    idx.tree[new_key] = [rowid]

    def lookup_eq(self, index_name: str, value: Any) -> List[int]:
        """Equality lookup -- returns matching rowids."""
        info = self.indexes.get(index_name)
        if not info:
            return []
        result = info.tree.get(value)
        return list(result) if result else []

    def lookup_range(self, index_name: str,
                     low: Any = None, high: Any = None,
                     low_inclusive: bool = True, high_inclusive: bool = True) -> List[int]:
        """Range lookup -- returns matching rowids in key order."""
        info = self.indexes.get(index_name)
        if not info:
            return []
        pairs = info.tree.range_query(
            low=low, high=high,
            include_low=low_inclusive, include_high=high_inclusive
        )
        rowids = []
        for _key, rids in pairs:
            rowids.extend(rids)
        return rowids

    def count(self, index_name: str) -> int:
        """Number of distinct keys in the index."""
        info = self.indexes.get(index_name)
        return len(info.tree) if info else 0

    def stats(self) -> Dict[str, Any]:
        result = {}
        for name, info in self.indexes.items():
            result[name] = {
                'table': info.table_name,
                'columns': info.columns,
                'unique': info.unique,
                'keys': len(info.tree),
                'height': info.tree.height(),
            }
        return result


# =============================================================================
# Query Optimizer -- decides when to use index scans
# =============================================================================

@dataclass
class IndexScanDecision:
    """Result of optimizer analysis."""
    use_index: bool = False
    index_info: Optional[IndexInfo] = None
    scan_type: str = 'seq'  # 'seq', 'eq', 'range'
    lookup_value: Any = None
    low: Any = None
    high: Any = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    remaining_filter: Any = None  # WHERE parts not covered by index


class QueryOptimizer:
    """Simple rule-based query optimizer that detects indexable WHERE conditions."""

    def __init__(self, index_manager: IndexManager):
        self.index_manager = index_manager

    def analyze_where(self, table_name: str, where_node) -> IndexScanDecision:
        """Analyze a WHERE clause and decide on index usage."""
        if where_node is None:
            return IndexScanDecision()

        # Try to find an indexable condition
        decision = self._try_single_condition(table_name, where_node)
        if decision.use_index:
            return decision

        # Try AND conditions: find one indexable part
        if isinstance(where_node, SqlLogic) and where_node.op == 'and' and len(where_node.operands) == 2:
            left_dec = self._try_single_condition(table_name, where_node.operands[0])
            if left_dec.use_index:
                left_dec.remaining_filter = where_node.operands[1]
                return left_dec
            right_dec = self._try_single_condition(table_name, where_node.operands[1])
            if right_dec.use_index:
                right_dec.remaining_filter = where_node.operands[0]
                return right_dec

        return IndexScanDecision()

    def _try_single_condition(self, table_name: str, node) -> IndexScanDecision:
        """Check if a single condition can use an index."""
        if not isinstance(node, SqlComparison):
            return IndexScanDecision()

        # Equality: col = value
        if node.op in ('=', '=='):
            col, val = self._extract_col_literal(table_name, node)
            if col:
                idx = self.index_manager.find_index_for_column(table_name, col)
                if idx:
                    return IndexScanDecision(
                        use_index=True, index_info=idx,
                        scan_type='eq', lookup_value=val
                    )

        # Range: col > value, col >= value, col < value, col <= value
        if node.op in ('<', '<=', '>', '>='):
            col, val = self._extract_col_literal(table_name, node)
            if col:
                idx = self.index_manager.find_index_for_column(table_name, col)
                if idx:
                    dec = IndexScanDecision(use_index=True, index_info=idx, scan_type='range')
                    if node.op in ('<', '<='):
                        dec.high = val
                        dec.high_inclusive = (node.op == '<=')
                    else:
                        dec.low = val
                        dec.low_inclusive = (node.op == '>=')
                    return dec

        # BETWEEN: col BETWEEN low AND high
        if node.op == 'between':
            col_name = self._get_column_name(node.left)
            if col_name:
                idx = self.index_manager.find_index_for_column(table_name, col_name)
                if idx and hasattr(node, 'right') and isinstance(node.right, (list, tuple)):
                    low_val = self._get_literal_value(node.right[0])
                    high_val = self._get_literal_value(node.right[1])
                    if low_val is not None and high_val is not None:
                        return IndexScanDecision(
                            use_index=True, index_info=idx, scan_type='range',
                            low=low_val, high=high_val,
                            low_inclusive=True, high_inclusive=True
                        )

        return IndexScanDecision()

    def _extract_col_literal(self, table_name: str, node: SqlComparison):
        """Extract (column_name, literal_value) from a comparison, or (None, None)."""
        # col op literal
        col = self._get_column_name(node.left)
        val = self._get_literal_value(node.right)
        if col and val is not None:
            return col, val
        # literal op col (reversed)
        col = self._get_column_name(node.right)
        val = self._get_literal_value(node.left)
        if col and val is not None:
            return col, val
        return None, None

    def _get_column_name(self, node) -> Optional[str]:
        if isinstance(node, SqlColumnRef):
            return node.column
        return None

    def _get_literal_value(self, node) -> Any:
        if isinstance(node, SqlLiteral):
            return node.value
        return None


# =============================================================================
# Indexed Database -- extends MiniDB with real indexes
# =============================================================================

class IndexedDB(MiniDB):
    """MiniDB extended with B+ tree indexes and query optimization."""

    def __init__(self, pool_size: int = 64):
        super().__init__(pool_size=pool_size)
        self.index_manager = IndexManager()
        self.optimizer = QueryOptimizer(self.index_manager)
        self._index_scan_count = 0
        self._seq_scan_count = 0

    def _exec_create_index(self, stmt: CreateIndexStmt) -> ResultSet:
        """Create a real B+ tree index, not just metadata."""
        schema = self.storage.catalog.get_table(stmt.table_name)

        # Determine columns (support composite: CREATE INDEX idx ON t (col1, col2))
        columns = [stmt.column] if isinstance(stmt.column, str) else list(stmt.column)

        # Validate columns exist
        col_names = schema.column_names()
        for c in columns:
            if c not in col_names:
                raise CatalogError(f"Column '{c}' does not exist in table '{stmt.table_name}'")

        # Detect UNIQUE from the statement (extend parser or use naming convention)
        unique = getattr(stmt, 'unique', False)

        # Create index structure
        info = self.index_manager.create_index(
            stmt.index_name, stmt.table_name, columns, unique=unique
        )

        # Populate from existing data
        txn_id = self._get_txn()
        try:
            rows = self.storage.scan_table(txn_id, stmt.table_name)
            self.index_manager.build_index(info, rows)

            # Also store in schema metadata for compatibility
            schema.indexes[stmt.index_name] = columns[0] if len(columns) == 1 else ','.join(columns)

            self._auto_commit(txn_id)
        except Exception:
            # Rollback index creation on failure
            self.index_manager.drop_index(stmt.index_name)
            self._auto_abort(txn_id)
            raise

        return ResultSet(columns=[], rows=[],
                         message=f"CREATE INDEX {stmt.index_name}")

    def _exec_drop_index(self, name: str) -> ResultSet:
        """Drop an index."""
        info = self.index_manager.get_index(name)
        if info:
            schema = self.storage.catalog.get_table(info.table_name)
            if name in schema.indexes:
                del schema.indexes[name]
        self.index_manager.drop_index(name)
        return ResultSet(columns=[], rows=[], message=f"DROP INDEX {name}")

    def _exec_insert(self, stmt: InsertStmt) -> ResultSet:
        """Insert with index maintenance."""
        schema = self.storage.catalog.get_table(stmt.table_name)
        indexes = self.index_manager.get_indexes_for_table(stmt.table_name)

        txn_id = self._get_txn()
        try:
            count = 0
            for values in stmt.values_list:
                row_data = {}
                cols = stmt.columns or schema.column_names()
                for i, col in enumerate(cols):
                    if i < len(values):
                        row_data[col] = self._eval_sql_value(values[i])
                    else:
                        row_data[col] = None

                rowid = self.storage.insert_row(txn_id, stmt.table_name, row_data)

                # Get the full row (with defaults applied) for indexing
                full_row = self.storage.get_row(txn_id, stmt.table_name, rowid)
                if full_row and indexes:
                    self.index_manager.on_insert(stmt.table_name, rowid, full_row)

                count += 1

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"INSERT {count}",
                             rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_update(self, stmt: UpdateStmt) -> ResultSet:
        """Update with index maintenance."""
        txn_id = self._get_txn()
        try:
            schema = self.storage.catalog.get_table(stmt.table_name)
            indexes = self.index_manager.get_indexes_for_table(stmt.table_name)
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0

            for rowid, row_data in all_rows:
                # Check WHERE
                if stmt.where is not None:
                    qe_row = Row(row_data)
                    qe_pred = self.compiler._sql_to_qe_expr(stmt.where)
                    from query_executor import eval_expr
                    if not eval_expr(qe_pred, qe_row):
                        continue

                # Compute updates
                updates = {}
                for col, val_expr in stmt.assignments:
                    if isinstance(val_expr, SqlLiteral):
                        updates[col] = val_expr.value
                    else:
                        qe_row = Row(row_data)
                        qe_expr = self.compiler._sql_to_qe_expr(val_expr)
                        from query_executor import eval_expr
                        updates[col] = eval_expr(qe_expr, qe_row)

                old_data = dict(row_data)
                self.storage.update_row(txn_id, stmt.table_name, rowid, updates)

                # Index maintenance
                if indexes:
                    new_data = self.storage.get_row(txn_id, stmt.table_name, rowid)
                    if new_data:
                        self.index_manager.on_update(
                            stmt.table_name, rowid, old_data, new_data
                        )

                count += 1

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"UPDATE {count}",
                             rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_delete(self, stmt: DeleteStmt) -> ResultSet:
        """Delete with index maintenance."""
        txn_id = self._get_txn()
        try:
            indexes = self.index_manager.get_indexes_for_table(stmt.table_name)
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0

            for rowid, row_data in all_rows:
                if stmt.where is not None:
                    qe_row = Row(row_data)
                    qe_pred = self.compiler._sql_to_qe_expr(stmt.where)
                    from query_executor import eval_expr
                    if not eval_expr(qe_pred, qe_row):
                        continue

                # Index maintenance before deletion
                if indexes:
                    self.index_manager.on_delete(stmt.table_name, rowid, row_data)

                self.storage.delete_row(txn_id, stmt.table_name, rowid)
                count += 1

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"DELETE {count}",
                             rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_select(self, stmt: SelectStmt) -> ResultSet:
        """Select with index-aware optimization."""
        txn_id = self._get_txn()
        try:
            # Check if we can use an index scan
            table_name = stmt.from_table.table_name if stmt.from_table else None
            decision = IndexScanDecision()

            if table_name and stmt.where and not stmt.joins:
                decision = self.optimizer.analyze_where(table_name, stmt.where)

            if decision.use_index:
                self._index_scan_count += 1
                result = self._exec_select_with_index(stmt, txn_id, decision)
            else:
                self._seq_scan_count += 1
                result = self._exec_select_seq(stmt, txn_id)

            self._auto_commit(txn_id)
            return result
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_select_seq(self, stmt: SelectStmt, txn_id: int) -> ResultSet:
        """Standard sequential scan path (delegates to parent)."""
        plan, engine = self.compiler.compile_select(stmt, txn_id)
        qe_rows = engine.execute(plan)
        return self._format_select_result(stmt, qe_rows)

    def _exec_select_with_index(self, stmt: SelectStmt, txn_id: int,
                                 decision: IndexScanDecision) -> ResultSet:
        """Index scan path: fetch only matching rows."""
        idx = decision.index_info
        table_name = stmt.from_table.table_name

        # Get matching rowids from index
        if decision.scan_type == 'eq':
            rowids = self.index_manager.lookup_eq(idx.name, decision.lookup_value)
        else:  # range
            rowids = self.index_manager.lookup_range(
                idx.name,
                low=decision.low, high=decision.high,
                low_inclusive=decision.low_inclusive,
                high_inclusive=decision.high_inclusive
            )

        # Fetch only the matching rows
        schema = self.storage.catalog.get_table(table_name)
        qe_db = QEDatabase()
        qe_table = qe_db.create_table(table_name, schema.column_names())

        for rid in rowids:
            row = self.storage.get_row(txn_id, table_name, rid)
            if row is not None:
                qe_table.insert(row)

        engine = ExecutionEngine(qe_db)

        # Build plan but use remaining filter instead of full WHERE
        modified_stmt = self._clone_select_with_filter(stmt, decision.remaining_filter)
        plan = self.compiler._build_plan(modified_stmt, qe_db, {})
        qe_rows = engine.execute(plan)

        return self._format_select_result(stmt, qe_rows)

    def _clone_select_with_filter(self, stmt: SelectStmt,
                                   remaining_filter) -> SelectStmt:
        """Clone a SelectStmt with a different WHERE clause."""
        new_stmt = SelectStmt(
            columns=stmt.columns,
            from_table=stmt.from_table,
            joins=stmt.joins,
            where=remaining_filter,
            group_by=stmt.group_by,
            having=stmt.having,
            order_by=stmt.order_by,
            limit=stmt.limit,
            offset=stmt.offset,
            distinct=stmt.distinct,
        )
        return new_stmt

    def _format_select_result(self, stmt: SelectStmt, qe_rows) -> ResultSet:
        """Convert QE rows to ResultSet (shared between seq and index paths)."""
        if qe_rows:
            if self.compiler._is_star_only(stmt.columns):
                ordered_cols = []
                if stmt.from_table:
                    schema = self.storage.catalog.get_table(stmt.from_table.table_name)
                    tname = stmt.from_table.table_name
                    for cn in schema.column_names():
                        ordered_cols.append((f"{tname}.{cn}", cn))
                for j in stmt.joins:
                    jschema = self.storage.catalog.get_table(j.table.table_name)
                    jtname = j.table.table_name
                    for cn in jschema.column_names():
                        ordered_cols.append((f"{jtname}.{cn}", cn))
                clean_cols = [c[1] for c in ordered_cols]
                qe_keys = [c[0] for c in ordered_cols]
                rows = []
                for r in qe_rows:
                    row_vals = [r.get(k) for k in qe_keys]
                    rows.append(row_vals)
            else:
                columns = qe_rows[0].columns()
                clean_cols = []
                for c in columns:
                    if '.' in c:
                        clean_cols.append(c.split('.')[-1])
                    else:
                        clean_cols.append(c)
                rows = [list(r.values()) for r in qe_rows]
        else:
            columns = []
            for se in stmt.columns:
                if isinstance(se.expr, SqlStar):
                    if stmt.from_table:
                        schema = self.storage.catalog.get_table(stmt.from_table.table_name)
                        columns.extend(schema.column_names())
                elif se.alias:
                    columns.append(se.alias)
                elif isinstance(se.expr, SqlColumnRef):
                    columns.append(se.expr.column)
                else:
                    columns.append(f"col_{len(columns)}")
            clean_cols = columns
            rows = []

        return ResultSet(columns=clean_cols, rows=rows)

    def _exec_explain(self, stmt) -> ResultSet:
        """Enhanced EXPLAIN that shows index usage."""
        inner = stmt.stmt if hasattr(stmt, 'stmt') else stmt
        if isinstance(inner, SelectStmt):
            table_name = inner.from_table.table_name if inner.from_table else None
            decision = IndexScanDecision()
            if table_name and inner.where and not inner.joins:
                decision = self.optimizer.analyze_where(table_name, inner.where)

            # Get base plan
            txn_id = self._get_txn()
            try:
                plan, engine = self.compiler.compile_select(inner, txn_id)
                base_explain = engine.explain(plan)
                self._auto_commit(txn_id)
            except Exception:
                self._auto_abort(txn_id)
                raise

            lines = []
            if decision.use_index:
                idx = decision.index_info
                lines.append(f"IndexScan using {idx.name} on {idx.table_name}")
                lines.append(f"  Index columns: {', '.join(idx.columns)}")
                lines.append(f"  Scan type: {decision.scan_type}")
                if decision.scan_type == 'eq':
                    lines.append(f"  Lookup value: {decision.lookup_value!r}")
                else:
                    if decision.low is not None:
                        op = '>=' if decision.low_inclusive else '>'
                        lines.append(f"  Low bound: {op} {decision.low!r}")
                    if decision.high is not None:
                        op = '<=' if decision.high_inclusive else '<'
                        lines.append(f"  High bound: {op} {decision.high!r}")
                if decision.remaining_filter:
                    lines.append(f"  Remaining filter: yes")
                lines.append(f"  Estimated rows: {self.index_manager.count(idx.name)} distinct keys")
            else:
                lines.append("SeqScan (no applicable index)")

            lines.append("")
            for line in base_explain.split('\n'):
                if line.strip():
                    lines.append(line)

            return ResultSet(columns=['plan'],
                             rows=[[line] for line in lines if line.strip()])
        return ResultSet(columns=['plan'], rows=[['EXPLAIN not supported']])

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with DROP INDEX support."""
        sql_stripped = sql.strip()

        # Handle DROP INDEX
        if sql_stripped.upper().startswith('DROP INDEX'):
            parts = sql_stripped.split()
            if len(parts) >= 3:
                idx_name = parts[2].rstrip(';')
                return self._exec_drop_index(idx_name)

        # Handle CREATE UNIQUE INDEX
        if sql_stripped.upper().startswith('CREATE UNIQUE INDEX'):
            return self._exec_create_unique_index(sql_stripped)

        return super().execute(sql)

    def _exec_create_unique_index(self, sql: str) -> ResultSet:
        """Parse and execute CREATE UNIQUE INDEX."""
        # CREATE UNIQUE INDEX name ON table (col1, col2, ...)
        import re
        m = re.match(
            r'CREATE\s+UNIQUE\s+INDEX\s+(\w+)\s+ON\s+(\w+)\s*\(([^)]+)\)',
            sql, re.IGNORECASE
        )
        if not m:
            raise CompileError(f"Invalid CREATE UNIQUE INDEX syntax: {sql}")

        idx_name = m.group(1)
        table_name = m.group(2)
        columns = [c.strip() for c in m.group(3).split(',')]

        schema = self.storage.catalog.get_table(table_name)
        col_names = schema.column_names()
        for c in columns:
            if c not in col_names:
                raise CatalogError(f"Column '{c}' does not exist in table '{table_name}'")

        info = self.index_manager.create_index(idx_name, table_name, columns, unique=True)

        txn_id = self._get_txn()
        try:
            rows = self.storage.scan_table(txn_id, table_name)
            self.index_manager.build_index(info, rows)
            schema.indexes[idx_name] = columns[0] if len(columns) == 1 else ','.join(columns)
            self._auto_commit(txn_id)
        except Exception:
            self.index_manager.drop_index(idx_name)
            self._auto_abort(txn_id)
            raise

        return ResultSet(columns=[], rows=[], message=f"CREATE UNIQUE INDEX {idx_name}")

    def execute_many(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        results = []
        # Split by semicolons (respecting strings)
        stmts = self._split_sql(sql)
        for s in stmts:
            s = s.strip()
            if s:
                results.append(self.execute(s))
        return results

    def _split_sql(self, sql: str) -> List[str]:
        """Split SQL by semicolons, respecting quoted strings."""
        stmts = []
        current = []
        in_string = False
        quote_char = None
        for ch in sql:
            if in_string:
                current.append(ch)
                if ch == quote_char:
                    in_string = False
            elif ch in ("'", '"'):
                in_string = True
                quote_char = ch
                current.append(ch)
            elif ch == ';':
                stmts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            stmts.append(''.join(current))
        return stmts

    def index_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            'indexes': self.index_manager.stats(),
            'index_scans': self._index_scan_count,
            'seq_scans': self._seq_scan_count,
        }
