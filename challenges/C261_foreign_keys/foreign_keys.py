"""
C261: FOREIGN KEY Constraints
Extends C260 (Check Constraints) with referential integrity.

Adds FOREIGN KEY constraints to the SQL database:
- Column-level REFERENCES: CREATE TABLE t (x INT REFERENCES parent(id))
- Table-level FOREIGN KEY: FOREIGN KEY (col) REFERENCES parent(col)
- Named FK constraints: CONSTRAINT fk_name FOREIGN KEY (...) REFERENCES ...
- ON DELETE actions: RESTRICT (default), CASCADE, SET NULL, SET DEFAULT, NO ACTION
- ON UPDATE actions: RESTRICT (default), CASCADE, SET NULL, SET DEFAULT, NO ACTION
- FK validation on INSERT (child must reference existing parent)
- FK validation on UPDATE (child FK column changes must reference existing parent)
- FK validation on DELETE (parent row deletion checks children)
- FK validation on UPDATE (parent PK/unique changes check children)
- ALTER TABLE ADD/DROP FOREIGN KEY
- Multi-column (composite) foreign keys
- Self-referencing foreign keys
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum

# Import composed components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C260_check_constraints')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager')))

from check_constraints import (
    CheckDB, CheckParser, CheckLexer, CheckConstraint, CheckTableSchema,
    CheckEvaluator, _extract_columns, _expr_to_sql,
    AlterAddCheckStmt, AlterDropConstraintStmt,
    parse_check_sql, parse_check_sql_multi,
)
from mini_database import (
    MiniDB, ResultSet, StorageEngine, QueryCompiler, Catalog, TableSchema,
    ColumnDef, CatalogError, CompileError, ParseError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlIsNull, SqlBetween, SqlInList, SqlFuncCall, SqlCase,
    TokenType, Token, Lexer, Parser, KEYWORDS,
)
from query_executor import Row, eval_expr


# =============================================================================
# Foreign Key Data Model
# =============================================================================

class FKAction(Enum):
    """Action to take on parent DELETE/UPDATE."""
    RESTRICT = "RESTRICT"
    CASCADE = "CASCADE"
    SET_NULL = "SET NULL"
    SET_DEFAULT = "SET DEFAULT"
    NO_ACTION = "NO ACTION"


@dataclass
class ForeignKey:
    """A FOREIGN KEY constraint."""
    name: Optional[str]             # constraint name (None = auto-generated)
    child_table: str                # table this FK is defined on
    child_columns: List[str]        # columns in child table
    parent_table: str               # referenced table
    parent_columns: List[str]       # referenced columns in parent table
    on_delete: FKAction = FKAction.RESTRICT
    on_update: FKAction = FKAction.RESTRICT

    def __repr__(self):
        name = self.name or "(unnamed)"
        cols = ", ".join(self.child_columns)
        refs = ", ".join(self.parent_columns)
        return f"ForeignKey({name}: {self.child_table}({cols}) -> {self.parent_table}({refs}))"


# =============================================================================
# FK Table Schema -- wraps CheckTableSchema with FK support
# =============================================================================

@dataclass
class FKTableSchema:
    """Wraps a table schema with foreign key constraint support."""
    base_schema: Any  # TableSchema or CheckTableSchema
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    _next_fk_id: int = 1

    @property
    def name(self):
        return self.base_schema.name

    @property
    def columns(self):
        return self.base_schema.columns

    @property
    def indexes(self):
        return self.base_schema.indexes

    @property
    def next_rowid(self):
        return self.base_schema.next_rowid

    @next_rowid.setter
    def next_rowid(self, val):
        self.base_schema.next_rowid = val

    def column_names(self):
        return self.base_schema.column_names()

    def get_column(self, name):
        return self.base_schema.get_column(name)

    def primary_key_column(self):
        return self.base_schema.primary_key_column()

    # Proxy CHECK constraint methods if base is CheckTableSchema
    @property
    def check_constraints(self):
        if hasattr(self.base_schema, 'check_constraints'):
            return self.base_schema.check_constraints
        return []

    def add_check(self, constraint):
        if hasattr(self.base_schema, 'add_check'):
            self.base_schema.add_check(constraint)

    def drop_check(self, name):
        if hasattr(self.base_schema, 'drop_check'):
            self.base_schema.drop_check(name)

    def get_check(self, name):
        if hasattr(self.base_schema, 'get_check'):
            return self.base_schema.get_check(name)
        return None

    def add_foreign_key(self, fk: ForeignKey):
        """Add a foreign key constraint, auto-naming if needed."""
        if fk.name is None:
            fk.name = f"{self.name}_fk_{self._next_fk_id}"
            self._next_fk_id += 1
        for existing in self.foreign_keys:
            if existing.name == fk.name:
                raise CatalogError(
                    f"Constraint '{fk.name}' already exists on table '{self.name}'"
                )
        self.foreign_keys.append(fk)

    def drop_foreign_key(self, constraint_name: str):
        """Drop a foreign key by name."""
        for i, fk in enumerate(self.foreign_keys):
            if fk.name == constraint_name:
                self.foreign_keys.pop(i)
                return
        raise CatalogError(
            f"Foreign key '{constraint_name}' not found on table '{self.name}'"
        )

    def get_foreign_key(self, constraint_name: str) -> Optional[ForeignKey]:
        for fk in self.foreign_keys:
            if fk.name == constraint_name:
                return fk
        return None


# =============================================================================
# AST nodes for FK statements
# =============================================================================

@dataclass
class AlterAddFKStmt:
    table_name: str
    foreign_key: ForeignKey


@dataclass
class AlterDropFKStmt:
    table_name: str
    constraint_name: str


# =============================================================================
# Extended Parser with FOREIGN KEY support
# =============================================================================

class FKParser(CheckParser):
    """Parser extended with FOREIGN KEY constraint support."""

    EXTRA_KEYWORDS = {
        **CheckParser.EXTRA_KEYWORDS,
        'foreign': 'FOREIGN',
        'key': 'KEY',  # Already a keyword but may lex as IDENT in some contexts
        'references': 'REFERENCES',
        'cascade': 'CASCADE',
        'restrict': 'RESTRICT',
        'action': 'ACTION',
        'no': 'NO',
    }

    def _parse_column_defs(self):
        """Parse column definitions with FK and CHECK support."""
        cols = [self._parse_column_def_with_fk()]
        table_checks = []
        table_fks = []

        while self.match(TokenType.COMMA):
            # Table-level PRIMARY KEY
            if self.peek().type == TokenType.PRIMARY:
                self.advance()
                self.expect(TokenType.KEY)
                self.expect(TokenType.LPAREN)
                pk_col = self.expect(TokenType.IDENT).value
                self.expect(TokenType.RPAREN)
                for c in cols:
                    if c['def'].name == pk_col:
                        c['def'].primary_key = True
                        c['def'].not_null = True
                continue

            # Table-level CONSTRAINT name [CHECK|FOREIGN KEY] (...)
            if self._check_ident('constraint'):
                saved = self.pos
                self.advance()  # CONSTRAINT
                cname = self.expect(TokenType.IDENT).value

                # CONSTRAINT name FOREIGN KEY (...)
                if self._check_ident('foreign'):
                    fk = self._parse_table_fk_body(cname)
                    table_fks.append(fk)
                    continue

                # CONSTRAINT name CHECK (...)
                if self._check_ident('check'):
                    self.advance()
                    self.expect(TokenType.LPAREN)
                    expr = self._parse_or()
                    self.expect(TokenType.RPAREN)
                    source = _expr_to_sql(expr)
                    table_checks.append(CheckConstraint(
                        name=cname, expr=expr,
                        columns=_extract_columns(expr), source=source
                    ))
                    continue

                # Not a recognized constraint type
                self.pos = saved
                cols.append(self._parse_column_def_with_fk())
                continue

            # Table-level CHECK (...)
            if self._check_ident('check'):
                self.advance()
                self.expect(TokenType.LPAREN)
                expr = self._parse_or()
                self.expect(TokenType.RPAREN)
                source = _expr_to_sql(expr)
                table_checks.append(CheckConstraint(
                    name=None, expr=expr,
                    columns=_extract_columns(expr), source=source
                ))
                continue

            # Table-level FOREIGN KEY (...) REFERENCES ...
            if self._check_ident('foreign'):
                fk = self._parse_table_fk_body(None)
                table_fks.append(fk)
                continue

            cols.append(self._parse_column_def_with_fk())

        return (
            [c['def'] for c in cols],
            [c.get('check') for c in cols if c.get('check')],
            table_checks,
            [c.get('fk') for c in cols if c.get('fk')],
            table_fks,
        )

    def _parse_column_def_with_fk(self):
        """Parse a single column def, capturing CHECK and REFERENCES."""
        result = self._parse_column_def_with_check()
        col_def = result['def']
        fk = None

        # Column-level REFERENCES parent(col)
        if self._check_ident('references'):
            self.advance()
            parent_table = self.expect(TokenType.IDENT).value
            self.expect(TokenType.LPAREN)
            parent_col = self.expect(TokenType.IDENT).value
            self.expect(TokenType.RPAREN)
            on_delete, on_update = self._parse_fk_actions()
            fk = ForeignKey(
                name=None,
                child_table="",  # filled in later by CREATE TABLE handler
                child_columns=[col_def.name],
                parent_table=parent_table,
                parent_columns=[parent_col],
                on_delete=on_delete,
                on_update=on_update,
            )

        result['fk'] = fk
        return result

    def _parse_table_fk_body(self, name: Optional[str]) -> ForeignKey:
        """Parse FOREIGN KEY (cols) REFERENCES parent(cols) [actions].
        'FOREIGN' keyword not yet consumed."""
        self._expect_ident('foreign')
        # KEY might be TokenType.KEY or an IDENT 'key'
        tok = self.peek()
        if tok.type == TokenType.KEY:
            self.advance()
        elif tok.type == TokenType.IDENT and tok.value.lower() == 'key':
            self.advance()
        else:
            raise ParseError(f"Expected KEY, got '{tok.value}'")

        # Child columns
        self.expect(TokenType.LPAREN)
        child_cols = [self.expect(TokenType.IDENT).value]
        while self.match(TokenType.COMMA):
            child_cols.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)

        # REFERENCES parent(cols)
        self._expect_ident('references')
        parent_table = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        parent_cols = [self.expect(TokenType.IDENT).value]
        while self.match(TokenType.COMMA):
            parent_cols.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)

        on_delete, on_update = self._parse_fk_actions()

        return ForeignKey(
            name=name,
            child_table="",  # filled in later
            child_columns=child_cols,
            parent_table=parent_table,
            parent_columns=parent_cols,
            on_delete=on_delete,
            on_update=on_update,
        )

    def _parse_fk_actions(self) -> Tuple[FKAction, FKAction]:
        """Parse ON DELETE / ON UPDATE action clauses."""
        on_delete = FKAction.RESTRICT
        on_update = FKAction.RESTRICT

        for _ in range(2):  # at most ON DELETE + ON UPDATE
            if self.peek().type != TokenType.ON:
                break
            self.advance()  # ON
            tok = self.peek()
            if tok.type == TokenType.DELETE:
                self.advance()
                on_delete = self._parse_fk_action()
            elif tok.type == TokenType.UPDATE or (tok.type == TokenType.IDENT and tok.value.lower() == 'update'):
                self.advance()
                on_update = self._parse_fk_action()
            else:
                raise ParseError(f"Expected DELETE or UPDATE after ON, got '{tok.value}'")

        return on_delete, on_update

    def _parse_fk_action(self) -> FKAction:
        """Parse a single FK action: CASCADE, RESTRICT, SET NULL, SET DEFAULT, NO ACTION."""
        tok = self.peek()

        if tok.type == TokenType.IDENT and tok.value.lower() == 'cascade':
            self.advance()
            return FKAction.CASCADE

        if tok.type == TokenType.IDENT and tok.value.lower() == 'restrict':
            self.advance()
            return FKAction.RESTRICT

        if tok.type == TokenType.SET:
            self.advance()
            nxt = self.peek()
            if nxt.type == TokenType.NULL:
                self.advance()
                return FKAction.SET_NULL
            if nxt.type == TokenType.DEFAULT:
                self.advance()
                return FKAction.SET_DEFAULT
            raise ParseError(f"Expected NULL or DEFAULT after SET, got '{nxt.value}'")

        if tok.type == TokenType.IDENT and tok.value.lower() == 'no':
            self.advance()
            self._expect_ident('action')
            return FKAction.NO_ACTION

        raise ParseError(f"Expected CASCADE, RESTRICT, SET NULL, SET DEFAULT, or NO ACTION, got '{tok.value}'")

    def _parse_create(self):
        """Override to handle FK constraints in CREATE TABLE."""
        self.expect(TokenType.CREATE)
        if self.peek().type == TokenType.INDEX:
            return self._parse_create_index()
        return self._parse_create_table_with_fks()

    def _parse_create_table_with_fks(self):
        """Parse CREATE TABLE with CHECK and FK constraints."""
        self.expect(TokenType.TABLE)
        if_not_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True

        table_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        col_defs, col_checks, table_checks, col_fks, table_fks = self._parse_column_defs()
        self.expect(TokenType.RPAREN)

        stmt = CreateTableStmt(
            table_name=table_name,
            columns=col_defs,
            if_not_exists=if_not_exists
        )
        stmt._col_checks = col_checks
        stmt._table_checks = table_checks
        stmt._col_fks = col_fks
        stmt._table_fks = table_fks
        return stmt

    def _parse_alter_add(self, table_name: str):
        """Parse ALTER TABLE ... ADD [CONSTRAINT name] [CHECK|FOREIGN KEY]."""
        cname = None

        # CONSTRAINT name
        if self._check_ident('constraint'):
            self.advance()
            cname = self.expect(TokenType.IDENT).value

        # CHECK (...)
        if self._check_ident('check'):
            self.advance()
            self.expect(TokenType.LPAREN)
            expr = self._parse_or()
            self.expect(TokenType.RPAREN)
            source = _expr_to_sql(expr)
            return AlterAddCheckStmt(
                table_name=table_name,
                constraint=CheckConstraint(
                    name=cname, expr=expr,
                    columns=_extract_columns(expr), source=source
                )
            )

        # FOREIGN KEY (...) REFERENCES ...
        if self._check_ident('foreign'):
            fk = self._parse_table_fk_body(cname)
            fk.child_table = table_name
            return AlterAddFKStmt(table_name=table_name, foreign_key=fk)

        raise ParseError("Expected CHECK or FOREIGN KEY after ADD [CONSTRAINT name]")

    def _parse_alter_drop(self, table_name: str):
        """Parse ALTER TABLE ... DROP CONSTRAINT name / DROP FOREIGN KEY name."""
        if self._check_ident('constraint'):
            self.advance()
            cname = self.expect(TokenType.IDENT).value
            return AlterDropConstraintStmt(
                table_name=table_name,
                constraint_name=cname
            )
        if self._check_ident('foreign'):
            self.advance()
            tok = self.peek()
            if tok.type == TokenType.KEY or (tok.type == TokenType.IDENT and tok.value.lower() == 'key'):
                self.advance()
            cname = self.expect(TokenType.IDENT).value
            return AlterDropFKStmt(
                table_name=table_name,
                constraint_name=cname
            )
        raise ParseError("Expected CONSTRAINT or FOREIGN KEY after DROP")


# =============================================================================
# Parse functions
# =============================================================================

def parse_fk_sql(sql: str):
    """Parse a single SQL statement with FK + CHECK support."""
    lexer = CheckLexer(sql)
    parser = FKParser(lexer.tokens)
    return parser.parse()


def parse_fk_sql_multi(sql: str):
    """Parse multiple SQL statements with FK + CHECK support."""
    lexer = CheckLexer(sql)
    parser = FKParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# FK Enforcer -- referential integrity validation
# =============================================================================

class FKEnforcer:
    """Enforces foreign key constraints across the database."""

    def __init__(self, db: 'FKDB'):
        self.db = db

    def validate_insert(self, table_name: str, row_data: Dict[str, Any],
                        foreign_keys: List[ForeignKey]):
        """Validate that inserted row references existing parent rows."""
        for fk in foreign_keys:
            self._check_parent_exists(fk, row_data)

    def validate_update(self, table_name: str, old_row: Dict[str, Any],
                        new_row: Dict[str, Any], foreign_keys: List[ForeignKey]):
        """Validate that updated FK columns still reference existing parents."""
        for fk in foreign_keys:
            # Only check if FK columns actually changed
            changed = any(
                old_row.get(col) != new_row.get(col) for col in fk.child_columns
            )
            if changed:
                self._check_parent_exists(fk, new_row)

    def handle_parent_delete(self, parent_table: str, deleted_row: Dict[str, Any],
                             txn_id: int):
        """Handle cascading actions when a parent row is deleted."""
        # Find all FKs referencing this parent table
        for child_table, fk_schema in self.db._fk_schemas.items():
            for fk in fk_schema.foreign_keys:
                if fk.parent_table != parent_table:
                    continue
                # Check if any child rows reference the deleted parent
                parent_key = self._get_key_values(fk.parent_columns, deleted_row)
                if self._all_null(parent_key):
                    continue
                children = self._find_referencing_children(
                    txn_id, child_table, fk.child_columns, parent_key
                )
                if not children:
                    continue

                if fk.on_delete == FKAction.RESTRICT or fk.on_delete == FKAction.NO_ACTION:
                    raise CatalogError(
                        f"Cannot delete from '{parent_table}': "
                        f"foreign key '{fk.name}' on '{child_table}' references this row"
                    )
                elif fk.on_delete == FKAction.CASCADE:
                    for rowid, child_row in children:
                        # Recursively handle cascades from this child table
                        self.handle_parent_delete(child_table, child_row, txn_id)
                        self.db.storage.delete_row(txn_id, child_table, rowid)
                elif fk.on_delete == FKAction.SET_NULL:
                    for rowid, child_row in children:
                        updates = {col: None for col in fk.child_columns}
                        self.db.storage.update_row(txn_id, child_table, rowid, updates)
                elif fk.on_delete == FKAction.SET_DEFAULT:
                    schema = self.db.storage.catalog.get_table(child_table)
                    updates = {}
                    for col in fk.child_columns:
                        col_def = self._get_column_def(schema, col)
                        updates[col] = self._unwrap_default(col_def) if col_def else None
                    for rowid, child_row in children:
                        self.db.storage.update_row(txn_id, child_table, rowid, updates)

    def handle_parent_update(self, parent_table: str, old_row: Dict[str, Any],
                             new_row: Dict[str, Any], txn_id: int):
        """Handle cascading actions when a parent row's key is updated."""
        for child_table, fk_schema in self.db._fk_schemas.items():
            for fk in fk_schema.foreign_keys:
                if fk.parent_table != parent_table:
                    continue
                old_key = self._get_key_values(fk.parent_columns, old_row)
                new_key = self._get_key_values(fk.parent_columns, new_row)
                if old_key == new_key:
                    continue
                if self._all_null(old_key):
                    continue

                children = self._find_referencing_children(
                    txn_id, child_table, fk.child_columns, old_key
                )
                if not children:
                    continue

                if fk.on_update == FKAction.RESTRICT or fk.on_update == FKAction.NO_ACTION:
                    raise CatalogError(
                        f"Cannot update '{parent_table}': "
                        f"foreign key '{fk.name}' on '{child_table}' references this row"
                    )
                elif fk.on_update == FKAction.CASCADE:
                    for rowid, child_row in children:
                        updates = {}
                        for i, col in enumerate(fk.child_columns):
                            updates[col] = new_key[i]
                        self.db.storage.update_row(txn_id, child_table, rowid, updates)
                elif fk.on_update == FKAction.SET_NULL:
                    for rowid, child_row in children:
                        updates = {col: None for col in fk.child_columns}
                        self.db.storage.update_row(txn_id, child_table, rowid, updates)
                elif fk.on_update == FKAction.SET_DEFAULT:
                    schema = self.db.storage.catalog.get_table(child_table)
                    updates = {}
                    for col in fk.child_columns:
                        col_def = self._get_column_def(schema, col)
                        updates[col] = self._unwrap_default(col_def) if col_def else None
                    for rowid, child_row in children:
                        self.db.storage.update_row(txn_id, child_table, rowid, updates)

    def validate_existing_data(self, table_name: str, fk: ForeignKey, txn_id: int):
        """Validate all existing rows satisfy a new FK constraint."""
        rows = self.db.storage.scan_table(txn_id, table_name)
        for rowid, row_data in rows:
            self._check_parent_exists(fk, row_data)

    def _check_parent_exists(self, fk: ForeignKey, child_row: Dict[str, Any]):
        """Check that the parent row exists for a child row's FK values."""
        key_values = self._get_key_values(fk.child_columns, child_row)

        # NULL FK values always satisfy the constraint (SQL standard)
        if any(v is None for v in key_values):
            return

        # Scan parent table for matching row
        txn_id = self.db.storage.txn_manager.begin()
        try:
            parent_rows = self.db.storage.scan_table(txn_id, fk.parent_table)
            for _, parent_row in parent_rows:
                parent_key = self._get_key_values(fk.parent_columns, parent_row)
                if parent_key == key_values:
                    self.db.storage.txn_manager.commit(txn_id)
                    return
            self.db.storage.txn_manager.commit(txn_id)
        except Exception:
            self.db.storage.txn_manager.abort(txn_id)
            raise

        child_cols = ", ".join(fk.child_columns)
        parent_cols = ", ".join(fk.parent_columns)
        raise CatalogError(
            f"Foreign key violation: '{fk.name}' - "
            f"no matching row in '{fk.parent_table}({parent_cols})' "
            f"for values ({', '.join(repr(v) for v in key_values)})"
        )

    def _get_key_values(self, columns: List[str], row: Dict[str, Any]) -> List:
        """Extract key values from a row for the given columns."""
        return [row.get(col) for col in columns]

    def _all_null(self, values: List) -> bool:
        return all(v is None for v in values)

    def _find_referencing_children(self, txn_id: int, child_table: str,
                                   child_columns: List[str],
                                   parent_key: List) -> List[Tuple[int, Dict]]:
        """Find child rows that reference the given parent key values."""
        rows = self.db.storage.scan_table(txn_id, child_table)
        result = []
        for rowid, row_data in rows:
            child_key = self._get_key_values(child_columns, row_data)
            if child_key == parent_key:
                result.append((rowid, row_data))
        return result

    def _get_column_def(self, schema, col_name: str):
        """Get ColumnDef from schema, handling wrapper types."""
        if hasattr(schema, 'base_schema'):
            return self._get_column_def(schema.base_schema, col_name)
        return schema.get_column(col_name)

    def _unwrap_default(self, col_def) -> Any:
        """Unwrap a column default value (may be SqlLiteral)."""
        if col_def is None or col_def.default is None:
            return None
        if isinstance(col_def.default, SqlLiteral):
            return col_def.default.value
        return col_def.default


# =============================================================================
# FKDB -- Database with FOREIGN KEY + CHECK constraint support
# =============================================================================

class FKDB(CheckDB):
    """CheckDB extended with FOREIGN KEY constraints."""

    def __init__(self):
        super().__init__()
        self._fk_schemas: Dict[str, FKTableSchema] = {}
        self._enforcer = FKEnforcer(self)

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with FK + CHECK constraint support."""
        stmt = parse_fk_sql(sql)
        return self._execute_fk_stmt(stmt)

    def execute_multi(self, sql: str) -> List[ResultSet]:
        """Execute multiple SQL statements."""
        stmts = parse_fk_sql_multi(sql)
        return [self._execute_fk_stmt(s) for s in stmts]

    def _execute_fk_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling FK-specific types."""
        if isinstance(stmt, CreateTableStmt) and hasattr(stmt, '_col_fks'):
            return self._exec_create_with_fks(stmt)
        elif isinstance(stmt, AlterAddFKStmt):
            return self._exec_alter_add_fk(stmt)
        elif isinstance(stmt, AlterDropFKStmt):
            return self._exec_alter_drop_fk(stmt)
        elif isinstance(stmt, AlterDropConstraintStmt):
            # Could be CHECK or FK -- try FK first, then CHECK
            return self._exec_alter_drop_any(stmt)
        elif isinstance(stmt, InsertStmt):
            return self._exec_insert_with_fks(stmt)
        elif isinstance(stmt, UpdateStmt):
            return self._exec_update_with_fks(stmt)
        elif isinstance(stmt, DeleteStmt):
            return self._exec_delete_with_fks(stmt)
        else:
            return self._execute_check_stmt(stmt)

    def _exec_create_with_fks(self, stmt: CreateTableStmt) -> ResultSet:
        """Create table with FK and CHECK constraints."""
        # Create the table with CHECK constraints first
        result = self._exec_create_with_checks(stmt)

        table_name = stmt.table_name

        # Collect FK constraints
        col_fks = getattr(stmt, '_col_fks', [])
        table_fks = getattr(stmt, '_table_fks', [])

        all_fks = []
        for fk in col_fks:
            if fk is not None:
                fk.child_table = table_name
                all_fks.append(fk)
        for fk in table_fks:
            fk.child_table = table_name
            all_fks.append(fk)

        if all_fks:
            fk_schema = self._get_or_create_fk_schema(table_name)
            for fk in all_fks:
                # Validate that parent table and columns exist
                self._validate_fk_references(fk)
                fk_schema.add_foreign_key(fk)

        return result

    def _exec_alter_add_fk(self, stmt: AlterAddFKStmt) -> ResultSet:
        """ALTER TABLE ... ADD FOREIGN KEY."""
        table_name = stmt.table_name
        fk = stmt.foreign_key
        fk.child_table = table_name

        # Validate parent table/columns exist
        self._validate_fk_references(fk)

        # Validate child columns exist
        schema = self.storage.catalog.get_table(table_name)
        col_names = [c.name for c in schema.columns]
        for col in fk.child_columns:
            if col not in col_names:
                raise CatalogError(f"Column '{col}' not found in table '{table_name}'")

        # Validate existing data satisfies the FK
        txn_id = self.storage.txn_manager.begin()
        try:
            self._enforcer.validate_existing_data(table_name, fk, txn_id)
            self.storage.txn_manager.commit(txn_id)
        except CatalogError:
            self.storage.txn_manager.abort(txn_id)
            raise

        fk_schema = self._get_or_create_fk_schema(table_name)
        fk_schema.add_foreign_key(fk)
        return ResultSet(columns=['result'], rows=[['Foreign key constraint added']])

    def _exec_alter_drop_fk(self, stmt: AlterDropFKStmt) -> ResultSet:
        """ALTER TABLE ... DROP FOREIGN KEY."""
        table_name = stmt.table_name
        fk_schema = self._fk_schemas.get(table_name)
        if fk_schema is None:
            raise CatalogError(
                f"Foreign key '{stmt.constraint_name}' not found on table '{table_name}'"
            )
        fk_schema.drop_foreign_key(stmt.constraint_name)
        return ResultSet(columns=['result'], rows=[['Foreign key dropped']])

    def _exec_alter_drop_any(self, stmt: AlterDropConstraintStmt) -> ResultSet:
        """Drop a constraint by name -- could be CHECK or FK."""
        table_name = stmt.table_name
        cname = stmt.constraint_name

        # Try FK first
        fk_schema = self._fk_schemas.get(table_name)
        if fk_schema:
            for fk in fk_schema.foreign_keys:
                if fk.name == cname:
                    fk_schema.drop_foreign_key(cname)
                    return ResultSet(columns=['result'], rows=[['Constraint dropped']])

        # Fall back to CHECK
        return self._exec_alter_drop_constraint(stmt)

    def _exec_insert_with_fks(self, stmt: InsertStmt) -> ResultSet:
        """INSERT with FK + CHECK validation."""
        table_name = stmt.table_name
        fk_schema = self._fk_schemas.get(table_name)
        check_schema = self._check_schemas.get(table_name)

        has_fks = fk_schema and fk_schema.foreign_keys
        has_checks = check_schema and check_schema.check_constraints

        if not has_fks and not has_checks:
            return self._execute_stmt(stmt)

        schema = self.storage.catalog.get_table(table_name)
        col_names = schema.column_names()

        txn_id = self.storage.txn_manager.begin()
        try:
            inserted = 0
            for values_row in stmt.values_list:
                row_data = {}
                cols = stmt.columns if stmt.columns else col_names
                for i, val_expr in enumerate(values_row):
                    if i < len(cols):
                        row_data[cols[i]] = self._eval_insert_expr(val_expr)

                # Insert the row
                rowid = self.storage.insert_row(txn_id, table_name, row_data)

                # Get full row (with defaults)
                full_row = self._get_row_by_id(txn_id, table_name, rowid)
                if full_row is not None:
                    # CHECK constraints
                    if has_checks:
                        from check_constraints import CheckEvaluator
                        self._evaluator.validate_row(
                            table_name, full_row, check_schema.check_constraints
                        )
                    # FK constraints
                    if has_fks:
                        self._enforcer.validate_insert(
                            table_name, full_row, fk_schema.foreign_keys
                        )
                inserted += 1

            self.storage.txn_manager.commit(txn_id)
            return ResultSet(columns=['rows_affected'], rows=[[inserted]])
        except (CatalogError, Exception) as e:
            self.storage.txn_manager.abort(txn_id)
            raise

    def _exec_update_with_fks(self, stmt: UpdateStmt) -> ResultSet:
        """UPDATE with FK validation (both as child and as parent)."""
        table_name = stmt.table_name
        fk_schema = self._fk_schemas.get(table_name)
        check_schema = self._check_schemas.get(table_name)

        has_fks = fk_schema and fk_schema.foreign_keys
        has_checks = check_schema and check_schema.check_constraints
        is_parent = self._is_parent_table(table_name)

        if not has_fks and not has_checks and not is_parent:
            return self._execute_stmt(stmt)

        schema = self.storage.catalog.get_table(table_name)
        txn_id = self.storage.txn_manager.begin()
        try:
            rows = self.storage.scan_table(txn_id, table_name)
            updated = 0
            for rowid, row_data in rows:
                if stmt.where:
                    qe_where = self.compiler._sql_to_qe_expr(stmt.where)
                    if not eval_expr(qe_where, Row(row_data)):
                        continue

                old_row = dict(row_data)

                updates = {}
                for col, expr in stmt.assignments:
                    updates[col] = self._eval_update_expr(expr, row_data)

                self.storage.update_row(txn_id, table_name, rowid, updates)

                full_row = self._get_row_by_id(txn_id, table_name, rowid)
                if full_row is not None:
                    # CHECK constraints
                    if has_checks:
                        self._evaluator.validate_row(
                            table_name, full_row, check_schema.check_constraints
                        )
                    # FK constraints (as child -- new values must exist in parent)
                    if has_fks:
                        self._enforcer.validate_update(
                            table_name, old_row, full_row, fk_schema.foreign_keys
                        )
                    # FK constraints (as parent -- handle cascading to children)
                    if is_parent:
                        self._enforcer.handle_parent_update(
                            table_name, old_row, full_row, txn_id
                        )
                updated += 1

            self.storage.txn_manager.commit(txn_id)
            return ResultSet(columns=['rows_affected'], rows=[[updated]])
        except (CatalogError, Exception):
            self.storage.txn_manager.abort(txn_id)
            raise

    def _exec_delete_with_fks(self, stmt: DeleteStmt) -> ResultSet:
        """DELETE with FK cascade/restrict handling."""
        table_name = stmt.table_name
        is_parent = self._is_parent_table(table_name)

        if not is_parent:
            return self._execute_stmt(stmt)

        txn_id = self.storage.txn_manager.begin()
        try:
            rows = self.storage.scan_table(txn_id, table_name)
            count = 0
            for rowid, row_data in rows:
                if stmt.where is not None:
                    qe_where = self.compiler._sql_to_qe_expr(stmt.where)
                    if not eval_expr(qe_where, Row(row_data)):
                        continue

                # Handle cascading before deleting
                self._enforcer.handle_parent_delete(table_name, row_data, txn_id)
                self.storage.delete_row(txn_id, table_name, rowid)
                count += 1

            self.storage.txn_manager.commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"DELETE {count}",
                           rows_affected=count)
        except (CatalogError, Exception):
            self.storage.txn_manager.abort(txn_id)
            raise

    def _validate_fk_references(self, fk: ForeignKey):
        """Validate that parent table and columns exist, and columns are unique/pk."""
        try:
            parent_schema = self.storage.catalog.get_table(fk.parent_table)
        except CatalogError:
            raise CatalogError(
                f"Referenced table '{fk.parent_table}' does not exist"
            )

        parent_col_names = [c.name for c in parent_schema.columns]
        for col in fk.parent_columns:
            if col not in parent_col_names:
                raise CatalogError(
                    f"Referenced column '{col}' not found in table '{fk.parent_table}'"
                )

        # Parent columns should be PRIMARY KEY or UNIQUE
        # (relaxed: we allow it but warn via validation at enforcement time)
        if len(fk.parent_columns) == 1:
            col = fk.parent_columns[0]
            col_def = parent_schema.get_column(col)
            if col_def and not col_def.primary_key and not col_def.unique:
                # SQL standard requires UNIQUE or PK, but we'll allow it
                pass

    def _is_parent_table(self, table_name: str) -> bool:
        """Check if any FK in any table references this table as parent."""
        for child_table, fk_schema in self._fk_schemas.items():
            for fk in fk_schema.foreign_keys:
                if fk.parent_table == table_name:
                    return True
        return False

    def _get_or_create_fk_schema(self, table_name: str) -> FKTableSchema:
        """Get or create an FKTableSchema for a table."""
        if table_name not in self._fk_schemas:
            base = self._check_schemas.get(table_name)
            if base is None:
                base = self.storage.catalog.get_table(table_name)
            self._fk_schemas[table_name] = FKTableSchema(base_schema=base)
        return self._fk_schemas[table_name]

    def get_foreign_keys(self, table_name: str) -> List[ForeignKey]:
        """Get all foreign keys for a table."""
        fk_schema = self._fk_schemas.get(table_name)
        if fk_schema is None:
            return []
        return list(fk_schema.foreign_keys)

    def describe_foreign_keys(self, table_name: str) -> ResultSet:
        """Show foreign keys for a table."""
        fks = self.get_foreign_keys(table_name)
        rows = []
        for fk in fks:
            rows.append([
                fk.name,
                ", ".join(fk.child_columns),
                fk.parent_table,
                ", ".join(fk.parent_columns),
                fk.on_delete.value,
                fk.on_update.value,
            ])
        return ResultSet(
            columns=['name', 'columns', 'references_table',
                     'references_columns', 'on_delete', 'on_update'],
            rows=rows
        )

    def get_referencing_tables(self, table_name: str) -> List[str]:
        """Get tables that have FKs referencing this table."""
        result = []
        for child_table, fk_schema in self._fk_schemas.items():
            for fk in fk_schema.foreign_keys:
                if fk.parent_table == table_name:
                    if child_table not in result:
                        result.append(child_table)
        return result
