"""
C260: CHECK Constraints
Extends C247 (Mini Database) with CHECK constraint support.

Adds data integrity constraints to the SQL database:
- Column-level CHECK: CREATE TABLE t (x INT CHECK (x > 0))
- Table-level CHECK: CREATE TABLE t (x INT, y INT, CHECK (x < y))
- Named constraints: CONSTRAINT positive_x CHECK (x > 0)
- CHECK validation on INSERT and UPDATE
- ALTER TABLE ADD CHECK / ADD CONSTRAINT
- DROP CONSTRAINT support
- Multi-column CHECK constraints
- CHECK with expressions: arithmetic, comparisons, AND/OR/NOT, IN, BETWEEN, IS NULL
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
    ColumnDef, CatalogError, CompileError, ParseError,
    SelectStmt, InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt,
    SqlColumnRef, SqlLiteral, SqlComparison, SqlLogic, SqlBinOp,
    SqlIsNull, SqlBetween, SqlInList, SqlFuncCall, SqlCase,
    SqlAggCall, SqlStar, SelectExpr,
    TokenType, Token, Lexer, Parser, parse_sql, KEYWORDS,
)
from query_executor import (
    Database as QEDatabase, Table as QETable, Row, ExecutionEngine,
    Operator, SeqScanOp, FilterOp, ProjectOp, SortOp,
    HashAggregateOp, HavingOp, LimitOp, DistinctOp,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr,
    CompOp, LogicOp, AggFunc, AggCall,
    eval_expr,
)


# =============================================================================
# CHECK Constraint AST and Storage
# =============================================================================

@dataclass
class CheckConstraint:
    """A CHECK constraint on a table."""
    name: Optional[str]         # constraint name (None = auto-generated)
    expr: Any                   # SQL AST expression node
    columns: List[str]          # columns referenced (for documentation)
    source: str = ""            # original SQL text of the expression

    def __repr__(self):
        name = self.name or "(unnamed)"
        return f"CheckConstraint({name}, {self.source})"


def _extract_columns(expr) -> List[str]:
    """Extract column names referenced in a SQL expression."""
    cols = []
    if isinstance(expr, SqlColumnRef):
        cols.append(expr.column)
    elif isinstance(expr, SqlComparison):
        cols.extend(_extract_columns(expr.left))
        cols.extend(_extract_columns(expr.right))
    elif isinstance(expr, SqlLogic):
        for op in expr.operands:
            cols.extend(_extract_columns(op))
    elif isinstance(expr, SqlBinOp):
        cols.extend(_extract_columns(expr.left))
        cols.extend(_extract_columns(expr.right))
    elif isinstance(expr, SqlIsNull):
        cols.extend(_extract_columns(expr.expr))
    elif isinstance(expr, SqlBetween):
        cols.extend(_extract_columns(expr.expr))
        cols.extend(_extract_columns(expr.low))
        cols.extend(_extract_columns(expr.high))
    elif isinstance(expr, SqlInList):
        cols.extend(_extract_columns(expr.expr))
        for v in expr.values:
            cols.extend(_extract_columns(v))
    elif isinstance(expr, SqlFuncCall):
        for a in expr.args:
            cols.extend(_extract_columns(a))
    elif isinstance(expr, SqlCase):
        for cond, result in expr.whens:
            cols.extend(_extract_columns(cond))
            cols.extend(_extract_columns(result))
        if expr.else_expr:
            cols.extend(_extract_columns(expr.else_expr))
    return list(dict.fromkeys(cols))  # unique, preserving order


def _expr_to_sql(expr) -> str:
    """Convert a SQL AST expression back to SQL text for display."""
    if isinstance(expr, SqlColumnRef):
        if expr.table:
            return f"{expr.table}.{expr.column}"
        return expr.column
    elif isinstance(expr, SqlLiteral):
        if isinstance(expr.value, str):
            return f"'{expr.value}'"
        elif expr.value is None:
            return "NULL"
        elif expr.value is True:
            return "TRUE"
        elif expr.value is False:
            return "FALSE"
        return str(expr.value)
    elif isinstance(expr, SqlComparison):
        return f"{_expr_to_sql(expr.left)} {expr.op} {_expr_to_sql(expr.right)}"
    elif isinstance(expr, SqlLogic):
        if expr.op == 'not':
            return f"NOT {_expr_to_sql(expr.operands[0])}"
        sep = f" {expr.op.upper()} "
        return f"({sep.join(_expr_to_sql(o) for o in expr.operands)})"
    elif isinstance(expr, SqlBinOp):
        return f"({_expr_to_sql(expr.left)} {expr.op} {_expr_to_sql(expr.right)})"
    elif isinstance(expr, SqlIsNull):
        if expr.negated:
            return f"{_expr_to_sql(expr.expr)} IS NOT NULL"
        return f"{_expr_to_sql(expr.expr)} IS NULL"
    elif isinstance(expr, SqlBetween):
        return f"{_expr_to_sql(expr.expr)} BETWEEN {_expr_to_sql(expr.low)} AND {_expr_to_sql(expr.high)}"
    elif isinstance(expr, SqlInList):
        vals = ", ".join(_expr_to_sql(v) for v in expr.values)
        return f"{_expr_to_sql(expr.expr)} IN ({vals})"
    elif isinstance(expr, SqlFuncCall):
        args = ", ".join(_expr_to_sql(a) for a in expr.args)
        return f"{expr.func_name}({args})"
    return str(expr)


# =============================================================================
# Extended Table Schema with CHECK constraints
# =============================================================================

@dataclass
class CheckTableSchema:
    """Wraps a TableSchema with CHECK constraint support."""
    base_schema: TableSchema
    check_constraints: List[CheckConstraint] = field(default_factory=list)
    _next_constraint_id: int = 1

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

    def add_check(self, constraint: CheckConstraint):
        """Add a CHECK constraint, auto-naming if needed."""
        if constraint.name is None:
            constraint.name = f"{self.name}_check_{self._next_constraint_id}"
            self._next_constraint_id += 1
        # Check for duplicate name
        for existing in self.check_constraints:
            if existing.name == constraint.name:
                raise CatalogError(
                    f"Constraint '{constraint.name}' already exists on table '{self.name}'"
                )
        self.check_constraints.append(constraint)

    def drop_check(self, constraint_name: str):
        """Drop a CHECK constraint by name."""
        for i, c in enumerate(self.check_constraints):
            if c.name == constraint_name:
                self.check_constraints.pop(i)
                return
        raise CatalogError(
            f"Constraint '{constraint_name}' not found on table '{self.name}'"
        )

    def get_check(self, constraint_name: str) -> Optional[CheckConstraint]:
        for c in self.check_constraints:
            if c.name == constraint_name:
                return c
        return None


# =============================================================================
# CHECK Constraint Evaluator
# =============================================================================

class CheckEvaluator:
    """Evaluates CHECK constraint expressions against row data."""

    def __init__(self):
        self._compiler = _CheckExprCompiler()

    def validate_row(self, table_name: str, row_data: Dict[str, Any],
                     constraints: List[CheckConstraint]):
        """Validate a row against all CHECK constraints. Raises CatalogError on violation."""
        for constraint in constraints:
            if not self._evaluate_check(row_data, constraint):
                name = constraint.name or "CHECK"
                raise CatalogError(
                    f"CHECK constraint '{name}' violated for table '{table_name}'"
                )

    def _evaluate_check(self, row_data: Dict[str, Any],
                        constraint: CheckConstraint) -> bool:
        """Evaluate a single CHECK constraint. Returns True if satisfied.
        NULL results are treated as satisfied (SQL standard: CHECK is satisfied
        unless the result is definitely FALSE)."""
        # Per SQL standard: if any referenced column is NULL and the constraint
        # doesn't explicitly check for NULL, the CHECK is considered satisfied
        # (three-valued logic: NULL comparison is not FALSE)
        if not self._expr_checks_null(constraint.expr):
            for col in constraint.columns:
                if row_data.get(col) is None:
                    return True

        qe_expr = self._compiler.compile(constraint.expr)
        row = Row(row_data)
        result = eval_expr(qe_expr, row)
        if result is None:
            return True
        return bool(result)

    def _expr_checks_null(self, expr) -> bool:
        """Check if the expression explicitly tests for NULL (IS NULL / IS NOT NULL)."""
        if isinstance(expr, SqlIsNull):
            return True
        if isinstance(expr, SqlLogic):
            return any(self._expr_checks_null(o) for o in expr.operands)
        return False


class _CheckExprCompiler:
    """Converts SQL AST expressions to QE expressions for evaluation."""

    def compile(self, node) -> Any:
        if isinstance(node, SqlColumnRef):
            return ColumnRef(node.table, node.column)

        if isinstance(node, SqlLiteral):
            return Literal(node.value)

        if isinstance(node, SqlBinOp):
            left = self.compile(node.left)
            right = self.compile(node.right)
            return ArithExpr(node.op, left, right)

        if isinstance(node, SqlComparison):
            left = self.compile(node.left)
            right = self.compile(node.right)
            op_map = {
                '=': CompOp.EQ, '!=': CompOp.NE, '<': CompOp.LT,
                '<=': CompOp.LE, '>': CompOp.GT, '>=': CompOp.GE,
                'like': CompOp.LIKE,
            }
            return Comparison(op_map[node.op], left, right)

        if isinstance(node, SqlLogic):
            if node.op == 'not':
                operand = self.compile(node.operands[0])
                return LogicExpr(LogicOp.NOT, [operand])
            operands = [self.compile(o) for o in node.operands]
            op_map = {'and': LogicOp.AND, 'or': LogicOp.OR}
            return LogicExpr(op_map[node.op], operands)

        if isinstance(node, SqlIsNull):
            expr = self.compile(node.expr)
            if node.negated:
                return Comparison(CompOp.IS_NOT_NULL, expr, Literal(None))
            return Comparison(CompOp.IS_NULL, expr, Literal(None))

        if isinstance(node, SqlBetween):
            expr = self.compile(node.expr)
            low = self.compile(node.low)
            high = self.compile(node.high)
            return LogicExpr(LogicOp.AND, [
                Comparison(CompOp.GE, expr, low),
                Comparison(CompOp.LE, expr, high),
            ])

        if isinstance(node, SqlInList):
            expr = self.compile(node.expr)
            if len(node.values) == 1:
                return Comparison(CompOp.EQ, expr, self.compile(node.values[0]))
            comparisons = [Comparison(CompOp.EQ, expr, self.compile(v))
                          for v in node.values]
            result = comparisons[0]
            for c in comparisons[1:]:
                result = LogicExpr(LogicOp.OR, [result, c])
            return result

        if isinstance(node, SqlFuncCall):
            args = [self.compile(a) for a in node.args]
            return FuncExpr(node.func_name, args)

        raise CompileError(f"Unsupported expression in CHECK: {type(node).__name__}")


# =============================================================================
# Extended Parser with CHECK support
# =============================================================================

class CheckParser(Parser):
    """Parser extended with CHECK constraint support."""

    # Add CHECK and CONSTRAINT keywords
    EXTRA_KEYWORDS = {
        'check': 'CHECK',
        'constraint': 'CONSTRAINT',
        'references': 'REFERENCES',
    }

    def __init__(self, tokens):
        # Remap CHECK/CONSTRAINT tokens before passing to parent
        remapped = []
        for t in tokens:
            if t.type == TokenType.IDENT and t.value.lower() in self.EXTRA_KEYWORDS:
                remapped.append(Token(TokenType.IDENT, t.value.lower(), t.pos))
            else:
                remapped.append(t)
        super().__init__(remapped)

    def _parse_column_defs(self):
        """Parse column definitions with CHECK constraint support."""
        cols = [self._parse_column_def_with_check()]
        table_checks = []

        while self.match(TokenType.COMMA):
            # Check for table-level PRIMARY KEY
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

            # Check for table-level CONSTRAINT name CHECK (...)
            if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'constraint':
                self.advance()  # consume CONSTRAINT
                cname = self.expect(TokenType.IDENT).value
                self._expect_ident('check')
                self.expect(TokenType.LPAREN)
                expr = self._parse_or()
                self.expect(TokenType.RPAREN)
                source = _expr_to_sql(expr)
                table_checks.append(CheckConstraint(
                    name=cname, expr=expr,
                    columns=_extract_columns(expr), source=source
                ))
                continue

            # Check for table-level CHECK (...)
            if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'check':
                self.advance()  # consume CHECK
                self.expect(TokenType.LPAREN)
                expr = self._parse_or()
                self.expect(TokenType.RPAREN)
                source = _expr_to_sql(expr)
                table_checks.append(CheckConstraint(
                    name=None, expr=expr,
                    columns=_extract_columns(expr), source=source
                ))
                continue

            cols.append(self._parse_column_def_with_check())

        return [c['def'] for c in cols], [c.get('check') for c in cols if c.get('check')], table_checks

    def _parse_column_def_with_check(self):
        """Parse a single column definition, capturing any CHECK constraint."""
        col_def = self._parse_column_def()
        check = None

        # Check for trailing CHECK on column
        if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'check':
            self.advance()
            self.expect(TokenType.LPAREN)
            expr = self._parse_or()
            self.expect(TokenType.RPAREN)
            source = _expr_to_sql(expr)
            check = CheckConstraint(
                name=None, expr=expr,
                columns=[col_def.name], source=source
            )

        # Check for CONSTRAINT name CHECK on column
        if (self.peek().type == TokenType.IDENT and
            self.peek().value.lower() == 'constraint'):
            saved_pos = self.pos
            self.advance()  # CONSTRAINT
            if self.peek().type == TokenType.IDENT and self.peek().value.lower() != 'check':
                cname = self.advance().value
                if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'check':
                    self.advance()
                    self.expect(TokenType.LPAREN)
                    expr = self._parse_or()
                    self.expect(TokenType.RPAREN)
                    source = _expr_to_sql(expr)
                    check = CheckConstraint(
                        name=cname, expr=expr,
                        columns=[col_def.name], source=source
                    )
                else:
                    self.pos = saved_pos
            else:
                self.pos = saved_pos

        return {'def': col_def, 'check': check}

    def _parse_statement(self):
        """Override to handle ALTER TABLE statements."""
        t = self.peek()
        if t.type == TokenType.ALTER:
            return self._parse_alter()
        return super()._parse_statement()

    def _expect_ident(self, value: str):
        """Expect an IDENT token with a specific value."""
        tok = self.advance()
        if tok.type != TokenType.IDENT or tok.value.lower() != value:
            raise ParseError(f"Expected '{value}', got '{tok.value}'")
        return tok

    def _parse_create(self):
        """Override to handle CHECK constraints in CREATE TABLE."""
        self.expect(TokenType.CREATE)
        if self.peek().type == TokenType.INDEX:
            return self._parse_create_index()
        # TABLE already consumed by expect below or we need to handle IF NOT EXISTS
        return self._parse_create_table_with_checks()

    def _parse_create_table_with_checks(self):
        """Parse CREATE TABLE with CHECK constraints. CREATE already consumed."""
        self.expect(TokenType.TABLE)
        if_not_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True

        table_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        col_defs, col_checks, table_checks = self._parse_column_defs()
        self.expect(TokenType.RPAREN)

        stmt = CreateTableStmt(
            table_name=table_name,
            columns=col_defs,
            if_not_exists=if_not_exists
        )
        # Attach checks as extra attributes
        stmt._col_checks = col_checks
        stmt._table_checks = table_checks
        return stmt

    def _expect_kw_or_ident(self, token_type):
        """Expect either a keyword token or an IDENT with matching name."""
        tok = self.peek()
        if tok.type == token_type:
            return self.advance()
        if tok.type == TokenType.IDENT:
            # Check if ident matches the keyword name
            expected = token_type.name.lower()
            if tok.value.lower() == expected:
                return self.advance()
        return self.expect(token_type)

    def _parse_alter(self):
        """Parse ALTER TABLE ... ADD CHECK / ADD CONSTRAINT / DROP CONSTRAINT."""
        self.expect(TokenType.ALTER)
        self.expect(TokenType.TABLE)
        table_name = self.expect(TokenType.IDENT).value

        if self.match(TokenType.ADD):
            return self._parse_alter_add(table_name)
        elif self.peek().type == TokenType.DROP:
            self.advance()  # consume DROP
            return self._parse_alter_drop(table_name)
        else:
            raise ParseError("Expected ADD or DROP after ALTER TABLE")

    def _check_ident(self, value: str) -> bool:
        tok = self.peek()
        return tok.type == TokenType.IDENT and tok.value.lower() == value

    def _parse_alter_add(self, table_name: str):
        """Parse ALTER TABLE ... ADD [CONSTRAINT name] CHECK (...)."""
        cname = None

        # CONSTRAINT name
        if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'constraint':
            self.advance()
            cname = self.expect(TokenType.IDENT).value

        # CHECK (...)
        if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'check':
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

        # If not CHECK, fall through to regular ADD COLUMN
        raise ParseError("Expected CHECK after ADD [CONSTRAINT name]")

    def _parse_alter_drop(self, table_name: str):
        """Parse ALTER TABLE ... DROP CONSTRAINT name."""
        if self.peek().type == TokenType.IDENT and self.peek().value.lower() == 'constraint':
            self.advance()
            cname = self.expect(TokenType.IDENT).value
            return AlterDropConstraintStmt(
                table_name=table_name,
                constraint_name=cname
            )
        raise ParseError("Expected CONSTRAINT after DROP")


# =============================================================================
# Additional AST nodes
# =============================================================================

@dataclass
class AlterAddCheckStmt:
    table_name: str
    constraint: CheckConstraint

@dataclass
class AlterDropConstraintStmt:
    table_name: str
    constraint_name: str


# =============================================================================
# Extended Lexer
# =============================================================================

class CheckLexer(Lexer):
    """Lexer that recognizes CHECK and CONSTRAINT as identifiers (handled by parser)."""
    pass  # No changes needed -- CHECK/CONSTRAINT lex as IDENT, parser handles them


# =============================================================================
# Parse function
# =============================================================================

def parse_check_sql(sql: str):
    """Parse a single SQL statement with CHECK constraint support."""
    lexer = CheckLexer(sql)
    parser = CheckParser(lexer.tokens)
    return parser.parse()


def parse_check_sql_multi(sql: str):
    """Parse multiple SQL statements with CHECK constraint support."""
    lexer = CheckLexer(sql)
    parser = CheckParser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# CheckDB -- Database with CHECK constraint support
# =============================================================================

class CheckDB(MiniDB):
    """MiniDB extended with CHECK constraints."""

    def __init__(self):
        super().__init__()
        self._check_schemas: Dict[str, CheckTableSchema] = {}
        self._evaluator = CheckEvaluator()

    def execute(self, sql: str) -> ResultSet:
        """Execute SQL with CHECK constraint support."""
        stmt = parse_check_sql(sql)
        return self._execute_check_stmt(stmt)

    def _execute_check_stmt(self, stmt) -> ResultSet:
        """Execute a statement, handling CHECK-specific types."""
        if isinstance(stmt, CreateTableStmt) and hasattr(stmt, '_col_checks'):
            return self._exec_create_with_checks(stmt)
        elif isinstance(stmt, AlterAddCheckStmt):
            return self._exec_alter_add_check(stmt)
        elif isinstance(stmt, AlterDropConstraintStmt):
            return self._exec_alter_drop_constraint(stmt)
        elif isinstance(stmt, InsertStmt):
            return self._exec_insert_with_checks(stmt)
        elif isinstance(stmt, UpdateStmt):
            return self._exec_update_with_checks(stmt)
        else:
            return self._execute_stmt(stmt)

    def _exec_create_with_checks(self, stmt: CreateTableStmt) -> ResultSet:
        """Create table with CHECK constraints."""
        # First, create the table normally
        result = self._execute_stmt(CreateTableStmt(
            table_name=stmt.table_name,
            columns=stmt.columns,
            if_not_exists=stmt.if_not_exists
        ))

        # Now attach CHECK constraints
        base_schema = self.storage.catalog.get_table(stmt.table_name)
        check_schema = CheckTableSchema(base_schema=base_schema)

        # Add column-level checks
        for check in getattr(stmt, '_col_checks', []):
            if check is not None:
                check_schema.add_check(check)

        # Add table-level checks
        for check in getattr(stmt, '_table_checks', []):
            check_schema.add_check(check)

        self._check_schemas[stmt.table_name] = check_schema
        return result

    def _exec_alter_add_check(self, stmt: AlterAddCheckStmt) -> ResultSet:
        """ALTER TABLE ... ADD [CONSTRAINT] CHECK."""
        table_name = stmt.table_name
        # Ensure table exists
        self.storage.catalog.get_table(table_name)

        check_schema = self._get_or_create_check_schema(table_name)

        # Validate against existing data
        txn_id = self.storage.txn_manager.begin()
        try:
            rows = self.storage.scan_table(txn_id, table_name)
            for rowid, row_data in rows:
                self._evaluator.validate_row(
                    table_name, row_data, [stmt.constraint]
                )
            self.storage.txn_manager.commit(txn_id)
        except CatalogError:
            self.storage.txn_manager.abort(txn_id)
            raise

        check_schema.add_check(stmt.constraint)
        return ResultSet(columns=['result'], rows=[['CHECK constraint added']])

    def _exec_alter_drop_constraint(self, stmt: AlterDropConstraintStmt) -> ResultSet:
        """ALTER TABLE ... DROP CONSTRAINT."""
        table_name = stmt.table_name
        self.storage.catalog.get_table(table_name)  # ensure exists
        check_schema = self._check_schemas.get(table_name)
        if check_schema is None:
            raise CatalogError(
                f"Constraint '{stmt.constraint_name}' not found on table '{table_name}'"
            )
        check_schema.drop_check(stmt.constraint_name)
        return ResultSet(columns=['result'], rows=[['Constraint dropped']])

    def _exec_insert_with_checks(self, stmt: InsertStmt) -> ResultSet:
        """INSERT with CHECK constraint validation."""
        table_name = stmt.table_name
        check_schema = self._check_schemas.get(table_name)

        if check_schema is None or not check_schema.check_constraints:
            return self._execute_stmt(stmt)

        # We need to intercept after row assembly but before commit
        # Strategy: let the parent do the insert, but wrap with check validation
        schema = self.storage.catalog.get_table(table_name)
        col_names = schema.column_names()

        txn_id = self.storage.txn_manager.begin()
        try:
            inserted = 0
            for values_row in stmt.values_list:
                # Evaluate value expressions
                row_data = {}
                cols = stmt.columns if stmt.columns else col_names
                for i, val_expr in enumerate(values_row):
                    if i < len(cols):
                        row_data[cols[i]] = self._eval_insert_expr(val_expr)

                # Let storage engine assemble the full row (defaults, auto-increment)
                rowid = self.storage.insert_row(txn_id, table_name, row_data)

                # Retrieve the full row to validate CHECK constraints
                full_row = self._get_row_by_id(txn_id, table_name, rowid)
                if full_row is not None:
                    self._evaluator.validate_row(
                        table_name, full_row, check_schema.check_constraints
                    )
                inserted += 1

            self.storage.txn_manager.commit(txn_id)
            return ResultSet(columns=['rows_affected'], rows=[[inserted]])
        except (CatalogError, Exception) as e:
            self.storage.txn_manager.abort(txn_id)
            raise

    def _exec_update_with_checks(self, stmt: UpdateStmt) -> ResultSet:
        """UPDATE with CHECK constraint validation."""
        table_name = stmt.table_name
        check_schema = self._check_schemas.get(table_name)

        if check_schema is None or not check_schema.check_constraints:
            return self._execute_stmt(stmt)

        schema = self.storage.catalog.get_table(table_name)
        txn_id = self.storage.txn_manager.begin()
        try:
            rows = self.storage.scan_table(txn_id, table_name)
            updated = 0
            for rowid, row_data in rows:
                # Check WHERE clause
                if stmt.where:
                    qe_where = self.compiler._sql_to_qe_expr(stmt.where)
                    if not eval_expr(qe_where, Row(row_data)):
                        continue

                # Build updates
                updates = {}
                for col, expr in stmt.assignments:
                    updates[col] = self._eval_update_expr(expr, row_data)

                # Apply update
                self.storage.update_row(txn_id, table_name, rowid, updates)

                # Retrieve and validate
                full_row = self._get_row_by_id(txn_id, table_name, rowid)
                if full_row is not None:
                    self._evaluator.validate_row(
                        table_name, full_row, check_schema.check_constraints
                    )
                updated += 1

            self.storage.txn_manager.commit(txn_id)
            return ResultSet(columns=['rows_affected'], rows=[[updated]])
        except (CatalogError, Exception) as e:
            self.storage.txn_manager.abort(txn_id)
            raise

    def _eval_insert_expr(self, expr) -> Any:
        """Evaluate an expression for INSERT values."""
        if isinstance(expr, SqlLiteral):
            return expr.value
        if isinstance(expr, SqlBinOp):
            left = self._eval_insert_expr(expr.left)
            right = self._eval_insert_expr(expr.right)
            if expr.op == '+': return left + right
            if expr.op == '-': return left - right
            if expr.op == '*': return left * right
            if expr.op == '/': return left / right if right != 0 else None
        if isinstance(expr, SqlFuncCall):
            # Limited function support for INSERT values
            pass
        return None

    def _eval_update_expr(self, expr, current_row: Dict[str, Any]) -> Any:
        """Evaluate an expression for UPDATE values, with access to current row."""
        if isinstance(expr, SqlLiteral):
            return expr.value
        if isinstance(expr, SqlColumnRef):
            return current_row.get(expr.column)
        if isinstance(expr, SqlBinOp):
            left = self._eval_update_expr(expr.left, current_row)
            right = self._eval_update_expr(expr.right, current_row)
            if left is None or right is None:
                return None
            if expr.op == '+': return left + right
            if expr.op == '-': return left - right
            if expr.op == '*': return left * right
            if expr.op == '/': return left / right if right != 0 else None
        return None

    def _get_row_by_id(self, txn_id: int, table: str, rowid: int) -> Optional[Dict[str, Any]]:
        """Get a row by its rowid within a transaction."""
        key = self.storage._row_key(table, rowid)
        return self.storage.txn_manager.get(txn_id, key)

    def _get_or_create_check_schema(self, table_name: str) -> CheckTableSchema:
        """Get or create a CheckTableSchema for a table."""
        if table_name not in self._check_schemas:
            base_schema = self.storage.catalog.get_table(table_name)
            self._check_schemas[table_name] = CheckTableSchema(base_schema=base_schema)
        return self._check_schemas[table_name]

    def get_constraints(self, table_name: str) -> List[CheckConstraint]:
        """Get all CHECK constraints for a table."""
        check_schema = self._check_schemas.get(table_name)
        if check_schema is None:
            return []
        return list(check_schema.check_constraints)

    def describe_constraints(self, table_name: str) -> ResultSet:
        """Show CHECK constraints for a table."""
        constraints = self.get_constraints(table_name)
        rows = []
        for c in constraints:
            rows.append([c.name, c.source, ', '.join(c.columns)])
        return ResultSet(
            columns=['constraint_name', 'expression', 'columns'],
            rows=rows
        )
