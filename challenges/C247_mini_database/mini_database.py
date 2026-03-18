"""
C247: Mini Database Engine
Composes C244 (Buffer Pool) + C245 (Query Executor) + C246 (Transaction Manager)

A complete relational database with:
- SQL parser (CREATE TABLE, INSERT, SELECT, UPDATE, DELETE, BEGIN/COMMIT/ROLLBACK)
- ACID transactions via C246
- Buffer pool page management via C244
- Volcano-model query execution via C245
- Indexes, joins, aggregation, subqueries
"""

import sys
import os
import re
import time
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Dict, List, Tuple, Set

# Import composed components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from buffer_pool import (
    BufferPoolManager, DiskManager, Page, EvictionPolicy
)
from query_executor import (
    Database as QEDatabase, Table as QETable, Row, ExecutionEngine,
    QueryPlan, Operator, SeqScanOp, FilterOp, ProjectOp, SortOp,
    HashAggregateOp, HavingOp, LimitOp, DistinctOp,
    HashJoinOp, NestedLoopJoinOp, SortMergeJoinOp,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr,
    CompOp, LogicOp, AggFunc, AggCall, IndexScanOp,
    UnionOp, SemiJoinOp, AntiJoinOp, TopNOp
)
from transaction_manager import (
    TransactionManager, Transaction, ReadOnlyTransaction,
    TransactionError, DeadlockError, ConflictError,
    IsolationLevel
)


# =============================================================================
# SQL Parser
# =============================================================================

class TokenType(Enum):
    # Keywords
    SELECT = auto(); FROM = auto(); WHERE = auto(); INSERT = auto()
    INTO = auto(); VALUES = auto(); UPDATE = auto(); SET = auto()
    DELETE = auto(); CREATE = auto(); TABLE = auto(); DROP = auto()
    ALTER = auto(); ADD = auto(); INDEX = auto(); ON = auto()
    AND = auto(); OR = auto(); NOT = auto(); IN = auto()
    BETWEEN = auto(); LIKE = auto(); IS = auto(); NULL = auto()
    TRUE = auto(); FALSE = auto(); AS = auto(); JOIN = auto()
    LEFT = auto(); RIGHT = auto(); INNER = auto(); CROSS = auto()
    OUTER = auto(); FULL = auto()
    ORDER = auto(); BY = auto(); ASC = auto(); DESC = auto()
    GROUP = auto(); HAVING = auto(); LIMIT = auto(); OFFSET = auto()
    DISTINCT = auto(); ALL = auto(); UNION = auto()
    INTERSECT = auto(); EXCEPT = auto()
    BEGIN = auto(); COMMIT = auto(); ROLLBACK = auto()
    SAVEPOINT = auto(); RELEASE = auto()
    COUNT = auto(); SUM = auto(); AVG = auto(); MIN = auto(); MAX = auto()
    EXISTS = auto(); CASE = auto(); WHEN = auto(); THEN = auto()
    ELSE = auto(); END = auto(); IF = auto()
    INT = auto(); TEXT = auto(); FLOAT = auto(); BOOL = auto()
    PRIMARY = auto(); KEY = auto(); UNIQUE = auto(); DEFAULT = auto()
    NOT_NULL = auto()
    SHOW = auto(); TABLES = auto(); DESCRIBE = auto(); EXPLAIN = auto()
    TRANSACTION = auto(); TO = auto()
    # Literals and identifiers
    IDENT = auto(); NUMBER = auto(); STRING = auto()
    # Operators
    EQ = auto(); NE = auto(); LT = auto(); LE = auto()
    GT = auto(); GE = auto()
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto()
    # Punctuation
    LPAREN = auto(); RPAREN = auto(); COMMA = auto(); DOT = auto()
    SEMICOLON = auto()
    # Special
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    pos: int = 0


KEYWORDS = {
    'select': TokenType.SELECT, 'from': TokenType.FROM, 'where': TokenType.WHERE,
    'insert': TokenType.INSERT, 'into': TokenType.INTO, 'values': TokenType.VALUES,
    'update': TokenType.UPDATE, 'set': TokenType.SET, 'delete': TokenType.DELETE,
    'create': TokenType.CREATE, 'table': TokenType.TABLE, 'drop': TokenType.DROP,
    'alter': TokenType.ALTER, 'add': TokenType.ADD, 'index': TokenType.INDEX,
    'on': TokenType.ON, 'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
    'in': TokenType.IN, 'between': TokenType.BETWEEN, 'like': TokenType.LIKE,
    'is': TokenType.IS, 'null': TokenType.NULL, 'true': TokenType.TRUE,
    'false': TokenType.FALSE, 'as': TokenType.AS, 'join': TokenType.JOIN,
    'left': TokenType.LEFT, 'right': TokenType.RIGHT, 'inner': TokenType.INNER,
    'cross': TokenType.CROSS, 'outer': TokenType.OUTER, 'full': TokenType.FULL,
    'order': TokenType.ORDER, 'by': TokenType.BY, 'asc': TokenType.ASC,
    'desc': TokenType.DESC, 'group': TokenType.GROUP, 'having': TokenType.HAVING,
    'limit': TokenType.LIMIT, 'offset': TokenType.OFFSET,
    'distinct': TokenType.DISTINCT, 'all': TokenType.ALL, 'union': TokenType.UNION,
    'intersect': TokenType.INTERSECT, 'except': TokenType.EXCEPT,
    'begin': TokenType.BEGIN, 'commit': TokenType.COMMIT, 'rollback': TokenType.ROLLBACK,
    'savepoint': TokenType.SAVEPOINT, 'release': TokenType.RELEASE,
    'count': TokenType.COUNT, 'sum': TokenType.SUM, 'avg': TokenType.AVG,
    'min': TokenType.MIN, 'max': TokenType.MAX,
    'exists': TokenType.EXISTS, 'case': TokenType.CASE, 'when': TokenType.WHEN,
    'then': TokenType.THEN, 'else': TokenType.ELSE, 'end': TokenType.END,
    'int': TokenType.INT, 'integer': TokenType.INT, 'text': TokenType.TEXT,
    'varchar': TokenType.TEXT, 'float': TokenType.FLOAT, 'real': TokenType.FLOAT,
    'double': TokenType.FLOAT, 'bool': TokenType.BOOL, 'boolean': TokenType.BOOL,
    'primary': TokenType.PRIMARY, 'key': TokenType.KEY, 'unique': TokenType.UNIQUE,
    'default': TokenType.DEFAULT,
    'show': TokenType.SHOW, 'tables': TokenType.TABLES, 'describe': TokenType.DESCRIBE,
    'explain': TokenType.EXPLAIN, 'transaction': TokenType.TRANSACTION, 'to': TokenType.TO,
    'if': TokenType.IF,
}


class Lexer:
    def __init__(self, sql: str):
        self.sql = sql
        self.pos = 0
        self.tokens: List[Token] = []
        self._tokenize()

    def _tokenize(self):
        s = self.sql
        n = len(s)
        while self.pos < n:
            c = s[self.pos]
            if c.isspace():
                self.pos += 1
                continue
            # Comments
            if c == '-' and self.pos + 1 < n and s[self.pos + 1] == '-':
                while self.pos < n and s[self.pos] != '\n':
                    self.pos += 1
                continue
            # Multi-char operators
            if self.pos + 1 < n:
                two = s[self.pos:self.pos + 2]
                if two == '!=':
                    self.tokens.append(Token(TokenType.NE, '!=', self.pos))
                    self.pos += 2
                    continue
                if two == '<>':
                    self.tokens.append(Token(TokenType.NE, '<>', self.pos))
                    self.pos += 2
                    continue
                if two == '<=':
                    self.tokens.append(Token(TokenType.LE, '<=', self.pos))
                    self.pos += 2
                    continue
                if two == '>=':
                    self.tokens.append(Token(TokenType.GE, '>=', self.pos))
                    self.pos += 2
                    continue
            # Single-char operators
            single = {
                '=': TokenType.EQ, '<': TokenType.LT, '>': TokenType.GT,
                '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
                '/': TokenType.SLASH, '(': TokenType.LPAREN, ')': TokenType.RPAREN,
                ',': TokenType.COMMA, '.': TokenType.DOT, ';': TokenType.SEMICOLON,
            }
            if c in single:
                self.tokens.append(Token(single[c], c, self.pos))
                self.pos += 1
                continue
            # Numbers
            if c.isdigit():
                start = self.pos
                while self.pos < n and (s[self.pos].isdigit() or s[self.pos] == '.'):
                    self.pos += 1
                num_str = s[start:self.pos]
                val = float(num_str) if '.' in num_str else int(num_str)
                self.tokens.append(Token(TokenType.NUMBER, val, start))
                continue
            # Strings
            if c in ("'", '"'):
                quote = c
                self.pos += 1
                start = self.pos
                parts = []
                while self.pos < n and s[self.pos] != quote:
                    if s[self.pos] == '\\' and self.pos + 1 < n:
                        self.pos += 1
                    parts.append(s[self.pos])
                    self.pos += 1
                if self.pos < n:
                    self.pos += 1  # closing quote
                self.tokens.append(Token(TokenType.STRING, ''.join(parts), start))
                continue
            # Identifiers / keywords
            if c.isalpha() or c == '_':
                start = self.pos
                while self.pos < n and (s[self.pos].isalnum() or s[self.pos] == '_'):
                    self.pos += 1
                word = s[start:self.pos]
                lower = word.lower()
                if lower in KEYWORDS:
                    self.tokens.append(Token(KEYWORDS[lower], word, start))
                else:
                    self.tokens.append(Token(TokenType.IDENT, word, start))
                continue
            # Unknown char -- skip
            self.pos += 1

        self.tokens.append(Token(TokenType.EOF, None, self.pos))


# =============================================================================
# SQL AST
# =============================================================================

@dataclass
class ColumnDef:
    name: str
    col_type: str
    primary_key: bool = False
    unique: bool = False
    not_null: bool = False
    default: Any = None

@dataclass
class CreateTableStmt:
    table_name: str
    columns: List[ColumnDef]
    if_not_exists: bool = False

@dataclass
class DropTableStmt:
    table_name: str
    if_exists: bool = False

@dataclass
class CreateIndexStmt:
    index_name: str
    table_name: str
    column: str

@dataclass
class InsertStmt:
    table_name: str
    columns: Optional[List[str]]
    values_list: List[List[Any]]  # list of value-rows

@dataclass
class SelectExpr:
    """A single item in SELECT clause"""
    expr: Any  # expression AST node
    alias: Optional[str] = None

@dataclass
class TableRef:
    table_name: str
    alias: Optional[str] = None

@dataclass
class JoinClause:
    join_type: str  # 'inner', 'left', 'cross'
    table: 'TableRef'
    condition: Any = None  # ON expression

@dataclass
class SelectStmt:
    columns: List[SelectExpr]  # SelectExpr or '*'
    from_table: Optional[TableRef] = None
    joins: List[JoinClause] = field(default_factory=list)
    where: Any = None
    group_by: Optional[List[Any]] = None
    having: Any = None
    order_by: Optional[List[Tuple[Any, bool]]] = None  # (expr, asc)
    limit: Optional[int] = None
    offset: Optional[int] = None
    distinct: bool = False

@dataclass
class UpdateStmt:
    table_name: str
    assignments: List[Tuple[str, Any]]  # (column, value_expr)
    where: Any = None

@dataclass
class DeleteStmt:
    table_name: str
    where: Any = None

@dataclass
class BeginStmt:
    pass

@dataclass
class CommitStmt:
    pass

@dataclass
class RollbackStmt:
    savepoint: Optional[str] = None

@dataclass
class SavepointStmt:
    name: str

@dataclass
class ShowTablesStmt:
    pass

@dataclass
class DescribeStmt:
    table_name: str

@dataclass
class ExplainStmt:
    stmt: Any  # wrapped statement


# SQL expression AST nodes
@dataclass
class SqlColumnRef:
    table: Optional[str]
    column: str

@dataclass
class SqlLiteral:
    value: Any

@dataclass
class SqlBinOp:
    op: str  # '+', '-', '*', '/'
    left: Any
    right: Any

@dataclass
class SqlComparison:
    op: str  # '=', '!=', '<', '<=', '>', '>=', 'like', 'in', 'between'
    left: Any
    right: Any

@dataclass
class SqlLogic:
    op: str  # 'and', 'or', 'not'
    operands: List[Any]

@dataclass
class SqlIsNull:
    expr: Any
    negated: bool = False

@dataclass
class SqlFuncCall:
    func_name: str
    args: List[Any]
    distinct: bool = False

@dataclass
class SqlAggCall:
    func: str  # 'count', 'sum', 'avg', 'min', 'max'
    arg: Any  # expression or None for count(*)
    distinct: bool = False
    alias: Optional[str] = None

@dataclass
class SqlBetween:
    expr: Any
    low: Any
    high: Any

@dataclass
class SqlInList:
    expr: Any
    values: List[Any]

@dataclass
class SqlCase:
    whens: List[Tuple[Any, Any]]  # (condition, result)
    else_result: Any = None

@dataclass
class SqlStar:
    table: Optional[str] = None


# =============================================================================
# SQL Parser
# =============================================================================

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt: TokenType) -> Token:
        t = self.advance()
        if t.type != tt:
            raise ParseError(f"Expected {tt.name}, got {t.type.name} ({t.value!r}) at pos {t.pos}")
        return t

    def match(self, *types: TokenType) -> Optional[Token]:
        if self.peek().type in types:
            return self.advance()
        return None

    def parse(self) -> Any:
        stmt = self._parse_statement()
        self.match(TokenType.SEMICOLON)
        return stmt

    def parse_multi(self) -> List[Any]:
        stmts = []
        while self.peek().type != TokenType.EOF:
            stmts.append(self._parse_statement())
            self.match(TokenType.SEMICOLON)
        return stmts

    def _parse_statement(self):
        t = self.peek()
        if t.type == TokenType.SELECT:
            return self._parse_select()
        elif t.type == TokenType.INSERT:
            return self._parse_insert()
        elif t.type == TokenType.UPDATE:
            return self._parse_update()
        elif t.type == TokenType.DELETE:
            return self._parse_delete()
        elif t.type == TokenType.CREATE:
            return self._parse_create()
        elif t.type == TokenType.DROP:
            return self._parse_drop()
        elif t.type == TokenType.BEGIN:
            self.advance()
            self.match(TokenType.TRANSACTION)
            return BeginStmt()
        elif t.type == TokenType.COMMIT:
            self.advance()
            return CommitStmt()
        elif t.type == TokenType.ROLLBACK:
            self.advance()
            if self.match(TokenType.TO):
                self.match(TokenType.SAVEPOINT)
                name = self.expect(TokenType.IDENT).value
                return RollbackStmt(savepoint=name)
            return RollbackStmt()
        elif t.type == TokenType.SAVEPOINT:
            self.advance()
            name = self.expect(TokenType.IDENT).value
            return SavepointStmt(name=name)
        elif t.type == TokenType.SHOW:
            self.advance()
            self.expect(TokenType.TABLES)
            return ShowTablesStmt()
        elif t.type == TokenType.DESCRIBE:
            self.advance()
            name = self.expect(TokenType.IDENT).value
            return DescribeStmt(table_name=name)
        elif t.type == TokenType.EXPLAIN:
            self.advance()
            inner = self._parse_statement()
            return ExplainStmt(stmt=inner)
        else:
            raise ParseError(f"Unexpected token {t.type.name} ({t.value!r}) at pos {t.pos}")

    def _parse_select(self) -> SelectStmt:
        self.expect(TokenType.SELECT)
        distinct = bool(self.match(TokenType.DISTINCT))

        # SELECT columns
        columns = self._parse_select_list()

        # FROM
        from_table = None
        joins = []
        if self.match(TokenType.FROM):
            from_table = self._parse_table_ref()
            # JOINs
            while True:
                join = self._try_parse_join()
                if join is None:
                    break
                joins.append(join)

        # WHERE
        where = None
        if self.match(TokenType.WHERE):
            where = self._parse_expr()

        # GROUP BY
        group_by = None
        if self.match(TokenType.GROUP):
            self.expect(TokenType.BY)
            group_by = [self._parse_expr()]
            while self.match(TokenType.COMMA):
                group_by.append(self._parse_expr())

        # HAVING
        having = None
        if self.match(TokenType.HAVING):
            having = self._parse_expr()

        # ORDER BY
        order_by = None
        if self.match(TokenType.ORDER):
            self.expect(TokenType.BY)
            order_by = self._parse_order_by_list()

        # LIMIT / OFFSET
        limit_val = None
        offset_val = None
        if self.match(TokenType.LIMIT):
            limit_val = int(self.expect(TokenType.NUMBER).value)
            if self.match(TokenType.OFFSET):
                offset_val = int(self.expect(TokenType.NUMBER).value)

        return SelectStmt(
            columns=columns, from_table=from_table, joins=joins,
            where=where, group_by=group_by, having=having,
            order_by=order_by, limit=limit_val, offset=offset_val,
            distinct=distinct
        )

    def _parse_select_list(self) -> List[SelectExpr]:
        cols = []
        first = self._parse_select_item()
        cols.append(first)
        while self.match(TokenType.COMMA):
            cols.append(self._parse_select_item())
        return cols

    def _parse_select_item(self) -> SelectExpr:
        # Check for * or table.*
        if self.peek().type == TokenType.STAR:
            self.advance()
            return SelectExpr(expr=SqlStar(), alias=None)

        expr = self._parse_expr()

        # Check if this is table.* pattern
        if isinstance(expr, SqlColumnRef) and self.peek().type == TokenType.DOT:
            self.advance()
            if self.match(TokenType.STAR):
                return SelectExpr(expr=SqlStar(table=expr.column), alias=None)

        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENT).value
        elif self.peek().type == TokenType.IDENT and self.peek().type not in (
            TokenType.FROM, TokenType.WHERE, TokenType.GROUP, TokenType.ORDER,
            TokenType.HAVING, TokenType.LIMIT, TokenType.JOIN,
        ):
            # Implicit alias (no AS keyword)
            next_t = self.peek()
            if next_t.value and next_t.value.lower() not in KEYWORDS:
                alias = self.advance().value

        return SelectExpr(expr=expr, alias=alias)

    def _parse_table_ref(self) -> TableRef:
        name = self.expect(TokenType.IDENT).value
        alias = None
        if self.match(TokenType.AS):
            alias = self.expect(TokenType.IDENT).value
        elif self.peek().type == TokenType.IDENT and self.peek().value.lower() not in KEYWORDS:
            alias = self.advance().value
        return TableRef(table_name=name, alias=alias)

    def _try_parse_join(self) -> Optional[JoinClause]:
        join_type = 'inner'
        if self.match(TokenType.INNER):
            join_type = 'inner'
        elif self.match(TokenType.LEFT):
            self.match(TokenType.OUTER)
            join_type = 'left'
        elif self.match(TokenType.CROSS):
            join_type = 'cross'
        elif self.peek().type == TokenType.JOIN:
            pass  # default inner
        elif self.match(TokenType.COMMA):
            # Implicit cross join (FROM a, b)
            table = self._parse_table_ref()
            return JoinClause(join_type='cross', table=table, condition=None)
        else:
            return None

        if not self.match(TokenType.JOIN):
            return None

        table = self._parse_table_ref()
        condition = None
        if self.match(TokenType.ON):
            condition = self._parse_expr()

        return JoinClause(join_type=join_type, table=table, condition=condition)

    def _parse_order_by_list(self) -> List[Tuple[Any, bool]]:
        items = []
        expr = self._parse_expr()
        asc = True
        if self.match(TokenType.ASC):
            asc = True
        elif self.match(TokenType.DESC):
            asc = False
        items.append((expr, asc))
        while self.match(TokenType.COMMA):
            expr = self._parse_expr()
            asc = True
            if self.match(TokenType.ASC):
                asc = True
            elif self.match(TokenType.DESC):
                asc = False
            items.append((expr, asc))
        return items

    def _parse_insert(self) -> InsertStmt:
        self.expect(TokenType.INSERT)
        self.expect(TokenType.INTO)
        table_name = self.expect(TokenType.IDENT).value

        columns = None
        if self.peek().type == TokenType.LPAREN:
            # Peek ahead to check if this is a column list or VALUES
            self.advance()  # consume '('
            # It's a column list
            if self.peek().type == TokenType.IDENT:
                columns = [self.expect(TokenType.IDENT).value]
                while self.match(TokenType.COMMA):
                    columns.append(self.expect(TokenType.IDENT).value)
                self.expect(TokenType.RPAREN)
            else:
                # Not a column list, push back
                self.pos -= 1

        self.expect(TokenType.VALUES)
        values_list = []
        values_list.append(self._parse_value_tuple())
        while self.match(TokenType.COMMA):
            values_list.append(self._parse_value_tuple())

        return InsertStmt(table_name=table_name, columns=columns, values_list=values_list)

    def _parse_value_tuple(self) -> List[Any]:
        self.expect(TokenType.LPAREN)
        vals = [self._parse_expr()]
        while self.match(TokenType.COMMA):
            vals.append(self._parse_expr())
        self.expect(TokenType.RPAREN)
        return vals

    def _parse_update(self) -> UpdateStmt:
        self.expect(TokenType.UPDATE)
        table_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.SET)

        assignments = []
        col = self.expect(TokenType.IDENT).value
        self.expect(TokenType.EQ)
        val = self._parse_expr()
        assignments.append((col, val))
        while self.match(TokenType.COMMA):
            col = self.expect(TokenType.IDENT).value
            self.expect(TokenType.EQ)
            val = self._parse_expr()
            assignments.append((col, val))

        where = None
        if self.match(TokenType.WHERE):
            where = self._parse_expr()

        return UpdateStmt(table_name=table_name, assignments=assignments, where=where)

    def _parse_delete(self) -> DeleteStmt:
        self.expect(TokenType.DELETE)
        self.expect(TokenType.FROM)
        table_name = self.expect(TokenType.IDENT).value

        where = None
        if self.match(TokenType.WHERE):
            where = self._parse_expr()

        return DeleteStmt(table_name=table_name, where=where)

    def _parse_create(self):
        self.expect(TokenType.CREATE)
        if self.peek().type == TokenType.INDEX:
            return self._parse_create_index()
        self.expect(TokenType.TABLE)
        if_not_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True
        table_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        columns = self._parse_column_defs()
        self.expect(TokenType.RPAREN)
        return CreateTableStmt(table_name=table_name, columns=columns, if_not_exists=if_not_exists)

    def _parse_column_defs(self) -> List[ColumnDef]:
        cols = [self._parse_column_def()]
        while self.match(TokenType.COMMA):
            # Check for table-level PRIMARY KEY
            if self.peek().type == TokenType.PRIMARY:
                self.advance()
                self.expect(TokenType.KEY)
                self.expect(TokenType.LPAREN)
                pk_col = self.expect(TokenType.IDENT).value
                self.expect(TokenType.RPAREN)
                for c in cols:
                    if c.name == pk_col:
                        c.primary_key = True
                        c.not_null = True
                continue
            cols.append(self._parse_column_def())
        return cols

    def _parse_column_def(self) -> ColumnDef:
        name = self.expect(TokenType.IDENT).value
        col_type = self._parse_type()
        pk = False
        unique = False
        not_null = False
        default = None

        while True:
            if self.match(TokenType.PRIMARY):
                self.expect(TokenType.KEY)
                pk = True
                not_null = True
            elif self.match(TokenType.UNIQUE):
                unique = True
            elif self.match(TokenType.NOT):
                if self.peek().type == TokenType.NULL:
                    self.advance()
                    not_null = True
                else:
                    self.pos -= 1
                    break
            elif self.match(TokenType.DEFAULT):
                default = self._parse_primary()
            else:
                break

        return ColumnDef(name=name, col_type=col_type, primary_key=pk,
                        unique=unique, not_null=not_null, default=default)

    def _parse_type(self) -> str:
        t = self.advance()
        if t.type in (TokenType.INT, TokenType.TEXT, TokenType.FLOAT, TokenType.BOOL):
            # Handle VARCHAR(n)
            if self.match(TokenType.LPAREN):
                self.expect(TokenType.NUMBER)
                self.expect(TokenType.RPAREN)
            return t.type.name.lower()
        elif t.type == TokenType.IDENT:
            return t.value.lower()
        raise ParseError(f"Expected type, got {t.type.name}")

    def _parse_create_index(self) -> CreateIndexStmt:
        self.expect(TokenType.INDEX)
        index_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ON)
        table_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        column = self.expect(TokenType.IDENT).value
        self.expect(TokenType.RPAREN)
        return CreateIndexStmt(index_name=index_name, table_name=table_name, column=column)

    def _parse_drop(self) -> DropTableStmt:
        self.expect(TokenType.DROP)
        self.expect(TokenType.TABLE)
        if_exists = False
        if self.match(TokenType.IF):
            self.expect(TokenType.EXISTS)
            if_exists = True
        table_name = self.expect(TokenType.IDENT).value
        return DropTableStmt(table_name=table_name, if_exists=if_exists)

    # Expression parsing (precedence climbing)
    def _parse_expr(self) -> Any:
        return self._parse_or()

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self.match(TokenType.OR):
            right = self._parse_and()
            left = SqlLogic(op='or', operands=[left, right])
        return left

    def _parse_and(self) -> Any:
        left = self._parse_not()
        while self.match(TokenType.AND):
            right = self._parse_not()
            left = SqlLogic(op='and', operands=[left, right])
        return left

    def _parse_not(self) -> Any:
        if self.match(TokenType.NOT):
            expr = self._parse_not()
            return SqlLogic(op='not', operands=[expr])
        return self._parse_comparison()

    def _parse_comparison(self) -> Any:
        left = self._parse_addition()

        # IS [NOT] NULL
        if self.peek().type == TokenType.IS:
            self.advance()
            negated = bool(self.match(TokenType.NOT))
            self.expect(TokenType.NULL)
            return SqlIsNull(expr=left, negated=negated)

        # NOT IN, NOT LIKE, NOT BETWEEN
        if self.peek().type == TokenType.NOT:
            self.advance()
            if self.peek().type == TokenType.IN:
                self.advance()
                self.expect(TokenType.LPAREN)
                vals = [self._parse_expr()]
                while self.match(TokenType.COMMA):
                    vals.append(self._parse_expr())
                self.expect(TokenType.RPAREN)
                return SqlLogic(op='not', operands=[SqlInList(expr=left, values=vals)])
            elif self.peek().type == TokenType.LIKE:
                self.advance()
                pattern = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlComparison(op='like', left=left, right=pattern)])
            elif self.peek().type == TokenType.BETWEEN:
                self.advance()
                low = self._parse_addition()
                self.expect(TokenType.AND)
                high = self._parse_addition()
                return SqlLogic(op='not', operands=[SqlBetween(expr=left, low=low, high=high)])
            else:
                self.pos -= 1  # push NOT back

        # IN
        if self.match(TokenType.IN):
            self.expect(TokenType.LPAREN)
            vals = [self._parse_expr()]
            while self.match(TokenType.COMMA):
                vals.append(self._parse_expr())
            self.expect(TokenType.RPAREN)
            return SqlInList(expr=left, values=vals)

        # BETWEEN
        if self.match(TokenType.BETWEEN):
            low = self._parse_addition()
            self.expect(TokenType.AND)
            high = self._parse_addition()
            return SqlBetween(expr=left, low=low, high=high)

        # LIKE
        if self.match(TokenType.LIKE):
            pattern = self._parse_addition()
            return SqlComparison(op='like', left=left, right=pattern)

        # Standard comparison operators
        op_map = {
            TokenType.EQ: '=', TokenType.NE: '!=', TokenType.LT: '<',
            TokenType.LE: '<=', TokenType.GT: '>', TokenType.GE: '>=',
        }
        for tt, op_str in op_map.items():
            if self.match(tt):
                right = self._parse_addition()
                return SqlComparison(op=op_str, left=left, right=right)

        return left

    def _parse_addition(self) -> Any:
        left = self._parse_multiplication()
        while True:
            if self.match(TokenType.PLUS):
                right = self._parse_multiplication()
                left = SqlBinOp(op='+', left=left, right=right)
            elif self.match(TokenType.MINUS):
                right = self._parse_multiplication()
                left = SqlBinOp(op='-', left=left, right=right)
            else:
                break
        return left

    def _parse_multiplication(self) -> Any:
        left = self._parse_unary()
        while True:
            if self.match(TokenType.STAR):
                right = self._parse_unary()
                left = SqlBinOp(op='*', left=left, right=right)
            elif self.match(TokenType.SLASH):
                right = self._parse_unary()
                left = SqlBinOp(op='/', left=left, right=right)
            else:
                break
        return left

    def _parse_unary(self) -> Any:
        if self.match(TokenType.MINUS):
            expr = self._parse_primary()
            return SqlBinOp(op='*', left=SqlLiteral(-1), right=expr)
        return self._parse_primary()

    def _parse_primary(self) -> Any:
        t = self.peek()

        # CASE expression
        if t.type == TokenType.CASE:
            return self._parse_case()

        # Number
        if t.type == TokenType.NUMBER:
            self.advance()
            return SqlLiteral(t.value)

        # String
        if t.type == TokenType.STRING:
            self.advance()
            return SqlLiteral(t.value)

        # NULL
        if t.type == TokenType.NULL:
            self.advance()
            return SqlLiteral(None)

        # TRUE / FALSE
        if t.type == TokenType.TRUE:
            self.advance()
            return SqlLiteral(True)
        if t.type == TokenType.FALSE:
            self.advance()
            return SqlLiteral(False)

        # Star (for count(*))
        if t.type == TokenType.STAR:
            self.advance()
            return SqlStar()

        # Parenthesized expression
        if t.type == TokenType.LPAREN:
            self.advance()
            expr = self._parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        # Aggregate functions
        if t.type in (TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX):
            return self._parse_agg_func()

        # Identifier (column ref, table.column, or function call)
        if t.type == TokenType.IDENT:
            name = self.advance().value
            # Function call
            if self.peek().type == TokenType.LPAREN:
                self.advance()  # consume '('
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self._parse_expr())
                    while self.match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self.expect(TokenType.RPAREN)
                return SqlFuncCall(func_name=name.upper(), args=args)
            # table.column
            if self.match(TokenType.DOT):
                col = self.advance().value
                return SqlColumnRef(table=name, column=col)
            return SqlColumnRef(table=None, column=name)

        raise ParseError(f"Unexpected token {t.type.name} ({t.value!r}) at pos {t.pos}")

    def _parse_agg_func(self) -> SqlAggCall:
        func_tok = self.advance()
        func_name = func_tok.type.name.lower()
        self.expect(TokenType.LPAREN)
        distinct = bool(self.match(TokenType.DISTINCT))

        if func_name == 'count' and self.peek().type == TokenType.STAR:
            self.advance()
            arg = None
        else:
            arg = self._parse_expr()

        self.expect(TokenType.RPAREN)
        return SqlAggCall(func=func_name, arg=arg, distinct=distinct)

    def _parse_case(self) -> SqlCase:
        self.expect(TokenType.CASE)
        whens = []
        while self.match(TokenType.WHEN):
            cond = self._parse_expr()
            self.expect(TokenType.THEN)
            result = self._parse_expr()
            whens.append((cond, result))
        else_result = None
        if self.match(TokenType.ELSE):
            else_result = self._parse_expr()
        self.expect(TokenType.END)
        return SqlCase(whens=whens, else_result=else_result)


def parse_sql(sql: str) -> Any:
    lexer = Lexer(sql)
    parser = Parser(lexer.tokens)
    return parser.parse()


def parse_sql_multi(sql: str) -> List[Any]:
    lexer = Lexer(sql)
    parser = Parser(lexer.tokens)
    return parser.parse_multi()


# =============================================================================
# Schema Catalog
# =============================================================================

@dataclass
class TableSchema:
    name: str
    columns: List[ColumnDef]
    indexes: Dict[str, str] = field(default_factory=dict)  # index_name -> column
    next_rowid: int = 1

    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnDef]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def primary_key_column(self) -> Optional[str]:
        for c in self.columns:
            if c.primary_key:
                return c.name
        return None


class CatalogError(Exception):
    pass


class Catalog:
    """Schema registry for tables and indexes."""

    def __init__(self):
        self.tables: Dict[str, TableSchema] = {}

    def create_table(self, name: str, columns: List[ColumnDef],
                     if_not_exists: bool = False) -> TableSchema:
        if name in self.tables:
            if if_not_exists:
                return self.tables[name]
            raise CatalogError(f"Table '{name}' already exists")
        schema = TableSchema(name=name, columns=columns)
        self.tables[name] = schema
        return schema

    def drop_table(self, name: str, if_exists: bool = False):
        if name not in self.tables:
            if if_exists:
                return
            raise CatalogError(f"Table '{name}' does not exist")
        del self.tables[name]

    def get_table(self, name: str) -> TableSchema:
        if name not in self.tables:
            raise CatalogError(f"Table '{name}' does not exist")
        return self.tables[name]

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def list_tables(self) -> List[str]:
        return sorted(self.tables.keys())


# =============================================================================
# Storage Engine (bridges C244 Buffer Pool with row storage)
# =============================================================================

class StorageEngine:
    """Row storage backed by C244 Buffer Pool and C246 Transaction Manager.

    Row storage strategy:
    - Each row is stored as a KV pair in the TransactionManager
    - Key format: "{table}.{rowid}" (enables prefix scans)
    - Value: dict of column values
    - Buffer pool manages page-level caching of frequently accessed data
    """

    def __init__(self, pool_size: int = 64, page_size: int = 4096,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        self.disk_manager = DiskManager(page_size=page_size)
        self.buffer_pool = BufferPoolManager(
            pool_size=pool_size,
            disk_manager=self.disk_manager,
            policy=EvictionPolicy.LRU
        )
        self.txn_manager = TransactionManager(
            isolation=isolation,
            lock_timeout=5.0,
            enable_deadlock_detection=True,
        )
        self.catalog = Catalog()
        # Track auto-increment per table
        self._auto_ids: Dict[str, int] = {}

    def _row_key(self, table: str, rowid: int) -> str:
        return f"{table}.{rowid}"

    def _table_prefix(self, table: str) -> str:
        return f"{table}."

    def begin(self) -> int:
        return self.txn_manager.begin()

    def commit(self, txn_id: int) -> bool:
        return self.txn_manager.commit(txn_id)

    def abort(self, txn_id: int):
        self.txn_manager.abort(txn_id)

    def savepoint(self, txn_id: int, name: str):
        self.txn_manager.savepoint(txn_id, name)

    def rollback_to_savepoint(self, txn_id: int, name: str):
        self.txn_manager.rollback_to_savepoint(txn_id, name)

    def insert_row(self, txn_id: int, table: str, row_data: Dict[str, Any]) -> int:
        """Insert a row, returns rowid."""
        schema = self.catalog.get_table(table)
        pk_col = schema.primary_key_column()

        # Determine rowid
        if pk_col and pk_col in row_data and row_data[pk_col] is not None:
            rowid = row_data[pk_col]
            # Check uniqueness
            key = self._row_key(table, rowid)
            existing = self.txn_manager.get(txn_id, key)
            if existing is not None:
                raise CatalogError(f"Duplicate primary key: {rowid}")
        else:
            # Auto-assign
            if table not in self._auto_ids:
                self._auto_ids[table] = schema.next_rowid
            rowid = self._auto_ids[table]
            self._auto_ids[table] = rowid + 1
            schema.next_rowid = self._auto_ids[table]
            if pk_col:
                row_data[pk_col] = rowid

        # Apply defaults and validate
        full_row = {}
        for col_def in schema.columns:
            if col_def.name in row_data:
                full_row[col_def.name] = row_data[col_def.name]
            elif col_def.default is not None:
                if isinstance(col_def.default, SqlLiteral):
                    full_row[col_def.name] = col_def.default.value
                else:
                    full_row[col_def.name] = col_def.default
            elif col_def.not_null:
                raise CatalogError(f"Column '{col_def.name}' cannot be NULL")
            else:
                full_row[col_def.name] = None

        # Validate NOT NULL
        for col_def in schema.columns:
            if col_def.not_null and full_row.get(col_def.name) is None:
                raise CatalogError(f"Column '{col_def.name}' cannot be NULL")

        # Validate UNIQUE constraints
        if any(c.unique for c in schema.columns):
            self._check_unique_constraints(txn_id, table, schema, full_row, exclude_rowid=None)

        key = self._row_key(table, rowid)
        self.txn_manager.put(txn_id, key, full_row)
        return rowid

    def _check_unique_constraints(self, txn_id: int, table: str, schema: TableSchema,
                                   row_data: Dict[str, Any], exclude_rowid: Any = None):
        """Check unique constraints against existing rows."""
        unique_cols = [c for c in schema.columns if c.unique and not c.primary_key]
        if not unique_cols:
            return
        existing = self.txn_manager.scan(txn_id, self._table_prefix(table))
        for key, existing_row in existing.items():
            rid = key.split('.', 1)[1]
            if exclude_rowid is not None and str(rid) == str(exclude_rowid):
                continue
            for col_def in unique_cols:
                if (col_def.name in row_data and col_def.name in existing_row and
                    row_data[col_def.name] is not None and
                    row_data[col_def.name] == existing_row[col_def.name]):
                    raise CatalogError(
                        f"UNIQUE constraint failed: {table}.{col_def.name} = {row_data[col_def.name]!r}")

    def get_row(self, txn_id: int, table: str, rowid: int) -> Optional[Dict[str, Any]]:
        key = self._row_key(table, rowid)
        return self.txn_manager.get(txn_id, key)

    def update_row(self, txn_id: int, table: str, rowid: int,
                   updates: Dict[str, Any]):
        key = self._row_key(table, rowid)
        row = self.txn_manager.get(txn_id, key)
        if row is None:
            return
        schema = self.catalog.get_table(table)
        row = dict(row)  # copy to avoid mutating MVCC committed version
        row.update(updates)

        # Validate NOT NULL
        for col_def in schema.columns:
            if col_def.not_null and row.get(col_def.name) is None:
                raise CatalogError(f"Column '{col_def.name}' cannot be NULL")

        # Validate UNIQUE
        if any(c.unique for c in schema.columns):
            pk = schema.primary_key_column()
            rid = row.get(pk, rowid) if pk else rowid
            self._check_unique_constraints(txn_id, table, schema, row, exclude_rowid=rid)

        self.txn_manager.put(txn_id, key, row)

    def delete_row(self, txn_id: int, table: str, rowid: int) -> bool:
        key = self._row_key(table, rowid)
        return self.txn_manager.delete(txn_id, key)

    def scan_table(self, txn_id: int, table: str) -> List[Tuple[int, Dict[str, Any]]]:
        """Returns list of (rowid, row_data) for all rows in table."""
        prefix = self._table_prefix(table)
        rows = self.txn_manager.scan(txn_id, prefix)
        result = []
        for key, value in rows.items():
            parts = key.split('.', 1)
            if len(parts) == 2:
                try:
                    rid = int(parts[1])
                except ValueError:
                    rid = parts[1]
                result.append((rid, value))
        return result

    def count_rows(self, txn_id: int, table: str) -> int:
        prefix = self._table_prefix(table)
        return self.txn_manager.count(txn_id, prefix)

    def stats(self) -> Dict[str, Any]:
        bp_stats = self.buffer_pool.get_stats()
        txn_stats = self.txn_manager.stats()
        return {
            'buffer_pool': {
                'hits': bp_stats.hits,
                'misses': bp_stats.misses,
                'hit_rate': bp_stats.hit_rate,
                'evictions': bp_stats.evictions,
            },
            'transactions': txn_stats,
            'tables': len(self.catalog.tables),
        }


# =============================================================================
# Query Compiler (SQL AST -> C245 Operator Tree)
# =============================================================================

class CompileError(Exception):
    pass


class QueryCompiler:
    """Compiles SQL AST into C245 query execution operator trees."""

    def __init__(self, storage: StorageEngine):
        self.storage = storage

    def compile_select(self, stmt: SelectStmt, txn_id: int) -> Tuple[Operator, ExecutionEngine]:
        """Compile a SELECT statement into an operator tree."""
        # Build a QEDatabase with table data from the transactional store
        qe_db = QEDatabase()
        table_aliases = {}  # alias -> real table name

        # Collect all referenced tables
        tables_needed = set()
        if stmt.from_table:
            tables_needed.add(stmt.from_table.table_name)
            if stmt.from_table.alias:
                table_aliases[stmt.from_table.alias] = stmt.from_table.table_name
        for j in stmt.joins:
            tables_needed.add(j.table.table_name)
            if j.table.alias:
                table_aliases[j.table.alias] = j.table.table_name

        # Load table data into QEDatabase
        for tname in tables_needed:
            schema = self.storage.catalog.get_table(tname)
            qe_table = qe_db.create_table(tname, schema.column_names())
            rows = self.storage.scan_table(txn_id, tname)
            for _rid, row_data in rows:
                qe_table.insert(row_data)

        engine = ExecutionEngine(qe_db)

        # Build the operator tree
        plan = self._build_plan(stmt, qe_db, table_aliases)
        return plan, engine

    def _build_plan(self, stmt: SelectStmt, qe_db: QEDatabase,
                    table_aliases: Dict[str, str]) -> Operator:
        # Start with FROM clause
        if stmt.from_table is None:
            # SELECT without FROM (e.g., SELECT 1+1)
            from query_executor import MaterializeOp
            # Create a dummy single-row source
            dummy_table = qe_db.create_table('__dual', ['__dummy'])
            dummy_table.insert({'__dummy': 1})
            plan = SeqScanOp(dummy_table)
        else:
            tname = stmt.from_table.table_name
            plan = SeqScanOp(qe_db.get_table(tname))

        # JOINs
        for j in stmt.joins:
            right_table = qe_db.get_table(j.table.table_name)
            right_scan = SeqScanOp(right_table)

            if j.join_type == 'cross':
                plan = NestedLoopJoinOp(plan, right_scan, predicate=None, join_type='cross')
            elif j.condition is not None:
                # Convert SQL condition to C245 expression
                qe_pred = self._sql_to_qe_expr(j.condition)
                if j.join_type == 'inner':
                    plan = HashJoinOp(
                        plan, right_scan,
                        left_key=self._extract_join_key(j.condition, 'left'),
                        right_key=self._extract_join_key(j.condition, 'right'),
                        join_type='inner'
                    ) if self._is_equijoin(j.condition) else NestedLoopJoinOp(
                        plan, right_scan, predicate=qe_pred, join_type='inner'
                    )
                elif j.join_type == 'left':
                    if self._is_equijoin(j.condition):
                        plan = HashJoinOp(
                            plan, right_scan,
                            left_key=self._extract_join_key(j.condition, 'left'),
                            right_key=self._extract_join_key(j.condition, 'right'),
                            join_type='left'
                        )
                    else:
                        plan = NestedLoopJoinOp(
                            plan, right_scan, predicate=qe_pred, join_type='left'
                        )
            else:
                plan = NestedLoopJoinOp(plan, right_scan, predicate=None, join_type=j.join_type)

        # WHERE
        if stmt.where is not None:
            qe_pred = self._sql_to_qe_expr(stmt.where)
            plan = FilterOp(plan, qe_pred)

        # GROUP BY + aggregates
        has_aggs = self._has_aggregates(stmt.columns)
        if stmt.group_by or has_aggs:
            group_exprs = []
            if stmt.group_by:
                group_exprs = [self._sql_to_qe_expr(g) for g in stmt.group_by]

            agg_calls = self._extract_aggregates(stmt.columns)
            plan = HashAggregateOp(plan, group_exprs, agg_calls)

            # HAVING
            if stmt.having:
                having_pred = self._sql_to_qe_expr(stmt.having)
                plan = HavingOp(plan, having_pred)

        # ORDER BY (before projection so sort keys can reference non-projected cols)
        if stmt.order_by:
            sort_keys = [(self._sql_to_qe_expr(expr), asc) for expr, asc in stmt.order_by]
            plan = SortOp(plan, sort_keys)

        # SELECT (projection)
        if not self._is_star_only(stmt.columns):
            projections = self._build_projections(stmt.columns, stmt.group_by, has_aggs)
            if projections:
                plan = ProjectOp(plan, projections)

        # DISTINCT (after projection)
        if stmt.distinct:
            plan = DistinctOp(plan)

        # LIMIT / OFFSET
        if stmt.limit is not None:
            plan = LimitOp(plan, stmt.limit, stmt.offset or 0)

        return plan

    def _is_star_only(self, columns: List[SelectExpr]) -> bool:
        return len(columns) == 1 and isinstance(columns[0].expr, SqlStar) and columns[0].expr.table is None

    def _has_aggregates(self, columns: List[SelectExpr]) -> bool:
        for col in columns:
            if isinstance(col.expr, SqlAggCall):
                return True
        return False

    def _extract_aggregates(self, columns: List[SelectExpr]) -> List[AggCall]:
        aggs = []
        for col in columns:
            if isinstance(col.expr, SqlAggCall):
                func_map = {
                    'count': AggFunc.COUNT_STAR if col.expr.arg is None else AggFunc.COUNT,
                    'sum': AggFunc.SUM, 'avg': AggFunc.AVG,
                    'min': AggFunc.MIN, 'max': AggFunc.MAX,
                }
                func = func_map.get(col.expr.func, AggFunc.COUNT)
                arg_expr = None
                if col.expr.arg is not None:
                    arg_expr = self._sql_to_qe_expr(col.expr.arg)
                alias = col.alias or self._agg_alias(col.expr)
                aggs.append(AggCall(func=func, column=arg_expr,
                                   distinct=col.expr.distinct, alias=alias))
        return aggs

    @staticmethod
    def _agg_alias(agg: SqlAggCall) -> str:
        """Deterministic alias for an aggregate call."""
        if agg.arg is None:
            return f"{agg.func}_star"
        if isinstance(agg.arg, SqlColumnRef):
            col_name = agg.arg.column
            return f"{agg.func}_{col_name}"
        return f"{agg.func}_expr"

    def _build_projections(self, columns: List[SelectExpr],
                           group_by: Optional[List], has_aggs: bool
                           ) -> List[Tuple[Any, str]]:
        projections = []
        used_aliases = set()
        for col in columns:
            if isinstance(col.expr, SqlStar):
                continue  # handled separately
            if isinstance(col.expr, SqlAggCall):
                alias = col.alias or self._agg_alias(col.expr)
                projections.append((ColumnRef(None, alias), alias))
                used_aliases.add(alias)
            else:
                qe_expr = self._sql_to_qe_expr(col.expr)
                alias = col.alias
                if alias is None:
                    if isinstance(col.expr, SqlColumnRef):
                        if col.expr.table:
                            alias = f"{col.expr.table}.{col.expr.column}"
                        else:
                            alias = col.expr.column
                    else:
                        alias = f"expr_{len(projections)}"
                # Deduplicate aliases
                base = alias
                counter = 1
                while alias in used_aliases:
                    alias = f"{base}_{counter}"
                    counter += 1
                used_aliases.add(alias)
                projections.append((qe_expr, alias))
        return projections

    def _sql_to_qe_expr(self, node) -> Any:
        """Convert SQL AST expression to C245 query executor expression."""
        if isinstance(node, SqlColumnRef):
            return ColumnRef(node.table, node.column)

        if isinstance(node, SqlLiteral):
            return Literal(node.value)

        if isinstance(node, SqlBinOp):
            left = self._sql_to_qe_expr(node.left)
            right = self._sql_to_qe_expr(node.right)
            return ArithExpr(node.op, left, right)

        if isinstance(node, SqlComparison):
            left = self._sql_to_qe_expr(node.left)
            right = self._sql_to_qe_expr(node.right)
            op_map = {
                '=': CompOp.EQ, '!=': CompOp.NE, '<': CompOp.LT,
                '<=': CompOp.LE, '>': CompOp.GT, '>=': CompOp.GE,
                'like': CompOp.LIKE,
            }
            return Comparison(op_map[node.op], left, right)

        if isinstance(node, SqlLogic):
            if node.op == 'not':
                operand = self._sql_to_qe_expr(node.operands[0])
                return LogicExpr(LogicOp.NOT, [operand])
            operands = [self._sql_to_qe_expr(o) for o in node.operands]
            op_map = {'and': LogicOp.AND, 'or': LogicOp.OR}
            return LogicExpr(op_map[node.op], operands)

        if isinstance(node, SqlIsNull):
            expr = self._sql_to_qe_expr(node.expr)
            if node.negated:
                return Comparison(CompOp.IS_NOT_NULL, expr, Literal(None))
            return Comparison(CompOp.IS_NULL, expr, Literal(None))

        if isinstance(node, SqlBetween):
            # Desugar: x BETWEEN low AND high -> x >= low AND x <= high
            expr = self._sql_to_qe_expr(node.expr)
            low = self._sql_to_qe_expr(node.low)
            high = self._sql_to_qe_expr(node.high)
            return LogicExpr(LogicOp.AND, [
                Comparison(CompOp.GE, expr, low),
                Comparison(CompOp.LE, expr, high),
            ])

        if isinstance(node, SqlInList):
            # Desugar: x IN (a, b, c) -> x = a OR x = b OR x = c
            expr = self._sql_to_qe_expr(node.expr)
            if len(node.values) == 1:
                return Comparison(CompOp.EQ, expr, self._sql_to_qe_expr(node.values[0]))
            comparisons = [Comparison(CompOp.EQ, expr, self._sql_to_qe_expr(v))
                          for v in node.values]
            result = comparisons[0]
            for c in comparisons[1:]:
                result = LogicExpr(LogicOp.OR, [result, c])
            return result

        if isinstance(node, SqlFuncCall):
            args = [self._sql_to_qe_expr(a) for a in node.args]
            return FuncExpr(node.func_name, args)

        if isinstance(node, SqlAggCall):
            # Reference aggregate by deterministic alias
            alias = self._agg_alias(node)
            return ColumnRef(None, alias)

        if isinstance(node, SqlCase):
            whens = [(self._sql_to_qe_expr(c), self._sql_to_qe_expr(r))
                     for c, r in node.whens]
            else_r = self._sql_to_qe_expr(node.else_result) if node.else_result else None
            from query_executor import CaseExpr
            return CaseExpr(whens=whens, else_result=else_r)

        if isinstance(node, SqlStar):
            return ColumnRef(None, '*')

        raise CompileError(f"Cannot convert SQL expr: {type(node).__name__}")

    def _is_equijoin(self, condition) -> bool:
        """Check if join condition is a simple equality (a.x = b.y)."""
        if isinstance(condition, SqlComparison) and condition.op == '=':
            return (isinstance(condition.left, SqlColumnRef) and
                    isinstance(condition.right, SqlColumnRef))
        return False

    def _extract_join_key(self, condition, side: str):
        """Extract join key expression for hash join."""
        if side == 'left':
            return self._sql_to_qe_expr(condition.left)
        return self._sql_to_qe_expr(condition.right)


# =============================================================================
# Mini Database Engine (unified facade)
# =============================================================================

class DatabaseError(Exception):
    pass


class MiniDB:
    """Complete relational database engine.

    Composes:
    - C244 Buffer Pool for page management
    - C245 Query Executor for query processing
    - C246 Transaction Manager for ACID transactions
    """

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        self.storage = StorageEngine(pool_size=pool_size, isolation=isolation)
        self.compiler = QueryCompiler(self.storage)
        self._active_txn: Optional[int] = None
        self._autocommit = True

    def execute(self, sql: str) -> 'ResultSet':
        """Execute a SQL statement and return results."""
        stmt = parse_sql(sql)
        return self._execute_stmt(stmt)

    def execute_many(self, sql: str) -> List['ResultSet']:
        """Execute multiple SQL statements separated by semicolons."""
        stmts = parse_sql_multi(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_stmt(stmt))
        return results

    def _execute_stmt(self, stmt) -> 'ResultSet':
        if isinstance(stmt, BeginStmt):
            return self._exec_begin()
        elif isinstance(stmt, CommitStmt):
            return self._exec_commit()
        elif isinstance(stmt, RollbackStmt):
            return self._exec_rollback(stmt)
        elif isinstance(stmt, SavepointStmt):
            return self._exec_savepoint(stmt)
        elif isinstance(stmt, CreateTableStmt):
            return self._exec_create_table(stmt)
        elif isinstance(stmt, DropTableStmt):
            return self._exec_drop_table(stmt)
        elif isinstance(stmt, CreateIndexStmt):
            return self._exec_create_index(stmt)
        elif isinstance(stmt, InsertStmt):
            return self._exec_insert(stmt)
        elif isinstance(stmt, SelectStmt):
            return self._exec_select(stmt)
        elif isinstance(stmt, UpdateStmt):
            return self._exec_update(stmt)
        elif isinstance(stmt, DeleteStmt):
            return self._exec_delete(stmt)
        elif isinstance(stmt, ShowTablesStmt):
            return self._exec_show_tables()
        elif isinstance(stmt, DescribeStmt):
            return self._exec_describe(stmt)
        elif isinstance(stmt, ExplainStmt):
            return self._exec_explain(stmt)
        else:
            raise DatabaseError(f"Unknown statement type: {type(stmt).__name__}")

    def _get_txn(self) -> int:
        """Get current transaction or start auto-commit transaction."""
        if self._active_txn is not None:
            return self._active_txn
        return self.storage.begin()

    def _auto_commit(self, txn_id: int):
        """Commit if in autocommit mode."""
        if self._active_txn is None:
            self.storage.commit(txn_id)

    def _auto_abort(self, txn_id: int):
        """Abort if in autocommit mode."""
        if self._active_txn is None:
            try:
                self.storage.abort(txn_id)
            except Exception:
                pass

    # -- Transaction control --

    def _exec_begin(self) -> 'ResultSet':
        if self._active_txn is not None:
            raise DatabaseError("Transaction already active")
        self._active_txn = self.storage.begin()
        self._autocommit = False
        return ResultSet(columns=[], rows=[], message="BEGIN")

    def _exec_commit(self) -> 'ResultSet':
        if self._active_txn is None:
            raise DatabaseError("No active transaction")
        self.storage.commit(self._active_txn)
        self._active_txn = None
        self._autocommit = True
        return ResultSet(columns=[], rows=[], message="COMMIT")

    def _exec_rollback(self, stmt: RollbackStmt) -> 'ResultSet':
        if self._active_txn is None:
            raise DatabaseError("No active transaction")
        if stmt.savepoint:
            self.storage.rollback_to_savepoint(self._active_txn, stmt.savepoint)
            return ResultSet(columns=[], rows=[], message=f"ROLLBACK TO {stmt.savepoint}")
        self.storage.abort(self._active_txn)
        self._active_txn = None
        self._autocommit = True
        return ResultSet(columns=[], rows=[], message="ROLLBACK")

    def _exec_savepoint(self, stmt: SavepointStmt) -> 'ResultSet':
        if self._active_txn is None:
            raise DatabaseError("No active transaction")
        self.storage.savepoint(self._active_txn, stmt.name)
        return ResultSet(columns=[], rows=[], message=f"SAVEPOINT {stmt.name}")

    # -- DDL --

    def _exec_create_table(self, stmt: CreateTableStmt) -> 'ResultSet':
        self.storage.catalog.create_table(
            stmt.table_name, stmt.columns, if_not_exists=stmt.if_not_exists
        )
        return ResultSet(columns=[], rows=[], message=f"CREATE TABLE {stmt.table_name}")

    def _exec_drop_table(self, stmt: DropTableStmt) -> 'ResultSet':
        self.storage.catalog.drop_table(stmt.table_name, if_exists=stmt.if_exists)
        # Also delete all rows
        txn_id = self._get_txn()
        try:
            prefix = self.storage._table_prefix(stmt.table_name)
            rows = self.storage.txn_manager.scan(txn_id, prefix)
            for key in rows:
                self.storage.txn_manager.delete(txn_id, key)
            self._auto_commit(txn_id)
        except Exception:
            self._auto_abort(txn_id)
            raise
        return ResultSet(columns=[], rows=[], message=f"DROP TABLE {stmt.table_name}")

    def _exec_create_index(self, stmt: CreateIndexStmt) -> 'ResultSet':
        schema = self.storage.catalog.get_table(stmt.table_name)
        schema.indexes[stmt.index_name] = stmt.column
        return ResultSet(columns=[], rows=[],
                        message=f"CREATE INDEX {stmt.index_name}")

    # -- DML --

    def _exec_insert(self, stmt: InsertStmt) -> 'ResultSet':
        schema = self.storage.catalog.get_table(stmt.table_name)
        txn_id = self._get_txn()
        try:
            count = 0
            for values in stmt.values_list:
                row_data = {}
                cols = stmt.columns or schema.column_names()
                for i, col in enumerate(cols):
                    if i < len(values):
                        val = values[i]
                        row_data[col] = self._eval_sql_value(val)
                    else:
                        row_data[col] = None
                self.storage.insert_row(txn_id, stmt.table_name, row_data)
                count += 1
            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"INSERT {count}",
                           rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_select(self, stmt: SelectStmt) -> 'ResultSet':
        txn_id = self._get_txn()
        try:
            plan, engine = self.compiler.compile_select(stmt, txn_id)
            qe_rows = engine.execute(plan)

            # Convert QE rows to result
            if qe_rows:
                # Determine column order from SELECT list or schema
                if self.compiler._is_star_only(stmt.columns):
                    # For SELECT *, use schema order from all referenced tables
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
                    # Use projected column names
                    columns = qe_rows[0].columns()
                    clean_cols = []
                    for c in columns:
                        if '.' in c:
                            clean_cols.append(c.split('.')[-1])
                        else:
                            clean_cols.append(c)
                    rows = [list(r.values()) for r in qe_rows]
            else:
                # Derive columns from SELECT list
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

            self._auto_commit(txn_id)
            return ResultSet(columns=clean_cols, rows=rows)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_update(self, stmt: UpdateStmt) -> 'ResultSet':
        txn_id = self._get_txn()
        try:
            schema = self.storage.catalog.get_table(stmt.table_name)
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0

            for rowid, row_data in all_rows:
                # Check WHERE condition
                if stmt.where is not None:
                    qe_row = Row(row_data)
                    qe_pred = self.compiler._sql_to_qe_expr(stmt.where)
                    from query_executor import eval_expr
                    if not eval_expr(qe_pred, qe_row):
                        continue

                # Apply assignments
                updates = {}
                for col, val_expr in stmt.assignments:
                    if isinstance(val_expr, SqlLiteral):
                        updates[col] = val_expr.value
                    else:
                        # Evaluate expression against current row
                        qe_row = Row(row_data)
                        qe_expr = self.compiler._sql_to_qe_expr(val_expr)
                        from query_executor import eval_expr
                        updates[col] = eval_expr(qe_expr, qe_row)
                self.storage.update_row(txn_id, stmt.table_name, rowid, updates)
                count += 1

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"UPDATE {count}",
                           rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _exec_delete(self, stmt: DeleteStmt) -> 'ResultSet':
        txn_id = self._get_txn()
        try:
            all_rows = self.storage.scan_table(txn_id, stmt.table_name)
            count = 0

            for rowid, row_data in all_rows:
                if stmt.where is not None:
                    qe_row = Row(row_data)
                    qe_pred = self.compiler._sql_to_qe_expr(stmt.where)
                    from query_executor import eval_expr
                    if not eval_expr(qe_pred, qe_row):
                        continue
                self.storage.delete_row(txn_id, stmt.table_name, rowid)
                count += 1

            self._auto_commit(txn_id)
            return ResultSet(columns=[], rows=[], message=f"DELETE {count}",
                           rows_affected=count)
        except Exception:
            self._auto_abort(txn_id)
            raise

    def _eval_sql_value(self, node) -> Any:
        """Evaluate a SQL expression to a Python value (for INSERT/UPDATE constants)."""
        if isinstance(node, SqlLiteral):
            return node.value
        if isinstance(node, SqlBinOp):
            left = self._eval_sql_value(node.left)
            right = self._eval_sql_value(node.right)
            if node.op == '+': return left + right
            if node.op == '-': return left - right
            if node.op == '*': return left * right
            if node.op == '/': return left / right
        if isinstance(node, SqlColumnRef):
            # Return column name as-is (for expression context)
            return node.column
        # Fallback: try to evaluate via QE
        if isinstance(node, (int, float, str, bool)):
            return node
        if node is None:
            return None
        # Convert to QE expr and evaluate against empty row
        qe_expr = self.compiler._sql_to_qe_expr(node)
        from query_executor import eval_expr
        return eval_expr(qe_expr, Row({}))

    # -- Utility --

    def _exec_show_tables(self) -> 'ResultSet':
        tables = self.storage.catalog.list_tables()
        return ResultSet(columns=['table_name'],
                        rows=[[t] for t in tables])

    def _exec_describe(self, stmt: DescribeStmt) -> 'ResultSet':
        schema = self.storage.catalog.get_table(stmt.table_name)
        rows = []
        for col in schema.columns:
            constraints = []
            if col.primary_key:
                constraints.append('PRIMARY KEY')
            if col.unique:
                constraints.append('UNIQUE')
            if col.not_null:
                constraints.append('NOT NULL')
            rows.append([col.name, col.col_type, ', '.join(constraints),
                        repr(col.default) if col.default else ''])
        return ResultSet(columns=['column', 'type', 'constraints', 'default'],
                        rows=rows)

    def _exec_explain(self, stmt: ExplainStmt) -> 'ResultSet':
        if isinstance(stmt.stmt, SelectStmt):
            txn_id = self._get_txn()
            try:
                plan, engine = self.compiler.compile_select(stmt.stmt, txn_id)
                explanation = engine.explain(plan)
                self._auto_commit(txn_id)
                return ResultSet(columns=['plan'],
                               rows=[[line] for line in explanation.split('\n') if line.strip()])
            except Exception:
                self._auto_abort(txn_id)
                raise
        return ResultSet(columns=['plan'], rows=[['EXPLAIN not supported for this statement']])

    def stats(self) -> Dict[str, Any]:
        return self.storage.stats()

    def close(self):
        """Clean up resources."""
        if self._active_txn is not None:
            try:
                self.storage.abort(self._active_txn)
            except Exception:
                pass
            self._active_txn = None
        self.storage.buffer_pool.flush_all()


@dataclass
class ResultSet:
    """Query result container."""
    columns: List[str]
    rows: List[List[Any]]
    message: Optional[str] = None
    rows_affected: int = 0

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def __bool__(self):
        return len(self.rows) > 0 or self.message is not None

    def scalar(self) -> Any:
        """Return single value from first row, first column."""
        if self.rows and self.rows[0]:
            return self.rows[0][0]
        return None

    def column(self, name: str) -> List[Any]:
        """Return all values for a column by name."""
        if name not in self.columns:
            raise KeyError(f"Column '{name}' not found")
        idx = self.columns.index(name)
        return [row[idx] for row in self.rows]

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Return rows as list of dicts."""
        return [dict(zip(self.columns, row)) for row in self.rows]

    def __repr__(self):
        if self.message:
            return f"ResultSet(message={self.message!r})"
        return f"ResultSet(columns={self.columns}, {len(self.rows)} rows)"
