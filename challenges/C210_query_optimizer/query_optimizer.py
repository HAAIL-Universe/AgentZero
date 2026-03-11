"""
C210: Database Query Optimizer

A cost-based query optimizer for SQL queries. Implements:
- SQL parser (SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT, subqueries)
- Logical plan representation (relational algebra)
- Physical plan operators (hash join, merge join, nested loop, index scan, etc.)
- Cost model with table statistics (cardinality, selectivity, histograms)
- Join ordering via dynamic programming (System R style)
- Index selection and access path planning
- Plan transformations (predicate pushdown, projection pruning, etc.)
- EXPLAIN output

No external dependencies. Pure Python.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import math
import re


# ============================================================
# SQL AST
# ============================================================

class TokenType(Enum):
    # Keywords
    SELECT = auto(); FROM = auto(); WHERE = auto(); JOIN = auto()
    INNER = auto(); LEFT = auto(); RIGHT = auto(); OUTER = auto()
    CROSS = auto(); ON = auto(); AND = auto(); OR = auto(); NOT = auto()
    AS = auto(); GROUP = auto(); BY = auto(); ORDER = auto()
    ASC = auto(); DESC = auto(); LIMIT = auto(); OFFSET = auto()
    HAVING = auto(); DISTINCT = auto(); UNION = auto(); ALL = auto()
    IN = auto(); EXISTS = auto(); BETWEEN = auto(); LIKE = auto()
    IS = auto(); NULL = auto(); TRUE = auto(); FALSE = auto()
    CASE = auto(); WHEN = auto(); THEN = auto(); ELSE = auto(); END = auto()
    # Aggregate functions
    COUNT = auto(); SUM = auto(); AVG = auto(); MIN = auto(); MAX = auto()
    # Literals and identifiers
    IDENT = auto(); NUMBER = auto(); STRING = auto()
    # Operators
    STAR = auto(); COMMA = auto(); DOT = auto(); LPAREN = auto(); RPAREN = auto()
    EQ = auto(); NEQ = auto(); LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    PLUS = auto(); MINUS = auto(); SLASH = auto(); PERCENT = auto()
    # Special
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int = 0


KEYWORDS = {
    'select': TokenType.SELECT, 'from': TokenType.FROM, 'where': TokenType.WHERE,
    'join': TokenType.JOIN, 'inner': TokenType.INNER, 'left': TokenType.LEFT,
    'right': TokenType.RIGHT, 'outer': TokenType.OUTER, 'cross': TokenType.CROSS,
    'on': TokenType.ON, 'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
    'as': TokenType.AS, 'group': TokenType.GROUP, 'by': TokenType.BY,
    'order': TokenType.ORDER, 'asc': TokenType.ASC, 'desc': TokenType.DESC,
    'limit': TokenType.LIMIT, 'offset': TokenType.OFFSET, 'having': TokenType.HAVING,
    'distinct': TokenType.DISTINCT, 'union': TokenType.UNION, 'all': TokenType.ALL,
    'in': TokenType.IN, 'exists': TokenType.EXISTS, 'between': TokenType.BETWEEN,
    'like': TokenType.LIKE, 'is': TokenType.IS, 'null': TokenType.NULL,
    'true': TokenType.TRUE, 'false': TokenType.FALSE,
    'case': TokenType.CASE, 'when': TokenType.WHEN, 'then': TokenType.THEN,
    'else': TokenType.ELSE, 'end': TokenType.END,
    'count': TokenType.COUNT, 'sum': TokenType.SUM, 'avg': TokenType.AVG,
    'min': TokenType.MIN, 'max': TokenType.MAX,
}


class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def _read_string(self) -> str:
        quote = self.text[self.pos]
        self.pos += 1
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != quote:
            if self.text[self.pos] == '\\':
                self.pos += 1
            self.pos += 1
        result = self.text[start:self.pos]
        self.pos += 1  # closing quote
        return result

    def _read_number(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
            self.pos += 1
        return self.text[start:self.pos]

    def _read_ident(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            self.pos += 1
        return self.text[start:self.pos]

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.text):
            self._skip_ws()
            if self.pos >= len(self.text):
                break
            ch = self.text[self.pos]
            p = self.pos

            if ch in ("'", '"'):
                tokens.append(Token(TokenType.STRING, self._read_string(), p))
            elif ch.isdigit():
                tokens.append(Token(TokenType.NUMBER, self._read_number(), p))
            elif ch.isalpha() or ch == '_':
                word = self._read_ident()
                tt = KEYWORDS.get(word.lower(), TokenType.IDENT)
                tokens.append(Token(tt, word, p))
            elif ch == '*':
                tokens.append(Token(TokenType.STAR, '*', p)); self.pos += 1
            elif ch == ',':
                tokens.append(Token(TokenType.COMMA, ',', p)); self.pos += 1
            elif ch == '.':
                tokens.append(Token(TokenType.DOT, '.', p)); self.pos += 1
            elif ch == '(':
                tokens.append(Token(TokenType.LPAREN, '(', p)); self.pos += 1
            elif ch == ')':
                tokens.append(Token(TokenType.RPAREN, ')', p)); self.pos += 1
            elif ch == '+':
                tokens.append(Token(TokenType.PLUS, '+', p)); self.pos += 1
            elif ch == '-':
                tokens.append(Token(TokenType.MINUS, '-', p)); self.pos += 1
            elif ch == '/':
                tokens.append(Token(TokenType.SLASH, '/', p)); self.pos += 1
            elif ch == '%':
                tokens.append(Token(TokenType.PERCENT, '%', p)); self.pos += 1
            elif ch == '=' :
                tokens.append(Token(TokenType.EQ, '=', p)); self.pos += 1
            elif ch == '!' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '=':
                tokens.append(Token(TokenType.NEQ, '!=', p)); self.pos += 2
            elif ch == '<' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '>':
                tokens.append(Token(TokenType.NEQ, '<>', p)); self.pos += 2
            elif ch == '<' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '=':
                tokens.append(Token(TokenType.LTE, '<=', p)); self.pos += 2
            elif ch == '>' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '=':
                tokens.append(Token(TokenType.GTE, '>=', p)); self.pos += 2
            elif ch == '<':
                tokens.append(Token(TokenType.LT, '<', p)); self.pos += 1
            elif ch == '>':
                tokens.append(Token(TokenType.GT, '>', p)); self.pos += 1
            elif ch == ';':
                self.pos += 1  # skip semicolons
            else:
                raise SyntaxError(f"Unexpected character: {ch!r} at position {p}")

        tokens.append(Token(TokenType.EOF, '', self.pos))
        return tokens


# ============================================================
# SQL AST Nodes
# ============================================================

@dataclass
class ColumnRef:
    """A column reference like 'table.column' or just 'column'."""
    table: Optional[str]
    column: str

    def __repr__(self):
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column

    def __eq__(self, other):
        if not isinstance(other, ColumnRef):
            return NotImplemented
        return self.table == other.table and self.column == other.column

    def __hash__(self):
        return hash((self.table, self.column))


@dataclass
class Literal:
    value: Any  # int, float, str, bool, None

    def __repr__(self):
        if isinstance(self.value, str):
            return f"'{self.value}'"
        if self.value is None:
            return 'NULL'
        return str(self.value)


@dataclass
class BinExpr:
    op: str  # =, !=, <, >, <=, >=, +, -, *, /, %, AND, OR, LIKE
    left: Any
    right: Any

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


@dataclass
class UnaryExpr:
    op: str  # NOT, -, IS NULL, IS NOT NULL
    operand: Any

    def __repr__(self):
        return f"({self.op} {self.operand})"


@dataclass
class FuncCall:
    name: str  # COUNT, SUM, AVG, MIN, MAX, or user func
    args: list
    distinct: bool = False

    def __repr__(self):
        d = "DISTINCT " if self.distinct else ""
        return f"{self.name}({d}{', '.join(str(a) for a in self.args)})"


@dataclass
class StarExpr:
    table: Optional[str] = None

    def __repr__(self):
        if self.table:
            return f"{self.table}.*"
        return "*"


@dataclass
class InExpr:
    expr: Any
    values: list
    negated: bool = False

    def __repr__(self):
        n = " NOT" if self.negated else ""
        return f"({self.expr}{n} IN ({', '.join(str(v) for v in self.values)}))"


@dataclass
class BetweenExpr:
    expr: Any
    low: Any
    high: Any
    negated: bool = False

    def __repr__(self):
        n = " NOT" if self.negated else ""
        return f"({self.expr}{n} BETWEEN {self.low} AND {self.high})"


@dataclass
class ExistsExpr:
    subquery: Any  # SelectStmt
    negated: bool = False

    def __repr__(self):
        n = "NOT " if self.negated else ""
        return f"({n}EXISTS (...))"


@dataclass
class SubqueryExpr:
    query: Any  # SelectStmt

    def __repr__(self):
        return "(subquery)"


@dataclass
class CaseExpr:
    operand: Any  # None for searched CASE
    whens: list  # [(condition, result), ...]
    else_result: Any  # None if no ELSE

    def __repr__(self):
        return "CASE ... END"


@dataclass
class AliasedExpr:
    expr: Any
    alias: Optional[str] = None

    def __repr__(self):
        if self.alias:
            return f"{self.expr} AS {self.alias}"
        return str(self.expr)


@dataclass
class TableRef:
    name: str
    alias: Optional[str] = None

    def __repr__(self):
        if self.alias:
            return f"{self.name} AS {self.alias}"
        return self.name


@dataclass
class JoinClause:
    type: str  # INNER, LEFT, RIGHT, CROSS
    table: Any  # TableRef or SubqueryTable
    condition: Any  # ON condition (None for CROSS)

    def __repr__(self):
        return f"{self.type} JOIN {self.table}"


@dataclass
class SubqueryTable:
    query: Any  # SelectStmt
    alias: str

    def __repr__(self):
        return f"(subquery) AS {self.alias}"


@dataclass
class OrderByItem:
    expr: Any
    direction: str = 'ASC'

    def __repr__(self):
        return f"{self.expr} {self.direction}"


@dataclass
class SelectStmt:
    columns: list  # [AliasedExpr, ...]
    from_clause: Optional[Any] = None  # TableRef, SubqueryTable
    joins: list = field(default_factory=list)  # [JoinClause, ...]
    where: Any = None
    group_by: list = field(default_factory=list)
    having: Any = None
    order_by: list = field(default_factory=list)
    limit: Optional[int] = None
    offset: Optional[int] = None
    distinct: bool = False


# ============================================================
# SQL Parser
# ============================================================

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, *types) -> bool:
        return self._cur().type in types

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, tt: TokenType) -> Token:
        t = self._cur()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt.name}, got {t.type.name} ({t.value!r}) at pos {t.pos}")
        return self._advance()

    def _match(self, *types) -> Optional[Token]:
        if self._cur().type in types:
            return self._advance()
        return None

    def parse(self) -> SelectStmt:
        stmt = self._parse_select()
        if not self._peek(TokenType.EOF):
            raise SyntaxError(f"Unexpected token: {self._cur().value!r}")
        return stmt

    def _parse_select(self) -> SelectStmt:
        self._expect(TokenType.SELECT)
        distinct = self._match(TokenType.DISTINCT) is not None
        columns = self._parse_select_list()
        from_clause = None
        joins = []
        where = None
        group_by = []
        having = None
        order_by = []
        limit = None
        offset = None

        if self._match(TokenType.FROM):
            from_clause, joins = self._parse_from()

        if self._match(TokenType.WHERE):
            where = self._parse_expr()

        if self._match(TokenType.GROUP):
            self._expect(TokenType.BY)
            group_by = self._parse_expr_list()

        if self._match(TokenType.HAVING):
            having = self._parse_expr()

        if self._match(TokenType.ORDER):
            self._expect(TokenType.BY)
            order_by = self._parse_order_by_list()

        if self._match(TokenType.LIMIT):
            limit = int(self._expect(TokenType.NUMBER).value)

        if self._match(TokenType.OFFSET):
            offset = int(self._expect(TokenType.NUMBER).value)

        return SelectStmt(columns=columns, from_clause=from_clause, joins=joins,
                          where=where, group_by=group_by, having=having,
                          order_by=order_by, limit=limit, offset=offset, distinct=distinct)

    def _parse_select_list(self) -> list:
        items = []
        items.append(self._parse_select_item())
        while self._match(TokenType.COMMA):
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> AliasedExpr:
        if self._peek(TokenType.STAR):
            self._advance()
            alias = None
            if self._match(TokenType.AS):
                alias = self._expect(TokenType.IDENT).value
            return AliasedExpr(StarExpr(), alias)

        expr = self._parse_expr()

        # Check for table.* pattern
        if isinstance(expr, ColumnRef) and self._peek(TokenType.DOT):
            self._advance()
            if self._peek(TokenType.STAR):
                self._advance()
                return AliasedExpr(StarExpr(expr.column), None)

        alias = None
        if self._match(TokenType.AS):
            alias = self._advance().value
        elif self._peek(TokenType.IDENT):
            # Implicit alias
            alias = self._advance().value
        return AliasedExpr(expr, alias)

    def _parse_from(self):
        table = self._parse_table_ref()
        joins = []
        while True:
            jt = self._parse_join_type()
            if jt is None:
                break
            jtable = self._parse_table_ref()
            cond = None
            if jt != 'CROSS' and self._match(TokenType.ON):
                cond = self._parse_expr()
            joins.append(JoinClause(type=jt, table=jtable, condition=cond))
        return table, joins

    def _parse_join_type(self) -> Optional[str]:
        if self._match(TokenType.CROSS):
            self._expect(TokenType.JOIN)
            return 'CROSS'
        if self._match(TokenType.INNER):
            self._expect(TokenType.JOIN)
            return 'INNER'
        if self._match(TokenType.LEFT):
            self._match(TokenType.OUTER)
            self._expect(TokenType.JOIN)
            return 'LEFT'
        if self._match(TokenType.RIGHT):
            self._match(TokenType.OUTER)
            self._expect(TokenType.JOIN)
            return 'RIGHT'
        if self._match(TokenType.JOIN):
            return 'INNER'
        return None

    def _parse_table_ref(self):
        if self._match(TokenType.LPAREN):
            if self._peek(TokenType.SELECT):
                sub = self._parse_select()
                self._expect(TokenType.RPAREN)
                alias = None
                if self._match(TokenType.AS):
                    alias = self._advance().value
                elif self._peek(TokenType.IDENT):
                    alias = self._advance().value
                return SubqueryTable(sub, alias or '_sub')
            else:
                # parenthesized table -- just a grouping
                tref = self._parse_table_ref()
                self._expect(TokenType.RPAREN)
                return tref

        name = self._advance().value
        alias = None
        if self._match(TokenType.AS):
            alias = self._advance().value
        elif self._peek(TokenType.IDENT) and not self._peek_join_keyword():
            alias = self._advance().value
        return TableRef(name, alias)

    def _peek_join_keyword(self) -> bool:
        t = self._cur().type
        return t in (TokenType.JOIN, TokenType.INNER, TokenType.LEFT,
                     TokenType.RIGHT, TokenType.CROSS, TokenType.ON,
                     TokenType.WHERE, TokenType.GROUP, TokenType.ORDER,
                     TokenType.LIMIT, TokenType.HAVING, TokenType.UNION)

    def _parse_expr_list(self) -> list:
        exprs = [self._parse_expr()]
        while self._match(TokenType.COMMA):
            exprs.append(self._parse_expr())
        return exprs

    def _parse_order_by_list(self) -> list:
        items = []
        expr = self._parse_expr()
        direction = 'ASC'
        if self._match(TokenType.ASC):
            direction = 'ASC'
        elif self._match(TokenType.DESC):
            direction = 'DESC'
        items.append(OrderByItem(expr, direction))
        while self._match(TokenType.COMMA):
            expr = self._parse_expr()
            direction = 'ASC'
            if self._match(TokenType.ASC):
                direction = 'ASC'
            elif self._match(TokenType.DESC):
                direction = 'DESC'
            items.append(OrderByItem(expr, direction))
        return items

    # Expression parsing with precedence climbing
    def _parse_expr(self) -> Any:
        return self._parse_or()

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self._match(TokenType.OR):
            right = self._parse_and()
            left = BinExpr('OR', left, right)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_not()
        while self._match(TokenType.AND):
            right = self._parse_not()
            left = BinExpr('AND', left, right)
        return left

    def _parse_not(self) -> Any:
        if self._match(TokenType.NOT):
            operand = self._parse_not()
            return UnaryExpr('NOT', operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> Any:
        left = self._parse_addition()

        # IS [NOT] NULL
        if self._peek(TokenType.IS):
            self._advance()
            if self._match(TokenType.NOT):
                self._expect(TokenType.NULL)
                return UnaryExpr('IS NOT NULL', left)
            self._expect(TokenType.NULL)
            return UnaryExpr('IS NULL', left)

        # [NOT] IN (...)
        negated = False
        if self._peek(TokenType.NOT):
            saved = self.pos
            self._advance()
            if self._peek(TokenType.IN, TokenType.BETWEEN, TokenType.LIKE):
                negated = True
            else:
                self.pos = saved

        if self._match(TokenType.IN):
            self._expect(TokenType.LPAREN)
            if self._peek(TokenType.SELECT):
                sub = self._parse_select()
                self._expect(TokenType.RPAREN)
                return InExpr(left, [SubqueryExpr(sub)], negated)
            values = self._parse_expr_list()
            self._expect(TokenType.RPAREN)
            return InExpr(left, values, negated)

        # [NOT] BETWEEN ... AND ...
        if self._match(TokenType.BETWEEN):
            low = self._parse_addition()
            self._expect(TokenType.AND)
            high = self._parse_addition()
            return BetweenExpr(left, low, high, negated)

        # [NOT] LIKE
        if self._match(TokenType.LIKE):
            pattern = self._parse_addition()
            return BinExpr('NOT LIKE' if negated else 'LIKE', left, pattern)

        # Comparison operators
        op_map = {
            TokenType.EQ: '=', TokenType.NEQ: '!=', TokenType.LT: '<',
            TokenType.GT: '>', TokenType.LTE: '<=', TokenType.GTE: '>='
        }
        for tt, op in op_map.items():
            if self._match(tt):
                right = self._parse_addition()
                return BinExpr(op, left, right)

        return left

    def _parse_addition(self) -> Any:
        left = self._parse_multiplication()
        while self._peek(TokenType.PLUS, TokenType.MINUS):
            op = '+' if self._cur().type == TokenType.PLUS else '-'
            self._advance()
            right = self._parse_multiplication()
            left = BinExpr(op, left, right)
        return left

    def _parse_multiplication(self) -> Any:
        left = self._parse_unary()
        while self._peek(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = {'*': '*', '/': '/', '%': '%'}[self._cur().value]
            self._advance()
            right = self._parse_unary()
            left = BinExpr(op, left, right)
        return left

    def _parse_unary(self) -> Any:
        if self._match(TokenType.MINUS):
            operand = self._parse_primary()
            return UnaryExpr('-', operand)
        if self._match(TokenType.EXISTS):
            self._expect(TokenType.LPAREN)
            sub = self._parse_select()
            self._expect(TokenType.RPAREN)
            return ExistsExpr(sub)
        if self._peek(TokenType.NOT):
            saved = self.pos
            self._advance()
            if self._peek(TokenType.EXISTS):
                self._advance()
                self._expect(TokenType.LPAREN)
                sub = self._parse_select()
                self._expect(TokenType.RPAREN)
                return ExistsExpr(sub, negated=True)
            self.pos = saved
        return self._parse_primary()

    def _parse_primary(self) -> Any:
        # CASE expression
        if self._match(TokenType.CASE):
            return self._parse_case()

        # Aggregate functions
        agg_types = {TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN, TokenType.MAX}
        if self._cur().type in agg_types:
            name = self._advance().value.upper()
            self._expect(TokenType.LPAREN)
            distinct = self._match(TokenType.DISTINCT) is not None
            if self._peek(TokenType.STAR):
                self._advance()
                args = [StarExpr()]
            else:
                args = self._parse_expr_list()
            self._expect(TokenType.RPAREN)
            return FuncCall(name, args, distinct)

        # Parenthesized expr or subquery
        if self._match(TokenType.LPAREN):
            if self._peek(TokenType.SELECT):
                sub = self._parse_select()
                self._expect(TokenType.RPAREN)
                return SubqueryExpr(sub)
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        # Literals
        if self._peek(TokenType.NUMBER):
            val = self._advance().value
            return Literal(float(val) if '.' in val else int(val))

        if self._peek(TokenType.STRING):
            return Literal(self._advance().value)

        if self._match(TokenType.NULL):
            return Literal(None)

        if self._match(TokenType.TRUE):
            return Literal(True)

        if self._match(TokenType.FALSE):
            return Literal(False)

        # Identifier (column or table.column or function call)
        if self._peek(TokenType.IDENT):
            name = self._advance().value

            # Function call
            if self._match(TokenType.LPAREN):
                args = []
                if not self._peek(TokenType.RPAREN):
                    args = self._parse_expr_list()
                self._expect(TokenType.RPAREN)
                return FuncCall(name.upper(), args)

            # table.column
            if self._match(TokenType.DOT):
                if self._peek(TokenType.STAR):
                    self._advance()
                    return StarExpr(name)
                col = self._advance().value
                return ColumnRef(name, col)

            return ColumnRef(None, name)

        raise SyntaxError(f"Unexpected token: {self._cur().type.name} ({self._cur().value!r})")

    def _parse_case(self) -> CaseExpr:
        operand = None
        if not self._peek(TokenType.WHEN):
            operand = self._parse_expr()
        whens = []
        while self._match(TokenType.WHEN):
            cond = self._parse_expr()
            self._expect(TokenType.THEN)
            result = self._parse_expr()
            whens.append((cond, result))
        else_result = None
        if self._match(TokenType.ELSE):
            else_result = self._parse_expr()
        self._expect(TokenType.END)
        return CaseExpr(operand, whens, else_result)


def parse_sql(sql: str) -> SelectStmt:
    tokens = Lexer(sql).tokenize()
    return Parser(tokens).parse()


# ============================================================
# Catalog: Table/Column/Index definitions and statistics
# ============================================================

@dataclass
class ColumnStats:
    """Statistics for a single column."""
    name: str
    distinct_count: int = 100
    null_count: int = 0
    min_value: Any = None
    max_value: Any = None
    avg_width: int = 8  # bytes
    histogram: list = field(default_factory=list)  # bucket boundaries


@dataclass
class IndexDef:
    """An index definition."""
    name: str
    table: str
    columns: list[str]
    unique: bool = False
    type: str = 'btree'  # btree, hash


@dataclass
class TableDef:
    """Table metadata and statistics."""
    name: str
    columns: list[ColumnStats]
    row_count: int = 1000
    indexes: list[IndexDef] = field(default_factory=list)
    page_count: int = 0  # 0 = auto-estimate

    def __post_init__(self):
        if self.page_count == 0:
            row_width = sum(c.avg_width for c in self.columns) + 8  # header
            rows_per_page = max(1, 8192 // row_width)
            self.page_count = max(1, math.ceil(self.row_count / rows_per_page))

    def get_column(self, name: str) -> Optional[ColumnStats]:
        for c in self.columns:
            if c.name == name:
                return c
        return None


class Catalog:
    """Database catalog holding table definitions, indexes, and statistics."""

    def __init__(self):
        self.tables: dict[str, TableDef] = {}
        self.indexes: dict[str, IndexDef] = {}  # index_name -> IndexDef

    def add_table(self, table: TableDef):
        self.tables[table.name] = table
        for idx in table.indexes:
            self.indexes[idx.name] = idx

    def get_table(self, name: str) -> Optional[TableDef]:
        return self.tables.get(name)

    def get_indexes_for_table(self, table_name: str) -> list[IndexDef]:
        return [idx for idx in self.indexes.values() if idx.table == table_name]

    def add_index(self, index: IndexDef):
        self.indexes[index.name] = index
        table = self.tables.get(index.table)
        if table and index not in table.indexes:
            table.indexes.append(index)


# ============================================================
# Logical Plan (Relational Algebra)
# ============================================================

class LogicalOp:
    """Base for logical plan operators."""
    def children(self) -> list['LogicalOp']:
        return []

    def schema(self) -> list[str]:
        """Returns output column names."""
        return []


@dataclass
class LogicalScan(LogicalOp):
    table: str
    alias: Optional[str] = None
    columns: list[str] = field(default_factory=list)

    def schema(self):
        prefix = self.alias or self.table
        return [f"{prefix}.{c}" for c in self.columns]


@dataclass
class LogicalFilter(LogicalOp):
    input: LogicalOp
    condition: Any  # AST expression

    def children(self):
        return [self.input]

    def schema(self):
        return self.input.schema()


@dataclass
class LogicalProject(LogicalOp):
    input: LogicalOp
    expressions: list  # [(expr, alias), ...]

    def children(self):
        return [self.input]

    def schema(self):
        result = []
        for expr, alias in self.expressions:
            if alias:
                result.append(alias)
            elif isinstance(expr, ColumnRef):
                result.append(f"{expr.table}.{expr.column}" if expr.table else expr.column)
            elif isinstance(expr, StarExpr):
                result.extend(self.input.schema())
            else:
                result.append(str(expr))
        return result


@dataclass
class LogicalJoin(LogicalOp):
    left: LogicalOp
    right: LogicalOp
    condition: Any  # join condition
    join_type: str = 'INNER'  # INNER, LEFT, RIGHT, CROSS

    def children(self):
        return [self.left, self.right]

    def schema(self):
        return self.left.schema() + self.right.schema()


@dataclass
class LogicalAggregate(LogicalOp):
    input: LogicalOp
    group_by: list  # expressions
    aggregates: list  # [(FuncCall, alias), ...]

    def children(self):
        return [self.input]

    def schema(self):
        result = []
        for expr in self.group_by:
            if isinstance(expr, ColumnRef):
                result.append(f"{expr.table}.{expr.column}" if expr.table else expr.column)
            else:
                result.append(str(expr))
        for _, alias in self.aggregates:
            result.append(alias or '?')
        return result


@dataclass
class LogicalSort(LogicalOp):
    input: LogicalOp
    order_by: list[OrderByItem]

    def children(self):
        return [self.input]

    def schema(self):
        return self.input.schema()


@dataclass
class LogicalLimit(LogicalOp):
    input: LogicalOp
    limit: int
    offset: int = 0

    def children(self):
        return [self.input]

    def schema(self):
        return self.input.schema()


@dataclass
class LogicalDistinct(LogicalOp):
    input: LogicalOp

    def children(self):
        return [self.input]

    def schema(self):
        return self.input.schema()


# ============================================================
# Physical Plan
# ============================================================

class PhysicalOp:
    """Base for physical plan operators."""
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self) -> list['PhysicalOp']:
        return []


@dataclass
class SeqScan(PhysicalOp):
    table: str
    alias: Optional[str] = None
    filter: Any = None  # pushed-down filter
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0


@dataclass
class IndexScan(PhysicalOp):
    table: str
    alias: Optional[str] = None
    index: str = ''
    lookup_columns: list[str] = field(default_factory=list)
    lookup_values: list = field(default_factory=list)
    scan_type: str = 'eq'  # eq, range
    range_low: Any = None
    range_high: Any = None
    filter: Any = None  # residual filter after index
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0


@dataclass
class HashJoin(PhysicalOp):
    left: PhysicalOp = None
    right: PhysicalOp = None
    condition: Any = None
    join_type: str = 'INNER'
    # Build side is right (smaller), probe side is left (larger)
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.left, self.right]


@dataclass
class MergeJoin(PhysicalOp):
    left: PhysicalOp = None
    right: PhysicalOp = None
    condition: Any = None
    join_type: str = 'INNER'
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.left, self.right]


@dataclass
class NestedLoopJoin(PhysicalOp):
    left: PhysicalOp = None
    right: PhysicalOp = None
    condition: Any = None
    join_type: str = 'INNER'
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.left, self.right]


@dataclass
class PhysicalFilter(PhysicalOp):
    input: PhysicalOp = None
    condition: Any = None
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input]


@dataclass
class PhysicalProject(PhysicalOp):
    input: PhysicalOp = None
    expressions: list = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input] if self.input else []


@dataclass
class PhysicalSort(PhysicalOp):
    input: PhysicalOp = None
    order_by: list = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input]


@dataclass
class HashAggregate(PhysicalOp):
    input: PhysicalOp = None
    group_by: list = field(default_factory=list)
    aggregates: list = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input]


@dataclass
class SortAggregate(PhysicalOp):
    input: PhysicalOp = None
    group_by: list = field(default_factory=list)
    aggregates: list = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input]


@dataclass
class PhysicalLimit(PhysicalOp):
    input: PhysicalOp = None
    limit: int = 0
    offset: int = 0
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input]


@dataclass
class PhysicalDistinct(PhysicalOp):
    input: PhysicalOp = None
    estimated_cost: float = 0.0
    estimated_rows: float = 0.0

    def children(self):
        return [self.input]


# ============================================================
# Cost Model
# ============================================================

@dataclass
class CostParams:
    """Tunable cost model parameters."""
    seq_page_cost: float = 1.0        # cost of sequential I/O page
    random_page_cost: float = 4.0     # cost of random I/O page
    cpu_tuple_cost: float = 0.01      # per-row CPU processing
    cpu_index_tuple_cost: float = 0.005  # per-row index processing
    cpu_operator_cost: float = 0.0025 # per-operator evaluation
    hash_build_cost: float = 0.02     # per-row hash build
    hash_probe_cost: float = 0.01     # per-row hash probe
    sort_cost_factor: float = 0.05    # n*log(n) sort factor
    effective_cache_size: int = 16384 # pages in cache


class CostEstimator:
    """Estimates costs and cardinalities for plan operators."""

    def __init__(self, catalog: Catalog, params: CostParams = None):
        self.catalog = catalog
        self.params = params or CostParams()

    def estimate_selectivity(self, condition: Any, tables: dict[str, TableDef]) -> float:
        """Estimate the selectivity of a filter condition (0.0 to 1.0)."""
        if condition is None:
            return 1.0

        if isinstance(condition, BinExpr):
            if condition.op == 'AND':
                s1 = self.estimate_selectivity(condition.left, tables)
                s2 = self.estimate_selectivity(condition.right, tables)
                return s1 * s2
            if condition.op == 'OR':
                s1 = self.estimate_selectivity(condition.left, tables)
                s2 = self.estimate_selectivity(condition.right, tables)
                return s1 + s2 - s1 * s2
            if condition.op == '=':
                return self._eq_selectivity(condition, tables)
            if condition.op == '!=':
                return 1.0 - self._eq_selectivity(condition, tables)
            if condition.op in ('<', '>', '<=', '>='):
                return self._range_selectivity(condition, tables)
            if condition.op == 'LIKE':
                return self._like_selectivity(condition)

        if isinstance(condition, UnaryExpr):
            if condition.op == 'NOT':
                return 1.0 - self.estimate_selectivity(condition.operand, tables)
            if condition.op == 'IS NULL':
                col = condition.operand
                if isinstance(col, ColumnRef):
                    stats = self._get_col_stats(col, tables)
                    if stats:
                        tbl = tables.get(col.table or '')
                        if tbl and tbl.row_count > 0:
                            return stats.null_count / tbl.row_count
                return 0.01
            if condition.op == 'IS NOT NULL':
                return 1.0 - self.estimate_selectivity(
                    UnaryExpr('IS NULL', condition.operand), tables)

        if isinstance(condition, InExpr):
            n = len(condition.values)
            eq_sel = self._eq_selectivity(
                BinExpr('=', condition.expr, Literal(0)), tables)
            sel = min(1.0, n * eq_sel)
            return 1.0 - sel if condition.negated else sel

        if isinstance(condition, BetweenExpr):
            sel = self._range_selectivity(
                BinExpr('>=', condition.expr, condition.low), tables)
            sel *= self._range_selectivity(
                BinExpr('<=', condition.expr, condition.high), tables)
            return 1.0 - sel if condition.negated else sel

        return 0.5  # default

    def _eq_selectivity(self, cond: BinExpr, tables: dict[str, TableDef]) -> float:
        col = cond.left if isinstance(cond.left, ColumnRef) else (
            cond.right if isinstance(cond.right, ColumnRef) else None)
        if col:
            stats = self._get_col_stats(col, tables)
            if stats and stats.distinct_count > 0:
                return 1.0 / stats.distinct_count
        return 0.1  # default equality selectivity

    def _range_selectivity(self, cond: BinExpr, tables: dict[str, TableDef]) -> float:
        col = cond.left if isinstance(cond.left, ColumnRef) else (
            cond.right if isinstance(cond.right, ColumnRef) else None)
        if col:
            stats = self._get_col_stats(col, tables)
            if stats and stats.min_value is not None and stats.max_value is not None:
                val = cond.right if isinstance(cond.left, ColumnRef) else cond.left
                if isinstance(val, Literal) and isinstance(val.value, (int, float)):
                    rng = stats.max_value - stats.min_value
                    if rng > 0:
                        frac = (val.value - stats.min_value) / rng
                        frac = max(0.0, min(1.0, frac))
                        if cond.op in ('<', '<='):
                            return frac
                        if cond.op in ('>', '>='):
                            return 1.0 - frac
        return 0.33  # default range selectivity

    def _like_selectivity(self, cond: BinExpr) -> float:
        if isinstance(cond.right, Literal) and isinstance(cond.right.value, str):
            pat = cond.right.value
            if '%' not in pat and '_' not in pat:
                return 0.01  # exact match
            if pat.endswith('%') and '%' not in pat[:-1]:
                return 0.1  # prefix match
        return 0.25  # general LIKE

    def _get_col_stats(self, col: ColumnRef, tables: dict[str, TableDef]) -> Optional[ColumnStats]:
        tname = col.table
        if tname and tname in tables:
            return tables[tname].get_column(col.column)
        # Search all tables
        for t in tables.values():
            s = t.get_column(col.column)
            if s:
                return s
        return None

    def estimate_join_rows(self, left_rows: float, right_rows: float,
                           condition: Any, tables: dict[str, TableDef]) -> float:
        """Estimate output rows from a join."""
        if condition is None:
            return left_rows * right_rows  # cross join

        sel = self.estimate_selectivity(condition, tables)
        return max(1.0, left_rows * right_rows * sel)

    def cost_seq_scan(self, table: TableDef, filter_sel: float = 1.0) -> tuple[float, float]:
        """Returns (cost, rows) for sequential scan."""
        rows = table.row_count * filter_sel
        cost = (self.params.seq_page_cost * table.page_count +
                self.params.cpu_tuple_cost * table.row_count)
        return cost, max(1.0, rows)

    def cost_index_scan(self, table: TableDef, index: IndexDef,
                        selectivity: float) -> tuple[float, float]:
        """Returns (cost, rows) for index scan."""
        rows = max(1.0, table.row_count * selectivity)
        # Index traversal + random page reads for matching rows
        index_pages = max(1, int(math.log2(table.row_count + 1)) + 1)
        data_pages = max(1, int(rows * table.page_count / max(1, table.row_count)))
        cost = (self.params.random_page_cost * (index_pages + data_pages) +
                self.params.cpu_index_tuple_cost * rows)
        return cost, rows

    def cost_hash_join(self, left_rows: float, right_rows: float,
                       join_rows: float) -> float:
        """Cost of hash join. Build on right (smaller), probe from left."""
        build_cost = self.params.hash_build_cost * right_rows
        probe_cost = self.params.hash_probe_cost * left_rows
        output_cost = self.params.cpu_tuple_cost * join_rows
        return build_cost + probe_cost + output_cost

    def cost_merge_join(self, left_rows: float, right_rows: float,
                        join_rows: float, presorted: bool = False) -> float:
        """Cost of merge join (includes sort cost if not presorted)."""
        cost = 0.0
        if not presorted:
            if left_rows > 1:
                cost += self.params.sort_cost_factor * left_rows * math.log2(left_rows)
            if right_rows > 1:
                cost += self.params.sort_cost_factor * right_rows * math.log2(right_rows)
        cost += self.params.cpu_tuple_cost * (left_rows + right_rows)
        cost += self.params.cpu_tuple_cost * join_rows
        return cost

    def cost_nested_loop(self, left_rows: float, right_rows: float,
                         join_rows: float) -> float:
        """Cost of nested loop join."""
        cost = self.params.cpu_tuple_cost * left_rows * right_rows
        cost += self.params.cpu_tuple_cost * join_rows
        return cost

    def cost_sort(self, rows: float) -> float:
        if rows <= 1:
            return 0.0
        return self.params.sort_cost_factor * rows * math.log2(rows)

    def cost_hash_aggregate(self, input_rows: float, groups: float) -> float:
        return (self.params.hash_build_cost * input_rows +
                self.params.cpu_tuple_cost * groups)


# ============================================================
# SQL -> Logical Plan converter
# ============================================================

class LogicalPlanner:
    """Converts SQL AST to logical plan."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.alias_map: dict[str, str] = {}  # alias -> table name

    def plan(self, stmt: SelectStmt) -> LogicalOp:
        # Build FROM clause
        node = None
        if stmt.from_clause:
            node = self._plan_table_ref(stmt.from_clause)

        # Apply JOINs
        for join in stmt.joins:
            right = self._plan_table_ref(join.table)
            node = LogicalJoin(
                left=node, right=right,
                condition=join.condition,
                join_type=join.type
            )

        # Apply WHERE
        if stmt.where:
            node = LogicalFilter(input=node, condition=stmt.where)

        # Apply GROUP BY / aggregates
        aggs = self._extract_aggregates(stmt)
        if stmt.group_by or aggs:
            node = LogicalAggregate(
                input=node,
                group_by=stmt.group_by,
                aggregates=aggs
            )

        # Apply HAVING
        if stmt.having:
            node = LogicalFilter(input=node, condition=stmt.having)

        # Apply DISTINCT
        if stmt.distinct:
            node = LogicalDistinct(input=node)

        # Apply ORDER BY
        if stmt.order_by:
            node = LogicalSort(input=node, order_by=stmt.order_by)

        # Apply LIMIT/OFFSET
        if stmt.limit is not None:
            node = LogicalLimit(input=node, limit=stmt.limit,
                               offset=stmt.offset or 0)

        # Apply projection
        exprs = [(ae.expr, ae.alias) for ae in stmt.columns]
        node = LogicalProject(input=node, expressions=exprs)

        return node

    def _plan_table_ref(self, ref) -> LogicalOp:
        if isinstance(ref, TableRef):
            table = self.catalog.get_table(ref.name)
            if table:
                cols = [c.name for c in table.columns]
            else:
                cols = []
            alias = ref.alias or ref.name
            self.alias_map[alias] = ref.name
            return LogicalScan(table=ref.name, alias=alias, columns=cols)

        if isinstance(ref, SubqueryTable):
            # Plan subquery
            sub_planner = LogicalPlanner(self.catalog)
            sub_plan = sub_planner.plan(ref.query)
            return sub_plan  # simplified

        raise ValueError(f"Unknown table ref type: {type(ref)}")

    def _extract_aggregates(self, stmt: SelectStmt) -> list:
        """Extract aggregate functions from SELECT list."""
        aggs = []
        for ae in stmt.columns:
            self._find_aggs(ae.expr, aggs, ae.alias)
        return aggs

    def _find_aggs(self, expr, aggs: list, alias=None):
        if isinstance(expr, FuncCall) and expr.name.upper() in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX'):
            aggs.append((expr, alias or str(expr)))
            return
        if isinstance(expr, BinExpr):
            self._find_aggs(expr.left, aggs)
            self._find_aggs(expr.right, aggs)


# ============================================================
# Plan Transformations (Logical -> Logical)
# ============================================================

class PlanTransformer:
    """Applies rule-based transformations to logical plans."""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def optimize(self, plan: LogicalOp) -> LogicalOp:
        plan = self.predicate_pushdown(plan)
        plan = self.join_reorder(plan)
        return plan

    def predicate_pushdown(self, plan: LogicalOp) -> LogicalOp:
        """Push filters as close to scans as possible."""
        if isinstance(plan, LogicalFilter):
            inner = self.predicate_pushdown(plan.input)
            return self._push_filter(plan.condition, inner)

        if isinstance(plan, LogicalJoin):
            left = self.predicate_pushdown(plan.left)
            right = self.predicate_pushdown(plan.right)
            return LogicalJoin(left=left, right=right,
                               condition=plan.condition, join_type=plan.join_type)

        if isinstance(plan, LogicalProject):
            return LogicalProject(
                input=self.predicate_pushdown(plan.input),
                expressions=plan.expressions)

        if isinstance(plan, LogicalSort):
            return LogicalSort(
                input=self.predicate_pushdown(plan.input),
                order_by=plan.order_by)

        if isinstance(plan, LogicalLimit):
            return LogicalLimit(
                input=self.predicate_pushdown(plan.input),
                limit=plan.limit, offset=plan.offset)

        if isinstance(plan, LogicalAggregate):
            return LogicalAggregate(
                input=self.predicate_pushdown(plan.input),
                group_by=plan.group_by, aggregates=plan.aggregates)

        if isinstance(plan, LogicalDistinct):
            return LogicalDistinct(input=self.predicate_pushdown(plan.input))

        return plan

    def _push_filter(self, condition: Any, plan: LogicalOp) -> LogicalOp:
        """Try to push a condition deeper into the plan tree."""
        conjuncts = self._split_and(condition)

        if isinstance(plan, LogicalJoin):
            left_tables = self._get_tables(plan.left)
            right_tables = self._get_tables(plan.right)

            left_filters = []
            right_filters = []
            join_filters = []
            remaining = []

            for cond in conjuncts:
                refs = self._get_table_refs(cond)
                if refs and refs.issubset(left_tables):
                    left_filters.append(cond)
                elif refs and refs.issubset(right_tables):
                    right_filters.append(cond)
                elif refs and refs.issubset(left_tables | right_tables):
                    join_filters.append(cond)
                else:
                    remaining.append(cond)

            left = plan.left
            if left_filters:
                left = LogicalFilter(input=left,
                                     condition=self._combine_and(left_filters))
            left = self.predicate_pushdown(left)

            right = plan.right
            if right_filters:
                right = LogicalFilter(input=right,
                                      condition=self._combine_and(right_filters))
            right = self.predicate_pushdown(right)

            # Merge join filters with existing condition
            join_cond = plan.condition
            if join_filters:
                all_jf = [join_cond] if join_cond else []
                all_jf.extend(join_filters)
                join_cond = self._combine_and(all_jf)

            result = LogicalJoin(left=left, right=right,
                                 condition=join_cond, join_type=plan.join_type)

            if remaining:
                result = LogicalFilter(input=result,
                                       condition=self._combine_and(remaining))
            return result

        if isinstance(plan, LogicalScan):
            return LogicalFilter(input=plan, condition=condition)

        if isinstance(plan, LogicalFilter):
            merged = BinExpr('AND', plan.condition, condition)
            return self._push_filter(merged, plan.input)

        return LogicalFilter(input=plan, condition=condition)

    def _split_and(self, cond: Any) -> list:
        """Split AND conditions into a flat list."""
        if isinstance(cond, BinExpr) and cond.op == 'AND':
            return self._split_and(cond.left) + self._split_and(cond.right)
        return [cond]

    def _combine_and(self, conds: list) -> Any:
        if not conds:
            return None
        result = conds[0]
        for c in conds[1:]:
            result = BinExpr('AND', result, c)
        return result

    def _get_tables(self, plan: LogicalOp) -> set[str]:
        """Get all table aliases referenced by a plan subtree."""
        if isinstance(plan, LogicalScan):
            return {plan.alias or plan.table}
        result = set()
        for child in plan.children():
            result |= self._get_tables(child)
        return result

    def _get_table_refs(self, expr: Any) -> set[str]:
        """Get table references from an expression."""
        refs = set()
        if isinstance(expr, ColumnRef):
            if expr.table:
                refs.add(expr.table)
        elif isinstance(expr, BinExpr):
            refs |= self._get_table_refs(expr.left)
            refs |= self._get_table_refs(expr.right)
        elif isinstance(expr, UnaryExpr):
            refs |= self._get_table_refs(expr.operand)
        elif isinstance(expr, FuncCall):
            for arg in expr.args:
                refs |= self._get_table_refs(arg)
        elif isinstance(expr, InExpr):
            refs |= self._get_table_refs(expr.expr)
        elif isinstance(expr, BetweenExpr):
            refs |= self._get_table_refs(expr.expr)
        return refs

    def join_reorder(self, plan: LogicalOp) -> LogicalOp:
        """Reorder joins using dynamic programming (System R style)."""
        if isinstance(plan, LogicalProject):
            return LogicalProject(
                input=self.join_reorder(plan.input),
                expressions=plan.expressions)
        if isinstance(plan, LogicalSort):
            return LogicalSort(
                input=self.join_reorder(plan.input),
                order_by=plan.order_by)
        if isinstance(plan, LogicalLimit):
            return LogicalLimit(
                input=self.join_reorder(plan.input),
                limit=plan.limit, offset=plan.offset)
        if isinstance(plan, LogicalAggregate):
            return LogicalAggregate(
                input=self.join_reorder(plan.input),
                group_by=plan.group_by, aggregates=plan.aggregates)
        if isinstance(plan, LogicalFilter):
            return LogicalFilter(
                input=self.join_reorder(plan.input),
                condition=plan.condition)
        if isinstance(plan, LogicalDistinct):
            return LogicalDistinct(input=self.join_reorder(plan.input))

        if not isinstance(plan, LogicalJoin):
            return plan

        # Collect all base relations and join conditions
        relations = []
        conditions = []
        self._collect_join_info(plan, relations, conditions)

        if len(relations) <= 1:
            return plan

        # DP join enumeration
        return self._dp_join_order(relations, conditions, plan)

    def _collect_join_info(self, plan: LogicalOp, relations: list, conditions: list):
        if isinstance(plan, LogicalJoin) and plan.join_type == 'INNER':
            self._collect_join_info(plan.left, relations, conditions)
            self._collect_join_info(plan.right, relations, conditions)
            if plan.condition:
                conditions.append(plan.condition)
        else:
            relations.append(plan)

    def _dp_join_order(self, relations: list, conditions: list,
                       original: LogicalJoin) -> LogicalOp:
        """System R style DP join ordering."""
        n = len(relations)
        if n <= 1:
            return relations[0] if relations else original
        if n > 12:  # too many relations for DP, fall back to greedy
            return self._greedy_join_order(relations, conditions)

        estimator = CostEstimator(self.catalog)

        # Build table sets for each relation
        rel_tables = [self._get_tables(r) for r in relations]

        # Map: frozenset of relation indices -> (cost, plan, estimated_rows)
        dp: dict[frozenset, tuple[float, LogicalOp, float]] = {}

        # Base case: single relations
        for i, rel in enumerate(relations):
            rows = self._estimate_base_rows(rel)
            dp[frozenset([i])] = (0.0, rel, rows)

        # Find applicable conditions for a pair of table sets
        def find_conditions(left_tables: set, right_tables: set) -> list:
            result = []
            for cond in conditions:
                refs = self._get_table_refs(cond)
                if refs and refs.issubset(left_tables | right_tables):
                    if refs & left_tables and refs & right_tables:
                        result.append(cond)
            return result

        # DP over subsets of increasing size
        for size in range(2, n + 1):
            for subset in self._subsets_of_size(n, size):
                best = None
                for left_sub in self._non_empty_subsets(subset):
                    right_sub = subset - left_sub
                    if not right_sub or left_sub not in dp or right_sub not in dp:
                        continue

                    left_cost, left_plan, left_rows = dp[left_sub]
                    right_cost, right_plan, right_rows = dp[right_sub]

                    left_tables = set()
                    for i in left_sub:
                        left_tables |= rel_tables[i]
                    right_tables = set()
                    for i in right_sub:
                        right_tables |= rel_tables[i]

                    jconds = find_conditions(left_tables, right_tables)
                    jcond = self._combine_and(jconds) if jconds else None

                    all_tables = {}
                    for i in subset:
                        for t in rel_tables[i]:
                            tdef = self.catalog.get_table(t)
                            if tdef:
                                all_tables[t] = tdef

                    join_rows = estimator.estimate_join_rows(
                        left_rows, right_rows, jcond, all_tables)

                    join_cost = estimator.cost_hash_join(left_rows, right_rows, join_rows)
                    total_cost = left_cost + right_cost + join_cost

                    if best is None or total_cost < best[0]:
                        join_plan = LogicalJoin(
                            left=left_plan, right=right_plan,
                            condition=jcond, join_type='INNER')
                        best = (total_cost, join_plan, join_rows)

                if best:
                    dp[subset] = best

        full_set = frozenset(range(n))
        if full_set in dp:
            _, plan, _ = dp[full_set]
            # Apply any remaining conditions not used in joins
            used_tables = set()
            for i in range(n):
                used_tables |= rel_tables[i]
            remaining = []
            for cond in conditions:
                refs = self._get_table_refs(cond)
                if not (refs and len(refs) >= 2):
                    remaining.append(cond)
            if remaining:
                plan = LogicalFilter(input=plan,
                                     condition=self._combine_and(remaining))
            return plan

        return original

    def _estimate_base_rows(self, plan: LogicalOp) -> float:
        if isinstance(plan, LogicalScan):
            tbl = self.catalog.get_table(plan.table)
            return float(tbl.row_count) if tbl else 1000.0
        if isinstance(plan, LogicalFilter):
            base_rows = self._estimate_base_rows(plan.input)
            estimator = CostEstimator(self.catalog)
            tables = {}
            for t in self._get_tables(plan):
                tdef = self.catalog.get_table(t)
                if tdef:
                    tables[t] = tdef
            sel = estimator.estimate_selectivity(plan.condition, tables)
            return max(1.0, base_rows * sel)
        return 1000.0

    def _subsets_of_size(self, n: int, size: int):
        """Generate all frozensets of {0..n-1} with given size."""
        def gen(start, remaining, current):
            if remaining == 0:
                yield frozenset(current)
                return
            for i in range(start, n):
                yield from gen(i + 1, remaining - 1, current + [i])
        yield from gen(0, size, [])

    def _non_empty_subsets(self, s: frozenset):
        """Generate all non-empty proper subsets of s."""
        elems = sorted(s)
        n = len(elems)
        for mask in range(1, (1 << n) - 1):
            yield frozenset(elems[i] for i in range(n) if mask & (1 << i))

    def _greedy_join_order(self, relations: list, conditions: list) -> LogicalOp:
        """Greedy join ordering for large join counts."""
        remaining = list(range(len(relations)))
        estimator = CostEstimator(self.catalog)
        rel_tables = [self._get_tables(r) for r in relations]

        current_idx = remaining.pop(0)
        current_plan = relations[current_idx]
        current_rows = self._estimate_base_rows(current_plan)
        current_tables = rel_tables[current_idx]

        while remaining:
            best = None
            for i, idx in enumerate(remaining):
                right_plan = relations[idx]
                right_tables = rel_tables[idx]
                right_rows = self._estimate_base_rows(right_plan)

                jconds = []
                for cond in conditions:
                    refs = self._get_table_refs(cond)
                    if refs and refs.issubset(current_tables | right_tables):
                        if refs & current_tables and refs & right_tables:
                            jconds.append(cond)

                jcond = self._combine_and(jconds) if jconds else None
                all_tables = {}
                for t in current_tables | right_tables:
                    tdef = self.catalog.get_table(t)
                    if tdef:
                        all_tables[t] = tdef

                join_rows = estimator.estimate_join_rows(
                    current_rows, right_rows, jcond, all_tables)
                cost = estimator.cost_hash_join(current_rows, right_rows, join_rows)

                if best is None or cost < best[0]:
                    best = (cost, i, idx, right_plan, jcond, join_rows)

            _, pos, idx, right_plan, jcond, join_rows = best
            remaining.pop(pos)
            current_plan = LogicalJoin(
                left=current_plan, right=right_plan,
                condition=jcond, join_type='INNER')
            current_rows = join_rows
            current_tables |= rel_tables[idx]

        return current_plan


# ============================================================
# Physical Plan Generator
# ============================================================

class PhysicalPlanner:
    """Converts logical plan to physical plan with cost-based operator selection."""

    def __init__(self, catalog: Catalog, cost_params: CostParams = None):
        self.catalog = catalog
        self.estimator = CostEstimator(catalog, cost_params)

    def plan(self, logical: LogicalOp) -> PhysicalOp:
        return self._plan_node(logical)

    def _plan_node(self, node: LogicalOp) -> PhysicalOp:
        if isinstance(node, LogicalScan):
            return self._plan_scan(node)
        if isinstance(node, LogicalFilter):
            return self._plan_filter(node)
        if isinstance(node, LogicalJoin):
            return self._plan_join(node)
        if isinstance(node, LogicalProject):
            if node.input is None:
                # SELECT without FROM -- single virtual row
                p = PhysicalProject(input=None, expressions=node.expressions)
                p.estimated_rows = 1.0
                p.estimated_cost = self.estimator.params.cpu_tuple_cost
                return p
            child = self._plan_node(node.input)
            p = PhysicalProject(input=child, expressions=node.expressions)
            p.estimated_rows = child.estimated_rows
            p.estimated_cost = child.estimated_cost + self.estimator.params.cpu_tuple_cost * child.estimated_rows
            return p
        if isinstance(node, LogicalAggregate):
            return self._plan_aggregate(node)
        if isinstance(node, LogicalSort):
            return self._plan_sort(node)
        if isinstance(node, LogicalLimit):
            child = self._plan_node(node.input)
            rows = min(node.limit, child.estimated_rows)
            p = PhysicalLimit(input=child, limit=node.limit, offset=node.offset)
            p.estimated_rows = rows
            p.estimated_cost = child.estimated_cost + self.estimator.params.cpu_tuple_cost * rows
            return p
        if isinstance(node, LogicalDistinct):
            child = self._plan_node(node.input)
            p = PhysicalDistinct(input=child)
            p.estimated_rows = child.estimated_rows * 0.8
            p.estimated_cost = child.estimated_cost + self.estimator.cost_hash_aggregate(
                child.estimated_rows, child.estimated_rows * 0.8)
            return p
        raise ValueError(f"Unknown logical node: {type(node)}")

    def _plan_scan(self, node: LogicalScan, filter_cond=None) -> PhysicalOp:
        table = self.catalog.get_table(node.table)
        if not table:
            s = SeqScan(table=node.table, alias=node.alias)
            s.estimated_rows = 1000.0
            s.estimated_cost = 10.0
            return s

        # Check for index scan opportunity
        if filter_cond:
            index_plan = self._try_index_scan(table, node, filter_cond)
            if index_plan:
                return index_plan

        cost, rows = self.estimator.cost_seq_scan(table)
        s = SeqScan(table=node.table, alias=node.alias, filter=filter_cond)
        s.estimated_rows = rows
        s.estimated_cost = cost
        if filter_cond:
            tables = {(node.alias or node.table): table}
            sel = self.estimator.estimate_selectivity(filter_cond, tables)
            s.estimated_rows = max(1.0, table.row_count * sel)
        return s

    def _try_index_scan(self, table: TableDef, scan: LogicalScan,
                        condition: Any) -> Optional[IndexScan]:
        """Try to use an index for the given condition."""
        indexes = self.catalog.get_indexes_for_table(table.name)
        if not indexes:
            return None

        # Extract equality and range conditions
        conjuncts = []
        if isinstance(condition, BinExpr) and condition.op == 'AND':
            self._flatten_and(condition, conjuncts)
        else:
            conjuncts = [condition]

        best_plan = None
        best_cost = float('inf')

        for idx in indexes:
            # Check if any conjunct matches the index's first column
            matched_cols = []
            matched_vals = []
            remaining = []
            scan_type = 'eq'
            range_low = None
            range_high = None

            for cond in conjuncts:
                if isinstance(cond, BinExpr) and cond.op == '=':
                    col = self._extract_column(cond, scan.alias or scan.table)
                    val = self._extract_literal(cond)
                    if col and val is not None and col in idx.columns:
                        matched_cols.append(col)
                        matched_vals.append(val)
                        continue
                if isinstance(cond, BinExpr) and cond.op in ('<', '<=', '>', '>='):
                    col = self._extract_column(cond, scan.alias or scan.table)
                    val = self._extract_literal(cond)
                    if col and val is not None and col in idx.columns:
                        scan_type = 'range'
                        if cond.op in ('>', '>='):
                            range_low = val
                        else:
                            range_high = val
                        continue
                remaining.append(cond)

            if not matched_cols and scan_type != 'range':
                continue

            # Estimate selectivity
            if scan_type == 'eq' and matched_cols:
                tables = {(scan.alias or scan.table): table}
                sel = 1.0
                for col in matched_cols:
                    cs = table.get_column(col)
                    if cs and cs.distinct_count > 0:
                        sel *= 1.0 / cs.distinct_count
                    else:
                        sel *= 0.1
            else:
                sel = 0.33  # range default

            cost, rows = self.estimator.cost_index_scan(table, idx, sel)

            # Compare with seq scan cost
            seq_cost, _ = self.estimator.cost_seq_scan(table)
            if cost < seq_cost and cost < best_cost:
                residual = self._combine_and_list(remaining) if remaining else None
                ip = IndexScan(
                    table=scan.table, alias=scan.alias,
                    index=idx.name, lookup_columns=matched_cols,
                    lookup_values=matched_vals, scan_type=scan_type,
                    range_low=range_low, range_high=range_high,
                    filter=residual)
                ip.estimated_cost = cost
                ip.estimated_rows = rows
                best_plan = ip
                best_cost = cost

        return best_plan

    def _extract_column(self, cond: BinExpr, table_alias: str) -> Optional[str]:
        for side in [cond.left, cond.right]:
            if isinstance(side, ColumnRef):
                if side.table is None or side.table == table_alias:
                    return side.column
        return None

    def _extract_literal(self, cond: BinExpr) -> Any:
        for side in [cond.left, cond.right]:
            if isinstance(side, Literal):
                return side.value
        return None

    def _flatten_and(self, cond, result):
        if isinstance(cond, BinExpr) and cond.op == 'AND':
            self._flatten_and(cond.left, result)
            self._flatten_and(cond.right, result)
        else:
            result.append(cond)

    def _combine_and_list(self, conds):
        if not conds:
            return None
        result = conds[0]
        for c in conds[1:]:
            result = BinExpr('AND', result, c)
        return result

    def _plan_filter(self, node: LogicalFilter) -> PhysicalOp:
        # Try to push filter into scan
        if isinstance(node.input, LogicalScan):
            return self._plan_scan(node.input, node.condition)

        child = self._plan_node(node.input)
        tables = {}
        for t in self._collect_tables(node.input):
            tdef = self.catalog.get_table(t)
            if tdef:
                tables[t] = tdef
        sel = self.estimator.estimate_selectivity(node.condition, tables)
        rows = max(1.0, child.estimated_rows * sel)

        p = PhysicalFilter(input=child, condition=node.condition)
        p.estimated_rows = rows
        p.estimated_cost = child.estimated_cost + self.estimator.params.cpu_operator_cost * child.estimated_rows
        return p

    def _plan_join(self, node: LogicalJoin) -> PhysicalOp:
        left = self._plan_node(node.left)
        right = self._plan_node(node.right)

        tables = {}
        for t in self._collect_tables(node.left) | self._collect_tables(node.right):
            tdef = self.catalog.get_table(t)
            if tdef:
                tables[t] = tdef

        join_rows = self.estimator.estimate_join_rows(
            left.estimated_rows, right.estimated_rows, node.condition, tables)

        # Ensure build side (right) is smaller for hash join
        if right.estimated_rows > left.estimated_rows and node.join_type == 'INNER':
            left, right = right, left

        # Cost all three join strategies
        hash_cost = self.estimator.cost_hash_join(
            left.estimated_rows, right.estimated_rows, join_rows)
        merge_cost = self.estimator.cost_merge_join(
            left.estimated_rows, right.estimated_rows, join_rows)
        nl_cost = self.estimator.cost_nested_loop(
            left.estimated_rows, right.estimated_rows, join_rows)

        # For non-equi joins, hash join isn't applicable
        is_equi = self._is_equi_join(node.condition)

        candidates = []
        if is_equi:
            candidates.append((left.estimated_cost + right.estimated_cost + hash_cost,
                               'hash'))
            candidates.append((left.estimated_cost + right.estimated_cost + merge_cost,
                               'merge'))
        candidates.append((left.estimated_cost + right.estimated_cost + nl_cost,
                           'nl'))

        candidates.sort(key=lambda x: x[0])
        best_cost, best_type = candidates[0]

        if best_type == 'hash':
            p = HashJoin(left=left, right=right, condition=node.condition,
                         join_type=node.join_type)
        elif best_type == 'merge':
            p = MergeJoin(left=left, right=right, condition=node.condition,
                          join_type=node.join_type)
        else:
            p = NestedLoopJoin(left=left, right=right, condition=node.condition,
                               join_type=node.join_type)

        p.estimated_rows = join_rows
        p.estimated_cost = best_cost
        return p

    def _is_equi_join(self, condition: Any) -> bool:
        if condition is None:
            return False
        if isinstance(condition, BinExpr):
            if condition.op == '=':
                return (isinstance(condition.left, ColumnRef) and
                        isinstance(condition.right, ColumnRef))
            if condition.op == 'AND':
                return self._is_equi_join(condition.left) or self._is_equi_join(condition.right)
        return False

    def _plan_aggregate(self, node: LogicalAggregate) -> PhysicalOp:
        child = self._plan_node(node.input)
        input_rows = child.estimated_rows

        # Estimate number of groups
        if node.group_by:
            groups = self._estimate_groups(node.group_by, node.input, input_rows)
        else:
            groups = 1.0

        hash_cost = self.estimator.cost_hash_aggregate(input_rows, groups)
        sort_cost = self.estimator.cost_sort(input_rows) + self.estimator.params.cpu_tuple_cost * input_rows

        if hash_cost <= sort_cost or not node.group_by:
            p = HashAggregate(input=child, group_by=node.group_by,
                              aggregates=node.aggregates)
            p.estimated_cost = child.estimated_cost + hash_cost
        else:
            p = SortAggregate(input=child, group_by=node.group_by,
                              aggregates=node.aggregates)
            p.estimated_cost = child.estimated_cost + sort_cost

        p.estimated_rows = groups
        return p

    def _estimate_groups(self, group_by: list, input_plan: LogicalOp,
                         input_rows: float) -> float:
        groups = input_rows
        for expr in group_by:
            if isinstance(expr, ColumnRef):
                for t in self._collect_tables(input_plan):
                    tdef = self.catalog.get_table(t)
                    if tdef:
                        cs = tdef.get_column(expr.column)
                        if cs:
                            groups = min(groups, float(cs.distinct_count))
                            break
        return max(1.0, groups)

    def _plan_sort(self, node: LogicalSort) -> PhysicalOp:
        child = self._plan_node(node.input)
        sort_cost = self.estimator.cost_sort(child.estimated_rows)
        p = PhysicalSort(input=child, order_by=node.order_by)
        p.estimated_rows = child.estimated_rows
        p.estimated_cost = child.estimated_cost + sort_cost
        return p

    def _collect_tables(self, plan: LogicalOp) -> set[str]:
        if isinstance(plan, LogicalScan):
            return {plan.alias or plan.table}
        result = set()
        for child in plan.children():
            result |= self._collect_tables(child)
        return result


# ============================================================
# EXPLAIN output
# ============================================================

def explain(plan: PhysicalOp, indent: int = 0) -> str:
    """Generate EXPLAIN output for a physical plan."""
    lines = []
    _explain_node(plan, lines, indent)
    return '\n'.join(lines)


def _explain_node(node: PhysicalOp, lines: list, indent: int):
    prefix = '  ' * indent
    arrow = '-> ' if indent > 0 else ''

    if isinstance(node, SeqScan):
        alias = f" {node.alias}" if node.alias and node.alias != node.table else ""
        filt = f"  Filter: {node.filter}" if node.filter else ""
        lines.append(f"{prefix}{arrow}Seq Scan on {node.table}{alias}"
                     f"  (cost={node.estimated_cost:.2f} rows={node.estimated_rows:.0f})")
        if filt:
            lines.append(f"{prefix}  {filt}")

    elif isinstance(node, IndexScan):
        alias = f" {node.alias}" if node.alias and node.alias != node.table else ""
        lines.append(f"{prefix}{arrow}Index Scan using {node.index} on {node.table}{alias}"
                     f"  (cost={node.estimated_cost:.2f} rows={node.estimated_rows:.0f})")
        if node.lookup_columns:
            conds = [f"{c} = {v}" for c, v in zip(node.lookup_columns, node.lookup_values)]
            lines.append(f"{prefix}    Index Cond: ({', '.join(conds)})")
        if node.filter:
            lines.append(f"{prefix}    Filter: {node.filter}")

    elif isinstance(node, HashJoin):
        lines.append(f"{prefix}{arrow}Hash Join  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        if node.condition:
            lines.append(f"{prefix}    Hash Cond: {node.condition}")
        _explain_node(node.left, lines, indent + 1)
        _explain_node(node.right, lines, indent + 1)

    elif isinstance(node, MergeJoin):
        lines.append(f"{prefix}{arrow}Merge Join  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        if node.condition:
            lines.append(f"{prefix}    Merge Cond: {node.condition}")
        _explain_node(node.left, lines, indent + 1)
        _explain_node(node.right, lines, indent + 1)

    elif isinstance(node, NestedLoopJoin):
        lines.append(f"{prefix}{arrow}Nested Loop  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        if node.condition:
            lines.append(f"{prefix}    Join Filter: {node.condition}")
        _explain_node(node.left, lines, indent + 1)
        _explain_node(node.right, lines, indent + 1)

    elif isinstance(node, PhysicalFilter):
        lines.append(f"{prefix}{arrow}Filter  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        lines.append(f"{prefix}    Filter: {node.condition}")
        _explain_node(node.input, lines, indent + 1)

    elif isinstance(node, PhysicalProject):
        lines.append(f"{prefix}{arrow}Project  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        if node.input:
            _explain_node(node.input, lines, indent + 1)

    elif isinstance(node, PhysicalSort):
        lines.append(f"{prefix}{arrow}Sort  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        keys = ', '.join(f"{o.expr} {o.direction}" for o in node.order_by)
        lines.append(f"{prefix}    Sort Key: {keys}")
        _explain_node(node.input, lines, indent + 1)

    elif isinstance(node, HashAggregate):
        lines.append(f"{prefix}{arrow}Hash Aggregate  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        if node.group_by:
            keys = ', '.join(str(g) for g in node.group_by)
            lines.append(f"{prefix}    Group Key: {keys}")
        _explain_node(node.input, lines, indent + 1)

    elif isinstance(node, SortAggregate):
        lines.append(f"{prefix}{arrow}Sort Aggregate  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        if node.group_by:
            keys = ', '.join(str(g) for g in node.group_by)
            lines.append(f"{prefix}    Group Key: {keys}")
        _explain_node(node.input, lines, indent + 1)

    elif isinstance(node, PhysicalLimit):
        lines.append(f"{prefix}{arrow}Limit  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        _explain_node(node.input, lines, indent + 1)

    elif isinstance(node, PhysicalDistinct):
        lines.append(f"{prefix}{arrow}Distinct  (cost={node.estimated_cost:.2f}"
                     f" rows={node.estimated_rows:.0f})")
        _explain_node(node.input, lines, indent + 1)


# ============================================================
# Query Optimizer (main entry point)
# ============================================================

class QueryOptimizer:
    """Main query optimizer: parses SQL, builds logical plan, optimizes, generates physical plan."""

    def __init__(self, catalog: Catalog, cost_params: CostParams = None):
        self.catalog = catalog
        self.cost_params = cost_params
        self.transformer = PlanTransformer(catalog)
        self.physical_planner = PhysicalPlanner(catalog, cost_params)

    def optimize(self, sql: str) -> PhysicalOp:
        """Parse SQL, optimize, and return physical plan."""
        ast = parse_sql(sql)
        logical = self.to_logical(ast)
        optimized = self.transform(logical)
        physical = self.to_physical(optimized)
        return physical

    def to_logical(self, ast: SelectStmt) -> LogicalOp:
        """Convert SQL AST to logical plan."""
        planner = LogicalPlanner(self.catalog)
        return planner.plan(ast)

    def transform(self, logical: LogicalOp) -> LogicalOp:
        """Apply logical transformations."""
        return self.transformer.optimize(logical)

    def to_physical(self, logical: LogicalOp) -> PhysicalOp:
        """Convert logical plan to physical plan."""
        return self.physical_planner.plan(logical)

    def explain(self, sql: str) -> str:
        """Return EXPLAIN output for a query."""
        plan = self.optimize(sql)
        return explain(plan)

    def explain_logical(self, sql: str) -> str:
        """Return logical plan representation."""
        ast = parse_sql(sql)
        logical = LogicalPlanner(self.catalog).plan(ast)
        return self._format_logical(logical)

    def explain_optimized(self, sql: str) -> str:
        """Return optimized logical plan."""
        ast = parse_sql(sql)
        logical = LogicalPlanner(self.catalog).plan(ast)
        optimized = self.transformer.optimize(logical)
        return self._format_logical(optimized)

    def _format_logical(self, node: LogicalOp, indent: int = 0) -> str:
        lines = []
        self._format_logical_node(node, lines, indent)
        return '\n'.join(lines)

    def _format_logical_node(self, node: LogicalOp, lines: list, indent: int):
        prefix = '  ' * indent
        arrow = '-> ' if indent > 0 else ''

        if isinstance(node, LogicalScan):
            alias = f" AS {node.alias}" if node.alias and node.alias != node.table else ""
            lines.append(f"{prefix}{arrow}Scan: {node.table}{alias}")
        elif isinstance(node, LogicalFilter):
            lines.append(f"{prefix}{arrow}Filter: {node.condition}")
            self._format_logical_node(node.input, lines, indent + 1)
        elif isinstance(node, LogicalProject):
            exprs = ', '.join(str(e) for e, a in node.expressions)
            lines.append(f"{prefix}{arrow}Project: {exprs}")
            self._format_logical_node(node.input, lines, indent + 1)
        elif isinstance(node, LogicalJoin):
            lines.append(f"{prefix}{arrow}{node.join_type} Join: {node.condition}")
            self._format_logical_node(node.left, lines, indent + 1)
            self._format_logical_node(node.right, lines, indent + 1)
        elif isinstance(node, LogicalAggregate):
            gb = ', '.join(str(g) for g in node.group_by) if node.group_by else '(none)'
            aggs = ', '.join(str(a) for a, _ in node.aggregates) if node.aggregates else '(none)'
            lines.append(f"{prefix}{arrow}Aggregate: group_by=[{gb}] aggs=[{aggs}]")
            self._format_logical_node(node.input, lines, indent + 1)
        elif isinstance(node, LogicalSort):
            keys = ', '.join(f"{o.expr} {o.direction}" for o in node.order_by)
            lines.append(f"{prefix}{arrow}Sort: {keys}")
            self._format_logical_node(node.input, lines, indent + 1)
        elif isinstance(node, LogicalLimit):
            lines.append(f"{prefix}{arrow}Limit: {node.limit} offset={node.offset}")
            self._format_logical_node(node.input, lines, indent + 1)
        elif isinstance(node, LogicalDistinct):
            lines.append(f"{prefix}{arrow}Distinct")
            self._format_logical_node(node.input, lines, indent + 1)
