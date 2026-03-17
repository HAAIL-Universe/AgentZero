"""
C249: Stored Procedures & User-Defined Functions
Extends C247 (Mini Database Engine)

Adds stored procedures and UDFs to the SQL database engine:
- CREATE FUNCTION: scalar UDFs with parameters, SQL body, RETURN
- CREATE PROCEDURE: multi-statement procedures with IN/OUT/INOUT params
- CALL procedure(args): execute procedures
- SELECT func(args): invoke UDFs in queries
- DROP FUNCTION / DROP PROCEDURE
- Control flow: IF/ELSEIF/ELSE, WHILE loop, DECLARE variables, SET
- Recursion with depth limiting
- Exception handling: DECLARE HANDLER
- Nested procedure/function calls
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set
from enum import Enum, auto

# Import C247
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C247_mini_database'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C245_query_executor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C244_buffer_pool'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C246_transaction_manager'))

from mini_database import (
    MiniDB, ResultSet, DatabaseError,
    Lexer, Parser, Token, TokenType,
    parse_sql, parse_sql_multi,
    # AST nodes
    SelectStmt, SelectExpr, TableRef, JoinClause,
    InsertStmt, UpdateStmt, DeleteStmt,
    CreateTableStmt, DropTableStmt, CreateIndexStmt,
    BeginStmt, CommitStmt, RollbackStmt, SavepointStmt,
    ShowTablesStmt, DescribeStmt, ExplainStmt,
    ColumnDef,
    # SQL expression AST
    SqlColumnRef, SqlLiteral, SqlBinOp, SqlComparison, SqlLogic,
    SqlIsNull, SqlFuncCall, SqlAggCall, SqlBetween, SqlInList,
    SqlCase, SqlStar,
    # Compiler / storage
    QueryCompiler, StorageEngine, CompileError, CatalogError,
    ParseError, KEYWORDS,
)

from query_executor import (
    Database as QEDatabase, Table as QETable, Row, ExecutionEngine,
    Operator, SeqScanOp, FilterOp, ProjectOp, SortOp,
    HashAggregateOp, HavingOp, LimitOp, DistinctOp,
    HashJoinOp, NestedLoopJoinOp,
    ColumnRef, Literal, Comparison, LogicExpr, ArithExpr, FuncExpr,
    CompOp, LogicOp, AggFunc, AggCall,
    eval_expr,
)

from transaction_manager import IsolationLevel


# =============================================================================
# Procedure/Function AST Nodes
# =============================================================================

class ParamMode(Enum):
    IN = auto()
    OUT = auto()
    INOUT = auto()


@dataclass
class ParamDef:
    name: str
    param_type: str  # 'int', 'text', 'float', 'bool'
    mode: ParamMode = ParamMode.IN
    default: Any = None


@dataclass
class CreateFunctionStmt:
    name: str
    params: List[ParamDef]
    return_type: str
    body: List[Any]  # list of body statements
    deterministic: bool = False
    replace: bool = False


@dataclass
class CreateProcedureStmt:
    name: str
    params: List[ParamDef]
    body: List[Any]  # list of body statements
    replace: bool = False


@dataclass
class DropFunctionStmt:
    name: str
    if_exists: bool = False


@dataclass
class DropProcedureStmt:
    name: str
    if_exists: bool = False


@dataclass
class CallStmt:
    name: str
    args: List[Any]  # SQL expressions


@dataclass
class DeclareVarStmt:
    name: str
    var_type: str
    default: Any = None


@dataclass
class DeclareHandlerStmt:
    handler_type: str  # 'continue', 'exit'
    condition: str     # 'sqlexception', 'not_found', or specific error
    body: List[Any]


@dataclass
class SetVarStmt:
    name: str
    expr: Any  # SQL expression


@dataclass
class ReturnStmt:
    expr: Any  # SQL expression


@dataclass
class IfStmt:
    condition: Any        # SQL expression
    then_body: List[Any]  # body statements
    elseif_clauses: List[Tuple[Any, List[Any]]]  # (condition, body) pairs
    else_body: Optional[List[Any]] = None


@dataclass
class WhileStmt:
    condition: Any   # SQL expression
    body: List[Any]  # body statements
    label: Optional[str] = None


@dataclass
class LeaveStmt:
    label: Optional[str] = None


@dataclass
class IterateStmt:
    label: Optional[str] = None


@dataclass
class SelectIntoStmt:
    """SELECT ... INTO var_name"""
    select: SelectStmt
    variables: List[str]


@dataclass
class ShowFunctionsStmt:
    pass


@dataclass
class ShowProceduresStmt:
    pass


# =============================================================================
# Stored routine catalog
# =============================================================================

@dataclass
class StoredFunction:
    name: str
    params: List[ParamDef]
    return_type: str
    body: List[Any]
    deterministic: bool = False


@dataclass
class StoredProcedure:
    name: str
    params: List[ParamDef]
    body: List[Any]


class RoutineCatalog:
    """Stores function and procedure definitions."""

    def __init__(self):
        self.functions: Dict[str, StoredFunction] = {}
        self.procedures: Dict[str, StoredProcedure] = {}

    def create_function(self, func: StoredFunction, replace: bool = False):
        if func.name in self.functions and not replace:
            raise DatabaseError(f"Function '{func.name}' already exists")
        self.functions[func.name] = func

    def create_procedure(self, proc: StoredProcedure, replace: bool = False):
        if proc.name in self.procedures and not replace:
            raise DatabaseError(f"Procedure '{proc.name}' already exists")
        self.procedures[proc.name] = proc

    def get_function(self, name: str) -> StoredFunction:
        if name not in self.functions:
            raise DatabaseError(f"Function '{name}' not found")
        return self.functions[name]

    def get_procedure(self, name: str) -> StoredProcedure:
        if name not in self.procedures:
            raise DatabaseError(f"Procedure '{name}' not found")
        return self.procedures[name]

    def drop_function(self, name: str, if_exists: bool = False):
        if name not in self.functions:
            if if_exists:
                return
            raise DatabaseError(f"Function '{name}' not found")
        del self.functions[name]

    def drop_procedure(self, name: str, if_exists: bool = False):
        if name not in self.procedures:
            if if_exists:
                return
            raise DatabaseError(f"Procedure '{name}' not found")
        del self.procedures[name]

    def list_functions(self) -> List[str]:
        return sorted(self.functions.keys())

    def list_procedures(self) -> List[str]:
        return sorted(self.procedures.keys())


# =============================================================================
# Extended Lexer -- adds procedure/function keywords
# =============================================================================

PROC_KEYWORDS = {
    'function': 'FUNCTION',
    'procedure': 'PROCEDURE',
    'returns': 'RETURNS',
    'return': 'RETURN',
    'call': 'CALL',
    'declare': 'DECLARE',
    'handler': 'HANDLER',
    'continue': 'CONTINUE',
    'exit': 'EXIT',
    'for': 'FOR',
    'sqlexception': 'SQLEXCEPTION',
    'not_found': 'NOT_FOUND',
    'while': 'WHILE',
    'do': 'DO',
    'if': 'IF',
    'elseif': 'ELSEIF',
    'deterministic': 'DETERMINISTIC',
    'replace': 'REPLACE',
    'out': 'OUT',
    'inout': 'INOUT',
    'iterate': 'ITERATE',
    'leave': 'LEAVE',
    'functions': 'FUNCTIONS',
    'procedures': 'PROCEDURES',
}


class ProcToken:
    """Extended token that can hold procedure keywords."""
    def __init__(self, type_val, value, pos=0):
        self.type = type_val
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"ProcToken({self.type}, {self.value!r})"


class ProcLexer:
    """Extended lexer that recognizes procedure/function keywords."""

    def __init__(self, sql: str):
        self.sql = sql
        self.pos = 0
        self.tokens: List[Any] = []
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
                if two == ':=':
                    self.tokens.append(ProcToken('ASSIGN', ':=', self.pos))
                    self.pos += 2
                    continue
            # Colon (for labels)
            if c == ':':
                self.tokens.append(ProcToken('COLON', ':', self.pos))
                self.pos += 1
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
            # Percent and modulo
            if c == '%':
                self.tokens.append(ProcToken('MOD', '%', self.pos))
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
                    self.pos += 1
                self.tokens.append(Token(TokenType.STRING, ''.join(parts), start))
                continue
            # Identifiers / keywords
            if c.isalpha() or c == '_' or c == '@':
                start = self.pos
                self.pos += 1
                while self.pos < n and (s[self.pos].isalnum() or s[self.pos] == '_'):
                    self.pos += 1
                word = s[start:self.pos]
                lower = word.lower()
                # Check procedure keywords first
                if lower in PROC_KEYWORDS:
                    self.tokens.append(ProcToken(PROC_KEYWORDS[lower], word, start))
                elif lower in KEYWORDS:
                    self.tokens.append(Token(KEYWORDS[lower], word, start))
                else:
                    self.tokens.append(Token(TokenType.IDENT, word, start))
                continue
            # Unknown char -- skip
            self.pos += 1

        self.tokens.append(Token(TokenType.EOF, None, self.pos))


# =============================================================================
# Extended Parser -- parses procedure/function DDL and body statements
# =============================================================================

class ProcParser(Parser):
    """Parser that handles CREATE FUNCTION/PROCEDURE, CALL, and body statements."""

    def __init__(self, tokens: List[Any]):
        # ProcParser uses its own token list (may contain ProcTokens)
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Any:
        return self.tokens[self.pos]

    def advance(self) -> Any:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt) -> Any:
        t = self.advance()
        tok_type = t.type if hasattr(t, 'type') else t.type
        if self._type_matches(tok_type, tt):
            return t
        raise ParseError(f"Expected {tt}, got {tok_type} ({t.value!r}) at pos {t.pos}")

    def _type_matches(self, actual, expected) -> bool:
        """Match token type, handling both TokenType enums and ProcToken strings."""
        if actual == expected:
            return True
        # Cross-matching: ProcToken string 'IF' should match TokenType.IF etc.
        # Build a mapping from proc keyword strings to their TokenType equivalents
        proc_to_tokentype = {
            'IF': TokenType.IF, 'ELSE': TokenType.ELSE,
        }
        tokentype_to_proc = {v: k for k, v in proc_to_tokentype.items()}
        if isinstance(expected, str) and isinstance(actual, TokenType):
            return actual == proc_to_tokentype.get(expected)
        if isinstance(expected, TokenType) and isinstance(actual, str):
            return actual == tokentype_to_proc.get(expected)
        return False

    def match(self, *types) -> Any:
        t = self.peek()
        tok_type = t.type if hasattr(t, 'type') else t.type
        for tt in types:
            if isinstance(tt, str):
                if isinstance(tok_type, str) and tok_type == tt:
                    return self.advance()
            else:
                if tok_type == tt:
                    return self.advance()
        return None

    def _peek_type(self):
        t = self.peek()
        return t.type if hasattr(t, 'type') else t.type

    def _parse_statement(self):
        t = self.peek()
        tt = self._peek_type()

        # Extended statement types
        if tt == 'CALL':
            return self._parse_call()
        if tt == 'FUNCTION' or tt == 'FUNCTIONS':
            # SHOW FUNCTIONS handled elsewhere
            pass
        if isinstance(tt, TokenType) and tt == TokenType.SHOW:
            return self._parse_show_extended()
        if isinstance(tt, TokenType) and tt == TokenType.CREATE:
            return self._parse_create_extended()
        if isinstance(tt, TokenType) and tt == TokenType.DROP:
            return self._parse_drop_extended()

        # Delegate to parent for standard SQL
        return super()._parse_statement()

    def _parse_show_extended(self):
        self.advance()  # SHOW
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.TABLES:
            self.advance()
            return ShowTablesStmt()
        if tt == 'FUNCTIONS':
            self.advance()
            return ShowFunctionsStmt()
        if tt == 'PROCEDURES':
            self.advance()
            return ShowProceduresStmt()
        raise ParseError(f"Expected TABLES/FUNCTIONS/PROCEDURES after SHOW")

    def _parse_create_extended(self):
        self.advance()  # CREATE
        tt = self._peek_type()

        # CREATE OR REPLACE
        replace = False
        if isinstance(tt, TokenType) and tt == TokenType.OR:
            self.advance()  # OR
            self.expect('REPLACE')
            replace = True
            tt = self._peek_type()

        if tt == 'FUNCTION':
            return self._parse_create_function(replace)
        if tt == 'PROCEDURE':
            return self._parse_create_procedure(replace)

        # Standard CREATE TABLE / CREATE INDEX
        if isinstance(tt, TokenType) and tt == TokenType.TABLE:
            # Re-parse using the standard parser path
            # We need to manually handle this since we consumed CREATE
            return self._parse_create_table()
        if isinstance(tt, TokenType) and tt == TokenType.INDEX:
            return self._parse_create_index()

        raise ParseError(f"Expected FUNCTION/PROCEDURE/TABLE/INDEX after CREATE")

    def _parse_create_table(self):
        """Parse CREATE TABLE after CREATE has been consumed."""
        self.advance()  # TABLE
        if_not_exists = False
        if self._peek_type() == TokenType.IF:
            self.advance()
            self.expect(TokenType.NOT)
            self.expect(TokenType.EXISTS)
            if_not_exists = True
        name = self._expect_ident()
        self.expect(TokenType.LPAREN)
        columns = []
        while True:
            col = self._parse_column_def()
            columns.append(col)
            if not self.match(TokenType.COMMA):
                break
        self.expect(TokenType.RPAREN)
        return CreateTableStmt(table_name=name, columns=columns,
                               if_not_exists=if_not_exists)

    def _parse_column_def(self):
        name = self._expect_ident()
        col_type = self._parse_type_name()
        pk = False
        unique = False
        not_null = False
        default = None
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.PRIMARY:
                self.advance()
                self.expect(TokenType.KEY)
                pk = True
            elif isinstance(tt, TokenType) and tt == TokenType.UNIQUE:
                self.advance()
                unique = True
            elif isinstance(tt, TokenType) and tt == TokenType.NOT:
                self.advance()
                self.expect(TokenType.NULL)
                not_null = True
            elif isinstance(tt, TokenType) and tt == TokenType.DEFAULT:
                self.advance()
                default = self._parse_expr()
            else:
                break
        return ColumnDef(name=name, col_type=col_type, primary_key=pk,
                        unique=unique, not_null=not_null, default=default)

    def _parse_create_index(self):
        self.advance()  # INDEX
        index_name = self._expect_ident()
        self.expect(TokenType.ON)
        table_name = self._expect_ident()
        self.expect(TokenType.LPAREN)
        column = self._expect_ident()
        self.expect(TokenType.RPAREN)
        return CreateIndexStmt(index_name=index_name, table_name=table_name,
                               column=column)

    def _parse_drop_extended(self):
        self.advance()  # DROP
        tt = self._peek_type()

        if_exists = False

        if tt == 'FUNCTION':
            self.advance()
            ptt = self._peek_type()
            if (isinstance(ptt, TokenType) and ptt == TokenType.IF) or ptt == 'IF':
                self.advance()
                self.expect(TokenType.EXISTS)
                if_exists = True
            name = self._expect_ident()
            return DropFunctionStmt(name=name, if_exists=if_exists)

        if tt == 'PROCEDURE':
            self.advance()
            ptt = self._peek_type()
            if (isinstance(ptt, TokenType) and ptt == TokenType.IF) or ptt == 'IF':
                self.advance()
                self.expect(TokenType.EXISTS)
                if_exists = True
            name = self._expect_ident()
            return DropProcedureStmt(name=name, if_exists=if_exists)

        if isinstance(tt, TokenType) and tt == TokenType.TABLE:
            self.advance()
            if self._peek_type() == TokenType.IF:
                self.advance()
                self.expect(TokenType.EXISTS)
                if_exists = True
            name = self._expect_ident()
            return DropTableStmt(table_name=name, if_exists=if_exists)

        raise ParseError(f"Expected FUNCTION/PROCEDURE/TABLE after DROP")

    def _parse_create_function(self, replace: bool) -> CreateFunctionStmt:
        self.advance()  # FUNCTION
        name = self._expect_ident()

        # Parameters
        self.expect(TokenType.LPAREN)
        params = []
        if self._peek_type() != TokenType.RPAREN:
            params = self._parse_param_list()
        self.expect(TokenType.RPAREN)

        # RETURNS type
        self.expect('RETURNS')
        return_type = self._parse_type_name()

        # Optional DETERMINISTIC
        deterministic = False
        if self._peek_type() == 'DETERMINISTIC':
            self.advance()
            deterministic = True

        # BEGIN ... END block
        body = self._parse_begin_end()

        return CreateFunctionStmt(
            name=name, params=params, return_type=return_type,
            body=body, deterministic=deterministic, replace=replace
        )

    def _parse_create_procedure(self, replace: bool) -> CreateProcedureStmt:
        self.advance()  # PROCEDURE
        name = self._expect_ident()

        # Parameters
        self.expect(TokenType.LPAREN)
        params = []
        if self._peek_type() != TokenType.RPAREN:
            params = self._parse_param_list()
        self.expect(TokenType.RPAREN)

        # BEGIN ... END block
        body = self._parse_begin_end()

        return CreateProcedureStmt(
            name=name, params=params, body=body, replace=replace
        )

    def _parse_param_list(self) -> List[ParamDef]:
        params = []
        while True:
            mode = ParamMode.IN
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.IN:
                self.advance()
                mode = ParamMode.IN
            elif tt == 'OUT':
                self.advance()
                mode = ParamMode.OUT
            elif tt == 'INOUT':
                self.advance()
                mode = ParamMode.INOUT

            name = self._expect_ident()
            param_type = self._parse_type_name()

            default = None
            if self._peek_type() == TokenType.DEFAULT:
                self.advance()
                default = self._parse_expr()

            params.append(ParamDef(name=name, param_type=param_type,
                                   mode=mode, default=default))
            if not self.match(TokenType.COMMA):
                break
        return params

    def _parse_type_name(self) -> str:
        """Parse a type name (INT, TEXT, FLOAT, BOOL, etc.)"""
        t = self.advance()
        tt = t.type if hasattr(t, 'type') else t.type
        type_map = {
            TokenType.INT: 'int',
            TokenType.TEXT: 'text',
            TokenType.FLOAT: 'float',
            TokenType.BOOL: 'bool',
        }
        if isinstance(tt, TokenType) and tt in type_map:
            return type_map[tt]
        # Accept IDENT as type name too
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            return t.value.lower()
        raise ParseError(f"Expected type name, got {tt} ({t.value!r})")

    def _parse_begin_end(self) -> List[Any]:
        """Parse BEGIN ... END block."""
        self.expect(TokenType.BEGIN)
        stmts = []
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.END:
                self.advance()
                break
            stmt = self._parse_body_statement()
            stmts.append(stmt)
            self.match(TokenType.SEMICOLON)
        return stmts

    def _parse_body_statement(self) -> Any:
        """Parse a statement inside a BEGIN..END block."""
        tt = self._peek_type()

        if tt == 'DECLARE':
            return self._parse_declare()

        if isinstance(tt, TokenType) and tt == TokenType.SET:
            return self._parse_set_var()

        if tt == 'IF':
            return self._parse_if()

        if isinstance(tt, TokenType) and tt == TokenType.IF:
            return self._parse_if()

        if tt == 'WHILE':
            return self._parse_while()

        if tt == 'RETURN':
            return self._parse_return()

        if tt == 'LEAVE':
            return self._parse_leave()

        if tt == 'ITERATE':
            return self._parse_iterate()

        if tt == 'CALL':
            return self._parse_call()

        # Check for label (ident followed by colon)
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            # Look ahead for colon
            if self.pos + 1 < len(self.tokens):
                next_tt = self.tokens[self.pos + 1]
                next_type = next_tt.type if hasattr(next_tt, 'type') else next_tt.type
                if isinstance(next_type, str) and next_type == 'COLON':
                    # This is a label
                    label = self.advance().value
                    self.advance()  # consume colon
                    # Next must be WHILE
                    return self._parse_while(label=label)

        # SELECT INTO
        if isinstance(tt, TokenType) and tt == TokenType.SELECT:
            return self._parse_select_or_select_into()

        # Standard SQL statements
        return self._parse_statement()

    def _parse_declare(self) -> Any:
        """Parse DECLARE var_name type [DEFAULT expr] or DECLARE HANDLER."""
        self.advance()  # DECLARE

        tt = self._peek_type()
        # DECLARE CONTINUE/EXIT HANDLER
        if tt in ('CONTINUE', 'EXIT'):
            handler_type = self.advance().value.lower()
            self.expect('HANDLER')
            self.expect('FOR')
            # Condition
            ctt = self._peek_type()
            if ctt == 'SQLEXCEPTION':
                condition = self.advance().value.lower()
            elif ctt == 'NOT_FOUND':
                condition = self.advance().value.lower()
            else:
                condition = self._expect_ident()
            # Handler body: single statement or BEGIN..END
            btt = self._peek_type()
            if isinstance(btt, TokenType) and btt == TokenType.BEGIN:
                body = self._parse_begin_end()
            else:
                stmt = self._parse_body_statement()
                body = [stmt]
            return DeclareHandlerStmt(
                handler_type=handler_type,
                condition=condition,
                body=body
            )

        # DECLARE var_name type [DEFAULT expr]
        name = self._expect_ident()
        var_type = self._parse_type_name()
        default = None
        if self._peek_type() == TokenType.DEFAULT:
            self.advance()
            default = self._parse_expr()
        return DeclareVarStmt(name=name, var_type=var_type, default=default)

    def _parse_set_var(self) -> SetVarStmt:
        self.advance()  # SET
        name = self._expect_ident()
        self.expect(TokenType.EQ)
        expr = self._parse_expr()
        return SetVarStmt(name=name, expr=expr)

    def _parse_return(self) -> ReturnStmt:
        self.advance()  # RETURN
        expr = self._parse_expr()
        return ReturnStmt(expr=expr)

    def _parse_leave(self) -> LeaveStmt:
        self.advance()  # LEAVE
        label = None
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            label = self.advance().value
        return LeaveStmt(label=label)

    def _parse_iterate(self) -> IterateStmt:
        self.advance()  # ITERATE
        label = None
        tt = self._peek_type()
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            label = self.advance().value
        return IterateStmt(label=label)

    def _parse_if(self) -> IfStmt:
        self.advance()  # IF
        condition = self._parse_expr()
        self.expect(TokenType.THEN)

        then_body = []
        elseif_clauses = []
        else_body = None

        while True:
            tt = self._peek_type()
            if tt == 'ELSEIF':
                self.advance()
                elif_cond = self._parse_expr()
                self.expect(TokenType.THEN)
                elif_body = []
                while True:
                    btt = self._peek_type()
                    if btt == 'ELSEIF' or (isinstance(btt, TokenType) and btt == TokenType.ELSE) or btt == 'ELSE':
                        break
                    if isinstance(btt, TokenType) and btt == TokenType.END:
                        break
                    elif_body.append(self._parse_body_statement())
                    self.match(TokenType.SEMICOLON)
                elseif_clauses.append((elif_cond, elif_body))
                continue
            if (isinstance(tt, TokenType) and tt == TokenType.ELSE) or tt == 'ELSE':
                self.advance()
                else_body = []
                while True:
                    btt = self._peek_type()
                    if isinstance(btt, TokenType) and btt == TokenType.END:
                        break
                    else_body.append(self._parse_body_statement())
                    self.match(TokenType.SEMICOLON)
                break
            if isinstance(tt, TokenType) and tt == TokenType.END:
                break
            then_body.append(self._parse_body_statement())
            self.match(TokenType.SEMICOLON)

        self.expect(TokenType.END)
        self.expect(TokenType.IF)

        return IfStmt(
            condition=condition,
            then_body=then_body,
            elseif_clauses=elseif_clauses,
            else_body=else_body
        )

    def _parse_while(self, label: Optional[str] = None) -> WhileStmt:
        self.advance()  # WHILE
        condition = self._parse_expr()
        self.expect('DO')

        body = []
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.END:
                break
            body.append(self._parse_body_statement())
            self.match(TokenType.SEMICOLON)

        self.expect(TokenType.END)
        self.expect('WHILE')

        return WhileStmt(condition=condition, body=body, label=label)

    def _parse_call(self) -> CallStmt:
        self.advance()  # CALL
        name = self._expect_ident()
        self.expect(TokenType.LPAREN)
        args = []
        if self._peek_type() != TokenType.RPAREN:
            while True:
                args.append(self._parse_expr())
                if not self.match(TokenType.COMMA):
                    break
        self.expect(TokenType.RPAREN)
        return CallStmt(name=name, args=args)

    def _parse_select_or_select_into(self):
        """Parse SELECT statement, checking for INTO clause."""
        # Save position to detect INTO
        save_pos = self.pos
        self.expect(TokenType.SELECT)

        # Parse select list
        columns = self._parse_select_list()

        # Check for INTO
        if self._peek_type() == TokenType.INTO:
            self.advance()  # INTO
            variables = []
            while True:
                variables.append(self._expect_ident())
                if not self.match(TokenType.COMMA):
                    break

            # Continue with FROM etc
            from_table = None
            if self.match(TokenType.FROM):
                from_table = self._parse_table_ref()

            where = None
            if self.match(TokenType.WHERE):
                where = self._parse_expr()

            select = SelectStmt(
                columns=columns,
                from_table=from_table,
                where=where
            )
            return SelectIntoStmt(select=select, variables=variables)

        # Normal SELECT -- restore and use standard parser
        self.pos = save_pos
        return self._parse_select()

    def _expect_ident(self) -> str:
        """Expect an identifier, allowing some keywords as identifiers in certain contexts."""
        t = self.peek()
        tt = self._peek_type()
        # Accept IDENT
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            return self.advance().value
        # Accept certain keywords as identifiers in param/variable contexts
        keyword_as_ident = {
            TokenType.COUNT, TokenType.SUM, TokenType.AVG, TokenType.MIN,
            TokenType.MAX, TokenType.KEY, TokenType.INDEX, TokenType.TABLE,
            TokenType.ADD, TokenType.SET, TokenType.IF, TokenType.ELSE,
            TokenType.IN, TokenType.ON, TokenType.ORDER, TokenType.BY,
            TokenType.GROUP, TokenType.HAVING, TokenType.LIMIT, TokenType.AS,
            TokenType.IS, TokenType.NOT, TokenType.NULL, TokenType.DEFAULT,
            TokenType.DELETE, TokenType.UPDATE, TokenType.INSERT,
            TokenType.SELECT, TokenType.FROM, TokenType.WHERE,
            TokenType.EXISTS, TokenType.BETWEEN, TokenType.LIKE,
            TokenType.BEGIN, TokenType.END, TokenType.CASE, TokenType.WHEN,
            TokenType.THEN, TokenType.DESC, TokenType.ASC,
            TokenType.FLOAT, TokenType.INT, TokenType.TEXT, TokenType.BOOL,
            TokenType.JOIN, TokenType.LEFT, TokenType.RIGHT, TokenType.INNER,
            TokenType.CROSS, TokenType.OUTER, TokenType.FULL,
            TokenType.DROP, TokenType.CREATE, TokenType.ALTER,
            TokenType.DISTINCT, TokenType.ALL, TokenType.UNION,
        }
        if isinstance(tt, TokenType) and tt in keyword_as_ident:
            return self.advance().value
        # Accept proc keyword strings as identifiers too
        proc_as_ident = {'FUNCTION', 'PROCEDURE', 'HANDLER', 'CONTINUE',
                         'EXIT', 'RETURNS', 'RETURN', 'CALL', 'DO', 'WHILE',
                         'DETERMINISTIC', 'DECLARE', 'OUT', 'INOUT',
                         'LEAVE', 'ITERATE', 'FUNCTIONS', 'PROCEDURES',
                         'REPLACE'}
        if isinstance(tt, str) and tt in proc_as_ident:
            return self.advance().value
        raise ParseError(f"Expected identifier, got {tt} ({t.value!r}) at pos {t.pos}")

    def _parse_expr(self):
        """Parse expression, extending parent with modulo support."""
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self.match(TokenType.OR):
            right = self._parse_and()
            left = SqlLogic(op='or', operands=[left, right])
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self.match(TokenType.AND):
            right = self._parse_not()
            left = SqlLogic(op='and', operands=[left, right])
        return left

    def _parse_not(self):
        if self.match(TokenType.NOT):
            operand = self._parse_not()
            return SqlLogic(op='not', operands=[operand])
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_addition()

        tt = self._peek_type()
        ops = {
            TokenType.EQ: '=', TokenType.NE: '!=', TokenType.LT: '<',
            TokenType.LE: '<=', TokenType.GT: '>', TokenType.GE: '>=',
        }
        if isinstance(tt, TokenType) and tt in ops:
            self.advance()
            right = self._parse_addition()
            return SqlComparison(op=ops[tt], left=left, right=right)

        # IS NULL / IS NOT NULL
        if isinstance(tt, TokenType) and tt == TokenType.IS:
            self.advance()
            negated = bool(self.match(TokenType.NOT))
            self.expect(TokenType.NULL)
            return SqlIsNull(expr=left, negated=negated)

        # BETWEEN
        if isinstance(tt, TokenType) and tt == TokenType.BETWEEN:
            self.advance()
            low = self._parse_addition()
            self.expect(TokenType.AND)
            high = self._parse_addition()
            return SqlBetween(expr=left, low=low, high=high)

        # IN
        if isinstance(tt, TokenType) and tt == TokenType.IN:
            self.advance()
            self.expect(TokenType.LPAREN)
            values = []
            while True:
                values.append(self._parse_expr())
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RPAREN)
            return SqlInList(expr=left, values=values)

        # NOT IN
        if isinstance(tt, TokenType) and tt == TokenType.NOT:
            save = self.pos
            self.advance()
            if self._peek_type() == TokenType.IN:
                self.advance()
                self.expect(TokenType.LPAREN)
                values = []
                while True:
                    values.append(self._parse_expr())
                    if not self.match(TokenType.COMMA):
                        break
                self.expect(TokenType.RPAREN)
                return SqlLogic(op='not', operands=[SqlInList(expr=left, values=values)])
            self.pos = save

        # LIKE
        if isinstance(tt, TokenType) and tt == TokenType.LIKE:
            self.advance()
            right = self._parse_addition()
            return SqlComparison(op='like', left=left, right=right)

        return left

    def _parse_addition(self):
        left = self._parse_multiplication()
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.PLUS:
                self.advance()
                right = self._parse_multiplication()
                left = SqlBinOp(op='+', left=left, right=right)
            elif isinstance(tt, TokenType) and tt == TokenType.MINUS:
                self.advance()
                right = self._parse_multiplication()
                left = SqlBinOp(op='-', left=left, right=right)
            else:
                break
        return left

    def _parse_multiplication(self):
        left = self._parse_unary()
        while True:
            tt = self._peek_type()
            if isinstance(tt, TokenType) and tt == TokenType.STAR:
                self.advance()
                right = self._parse_unary()
                left = SqlBinOp(op='*', left=left, right=right)
            elif isinstance(tt, TokenType) and tt == TokenType.SLASH:
                self.advance()
                right = self._parse_unary()
                left = SqlBinOp(op='/', left=left, right=right)
            elif isinstance(tt, str) and tt == 'MOD':
                self.advance()
                right = self._parse_unary()
                left = SqlBinOp(op='%', left=left, right=right)
            else:
                break
        return left

    def _parse_unary(self):
        if self.match(TokenType.MINUS):
            operand = self._parse_primary()
            return SqlBinOp(op='*', left=SqlLiteral(-1), right=operand)
        return self._parse_primary()

    def _parse_primary(self):
        tt = self._peek_type()

        # Numbers
        if isinstance(tt, TokenType) and tt == TokenType.NUMBER:
            return SqlLiteral(self.advance().value)

        # Strings
        if isinstance(tt, TokenType) and tt == TokenType.STRING:
            return SqlLiteral(self.advance().value)

        # NULL
        if isinstance(tt, TokenType) and tt == TokenType.NULL:
            self.advance()
            return SqlLiteral(None)

        # TRUE/FALSE
        if isinstance(tt, TokenType) and tt == TokenType.TRUE:
            self.advance()
            return SqlLiteral(True)
        if isinstance(tt, TokenType) and tt == TokenType.FALSE:
            self.advance()
            return SqlLiteral(False)

        # CASE expression
        if isinstance(tt, TokenType) and tt == TokenType.CASE:
            return self._parse_case()

        # EXISTS
        if isinstance(tt, TokenType) and tt == TokenType.EXISTS:
            self.advance()
            self.expect(TokenType.LPAREN)
            inner = self._parse_select()
            self.expect(TokenType.RPAREN)
            # Return as func call placeholder
            return SqlFuncCall(func_name='exists', args=[inner])

        # Aggregate functions -- only if followed by (
        agg_types = {TokenType.COUNT, TokenType.SUM, TokenType.AVG,
                     TokenType.MIN, TokenType.MAX}
        if isinstance(tt, TokenType) and tt in agg_types:
            if self.pos + 1 < len(self.tokens):
                next_t = self.tokens[self.pos + 1]
                next_tt = next_t.type if hasattr(next_t, 'type') else next_t.type
                if isinstance(next_tt, TokenType) and next_tt == TokenType.LPAREN:
                    return self._parse_agg_call()
            # Not followed by ( -- treat as variable/column name
            name = self.advance().value
            return SqlColumnRef(table=None, column=name)

        # Parenthesized expression or subquery
        if isinstance(tt, TokenType) and tt == TokenType.LPAREN:
            self.advance()
            # Check for subquery
            if self._peek_type() == TokenType.SELECT:
                inner = self._parse_select()
                self.expect(TokenType.RPAREN)
                return inner  # Return SelectStmt as expression
            expr = self._parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        # Identifier or function call
        if isinstance(tt, TokenType) and tt == TokenType.IDENT:
            return self._parse_ident_or_func()

        # Also allow proc keywords as identifiers
        if isinstance(tt, str):
            # Try as identifier
            val = self.advance().value
            # Check for dot notation
            if self._peek_type() == TokenType.DOT:
                self.advance()
                col = self._expect_ident()
                return SqlColumnRef(table=val, column=col)
            # Check for function call
            if isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.LPAREN:
                self.advance()
                args = []
                if self._peek_type() != TokenType.RPAREN:
                    while True:
                        args.append(self._parse_expr())
                        if not self.match(TokenType.COMMA):
                            break
                self.expect(TokenType.RPAREN)
                return SqlFuncCall(func_name=val, args=args)
            return SqlColumnRef(table=None, column=val)

        # Star
        if isinstance(tt, TokenType) and tt == TokenType.STAR:
            self.advance()
            return SqlStar(table=None)

        # Keywords that could be variable/function names in procedure context
        keyword_as_expr = {
            TokenType.ADD, TokenType.SET, TokenType.IN, TokenType.ON,
            TokenType.ORDER, TokenType.BY, TokenType.AS, TokenType.DEFAULT,
            TokenType.FLOAT, TokenType.INT, TokenType.TEXT, TokenType.BOOL,
            TokenType.JOIN, TokenType.LEFT, TokenType.RIGHT, TokenType.INNER,
            TokenType.DROP, TokenType.CREATE, TokenType.ALTER,
            TokenType.DISTINCT, TokenType.ALL, TokenType.UNION,
            TokenType.KEY, TokenType.INDEX, TokenType.TABLE,
            TokenType.DELETE, TokenType.UPDATE, TokenType.INSERT,
            TokenType.BEGIN, TokenType.END,
        }
        if isinstance(tt, TokenType) and tt in keyword_as_expr:
            val = self.advance().value
            # Check for function call
            if isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.LPAREN:
                self.advance()
                args = []
                if self._peek_type() != TokenType.RPAREN:
                    while True:
                        args.append(self._parse_expr())
                        if not self.match(TokenType.COMMA):
                            break
                self.expect(TokenType.RPAREN)
                return SqlFuncCall(func_name=val, args=args)
            return SqlColumnRef(table=None, column=val)

        raise ParseError(f"Unexpected token in expression: {tt} ({self.peek().value!r})")

    def _parse_ident_or_func(self):
        name = self.advance().value
        # Check for dot notation (table.column)
        if isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.DOT:
            self.advance()
            col = self._expect_ident()
            return SqlColumnRef(table=name, column=col)
        # Check for function call
        if isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.LPAREN:
            self.advance()
            args = []
            if self._peek_type() != TokenType.RPAREN:
                while True:
                    args.append(self._parse_expr())
                    if not self.match(TokenType.COMMA):
                        break
            self.expect(TokenType.RPAREN)
            return SqlFuncCall(func_name=name, args=args)
        return SqlColumnRef(table=None, column=name)

    def _parse_agg_call(self):
        func_name = self.advance().value.lower()
        self.expect(TokenType.LPAREN)
        distinct = bool(self.match(TokenType.DISTINCT))
        if isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.STAR:
            self.advance()
            arg = None
        else:
            arg = self._parse_expr()
        self.expect(TokenType.RPAREN)
        return SqlAggCall(func=func_name, arg=arg, distinct=distinct)

    def _parse_case(self):
        self.advance()  # CASE
        whens = []
        else_result = None
        while self.match(TokenType.WHEN):
            cond = self._parse_expr()
            self.expect(TokenType.THEN)
            result = self._parse_expr()
            whens.append((cond, result))
        if self.match(TokenType.ELSE):
            else_result = self._parse_expr()
        self.expect(TokenType.END)
        return SqlCase(whens=whens, else_result=else_result)

    def _parse_select_list(self):
        """Parse SELECT column list."""
        columns = []
        while True:
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
                    # Implicit alias
                    next_val = self.peek().value.lower()
                    if next_val not in ('from', 'where', 'into', 'group', 'order',
                                        'having', 'limit', 'union'):
                        alias = self.advance().value
                columns.append(SelectExpr(expr=expr, alias=alias))
            if not self.match(TokenType.COMMA):
                break
        return columns

    def _parse_table_ref(self):
        name = self._expect_ident()
        alias = None
        if self.match(TokenType.AS):
            alias = self._expect_ident()
        elif isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.IDENT:
            peek_val = self.peek().value.lower()
            if peek_val not in ('where', 'on', 'join', 'left', 'right', 'inner',
                                'cross', 'full', 'order', 'group', 'having',
                                'limit', 'union', 'set'):
                alias = self.advance().value
        return TableRef(table_name=name, alias=alias)

    def _try_parse_join(self):
        tt = self._peek_type()
        join_type = None

        if isinstance(tt, TokenType):
            if tt == TokenType.JOIN:
                join_type = 'inner'
            elif tt == TokenType.LEFT:
                join_type = 'left'
            elif tt == TokenType.RIGHT:
                join_type = 'right'
            elif tt == TokenType.CROSS:
                join_type = 'cross'
            elif tt == TokenType.INNER:
                join_type = 'inner'
            elif tt == TokenType.FULL:
                join_type = 'full'

        if join_type is None:
            return None

        self.advance()
        if join_type != 'cross':
            self.match(TokenType.OUTER)
        self.match(TokenType.JOIN)

        table = self._parse_table_ref()
        condition = None
        if self.match(TokenType.ON):
            condition = self._parse_expr()

        return JoinClause(join_type=join_type, table=table, condition=condition)

    def _parse_select(self) -> SelectStmt:
        self.expect(TokenType.SELECT)
        distinct = bool(self.match(TokenType.DISTINCT))
        columns = self._parse_select_list()

        from_table = None
        joins = []
        if self.match(TokenType.FROM):
            from_table = self._parse_table_ref()
            while True:
                join = self._try_parse_join()
                if join is None:
                    break
                joins.append(join)

        where = None
        if self.match(TokenType.WHERE):
            where = self._parse_expr()

        group_by = None
        if self.match(TokenType.GROUP):
            self.expect(TokenType.BY)
            group_by = []
            while True:
                group_by.append(self._parse_expr())
                if not self.match(TokenType.COMMA):
                    break

        having = None
        if self.match(TokenType.HAVING):
            having = self._parse_expr()

        order_by = None
        if self.match(TokenType.ORDER):
            self.expect(TokenType.BY)
            order_by = []
            while True:
                expr = self._parse_expr()
                asc = True
                if self.match(TokenType.DESC):
                    asc = False
                elif self.match(TokenType.ASC):
                    asc = True
                order_by.append((expr, asc))
                if not self.match(TokenType.COMMA):
                    break

        limit = None
        offset = None
        if self.match(TokenType.LIMIT):
            limit = int(self.expect(TokenType.NUMBER).value)
            if self.match(TokenType.OFFSET):
                offset = int(self.expect(TokenType.NUMBER).value)

        return SelectStmt(
            columns=columns, from_table=from_table, joins=joins,
            where=where, group_by=group_by, having=having,
            order_by=order_by, limit=limit, offset=offset,
            distinct=distinct
        )

    def _parse_insert(self):
        self.expect(TokenType.INSERT)
        self.expect(TokenType.INTO)
        table_name = self._expect_ident()

        columns = None
        if isinstance(self._peek_type(), TokenType) and self._peek_type() == TokenType.LPAREN:
            self.advance()
            columns = []
            while True:
                columns.append(self._expect_ident())
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RPAREN)

        self.expect(TokenType.VALUES)
        values_list = []
        while True:
            self.expect(TokenType.LPAREN)
            values = []
            while True:
                values.append(self._parse_expr())
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RPAREN)
            values_list.append(values)
            if not self.match(TokenType.COMMA):
                break

        return InsertStmt(table_name=table_name, columns=columns,
                         values_list=values_list)

    def _parse_update(self):
        self.expect(TokenType.UPDATE)
        table_name = self._expect_ident()
        self.expect(TokenType.SET)

        assignments = []
        while True:
            col = self._expect_ident()
            self.expect(TokenType.EQ)
            val = self._parse_expr()
            assignments.append((col, val))
            if not self.match(TokenType.COMMA):
                break

        where = None
        if self.match(TokenType.WHERE):
            where = self._parse_expr()

        return UpdateStmt(table_name=table_name, assignments=assignments,
                         where=where)

    def _parse_delete(self):
        self.expect(TokenType.DELETE)
        self.expect(TokenType.FROM)
        table_name = self._expect_ident()

        where = None
        if self.match(TokenType.WHERE):
            where = self._parse_expr()

        return DeleteStmt(table_name=table_name, where=where)


# =============================================================================
# Procedure/Function Execution Engine
# =============================================================================

class ReturnSignal(Exception):
    """Raised when RETURN is executed in a function body."""
    def __init__(self, value):
        self.value = value


class LeaveSignal(Exception):
    """Raised when LEAVE is executed."""
    def __init__(self, label=None):
        self.label = label


class IterateSignal(Exception):
    """Raised when ITERATE is executed."""
    def __init__(self, label=None):
        self.label = label


class HandlerSignal(Exception):
    """Raised when a handler catches an error."""
    def __init__(self, handler_type, body):
        self.handler_type = handler_type
        self.body = body


class ProcExecutor:
    """Executes stored procedures and functions."""

    MAX_RECURSION_DEPTH = 100

    def __init__(self, db: 'ProcDB'):
        self.db = db
        self._call_depth = 0

    def call_function(self, name: str, args: List[Any]) -> Any:
        """Call a stored function and return its scalar result."""
        func = self.db.routines.get_function(name)

        if self._call_depth >= self.MAX_RECURSION_DEPTH:
            raise DatabaseError(f"Maximum recursion depth ({self.MAX_RECURSION_DEPTH}) exceeded")

        # Bind parameters
        local_vars = {}
        for i, param in enumerate(func.params):
            if i < len(args):
                local_vars[param.name] = self._coerce_type(args[i], param.param_type)
            elif param.default is not None:
                local_vars[param.name] = self._eval_default(param.default)
            else:
                local_vars[param.name] = self._default_for_type(param.param_type)

        self._call_depth += 1
        try:
            self._exec_body(func.body, local_vars)
            # If no RETURN was hit, return default for type
            return self._default_for_type(func.return_type)
        except ReturnSignal as rs:
            return self._coerce_type(rs.value, func.return_type)
        finally:
            self._call_depth -= 1

    def call_procedure(self, name: str, args: List[Any]) -> Dict[str, Any]:
        """Call a stored procedure. Returns dict of OUT/INOUT parameter values."""
        proc = self.db.routines.get_procedure(name)

        if self._call_depth >= self.MAX_RECURSION_DEPTH:
            raise DatabaseError(f"Maximum recursion depth ({self.MAX_RECURSION_DEPTH}) exceeded")

        # Bind parameters
        local_vars = {}
        out_params = {}
        for i, param in enumerate(proc.params):
            if i < len(args):
                local_vars[param.name] = args[i]
            elif param.default is not None:
                local_vars[param.name] = self._eval_default(param.default)
            else:
                local_vars[param.name] = None
            if param.mode in (ParamMode.OUT, ParamMode.INOUT):
                out_params[param.name] = param

        self._call_depth += 1
        try:
            self._exec_body(proc.body, local_vars)
        except ReturnSignal:
            pass  # Procedures can use RETURN without value to exit early
        finally:
            self._call_depth -= 1

        # Return OUT/INOUT values
        result = {}
        for pname in out_params:
            result[pname] = local_vars.get(pname)
        return result

    def _exec_body(self, stmts: List[Any], local_vars: Dict[str, Any]):
        """Execute a list of body statements."""
        handlers = []

        for stmt in stmts:
            if isinstance(stmt, DeclareVarStmt):
                if stmt.default is not None:
                    local_vars[stmt.name] = self._eval_sql_expr(stmt.default, local_vars)
                else:
                    local_vars[stmt.name] = self._default_for_type(stmt.var_type)
                continue

            if isinstance(stmt, DeclareHandlerStmt):
                handlers.append(stmt)
                continue

            try:
                self._exec_stmt(stmt, local_vars)
            except ReturnSignal:
                raise
            except LeaveSignal:
                raise
            except IterateSignal:
                raise
            except (DatabaseError, CatalogError, CompileError, ParseError) as e:
                # Check handlers
                handled = False
                for h in handlers:
                    if h.condition == 'sqlexception':
                        handled = True
                        self._exec_body(h.body, local_vars)
                        if h.handler_type == 'exit':
                            return
                        break
                if not handled:
                    raise

    def _exec_stmt(self, stmt: Any, local_vars: Dict[str, Any]):
        """Execute a single body statement."""
        if isinstance(stmt, SetVarStmt):
            value = self._eval_sql_expr(stmt.expr, local_vars)
            local_vars[stmt.name] = value

        elif isinstance(stmt, ReturnStmt):
            value = self._eval_sql_expr(stmt.expr, local_vars)
            raise ReturnSignal(value)

        elif isinstance(stmt, IfStmt):
            self._exec_if(stmt, local_vars)

        elif isinstance(stmt, WhileStmt):
            self._exec_while(stmt, local_vars)

        elif isinstance(stmt, LeaveStmt):
            raise LeaveSignal(stmt.label)

        elif isinstance(stmt, IterateStmt):
            raise IterateSignal(stmt.label)

        elif isinstance(stmt, CallStmt):
            # Execute nested call
            args = [self._eval_sql_expr(a, local_vars) for a in stmt.args]
            if stmt.name in self.db.routines.procedures:
                result = self.call_procedure(stmt.name, args)
                # Copy OUT params back to local vars
                for pname, val in result.items():
                    if pname in local_vars:
                        local_vars[pname] = val
            elif stmt.name in self.db.routines.functions:
                self.call_function(stmt.name, args)
            else:
                raise DatabaseError(f"Routine '{stmt.name}' not found")

        elif isinstance(stmt, SelectIntoStmt):
            self._exec_select_into(stmt, local_vars)

        elif isinstance(stmt, InsertStmt):
            self._exec_sql_in_body(stmt, local_vars)

        elif isinstance(stmt, UpdateStmt):
            self._exec_sql_in_body(stmt, local_vars)

        elif isinstance(stmt, DeleteStmt):
            self._exec_sql_in_body(stmt, local_vars)

        elif isinstance(stmt, SelectStmt):
            # SELECT without INTO -- just execute (side effects or discard)
            self._exec_sql_in_body(stmt, local_vars)

        elif isinstance(stmt, DeclareVarStmt):
            if stmt.default is not None:
                local_vars[stmt.name] = self._eval_sql_expr(stmt.default, local_vars)
            else:
                local_vars[stmt.name] = self._default_for_type(stmt.var_type)

        elif isinstance(stmt, DeclareHandlerStmt):
            pass  # Handlers are collected at block level

        else:
            raise DatabaseError(f"Unsupported body statement: {type(stmt).__name__}")

    def _exec_if(self, stmt: IfStmt, local_vars: Dict[str, Any]):
        cond = self._eval_sql_expr(stmt.condition, local_vars)
        if cond:
            self._exec_body(stmt.then_body, local_vars)
            return

        for elif_cond, elif_body in stmt.elseif_clauses:
            if self._eval_sql_expr(elif_cond, local_vars):
                self._exec_body(elif_body, local_vars)
                return

        if stmt.else_body:
            self._exec_body(stmt.else_body, local_vars)

    def _exec_while(self, stmt: WhileStmt, local_vars: Dict[str, Any]):
        iteration = 0
        max_iterations = 100000

        while iteration < max_iterations:
            cond = self._eval_sql_expr(stmt.condition, local_vars)
            if not cond:
                break

            try:
                self._exec_body(stmt.body, local_vars)
            except LeaveSignal as ls:
                if ls.label is None or ls.label == stmt.label:
                    break
                raise
            except IterateSignal as it:
                if it.label is None or it.label == stmt.label:
                    iteration += 1
                    continue
                raise

            iteration += 1

        if iteration >= max_iterations:
            raise DatabaseError("WHILE loop exceeded maximum iterations")

    def _exec_select_into(self, stmt: SelectIntoStmt, local_vars: Dict[str, Any]):
        """Execute SELECT ... INTO vars."""
        # Substitute local vars into the query
        modified_select = self._substitute_vars_in_select(stmt.select, local_vars)
        result = self.db._exec_select(modified_select)
        if result.rows:
            row = result.rows[0]
            for i, var_name in enumerate(stmt.variables):
                if i < len(row):
                    local_vars[var_name] = row[i]

    def _exec_sql_in_body(self, stmt: Any, local_vars: Dict[str, Any]):
        """Execute a SQL DML statement inside a procedure body, substituting variables."""
        # Substitute variable references in the statement
        modified = self._substitute_vars_in_stmt(stmt, local_vars)
        self.db._execute_stmt(modified)

    def _substitute_vars_in_stmt(self, stmt: Any, local_vars: Dict[str, Any]) -> Any:
        """Replace variable references with literal values in SQL statements."""
        if isinstance(stmt, InsertStmt):
            new_values = []
            for row_vals in stmt.values_list:
                new_row = [self._substitute_expr(v, local_vars) for v in row_vals]
                new_values.append(new_row)
            return InsertStmt(
                table_name=stmt.table_name,
                columns=stmt.columns,
                values_list=new_values
            )
        if isinstance(stmt, UpdateStmt):
            new_assignments = [
                (col, self._substitute_expr(val, local_vars))
                for col, val in stmt.assignments
            ]
            new_where = self._substitute_expr(stmt.where, local_vars) if stmt.where else None
            return UpdateStmt(
                table_name=stmt.table_name,
                assignments=new_assignments,
                where=new_where
            )
        if isinstance(stmt, DeleteStmt):
            new_where = self._substitute_expr(stmt.where, local_vars) if stmt.where else None
            return DeleteStmt(table_name=stmt.table_name, where=new_where)
        if isinstance(stmt, SelectStmt):
            return self._substitute_vars_in_select(stmt, local_vars)
        return stmt

    def _substitute_vars_in_select(self, stmt: SelectStmt, local_vars: Dict[str, Any]) -> SelectStmt:
        """Substitute variable references in a SELECT statement."""
        new_cols = []
        for se in stmt.columns:
            new_expr = self._substitute_expr(se.expr, local_vars)
            new_cols.append(SelectExpr(expr=new_expr, alias=se.alias))

        new_where = self._substitute_expr(stmt.where, local_vars) if stmt.where else None
        new_having = self._substitute_expr(stmt.having, local_vars) if stmt.having else None

        return SelectStmt(
            columns=new_cols,
            from_table=stmt.from_table,
            joins=stmt.joins,
            where=new_where,
            group_by=stmt.group_by,
            having=new_having,
            order_by=stmt.order_by,
            limit=stmt.limit,
            offset=stmt.offset,
            distinct=stmt.distinct
        )

    def _substitute_expr(self, expr: Any, local_vars: Dict[str, Any]) -> Any:
        """Replace SqlColumnRef with SqlLiteral if the column name is a local variable."""
        if expr is None:
            return None

        if isinstance(expr, SqlColumnRef):
            if expr.table is None and expr.column in local_vars:
                return SqlLiteral(local_vars[expr.column])
            return expr

        if isinstance(expr, SqlBinOp):
            return SqlBinOp(
                op=expr.op,
                left=self._substitute_expr(expr.left, local_vars),
                right=self._substitute_expr(expr.right, local_vars)
            )

        if isinstance(expr, SqlComparison):
            return SqlComparison(
                op=expr.op,
                left=self._substitute_expr(expr.left, local_vars),
                right=self._substitute_expr(expr.right, local_vars)
            )

        if isinstance(expr, SqlLogic):
            return SqlLogic(
                op=expr.op,
                operands=[self._substitute_expr(o, local_vars) for o in expr.operands]
            )

        if isinstance(expr, SqlIsNull):
            return SqlIsNull(
                expr=self._substitute_expr(expr.expr, local_vars),
                negated=expr.negated
            )

        if isinstance(expr, SqlFuncCall):
            return SqlFuncCall(
                func_name=expr.func_name,
                args=[self._substitute_expr(a, local_vars) for a in expr.args]
            )

        if isinstance(expr, SqlAggCall):
            new_arg = self._substitute_expr(expr.arg, local_vars) if expr.arg else None
            return SqlAggCall(func=expr.func, arg=new_arg, distinct=expr.distinct)

        if isinstance(expr, SqlCase):
            new_whens = [
                (self._substitute_expr(c, local_vars), self._substitute_expr(r, local_vars))
                for c, r in expr.whens
            ]
            new_else = self._substitute_expr(expr.else_result, local_vars) if expr.else_result else None
            return SqlCase(whens=new_whens, else_result=new_else)

        if isinstance(expr, SqlBetween):
            return SqlBetween(
                expr=self._substitute_expr(expr.expr, local_vars),
                low=self._substitute_expr(expr.low, local_vars),
                high=self._substitute_expr(expr.high, local_vars)
            )

        if isinstance(expr, SqlInList):
            return SqlInList(
                expr=self._substitute_expr(expr.expr, local_vars),
                values=[self._substitute_expr(v, local_vars) for v in expr.values]
            )

        return expr

    def _eval_sql_expr(self, expr: Any, local_vars: Dict[str, Any]) -> Any:
        """Evaluate a SQL expression against local variables."""
        if isinstance(expr, SqlLiteral):
            return expr.value

        if isinstance(expr, SqlColumnRef):
            if expr.table is None and expr.column in local_vars:
                return local_vars[expr.column]
            raise DatabaseError(f"Variable '{expr.column}' not declared")

        if isinstance(expr, SqlBinOp):
            left = self._eval_sql_expr(expr.left, local_vars)
            right = self._eval_sql_expr(expr.right, local_vars)
            if expr.op == '+': return left + right
            if expr.op == '-': return left - right
            if expr.op == '*': return left * right
            if expr.op == '/':
                if right == 0:
                    raise DatabaseError("Division by zero")
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                return left / right
            if expr.op == '%':
                if right == 0:
                    raise DatabaseError("Division by zero")
                return left % right

        if isinstance(expr, SqlComparison):
            left = self._eval_sql_expr(expr.left, local_vars)
            right = self._eval_sql_expr(expr.right, local_vars)
            if expr.op == '=': return left == right
            if expr.op == '!=': return left != right
            if expr.op == '<': return left < right
            if expr.op == '<=': return left <= right
            if expr.op == '>': return left > right
            if expr.op == '>=': return left >= right

        if isinstance(expr, SqlLogic):
            if expr.op == 'and':
                return all(self._eval_sql_expr(o, local_vars) for o in expr.operands)
            if expr.op == 'or':
                return any(self._eval_sql_expr(o, local_vars) for o in expr.operands)
            if expr.op == 'not':
                return not self._eval_sql_expr(expr.operands[0], local_vars)

        if isinstance(expr, SqlIsNull):
            val = self._eval_sql_expr(expr.expr, local_vars)
            if expr.negated:
                return val is not None
            return val is None

        if isinstance(expr, SqlFuncCall):
            # Check if it's a stored function
            if expr.func_name in self.db.routines.functions:
                args = [self._eval_sql_expr(a, local_vars) for a in expr.args]
                return self.call_function(expr.func_name, args)
            # Built-in functions
            args = [self._eval_sql_expr(a, local_vars) for a in expr.args]
            return self._eval_builtin_func(expr.func_name, args)

        if isinstance(expr, SqlCase):
            for cond, result in expr.whens:
                if self._eval_sql_expr(cond, local_vars):
                    return self._eval_sql_expr(result, local_vars)
            if expr.else_result:
                return self._eval_sql_expr(expr.else_result, local_vars)
            return None

        if isinstance(expr, SqlBetween):
            val = self._eval_sql_expr(expr.expr, local_vars)
            low = self._eval_sql_expr(expr.low, local_vars)
            high = self._eval_sql_expr(expr.high, local_vars)
            return low <= val <= high

        if isinstance(expr, SqlInList):
            val = self._eval_sql_expr(expr.expr, local_vars)
            values = [self._eval_sql_expr(v, local_vars) for v in expr.values]
            return val in values

        raise DatabaseError(f"Cannot evaluate expression: {type(expr).__name__}")

    def _eval_builtin_func(self, name: str, args: List[Any]) -> Any:
        """Evaluate built-in scalar functions."""
        name_lower = name.lower()
        if name_lower == 'abs':
            return abs(args[0])
        if name_lower == 'upper':
            return str(args[0]).upper()
        if name_lower == 'lower':
            return str(args[0]).lower()
        if name_lower == 'length' or name_lower == 'len':
            return len(str(args[0]))
        if name_lower == 'concat':
            return ''.join(str(a) for a in args)
        if name_lower == 'coalesce':
            for a in args:
                if a is not None:
                    return a
            return None
        if name_lower == 'ifnull':
            return args[0] if args[0] is not None else args[1]
        if name_lower == 'nullif':
            return None if args[0] == args[1] else args[0]
        if name_lower == 'cast':
            return args[0]  # Simplified
        if name_lower == 'substr' or name_lower == 'substring':
            s = str(args[0])
            start = int(args[1]) - 1  # 1-indexed
            if len(args) > 2:
                length = int(args[2])
                return s[start:start + length]
            return s[start:]
        if name_lower == 'replace':
            return str(args[0]).replace(str(args[1]), str(args[2]))
        if name_lower == 'trim':
            return str(args[0]).strip()
        if name_lower == 'round':
            if len(args) > 1:
                return round(args[0], int(args[1]))
            return round(args[0])
        if name_lower == 'floor':
            import math
            return math.floor(args[0])
        if name_lower == 'ceil' or name_lower == 'ceiling':
            import math
            return math.ceil(args[0])
        if name_lower == 'mod':
            return args[0] % args[1]
        if name_lower == 'power' or name_lower == 'pow':
            return args[0] ** args[1]
        if name_lower == 'sqrt':
            return args[0] ** 0.5
        if name_lower == 'greatest':
            return max(args)
        if name_lower == 'least':
            return min(args)
        raise DatabaseError(f"Unknown function: {name}")

    def _eval_default(self, default_expr):
        """Evaluate a default value expression."""
        if isinstance(default_expr, SqlLiteral):
            return default_expr.value
        return self._eval_sql_expr(default_expr, {})

    @staticmethod
    def _coerce_type(value: Any, type_name: str) -> Any:
        """Coerce a value to the specified type."""
        if value is None:
            return None
        type_lower = type_name.lower()
        if type_lower in ('int', 'integer'):
            return int(value)
        if type_lower in ('float', 'real', 'double'):
            return float(value)
        if type_lower in ('text', 'varchar', 'string'):
            return str(value)
        if type_lower in ('bool', 'boolean'):
            return bool(value)
        return value

    @staticmethod
    def _default_for_type(type_name: str) -> Any:
        """Return default value for a type."""
        type_lower = type_name.lower()
        if type_lower in ('int', 'integer'):
            return 0
        if type_lower in ('float', 'real', 'double'):
            return 0.0
        if type_lower in ('text', 'varchar', 'string'):
            return ''
        if type_lower in ('bool', 'boolean'):
            return False
        return None


# =============================================================================
# Extended MiniDB with Procedure/Function Support
# =============================================================================

class ProcDB(MiniDB):
    """MiniDB extended with stored procedures and user-defined functions."""

    def __init__(self, pool_size: int = 64,
                 isolation: IsolationLevel = IsolationLevel.REPEATABLE_READ):
        super().__init__(pool_size=pool_size, isolation=isolation)
        self.routines = RoutineCatalog()
        self.executor = ProcExecutor(self)
        # Replace compiler with proc-aware compiler
        self.proc_compiler = ProcQueryCompiler(self.storage, self)

    def execute(self, sql: str) -> 'ResultSet':
        """Execute SQL with procedure/function support."""
        stmts = self._parse_proc(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_stmt(stmt))
        # Return last result
        return results[-1] if results else ResultSet(columns=[], rows=[], message="OK")

    def execute_many(self, sql: str) -> List['ResultSet']:
        """Execute multiple SQL statements."""
        stmts = self._parse_proc(sql)
        results = []
        for stmt in stmts:
            results.append(self._execute_stmt(stmt))
        return results

    def _parse_proc(self, sql: str) -> List[Any]:
        """Parse SQL using the extended lexer/parser."""
        lexer = ProcLexer(sql)
        parser = ProcParser(lexer.tokens)
        stmts = []
        while parser._peek_type() != TokenType.EOF:
            stmts.append(parser._parse_statement())
            parser.match(TokenType.SEMICOLON)
        return stmts

    def _execute_stmt(self, stmt) -> 'ResultSet':
        # Extended statement types
        if isinstance(stmt, CreateFunctionStmt):
            return self._exec_create_function(stmt)
        if isinstance(stmt, CreateProcedureStmt):
            return self._exec_create_procedure(stmt)
        if isinstance(stmt, DropFunctionStmt):
            return self._exec_drop_function(stmt)
        if isinstance(stmt, DropProcedureStmt):
            return self._exec_drop_procedure(stmt)
        if isinstance(stmt, CallStmt):
            return self._exec_call(stmt)
        if isinstance(stmt, ShowFunctionsStmt):
            return self._exec_show_functions()
        if isinstance(stmt, ShowProceduresStmt):
            return self._exec_show_procedures()

        # For SELECT statements, use proc-aware compiler to resolve UDFs
        if isinstance(stmt, SelectStmt):
            return self._exec_select_with_udfs(stmt)

        # Standard statements handled by parent
        return super()._execute_stmt(stmt)

    def _exec_create_function(self, stmt: CreateFunctionStmt) -> ResultSet:
        func = StoredFunction(
            name=stmt.name,
            params=stmt.params,
            return_type=stmt.return_type,
            body=stmt.body,
            deterministic=stmt.deterministic
        )
        self.routines.create_function(func, replace=stmt.replace)
        return ResultSet(columns=[], rows=[], message=f"CREATE FUNCTION {stmt.name}")

    def _exec_create_procedure(self, stmt: CreateProcedureStmt) -> ResultSet:
        proc = StoredProcedure(
            name=stmt.name,
            params=stmt.params,
            body=stmt.body
        )
        self.routines.create_procedure(proc, replace=stmt.replace)
        return ResultSet(columns=[], rows=[], message=f"CREATE PROCEDURE {stmt.name}")

    def _exec_drop_function(self, stmt: DropFunctionStmt) -> ResultSet:
        self.routines.drop_function(stmt.name, if_exists=stmt.if_exists)
        return ResultSet(columns=[], rows=[], message=f"DROP FUNCTION {stmt.name}")

    def _exec_drop_procedure(self, stmt: DropProcedureStmt) -> ResultSet:
        self.routines.drop_procedure(stmt.name, if_exists=stmt.if_exists)
        return ResultSet(columns=[], rows=[], message=f"DROP PROCEDURE {stmt.name}")

    def _exec_call(self, stmt: CallStmt) -> ResultSet:
        args = []
        for arg in stmt.args:
            if isinstance(arg, SqlLiteral):
                args.append(arg.value)
            else:
                # Evaluate expression
                args.append(self.executor._eval_sql_expr(arg, {}))

        if stmt.name in self.routines.procedures:
            out_values = self.executor.call_procedure(stmt.name, args)
            if out_values:
                columns = list(out_values.keys())
                rows = [list(out_values.values())]
                return ResultSet(columns=columns, rows=rows,
                                message=f"CALL {stmt.name}")
            return ResultSet(columns=[], rows=[], message=f"CALL {stmt.name}")
        elif stmt.name in self.routines.functions:
            result = self.executor.call_function(stmt.name, args)
            return ResultSet(columns=['result'], rows=[[result]],
                            message=f"CALL {stmt.name}")
        else:
            raise DatabaseError(f"Routine '{stmt.name}' not found")

    def _exec_show_functions(self) -> ResultSet:
        funcs = self.routines.list_functions()
        return ResultSet(columns=['function_name'],
                        rows=[[f] for f in funcs])

    def _exec_show_procedures(self) -> ResultSet:
        procs = self.routines.list_procedures()
        return ResultSet(columns=['procedure_name'],
                        rows=[[p] for p in procs])

    def _exec_select_with_udfs(self, stmt: SelectStmt) -> ResultSet:
        """Execute SELECT with UDF resolution in expressions."""
        # First resolve any UDF calls in the SELECT columns
        resolved_stmt = self._resolve_udfs_in_select(stmt)
        return super()._exec_select(resolved_stmt)

    def _resolve_udfs_in_select(self, stmt: SelectStmt) -> SelectStmt:
        """Pre-evaluate UDF calls in SELECT expressions, replacing with literals."""
        new_cols = []
        for se in stmt.columns:
            new_expr = self._resolve_expr_udfs(se.expr)
            new_cols.append(SelectExpr(expr=new_expr, alias=se.alias))

        new_where = self._resolve_expr_udfs(stmt.where) if stmt.where else None

        return SelectStmt(
            columns=new_cols,
            from_table=stmt.from_table,
            joins=stmt.joins,
            where=new_where,
            group_by=stmt.group_by,
            having=stmt.having,
            order_by=stmt.order_by,
            limit=stmt.limit,
            offset=stmt.offset,
            distinct=stmt.distinct
        )

    def _resolve_expr_udfs(self, expr: Any) -> Any:
        """Resolve UDF calls in expressions that don't reference table columns."""
        if expr is None:
            return None

        if isinstance(expr, SqlFuncCall):
            if expr.func_name in self.routines.functions:
                # Check if all args are constant (no column refs)
                args = []
                all_const = True
                for a in expr.args:
                    resolved = self._resolve_expr_udfs(a)
                    if self._has_column_ref(resolved):
                        all_const = False
                    args.append(resolved)

                if all_const:
                    # Evaluate now
                    arg_vals = [self._eval_const_expr(a) for a in args]
                    result = self.executor.call_function(expr.func_name, arg_vals)
                    return SqlLiteral(result)
                else:
                    return SqlFuncCall(func_name=expr.func_name, args=args)

            # Built-in function -- recurse into args
            new_args = [self._resolve_expr_udfs(a) for a in expr.args]
            return SqlFuncCall(func_name=expr.func_name, args=new_args)

        if isinstance(expr, SqlBinOp):
            return SqlBinOp(
                op=expr.op,
                left=self._resolve_expr_udfs(expr.left),
                right=self._resolve_expr_udfs(expr.right)
            )

        if isinstance(expr, SqlComparison):
            return SqlComparison(
                op=expr.op,
                left=self._resolve_expr_udfs(expr.left),
                right=self._resolve_expr_udfs(expr.right)
            )

        if isinstance(expr, SqlLogic):
            return SqlLogic(
                op=expr.op,
                operands=[self._resolve_expr_udfs(o) for o in expr.operands]
            )

        if isinstance(expr, SqlCase):
            new_whens = [
                (self._resolve_expr_udfs(c), self._resolve_expr_udfs(r))
                for c, r in expr.whens
            ]
            new_else = self._resolve_expr_udfs(expr.else_result) if expr.else_result else None
            return SqlCase(whens=new_whens, else_result=new_else)

        return expr

    def _has_column_ref(self, expr: Any) -> bool:
        if isinstance(expr, SqlColumnRef):
            return True
        if isinstance(expr, SqlBinOp):
            return self._has_column_ref(expr.left) or self._has_column_ref(expr.right)
        if isinstance(expr, SqlComparison):
            return self._has_column_ref(expr.left) or self._has_column_ref(expr.right)
        if isinstance(expr, SqlLogic):
            return any(self._has_column_ref(o) for o in expr.operands)
        if isinstance(expr, SqlFuncCall):
            return any(self._has_column_ref(a) for a in expr.args)
        return False

    def _eval_const_expr(self, expr: Any) -> Any:
        """Evaluate a constant expression (no column refs)."""
        if isinstance(expr, SqlLiteral):
            return expr.value
        if isinstance(expr, SqlBinOp):
            left = self._eval_const_expr(expr.left)
            right = self._eval_const_expr(expr.right)
            if expr.op == '+': return left + right
            if expr.op == '-': return left - right
            if expr.op == '*': return left * right
            if expr.op == '/': return left / right
            if expr.op == '%': return left % right
        if isinstance(expr, SqlFuncCall):
            args = [self._eval_const_expr(a) for a in expr.args]
            if expr.func_name in self.routines.functions:
                return self.executor.call_function(expr.func_name, args)
            return self.executor._eval_builtin_func(expr.func_name, args)
        return None


class ProcQueryCompiler(QueryCompiler):
    """QueryCompiler extended with UDF awareness."""

    def __init__(self, storage: StorageEngine, proc_db: ProcDB):
        super().__init__(storage)
        self.proc_db = proc_db


# =============================================================================
# Public API
# =============================================================================

def proc_parse(sql: str) -> List[Any]:
    """Parse SQL with procedure/function extensions."""
    lexer = ProcLexer(sql)
    parser = ProcParser(lexer.tokens)
    stmts = []
    while parser._peek_type() != TokenType.EOF:
        stmts.append(parser._parse_statement())
        parser.match(TokenType.SEMICOLON)
    return stmts
