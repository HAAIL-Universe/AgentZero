"""
Effect Handler Runtime
Challenge C033 -- AgentZero Session 034

Composes C032 (Effect System) + C010 (Stack VM) to create a runtime that
actually executes programs with algebraic effects, handlers, and resumable
continuations.

Features:
  - Full C010 base language (ints, floats, bools, strings, let, if/else, while,
    fn, print, return)
  - Effect declarations: effect Name { op(params); ... }
  - Perform: perform Name.op(args) -- triggers an effect operation
  - Handle/with: handle { body } with { Name.op(params, k) -> expr; ... }
  - Resume: resume(k, value) -- resumes a captured continuation
  - Multi-shot continuations: resume the same continuation multiple times
  - Nested handlers: inner handlers shadow outer ones
  - Effect forwarding: re-perform unhandled effects to outer handlers
  - Try/catch for Error effects
  - State effects with get/set operations

Architecture:
  Source -> Lex -> Parse -> AST -> Compiler -> Bytecode -> EffectVM -> Result

The key innovation is the continuation capture mechanism:
  When `perform` is executed, the VM captures everything between the perform
  site and the handler installation point as a delimited continuation. This
  continuation is a first-class value that can be passed to the handler and
  resumed zero or more times.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import IntEnum, auto
import copy


# ============================================================
# Instruction Set (extends C010 with effect operations)
# ============================================================

class Op(IntEnum):
    # Stack
    CONST = auto()
    POP = auto()
    DUP = auto()

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()

    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    # Logic
    NOT = auto()
    AND = auto()
    OR = auto()

    # Variables
    LOAD = auto()
    STORE = auto()

    # Control flow
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    JUMP_IF_TRUE = auto()

    # Functions
    CALL = auto()
    RETURN = auto()

    # I/O
    PRINT = auto()

    # Halt
    HALT = auto()

    # === Effect operations ===
    PERFORM = auto()          # perform effect.op -- args on stack, effect_key in constants
    INSTALL_HANDLER = auto()  # install handler frame, operand = handler descriptor index
    REMOVE_HANDLER = auto()   # remove top handler frame
    RESUME = auto()           # resume(continuation, value)
    MAKE_CONTINUATION = auto()  # internal: package continuation object


# ============================================================
# Bytecode
# ============================================================

@dataclass
class Chunk:
    code: list = field(default_factory=list)
    constants: list = field(default_factory=list)
    names: list = field(default_factory=list)
    lines: list = field(default_factory=list)

    def emit(self, op, operand=None, line=0):
        addr = len(self.code)
        self.code.append(op)
        self.lines.append(line)
        if operand is not None:
            self.code.append(operand)
            self.lines.append(line)
        return addr

    def add_constant(self, value):
        for i, c in enumerate(self.constants):
            if c is value or (type(c) is type(value) and c == value):
                return i
        self.constants.append(value)
        return len(self.constants) - 1

    def add_name(self, name):
        if name in self.names:
            return self.names.index(name)
        self.names.append(name)
        return len(self.names) - 1

    def patch(self, addr, value):
        self.code[addr] = value


# ============================================================
# Lexer
# ============================================================

class TokenType(IntEnum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    IDENT = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    ASSIGN = auto()
    NOT = auto()
    AND = auto()
    OR = auto()

    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()
    DOT = auto()
    ARROW = auto()

    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FN = auto()
    RETURN = auto()
    PRINT = auto()
    EFFECT = auto()
    PERFORM = auto()
    HANDLE = auto()
    WITH = auto()
    RESUME = auto()
    TRY = auto()
    CATCH = auto()
    THROW = auto()

    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int


KEYWORDS = {
    'let': TokenType.LET,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'while': TokenType.WHILE,
    'fn': TokenType.FN,
    'return': TokenType.RETURN,
    'print': TokenType.PRINT,
    'true': TokenType.TRUE,
    'false': TokenType.FALSE,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'effect': TokenType.EFFECT,
    'perform': TokenType.PERFORM,
    'handle': TokenType.HANDLE,
    'with': TokenType.WITH,
    'resume': TokenType.RESUME,
    'try': TokenType.TRY,
    'catch': TokenType.CATCH,
    'throw': TokenType.THROW,
}


class LexError(Exception):
    pass


def lex(source: str) -> list:
    tokens = []
    i = 0
    line = 1

    while i < len(source):
        c = source[i]

        if c == '\n':
            line += 1
            i += 1
        elif c in ' \t\r':
            i += 1
        elif c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
        elif c.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            if i < len(source) and source[i] == '.' and i + 1 < len(source) and source[i + 1].isdigit():
                i += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
                tokens.append(Token(TokenType.FLOAT, float(source[start:i]), line))
            else:
                tokens.append(Token(TokenType.INT, int(source[start:i]), line))
        elif c == '"':
            i += 1
            start = i
            while i < len(source) and source[i] != '"':
                if source[i] == '\n':
                    line += 1
                i += 1
            if i >= len(source):
                raise LexError(f"Unterminated string at line {line}")
            tokens.append(Token(TokenType.STRING, source[start:i], line))
            i += 1
        elif c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line))
        # Two-char operators
        elif c == '=' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.EQ, '==', line)); i += 2
        elif c == '!' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.NE, '!=', line)); i += 2
        elif c == '<' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.LE, '<=', line)); i += 2
        elif c == '>' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.GE, '>=', line)); i += 2
        elif c == '-' and i + 1 < len(source) and source[i + 1] == '>':
            tokens.append(Token(TokenType.ARROW, '->', line)); i += 2
        # Single-char
        elif c == '+':
            tokens.append(Token(TokenType.PLUS, '+', line)); i += 1
        elif c == '-':
            tokens.append(Token(TokenType.MINUS, '-', line)); i += 1
        elif c == '*':
            tokens.append(Token(TokenType.STAR, '*', line)); i += 1
        elif c == '/':
            tokens.append(Token(TokenType.SLASH, '/', line)); i += 1
        elif c == '%':
            tokens.append(Token(TokenType.PERCENT, '%', line)); i += 1
        elif c == '<':
            tokens.append(Token(TokenType.LT, '<', line)); i += 1
        elif c == '>':
            tokens.append(Token(TokenType.GT, '>', line)); i += 1
        elif c == '=':
            tokens.append(Token(TokenType.ASSIGN, '=', line)); i += 1
        elif c == '(':
            tokens.append(Token(TokenType.LPAREN, '(', line)); i += 1
        elif c == ')':
            tokens.append(Token(TokenType.RPAREN, ')', line)); i += 1
        elif c == '{':
            tokens.append(Token(TokenType.LBRACE, '{', line)); i += 1
        elif c == '}':
            tokens.append(Token(TokenType.RBRACE, '}', line)); i += 1
        elif c == ',':
            tokens.append(Token(TokenType.COMMA, ',', line)); i += 1
        elif c == ';':
            tokens.append(Token(TokenType.SEMICOLON, ';', line)); i += 1
        elif c == '.':
            tokens.append(Token(TokenType.DOT, '.', line)); i += 1
        else:
            raise LexError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens


# ============================================================
# AST Nodes
# ============================================================

@dataclass
class IntLit:
    value: int
    line: int = 0

@dataclass
class FloatLit:
    value: float
    line: int = 0

@dataclass
class StringLit:
    value: str
    line: int = 0

@dataclass
class BoolLit:
    value: bool
    line: int = 0

@dataclass
class Var:
    name: str
    line: int = 0

@dataclass
class UnaryOp:
    op: str
    operand: Any
    line: int = 0

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    line: int = 0

@dataclass
class Assign:
    name: str
    value: Any
    line: int = 0

@dataclass
class LetDecl:
    name: str
    value: Any
    line: int = 0

@dataclass
class Block:
    stmts: list
    line: int = 0

@dataclass
class IfStmt:
    cond: Any
    then_body: Any
    else_body: Any = None
    line: int = 0

@dataclass
class WhileStmt:
    cond: Any
    body: Any
    line: int = 0

@dataclass
class FnDecl:
    name: str
    params: list
    body: Any
    line: int = 0

@dataclass
class CallExpr:
    callee: str
    args: list
    line: int = 0

@dataclass
class ReturnStmt:
    value: Any = None
    line: int = 0

@dataclass
class PrintStmt:
    value: Any
    line: int = 0

@dataclass
class Program:
    stmts: list

# Effect-specific AST nodes

@dataclass
class EffectDecl:
    """effect Name { op1(params); op2(params); ... }"""
    name: str
    operations: list  # list of (op_name, param_names)
    line: int = 0

@dataclass
class PerformExpr:
    """perform Name.op(args)"""
    effect: str
    operation: str
    args: list
    line: int = 0

@dataclass
class HandleWith:
    """handle { body } with { Name.op(params, k) -> expr; ... }"""
    body: Any
    handlers: list  # list of HandlerClause
    return_clause: Any = None  # optional: return(x) -> expr
    line: int = 0

@dataclass
class HandlerClause:
    """Name.op(params, k) -> body"""
    effect: str
    operation: str
    params: list      # parameter names for the operation args
    cont_name: str    # name for the continuation parameter
    body: Any         # handler body
    line: int = 0

@dataclass
class ReturnClause:
    """return(x) -> body -- transforms the handled body's return value"""
    param: str
    body: Any
    line: int = 0

@dataclass
class ResumeExpr:
    """resume(k, value)"""
    cont: Any   # continuation expression
    value: Any  # value to resume with
    line: int = 0

@dataclass
class ThrowExpr:
    """throw(value)"""
    value: Any
    line: int = 0

@dataclass
class TryCatch:
    """try { body } catch(name) { handler }"""
    body: Any
    catch_name: str
    catch_body: Any
    line: int = 0


# ============================================================
# Parser
# ============================================================

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt):
        t = self.advance()
        if t.type != tt:
            raise ParseError(f"Expected {tt.name}, got {t.type.name} '{t.value}' at line {t.line}")
        return t

    def match(self, *types):
        if self.peek().type in types:
            return self.advance()
        return None

    def parse(self):
        stmts = []
        while self.peek().type != TokenType.EOF:
            stmts.append(self.declaration())
        return Program(stmts)

    def declaration(self):
        if self.peek().type == TokenType.FN:
            return self.fn_decl()
        if self.peek().type == TokenType.LET:
            return self.let_decl()
        if self.peek().type == TokenType.EFFECT:
            return self.effect_decl()
        return self.statement()

    def effect_decl(self):
        tok = self.advance()  # effect
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LBRACE)
        operations = []
        while self.peek().type != TokenType.RBRACE:
            op_name = self.expect(TokenType.IDENT).value
            self.expect(TokenType.LPAREN)
            params = []
            if self.peek().type != TokenType.RPAREN:
                params.append(self.expect(TokenType.IDENT).value)
                while self.match(TokenType.COMMA):
                    params.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.RPAREN)
            self.expect(TokenType.SEMICOLON)
            operations.append((op_name, params))
        self.expect(TokenType.RBRACE)
        return EffectDecl(name=name, operations=operations, line=tok.line)

    def fn_decl(self):
        self.advance()  # fn
        name_tok = self.expect(TokenType.IDENT)
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)
        body = self.block()
        return FnDecl(name=name_tok.value, params=params, body=body, line=name_tok.line)

    def let_decl(self):
        tok = self.advance()  # let
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return LetDecl(name=name, value=value, line=tok.line)

    def statement(self):
        if self.peek().type == TokenType.IF:
            return self.if_stmt()
        if self.peek().type == TokenType.WHILE:
            return self.while_stmt()
        if self.peek().type == TokenType.RETURN:
            return self.return_stmt()
        if self.peek().type == TokenType.PRINT:
            return self.print_stmt()
        if self.peek().type == TokenType.LBRACE:
            return self.block()
        if self.peek().type == TokenType.TRY:
            return self.try_catch()
        return self.expr_stmt()

    def if_stmt(self):
        tok = self.advance()  # if
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        self.expect(TokenType.RPAREN)
        then_body = self.block()
        else_body = None
        if self.match(TokenType.ELSE):
            if self.peek().type == TokenType.IF:
                else_body = self.if_stmt()
            else:
                else_body = self.block()
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body, line=tok.line)

    def while_stmt(self):
        tok = self.advance()  # while
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return WhileStmt(cond=cond, body=body, line=tok.line)

    def return_stmt(self):
        tok = self.advance()  # return
        value = None
        if self.peek().type != TokenType.SEMICOLON:
            value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(value=value, line=tok.line)

    def print_stmt(self):
        tok = self.advance()  # print
        self.expect(TokenType.LPAREN)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(value=value, line=tok.line)

    def try_catch(self):
        tok = self.advance()  # try
        body = self.block()
        self.expect(TokenType.CATCH)
        self.expect(TokenType.LPAREN)
        catch_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.RPAREN)
        catch_body = self.block()
        return TryCatch(body=body, catch_name=catch_name, catch_body=catch_body, line=tok.line)

    def block(self):
        tok = self.expect(TokenType.LBRACE)
        stmts = []
        while self.peek().type != TokenType.RBRACE:
            stmts.append(self.declaration())
        self.expect(TokenType.RBRACE)
        return Block(stmts=stmts, line=tok.line)

    def expr_stmt(self):
        expr = self.expression()
        self.expect(TokenType.SEMICOLON)
        return expr

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.or_expr()
        if self.match(TokenType.ASSIGN):
            if isinstance(expr, Var):
                value = self.assignment()
                return Assign(name=expr.name, value=value, line=expr.line)
            raise ParseError(f"Invalid assignment target at line {self.peek().line}")
        return expr

    def or_expr(self):
        left = self.and_expr()
        while self.match(TokenType.OR):
            right = self.and_expr()
            left = BinOp(op='or', left=left, right=right, line=left.line)
        return left

    def and_expr(self):
        left = self.equality()
        while self.match(TokenType.AND):
            right = self.equality()
            left = BinOp(op='and', left=left, right=right, line=left.line)
        return left

    def equality(self):
        left = self.comparison()
        while True:
            if self.match(TokenType.EQ):
                left = BinOp(op='==', left=left, right=self.comparison(), line=left.line)
            elif self.match(TokenType.NE):
                left = BinOp(op='!=', left=left, right=self.comparison(), line=left.line)
            else:
                break
        return left

    def comparison(self):
        left = self.addition()
        while True:
            if self.match(TokenType.LT):
                left = BinOp(op='<', left=left, right=self.addition(), line=left.line)
            elif self.match(TokenType.GT):
                left = BinOp(op='>', left=left, right=self.addition(), line=left.line)
            elif self.match(TokenType.LE):
                left = BinOp(op='<=', left=left, right=self.addition(), line=left.line)
            elif self.match(TokenType.GE):
                left = BinOp(op='>=', left=left, right=self.addition(), line=left.line)
            else:
                break
        return left

    def addition(self):
        left = self.multiplication()
        while True:
            if self.match(TokenType.PLUS):
                left = BinOp(op='+', left=left, right=self.multiplication(), line=left.line)
            elif self.match(TokenType.MINUS):
                left = BinOp(op='-', left=left, right=self.multiplication(), line=left.line)
            else:
                break
        return left

    def multiplication(self):
        left = self.unary()
        while True:
            if self.match(TokenType.STAR):
                left = BinOp(op='*', left=left, right=self.unary(), line=left.line)
            elif self.match(TokenType.SLASH):
                left = BinOp(op='/', left=left, right=self.unary(), line=left.line)
            elif self.match(TokenType.PERCENT):
                left = BinOp(op='%', left=left, right=self.unary(), line=left.line)
            else:
                break
        return left

    def unary(self):
        if self.match(TokenType.MINUS):
            return UnaryOp(op='-', operand=self.unary(), line=self.tokens[self.pos - 1].line)
        if self.match(TokenType.NOT):
            return UnaryOp(op='not', operand=self.unary(), line=self.tokens[self.pos - 1].line)
        return self.primary()

    def primary(self):
        tok = self.peek()

        if tok.type == TokenType.INT:
            self.advance()
            return IntLit(value=tok.value, line=tok.line)
        if tok.type == TokenType.FLOAT:
            self.advance()
            return FloatLit(value=tok.value, line=tok.line)
        if tok.type == TokenType.STRING:
            self.advance()
            return StringLit(value=tok.value, line=tok.line)
        if tok.type == TokenType.TRUE:
            self.advance()
            return BoolLit(value=True, line=tok.line)
        if tok.type == TokenType.FALSE:
            self.advance()
            return BoolLit(value=False, line=tok.line)

        if tok.type == TokenType.PERFORM:
            return self.perform_expr()

        if tok.type == TokenType.HANDLE:
            return self.handle_expr()

        if tok.type == TokenType.RESUME:
            return self.resume_expr()

        if tok.type == TokenType.THROW:
            return self.throw_expr()

        if tok.type == TokenType.IDENT:
            self.advance()
            if self.peek().type == TokenType.LPAREN:
                self.advance()  # (
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.expression())
                self.expect(TokenType.RPAREN)
                return CallExpr(callee=tok.value, args=args, line=tok.line)
            return Var(name=tok.value, line=tok.line)

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN)
            return expr

        raise ParseError(f"Unexpected token {tok.type.name} '{tok.value}' at line {tok.line}")

    def perform_expr(self):
        tok = self.advance()  # perform
        effect_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.DOT)
        # Operation name might be a keyword like 'print' -- accept IDENT or keywords
        op_tok = self.advance()
        op_name = op_tok.value
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.expression())
            while self.match(TokenType.COMMA):
                args.append(self.expression())
        self.expect(TokenType.RPAREN)
        return PerformExpr(effect=effect_name, operation=op_name, args=args, line=tok.line)

    def handle_expr(self):
        tok = self.advance()  # handle
        body = self.block()
        self.expect(TokenType.WITH)
        self.expect(TokenType.LBRACE)
        handlers = []
        return_clause = None
        while self.peek().type != TokenType.RBRACE:
            # Check for return clause: return(x) -> body
            if self.peek().type == TokenType.RETURN:
                self.advance()  # return
                self.expect(TokenType.LPAREN)
                param = self.expect(TokenType.IDENT).value
                self.expect(TokenType.RPAREN)
                self.expect(TokenType.ARROW)
                rc_body = self.handler_body()
                return_clause = ReturnClause(param=param, body=rc_body)
                # Optional semicolon between clauses
                self.match(TokenType.SEMICOLON)
                continue
            # Effect.op(params, k) -> body
            effect_name = self.expect(TokenType.IDENT).value
            self.expect(TokenType.DOT)
            op_tok = self.advance()
            op_name = op_tok.value
            self.expect(TokenType.LPAREN)
            params = []
            if self.peek().type != TokenType.RPAREN:
                params.append(self.expect(TokenType.IDENT).value)
                while self.match(TokenType.COMMA):
                    params.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.RPAREN)
            self.expect(TokenType.ARROW)
            h_body = self.handler_body()
            # Last param is the continuation name
            if len(params) < 1:
                raise ParseError(f"Handler must have at least a continuation parameter at line {tok.line}")
            cont_name = params[-1]
            op_params = params[:-1]
            handlers.append(HandlerClause(
                effect=effect_name, operation=op_name,
                params=op_params, cont_name=cont_name,
                body=h_body, line=tok.line
            ))
            self.match(TokenType.SEMICOLON)
        self.expect(TokenType.RBRACE)
        return HandleWith(body=body, handlers=handlers, return_clause=return_clause, line=tok.line)

    def handler_body(self):
        """Parse handler body -- either a block or a single expression."""
        if self.peek().type == TokenType.LBRACE:
            return self.block()
        # Single expression -- parse until semicolon or closing brace
        return self.expression()

    def resume_expr(self):
        tok = self.advance()  # resume
        self.expect(TokenType.LPAREN)
        cont = self.expression()
        self.expect(TokenType.COMMA)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        return ResumeExpr(cont=cont, value=value, line=tok.line)

    def throw_expr(self):
        tok = self.advance()  # throw
        self.expect(TokenType.LPAREN)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        return ThrowExpr(value=value, line=tok.line)


# ============================================================
# Runtime Data Structures
# ============================================================

@dataclass
class FnObject:
    """Compiled function."""
    name: str
    arity: int
    chunk: Chunk


@dataclass
class HandlerDescriptor:
    """Describes a handler installed at runtime."""
    clauses: list      # list of (effect, operation, param_names, cont_name, handler_chunk)
    return_clause: Any  # None or (param_name, return_chunk)


@dataclass
class HandlerFrame:
    """A handler frame on the handler stack."""
    descriptor: HandlerDescriptor
    stack_depth: int     # stack depth when handler was installed
    call_depth: int      # call stack depth when handler was installed
    env_snapshot: dict   # env at handler installation time
    chunk: Any           # chunk at handler installation time
    ip: int              # ip at handler installation time (after INSTALL_HANDLER)
    end_ip: int = 0      # ip after REMOVE_HANDLER (where to skip to when handler doesn't resume)
    is_resumed: bool = False  # True if this handler was re-installed via resume
    capture_env: dict = None  # env at the time perform captured the continuation


@dataclass
class Continuation:
    """A captured delimited continuation."""
    stack_slice: list       # stack values between handler and perform
    call_frames: list       # call frames between handler and perform
    handler_frames: list    # handler frames between this handler and perform site
    chunk: Any              # chunk to return to (perform site)
    ip: int                 # ip to return to (after PERFORM)
    handler_frame: Any = None  # the handler frame that caught this (for re-installation on resume)


class EffectError(Exception):
    """Raised when an effect is performed with no handler."""
    pass


class ThrowError(Exception):
    """Raised by throw() to unwind to try/catch."""
    def __init__(self, value):
        self.value = value
        super().__init__(str(value))


# ============================================================
# Compiler
# ============================================================

class CompileError(Exception):
    pass


class Compiler:
    def __init__(self):
        self.chunk = Chunk()
        self.functions = {}
        self.effects = {}  # effect_name -> list of (op_name, param_names)

    def compile(self, program: Program) -> Chunk:
        for stmt in program.stmts:
            self.compile_node(stmt)
        self.chunk.emit(Op.HALT)
        return self.chunk

    def compile_node(self, node):
        method = f'compile_{type(node).__name__}'
        handler = getattr(self, method, None)
        if handler is None:
            raise CompileError(f"Cannot compile {type(node).__name__}")
        handler(node)

    def compile_IntLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_FloatLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_StringLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_BoolLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_Var(self, node):
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.LOAD, idx, node.line)

    def compile_UnaryOp(self, node):
        self.compile_node(node.operand)
        if node.op == '-':
            self.chunk.emit(Op.NEG, line=node.line)
        elif node.op == 'not':
            self.chunk.emit(Op.NOT, line=node.line)

    def compile_BinOp(self, node):
        if node.op == 'and':
            self.compile_node(node.left)
            jump = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
            self.chunk.emit(Op.POP, line=node.line)
            self.compile_node(node.right)
            self.chunk.patch(jump + 1, len(self.chunk.code))
            return
        if node.op == 'or':
            self.compile_node(node.left)
            jump = self.chunk.emit(Op.JUMP_IF_TRUE, 0, node.line)
            self.chunk.emit(Op.POP, line=node.line)
            self.compile_node(node.right)
            self.chunk.patch(jump + 1, len(self.chunk.code))
            return
        self.compile_node(node.left)
        self.compile_node(node.right)
        op_map = {
            '+': Op.ADD, '-': Op.SUB, '*': Op.MUL, '/': Op.DIV, '%': Op.MOD,
            '==': Op.EQ, '!=': Op.NE, '<': Op.LT, '>': Op.GT,
            '<=': Op.LE, '>=': Op.GE,
        }
        self.chunk.emit(op_map[node.op], line=node.line)

    def compile_Assign(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, idx, node.line)

    def compile_LetDecl(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, idx, node.line)

    def compile_Block(self, node):
        for stmt in node.stmts:
            self.compile_node(stmt)

    def compile_IfStmt(self, node):
        self.compile_node(node.cond)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.then_body)
        if node.else_body:
            jump_end = self.chunk.emit(Op.JUMP, 0, node.line)
            self.chunk.patch(jump_false + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)
            self.compile_node(node.else_body)
            self.chunk.patch(jump_end + 1, len(self.chunk.code))
        else:
            self.chunk.patch(jump_false + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)

    def compile_WhileStmt(self, node):
        loop_start = len(self.chunk.code)
        self.compile_node(node.cond)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.body)
        self.chunk.emit(Op.JUMP, loop_start, node.line)
        self.chunk.patch(jump_false + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=node.line)

    def compile_FnDecl(self, node):
        fn_compiler = Compiler()
        fn_compiler.effects = self.effects
        for param in node.params:
            fn_compiler.chunk.add_name(param)
        fn_compiler.compile_node(node.body)
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx)
        fn_compiler.chunk.emit(Op.RETURN)
        fn_obj = FnObject(name=node.name, arity=len(node.params), chunk=fn_compiler.chunk)
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v
        self.functions[node.name] = fn_obj
        fn_idx = self.chunk.add_constant(fn_obj)
        name_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.STORE, name_idx, node.line)

    def compile_CallExpr(self, node):
        fn_idx = self.chunk.add_name(node.callee)
        self.chunk.emit(Op.LOAD, fn_idx, node.line)
        for arg in node.args:
            self.compile_node(arg)
        self.chunk.emit(Op.CALL, len(node.args), node.line)

    def compile_ReturnStmt(self, node):
        if node.value:
            self.compile_node(node.value)
        else:
            idx = self.chunk.add_constant(None)
            self.chunk.emit(Op.CONST, idx, node.line)
        self.chunk.emit(Op.RETURN, line=node.line)

    def compile_PrintStmt(self, node):
        self.compile_node(node.value)
        self.chunk.emit(Op.PRINT, line=node.line)

    def compile_EffectDecl(self, node):
        """Register effect at compile time (no bytecode emitted)."""
        self.effects[node.name] = node.operations

    def compile_PerformExpr(self, node):
        """Compile perform Effect.op(args)."""
        # Push args onto stack
        for arg in node.args:
            self.compile_node(arg)
        # Store effect key as constant
        effect_key = (node.effect, node.operation, len(node.args))
        key_idx = self.chunk.add_constant(effect_key)
        self.chunk.emit(Op.PERFORM, key_idx, node.line)

    def compile_HandleWith(self, node):
        """Compile handle { body } with { handlers }."""
        # Compile each handler clause into its own chunk
        clauses = []
        for hc in node.handlers:
            h_compiler = Compiler()
            h_compiler.effects = self.effects
            # Parameters: op params + continuation
            for p in hc.params:
                h_compiler.chunk.add_name(p)
            h_compiler.chunk.add_name(hc.cont_name)
            h_compiler.compile_node(hc.body)
            # Ensure handler returns a value
            none_idx = h_compiler.chunk.add_constant(None)
            h_compiler.chunk.emit(Op.CONST, none_idx)
            h_compiler.chunk.emit(Op.RETURN)
            for k, v in h_compiler.functions.items():
                self.functions[k] = v
            clauses.append((
                hc.effect, hc.operation,
                hc.params, hc.cont_name,
                h_compiler.chunk
            ))

        # Compile return clause if present
        ret_clause = None
        if node.return_clause:
            rc = node.return_clause
            rc_compiler = Compiler()
            rc_compiler.effects = self.effects
            rc_compiler.chunk.add_name(rc.param)
            rc_compiler.compile_node(rc.body)
            none_idx = rc_compiler.chunk.add_constant(None)
            rc_compiler.chunk.emit(Op.CONST, none_idx)
            rc_compiler.chunk.emit(Op.RETURN)
            for k, v in rc_compiler.functions.items():
                self.functions[k] = v
            ret_clause = (rc.param, rc_compiler.chunk)

        descriptor = HandlerDescriptor(clauses=clauses, return_clause=ret_clause)
        desc_idx = self.chunk.add_constant(descriptor)
        # INSTALL_HANDLER takes desc_idx, then end_addr (patched later)
        install_addr = self.chunk.emit(Op.INSTALL_HANDLER, desc_idx, node.line)
        # Reserve slot for end_addr
        end_addr_slot = len(self.chunk.code)
        self.chunk.code.append(0)
        self.chunk.lines.append(node.line)

        # Compile the body
        self.compile_node(node.body)

        # Remove handler and handle return value
        self.chunk.emit(Op.REMOVE_HANDLER, line=node.line)

        # Patch end_addr to point here (after REMOVE_HANDLER)
        self.chunk.patch(end_addr_slot, len(self.chunk.code))

    def compile_ResumeExpr(self, node):
        """Compile resume(k, value)."""
        self.compile_node(node.cont)
        self.compile_node(node.value)
        self.chunk.emit(Op.RESUME, line=node.line)

    def compile_ThrowExpr(self, node):
        """Compile throw(value) as perform __error__.throw(value)."""
        self.compile_node(node.value)
        effect_key = ("__error__", "throw", 1)
        key_idx = self.chunk.add_constant(effect_key)
        self.chunk.emit(Op.PERFORM, key_idx, node.line)

    def compile_TryCatch(self, node):
        """Try/catch compiles to a handler for the Error effect internally."""
        # We handle try/catch at VM level using Python exceptions
        # Store as a special handler descriptor
        catch_compiler = Compiler()
        catch_compiler.effects = self.effects
        catch_compiler.chunk.add_name(node.catch_name)
        catch_compiler.compile_node(node.catch_body)
        none_idx = catch_compiler.chunk.add_constant(None)
        catch_compiler.chunk.emit(Op.CONST, none_idx)
        catch_compiler.chunk.emit(Op.RETURN)
        for k, v in catch_compiler.functions.items():
            self.functions[k] = v

        # Use __try__ as a special handler
        clauses = [("__error__", "throw", [node.catch_name], "__k__", catch_compiler.chunk)]
        descriptor = HandlerDescriptor(
            clauses=clauses,
            return_clause=None
        )
        desc_idx = self.chunk.add_constant(descriptor)
        install_addr = self.chunk.emit(Op.INSTALL_HANDLER, desc_idx, node.line)
        end_addr_slot = len(self.chunk.code)
        self.chunk.code.append(0)
        self.chunk.lines.append(node.line)
        self.compile_node(node.body)
        self.chunk.emit(Op.REMOVE_HANDLER, line=node.line)
        self.chunk.patch(end_addr_slot, len(self.chunk.code))



# ============================================================
# Virtual Machine with Effect Handling
# ============================================================

class VMError(Exception):
    pass


FRAME_NORMAL = 0
FRAME_HANDLER = 1    # From PERFORM: don't restore env on return
FRAME_RESUME = 2     # From RESUME: restore env on return

_NOT_PRESENT = object()  # sentinel for param restoration

@dataclass
class CallFrame:
    chunk: Any
    ip: int
    base_env: dict
    kind: int = FRAME_NORMAL
    fn_param_saves: dict = None  # saved param values for function frames


class EffectVM:
    """Stack-based VM with algebraic effect handling."""

    def __init__(self, chunk: Chunk, trace=False):
        self.chunk = chunk
        self.stack = []
        self.env = {}
        self.call_stack = []
        self.handler_stack = []  # list of HandlerFrame
        self.output = []
        self.trace = trace
        self.ip = 0
        self.current_chunk = chunk
        self.step_count = 0
        self.max_steps = 500000

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if not self.stack:
            raise VMError("Stack underflow")
        return self.stack.pop()

    def peek(self):
        if not self.stack:
            raise VMError("Stack underflow on peek")
        return self.stack[-1]

    def run(self):
        while True:
            self.step_count += 1
            if self.step_count > self.max_steps:
                raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

            if self.ip >= len(self.current_chunk.code):
                break

            op = self.current_chunk.code[self.ip]
            self.ip += 1

            if self.trace:
                self._trace_op(op)

            if op == Op.HALT:
                break

            elif op == Op.CONST:
                idx = self.current_chunk.code[self.ip]
                self.ip += 1
                self.push(self.current_chunk.constants[idx])

            elif op == Op.POP:
                if self.stack:
                    self.pop()

            elif op == Op.DUP:
                self.push(self.peek())

            elif op == Op.ADD:
                b, a = self.pop(), self.pop()
                self.push(a + b)
            elif op == Op.SUB:
                b, a = self.pop(), self.pop()
                self.push(a - b)
            elif op == Op.MUL:
                b, a = self.pop(), self.pop()
                self.push(a * b)
            elif op == Op.DIV:
                b, a = self.pop(), self.pop()
                if b == 0:
                    raise VMError("Division by zero")
                if isinstance(a, int) and isinstance(b, int):
                    self.push(a // b)
                else:
                    self.push(a / b)
            elif op == Op.MOD:
                b, a = self.pop(), self.pop()
                if b == 0:
                    raise VMError("Modulo by zero")
                self.push(a % b)
            elif op == Op.NEG:
                self.push(-self.pop())

            elif op == Op.EQ:
                b, a = self.pop(), self.pop()
                self.push(a == b)
            elif op == Op.NE:
                b, a = self.pop(), self.pop()
                self.push(a != b)
            elif op == Op.LT:
                b, a = self.pop(), self.pop()
                self.push(a < b)
            elif op == Op.GT:
                b, a = self.pop(), self.pop()
                self.push(a > b)
            elif op == Op.LE:
                b, a = self.pop(), self.pop()
                self.push(a <= b)
            elif op == Op.GE:
                b, a = self.pop(), self.pop()
                self.push(a >= b)

            elif op == Op.NOT:
                self.push(not self.pop())
            elif op == Op.AND:
                b, a = self.pop(), self.pop()
                self.push(a and b)
            elif op == Op.OR:
                b, a = self.pop(), self.pop()
                self.push(a or b)

            elif op == Op.LOAD:
                idx = self.current_chunk.code[self.ip]
                self.ip += 1
                name = self.current_chunk.names[idx]
                if name not in self.env:
                    raise VMError(f"Undefined variable '{name}'")
                self.push(self.env[name])

            elif op == Op.STORE:
                idx = self.current_chunk.code[self.ip]
                self.ip += 1
                name = self.current_chunk.names[idx]
                value = self.pop()
                self.env[name] = value

            elif op == Op.JUMP:
                target = self.current_chunk.code[self.ip]
                self.ip = target

            elif op == Op.JUMP_IF_FALSE:
                target = self.current_chunk.code[self.ip]
                self.ip += 1
                if not self.peek():
                    self.ip = target

            elif op == Op.JUMP_IF_TRUE:
                target = self.current_chunk.code[self.ip]
                self.ip += 1
                if self.peek():
                    self.ip = target

            elif op == Op.CALL:
                arg_count = self.current_chunk.code[self.ip]
                self.ip += 1
                args = []
                for _ in range(arg_count):
                    args.insert(0, self.pop())
                fn_obj = self.pop()
                if not isinstance(fn_obj, FnObject):
                    raise VMError(f"Cannot call non-function: {fn_obj}")
                if fn_obj.arity != arg_count:
                    raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")
                # Save param values so we can restore them on return
                fn_param_saves = {}
                for param_name in fn_obj.chunk.names[:fn_obj.arity]:
                    fn_param_saves[param_name] = self.env.get(param_name, _NOT_PRESENT)
                # Also save all function-local names (added during compilation)
                fn_local_names = set(fn_obj.chunk.names) - set(fn_obj.chunk.names[:fn_obj.arity])
                for local_name in fn_local_names:
                    fn_param_saves[local_name] = self.env.get(local_name, _NOT_PRESENT)
                frame = CallFrame(
                    chunk=self.current_chunk, ip=self.ip,
                    base_env=dict(self.env),
                    fn_param_saves=fn_param_saves,
                )
                self.call_stack.append(frame)
                self.current_chunk = fn_obj.chunk
                self.ip = 0
                for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                    self.env[param_name] = args[i]

            elif op == Op.RETURN:
                return_val = self.pop()
                if not self.call_stack:
                    self.push(return_val)
                    break
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                if frame.kind == FRAME_HANDLER:
                    # Handler dispatch frame: keep current env (don't restore)
                    pass
                elif frame.kind == FRAME_RESUME:
                    # Resume frame: keep current env (shared mutable)
                    pass
                else:
                    # Normal function frame: restore function params/locals
                    # to their pre-call values, but keep other env changes
                    # (which may have been made by effect handlers).
                    if frame.fn_param_saves is not None:
                        for name, prev_val in frame.fn_param_saves.items():
                            if prev_val is _NOT_PRESENT:
                                self.env.pop(name, None)
                            else:
                                self.env[name] = prev_val
                    else:
                        # Fallback: full restore (no fn_param_saves)
                        self.env = frame.base_env
                self.push(return_val)

            elif op == Op.PRINT:
                value = self.pop()
                text = str(value) if value is not None else "None"
                if isinstance(value, bool):
                    text = "true" if value else "false"
                self.output.append(text)

            # === Effect operations ===

            elif op == Op.INSTALL_HANDLER:
                desc_idx = self.current_chunk.code[self.ip]
                self.ip += 1
                end_ip = self.current_chunk.code[self.ip]
                self.ip += 1
                descriptor = self.current_chunk.constants[desc_idx]
                frame = HandlerFrame(
                    descriptor=descriptor,
                    stack_depth=len(self.stack),
                    call_depth=len(self.call_stack),
                    env_snapshot=dict(self.env),
                    chunk=self.current_chunk,
                    ip=self.ip,
                    end_ip=end_ip,
                )
                self.handler_stack.append(frame)

            elif op == Op.REMOVE_HANDLER:
                if self.handler_stack:
                    hf = self.handler_stack.pop()
                    if hf.is_resumed:
                        # Resumed continuation finished. Return to resume call site.
                        # Don't touch env -- shared mutable env is already correct.
                        return_val = None
                        while self.call_stack:
                            frame = self.call_stack.pop()
                            if frame.kind == FRAME_RESUME:
                                self.current_chunk = frame.chunk
                                self.ip = frame.ip
                                # Don't restore env -- keep shared mutable env
                                self.push(return_val)
                                break
                        else:
                            self.push(return_val)
                            break
                    else:
                        # Normal (non-resumed) handler removal
                        if hf.descriptor.return_clause:
                            return_val = self.pop() if self.stack else None
                            param_name, ret_chunk = hf.descriptor.return_clause
                            frame = CallFrame(
                                chunk=self.current_chunk,
                                ip=self.ip,
                                base_env=None,
                                kind=FRAME_HANDLER,
                            )
                            self.call_stack.append(frame)
                            self.current_chunk = ret_chunk
                            self.ip = 0
                            self.env[param_name] = return_val

            elif op == Op.PERFORM:
                key_idx = self.current_chunk.code[self.ip]
                self.ip += 1
                effect_key = self.current_chunk.constants[key_idx]
                effect_name, op_name, arg_count = effect_key

                # Pop args
                args = []
                for _ in range(arg_count):
                    args.insert(0, self.pop())

                # Search handler stack from top
                handler_idx = None
                matching_clause = None
                for i in range(len(self.handler_stack) - 1, -1, -1):
                    hf = self.handler_stack[i]
                    for clause in hf.descriptor.clauses:
                        c_effect, c_op, c_params, c_cont, c_chunk = clause
                        if c_effect == effect_name and c_op == op_name:
                            handler_idx = i
                            matching_clause = clause
                            break
                    if handler_idx is not None:
                        break

                if handler_idx is None:
                    raise EffectError(f"Unhandled effect: {effect_name}.{op_name}")

                hf = self.handler_stack[handler_idx]
                c_effect, c_op, c_params, c_cont, c_chunk = matching_clause

                # Capture continuation: stack/frames between handler and perform
                stack_slice = list(self.stack[hf.stack_depth:])
                self.stack = self.stack[:hf.stack_depth]

                call_frames_slice = list(self.call_stack[hf.call_depth:])
                self.call_stack = self.call_stack[:hf.call_depth]

                handler_frames_slice = list(self.handler_stack[handler_idx + 1:])
                self.handler_stack = self.handler_stack[:handler_idx]

                # Create continuation (no env snapshot -- shared mutable env)
                cont = Continuation(
                    stack_slice=stack_slice,
                    call_frames=call_frames_slice,
                    handler_frames=handler_frames_slice,
                    chunk=self.current_chunk,
                    ip=self.ip,
                    handler_frame=hf,
                )

                # Don't restore env -- shared mutable env. Just bind handler params.
                for pi, pname in enumerate(c_params):
                    if pi < len(args):
                        self.env[pname] = args[pi]
                self.env[c_cont] = cont

                # Execute handler chunk. FRAME_HANDLER so RETURN doesn't restore env.
                frame = CallFrame(
                    chunk=hf.chunk,
                    ip=hf.end_ip,
                    base_env=None,
                    kind=FRAME_HANDLER,
                )
                self.call_stack.append(frame)
                self.current_chunk = c_chunk
                self.ip = 0

            elif op == Op.RESUME:
                value = self.pop()
                cont = self.pop()
                if not isinstance(cont, Continuation):
                    raise VMError(f"Cannot resume non-continuation: {cont}")

                # Save return point. Don't snapshot env -- shared mutable.
                frame = CallFrame(
                    chunk=self.current_chunk,
                    ip=self.ip,
                    base_env=None,
                    kind=FRAME_RESUME,
                )
                self.call_stack.append(frame)

                # Restore continuation's stack/frames
                self.stack.extend(cont.stack_slice)
                self.call_stack.extend(cont.call_frames)
                self.handler_stack.extend(cont.handler_frames)

                # Re-install the handler (marked as resumed)
                reinstalled = HandlerFrame(
                    descriptor=cont.handler_frame.descriptor,
                    stack_depth=len(self.stack),
                    call_depth=len(self.call_stack),
                    env_snapshot=None,
                    chunk=cont.handler_frame.chunk,
                    ip=cont.handler_frame.ip,
                    end_ip=cont.handler_frame.end_ip,
                    is_resumed=True,
                )
                self.handler_stack.append(reinstalled)

                # Don't restore env -- shared mutable. Just continue from perform site.
                self.current_chunk = cont.chunk
                self.ip = cont.ip

                # Push the resumed value (result of perform expression)
                self.push(value)

            elif op == Op.MAKE_CONTINUATION:
                pass  # unused

            else:
                raise VMError(f"Unknown opcode: {op}")

        return self.stack[-1] if self.stack else None

    def _trace_op(self, op):
        try:
            name = Op(op).name
        except ValueError:
            name = f"??({op})"
        print(f"  [{self.ip-1:04d}] {name:20s} stack={self.stack[-5:]} handlers={len(self.handler_stack)}")


# ============================================================
# Public API
# ============================================================

def compile_source(source: str) -> tuple:
    """Compile source to bytecode. Returns (chunk, compiler)."""
    tokens = lex(source)
    parser = Parser(tokens)
    ast = parser.parse()
    compiler = Compiler()
    chunk = compiler.compile(ast)
    return chunk, compiler


def execute(source: str, trace=False) -> dict:
    """Compile and execute source code. Returns result dict."""
    chunk, compiler = compile_source(source)
    vm = EffectVM(chunk, trace=trace)
    result = vm.run()
    return {
        'result': result,
        'output': vm.output,
        'env': vm.env,
        'steps': vm.step_count,
    }


def disassemble(chunk: Chunk) -> str:
    """Disassemble a chunk to human-readable form."""
    lines = []
    i = 0
    while i < len(chunk.code):
        op_val = chunk.code[i]
        try:
            name = Op(op_val).name
        except ValueError:
            name = f"??({op_val})"

        if op_val == Op.INSTALL_HANDLER and i + 2 < len(chunk.code):
            desc = chunk.code[i + 1]
            end = chunk.code[i + 2]
            lines.append(f"{i:04d}  {name:20s} desc={desc} end={end}")
            i += 3
        elif op_val in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
                        Op.JUMP_IF_TRUE, Op.CALL, Op.PERFORM) and i + 1 < len(chunk.code):
            operand = chunk.code[i + 1]
            if op_val == Op.CONST:
                val = chunk.constants[operand]
                if isinstance(val, FnObject):
                    lines.append(f"{i:04d}  {name:20s} {operand} (fn:{val.name})")
                elif isinstance(val, HandlerDescriptor):
                    lines.append(f"{i:04d}  {name:20s} {operand} (handler)")
                else:
                    lines.append(f"{i:04d}  {name:20s} {operand} ({val!r})")
            elif op_val in (Op.LOAD, Op.STORE):
                nm = chunk.names[operand] if operand < len(chunk.names) else "?"
                lines.append(f"{i:04d}  {name:20s} {operand} ({nm})")
            else:
                lines.append(f"{i:04d}  {name:20s} {operand}")
            i += 2
        else:
            lines.append(f"{i:04d}  {name}")
            i += 1
    return '\n'.join(lines)
