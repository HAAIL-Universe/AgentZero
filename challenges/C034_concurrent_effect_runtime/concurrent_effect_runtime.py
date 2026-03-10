"""
C034: Concurrent Effect Runtime
Composes C033 (Effect Runtime) + C029 (Concurrent Runtime)

Merges algebraic effects with cooperative concurrency:
- Task-local effect handler stacks
- Handler inheritance on spawn (deep copy)
- Built-in Async effect: spawn, yield, send, recv as algebraic effects
- Effect-aware scheduling: perform/resume within task time slices
- Continuation capture/restore per-task
- Nested handlers within concurrent tasks
- Select with active effect handlers
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
from enum import IntEnum
import copy

# ============================================================
# Tokens & Lexer
# ============================================================

class TokenType(IntEnum):
    # Literals & identifiers
    NUMBER = 1
    STRING = 2
    IDENT = 3
    BOOL = 4
    # Operators
    PLUS = 10; MINUS = 11; STAR = 12; SLASH = 13
    EQ = 14; NEQ = 15; LT = 16; GT = 17; LTE = 18; GTE = 19
    ASSIGN = 20; BANG = 21; AND = 22; OR = 23; MOD = 24
    # Delimiters
    LPAREN = 30; RPAREN = 31; LBRACE = 32; RBRACE = 33
    COMMA = 34; SEMICOLON = 35; DOT = 36; ARROW = 37
    # Keywords
    LET = 50; FN = 51; IF = 52; ELSE = 53; WHILE = 54
    RETURN = 55; PRINT = 56; TRUE = 57; FALSE = 58
    # Concurrency keywords
    SPAWN = 60; YIELD = 61; CHAN = 62; SEND = 63; RECV = 64
    JOIN = 65; TASK_ID = 66; SELECT = 67; CASE = 68; DEFAULT = 69
    # Effect keywords
    EFFECT = 70; PERFORM = 71; HANDLE = 72; WITH = 73; RESUME = 74
    # EOF
    EOF = 99

@dataclass
class Token:
    type: TokenType
    value: Any
    line: int

KEYWORDS = {
    'let': TokenType.LET, 'fn': TokenType.FN, 'if': TokenType.IF,
    'else': TokenType.ELSE, 'while': TokenType.WHILE, 'return': TokenType.RETURN,
    'print': TokenType.PRINT, 'true': TokenType.TRUE, 'false': TokenType.FALSE,
    'spawn': TokenType.SPAWN, 'yield': TokenType.YIELD, 'chan': TokenType.CHAN,
    'send': TokenType.SEND, 'recv': TokenType.RECV, 'join': TokenType.JOIN,
    'task_id': TokenType.TASK_ID, 'select': TokenType.SELECT,
    'case': TokenType.CASE, 'default': TokenType.DEFAULT,
    'effect': TokenType.EFFECT, 'perform': TokenType.PERFORM,
    'handle': TokenType.HANDLE, 'with': TokenType.WITH, 'resume': TokenType.RESUME,
}

def lex(source):
    tokens = []
    i = 0
    line = 1
    while i < len(source):
        c = source[i]
        if c == '\n':
            line += 1; i += 1
        elif c in ' \t\r':
            i += 1
        elif c == '/' and i + 1 < len(source) and source[i+1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
        elif c.isdigit():
            j = i
            while j < len(source) and (source[j].isdigit() or source[j] == '.'):
                j += 1
            val = source[i:j]
            tokens.append(Token(TokenType.NUMBER, float(val) if '.' in val else int(val), line))
            i = j
        elif c == '"':
            j = i + 1
            s = ''
            while j < len(source) and source[j] != '"':
                if source[j] == '\\' and j + 1 < len(source):
                    nc = source[j+1]
                    if nc == 'n': s += '\n'
                    elif nc == 't': s += '\t'
                    elif nc == '"': s += '"'
                    elif nc == '\\': s += '\\'
                    else: s += nc
                    j += 2
                else:
                    s += source[j]; j += 1
            tokens.append(Token(TokenType.STRING, s, line))
            i = j + 1
        elif c.isalpha() or c == '_':
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            if word in KEYWORDS:
                tt = KEYWORDS[word]
                if tt == TokenType.TRUE:
                    tokens.append(Token(TokenType.BOOL, True, line))
                elif tt == TokenType.FALSE:
                    tokens.append(Token(TokenType.BOOL, False, line))
                else:
                    tokens.append(Token(tt, word, line))
            else:
                tokens.append(Token(TokenType.IDENT, word, line))
            i = j
        elif c == '=' and i + 1 < len(source) and source[i+1] == '=':
            tokens.append(Token(TokenType.EQ, '==', line)); i += 2
        elif c == '!' and i + 1 < len(source) and source[i+1] == '=':
            tokens.append(Token(TokenType.NEQ, '!=', line)); i += 2
        elif c == '<' and i + 1 < len(source) and source[i+1] == '=':
            tokens.append(Token(TokenType.LTE, '<=', line)); i += 2
        elif c == '>' and i + 1 < len(source) and source[i+1] == '=':
            tokens.append(Token(TokenType.GTE, '>=', line)); i += 2
        elif c == '-' and i + 1 < len(source) and source[i+1] == '>':
            tokens.append(Token(TokenType.ARROW, '->', line)); i += 2
        elif c == '&' and i + 1 < len(source) and source[i+1] == '&':
            tokens.append(Token(TokenType.AND, '&&', line)); i += 2
        elif c == '|' and i + 1 < len(source) and source[i+1] == '|':
            tokens.append(Token(TokenType.OR, '||', line)); i += 2
        else:
            simple = {'+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
                      '/': TokenType.SLASH, '%': TokenType.MOD, '(': TokenType.LPAREN,
                      ')': TokenType.RPAREN, '{': TokenType.LBRACE, '}': TokenType.RBRACE,
                      ',': TokenType.COMMA, ';': TokenType.SEMICOLON, '.': TokenType.DOT,
                      '=': TokenType.ASSIGN, '!': TokenType.BANG, '<': TokenType.LT,
                      '>': TokenType.GT}
            if c in simple:
                tokens.append(Token(simple[c], c, line)); i += 1
            else:
                i += 1  # skip unknown
    tokens.append(Token(TokenType.EOF, None, line))
    return tokens

# ============================================================
# AST Nodes
# ============================================================

@dataclass
class NumberLit:
    value: float

@dataclass
class StringLit:
    value: str

@dataclass
class BoolLit:
    value: bool

@dataclass
class Var:
    name: str

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    op: str
    operand: Any

@dataclass
class CallExpr:
    callee: Any
    args: list

@dataclass
class LetDecl:
    name: str
    value: Any

@dataclass
class Assignment:
    name: str
    value: Any

@dataclass
class PrintStmt:
    value: Any

@dataclass
class IfStmt:
    cond: Any
    then_body: list
    else_body: list = None

@dataclass
class WhileStmt:
    cond: Any
    body: list

@dataclass
class FnDecl:
    name: str
    params: list
    body: list

@dataclass
class ReturnStmt:
    value: Any = None

@dataclass
class Block:
    stmts: list

# Concurrency AST
@dataclass
class SpawnExpr:
    callee: str
    args: list

@dataclass
class YieldStmt:
    pass

@dataclass
class ChanExpr:
    size: Any = None

@dataclass
class SendExpr:
    channel: Any
    value: Any

@dataclass
class RecvExpr:
    channel: Any

@dataclass
class JoinExpr:
    task: Any

@dataclass
class TaskIdExpr:
    pass

@dataclass
class SelectCase:
    kind: str  # 'send' or 'recv'
    channel: Any
    value: Any = None  # for send
    var_name: str = None  # for recv
    body: list = None

@dataclass
class SelectStmt:
    cases: list
    default_body: list = None

# Effect AST
@dataclass
class EffectOp:
    name: str
    params: list

@dataclass
class EffectDecl:
    name: str
    operations: list  # [EffectOp]

@dataclass
class PerformExpr:
    effect: str
    operation: str
    args: list

@dataclass
class HandlerClause:
    effect: str
    operation: str
    params: list
    body: list

@dataclass
class HandleWith:
    body: list
    handlers: list  # [HandlerClause]

@dataclass
class ResumeExpr:
    value: Any

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
            raise ParseError(f"Expected {tt}, got {t.type} ({t.value!r}) at line {t.line}")
        return t

    def expect_ident_like(self):
        """Accept IDENT or any keyword token with a string value as identifier."""
        t = self.peek()
        if t.type == TokenType.IDENT:
            return self.advance().value
        if isinstance(t.value, str) and t.type not in (TokenType.EOF, TokenType.STRING):
            return self.advance().value
        raise ParseError(f"Expected identifier, got {t.type} at line {t.line}")

    def match(self, tt):
        if self.peek().type == tt:
            return self.advance()
        return None

    def parse(self):
        stmts = []
        while self.peek().type != TokenType.EOF:
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        t = self.peek()
        if t.type == TokenType.LET:
            return self.parse_let()
        if t.type == TokenType.FN:
            return self.parse_fn()
        if t.type == TokenType.IF:
            return self.parse_if()
        if t.type == TokenType.WHILE:
            return self.parse_while()
        if t.type == TokenType.RETURN:
            return self.parse_return()
        if t.type == TokenType.PRINT:
            return self.parse_print()
        if t.type == TokenType.YIELD:
            return self.parse_yield()
        if t.type == TokenType.SELECT:
            return self.parse_select()
        if t.type == TokenType.EFFECT:
            return self.parse_effect_decl()
        if t.type == TokenType.HANDLE:
            return self.parse_handle()
        # Expression statement
        expr = self.parse_expr()
        self.match(TokenType.SEMICOLON)
        return expr

    def parse_let(self):
        self.advance()  # let
        name = self.expect_ident_like()
        self.expect(TokenType.ASSIGN)
        val = self.parse_expr()
        self.match(TokenType.SEMICOLON)
        return LetDecl(name=name, value=val)

    def parse_fn(self):
        self.advance()  # fn
        name = self.expect_ident_like()
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.expect_ident_like())
            while self.match(TokenType.COMMA):
                params.append(self.expect_ident_like())
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return FnDecl(name=name, params=params, body=body)

    def parse_block(self):
        self.expect(TokenType.LBRACE)
        stmts = []
        while self.peek().type != TokenType.RBRACE:
            stmts.append(self.parse_stmt())
        self.expect(TokenType.RBRACE)
        return stmts

    def parse_if(self):
        self.advance()  # if
        self.expect(TokenType.LPAREN)
        cond = self.parse_expr()
        self.expect(TokenType.RPAREN)
        then_body = self.parse_block()
        else_body = None
        if self.match(TokenType.ELSE):
            if self.peek().type == TokenType.IF:
                else_body = [self.parse_if()]
            else:
                else_body = self.parse_block()
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body)

    def parse_while(self):
        self.advance()  # while
        self.expect(TokenType.LPAREN)
        cond = self.parse_expr()
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return WhileStmt(cond=cond, body=body)

    def parse_return(self):
        self.advance()  # return
        val = None
        if self.peek().type not in (TokenType.SEMICOLON, TokenType.RBRACE, TokenType.EOF):
            val = self.parse_expr()
        self.match(TokenType.SEMICOLON)
        return ReturnStmt(value=val)

    def parse_print(self):
        self.advance()  # print
        self.expect(TokenType.LPAREN)
        val = self.parse_expr()
        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)
        return PrintStmt(value=val)

    def parse_yield(self):
        self.advance()  # yield
        self.match(TokenType.SEMICOLON)
        return YieldStmt()

    def parse_select(self):
        self.advance()  # select
        self.expect(TokenType.LBRACE)
        cases = []
        default_body = None
        while self.peek().type != TokenType.RBRACE:
            if self.peek().type == TokenType.CASE:
                self.advance()
                t = self.peek()
                if t.type == TokenType.SEND:
                    self.advance()
                    self.expect(TokenType.LPAREN)
                    ch = self.parse_expr()
                    self.expect(TokenType.COMMA)
                    val = self.parse_expr()
                    self.expect(TokenType.RPAREN)
                    self.expect(TokenType.ARROW)
                    body = self.parse_block()
                    cases.append(SelectCase(kind='send', channel=ch, value=val, body=body))
                elif t.type == TokenType.RECV:
                    self.advance()
                    self.expect(TokenType.LPAREN)
                    ch = self.parse_expr()
                    self.expect(TokenType.RPAREN)
                    self.expect(TokenType.ARROW)
                    var_name = self.expect_ident_like()
                    body = self.parse_block()
                    cases.append(SelectCase(kind='recv', channel=ch, var_name=var_name, body=body))
                else:
                    raise ParseError(f"Expected send or recv in select case at line {t.line}")
            elif self.peek().type == TokenType.DEFAULT:
                self.advance()
                self.expect(TokenType.ARROW)
                default_body = self.parse_block()
            else:
                raise ParseError(f"Expected case or default in select at line {self.peek().line}")
        self.expect(TokenType.RBRACE)
        return SelectStmt(cases=cases, default_body=default_body)

    def parse_effect_decl(self):
        self.advance()  # effect
        name = self.expect_ident_like()
        self.expect(TokenType.LBRACE)
        ops = []
        while self.peek().type != TokenType.RBRACE:
            op_name = self.expect_ident_like()
            self.expect(TokenType.LPAREN)
            params = []
            if self.peek().type != TokenType.RPAREN:
                params.append(self.expect_ident_like())
                while self.match(TokenType.COMMA):
                    params.append(self.expect_ident_like())
            self.expect(TokenType.RPAREN)
            self.match(TokenType.SEMICOLON)
            ops.append(EffectOp(name=op_name, params=params))
        self.expect(TokenType.RBRACE)
        return EffectDecl(name=name, operations=ops)

    def parse_handle(self):
        self.advance()  # handle
        body = self.parse_block()
        self.expect(TokenType.WITH)
        self.expect(TokenType.LBRACE)
        handlers = []
        while self.peek().type != TokenType.RBRACE:
            effect_name = self.expect_ident_like()
            self.expect(TokenType.DOT)
            op_name = self.expect_ident_like()
            self.expect(TokenType.LPAREN)
            params = []
            if self.peek().type != TokenType.RPAREN:
                params.append(self.expect_ident_like())
                while self.match(TokenType.COMMA):
                    params.append(self.expect_ident_like())
            self.expect(TokenType.RPAREN)
            self.expect(TokenType.ARROW)
            handler_body = self.parse_block()
            handlers.append(HandlerClause(
                effect=effect_name, operation=op_name,
                params=params, body=handler_body
            ))
        self.expect(TokenType.RBRACE)
        return HandleWith(body=body, handlers=handlers)

    def parse_expr(self):
        return self.parse_assignment_expr()

    def parse_assignment_expr(self):
        expr = self.parse_or()
        if isinstance(expr, Var) and self.peek().type == TokenType.ASSIGN:
            self.advance()
            val = self.parse_expr()
            return Assignment(name=expr.name, value=val)
        return expr

    def parse_or(self):
        left = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            left = BinOp(op='||', left=left, right=right)
        return left

    def parse_and(self):
        left = self.parse_equality()
        while self.match(TokenType.AND):
            right = self.parse_equality()
            left = BinOp(op='&&', left=left, right=right)
        return left

    def parse_equality(self):
        left = self.parse_comparison()
        while self.peek().type in (TokenType.EQ, TokenType.NEQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinOp(op=op, left=left, right=right)
        return left

    def parse_comparison(self):
        left = self.parse_additive()
        while self.peek().type in (TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            op = self.advance().value
            right = self.parse_additive()
            left = BinOp(op=op, left=left, right=right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinOp(op=op, left=left, right=right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH, TokenType.MOD):
            op = self.advance().value
            right = self.parse_unary()
            left = BinOp(op=op, left=left, right=right)
        return left

    def parse_unary(self):
        if self.peek().type == TokenType.MINUS:
            self.advance()
            return UnaryOp(op='-', operand=self.parse_unary())
        if self.peek().type == TokenType.BANG:
            self.advance()
            return UnaryOp(op='!', operand=self.parse_unary())
        return self.parse_call()

    def parse_call(self):
        expr = self.parse_primary()
        while self.peek().type == TokenType.LPAREN:
            self.advance()
            args = []
            if self.peek().type != TokenType.RPAREN:
                args.append(self.parse_expr())
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expr())
            self.expect(TokenType.RPAREN)
            expr = CallExpr(callee=expr, args=args)
        return expr

    def parse_primary(self):
        t = self.peek()
        if t.type == TokenType.NUMBER:
            self.advance(); return NumberLit(value=t.value)
        if t.type == TokenType.STRING:
            self.advance(); return StringLit(value=t.value)
        if t.type == TokenType.BOOL:
            self.advance(); return BoolLit(value=t.value)
        if t.type == TokenType.IDENT:
            self.advance(); return Var(name=t.value)
        if t.type == TokenType.SPAWN:
            return self.parse_spawn()
        if t.type == TokenType.CHAN:
            return self.parse_chan()
        if t.type == TokenType.SEND:
            return self.parse_send()
        if t.type == TokenType.RECV:
            return self.parse_recv()
        if t.type == TokenType.JOIN:
            return self.parse_join()
        if t.type == TokenType.TASK_ID:
            self.advance()
            self.expect(TokenType.LPAREN)
            self.expect(TokenType.RPAREN)
            return TaskIdExpr()
        if t.type == TokenType.PERFORM:
            return self.parse_perform()
        if t.type == TokenType.RESUME:
            return self.parse_resume()
        if t.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr
        raise ParseError(f"Unexpected token {t.type} ({t.value!r}) at line {t.line}")

    def parse_spawn(self):
        self.advance()  # spawn
        name = self.expect_ident_like()
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN)
        return SpawnExpr(callee=name, args=args)

    def parse_chan(self):
        self.advance()  # chan
        self.expect(TokenType.LPAREN)
        size = None
        if self.peek().type != TokenType.RPAREN:
            size = self.parse_expr()
        self.expect(TokenType.RPAREN)
        return ChanExpr(size=size)

    def parse_send(self):
        self.advance()  # send
        self.expect(TokenType.LPAREN)
        ch = self.parse_expr()
        self.expect(TokenType.COMMA)
        val = self.parse_expr()
        self.expect(TokenType.RPAREN)
        return SendExpr(channel=ch, value=val)

    def parse_recv(self):
        self.advance()  # recv
        self.expect(TokenType.LPAREN)
        ch = self.parse_expr()
        self.expect(TokenType.RPAREN)
        return RecvExpr(channel=ch)

    def parse_join(self):
        self.advance()  # join
        self.expect(TokenType.LPAREN)
        task = self.parse_expr()
        self.expect(TokenType.RPAREN)
        return JoinExpr(task=task)

    def parse_perform(self):
        self.advance()  # perform
        effect = self.expect_ident_like()
        self.expect(TokenType.DOT)
        op = self.expect_ident_like()
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN)
        return PerformExpr(effect=effect, operation=op, args=args)

    def parse_resume(self):
        self.advance()  # resume
        self.expect(TokenType.LPAREN)
        val = self.parse_expr()
        self.expect(TokenType.RPAREN)
        return ResumeExpr(value=val)


# ============================================================
# Bytecode
# ============================================================

class Op(IntEnum):
    CONST = 1
    ADD = 2; SUB = 3; MUL = 4; DIV = 5; MOD = 6
    EQ = 7; NEQ = 8; LT = 9; GT = 10; LTE = 11; GTE = 12
    AND = 13; OR = 14; NOT = 15; NEG = 16
    LOAD = 17; STORE = 18
    JUMP = 19; JUMP_IF_FALSE = 20
    CALL = 21; RETURN = 22
    PRINT = 23; POP = 24; DUP = 25; HALT = 26
    # Concurrency ops
    SPAWN = 100; YIELD_OP = 101; CHAN_NEW = 102
    CHAN_SEND = 103; CHAN_RECV = 104; TASK_JOIN = 105
    CHAN_TRY_SEND = 107; CHAN_TRY_RECV = 108; TASK_ID_OP = 109
    # Effect ops
    PERFORM = 120; INSTALL_HANDLER = 121; REMOVE_HANDLER = 122; RESUME = 123

OP_NAMES = {v: v.name for v in Op}

@dataclass
class Chunk:
    code: list = field(default_factory=list)
    constants: list = field(default_factory=list)
    names: list = field(default_factory=list)

    def add_constant(self, value):
        # Type-aware to avoid True==1 collision
        for i, c in enumerate(self.constants):
            if type(c) is type(value) and c == value:
                return i
        self.constants.append(value)
        return len(self.constants) - 1

    def add_name(self, name):
        if name in self.names:
            return self.names.index(name)
        self.names.append(name)
        return len(self.names) - 1

    def emit(self, op, operand=None):
        addr = len(self.code)
        self.code.append(op)
        if operand is not None:
            self.code.append(operand)
        return addr

    def patch(self, addr, value):
        self.code[addr] = value

@dataclass
class FnObject:
    name: str
    params: list
    chunk: Chunk

@dataclass
class HandlerClauseObj:
    effect: str
    operation: str
    param_names: list
    chunk: Chunk
    arity: int

@dataclass
class HandlerObject:
    clauses: dict  # {(effect, op): HandlerClauseObj}

# ============================================================
# Compiler
# ============================================================

class CompileError(Exception):
    pass

class Compiler:
    def __init__(self):
        self.chunk = Chunk()
        self.functions = {}
        self.effects = {}  # effect_name -> [op_names]

    def compile(self, stmts):
        for s in stmts:
            self.compile_node(s, self.chunk)
        self.chunk.emit(Op.HALT)
        return self.chunk

    def compile_node(self, node, chunk):
        method = f'compile_{type(node).__name__}'
        if hasattr(self, method):
            getattr(self, method)(node, chunk)
        else:
            raise CompileError(f"Cannot compile {type(node).__name__}")

    def compile_NumberLit(self, node, chunk):
        idx = chunk.add_constant(node.value)
        chunk.emit(Op.CONST, idx)

    def compile_StringLit(self, node, chunk):
        idx = chunk.add_constant(node.value)
        chunk.emit(Op.CONST, idx)

    def compile_BoolLit(self, node, chunk):
        idx = chunk.add_constant(node.value)
        chunk.emit(Op.CONST, idx)

    def compile_Var(self, node, chunk):
        idx = chunk.add_name(node.name)
        chunk.emit(Op.LOAD, idx)

    def compile_BinOp(self, node, chunk):
        # Short-circuit for && and ||
        if node.op == '&&':
            self.compile_node(node.left, chunk)
            chunk.emit(Op.DUP)
            jump_addr = chunk.emit(Op.JUMP_IF_FALSE, 0)
            chunk.emit(Op.POP)
            self.compile_node(node.right, chunk)
            chunk.patch(jump_addr + 1, len(chunk.code))
            return
        if node.op == '||':
            self.compile_node(node.left, chunk)
            chunk.emit(Op.DUP)
            # Jump if true (invert: jump_if_false to skip, else take)
            chunk.emit(Op.NOT)
            jump_addr = chunk.emit(Op.JUMP_IF_FALSE, 0)
            chunk.emit(Op.POP)
            self.compile_node(node.right, chunk)
            chunk.patch(jump_addr + 1, len(chunk.code))
            return
        self.compile_node(node.left, chunk)
        self.compile_node(node.right, chunk)
        ops = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL, '/': Op.DIV, '%': Op.MOD,
               '==': Op.EQ, '!=': Op.NEQ, '<': Op.LT, '>': Op.GT, '<=': Op.LTE, '>=': Op.GTE}
        if node.op in ops:
            chunk.emit(ops[node.op])
        else:
            raise CompileError(f"Unknown binop {node.op}")

    def compile_UnaryOp(self, node, chunk):
        self.compile_node(node.operand, chunk)
        if node.op == '-':
            chunk.emit(Op.NEG)
        elif node.op == '!':
            chunk.emit(Op.NOT)

    def compile_CallExpr(self, node, chunk):
        self.compile_node(node.callee, chunk)
        for arg in node.args:
            self.compile_node(arg, chunk)
        idx = chunk.add_constant(len(node.args))
        chunk.emit(Op.CALL, idx)

    def compile_LetDecl(self, node, chunk):
        self.compile_node(node.value, chunk)
        idx = chunk.add_name(node.name)
        chunk.emit(Op.STORE, idx)

    def compile_Assignment(self, node, chunk):
        self.compile_node(node.value, chunk)
        idx = chunk.add_name(node.name)
        chunk.emit(Op.DUP)  # assignment is expression, leave value
        chunk.emit(Op.STORE, idx)

    def compile_PrintStmt(self, node, chunk):
        self.compile_node(node.value, chunk)
        chunk.emit(Op.PRINT)

    def compile_IfStmt(self, node, chunk):
        self.compile_node(node.cond, chunk)
        jump_false = chunk.emit(Op.JUMP_IF_FALSE, 0)
        for s in node.then_body:
            self.compile_node(s, chunk)
        if node.else_body:
            jump_end = chunk.emit(Op.JUMP, 0)
            chunk.patch(jump_false + 1, len(chunk.code))
            for s in node.else_body:
                self.compile_node(s, chunk)
            chunk.patch(jump_end + 1, len(chunk.code))
        else:
            chunk.patch(jump_false + 1, len(chunk.code))

    def compile_WhileStmt(self, node, chunk):
        loop_start = len(chunk.code)
        self.compile_node(node.cond, chunk)
        jump_false = chunk.emit(Op.JUMP_IF_FALSE, 0)
        for s in node.body:
            self.compile_node(s, chunk)
        chunk.emit(Op.JUMP, loop_start)
        chunk.patch(jump_false + 1, len(chunk.code))

    def compile_FnDecl(self, node, chunk):
        fn_chunk = Chunk()
        for p in node.params:
            fn_chunk.add_name(p)
        for s in node.body:
            self.compile_node(s, fn_chunk)
        # Implicit None return
        none_idx = fn_chunk.add_constant(None)
        fn_chunk.emit(Op.CONST, none_idx)
        fn_chunk.emit(Op.RETURN)
        fn_obj = FnObject(name=node.name, params=node.params, chunk=fn_chunk)
        self.functions[node.name] = fn_obj
        idx = chunk.add_constant(fn_obj)
        chunk.emit(Op.CONST, idx)
        name_idx = chunk.add_name(node.name)
        chunk.emit(Op.STORE, name_idx)

    def compile_ReturnStmt(self, node, chunk):
        if node.value:
            self.compile_node(node.value, chunk)
        else:
            idx = chunk.add_constant(None)
            chunk.emit(Op.CONST, idx)
        chunk.emit(Op.RETURN)

    def compile_Block(self, node, chunk):
        for s in node.stmts:
            self.compile_node(s, chunk)

    # Concurrency compilation
    def compile_SpawnExpr(self, node, chunk):
        name_idx = chunk.add_name(node.callee)
        chunk.emit(Op.LOAD, name_idx)
        for arg in node.args:
            self.compile_node(arg, chunk)
        argc_idx = chunk.add_constant(len(node.args))
        chunk.emit(Op.SPAWN, argc_idx)

    def compile_YieldStmt(self, node, chunk):
        chunk.emit(Op.YIELD_OP)

    def compile_ChanExpr(self, node, chunk):
        if node.size is not None:
            self.compile_node(node.size, chunk)
        else:
            idx = chunk.add_constant(1)
            chunk.emit(Op.CONST, idx)
        chunk.emit(Op.CHAN_NEW)

    def compile_SendExpr(self, node, chunk):
        self.compile_node(node.channel, chunk)
        self.compile_node(node.value, chunk)
        chunk.emit(Op.CHAN_SEND)

    def compile_RecvExpr(self, node, chunk):
        self.compile_node(node.channel, chunk)
        chunk.emit(Op.CHAN_RECV)

    def compile_JoinExpr(self, node, chunk):
        self.compile_node(node.task, chunk)
        chunk.emit(Op.TASK_JOIN)

    def compile_TaskIdExpr(self, node, chunk):
        chunk.emit(Op.TASK_ID_OP)

    def compile_SelectStmt(self, node, chunk):
        end_jumps = []
        for case in node.cases:
            if case.kind == 'send':
                self.compile_node(case.channel, chunk)
                self.compile_node(case.value, chunk)
                chunk.emit(Op.CHAN_TRY_SEND)
                skip = chunk.emit(Op.JUMP_IF_FALSE, 0)
                for s in case.body:
                    self.compile_node(s, chunk)
                end_jumps.append(chunk.emit(Op.JUMP, 0))
                chunk.patch(skip + 1, len(chunk.code))
            elif case.kind == 'recv':
                self.compile_node(case.channel, chunk)
                chunk.emit(Op.CHAN_TRY_RECV)
                # Stack: [value, success]
                skip = chunk.emit(Op.JUMP_IF_FALSE, 0)
                # Store received value
                if case.var_name:
                    name_idx = chunk.add_name(case.var_name)
                    chunk.emit(Op.STORE, name_idx)
                else:
                    chunk.emit(Op.POP)
                for s in case.body:
                    self.compile_node(s, chunk)
                end_jumps.append(chunk.emit(Op.JUMP, 0))
                chunk.patch(skip + 1, len(chunk.code))
                # Pop the value that wasn't used
                chunk.emit(Op.POP)
        if node.default_body:
            for s in node.default_body:
                self.compile_node(s, chunk)
        for j in end_jumps:
            chunk.patch(j + 1, len(chunk.code))

    # Effect compilation
    def compile_EffectDecl(self, node, chunk):
        self.effects[node.name] = [op.name for op in node.operations]

    def compile_PerformExpr(self, node, chunk):
        for arg in node.args:
            self.compile_node(arg, chunk)
        eff_idx = chunk.add_constant(node.effect)
        op_idx = chunk.add_constant(node.operation)
        argc_idx = chunk.add_constant(len(node.args))
        chunk.emit(Op.PERFORM, eff_idx)
        chunk.code.append(op_idx)
        chunk.code.append(argc_idx)

    def compile_HandleWith(self, node, chunk):
        # Compile handler clauses into chunks
        clauses = {}
        for hc in node.handlers:
            hc_chunk = Chunk()
            for p in hc.params:
                hc_chunk.add_name(p)
            hc_chunk.add_name('resume')  # always available
            for s in hc.body:
                self.compile_node(s, hc_chunk)
            none_idx = hc_chunk.add_constant(None)
            hc_chunk.emit(Op.CONST, none_idx)
            hc_chunk.emit(Op.RETURN)
            clause_obj = HandlerClauseObj(
                effect=hc.effect, operation=hc.operation,
                param_names=hc.params, chunk=hc_chunk,
                arity=len(hc.params)
            )
            clauses[(hc.effect, hc.operation)] = clause_obj
        handler_obj = HandlerObject(clauses=clauses)
        handler_idx = chunk.add_constant(handler_obj)
        chunk.emit(Op.INSTALL_HANDLER, handler_idx)
        for s in node.body:
            self.compile_node(s, chunk)
        chunk.emit(Op.REMOVE_HANDLER)

    def compile_ResumeExpr(self, node, chunk):
        self.compile_node(node.value, chunk)
        chunk.emit(Op.RESUME)


# ============================================================
# Runtime: Channel
# ============================================================

class Channel:
    _next_id = 0

    def __init__(self, buffer_size=1):
        self.buffer_size = max(1, buffer_size)
        self.buffer = deque()
        Channel._next_id += 1
        self.id = Channel._next_id
        self.closed = False

    def can_send(self):
        return not self.closed and len(self.buffer) < self.buffer_size

    def can_recv(self):
        return len(self.buffer) > 0

    def try_send(self, value):
        if self.can_send():
            self.buffer.append(value)
            return True
        return False

    def try_recv(self):
        if self.can_recv():
            return True, self.buffer.popleft()
        return False, None

    def close(self):
        self.closed = True

    def __repr__(self):
        return f"Channel({self.id}, buf={self.buffer_size})"

# ============================================================
# Runtime: Task
# ============================================================

class TaskState(IntEnum):
    READY = 0
    RUNNING = 1
    BLOCKED_SEND = 2
    BLOCKED_RECV = 3
    BLOCKED_JOIN = 4
    COMPLETED = 5
    FAILED = 6

@dataclass
class CallFrame:
    chunk: Chunk
    ip: int
    base_env: dict

@dataclass
class HandlerFrame:
    handler: HandlerObject
    chunk: Chunk
    ip: int
    stack_depth: int
    env: dict
    call_stack_depth: int

@dataclass
class Continuation:
    chunk: Chunk
    ip: int
    stack: list
    env: dict
    call_stack: list
    handler_stack: list  # handlers above + matched handler

class Task:
    def __init__(self, task_id, chunk, env=None):
        self.id = task_id
        self.chunk = chunk
        self.ip = 0
        self.stack = []
        self.env = env if env is not None else {}
        self.call_stack = []
        self.handler_stack = []
        self.resume_continuations = []
        self.state = TaskState.READY
        self.result = None
        self.error = None
        self.step_count = 0
        self.blocked_channel = None
        self.blocked_value = None
        self.blocked_on_task = None
        self.current_chunk = chunk

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if not self.stack:
            raise RuntimeError(f"Task {self.id}: stack underflow")
        return self.stack.pop()

    def peek(self):
        if not self.stack:
            return None
        return self.stack[-1]


# ============================================================
# ConcurrentEffectVM
# ============================================================

class EffectError(Exception):
    pass

class ResumeError(Exception):
    pass

class ConcurrentEffectVM:
    def __init__(self, chunk, functions=None, max_steps_per_task=1000,
                 max_total_steps=500000, trace=False):
        self.functions = functions or {}
        self.max_steps_per_task = max_steps_per_task
        self.max_total_steps = max_total_steps
        self.trace = trace
        self.output = []
        self.total_steps = 0
        self.next_task_id = 0
        self.tasks = {}
        self.run_queue = deque()

        # Create main task
        main = self._create_task(chunk)
        self.main_task_id = main.id
        # Register functions in main task env
        for name, fn_obj in self.functions.items():
            main.env[name] = fn_obj

    def _create_task(self, chunk, env=None):
        tid = self.next_task_id
        self.next_task_id += 1
        task = Task(tid, chunk, env)
        self.tasks[tid] = task
        task.state = TaskState.READY
        self.run_queue.append(tid)
        return task

    def run(self):
        while self.run_queue:
            if self.total_steps >= self.max_total_steps:
                break
            tid = self.run_queue.popleft()
            task = self.tasks[tid]
            if task.state not in (TaskState.READY,):
                continue
            task.state = TaskState.RUNNING
            try:
                yielded = self._run_task(task)
            except (EffectError, ResumeError) as e:
                task.state = TaskState.FAILED
                task.error = str(e)
                if tid == self.main_task_id:
                    raise
                self._wake_joiners(task)
                continue
            except Exception as e:
                task.state = TaskState.FAILED
                task.error = str(e)
                if tid == self.main_task_id:
                    raise
                self._wake_joiners(task)
                continue

            if yielded:
                if task.state == TaskState.RUNNING:
                    task.state = TaskState.READY
                    self.run_queue.append(tid)
                # If blocked (BLOCKED_SEND/RECV/JOIN), don't re-queue
            else:
                task.state = TaskState.COMPLETED
                task.result = task.pop() if task.stack else None
                self._wake_joiners(task)

            self._process_blocked_tasks()

        main = self.tasks[self.main_task_id]
        return main.result

    def _run_task(self, task):
        """Execute task instructions. Returns True if yielded/blocked, False if completed."""
        steps = 0
        while steps < self.max_steps_per_task and self.total_steps < self.max_total_steps:
            chunk = task.current_chunk
            if task.ip >= len(chunk.code):
                if task.call_stack:
                    frame = task.call_stack.pop()
                    task.current_chunk = frame.chunk
                    task.ip = frame.ip
                    task.env = frame.base_env
                    continue
                return False  # completed

            op = chunk.code[task.ip]
            task.ip += 1
            steps += 1
            self.total_steps += 1

            if self.trace:
                name = OP_NAMES.get(op, f"?{op}")
                stack_preview = task.stack[-5:] if task.stack else []
                print(f"T{task.id} [{task.ip-1:04d}] {name:20s} stack={stack_preview}")

            if op == Op.CONST:
                idx = chunk.code[task.ip]; task.ip += 1
                task.push(chunk.constants[idx])

            elif op == Op.ADD:
                b, a = task.pop(), task.pop()
                if isinstance(a, str) or isinstance(b, str):
                    task.push(str(a) + str(b))
                else:
                    task.push(a + b)
            elif op == Op.SUB:
                b, a = task.pop(), task.pop()
                task.push(a - b)
            elif op == Op.MUL:
                b, a = task.pop(), task.pop()
                task.push(a * b)
            elif op == Op.DIV:
                b, a = task.pop(), task.pop()
                task.push(a / b if isinstance(a, float) or isinstance(b, float) else a // b)
            elif op == Op.MOD:
                b, a = task.pop(), task.pop()
                task.push(a % b)
            elif op == Op.EQ:
                b, a = task.pop(), task.pop()
                task.push(type(a) is type(b) and a == b if isinstance(a, bool) or isinstance(b, bool) else a == b)
            elif op == Op.NEQ:
                b, a = task.pop(), task.pop()
                task.push(not (type(a) is type(b) and a == b) if isinstance(a, bool) or isinstance(b, bool) else a != b)
            elif op == Op.LT:
                b, a = task.pop(), task.pop()
                task.push(a < b)
            elif op == Op.GT:
                b, a = task.pop(), task.pop()
                task.push(a > b)
            elif op == Op.LTE:
                b, a = task.pop(), task.pop()
                task.push(a <= b)
            elif op == Op.GTE:
                b, a = task.pop(), task.pop()
                task.push(a >= b)
            elif op == Op.AND:
                b, a = task.pop(), task.pop()
                task.push(a and b)
            elif op == Op.OR:
                b, a = task.pop(), task.pop()
                task.push(a or b)
            elif op == Op.NOT:
                task.push(not task.pop())
            elif op == Op.NEG:
                task.push(-task.pop())

            elif op == Op.LOAD:
                idx = chunk.code[task.ip]; task.ip += 1
                name = chunk.names[idx]
                if name in task.env:
                    task.push(task.env[name])
                else:
                    raise RuntimeError(f"Undefined variable: {name}")

            elif op == Op.STORE:
                idx = chunk.code[task.ip]; task.ip += 1
                name = chunk.names[idx]
                task.env[name] = task.pop()

            elif op == Op.JUMP:
                addr = chunk.code[task.ip]; task.ip += 1
                task.ip = addr

            elif op == Op.JUMP_IF_FALSE:
                addr = chunk.code[task.ip]; task.ip += 1
                if not task.pop():
                    task.ip = addr

            elif op == Op.CALL:
                argc_idx = chunk.code[task.ip]; task.ip += 1
                argc = chunk.constants[argc_idx]
                args = []
                for _ in range(argc):
                    args.insert(0, task.pop())
                fn = task.pop()
                if not isinstance(fn, FnObject):
                    raise RuntimeError(f"Not callable: {fn}")
                # Save frame
                frame = CallFrame(chunk=task.current_chunk, ip=task.ip, base_env=dict(task.env))
                task.call_stack.append(frame)
                # Set up new frame
                task.current_chunk = fn.chunk
                task.ip = 0
                new_env = dict(task.env)
                for i, param in enumerate(fn.params):
                    if i < len(args):
                        new_env[param] = args[i]
                task.env = new_env

            elif op == Op.RETURN:
                ret_val = task.pop()
                if not task.call_stack:
                    task.push(ret_val)
                    return False  # task completed
                frame = task.call_stack.pop()
                task.current_chunk = frame.chunk
                task.ip = frame.ip
                task.env = frame.base_env
                task.push(ret_val)

            elif op == Op.PRINT:
                val = task.pop()
                self.output.append(str(val))

            elif op == Op.POP:
                task.pop()
            elif op == Op.DUP:
                v = task.pop()
                task.push(v)
                task.push(v)
            elif op == Op.HALT:
                if task.stack:
                    return False
                task.push(None)
                return False

            # ---- Concurrency ops ----
            elif op == Op.SPAWN:
                argc_idx = chunk.code[task.ip]; task.ip += 1
                argc = chunk.constants[argc_idx]
                args = []
                for _ in range(argc):
                    args.insert(0, task.pop())
                fn = task.pop()
                if not isinstance(fn, FnObject):
                    raise RuntimeError(f"Cannot spawn non-function: {fn}")
                # Create new task with copy of parent env
                new_env = dict(task.env)
                for i, param in enumerate(fn.params):
                    if i < len(args):
                        new_env[param] = args[i]
                new_task = self._create_task(fn.chunk, new_env)
                # Inherit handler stack (deep copy)
                new_task.handler_stack = [
                    HandlerFrame(
                        handler=hf.handler,
                        chunk=hf.chunk,
                        ip=hf.ip,
                        stack_depth=0,
                        env=dict(hf.env),
                        call_stack_depth=0
                    ) for hf in task.handler_stack
                ]
                task.push(new_task.id)

            elif op == Op.YIELD_OP:
                return True  # yield to scheduler

            elif op == Op.CHAN_NEW:
                size = task.pop()
                size = int(size) if isinstance(size, (int, float)) else 1
                task.push(Channel(buffer_size=size))

            elif op == Op.CHAN_SEND:
                val = task.pop()
                ch = task.pop()
                if not isinstance(ch, Channel):
                    raise RuntimeError(f"Not a channel: {ch}")
                if ch.try_send(val):
                    task.push(True)
                else:
                    # Block
                    task.state = TaskState.BLOCKED_SEND
                    task.blocked_channel = ch
                    task.blocked_value = val
                    return True

            elif op == Op.CHAN_RECV:
                ch = task.pop()
                if not isinstance(ch, Channel):
                    raise RuntimeError(f"Not a channel: {ch}")
                ok, val = ch.try_recv()
                if ok:
                    task.push(val)
                else:
                    # Block
                    task.state = TaskState.BLOCKED_RECV
                    task.blocked_channel = ch
                    return True

            elif op == Op.TASK_JOIN:
                target_id = task.pop()
                target_id = int(target_id)
                if target_id not in self.tasks:
                    raise RuntimeError(f"Unknown task: {target_id}")
                target = self.tasks[target_id]
                if target.state in (TaskState.COMPLETED, TaskState.FAILED):
                    task.push(target.result)
                else:
                    task.state = TaskState.BLOCKED_JOIN
                    task.blocked_on_task = target_id
                    return True

            elif op == Op.CHAN_TRY_SEND:
                val = task.pop()
                ch = task.pop()
                if not isinstance(ch, Channel):
                    raise RuntimeError(f"Not a channel: {ch}")
                task.push(ch.try_send(val))

            elif op == Op.CHAN_TRY_RECV:
                ch = task.pop()
                if not isinstance(ch, Channel):
                    raise RuntimeError(f"Not a channel: {ch}")
                ok, val = ch.try_recv()
                if ok:
                    task.push(val)
                    task.push(True)
                else:
                    task.push(None)
                    task.push(False)

            elif op == Op.TASK_ID_OP:
                task.push(task.id)

            # ---- Effect ops ----
            elif op == Op.PERFORM:
                eff_idx = chunk.code[task.ip]; task.ip += 1
                op_idx = chunk.code[task.ip]; task.ip += 1
                argc_idx = chunk.code[task.ip]; task.ip += 1
                effect_name = chunk.constants[eff_idx]
                op_name = chunk.constants[op_idx]
                argc = chunk.constants[argc_idx]
                args = []
                for _ in range(argc):
                    args.insert(0, task.pop())
                # Find handler in task's handler stack
                handler_frame, handler_idx = self._find_handler(task, effect_name, op_name)
                if handler_frame is None:
                    raise EffectError(f"Unhandled effect: {effect_name}.{op_name}")
                self._invoke_handler(task, handler_frame, handler_idx, effect_name, op_name, args)
                # After handler setup, continue execution in handler body

            elif op == Op.INSTALL_HANDLER:
                handler_idx = chunk.code[task.ip]; task.ip += 1
                handler_obj = chunk.constants[handler_idx]
                frame = HandlerFrame(
                    handler=handler_obj,
                    chunk=task.current_chunk,
                    ip=task.ip,
                    stack_depth=len(task.stack),
                    env=dict(task.env),
                    call_stack_depth=len(task.call_stack)
                )
                task.handler_stack.append(frame)

            elif op == Op.REMOVE_HANDLER:
                if task.handler_stack:
                    task.handler_stack.pop()

            elif op == Op.RESUME:
                val = task.pop()
                if not task.resume_continuations:
                    raise ResumeError("resume called outside handler context")
                cont = task.resume_continuations.pop()
                self._restore_continuation(task, cont, val)

        # Time slice exhausted
        return True

    def _find_handler(self, task, effect_name, op_name):
        """Search task's handler stack top-down for matching handler."""
        for i in range(len(task.handler_stack) - 1, -1, -1):
            frame = task.handler_stack[i]
            if (effect_name, op_name) in frame.handler.clauses:
                return frame, i
        return None, -1

    def _invoke_handler(self, task, handler_frame, handler_idx, effect_name, op_name, args):
        """Capture continuation and set up handler clause execution."""
        clause = handler_frame.handler.clauses[(effect_name, op_name)]

        # Capture continuation: save all state above handler
        handlers_above = task.handler_stack[handler_idx:]  # matched handler + above
        cont = Continuation(
            chunk=task.current_chunk,
            ip=task.ip,
            stack=list(task.stack),
            env=dict(task.env),
            call_stack=list(task.call_stack),
            handler_stack=[
                HandlerFrame(
                    handler=hf.handler,
                    chunk=hf.chunk,
                    ip=hf.ip,
                    stack_depth=hf.stack_depth,
                    env=dict(hf.env),
                    call_stack_depth=hf.call_stack_depth
                ) for hf in handlers_above
            ]
        )
        task.resume_continuations.append(cont)

        # Unwind to handler installation point
        task.handler_stack = task.handler_stack[:handler_idx]
        task.call_stack = task.call_stack[:handler_frame.call_stack_depth]
        task.stack = task.stack[:handler_frame.stack_depth]
        task.env = dict(handler_frame.env)

        # Set up handler clause execution
        task.current_chunk = clause.chunk
        task.ip = 0
        new_env = dict(task.env)
        for i, param in enumerate(clause.param_names):
            if i < len(args):
                new_env[param] = args[i]
        new_env['resume'] = '__resume_sentinel__'
        task.env = new_env

    def _restore_continuation(self, task, cont, value):
        """Restore task state from continuation after resume."""
        # Write back handler clause env changes to the matched handler frame.
        # The matched handler is cont.handler_stack[0]. Any variables that
        # existed in the handler frame's env should be updated from the
        # current env (the handler clause's env after execution).
        if cont.handler_stack:
            matched_hf = cont.handler_stack[0]
            for key in list(matched_hf.env.keys()):
                if key in task.env:
                    matched_hf.env[key] = task.env[key]
            # Propagate handler env changes to continuation env.
            # Variables in the enclosing scope (handler frame) that were
            # modified by the handler clause must be visible in the
            # restored continuation's env.
            for key in matched_hf.env:
                if key in cont.env:
                    cont.env[key] = matched_hf.env[key]

        task.current_chunk = cont.chunk
        task.ip = cont.ip
        task.stack = cont.stack
        task.env = cont.env
        task.call_stack = cont.call_stack
        # Re-install saved handlers
        task.handler_stack.extend(cont.handler_stack)
        # Push the resume value (result of perform expression)
        task.push(value)

    def _wake_joiners(self, completed_task):
        """Wake all tasks blocked on join for completed_task."""
        for tid, t in self.tasks.items():
            if t.state == TaskState.BLOCKED_JOIN and t.blocked_on_task == completed_task.id:
                t.state = TaskState.READY
                t.blocked_on_task = None
                t.push(completed_task.result)
                self.run_queue.append(tid)

    def _process_blocked_tasks(self):
        """Try to unblock tasks waiting on channels."""
        changed = True
        while changed:
            changed = False
            for tid, t in self.tasks.items():
                if t.state == TaskState.BLOCKED_SEND:
                    ch = t.blocked_channel
                    if ch.can_send():
                        ch.try_send(t.blocked_value)
                        t.blocked_channel = None
                        t.blocked_value = None
                        t.state = TaskState.READY
                        t.push(True)
                        self.run_queue.append(tid)
                        changed = True
                elif t.state == TaskState.BLOCKED_RECV:
                    ch = t.blocked_channel
                    if ch.can_recv():
                        ok, val = ch.try_recv()
                        t.blocked_channel = None
                        t.state = TaskState.READY
                        t.push(val)
                        self.run_queue.append(tid)
                        changed = True


# ============================================================
# Public API
# ============================================================

def parse(source):
    """Parse source code into AST."""
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()

def compile_program(ast):
    """Compile AST to bytecode."""
    compiler = Compiler()
    chunk = compiler.compile(ast)
    return chunk, compiler

def run(source, trace=False, max_steps_per_task=1000, max_total_steps=500000):
    """Full pipeline: parse -> compile -> execute.
    Returns (result, output, task_info)."""
    ast = parse(source)
    chunk, compiler = compile_program(ast)
    vm = ConcurrentEffectVM(
        chunk, functions=compiler.functions,
        max_steps_per_task=max_steps_per_task,
        max_total_steps=max_total_steps,
        trace=trace
    )
    result = vm.run()
    task_info = {}
    for tid, t in vm.tasks.items():
        task_info[tid] = {
            'state': t.state.name,
            'steps': t.step_count,
            'result': t.result,
            'error': t.error
        }
    return result, vm.output, task_info
