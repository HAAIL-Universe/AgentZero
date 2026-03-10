"""
Concurrent Task Runtime for Stack VM
Challenge C029 -- AgentZero Session 030

Adds cooperative concurrency to the C010 stack VM:
  - Tasks (lightweight coroutines with own stack/env)
  - Channels (buffered communication between tasks)
  - Cooperative yielding + auto-preemption for fairness
  - Select (multiplexed channel operations)
  - Join (wait for task completion)

New language constructs:
  spawn func(args)    -- create task, returns task ID
  yield;              -- cooperatively yield to scheduler
  chan(size)           -- create buffered channel (default size=1)
  send(ch, val)       -- send to channel (blocks if full)
  recv(ch)            -- receive from channel (blocks if empty)
  select { ... }      -- multiplex channel operations
  join(task_id)       -- wait for task to finish, get result

Architecture:
  Extended Lexer/Parser/Compiler -> Extended Bytecode -> ConcurrentVM (scheduler + tasks)
"""

import sys
import os
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import deque
import copy

# Import C010 stack VM components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op, Chunk, TokenType, Token, KEYWORDS,
    lex as base_lex, LexError, ParseError, CompileError, VMError,
    FnObject, CallFrame,
    # AST nodes
    Program, IntLit, FloatLit, StringLit, BoolLit,
    Var, BinOp, UnaryOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, ReturnStmt, PrintStmt,
    FnDecl, CallExpr,
)


# ============================================================
# Extended Instruction Set
# ============================================================

class ConcOp(IntEnum):
    """Concurrency opcodes, starting after C010's Op range."""
    SPAWN = 100       # spawn task from function on stack
    YIELD = 101       # cooperative yield
    CHAN_NEW = 102     # create channel (buffer size on stack)
    CHAN_SEND = 103    # send value to channel
    CHAN_RECV = 104    # receive from channel
    TASK_JOIN = 105    # join task (task ID on stack)
    SELECT = 106      # select over channels (operand = case count)
    CHAN_TRY_SEND = 107  # non-blocking send (pushes bool)
    CHAN_TRY_RECV = 108  # non-blocking recv (pushes [bool, val])
    TASK_ID = 109     # push current task ID


# ============================================================
# Extended AST Nodes
# ============================================================

@dataclass
class SpawnExpr:
    callee: str
    args: list
    line: int


@dataclass
class YieldStmt:
    line: int


@dataclass
class ChanExpr:
    size: Any  # expression for buffer size
    line: int


@dataclass
class SendExpr:
    channel: Any  # expression evaluating to channel
    value: Any    # expression for value to send
    line: int


@dataclass
class RecvExpr:
    channel: Any  # expression evaluating to channel
    line: int


@dataclass
class JoinExpr:
    task: Any  # expression evaluating to task ID
    line: int


@dataclass
class SelectStmt:
    """Select over channel operations.
    cases: list of (op, channel_expr, value_expr_or_None, var_name_or_None, body)
      op is 'send' or 'recv'
    default_body: optional Block for default case
    """
    cases: list
    default_body: Any
    line: int


@dataclass
class TaskIdExpr:
    line: int


# ============================================================
# Extended Lexer
# ============================================================

# Add new keywords
CONC_KEYWORDS = {
    'spawn': 'SPAWN',
    'yield': 'YIELD',
    'chan': 'CHAN',
    'send': 'SEND',
    'recv': 'RECV',
    'join': 'JOIN',
    'select': 'SELECT',
    'case': 'CASE',
    'default': 'DEFAULT',
    'task_id': 'TASK_ID',
}

# Extended token types
class ConcTokenType(IntEnum):
    # Include all base types by value
    SPAWN = 200
    YIELD = 201
    CHAN = 202
    SEND = 203
    RECV = 204
    JOIN = 205
    SELECT = 206
    CASE = 207
    DEFAULT = 208
    TASK_ID = 209
    ARROW = 210   # => for select cases


def conc_lex(source: str) -> list:
    """Extended lexer that handles concurrency keywords."""
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
        elif c == '=' and i + 1 < len(source) and source[i + 1] == '>':
            tokens.append(Token(ConcTokenType.ARROW, '=>', line))
            i += 2
        elif c == '=' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.EQ, '==', line))
            i += 2
        elif c == '!' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.NE, '!=', line))
            i += 2
        elif c == '<' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.LE, '<=', line))
            i += 2
        elif c == '>' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.GE, '>=', line))
            i += 2
        elif c == '<':
            tokens.append(Token(TokenType.LT, '<', line))
            i += 1
        elif c == '>':
            tokens.append(Token(TokenType.GT, '>', line))
            i += 1
        elif c == '=':
            tokens.append(Token(TokenType.ASSIGN, '=', line))
            i += 1
        elif c.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            if i < len(source) and source[i] == '.':
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
            # Check concurrency keywords first
            if word in CONC_KEYWORDS:
                tt = getattr(ConcTokenType, CONC_KEYWORDS[word])
                tokens.append(Token(tt, word, line))
            elif word in KEYWORDS:
                tokens.append(Token(KEYWORDS[word], word, line))
            else:
                tokens.append(Token(TokenType.IDENT, word, line))
        elif c == '+':
            tokens.append(Token(TokenType.PLUS, '+', line))
            i += 1
        elif c == '-':
            tokens.append(Token(TokenType.MINUS, '-', line))
            i += 1
        elif c == '*':
            tokens.append(Token(TokenType.STAR, '*', line))
            i += 1
        elif c == '/':
            tokens.append(Token(TokenType.SLASH, '/', line))
            i += 1
        elif c == '%':
            tokens.append(Token(TokenType.PERCENT, '%', line))
            i += 1
        elif c == '(':
            tokens.append(Token(TokenType.LPAREN, '(', line))
            i += 1
        elif c == ')':
            tokens.append(Token(TokenType.RPAREN, ')', line))
            i += 1
        elif c == '{':
            tokens.append(Token(TokenType.LBRACE, '{', line))
            i += 1
        elif c == '}':
            tokens.append(Token(TokenType.RBRACE, '}', line))
            i += 1
        elif c == ',':
            tokens.append(Token(TokenType.COMMA, ',', line))
            i += 1
        elif c == ';':
            tokens.append(Token(TokenType.SEMICOLON, ';', line))
            i += 1
        elif c == '!':
            tokens.append(Token(TokenType.NOT, '!', line))
            i += 1
        else:
            raise LexError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TokenType.EOF, '', line))
    return tokens


# ============================================================
# Extended Parser
# ============================================================

class ConcParser:
    """Parser extended with concurrency constructs."""

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
            raise ParseError(
                f"Expected {tt}, got {t.type} '{t.value}' at line {t.line}")
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
        return self.statement()

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
        return FnDecl(name_tok.value, params, body, name_tok.line)

    def let_decl(self):
        tok = self.advance()  # let
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return LetDecl(name, value, tok.line)

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
        if self.peek().type == ConcTokenType.YIELD:
            return self.yield_stmt()
        if self.peek().type == ConcTokenType.SELECT:
            return self.select_stmt()
        return self.expr_stmt()

    def if_stmt(self):
        tok = self.advance()
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
        return IfStmt(cond, then_body, else_body, tok.line)

    def while_stmt(self):
        tok = self.advance()
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return WhileStmt(cond, body, tok.line)

    def return_stmt(self):
        tok = self.advance()
        value = None
        if self.peek().type != TokenType.SEMICOLON:
            value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(value, tok.line)

    def print_stmt(self):
        tok = self.advance()
        self.expect(TokenType.LPAREN)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(value, tok.line)

    def yield_stmt(self):
        tok = self.advance()  # yield
        self.expect(TokenType.SEMICOLON)
        return YieldStmt(tok.line)

    def select_stmt(self):
        tok = self.advance()  # select
        self.expect(TokenType.LBRACE)
        cases = []
        default_body = None
        while self.peek().type != TokenType.RBRACE:
            if self.peek().type == ConcTokenType.DEFAULT:
                self.advance()  # default
                self.expect(ConcTokenType.ARROW)
                default_body = self.block()
            elif self.peek().type == ConcTokenType.CASE:
                self.advance()  # case
                # case send(ch, val) => { ... }
                # case recv(ch) => name { ... }
                if self.peek().type == ConcTokenType.SEND:
                    self.advance()  # send
                    self.expect(TokenType.LPAREN)
                    ch_expr = self.expression()
                    self.expect(TokenType.COMMA)
                    val_expr = self.expression()
                    self.expect(TokenType.RPAREN)
                    self.expect(ConcTokenType.ARROW)
                    body = self.block()
                    cases.append(('send', ch_expr, val_expr, None, body))
                elif self.peek().type == ConcTokenType.RECV:
                    self.advance()  # recv
                    self.expect(TokenType.LPAREN)
                    ch_expr = self.expression()
                    self.expect(TokenType.RPAREN)
                    var_name = None
                    if self.peek().type == ConcTokenType.ARROW:
                        self.advance()  # =>
                        if self.peek().type == TokenType.IDENT:
                            var_name = self.advance().value
                    body = self.block()
                    cases.append(('recv', ch_expr, None, var_name, body))
                else:
                    raise ParseError(
                        f"Expected 'send' or 'recv' in select case at line {self.peek().line}")
            else:
                raise ParseError(
                    f"Expected 'case' or 'default' in select at line {self.peek().line}")
        self.expect(TokenType.RBRACE)
        return SelectStmt(cases, default_body, tok.line)

    def block(self):
        tok = self.expect(TokenType.LBRACE)
        stmts = []
        while self.peek().type != TokenType.RBRACE:
            stmts.append(self.declaration())
        self.expect(TokenType.RBRACE)
        return Block(stmts, tok.line)

    def expr_stmt(self):
        expr = self.expression()
        self.expect(TokenType.SEMICOLON)
        return expr

    # Expression parsing (precedence climbing)
    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.or_expr()
        if self.match(TokenType.ASSIGN):
            if isinstance(expr, Var):
                value = self.assignment()
                return Assign(expr.name, value, expr.line)
            raise ParseError(f"Invalid assignment target at line {self.peek().line}")
        return expr

    def or_expr(self):
        left = self.and_expr()
        while self.match(TokenType.OR):
            right = self.and_expr()
            left = BinOp('or', left, right, left.line)
        return left

    def and_expr(self):
        left = self.equality()
        while self.match(TokenType.AND):
            right = self.equality()
            left = BinOp('and', left, right, left.line)
        return left

    def equality(self):
        left = self.comparison()
        while True:
            if self.match(TokenType.EQ):
                left = BinOp('==', left, self.comparison(), left.line)
            elif self.match(TokenType.NE):
                left = BinOp('!=', left, self.comparison(), left.line)
            else:
                break
        return left

    def comparison(self):
        left = self.addition()
        while True:
            if self.match(TokenType.LT):
                left = BinOp('<', left, self.addition(), left.line)
            elif self.match(TokenType.GT):
                left = BinOp('>', left, self.addition(), left.line)
            elif self.match(TokenType.LE):
                left = BinOp('<=', left, self.addition(), left.line)
            elif self.match(TokenType.GE):
                left = BinOp('>=', left, self.addition(), left.line)
            else:
                break
        return left

    def addition(self):
        left = self.multiplication()
        while True:
            if self.match(TokenType.PLUS):
                left = BinOp('+', left, self.multiplication(), left.line)
            elif self.match(TokenType.MINUS):
                left = BinOp('-', left, self.multiplication(), left.line)
            else:
                break
        return left

    def multiplication(self):
        left = self.unary()
        while True:
            if self.match(TokenType.STAR):
                left = BinOp('*', left, self.unary(), left.line)
            elif self.match(TokenType.SLASH):
                left = BinOp('/', left, self.unary(), left.line)
            elif self.match(TokenType.PERCENT):
                left = BinOp('%', left, self.unary(), left.line)
            else:
                break
        return left

    def unary(self):
        if self.match(TokenType.MINUS):
            return UnaryOp('-', self.unary(), self.tokens[self.pos - 1].line)
        if self.match(TokenType.NOT):
            return UnaryOp('not', self.unary(), self.tokens[self.pos - 1].line)
        return self.primary()

    def primary(self):
        tok = self.peek()

        if tok.type == TokenType.INT:
            self.advance()
            return IntLit(tok.value, tok.line)
        if tok.type == TokenType.FLOAT:
            self.advance()
            return FloatLit(tok.value, tok.line)
        if tok.type == TokenType.STRING:
            self.advance()
            return StringLit(tok.value, tok.line)
        if tok.type == TokenType.TRUE:
            self.advance()
            return BoolLit(True, tok.line)
        if tok.type == TokenType.FALSE:
            self.advance()
            return BoolLit(False, tok.line)

        # Concurrency builtins
        if tok.type == ConcTokenType.SPAWN:
            return self.spawn_expr()
        if tok.type == ConcTokenType.CHAN:
            return self.chan_expr()
        if tok.type == ConcTokenType.SEND:
            return self.send_expr()
        if tok.type == ConcTokenType.RECV:
            return self.recv_expr()
        if tok.type == ConcTokenType.JOIN:
            return self.join_expr()
        if tok.type == ConcTokenType.TASK_ID:
            self.advance()
            return TaskIdExpr(tok.line)

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
                return CallExpr(tok.value, args, tok.line)
            return Var(tok.value, tok.line)

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN)
            return expr

        raise ParseError(f"Unexpected token {tok.type} '{tok.value}' at line {tok.line}")

    def spawn_expr(self):
        tok = self.advance()  # spawn
        callee = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.expression())
            while self.match(TokenType.COMMA):
                args.append(self.expression())
        self.expect(TokenType.RPAREN)
        return SpawnExpr(callee, args, tok.line)

    def chan_expr(self):
        tok = self.advance()  # chan
        self.expect(TokenType.LPAREN)
        if self.peek().type != TokenType.RPAREN:
            size = self.expression()
        else:
            size = IntLit(1, tok.line)  # default buffer size
        self.expect(TokenType.RPAREN)
        return ChanExpr(size, tok.line)

    def send_expr(self):
        tok = self.advance()  # send
        self.expect(TokenType.LPAREN)
        channel = self.expression()
        self.expect(TokenType.COMMA)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        return SendExpr(channel, value, tok.line)

    def recv_expr(self):
        tok = self.advance()  # recv
        self.expect(TokenType.LPAREN)
        channel = self.expression()
        self.expect(TokenType.RPAREN)
        return RecvExpr(channel, tok.line)

    def join_expr(self):
        tok = self.advance()  # join
        self.expect(TokenType.LPAREN)
        task = self.expression()
        self.expect(TokenType.RPAREN)
        return JoinExpr(task, tok.line)


# ============================================================
# Extended Compiler
# ============================================================

class ConcCompiler:
    """Compiler extended with concurrency opcodes."""

    def __init__(self):
        self.chunk = Chunk()
        self.functions = {}

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

    # -- Literals --
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

    # -- Variables --
    def compile_Var(self, node):
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.LOAD, idx, node.line)

    def compile_Assign(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, idx, node.line)

    def compile_LetDecl(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, idx, node.line)

    # -- Expressions --
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

    def compile_CallExpr(self, node):
        fn_idx = self.chunk.add_name(node.callee)
        self.chunk.emit(Op.LOAD, fn_idx, node.line)
        for arg in node.args:
            self.compile_node(arg)
        self.chunk.emit(Op.CALL, len(node.args), node.line)

    # -- Statements --
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

    def compile_FnDecl(self, node):
        fn_compiler = ConcCompiler()
        for param in node.params:
            fn_compiler.chunk.add_name(param)
        fn_compiler.compile_node(node.body)
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx)
        fn_compiler.chunk.emit(Op.RETURN)

        fn_obj = FnObject(node.name, len(node.params), fn_compiler.chunk)
        self.functions[node.name] = fn_obj
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v

        fn_idx = self.chunk.add_constant(fn_obj)
        name_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.STORE, name_idx, node.line)

    # -- Concurrency --
    def compile_SpawnExpr(self, node):
        # Push function reference, then args, then SPAWN with arg count
        fn_idx = self.chunk.add_name(node.callee)
        self.chunk.emit(Op.LOAD, fn_idx, node.line)
        for arg in node.args:
            self.compile_node(arg)
        self.chunk.emit(ConcOp.SPAWN, len(node.args), node.line)

    def compile_YieldStmt(self, node):
        self.chunk.emit(ConcOp.YIELD, line=node.line)

    def compile_ChanExpr(self, node):
        self.compile_node(node.size)
        self.chunk.emit(ConcOp.CHAN_NEW, line=node.line)

    def compile_SendExpr(self, node):
        self.compile_node(node.channel)
        self.compile_node(node.value)
        self.chunk.emit(ConcOp.CHAN_SEND, line=node.line)

    def compile_RecvExpr(self, node):
        self.compile_node(node.channel)
        self.chunk.emit(ConcOp.CHAN_RECV, line=node.line)

    def compile_JoinExpr(self, node):
        self.compile_node(node.task)
        self.chunk.emit(ConcOp.TASK_JOIN, line=node.line)

    def compile_TaskIdExpr(self, node):
        self.chunk.emit(ConcOp.TASK_ID, line=node.line)

    def compile_SelectStmt(self, node):
        # Compile select as a sequence of try-send/try-recv checks
        # For each case: compile channel expr, try op, jump to body if successful
        # Fall through to default if no case matches

        case_jumps = []  # (jump_to_body_addr, body_node, var_name)
        end_jumps = []

        for op_type, ch_expr, val_expr, var_name, body in node.cases:
            self.compile_node(ch_expr)
            if op_type == 'send':
                self.compile_node(val_expr)
                self.chunk.emit(ConcOp.CHAN_TRY_SEND, line=node.line)
                # TRY_SEND pushes bool (success)
                jump_next = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
                self.chunk.emit(Op.POP, line=node.line)  # pop true
                # Execute body
                self.compile_node(body)
                end_jump = self.chunk.emit(Op.JUMP, 0, node.line)
                end_jumps.append(end_jump)
                self.chunk.patch(jump_next + 1, len(self.chunk.code))
                self.chunk.emit(Op.POP, line=node.line)  # pop false
            elif op_type == 'recv':
                self.chunk.emit(ConcOp.CHAN_TRY_RECV, line=node.line)
                # TRY_RECV pushes [success_bool, value]
                # We need to check the bool
                # Stack: ... success value
                # Swap to get success on top -- emit DUP of second-from-top
                # Actually, let's push [value, success] with success on top
                jump_next = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
                self.chunk.emit(Op.POP, line=node.line)  # pop true
                # value is on stack
                if var_name:
                    name_idx = self.chunk.add_name(var_name)
                    self.chunk.emit(Op.STORE, name_idx, node.line)
                else:
                    self.chunk.emit(Op.POP, line=node.line)  # discard value
                self.compile_node(body)
                end_jump = self.chunk.emit(Op.JUMP, 0, node.line)
                end_jumps.append(end_jump)
                self.chunk.patch(jump_next + 1, len(self.chunk.code))
                self.chunk.emit(Op.POP, line=node.line)  # pop false
                self.chunk.emit(Op.POP, line=node.line)  # pop None value

        # Default case
        if node.default_body:
            self.compile_node(node.default_body)

        # Patch end jumps
        end_addr = len(self.chunk.code)
        for ej in end_jumps:
            self.chunk.patch(ej + 1, end_addr)


# ============================================================
# Channel
# ============================================================

class Channel:
    """Buffered channel for inter-task communication."""

    _next_id = 0

    def __init__(self, buffer_size=1):
        self.buffer_size = max(1, buffer_size)
        self.buffer = deque()
        self.id = Channel._next_id
        Channel._next_id += 1
        self.closed = False
        # Waiters: tasks blocked on this channel
        self.send_waiters = deque()  # (task_id, value)
        self.recv_waiters = deque()  # task_id

    def can_send(self):
        return not self.closed and len(self.buffer) < self.buffer_size

    def can_recv(self):
        return len(self.buffer) > 0

    def try_send(self, value):
        if self.closed:
            return False
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(value)
            return True
        return False

    def try_recv(self):
        if self.buffer:
            return True, self.buffer.popleft()
        return False, None

    def close(self):
        self.closed = True

    def __repr__(self):
        return f"<chan:{self.id} buf={len(self.buffer)}/{self.buffer_size}>"


# ============================================================
# Task
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
class Task:
    """A lightweight concurrent task (coroutine)."""
    id: int
    chunk: Chunk
    ip: int = 0
    stack: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    call_stack: list = field(default_factory=list)
    state: TaskState = TaskState.READY
    result: Any = None
    error: Optional[str] = None
    step_count: int = 0
    # Blocking info
    blocked_channel: Optional[Channel] = None
    blocked_value: Any = None  # value to send when unblocked
    blocked_on_task: Optional[int] = None  # task ID we're joining
    current_chunk: Chunk = None

    def __post_init__(self):
        if self.current_chunk is None:
            self.current_chunk = self.chunk


# ============================================================
# Concurrent VM (Scheduler + Task Execution)
# ============================================================

class ConcurrentVM:
    """VM that manages multiple concurrent tasks with cooperative scheduling."""

    def __init__(self, chunk: Chunk, functions=None, max_steps_per_task=1000,
                 max_total_steps=500000, trace=False):
        self.functions = functions or {}
        self.max_steps_per_task = max_steps_per_task
        self.max_total_steps = max_total_steps
        self.trace = trace
        self.output = []
        self.total_steps = 0

        # Task management
        self.next_task_id = 0
        self.tasks = {}  # id -> Task
        self.run_queue = deque()  # task IDs ready to run

        # Create main task
        main_task = self._create_task(chunk)
        # Store function references in main task env
        for name, fn_obj in self.functions.items():
            main_task.env[name] = fn_obj
        self.main_task_id = main_task.id

    def _create_task(self, chunk, env=None):
        task_id = self.next_task_id
        self.next_task_id += 1
        task = Task(id=task_id, chunk=chunk, env=env or {})
        self.tasks[task_id] = task
        self.run_queue.append(task_id)
        return task

    def run(self):
        """Run all tasks to completion using round-robin scheduling."""
        while self.run_queue:
            if self.total_steps > self.max_total_steps:
                raise VMError(f"Total execution limit exceeded ({self.max_total_steps} steps)")

            task_id = self.run_queue.popleft()
            task = self.tasks.get(task_id)
            if task is None or task.state in (TaskState.COMPLETED, TaskState.FAILED):
                continue

            task.state = TaskState.RUNNING
            try:
                yielded = self._run_task(task)
            except VMError as e:
                task.state = TaskState.FAILED
                task.error = str(e)
                self._wake_joiners(task)
                if task.id == self.main_task_id:
                    raise
                continue

            if task.state == TaskState.RUNNING:
                # Task completed normally or yielded
                if yielded:
                    task.state = TaskState.READY
                    self.run_queue.append(task.id)
                else:
                    task.state = TaskState.COMPLETED
                    self._wake_joiners(task)
            elif task.state in (TaskState.BLOCKED_SEND, TaskState.BLOCKED_RECV,
                                TaskState.BLOCKED_JOIN):
                # Task is blocked, don't re-queue until unblocked
                pass

            # Try to unblock waiting tasks
            self._process_blocked_tasks()

        # Return main task result
        main_task = self.tasks[self.main_task_id]
        if main_task.state == TaskState.FAILED:
            raise VMError(main_task.error)
        return main_task.result

    def _wake_joiners(self, completed_task):
        """Wake any tasks that are joining on the completed task."""
        for tid, t in self.tasks.items():
            if t.state == TaskState.BLOCKED_JOIN and t.blocked_on_task == completed_task.id:
                t.state = TaskState.READY
                t.blocked_on_task = None
                # Push the completed task's result onto the joiner's stack
                result = completed_task.result
                if completed_task.state == TaskState.FAILED:
                    result = None  # Could push error, but keep simple
                t.stack.append(result)
                if tid not in self.run_queue:
                    self.run_queue.append(tid)

    def _process_blocked_tasks(self):
        """Try to unblock tasks waiting on channels."""
        changed = True
        while changed:
            changed = False
            for tid, task in list(self.tasks.items()):
                if task.state == TaskState.BLOCKED_SEND:
                    ch = task.blocked_channel
                    if ch.can_send():
                        ch.buffer.append(task.blocked_value)
                        task.blocked_channel = None
                        task.blocked_value = None
                        task.state = TaskState.READY
                        task.stack.append(True)  # send succeeded
                        if tid not in self.run_queue:
                            self.run_queue.append(tid)
                        changed = True
                elif task.state == TaskState.BLOCKED_RECV:
                    ch = task.blocked_channel
                    success, value = ch.try_recv()
                    if success:
                        task.blocked_channel = None
                        task.state = TaskState.READY
                        task.stack.append(value)
                        if tid not in self.run_queue:
                            self.run_queue.append(tid)
                        changed = True

    def _run_task(self, task):
        """Execute a task for up to max_steps_per_task instructions.
        Returns True if yielded (should be re-queued), False if completed."""
        steps_this_slice = 0
        chunk = task.current_chunk

        while steps_this_slice < self.max_steps_per_task:
            self.total_steps += 1
            task.step_count += 1
            steps_this_slice += 1

            if self.total_steps > self.max_total_steps:
                raise VMError(f"Total execution limit exceeded ({self.max_total_steps} steps)")

            if task.ip >= len(chunk.code):
                task.result = task.stack[-1] if task.stack else None
                return False  # completed

            op = chunk.code[task.ip]
            task.ip += 1

            if self.trace:
                name = (Op(op).name if op in Op._value2member_map_
                        else ConcOp(op).name if op in ConcOp._value2member_map_
                        else f"??({op})")
                print(f"  T{task.id} [{task.ip-1:04d}] {name:20s} stack={task.stack[-5:]}")

            # === Standard opcodes ===
            if op == Op.HALT:
                task.result = task.stack[-1] if task.stack else None
                return False  # completed

            elif op == Op.CONST:
                idx = chunk.code[task.ip]
                task.ip += 1
                task.stack.append(chunk.constants[idx])

            elif op == Op.POP:
                if task.stack:
                    task.stack.pop()

            elif op == Op.DUP:
                if not task.stack:
                    raise VMError("Stack underflow on DUP")
                task.stack.append(task.stack[-1])

            elif op == Op.ADD:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a + b)
            elif op == Op.SUB:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a - b)
            elif op == Op.MUL:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a * b)
            elif op == Op.DIV:
                b, a = task.stack.pop(), task.stack.pop()
                if b == 0:
                    raise VMError("Division by zero")
                if isinstance(a, int) and isinstance(b, int):
                    task.stack.append(a // b)
                else:
                    task.stack.append(a / b)
            elif op == Op.MOD:
                b, a = task.stack.pop(), task.stack.pop()
                if b == 0:
                    raise VMError("Modulo by zero")
                task.stack.append(a % b)
            elif op == Op.NEG:
                task.stack.append(-task.stack.pop())

            elif op == Op.EQ:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a == b)
            elif op == Op.NE:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a != b)
            elif op == Op.LT:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a < b)
            elif op == Op.GT:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a > b)
            elif op == Op.LE:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a <= b)
            elif op == Op.GE:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a >= b)

            elif op == Op.NOT:
                task.stack.append(not task.stack.pop())
            elif op == Op.AND:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a and b)
            elif op == Op.OR:
                b, a = task.stack.pop(), task.stack.pop()
                task.stack.append(a or b)

            elif op == Op.LOAD:
                idx = chunk.code[task.ip]
                task.ip += 1
                name = chunk.names[idx]
                if name not in task.env:
                    raise VMError(f"Undefined variable '{name}'")
                task.stack.append(task.env[name])

            elif op == Op.STORE:
                idx = chunk.code[task.ip]
                task.ip += 1
                name = chunk.names[idx]
                value = task.stack.pop()
                task.env[name] = value

            elif op == Op.JUMP:
                target = chunk.code[task.ip]
                task.ip = target

            elif op == Op.JUMP_IF_FALSE:
                target = chunk.code[task.ip]
                task.ip += 1
                if not task.stack[-1]:
                    task.ip = target

            elif op == Op.JUMP_IF_TRUE:
                target = chunk.code[task.ip]
                task.ip += 1
                if task.stack[-1]:
                    task.ip = target

            elif op == Op.CALL:
                arg_count = chunk.code[task.ip]
                task.ip += 1
                args = []
                for _ in range(arg_count):
                    args.insert(0, task.stack.pop())
                fn_obj = task.stack.pop()
                if not isinstance(fn_obj, FnObject):
                    raise VMError(f"Cannot call non-function: {fn_obj}")
                if fn_obj.arity != arg_count:
                    raise VMError(
                        f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")

                frame = CallFrame(chunk, task.ip, dict(task.env))
                task.call_stack.append(frame)
                chunk = fn_obj.chunk
                task.current_chunk = chunk
                task.ip = 0
                for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                    task.env[param_name] = args[i]

            elif op == Op.RETURN:
                return_val = task.stack.pop()
                if not task.call_stack:
                    task.stack.append(return_val)
                    task.result = return_val
                    return False  # completed
                frame = task.call_stack.pop()
                chunk = frame.chunk
                task.current_chunk = chunk
                task.ip = frame.ip
                task.env = frame.base_env
                task.stack.append(return_val)

            elif op == Op.PRINT:
                value = task.stack.pop()
                text = str(value) if value is not None else "None"
                if isinstance(value, bool):
                    text = "true" if value else "false"
                self.output.append(text)

            # === Concurrency opcodes ===
            elif op == ConcOp.SPAWN:
                arg_count = chunk.code[task.ip]
                task.ip += 1
                args = []
                for _ in range(arg_count):
                    args.insert(0, task.stack.pop())
                fn_obj = task.stack.pop()
                if not isinstance(fn_obj, FnObject):
                    raise VMError(f"Cannot spawn non-function: {fn_obj}")
                if fn_obj.arity != arg_count:
                    raise VMError(
                        f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")

                # Create new task with function's chunk
                new_task = self._create_task(fn_obj.chunk)
                # Copy parent env so spawned task can access globals/channels
                new_task.env = dict(task.env)
                # Bind parameters
                for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                    new_task.env[param_name] = args[i]
                # Push task ID onto parent's stack
                task.stack.append(new_task.id)

            elif op == ConcOp.YIELD:
                task.current_chunk = chunk
                return True  # yielded

            elif op == ConcOp.CHAN_NEW:
                size = task.stack.pop()
                if not isinstance(size, int) or size < 1:
                    raise VMError(f"Channel buffer size must be positive integer, got {size}")
                ch = Channel(buffer_size=size)
                task.stack.append(ch)

            elif op == ConcOp.CHAN_SEND:
                value = task.stack.pop()
                ch = task.stack.pop()
                if not isinstance(ch, Channel):
                    raise VMError(f"Cannot send to non-channel: {ch}")
                if ch.closed:
                    raise VMError("Cannot send to closed channel")
                if ch.try_send(value):
                    task.stack.append(True)  # send succeeded
                else:
                    # Block this task
                    task.state = TaskState.BLOCKED_SEND
                    task.blocked_channel = ch
                    task.blocked_value = value
                    task.current_chunk = chunk
                    return True  # suspend (but not re-queued as READY)

            elif op == ConcOp.CHAN_RECV:
                ch = task.stack.pop()
                if not isinstance(ch, Channel):
                    raise VMError(f"Cannot recv from non-channel: {ch}")
                success, value = ch.try_recv()
                if success:
                    task.stack.append(value)
                else:
                    # Block this task
                    task.state = TaskState.BLOCKED_RECV
                    task.blocked_channel = ch
                    task.current_chunk = chunk
                    return True  # suspend

            elif op == ConcOp.TASK_JOIN:
                target_id = task.stack.pop()
                if not isinstance(target_id, int):
                    raise VMError(f"Cannot join non-task-id: {target_id}")
                target = self.tasks.get(target_id)
                if target is None:
                    raise VMError(f"Unknown task ID: {target_id}")
                if target.state in (TaskState.COMPLETED, TaskState.FAILED):
                    # Already done
                    task.stack.append(target.result)
                else:
                    # Block until target completes
                    task.state = TaskState.BLOCKED_JOIN
                    task.blocked_on_task = target_id
                    task.current_chunk = chunk
                    return True  # suspend

            elif op == ConcOp.TASK_ID:
                task.stack.append(task.id)

            elif op == ConcOp.CHAN_TRY_SEND:
                value = task.stack.pop()
                ch = task.stack.pop()
                if not isinstance(ch, Channel):
                    raise VMError(f"Cannot send to non-channel: {ch}")
                success = ch.try_send(value)
                task.stack.append(success)

            elif op == ConcOp.CHAN_TRY_RECV:
                ch = task.stack.pop()
                if not isinstance(ch, Channel):
                    raise VMError(f"Cannot recv from non-channel: {ch}")
                success, value = ch.try_recv()
                # Push value first, then success (success on top for JUMP_IF_FALSE)
                task.stack.append(value)
                task.stack.append(success)

            elif op == ConcOp.SELECT:
                # SELECT is handled at compile level as try operations
                # This opcode is reserved but not directly used
                pass

            else:
                raise VMError(f"Unknown opcode: {op}")

        # Time slice exhausted -- auto-yield for fairness
        task.current_chunk = chunk
        return True

    def get_task_info(self):
        """Return info about all tasks."""
        info = {}
        for tid, task in self.tasks.items():
            info[tid] = {
                'state': task.state.name,
                'steps': task.step_count,
                'result': task.result,
                'error': task.error,
            }
        return info


# ============================================================
# Public API
# ============================================================

def compile_concurrent(source: str) -> tuple:
    """Compile source with concurrency extensions. Returns (chunk, compiler)."""
    tokens = conc_lex(source)
    parser = ConcParser(tokens)
    ast = parser.parse()
    compiler = ConcCompiler()
    chunk = compiler.compile(ast)
    return chunk, compiler


def execute_concurrent(source: str, max_steps_per_task=1000,
                       max_total_steps=500000, trace=False) -> dict:
    """Compile and execute source with concurrency. Returns result dict."""
    chunk, compiler = compile_concurrent(source)
    vm = ConcurrentVM(
        chunk,
        functions=compiler.functions,
        max_steps_per_task=max_steps_per_task,
        max_total_steps=max_total_steps,
        trace=trace,
    )
    result = vm.run()
    return {
        'result': result,
        'output': vm.output,
        'tasks': vm.get_task_info(),
        'total_steps': vm.total_steps,
    }


def create_channel(buffer_size=1):
    """Create a channel (for use in Python-level testing)."""
    return Channel(buffer_size=buffer_size)
