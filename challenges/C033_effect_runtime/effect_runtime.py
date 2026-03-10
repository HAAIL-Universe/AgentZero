"""
C033: Effect Handlers Runtime
Composes C032 (effect system type checker) + C010 (stack VM)

A complete runtime for algebraic effects:
- Programs are parsed, type-checked for effect safety, compiled to bytecode, and executed
- Runtime effect handlers with perform, handle, and resume
- Continuation capture for resumable effects
- Nested handler scopes with proper unwinding
- Built-in effects: IO, State, Error + custom effects
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Import C010 Stack VM (we extend it, not wrap it)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op as BaseOp, Token, TokenType, Chunk, VM as BaseVM,
    IntLit, FloatLit, StringLit, BoolLit, Var, Assign, UnaryOp, BinOp,
    LetDecl, FnDecl, Block, IfStmt, WhileStmt, ReturnStmt, PrintStmt,
    CallExpr, Program, FnObject,
)

# ---------------------------------------------------------------------------
# Extended Opcodes
# ---------------------------------------------------------------------------
class Op(IntEnum):
    """All opcodes -- base + effect extensions."""
    # Base ops (must match C010 values)
    CONST = 1
    POP = 2
    DUP = 3
    ADD = 4
    SUB = 5
    MUL = 6
    DIV = 7
    MOD = 8
    NEG = 9
    EQ = 10
    NE = 11
    LT = 12
    GT = 13
    LE = 14
    GE = 15
    NOT = 16
    AND = 17
    OR = 18
    LOAD = 19
    STORE = 20
    JUMP = 21
    JUMP_IF_FALSE = 22
    JUMP_IF_TRUE = 23
    CALL = 24
    RETURN = 25
    PRINT = 26
    HALT = 27
    # Effect extensions
    PERFORM = 28        # operand: constant index (effect_name), operand2: constant index (op_name), pops arg_count args
    INSTALL_HANDLER = 29  # operand: handler object index in constants
    REMOVE_HANDLER = 30   # no operand, pops topmost handler
    RESUME = 31          # pops value, resumes captured continuation

# ---------------------------------------------------------------------------
# Extended AST nodes for effects
# ---------------------------------------------------------------------------
@dataclass
class EffectDecl:
    """Declare a custom effect with operations."""
    name: str
    operations: list  # [(op_name, param_names)]
    line: int = 0

@dataclass
class PerformExpr:
    """Perform an effect operation: perform Effect.op(args)"""
    effect: str
    operation: str
    args: list
    line: int = 0

@dataclass
class HandleWith:
    """Handle effects in a body block with handler clauses.
    handle { body } with { Effect.op(params) -> handler_body }
    """
    body: Any  # Block
    handlers: list  # [HandlerClause]
    line: int = 0

@dataclass
class HandlerClause:
    """A single handler clause: Effect.op(params) -> body"""
    effect: str
    operation: str
    params: list  # parameter names (not including implicit 'resume')
    body: Any  # Block
    line: int = 0

@dataclass
class ResumeExpr:
    """Resume from a handler with a value: resume(value)"""
    value: Any
    line: int = 0

# ---------------------------------------------------------------------------
# Handler object (stored in constants pool)
# ---------------------------------------------------------------------------
@dataclass
class HandlerObject:
    """Runtime handler -- compiled handler clauses."""
    clauses: dict  # {(effect, op): HandlerClauseObj}

@dataclass
class HandlerClauseObj:
    """Compiled handler clause with its own chunk."""
    effect: str
    operation: str
    param_names: list  # parameter names for args
    chunk: Any  # Chunk -- compiled handler body
    arity: int  # number of params (not including resume)

# ---------------------------------------------------------------------------
# Continuation (captured when perform hits a handler)
# ---------------------------------------------------------------------------
@dataclass
class Continuation:
    """Captured continuation for resume."""
    chunk: Any
    ip: int
    stack: list
    env: dict
    call_stack: list
    handler_stack: list  # handlers above the handling one

# ---------------------------------------------------------------------------
# Extended Token Types
# ---------------------------------------------------------------------------
class ExtTokenType(IntEnum):
    """Extra token types for effect syntax."""
    # Copy base types
    INT = TokenType.INT
    FLOAT = TokenType.FLOAT
    STRING = TokenType.STRING
    TRUE = TokenType.TRUE
    FALSE = TokenType.FALSE
    IDENT = TokenType.IDENT
    PLUS = TokenType.PLUS
    MINUS = TokenType.MINUS
    STAR = TokenType.STAR
    SLASH = TokenType.SLASH
    PERCENT = TokenType.PERCENT
    EQ = TokenType.EQ
    NE = TokenType.NE
    LT = TokenType.LT
    GT = TokenType.GT
    LE = TokenType.LE
    GE = TokenType.GE
    ASSIGN = TokenType.ASSIGN
    NOT = TokenType.NOT
    AND = TokenType.AND
    OR = TokenType.OR
    LPAREN = TokenType.LPAREN
    RPAREN = TokenType.RPAREN
    LBRACE = TokenType.LBRACE
    RBRACE = TokenType.RBRACE
    COMMA = TokenType.COMMA
    SEMICOLON = TokenType.SEMICOLON
    LET = TokenType.LET
    IF = TokenType.IF
    ELSE = TokenType.ELSE
    WHILE = TokenType.WHILE
    FN = TokenType.FN
    RETURN = TokenType.RETURN
    PRINT = TokenType.PRINT
    EOF = TokenType.EOF
    # New effect tokens
    EFFECT = 100
    PERFORM = 101
    HANDLE = 102
    WITH = 103
    RESUME = 104
    DOT = 105
    ARROW = 106

EFFECT_KEYWORDS = {
    'effect': ExtTokenType.EFFECT,
    'perform': ExtTokenType.PERFORM,
    'handle': ExtTokenType.HANDLE,
    'with': ExtTokenType.WITH,
    'resume': ExtTokenType.RESUME,
}

# ---------------------------------------------------------------------------
# Extended Lexer
# ---------------------------------------------------------------------------
def lex(source):
    """Tokenize source with effect syntax support."""
    tokens = []
    i = 0
    line = 1

    # Base keywords (from C010)
    base_keywords = {
        'let': TokenType.LET, 'if': TokenType.IF, 'else': TokenType.ELSE,
        'while': TokenType.WHILE, 'fn': TokenType.FN, 'return': TokenType.RETURN,
        'print': TokenType.PRINT, 'true': TokenType.TRUE, 'false': TokenType.FALSE,
        'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
    }

    while i < len(source):
        c = source[i]

        # Whitespace
        if c in ' \t\r':
            i += 1
            continue
        if c == '\n':
            line += 1
            i += 1
            continue

        # Comments
        if c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Numbers
        if c.isdigit():
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
            continue

        # Strings
        if c == '"':
            i += 1
            start = i
            while i < len(source) and source[i] != '"':
                if source[i] == '\n':
                    line += 1
                i += 1
            tokens.append(Token(TokenType.STRING, source[start:i], line))
            i += 1  # skip closing quote
            continue

        # Identifiers / keywords
        if c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            if word in EFFECT_KEYWORDS:
                tokens.append(Token(EFFECT_KEYWORDS[word], word, line))
            elif word in base_keywords:
                tokens.append(Token(base_keywords[word], word, line))
            else:
                tokens.append(Token(TokenType.IDENT, word, line))
            continue

        # Two-char operators
        if i + 1 < len(source):
            two = source[i:i+2]
            if two == '==':
                tokens.append(Token(TokenType.EQ, '==', line)); i += 2; continue
            if two == '!=':
                tokens.append(Token(TokenType.NE, '!=', line)); i += 2; continue
            if two == '<=':
                tokens.append(Token(TokenType.LE, '<=', line)); i += 2; continue
            if two == '>=':
                tokens.append(Token(TokenType.GE, '>=', line)); i += 2; continue
            if two == '->':
                tokens.append(Token(ExtTokenType.ARROW, '->', line)); i += 2; continue

        # Single-char
        singles = {
            '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
            '/': TokenType.SLASH, '%': TokenType.PERCENT, '<': TokenType.LT,
            '>': TokenType.GT, '=': TokenType.ASSIGN, '(': TokenType.LPAREN,
            ')': TokenType.RPAREN, '{': TokenType.LBRACE, '}': TokenType.RBRACE,
            ',': TokenType.COMMA, ';': TokenType.SEMICOLON, '.': ExtTokenType.DOT,
        }
        if c in singles:
            tokens.append(Token(singles[c], c, line))
            i += 1
            continue

        raise SyntaxError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens

# ---------------------------------------------------------------------------
# Extended Parser
# ---------------------------------------------------------------------------
class Parser:
    """Parse tokens into AST with effect syntax support."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, ttype):
        t = self.advance()
        if t.type != ttype:
            raise SyntaxError(f"Expected {ttype}, got {t.type} ({t.value!r}) at line {t.line}")
        return t

    def match(self, ttype):
        if self.peek().type == ttype:
            return self.advance()
        return None

    def expect_ident_like(self):
        """Expect an identifier or a keyword usable as identifier (e.g. 'print' as effect op name)."""
        t = self.advance()
        # Accept IDENT or any keyword token that has a string value
        if t.type == TokenType.IDENT or isinstance(t.value, str) and t.value.isalpha():
            return t
        raise SyntaxError(f"Expected identifier, got {t.type} ({t.value!r}) at line {t.line}")

    def parse(self):
        stmts = []
        while self.peek().type != TokenType.EOF:
            stmts.append(self.parse_stmt())
        return Program(stmts)

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
        if t.type == ExtTokenType.EFFECT:
            return self.parse_effect_decl()
        if t.type == ExtTokenType.HANDLE:
            return self.parse_handle()

        # Expression statement
        expr = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return expr

    def parse_let(self):
        self.advance()  # 'let'
        line = self.peek().line
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        value = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return LetDecl(name=name, value=value, line=line)

    def parse_fn(self):
        self.advance()  # 'fn'
        line = self.peek().line
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return FnDecl(name=name, params=params, body=body, line=line)

    def parse_if(self):
        self.advance()  # 'if'
        line = self.peek().line
        self.expect(TokenType.LPAREN)
        cond = self.parse_expr()
        self.expect(TokenType.RPAREN)
        then_body = self.parse_block()
        else_body = None
        if self.match(TokenType.ELSE):
            if self.peek().type == TokenType.IF:
                else_body = self.parse_if()
            else:
                else_body = self.parse_block()
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body, line=line)

    def parse_while(self):
        self.advance()  # 'while'
        line = self.peek().line
        self.expect(TokenType.LPAREN)
        cond = self.parse_expr()
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return WhileStmt(cond=cond, body=body, line=line)

    def parse_return(self):
        t = self.advance()  # 'return'
        if self.peek().type == TokenType.SEMICOLON:
            self.advance()
            return ReturnStmt(value=None, line=t.line)
        value = self.parse_expr()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(value=value, line=t.line)

    def parse_print(self):
        t = self.advance()  # 'print'
        self.expect(TokenType.LPAREN)
        value = self.parse_expr()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(value=value, line=t.line)

    def parse_effect_decl(self):
        self.advance()  # 'effect'
        line = self.peek().line
        name = self.expect_ident_like().value
        self.expect(TokenType.LBRACE)
        operations = []
        while self.peek().type != TokenType.RBRACE:
            op_name = self.expect_ident_like().value
            self.expect(TokenType.LPAREN)
            params = []
            if self.peek().type != TokenType.RPAREN:
                params.append(self.expect_ident_like().value)
                while self.match(TokenType.COMMA):
                    params.append(self.expect_ident_like().value)
            self.expect(TokenType.RPAREN)
            self.expect(TokenType.SEMICOLON)
            operations.append((op_name, params))
        self.expect(TokenType.RBRACE)
        return EffectDecl(name=name, operations=operations, line=line)

    def parse_handle(self):
        t = self.advance()  # 'handle'
        body = self.parse_block()
        self.expect(ExtTokenType.WITH)
        self.expect(TokenType.LBRACE)
        handlers = []
        while self.peek().type != TokenType.RBRACE:
            handlers.append(self.parse_handler_clause())
        self.expect(TokenType.RBRACE)
        return HandleWith(body=body, handlers=handlers, line=t.line)

    def parse_handler_clause(self):
        """Parse: Effect.op(params) -> { body }"""
        line = self.peek().line
        effect = self.expect_ident_like().value
        self.expect(ExtTokenType.DOT)
        operation = self.expect_ident_like().value
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.expect_ident_like().value)
            while self.match(TokenType.COMMA):
                params.append(self.expect_ident_like().value)
        self.expect(TokenType.RPAREN)
        self.expect(ExtTokenType.ARROW)
        body = self.parse_block()
        return HandlerClause(effect=effect, operation=operation, params=params,
                           body=body, line=line)

    def parse_block(self):
        self.expect(TokenType.LBRACE)
        line = self.peek().line
        stmts = []
        while self.peek().type != TokenType.RBRACE:
            stmts.append(self.parse_stmt())
        self.expect(TokenType.RBRACE)
        return Block(stmts=stmts, line=line)

    def parse_expr(self):
        return self.parse_assignment()

    def parse_assignment(self):
        expr = self.parse_or()
        if isinstance(expr, Var) and self.peek().type == TokenType.ASSIGN:
            self.advance()
            value = self.parse_expr()
            return Assign(name=expr.name, value=value, line=expr.line)
        return expr

    def parse_or(self):
        left = self.parse_and()
        while self.peek().type == TokenType.OR:
            self.advance()
            right = self.parse_and()
            left = BinOp(op='or', left=left, right=right, line=left.line)
        return left

    def parse_and(self):
        left = self.parse_equality()
        while self.peek().type == TokenType.AND:
            self.advance()
            right = self.parse_equality()
            left = BinOp(op='and', left=left, right=right, line=left.line)
        return left

    def parse_equality(self):
        left = self.parse_comparison()
        while self.peek().type in (TokenType.EQ, TokenType.NE):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinOp(op=op, left=left, right=right, line=left.line)
        return left

    def parse_comparison(self):
        left = self.parse_addition()
        while self.peek().type in (TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op = self.advance().value
            right = self.parse_addition()
            left = BinOp(op=op, left=left, right=right, line=left.line)
        return left

    def parse_addition(self):
        left = self.parse_multiplication()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplication()
            left = BinOp(op=op, left=left, right=right, line=left.line)
        return left

    def parse_multiplication(self):
        left = self.parse_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = BinOp(op=op, left=left, right=right, line=left.line)
        return left

    def parse_unary(self):
        if self.peek().type == TokenType.MINUS:
            t = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op='-', operand=operand, line=t.line)
        if self.peek().type == TokenType.NOT:
            t = self.advance()
            operand = self.parse_unary()
            return UnaryOp(op='not', operand=operand, line=t.line)
        return self.parse_call()

    def parse_call(self):
        expr = self.parse_primary()
        if isinstance(expr, Var) and self.peek().type == TokenType.LPAREN:
            self.advance()
            args = []
            if self.peek().type != TokenType.RPAREN:
                args.append(self.parse_expr())
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expr())
            self.expect(TokenType.RPAREN)
            return CallExpr(callee=expr.name, args=args, line=expr.line)
        return expr

    def parse_primary(self):
        t = self.peek()

        if t.type == TokenType.INT:
            self.advance()
            return IntLit(value=t.value, line=t.line)
        if t.type == TokenType.FLOAT:
            self.advance()
            return FloatLit(value=t.value, line=t.line)
        if t.type == TokenType.STRING:
            self.advance()
            return StringLit(value=t.value, line=t.line)
        if t.type == TokenType.TRUE:
            self.advance()
            return BoolLit(value=True, line=t.line)
        if t.type == TokenType.FALSE:
            self.advance()
            return BoolLit(value=False, line=t.line)
        if t.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expr()
            self.expect(TokenType.RPAREN)
            return expr

        # perform Effect.op(args)
        if t.type == ExtTokenType.PERFORM:
            return self.parse_perform()

        # resume(value)
        if t.type == ExtTokenType.RESUME:
            return self.parse_resume()

        if t.type == TokenType.IDENT:
            self.advance()
            return Var(name=t.value, line=t.line)

        raise SyntaxError(f"Unexpected token {t.type} ({t.value!r}) at line {t.line}")

    def parse_perform(self):
        t = self.advance()  # 'perform'
        effect = self.expect_ident_like().value
        self.expect(ExtTokenType.DOT)
        operation = self.expect_ident_like().value
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.parse_expr())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expr())
        self.expect(TokenType.RPAREN)
        return PerformExpr(effect=effect, operation=operation, args=args, line=t.line)

    def parse_resume(self):
        t = self.advance()  # 'resume'
        self.expect(TokenType.LPAREN)
        value = self.parse_expr()
        self.expect(TokenType.RPAREN)
        return ResumeExpr(value=value, line=t.line)


# ---------------------------------------------------------------------------
# Compiler -- extends C010 compiler with effect opcodes
# ---------------------------------------------------------------------------
class Compiler:
    """Compile AST to bytecode with effect support."""

    def __init__(self):
        self.chunk = Chunk()
        self.functions = {}
        self.effects = {}  # {effect_name: [(op_name, param_names)]}

    def compile(self, program):
        for stmt in program.stmts:
            self.compile_node(stmt)
        self.chunk.emit(Op.HALT, line=0)
        return self.chunk

    def compile_node(self, node):
        method = f'compile_{type(node).__name__}'
        compiler = getattr(self, method, None)
        if compiler is None:
            raise CompileError(f"Cannot compile {type(node).__name__}")
        return compiler(node)

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

    def compile_Assign(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, idx, node.line)

    def compile_UnaryOp(self, node):
        self.compile_node(node.operand)
        if node.op == '-':
            self.chunk.emit(Op.NEG, line=node.line)
        elif node.op == 'not':
            self.chunk.emit(Op.NOT, line=node.line)

    def compile_BinOp(self, node):
        # Short-circuit for and/or
        if node.op == 'and':
            self.compile_node(node.left)
            jump_addr = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
            self.chunk.emit(Op.POP, line=node.line)
            self.compile_node(node.right)
            self.chunk.patch(jump_addr + 1, len(self.chunk.code))
            return
        if node.op == 'or':
            self.compile_node(node.left)
            jump_addr = self.chunk.emit(Op.JUMP_IF_TRUE, 0, node.line)
            self.chunk.emit(Op.POP, line=node.line)
            self.compile_node(node.right)
            self.chunk.patch(jump_addr + 1, len(self.chunk.code))
            return

        self.compile_node(node.left)
        self.compile_node(node.right)
        op_map = {
            '+': Op.ADD, '-': Op.SUB, '*': Op.MUL, '/': Op.DIV, '%': Op.MOD,
            '==': Op.EQ, '!=': Op.NE, '<': Op.LT, '>': Op.GT,
            '<=': Op.LE, '>=': Op.GE,
        }
        self.chunk.emit(op_map[node.op], line=node.line)

    def compile_LetDecl(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, idx, node.line)

    def compile_Block(self, node):
        for stmt in node.stmts:
            self.compile_node(stmt)

    def compile_IfStmt(self, node):
        self.compile_node(node.cond)
        false_jump = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.then_body)
        if node.else_body:
            end_jump = self.chunk.emit(Op.JUMP, 0, node.line)
            self.chunk.patch(false_jump + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)
            self.compile_node(node.else_body)
            self.chunk.patch(end_jump + 1, len(self.chunk.code))
        else:
            self.chunk.patch(false_jump + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)

    def compile_WhileStmt(self, node):
        loop_start = len(self.chunk.code)
        self.compile_node(node.cond)
        exit_jump = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.body)
        self.chunk.emit(Op.JUMP, loop_start, node.line)
        self.chunk.patch(exit_jump + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=node.line)

    def compile_FnDecl(self, node):
        # Compile function body into separate chunk
        fn_compiler = Compiler()
        fn_compiler.effects = self.effects
        for p in node.params:
            fn_compiler.chunk.add_name(p)
        fn_compiler.compile_node(node.body)
        # Implicit return None
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx, node.line)
        fn_compiler.chunk.emit(Op.RETURN, line=node.line)

        fn_obj = FnObject(name=node.name, arity=len(node.params), chunk=fn_compiler.chunk)
        cidx = self.chunk.add_constant(fn_obj)
        self.chunk.emit(Op.CONST, cidx, node.line)
        nidx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, nidx, node.line)
        self.functions[node.name] = fn_obj

    def compile_CallExpr(self, node):
        idx = self.chunk.add_name(node.callee)
        self.chunk.emit(Op.LOAD, idx, node.line)
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
        """Register effect -- no bytecode emitted."""
        self.effects[node.name] = node.operations

    def compile_PerformExpr(self, node):
        """Compile perform Effect.op(args).
        Push args, then emit PERFORM with effect/op indices.
        """
        for arg in node.args:
            self.compile_node(arg)
        effect_idx = self.chunk.add_constant(node.effect)
        op_idx = self.chunk.add_constant(node.operation)
        arg_count_idx = self.chunk.add_constant(len(node.args))
        self.chunk.emit(Op.PERFORM, effect_idx, node.line)
        # Encode op name and arg count as additional operands
        self.chunk.code.append(op_idx)
        self.chunk.code.append(arg_count_idx)

    def compile_HandleWith(self, node):
        """Compile handle { body } with { handlers }.
        1. Compile handler clauses into HandlerObject
        2. INSTALL_HANDLER (push handler)
        3. Compile body
        4. REMOVE_HANDLER (pop handler)
        """
        # Compile each handler clause
        clauses = {}
        for hc in node.handlers:
            hc_compiler = Compiler()
            hc_compiler.effects = self.effects
            # Add params as names (resume is implicit)
            for p in hc.params:
                hc_compiler.chunk.add_name(p)
            hc_compiler.chunk.add_name('resume')  # always available
            hc_compiler.compile_node(hc.body)
            # Handler body returns its last value implicitly via RETURN
            ret_idx = hc_compiler.chunk.add_constant(None)
            hc_compiler.chunk.emit(Op.CONST, ret_idx, hc.line)
            hc_compiler.chunk.emit(Op.RETURN, line=hc.line)

            clause_obj = HandlerClauseObj(
                effect=hc.effect, operation=hc.operation,
                param_names=hc.params, chunk=hc_compiler.chunk,
                arity=len(hc.params),
            )
            clauses[(hc.effect, hc.operation)] = clause_obj

        handler_obj = HandlerObject(clauses=clauses)
        hidx = self.chunk.add_constant(handler_obj)
        self.chunk.emit(Op.INSTALL_HANDLER, hidx, node.line)
        self.compile_node(node.body)
        self.chunk.emit(Op.REMOVE_HANDLER, line=node.line)

    def compile_ResumeExpr(self, node):
        """Compile resume(value) -- push value, emit RESUME."""
        self.compile_node(node.value)
        self.chunk.emit(Op.RESUME, line=node.line)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------
class CompileError(Exception):
    pass

class EffectError(Exception):
    """Unhandled effect at runtime."""
    pass

class ResumeError(Exception):
    """Resume called outside handler context."""
    pass

# ---------------------------------------------------------------------------
# Extended VM with effect handling
# ---------------------------------------------------------------------------
@dataclass
class HandlerFrame:
    """Active handler on the handler stack."""
    handler: HandlerObject
    # Saved state for unwinding when effect is performed
    chunk: Any
    ip: int
    stack_depth: int
    env: dict
    call_stack_depth: int

@dataclass
class CallFrame:
    """Saved call frame."""
    chunk: Any
    ip: int
    base_env: dict

class EffectVM:
    """Stack VM with algebraic effect handling."""

    def __init__(self, chunk, trace=False):
        self.chunk = chunk
        self.stack = []
        self.env = {}
        self.call_stack = []  # [CallFrame]
        self.handler_stack = []  # [HandlerFrame]
        self.output = []
        self.ip = 0
        self.current_chunk = chunk
        self.step_count = 0
        self.max_steps = 100000
        self.trace = trace
        self.effects = {}  # registered effects
        self._resume_continuations = []  # stack of Continuation for resume

    def push(self, value):
        self.stack.append(value)

    def pop(self):
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack.pop()

    def peek(self):
        if not self.stack:
            raise RuntimeError("Stack underflow")
        return self.stack[-1]

    def run(self):
        """Execute bytecode."""
        while True:
            if self.step_count >= self.max_steps:
                raise RuntimeError("Execution limit exceeded")
            self.step_count += 1

            if self.ip >= len(self.current_chunk.code):
                break

            op = self.current_chunk.code[self.ip]
            self.ip += 1

            if self.trace:
                print(f"  [{self.ip-1}] {Op(op).name} stack={self.stack}")

            if op == Op.CONST:
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
                if isinstance(a, int) and isinstance(b, int):
                    self.push(a // b if b != 0 else 0)
                else:
                    self.push(a / b if b != 0 else 0.0)
            elif op == Op.MOD:
                b, a = self.pop(), self.pop()
                self.push(a % b if b != 0 else 0)

            elif op == Op.NEG:
                self.push(-self.pop())
            elif op == Op.NOT:
                self.push(not self.pop())

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
                    raise RuntimeError(f"Undefined variable: {name}")
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
                fn = self.pop()

                if not isinstance(fn, FnObject):
                    raise RuntimeError(f"Cannot call {fn}")
                if fn.arity != arg_count:
                    raise RuntimeError(f"{fn.name} expects {fn.arity} args, got {arg_count}")

                # Save current frame
                frame = CallFrame(
                    chunk=self.current_chunk,
                    ip=self.ip,
                    base_env=dict(self.env),
                )
                self.call_stack.append(frame)

                # Set up new frame
                self.current_chunk = fn.chunk
                self.ip = 0
                # Bind parameters
                for i, name in enumerate(fn.chunk.names[:fn.arity]):
                    self.env[name] = args[i]

            elif op == Op.RETURN:
                ret_val = self.pop()
                if not self.call_stack:
                    # Top level return
                    self.push(ret_val)
                    break
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                self.env = frame.base_env
                self.push(ret_val)

            elif op == Op.PRINT:
                val = self.pop()
                s = str(val)
                self.output.append(s)

            elif op == Op.HALT:
                break

            # --- Effect opcodes ---
            elif op == Op.PERFORM:
                effect_idx = self.current_chunk.code[self.ip]
                self.ip += 1
                op_idx = self.current_chunk.code[self.ip]
                self.ip += 1
                arg_count_idx = self.current_chunk.code[self.ip]
                self.ip += 1

                effect_name = self.current_chunk.constants[effect_idx]
                op_name = self.current_chunk.constants[op_idx]
                arg_count = self.current_chunk.constants[arg_count_idx]

                # Pop arguments
                args = []
                for _ in range(arg_count):
                    args.insert(0, self.pop())

                # Find handler
                handler_frame = self._find_handler(effect_name, op_name)
                if handler_frame is None:
                    raise EffectError(f"Unhandled effect: {effect_name}.{op_name}")

                self._invoke_handler(handler_frame, effect_name, op_name, args)

            elif op == Op.INSTALL_HANDLER:
                idx = self.current_chunk.code[self.ip]
                self.ip += 1
                handler_obj = self.current_chunk.constants[idx]
                frame = HandlerFrame(
                    handler=handler_obj,
                    chunk=self.current_chunk,
                    ip=self.ip,
                    stack_depth=len(self.stack),
                    env=dict(self.env),
                    call_stack_depth=len(self.call_stack),
                )
                self.handler_stack.append(frame)

            elif op == Op.REMOVE_HANDLER:
                if self.handler_stack:
                    self.handler_stack.pop()

            elif op == Op.RESUME:
                value = self.pop()
                if not self._resume_continuations:
                    raise ResumeError("resume called outside handler context")
                cont = self._resume_continuations.pop()
                self._restore_continuation(cont, value)

            else:
                raise RuntimeError(f"Unknown opcode: {op}")

        return self.stack[-1] if self.stack else None

    def _find_handler(self, effect_name, op_name):
        """Search handler stack (top to bottom) for matching handler."""
        for i in range(len(self.handler_stack) - 1, -1, -1):
            frame = self.handler_stack[i]
            if (effect_name, op_name) in frame.handler.clauses:
                return frame
        return None

    def _invoke_handler(self, handler_frame, effect_name, op_name, args):
        """Invoke a handler clause, capturing continuation for resume."""
        clause = handler_frame.handler.clauses[(effect_name, op_name)]

        # Capture continuation (everything needed to resume where perform was called)
        # Find handler's position in the stack
        handler_idx = self.handler_stack.index(handler_frame)

        # Save handlers above the matched one (they'll be restored on resume)
        handlers_above = self.handler_stack[handler_idx + 1:]

        # Include the matched handler frame so it can be re-installed on resume
        cont = Continuation(
            chunk=self.current_chunk,
            ip=self.ip,
            stack=list(self.stack),
            env=dict(self.env),
            call_stack=list(self.call_stack),
            handler_stack=[handler_frame] + list(handlers_above),
        )
        self._resume_continuations.append(cont)

        # Unwind to handler's scope (remove matched handler and above)
        self.handler_stack = self.handler_stack[:handler_idx]
        self.call_stack = self.call_stack[:handler_frame.call_stack_depth]
        self.stack = self.stack[:handler_frame.stack_depth]
        self.env = dict(handler_frame.env)

        # Set up handler clause execution
        self.current_chunk = clause.chunk
        self.ip = 0

        # Bind parameters
        for i, name in enumerate(clause.param_names):
            if i < len(args):
                self.env[name] = args[i]

        # Bind resume as a sentinel -- actual resume happens via RESUME opcode
        self.env['resume'] = '<resume_continuation>'

    def _restore_continuation(self, cont, value):
        """Restore a captured continuation with a value."""
        self.current_chunk = cont.chunk
        self.ip = cont.ip
        self.stack = cont.stack
        self.env = cont.env
        self.call_stack = cont.call_stack
        # Restore handlers that were above the handler
        self.handler_stack.extend(cont.handler_stack)
        # Push the resume value (it's the result of the perform expression)
        self.push(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse(source):
    """Parse source code into AST."""
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()

def compile_program(ast):
    """Compile AST to bytecode chunk."""
    compiler = Compiler()
    return compiler.compile(ast)

def run(source, trace=False):
    """Parse, compile, and run source code. Returns (result, output)."""
    ast = parse(source)
    chunk = compile_program(ast)
    vm = EffectVM(chunk, trace=trace)
    result = vm.run()
    return result, vm.output

def run_checked(source, trace=False):
    """Parse, type-check effects, compile, and run.
    Returns (result, output, errors) where errors is from effect checking.
    """
    # Try to import C032 for type checking
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C032_effect_system'))
        from effect_system import check_source, format_errors
        errors, checker = check_source(source)
        if errors:
            return None, [], errors
    except ImportError:
        errors = []

    ast = parse(source)
    chunk = compile_program(ast)
    vm = EffectVM(chunk, trace=trace)
    result = vm.run()
    return result, vm.output, errors
