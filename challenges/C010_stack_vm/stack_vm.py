"""
Stack-based Virtual Machine with Bytecode Compiler
Challenge C010 -- AgentZero Session 011

Architecture:
  Source code -> Lexer -> Parser -> AST -> Compiler -> Bytecode -> VM -> Result

Language features:
  - Integers, floats, booleans, strings
  - Arithmetic: + - * / %
  - Comparison: == != < > <= >=
  - Logic: and or not
  - Variables: let x = expr
  - Assignment: x = expr
  - Control flow: if/else, while
  - Functions: fn name(args) { body }
  - Print: print(expr)
  - Return values from functions
  - Block scoping
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Instruction Set
# ============================================================

class Op(IntEnum):
    # Stack
    CONST = auto()       # push constant
    POP = auto()         # discard top
    DUP = auto()         # duplicate top

    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    NEG = auto()         # unary negate

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
    LOAD = auto()        # push variable value (by name index)
    STORE = auto()       # pop and store to variable (by name index)

    # Control flow
    JUMP = auto()        # unconditional jump
    JUMP_IF_FALSE = auto()  # conditional jump
    JUMP_IF_TRUE = auto()

    # Functions
    CALL = auto()        # call function (arg count follows)
    RETURN = auto()      # return from function

    # I/O
    PRINT = auto()

    # Halt
    HALT = auto()


# ============================================================
# Bytecode
# ============================================================

@dataclass
class Chunk:
    """A compiled unit of bytecode."""
    code: list = field(default_factory=list)       # list of Op + operands
    constants: list = field(default_factory=list)   # constant pool
    names: list = field(default_factory=list)       # variable name pool
    lines: list = field(default_factory=list)       # source line for each op (for errors)

    def emit(self, op, operand=None, line=0):
        addr = len(self.code)
        self.code.append(op)
        self.lines.append(line)
        if operand is not None:
            self.code.append(operand)
            self.lines.append(line)
        return addr

    def add_constant(self, value):
        # Must match both value AND type to avoid Python's True==1, False==0
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
        """Patch an operand at a given address."""
        self.code[addr] = value


# ============================================================
# Lexer
# ============================================================

class TokenType(IntEnum):
    # Literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    IDENT = auto()

    # Operators
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

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()

    # Keywords
    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FN = auto()
    RETURN = auto()
    PRINT = auto()

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
}


class LexError(Exception):
    pass


def lex(source: str) -> list:
    tokens = []
    i = 0
    line = 1

    while i < len(source):
        c = source[i]

        # Whitespace
        if c == '\n':
            line += 1
            i += 1
        elif c in ' \t\r':
            i += 1

        # Comments
        elif c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1

        # Numbers
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

        # Strings
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

        # Identifiers and keywords
        elif c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line))

        # Two-char operators
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

        # Single-char operators/delimiters
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
        else:
            raise LexError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens


# ============================================================
# AST
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


# ============================================================
# Parser (recursive descent)
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
        return IfStmt(cond, then_body, else_body, tok.line)

    def while_stmt(self):
        tok = self.advance()  # while
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return WhileStmt(cond, body, tok.line)

    def return_stmt(self):
        tok = self.advance()  # return
        value = None
        if self.peek().type != TokenType.SEMICOLON:
            value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return ReturnStmt(value, tok.line)

    def print_stmt(self):
        tok = self.advance()  # print
        self.expect(TokenType.LPAREN)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return PrintStmt(value, tok.line)

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

    # Precedence climbing
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

        if tok.type == TokenType.IDENT:
            self.advance()
            # Check for function call
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

        raise ParseError(f"Unexpected token {tok.type.name} '{tok.value}' at line {tok.line}")


# ============================================================
# Compiler (AST -> Bytecode)
# ============================================================

class CompileError(Exception):
    pass


@dataclass
class FnObject:
    """Compiled function."""
    name: str
    arity: int
    chunk: Chunk


class Compiler:
    def __init__(self):
        self.chunk = Chunk()
        self.functions = {}  # name -> FnObject

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
        # Short-circuit for and/or
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
        # Jump to else/end if false
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)  # pop condition
        self.compile_node(node.then_body)
        if node.else_body:
            jump_end = self.chunk.emit(Op.JUMP, 0, node.line)
            self.chunk.patch(jump_false + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)  # pop condition
            self.compile_node(node.else_body)
            self.chunk.patch(jump_end + 1, len(self.chunk.code))
        else:
            self.chunk.patch(jump_false + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)  # pop condition

    def compile_WhileStmt(self, node):
        loop_start = len(self.chunk.code)
        self.compile_node(node.cond)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)  # pop condition
        self.compile_node(node.body)
        self.chunk.emit(Op.JUMP, loop_start, node.line)
        self.chunk.patch(jump_false + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=node.line)  # pop condition

    def compile_FnDecl(self, node):
        # Compile function to its own chunk
        fn_compiler = Compiler()
        # Store params as local variables
        for param in node.params:
            fn_compiler.chunk.add_name(param)
        fn_compiler.compile_node(node.body)
        # Implicit return None
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx)
        fn_compiler.chunk.emit(Op.RETURN)

        fn_obj = FnObject(node.name, len(node.params), fn_compiler.chunk)
        fn_obj_functions = fn_compiler.functions
        self.functions[node.name] = fn_obj
        # Also store sub-functions
        for k, v in fn_obj_functions.items():
            self.functions[k] = v

        # Store function reference in main chunk
        fn_idx = self.chunk.add_constant(fn_obj)
        name_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.STORE, name_idx, node.line)

    def compile_CallExpr(self, node):
        # Push function
        fn_idx = self.chunk.add_name(node.callee)
        self.chunk.emit(Op.LOAD, fn_idx, node.line)
        # Push args
        for arg in node.args:
            self.compile_node(arg)
        # Call
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


# ============================================================
# Virtual Machine
# ============================================================

class VMError(Exception):
    pass


@dataclass
class CallFrame:
    """A function call frame."""
    chunk: Chunk
    ip: int
    base_env: dict  # snapshot of variables at call time


class VM:
    def __init__(self, chunk: Chunk, trace=False):
        self.chunk = chunk
        self.stack = []
        self.env = {}  # global variables
        self.call_stack = []  # CallFrame stack
        self.output = []  # captured print output
        self.trace = trace
        self.ip = 0
        self.current_chunk = chunk
        self.step_count = 0
        self.max_steps = 100000  # safety limit

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

            # Arithmetic
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

            # Comparison
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

            # Logic
            elif op == Op.NOT:
                self.push(not self.pop())
            elif op == Op.AND:
                b, a = self.pop(), self.pop()
                self.push(a and b)
            elif op == Op.OR:
                b, a = self.pop(), self.pop()
                self.push(a or b)

            # Variables
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

            # Control flow
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

            # Functions
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

                # Save current state
                frame = CallFrame(self.current_chunk, self.ip, dict(self.env))
                self.call_stack.append(frame)

                # Set up function execution
                self.current_chunk = fn_obj.chunk
                self.ip = 0
                # Bind parameters
                for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                    self.env[param_name] = args[i]

            elif op == Op.RETURN:
                return_val = self.pop()
                if not self.call_stack:
                    # Return from top-level
                    self.push(return_val)
                    break
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                self.env = frame.base_env
                self.push(return_val)

            # I/O
            elif op == Op.PRINT:
                value = self.pop()
                text = str(value) if value is not None else "None"
                if isinstance(value, bool):
                    text = "true" if value else "false"
                self.output.append(text)

            else:
                raise VMError(f"Unknown opcode: {op}")

        return self.stack[-1] if self.stack else None

    def _trace_op(self, op):
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"
        print(f"  [{self.ip-1:04d}] {name:20s} stack={self.stack[-5:]}")


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
    vm = VM(chunk, trace=trace)
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
        op = chunk.code[i]
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"

        if op in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
                  Op.JUMP_IF_TRUE, Op.CALL):
            operand = chunk.code[i + 1]
            if op == Op.CONST:
                val = chunk.constants[operand]
                if isinstance(val, FnObject):
                    lines.append(f"{i:04d}  {name:20s} {operand} (fn:{val.name})")
                else:
                    lines.append(f"{i:04d}  {name:20s} {operand} ({val!r})")
            elif op in (Op.LOAD, Op.STORE):
                nm = chunk.names[operand]
                lines.append(f"{i:04d}  {name:20s} {operand} ({nm})")
            else:
                lines.append(f"{i:04d}  {name:20s} {operand}")
            i += 2
        else:
            lines.append(f"{i:04d}  {name}")
            i += 1
    return '\n'.join(lines)
