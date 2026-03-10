"""
Self-Hosting Compiler -- Challenge C018 -- AgentZero Session 019

A compiler for a subset of the VM language, written IN the VM language itself.
Extends C010's stack VM with arrays and string builtins to make self-hosting possible.

Architecture:
  1. Extended VM (C010 + arrays + string builtins)
  2. Host compiler (Python) for the full language
  3. Self-compiler (MiniLang source) for a subset: ints, vars, arithmetic,
     comparison, if/else, while, print
  4. Bootstrap verification: both compilers produce identical bytecode

The subset the self-compiler COMPILES:
  - Integer literals
  - Variables (let, assign)
  - Arithmetic: + - * / %
  - Comparison: == != < > <= >=
  - If/else, while
  - Print
  - No functions, no strings, no booleans, no floats

The self-compiler USES (runs on the full extended VM):
  - All of the above + functions, strings, arrays, builtins
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Any
import copy


# ============================================================
# Instruction Set (extended from C010)
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
    JUMP_IF_FALSE = auto()
    JUMP_IF_TRUE = auto()

    # Functions
    CALL = auto()        # call function (arg count follows)
    RETURN = auto()      # return from function

    # I/O
    PRINT = auto()

    # Halt
    HALT = auto()

    # Arrays (new)
    ARRAY_NEW = auto()   # push empty array
    ARRAY_GET = auto()   # pop idx, pop arr, push arr[idx]
    ARRAY_SET = auto()   # pop val, pop idx, pop arr; arr[idx]=val

    # Builtins (new)
    BUILTIN = auto()     # operand = builtin ID, arg count follows


# ============================================================
# Builtin Functions
# ============================================================

class Builtin(IntEnum):
    LEN = auto()
    PUSH = auto()
    CHAR_AT = auto()
    CHAR_CODE = auto()
    FROM_CODE = auto()
    SUBSTR = auto()
    TO_STR = auto()
    TO_INT = auto()
    TYPE_OF = auto()

BUILTIN_NAMES = {
    'len': (Builtin.LEN, 1),
    'push': (Builtin.PUSH, 2),
    'char_at': (Builtin.CHAR_AT, 2),
    'char_code': (Builtin.CHAR_CODE, 1),
    'from_code': (Builtin.FROM_CODE, 1),
    'substr': (Builtin.SUBSTR, 3),
    'to_str': (Builtin.TO_STR, 1),
    'to_int': (Builtin.TO_INT, 1),
    'type_of': (Builtin.TYPE_OF, 1),
}


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
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMICOLON = auto()

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
            result = []
            while i < len(source) and source[i] != '"':
                if source[i] == '\\' and i + 1 < len(source):
                    nc = source[i + 1]
                    if nc == 'n':
                        result.append('\n')
                    elif nc == 't':
                        result.append('\t')
                    elif nc == '\\':
                        result.append('\\')
                    elif nc == '"':
                        result.append('"')
                    else:
                        result.append('\\')
                        result.append(nc)
                    i += 2
                else:
                    if source[i] == '\n':
                        line += 1
                    result.append(source[i])
                    i += 1
            if i >= len(source):
                raise LexError(f"Unterminated string at line {line}")
            tokens.append(Token(TokenType.STRING, ''.join(result), line))
            i += 1
        elif c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line))
        elif c == '=' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.EQ, '==', line)); i += 2
        elif c == '!' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.NE, '!=', line)); i += 2
        elif c == '<' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.LE, '<=', line)); i += 2
        elif c == '>' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.GE, '>=', line)); i += 2
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
        elif c == '[':
            tokens.append(Token(TokenType.LBRACKET, '[', line)); i += 1
        elif c == ']':
            tokens.append(Token(TokenType.RBRACKET, ']', line)); i += 1
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
class ArrayLit:
    elements: list
    line: int = 0

@dataclass
class IndexExpr:
    obj: Any
    index: Any
    line: int = 0

@dataclass
class IndexAssign:
    obj: Any
    index: Any
    value: Any
    line: int = 0

@dataclass
class ExprStmt:
    expr: Any
    line: int = 0

@dataclass
class Program:
    stmts: list


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
        return self.statement()

    def fn_decl(self):
        self.advance()
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
        tok = self.advance()
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
        # Assign/IndexAssign already pop via STORE/ARRAY_SET, don't add extra POP
        if isinstance(expr, (Assign, IndexAssign)):
            return expr
        return ExprStmt(expr, expr.line if hasattr(expr, 'line') else 0)

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.or_expr()
        if self.match(TokenType.ASSIGN):
            if isinstance(expr, Var):
                value = self.assignment()
                return Assign(expr.name, value, expr.line)
            if isinstance(expr, IndexExpr):
                value = self.assignment()
                return IndexAssign(expr.obj, expr.index, value, expr.line)
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
        return self.postfix()

    def postfix(self):
        expr = self.primary()
        while self.peek().type == TokenType.LBRACKET:
            self.advance()
            index = self.expression()
            self.expect(TokenType.RBRACKET)
            expr = IndexExpr(expr, index, expr.line)
        return expr

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
            if self.peek().type == TokenType.LPAREN:
                self.advance()
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

        if tok.type == TokenType.LBRACKET:
            self.advance()
            elements = []
            if self.peek().type != TokenType.RBRACKET:
                elements.append(self.expression())
                while self.match(TokenType.COMMA):
                    elements.append(self.expression())
            self.expect(TokenType.RBRACKET)
            return ArrayLit(elements, tok.line)

        raise ParseError(f"Unexpected token {tok.type.name} '{tok.value}' at line {tok.line}")


# ============================================================
# Compiler (AST -> Bytecode)
# ============================================================

class CompileError(Exception):
    pass


@dataclass
class FnObject:
    name: str
    arity: int
    chunk: Chunk


class Compiler:
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
            jump_end = self.chunk.emit(Op.JUMP, 0, node.line)
            self.chunk.patch(jump_false + 1, len(self.chunk.code))
            self.chunk.emit(Op.POP, line=node.line)
            self.chunk.patch(jump_end + 1, len(self.chunk.code))

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

    def compile_CallExpr(self, node):
        # Check for builtins
        if node.callee in BUILTIN_NAMES:
            builtin_id, expected_arity = BUILTIN_NAMES[node.callee]
            for arg in node.args:
                self.compile_node(arg)
            self.chunk.emit(Op.BUILTIN, builtin_id, node.line)
            # Arg count follows
            self.chunk.code.append(len(node.args))
            self.chunk.lines.append(node.line)
            return

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

    def compile_ExprStmt(self, node):
        self.compile_node(node.expr)
        self.chunk.emit(Op.POP, line=node.line)

    def compile_PrintStmt(self, node):
        self.compile_node(node.value)
        self.chunk.emit(Op.PRINT, line=node.line)

    def compile_ArrayLit(self, node):
        self.chunk.emit(Op.ARRAY_NEW, line=node.line)
        for elem in node.elements:
            self.chunk.emit(Op.DUP, line=node.line)
            self.compile_node(elem)
            # Use BUILTIN PUSH
            self.chunk.emit(Op.BUILTIN, Builtin.PUSH, node.line)
            self.chunk.code.append(2)
            self.chunk.lines.append(node.line)
            self.chunk.emit(Op.POP, line=node.line)  # pop returned array (dup still on stack)

    def compile_IndexExpr(self, node):
        self.compile_node(node.obj)
        self.compile_node(node.index)
        self.chunk.emit(Op.ARRAY_GET, line=node.line)

    def compile_IndexAssign(self, node):
        self.compile_node(node.obj)
        self.compile_node(node.index)
        self.compile_node(node.value)
        self.chunk.emit(Op.ARRAY_SET, line=node.line)


# ============================================================
# Virtual Machine
# ============================================================

class VMError(Exception):
    pass


@dataclass
class CallFrame:
    chunk: Chunk
    ip: int
    base_env: dict


class VM:
    def __init__(self, chunk: Chunk, trace=False):
        self.chunk = chunk
        self.stack = []
        self.env = {}
        self.call_stack = []
        self.output = []
        self.trace = trace
        self.ip = 0
        self.current_chunk = chunk
        self.step_count = 0
        self.max_steps = 5000000  # higher limit for self-hosting

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
                frame = CallFrame(self.current_chunk, self.ip, dict(self.env))
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
                self.env = frame.base_env
                self.push(return_val)

            # I/O
            elif op == Op.PRINT:
                value = self.pop()
                text = str(value) if value is not None else "None"
                if isinstance(value, bool):
                    text = "true" if value else "false"
                self.output.append(text)

            # Arrays
            elif op == Op.ARRAY_NEW:
                self.push([])
            elif op == Op.ARRAY_GET:
                idx = self.pop()
                arr = self.pop()
                if not isinstance(arr, list):
                    raise VMError(f"Cannot index non-array: {type(arr)}")
                if not isinstance(idx, int) or idx < 0 or idx >= len(arr):
                    raise VMError(f"Array index out of bounds: {idx} (len={len(arr)})")
                self.push(arr[idx])
            elif op == Op.ARRAY_SET:
                val = self.pop()
                idx = self.pop()
                arr = self.pop()
                if not isinstance(arr, list):
                    raise VMError(f"Cannot index non-array: {type(arr)}")
                if not isinstance(idx, int) or idx < 0 or idx >= len(arr):
                    raise VMError(f"Array index out of bounds: {idx} (len={len(arr)})")
                arr[idx] = val

            # Builtins
            elif op == Op.BUILTIN:
                builtin_id = self.current_chunk.code[self.ip]
                self.ip += 1
                arg_count = self.current_chunk.code[self.ip]
                self.ip += 1
                args = []
                for _ in range(arg_count):
                    args.insert(0, self.pop())
                result = self._exec_builtin(builtin_id, args)
                self.push(result)

            else:
                raise VMError(f"Unknown opcode: {op}")

        return self.stack[-1] if self.stack else None

    def _exec_builtin(self, builtin_id, args):
        if builtin_id == Builtin.LEN:
            val = args[0]
            if isinstance(val, list):
                return len(val)
            if isinstance(val, str):
                return len(val)
            raise VMError(f"len() requires array or string, got {type(val)}")

        elif builtin_id == Builtin.PUSH:
            arr, val = args[0], args[1]
            if not isinstance(arr, list):
                raise VMError(f"push() requires array, got {type(arr)}")
            arr.append(val)
            return arr

        elif builtin_id == Builtin.CHAR_AT:
            s, idx = args[0], args[1]
            if not isinstance(s, str):
                raise VMError(f"char_at() requires string, got {type(s)}")
            if idx < 0 or idx >= len(s):
                raise VMError(f"char_at() index out of bounds: {idx}")
            return s[idx]

        elif builtin_id == Builtin.CHAR_CODE:
            s = args[0]
            if not isinstance(s, str) or len(s) != 1:
                raise VMError(f"char_code() requires single character string")
            return ord(s)

        elif builtin_id == Builtin.FROM_CODE:
            n = args[0]
            if not isinstance(n, int):
                raise VMError(f"from_code() requires int")
            return chr(n)

        elif builtin_id == Builtin.SUBSTR:
            s, start, count = args[0], args[1], args[2]
            if not isinstance(s, str):
                raise VMError(f"substr() requires string")
            return s[start:start + count]

        elif builtin_id == Builtin.TO_STR:
            val = args[0]
            if isinstance(val, bool):
                return "true" if val else "false"
            return str(val)

        elif builtin_id == Builtin.TO_INT:
            val = args[0]
            if isinstance(val, str):
                return int(val)
            if isinstance(val, float):
                return int(val)
            if isinstance(val, int):
                return val
            raise VMError(f"to_int() cannot convert {type(val)}")

        elif builtin_id == Builtin.TYPE_OF:
            val = args[0]
            if isinstance(val, bool):
                return "bool"
            if isinstance(val, int):
                return "int"
            if isinstance(val, float):
                return "float"
            if isinstance(val, str):
                return "string"
            if isinstance(val, list):
                return "array"
            if isinstance(val, FnObject):
                return "fn"
            if val is None:
                return "none"
            return "unknown"

        raise VMError(f"Unknown builtin: {builtin_id}")

    def _trace_op(self, op):
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"
        print(f"  [{self.ip-1:04d}] {name:20s} stack={self.stack[-5:]}")


# ============================================================
# Public API
# ============================================================

def compile_source(source: str) -> tuple:
    tokens = lex(source)
    parser = Parser(tokens)
    ast = parser.parse()
    compiler = Compiler()
    chunk = compiler.compile(ast)
    return chunk, compiler


def execute(source: str, trace=False, env=None) -> dict:
    chunk, compiler = compile_source(source)
    vm = VM(chunk, trace=trace)
    if env:
        vm.env.update(env)
    result = vm.run()
    return {
        'result': result,
        'output': vm.output,
        'env': vm.env,
        'steps': vm.step_count,
    }


def disassemble(chunk: Chunk) -> str:
    lines = []
    i = 0
    while i < len(chunk.code):
        op = chunk.code[i]
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"

        if op == Op.BUILTIN:
            builtin_id = chunk.code[i + 1]
            arg_count = chunk.code[i + 2]
            bname = Builtin(builtin_id).name if builtin_id in Builtin._value2member_map_ else f"??({builtin_id})"
            lines.append(f"{i:04d}  {name:20s} {bname} argc={arg_count}")
            i += 3
        elif op in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
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


# ============================================================
# Self-Compiler Source (MiniLang)
# ============================================================
# This is a compiler for the integer subset of the language,
# written in the language itself. It runs on the extended VM.

SELF_COMPILER_SOURCE = r'''
// ============================================================
// Self-Hosting Compiler for Integer Subset
// Written in MiniLang, runs on the extended VM
// Compiles: ints, vars, arithmetic, comparison, if/else, while, print
// ============================================================

// Op codes -- must match VM's Op enum values exactly
let OP_CONST = 1;
let OP_POP = 2;
let OP_ADD = 4;
let OP_SUB = 5;
let OP_MUL = 6;
let OP_DIV = 7;
let OP_MOD = 8;
let OP_NEG = 9;
let OP_EQ = 10;
let OP_NE = 11;
let OP_LT = 12;
let OP_GT = 13;
let OP_LE = 14;
let OP_GE = 15;
let OP_LOAD = 19;
let OP_STORE = 20;
let OP_JUMP = 21;
let OP_JUMP_IF_FALSE = 22;
let OP_PRINT = 26;
let OP_HALT = 27;

// Token types
let TK_INT = 1;
let TK_IDENT = 2;
let TK_PLUS = 3;
let TK_MINUS = 4;
let TK_STAR = 5;
let TK_SLASH = 6;
let TK_PERCENT = 7;
let TK_EQ = 8;
let TK_NE = 9;
let TK_LT = 10;
let TK_GT = 11;
let TK_LE = 12;
let TK_GE = 13;
let TK_ASSIGN = 14;
let TK_LPAREN = 15;
let TK_RPAREN = 16;
let TK_LBRACE = 17;
let TK_RBRACE = 18;
let TK_SEMICOLON = 19;
let TK_LET = 20;
let TK_IF = 21;
let TK_ELSE = 22;
let TK_WHILE = 23;
let TK_PRINT = 24;
let TK_EOF = 25;

// AST node types
let N_INT = 1;
let N_VAR = 2;
let N_BINOP = 3;
let N_UNARY = 4;
let N_ASSIGN = 5;
let N_LET = 6;
let N_IF = 7;
let N_WHILE = 8;
let N_PRINT = 9;
let N_BLOCK = 10;
let N_PROGRAM = 11;

// ============================================================
// Lexer
// ============================================================

let tokens = [];
let tok_state = [0];

fn is_digit(c) {
    let code = char_code(c);
    return code >= 48 and code <= 57;
}

fn is_alpha(c) {
    let code = char_code(c);
    return (code >= 65 and code <= 90) or (code >= 97 and code <= 122) or code == 95;
}

fn is_alnum(c) {
    return is_digit(c) or is_alpha(c);
}

fn lex_source(src) {
    let pos = 0;
    let line = 1;
    let src_len = len(src);
    while (pos < src_len) {
        let c = char_at(src, pos);
        if (c == " " or c == "\t" or c == "\r") {
            pos = pos + 1;
        } else if (c == "\n") {
            line = line + 1;
            pos = pos + 1;
        } else if (is_digit(c)) {
            let start = pos;
            while (pos < src_len and is_digit(char_at(src, pos))) {
                pos = pos + 1;
            }
            let num_str = substr(src, start, pos - start);
            push(tokens, [TK_INT, to_int(num_str), line]);
        } else if (is_alpha(c)) {
            let start = pos;
            while (pos < src_len and is_alnum(char_at(src, pos))) {
                pos = pos + 1;
            }
            let word = substr(src, start, pos - start);
            if (word == "let") {
                push(tokens, [TK_LET, word, line]);
            } else if (word == "if") {
                push(tokens, [TK_IF, word, line]);
            } else if (word == "else") {
                push(tokens, [TK_ELSE, word, line]);
            } else if (word == "while") {
                push(tokens, [TK_WHILE, word, line]);
            } else if (word == "print") {
                push(tokens, [TK_PRINT, word, line]);
            } else {
                push(tokens, [TK_IDENT, word, line]);
            }
        } else if (c == "/" and pos + 1 < src_len and char_at(src, pos + 1) == "/") {
            while (pos < src_len and char_at(src, pos) != "\n") {
                pos = pos + 1;
            }
        } else if (c == "=") {
            if (pos + 1 < src_len and char_at(src, pos + 1) == "=") {
                push(tokens, [TK_EQ, "==", line]);
                pos = pos + 2;
            } else {
                push(tokens, [TK_ASSIGN, "=", line]);
                pos = pos + 1;
            }
        } else if (c == "!") {
            if (pos + 1 < src_len and char_at(src, pos + 1) == "=") {
                push(tokens, [TK_NE, "!=", line]);
                pos = pos + 2;
            } else {
                pos = pos + 1;
            }
        } else if (c == "<") {
            if (pos + 1 < src_len and char_at(src, pos + 1) == "=") {
                push(tokens, [TK_LE, "<=", line]);
                pos = pos + 2;
            } else {
                push(tokens, [TK_LT, "<", line]);
                pos = pos + 1;
            }
        } else if (c == ">") {
            if (pos + 1 < src_len and char_at(src, pos + 1) == "=") {
                push(tokens, [TK_GE, ">=", line]);
                pos = pos + 2;
            } else {
                push(tokens, [TK_GT, ">", line]);
                pos = pos + 1;
            }
        } else if (c == "+") {
            push(tokens, [TK_PLUS, "+", line]);
            pos = pos + 1;
        } else if (c == "-") {
            push(tokens, [TK_MINUS, "-", line]);
            pos = pos + 1;
        } else if (c == "*") {
            push(tokens, [TK_STAR, "*", line]);
            pos = pos + 1;
        } else if (c == "/") {
            push(tokens, [TK_SLASH, "/", line]);
            pos = pos + 1;
        } else if (c == "%") {
            push(tokens, [TK_PERCENT, "%", line]);
            pos = pos + 1;
        } else if (c == "(") {
            push(tokens, [TK_LPAREN, "(", line]);
            pos = pos + 1;
        } else if (c == ")") {
            push(tokens, [TK_RPAREN, ")", line]);
            pos = pos + 1;
        } else if (c == "{") {
            push(tokens, [TK_LBRACE, "{", line]);
            pos = pos + 1;
        } else if (c == "}") {
            push(tokens, [TK_RBRACE, "}", line]);
            pos = pos + 1;
        } else if (c == ";") {
            push(tokens, [TK_SEMICOLON, ";", line]);
            pos = pos + 1;
        } else {
            pos = pos + 1;
        }
    }
    push(tokens, [TK_EOF, 0, line]);
}

// ============================================================
// Parser helpers
// ============================================================

fn peek_type() {
    return tokens[tok_state[0]][0];
}

fn peek_val() {
    return tokens[tok_state[0]][1];
}

fn advance_tok() {
    let t = tokens[tok_state[0]];
    tok_state[0] = tok_state[0] + 1;
    return t;
}

fn expect_tok(tt) {
    let t = tokens[tok_state[0]];
    tok_state[0] = tok_state[0] + 1;
    return t;
}

// ============================================================
// Parser
// ============================================================

fn parse_program() {
    let stmts = [];
    while (peek_type() != TK_EOF) {
        push(stmts, parse_declaration());
    }
    return [N_PROGRAM, stmts];
}

fn parse_declaration() {
    if (peek_type() == TK_LET) {
        return parse_let();
    }
    return parse_statement();
}

fn parse_let() {
    advance_tok();
    let name = advance_tok()[1];
    expect_tok(TK_ASSIGN);
    let val = parse_expression();
    expect_tok(TK_SEMICOLON);
    return [N_LET, name, val];
}

fn parse_statement() {
    if (peek_type() == TK_IF) { return parse_if(); }
    if (peek_type() == TK_WHILE) { return parse_while(); }
    if (peek_type() == TK_PRINT) { return parse_print(); }
    if (peek_type() == TK_LBRACE) { return parse_block(); }
    return parse_expr_stmt();
}

fn parse_if() {
    advance_tok();
    expect_tok(TK_LPAREN);
    let cond = parse_expression();
    expect_tok(TK_RPAREN);
    let then_body = parse_block();
    let else_body = 0;
    if (peek_type() == TK_ELSE) {
        advance_tok();
        if (peek_type() == TK_IF) {
            else_body = parse_if();
        } else {
            else_body = parse_block();
        }
    }
    return [N_IF, cond, then_body, else_body];
}

fn parse_while() {
    advance_tok();
    expect_tok(TK_LPAREN);
    let cond = parse_expression();
    expect_tok(TK_RPAREN);
    let body = parse_block();
    return [N_WHILE, cond, body];
}

fn parse_print() {
    advance_tok();
    expect_tok(TK_LPAREN);
    let val = parse_expression();
    expect_tok(TK_RPAREN);
    expect_tok(TK_SEMICOLON);
    return [N_PRINT, val];
}

fn parse_block() {
    expect_tok(TK_LBRACE);
    let stmts = [];
    while (peek_type() != TK_RBRACE) {
        push(stmts, parse_declaration());
    }
    expect_tok(TK_RBRACE);
    return [N_BLOCK, stmts];
}

fn parse_expr_stmt() {
    let expr = parse_expression();
    expect_tok(TK_SEMICOLON);
    return expr;
}

fn parse_expression() {
    return parse_assignment();
}

fn parse_assignment() {
    let left = parse_comparison();
    if (peek_type() == TK_ASSIGN) {
        advance_tok();
        let val = parse_assignment();
        return [N_ASSIGN, left[1], val];
    }
    return left;
}

fn parse_comparison() {
    let left = parse_addition();
    while (peek_type() == TK_EQ or peek_type() == TK_NE or peek_type() == TK_LT or peek_type() == TK_GT or peek_type() == TK_LE or peek_type() == TK_GE) {
        let op_tok = advance_tok();
        let right = parse_addition();
        left = [N_BINOP, op_tok[1], left, right];
    }
    return left;
}

fn parse_addition() {
    let left = parse_multiplication();
    while (peek_type() == TK_PLUS or peek_type() == TK_MINUS) {
        let op_tok = advance_tok();
        let right = parse_multiplication();
        left = [N_BINOP, op_tok[1], left, right];
    }
    return left;
}

fn parse_multiplication() {
    let left = parse_unary();
    while (peek_type() == TK_STAR or peek_type() == TK_SLASH or peek_type() == TK_PERCENT) {
        let op_tok = advance_tok();
        let right = parse_unary();
        left = [N_BINOP, op_tok[1], left, right];
    }
    return left;
}

fn parse_unary() {
    if (peek_type() == TK_MINUS) {
        advance_tok();
        let operand = parse_unary();
        return [N_UNARY, "-", operand];
    }
    return parse_primary();
}

fn parse_primary() {
    if (peek_type() == TK_INT) {
        let t = advance_tok();
        return [N_INT, t[1]];
    }
    if (peek_type() == TK_IDENT) {
        let t = advance_tok();
        return [N_VAR, t[1]];
    }
    if (peek_type() == TK_LPAREN) {
        advance_tok();
        let expr = parse_expression();
        expect_tok(TK_RPAREN);
        return expr;
    }
    advance_tok();
    return [N_INT, 0];
}

// ============================================================
// Code Generator
// ============================================================

let bytecode = [];
let constants = [];
let names = [];

fn add_const(val) {
    let i = 0;
    while (i < len(constants)) {
        if (constants[i] == val) {
            return i;
        }
        i = i + 1;
    }
    push(constants, val);
    return len(constants) - 1;
}

fn add_name(nm) {
    let i = 0;
    while (i < len(names)) {
        if (names[i] == nm) {
            return i;
        }
        i = i + 1;
    }
    push(names, nm);
    return len(names) - 1;
}

fn emit1(op) {
    push(bytecode, op);
    return len(bytecode) - 1;
}

fn emit2(op, operand) {
    push(bytecode, op);
    push(bytecode, operand);
    return len(bytecode) - 2;
}

fn patch(addr, val) {
    bytecode[addr] = val;
}

fn compile_node(node) {
    let ntype = node[0];
    if (ntype == N_INT) {
        let idx = add_const(node[1]);
        emit2(OP_CONST, idx);
    } else if (ntype == N_VAR) {
        let idx = add_name(node[1]);
        emit2(OP_LOAD, idx);
    } else if (ntype == N_BINOP) {
        compile_node(node[2]);
        compile_node(node[3]);
        let op = node[1];
        if (op == "+") { emit1(OP_ADD); }
        else if (op == "-") { emit1(OP_SUB); }
        else if (op == "*") { emit1(OP_MUL); }
        else if (op == "/") { emit1(OP_DIV); }
        else if (op == "%") { emit1(OP_MOD); }
        else if (op == "==") { emit1(OP_EQ); }
        else if (op == "!=") { emit1(OP_NE); }
        else if (op == "<") { emit1(OP_LT); }
        else if (op == ">") { emit1(OP_GT); }
        else if (op == "<=") { emit1(OP_LE); }
        else if (op == ">=") { emit1(OP_GE); }
    } else if (ntype == N_UNARY) {
        compile_node(node[2]);
        emit1(OP_NEG);
    } else if (ntype == N_ASSIGN) {
        compile_node(node[2]);
        let idx = add_name(node[1]);
        emit2(OP_STORE, idx);
    } else if (ntype == N_LET) {
        compile_node(node[2]);
        let idx = add_name(node[1]);
        emit2(OP_STORE, idx);
    } else if (ntype == N_IF) {
        compile_node(node[1]);
        let jf = emit2(OP_JUMP_IF_FALSE, 0);
        emit1(OP_POP);
        compile_node(node[2]);
        if (node[3] != 0) {
            let je = emit2(OP_JUMP, 0);
            patch(jf + 1, len(bytecode));
            emit1(OP_POP);
            compile_node(node[3]);
            patch(je + 1, len(bytecode));
        } else {
            let je = emit2(OP_JUMP, 0);
            patch(jf + 1, len(bytecode));
            emit1(OP_POP);
            patch(je + 1, len(bytecode));
        }
    } else if (ntype == N_WHILE) {
        let loop_start = len(bytecode);
        compile_node(node[1]);
        let jf = emit2(OP_JUMP_IF_FALSE, 0);
        emit1(OP_POP);
        compile_node(node[2]);
        emit2(OP_JUMP, loop_start);
        patch(jf + 1, len(bytecode));
        emit1(OP_POP);
    } else if (ntype == N_PRINT) {
        compile_node(node[1]);
        emit1(OP_PRINT);
    } else if (ntype == N_BLOCK) {
        let stmts = node[1];
        let i = 0;
        while (i < len(stmts)) {
            compile_node(stmts[i]);
            i = i + 1;
        }
    } else if (ntype == N_PROGRAM) {
        let stmts = node[1];
        let i = 0;
        while (i < len(stmts)) {
            compile_node(stmts[i]);
            i = i + 1;
        }
        emit1(OP_HALT);
    }
}

// ============================================================
// Main: compile the source code stored in global 'source'
// ============================================================

lex_source(source);
let ast = parse_program();
compile_node(ast);

// Results are in globals: bytecode, constants, names
'''


def compile_subset_with_host(source: str) -> tuple:
    """Compile an integer-subset program using the Python host compiler.
    Returns (bytecode_list, constants_list, names_list) matching what the
    self-compiler would produce."""
    tokens = lex(source)
    parser = Parser(tokens)
    ast = parser.parse()
    compiler = Compiler()
    chunk = compiler.compile(ast)
    return list(chunk.code), list(chunk.constants), list(chunk.names)


def run_self_compiler(source_to_compile: str) -> tuple:
    """Run the self-compiler (in MiniLang) to compile source_to_compile.
    Returns (bytecode_list, constants_list, names_list)."""
    # Compile the self-compiler with the host compiler
    chunk, compiler = compile_source(SELF_COMPILER_SOURCE)
    # Run it with 'source' pre-loaded in the environment
    vm = VM(chunk)
    vm.env['source'] = source_to_compile
    vm.run()
    # Extract results
    bytecode = vm.env.get('bytecode', [])
    constants = vm.env.get('constants', [])
    names = vm.env.get('names', [])
    return bytecode, constants, names


def bootstrap_verify(source: str) -> dict:
    """Verify that host compiler and self-compiler produce identical output
    for the given source program."""
    host_code, host_consts, host_names = compile_subset_with_host(source)
    self_code, self_consts, self_names = run_self_compiler(source)

    match = (host_code == self_code and
             host_consts == self_consts and
             host_names == self_names)

    return {
        'match': match,
        'host': {'code': host_code, 'constants': host_consts, 'names': host_names},
        'self': {'code': self_code, 'constants': self_consts, 'names': self_names},
    }
