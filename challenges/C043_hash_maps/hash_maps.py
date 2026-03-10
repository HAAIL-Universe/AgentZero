"""
Hash Maps -- First-class dictionary type for the VM language
Challenge C043 -- AgentZero Session 044

Extends C042 (Arrays/Closures/C010 Stack VM) with:
  - Hash map literals: {key: value, key2: value2}
  - Dot access: obj.key, obj.key = value
  - String key indexing: obj["key"], obj["key"] = value
  - Nested hash maps: {a: {b: 1}}
  - Hash + array composition: {items: [1, 2, 3]}
  - Builtins: keys(), values(), has(), delete(), merge(), entries(), size()

New opcodes: MAKE_HASH
New AST: HashLit, DotExpr, DotAssign
New tokens: COLON, DOT
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Instruction Set
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

    # Closures (from C041)
    MAKE_CLOSURE = auto()

    # Arrays (from C042)
    MAKE_ARRAY = auto()
    INDEX_GET = auto()
    INDEX_SET = auto()

    # Hash maps (new in C043)
    MAKE_HASH = auto()


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
        # Type-aware comparison to avoid True==1 collisions
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
    COLON = auto()      # new
    DOT = auto()         # new

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
        elif c == ':':
            tokens.append(Token(TokenType.COLON, ':', line)); i += 1
        elif c == '.':
            tokens.append(Token(TokenType.DOT, '.', line)); i += 1
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
    callee: Any
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

@dataclass
class LambdaExpr:
    params: list
    body: Any
    line: int = 0

# C042 AST nodes
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

# C043 AST nodes
@dataclass
class HashLit:
    """Hash map literal: {key: value, ...}"""
    pairs: list  # list of (key_expr, value_expr) tuples
    line: int = 0

@dataclass
class DotExpr:
    """Dot access: obj.key"""
    obj: Any
    key: str
    line: int = 0

@dataclass
class DotAssign:
    """Dot assignment: obj.key = value"""
    obj: Any
    key: str
    value: Any
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
            if self._is_fn_decl():
                return self.fn_decl()
        if self.peek().type == TokenType.LET:
            return self.let_decl()
        return self.statement()

    def _is_fn_decl(self):
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1].type == TokenType.IDENT
        return False

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
            # Disambiguate: block vs hash literal in statement position
            # If next is RBRACE or (IDENT|STRING) COLON, it's a hash literal expr stmt
            if self._is_hash_literal():
                return self.expr_stmt()
            return self.block()
        return self.expr_stmt()

    def _is_hash_literal(self):
        """Look ahead to determine if { starts a hash literal vs a block."""
        # Save position
        saved = self.pos
        # We're at LBRACE
        if self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]
            # {} is empty hash
            if next_tok.type == TokenType.RBRACE:
                return True
            # {ident: or {string: is hash
            if next_tok.type in (TokenType.IDENT, TokenType.STRING):
                if self.pos + 2 < len(self.tokens):
                    after = self.tokens[self.pos + 2]
                    if after.type == TokenType.COLON:
                        return True
            # {int: is hash (numeric keys)
            if next_tok.type == TokenType.INT:
                if self.pos + 2 < len(self.tokens):
                    after = self.tokens[self.pos + 2]
                    if after.type == TokenType.COLON:
                        return True
        return False

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
        return expr

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
                return IndexAssign(obj=expr.obj, index=expr.index, value=value, line=expr.line)
            if isinstance(expr, DotExpr):
                value = self.assignment()
                return DotAssign(obj=expr.obj, key=expr.key, value=value, line=expr.line)
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
        """Parse primary followed by call (), index [], and dot . expressions."""
        expr = self.primary()
        while True:
            if self.peek().type == TokenType.LPAREN:
                self.advance()  # (
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.expression())
                self.expect(TokenType.RPAREN)
                expr = CallExpr(callee=expr, args=args, line=expr.line)
            elif self.peek().type == TokenType.LBRACKET:
                self.advance()  # [
                index = self.expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexExpr(obj=expr, index=index, line=expr.line)
            elif self.peek().type == TokenType.DOT:
                self.advance()  # .
                # Accept any identifier-like token after dot (including keywords)
                tok = self.peek()
                if tok.type == TokenType.IDENT or tok.value in KEYWORDS:
                    key = self.advance().value
                else:
                    raise ParseError(f"Expected identifier after '.', got {tok.type.name} at line {tok.line}")
                expr = DotExpr(obj=expr, key=key, line=expr.line)
            else:
                break
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
            return Var(tok.value, tok.line)

        # Lambda: fn(params) { body }
        if tok.type == TokenType.FN:
            self.advance()  # fn
            self.expect(TokenType.LPAREN)
            params = []
            if self.peek().type != TokenType.RPAREN:
                params.append(self.expect(TokenType.IDENT).value)
                while self.match(TokenType.COMMA):
                    params.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.RPAREN)
            body = self.block()
            return LambdaExpr(params=params, body=body, line=tok.line)

        # Array literal: [expr, expr, ...]
        if tok.type == TokenType.LBRACKET:
            self.advance()  # [
            elements = []
            if self.peek().type != TokenType.RBRACKET:
                elements.append(self.expression())
                while self.match(TokenType.COMMA):
                    if self.peek().type == TokenType.RBRACKET:
                        break
                    elements.append(self.expression())
            self.expect(TokenType.RBRACKET)
            return ArrayLit(elements=elements, line=tok.line)

        # Hash map literal: {key: value, ...}
        if tok.type == TokenType.LBRACE:
            return self._parse_hash_literal()

        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN)
            return expr

        raise ParseError(f"Unexpected token {tok.type.name} '{tok.value}' at line {tok.line}")

    def _parse_hash_literal(self):
        """Parse a hash map literal: {key: value, ...}"""
        tok = self.advance()  # {
        pairs = []
        if self.peek().type != TokenType.RBRACE:
            key = self.expression()
            self.expect(TokenType.COLON)
            value = self.expression()
            pairs.append((key, value))
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RBRACE:
                    break  # trailing comma
                key = self.expression()
                self.expect(TokenType.COLON)
                value = self.expression()
                pairs.append((key, value))
        self.expect(TokenType.RBRACE)
        return HashLit(pairs=pairs, line=tok.line)


# ============================================================
# Compiler
# ============================================================

class CompileError(Exception):
    pass


@dataclass
class FnObject:
    name: str
    arity: int
    chunk: Chunk


@dataclass
class ClosureObject:
    fn: FnObject
    env: dict


# Builtin function names
BUILTINS = {
    'len', 'push', 'pop', 'map', 'filter', 'reduce', 'range',
    'slice', 'concat', 'sort', 'reverse', 'find', 'each',
    # C043 hash map builtins
    'keys', 'values', 'has', 'delete', 'merge', 'entries', 'size',
}


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

    def _compile_function_body(self, params, body):
        fn_compiler = self.__class__()
        for param in params:
            fn_compiler.chunk.add_name(param)
        fn_compiler.compile_node(body)
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx)
        fn_compiler.chunk.emit(Op.RETURN)
        return fn_compiler

    def compile_FnDecl(self, node):
        fn_compiler = self._compile_function_body(node.params, node.body)
        fn_obj = FnObject(node.name, len(node.params), fn_compiler.chunk)
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v
        self.functions[node.name] = fn_obj

        fn_idx = self.chunk.add_constant(fn_obj)
        name_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.MAKE_CLOSURE, line=node.line)
        self.chunk.emit(Op.STORE, name_idx, node.line)

    def compile_LambdaExpr(self, node):
        fn_compiler = self._compile_function_body(node.params, node.body)
        fn_obj = FnObject("<lambda>", len(node.params), fn_compiler.chunk)
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v

        fn_idx = self.chunk.add_constant(fn_obj)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.MAKE_CLOSURE, line=node.line)

    def compile_CallExpr(self, node):
        self.compile_node(node.callee)
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

    # C042 compilation
    def compile_ArrayLit(self, node):
        for elem in node.elements:
            self.compile_node(elem)
        self.chunk.emit(Op.MAKE_ARRAY, len(node.elements), node.line)

    def compile_IndexExpr(self, node):
        self.compile_node(node.obj)
        self.compile_node(node.index)
        self.chunk.emit(Op.INDEX_GET, line=node.line)

    def compile_IndexAssign(self, node):
        self.compile_node(node.obj)
        self.compile_node(node.index)
        self.compile_node(node.value)
        self.chunk.emit(Op.INDEX_SET, line=node.line)

    # C043 compilation
    def compile_HashLit(self, node):
        for key_expr, value_expr in node.pairs:
            self.compile_node(key_expr)
            self.compile_node(value_expr)
        self.chunk.emit(Op.MAKE_HASH, len(node.pairs), node.line)

    def compile_DotExpr(self, node):
        """obj.key compiles to: push obj, push "key", INDEX_GET"""
        self.compile_node(node.obj)
        idx = self.chunk.add_constant(node.key)
        self.chunk.emit(Op.CONST, idx, node.line)
        self.chunk.emit(Op.INDEX_GET, line=node.line)

    def compile_DotAssign(self, node):
        """obj.key = value compiles to: push obj, push "key", push value, INDEX_SET"""
        self.compile_node(node.obj)
        idx = self.chunk.add_constant(node.key)
        self.chunk.emit(Op.CONST, idx, node.line)
        self.compile_node(node.value)
        self.chunk.emit(Op.INDEX_SET, line=node.line)


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


def _format_value(value):
    """Format a value for printing."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        parts = [_format_value(v) for v in value]
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, dict):
        parts = [f"{_format_value(k)}: {_format_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, ClosureObject):
        return f"<closure:{value.fn.name}>"
    if isinstance(value, FnObject):
        return f"<fn:{value.name}>"
    return str(value)


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
        self.max_steps = 100000

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

    def _call_builtin(self, name, args):
        """Execute a built-in function. Returns result value."""
        if name == 'len':
            if len(args) != 1:
                raise VMError(f"len() takes 1 argument, got {len(args)}")
            arg = args[0]
            if isinstance(arg, (list, str, dict)):
                return len(arg)
            raise VMError(f"len() requires array, string, or hash map, got {type(arg).__name__}")

        elif name == 'push':
            if len(args) != 2:
                raise VMError(f"push() takes 2 arguments, got {len(args)}")
            arr, val = args
            if not isinstance(arr, list):
                raise VMError(f"push() requires array as first argument")
            arr.append(val)
            return arr

        elif name == 'pop':
            if len(args) != 1:
                raise VMError(f"pop() takes 1 argument, got {len(args)}")
            arr = args[0]
            if not isinstance(arr, list):
                raise VMError(f"pop() requires array")
            if len(arr) == 0:
                raise VMError("pop() on empty array")
            return arr.pop()

        elif name == 'range':
            if len(args) == 1:
                return list(range(args[0]))
            elif len(args) == 2:
                return list(range(args[0], args[1]))
            elif len(args) == 3:
                return list(range(args[0], args[1], args[2]))
            else:
                raise VMError(f"range() takes 1-3 arguments, got {len(args)}")

        elif name == 'map':
            if len(args) != 2:
                raise VMError(f"map() takes 2 arguments, got {len(args)}")
            arr, fn = args
            if not isinstance(arr, list):
                raise VMError(f"map() requires array as first argument")
            return [self._call_function(fn, [elem]) for elem in arr]

        elif name == 'filter':
            if len(args) != 2:
                raise VMError(f"filter() takes 2 arguments, got {len(args)}")
            arr, fn = args
            if not isinstance(arr, list):
                raise VMError(f"filter() requires array as first argument")
            result = []
            for elem in arr:
                if self._call_function(fn, [elem]):
                    result.append(elem)
            return result

        elif name == 'reduce':
            if len(args) != 3:
                raise VMError(f"reduce() takes 3 arguments, got {len(args)}")
            arr, fn, init = args
            if not isinstance(arr, list):
                raise VMError(f"reduce() requires array as first argument")
            acc = init
            for elem in arr:
                acc = self._call_function(fn, [acc, elem])
            return acc

        elif name == 'slice':
            if len(args) < 2 or len(args) > 3:
                raise VMError(f"slice() takes 2-3 arguments, got {len(args)}")
            arr = args[0]
            if not isinstance(arr, list):
                raise VMError(f"slice() requires array as first argument")
            start = args[1]
            end = args[2] if len(args) == 3 else len(arr)
            return arr[start:end]

        elif name == 'concat':
            if len(args) != 2:
                raise VMError(f"concat() takes 2 arguments, got {len(args)}")
            a, b = args
            if not isinstance(a, list) or not isinstance(b, list):
                raise VMError(f"concat() requires two arrays")
            return a + b

        elif name == 'sort':
            if len(args) != 1:
                raise VMError(f"sort() takes 1 argument, got {len(args)}")
            arr = args[0]
            if not isinstance(arr, list):
                raise VMError(f"sort() requires array")
            return sorted(arr)

        elif name == 'reverse':
            if len(args) != 1:
                raise VMError(f"reverse() takes 1 argument, got {len(args)}")
            arr = args[0]
            if not isinstance(arr, list):
                raise VMError(f"reverse() requires array")
            return list(reversed(arr))

        elif name == 'find':
            if len(args) != 2:
                raise VMError(f"find() takes 2 arguments, got {len(args)}")
            arr, fn = args
            if not isinstance(arr, list):
                raise VMError(f"find() requires array as first argument")
            for elem in arr:
                if self._call_function(fn, [elem]):
                    return elem
            return None

        elif name == 'each':
            if len(args) != 2:
                raise VMError(f"each() takes 2 arguments, got {len(args)}")
            arr, fn = args
            if not isinstance(arr, list):
                raise VMError(f"each() requires array as first argument")
            for elem in arr:
                self._call_function(fn, [elem])
            return None

        # C043 hash map builtins
        elif name == 'keys':
            if len(args) != 1:
                raise VMError(f"keys() takes 1 argument, got {len(args)}")
            obj = args[0]
            if not isinstance(obj, dict):
                raise VMError(f"keys() requires hash map, got {type(obj).__name__}")
            return list(obj.keys())

        elif name == 'values':
            if len(args) != 1:
                raise VMError(f"values() takes 1 argument, got {len(args)}")
            obj = args[0]
            if not isinstance(obj, dict):
                raise VMError(f"values() requires hash map, got {type(obj).__name__}")
            return list(obj.values())

        elif name == 'has':
            if len(args) != 2:
                raise VMError(f"has() takes 2 arguments, got {len(args)}")
            obj, key = args
            if not isinstance(obj, dict):
                raise VMError(f"has() requires hash map as first argument, got {type(obj).__name__}")
            return key in obj

        elif name == 'delete':
            if len(args) != 2:
                raise VMError(f"delete() takes 2 arguments, got {len(args)}")
            obj, key = args
            if not isinstance(obj, dict):
                raise VMError(f"delete() requires hash map as first argument, got {type(obj).__name__}")
            if key in obj:
                del obj[key]
            return obj

        elif name == 'merge':
            if len(args) != 2:
                raise VMError(f"merge() takes 2 arguments, got {len(args)}")
            a, b = args
            if not isinstance(a, dict) or not isinstance(b, dict):
                raise VMError(f"merge() requires two hash maps")
            result = dict(a)
            result.update(b)
            return result

        elif name == 'entries':
            if len(args) != 1:
                raise VMError(f"entries() takes 1 argument, got {len(args)}")
            obj = args[0]
            if not isinstance(obj, dict):
                raise VMError(f"entries() requires hash map, got {type(obj).__name__}")
            return [[k, v] for k, v in obj.items()]

        elif name == 'size':
            if len(args) != 1:
                raise VMError(f"size() takes 1 argument, got {len(args)}")
            obj = args[0]
            if isinstance(obj, (dict, list, str)):
                return len(obj)
            raise VMError(f"size() requires hash map, array, or string, got {type(obj).__name__}")

        else:
            raise VMError(f"Unknown builtin: {name}")

    def _call_function(self, fn_val, args):
        """Call a function/closure with args and return result."""
        captured_env = None
        if isinstance(fn_val, ClosureObject):
            captured_env = fn_val.env
            fn_obj = fn_val.fn
        elif isinstance(fn_val, FnObject):
            fn_obj = fn_val
        else:
            raise VMError(f"Cannot call non-function: {fn_val}")

        if fn_obj.arity != len(args):
            raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {len(args)}")

        saved_chunk = self.current_chunk
        saved_ip = self.ip
        saved_env = self.env
        saved_call_stack_depth = len(self.call_stack)

        self.current_chunk = fn_obj.chunk
        self.ip = 0

        if captured_env is not None:
            self.env = dict(captured_env)
        else:
            self.env = dict(self.env)

        for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
            self.env[param_name] = args[i]

        result = self._run_until_return(saved_call_stack_depth)

        self.current_chunk = saved_chunk
        self.ip = saved_ip
        self.env = saved_env

        return result

    def _run_until_return(self, base_depth):
        """Run VM until we return to the given call stack depth."""
        while True:
            self.step_count += 1
            if self.step_count > self.max_steps:
                raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

            if self.ip >= len(self.current_chunk.code):
                return self.stack[-1] if self.stack else None

            op = self.current_chunk.code[self.ip]
            self.ip += 1

            if op == Op.RETURN:
                return_val = self.pop()
                if len(self.call_stack) <= base_depth:
                    return return_val
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                self.env = frame.base_env
                self.push(return_val)
                continue

            self._execute_op(op)

    def _execute_op(self, op):
        """Execute a single opcode."""
        if op == Op.HALT:
            return 'halt'

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
            if isinstance(a, list) and isinstance(b, int):
                self.push(a * b)
            else:
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
            if name in self.env:
                self.push(self.env[name])
            elif name in BUILTINS:
                self.push(('__builtin__', name))
            else:
                raise VMError(f"Undefined variable '{name}'")

        elif op == Op.STORE:
            idx = self.current_chunk.code[self.ip]
            self.ip += 1
            name = self.current_chunk.names[idx]
            value = self.pop()
            self.env[name] = value
            if isinstance(value, ClosureObject):
                value.env[name] = value

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
            fn_val = self.pop()

            # Check for builtin call
            if isinstance(fn_val, tuple) and len(fn_val) == 2 and fn_val[0] == '__builtin__':
                result = self._call_builtin(fn_val[1], args)
                self.push(result)
                return

            captured_env = None
            if isinstance(fn_val, ClosureObject):
                captured_env = fn_val.env
                fn_obj = fn_val.fn
            elif isinstance(fn_val, FnObject):
                fn_obj = fn_val
            else:
                raise VMError(f"Cannot call non-function: {fn_val}")

            if fn_obj.arity != arg_count:
                raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")

            frame = CallFrame(self.current_chunk, self.ip, self.env)
            self.call_stack.append(frame)

            self.current_chunk = fn_obj.chunk
            self.ip = 0

            if captured_env is not None:
                self.env = dict(captured_env)
            else:
                self.env = dict(self.env)

            for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                self.env[param_name] = args[i]

        elif op == Op.RETURN:
            return_val = self.pop()
            if not self.call_stack:
                self.push(return_val)
                return 'halt'
            frame = self.call_stack.pop()
            self.current_chunk = frame.chunk
            self.ip = frame.ip
            self.env = frame.base_env
            self.push(return_val)

        elif op == Op.PRINT:
            value = self.pop()
            self.output.append(_format_value(value))

        elif op == Op.MAKE_CLOSURE:
            fn_obj = self.pop()
            if not isinstance(fn_obj, FnObject):
                raise VMError(f"MAKE_CLOSURE expects FnObject, got {type(fn_obj).__name__}")
            closure = ClosureObject(fn=fn_obj, env=dict(self.env))
            self.push(closure)

        elif op == Op.MAKE_ARRAY:
            count = self.current_chunk.code[self.ip]
            self.ip += 1
            elements = []
            for _ in range(count):
                elements.insert(0, self.pop())
            self.push(elements)

        elif op == Op.INDEX_GET:
            index = self.pop()
            obj = self.pop()
            if isinstance(obj, list):
                if not isinstance(index, int):
                    raise VMError(f"Array index must be integer, got {type(index).__name__}")
                if index < 0 or index >= len(obj):
                    raise VMError(f"Array index {index} out of bounds (length {len(obj)})")
                self.push(obj[index])
            elif isinstance(obj, str):
                if not isinstance(index, int):
                    raise VMError(f"String index must be integer, got {type(index).__name__}")
                if index < 0 or index >= len(obj):
                    raise VMError(f"String index {index} out of bounds (length {len(obj)})")
                self.push(obj[index])
            elif isinstance(obj, dict):
                if index not in obj:
                    raise VMError(f"Key {_format_value(index)} not found in hash map")
                self.push(obj[index])
            else:
                raise VMError(f"Cannot index type {type(obj).__name__}")

        elif op == Op.INDEX_SET:
            value = self.pop()
            index = self.pop()
            obj = self.pop()
            if isinstance(obj, list):
                if not isinstance(index, int):
                    raise VMError(f"Array index must be integer, got {type(index).__name__}")
                if index < 0 or index >= len(obj):
                    raise VMError(f"Array index {index} out of bounds (length {len(obj)})")
                obj[index] = value
                self.push(value)
            elif isinstance(obj, dict):
                obj[index] = value
                self.push(value)
            else:
                raise VMError(f"Cannot assign to index of type {type(obj).__name__}")

        elif op == Op.MAKE_HASH:
            count = self.current_chunk.code[self.ip]
            self.ip += 1
            result = {}
            pairs = []
            for _ in range(count):
                v = self.pop()
                k = self.pop()
                pairs.append((k, v))
            # Reverse to preserve insertion order
            for k, v in reversed(pairs):
                result[k] = v
            self.push(result)

        else:
            raise VMError(f"Unknown opcode: {op}")

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

            result = self._execute_op(op)
            if result == 'halt':
                break

        return self.stack[-1] if self.stack else None

    def _trace_op(self, op):
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"
        print(f"  [{self.ip-1:04d}] {name:20s} stack={self.stack[-5:]}")


# ============================================================
# Public API
# ============================================================

def parse(source: str) -> Program:
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()


def compile_source(source: str) -> tuple:
    ast = parse(source)
    compiler = Compiler()
    chunk = compiler.compile(ast)
    return chunk, compiler


def execute(source: str, trace=False) -> dict:
    chunk, compiler = compile_source(source)
    vm = VM(chunk, trace=trace)
    result = vm.run()
    return {
        'result': result,
        'output': vm.output,
        'env': vm.env,
        'steps': vm.step_count,
    }


def run(source: str) -> tuple:
    r = execute(source)
    return r['result'], r['output']


def disassemble(chunk: Chunk) -> str:
    lines = []
    i = 0
    while i < len(chunk.code):
        op = chunk.code[i]
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"

        if op in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
                  Op.JUMP_IF_TRUE, Op.CALL, Op.MAKE_ARRAY, Op.MAKE_HASH):
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
