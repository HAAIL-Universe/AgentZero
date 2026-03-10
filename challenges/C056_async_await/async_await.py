"""
Async/Await -- cooperative async functions with promises
Challenge C056 -- AgentZero Session 057

Extends C055 (Finally Blocks) with:
  - async fn name() { }  -- async function declaration
  - await expr  -- suspend until promise resolves
  - Calling async fn returns a PromiseObject
  - PromiseObject: resolve/reject, then/catch chaining, Promise.all/race
  - Event loop: runs pending async continuations cooperatively
  - Async generators NOT supported (async fn* is an error)
  - await in non-async context is a compile error
  - try/catch works with rejected promises via await

New tokens: ASYNC, AWAIT
New opcode: AWAIT
New AST: AsyncFnDecl, AwaitExpr
New classes: PromiseObject
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

    # Hash maps (from C043)
    MAKE_HASH = auto()

    # Iteration (from C044)
    ITER_PREPARE = auto()
    ITER_LENGTH = auto()

    # Error handling (from C045)
    SETUP_TRY = auto()
    POP_TRY = auto()
    THROW = auto()

    # Generators (C047)
    YIELD = auto()

    # Spread (C050)
    ARRAY_SPREAD = auto()
    HASH_SPREAD = auto()
    CALL_SPREAD = auto()

    # Classes (C052)
    MAKE_CLASS = auto()
    LOOKUP_METHOD = auto()
    SUPER_INVOKE = auto()

    # Null coalescing (C054)
    NULL_COALESCE = auto()

    # Finally blocks (C055)
    SETUP_FINALLY = auto()
    POP_FINALLY = auto()
    END_FINALLY = auto()

    # Async/Await (C056)
    AWAIT = auto()


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
            if c is value or (type(c) == type(value) and c == value):
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
# Tokens
# ============================================================

class TokenType(IntEnum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
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
    COLON = auto()
    DOT = auto()

    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FN = auto()
    RETURN = auto()
    PRINT = auto()
    FOR = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    TRY = auto()
    CATCH = auto()
    THROW = auto()
    FINALLY = auto()

    # C046 module system
    IMPORT = auto()
    EXPORT = auto()
    FROM = auto()

    # C047 generators
    YIELD = auto()

    # C048 destructuring
    DOTDOTDOT = auto()

    # C049 string interpolation
    FSTRING = auto()

    # C051 pipe operator
    PIPE = auto()

    # C052 classes
    CLASS = auto()
    SUPER = auto()

    # C053 optional chaining
    QUESTION_DOT = auto()

    # C054 null coalescing
    QUESTION_QUESTION = auto()
    QUESTION_QUESTION_ASSIGN = auto()

    # C056 async/await
    ASYNC = auto()
    AWAIT = auto()

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
    'null': TokenType.NULL,
    'and': TokenType.AND,
    'or': TokenType.OR,
    'not': TokenType.NOT,
    'for': TokenType.FOR,
    'in': TokenType.IN,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'try': TokenType.TRY,
    'catch': TokenType.CATCH,
    'throw': TokenType.THROW,
    'finally': TokenType.FINALLY,
    # C046
    'import': TokenType.IMPORT,
    'export': TokenType.EXPORT,
    'from': TokenType.FROM,
    # C047
    'yield': TokenType.YIELD,
    # C052
    'class': TokenType.CLASS,
    'super': TokenType.SUPER,
    # C056
    'async': TokenType.ASYNC,
    'await': TokenType.AWAIT,
}


class LexError(Exception):
    pass


def lex(source: str) -> list:
    tokens = []
    i = 0
    line = 1
    while i < len(source):
        ch = source[i]

        if ch == '\n':
            line += 1
            i += 1
            continue
        if ch in ' \t\r':
            i += 1
            continue

        # Comments
        if ch == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Numbers
        if ch.isdigit():
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
            continue

        # Strings
        if ch == '"':
            i += 1
            start = i
            while i < len(source) and source[i] != '"':
                if source[i] == '\\':
                    i += 1
                i += 1
            if i >= len(source):
                raise LexError(f"Unterminated string at line {line}")
            raw = source[start:i]
            val = raw.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
            tokens.append(Token(TokenType.STRING, val, line))
            i += 1
            continue

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            # C049: f-string detection
            if word == 'f' and i < len(source) and source[i] == '"':
                i += 1  # skip opening "
                parts = []  # list of ("text", str) or ("expr", str)
                text_buf = []
                while i < len(source) and source[i] != '"':
                    # Escaped characters
                    if source[i] == '\\':
                        if i + 1 < len(source):
                            nxt = source[i + 1]
                            if nxt == '$':
                                # \$ -> literal $ (also consume { if present to escape ${)
                                text_buf.append('$')
                                i += 2
                                if i < len(source) and source[i] == '{':
                                    i += 1  # skip { after \$
                                continue
                            elif nxt == 'n':
                                text_buf.append('\n')
                                i += 2
                                continue
                            elif nxt == 't':
                                text_buf.append('\t')
                                i += 2
                                continue
                            elif nxt == '"':
                                text_buf.append('"')
                                i += 2
                                continue
                            elif nxt == '\\':
                                text_buf.append('\\')
                                i += 2
                                continue
                        text_buf.append(source[i])
                        i += 1
                        continue
                    # Interpolation: ${...}
                    if source[i] == '$' and i + 1 < len(source) and source[i + 1] == '{':
                        # Flush text buffer
                        if text_buf:
                            parts.append(("text", ''.join(text_buf)))
                            text_buf = []
                        i += 2  # skip ${
                        # Scan expression, matching braces and nested strings
                        depth = 1
                        expr_start = i
                        while i < len(source) and depth > 0:
                            c = source[i]
                            if c == '{':
                                depth += 1
                            elif c == '}':
                                depth -= 1
                                if depth == 0:
                                    break
                            elif c == '"':
                                # Skip over nested string literal
                                i += 1
                                while i < len(source) and source[i] != '"':
                                    if source[i] == '\\':
                                        i += 1
                                    i += 1
                            elif c == '\n':
                                line += 1
                            i += 1
                        expr_text = source[expr_start:i]
                        if not expr_text.strip():
                            raise LexError(f"Empty interpolation at line {line}")
                        parts.append(("expr", expr_text))
                        if i < len(source) and source[i] == '}':
                            i += 1  # skip closing }
                        else:
                            raise LexError(f"Unterminated interpolation at line {line}")
                        continue
                    # Normal character
                    if source[i] == '\n':
                        line += 1
                    text_buf.append(source[i])
                    i += 1
                if i >= len(source):
                    raise LexError(f"Unterminated f-string at line {line}")
                # Flush remaining text
                if text_buf:
                    parts.append(("text", ''.join(text_buf)))
                tokens.append(Token(TokenType.FSTRING, parts, line))
                i += 1  # skip closing "
                continue
            if word in KEYWORDS:
                tokens.append(Token(KEYWORDS[word], word, line))
            else:
                tokens.append(Token(TokenType.IDENT, word, line))
            continue

        # Three-char operators (check before two-char to avoid partial matches)
        three = source[i:i+3] if i + 2 < len(source) else ''
        if three == '??=':
            tokens.append(Token(TokenType.QUESTION_QUESTION_ASSIGN, '??=', line)); i += 3; continue
        if three == '...':
            tokens.append(Token(TokenType.DOTDOTDOT, '...', line)); i += 3; continue

        # Two-char operators
        two = source[i:i+2] if i + 1 < len(source) else ''
        if two == '??':
            tokens.append(Token(TokenType.QUESTION_QUESTION, '??', line)); i += 2; continue
        if two == '?.':
            tokens.append(Token(TokenType.QUESTION_DOT, '?.', line)); i += 2; continue
        if two == '==':
            tokens.append(Token(TokenType.EQ, '==', line)); i += 2; continue
        if two == '!=':
            tokens.append(Token(TokenType.NE, '!=', line)); i += 2; continue
        if two == '<=':
            tokens.append(Token(TokenType.LE, '<=', line)); i += 2; continue
        if two == '>=':
            tokens.append(Token(TokenType.GE, '>=', line)); i += 2; continue
        if two == '&&':
            tokens.append(Token(TokenType.AND, '&&', line)); i += 2; continue
        if two == '|>':
            tokens.append(Token(TokenType.PIPE, '|>', line)); i += 2; continue
        if two == '||':
            tokens.append(Token(TokenType.OR, '||', line)); i += 2; continue
        if two == '//':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Single-char operators/punctuation
        singles = {
            '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
            '/': TokenType.SLASH, '%': TokenType.PERCENT, '<': TokenType.LT,
            '>': TokenType.GT, '=': TokenType.ASSIGN, '!': TokenType.NOT,
            '(': TokenType.LPAREN, ')': TokenType.RPAREN,
            '{': TokenType.LBRACE, '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET, ']': TokenType.RBRACKET,
            ',': TokenType.COMMA, ';': TokenType.SEMICOLON,
            ':': TokenType.COLON, '.': TokenType.DOT,
        }
        if ch in singles:
            tokens.append(Token(singles[ch], ch, line))
            i += 1
            continue

        raise LexError(f"Unexpected character '{ch}' at line {line}")

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
class InterpolatedString:
    parts: list  # list of AST nodes (StringLit for text, expressions for ${...})
    line: int = 0

@dataclass
class IfExpr:
    cond: Any
    then_expr: Any
    else_expr: Any
    line: int = 0

@dataclass
class BoolLit:
    value: bool
    line: int = 0

@dataclass
class NullLit:
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
    else_body: Any
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
    optional: bool = False

@dataclass
class ReturnStmt:
    value: Any
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

# C042 arrays
@dataclass
class ArrayLit:
    elements: list
    line: int = 0

@dataclass
class IndexExpr:
    obj: Any
    index: Any
    line: int = 0
    optional: bool = False

@dataclass
class IndexAssign:
    obj: Any
    index: Any
    value: Any
    line: int = 0

# C043 hash maps
@dataclass
class HashLit:
    pairs: list  # [(key_expr, value_expr), ...]
    line: int = 0

@dataclass
class DotExpr:
    obj: Any
    key: str
    line: int = 0
    optional: bool = False

@dataclass
class DotAssign:
    obj: Any
    key: str
    value: Any
    line: int = 0

# C044 for-in loops
@dataclass
class ForInStmt:
    var_name: str
    var_name2: str
    iterable: Any
    body: Any
    line: int = 0

@dataclass
class BreakStmt:
    line: int = 0

@dataclass
class ContinueStmt:
    line: int = 0

# C045 error handling + C055 finally
@dataclass
class TryCatchStmt:
    try_body: Any
    catch_var: Any = None    # str or None (no catch)
    catch_body: Any = None   # Block or None (no catch)
    finally_body: Any = None # Block or None (no finally)
    line: int = 0

@dataclass
class ThrowStmt:
    value: Any
    line: int = 0

# C046 module system
@dataclass
class ImportStmt:
    module_name: str
    names: list
    line: int = 0

@dataclass
class ExportFnDecl:
    fn_decl: FnDecl
    line: int = 0

@dataclass
class ExportLetDecl:
    let_decl: LetDecl
    line: int = 0

@dataclass
class ExportClassDecl:
    class_decl: 'ClassDecl'
    line: int = 0

# C047 generators
@dataclass
class YieldExpr:
    value: Any  # expression to yield (None for bare yield)
    line: int = 0

# C050 spread operator
@dataclass
class SpreadExpr:
    """...expr in array literals, hash literals, or function calls."""
    expr: Any
    line: int = 0

# C048 destructuring
@dataclass
class PatternElement:
    """Single element in a destructuring pattern."""
    name: str           # variable name to bind (or None for nested)
    default: Any        # default value expression (or None)
    rest: bool = False  # is this a ...rest element?
    nested: Any = None  # nested ArrayPattern or HashPattern
    line: int = 0

@dataclass
class ArrayPattern:
    """[a, b, ...rest] destructuring pattern."""
    elements: list  # list of PatternElement
    line: int = 0

@dataclass
class HashPatternEntry:
    """Single entry in hash destructuring: key or key: alias."""
    key: str
    alias: str          # variable name (defaults to key)
    default: Any        # default value expression (or None)
    nested: Any = None  # nested pattern
    line: int = 0

@dataclass
class HashPattern:
    """{x, y: alias} destructuring pattern."""
    entries: list  # list of HashPatternEntry
    line: int = 0

@dataclass
class LetDestructure:
    """let [a, b] = expr; or let {x, y} = expr;"""
    pattern: Any  # ArrayPattern or HashPattern
    value: Any
    line: int = 0

@dataclass
class AssignDestructure:
    """[a, b] = expr; destructuring assignment."""
    pattern: Any  # ArrayPattern or HashPattern
    value: Any
    line: int = 0

@dataclass
class ForInDestructure:
    """for ([a, b] in iterable) { ... }"""
    pattern: Any  # ArrayPattern or HashPattern
    iterable: Any
    body: Any
    line: int = 0


# C052: Classes
@dataclass
class ClassDecl:
    name: str
    parent: str  # parent class name or None
    methods: list  # list of (name, params, body) tuples
    line: int


@dataclass
class SuperCall:
    method: str
    args: list
    line: int


# C056: Async/Await AST nodes
@dataclass
class AsyncFnDecl:
    name: str
    params: list
    body: Any
    line: int


@dataclass
class AwaitExpr:
    value: Any
    line: int


@dataclass
class ExportAsyncFnDecl:
    fn_decl: Any  # AsyncFnDecl
    line: int


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
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tt):
        tok = self.tokens[self.pos]
        if tok.type != tt:
            raise ParseError(f"Expected {tt.name}, got {tok.type.name} ({tok.value!r}) at line {tok.line}")
        self.pos += 1
        return tok

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
        if self.peek().type == TokenType.LET:
            return self.let_decl()
        if self._is_fn_decl():
            return self.fn_decl()
        if self._is_async_fn_decl():
            return self.async_fn_decl()
        if self.peek().type == TokenType.CLASS:
            return self.class_decl()
        if self.peek().type == TokenType.IMPORT:
            return self.import_stmt()
        if self.peek().type == TokenType.EXPORT:
            return self.export_decl()
        return self.statement()

    def _is_fn_decl(self):
        if self.peek().type != TokenType.FN:
            return False
        # fn name(...) or fn* name(...)
        next_pos = self.pos + 1
        if next_pos < len(self.tokens) and self.tokens[next_pos].type == TokenType.STAR:
            next_pos += 1
        return (next_pos < len(self.tokens) and
                self.tokens[next_pos].type == TokenType.IDENT)

    def _is_async_fn_decl(self):
        """Check for 'async fn name(' pattern."""
        if self.peek().type != TokenType.ASYNC:
            return False
        next_pos = self.pos + 1
        if next_pos >= len(self.tokens) or self.tokens[next_pos].type != TokenType.FN:
            return False
        next_pos += 1
        return (next_pos < len(self.tokens) and
                self.tokens[next_pos].type == TokenType.IDENT)

    def async_fn_decl(self):
        """Parse 'async fn name(...) { ... }'."""
        tok = self.advance()  # consume 'async'
        self.expect(TokenType.FN)
        name_tok = self.expect(TokenType.IDENT)
        self.expect(TokenType.LPAREN)
        params = self._parse_params()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return AsyncFnDecl(name_tok.value, params, body, tok.line)

    def import_stmt(self):
        tok = self.advance()
        if self.peek().type == TokenType.LBRACE:
            self.advance()
            names = []
            names.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RBRACE:
                    break
                names.append(self.expect(TokenType.IDENT).value)
            self.expect(TokenType.RBRACE)
            self.expect(TokenType.FROM)
            module_name = self.expect(TokenType.STRING).value
            self.expect(TokenType.SEMICOLON)
            return ImportStmt(module_name=module_name, names=names, line=tok.line)
        else:
            module_name = self.expect(TokenType.STRING).value
            self.expect(TokenType.SEMICOLON)
            return ImportStmt(module_name=module_name, names=[], line=tok.line)

    def export_decl(self):
        tok = self.advance()
        if self._is_fn_decl():
            fn = self.fn_decl()
            return ExportFnDecl(fn_decl=fn, line=tok.line)
        elif self._is_async_fn_decl():
            fn = self.async_fn_decl()
            return ExportAsyncFnDecl(fn_decl=fn, line=tok.line)
        elif self.peek().type == TokenType.LET:
            let = self.let_decl()
            return ExportLetDecl(let_decl=let, line=tok.line)
        elif self.peek().type == TokenType.CLASS:
            cls = self.class_decl()
            return ExportClassDecl(class_decl=cls, line=tok.line)
        else:
            raise ParseError(f"Expected 'fn', 'let', 'class', or 'async fn' after 'export' at line {tok.line}")

    def fn_decl(self):
        tok = self.advance()
        # fn* name() for generator syntax
        self.match(TokenType.STAR)
        name_tok = self.expect(TokenType.IDENT)
        self.expect(TokenType.LPAREN)
        params = self._parse_params()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return FnDecl(name_tok.value, params, body, tok.line)

    def class_decl(self):
        tok = self.advance()  # consume 'class'
        name = self.expect(TokenType.IDENT).value
        parent = None
        if self.match(TokenType.LT):
            parent = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LBRACE)
        methods = []
        while self.peek().type != TokenType.RBRACE:
            method_name = self.expect(TokenType.IDENT).value
            self.expect(TokenType.LPAREN)
            params = self._parse_params()
            self.expect(TokenType.RPAREN)
            body = self.block()
            methods.append((method_name, params, body))
        self.expect(TokenType.RBRACE)
        return ClassDecl(name=name, parent=parent, methods=methods, line=tok.line)

    def _parse_params(self):
        """Parse function parameters, supporting destructuring patterns."""
        params = []
        if self.peek().type == TokenType.RPAREN:
            return params
        params.append(self._parse_param())
        while self.match(TokenType.COMMA):
            params.append(self._parse_param())
        return params

    def _parse_param(self):
        """Parse a single parameter: identifier, [pattern], or {pattern}."""
        if self.peek().type == TokenType.LBRACKET:
            return self.parse_array_pattern()
        if self.peek().type == TokenType.LBRACE:
            return self.parse_hash_pattern()
        return self.expect(TokenType.IDENT).value

    def let_decl(self):
        tok = self.advance()
        # C048: destructuring patterns
        if self.peek().type == TokenType.LBRACKET:
            pattern = self.parse_array_pattern()
            self.expect(TokenType.ASSIGN)
            value = self.expression()
            self.expect(TokenType.SEMICOLON)
            return LetDestructure(pattern=pattern, value=value, line=tok.line)
        if self.peek().type == TokenType.LBRACE:
            pattern = self.parse_hash_pattern()
            self.expect(TokenType.ASSIGN)
            value = self.expression()
            self.expect(TokenType.SEMICOLON)
            return LetDestructure(pattern=pattern, value=value, line=tok.line)
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.ASSIGN)
        value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return LetDecl(name, value, tok.line)

    def parse_array_pattern(self):
        """Parse [a, b, ...rest] pattern."""
        tok = self.expect(TokenType.LBRACKET)
        elements = []
        while self.peek().type != TokenType.RBRACKET:
            if elements:
                self.expect(TokenType.COMMA)
                if self.peek().type == TokenType.RBRACKET:
                    break  # trailing comma
            if self.peek().type == TokenType.DOTDOTDOT:
                self.advance()
                name = self.expect(TokenType.IDENT).value
                elements.append(PatternElement(name=name, default=None, rest=True, line=tok.line))
                break  # rest must be last
            elif self.peek().type == TokenType.LBRACKET:
                nested = self.parse_array_pattern()
                elem = PatternElement(name=None, default=None, nested=nested, line=tok.line)
                if self.peek().type == TokenType.ASSIGN:
                    self.advance()
                    elem.default = self.expression()
                elements.append(elem)
            elif self.peek().type == TokenType.LBRACE:
                nested = self.parse_hash_pattern()
                elem = PatternElement(name=None, default=None, nested=nested, line=tok.line)
                if self.peek().type == TokenType.ASSIGN:
                    self.advance()
                    elem.default = self.expression()
                elements.append(elem)
            else:
                name = self.expect(TokenType.IDENT).value
                default = None
                if self.peek().type == TokenType.ASSIGN:
                    self.advance()
                    default = self.expression()
                elements.append(PatternElement(name=name, default=default, line=tok.line))
        self.expect(TokenType.RBRACKET)
        return ArrayPattern(elements=elements, line=tok.line)

    def parse_hash_pattern(self):
        """Parse {x, y: alias, z = default} pattern."""
        tok = self.expect(TokenType.LBRACE)
        entries = []
        while self.peek().type != TokenType.RBRACE:
            if entries:
                self.expect(TokenType.COMMA)
                if self.peek().type == TokenType.RBRACE:
                    break  # trailing comma
            if self.peek().type == TokenType.LBRACKET:
                # nested array pattern: { arr: [a, b] }
                # not valid at top level of hash pattern without key
                raise ParseError(f"Expected identifier in hash pattern at line {self.peek().line}")
            key = self.expect(TokenType.IDENT).value
            alias = key
            default = None
            nested = None
            if self.peek().type == TokenType.COLON:
                self.advance()
                if self.peek().type == TokenType.LBRACKET:
                    nested = self.parse_array_pattern()
                elif self.peek().type == TokenType.LBRACE:
                    nested = self.parse_hash_pattern()
                else:
                    alias = self.expect(TokenType.IDENT).value
            if self.peek().type == TokenType.ASSIGN:
                self.advance()
                default = self.expression()
            entries.append(HashPatternEntry(key=key, alias=alias, default=default, nested=nested, line=tok.line))
        self.expect(TokenType.RBRACE)
        return HashPattern(entries=entries, line=tok.line)

    def statement(self):
        if self.peek().type == TokenType.IF:
            return self.if_stmt()
        if self.peek().type == TokenType.WHILE:
            return self.while_stmt()
        if self.peek().type == TokenType.FOR:
            return self.for_in_stmt()
        if self.peek().type == TokenType.RETURN:
            return self.return_stmt()
        if self.peek().type == TokenType.PRINT:
            return self.print_stmt()
        if self.peek().type == TokenType.BREAK:
            return self.break_stmt()
        if self.peek().type == TokenType.CONTINUE:
            return self.continue_stmt()
        if self.peek().type == TokenType.TRY:
            return self.try_catch_stmt()
        if self.peek().type == TokenType.THROW:
            return self.throw_stmt()
        if self.peek().type == TokenType.LBRACE:
            if self._is_hash_literal():
                return self.expr_stmt()
            return self.block()
        return self.expr_stmt()

    def _is_hash_literal(self):
        if self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]
            if next_tok.type == TokenType.RBRACE:
                return True
            # C050: {...expr} is a hash literal with spread
            if next_tok.type == TokenType.DOTDOTDOT:
                return True
            if next_tok.type in (TokenType.IDENT, TokenType.STRING):
                if self.pos + 2 < len(self.tokens):
                    after = self.tokens[self.pos + 2]
                    if after.type == TokenType.COLON:
                        return True
            if next_tok.type == TokenType.INT:
                if self.pos + 2 < len(self.tokens):
                    after = self.tokens[self.pos + 2]
                    if after.type == TokenType.COLON:
                        return True
        return False

    def try_catch_stmt(self):
        tok = self.advance()
        try_body = self.block()
        catch_var = None
        catch_body = None
        finally_body = None
        if self.peek().type == TokenType.CATCH:
            self.advance()
            self.expect(TokenType.LPAREN)
            catch_var = self.expect(TokenType.IDENT).value
            self.expect(TokenType.RPAREN)
            catch_body = self.block()
        if self.peek().type == TokenType.FINALLY:
            self.advance()
            finally_body = self.block()
        if catch_body is None and finally_body is None:
            raise ParseError("try requires catch or finally", tok.line)
        return TryCatchStmt(
            try_body=try_body,
            catch_var=catch_var,
            catch_body=catch_body,
            finally_body=finally_body,
            line=tok.line,
        )

    def throw_stmt(self):
        tok = self.advance()
        value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return ThrowStmt(value=value, line=tok.line)

    def for_in_stmt(self):
        tok = self.advance()
        self.expect(TokenType.LPAREN)
        # C048: destructuring in for-in
        if self.peek().type == TokenType.LBRACKET:
            pattern = self.parse_array_pattern()
            self.expect(TokenType.IN)
            iterable = self.expression()
            self.expect(TokenType.RPAREN)
            body = self.block()
            return ForInDestructure(pattern=pattern, iterable=iterable, body=body, line=tok.line)
        if self.peek().type == TokenType.LBRACE:
            pattern = self.parse_hash_pattern()
            self.expect(TokenType.IN)
            iterable = self.expression()
            self.expect(TokenType.RPAREN)
            body = self.block()
            return ForInDestructure(pattern=pattern, iterable=iterable, body=body, line=tok.line)
        var1 = self.expect(TokenType.IDENT).value
        var2 = None
        if self.match(TokenType.COMMA):
            var2_tok = self.expect(TokenType.IDENT)
            var2 = var2_tok.value
        self.expect(TokenType.IN)
        iterable = self.expression()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return ForInStmt(var_name=var1, var_name2=var2, iterable=iterable, body=body, line=tok.line)

    def break_stmt(self):
        tok = self.advance()
        self.expect(TokenType.SEMICOLON)
        return BreakStmt(line=tok.line)

    def continue_stmt(self):
        tok = self.advance()
        self.expect(TokenType.SEMICOLON)
        return ContinueStmt(line=tok.line)

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
        if self.peek().type == TokenType.LPAREN:
            self.advance()
            value = self.expression()
            self.expect(TokenType.RPAREN)
        else:
            value = self.expression()
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
        # Check for yield expression at top level
        if self.peek().type == TokenType.YIELD:
            return self.yield_expr()
        expr = self.pipe_expr()
        if self.match(TokenType.ASSIGN):
            if isinstance(expr, Var):
                value = self.assignment()
                return Assign(expr.name, value, expr.line)
            elif isinstance(expr, IndexExpr):
                if expr.optional:
                    raise ParseError(f"Cannot assign to optional chain at line {expr.line}")
                value = self.assignment()
                return IndexAssign(expr.obj, expr.index, value, expr.line)
            elif isinstance(expr, DotExpr):
                if expr.optional:
                    raise ParseError(f"Cannot assign to optional chain at line {expr.line}")
                value = self.assignment()
                return DotAssign(expr.obj, expr.key, value, expr.line)
            # C048: destructuring assignment [a, b] = expr
            elif isinstance(expr, ArrayLit):
                pattern = self._array_lit_to_pattern(expr)
                value = self.assignment()
                return AssignDestructure(pattern=pattern, value=value, line=expr.line)
            elif isinstance(expr, HashLit):
                pattern = self._hash_lit_to_pattern(expr)
                value = self.assignment()
                return AssignDestructure(pattern=pattern, value=value, line=expr.line)
            raise ParseError(f"Invalid assignment target at line {expr.line}")
        # C054: x ??= y  ->  x = x ?? y
        if self.match(TokenType.QUESTION_QUESTION_ASSIGN):
            value = self.assignment()
            coalesced = BinOp('??', expr, value, expr.line)
            if isinstance(expr, Var):
                return Assign(expr.name, coalesced, expr.line)
            elif isinstance(expr, IndexExpr):
                if expr.optional:
                    raise ParseError(f"Cannot assign to optional chain at line {expr.line}")
                return IndexAssign(expr.obj, expr.index, coalesced, expr.line)
            elif isinstance(expr, DotExpr):
                if expr.optional:
                    raise ParseError(f"Cannot assign to optional chain at line {expr.line}")
                return DotAssign(expr.obj, expr.key, coalesced, expr.line)
            raise ParseError(f"Invalid ??= target at line {expr.line}")
        return expr

    def _array_lit_to_pattern(self, node):
        """Convert parsed ArrayLit to ArrayPattern for destructuring assignment."""
        elements = []
        for elem in node.elements:
            if isinstance(elem, SpreadExpr):
                # ...name in destructuring context -> rest pattern
                if isinstance(elem.expr, Var):
                    elements.append(PatternElement(name=elem.expr.name, default=None, rest=True, line=elem.line))
                else:
                    raise ParseError(f"Rest element must be a simple identifier at line {elem.line}")
            elif isinstance(elem, Var):
                elements.append(PatternElement(name=elem.name, default=None, line=elem.line))
            elif isinstance(elem, ArrayLit):
                nested = self._array_lit_to_pattern(elem)
                elements.append(PatternElement(name=None, default=None, nested=nested, line=elem.line))
            elif isinstance(elem, HashLit):
                nested = self._hash_lit_to_pattern(elem)
                elements.append(PatternElement(name=None, default=None, nested=nested, line=elem.line))
            else:
                raise ParseError(f"Invalid destructuring target at line {node.line}")
        return ArrayPattern(elements=elements, line=node.line)

    def _hash_lit_to_pattern(self, node):
        """Convert parsed HashLit to HashPattern for destructuring assignment."""
        entries = []
        for key_expr, val_expr in node.pairs:
            if isinstance(key_expr, Var):
                key = key_expr.name
            elif isinstance(key_expr, StringLit):
                key = key_expr.value
            else:
                raise ParseError(f"Invalid hash destructuring key at line {node.line}")
            if isinstance(val_expr, Var):
                entries.append(HashPatternEntry(key=key, alias=val_expr.name, default=None, line=node.line))
            else:
                entries.append(HashPatternEntry(key=key, alias=key, default=None, line=node.line))
        return HashPattern(entries=entries, line=node.line)

    def yield_expr(self):
        """Parse: yield expr or bare yield"""
        tok = self.advance()  # yield
        # yield can have a value or be bare (yields None)
        # bare yield: followed by ; or )
        if self.peek().type in (TokenType.SEMICOLON, TokenType.RPAREN):
            return YieldExpr(value=None, line=tok.line)
        value = self.expression()
        return YieldExpr(value=value, line=tok.line)

    def pipe_expr(self):
        """Parse: expr |> expr |> expr ... (left-associative)
        Desugars: a |> b -> CallExpr(b, [a])
                  a |> b(x) -> CallExpr(b, [a, x])  (prepend a as first arg)
        """
        left = self.or_expr()
        while self.match(TokenType.PIPE):
            right = self.or_expr()
            if isinstance(right, CallExpr):
                # a |> fn(x, y) -> fn(a, x, y)
                left = CallExpr(right.callee, [left] + right.args, left.line)
            else:
                # a |> fn -> fn(a)
                left = CallExpr(right, [left], left.line)
        return left

    def or_expr(self):
        left = self.null_coalesce_expr()
        while self.match(TokenType.OR):
            right = self.null_coalesce_expr()
            left = BinOp('or', left, right, left.line)
        return left

    def null_coalesce_expr(self):
        """Parse: a ?? b ?? c (left-associative). Returns a if not null, else b."""
        left = self.and_expr()
        while self.match(TokenType.QUESTION_QUESTION):
            right = self.and_expr()
            left = BinOp('??', left, right, left.line)
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
        if self.match(TokenType.AWAIT):
            tok = self.tokens[self.pos - 1]
            return AwaitExpr(value=self.unary(), line=tok.line)
        return self.postfix()

    def postfix(self):
        expr = self.primary()
        while True:
            if self.match(TokenType.LPAREN):
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self._parse_call_arg())
                    while self.match(TokenType.COMMA):
                        args.append(self._parse_call_arg())
                self.expect(TokenType.RPAREN)
                expr = CallExpr(expr, args, expr.line)
            elif self.match(TokenType.LBRACKET):
                index = self.expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexExpr(expr, index, expr.line)
            elif self.match(TokenType.DOT):
                name_tok = self.expect(TokenType.IDENT)
                expr = DotExpr(expr, name_tok.value, expr.line)
            elif self.match(TokenType.QUESTION_DOT):
                # Optional chaining: ?.prop, ?.[expr], ?.(args)
                if self.match(TokenType.LPAREN):
                    # Optional call: obj?.(args)
                    args = []
                    if self.peek().type != TokenType.RPAREN:
                        args.append(self._parse_call_arg())
                        while self.match(TokenType.COMMA):
                            args.append(self._parse_call_arg())
                    self.expect(TokenType.RPAREN)
                    expr = CallExpr(expr, args, expr.line, optional=True)
                elif self.match(TokenType.LBRACKET):
                    # Optional index: obj?.[expr]
                    index = self.expression()
                    self.expect(TokenType.RBRACKET)
                    expr = IndexExpr(expr, index, expr.line, optional=True)
                else:
                    # Optional dot: obj?.prop
                    name_tok = self.expect(TokenType.IDENT)
                    expr = DotExpr(expr, name_tok.value, expr.line, optional=True)
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
        if tok.type == TokenType.FSTRING:
            self.advance()
            return self._parse_fstring(tok)
        if tok.type == TokenType.TRUE:
            self.advance()
            return BoolLit(True, tok.line)
        if tok.type == TokenType.FALSE:
            self.advance()
            return BoolLit(False, tok.line)
        if tok.type == TokenType.NULL:
            self.advance()
            return NullLit(tok.line)
        if tok.type == TokenType.IDENT:
            self.advance()
            return Var(tok.value, tok.line)
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN)
            return expr
        if tok.type == TokenType.LBRACKET:
            return self._parse_array_literal()
        if tok.type == TokenType.LBRACE:
            return self._parse_hash_literal()
        if tok.type == TokenType.FN:
            return self._parse_lambda()
        if tok.type == TokenType.IF:
            return self._parse_if_expr()
        if tok.type == TokenType.SUPER:
            return self._parse_super_call()
        raise ParseError(f"Unexpected token {tok.type.name} ({tok.value!r}) at line {tok.line}")

    def _parse_array_literal(self):
        tok = self.advance()
        elements = []
        if self.peek().type != TokenType.RBRACKET:
            elements.append(self._parse_array_element())
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RBRACKET:
                    break
                elements.append(self._parse_array_element())
        self.expect(TokenType.RBRACKET)
        return ArrayLit(elements, tok.line)

    def _parse_array_element(self):
        """Parse a single array element, which may be ...expr (spread)."""
        if self.peek().type == TokenType.DOTDOTDOT:
            tok = self.advance()
            expr = self.expression()
            return SpreadExpr(expr=expr, line=tok.line)
        return self.expression()

    def _parse_hash_key(self):
        tok = self.peek()
        if tok.type == TokenType.IDENT and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.COLON:
            self.advance()
            return StringLit(tok.value, tok.line)
        return self.expression()

    def _parse_hash_literal(self):
        tok = self.advance()
        pairs = []  # list of (key, value) tuples or SpreadExpr
        if self.peek().type != TokenType.RBRACE:
            pairs.append(self._parse_hash_entry())
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RBRACE:
                    break
                pairs.append(self._parse_hash_entry())
        self.expect(TokenType.RBRACE)
        return HashLit(pairs, tok.line)

    def _parse_hash_entry(self):
        """Parse a single hash entry: key: value or ...expr (spread)."""
        if self.peek().type == TokenType.DOTDOTDOT:
            tok = self.advance()
            expr = self.expression()
            return SpreadExpr(expr=expr, line=tok.line)
        key = self._parse_hash_key()
        self.expect(TokenType.COLON)
        value = self.expression()
        return (key, value)

    def _parse_lambda(self):
        tok = self.advance()  # consume 'fn'
        self.match(TokenType.STAR)  # consume optional '*' for generators
        self.expect(TokenType.LPAREN)
        params = self._parse_params()
        self.expect(TokenType.RPAREN)
        body = self.block()
        return LambdaExpr(params, body, tok.line)

    def _parse_if_expr(self):
        """Parse if (cond) expr else expr as an expression."""
        tok = self.advance()  # consume 'if'
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        self.expect(TokenType.RPAREN)
        then_expr = self.expression()
        self.expect(TokenType.ELSE)
        else_expr = self.expression()
        return IfExpr(cond=cond, then_expr=then_expr, else_expr=else_expr, line=tok.line)

    def _parse_super_call(self):
        """Parse super.method(args)."""
        tok = self.advance()  # consume 'super'
        self.expect(TokenType.DOT)
        method = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self._parse_call_arg())
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RPAREN:
                    break
                args.append(self._parse_call_arg())
        self.expect(TokenType.RPAREN)
        return SuperCall(method=method, args=args, line=tok.line)

    def _parse_call_arg(self):
        """Parse a single call argument, which may be ...expr (spread)."""
        if self.peek().type == TokenType.DOTDOTDOT:
            tok = self.advance()
            expr = self.expression()
            return SpreadExpr(expr=expr, line=tok.line)
        return self.expression()

    def _parse_fstring(self, tok):
        """Parse an f-string token into an InterpolatedString AST node."""
        parts_raw = tok.value  # list of ("text", str) or ("expr", str)
        if not parts_raw:
            return StringLit("", tok.line)
        # Optimization: if only one text part, return plain StringLit
        if len(parts_raw) == 1 and parts_raw[0][0] == "text":
            return StringLit(parts_raw[0][1], tok.line)
        parts = []
        for kind, text in parts_raw:
            if kind == "text":
                parts.append(StringLit(text, tok.line))
            else:
                # Parse the expression text
                expr_tokens = lex(text)
                sub_parser = Parser(expr_tokens)
                expr = sub_parser.expression()
                parts.append(expr)
        return InterpolatedString(parts, tok.line)


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
    is_generator: bool = False
    is_async: bool = False


@dataclass
class ClosureObject:
    fn: FnObject
    env: dict


# C047: Generator runtime object
@dataclass
class GeneratorObject:
    fn: FnObject
    env: dict
    ip: int = 0
    stack: list = field(default_factory=list)
    call_stack: list = field(default_factory=list)
    handler_stack: list = field(default_factory=list)
    done: bool = False
    finally_pending: Any = None  # C055: saved _finally_pending state


# C056: Promise runtime object
class PromiseObject:
    PENDING = 'pending'
    RESOLVED = 'resolved'
    REJECTED = 'rejected'

    def __init__(self):
        self.state = self.PENDING
        self.value = None
        self.callbacks = []  # list of (on_resolve, on_reject) tuples
        self.chained = []  # list of (next_promise, on_resolve, on_reject)

    def resolve(self, value):
        if self.state != self.PENDING:
            return
        self.state = self.RESOLVED
        self.value = value
        self._flush_callbacks()
        self._flush_chained()

    def reject(self, reason):
        if self.state != self.PENDING:
            return
        self.state = self.REJECTED
        self.value = reason
        self._flush_callbacks()
        self._flush_chained()

    def _flush_callbacks(self):
        for on_resolve, on_reject in self.callbacks:
            if self.state == self.RESOLVED and on_resolve:
                on_resolve(self.value)
            elif self.state == self.REJECTED and on_reject:
                on_reject(self.value)
        self.callbacks.clear()

    def _flush_chained(self):
        for next_promise, on_resolve, on_reject in self.chained:
            if self.state == self.RESOLVED and on_resolve:
                try:
                    result = on_resolve(self.value)
                    if isinstance(result, PromiseObject):
                        result.chained.append((next_promise, lambda v: next_promise.resolve(v), lambda e: next_promise.reject(e)))
                        if result.state == PromiseObject.RESOLVED:
                            next_promise.resolve(result.value)
                        elif result.state == PromiseObject.REJECTED:
                            next_promise.reject(result.value)
                    else:
                        next_promise.resolve(result)
                except Exception as e:
                    next_promise.reject(str(e))
            elif self.state == self.REJECTED and on_reject:
                try:
                    result = on_reject(self.value)
                    next_promise.resolve(result)
                except Exception as e:
                    next_promise.reject(str(e))
            elif self.state == self.REJECTED:
                next_promise.reject(self.value)
            elif self.state == self.RESOLVED:
                next_promise.resolve(self.value)
        self.chained.clear()


# C056: Async coroutine (suspended async function)
@dataclass
class AsyncCoroutine:
    fn: FnObject
    env: dict
    ip: int = 0
    stack: list = field(default_factory=list)
    call_stack: list = field(default_factory=list)
    handler_stack: list = field(default_factory=list)
    done: bool = False
    finally_pending: Any = None
    promise: Any = None  # The PromiseObject this coroutine will resolve


# C052: Class runtime objects
@dataclass
class ClassObject:
    name: str
    methods: dict  # name -> FnObject
    parent: Any = None  # ClassObject or None


@dataclass
class BoundMethod:
    instance: dict  # the instance
    method: Any  # FnObject or ClosureObject
    klass: Any  # ClassObject where method was found


# Builtin function names
BUILTINS = {
    'len', 'push', 'pop', 'map', 'filter', 'reduce', 'range',
    'slice', 'concat', 'sort', 'reverse', 'find', 'each',
    # C043 hash map builtins
    'keys', 'values', 'has', 'delete', 'merge', 'entries', 'size',
    # C045 error handling builtins
    'type', 'string',
    # C047 generator builtins
    'next',
    # C048 destructuring builtins
    '__slice_from',
    # C052 class builtins
    'instanceof',
}


def _ast_contains_yield(node):
    """Check if an AST node contains any yield expression."""
    if isinstance(node, YieldExpr):
        return True
    if isinstance(node, Block):
        return any(_ast_contains_yield(s) for s in node.stmts)
    if isinstance(node, IfStmt):
        if _ast_contains_yield(node.cond):
            return True
        if _ast_contains_yield(node.then_body):
            return True
        if node.else_body and _ast_contains_yield(node.else_body):
            return True
        return False
    if isinstance(node, WhileStmt):
        return _ast_contains_yield(node.cond) or _ast_contains_yield(node.body)
    if isinstance(node, ForInStmt):
        return _ast_contains_yield(node.body)
    if isinstance(node, TryCatchStmt):
        result = _ast_contains_yield(node.try_body)
        if node.catch_body and _ast_contains_yield(node.catch_body):
            result = True
        if node.finally_body and _ast_contains_yield(node.finally_body):
            result = True
        return result
    if isinstance(node, LetDecl):
        return _ast_contains_yield(node.value)
    if isinstance(node, ReturnStmt):
        return node.value is not None and _ast_contains_yield(node.value)
    if isinstance(node, PrintStmt):
        return _ast_contains_yield(node.value)
    if isinstance(node, ThrowStmt):
        return _ast_contains_yield(node.value)
    if isinstance(node, Assign):
        return _ast_contains_yield(node.value)
    if isinstance(node, BinOp):
        return _ast_contains_yield(node.left) or _ast_contains_yield(node.right)
    if isinstance(node, UnaryOp):
        return _ast_contains_yield(node.operand)
    if isinstance(node, CallExpr):
        if _ast_contains_yield(node.callee):
            return True
        return any(_ast_contains_yield(a) for a in node.args)
    if isinstance(node, IndexExpr):
        return _ast_contains_yield(node.obj) or _ast_contains_yield(node.index)
    if isinstance(node, IndexAssign):
        return _ast_contains_yield(node.obj) or _ast_contains_yield(node.index) or _ast_contains_yield(node.value)
    if isinstance(node, SpreadExpr):
        return _ast_contains_yield(node.expr)
    if isinstance(node, ArrayLit):
        return any(_ast_contains_yield(e) for e in node.elements)
    if isinstance(node, HashLit):
        for p in node.pairs:
            if isinstance(p, SpreadExpr):
                if _ast_contains_yield(p.expr):
                    return True
            else:
                if _ast_contains_yield(p[0]) or _ast_contains_yield(p[1]):
                    return True
        return False
    if isinstance(node, DotExpr):
        return _ast_contains_yield(node.obj)
    if isinstance(node, DotAssign):
        return _ast_contains_yield(node.obj) or _ast_contains_yield(node.value)
    # FnDecl/LambdaExpr: yield inside a nested function does NOT make the outer one a generator
    # ExportFnDecl/ExportLetDecl: check inner
    if isinstance(node, ExportFnDecl):
        return False  # nested function
    if isinstance(node, ExportLetDecl):
        return _ast_contains_yield(node.let_decl)
    # C048 destructuring
    if isinstance(node, LetDestructure):
        return _ast_contains_yield(node.value)
    if isinstance(node, AssignDestructure):
        return _ast_contains_yield(node.value)
    if isinstance(node, ForInDestructure):
        return _ast_contains_yield(node.body)
    if isinstance(node, ClassDecl):
        return False  # class body is its own scope
    if isinstance(node, SuperCall):
        return any(_ast_contains_yield(a) for a in node.args)
    if isinstance(node, AsyncFnDecl):
        return False  # async function body is its own scope
    if isinstance(node, ExportAsyncFnDecl):
        return False
    if isinstance(node, AwaitExpr):
        return _ast_contains_yield(node.value)
    return False


class Compiler:
    def __init__(self):
        self.chunk = Chunk()
        self.functions = {}
        self.loop_stack = []
        self.exports = []

    def compile(self, program: Program) -> Chunk:
        for stmt in program.stmts:
            self.compile_node(stmt)
        self.chunk.emit(Op.HALT)
        return self.chunk

    def compile_node(self, node):
        method_name = f'compile_{type(node).__name__}'
        method = getattr(self, method_name, None)
        if method is None:
            raise CompileError(f"Cannot compile {type(node).__name__}")
        method(node)

    def compile_IntLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_FloatLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_StringLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_InterpolatedString(self, node):
        if not node.parts:
            idx = self.chunk.add_constant("")
            self.chunk.emit(Op.CONST, idx, node.line)
            return
        # Compile first part
        self._compile_interp_part(node.parts[0], node.line)
        # Compile remaining parts and concatenate
        for part in node.parts[1:]:
            self._compile_interp_part(part, node.line)
            self.chunk.emit(Op.ADD, line=node.line)

    def _compile_interp_part(self, part, line):
        """Compile one part of an interpolated string, coercing to string if needed."""
        if isinstance(part, StringLit):
            self.compile_node(part)
        else:
            # Emit: LOAD string, <expr>, CALL 1
            # This puts string_fn on stack, then value, then calls string(value)
            str_idx = self.chunk.add_name("string")
            self.chunk.emit(Op.LOAD, str_idx, line)
            self.compile_node(part)
            self.chunk.emit(Op.CALL, 1, line)

    def compile_BoolLit(self, node):
        idx = self.chunk.add_constant(node.value)
        self.chunk.emit(Op.CONST, idx, node.line)

    def compile_NullLit(self, node):
        idx = self.chunk.add_constant(None)
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
        self.compile_node(node.left)
        self.compile_node(node.right)
        ops = {
            '+': Op.ADD, '-': Op.SUB, '*': Op.MUL, '/': Op.DIV, '%': Op.MOD,
            '==': Op.EQ, '!=': Op.NE, '<': Op.LT, '>': Op.GT,
            '<=': Op.LE, '>=': Op.GE,
            'and': Op.AND, 'or': Op.OR,
            '??': Op.NULL_COALESCE,
        }
        if node.op in ops:
            self.chunk.emit(ops[node.op], line=node.line)
        else:
            raise CompileError(f"Unknown binary operator: {node.op}")

    def compile_Assign(self, node):
        self.compile_node(node.value)
        idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.DUP, line=node.line)
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

    def compile_IfExpr(self, node):
        self.compile_node(node.cond)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.then_expr)
        jump_end = self.chunk.emit(Op.JUMP, 0, node.line)
        self.chunk.patch(jump_false + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.else_expr)
        self.chunk.patch(jump_end + 1, len(self.chunk.code))

    def compile_WhileStmt(self, node):
        loop_start = len(self.chunk.code)
        break_patches = []
        continue_patches = []
        self.loop_stack.append((break_patches, continue_patches, loop_start))

        self.compile_node(node.cond)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, node.line)
        self.chunk.emit(Op.POP, line=node.line)
        self.compile_node(node.body)
        self.chunk.emit(Op.JUMP, loop_start, node.line)
        loop_end = len(self.chunk.code)
        self.chunk.patch(jump_false + 1, loop_end)
        self.chunk.emit(Op.POP, line=node.line)

        for bp in break_patches:
            self.chunk.patch(bp + 1, loop_end)
        for cp in continue_patches:
            self.chunk.patch(cp + 1, loop_start)
        self.loop_stack.pop()

    def compile_ForInStmt(self, node):
        line = node.line
        iter_name = f'__iter_{id(node)}'
        keys_name = f'__keys_{id(node)}'
        idx_name = f'__idx_{id(node)}'
        len_name = f'__len_{id(node)}'

        iter_idx = self.chunk.add_name(iter_name)
        keys_idx = self.chunk.add_name(keys_name)
        idx_idx = self.chunk.add_name(idx_name)
        len_idx = self.chunk.add_name(len_name)

        var_idx = self.chunk.add_name(node.var_name)
        var2_idx = self.chunk.add_name(node.var_name2) if node.var_name2 else None

        self.compile_node(node.iterable)
        self.chunk.emit(Op.DUP, line=line)
        self.chunk.emit(Op.STORE, iter_idx, line)
        self.chunk.emit(Op.ITER_PREPARE, line=line)
        self.chunk.emit(Op.STORE, keys_idx, line)

        zero_idx = self.chunk.add_constant(0)
        self.chunk.emit(Op.CONST, zero_idx, line)
        self.chunk.emit(Op.STORE, idx_idx, line)

        self.chunk.emit(Op.LOAD, keys_idx, line)
        self.chunk.emit(Op.ITER_LENGTH, line=line)
        self.chunk.emit(Op.STORE, len_idx, line)

        loop_start = len(self.chunk.code)

        break_patches = []
        continue_patches = []
        self.loop_stack.append((break_patches, continue_patches, None))

        self.chunk.emit(Op.LOAD, idx_idx, line)
        self.chunk.emit(Op.LOAD, len_idx, line)
        self.chunk.emit(Op.LT, line=line)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
        self.chunk.emit(Op.POP, line=line)

        if node.var_name2:
            self.chunk.emit(Op.LOAD, keys_idx, line)
            self.chunk.emit(Op.LOAD, idx_idx, line)
            self.chunk.emit(Op.INDEX_GET, line=line)
            self.chunk.emit(Op.DUP, line=line)
            self.chunk.emit(Op.STORE, var_idx, line)

            self.chunk.emit(Op.LOAD, iter_idx, line)
            self.chunk.emit(Op.LOAD, var_idx, line)
            self.chunk.emit(Op.INDEX_GET, line=line)
            self.chunk.emit(Op.STORE, var2_idx, line)

            self.chunk.emit(Op.POP, line=line)
        else:
            self.chunk.emit(Op.LOAD, keys_idx, line)
            self.chunk.emit(Op.LOAD, idx_idx, line)
            self.chunk.emit(Op.INDEX_GET, line=line)
            self.chunk.emit(Op.STORE, var_idx, line)

        self.compile_node(node.body)

        # Increment index
        inc_start = len(self.chunk.code)
        one_idx = self.chunk.add_constant(1)
        self.chunk.emit(Op.LOAD, idx_idx, line)
        self.chunk.emit(Op.CONST, one_idx, line)
        self.chunk.emit(Op.ADD, line=line)
        self.chunk.emit(Op.STORE, idx_idx, line)

        self.chunk.emit(Op.JUMP, loop_start, line)
        loop_end = len(self.chunk.code)
        self.chunk.patch(jump_false + 1, loop_end)
        self.chunk.emit(Op.POP, line=line)

        for bp in break_patches:
            self.chunk.patch(bp + 1, loop_end)
        for cp in continue_patches:
            self.chunk.patch(cp + 1, inc_start)
        self.loop_stack.pop()

    def compile_ForInDestructure(self, node):
        """Compile for ([a, b] in iterable) { ... } -- destructuring for-in."""
        line = node.line
        iter_name = f'__iter_{id(node)}'
        keys_name = f'__keys_{id(node)}'
        idx_name = f'__idx_{id(node)}'
        len_name = f'__len_{id(node)}'

        iter_idx = self.chunk.add_name(iter_name)
        keys_idx = self.chunk.add_name(keys_name)
        idx_idx = self.chunk.add_name(idx_name)
        len_idx = self.chunk.add_name(len_name)

        self.compile_node(node.iterable)
        self.chunk.emit(Op.DUP, line=line)
        self.chunk.emit(Op.STORE, iter_idx, line)
        self.chunk.emit(Op.ITER_PREPARE, line=line)
        self.chunk.emit(Op.STORE, keys_idx, line)

        zero_idx = self.chunk.add_constant(0)
        self.chunk.emit(Op.CONST, zero_idx, line)
        self.chunk.emit(Op.STORE, idx_idx, line)

        self.chunk.emit(Op.LOAD, keys_idx, line)
        self.chunk.emit(Op.ITER_LENGTH, line=line)
        self.chunk.emit(Op.STORE, len_idx, line)

        loop_start = len(self.chunk.code)

        break_patches = []
        continue_patches = []
        self.loop_stack.append((break_patches, continue_patches, None))

        self.chunk.emit(Op.LOAD, idx_idx, line)
        self.chunk.emit(Op.LOAD, len_idx, line)
        self.chunk.emit(Op.LT, line=line)
        jump_false = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
        self.chunk.emit(Op.POP, line=line)

        # Get the current element
        self.chunk.emit(Op.LOAD, keys_idx, line)
        self.chunk.emit(Op.LOAD, idx_idx, line)
        self.chunk.emit(Op.INDEX_GET, line=line)

        # Destructure the element into pattern bindings
        self._compile_pattern_destructure(node.pattern, line)

        self.compile_node(node.body)

        # Increment index
        inc_start = len(self.chunk.code)
        one_idx = self.chunk.add_constant(1)
        self.chunk.emit(Op.LOAD, idx_idx, line)
        self.chunk.emit(Op.CONST, one_idx, line)
        self.chunk.emit(Op.ADD, line=line)
        self.chunk.emit(Op.STORE, idx_idx, line)

        self.chunk.emit(Op.JUMP, loop_start, line)
        loop_end = len(self.chunk.code)
        self.chunk.patch(jump_false + 1, loop_end)
        self.chunk.emit(Op.POP, line=line)

        for bp in break_patches:
            self.chunk.patch(bp + 1, loop_end)
        for cp in continue_patches:
            self.chunk.patch(cp + 1, inc_start)
        self.loop_stack.pop()

    def compile_BreakStmt(self, node):
        if not self.loop_stack:
            raise CompileError("'break' outside of loop")
        break_patches, _, _ = self.loop_stack[-1]
        addr = self.chunk.emit(Op.JUMP, 0, node.line)
        break_patches.append(addr)

    def compile_ContinueStmt(self, node):
        if not self.loop_stack:
            raise CompileError("'continue' outside of loop")
        _, continue_patches, _ = self.loop_stack[-1]
        addr = self.chunk.emit(Op.JUMP, 0, node.line)
        continue_patches.append(addr)

    def compile_TryCatchStmt(self, node):
        line = node.line
        has_catch = node.catch_body is not None
        has_finally = node.finally_body is not None

        if has_finally:
            setup_finally_addr = self.chunk.emit(Op.SETUP_FINALLY, 0, line)

        if has_catch:
            setup_try_addr = self.chunk.emit(Op.SETUP_TRY, 0, line)

        # -- try body --
        self.compile_node(node.try_body)

        if has_catch:
            self.chunk.emit(Op.POP_TRY, line=line)

        if has_catch:
            jump_after_catch = self.chunk.emit(Op.JUMP, 0, line)
            # -- catch body --
            catch_addr = len(self.chunk.code)
            self.chunk.patch(setup_try_addr + 1, catch_addr)
            catch_var_idx = self.chunk.add_name(node.catch_var)
            self.chunk.emit(Op.STORE, catch_var_idx, line)
            self.compile_node(node.catch_body)
            # after_catch:
            self.chunk.patch(jump_after_catch + 1, len(self.chunk.code))

        if has_finally:
            # Normal path: pop finally handler, run finally inline, jump to end
            self.chunk.emit(Op.POP_FINALLY, line=line)
            self.compile_node(node.finally_body)
            jump_end = self.chunk.emit(Op.JUMP, 0, line)

            # Abnormal path: FinallyHandler caught exception or intercepted return
            # _finally_pending holds the pending action, stack is clean
            finally_addr = len(self.chunk.code)
            self.chunk.patch(setup_finally_addr + 1, finally_addr)
            self.compile_node(node.finally_body)
            self.chunk.emit(Op.END_FINALLY, line=line)

            # end:
            self.chunk.patch(jump_end + 1, len(self.chunk.code))
        else:
            # No finally -- original simple try/catch behavior
            pass

    def compile_ThrowStmt(self, node):
        self.compile_node(node.value)
        self.chunk.emit(Op.THROW, line=node.line)

    # C046 module system
    def compile_ImportStmt(self, node):
        pass

    def compile_ExportFnDecl(self, node):
        self.compile_node(node.fn_decl)
        self.exports.append(node.fn_decl.name)

    def compile_ExportLetDecl(self, node):
        self.compile_node(node.let_decl)
        if isinstance(node.let_decl, LetDestructure):
            # Export all names bound by the destructuring pattern
            for name in self._pattern_names(node.let_decl.pattern):
                self.exports.append(name)
        else:
            self.exports.append(node.let_decl.name)

    def _pattern_names(self, pattern):
        """Extract all variable names from a destructuring pattern."""
        names = []
        if isinstance(pattern, ArrayPattern):
            for elem in pattern.elements:
                if elem.nested:
                    names.extend(self._pattern_names(elem.nested))
                elif elem.name:
                    names.append(elem.name)
        elif isinstance(pattern, HashPattern):
            for entry in pattern.entries:
                if entry.nested:
                    names.extend(self._pattern_names(entry.nested))
                else:
                    names.append(entry.alias)
        return names

    # C047 generators
    def compile_YieldExpr(self, node):
        """Compile yield expr -- emits YIELD opcode.
        The value to yield is pushed on stack, YIELD suspends execution.
        When resumed, the value passed to next() is left on the stack."""
        if node.value is not None:
            self.compile_node(node.value)
        else:
            idx = self.chunk.add_constant(None)
            self.chunk.emit(Op.CONST, idx, node.line)
        self.chunk.emit(Op.YIELD, line=node.line)

    # C048 destructuring compilation
    def compile_LetDestructure(self, node):
        """Compile: let [a, b] = expr; or let {x, y} = expr;"""
        self.compile_node(node.value)
        self._compile_pattern_destructure(node.pattern)

    def compile_AssignDestructure(self, node):
        """Compile: [a, b] = [b, a]; destructuring assignment."""
        self.compile_node(node.value)
        self._compile_pattern_destructure(node.pattern)

    def _compile_pattern_destructure(self, pattern, line=0):
        """Emit bytecode to destructure the value on top of stack into pattern bindings.
        After this, the original value is consumed from the stack."""
        if isinstance(pattern, ArrayPattern):
            self._compile_array_destructure(pattern)
        elif isinstance(pattern, HashPattern):
            self._compile_hash_destructure(pattern)

    def _compile_array_destructure(self, pattern):
        """Destructure array: stack top is array value.
        Store in temp, then LOAD temp for each element access."""
        line = pattern.line
        temp_name = f'__destr_{id(pattern)}'
        temp_idx = self.chunk.add_name(temp_name)
        self.chunk.emit(Op.STORE, temp_idx, line)

        for i, elem in enumerate(pattern.elements):
            if elem.rest:
                # ...rest: call __slice_from(arr, start_index)
                slice_fn_idx = self.chunk.add_name('__slice_from')
                self.chunk.emit(Op.LOAD, slice_fn_idx, line)
                self.chunk.emit(Op.LOAD, temp_idx, line)
                start_const = self.chunk.add_constant(i)
                self.chunk.emit(Op.CONST, start_const, line)
                self.chunk.emit(Op.CALL, 2, line)
                name_idx = self.chunk.add_name(elem.name)
                self.chunk.emit(Op.STORE, name_idx, line)
            elif elem.nested:
                self.chunk.emit(Op.LOAD, temp_idx, line)
                idx_const = self.chunk.add_constant(i)
                self.chunk.emit(Op.CONST, idx_const, line)
                self.chunk.emit(Op.INDEX_GET, line=line)
                if elem.default is not None:
                    self._emit_default_check(elem.default, line)
                self._compile_pattern_destructure(elem.nested, line)
            else:
                if elem.default is not None:
                    # Safe access: check bounds first, use None if out of bounds
                    self._emit_safe_array_access(temp_idx, i, elem.default, line)
                    name_idx = self.chunk.add_name(elem.name)
                    self.chunk.emit(Op.STORE, name_idx, line)
                else:
                    self.chunk.emit(Op.LOAD, temp_idx, line)
                    idx_const = self.chunk.add_constant(i)
                    self.chunk.emit(Op.CONST, idx_const, line)
                    self.chunk.emit(Op.INDEX_GET, line=line)
                    name_idx = self.chunk.add_name(elem.name)
                    self.chunk.emit(Op.STORE, name_idx, line)

    def _emit_safe_array_access(self, arr_temp_idx, index, default_expr, line):
        """Emit: if index < len(arr), arr[index]; else default_value.
        Handles out-of-bounds gracefully for destructuring defaults."""
        # Load len(arr)
        len_fn_idx = self.chunk.add_name('len')
        self.chunk.emit(Op.LOAD, len_fn_idx, line)
        self.chunk.emit(Op.LOAD, arr_temp_idx, line)
        self.chunk.emit(Op.CALL, 1, line)
        # Stack: [arr_len]
        idx_const = self.chunk.add_constant(index)
        self.chunk.emit(Op.CONST, idx_const, line)
        # Stack: [arr_len, index]
        self.chunk.emit(Op.GT, line=line)
        # Stack: [arr_len > index] -- true if element exists
        jump_to_default = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
        self.chunk.emit(Op.POP, line=line)  # pop condition
        # Element exists: load it
        self.chunk.emit(Op.LOAD, arr_temp_idx, line)
        idx_const2 = self.chunk.add_constant(index)
        self.chunk.emit(Op.CONST, idx_const2, line)
        self.chunk.emit(Op.INDEX_GET, line=line)
        # Check if the value is None (for explicit None values)
        self._emit_default_check(default_expr, line)
        jump_end = self.chunk.emit(Op.JUMP, 0, line)
        # Default branch
        self.chunk.patch(jump_to_default + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=line)  # pop condition
        self.compile_node(default_expr)
        self.chunk.patch(jump_end + 1, len(self.chunk.code))

    def _compile_hash_destructure(self, pattern):
        """Destructure hash: stack top is hash value.
        Store in temp, then LOAD temp for each key access."""
        line = pattern.line
        temp_name = f'__destr_{id(pattern)}'
        temp_idx = self.chunk.add_name(temp_name)
        self.chunk.emit(Op.STORE, temp_idx, line)

        for entry in pattern.entries:
            if entry.default is not None:
                # Safe access: check if key exists first
                self._emit_safe_hash_access(temp_idx, entry.key, entry.default, line)
            else:
                self.chunk.emit(Op.LOAD, temp_idx, line)
                key_const = self.chunk.add_constant(entry.key)
                self.chunk.emit(Op.CONST, key_const, line)
                self.chunk.emit(Op.INDEX_GET, line=line)
            if entry.nested:
                self._compile_pattern_destructure(entry.nested, line)
            else:
                name_idx = self.chunk.add_name(entry.alias)
                self.chunk.emit(Op.STORE, name_idx, line)

    def _emit_safe_hash_access(self, hash_temp_idx, key, default_expr, line):
        """Emit: if key exists in hash, hash[key]; else default_value."""
        has_fn_idx = self.chunk.add_name('has')
        self.chunk.emit(Op.LOAD, has_fn_idx, line)
        self.chunk.emit(Op.LOAD, hash_temp_idx, line)
        key_const = self.chunk.add_constant(key)
        self.chunk.emit(Op.CONST, key_const, line)
        self.chunk.emit(Op.CALL, 2, line)
        # Stack: [true/false]
        jump_to_default = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
        self.chunk.emit(Op.POP, line=line)  # pop condition
        # Key exists: load it
        self.chunk.emit(Op.LOAD, hash_temp_idx, line)
        key_const2 = self.chunk.add_constant(key)
        self.chunk.emit(Op.CONST, key_const2, line)
        self.chunk.emit(Op.INDEX_GET, line=line)
        # Also check for None value
        self._emit_default_check(default_expr, line)
        jump_end = self.chunk.emit(Op.JUMP, 0, line)
        # Default branch
        self.chunk.patch(jump_to_default + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=line)  # pop condition
        self.compile_node(default_expr)
        self.chunk.patch(jump_end + 1, len(self.chunk.code))

    def _emit_default_check(self, default_expr, line):
        """Emit: if top of stack is None, replace with default value."""
        # Stack: [value]
        # DUP, CONST None, NE -> if true (not None), jump over default
        self.chunk.emit(Op.DUP, line=line)
        none_idx = self.chunk.add_constant(None)
        self.chunk.emit(Op.CONST, none_idx, line)
        self.chunk.emit(Op.NE, line=line)
        jump_skip = self.chunk.emit(Op.JUMP_IF_TRUE, 0, line)
        self.chunk.emit(Op.POP, line=line)  # pop NE result (false)
        self.chunk.emit(Op.POP, line=line)  # pop None value
        self.compile_node(default_expr)
        jump_end = self.chunk.emit(Op.JUMP, 0, line)
        self.chunk.patch(jump_skip + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=line)  # pop NE result (true)
        self.chunk.patch(jump_end + 1, len(self.chunk.code))

    def _compile_function_body(self, params, body):
        fn_compiler = self.__class__()
        fn_compiler.chunk = Chunk()

        # Register param names; pattern params get synthetic temp names
        param_patterns = []  # (index, pattern) pairs for destructuring
        for i, p in enumerate(params):
            if isinstance(p, str):
                fn_compiler.chunk.add_name(p)
            else:
                # Pattern parameter -- use synthetic name
                temp_name = f'__destruct_param_{i}'
                fn_compiler.chunk.add_name(temp_name)
                param_patterns.append((i, temp_name, p))

        # Emit destructuring for pattern params at the start
        for _, temp_name, pattern in param_patterns:
            temp_idx = fn_compiler.chunk.add_name(temp_name)
            fn_compiler.chunk.emit(Op.LOAD, temp_idx)
            fn_compiler._compile_pattern_destructure(pattern)

        if isinstance(body, Block):
            for stmt in body.stmts:
                fn_compiler.compile_node(stmt)
        else:
            fn_compiler.compile_node(body)

        # Implicit return None
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx)
        fn_compiler.chunk.emit(Op.RETURN)
        return fn_compiler

    def compile_FnDecl(self, node):
        fn_compiler = self._compile_function_body(node.params, node.body)

        # Check if this function contains yield
        is_gen = _ast_contains_yield(node.body)

        fn_obj = FnObject(node.name, len(node.params), fn_compiler.chunk, is_generator=is_gen)
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

        is_gen = _ast_contains_yield(node.body)

        fn_obj = FnObject("<lambda>", len(node.params), fn_compiler.chunk, is_generator=is_gen)
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v

        fn_idx = self.chunk.add_constant(fn_obj)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.MAKE_CLOSURE, line=node.line)

    def compile_AsyncFnDecl(self, node):
        fn_compiler = self._compile_function_body(node.params, node.body)
        fn_obj = FnObject(node.name, len(node.params), fn_compiler.chunk, is_async=True)
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v
        self.functions[node.name] = fn_obj

        fn_idx = self.chunk.add_constant(fn_obj)
        name_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.MAKE_CLOSURE, line=node.line)
        self.chunk.emit(Op.STORE, name_idx, node.line)

    def compile_ExportAsyncFnDecl(self, node):
        self.compile_AsyncFnDecl(node.fn_decl)
        self.exports.append(node.fn_decl.name)

    def compile_AwaitExpr(self, node):
        self.compile_node(node.value)
        self.chunk.emit(Op.AWAIT, line=node.line)

    def compile_CallExpr(self, node):
        has_spread = any(isinstance(a, SpreadExpr) for a in node.args)
        if node.optional:
            # Optional call: obj?.(args) -- if obj is null, result is null
            self.compile_node(node.callee)
            null_label = self._emit_optional_null_check(node.line)
            if not has_spread:
                for arg in node.args:
                    self.compile_node(arg)
                self.chunk.emit(Op.CALL, len(node.args), node.line)
            else:
                self._compile_spread_call_args(node)
            self._patch_optional_end(null_label, node.line)
        elif not has_spread:
            self.compile_node(node.callee)
            for arg in node.args:
                self.compile_node(arg)
            self.chunk.emit(Op.CALL, len(node.args), node.line)
        else:
            # Build args array using ARRAY_SPREAD, then CALL_SPREAD
            self.compile_node(node.callee)
            self._compile_spread_call_args(node)

    def _compile_spread_call_args(self, node):
        """Helper for spread call arguments."""
        self.chunk.emit(Op.MAKE_ARRAY, 0, node.line)
        group = []
        for arg in node.args:
            if isinstance(arg, SpreadExpr):
                if group:
                    for g in group:
                        self.compile_node(g)
                    self.chunk.emit(Op.MAKE_ARRAY, len(group), node.line)
                    self.chunk.emit(Op.ARRAY_SPREAD, line=node.line)
                    group = []
                self.compile_node(arg.expr)
                self.chunk.emit(Op.ARRAY_SPREAD, line=node.line)
            else:
                group.append(arg)
        if group:
            for g in group:
                self.compile_node(g)
            self.chunk.emit(Op.MAKE_ARRAY, len(group), node.line)
            self.chunk.emit(Op.ARRAY_SPREAD, line=node.line)
        self.chunk.emit(Op.CALL_SPREAD, line=node.line)

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

    def compile_ArrayLit(self, node):
        has_spread = any(isinstance(e, SpreadExpr) for e in node.elements)
        if not has_spread:
            for elem in node.elements:
                self.compile_node(elem)
            self.chunk.emit(Op.MAKE_ARRAY, len(node.elements), node.line)
        else:
            # Start with empty array, then extend with groups/spreads
            self.chunk.emit(Op.MAKE_ARRAY, 0, node.line)
            group = []
            for elem in node.elements:
                if isinstance(elem, SpreadExpr):
                    if group:
                        for g in group:
                            self.compile_node(g)
                        self.chunk.emit(Op.MAKE_ARRAY, len(group), node.line)
                        self.chunk.emit(Op.ARRAY_SPREAD, line=node.line)
                        group = []
                    self.compile_node(elem.expr)
                    self.chunk.emit(Op.ARRAY_SPREAD, line=node.line)
                else:
                    group.append(elem)
            if group:
                for g in group:
                    self.compile_node(g)
                self.chunk.emit(Op.MAKE_ARRAY, len(group), node.line)
                self.chunk.emit(Op.ARRAY_SPREAD, line=node.line)

    def _emit_optional_null_check(self, line):
        """Emit null check for optional chaining. Returns (null_label, end_label).
        Stack before: [obj]. After: [obj] with jump to null_label if obj is None.
        Caller must POP the comparison result on both branches."""
        self.chunk.emit(Op.DUP, line=line)
        none_idx = self.chunk.add_constant(None)
        self.chunk.emit(Op.CONST, none_idx, line)
        self.chunk.emit(Op.EQ, line=line)
        null_label = self.chunk.emit(Op.JUMP_IF_TRUE, 0, line)
        # Fall-through: obj is not None. Pop the comparison result (False).
        self.chunk.emit(Op.POP, line=line)
        return null_label

    def _patch_optional_end(self, null_label, line):
        """Emit the end of an optional chain expression.
        After the non-null path, jump over the null path."""
        end_label = self.chunk.emit(Op.JUMP, 0, line)
        # null_label target: obj is None, pop comparison result (True), leave None on stack
        self.chunk.patch(null_label + 1, len(self.chunk.code))
        self.chunk.emit(Op.POP, line=line)
        # None (the obj) is already on the stack
        self.chunk.patch(end_label + 1, len(self.chunk.code))

    def compile_IndexExpr(self, node):
        self.compile_node(node.obj)
        if node.optional:
            null_label = self._emit_optional_null_check(node.line)
            self.compile_node(node.index)
            self.chunk.emit(Op.INDEX_GET, line=node.line)
            self._patch_optional_end(null_label, node.line)
        else:
            self.compile_node(node.index)
            self.chunk.emit(Op.INDEX_GET, line=node.line)

    def compile_IndexAssign(self, node):
        self.compile_node(node.obj)
        self.compile_node(node.index)
        self.compile_node(node.value)
        self.chunk.emit(Op.INDEX_SET, line=node.line)

    def compile_HashLit(self, node):
        has_spread = any(isinstance(p, SpreadExpr) for p in node.pairs)
        if not has_spread:
            for key_expr, value_expr in node.pairs:
                self.compile_node(key_expr)
                self.compile_node(value_expr)
            self.chunk.emit(Op.MAKE_HASH, len(node.pairs), node.line)
        else:
            # Start with empty hash, then merge with groups/spreads
            self.chunk.emit(Op.MAKE_HASH, 0, node.line)
            group = []
            for pair in node.pairs:
                if isinstance(pair, SpreadExpr):
                    if group:
                        for k, v in group:
                            self.compile_node(k)
                            self.compile_node(v)
                        self.chunk.emit(Op.MAKE_HASH, len(group), node.line)
                        self.chunk.emit(Op.HASH_SPREAD, line=node.line)
                        group = []
                    self.compile_node(pair.expr)
                    self.chunk.emit(Op.HASH_SPREAD, line=node.line)
                else:
                    group.append(pair)
            if group:
                for k, v in group:
                    self.compile_node(k)
                    self.compile_node(v)
                self.chunk.emit(Op.MAKE_HASH, len(group), node.line)
                self.chunk.emit(Op.HASH_SPREAD, line=node.line)

    def compile_DotExpr(self, node):
        self.compile_node(node.obj)
        if node.optional:
            null_label = self._emit_optional_null_check(node.line)
            idx = self.chunk.add_constant(node.key)
            self.chunk.emit(Op.CONST, idx, node.line)
            self.chunk.emit(Op.INDEX_GET, line=node.line)
            self._patch_optional_end(null_label, node.line)
        else:
            idx = self.chunk.add_constant(node.key)
            self.chunk.emit(Op.CONST, idx, node.line)
            self.chunk.emit(Op.INDEX_GET, line=node.line)

    def compile_DotAssign(self, node):
        self.compile_node(node.obj)
        idx = self.chunk.add_constant(node.key)
        self.chunk.emit(Op.CONST, idx, node.line)
        self.compile_node(node.value)
        self.chunk.emit(Op.INDEX_SET, line=node.line)

    # C052: Classes
    def compile_ClassDecl(self, node):
        line = node.line

        # Push class name
        name_const = self.chunk.add_constant(node.name)
        self.chunk.emit(Op.CONST, name_const, line)

        # Push parent (None or class reference)
        if node.parent:
            parent_idx = self.chunk.add_name(node.parent)
            self.chunk.emit(Op.LOAD, parent_idx, line)
        else:
            none_idx = self.chunk.add_constant(None)
            self.chunk.emit(Op.CONST, none_idx, line)

        # Compile each method
        for method_name, params, body in node.methods:
            # Methods get implicit 'this' as first param
            all_params = ['this'] + list(params)
            fn_compiler = self._compile_function_body(all_params, body)

            is_gen = _ast_contains_yield(body)
            fn_obj = FnObject(method_name, len(all_params), fn_compiler.chunk,
                              is_generator=is_gen)
            for k, v in fn_compiler.functions.items():
                self.functions[k] = v

            # Push method name string
            mn_idx = self.chunk.add_constant(method_name)
            self.chunk.emit(Op.CONST, mn_idx, line)
            # Push fn object
            fn_idx = self.chunk.add_constant(fn_obj)
            self.chunk.emit(Op.CONST, fn_idx, line)

        # MAKE_CLASS operand: method_count
        self.chunk.emit(Op.MAKE_CLASS, len(node.methods), line)

        # Store class in variable
        cls_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.STORE, cls_idx, line)

    def compile_SuperCall(self, node):
        line = node.line
        # Load parent class
        super_idx = self.chunk.add_name('__super__')
        self.chunk.emit(Op.LOAD, super_idx, line)
        # Push this
        this_idx = self.chunk.add_name('this')
        self.chunk.emit(Op.LOAD, this_idx, line)
        # Push args
        for arg in node.args:
            self.compile_node(arg)
        # SUPER_INVOKE: method_name_idx, arg_count (not including this)
        method_name_idx = self.chunk.add_name(node.method)
        self.chunk.emit(Op.SUPER_INVOKE, method_name_idx, line)
        # Second operand: arg count
        self.chunk.code.append(len(node.args))
        self.chunk.lines.append(line)

    def compile_ExportClassDecl(self, node):
        self.compile_ClassDecl(node.class_decl)
        self.exports.append(node.class_decl.name)


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


@dataclass
class TryHandler:
    catch_addr: int
    catch_chunk: Any
    call_depth: int
    stack_depth: int
    env: dict


@dataclass
class FinallyHandler:
    finally_addr: int
    finally_chunk: Any
    call_depth: int
    stack_depth: int
    env: dict


# Sentinel for generator exhaustion
_GENERATOR_DONE = object()


def _format_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        parts = [_format_value(v) for v in value]
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, ClassObject):
        return f"<class:{value.name}>"
    if isinstance(value, BoundMethod):
        fn = value.method
        name = fn.name if isinstance(fn, FnObject) else fn.fn.name
        return f"<method:{name}>"
    if isinstance(value, dict):
        if '__class__' in value:
            cls = value['__class__']
            props = {k: v for k, v in value.items() if k != '__class__'}
            parts = [f"{k}: {_format_value(v)}" for k, v in props.items()]
            return f"<{cls.name} {{{', '.join(parts)}}}>"
        parts = [f"{_format_value(k)}: {_format_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, ClosureObject):
        return f"<closure:{value.fn.name}>"
    if isinstance(value, FnObject):
        return f"<fn:{value.fn.name}>" if hasattr(value, 'fn') else f"<fn:{value.name}>"
    if isinstance(value, GeneratorObject):
        status = "done" if value.done else "suspended"
        return f"<generator:{value.fn.name}:{status}>"
    if isinstance(value, PromiseObject):
        return f"<promise:{value.state}>"
    if isinstance(value, AsyncCoroutine):
        status = "done" if value.done else "suspended"
        return f"<async:{value.fn.name}:{status}>"
    return str(value)


class VM:
    def __init__(self, chunk: Chunk, trace=False):
        self.chunk = chunk
        self.stack = []
        self.env = {}
        self.call_stack = []
        self.handler_stack = []
        self.output = []
        self.trace = trace
        self.ip = 0
        self.current_chunk = chunk
        self.step_count = 0
        self.max_steps = 100000
        self._finally_pending = None  # None | ('throw', value) | ('return', value)
        # C056: Async support
        self._async_queue = []  # list of (AsyncCoroutine, value, error) pending resume
        self._current_async = None  # AsyncCoroutine being executed, or None
        # Promise namespace -- accessible as 'Promise' variable
        self.env['Promise'] = self._make_promise_namespace()

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

    def _lookup_method(self, klass, name):
        """Look up a method in the class chain. Returns (fn, class_where_found) or None."""
        while klass is not None:
            if name in klass.methods:
                return (klass.methods[name], klass)
            klass = klass.parent
        return None

    def _throw(self, value):
        if not self.handler_stack:
            raise VMError(f"Uncaught exception: {_format_value(value)}")

        handler = self.handler_stack.pop()

        while len(self.call_stack) > handler.call_depth:
            self.call_stack.pop()

        if isinstance(handler, FinallyHandler):
            # Route through finally block, then re-throw
            self._finally_pending = ('throw', value)
            self.current_chunk = handler.finally_chunk
            self.ip = handler.finally_addr
            # Don't restore env -- finally should see current variable state
            # (call_stack unwinding already handles scope changes)
            del self.stack[handler.stack_depth:]
        else:
            # Normal TryHandler -- jump to catch
            self.current_chunk = handler.catch_chunk
            self.ip = handler.catch_addr
            self.env = handler.env
            del self.stack[handler.stack_depth:]
            self.push(value)

    def _route_return_through_finally(self, return_val, call_depth):
        """Check for FinallyHandler at current call depth. If found, route through it."""
        # Scan from top of handler_stack for FinallyHandler at this depth
        handlers_to_remove = []
        found_finally = None
        for i in range(len(self.handler_stack) - 1, -1, -1):
            h = self.handler_stack[i]
            if h.call_depth < call_depth:
                break  # handler belongs to caller, stop
            if h.call_depth == call_depth and isinstance(h, FinallyHandler):
                found_finally = (i, h)
                break
            handlers_to_remove.append(i)

        if found_finally is None:
            return False

        idx, handler = found_finally
        # Pop all handlers from top down to and including the FinallyHandler
        while len(self.handler_stack) > idx:
            self.handler_stack.pop()

        self._finally_pending = ('return', return_val)
        self.current_chunk = handler.finally_chunk
        self.ip = handler.finally_addr
        # Don't restore env -- finally should see current variable state
        del self.stack[handler.stack_depth:]
        return True

    def _vm_error_to_throw(self, msg):
        if self.handler_stack:
            self._throw(msg)
            return True
        return False

    def _call_builtin(self, name, args):
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

        # C045 utility builtins
        elif name == 'type':
            if len(args) != 1:
                raise VMError(f"type() takes 1 argument, got {len(args)}")
            val = args[0]
            if val is None:
                return "null"
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
            if isinstance(val, ClassObject):
                return "class"
            if isinstance(val, BoundMethod):
                return "function"
            if isinstance(val, dict):
                if '__class__' in val:
                    return val['__class__'].name
                return "hash"
            if isinstance(val, (ClosureObject, FnObject)):
                return "function"
            if isinstance(val, GeneratorObject):
                return "generator"
            if isinstance(val, PromiseObject):
                return "promise"
            if isinstance(val, AsyncCoroutine):
                return "async"
            return "unknown"

        # C052 class builtins
        elif name == 'instanceof':
            if len(args) != 2:
                raise VMError(f"instanceof() takes 2 arguments, got {len(args)}")
            obj, cls = args
            if not isinstance(cls, ClassObject):
                raise VMError(f"instanceof() second argument must be a class")
            if not isinstance(obj, dict) or '__class__' not in obj:
                return False
            klass = obj['__class__']
            while klass is not None:
                if klass is cls:
                    return True
                klass = klass.parent
            return False

        elif name == 'string':
            if len(args) != 1:
                raise VMError(f"string() takes 1 argument, got {len(args)}")
            return _format_value(args[0])

        # C047 generator builtin
        elif name == 'next':
            if len(args) < 1 or len(args) > 2:
                raise VMError(f"next() takes 1-2 arguments, got {len(args)}")
            gen = args[0]
            if not isinstance(gen, GeneratorObject):
                raise VMError(f"next() requires generator, got {type(gen).__name__}")
            default = args[1] if len(args) == 2 else _GENERATOR_DONE
            return self._resume_generator(gen, default)

        # C048 destructuring builtins
        elif name == '__slice_from':
            if len(args) != 2:
                raise VMError(f"__slice_from() takes 2 arguments, got {len(args)}")
            arr, start = args
            if not isinstance(arr, list):
                raise VMError(f"__slice_from() requires array as first argument")
            return arr[int(start):]

        else:
            raise VMError(f"Unknown builtin: {name}")

    def _create_generator(self, fn_obj, captured_env, args):
        """Create a GeneratorObject from a generator function call."""
        # Set up initial env with parameters
        env = dict(captured_env) if captured_env is not None else dict(self.env)
        for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
            env[param_name] = args[i]

        gen = GeneratorObject(
            fn=fn_obj,
            env=env,
            ip=0,
            stack=[],
            call_stack=[],
            handler_stack=[],
            done=False,
        )
        return gen

    def _resume_generator(self, gen, default=_GENERATOR_DONE):
        """Resume a generator, running until next yield or return.
        Returns the yielded value, or default if generator is done.
        If no default and generator is done, returns None."""
        if gen.done:
            if default is not _GENERATOR_DONE:
                return default
            return None

        # Save current VM state
        saved_chunk = self.current_chunk
        saved_ip = self.ip
        saved_env = self.env
        saved_stack = self.stack
        saved_call_stack = self.call_stack
        saved_handler_stack = self.handler_stack
        saved_finally_pending = self._finally_pending

        # Restore generator state
        self.current_chunk = gen.fn.chunk
        self.ip = gen.ip
        self.env = gen.env
        self.stack = gen.stack
        self.call_stack = gen.call_stack
        self.handler_stack = gen.handler_stack
        self._finally_pending = gen.finally_pending

        result = None
        yielded = False

        try:
            while True:
                self.step_count += 1
                if self.step_count > self.max_steps:
                    raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

                if self.ip >= len(self.current_chunk.code):
                    # Generator finished by reaching end of code
                    gen.done = True
                    result = None
                    break

                op = self.current_chunk.code[self.ip]
                self.ip += 1

                if op == Op.YIELD:
                    # Pop the value to yield
                    result = self.pop()
                    yielded = True
                    break

                if op == Op.RETURN:
                    # Generator returns -- mark as done
                    result_val = self.pop()
                    if self.call_stack:
                        # Return from nested function call within generator
                        cur_depth = len(self.call_stack)
                        if self._route_return_through_finally(result_val, cur_depth):
                            continue  # routed to finally, keep running
                        frame = self.call_stack.pop()
                        self.current_chunk = frame.chunk
                        self.ip = frame.ip
                        self.env = frame.base_env
                        self.push(result_val)
                        continue
                    else:
                        # Return from generator itself -- check for finally
                        if self._route_return_through_finally(result_val, 0):
                            continue  # routed to finally, keep running
                        gen.done = True
                        result = None  # Generator return value is discarded
                        break

                exec_result = self._execute_op(op)
                if exec_result == 'halt':
                    gen.done = True
                    result = None
                    break

        finally:
            # Save generator state
            gen.ip = self.ip
            gen.env = self.env
            gen.stack = self.stack
            gen.call_stack = self.call_stack
            gen.handler_stack = self.handler_stack
            gen.finally_pending = self._finally_pending

            # Restore VM state
            self.current_chunk = saved_chunk
            self.ip = saved_ip
            self.env = saved_env
            self.stack = saved_stack
            self.call_stack = saved_call_stack
            self.handler_stack = saved_handler_stack
            self._finally_pending = saved_finally_pending

        if gen.done and not yielded:
            if default is not _GENERATOR_DONE:
                return default
            return None

        return result

    def _start_async(self, fn_obj, captured_env, args):
        """Start an async function. Returns a PromiseObject."""
        env = dict(captured_env) if captured_env is not None else dict(self.env)
        for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
            env[param_name] = args[i]

        promise = PromiseObject()
        coro = AsyncCoroutine(
            fn=fn_obj,
            env=env,
            ip=0,
            stack=[],
            call_stack=[],
            handler_stack=[],
            done=False,
            promise=promise,
        )

        # Eagerly run until first await or completion
        self._resume_async(coro, None)
        return promise

    def _resume_async(self, coro, send_value):
        """Resume an async coroutine. Runs until AWAIT, RETURN, or error."""
        if coro.done:
            return

        # Save current VM state
        saved_chunk = self.current_chunk
        saved_ip = self.ip
        saved_env = self.env
        saved_stack = self.stack
        saved_call_stack = self.call_stack
        saved_handler_stack = self.handler_stack
        saved_finally_pending = self._finally_pending
        saved_async = self._current_async

        # Restore coroutine state
        self.current_chunk = coro.fn.chunk
        self.ip = coro.ip
        self.env = coro.env
        self.stack = coro.stack
        self.call_stack = coro.call_stack
        self.handler_stack = coro.handler_stack
        self._finally_pending = coro.finally_pending
        self._current_async = coro

        # If resuming from an await, replace the promise on the stack with the resolved value
        if send_value is not None or (coro.ip > 0):
            # Pop the promise that was left on stack by AWAIT
            if self.stack:
                self.stack.pop()
            self.push(send_value)

        suspended = False
        try:
            while True:
                self.step_count += 1
                if self.step_count > self.max_steps:
                    raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

                if self.ip >= len(self.current_chunk.code):
                    coro.done = True
                    coro.promise.resolve(None)
                    break

                op = self.current_chunk.code[self.ip]
                self.ip += 1

                if op == Op.RETURN:
                    return_val = self.pop()
                    if self.call_stack:
                        cur_depth = len(self.call_stack)
                        if self._route_return_through_finally(return_val, cur_depth):
                            continue
                        frame = self.call_stack.pop()
                        self.current_chunk = frame.chunk
                        self.ip = frame.ip
                        self.env = frame.base_env
                        self.push(return_val)
                        continue
                    else:
                        if self._route_return_through_finally(return_val, 0):
                            continue
                        coro.done = True
                        coro.promise.resolve(return_val)
                        break

                # Handle THROW explicitly -- if no handler in coroutine, reject promise
                if op == Op.THROW:
                    thrown_value = self.pop()
                    if self.handler_stack:
                        self._throw(thrown_value)
                        continue
                    else:
                        # No handler -- reject the promise with the thrown value
                        coro.done = True
                        coro.promise.reject(thrown_value)
                        break

                exec_result = self._execute_op(op)
                if exec_result == 'halt':
                    coro.done = True
                    result_val = self.stack.pop() if self.stack else None
                    coro.promise.resolve(result_val)
                    break
                if isinstance(exec_result, tuple) and exec_result[0] == 'async_reject':
                    coro.done = True
                    coro.promise.reject(exec_result[1])
                    break
                if exec_result == 'await_suspend':
                    # Coroutine hit an await on a pending promise
                    awaited_promise = self.peek()  # promise is on top of stack
                    suspended = True

                    # When the awaited promise resolves, schedule coroutine resumption
                    def make_resume(c=coro):
                        def on_resolve(val):
                            self._async_queue.append((c, val, None))
                        def on_reject(err):
                            self._async_queue.append((c, None, err))
                        return on_resolve, on_reject

                    on_res, on_rej = make_resume()

                    # Direct callback registration
                    if awaited_promise.state == PromiseObject.PENDING:
                        awaited_promise.callbacks.append((on_res, on_rej))
                    elif awaited_promise.state == PromiseObject.RESOLVED:
                        self._async_queue.append((coro, awaited_promise.value, None))
                    else:
                        self._async_queue.append((coro, None, awaited_promise.value))
                    break
                if exec_result == 'finally_returned':
                    result_val = self.stack.pop() if self.stack else None
                    coro.done = True
                    coro.promise.resolve(result_val)
                    break

        except VMError as e:
            coro.done = True
            coro.promise.reject(str(e))

        finally:
            # Save coroutine state
            coro.ip = self.ip
            coro.env = self.env
            coro.stack = self.stack
            coro.call_stack = self.call_stack
            coro.handler_stack = self.handler_stack
            coro.finally_pending = self._finally_pending

            # Restore VM state
            self.current_chunk = saved_chunk
            self.ip = saved_ip
            self.env = saved_env
            self.stack = saved_stack
            self.call_stack = saved_call_stack
            self.handler_stack = saved_handler_stack
            self._finally_pending = saved_finally_pending
            self._current_async = saved_async

    def _make_promise_namespace(self):
        """Create the Promise namespace object with resolve/reject/all/race methods."""
        return {
            '__promise_ns__': True,
            'resolve': ('__promise_builtin__', 'resolve'),
            'reject': ('__promise_builtin__', 'reject'),
            'all': ('__promise_builtin__', 'all'),
            'race': ('__promise_builtin__', 'race'),
        }

    def _call_promise_builtin(self, method, args):
        """Handle Promise.resolve/reject/all/race calls."""
        if method == 'resolve':
            if len(args) != 1:
                raise VMError(f"Promise.resolve() takes 1 argument, got {len(args)}")
            p = PromiseObject()
            p.resolve(args[0])
            return p
        elif method == 'reject':
            if len(args) != 1:
                raise VMError(f"Promise.reject() takes 1 argument, got {len(args)}")
            p = PromiseObject()
            p.reject(args[0])
            return p
        elif method == 'all':
            if len(args) != 1 or not isinstance(args[0], list):
                raise VMError("Promise.all() takes an array of promises")
            promises = args[0]
            result_promise = PromiseObject()
            if not promises:
                result_promise.resolve([])
                return result_promise
            results = [None] * len(promises)
            resolved_count = [0]

            for i, p in enumerate(promises):
                if not isinstance(p, PromiseObject):
                    # Non-promise values are treated as resolved
                    results[i] = p
                    resolved_count[0] += 1
                    if resolved_count[0] == len(promises):
                        result_promise.resolve(list(results))
                elif p.state == PromiseObject.RESOLVED:
                    results[i] = p.value
                    resolved_count[0] += 1
                    if resolved_count[0] == len(promises):
                        result_promise.resolve(list(results))
                elif p.state == PromiseObject.REJECTED:
                    result_promise.reject(p.value)
                    return result_promise
                else:
                    idx = i
                    def make_handler(idx=idx):
                        def on_resolve(val):
                            results[idx] = val
                            resolved_count[0] += 1
                            if resolved_count[0] == len(promises):
                                result_promise.resolve(list(results))
                        def on_reject(err):
                            result_promise.reject(err)
                        return on_resolve, on_reject
                    on_res, on_rej = make_handler()
                    p.callbacks.append((on_res, on_rej))
            return result_promise
        elif method == 'race':
            if len(args) != 1 or not isinstance(args[0], list):
                raise VMError("Promise.race() takes an array of promises")
            promises = args[0]
            result_promise = PromiseObject()
            if not promises:
                return result_promise  # never resolves
            for p in promises:
                if not isinstance(p, PromiseObject):
                    result_promise.resolve(p)
                    return result_promise
                if p.state == PromiseObject.RESOLVED:
                    result_promise.resolve(p.value)
                    return result_promise
                if p.state == PromiseObject.REJECTED:
                    result_promise.reject(p.value)
                    return result_promise
                def on_resolve(val, rp=result_promise):
                    rp.resolve(val)
                def on_reject(err, rp=result_promise):
                    rp.reject(err)
                p.callbacks.append((on_resolve, on_reject))
            return result_promise
        else:
            raise VMError(f"Promise.{method} is not a function")

    def _drain_async_queue(self):
        """Process all pending async continuations until the queue is empty."""
        max_iterations = 10000
        iterations = 0
        while self._async_queue:
            iterations += 1
            if iterations > max_iterations:
                raise VMError("Async queue exceeded maximum iterations (possible infinite loop)")
            coro, value, error = self._async_queue.pop(0)
            if coro.done:
                continue
            if error is not None:
                # Resume with an error -- need to throw in the coroutine
                self._resume_async_with_error(coro, error)
            else:
                self._resume_async(coro, value)

    def _resume_async_with_error(self, coro, error):
        """Resume an async coroutine by throwing an error into it."""
        if coro.done:
            return

        # Save current VM state
        saved_chunk = self.current_chunk
        saved_ip = self.ip
        saved_env = self.env
        saved_stack = self.stack
        saved_call_stack = self.call_stack
        saved_handler_stack = self.handler_stack
        saved_finally_pending = self._finally_pending
        saved_async = self._current_async

        # Restore coroutine state
        self.current_chunk = coro.fn.chunk
        self.ip = coro.ip
        self.env = coro.env
        self.stack = coro.stack
        self.call_stack = coro.call_stack
        self.handler_stack = coro.handler_stack
        self._finally_pending = coro.finally_pending
        self._current_async = coro

        # Pop the promise left on the stack by AWAIT
        if self.stack:
            self.stack.pop()

        try:
            # Try to throw the error -- if handler exists, continue execution
            if self.handler_stack:
                self._throw(error)
                # Continue execution after throwing
                while True:
                    self.step_count += 1
                    if self.step_count > self.max_steps:
                        raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

                    if self.ip >= len(self.current_chunk.code):
                        coro.done = True
                        coro.promise.resolve(None)
                        break

                    op = self.current_chunk.code[self.ip]
                    self.ip += 1

                    if op == Op.RETURN:
                        return_val = self.pop()
                        if self.call_stack:
                            cur_depth = len(self.call_stack)
                            if self._route_return_through_finally(return_val, cur_depth):
                                continue
                            frame = self.call_stack.pop()
                            self.current_chunk = frame.chunk
                            self.ip = frame.ip
                            self.env = frame.base_env
                            self.push(return_val)
                            continue
                        else:
                            if self._route_return_through_finally(return_val, 0):
                                continue
                            coro.done = True
                            coro.promise.resolve(return_val)
                            break

                    # Handle THROW explicitly in async context
                    if op == Op.THROW:
                        thrown_value = self.pop()
                        if self.handler_stack:
                            self._throw(thrown_value)
                            continue
                        else:
                            coro.done = True
                            coro.promise.reject(thrown_value)
                            break

                    exec_result = self._execute_op(op)
                    if exec_result == 'halt':
                        coro.done = True
                        result_val = self.stack.pop() if self.stack else None
                        coro.promise.resolve(result_val)
                        break
                    if isinstance(exec_result, tuple) and exec_result[0] == 'async_reject':
                        coro.done = True
                        coro.promise.reject(exec_result[1])
                        break
                    if exec_result == 'await_suspend':
                        awaited_promise = self.peek()
                        def make_resume(c=coro):
                            def on_resolve(val):
                                self._async_queue.append((c, val, None))
                            def on_reject(err):
                                self._async_queue.append((c, None, err))
                            return on_resolve, on_reject
                        on_res, on_rej = make_resume()
                        if awaited_promise.state == PromiseObject.PENDING:
                            awaited_promise.callbacks.append((on_res, on_rej))
                        elif awaited_promise.state == PromiseObject.RESOLVED:
                            self._async_queue.append((coro, awaited_promise.value, None))
                        else:
                            self._async_queue.append((coro, None, awaited_promise.value))
                        break
                    if exec_result == 'finally_returned':
                        result_val = self.stack.pop() if self.stack else None
                        coro.done = True
                        coro.promise.resolve(result_val)
                        break
            else:
                # No handler -- reject the promise
                coro.done = True
                coro.promise.reject(error)

        except VMError as e:
            coro.done = True
            coro.promise.reject(str(e))

        finally:
            # Save coroutine state
            coro.ip = self.ip
            coro.env = self.env
            coro.stack = self.stack
            coro.call_stack = self.call_stack
            coro.handler_stack = self.handler_stack
            coro.finally_pending = self._finally_pending

            # Restore VM state
            self.current_chunk = saved_chunk
            self.ip = saved_ip
            self.env = saved_env
            self.stack = saved_stack
            self.call_stack = saved_call_stack
            self.handler_stack = saved_handler_stack
            self._finally_pending = saved_finally_pending
            self._current_async = saved_async

    def _call_function(self, fn_val, args):
        # C052: Handle BoundMethod
        method_class = None
        if isinstance(fn_val, BoundMethod):
            args = [fn_val.instance] + list(args)
            method_class = fn_val.klass
            fn_val = fn_val.method

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

        # Generator functions return a GeneratorObject instead of executing
        if fn_obj.is_generator:
            return self._create_generator(fn_obj, captured_env, args)

        # C056: Async functions return a Promise and start running as a coroutine
        if fn_obj.is_async:
            return self._start_async(fn_obj, captured_env, args)

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

        # C052: Inject __super__ for methods
        if method_class and method_class.parent:
            self.env['__super__'] = method_class.parent

        result = self._run_until_return(saved_call_stack_depth)

        self.current_chunk = saved_chunk
        self.ip = saved_ip
        self.env = saved_env

        return result

    def _run_until_return(self, base_depth):
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
                    # Check for FinallyHandler before exiting
                    if self._route_return_through_finally(return_val, len(self.call_stack)):
                        continue  # routed to finally, keep running
                    return return_val
                cur_depth = len(self.call_stack)
                if self._route_return_through_finally(return_val, cur_depth):
                    continue  # routed to finally
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                self.env = frame.base_env
                self.push(return_val)
                continue

            exec_result = self._execute_op(op)
            if exec_result == 'halt':
                return self.stack[-1] if self.stack else None
            # C055: END_FINALLY may have completed a pending return
            if exec_result == 'finally_returned':
                return self.stack.pop() if self.stack else None

    def _execute_op(self, op):
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
            self.push(a * b)
        elif op == Op.DIV:
            b, a = self.pop(), self.pop()
            if b == 0:
                if self._vm_error_to_throw("Division by zero"):
                    return
                raise VMError("Division by zero")
            if isinstance(a, int) and isinstance(b, int):
                self.push(a // b)
            else:
                self.push(a / b)
        elif op == Op.MOD:
            b, a = self.pop(), self.pop()
            if b == 0:
                if self._vm_error_to_throw("Modulo by zero"):
                    return
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

        elif op == Op.NULL_COALESCE:
            b, a = self.pop(), self.pop()
            self.push(a if a is not None else b)

        elif op == Op.LOAD:
            idx = self.current_chunk.code[self.ip]
            self.ip += 1
            name = self.current_chunk.names[idx]
            if name in self.env:
                self.push(self.env[name])
            elif name in BUILTINS:
                self.push(('__builtin__', name))
            else:
                if self._vm_error_to_throw(f"Undefined variable '{name}'"):
                    return
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
                try:
                    result = self._call_builtin(fn_val[1], args)
                    self.push(result)
                except VMError as e:
                    if self._vm_error_to_throw(str(e)):
                        return
                    raise
                return

            # C056: Promise builtin call
            if isinstance(fn_val, tuple) and len(fn_val) == 2 and fn_val[0] == '__promise_builtin__':
                try:
                    result = self._call_promise_builtin(fn_val[1], args)
                    self.push(result)
                except VMError as e:
                    if self._vm_error_to_throw(str(e)):
                        return
                    raise
                return

            # C052: ClassObject constructor call
            if isinstance(fn_val, ClassObject):
                instance = {'__class__': fn_val}
                init_result = self._lookup_method(fn_val, 'init')
                if init_result:
                    init_fn, init_class = init_result
                    expected = init_fn.arity - 1  # subtract 'this'
                    if expected != arg_count:
                        if self._vm_error_to_throw(
                            f"Class '{fn_val.name}' init expects {expected} args, got {arg_count}"):
                            return
                        raise VMError(
                            f"Class '{fn_val.name}' init expects {expected} args, got {arg_count}")
                    # Synchronous call to init
                    saved_chunk = self.current_chunk
                    saved_ip = self.ip
                    saved_env = self.env
                    saved_depth = len(self.call_stack)
                    saved_stack_depth = len(self.stack)

                    self.current_chunk = init_fn.chunk
                    self.ip = 0
                    self.env = dict(self.env)

                    # Bind params: this + user args
                    all_args = [instance] + list(args)
                    for i, pname in enumerate(init_fn.chunk.names[:init_fn.arity]):
                        self.env[pname] = all_args[i]

                    # Inject __super__
                    if init_class.parent:
                        self.env['__super__'] = init_class.parent

                    self._run_until_return(saved_depth)

                    # Clean up any extra stack values left by init body
                    del self.stack[saved_stack_depth:]

                    self.current_chunk = saved_chunk
                    self.ip = saved_ip
                    self.env = saved_env
                elif arg_count > 0:
                    if self._vm_error_to_throw(
                        f"Class '{fn_val.name}' has no init but received {arg_count} args"):
                        return
                    raise VMError(
                        f"Class '{fn_val.name}' has no init but received {arg_count} args")
                self.push(instance)
                return

            # C052: BoundMethod call -- prepend instance
            method_class = None
            if isinstance(fn_val, BoundMethod):
                args = [fn_val.instance] + args
                arg_count = len(args)
                method_class = fn_val.klass
                fn_val = fn_val.method

            captured_env = None
            if isinstance(fn_val, ClosureObject):
                captured_env = fn_val.env
                fn_obj = fn_val.fn
            elif isinstance(fn_val, FnObject):
                fn_obj = fn_val
            else:
                if self._vm_error_to_throw(f"Cannot call non-function: {fn_val}"):
                    return
                raise VMError(f"Cannot call non-function: {fn_val}")

            if fn_obj.arity != arg_count:
                if self._vm_error_to_throw(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}"):
                    return
                raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")

            # C047: Generator functions return a GeneratorObject
            if fn_obj.is_generator:
                gen = self._create_generator(fn_obj, captured_env, args)
                if method_class and method_class.parent:
                    gen.env['__super__'] = method_class.parent
                self.push(gen)
                return

            # C056: Async functions return a Promise
            if fn_obj.is_async:
                promise = self._start_async(fn_obj, captured_env, args)
                self.push(promise)
                return

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

            # C052: Inject __super__ for methods
            if method_class and method_class.parent:
                self.env['__super__'] = method_class.parent

        elif op == Op.RETURN:
            return_val = self.pop()
            if not self.call_stack:
                # Top-level return -- check for FinallyHandler
                if self._route_return_through_finally(return_val, 0):
                    pass  # routed to finally, don't halt
                else:
                    self.push(return_val)
                    return 'halt'
            else:
                cur_depth = len(self.call_stack)
                if self._route_return_through_finally(return_val, cur_depth):
                    pass  # routed to finally
                else:
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
                    if self._vm_error_to_throw(f"Array index must be integer, got {type(index).__name__}"):
                        return
                    raise VMError(f"Array index must be integer, got {type(index).__name__}")
                if index < 0 or index >= len(obj):
                    if self._vm_error_to_throw(f"Array index {index} out of bounds (length {len(obj)})"):
                        return
                    raise VMError(f"Array index {index} out of bounds (length {len(obj)})")
                self.push(obj[index])
            elif isinstance(obj, str):
                if not isinstance(index, int):
                    if self._vm_error_to_throw(f"String index must be integer, got {type(index).__name__}"):
                        return
                    raise VMError(f"String index must be integer, got {type(index).__name__}")
                if index < 0 or index >= len(obj):
                    if self._vm_error_to_throw(f"String index {index} out of bounds (length {len(obj)})"):
                        return
                    raise VMError(f"String index {index} out of bounds (length {len(obj)})")
                self.push(obj[index])
            elif isinstance(obj, dict):
                if index in obj:
                    self.push(obj[index])
                elif '__class__' in obj:
                    # Instance: look up method in class chain
                    result = self._lookup_method(obj['__class__'], index)
                    if result:
                        fn, found_class = result
                        self.push(BoundMethod(instance=obj, method=fn, klass=found_class))
                    else:
                        if self._vm_error_to_throw(f"'{obj['__class__'].name}' has no property '{index}'"):
                            return
                        raise VMError(f"'{obj['__class__'].name}' has no property '{index}'")
                else:
                    if self._vm_error_to_throw(f"Key {_format_value(index)} not found in hash map"):
                        return
                    raise VMError(f"Key {_format_value(index)} not found in hash map")
            else:
                if self._vm_error_to_throw(f"Cannot index type {type(obj).__name__}"):
                    return
                raise VMError(f"Cannot index type {type(obj).__name__}")

        elif op == Op.INDEX_SET:
            value = self.pop()
            index = self.pop()
            obj = self.pop()
            if isinstance(obj, list):
                if not isinstance(index, int):
                    if self._vm_error_to_throw(f"Array index must be integer, got {type(index).__name__}"):
                        return
                    raise VMError(f"Array index must be integer, got {type(index).__name__}")
                if index < 0 or index >= len(obj):
                    if self._vm_error_to_throw(f"Array index {index} out of bounds (length {len(obj)})"):
                        return
                    raise VMError(f"Array index {index} out of bounds (length {len(obj)})")
                obj[index] = value
                self.push(value)
            elif isinstance(obj, dict):
                obj[index] = value
                self.push(value)
            else:
                if self._vm_error_to_throw(f"Cannot assign to index of type {type(obj).__name__}"):
                    return
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
            for k, v in reversed(pairs):
                result[k] = v
            self.push(result)

        elif op == Op.ITER_PREPARE:
            value = self.pop()
            if isinstance(value, list):
                self.push(value)
            elif isinstance(value, dict):
                self.push(list(value.keys()))
            elif isinstance(value, str):
                self.push(list(value))
            elif isinstance(value, GeneratorObject):
                # Collect all values from generator into a list
                items = []
                while not value.done:
                    item = self._resume_generator(value)
                    if not value.done or item is not None:
                        items.append(item)
                    if value.done:
                        break
                self.push(items)
            else:
                if self._vm_error_to_throw(f"Cannot iterate over {type(value).__name__}"):
                    return
                raise VMError(f"Cannot iterate over {type(value).__name__}")

        elif op == Op.ITER_LENGTH:
            value = self.pop()
            if isinstance(value, list):
                self.push(len(value))
            else:
                if self._vm_error_to_throw(f"ITER_LENGTH requires array, got {type(value).__name__}"):
                    return
                raise VMError(f"ITER_LENGTH requires array, got {type(value).__name__}")

        # C045 error handling opcodes
        elif op == Op.SETUP_TRY:
            catch_addr = self.current_chunk.code[self.ip]
            self.ip += 1
            handler = TryHandler(
                catch_addr=catch_addr,
                catch_chunk=self.current_chunk,
                call_depth=len(self.call_stack),
                stack_depth=len(self.stack),
                env=dict(self.env),
            )
            self.handler_stack.append(handler)

        elif op == Op.POP_TRY:
            if self.handler_stack:
                self.handler_stack.pop()

        elif op == Op.THROW:
            value = self.pop()
            self._throw(value)

        # C055 finally blocks
        elif op == Op.SETUP_FINALLY:
            finally_addr = self.current_chunk.code[self.ip]
            self.ip += 1
            handler = FinallyHandler(
                finally_addr=finally_addr,
                finally_chunk=self.current_chunk,
                call_depth=len(self.call_stack),
                stack_depth=len(self.stack),
                env=dict(self.env),
            )
            self.handler_stack.append(handler)

        elif op == Op.POP_FINALLY:
            if self.handler_stack and isinstance(self.handler_stack[-1], FinallyHandler):
                self.handler_stack.pop()

        elif op == Op.END_FINALLY:
            pending = self._finally_pending
            self._finally_pending = None
            if pending is None:
                pass  # nothing pending, continue normally
            elif pending[0] == 'throw':
                self._throw(pending[1])
            elif pending[0] == 'return':
                return_val = pending[1]
                if not self.call_stack:
                    if self._route_return_through_finally(return_val, 0):
                        pass  # routed to next finally
                    else:
                        self.push(return_val)
                        return 'halt'
                else:
                    cur_depth = len(self.call_stack)
                    if self._route_return_through_finally(return_val, cur_depth):
                        pass  # routed to next finally
                    else:
                        frame = self.call_stack.pop()
                        self.current_chunk = frame.chunk
                        self.ip = frame.ip
                        self.env = frame.base_env
                        self.push(return_val)
                        return 'finally_returned'

        # C047 YIELD -- only meaningful inside generator execution
        elif op == Op.YIELD:
            # When encountered in normal (non-generator) execution, this is an error
            raise VMError("yield outside of generator function")

        # C056: AWAIT opcode
        elif op == Op.AWAIT:
            value = self.pop()
            if self._current_async is not None:
                # Inside an async coroutine -- suspend and wait for promise
                if isinstance(value, PromiseObject):
                    if value.state == PromiseObject.RESOLVED:
                        # Already resolved -- push value and continue
                        self.push(value.value)
                    elif value.state == PromiseObject.REJECTED:
                        # Already rejected -- throw or reject coroutine
                        if self.handler_stack:
                            self._throw(value.value)
                        else:
                            # No handler -- signal rejection to _resume_async
                            return ('async_reject', value.value)
                    else:
                        # Pending -- return 'await_suspend' signal
                        self.push(value)  # leave promise on stack for resume
                        return 'await_suspend'
                else:
                    # Awaiting a non-promise -- just push the value
                    self.push(value)
            else:
                # Await at top-level -- run async queue then get result
                if isinstance(value, PromiseObject):
                    self._drain_async_queue()
                    if value.state == PromiseObject.RESOLVED:
                        self.push(value.value)
                    elif value.state == PromiseObject.REJECTED:
                        self._throw(value.value)
                    else:
                        raise VMError("Await on unresolved promise (deadlock)")
                else:
                    self.push(value)

        # C050 Spread opcodes
        elif op == Op.ARRAY_SPREAD:
            # Pop source (array/string), pop target array, extend target, push result
            source = self.pop()
            target = self.pop()
            if not isinstance(target, list):
                if self._vm_error_to_throw(f"ARRAY_SPREAD target must be array, got {type(target).__name__}"):
                    return
                raise VMError(f"ARRAY_SPREAD target must be array, got {type(target).__name__}")
            if isinstance(source, list):
                target = target + source
            elif isinstance(source, str):
                target = target + list(source)
            else:
                if self._vm_error_to_throw(f"Cannot spread {type(source).__name__} into array"):
                    return
                raise VMError(f"Cannot spread {type(source).__name__} into array")
            self.push(target)

        elif op == Op.HASH_SPREAD:
            # Pop source hash, pop target hash, merge source into target, push result
            source = self.pop()
            target = self.pop()
            if not isinstance(target, dict):
                if self._vm_error_to_throw(f"HASH_SPREAD target must be hash, got {type(target).__name__}"):
                    return
                raise VMError(f"HASH_SPREAD target must be hash, got {type(target).__name__}")
            if not isinstance(source, dict):
                if self._vm_error_to_throw(f"Cannot spread {type(source).__name__} into hash"):
                    return
                raise VMError(f"Cannot spread {type(source).__name__} into hash")
            result = dict(target)
            result.update(source)
            self.push(result)

        elif op == Op.CALL_SPREAD:
            # Pop args_array, pop callee, call with unpacked args
            args_array = self.pop()
            fn_val = self.pop()
            if not isinstance(args_array, list):
                if self._vm_error_to_throw(f"CALL_SPREAD requires array of args"):
                    return
                raise VMError(f"CALL_SPREAD requires array of args")

            # Check for builtin call
            if isinstance(fn_val, tuple) and len(fn_val) == 2 and fn_val[0] == '__builtin__':
                try:
                    result = self._call_builtin(fn_val[1], args_array)
                    self.push(result)
                except VMError as e:
                    if self._vm_error_to_throw(str(e)):
                        return
                    raise
                return

            # C056: Promise builtin via spread
            if isinstance(fn_val, tuple) and len(fn_val) == 2 and fn_val[0] == '__promise_builtin__':
                try:
                    result = self._call_promise_builtin(fn_val[1], args_array)
                    self.push(result)
                except VMError as e:
                    if self._vm_error_to_throw(str(e)):
                        return
                    raise
                return

            # C052: ClassObject constructor via spread
            if isinstance(fn_val, ClassObject):
                instance = {'__class__': fn_val}
                init_result = self._lookup_method(fn_val, 'init')
                if init_result:
                    init_fn, init_class = init_result
                    expected = init_fn.arity - 1
                    if expected != len(args_array):
                        if self._vm_error_to_throw(
                            f"Class '{fn_val.name}' init expects {expected} args, got {len(args_array)}"):
                            return
                        raise VMError(
                            f"Class '{fn_val.name}' init expects {expected} args, got {len(args_array)}")
                    saved_chunk = self.current_chunk
                    saved_ip = self.ip
                    saved_env = self.env
                    saved_depth = len(self.call_stack)
                    saved_stack_depth = len(self.stack)
                    self.current_chunk = init_fn.chunk
                    self.ip = 0
                    self.env = dict(self.env)
                    all_args = [instance] + list(args_array)
                    for i, pname in enumerate(init_fn.chunk.names[:init_fn.arity]):
                        self.env[pname] = all_args[i]
                    if init_class.parent:
                        self.env['__super__'] = init_class.parent
                    self._run_until_return(saved_depth)
                    del self.stack[saved_stack_depth:]
                    self.current_chunk = saved_chunk
                    self.ip = saved_ip
                    self.env = saved_env
                elif len(args_array) > 0:
                    if self._vm_error_to_throw(
                        f"Class '{fn_val.name}' has no init but received {len(args_array)} args"):
                        return
                    raise VMError(
                        f"Class '{fn_val.name}' has no init but received {len(args_array)} args")
                self.push(instance)
                return

            # C052: BoundMethod via spread
            method_class = None
            if isinstance(fn_val, BoundMethod):
                args_array = [fn_val.instance] + list(args_array)
                method_class = fn_val.klass
                fn_val = fn_val.method

            captured_env = None
            if isinstance(fn_val, ClosureObject):
                captured_env = fn_val.env
                fn_obj = fn_val.fn
            elif isinstance(fn_val, FnObject):
                fn_obj = fn_val
            else:
                if self._vm_error_to_throw(f"Cannot call non-function: {fn_val}"):
                    return
                raise VMError(f"Cannot call non-function: {fn_val}")

            if fn_obj.arity != len(args_array):
                if self._vm_error_to_throw(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {len(args_array)}"):
                    return
                raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {len(args_array)}")

            # C047: Generator functions return a GeneratorObject
            if fn_obj.is_generator:
                gen = self._create_generator(fn_obj, captured_env, args_array)
                if method_class and method_class.parent:
                    gen.env['__super__'] = method_class.parent
                self.push(gen)
                return

            frame = CallFrame(self.current_chunk, self.ip, self.env)
            self.call_stack.append(frame)

            self.current_chunk = fn_obj.chunk
            self.ip = 0

            if captured_env is not None:
                self.env = dict(captured_env)
            else:
                self.env = dict(self.env)

            for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                self.env[param_name] = args_array[i]

            if method_class and method_class.parent:
                self.env['__super__'] = method_class.parent

        # C052: MAKE_CLASS opcode
        elif op == Op.MAKE_CLASS:
            method_count = self.current_chunk.code[self.ip]
            self.ip += 1
            methods = {}
            for _ in range(method_count):
                fn_obj = self.pop()
                name = self.pop()
                methods[name] = fn_obj
            parent = self.pop()
            if parent is not None and not isinstance(parent, ClassObject):
                if self._vm_error_to_throw(f"Superclass must be a class, got {type(parent).__name__}"):
                    return
                raise VMError(f"Superclass must be a class, got {type(parent).__name__}")
            class_name = self.pop()
            cls = ClassObject(name=class_name, methods=methods, parent=parent)
            self.push(cls)

        # C052: LOOKUP_METHOD opcode (for super calls)
        elif op == Op.LOOKUP_METHOD:
            method_name = self.pop()
            klass = self.pop()
            if not isinstance(klass, ClassObject):
                if self._vm_error_to_throw(f"LOOKUP_METHOD requires class, got {type(klass).__name__}"):
                    return
                raise VMError(f"LOOKUP_METHOD requires class, got {type(klass).__name__}")
            result = self._lookup_method(klass, method_name)
            if result is None:
                if self._vm_error_to_throw(f"Method '{method_name}' not found in class '{klass.name}'"):
                    return
                raise VMError(f"Method '{method_name}' not found in class '{klass.name}'")
            fn, _ = result
            self.push(fn)

        # C052: SUPER_INVOKE opcode
        elif op == Op.SUPER_INVOKE:
            name_idx = self.current_chunk.code[self.ip]
            self.ip += 1
            arg_count = self.current_chunk.code[self.ip]
            self.ip += 1
            method_name = self.current_chunk.names[name_idx]

            # Pop args, this, class from stack
            args = []
            for _ in range(arg_count):
                args.insert(0, self.pop())
            instance = self.pop()
            klass = self.pop()

            if not isinstance(klass, ClassObject):
                if self._vm_error_to_throw(f"SUPER_INVOKE requires class, got {type(klass).__name__}"):
                    return
                raise VMError(f"SUPER_INVOKE requires class, got {type(klass).__name__}")

            result = self._lookup_method(klass, method_name)
            if result is None:
                if self._vm_error_to_throw(f"Method '{method_name}' not found in super class '{klass.name}'"):
                    return
                raise VMError(f"Method '{method_name}' not found in super class '{klass.name}'")

            fn, found_class = result
            all_args = [instance] + args

            if fn.arity != len(all_args):
                if self._vm_error_to_throw(
                    f"Method '{method_name}' expects {fn.arity - 1} args, got {arg_count}"):
                    return
                raise VMError(
                    f"Method '{method_name}' expects {fn.arity - 1} args, got {arg_count}")

            # Set up call frame
            frame = CallFrame(self.current_chunk, self.ip, self.env)
            self.call_stack.append(frame)

            self.current_chunk = fn.chunk
            self.ip = 0
            self.env = dict(self.env)

            for i, pname in enumerate(fn.chunk.names[:fn.arity]):
                self.env[pname] = all_args[i]

            # Inject __super__ for further chain
            if found_class.parent:
                self.env['__super__'] = found_class.parent

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

        # C056: Drain async queue after main execution
        self._drain_async_queue()

        return self.stack[-1] if self.stack else None

    def _trace_op(self, op):
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"
        print(f"  [{self.ip-1:04d}] {name:20s} stack={self.stack[-5:]}")


# ============================================================
# Module Registry & Loader (C046)
# ============================================================

class ModuleError(Exception):
    pass


class ModuleRegistry:
    def __init__(self):
        self.sources = {}
        self.cache = {}
        self._loading = set()

    def register(self, name: str, source: str):
        self.sources[name] = source

    def register_many(self, modules: dict):
        self.sources.update(modules)

    def load(self, name: str) -> dict:
        if name in self.cache:
            return self.cache[name]

        if name not in self.sources:
            raise ModuleError(f"Module '{name}' not found")

        if name in self._loading:
            raise ModuleError(f"Circular import detected: '{name}'")

        self._loading.add(name)
        try:
            source = self.sources[name]
            exports = self._execute_module(name, source)
            self.cache[name] = exports
            return exports
        finally:
            self._loading.discard(name)

    def _execute_module(self, name: str, source: str) -> dict:
        ast = parse(source)
        compiler = Compiler()
        chunk = compiler.compile(ast)

        imports_to_resolve = []
        for stmt in ast.stmts:
            if isinstance(stmt, ImportStmt):
                imports_to_resolve.append(stmt)

        vm = VM(chunk)

        for imp in imports_to_resolve:
            module_exports = self.load(imp.module_name)
            if imp.names:
                for n in imp.names:
                    if n not in module_exports:
                        raise ModuleError(
                            f"Module '{imp.module_name}' does not export '{n}'"
                        )
                    vm.env[n] = module_exports[n]
            else:
                vm.env.update(module_exports)

        vm.run()

        exports = {}
        for export_name in compiler.exports:
            if export_name in vm.env:
                exports[export_name] = vm.env[export_name]
            else:
                raise ModuleError(
                    f"Exported name '{export_name}' not found in module '{name}' environment"
                )

        return exports

    def clear_cache(self):
        self.cache.clear()


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


def execute(source: str, trace=False, registry=None) -> dict:
    ast = parse(source)
    compiler = Compiler()
    chunk = compiler.compile(ast)
    vm = VM(chunk, trace=trace)

    if registry is not None:
        for stmt in ast.stmts:
            if isinstance(stmt, ImportStmt):
                module_exports = registry.load(stmt.module_name)
                if stmt.names:
                    for n in stmt.names:
                        if n not in module_exports:
                            raise ModuleError(
                                f"Module '{stmt.module_name}' does not export '{n}'"
                            )
                        vm.env[n] = module_exports[n]
                else:
                    vm.env.update(module_exports)

    result = vm.run()
    return {
        'result': result,
        'output': vm.output,
        'env': vm.env,
        'steps': vm.step_count,
        'exports': {name: vm.env[name] for name in compiler.exports if name in vm.env},
    }


def run(source: str, registry=None) -> tuple:
    r = execute(source, registry=registry)
    return r['result'], r['output']


def disassemble(chunk: Chunk) -> str:
    lines = []
    i = 0
    while i < len(chunk.code):
        op = chunk.code[i]
        name = Op(op).name if op in Op._value2member_map_ else f"??({op})"

        if op in (Op.CONST, Op.LOAD, Op.STORE, Op.JUMP, Op.JUMP_IF_FALSE,
                  Op.JUMP_IF_TRUE, Op.CALL, Op.MAKE_ARRAY, Op.MAKE_HASH,
                  Op.SETUP_TRY, Op.MAKE_CLASS, Op.SETUP_FINALLY):
            # Note: ARRAY_SPREAD, HASH_SPREAD, CALL_SPREAD have no operands
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
        elif op == Op.SUPER_INVOKE:
            name_idx = chunk.code[i + 1]
            arg_count = chunk.code[i + 2]
            nm = chunk.names[name_idx]
            lines.append(f"{i:04d}  {name:20s} {name_idx} ({nm}) args={arg_count}")
            i += 3
        else:
            lines.append(f"{i:04d}  {name}")
            i += 1
    return '\n'.join(lines)
