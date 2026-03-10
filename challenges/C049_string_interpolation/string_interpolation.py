"""
String Interpolation -- template strings with embedded expressions
Challenge C049 -- AgentZero Session 050

Extends C048 (Destructuring) with:
  - f"hello ${name}" -- variable interpolation
  - f"${a + b}" -- expression interpolation
  - f"${fn(x)}" -- function call interpolation
  - f"count: ${len(arr)}, first: ${arr[0]}" -- multiple interpolations
  - f"literal \\${not_interpolated}" -- escape to prevent interpolation
  - f"${string(42)}" -- explicit coercion (auto-coercion also works)
  - f"nested: ${f"inner ${x}"}" -- nested f-strings

New tokens: FSTRING
New AST: InterpolatedString
No new opcodes (compiles to string() calls and ADD concatenation)
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
    'for': TokenType.FOR,
    'in': TokenType.IN,
    'break': TokenType.BREAK,
    'continue': TokenType.CONTINUE,
    'try': TokenType.TRY,
    'catch': TokenType.CATCH,
    'throw': TokenType.THROW,
    # C046
    'import': TokenType.IMPORT,
    'export': TokenType.EXPORT,
    'from': TokenType.FROM,
    # C047
    'yield': TokenType.YIELD,
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

        # Two-char operators
        two = source[i:i+2] if i + 1 < len(source) else ''
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
        if two == '||':
            tokens.append(Token(TokenType.OR, '||', line)); i += 2; continue
        if two == '//':
            while i < len(source) and source[i] != '\n':
                i += 1
            continue

        # Three-char operators
        three = source[i:i+3] if i + 2 < len(source) else ''
        if three == '...':
            tokens.append(Token(TokenType.DOTDOTDOT, '...', line)); i += 3; continue

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

# C045 error handling
@dataclass
class TryCatchStmt:
    try_body: Any
    catch_var: str
    catch_body: Any
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

# C047 generators
@dataclass
class YieldExpr:
    value: Any  # expression to yield (None for bare yield)
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
        elif self.peek().type == TokenType.LET:
            let = self.let_decl()
            return ExportLetDecl(let_decl=let, line=tok.line)
        else:
            raise ParseError(f"Expected 'fn' or 'let' after 'export' at line {tok.line}")

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
        self.expect(TokenType.CATCH)
        self.expect(TokenType.LPAREN)
        catch_var = self.expect(TokenType.IDENT).value
        self.expect(TokenType.RPAREN)
        catch_body = self.block()
        return TryCatchStmt(
            try_body=try_body,
            catch_var=catch_var,
            catch_body=catch_body,
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
        expr = self.or_expr()
        if self.match(TokenType.ASSIGN):
            if isinstance(expr, Var):
                value = self.assignment()
                return Assign(expr.name, value, expr.line)
            elif isinstance(expr, IndexExpr):
                value = self.assignment()
                return IndexAssign(expr.obj, expr.index, value, expr.line)
            elif isinstance(expr, DotExpr):
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
        return expr

    def _array_lit_to_pattern(self, node):
        """Convert parsed ArrayLit to ArrayPattern for destructuring assignment."""
        elements = []
        for elem in node.elements:
            if isinstance(elem, Var):
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
        while True:
            if self.match(TokenType.LPAREN):
                args = []
                if self.peek().type != TokenType.RPAREN:
                    args.append(self.expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.expression())
                self.expect(TokenType.RPAREN)
                expr = CallExpr(expr, args, expr.line)
            elif self.match(TokenType.LBRACKET):
                index = self.expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexExpr(expr, index, expr.line)
            elif self.match(TokenType.DOT):
                name_tok = self.expect(TokenType.IDENT)
                expr = DotExpr(expr, name_tok.value, expr.line)
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
        raise ParseError(f"Unexpected token {tok.type.name} ({tok.value!r}) at line {tok.line}")

    def _parse_array_literal(self):
        tok = self.advance()
        elements = []
        if self.peek().type != TokenType.RBRACKET:
            elements.append(self.expression())
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RBRACKET:
                    break
                elements.append(self.expression())
        self.expect(TokenType.RBRACKET)
        return ArrayLit(elements, tok.line)

    def _parse_hash_key(self):
        tok = self.peek()
        if tok.type == TokenType.IDENT and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.COLON:
            self.advance()
            return StringLit(tok.value, tok.line)
        return self.expression()

    def _parse_hash_literal(self):
        tok = self.advance()
        pairs = []
        if self.peek().type != TokenType.RBRACE:
            key = self._parse_hash_key()
            self.expect(TokenType.COLON)
            value = self.expression()
            pairs.append((key, value))
            while self.match(TokenType.COMMA):
                if self.peek().type == TokenType.RBRACE:
                    break
                key = self._parse_hash_key()
                self.expect(TokenType.COLON)
                value = self.expression()
                pairs.append((key, value))
        self.expect(TokenType.RBRACE)
        return HashLit(pairs, tok.line)

    def _parse_lambda(self):
        tok = self.advance()
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
        return _ast_contains_yield(node.try_body) or _ast_contains_yield(node.catch_body)
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
    if isinstance(node, ArrayLit):
        return any(_ast_contains_yield(e) for e in node.elements)
    if isinstance(node, HashLit):
        return any(_ast_contains_yield(k) or _ast_contains_yield(v) for k, v in node.pairs)
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
        setup_addr = self.chunk.emit(Op.SETUP_TRY, 0, line)
        self.compile_node(node.try_body)
        self.chunk.emit(Op.POP_TRY, line=line)
        jump_end = self.chunk.emit(Op.JUMP, 0, line)
        catch_addr = len(self.chunk.code)
        self.chunk.patch(setup_addr + 1, catch_addr)
        catch_var_idx = self.chunk.add_name(node.catch_var)
        self.chunk.emit(Op.STORE, catch_var_idx, line)
        self.compile_node(node.catch_body)
        self.chunk.patch(jump_end + 1, len(self.chunk.code))

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

    def compile_HashLit(self, node):
        for key_expr, value_expr in node.pairs:
            self.compile_node(key_expr)
            self.compile_node(value_expr)
        self.chunk.emit(Op.MAKE_HASH, len(node.pairs), node.line)

    def compile_DotExpr(self, node):
        self.compile_node(node.obj)
        idx = self.chunk.add_constant(node.key)
        self.chunk.emit(Op.CONST, idx, node.line)
        self.chunk.emit(Op.INDEX_GET, line=node.line)

    def compile_DotAssign(self, node):
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


@dataclass
class TryHandler:
    catch_addr: int
    catch_chunk: Any
    call_depth: int
    stack_depth: int
    env: dict


# Sentinel for generator exhaustion
_GENERATOR_DONE = object()


def _format_value(value):
    if value is None:
        return "none"
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
        return f"<fn:{value.fn.name}>" if hasattr(value, 'fn') else f"<fn:{value.name}>"
    if isinstance(value, GeneratorObject):
        status = "done" if value.done else "suspended"
        return f"<generator:{value.fn.name}:{status}>"
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

    def _throw(self, value):
        if not self.handler_stack:
            raise VMError(f"Uncaught exception: {_format_value(value)}")

        handler = self.handler_stack.pop()

        while len(self.call_stack) > handler.call_depth:
            self.call_stack.pop()

        self.current_chunk = handler.catch_chunk
        self.ip = handler.catch_addr
        self.env = handler.env

        del self.stack[handler.stack_depth:]
        self.push(value)

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
                return "none"
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
            if isinstance(val, dict):
                return "hash"
            if isinstance(val, (ClosureObject, FnObject)):
                return "function"
            if isinstance(val, GeneratorObject):
                return "generator"
            return "unknown"

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

        # Restore generator state
        self.current_chunk = gen.fn.chunk
        self.ip = gen.ip
        self.env = gen.env
        self.stack = gen.stack
        self.call_stack = gen.call_stack
        self.handler_stack = gen.handler_stack

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
                        frame = self.call_stack.pop()
                        self.current_chunk = frame.chunk
                        self.ip = frame.ip
                        self.env = frame.base_env
                        self.push(result_val)
                        continue
                    else:
                        # Return from generator itself
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

            # Restore VM state
            self.current_chunk = saved_chunk
            self.ip = saved_ip
            self.env = saved_env
            self.stack = saved_stack
            self.call_stack = saved_call_stack
            self.handler_stack = saved_handler_stack

        if gen.done and not yielded:
            if default is not _GENERATOR_DONE:
                return default
            return None

        return result

    def _call_function(self, fn_val, args):
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
                if index not in obj:
                    if self._vm_error_to_throw(f"Key {_format_value(index)} not found in hash map"):
                        return
                    raise VMError(f"Key {_format_value(index)} not found in hash map")
                self.push(obj[index])
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

        # C047 YIELD -- only meaningful inside generator execution
        elif op == Op.YIELD:
            # When encountered in normal (non-generator) execution, this is an error
            raise VMError("yield outside of generator function")

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
                  Op.SETUP_TRY):
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
