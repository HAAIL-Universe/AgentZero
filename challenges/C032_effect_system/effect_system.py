"""
Effect System Type Checker
Challenge C032 -- AgentZero Session 033

An algebraic effect system that tracks computational side effects at the type level.
Functions don't just have return types -- they declare what effects they perform.

Features:
  - Effect types: IO, State, Error, Async, and user-defined effects
  - Effect rows: sets of effects with optional row variables for polymorphism
  - Effect inference: automatically determines effects from function bodies
  - Effect annotations: explicit effect declarations on functions
  - Effect handlers: discharge effects via handle/with blocks
  - Effect subtyping: Pure (empty) is a subtype of any effect set
  - Effect polymorphism: functions generic over effect sets via row variables
  - Resume/continue in handlers (algebraic effect semantics)
  - Composed effect checking through call chains

Language syntax:
  let x = expr;
  fn name(params) { body }
  fn name(params) -> RetType ! Effect1, Effect2 { body }
  perform Effect.operation(args)
  handle { body } with { Effect.op(args) -> resume(val) }
  if (cond) { then } else { else }
  while (cond) { body }
  return expr;
  print(expr);
  throw(expr);
  try { body } catch(e) { handler }

Architecture:
  Source -> Lex -> Parse -> AST -> EffectCheck -> (typed+effected AST or errors)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import re


# ============================================================
# Effect Representations
# ============================================================

@dataclass(frozen=True)
class Effect:
    """A named effect (e.g., IO, State, Error)."""
    name: str

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        if isinstance(other, Effect):
            return self.name < other.name
        return NotImplemented


# Built-in effects
IO = Effect("IO")
STATE = Effect("State")
ERROR = Effect("Error")
ASYNC = Effect("Async")

BUILTIN_EFFECTS = {"IO": IO, "State": STATE, "Error": ERROR, "Async": ASYNC}


@dataclass
class EffectRow:
    """
    A set of effects, optionally open via a row variable.

    effects: frozenset of Effect -- known concrete effects
    row_var: optional EffectVar -- if present, the row is open (polymorphic)
    """
    effects: frozenset = field(default_factory=frozenset)
    row_var: Any = None  # None or EffectVar

    def __repr__(self):
        parts = sorted(repr(e) for e in self.effects)
        if self.row_var is not None:
            parts.append(repr(self.row_var))
        if not parts:
            return "Pure"
        return "{" + ", ".join(parts) + "}"

    @property
    def is_pure(self):
        return len(self.effects) == 0 and self.row_var is None

    def contains(self, effect):
        return effect in self.effects

    def union(self, other):
        """Combine two effect rows."""
        combined = self.effects | other.effects
        # If either has a row variable, keep it
        rv = self.row_var or other.row_var
        return EffectRow(combined, rv)

    def without(self, effect):
        """Remove an effect (used when handling)."""
        return EffectRow(self.effects - {effect}, self.row_var)

    def add(self, effect):
        """Add an effect."""
        return EffectRow(self.effects | {effect}, self.row_var)


PURE = EffectRow()  # Empty effect row = pure


@dataclass
class EffectVar:
    """Effect row variable for polymorphism."""
    id: int
    bound: Any = None  # None or EffectRow

    def __repr__(self):
        if self.bound is not None:
            return repr(self.bound)
        return f"?e{self.id}"

    def __eq__(self, other):
        if isinstance(other, EffectVar):
            return self.id == other.id
        return NotImplemented

    def __hash__(self):
        return hash(self.id)


# ============================================================
# Type Representations
# ============================================================

@dataclass(frozen=True)
class TInt:
    def __repr__(self): return "int"

@dataclass(frozen=True)
class TFloat:
    def __repr__(self): return "float"

@dataclass(frozen=True)
class TString:
    def __repr__(self): return "string"

@dataclass(frozen=True)
class TBool:
    def __repr__(self): return "bool"

@dataclass(frozen=True)
class TVoid:
    def __repr__(self): return "void"

@dataclass(frozen=True)
class TFunc:
    """Function type with effect annotation."""
    params: tuple       # tuple of types
    ret: Any           # return type
    effects: Any = None  # EffectRow or None (inferred)

    def __repr__(self):
        params_str = ", ".join(repr(p) for p in self.params)
        eff = f" ! {self.effects}" if self.effects and not self.effects.is_pure else ""
        return f"fn({params_str}) -> {self.ret!r}{eff}"

@dataclass
class TVar:
    """Type variable for inference."""
    id: int
    bound: Any = None

    def __repr__(self):
        if self.bound is not None:
            return repr(self.bound)
        return f"?T{self.id}"

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.id == other.id
        return NotImplemented

    def __hash__(self):
        return hash(self.id)


# Singleton types
INT = TInt()
FLOAT = TFloat()
STRING = TString()
BOOL = TBool()
VOID = TVoid()

TYPE_NAMES = {
    "int": INT, "float": FLOAT, "string": STRING,
    "bool": BOOL, "void": VOID,
}


# ============================================================
# Errors
# ============================================================

@dataclass
class EffectError:
    """An effect or type error with location."""
    message: str
    line: int
    kind: str = "type"  # "type", "effect", "syntax"

    def __repr__(self):
        return f"{self.kind.title()}Error at line {self.line}: {self.message}"


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
    type_ann: Any = None  # optional type annotation
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
    params: list        # list of (name, type_or_None) tuples
    body: Any
    ret_ann: Any = None     # return type annotation
    effect_ann: Any = None  # EffectRow annotation or None
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
class Perform:
    """Perform an effect operation: perform Effect.op(args)"""
    effect: str     # effect name
    operation: str  # operation name
    args: list
    line: int = 0

@dataclass
class HandleWith:
    """Handle effects: handle { body } with { Effect.op(args) -> expr, ... }"""
    body: Any           # Block to execute
    handlers: list      # list of EffectHandler
    line: int = 0

@dataclass
class EffectHandler:
    """A single handler clause: Effect.op(params) -> expr"""
    effect: str
    operation: str
    params: list     # parameter names
    body: Any        # handler body (has access to 'resume')
    line: int = 0

@dataclass
class Resume:
    """Resume from an effect handler: resume(value)"""
    value: Any
    line: int = 0

@dataclass
class ThrowExpr:
    """Throw an error: throw(expr)"""
    value: Any
    line: int = 0

@dataclass
class TryCatch:
    """try { body } catch(name) { handler }"""
    body: Any
    catch_name: str
    catch_body: Any
    line: int = 0

@dataclass
class EffectDecl:
    """Declare a custom effect: effect Name { op1(params) -> RetType; ... }"""
    name: str
    operations: list  # list of (op_name, param_types, ret_type)
    line: int = 0

@dataclass
class Program:
    stmts: list


# ============================================================
# Lexer
# ============================================================

class TokenType:
    # Literals
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    # Identifiers and keywords
    IDENT = "IDENT"
    # Keywords
    LET = "LET"
    FN = "FN"
    IF = "IF"
    ELSE = "ELSE"
    WHILE = "WHILE"
    RETURN = "RETURN"
    TRUE = "TRUE"
    FALSE = "FALSE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    PRINT = "PRINT"
    PERFORM = "PERFORM"
    HANDLE = "HANDLE"
    WITH = "WITH"
    RESUME = "RESUME"
    THROW = "THROW"
    TRY = "TRY"
    CATCH = "CATCH"
    EFFECT = "EFFECT"
    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    STAR = "STAR"
    SLASH = "SLASH"
    PERCENT = "PERCENT"
    ASSIGN = "ASSIGN"
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    LTE = "LTE"
    GTE = "GTE"
    BANG = "BANG"
    ARROW = "ARROW"
    DOT = "DOT"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    EOF = "EOF"


KEYWORDS = {
    "let": TokenType.LET, "fn": TokenType.FN, "if": TokenType.IF,
    "else": TokenType.ELSE, "while": TokenType.WHILE, "return": TokenType.RETURN,
    "true": TokenType.TRUE, "false": TokenType.FALSE,
    "and": TokenType.AND, "or": TokenType.OR, "not": TokenType.NOT,
    "print": TokenType.PRINT, "perform": TokenType.PERFORM,
    "handle": TokenType.HANDLE, "with": TokenType.WITH,
    "resume": TokenType.RESUME, "throw": TokenType.THROW,
    "try": TokenType.TRY, "catch": TokenType.CATCH,
    "effect": TokenType.EFFECT,
}


@dataclass
class Token:
    type: str
    value: Any
    line: int


class LexError(Exception):
    pass


def lex(source):
    """Tokenize source code."""
    tokens = []
    i = 0
    line = 1
    n = len(source)

    while i < n:
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
        if c == '/' and i + 1 < n and source[i + 1] == '/':
            while i < n and source[i] != '\n':
                i += 1
            continue

        # Numbers
        if c.isdigit():
            start = i
            while i < n and source[i].isdigit():
                i += 1
            if i < n and source[i] == '.' and i + 1 < n and source[i + 1].isdigit():
                i += 1
                while i < n and source[i].isdigit():
                    i += 1
                tokens.append(Token(TokenType.FLOAT, float(source[start:i]), line))
            else:
                tokens.append(Token(TokenType.INT, int(source[start:i]), line))
            continue

        # Strings
        if c == '"':
            i += 1
            start = i
            while i < n and source[i] != '"':
                if source[i] == '\\':
                    i += 1
                i += 1
            if i >= n:
                raise LexError(f"Unterminated string at line {line}")
            val = source[start:i].replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            tokens.append(Token(TokenType.STRING, val, line))
            i += 1
            continue

        # Identifiers and keywords
        if c.isalpha() or c == '_':
            start = i
            while i < n and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line))
            continue

        # Two-char operators
        if i + 1 < n:
            two = source[i:i+2]
            if two == '==':
                tokens.append(Token(TokenType.EQ, '==', line))
                i += 2
                continue
            if two == '!=':
                tokens.append(Token(TokenType.NEQ, '!=', line))
                i += 2
                continue
            if two == '<=':
                tokens.append(Token(TokenType.LTE, '<=', line))
                i += 2
                continue
            if two == '>=':
                tokens.append(Token(TokenType.GTE, '>=', line))
                i += 2
                continue
            if two == '->':
                tokens.append(Token(TokenType.ARROW, '->', line))
                i += 2
                continue

        # Single-char operators
        singles = {
            '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
            '/': TokenType.SLASH, '%': TokenType.PERCENT, '=': TokenType.ASSIGN,
            '<': TokenType.LT, '>': TokenType.GT, '!': TokenType.BANG,
            '.': TokenType.DOT, ',': TokenType.COMMA, ';': TokenType.SEMICOLON,
            ':': TokenType.COLON, '(': TokenType.LPAREN, ')': TokenType.RPAREN,
            '{': TokenType.LBRACE, '}': TokenType.RBRACE,
        }
        if c in singles:
            tokens.append(Token(singles[c], c, line))
            i += 1
            continue

        raise LexError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens


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
        tok = self.peek()
        if tok.type != tt:
            raise ParseError(f"Expected {tt}, got {tok.type} ('{tok.value}') at line {tok.line}")
        return self.advance()

    def match(self, tt):
        if self.peek().type == tt:
            return self.advance()
        return None

    def parse(self):
        stmts = []
        while self.peek().type != TokenType.EOF:
            stmts.append(self.declaration())
        return Program(stmts)

    def declaration(self):
        tt = self.peek().type
        if tt == TokenType.LET:
            return self.let_decl()
        if tt == TokenType.FN:
            return self.fn_decl()
        if tt == TokenType.EFFECT:
            return self.effect_decl()
        return self.statement()

    def let_decl(self):
        tok = self.advance()  # let
        name = self.expect(TokenType.IDENT).value
        # Optional type annotation: let x: int = ...
        type_ann = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type()
        self.expect(TokenType.ASSIGN)
        value = self.expression()
        self.expect(TokenType.SEMICOLON)
        return LetDecl(name=name, value=value, type_ann=type_ann, line=tok.line)

    def fn_decl(self):
        tok = self.advance()  # fn
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.parse_param())
            while self.match(TokenType.COMMA):
                params.append(self.parse_param())
        self.expect(TokenType.RPAREN)

        # Optional return type and effect annotation: -> RetType ! Effect1, Effect2
        ret_ann = None
        effect_ann = None
        if self.match(TokenType.ARROW):
            ret_ann = self.parse_type()
            if self.match(TokenType.BANG):
                effect_ann = self.parse_effect_row()

        body = self.block()
        return FnDecl(name=name, params=params, body=body,
                      ret_ann=ret_ann, effect_ann=effect_ann, line=tok.line)

    def parse_param(self):
        """Parse a function parameter: name or name: type"""
        name = self.expect(TokenType.IDENT).value
        type_ann = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type()
        return (name, type_ann)

    def parse_type(self):
        """Parse a type expression."""
        tok = self.expect(TokenType.IDENT)
        if tok.value in TYPE_NAMES:
            return TYPE_NAMES[tok.value]
        raise ParseError(f"Unknown type '{tok.value}' at line {tok.line}")

    def parse_effect_row(self):
        """Parse an effect row: Effect1, Effect2, ..."""
        effects = set()
        eff_name = self.expect(TokenType.IDENT).value
        effects.add(Effect(eff_name))
        while self.match(TokenType.COMMA):
            # Check if next is an identifier (part of effect list)
            # vs something else (could be end of list)
            if self.peek().type == TokenType.IDENT:
                eff_name = self.advance().value
                effects.add(Effect(eff_name))
            else:
                break
        return EffectRow(frozenset(effects))

    def effect_decl(self):
        """Parse: effect Name { op(params) -> RetType; ... }"""
        tok = self.advance()  # effect
        name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.LBRACE)
        operations = []
        while self.peek().type != TokenType.RBRACE:
            op_name = self.expect(TokenType.IDENT).value
            self.expect(TokenType.LPAREN)
            param_types = []
            if self.peek().type != TokenType.RPAREN:
                param_types.append(self.parse_type())
                while self.match(TokenType.COMMA):
                    param_types.append(self.parse_type())
            self.expect(TokenType.RPAREN)
            ret_type = VOID
            if self.match(TokenType.ARROW):
                ret_type = self.parse_type()
            self.expect(TokenType.SEMICOLON)
            operations.append((op_name, param_types, ret_type))
        self.expect(TokenType.RBRACE)
        return EffectDecl(name=name, operations=operations, line=tok.line)

    def statement(self):
        tt = self.peek().type
        if tt == TokenType.IF:
            return self.if_stmt()
        if tt == TokenType.WHILE:
            return self.while_stmt()
        if tt == TokenType.RETURN:
            return self.return_stmt()
        if tt == TokenType.PRINT:
            return self.print_stmt()
        if tt == TokenType.PERFORM:
            return self.perform_stmt()
        if tt == TokenType.HANDLE:
            return self.handle_stmt()
        if tt == TokenType.THROW:
            return self.throw_stmt()
        if tt == TokenType.TRY:
            return self.try_catch()
        if tt == TokenType.LBRACE:
            return self.block()
        return self.expr_stmt()

    def block(self):
        tok = self.expect(TokenType.LBRACE)
        stmts = []
        while self.peek().type != TokenType.RBRACE:
            stmts.append(self.declaration())
        self.expect(TokenType.RBRACE)
        return Block(stmts, tok.line)

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

    def _expect_ident_or_keyword(self):
        """Accept an identifier or keyword token as a name (for dotted access)."""
        tok = self.peek()
        if tok.type == TokenType.IDENT or tok.type in KEYWORDS.values():
            return self.advance()
        raise ParseError(f"Expected identifier, got {tok.type} ('{tok.value}') at line {tok.line}")

    def perform_stmt(self):
        """Parse: perform Effect.operation(args);"""
        tok = self.advance()  # perform
        effect = self.expect(TokenType.IDENT).value
        self.expect(TokenType.DOT)
        operation = self._expect_ident_or_keyword().value
        self.expect(TokenType.LPAREN)
        args = []
        if self.peek().type != TokenType.RPAREN:
            args.append(self.expression())
            while self.match(TokenType.COMMA):
                args.append(self.expression())
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return Perform(effect=effect, operation=operation, args=args, line=tok.line)

    def handle_stmt(self):
        """Parse: handle { body } with { Effect.op(params) -> { expr }, ... }"""
        tok = self.advance()  # handle
        body = self.block()
        self.expect(TokenType.WITH)
        self.expect(TokenType.LBRACE)
        handlers = []
        while self.peek().type != TokenType.RBRACE:
            handlers.append(self.parse_handler_clause())
        self.expect(TokenType.RBRACE)
        return HandleWith(body=body, handlers=handlers, line=tok.line)

    def parse_handler_clause(self):
        """Parse: Effect.op(params) -> { body }"""
        line = self.peek().line
        effect = self.expect(TokenType.IDENT).value
        self.expect(TokenType.DOT)
        operation = self._expect_ident_or_keyword().value
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.ARROW)
        body = self.block()
        return EffectHandler(effect=effect, operation=operation,
                            params=params, body=body, line=line)

    def throw_stmt(self):
        """Parse: throw(expr);"""
        tok = self.advance()  # throw
        self.expect(TokenType.LPAREN)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return ThrowExpr(value=value, line=tok.line)

    def try_catch(self):
        """Parse: try { body } catch(name) { handler }"""
        tok = self.advance()  # try
        body = self.block()
        self.expect(TokenType.CATCH)
        self.expect(TokenType.LPAREN)
        catch_name = self.expect(TokenType.IDENT).value
        self.expect(TokenType.RPAREN)
        catch_body = self.block()
        return TryCatch(body=body, catch_name=catch_name,
                       catch_body=catch_body, line=tok.line)

    # ---- Expressions ----

    def expr_stmt(self):
        """Expression statement."""
        expr = self.expression()
        self.expect(TokenType.SEMICOLON)
        return expr

    def expression(self):
        return self.assignment()

    def assignment(self):
        expr = self.or_expr()
        if isinstance(expr, Var) and self.match(TokenType.ASSIGN):
            value = self.expression()
            return Assign(name=expr.name, value=value, line=expr.line)
        return expr

    def or_expr(self):
        left = self.and_expr()
        while self.match(TokenType.OR):
            right = self.and_expr()
            left = BinOp(op="or", left=left, right=right, line=left.line)
        return left

    def and_expr(self):
        left = self.equality()
        while self.match(TokenType.AND):
            right = self.equality()
            left = BinOp(op="and", left=left, right=right, line=left.line)
        return left

    def equality(self):
        left = self.comparison()
        while True:
            if self.match(TokenType.EQ):
                right = self.comparison()
                left = BinOp(op="==", left=left, right=right, line=left.line)
            elif self.match(TokenType.NEQ):
                right = self.comparison()
                left = BinOp(op="!=", left=left, right=right, line=left.line)
            else:
                break
        return left

    def comparison(self):
        left = self.addition()
        while True:
            if self.match(TokenType.LT):
                right = self.addition()
                left = BinOp(op="<", left=left, right=right, line=left.line)
            elif self.match(TokenType.GT):
                right = self.addition()
                left = BinOp(op=">", left=left, right=right, line=left.line)
            elif self.match(TokenType.LTE):
                right = self.addition()
                left = BinOp(op="<=", left=left, right=right, line=left.line)
            elif self.match(TokenType.GTE):
                right = self.addition()
                left = BinOp(op=">=", left=left, right=right, line=left.line)
            else:
                break
        return left

    def addition(self):
        left = self.multiplication()
        while True:
            if self.match(TokenType.PLUS):
                right = self.multiplication()
                left = BinOp(op="+", left=left, right=right, line=left.line)
            elif self.match(TokenType.MINUS):
                right = self.multiplication()
                left = BinOp(op="-", left=left, right=right, line=left.line)
            else:
                break
        return left

    def multiplication(self):
        left = self.unary()
        while True:
            if self.match(TokenType.STAR):
                right = self.unary()
                left = BinOp(op="*", left=left, right=right, line=left.line)
            elif self.match(TokenType.SLASH):
                right = self.unary()
                left = BinOp(op="/", left=left, right=right, line=left.line)
            elif self.match(TokenType.PERCENT):
                right = self.unary()
                left = BinOp(op="%", left=left, right=right, line=left.line)
            else:
                break
        return left

    def unary(self):
        if self.match(TokenType.MINUS):
            operand = self.unary()
            return UnaryOp(op="-", operand=operand, line=operand.line)
        if self.match(TokenType.NOT):
            operand = self.unary()
            return UnaryOp(op="not", operand=operand, line=operand.line)
        return self.call()

    def call(self):
        expr = self.primary()
        if isinstance(expr, Var) and self.peek().type == TokenType.LPAREN:
            self.advance()  # (
            args = []
            if self.peek().type != TokenType.RPAREN:
                args.append(self.expression())
                while self.match(TokenType.COMMA):
                    args.append(self.expression())
            self.expect(TokenType.RPAREN)
            return CallExpr(callee=expr.name, args=args, line=expr.line)
        return expr

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
        if tok.type == TokenType.IDENT:
            self.advance()
            return Var(name=tok.value, line=tok.line)
        if tok.type == TokenType.RESUME:
            return self.parse_resume()
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.expression()
            self.expect(TokenType.RPAREN)
            return expr
        raise ParseError(f"Unexpected token '{tok.value}' ({tok.type}) at line {tok.line}")

    def parse_resume(self):
        """Parse resume(value)"""
        tok = self.advance()  # resume
        self.expect(TokenType.LPAREN)
        value = self.expression()
        self.expect(TokenType.RPAREN)
        return Resume(value=value, line=tok.line)


# ============================================================
# Type Environment
# ============================================================

class TypeEnv:
    """Scoped type environment with effect tracking."""

    def __init__(self, parent=None):
        self.bindings = {}
        self.parent = parent

    def define(self, name, typ):
        self.bindings[name] = typ

    def lookup(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def child(self):
        return TypeEnv(parent=self)


# ============================================================
# Effect Registry
# ============================================================

class EffectRegistry:
    """Tracks declared effects and their operations."""

    def __init__(self):
        self.effects = {}  # name -> {op_name: (param_types, ret_type)}
        # Register built-in effects
        self.register("IO", [
            ("print", [STRING], VOID),
            ("read", [], STRING),
        ])
        self.register("State", [
            ("get", [STRING], INT),  # get(key) -> int (simplified)
            ("set", [STRING, INT], VOID),  # set(key, value)
        ])
        self.register("Error", [
            ("raise", [STRING], VOID),
        ])
        self.register("Async", [
            ("await", [], VOID),
            ("yield_", [], VOID),
        ])

    def register(self, name, operations):
        """Register an effect with operations: [(op_name, param_types, ret_type)]"""
        self.effects[name] = {}
        for op_name, param_types, ret_type in operations:
            self.effects[name][op_name] = (param_types, ret_type)

    def lookup_operation(self, effect_name, op_name):
        """Look up an effect operation. Returns (param_types, ret_type) or None."""
        if effect_name in self.effects:
            return self.effects[effect_name].get(op_name)
        return None

    def has_effect(self, name):
        return name in self.effects


# ============================================================
# Type Unification
# ============================================================

def resolve(t):
    """Follow TVar chains."""
    while isinstance(t, TVar) and t.bound is not None:
        t = t.bound
    return t


def occurs_in(tvar, t):
    t = resolve(t)
    if isinstance(t, TVar):
        return t.id == tvar.id
    if isinstance(t, TFunc):
        return any(occurs_in(tvar, p) for p in t.params) or occurs_in(tvar, t.ret)
    return False


class UnificationError(Exception):
    pass


def unify(t1, t2):
    """Unify two types."""
    t1 = resolve(t1)
    t2 = resolve(t2)

    if t1 == t2:
        return t1

    if isinstance(t1, TVar):
        if occurs_in(t1, t2):
            raise UnificationError(f"Infinite type: {t1} ~ {t2}")
        t1.bound = t2
        return t2

    if isinstance(t2, TVar):
        if occurs_in(t2, t1):
            raise UnificationError(f"Infinite type: {t2} ~ {t1}")
        t2.bound = t1
        return t1

    # Int promotes to float
    if isinstance(t1, TInt) and isinstance(t2, TFloat):
        return FLOAT
    if isinstance(t1, TFloat) and isinstance(t2, TInt):
        return FLOAT

    # Function types
    if isinstance(t1, TFunc) and isinstance(t2, TFunc):
        if len(t1.params) != len(t2.params):
            raise UnificationError(f"Arity mismatch: {t1} vs {t2}")
        unified_params = tuple(unify(p1, p2) for p1, p2 in zip(t1.params, t2.params))
        unified_ret = unify(t1.ret, t2.ret)
        # Combine effects
        eff = None
        if t1.effects and t2.effects:
            eff = t1.effects.union(t2.effects)
        elif t1.effects:
            eff = t1.effects
        elif t2.effects:
            eff = t2.effects
        return TFunc(unified_params, unified_ret, eff)

    raise UnificationError(f"Cannot unify {t1!r} with {t2!r}")


# ============================================================
# Effect Unification
# ============================================================

def resolve_effect_var(ev):
    """Follow EffectVar chains."""
    while isinstance(ev, EffectVar) and ev.bound is not None:
        ev = ev.bound
    return ev


def unify_effects(e1, e2):
    """
    Unify two effect rows.
    e1 must be a subeffect of e2 (e1's effects must be subset of e2's).
    """
    if e1 is None or e2 is None:
        return e2 or e1 or PURE

    # Resolve row variables
    if e1.row_var:
        rv = resolve_effect_var(e1.row_var)
        if isinstance(rv, EffectRow):
            e1 = EffectRow(e1.effects | rv.effects, rv.row_var)
        elif isinstance(rv, EffectVar):
            e1 = EffectRow(e1.effects, rv)

    if e2.row_var:
        rv = resolve_effect_var(e2.row_var)
        if isinstance(rv, EffectRow):
            e2 = EffectRow(e2.effects | rv.effects, rv.row_var)
        elif isinstance(rv, EffectVar):
            e2 = EffectRow(e2.effects, rv)

    return e1.union(e2)


def effect_subset(sub, sup):
    """
    Check if sub's effects are a subset of sup's effects.
    Pure is a subset of everything.
    """
    if sub is None or sub.is_pure:
        return True
    if sup is None:
        return sub.is_pure

    # All concrete effects in sub must be in sup
    for eff in sub.effects:
        if eff not in sup.effects:
            # Check if sup has a row variable (open row)
            if sup.row_var is not None:
                continue  # Open row can absorb unknown effects
            return False

    return True


# ============================================================
# Effect Checker
# ============================================================

class EffectChecker:
    """
    Combined type and effect checker.

    Walks the AST, infers types AND effects simultaneously.
    Reports all errors (does not stop at first).
    """

    def __init__(self):
        self.errors = []
        self.env = TypeEnv()
        self.registry = EffectRegistry()
        self._tvar_counter = 0
        self._evar_counter = 0
        self._return_type_stack = []
        self._current_effects = PURE  # effects accumulated in current function
        self._effect_stack = []  # stack for nested function effects
        self._handler_stack = []  # stack of handled effects
        self._in_handler = False
        self._function_effects = {}  # fn_name -> EffectRow (inferred)

    def fresh_tvar(self):
        self._tvar_counter += 1
        return TVar(self._tvar_counter)

    def fresh_evar(self):
        self._evar_counter += 1
        return EffectVar(self._evar_counter)

    def error(self, msg, line, kind="type"):
        self.errors.append(EffectError(msg, line, kind))

    def add_effect(self, effect):
        """Record that the current function performs this effect."""
        self._current_effects = self._current_effects.add(effect)

    def is_handled(self, effect):
        """Check if an effect is currently handled by an enclosing handler."""
        for handled in self._handler_stack:
            if effect in handled:
                return True
        return False

    # ---- Public API ----

    def check(self, program):
        """Type-and-effect check a program. Returns list of EffectError."""
        self.errors = []
        # First pass: register effect declarations
        for stmt in program.stmts:
            if isinstance(stmt, EffectDecl):
                self._register_effect(stmt)
        # Second pass: check everything
        for stmt in program.stmts:
            self.check_stmt(stmt)
        return self.errors

    def _register_effect(self, node):
        """Register a custom effect declaration."""
        ops = []
        for op_name, param_types, ret_type in node.operations:
            ops.append((op_name, param_types, ret_type))
        self.registry.register(node.name, ops)

    # ---- Statements ----

    def check_stmt(self, node):
        method = f"check_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler:
            handler(node)
        else:
            self.infer(node)

    def check_LetDecl(self, node):
        value_type = self.infer(node.value)
        if node.type_ann is not None:
            if value_type is not None:
                try:
                    unify(value_type, node.type_ann)
                except UnificationError:
                    self.error(
                        f"Type mismatch: declared {node.type_ann!r} but got {value_type!r}",
                        node.line)
            self.env.define(node.name, node.type_ann)
        else:
            self.env.define(node.name, value_type if value_type else self.fresh_tvar())

    def check_Assign(self, node):
        value_type = self.infer(node.value)
        existing = self.env.lookup(node.name)
        if existing is None:
            self.error(f"Assignment to undefined variable '{node.name}'", node.line)
            self.env.define(node.name, value_type if value_type else self.fresh_tvar())
        else:
            if value_type is not None:
                try:
                    unify(existing, value_type)
                except UnificationError:
                    self.error(
                        f"Cannot assign {value_type!r} to '{node.name}' of type {resolve(existing)!r}",
                        node.line)

    def check_Block(self, node):
        child_env = self.env.child()
        old_env = self.env
        self.env = child_env
        for stmt in node.stmts:
            self.check_stmt(stmt)
        self.env = old_env

    def check_IfStmt(self, node):
        cond_type = self.infer(node.cond)
        if cond_type is not None:
            resolved = resolve(cond_type)
            if not isinstance(resolved, (TBool, TVar)):
                self.error(f"Condition must be bool, got {resolved!r}", node.line)
        self.check_stmt(node.then_body)
        if node.else_body:
            self.check_stmt(node.else_body)

    def check_WhileStmt(self, node):
        cond_type = self.infer(node.cond)
        if cond_type is not None:
            resolved = resolve(cond_type)
            if not isinstance(resolved, (TBool, TVar)):
                self.error(f"While condition must be bool, got {resolved!r}", node.line)
        self.check_stmt(node.body)

    def check_FnDecl(self, node):
        # Create function type
        param_types = []
        fn_env = self.env.child()

        for param_name, param_type_ann in node.params:
            if param_type_ann is not None:
                pt = param_type_ann
            else:
                pt = self.fresh_tvar()
            param_types.append(pt)
            fn_env.define(param_name, pt)

        ret_type = node.ret_ann if node.ret_ann is not None else self.fresh_tvar()

        # Effect annotation becomes the declared effects
        declared_effects = node.effect_ann  # may be None

        fn_type = TFunc(tuple(param_types), ret_type, declared_effects)
        self.env.define(node.name, fn_type)

        # Push effect context for this function
        self._effect_stack.append(self._current_effects)
        self._current_effects = PURE

        # Check body
        old_env = self.env
        self.env = fn_env
        self._return_type_stack.append(ret_type)

        self.check_stmt(node.body)

        self._return_type_stack.pop()
        self.env = old_env

        # Collect inferred effects
        inferred = self._current_effects
        self._function_effects[node.name] = inferred
        self._current_effects = self._effect_stack.pop()

        # Check declared vs inferred effects
        if declared_effects is not None:
            for eff in inferred.effects:
                if not declared_effects.contains(eff):
                    self.error(
                        f"Function '{node.name}' performs undeclared effect {eff!r}",
                        node.line, kind="effect")

    def check_ReturnStmt(self, node):
        if node.value is not None:
            ret_type = self.infer(node.value)
        else:
            ret_type = VOID

        if self._return_type_stack:
            expected = self._return_type_stack[-1]
            if ret_type is not None:
                try:
                    unify(expected, ret_type)
                except UnificationError:
                    self.error(
                        f"Return type mismatch: expected {resolve(expected)!r}, got {ret_type!r}",
                        node.line)
        else:
            self.error("Return statement outside of function", node.line)

    def check_PrintStmt(self, node):
        self.infer(node.value)
        self.add_effect(IO)

    def check_Perform(self, node):
        effect = Effect(node.effect)
        # Verify the effect and operation exist
        op_info = self.registry.lookup_operation(node.effect, node.operation)
        if op_info is None:
            if not self.registry.has_effect(node.effect):
                self.error(f"Unknown effect '{node.effect}'", node.line, kind="effect")
            else:
                self.error(
                    f"Unknown operation '{node.operation}' on effect '{node.effect}'",
                    node.line, kind="effect")
            return
        param_types, ret_type = op_info
        # Check argument count
        if len(node.args) != len(param_types):
            self.error(
                f"Effect operation '{node.effect}.{node.operation}' expects "
                f"{len(param_types)} args, got {len(node.args)}",
                node.line, kind="effect")
        else:
            # Check argument types
            for i, (arg, expected) in enumerate(zip(node.args, param_types)):
                arg_type = self.infer(arg)
                if arg_type is not None:
                    try:
                        unify(arg_type, expected)
                    except UnificationError:
                        self.error(
                            f"Arg {i+1} of '{node.effect}.{node.operation}': "
                            f"expected {expected!r}, got {arg_type!r}",
                            node.line)
        # Record the effect (unless it's handled)
        if not self.is_handled(effect):
            self.add_effect(effect)

    def check_HandleWith(self, node):
        # Collect which effects are handled
        handled_effects = set()
        for h in node.handlers:
            handled_effects.add(Effect(h.effect))

        # Push handled effects
        self._handler_stack.append(handled_effects)

        # Check the body
        self.check_stmt(node.body)

        # Pop handled effects
        self._handler_stack.pop()

        # Check handler bodies
        for h in node.handlers:
            self._check_handler_clause(h)

    def _check_handler_clause(self, handler):
        """Type-check a handler clause."""
        effect = handler.effect
        operation = handler.operation

        # Verify effect/operation
        op_info = self.registry.lookup_operation(effect, operation)
        if op_info is None:
            if not self.registry.has_effect(effect):
                self.error(f"Unknown effect '{effect}' in handler", handler.line, kind="effect")
            else:
                self.error(
                    f"Unknown operation '{operation}' on effect '{effect}' in handler",
                    handler.line, kind="effect")
            return

        param_types, ret_type = op_info

        # Check parameter count
        if len(handler.params) != len(param_types):
            self.error(
                f"Handler for '{effect}.{operation}' expects "
                f"{len(param_types)} params, got {len(handler.params)}",
                handler.line, kind="effect")

        # Create handler body environment with params bound
        handler_env = self.env.child()
        for i, pname in enumerate(handler.params):
            if i < len(param_types):
                handler_env.define(pname, param_types[i])
            else:
                handler_env.define(pname, self.fresh_tvar())

        # 'resume' is available in handler -- it's a function that takes ret_type and returns void
        handler_env.define("resume", TFunc((ret_type,), VOID))

        old_env = self.env
        self._in_handler = True
        self.env = handler_env
        self.check_stmt(handler.body)
        self.env = old_env
        self._in_handler = False

    def check_ThrowExpr(self, node):
        self.infer(node.value)
        if not self.is_handled(ERROR):
            self.add_effect(ERROR)

    def check_TryCatch(self, node):
        # try block may perform Error effects -- they're handled here
        self._handler_stack.append({ERROR})
        self.check_stmt(node.body)
        self._handler_stack.pop()

        # catch block has the error bound
        catch_env = self.env.child()
        catch_env.define(node.catch_name, STRING)  # error values are strings
        old_env = self.env
        self.env = catch_env
        self.check_stmt(node.catch_body)
        self.env = old_env

    def check_EffectDecl(self, node):
        # Already processed in first pass
        pass

    # ---- Expressions ----

    def infer(self, node):
        method = f"infer_{type(node).__name__}"
        handler = getattr(self, method, None)
        if handler:
            return handler(node)

        stmt_method = f"check_{type(node).__name__}"
        stmt_handler = getattr(self, stmt_method, None)
        if stmt_handler:
            stmt_handler(node)
            return VOID

        line = getattr(node, 'line', 0)
        self.error(f"Cannot type-check {type(node).__name__}", line)
        return None

    def infer_IntLit(self, node):
        return INT

    def infer_FloatLit(self, node):
        return FLOAT

    def infer_StringLit(self, node):
        return STRING

    def infer_BoolLit(self, node):
        return BOOL

    def infer_Var(self, node):
        t = self.env.lookup(node.name)
        if t is None:
            self.error(f"Undefined variable '{node.name}'", node.line)
            return None
        return resolve(t)

    def infer_UnaryOp(self, node):
        operand_type = self.infer(node.operand)
        if operand_type is None:
            return None
        operand_type = resolve(operand_type)

        if node.op == '-':
            if isinstance(operand_type, TInt): return INT
            if isinstance(operand_type, TFloat): return FLOAT
            if isinstance(operand_type, TVar): return operand_type
            self.error(f"Cannot negate type {operand_type!r}", node.line)
            return None

        if node.op == 'not':
            if isinstance(operand_type, TBool): return BOOL
            if isinstance(operand_type, TVar):
                try: unify(operand_type, BOOL)
                except UnificationError: pass
                return BOOL
            self.error(f"'not' requires bool, got {operand_type!r}", node.line)
            return None

        return None

    def infer_BinOp(self, node):
        left_type = self.infer(node.left)
        right_type = self.infer(node.right)
        if left_type is None or right_type is None:
            return None
        left_type = resolve(left_type)
        right_type = resolve(right_type)

        if node.op in ('+', '-', '*', '/', '%'):
            return self._check_arithmetic(node.op, left_type, right_type, node.line)
        if node.op in ('<', '>', '<=', '>='):
            return self._check_comparison(left_type, right_type, node.line)
        if node.op in ('==', '!='):
            return self._check_equality(left_type, right_type, node.line)
        if node.op in ('and', 'or'):
            return self._check_logical(left_type, right_type, node.line)

        self.error(f"Unknown operator '{node.op}'", node.line)
        return None

    def _check_arithmetic(self, op, lt, rt, line):
        if op == '+' and isinstance(lt, TString) and isinstance(rt, TString):
            return STRING
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            if isinstance(lt, TFloat) or isinstance(rt, TFloat):
                return FLOAT
            return INT
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            try:
                result = unify(lt, rt)
                return resolve(result)
            except UnificationError:
                pass
            return self.fresh_tvar()
        self.error(f"Cannot apply '{op}' to {lt!r} and {rt!r}", line)
        return None

    def _check_comparison(self, lt, rt, line):
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            return BOOL
        if isinstance(lt, TString) and isinstance(rt, TString):
            return BOOL
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            return BOOL
        self.error(f"Cannot compare {lt!r} and {rt!r}", line)
        return None

    def _check_equality(self, lt, rt, line):
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            return BOOL
        if type(lt) == type(rt):
            return BOOL
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            return BOOL
        self.error(f"Cannot compare {lt!r} and {rt!r} for equality", line)
        return None

    def _check_logical(self, lt, rt, line):
        if isinstance(lt, TVar):
            try: unify(lt, BOOL)
            except UnificationError: pass
        elif not isinstance(lt, TBool):
            self.error(f"Logical operator requires bool, got {lt!r}", line)
            return None
        if isinstance(rt, TVar):
            try: unify(rt, BOOL)
            except UnificationError: pass
        elif not isinstance(rt, TBool):
            self.error(f"Logical operator requires bool, got {rt!r}", line)
            return None
        return BOOL

    def infer_CallExpr(self, node):
        fn_type = self.env.lookup(node.callee)
        if fn_type is None:
            self.error(f"Undefined function '{node.callee}'", node.line)
            return None

        fn_type = resolve(fn_type)

        if isinstance(fn_type, TVar):
            arg_types = []
            for arg in node.args:
                at = self.infer(arg)
                arg_types.append(at if at else self.fresh_tvar())
            ret = self.fresh_tvar()
            fn_t = TFunc(tuple(arg_types), ret)
            try:
                unify(fn_type, fn_t)
            except UnificationError as e:
                self.error(str(e), node.line)
            return ret

        if not isinstance(fn_type, TFunc):
            self.error(f"'{node.callee}' is not a function (type: {fn_type!r})", node.line)
            return None

        # Check argument count
        if len(node.args) != len(fn_type.params):
            self.error(
                f"Function '{node.callee}' expects {len(fn_type.params)} args, got {len(node.args)}",
                node.line)
            return resolve(fn_type.ret)

        # Check argument types
        for i, (arg, param_type) in enumerate(zip(node.args, fn_type.params)):
            arg_type = self.infer(arg)
            if arg_type is not None:
                try:
                    unify(arg_type, param_type)
                except UnificationError:
                    self.error(
                        f"Argument {i+1} of '{node.callee}': expected {resolve(param_type)!r}, got {arg_type!r}",
                        node.line)

        # Propagate callee's effects to current context
        callee_effects = fn_type.effects
        if callee_effects:
            for eff in callee_effects.effects:
                if not self.is_handled(eff):
                    self.add_effect(eff)
        # Also check if we have inferred effects for this function
        if node.callee in self._function_effects:
            inferred = self._function_effects[node.callee]
            for eff in inferred.effects:
                if not self.is_handled(eff):
                    self.add_effect(eff)

        return resolve(fn_type.ret)

    def infer_Assign(self, node):
        self.check_Assign(node)
        t = self.env.lookup(node.name)
        return resolve(t) if t else None

    def infer_Resume(self, node):
        if not self._in_handler:
            self.error("'resume' used outside of effect handler", node.line, kind="effect")
            return VOID
        self.infer(node.value)
        return VOID

    # ---- Query API ----

    def get_function_effects(self, name):
        """Get the inferred effects for a function."""
        return self._function_effects.get(name, PURE)

    def get_function_type(self, name):
        """Get the type of a function."""
        t = self.env.lookup(name)
        if t:
            return resolve(t)
        return None


# ============================================================
# Public API
# ============================================================

def parse_source(source):
    """Parse source code into AST."""
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()


def check_source(source):
    """
    Type-and-effect check source code.
    Returns (errors, checker).
    """
    program = parse_source(source)
    checker = EffectChecker()
    errors = checker.check(program)
    return errors, checker


def check_program(program):
    """
    Type-and-effect check a pre-parsed Program.
    Returns (errors, checker).
    """
    checker = EffectChecker()
    errors = checker.check(program)
    return errors, checker


def format_errors(errors):
    """Format errors as human-readable string."""
    if not errors:
        return "No errors."
    lines = [f"Found {len(errors)} error(s):"]
    for e in errors:
        lines.append(f"  Line {e.line} [{e.kind}]: {e.message}")
    return "\n".join(lines)
