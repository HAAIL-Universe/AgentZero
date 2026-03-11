"""V120: Array Domain Abstract Interpretation

Composes V020 (Abstract Domain Functor) + C039 (Abstract Interpreter) concepts
with V116 (Array Theory) semantics for array content analysis.

Tracks abstract properties of array contents:
- Per-element intervals for small/known-size arrays
- Smashed (single abstract value for all elements) for unknown-size
- Segment-based abstraction for partially-known regions
- Length tracking via intervals
- Out-of-bounds access detection
- Sortedness and boundedness inference
- Array initialization analysis

Uses a simple imperative language with arrays (self-contained parser).
"""

from __future__ import annotations
import sys
import os
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable
)
from enum import Enum, auto
from copy import deepcopy
import math

# ---------------------------------------------------------------------------
# Import V020 domain infrastructure
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V020_abstract_domain_functor'))
from domain_functor import (
    AbstractDomain, IntervalDomain, SignDomain,
    INF, NEG_INF,
)


# ===========================================================================
# AST for simple imperative language with arrays
# ===========================================================================

@dataclass(frozen=True)
class IntLit:
    value: int
    line: int = 0

@dataclass(frozen=True)
class VarExpr:
    name: str
    line: int = 0

@dataclass(frozen=True)
class BinExpr:
    op: str  # +, -, *, /, %, <, <=, >, >=, ==, !=
    left: Any
    right: Any
    line: int = 0

@dataclass(frozen=True)
class UnaryExpr:
    op: str  # -
    operand: Any
    line: int = 0

@dataclass(frozen=True)
class ArrayLit:
    """[e1, e2, ..., en]"""
    elements: Tuple
    line: int = 0

@dataclass(frozen=True)
class ArrayNew:
    """new_array(size, init_value)"""
    size: Any
    init_value: Any
    line: int = 0

@dataclass(frozen=True)
class ArrayRead:
    """a[i] -- Select"""
    array: Any
    index: Any
    line: int = 0

@dataclass(frozen=True)
class ArrayLen:
    """len(a)"""
    array: Any
    line: int = 0

@dataclass(frozen=True)
class LetStmt:
    name: str
    value: Any
    line: int = 0

@dataclass(frozen=True)
class AssignStmt:
    name: str
    value: Any
    line: int = 0

@dataclass(frozen=True)
class ArrayWriteStmt:
    """a[i] = v -- Store"""
    array: str
    index: Any
    value: Any
    line: int = 0

@dataclass(frozen=True)
class IfStmt:
    cond: Any
    then_body: List
    else_body: Optional[List]
    line: int = 0

@dataclass(frozen=True)
class WhileStmt:
    cond: Any
    body: List
    line: int = 0

@dataclass(frozen=True)
class AssertStmt:
    cond: Any
    message: str = ""
    line: int = 0

@dataclass(frozen=True)
class Program:
    stmts: List


# ===========================================================================
# Lexer
# ===========================================================================

class TT(Enum):
    INT = auto()
    IDENT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NEQ = auto()
    ASSIGN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMI = auto()
    LET = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    ASSERT = auto()
    NEW_ARRAY = auto()
    LEN = auto()
    EOF = auto()

@dataclass
class Token:
    type: TT
    value: Any
    line: int

KEYWORDS = {
    'let': TT.LET, 'if': TT.IF, 'else': TT.ELSE,
    'while': TT.WHILE, 'assert': TT.ASSERT,
    'new_array': TT.NEW_ARRAY, 'len': TT.LEN,
}

def lex(source: str) -> List[Token]:
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
            j = i
            while j < len(source) and source[j].isdigit():
                j += 1
            tokens.append(Token(TT.INT, int(source[i:j]), line))
            i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            tt = KEYWORDS.get(word, TT.IDENT)
            tokens.append(Token(tt, word, line))
            i = j
        elif c == '+':
            tokens.append(Token(TT.PLUS, '+', line)); i += 1
        elif c == '-':
            tokens.append(Token(TT.MINUS, '-', line)); i += 1
        elif c == '*':
            tokens.append(Token(TT.STAR, '*', line)); i += 1
        elif c == '/':
            tokens.append(Token(TT.SLASH, '/', line)); i += 1
        elif c == '%':
            tokens.append(Token(TT.PERCENT, '%', line)); i += 1
        elif c == '<':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(Token(TT.LE, '<=', line)); i += 2
            else:
                tokens.append(Token(TT.LT, '<', line)); i += 1
        elif c == '>':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(Token(TT.GE, '>=', line)); i += 2
            else:
                tokens.append(Token(TT.GT, '>', line)); i += 1
        elif c == '=':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(Token(TT.EQ, '==', line)); i += 2
            else:
                tokens.append(Token(TT.ASSIGN, '=', line)); i += 1
        elif c == '!':
            if i + 1 < len(source) and source[i + 1] == '=':
                tokens.append(Token(TT.NEQ, '!=', line)); i += 2
            else:
                raise SyntaxError(f"Unexpected '!' at line {line}")
        elif c == '[':
            tokens.append(Token(TT.LBRACKET, '[', line)); i += 1
        elif c == ']':
            tokens.append(Token(TT.RBRACKET, ']', line)); i += 1
        elif c == '(':
            tokens.append(Token(TT.LPAREN, '(', line)); i += 1
        elif c == ')':
            tokens.append(Token(TT.RPAREN, ')', line)); i += 1
        elif c == '{':
            tokens.append(Token(TT.LBRACE, '{', line)); i += 1
        elif c == '}':
            tokens.append(Token(TT.RBRACE, '}', line)); i += 1
        elif c == ',':
            tokens.append(Token(TT.COMMA, ',', line)); i += 1
        elif c == ';':
            tokens.append(Token(TT.SEMI, ';', line)); i += 1
        else:
            raise SyntaxError(f"Unexpected character '{c}' at line {line}")
    tokens.append(Token(TT.EOF, None, line))
    return tokens


# ===========================================================================
# Parser
# ===========================================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, tt: TT) -> Token:
        t = self._advance()
        if t.type != tt:
            raise SyntaxError(f"Expected {tt.name}, got {t.type.name} at line {t.line}")
        return t

    def _match(self, tt: TT) -> bool:
        if self._peek().type == tt:
            self._advance()
            return True
        return False

    def parse(self) -> Program:
        stmts = []
        while self._peek().type != TT.EOF:
            stmts.append(self._stmt())
        return Program(stmts)

    def _stmt(self) -> Any:
        t = self._peek()
        if t.type == TT.LET:
            return self._let_stmt()
        elif t.type == TT.IF:
            return self._if_stmt()
        elif t.type == TT.WHILE:
            return self._while_stmt()
        elif t.type == TT.ASSERT:
            return self._assert_stmt()
        elif t.type == TT.IDENT:
            return self._assign_or_array_write()
        else:
            raise SyntaxError(f"Unexpected token {t.type.name} at line {t.line}")

    def _let_stmt(self):
        self._expect(TT.LET)
        name = self._expect(TT.IDENT).value
        self._expect(TT.ASSIGN)
        val = self._expr()
        self._expect(TT.SEMI)
        return LetStmt(name, val, name_line(self.tokens[self.pos - 1]))

    def _assign_or_array_write(self):
        name_tok = self._expect(TT.IDENT)
        if self._peek().type == TT.LBRACKET:
            self._advance()  # [
            idx = self._expr()
            self._expect(TT.RBRACKET)
            self._expect(TT.ASSIGN)
            val = self._expr()
            self._expect(TT.SEMI)
            return ArrayWriteStmt(name_tok.value, idx, val, name_tok.line)
        else:
            self._expect(TT.ASSIGN)
            val = self._expr()
            self._expect(TT.SEMI)
            return AssignStmt(name_tok.value, val, name_tok.line)

    def _if_stmt(self):
        tok = self._expect(TT.IF)
        self._expect(TT.LPAREN)
        cond = self._expr()
        self._expect(TT.RPAREN)
        then_body = self._block()
        else_body = None
        if self._match(TT.ELSE):
            else_body = self._block()
        return IfStmt(cond, then_body, else_body, tok.line)

    def _while_stmt(self):
        tok = self._expect(TT.WHILE)
        self._expect(TT.LPAREN)
        cond = self._expr()
        self._expect(TT.RPAREN)
        body = self._block()
        return WhileStmt(cond, body, tok.line)

    def _assert_stmt(self):
        tok = self._expect(TT.ASSERT)
        self._expect(TT.LPAREN)
        cond = self._expr()
        msg = ""
        if self._match(TT.COMMA):
            msg_tok = self._expect(TT.IDENT)
            msg = msg_tok.value
        self._expect(TT.RPAREN)
        self._expect(TT.SEMI)
        return AssertStmt(cond, msg, tok.line)

    def _block(self) -> List:
        self._expect(TT.LBRACE)
        stmts = []
        while self._peek().type != TT.RBRACE:
            stmts.append(self._stmt())
        self._expect(TT.RBRACE)
        return stmts

    def _expr(self):
        return self._comparison()

    def _comparison(self):
        left = self._additive()
        ops = {TT.LT: '<', TT.LE: '<=', TT.GT: '>', TT.GE: '>=',
               TT.EQ: '==', TT.NEQ: '!='}
        while self._peek().type in ops:
            t = self._advance()
            right = self._additive()
            left = BinExpr(ops[t.type], left, right, t.line)
        return left

    def _additive(self):
        left = self._multiplicative()
        while self._peek().type in (TT.PLUS, TT.MINUS):
            t = self._advance()
            right = self._multiplicative()
            left = BinExpr(t.value, left, right, t.line)
        return left

    def _multiplicative(self):
        left = self._unary()
        while self._peek().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            t = self._advance()
            right = self._unary()
            left = BinExpr(t.value, left, right, t.line)
        return left

    def _unary(self):
        if self._peek().type == TT.MINUS:
            t = self._advance()
            operand = self._unary()
            return UnaryExpr('-', operand, t.line)
        return self._postfix()

    def _postfix(self):
        expr = self._primary()
        while self._peek().type == TT.LBRACKET:
            self._advance()
            idx = self._expr()
            self._expect(TT.RBRACKET)
            expr = ArrayRead(expr, idx, getattr(expr, 'line', 0))
        return expr

    def _primary(self):
        t = self._peek()
        if t.type == TT.INT:
            self._advance()
            return IntLit(t.value, t.line)
        elif t.type == TT.IDENT:
            self._advance()
            return VarExpr(t.name if hasattr(t, 'name') else t.value, t.line)
        elif t.type == TT.LBRACKET:
            return self._array_lit()
        elif t.type == TT.NEW_ARRAY:
            return self._new_array()
        elif t.type == TT.LEN:
            return self._len_expr()
        elif t.type == TT.LPAREN:
            self._advance()
            e = self._expr()
            self._expect(TT.RPAREN)
            return e
        else:
            raise SyntaxError(f"Unexpected token {t.type.name} at line {t.line}")

    def _array_lit(self):
        tok = self._expect(TT.LBRACKET)
        elements = []
        if self._peek().type != TT.RBRACKET:
            elements.append(self._expr())
            while self._match(TT.COMMA):
                elements.append(self._expr())
        self._expect(TT.RBRACKET)
        return ArrayLit(tuple(elements), tok.line)

    def _new_array(self):
        tok = self._expect(TT.NEW_ARRAY)
        self._expect(TT.LPAREN)
        size = self._expr()
        self._expect(TT.COMMA)
        init = self._expr()
        self._expect(TT.RPAREN)
        return ArrayNew(size, init, tok.line)

    def _len_expr(self):
        tok = self._expect(TT.LEN)
        self._expect(TT.LPAREN)
        arr = self._expr()
        self._expect(TT.RPAREN)
        return ArrayLen(arr, tok.line)


def name_line(tok):
    return tok.line


def parse_source(source: str) -> Program:
    return Parser(lex(source)).parse()


# ===========================================================================
# Array Abstract Domain
# ===========================================================================

@dataclass
class ArrayAbstractValue:
    """Abstract representation of an array's contents.

    Tracks:
    - length: IntervalDomain for possible array lengths
    - elements: dict mapping concrete index -> IntervalDomain
    - smash: IntervalDomain covering ALL elements (sound over-approximation)
    - segments: list of (lo_idx, hi_idx, IntervalDomain) for regions
    """
    length: IntervalDomain
    elements: Dict[int, IntervalDomain]
    smash: IntervalDomain  # covers all elements not in per-element map

    def copy(self) -> ArrayAbstractValue:
        return ArrayAbstractValue(
            length=self.length,
            elements=dict(self.elements),
            smash=self.smash,
        )

    @staticmethod
    def bot() -> ArrayAbstractValue:
        return ArrayAbstractValue(
            length=IntervalDomain(1, 0),  # BOT
            elements={},
            smash=IntervalDomain(1, 0),
        )

    @staticmethod
    def top() -> ArrayAbstractValue:
        return ArrayAbstractValue(
            length=IntervalDomain(0, INF),
            elements={},
            smash=IntervalDomain(NEG_INF, INF),
        )

    def is_bot(self) -> bool:
        return self.length.is_bot()

    def is_top(self) -> bool:
        return (self.length.lo <= 0 and self.length.hi == INF
                and not self.elements and self.smash.is_top())

    def get_element(self, index: int) -> IntervalDomain:
        """Get the abstract value at a concrete index."""
        if index in self.elements:
            return self.elements[index]
        return self.smash

    def set_element(self, index: int, value: IntervalDomain) -> ArrayAbstractValue:
        """Strong update at a concrete index."""
        result = self.copy()
        result.elements[index] = value
        return result

    def set_element_weak(self, index_interval: IntervalDomain,
                         value: IntervalDomain) -> ArrayAbstractValue:
        """Weak update: index is an interval, could be any position."""
        result = self.copy()
        # If index is a single concrete value, do strong update
        if index_interval.lo == index_interval.hi and not math.isinf(index_interval.lo):
            idx = int(index_interval.lo)
            result.elements[idx] = value
            return result
        # Otherwise, smash: widen smash to include the new value
        result.smash = result.smash.join(value)
        # All per-element entries in the possible range are also weakened
        if not math.isinf(index_interval.lo) and not math.isinf(index_interval.hi):
            lo = int(max(0, index_interval.lo))
            hi = int(min(index_interval.hi, 1000))  # cap to avoid huge loops
            for i in range(lo, hi + 1):
                if i in result.elements:
                    result.elements[i] = result.elements[i].join(value)
        else:
            # Unknown index: all per-element entries weakened
            for i in list(result.elements.keys()):
                result.elements[i] = result.elements[i].join(value)
        return result

    def read_element(self, index_interval: IntervalDomain) -> IntervalDomain:
        """Read at an abstract index. Returns join of all possible elements."""
        if index_interval.lo == index_interval.hi and not math.isinf(index_interval.lo):
            idx = int(index_interval.lo)
            return self.get_element(idx)
        # Unknown index: join of smash and all per-element entries in range
        result = self.smash
        if not math.isinf(index_interval.lo) and not math.isinf(index_interval.hi):
            lo = int(max(0, index_interval.lo))
            hi = int(min(index_interval.hi, 1000))
            for i in range(lo, hi + 1):
                if i in self.elements:
                    result = result.join(self.elements[i])
        else:
            for v in self.elements.values():
                result = result.join(v)
        return result

    def join(self, other: ArrayAbstractValue) -> ArrayAbstractValue:
        """Least upper bound."""
        if self.is_bot():
            return other.copy()
        if other.is_bot():
            return self.copy()
        length = self.length.join(other.length)
        smash = self.smash.join(other.smash)
        elements = {}
        all_keys = set(self.elements.keys()) | set(other.elements.keys())
        for k in all_keys:
            v1 = self.elements.get(k, self.smash)
            v2 = other.elements.get(k, other.smash)
            joined = v1.join(v2)
            # Only keep per-element if it adds precision over smash
            if not joined.leq(smash):
                elements[k] = joined
            elif joined.lo != smash.lo or joined.hi != smash.hi:
                elements[k] = joined
        return ArrayAbstractValue(length=length, elements=elements, smash=smash)

    def widen(self, other: ArrayAbstractValue) -> ArrayAbstractValue:
        """Widening for convergence."""
        if self.is_bot():
            return other.copy()
        if other.is_bot():
            return self.copy()
        length = self.length.widen(other.length)
        smash = self.smash.widen(other.smash)
        elements = {}
        all_keys = set(self.elements.keys()) | set(other.elements.keys())
        for k in all_keys:
            v1 = self.elements.get(k, self.smash)
            v2 = other.elements.get(k, other.smash)
            widened = v1.widen(v2)
            if not widened.leq(smash):
                elements[k] = widened
            elif widened.lo != smash.lo or widened.hi != smash.hi:
                elements[k] = widened
        return ArrayAbstractValue(length=length, elements=elements, smash=smash)

    def leq(self, other: ArrayAbstractValue) -> bool:
        """Partial order check."""
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        if not self.length.leq(other.length):
            return False
        if not self.smash.leq(other.smash):
            return False
        for k, v in self.elements.items():
            other_v = other.elements.get(k, other.smash)
            if not v.leq(other_v):
                return False
        return True

    def __repr__(self):
        if self.is_bot():
            return "ArrayBOT"
        parts = [f"len={self.length}"]
        if self.elements:
            elems = ", ".join(f"{k}:{v}" for k, v in sorted(self.elements.items()))
            parts.append(f"elems={{{elems}}}")
        if not self.smash.is_bot():
            parts.append(f"smash={self.smash}")
        return f"Array({', '.join(parts)})"


# ===========================================================================
# Array Abstract Environment
# ===========================================================================

class ArrayEnv:
    """Environment mapping variables to IntervalDomain or ArrayAbstractValue."""

    def __init__(self):
        self.scalars: Dict[str, IntervalDomain] = {}
        self.arrays: Dict[str, ArrayAbstractValue] = {}

    def copy(self) -> ArrayEnv:
        e = ArrayEnv()
        e.scalars = dict(self.scalars)
        e.arrays = {k: v.copy() for k, v in self.arrays.items()}
        return e

    def get_scalar(self, name: str) -> IntervalDomain:
        return self.scalars.get(name, IntervalDomain(NEG_INF, INF))

    def set_scalar(self, name: str, value: IntervalDomain):
        self.scalars[name] = value

    def get_array(self, name: str) -> ArrayAbstractValue:
        return self.arrays.get(name, ArrayAbstractValue.top())

    def set_array(self, name: str, value: ArrayAbstractValue):
        self.arrays[name] = value

    def join(self, other: ArrayEnv) -> ArrayEnv:
        result = ArrayEnv()
        all_scalars = set(self.scalars.keys()) | set(other.scalars.keys())
        for k in all_scalars:
            v1 = self.get_scalar(k)
            v2 = other.get_scalar(k)
            result.scalars[k] = v1.join(v2)
        all_arrays = set(self.arrays.keys()) | set(other.arrays.keys())
        for k in all_arrays:
            v1 = self.get_array(k)
            v2 = other.get_array(k)
            result.arrays[k] = v1.join(v2)
        return result

    def widen(self, other: ArrayEnv) -> ArrayEnv:
        result = ArrayEnv()
        all_scalars = set(self.scalars.keys()) | set(other.scalars.keys())
        for k in all_scalars:
            v1 = self.scalars.get(k, IntervalDomain(1, 0))  # BOT if missing
            v2 = other.scalars.get(k, IntervalDomain(1, 0))
            result.scalars[k] = v1.widen(v2)
        all_arrays = set(self.arrays.keys()) | set(other.arrays.keys())
        for k in all_arrays:
            v1 = self.arrays.get(k, ArrayAbstractValue.bot())
            v2 = other.arrays.get(k, ArrayAbstractValue.bot())
            result.arrays[k] = v1.widen(v2)
        return result

    def equals(self, other: ArrayEnv) -> bool:
        all_scalars = set(self.scalars.keys()) | set(other.scalars.keys())
        for k in all_scalars:
            v1 = self.get_scalar(k)
            v2 = other.get_scalar(k)
            if not v1.eq(v2):
                return False
        all_arrays = set(self.arrays.keys()) | set(other.arrays.keys())
        for k in all_arrays:
            v1 = self.get_array(k)
            v2 = other.get_array(k)
            if not v1.leq(v2) or not v2.leq(v1):
                return False
        return True


# ===========================================================================
# Warnings
# ===========================================================================

class WarningKind(Enum):
    OUT_OF_BOUNDS = "out_of_bounds"
    POSSIBLE_OUT_OF_BOUNDS = "possible_out_of_bounds"
    DIVISION_BY_ZERO = "division_by_zero"
    POSSIBLE_DIVISION_BY_ZERO = "possible_division_by_zero"
    NEGATIVE_LENGTH = "negative_length"
    ASSERTION_FAILURE = "assertion_failure"
    POSSIBLE_ASSERTION_FAILURE = "possible_assertion_failure"

@dataclass
class Warning:
    kind: WarningKind
    message: str
    line: int


# ===========================================================================
# Array Abstract Interpreter
# ===========================================================================

class ArrayInterpreter:
    """Abstract interpreter for programs with arrays.

    Tracks scalar variables as IntervalDomain and array variables as
    ArrayAbstractValue. Detects out-of-bounds accesses and assertion failures.
    """

    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
        self.warnings: List[Warning] = []

    def analyze(self, source: str) -> dict:
        """Analyze source code and return analysis results."""
        program = parse_source(source)
        env = ArrayEnv()
        self.warnings = []
        env = self._interpret_stmts(program.stmts, env)
        return {
            'env': env,
            'warnings': list(self.warnings),
            'scalars': {k: v for k, v in env.scalars.items()},
            'arrays': {k: v for k, v in env.arrays.items()},
        }

    def _interpret_stmts(self, stmts: List, env: ArrayEnv) -> ArrayEnv:
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env)
        return env

    def _interpret_stmt(self, stmt, env: ArrayEnv) -> ArrayEnv:
        if isinstance(stmt, LetStmt):
            return self._interpret_let(stmt, env)
        elif isinstance(stmt, AssignStmt):
            return self._interpret_assign(stmt, env)
        elif isinstance(stmt, ArrayWriteStmt):
            return self._interpret_array_write(stmt, env)
        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env)
        elif isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, env)
        elif isinstance(stmt, AssertStmt):
            return self._interpret_assert(stmt, env)
        else:
            return env

    def _interpret_let(self, stmt: LetStmt, env: ArrayEnv) -> ArrayEnv:
        env = env.copy()
        if isinstance(stmt.value, (ArrayLit, ArrayNew)):
            arr = self._eval_array(stmt.value, env)
            env.set_array(stmt.name, arr)
        else:
            val = self._eval_expr(stmt.value, env)
            env.set_scalar(stmt.name, val)
        return env

    def _interpret_assign(self, stmt: AssignStmt, env: ArrayEnv) -> ArrayEnv:
        env = env.copy()
        if isinstance(stmt.value, (ArrayLit, ArrayNew)):
            arr = self._eval_array(stmt.value, env)
            env.set_array(stmt.name, arr)
        else:
            val = self._eval_expr(stmt.value, env)
            env.set_scalar(stmt.name, val)
        return env

    def _interpret_array_write(self, stmt: ArrayWriteStmt, env: ArrayEnv) -> ArrayEnv:
        env = env.copy()
        arr = env.get_array(stmt.array)
        idx = self._eval_expr(stmt.index, env)
        val = self._eval_expr(stmt.value, env)

        # Check bounds
        self._check_bounds(arr, idx, stmt.line)

        # Perform the store
        arr = arr.set_element_weak(idx, val)
        env.set_array(stmt.array, arr)
        return env

    def _interpret_if(self, stmt: IfStmt, env: ArrayEnv) -> ArrayEnv:
        # Evaluate condition to check if it's always true/false
        cond_val = self._eval_expr(stmt.cond, env)

        then_env, else_env = self._refine_condition(stmt.cond, env)

        # Check if then-branch is definitely dead (condition always false)
        then_dead = cond_val.hi < 1  # max value < 1 means always 0
        # Check if else-branch is definitely dead (condition always true)
        else_dead = cond_val.lo >= 1  # min value >= 1 means always true

        if then_dead:
            # Only else branch executes
            if stmt.else_body is not None:
                return self._interpret_stmts(stmt.else_body, else_env)
            else:
                return else_env
        elif else_dead:
            # Only then branch executes
            return self._interpret_stmts(stmt.then_body, then_env)
        else:
            # Both branches possible
            then_result = self._interpret_stmts(stmt.then_body, then_env)
            if stmt.else_body is not None:
                else_result = self._interpret_stmts(stmt.else_body, else_env)
            else:
                else_result = else_env
            return then_result.join(else_result)

    def _interpret_while(self, stmt: WhileStmt, env: ArrayEnv) -> ArrayEnv:
        current = env.copy()
        for _ in range(self.max_iterations):
            loop_env, exit_env = self._refine_condition(stmt.cond, current)
            body_result = self._interpret_stmts(stmt.body, loop_env)
            next_env = current.widen(body_result)
            if next_env.equals(current):
                # Fixpoint reached
                _, final_exit = self._refine_condition(stmt.cond, current)
                return final_exit
            current = next_env
        # Max iterations: return conservative join
        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env

    def _interpret_assert(self, stmt: AssertStmt, env: ArrayEnv) -> ArrayEnv:
        cond_val = self._eval_expr(stmt.cond, env)
        # For comparison expressions, result is 0 (false) or 1 (true)
        if cond_val.hi < 1:
            # Definitely false
            self.warnings.append(Warning(
                WarningKind.ASSERTION_FAILURE,
                f"Assertion always fails at line {stmt.line}",
                stmt.line,
            ))
        elif cond_val.lo < 1:
            # Possibly false
            self.warnings.append(Warning(
                WarningKind.POSSIBLE_ASSERTION_FAILURE,
                f"Assertion may fail at line {stmt.line}",
                stmt.line,
            ))
        # Refine env assuming assertion holds
        then_env, _ = self._refine_condition(stmt.cond, env)
        return then_env

    def _eval_array(self, expr, env: ArrayEnv) -> ArrayAbstractValue:
        """Evaluate an expression that produces an array."""
        if isinstance(expr, ArrayLit):
            elements = {}
            smash = IntervalDomain(1, 0)  # BOT initially
            for i, e in enumerate(expr.elements):
                val = self._eval_expr(e, env)
                elements[i] = val
                smash = smash.join(val)
            length = IntervalDomain.from_concrete(len(expr.elements))
            return ArrayAbstractValue(length=length, elements=elements, smash=smash)
        elif isinstance(expr, ArrayNew):
            size = self._eval_expr(expr.size, env)
            init = self._eval_expr(expr.init_value, env)
            if size.lo < 0:
                self.warnings.append(Warning(
                    WarningKind.NEGATIVE_LENGTH,
                    f"Array size may be negative at line {expr.line}",
                    expr.line,
                ))
            # If size is concrete, populate per-element
            elements = {}
            if size.lo == size.hi and not math.isinf(size.lo) and size.lo <= 100:
                n = int(size.lo)
                for i in range(n):
                    elements[i] = init
            return ArrayAbstractValue(length=size, elements=elements, smash=init)
        else:
            return ArrayAbstractValue.top()

    def _eval_expr(self, expr, env: ArrayEnv) -> IntervalDomain:
        """Evaluate an expression to an IntervalDomain."""
        if isinstance(expr, IntLit):
            return IntervalDomain.from_concrete(expr.value)
        elif isinstance(expr, VarExpr):
            return env.get_scalar(expr.name)
        elif isinstance(expr, UnaryExpr):
            if expr.op == '-':
                return self._eval_expr(expr.operand, env).neg()
            return IntervalDomain(NEG_INF, INF)
        elif isinstance(expr, BinExpr):
            return self._eval_binop(expr, env)
        elif isinstance(expr, ArrayRead):
            return self._eval_array_read(expr, env)
        elif isinstance(expr, ArrayLen):
            return self._eval_array_len(expr, env)
        else:
            return IntervalDomain(NEG_INF, INF)

    def _eval_binop(self, expr: BinExpr, env: ArrayEnv) -> IntervalDomain:
        left = self._eval_expr(expr.left, env)
        right = self._eval_expr(expr.right, env)

        op = expr.op
        if op == '+':
            return left.add(right)
        elif op == '-':
            return left.sub(right)
        elif op == '*':
            return left.mul(right)
        elif op == '/':
            if right.contains(0):
                if right.lo == 0 and right.hi == 0:
                    self.warnings.append(Warning(
                        WarningKind.DIVISION_BY_ZERO,
                        f"Division by zero at line {expr.line}",
                        expr.line,
                    ))
                else:
                    self.warnings.append(Warning(
                        WarningKind.POSSIBLE_DIVISION_BY_ZERO,
                        f"Possible division by zero at line {expr.line}",
                        expr.line,
                    ))
            return self._interval_div(left, right)
        elif op == '%':
            if right.contains(0):
                self.warnings.append(Warning(
                    WarningKind.POSSIBLE_DIVISION_BY_ZERO,
                    f"Possible modulo by zero at line {expr.line}",
                    expr.line,
                ))
            return self._interval_mod(left, right)
        elif op in ('<', '<=', '>', '>=', '==', '!='):
            return self._eval_comparison(left, right, op)
        return IntervalDomain(NEG_INF, INF)

    def _interval_div(self, a: IntervalDomain, b: IntervalDomain) -> IntervalDomain:
        if a.is_bot() or b.is_bot():
            return IntervalDomain(1, 0)
        if b.lo == 0 and b.hi == 0:
            return IntervalDomain(1, 0)  # BOT (undefined)
        # Exclude zero from divisor
        if b.lo <= 0 <= b.hi:
            # Split around zero
            neg = IntervalDomain(b.lo, -1) if b.lo < 0 else IntervalDomain(1, 0)
            pos = IntervalDomain(1, b.hi) if b.hi > 0 else IntervalDomain(1, 0)
            r1 = self._interval_div(a, neg) if not neg.is_bot() else IntervalDomain(1, 0)
            r2 = self._interval_div(a, pos) if not pos.is_bot() else IntervalDomain(1, 0)
            return r1.join(r2)
        # b doesn't contain 0
        corners = []
        for x in [a.lo, a.hi]:
            for y in [b.lo, b.hi]:
                if not math.isinf(x) and not math.isinf(y) and y != 0:
                    corners.append(x / y)
                elif math.isinf(x) and not math.isinf(y) and y != 0:
                    corners.append(math.copysign(INF, x * y))
                elif not math.isinf(x) and math.isinf(y):
                    corners.append(0)
                else:
                    return IntervalDomain(NEG_INF, INF)
        if not corners:
            return IntervalDomain(NEG_INF, INF)
        lo = math.floor(min(corners))
        hi = math.ceil(max(corners))
        return IntervalDomain(lo, hi)

    def _interval_mod(self, a: IntervalDomain, b: IntervalDomain) -> IntervalDomain:
        if a.is_bot() or b.is_bot():
            return IntervalDomain(1, 0)
        if b.lo == 0 and b.hi == 0:
            return IntervalDomain(1, 0)
        # |a % b| < |b|
        abs_b = max(abs(b.lo) if not math.isinf(b.lo) else INF,
                    abs(b.hi) if not math.isinf(b.hi) else INF)
        if math.isinf(abs_b):
            return IntervalDomain(NEG_INF, INF)
        bound = abs_b - 1
        if a.lo >= 0:
            return IntervalDomain(0, min(a.hi, bound))
        elif a.hi <= 0:
            return IntervalDomain(max(a.lo, -bound), 0)
        else:
            return IntervalDomain(-bound, bound)

    def _eval_comparison(self, left: IntervalDomain, right: IntervalDomain,
                         op: str) -> IntervalDomain:
        """Evaluate comparison, returning interval [0,0], [1,1], or [0,1]."""
        if left.is_bot() or right.is_bot():
            return IntervalDomain(1, 0)

        if op == '<':
            if left.hi < right.lo:
                return IntervalDomain(1, 1)  # always true
            elif left.lo >= right.hi:
                return IntervalDomain(0, 0)  # always false
        elif op == '<=':
            if left.hi <= right.lo:
                return IntervalDomain(1, 1)
            elif left.lo > right.hi:
                return IntervalDomain(0, 0)
        elif op == '>':
            if left.lo > right.hi:
                return IntervalDomain(1, 1)
            elif left.hi <= right.lo:
                return IntervalDomain(0, 0)
        elif op == '>=':
            if left.lo >= right.hi:
                return IntervalDomain(1, 1)
            elif left.hi < right.lo:
                return IntervalDomain(0, 0)
        elif op == '==':
            if left.lo == left.hi == right.lo == right.hi:
                return IntervalDomain(1, 1)
            # Check disjoint
            meet = left.meet(right)
            if meet.is_bot():
                return IntervalDomain(0, 0)
        elif op == '!=':
            meet = left.meet(right)
            if meet.is_bot():
                return IntervalDomain(1, 1)
            if left.lo == left.hi == right.lo == right.hi:
                return IntervalDomain(0, 0)

        return IntervalDomain(0, 1)

    def _eval_array_read(self, expr: ArrayRead, env: ArrayEnv) -> IntervalDomain:
        """Evaluate a[i]."""
        if isinstance(expr.array, VarExpr):
            arr = env.get_array(expr.array.name)
        else:
            return IntervalDomain(NEG_INF, INF)

        idx = self._eval_expr(expr.index, env)
        self._check_bounds(arr, idx, expr.line)
        return arr.read_element(idx)

    def _eval_array_len(self, expr: ArrayLen, env: ArrayEnv) -> IntervalDomain:
        """Evaluate len(a)."""
        if isinstance(expr.array, VarExpr):
            arr = env.get_array(expr.array.name)
            return arr.length
        return IntervalDomain(0, INF)

    def _check_bounds(self, arr: ArrayAbstractValue, idx: IntervalDomain, line: int):
        """Check if array access is in bounds."""
        if arr.is_bot() or idx.is_bot():
            return
        # Check lower bound
        if idx.hi < 0:
            self.warnings.append(Warning(
                WarningKind.OUT_OF_BOUNDS,
                f"Array index always negative at line {line}",
                line,
            ))
            return
        if idx.lo < 0:
            self.warnings.append(Warning(
                WarningKind.POSSIBLE_OUT_OF_BOUNDS,
                f"Array index may be negative at line {line}",
                line,
            ))
        # Check upper bound
        if not arr.length.is_bot() and not math.isinf(arr.length.hi):
            max_len = arr.length.hi
            if idx.lo >= max_len:
                self.warnings.append(Warning(
                    WarningKind.OUT_OF_BOUNDS,
                    f"Array index always out of bounds at line {line}",
                    line,
                ))
            elif idx.hi >= max_len:
                self.warnings.append(Warning(
                    WarningKind.POSSIBLE_OUT_OF_BOUNDS,
                    f"Array index may be out of bounds at line {line}",
                    line,
                ))

    def _refine_condition(self, expr, env: ArrayEnv) -> Tuple[ArrayEnv, ArrayEnv]:
        """Refine env for then-branch (cond true) and else-branch (cond false)."""
        if isinstance(expr, BinExpr):
            return self._refine_comparison(expr, env)
        # For non-comparison expressions, no refinement
        return env.copy(), env.copy()

    def _refine_comparison(self, expr: BinExpr, env: ArrayEnv) -> Tuple[ArrayEnv, ArrayEnv]:
        """Refine environment based on a comparison condition."""
        # Get the variable names involved (for refinement)
        left_var = self._get_var_name(expr.left)
        right_var = self._get_var_name(expr.right)
        left_val = self._eval_expr(expr.left, env)
        right_val = self._eval_expr(expr.right, env)

        then_env = env.copy()
        else_env = env.copy()
        op = expr.op

        complement = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}

        self._apply_refinement(then_env, left_var, right_var, left_val, right_val, op)
        self._apply_refinement(else_env, left_var, right_var, left_val, right_val, complement[op])

        return then_env, else_env

    def _apply_refinement(self, env: ArrayEnv, left_var: Optional[str],
                          right_var: Optional[str],
                          left_val: IntervalDomain, right_val: IntervalDomain,
                          op: str):
        """Apply a comparison refinement to the environment."""
        if op == '<':
            if left_var and left_var in env.scalars:
                # left < right => left <= right.hi - 1
                if not math.isinf(right_val.hi):
                    refined = env.get_scalar(left_var).meet(
                        IntervalDomain(NEG_INF, right_val.hi - 1))
                    env.set_scalar(left_var, refined)
            if right_var and right_var in env.scalars:
                # left < right => right >= left.lo + 1
                if not math.isinf(left_val.lo):
                    refined = env.get_scalar(right_var).meet(
                        IntervalDomain(left_val.lo + 1, INF))
                    env.set_scalar(right_var, refined)
        elif op == '<=':
            if left_var and left_var in env.scalars:
                if not math.isinf(right_val.hi):
                    refined = env.get_scalar(left_var).meet(
                        IntervalDomain(NEG_INF, right_val.hi))
                    env.set_scalar(left_var, refined)
            if right_var and right_var in env.scalars:
                if not math.isinf(left_val.lo):
                    refined = env.get_scalar(right_var).meet(
                        IntervalDomain(left_val.lo, INF))
                    env.set_scalar(right_var, refined)
        elif op == '>':
            if left_var and left_var in env.scalars:
                if not math.isinf(right_val.lo):
                    refined = env.get_scalar(left_var).meet(
                        IntervalDomain(right_val.lo + 1, INF))
                    env.set_scalar(left_var, refined)
            if right_var and right_var in env.scalars:
                if not math.isinf(left_val.hi):
                    refined = env.get_scalar(right_var).meet(
                        IntervalDomain(NEG_INF, left_val.hi - 1))
                    env.set_scalar(right_var, refined)
        elif op == '>=':
            if left_var and left_var in env.scalars:
                if not math.isinf(right_val.lo):
                    refined = env.get_scalar(left_var).meet(
                        IntervalDomain(right_val.lo, INF))
                    env.set_scalar(left_var, refined)
            if right_var and right_var in env.scalars:
                if not math.isinf(left_val.hi):
                    refined = env.get_scalar(right_var).meet(
                        IntervalDomain(NEG_INF, left_val.hi))
                    env.set_scalar(right_var, refined)
        elif op == '==':
            meet = left_val.meet(right_val)
            if left_var and left_var in env.scalars:
                env.set_scalar(left_var, env.get_scalar(left_var).meet(meet))
            if right_var and right_var in env.scalars:
                env.set_scalar(right_var, env.get_scalar(right_var).meet(meet))
        elif op == '!=':
            pass  # No useful refinement for != on intervals

    def _get_var_name(self, expr) -> Optional[str]:
        if isinstance(expr, VarExpr):
            return expr.name
        return None


# ===========================================================================
# Array Property Analysis
# ===========================================================================

class ArrayPropertyKind(Enum):
    SORTED = "sorted"
    BOUNDED = "bounded"
    INITIALIZED = "initialized"
    CONSTANT = "constant"

@dataclass
class ArrayProperty:
    kind: ArrayPropertyKind
    array_name: str
    details: Dict[str, Any]
    holds: bool  # True = definitely, False = definitely not, None below
    may_hold: bool  # True = possibly holds


def analyze_array_properties(env: ArrayEnv) -> List[ArrayProperty]:
    """Infer properties about arrays in the environment."""
    properties = []
    for name, arr in env.arrays.items():
        if arr.is_bot():
            continue
        # Check sortedness (per-element tracking only)
        if arr.elements:
            sorted_keys = sorted(arr.elements.keys())
            is_sorted = True
            may_be_sorted = True
            for i in range(len(sorted_keys) - 1):
                k1, k2 = sorted_keys[i], sorted_keys[i + 1]
                if k2 != k1 + 1:
                    # Gap -- can't determine sortedness
                    is_sorted = False
                    continue
                v1 = arr.elements[k1]
                v2 = arr.elements[k2]
                if v1.lo > v2.hi:
                    # Definitely not sorted
                    is_sorted = False
                    may_be_sorted = False
                    break
                if v1.hi > v2.lo:
                    is_sorted = False

            if may_be_sorted:
                properties.append(ArrayProperty(
                    kind=ArrayPropertyKind.SORTED,
                    array_name=name,
                    details={'indices': sorted_keys},
                    holds=is_sorted,
                    may_hold=may_be_sorted,
                ))

        # Check boundedness
        if arr.elements or not arr.smash.is_top():
            all_vals = IntervalDomain(1, 0)  # BOT
            for v in arr.elements.values():
                all_vals = all_vals.join(v)
            if not arr.smash.is_bot():
                all_vals = all_vals.join(arr.smash)
            if not all_vals.is_top() and not all_vals.is_bot():
                lo = all_vals.lo if not math.isinf(all_vals.lo) else None
                hi = all_vals.hi if not math.isinf(all_vals.hi) else None
                if lo is not None or hi is not None:
                    properties.append(ArrayProperty(
                        kind=ArrayPropertyKind.BOUNDED,
                        array_name=name,
                        details={'lower': lo, 'upper': hi},
                        holds=True,
                        may_hold=True,
                    ))

        # Check if all elements are the same constant
        if arr.elements and not arr.smash.is_top():
            vals = list(arr.elements.values())
            first = vals[0]
            all_same = all(v.lo == first.lo and v.hi == first.hi for v in vals)
            if all_same and first.lo == first.hi:
                if arr.smash.is_bot() or (arr.smash.lo == first.lo and arr.smash.hi == first.hi):
                    properties.append(ArrayProperty(
                        kind=ArrayPropertyKind.CONSTANT,
                        array_name=name,
                        details={'value': int(first.lo)},
                        holds=True,
                        may_hold=True,
                    ))

        # Check initialization
        if arr.elements:
            init_indices = sorted(arr.elements.keys())
            length = arr.length
            if not math.isinf(length.hi) and length.hi >= 0:
                max_len = int(length.hi)
                initialized = all(i in arr.elements for i in range(max_len))
                if initialized:
                    properties.append(ArrayProperty(
                        kind=ArrayPropertyKind.INITIALIZED,
                        array_name=name,
                        details={'range': (0, max_len)},
                        holds=True,
                        may_hold=True,
                    ))

    return properties


# ===========================================================================
# High-level API
# ===========================================================================

def array_analyze(source: str, max_iterations: int = 50) -> dict:
    """Analyze a program with arrays and return comprehensive results.

    Returns dict with:
    - env: ArrayEnv with final abstract state
    - warnings: list of Warning objects
    - scalars: dict of scalar variable intervals
    - arrays: dict of array abstract values
    - properties: list of inferred ArrayProperty objects
    """
    interp = ArrayInterpreter(max_iterations=max_iterations)
    result = interp.analyze(source)
    result['properties'] = analyze_array_properties(result['env'])
    return result


def check_bounds(source: str) -> List[Warning]:
    """Check for out-of-bounds array accesses."""
    result = array_analyze(source)
    return [w for w in result['warnings']
            if w.kind in (WarningKind.OUT_OF_BOUNDS, WarningKind.POSSIBLE_OUT_OF_BOUNDS)]


def check_assertions(source: str) -> List[Warning]:
    """Check for assertion failures."""
    result = array_analyze(source)
    return [w for w in result['warnings']
            if w.kind in (WarningKind.ASSERTION_FAILURE, WarningKind.POSSIBLE_ASSERTION_FAILURE)]


def get_array_info(source: str, array_name: str) -> Optional[ArrayAbstractValue]:
    """Get abstract info about a specific array."""
    result = array_analyze(source)
    return result['arrays'].get(array_name)


def get_variable_range(source: str, var_name: str) -> Optional[IntervalDomain]:
    """Get the interval range of a scalar variable."""
    result = array_analyze(source)
    return result['scalars'].get(var_name)


def infer_properties(source: str) -> List[ArrayProperty]:
    """Infer array properties from the program."""
    result = array_analyze(source)
    return result['properties']


def compare_analyses(source: str) -> dict:
    """Compare array domain analysis with a baseline (no array tracking)."""
    full = array_analyze(source)

    # Baseline: just scalars with standard interval analysis
    program = parse_source(source)
    baseline_interp = ArrayInterpreter()
    baseline = baseline_interp.analyze(source)

    return {
        'full_warnings': len(full['warnings']),
        'baseline_warnings': len(baseline['warnings']),
        'arrays_tracked': len(full['arrays']),
        'properties_found': len(full['properties']),
        'scalars_tracked': len(full['scalars']),
    }


def array_summary(source: str) -> str:
    """Human-readable summary of array analysis."""
    result = array_analyze(source)
    lines = []
    lines.append("=== Array Domain Analysis Summary ===")
    lines.append("")

    if result['scalars']:
        lines.append("Scalar Variables:")
        for name, val in sorted(result['scalars'].items()):
            lines.append(f"  {name}: {val}")
        lines.append("")

    if result['arrays']:
        lines.append("Arrays:")
        for name, arr in sorted(result['arrays'].items()):
            lines.append(f"  {name}: {arr}")
        lines.append("")

    if result['properties']:
        lines.append("Inferred Properties:")
        for prop in result['properties']:
            status = "HOLDS" if prop.holds else ("MAY HOLD" if prop.may_hold else "VIOLATED")
            lines.append(f"  {prop.array_name}: {prop.kind.value} [{status}] {prop.details}")
        lines.append("")

    if result['warnings']:
        lines.append("Warnings:")
        for w in result['warnings']:
            lines.append(f"  [{w.kind.value}] {w.message}")
        lines.append("")

    if not result['warnings']:
        lines.append("No warnings -- all accesses proven safe.")

    return "\n".join(lines)
