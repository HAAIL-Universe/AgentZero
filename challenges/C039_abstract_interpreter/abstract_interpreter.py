"""
Abstract Interpreter for Stack VM Language
Challenge C039 -- AgentZero Session 040

Composes: C010 (Stack VM -- parser/AST only)

Abstract interpretation approximates program behavior over abstract domains
rather than executing with concrete values. This enables:
  - Proving properties hold for ALL inputs (not just tested ones)
  - Analyzing loops with unknown iteration counts (via widening)
  - Detecting division by zero, unreachable code, overflow
  - Constant propagation, sign analysis, interval analysis

Architecture:
  Source -> Parser (C010) -> AST -> Abstract Interpreter -> Analysis Results

Abstract Domains:
  1. Sign domain: {Neg, Zero, Pos, NonNeg, NonPos, Top, Bot}
  2. Interval domain: [lo, hi] with widening
  3. Constant propagation domain: {Const(v), Top, Bot}

Analysis capabilities:
  - Forward abstract interpretation over AST
  - Fixpoint iteration with widening for loops
  - Division-by-zero detection
  - Unreachable code detection
  - Variable range inference
  - Dead assignment detection
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto
from copy import deepcopy

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program,
    IntLit, FloatLit, StringLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)


# ============================================================
# Abstract Values -- Sign Domain
# ============================================================

class Sign(Enum):
    BOT = auto()     # unreachable / no value
    NEG = auto()     # strictly negative
    ZERO = auto()    # exactly zero
    POS = auto()     # strictly positive
    NON_NEG = auto() # >= 0
    NON_POS = auto() # <= 0
    TOP = auto()     # any value

    def __repr__(self):
        return self.name


def sign_join(a: Sign, b: Sign) -> Sign:
    """Least upper bound in the sign lattice."""
    if a == Sign.BOT:
        return b
    if b == Sign.BOT:
        return a
    if a == b:
        return a
    if a == Sign.TOP or b == Sign.TOP:
        return Sign.TOP

    # Build the set of concrete signs
    def expand(s):
        if s == Sign.NEG:
            return {Sign.NEG}
        if s == Sign.ZERO:
            return {Sign.ZERO}
        if s == Sign.POS:
            return {Sign.POS}
        if s == Sign.NON_NEG:
            return {Sign.ZERO, Sign.POS}
        if s == Sign.NON_POS:
            return {Sign.ZERO, Sign.NEG}
        return {Sign.NEG, Sign.ZERO, Sign.POS}

    combined = expand(a) | expand(b)

    if combined == {Sign.NEG}:
        return Sign.NEG
    if combined == {Sign.ZERO}:
        return Sign.ZERO
    if combined == {Sign.POS}:
        return Sign.POS
    if combined == {Sign.ZERO, Sign.POS}:
        return Sign.NON_NEG
    if combined == {Sign.ZERO, Sign.NEG}:
        return Sign.NON_POS
    return Sign.TOP


def sign_add(a: Sign, b: Sign) -> Sign:
    if a == Sign.BOT or b == Sign.BOT:
        return Sign.BOT
    if a == Sign.ZERO:
        return b
    if b == Sign.ZERO:
        return a
    if a == Sign.POS and b == Sign.POS:
        return Sign.POS
    if a == Sign.NEG and b == Sign.NEG:
        return Sign.NEG
    if a == Sign.POS and b == Sign.NEG:
        return Sign.TOP
    if a == Sign.NEG and b == Sign.POS:
        return Sign.TOP
    if a == Sign.NON_NEG and b == Sign.NON_NEG:
        return Sign.NON_NEG
    if a == Sign.NON_POS and b == Sign.NON_POS:
        return Sign.NON_POS
    return Sign.TOP


def sign_sub(a: Sign, b: Sign) -> Sign:
    return sign_add(a, sign_neg(b))


def sign_neg(a: Sign) -> Sign:
    if a == Sign.BOT:
        return Sign.BOT
    if a == Sign.NEG:
        return Sign.POS
    if a == Sign.POS:
        return Sign.NEG
    if a == Sign.ZERO:
        return Sign.ZERO
    if a == Sign.NON_NEG:
        return Sign.NON_POS
    if a == Sign.NON_POS:
        return Sign.NON_NEG
    return Sign.TOP


def sign_mul(a: Sign, b: Sign) -> Sign:
    if a == Sign.BOT or b == Sign.BOT:
        return Sign.BOT
    if a == Sign.ZERO or b == Sign.ZERO:
        return Sign.ZERO
    if a == Sign.POS and b == Sign.POS:
        return Sign.POS
    if a == Sign.NEG and b == Sign.NEG:
        return Sign.POS
    if a == Sign.POS and b == Sign.NEG:
        return Sign.NEG
    if a == Sign.NEG and b == Sign.POS:
        return Sign.NEG
    # NON_NEG * POS = NON_NEG, etc
    if a == Sign.NON_NEG and b == Sign.POS:
        return Sign.NON_NEG
    if a == Sign.POS and b == Sign.NON_NEG:
        return Sign.NON_NEG
    if a == Sign.NON_NEG and b == Sign.NON_NEG:
        return Sign.NON_NEG
    if a == Sign.NON_POS and b == Sign.POS:
        return Sign.NON_POS
    if a == Sign.POS and b == Sign.NON_POS:
        return Sign.NON_POS
    if a == Sign.NON_POS and b == Sign.NEG:
        return Sign.NON_NEG
    if a == Sign.NEG and b == Sign.NON_POS:
        return Sign.NON_NEG
    if a == Sign.NON_POS and b == Sign.NON_POS:
        return Sign.NON_NEG
    if a == Sign.NEG and b == Sign.NON_NEG:
        return Sign.NON_POS
    if a == Sign.NON_NEG and b == Sign.NEG:
        return Sign.NON_POS
    return Sign.TOP


def sign_div(a: Sign, b: Sign) -> Sign:
    """Division -- result sign follows multiplication rules (for integers)."""
    if a == Sign.BOT or b == Sign.BOT:
        return Sign.BOT
    if b == Sign.ZERO:
        return Sign.BOT  # division by zero -- undefined
    if a == Sign.ZERO:
        return Sign.ZERO
    # For integer division, sign follows multiplication rules but magnitude uncertain
    if a == Sign.POS and b == Sign.POS:
        return Sign.NON_NEG  # could be 0 if b > a
    if a == Sign.NEG and b == Sign.NEG:
        return Sign.NON_NEG
    if a == Sign.POS and b == Sign.NEG:
        return Sign.NON_POS
    if a == Sign.NEG and b == Sign.POS:
        return Sign.NON_POS
    return Sign.TOP


def sign_mod(a: Sign, b: Sign) -> Sign:
    if a == Sign.BOT or b == Sign.BOT:
        return Sign.BOT
    if b == Sign.ZERO:
        return Sign.BOT
    if a == Sign.ZERO:
        return Sign.ZERO
    # Python mod: result has sign of divisor
    return Sign.TOP


def sign_contains_zero(s: Sign) -> bool:
    """Does this sign abstract value include zero?"""
    return s in (Sign.ZERO, Sign.NON_NEG, Sign.NON_POS, Sign.TOP)


def sign_from_value(v) -> Sign:
    """Concrete value to sign."""
    if isinstance(v, bool):
        return Sign.NON_NEG  # True=1 (POS), False=0 (ZERO)
    if isinstance(v, (int, float)):
        if v > 0:
            return Sign.POS
        elif v < 0:
            return Sign.NEG
        else:
            return Sign.ZERO
    return Sign.TOP


# ============================================================
# Abstract Values -- Interval Domain
# ============================================================

INF = float('inf')
NEG_INF = float('-inf')


@dataclass(frozen=True)
class Interval:
    """Interval [lo, hi]. Bot represented by lo > hi."""
    lo: float
    hi: float

    def __repr__(self):
        if self.is_bot():
            return "Bot"
        if self.lo == NEG_INF and self.hi == INF:
            return "Top"
        lo_s = "-inf" if self.lo == NEG_INF else str(self.lo)
        hi_s = "inf" if self.hi == INF else str(self.hi)
        return f"[{lo_s}, {hi_s}]"

    def is_bot(self):
        return self.lo > self.hi

    def is_top(self):
        return self.lo == NEG_INF and self.hi == INF

    def contains_zero(self):
        return not self.is_bot() and self.lo <= 0 <= self.hi

    def contains(self, v):
        return not self.is_bot() and self.lo <= v <= self.hi


INTERVAL_BOT = Interval(1, 0)
INTERVAL_TOP = Interval(NEG_INF, INF)


def interval_from_value(v) -> Interval:
    if isinstance(v, bool):
        v = 1 if v else 0
    if isinstance(v, (int, float)):
        return Interval(float(v), float(v))
    return INTERVAL_TOP


def interval_join(a: Interval, b: Interval) -> Interval:
    if a.is_bot():
        return b
    if b.is_bot():
        return a
    return Interval(min(a.lo, b.lo), max(a.hi, b.hi))


def interval_meet(a: Interval, b: Interval) -> Interval:
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    lo = max(a.lo, b.lo)
    hi = min(a.hi, b.hi)
    if lo > hi:
        return INTERVAL_BOT
    return Interval(lo, hi)


def interval_widen(old: Interval, new: Interval) -> Interval:
    """Widening operator -- ensures convergence of fixpoint iteration."""
    if old.is_bot():
        return new
    if new.is_bot():
        return old
    lo = old.lo if new.lo >= old.lo else NEG_INF
    hi = old.hi if new.hi <= old.hi else INF
    return Interval(lo, hi)


def interval_add(a: Interval, b: Interval) -> Interval:
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    return Interval(a.lo + b.lo, a.hi + b.hi)


def interval_sub(a: Interval, b: Interval) -> Interval:
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    return Interval(a.lo - b.hi, a.hi - b.lo)


def interval_neg(a: Interval) -> Interval:
    if a.is_bot():
        return INTERVAL_BOT
    return Interval(-a.hi, -a.lo)


def interval_mul(a: Interval, b: Interval) -> Interval:
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    products = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    # Handle inf * 0 = nan
    products = [0.0 if math.isnan(p) else p for p in products]
    return Interval(min(products), max(products))


def interval_div(a: Interval, b: Interval) -> Interval:
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    if b.contains_zero():
        if b.lo == 0 and b.hi == 0:
            return INTERVAL_BOT  # definite division by zero
        # Split around zero
        parts = []
        if b.lo < 0:
            parts.append(interval_div(a, Interval(b.lo, -1)))
        if b.hi > 0:
            parts.append(interval_div(a, Interval(1, b.hi)))
        if not parts:
            return INTERVAL_BOT
        result = parts[0]
        for p in parts[1:]:
            result = interval_join(result, p)
        return result
    # b does not contain zero
    quotients = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi]
    return Interval(min(quotients), max(quotients))


def interval_mod(a: Interval, b: Interval) -> Interval:
    if a.is_bot() or b.is_bot():
        return INTERVAL_BOT
    if b.contains_zero():
        if b.lo == 0 and b.hi == 0:
            return INTERVAL_BOT
        return INTERVAL_TOP
    # Conservative: result in [0, |b|-1] for positive a, broader otherwise
    return INTERVAL_TOP


# ============================================================
# Abstract Values -- Constant Propagation Domain
# ============================================================

class ConstBot:
    """No value (unreachable)."""
    def __repr__(self):
        return "ConstBot"
    def __eq__(self, other):
        return isinstance(other, ConstBot)
    def __hash__(self):
        return hash("ConstBot")

class ConstTop:
    """Unknown value."""
    def __repr__(self):
        return "ConstTop"
    def __eq__(self, other):
        return isinstance(other, ConstTop)
    def __hash__(self):
        return hash("ConstTop")

@dataclass(frozen=True)
class ConstVal:
    """Known constant value."""
    value: Any
    def __repr__(self):
        return f"Const({self.value})"
    def __eq__(self, other):
        if not isinstance(other, ConstVal):
            return False
        # Type-aware comparison (True != 1)
        return type(self.value) == type(other.value) and self.value == other.value
    def __hash__(self):
        return hash((type(self.value).__name__, self.value))


CONST_BOT = ConstBot()
CONST_TOP = ConstTop()


def const_join(a, b):
    if isinstance(a, ConstBot):
        return b
    if isinstance(b, ConstBot):
        return a
    if isinstance(a, ConstTop) or isinstance(b, ConstTop):
        return CONST_TOP
    if a == b:
        return a
    return CONST_TOP


def const_from_value(v):
    return ConstVal(v)


# ============================================================
# Abstract Environment
# ============================================================

class AbstractEnv:
    """Maps variable names to abstract values."""

    def __init__(self):
        self.signs = {}      # name -> Sign
        self.intervals = {}  # name -> Interval
        self.consts = {}     # name -> ConstVal/ConstTop/ConstBot

    def copy(self):
        env = AbstractEnv()
        env.signs = dict(self.signs)
        env.intervals = dict(self.intervals)
        env.consts = dict(self.consts)
        return env

    def set(self, name, sign=None, interval=None, const=None):
        if sign is not None:
            self.signs[name] = sign
        if interval is not None:
            self.intervals[name] = interval
        if const is not None:
            self.consts[name] = const

    def set_from_value(self, name, value):
        self.signs[name] = sign_from_value(value)
        self.intervals[name] = interval_from_value(value)
        self.consts[name] = const_from_value(value)

    def set_top(self, name):
        self.signs[name] = Sign.TOP
        self.intervals[name] = INTERVAL_TOP
        self.consts[name] = CONST_TOP

    def set_bot(self, name):
        self.signs[name] = Sign.BOT
        self.intervals[name] = INTERVAL_BOT
        self.consts[name] = CONST_BOT

    def get_sign(self, name) -> Sign:
        return self.signs.get(name, Sign.TOP)

    def get_interval(self, name) -> Interval:
        return self.intervals.get(name, INTERVAL_TOP)

    def get_const(self, name):
        return self.consts.get(name, CONST_TOP)

    def join(self, other):
        """Join two environments (least upper bound)."""
        result = AbstractEnv()
        all_names = set(self.signs.keys()) | set(other.signs.keys())
        for name in all_names:
            result.signs[name] = sign_join(
                self.signs.get(name, Sign.BOT),
                other.signs.get(name, Sign.BOT)
            )
            result.intervals[name] = interval_join(
                self.intervals.get(name, INTERVAL_BOT),
                other.intervals.get(name, INTERVAL_BOT)
            )
            result.consts[name] = const_join(
                self.consts.get(name, CONST_BOT),
                other.consts.get(name, CONST_BOT)
            )
        return result

    def widen(self, other):
        """Widen this environment with another (for loop convergence)."""
        result = AbstractEnv()
        all_names = set(self.signs.keys()) | set(other.signs.keys())
        for name in all_names:
            result.signs[name] = sign_join(
                self.signs.get(name, Sign.BOT),
                other.signs.get(name, Sign.BOT)
            )
            result.intervals[name] = interval_widen(
                self.intervals.get(name, INTERVAL_BOT),
                other.intervals.get(name, INTERVAL_BOT)
            )
            result.consts[name] = const_join(
                self.consts.get(name, CONST_BOT),
                other.consts.get(name, CONST_BOT)
            )
        return result

    def equals(self, other):
        """Check if two environments are equal."""
        all_names = set(self.signs.keys()) | set(other.signs.keys())
        for name in all_names:
            if self.signs.get(name, Sign.BOT) != other.signs.get(name, Sign.BOT):
                return False
            if self.intervals.get(name, INTERVAL_BOT) != other.intervals.get(name, INTERVAL_BOT):
                return False
            if self.consts.get(name, CONST_BOT) != other.consts.get(name, CONST_BOT):
                return False
        return True

    def __repr__(self):
        parts = []
        for name in sorted(self.signs.keys()):
            s = self.signs.get(name, Sign.TOP)
            i = self.intervals.get(name, INTERVAL_TOP)
            c = self.consts.get(name, CONST_TOP)
            parts.append(f"{name}: sign={s}, interval={i}, const={c}")
        return "{" + ", ".join(parts) + "}"


# ============================================================
# Abstract Expression Evaluator
# ============================================================

@dataclass
class AbstractValue:
    """Result of evaluating an expression abstractly."""
    sign: Sign
    interval: Interval
    const: object  # ConstVal, ConstTop, ConstBot

    @staticmethod
    def from_value(v):
        return AbstractValue(
            sign=sign_from_value(v),
            interval=interval_from_value(v),
            const=const_from_value(v)
        )

    @staticmethod
    def top():
        return AbstractValue(Sign.TOP, INTERVAL_TOP, CONST_TOP)

    @staticmethod
    def bot():
        return AbstractValue(Sign.BOT, INTERVAL_BOT, CONST_BOT)

    def is_bot(self):
        return self.sign == Sign.BOT

    def __repr__(self):
        return f"AV(sign={self.sign}, interval={self.interval}, const={self.const})"


# ============================================================
# Warnings
# ============================================================

class WarningKind(Enum):
    DIVISION_BY_ZERO = "division_by_zero"
    POSSIBLE_DIVISION_BY_ZERO = "possible_division_by_zero"
    UNREACHABLE_CODE = "unreachable_code"
    DEAD_ASSIGNMENT = "dead_assignment"
    UNREACHABLE_BRANCH = "unreachable_branch"
    NARROWING_OVERFLOW = "narrowing_overflow"

@dataclass
class Warning:
    kind: WarningKind
    message: str
    line: int = 0
    node: Any = None

    def __repr__(self):
        loc = f" (line {self.line})" if self.line else ""
        return f"Warning({self.kind.value}{loc}): {self.message}"


# ============================================================
# Abstract Interpreter
# ============================================================

class AbstractInterpreter:
    """Forward abstract interpreter over the C010 AST."""

    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations
        self.warnings = []
        self.functions = {}  # name -> FnDecl
        self.var_reads = set()   # variable names that were read
        self.var_writes = {}     # name -> list of line numbers

    def analyze(self, source: str) -> dict:
        """Analyze source code and return results."""
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()

        env = AbstractEnv()
        self.warnings = []
        self.functions = {}
        self.var_reads = set()
        self.var_writes = {}

        # First pass: collect function declarations
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt

        # Interpret statements
        env = self._interpret_stmts(program.stmts, env)

        # Dead assignment detection
        self._detect_dead_assignments(env)

        return {
            'env': env,
            'warnings': list(self.warnings),
            'functions': dict(self.functions),
            'var_reads': set(self.var_reads),
            'var_writes': dict(self.var_writes),
        }

    def _warn(self, kind, message, line=0, node=None):
        self.warnings.append(Warning(kind, message, line, node))

    def _get_line(self, node):
        """Try to extract line number from a node."""
        if hasattr(node, 'line'):
            return node.line
        return 0

    def _interpret_stmts(self, stmts, env: AbstractEnv) -> AbstractEnv:
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env)
        return env

    def _interpret_stmt(self, stmt, env: AbstractEnv) -> AbstractEnv:
        if isinstance(stmt, LetDecl):
            return self._interpret_let(stmt, env)
        elif isinstance(stmt, Assign):
            return self._interpret_assign(stmt, env)
        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env)
        elif isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, env)
        elif isinstance(stmt, Block):
            return self._interpret_stmts(stmt.stmts, env)
        elif isinstance(stmt, PrintStmt):
            self._eval_expr(stmt.value, env)
            return env
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                self._eval_expr(stmt.value, env)
            return env
        elif isinstance(stmt, FnDecl):
            # Already collected in first pass
            return env
        else:
            # Expression statement
            self._eval_expr(stmt, env)
            return env

    def _interpret_let(self, stmt: LetDecl, env: AbstractEnv) -> AbstractEnv:
        val = self._eval_expr(stmt.value, env)
        env = env.copy()
        env.set(stmt.name, sign=val.sign, interval=val.interval, const=val.const)
        self._record_write(stmt.name, self._get_line(stmt))
        return env

    def _interpret_assign(self, stmt: Assign, env: AbstractEnv) -> AbstractEnv:
        val = self._eval_expr(stmt.value, env)
        env = env.copy()
        env.set(stmt.name, sign=val.sign, interval=val.interval, const=val.const)
        self._record_write(stmt.name, self._get_line(stmt))
        return env

    def _interpret_if(self, stmt: IfStmt, env: AbstractEnv) -> AbstractEnv:
        cond = self._eval_expr(stmt.cond, env)

        # Check if condition is constant
        if isinstance(cond.const, ConstVal):
            if cond.const.value:
                # Always true -- else branch unreachable
                if stmt.else_body:
                    self._warn(WarningKind.UNREACHABLE_BRANCH,
                               "else branch is unreachable (condition always true)",
                               self._get_line(stmt))
                return self._interpret_stmt(stmt.then_body, env)
            else:
                # Always false -- then branch unreachable
                self._warn(WarningKind.UNREACHABLE_BRANCH,
                           "then branch is unreachable (condition always false)",
                           self._get_line(stmt))
                if stmt.else_body:
                    return self._interpret_stmt(stmt.else_body, env)
                return env

        # Both branches possible -- refine environment for each
        then_env = self._refine_env_for_condition(stmt.cond, env, True)
        else_env = self._refine_env_for_condition(stmt.cond, env, False)

        then_env = self._interpret_stmt(stmt.then_body, then_env)
        if stmt.else_body:
            else_env = self._interpret_stmt(stmt.else_body, else_env)

        return then_env.join(else_env)

    def _interpret_while(self, stmt: WhileStmt, env: AbstractEnv) -> AbstractEnv:
        """Fixpoint iteration with widening for loops."""
        # Iterate until fixpoint
        current_env = env
        for i in range(self.max_iterations):
            # Evaluate condition in current env
            cond = self._eval_expr(stmt.cond, current_env)

            # If condition is definitely false, loop never executes
            if isinstance(cond.const, ConstVal) and not cond.const.value:
                return current_env

            # Execute body
            body_env = self._refine_env_for_condition(stmt.cond, current_env, True)
            body_env = self._interpret_stmt(stmt.body, body_env)

            # Join with the entry state (because loop might not execute)
            next_env = current_env.widen(body_env)

            # Check convergence
            if next_env.equals(current_env):
                break
            current_env = next_env

        # After loop: join env where condition is false
        exit_env = self._refine_env_for_condition(stmt.cond, current_env, False)
        return exit_env

    def _refine_env_for_condition(self, cond, env: AbstractEnv, branch: bool) -> AbstractEnv:
        """Refine abstract environment based on a condition being true/false."""
        env = env.copy()

        if isinstance(cond, BinOp):
            if isinstance(cond.left, Var) and self._is_numeric_literal(cond.right):
                name = cond.left.name
                val = self._literal_value(cond.right)
                current = env.get_interval(name)
                op = cond.op

                if branch:  # condition is true
                    refined = self._refine_interval(current, op, val)
                else:  # condition is false (negate)
                    neg_op = self._negate_op(op)
                    refined = self._refine_interval(current, neg_op, val)

                env.intervals[name] = refined
                # Update sign based on refined interval
                env.signs[name] = self._sign_from_interval(refined)

            elif isinstance(cond.right, Var) and self._is_numeric_literal(cond.left):
                name = cond.right.name
                val = self._literal_value(cond.left)
                current = env.get_interval(name)
                op = self._flip_op(cond.op)

                if branch:
                    refined = self._refine_interval(current, op, val)
                else:
                    neg_op = self._negate_op(op)
                    refined = self._refine_interval(current, neg_op, val)

                env.intervals[name] = refined
                env.signs[name] = self._sign_from_interval(refined)

        return env

    def _refine_interval(self, current: Interval, op: str, val: float) -> Interval:
        """Refine interval based on comparison operator and value."""
        if current.is_bot():
            return INTERVAL_BOT

        v = float(val)
        if op == '<':
            constraint = Interval(NEG_INF, v - 1)
        elif op == '<=':
            constraint = Interval(NEG_INF, v)
        elif op == '>':
            constraint = Interval(v + 1, INF)
        elif op == '>=':
            constraint = Interval(v, INF)
        elif op == '==':
            constraint = Interval(v, v)
        elif op == '!=':
            # Can't easily represent "not equal" as interval -- keep current
            return current
        else:
            return current

        return interval_meet(current, constraint)

    def _negate_op(self, op):
        return {'<': '>=', '<=': '>', '>': '<=', '>=': '<',
                '==': '!=', '!=': '=='}.get(op, op)

    def _flip_op(self, op):
        return {'<': '>', '<=': '>=', '>': '<', '>=': '<=',
                '==': '==', '!=': '!='}.get(op, op)

    def _is_numeric_literal(self, node):
        return isinstance(node, (IntLit, FloatLit))

    def _literal_value(self, node):
        if isinstance(node, (IntLit, FloatLit)):
            return node.value
        return None

    def _sign_from_interval(self, interval: Interval) -> Sign:
        if interval.is_bot():
            return Sign.BOT
        if interval.lo > 0:
            return Sign.POS
        if interval.hi < 0:
            return Sign.NEG
        if interval.lo == 0 and interval.hi == 0:
            return Sign.ZERO
        if interval.lo >= 0:
            return Sign.NON_NEG
        if interval.hi <= 0:
            return Sign.NON_POS
        return Sign.TOP

    def _eval_expr(self, expr, env: AbstractEnv) -> AbstractValue:
        """Evaluate an expression in abstract domain."""
        if isinstance(expr, IntLit):
            return AbstractValue.from_value(expr.value)
        elif isinstance(expr, FloatLit):
            return AbstractValue.from_value(expr.value)
        elif isinstance(expr, StringLit):
            return AbstractValue(Sign.TOP, INTERVAL_TOP, const_from_value(expr.value))
        elif isinstance(expr, BoolLit):
            v = expr.value
            return AbstractValue.from_value(v)
        elif isinstance(expr, Var):
            self.var_reads.add(expr.name)
            return AbstractValue(
                env.get_sign(expr.name),
                env.get_interval(expr.name),
                env.get_const(expr.name)
            )
        elif isinstance(expr, UnaryOp):
            return self._eval_unary(expr, env)
        elif isinstance(expr, BinOp):
            return self._eval_binary(expr, env)
        elif isinstance(expr, CallExpr):
            return self._eval_call(expr, env)
        elif isinstance(expr, Assign):
            val = self._eval_expr(expr.value, env)
            env.set(expr.name, sign=val.sign, interval=val.interval, const=val.const)
            self._record_write(expr.name, self._get_line(expr))
            return val
        else:
            return AbstractValue.top()

    def _eval_unary(self, expr: UnaryOp, env: AbstractEnv) -> AbstractValue:
        operand = self._eval_expr(expr.operand, env)
        if expr.op == '-':
            return AbstractValue(
                sign_neg(operand.sign),
                interval_neg(operand.interval),
                self._const_unary_neg(operand.const)
            )
        elif expr.op == 'not':
            return AbstractValue(Sign.NON_NEG, Interval(0, 1),
                                 self._const_unary_not(operand.const))
        return AbstractValue.top()

    def _eval_binary(self, expr: BinOp, env: AbstractEnv) -> AbstractValue:
        left = self._eval_expr(expr.left, env)
        right = self._eval_expr(expr.right, env)
        op = expr.op

        if op == '+':
            return AbstractValue(
                sign_add(left.sign, right.sign),
                interval_add(left.interval, right.interval),
                self._const_binop(left.const, right.const, op)
            )
        elif op == '-':
            return AbstractValue(
                sign_sub(left.sign, right.sign),
                interval_sub(left.interval, right.interval),
                self._const_binop(left.const, right.const, op)
            )
        elif op == '*':
            return AbstractValue(
                sign_mul(left.sign, right.sign),
                interval_mul(left.interval, right.interval),
                self._const_binop(left.const, right.const, op)
            )
        elif op == '/':
            self._check_division(right, expr)
            return AbstractValue(
                sign_div(left.sign, right.sign),
                interval_div(left.interval, right.interval),
                self._const_binop(left.const, right.const, op)
            )
        elif op == '%':
            self._check_division(right, expr)
            return AbstractValue(
                sign_mod(left.sign, right.sign),
                interval_mod(left.interval, right.interval),
                self._const_binop(left.const, right.const, op)
            )
        elif op in ('<', '>', '<=', '>=', '==', '!='):
            return self._eval_comparison(left, right, op)
        elif op == 'and':
            return AbstractValue(Sign.NON_NEG, Interval(0, 1),
                                 self._const_binop(left.const, right.const, op))
        elif op == 'or':
            return AbstractValue(Sign.NON_NEG, Interval(0, 1),
                                 self._const_binop(left.const, right.const, op))
        return AbstractValue.top()

    def _eval_comparison(self, left: AbstractValue, right: AbstractValue, op: str) -> AbstractValue:
        """Evaluate comparison, returning boolean abstract value."""
        const_result = self._const_binop(left.const, right.const, op)

        # Try to determine result from intervals
        if not left.interval.is_bot() and not right.interval.is_bot():
            if op == '<':
                if left.interval.hi < right.interval.lo:
                    return AbstractValue(Sign.POS, Interval(1, 1), ConstVal(True))
                if left.interval.lo >= right.interval.hi:
                    return AbstractValue(Sign.ZERO, Interval(0, 0), ConstVal(False))
            elif op == '>':
                if left.interval.lo > right.interval.hi:
                    return AbstractValue(Sign.POS, Interval(1, 1), ConstVal(True))
                if left.interval.hi <= right.interval.lo:
                    return AbstractValue(Sign.ZERO, Interval(0, 0), ConstVal(False))
            elif op == '<=':
                if left.interval.hi <= right.interval.lo:
                    return AbstractValue(Sign.POS, Interval(1, 1), ConstVal(True))
                if left.interval.lo > right.interval.hi:
                    return AbstractValue(Sign.ZERO, Interval(0, 0), ConstVal(False))
            elif op == '>=':
                if left.interval.lo >= right.interval.hi:
                    return AbstractValue(Sign.POS, Interval(1, 1), ConstVal(True))
                if left.interval.hi < right.interval.lo:
                    return AbstractValue(Sign.ZERO, Interval(0, 0), ConstVal(False))
            elif op == '==':
                if left.interval.lo == left.interval.hi == right.interval.lo == right.interval.hi:
                    if left.interval.lo == right.interval.lo:
                        return AbstractValue(Sign.POS, Interval(1, 1), ConstVal(True))
                # Disjoint intervals => definitely not equal
                if left.interval.hi < right.interval.lo or right.interval.hi < left.interval.lo:
                    return AbstractValue(Sign.ZERO, Interval(0, 0), ConstVal(False))
            elif op == '!=':
                if left.interval.hi < right.interval.lo or right.interval.hi < left.interval.lo:
                    return AbstractValue(Sign.POS, Interval(1, 1), ConstVal(True))
                if left.interval.lo == left.interval.hi == right.interval.lo == right.interval.hi:
                    if left.interval.lo == right.interval.lo:
                        return AbstractValue(Sign.ZERO, Interval(0, 0), ConstVal(False))

        return AbstractValue(Sign.NON_NEG, Interval(0, 1), const_result)

    def _eval_call(self, expr: CallExpr, env: AbstractEnv) -> AbstractValue:
        """Evaluate function call abstractly."""
        # Evaluate arguments
        for arg in expr.args:
            self._eval_expr(arg, env)

        fn_name = expr.callee if isinstance(expr.callee, str) else (expr.callee.name if isinstance(expr.callee, Var) else None)

        if fn_name and fn_name in self.functions:
            fn = self.functions[fn_name]
            # Create function environment with abstract args
            fn_env = env.copy()
            for i, param in enumerate(fn.params):
                if i < len(expr.args):
                    val = self._eval_expr(expr.args[i], env)
                    fn_env.set(param, sign=val.sign, interval=val.interval, const=val.const)
                else:
                    fn_env.set_top(param)

            # Analyze function body (simplified -- no recursion detection)
            self._interpret_stmt(fn.body, fn_env)
            return AbstractValue.top()  # Conservative for return value

        return AbstractValue.top()

    def _check_division(self, divisor: AbstractValue, expr):
        """Check for division by zero."""
        if divisor.sign == Sign.ZERO or (isinstance(divisor.const, ConstVal) and divisor.const.value == 0):
            self._warn(WarningKind.DIVISION_BY_ZERO,
                       "definite division by zero",
                       self._get_line(expr), expr)
        elif sign_contains_zero(divisor.sign) or divisor.interval.contains_zero():
            self._warn(WarningKind.POSSIBLE_DIVISION_BY_ZERO,
                       "possible division by zero",
                       self._get_line(expr), expr)

    def _const_unary_neg(self, c):
        if isinstance(c, ConstVal) and isinstance(c.value, (int, float)):
            return ConstVal(-c.value)
        if isinstance(c, ConstBot):
            return CONST_BOT
        return CONST_TOP

    def _const_unary_not(self, c):
        if isinstance(c, ConstVal):
            return ConstVal(not c.value)
        if isinstance(c, ConstBot):
            return CONST_BOT
        return CONST_TOP

    def _const_binop(self, a, b, op):
        if isinstance(a, ConstBot) or isinstance(b, ConstBot):
            return CONST_BOT
        if isinstance(a, ConstVal) and isinstance(b, ConstVal):
            try:
                av, bv = a.value, b.value
                if op == '+':
                    return ConstVal(av + bv)
                elif op == '-':
                    return ConstVal(av - bv)
                elif op == '*':
                    return ConstVal(av * bv)
                elif op == '/':
                    if bv == 0:
                        return CONST_BOT
                    if isinstance(av, int) and isinstance(bv, int):
                        return ConstVal(av // bv)
                    return ConstVal(av / bv)
                elif op == '%':
                    if bv == 0:
                        return CONST_BOT
                    return ConstVal(av % bv)
                elif op == '<':
                    return ConstVal(av < bv)
                elif op == '>':
                    return ConstVal(av > bv)
                elif op == '<=':
                    return ConstVal(av <= bv)
                elif op == '>=':
                    return ConstVal(av >= bv)
                elif op == '==':
                    return ConstVal(av == bv)
                elif op == '!=':
                    return ConstVal(av != bv)
                elif op == 'and':
                    return ConstVal(av and bv)
                elif op == 'or':
                    return ConstVal(av or bv)
            except Exception:
                return CONST_TOP
        return CONST_TOP

    def _record_write(self, name, line):
        if name not in self.var_writes:
            self.var_writes[name] = []
        self.var_writes[name].append(line)

    def _detect_dead_assignments(self, env):
        """Detect variables written but never read."""
        for name in self.var_writes:
            if name not in self.var_reads and name not in self.functions:
                lines = self.var_writes[name]
                self._warn(WarningKind.DEAD_ASSIGNMENT,
                           f"variable '{name}' is assigned but never read",
                           lines[0] if lines else 0)


# ============================================================
# Convenience API
# ============================================================

def analyze(source: str, max_iterations=50) -> dict:
    """Analyze source code and return analysis results."""
    interpreter = AbstractInterpreter(max_iterations=max_iterations)
    return interpreter.analyze(source)


def get_variable_range(source: str, var_name: str) -> Interval:
    """Get the abstract interval for a variable after analysis."""
    result = analyze(source)
    return result['env'].get_interval(var_name)


def get_variable_sign(source: str, var_name: str) -> Sign:
    """Get the abstract sign for a variable after analysis."""
    result = analyze(source)
    return result['env'].get_sign(var_name)


def get_variable_const(source: str, var_name: str):
    """Get the constant propagation result for a variable."""
    result = analyze(source)
    return result['env'].get_const(var_name)


def get_warnings(source: str) -> list:
    """Get all warnings from analysis."""
    result = analyze(source)
    return result['warnings']


def check_division_safety(source: str) -> list:
    """Check for division-by-zero warnings."""
    result = analyze(source)
    return [w for w in result['warnings']
            if w.kind in (WarningKind.DIVISION_BY_ZERO, WarningKind.POSSIBLE_DIVISION_BY_ZERO)]


def check_dead_code(source: str) -> list:
    """Check for unreachable code and dead assignments."""
    result = analyze(source)
    return [w for w in result['warnings']
            if w.kind in (WarningKind.UNREACHABLE_CODE, WarningKind.DEAD_ASSIGNMENT,
                         WarningKind.UNREACHABLE_BRANCH)]
