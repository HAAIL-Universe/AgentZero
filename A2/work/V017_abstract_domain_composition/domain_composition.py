"""
V017: Abstract Domain Composition (Reduced Product)

Composes multiple abstract domains (sign, interval, constant propagation) in a
reduced product for more precise analysis than any single domain alone.

Key insight: C039 runs domains independently. A reduced product exchanges
information between domains after each operation, tightening all domains
using cross-domain knowledge.

Examples of precision gains:
- Constant x=5 -> interval [5,5], sign POS (not TOP)
- Interval [0,10] -> sign NON_NEG (not TOP)
- Sign NEG -> interval [-inf,-1] (not [-inf,inf])
- Interval [3,3] -> constant 3 (not TOP)

Composes: C039 (abstract interpreter) + C010 (parser)
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHALLENGES = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'challenges'))
sys.path.insert(0, os.path.join(_CHALLENGES, 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(_CHALLENGES, 'C010_stack_vm'))

from abstract_interpreter import (
    Sign, Interval, AbstractValue, AbstractEnv, AbstractInterpreter,
    ConstBot, ConstTop, ConstVal,
    INTERVAL_BOT, INTERVAL_TOP, CONST_BOT, CONST_TOP,
    sign_join, sign_add, sign_sub, sign_neg, sign_mul,
    interval_join, interval_meet, interval_widen,
    interval_add, interval_sub, interval_neg, interval_mul,
    const_join, const_from_value,
    sign_from_value, interval_from_value,
    INF, NEG_INF,
    Warning, WarningKind,
    analyze as c039_analyze,
)
from stack_vm import lex, Parser


# ============================================================
# Reduction operators: cross-domain information exchange
# ============================================================

def sign_from_interval(iv):
    """Derive sign from interval bounds."""
    if iv.is_bot():
        return Sign.BOT
    if iv.lo > 0:
        return Sign.POS
    if iv.hi < 0:
        return Sign.NEG
    if iv.lo == 0 and iv.hi == 0:
        return Sign.ZERO
    if iv.lo >= 0:
        return Sign.NON_NEG
    if iv.hi <= 0:
        return Sign.NON_POS
    return Sign.TOP


def interval_from_sign(s):
    """Derive interval from sign."""
    if s == Sign.BOT:
        return INTERVAL_BOT
    if s == Sign.POS:
        return Interval(1, INF)
    if s == Sign.NEG:
        return Interval(NEG_INF, -1)
    if s == Sign.ZERO:
        return Interval(0, 0)
    if s == Sign.NON_NEG:
        return Interval(0, INF)
    if s == Sign.NON_POS:
        return Interval(NEG_INF, 0)
    return INTERVAL_TOP


def sign_from_const(c):
    """Derive sign from constant."""
    if isinstance(c, ConstBot):
        return Sign.BOT
    if isinstance(c, ConstTop):
        return Sign.TOP
    # ConstVal
    v = c.value
    if not isinstance(v, (int, float)):
        return Sign.TOP
    if v > 0:
        return Sign.POS
    if v < 0:
        return Sign.NEG
    return Sign.ZERO


def interval_from_const(c):
    """Derive interval from constant."""
    if isinstance(c, ConstBot):
        return INTERVAL_BOT
    if isinstance(c, ConstTop):
        return INTERVAL_TOP
    v = c.value
    if isinstance(v, (int, float)):
        return Interval(v, v)
    return INTERVAL_TOP


def const_from_interval(iv):
    """Derive constant from interval (singleton intervals only)."""
    if iv.is_bot():
        return CONST_BOT
    if iv.lo == iv.hi and iv.lo != NEG_INF and iv.lo != INF:
        return ConstVal(int(iv.lo) if iv.lo == int(iv.lo) else iv.lo)
    return CONST_TOP


def sign_meet(a, b):
    """Meet (greatest lower bound) of two signs."""
    if a == Sign.BOT or b == Sign.BOT:
        return Sign.BOT
    if a == Sign.TOP:
        return b
    if b == Sign.TOP:
        return a
    if a == b:
        return a

    # Compute via set containment
    sign_sets = {
        Sign.POS: {Sign.POS},
        Sign.NEG: {Sign.NEG},
        Sign.ZERO: {Sign.ZERO},
        Sign.NON_NEG: {Sign.ZERO, Sign.POS},
        Sign.NON_POS: {Sign.ZERO, Sign.NEG},
    }
    sa = sign_sets.get(a, set())
    sb = sign_sets.get(b, set())
    inter = sa & sb
    if not inter:
        return Sign.BOT
    if inter == {Sign.POS}:
        return Sign.POS
    if inter == {Sign.NEG}:
        return Sign.NEG
    if inter == {Sign.ZERO}:
        return Sign.ZERO
    if inter == {Sign.ZERO, Sign.POS}:
        return Sign.NON_NEG
    if inter == {Sign.ZERO, Sign.NEG}:
        return Sign.NON_POS
    return Sign.TOP


def reduce_value(av):
    """Apply reduction to an AbstractValue: tighten all domains using cross-domain info.

    This is the core of the reduced product. After each operation, we propagate
    information between domains to maximize precision.
    """
    sign = av.sign
    interval = av.interval
    const = av.const

    # Phase 1: Constant -> everything (most precise source)
    if isinstance(const, ConstVal) and isinstance(const.value, (int, float)):
        s_from_c = sign_from_const(const)
        sign = sign_meet(sign, s_from_c)
        i_from_c = interval_from_const(const)
        interval = interval_meet(interval, i_from_c)

    # Phase 2: Interval -> sign, constant
    if not interval.is_bot():
        s_from_i = sign_from_interval(interval)
        sign = sign_meet(sign, s_from_i)
        c_from_i = const_from_interval(interval)
        if isinstance(c_from_i, ConstVal):
            if isinstance(const, ConstTop):
                const = c_from_i

    # Phase 3: Sign -> interval
    if sign != Sign.TOP and sign != Sign.BOT:
        i_from_s = interval_from_sign(sign)
        interval = interval_meet(interval, i_from_s)

    # Phase 4: BOT propagation (any domain BOT -> all BOT)
    if sign == Sign.BOT or interval.is_bot() or isinstance(const, ConstBot):
        return AbstractValue(Sign.BOT, INTERVAL_BOT, CONST_BOT)

    return AbstractValue(sign, interval, const)


def reduce_env(env):
    """Apply reduction to every variable in an AbstractEnv."""
    reduced = env.copy()
    all_vars = set(env.signs.keys()) | set(env.intervals.keys()) | set(env.consts.keys())
    for name in all_vars:
        av = AbstractValue(
            env.get_sign(name),
            env.get_interval(name),
            env.get_const(name),
        )
        rv = reduce_value(av)
        reduced.set(name, sign=rv.sign, interval=rv.interval, const=rv.const)
    return reduced


# ============================================================
# Composed Abstract Interpreter (with reduction)
# ============================================================

class ComposedInterpreter(AbstractInterpreter):
    """Abstract interpreter with reduced product domain composition.

    Overrides key methods to apply reduction after each transfer function,
    propagating information between sign, interval, and constant domains.
    """

    def __init__(self, max_iterations=50, reduce_frequency='every_stmt'):
        super().__init__(max_iterations=max_iterations)
        self.reduce_frequency = reduce_frequency  # 'every_stmt', 'every_expr', 'end_only'
        self.reduction_count = 0

    def _reduce_env(self, env):
        """Apply reduction to environment and count."""
        self.reduction_count += 1
        return reduce_env(env)

    def _reduce_value(self, av):
        """Apply reduction to abstract value."""
        self.reduction_count += 1
        return reduce_value(av)

    def _interpret_stmts(self, stmts, env):
        """Override: reduce after each statement."""
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env)
            if self.reduce_frequency in ('every_stmt', 'every_expr'):
                env = self._reduce_env(env)
        return env

    def _eval_expr(self, expr, env):
        """Override: reduce after expression evaluation."""
        result = super()._eval_expr(expr, env)
        if self.reduce_frequency == 'every_expr':
            result = self._reduce_value(result)
        return result

    def _interpret_let(self, stmt, env):
        """Override: reduce after let binding."""
        env = super()._interpret_let(stmt, env)
        return env  # reduction handled by _interpret_stmts

    def _interpret_assign(self, stmt, env):
        """Override: reduce after assignment."""
        env = super()._interpret_assign(stmt, env)
        return env  # reduction handled by _interpret_stmts

    def _interpret_if(self, stmt, env):
        """Override: reduce branch environments before join."""
        from stack_vm import IfStmt

        # Check for constant condition
        cond_val = self._eval_expr(stmt.cond, env)
        if isinstance(cond_val.const, ConstVal):
            if cond_val.const.value:
                then_env = self._interpret_stmts(stmt.then_body if isinstance(stmt.then_body, list) else stmt.then_body.stmts if hasattr(stmt.then_body, 'stmts') else [stmt.then_body], env.copy())
                then_env = self._reduce_env(then_env)
                self.warnings.append(Warning(WarningKind.UNREACHABLE_BRANCH,
                    f"Condition always true", node=stmt))
                return then_env
            else:
                if stmt.else_body:
                    else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else stmt.else_body.stmts if hasattr(stmt.else_body, 'stmts') else [stmt.else_body]
                    else_env = self._interpret_stmts(else_stmts, env.copy())
                    else_env = self._reduce_env(else_env)
                    self.warnings.append(Warning(WarningKind.UNREACHABLE_BRANCH,
                        f"Condition always false", node=stmt))
                    return else_env
                self.warnings.append(Warning(WarningKind.UNREACHABLE_BRANCH,
                    f"Condition always false, no else branch", node=stmt))
                return env

        # Refine environments for each branch
        then_env = self._refine_env_for_condition(stmt.cond, env.copy(), True)
        then_env = self._reduce_env(then_env)

        then_stmts = stmt.then_body if isinstance(stmt.then_body, list) else stmt.then_body.stmts if hasattr(stmt.then_body, 'stmts') else [stmt.then_body]
        then_env = self._interpret_stmts(then_stmts, then_env)
        then_env = self._reduce_env(then_env)

        if stmt.else_body:
            else_env = self._refine_env_for_condition(stmt.cond, env.copy(), False)
            else_env = self._reduce_env(else_env)
            else_stmts = stmt.else_body if isinstance(stmt.else_body, list) else stmt.else_body.stmts if hasattr(stmt.else_body, 'stmts') else [stmt.else_body]
            else_env = self._interpret_stmts(else_stmts, else_env)
            else_env = self._reduce_env(else_env)
        else:
            else_env = self._refine_env_for_condition(stmt.cond, env.copy(), False)
            else_env = self._reduce_env(else_env)

        # Join
        joined = then_env.join(else_env)
        return self._reduce_env(joined)

    def _interpret_while(self, stmt, env):
        """Override: reduce inside fixpoint loop."""
        # Check for immediately false condition
        cond_val = self._eval_expr(stmt.cond, env)
        if isinstance(cond_val.const, ConstVal) and not cond_val.const.value:
            return env

        current = env.copy()
        for _ in range(self.max_iterations):
            # Refine env where condition is true
            loop_env = self._refine_env_for_condition(stmt.cond, current.copy(), True)
            loop_env = self._reduce_env(loop_env)

            # Execute body
            body_stmts = stmt.body if isinstance(stmt.body, list) else stmt.body.stmts if hasattr(stmt.body, 'stmts') else [stmt.body]
            body_env = self._interpret_stmts(body_stmts, loop_env)
            body_env = self._reduce_env(body_env)

            # Widen
            next_env = current.widen(body_env)
            next_env = self._reduce_env(next_env)

            if next_env.equals(current):
                break
            current = next_env

        # Post-loop: refine where condition is false
        exit_env = self._refine_env_for_condition(stmt.cond, current.copy(), False)
        return self._reduce_env(exit_env)


# ============================================================
# Custom Domain Extension: Parity Domain
# ============================================================

class Parity(Enum):
    """Parity abstract domain: tracks even/odd."""
    BOT = "bot"
    EVEN = "even"
    ODD = "odd"
    TOP = "top"


def parity_from_value(v):
    """Create parity from concrete value."""
    if not isinstance(v, int):
        return Parity.TOP
    return Parity.EVEN if v % 2 == 0 else Parity.ODD


def parity_join(a, b):
    if a == Parity.BOT:
        return b
    if b == Parity.BOT:
        return a
    if a == b:
        return a
    return Parity.TOP


def parity_meet(a, b):
    if a == Parity.TOP:
        return b
    if b == Parity.TOP:
        return a
    if a == b:
        return a
    return Parity.BOT


def parity_add(a, b):
    if a == Parity.BOT or b == Parity.BOT:
        return Parity.BOT
    if a == Parity.TOP or b == Parity.TOP:
        return Parity.TOP
    if a == b:
        return Parity.EVEN  # even+even=even, odd+odd=even
    return Parity.ODD  # even+odd=odd, odd+even=odd


def parity_sub(a, b):
    return parity_add(a, b)  # same parity rules for subtraction


def parity_mul(a, b):
    if a == Parity.BOT or b == Parity.BOT:
        return Parity.BOT
    if a == Parity.EVEN or b == Parity.EVEN:
        if a != Parity.TOP and b != Parity.TOP:
            return Parity.EVEN  # even * anything = even
        if a == Parity.EVEN or b == Parity.EVEN:
            return Parity.EVEN
    if a == Parity.TOP or b == Parity.TOP:
        return Parity.TOP
    # odd * odd = odd
    return Parity.ODD


def parity_neg(a):
    return a  # -even = even, -odd = odd


# ============================================================
# Extended Reduced Product (with parity)
# ============================================================

@dataclass(frozen=True)
class ExtendedValue:
    """Abstract value with four domains: sign, interval, constant, parity."""
    sign: Sign
    interval: Interval
    const: object  # ConstVal, ConstTop, ConstBot
    parity: Parity

    @staticmethod
    def from_value(v):
        return ExtendedValue(
            sign_from_value(v),
            interval_from_value(v),
            const_from_value(v),
            parity_from_value(v) if isinstance(v, int) else Parity.TOP,
        )

    @staticmethod
    def top():
        return ExtendedValue(Sign.TOP, INTERVAL_TOP, CONST_TOP, Parity.TOP)

    @staticmethod
    def bot():
        return ExtendedValue(Sign.BOT, INTERVAL_BOT, CONST_BOT, Parity.BOT)

    def is_bot(self):
        return self.sign == Sign.BOT or self.interval.is_bot() or isinstance(self.const, ConstBot) or self.parity == Parity.BOT


def parity_from_interval(iv):
    """If interval is singleton, derive parity."""
    if iv.is_bot():
        return Parity.BOT
    if iv.lo == iv.hi and isinstance(iv.lo, (int, float)) and iv.lo == int(iv.lo):
        return parity_from_value(int(iv.lo))
    return Parity.TOP


def interval_refine_by_parity(iv, par):
    """Refine interval bounds using parity.

    If interval is [lo, hi] and parity is EVEN, we can tighten:
    - if lo is odd, lo becomes lo+1
    - if hi is odd, hi becomes hi-1
    And vice versa for ODD.
    """
    if iv.is_bot() or par in (Parity.BOT, Parity.TOP):
        return iv
    lo, hi = iv.lo, iv.hi
    if lo == NEG_INF or hi == INF:
        return iv  # can't refine infinite bounds
    # Only refine finite integer bounds
    if isinstance(lo, (int, float)) and lo != NEG_INF:
        lo = int(lo)
        if par == Parity.EVEN and lo % 2 != 0:
            lo += 1
        elif par == Parity.ODD and lo % 2 == 0:
            lo += 1
    if isinstance(hi, (int, float)) and hi != INF:
        hi = int(hi)
        if par == Parity.EVEN and hi % 2 != 0:
            hi -= 1
        elif par == Parity.ODD and hi % 2 == 0:
            hi -= 1
    if lo > hi:
        return INTERVAL_BOT
    return Interval(lo, hi)


def reduce_extended(ev):
    """Full reduction for the 4-domain extended product."""
    sign, interval, const, parity = ev.sign, ev.interval, ev.const, ev.parity

    # Constant -> everything
    if isinstance(const, ConstVal) and isinstance(const.value, (int, float)):
        sign = sign_meet(sign, sign_from_const(const))
        interval = interval_meet(interval, interval_from_const(const))
        p = parity_from_value(const.value) if isinstance(const.value, int) else Parity.TOP
        parity = parity_meet(parity, p)

    # Interval -> sign, constant, parity
    if not interval.is_bot():
        sign = sign_meet(sign, sign_from_interval(interval))
        c_from_i = const_from_interval(interval)
        if isinstance(c_from_i, ConstVal) and isinstance(const, ConstTop):
            const = c_from_i
        parity = parity_meet(parity, parity_from_interval(interval))

    # Sign -> interval
    if sign != Sign.TOP and sign != Sign.BOT:
        interval = interval_meet(interval, interval_from_sign(sign))

    # Parity -> interval refinement
    interval = interval_refine_by_parity(interval, parity)

    # Check for new constant after parity refinement
    if not interval.is_bot():
        c_from_i = const_from_interval(interval)
        if isinstance(c_from_i, ConstVal) and isinstance(const, ConstTop):
            const = c_from_i

    # BOT propagation
    if sign == Sign.BOT or interval.is_bot() or isinstance(const, ConstBot) or parity == Parity.BOT:
        return ExtendedValue(Sign.BOT, INTERVAL_BOT, CONST_BOT, Parity.BOT)

    return ExtendedValue(sign, interval, const, parity)


# ============================================================
# Extended Environment (4-domain)
# ============================================================

class ExtendedEnv:
    """Environment tracking four abstract domains per variable."""

    def __init__(self):
        self.signs = {}
        self.intervals = {}
        self.consts = {}
        self.parities = {}

    def set(self, name, sign=None, interval=None, const=None, parity=None):
        if sign is not None:
            self.signs[name] = sign
        if interval is not None:
            self.intervals[name] = interval
        if const is not None:
            self.consts[name] = const
        if parity is not None:
            self.parities[name] = parity

    def set_from_value(self, name, value):
        self.signs[name] = sign_from_value(value)
        self.intervals[name] = interval_from_value(value)
        self.consts[name] = const_from_value(value)
        self.parities[name] = parity_from_value(value) if isinstance(value, int) else Parity.TOP

    def set_top(self, name):
        self.signs[name] = Sign.TOP
        self.intervals[name] = INTERVAL_TOP
        self.consts[name] = CONST_TOP
        self.parities[name] = Parity.TOP

    def get_sign(self, name):
        return self.signs.get(name, Sign.TOP)

    def get_interval(self, name):
        return self.intervals.get(name, INTERVAL_TOP)

    def get_const(self, name):
        return self.consts.get(name, CONST_TOP)

    def get_parity(self, name):
        return self.parities.get(name, Parity.TOP)

    def get_extended(self, name):
        return ExtendedValue(
            self.get_sign(name),
            self.get_interval(name),
            self.get_const(name),
            self.get_parity(name),
        )

    def set_extended(self, name, ev):
        self.signs[name] = ev.sign
        self.intervals[name] = ev.interval
        self.consts[name] = ev.const
        self.parities[name] = ev.parity

    def all_vars(self):
        return set(self.signs.keys()) | set(self.intervals.keys()) | set(self.consts.keys()) | set(self.parities.keys())

    def join(self, other):
        result = ExtendedEnv()
        for name in self.all_vars() | other.all_vars():
            result.signs[name] = sign_join(self.get_sign(name), other.get_sign(name))
            result.intervals[name] = interval_join(self.get_interval(name), other.get_interval(name))
            result.consts[name] = const_join(self.get_const(name), other.get_const(name))
            result.parities[name] = parity_join(self.get_parity(name), other.get_parity(name))
        return result

    def widen(self, other):
        result = ExtendedEnv()
        for name in self.all_vars() | other.all_vars():
            result.signs[name] = sign_join(self.get_sign(name), other.get_sign(name))
            result.intervals[name] = interval_widen(self.get_interval(name), other.get_interval(name))
            result.consts[name] = const_join(self.get_const(name), other.get_const(name))
            result.parities[name] = parity_join(self.get_parity(name), other.get_parity(name))
        return result

    def equals(self, other):
        for name in self.all_vars() | other.all_vars():
            if self.get_sign(name) != other.get_sign(name):
                return False
            if self.get_interval(name) != other.get_interval(name):
                return False
            sc, oc = self.get_const(name), other.get_const(name)
            if type(sc) != type(oc):
                return False
            if isinstance(sc, ConstVal) and sc.value != oc.value:
                return False
            if self.get_parity(name) != other.get_parity(name):
                return False
        return True

    def copy(self):
        result = ExtendedEnv()
        result.signs = dict(self.signs)
        result.intervals = dict(self.intervals)
        result.consts = dict(self.consts)
        result.parities = dict(self.parities)
        return result

    def reduce(self):
        """Apply full reduction to all variables."""
        for name in self.all_vars():
            ev = self.get_extended(name)
            rv = reduce_extended(ev)
            self.set_extended(name, rv)
        return self


# ============================================================
# Full Composed Analysis (4-domain interpreter)
# ============================================================

class ExtendedInterpreter:
    """Abstract interpreter with 4-domain reduced product.

    Runs a complete abstract interpretation with sign, interval, constant,
    and parity domains, applying reduction after each transfer function.
    """

    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations
        self.warnings = []
        self.functions = {}
        self.reduction_count = 0

    def _reduce(self, env):
        self.reduction_count += 1
        return env.reduce()

    def analyze(self, source):
        """Run extended composed analysis."""
        tokens = lex(source)
        program = Parser(tokens).parse()

        # Collect functions
        from stack_vm import FnDecl
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt

        env = ExtendedEnv()
        env = self._interpret_stmts(program.stmts, env)

        return {
            'env': env,
            'warnings': self.warnings,
            'reductions': self.reduction_count,
        }

    def _interpret_stmts(self, stmts, env):
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env)
            env = self._reduce(env)
        return env

    def _interpret_stmt(self, stmt, env):
        from stack_vm import LetDecl, Assign, IfStmt, WhileStmt, PrintStmt, ReturnStmt, FnDecl, Block

        if isinstance(stmt, LetDecl):
            val = self._eval_expr(stmt.value, env)
            env.set_extended(stmt.name, val)
            return env

        if isinstance(stmt, Assign):
            val = self._eval_expr(stmt.value, env)
            env.set_extended(stmt.name, val)
            return env

        if isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env)

        if isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, env)

        if isinstance(stmt, PrintStmt):
            self._eval_expr(stmt.value, env)
            return env

        if isinstance(stmt, ReturnStmt):
            if stmt.value:
                self._eval_expr(stmt.value, env)
            return env

        if isinstance(stmt, FnDecl):
            return env  # already collected

        if isinstance(stmt, Block):
            return self._interpret_stmts(stmt.stmts, env)

        return env

    def _interpret_if(self, stmt, env):
        cond_val = self._eval_expr(stmt.cond, env)

        # Constant condition
        if isinstance(cond_val.const, ConstVal):
            if cond_val.const.value:
                self.warnings.append(Warning(WarningKind.UNREACHABLE_BRANCH, "Condition always true"))
                then_stmts = self._get_stmts(stmt.then_body)
                return self._interpret_stmts(then_stmts, env.copy())
            else:
                self.warnings.append(Warning(WarningKind.UNREACHABLE_BRANCH, "Condition always false"))
                if stmt.else_body:
                    return self._interpret_stmts(self._get_stmts(stmt.else_body), env.copy())
                return env

        # Both branches
        then_env = self._refine_for_condition(stmt.cond, env.copy(), True)
        then_env = self._reduce(then_env)
        then_env = self._interpret_stmts(self._get_stmts(stmt.then_body), then_env)
        then_env = self._reduce(then_env)

        if stmt.else_body:
            else_env = self._refine_for_condition(stmt.cond, env.copy(), False)
            else_env = self._reduce(else_env)
            else_env = self._interpret_stmts(self._get_stmts(stmt.else_body), else_env)
            else_env = self._reduce(else_env)
        else:
            else_env = self._refine_for_condition(stmt.cond, env.copy(), False)
            else_env = self._reduce(else_env)

        joined = then_env.join(else_env)
        return self._reduce(joined)

    def _interpret_while(self, stmt, env):
        current = env.copy()
        for _ in range(self.max_iterations):
            loop_env = self._refine_for_condition(stmt.cond, current.copy(), True)
            loop_env = self._reduce(loop_env)

            body_env = self._interpret_stmts(self._get_stmts(stmt.body), loop_env)
            body_env = self._reduce(body_env)

            next_env = current.widen(body_env)
            next_env = self._reduce(next_env)

            if next_env.equals(current):
                break
            current = next_env

        exit_env = self._refine_for_condition(stmt.cond, current.copy(), False)
        return self._reduce(exit_env)

    def _get_stmts(self, body):
        if isinstance(body, list):
            return body
        if hasattr(body, 'stmts'):
            return body.stmts
        return [body]

    def _eval_expr(self, expr, env):
        from stack_vm import IntLit, BoolLit, StringLit, Var, BinOp, UnaryOp, CallExpr, Assign as ASTAssign

        if isinstance(expr, IntLit):
            return reduce_extended(ExtendedValue.from_value(expr.value))

        if isinstance(expr, BoolLit):
            v = 1 if expr.value else 0
            return ExtendedValue(
                sign_from_value(v),
                interval_from_value(v),
                const_from_value(expr.value),
                parity_from_value(v),
            )

        if isinstance(expr, StringLit):
            return ExtendedValue(Sign.TOP, INTERVAL_TOP, const_from_value(expr.value), Parity.TOP)

        if isinstance(expr, Var):
            return env.get_extended(expr.name)

        if isinstance(expr, UnaryOp):
            operand = self._eval_expr(expr.operand, env)
            if expr.op == '-':
                return reduce_extended(ExtendedValue(
                    sign_neg(operand.sign),
                    interval_neg(operand.interval),
                    ConstVal(-operand.const.value) if isinstance(operand.const, ConstVal) and isinstance(operand.const.value, (int, float)) else CONST_TOP,
                    parity_neg(operand.parity),
                ))
            if expr.op == 'not':
                return ExtendedValue.top()
            return ExtendedValue.top()

        if isinstance(expr, BinOp):
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            return self._eval_binary(expr.op, left, right)

        if isinstance(expr, CallExpr):
            for arg in expr.args:
                self._eval_expr(arg, env)
            return ExtendedValue.top()

        if isinstance(expr, ASTAssign):
            val = self._eval_expr(expr.value, env)
            env.set_extended(expr.name, val)
            return val

        return ExtendedValue.top()

    def _eval_binary(self, op, left, right):
        """Evaluate binary operation across all four domains."""
        if op == '+':
            return reduce_extended(ExtendedValue(
                sign_add(left.sign, right.sign),
                interval_add(left.interval, right.interval),
                self._const_arith(left.const, right.const, lambda a, b: a + b),
                parity_add(left.parity, right.parity),
            ))
        if op == '-':
            return reduce_extended(ExtendedValue(
                sign_sub(left.sign, right.sign),
                interval_sub(left.interval, right.interval),
                self._const_arith(left.const, right.const, lambda a, b: a - b),
                parity_sub(left.parity, right.parity),
            ))
        if op == '*':
            return reduce_extended(ExtendedValue(
                sign_mul(left.sign, right.sign),
                interval_mul(left.interval, right.interval),
                self._const_arith(left.const, right.const, lambda a, b: a * b),
                parity_mul(left.parity, right.parity),
            ))
        if op in ('/', '%'):
            # Check division by zero
            if right.sign == Sign.ZERO or (isinstance(right.const, ConstVal) and right.const.value == 0):
                self.warnings.append(Warning(WarningKind.DIVISION_BY_ZERO, "Division by zero"))
            elif right.interval.contains(0) if hasattr(right.interval, 'contains') else False:
                self.warnings.append(Warning(WarningKind.POSSIBLE_DIVISION_BY_ZERO, "Possible division by zero"))
            return ExtendedValue.top()  # conservative for div/mod

        # Comparison operators -> result in {0, 1}
        if op in ('<', '>', '<=', '>=', '==', '!='):
            result = self._eval_comparison(op, left, right)
            return result

        # Boolean operators
        if op in ('and', 'or'):
            return ExtendedValue(Sign.NON_NEG, Interval(0, 1), CONST_TOP, Parity.TOP)

        return ExtendedValue.top()

    def _eval_comparison(self, op, left, right):
        """Evaluate comparison with cross-domain precision."""
        # Try constant comparison first
        if isinstance(left.const, ConstVal) and isinstance(right.const, ConstVal):
            lv, rv = left.const.value, right.const.value
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                result = {
                    '<': lv < rv, '>': lv > rv, '<=': lv <= rv,
                    '>=': lv >= rv, '==': lv == rv, '!=': lv != rv,
                }[op]
                v = 1 if result else 0
                return ExtendedValue.from_value(v)

        # Try interval comparison
        li, ri = left.interval, right.interval
        if not li.is_bot() and not ri.is_bot() and not li.is_top() and not ri.is_top():
            if op == '<' and li.hi < ri.lo:
                return ExtendedValue.from_value(1)
            if op == '<' and li.lo >= ri.hi:
                return ExtendedValue.from_value(0)
            if op == '>' and li.lo > ri.hi:
                return ExtendedValue.from_value(1)
            if op == '>' and li.hi <= ri.lo:
                return ExtendedValue.from_value(0)
            if op == '<=' and li.hi <= ri.lo:
                return ExtendedValue.from_value(1)
            if op == '<=' and li.lo > ri.hi:
                return ExtendedValue.from_value(0)
            if op == '>=' and li.lo >= ri.hi:
                return ExtendedValue.from_value(1)
            if op == '>=' and li.hi < ri.lo:
                return ExtendedValue.from_value(0)
            if op == '==' and li.lo == li.hi == ri.lo == ri.hi:
                return ExtendedValue.from_value(1)
            if op == '==' and (li.hi < ri.lo or li.lo > ri.hi):
                return ExtendedValue.from_value(0)
            if op == '!=' and (li.hi < ri.lo or li.lo > ri.hi):
                return ExtendedValue.from_value(1)
            if op == '!=' and li.lo == li.hi == ri.lo == ri.hi:
                return ExtendedValue.from_value(0)

        # Parity can resolve some equality checks
        if op == '!=' and left.parity != Parity.TOP and right.parity != Parity.TOP:
            if left.parity != right.parity and left.parity != Parity.BOT and right.parity != Parity.BOT:
                # Different parity -> definitely not equal
                return ExtendedValue.from_value(1)
        if op == '==' and left.parity != Parity.TOP and right.parity != Parity.TOP:
            if left.parity != right.parity and left.parity != Parity.BOT and right.parity != Parity.BOT:
                return ExtendedValue.from_value(0)

        # Unknown comparison result in [0, 1]
        return ExtendedValue(Sign.NON_NEG, Interval(0, 1), CONST_TOP, Parity.TOP)

    def _const_arith(self, a, b, op):
        """Constant propagation for arithmetic."""
        if isinstance(a, ConstVal) and isinstance(b, ConstVal):
            av, bv = a.value, b.value
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                try:
                    return ConstVal(op(av, bv))
                except (ZeroDivisionError, OverflowError):
                    return CONST_TOP
        return CONST_TOP

    def _refine_for_condition(self, cond, env, branch):
        """Refine extended environment based on branch condition."""
        from stack_vm import BinOp, Var, IntLit, UnaryOp

        if isinstance(cond, BinOp):
            if isinstance(cond.left, Var) and isinstance(cond.right, IntLit):
                name = cond.left.name
                val = cond.right.value
                iv = env.get_interval(name)
                op = cond.op

                if branch:
                    if op == '<':
                        new_iv = interval_meet(iv, Interval(NEG_INF, val - 1))
                    elif op == '<=':
                        new_iv = interval_meet(iv, Interval(NEG_INF, val))
                    elif op == '>':
                        new_iv = interval_meet(iv, Interval(val + 1, INF))
                    elif op == '>=':
                        new_iv = interval_meet(iv, Interval(val, INF))
                    elif op == '==':
                        new_iv = interval_meet(iv, Interval(val, val))
                    elif op == '!=':
                        new_iv = iv  # can't represent != as interval
                    else:
                        new_iv = iv
                else:
                    if op == '<':
                        new_iv = interval_meet(iv, Interval(val, INF))
                    elif op == '<=':
                        new_iv = interval_meet(iv, Interval(val + 1, INF))
                    elif op == '>':
                        new_iv = interval_meet(iv, Interval(NEG_INF, val))
                    elif op == '>=':
                        new_iv = interval_meet(iv, Interval(NEG_INF, val - 1))
                    elif op == '==':
                        new_iv = iv  # can't represent != as interval
                    elif op == '!=':
                        new_iv = interval_meet(iv, Interval(val, val))
                    else:
                        new_iv = iv

                env.intervals[name] = new_iv
                # Reduce: propagate interval info to other domains
                ev = env.get_extended(name)
                rv = reduce_extended(ev)
                env.set_extended(name, rv)

            # Var cmp Var
            elif isinstance(cond.left, Var) and isinstance(cond.right, Var):
                # Less precise but still useful
                lname, rname = cond.left.name, cond.right.name
                liv, riv = env.get_interval(lname), env.get_interval(rname)
                op = cond.op

                if branch:
                    if op == '<':
                        # left < right => left.hi < right.lo at most
                        if not riv.is_top() and riv.hi != INF:
                            new_l = interval_meet(liv, Interval(NEG_INF, riv.hi - 1))
                            env.intervals[lname] = new_l
                        if not liv.is_top() and liv.lo != NEG_INF:
                            new_r = interval_meet(riv, Interval(liv.lo + 1, INF))
                            env.intervals[rname] = new_r
                    elif op == '>':
                        if not riv.is_top() and riv.lo != NEG_INF:
                            new_l = interval_meet(liv, Interval(riv.lo + 1, INF))
                            env.intervals[lname] = new_l
                        if not liv.is_top() and liv.hi != INF:
                            new_r = interval_meet(riv, Interval(NEG_INF, liv.hi - 1))
                            env.intervals[rname] = new_r

        return env


# ============================================================
# Comparison: C039 baseline vs composed analysis
# ============================================================

@dataclass
class ComparisonResult:
    """Result of comparing baseline vs composed analysis."""
    source: str
    baseline_env: object  # AbstractEnv from C039
    composed_env: object  # ExtendedEnv from V017
    precision_gains: list  # list of (var, domain, baseline_val, composed_val)
    baseline_warnings: list
    composed_warnings: list
    extra_warnings: list  # warnings only in composed
    reductions: int

    @property
    def has_gains(self):
        return len(self.precision_gains) > 0


def compare_analyses(source):
    """Run both C039 baseline and V017 composed analysis, report precision gains."""
    # Baseline: C039
    baseline = c039_analyze(source)
    baseline_env = baseline['env']
    baseline_warnings = baseline['warnings']

    # Composed: V017
    interp = ExtendedInterpreter()
    composed = interp.analyze(source)
    composed_env = composed['env']
    composed_warnings = composed['warnings']

    # Find precision gains
    gains = []
    all_vars = set(baseline_env.signs.keys()) | set(baseline_env.intervals.keys()) | set(baseline_env.consts.keys())
    all_vars |= composed_env.all_vars()

    for var in sorted(all_vars):
        # Sign precision
        bs = baseline_env.get_sign(var)
        cs = composed_env.get_sign(var)
        if cs != bs and cs != Sign.TOP and (bs == Sign.TOP or _sign_more_precise(cs, bs)):
            gains.append((var, 'sign', str(bs), str(cs)))

        # Interval precision
        bi = baseline_env.get_interval(var)
        ci = composed_env.get_interval(var)
        if ci != bi and not ci.is_top() and (bi.is_top() or _interval_more_precise(ci, bi)):
            gains.append((var, 'interval', str(bi), str(ci)))

        # Constant precision
        bc = baseline_env.get_const(var)
        cc = composed_env.get_const(var)
        if isinstance(cc, ConstVal) and isinstance(bc, ConstTop):
            gains.append((var, 'constant', 'TOP', str(cc.value)))

        # Parity (new domain, always a gain if non-TOP)
        cp = composed_env.get_parity(var)
        if cp not in (Parity.TOP, Parity.BOT):
            gains.append((var, 'parity', 'N/A', str(cp)))

    # Extra warnings
    baseline_strs = {str(w) for w in baseline_warnings}
    extra = [w for w in composed_warnings if str(w) not in baseline_strs]

    return ComparisonResult(
        source=source,
        baseline_env=baseline_env,
        composed_env=composed_env,
        precision_gains=gains,
        baseline_warnings=baseline_warnings,
        composed_warnings=composed_warnings,
        extra_warnings=extra,
        reductions=interp.reduction_count,
    )


def _sign_more_precise(a, b):
    """Is sign a strictly more precise than b?"""
    precise_order = {
        Sign.POS: {Sign.NON_NEG, Sign.TOP},
        Sign.NEG: {Sign.NON_POS, Sign.TOP},
        Sign.ZERO: {Sign.NON_NEG, Sign.NON_POS, Sign.TOP},
        Sign.NON_NEG: {Sign.TOP},
        Sign.NON_POS: {Sign.TOP},
    }
    return b in precise_order.get(a, set())


def _interval_more_precise(a, b):
    """Is interval a strictly more precise (subset of) b?"""
    if b.is_top():
        return True
    if a.is_bot():
        return True
    return a.lo >= b.lo and a.hi <= b.hi and (a.lo > b.lo or a.hi < b.hi)


# ============================================================
# High-level API
# ============================================================

def composed_analyze(source, max_iterations=50):
    """Run composed 4-domain abstract interpretation.

    Returns dict with 'env' (ExtendedEnv), 'warnings', 'reductions'.
    """
    interp = ExtendedInterpreter(max_iterations=max_iterations)
    return interp.analyze(source)


def get_variable_info(source, var_name):
    """Get full abstract information about a variable.

    Returns ExtendedValue with sign, interval, constant, and parity.
    """
    result = composed_analyze(source)
    return result['env'].get_extended(var_name)


def get_precision_gains(source):
    """Compare C039 baseline vs V017 composed, return list of precision gains."""
    cr = compare_analyses(source)
    return cr.precision_gains


def composed_reduce(sign=None, interval=None, const=None, parity=None):
    """Apply reduction to explicitly provided domain values.

    Useful for testing reduction in isolation.
    """
    ev = ExtendedValue(
        sign or Sign.TOP,
        interval or INTERVAL_TOP,
        const or CONST_TOP,
        parity or Parity.TOP,
    )
    return reduce_extended(ev)


def analyze_with_comparison(source):
    """Run both analyses and return full comparison."""
    return compare_analyses(source)
