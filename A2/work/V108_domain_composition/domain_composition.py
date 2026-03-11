"""
V108: Abstract Domain Composition Framework
============================================

A framework for composing abstract domains with configurable reduction,
built-in cross-domain reducers, and advanced combinators.

Composes: V020 (AbstractDomain protocol), C010 (parser)

Components:
1. Built-in reducers for common domain pairs (sign<->interval, const<->interval, etc.)
2. ReducedProductBuilder -- declarative domain composition with auto-reduction
3. DisjunctiveDomain -- bounded disjunctive completion (powerset with cardinality limit)
4. LiftedDomain -- adds error/exception abstract state
5. CardinalPowerDomain -- map from a finite abstract set to another domain
6. CompositionInterpreter -- generic C10 interpreter for composed domains
7. PrecisionComparator -- analyze which composition catches more
"""

import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional, Sequence,
    Set, Tuple, Type, TypeVar, Union,
)
from enum import Enum
from copy import deepcopy
import math

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_a2_work = os.path.dirname(_here)
_agent_zero = os.path.normpath(os.path.join(_a2_work, '..', '..'))
_challenges = os.path.join(_agent_zero, 'challenges')

_c010 = os.path.normpath(os.path.join(_challenges, 'C010_stack_vm'))
if _c010 not in sys.path:
    sys.path.insert(0, _c010)

_v020 = os.path.normpath(os.path.join(_a2_work, 'V020_abstract_domain_functor'))
if _v020 not in sys.path:
    sys.path.insert(0, _v020)

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, Var, IntLit, BinOp, UnaryOp,
    CallExpr, BoolLit,
)

from domain_functor import (
    AbstractDomain, SignDomain, SignValue, IntervalDomain,
    ConstDomain, ParityDomain, ParityValue,
    ProductDomain, ReducedProductDomain, PowersetDomain,
    FlatDomain, DomainEnv, FunctorInterpreter,
    INF, NEG_INF,
)


# ===========================================================================
# Part 1: Built-in Cross-Domain Reducers
# ===========================================================================

def reduce_sign_interval(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Reduce sign information into interval bounds and vice versa.

    Expects components[0] = SignDomain, components[1] = IntervalDomain.
    """
    result = list(components)
    sign_idx = interval_idx = None
    for i, c in enumerate(result):
        if isinstance(c, SignDomain):
            sign_idx = i
        elif isinstance(c, IntervalDomain):
            interval_idx = i
    if sign_idx is None or interval_idx is None:
        return result

    sign = result[sign_idx]
    interval = result[interval_idx]

    if sign.is_bot() or interval.is_bot():
        return [c.bot() for c in result]

    # Sign -> Interval reduction
    sv = sign.value
    if sv == SignValue.POS:
        interval = interval.meet(IntervalDomain(1, INF))
    elif sv == SignValue.NEG:
        interval = interval.meet(IntervalDomain(NEG_INF, -1))
    elif sv == SignValue.ZERO:
        interval = interval.meet(IntervalDomain(0, 0))
    elif sv == SignValue.NON_NEG:
        interval = interval.meet(IntervalDomain(0, INF))
    elif sv == SignValue.NON_POS:
        interval = interval.meet(IntervalDomain(NEG_INF, 0))

    if interval.is_bot():
        return [c.bot() for c in result]

    # Interval -> Sign reduction
    if not interval.is_top():
        lo, hi = interval.lo, interval.hi
        if lo > 0:
            sign = sign.meet(SignDomain(SignValue.POS))
        elif hi < 0:
            sign = sign.meet(SignDomain(SignValue.NEG))
        elif lo == 0 and hi == 0:
            sign = sign.meet(SignDomain(SignValue.ZERO))
        elif lo >= 0:
            sign = sign.meet(SignDomain(SignValue.NON_NEG))
        elif hi <= 0:
            sign = sign.meet(SignDomain(SignValue.NON_POS))

    if sign.is_bot():
        return [c.bot() for c in result]

    result[sign_idx] = sign
    result[interval_idx] = interval
    return result


def reduce_const_interval(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Reduce constant propagation info into intervals and vice versa.

    Looks for ConstDomain and IntervalDomain in components.
    """
    result = list(components)
    const_idx = interval_idx = None
    for i, c in enumerate(result):
        if isinstance(c, ConstDomain):
            const_idx = i
        elif isinstance(c, IntervalDomain):
            interval_idx = i
    if const_idx is None or interval_idx is None:
        return result

    const = result[const_idx]
    interval = result[interval_idx]

    if const.is_bot() or interval.is_bot():
        return [c.bot() for c in result]

    # Const -> Interval: if constant is known, interval is [c, c]
    if const.is_const() and isinstance(const.value, (int, float)):
        v = const.value
        interval = interval.meet(IntervalDomain(v, v))
        if interval.is_bot():
            return [c.bot() for c in result]

    # Interval -> Const: if interval is [c, c], constant is c
    if not interval.is_bot() and not interval.is_top():
        if interval.lo == interval.hi and interval.lo != NEG_INF and interval.lo != INF:
            v = int(interval.lo) if interval.lo == int(interval.lo) else interval.lo
            const = const.meet(ConstDomain.from_concrete(v))
            if const.is_bot():
                return [c.bot() for c in result]

    result[const_idx] = const
    result[interval_idx] = interval
    return result


def reduce_const_sign(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Reduce constant propagation into sign and vice versa."""
    result = list(components)
    const_idx = sign_idx = None
    for i, c in enumerate(result):
        if isinstance(c, ConstDomain):
            const_idx = i
        elif isinstance(c, SignDomain):
            sign_idx = i
    if const_idx is None or sign_idx is None:
        return result

    const = result[const_idx]
    sign = result[sign_idx]

    if const.is_bot() or sign.is_bot():
        return [c.bot() for c in result]

    # Const -> Sign
    if const.is_const() and isinstance(const.value, (int, float)):
        v = const.value
        if v > 0:
            sign = sign.meet(SignDomain(SignValue.POS))
        elif v < 0:
            sign = sign.meet(SignDomain(SignValue.NEG))
        else:
            sign = sign.meet(SignDomain(SignValue.ZERO))
        if sign.is_bot():
            return [c.bot() for c in result]

    # Sign -> Const: if sign is ZERO, constant is 0
    if sign.value == SignValue.ZERO:
        const = const.meet(ConstDomain.from_concrete(0))
        if const.is_bot():
            return [c.bot() for c in result]

    result[const_idx] = const
    result[sign_idx] = sign
    return result


def reduce_parity_interval(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Reduce parity into interval and vice versa."""
    result = list(components)
    parity_idx = interval_idx = None
    for i, c in enumerate(result):
        if isinstance(c, ParityDomain):
            parity_idx = i
        elif isinstance(c, IntervalDomain):
            interval_idx = i
    if parity_idx is None or interval_idx is None:
        return result

    parity = result[parity_idx]
    interval = result[interval_idx]

    if parity.is_bot() or interval.is_bot():
        return [c.bot() for c in result]

    # Interval -> Parity: singleton interval determines parity
    if not interval.is_bot() and not interval.is_top():
        if interval.lo == interval.hi and interval.lo != NEG_INF:
            v = int(interval.lo)
            expected = ParityDomain(ParityValue.EVEN if v % 2 == 0 else ParityValue.ODD)
            parity = parity.meet(expected)
            if parity.is_bot():
                return [c.bot() for c in result]

    # Parity -> Interval: tighten bounds by 1 if parity mismatches endpoint
    pv = parity.value
    if pv in (ParityValue.EVEN, ParityValue.ODD) and not interval.is_top():
        lo, hi = interval.lo, interval.hi
        target_rem = 0 if pv == ParityValue.EVEN else 1
        if lo != NEG_INF:
            ilo = int(lo)
            if ilo % 2 != target_rem and (ilo + target_rem) % 2 != target_rem:
                ilo += 1
            while ilo % 2 != target_rem:
                ilo += 1
            if ilo > hi:
                return [c.bot() for c in result]
            if ilo > lo:
                interval = IntervalDomain(ilo, interval.hi)
        if hi != INF:
            ihi = int(hi)
            while ihi % 2 != target_rem:
                ihi -= 1
            if ihi < interval.lo:
                return [c.bot() for c in result]
            if ihi < hi:
                interval = IntervalDomain(interval.lo, ihi)

    result[parity_idx] = parity
    result[interval_idx] = interval
    return result


def reduce_parity_sign(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Reduce parity + sign: zero is even."""
    result = list(components)
    parity_idx = sign_idx = None
    for i, c in enumerate(result):
        if isinstance(c, ParityDomain):
            parity_idx = i
        elif isinstance(c, SignDomain):
            sign_idx = i
    if parity_idx is None or sign_idx is None:
        return result

    parity = result[parity_idx]
    sign = result[sign_idx]

    if parity.is_bot() or sign.is_bot():
        return [c.bot() for c in result]

    # Sign ZERO -> Parity EVEN
    if sign.value == SignValue.ZERO:
        parity = parity.meet(ParityDomain(ParityValue.EVEN))
        if parity.is_bot():
            return [c.bot() for c in result]

    # Parity ODD -> Sign != ZERO
    if parity.value == ParityValue.ODD:
        if sign.value == SignValue.ZERO:
            return [c.bot() for c in result]
        if sign.value == SignValue.NON_NEG:
            sign = SignDomain(SignValue.POS)
        elif sign.value == SignValue.NON_POS:
            sign = SignDomain(SignValue.NEG)

    result[parity_idx] = parity
    result[sign_idx] = sign
    return result


# Registry of built-in reducers keyed by (domain_type_1, domain_type_2)
BUILTIN_REDUCERS = {
    (SignDomain, IntervalDomain): reduce_sign_interval,
    (ConstDomain, IntervalDomain): reduce_const_interval,
    (ConstDomain, SignDomain): reduce_const_sign,
    (ParityDomain, IntervalDomain): reduce_parity_interval,
    (ParityDomain, SignDomain): reduce_parity_sign,
}


def find_builtin_reducers(domain_types: List[Type[AbstractDomain]]) -> List[Callable]:
    """Auto-discover applicable built-in reducers for a list of domain types."""
    reducers = []
    for i, t1 in enumerate(domain_types):
        for j, t2 in enumerate(domain_types):
            if i >= j:
                continue
            key = (t1, t2)
            if key in BUILTIN_REDUCERS:
                reducers.append(BUILTIN_REDUCERS[key])
            else:
                rev_key = (t2, t1)
                if rev_key in BUILTIN_REDUCERS:
                    reducers.append(BUILTIN_REDUCERS[rev_key])
    return reducers


# ===========================================================================
# Part 2: ReducedProductBuilder -- Declarative Domain Composition
# ===========================================================================

class ReducedProductBuilder:
    """Declarative builder for reduced product domains.

    Usage:
        builder = ReducedProductBuilder()
        builder.add(SignDomain)
        builder.add(IntervalDomain)
        builder.add(ParityDomain)
        builder.auto_reduce()  # auto-discover reducers
        factory = builder.build()
        # factory(5) -> ReducedProductDomain([Sign(pos), Interval(5,5), Parity(odd)])
        # factory()  -> ReducedProductDomain([Sign(top), Interval(-inf,inf), Parity(top)])
    """

    def __init__(self):
        self._domain_types: List[Type[AbstractDomain]] = []
        self._reducers: List[Callable] = []
        self._fixpoint_reduce = False
        self._max_reduce_iterations = 3

    def add(self, domain_type: Type[AbstractDomain]) -> 'ReducedProductBuilder':
        """Add a domain to the product."""
        self._domain_types.append(domain_type)
        return self

    def add_reducer(self, reducer: Callable) -> 'ReducedProductBuilder':
        """Add a custom reducer function."""
        self._reducers.append(reducer)
        return self

    def auto_reduce(self) -> 'ReducedProductBuilder':
        """Auto-discover and add built-in reducers for all domain pairs."""
        self._reducers.extend(find_builtin_reducers(self._domain_types))
        return self

    def fixpoint(self, max_iterations: int = 3) -> 'ReducedProductBuilder':
        """Enable fixpoint reduction (iterate reducers until stable)."""
        self._fixpoint_reduce = True
        self._max_reduce_iterations = max_iterations
        return self

    def build(self) -> Callable:
        """Build a domain factory function.

        Returns a callable: Optional[int] -> ReducedProductDomain
        - factory(None) or factory() returns TOP
        - factory(5) returns abstract value for 5
        """
        domain_types = list(self._domain_types)
        reducers = list(self._reducers)
        fixpoint = self._fixpoint_reduce
        max_iter = self._max_reduce_iterations

        if fixpoint and reducers:
            def fixpoint_reducer(components):
                for _ in range(max_iter):
                    old = [repr(c) for c in components]
                    for r in reducers:
                        components = r(components)
                        if any(c.is_bot() for c in components):
                            return [c.bot() for c in components]
                    if [repr(c) for c in components] == old:
                        break
                return components
            final_reducers = [fixpoint_reducer]
        else:
            final_reducers = reducers

        def factory(value=None):
            if value is None:
                components = [dt(dt().top()._v) if hasattr(dt(), '_v')
                              else dt()
                              for dt in domain_types]
                # Simpler: just use from_concrete approach
                components = []
                for dt in domain_types:
                    inst = dt.__new__(dt)
                    # Use top() from a throwaway instance
                    components.append(dt.from_concrete(0).top())
                return ReducedProductDomain(components, final_reducers)
            else:
                components = [dt.from_concrete(value) for dt in domain_types]
                return ReducedProductDomain(components, final_reducers)

        return factory

    def build_interpreter(self, **kwargs) -> 'CompositionInterpreter':
        """Build an interpreter using this domain composition."""
        factory = self.build()
        return CompositionInterpreter(factory, **kwargs)


# ===========================================================================
# Part 3: DisjunctiveDomain -- Bounded Disjunctive Completion
# ===========================================================================

class DisjunctiveDomain(AbstractDomain):
    """Bounded disjunctive completion of an abstract domain.

    Maintains a set of abstract values (disjuncts) up to a cardinality bound.
    When the bound is exceeded, disjuncts are merged using join.

    More precise than a single abstract value (tracks multiple paths),
    but bounded to prevent exponential blowup.
    """

    def __init__(self, disjuncts: Optional[List[AbstractDomain]] = None,
                 max_disjuncts: int = 4,
                 _is_bot: bool = False):
        self._disjuncts = []
        self._max = max_disjuncts
        self._bot = _is_bot

        if disjuncts and not _is_bot:
            # Filter out bots, deduplicate
            for d in disjuncts:
                if not d.is_bot():
                    self._disjuncts.append(d)
            if not self._disjuncts:
                self._bot = True
            self._compact()

    def _compact(self):
        """Merge disjuncts down to max cardinality."""
        while len(self._disjuncts) > self._max:
            # Find the closest pair (by join size heuristic: merge smallest)
            best_i, best_j = 0, 1
            self._disjuncts[best_i] = self._disjuncts[best_i].join(self._disjuncts[best_j])
            self._disjuncts.pop(best_j)

    @property
    def disjuncts(self):
        return list(self._disjuncts)

    def top(self):
        if not self._disjuncts:
            return DisjunctiveDomain(max_disjuncts=self._max, _is_bot=False)
        return DisjunctiveDomain([self._disjuncts[0].top()], self._max)

    def bot(self):
        return DisjunctiveDomain(max_disjuncts=self._max, _is_bot=True)

    def is_top(self):
        return len(self._disjuncts) == 1 and self._disjuncts[0].is_top()

    def is_bot(self):
        return self._bot or (not self._disjuncts)

    def leq(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return NotImplemented
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        # Each disjunct of self must be leq some disjunct of other
        for d in self._disjuncts:
            if not any(d.leq(o) for o in other._disjuncts):
                return False
        return True

    def join(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return NotImplemented
        if self.is_bot():
            return DisjunctiveDomain(other._disjuncts[:], self._max)
        if other.is_bot():
            return DisjunctiveDomain(self._disjuncts[:], self._max)
        combined = self._disjuncts + other._disjuncts
        return DisjunctiveDomain(combined, self._max)

    def meet(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return NotImplemented
        if self.is_bot() or other.is_bot():
            return self.bot()
        # Pairwise meets, keep non-bot results
        results = []
        for a in self._disjuncts:
            for b in other._disjuncts:
                m = a.meet(b)
                if not m.is_bot():
                    results.append(m)
        if not results:
            return self.bot()
        return DisjunctiveDomain(results, self._max)

    def widen(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return NotImplemented
        if self.is_bot():
            return DisjunctiveDomain(other._disjuncts[:], self._max)
        if other.is_bot():
            return DisjunctiveDomain(self._disjuncts[:], self._max)
        # Pairwise widen where possible, then add new disjuncts
        result = []
        other_used = [False] * len(other._disjuncts)
        for a in self._disjuncts:
            # Find best matching disjunct in other
            best = None
            best_idx = -1
            for j, b in enumerate(other._disjuncts):
                if not other_used[j] and a.leq(b.join(a)):
                    best = b
                    best_idx = j
                    break
            if best is not None:
                result.append(a.widen(best))
                other_used[best_idx] = True
            else:
                result.append(a)
        # Add unmatched disjuncts from other
        for j, b in enumerate(other._disjuncts):
            if not other_used[j]:
                result.append(b)
        return DisjunctiveDomain(result, self._max)

    @classmethod
    def from_concrete(cls, value: int):
        raise NotImplementedError("Use DisjunctiveDomain.wrap(domain.from_concrete(value))")

    @staticmethod
    def wrap(value: AbstractDomain, max_disjuncts: int = 4) -> 'DisjunctiveDomain':
        """Wrap a single abstract value as a disjunctive domain."""
        return DisjunctiveDomain([value], max_disjuncts)

    def apply_unary(self, op_name: str) -> 'DisjunctiveDomain':
        """Apply a unary operation to each disjunct."""
        if self.is_bot():
            return self.bot()
        results = [getattr(d, op_name)() for d in self._disjuncts]
        return DisjunctiveDomain(results, self._max)

    def apply_binary(self, other: 'DisjunctiveDomain', op_name: str) -> 'DisjunctiveDomain':
        """Apply a binary operation across all disjunct pairs."""
        if self.is_bot() or other.is_bot():
            return self.bot()
        results = []
        for a in self._disjuncts:
            for b in other._disjuncts:
                r = getattr(a, op_name)(b)
                if not r.is_bot():
                    results.append(r)
        if not results:
            return self.bot()
        return DisjunctiveDomain(results, self._max)

    def add(self, other):
        return self.apply_binary(other, 'add')

    def sub(self, other):
        return self.apply_binary(other, 'sub')

    def mul(self, other):
        return self.apply_binary(other, 'mul')

    def neg(self):
        return self.apply_unary('neg')

    def refine_lt(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return (self, other)
        if self.is_bot() or other.is_bot():
            return (self, other)
        # Refine each pair, keep non-bot
        self_results = []
        other_results = []
        for a in self._disjuncts:
            for b in other._disjuncts:
                ra, rb = a.refine_lt(b)
                if not ra.is_bot():
                    self_results.append(ra)
                if not rb.is_bot():
                    other_results.append(rb)
        s = DisjunctiveDomain(self_results, self._max) if self_results else self.bot()
        o = DisjunctiveDomain(other_results, self._max) if other_results else other.bot()
        return (s, o)

    def refine_eq(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return (self, other)
        if self.is_bot() or other.is_bot():
            return (self, other)
        self_results = []
        other_results = []
        for a in self._disjuncts:
            for b in other._disjuncts:
                ra, rb = a.refine_eq(b)
                if not ra.is_bot():
                    self_results.append(ra)
                if not rb.is_bot():
                    other_results.append(rb)
        s = DisjunctiveDomain(self_results, self._max) if self_results else self.bot()
        o = DisjunctiveDomain(other_results, self._max) if other_results else other.bot()
        return (s, o)

    def collapse(self) -> AbstractDomain:
        """Collapse all disjuncts into a single value via join."""
        if self.is_bot() or not self._disjuncts:
            return self._disjuncts[0].bot() if self._disjuncts else None
        result = self._disjuncts[0]
        for d in self._disjuncts[1:]:
            result = result.join(d)
        return result

    def __repr__(self):
        if self.is_bot():
            return "Disj(bot)"
        inner = ' | '.join(repr(d) for d in self._disjuncts)
        return f"Disj({inner})"

    def eq(self, other):
        if not isinstance(other, DisjunctiveDomain):
            return False
        if self.is_bot() and other.is_bot():
            return True
        return self.leq(other) and other.leq(self)


# ===========================================================================
# Part 4: LiftedDomain -- Error States
# ===========================================================================

class LiftState(Enum):
    NORMAL = 'normal'
    ERROR = 'error'
    BOTH = 'both'  # may be normal or error
    BOT = 'bot'


class LiftedDomain(AbstractDomain):
    """Adds error tracking to any abstract domain.

    Tracks whether computation may produce an error (division by zero,
    overflow, out of bounds, etc.) alongside normal abstract value.
    """

    def __init__(self, value: AbstractDomain, state: LiftState = LiftState.NORMAL,
                 error_info: Optional[str] = None):
        self._value = value
        self._state = state
        self._error_info = error_info

    @property
    def value(self):
        return self._value

    @property
    def state(self):
        return self._state

    @property
    def error_info(self):
        return self._error_info

    def has_error(self):
        return self._state in (LiftState.ERROR, LiftState.BOTH)

    def may_be_normal(self):
        return self._state in (LiftState.NORMAL, LiftState.BOTH)

    def top(self):
        return LiftedDomain(self._value.top(), LiftState.BOTH)

    def bot(self):
        return LiftedDomain(self._value.bot(), LiftState.BOT)

    def is_top(self):
        return self._value.is_top() and self._state == LiftState.BOTH

    def is_bot(self):
        return self._state == LiftState.BOT

    def _join_state(self, s1, s2):
        if s1 == LiftState.BOT:
            return s2
        if s2 == LiftState.BOT:
            return s1
        if s1 == s2:
            return s1
        return LiftState.BOTH

    def _meet_state(self, s1, s2):
        if s1 == LiftState.BOT or s2 == LiftState.BOT:
            return LiftState.BOT
        if s1 == s2:
            return s1
        if s1 == LiftState.BOTH:
            return s2
        if s2 == LiftState.BOTH:
            return s1
        return LiftState.BOT

    def leq(self, other):
        if not isinstance(other, LiftedDomain):
            return NotImplemented
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        # State ordering: BOT <= NORMAL/ERROR <= BOTH
        state_ok = (self._state == other._state or other._state == LiftState.BOTH
                     or self._state == LiftState.BOT)
        if not state_ok:
            return False
        if self._state in (LiftState.NORMAL, LiftState.BOTH):
            if not self._value.leq(other._value):
                return False
        return True

    def join(self, other):
        if not isinstance(other, LiftedDomain):
            return NotImplemented
        if self.is_bot():
            return LiftedDomain(other._value, other._state, other._error_info)
        if other.is_bot():
            return LiftedDomain(self._value, self._state, self._error_info)
        state = self._join_state(self._state, other._state)
        val = self._value.join(other._value)
        err = self._error_info or other._error_info
        return LiftedDomain(val, state, err)

    def meet(self, other):
        if not isinstance(other, LiftedDomain):
            return NotImplemented
        state = self._meet_state(self._state, other._state)
        if state == LiftState.BOT:
            return self.bot()
        val = self._value.meet(other._value)
        if val.is_bot() and state == LiftState.NORMAL:
            return self.bot()
        return LiftedDomain(val, state)

    def widen(self, other):
        if not isinstance(other, LiftedDomain):
            return NotImplemented
        if self.is_bot():
            return LiftedDomain(other._value, other._state, other._error_info)
        if other.is_bot():
            return LiftedDomain(self._value, self._state, self._error_info)
        state = self._join_state(self._state, other._state)
        val = self._value.widen(other._value)
        return LiftedDomain(val, state, self._error_info or other._error_info)

    @classmethod
    def from_concrete(cls, value: int):
        raise NotImplementedError("Use LiftedDomain.lift(domain.from_concrete(value))")

    @staticmethod
    def lift(value: AbstractDomain) -> 'LiftedDomain':
        return LiftedDomain(value, LiftState.NORMAL)

    @staticmethod
    def error(template: AbstractDomain, info: str = "error") -> 'LiftedDomain':
        return LiftedDomain(template.bot(), LiftState.ERROR, info)

    def add(self, other):
        if self.is_bot() or other.is_bot():
            return self.bot()
        val = self._value.add(other._value)
        state = self._join_state(self._state, other._state)
        return LiftedDomain(val, state if not val.is_bot() else LiftState.BOT)

    def sub(self, other):
        if self.is_bot() or other.is_bot():
            return self.bot()
        val = self._value.sub(other._value)
        state = self._join_state(self._state, other._state)
        return LiftedDomain(val, state if not val.is_bot() else LiftState.BOT)

    def mul(self, other):
        if self.is_bot() or other.is_bot():
            return self.bot()
        val = self._value.mul(other._value)
        state = self._join_state(self._state, other._state)
        return LiftedDomain(val, state if not val.is_bot() else LiftState.BOT)

    def neg(self):
        if self.is_bot():
            return self.bot()
        return LiftedDomain(self._value.neg(), self._state, self._error_info)

    def refine_lt(self, other):
        if not isinstance(other, LiftedDomain):
            return (self, other)
        ra, rb = self._value.refine_lt(other._value)
        return (LiftedDomain(ra, self._state), LiftedDomain(rb, other._state))

    def refine_eq(self, other):
        if not isinstance(other, LiftedDomain):
            return (self, other)
        ra, rb = self._value.refine_eq(other._value)
        return (LiftedDomain(ra, self._state), LiftedDomain(rb, other._state))

    def __repr__(self):
        if self.is_bot():
            return "Lifted(bot)"
        st = self._state.value
        err = f", err={self._error_info}" if self._error_info else ""
        return f"Lifted({self._value}, {st}{err})"

    def eq(self, other):
        if not isinstance(other, LiftedDomain):
            return False
        return self._state == other._state and self._value.eq(other._value)


# ===========================================================================
# Part 5: CardinalPowerDomain
# ===========================================================================

class CardinalPowerDomain(AbstractDomain):
    """Cardinal power domain: maps elements of a finite set to abstract values.

    Used for array-like abstractions: index -> abstract value, or
    enum variant -> abstract state.
    """

    def __init__(self, mapping: Dict[Any, AbstractDomain],
                 default: AbstractDomain,
                 keys: Optional[FrozenSet] = None):
        self._mapping = dict(mapping)
        self._default = default
        self._keys = keys or frozenset(mapping.keys())

    @property
    def mapping(self):
        return dict(self._mapping)

    @property
    def default(self):
        return self._default

    def get(self, key) -> AbstractDomain:
        return self._mapping.get(key, self._default)

    def set(self, key, value: AbstractDomain) -> 'CardinalPowerDomain':
        m = dict(self._mapping)
        m[key] = value
        return CardinalPowerDomain(m, self._default, self._keys | frozenset([key]))

    def top(self):
        return CardinalPowerDomain(
            {k: v.top() for k, v in self._mapping.items()},
            self._default.top(),
            self._keys
        )

    def bot(self):
        return CardinalPowerDomain(
            {k: v.bot() for k, v in self._mapping.items()},
            self._default.bot(),
            self._keys
        )

    def is_top(self):
        return self._default.is_top() and all(v.is_top() for v in self._mapping.values())

    def is_bot(self):
        return self._default.is_bot() and all(v.is_bot() for v in self._mapping.values())

    def _all_keys(self, other):
        return self._keys | other._keys

    def leq(self, other):
        if not isinstance(other, CardinalPowerDomain):
            return NotImplemented
        for k in self._all_keys(other):
            if not self.get(k).leq(other.get(k)):
                return False
        if not self._default.leq(other._default):
            return False
        return True

    def join(self, other):
        if not isinstance(other, CardinalPowerDomain):
            return NotImplemented
        keys = self._all_keys(other)
        m = {k: self.get(k).join(other.get(k)) for k in keys}
        default = self._default.join(other._default)
        return CardinalPowerDomain(m, default, keys)

    def meet(self, other):
        if not isinstance(other, CardinalPowerDomain):
            return NotImplemented
        keys = self._all_keys(other)
        m = {k: self.get(k).meet(other.get(k)) for k in keys}
        default = self._default.meet(other._default)
        return CardinalPowerDomain(m, default, keys)

    def widen(self, other):
        if not isinstance(other, CardinalPowerDomain):
            return NotImplemented
        keys = self._all_keys(other)
        m = {k: self.get(k).widen(other.get(k)) for k in keys}
        default = self._default.widen(other._default)
        return CardinalPowerDomain(m, default, keys)

    @classmethod
    def from_concrete(cls, value: int):
        raise NotImplementedError("Use CardinalPowerDomain constructor")

    def add(self, other):
        # Not meaningful for cardinal power
        return self.top()

    def sub(self, other):
        return self.top()

    def mul(self, other):
        return self.top()

    def neg(self):
        return self.top()

    def __repr__(self):
        if self.is_bot():
            return "CardPow(bot)"
        items = ', '.join(f"{k}: {v}" for k, v in self._mapping.items())
        return f"CardPow({{{items}}}, default={self._default})"

    def eq(self, other):
        if not isinstance(other, CardinalPowerDomain):
            return False
        if not self._default.eq(other._default):
            return False
        for k in self._all_keys(other):
            if not self.get(k).eq(other.get(k)):
                return False
        return True


# ===========================================================================
# Part 6: CompositionInterpreter -- C10 Interpreter for Composed Domains
# ===========================================================================

class CompositionEnv:
    """Environment for composition interpreter, tracks per-variable domain values."""

    def __init__(self, factory: Callable):
        self._bindings: Dict[str, AbstractDomain] = {}
        self._factory = factory

    def copy(self):
        e = CompositionEnv(self._factory)
        e._bindings = dict(self._bindings)
        return e

    def get(self, name: str) -> AbstractDomain:
        return self._bindings.get(name, self._factory())

    def set(self, name: str, value: AbstractDomain):
        self._bindings[name] = value

    def join(self, other: 'CompositionEnv') -> 'CompositionEnv':
        e = CompositionEnv(self._factory)
        all_vars = set(self._bindings.keys()) | set(other._bindings.keys())
        for v in all_vars:
            e._bindings[v] = self.get(v).join(other.get(v))
        return e

    def widen(self, other: 'CompositionEnv') -> 'CompositionEnv':
        e = CompositionEnv(self._factory)
        all_vars = set(self._bindings.keys()) | set(other._bindings.keys())
        for v in all_vars:
            e._bindings[v] = self.get(v).widen(other.get(v))
        return e

    def equals(self, other: 'CompositionEnv') -> bool:
        all_vars = set(self._bindings.keys()) | set(other._bindings.keys())
        for v in all_vars:
            if not self.get(v).eq(other.get(v)):
                return False
        return True

    @property
    def bindings(self):
        return dict(self._bindings)


class CompositionInterpreter:
    """Generic C10 abstract interpreter for composed domains.

    Works with any domain conforming to the AbstractDomain protocol.
    Parameterized by a factory: Optional[int] -> AbstractDomain.
    """

    def __init__(self, factory: Callable, max_iterations: int = 50,
                 div_zero_check: bool = True):
        self._factory = factory
        self._max_iterations = max_iterations
        self._div_zero_check = div_zero_check
        self._warnings: List[str] = []
        self._functions: Dict[str, Any] = {}

    def analyze(self, source: str) -> dict:
        """Analyze C10 source, return env + warnings + metadata."""
        tokens = lex(source)
        program = Parser(tokens).parse()
        env = CompositionEnv(self._factory)
        self._warnings = []
        self._functions = {}

        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self._functions[stmt.name] = stmt

        env = self._interpret_stmts(program.stmts, env)

        return {
            'env': env,
            'warnings': list(self._warnings),
            'functions': list(self._functions.keys()),
            'bindings': {k: repr(v) for k, v in env.bindings.items()},
        }

    def _interpret_stmts(self, stmts, env):
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env)
        return env

    def _interpret_stmt(self, stmt, env):
        if isinstance(stmt, LetDecl):
            val = self._eval_expr(stmt.value, env)
            env = env.copy()
            env.set(stmt.name, val)
            return env

        elif isinstance(stmt, Assign):
            val = self._eval_expr(stmt.value, env)
            env = env.copy()
            env.set(stmt.name, val)
            return env

        elif isinstance(stmt, IfStmt):
            return self._interpret_if(stmt, env)

        elif isinstance(stmt, WhileStmt):
            return self._interpret_while(stmt, env)

        elif isinstance(stmt, FnDecl):
            self._functions[stmt.name] = stmt
            return env

        elif isinstance(stmt, PrintStmt):
            self._eval_expr(stmt.value, env)
            return env

        elif isinstance(stmt, ReturnStmt):
            return env

        return env

    def _interpret_if(self, stmt, env):
        then_env, else_env = self._refine_condition(stmt.cond, env)

        then_stmts = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
        then_result = self._interpret_stmts(then_stmts, then_env)

        if stmt.else_body:
            else_stmts = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
            else_result = self._interpret_stmts(else_stmts, else_env)
        else:
            else_result = else_env

        return then_result.join(else_result)

    def _interpret_while(self, stmt, env):
        body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]

        current = env.copy()
        for _ in range(self._max_iterations):
            body_env, _ = self._refine_condition(stmt.cond, current)
            after_body = self._interpret_stmts(body_stmts, body_env)
            next_env = current.widen(after_body)
            if current.equals(next_env):
                break
            current = next_env

        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env

    def _eval_expr(self, expr, env):
        if isinstance(expr, IntLit):
            return self._factory(expr.value)

        elif isinstance(expr, BoolLit):
            return self._factory(1 if expr.value else 0)

        elif isinstance(expr, Var):
            return env.get(expr.name)

        elif isinstance(expr, BinOp):
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
                if self._div_zero_check and self._may_be_zero(right):
                    self._warnings.append("possible division by zero")
                return self._factory()  # TOP for division result
            elif op == '%':
                if self._div_zero_check and self._may_be_zero(right):
                    self._warnings.append("possible modulo by zero")
                return self._factory()
            elif op in ('<', '>', '<=', '>=', '==', '!='):
                return self._factory()
            return self._factory()

        elif isinstance(expr, UnaryOp):
            operand = self._eval_expr(expr.operand, env)
            if expr.op == '-':
                return operand.neg()
            return self._factory()

        elif isinstance(expr, CallExpr):
            fn_name = expr.callee if isinstance(expr.callee, str) else \
                      expr.callee.name if isinstance(expr.callee, Var) else None
            if fn_name and fn_name in self._functions:
                fn = self._functions[fn_name]
                fn_env = env.copy()
                for i, param in enumerate(fn.params):
                    if i < len(expr.args):
                        fn_env.set(param, self._eval_expr(expr.args[i], env))
                body_stmts = fn.body.stmts if isinstance(fn.body, Block) else [fn.body]
                for s in body_stmts:
                    fn_env = self._interpret_stmt(s, fn_env)
                    if isinstance(s, ReturnStmt):
                        return self._eval_expr(s.value, fn_env)
                return self._factory()
            return self._factory()

        return self._factory()

    def _may_be_zero(self, value: AbstractDomain) -> bool:
        """Check if an abstract value may contain zero."""
        # Try various domain-specific checks
        if isinstance(value, ReducedProductDomain) or isinstance(value, ProductDomain):
            for c in value.components:
                if self._may_be_zero_single(c) is False:
                    return False
            return True
        return self._may_be_zero_single(value)

    def _may_be_zero_single(self, value: AbstractDomain) -> bool:
        if isinstance(value, IntervalDomain):
            if value.is_bot():
                return False
            return value.lo <= 0 <= value.hi
        elif isinstance(value, SignDomain):
            return value.value in (SignValue.ZERO, SignValue.NON_NEG, SignValue.NON_POS, SignValue.TOP)
        elif isinstance(value, ConstDomain):
            if value.is_const():
                return value.value == 0
            return not value.is_bot()  # TOP -> may be zero
        elif isinstance(value, ParityDomain):
            return value.value in (ParityValue.EVEN, ParityValue.TOP)
        elif isinstance(value, LiftedDomain):
            if value.is_bot():
                return False
            return self._may_be_zero(value.value)
        elif isinstance(value, DisjunctiveDomain):
            return any(self._may_be_zero(d) for d in value.disjuncts)
        return True  # Unknown domain: conservative

    def _refine_condition(self, cond, env):
        """Returns (then_env, else_env) refined by condition."""
        then_env = env.copy()
        else_env = env.copy()

        if isinstance(cond, BinOp):
            left_val = self._eval_expr(cond.left, env)
            right_val = self._eval_expr(cond.right, env)

            if isinstance(cond.left, Var) and isinstance(cond.right, Var):
                lname, rname = cond.left.name, cond.right.name
                self._refine_var_var(cond.op, lname, rname, left_val, right_val,
                                     then_env, else_env)
            elif isinstance(cond.left, Var) and isinstance(cond.right, IntLit):
                lname = cond.left.name
                rval = self._factory(cond.right.value)
                self._refine_var_const(cond.op, lname, left_val, rval, then_env, else_env)
            elif isinstance(cond.left, IntLit) and isinstance(cond.right, Var):
                rname = cond.right.name
                lval = self._factory(cond.left.value)
                # Flip the operator
                flipped = {'<': '>', '>': '<', '<=': '>=', '>=': '<=', '==': '==', '!=': '!='}
                self._refine_var_const(flipped.get(cond.op, cond.op), rname, right_val, lval,
                                       then_env, else_env)

        return (then_env, else_env)

    def _refine_var_var(self, op, lname, rname, lval, rval, then_env, else_env):
        if op == '<':
            lt, rt = lval.refine_lt(rval)
            then_env.set(lname, lt)
            then_env.set(rname, rt)
            el, er = rval.refine_le(lval)
            else_env.set(rname, el)
            else_env.set(lname, er)
        elif op == '<=':
            lt, rt = lval.refine_le(rval) if hasattr(lval, 'refine_le') else (lval, rval)
            then_env.set(lname, lt)
            then_env.set(rname, rt)
            el, er = rval.refine_lt(lval)
            else_env.set(rname, el)
            else_env.set(lname, er)
        elif op == '>':
            lt, rt = rval.refine_lt(lval)
            then_env.set(rname, lt)
            then_env.set(lname, rt)
            el, er = lval.refine_le(rval) if hasattr(lval, 'refine_le') else (lval, rval)
            else_env.set(lname, el)
            else_env.set(rname, er)
        elif op == '>=':
            lt, rt = rval.refine_le(lval) if hasattr(rval, 'refine_le') else (rval, lval)
            then_env.set(rname, lt)
            then_env.set(lname, rt)
            el, er = lval.refine_lt(rval)
            else_env.set(lname, el)
            else_env.set(rname, er)
        elif op == '==':
            lt, rt = lval.refine_eq(rval)
            then_env.set(lname, lt)
            then_env.set(rname, rt)
            el, er = lval.refine_ne(rval)
            else_env.set(lname, el)
            else_env.set(rname, er)
        elif op == '!=':
            lt, rt = lval.refine_ne(rval)
            then_env.set(lname, lt)
            then_env.set(rname, rt)
            el, er = lval.refine_eq(rval)
            else_env.set(lname, el)
            else_env.set(rname, er)

    def _refine_var_const(self, op, vname, vval, cval, then_env, else_env):
        if op == '<':
            lt, _ = vval.refine_lt(cval)
            then_env.set(vname, lt)
            er, _ = cval.refine_le(vval)
            else_env.set(vname, er)
        elif op == '<=':
            lt, _ = vval.refine_le(cval) if hasattr(vval, 'refine_le') else (vval, cval)
            then_env.set(vname, lt)
            er, _ = cval.refine_lt(vval)
            else_env.set(vname, er)
        elif op == '>':
            lt, _ = cval.refine_lt(vval)
            then_env.set(vname, lt)
            er, _ = vval.refine_le(cval) if hasattr(vval, 'refine_le') else (vval, cval)
            else_env.set(vname, er)
        elif op == '>=':
            lt, _ = cval.refine_le(vval) if hasattr(cval, 'refine_le') else (cval, vval)
            then_env.set(vname, lt)
            er, _ = vval.refine_lt(cval)
            else_env.set(vname, er)
        elif op == '==':
            lt, _ = vval.refine_eq(cval)
            then_env.set(vname, lt)
            er, _ = vval.refine_ne(cval)
            else_env.set(vname, er)
        elif op == '!=':
            lt, _ = vval.refine_ne(cval)
            then_env.set(vname, lt)
            er, _ = vval.refine_eq(cval)
            else_env.set(vname, er)


# ===========================================================================
# Part 7: PrecisionComparator -- Compare Domain Compositions
# ===========================================================================

@dataclass
class ComparisonResult:
    """Result of comparing two domain compositions."""
    source: str
    domain1_name: str
    domain2_name: str
    domain1_bindings: Dict[str, str]
    domain2_bindings: Dict[str, str]
    domain1_warnings: List[str]
    domain2_warnings: List[str]
    precision_winner: str  # 'domain1', 'domain2', 'equal', 'incomparable'
    warning_diff: List[str]


def compare_compositions(source: str,
                         factory1: Callable, name1: str,
                         factory2: Callable, name2: str,
                         **kwargs) -> ComparisonResult:
    """Compare two domain compositions on the same source code.

    Returns which composition is more precise (tighter bindings, more warnings).
    """
    interp1 = CompositionInterpreter(factory1, **kwargs)
    interp2 = CompositionInterpreter(factory2, **kwargs)

    r1 = interp1.analyze(source)
    r2 = interp2.analyze(source)

    b1 = r1['bindings']
    b2 = r2['bindings']
    w1 = r1['warnings']
    w2 = r2['warnings']

    # More warnings = more precise (catching more potential issues)
    w1_extra = [w for w in w1 if w not in w2]
    w2_extra = [w for w in w2 if w not in w1]

    if len(w1) > len(w2):
        winner = 'domain1'
    elif len(w2) > len(w1):
        winner = 'domain2'
    else:
        winner = 'equal'

    return ComparisonResult(
        source=source,
        domain1_name=name1,
        domain2_name=name2,
        domain1_bindings=b1,
        domain2_bindings=b2,
        domain1_warnings=w1,
        domain2_warnings=w2,
        precision_winner=winner,
        warning_diff=w1_extra + w2_extra,
    )


# ===========================================================================
# Part 8: High-Level API Functions
# ===========================================================================

def compose_domains(*domain_types: Type[AbstractDomain],
                    auto_reduce: bool = True,
                    fixpoint: bool = False) -> Callable:
    """Create a domain factory from multiple domain types.

    Usage:
        factory = compose_domains(SignDomain, IntervalDomain, ParityDomain)
        interp = CompositionInterpreter(factory)
        result = interp.analyze("let x = 5; let y = x + 1;")
    """
    builder = ReducedProductBuilder()
    for dt in domain_types:
        builder.add(dt)
    if auto_reduce:
        builder.auto_reduce()
    if fixpoint:
        builder.fixpoint()
    return builder.build()


def analyze_with_composition(source: str,
                             *domain_types: Type[AbstractDomain],
                             auto_reduce: bool = True,
                             **kwargs) -> dict:
    """Analyze C10 source with a composed domain."""
    factory = compose_domains(*domain_types, auto_reduce=auto_reduce)
    interp = CompositionInterpreter(factory, **kwargs)
    return interp.analyze(source)


def analyze_single_domain(source: str,
                          domain_type: Type[AbstractDomain],
                          **kwargs) -> dict:
    """Analyze C10 source with a single domain."""
    def factory(value=None):
        if value is None:
            return domain_type.from_concrete(0).top()
        return domain_type.from_concrete(value)
    interp = CompositionInterpreter(factory, **kwargs)
    return interp.analyze(source)


def full_composition_analysis(source: str) -> dict:
    """Run analysis with multiple domain compositions and compare.

    Returns results from:
    1. Sign only
    2. Interval only
    3. Sign + Interval (reduced)
    4. Sign + Interval + Parity (reduced)
    5. Sign + Interval + Const (reduced)
    6. Sign + Interval + Const + Parity (full, reduced, fixpoint)
    """
    configs = [
        ('Sign', [SignDomain]),
        ('Interval', [IntervalDomain]),
        ('Sign+Interval', [SignDomain, IntervalDomain]),
        ('Sign+Interval+Parity', [SignDomain, IntervalDomain, ParityDomain]),
        ('Sign+Interval+Const', [SignDomain, IntervalDomain, ConstDomain]),
        ('Full (Sign+Interval+Const+Parity)', [SignDomain, IntervalDomain, ConstDomain, ParityDomain]),
    ]

    results = {}
    for name, domains in configs:
        if len(domains) == 1:
            r = analyze_single_domain(source, domains[0])
        else:
            r = analyze_with_composition(source, *domains, auto_reduce=True)
        results[name] = {
            'bindings': r['bindings'],
            'warnings': r['warnings'],
            'warning_count': len(r['warnings']),
        }

    return {
        'source': source,
        'results': results,
    }


def composition_summary(source: str) -> str:
    """Human-readable summary of composition analysis."""
    analysis = full_composition_analysis(source)
    lines = ["=== Domain Composition Analysis ===", ""]
    lines.append(f"Source: {source[:80]}...")
    lines.append("")

    for name, data in analysis['results'].items():
        lines.append(f"--- {name} ---")
        lines.append(f"  Warnings: {data['warning_count']}")
        for var, val in data['bindings'].items():
            lines.append(f"  {var} = {val}")
        lines.append("")

    return '\n'.join(lines)
