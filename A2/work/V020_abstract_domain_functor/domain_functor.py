"""
V020: Abstract Domain Functor
=============================

A composable algebra of abstract domains. Provides:

1. **Domain protocol** -- ABC that every abstract domain must implement
   (lattice operations + arithmetic transfer functions + concretization)

2. **Concrete domains** -- Sign, Interval, Constant, Parity wrapped in protocol

3. **Domain functors** -- Higher-order constructors that build new domains
   from existing ones:
   - Product(D1, D2, ..., Dn)  -- independent product of N domains
   - ReducedProduct(D1, ..., Dn, reducers)  -- product with cross-domain reduction
   - Flat(S)  -- flat lattice from a finite set S
   - Powerset(D)  -- powerset domain (disjunctive completion)
   - Optional(D)  -- adds explicit bottom to any domain

4. **FunctorInterpreter** -- Generic C10 AST interpreter parameterized by
   any domain conforming to the protocol. Write the domain once, get the
   interpreter for free.

Composes: C010 (parser/AST), C039 (reference for domain logic)
"""

import sys, os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, FrozenSet, Generic, List, Optional, Sequence,
    Set, Tuple, Type, TypeVar, Union,
)
from enum import Enum
from copy import deepcopy
import math

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_challenges = os.path.join(os.path.dirname(os.path.dirname(_here)), '..', 'challenges')
_c010 = os.path.normpath(os.path.join(_challenges, 'C010_stack_vm'))
if _c010 not in sys.path:
    sys.path.insert(0, _c010)

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, Var, IntLit, BinOp, UnaryOp,
    CallExpr, BoolLit,
)

# ---------------------------------------------------------------------------
# Domain Protocol (ABC)
# ---------------------------------------------------------------------------

T = TypeVar('T')


class AbstractDomain(ABC):
    """Protocol that every abstract domain must implement.

    An abstract domain is a lattice with:
    - top / bot elements
    - join (least upper bound), meet (greatest lower bound)
    - widen (for convergence), narrow (optional refinement)
    - Transfer functions for arithmetic and comparisons
    - Concretization / abstraction interface
    """

    # -- Lattice elements --------------------------------------------------

    @abstractmethod
    def top(self) -> 'AbstractDomain':
        """Return the top element (least precise / all values)."""

    @abstractmethod
    def bot(self) -> 'AbstractDomain':
        """Return the bottom element (unreachable / no values)."""

    @abstractmethod
    def is_top(self) -> bool: ...

    @abstractmethod
    def is_bot(self) -> bool: ...

    # -- Lattice operations ------------------------------------------------

    @abstractmethod
    def join(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Least upper bound."""

    @abstractmethod
    def meet(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Greatest lower bound."""

    @abstractmethod
    def widen(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Widening operator for convergence."""

    def narrow(self, other: 'AbstractDomain') -> 'AbstractDomain':
        """Narrowing (optional, default = meet)."""
        return self.meet(other)

    @abstractmethod
    def leq(self, other: 'AbstractDomain') -> bool:
        """Partial order: self <= other (self is more precise)."""

    def eq(self, other: 'AbstractDomain') -> bool:
        """Equality in the lattice."""
        return self.leq(other) and other.leq(self)

    # -- Abstraction / concretization --------------------------------------

    @classmethod
    @abstractmethod
    def from_concrete(cls, value: int) -> 'AbstractDomain':
        """Abstract a concrete integer value."""

    # -- Transfer functions ------------------------------------------------

    @abstractmethod
    def add(self, other: 'AbstractDomain') -> 'AbstractDomain': ...

    @abstractmethod
    def sub(self, other: 'AbstractDomain') -> 'AbstractDomain': ...

    @abstractmethod
    def mul(self, other: 'AbstractDomain') -> 'AbstractDomain': ...

    @abstractmethod
    def neg(self) -> 'AbstractDomain': ...

    # -- Comparison refinement ---------------------------------------------

    def refine_lt(self, other: 'AbstractDomain') -> Tuple['AbstractDomain', 'AbstractDomain']:
        """Refine self and other under constraint self < other.
        Returns (refined_self, refined_other). Default: no refinement."""
        return (self, other)

    def refine_le(self, other: 'AbstractDomain') -> Tuple['AbstractDomain', 'AbstractDomain']:
        return (self, other)

    def refine_eq(self, other: 'AbstractDomain') -> Tuple['AbstractDomain', 'AbstractDomain']:
        m = self.meet(other)
        return (m, m)

    def refine_ne(self, other: 'AbstractDomain') -> Tuple['AbstractDomain', 'AbstractDomain']:
        return (self, other)

    # -- String representation ---------------------------------------------

    @abstractmethod
    def __repr__(self) -> str: ...

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.eq(other)

    def __hash__(self):
        return id(self)


# ---------------------------------------------------------------------------
# Concrete Domain: Sign
# ---------------------------------------------------------------------------

class SignValue(Enum):
    BOT = 'bot'
    NEG = 'neg'
    ZERO = 'zero'
    POS = 'pos'
    NON_NEG = 'non_neg'
    NON_POS = 'non_pos'
    TOP = 'top'


class SignDomain(AbstractDomain):
    """Sign abstract domain: {bot, neg, zero, pos, non_neg, non_pos, top}."""

    __slots__ = ('_v',)

    def __init__(self, v: SignValue = SignValue.TOP):
        self._v = v

    @property
    def value(self):
        return self._v

    def top(self):
        return SignDomain(SignValue.TOP)

    def bot(self):
        return SignDomain(SignValue.BOT)

    def is_top(self):
        return self._v == SignValue.TOP

    def is_bot(self):
        return self._v == SignValue.BOT

    # Lattice ordering
    _order = {
        SignValue.BOT: set(),
        SignValue.NEG: {SignValue.BOT},
        SignValue.ZERO: {SignValue.BOT},
        SignValue.POS: {SignValue.BOT},
        SignValue.NON_NEG: {SignValue.BOT, SignValue.ZERO, SignValue.POS},
        SignValue.NON_POS: {SignValue.BOT, SignValue.ZERO, SignValue.NEG},
        SignValue.TOP: {SignValue.BOT, SignValue.NEG, SignValue.ZERO, SignValue.POS,
                        SignValue.NON_NEG, SignValue.NON_POS},
    }

    def leq(self, other):
        if not isinstance(other, SignDomain):
            return NotImplemented
        if self._v == other._v:
            return True
        return self._v in self._order.get(other._v, set())

    def join(self, other):
        if not isinstance(other, SignDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == b:
            return SignDomain(a)
        if a == SignValue.BOT:
            return SignDomain(b)
        if b == SignValue.BOT:
            return SignDomain(a)
        if a == SignValue.TOP or b == SignValue.TOP:
            return SignDomain(SignValue.TOP)
        # Both are non-bot, non-top
        tbl = {
            frozenset({SignValue.NEG, SignValue.ZERO}): SignValue.NON_POS,
            frozenset({SignValue.POS, SignValue.ZERO}): SignValue.NON_NEG,
            frozenset({SignValue.NEG, SignValue.POS}): SignValue.TOP,
            frozenset({SignValue.NEG, SignValue.NON_NEG}): SignValue.TOP,
            frozenset({SignValue.POS, SignValue.NON_POS}): SignValue.TOP,
            frozenset({SignValue.NEG, SignValue.NON_POS}): SignValue.NON_POS,
            frozenset({SignValue.POS, SignValue.NON_NEG}): SignValue.NON_NEG,
            frozenset({SignValue.ZERO, SignValue.NON_NEG}): SignValue.NON_NEG,
            frozenset({SignValue.ZERO, SignValue.NON_POS}): SignValue.NON_POS,
            frozenset({SignValue.NON_NEG, SignValue.NON_POS}): SignValue.TOP,
        }
        return SignDomain(tbl.get(frozenset({a, b}), SignValue.TOP))

    def meet(self, other):
        if not isinstance(other, SignDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == b:
            return SignDomain(a)
        if a == SignValue.TOP:
            return SignDomain(b)
        if b == SignValue.TOP:
            return SignDomain(a)
        if a == SignValue.BOT or b == SignValue.BOT:
            return SignDomain(SignValue.BOT)
        # Check if one is contained in the other
        if a in self._order.get(b, set()):
            return SignDomain(a)
        if b in self._order.get(a, set()):
            return SignDomain(b)
        # Both are incomparable or overlapping
        tbl = {
            frozenset({SignValue.NON_NEG, SignValue.NON_POS}): SignValue.ZERO,
        }
        return SignDomain(tbl.get(frozenset({a, b}), SignValue.BOT))

    def widen(self, other):
        return self.join(other)

    @classmethod
    def from_concrete(cls, value: int):
        if value > 0:
            return cls(SignValue.POS)
        elif value < 0:
            return cls(SignValue.NEG)
        else:
            return cls(SignValue.ZERO)

    def add(self, other):
        if not isinstance(other, SignDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == SignValue.BOT or b == SignValue.BOT:
            return SignDomain(SignValue.BOT)
        if a == SignValue.ZERO:
            return SignDomain(b)
        if b == SignValue.ZERO:
            return SignDomain(a)
        if a == SignValue.POS and b == SignValue.POS:
            return SignDomain(SignValue.POS)
        if a == SignValue.NEG and b == SignValue.NEG:
            return SignDomain(SignValue.NEG)
        if a == SignValue.POS and b == SignValue.NON_NEG:
            return SignDomain(SignValue.POS)
        if a == SignValue.NON_NEG and b == SignValue.POS:
            return SignDomain(SignValue.POS)
        if a == SignValue.NEG and b == SignValue.NON_POS:
            return SignDomain(SignValue.NEG)
        if a == SignValue.NON_POS and b == SignValue.NEG:
            return SignDomain(SignValue.NEG)
        if a == SignValue.NON_NEG and b == SignValue.NON_NEG:
            return SignDomain(SignValue.NON_NEG)
        if a == SignValue.NON_POS and b == SignValue.NON_POS:
            return SignDomain(SignValue.NON_POS)
        return SignDomain(SignValue.TOP)

    def sub(self, other):
        if not isinstance(other, SignDomain):
            return NotImplemented
        return self.add(other.neg())

    def mul(self, other):
        if not isinstance(other, SignDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == SignValue.BOT or b == SignValue.BOT:
            return SignDomain(SignValue.BOT)
        if a == SignValue.ZERO or b == SignValue.ZERO:
            return SignDomain(SignValue.ZERO)
        if a == SignValue.TOP or b == SignValue.TOP:
            return SignDomain(SignValue.TOP)
        pos_set = {SignValue.POS, SignValue.NON_NEG}
        neg_set = {SignValue.NEG, SignValue.NON_POS}
        if a in pos_set and b in pos_set:
            return SignDomain(SignValue.NON_NEG)
        if a in neg_set and b in neg_set:
            return SignDomain(SignValue.NON_NEG)
        if (a in pos_set and b in neg_set) or (a in neg_set and b in pos_set):
            return SignDomain(SignValue.NON_POS)
        return SignDomain(SignValue.TOP)

    def neg(self):
        flip = {
            SignValue.BOT: SignValue.BOT,
            SignValue.POS: SignValue.NEG,
            SignValue.NEG: SignValue.POS,
            SignValue.ZERO: SignValue.ZERO,
            SignValue.NON_NEG: SignValue.NON_POS,
            SignValue.NON_POS: SignValue.NON_NEG,
            SignValue.TOP: SignValue.TOP,
        }
        return SignDomain(flip[self._v])

    def refine_lt(self, other):
        # self < other
        a, b = self._v, other._v
        if b == SignValue.ZERO or b == SignValue.NEG or b == SignValue.NON_POS:
            a_ref = SignDomain(self.meet(SignDomain(SignValue.NEG))._v)
        elif b == SignValue.POS or b == SignValue.NON_NEG:
            a_ref = self  # self could be anything < something positive
        else:
            a_ref = self
        return (a_ref, other)

    def refine_eq(self, other):
        m = self.meet(other)
        return (m, m)

    def __repr__(self):
        return f"Sign({self._v.value})"

    def __hash__(self):
        return hash(('SignDomain', self._v))

    def eq(self, other):
        if not isinstance(other, SignDomain):
            return False
        return self._v == other._v


# ---------------------------------------------------------------------------
# Concrete Domain: Interval
# ---------------------------------------------------------------------------

INF = float('inf')
NEG_INF = float('-inf')


class IntervalDomain(AbstractDomain):
    """Interval abstract domain: [lo, hi] with lo > hi = bot."""

    __slots__ = ('lo', 'hi')

    def __init__(self, lo: float = NEG_INF, hi: float = INF):
        self.lo = lo
        self.hi = hi

    def top(self):
        return IntervalDomain(NEG_INF, INF)

    def bot(self):
        return IntervalDomain(1, 0)  # lo > hi = bot

    def is_top(self):
        return self.lo == NEG_INF and self.hi == INF

    def is_bot(self):
        return self.lo > self.hi

    def leq(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        return other.lo <= self.lo and self.hi <= other.hi

    def join(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot():
            return IntervalDomain(other.lo, other.hi)
        if other.is_bot():
            return IntervalDomain(self.lo, self.hi)
        return IntervalDomain(min(self.lo, other.lo), max(self.hi, other.hi))

    def meet(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot() or other.is_bot():
            return self.bot()
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        return IntervalDomain(lo, hi)

    def widen(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot():
            return IntervalDomain(other.lo, other.hi)
        if other.is_bot():
            return IntervalDomain(self.lo, self.hi)
        lo = self.lo if other.lo >= self.lo else NEG_INF
        hi = self.hi if other.hi <= self.hi else INF
        return IntervalDomain(lo, hi)

    def narrow(self, other):
        if self.is_bot() or other.is_bot():
            return self.bot()
        lo = other.lo if self.lo == NEG_INF else self.lo
        hi = other.hi if self.hi == INF else self.hi
        return IntervalDomain(lo, hi)

    @classmethod
    def from_concrete(cls, value: int):
        return cls(value, value)

    def add(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot() or other.is_bot():
            return self.bot()
        return IntervalDomain(self.lo + other.lo, self.hi + other.hi)

    def sub(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot() or other.is_bot():
            return self.bot()
        return IntervalDomain(self.lo - other.hi, self.hi - other.lo)

    def mul(self, other):
        if not isinstance(other, IntervalDomain):
            return NotImplemented
        if self.is_bot() or other.is_bot():
            return self.bot()
        products = [
            self.lo * other.lo, self.lo * other.hi,
            self.hi * other.lo, self.hi * other.hi,
        ]
        # Handle inf * 0 cases
        products = [0 if math.isnan(p) else p for p in products]
        return IntervalDomain(min(products), max(products))

    def neg(self):
        if self.is_bot():
            return self.bot()
        return IntervalDomain(-self.hi, -self.lo)

    def refine_lt(self, other):
        if self.is_bot() or other.is_bot():
            return (self.bot(), other.bot())
        # self < other: self.hi < other.hi, other.lo > self.lo
        s = IntervalDomain(self.lo, min(self.hi, other.hi - 1))
        o = IntervalDomain(max(other.lo, self.lo + 1), other.hi)
        return (s, o)

    def refine_le(self, other):
        if self.is_bot() or other.is_bot():
            return (self.bot(), other.bot())
        s = IntervalDomain(self.lo, min(self.hi, other.hi))
        o = IntervalDomain(max(other.lo, self.lo), other.hi)
        return (s, o)

    def refine_eq(self, other):
        m = self.meet(other)
        return (m, m)

    def refine_ne(self, other):
        # Only useful if one is a singleton
        if self.lo == self.hi and other.lo == self.hi and other.hi == self.hi:
            return (self.bot(), other.bot())
        return (self, other)

    def contains(self, value: int) -> bool:
        if self.is_bot():
            return False
        return self.lo <= value <= self.hi

    def __repr__(self):
        if self.is_bot():
            return "Interval(bot)"
        lo_s = '-inf' if self.lo == NEG_INF else str(int(self.lo))
        hi_s = '+inf' if self.hi == INF else str(int(self.hi))
        return f"Interval([{lo_s}, {hi_s}])"

    def __hash__(self):
        return hash(('IntervalDomain', self.lo, self.hi))

    def eq(self, other):
        if not isinstance(other, IntervalDomain):
            return False
        if self.is_bot() and other.is_bot():
            return True
        return self.lo == other.lo and self.hi == other.hi


# ---------------------------------------------------------------------------
# Concrete Domain: Constant Propagation
# ---------------------------------------------------------------------------

class ConstDomain(AbstractDomain):
    """Constant propagation domain: bot < {concrete values} < top."""

    __slots__ = ('_val', '_kind')  # _kind: 'bot', 'val', 'top'

    def __init__(self, val=None, kind='top'):
        self._val = val
        self._kind = kind

    def top(self):
        return ConstDomain(kind='top')

    def bot(self):
        return ConstDomain(kind='bot')

    def is_top(self):
        return self._kind == 'top'

    def is_bot(self):
        return self._kind == 'bot'

    def is_const(self):
        return self._kind == 'val'

    @property
    def value(self):
        return self._val if self._kind == 'val' else None

    def leq(self, other):
        if not isinstance(other, ConstDomain):
            return NotImplemented
        if self._kind == 'bot':
            return True
        if other._kind == 'top':
            return True
        if self._kind == 'val' and other._kind == 'val':
            return self._val == other._val
        return False

    def join(self, other):
        if not isinstance(other, ConstDomain):
            return NotImplemented
        if self._kind == 'bot':
            return ConstDomain(other._val, other._kind)
        if other._kind == 'bot':
            return ConstDomain(self._val, self._kind)
        if self._kind == 'val' and other._kind == 'val' and self._val == other._val:
            return ConstDomain(self._val, 'val')
        return ConstDomain(kind='top')

    def meet(self, other):
        if not isinstance(other, ConstDomain):
            return NotImplemented
        if self._kind == 'bot' or other._kind == 'bot':
            return self.bot()
        if self._kind == 'top':
            return ConstDomain(other._val, other._kind)
        if other._kind == 'top':
            return ConstDomain(self._val, self._kind)
        if self._kind == 'val' and other._kind == 'val':
            if self._val == other._val:
                return ConstDomain(self._val, 'val')
            return self.bot()
        return self.bot()

    def widen(self, other):
        return self.join(other)

    @classmethod
    def from_concrete(cls, value: int):
        return cls(value, 'val')

    def _binop(self, other, op):
        if self._kind == 'bot' or other._kind == 'bot':
            return self.bot()
        if self._kind == 'val' and other._kind == 'val':
            try:
                return ConstDomain(op(self._val, other._val), 'val')
            except (ZeroDivisionError, OverflowError):
                return self.top()
        return self.top()

    def add(self, other):
        return self._binop(other, lambda a, b: a + b)

    def sub(self, other):
        return self._binop(other, lambda a, b: a - b)

    def mul(self, other):
        return self._binop(other, lambda a, b: a * b)

    def neg(self):
        if self._kind == 'bot':
            return self.bot()
        if self._kind == 'val':
            return ConstDomain(-self._val, 'val')
        return self.top()

    def __repr__(self):
        if self._kind == 'bot':
            return "Const(bot)"
        if self._kind == 'top':
            return "Const(top)"
        return f"Const({self._val})"

    def __hash__(self):
        return hash(('ConstDomain', self._kind, self._val))

    def eq(self, other):
        if not isinstance(other, ConstDomain):
            return False
        if self._kind != other._kind:
            return False
        if self._kind == 'val':
            return self._val == other._val
        return True


# ---------------------------------------------------------------------------
# Concrete Domain: Parity
# ---------------------------------------------------------------------------

class ParityValue(Enum):
    BOT = 'bot'
    EVEN = 'even'
    ODD = 'odd'
    TOP = 'top'


class ParityDomain(AbstractDomain):
    """Parity domain: tracks even/odd."""

    __slots__ = ('_v',)

    def __init__(self, v: ParityValue = ParityValue.TOP):
        self._v = v

    @property
    def value(self):
        return self._v

    def top(self):
        return ParityDomain(ParityValue.TOP)

    def bot(self):
        return ParityDomain(ParityValue.BOT)

    def is_top(self):
        return self._v == ParityValue.TOP

    def is_bot(self):
        return self._v == ParityValue.BOT

    def leq(self, other):
        if not isinstance(other, ParityDomain):
            return NotImplemented
        if self._v == ParityValue.BOT:
            return True
        if other._v == ParityValue.TOP:
            return True
        return self._v == other._v

    def join(self, other):
        if not isinstance(other, ParityDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == b:
            return ParityDomain(a)
        if a == ParityValue.BOT:
            return ParityDomain(b)
        if b == ParityValue.BOT:
            return ParityDomain(a)
        return ParityDomain(ParityValue.TOP)

    def meet(self, other):
        if not isinstance(other, ParityDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == b:
            return ParityDomain(a)
        if a == ParityValue.TOP:
            return ParityDomain(b)
        if b == ParityValue.TOP:
            return ParityDomain(a)
        return ParityDomain(ParityValue.BOT)

    def widen(self, other):
        return self.join(other)

    @classmethod
    def from_concrete(cls, value: int):
        return cls(ParityValue.EVEN if value % 2 == 0 else ParityValue.ODD)

    def add(self, other):
        if not isinstance(other, ParityDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == ParityValue.BOT or b == ParityValue.BOT:
            return ParityDomain(ParityValue.BOT)
        if a == ParityValue.TOP or b == ParityValue.TOP:
            return ParityDomain(ParityValue.TOP)
        if a == b:
            return ParityDomain(ParityValue.EVEN)
        return ParityDomain(ParityValue.ODD)

    def sub(self, other):
        return self.add(other)  # parity of a-b = parity of a+b

    def mul(self, other):
        if not isinstance(other, ParityDomain):
            return NotImplemented
        a, b = self._v, other._v
        if a == ParityValue.BOT or b == ParityValue.BOT:
            return ParityDomain(ParityValue.BOT)
        if a == ParityValue.EVEN or b == ParityValue.EVEN:
            return ParityDomain(ParityValue.EVEN)
        if a == ParityValue.ODD and b == ParityValue.ODD:
            return ParityDomain(ParityValue.ODD)
        return ParityDomain(ParityValue.TOP)

    def neg(self):
        return ParityDomain(self._v)  # parity unchanged by negation

    def __repr__(self):
        return f"Parity({self._v.value})"

    def __hash__(self):
        return hash(('ParityDomain', self._v))

    def eq(self, other):
        if not isinstance(other, ParityDomain):
            return False
        return self._v == other._v


# ---------------------------------------------------------------------------
# Concrete Domain: Flat Lattice
# ---------------------------------------------------------------------------

class FlatDomain(AbstractDomain):
    """Flat lattice from a finite set of values: bot < {v1, v2, ...} < top.
    Useful for enumerations, string constants, etc."""

    __slots__ = ('_val', '_kind')

    def __init__(self, val=None, kind='top'):
        self._val = val
        self._kind = kind

    def top(self):
        return FlatDomain(kind='top')

    def bot(self):
        return FlatDomain(kind='bot')

    def is_top(self):
        return self._kind == 'top'

    def is_bot(self):
        return self._kind == 'bot'

    def is_val(self):
        return self._kind == 'val'

    @property
    def value(self):
        return self._val if self._kind == 'val' else None

    def leq(self, other):
        if not isinstance(other, FlatDomain):
            return NotImplemented
        if self._kind == 'bot':
            return True
        if other._kind == 'top':
            return True
        if self._kind == 'val' and other._kind == 'val':
            return self._val == other._val
        return False

    def join(self, other):
        if not isinstance(other, FlatDomain):
            return NotImplemented
        if self._kind == 'bot':
            return FlatDomain(other._val, other._kind)
        if other._kind == 'bot':
            return FlatDomain(self._val, self._kind)
        if self._kind == 'val' and other._kind == 'val' and self._val == other._val:
            return FlatDomain(self._val, 'val')
        return FlatDomain(kind='top')

    def meet(self, other):
        if not isinstance(other, FlatDomain):
            return NotImplemented
        if self._kind == 'bot' or other._kind == 'bot':
            return self.bot()
        if self._kind == 'top':
            return FlatDomain(other._val, other._kind)
        if other._kind == 'top':
            return FlatDomain(self._val, self._kind)
        if self._kind == 'val' and other._kind == 'val' and self._val == other._val:
            return FlatDomain(self._val, 'val')
        return self.bot()

    def widen(self, other):
        return self.join(other)

    @classmethod
    def from_concrete(cls, value):
        return cls(value, 'val')

    def add(self, other):
        return self.top()

    def sub(self, other):
        return self.top()

    def mul(self, other):
        return self.top()

    def neg(self):
        return self.top()

    def __repr__(self):
        if self._kind == 'bot':
            return "Flat(bot)"
        if self._kind == 'top':
            return "Flat(top)"
        return f"Flat({self._val!r})"

    def __hash__(self):
        return hash(('FlatDomain', self._kind, self._val))

    def eq(self, other):
        if not isinstance(other, FlatDomain):
            return False
        if self._kind != other._kind:
            return False
        if self._kind == 'val':
            return self._val == other._val
        return True


# ===================================================================
#  DOMAIN FUNCTORS -- Higher-order domain constructors
# ===================================================================

# ---------------------------------------------------------------------------
# Functor: Product Domain
# ---------------------------------------------------------------------------

class ProductDomain(AbstractDomain):
    """Independent product of N abstract domains.

    ProductDomain([d1, d2, ...]) where each di is an AbstractDomain.
    All operations are applied component-wise. No cross-domain reduction.
    """

    __slots__ = ('_components',)

    def __init__(self, components: List[AbstractDomain]):
        self._components = list(components)

    @property
    def components(self):
        return self._components

    def __getitem__(self, idx):
        return self._components[idx]

    def __len__(self):
        return len(self._components)

    def top(self):
        return ProductDomain([c.top() for c in self._components])

    def bot(self):
        return ProductDomain([c.bot() for c in self._components])

    def is_top(self):
        return all(c.is_top() for c in self._components)

    def is_bot(self):
        return any(c.is_bot() for c in self._components)

    def leq(self, other):
        if not isinstance(other, ProductDomain):
            return NotImplemented
        return all(a.leq(b) for a, b in zip(self._components, other._components))

    def join(self, other):
        if not isinstance(other, ProductDomain):
            return NotImplemented
        return ProductDomain([a.join(b) for a, b in zip(self._components, other._components)])

    def meet(self, other):
        if not isinstance(other, ProductDomain):
            return NotImplemented
        return ProductDomain([a.meet(b) for a, b in zip(self._components, other._components)])

    def widen(self, other):
        if not isinstance(other, ProductDomain):
            return NotImplemented
        return ProductDomain([a.widen(b) for a, b in zip(self._components, other._components)])

    def narrow(self, other):
        if not isinstance(other, ProductDomain):
            return NotImplemented
        return ProductDomain([a.narrow(b) for a, b in zip(self._components, other._components)])

    @classmethod
    def from_concrete(cls, value: int):
        raise NotImplementedError("Use ProductDomain.from_concrete_with_types")

    @staticmethod
    def from_concrete_with_types(value: int, domain_types: List[Type[AbstractDomain]]):
        return ProductDomain([dt.from_concrete(value) for dt in domain_types])

    def add(self, other):
        return ProductDomain([a.add(b) for a, b in zip(self._components, other._components)])

    def sub(self, other):
        return ProductDomain([a.sub(b) for a, b in zip(self._components, other._components)])

    def mul(self, other):
        return ProductDomain([a.mul(b) for a, b in zip(self._components, other._components)])

    def neg(self):
        return ProductDomain([c.neg() for c in self._components])

    def refine_lt(self, other):
        pairs = [a.refine_lt(b) for a, b in zip(self._components, other._components)]
        return (ProductDomain([p[0] for p in pairs]),
                ProductDomain([p[1] for p in pairs]))

    def refine_le(self, other):
        pairs = [a.refine_le(b) for a, b in zip(self._components, other._components)]
        return (ProductDomain([p[0] for p in pairs]),
                ProductDomain([p[1] for p in pairs]))

    def refine_eq(self, other):
        pairs = [a.refine_eq(b) for a, b in zip(self._components, other._components)]
        return (ProductDomain([p[0] for p in pairs]),
                ProductDomain([p[1] for p in pairs]))

    def __repr__(self):
        inner = ', '.join(repr(c) for c in self._components)
        return f"Product({inner})"

    def __hash__(self):
        return hash(('ProductDomain', tuple(hash(c) for c in self._components)))

    def eq(self, other):
        if not isinstance(other, ProductDomain):
            return False
        return all(a.eq(b) for a, b in zip(self._components, other._components))


# ---------------------------------------------------------------------------
# Functor: Reduced Product Domain
# ---------------------------------------------------------------------------

# A reducer is a callable: (list[AbstractDomain]) -> list[AbstractDomain]
Reducer = Callable[[List[AbstractDomain]], List[AbstractDomain]]


class ReducedProductDomain(ProductDomain):
    """Product domain with cross-domain reduction.

    After every operation, the reducer is applied to tighten all components
    using cross-domain information.
    """

    def __init__(self, components: List[AbstractDomain],
                 reducers: Optional[List[Reducer]] = None):
        self._reducers = reducers or []
        # Apply reduction on construction
        components = self._reduce_raw(components)
        super().__init__(components)

    def _reduce_raw(self, components: List[AbstractDomain]) -> List[AbstractDomain]:
        for reducer in self._reducers:
            components = reducer(components)
        # BOT propagation: if any component is BOT, all become BOT
        if any(c.is_bot() for c in components):
            return [c.bot() for c in components]
        return components

    def _reduce(self, components: List[AbstractDomain]) -> List[AbstractDomain]:
        return self._reduce_raw(components)

    def _wrap(self, components):
        return ReducedProductDomain(self._reduce(components), self._reducers)

    def top(self):
        return ReducedProductDomain([c.top() for c in self._components], self._reducers)

    def bot(self):
        return ReducedProductDomain([c.bot() for c in self._components], self._reducers)

    def join(self, other):
        if not isinstance(other, ReducedProductDomain):
            return NotImplemented
        components = [a.join(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def meet(self, other):
        if not isinstance(other, ReducedProductDomain):
            return NotImplemented
        components = [a.meet(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def widen(self, other):
        if not isinstance(other, ReducedProductDomain):
            return NotImplemented
        components = [a.widen(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def narrow(self, other):
        if not isinstance(other, ReducedProductDomain):
            return NotImplemented
        components = [a.narrow(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def add(self, other):
        components = [a.add(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def sub(self, other):
        components = [a.sub(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def mul(self, other):
        components = [a.mul(b) for a, b in zip(self._components, other._components)]
        return self._wrap(components)

    def neg(self):
        components = [c.neg() for c in self._components]
        return self._wrap(components)

    def refine_lt(self, other):
        pairs = [a.refine_lt(b) for a, b in zip(self._components, other._components)]
        left = self._wrap([p[0] for p in pairs])
        right = ReducedProductDomain(
            self._reduce([p[1] for p in pairs]), self._reducers)
        return (left, right)

    def refine_le(self, other):
        pairs = [a.refine_le(b) for a, b in zip(self._components, other._components)]
        left = self._wrap([p[0] for p in pairs])
        right = ReducedProductDomain(
            self._reduce([p[1] for p in pairs]), self._reducers)
        return (left, right)

    def refine_eq(self, other):
        pairs = [a.refine_eq(b) for a, b in zip(self._components, other._components)]
        left = self._wrap([p[0] for p in pairs])
        right = ReducedProductDomain(
            self._reduce([p[1] for p in pairs]), self._reducers)
        return (left, right)

    @staticmethod
    def from_concrete_with_types(value: int, domain_types: List[Type[AbstractDomain]],
                                  reducers: Optional[List[Reducer]] = None):
        components = [dt.from_concrete(value) for dt in domain_types]
        return ReducedProductDomain(components, reducers or [])

    def __repr__(self):
        inner = ', '.join(repr(c) for c in self._components)
        return f"ReducedProduct({inner})"

    def eq(self, other):
        if not isinstance(other, ReducedProductDomain):
            return False
        return all(a.eq(b) for a, b in zip(self._components, other._components))


# ---------------------------------------------------------------------------
# Functor: Powerset Domain (Disjunctive Completion)
# ---------------------------------------------------------------------------

class PowersetDomain(AbstractDomain):
    """Powerset domain: maintains a set of abstract values (disjuncts).

    Represents a disjunction of abstract values. More precise than join
    (which loses information). Bounded by max_size to prevent explosion.
    """

    def __init__(self, elements: Optional[Set[Any]] = None,
                 element_type: Optional[Type[AbstractDomain]] = None,
                 max_size: int = 8, _is_top: bool = False, _is_bot: bool = False):
        self._elements = frozenset(elements) if elements else frozenset()
        self._element_type = element_type
        self._max_size = max_size
        self._is_top_flag = _is_top
        self._is_bot_flag = _is_bot or (not elements and not _is_top)

    @property
    def elements(self):
        return self._elements

    def top(self):
        return PowersetDomain(element_type=self._element_type,
                              max_size=self._max_size, _is_top=True)

    def bot(self):
        return PowersetDomain(element_type=self._element_type,
                              max_size=self._max_size, _is_bot=True)

    def is_top(self):
        return self._is_top_flag

    def is_bot(self):
        return self._is_bot_flag

    def _collapse(self, elements):
        """If too many elements, join them down to max_size."""
        elems = list(elements)
        while len(elems) > self._max_size:
            # Join the two most similar (smallest) elements
            best_i, best_j = 0, 1
            elems[best_i] = elems[best_i].join(elems[best_j])
            elems.pop(best_j)
        return frozenset(elems)

    def leq(self, other):
        if not isinstance(other, PowersetDomain):
            return NotImplemented
        if self.is_bot():
            return True
        if other.is_top():
            return True
        if other.is_bot():
            return False
        # Every element of self must be leq some element of other
        for e in self._elements:
            if not any(e.leq(o) for o in other._elements):
                return False
        return True

    def join(self, other):
        if not isinstance(other, PowersetDomain):
            return NotImplemented
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        if self.is_top() or other.is_top():
            return self.top()
        combined = set(self._elements) | set(other._elements)
        collapsed = self._collapse(combined)
        return PowersetDomain(collapsed, self._element_type, self._max_size)

    def meet(self, other):
        if not isinstance(other, PowersetDomain):
            return NotImplemented
        if self.is_bot() or other.is_bot():
            return self.bot()
        if self.is_top():
            return other
        if other.is_top():
            return self
        # Meet = all pairwise meets (that are non-bot)
        result = set()
        for a in self._elements:
            for b in other._elements:
                m = a.meet(b)
                if not m.is_bot():
                    result.add(m)
        if not result:
            return self.bot()
        return PowersetDomain(self._collapse(result), self._element_type, self._max_size)

    def widen(self, other):
        if not isinstance(other, PowersetDomain):
            return NotImplemented
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        if self.is_top() or other.is_top():
            return self.top()
        # Widen: join matching pairs, add unmatched
        self_list = list(self._elements)
        other_list = list(other._elements)
        result = []
        used = set()
        for a in self_list:
            best = None
            best_idx = -1
            for i, b in enumerate(other_list):
                if i in used:
                    continue
                if a.eq(b):
                    best = a
                    best_idx = i
                    break
                if best is None:
                    best = a.widen(b)
                    best_idx = i
            if best is not None:
                result.append(best)
                if best_idx >= 0:
                    used.add(best_idx)
        for i, b in enumerate(other_list):
            if i not in used:
                result.append(b)
        collapsed = self._collapse(frozenset(result))
        return PowersetDomain(collapsed, self._element_type, self._max_size)

    @classmethod
    def from_concrete(cls, value: int):
        raise NotImplementedError("Use PowersetDomain.singleton()")

    @staticmethod
    def singleton(value: AbstractDomain, max_size: int = 8):
        return PowersetDomain({value}, type(value), max_size)

    def _apply_binop(self, other, op_name):
        if self.is_bot() or other.is_bot():
            return self.bot()
        if self.is_top() or other.is_top():
            return self.top()
        result = set()
        for a in self._elements:
            for b in other._elements:
                r = getattr(a, op_name)(b)
                if not r.is_bot():
                    result.add(r)
        if not result:
            return self.bot()
        return PowersetDomain(self._collapse(result), self._element_type, self._max_size)

    def add(self, other):
        return self._apply_binop(other, 'add')

    def sub(self, other):
        return self._apply_binop(other, 'sub')

    def mul(self, other):
        return self._apply_binop(other, 'mul')

    def neg(self):
        if self.is_bot():
            return self.bot()
        if self.is_top():
            return self.top()
        return PowersetDomain(
            frozenset(e.neg() for e in self._elements),
            self._element_type, self._max_size)

    def __repr__(self):
        if self.is_bot():
            return "Powerset(bot)"
        if self.is_top():
            return "Powerset(top)"
        inner = ', '.join(sorted(repr(e) for e in self._elements))
        return f"Powerset({{{inner}}})"

    def __hash__(self):
        return hash(('PowersetDomain', self._elements, self._is_top_flag, self._is_bot_flag))

    def eq(self, other):
        if not isinstance(other, PowersetDomain):
            return False
        if self.is_bot() and other.is_bot():
            return True
        if self.is_top() and other.is_top():
            return True
        return self._elements == other._elements


# ===================================================================
#  STANDARD REDUCERS
# ===================================================================

def sign_interval_reducer(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Reduce between sign and interval domains.
    Expects components = [SignDomain, IntervalDomain, ...]."""
    result = list(components)
    sign_idx = None
    interval_idx = None
    for i, c in enumerate(result):
        if isinstance(c, SignDomain):
            sign_idx = i
        elif isinstance(c, IntervalDomain):
            interval_idx = i
    if sign_idx is None or interval_idx is None:
        return result

    sign = result[sign_idx]
    interval = result[interval_idx]

    # Interval -> Sign
    if not interval.is_bot() and not interval.is_top():
        if interval.lo >= 1:
            sign = sign.meet(SignDomain(SignValue.POS))
        elif interval.lo >= 0:
            if interval.hi == 0:
                sign = sign.meet(SignDomain(SignValue.ZERO))
            else:
                sign = sign.meet(SignDomain(SignValue.NON_NEG))
        elif interval.hi <= -1:
            sign = sign.meet(SignDomain(SignValue.NEG))
        elif interval.hi <= 0:
            sign = sign.meet(SignDomain(SignValue.NON_POS))

    # Sign -> Interval
    if not sign.is_bot() and not sign.is_top():
        v = sign.value
        if v == SignValue.POS:
            interval = interval.meet(IntervalDomain(1, INF))
        elif v == SignValue.NEG:
            interval = interval.meet(IntervalDomain(NEG_INF, -1))
        elif v == SignValue.ZERO:
            interval = interval.meet(IntervalDomain(0, 0))
        elif v == SignValue.NON_NEG:
            interval = interval.meet(IntervalDomain(0, INF))
        elif v == SignValue.NON_POS:
            interval = interval.meet(IntervalDomain(NEG_INF, 0))

    result[sign_idx] = sign
    result[interval_idx] = interval
    return result


def const_all_reducer(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """If constant is known, tighten all other domains."""
    result = list(components)
    const_idx = None
    for i, c in enumerate(result):
        if isinstance(c, ConstDomain):
            const_idx = i
            break
    if const_idx is None or not result[const_idx].is_const():
        return result

    val = result[const_idx].value
    for i, c in enumerate(result):
        if i == const_idx:
            continue
        if isinstance(c, SignDomain):
            result[i] = c.meet(SignDomain.from_concrete(val))
        elif isinstance(c, IntervalDomain):
            result[i] = c.meet(IntervalDomain.from_concrete(val))
        elif isinstance(c, ParityDomain):
            result[i] = c.meet(ParityDomain.from_concrete(val))
    return result


def interval_const_reducer(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """If interval is a singleton, derive constant."""
    result = list(components)
    interval_idx = None
    const_idx = None
    for i, c in enumerate(result):
        if isinstance(c, IntervalDomain):
            interval_idx = i
        elif isinstance(c, ConstDomain):
            const_idx = i
    if interval_idx is None or const_idx is None:
        return result

    interval = result[interval_idx]
    const = result[const_idx]
    if not interval.is_bot() and interval.lo == interval.hi and not const.is_const():
        result[const_idx] = ConstDomain(int(interval.lo), 'val')
    return result


def parity_interval_reducer(components: List[AbstractDomain]) -> List[AbstractDomain]:
    """Tighten interval bounds using parity info."""
    result = list(components)
    parity_idx = None
    interval_idx = None
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
        return result

    # Interval -> Parity (singleton)
    if not interval.is_bot() and interval.lo == interval.hi:
        concrete_parity = ParityDomain.from_concrete(int(interval.lo))
        new_parity = parity.meet(concrete_parity)
        result[parity_idx] = new_parity
        return result

    # Parity -> Interval (tighten bounds)
    if parity.value in (ParityValue.EVEN, ParityValue.ODD) and not interval.is_top():
        lo, hi = interval.lo, interval.hi
        target_even = parity.value == ParityValue.EVEN
        if lo != NEG_INF:
            ilo = int(lo)
            if (ilo % 2 == 0) != target_even:
                lo = ilo + 1
        if hi != INF:
            ihi = int(hi)
            if (ihi % 2 == 0) != target_even:
                hi = ihi - 1
        result[interval_idx] = IntervalDomain(lo, hi)

    return result


def standard_reducers():
    """Standard set of reducers for Sign+Interval+Const+Parity."""
    return [const_all_reducer, sign_interval_reducer,
            interval_const_reducer, parity_interval_reducer]


# ===================================================================
#  DOMAIN FACTORY -- convenience constructors
# ===================================================================

def make_sign_interval():
    """Create a Sign x Interval reduced product domain factory."""
    def factory(value: Optional[int] = None):
        if value is not None:
            return ReducedProductDomain(
                [SignDomain.from_concrete(value), IntervalDomain.from_concrete(value)],
                [sign_interval_reducer])
        return ReducedProductDomain(
            [SignDomain(), IntervalDomain()],
            [sign_interval_reducer])
    return factory


def make_full_product():
    """Create a Sign x Interval x Const x Parity reduced product domain factory."""
    reducers = standard_reducers()

    def factory(value: Optional[int] = None):
        if value is not None:
            return ReducedProductDomain(
                [SignDomain.from_concrete(value), IntervalDomain.from_concrete(value),
                 ConstDomain.from_concrete(value), ParityDomain.from_concrete(value)],
                reducers)
        return ReducedProductDomain(
            [SignDomain(), IntervalDomain(), ConstDomain(), ParityDomain()],
            reducers)
    return factory


# ===================================================================
#  FUNCTOR INTERPRETER -- Generic interpreter parameterized by domain
# ===================================================================

class DomainEnv:
    """Abstract environment mapping variable names to domain values."""

    def __init__(self, domain_factory):
        self._vars = {}  # name -> AbstractDomain
        self._factory = domain_factory

    def copy(self):
        e = DomainEnv(self._factory)
        e._vars = dict(self._vars)
        return e

    def get(self, name: str) -> AbstractDomain:
        return self._vars.get(name, self._factory())  # TOP by default

    def set(self, name: str, value: AbstractDomain):
        self._vars[name] = value

    @property
    def names(self):
        return set(self._vars.keys())

    def join(self, other: 'DomainEnv') -> 'DomainEnv':
        result = DomainEnv(self._factory)
        all_names = self.names | other.names
        for name in all_names:
            result._vars[name] = self.get(name).join(other.get(name))
        return result

    def widen(self, other: 'DomainEnv') -> 'DomainEnv':
        result = DomainEnv(self._factory)
        all_names = self.names | other.names
        for name in all_names:
            result._vars[name] = self.get(name).widen(other.get(name))
        return result

    def narrow(self, other: 'DomainEnv') -> 'DomainEnv':
        result = DomainEnv(self._factory)
        all_names = self.names | other.names
        for name in all_names:
            result._vars[name] = self.get(name).narrow(other.get(name))
        return result

    def equals(self, other: 'DomainEnv') -> bool:
        all_names = self.names | other.names
        for name in all_names:
            if not self.get(name).eq(other.get(name)):
                return False
        return True

    def __repr__(self):
        items = ', '.join(f'{k}: {v}' for k, v in sorted(self._vars.items()))
        return f"Env({{{items}}})"


class FunctorInterpreter:
    """Generic abstract interpreter parameterized by a domain factory.

    The domain factory is a callable: Optional[int] -> AbstractDomain.
    - factory(None) returns TOP
    - factory(5) returns the abstract value for concrete 5
    """

    def __init__(self, domain_factory, max_iterations: int = 50):
        self._factory = domain_factory
        self._max_iterations = max_iterations
        self._warnings = []
        self._functions = {}  # name -> FnDecl

    def analyze(self, source: str) -> dict:
        """Analyze C10 source code, return dict with env, warnings, etc."""
        tokens = lex(source)
        program = Parser(tokens).parse()
        env = DomainEnv(self._factory)
        self._warnings = []
        self._functions = {}

        # Collect functions first
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self._functions[stmt.name] = stmt

        env = self._interpret_stmts(program.stmts, env)
        return {
            'env': env,
            'warnings': list(self._warnings),
            'functions': list(self._functions.keys()),
        }

    def _interpret_stmts(self, stmts, env: DomainEnv) -> DomainEnv:
        for stmt in stmts:
            env = self._interpret_stmt(stmt, env)
        return env

    def _interpret_stmt(self, stmt, env: DomainEnv) -> DomainEnv:
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
            return env

        elif isinstance(stmt, ReturnStmt):
            return env

        else:
            return env

    def _interpret_if(self, stmt: IfStmt, env: DomainEnv) -> DomainEnv:
        then_env, else_env = self._refine_condition(stmt.cond, env)

        then_stmts = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
        then_result = self._interpret_stmts(then_stmts, then_env)

        if stmt.else_body:
            else_stmts = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
            else_result = self._interpret_stmts(else_stmts, else_env)
        else:
            else_result = else_env

        return then_result.join(else_result)

    def _interpret_while(self, stmt: WhileStmt, env: DomainEnv) -> DomainEnv:
        body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]

        # Widening fixpoint
        current = env.copy()
        for iteration in range(self._max_iterations):
            # Refine by loop condition
            body_env, _ = self._refine_condition(stmt.cond, current)
            # Execute body
            after_body = self._interpret_stmts(body_stmts, body_env)
            # Widen
            next_env = current.widen(after_body)
            if current.equals(next_env):
                break
            current = next_env

        # Exit condition: NOT(cond)
        _, exit_env = self._refine_condition(stmt.cond, current)
        return exit_env

    def _eval_expr(self, expr, env: DomainEnv) -> AbstractDomain:
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
            elif op in ('<', '>', '<=', '>=', '==', '!='):
                return self._factory()  # TOP for comparison results
            else:
                return self._factory()

        elif isinstance(expr, UnaryOp):
            operand = self._eval_expr(expr.operand, env)
            if expr.op == '-':
                return operand.neg()
            return self._factory()

        elif isinstance(expr, CallExpr):
            # Simple function inlining for known functions
            fn_name = expr.callee if isinstance(expr.callee, str) else expr.callee.name if isinstance(expr.callee, Var) else None
            if fn_name and fn_name in self._functions:
                fn = self._functions[fn_name]
                fn_env = env.copy()
                for i, param in enumerate(fn.params):
                    if i < len(expr.args):
                        fn_env.set(param, self._eval_expr(expr.args[i], env))
                body_stmts = fn.body.stmts if isinstance(fn.body, Block) else [fn.body]
                # Look for return statement
                for s in body_stmts:
                    fn_env = self._interpret_stmt(s, fn_env)
                    if isinstance(s, ReturnStmt):
                        return self._eval_expr(s.value, fn_env)
                return self._factory()
            return self._factory()

        return self._factory()

    def _refine_condition(self, cond, env: DomainEnv):
        """Returns (then_env, else_env) refined by condition."""
        then_env = env.copy()
        else_env = env.copy()

        if isinstance(cond, BinOp):
            left_val = self._eval_expr(cond.left, env)
            right_val = self._eval_expr(cond.right, env)

            if isinstance(cond.left, Var) and isinstance(cond.right, Var):
                lname, rname = cond.left.name, cond.right.name
                if cond.op == '<':
                    lt, rt = left_val.refine_lt(right_val)
                    then_env.set(lname, lt)
                    then_env.set(rname, rt)
                    # else: >=
                    el, er = right_val.refine_le(left_val)
                    else_env.set(rname, el)
                    else_env.set(lname, er)
                elif cond.op == '<=':
                    lt, rt = left_val.refine_le(right_val)
                    then_env.set(lname, lt)
                    then_env.set(rname, rt)
                    el, er = right_val.refine_lt(left_val)
                    else_env.set(rname, el)
                    else_env.set(lname, er)
                elif cond.op == '>':
                    lt, rt = right_val.refine_lt(left_val)
                    then_env.set(rname, lt)
                    then_env.set(lname, rt)
                    el, er = left_val.refine_le(right_val)
                    else_env.set(lname, el)
                    else_env.set(rname, er)
                elif cond.op == '>=':
                    lt, rt = right_val.refine_le(left_val)
                    then_env.set(rname, lt)
                    then_env.set(lname, rt)
                    el, er = left_val.refine_lt(right_val)
                    else_env.set(lname, el)
                    else_env.set(rname, er)
                elif cond.op == '==':
                    lt, rt = left_val.refine_eq(right_val)
                    then_env.set(lname, lt)
                    then_env.set(rname, rt)
                    el, er = left_val.refine_ne(right_val)
                    else_env.set(lname, el)
                    else_env.set(rname, er)
                elif cond.op == '!=':
                    lt, rt = left_val.refine_ne(right_val)
                    then_env.set(lname, lt)
                    then_env.set(rname, rt)
                    el, er = left_val.refine_eq(right_val)
                    else_env.set(lname, el)
                    else_env.set(rname, er)

            elif isinstance(cond.left, Var) and isinstance(cond.right, IntLit):
                lname = cond.left.name
                rval = self._factory(cond.right.value)
                if cond.op == '<':
                    lt, _ = left_val.refine_lt(rval)
                    then_env.set(lname, lt)
                    el, _ = rval.refine_le(left_val)
                    else_env.set(lname, el)
                elif cond.op == '<=':
                    lt, _ = left_val.refine_le(rval)
                    then_env.set(lname, lt)
                    el, _ = rval.refine_lt(left_val)
                    else_env.set(lname, el)
                elif cond.op == '>':
                    lt, _ = rval.refine_lt(left_val)
                    then_env.set(lname, lt)
                    el, _ = left_val.refine_le(rval)
                    else_env.set(lname, el)
                elif cond.op == '>=':
                    lt, _ = rval.refine_le(left_val)
                    then_env.set(lname, lt)
                    el, _ = left_val.refine_lt(rval)
                    else_env.set(lname, el)
                elif cond.op == '==':
                    lt, _ = left_val.refine_eq(rval)
                    then_env.set(lname, lt)
                    el, _ = left_val.refine_ne(rval)
                    else_env.set(lname, el)
                elif cond.op == '!=':
                    lt, _ = left_val.refine_ne(rval)
                    then_env.set(lname, lt)
                    el, _ = left_val.refine_eq(rval)
                    else_env.set(lname, el)

            elif isinstance(cond.left, IntLit) and isinstance(cond.right, Var):
                # Flip: (5 < x) is (x > 5)
                rname = cond.right.name
                lval = self._factory(cond.left.value)
                if cond.op == '<':
                    _, rt = lval.refine_lt(right_val)
                    then_env.set(rname, rt)
                    _, er = right_val.refine_le(lval)
                    else_env.set(rname, er)
                elif cond.op == '>':
                    _, rt = right_val.refine_lt(lval)
                    then_env.set(rname, rt)
                    _, er = lval.refine_le(right_val)
                    else_env.set(rname, er)
                elif cond.op == '<=':
                    _, rt = lval.refine_le(right_val)
                    then_env.set(rname, rt)
                    _, er = right_val.refine_lt(lval)
                    else_env.set(rname, er)
                elif cond.op == '>=':
                    _, rt = right_val.refine_le(lval)
                    then_env.set(rname, rt)
                    _, er = lval.refine_lt(right_val)
                    else_env.set(rname, er)
                elif cond.op == '==':
                    _, rt = lval.refine_eq(right_val)
                    then_env.set(rname, rt)
                    _, er = lval.refine_ne(right_val)
                    else_env.set(rname, er)
                elif cond.op == '!=':
                    _, rt = lval.refine_ne(right_val)
                    then_env.set(rname, rt)
                    _, er = lval.refine_eq(right_val)
                    else_env.set(rname, er)

        return (then_env, else_env)


# ===================================================================
#  HIGH-LEVEL API
# ===================================================================

def analyze_with_domain(source: str, domain_factory, max_iterations: int = 50) -> dict:
    """Analyze source using any domain that conforms to the protocol."""
    interp = FunctorInterpreter(domain_factory, max_iterations)
    return interp.analyze(source)


def analyze_sign(source: str) -> dict:
    """Analyze with sign domain only."""
    return analyze_with_domain(source, lambda v=None: SignDomain.from_concrete(v) if v is not None else SignDomain())


def analyze_interval(source: str) -> dict:
    """Analyze with interval domain only."""
    return analyze_with_domain(source, lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain())


def analyze_sign_interval(source: str) -> dict:
    """Analyze with sign x interval reduced product."""
    return analyze_with_domain(source, make_sign_interval())


def analyze_full(source: str) -> dict:
    """Analyze with sign x interval x const x parity reduced product."""
    return analyze_with_domain(source, make_full_product())


def compare_domains(source: str) -> dict:
    """Compare results across different domain configurations."""
    results = {}
    configs = {
        'sign': lambda v=None: SignDomain.from_concrete(v) if v is not None else SignDomain(),
        'interval': lambda v=None: IntervalDomain.from_concrete(v) if v is not None else IntervalDomain(),
        'sign_interval': make_sign_interval(),
        'full': make_full_product(),
    }
    for name, factory in configs.items():
        r = analyze_with_domain(source, factory)
        results[name] = r
    return results


def get_variable_info(source: str, var_name: str, domain_factory=None) -> AbstractDomain:
    """Get the abstract value of a variable after analysis."""
    if domain_factory is None:
        domain_factory = make_full_product()
    result = analyze_with_domain(source, domain_factory)
    return result['env'].get(var_name)


def create_custom_domain(*domain_types, reducers=None):
    """Create a custom reduced product domain factory from domain types.

    Usage:
        factory = create_custom_domain(SignDomain, IntervalDomain, reducers=[...])
        result = analyze_with_domain(source, factory)
    """
    reds = reducers or []

    def factory(value=None):
        if value is not None:
            components = [dt.from_concrete(value) for dt in domain_types]
        else:
            components = [dt() for dt in domain_types]
        if reds:
            return ReducedProductDomain(components, reds)
        if len(components) == 1:
            return components[0]
        return ProductDomain(components)

    return factory
