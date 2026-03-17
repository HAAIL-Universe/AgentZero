"""
V176: Runtime Verification Monitor

Monitors execution traces against temporal specifications in real-time.
Supports past-time LTL (ptLTL), future-time LTL with finite-trace semantics,
and safety/liveness property monitoring.

Key capabilities:
- Online monitoring: process events one-at-a-time, O(|formula|) per event
- Past-time LTL: once, historically, since, previous -- evaluated in O(1) per step
- Future-time LTL with 3-valued semantics (true/false/inconclusive)
- Safety properties: detect violations immediately
- Automaton-based monitoring: convert LTL to monitor automaton
- Parametric monitoring: track property instances per parameter binding
- Statistical monitoring: frequency, timing, rate analysis
- Trace slicing: monitor sub-traces filtered by predicate

Composition:
- Standalone (no dependencies on A1's challenges)
- Can be composed with V023 (LTL model checking) for offline verification
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from fractions import Fraction
import re


# ============================================================
# Core Types
# ============================================================

class Verdict(Enum):
    """Three-valued monitoring verdict."""
    TRUE = auto()       # Property satisfied (no future can violate)
    FALSE = auto()      # Property violated (no future can satisfy)
    UNKNOWN = auto()    # Inconclusive (depends on future events)

    def __and__(self, other):
        if self is Verdict.FALSE or other is Verdict.FALSE:
            return Verdict.FALSE
        if self is Verdict.UNKNOWN or other is Verdict.UNKNOWN:
            return Verdict.UNKNOWN
        return Verdict.TRUE

    def __or__(self, other):
        if self is Verdict.TRUE or other is Verdict.TRUE:
            return Verdict.TRUE
        if self is Verdict.UNKNOWN or other is Verdict.UNKNOWN:
            return Verdict.UNKNOWN
        return Verdict.FALSE

    def __invert__(self):
        if self is Verdict.TRUE:
            return Verdict.FALSE
        if self is Verdict.FALSE:
            return Verdict.TRUE
        return Verdict.UNKNOWN


@dataclass
class Event:
    """A single event in a trace."""
    name: str
    data: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def satisfies(self, predicate):
        """Check if event satisfies a predicate string or callable."""
        if callable(predicate):
            return predicate(self)
        # String predicate: event name match or key=value match
        if '=' in predicate and not predicate.startswith('!'):
            key, val = predicate.split('=', 1)
            return str(self.data.get(key.strip())) == val.strip()
        if predicate.startswith('!'):
            return self.name != predicate[1:]
        return self.name == predicate


@dataclass
class MonitorState:
    """State of a monitor at a point in time."""
    verdict: Verdict = Verdict.UNKNOWN
    step: int = 0
    metadata: dict = field(default_factory=dict)


# ============================================================
# Temporal Formulas (AST)
# ============================================================

class Formula:
    """Base class for temporal logic formulas."""
    pass


class Atom(Formula):
    """Atomic proposition: event name or predicate."""
    def __init__(self, predicate):
        self.predicate = predicate

    def __repr__(self):
        return f"Atom({self.predicate!r})"

    def __eq__(self, other):
        return isinstance(other, Atom) and self.predicate == other.predicate

    def __hash__(self):
        return hash(('Atom', self.predicate))


class TrueF(Formula):
    """Always true."""
    def __repr__(self):
        return "True"

    def __eq__(self, other):
        return isinstance(other, TrueF)

    def __hash__(self):
        return hash('TrueF')


class FalseF(Formula):
    """Always false."""
    def __repr__(self):
        return "False"

    def __eq__(self, other):
        return isinstance(other, FalseF)

    def __hash__(self):
        return hash('FalseF')


class Not(Formula):
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Not({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Not) and self.sub == other.sub

    def __hash__(self):
        return hash(('Not', self.sub))


class And(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"And({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return isinstance(other, And) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('And', self.left, self.right))


class Or(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Or({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return isinstance(other, Or) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('Or', self.left, self.right))


class Implies(Formula):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Implies({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return isinstance(other, Implies) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('Implies', self.left, self.right))


# Future-time temporal operators
class Next(Formula):
    """X phi: phi holds at next step."""
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Next({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Next) and self.sub == other.sub

    def __hash__(self):
        return hash(('Next', self.sub))


class Eventually(Formula):
    """F phi: phi holds at some future step."""
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Eventually({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Eventually) and self.sub == other.sub

    def __hash__(self):
        return hash(('Eventually', self.sub))


class Always(Formula):
    """G phi: phi holds at all future steps."""
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Always({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Always) and self.sub == other.sub

    def __hash__(self):
        return hash(('Always', self.sub))


class Until(Formula):
    """phi U psi: phi holds until psi holds."""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Until({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return isinstance(other, Until) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('Until', self.left, self.right))


class Release(Formula):
    """phi R psi: psi holds until (and including) phi holds, or forever."""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Release({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return isinstance(other, Release) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('Release', self.left, self.right))


# Past-time temporal operators
class Previous(Formula):
    """Y phi: phi held at previous step (false at step 0)."""
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Previous({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Previous) and self.sub == other.sub

    def __hash__(self):
        return hash(('Previous', self.sub))


class Once(Formula):
    """O phi: phi held at some past step (or now)."""
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Once({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Once) and self.sub == other.sub

    def __hash__(self):
        return hash(('Once', self.sub))


class Historically(Formula):
    """H phi: phi held at all past steps (and now)."""
    def __init__(self, sub):
        self.sub = sub

    def __repr__(self):
        return f"Historically({self.sub!r})"

    def __eq__(self, other):
        return isinstance(other, Historically) and self.sub == other.sub

    def __hash__(self):
        return hash(('Historically', self.sub))


class Since(Formula):
    """phi S psi: phi has held since psi last held."""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Since({self.left!r}, {self.right!r})"

    def __eq__(self, other):
        return isinstance(other, Since) and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(('Since', self.left, self.right))


# Bounded temporal operators
class BoundedEventually(Formula):
    """F[<=k] phi: phi holds within k steps."""
    def __init__(self, sub, bound):
        self.sub = sub
        self.bound = bound

    def __repr__(self):
        return f"BoundedEventually({self.sub!r}, {self.bound})"

    def __eq__(self, other):
        return isinstance(other, BoundedEventually) and self.sub == other.sub and self.bound == other.bound

    def __hash__(self):
        return hash(('BoundedEventually', self.sub, self.bound))


class BoundedAlways(Formula):
    """G[<=k] phi: phi holds for the next k steps."""
    def __init__(self, sub, bound):
        self.sub = sub
        self.bound = bound

    def __repr__(self):
        return f"BoundedAlways({self.sub!r}, {self.bound})"

    def __eq__(self, other):
        return isinstance(other, BoundedAlways) and self.sub == other.sub and self.bound == other.bound

    def __hash__(self):
        return hash(('BoundedAlways', self.sub, self.bound))


# ============================================================
# Formula Parser
# ============================================================

def parse_formula(text):
    """Parse a temporal logic formula from string.

    Syntax:
        atom     := identifier | 'true' | 'false'
        unary    := '!' unary | 'X' unary | 'F' unary | 'G' unary
                  | 'Y' unary | 'O' unary | 'H' unary
                  | 'F[' num ']' unary | 'G[' num ']' unary
                  | '(' expr ')'
        binary   := unary ('U' unary | 'R' unary | 'S' unary
                          | '&&' unary | '||' unary | '->' unary)*
        expr     := binary
    """
    tokens = _tokenize(text)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def advance():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(t):
        if peek() != t:
            raise ValueError(f"Expected {t!r}, got {peek()!r}")
        return advance()

    def parse_expr():
        return parse_implies()

    def parse_implies():
        left = parse_or()
        while peek() == '->':
            advance()
            right = parse_or()
            left = Implies(left, right)
        return left

    def parse_or():
        left = parse_and()
        while peek() == '||':
            advance()
            right = parse_and()
            left = Or(left, right)
        return left

    def parse_and():
        left = parse_until()
        while peek() == '&&':
            advance()
            right = parse_until()
            left = And(left, right)
        return left

    def parse_until():
        left = parse_unary()
        while peek() in ('U', 'R', 'S'):
            op = advance()
            right = parse_unary()
            if op == 'U':
                left = Until(left, right)
            elif op == 'R':
                left = Release(left, right)
            elif op == 'S':
                left = Since(left, right)
        return left

    def parse_unary():
        t = peek()
        if t == '!':
            advance()
            return Not(parse_unary())
        if t == 'X':
            advance()
            return Next(parse_unary())
        if t == 'F':
            advance()
            if peek() == '[':
                advance()
                bound = int(advance())
                expect(']')
                return BoundedEventually(parse_unary(), bound)
            return Eventually(parse_unary())
        if t == 'G':
            advance()
            if peek() == '[':
                advance()
                bound = int(advance())
                expect(']')
                return BoundedAlways(parse_unary(), bound)
            return Always(parse_unary())
        if t == 'Y':
            advance()
            return Previous(parse_unary())
        if t == 'O':
            advance()
            return Once(parse_unary())
        if t == 'H':
            advance()
            return Historically(parse_unary())
        if t == '(':
            advance()
            e = parse_expr()
            expect(')')
            return e
        if t == 'true':
            advance()
            return TrueF()
        if t == 'false':
            advance()
            return FalseF()
        if t is not None:
            advance()
            return Atom(t)
        raise ValueError("Unexpected end of formula")

    result = parse_expr()
    if pos[0] < len(tokens):
        raise ValueError(f"Unexpected token: {tokens[pos[0]]!r}")
    return result


def _tokenize(text):
    """Tokenize a formula string."""
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text[i:i+2] == '->':
            tokens.append('->')
            i += 2
            continue
        if text[i:i+2] == '&&':
            tokens.append('&&')
            i += 2
            continue
        if text[i:i+2] == '||':
            tokens.append('||')
            i += 2
            continue
        if text[i] in '!()[]':
            tokens.append(text[i])
            i += 1
            continue
        # Identifier or number
        j = i
        while j < len(text) and (text[j].isalnum() or text[j] in '_'):
            j += 1
        if j > i:
            tokens.append(text[i:j])
            i = j
            continue
        raise ValueError(f"Unexpected character: {text[i]!r}")
    return tokens


# ============================================================
# Past-Time Monitor (O(|formula|) per step, definitive verdicts)
# ============================================================

class PastTimeMonitor:
    """Monitors past-time LTL formulas.

    Past-time formulas can be evaluated definitively at each step
    because they only reference the past. O(|formula|) per event.
    """

    def __init__(self, formula):
        if isinstance(formula, str):
            formula = parse_formula(formula)
        self.formula = formula
        self.step = 0
        # prev_val[f] = value of formula f at previous step
        self._prev_val = {}
        self._cur_val = {}
        self.history = []  # list of (step, verdict)

    def process(self, event):
        """Process one event. Returns Verdict.TRUE or Verdict.FALSE."""
        if isinstance(event, str):
            event = Event(event)
        self._cur_val = {}
        v = self._eval(self.formula, event)
        verdict = Verdict.TRUE if v else Verdict.FALSE
        self.history.append((self.step, verdict))
        self._prev_val = dict(self._cur_val)
        self.step += 1
        return verdict

    def _eval(self, f, event):
        """Evaluate formula at current step. Returns bool."""
        key = id(f)
        if key in self._cur_val:
            return self._cur_val[key]

        if isinstance(f, TrueF):
            result = True
        elif isinstance(f, FalseF):
            result = False
        elif isinstance(f, Atom):
            result = event.satisfies(f.predicate)
        elif isinstance(f, Not):
            result = not self._eval(f.sub, event)
        elif isinstance(f, And):
            # Must evaluate BOTH sides to maintain state for temporal operators
            lv = self._eval(f.left, event)
            rv = self._eval(f.right, event)
            result = lv and rv
        elif isinstance(f, Or):
            lv = self._eval(f.left, event)
            rv = self._eval(f.right, event)
            result = lv or rv
        elif isinstance(f, Implies):
            # Must evaluate BOTH sides to maintain state for temporal operators
            lv = self._eval(f.left, event)
            rv = self._eval(f.right, event)
            result = (not lv) or rv
        elif isinstance(f, Previous):
            # Y phi: value of phi at previous step (false at step 0)
            # Must evaluate sub now (to store it for next step) and return prev value
            self._eval(f.sub, event)  # Ensure sub is evaluated and stored
            result = self._prev_val.get(id(f.sub), False) if self.step > 0 else False
        elif isinstance(f, Once):
            # O phi: phi now OR O phi was true before
            result = self._eval(f.sub, event) or self._prev_val.get(key, False)
        elif isinstance(f, Historically):
            # H phi: phi now AND H phi was true before (true at step 0 if phi holds)
            if self.step == 0:
                result = self._eval(f.sub, event)
            else:
                result = self._eval(f.sub, event) and self._prev_val.get(key, True)
        elif isinstance(f, Since):
            # phi S psi: psi now, OR (phi now AND (phi S psi) was true before)
            result = self._eval(f.right, event) or (
                self._eval(f.left, event) and self._prev_val.get(key, False)
            )
        else:
            raise ValueError(f"PastTimeMonitor does not support future-time operator: {type(f).__name__}")

        self._cur_val[key] = result
        return result

    def reset(self):
        self.step = 0
        self._prev_val = {}
        self._cur_val = {}
        self.history = []


# ============================================================
# Future-Time Monitor (3-valued, finite-trace semantics)
# ============================================================

class FutureTimeMonitor:
    """Monitors future-time LTL formulas with 3-valued semantics.

    Uses formula rewriting: at each step, the formula is rewritten
    based on the current event, producing a residual formula that
    represents the remaining obligation.

    Verdict: TRUE (satisfied regardless of future), FALSE (violated
    regardless of future), UNKNOWN (depends on future events).
    """

    def __init__(self, formula):
        if isinstance(formula, str):
            formula = parse_formula(formula)
        self.original = formula
        self.current = formula
        self.step = 0
        self.history = []
        self._verdict = Verdict.UNKNOWN

    @property
    def verdict(self):
        return self._verdict

    def process(self, event):
        """Process one event. Returns current 3-valued verdict."""
        if isinstance(event, str):
            event = Event(event)
        self.current = self._rewrite(self.current, event)
        self.current = self._simplify(self.current)
        self._verdict = self._evaluate_verdict(self.current)
        self.history.append((self.step, self._verdict))
        self.step += 1
        return self._verdict

    def finalize(self):
        """Finalize monitoring (trace ended). Resolves UNKNOWN under finite-trace semantics.

        Under finite-trace LTL:
        - F phi at end of trace: FALSE (no more chances)
        - G phi at end of trace: TRUE (survived all steps)
        - X phi at end of trace: weak next = TRUE, strong next = FALSE
        """
        self._verdict = self._finite_eval(self.current)
        return self._verdict

    def _rewrite(self, f, event):
        """Rewrite formula after observing event."""
        if isinstance(f, TrueF) or isinstance(f, FalseF):
            return f
        if isinstance(f, Atom):
            return TrueF() if event.satisfies(f.predicate) else FalseF()
        if isinstance(f, Not):
            return Not(self._rewrite(f.sub, event))
        if isinstance(f, And):
            return And(self._rewrite(f.left, event), self._rewrite(f.right, event))
        if isinstance(f, Or):
            return Or(self._rewrite(f.left, event), self._rewrite(f.right, event))
        if isinstance(f, Implies):
            return Or(Not(self._rewrite(f.left, event)), self._rewrite(f.right, event))
        if isinstance(f, Next):
            # X phi: after seeing current event, obligation is just phi
            return f.sub
        if isinstance(f, Eventually):
            # F phi = phi || X(F phi)
            return Or(self._rewrite(f.sub, event), Eventually(f.sub))
        if isinstance(f, Always):
            # G phi = phi && X(G phi)
            return And(self._rewrite(f.sub, event), Always(f.sub))
        if isinstance(f, Until):
            # phi U psi = psi || (phi && X(phi U psi))
            return Or(
                self._rewrite(f.right, event),
                And(self._rewrite(f.left, event), Until(f.left, f.right))
            )
        if isinstance(f, Release):
            # phi R psi = (phi && psi) || (psi && X(phi R psi))
            return Or(
                And(self._rewrite(f.left, event), self._rewrite(f.right, event)),
                And(self._rewrite(f.right, event), Release(f.left, f.right))
            )
        if isinstance(f, BoundedEventually):
            # F[<=k] phi: if k==0, just check now; else phi || X(F[<=k-1] phi)
            if f.bound <= 0:
                return self._rewrite(f.sub, event)
            return Or(self._rewrite(f.sub, event), BoundedEventually(f.sub, f.bound - 1))
        if isinstance(f, BoundedAlways):
            # G[<=k] phi: if k==0, just check now; else phi && X(G[<=k-1] phi)
            if f.bound <= 0:
                return self._rewrite(f.sub, event)
            return And(self._rewrite(f.sub, event), BoundedAlways(f.sub, f.bound - 1))
        # Past-time operators in future monitor: evaluate definitively
        if isinstance(f, (Previous, Once, Historically, Since)):
            raise ValueError(f"Use PastTimeMonitor for past-time operators: {type(f).__name__}")
        raise ValueError(f"Unknown formula type: {type(f).__name__}")

    def _simplify(self, f):
        """Simplify formula (constant folding)."""
        if isinstance(f, (TrueF, FalseF, Atom)):
            return f
        if isinstance(f, Not):
            sub = self._simplify(f.sub)
            if isinstance(sub, TrueF):
                return FalseF()
            if isinstance(sub, FalseF):
                return TrueF()
            if isinstance(sub, Not):
                return sub.sub
            return Not(sub)
        if isinstance(f, And):
            l = self._simplify(f.left)
            r = self._simplify(f.right)
            if isinstance(l, FalseF) or isinstance(r, FalseF):
                return FalseF()
            if isinstance(l, TrueF):
                return r
            if isinstance(r, TrueF):
                return l
            return And(l, r)
        if isinstance(f, Or):
            l = self._simplify(f.left)
            r = self._simplify(f.right)
            if isinstance(l, TrueF) or isinstance(r, TrueF):
                return TrueF()
            if isinstance(l, FalseF):
                return r
            if isinstance(r, FalseF):
                return l
            return Or(l, r)
        if isinstance(f, Eventually):
            sub = self._simplify(f.sub)
            if isinstance(sub, TrueF):
                return TrueF()
            if isinstance(sub, FalseF):
                return FalseF()
            return Eventually(sub)
        if isinstance(f, Always):
            sub = self._simplify(f.sub)
            if isinstance(sub, TrueF):
                return TrueF()
            if isinstance(sub, FalseF):
                return FalseF()
            return Always(sub)
        if isinstance(f, Until):
            l = self._simplify(f.left)
            r = self._simplify(f.right)
            if isinstance(r, TrueF):
                return TrueF()
            if isinstance(l, FalseF):
                return r
            return Until(l, r)
        if isinstance(f, Release):
            l = self._simplify(f.left)
            r = self._simplify(f.right)
            if isinstance(r, FalseF):
                return FalseF()
            if isinstance(l, TrueF):
                return r
            return Release(l, r)
        if isinstance(f, (BoundedEventually, BoundedAlways)):
            sub = self._simplify(f.sub)
            if isinstance(sub, TrueF):
                return TrueF()
            if isinstance(sub, FalseF):
                return FalseF()
            return type(f)(sub, f.bound)
        return f

    def _evaluate_verdict(self, f):
        """Determine 3-valued verdict from residual formula."""
        if isinstance(f, TrueF):
            return Verdict.TRUE
        if isinstance(f, FalseF):
            return Verdict.FALSE
        # Any remaining temporal obligations mean inconclusive
        return Verdict.UNKNOWN

    def _finite_eval(self, f):
        """Evaluate under finite-trace (weak) semantics."""
        if isinstance(f, TrueF):
            return Verdict.TRUE
        if isinstance(f, FalseF):
            return Verdict.FALSE
        if isinstance(f, Atom):
            return Verdict.FALSE  # Unresolved atom at end
        if isinstance(f, Not):
            v = self._finite_eval(f.sub)
            return ~v
        if isinstance(f, And):
            return self._finite_eval(f.left) & self._finite_eval(f.right)
        if isinstance(f, Or):
            return self._finite_eval(f.left) | self._finite_eval(f.right)
        if isinstance(f, Next):
            return Verdict.TRUE  # Weak next: vacuously true at end
        if isinstance(f, Eventually):
            return Verdict.FALSE  # Never saw it
        if isinstance(f, Always):
            return Verdict.TRUE  # Survived all steps
        if isinstance(f, Until):
            return Verdict.FALSE  # RHS never held
        if isinstance(f, Release):
            return Verdict.TRUE  # RHS held throughout
        if isinstance(f, BoundedEventually):
            return Verdict.FALSE
        if isinstance(f, BoundedAlways):
            return Verdict.TRUE
        return Verdict.UNKNOWN

    def reset(self):
        self.current = self.original
        self.step = 0
        self.history = []
        self._verdict = Verdict.UNKNOWN


# ============================================================
# Safety Monitor (specialized for G phi -- immediate violation detection)
# ============================================================

class SafetyMonitor:
    """Monitors safety properties (G phi) with immediate violation detection.

    Optimized for the common case of "something bad never happens" or
    "an invariant always holds". Reports the first violation with context.
    """

    def __init__(self, invariant, name=None):
        """invariant: callable(Event) -> bool, or formula string for Atom check."""
        if isinstance(invariant, str):
            pred_str = invariant
            invariant = lambda e, p=pred_str: e.satisfies(p)
        self.invariant = invariant
        self.name = name or "safety"
        self.violated = False
        self.violation_event = None
        self.violation_step = None
        self.step = 0

    def process(self, event):
        """Process one event. Returns True if safe, False if violated."""
        if isinstance(event, str):
            event = Event(event)
        if not self.violated:
            if not self.invariant(event):
                self.violated = True
                self.violation_event = event
                self.violation_step = self.step
        self.step += 1
        return not self.violated

    @property
    def verdict(self):
        return Verdict.FALSE if self.violated else Verdict.UNKNOWN

    def reset(self):
        self.violated = False
        self.violation_event = None
        self.violation_step = None
        self.step = 0


# ============================================================
# Parametric Monitor (track per-parameter instances)
# ============================================================

class ParametricMonitor:
    """Monitors parametric properties: one monitor instance per parameter binding.

    Example: "for every request r, eventually r gets a response"
    Each unique request ID gets its own Eventually monitor.
    """

    def __init__(self, formula, param_key, monitor_class=None):
        """
        formula: the temporal formula to monitor per instance
        param_key: event data key to extract parameter value
        monitor_class: FutureTimeMonitor or PastTimeMonitor (default: FutureTimeMonitor)
        """
        if isinstance(formula, str):
            formula = parse_formula(formula)
        self.formula = formula
        self.param_key = param_key
        self.monitor_class = monitor_class or FutureTimeMonitor
        self.instances = {}  # param_value -> monitor
        self.step = 0

    def process(self, event):
        """Process event. Creates new monitor instance if new parameter value seen."""
        if isinstance(event, str):
            event = Event(event)
        param_val = event.data.get(self.param_key)
        if param_val is None:
            self.step += 1
            return {}

        if param_val not in self.instances:
            self.instances[param_val] = self.monitor_class(self.formula)

        verdict = self.instances[param_val].process(event)
        self.step += 1
        return {param_val: verdict}

    def get_verdicts(self):
        """Get current verdicts for all instances."""
        result = {}
        for k, m in self.instances.items():
            if hasattr(m, 'verdict'):
                result[k] = m.verdict
            elif m.history:
                result[k] = m.history[-1][1]
        return result

    def get_violations(self):
        """Get parameter values whose monitors report FALSE."""
        return {k: v for k, v in self.get_verdicts().items() if v is Verdict.FALSE}

    def finalize(self):
        """Finalize all instances (for FutureTimeMonitor)."""
        results = {}
        for k, m in self.instances.items():
            if isinstance(m, FutureTimeMonitor):
                results[k] = m.finalize()
            elif m.history:
                results[k] = m.history[-1][1]
        return results


# ============================================================
# Statistical Monitor (frequency, timing, rate)
# ============================================================

class StatisticalMonitor:
    """Monitors statistical properties of event streams.

    - Event frequency / count
    - Inter-event timing
    - Rate monitoring (events per time window)
    - Sequence detection
    """

    def __init__(self):
        self.counts = {}          # event_name -> count
        self.first_seen = {}      # event_name -> timestamp
        self.last_seen = {}       # event_name -> timestamp
        self.intervals = {}       # event_name -> list of inter-event intervals
        self.total_events = 0
        self.window_events = []   # (timestamp, event_name) for rate monitoring
        self.step = 0

    def process(self, event):
        """Process one event, updating all statistics."""
        if isinstance(event, str):
            event = Event(event)
        name = event.name
        ts = event.timestamp

        self.counts[name] = self.counts.get(name, 0) + 1
        if name not in self.first_seen:
            self.first_seen[name] = ts
        else:
            interval = ts - self.last_seen[name]
            if name not in self.intervals:
                self.intervals[name] = []
            self.intervals[name].append(interval)
        self.last_seen[name] = ts
        self.total_events += 1
        self.window_events.append((ts, name))
        self.step += 1

    def count(self, event_name):
        return self.counts.get(event_name, 0)

    def rate(self, event_name, window_start, window_end):
        """Count events in a time window."""
        return sum(1 for ts, n in self.window_events
                   if n == event_name and window_start <= ts <= window_end)

    def mean_interval(self, event_name):
        """Average time between consecutive events of the same type."""
        intervals = self.intervals.get(event_name, [])
        if not intervals:
            return None
        return sum(intervals) / len(intervals)

    def max_interval(self, event_name):
        intervals = self.intervals.get(event_name, [])
        return max(intervals) if intervals else None

    def min_interval(self, event_name):
        intervals = self.intervals.get(event_name, [])
        return min(intervals) if intervals else None

    def frequency(self, event_name):
        """Fraction of events that are this type."""
        if self.total_events == 0:
            return 0.0
        return self.counts.get(event_name, 0) / self.total_events


# ============================================================
# Trace Slicer (filter traces by predicate)
# ============================================================

class TraceSlicer:
    """Slices a trace to monitor properties on sub-traces.

    Only forwards events matching the filter to the inner monitor.
    Useful for monitoring properties of specific event categories.
    """

    def __init__(self, monitor, filter_pred):
        """
        monitor: inner monitor to forward filtered events to
        filter_pred: callable(Event) -> bool, or event name string
        """
        self.monitor = monitor
        if isinstance(filter_pred, str):
            name = filter_pred
            self.filter_pred = lambda e, n=name: e.name == n or e.name.startswith(n + '.')
        else:
            self.filter_pred = filter_pred
        self.total_events = 0
        self.forwarded_events = 0

    def process(self, event):
        """Process event, forwarding to inner monitor if it passes filter."""
        if isinstance(event, str):
            event = Event(event)
        self.total_events += 1
        if self.filter_pred(event):
            self.forwarded_events += 1
            return self.monitor.process(event)
        return None

    @property
    def verdict(self):
        if hasattr(self.monitor, 'verdict'):
            return self.monitor.verdict
        return Verdict.UNKNOWN


# ============================================================
# Composite Monitor (monitor multiple properties simultaneously)
# ============================================================

class CompositeMonitor:
    """Monitors multiple properties simultaneously over the same trace."""

    def __init__(self):
        self.monitors = {}  # name -> monitor

    def add(self, name, monitor):
        self.monitors[name] = monitor
        return self

    def process(self, event):
        """Process event through all monitors. Returns dict of verdicts."""
        results = {}
        for name, mon in self.monitors.items():
            v = mon.process(event)
            if v is not None:
                results[name] = v
        return results

    def get_verdicts(self):
        """Get current verdicts for all monitors."""
        results = {}
        for name, mon in self.monitors.items():
            if hasattr(mon, 'verdict'):
                results[name] = mon.verdict
            elif hasattr(mon, 'history') and mon.history:
                results[name] = mon.history[-1][1]
        return results

    def get_violations(self):
        """Get monitors reporting FALSE."""
        return {k: v for k, v in self.get_verdicts().items() if v is Verdict.FALSE}


# ============================================================
# Response Pattern Monitor (request-response matching)
# ============================================================

class ResponsePatternMonitor:
    """Monitors request-response patterns with timeouts.

    For each request, expects a matching response within a deadline.
    Tracks: pending requests, matched pairs, timeouts.
    """

    def __init__(self, request_name, response_name, match_key=None, deadline=None):
        self.request_name = request_name
        self.response_name = response_name
        self.match_key = match_key  # event data key for matching
        self.deadline = deadline    # max time between request and response
        self.pending = {}          # match_value -> (request_event, request_step)
        self.matched = []          # (request, response, latency)
        self.timed_out = []        # (request, step)
        self.step = 0

    def process(self, event):
        """Process event, matching requests to responses."""
        if isinstance(event, str):
            event = Event(event)

        # Check for timeouts
        if self.deadline is not None:
            expired = []
            for k, (req, req_step) in self.pending.items():
                if event.timestamp - req.timestamp > self.deadline:
                    expired.append(k)
                    self.timed_out.append((req, req_step))
            for k in expired:
                del self.pending[k]

        if event.name == self.request_name:
            key = event.data.get(self.match_key, self.step) if self.match_key else self.step
            self.pending[key] = (event, self.step)
        elif event.name == self.response_name:
            key = event.data.get(self.match_key) if self.match_key else None
            if key is not None and key in self.pending:
                req, req_step = self.pending.pop(key)
                latency = event.timestamp - req.timestamp
                self.matched.append((req, event, latency))
            elif key is None and self.pending:
                # FIFO matching when no key
                first_key = min(self.pending.keys())
                req, req_step = self.pending.pop(first_key)
                latency = event.timestamp - req.timestamp
                self.matched.append((req, event, latency))

        self.step += 1

    @property
    def verdict(self):
        if self.timed_out:
            return Verdict.FALSE
        if self.pending:
            return Verdict.UNKNOWN
        return Verdict.TRUE

    def mean_latency(self):
        if not self.matched:
            return None
        return sum(lat for _, _, lat in self.matched) / len(self.matched)

    def max_latency(self):
        if not self.matched:
            return None
        return max(lat for _, _, lat in self.matched)


# ============================================================
# Convenience Functions
# ============================================================

def monitor_trace(formula, events, monitor_type='auto'):
    """Monitor a complete trace against a formula.

    Returns: (final_verdict, history)
    """
    if isinstance(formula, str):
        formula = parse_formula(formula)

    # Auto-detect: use past-time if formula is purely past-time
    if monitor_type == 'auto':
        if _is_past_time(formula):
            monitor_type = 'past'
        else:
            monitor_type = 'future'

    if monitor_type == 'past':
        mon = PastTimeMonitor(formula)
    else:
        mon = FutureTimeMonitor(formula)

    for event in events:
        if isinstance(event, str):
            event = Event(event)
        mon.process(event)

    if isinstance(mon, FutureTimeMonitor):
        final = mon.finalize()
    else:
        final = mon.history[-1][1] if mon.history else Verdict.UNKNOWN

    return final, mon.history


def _is_past_time(f):
    """Check if formula only uses past-time operators."""
    if isinstance(f, (TrueF, FalseF, Atom)):
        return True
    if isinstance(f, (Not,)):
        return _is_past_time(f.sub)
    if isinstance(f, (And, Or, Implies)):
        return _is_past_time(f.left) and _is_past_time(f.right)
    if isinstance(f, (Previous, Once, Historically)):
        return _is_past_time(f.sub)
    if isinstance(f, Since):
        return _is_past_time(f.left) and _is_past_time(f.right)
    # Future-time operators
    return False


def check_safety(invariant, events):
    """Check a safety property over a trace.

    Returns: (safe: bool, violation_step: int or None, violation_event: Event or None)
    """
    mon = SafetyMonitor(invariant)
    for event in events:
        if isinstance(event, str):
            event = Event(event)
        if not mon.process(event):
            return False, mon.violation_step, mon.violation_event
    return True, None, None


def check_response_pattern(events, request_name, response_name,
                           match_key=None, deadline=None):
    """Check request-response pattern over a trace.

    Returns: dict with matched_count, timeout_count, pending_count, mean_latency
    """
    mon = ResponsePatternMonitor(request_name, response_name, match_key, deadline)
    for event in events:
        if isinstance(event, str):
            event = Event(event)
        mon.process(event)
    return {
        'matched': len(mon.matched),
        'timed_out': len(mon.timed_out),
        'pending': len(mon.pending),
        'mean_latency': mon.mean_latency(),
        'max_latency': mon.max_latency(),
        'verdict': mon.verdict,
    }
