"""V098: IDE Framework (Interprocedural Distributive Environment)

Extends V096 IFDS to IDE: facts carry values (environments), not just sets.
Enables copy-constant propagation, linear constant propagation, and other
value-carrying interprocedural analyses.

Key idea (Sagiv, Reps, Horwitz 1996):
- IFDS tracks which facts are reachable (boolean: in set or not)
- IDE tracks which facts hold AND what value they carry
- Edge functions: micro-functions (D -> L) that transform lattice values
- Micro-function composition replaces set union

Features:
- IDE tabulation algorithm with micro-function composition
- Copy-constant propagation (assignments propagate constants)
- Linear constant propagation (y = a*x + b tracked precisely)
- Lattice framework: TOP/BOT/concrete values with meet
- Function summaries as composed micro-functions
- Context-sensitive via IFDS-style call/return matching
- C10 source-level API via parser
"""

import sys
import os
from dataclasses import dataclass, field
from typing import (Dict, Set, List, Tuple, Optional, FrozenSet,
                    Callable, Any, Union)
from enum import Enum, auto
from collections import defaultdict, deque
from copy import deepcopy

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser


# ---------------------------------------------------------------------------
# Value lattice
# ---------------------------------------------------------------------------

class LatticeValue:
    """Abstract lattice value for IDE.

    Lattice: TOP > concrete values > BOT
    - TOP: unknown/any value
    - BOT: unreachable/no value
    - Concrete: specific integer or constant
    """
    pass


@dataclass(frozen=True)
class Top(LatticeValue):
    """Top of lattice (unknown value)."""
    def __repr__(self):
        return "TOP"


@dataclass(frozen=True)
class Bot(LatticeValue):
    """Bottom of lattice (unreachable)."""
    def __repr__(self):
        return "BOT"


@dataclass(frozen=True)
class Const(LatticeValue):
    """Concrete constant value."""
    value: int

    def __repr__(self):
        return f"Const({self.value})"


TOP = Top()
BOT = Bot()


def lattice_meet(a: LatticeValue, b: LatticeValue) -> LatticeValue:
    """Meet (greatest lower bound) of two lattice values."""
    if isinstance(a, Bot) or isinstance(b, Bot):
        return BOT
    if isinstance(a, Top):
        return b
    if isinstance(b, Top):
        return a
    # Both are Const
    if a == b:
        return a
    return BOT  # Different constants -> bottom (NAC)


def lattice_join(a: LatticeValue, b: LatticeValue) -> LatticeValue:
    """Join (least upper bound) of two lattice values."""
    if isinstance(a, Top) or isinstance(b, Top):
        return TOP
    if isinstance(a, Bot):
        return b
    if isinstance(b, Bot):
        return a
    if a == b:
        return a
    return TOP  # Different constants -> top


def lattice_leq(a: LatticeValue, b: LatticeValue) -> bool:
    """Check if a <= b in the lattice ordering (BOT <= anything <= TOP)."""
    if isinstance(a, Bot):
        return True
    if isinstance(b, Top):
        return True
    if isinstance(a, Top):
        return isinstance(b, Top)
    if isinstance(b, Bot):
        return isinstance(a, Bot)
    return a == b


# ---------------------------------------------------------------------------
# Micro-functions (edge functions in IDE)
# ---------------------------------------------------------------------------

class MicroFunction:
    """Abstract base for micro-functions mapping LatticeValue -> LatticeValue.

    IDE associates a micro-function with each edge in the exploded supergraph.
    The composition of micro-functions along a path gives the value transformation.

    Key property: micro-functions must be distributive (f(meet(a,b)) = meet(f(a),f(b))).
    """

    def apply(self, val: LatticeValue) -> LatticeValue:
        raise NotImplementedError

    def compose(self, inner: 'MicroFunction') -> 'MicroFunction':
        """Compose: self(inner(x)). Returns self . inner."""
        return ComposedFunction(self, inner)

    def meet(self, other: 'MicroFunction') -> 'MicroFunction':
        """Pointwise meet of two micro-functions."""
        return MeetFunction(self, other)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self).__name__)


class IdFunction(MicroFunction):
    """Identity function: f(x) = x."""

    def apply(self, val: LatticeValue) -> LatticeValue:
        return val

    def compose(self, inner: MicroFunction) -> MicroFunction:
        return inner

    def meet(self, other: MicroFunction) -> MicroFunction:
        if isinstance(other, IdFunction):
            return self
        return other.meet(self)

    def __repr__(self):
        return "Id"

    def __eq__(self, other):
        return isinstance(other, IdFunction)

    def __hash__(self):
        return hash("IdFunction")


class ConstFunction(MicroFunction):
    """Constant function: f(x) = c for all x."""

    def __init__(self, value: LatticeValue):
        self.value = value

    def apply(self, val: LatticeValue) -> LatticeValue:
        return self.value

    def compose(self, inner: MicroFunction) -> MicroFunction:
        # const . anything = const
        return self

    def meet(self, other: MicroFunction) -> MicroFunction:
        if isinstance(other, ConstFunction):
            m = lattice_meet(self.value, other.value)
            return ConstFunction(m)
        if isinstance(other, IdFunction):
            # meet(const(c), id) needs to be computed pointwise
            return MeetFunction(self, other)
        if isinstance(other, TopFunction):
            return self
        return MeetFunction(self, other)

    def __repr__(self):
        return f"Const({self.value})"

    def __eq__(self, other):
        return isinstance(other, ConstFunction) and self.value == other.value

    def __hash__(self):
        return hash(("ConstFunction", self.value))


class TopFunction(MicroFunction):
    """Top function: f(x) = TOP for all x. Identity in meet."""

    def apply(self, val: LatticeValue) -> LatticeValue:
        return TOP

    def compose(self, inner: MicroFunction) -> MicroFunction:
        return self

    def meet(self, other: MicroFunction) -> MicroFunction:
        return other

    def __repr__(self):
        return "TopFn"

    def __eq__(self, other):
        return isinstance(other, TopFunction)

    def __hash__(self):
        return hash("TopFunction")


class BotFunction(MicroFunction):
    """Bottom function: f(x) = BOT for all x. Annihilator in meet."""

    def apply(self, val: LatticeValue) -> LatticeValue:
        return BOT

    def compose(self, inner: MicroFunction) -> MicroFunction:
        return self

    def meet(self, other: MicroFunction) -> MicroFunction:
        return self

    def __repr__(self):
        return "BotFn"

    def __eq__(self, other):
        return isinstance(other, BotFunction)

    def __hash__(self):
        return hash("BotFunction")


class LinearFunction(MicroFunction):
    """Linear function: f(x) = a * x + b.

    For constant propagation:
    - f(Const(c)) = Const(a * c + b)
    - f(TOP) = TOP (if a != 0), Const(b) (if a == 0)
    - f(BOT) = BOT
    """

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b

    def apply(self, val: LatticeValue) -> LatticeValue:
        if isinstance(val, Bot):
            return BOT
        if isinstance(val, Top):
            if self.a == 0:
                return Const(self.b)
            return TOP
        # Const
        return Const(self.a * val.value + self.b)

    def compose(self, inner: MicroFunction) -> MicroFunction:
        """(a*x + b) . inner(x)"""
        if isinstance(inner, LinearFunction):
            # a*(a2*x + b2) + b = (a*a2)*x + (a*b2 + b)
            return LinearFunction(self.a * inner.a, self.a * inner.b + self.b)
        if isinstance(inner, ConstFunction):
            return ConstFunction(self.apply(inner.value))
        if isinstance(inner, IdFunction):
            return self
        if isinstance(inner, TopFunction):
            return TopFunction()
        if isinstance(inner, BotFunction):
            return BotFunction()
        return ComposedFunction(self, inner)

    def meet(self, other: MicroFunction) -> MicroFunction:
        if isinstance(other, LinearFunction):
            if self.a == other.a and self.b == other.b:
                return self
            # Different linear functions -> pointwise meet
            return MeetFunction(self, other)
        if isinstance(other, TopFunction):
            return self
        if isinstance(other, BotFunction):
            return other
        return MeetFunction(self, other)

    def __repr__(self):
        if self.a == 1 and self.b == 0:
            return "Id"
        if self.a == 0:
            return f"Const({self.b})"
        if self.b == 0:
            return f"Lin({self.a}*x)"
        return f"Lin({self.a}*x+{self.b})"

    def __eq__(self, other):
        if isinstance(other, LinearFunction):
            return self.a == other.a and self.b == other.b
        if isinstance(other, IdFunction) and self.a == 1 and self.b == 0:
            return True
        if isinstance(other, ConstFunction) and self.a == 0:
            return Const(self.b) == other.value
        return False

    def __hash__(self):
        return hash(("LinearFunction", self.a, self.b))


class ComposedFunction(MicroFunction):
    """Composition of two micro-functions: outer(inner(x))."""

    def __init__(self, outer: MicroFunction, inner: MicroFunction):
        self.outer = outer
        self.inner = inner

    def apply(self, val: LatticeValue) -> LatticeValue:
        return self.outer.apply(self.inner.apply(val))

    def __repr__(self):
        return f"({self.outer} . {self.inner})"

    def __eq__(self, other):
        return (isinstance(other, ComposedFunction) and
                self.outer == other.outer and self.inner == other.inner)

    def __hash__(self):
        return hash(("ComposedFunction", self.outer, self.inner))


class MeetFunction(MicroFunction):
    """Pointwise meet of two micro-functions."""

    def __init__(self, f1: MicroFunction, f2: MicroFunction):
        self.f1 = f1
        self.f2 = f2

    def apply(self, val: LatticeValue) -> LatticeValue:
        return lattice_meet(self.f1.apply(val), self.f2.apply(val))

    def __repr__(self):
        return f"meet({self.f1}, {self.f2})"

    def __eq__(self, other):
        if not isinstance(other, MeetFunction):
            return False
        return ((self.f1 == other.f1 and self.f2 == other.f2) or
                (self.f1 == other.f2 and self.f2 == other.f1))

    def __hash__(self):
        return hash(("MeetFunction", frozenset([hash(self.f1), hash(self.f2)])))


# Singleton
ID_FN = IdFunction()
TOP_FN = TopFunction()
BOT_FN = BotFunction()


# ---------------------------------------------------------------------------
# IDE problem specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    """A dataflow fact (variable name or ZERO for the special lambda fact)."""
    name: str

    def __repr__(self):
        return self.name if self.name != "__ZERO__" else "ZERO"


ZERO = Fact("__ZERO__")


@dataclass
class IDEProblem:
    """Specification of an IDE problem.

    Users subclass this or provide callbacks to define:
    - Normal flow functions (intraprocedural edges)
    - Call flow functions (caller -> callee entry)
    - Return flow functions (callee exit -> caller return site)
    - Call-to-return flow functions (across call, for non-callee effects)
    """
    # ICFG structure
    functions: Dict[str, Dict] = field(default_factory=dict)
    # function -> {entry, exit, params, points, stmts}
    edges: List[Tuple[str, str, str, str]] = field(default_factory=list)
    # (source, target, edge_type, callee)
    entry_function: str = "main"
    all_facts: Set[Fact] = field(default_factory=set)  # domain D

    # Flow function callbacks
    normal_flow: Optional[Callable] = None
    # (src_point, tgt_point, fact) -> Dict[Fact, MicroFunction]
    call_flow: Optional[Callable] = None
    # (call_point, callee_entry, fact, callee) -> Dict[Fact, MicroFunction]
    return_flow: Optional[Callable] = None
    # (callee_exit, return_site, fact, callee, call_point) -> Dict[Fact, MicroFunction]
    call_to_return_flow: Optional[Callable] = None
    # (call_point, return_site, fact, callee) -> Dict[Fact, MicroFunction]


@dataclass
class IDEResult:
    """Result of an IDE analysis."""
    values: Dict[str, Dict[Fact, LatticeValue]]
    # point_id -> fact -> lattice value
    summaries: Dict[str, Dict[Tuple[Fact, Fact], MicroFunction]]
    # function -> (d1, d2) -> composed micro-function
    reachable_facts: Dict[str, Set[Fact]]
    # point_id -> set of reachable facts (IFDS part)
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IDE tabulation algorithm (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------

class IDESolver:
    """IDE tabulation algorithm solver.

    Phase 1: Forward tabulation (like IFDS) computing jump functions
    Phase 2: Value computation using composed edge functions
    """

    def __init__(self, problem: IDEProblem):
        self.problem = problem
        self.jump_fn: Dict[Tuple[str, Fact, str, Fact], MicroFunction] = {}
        # (sp, d1, n, d2) -> micro-function
        # sp = start point (procedure entry), d1 = entry fact
        # n = current point, d2 = current fact
        self.summary_fn: Dict[Tuple[str, Fact, Fact], MicroFunction] = {}
        # (callee, d1, d2) -> summary micro-function from callee entry to exit
        self.path_edges: Set[Tuple[str, Fact, str, Fact]] = set()
        self.worklist: deque = deque()
        self.end_summary: Dict[Tuple[str, Fact, str, Fact], MicroFunction] = {}
        # incoming: for each callee entry, which callers called it with which facts
        self.incoming: Dict[Tuple[str, Fact], Set[Tuple[str, Fact]]] = defaultdict(set)
        # callee_entry, d3 -> set of (call_node, d1)

        # Build adjacency for quick lookup
        self._succ: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        # point -> [(target, edge_type, callee)]
        self._pred: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for (src, tgt, etype, callee) in problem.edges:
            self._succ[src].append((tgt, etype, callee))
            self._pred[tgt].append((src, etype, callee))

        # Function info
        self._entry_of: Dict[str, str] = {}  # function -> entry point
        self._exit_of: Dict[str, str] = {}   # function -> exit point
        self._func_of: Dict[str, str] = {}   # point -> function
        self._return_sites: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # call_node -> [(return_site, callee)]

        for fn_name, fn_info in problem.functions.items():
            self._entry_of[fn_name] = fn_info['entry']
            self._exit_of[fn_name] = fn_info['exit']
            for pt in fn_info.get('points', []):
                self._func_of[pt] = fn_name

        # Build return site map from call-to-return edges
        for (src, tgt, etype, callee) in problem.edges:
            if etype == 'call_to_return':
                self._return_sites[src].append((tgt, callee))

        self._stats = {'path_edges': 0, 'summary_apps': 0, 'worklist_pops': 0}

    def _propagate(self, sp: str, d1: Fact, n: str, d2: Fact, f: MicroFunction):
        """Propagate a path edge with its edge function."""
        key = (sp, d1, n, d2)
        if key in self.jump_fn:
            old_f = self.jump_fn[key]
            new_f = old_f.meet(f)
            if new_f == old_f:
                return  # No change
            self.jump_fn[key] = new_f
        else:
            self.jump_fn[key] = f
            self._stats['path_edges'] += 1

        if key not in self.path_edges:
            self.path_edges.add(key)
            self.worklist.append(key)

    def _get_normal_flow(self, src: str, tgt: str, fact: Fact) -> Dict[Fact, MicroFunction]:
        """Get normal (intraprocedural) flow function results."""
        if self.problem.normal_flow:
            return self.problem.normal_flow(src, tgt, fact)
        # Default: identity for all existing facts
        return {fact: ID_FN}

    def _get_call_flow(self, call_pt: str, callee_entry: str, fact: Fact,
                       callee: str) -> Dict[Fact, MicroFunction]:
        """Get call flow function results."""
        if self.problem.call_flow:
            return self.problem.call_flow(call_pt, callee_entry, fact, callee)
        return {}

    def _get_return_flow(self, callee_exit: str, return_site: str, fact: Fact,
                         callee: str, call_pt: str) -> Dict[Fact, MicroFunction]:
        """Get return flow function results."""
        if self.problem.return_flow:
            return self.problem.return_flow(callee_exit, return_site, fact, callee, call_pt)
        return {}

    def _get_call_to_return_flow(self, call_pt: str, return_site: str, fact: Fact,
                                  callee: str) -> Dict[Fact, MicroFunction]:
        """Get call-to-return flow function results."""
        if self.problem.call_to_return_flow:
            return self.problem.call_to_return_flow(call_pt, return_site, fact, callee)
        return {fact: ID_FN}

    def solve(self) -> IDEResult:
        """Run the IDE tabulation algorithm (Phase 1 + Phase 2)."""
        # Phase 1: Forward tabulation
        self._phase1()
        # Phase 2: Value computation
        values = self._phase2()
        # Build result
        reachable = defaultdict(set)
        for (sp, d1, n, d2) in self.path_edges:
            reachable[n].add(d2)

        summaries = {}
        for (callee, d1, d2), f in self.end_summary.items():
            if callee not in summaries:
                summaries[callee] = {}
            summaries[callee][(d1, d2)] = f

        self._stats['total_jump_fns'] = len(self.jump_fn)
        self._stats['total_summaries'] = len(self.end_summary)

        return IDEResult(
            values=dict(values),
            summaries=summaries,
            reachable_facts=dict(reachable),
            stats=self._stats
        )

    def _phase1(self):
        """Phase 1: Forward tabulation computing jump functions.

        Similar to IFDS but edges carry micro-functions instead of just
        reachability. Jump functions compose along paths.
        """
        # Seed: entry of main function with ZERO fact
        entry = self._entry_of.get(self.problem.entry_function)
        if not entry:
            return

        self._propagate(entry, ZERO, entry, ZERO, ID_FN)

        while self.worklist:
            (sp, d1, n, d2) = self.worklist.popleft()
            self._stats['worklist_pops'] += 1
            f_jump = self.jump_fn.get((sp, d1, n, d2), TOP_FN)

            fn = self._func_of.get(n, "")
            exit_pt = self._exit_of.get(fn, "")

            for (tgt, etype, callee) in self._succ.get(n, []):
                if etype == 'intra' or etype == 'normal':
                    # Normal intraprocedural edge
                    flow = self._get_normal_flow(n, tgt, d2)
                    for d3, f_edge in flow.items():
                        # Compose: jump to n with d2, then edge to tgt with d3
                        f_new = f_edge.compose(f_jump)
                        self._propagate(sp, d1, tgt, d3, f_new)

                elif etype == 'call':
                    # Call edge: n is call site, tgt is callee entry
                    flow = self._get_call_flow(n, tgt, d2, callee)
                    for d3, f_edge in flow.items():
                        # Start new path from callee entry
                        callee_entry = tgt
                        self._propagate(callee_entry, d3, callee_entry, d3, ID_FN)
                        # Record incoming
                        self.incoming[(callee_entry, d3)].add((n, d2))

                        # Apply existing summaries
                        callee_exit = self._exit_of.get(callee, "")
                        if callee_exit:
                            for d4 in self.problem.all_facts | {ZERO}:
                                sum_key = (callee, d3, d4)
                                if sum_key in self.end_summary:
                                    f_sum = self.end_summary[sum_key]
                                    # Process return
                                    for (ret_site, _) in self._return_sites.get(n, []):
                                        ret_flow = self._get_return_flow(
                                            callee_exit, ret_site, d4, callee, n)
                                        for d5, f_ret in ret_flow.items():
                                            f_composed = f_ret.compose(f_sum).compose(f_edge).compose(f_jump)
                                            self._propagate(sp, d1, ret_site, d5, f_composed)
                                            self._stats['summary_apps'] += 1

                elif etype == 'call_to_return':
                    # Call-to-return edge (for local variables not affected by callee)
                    flow = self._get_call_to_return_flow(n, tgt, d2, callee)
                    for d3, f_edge in flow.items():
                        f_new = f_edge.compose(f_jump)
                        self._propagate(sp, d1, tgt, d3, f_new)

            # Check if n is an exit point -- update end summaries
            if n == exit_pt and fn:
                # This is a procedure exit: update end summary
                # sp should be the entry of this function
                sum_key = (fn, d1, d2)
                old_f = self.end_summary.get(sum_key)
                if old_f is None:
                    self.end_summary[sum_key] = f_jump
                else:
                    new_f = old_f.meet(f_jump)
                    if new_f == old_f:
                        continue
                    self.end_summary[sum_key] = new_f

                # Apply summary to all callers
                callee_entry = self._entry_of.get(fn, "")
                for (call_node, d_call) in self.incoming.get((callee_entry, d1), set()):
                    caller_fn = self._func_of.get(call_node, "")
                    caller_entry = self._entry_of.get(caller_fn, "")

                    # Find the call edge function
                    f_call_edge = ID_FN
                    for (ce_tgt, ce_etype, ce_callee) in self._succ.get(call_node, []):
                        if ce_etype == 'call' and ce_callee == fn:
                            cf = self._get_call_flow(call_node, ce_tgt, d_call, fn)
                            if d1 in cf:
                                f_call_edge = cf[d1]
                            break

                    # Get return flow
                    for (ret_site, ret_callee) in self._return_sites.get(call_node, []):
                        if ret_callee == fn:
                            ret_flow = self._get_return_flow(n, ret_site, d2, fn, call_node)
                            for d5, f_ret in ret_flow.items():
                                # Find caller's jump function to call_node
                                for d0 in self.problem.all_facts | {ZERO}:
                                    caller_jf_key = (caller_entry, d0, call_node, d_call)
                                    if caller_jf_key in self.jump_fn:
                                        f_caller = self.jump_fn[caller_jf_key]
                                        f_composed = f_ret.compose(f_jump).compose(f_call_edge).compose(f_caller)
                                        self._propagate(caller_entry, d0, ret_site, d5, f_composed)
                                        self._stats['summary_apps'] += 1

    def _phase2(self) -> Dict[str, Dict[Fact, LatticeValue]]:
        """Phase 2: Value computation.

        Uses jump functions from Phase 1 to compute actual lattice values
        at each program point.
        """
        values: Dict[str, Dict[Fact, LatticeValue]] = defaultdict(dict)

        # Seed entry point
        entry = self._entry_of.get(self.problem.entry_function)
        if not entry:
            return values

        # For each reachable (n, d), compute value by applying jump function
        # to the initial value at the procedure entry.
        # Initial value: ZERO fact has value BOT (it's just a generator),
        # all other facts at entry start with TOP.

        for (sp, d1, n, d2), f in self.jump_fn.items():
            # The value at (n, d2) = f(initial_value_at(sp, d1))
            if d1 == ZERO:
                init_val = BOT  # ZERO generates, doesn't carry value
                result_val = f.apply(init_val)
                # For ZERO-generated facts, the function IS the value
                # (e.g., ConstFunction(5) applied to BOT = Const(5) is wrong,
                #  it should give Const(5). But ConstFunction(5).apply(BOT) = Const(5))
                # Actually: for ZERO-seeded paths, the micro-function gives the value
                # Let's use a special convention: ZERO fact carries a dummy value
                # The const function handles it correctly.
                if isinstance(result_val, Bot):
                    # ZERO->d2 path: value comes from the micro-function applied to TOP
                    # (since ZERO is the universal generator)
                    result_val = f.apply(TOP)
            else:
                # For non-ZERO starts, the value depends on what d1 holds at sp
                init_val = values.get(sp, {}).get(d1, TOP)
                result_val = f.apply(init_val)

            if d2 == ZERO:
                continue  # Don't store values for ZERO fact

            if n in values and d2 in values[n]:
                values[n][d2] = lattice_meet(values[n][d2], result_val)
            else:
                values[n][d2] = result_val

        return values


# ---------------------------------------------------------------------------
# C10 source -> IDE problem construction
# ---------------------------------------------------------------------------

def _extract_vars_from_expr(expr) -> Set[str]:
    """Extract variable names used in an expression."""
    cls = type(expr).__name__
    if cls == 'Var':
        return {expr.name}
    if cls == 'BinOp':
        return _extract_vars_from_expr(expr.left) | _extract_vars_from_expr(expr.right)
    if cls == 'UnaryOp':
        return _extract_vars_from_expr(expr.operand)
    if cls == 'CallExpr':
        result = set()
        for arg in expr.args:
            result |= _extract_vars_from_expr(arg)
        return result
    if cls == 'IntLit':
        return set()
    return set()


def _eval_const_expr(expr, known_consts: Dict[str, int]) -> Optional[int]:
    """Try to evaluate an expression to a constant given known constants."""
    cls = type(expr).__name__
    if cls == 'IntLit':
        return expr.value
    if cls == 'Var':
        return known_consts.get(expr.name)
    if cls == 'BinOp':
        left = _eval_const_expr(expr.left, known_consts)
        right = _eval_const_expr(expr.right, known_consts)
        if left is not None and right is not None:
            op = expr.op
            if op == '+': return left + right
            if op == '-': return left - right
            if op == '*': return left * right
            if op == '/' and right != 0: return left // right
            if op == '%' and right != 0: return left % right
        return None
    if cls == 'UnaryOp':
        val = _eval_const_expr(expr.operand, known_consts)
        if val is not None:
            if expr.op == '-': return -val
        return None
    return None


def _extract_linear_form(expr, target_var: str) -> Optional[Tuple[int, int]]:
    """Try to express expr as a*target_var + b. Returns (a, b) or None.

    For copy-constant propagation:
    - IntLit(c) -> (0, c)
    - Var(target_var) -> (1, 0)
    - target_var + c -> (1, c)
    - c * target_var -> (c, 0)
    - a * target_var + b -> (a, b)
    """
    cls = type(expr).__name__
    if cls == 'IntLit':
        return (0, expr.value)
    if cls == 'Var':
        if expr.name == target_var:
            return (1, 0)
        return None  # Different variable - not expressible as linear fn of target
    if cls == 'BinOp':
        op = expr.op
        # Check for target_var +/- const
        left_cls = type(expr.left).__name__
        right_cls = type(expr.right).__name__

        if op == '+':
            if left_cls == 'Var' and expr.left.name == target_var and right_cls == 'IntLit':
                return (1, expr.right.value)
            if right_cls == 'Var' and expr.right.name == target_var and left_cls == 'IntLit':
                return (1, expr.left.value)
        if op == '-':
            if left_cls == 'Var' and expr.left.name == target_var and right_cls == 'IntLit':
                return (1, -expr.right.value)
        if op == '*':
            if left_cls == 'Var' and expr.left.name == target_var and right_cls == 'IntLit':
                return (expr.right.value, 0)
            if right_cls == 'Var' and expr.right.name == target_var and left_cls == 'IntLit':
                return (expr.left.value, 0)
        # Recursive: (a1*x + b1) op (a2*x + b2)
        left_lin = _extract_linear_form(expr.left, target_var)
        right_lin = _extract_linear_form(expr.right, target_var)
        if left_lin and right_lin:
            a1, b1 = left_lin
            a2, b2 = right_lin
            if op == '+':
                return (a1 + a2, b1 + b2)
            if op == '-':
                return (a1 - a2, b1 - b2)
            if op == '*' and a1 == 0:
                return (b1 * a2, b1 * b2)
            if op == '*' and a2 == 0:
                return (a1 * b2, b1 * b2)
        return None
    if cls == 'UnaryOp':
        if expr.op == '-':
            inner = _extract_linear_form(expr.operand, target_var)
            if inner:
                return (-inner[0], -inner[1])
        return None
    return None


def _get_call_info(stmt) -> Optional[Tuple[str, list]]:
    """Extract function call info from a statement."""
    cls = type(stmt).__name__
    if cls == 'LetDecl' and type(stmt.value).__name__ == 'CallExpr':
        callee = stmt.value.callee
        if isinstance(callee, str):
            return (callee, stmt.value.args)
    if cls == 'Assign' and type(stmt.value).__name__ == 'CallExpr':
        callee = stmt.value.callee
        if isinstance(callee, str):
            return (callee, stmt.value.args)
    if cls == 'ReturnStmt' and stmt.value and type(stmt.value).__name__ == 'CallExpr':
        callee = stmt.value.callee
        if isinstance(callee, str):
            return (callee, stmt.value.args)
    if cls == 'CallExpr':
        callee = stmt.callee
        if isinstance(callee, str):
            return (callee, stmt.args)
    return None


def build_ide_problem(source: str, analysis: str = "copy_const") -> IDEProblem:
    """Build an IDE problem from C10 source code.

    Args:
        source: C10 source code
        analysis: Type of analysis:
            - "copy_const": Copy-constant propagation
            - "linear_const": Linear constant propagation
    """
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()

    problem = IDEProblem()
    functions = {}
    main_stmts = []

    # Collect function declarations and main statements
    for stmt in program.stmts:
        cls = type(stmt).__name__
        if cls == 'FnDecl':
            functions[stmt.name] = stmt
        else:
            main_stmts.append(stmt)

    # Build ICFG structure
    all_vars = set()

    def _build_fn(fn_name, params, stmts):
        entry = f"{fn_name}.entry"
        exit_pt = f"{fn_name}.exit"
        problem.functions[fn_name] = {
            'entry': entry,
            'exit': exit_pt,
            'params': params,
            'points': [entry, exit_pt],
            'stmts': {},  # label -> stmt
        }

        if not stmts:
            problem.edges.append((entry, exit_pt, 'normal', ''))
            return

        # Create statement nodes
        labels = []
        for i, stmt in enumerate(stmts):
            label = f"{fn_name}.s{i}"
            labels.append(label)
            problem.functions[fn_name]['points'].append(label)
            problem.functions[fn_name]['stmts'][label] = stmt

            # Collect variables
            cls = type(stmt).__name__
            if cls == 'LetDecl':
                all_vars.add(stmt.name)
            elif cls == 'Assign':
                all_vars.add(stmt.name)

            # Collect vars from expressions
            for v in _extract_vars_from_expr_stmt(stmt):
                all_vars.add(v)

        # Entry -> first statement
        problem.edges.append((entry, labels[0], 'normal', ''))

        # Sequential edges between statements
        for i in range(len(labels) - 1):
            stmt = stmts[i]
            call_info = _get_call_info(stmt)
            cls = type(stmt).__name__

            if call_info and call_info[0] in functions:
                callee_name = call_info[0]
                callee_entry = f"{callee_name}.entry"
                ret_site = labels[i + 1]
                # Call edge
                problem.edges.append((labels[i], callee_entry, 'call', callee_name))
                # Call-to-return edge
                problem.edges.append((labels[i], ret_site, 'call_to_return', callee_name))
                # Return edge (from callee exit)
                callee_exit = f"{callee_name}.exit"
                problem.edges.append((callee_exit, ret_site, 'return', callee_name))
            elif cls == 'ReturnStmt':
                # Check if return value is a call
                if call_info and call_info[0] in functions:
                    callee_name = call_info[0]
                    callee_entry = f"{callee_name}.entry"
                    problem.edges.append((labels[i], callee_entry, 'call', callee_name))
                    problem.edges.append((labels[i], exit_pt, 'call_to_return', callee_name))
                    callee_exit = f"{callee_name}.exit"
                    problem.edges.append((callee_exit, exit_pt, 'return', callee_name))
                else:
                    problem.edges.append((labels[i], exit_pt, 'normal', ''))
            else:
                problem.edges.append((labels[i], labels[i + 1], 'normal', ''))

        # Last statement -> exit (if not already handled)
        last_stmt = stmts[-1]
        last_cls = type(last_stmt).__name__
        if last_cls != 'ReturnStmt':
            call_info = _get_call_info(last_stmt)
            if call_info and call_info[0] in functions:
                callee_name = call_info[0]
                callee_entry = f"{callee_name}.entry"
                problem.edges.append((labels[-1], callee_entry, 'call', callee_name))
                problem.edges.append((labels[-1], exit_pt, 'call_to_return', callee_name))
                callee_exit = f"{callee_name}.exit"
                problem.edges.append((callee_exit, exit_pt, 'return', callee_name))
            else:
                problem.edges.append((labels[-1], exit_pt, 'normal', ''))

    # Build all functions
    for fn_name, fn_decl in functions.items():
        params = fn_decl.params if hasattr(fn_decl, 'params') else []
        body_stmts = fn_decl.body.stmts if hasattr(fn_decl.body, 'stmts') else []
        _build_fn(fn_name, params, body_stmts)

    if main_stmts:
        _build_fn("main", [], main_stmts)
        problem.entry_function = "main"
    elif functions:
        problem.entry_function = next(iter(functions))

    # Set up facts (one per variable + ZERO)
    problem.all_facts = {Fact(v) for v in all_vars}

    # Set up flow functions based on analysis type
    if analysis == "copy_const":
        _setup_copy_const_flow(problem, functions)
    elif analysis == "linear_const":
        _setup_linear_const_flow(problem, functions)
    else:
        _setup_copy_const_flow(problem, functions)

    return problem


def _extract_vars_from_expr_stmt(stmt) -> Set[str]:
    """Extract all variable names from a statement."""
    cls = type(stmt).__name__
    if cls == 'LetDecl':
        return _extract_vars_from_expr(stmt.value)
    if cls == 'Assign':
        return _extract_vars_from_expr(stmt.value)
    if cls == 'ReturnStmt' and stmt.value:
        return _extract_vars_from_expr(stmt.value)
    return set()


def _setup_copy_const_flow(problem: IDEProblem, functions: Dict):
    """Set up flow functions for copy-constant propagation.

    Copy-constant propagation tracks:
    - x = c (constant assignment): x gets Const(c)
    - x = y (copy): x gets whatever y has
    - x = expr: x gets TOP (unknown) unless expr is constant
    """

    def normal_flow(src: str, tgt: str, fact: Fact) -> Dict[Fact, MicroFunction]:
        # Find the statement at src
        for fn_info in problem.functions.values():
            if src in fn_info.get('stmts', {}):
                stmt = fn_info['stmts'][src]
                return _copy_const_normal(stmt, fact, problem.all_facts)
        # Entry/exit nodes: identity
        return {fact: ID_FN}

    def call_flow(call_pt: str, callee_entry: str, fact: Fact,
                  callee: str) -> Dict[Fact, MicroFunction]:
        if callee not in functions:
            return {}
        fn_decl = functions[callee]
        params = fn_decl.params if hasattr(fn_decl, 'params') else []

        # Find the call statement
        call_stmt = None
        for fn_info in problem.functions.values():
            if call_pt in fn_info.get('stmts', {}):
                call_stmt = fn_info['stmts'][call_pt]
                break

        if not call_stmt:
            return {}

        call_info = _get_call_info(call_stmt)
        if not call_info:
            return {}

        _, args = call_info
        result = {}

        if fact == ZERO:
            # Map ZERO to ZERO (seed the callee)
            result[ZERO] = ID_FN
            # Map arguments to parameters
            for i, param in enumerate(params):
                if i < len(args):
                    arg = args[i]
                    arg_cls = type(arg).__name__
                    if arg_cls == 'IntLit':
                        result[Fact(param)] = ConstFunction(Const(arg.value))
                    elif arg_cls == 'Var':
                        # Copy: param gets whatever the argument var has
                        # This is tricky: we need to map Fact(arg.name) -> Fact(param)
                        # But in IDE, the fact here is ZERO, so we generate param with TOP
                        result[Fact(param)] = TopFunction()
                    else:
                        result[Fact(param)] = TopFunction()
        else:
            # Non-ZERO fact: map argument variables to parameters
            for i, param in enumerate(params):
                if i < len(args):
                    arg = args[i]
                    if type(arg).__name__ == 'Var' and arg.name == fact.name:
                        # Copy: param = arg_var -> identity transfer
                        result[Fact(param)] = ID_FN

        return result

    def return_flow(callee_exit: str, return_site: str, fact: Fact,
                    callee: str, call_pt: str) -> Dict[Fact, MicroFunction]:
        if callee not in functions:
            return {}

        # Find the call statement to get the target variable
        call_stmt = None
        for fn_info in problem.functions.values():
            if call_pt in fn_info.get('stmts', {}):
                call_stmt = fn_info['stmts'][call_pt]
                break

        if not call_stmt:
            return {}

        result = {}
        call_cls = type(call_stmt).__name__

        # Map return value to target variable
        target_var = None
        if call_cls == 'LetDecl':
            target_var = call_stmt.name
        elif call_cls == 'Assign':
            target_var = call_stmt.name

        # Look at the callee's return statement
        fn_decl = functions[callee]
        body_stmts = fn_decl.body.stmts if hasattr(fn_decl.body, 'stmts') else []
        return_stmt = None
        for s in body_stmts:
            if type(s).__name__ == 'ReturnStmt':
                return_stmt = s
                break

        if fact == ZERO:
            result[ZERO] = ID_FN
            if target_var and return_stmt and return_stmt.value:
                ret_expr = return_stmt.value
                ret_cls = type(ret_expr).__name__
                if ret_cls == 'IntLit':
                    result[Fact(target_var)] = ConstFunction(Const(ret_expr.value))
                else:
                    result[Fact(target_var)] = TopFunction()
        else:
            # Non-ZERO: if return expression references the fact's variable
            if target_var and return_stmt and return_stmt.value:
                ret_expr = return_stmt.value
                if type(ret_expr).__name__ == 'Var' and ret_expr.name == fact.name:
                    result[Fact(target_var)] = ID_FN

        return result

    def call_to_return_flow(call_pt: str, return_site: str, fact: Fact,
                            callee: str) -> Dict[Fact, MicroFunction]:
        # Find the call statement
        call_stmt = None
        for fn_info in problem.functions.values():
            if call_pt in fn_info.get('stmts', {}):
                call_stmt = fn_info['stmts'][call_pt]
                break

        if not call_stmt:
            return {fact: ID_FN}

        # Kill the target variable (it will be set by return flow)
        target_var = None
        call_cls = type(call_stmt).__name__
        if call_cls == 'LetDecl':
            target_var = call_stmt.name
        elif call_cls == 'Assign':
            target_var = call_stmt.name

        if fact == ZERO:
            return {ZERO: ID_FN}

        if target_var and fact.name == target_var:
            return {}  # Kill: target will be set by return flow

        return {fact: ID_FN}  # Pass through

    problem.normal_flow = normal_flow
    problem.call_flow = call_flow
    problem.return_flow = return_flow
    problem.call_to_return_flow = call_to_return_flow


def _copy_const_normal(stmt, fact: Fact, all_facts: Set[Fact]) -> Dict[Fact, MicroFunction]:
    """Normal flow function for copy-constant propagation."""
    cls = type(stmt).__name__
    result = {}

    if cls in ('LetDecl', 'Assign'):
        var_name = stmt.name
        expr = stmt.value

        if fact == ZERO:
            result[ZERO] = ID_FN
            # Generate the assigned variable
            expr_cls = type(expr).__name__
            if expr_cls == 'IntLit':
                result[Fact(var_name)] = ConstFunction(Const(expr.value))
            elif expr_cls == 'Var':
                # Copy: target gets TOP from ZERO (actual value via non-ZERO path)
                result[Fact(var_name)] = TopFunction()
            else:
                # Try to evaluate constant expression
                const_val = _eval_const_expr(expr, {})
                if const_val is not None:
                    result[Fact(var_name)] = ConstFunction(Const(const_val))
                else:
                    result[Fact(var_name)] = TopFunction()
            # Pass through all other facts
            for f in all_facts:
                if f.name != var_name:
                    result[Fact(f.name)] = ID_FN
        else:
            if fact.name == var_name:
                # Variable being assigned: kill old value
                # If RHS references this var (x = x + 1), create linear fn
                expr_cls = type(expr).__name__
                if expr_cls == 'Var' and expr.name == var_name:
                    result[Fact(var_name)] = ID_FN  # x = x (identity)
                else:
                    # Check if expr is linear in fact.name
                    lin = _extract_linear_form(expr, fact.name)
                    if lin:
                        a, b = lin
                        if a == 1 and b == 0:
                            result[Fact(var_name)] = ID_FN
                        else:
                            result[Fact(var_name)] = LinearFunction(a, b)
                    # Otherwise, the old value is killed (no edge)
            else:
                # Different variable: pass through
                result[fact] = ID_FN
                # If RHS copies this variable
                expr_cls = type(expr).__name__
                if expr_cls == 'Var' and expr.name == fact.name:
                    result[Fact(var_name)] = ID_FN  # Copy: target = source

    elif cls == 'ReturnStmt':
        result[fact] = ID_FN

    else:
        # Default: identity
        result[fact] = ID_FN

    return result


def _setup_linear_const_flow(problem: IDEProblem, functions: Dict):
    """Set up flow functions for linear constant propagation.

    Extends copy-constant propagation to track linear transformations:
    - x = a * y + b tracks precisely as LinearFunction(a, b)
    """

    def normal_flow(src: str, tgt: str, fact: Fact) -> Dict[Fact, MicroFunction]:
        for fn_info in problem.functions.values():
            if src in fn_info.get('stmts', {}):
                stmt = fn_info['stmts'][src]
                return _linear_const_normal(stmt, fact, problem.all_facts)
        return {fact: ID_FN}

    def call_flow(call_pt: str, callee_entry: str, fact: Fact,
                  callee: str) -> Dict[Fact, MicroFunction]:
        if callee not in functions:
            return {}
        fn_decl = functions[callee]
        params = fn_decl.params if hasattr(fn_decl, 'params') else []

        call_stmt = None
        for fn_info in problem.functions.values():
            if call_pt in fn_info.get('stmts', {}):
                call_stmt = fn_info['stmts'][call_pt]
                break

        if not call_stmt:
            return {}

        call_info = _get_call_info(call_stmt)
        if not call_info:
            return {}

        _, args = call_info
        result = {}

        if fact == ZERO:
            result[ZERO] = ID_FN
            for i, param in enumerate(params):
                if i < len(args):
                    arg = args[i]
                    arg_cls = type(arg).__name__
                    if arg_cls == 'IntLit':
                        result[Fact(param)] = ConstFunction(Const(arg.value))
                    else:
                        result[Fact(param)] = TopFunction()
        else:
            for i, param in enumerate(params):
                if i < len(args):
                    arg = args[i]
                    if type(arg).__name__ == 'Var' and arg.name == fact.name:
                        result[Fact(param)] = ID_FN
                    elif type(arg).__name__ == 'BinOp':
                        lin = _extract_linear_form(arg, fact.name)
                        if lin:
                            a, b = lin
                            result[Fact(param)] = LinearFunction(a, b)

        return result

    def return_flow(callee_exit: str, return_site: str, fact: Fact,
                    callee: str, call_pt: str) -> Dict[Fact, MicroFunction]:
        if callee not in functions:
            return {}

        call_stmt = None
        for fn_info in problem.functions.values():
            if call_pt in fn_info.get('stmts', {}):
                call_stmt = fn_info['stmts'][call_pt]
                break

        if not call_stmt:
            return {}

        target_var = None
        call_cls = type(call_stmt).__name__
        if call_cls == 'LetDecl':
            target_var = call_stmt.name
        elif call_cls == 'Assign':
            target_var = call_stmt.name

        fn_decl = functions[callee]
        body_stmts = fn_decl.body.stmts if hasattr(fn_decl.body, 'stmts') else []
        return_stmt = None
        for s in body_stmts:
            if type(s).__name__ == 'ReturnStmt':
                return_stmt = s
                break

        result = {}
        if fact == ZERO:
            result[ZERO] = ID_FN
            if target_var and return_stmt and return_stmt.value:
                ret_expr = return_stmt.value
                ret_cls = type(ret_expr).__name__
                if ret_cls == 'IntLit':
                    result[Fact(target_var)] = ConstFunction(Const(ret_expr.value))
                else:
                    result[Fact(target_var)] = TopFunction()
        else:
            if target_var and return_stmt and return_stmt.value:
                ret_expr = return_stmt.value
                if type(ret_expr).__name__ == 'Var' and ret_expr.name == fact.name:
                    result[Fact(target_var)] = ID_FN
                elif type(ret_expr).__name__ == 'BinOp':
                    lin = _extract_linear_form(ret_expr, fact.name)
                    if lin:
                        a, b = lin
                        result[Fact(target_var)] = LinearFunction(a, b)

        return result

    def call_to_return_flow(call_pt: str, return_site: str, fact: Fact,
                            callee: str) -> Dict[Fact, MicroFunction]:
        call_stmt = None
        for fn_info in problem.functions.values():
            if call_pt in fn_info.get('stmts', {}):
                call_stmt = fn_info['stmts'][call_pt]
                break

        if not call_stmt:
            return {fact: ID_FN}

        target_var = None
        call_cls = type(call_stmt).__name__
        if call_cls == 'LetDecl':
            target_var = call_stmt.name
        elif call_cls == 'Assign':
            target_var = call_stmt.name

        if fact == ZERO:
            return {ZERO: ID_FN}
        if target_var and fact.name == target_var:
            return {}
        return {fact: ID_FN}

    problem.normal_flow = normal_flow
    problem.call_flow = call_flow
    problem.return_flow = return_flow
    problem.call_to_return_flow = call_to_return_flow


def _linear_const_normal(stmt, fact: Fact, all_facts: Set[Fact]) -> Dict[Fact, MicroFunction]:
    """Normal flow function for linear constant propagation."""
    cls = type(stmt).__name__
    result = {}

    if cls in ('LetDecl', 'Assign'):
        var_name = stmt.name
        expr = stmt.value

        if fact == ZERO:
            result[ZERO] = ID_FN
            expr_cls = type(expr).__name__
            if expr_cls == 'IntLit':
                result[Fact(var_name)] = ConstFunction(Const(expr.value))
            elif expr_cls == 'Var':
                result[Fact(var_name)] = TopFunction()
            else:
                const_val = _eval_const_expr(expr, {})
                if const_val is not None:
                    result[Fact(var_name)] = ConstFunction(Const(const_val))
                else:
                    result[Fact(var_name)] = TopFunction()
            for f in all_facts:
                if f.name != var_name:
                    result[Fact(f.name)] = ID_FN
        else:
            if fact.name == var_name:
                # Check for linear self-reference
                lin = _extract_linear_form(expr, var_name)
                if lin:
                    a, b = lin
                    result[Fact(var_name)] = LinearFunction(a, b)
                # Otherwise killed
            else:
                result[fact] = ID_FN
                # Check if expr references this fact as linear
                lin = _extract_linear_form(expr, fact.name)
                if lin:
                    a, b = lin
                    if a == 1 and b == 0:
                        result[Fact(var_name)] = ID_FN
                    else:
                        result[Fact(var_name)] = LinearFunction(a, b)

    elif cls == 'ReturnStmt':
        result[fact] = ID_FN
    else:
        result[fact] = ID_FN

    return result


# ---------------------------------------------------------------------------
# High-level APIs
# ---------------------------------------------------------------------------

def ide_analyze(source: str, analysis: str = "copy_const") -> IDEResult:
    """Run IDE analysis on C10 source code.

    Args:
        source: C10 source code
        analysis: "copy_const" or "linear_const"

    Returns:
        IDEResult with values, summaries, and reachable facts
    """
    problem = build_ide_problem(source, analysis)
    solver = IDESolver(problem)
    return solver.solve()


def get_constants(source: str) -> Dict[str, Dict[str, LatticeValue]]:
    """Get constant values at each program point.

    Returns: point_id -> var_name -> value
    """
    result = ide_analyze(source, "copy_const")
    constants = {}
    for pt, facts in result.values.items():
        pt_consts = {}
        for fact, val in facts.items():
            if fact != ZERO and not isinstance(val, (Top, Bot)):
                pt_consts[fact.name] = val
        if pt_consts:
            constants[pt] = pt_consts
    return constants


def get_variable_value(source: str, var_name: str,
                       point: Optional[str] = None) -> LatticeValue:
    """Get the value of a variable at a specific program point.

    If point is None, returns the value at the last program point.
    """
    result = ide_analyze(source, "copy_const")
    target_fact = Fact(var_name)

    if point:
        return result.values.get(point, {}).get(target_fact, TOP)

    # Find last point in main
    main_exit = None
    if 'main' in result.values:
        return result.values.get('main', {}).get(target_fact, TOP)

    # Search all points for the last one containing the variable
    last_val = TOP
    for pt, facts in result.values.items():
        if target_fact in facts:
            last_val = facts[target_fact]

    return last_val


def linear_const_analyze(source: str) -> IDEResult:
    """Run linear constant propagation analysis."""
    return ide_analyze(source, "linear_const")


def compare_analyses(source: str) -> Dict[str, Any]:
    """Compare copy-constant vs linear constant propagation precision."""
    copy_result = ide_analyze(source, "copy_const")
    linear_result = ide_analyze(source, "linear_const")

    comparison = {
        'copy_const': {},
        'linear_const': {},
        'precision_gains': [],
    }

    # Collect values at exit points
    for fn_name, fn_info in copy_result.reachable_facts.items():
        pass

    all_points = set(copy_result.values.keys()) | set(linear_result.values.keys())
    for pt in all_points:
        copy_vals = copy_result.values.get(pt, {})
        linear_vals = linear_result.values.get(pt, {})

        for fact in set(copy_vals.keys()) | set(linear_vals.keys()):
            if fact == ZERO:
                continue
            cv = copy_vals.get(fact, TOP)
            lv = linear_vals.get(fact, TOP)

            comparison['copy_const'][f"{pt}.{fact.name}"] = str(cv)
            comparison['linear_const'][f"{pt}.{fact.name}"] = str(lv)

            # Linear is more precise if it's const where copy is TOP
            if isinstance(cv, Top) and isinstance(lv, Const):
                comparison['precision_gains'].append({
                    'point': pt,
                    'var': fact.name,
                    'copy': str(cv),
                    'linear': str(lv),
                })

    comparison['stats'] = {
        'copy_const': copy_result.stats,
        'linear_const': linear_result.stats,
    }

    return comparison


def get_function_summary(source: str, fn_name: str,
                         analysis: str = "copy_const") -> Dict:
    """Get the IDE summary of a function.

    Returns a dict mapping (input_fact, output_fact) to the micro-function
    that transforms input values to output values through the function.
    """
    result = ide_analyze(source, analysis)
    if fn_name in result.summaries:
        summary = {}
        for (d1, d2), f in result.summaries[fn_name].items():
            summary[f"({d1}, {d2})"] = str(f)
        return summary
    return {}


def ide_verify_constant(source: str, var_name: str,
                        expected_value: int) -> Dict[str, Any]:
    """Verify that a variable holds a specific constant value at program exit.

    Returns dict with 'verified' (bool), 'actual_value', and 'analysis'.
    """
    result = ide_analyze(source, "linear_const")

    # Find the value at exit of main or last function
    target = Fact(var_name)
    actual = TOP

    # Check main exit first
    main_exit = "main.exit"
    if main_exit in result.values and target in result.values[main_exit]:
        actual = result.values[main_exit][target]
    else:
        # Check last statement in main
        for pt in sorted(result.values.keys(), reverse=True):
            if pt.startswith("main.") and target in result.values[pt]:
                actual = result.values[pt][target]
                break

    verified = isinstance(actual, Const) and actual.value == expected_value
    return {
        'verified': verified,
        'actual_value': str(actual),
        'expected_value': expected_value,
        'analysis': 'linear_const',
    }
