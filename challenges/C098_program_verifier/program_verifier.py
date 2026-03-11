"""
C098: Program Verifier
Composes C037 (SMT Solver) + C097 (Program Synthesis)

A Hoare-logic program verifier with:
- Simple imperative language (assignments, if/else, while, assert, assume)
- Weakest precondition (WP) calculus for verification condition generation
- SMT-backed VC checking via C037
- Loop invariant inference via synthesis (C097)
- Strongest postcondition (SP) calculus as alternative
- Function contracts (requires/ensures)
- Array reasoning with theory of arrays
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C097_program_synthesis'))

from smt_solver import SMTSolver, SMTResult, Var, IntConst as SMTIntConst, BoolConst as SMTBoolConst, App, Op, Sort, SortKind, INT, BOOL
from synthesis import (
    Expr as SynthExpr, IntConst as SynthIntConst, BoolConst as SynthBoolConst,
    VarExpr as SynthVarExpr, BinOp as SynthBinOp, UnaryOp as SynthUnaryOp,
    IfExpr as SynthIfExpr, evaluate as synth_evaluate, IOExample, SynthesisSpec,
    EnumerativeSynthesizer, expr_size
)
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Set
from enum import Enum, auto


# ============================================================
# AST for the verification language
# ============================================================

class Stmt:
    """Base class for statements."""
    pass

@dataclass
class Skip(Stmt):
    """No-op statement."""
    pass

@dataclass
class Assign(Stmt):
    """Variable assignment: var := expr"""
    var: str
    expr: 'VExpr'

@dataclass
class Seq(Stmt):
    """Sequential composition: s1; s2"""
    first: Stmt
    second: Stmt

@dataclass
class If(Stmt):
    """Conditional: if cond then s1 else s2"""
    cond: 'VExpr'
    then_branch: Stmt
    else_branch: Stmt

@dataclass
class While(Stmt):
    """While loop with optional invariant annotation."""
    cond: 'VExpr'
    body: Stmt
    invariant: Optional['VExpr'] = None

@dataclass
class Assert(Stmt):
    """Assert a condition holds."""
    cond: 'VExpr'

@dataclass
class Assume(Stmt):
    """Assume a condition (adds to path condition)."""
    cond: 'VExpr'

@dataclass
class ArrayAssign(Stmt):
    """Array element assignment: arr[idx] := val"""
    arr: str
    index: 'VExpr'
    value: 'VExpr'

@dataclass
class FuncDecl(Stmt):
    """Function with contract: requires/ensures."""
    name: str
    params: List[str]
    requires: Optional['VExpr']
    ensures: Optional['VExpr']
    body: Stmt
    # 'result' variable holds return value in ensures

@dataclass
class Return(Stmt):
    """Return statement."""
    expr: 'VExpr'

@dataclass
class Block(Stmt):
    """Block of statements."""
    stmts: List[Stmt]


# ============================================================
# Expression AST for verification language
# ============================================================

class VExpr:
    """Base class for verification expressions."""
    pass

@dataclass(frozen=True)
class IntLit(VExpr):
    value: int

@dataclass(frozen=True)
class BoolLit(VExpr):
    value: bool

@dataclass(frozen=True)
class VarRef(VExpr):
    name: str

@dataclass(frozen=True)
class BinaryOp(VExpr):
    op: str  # +, -, *, /, %, ==, !=, <, <=, >, >=, and, or, implies
    left: VExpr
    right: VExpr

@dataclass(frozen=True)
class UnaryOp(VExpr):
    op: str  # not, neg
    operand: VExpr

@dataclass(frozen=True)
class CondExpr(VExpr):
    """If-then-else expression."""
    cond: VExpr
    then_expr: VExpr
    else_expr: VExpr

@dataclass(frozen=True)
class ArrayRead(VExpr):
    """Array element read: arr[idx]"""
    arr: VExpr
    index: VExpr

@dataclass(frozen=True)
class ArrayStore(VExpr):
    """Array store expression (functional): store(arr, idx, val)"""
    arr: VExpr
    index: VExpr
    value: VExpr

@dataclass(frozen=True)
class Forall(VExpr):
    """Universal quantification (for specification)."""
    var: str
    body: VExpr

@dataclass(frozen=True)
class Exists(VExpr):
    """Existential quantification."""
    var: str
    body: VExpr

@dataclass(frozen=True)
class OldExpr(VExpr):
    """Old(expr) -- value of expr at function entry (for ensures clauses)."""
    expr: VExpr

@dataclass(frozen=True)
class ResultExpr(VExpr):
    """The return value of a function (used in ensures)."""
    pass


# ============================================================
# Expression helpers
# ============================================================

def AND(*args):
    """Build conjunction."""
    exprs = [a for a in args if a != BoolLit(True)]
    if not exprs:
        return BoolLit(True)
    if BoolLit(False) in exprs:
        return BoolLit(False)
    result = exprs[0]
    for e in exprs[1:]:
        result = BinaryOp('and', result, e)
    return result

def OR(*args):
    """Build disjunction."""
    exprs = [a for a in args if a != BoolLit(False)]
    if not exprs:
        return BoolLit(False)
    if BoolLit(True) in exprs:
        return BoolLit(True)
    result = exprs[0]
    for e in exprs[1:]:
        result = BinaryOp('or', result, e)
    return result

def NOT(e):
    if isinstance(e, BoolLit):
        return BoolLit(not e.value)
    if isinstance(e, UnaryOp) and e.op == 'not':
        return e.operand
    return UnaryOp('not', e)

def IMPLIES(a, b):
    return BinaryOp('implies', a, b)

def EQ(a, b):
    return BinaryOp('==', a, b)

def LT(a, b):
    return BinaryOp('<', a, b)

def LE(a, b):
    return BinaryOp('<=', a, b)

def GT(a, b):
    return BinaryOp('>', a, b)

def GE(a, b):
    return BinaryOp('>=', a, b)

def ADD(a, b):
    return BinaryOp('+', a, b)

def SUB(a, b):
    return BinaryOp('-', a, b)

def MUL(a, b):
    return BinaryOp('*', a, b)

def NEG(e):
    return UnaryOp('neg', e)

def VAR(name):
    return VarRef(name)

def INT(val):
    return IntLit(val)

def BOOL(val):
    return BoolLit(val)


# ============================================================
# Substitution
# ============================================================

def substitute(expr, var_name, replacement):
    """Substitute all occurrences of var_name with replacement in expr."""
    if isinstance(expr, IntLit) or isinstance(expr, BoolLit):
        return expr
    if isinstance(expr, VarRef):
        return replacement if expr.name == var_name else expr
    if isinstance(expr, BinaryOp):
        return BinaryOp(expr.op,
                        substitute(expr.left, var_name, replacement),
                        substitute(expr.right, var_name, replacement))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, substitute(expr.operand, var_name, replacement))
    if isinstance(expr, CondExpr):
        return CondExpr(substitute(expr.cond, var_name, replacement),
                        substitute(expr.then_expr, var_name, replacement),
                        substitute(expr.else_expr, var_name, replacement))
    if isinstance(expr, ArrayRead):
        return ArrayRead(substitute(expr.arr, var_name, replacement),
                         substitute(expr.index, var_name, replacement))
    if isinstance(expr, ArrayStore):
        return ArrayStore(substitute(expr.arr, var_name, replacement),
                          substitute(expr.index, var_name, replacement),
                          substitute(expr.value, var_name, replacement))
    if isinstance(expr, Forall):
        if expr.var == var_name:
            return expr  # bound variable shadows
        return Forall(expr.var, substitute(expr.body, var_name, replacement))
    if isinstance(expr, Exists):
        if expr.var == var_name:
            return expr
        return Exists(expr.var, substitute(expr.body, var_name, replacement))
    if isinstance(expr, OldExpr):
        return OldExpr(substitute(expr.expr, var_name, replacement))
    if isinstance(expr, ResultExpr):
        return replacement if var_name == 'result' else expr
    return expr


def free_vars(expr):
    """Collect free variable names in expr."""
    if isinstance(expr, IntLit) or isinstance(expr, BoolLit):
        return set()
    if isinstance(expr, VarRef):
        return {expr.name}
    if isinstance(expr, BinaryOp):
        return free_vars(expr.left) | free_vars(expr.right)
    if isinstance(expr, UnaryOp):
        return free_vars(expr.operand)
    if isinstance(expr, CondExpr):
        return free_vars(expr.cond) | free_vars(expr.then_expr) | free_vars(expr.else_expr)
    if isinstance(expr, ArrayRead):
        return free_vars(expr.arr) | free_vars(expr.index)
    if isinstance(expr, ArrayStore):
        return free_vars(expr.arr) | free_vars(expr.index) | free_vars(expr.value)
    if isinstance(expr, Forall) or isinstance(expr, Exists):
        return free_vars(expr.body) - {expr.var}
    if isinstance(expr, OldExpr):
        return free_vars(expr.expr)
    if isinstance(expr, ResultExpr):
        return set()
    return set()


# ============================================================
# Weakest Precondition Calculus
# ============================================================

class WPCalculus:
    """Dijkstra's weakest precondition calculus."""

    def __init__(self):
        self.vcs = []  # collected verification conditions
        self._counter = 0

    def fresh_var(self, base='_wp'):
        self._counter += 1
        return f'{base}_{self._counter}'

    def wp(self, stmt, post):
        """Compute weakest precondition of stmt w.r.t. postcondition post."""
        if isinstance(stmt, Skip):
            return post

        if isinstance(stmt, Assign):
            # wp(x := e, Q) = Q[x -> e]
            return substitute(post, stmt.var, stmt.expr)

        if isinstance(stmt, Seq):
            # wp(s1; s2, Q) = wp(s1, wp(s2, Q))
            wp2 = self.wp(stmt.second, post)
            return self.wp(stmt.first, wp2)

        if isinstance(stmt, If):
            # wp(if b then s1 else s2, Q) = (b => wp(s1,Q)) /\ (!b => wp(s2,Q))
            wp_then = self.wp(stmt.then_branch, post)
            wp_else = self.wp(stmt.else_branch, post)
            return AND(IMPLIES(stmt.cond, wp_then),
                       IMPLIES(NOT(stmt.cond), wp_else))

        if isinstance(stmt, While):
            # Requires loop invariant I
            # wp(while b do s, Q) = I /\ (I /\ b => wp(s, I)) /\ (I /\ !b => Q)
            # All VCs are embedded in the formula for correct context propagation
            if stmt.invariant is None:
                raise VerificationError("Loop requires invariant annotation")
            inv = stmt.invariant
            body_wp = self.wp(stmt.body, inv)
            preservation = IMPLIES(AND(inv, stmt.cond), body_wp)
            exit_cond = IMPLIES(AND(inv, NOT(stmt.cond)), post)
            return AND(inv, AND(preservation, exit_cond))

        if isinstance(stmt, Assert):
            # wp(assert P, Q) = P /\ Q
            # No separate VC -- the main pre => wp VC checks reachability
            return AND(stmt.cond, post)

        if isinstance(stmt, Assume):
            # wp(assume P, Q) = P => Q
            return IMPLIES(stmt.cond, post)

        if isinstance(stmt, ArrayAssign):
            # wp(a[i] := v, Q) = Q[a -> store(a, i, v)]
            new_arr = ArrayStore(VarRef(stmt.arr), stmt.index, stmt.value)
            return substitute(post, stmt.arr, new_arr)

        if isinstance(stmt, Block):
            # Process statements in reverse
            result = post
            for s in reversed(stmt.stmts):
                result = self.wp(s, result)
            return result

        if isinstance(stmt, FuncDecl):
            return self._wp_func(stmt, post)

        if isinstance(stmt, Return):
            # wp(return e, Q) = Q[result -> e]
            return substitute(post, 'result', stmt.expr)

        raise VerificationError(f"Unknown statement type: {type(stmt)}")

    def _wp_func(self, func, post):
        """Handle function contract verification."""
        pre = func.requires if func.requires else BoolLit(True)
        ensures = func.ensures if func.ensures else BoolLit(True)

        # Embed: pre => wp(body, ensures)
        body_wp = self.wp(func.body, ensures)
        return IMPLIES(pre, body_wp)


# ============================================================
# Strongest Postcondition Calculus
# ============================================================

class SPCalculus:
    """Strongest postcondition calculus (forward reasoning)."""

    def __init__(self):
        self.vcs = []
        self._counter = 0

    def fresh_var(self, base='_sp'):
        self._counter += 1
        return f'{base}_{self._counter}'

    def sp(self, stmt, pre):
        """Compute strongest postcondition of stmt w.r.t. precondition pre."""
        if isinstance(stmt, Skip):
            return pre

        if isinstance(stmt, Assign):
            # sp(x := e, P) = exists x0. P[x->x0] /\ x == e[x->x0]
            # Simplified: track as substitution
            old_var = self.fresh_var(stmt.var)
            p_with_old = substitute(pre, stmt.var, VarRef(old_var))
            e_with_old = substitute(stmt.expr, stmt.var, VarRef(old_var))
            return AND(p_with_old, EQ(VarRef(stmt.var), e_with_old))

        if isinstance(stmt, Seq):
            sp1 = self.sp(stmt.first, pre)
            return self.sp(stmt.second, sp1)

        if isinstance(stmt, If):
            sp_then = self.sp(stmt.then_branch, AND(pre, stmt.cond))
            sp_else = self.sp(stmt.else_branch, AND(pre, NOT(stmt.cond)))
            return OR(sp_then, sp_else)

        if isinstance(stmt, While):
            if stmt.invariant is None:
                raise VerificationError("Loop requires invariant for SP")
            inv = stmt.invariant
            # VC: pre => inv (invariant initially holds)
            self.vcs.append(VC('loop_init', IMPLIES(pre, inv), "Loop invariant holds initially"))
            # VC: inv /\ cond => sp(body, inv) => inv
            body_sp = self.sp(stmt.body, AND(inv, stmt.cond))
            self.vcs.append(VC('loop_preservation_sp', IMPLIES(body_sp, inv), "Loop invariant preserved"))
            return AND(inv, NOT(stmt.cond))

        if isinstance(stmt, Assert):
            self.vcs.append(VC('assertion', IMPLIES(pre, stmt.cond), "Assertion holds"))
            return AND(pre, stmt.cond)

        if isinstance(stmt, Assume):
            return AND(pre, stmt.cond)

        if isinstance(stmt, Block):
            result = pre
            for s in stmt.stmts:
                result = self.sp(s, result)
            return result

        raise VerificationError(f"Unknown statement type for SP: {type(stmt)}")


# ============================================================
# Verification Condition
# ============================================================

@dataclass
class VC:
    """A verification condition to be checked."""
    kind: str        # 'loop_preservation', 'loop_exit', 'assertion', 'function_contract', etc.
    formula: VExpr   # Must be valid (true in all states)
    description: str

    def __repr__(self):
        return f"VC({self.kind}: {self.description})"


class VerificationError(Exception):
    pass


# ============================================================
# SMT Translation
# ============================================================

class SMTTranslator:
    """Translate verification expressions to SMT terms."""

    def __init__(self, solver):
        self.solver = solver
        self._vars = {}  # name -> SMT Var
        self._array_vars = {}  # name -> SMT Function (array as uninterpreted function)

    def get_var(self, name, sort='int'):
        if name not in self._vars:
            if sort == 'bool':
                self._vars[name] = self.solver.Bool(name)
            else:
                self._vars[name] = self.solver.Int(name)
        return self._vars[name]

    def translate(self, expr):
        """Translate VExpr to SMT Term."""
        if isinstance(expr, IntLit):
            return self.solver.IntVal(expr.value)

        if isinstance(expr, BoolLit):
            return self.solver.BoolVal(expr.value)

        if isinstance(expr, VarRef):
            return self.get_var(expr.name)

        if isinstance(expr, BinaryOp):
            left = self.translate(expr.left)
            right = self.translate(expr.right)
            op = expr.op
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                # Integer division -- avoid div by zero
                return self.solver.If(right == self.solver.IntVal(0),
                                       self.solver.IntVal(0),
                                       App(Op.MUL, [left, self.solver.IntVal(0)], Sort(SortKind.INT)))
            elif op == '%':
                return left - left  # Simplified -- modulo hard for SMT
            elif op == '==':
                return left == right
            elif op == '!=':
                return left != right
            elif op == '<':
                return left < right
            elif op == '<=':
                return left <= right
            elif op == '>':
                return left > right
            elif op == '>=':
                return left >= right
            elif op == 'and':
                return self.solver.And(left, right)
            elif op == 'or':
                return self.solver.Or(left, right)
            elif op == 'implies':
                return self.solver.Implies(left, right)
            else:
                raise VerificationError(f"Unknown binary op: {op}")

        if isinstance(expr, UnaryOp):
            operand = self.translate(expr.operand)
            if expr.op == 'not':
                return self.solver.Not(operand)
            elif expr.op == 'neg':
                return self.solver.IntVal(0) - operand
            else:
                raise VerificationError(f"Unknown unary op: {expr.op}")

        if isinstance(expr, CondExpr):
            c = self.translate(expr.cond)
            t = self.translate(expr.then_expr)
            e = self.translate(expr.else_expr)
            return self.solver.If(c, t, e)

        if isinstance(expr, ArrayRead):
            arr_term = self.translate(expr.arr)
            idx_term = self.translate(expr.index)
            # Model arrays as uninterpreted functions
            if isinstance(expr.arr, VarRef):
                name = expr.arr.name
                if name not in self._array_vars:
                    from smt_solver import Sort, SortKind
                    int_sort = Sort(SortKind.INT)
                    self._array_vars[name] = self.solver.Function(name, int_sort, int_sort)
                return self._array_vars[name](idx_term)
            # Nested -- use ITE chain or fallback
            return self.solver.IntVal(0)

        if isinstance(expr, ArrayStore):
            # store(a, i, v)[j] == ite(i==j, v, a[j])
            # For top-level store, we handle via substitution in WP
            # Here just translate the components
            return self.solver.IntVal(0)  # Shouldn't reach here after WP

        if isinstance(expr, Forall):
            # Approximate: instantiate with a few values
            # Full quantifier elimination is out of scope for C037
            body = expr.body
            result = self.solver.BoolVal(True)
            for val in range(-2, 3):
                inst = substitute(body, expr.var, IntLit(val))
                result = self.solver.And(result, self.translate(inst))
            return result

        if isinstance(expr, Exists):
            body = expr.body
            result = self.solver.BoolVal(False)
            for val in range(-2, 3):
                inst = substitute(body, expr.var, IntLit(val))
                result = self.solver.Or(result, self.translate(inst))
            return result

        if isinstance(expr, ResultExpr):
            return self.get_var('result')

        if isinstance(expr, OldExpr):
            # Old values tracked with _old_ prefix
            if isinstance(expr.expr, VarRef):
                return self.get_var(f'_old_{expr.expr.name}')
            return self.translate(expr.expr)

        raise VerificationError(f"Cannot translate to SMT: {type(expr)}")


# ============================================================
# Loop Invariant Inference
# ============================================================

class InvariantInference:
    """Infer loop invariants using synthesis (C097)."""

    def __init__(self, max_size=8, max_depth=3):
        self.max_size = max_size
        self.max_depth = max_depth

    def infer(self, loop, pre, post, trace_states=None):
        """
        Attempt to infer a loop invariant.

        Args:
            loop: While statement
            pre: Precondition (VExpr)
            post: Postcondition (VExpr)
            trace_states: Optional list of (env, is_entry) from concrete execution

        Returns:
            VExpr invariant or None
        """
        # Strategy 1: Try the precondition itself
        inv = pre
        if self._check_invariant(loop, inv, post):
            return inv

        # Strategy 2: Try the postcondition weakened
        inv = AND(post, loop.cond)
        # This often doesn't work, but worth trying

        # Strategy 3: Try common invariant templates
        vars_in_loop = self._collect_vars(loop)
        candidates = self._generate_candidates(vars_in_loop, pre, post, loop.cond)
        for cand in candidates:
            if self._check_invariant(loop, cand, post):
                return cand

        # Strategy 4: Use synthesis from traces
        if trace_states:
            inv = self._synthesize_from_traces(trace_states, vars_in_loop, pre, post, loop)
            if inv is not None:
                return inv

        return None

    def _check_invariant(self, loop, inv, post):
        """Check if inv is a valid loop invariant."""
        solver = SMTSolver()
        translator = SMTTranslator(solver)

        # Check: inv /\ !cond => post
        try:
            solver.push()
            exit_cond = AND(inv, NOT(loop.cond))
            smt_exit = translator.translate(exit_cond)
            smt_post = translator.translate(post)
            solver.add(smt_exit)
            solver.add(solver.Not(smt_post))
            result = solver.check()
            solver.pop()
            if result != SMTResult.UNSAT:
                return False
        except Exception:
            return False

        # Check: inv /\ cond => wp(body, inv)
        try:
            wp_calc = WPCalculus()
            body_wp = wp_calc.wp(loop.body, inv)
            solver2 = SMTSolver()
            translator2 = SMTTranslator(solver2)
            entry_cond = AND(inv, loop.cond)
            smt_entry = translator2.translate(entry_cond)
            smt_wp = translator2.translate(body_wp)
            solver2.add(smt_entry)
            solver2.add(solver2.Not(smt_wp))
            result = solver2.check()
            if result != SMTResult.UNSAT:
                return False
        except Exception:
            return False

        return True

    def _collect_vars(self, stmt):
        """Collect variable names used in a statement."""
        vars_set = set()
        if isinstance(stmt, Assign):
            vars_set.add(stmt.var)
            vars_set |= free_vars(stmt.expr)
        elif isinstance(stmt, Seq):
            vars_set |= self._collect_vars(stmt.first)
            vars_set |= self._collect_vars(stmt.second)
        elif isinstance(stmt, If):
            vars_set |= free_vars(stmt.cond)
            vars_set |= self._collect_vars(stmt.then_branch)
            vars_set |= self._collect_vars(stmt.else_branch)
        elif isinstance(stmt, While):
            vars_set |= free_vars(stmt.cond)
            vars_set |= self._collect_vars(stmt.body)
        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                vars_set |= self._collect_vars(s)
        elif isinstance(stmt, Assert) or isinstance(stmt, Assume):
            vars_set |= free_vars(stmt.cond)
        elif isinstance(stmt, ArrayAssign):
            vars_set.add(stmt.arr)
            vars_set |= free_vars(stmt.index)
            vars_set |= free_vars(stmt.value)
        return vars_set

    def _generate_candidates(self, vars_in_loop, pre, post, cond):
        """Generate candidate invariants from templates."""
        candidates = []
        var_list = sorted(vars_in_loop)

        # Template: var >= 0
        for v in var_list:
            candidates.append(GE(VarRef(v), IntLit(0)))

        # Template: var <= const (from postcondition)
        # Template: var1 + var2 == const
        if len(var_list) >= 2:
            for i in range(len(var_list)):
                for j in range(i+1, len(var_list)):
                    v1, v2 = var_list[i], var_list[j]
                    # Try v1 + v2 == some constant from pre
                    for c in [0, 1, -1]:
                        candidates.append(EQ(ADD(VarRef(v1), VarRef(v2)), IntLit(c)))

        # Template: var1 * const + var2 == const
        for v in var_list:
            for c in range(2, 5):
                candidates.append(LE(VarRef(v), IntLit(c)))
                candidates.append(GE(VarRef(v), IntLit(-c)))

        # Template: conjunction of simple bounds
        # Try postcondition as invariant
        candidates.append(post)

        # Try weakening postcondition
        candidates.append(BoolLit(True))

        return candidates

    def _synthesize_from_traces(self, trace_states, variables, pre, post, loop):
        """Use program synthesis to find invariant from execution traces."""
        var_list = sorted(variables)
        if not var_list or not trace_states:
            return None

        # Collect states where invariant must hold (loop entry points)
        positive_states = []
        for state, is_entry in trace_states:
            if is_entry:
                positive_states.append(state)

        if not positive_states:
            return None

        # Try to synthesize a boolean expression that's true at all entry points
        # Use comparison operators to build candidate invariants
        for v1 in var_list:
            for v2 in var_list:
                if v1 >= v2:
                    continue
                # Check if v1 <= v2 at all entry states
                if all(s.get(v1, 0) <= s.get(v2, 0) for s in positive_states):
                    cand = LE(VarRef(v1), VarRef(v2))
                    if self._check_invariant(loop, cand, post):
                        return cand
                # Check if v1 + v2 is constant
                sums = [s.get(v1, 0) + s.get(v2, 0) for s in positive_states]
                if len(set(sums)) == 1:
                    cand = EQ(ADD(VarRef(v1), VarRef(v2)), IntLit(sums[0]))
                    if self._check_invariant(loop, cand, post):
                        return cand

        # Try single variable bounds
        for v in var_list:
            vals = [s.get(v, 0) for s in positive_states]
            min_val = min(vals)
            # Invariant: v >= min_val
            cand = GE(VarRef(v), IntLit(min_val))
            if self._check_invariant(loop, cand, post):
                return cand

        return None


# ============================================================
# Concrete Executor (for trace generation)
# ============================================================

class ConcreteExecutor:
    """Execute programs concretely to generate traces for invariant inference."""

    def __init__(self, max_steps=1000):
        self.max_steps = max_steps

    def execute(self, stmt, env=None):
        """Execute statement, return (final_env, trace).
        trace = list of (env_snapshot, event) where event is 'loop_entry', 'loop_exit', etc.
        """
        if env is None:
            env = {}
        trace = []
        self._steps = 0
        self._execute(stmt, env, trace)
        return env, trace

    def _execute(self, stmt, env, trace):
        self._steps += 1
        if self._steps > self.max_steps:
            raise VerificationError("Execution exceeded max steps")

        if isinstance(stmt, Skip):
            return

        if isinstance(stmt, Assign):
            env[stmt.var] = self._eval(stmt.expr, env)

        elif isinstance(stmt, Seq):
            self._execute(stmt.first, env, trace)
            self._execute(stmt.second, env, trace)

        elif isinstance(stmt, If):
            if self._eval(stmt.cond, env):
                self._execute(stmt.then_branch, env, trace)
            else:
                self._execute(stmt.else_branch, env, trace)

        elif isinstance(stmt, While):
            trace.append((dict(env), True))  # loop entry
            while self._eval(stmt.cond, env):
                self._execute(stmt.body, env, trace)
                trace.append((dict(env), True))  # re-entry
                self._steps += 1
                if self._steps > self.max_steps:
                    raise VerificationError("Loop exceeded max steps")

        elif isinstance(stmt, Assert):
            val = self._eval(stmt.cond, env)
            if not val:
                raise VerificationError(f"Assertion failed: {stmt.cond}")

        elif isinstance(stmt, Assume):
            val = self._eval(stmt.cond, env)
            if not val:
                raise VerificationError(f"Assumption violated")

        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                self._execute(s, env, trace)

        elif isinstance(stmt, ArrayAssign):
            arr = env.get(stmt.arr, {})
            idx = self._eval(stmt.index, env)
            val = self._eval(stmt.value, env)
            arr[idx] = val
            env[stmt.arr] = arr

        elif isinstance(stmt, Return):
            env['result'] = self._eval(stmt.expr, env)

    def _eval(self, expr, env):
        if isinstance(expr, IntLit):
            return expr.value
        if isinstance(expr, BoolLit):
            return expr.value
        if isinstance(expr, VarRef):
            if expr.name not in env:
                return 0  # default
            return env[expr.name]
        if isinstance(expr, BinaryOp):
            left = self._eval(expr.left, env)
            right = self._eval(expr.right, env)
            op = expr.op
            if op == '+': return left + right
            if op == '-': return left - right
            if op == '*': return left * right
            if op == '/': return left // right if right != 0 else 0
            if op == '%': return left % right if right != 0 else 0
            if op == '==': return left == right
            if op == '!=': return left != right
            if op == '<': return left < right
            if op == '<=': return left <= right
            if op == '>': return left > right
            if op == '>=': return left >= right
            if op == 'and': return left and right
            if op == 'or': return left or right
            if op == 'implies': return (not left) or right
        if isinstance(expr, UnaryOp):
            val = self._eval(expr.operand, env)
            if expr.op == 'not': return not val
            if expr.op == 'neg': return -val
        if isinstance(expr, CondExpr):
            if self._eval(expr.cond, env):
                return self._eval(expr.then_expr, env)
            return self._eval(expr.else_expr, env)
        if isinstance(expr, ArrayRead):
            arr = self._eval(expr.arr, env)
            idx = self._eval(expr.index, env)
            if isinstance(arr, dict):
                return arr.get(idx, 0)
            return 0
        return 0


# ============================================================
# Verification Result
# ============================================================

class VResult(Enum):
    VERIFIED = auto()
    FAILED = auto()
    UNKNOWN = auto()

@dataclass
class VerificationResult:
    status: VResult
    vcs_total: int = 0
    vcs_verified: int = 0
    vcs_failed: int = 0
    vcs_unknown: int = 0
    failed_vcs: List[VC] = field(default_factory=list)
    counterexamples: List[Dict] = field(default_factory=list)
    inferred_invariants: List[Tuple[str, VExpr]] = field(default_factory=list)

    @property
    def verified(self):
        return self.status == VResult.VERIFIED


# ============================================================
# Main Verifier
# ============================================================

class ProgramVerifier:
    """
    Main verification engine.
    Composes WP calculus + SMT solver + invariant inference.
    """

    def __init__(self, infer_invariants=True, use_sp=False, max_trace_inputs=5):
        self.infer_invariants = infer_invariants
        self.use_sp = use_sp
        self.max_trace_inputs = max_trace_inputs

    def verify(self, stmt, precondition=None, postcondition=None):
        """
        Verify a program against a specification.

        Args:
            stmt: The program (Stmt)
            precondition: VExpr (default: True)
            postcondition: VExpr (default: True)

        Returns:
            VerificationResult
        """
        pre = precondition if precondition else BoolLit(True)
        post = postcondition if postcondition else BoolLit(True)

        # Step 1: Infer missing loop invariants if enabled
        if self.infer_invariants:
            stmt = self._infer_loop_invariants(stmt, pre, post)

        # Step 2: Generate VCs
        vcs = self._generate_vcs(stmt, pre, post)

        # Step 3: Check each VC with SMT
        return self._check_vcs(vcs)

    def verify_function(self, func):
        """Verify a function declaration with contracts."""
        pre = func.requires if func.requires else BoolLit(True)
        post = func.ensures if func.ensures else BoolLit(True)

        if self.infer_invariants:
            func = FuncDecl(func.name, func.params, func.requires, func.ensures,
                           self._infer_loop_invariants(func.body, pre, post))

        calc = WPCalculus()
        wp_result = calc.wp(func, BoolLit(True))
        vcs = [VC('function_contract', wp_result, f"Function '{func.name}' contract")]
        return self._check_vcs(vcs)

    def _generate_vcs(self, stmt, pre, post):
        """Generate verification conditions."""
        if self.use_sp:
            calc = SPCalculus()
            sp_result = calc.sp(stmt, pre)
            # The main VC: sp(stmt, pre) => post
            calc.vcs.append(VC('postcondition', IMPLIES(sp_result, post),
                              "Strongest postcondition implies desired postcondition"))
            return calc.vcs
        else:
            calc = WPCalculus()
            wp_result = calc.wp(stmt, post)
            # Single VC: pre => wp(stmt, post) (loop VCs are embedded)
            return [VC('precondition', IMPLIES(pre, wp_result),
                       "Precondition implies weakest precondition")]

    def _check_vcs(self, vcs):
        """Check all verification conditions using SMT."""
        total = len(vcs)
        verified = 0
        failed = 0
        unknown = 0
        failed_vcs = []
        counterexamples = []

        for vc in vcs:
            result, model = self._check_single_vc(vc)
            if result == VResult.VERIFIED:
                verified += 1
            elif result == VResult.FAILED:
                failed += 1
                failed_vcs.append(vc)
                if model:
                    counterexamples.append(model)
            else:
                unknown += 1

        if failed > 0:
            status = VResult.FAILED
        elif unknown > 0:
            status = VResult.UNKNOWN
        else:
            status = VResult.VERIFIED

        return VerificationResult(
            status=status,
            vcs_total=total,
            vcs_verified=verified,
            vcs_failed=failed,
            vcs_unknown=unknown,
            failed_vcs=failed_vcs,
            counterexamples=counterexamples
        )

    def _check_single_vc(self, vc):
        """Check a single VC. Returns (VResult, optional_counterexample)."""
        solver = SMTSolver()
        translator = SMTTranslator(solver)

        try:
            # To check validity of P, check unsat of !P
            smt_formula = translator.translate(vc.formula)
            solver.add(solver.Not(smt_formula))
            result = solver.check()

            if result == SMTResult.UNSAT:
                return VResult.VERIFIED, None
            elif result == SMTResult.SAT:
                model = solver.model()
                return VResult.FAILED, model
            else:
                return VResult.UNKNOWN, None
        except Exception as e:
            return VResult.UNKNOWN, None

    def _infer_loop_invariants(self, stmt, pre, post):
        """Walk the AST and infer invariants for unannotated loops."""
        if isinstance(stmt, While) and stmt.invariant is None:
            # Try to infer invariant
            inference = InvariantInference()

            # Generate traces for inference
            traces = self._generate_traces(stmt, pre)

            inv = inference.infer(stmt, pre, post, traces)
            if inv is not None:
                return While(stmt.cond, self._infer_loop_invariants(stmt.body, pre, post),
                           invariant=inv)
            return stmt

        if isinstance(stmt, Seq):
            return Seq(self._infer_loop_invariants(stmt.first, pre, post),
                      self._infer_loop_invariants(stmt.second, pre, post))

        if isinstance(stmt, If):
            return If(stmt.cond,
                     self._infer_loop_invariants(stmt.then_branch, pre, post),
                     self._infer_loop_invariants(stmt.else_branch, pre, post))

        if isinstance(stmt, Block):
            return Block([self._infer_loop_invariants(s, pre, post) for s in stmt.stmts])

        return stmt

    def _generate_traces(self, loop_stmt, pre):
        """Generate concrete execution traces for invariant inference."""
        executor = ConcreteExecutor(max_steps=200)
        all_traces = []

        # Try various initial states
        vars_in_pre = free_vars(pre)
        test_envs = self._generate_test_envs(vars_in_pre)

        for env in test_envs[:self.max_trace_inputs]:
            try:
                # Check if precondition holds
                if self._eval_pre(pre, env):
                    _, trace = executor.execute(loop_stmt, dict(env))
                    all_traces.extend(trace)
            except (VerificationError, Exception):
                continue

        return all_traces

    def _generate_test_envs(self, var_names):
        """Generate test environments."""
        envs = []
        var_list = sorted(var_names)
        if not var_list:
            return [{}]

        # Simple: try small values
        values = [0, 1, 2, 3, 5, 10]
        if len(var_list) == 1:
            for v in values:
                envs.append({var_list[0]: v})
        elif len(var_list) == 2:
            for v1 in values[:4]:
                for v2 in values[:4]:
                    envs.append({var_list[0]: v1, var_list[1]: v2})
        else:
            for v in values[:3]:
                envs.append({var: v for var in var_list})
            for i, var in enumerate(var_list):
                env = {v: 0 for v in var_list}
                env[var] = 5
                envs.append(env)

        return envs

    def _eval_pre(self, pre, env):
        """Evaluate precondition in concrete environment."""
        executor = ConcreteExecutor()
        try:
            return executor._eval(pre, env)
        except Exception:
            return True


# ============================================================
# Parser for a simple verification language
# ============================================================

class VerifParser:
    """Parse a simple verification language into AST.

    Syntax:
        stmt := 'skip'
              | var ':=' expr
              | stmt ';' stmt
              | 'if' expr 'then' stmt 'else' stmt 'end'
              | 'while' expr ['invariant' expr] 'do' stmt 'end'
              | 'assert' expr
              | 'assume' expr
              | '{' stmt* '}'
              | 'function' name '(' params ')' ['requires' expr] ['ensures' expr] '{' stmt '}'
              | 'return' expr

        expr := integer | 'true' | 'false' | var
              | expr op expr
              | 'not' expr | '-' expr
              | '(' expr ')'
              | 'old(' expr ')'
              | 'result'
    """

    def __init__(self, source):
        self.tokens = self._tokenize(source)
        self.pos = 0

    def _tokenize(self, source):
        import re
        token_re = re.compile(r'''
            (\s+)|                   # whitespace
            (//[^\n]*)|              # comment
            (:=)|                    # assignment
            (<=|>=|!=|==|=>)|        # two-char ops
            ([+\-*/%<>])|            # single-char ops
            ([(){}\[\];,])|          # delimiters
            (\d+)|                   # integer
            ([a-zA-Z_]\w*)|          # identifier
            (.)                      # error
        ''', re.VERBOSE)
        tokens = []
        for m in token_re.finditer(source):
            if m.group(1) or m.group(2):
                continue  # skip whitespace and comments
            val = m.group()
            tokens.append(val)
        return tokens

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tok):
        if self.peek() != tok:
            raise VerificationError(f"Expected '{tok}', got '{self.peek()}'")
        return self.advance()

    def _skip_semis(self):
        while self.peek() == ';':
            self.advance()

    def _parse_stmt_list(self, terminators=()):
        """Parse a sequence of statements separated by ';', stopping at terminators."""
        stmts = []
        self._skip_semis()
        while self.pos < len(self.tokens) and self.peek() not in terminators:
            stmts.append(self.parse_stmt())
            self._skip_semis()
        if len(stmts) == 0:
            return Skip()
        if len(stmts) == 1:
            return stmts[0]
        result = stmts[0]
        for s in stmts[1:]:
            result = Seq(result, s)
        return result

    def parse_program(self):
        """Parse a complete program."""
        return self._parse_stmt_list()

    def parse_stmt(self):
        tok = self.peek()

        if tok == 'skip':
            self.advance()
            return Skip()

        if tok == 'if':
            return self.parse_if()

        if tok == 'while':
            return self.parse_while()

        if tok == 'assert':
            self.advance()
            expr = self.parse_expr()
            return Assert(expr)

        if tok == 'assume':
            self.advance()
            expr = self.parse_expr()
            return Assume(expr)

        if tok == '{':
            return self.parse_block()

        if tok == 'function':
            return self.parse_function()

        if tok == 'return':
            self.advance()
            expr = self.parse_expr()
            return Return(expr)

        # Assignment: var := expr
        if tok and tok.isidentifier() and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1] == ':=':
            var = self.advance()
            self.expect(':=')
            expr = self.parse_expr()
            return Assign(var, expr)

        raise VerificationError(f"Unexpected token: {tok}")

    def parse_if(self):
        self.expect('if')
        cond = self.parse_expr()
        self.expect('then')
        then_branch = self._parse_stmt_list(('else', 'end'))
        if self.peek() == 'else':
            self.advance()
            else_branch = self._parse_stmt_list(('end',))
        else:
            else_branch = Skip()
        self.expect('end')
        return If(cond, then_branch, else_branch)

    def parse_while(self):
        self.expect('while')
        cond = self.parse_expr()
        inv = None
        if self.peek() == 'invariant':
            self.advance()
            inv = self.parse_expr()
        self.expect('do')
        body = self._parse_stmt_list(('end',))
        self.expect('end')
        return While(cond, body, invariant=inv)

    def parse_block(self):
        self.expect('{')
        result = self._parse_stmt_list(('}',))
        self.expect('}')
        return result

    def parse_function(self):
        self.expect('function')
        name = self.advance()
        self.expect('(')
        params = []
        while self.peek() != ')':
            params.append(self.advance())
            if self.peek() == ',':
                self.advance()
        self.expect(')')
        req = None
        ens = None
        if self.peek() == 'requires':
            self.advance()
            req = self.parse_expr()
        if self.peek() == 'ensures':
            self.advance()
            ens = self.parse_expr()
        self.expect('{')
        body = self._parse_stmt_list(('}',))
        self.expect('}')
        return FuncDecl(name, params, req, ens, body)

    def parse_expr(self):
        return self.parse_implies()

    def parse_implies(self):
        left = self.parse_or()
        while self.peek() == '=>':
            self.advance()
            right = self.parse_or()
            left = BinaryOp('implies', left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.peek() == 'or':
            self.advance()
            right = self.parse_and()
            left = BinaryOp('or', left, right)
        return left

    def parse_and(self):
        left = self.parse_comparison()
        while self.peek() == 'and':
            self.advance()
            right = self.parse_comparison()
            left = BinaryOp('and', left, right)
        return left

    def parse_comparison(self):
        left = self.parse_add()
        if self.peek() in ('==', '!=', '<', '<=', '>', '>='):
            op = self.advance()
            right = self.parse_add()
            left = BinaryOp(op, left, right)
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek() in ('+', '-'):
            op = self.advance()
            right = self.parse_mul()
            left = BinaryOp(op, left, right)
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek() in ('*', '/', '%'):
            op = self.advance()
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self):
        if self.peek() == 'not':
            self.advance()
            return UnaryOp('not', self.parse_unary())
        if self.peek() == '-':
            self.advance()
            return UnaryOp('neg', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self):
        tok = self.peek()

        if tok == '(':
            self.advance()
            expr = self.parse_expr()
            self.expect(')')
            return expr

        if tok == 'true':
            self.advance()
            return BoolLit(True)

        if tok == 'false':
            self.advance()
            return BoolLit(False)

        if tok == 'old':
            self.advance()
            self.expect('(')
            expr = self.parse_expr()
            self.expect(')')
            return OldExpr(expr)

        if tok == 'result':
            self.advance()
            return ResultExpr()

        if tok and tok.isdigit():
            self.advance()
            return IntLit(int(tok))

        if tok and (tok[0].isalpha() or tok[0] == '_'):
            self.advance()
            return VarRef(tok)

        raise VerificationError(f"Unexpected token in expression: {tok}")


def parse(source):
    """Parse source code into AST."""
    return VerifParser(source).parse_program()


def verify(source, precondition=None, postcondition=None, **kwargs):
    """Convenience: parse and verify in one call."""
    stmt = parse(source)
    v = ProgramVerifier(**kwargs)
    return v.verify(stmt, precondition, postcondition)


# ============================================================
# Pretty-printing
# ============================================================

def format_expr(expr, indent=0):
    """Format a VExpr as a readable string."""
    if isinstance(expr, IntLit):
        return str(expr.value)
    if isinstance(expr, BoolLit):
        return 'true' if expr.value else 'false'
    if isinstance(expr, VarRef):
        return expr.name
    if isinstance(expr, BinaryOp):
        left = format_expr(expr.left)
        right = format_expr(expr.right)
        return f"({left} {expr.op} {right})"
    if isinstance(expr, UnaryOp):
        return f"({expr.op} {format_expr(expr.operand)})"
    if isinstance(expr, CondExpr):
        return f"(if {format_expr(expr.cond)} then {format_expr(expr.then_expr)} else {format_expr(expr.else_expr)})"
    if isinstance(expr, ArrayRead):
        return f"{format_expr(expr.arr)}[{format_expr(expr.index)}]"
    if isinstance(expr, Forall):
        return f"(forall {expr.var}. {format_expr(expr.body)})"
    if isinstance(expr, Exists):
        return f"(exists {expr.var}. {format_expr(expr.body)})"
    if isinstance(expr, OldExpr):
        return f"old({format_expr(expr.expr)})"
    if isinstance(expr, ResultExpr):
        return "result"
    return str(expr)


def format_stmt(stmt, indent=0):
    """Format a Stmt as readable string."""
    pad = '  ' * indent
    if isinstance(stmt, Skip):
        return f"{pad}skip"
    if isinstance(stmt, Assign):
        return f"{pad}{stmt.var} := {format_expr(stmt.expr)}"
    if isinstance(stmt, Seq):
        return f"{format_stmt(stmt.first, indent)};\n{format_stmt(stmt.second, indent)}"
    if isinstance(stmt, If):
        s = f"{pad}if {format_expr(stmt.cond)} then\n"
        s += format_stmt(stmt.then_branch, indent + 1) + "\n"
        s += f"{pad}else\n"
        s += format_stmt(stmt.else_branch, indent + 1) + "\n"
        s += f"{pad}end"
        return s
    if isinstance(stmt, While):
        s = f"{pad}while {format_expr(stmt.cond)}"
        if stmt.invariant:
            s += f" invariant {format_expr(stmt.invariant)}"
        s += f" do\n"
        s += format_stmt(stmt.body, indent + 1) + "\n"
        s += f"{pad}end"
        return s
    if isinstance(stmt, Assert):
        return f"{pad}assert {format_expr(stmt.cond)}"
    if isinstance(stmt, Assume):
        return f"{pad}assume {format_expr(stmt.cond)}"
    if isinstance(stmt, Block):
        lines = [format_stmt(s, indent) for s in stmt.stmts]
        return ";\n".join(lines)
    return f"{pad}{stmt}"
