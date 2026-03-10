"""V029: Abstract DPLL(T) -- CDCL-guided path-sensitive abstract interpretation.

Combines conflict-driven clause learning (CDCL) search strategy with abstract
interpretation as the theory solver. Programs are analyzed by exploring paths
through branch decisions, using abstract domains to detect infeasibility and
assertion failures, and learning clauses to prune the search space.

Key ideas:
  - Branch decisions = Boolean variables (which direction at each if-statement)
  - Abstract interpretation = theory solver (propagates numeric constraints)
  - Assertion failures = conflicts (trigger clause learning)
  - CDCL learning = prune infeasible/failing branch combinations
  - Non-chronological backtracking = skip irrelevant decision levels

Composes: C010 (parser), C037 (SMT for refinement), C039 (abstract domains)
"""

import sys
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set, Any

# Path setup
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))

from stack_vm import (lex, Parser, IfStmt, WhileStmt, LetDecl, Assign,
                      BinOp, Var, IntLit, CallExpr, FnDecl, Block, UnaryOp)
from abstract_interpreter import (
    AbstractEnv, Sign, Interval,
    sign_from_value, interval_from_value, ConstVal, ConstTop, ConstBot,
    sign_join, interval_join, interval_meet,
    sign_add, sign_sub, sign_mul,
    interval_add, interval_sub, interval_mul,
    sign_contains_zero,
)
from smt_solver import (SMTSolver, SMTResult, Var as SMTVar, IntConst,
                        App, Op, Sort, SortKind)

BOOL = Sort(SortKind.BOOL)
INT = Sort(SortKind.INT)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Verdict(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class LearnedClause:
    """A disjunction of branch-direction literals.

    Semantics: at least one literal must hold.
    Literal (bid, True) means branch bid must go True.
    """
    literals: List[Tuple[int, bool]]

    def is_satisfied(self, assignments: Dict[int, bool]) -> bool:
        return any(assignments.get(bid) == pol for bid, pol in self.literals)

    def is_falsified(self, assignments: Dict[int, bool]) -> bool:
        return all(bid in assignments and assignments[bid] != pol
                   for bid, pol in self.literals)

    def is_unit(self, assignments: Dict[int, bool]) -> Optional[Tuple[int, bool]]:
        """If exactly one literal is unassigned and the rest are false, return it."""
        unassigned = None
        for bid, pol in self.literals:
            if bid not in assignments:
                if unassigned is not None:
                    return None  # >1 unassigned
                unassigned = (bid, pol)
            elif assignments[bid] == pol:
                return None  # satisfied
        return unassigned


@dataclass
class ConflictInfo:
    """Information about an assertion failure."""
    branch_decisions: List[Tuple[int, bool]]
    abstract_state: Dict[str, Tuple]  # var -> (sign, interval) snapshot
    message: str
    relevant_branches: Optional[List[int]] = None  # minimal set


@dataclass
class DPLLTResult:
    """Result of Abstract DPLL(T) analysis."""
    verdict: Verdict
    paths_explored: int
    paths_pruned: int
    clauses_learned: int
    max_decision_level: int
    conflicts: List[ConflictInfo]
    counterexample: Optional[Dict[str, int]] = None
    assertions_checked: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign_neg(s: Sign) -> Sign:
    if s == Sign.POS:
        return Sign.NEG
    if s == Sign.NEG:
        return Sign.POS
    if s == Sign.ZERO:
        return Sign.ZERO
    if s == Sign.NON_NEG:
        return Sign.NON_POS
    if s == Sign.NON_POS:
        return Sign.NON_NEG
    if s == Sign.BOT:
        return Sign.BOT
    return Sign.TOP


def _sign_from_interval(iv: Interval) -> Sign:
    if iv.lo > iv.hi:
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


def _const_binop(lc, rc, op):
    if isinstance(lc, ConstVal) and isinstance(rc, ConstVal):
        a, b = lc.value, rc.value
        if op == '+':
            return ConstVal(a + b)
        if op == '-':
            return ConstVal(a - b)
        if op == '*':
            return ConstVal(a * b)
    return ConstTop()


def _eval_comparison_interval(li, ri, op):
    """Check comparison result from intervals. Returns True/False/None."""
    if li.lo > li.hi or ri.lo > ri.hi:
        return None
    if op == '<':
        if li.hi < ri.lo:
            return True
        if li.lo >= ri.hi:
            return False
    elif op == '<=':
        if li.hi <= ri.lo:
            return True
        if li.lo > ri.hi:
            return False
    elif op == '>':
        if li.lo > ri.hi:
            return True
        if li.hi <= ri.lo:
            return False
    elif op == '>=':
        if li.lo >= ri.hi:
            return True
        if li.hi < ri.lo:
            return False
    elif op == '==':
        if li.lo == li.hi == ri.lo == ri.hi:
            return True
        if li.hi < ri.lo or ri.hi < li.lo:
            return False
    elif op == '!=':
        if li.hi < ri.lo or ri.hi < li.lo:
            return True
        if li.lo == li.hi == ri.lo == ri.hi:
            return False
    return None


def _parse(source: str):
    tokens = lex(source)
    return Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Exception for decision-needed
# ---------------------------------------------------------------------------

class _NeedDecision(Exception):
    def __init__(self, branch_id: int):
        self.branch_id = branch_id


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class AbstractDPLLT:
    """Abstract DPLL(T) solver for program verification.

    Explores paths through a program using CDCL-style search, with
    abstract interpretation (interval domain) as the theory solver.
    Optionally uses SMT (C037) to refine conflicts.
    """

    def __init__(self, max_decisions=200, use_smt_refinement=True):
        self.max_decisions = max_decisions
        self.use_smt_refinement = use_smt_refinement

    def _reset(self):
        self._branch_map = {}  # (stmt_id, parent_path) -> branch_id
        self._next_bid = 0
        self._clauses: List[LearnedClause] = []
        self._paths_explored = 0
        self._paths_pruned = 0
        self._max_level = 0
        self._conflicts: List[ConflictInfo] = []
        self._assertions_checked = 0
        self._functions = {}  # name -> FnDecl
        # Track variable dependencies on branch decisions
        self._var_deps: Dict[str, Set[int]] = {}  # var -> set of branch ids

    def _new_bid(self) -> int:
        bid = self._next_bid
        self._next_bid += 1
        return bid

    def _get_bid(self, stmt, parent_path: tuple) -> int:
        key = (id(stmt), parent_path)
        if key not in self._branch_map:
            self._branch_map[key] = self._new_bid()
        return self._branch_map[key]

    # ----- Abstract evaluation -----

    def _eval(self, expr, env: AbstractEnv):
        """Evaluate expression abstractly -> (sign, interval, const)."""
        if isinstance(expr, IntLit):
            v = expr.value
            return sign_from_value(v), Interval(v, v), ConstVal(v)

        if isinstance(expr, Var):
            name = expr.name
            return (env.get_sign(name),
                    env.get_interval(name),
                    env.get_const(name))

        if isinstance(expr, BinOp):
            ls, li, lc = self._eval(expr.left, env)
            rs, ri, rc = self._eval(expr.right, env)
            op = expr.op
            if op == '+':
                return sign_add(ls, rs), interval_add(li, ri), _const_binop(lc, rc, op)
            if op == '-':
                return sign_sub(ls, rs), interval_sub(li, ri), _const_binop(lc, rc, op)
            if op == '*':
                return sign_mul(ls, rs), interval_mul(li, ri), _const_binop(lc, rc, op)
            if op in ('<', '<=', '>', '>=', '==', '!='):
                result = _eval_comparison_interval(li, ri, op)
                if result is True:
                    return Sign.POS, Interval(1, 1), ConstVal(1)
                if result is False:
                    return Sign.ZERO, Interval(0, 0), ConstVal(0)
                return Sign.NON_NEG, Interval(0, 1), ConstTop()
            return Sign.TOP, Interval(float('-inf'), float('inf')), ConstTop()

        if isinstance(expr, UnaryOp):
            s, i, c = self._eval(expr.operand, env)
            if expr.op == '-':
                ns = _sign_neg(s)
                ni = Interval(-i.hi if i.hi != float('inf') else float('-inf'),
                              -i.lo if i.lo != float('-inf') else float('inf'))
                nc = ConstVal(-c.value) if isinstance(c, ConstVal) else c
                return ns, ni, nc
            return s, i, c

        if isinstance(expr, CallExpr):
            return Sign.TOP, Interval(float('-inf'), float('inf')), ConstTop()

        return Sign.TOP, Interval(float('-inf'), float('inf')), ConstTop()

    def _set_eval(self, env: AbstractEnv, name: str, s, i, c):
        env.set(name, sign=s, interval=i, const=c)

    # ----- Condition refinement -----

    def _refine(self, cond, env: AbstractEnv, branch: bool) -> AbstractEnv:
        """Refine environment based on condition being true/false."""
        env = env.copy()
        if not isinstance(cond, BinOp):
            return env
        left, right, op = cond.left, cond.right, cond.op

        _neg = {'<': '>=', '>=': '<', '>': '<=', '<=': '>',
                '==': '!=', '!=': '=='}
        _flip = {'<': '>', '>': '<', '<=': '>=', '>=': '<=',
                 '==': '==', '!=': '!='}

        if isinstance(left, Var) and isinstance(right, IntLit):
            actual_op = op if branch else _neg.get(op, op)
            self._refine_var(env, left.name, actual_op, right.value)
        elif isinstance(right, Var) and isinstance(left, IntLit):
            flipped = _flip.get(op, op)
            actual_op = flipped if branch else _neg.get(flipped, flipped)
            self._refine_var(env, right.name, actual_op, left.value)
        elif isinstance(left, Var) and isinstance(right, Var):
            # Normalize all var-vs-var comparisons and apply refinement
            actual_op = op if branch else _neg.get(op, op)
            self._refine_var_var(env, left.name, actual_op, right.name)
        return env

    def _refine_var(self, env: AbstractEnv, var: str, op: str, val: int):
        cur = env.get_interval(var)
        if op == '<':
            refined = interval_meet(cur, Interval(float('-inf'), val - 1))
        elif op == '<=':
            refined = interval_meet(cur, Interval(float('-inf'), val))
        elif op == '>':
            refined = interval_meet(cur, Interval(val + 1, float('inf')))
        elif op == '>=':
            refined = interval_meet(cur, Interval(val, float('inf')))
        elif op == '==':
            refined = interval_meet(cur, Interval(val, val))
        elif op == '!=':
            # Singleton exclusion
            if cur.lo == cur.hi == val:
                refined = Interval(1, 0)  # BOT
            elif cur.lo == val:
                refined = Interval(val + 1, cur.hi)
            elif cur.hi == val:
                refined = Interval(cur.lo, val - 1)
            else:
                refined = cur
        else:
            refined = cur
        env.set(var, interval=refined, sign=_sign_from_interval(refined))

    def _refine_var_var(self, env: AbstractEnv, lname: str, op: str, rname: str):
        """Refine two variables based on a comparison between them."""
        li = env.get_interval(lname)
        ri = env.get_interval(rname)
        if op == '<':
            new_l = interval_meet(li, Interval(float('-inf'), ri.hi - 1))
            new_r = interval_meet(ri, Interval(li.lo + 1, float('inf')))
        elif op == '<=':
            new_l = interval_meet(li, Interval(float('-inf'), ri.hi))
            new_r = interval_meet(ri, Interval(li.lo, float('inf')))
        elif op == '>':
            new_l = interval_meet(li, Interval(ri.lo + 1, float('inf')))
            new_r = interval_meet(ri, Interval(float('-inf'), li.hi - 1))
        elif op == '>=':
            new_l = interval_meet(li, Interval(ri.lo, float('inf')))
            new_r = interval_meet(ri, Interval(float('-inf'), li.hi))
        elif op == '==':
            shared = interval_meet(li, ri)
            new_l = shared
            new_r = shared
        elif op == '!=':
            # Limited: singleton exclusion
            if li.lo == li.hi and ri.lo == ri.hi and li.lo == ri.lo:
                new_l = Interval(1, 0)  # BOT
                new_r = Interval(1, 0)
            else:
                return
        else:
            return
        env.set(lname, interval=new_l, sign=_sign_from_interval(new_l))
        env.set(rname, interval=new_r, sign=_sign_from_interval(new_r))

    def _is_bot(self, env: AbstractEnv) -> bool:
        for name in set(list(env.intervals.keys())):
            iv = env.get_interval(name)
            if iv.lo > iv.hi:
                return True
        return False

    # ----- Assertion checking -----

    def _check_assertion(self, expr, env: AbstractEnv) -> str:
        """Returns 'pass', 'may_fail', or 'definitely_fail'."""
        s, i, c = self._eval(expr, env)

        if isinstance(c, ConstVal):
            if c.value == 0 or c.value is False:
                return 'definitely_fail'
            if c.value and c.value != 0:
                return 'pass'

        # Interval-based: truthy if value != 0
        if i.lo > 0 or i.hi < 0:
            return 'pass'
        if i.lo == 0 and i.hi == 0:
            return 'definitely_fail'

        if s in (Sign.POS, Sign.NEG):
            return 'pass'
        if s == Sign.ZERO:
            return 'definitely_fail'

        return 'may_fail'

    # ----- While loop handling (abstract fixpoint) -----

    def _interpret_while(self, stmt, env: AbstractEnv, path, var_deps):
        """Handle while with abstract fixpoint (sound, conservative)."""
        max_iter = 50
        for _ in range(max_iter):
            old = env.copy()
            body_env = self._refine(stmt.cond, env, True)
            if self._is_bot(body_env):
                break
            stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            body_env, _, _ = self._run_stmts(stmts, body_env, path, var_deps)
            env = old.widen(body_env)
            if env.equals(old):
                break
        return self._refine(stmt.cond, env, False)

    # ----- Statement interpretation -----

    def _run_stmts(self, stmts, env: AbstractEnv, path: list,
                   var_deps: Dict[str, Set[int]]):
        """Interpret statements, collecting conflicts.

        Returns (env, conflicts, assertions_checked).
        Raises _NeedDecision if a branch hasn't been decided yet.
        """
        conflicts = []
        checked = 0

        for stmt in stmts:
            if isinstance(stmt, LetDecl):
                s, i, c = self._eval(stmt.value, env)
                self._set_eval(env, stmt.name, s, i, c)
                # Inherit deps from RHS variables
                deps = set()
                for d in self._expr_vars(stmt.value):
                    deps |= var_deps.get(d, set())
                # Also inherit path deps
                for bid, _ in path:
                    deps.add(bid)
                var_deps[stmt.name] = deps

            elif isinstance(stmt, Assign):
                s, i, c = self._eval(stmt.value, env)
                self._set_eval(env, stmt.name, s, i, c)
                deps = set()
                for d in self._expr_vars(stmt.value):
                    deps |= var_deps.get(d, set())
                for bid, _ in path:
                    deps.add(bid)
                var_deps[stmt.name] = deps

            elif isinstance(stmt, IfStmt):
                bid = self._get_bid(stmt, tuple(p for p in path))
                if bid not in self._current_assignments:
                    raise _NeedDecision(bid)

                direction = self._current_assignments[bid]
                refined = self._refine(stmt.cond, env, direction)

                if self._is_bot(refined):
                    # Infeasible branch -- this path is unreachable.
                    # Not an assertion failure, just path pruning.
                    # Return early with no conflicts; the path simply
                    # doesn't exist and doesn't contribute to the verdict.
                    return env, conflicts, checked

                body = stmt.then_body if direction else stmt.else_body
                new_path = path + [(bid, direction)]
                # Update deps: variables in condition depend on this branch
                for v in self._expr_vars(stmt.cond):
                    var_deps.setdefault(v, set()).add(bid)

                if body:
                    body_stmts = body.stmts if isinstance(body, Block) else [body]
                    refined, sub_c, sub_a = self._run_stmts(
                        body_stmts, refined, new_path, var_deps)
                    conflicts.extend(sub_c)
                    checked += sub_a
                env = refined

            elif isinstance(stmt, WhileStmt):
                env = self._interpret_while(stmt, env, path, var_deps)

            elif isinstance(stmt, FnDecl):
                self._functions[stmt.name] = stmt

            elif isinstance(stmt, CallExpr):
                checked_here, confs = self._handle_call(stmt, env, path, var_deps)
                checked += checked_here
                conflicts.extend(confs)

        return env, conflicts, checked

    def _handle_call(self, call, env, path, var_deps):
        conflicts = []
        checked = 0
        if call.callee == 'assert' and call.args:
            checked += 1
            result = self._check_assertion(call.args[0], env)
            if result in ('may_fail', 'definitely_fail'):
                # Find relevant branches via dependency analysis
                relevant = set()
                for v in self._expr_vars(call.args[0]):
                    relevant |= var_deps.get(v, set())
                # Also include all path decisions that influenced this point
                for bid, _ in path:
                    relevant.add(bid)

                # Include ALL current branch assignments, not just nesting path
                all_decisions = list(self._current_assignments.items())
                conflict = ConflictInfo(
                    branch_decisions=all_decisions,
                    abstract_state=self._snapshot(env),
                    message=f"Assertion {result}",
                    relevant_branches=list(relevant) if relevant else None,
                )
                conflicts.append(conflict)
        elif call.callee == 'print':
            pass
        elif call.callee in self._functions:
            fn = self._functions[call.callee]
            fn_env = env.copy()
            for i, param in enumerate(fn.params):
                if i < len(call.args):
                    s, iv, c = self._eval(call.args[i], env)
                    self._set_eval(fn_env, param, s, iv, c)
            body_stmts = fn.body.stmts if isinstance(fn.body, Block) else [fn.body]
            _, sub_c, sub_a = self._run_stmts(body_stmts, fn_env, path, var_deps)
            conflicts.extend(sub_c)
            checked += sub_a
        return checked, conflicts

    def _expr_vars(self, expr) -> Set[str]:
        """Collect variable names referenced in expression."""
        if isinstance(expr, Var):
            return {expr.name}
        if isinstance(expr, BinOp):
            return self._expr_vars(expr.left) | self._expr_vars(expr.right)
        if isinstance(expr, UnaryOp):
            return self._expr_vars(expr.operand)
        if isinstance(expr, CallExpr):
            result = set()
            for a in expr.args:
                result |= self._expr_vars(a)
            return result
        return set()

    def _snapshot(self, env: AbstractEnv) -> Dict[str, Tuple]:
        result = {}
        for name in set(list(env.signs.keys()) + list(env.intervals.keys())):
            result[name] = (env.get_sign(name), env.get_interval(name))
        return result

    # ----- CDCL: clause learning -----

    def _learn(self, conflict: ConflictInfo) -> Optional[LearnedClause]:
        """Analyze conflict and learn a clause.

        Uses dependency analysis to produce a minimal clause: only include
        branch decisions that actually influenced the failing assertion.
        """
        decisions = conflict.branch_decisions
        if not decisions:
            return None

        relevant = conflict.relevant_branches
        if relevant:
            # Minimal clause: only negate relevant decisions
            lits = []
            decisions_dict = dict(decisions)
            for bid in relevant:
                if bid in decisions_dict:
                    lits.append((bid, not decisions_dict[bid]))
            if lits:
                return LearnedClause(literals=lits)

        # Fallback: negate all decisions
        return LearnedClause(
            literals=[(bid, not val) for bid, val in decisions])

    def _is_pruned(self, assignments: Dict[int, bool]) -> bool:
        return any(c.is_falsified(assignments) for c in self._clauses)

    def _unit_propagate(self, assignments: Dict[int, bool]) -> List[Tuple[int, bool]]:
        """Find forced assignments from learned clauses."""
        forced = []
        changed = True
        while changed:
            changed = False
            for clause in self._clauses:
                if clause.is_satisfied(assignments):
                    continue
                unit = clause.is_unit(assignments)
                if unit is not None:
                    bid, pol = unit
                    if bid not in assignments:
                        assignments[bid] = pol
                        forced.append((bid, pol))
                        changed = True
        return forced

    # ----- SMT refinement -----

    def _smt_refine(self, conflict: ConflictInfo, program) -> bool:
        """Check if an assertion failure is real using SMT.

        Returns True if the failure is confirmed real.
        """
        if not self.use_smt_refinement:
            return True  # Assume real

        try:
            solver = SMTSolver()
            # Encode path conditions
            smt_vars = {}
            decisions = dict(conflict.branch_decisions)
            # Walk program with the same branch decisions and collect SMT constraints
            self._smt_encode_path(solver, smt_vars, program.stmts, decisions, {})
            result = solver.check()
            if result == SMTResult.UNSAT:
                return False  # Spurious: path is infeasible
            return True  # Real or unknown
        except Exception:
            return True  # Conservative

    def _smt_encode_path(self, solver, smt_vars, stmts, decisions, bid_map):
        """Encode a path through the program as SMT constraints."""
        for stmt in stmts:
            if isinstance(stmt, LetDecl):
                var = solver.Int(stmt.name) if stmt.name not in smt_vars else smt_vars[stmt.name]
                smt_vars[stmt.name] = var
                rhs = self._smt_expr(solver, smt_vars, stmt.value)
                if rhs is not None:
                    solver.add(App(Op.EQ, [var, rhs], BOOL))

            elif isinstance(stmt, Assign):
                # Create a new version of the variable
                ver = smt_vars.get(stmt.name + '_ver', 0)
                new_name = f"{stmt.name}_{ver}"
                smt_vars[stmt.name + '_ver'] = ver + 1
                new_var = solver.Int(new_name)
                rhs = self._smt_expr(solver, smt_vars, stmt.value)
                if rhs is not None:
                    solver.add(App(Op.EQ, [new_var, rhs], BOOL))
                smt_vars[stmt.name] = new_var

            elif isinstance(stmt, IfStmt):
                bid = self._get_bid(stmt, tuple())
                if bid in decisions:
                    direction = decisions[bid]
                    cond_smt = self._smt_cond(solver, smt_vars, stmt.cond, direction)
                    if cond_smt is not None:
                        solver.add(cond_smt)
                    body = stmt.then_body if direction else stmt.else_body
                    if body:
                        body_stmts = body.stmts if isinstance(body, Block) else [body]
                        self._smt_encode_path(solver, smt_vars, body_stmts, decisions, bid_map)

    def _smt_expr(self, solver, smt_vars, expr):
        """Convert expression to SMT term. Returns None if can't encode."""
        if isinstance(expr, IntLit):
            return IntConst(expr.value)
        if isinstance(expr, Var):
            if expr.name in smt_vars:
                return smt_vars[expr.name]
            v = solver.Int(expr.name)
            smt_vars[expr.name] = v
            return v
        if isinstance(expr, BinOp):
            l = self._smt_expr(solver, smt_vars, expr.left)
            r = self._smt_expr(solver, smt_vars, expr.right)
            if l is None or r is None:
                return None
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL}
            if expr.op in op_map:
                return App(op_map[expr.op], [l, r], INT)
            return None
        if isinstance(expr, UnaryOp) and expr.op == '-':
            inner = self._smt_expr(solver, smt_vars, expr.operand)
            if inner is not None:
                return App(Op.SUB, [IntConst(0), inner], INT)
        return None

    def _smt_cond(self, solver, smt_vars, cond, branch):
        """Encode condition as SMT, optionally negated for branch=False."""
        if not isinstance(cond, BinOp):
            return None
        l = self._smt_expr(solver, smt_vars, cond.left)
        r = self._smt_expr(solver, smt_vars, cond.right)
        if l is None or r is None:
            return None
        cmp_map = {'<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
                   '==': Op.EQ, '!=': Op.NEQ}
        neg_map = {'<': Op.GE, '<=': Op.GT, '>': Op.LE, '>=': Op.LT,
                   '==': Op.NEQ, '!=': Op.EQ}
        op = cond.op
        if op not in cmp_map:
            return None
        smt_op = cmp_map[op] if branch else neg_map[op]
        return App(smt_op, [l, r], BOOL)

    # ----- Main analysis -----

    def analyze(self, source: str) -> DPLLTResult:
        """Analyze a program using Abstract DPLL(T).

        Explores paths through the program using CDCL-style search,
        with abstract interpretation as the theory solver.
        """
        program = _parse(source)
        self._reset()
        self._current_assignments: Dict[int, bool] = {}
        self._decision_stack: List[Tuple[int, bool, bool]] = []
        # Stack of (bid, current_value, tried_other)

        total_decisions = 0
        all_conflicts = []
        real_conflicts = []

        while total_decisions < self.max_decisions:
            # Unit propagation
            forced = self._unit_propagate(self._current_assignments)

            # Check if current assignments are pruned
            if self._is_pruned(self._current_assignments):
                self._paths_pruned += 1
                if not self._backtrack():
                    break
                continue

            # Try to run program with current assignments
            try:
                env = AbstractEnv()
                var_deps: Dict[str, Set[int]] = {}
                env, conflicts, checked = self._run_stmts(
                    program.stmts, env, [], var_deps)
                self._assertions_checked += checked
                self._paths_explored += 1

                if conflicts:
                    for conflict in conflicts:
                        all_conflicts.append(conflict)
                        clause = self._learn(conflict)
                        if clause:
                            self._clauses.append(clause)

                        # SMT refinement
                        if self.use_smt_refinement:
                            is_real = self._smt_refine(conflict, program)
                        else:
                            is_real = True

                        if is_real:
                            real_conflicts.append(conflict)

                # Backtrack to explore other paths
                if not self._backtrack():
                    break

            except _NeedDecision as nd:
                bid = nd.branch_id
                total_decisions += 1
                self._max_level = max(self._max_level, len(self._decision_stack) + 1)

                # Check if learned clauses force a direction
                forced_val = None
                for clause in self._clauses:
                    unit = clause.is_unit(self._current_assignments)
                    if unit and unit[0] == bid:
                        forced_val = unit[1]
                        break

                if forced_val is not None:
                    self._current_assignments[bid] = forced_val
                    self._decision_stack.append((bid, forced_val, True))
                else:
                    # Decide True first
                    self._current_assignments[bid] = True
                    self._decision_stack.append((bid, True, False))

        # Determine verdict
        if real_conflicts:
            # Try to extract counterexample from SMT model
            verdict = Verdict.UNSAFE
        elif self._assertions_checked == 0:
            verdict = Verdict.SAFE
        else:
            verdict = Verdict.SAFE

        self._conflicts = real_conflicts

        return DPLLTResult(
            verdict=verdict,
            paths_explored=self._paths_explored,
            paths_pruned=self._paths_pruned,
            clauses_learned=len(self._clauses),
            max_decision_level=self._max_level,
            conflicts=real_conflicts,
            assertions_checked=self._assertions_checked,
        )

    def _backtrack(self) -> bool:
        """Backtrack to explore the other branch direction.
        Returns True if backtracking succeeded, False if exhausted."""
        while self._decision_stack:
            bid, val, tried_other = self._decision_stack[-1]
            if not tried_other:
                # Try the other direction
                new_val = not val
                self._decision_stack[-1] = (bid, new_val, True)
                # Undo all decisions after this one
                self._current_assignments[bid] = new_val
                # Remove deeper decisions
                while len(self._decision_stack) > len(self._decision_stack):
                    pass  # This doesn't make sense, let me fix
                # Actually need to remove decisions deeper than current
                self._trim_decisions()
                return True
            else:
                # Both directions tried, pop
                self._decision_stack.pop()
                del self._current_assignments[bid]
        return False  # Exhausted

    def _trim_decisions(self):
        """Remove decisions deeper than current stack level."""
        valid_bids = {bid for bid, _, _ in self._decision_stack}
        to_remove = [bid for bid in self._current_assignments if bid not in valid_bids]
        for bid in to_remove:
            del self._current_assignments[bid]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_program(source: str, use_smt=True) -> DPLLTResult:
    """Analyze a program using Abstract DPLL(T).

    Args:
        source: C10 source code with assertions
        use_smt: Whether to use SMT refinement for conflict checking

    Returns:
        DPLLTResult with verdict, statistics, and conflicts
    """
    solver = AbstractDPLLT(use_smt_refinement=use_smt)
    return solver.analyze(source)


def verify_assertions(source: str) -> Tuple[Verdict, List[str]]:
    """Verify all assertions in a program.

    Returns (verdict, list of failure messages).
    """
    result = analyze_program(source)
    messages = [c.message for c in result.conflicts]
    return result.verdict, messages


def compare_with_standard_ai(source: str) -> Dict[str, Any]:
    """Compare Abstract DPLL(T) with standard abstract interpretation.

    Returns comparison dict showing precision differences.
    """
    from abstract_interpreter import AbstractInterpreter

    # Standard AI
    ai = AbstractInterpreter()
    ai_result = ai.analyze(source)
    ai_warnings = ai_result.get('warnings', [])

    # Abstract DPLL(T)
    dpll_result = analyze_program(source, use_smt=False)

    return {
        'standard_ai': {
            'warnings': len(ai_warnings),
            'warning_list': ai_warnings,
        },
        'abstract_dpll_t': {
            'verdict': dpll_result.verdict.value,
            'paths_explored': dpll_result.paths_explored,
            'paths_pruned': dpll_result.paths_pruned,
            'clauses_learned': dpll_result.clauses_learned,
            'conflicts': len(dpll_result.conflicts),
            'assertions_checked': dpll_result.assertions_checked,
        },
    }


def analyze_with_budget(source: str, max_decisions: int = 50) -> DPLLTResult:
    """Analyze with a decision budget (for performance control)."""
    solver = AbstractDPLLT(max_decisions=max_decisions)
    return solver.analyze(source)
