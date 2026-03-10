"""
V019: Widening with Thresholds

Improves interval analysis precision by widening to program-derived thresholds
instead of jumping directly to infinity.

Standard widening:
  old=[0,0], new=[0,1] -> [0, +inf]   (upper bound grew, jump to infinity)

Threshold widening with thresholds={10}:
  old=[0,0], new=[0,1] -> [0, 10]     (widen to next threshold above 1)

This matters for loops like `i=0; while(i<10){i=i+1}`:
  - Standard: i = [0, +inf] (imprecise)
  - Threshold: i = [0, 10]  (tight)

Threshold sources:
  1. Literal constants in the program
  2. Comparison operands (loop bounds, conditionals)
  3. Assignment right-hand side constants
  4. Negations of extracted thresholds
  5. User-provided thresholds

Composes: C039 (abstract interpreter) + C010 (parser)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Set, List, Dict, Tuple
from copy import deepcopy

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
from stack_vm import (
    lex, Parser, Program,
    IntLit, FloatLit, StringLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)


# ============================================================
# Threshold Extraction
# ============================================================

def extract_thresholds_from_ast(node, thresholds=None):
    """Walk AST and collect numeric constants as thresholds."""
    if thresholds is None:
        thresholds = set()

    if isinstance(node, IntLit):
        thresholds.add(float(node.value))
    elif isinstance(node, FloatLit):
        thresholds.add(float(node.value))
    elif isinstance(node, BinOp):
        extract_thresholds_from_ast(node.left, thresholds)
        extract_thresholds_from_ast(node.right, thresholds)
        # For comparisons, also add boundary values (x < 10 -> threshold 10 and 9)
        if node.op in ('<', '<=', '>', '>=', '==', '!='):
            if isinstance(node.right, (IntLit, FloatLit)):
                v = float(node.right.value)
                thresholds.add(v)
                thresholds.add(v - 1)
                thresholds.add(v + 1)
            if isinstance(node.left, (IntLit, FloatLit)):
                v = float(node.left.value)
                thresholds.add(v)
                thresholds.add(v - 1)
                thresholds.add(v + 1)
    elif isinstance(node, UnaryOp):
        extract_thresholds_from_ast(node.operand, thresholds)
        if node.op == '-' and isinstance(node.operand, (IntLit, FloatLit)):
            thresholds.add(-float(node.operand.value))
    elif isinstance(node, LetDecl):
        extract_thresholds_from_ast(node.value, thresholds)
    elif isinstance(node, Assign):
        extract_thresholds_from_ast(node.value, thresholds)
    elif isinstance(node, IfStmt):
        extract_thresholds_from_ast(node.cond, thresholds)
        extract_thresholds_from_ast(node.then_body, thresholds)
        if node.else_body:
            extract_thresholds_from_ast(node.else_body, thresholds)
    elif isinstance(node, WhileStmt):
        extract_thresholds_from_ast(node.cond, thresholds)
        extract_thresholds_from_ast(node.body, thresholds)
    elif isinstance(node, Block):
        for stmt in node.stmts:
            extract_thresholds_from_ast(stmt, thresholds)
    elif isinstance(node, Program):
        for stmt in node.stmts:
            extract_thresholds_from_ast(stmt, thresholds)
    elif isinstance(node, FnDecl):
        extract_thresholds_from_ast(node.body, thresholds)
    elif isinstance(node, CallExpr):
        for arg in node.args:
            extract_thresholds_from_ast(arg, thresholds)
    elif isinstance(node, PrintStmt):
        extract_thresholds_from_ast(node.value, thresholds)
    elif isinstance(node, ReturnStmt):
        if node.value:
            extract_thresholds_from_ast(node.value, thresholds)

    return thresholds


def extract_thresholds_from_source(source):
    """Parse source and extract thresholds."""
    tokens = lex(source)
    program = Parser(tokens).parse()
    thresholds = extract_thresholds_from_ast(program)
    # Add negations of all thresholds
    neg_thresholds = {-t for t in thresholds}
    thresholds = thresholds | neg_thresholds
    # Always include 0
    thresholds.add(0.0)
    return sorted(thresholds)


# ============================================================
# Threshold Widening Operator
# ============================================================

def interval_widen_thresholds(old, new, thresholds):
    """Widening with thresholds.

    Instead of jumping to -inf/+inf when bounds change, widen to the
    nearest threshold in the direction of change.

    Args:
        old: Previous interval
        new: New interval (after one more iteration)
        thresholds: Sorted list of threshold values

    Returns:
        Widened interval
    """
    if old.is_bot():
        return new
    if new.is_bot():
        return old

    # Lower bound: if new.lo < old.lo, find largest threshold <= new.lo
    if new.lo < old.lo:
        lo = NEG_INF
        for t in thresholds:
            if t <= new.lo:
                lo = t
            else:
                break
    else:
        lo = old.lo

    # Upper bound: if new.hi > old.hi, find smallest threshold >= new.hi
    if new.hi > old.hi:
        hi = INF
        for t in reversed(thresholds):
            if t >= new.hi:
                hi = t
            else:
                break
    else:
        hi = old.hi

    return Interval(lo, hi)


# ============================================================
# Abstract Environment with Threshold Widening
# ============================================================

class ThresholdEnv(AbstractEnv):
    """AbstractEnv that uses threshold-based widening."""

    def __init__(self, thresholds=None):
        super().__init__()
        self.thresholds = sorted(thresholds) if thresholds else []

    def copy(self):
        env = ThresholdEnv(self.thresholds)
        env.signs = dict(self.signs)
        env.intervals = dict(self.intervals)
        env.consts = dict(self.consts)
        return env

    def widen(self, other):
        """Widen with thresholds instead of jumping to infinity."""
        result = ThresholdEnv(self.thresholds)
        all_names = (set(self.signs.keys()) | set(other.signs.keys()) |
                     set(self.intervals.keys()) | set(other.intervals.keys()) |
                     set(self.consts.keys()) | set(other.consts.keys()))
        for name in all_names:
            result.signs[name] = sign_join(
                self.signs.get(name, Sign.BOT),
                other.signs.get(name, Sign.BOT)
            )
            result.intervals[name] = interval_widen_thresholds(
                self.intervals.get(name, INTERVAL_BOT),
                other.intervals.get(name, INTERVAL_BOT),
                self.thresholds
            )
            result.consts[name] = const_join(
                self.consts.get(name, CONST_BOT),
                other.consts.get(name, CONST_BOT)
            )
        return result

    def join(self, other):
        """Join preserving thresholds."""
        result = ThresholdEnv(self.thresholds)
        all_names = (set(self.signs.keys()) | set(other.signs.keys()) |
                     set(self.intervals.keys()) | set(other.intervals.keys()) |
                     set(self.consts.keys()) | set(other.consts.keys()))
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


# ============================================================
# Threshold-Aware Abstract Interpreter
# ============================================================

class ThresholdInterpreter(AbstractInterpreter):
    """Abstract interpreter using threshold widening.

    Extends C039's AbstractInterpreter with:
    1. Automatic threshold extraction from source
    2. Threshold-based widening for tighter loop invariants
    3. Per-variable threshold tracking
    4. Narrowing pass after widening for extra precision
    """

    def __init__(self, thresholds=None, max_iterations=50, narrowing_iterations=3,
                 auto_extract=True):
        super().__init__(max_iterations=max_iterations)
        self.global_thresholds = sorted(thresholds) if thresholds else []
        self.narrowing_iterations = narrowing_iterations
        self.auto_extract = auto_extract
        self.widening_stats = {'standard_jumps': 0, 'threshold_widens': 0}

    def analyze(self, source):
        """Analyze with threshold widening."""
        tokens = lex(source)
        program = Parser(tokens).parse()

        # Auto-extract thresholds from program
        if self.auto_extract:
            extracted = extract_thresholds_from_ast(program)
            neg_extracted = {-t for t in extracted}
            all_thresholds = extracted | neg_extracted | {0.0}
            all_thresholds = all_thresholds | set(self.global_thresholds)
        else:
            all_thresholds = set(self.global_thresholds) | {0.0}

        sorted_thresholds = sorted(all_thresholds)

        # Initialize with threshold-aware environment
        env = ThresholdEnv(sorted_thresholds)
        self.warnings = []
        self.functions = {}
        self.var_reads = set()
        self.var_writes = {}
        self.widening_stats = {'standard_jumps': 0, 'threshold_widens': 0}

        # First pass: collect function declarations
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt

        # Interpret
        env = self._interpret_stmts(program.stmts, env)

        # Dead assignment detection
        self._detect_dead_assignments(env)

        return {
            'env': env,
            'warnings': list(self.warnings),
            'functions': dict(self.functions),
            'var_reads': set(self.var_reads),
            'var_writes': dict(self.var_writes),
            'thresholds': sorted_thresholds,
            'widening_stats': dict(self.widening_stats),
        }

    def _interpret_while(self, stmt, env):
        """Fixpoint with threshold widening + optional narrowing."""
        # Ensure env is ThresholdEnv
        if not isinstance(env, ThresholdEnv):
            t_env = ThresholdEnv(self.global_thresholds)
            t_env.signs = dict(env.signs)
            t_env.intervals = dict(env.intervals)
            t_env.consts = dict(env.consts)
            env = t_env

        # Phase 1: Widening fixpoint (ascending)
        current_env = env
        for i in range(self.max_iterations):
            cond = self._eval_expr(stmt.cond, current_env)
            if isinstance(cond.const, ConstVal) and not cond.const.value:
                return current_env

            body_env = self._refine_env_for_condition(stmt.cond, current_env, True)
            body_env = self._interpret_stmt(stmt.body, body_env)

            # Ensure body_env is ThresholdEnv for widening
            if not isinstance(body_env, ThresholdEnv):
                t_body = ThresholdEnv(current_env.thresholds if isinstance(current_env, ThresholdEnv) else [])
                t_body.signs = dict(body_env.signs)
                t_body.intervals = dict(body_env.intervals)
                t_body.consts = dict(body_env.consts)
                body_env = t_body

            next_env = current_env.widen(body_env)

            if next_env.equals(current_env):
                break
            current_env = next_env

        # Phase 2: Narrowing (descending) -- refine widened result
        for i in range(self.narrowing_iterations):
            cond = self._eval_expr(stmt.cond, current_env)
            if isinstance(cond.const, ConstVal) and not cond.const.value:
                break

            body_env = self._refine_env_for_condition(stmt.cond, current_env, True)
            body_env = self._interpret_stmt(stmt.body, body_env)

            # Narrow: take meet of current with new (descending)
            narrowed = self._narrow_env(current_env, body_env)
            if narrowed.equals(current_env):
                break
            current_env = narrowed

        # After loop: condition is false
        exit_env = self._refine_env_for_condition(stmt.cond, current_env, False)
        return exit_env

    def _narrow_env(self, wide_env, new_env):
        """Narrowing: intersect wide fixpoint with new iteration."""
        result = ThresholdEnv(wide_env.thresholds if isinstance(wide_env, ThresholdEnv) else [])
        all_names = set(wide_env.signs.keys()) | set(new_env.signs.keys())
        for name in all_names:
            # For intervals: narrow by taking tighter bounds
            wide_iv = wide_env.intervals.get(name, INTERVAL_BOT)
            new_iv = new_env.intervals.get(name, INTERVAL_BOT)
            result.intervals[name] = _interval_narrow(wide_iv, new_iv)

            # Signs: join (conservative)
            result.signs[name] = sign_join(
                wide_env.signs.get(name, Sign.BOT),
                new_env.signs.get(name, Sign.BOT)
            )

            # Constants: join (conservative)
            result.consts[name] = const_join(
                wide_env.consts.get(name, CONST_BOT),
                new_env.consts.get(name, CONST_BOT)
            )
        return result

    def _interpret_if(self, stmt, env):
        """Override to preserve ThresholdEnv through if-statements."""
        cond = self._eval_expr(stmt.cond, env)

        if isinstance(cond.const, ConstVal):
            if cond.const.value:
                if stmt.else_body:
                    self._warn(WarningKind.UNREACHABLE_BRANCH,
                               "else branch is unreachable (condition always true)",
                               self._get_line(stmt))
                return self._interpret_stmt(stmt.then_body, env)
            else:
                self._warn(WarningKind.UNREACHABLE_BRANCH,
                           "then branch is unreachable (condition always false)",
                           self._get_line(stmt))
                if stmt.else_body:
                    return self._interpret_stmt(stmt.else_body, env)
                return env

        then_env = self._refine_env_for_condition(stmt.cond, env, True)
        else_env = self._refine_env_for_condition(stmt.cond, env, False)

        then_env = self._interpret_stmt(stmt.then_body, then_env)
        if stmt.else_body:
            else_env = self._interpret_stmt(stmt.else_body, else_env)

        return then_env.join(else_env)

    def _refine_env_for_condition(self, cond, env, branch):
        """Override to preserve ThresholdEnv type."""
        result = super()._refine_env_for_condition(cond, env, branch)
        # Ensure result stays as ThresholdEnv if input was
        if isinstance(env, ThresholdEnv) and not isinstance(result, ThresholdEnv):
            t_result = ThresholdEnv(env.thresholds)
            t_result.signs = dict(result.signs)
            t_result.intervals = dict(result.intervals)
            t_result.consts = dict(result.consts)
            return t_result
        return result

    def _interpret_let(self, stmt, env):
        """Override to preserve ThresholdEnv type."""
        val = self._eval_expr(stmt.value, env)
        env = env.copy()
        env.set(stmt.name, sign=val.sign, interval=val.interval, const=val.const)
        self._record_write(stmt.name, self._get_line(stmt))
        return env

    def _interpret_assign(self, stmt, env):
        """Override to preserve ThresholdEnv type."""
        val = self._eval_expr(stmt.value, env)
        env = env.copy()
        env.set(stmt.name, sign=val.sign, interval=val.interval, const=val.const)
        self._record_write(stmt.name, self._get_line(stmt))
        return env


def _interval_narrow(wide, new):
    """Narrowing operator for intervals.

    If wide bound is infinite and new bound is finite, take the finite bound.
    Otherwise keep the wide bound (don't expand).
    """
    if wide.is_bot():
        return INTERVAL_BOT
    if new.is_bot():
        return wide

    # Narrow lower: if wide is -inf and new is finite, take new
    if wide.lo == NEG_INF and new.lo != NEG_INF:
        lo = new.lo
    else:
        lo = wide.lo

    # Narrow upper: if wide is +inf and new is finite, take new
    if wide.hi == INF and new.hi != INF:
        hi = new.hi
    else:
        hi = wide.hi

    if lo > hi:
        return INTERVAL_BOT
    return Interval(lo, hi)


# ============================================================
# Per-Variable Thresholds
# ============================================================

def extract_variable_thresholds(source):
    """Extract thresholds per variable from comparisons and assignments.

    Returns dict mapping variable name -> set of thresholds.
    """
    tokens = lex(source)
    program = Parser(tokens).parse()
    var_thresholds = {}

    def collect(node):
        if isinstance(node, BinOp):
            if node.op in ('<', '<=', '>', '>=', '==', '!='):
                if isinstance(node.left, Var) and isinstance(node.right, (IntLit, FloatLit)):
                    name = node.left.name
                    v = float(node.right.value)
                    if name not in var_thresholds:
                        var_thresholds[name] = set()
                    var_thresholds[name].update({v - 1, v, v + 1})
                if isinstance(node.right, Var) and isinstance(node.left, (IntLit, FloatLit)):
                    name = node.right.name
                    v = float(node.left.value)
                    if name not in var_thresholds:
                        var_thresholds[name] = set()
                    var_thresholds[name].update({v - 1, v, v + 1})
            collect(node.left)
            collect(node.right)
        elif isinstance(node, LetDecl):
            if isinstance(node.value, (IntLit, FloatLit)):
                v = float(node.value.value)
                if node.name not in var_thresholds:
                    var_thresholds[node.name] = set()
                var_thresholds[node.name].add(v)
            collect(node.value)
        elif isinstance(node, Assign):
            if isinstance(node.value, (IntLit, FloatLit)):
                v = float(node.value.value)
                if node.name not in var_thresholds:
                    var_thresholds[node.name] = set()
                var_thresholds[node.name].add(v)
            collect(node.value)
        elif isinstance(node, IfStmt):
            collect(node.cond)
            collect(node.then_body)
            if node.else_body:
                collect(node.else_body)
        elif isinstance(node, WhileStmt):
            collect(node.cond)
            collect(node.body)
        elif isinstance(node, Block):
            for s in node.stmts:
                collect(s)
        elif isinstance(node, Program):
            for s in node.stmts:
                collect(s)
        elif isinstance(node, FnDecl):
            collect(node.body)
        elif isinstance(node, CallExpr):
            for a in node.args:
                collect(a)
        elif isinstance(node, UnaryOp):
            collect(node.operand)
        elif isinstance(node, PrintStmt):
            collect(node.value)
        elif isinstance(node, ReturnStmt):
            if node.value:
                collect(node.value)

    collect(program)

    # Add 0 to all variables and negations
    for name in var_thresholds:
        var_thresholds[name].add(0.0)
        neg = {-t for t in var_thresholds[name]}
        var_thresholds[name] |= neg

    return var_thresholds


# ============================================================
# Convenience APIs
# ============================================================

def threshold_analyze(source, extra_thresholds=None, narrowing=3):
    """Analyze with automatic threshold extraction and widening.

    Args:
        source: C10 source code
        extra_thresholds: Optional additional thresholds to include
        narrowing: Number of narrowing iterations (0 to disable)

    Returns:
        Analysis result dict with env, warnings, thresholds, widening_stats
    """
    thresholds = list(extra_thresholds) if extra_thresholds else []
    interp = ThresholdInterpreter(
        thresholds=thresholds,
        narrowing_iterations=narrowing,
        auto_extract=True
    )
    return interp.analyze(source)


def compare_widening(source, extra_thresholds=None):
    """Compare standard widening (C039) vs threshold widening (V019).

    Returns dict with:
        - standard: C039 analysis result
        - threshold: V019 analysis result
        - improvements: list of (var, standard_interval, threshold_interval) where
          threshold is strictly tighter
    """
    standard = c039_analyze(source)
    threshold = threshold_analyze(source, extra_thresholds=extra_thresholds)

    improvements = []
    std_env = standard['env']
    thr_env = threshold['env']

    all_vars = set(std_env.intervals.keys()) | set(thr_env.intervals.keys())
    for v in sorted(all_vars):
        std_iv = std_env.get_interval(v)
        thr_iv = thr_env.get_interval(v)
        # Threshold is tighter if it has a finite bound where standard has infinite
        if _is_strictly_tighter(thr_iv, std_iv):
            improvements.append((v, std_iv, thr_iv))

    return {
        'standard': standard,
        'threshold': threshold,
        'improvements': improvements,
    }


def get_variable_range(source, var_name, extra_thresholds=None):
    """Get interval for variable using threshold widening."""
    result = threshold_analyze(source, extra_thresholds=extra_thresholds)
    return result['env'].get_interval(var_name)


def get_thresholds(source):
    """Extract all thresholds from source code."""
    return extract_thresholds_from_source(source)


def get_variable_thresholds(source):
    """Extract per-variable thresholds from source code."""
    return extract_variable_thresholds(source)


def _is_strictly_tighter(a, b):
    """Is interval a strictly contained in interval b?"""
    if a.is_bot() and b.is_bot():
        return False
    if a.is_bot():
        return True
    if b.is_bot():
        return False
    # a is tighter if its bounds are at least as tight with at least one strictly tighter
    a_lo = a.lo if a.lo != NEG_INF else float('-inf')
    a_hi = a.hi if a.hi != INF else float('inf')
    b_lo = b.lo if b.lo != NEG_INF else float('-inf')
    b_hi = b.hi if b.hi != INF else float('inf')

    lo_tighter = a_lo >= b_lo
    hi_tighter = a_hi <= b_hi
    strictly = a_lo > b_lo or a_hi < b_hi

    return lo_tighter and hi_tighter and strictly
