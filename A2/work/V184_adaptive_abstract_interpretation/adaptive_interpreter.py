"""
V184: Adaptive Abstract Interpretation
Composes V179 (domain hierarchy) + C039 (abstract interpreter) + C010 (parser)

An abstract interpreter that automatically selects the optimal abstract domain
per program point based on precision demands. Starts cheap (sign/interval),
promotes to relational domains (zone/octagon/polyhedra) only where needed.

Key innovations:
1. Per-point domain tracking: each program point maintains its own domain level
2. Demand-driven promotion: relational assignments/guards trigger upgrade
3. Convergence-driven promotion: if widening loses too much, escalate
4. Cost-aware: tracks analysis cost at each level
5. Precision diagnostics: reports where promotions happened and why

Architecture:
  Source -> C010 Parser -> AST -> AdaptiveInterpreter -> Results
  Each program point gets an AdaptiveEnv (wraps V179 domains)
  Promotion triggers: relational ops, widening precision loss, guard failures
"""

import sys, os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import IntEnum, auto
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V179_abstract_domain_hierarchy'))

from stack_vm import lex, Parser, IntLit, Var, BinOp, Assign, LetDecl, Block, IfStmt, WhileStmt, FnDecl, ReturnStmt
try:
    from stack_vm import FloatLit, BoolLit, StringLit, UnaryOp, PrintStmt, CallExpr
except ImportError:
    FloatLit = BoolLit = StringLit = UnaryOp = PrintStmt = CallExpr = None

from abstract_interpreter import (
    Sign, Interval, INTERVAL_TOP, INTERVAL_BOT, INF, NEG_INF,
    sign_join, sign_add, sign_sub, sign_mul, sign_div, sign_neg,
    sign_contains_zero,
    interval_join, interval_meet, interval_widen,
    interval_add, interval_sub, interval_mul, interval_div, interval_neg,
    ConstVal, ConstTop, ConstBot, CONST_TOP, CONST_BOT, const_join,
    AbstractValue, AbstractEnv, Warning, WarningKind
)

from domain_hierarchy import (
    DomainLevel, LinearConstraint, DomainHierarchy,
    sign_domain, interval_domain, zone_domain, octagon_domain, polyhedra_domain,
    adaptive_domain as create_adaptive_domain
)


# ============================================================
# Promotion Reasons
# ============================================================

class PromotionReason(IntEnum):
    RELATIONAL_ASSIGN = 1      # x = y + z (need relational domain)
    RELATIONAL_GUARD = 2       # if (x < y) (need difference constraints)
    WIDENING_LOSS = 3          # widening jumped to infinity, try higher domain
    EXPLICIT_REQUEST = 4       # user/analysis requested specific level
    CONSTRAINT_DEMAND = 5      # constraint requires higher domain


@dataclass(frozen=True)
class PromotionEvent:
    """Records when and why a domain promotion occurred."""
    line: int
    variable: str
    from_level: DomainLevel
    to_level: DomainLevel
    reason: PromotionReason
    detail: str = ""


# ============================================================
# Adaptive Environment
# ============================================================

class AdaptiveEnv:
    """
    Abstract environment with per-variable domain level tracking.

    Each variable has:
    - An AbstractValue (sign + interval + const) from C039
    - A domain level indicating the precision tier
    - Relational constraints tracked at the env level via V179 domain
    """

    def __init__(self, max_level=DomainLevel.POLYHEDRA):
        self._abs_env = AbstractEnv()  # C039 non-relational values
        self._var_levels: Dict[str, DomainLevel] = {}  # per-var domain level
        self._relational = None  # V179 domain for relational constraints (lazy)
        self._rel_level = DomainLevel.INTERVAL  # current relational domain level
        self._max_level = max_level
        self._promotions: List[PromotionEvent] = []
        self._relational_vars: Set[str] = set()  # vars needing relational tracking

    def copy(self):
        e = AdaptiveEnv(self._max_level)
        e._abs_env = self._abs_env.copy()
        e._var_levels = dict(self._var_levels)
        e._rel_level = self._rel_level
        e._relational_vars = set(self._relational_vars)
        e._promotions = list(self._promotions)
        if self._relational is not None:
            # Deep copy relational domain
            e._relational = self._copy_relational()
        return e

    def _copy_relational(self):
        """Copy the relational domain by extracting and re-creating."""
        if self._relational is None:
            return None
        constraints = self._relational.extract_constraints()
        dom = DomainHierarchy.create(self._rel_level, constraints)
        return dom

    def _ensure_relational(self, level=None):
        """Ensure relational domain exists at given level."""
        target = level or DomainLevel.ZONE
        if target <= DomainLevel.INTERVAL:
            return
        if target > self._max_level:
            target = self._max_level
        if self._relational is None:
            self._relational = DomainHierarchy.create(target)
            self._rel_level = target
        elif target > self._rel_level:
            # Promote relational domain
            constraints = self._relational.extract_constraints()
            self._relational = DomainHierarchy.create(target, constraints)
            self._rel_level = target

    def _promote_relational(self, reason, line=0, var="", detail=""):
        """Promote relational domain to next level."""
        if self._rel_level >= self._max_level:
            return False
        old_level = self._rel_level
        new_level = DomainLevel(min(self._rel_level + 1, self._max_level))
        self._ensure_relational(new_level)
        self._promotions.append(PromotionEvent(
            line=line, variable=var,
            from_level=old_level, to_level=new_level,
            reason=reason, detail=detail
        ))
        return True

    def set_var(self, name, value, line=0):
        """Set a variable to an abstract value."""
        self._abs_env.set(
            name,
            sign=value.sign,
            interval=value.interval,
            const=value.const
        )
        if name not in self._var_levels:
            self._var_levels[name] = DomainLevel.INTERVAL
        # Forget old relational constraints for this variable before adding new
        if self._relational is not None and name in self._relational_vars:
            try:
                self._relational = self._relational.forget(name)
            except Exception:
                pass
            iv = value.interval
            if iv.lo != NEG_INF or iv.hi != INF:
                self._update_relational_bounds(name, iv)

    def _update_relational_bounds(self, name, iv):
        """Push interval bounds into relational domain."""
        try:
            if iv.lo != NEG_INF:
                c = LinearConstraint.var_ge(name, iv.lo)
                self._relational = self._relational.guard(c)
            if iv.hi != INF:
                c = LinearConstraint.var_le(name, iv.hi)
                self._relational = self._relational.guard(c)
        except Exception:
            pass  # relational domain may reject

    def set_relational(self, name, expr_vars, line=0):
        """
        Mark that 'name' has a relational dependency on expr_vars.
        Promotes to zone/octagon if needed.
        """
        if len(expr_vars) == 0:
            return

        self._relational_vars.add(name)
        for v in expr_vars:
            self._relational_vars.add(v)

        # Need at least zone for difference constraints
        if self._max_level < DomainLevel.ZONE:
            return  # can't promote, max level too low
        if self._rel_level < DomainLevel.ZONE:
            self._ensure_relational(DomainLevel.ZONE)
            self._promotions.append(PromotionEvent(
                line=line, variable=name,
                from_level=DomainLevel.INTERVAL, to_level=DomainLevel.ZONE,
                reason=PromotionReason.RELATIONAL_ASSIGN,
                detail=f"{name} depends on {expr_vars}"
            ))

    def add_constraint(self, constraint, line=0):
        """Add a relational constraint, promoting if needed."""
        needed = constraint.classify()
        if needed > self._max_level:
            needed = self._max_level
        if needed > DomainLevel.INTERVAL:
            if self._rel_level < needed:
                old = self._rel_level
                self._ensure_relational(needed)
                self._promotions.append(PromotionEvent(
                    line=line, variable="",
                    from_level=old, to_level=needed,
                    reason=PromotionReason.CONSTRAINT_DEMAND,
                    detail=str(constraint)
                ))
            else:
                self._ensure_relational(needed)
            try:
                self._relational = self._relational.guard(constraint)
            except Exception:
                pass

    def get_sign(self, name):
        return self._abs_env.get_sign(name)

    def get_interval(self, name):
        return self._abs_env.get_interval(name)

    def get_const(self, name):
        return self._abs_env.get_const(name)

    def get_bounds(self, name):
        """Get tightest bounds from both non-relational and relational."""
        iv = self._abs_env.get_interval(name)
        lo, hi = iv.lo, iv.hi

        if self._relational is not None and name in self._relational_vars:
            try:
                rlo, rhi = self._relational.get_bounds(name)
                if rlo is not None and (lo == NEG_INF or rlo > lo):
                    lo = rlo
                if rhi is not None and (hi == INF or rhi < hi):
                    hi = rhi
            except Exception:
                pass

        return Interval(lo, hi)

    def get_relational_constraints(self, var=None):
        """Get relational constraints, optionally filtered by variable."""
        if self._relational is None:
            return []
        try:
            constraints = self._relational.extract_constraints()
            if var:
                return [c for c in constraints if var in c.coeffs]
            return constraints
        except Exception:
            return []

    @property
    def variables(self):
        return set(self._var_levels.keys())

    @property
    def promotions(self):
        return list(self._promotions)

    @property
    def relational_level(self):
        return self._rel_level

    def join(self, other):
        """Join two adaptive environments (LUB)."""
        result = AdaptiveEnv(self._max_level)
        result._abs_env = self._abs_env.join(other._abs_env)

        # Merge variable levels (take max)
        all_vars = set(self._var_levels) | set(other._var_levels)
        for v in all_vars:
            l1 = self._var_levels.get(v, DomainLevel.INTERVAL)
            l2 = other._var_levels.get(v, DomainLevel.INTERVAL)
            result._var_levels[v] = max(l1, l2)

        # Merge relational domains
        result._relational_vars = self._relational_vars | other._relational_vars
        result._rel_level = max(self._rel_level, other._rel_level)

        if self._relational is not None and other._relational is not None:
            try:
                # Promote both to same level before join
                r1 = self._relational
                r2 = other._relational
                if self._rel_level < result._rel_level:
                    c1 = r1.extract_constraints()
                    r1 = DomainHierarchy.create(result._rel_level, c1)
                if other._rel_level < result._rel_level:
                    c2 = r2.extract_constraints()
                    r2 = DomainHierarchy.create(result._rel_level, c2)
                result._relational = r1.join(r2)
            except Exception:
                result._relational = None
        elif self._relational is not None:
            result._relational = self._copy_relational()
        elif other._relational is not None:
            result._relational = other._copy_relational()

        # Merge promotions
        result._promotions = self._promotions + other._promotions

        return result

    def widen(self, other):
        """Widen this environment with other (for loop convergence)."""
        result = AdaptiveEnv(self._max_level)
        result._abs_env = self._abs_env.widen(other._abs_env)

        all_vars = set(self._var_levels) | set(other._var_levels)
        for v in all_vars:
            l1 = self._var_levels.get(v, DomainLevel.INTERVAL)
            l2 = other._var_levels.get(v, DomainLevel.INTERVAL)
            result._var_levels[v] = max(l1, l2)

        result._relational_vars = self._relational_vars | other._relational_vars
        result._rel_level = max(self._rel_level, other._rel_level)

        if self._relational is not None and other._relational is not None:
            try:
                r1 = self._relational
                r2 = other._relational
                if self._rel_level < result._rel_level:
                    c1 = r1.extract_constraints()
                    r1 = DomainHierarchy.create(result._rel_level, c1)
                if other._rel_level < result._rel_level:
                    c2 = r2.extract_constraints()
                    r2 = DomainHierarchy.create(result._rel_level, c2)
                result._relational = r1.widen(r2)
            except Exception:
                result._relational = None
        elif self._relational is not None:
            result._relational = self._copy_relational()
        elif other._relational is not None:
            result._relational = other._copy_relational()

        result._promotions = list(self._promotions)
        return result

    def equals(self, other):
        """Check if two environments are equal."""
        if not self._abs_env.equals(other._abs_env):
            return False
        if self._rel_level != other._rel_level:
            return False
        # Check relational equality
        if self._relational is None and other._relational is None:
            return True
        if self._relational is None or other._relational is None:
            return False
        try:
            return self._relational.equals(other._relational)
        except Exception:
            return False


# ============================================================
# Adaptive Abstract Interpreter
# ============================================================

class AdaptiveInterpreter:
    """
    Abstract interpreter that auto-selects domain precision per program point.

    Features:
    - Non-relational analysis (sign + interval + const) everywhere
    - Relational analysis (zone/octagon/polyhedra) only where needed
    - Demand-driven promotion: relational ops trigger upgrade
    - Convergence-driven promotion: escalate if widening loses too much
    - Per-point precision diagnostics
    """

    def __init__(self, max_level=DomainLevel.POLYHEDRA, max_iterations=50,
                 promote_on_widening_loss=True, promotion_threshold=2):
        self.max_level = max_level
        self.max_iterations = max_iterations
        self.promote_on_widening_loss = promote_on_widening_loss
        self.promotion_threshold = promotion_threshold  # consecutive widenings before promote
        self.warnings: List[Warning] = []
        self.promotions: List[PromotionEvent] = []
        self.var_reads: Set[str] = set()
        self.var_writes: Set[str] = set()
        self.loop_iterations: Dict[int, int] = {}  # line -> iteration count
        self.domain_costs: Dict[str, int] = {}  # level_name -> operation count

    def analyze(self, source):
        """Analyze a program, returning results dict."""
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()

        self.warnings = []
        self.promotions = []
        self.var_reads = set()
        self.var_writes = set()
        self.loop_iterations = {}
        self.domain_costs = {}

        env = AdaptiveEnv(self.max_level)
        env = self._interpret(ast, env)

        # Collect promotions from env
        self.promotions.extend(env.promotions)

        # Check dead assignments
        for v in self.var_writes:
            if v not in self.var_reads:
                self.warnings.append(Warning(
                    kind=WarningKind.DEAD_ASSIGNMENT,
                    message=f"Variable '{v}' is assigned but never read"
                ))

        return {
            'env': env,
            'warnings': self.warnings,
            'promotions': self.promotions,
            'loop_iterations': self.loop_iterations,
            'domain_costs': self.domain_costs,
            'var_reads': self.var_reads,
            'var_writes': self.var_writes,
        }

    def _interpret(self, node, env):
        """Interpret an AST node, returning updated environment."""
        if node is None:
            return env

        name = type(node).__name__

        if name == 'Program':
            for stmt in node.stmts:
                env = self._interpret(stmt, env)
            return env

        if name == 'Block':
            for stmt in node.stmts:
                env = self._interpret(stmt, env)
            return env

        if name == 'LetDecl':
            return self._interpret_let(node, env)

        if name == 'Assign':
            return self._interpret_assign(node, env)

        if name == 'IfStmt':
            return self._interpret_if(node, env)

        if name == 'WhileStmt':
            return self._interpret_while(node, env)

        if name == 'FnDecl':
            # Just track function existence
            return env

        if name == 'ReturnStmt':
            if node.value:
                self._eval_expr(node.value, env)
            return env

        if name == 'PrintStmt' or (PrintStmt and isinstance(node, PrintStmt)):
            if hasattr(node, 'value') and node.value:
                self._eval_expr(node.value, env)
            elif hasattr(node, 'expr') and node.expr:
                self._eval_expr(node.expr, env)
            return env

        # Expression statement
        if hasattr(node, '__class__') and name in ('CallExpr', 'BinOp', 'Var', 'IntLit'):
            self._eval_expr(node, env)
            return env

        return env

    def _interpret_let(self, node, env):
        """Interpret let declaration."""
        line = getattr(node, 'line', 0)
        name = node.name
        self.var_writes.add(name)

        val, expr_vars = self._eval_expr_with_deps(node.value, env)
        env.set_var(name, val, line)

        # If RHS involves other variables, track relational dependency
        if len(expr_vars) > 0:
            env.set_relational(name, expr_vars, line)
            self._add_relational_from_expr(env, name, node.value, line)

        self._track_cost('assign')
        return env

    def _interpret_assign(self, node, env):
        """Interpret assignment."""
        line = getattr(node, 'line', 0)
        name = node.name
        self.var_writes.add(name)

        val, expr_vars = self._eval_expr_with_deps(node.value, env)
        env.set_var(name, val, line)

        if len(expr_vars) > 0:
            env.set_relational(name, expr_vars, line)
            self._add_relational_from_expr(env, name, node.value, line)

        self._track_cost('assign')
        return env

    def _add_relational_from_expr(self, env, target, expr, line):
        """Extract and add relational constraints from assignment target = expr."""
        # Pattern: x = y + c or x = y - c -> difference constraint
        if isinstance(expr, BinOp):
            left_var = self._get_var_name(expr.left)
            right_var = self._get_var_name(expr.right)
            left_const = self._get_const_value(expr.right)
            right_const = self._get_const_value(expr.left)

            op = expr.op

            # x = y + c -> x - y = c -> x - y <= c AND y - x <= -c
            if op == '+' and left_var and left_const is not None:
                c = Fraction(left_const)
                env.add_constraint(LinearConstraint.diff_le(target, left_var, c), line)
                env.add_constraint(LinearConstraint.diff_le(left_var, target, -c), line)

            # x = y - c -> x - y = -c
            elif op == '-' and left_var and left_const is not None:
                c = Fraction(left_const)
                env.add_constraint(LinearConstraint.diff_le(target, left_var, -c), line)
                env.add_constraint(LinearConstraint.diff_le(left_var, target, c), line)

            # x = c + y -> same as x = y + c
            elif op == '+' and right_var and right_const is not None:
                c = Fraction(right_const)
                env.add_constraint(LinearConstraint.diff_le(target, right_var, c), line)
                env.add_constraint(LinearConstraint.diff_le(right_var, target, -c), line)

            # x = y + z -> need octagon (sum constraint)
            elif op == '+' and left_var and right_var:
                if env._rel_level < DomainLevel.OCTAGON and env._max_level >= DomainLevel.OCTAGON:
                    env._promote_relational(
                        PromotionReason.RELATIONAL_ASSIGN, line, target,
                        f"{target} = {left_var} + {right_var} needs sum constraints"
                    )
                # x = y + z -> x - y - z = 0 (polyhedra level)
                if env._rel_level >= DomainLevel.POLYHEDRA:
                    try:
                        # x <= y + z and x >= y + z
                        c1 = LinearConstraint({target: Fraction(1), left_var: Fraction(-1), right_var: Fraction(-1)}, Fraction(0))
                        c2 = LinearConstraint({target: Fraction(-1), left_var: Fraction(1), right_var: Fraction(1)}, Fraction(0))
                        env.add_constraint(c1, line)
                        env.add_constraint(c2, line)
                    except Exception:
                        pass

            # x = y - z -> difference constraint (zone level)
            elif op == '-' and left_var and right_var:
                # x = y - z -> x - y + z = 0
                # At zone level: x - y <= 0 (if z >= 0) -- imprecise
                # At polyhedra: exact
                if env._rel_level >= DomainLevel.POLYHEDRA:
                    try:
                        c1 = LinearConstraint({target: Fraction(1), left_var: Fraction(-1), right_var: Fraction(1)}, Fraction(0))
                        c2 = LinearConstraint({target: Fraction(-1), left_var: Fraction(1), right_var: Fraction(-1)}, Fraction(0))
                        env.add_constraint(c1, line)
                        env.add_constraint(c2, line)
                    except Exception:
                        pass

        # Simple copy: x = y -> x - y = 0
        elif isinstance(expr, Var):
            src = expr.name
            self.var_reads.add(src)
            env.add_constraint(LinearConstraint.diff_le(target, src, Fraction(0)), line)
            env.add_constraint(LinearConstraint.diff_le(src, target, Fraction(0)), line)

    def _interpret_if(self, node, env):
        """Interpret if/else statement with guard refinement."""
        line = getattr(node, 'line', 0)

        # Evaluate condition for warnings
        cond_val = self._eval_expr(node.cond, env)

        # Try to refine environments per branch
        then_env = env.copy()
        else_env = env.copy()

        self._refine_from_condition(then_env, node.cond, True, line)
        self._refine_from_condition(else_env, node.cond, False, line)

        # Check for unreachable branches
        cond_sign = cond_val.sign
        if cond_val.const and isinstance(cond_val.const, ConstVal):
            if cond_val.const.value is True or (isinstance(cond_val.const.value, int) and cond_val.const.value != 0):
                self.warnings.append(Warning(
                    kind=WarningKind.UNREACHABLE_BRANCH,
                    message="Else branch is unreachable (condition always true)",
                    line=line
                ))
            elif cond_val.const.value is False or cond_val.const.value == 0:
                self.warnings.append(Warning(
                    kind=WarningKind.UNREACHABLE_BRANCH,
                    message="Then branch is unreachable (condition always false)",
                    line=line
                ))

        then_env = self._interpret(node.then_body, then_env)

        if node.else_body:
            else_env = self._interpret(node.else_body, else_env)
            result = then_env.join(else_env)
        else:
            result = then_env.join(else_env)

        self._track_cost('branch')
        return result

    def _refine_from_condition(self, env, cond, is_true, line):
        """Refine environment based on condition being true/false."""
        if not isinstance(cond, BinOp):
            return

        op = cond.op
        left_var = self._get_var_name(cond.left)
        right_var = self._get_var_name(cond.right)
        left_const = self._get_const_value(cond.left)
        right_const = self._get_const_value(cond.right)

        # Comparison operators for refinement
        if op in ('<', '<=', '>', '>=', '==', '!='):
            self._refine_comparison(env, op, cond.left, cond.right,
                                     left_var, right_var, left_const, right_const,
                                     is_true, line)

    def _refine_comparison(self, env, op, left, right,
                           left_var, right_var, left_const, right_const,
                           is_true, line):
        """Refine environment from a comparison."""
        # Negate if branch is false
        if not is_true:
            neg_map = {'<': '>=', '<=': '>', '>': '<=', '>=': '<', '==': '!=', '!=': '=='}
            op = neg_map.get(op, op)

        # var < const -> var <= const - 1 (integer)
        if left_var and right_const is not None:
            c = Fraction(right_const)
            if op == '<':
                iv = env.get_interval(left_var)
                new_hi = min(iv.hi, c - 1) if iv.hi != INF else c - 1
                env._abs_env.set(left_var, interval=Interval(iv.lo, new_hi))
            elif op == '<=':
                iv = env.get_interval(left_var)
                new_hi = min(iv.hi, c) if iv.hi != INF else c
                env._abs_env.set(left_var, interval=Interval(iv.lo, new_hi))
            elif op == '>':
                iv = env.get_interval(left_var)
                new_lo = max(iv.lo, c + 1) if iv.lo != NEG_INF else c + 1
                env._abs_env.set(left_var, interval=Interval(new_lo, iv.hi))
            elif op == '>=':
                iv = env.get_interval(left_var)
                new_lo = max(iv.lo, c) if iv.lo != NEG_INF else c
                env._abs_env.set(left_var, interval=Interval(new_lo, iv.hi))
            elif op == '==':
                env._abs_env.set(left_var, interval=Interval(c, c))

        # const < var
        elif right_var and left_const is not None:
            c = Fraction(left_const)
            if op == '<':
                iv = env.get_interval(right_var)
                new_lo = max(iv.lo, c + 1) if iv.lo != NEG_INF else c + 1
                env._abs_env.set(right_var, interval=Interval(new_lo, iv.hi))
            elif op == '<=':
                iv = env.get_interval(right_var)
                new_lo = max(iv.lo, c) if iv.lo != NEG_INF else c
                env._abs_env.set(right_var, interval=Interval(new_lo, iv.hi))
            elif op == '>':
                iv = env.get_interval(right_var)
                new_hi = min(iv.hi, c - 1) if iv.hi != INF else c - 1
                env._abs_env.set(right_var, interval=Interval(iv.lo, new_hi))
            elif op == '>=':
                iv = env.get_interval(right_var)
                new_hi = min(iv.hi, c) if iv.hi != INF else c
                env._abs_env.set(right_var, interval=Interval(iv.lo, new_hi))
            elif op == '==':
                env._abs_env.set(right_var, interval=Interval(c, c))

        # var < var -> relational constraint + interval refinement from known bounds
        elif left_var and right_var:
            left_iv = env.get_interval(left_var)
            right_iv = env.get_interval(right_var)
            if op == '<':
                # x < y -> x - y <= -1
                env.add_constraint(LinearConstraint.diff_le(left_var, right_var, Fraction(-1)), line)
                # x < y: x.hi <= y.hi - 1, y.lo >= x.lo + 1
                if right_iv.hi != INF:
                    new_hi = min(left_iv.hi, right_iv.hi - 1) if left_iv.hi != INF else right_iv.hi - 1
                    env._abs_env.set(left_var, interval=Interval(left_iv.lo, new_hi))
                if left_iv.lo != NEG_INF:
                    new_lo = max(right_iv.lo, left_iv.lo + 1) if right_iv.lo != NEG_INF else left_iv.lo + 1
                    env._abs_env.set(right_var, interval=Interval(new_lo, right_iv.hi))
            elif op == '<=':
                # x <= y -> x - y <= 0
                env.add_constraint(LinearConstraint.diff_le(left_var, right_var, Fraction(0)), line)
                if right_iv.hi != INF:
                    new_hi = min(left_iv.hi, right_iv.hi) if left_iv.hi != INF else right_iv.hi
                    env._abs_env.set(left_var, interval=Interval(left_iv.lo, new_hi))
                if left_iv.lo != NEG_INF:
                    new_lo = max(right_iv.lo, left_iv.lo) if right_iv.lo != NEG_INF else left_iv.lo
                    env._abs_env.set(right_var, interval=Interval(new_lo, right_iv.hi))
            elif op == '>':
                # x > y -> y - x <= -1
                env.add_constraint(LinearConstraint.diff_le(right_var, left_var, Fraction(-1)), line)
                if right_iv.lo != NEG_INF:
                    new_lo = max(left_iv.lo, right_iv.lo + 1) if left_iv.lo != NEG_INF else right_iv.lo + 1
                    env._abs_env.set(left_var, interval=Interval(new_lo, left_iv.hi))
                if left_iv.hi != INF:
                    new_hi = min(right_iv.hi, left_iv.hi - 1) if right_iv.hi != INF else left_iv.hi - 1
                    env._abs_env.set(right_var, interval=Interval(right_iv.lo, new_hi))
            elif op == '>=':
                # x >= y -> y - x <= 0
                env.add_constraint(LinearConstraint.diff_le(right_var, left_var, Fraction(0)), line)
                if right_iv.lo != NEG_INF:
                    new_lo = max(left_iv.lo, right_iv.lo) if left_iv.lo != NEG_INF else right_iv.lo
                    env._abs_env.set(left_var, interval=Interval(new_lo, left_iv.hi))
                if left_iv.hi != INF:
                    new_hi = min(right_iv.hi, left_iv.hi) if right_iv.hi != INF else left_iv.hi
                    env._abs_env.set(right_var, interval=Interval(right_iv.lo, new_hi))
            elif op == '==':
                # x == y -> x - y <= 0 AND y - x <= 0
                env.add_constraint(LinearConstraint.diff_le(left_var, right_var, Fraction(0)), line)
                env.add_constraint(LinearConstraint.diff_le(right_var, left_var, Fraction(0)), line)
                # Intersect intervals
                meet_lo = max(left_iv.lo, right_iv.lo) if left_iv.lo != NEG_INF and right_iv.lo != NEG_INF else max(left_iv.lo, right_iv.lo)
                meet_hi = min(left_iv.hi, right_iv.hi) if left_iv.hi != INF and right_iv.hi != INF else min(left_iv.hi, right_iv.hi)
                env._abs_env.set(left_var, interval=Interval(meet_lo, meet_hi))
                env._abs_env.set(right_var, interval=Interval(meet_lo, meet_hi))

    def _interpret_while(self, node, env):
        """Interpret while loop with adaptive widening."""
        line = getattr(node, 'line', 0)

        current = env.copy()
        widening_count = 0
        prev_bounds = {}

        for iteration in range(self.max_iterations):
            # Refine by condition being true
            body_env = current.copy()
            self._refine_from_condition(body_env, node.cond, True, line)

            # Execute body
            body_env = self._interpret(node.body, body_env)

            # Widen
            next_env = current.widen(body_env)

            # Check convergence-driven promotion
            if self.promote_on_widening_loss and iteration > 0:
                lost_precision = self._check_widening_loss(current, next_env, prev_bounds)
                if lost_precision:
                    widening_count += 1
                    if widening_count >= self.promotion_threshold:
                        promoted = next_env._promote_relational(
                            PromotionReason.WIDENING_LOSS, line, "",
                            f"Widening lost precision {widening_count} times"
                        )
                        if promoted:
                            widening_count = 0

            # Record bounds for next iteration comparison
            prev_bounds = {}
            for v in next_env.variables:
                prev_bounds[v] = next_env.get_bounds(v)

            if next_env.equals(current):
                self.loop_iterations[line] = iteration + 1
                break

            current = next_env
        else:
            self.loop_iterations[line] = self.max_iterations

        # Exit: condition is false
        exit_env = current.copy()
        self._refine_from_condition(exit_env, node.cond, False, line)

        self._track_cost('loop')
        return exit_env

    def _check_widening_loss(self, old_env, new_env, prev_bounds):
        """Check if widening lost significant precision."""
        for v in old_env.variables:
            old_iv = old_env.get_bounds(v)
            new_iv = new_env.get_bounds(v)

            # Jumped to infinity = precision loss
            if old_iv.lo != NEG_INF and new_iv.lo == NEG_INF:
                return True
            if old_iv.hi != INF and new_iv.hi == INF:
                return True

        return False

    def _eval_expr(self, node, env):
        """Evaluate expression abstractly, returning AbstractValue."""
        val, _ = self._eval_expr_with_deps(node, env)
        return val

    def _eval_expr_with_deps(self, node, env):
        """Evaluate expression, returning (AbstractValue, set of variable deps)."""
        if node is None:
            return AbstractValue.top(), set()

        name = type(node).__name__

        if name == 'IntLit':
            v = node.value
            return AbstractValue.from_value(v), set()

        if FloatLit and name == 'FloatLit':
            v = node.value
            return AbstractValue.from_value(v), set()

        if BoolLit and name == 'BoolLit':
            v = node.value
            return AbstractValue.from_value(v), set()

        if name == 'Var':
            self.var_reads.add(node.name)
            s = env.get_sign(node.name)
            iv = env.get_bounds(node.name)
            c = env.get_const(node.name)
            return AbstractValue(sign=s, interval=iv, const=c), {node.name}

        if name == 'BinOp':
            left_val, left_deps = self._eval_expr_with_deps(node.left, env)
            right_val, right_deps = self._eval_expr_with_deps(node.right, env)

            result = self._eval_binop(node.op, left_val, right_val)
            deps = left_deps | right_deps

            # Check division by zero
            if node.op in ('/', '%'):
                line = getattr(node, 'line', 0)
                if right_val.sign == Sign.ZERO:
                    self.warnings.append(Warning(
                        kind=WarningKind.DIVISION_BY_ZERO,
                        message="Division by zero",
                        line=line
                    ))
                elif sign_contains_zero(right_val.sign):
                    self.warnings.append(Warning(
                        kind=WarningKind.POSSIBLE_DIVISION_BY_ZERO,
                        message="Possible division by zero",
                        line=line
                    ))

            return result, deps

        if UnaryOp and name == 'UnaryOp':
            inner_val, inner_deps = self._eval_expr_with_deps(node.operand if hasattr(node, 'operand') else node.expr, env)
            op = node.op
            if op == '-':
                s = sign_neg(inner_val.sign)
                iv = interval_neg(inner_val.interval)
                c = CONST_TOP
                if isinstance(inner_val.const, ConstVal):
                    c = ConstVal(-inner_val.const.value)
                return AbstractValue(sign=s, interval=iv, const=c), inner_deps
            elif op == 'not':
                return AbstractValue.top(), inner_deps
            return AbstractValue.top(), inner_deps

        if CallExpr and name == 'CallExpr':
            # Evaluate arguments for side effects
            deps = set()
            if hasattr(node, 'args'):
                for arg in node.args:
                    _, d = self._eval_expr_with_deps(arg, env)
                    deps |= d
            return AbstractValue.top(), deps

        return AbstractValue.top(), set()

    def _eval_binop(self, op, left, right):
        """Evaluate binary operation on abstract values."""
        ls, li, lc = left.sign, left.interval, left.const
        rs, ri, rc = right.sign, right.interval, right.const

        # Sign domain
        sign_ops = {'+': sign_add, '-': sign_sub, '*': sign_mul, '/': sign_div}
        s = sign_ops.get(op, lambda a, b: Sign.TOP)(ls, rs)

        # Interval domain
        iv_ops = {'+': interval_add, '-': interval_sub, '*': interval_mul, '/': interval_div}
        iv = iv_ops.get(op, lambda a, b: INTERVAL_TOP)(li, ri)

        # Constant domain
        c = CONST_TOP
        if isinstance(lc, ConstVal) and isinstance(rc, ConstVal):
            try:
                py_ops = {'+': lambda a, b: a + b, '-': lambda a, b: a - b,
                          '*': lambda a, b: a * b, '/': lambda a, b: a // b if b != 0 else None,
                          '%': lambda a, b: a % b if b != 0 else None,
                          '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
                          '>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
                          '==': lambda a, b: a == b, '!=': lambda a, b: a != b}
                fn = py_ops.get(op)
                if fn:
                    result = fn(lc.value, rc.value)
                    if result is not None:
                        c = ConstVal(result)
            except Exception:
                pass

        # Comparison operators sign
        if op in ('<', '<=', '>', '>=', '==', '!='):
            s = Sign.TOP  # boolean result
            # Try to determine from intervals
            if op == '<' and li.hi < ri.lo:
                c = ConstVal(True)
            elif op == '<' and li.lo >= ri.hi:
                c = ConstVal(False)
            elif op == '<=' and li.hi <= ri.lo:
                c = ConstVal(True)
            elif op == '<=' and li.lo > ri.hi:
                c = ConstVal(False)

            iv = INTERVAL_TOP

        return AbstractValue(sign=s, interval=iv, const=c)

    def _get_var_name(self, node):
        """Extract variable name from AST node, or None."""
        if isinstance(node, Var):
            return node.name
        return None

    def _get_const_value(self, node):
        """Extract constant value from AST node, or None."""
        if isinstance(node, IntLit):
            return node.value
        if FloatLit and isinstance(node, FloatLit):
            return node.value
        return None

    def _track_cost(self, operation):
        """Track analysis cost by operation type."""
        self.domain_costs[operation] = self.domain_costs.get(operation, 0) + 1


# ============================================================
# Comparison Framework
# ============================================================

class DomainComparison:
    """Compare analysis results across different domain configurations."""

    @staticmethod
    def compare_fixed_vs_adaptive(source, variables=None, max_level=DomainLevel.POLYHEDRA):
        """
        Run analysis at each fixed domain level AND with adaptive,
        comparing precision for specified variables.

        Returns dict with bounds per variable per strategy.
        """
        results = {}

        # Fixed-level analyses (interval only, no relational)
        interp_interval = AdaptiveInterpreter(max_level=DomainLevel.INTERVAL)
        r_interval = interp_interval.analyze(source)
        results['interval'] = {
            'env': r_interval['env'],
            'warnings': len(r_interval['warnings']),
            'promotions': 0,
        }

        # Adaptive analyses at different max levels
        for level in [DomainLevel.ZONE, DomainLevel.OCTAGON, DomainLevel.POLYHEDRA]:
            if level > max_level:
                break
            interp = AdaptiveInterpreter(max_level=level)
            r = interp.analyze(source)
            level_name = level.name.lower()
            results[f'adaptive_{level_name}'] = {
                'env': r['env'],
                'warnings': len(r['warnings']),
                'promotions': len(r['promotions']),
            }

        # Extract bounds comparison
        if variables:
            bounds = {}
            for v in variables:
                bounds[v] = {}
                for strategy, data in results.items():
                    env = data['env']
                    b = env.get_bounds(v)
                    bounds[v][strategy] = (b.lo, b.hi)
            results['bounds'] = bounds

        return results

    @staticmethod
    def precision_gain(source, variables):
        """
        Measure precision gained by adaptive analysis over interval-only.
        Returns dict: {var: {strategy: width_reduction_percent}}
        """
        comparison = DomainComparison.compare_fixed_vs_adaptive(source, variables)
        bounds = comparison.get('bounds', {})

        gains = {}
        for var, strats in bounds.items():
            gains[var] = {}
            iv_bounds = strats.get('interval', (NEG_INF, INF))
            iv_width = iv_bounds[1] - iv_bounds[0] if iv_bounds[0] != NEG_INF and iv_bounds[1] != INF else None

            for strat, (lo, hi) in strats.items():
                if strat == 'interval':
                    continue
                width = hi - lo if lo != NEG_INF and hi != INF else None
                if iv_width is not None and width is not None and iv_width > 0:
                    reduction = float((iv_width - width) / iv_width * 100)
                    gains[var][strat] = reduction
                elif iv_width is None and width is not None:
                    gains[var][strat] = float('inf')  # from infinite to finite
                else:
                    gains[var][strat] = 0.0

        return gains


# ============================================================
# Program Point Analysis
# ============================================================

class PointAnalysis:
    """Analyze individual program points for domain selection."""

    @staticmethod
    def classify_program(source):
        """
        Classify a program's domain requirements at each point.
        Returns list of (line, needed_level, reason).
        """
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()

        points = []
        PointAnalysis._classify_node(ast, points)
        return points

    @staticmethod
    def _classify_node(node, points):
        """Recursively classify AST nodes."""
        if node is None:
            return

        name = type(node).__name__
        line = getattr(node, 'line', 0)

        if name == 'Program' or name == 'Block':
            for stmt in node.stmts:
                PointAnalysis._classify_node(stmt, points)
            return

        if name in ('LetDecl', 'Assign'):
            # Check if RHS involves multiple variables
            vars_in_rhs = PointAnalysis._count_vars(node.value)
            if vars_in_rhs >= 2:
                # Check if sum or difference
                if PointAnalysis._has_sum(node.value):
                    points.append((line, DomainLevel.OCTAGON, "sum of variables"))
                else:
                    points.append((line, DomainLevel.ZONE, "difference of variables"))
            elif vars_in_rhs == 1:
                points.append((line, DomainLevel.INTERVAL, "single variable expression"))
            else:
                points.append((line, DomainLevel.INTERVAL, "constant expression"))

        if name == 'IfStmt':
            if PointAnalysis._is_relational_cond(node.cond):
                points.append((line, DomainLevel.ZONE, "relational condition"))
            PointAnalysis._classify_node(node.then_body, points)
            if node.else_body:
                PointAnalysis._classify_node(node.else_body, points)

        if name == 'WhileStmt':
            if PointAnalysis._is_relational_cond(node.cond):
                points.append((line, DomainLevel.ZONE, "relational loop condition"))
            PointAnalysis._classify_node(node.body, points)

    @staticmethod
    def _count_vars(node):
        if node is None:
            return 0
        if isinstance(node, Var):
            return 1
        if isinstance(node, BinOp):
            return PointAnalysis._count_vars(node.left) + PointAnalysis._count_vars(node.right)
        return 0

    @staticmethod
    def _has_sum(node):
        """Check if expression has var + var pattern."""
        if isinstance(node, BinOp) and node.op == '+':
            if isinstance(node.left, Var) and isinstance(node.right, Var):
                return True
        if isinstance(node, BinOp):
            return PointAnalysis._has_sum(node.left) or PointAnalysis._has_sum(node.right)
        return False

    @staticmethod
    def _is_relational_cond(node):
        """Check if condition compares two variables."""
        if isinstance(node, BinOp) and node.op in ('<', '<=', '>', '>=', '==', '!='):
            left_is_var = isinstance(node.left, Var)
            right_is_var = isinstance(node.right, Var)
            return left_is_var and right_is_var
        return False


# ============================================================
# Public API
# ============================================================

def adaptive_analyze(source, max_level=DomainLevel.POLYHEDRA, max_iterations=50):
    """Analyze a program with adaptive domain selection."""
    interp = AdaptiveInterpreter(max_level=max_level, max_iterations=max_iterations)
    return interp.analyze(source)


def analyze_with_comparison(source, variables=None):
    """Analyze and compare fixed vs adaptive domain strategies."""
    return DomainComparison.compare_fixed_vs_adaptive(source, variables)


def precision_report(source, variables):
    """Report precision gains from adaptive analysis."""
    return DomainComparison.precision_gain(source, variables)


def classify_points(source):
    """Classify program points by domain requirements."""
    return PointAnalysis.classify_program(source)


def get_promotions(source, max_level=DomainLevel.POLYHEDRA):
    """Get list of all promotions that occurred during analysis."""
    result = adaptive_analyze(source, max_level)
    return result['promotions']


def get_relational_bounds(source, variable, max_level=DomainLevel.POLYHEDRA):
    """Get bounds for a variable using adaptive relational analysis."""
    result = adaptive_analyze(source, max_level)
    return result['env'].get_bounds(variable)


def get_relational_constraints(source, variable=None, max_level=DomainLevel.POLYHEDRA):
    """Get relational constraints discovered during analysis."""
    result = adaptive_analyze(source, max_level)
    return result['env'].get_relational_constraints(variable)


def compare_strategies(source, variables):
    """Compare interval-only vs adaptive strategies for given variables."""
    return DomainComparison.compare_fixed_vs_adaptive(source, variables)
