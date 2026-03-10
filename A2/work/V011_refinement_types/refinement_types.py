"""V011: Refinement Type Checking (Liquid Types)

Composes C013 (type checker) + C037 (SMT solver) + V004 (SExpr/WP) + C010 (parser/AST).

Refinement types augment base types with logical predicates:
  {v: int | v >= 0}    -- non-negative integer
  {v: int | v > 0}     -- positive integer
  {v: bool | v == 1}   -- true boolean (encoded as int)

Subtype checking becomes SMT implication:
  {v|P} <: {v|Q}  iff  forall v. P(v) => Q(v)

Features:
- Refined base types with SMT-checked predicates
- Refined function signatures (parameter + return types)
- Path-sensitive refinement (branch conditions strengthen context)
- Automatic refinement inference for let-bindings
- Subtype checking via SMT implication
- Counterexample generation on type errors
- Source-level annotation API (requires/ensures style)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C013_type_checker'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr, Var, IntLit, FloatLit,
    StringLit, BoolLit, BinOp, UnaryOp
)
from type_checker import TypeChecker, TInt, TBool, TFloat, TString, TVoid, TFunc, INT, BOOL, FLOAT, STRING, VOID
from smt_solver import SMTSolver, SMTResult, Op, App, IntConst, BoolConst, Var as SMTVar
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, lower_to_smt, ast_to_sexpr
)


# ---------------------------------------------------------------------------
# Refinement Type Representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinedType:
    """A base type with a refinement predicate.

    {binder: base | predicate}

    Example: RefinedType(INT, 'v', SBinOp('>=', SVar('v'), SInt(0)))
             represents {v: int | v >= 0}
    """
    base: Any           # TInt, TBool, etc.
    binder: str         # Variable name in predicate
    predicate: SExpr    # Logical predicate over binder

    def __repr__(self):
        return f"{{{self.binder}: {self.base} | {self.predicate}}}"


@dataclass(frozen=True)
class RefinedFuncType:
    """A function type with refined parameters and return type.

    (x: {int | x > 0}, y: {int | y > 0}) -> {result: int | result > 0}
    """
    params: List[Tuple[str, RefinedType]]  # [(name, refined_type), ...]
    ret: RefinedType                        # Return refinement

    def __repr__(self):
        params_str = ', '.join(f"{n}: {t}" for n, t in self.params)
        return f"({params_str}) -> {self.ret}"


# Convenience constructors for common refinement types
def refined_int(pred_str_or_sexpr, binder='v'):
    """Create a refined int type. pred can be SExpr or will be parsed."""
    if isinstance(pred_str_or_sexpr, SExpr):
        return RefinedType(INT, binder, pred_str_or_sexpr)
    return RefinedType(INT, binder, pred_str_or_sexpr)

def unrefined(base):
    """Create an unrefined type (predicate is True)."""
    return RefinedType(base, 'v', SBool(True))

def nat_type(binder='v'):
    """Non-negative integer: {v: int | v >= 0}"""
    return RefinedType(INT, binder, SBinOp('>=', SVar(binder), SInt(0)))

def pos_type(binder='v'):
    """Positive integer: {v: int | v > 0}"""
    return RefinedType(INT, binder, SBinOp('>', SVar(binder), SInt(0)))

def range_type(lo, hi, binder='v'):
    """Range type: {v: int | lo <= v && v <= hi}"""
    return RefinedType(INT, binder, s_and(
        SBinOp('>=', SVar(binder), SInt(lo)),
        SBinOp('<=', SVar(binder), SInt(hi))
    ))

def eq_type(val, binder='v'):
    """Singleton type: {v: int | v == val}"""
    return RefinedType(INT, binder, SBinOp('==', SVar(binder), SInt(val)))


# ---------------------------------------------------------------------------
# Refinement Environment
# ---------------------------------------------------------------------------

class RefinementEnv:
    """Tracks refinement types for variables, with lexical scoping."""

    def __init__(self, parent=None):
        self.bindings: Dict[str, RefinedType] = {}
        self.assumptions: List[SExpr] = []  # Path conditions
        self.parent = parent

    def bind(self, name: str, rtype: RefinedType):
        self.bindings[name] = rtype

    def lookup(self, name: str) -> Optional[RefinedType]:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def assume(self, condition: SExpr):
        """Add a path condition (e.g., from if-branch)."""
        self.assumptions.append(condition)

    def all_assumptions(self) -> List[SExpr]:
        """Collect all assumptions from this env and parents."""
        result = list(self.assumptions)
        if self.parent:
            result = self.parent.all_assumptions() + result
        return result

    def child(self) -> 'RefinementEnv':
        return RefinementEnv(parent=self)

    def all_bindings(self) -> Dict[str, RefinedType]:
        """Collect all bindings (child overrides parent)."""
        result = {}
        if self.parent:
            result = self.parent.all_bindings()
        result.update(self.bindings)
        return result


# ---------------------------------------------------------------------------
# Substitution in SExpr
# ---------------------------------------------------------------------------

def subst_sexpr(expr: SExpr, var: str, replacement: SExpr) -> SExpr:
    """Substitute var with replacement in expr."""
    if isinstance(expr, SVar):
        return replacement if expr.name == var else expr
    elif isinstance(expr, (SInt, SBool)):
        return expr
    elif isinstance(expr, SBinOp):
        return SBinOp(expr.op, subst_sexpr(expr.left, var, replacement),
                       subst_sexpr(expr.right, var, replacement))
    elif isinstance(expr, SUnaryOp):
        return SUnaryOp(expr.op, subst_sexpr(expr.operand, var, replacement))
    elif isinstance(expr, SImplies):
        return SImplies(subst_sexpr(expr.antecedent, var, replacement),
                        subst_sexpr(expr.consequent, var, replacement))
    elif isinstance(expr, SAnd):
        return SAnd(tuple(subst_sexpr(c, var, replacement) for c in expr.conjuncts))
    elif isinstance(expr, SOr):
        return SOr(tuple(subst_sexpr(c, var, replacement) for c in expr.disjuncts))
    elif isinstance(expr, SNot):
        return SNot(subst_sexpr(expr.operand, var, replacement))
    elif isinstance(expr, SIte):
        return SIte(subst_sexpr(expr.cond, var, replacement),
                     subst_sexpr(expr.then_val, var, replacement),
                     subst_sexpr(expr.else_val, var, replacement))
    return expr


def selfify(rtype: RefinedType, name: str) -> SExpr:
    """Convert a refinement type's predicate to use the actual variable name.

    Given {v: int | v >= 0} and name='x', returns x >= 0.
    """
    return subst_sexpr(rtype.predicate, rtype.binder, SVar(name))


# ---------------------------------------------------------------------------
# Subtype Checking via SMT
# ---------------------------------------------------------------------------

@dataclass
class SubtypeResult:
    is_subtype: bool
    counterexample: Optional[Dict[str, int]] = None
    reason: str = ""


def check_subtype(
    sub: RefinedType,
    sup: RefinedType,
    assumptions: Optional[List[SExpr]] = None,
    extra_vars: Optional[Dict[str, RefinedType]] = None
) -> SubtypeResult:
    """Check if sub <: sup under assumptions.

    {v|P} <: {v|Q} under assumptions A iff:
      forall v. (A AND P(v)) => Q(v)

    Checked by: NOT((A AND P(v)) => Q(v)) is UNSAT.
    Equivalently: A AND P(v) AND NOT(Q(v)) is UNSAT.
    """
    solver = SMTSolver()

    # Use a common binder variable
    binder = '__subtype_v'
    sub_pred = subst_sexpr(sub.predicate, sub.binder, SVar(binder))
    sup_pred = subst_sexpr(sup.predicate, sup.binder, SVar(binder))

    # Collect all constraints: assumptions + sub predicate
    constraints = []
    if assumptions:
        constraints.extend(assumptions)
    constraints.append(sub_pred)

    # Add bindings for extra variables (context refinements)
    if extra_vars:
        for vname, vtype in extra_vars.items():
            if vname != binder:
                constraints.append(selfify(vtype, vname))

    # Check: constraints AND NOT(sup_pred) is UNSAT?
    neg_sup = negate_sexpr(sup_pred)

    all_constraints = s_and(*constraints, neg_sup) if constraints else neg_sup

    # Lower to SMT
    var_cache = {}
    smt_formula = lower_to_smt(solver, all_constraints, var_cache)

    # Ensure the binder variable exists
    if binder not in var_cache:
        var_cache[binder] = solver.Int(binder)

    # Ensure all referenced variables exist
    _ensure_vars(all_constraints, solver, var_cache)

    solver.add(smt_formula)
    result = solver.check()

    if result == SMTResult.UNSAT:
        return SubtypeResult(is_subtype=True)
    elif result == SMTResult.SAT:
        model = solver.model() or {}
        return SubtypeResult(
            is_subtype=False,
            counterexample=model,
            reason=f"Counterexample: {model}"
        )
    else:
        return SubtypeResult(
            is_subtype=False,
            reason="SMT solver returned UNKNOWN"
        )


def negate_sexpr(expr: SExpr) -> SExpr:
    """Negate an SExpr, using complement operators where possible."""
    if isinstance(expr, SBool):
        return SBool(not expr.value)
    if isinstance(expr, SNot):
        return expr.operand
    if isinstance(expr, SBinOp):
        complements = {
            '==': '!=', '!=': '==',
            '<': '>=', '>=': '<',
            '>': '<=', '<=': '>',
        }
        if expr.op in complements:
            return SBinOp(complements[expr.op], expr.left, expr.right)
        if expr.op == 'and':
            return s_or(negate_sexpr(expr.left), negate_sexpr(expr.right))
        if expr.op == 'or':
            return s_and(negate_sexpr(expr.left), negate_sexpr(expr.right))
    if isinstance(expr, SAnd):
        return SOr(tuple(negate_sexpr(c) for c in expr.conjuncts))
    if isinstance(expr, SOr):
        return SAnd(tuple(negate_sexpr(d) for d in expr.disjuncts))
    if isinstance(expr, SImplies):
        # NOT(A => B) = A AND NOT(B)
        return s_and(expr.antecedent, negate_sexpr(expr.consequent))
    return SNot(expr)


def _ensure_vars(expr: SExpr, solver: SMTSolver, var_cache: dict):
    """Ensure all variables in expr are registered with the solver."""
    if isinstance(expr, SVar):
        if expr.name not in var_cache:
            var_cache[expr.name] = solver.Int(expr.name)
    elif isinstance(expr, SBinOp):
        _ensure_vars(expr.left, solver, var_cache)
        _ensure_vars(expr.right, solver, var_cache)
    elif isinstance(expr, SUnaryOp):
        _ensure_vars(expr.operand, solver, var_cache)
    elif isinstance(expr, SAnd):
        for c in expr.conjuncts:
            _ensure_vars(c, solver, var_cache)
    elif isinstance(expr, SOr):
        for d in expr.disjuncts:
            _ensure_vars(d, solver, var_cache)
    elif isinstance(expr, SNot):
        _ensure_vars(expr.operand, solver, var_cache)
    elif isinstance(expr, SImplies):
        _ensure_vars(expr.antecedent, solver, var_cache)
        _ensure_vars(expr.consequent, solver, var_cache)
    elif isinstance(expr, SIte):
        _ensure_vars(expr.cond, solver, var_cache)
        _ensure_vars(expr.then_val, solver, var_cache)
        _ensure_vars(expr.else_val, solver, var_cache)


# ---------------------------------------------------------------------------
# Refinement Type Checker
# ---------------------------------------------------------------------------

@dataclass
class RefinementError:
    message: str
    location: str = ""
    counterexample: Optional[Dict[str, int]] = None

    def __repr__(self):
        loc = f" at {self.location}" if self.location else ""
        ce = f" (counterexample: {self.counterexample})" if self.counterexample else ""
        return f"RefinementError: {self.message}{loc}{ce}"


@dataclass
class CheckResult:
    """Result of refinement type checking."""
    errors: List[RefinementError]
    verified_obligations: int = 0
    total_obligations: int = 0
    function_types: Dict[str, RefinedFuncType] = field(default_factory=dict)

    @property
    def ok(self):
        return len(self.errors) == 0

    def __repr__(self):
        status = "OK" if self.ok else f"{len(self.errors)} errors"
        return f"CheckResult({status}, {self.verified_obligations}/{self.total_obligations} obligations verified)"


class RefinementChecker:
    """Type checks a program with refinement types.

    Walks AST, maintains a refinement environment, and generates
    subtype obligations checked via SMT.
    """

    def __init__(self):
        self.errors: List[RefinementError] = []
        self.verified = 0
        self.total = 0
        self.env = RefinementEnv()
        self.func_specs: Dict[str, RefinedFuncType] = {}
        self.return_type: Optional[RefinedType] = None

    def check_program(self, stmts, specs: Optional[Dict[str, RefinedFuncType]] = None):
        """Check a list of statements with optional function specs."""
        if specs:
            self.func_specs.update(specs)
        for stmt in stmts:
            self.check_stmt(stmt)

    def check_stmt(self, stmt):
        """Check a statement, updating the refinement environment."""
        if isinstance(stmt, LetDecl):
            self._check_let(stmt)
        elif isinstance(stmt, Assign):
            self._check_assign(stmt)
        elif isinstance(stmt, IfStmt):
            self._check_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self._check_while(stmt)
        elif isinstance(stmt, FnDecl):
            self._check_fn_decl(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._check_return(stmt)
        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                self.check_stmt(s)
        elif isinstance(stmt, PrintStmt):
            pass  # No refinement constraints
        elif isinstance(stmt, CallExpr):
            self._infer_expr(stmt)  # Check call side effects
        else:
            pass  # Unknown statement type

    def _check_let(self, stmt: LetDecl):
        """Check let binding: infer refinement from value."""
        rtype = self._infer_expr(stmt.value)
        # Selfify: the refinement predicate now refers to the variable name
        actual_pred = subst_sexpr(rtype.predicate, rtype.binder, SVar(stmt.name))
        bound_type = RefinedType(rtype.base, stmt.name, actual_pred)
        self.env.bind(stmt.name, bound_type)

    def _check_assign(self, stmt: Assign):
        """Check assignment: infer new refinement for variable."""
        rtype = self._infer_expr(stmt.value)
        actual_pred = subst_sexpr(rtype.predicate, rtype.binder, SVar(stmt.name))
        bound_type = RefinedType(rtype.base, stmt.name, actual_pred)
        self.env.bind(stmt.name, bound_type)

    def _check_if(self, stmt: IfStmt):
        """Check if statement with path-sensitive refinement."""
        cond_sexpr = ast_to_sexpr(stmt.cond)

        # Then branch: assume condition
        then_env = self.env.child()
        then_env.assume(cond_sexpr)
        saved_env = self.env
        self.env = then_env
        if isinstance(stmt.then_body, Block):
            for s in stmt.then_body.stmts:
                self.check_stmt(s)
        else:
            self.check_stmt(stmt.then_body)
        self.env = saved_env

        # Else branch: assume negation
        if stmt.else_body:
            else_env = self.env.child()
            else_env.assume(negate_sexpr(cond_sexpr))
            self.env = else_env
            if isinstance(stmt.else_body, Block):
                for s in stmt.else_body.stmts:
                    self.check_stmt(s)
            else:
                self.check_stmt(stmt.else_body)
            self.env = saved_env

    def _check_while(self, stmt: WhileStmt):
        """Check while loop. Weakens refinements for modified variables."""
        # Variables modified in the loop body lose their refinements
        modified = self._modified_vars(stmt.body)
        for var in modified:
            rt = self.env.lookup(var)
            if rt:
                self.env.bind(var, unrefined(rt.base))

        # Check body
        cond_sexpr = ast_to_sexpr(stmt.cond)
        body_env = self.env.child()
        body_env.assume(cond_sexpr)
        saved_env = self.env
        self.env = body_env
        if isinstance(stmt.body, Block):
            for s in stmt.body.stmts:
                self.check_stmt(s)
        else:
            self.check_stmt(stmt.body)
        self.env = saved_env

    def _check_fn_decl(self, stmt: FnDecl):
        """Check function declaration against its refinement spec."""
        spec = self.func_specs.get(stmt.name)
        if not spec:
            return  # No spec, nothing to check

        # Create function body environment with parameter refinements
        fn_env = self.env.child()
        for pname, ptype in spec.params:
            # Selfify: bind the parameter name in the environment
            actual_pred = subst_sexpr(ptype.predicate, ptype.binder, SVar(pname))
            fn_env.bind(pname, RefinedType(ptype.base, pname, actual_pred))

        saved_env = self.env
        saved_ret = self.return_type
        self.env = fn_env
        self.return_type = spec.ret

        # Check body
        if isinstance(stmt.body, Block):
            for s in stmt.body.stmts:
                self.check_stmt(s)
        else:
            self.check_stmt(stmt.body)

        self.env = saved_env
        self.return_type = saved_ret

        # Store spec for call-site checking
        self.func_specs[stmt.name] = spec

    def _check_return(self, stmt: ReturnStmt):
        """Check return value against function's return refinement type."""
        if not self.return_type:
            return
        val_type = self._infer_expr(stmt.value)
        self._check_subtype_obligation(
            val_type, self.return_type,
            f"return value",
            self.env
        )

    def _infer_expr(self, expr) -> RefinedType:
        """Infer the refinement type of an expression."""
        if isinstance(expr, IntLit):
            return RefinedType(INT, 'v', SBinOp('==', SVar('v'), SInt(expr.value)))

        elif isinstance(expr, BoolLit):
            val = 1 if expr.value else 0
            return RefinedType(BOOL, 'v', SBinOp('==', SVar('v'), SInt(val)))

        elif isinstance(expr, Var):
            rt = self.env.lookup(expr.name)
            if rt:
                # Return type that captures both: value IS this variable,
                # AND whatever we know about the variable from its binding.
                # This ensures the subtype checker can connect the value
                # to the variable's known refinements and path conditions.
                known_pred = selfify(rt, expr.name)
                value_eq = SBinOp('==', SVar('v'), SVar(expr.name))
                if isinstance(known_pred, SBool) and known_pred.value:
                    pred = value_eq
                else:
                    pred = s_and(value_eq, known_pred)
                return RefinedType(rt.base, 'v', pred)
            return unrefined(INT)

        elif isinstance(expr, UnaryOp):
            operand_type = self._infer_expr(expr.operand)
            if expr.op == '-':
                # {v | v == -(operand)}
                operand_expr = _type_to_expr(operand_type)
                return RefinedType(INT, 'v',
                    SBinOp('==', SVar('v'), SUnaryOp('-', operand_expr)))
            elif expr.op == 'not':
                return unrefined(BOOL)
            return unrefined(operand_type.base)

        elif isinstance(expr, BinOp):
            return self._infer_binop(expr)

        elif isinstance(expr, CallExpr):
            return self._infer_call(expr)

        elif isinstance(expr, StringLit):
            return unrefined(STRING)

        elif isinstance(expr, FloatLit):
            return unrefined(FLOAT)

        return unrefined(INT)

    def _infer_binop(self, expr: BinOp) -> RefinedType:
        """Infer refinement for binary operations."""
        left_type = self._infer_expr(expr.left)
        right_type = self._infer_expr(expr.right)

        left_expr = _type_to_expr(left_type)
        right_expr = _type_to_expr(right_type)

        # Arithmetic operations: result is exact
        if expr.op in ('+', '-', '*'):
            result_expr = SBinOp(expr.op, left_expr, right_expr)
            return RefinedType(INT, 'v', SBinOp('==', SVar('v'), result_expr))

        # Comparison operations: result is boolean
        if expr.op in ('<', '>', '<=', '>=', '==', '!='):
            return unrefined(BOOL)

        # Logical operations
        if expr.op in ('and', 'or'):
            return unrefined(BOOL)

        return unrefined(INT)

    def _infer_call(self, expr: CallExpr) -> RefinedType:
        """Infer refinement for function call, checking argument types."""
        spec = self.func_specs.get(expr.callee)
        if not spec:
            return unrefined(INT)

        # Check arguments against parameter refinement types
        for i, (pname, ptype) in enumerate(spec.params):
            if i < len(expr.args):
                arg_type = self._infer_expr(expr.args[i])
                self._check_subtype_obligation(
                    arg_type, ptype,
                    f"argument '{pname}' in call to {expr.callee}",
                    self.env
                )

        # Return type may depend on parameter names -- substitute actual args
        ret = spec.ret
        ret_pred = ret.predicate
        for i, (pname, _) in enumerate(spec.params):
            if i < len(expr.args):
                arg_sexpr = ast_to_sexpr(expr.args[i])
                ret_pred = subst_sexpr(ret_pred, pname, arg_sexpr)

        return RefinedType(ret.base, ret.binder, ret_pred)

    def _check_subtype_obligation(self, actual: RefinedType, expected: RefinedType,
                                    location: str, env: RefinementEnv):
        """Generate and check a subtype obligation."""
        self.total += 1

        # Collect assumptions from the environment
        assumptions = env.all_assumptions()

        # Add refinements of all bound variables as assumptions
        for vname, vtype in env.all_bindings().items():
            pred = selfify(vtype, vname)
            if not isinstance(pred, SBool) or pred.value != True:
                assumptions.append(pred)

        result = check_subtype(actual, expected, assumptions if assumptions else None)
        if result.is_subtype:
            self.verified += 1
        else:
            self.errors.append(RefinementError(
                message=f"Subtype check failed for {location}: "
                        f"{actual} is not a subtype of {expected}",
                location=location,
                counterexample=result.counterexample
            ))

    def _modified_vars(self, body) -> set:
        """Find all variables modified in a block."""
        modified = set()
        stmts = body.stmts if isinstance(body, Block) else [body]
        for stmt in stmts:
            if isinstance(stmt, Assign):
                modified.add(stmt.name)
            elif isinstance(stmt, LetDecl):
                modified.add(stmt.name)
            elif isinstance(stmt, IfStmt):
                modified |= self._modified_vars(stmt.then_body)
                if stmt.else_body:
                    modified |= self._modified_vars(stmt.else_body)
            elif isinstance(stmt, WhileStmt):
                modified |= self._modified_vars(stmt.body)
            elif isinstance(stmt, Block):
                for s in stmt.stmts:
                    if isinstance(s, (Assign, LetDecl)):
                        modified.add(s.name if isinstance(s, Assign) else s.name)
        return modified

    def get_result(self) -> CheckResult:
        return CheckResult(
            errors=self.errors,
            verified_obligations=self.verified,
            total_obligations=self.total,
            function_types=dict(self.func_specs)
        )


def _type_to_expr(rtype: RefinedType) -> SExpr:
    """Extract the 'value' expression from a refined type.

    If the predicate is v == <expr>, return <expr>.
    Otherwise return SVar(binder).
    """
    pred = rtype.predicate
    if isinstance(pred, SBinOp) and pred.op == '==':
        if isinstance(pred.left, SVar) and pred.left.name == rtype.binder:
            return pred.right
        if isinstance(pred.right, SVar) and pred.right.name == rtype.binder:
            return pred.left
    # For complex predicates, just use the binder variable
    return SVar(rtype.binder)


# ---------------------------------------------------------------------------
# Source-Level Annotation API
# ---------------------------------------------------------------------------

def parse_source(source: str) -> List:
    """Parse C10 source into statement list."""
    tokens = lex(source)
    program = Parser(tokens).parse()
    return program.stmts


def extract_refinement_specs(stmts) -> Dict[str, RefinedFuncType]:
    """Extract refinement specs from annotated function declarations.

    Annotation format (using requires/ensures):
      fn abs(x) {
          requires(x == x);  // trivial precondition (just declares x: int)
          ensures(result >= 0);
          ...
      }

    With refined parameters:
      fn safe_div(x, y) {
          requires(y != 0);
          ensures(result * y <= x);
          ...
      }
    """
    specs = {}
    for stmt in stmts:
        if isinstance(stmt, FnDecl):
            pre_preds = []
            post_preds = []
            param_preds = {}  # param_name -> list of predicates

            body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            for s in body_stmts:
                if isinstance(s, CallExpr):
                    if s.callee == 'requires' and s.args:
                        pre_sexpr = ast_to_sexpr(s.args[0])
                        pre_preds.append(pre_sexpr)
                        # Classify: which parameter does this constrain?
                        for pname in stmt.params:
                            if _mentions_var(pre_sexpr, pname):
                                if pname not in param_preds:
                                    param_preds[pname] = []
                                param_preds[pname].append(pre_sexpr)
                    elif s.callee == 'ensures' and s.args:
                        post_preds.append(ast_to_sexpr(s.args[0]))

            if not pre_preds and not post_preds:
                continue  # No annotations

            # Build parameter refined types
            params = []
            for pname in stmt.params:
                if pname in param_preds:
                    pred = s_and(*param_preds[pname]) if len(param_preds[pname]) > 1 else param_preds[pname][0]
                    # Rename parameter to binder 'v'
                    pred_v = subst_sexpr(pred, pname, SVar('v'))
                    params.append((pname, RefinedType(INT, 'v', pred_v)))
                else:
                    params.append((pname, unrefined(INT)))

            # Build return refined type
            if post_preds:
                ret_pred = s_and(*post_preds) if len(post_preds) > 1 else post_preds[0]
                # Rename 'result' to binder 'v' if present
                ret_pred_v = subst_sexpr(ret_pred, 'result', SVar('v'))
                ret_type = RefinedType(INT, 'v', ret_pred_v)
            else:
                ret_type = unrefined(INT)

            specs[stmt.name] = RefinedFuncType(params=params, ret=ret_type)

    return specs


def _mentions_var(expr: SExpr, name: str) -> bool:
    """Check if an SExpr mentions a variable."""
    if isinstance(expr, SVar):
        return expr.name == name
    elif isinstance(expr, SBinOp):
        return _mentions_var(expr.left, name) or _mentions_var(expr.right, name)
    elif isinstance(expr, SUnaryOp):
        return _mentions_var(expr.operand, name)
    elif isinstance(expr, SAnd):
        return any(_mentions_var(c, name) for c in expr.conjuncts)
    elif isinstance(expr, SOr):
        return any(_mentions_var(d, name) for d in expr.disjuncts)
    elif isinstance(expr, SNot):
        return _mentions_var(expr.operand, name)
    elif isinstance(expr, SImplies):
        return _mentions_var(expr.antecedent, name) or _mentions_var(expr.consequent, name)
    elif isinstance(expr, SIte):
        return _mentions_var(expr.cond, name) or _mentions_var(expr.then_val, name) or _mentions_var(expr.else_val, name)
    return False


def strip_annotations(stmts) -> List:
    """Remove requires/ensures/invariant annotations from statements."""
    result = []
    for stmt in stmts:
        if isinstance(stmt, CallExpr) and stmt.callee in ('requires', 'ensures', 'invariant'):
            continue
        if isinstance(stmt, FnDecl):
            body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            clean_body = [s for s in body_stmts
                          if not (isinstance(s, CallExpr) and s.callee in ('requires', 'ensures', 'invariant'))]
            clean_fn = FnDecl(stmt.name, stmt.params, Block(clean_body, getattr(stmt.body, 'line', 0)),
                              getattr(stmt, 'line', 0))
            result.append(clean_fn)
        else:
            result.append(stmt)
    return result


# ---------------------------------------------------------------------------
# High-Level APIs
# ---------------------------------------------------------------------------

def check_refinements(source: str, specs: Dict[str, RefinedFuncType]) -> CheckResult:
    """Check a program against provided refinement specs.

    Args:
        source: C10 source code
        specs: Dict mapping function names to RefinedFuncType specs

    Returns:
        CheckResult with errors, verification stats
    """
    stmts = parse_source(source)
    checker = RefinementChecker()
    checker.check_program(stmts, specs)
    return checker.get_result()


def check_program_refinements(source: str) -> CheckResult:
    """Check a program using inline annotations (requires/ensures).

    Example:
        fn abs(x) {
            requires(x == x);
            ensures(result >= 0);
            if (x >= 0) { return x; } else { return (0 - x); }
        }
    """
    stmts = parse_source(source)
    specs = extract_refinement_specs(stmts)
    clean_stmts = strip_annotations(stmts)

    checker = RefinementChecker()
    checker.check_program(clean_stmts, specs)
    return checker.get_result()


def check_function_refinements(source: str, fn_name: str,
                                params: List[Tuple[str, RefinedType]],
                                ret: RefinedType) -> CheckResult:
    """Check a single function against a refinement spec.

    Args:
        source: C10 source code containing the function
        fn_name: Name of the function to check
        params: List of (param_name, RefinedType) pairs
        ret: Return refinement type

    Returns:
        CheckResult
    """
    spec = RefinedFuncType(params=params, ret=ret)
    return check_refinements(source, {fn_name: spec})


def check_subtype_valid(sub: RefinedType, sup: RefinedType,
                         assumptions: Optional[List[SExpr]] = None) -> SubtypeResult:
    """Standalone subtype check (no program context needed)."""
    return check_subtype(sub, sup, assumptions)


def infer_refinement(source: str, var_name: str) -> Optional[RefinedType]:
    """Infer the refinement type of a variable after executing source.

    Returns the refinement type bound to var_name in the final environment.
    """
    stmts = parse_source(source)
    checker = RefinementChecker()
    checker.check_program(stmts)
    return checker.env.lookup(var_name)
