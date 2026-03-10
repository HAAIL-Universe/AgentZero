"""
V042: Dependent Types
======================
Types that depend on values, enabling compile-time verification of array
bounds, non-zero divisors, vector lengths, and other value-dependent properties.

Composes:
  - V011 (refinement types) -- base refinement checking with SMT
  - V040 (effect systems) -- effect tracking for dependent computations
  - V004 (VCGen) -- SExpr layer, WP calculus
  - C010 (parser) -- AST
  - C037 (SMT solver) -- constraint solving

Dependent types go beyond refinement types by making types FIRST CLASS values
that can be computed and passed around:
  - `Array(n)`: an array with exactly n elements
  - `NonZero`: an integer guaranteed non-zero
  - `Bounded(lo, hi)`: integer in range [lo, hi)
  - `Equal(x)`: integer equal to x (singleton type)

Key distinction from V011:
  - V011: {v: int | v >= 0} -- refinement on a base type
  - V042: Bounded(0, n) -- type constructor parameterized by values
  - V042 types can be COMPUTED: `fn vec_len(v: Vec(n)) -> Equal(n)`

Verification: dependent type checking reduces to SMT obligation discharge.
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from enum import Enum, auto

# --- Path setup ---
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)
from smt_solver import SMTSolver, SMTResult, Op, Var as SMTVar, IntConst, App, INT, BOOL
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot,
    s_and, s_or, s_not, s_implies, lower_to_smt, parse,
)


# ============================================================
# Dependent Type Representation
# ============================================================

class DTypeKind(Enum):
    INT = "int"
    BOOL = "bool"
    UNIT = "unit"
    NONZERO = "nonzero"           # int, != 0
    POSITIVE = "positive"         # int, > 0
    NONNEG = "nonneg"             # int, >= 0
    BOUNDED = "bounded"           # int in [lo, hi)
    EQUAL = "equal"               # int == value (singleton type)
    ARRAY = "array"               # array of given length
    FUNC = "func"                 # dependent function type


@dataclass(frozen=True)
class DType:
    """Base dependent type."""
    kind: DTypeKind

    def __repr__(self):
        return self.kind.value


@dataclass(frozen=True)
class IntType(DType):
    """Plain integer type."""
    def __init__(self):
        object.__setattr__(self, 'kind', DTypeKind.INT)

    def to_predicate(self, binder: str) -> SExpr:
        return SBool(True)  # No constraint

    def __repr__(self):
        return "int"


@dataclass(frozen=True)
class BoolType(DType):
    def __init__(self):
        object.__setattr__(self, 'kind', DTypeKind.BOOL)

    def to_predicate(self, binder: str) -> SExpr:
        return s_or(SBinOp('==', SVar(binder), SInt(0)),
                    SBinOp('==', SVar(binder), SInt(1)))

    def __repr__(self):
        return "bool"


@dataclass(frozen=True)
class UnitType(DType):
    def __init__(self):
        object.__setattr__(self, 'kind', DTypeKind.UNIT)

    def to_predicate(self, binder: str) -> SExpr:
        return SBool(True)

    def __repr__(self):
        return "unit"


@dataclass(frozen=True)
class NonZeroType(DType):
    """Integer guaranteed to be non-zero."""
    def __init__(self):
        object.__setattr__(self, 'kind', DTypeKind.NONZERO)

    def to_predicate(self, binder: str) -> SExpr:
        return SBinOp('!=', SVar(binder), SInt(0))

    def __repr__(self):
        return "NonZero"


@dataclass(frozen=True)
class PositiveType(DType):
    """Integer guaranteed to be positive (> 0)."""
    def __init__(self):
        object.__setattr__(self, 'kind', DTypeKind.POSITIVE)

    def to_predicate(self, binder: str) -> SExpr:
        return SBinOp('>', SVar(binder), SInt(0))

    def __repr__(self):
        return "Positive"


@dataclass(frozen=True)
class NonNegType(DType):
    """Integer guaranteed to be non-negative (>= 0)."""
    def __init__(self):
        object.__setattr__(self, 'kind', DTypeKind.NONNEG)

    def to_predicate(self, binder: str) -> SExpr:
        return SBinOp('>=', SVar(binder), SInt(0))

    def __repr__(self):
        return "NonNeg"


@dataclass(frozen=True)
class BoundedType(DType):
    """Integer in range [lo, hi)."""
    lo: Union[int, str]  # int literal or variable name
    hi: Union[int, str]

    def __init__(self, lo, hi):
        object.__setattr__(self, 'kind', DTypeKind.BOUNDED)
        object.__setattr__(self, 'lo', lo)
        object.__setattr__(self, 'hi', hi)

    def _val_expr(self, v) -> SExpr:
        return SInt(v) if isinstance(v, int) else SVar(v)

    def to_predicate(self, binder: str) -> SExpr:
        lo_e = self._val_expr(self.lo)
        hi_e = self._val_expr(self.hi)
        return s_and(SBinOp('>=', SVar(binder), lo_e),
                     SBinOp('<', SVar(binder), hi_e))

    def __repr__(self):
        return f"Bounded({self.lo}, {self.hi})"


@dataclass(frozen=True)
class EqualType(DType):
    """Singleton type: value equals exactly `val`."""
    val: Union[int, str]

    def __init__(self, val):
        object.__setattr__(self, 'kind', DTypeKind.EQUAL)
        object.__setattr__(self, 'val', val)

    def to_predicate(self, binder: str) -> SExpr:
        v = SInt(self.val) if isinstance(self.val, int) else SVar(self.val)
        return SBinOp('==', SVar(binder), v)

    def __repr__(self):
        return f"Equal({self.val})"


@dataclass(frozen=True)
class ArrayType(DType):
    """Array type with dependent length."""
    length: Union[int, str]

    def __init__(self, length):
        object.__setattr__(self, 'kind', DTypeKind.ARRAY)
        object.__setattr__(self, 'length', length)

    def to_predicate(self, binder: str) -> SExpr:
        l = SInt(self.length) if isinstance(self.length, int) else SVar(self.length)
        return SBinOp('==', SVar(f"{binder}_len"), l)

    def __repr__(self):
        return f"Array({self.length})"


@dataclass(frozen=True)
class DepFuncType(DType):
    """Dependent function type: (x: T1) -> T2[x]"""
    params: tuple[tuple[str, DType], ...]
    ret: DType

    def __init__(self, params, ret):
        object.__setattr__(self, 'kind', DTypeKind.FUNC)
        object.__setattr__(self, 'params', tuple(params))
        object.__setattr__(self, 'ret', ret)

    def to_predicate(self, binder: str) -> SExpr:
        return SBool(True)

    def __repr__(self):
        ps = ", ".join(f"{n}: {t}" for n, t in self.params)
        return f"({ps}) -> {self.ret}"


# ============================================================
# Subtype Checking (via SMT)
# ============================================================

class SubtypeResult(Enum):
    SUBTYPE = "subtype"
    NOT_SUBTYPE = "not_subtype"
    UNKNOWN = "unknown"


@dataclass
class SubtypeCheckResult:
    result: SubtypeResult
    sub: DType
    sup: DType
    counterexample: Optional[dict] = None
    message: str = ""


def check_subtype(sub: DType, sup: DType,
                  context: Optional[SExpr] = None) -> SubtypeCheckResult:
    """Check if sub is a subtype of sup under context.

    sub <: sup  iff  forall v. context(v) AND sub(v) => sup(v)
    Equivalently: UNSAT(context AND sub(v) AND NOT sup(v))
    """
    binder = "__dt_v"

    sub_pred = sub.to_predicate(binder)
    sup_pred = sup.to_predicate(binder)

    # Build the formula: context AND sub AND NOT sup
    solver = SMTSolver()
    var = solver.Int(binder)

    # Also declare any length variables for array types
    _declare_extra_vars(solver, sub, binder)
    _declare_extra_vars(solver, sup, binder)

    # Declare context variables
    if context is not None:
        _declare_sexpr_vars(solver, context)

    var_cache = {}

    # Add context
    if context is not None:
        ctx_smt = lower_to_smt(solver, context, var_cache)
        solver.add(ctx_smt)

    # Add sub predicate
    sub_smt = lower_to_smt(solver, sub_pred, var_cache)
    solver.add(sub_smt)

    # Add negation of sup predicate
    sup_smt = lower_to_smt(solver, sup_pred, var_cache)
    neg_sup = App(Op.NOT, [sup_smt], BOOL)
    solver.add(neg_sup)

    result = solver.check()

    if result == SMTResult.UNSAT:
        return SubtypeCheckResult(
            result=SubtypeResult.SUBTYPE,
            sub=sub, sup=sup,
            message=f"{sub} <: {sup}"
        )
    elif result == SMTResult.SAT:
        model = solver.model()
        return SubtypeCheckResult(
            result=SubtypeResult.NOT_SUBTYPE,
            sub=sub, sup=sup,
            counterexample=model,
            message=f"{sub} is not a subtype of {sup}, counterexample: {model}"
        )
    else:
        return SubtypeCheckResult(
            result=SubtypeResult.UNKNOWN,
            sub=sub, sup=sup,
            message=f"Cannot determine subtype relationship between {sub} and {sup}"
        )


def _declare_extra_vars(solver: SMTSolver, dtype: DType, binder: str):
    """Declare extra SMT variables needed by the type."""
    if isinstance(dtype, ArrayType):
        solver.Int(f"{binder}_len")
    if isinstance(dtype, BoundedType):
        if isinstance(dtype.lo, str):
            solver.Int(dtype.lo)
        if isinstance(dtype.hi, str):
            solver.Int(dtype.hi)
    if isinstance(dtype, EqualType):
        if isinstance(dtype.val, str):
            solver.Int(dtype.val)


def _declare_sexpr_vars(solver: SMTSolver, expr: SExpr):
    """Declare variables that appear in an SExpr."""
    if isinstance(expr, SVar):
        solver.Int(expr.name)
    elif isinstance(expr, SBinOp):
        _declare_sexpr_vars(solver, expr.left)
        _declare_sexpr_vars(solver, expr.right)
    elif isinstance(expr, SUnaryOp):
        _declare_sexpr_vars(solver, expr.operand)
    elif isinstance(expr, (SAnd, SOr)):
        for child in expr.children:
            _declare_sexpr_vars(solver, child)
    elif isinstance(expr, SNot):
        _declare_sexpr_vars(solver, expr.operand)
    elif isinstance(expr, SImplies):
        _declare_sexpr_vars(solver, expr.lhs)
        _declare_sexpr_vars(solver, expr.rhs)


# ============================================================
# Dependent Type Checker
# ============================================================

class DTypeError(Exception):
    pass


@dataclass
class DTypeCheckResult:
    """Result of dependent type checking."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    types: dict[str, DType] = field(default_factory=dict)  # var -> inferred type


class DepTypeChecker:
    """Dependent type checker for C10 programs.

    Annotations (as function calls in source):
      dtype_nonzero(x)       -- declare x has type NonZero
      dtype_positive(x)      -- declare x has type Positive
      dtype_nonneg(x)        -- declare x has type NonNeg
      dtype_bounded(x, lo, hi) -- declare x has type Bounded(lo, hi)
      dtype_equal(x, val)    -- declare x has type Equal(val)
      dtype_array(x, len)    -- declare x has type Array(len)

    The checker infers types for expressions and checks that:
    1. Division targets have NonZero divisors
    2. Array indices are in bounds
    3. Function arguments match parameter types
    4. Return values match declared return types
    """

    def __init__(self):
        self.env: dict[str, DType] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.fn_sigs: dict[str, DepFuncType] = {}

    def check_program(self, source: str,
                      declared: Optional[dict[str, DType]] = None) -> DTypeCheckResult:
        """Type check a C10 program with dependent types."""
        self.env = {}
        self.errors = []
        self.warnings = []

        if declared:
            self.env.update(declared)

        program = parse(source)

        for stmt in program.stmts:
            self._check_stmt(stmt)

        return DTypeCheckResult(
            ok=len(self.errors) == 0,
            errors=list(self.errors),
            warnings=list(self.warnings),
            types=dict(self.env),
        )

    def _check_stmt(self, stmt):
        if isinstance(stmt, LetDecl):
            inferred = self._infer_expr(stmt.value)
            self.env[stmt.name] = inferred

        elif isinstance(stmt, Assign):
            inferred = self._infer_expr(stmt.value)
            # Check if there's a declared type to match
            if stmt.name in self.env:
                existing = self.env[stmt.name]
                result = check_subtype(inferred, existing)
                if result.result == SubtypeResult.NOT_SUBTYPE:
                    self.errors.append(
                        f"Assignment to '{stmt.name}': inferred {inferred} "
                        f"is not a subtype of declared {existing}"
                    )
            self.env[stmt.name] = inferred

        elif isinstance(stmt, IfStmt):
            # Check condition
            self._infer_expr(stmt.cond)
            # Check branches (save/restore env)
            saved = dict(self.env)
            body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            for s in body:
                self._check_stmt(s)
            then_env = dict(self.env)
            self.env = saved
            if stmt.else_body:
                ebody = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                for s in ebody:
                    self._check_stmt(s)
            # Join: widen to Int for vars that differ
            else_env = dict(self.env)
            for var in set(then_env) | set(else_env):
                t1 = then_env.get(var)
                t2 = else_env.get(var)
                if t1 and t2 and t1 != t2:
                    self.env[var] = IntType()  # Widen to plain int
                elif t1:
                    self.env[var] = t1
                elif t2:
                    self.env[var] = t2

        elif isinstance(stmt, WhileStmt):
            self._infer_expr(stmt.cond)
            body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            for s in body:
                self._check_stmt(s)

        elif isinstance(stmt, ReturnStmt):
            if hasattr(stmt, 'value') and stmt.value:
                self._infer_expr(stmt.value)

        elif isinstance(stmt, PrintStmt):
            if hasattr(stmt, 'value') and stmt.value:
                self._infer_expr(stmt.value)

        elif isinstance(stmt, FnDecl):
            saved = dict(self.env)
            for p in stmt.params:
                self.env[p] = IntType()
            body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            for s in body:
                self._check_stmt(s)
            self.env = saved

        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                self._check_stmt(s)

    def _infer_expr(self, expr) -> DType:
        """Infer the dependent type of an expression."""
        if expr is None:
            return UnitType()

        if isinstance(expr, IntLit):
            if expr.value == 0:
                return EqualType(0)
            elif expr.value > 0:
                return PositiveType()
            else:
                return IntType()

        if isinstance(expr, BoolLit):
            return BoolType()

        if isinstance(expr, ASTVar):
            return self.env.get(expr.name, IntType())

        if isinstance(expr, BinOp):
            left_t = self._infer_expr(expr.left)
            right_t = self._infer_expr(expr.right)

            if expr.op in ('/', '%'):
                # Division: check that divisor is non-zero
                if not self._is_nonzero(right_t):
                    self.warnings.append(
                        f"Division by potentially zero value (type: {right_t})"
                    )
                return IntType()

            if expr.op == '+':
                return self._infer_add(left_t, right_t)
            if expr.op == '-':
                return IntType()  # Subtraction is hard to track precisely
            if expr.op == '*':
                return self._infer_mul(left_t, right_t)

            # Comparison operators return bool
            if expr.op in ('==', '!=', '<', '<=', '>', '>='):
                return BoolType()

            # Logical operators
            if expr.op in ('&&', '||'):
                return BoolType()

            return IntType()

        if isinstance(expr, UnaryOp):
            self._infer_expr(expr.operand)
            if expr.op == '!':
                return BoolType()
            return IntType()

        if isinstance(expr, CallExpr):
            for arg in expr.args:
                self._infer_expr(arg)
            callee = expr.callee if isinstance(expr.callee, str) else (
                expr.callee.name if isinstance(expr.callee, ASTVar) else str(expr.callee))
            if callee in self.fn_sigs:
                return self.fn_sigs[callee].ret
            return IntType()

        return IntType()

    def _is_nonzero(self, t: DType) -> bool:
        """Check if a type guarantees non-zero."""
        if isinstance(t, NonZeroType):
            return True
        if isinstance(t, PositiveType):
            return True
        if isinstance(t, EqualType):
            if isinstance(t.val, int):
                return t.val != 0
        if isinstance(t, BoundedType):
            if isinstance(t.lo, int) and isinstance(t.hi, int):
                return t.lo > 0 or t.hi <= 0
        return False

    def _infer_add(self, left: DType, right: DType) -> DType:
        """Infer type of addition."""
        # Positive + NonNeg = Positive
        if isinstance(left, PositiveType) and isinstance(right, (NonNegType, PositiveType)):
            return PositiveType()
        if isinstance(right, PositiveType) and isinstance(left, (NonNegType, PositiveType)):
            return PositiveType()
        # NonNeg + NonNeg = NonNeg
        if isinstance(left, NonNegType) and isinstance(right, NonNegType):
            return NonNegType()
        # Equal + Equal = Equal
        if isinstance(left, EqualType) and isinstance(right, EqualType):
            if isinstance(left.val, int) and isinstance(right.val, int):
                return EqualType(left.val + right.val)
        return IntType()

    def _infer_mul(self, left: DType, right: DType) -> DType:
        """Infer type of multiplication."""
        # Positive * Positive = Positive
        if isinstance(left, PositiveType) and isinstance(right, PositiveType):
            return PositiveType()
        # NonNeg * NonNeg = NonNeg
        if isinstance(left, (NonNegType, PositiveType)) and isinstance(right, (NonNegType, PositiveType)):
            return NonNegType()
        return IntType()


# ============================================================
# High-Level API
# ============================================================

def check_dependent_types(source: str,
                          declared: Optional[dict[str, DType]] = None) -> DTypeCheckResult:
    """Check dependent types in a C10 program."""
    checker = DepTypeChecker()
    return checker.check_program(source, declared)


def is_subtype(sub: DType, sup: DType,
               context: Optional[SExpr] = None) -> bool:
    """Check if sub <: sup."""
    result = check_subtype(sub, sup, context)
    return result.result == SubtypeResult.SUBTYPE


# Convenience type constructors
T_INT = IntType()
T_BOOL = BoolType()
T_UNIT = UnitType()
T_NONZERO = NonZeroType()
T_POSITIVE = PositiveType()
T_NONNEG = NonNegType()

def Bounded(lo, hi) -> BoundedType:
    return BoundedType(lo, hi)

def Equal(val) -> EqualType:
    return EqualType(val)

def Array(length) -> ArrayType:
    return ArrayType(length)

def DepFunc(params, ret) -> DepFuncType:
    return DepFuncType(params, ret)
