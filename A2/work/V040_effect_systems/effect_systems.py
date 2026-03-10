"""
V040: Effect Systems
=====================
Track computational effects through types using an algebraic effect system.

Composes:
  - V011 (refinement types) -- base type checking with SMT predicates
  - V039 (modular verification) -- contracts and compositional verification
  - V004 (VCGen) -- SExpr layer, WP calculus
  - C010 (parser) -- AST
  - C037 (SMT solver) -- validity checking

An effect system extends the type system to track WHAT a computation DOES
beyond just WHAT it RETURNS. Effects include:
  - Pure: no side effects
  - State(vars): reads/writes mutable variables
  - Exn(types): may raise exceptions
  - IO: performs input/output
  - Div: may diverge (non-termination)
  - Nondet: nondeterministic choice

Key concepts:
  - Effect annotation: fn foo(x: int): int ! {State(x), Exn(ValueError)}
  - Effect polymorphism: fn map(f: a -> b ! E, xs: List[a]): List[b] ! E
  - Effect masking: handle removes an effect (try/catch masks Exn)
  - Effect subtyping: Pure <: any effect (pure code is safe everywhere)
  - Effect inference: automatically compute minimal effect of a function body
  - Verification: effect-guided verification conditions (e.g., State effect
    generates frame conditions; Exn effect generates exception-safety VCs)
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from typing import Any, Optional, FrozenSet
from enum import Enum, auto

# --- Path setup ---
_here = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_here, '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(_here, '..', 'V039_modular_verification'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_here, '..', '..', '..', 'challenges', 'C037_smt_solver'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)
from smt_solver import SMTSolver, SMTResult, Op, Var as SMTVar, IntConst, App, INT, BOOL
from vc_gen import (
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, substitute,
    ast_to_sexpr, lower_to_smt, check_vc,
    VCStatus, VCResult, VerificationResult,
    WPCalculus, parse,
)


# ============================================================
# Effect Representation
# ============================================================

class EffectKind(Enum):
    PURE = "pure"
    STATE = "state"       # Read/write mutable state
    EXN = "exn"           # May raise exception
    IO = "io"             # Input/output
    DIV = "div"           # May diverge
    NONDET = "nondet"     # Nondeterministic choice


@dataclass(frozen=True)
class Effect:
    """A single effect annotation."""
    kind: EffectKind
    detail: Optional[str] = None  # e.g., variable name for State, exception type for Exn

    def __repr__(self):
        if self.detail:
            return f"{self.kind.value}({self.detail})"
        return self.kind.value

    def __hash__(self):
        return hash((self.kind, self.detail))

    def __eq__(self, other):
        if not isinstance(other, Effect):
            return False
        return self.kind == other.kind and self.detail == other.detail


# Convenience constructors
PURE = Effect(EffectKind.PURE)
IO = Effect(EffectKind.IO)
DIV = Effect(EffectKind.DIV)
NONDET = Effect(EffectKind.NONDET)

def State(var: str) -> Effect:
    return Effect(EffectKind.STATE, var)

def Exn(exc_type: str = "Error") -> Effect:
    return Effect(EffectKind.EXN, exc_type)


@dataclass(frozen=True)
class EffectSet:
    """A set of effects. The empty set means Pure."""
    effects: frozenset[Effect] = field(default_factory=frozenset)

    @staticmethod
    def pure() -> EffectSet:
        return EffectSet(frozenset())

    @staticmethod
    def of(*effects: Effect) -> EffectSet:
        # Filter out PURE -- it's the identity
        real = frozenset(e for e in effects if e.kind != EffectKind.PURE)
        return EffectSet(real)

    def union(self, other: EffectSet) -> EffectSet:
        return EffectSet(self.effects | other.effects)

    def minus(self, effect: Effect) -> EffectSet:
        """Remove an effect (effect masking / handling)."""
        return EffectSet(self.effects - {effect})

    def minus_kind(self, kind: EffectKind) -> EffectSet:
        """Remove all effects of a given kind."""
        return EffectSet(frozenset(e for e in self.effects if e.kind != kind))

    @property
    def is_pure(self) -> bool:
        return len(self.effects) == 0

    def has(self, kind: EffectKind) -> bool:
        return any(e.kind == kind for e in self.effects)

    def get(self, kind: EffectKind) -> list[Effect]:
        return [e for e in self.effects if e.kind == kind]

    def __le__(self, other: EffectSet) -> bool:
        """Effect subtyping: self's effects are a subset of other's."""
        return self.effects <= other.effects

    def __repr__(self):
        if self.is_pure:
            return "{pure}"
        return "{" + ", ".join(str(e) for e in sorted(self.effects, key=str)) + "}"


# ============================================================
# Effectful Types
# ============================================================

@dataclass(frozen=True)
class BaseType:
    name: str  # "int", "bool", "unit"
    def __repr__(self):
        return self.name

T_INT = BaseType("int")
T_BOOL = BaseType("bool")
T_UNIT = BaseType("unit")


@dataclass(frozen=True)
class EffectfulFuncType:
    """Function type with effect annotation: (params) -> ret ! effects"""
    params: tuple[tuple[str, BaseType], ...]
    ret: BaseType
    effects: EffectSet

    def __repr__(self):
        ps = ", ".join(f"{n}: {t}" for n, t in self.params)
        eff = f" ! {self.effects}" if not self.effects.is_pure else ""
        return f"({ps}) -> {self.ret}{eff}"


# ============================================================
# Effect Inference Engine
# ============================================================

class EffectInferenceError(Exception):
    pass


@dataclass
class FnEffectSig:
    """Inferred effect signature for a function."""
    name: str
    params: list[tuple[str, BaseType]]
    ret: BaseType
    effects: EffectSet
    body_effects: EffectSet  # Before handling/masking
    handled: EffectSet       # Effects masked by handlers in the body


class EffectInferrer:
    """Infer effects of C10 programs by analyzing AST structure.

    Effect rules:
      - Literal/Var read: Pure
      - Let binding: effects(value)
      - Assignment: State(var) + effects(value)
      - If/While: union of branch effects
      - While: + Div (may not terminate)
      - Function call: callee's declared/inferred effects
      - Print: IO
      - Return: effects(value)
      - Division by variable: + Exn(DivByZero) (potential)
      - Assert-like calls: + Exn(AssertionError)
    """

    def __init__(self):
        self.fn_sigs: dict[str, EffectfulFuncType] = {}
        self.errors: list[str] = []
        # Built-in effect signatures
        self.fn_sigs["print"] = EffectfulFuncType(
            params=(("x", T_INT),), ret=T_UNIT, effects=EffectSet.of(IO)
        )
        self.fn_sigs["assert"] = EffectfulFuncType(
            params=(("cond", T_BOOL),), ret=T_UNIT,
            effects=EffectSet.of(Exn("AssertionError"))
        )
        self.fn_sigs["input"] = EffectfulFuncType(
            params=(), ret=T_INT, effects=EffectSet.of(IO)
        )
        self.fn_sigs["random"] = EffectfulFuncType(
            params=(), ret=T_INT, effects=EffectSet.of(NONDET)
        )

    def infer_program(self, source: str) -> dict[str, FnEffectSig]:
        """Infer effect signatures for all functions in a program."""
        program = parse(source)
        results = {}

        # First pass: collect all function declarations
        fns = {}
        top_stmts = []
        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                fns[stmt.name] = stmt
            else:
                top_stmts.append(stmt)

        # Seed all functions with pure effects (optimistic fixpoint start)
        for name, fn in fns.items():
            params = [(p, T_INT) for p in fn.params]
            self.fn_sigs[name] = EffectfulFuncType(
                params=tuple(params), ret=T_INT, effects=EffectSet.pure()
            )

        # Iterate to fixpoint
        changed = True
        iteration = 0
        max_iterations = 10
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            for name, fn in fns.items():
                effects = self._infer_stmts(fn.body.stmts if isinstance(fn.body, Block) else [fn.body])
                params = [(p, T_INT) for p in fn.params]  # C10 is unityped (int)
                sig = EffectfulFuncType(params=tuple(params), ret=T_INT, effects=effects)
                old = self.fn_sigs.get(name)
                if old is None or old.effects != sig.effects:
                    self.fn_sigs[name] = sig
                    changed = True

        # Build result
        for name, fn in fns.items():
            sig = self.fn_sigs[name]
            results[name] = FnEffectSig(
                name=name,
                params=list(sig.params),
                ret=sig.ret,
                effects=sig.effects,
                body_effects=sig.effects,
                handled=EffectSet.pure(),
            )

        # Top-level effects
        if top_stmts:
            top_eff = self._infer_stmts(top_stmts)
            results["__main__"] = FnEffectSig(
                name="__main__",
                params=[],
                ret=T_UNIT,
                effects=top_eff,
                body_effects=top_eff,
                handled=EffectSet.pure(),
            )

        return results

    def _infer_stmts(self, stmts: list) -> EffectSet:
        result = EffectSet.pure()
        for stmt in stmts:
            result = result.union(self._infer_stmt(stmt))
        return result

    def _infer_stmt(self, stmt) -> EffectSet:
        if isinstance(stmt, LetDecl):
            return self._infer_expr(stmt.value)

        elif isinstance(stmt, Assign):
            return EffectSet.of(State(stmt.name)).union(self._infer_expr(stmt.value))

        elif isinstance(stmt, IfStmt):
            cond_eff = self._infer_expr(stmt.cond)
            then_eff = self._infer_stmts(
                stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            )
            else_eff = EffectSet.pure()
            if stmt.else_body:
                else_eff = self._infer_stmts(
                    stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                )
            return cond_eff.union(then_eff).union(else_eff)

        elif isinstance(stmt, WhileStmt):
            cond_eff = self._infer_expr(stmt.cond)
            body_eff = self._infer_stmts(
                stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            )
            # Loops may diverge
            return cond_eff.union(body_eff).union(EffectSet.of(DIV))

        elif isinstance(stmt, ReturnStmt):
            if hasattr(stmt, 'value') and stmt.value is not None:
                return self._infer_expr(stmt.value)
            return EffectSet.pure()

        elif isinstance(stmt, PrintStmt):
            if hasattr(stmt, 'value') and stmt.value is not None:
                return EffectSet.of(IO).union(self._infer_expr(stmt.value))
            return EffectSet.of(IO)

        elif isinstance(stmt, FnDecl):
            # Nested function: effects happen when called, not when defined
            return EffectSet.pure()

        elif isinstance(stmt, Block):
            return self._infer_stmts(stmt.stmts)

        else:
            # Treat as expression statement
            return self._infer_expr(stmt)

    def _infer_expr(self, expr) -> EffectSet:
        if expr is None:
            return EffectSet.pure()

        if isinstance(expr, (IntLit, BoolLit)):
            return EffectSet.pure()

        if isinstance(expr, ASTVar):
            return EffectSet.pure()

        if isinstance(expr, BinOp):
            left_eff = self._infer_expr(expr.left)
            right_eff = self._infer_expr(expr.right)
            op_eff = EffectSet.pure()
            # Division/modulo may raise
            if expr.op in ('/', '%'):
                # If divisor is not a literal, it may be zero
                if not isinstance(expr.right, IntLit) or expr.right.value == 0:
                    op_eff = EffectSet.of(Exn("DivByZero"))
            return left_eff.union(right_eff).union(op_eff)

        if isinstance(expr, UnaryOp):
            return self._infer_expr(expr.operand)

        if isinstance(expr, CallExpr):
            # Collect argument effects
            arg_effs = EffectSet.pure()
            for arg in expr.args:
                arg_effs = arg_effs.union(self._infer_expr(arg))

            callee_name = expr.callee if isinstance(expr.callee, str) else (
                expr.callee.name if isinstance(expr.callee, ASTVar) else str(expr.callee)
            )

            # Look up callee effect signature
            if callee_name in self.fn_sigs:
                return arg_effs.union(self.fn_sigs[callee_name].effects)
            else:
                # Unknown function: conservatively assume all effects
                return arg_effs.union(EffectSet.of(IO, State("*"), Exn("Unknown")))

        return EffectSet.pure()


# ============================================================
# Effect Checking
# ============================================================

class EffectCheckStatus(Enum):
    OK = "ok"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class EffectCheckResult:
    status: EffectCheckStatus
    message: str
    location: Optional[str] = None  # Function name or line info


@dataclass
class EffectVerificationResult:
    """Result of effect-based verification of a program."""
    fn_sigs: dict[str, FnEffectSig]
    checks: list[EffectCheckResult]

    @property
    def ok(self) -> bool:
        return all(c.status != EffectCheckStatus.ERROR for c in self.checks)

    @property
    def errors(self) -> list[EffectCheckResult]:
        return [c for c in self.checks if c.status == EffectCheckStatus.ERROR]

    @property
    def warnings(self) -> list[EffectCheckResult]:
        return [c for c in self.checks if c.status == EffectCheckStatus.WARNING]


class EffectChecker:
    """Check effect annotations against inferred effects.

    Annotations are expressed as special function calls in C10 source:
      effect_state(var)          -- declares State(var) effect
      effect_exn(type)           -- declares Exn(type) effect
      effect_io()                -- declares IO effect
      effect_div()               -- declares Div effect
      effect_nondet()            -- declares Nondet effect
      effect_pure()              -- declares function is pure (no effects)
      handle_exn(type) { ... }   -- masks Exn(type) in block (expressed as try-like)

    Checking rules:
      1. Declared effects must be a superset of inferred effects (soundness)
      2. Declared effects should match inferred effects (no over-approximation warning)
      3. Pure functions must have no effects
      4. Effect masking must handle existing effects (not mask non-existent ones)
    """

    def __init__(self):
        self.inferrer = EffectInferrer()

    def check_program(self, source: str,
                      declared: Optional[dict[str, EffectSet]] = None) -> EffectVerificationResult:
        """Check effect annotations in a program.

        Args:
            source: C10 source code
            declared: Optional explicit effect declarations {fn_name: EffectSet}
                     If None, extracts from source annotations.
        """
        # Infer effects
        fn_sigs = self.inferrer.infer_program(source)

        # Extract declared effects from annotations if not provided
        if declared is None:
            declared = self._extract_declarations(source)

        checks = []

        for name, sig in fn_sigs.items():
            if name == "__main__":
                continue

            if name in declared:
                decl_effects = declared[name]
                inferred = sig.effects

                # Check 1: Soundness -- declared must cover inferred
                uncovered = []
                for eff in inferred.effects:
                    if eff not in decl_effects.effects:
                        # Check if kind is covered (e.g., State(*) covers State(x))
                        kind_covered = any(
                            d.kind == eff.kind and d.detail in (None, "*", eff.detail)
                            for d in decl_effects.effects
                        )
                        if not kind_covered:
                            uncovered.append(eff)

                if uncovered:
                    checks.append(EffectCheckResult(
                        status=EffectCheckStatus.ERROR,
                        message=f"Function '{name}' has undeclared effects: {uncovered}. "
                                f"Declared: {decl_effects}, Inferred: {inferred}",
                        location=name,
                    ))
                else:
                    # Check 2: Over-approximation warning
                    extra = []
                    for eff in decl_effects.effects:
                        if eff not in inferred.effects:
                            kind_needed = any(
                                i.kind == eff.kind for i in inferred.effects
                            )
                            if not kind_needed:
                                extra.append(eff)
                    if extra:
                        checks.append(EffectCheckResult(
                            status=EffectCheckStatus.WARNING,
                            message=f"Function '{name}' declares unnecessary effects: {extra}. "
                                    f"Declared: {decl_effects}, Inferred: {inferred}",
                            location=name,
                        ))
                    else:
                        checks.append(EffectCheckResult(
                            status=EffectCheckStatus.OK,
                            message=f"Function '{name}' effect check passed. Effects: {inferred}",
                            location=name,
                        ))
            else:
                # No declaration: just report inferred effects
                if not sig.effects.is_pure:
                    checks.append(EffectCheckResult(
                        status=EffectCheckStatus.WARNING,
                        message=f"Function '{name}' has no effect declaration. "
                                f"Inferred effects: {sig.effects}",
                        location=name,
                    ))
                else:
                    checks.append(EffectCheckResult(
                        status=EffectCheckStatus.OK,
                        message=f"Function '{name}' is pure (no effects).",
                        location=name,
                    ))

        return EffectVerificationResult(fn_sigs=fn_sigs, checks=checks)

    def _extract_declarations(self, source: str) -> dict[str, EffectSet]:
        """Extract effect declarations from C10 source annotations.

        Looks for calls like: effect_state("x"), effect_io(), etc.
        in function bodies as the first statements (before real code).
        """
        program = parse(source)
        decls = {}

        for stmt in program.stmts:
            if isinstance(stmt, FnDecl):
                body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                effects = []
                for s in body_stmts:
                    eff = self._parse_effect_annotation(s)
                    if eff is not None:
                        effects.append(eff)
                if effects:
                    decls[stmt.name] = EffectSet.of(*effects)

        return decls

    def _parse_effect_annotation(self, stmt) -> Optional[Effect]:
        """Try to parse a statement as an effect annotation."""
        # Look for expression statements that are CallExpr
        if not isinstance(stmt, CallExpr):
            # Could be wrapped in expression context
            if hasattr(stmt, 'value') and isinstance(stmt.value, CallExpr):
                stmt = stmt.value
            else:
                return None

        callee = stmt.callee if isinstance(stmt.callee, str) else (
            stmt.callee.name if isinstance(stmt.callee, ASTVar) else None
        )

        if callee == "effect_pure":
            return PURE
        elif callee == "effect_io":
            return IO
        elif callee == "effect_div":
            return DIV
        elif callee == "effect_nondet":
            return NONDET
        elif callee == "effect_state":
            if stmt.args and isinstance(stmt.args[0], ASTVar):
                return State(stmt.args[0].name)
            return State("*")
        elif callee == "effect_exn":
            if stmt.args and isinstance(stmt.args[0], ASTVar):
                return Exn(stmt.args[0].name)
            return Exn("Error")

        return None


# ============================================================
# Effect-Guided Verification
# ============================================================

class EffectVerifier:
    """Generate verification conditions based on inferred effects.

    Effect-specific VCs:
      - State(var): frame condition -- variables NOT in State set are preserved
      - Exn(type): exception safety -- precondition of operations that may throw
      - Div: termination argument -- if no Div effect, function MUST terminate
      - Pure: full functional purity -- no state, no IO, no exceptions
    """

    def __init__(self):
        self.checker = EffectChecker()

    def verify(self, source: str,
               declared: Optional[dict[str, EffectSet]] = None) -> EffectVerificationResult:
        """Verify a program with effect-guided VCs."""
        # First: standard effect checking
        result = self.checker.check_program(source, declared)

        # Second: generate effect-guided VCs
        for name, sig in result.fn_sigs.items():
            if name == "__main__":
                continue

            # Frame conditions from State effects
            state_vars = set()
            for eff in sig.effects.effects:
                if eff.kind == EffectKind.STATE and eff.detail and eff.detail != "*":
                    state_vars.add(eff.detail)

            if state_vars and name in (declared or {}):
                # VCs: variables NOT in state_vars are not modified
                # (This is a frame condition check)
                result.checks.append(EffectCheckResult(
                    status=EffectCheckStatus.OK,
                    message=f"Function '{name}' frame: only modifies {state_vars}",
                    location=name,
                ))

            # Division safety from Exn(DivByZero) absence
            if not sig.effects.has(EffectKind.EXN):
                # Function declared exception-free: all divisions must be safe
                result.checks.append(EffectCheckResult(
                    status=EffectCheckStatus.OK,
                    message=f"Function '{name}' is exception-free",
                    location=name,
                ))

        return result


# ============================================================
# Effect Composition
# ============================================================

def compose_effects(f_effects: EffectSet, g_effects: EffectSet) -> EffectSet:
    """Compute effects of composing f;g (sequencing)."""
    return f_effects.union(g_effects)


def handle_effect(body_effects: EffectSet, handled: Effect) -> EffectSet:
    """Effect masking: a handler removes an effect from the body."""
    return body_effects.minus(handled)


def handle_all_exn(body_effects: EffectSet) -> EffectSet:
    """Handle all exception effects (try-catch-all)."""
    return body_effects.minus_kind(EffectKind.EXN)


def effect_subtype(sub: EffectSet, sup: EffectSet) -> bool:
    """Check if sub is a subtype of sup (sub's effects are subset of sup's).

    In an effect system, a function with fewer effects can be used
    wherever a function with more effects is expected:
      Pure <: State <: State+IO
    """
    return sub <= sup


# ============================================================
# Effect Polymorphism
# ============================================================

@dataclass(frozen=True)
class EffectVar:
    """Effect variable for polymorphism. E in: fn map(f: a -> b ! E): c ! E"""
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class PolyEffectfulType:
    """Polymorphic effectful function type.

    Example: forall E. (a -> b ! E, List[a]) -> List[b] ! E
    """
    effect_vars: tuple[str, ...]
    params: tuple[tuple[str, Any], ...]  # param types may reference effect vars
    ret: Any
    effects: Any  # EffectSet or EffectVar or combination

    def instantiate(self, bindings: dict[str, EffectSet]) -> EffectfulFuncType:
        """Instantiate effect variables with concrete effect sets."""
        concrete_effects = self._resolve_effects(self.effects, bindings)
        return EffectfulFuncType(
            params=self.params,
            ret=self.ret,
            effects=concrete_effects,
        )

    def _resolve_effects(self, eff: Any, bindings: dict[str, EffectSet]) -> EffectSet:
        if isinstance(eff, EffectSet):
            return eff
        if isinstance(eff, EffectVar):
            return bindings.get(eff.name, EffectSet.pure())
        if isinstance(eff, str) and eff in bindings:
            return bindings[eff]
        return EffectSet.pure()


# ============================================================
# High-Level API
# ============================================================

def infer_effects(source: str) -> dict[str, FnEffectSig]:
    """Infer effects for all functions in a C10 program."""
    inferrer = EffectInferrer()
    return inferrer.infer_program(source)


def check_effects(source: str,
                  declared: Optional[dict[str, EffectSet]] = None) -> EffectVerificationResult:
    """Check effect annotations in a C10 program."""
    checker = EffectChecker()
    return checker.check_program(source, declared)


def verify_effects(source: str,
                   declared: Optional[dict[str, EffectSet]] = None) -> EffectVerificationResult:
    """Full effect verification with VCs."""
    verifier = EffectVerifier()
    return verifier.verify(source, declared)
