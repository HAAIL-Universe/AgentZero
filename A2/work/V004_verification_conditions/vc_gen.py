"""
V004: Verification Condition Generation
========================================
Hoare-logic verification via Weakest Precondition calculus.

Composes:
  - C010 parser (AST)
  - C037 SMT solver (validity checking)

Given a program annotated with preconditions, postconditions, loop invariants,
and assertions, generates verification conditions (VCs) and checks them via SMT.

Annotations are expressed as special function calls in the source:
  requires(expr)   -- function precondition
  ensures(expr)    -- function postcondition (use 'result' to refer to return value)
  invariant(expr)  -- loop invariant (must appear as first statement in while body)
  assert(expr)     -- inline assertion

A VC is valid iff its negation is UNSAT.
"""

from __future__ import annotations
import sys, os, copy
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum, auto

# --- Import C010 parser and C037 SMT solver ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit, FloatLit, StringLit,
)

def parse(source: str) -> Program:
    """Lex and parse a source string into an AST."""
    tokens = lex(source)
    return Parser(tokens).parse()
from smt_solver import (
    SMTSolver, SMTResult, Op, Var as SMTVar, IntConst, BoolConst, App, INT, BOOL,
)


# ============================================================
# Result types
# ============================================================

class VCStatus(Enum):
    VALID = auto()      # VC holds (negation is UNSAT)
    INVALID = auto()    # VC fails (negation is SAT, counterexample available)
    UNKNOWN = auto()    # Solver couldn't determine

@dataclass
class VCResult:
    """Result of checking a single verification condition."""
    name: str               # Human-readable description
    status: VCStatus
    counterexample: Optional[dict] = None  # Variable assignments if INVALID
    formula_str: Optional[str] = None      # String representation of VC formula

@dataclass
class VerificationResult:
    """Result of verifying a program or function."""
    verified: bool              # True iff ALL VCs are valid
    vcs: list[VCResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_vcs(self):
        return len(self.vcs)

    @property
    def valid_vcs(self):
        return sum(1 for vc in self.vcs if vc.status == VCStatus.VALID)

    @property
    def invalid_vcs(self):
        return sum(1 for vc in self.vcs if vc.status == VCStatus.INVALID)


# ============================================================
# Symbolic expression representation (SMT-free, for WP calc)
# ============================================================
# We build symbolic expressions first, then lower to SMT terms.

class SExpr:
    """Base class for symbolic expressions used in WP calculus."""
    pass

@dataclass(frozen=True)
class SVar(SExpr):
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class SInt(SExpr):
    value: int
    def __repr__(self): return str(self.value)

@dataclass(frozen=True)
class SBool(SExpr):
    value: bool
    def __repr__(self): return str(self.value)

@dataclass(frozen=True)
class SBinOp(SExpr):
    op: str
    left: SExpr
    right: SExpr
    def __repr__(self): return f'({self.left} {self.op} {self.right})'

@dataclass(frozen=True)
class SUnaryOp(SExpr):
    op: str
    operand: SExpr
    def __repr__(self): return f'({self.op} {self.operand})'

@dataclass(frozen=True)
class SImplies(SExpr):
    antecedent: SExpr
    consequent: SExpr
    def __repr__(self): return f'({self.antecedent} => {self.consequent})'

@dataclass(frozen=True)
class SAnd(SExpr):
    conjuncts: tuple  # tuple of SExpr
    def __repr__(self): return f'({" and ".join(str(c) for c in self.conjuncts)})'

@dataclass(frozen=True)
class SOr(SExpr):
    disjuncts: tuple  # tuple of SExpr
    def __repr__(self): return f'({" or ".join(str(d) for d in self.disjuncts)})'

@dataclass(frozen=True)
class SNot(SExpr):
    operand: SExpr
    def __repr__(self): return f'(not {self.operand})'

@dataclass(frozen=True)
class SIte(SExpr):
    """If-then-else expression."""
    cond: SExpr
    then_val: SExpr
    else_val: SExpr
    def __repr__(self): return f'(if {self.cond} then {self.then_val} else {self.else_val})'


# Helpers for building symbolic expressions
def s_and(*args: SExpr) -> SExpr:
    flat = []
    for a in args:
        if isinstance(a, SBool) and a.value:
            continue  # skip True
        if isinstance(a, SBool) and not a.value:
            return SBool(False)
        if isinstance(a, SAnd):
            flat.extend(a.conjuncts)
        else:
            flat.append(a)
    if not flat:
        return SBool(True)
    if len(flat) == 1:
        return flat[0]
    return SAnd(tuple(flat))

def s_or(*args: SExpr) -> SExpr:
    flat = []
    for a in args:
        if isinstance(a, SBool) and not a.value:
            continue
        if isinstance(a, SBool) and a.value:
            return SBool(True)
        if isinstance(a, SOr):
            flat.extend(a.disjuncts)
        else:
            flat.append(a)
    if not flat:
        return SBool(False)
    if len(flat) == 1:
        return flat[0]
    return SOr(tuple(flat))

def s_not(e: SExpr) -> SExpr:
    if isinstance(e, SBool):
        return SBool(not e.value)
    if isinstance(e, SNot):
        return e.operand
    return SNot(e)

def s_implies(a: SExpr, b: SExpr) -> SExpr:
    if isinstance(a, SBool) and a.value:
        return b
    if isinstance(a, SBool) and not a.value:
        return SBool(True)
    if isinstance(b, SBool) and b.value:
        return SBool(True)
    return SImplies(a, b)


# ============================================================
# Substitution
# ============================================================

def substitute(expr: SExpr, var: str, replacement: SExpr) -> SExpr:
    """Substitute all occurrences of variable 'var' with 'replacement' in expr."""
    if isinstance(expr, SVar):
        return replacement if expr.name == var else expr
    if isinstance(expr, (SInt, SBool)):
        return expr
    if isinstance(expr, SBinOp):
        return SBinOp(expr.op, substitute(expr.left, var, replacement),
                       substitute(expr.right, var, replacement))
    if isinstance(expr, SUnaryOp):
        return SUnaryOp(expr.op, substitute(expr.operand, var, replacement))
    if isinstance(expr, SImplies):
        return SImplies(substitute(expr.antecedent, var, replacement),
                        substitute(expr.consequent, var, replacement))
    if isinstance(expr, SAnd):
        return SAnd(tuple(substitute(c, var, replacement) for c in expr.conjuncts))
    if isinstance(expr, SOr):
        return SOr(tuple(substitute(d, var, replacement) for d in expr.disjuncts))
    if isinstance(expr, SNot):
        return SNot(substitute(expr.operand, var, replacement))
    if isinstance(expr, SIte):
        return SIte(substitute(expr.cond, var, replacement),
                    substitute(expr.then_val, var, replacement),
                    substitute(expr.else_val, var, replacement))
    raise ValueError(f"Unknown SExpr type: {type(expr)}")


# ============================================================
# AST -> Symbolic expression conversion
# ============================================================

def ast_to_sexpr(node) -> SExpr:
    """Convert a C010 AST expression to a symbolic expression."""
    if isinstance(node, IntLit):
        return SInt(node.value)
    if isinstance(node, BoolLit):
        return SBool(node.value)
    if isinstance(node, ASTVar):
        return SVar(node.name)
    if isinstance(node, BinOp):
        return SBinOp(node.op, ast_to_sexpr(node.left), ast_to_sexpr(node.right))
    if isinstance(node, UnaryOp):
        return SUnaryOp(node.op, ast_to_sexpr(node.operand))
    raise ValueError(f"Cannot convert AST node to SExpr: {type(node).__name__}")


# ============================================================
# Annotation extraction
# ============================================================

@dataclass
class FnSpec:
    """Specification for a function."""
    name: str
    params: list[str]
    preconditions: list[SExpr]    # requires(...)
    postconditions: list[SExpr]   # ensures(...) -- may reference 'result'
    body_stmts: list              # AST statements (without annotation stmts)

@dataclass
class LoopSpec:
    """Specification for a while loop."""
    invariants: list[SExpr]
    cond: Any  # AST node
    body_stmts: list  # AST statements (without invariant stmts)


def extract_fn_spec(fn: FnDecl) -> FnSpec:
    """Extract requires/ensures annotations from a function declaration."""
    preconditions = []
    postconditions = []
    body_stmts = []

    stmts = fn.body.stmts if isinstance(fn.body, Block) else [fn.body]

    for stmt in stmts:
        if isinstance(stmt, CallExpr) and stmt.callee == 'requires':
            preconditions.append(ast_to_sexpr(stmt.args[0]))
        elif isinstance(stmt, CallExpr) and stmt.callee == 'ensures':
            postconditions.append(ast_to_sexpr(stmt.args[0]))
        else:
            body_stmts.append(stmt)

    return FnSpec(
        name=fn.name,
        params=fn.params,
        preconditions=preconditions,
        postconditions=postconditions,
        body_stmts=body_stmts,
    )


def extract_loop_invariants(while_stmt: WhileStmt) -> LoopSpec:
    """Extract invariant annotations from a while loop body."""
    invariants = []
    body_stmts = []

    stmts = while_stmt.body.stmts if isinstance(while_stmt.body, Block) else [while_stmt.body]

    for stmt in stmts:
        if isinstance(stmt, CallExpr) and stmt.callee == 'invariant':
            invariants.append(ast_to_sexpr(stmt.args[0]))
        else:
            body_stmts.append(stmt)

    return LoopSpec(
        invariants=invariants,
        cond=while_stmt.cond,
        body_stmts=body_stmts,
    )


# ============================================================
# Weakest Precondition calculus
# ============================================================

class WPCalculus:
    """Weakest precondition transformer over C010 AST."""

    def __init__(self):
        self.vcs: list[tuple[str, SExpr]] = []  # (description, VC formula)

    def wp_stmts(self, stmts: list, postcond: SExpr) -> SExpr:
        """WP for a sequence of statements (processed right-to-left)."""
        result = postcond
        for stmt in reversed(stmts):
            result = self.wp_stmt(stmt, result)
        return result

    def wp_stmt(self, stmt, postcond: SExpr) -> SExpr:
        """Compute WP(stmt, postcond)."""

        # --- Assignment: WP(x = e, Q) = Q[x/e] ---
        if isinstance(stmt, (LetDecl, Assign)):
            rhs = ast_to_sexpr(stmt.value)
            return substitute(postcond, stmt.name, rhs)

        # --- Block ---
        if isinstance(stmt, Block):
            return self.wp_stmts(stmt.stmts, postcond)

        # --- If-then-else ---
        if isinstance(stmt, IfStmt):
            cond = ast_to_sexpr(stmt.cond)
            wp_then = self.wp_stmt(stmt.then_body, postcond)
            if stmt.else_body is not None:
                wp_else = self.wp_stmt(stmt.else_body, postcond)
            else:
                wp_else = postcond  # no else branch = skip
            return s_and(s_implies(cond, wp_then), s_implies(s_not(cond), wp_else))

        # --- While loop with invariant ---
        if isinstance(stmt, WhileStmt):
            loop = extract_loop_invariants(stmt)
            cond = ast_to_sexpr(stmt.cond)

            if not loop.invariants:
                raise ValueError(
                    "While loop requires at least one invariant annotation. "
                    "Add invariant(expr) as the first statement in the loop body."
                )

            inv = s_and(*loop.invariants) if len(loop.invariants) > 1 else loop.invariants[0]

            # VC 1: Invariant is preserved by loop body
            # I AND cond => WP(body, I)
            wp_body = self.wp_stmts(loop.body_stmts, inv)
            preservation = s_implies(s_and(inv, cond), wp_body)
            self.vcs.append(("Loop invariant preservation", preservation))

            # VC 2: Invariant + !cond => postcondition
            termination = s_implies(s_and(inv, s_not(cond)), postcond)
            self.vcs.append(("Loop postcondition establishment", termination))

            # WP of the while = the invariant (must hold on entry)
            return inv

        # --- Assert ---
        if isinstance(stmt, CallExpr) and stmt.callee == 'assert':
            assertion = ast_to_sexpr(stmt.args[0])
            # VC: the assertion must hold
            self.vcs.append(("Assertion", s_implies(postcond, assertion)))
            # Actually WP(assert(e), Q) = e AND Q
            return s_and(assertion, postcond)

        # --- Print (skip) ---
        if isinstance(stmt, PrintStmt):
            return postcond

        # --- Return ---
        if isinstance(stmt, ReturnStmt):
            if stmt.value is not None:
                ret_expr = ast_to_sexpr(stmt.value)
                return substitute(postcond, 'result', ret_expr)
            return postcond

        # --- Function declaration (skip in WP, handled separately) ---
        if isinstance(stmt, FnDecl):
            return postcond

        # --- Bare expression statement (skip) ---
        if isinstance(stmt, CallExpr):
            return postcond

        raise ValueError(f"WP not implemented for: {type(stmt).__name__}")


# ============================================================
# SExpr -> SMT term lowering
# ============================================================

def lower_to_smt(solver: SMTSolver, expr: SExpr, var_cache: dict[str, Any] = None) -> Any:
    """Convert symbolic expression to SMT term."""
    if var_cache is None:
        var_cache = {}

    def lower(e: SExpr):
        if isinstance(e, SVar):
            if e.name not in var_cache:
                var_cache[e.name] = solver.Int(e.name)
            return var_cache[e.name]
        if isinstance(e, SInt):
            return solver.IntVal(e.value)
        if isinstance(e, SBool):
            return solver.BoolVal(e.value)
        if isinstance(e, SBinOp):
            l, r = lower(e.left), lower(e.right)
            op_map = {
                '+': lambda: l + r,
                '-': lambda: l - r,
                '*': lambda: l * r,
                '/': lambda: App(Op.ITE, [App(Op.EQ, [r, IntConst(0)], BOOL),
                                          IntConst(0), App(Op.MUL, [l, App(Op.ITE, [App(Op.GE, [r, IntConst(0)], BOOL), IntConst(1), IntConst(-1)], INT)], INT)], INT),
                '%': lambda: l - App(Op.MUL, [App(Op.ITE, [App(Op.EQ, [r, IntConst(0)], BOOL), IntConst(0), r], INT), IntConst(0)], INT),
                '==': lambda: App(Op.EQ, [l, r], BOOL),
                '!=': lambda: App(Op.NEQ, [l, r], BOOL),
                '<': lambda: App(Op.LT, [l, r], BOOL),
                '>': lambda: App(Op.GT, [l, r], BOOL),
                '<=': lambda: App(Op.LE, [l, r], BOOL),
                '>=': lambda: App(Op.GE, [l, r], BOOL),
                'and': lambda: solver.And(l, r),
                'or': lambda: solver.Or(l, r),
            }
            if e.op in op_map:
                return op_map[e.op]()
            raise ValueError(f"Unknown binary op: {e.op}")
        if isinstance(e, SUnaryOp):
            operand = lower(e.operand)
            if e.op == '-':
                return IntConst(0) - operand
            if e.op == 'not':
                return solver.Not(operand)
            raise ValueError(f"Unknown unary op: {e.op}")
        if isinstance(e, SImplies):
            return solver.Implies(lower(e.antecedent), lower(e.consequent))
        if isinstance(e, SAnd):
            return solver.And(*(lower(c) for c in e.conjuncts))
        if isinstance(e, SOr):
            return solver.Or(*(lower(d) for d in e.disjuncts))
        if isinstance(e, SNot):
            return solver.Not(lower(e.operand))
        if isinstance(e, SIte):
            return solver.If(lower(e.cond), lower(e.then_val), lower(e.else_val))
        raise ValueError(f"Cannot lower to SMT: {type(e).__name__}")

    return lower(expr)


# ============================================================
# VC checking
# ============================================================

def check_vc(name: str, formula: SExpr) -> VCResult:
    """Check if a VC formula is valid (negation is UNSAT)."""
    solver = SMTSolver()
    var_cache = {}
    smt_formula = lower_to_smt(solver, formula, var_cache)

    # VC is valid iff NOT(formula) is UNSAT
    solver.add(solver.Not(smt_formula))
    result = solver.check()

    if result == SMTResult.UNSAT:
        return VCResult(name=name, status=VCStatus.VALID,
                       formula_str=str(formula))
    elif result == SMTResult.SAT:
        model = solver.model()
        return VCResult(name=name, status=VCStatus.INVALID,
                       counterexample=model, formula_str=str(formula))
    else:
        return VCResult(name=name, status=VCStatus.UNKNOWN,
                       formula_str=str(formula))


# ============================================================
# Main verification API
# ============================================================

def verify_function(source: str, fn_name: str = None) -> VerificationResult:
    """
    Verify a function's specification (requires/ensures) via WP calculus.

    If fn_name is None, verifies all annotated functions.
    Returns VerificationResult with all VCs and their status.
    """
    try:
        program = parse(source)
    except Exception as e:
        return VerificationResult(verified=False, errors=[f"Parse error: {e}"])

    functions = [s for s in program.stmts if isinstance(s, FnDecl)]

    if fn_name:
        functions = [f for f in functions if f.name == fn_name]
        if not functions:
            return VerificationResult(verified=False,
                                      errors=[f"Function '{fn_name}' not found"])

    results = VerificationResult(verified=True)

    for fn in functions:
        spec = extract_fn_spec(fn)

        if not spec.preconditions and not spec.postconditions:
            continue  # No spec to verify

        # Build postcondition
        postcond = s_and(*spec.postconditions) if spec.postconditions else SBool(True)

        # Compute WP
        wp_calc = WPCalculus()
        try:
            wp = wp_calc.wp_stmts(spec.body_stmts, postcond)
        except Exception as e:
            results.errors.append(f"WP computation error in {spec.name}: {e}")
            results.verified = False
            continue

        # Build precondition
        precond = s_and(*spec.preconditions) if spec.preconditions else SBool(True)

        # Main VC: precondition => WP(body, postcondition)
        main_vc = s_implies(precond, wp)
        vc_result = check_vc(f"{spec.name}: precondition => wp(body, postcondition)", main_vc)
        results.vcs.append(vc_result)
        if vc_result.status != VCStatus.VALID:
            results.verified = False

        # Additional VCs from loops/assertions
        for (desc, vc_formula) in wp_calc.vcs:
            # These VCs must hold under the precondition
            full_vc = s_implies(precond, vc_formula)
            vc_result = check_vc(f"{spec.name}: {desc}", full_vc)
            results.vcs.append(vc_result)
            if vc_result.status != VCStatus.VALID:
                results.verified = False

    return results


def verify_program(source: str) -> VerificationResult:
    """
    Verify all annotated constructs in a program:
    - Function specs (requires/ensures)
    - Inline assertions
    - Loop invariants

    For top-level code (not in functions), treats the entire program as a
    single block with postcondition True and checks assertions/invariants.
    """
    try:
        program = parse(source)
    except Exception as e:
        return VerificationResult(verified=False, errors=[f"Parse error: {e}"])

    results = VerificationResult(verified=True)

    # Verify functions
    functions = [s for s in program.stmts if isinstance(s, FnDecl)]
    for fn in functions:
        fn_result = verify_function(source, fn.name)
        results.vcs.extend(fn_result.vcs)
        results.errors.extend(fn_result.errors)
        if not fn_result.verified:
            results.verified = False

    # Verify top-level code (assertions, loop invariants)
    top_stmts = [s for s in program.stmts if not isinstance(s, FnDecl)]
    if top_stmts:
        wp_calc = WPCalculus()
        try:
            wp = wp_calc.wp_stmts(top_stmts, SBool(True))
        except Exception as e:
            results.errors.append(f"WP error in top-level code: {e}")
            results.verified = False
        else:
            # Check that WP is satisfiable (top-level has no precondition)
            if wp_calc.vcs:
                for (desc, vc_formula) in wp_calc.vcs:
                    vc_result = check_vc(f"top-level: {desc}", vc_formula)
                    results.vcs.append(vc_result)
                    if vc_result.status != VCStatus.VALID:
                        results.verified = False

    return results


def verify_hoare_triple(precond_src: str, program_src: str, postcond_src: str,
                        var_types: dict[str, str] = None) -> VerificationResult:
    """
    Verify a Hoare triple {P} S {Q} directly.

    precond_src:  expression string for precondition (e.g., "x > 0")
    program_src:  statement(s) as source code
    postcond_src: expression string for postcondition

    This wraps the statements in a function with requires/ensures and verifies.
    """
    # Detect variables used
    if var_types is None:
        var_types = {}

    # Build a wrapper function
    params = list(var_types.keys()) if var_types else []
    param_str = ', '.join(params)

    # Wrap in function with annotations
    fn_source = f"fn _hoare_({param_str}) {{\n"
    fn_source += f"  requires({precond_src});\n"
    fn_source += f"  ensures({postcond_src});\n"
    fn_source += f"  {program_src}\n"
    fn_source += "}\n"

    return verify_function(fn_source, '_hoare_')
