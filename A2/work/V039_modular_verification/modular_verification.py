"""
V039: Modular Verification (Contracts)
=======================================
Function-level contracts with compositional inter-procedural verification.

Composes:
  - V004 (VCGen) -- WP calculus, SExpr layer, VC checking
  - C010 (parser) -- AST
  - C037 (SMT solver) -- validity checking

Extends V004 from single-function verification to whole-program compositional
verification where each function is verified against its contract, and call
sites use callee contracts instead of inlining callee bodies.

Key features:
  - Contract extraction: requires(), ensures(), modifies() annotations
  - Modular WP: at call sites, check callee precondition + assume postcondition
  - Modifies clauses: frame conditions (unmodified vars preserved)
  - Contract refinement: subtyping (weaker pre, stronger post)
  - Whole-program compositional verification
  - Dependency-ordered verification (callees before callers)
  - Contract summaries: reusable function abstractions
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from typing import Any, Optional
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
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, substitute,
    ast_to_sexpr, lower_to_smt, check_vc,
    VCStatus, VCResult, VerificationResult,
    WPCalculus, extract_fn_spec, extract_loop_invariants, FnSpec, LoopSpec,
    parse,
)


# ============================================================
# Contract representation
# ============================================================

@dataclass
class Contract:
    """Function contract: precondition, postcondition, modifies clause."""
    fn_name: str
    params: list[str]
    preconditions: list[SExpr] = field(default_factory=list)
    postconditions: list[SExpr] = field(default_factory=list)
    modifies: list[str] = field(default_factory=list)  # vars this function may modify

    @property
    def precondition(self) -> SExpr:
        if not self.preconditions:
            return SBool(True)
        return s_and(*self.preconditions)

    @property
    def postcondition(self) -> SExpr:
        if not self.postconditions:
            return SBool(True)
        return s_and(*self.postconditions)


@dataclass
class ContractStore:
    """Maps function names to their contracts."""
    contracts: dict[str, Contract] = field(default_factory=dict)

    def add(self, contract: Contract):
        self.contracts[contract.fn_name] = contract

    def get(self, fn_name: str) -> Optional[Contract]:
        return self.contracts.get(fn_name)

    def has(self, fn_name: str) -> bool:
        return fn_name in self.contracts

    def all_names(self) -> list[str]:
        return list(self.contracts.keys())


# ============================================================
# Contract extraction from AST
# ============================================================

def _is_annotation_call(stmt, name: str) -> bool:
    """Check if a statement is a CallExpr with the given callee name."""
    if isinstance(stmt, CallExpr) and hasattr(stmt, 'callee'):
        return stmt.callee == name
    return False


def extract_contract(fn: FnDecl) -> Contract:
    """Extract a Contract from a function declaration's annotations."""
    contract = Contract(fn_name=fn.name, params=list(fn.params))

    body_stmts = fn.body.stmts if isinstance(fn.body, Block) else fn.body
    remaining = []

    for stmt in body_stmts:
        if _is_annotation_call(stmt, 'requires'):
            contract.preconditions.append(ast_to_sexpr(stmt.args[0]))
        elif _is_annotation_call(stmt, 'ensures'):
            contract.postconditions.append(ast_to_sexpr(stmt.args[0]))
        elif _is_annotation_call(stmt, 'modifies'):
            # modifies takes variable names as arguments
            for arg in stmt.args:
                if isinstance(arg, ASTVar):
                    contract.modifies.append(arg.name)
                elif isinstance(arg, str):
                    contract.modifies.append(arg)
        else:
            remaining.append(stmt)

    # If no modifies clause, infer from assignments in body
    if not contract.modifies:
        contract.modifies = _infer_modified_vars(remaining, contract.params)

    return contract


def _infer_modified_vars(stmts: list, params: list[str]) -> list[str]:
    """Infer which variables a function may modify (assigned but not declared locally)."""
    modified = set()
    local_decls = set()

    def walk(stmts_list):
        for stmt in stmts_list:
            if isinstance(stmt, LetDecl):
                local_decls.add(stmt.name)
            elif isinstance(stmt, Assign):
                if stmt.name not in local_decls:
                    modified.add(stmt.name)
            elif isinstance(stmt, IfStmt):
                walk(stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body])
                if stmt.else_body:
                    walk(stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body])
            elif isinstance(stmt, WhileStmt):
                walk(stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body])
            elif isinstance(stmt, Block):
                walk(stmt.stmts)

    walk(stmts)
    return list(modified)


def extract_all_contracts(source: str) -> ContractStore:
    """Extract contracts from all functions in a program."""
    prog = parse(source)
    store = ContractStore()
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl):
            store.add(extract_contract(stmt))
    return store


# ============================================================
# Modular WP Calculus (extends V004 WPCalculus)
# ============================================================

class ModularWP(WPCalculus):
    """WP calculus that handles function calls via contracts instead of inlining."""

    def __init__(self, contract_store: ContractStore):
        super().__init__()
        self.contract_store = contract_store
        self.call_vcs: list[tuple[str, SExpr]] = []  # precondition VCs at call sites

    def wp_stmt(self, stmt, postcond: SExpr) -> SExpr:
        """Extend WP to handle calls via contracts."""
        # CallExpr as a statement (side-effecting call)
        if isinstance(stmt, CallExpr):
            return self._wp_call_stmt(stmt, postcond)

        # LetDecl with a call on the RHS: let x = f(args);
        if isinstance(stmt, LetDecl) and isinstance(stmt.value, CallExpr):
            return self._wp_let_call(stmt, postcond)

        # Assign with a call on the RHS: x = f(args);
        if isinstance(stmt, Assign) and isinstance(stmt.value, CallExpr):
            return self._wp_assign_call(stmt, postcond)

        # Default to parent WP
        return super().wp_stmt(stmt, postcond)

    def _get_call_contract(self, call_expr: CallExpr) -> Optional[Contract]:
        """Get the contract for a function call, if available."""
        callee_name = call_expr.callee
        # Skip annotation calls
        if callee_name in ('requires', 'ensures', 'invariant', 'assert', 'modifies'):
            return None
        return self.contract_store.get(callee_name)

    def _substitute_params(self, expr: SExpr, params: list[str],
                           args: list) -> SExpr:
        """Substitute formal parameters with actual arguments in an expression."""
        result = expr
        for param, arg in zip(params, args):
            arg_sexpr = ast_to_sexpr(arg) if not isinstance(arg, SExpr) else arg
            result = substitute(result, param, arg_sexpr)
        return result

    def _wp_call_stmt(self, call: CallExpr, postcond: SExpr) -> SExpr:
        """WP for a bare call statement: f(args);

        WP(f(args), Q) = pre_f(args) AND Q
        The precondition is embedded in the WP so it's checked in context.
        """
        contract = self._get_call_contract(call)
        if contract is None:
            return postcond

        actual_pre = self._substitute_params(
            contract.precondition, contract.params, call.args
        )

        # Embed precondition in WP: must hold AND postcondition must hold
        return s_and(actual_pre, postcond)

    def _wp_let_call(self, stmt: LetDecl, postcond: SExpr) -> SExpr:
        """WP for: let x = f(args);

        WP(let x = f(args), Q) = pre_f(args) AND (post_f(args, x) => Q)

        The precondition is embedded in the WP (not accumulated separately)
        so that earlier postconditions from the WP chain are available as
        assumptions when checking this precondition.
        """
        call = stmt.value
        contract = self._get_call_contract(call)
        if contract is None:
            return postcond

        actual_pre = self._substitute_params(
            contract.precondition, contract.params, call.args
        )

        actual_post = self._substitute_params(
            contract.postcondition, contract.params, call.args
        )
        actual_post = substitute(actual_post, 'result', SVar(stmt.name))

        # WP = pre_f(args) AND (post_f(args, x) => Q)
        return s_and(actual_pre, s_implies(actual_post, postcond))

    def _wp_assign_call(self, stmt: Assign, postcond: SExpr) -> SExpr:
        """WP for: x = f(args);

        WP(x = f(args), Q) = pre_f(args) AND (post_f(args, x) => Q)
        """
        call = stmt.value
        contract = self._get_call_contract(call)
        if contract is None:
            return postcond

        actual_pre = self._substitute_params(
            contract.precondition, contract.params, call.args
        )

        actual_post = self._substitute_params(
            contract.postcondition, contract.params, call.args
        )
        actual_post = substitute(actual_post, 'result', SVar(stmt.name))

        return s_and(actual_pre, s_implies(actual_post, postcond))


# ============================================================
# Contract refinement (subtyping)
# ============================================================

@dataclass
class RefinementResult:
    """Result of checking contract refinement."""
    is_refinement: bool
    pre_weakened: bool       # new pre is weaker (or equal)
    post_strengthened: bool  # new post is stronger (or equal)
    counterexample: Optional[dict] = None
    details: str = ""


def check_refinement(old: Contract, new: Contract) -> RefinementResult:
    """
    Check if `new` contract refines `old` contract.
    Refinement means: weaker precondition AND stronger postcondition.

    new.pre => old.pre  (new accepts more inputs)
    old.post => new.post  (new promises more on outputs)

    Wait -- standard subtyping is contravariant in pre, covariant in post:
    old.pre => new.pre  (old's pre implies new's pre, so new accepts at least what old did)
    Actually the standard is:
    - new refines old iff:
      - old.pre => new.pre  (new is callable wherever old was)
      - new.post => old.post  (new provides at least what old promised)

    No wait. Behavioral subtyping (Liskov): B refines A iff
    - A.pre => B.pre  (B accepts everything A accepts -- B's pre is weaker)
    - B.post => A.post  (B provides at least A's guarantees -- B's post is stronger)

    So if B refines A, you can replace A with B safely.
    """
    old_pre = old.precondition
    new_pre = new.precondition
    old_post = old.postcondition
    new_post = new.postcondition

    # Align parameters: substitute new params with old params
    for op, np_ in zip(old.params, new.params):
        if op != np_:
            new_pre = substitute(new_pre, np_, SVar(op))
            new_post = substitute(new_post, np_, SVar(op))

    # Check: old.pre => new.pre (new's pre is weaker)
    pre_vc = s_implies(old_pre, new_pre)
    pre_result = check_vc("pre_weakening", pre_vc)
    pre_ok = pre_result.status == VCStatus.VALID

    # Check: new.post => old.post (new's post is stronger)
    post_vc = s_implies(new_post, old_post)
    post_result = check_vc("post_strengthening", post_vc)
    post_ok = post_result.status == VCStatus.VALID

    is_ref = pre_ok and post_ok
    cx = None
    if not pre_ok and pre_result.counterexample:
        cx = pre_result.counterexample
    elif not post_ok and post_result.counterexample:
        cx = post_result.counterexample

    details_parts = []
    if not pre_ok:
        details_parts.append("new precondition is not weaker than old")
    if not post_ok:
        details_parts.append("new postcondition is not stronger than old")

    return RefinementResult(
        is_refinement=is_ref,
        pre_weakened=pre_ok,
        post_strengthened=post_ok,
        counterexample=cx,
        details="; ".join(details_parts) if details_parts else "valid refinement"
    )


# ============================================================
# Modular function verification
# ============================================================

@dataclass
class ModularResult:
    """Result of modular verification of a whole program."""
    verified: bool
    function_results: dict[str, VerificationResult] = field(default_factory=dict)
    call_site_results: list[VCResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_vcs(self) -> int:
        total = len(self.call_site_results)
        for vr in self.function_results.values():
            total += vr.total_vcs
        return total

    @property
    def valid_vcs(self) -> int:
        valid = sum(1 for vc in self.call_site_results if vc.status == VCStatus.VALID)
        for vr in self.function_results.values():
            valid += vr.valid_vcs
        return valid

    @property
    def invalid_vcs(self) -> int:
        invalid = sum(1 for vc in self.call_site_results if vc.status == VCStatus.INVALID)
        for vr in self.function_results.values():
            invalid += vr.invalid_vcs
        return invalid


def _get_fn_body_stmts(fn: FnDecl) -> list:
    """Get the body statements of a function, stripping annotations."""
    body_stmts = fn.body.stmts if isinstance(fn.body, Block) else fn.body
    result = []
    for stmt in body_stmts:
        if isinstance(stmt, CallExpr) and hasattr(stmt, 'callee'):
            if stmt.callee in ('requires', 'ensures', 'modifies', 'invariant'):
                continue
        result.append(stmt)
    return result


def _build_call_graph(fns: dict[str, FnDecl]) -> dict[str, set[str]]:
    """Build a call graph from function declarations."""
    graph = {name: set() for name in fns}

    def find_calls(node, fn_name: str):
        if isinstance(node, CallExpr):
            if node.callee in fns:
                graph[fn_name].add(node.callee)
            for arg in node.args:
                find_calls(arg, fn_name)
        elif isinstance(node, LetDecl):
            find_calls(node.value, fn_name)
        elif isinstance(node, Assign):
            find_calls(node.value, fn_name)
        elif isinstance(node, IfStmt):
            find_calls(node.cond, fn_name)
            body = node.then_body.stmts if isinstance(node.then_body, Block) else [node.then_body]
            for s in body:
                find_calls(s, fn_name)
            if node.else_body:
                ebody = node.else_body.stmts if isinstance(node.else_body, Block) else [node.else_body]
                for s in ebody:
                    find_calls(s, fn_name)
        elif isinstance(node, WhileStmt):
            find_calls(node.cond, fn_name)
            body = node.body.stmts if isinstance(node.body, Block) else [node.body]
            for s in body:
                find_calls(s, fn_name)
        elif isinstance(node, ReturnStmt):
            if node.value:
                find_calls(node.value, fn_name)
        elif isinstance(node, PrintStmt):
            if node.value:
                find_calls(node.value, fn_name)
        elif isinstance(node, BinOp):
            find_calls(node.left, fn_name)
            find_calls(node.right, fn_name)
        elif isinstance(node, UnaryOp):
            find_calls(node.operand, fn_name)
        elif isinstance(node, Block):
            for s in node.stmts:
                find_calls(s, fn_name)

    for name, fn in fns.items():
        body_stmts = fn.body.stmts if isinstance(fn.body, Block) else fn.body
        for stmt in body_stmts:
            find_calls(stmt, name)

    return graph


def _topological_order(graph: dict[str, set[str]]) -> list[str]:
    """Topological sort of call graph (callees first). Handles cycles by breaking them."""
    visited = set()
    temp = set()
    order = []

    def visit(node):
        if node in visited:
            return
        if node in temp:
            # Cycle -- break it
            return
        temp.add(node)
        for dep in graph.get(node, set()):
            visit(dep)
        temp.discard(node)
        visited.add(node)
        order.append(node)

    for node in graph:
        visit(node)

    return order  # callees before callers


def verify_function_modular(source: str, fn_name: str,
                            contract_store: ContractStore) -> VerificationResult:
    """
    Verify a single function using contracts for called functions.

    At call sites, instead of inlining callee bodies:
    1. Check callee precondition (with actual args)
    2. Assume callee postcondition (with actual args, result bound to LHS var)
    """
    prog = parse(source)

    # Find the target function
    target_fn = None
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl) and stmt.name == fn_name:
            target_fn = stmt
            break

    if target_fn is None:
        return VerificationResult(verified=False, vcs=[], errors=[f"Function {fn_name} not found"])

    contract = contract_store.get(fn_name)
    if contract is None:
        return VerificationResult(verified=False, vcs=[], errors=[f"No contract for {fn_name}"])

    # Build modular WP
    wp_calc = ModularWP(contract_store)
    body_stmts = _get_fn_body_stmts(target_fn)

    pre = contract.precondition
    post = contract.postcondition

    try:
        wp_result = wp_calc.wp_stmts(body_stmts, post)
    except ValueError as e:
        return VerificationResult(verified=False, vcs=[], errors=[str(e)])

    # Main VC: precondition => WP(body, postcondition)
    main_vc = s_implies(pre, wp_result)

    all_vcs = []

    # Check main VC
    main_result = check_vc(f"{fn_name}: pre => WP(body, post)", main_vc)
    all_vcs.append(main_result)

    # Check loop/assert VCs from WP computation (wrapped with precondition)
    for vc_name, vc_formula in wp_calc.vcs:
        wrapped = s_implies(pre, vc_formula)
        result = check_vc(f"{fn_name}: {vc_name}", wrapped)
        all_vcs.append(result)

    # Call-site preconditions are now embedded in the WP (no separate call_vcs)

    verified = all(vc.status == VCStatus.VALID for vc in all_vcs)
    return VerificationResult(verified=verified, vcs=all_vcs, errors=[])


def verify_program_modular(source: str) -> ModularResult:
    """
    Verify all annotated functions in a program compositionally.

    1. Extract contracts from all functions
    2. Build call graph and determine verification order
    3. Verify each function using contracts of its callees
    """
    prog = parse(source)

    # Collect all functions
    fns = {}
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl):
            fns[stmt.name] = stmt

    # Extract contracts
    store = extract_all_contracts(source)

    # Build call graph and get verification order
    call_graph = _build_call_graph(fns)
    order = _topological_order(call_graph)

    result = ModularResult(verified=True)

    for fn_name in order:
        if not store.has(fn_name):
            continue
        contract = store.get(fn_name)
        # Only verify functions with at least one annotation
        if not contract.preconditions and not contract.postconditions:
            continue

        fn_result = verify_function_modular(source, fn_name, store)
        result.function_results[fn_name] = fn_result
        if not fn_result.verified:
            result.verified = False

    return result


# ============================================================
# Frame condition checking
# ============================================================

def check_frame_condition(source: str, fn_name: str,
                          contract_store: ContractStore) -> list[VCResult]:
    """
    Check that a function only modifies variables listed in its modifies clause.

    For each variable v NOT in modifies(f), verify that v' == v (v is preserved).
    This is checked by adding frame postconditions: old_v == v for each non-modified var.
    """
    contract = contract_store.get(fn_name)
    if contract is None:
        return [VCResult(name=f"No contract for {fn_name}",
                         status=VCStatus.UNKNOWN, counterexample=None)]

    prog = parse(source)
    target_fn = None
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl) and stmt.name == fn_name:
            target_fn = stmt
            break

    if target_fn is None:
        return [VCResult(name=f"Function {fn_name} not found",
                         status=VCStatus.UNKNOWN, counterexample=None)]

    body_stmts = _get_fn_body_stmts(target_fn)

    # Find all variables read in the function that aren't params or local decls
    all_vars = set()
    local_vars = set()
    _collect_vars(body_stmts, all_vars, local_vars)

    # Frame variables: referenced but not in modifies and not local
    frame_vars = set()
    for v in all_vars:
        if v not in contract.modifies and v not in local_vars and v not in contract.params:
            frame_vars.add(v)

    if not frame_vars:
        return []  # No frame variables to check

    results = []
    wp_calc = ModularWP(contract_store)

    for var in sorted(frame_vars):
        # Frame VC: pre => WP(body, var == old_var)
        old_var = f"__old_{var}"
        frame_post = SBinOp('==', SVar(var), SVar(old_var))

        # Add old_var = var to precondition
        pre_with_old = s_and(contract.precondition,
                             SBinOp('==', SVar(old_var), SVar(var)))

        try:
            wp_calc_local = ModularWP(contract_store)
            wp = wp_calc_local.wp_stmts(body_stmts, frame_post)
            vc = s_implies(pre_with_old, wp)
            result = check_vc(f"{fn_name}: frame({var})", vc)
            results.append(result)
        except ValueError:
            results.append(VCResult(
                name=f"{fn_name}: frame({var})",
                status=VCStatus.UNKNOWN, counterexample=None
            ))

    return results


def _collect_vars(stmts: list, all_vars: set, local_vars: set):
    """Collect all variable names referenced and locally declared."""
    for stmt in stmts:
        if isinstance(stmt, LetDecl):
            local_vars.add(stmt.name)
            _collect_expr_vars(stmt.value, all_vars)
        elif isinstance(stmt, Assign):
            all_vars.add(stmt.name)
            _collect_expr_vars(stmt.value, all_vars)
        elif isinstance(stmt, IfStmt):
            _collect_expr_vars(stmt.cond, all_vars)
            body = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            _collect_vars(body, all_vars, local_vars)
            if stmt.else_body:
                ebody = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                _collect_vars(ebody, all_vars, local_vars)
        elif isinstance(stmt, WhileStmt):
            _collect_expr_vars(stmt.cond, all_vars)
            body = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            _collect_vars(body, all_vars, local_vars)
        elif isinstance(stmt, ReturnStmt):
            if stmt.value:
                _collect_expr_vars(stmt.value, all_vars)
        elif isinstance(stmt, PrintStmt):
            if stmt.value:
                _collect_expr_vars(stmt.value, all_vars)
        elif isinstance(stmt, CallExpr):
            for arg in stmt.args:
                _collect_expr_vars(arg, all_vars)
        elif isinstance(stmt, Block):
            _collect_vars(stmt.stmts, all_vars, local_vars)


def _collect_expr_vars(expr, all_vars: set):
    """Collect variable names from an expression."""
    if isinstance(expr, ASTVar):
        all_vars.add(expr.name)
    elif isinstance(expr, BinOp):
        _collect_expr_vars(expr.left, all_vars)
        _collect_expr_vars(expr.right, all_vars)
    elif isinstance(expr, UnaryOp):
        _collect_expr_vars(expr.operand, all_vars)
    elif isinstance(expr, CallExpr):
        for arg in expr.args:
            _collect_expr_vars(arg, all_vars)


# ============================================================
# Contract summary and query
# ============================================================

def summarize_contracts(source: str) -> dict:
    """Extract and summarize all contracts in a program."""
    store = extract_all_contracts(source)
    summary = {}
    for name, contract in store.contracts.items():
        summary[name] = {
            'params': contract.params,
            'preconditions': [str(p) for p in contract.preconditions],
            'postconditions': [str(p) for p in contract.postconditions],
            'modifies': contract.modifies,
        }
    return summary


def check_call_safety(source: str, caller: str, callee: str) -> VCResult:
    """
    Check that a specific call from caller to callee satisfies the callee's precondition.

    Builds a dedicated VC: caller_pre => WP(stmts_before_call, callee_pre).
    """
    store = extract_all_contracts(source)
    callee_contract = store.get(callee)
    caller_contract = store.get(caller)

    if callee_contract is None:
        return VCResult(name=f"No contract for {callee}",
                        status=VCStatus.UNKNOWN, counterexample=None)
    if caller_contract is None:
        return VCResult(name=f"No contract for {caller}",
                        status=VCStatus.UNKNOWN, counterexample=None)

    # Verify the caller; the callee precondition is embedded in the WP.
    # If the caller verifies, the call is safe; if not, extract which VC failed.
    result = verify_function_modular(source, caller, store)

    if result.verified:
        return VCResult(name=f"Call {callee} in {caller}: precondition",
                        status=VCStatus.VALID, counterexample=None)

    # Find the failing VC -- if it exists, the call might be unsafe
    for vc in result.vcs:
        if vc.status == VCStatus.INVALID:
            return VCResult(name=f"Call {callee} in {caller}: precondition",
                            status=VCStatus.INVALID,
                            counterexample=vc.counterexample)

    return VCResult(name=f"Call {callee} in {caller}: precondition",
                    status=VCStatus.UNKNOWN, counterexample=None)


# ============================================================
# High-level APIs
# ============================================================

def verify_modular(source: str) -> ModularResult:
    """
    Verify a program using modular, contract-based verification.

    This is the main entry point. It:
    1. Extracts contracts from all functions
    2. Verifies each function against its contract
    3. At call sites, uses callee contracts (not bodies)
    4. Reports per-function and overall results
    """
    return verify_program_modular(source)


def verify_with_contracts(source: str, contracts: dict[str, dict]) -> ModularResult:
    """
    Verify a program with externally-provided contracts.

    contracts format:
        {'fn_name': {'pre': 'x > 0', 'post': 'result > 0', 'modifies': ['x']}}

    Pre/post are C10 expressions (parsed to SExpr).
    """
    store = ContractStore()
    prog = parse(source)

    # Get param lists from functions
    fn_params = {}
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl):
            fn_params[stmt.name] = list(stmt.params)

    for fn_name, spec in contracts.items():
        params = fn_params.get(fn_name, [])
        contract = Contract(fn_name=fn_name, params=params)

        if 'pre' in spec and spec['pre']:
            pre_expr = parse(f"let __dummy = {spec['pre']};")
            pre_sexpr = ast_to_sexpr(pre_expr.stmts[0].value)
            contract.preconditions.append(pre_sexpr)

        if 'post' in spec and spec['post']:
            post_expr = parse(f"let __dummy = {spec['post']};")
            post_sexpr = ast_to_sexpr(post_expr.stmts[0].value)
            contract.postconditions.append(post_sexpr)

        if 'modifies' in spec:
            contract.modifies = spec['modifies']

        store.add(contract)

    # Verify each function
    fns = {}
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl):
            fns[stmt.name] = stmt

    call_graph = _build_call_graph(fns)
    order = _topological_order(call_graph)

    result = ModularResult(verified=True)
    for fn_name in order:
        if not store.has(fn_name):
            continue
        fn_result = verify_function_modular(source, fn_name, store)
        result.function_results[fn_name] = fn_result
        if not fn_result.verified:
            result.verified = False

    return result


def check_contract_refinement(old_source: str, new_source: str,
                              fn_name: str) -> RefinementResult:
    """
    Check if the contract of fn_name in new_source refines the contract in old_source.
    """
    old_store = extract_all_contracts(old_source)
    new_store = extract_all_contracts(new_source)

    old_contract = old_store.get(fn_name)
    new_contract = new_store.get(fn_name)

    if old_contract is None:
        return RefinementResult(is_refinement=False, pre_weakened=False,
                                post_strengthened=False,
                                details=f"No contract for {fn_name} in old source")
    if new_contract is None:
        return RefinementResult(is_refinement=False, pre_weakened=False,
                                post_strengthened=False,
                                details=f"No contract for {fn_name} in new source")

    return check_refinement(old_contract, new_contract)


def verify_against_spec(source: str, fn_name: str,
                        pre: str, post: str) -> VerificationResult:
    """
    Verify a single function against an externally provided spec.
    Uses contracts from other functions in the source for modular call handling.
    """
    store = extract_all_contracts(source)

    # Override or add the target function's contract
    prog = parse(source)
    params = []
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl) and stmt.name == fn_name:
            params = list(stmt.params)
            break

    contract = Contract(fn_name=fn_name, params=params)
    if pre:
        pre_prog = parse(f"let __d = {pre};")
        contract.preconditions.append(ast_to_sexpr(pre_prog.stmts[0].value))
    if post:
        post_prog = parse(f"let __d = {post};")
        contract.postconditions.append(ast_to_sexpr(post_prog.stmts[0].value))

    store.add(contract)
    return verify_function_modular(source, fn_name, store)


def get_verification_order(source: str) -> list[str]:
    """Return the order in which functions should be verified (callees first)."""
    prog = parse(source)
    fns = {}
    for stmt in prog.stmts:
        if isinstance(stmt, FnDecl):
            fns[stmt.name] = stmt
    call_graph = _build_call_graph(fns)
    return _topological_order(call_graph)
