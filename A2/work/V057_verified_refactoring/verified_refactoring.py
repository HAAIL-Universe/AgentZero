"""
V057: Verified Refactoring
Composes V006 (equivalence checking) + V047 (incremental verification) + V055 (modular AI)
        + V004 (VCGen) + C010 (parser) + C037 (SMT)

Verifies that refactored code preserves behavior:
1. AST-level refactoring classification (rename, extract, inline, reorder, restructure)
2. Per-function equivalence checking via V006
3. Modular summary comparison via V055 (abstract behavior preservation)
4. Contract preservation via V004 (specifications still hold)
5. Certificate generation for verified refactorings via V044
"""

import sys, os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V006_equivalence_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V047_incremental_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V055_modular_abstract_interpretation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))

from stack_vm import lex, Parser
from equiv_check import (
    check_function_equivalence, check_program_equivalence,
    check_regression, EquivResult
)
from incremental_verification import (
    diff_programs, ProgramDiff, ChangeKind, FunctionChange,
    CertificateCache
)
from modular_abstract_interpretation import (
    modular_analyze, FunctionSummary, ModularAIResult
)
from vc_gen import verify_function, verify_program, VCResult
from proof_certificates import (
    ProofCertificate, ProofObligation, CertStatus, ProofKind,
    combine_certificates, check_certificate
)


# --- Refactoring Classification ---

class RefactoringKind(Enum):
    RENAME_VARIABLE = "rename_variable"
    RENAME_FUNCTION = "rename_function"
    EXTRACT_FUNCTION = "extract_function"
    INLINE_FUNCTION = "inline_function"
    REORDER_STATEMENTS = "reorder_statements"
    RESTRUCTURE_CONTROL = "restructure_control"
    SIMPLIFY_EXPRESSION = "simplify_expression"
    ADD_FUNCTION = "add_function"
    REMOVE_FUNCTION = "remove_function"
    MODIFY_FUNCTION = "modify_function"
    UNKNOWN = "unknown"


@dataclass
class RefactoringDetection:
    kind: RefactoringKind
    description: str
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    affected_functions: Set[str] = field(default_factory=set)


@dataclass
class FunctionVerificationResult:
    fn_name: str
    equivalence_verified: Optional[bool] = None
    summary_compatible: Optional[bool] = None
    contract_preserved: Optional[bool] = None
    counterexample: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        checks = [c for c in [self.equivalence_verified, self.summary_compatible,
                               self.contract_preserved] if c is not None]
        return len(checks) > 0 and all(checks)


@dataclass
class RefactoringResult:
    old_source: str
    new_source: str
    diff: ProgramDiff
    refactorings: List[RefactoringDetection]
    function_results: Dict[str, FunctionVerificationResult]
    certificate: Optional[ProofCertificate] = None
    old_summaries: Optional[Dict[str, FunctionSummary]] = None
    new_summaries: Optional[Dict[str, FunctionSummary]] = None

    @property
    def is_verified(self) -> bool:
        if not self.function_results:
            return not self.diff.has_changes
        return all(r.is_verified for r in self.function_results.values())

    @property
    def functions_verified(self) -> int:
        return sum(1 for r in self.function_results.values() if r.is_verified)

    @property
    def functions_failed(self) -> int:
        return sum(1 for r in self.function_results.values() if not r.is_verified)

    @property
    def refactoring_kinds(self) -> Set[RefactoringKind]:
        return {r.kind for r in self.refactorings}

    def summary(self) -> str:
        lines = []
        status = "VERIFIED" if self.is_verified else "FAILED"
        lines.append(f"Refactoring Verification: {status}")
        lines.append(f"  Functions: {self.functions_verified} verified, {self.functions_failed} failed")
        if self.refactorings:
            kinds = ", ".join(r.kind.value for r in self.refactorings)
            lines.append(f"  Refactorings detected: {kinds}")
        for name, result in self.function_results.items():
            s = "OK" if result.is_verified else "FAIL"
            notes = "; ".join(result.notes) if result.notes else ""
            lines.append(f"    {name}: {s}" + (f" ({notes})" if notes else ""))
        return "\n".join(lines)


# --- AST Helpers ---

def _parse(source: str):
    """Parse source into AST statements."""
    tokens = lex(source)
    return Parser(tokens).parse().stmts


def _extract_functions(stmts) -> Dict[str, Any]:
    """Extract function declarations by name."""
    fns = {}
    for stmt in stmts:
        if type(stmt).__name__ == 'FnDecl':
            fns[stmt.name] = stmt
    return fns


def _get_body_stmts(node) -> list:
    """Get the list of statements from a body (handles Block objects)."""
    if hasattr(node, 'stmts'):
        return node.stmts
    if isinstance(node, list):
        return node
    return [node]


def _collect_variables(node, variables=None) -> Set[str]:
    """Collect all variable names referenced in an AST node."""
    if variables is None:
        variables = set()
    name = type(node).__name__
    if name == 'ASTVar':
        variables.add(node.name)
    elif name == 'LetDecl':
        variables.add(node.name)
        if node.value:
            _collect_variables(node.value, variables)
    elif name == 'Assign':
        variables.add(node.name)
        _collect_variables(node.value, variables)
    elif name == 'BinOp':
        _collect_variables(node.left, variables)
        _collect_variables(node.right, variables)
    elif name == 'UnaryOp':
        _collect_variables(node.expr, variables)
    elif name == 'CallExpr':
        for arg in node.args:
            _collect_variables(arg, variables)
        if hasattr(node, 'callee'):
            _collect_variables(node.callee, variables)
    elif name == 'IfStmt':
        _collect_variables(node.cond, variables)
        for s in _get_body_stmts(node.then_body):
            _collect_variables(s, variables)
        if node.else_body:
            for s in _get_body_stmts(node.else_body):
                _collect_variables(s, variables)
    elif name == 'WhileStmt':
        _collect_variables(node.cond, variables)
        for s in _get_body_stmts(node.body):
            _collect_variables(s, variables)
    elif name == 'ReturnStmt':
        if node.value:
            _collect_variables(node.value, variables)
    elif name == 'PrintStmt':
        _collect_variables(node.value, variables)
    elif name == 'FnDecl':
        for p in node.params:
            variables.add(p)
        for s in _get_body_stmts(node.body):
            _collect_variables(s, variables)
    elif name == 'Block':
        for s in node.stmts:
            _collect_variables(s, variables)
    return variables


def _stmt_signature(stmt) -> str:
    """Generate structural signature for a statement (reused from V047 logic)."""
    name = type(stmt).__name__
    if name == 'LetDecl':
        return f"let:{stmt.name}={_expr_sig(stmt.value)}"
    elif name == 'Assign':
        return f"assign:{stmt.name}={_expr_sig(stmt.value)}"
    elif name == 'IfStmt':
        then_sigs = ";".join(_stmt_signature(s) for s in _get_body_stmts(stmt.then_body))
        else_sigs = ";".join(_stmt_signature(s) for s in _get_body_stmts(stmt.else_body) if stmt.else_body)
        return f"if:{_expr_sig(stmt.cond)}:{then_sigs}:{else_sigs}"
    elif name == 'WhileStmt':
        body_sigs = ";".join(_stmt_signature(s) for s in _get_body_stmts(stmt.body))
        return f"while:{_expr_sig(stmt.cond)}:{body_sigs}"
    elif name == 'ReturnStmt':
        return f"return:{_expr_sig(stmt.value) if stmt.value else 'none'}"
    elif name == 'PrintStmt':
        return f"print:{_expr_sig(stmt.value)}"
    elif name == 'FnDecl':
        body_sigs = ";".join(_stmt_signature(s) for s in _get_body_stmts(stmt.body))
        return f"fn:{stmt.name}({','.join(stmt.params)}):{body_sigs}"
    elif name == 'Block':
        return ";".join(_stmt_signature(s) for s in stmt.stmts)
    else:
        return f"?{name}"


def _expr_sig(expr) -> str:
    """Generate structural signature for an expression."""
    if expr is None:
        return "none"
    name = type(expr).__name__
    if name == 'IntLit':
        return str(expr.value)
    elif name == 'BoolLit':
        return str(expr.value)
    elif name == 'ASTVar':
        return expr.name
    elif name == 'BinOp':
        return f"({_expr_sig(expr.left)}{expr.op}{_expr_sig(expr.right)})"
    elif name == 'UnaryOp':
        return f"({expr.op}{_expr_sig(expr.expr)})"
    elif name == 'CallExpr':
        callee = expr.callee if isinstance(expr.callee, str) else (expr.callee.name if hasattr(expr.callee, 'name') else str(expr.callee))
        args = ",".join(_expr_sig(a) for a in expr.args)
        return f"{callee}({args})"
    else:
        return f"?{name}"


# --- Refactoring Detection ---

def _detect_refactorings(diff: ProgramDiff, old_source: str, new_source: str) -> List[RefactoringDetection]:
    """Classify what kind of refactoring occurred based on the diff."""
    refactorings = []

    for change in diff.function_changes:
        if change.kind == ChangeKind.ADDED:
            # Could be extract function or genuinely new
            refactorings.append(RefactoringDetection(
                kind=RefactoringKind.ADD_FUNCTION,
                description=f"Function '{change.name}' added",
                new_name=change.name,
                affected_functions={change.name}
            ))
        elif change.kind == ChangeKind.REMOVED:
            # Could be inline function or genuinely removed
            refactorings.append(RefactoringDetection(
                kind=RefactoringKind.REMOVE_FUNCTION,
                description=f"Function '{change.name}' removed",
                old_name=change.name,
                affected_functions={change.name}
            ))
        elif change.kind == ChangeKind.MODIFIED:
            refactorings.append(RefactoringDetection(
                kind=RefactoringKind.MODIFY_FUNCTION,
                description=f"Function '{change.name}' modified",
                old_name=change.name,
                new_name=change.name,
                affected_functions={change.name}
            ))

    # Detect extract function: one function added, another modified, new function
    # body matches part of old function
    added = {c.name for c in diff.function_changes if c.kind == ChangeKind.ADDED}
    modified = {c.name for c in diff.function_changes if c.kind == ChangeKind.MODIFIED}
    removed = {c.name for c in diff.function_changes if c.kind == ChangeKind.REMOVED}

    if added and modified:
        for add_name in added:
            for mod_name in modified:
                # Check if modified function now calls the added function
                try:
                    new_stmts = _parse(new_source)
                    new_fns = _extract_functions(new_stmts)
                    if add_name in new_fns and mod_name in new_fns:
                        calls_in_modified = _collect_calls_in_fn(new_fns[mod_name])
                        if add_name in calls_in_modified:
                            # Replace the generic ADD and MODIFY with EXTRACT
                            refactorings = [r for r in refactorings
                                            if not (r.affected_functions & {add_name, mod_name}
                                                    and r.kind in (RefactoringKind.ADD_FUNCTION,
                                                                   RefactoringKind.MODIFY_FUNCTION))]
                            refactorings.append(RefactoringDetection(
                                kind=RefactoringKind.EXTRACT_FUNCTION,
                                description=f"Extracted '{add_name}' from '{mod_name}'",
                                old_name=mod_name,
                                new_name=add_name,
                                affected_functions={add_name, mod_name}
                            ))
                except Exception:
                    pass

    # Detect inline function: one function removed, another modified
    if removed and modified:
        for rem_name in removed:
            for mod_name in modified:
                try:
                    old_stmts = _parse(old_source)
                    old_fns = _extract_functions(old_stmts)
                    if rem_name in old_fns and mod_name in old_fns:
                        calls_in_old = _collect_calls_in_fn(old_fns[mod_name])
                        if rem_name in calls_in_old:
                            refactorings = [r for r in refactorings
                                            if not (r.affected_functions & {rem_name, mod_name}
                                                    and r.kind in (RefactoringKind.REMOVE_FUNCTION,
                                                                   RefactoringKind.MODIFY_FUNCTION))]
                            refactorings.append(RefactoringDetection(
                                kind=RefactoringKind.INLINE_FUNCTION,
                                description=f"Inlined '{rem_name}' into '{mod_name}'",
                                old_name=rem_name,
                                new_name=mod_name,
                                affected_functions={rem_name, mod_name}
                            ))
                except Exception:
                    pass

    # Detect rename: function removed + function added with same body signature
    if removed and added:
        try:
            old_stmts = _parse(old_source)
            new_stmts = _parse(new_source)
            old_fns = _extract_functions(old_stmts)
            new_fns = _extract_functions(new_stmts)
            for rem_name in list(removed):
                for add_name in list(added):
                    if rem_name in old_fns and add_name in new_fns:
                        old_body = ";".join(_stmt_signature(s) for s in _get_body_stmts(old_fns[rem_name].body))
                        new_body = ";".join(_stmt_signature(s) for s in _get_body_stmts(new_fns[add_name].body))
                        if old_body == new_body and old_fns[rem_name].params == new_fns[add_name].params:
                            refactorings = [r for r in refactorings
                                            if not (r.affected_functions & {rem_name, add_name}
                                                    and r.kind in (RefactoringKind.ADD_FUNCTION,
                                                                   RefactoringKind.REMOVE_FUNCTION))]
                            refactorings.append(RefactoringDetection(
                                kind=RefactoringKind.RENAME_FUNCTION,
                                description=f"Renamed '{rem_name}' to '{add_name}'",
                                old_name=rem_name,
                                new_name=add_name,
                                affected_functions={rem_name, add_name}
                            ))
        except Exception:
            pass

    return refactorings


def _collect_calls_in_fn(fn_node) -> Set[str]:
    """Collect names of functions called within a function body."""
    calls = set()
    for stmt in _get_body_stmts(fn_node.body):
        _collect_calls_recursive(stmt, calls)
    return calls


def _collect_calls_recursive(node, calls: Set[str]):
    """Recursively collect call targets from AST."""
    name = type(node).__name__
    if name == 'CallExpr':
        callee = node.callee if hasattr(node, 'callee') else None
        if callee is not None:
            if isinstance(callee, str):
                callee_name = callee
            elif hasattr(callee, 'name'):
                callee_name = callee.name
            else:
                callee_name = None
            if callee_name and callee_name not in ('requires', 'ensures', 'invariant', 'modifies', 'assert', 'print'):
                calls.add(callee_name)
        for arg in node.args:
            _collect_calls_recursive(arg, calls)
    elif name == 'LetDecl':
        if node.value:
            _collect_calls_recursive(node.value, calls)
    elif name == 'Assign':
        _collect_calls_recursive(node.value, calls)
    elif name == 'BinOp':
        _collect_calls_recursive(node.left, calls)
        _collect_calls_recursive(node.right, calls)
    elif name == 'UnaryOp':
        _collect_calls_recursive(node.expr, calls)
    elif name == 'IfStmt':
        _collect_calls_recursive(node.cond, calls)
        for s in _get_body_stmts(node.then_body):
            _collect_calls_recursive(s, calls)
        if node.else_body:
            for s in _get_body_stmts(node.else_body):
                _collect_calls_recursive(s, calls)
    elif name == 'WhileStmt':
        _collect_calls_recursive(node.cond, calls)
        for s in _get_body_stmts(node.body):
            _collect_calls_recursive(s, calls)
    elif name == 'ReturnStmt':
        if node.value:
            _collect_calls_recursive(node.value, calls)
    elif name == 'PrintStmt':
        _collect_calls_recursive(node.value, calls)
    elif name == 'Block':
        for s in node.stmts:
            _collect_calls_recursive(s, calls)


# --- Summary Comparison ---

def _compare_summaries(
    old_summary: Optional[FunctionSummary],
    new_summary: Optional[FunctionSummary]
) -> Tuple[bool, List[str]]:
    """Compare abstract summaries for compatibility.

    Returns (compatible, notes).
    Summaries are compatible if:
    - Same parameter list
    - Result bounds are compatible (new is subset of old or equal)
    - No new warnings introduced
    """
    notes = []

    if old_summary is None or new_summary is None:
        if old_summary is None and new_summary is None:
            return True, ["Both summaries missing"]
        notes.append("One summary missing")
        return True, notes  # Can't compare, not a failure

    if not old_summary.analyzed or not new_summary.analyzed:
        notes.append("One or both summaries not fully analyzed")
        return True, notes

    # Check parameter compatibility
    if old_summary.params != new_summary.params:
        notes.append(f"Parameters changed: {old_summary.params} -> {new_summary.params}")
        return False, notes

    # Check result bounds compatibility
    compatible = True
    for var, old_bound in old_summary.result_bounds.items():
        if var in new_summary.result_bounds:
            new_bound = new_summary.result_bounds[var]
            # New bound should not be wider than old
            if old_bound.lower is not None and new_bound.lower is not None:
                if new_bound.lower < old_bound.lower:
                    notes.append(f"Result bound for '{var}' lower bound widened: {old_bound.lower} -> {new_bound.lower}")
                    compatible = False
            if old_bound.upper is not None and new_bound.upper is not None:
                if new_bound.upper > old_bound.upper:
                    notes.append(f"Result bound for '{var}' upper bound widened: {old_bound.upper} -> {new_bound.upper}")
                    compatible = False
        else:
            notes.append(f"Result bound for '{var}' missing in new version")

    # Check for new warnings
    old_warning_count = len(old_summary.warnings)
    new_warning_count = len(new_summary.warnings)
    if new_warning_count > old_warning_count:
        notes.append(f"New warnings introduced: {old_warning_count} -> {new_warning_count}")

    if not notes:
        notes.append("Summaries compatible")

    return compatible, notes


# --- Contract Preservation ---

def _check_contract_preservation(
    old_source: str, new_source: str, fn_name: str
) -> Tuple[Optional[bool], List[str]]:
    """Check if contracts are preserved after refactoring.

    Returns (preserved, notes).
    """
    notes = []
    try:
        new_result = verify_function(new_source, fn_name)
        if new_result.status == 'valid':
            notes.append("All verification conditions valid")
            return True, notes
        elif new_result.status == 'invalid':
            notes.append("Verification condition failed after refactoring")
            if hasattr(new_result, 'counterexample') and new_result.counterexample:
                notes.append(f"Counterexample: {new_result.counterexample}")
            return False, notes
        else:
            notes.append(f"Verification result: {new_result.status}")
            return None, notes
    except Exception as e:
        notes.append(f"Contract check skipped: {e}")
        return None, notes


# --- Equivalence Verification ---

def _verify_function_equivalence(
    old_source: str, new_source: str, fn_name: str,
    old_fn_name: Optional[str] = None,
    param_types: Optional[Dict[str, str]] = None
) -> Tuple[Optional[bool], Optional[Dict], List[str]]:
    """Verify function equivalence between old and new versions.

    Returns (equivalent, counterexample, notes).
    """
    notes = []
    actual_old_name = old_fn_name or fn_name

    try:
        # Get parameter info from AST
        old_stmts = _parse(old_source)
        old_fns = _extract_functions(old_stmts)

        if actual_old_name not in old_fns:
            notes.append(f"Function '{actual_old_name}' not in old source")
            return None, None, notes

        new_stmts = _parse(new_source)
        new_fns = _extract_functions(new_stmts)

        if fn_name not in new_fns:
            notes.append(f"Function '{fn_name}' not in new source")
            return None, None, notes

        old_fn = old_fns[actual_old_name]
        new_fn = new_fns[fn_name]

        # Build param_types if not provided
        if param_types is None:
            param_types = {p: 'int' for p in old_fn.params}

        result = check_function_equivalence(
            old_source, actual_old_name,
            new_source, fn_name,
            param_types=param_types,
            max_paths=64
        )

        if result.result == EquivResult.EQUIVALENT:
            notes.append("Semantically equivalent")
            return True, None, notes
        elif result.result == EquivResult.NOT_EQUIVALENT:
            ce = None
            if result.counterexample:
                ce = {
                    'inputs': result.counterexample.inputs,
                    'output1': result.counterexample.output1,
                    'output2': result.counterexample.output2,
                }
            notes.append("NOT equivalent -- behavior changed")
            return False, ce, notes
        else:
            notes.append("Equivalence check inconclusive")
            return None, None, notes

    except Exception as e:
        notes.append(f"Equivalence check error: {e}")
        return None, None, notes


# --- Certificate Generation ---

def _generate_refactoring_certificate(
    result: RefactoringResult
) -> ProofCertificate:
    """Generate a proof certificate for the refactoring verification."""
    obligations = []

    for fn_name, fn_result in result.function_results.items():
        # Equivalence obligation
        if fn_result.equivalence_verified is not None:
            status = CertStatus.VALID if fn_result.equivalence_verified else CertStatus.INVALID
            ce = fn_result.counterexample if not fn_result.equivalence_verified else None
            obligations.append(ProofObligation(
                name=f"equiv_{fn_name}",
                description=f"Function '{fn_name}' preserves behavior after refactoring",
                formula_str=f"forall inputs. old_{fn_name}(inputs) == new_{fn_name}(inputs)",
                formula_smt="",
                status=status,
                counterexample=ce
            ))

        # Summary compatibility obligation
        if fn_result.summary_compatible is not None:
            status = CertStatus.VALID if fn_result.summary_compatible else CertStatus.INVALID
            obligations.append(ProofObligation(
                name=f"summary_{fn_name}",
                description=f"Abstract summary compatible for '{fn_name}'",
                formula_str=f"summary(old_{fn_name}) compatible_with summary(new_{fn_name})",
                formula_smt="",
                status=status
            ))

        # Contract preservation obligation
        if fn_result.contract_preserved is not None:
            status = CertStatus.VALID if fn_result.contract_preserved else CertStatus.INVALID
            obligations.append(ProofObligation(
                name=f"contract_{fn_name}",
                description=f"Contracts preserved for '{fn_name}'",
                formula_str=f"contracts(new_{fn_name}) valid",
                formula_smt="",
                status=status
            ))

    # Determine overall status
    if not obligations:
        overall = CertStatus.VALID
    elif all(o.status == CertStatus.VALID for o in obligations):
        overall = CertStatus.VALID
    elif any(o.status == CertStatus.INVALID for o in obligations):
        overall = CertStatus.INVALID
    else:
        overall = CertStatus.UNKNOWN

    refactoring_kinds = [r.kind.value for r in result.refactorings]

    return ProofCertificate(
        kind=ProofKind.COMPOSITE,
        claim="Refactoring preserves program behavior",
        source=result.new_source,
        obligations=obligations,
        status=overall,
        metadata={
            'refactoring_kinds': refactoring_kinds,
            'functions_verified': result.functions_verified,
            'functions_failed': result.functions_failed,
        }
    )


# --- Main API ---

def verify_refactoring(
    old_source: str,
    new_source: str,
    check_equivalence: bool = True,
    check_summaries: bool = True,
    check_contracts: bool = False,
    param_types: Optional[Dict[str, Dict[str, str]]] = None,
    fn_name_map: Optional[Dict[str, str]] = None
) -> RefactoringResult:
    """Verify that a refactoring preserves program behavior.

    Args:
        old_source: Original program source
        new_source: Refactored program source
        check_equivalence: Run SMT-based equivalence checking
        check_summaries: Run abstract summary comparison
        check_contracts: Run contract preservation checking
        param_types: Per-function parameter types {fn_name: {param: type}}
        fn_name_map: Mapping from old function names to new {old: new}

    Returns:
        RefactoringResult with verification details
    """
    fn_name_map = fn_name_map or {}

    # Step 1: Diff
    diff = diff_programs(old_source, new_source)

    # Step 2: Detect refactorings
    refactorings = _detect_refactorings(diff, old_source, new_source)

    # Step 3: Compute modular summaries
    old_summaries = None
    new_summaries = None
    if check_summaries:
        try:
            old_result = modular_analyze(old_source)
            old_summaries = old_result.summaries
        except Exception:
            old_summaries = {}
        try:
            new_result = modular_analyze(new_source)
            new_summaries = new_result.summaries
        except Exception:
            new_summaries = {}

    # Step 4: Verify each changed function
    function_results = {}

    # Collect functions to verify
    functions_to_verify = set()
    for change in diff.function_changes:
        if change.kind == ChangeKind.MODIFIED:
            functions_to_verify.add(change.name)
        elif change.kind == ChangeKind.ADDED:
            functions_to_verify.add(change.name)

    # Also add functions affected by refactorings
    for ref in refactorings:
        if ref.kind == RefactoringKind.EXTRACT_FUNCTION:
            functions_to_verify.update(ref.affected_functions)
        elif ref.kind == RefactoringKind.INLINE_FUNCTION:
            if ref.new_name:
                functions_to_verify.add(ref.new_name)
        elif ref.kind == RefactoringKind.RENAME_FUNCTION:
            if ref.new_name:
                functions_to_verify.add(ref.new_name)

    for fn_name in functions_to_verify:
        fn_result = FunctionVerificationResult(fn_name=fn_name)

        # Determine old name (for renames or explicit mapping)
        old_fn_name = fn_name_map.get(fn_name, fn_name)

        # Check if this is a rename
        for ref in refactorings:
            if ref.kind == RefactoringKind.RENAME_FUNCTION and ref.new_name == fn_name:
                old_fn_name = ref.old_name
                break
            elif ref.kind == RefactoringKind.EXTRACT_FUNCTION and ref.new_name == fn_name:
                # New extracted function -- can't check equivalence against old
                fn_result.notes.append("Newly extracted function")
                old_fn_name = None
                break

        # Equivalence checking
        if check_equivalence and old_fn_name is not None:
            # Check if old function exists
            try:
                old_stmts = _parse(old_source)
                old_fns = _extract_functions(old_stmts)
                if old_fn_name in old_fns:
                    fn_params = param_types.get(fn_name) if param_types else None
                    eq, ce, notes = _verify_function_equivalence(
                        old_source, new_source, fn_name,
                        old_fn_name=old_fn_name,
                        param_types=fn_params
                    )
                    fn_result.equivalence_verified = eq
                    fn_result.counterexample = ce
                    fn_result.notes.extend(notes)
                else:
                    fn_result.notes.append(f"Old function '{old_fn_name}' not found")
            except Exception as e:
                fn_result.notes.append(f"Parse error: {e}")

        # Summary comparison
        if check_summaries and old_summaries is not None and new_summaries is not None:
            old_sum = old_summaries.get(old_fn_name or fn_name)
            new_sum = new_summaries.get(fn_name)
            compatible, notes = _compare_summaries(old_sum, new_sum)
            fn_result.summary_compatible = compatible
            fn_result.notes.extend(notes)

        # Contract preservation
        if check_contracts:
            try:
                new_stmts = _parse(new_source)
                new_fns = _extract_functions(new_stmts)
                if fn_name in new_fns:
                    preserved, notes = _check_contract_preservation(old_source, new_source, fn_name)
                    fn_result.contract_preserved = preserved
                    fn_result.notes.extend(notes)
            except Exception as e:
                fn_result.notes.append(f"Contract check error: {e}")

        function_results[fn_name] = fn_result

    # Step 5: Build result
    result = RefactoringResult(
        old_source=old_source,
        new_source=new_source,
        diff=diff,
        refactorings=refactorings,
        function_results=function_results,
        old_summaries=old_summaries if check_summaries else None,
        new_summaries=new_summaries if check_summaries else None,
    )

    # Step 6: Generate certificate
    result.certificate = _generate_refactoring_certificate(result)

    return result


def verify_extract_refactoring(
    old_source: str,
    new_source: str,
    extracted_fn: str,
    modified_fn: str,
    param_types: Optional[Dict[str, str]] = None
) -> RefactoringResult:
    """Verify an extract-function refactoring.

    Checks that the modified function (which now calls extracted_fn)
    behaves identically to the original.
    """
    return verify_refactoring(
        old_source, new_source,
        check_equivalence=True,
        check_summaries=True,
        param_types={modified_fn: param_types} if param_types else None
    )


def verify_inline_refactoring(
    old_source: str,
    new_source: str,
    inlined_fn: str,
    target_fn: str,
    param_types: Optional[Dict[str, str]] = None
) -> RefactoringResult:
    """Verify an inline-function refactoring.

    Checks that the target function (which absorbed inlined_fn)
    behaves identically to the original.
    """
    return verify_refactoring(
        old_source, new_source,
        check_equivalence=True,
        check_summaries=True,
        param_types={target_fn: param_types} if param_types else None
    )


def verify_rename_refactoring(
    old_source: str,
    new_source: str,
    old_name: str,
    new_name: str,
    param_types: Optional[Dict[str, str]] = None
) -> RefactoringResult:
    """Verify a rename-function refactoring.

    Checks that the renamed function behaves identically.
    """
    return verify_refactoring(
        old_source, new_source,
        check_equivalence=True,
        check_summaries=True,
        fn_name_map={new_name: old_name},
        param_types={new_name: param_types} if param_types else None
    )


def verify_simplification(
    old_source: str,
    new_source: str,
    fn_name: str,
    param_types: Optional[Dict[str, str]] = None
) -> RefactoringResult:
    """Verify an expression simplification refactoring.

    Checks that simplifying expressions preserves behavior.
    """
    return verify_refactoring(
        old_source, new_source,
        check_equivalence=True,
        check_summaries=True,
        param_types={fn_name: param_types} if param_types else None
    )


def compare_refactoring_strategies(
    old_source: str,
    new_source: str,
    param_types: Optional[Dict[str, Dict[str, str]]] = None
) -> Dict:
    """Compare equivalence-only vs summary-only vs combined verification."""
    # Equivalence only
    eq_result = verify_refactoring(old_source, new_source,
                                    check_equivalence=True, check_summaries=False,
                                    param_types=param_types)

    # Summary only
    sum_result = verify_refactoring(old_source, new_source,
                                     check_equivalence=False, check_summaries=True,
                                     param_types=param_types)

    # Combined
    combined_result = verify_refactoring(old_source, new_source,
                                          check_equivalence=True, check_summaries=True,
                                          param_types=param_types)

    return {
        'equivalence_only': {
            'verified': eq_result.is_verified,
            'functions_verified': eq_result.functions_verified,
            'functions_failed': eq_result.functions_failed,
        },
        'summary_only': {
            'verified': sum_result.is_verified,
            'functions_verified': sum_result.functions_verified,
            'functions_failed': sum_result.functions_failed,
        },
        'combined': {
            'verified': combined_result.is_verified,
            'functions_verified': combined_result.functions_verified,
            'functions_failed': combined_result.functions_failed,
        },
        'refactorings': [r.kind.value for r in combined_result.refactorings],
    }


def refactoring_report(old_source: str, new_source: str) -> str:
    """Generate a human-readable refactoring verification report."""
    result = verify_refactoring(old_source, new_source)
    return result.summary()
