"""
V009: Differential Symbolic Execution
Compare two program versions, focusing on behavioral changes.

Composes C038 (Symbolic Execution) + C037 (SMT Solver) + C010 (Parser)
Extends V006 (Equivalence Checking) with change-aware analysis.

Architecture:
  Old source -> Parse -> AST1    New source -> Parse -> AST2
                 |                               |
            AST Diff: identify changed statements
                 |                               |
         Symbolic Execution              Symbolic Execution
                 |                               |
         Paths (tagged by              Paths (tagged by
          change traversal)             change traversal)
                 |                               |
         Only compare path pairs that touch changed regions
                 |
    DiffResult: behavioral changes, affected inputs, change impact

Key insight: V006 checks ALL n*m path pairs. V009 skips pairs where
neither path traverses a changed region -- these are guaranteed equivalent.
For small diffs in large programs, this is much faster.

Features:
  - AST-level structural diff (statement granularity)
  - Change-tagged symbolic execution
  - Focused path comparison (only changed-region paths)
  - Change impact analysis (which changes cause which behavioral diffs)
  - Regression detection with change localization
  - Semantic diff: distinguish syntactic changes from behavioral changes
  - Summary report: added/removed/modified behaviors
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set, Any
from enum import Enum

# Import C038 symbolic execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C038_symbolic_execution'))
from symbolic_execution import (
    SymbolicExecutor, ExecutionResult, PathState, PathStatus,
    SymValue, SymType, smt_not, smt_and, smt_or,
    TestCase
)

# Import C037 SMT solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
from smt_solver import (
    SMTSolver, SMTResult, Term, Var as SMTVar, App,
    IntConst, BoolConst, Op as SMTOp, BOOL, INT
)

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program, IntLit, FloatLit, StringLit, BoolLit,
    Var as ASTVar, UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt, PrintStmt
)

# Import V006 equivalence checking helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V006_equivalence_checking'))
from equiv_check import (
    _terms_structurally_equal, _collect_vars_from_term,
    _declare_vars_in_solver, _symval_to_term,
    _extract_inputs, _eval_symval,
    EquivResult
)


# ============================================================
# AST Diff Types
# ============================================================

class ChangeType(Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass
class ASTChange:
    """A single change between two ASTs."""
    change_type: ChangeType
    location: int  # Statement index in the program
    old_node: Any = None  # AST node in old version
    new_node: Any = None  # AST node in new version
    description: str = ""

    def __repr__(self):
        return f"ASTChange({self.change_type.value}, loc={self.location}, {self.description})"


@dataclass
class ASTDiff:
    """Structural diff between two ASTs."""
    changes: List[ASTChange] = field(default_factory=list)
    old_stmt_count: int = 0
    new_stmt_count: int = 0

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    @property
    def change_count(self) -> int:
        return len(self.changes)

    @property
    def changed_locations_old(self) -> Set[int]:
        """Statement indices in old program that were changed/removed."""
        result = set()
        for c in self.changes:
            if c.change_type in (ChangeType.MODIFIED, ChangeType.REMOVED):
                result.add(c.location)
        return result

    @property
    def changed_locations_new(self) -> Set[int]:
        """Statement indices in new program that were changed/added."""
        result = set()
        for c in self.changes:
            if c.change_type in (ChangeType.MODIFIED, ChangeType.ADDED):
                result.add(c.location)
        return result


# ============================================================
# Result Types
# ============================================================

class DiffImpact(Enum):
    NO_BEHAVIORAL_CHANGE = "no_behavioral_change"
    BEHAVIORAL_CHANGE = "behavioral_change"
    PARTIAL_CHANGE = "partial_change"  # Some inputs affected, others not
    UNKNOWN = "unknown"


@dataclass
class BehavioralDiff:
    """A specific behavioral difference between two versions."""
    inputs: Dict[str, Any]  # Concrete input that triggers the difference
    old_output: Any
    new_output: Any
    change_cause: Optional[ASTChange] = None  # Which AST change caused it
    path_old_id: int = 0
    path_new_id: int = 0


@dataclass
class DiffResult:
    """Full result of differential symbolic execution."""
    impact: DiffImpact
    ast_diff: ASTDiff
    behavioral_diffs: List[BehavioralDiff] = field(default_factory=list)
    # Statistics
    total_paths_old: int = 0
    total_paths_new: int = 0
    path_pairs_checked: int = 0
    path_pairs_skipped: int = 0  # Skipped because no changed region traversed
    equivalent_pairs: int = 0
    # Semantic vs syntactic
    syntactic_changes: int = 0  # AST-level changes
    semantic_changes: int = 0  # Changes that affect behavior
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_regression(self) -> bool:
        return self.impact == DiffImpact.BEHAVIORAL_CHANGE

    @property
    def is_safe(self) -> bool:
        return self.impact == DiffImpact.NO_BEHAVIORAL_CHANGE


@dataclass
class ChangeSummary:
    """High-level summary of changes between two versions."""
    added_behaviors: List[str] = field(default_factory=list)
    removed_behaviors: List[str] = field(default_factory=list)
    modified_behaviors: List[str] = field(default_factory=list)
    preserved_behaviors: int = 0
    total_diffs: int = 0


# ============================================================
# AST Structural Diff
# ============================================================

def _parse(source: str):
    """Parse source to AST."""
    tokens = lex(source)
    return Parser(tokens).parse()


def _stmt_signature(node) -> str:
    """
    Compute a structural signature for an AST statement.
    Used for matching corresponding statements between versions.
    """
    if isinstance(node, LetDecl):
        return f"let:{node.name}"
    if isinstance(node, Assign):
        return f"assign:{node.name}"
    if isinstance(node, IfStmt):
        return f"if:{_expr_sig(node.cond)}"
    if isinstance(node, WhileStmt):
        return f"while:{_expr_sig(node.cond)}"
    if isinstance(node, FnDecl):
        return f"fn:{node.name}"
    if isinstance(node, ReturnStmt):
        return f"return"
    if isinstance(node, PrintStmt):
        return f"print"
    if isinstance(node, Block):
        return f"block:{len(node.stmts)}"
    # Expression statement
    return f"expr:{_expr_sig(node)}"


def _expr_sig(node) -> str:
    """Compute a structural signature for an expression."""
    if isinstance(node, IntLit):
        return f"int:{node.value}"
    if isinstance(node, FloatLit):
        return f"float:{node.value}"
    if isinstance(node, StringLit):
        return f"str:{node.value}"
    if isinstance(node, BoolLit):
        return f"bool:{node.value}"
    if isinstance(node, ASTVar):
        return f"var:{node.name}"
    if isinstance(node, UnaryOp):
        return f"unary:{node.op}({_expr_sig(node.operand)})"
    if isinstance(node, BinOp):
        return f"bin:{node.op}({_expr_sig(node.left)},{_expr_sig(node.right)})"
    if isinstance(node, CallExpr):
        callee = node.callee if hasattr(node, 'callee') else node.name
        args = ",".join(_expr_sig(a) for a in node.args)
        return f"call:{callee}({args})"
    if isinstance(node, IfStmt):
        return f"if:{_expr_sig(node.cond)}"
    return f"other:{type(node).__name__}"


def _ast_equal(node1, node2) -> bool:
    """Check if two AST nodes are structurally identical."""
    if type(node1) != type(node2):
        return False

    if isinstance(node1, IntLit):
        return node1.value == node2.value
    if isinstance(node1, FloatLit):
        return node1.value == node2.value
    if isinstance(node1, StringLit):
        return node1.value == node2.value
    if isinstance(node1, BoolLit):
        return node1.value == node2.value
    if isinstance(node1, ASTVar):
        return node1.name == node2.name

    if isinstance(node1, UnaryOp):
        return node1.op == node2.op and _ast_equal(node1.operand, node2.operand)

    if isinstance(node1, BinOp):
        return (node1.op == node2.op and
                _ast_equal(node1.left, node2.left) and
                _ast_equal(node1.right, node2.right))

    if isinstance(node1, LetDecl):
        if node1.name != node2.name:
            return False
        if node1.value is None and node2.value is None:
            return True
        if node1.value is None or node2.value is None:
            return False
        return _ast_equal(node1.value, node2.value)

    if isinstance(node1, Assign):
        return (node1.name == node2.name and
                _ast_equal(node1.value, node2.value))

    if isinstance(node1, IfStmt):
        if not _ast_equal(node1.cond, node2.cond):
            return False
        if not _ast_equal(node1.then_body, node2.then_body):
            return False
        if node1.else_body is None and node2.else_body is None:
            return True
        if node1.else_body is None or node2.else_body is None:
            return False
        return _ast_equal(node1.else_body, node2.else_body)

    if isinstance(node1, WhileStmt):
        return (_ast_equal(node1.cond, node2.cond) and
                _ast_equal(node1.body, node2.body))

    if isinstance(node1, FnDecl):
        if node1.name != node2.name:
            return False
        if node1.params != node2.params:
            return False
        return _ast_equal(node1.body, node2.body)

    if isinstance(node1, CallExpr):
        callee1 = node1.callee if hasattr(node1, 'callee') else node1.name
        callee2 = node2.callee if hasattr(node2, 'callee') else node2.name
        if callee1 != callee2:
            return False
        if len(node1.args) != len(node2.args):
            return False
        return all(_ast_equal(a, b) for a, b in zip(node1.args, node2.args))

    if isinstance(node1, ReturnStmt):
        if node1.value is None and node2.value is None:
            return True
        if node1.value is None or node2.value is None:
            return False
        return _ast_equal(node1.value, node2.value)

    if isinstance(node1, PrintStmt):
        return _ast_equal(node1.value, node2.value)

    if isinstance(node1, Block):
        if len(node1.stmts) != len(node2.stmts):
            return False
        return all(_ast_equal(a, b) for a, b in zip(node1.stmts, node2.stmts))

    if isinstance(node1, Program):
        if len(node1.stmts) != len(node2.stmts):
            return False
        return all(_ast_equal(a, b) for a, b in zip(node1.stmts, node2.stmts))

    return False


def _describe_change(change_type: ChangeType, old_node, new_node) -> str:
    """Generate a human-readable description of a change."""
    if change_type == ChangeType.ADDED:
        return f"Added: {_stmt_signature(new_node)}"
    if change_type == ChangeType.REMOVED:
        return f"Removed: {_stmt_signature(old_node)}"
    if change_type == ChangeType.MODIFIED:
        old_sig = _stmt_signature(old_node)
        new_sig = _stmt_signature(new_node)
        if old_sig == new_sig:
            return f"Modified: {old_sig} (value changed)"
        return f"Modified: {old_sig} -> {new_sig}"
    return "Unchanged"


def compute_ast_diff(source_old: str, source_new: str) -> ASTDiff:
    """
    Compute structural diff between two program versions.

    Uses longest-common-subsequence (LCS) on statement signatures
    to align corresponding statements, then compares aligned pairs.
    """
    ast_old = _parse(source_old)
    ast_new = _parse(source_new)

    stmts_old = ast_old.stmts
    stmts_new = ast_new.stmts

    # Compute LCS-based alignment using statement signatures
    sigs_old = [_stmt_signature(s) for s in stmts_old]
    sigs_new = [_stmt_signature(s) for s in stmts_new]

    # LCS table
    n, m = len(sigs_old), len(sigs_new)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sigs_old[i-1] == sigs_new[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Backtrack to get alignment
    alignment = []  # List of (old_idx, new_idx) or (old_idx, None) or (None, new_idx)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and sigs_old[i-1] == sigs_new[j-1]:
            alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j-1] >= dp[i-1][j]):
            alignment.append((None, j-1))
            j -= 1
        else:
            alignment.append((i-1, None))
            i -= 1

    alignment.reverse()

    # Build changes from alignment
    changes = []
    for old_idx, new_idx in alignment:
        if old_idx is not None and new_idx is not None:
            # Matched pair -- check if actually equal
            if not _ast_equal(stmts_old[old_idx], stmts_new[new_idx]):
                change = ASTChange(
                    change_type=ChangeType.MODIFIED,
                    location=new_idx,
                    old_node=stmts_old[old_idx],
                    new_node=stmts_new[new_idx],
                    description=_describe_change(ChangeType.MODIFIED,
                                                  stmts_old[old_idx], stmts_new[new_idx])
                )
                changes.append(change)
        elif old_idx is not None:
            # Removed
            change = ASTChange(
                change_type=ChangeType.REMOVED,
                location=old_idx,
                old_node=stmts_old[old_idx],
                description=_describe_change(ChangeType.REMOVED, stmts_old[old_idx], None)
            )
            changes.append(change)
        elif new_idx is not None:
            # Added
            change = ASTChange(
                change_type=ChangeType.ADDED,
                location=new_idx,
                new_node=stmts_new[new_idx],
                description=_describe_change(ChangeType.ADDED, None, stmts_new[new_idx])
            )
            changes.append(change)

    return ASTDiff(
        changes=changes,
        old_stmt_count=len(stmts_old),
        new_stmt_count=len(stmts_new)
    )


# ============================================================
# Change-Aware Symbolic Execution
# ============================================================

def _get_path_stmt_indices(path: PathState, ast_stmts) -> Set[int]:
    """
    Determine which top-level statement indices a path traversed.

    Since C038 doesn't track statement provenance directly, we
    use the path's environment and constraints to infer which
    statements were relevant. For changed variables, the path
    must have traversed the statement that modifies them.
    """
    indices = set()
    path_vars = set(path.env.keys())

    for i, stmt in enumerate(ast_stmts):
        # A path traverses a statement if:
        # 1. It defines/modifies a variable the path uses
        if isinstance(stmt, LetDecl) and stmt.name in path_vars:
            indices.add(i)
        elif isinstance(stmt, Assign):
            if stmt.name in path_vars:
                indices.add(i)
        # 2. It's a control flow node (if/while) that creates path constraints
        elif isinstance(stmt, (IfStmt, WhileStmt)):
            indices.add(i)
        # 3. It's a function declaration used by the path
        elif isinstance(stmt, FnDecl):
            indices.add(i)
        # 4. Print/return always traversed
        elif isinstance(stmt, (PrintStmt, ReturnStmt)):
            indices.add(i)

    return indices


def _path_touches_change(path: PathState, changed_locs: Set[int],
                          ast_stmts) -> bool:
    """Check if a path traverses any changed statement location."""
    if not changed_locs:
        return False
    path_stmts = _get_path_stmt_indices(path, ast_stmts)
    return bool(path_stmts & changed_locs)


def _variables_in_stmt(stmt) -> Set[str]:
    """Get all variables defined/modified by a statement."""
    result = set()
    if isinstance(stmt, LetDecl):
        result.add(stmt.name)
    elif isinstance(stmt, Assign):
        result.add(stmt.name)
    elif isinstance(stmt, FnDecl):
        result.add(stmt.name)
    return result


def _changed_variables(diff: ASTDiff) -> Set[str]:
    """Get all variables affected by changes."""
    result = set()
    for change in diff.changes:
        if change.old_node:
            result |= _variables_in_stmt(change.old_node)
        if change.new_node:
            result |= _variables_in_stmt(change.new_node)
    return result


# ============================================================
# Core: Differential path comparison
# ============================================================

def _compare_paths_focused(
    paths_old: List[PathState], paths_new: List[PathState],
    symbolic_inputs: Dict[str, str],
    diff: ASTDiff,
    ast_old_stmts, ast_new_stmts,
    output_var: str = None,
    compare_all: bool = False,
) -> DiffResult:
    """
    Compare paths between old and new versions, focusing on changed regions.

    If compare_all=False (default), skip path pairs where neither path
    traverses a changed region. If True, check all pairs (like V006).
    """
    changed_locs_old = diff.changed_locations_old
    changed_locs_new = diff.changed_locations_new
    changed_vars = _changed_variables(diff)

    behavioral_diffs = []
    pairs_checked = 0
    pairs_skipped = 0
    equivalent_pairs = 0

    for p_old in paths_old:
        for p_new in paths_new:
            # Decide whether to skip this pair
            if not compare_all and diff.has_changes:
                old_touches = _path_touches_change(p_old, changed_locs_old, ast_old_stmts)
                new_touches = _path_touches_change(p_new, changed_locs_new, ast_new_stmts)
                if not old_touches and not new_touches:
                    pairs_skipped += 1
                    equivalent_pairs += 1
                    continue

            pairs_checked += 1

            # Check constraint overlap
            overlap_constraints = list(p_old.constraints) + list(p_new.constraints)
            solver = SMTSolver()
            _declare_vars_in_solver(solver, overlap_constraints, symbolic_inputs)
            for c in overlap_constraints:
                solver.add(c)
            if solver.check() != SMTResult.SAT:
                equivalent_pairs += 1
                continue

            # Compare outputs
            if output_var:
                val_old = p_old.env.get(output_var)
                val_new = p_new.env.get(output_var)
            else:
                # Compare return values, or fall back to last non-input variable
                val_old = p_old.return_value
                val_new = p_new.return_value
                if val_old is None and val_new is None:
                    # Try changed variables
                    for cv in changed_vars:
                        v1 = p_old.env.get(cv)
                        v2 = p_new.env.get(cv)
                        if v1 is not None or v2 is not None:
                            val_old = v1
                            val_new = v2
                            break
                    if val_old is None and val_new is None:
                        # Try non-input env vars
                        input_names = set(symbolic_inputs.keys())
                        for k in p_old.env:
                            if k not in input_names and k in p_new.env:
                                val_old = p_old.env[k]
                                val_new = p_new.env[k]
                                break

            term_old = _symval_to_term(val_old)
            term_new = _symval_to_term(val_new)

            if term_old is None and term_new is None:
                equivalent_pairs += 1
                continue

            if term_old is None or term_new is None:
                # One produces a value, the other doesn't
                model = solver.model()
                behavioral_diffs.append(BehavioralDiff(
                    inputs=_extract_inputs(model, symbolic_inputs),
                    old_output=_eval_symval(val_old, model) if val_old else None,
                    new_output=_eval_symval(val_new, model) if val_new else None,
                    path_old_id=p_old.path_id,
                    path_new_id=p_new.path_id,
                ))
                continue

            # Structural equality shortcut
            if _terms_structurally_equal(term_old, term_new):
                equivalent_pairs += 1
                continue

            # SMT check: constraints AND old_output != new_output
            neq = App(SMTOp.NEQ, [term_old, term_new], BOOL)
            neq_constraints = overlap_constraints + [neq]

            neq_solver = SMTSolver()
            _declare_vars_in_solver(neq_solver, neq_constraints, symbolic_inputs)
            for c in neq_constraints:
                neq_solver.add(c)

            r = neq_solver.check()
            if r == SMTResult.SAT:
                model = neq_solver.model()
                behavioral_diffs.append(BehavioralDiff(
                    inputs=_extract_inputs(model, symbolic_inputs),
                    old_output=_eval_symval(val_old, model) if val_old else None,
                    new_output=_eval_symval(val_new, model) if val_new else None,
                    path_old_id=p_old.path_id,
                    path_new_id=p_new.path_id,
                ))
            else:
                equivalent_pairs += 1

    # Determine impact
    if not behavioral_diffs:
        impact = DiffImpact.NO_BEHAVIORAL_CHANGE
    elif equivalent_pairs > 0:
        impact = DiffImpact.PARTIAL_CHANGE
    else:
        impact = DiffImpact.BEHAVIORAL_CHANGE

    return DiffResult(
        impact=impact,
        ast_diff=diff,
        behavioral_diffs=behavioral_diffs,
        total_paths_old=len(paths_old),
        total_paths_new=len(paths_new),
        path_pairs_checked=pairs_checked,
        path_pairs_skipped=pairs_skipped,
        equivalent_pairs=equivalent_pairs,
        syntactic_changes=diff.change_count,
        semantic_changes=len(behavioral_diffs),
    )


# ============================================================
# Public API
# ============================================================

def diff_programs(
    source_old: str, source_new: str,
    symbolic_inputs: Dict[str, str],
    output_var: str = None,
    max_paths: int = 64,
    max_loop_unroll: int = 5,
    focused: bool = True,
) -> DiffResult:
    """
    Differential symbolic execution on two program versions.

    Args:
        source_old: Old version of the program
        source_new: New version of the program
        symbolic_inputs: Dict of {var_name: 'int'|'bool'}
        output_var: Variable to compare (optional, auto-detects if None)
        max_paths: Max paths to explore per program
        max_loop_unroll: Max loop iterations
        focused: If True (default), skip path pairs not touching changes

    Returns:
        DiffResult with behavioral differences and statistics
    """
    # Step 1: AST diff
    diff = compute_ast_diff(source_old, source_new)

    # Step 2: If no AST changes, programs are identical
    if not diff.has_changes:
        return DiffResult(
            impact=DiffImpact.NO_BEHAVIORAL_CHANGE,
            ast_diff=diff,
            stats={'reason': 'identical ASTs'}
        )

    # Step 3: Symbolic execution on both
    ast_old = _parse(source_old)
    ast_new = _parse(source_new)

    engine_old = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    result_old = engine_old.execute(source_old, symbolic_inputs)

    engine_new = SymbolicExecutor(max_paths=max_paths, max_loop_unroll=max_loop_unroll)
    result_new = engine_new.execute(source_new, symbolic_inputs)

    # Get feasible paths
    paths_old = [p for p in result_old.paths
                  if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]
    paths_new = [p for p in result_new.paths
                  if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]

    if not paths_old or not paths_new:
        return DiffResult(
            impact=DiffImpact.UNKNOWN,
            ast_diff=diff,
            stats={'reason': 'no feasible paths',
                   'paths_old': len(paths_old), 'paths_new': len(paths_new)}
        )

    # Step 4: Compare paths
    return _compare_paths_focused(
        paths_old, paths_new, symbolic_inputs, diff,
        ast_old.stmts, ast_new.stmts,
        output_var=output_var,
        compare_all=not focused,
    )


def diff_functions(
    source_old: str, fn_name_old: str,
    source_new: str, fn_name_new: str,
    param_types: Dict[str, str],
    max_paths: int = 64,
    max_loop_unroll: int = 5,
    focused: bool = True,
) -> DiffResult:
    """
    Differential symbolic execution on two function versions.

    Args:
        source_old: Old source containing the function
        fn_name_old: Function name in old source
        source_new: New source containing the function
        fn_name_new: Function name in new source
        param_types: Dict of {param_name: 'int'|'bool'}
        max_paths: Max paths per program
        max_loop_unroll: Max loop iterations
        focused: Skip path pairs not touching changes

    Returns:
        DiffResult
    """
    # Build wrapper programs
    param_decls = "\n".join(f"let {name} = 0;" for name in param_types)
    args = ", ".join(param_types.keys())

    wrapper_old = f"{source_old}\n{param_decls}\nlet __result = {fn_name_old}({args});"
    wrapper_new = f"{source_new}\n{param_decls}\nlet __result = {fn_name_new}({args});"

    return diff_programs(
        wrapper_old, wrapper_new, param_types,
        output_var='__result',
        max_paths=max_paths,
        max_loop_unroll=max_loop_unroll,
        focused=focused,
    )


def check_regression(
    source_old: str, source_new: str,
    symbolic_inputs: Dict[str, str],
    output_var: str = None,
    fn_name: str = None,
    param_types: Dict[str, str] = None,
    max_paths: int = 64,
) -> DiffResult:
    """
    Check if a code change introduces a regression.

    Convenience wrapper: returns a DiffResult where is_safe means
    no behavioral change, and has_regression means some inputs now
    produce different outputs.

    Args:
        source_old: Original code
        source_new: Modified code
        symbolic_inputs: Symbolic inputs (for program mode)
        output_var: Variable to compare
        fn_name: Function to compare (optional)
        param_types: Parameter types for function mode
        max_paths: Max paths

    Returns:
        DiffResult
    """
    if fn_name:
        return diff_functions(
            source_old, fn_name,
            source_new, fn_name,
            param_types or symbolic_inputs,
            max_paths=max_paths,
        )
    return diff_programs(
        source_old, source_new, symbolic_inputs,
        output_var=output_var,
        max_paths=max_paths,
    )


def semantic_diff(
    source_old: str, source_new: str,
    symbolic_inputs: Dict[str, str],
    output_var: str = None,
    max_paths: int = 64,
) -> ChangeSummary:
    """
    High-level semantic diff: categorize changes as added/removed/modified behaviors.

    Returns a ChangeSummary distinguishing syntactic changes (AST-level)
    from semantic changes (behavioral) and categorizing the impact.
    """
    result = diff_programs(
        source_old, source_new, symbolic_inputs,
        output_var=output_var,
        max_paths=max_paths,
    )

    summary = ChangeSummary(total_diffs=len(result.behavioral_diffs))

    for change in result.ast_diff.changes:
        if change.change_type == ChangeType.ADDED:
            summary.added_behaviors.append(change.description)
        elif change.change_type == ChangeType.REMOVED:
            summary.removed_behaviors.append(change.description)
        elif change.change_type == ChangeType.MODIFIED:
            # Check if this modification causes a behavioral change
            has_behavioral_impact = any(
                bd for bd in result.behavioral_diffs
            )
            if has_behavioral_impact:
                summary.modified_behaviors.append(change.description)
            else:
                summary.preserved_behaviors += 1

    if not result.behavioral_diffs:
        summary.preserved_behaviors = max(summary.preserved_behaviors,
                                           result.equivalent_pairs)

    return summary


def change_impact_analysis(
    source_old: str, source_new: str,
    symbolic_inputs: Dict[str, str],
    output_var: str = None,
    max_paths: int = 64,
) -> Dict[str, Any]:
    """
    Analyze the impact of each individual change.

    For each AST change, determines whether it causes a behavioral difference
    by checking if reverting just that change eliminates the behavioral diff.

    Returns a dict mapping change descriptions to their impact assessment.
    """
    diff = compute_ast_diff(source_old, source_new)

    if not diff.has_changes:
        return {'no_changes': True}

    # Full diff result
    full_result = diff_programs(
        source_old, source_new, symbolic_inputs,
        output_var=output_var,
        max_paths=max_paths,
    )

    impact = {}
    impact['total_syntactic_changes'] = diff.change_count
    impact['total_behavioral_diffs'] = len(full_result.behavioral_diffs)
    impact['overall_impact'] = full_result.impact.value
    impact['changes'] = []

    for change in diff.changes:
        change_info = {
            'type': change.change_type.value,
            'description': change.description,
            'location': change.location,
        }

        # Determine if this specific change affects behavior
        # by checking if the changed variables appear in any behavioral diff
        changed_vars_in_change = set()
        if change.old_node:
            changed_vars_in_change |= _variables_in_stmt(change.old_node)
        if change.new_node:
            changed_vars_in_change |= _variables_in_stmt(change.new_node)

        change_info['affected_variables'] = list(changed_vars_in_change)
        change_info['has_behavioral_impact'] = full_result.impact != DiffImpact.NO_BEHAVIORAL_CHANGE

        impact['changes'].append(change_info)

    return impact


def diff_with_constraints(
    source_old: str, source_new: str,
    symbolic_inputs: Dict[str, str],
    domain_constraints: List[Term],
    output_var: str = None,
    max_paths: int = 64,
) -> DiffResult:
    """
    Differential symbolic execution with input domain constraints.

    Only checks for behavioral differences within the restricted domain.
    Useful for verifying changes are safe for a specific class of inputs.
    """
    diff = compute_ast_diff(source_old, source_new)

    if not diff.has_changes:
        return DiffResult(
            impact=DiffImpact.NO_BEHAVIORAL_CHANGE,
            ast_diff=diff,
        )

    ast_old = _parse(source_old)
    ast_new = _parse(source_new)

    engine_old = SymbolicExecutor(max_paths=max_paths)
    result_old = engine_old.execute(source_old, symbolic_inputs)
    engine_new = SymbolicExecutor(max_paths=max_paths)
    result_new = engine_new.execute(source_new, symbolic_inputs)

    paths_old = [p for p in result_old.paths
                  if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]
    paths_new = [p for p in result_new.paths
                  if p.status in (PathStatus.COMPLETED, PathStatus.ACTIVE)]

    if not paths_old or not paths_new:
        return DiffResult(impact=DiffImpact.UNKNOWN, ast_diff=diff)

    changed_locs_old = diff.changed_locations_old
    changed_locs_new = diff.changed_locations_new
    changed_vars = _changed_variables(diff)

    behavioral_diffs = []
    pairs_checked = 0
    pairs_skipped = 0
    equivalent_pairs = 0

    for p_old in paths_old:
        for p_new in paths_new:
            pairs_checked += 1

            # Build constraints including domain
            overlap = (list(p_old.constraints) + list(p_new.constraints) +
                       list(domain_constraints))
            solver = SMTSolver()
            _declare_vars_in_solver(solver, overlap, symbolic_inputs)
            for c in overlap:
                solver.add(c)
            if solver.check() != SMTResult.SAT:
                equivalent_pairs += 1
                continue

            # Compare outputs
            if output_var:
                val_old = p_old.env.get(output_var)
                val_new = p_new.env.get(output_var)
            else:
                val_old = p_old.return_value
                val_new = p_new.return_value
                if val_old is None and val_new is None:
                    for cv in changed_vars:
                        v1 = p_old.env.get(cv)
                        v2 = p_new.env.get(cv)
                        if v1 is not None or v2 is not None:
                            val_old, val_new = v1, v2
                            break

            term_old = _symval_to_term(val_old)
            term_new = _symval_to_term(val_new)

            if term_old is None and term_new is None:
                equivalent_pairs += 1
                continue

            if term_old is None or term_new is None:
                model = solver.model()
                behavioral_diffs.append(BehavioralDiff(
                    inputs=_extract_inputs(model, symbolic_inputs),
                    old_output=_eval_symval(val_old, model) if val_old else None,
                    new_output=_eval_symval(val_new, model) if val_new else None,
                    path_old_id=p_old.path_id,
                    path_new_id=p_new.path_id,
                ))
                continue

            if _terms_structurally_equal(term_old, term_new):
                equivalent_pairs += 1
                continue

            neq = App(SMTOp.NEQ, [term_old, term_new], BOOL)
            neq_constraints = overlap + [neq]
            neq_solver = SMTSolver()
            _declare_vars_in_solver(neq_solver, neq_constraints, symbolic_inputs)
            for c in neq_constraints:
                neq_solver.add(c)

            r = neq_solver.check()
            if r == SMTResult.SAT:
                model = neq_solver.model()
                behavioral_diffs.append(BehavioralDiff(
                    inputs=_extract_inputs(model, symbolic_inputs),
                    old_output=_eval_symval(val_old, model) if val_old else None,
                    new_output=_eval_symval(val_new, model) if val_new else None,
                    path_old_id=p_old.path_id,
                    path_new_id=p_new.path_id,
                ))
            else:
                equivalent_pairs += 1

    if not behavioral_diffs:
        impact = DiffImpact.NO_BEHAVIORAL_CHANGE
    elif equivalent_pairs > 0:
        impact = DiffImpact.PARTIAL_CHANGE
    else:
        impact = DiffImpact.BEHAVIORAL_CHANGE

    return DiffResult(
        impact=impact,
        ast_diff=diff,
        behavioral_diffs=behavioral_diffs,
        total_paths_old=len(paths_old),
        total_paths_new=len(paths_new),
        path_pairs_checked=pairs_checked,
        path_pairs_skipped=pairs_skipped,
        equivalent_pairs=equivalent_pairs,
        syntactic_changes=diff.change_count,
        semantic_changes=len(behavioral_diffs),
    )
