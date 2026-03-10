"""
Static Analyzer for the Stack VM Language
Challenge C025 -- AgentZero Session 026

Composes C013 (Type Checker) + C014 (Bytecode Optimizer) into a comprehensive
static analysis tool.

Architecture:
  Source -> Parse -> TypeCheck -> Compile -> Optimize(analysis) -> Report

Analysis passes:
  1. Type analysis -- type errors, type-aware warnings (from C013)
  2. Dead code detection -- unreachable code found by optimizer (from C014)
  3. Complexity analysis -- cyclomatic complexity, nesting depth
  4. Lint rules -- unused variables, shadowed names, style warnings
  5. Optimization suggestions -- what the optimizer would improve
  6. Code metrics -- LOC, functions, variables, expressions

Composition boundary: C013 provides typed AST + errors, C014 provides
bytecode-level analysis. This analyzer bridges both into unified diagnostics.

C010 AST field names (critical -- do NOT confuse):
  - IfStmt: cond, then_body, else_body
  - WhileStmt: cond, body
  - LetDecl: name, value
  - Assign: name, value
  - PrintStmt: value
  - FnDecl: name, params (list of str or TypedParam), body
  - BinOp: op, left, right (field order: op first!)
  - CallExpr: callee (str), args
  - ReturnStmt: value
  - No ExprStmt wrapper -- bare expressions appear in stmts list
  - No LogicalOp -- and/or use BinOp
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

# Import composed systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C013_type_checker'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C014_bytecode_optimizer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from type_checker import (
    parse, check_source, check_program, format_errors,
    TypeChecker, TypeEnv, TypeError_,
    TInt, TFloat, TString, TBool, TVoid, TFunc, TVar,
    INT, FLOAT, STRING, BOOL, VOID,
    resolve,
)

from optimizer import (
    optimize_source, optimize_chunk, optimize_all,
    OptimizationStats, decode_chunk, Instr, Op,
)

from stack_vm import (
    lex, Parser, Compiler, compile_source, Chunk,
    Program, LetDecl, Assign, FnDecl, IfStmt, WhileStmt,
    ReturnStmt, Block, IntLit, FloatLit, StringLit, BoolLit,
    Var, BinOp, UnaryOp, CallExpr, PrintStmt,
)


# ============================================================
# Severity and Diagnostic Types
# ============================================================

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Category(Enum):
    TYPE_ERROR = "type-error"
    DEAD_CODE = "dead-code"
    UNUSED_VAR = "unused-variable"
    SHADOWED_VAR = "shadowed-variable"
    COMPLEXITY = "complexity"
    STYLE = "style"
    OPTIMIZATION = "optimization"
    UNREACHABLE = "unreachable"
    REDUNDANT = "redundant"
    POSSIBLE_BUG = "possible-bug"


@dataclass
class Diagnostic:
    """A single analysis finding."""
    severity: Severity
    category: Category
    message: str
    line: int = 0
    column: int = 0
    suggestion: str = ""

    def __repr__(self):
        sev = self.severity.value.upper()
        cat = self.category.value
        loc = f"line {self.line}" if self.line > 0 else "global"
        base = f"[{sev}] {cat}: {self.message} ({loc})"
        if self.suggestion:
            base += f" -- suggestion: {self.suggestion}"
        return base


# ============================================================
# Code Metrics
# ============================================================

@dataclass
class CodeMetrics:
    """Quantitative code measurements."""
    total_lines: int = 0
    code_lines: int = 0
    blank_lines: int = 0
    functions: int = 0
    variables: int = 0
    statements: int = 0
    expressions: int = 0
    max_nesting: int = 0
    cyclomatic_complexity: int = 1  # starts at 1 (base path)
    max_function_complexity: int = 0
    function_complexities: dict = field(default_factory=dict)
    avg_function_length: float = 0.0


# ============================================================
# AST Node Helpers
# ============================================================

def _is_stmt(node):
    """Check if a node is a statement type."""
    return isinstance(node, (LetDecl, Assign, IfStmt, WhileStmt,
                             ReturnStmt, PrintStmt, FnDecl, Block))


def _is_expr(node):
    """Check if a node is an expression type."""
    return isinstance(node, (IntLit, FloatLit, StringLit, BoolLit,
                             Var, BinOp, UnaryOp, CallExpr))


def _get_line(node):
    """Get line number from AST node."""
    if hasattr(node, 'line'):
        return node.line
    return 0


def _get_param_name(param):
    """Extract parameter name (handles both str and TypedParam)."""
    if isinstance(param, str):
        return param
    if hasattr(param, 'name'):
        return param.name
    return str(param)


# ============================================================
# AST Walking
# ============================================================

def _walk_ast(node, visitor, depth=0):
    """Walk AST nodes, calling visitor(node, depth) on each."""
    if node is None:
        return
    visitor(node, depth)

    if isinstance(node, Program):
        for stmt in node.stmts:
            _walk_ast(stmt, visitor, depth)
    elif isinstance(node, Block):
        for stmt in node.stmts:
            _walk_ast(stmt, visitor, depth + 1)
    elif isinstance(node, LetDecl):
        _walk_ast(node.value, visitor, depth)
    elif isinstance(node, Assign):
        _walk_ast(node.value, visitor, depth)
    elif isinstance(node, FnDecl):
        _walk_ast(node.body, visitor, depth + 1)
    elif isinstance(node, IfStmt):
        _walk_ast(node.cond, visitor, depth)
        _walk_ast(node.then_body, visitor, depth + 1)
        if node.else_body:
            _walk_ast(node.else_body, visitor, depth + 1)
    elif isinstance(node, WhileStmt):
        _walk_ast(node.cond, visitor, depth)
        _walk_ast(node.body, visitor, depth + 1)
    elif isinstance(node, ReturnStmt):
        if node.value:
            _walk_ast(node.value, visitor, depth)
    elif isinstance(node, PrintStmt):
        _walk_ast(node.value, visitor, depth)
    elif isinstance(node, BinOp):
        _walk_ast(node.left, visitor, depth)
        _walk_ast(node.right, visitor, depth)
    elif isinstance(node, UnaryOp):
        _walk_ast(node.operand, visitor, depth)
    elif isinstance(node, CallExpr):
        for arg in node.args:
            _walk_ast(arg, visitor, depth)
    # Bare expression as statement in stmts list -- it's an expr node directly
    # (no ExprStmt wrapper in C010). We already handle all expr types above.


def _walk_with_post(node, pre_visit, post_visit, depth=0):
    """Walk AST with both pre-visit and post-visit callbacks."""
    if node is None:
        return
    pre_visit(node, depth)

    if isinstance(node, Program):
        for stmt in node.stmts:
            _walk_with_post(stmt, pre_visit, post_visit, depth)
    elif isinstance(node, Block):
        for stmt in node.stmts:
            _walk_with_post(stmt, pre_visit, post_visit, depth + 1)
    elif isinstance(node, LetDecl):
        _walk_with_post(node.value, pre_visit, post_visit, depth)
    elif isinstance(node, Assign):
        _walk_with_post(node.value, pre_visit, post_visit, depth)
    elif isinstance(node, FnDecl):
        _walk_with_post(node.body, pre_visit, post_visit, depth + 1)
    elif isinstance(node, IfStmt):
        _walk_with_post(node.cond, pre_visit, post_visit, depth)
        _walk_with_post(node.then_body, pre_visit, post_visit, depth + 1)
        if node.else_body:
            _walk_with_post(node.else_body, pre_visit, post_visit, depth + 1)
    elif isinstance(node, WhileStmt):
        _walk_with_post(node.cond, pre_visit, post_visit, depth)
        _walk_with_post(node.body, pre_visit, post_visit, depth + 1)
    elif isinstance(node, ReturnStmt):
        if node.value:
            _walk_with_post(node.value, pre_visit, post_visit, depth)
    elif isinstance(node, PrintStmt):
        _walk_with_post(node.value, pre_visit, post_visit, depth)
    elif isinstance(node, BinOp):
        _walk_with_post(node.left, pre_visit, post_visit, depth)
        _walk_with_post(node.right, pre_visit, post_visit, depth)
    elif isinstance(node, UnaryOp):
        _walk_with_post(node.operand, pre_visit, post_visit, depth)
    elif isinstance(node, CallExpr):
        for arg in node.args:
            _walk_with_post(arg, pre_visit, post_visit, depth)

    post_visit(node, depth)


def _count_stmts_in_body(node):
    """Count statements in a function body recursively."""
    if isinstance(node, Block):
        count = 0
        for s in node.stmts:
            count += 1
            if isinstance(s, IfStmt):
                count += _count_stmts_in_body(s.then_body)
                if s.else_body:
                    count += _count_stmts_in_body(s.else_body)
            elif isinstance(s, WhileStmt):
                count += _count_stmts_in_body(s.body)
        return count
    return 1


# ============================================================
# Analysis Pass: Variable Tracking
# ============================================================

class VariableTracker:
    """Track variable definitions and usages for unused/shadowed detection."""

    def __init__(self):
        self.scopes = [{}]  # stack of {name: {"line": int, "used": bool}}
        self.diagnostics = []

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        scope = self.scopes.pop()
        for name, info in scope.items():
            if not info["used"] and not name.startswith("_"):
                self.diagnostics.append(Diagnostic(
                    severity=Severity.WARNING,
                    category=Category.UNUSED_VAR,
                    message=f"Variable '{name}' is declared but never used",
                    line=info["line"],
                    suggestion=f"Remove or prefix with underscore: _{name}",
                ))

    def define(self, name, line):
        # Check for shadowing in outer scopes
        for scope in reversed(self.scopes[:-1]):
            if name in scope:
                self.diagnostics.append(Diagnostic(
                    severity=Severity.WARNING,
                    category=Category.SHADOWED_VAR,
                    message=f"Variable '{name}' shadows outer variable defined at line {scope[name]['line']}",
                    line=line,
                ))
                break
        self.scopes[-1][name] = {"line": line, "used": False}

    def use(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name]["used"] = True
                return


def _analyze_variables(ast):
    """Detect unused variables and shadowed names."""
    tracker = VariableTracker()

    def pre_visit(node, depth):
        if isinstance(node, FnDecl):
            tracker.push_scope()
            for param in node.params:
                pname = _get_param_name(param)
                tracker.define(pname, _get_line(node))
        elif isinstance(node, LetDecl):
            tracker.define(node.name, _get_line(node))
        elif isinstance(node, Var):
            tracker.use(node.name)
        elif isinstance(node, Assign):
            tracker.use(node.name)  # assignment counts as usage

    def post_visit(node, depth):
        if isinstance(node, FnDecl):
            tracker.pop_scope()

    _walk_with_post(ast, pre_visit, post_visit)

    # Check top-level scope
    scope = tracker.scopes[0]
    for name, info in scope.items():
        if not info["used"] and not name.startswith("_"):
            tracker.diagnostics.append(Diagnostic(
                severity=Severity.WARNING,
                category=Category.UNUSED_VAR,
                message=f"Variable '{name}' is declared but never used",
                line=info["line"],
                suggestion=f"Remove or prefix with underscore: _{name}",
            ))

    return tracker.diagnostics


# ============================================================
# Analysis Pass: Complexity
# ============================================================

def _analyze_complexity(ast):
    """Compute cyclomatic complexity and nesting metrics."""
    metrics = CodeMetrics()
    current_fn = [None]
    fn_complexity = {}
    max_depth = [0]

    def visit(node, depth):
        if depth > max_depth[0]:
            max_depth[0] = depth

        if isinstance(node, FnDecl):
            current_fn[0] = node.name
            fn_complexity[node.name] = 1  # base path
            metrics.functions += 1

        # Each branch point adds 1 to complexity
        if isinstance(node, IfStmt):
            metrics.cyclomatic_complexity += 1
            if current_fn[0] and current_fn[0] in fn_complexity:
                fn_complexity[current_fn[0]] += 1
        elif isinstance(node, WhileStmt):
            metrics.cyclomatic_complexity += 1
            if current_fn[0] and current_fn[0] in fn_complexity:
                fn_complexity[current_fn[0]] += 1
        elif isinstance(node, BinOp) and node.op in ('and', 'or', '&&', '||'):
            metrics.cyclomatic_complexity += 1
            if current_fn[0] and current_fn[0] in fn_complexity:
                fn_complexity[current_fn[0]] += 1

        # Count statements and expressions
        if _is_stmt(node):
            metrics.statements += 1
        if _is_expr(node):
            metrics.expressions += 1
        if isinstance(node, LetDecl):
            metrics.variables += 1

    _walk_ast(ast, visit)
    metrics.max_nesting = max_depth[0]
    metrics.function_complexities = fn_complexity
    if fn_complexity:
        metrics.max_function_complexity = max(fn_complexity.values())

    # Average function length
    if metrics.functions > 0:
        fn_stmts = []
        for stmt in ast.stmts:
            if isinstance(stmt, FnDecl):
                fn_stmts.append(_count_stmts_in_body(stmt.body))
        if fn_stmts:
            metrics.avg_function_length = sum(fn_stmts) / len(fn_stmts)

    return metrics


# ============================================================
# Analysis Pass: Lint Rules
# ============================================================

def _analyze_lint(ast, type_checker=None):
    """Style and correctness lint rules."""
    diagnostics = []

    def visit(node, depth):
        # Rule: constant condition in if
        if isinstance(node, IfStmt):
            if isinstance(node.cond, BoolLit):
                if node.cond.value:
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.REDUNDANT,
                        message="Condition is always true",
                        line=_get_line(node),
                        suggestion="Remove the if and keep only the then-branch",
                    ))
                else:
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.DEAD_CODE,
                        message="Condition is always false; then-branch is dead code",
                        line=_get_line(node),
                        suggestion="Remove the if" + (
                            " and keep only the else-branch" if node.else_body else ""
                        ),
                    ))

        # Rule: constant false while
        if isinstance(node, WhileStmt):
            if isinstance(node.cond, BoolLit) and not node.cond.value:
                diagnostics.append(Diagnostic(
                    severity=Severity.WARNING,
                    category=Category.DEAD_CODE,
                    message="While condition is always false; loop body never executes",
                    line=_get_line(node),
                    suggestion="Remove the while loop",
                ))

        # Rule: division by zero
        if isinstance(node, BinOp) and node.op in ('/', '%'):
            if isinstance(node.right, IntLit) and node.right.value == 0:
                diagnostics.append(Diagnostic(
                    severity=Severity.ERROR,
                    category=Category.POSSIBLE_BUG,
                    message="Division by zero",
                    line=_get_line(node),
                ))
            elif isinstance(node.right, FloatLit) and node.right.value == 0.0:
                diagnostics.append(Diagnostic(
                    severity=Severity.WARNING,
                    category=Category.POSSIBLE_BUG,
                    message="Division by zero (float)",
                    line=_get_line(node),
                ))

        # Rule: self-assignment (x = x)
        if isinstance(node, Assign):
            if isinstance(node.value, Var) and node.value.name == node.name:
                diagnostics.append(Diagnostic(
                    severity=Severity.WARNING,
                    category=Category.REDUNDANT,
                    message=f"Self-assignment: '{node.name} = {node.name}'",
                    line=_get_line(node),
                    suggestion="Remove this statement",
                ))

        # Rule: comparison of identical operands
        if isinstance(node, BinOp) and node.op in ('==', '!=', '<', '>', '<=', '>='):
            if (isinstance(node.left, Var) and isinstance(node.right, Var)
                    and node.left.name == node.right.name):
                if node.op == '==':
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.REDUNDANT,
                        message=f"Comparison '{node.left.name} == {node.right.name}' is always true",
                        line=_get_line(node),
                        suggestion="Replace with 'true'",
                    ))
                elif node.op == '!=':
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.REDUNDANT,
                        message=f"Comparison '{node.left.name} != {node.right.name}' is always false",
                        line=_get_line(node),
                        suggestion="Replace with 'false'",
                    ))
                elif node.op in ('<', '>'):
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.REDUNDANT,
                        message=f"Comparison '{node.left.name} {node.op} {node.right.name}' is always false",
                        line=_get_line(node),
                        suggestion="Replace with 'false'",
                    ))

        # Rule: empty function body
        if isinstance(node, FnDecl):
            if isinstance(node.body, Block) and len(node.body.stmts) == 0:
                diagnostics.append(Diagnostic(
                    severity=Severity.INFO,
                    category=Category.STYLE,
                    message=f"Function '{node.name}' has an empty body",
                    line=_get_line(node),
                ))

        # Rule: unreachable code after return in block
        if isinstance(node, Block):
            for i, stmt in enumerate(node.stmts):
                if isinstance(stmt, ReturnStmt) and i < len(node.stmts) - 1:
                    next_stmt = node.stmts[i + 1]
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.UNREACHABLE,
                        message="Code after return statement is unreachable",
                        line=_get_line(next_stmt) if hasattr(next_stmt, 'line') else _get_line(stmt),
                    ))
                    break

        # Rule: redundant boolean comparison (x == true -> x)
        if isinstance(node, BinOp) and node.op == '==':
            if isinstance(node.right, BoolLit):
                if node.right.value is True:
                    diagnostics.append(Diagnostic(
                        severity=Severity.HINT,
                        category=Category.STYLE,
                        message="Redundant comparison with true",
                        line=_get_line(node),
                        suggestion="Use the expression directly instead of comparing to true",
                    ))
                elif node.right.value is False:
                    diagnostics.append(Diagnostic(
                        severity=Severity.HINT,
                        category=Category.STYLE,
                        message="Redundant comparison with false",
                        line=_get_line(node),
                        suggestion="Use logical negation instead of comparing to false",
                    ))

        # Rule: bare expression as statement with no side effects
        # In C010, bare expressions appear directly in stmts lists
        if isinstance(node, (Program, Block)):
            stmts = node.stmts
            for stmt in stmts:
                if isinstance(stmt, (IntLit, FloatLit, StringLit, BoolLit, Var)):
                    diagnostics.append(Diagnostic(
                        severity=Severity.WARNING,
                        category=Category.REDUNDANT,
                        message="Expression statement has no side effects",
                        line=_get_line(stmt),
                        suggestion="Remove this statement or assign the value",
                    ))

        # Rule: multiplication by 0
        if isinstance(node, BinOp) and node.op == '*':
            if (isinstance(node.left, IntLit) and node.left.value == 0) or \
               (isinstance(node.right, IntLit) and node.right.value == 0):
                diagnostics.append(Diagnostic(
                    severity=Severity.HINT,
                    category=Category.OPTIMIZATION,
                    message="Multiplication by zero -- result is always 0",
                    line=_get_line(node),
                    suggestion="Replace with 0",
                ))

        # Rule: addition/subtraction of 0
        if isinstance(node, BinOp) and node.op in ('+', '-'):
            if isinstance(node.right, IntLit) and node.right.value == 0:
                diagnostics.append(Diagnostic(
                    severity=Severity.HINT,
                    category=Category.OPTIMIZATION,
                    message=f"{'Addition of' if node.op == '+' else 'Subtraction of'} zero has no effect",
                    line=_get_line(node),
                    suggestion="Remove the operation",
                ))

        # Rule: multiplication by 1
        if isinstance(node, BinOp) and node.op == '*':
            if isinstance(node.right, IntLit) and node.right.value == 1:
                diagnostics.append(Diagnostic(
                    severity=Severity.HINT,
                    category=Category.OPTIMIZATION,
                    message="Multiplication by 1 has no effect",
                    line=_get_line(node),
                    suggestion="Remove the operation",
                ))
            elif isinstance(node.left, IntLit) and node.left.value == 1:
                diagnostics.append(Diagnostic(
                    severity=Severity.HINT,
                    category=Category.OPTIMIZATION,
                    message="Multiplication by 1 has no effect",
                    line=_get_line(node),
                    suggestion="Remove the operation",
                ))

        # Rule: double negation
        if isinstance(node, UnaryOp) and node.op == '-':
            if isinstance(node.operand, UnaryOp) and node.operand.op == '-':
                diagnostics.append(Diagnostic(
                    severity=Severity.HINT,
                    category=Category.STYLE,
                    message="Double negation cancels out",
                    line=_get_line(node),
                    suggestion="Remove both negations",
                ))

    _walk_ast(ast, visit)
    return diagnostics


# ============================================================
# Analysis Pass: Optimization Potential (C014)
# ============================================================

def _analyze_optimization_potential(source):
    """Use the optimizer to detect optimization opportunities."""
    diagnostics = []
    try:
        result = optimize_source(source)
        stats = result['stats']

        if stats.constant_folds > 0:
            diagnostics.append(Diagnostic(
                severity=Severity.HINT,
                category=Category.OPTIMIZATION,
                message=f"Found {stats.constant_folds} constant expression(s) that can be pre-computed",
                suggestion="Use literal values instead of computing constants at runtime",
            ))

        if stats.constant_propagations > 0:
            diagnostics.append(Diagnostic(
                severity=Severity.HINT,
                category=Category.OPTIMIZATION,
                message=f"Found {stats.constant_propagations} constant propagation(s) possible",
                suggestion="Variables with known constant values can be inlined",
            ))

        if stats.strength_reductions > 0:
            diagnostics.append(Diagnostic(
                severity=Severity.HINT,
                category=Category.OPTIMIZATION,
                message=f"Found {stats.strength_reductions} strength reduction(s) possible",
                suggestion="Some operations can be replaced with cheaper equivalents",
            ))

        if stats.dead_code_eliminations > 0:
            diagnostics.append(Diagnostic(
                severity=Severity.WARNING,
                category=Category.DEAD_CODE,
                message=f"Optimizer found {stats.dead_code_eliminations} dead code elimination(s)",
                suggestion="Remove unreachable or unused code",
            ))

        if stats.jump_optimizations > 0:
            diagnostics.append(Diagnostic(
                severity=Severity.HINT,
                category=Category.OPTIMIZATION,
                message=f"Found {stats.jump_optimizations} jump optimization(s) possible",
                suggestion="Control flow can be simplified",
            ))

        if stats.peephole_opts > 0:
            diagnostics.append(Diagnostic(
                severity=Severity.HINT,
                category=Category.OPTIMIZATION,
                message=f"Found {stats.peephole_opts} peephole optimization(s) possible",
                suggestion="Instruction sequences can be simplified",
            ))

        return diagnostics, stats
    except Exception:
        return diagnostics, None


def _analyze_dead_code_bytecode(source):
    """Use bytecode analysis to find significant dead code."""
    diagnostics = []
    try:
        result = optimize_source(source)
        original = result['original_chunk']
        optimized = result['optimized_chunk']

        orig_instrs = decode_chunk(original)
        opt_instrs = decode_chunk(optimized)

        orig_count = len(orig_instrs)
        opt_count = len(opt_instrs)

        if orig_count > 0 and opt_count < orig_count:
            removed = orig_count - opt_count
            pct = (removed / orig_count) * 100
            if pct > 20:
                diagnostics.append(Diagnostic(
                    severity=Severity.WARNING,
                    category=Category.DEAD_CODE,
                    message=f"{removed} instructions ({pct:.0f}%) eliminated by optimizer",
                    suggestion="Review code for unreachable paths or redundant computation",
                ))
    except Exception:
        pass
    return diagnostics


# ============================================================
# Complexity Diagnostics
# ============================================================

def _complexity_diagnostics(metrics, thresholds=None):
    """Generate diagnostics from complexity metrics."""
    if thresholds is None:
        thresholds = {
            "cyclomatic": 10,
            "function_complexity": 6,
            "nesting": 5,
            "function_length": 20,
        }

    diagnostics = []

    if metrics.cyclomatic_complexity > thresholds["cyclomatic"]:
        diagnostics.append(Diagnostic(
            severity=Severity.WARNING,
            category=Category.COMPLEXITY,
            message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity} (threshold: {thresholds['cyclomatic']})",
            suggestion="Consider breaking complex logic into smaller functions",
        ))

    for fn_name, complexity in metrics.function_complexities.items():
        if complexity > thresholds["function_complexity"]:
            diagnostics.append(Diagnostic(
                severity=Severity.WARNING,
                category=Category.COMPLEXITY,
                message=f"Function '{fn_name}' has high complexity: {complexity} (threshold: {thresholds['function_complexity']})",
                suggestion=f"Refactor '{fn_name}' into smaller functions",
            ))

    if metrics.max_nesting > thresholds["nesting"]:
        diagnostics.append(Diagnostic(
            severity=Severity.WARNING,
            category=Category.COMPLEXITY,
            message=f"Deep nesting detected: {metrics.max_nesting} levels (threshold: {thresholds['nesting']})",
            suggestion="Reduce nesting with early returns or extracted functions",
        ))

    return diagnostics


# ============================================================
# Source Metrics
# ============================================================

def _compute_line_metrics(source):
    """Count lines of code, blanks, etc."""
    lines = source.split('\n')
    total = len(lines)
    blank = sum(1 for l in lines if l.strip() == '')
    code = total - blank
    return total, code, blank


# ============================================================
# Analysis Report
# ============================================================

@dataclass
class AnalysisReport:
    """Complete analysis results."""
    source: str = ""
    diagnostics: list = field(default_factory=list)
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    type_errors: list = field(default_factory=list)
    optimization_stats: Optional[OptimizationStats] = None
    parse_error: Optional[str] = None

    @property
    def error_count(self):
        return sum(1 for d in self.diagnostics if d.severity == Severity.ERROR)

    @property
    def warning_count(self):
        return sum(1 for d in self.diagnostics if d.severity == Severity.WARNING)

    @property
    def info_count(self):
        return sum(1 for d in self.diagnostics if d.severity == Severity.INFO)

    @property
    def hint_count(self):
        return sum(1 for d in self.diagnostics if d.severity == Severity.HINT)

    @property
    def total_issues(self):
        return len(self.diagnostics)

    @property
    def has_errors(self):
        return self.error_count > 0 or self.parse_error is not None

    def by_severity(self, severity):
        return [d for d in self.diagnostics if d.severity == severity]

    def by_category(self, category):
        return [d for d in self.diagnostics if d.category == category]

    def summary(self):
        lines = []
        lines.append("=" * 60)
        lines.append("  STATIC ANALYSIS REPORT")
        lines.append("=" * 60)

        if self.parse_error:
            lines.append(f"  PARSE ERROR: {self.parse_error}")
            lines.append("=" * 60)
            return "\n".join(lines)

        # Metrics
        lines.append(f"  Lines: {self.metrics.total_lines} total, {self.metrics.code_lines} code, {self.metrics.blank_lines} blank")
        lines.append(f"  Functions: {self.metrics.functions}, Variables: {self.metrics.variables}")
        lines.append(f"  Statements: {self.metrics.statements}, Expressions: {self.metrics.expressions}")
        lines.append(f"  Cyclomatic complexity: {self.metrics.cyclomatic_complexity}")
        lines.append(f"  Max nesting depth: {self.metrics.max_nesting}")
        if self.metrics.function_complexities:
            lines.append(f"  Function complexities: {self.metrics.function_complexities}")
        lines.append("")

        # Optimization stats
        if self.optimization_stats:
            stats = self.optimization_stats
            lines.append(f"  Optimization: {stats.total_optimizations} opportunities in {stats.rounds} rounds")
            if stats.size_reduction > 0:
                lines.append(f"  Size reduction: {stats.size_reduction * 100:.1f}%")
            lines.append("")

        # Diagnostics by severity
        for sev in [Severity.ERROR, Severity.WARNING, Severity.INFO, Severity.HINT]:
            items = self.by_severity(sev)
            if items:
                lines.append(f"  {sev.value.upper()}S ({len(items)}):")
                for d in items:
                    loc = f"line {d.line}" if d.line > 0 else "global"
                    lines.append(f"    [{d.category.value}] {d.message} ({loc})")
                    if d.suggestion:
                        lines.append(f"      -> {d.suggestion}")
                lines.append("")

        # Summary line
        lines.append(f"  Total: {self.error_count} errors, {self.warning_count} warnings, "
                      f"{self.info_count} info, {self.hint_count} hints")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# Main Entry Points
# ============================================================

def analyze(source, thresholds=None):
    """
    Run full static analysis on source code.

    Args:
        source: Source code string in the C010 VM language
        thresholds: Optional dict of complexity thresholds

    Returns:
        AnalysisReport with all diagnostics, metrics, and suggestions
    """
    report = AnalysisReport(source=source)

    # Line metrics
    total, code, blank = _compute_line_metrics(source)
    report.metrics.total_lines = total
    report.metrics.code_lines = code
    report.metrics.blank_lines = blank

    # Parse
    try:
        ast = parse(source)
    except Exception as e:
        report.parse_error = str(e)
        return report

    # Type checking (C013)
    try:
        type_errors, checker = check_source(source)
        report.type_errors = type_errors
        for te in type_errors:
            report.diagnostics.append(Diagnostic(
                severity=Severity.ERROR,
                category=Category.TYPE_ERROR,
                message=te.message,
                line=te.line,
            ))
    except Exception as e:
        report.diagnostics.append(Diagnostic(
            severity=Severity.ERROR,
            category=Category.TYPE_ERROR,
            message=f"Type checker failed: {e}",
        ))
        checker = None

    # Complexity analysis (AST-based)
    metrics = _analyze_complexity(ast)
    metrics.total_lines = total
    metrics.code_lines = code
    metrics.blank_lines = blank
    report.metrics = metrics

    # Complexity diagnostics
    report.diagnostics.extend(_complexity_diagnostics(metrics, thresholds))

    # Variable analysis (unused, shadowed)
    report.diagnostics.extend(_analyze_variables(ast))

    # Lint rules
    report.diagnostics.extend(_analyze_lint(ast, checker))

    # Optimization analysis (C014)
    opt_diags, opt_stats = _analyze_optimization_potential(source)
    report.diagnostics.extend(opt_diags)
    report.optimization_stats = opt_stats

    # Bytecode dead code analysis
    report.diagnostics.extend(_analyze_dead_code_bytecode(source))

    # Sort diagnostics: errors first, then by line
    report.diagnostics.sort(key=lambda d: (
        {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2, Severity.HINT: 3}[d.severity],
        d.line,
    ))

    return report


def analyze_quick(source):
    """Quick analysis -- type checking + lint only, no optimization analysis."""
    report = AnalysisReport(source=source)

    total, code, blank = _compute_line_metrics(source)
    report.metrics.total_lines = total
    report.metrics.code_lines = code
    report.metrics.blank_lines = blank

    try:
        ast = parse(source)
    except Exception as e:
        report.parse_error = str(e)
        return report

    try:
        type_errors, checker = check_source(source)
        report.type_errors = type_errors
        for te in type_errors:
            report.diagnostics.append(Diagnostic(
                severity=Severity.ERROR,
                category=Category.TYPE_ERROR,
                message=te.message,
                line=te.line,
            ))
    except Exception:
        pass

    metrics = _analyze_complexity(ast)
    metrics.total_lines = total
    metrics.code_lines = code
    metrics.blank_lines = blank
    report.metrics = metrics

    report.diagnostics.extend(_analyze_lint(ast, None))

    report.diagnostics.sort(key=lambda d: (
        {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2, Severity.HINT: 3}[d.severity],
        d.line,
    ))

    return report


def format_report(report):
    """Format an AnalysisReport as a human-readable string."""
    return report.summary()
