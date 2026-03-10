"""
V034: Deep Python Taint Analysis

Path-sensitive, inter-procedural taint tracking for Python source code.
Extends V033's basic taint analysis with:

1. Path sensitivity: per-branch taint state (if sanitized on one path, still
   tainted on the other)
2. Inter-procedural: taint flows through function calls/returns with
   context sensitivity (k-CFA style call-string approach)
3. Sanitizer support: functions that clean taint (e.g., escape, validate)
4. Taint labels: track WHICH source tainted a value (not just yes/no)
5. Implicit flows: taint from conditions (if tainted: x = safe_val)

Uses Python's ast module to analyze real Python code.
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from typing import (
    List, Dict, Set, Optional, Tuple, FrozenSet, Any, Union
)
from enum import Enum
from collections import defaultdict, deque
from copy import deepcopy


# ============================================================
# Taint Labels and Lattice
# ============================================================

@dataclass(frozen=True)
class TaintLabel:
    """A taint source identifier."""
    source: str       # e.g., "user_input", "file_read"
    origin_line: int   # where the taint originated
    origin_var: str    # which variable introduced it

    def __str__(self):
        return f"{self.source}@{self.origin_var}:{self.origin_line}"


@dataclass(frozen=True)
class TaintValue:
    """Taint state for a single variable: a set of labels (or clean)."""
    labels: FrozenSet[TaintLabel] = frozenset()

    @property
    def is_tainted(self) -> bool:
        return len(self.labels) > 0

    def join(self, other: 'TaintValue') -> 'TaintValue':
        """Join = union of labels (may-taint)."""
        return TaintValue(self.labels | other.labels)

    def __str__(self):
        if not self.labels:
            return "clean"
        return "{" + ", ".join(str(l) for l in sorted(self.labels, key=str)) + "}"


CLEAN = TaintValue()


class TaintEnv:
    """Mapping from variable names to taint values. Immutable-style."""

    def __init__(self, bindings: Optional[Dict[str, TaintValue]] = None):
        self._bindings: Dict[str, TaintValue] = dict(bindings) if bindings else {}

    def get(self, name: str) -> TaintValue:
        return self._bindings.get(name, CLEAN)

    def set(self, name: str, value: TaintValue) -> 'TaintEnv':
        new_bindings = dict(self._bindings)
        new_bindings[name] = value
        return TaintEnv(new_bindings)

    def join(self, other: 'TaintEnv') -> 'TaintEnv':
        """Join two environments (union of taint at each variable)."""
        all_vars = set(self._bindings) | set(other._bindings)
        result = {}
        for v in all_vars:
            result[v] = self.get(v).join(other.get(v))
        return TaintEnv(result)

    def copy(self) -> 'TaintEnv':
        return TaintEnv(dict(self._bindings))

    def tainted_vars(self) -> Set[str]:
        return {k for k, v in self._bindings.items() if v.is_tainted}

    def __eq__(self, other):
        if not isinstance(other, TaintEnv):
            return False
        all_vars = set(self._bindings) | set(other._bindings)
        return all(self.get(v) == other.get(v) for v in all_vars)

    def __repr__(self):
        tainted = {k: str(v) for k, v in self._bindings.items() if v.is_tainted}
        return f"TaintEnv({tainted})"


# ============================================================
# Findings
# ============================================================

class TaintSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TaintFindingKind(Enum):
    TAINT_PROPAGATION = "taint_propagation"
    TAINT_SINK = "taint_sink"
    IMPLICIT_FLOW = "implicit_flow"
    TAINT_THROUGH_CALL = "taint_through_call"
    TAINT_FROM_RETURN = "taint_from_return"
    UNSANITIZED_USE = "unsanitized_use"
    PARTIAL_SANITIZATION = "partial_sanitization"


@dataclass
class TaintFinding:
    kind: TaintFindingKind
    severity: TaintSeverity
    file: str
    line: int
    col: int
    message: str
    labels: FrozenSet[TaintLabel] = frozenset()
    path_condition: str = ""  # human-readable path condition
    function: str = ""
    call_chain: Tuple[str, ...] = ()  # inter-procedural call context

    def __str__(self):
        loc = f"{os.path.basename(self.file)}:{self.line}"
        if self.function:
            loc += f" in {self.function}"
        ctx = ""
        if self.call_chain:
            ctx = f" [via {' -> '.join(self.call_chain)}]"
        path = ""
        if self.path_condition:
            path = f" (when {self.path_condition})"
        return f"[{self.severity.value.upper()}] {loc}: {self.message}{ctx}{path}"


# ============================================================
# Configuration
# ============================================================

@dataclass
class TaintConfig:
    """Configure taint sources, sinks, and sanitizers."""
    # Source patterns: function names or parameter names that introduce taint
    source_functions: Dict[str, str] = field(default_factory=lambda: {
        'input': 'user_input',
        'raw_input': 'user_input',
        'read': 'file_read',
        'readline': 'file_read',
        'readlines': 'file_read',
        'recv': 'network',
        'recvfrom': 'network',
        'urlopen': 'network',
        'get': 'http_input',   # requests.get
    })
    source_params: Set[str] = field(default_factory=lambda: {
        'user_input', 'raw_data', 'untrusted', 'request', 'query',
        'form_data', 'payload', 'body',
    })
    # Additional explicit sources (variable names)
    source_vars: Set[str] = field(default_factory=set)

    # Sink patterns: functions where tainted data is dangerous
    sinks: Dict[str, TaintSeverity] = field(default_factory=lambda: {
        'exec': TaintSeverity.CRITICAL,
        'eval': TaintSeverity.CRITICAL,
        'compile': TaintSeverity.ERROR,
        'system': TaintSeverity.CRITICAL,
        'popen': TaintSeverity.CRITICAL,
        'run': TaintSeverity.ERROR,
        'call': TaintSeverity.ERROR,
        'check_output': TaintSeverity.ERROR,
        'execute': TaintSeverity.ERROR,     # SQL
        'executemany': TaintSeverity.ERROR,
        'executescript': TaintSeverity.ERROR,
        'write': TaintSeverity.WARNING,
        'send': TaintSeverity.WARNING,
        'sendall': TaintSeverity.WARNING,
    })

    # Sanitizer functions: remove or reduce taint
    sanitizers: Dict[str, Set[str]] = field(default_factory=lambda: {
        # sanitizer_name -> set of taint sources it sanitizes ('*' = all)
        'escape': {'*'},
        'html_escape': {'*'},
        'quote': {'*'},
        'sanitize': {'*'},
        'clean': {'*'},
        'validate': {'*'},
        'strip_tags': {'user_input'},
        'parameterize': {'user_input'},  # SQL parameterization
        'int': {'*'},       # type coercion sanitizes string injection
        'float': {'*'},
        'bool': {'*'},
    })

    # Max call depth for inter-procedural analysis
    max_call_depth: int = 5

    # Track implicit flows (taint from conditions)
    track_implicit: bool = True

    # Context sensitivity depth (k in k-CFA)
    context_depth: int = 2


# ============================================================
# Function Summaries (for inter-procedural analysis)
# ============================================================

@dataclass
class FunctionSummary:
    """Summary of a function's taint behavior."""
    name: str
    params: List[str]
    # Which params taint the return value? (param_index -> True)
    param_taints_return: Set[int] = field(default_factory=set)
    # Which params taint which other params (via mutation)?
    param_taints_param: Dict[int, Set[int]] = field(default_factory=dict)
    # Does the function introduce new taint? (from sources)
    introduces_taint: bool = False
    # Taint labels introduced
    introduced_labels: FrozenSet[TaintLabel] = frozenset()
    # Is this a sanitizer?
    is_sanitizer: bool = False
    sanitizes: Set[str] = field(default_factory=set)  # which taint sources
    # Internal findings (for reporting)
    internal_findings: List[TaintFinding] = field(default_factory=list)


# ============================================================
# Path-Sensitive Intra-Procedural Analysis
# ============================================================

class PathSensitiveTaintAnalyzer(ast.NodeVisitor):
    """Analyze a function body with path-sensitive taint tracking.

    At each branch (if/else, try/except), we fork the taint environment
    and analyze each path independently. At the join point, we merge
    with union semantics (may-taint).
    """

    def __init__(self, config: TaintConfig, filename: str,
                 func_name: str, initial_env: TaintEnv,
                 func_summaries: Dict[str, FunctionSummary],
                 call_chain: Tuple[str, ...] = (),
                 implicit_taint: Optional[TaintValue] = None):
        self.config = config
        self.filename = filename
        self.func_name = func_name
        self.env = initial_env
        self.func_summaries = func_summaries
        self.call_chain = call_chain
        self.implicit_taint = implicit_taint or CLEAN
        self.findings: List[TaintFinding] = []
        self.return_taint = CLEAN
        self.path_conditions: List[str] = []

    def _path_str(self) -> str:
        if not self.path_conditions:
            return ""
        return " and ".join(self.path_conditions)

    def _expr_taint(self, node: ast.expr) -> TaintValue:
        """Compute the taint of an expression."""
        if isinstance(node, ast.Name):
            result = self.env.get(node.id)
            # source_vars are always tainted regardless of env
            if node.id in self.config.source_vars:
                label = TaintLabel(
                    source='explicit_source',
                    origin_line=getattr(node, 'lineno', 0),
                    origin_var=node.id,
                )
                result = result.join(TaintValue(frozenset([label])))
            return result

        elif isinstance(node, ast.Call):
            return self._call_taint(node)

        elif isinstance(node, ast.Constant):
            return CLEAN

        elif isinstance(node, ast.BinOp):
            left = self._expr_taint(node.left)
            right = self._expr_taint(node.right)
            return left.join(right)

        elif isinstance(node, ast.UnaryOp):
            return self._expr_taint(node.operand)

        elif isinstance(node, ast.BoolOp):
            result = CLEAN
            for val in node.values:
                result = result.join(self._expr_taint(val))
            return result

        elif isinstance(node, ast.Compare):
            result = self._expr_taint(node.left)
            for comp in node.comparators:
                result = result.join(self._expr_taint(comp))
            return result

        elif isinstance(node, ast.IfExp):
            # Path-sensitive: condition taint + union of branches
            cond_taint = self._expr_taint(node.test)
            then_taint = self._expr_taint(node.body)
            else_taint = self._expr_taint(node.orelse)
            result = then_taint.join(else_taint)
            if self.config.track_implicit and cond_taint.is_tainted:
                result = result.join(cond_taint)
            return result

        elif isinstance(node, ast.Subscript):
            val_taint = self._expr_taint(node.value)
            if isinstance(node.slice, ast.Index):
                # Python 3.8 compat
                idx_taint = self._expr_taint(node.slice.value)
            else:
                idx_taint = self._expr_taint(node.slice)
            return val_taint.join(idx_taint)

        elif isinstance(node, ast.Attribute):
            return self._expr_taint(node.value)

        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            result = CLEAN
            for elt in node.elts:
                result = result.join(self._expr_taint(elt))
            return result

        elif isinstance(node, ast.Dict):
            result = CLEAN
            for k in node.keys:
                if k is not None:
                    result = result.join(self._expr_taint(k))
            for v in node.values:
                result = result.join(self._expr_taint(v))
            return result

        elif isinstance(node, ast.JoinedStr):
            # f-string
            result = CLEAN
            for val in node.values:
                if isinstance(val, ast.FormattedValue):
                    result = result.join(self._expr_taint(val.value))
            return result

        elif isinstance(node, ast.FormattedValue):
            return self._expr_taint(node.value)

        elif isinstance(node, ast.Starred):
            return self._expr_taint(node.value)

        elif isinstance(node, ast.NamedExpr):
            # Walrus operator: (x := expr)
            val_taint = self._expr_taint(node.value)
            if isinstance(node.target, ast.Name):
                self.env = self.env.set(node.target.id, val_taint)
                if val_taint.is_tainted:
                    self.findings.append(TaintFinding(
                        kind=TaintFindingKind.TAINT_PROPAGATION,
                        severity=TaintSeverity.INFO,
                        file=self.filename,
                        line=getattr(node, 'lineno', 0),
                        col=getattr(node, 'col_offset', 0),
                        message=f"Taint propagates to '{node.target.id}' via walrus operator",
                        labels=val_taint.labels,
                        function=self.func_name,
                        path_condition=self._path_str(),
                        call_chain=self.call_chain,
                    ))
            return val_taint

        elif isinstance(node, ast.Lambda):
            return CLEAN  # lambda itself isn't tainted

        # Default: walk children
        result = CLEAN
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                result = result.join(self._expr_taint(child))
        return result

    def _call_taint(self, node: ast.Call) -> TaintValue:
        """Compute taint from a function call."""
        func_name = self._resolve_func_name(node)

        # Check if it's a source function
        if func_name in self.config.source_functions:
            source_label = self.config.source_functions[func_name]
            label = TaintLabel(
                source=source_label,
                origin_line=node.lineno,
                origin_var=func_name,
            )
            return TaintValue(frozenset([label]))

        # Check if it's a sanitizer
        if func_name in self.config.sanitizers:
            sanitized_sources = self.config.sanitizers[func_name]
            if '*' in sanitized_sources:
                return CLEAN
            # Partial sanitization: remove specific labels
            if node.args:
                arg_taint = self._expr_taint(node.args[0])
                remaining = frozenset(
                    l for l in arg_taint.labels
                    if l.source not in sanitized_sources
                )
                if remaining and arg_taint.is_tainted:
                    self.findings.append(TaintFinding(
                        kind=TaintFindingKind.PARTIAL_SANITIZATION,
                        severity=TaintSeverity.WARNING,
                        file=self.filename,
                        line=node.lineno,
                        col=getattr(node, 'col_offset', 0),
                        message=f"Sanitizer '{func_name}' does not clean all taint sources",
                        labels=remaining,
                        function=self.func_name,
                        path_condition=self._path_str(),
                        call_chain=self.call_chain,
                    ))
                return TaintValue(remaining)
            return CLEAN

        # Check for a function summary (inter-procedural)
        if func_name in self.func_summaries:
            summary = self.func_summaries[func_name]
            result = CLEAN
            if summary.introduces_taint:
                result = result.join(TaintValue(summary.introduced_labels))
            # Apply param-to-return taint transfer
            for i in summary.param_taints_return:
                if i < len(node.args):
                    result = result.join(self._expr_taint(node.args[i]))
            return result

        # Check if it's a sink
        if func_name in self.config.sinks:
            for i, arg in enumerate(node.args):
                arg_taint = self._expr_taint(arg)
                if arg_taint.is_tainted:
                    self.findings.append(TaintFinding(
                        kind=TaintFindingKind.TAINT_SINK,
                        severity=self.config.sinks[func_name],
                        file=self.filename,
                        line=node.lineno,
                        col=getattr(node, 'col_offset', 0),
                        message=f"Tainted data flows into sink '{func_name}' (arg {i})",
                        labels=arg_taint.labels,
                        function=self.func_name,
                        path_condition=self._path_str(),
                        call_chain=self.call_chain,
                    ))

        # Default: taint propagates through args AND object (for method calls)
        result = CLEAN
        # For method calls (obj.method()), the object's taint propagates
        if isinstance(node.func, ast.Attribute):
            result = result.join(self._expr_taint(node.func.value))
        for arg in node.args:
            result = result.join(self._expr_taint(arg))
        for kw in node.keywords:
            result = result.join(self._expr_taint(kw.value))
        return result

    def _resolve_func_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _check_sink(self, name: str, value_taint: TaintValue, line: int):
        """Check if assigning to a variable constitutes a sink."""
        pass  # Sinks are checked at call sites

    def analyze_stmts(self, stmts: List[ast.stmt]) -> TaintEnv:
        """Analyze a sequence of statements, returning the final environment."""
        for stmt in stmts:
            self._analyze_stmt(stmt)
        return self.env

    def _analyze_stmt(self, node: ast.stmt):
        """Analyze a single statement."""
        if isinstance(node, ast.Assign):
            self._analyze_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._analyze_aug_assign(node)
        elif isinstance(node, ast.AnnAssign):
            self._analyze_ann_assign(node)
        elif isinstance(node, ast.If):
            self._analyze_if(node)
        elif isinstance(node, ast.While):
            self._analyze_while(node)
        elif isinstance(node, ast.For):
            self._analyze_for(node)
        elif isinstance(node, ast.With):
            self._analyze_with(node)
        elif isinstance(node, ast.Try):
            self._analyze_try(node)
        elif isinstance(node, ast.Return):
            self._analyze_return(node)
        elif isinstance(node, ast.Expr):
            # Expression statement (e.g., function call)
            if isinstance(node.value, ast.Call):
                self._call_taint(node.value)
            else:
                self._expr_taint(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pass  # Analyzed separately
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env = self.env.set(target.id, CLEAN)
        elif isinstance(node, ast.Global):
            pass
        elif isinstance(node, ast.Nonlocal):
            pass
        elif isinstance(node, ast.Assert):
            self._expr_taint(node.test)
        elif isinstance(node, ast.Raise):
            if node.exc:
                self._expr_taint(node.exc)
        elif isinstance(node, ast.Pass):
            pass
        elif isinstance(node, ast.Break):
            pass
        elif isinstance(node, ast.Continue):
            pass
        elif isinstance(node, ast.Import):
            pass
        elif isinstance(node, ast.ImportFrom):
            pass

    def _analyze_assign(self, node: ast.Assign):
        val_taint = self._expr_taint(node.value)

        # Add implicit taint from enclosing conditions
        if self.config.track_implicit and self.implicit_taint.is_tainted:
            combined = val_taint.join(self.implicit_taint)
            if not val_taint.is_tainted and combined.is_tainted:
                for target in node.targets:
                    tgt_name = self._target_name(target)
                    if tgt_name:
                        self.findings.append(TaintFinding(
                            kind=TaintFindingKind.IMPLICIT_FLOW,
                            severity=TaintSeverity.WARNING,
                            file=self.filename,
                            line=node.lineno,
                            col=getattr(node, 'col_offset', 0),
                            message=f"Implicit taint flow to '{tgt_name}' via condition",
                            labels=self.implicit_taint.labels,
                            function=self.func_name,
                            path_condition=self._path_str(),
                            call_chain=self.call_chain,
                        ))
            val_taint = combined

        for target in node.targets:
            self._assign_target(target, val_taint, node.lineno)

    def _analyze_aug_assign(self, node: ast.AugAssign):
        val_taint = self._expr_taint(node.value)
        if isinstance(node.target, ast.Name):
            existing = self.env.get(node.target.id)
            combined = existing.join(val_taint)
            if self.config.track_implicit:
                combined = combined.join(self.implicit_taint)
            self.env = self.env.set(node.target.id, combined)
            if combined.is_tainted and not existing.is_tainted:
                self.findings.append(TaintFinding(
                    kind=TaintFindingKind.TAINT_PROPAGATION,
                    severity=TaintSeverity.INFO,
                    file=self.filename,
                    line=node.lineno,
                    col=getattr(node, 'col_offset', 0),
                    message=f"Taint propagates to '{node.target.id}' via augmented assignment",
                    labels=combined.labels,
                    function=self.func_name,
                    path_condition=self._path_str(),
                    call_chain=self.call_chain,
                ))

    def _analyze_ann_assign(self, node: ast.AnnAssign):
        if node.value is not None:
            val_taint = self._expr_taint(node.value)
            if self.config.track_implicit:
                val_taint = val_taint.join(self.implicit_taint)
            if isinstance(node.target, ast.Name):
                self.env = self.env.set(node.target.id, val_taint)

    def _assign_target(self, target: ast.expr, taint: TaintValue, line: int):
        if isinstance(target, ast.Name):
            old_taint = self.env.get(target.id)
            self.env = self.env.set(target.id, taint)
            if taint.is_tainted and not old_taint.is_tainted:
                self.findings.append(TaintFinding(
                    kind=TaintFindingKind.TAINT_PROPAGATION,
                    severity=TaintSeverity.INFO,
                    file=self.filename,
                    line=line,
                    col=getattr(target, 'col_offset', 0),
                    message=f"Taint propagates to '{target.id}'",
                    labels=taint.labels,
                    function=self.func_name,
                    path_condition=self._path_str(),
                    call_chain=self.call_chain,
                ))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign_target(elt, taint, line)
        elif isinstance(target, ast.Starred):
            self._assign_target(target.value, taint, line)

    def _target_name(self, target) -> Optional[str]:
        if isinstance(target, ast.Name):
            return target.id
        return None

    def _analyze_if(self, node: ast.If):
        """Path-sensitive: fork env for each branch, join at exit."""
        cond_taint = self._expr_taint(node.test)
        cond_str = self._condition_str(node.test)

        # Save state
        pre_env = self.env.copy()
        pre_implicit = self.implicit_taint

        # True branch
        if self.config.track_implicit and cond_taint.is_tainted:
            self.implicit_taint = pre_implicit.join(cond_taint)
        self.path_conditions.append(cond_str)
        self.analyze_stmts(node.body)
        then_env = self.env
        self.path_conditions.pop()

        # False branch
        self.env = pre_env.copy()
        self.implicit_taint = pre_implicit
        if self.config.track_implicit and cond_taint.is_tainted:
            self.implicit_taint = pre_implicit.join(cond_taint)
        self.path_conditions.append(f"not ({cond_str})")
        if node.orelse:
            self.analyze_stmts(node.orelse)
        else_env = self.env
        self.path_conditions.pop()

        # Join
        self.env = then_env.join(else_env)
        self.implicit_taint = pre_implicit

    def _analyze_while(self, node: ast.While):
        """Iterate to fixpoint for while loops."""
        cond_taint = self._expr_taint(node.test)

        # Iterate until env stabilizes (max 10 iterations)
        for _ in range(10):
            old_env = self.env.copy()
            pre_implicit = self.implicit_taint
            if self.config.track_implicit and cond_taint.is_tainted:
                self.implicit_taint = pre_implicit.join(cond_taint)
            self.analyze_stmts(node.body)
            self.implicit_taint = pre_implicit
            new_env = old_env.join(self.env)
            if new_env == self.env:
                break
            self.env = new_env

        if node.orelse:
            self.analyze_stmts(node.orelse)

    def _analyze_for(self, node: ast.For):
        """For loops: iterator taint flows to loop variable."""
        iter_taint = self._expr_taint(node.iter)
        self._assign_target(node.target, iter_taint, node.lineno)

        # Iterate to fixpoint
        for _ in range(10):
            old_env = self.env.copy()
            self.analyze_stmts(node.body)
            new_env = old_env.join(self.env)
            if new_env == self.env:
                break
            self.env = new_env

        if node.orelse:
            self.analyze_stmts(node.orelse)

    def _analyze_with(self, node: ast.With):
        for item in node.items:
            ctx_taint = self._expr_taint(item.context_expr)
            if item.optional_vars:
                self._assign_target(item.optional_vars, ctx_taint, node.lineno)
        self.analyze_stmts(node.body)

    def _analyze_try(self, node: ast.Try):
        """Path-sensitive: each handler is a separate path."""
        # Try body
        pre_env = self.env.copy()
        self.analyze_stmts(node.body)
        try_env = self.env

        # Each handler gets the pre-try env (exception may happen at any point)
        handler_envs = []
        for handler in node.handlers:
            self.env = pre_env.copy()
            if handler.name:
                # Exception variable is tainted if it comes from external
                self.env = self.env.set(handler.name, CLEAN)
            self.analyze_stmts(handler.body)
            handler_envs.append(self.env)

        # Join try-env with all handler envs
        result_env = try_env
        for henv in handler_envs:
            result_env = result_env.join(henv)

        # Else clause (only runs if no exception)
        if node.orelse:
            self.env = try_env
            self.analyze_stmts(node.orelse)
            result_env = result_env.join(self.env)

        self.env = result_env

        # Finally always runs
        if node.finalbody:
            self.analyze_stmts(node.finalbody)

    def _analyze_return(self, node: ast.Return):
        if node.value:
            ret_taint = self._expr_taint(node.value)
            self.return_taint = self.return_taint.join(ret_taint)

    def _condition_str(self, node: ast.expr) -> str:
        """Human-readable condition string."""
        try:
            return ast.unparse(node)
        except Exception:
            return "<?>"


# ============================================================
# Inter-Procedural Analysis
# ============================================================

class InterProceduralAnalyzer:
    """Analyze a module with inter-procedural taint tracking.

    Uses a worklist algorithm:
    1. Build call graph
    2. Compute function summaries bottom-up
    3. Re-analyze callers when callee summaries change
    """

    def __init__(self, config: Optional[TaintConfig] = None):
        self.config = config or TaintConfig()
        self.func_defs: Dict[str, ast.FunctionDef] = {}
        self.func_summaries: Dict[str, FunctionSummary] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.findings: List[TaintFinding] = []
        self.filename = ""

    def analyze_source(self, source: str, filename: str = "<string>",
                       entry_points: Optional[List[str]] = None,
                       taint_args: Optional[Dict[str, Dict[str, str]]] = None
                       ) -> 'TaintAnalysisResult':
        """Analyze Python source code.

        Args:
            source: Python source code
            filename: Filename for error reporting
            entry_points: Functions to analyze (default: all)
            taint_args: Per-function tainted arguments
                        {func_name: {param_name: taint_source}}
        """
        self.filename = filename
        self.findings = []

        tree = ast.parse(source, filename=filename)
        return self._analyze_tree(tree, filename, entry_points, taint_args)

    def analyze_file(self, filepath: str,
                     entry_points: Optional[List[str]] = None,
                     taint_args: Optional[Dict[str, Dict[str, str]]] = None
                     ) -> 'TaintAnalysisResult':
        """Analyze a Python file."""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        return self.analyze_source(source, filepath, entry_points, taint_args)

    def _analyze_tree(self, tree: ast.Module, filename: str,
                      entry_points: Optional[List[str]],
                      taint_args: Optional[Dict[str, Dict[str, str]]]
                      ) -> 'TaintAnalysisResult':
        # Phase 1: Collect function definitions
        self.func_defs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.func_defs[node.name] = node

        # Phase 2: Build call graph
        self.call_graph = defaultdict(set)
        self.reverse_call_graph = defaultdict(set)
        for fname, fnode in self.func_defs.items():
            for child in ast.walk(fnode):
                if isinstance(child, ast.Call):
                    callee = self._resolve_call_name(child)
                    if callee and callee in self.func_defs:
                        self.call_graph[fname].add(callee)
                        self.reverse_call_graph[callee].add(fname)

        # Phase 3: Topological order (bottom-up analysis)
        order = self._topo_sort()

        # Phase 4: Compute initial summaries
        self.func_summaries = {}
        for fname in order:
            self._compute_summary(fname)

        # Phase 5: Worklist refinement (re-analyze when summaries change)
        worklist = deque(order)
        iterations = 0
        max_iterations = len(order) * 3  # prevent infinite loops
        while worklist and iterations < max_iterations:
            fname = worklist.popleft()
            old_summary = self.func_summaries.get(fname)
            self._compute_summary(fname)
            new_summary = self.func_summaries.get(fname)
            if self._summary_changed(old_summary, new_summary):
                for caller in self.reverse_call_graph.get(fname, set()):
                    if caller not in worklist:
                        worklist.append(caller)
            iterations += 1

        # Phase 6: Final analysis of entry points with full summaries
        self.findings = []
        targets = entry_points or list(self.func_defs.keys())
        for fname in targets:
            if fname not in self.func_defs:
                continue
            fnode = self.func_defs[fname]

            # Build initial taint env for this function
            env = TaintEnv()
            params = [arg.arg for arg in fnode.args.args]

            # Apply configured taint sources
            func_taint_args = (taint_args or {}).get(fname, {})
            for i, param in enumerate(params):
                if param in self.config.source_params or param in func_taint_args:
                    source = func_taint_args.get(param, 'parameter')
                    label = TaintLabel(
                        source=source,
                        origin_line=fnode.lineno,
                        origin_var=param,
                    )
                    env = env.set(param, TaintValue(frozenset([label])))
                elif param in self.config.source_vars:
                    label = TaintLabel(
                        source='explicit_source',
                        origin_line=fnode.lineno,
                        origin_var=param,
                    )
                    env = env.set(param, TaintValue(frozenset([label])))

            analyzer = PathSensitiveTaintAnalyzer(
                config=self.config,
                filename=filename,
                func_name=fname,
                initial_env=env,
                func_summaries=self.func_summaries,
                call_chain=(),
            )
            analyzer.analyze_stmts(fnode.body)
            self.findings.extend(analyzer.findings)

        # Phase 7: Also analyze module-level code
        module_stmts = [s for s in tree.body
                        if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                                              ast.ClassDef))]
        if module_stmts:
            env = TaintEnv()
            for var in self.config.source_vars:
                label = TaintLabel(source='explicit_source', origin_line=0, origin_var=var)
                env = env.set(var, TaintValue(frozenset([label])))
            analyzer = PathSensitiveTaintAnalyzer(
                config=self.config,
                filename=filename,
                func_name="<module>",
                initial_env=env,
                func_summaries=self.func_summaries,
                call_chain=(),
            )
            analyzer.analyze_stmts(module_stmts)
            self.findings.extend(analyzer.findings)

        return TaintAnalysisResult(
            findings=list(self.findings),
            func_summaries=dict(self.func_summaries),
            call_graph={k: set(v) for k, v in self.call_graph.items()},
            filename=filename,
        )

    def _resolve_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _topo_sort(self) -> List[str]:
        """Topological sort of call graph (callees before callers).
        Handles cycles by breaking them."""
        visited = set()
        in_stack = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            if node in in_stack:
                return  # cycle, break it
            in_stack.add(node)
            for callee in self.call_graph.get(node, set()):
                if callee in self.func_defs:
                    dfs(callee)
            in_stack.discard(node)
            visited.add(node)
            order.append(node)

        for fname in self.func_defs:
            dfs(fname)
        return order

    def _compute_summary(self, fname: str):
        """Compute a function summary by analyzing with tainted params."""
        if fname not in self.func_defs:
            return
        fnode = self.func_defs[fname]
        params = [arg.arg for arg in fnode.args.args]

        # Analyze with each param tainted individually to determine transfer
        param_taints_return = set()
        for i, param in enumerate(params):
            env = TaintEnv()
            label = TaintLabel(source=f'param_{i}', origin_line=fnode.lineno,
                               origin_var=param)
            env = env.set(param, TaintValue(frozenset([label])))

            analyzer = PathSensitiveTaintAnalyzer(
                config=self.config,
                filename=self.filename,
                func_name=fname,
                initial_env=env,
                func_summaries=self.func_summaries,
                call_chain=(fname,),
            )
            analyzer.analyze_stmts(fnode.body)

            if analyzer.return_taint.is_tainted:
                param_taints_return.add(i)

        # Check if function introduces new taint (calls source functions)
        introduces_taint = False
        introduced_labels = frozenset()
        env = TaintEnv()
        analyzer = PathSensitiveTaintAnalyzer(
            config=self.config,
            filename=self.filename,
            func_name=fname,
            initial_env=env,
            func_summaries=self.func_summaries,
            call_chain=(fname,),
        )
        analyzer.analyze_stmts(fnode.body)
        if analyzer.return_taint.is_tainted:
            introduces_taint = True
            introduced_labels = analyzer.return_taint.labels

        # Check if it's a sanitizer-like function
        is_sanitizer = False
        sanitizes = set()
        if params:
            # If first param is tainted and return is clean, it sanitizes
            env = TaintEnv()
            all_labels = set()
            for src in ['user_input', 'file_read', 'network']:
                all_labels.add(TaintLabel(source=src, origin_line=0,
                                          origin_var='test'))
            env = env.set(params[0], TaintValue(frozenset(all_labels)))
            analyzer = PathSensitiveTaintAnalyzer(
                config=self.config,
                filename=self.filename,
                func_name=fname,
                initial_env=env,
                func_summaries=self.func_summaries,
                call_chain=(fname,),
            )
            analyzer.analyze_stmts(fnode.body)
            if not analyzer.return_taint.is_tainted:
                is_sanitizer = True
                sanitizes = {'*'}

        self.func_summaries[fname] = FunctionSummary(
            name=fname,
            params=params,
            param_taints_return=param_taints_return,
            introduces_taint=introduces_taint,
            introduced_labels=introduced_labels,
            is_sanitizer=is_sanitizer,
            sanitizes=sanitizes,
        )

    def _summary_changed(self, old: Optional[FunctionSummary],
                         new: Optional[FunctionSummary]) -> bool:
        if old is None or new is None:
            return old is not new
        return (old.param_taints_return != new.param_taints_return or
                old.introduces_taint != new.introduces_taint or
                old.is_sanitizer != new.is_sanitizer)


# ============================================================
# Results
# ============================================================

@dataclass
class TaintAnalysisResult:
    findings: List[TaintFinding]
    func_summaries: Dict[str, FunctionSummary]
    call_graph: Dict[str, Set[str]]
    filename: str

    @property
    def num_findings(self) -> int:
        return len(self.findings)

    def findings_by_severity(self) -> Dict[TaintSeverity, List[TaintFinding]]:
        result = defaultdict(list)
        for f in self.findings:
            result[f.severity].append(f)
        return dict(result)

    def findings_by_kind(self) -> Dict[TaintFindingKind, List[TaintFinding]]:
        result = defaultdict(list)
        for f in self.findings:
            result[f.kind].append(f)
        return dict(result)

    def taint_sinks(self) -> List[TaintFinding]:
        return [f for f in self.findings if f.kind == TaintFindingKind.TAINT_SINK]

    def critical_findings(self) -> List[TaintFinding]:
        return [f for f in self.findings
                if f.severity in (TaintSeverity.CRITICAL, TaintSeverity.ERROR)]

    def summary(self) -> str:
        lines = [f"=== Taint Analysis: {os.path.basename(self.filename)} ==="]
        lines.append(f"Total findings: {self.num_findings}")

        by_sev = self.findings_by_severity()
        for sev in [TaintSeverity.CRITICAL, TaintSeverity.ERROR,
                     TaintSeverity.WARNING, TaintSeverity.INFO]:
            if sev in by_sev:
                lines.append(f"  {sev.value.upper()}: {len(by_sev[sev])}")

        by_kind = self.findings_by_kind()
        if by_kind:
            lines.append("By kind:")
            for kind, findings in sorted(by_kind.items(),
                                         key=lambda x: -len(x[1])):
                lines.append(f"  {kind.value}: {len(findings)}")

        sinks = self.taint_sinks()
        if sinks:
            lines.append(f"\nDangerous sinks reached: {len(sinks)}")
            for s in sinks:
                lines.append(f"  {s}")

        if self.func_summaries:
            taint_funcs = [s for s in self.func_summaries.values()
                           if s.param_taints_return or s.introduces_taint]
            if taint_funcs:
                lines.append(f"\nFunctions that propagate taint: {len(taint_funcs)}")
                for s in taint_funcs:
                    if s.param_taints_return:
                        params = [s.params[i] for i in s.param_taints_return
                                  if i < len(s.params)]
                        lines.append(f"  {s.name}: params {params} -> return")
                    if s.introduces_taint:
                        lines.append(f"  {s.name}: introduces taint")

        return "\n".join(lines)

    def report(self) -> str:
        lines = [self.summary(), ""]
        lines.append("=== Detailed Findings ===")
        for f in sorted(self.findings, key=lambda x: (x.severity.value, x.line)):
            lines.append(str(f))
            if f.labels:
                lines.append(f"  Sources: {', '.join(str(l) for l in f.labels)}")
        return "\n".join(lines)


# ============================================================
# Convenience Functions
# ============================================================

def analyze_taint(source: str, filename: str = "<string>",
                  config: Optional[TaintConfig] = None,
                  entry_points: Optional[List[str]] = None,
                  taint_args: Optional[Dict[str, Dict[str, str]]] = None
                  ) -> TaintAnalysisResult:
    """Analyze Python source for taint flows.

    Args:
        source: Python source code
        filename: For error reporting
        config: Taint configuration (sources, sinks, sanitizers)
        entry_points: Functions to analyze (default: all)
        taint_args: Per-function tainted arguments
    """
    analyzer = InterProceduralAnalyzer(config)
    return analyzer.analyze_source(source, filename, entry_points, taint_args)


def analyze_taint_file(filepath: str,
                       config: Optional[TaintConfig] = None,
                       entry_points: Optional[List[str]] = None,
                       taint_args: Optional[Dict[str, Dict[str, str]]] = None
                       ) -> TaintAnalysisResult:
    """Analyze a Python file for taint flows."""
    analyzer = InterProceduralAnalyzer(config)
    return analyzer.analyze_file(filepath, entry_points, taint_args)
