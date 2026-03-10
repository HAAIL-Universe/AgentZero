"""
V033: Python Code Analyzer
Applies formal-methods-inspired analysis to real Python code.

This is A2's first tool that operates on Python source (via ast module),
not C10 source. Built to analyze A1's actual challenge codebase.

Analyses:
1. Data flow / taint tracking: where do inputs flow?
2. Complexity metrics: cyclomatic, cognitive, nesting depth
3. Def-use analysis: dead code, uninitialized vars, shadowing
4. Exception safety: uncaught paths, bare excepts
5. Mutation analysis: mutable default args, aliasing risks
"""

import ast
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict


# ============================================================
# Core Data Structures
# ============================================================

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class FindingKind(Enum):
    # Data flow
    TAINTED_FLOW = "tainted_flow"
    UNINITIALIZED_USE = "uninitialized_use"
    DEAD_ASSIGNMENT = "dead_assignment"
    SHADOWED_NAME = "shadowed_name"
    # Complexity
    HIGH_CYCLOMATIC = "high_cyclomatic"
    DEEP_NESTING = "deep_nesting"
    HIGH_COGNITIVE = "high_cognitive"
    LONG_FUNCTION = "long_function"
    LONG_PARAMETER_LIST = "long_parameter_list"
    # Exception safety
    BARE_EXCEPT = "bare_except"
    BROAD_EXCEPT = "broad_except"
    MISSING_RAISE_FROM = "missing_raise_from"
    # Mutation hazards
    MUTABLE_DEFAULT = "mutable_default"
    # Type confusion
    INCONSISTENT_RETURN = "inconsistent_return"
    # Import
    UNUSED_IMPORT = "unused_import"


@dataclass
class Finding:
    kind: FindingKind
    severity: Severity
    file: str
    line: int
    col: int
    message: str
    function: str = ""
    details: str = ""

    def __str__(self):
        loc = f"{self.file}:{self.line}"
        if self.function:
            loc += f" ({self.function})"
        return f"[{self.severity.value.upper()}] {loc}: {self.message}"


@dataclass
class FunctionMetrics:
    name: str
    file: str
    line: int
    end_line: int
    num_lines: int
    num_params: int
    cyclomatic: int
    cognitive: int
    max_nesting: int
    num_returns: int
    num_branches: int
    has_loop: bool
    calls: List[str] = field(default_factory=list)


@dataclass
class FileAnalysis:
    file: str
    num_lines: int
    num_functions: int
    num_classes: int
    findings: List[Finding] = field(default_factory=list)
    function_metrics: List[FunctionMetrics] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    files: List[FileAnalysis] = field(default_factory=list)
    total_findings: int = 0
    findings_by_kind: Dict[str, int] = field(default_factory=dict)
    findings_by_severity: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"=== Analysis Summary ({len(self.files)} files) ==="]
        lines.append(f"Total findings: {self.total_findings}")
        if self.findings_by_severity:
            lines.append("By severity:")
            for sev, count in sorted(self.findings_by_severity.items()):
                lines.append(f"  {sev}: {count}")
        if self.findings_by_kind:
            lines.append("By kind:")
            for kind, count in sorted(self.findings_by_kind.items(),
                                       key=lambda x: -x[1]):
                lines.append(f"  {kind}: {count}")
        return "\n".join(lines)


# ============================================================
# AST Visitors
# ============================================================

class ComplexityVisitor(ast.NodeVisitor):
    """Compute cyclomatic and cognitive complexity for a function."""

    def __init__(self):
        self.cyclomatic = 1  # base path
        self.cognitive = 0
        self.max_nesting = 0
        self._nesting = 0
        self.num_returns = 0
        self.num_branches = 0
        self.has_loop = False
        self.calls = []

    def _enter_nesting(self):
        self._nesting += 1
        self.max_nesting = max(self.max_nesting, self._nesting)

    def _exit_nesting(self):
        self._nesting -= 1

    def visit_If(self, node):
        self.cyclomatic += 1
        self.num_branches += 1
        self.cognitive += 1 + self._nesting  # nesting penalty
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_For(self, node):
        self.cyclomatic += 1
        self.has_loop = True
        self.cognitive += 1 + self._nesting
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_While(self, node):
        self.cyclomatic += 1
        self.has_loop = True
        self.cognitive += 1 + self._nesting
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_ExceptHandler(self, node):
        self.cyclomatic += 1
        self.cognitive += 1 + self._nesting
        self._enter_nesting()
        self.generic_visit(node)
        self._exit_nesting()

    def visit_BoolOp(self, node):
        # Each additional boolean operand adds a branch
        self.cyclomatic += len(node.values) - 1
        self.cognitive += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.cyclomatic += 1
        self.cognitive += 1 + self._nesting
        self.generic_visit(node)

    def visit_ListComp(self, node):
        for gen in node.generators:
            self.cyclomatic += 1
            if gen.ifs:
                self.cyclomatic += len(gen.ifs)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.visit_ListComp(node)  # same structure

    def visit_DictComp(self, node):
        for gen in node.generators:
            self.cyclomatic += 1
            if gen.ifs:
                self.cyclomatic += len(gen.ifs)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.visit_ListComp(node)

    def visit_Return(self, node):
        self.num_returns += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.cyclomatic += 1
        self.generic_visit(node)


class DefUseVisitor(ast.NodeVisitor):
    """Track variable definitions and uses within a function scope."""

    def __init__(self):
        self.defs: Dict[str, List[int]] = defaultdict(list)  # name -> [lines]
        self.uses: Dict[str, List[int]] = defaultdict(list)
        self.imports: Set[str] = set()
        self._in_target = False

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.defs[node.id].append(node.lineno)
        elif isinstance(node.ctx, ast.Load):
            self.uses[node.id].append(node.lineno)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.add(name)
            self.defs[name].append(node.lineno)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports.add(name)
            self.defs[name].append(node.lineno)

    def visit_FunctionDef(self, node):
        self.defs[node.name].append(node.lineno)
        for arg in node.args.args:
            self.defs[arg.arg].append(node.lineno)
        for arg in node.args.kwonlyargs:
            self.defs[arg.arg].append(node.lineno)
        if node.args.vararg:
            self.defs[node.args.vararg.arg].append(node.lineno)
        if node.args.kwarg:
            self.defs[node.args.kwarg.arg].append(node.lineno)
        # Don't recurse into nested functions
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.defs[node.name].append(node.lineno)
        # Don't recurse into class bodies for outer scope analysis


class DataFlowVisitor(ast.NodeVisitor):
    """Track data flow through assignments, function calls, returns."""

    def __init__(self):
        self.flows: List[Tuple[str, str, int, str]] = []  # (src, dst, line, kind)
        self.assignments: Dict[str, List[Tuple[int, Set[str]]]] = defaultdict(list)

    def _extract_names(self, node) -> Set[str]:
        """Extract all Name references from an expression."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
        return names

    def visit_Assign(self, node):
        sources = self._extract_names(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id].append((node.lineno, sources))
                for src in sources:
                    self.flows.append((src, target.id, node.lineno, "assign"))
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assignments[elt.id].append((node.lineno, sources))
                        for src in sources:
                            self.flows.append((src, elt.id, node.lineno, "assign"))
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            sources = self._extract_names(node.value) | {node.target.id}
            self.assignments[node.target.id].append((node.lineno, sources))
            for src in sources:
                self.flows.append((src, node.target.id, node.lineno, "aug_assign"))
        self.generic_visit(node)


# ============================================================
# Analyzers
# ============================================================

def analyze_function(node: ast.FunctionDef, filename: str) -> Tuple[FunctionMetrics, List[Finding]]:
    """Analyze a single function for metrics and findings."""
    findings = []

    # Complexity
    cv = ComplexityVisitor()
    cv.visit(node)

    end_line = getattr(node, 'end_lineno', node.lineno)
    num_lines = end_line - node.lineno + 1

    num_params = len(node.args.args)
    if node.args.vararg:
        num_params += 1
    if node.args.kwarg:
        num_params += 1
    num_params += len(node.args.kwonlyargs)

    metrics = FunctionMetrics(
        name=node.name,
        file=filename,
        line=node.lineno,
        end_line=end_line,
        num_lines=num_lines,
        num_params=num_params,
        cyclomatic=cv.cyclomatic,
        cognitive=cv.cognitive,
        max_nesting=cv.max_nesting,
        num_returns=cv.num_returns,
        num_branches=cv.num_branches,
        has_loop=cv.has_loop,
        calls=cv.calls,
    )

    # Findings
    if cv.cyclomatic > 10:
        findings.append(Finding(
            kind=FindingKind.HIGH_CYCLOMATIC,
            severity=Severity.WARNING if cv.cyclomatic <= 20 else Severity.ERROR,
            file=filename, line=node.lineno, col=0,
            message=f"Function '{node.name}' has cyclomatic complexity {cv.cyclomatic} (>10)",
            function=node.name,
        ))

    if cv.max_nesting > 4:
        findings.append(Finding(
            kind=FindingKind.DEEP_NESTING,
            severity=Severity.WARNING,
            file=filename, line=node.lineno, col=0,
            message=f"Function '{node.name}' has max nesting depth {cv.max_nesting} (>4)",
            function=node.name,
        ))

    if cv.cognitive > 15:
        findings.append(Finding(
            kind=FindingKind.HIGH_COGNITIVE,
            severity=Severity.WARNING if cv.cognitive <= 30 else Severity.ERROR,
            file=filename, line=node.lineno, col=0,
            message=f"Function '{node.name}' has cognitive complexity {cv.cognitive} (>15)",
            function=node.name,
        ))

    if num_lines > 60:
        findings.append(Finding(
            kind=FindingKind.LONG_FUNCTION,
            severity=Severity.WARNING if num_lines <= 100 else Severity.ERROR,
            file=filename, line=node.lineno, col=0,
            message=f"Function '{node.name}' is {num_lines} lines (>60)",
            function=node.name,
        ))

    if num_params > 5:
        findings.append(Finding(
            kind=FindingKind.LONG_PARAMETER_LIST,
            severity=Severity.WARNING,
            file=filename, line=node.lineno, col=0,
            message=f"Function '{node.name}' has {num_params} parameters (>5)",
            function=node.name,
        ))

    # Mutable default arguments
    for default in node.args.defaults + node.args.kw_defaults:
        if default and isinstance(default, (ast.List, ast.Dict, ast.Set, ast.Call)):
            if isinstance(default, ast.Call):
                # Allow specific safe calls like field(default_factory=...)
                if isinstance(default.func, ast.Name) and default.func.id in ('field',):
                    continue
            findings.append(Finding(
                kind=FindingKind.MUTABLE_DEFAULT,
                severity=Severity.WARNING,
                file=filename, line=node.lineno, col=0,
                message=f"Function '{node.name}' has mutable default argument",
                function=node.name,
            ))
            break

    # Inconsistent returns (some return value, some don't)
    returns_with_value = []
    returns_without_value = []
    for child in ast.walk(node):
        if isinstance(child, ast.Return):
            if child.value is None:
                returns_without_value.append(child.lineno)
            else:
                returns_with_value.append(child.lineno)
    if returns_with_value and returns_without_value:
        findings.append(Finding(
            kind=FindingKind.INCONSISTENT_RETURN,
            severity=Severity.WARNING,
            file=filename, line=node.lineno, col=0,
            message=f"Function '{node.name}' has both value-returns and bare returns",
            function=node.name,
            details=f"value-returns at lines {returns_with_value[:3]}, bare at {returns_without_value[:3]}",
        ))

    return metrics, findings


def analyze_exception_safety(tree: ast.AST, filename: str) -> List[Finding]:
    """Check for exception handling anti-patterns."""
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(Finding(
                    kind=FindingKind.BARE_EXCEPT,
                    severity=Severity.WARNING,
                    file=filename, line=node.lineno, col=0,
                    message="Bare 'except:' catches all exceptions including SystemExit/KeyboardInterrupt",
                ))
            elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                findings.append(Finding(
                    kind=FindingKind.BROAD_EXCEPT,
                    severity=Severity.INFO,
                    file=filename, line=node.lineno, col=0,
                    message="Broad 'except Exception' -- consider catching specific exceptions",
                ))
    return findings


def analyze_imports(tree: ast.AST, filename: str) -> Tuple[List[str], List[Finding]]:
    """Analyze imports and detect unused ones."""
    findings = []
    imports = []

    import_names = {}  # name -> line
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                import_names[name] = node.lineno
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                import_names[name] = node.lineno
                imports.append(f"{node.module}.{alias.name}" if node.module else alias.name)

    # Collect all Name references
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            # Check if the root is a used import
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used_names.add(root.id)

    for name, line in import_names.items():
        if name not in used_names and name != '__all__' and not name.startswith('_'):
            findings.append(Finding(
                kind=FindingKind.UNUSED_IMPORT,
                severity=Severity.INFO,
                file=filename, line=line, col=0,
                message=f"Import '{name}' appears unused",
            ))

    return imports, findings


def taint_analysis(tree: ast.AST, filename: str, taint_sources: Set[str]) -> List[Finding]:
    """Track how taint sources flow through the code.

    taint_sources: set of parameter names or variable names considered 'tainted'
    (e.g., user inputs, external data).
    """
    findings = []
    tainted: Set[str] = set(taint_sources)

    class TaintTracker(ast.NodeVisitor):
        def __init__(self):
            self.tainted = set(taint_sources)

        def _is_tainted_expr(self, node) -> bool:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in self.tainted:
                    return True
            return False

        def _taint_sources_in(self, node) -> Set[str]:
            sources = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in self.tainted:
                    sources.add(child.id)
            return sources

        def visit_Assign(self, node):
            if self._is_tainted_expr(node.value):
                sources = self._taint_sources_in(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.tainted.add(target.id)
                        findings.append(Finding(
                            kind=FindingKind.TAINTED_FLOW,
                            severity=Severity.INFO,
                            file=filename, line=node.lineno, col=0,
                            message=f"Taint flows from {sources} to '{target.id}'",
                        ))
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check if tainted data flows into sensitive sinks
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            sensitive_sinks = {'exec', 'eval', 'compile', 'system', 'popen',
                             'run', 'subprocess', 'execute', 'raw_input'}

            if func_name in sensitive_sinks:
                for arg in node.args:
                    if self._is_tainted_expr(arg):
                        sources = self._taint_sources_in(arg)
                        findings.append(Finding(
                            kind=FindingKind.TAINTED_FLOW,
                            severity=Severity.ERROR,
                            file=filename, line=node.lineno, col=0,
                            message=f"Tainted data from {sources} flows into sensitive sink '{func_name}'",
                        ))
            self.generic_visit(node)

    tracker = TaintTracker()
    tracker.visit(tree)
    return findings


# ============================================================
# File Analysis
# ============================================================

def analyze_file(filepath: str, taint_sources: Optional[Set[str]] = None) -> FileAnalysis:
    """Analyze a single Python file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return FileAnalysis(
            file=filepath,
            num_lines=source.count('\n') + 1,
            num_functions=0,
            num_classes=0,
            findings=[Finding(
                kind=FindingKind.HIGH_CYCLOMATIC,  # reuse for syntax errors
                severity=Severity.ERROR,
                file=filepath, line=e.lineno or 0, col=e.offset or 0,
                message=f"Syntax error: {e.msg}",
            )],
        )

    lines = source.split('\n')
    num_lines = len(lines)

    # Count top-level constructs
    num_functions = 0
    num_classes = 0
    findings = []
    function_metrics = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            num_functions += 1
            metrics, fn_findings = analyze_function(node, filepath)
            function_metrics.append(metrics)
            findings.extend(fn_findings)
        elif isinstance(node, ast.ClassDef):
            num_classes += 1

    # Exception safety
    findings.extend(analyze_exception_safety(tree, filepath))

    # Imports
    imports, import_findings = analyze_imports(tree, filepath)
    findings.extend(import_findings)

    # Taint analysis
    if taint_sources:
        findings.extend(taint_analysis(tree, filepath, taint_sources))

    return FileAnalysis(
        file=filepath,
        num_lines=num_lines,
        num_functions=num_functions,
        num_classes=num_classes,
        findings=findings,
        function_metrics=function_metrics,
        imports=imports,
    )


def analyze_directory(dirpath: str, taint_sources: Optional[Set[str]] = None,
                      exclude_tests: bool = True) -> AnalysisResult:
    """Analyze all Python files in a directory."""
    result = AnalysisResult()

    for root, dirs, files in os.walk(dirpath):
        for f in files:
            if not f.endswith('.py'):
                continue
            if exclude_tests and (f.startswith('test_') or f.endswith('_test.py')):
                continue

            filepath = os.path.join(root, f)
            file_analysis = analyze_file(filepath, taint_sources)
            result.files.append(file_analysis)

    # Aggregate
    for fa in result.files:
        result.total_findings += len(fa.findings)
        for finding in fa.findings:
            kind_name = finding.kind.value
            result.findings_by_kind[kind_name] = result.findings_by_kind.get(kind_name, 0) + 1
            sev_name = finding.severity.value
            result.findings_by_severity[sev_name] = result.findings_by_severity.get(sev_name, 0) + 1

    return result


# ============================================================
# Targeted Analysis: A1's Verification Stack
# ============================================================

def analyze_a1_challenges(challenge_dir: str) -> AnalysisResult:
    """Analyze A1's challenge codebase with domain-specific checks."""
    targets = [
        'C010_stack_vm',
        'C013_type_checker',
        'C014_bytecode_optimizer',
        'C035_sat_solver',
        'C036_model_checker',
        'C037_smt_solver',
        'C038_symbolic_execution',
        'C039_abstract_interpreter',
    ]

    result = AnalysisResult()

    for target in targets:
        target_dir = os.path.join(challenge_dir, target)
        if not os.path.isdir(target_dir):
            continue
        dir_result = analyze_directory(target_dir)
        result.files.extend(dir_result.files)

    # Re-aggregate
    result.total_findings = 0
    result.findings_by_kind = {}
    result.findings_by_severity = {}
    for fa in result.files:
        result.total_findings += len(fa.findings)
        for finding in fa.findings:
            kind_name = finding.kind.value
            result.findings_by_kind[kind_name] = result.findings_by_kind.get(kind_name, 0) + 1
            sev_name = finding.severity.value
            result.findings_by_severity[sev_name] = result.findings_by_severity.get(sev_name, 0) + 1

    return result


def complexity_report(result: AnalysisResult) -> str:
    """Generate a detailed complexity report from analysis results."""
    lines = ["=== Complexity Report ===\n"]

    # Collect all function metrics
    all_metrics = []
    for fa in result.files:
        all_metrics.extend(fa.function_metrics)

    if not all_metrics:
        return "No functions found."

    # Sort by cyclomatic complexity
    all_metrics.sort(key=lambda m: m.cyclomatic, reverse=True)

    lines.append(f"Total functions analyzed: {len(all_metrics)}")
    lines.append(f"Total source files: {len(result.files)}")
    total_lines = sum(fa.num_lines for fa in result.files)
    lines.append(f"Total lines of code: {total_lines}")
    lines.append("")

    # Top 20 by cyclomatic complexity
    lines.append("--- Top 20 by Cyclomatic Complexity ---")
    for m in all_metrics[:20]:
        short_file = os.path.basename(m.file)
        lines.append(
            f"  CC={m.cyclomatic:3d}  Cog={m.cognitive:3d}  "
            f"Nest={m.max_nesting}  Lines={m.num_lines:3d}  "
            f"Params={m.num_params}  "
            f"{short_file}:{m.line} {m.name}"
        )
    lines.append("")

    # Top 20 by cognitive complexity
    all_metrics.sort(key=lambda m: m.cognitive, reverse=True)
    lines.append("--- Top 20 by Cognitive Complexity ---")
    for m in all_metrics[:20]:
        short_file = os.path.basename(m.file)
        lines.append(
            f"  Cog={m.cognitive:3d}  CC={m.cyclomatic:3d}  "
            f"Nest={m.max_nesting}  Lines={m.num_lines:3d}  "
            f"{short_file}:{m.line} {m.name}"
        )
    lines.append("")

    # Longest functions
    all_metrics.sort(key=lambda m: m.num_lines, reverse=True)
    lines.append("--- Top 20 by Lines ---")
    for m in all_metrics[:20]:
        short_file = os.path.basename(m.file)
        lines.append(
            f"  Lines={m.num_lines:3d}  CC={m.cyclomatic:3d}  "
            f"Cog={m.cognitive:3d}  "
            f"{short_file}:{m.line} {m.name}"
        )
    lines.append("")

    # Summary statistics
    ccs = [m.cyclomatic for m in all_metrics]
    cogs = [m.cognitive for m in all_metrics]
    lns = [m.num_lines for m in all_metrics]
    lines.append("--- Statistics ---")
    lines.append(f"  Cyclomatic: avg={sum(ccs)/len(ccs):.1f}, max={max(ccs)}, "
                 f">{10}={sum(1 for c in ccs if c > 10)}/{len(ccs)}")
    lines.append(f"  Cognitive:  avg={sum(cogs)/len(cogs):.1f}, max={max(cogs)}, "
                 f">{15}={sum(1 for c in cogs if c > 15)}/{len(cogs)}")
    lines.append(f"  Lines:      avg={sum(lns)/len(lns):.1f}, max={max(lns)}, "
                 f">{60}={sum(1 for l in lns if l > 60)}/{len(lns)}")

    return "\n".join(lines)


def findings_report(result: AnalysisResult) -> str:
    """Generate a detailed findings report."""
    lines = ["=== Findings Report ===\n"]

    # Group by file
    for fa in result.files:
        if not fa.findings:
            continue
        short_file = os.path.basename(fa.file)
        lines.append(f"--- {short_file} ({len(fa.findings)} findings) ---")
        for f in sorted(fa.findings, key=lambda x: x.line):
            lines.append(f"  {f}")
            if f.details:
                lines.append(f"    {f.details}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Default: analyze A1's verification stack
        challenge_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                     'challenges')
        if not os.path.isdir(challenge_dir):
            challenge_dir = r'Z:\AgentZero\challenges'

        print("Analyzing A1's verification stack...\n")
        result = analyze_a1_challenges(challenge_dir)
    else:
        target = sys.argv[1]
        if os.path.isdir(target):
            result = analyze_directory(target)
        else:
            fa = analyze_file(target)
            result = AnalysisResult(
                files=[fa],
                total_findings=len(fa.findings),
            )

    print(result.summary())
    print()
    print(complexity_report(result))
    print()
    print(findings_report(result))
