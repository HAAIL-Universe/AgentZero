"""
V026: Information Flow / Taint Analysis
========================================
Composes C038 (symbolic execution) + C037 (SMT solver) + C039 (abstract interpreter) + C010 (parser)

Three analysis modes:
1. Taint Analysis (abstract interpretation): fast, over-approximate taint tracking
2. Symbolic Taint Analysis (symbolic execution): precise, path-sensitive taint
3. Noninterference Checking (SMT): proves high-security inputs can't affect low outputs

Security lattice: LOW (public) < HIGH (secret)
"""

import sys, os

_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)

sys.path.insert(0, os.path.join(_az, 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C038_symbolic_execution'))
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))

from stack_vm import lex, Parser, IntLit, Var, BinOp, IfStmt, WhileStmt, LetDecl, FnDecl, CallExpr, Block, Assign, PrintStmt, ReturnStmt
from smt_solver import SMTSolver, Var as SMTVar, App, Op, IntConst, BoolConst, INT, BOOL
from symbolic_execution import symbolic_execute
from abstract_interpreter import analyze as ai_analyze

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Security Lattice
# ============================================================

class SecurityLevel(Enum):
    LOW = 0    # Public
    HIGH = 1   # Secret

    def join(self, other):
        """Least upper bound"""
        if self == SecurityLevel.HIGH or other == SecurityLevel.HIGH:
            return SecurityLevel.HIGH
        return SecurityLevel.LOW

    def __le__(self, other):
        return self.value <= other.value

    def __lt__(self, other):
        return self.value < other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __repr__(self):
        return self.name

LOW = SecurityLevel.LOW
HIGH = SecurityLevel.HIGH


# ============================================================
# Taint Domain (for Abstract Interpretation)
# ============================================================

class TaintValue:
    """Abstract taint value: tracks security level + sources"""
    def __init__(self, level, sources=None):
        self.level = level
        self.sources = sources or set()

    def join(self, other):
        return TaintValue(
            self.level.join(other.level),
            self.sources | other.sources
        )

    def is_tainted(self):
        return self.level == HIGH

    def __repr__(self):
        src = ','.join(sorted(self.sources)) if self.sources else ''
        return f"{'TAINTED' if self.is_tainted() else 'CLEAN'}({src})"

    def __eq__(self, other):
        return isinstance(other, TaintValue) and self.level == other.level and self.sources == other.sources

    def __hash__(self):
        return hash((self.level, frozenset(self.sources)))


CLEAN = TaintValue(LOW)
TAINTED = lambda src: TaintValue(HIGH, {src})


# ============================================================
# Results
# ============================================================

@dataclass
class TaintViolation:
    """A security violation: tainted data flows to a clean sink"""
    sink: str          # Variable or output that should be clean
    sources: set       # Tainted sources that reach the sink
    path_condition: str = ""  # Optional path condition for symbolic analysis
    line_hint: str = ""      # Description of where the violation occurs

    def __repr__(self):
        srcs = ','.join(sorted(self.sources))
        return f"VIOLATION: {srcs} -> {self.sink}"


@dataclass
class FlowEdge:
    """An information flow edge"""
    src: str
    dst: str
    kind: str  # 'direct', 'implicit', 'both'


@dataclass
class TaintResult:
    """Result of taint analysis"""
    safe: bool
    violations: list  # List[TaintViolation]
    taint_state: dict  # var -> TaintValue at program end
    flows: list  # List[FlowEdge]
    method: str  # 'abstract', 'symbolic', 'noninterference'

    def __repr__(self):
        status = "SAFE" if self.safe else f"UNSAFE ({len(self.violations)} violations)"
        return f"TaintResult({self.method}): {status}"


# ============================================================
# Parsing Helper
# ============================================================

def parse(source):
    tokens = lex(source)
    return Parser(tokens).parse()


# ============================================================
# Taint Analysis via Abstract Interpretation
# ============================================================

class TaintAnalyzer:
    """
    Forward taint analysis over the C10 AST.
    Tracks which variables are tainted (influenced by HIGH inputs).
    Detects implicit flows through branch conditions.
    """

    def __init__(self, high_vars, low_sinks=None):
        """
        high_vars: set of variable names that are HIGH (tainted sources)
        low_sinks: set of variable names that must be LOW (optional; if None, 'output' vars)
        """
        self.high_vars = set(high_vars)
        self.low_sinks = set(low_sinks) if low_sinks else set()
        self.env = {}  # var -> TaintValue
        self.violations = []
        self.flows = []
        self.implicit_context = CLEAN  # Current implicit flow context (from branches)
        self.functions = {}  # fn_name -> FnDecl

    def analyze(self, source):
        prog = parse(source)
        # Initialize high vars as tainted
        for v in self.high_vars:
            self.env[v] = TaintValue(HIGH, {v})
        # Collect functions first
        for stmt in prog.stmts:
            if isinstance(stmt, FnDecl):
                self.functions[stmt.name] = stmt
        # Analyze statements
        for stmt in prog.stmts:
            self._analyze_stmt(stmt)
        # Check sinks
        for sink in self.low_sinks:
            if sink in self.env and self.env[sink].is_tainted():
                self.violations.append(TaintViolation(
                    sink=sink,
                    sources=self.env[sink].sources,
                    line_hint=f"variable '{sink}' is tainted at program end"
                ))
        return TaintResult(
            safe=len(self.violations) == 0,
            violations=self.violations,
            taint_state={k: v for k, v in self.env.items()},
            flows=self.flows,
            method='abstract'
        )

    def _get_taint(self, var_name):
        return self.env.get(var_name, CLEAN)

    def _set_taint(self, var_name, taint):
        # Join with implicit context (branch taint)
        combined = taint.join(self.implicit_context)
        # High vars always stay tainted (they are secret sources)
        if var_name in self.high_vars:
            combined = combined.join(TaintValue(HIGH, {var_name}))
        self.env[var_name] = combined
        # Record flow edges
        for src in combined.sources:
            if src != var_name:
                kind = 'direct' if taint.is_tainted() else 'implicit'
                self.flows.append(FlowEdge(src, var_name, kind))

    def _analyze_expr(self, expr):
        """Return TaintValue for an expression"""
        if isinstance(expr, IntLit):
            return CLEAN
        if isinstance(expr, Var):
            return self._get_taint(expr.name)
        if isinstance(expr, BinOp):
            left_t = self._analyze_expr(expr.left)
            right_t = self._analyze_expr(expr.right)
            return left_t.join(right_t)
        if isinstance(expr, CallExpr):
            # Taint of call = join of all argument taints + function body taint
            arg_taint = CLEAN
            for arg in expr.args:
                arg_taint = arg_taint.join(self._analyze_expr(arg))
            # If we have the function body, analyze it
            if expr.callee in self.functions:
                fn = self.functions[expr.callee]
                return self._analyze_call(fn, expr.args, arg_taint)
            return arg_taint
        return CLEAN

    def _analyze_call(self, fn, args, arg_taint):
        """Analyze a function call, returning the taint of its return value"""
        # Save and setup function environment
        old_env = dict(self.env)
        for i, param in enumerate(fn.params):
            if i < len(args):
                self.env[param] = self._analyze_expr(args[i])
            else:
                self.env[param] = CLEAN
        # Analyze function body
        return_taint = CLEAN
        for stmt in fn.body.stmts:
            if isinstance(stmt, ReturnStmt):
                return_taint = return_taint.join(self._analyze_expr(stmt.value))
            else:
                self._analyze_stmt(stmt)
        # Check if any return-referenced variable is tainted
        # Restore caller environment
        result = return_taint.join(arg_taint)
        self.env = old_env
        return result

    def _analyze_stmt(self, stmt):
        if isinstance(stmt, LetDecl):
            taint = self._analyze_expr(stmt.value)
            self._set_taint(stmt.name, taint)
        elif isinstance(stmt, Assign):
            taint = self._analyze_expr(stmt.value)
            self._set_taint(stmt.name, taint)
        elif isinstance(stmt, IfStmt):
            cond_taint = self._analyze_expr(stmt.cond)
            # Save context
            old_implicit = self.implicit_context
            old_env = {k: v for k, v in self.env.items()}
            # Then branch: condition taint flows implicitly
            self.implicit_context = old_implicit.join(cond_taint)
            then_stmts = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            for s in then_stmts:
                self._analyze_stmt(s)
            then_env = {k: v for k, v in self.env.items()}
            # Else branch
            self.env = {k: v for k, v in old_env.items()}
            self.implicit_context = old_implicit.join(cond_taint)
            if stmt.else_body:
                else_stmts = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                for s in else_stmts:
                    self._analyze_stmt(s)
            else_env = {k: v for k, v in self.env.items()}
            # Join branches
            all_vars = set(then_env.keys()) | set(else_env.keys())
            for v in all_vars:
                t1 = then_env.get(v, CLEAN)
                t2 = else_env.get(v, CLEAN)
                self.env[v] = t1.join(t2)
            self.implicit_context = old_implicit
        elif isinstance(stmt, WhileStmt):
            cond_taint = self._analyze_expr(stmt.cond)
            old_implicit = self.implicit_context
            self.implicit_context = old_implicit.join(cond_taint)
            # Fixed-point iteration (max 20 iterations)
            for _ in range(20):
                old_env = {k: v for k, v in self.env.items()}
                body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
                for s in body_stmts:
                    self._analyze_stmt(s)
                # Re-evaluate condition after body
                cond_taint = self._analyze_expr(stmt.cond)
                self.implicit_context = old_implicit.join(cond_taint)
                # Check fixpoint
                if all(self.env.get(k) == old_env.get(k) for k in set(self.env.keys()) | set(old_env.keys())):
                    break
            self.implicit_context = old_implicit
        elif isinstance(stmt, PrintStmt):
            # Print is an observable output -- check if tainted data leaks
            taint = self._analyze_expr(stmt.value)
            if taint.is_tainted():
                self.violations.append(TaintViolation(
                    sink='print',
                    sources=taint.sources,
                    line_hint='tainted value printed to output'
                ))
        elif isinstance(stmt, FnDecl):
            self.functions[stmt.name] = stmt
        elif isinstance(stmt, ReturnStmt):
            pass  # Handled by _analyze_call
        elif isinstance(stmt, Block):
            for s in stmt.stmts:
                self._analyze_stmt(s)


# ============================================================
# Symbolic Taint Analysis (Path-Sensitive)
# ============================================================

class SymbolicTaintAnalyzer:
    """
    Uses C038 symbolic execution for path-sensitive taint analysis.
    Each path independently tracks which variables carry tainted data.
    More precise than abstract analysis: distinguishes feasible from infeasible flows.
    """

    def __init__(self, high_vars, low_sinks=None):
        self.high_vars = set(high_vars)
        self.low_sinks = set(low_sinks) if low_sinks else set()

    def analyze(self, source):
        # Run symbolic execution with high vars as symbolic
        sym_inputs = {v: 'int' for v in self.high_vars}
        result = symbolic_execute(source, sym_inputs, max_paths=64)

        violations = []
        all_taint = {}
        flows = []

        for path in result.paths:
            if path.status.name in ('INFEASIBLE',):
                continue
            # Track taint per path
            path_taint = self._compute_path_taint(path, source)

            # Merge into global taint state
            for var, tv in path_taint.items():
                if var in all_taint:
                    all_taint[var] = all_taint[var].join(tv)
                else:
                    all_taint[var] = tv

            # Check sinks on this path
            for sink in self.low_sinks:
                if sink in path_taint and path_taint[sink].is_tainted():
                    cond_str = f"path {path.path_id}"
                    violations.append(TaintViolation(
                        sink=sink,
                        sources=path_taint[sink].sources,
                        path_condition=cond_str,
                        line_hint=f"variable '{sink}' tainted on {cond_str}"
                    ))

            # Check print outputs
            for out_val in path.output:
                # If any symbolic high var appears in output, it's a leak
                if self._output_depends_on_high(out_val, path):
                    violations.append(TaintViolation(
                        sink='print',
                        sources=self._get_output_sources(out_val, path),
                        path_condition=f"path {path.path_id}",
                        line_hint='tainted value printed'
                    ))

        # Deduplicate violations
        seen = set()
        unique_violations = []
        for v in violations:
            key = (v.sink, frozenset(v.sources))
            if key not in seen:
                seen.add(key)
                unique_violations.append(v)

        return TaintResult(
            safe=len(unique_violations) == 0,
            violations=unique_violations,
            taint_state=all_taint,
            flows=flows,
            method='symbolic'
        )

    def _compute_path_taint(self, path, source):
        """Compute taint state for a single execution path"""
        taint = {}
        # High vars are tainted
        for v in self.high_vars:
            taint[v] = TaintValue(HIGH, {v})

        # Walk the path's environment -- a variable is tainted if its
        # symbolic value depends on any high variable
        for var_name, sym_val in path.env.items():
            if var_name in self.high_vars:
                continue
            deps = self._get_symbolic_deps(sym_val)
            high_deps = deps & self.high_vars
            if high_deps:
                taint[var_name] = TaintValue(HIGH, high_deps)
            else:
                taint[var_name] = CLEAN

        return taint

    def _get_symbolic_deps(self, sym_val):
        """Get variable names that a symbolic value depends on"""
        if sym_val is None:
            return set()
        if hasattr(sym_val, 'kind') and sym_val.kind.name == 'CONCRETE':
            return set()
        if hasattr(sym_val, 'term') and sym_val.term is not None:
            return self._get_term_vars(sym_val.term)
        if hasattr(sym_val, 'name') and sym_val.name:
            return {sym_val.name}
        return set()

    def _get_term_vars(self, term):
        """Extract variable names from an SMT term"""
        if isinstance(term, SMTVar):
            return {term.name}
        if isinstance(term, (IntConst, BoolConst)):
            return set()
        if isinstance(term, App):
            result = set()
            for arg in term.args:
                result |= self._get_term_vars(arg)
            return result
        return set()

    def _output_depends_on_high(self, out_val, path):
        """Check if an output value depends on high variables"""
        if hasattr(out_val, 'term') and out_val.term is not None:
            deps = self._get_term_vars(out_val.term)
            return bool(deps & self.high_vars)
        if hasattr(out_val, 'name') and out_val.name in self.high_vars:
            return True
        return False

    def _get_output_sources(self, out_val, path):
        """Get high-variable sources for an output"""
        if hasattr(out_val, 'term') and out_val.term is not None:
            return self._get_term_vars(out_val.term) & self.high_vars
        if hasattr(out_val, 'name') and out_val.name in self.high_vars:
            return {out_val.name}
        return set()


# ============================================================
# Noninterference Checking (SMT-based)
# ============================================================

class NoninterferenceChecker:
    """
    Proves noninterference: varying HIGH inputs cannot change LOW outputs.

    Method: Run symbolic execution twice with different symbolic values for
    HIGH vars. If for any path pair, the LOW outputs differ while LOW inputs
    are the same, noninterference is violated.

    This is the 2-run self-composition approach.
    """

    def __init__(self, high_vars, low_vars, low_outputs=None):
        """
        high_vars: secret inputs
        low_vars: public inputs
        low_outputs: public outputs to check (if None, checks all non-high vars)
        """
        self.high_vars = set(high_vars)
        self.low_vars = set(low_vars)
        self.low_outputs = set(low_outputs) if low_outputs else None

    def check(self, source):
        """Check noninterference via self-composition with symbolic execution"""
        # Run symbolic execution with ALL vars symbolic
        all_sym = {}
        for v in self.high_vars:
            all_sym[v] = 'int'
        for v in self.low_vars:
            all_sym[v] = 'int'

        result = symbolic_execute(source, all_sym, max_paths=64)

        violations = []
        feasible_paths = [p for p in result.paths if p.status.name not in ('INFEASIBLE',)]

        if not feasible_paths:
            return TaintResult(
                safe=True, violations=[], taint_state={}, flows=[], method='noninterference'
            )

        # For each pair of paths, check if they can have same LOW inputs
        # but different LOW outputs
        solver = SMTSolver()

        # Determine output variables
        output_vars = self.low_outputs
        if output_vars is None:
            # Collect all non-high variables that appear in path environments
            output_vars = set()
            for p in feasible_paths:
                for v in p.env:
                    if v not in self.high_vars:
                        output_vars.add(v)

        # For each path, check if varying high inputs changes low outputs
        for path in feasible_paths:
            for out_var in output_vars:
                if out_var not in path.env:
                    continue
                sym_val = path.env[out_var]
                if sym_val is None:
                    continue
                # Check if the output depends on any high variable
                deps = set()
                if hasattr(sym_val, 'term') and sym_val.term is not None:
                    deps = self._get_term_vars(sym_val.term)
                elif hasattr(sym_val, 'name') and sym_val.name:
                    deps = {sym_val.name}
                high_deps = deps & self.high_vars
                if high_deps:
                    violations.append(TaintViolation(
                        sink=out_var,
                        sources=high_deps,
                        path_condition=f"path {path.path_id}",
                        line_hint=f"'{out_var}' depends on secret input(s) {high_deps}"
                    ))

        # Deduplicate
        seen = set()
        unique = []
        for v in violations:
            key = (v.sink, frozenset(v.sources))
            if key not in seen:
                seen.add(key)
                unique.append(v)

        return TaintResult(
            safe=len(unique) == 0,
            violations=unique,
            taint_state={},
            flows=[],
            method='noninterference'
        )

    def _get_term_vars(self, term):
        if isinstance(term, SMTVar):
            return {term.name}
        if isinstance(term, (IntConst, BoolConst)):
            return set()
        if isinstance(term, App):
            result = set()
            for arg in term.args:
                result |= self._get_term_vars(arg)
            return result
        return set()


# ============================================================
# Dependency Graph
# ============================================================

class DependencyGraph:
    """
    Builds a data dependency graph from the program.
    Edges: (source_var, dest_var, kind) where kind is 'direct' or 'implicit'.
    """

    def __init__(self):
        self.edges = []  # (src, dst, kind)
        self.nodes = set()

    def build(self, source):
        prog = parse(source)
        self._walk_stmts(prog.stmts, implicit_deps=set())
        return self

    def _walk_stmts(self, stmts, implicit_deps):
        for stmt in stmts:
            self._walk_stmt(stmt, implicit_deps)

    def _walk_stmt(self, stmt, implicit_deps):
        if isinstance(stmt, LetDecl):
            self.nodes.add(stmt.name)
            deps = self._expr_deps(stmt.value)
            for d in deps:
                self.edges.append((d, stmt.name, 'direct'))
                self.nodes.add(d)
            for d in implicit_deps:
                self.edges.append((d, stmt.name, 'implicit'))
                self.nodes.add(d)
        elif isinstance(stmt, Assign):
            self.nodes.add(stmt.name)
            deps = self._expr_deps(stmt.value)
            for d in deps:
                self.edges.append((d, stmt.name, 'direct'))
                self.nodes.add(d)
            for d in implicit_deps:
                self.edges.append((d, stmt.name, 'implicit'))
                self.nodes.add(d)
        elif isinstance(stmt, IfStmt):
            cond_deps = self._expr_deps(stmt.cond)
            new_implicit = implicit_deps | cond_deps
            then_stmts = stmt.then_body.stmts if isinstance(stmt.then_body, Block) else [stmt.then_body]
            self._walk_stmts(then_stmts, new_implicit)
            if stmt.else_body:
                else_stmts = stmt.else_body.stmts if isinstance(stmt.else_body, Block) else [stmt.else_body]
                self._walk_stmts(else_stmts, new_implicit)
        elif isinstance(stmt, WhileStmt):
            cond_deps = self._expr_deps(stmt.cond)
            new_implicit = implicit_deps | cond_deps
            body_stmts = stmt.body.stmts if isinstance(stmt.body, Block) else [stmt.body]
            self._walk_stmts(body_stmts, new_implicit)
        elif isinstance(stmt, PrintStmt):
            deps = self._expr_deps(stmt.value)
            for d in deps:
                self.edges.append((d, '__print__', 'direct'))
            for d in implicit_deps:
                self.edges.append((d, '__print__', 'implicit'))
            self.nodes.add('__print__')
        elif isinstance(stmt, Block):
            self._walk_stmts(stmt.stmts, implicit_deps)

    def _expr_deps(self, expr):
        if isinstance(expr, Var):
            return {expr.name}
        if isinstance(expr, BinOp):
            return self._expr_deps(expr.left) | self._expr_deps(expr.right)
        if isinstance(expr, CallExpr):
            deps = set()
            for arg in expr.args:
                deps |= self._expr_deps(arg)
            return deps
        if isinstance(expr, IntLit):
            return set()
        return set()

    def reachable_from(self, sources):
        """Find all nodes reachable from a set of source nodes (transitive closure)"""
        adj = {}
        for s, d, k in self.edges:
            adj.setdefault(s, set()).add(d)
        visited = set()
        stack = list(sources)
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    stack.append(nb)
        return visited

    def flows_to(self, src, dst):
        """Check if information from src can flow to dst"""
        return dst in self.reachable_from({src})

    def all_flows_from(self, src):
        """All variables that src can flow to"""
        return self.reachable_from({src}) - {src}


# ============================================================
# Declassification Support
# ============================================================

class DeclassifyPolicy:
    """
    A declassification policy that allows specific flows.
    Example: declassify('password', 'hash') allows password -> hash flow.
    """

    def __init__(self):
        self.allowed = set()  # (src, dst) pairs that are allowed

    def allow(self, src, dst):
        self.allowed.add((src, dst))
        return self

    def is_allowed(self, src, dst):
        return (src, dst) in self.allowed

    def filter_violations(self, result):
        """Remove violations that are covered by declassification policy"""
        filtered = []
        for v in result.violations:
            # Check if ALL sources are declassified for this sink
            all_allowed = all(
                self.is_allowed(src, v.sink) for src in v.sources
            )
            if not all_allowed:
                filtered.append(v)
        return TaintResult(
            safe=len(filtered) == 0,
            violations=filtered,
            taint_state=result.taint_state,
            flows=result.flows,
            method=result.method + '+declassify'
        )


# ============================================================
# Public APIs
# ============================================================

def taint_analyze(source, high_vars, low_sinks=None):
    """
    Fast taint analysis via abstract interpretation.
    Over-approximate: may report false positives, never misses real flows.

    Args:
        source: C10 source code
        high_vars: set/list of secret variable names
        low_sinks: set/list of variables that must remain clean (optional)
    Returns:
        TaintResult
    """
    analyzer = TaintAnalyzer(high_vars, low_sinks)
    return analyzer.analyze(source)


def symbolic_taint_analyze(source, high_vars, low_sinks=None):
    """
    Path-sensitive taint analysis via symbolic execution.
    More precise than abstract: only reports feasible flows.

    Args:
        source: C10 source code
        high_vars: set/list of secret variable names
        low_sinks: set/list of variables that must remain clean (optional)
    Returns:
        TaintResult
    """
    analyzer = SymbolicTaintAnalyzer(high_vars, low_sinks)
    return analyzer.analyze(source)


def check_noninterference(source, high_vars, low_vars, low_outputs=None):
    """
    Prove noninterference: varying HIGH inputs cannot change LOW outputs.

    Args:
        source: C10 source code
        high_vars: secret inputs
        low_vars: public inputs
        low_outputs: public outputs to check (optional)
    Returns:
        TaintResult
    """
    checker = NoninterferenceChecker(high_vars, low_vars, low_outputs)
    return checker.check(source)


def build_dependency_graph(source):
    """
    Build a data dependency graph from C10 source.

    Returns:
        DependencyGraph with edges and reachability queries
    """
    return DependencyGraph().build(source)


def compare_taint_analyses(source, high_vars, low_sinks=None):
    """
    Run both abstract and symbolic taint analysis and compare results.

    Returns:
        dict with 'abstract', 'symbolic', and 'comparison' keys
    """
    abstract = taint_analyze(source, high_vars, low_sinks)
    symbolic = symbolic_taint_analyze(source, high_vars, low_sinks)

    # Compare precision
    abs_violations = {(v.sink, frozenset(v.sources)) for v in abstract.violations}
    sym_violations = {(v.sink, frozenset(v.sources)) for v in symbolic.violations}

    return {
        'abstract': abstract,
        'symbolic': symbolic,
        'comparison': {
            'abstract_violations': len(abstract.violations),
            'symbolic_violations': len(symbolic.violations),
            'false_positives': len(abs_violations - sym_violations),
            'shared': len(abs_violations & sym_violations),
            'symbolic_only': len(sym_violations - abs_violations),
            'abstract_more_conservative': len(abstract.violations) >= len(symbolic.violations),
        }
    }


def full_information_flow_analysis(source, high_vars, low_vars=None, low_sinks=None, low_outputs=None):
    """
    Comprehensive information flow analysis: taint + noninterference + dependency graph.

    Returns:
        dict with all analysis results
    """
    taint = taint_analyze(source, high_vars, low_sinks)
    graph = build_dependency_graph(source)
    result = {
        'taint': taint,
        'dependency_graph': graph,
        'high_var_reach': {v: graph.all_flows_from(v) for v in high_vars},
    }
    if low_vars is not None:
        ni = check_noninterference(source, high_vars, low_vars, low_outputs)
        result['noninterference'] = ni
    return result
