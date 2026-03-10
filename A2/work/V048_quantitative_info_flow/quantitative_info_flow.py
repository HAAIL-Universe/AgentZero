"""
V048: Quantitative Information Flow Analysis

Composes V034 (taint analysis) + V011 (refinement types) + C037 (SMT solver)
to measure not just WHETHER information leaks, but HOW MUCH.

Key concepts:
- Security levels (HIGH/LOW) assigned to variables
- Channel capacity: how many bits an observer can learn about a secret
- Min-entropy leakage: worst-case information revealed per observation
- Quantitative non-interference: bound bits leaked via outputs

Approach:
- Parse Python source with ast module (like V034)
- Track security levels (HIGH=secret, LOW=public) per variable
- For each LOW output, compute how many distinct HIGH-dependent values it can take
- log2(distinct_values) = bits leaked
- Uses SMT (C037) to count/bound distinct output values
"""

import ast
import math
import sys
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, FrozenSet, Any, Union
)
from collections import defaultdict

# --- Path setup ---
_dir = os.path.dirname(os.path.abspath(__file__))
_work = os.path.dirname(_dir)
_a2 = os.path.dirname(_work)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, "challenges", "C037_smt_solver"))
sys.path.insert(0, os.path.join(_work, "V034_deep_taint_analysis"))
sys.path.insert(0, os.path.join(_work, "V011_refinement_types"))

from smt_solver import (
    SMTSolver, SMTResult, Var, App, Op, IntConst, BoolConst,
    Sort, SortKind,
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ============================================================
# Core Types
# ============================================================

class SecurityLevel(Enum):
    """Two-point security lattice: LOW (public) <= HIGH (secret)."""
    LOW = "low"
    HIGH = "high"

    def __le__(self, other):
        if self == SecurityLevel.LOW:
            return True
        return other == SecurityLevel.HIGH

    def __lt__(self, other):
        return self != other and self <= other

    def join(self, other):
        if self == SecurityLevel.HIGH or other == SecurityLevel.HIGH:
            return SecurityLevel.HIGH
        return SecurityLevel.LOW

    def meet(self, other):
        if self == SecurityLevel.LOW or other == SecurityLevel.LOW:
            return SecurityLevel.LOW
        return SecurityLevel.HIGH


@dataclass(frozen=True)
class InfoFlowLabel:
    """Tracks which secret variable influenced a value and how."""
    secret_var: str          # name of the HIGH variable
    flow_kind: str           # "direct" or "implicit"
    origin_line: int = 0     # where the flow originated


@dataclass
class FlowValue:
    """A value with security level and flow provenance."""
    level: SecurityLevel
    labels: FrozenSet[InfoFlowLabel] = field(default_factory=frozenset)

    @property
    def is_high(self):
        return self.level == SecurityLevel.HIGH

    def join(self, other: "FlowValue") -> "FlowValue":
        return FlowValue(
            level=self.level.join(other.level),
            labels=self.labels | other.labels,
        )

    @staticmethod
    def low():
        return FlowValue(SecurityLevel.LOW)

    @staticmethod
    def high(var: str, line: int = 0, kind: str = "direct"):
        return FlowValue(
            SecurityLevel.HIGH,
            frozenset([InfoFlowLabel(var, kind, line)]),
        )


@dataclass
class FlowEnv:
    """Environment mapping variables to their flow values."""
    bindings: Dict[str, FlowValue] = field(default_factory=dict)
    pc_level: SecurityLevel = SecurityLevel.LOW  # program counter level
    pc_labels: FrozenSet[InfoFlowLabel] = field(default_factory=frozenset)

    def get(self, name: str) -> FlowValue:
        return self.bindings.get(name, FlowValue.low())

    def set(self, name: str, value: FlowValue) -> "FlowEnv":
        new_bindings = dict(self.bindings)
        # Join with PC level for implicit flows
        if self.pc_level == SecurityLevel.HIGH:
            value = FlowValue(
                level=SecurityLevel.HIGH,
                labels=value.labels | self.pc_labels,
            )
        new_bindings[name] = value
        return FlowEnv(new_bindings, self.pc_level, self.pc_labels)

    def join(self, other: "FlowEnv") -> "FlowEnv":
        all_vars = set(self.bindings) | set(other.bindings)
        merged = {}
        for v in all_vars:
            merged[v] = self.get(v).join(other.get(v))
        return FlowEnv(
            merged,
            self.pc_level.join(other.pc_level),
            self.pc_labels | other.pc_labels,
        )

    def with_pc(self, level: SecurityLevel,
                labels: FrozenSet[InfoFlowLabel]) -> "FlowEnv":
        return FlowEnv(
            dict(self.bindings),
            self.pc_level.join(level),
            self.pc_labels | labels,
        )

    def copy(self) -> "FlowEnv":
        return FlowEnv(dict(self.bindings), self.pc_level, self.pc_labels)


# ============================================================
# Leakage Findings
# ============================================================

class LeakageKind(Enum):
    DIRECT_FLOW = "direct_flow"           # x_low = x_high
    IMPLICIT_FLOW = "implicit_flow"       # if(high) { low = ... }
    DECLASSIFICATION = "declassification" # intentional downgrade
    OBSERVABLE_OUTPUT = "observable_output" # print/return of HIGH


@dataclass
class LeakageFinding:
    """A quantified information leak."""
    kind: LeakageKind
    line: int
    variable: str                # the LOW variable receiving leaked info
    secret_sources: Set[str]     # HIGH variables that contribute
    bits_leaked: float           # upper bound on bits leaked (log2 of distinct vals)
    message: str
    channel: str = ""            # "output", "branch", "timing"
    col: int = 0


@dataclass
class ChannelCapacity:
    """Capacity of an information channel."""
    channel_name: str            # e.g., "return_value", "print_output"
    bits: float                  # max bits leaked through this channel
    secret_sources: Set[str]     # which secrets feed into it
    distinct_values: int         # number of distinct observable values
    line: int = 0


@dataclass
class QIFResult:
    """Result of quantitative information flow analysis."""
    findings: List[LeakageFinding] = field(default_factory=list)
    channels: List[ChannelCapacity] = field(default_factory=list)
    total_leakage_bits: float = 0.0
    high_vars: Set[str] = field(default_factory=set)
    low_vars: Set[str] = field(default_factory=set)
    function_leakages: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self):
        return self.total_leakage_bits == 0.0

    @property
    def max_channel_leakage(self) -> float:
        if not self.channels:
            return 0.0
        return max(c.bits for c in self.channels)

    def summary(self) -> str:
        lines = ["=== Quantitative Information Flow Analysis ==="]
        lines.append(f"HIGH variables: {self.high_vars}")
        lines.append(f"LOW variables: {self.low_vars}")
        lines.append(f"Total leakage: {self.total_leakage_bits:.2f} bits")
        lines.append(f"Channels: {len(self.channels)}")
        for ch in self.channels:
            lines.append(
                f"  {ch.channel_name}: {ch.bits:.2f} bits "
                f"({ch.distinct_values} distinct values) "
                f"from {ch.secret_sources}"
            )
        lines.append(f"Findings: {len(self.findings)}")
        for f in self.findings:
            lines.append(
                f"  L{f.line} [{f.kind.value}] {f.message} "
                f"({f.bits_leaked:.2f} bits)"
            )
        return "\n".join(lines)


# ============================================================
# SMT-Based Leakage Quantification
# ============================================================

class LeakageQuantifier:
    """Uses SMT to bound how many distinct LOW outputs are possible
    for different HIGH inputs, given a set of constraints."""

    def __init__(self):
        pass

    def count_distinct_outputs(
        self,
        high_vars: List[str],
        low_var: str,
        constraints: List[Any],
        high_domain: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Tuple[int, float]:
        """Count distinct values of low_var across all HIGH inputs.

        Returns (distinct_count, bits_leaked).
        Uses iterative SMT: find a value, exclude it, repeat.
        """
        if not high_vars:
            return 1, 0.0

        found_values = set()
        max_iterations = 32  # cap to avoid slow SMT enumeration

        for _ in range(max_iterations):
            solver = SMTSolver()
            out_var = solver.Int(low_var)

            # Declare high vars
            smt_highs = {}
            for h in high_vars:
                smt_highs[h] = solver.Int(h)

            # Apply domain constraints
            if high_domain:
                for h, (lo, hi) in high_domain.items():
                    if h in smt_highs:
                        v = smt_highs[h]
                        solver.add(App(Op.GE, [v, IntConst(lo)], BOOL))
                        solver.add(App(Op.LE, [v, IntConst(hi)], BOOL))

            # Apply program constraints
            for c in constraints:
                solver.add(c)

            # Exclude already-found values
            for val in found_values:
                solver.add(App(Op.NEQ, [out_var, IntConst(val)], BOOL))

            result = solver.check()
            if result != SMTResult.SAT:
                break
            model = solver.model()
            if low_var in model:
                found_values.add(model[low_var])
            else:
                break

        distinct = len(found_values) if found_values else 1
        bits = math.log2(distinct) if distinct > 1 else 0.0
        return distinct, bits

    def check_noninterference(
        self,
        high_vars: List[str],
        low_var: str,
        constraints_builder,
        high_domain: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Tuple[bool, Optional[Dict[str, int]]]:
        """Check if low_var is independent of high_vars.

        Uses the self-composition technique:
        Run program twice with same LOW inputs but different HIGH inputs.
        If LOW outputs can differ -> interference exists.

        Returns (is_noninterfering, counterexample).
        """
        solver = SMTSolver()

        # Two runs: variables suffixed with _1 and _2
        highs_1 = {h: solver.Int(h + "_1") for h in high_vars}
        highs_2 = {h: solver.Int(h + "_2") for h in high_vars}
        out_1 = solver.Int(low_var + "_1")
        out_2 = solver.Int(low_var + "_2")

        # HIGH inputs differ in at least one variable
        differ_clauses = []
        for h in high_vars:
            differ_clauses.append(
                App(Op.NEQ, [highs_1[h], highs_2[h]], BOOL)
            )
        if differ_clauses:
            # At least one must differ
            disj = differ_clauses[0]
            for d in differ_clauses[1:]:
                disj = App(Op.OR, [disj, d], BOOL)
            solver.add(disj)

        # Domain constraints
        if high_domain:
            for h, (lo, hi) in high_domain.items():
                for suffix, highs in [("_1", highs_1), ("_2", highs_2)]:
                    if h in highs:
                        v = highs[h]
                        solver.add(App(Op.GE, [v, IntConst(lo)], BOOL))
                        solver.add(App(Op.LE, [v, IntConst(hi)], BOOL))

        # Apply constraints for both runs
        c1 = constraints_builder("_1")
        c2 = constraints_builder("_2")
        for c in c1:
            solver.add(c)
        for c in c2:
            solver.add(c)

        # LOW outputs differ
        solver.add(App(Op.NEQ, [out_1, out_2], BOOL))

        result = solver.check()
        if result == SMTResult.SAT:
            model = solver.model()
            return False, model
        return True, None


# ============================================================
# Information Flow Analyzer (AST-based)
# ============================================================

class QIFAnalyzer(ast.NodeVisitor):
    """Analyzes Python source for quantitative information flow.

    Tracks security levels through the program and quantifies
    leakage at observable outputs (print, return, assignments to LOW vars).
    """

    def __init__(
        self,
        high_vars: Set[str],
        low_vars: Optional[Set[str]] = None,
        high_domain: Optional[Dict[str, Tuple[int, int]]] = None,
        track_implicit: bool = True,
    ):
        self.high_vars = high_vars
        self.low_vars = low_vars or set()
        self.high_domain = high_domain or {}
        self.track_implicit = track_implicit
        self.env = FlowEnv()
        self.findings: List[LeakageFinding] = []
        self.channels: List[ChannelCapacity] = []
        self.quantifier = LeakageQuantifier()
        self._output_exprs: List[Tuple[int, str, ast.expr]] = []  # (line, channel, expr)
        self._assignments: List[Tuple[int, str, ast.expr]] = []   # (line, var, expr)
        self._func_returns: Dict[str, List[Tuple[int, ast.expr]]] = defaultdict(list)
        self._current_func: Optional[str] = None
        self._branch_depth = 0

    def analyze(self, source: str) -> QIFResult:
        """Analyze source code for quantitative information flow."""
        tree = ast.parse(source)

        # Initialize HIGH variables
        for h in self.high_vars:
            self.env = self.env.set(
                h, FlowValue.high(h, 0, "source")
            )

        # Phase 1: Flow analysis (track levels through program)
        for stmt in tree.body:
            self._analyze_stmt(stmt)

        # Phase 2: Quantify leakage at each output/channel
        self._quantify_all(source)

        # Build result
        total_bits = sum(c.bits for c in self.channels)
        func_leakages = {}
        for f in self.findings:
            if f.variable not in func_leakages:
                func_leakages[f.variable] = 0.0
            func_leakages[f.variable] += f.bits_leaked

        return QIFResult(
            findings=self.findings,
            channels=self.channels,
            total_leakage_bits=total_bits,
            high_vars=set(self.high_vars),
            low_vars=set(self.low_vars),
            function_leakages=func_leakages,
        )

    def _analyze_stmt(self, node: ast.stmt):
        """Analyze a statement for information flow."""
        if isinstance(node, ast.Assign):
            self._analyze_assign(node)
        elif isinstance(node, ast.AugAssign):
            self._analyze_aug_assign(node)
        elif isinstance(node, ast.If):
            self._analyze_if(node)
        elif isinstance(node, ast.While):
            self._analyze_while(node)
        elif isinstance(node, ast.For):
            self._analyze_for(node)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Call):
                self._analyze_call_stmt(node.value, node.lineno)
        elif isinstance(node, ast.Return):
            self._analyze_return(node)
        elif isinstance(node, ast.FunctionDef):
            self._analyze_funcdef(node)
        elif isinstance(node, (ast.AnnAssign,)):
            if node.value is not None:
                val_flow = self._expr_flow(node.value)
                if node.target and isinstance(node.target, ast.Name):
                    self.env = self.env.set(node.target.id, val_flow)

    def _analyze_assign(self, node: ast.Assign):
        val_flow = self._expr_flow(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                # HIGH vars stay HIGH regardless of what is assigned
                if name in self.high_vars:
                    effective = val_flow.join(FlowValue.high(name, node.lineno, "source"))
                else:
                    effective = val_flow
                self.env = self.env.set(name, effective)
                # Check: assigning HIGH to a LOW variable
                # Use stored value (includes PC-level implicit flows)
                stored = self.env.get(name)
                if name in self.low_vars and stored.is_high:
                    sources = {l.secret_var for l in stored.labels}
                    self._record_direct_leak(
                        node.lineno, name, sources, node.value
                    )
                self._assignments.append((node.lineno, name, node.value))
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        ename = elt.id
                        if ename in self.high_vars:
                            eff = val_flow.join(FlowValue.high(ename, node.lineno, "source"))
                        else:
                            eff = val_flow
                        self.env = self.env.set(ename, eff)
                        if ename in self.low_vars and eff.is_high:
                            sources = {l.secret_var for l in eff.labels}
                            self._record_direct_leak(
                                node.lineno, ename, sources, node.value
                            )

    def _analyze_aug_assign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            name = node.target.id
            cur = self.env.get(name)
            val = self._expr_flow(node.value)
            combined = cur.join(val)
            self.env = self.env.set(name, combined)
            if name in self.low_vars and combined.is_high:
                sources = {l.secret_var for l in combined.labels}
                self._record_direct_leak(
                    node.lineno, name, sources, node.value
                )

    def _analyze_if(self, node: ast.If):
        cond_flow = self._expr_flow(node.test)
        if self.track_implicit and cond_flow.is_high:
            # HIGH condition -> implicit flow in both branches
            saved_env = self.env.copy()
            then_env = self.env.with_pc(
                SecurityLevel.HIGH, cond_flow.labels
            )
            self.env = then_env
            self._branch_depth += 1
            for s in node.body:
                self._analyze_stmt(s)
            then_result = self.env
            self._branch_depth -= 1

            self.env = saved_env.with_pc(
                SecurityLevel.HIGH, cond_flow.labels
            )
            self._branch_depth += 1
            for s in node.orelse:
                self._analyze_stmt(s)
            else_result = self.env
            self._branch_depth -= 1

            self.env = then_result.join(else_result)
        else:
            saved_env = self.env.copy()
            for s in node.body:
                self._analyze_stmt(s)
            then_result = self.env
            self.env = saved_env
            for s in node.orelse:
                self._analyze_stmt(s)
            else_result = self.env
            self.env = then_result.join(else_result)

    def _analyze_while(self, node: ast.While):
        cond_flow = self._expr_flow(node.test)
        # Fixed-point iteration (simplified: 2 iterations)
        for _ in range(2):
            if self.track_implicit and cond_flow.is_high:
                self.env = self.env.with_pc(
                    SecurityLevel.HIGH, cond_flow.labels
                )
            for s in node.body:
                self._analyze_stmt(s)
            cond_flow = self._expr_flow(node.test)

    def _analyze_for(self, node: ast.For):
        iter_flow = self._expr_flow(node.iter)
        if isinstance(node.target, ast.Name):
            self.env = self.env.set(node.target.id, iter_flow)
        for _ in range(2):
            if self.track_implicit and iter_flow.is_high:
                self.env = self.env.with_pc(
                    SecurityLevel.HIGH, iter_flow.labels
                )
            for s in node.body:
                self._analyze_stmt(s)

    def _analyze_call_stmt(self, node: ast.Call, line: int):
        """Handle print() and other observable outputs."""
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            for arg in node.args:
                flow = self._expr_flow(arg)
                if flow.is_high:
                    sources = {l.secret_var for l in flow.labels}
                    self._output_exprs.append(
                        (line, "print_output", arg)
                    )
                    self.findings.append(LeakageFinding(
                        kind=LeakageKind.OBSERVABLE_OUTPUT,
                        line=line,
                        variable="<print>",
                        secret_sources=sources,
                        bits_leaked=0.0,  # quantified later
                        message=f"print() outputs HIGH-dependent value from {sources}",
                        channel="output",
                    ))

    def _analyze_return(self, node: ast.Return):
        if node.value is not None:
            flow = self._expr_flow(node.value)
            if flow.is_high:
                sources = {l.secret_var for l in flow.labels}
                fname = self._current_func or "<module>"
                self._func_returns[fname].append(
                    (node.lineno, node.value)
                )
                self.findings.append(LeakageFinding(
                    kind=LeakageKind.OBSERVABLE_OUTPUT,
                    line=node.lineno,
                    variable=f"return({fname})",
                    secret_sources=sources,
                    bits_leaked=0.0,
                    message=f"Function {fname} returns HIGH-dependent value from {sources}",
                    channel="return",
                ))

    def _analyze_funcdef(self, node: ast.FunctionDef):
        saved_func = self._current_func
        self._current_func = node.name
        for s in node.body:
            self._analyze_stmt(s)
        self._current_func = saved_func

    def _expr_flow(self, node: ast.expr) -> FlowValue:
        """Compute the security level of an expression."""
        if isinstance(node, ast.Name):
            val = self.env.get(node.id)
            # HIGH vars are always HIGH even if env was cleared
            if node.id in self.high_vars and not val.is_high:
                return FlowValue.high(node.id, getattr(node, 'lineno', 0), "source")
            return val
        elif isinstance(node, (ast.Constant,)):
            return FlowValue.low()
        elif isinstance(node, ast.BinOp):
            left = self._expr_flow(node.left)
            right = self._expr_flow(node.right)
            return left.join(right)
        elif isinstance(node, ast.UnaryOp):
            return self._expr_flow(node.operand)
        elif isinstance(node, ast.BoolOp):
            result = FlowValue.low()
            for v in node.values:
                result = result.join(self._expr_flow(v))
            return result
        elif isinstance(node, ast.Compare):
            result = self._expr_flow(node.left)
            for comp in node.comparators:
                result = result.join(self._expr_flow(comp))
            return result
        elif isinstance(node, ast.IfExp):
            test = self._expr_flow(node.test)
            body = self._expr_flow(node.body)
            orelse = self._expr_flow(node.orelse)
            return test.join(body).join(orelse)
        elif isinstance(node, ast.Call):
            result = FlowValue.low()
            for arg in node.args:
                result = result.join(self._expr_flow(arg))
            if isinstance(node.func, ast.Name):
                result = result.join(self.env.get(node.func.id))
            return result
        elif isinstance(node, ast.Subscript):
            return self._expr_flow(node.value)
        elif isinstance(node, ast.Attribute):
            return self._expr_flow(node.value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            result = FlowValue.low()
            for elt in node.elts:
                result = result.join(self._expr_flow(elt))
            return result
        return FlowValue.low()

    def _record_direct_leak(
        self, line: int, var: str, sources: Set[str], expr: ast.expr
    ):
        self.findings.append(LeakageFinding(
            kind=LeakageKind.DIRECT_FLOW,
            line=line,
            variable=var,
            secret_sources=sources,
            bits_leaked=0.0,  # quantified later
            message=f"HIGH flows to LOW variable '{var}' from {sources}",
            channel="assignment",
        ))

    # --- Phase 2: Quantification ---

    def _quantify_all(self, source: str):
        """Quantify leakage at each detected leak point using SMT."""
        # Quantify output channels
        for line, channel, expr in self._output_exprs:
            bits, distinct = self._quantify_expr(expr, source)
            self.channels.append(ChannelCapacity(
                channel_name=f"{channel}@L{line}",
                bits=bits,
                secret_sources=self._expr_sources(expr),
                distinct_values=distinct,
                line=line,
            ))
            # Update the finding
            for f in self.findings:
                if f.line == line and f.bits_leaked == 0.0:
                    f.bits_leaked = bits
                    break

        # Quantify assignment leaks (direct flows)
        seen_vars = set()
        for f in self.findings:
            if f.kind == LeakageKind.DIRECT_FLOW and f.bits_leaked == 0.0:
                key = (f.line, f.variable)
                if key in seen_vars:
                    continue
                seen_vars.add(key)
                # Find the corresponding assignment expr
                for aline, avar, aexpr in self._assignments:
                    if aline == f.line and avar == f.variable:
                        bits, distinct = self._quantify_expr(aexpr, source)
                        f.bits_leaked = bits
                        if bits > 0:
                            self.channels.append(ChannelCapacity(
                                channel_name=f"assign_{f.variable}@L{f.line}",
                                bits=bits,
                                secret_sources=f.secret_sources,
                                distinct_values=distinct,
                                line=f.line,
                            ))
                        break

        # Quantify return channels
        for f in self.findings:
            if f.kind == LeakageKind.OBSERVABLE_OUTPUT and f.channel == "return" and f.bits_leaked == 0.0:
                fname = f.variable.replace("return(", "").rstrip(")")
                for rline, rexpr in self._func_returns.get(fname, []):
                    if rline == f.line:
                        bits, distinct = self._quantify_expr(rexpr, source)
                        f.bits_leaked = bits
                        self.channels.append(ChannelCapacity(
                            channel_name=f"return_{fname}@L{f.line}",
                            bits=bits,
                            secret_sources=f.secret_sources,
                            distinct_values=distinct,
                            line=f.line,
                        ))
                        break

    def _quantify_expr(
        self, expr: ast.expr, source: str
    ) -> Tuple[float, int]:
        """Quantify how many distinct values expr can take over HIGH inputs.

        Uses lightweight symbolic evaluation:
        - Constants -> 1 distinct value
        - HIGH variable -> domain size
        - Arithmetic on HIGH -> SMT counting or structural analysis
        """
        involved_highs = self._expr_high_vars(expr)
        if not involved_highs:
            return 0.0, 1

        # Structural quantification (faster than SMT for simple cases)
        distinct = self._structural_count(expr, involved_highs)
        if distinct is not None:
            bits = math.log2(distinct) if distinct > 1 else 0.0
            return bits, distinct

        # SMT-based counting for complex expressions
        constraints = self._expr_to_smt_constraints(expr, "out")
        if constraints is not None:
            distinct, bits = self.quantifier.count_distinct_outputs(
                list(involved_highs), "out", constraints, self.high_domain
            )
            return bits, distinct

        # Fallback: assume full domain leaked
        total = 1
        for h in involved_highs:
            if h in self.high_domain:
                lo, hi = self.high_domain[h]
                total *= (hi - lo + 1)
            else:
                total = 256  # conservative default
        bits = math.log2(total) if total > 1 else 0.0
        return bits, total

    def _structural_count(
        self, expr: ast.expr, highs: Set[str]
    ) -> Optional[int]:
        """Structurally determine distinct output count for simple expressions."""
        if isinstance(expr, ast.Name):
            if expr.id in highs and expr.id in self.high_domain:
                lo, hi = self.high_domain[expr.id]
                return hi - lo + 1
            return None

        if isinstance(expr, ast.Constant):
            return 1

        if isinstance(expr, ast.BinOp):
            if isinstance(expr.op, ast.Mod):
                # x % k has at most k distinct values
                if isinstance(expr.right, ast.Constant) and isinstance(expr.right.value, int):
                    k = abs(expr.right.value)
                    return k if k > 0 else None
            if isinstance(expr.op, ast.FloorDiv):
                # x // k reduces domain by factor k
                if isinstance(expr.right, ast.Constant) and isinstance(expr.right.value, int):
                    k = abs(expr.right.value)
                    if k > 0:
                        left_count = self._structural_count(expr.left, highs)
                        if left_count is not None:
                            return max(1, (left_count + k - 1) // k)
            if isinstance(expr.op, ast.BitAnd):
                # x & mask has at most 2^popcount(mask) values
                if isinstance(expr.right, ast.Constant) and isinstance(expr.right.value, int):
                    mask = expr.right.value
                    return 1 << bin(mask).count('1')

        if isinstance(expr, ast.UnaryOp):
            # Unary ops preserve count (negation, not, etc.)
            return self._structural_count(expr.operand, highs)

        if isinstance(expr, ast.Compare):
            # Comparison produces bool: 2 distinct values max
            return 2

        if isinstance(expr, ast.IfExp):
            # Ternary: distinct values = union of body + orelse
            body_count = self._structural_count(expr.body, highs)
            else_count = self._structural_count(expr.orelse, highs)
            if body_count is not None and else_count is not None:
                return body_count + else_count  # upper bound

        return None

    def _expr_to_smt_constraints(
        self, expr: ast.expr, out_name: str
    ) -> Optional[List]:
        """Convert a Python expression to SMT constraints: out_name == expr.

        Returns None if the expression can't be translated.
        """
        try:
            smt_expr = self._ast_to_smt(expr)
            if smt_expr is None:
                return None
            out_var = Var(out_name, INT)
            return [App(Op.EQ, [out_var, smt_expr], BOOL)]
        except Exception:
            return None

    def _ast_to_smt(self, node: ast.expr) -> Optional[Any]:
        """Convert AST expression to SMT term."""
        if isinstance(node, ast.Name):
            return Var(node.id, INT)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return IntConst(node.value)
        if isinstance(node, ast.BinOp):
            left = self._ast_to_smt(node.left)
            right = self._ast_to_smt(node.right)
            if left is None or right is None:
                return None
            op_map = {
                ast.Add: Op.ADD,
                ast.Sub: Op.SUB,
                ast.Mult: Op.MUL,
            }
            smt_op = op_map.get(type(node.op))
            if smt_op is None:
                return None
            return App(smt_op, [left, right], INT)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._ast_to_smt(node.operand)
            if operand is None:
                return None
            return App(Op.SUB, [IntConst(0), operand], INT)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = self._ast_to_smt(node.left)
            right = self._ast_to_smt(node.comparators[0])
            if left is None or right is None:
                return None
            cmp_map = {
                ast.Eq: Op.EQ,
                ast.NotEq: Op.NEQ,
                ast.Lt: Op.LT,
                ast.LtE: Op.LE,
                ast.Gt: Op.GT,
                ast.GtE: Op.GE,
            }
            smt_op = cmp_map.get(type(node.ops[0]))
            if smt_op is None:
                return None
            cmp = App(smt_op, [left, right], BOOL)
            return App(Op.ITE, [cmp, IntConst(1), IntConst(0)], INT)
        if isinstance(node, ast.IfExp):
            test = self._ast_to_smt(node.test)
            body = self._ast_to_smt(node.body)
            orelse = self._ast_to_smt(node.orelse)
            if test is None or body is None or orelse is None:
                return None
            # test might be a comparison (BOOL) or variable (need != 0 check)
            if not self._is_bool_expr(node.test):
                test = App(Op.NEQ, [test, IntConst(0)], BOOL)
            return App(Op.ITE, [test, body, orelse], INT)
        return None

    def _is_bool_expr(self, node: ast.expr) -> bool:
        """Check if a Python expression naturally produces a boolean."""
        return isinstance(node, (ast.Compare, ast.BoolOp))

    def _expr_high_vars(self, node: ast.expr) -> Set[str]:
        """Collect all HIGH variables referenced in an expression."""
        result = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.high_vars:
                result.add(child.id)
        return result

    def _expr_sources(self, node: ast.expr) -> Set[str]:
        """Get the set of HIGH source variables for an expression."""
        return self._expr_high_vars(node)


# ============================================================
# Declassification Support
# ============================================================

class DeclassificationPolicy:
    """Policy for intentional information release (declassification).

    Allows controlled downgrading of security levels through
    explicit declassification annotations.
    """

    def __init__(self):
        self.allowed: List[Tuple[str, str, float]] = []
        # (secret_var, context, max_bits)

    def allow(self, secret: str, context: str = "*", max_bits: float = float('inf')):
        """Allow declassification of secret in context up to max_bits."""
        self.allowed.append((secret, context, max_bits))

    def is_allowed(self, secret: str, context: str, bits: float) -> bool:
        for s, c, mb in self.allowed:
            if (s == secret or s == "*") and (c == context or c == "*"):
                if bits <= mb:
                    return True
        return False


# ============================================================
# Convenience API
# ============================================================

def analyze_qif(
    source: str,
    high_vars: Set[str],
    low_vars: Optional[Set[str]] = None,
    high_domain: Optional[Dict[str, Tuple[int, int]]] = None,
    track_implicit: bool = True,
) -> QIFResult:
    """Analyze quantitative information flow in Python source.

    Args:
        source: Python source code
        high_vars: Set of HIGH (secret) variable names
        low_vars: Set of LOW (public) variable names (for detecting flows)
        high_domain: Optional bounds on HIGH variables {name: (lo, hi)}
        track_implicit: Whether to track implicit flows (branches on HIGH)

    Returns:
        QIFResult with findings, channels, and total leakage
    """
    analyzer = QIFAnalyzer(
        high_vars=high_vars,
        low_vars=low_vars,
        high_domain=high_domain,
        track_implicit=track_implicit,
    )
    return analyzer.analyze(source)


def check_noninterference(
    source: str,
    high_vars: Set[str],
    low_var: str,
    high_domain: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[bool, Optional[Dict[str, int]]]:
    """Check if low_var is independent of high_vars in the given program.

    Uses self-composition: run program twice with different HIGH inputs,
    check if LOW output can differ.

    Returns (is_noninterfering, counterexample).
    """
    # First do flow analysis to get the expression
    result = analyze_qif(source, high_vars, {low_var}, high_domain)
    # If no HIGH flows to the variable, it's noninterfering
    has_leak = any(
        f.variable == low_var or f.variable == "<print>"
        for f in result.findings
    )
    if not has_leak:
        return True, None

    # Use the quantifier for self-composition check
    q = LeakageQuantifier()

    # Simple structural approach: parse and extract the assignment
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == low_var:
                    constraints = _build_self_composition_constraints(
                        node.value, high_vars, low_var
                    )
                    if constraints is not None:
                        return constraints

    # Fallback: report based on flow analysis
    return not has_leak, None


def _build_self_composition_constraints(
    expr: ast.expr,
    high_vars: Set[str],
    low_var: str,
) -> Optional[Tuple[bool, Optional[Dict[str, int]]]]:
    """Build self-composition check for an expression."""
    solver = SMTSolver()
    analyzer = QIFAnalyzer(high_vars, set())

    # Two copies of high vars
    smt1 = {}
    smt2 = {}
    for h in high_vars:
        smt1[h] = solver.Int(h + "_1")
        smt2[h] = solver.Int(h + "_2")

    # At least one high var differs
    differ = []
    for h in high_vars:
        differ.append(App(Op.NEQ, [smt1[h], smt2[h]], BOOL))
    if differ:
        disj = differ[0]
        for d in differ[1:]:
            disj = App(Op.OR, [disj, d], BOOL)
        solver.add(disj)

    # Convert expr to SMT for both copies
    def ast_to_smt_with_suffix(node, suffix, var_map):
        if isinstance(node, ast.Name):
            if node.id in var_map:
                return var_map[node.id]
            return Var(node.id + suffix, INT)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return IntConst(node.value)
        if isinstance(node, ast.BinOp):
            left = ast_to_smt_with_suffix(node.left, suffix, var_map)
            right = ast_to_smt_with_suffix(node.right, suffix, var_map)
            if left is None or right is None:
                return None
            op_map = {ast.Add: Op.ADD, ast.Sub: Op.SUB, ast.Mult: Op.MUL}
            smt_op = op_map.get(type(node.op))
            if smt_op is None:
                return None
            return App(smt_op, [left, right], INT)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = ast_to_smt_with_suffix(node.operand, suffix, var_map)
            return App(Op.SUB, [IntConst(0), operand], INT) if operand else None
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = ast_to_smt_with_suffix(node.left, suffix, var_map)
            right = ast_to_smt_with_suffix(node.comparators[0], suffix, var_map)
            if left is None or right is None:
                return None
            cmp_map = {
                ast.Eq: Op.EQ, ast.NotEq: Op.NEQ,
                ast.Lt: Op.LT, ast.LtE: Op.LE,
                ast.Gt: Op.GT, ast.GtE: Op.GE,
            }
            smt_op = cmp_map.get(type(node.ops[0]))
            if smt_op is None:
                return None
            cmp = App(smt_op, [left, right], BOOL)
            return App(Op.ITE, [cmp, IntConst(1), IntConst(0)], INT)
        if isinstance(node, ast.IfExp):
            test = ast_to_smt_with_suffix(node.test, suffix, var_map)
            body = ast_to_smt_with_suffix(node.body, suffix, var_map)
            orelse = ast_to_smt_with_suffix(node.orelse, suffix, var_map)
            if test is None or body is None or orelse is None:
                return None
            if not isinstance(node.test, (ast.Compare, ast.BoolOp)):
                test = App(Op.NEQ, [test, IntConst(0)], BOOL)
            return App(Op.ITE, [test, body, orelse], INT)
        return None

    e1 = ast_to_smt_with_suffix(expr, "_1", smt1)
    e2 = ast_to_smt_with_suffix(expr, "_2", smt2)
    if e1 is None or e2 is None:
        return None

    out1 = solver.Int(low_var + "_1")
    out2 = solver.Int(low_var + "_2")
    solver.add(App(Op.EQ, [out1, e1], BOOL))
    solver.add(App(Op.EQ, [out2, e2], BOOL))
    solver.add(App(Op.NEQ, [out1, out2], BOOL))

    result = solver.check()
    if result == SMTResult.SAT:
        return False, solver.model()
    return True, None


def min_entropy_leakage(
    source: str,
    high_vars: Set[str],
    low_var: str,
    high_domain: Dict[str, Tuple[int, int]],
) -> float:
    """Compute min-entropy leakage: log2(max_prob_guess_after / max_prob_guess_before).

    Min-entropy measures the worst-case vulnerability: how much easier
    it is to guess the secret after observing the output.

    For deterministic programs with uniform HIGH inputs:
      leakage = log2(|HIGH_domain|) - log2(max partition size)
    where partitions are equivalence classes of HIGH inputs producing the same output.
    """
    result = analyze_qif(source, high_vars, {low_var}, high_domain)

    # Find the channel for this variable
    for ch in result.channels:
        if low_var in ch.channel_name or ch.channel_name.startswith(f"assign_{low_var}"):
            if ch.distinct_values <= 1:
                return 0.0
            # For uniform input: min-entropy leakage = log2(distinct_values)
            return ch.bits

    return 0.0


def channel_capacity(
    source: str,
    high_vars: Set[str],
    output_var: str,
    high_domain: Dict[str, Tuple[int, int]],
) -> float:
    """Compute channel capacity: max mutual information over all input distributions.

    For deterministic programs:
      capacity = log2(number of distinct outputs)
    """
    result = analyze_qif(source, high_vars, {output_var}, high_domain)
    for ch in result.channels:
        if output_var in ch.channel_name:
            return ch.bits
    return 0.0
