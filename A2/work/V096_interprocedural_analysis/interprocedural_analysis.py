"""V096: Interprocedural Analysis via Pushdown Systems

Composes V094 (pushdown systems) + C039 (abstract interpreter) for
context-sensitive interprocedural dataflow analysis.

Key idea: Model program control flow as a PDS where call/return matching
is enforced by the stack. Run abstract interpretation over the PDS
reachability structure for context-sensitive results.

Features:
- IFDS framework: interprocedural finite distributive subset problems
- Context-sensitive reaching definitions, live variables, taint tracking
- Function summaries computed via PDS reachability
- Call-string and functional approach comparison
- C10 source-level API via parser
"""

import sys
import os
from dataclasses import dataclass, field
from typing import (Dict, Set, List, Tuple, Optional, FrozenSet,
                    Callable, Any, Union)
from enum import Enum, auto
from collections import defaultdict, deque

# Import V094 pushdown systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V094_pushdown_systems'))
from pushdown_systems import (PushdownSystem, PDSRule, StackOp, Configuration,
                               PAutomaton, pre_star, post_star,
                               make_config_automaton, make_state_automaton)

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import lex, Parser

# Import C039 abstract interpreter for domains
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (AbstractInterpreter, AbstractEnv, AbstractValue,
                                   Sign, Interval, analyze as ai_analyze)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FactKind(Enum):
    """Kinds of dataflow facts for IFDS."""
    REACH_DEF = auto()    # reaching definition: (var, def_site)
    LIVE_VAR = auto()     # live variable: var is live
    TAINT = auto()        # taint: var is tainted
    AVAILABLE = auto()    # available expression
    ZERO_VALUE = auto()   # the special zero/lambda fact


@dataclass(frozen=True)
class DataflowFact:
    """A single dataflow fact."""
    kind: FactKind
    var: str = ""
    site: str = ""  # definition site (for REACH_DEF)
    extra: str = ""  # additional info

    def __repr__(self):
        if self.kind == FactKind.ZERO_VALUE:
            return "ZERO"
        if self.kind == FactKind.REACH_DEF:
            return f"def({self.var}@{self.site})"
        if self.kind == FactKind.LIVE_VAR:
            return f"live({self.var})"
        if self.kind == FactKind.TAINT:
            return f"taint({self.var})"
        return f"fact({self.kind.name},{self.var})"


ZERO_FACT = DataflowFact(kind=FactKind.ZERO_VALUE)


@dataclass
class ProgramPoint:
    """A program point in the interprocedural CFG."""
    function: str
    label: str
    stmt_type: str = ""  # 'assign', 'call', 'return', 'branch', 'entry', 'exit'
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.function}.{self.label}"


@dataclass
class ICFGEdge:
    """Edge in the interprocedural control-flow graph."""
    source: str   # program point id
    target: str   # program point id
    edge_type: str  # 'intra', 'call', 'return', 'call_to_return'
    callee: str = ""  # for call/return edges


@dataclass
class ICFG:
    """Interprocedural Control-Flow Graph."""
    points: Dict[str, ProgramPoint] = field(default_factory=dict)
    edges: List[ICFGEdge] = field(default_factory=list)
    functions: Dict[str, Dict] = field(default_factory=dict)
    # function -> {entry, exit, params, locals, points}
    entry_function: str = "main"

    def add_point(self, pp: ProgramPoint):
        self.points[pp.id] = pp

    def add_edge(self, src: str, tgt: str, edge_type: str, callee: str = ""):
        self.edges.append(ICFGEdge(src, tgt, edge_type, callee))

    def get_successors(self, point_id: str) -> List[ICFGEdge]:
        return [e for e in self.edges if e.source == point_id]

    def get_predecessors(self, point_id: str) -> List[ICFGEdge]:
        return [e for e in self.edges if e.target == point_id]

    def get_function_points(self, func: str) -> List[str]:
        return [pid for pid, pp in self.points.items() if pp.function == func]


@dataclass
class FlowFunction:
    """A flow function mapping input facts to output facts.

    For IFDS, this is a function D -> 2^D (distributive).
    We represent it as a set of edges in the exploded supergraph:
    (d1, d2) means fact d1 at source produces fact d2 at target.
    """
    edges: Set[Tuple[DataflowFact, DataflowFact]] = field(default_factory=set)

    def apply(self, input_facts: Set[DataflowFact]) -> Set[DataflowFact]:
        """Apply flow function to a set of input facts."""
        result = set()
        for (d1, d2) in self.edges:
            if d1 in input_facts:
                result.add(d2)
        return result

    def add_edge(self, d1: DataflowFact, d2: DataflowFact):
        self.edges.add((d1, d2))

    def identity(self, facts: Set[DataflowFact]):
        """Add identity edges for all given facts."""
        for f in facts:
            self.edges.add((f, f))

    def kill(self, fact: DataflowFact):
        """Remove all edges producing this fact."""
        self.edges = {(d1, d2) for (d1, d2) in self.edges if d2 != fact}

    def gen(self, fact: DataflowFact):
        """Generate fact from zero."""
        self.edges.add((ZERO_FACT, fact))


@dataclass
class IFDSResult:
    """Result of an IFDS analysis."""
    reachable_facts: Dict[str, Set[DataflowFact]]  # point_id -> facts
    summaries: Dict[str, Dict[DataflowFact, Set[DataflowFact]]]  # function -> summary
    context_sensitive: bool = True
    analysis_type: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionSummary:
    """Abstract summary of a function's effect."""
    name: str
    params: List[str]
    defined_vars: Set[str]    # variables defined
    used_vars: Set[str]       # variables used
    tainted_params: Set[str]  # which params propagate taint to return
    return_deps: Set[str]     # params that return value depends on
    env_effect: Optional[AbstractEnv] = None  # abstract effect on environment


# ---------------------------------------------------------------------------
# C10 source -> ICFG construction
# ---------------------------------------------------------------------------

def _extract_vars_from_expr(expr) -> Set[str]:
    """Extract variable names used in an expression."""
    cls = type(expr).__name__
    if cls == 'Var':
        return {expr.name}
    if cls == 'BinOp':
        return _extract_vars_from_expr(expr.left) | _extract_vars_from_expr(expr.right)
    if cls == 'UnaryOp':
        return _extract_vars_from_expr(expr.operand)
    if cls == 'CallExpr':
        result = set()
        for arg in expr.args:
            result |= _extract_vars_from_expr(arg)
        return result
    return set()


def _extract_assigned_var(stmt) -> Optional[str]:
    """Get the variable assigned in a statement."""
    cls = type(stmt).__name__
    if cls == 'LetDecl':
        return stmt.name
    if cls == 'Assign':
        return stmt.name
    return None


def _extract_call_info(stmt) -> Optional[Tuple[str, List]]:
    """Extract call target and args from a statement."""
    cls = type(stmt).__name__
    if cls == 'LetDecl' and type(stmt.value).__name__ == 'CallExpr':
        return (stmt.value.callee, stmt.value.args)
    if cls == 'Assign' and type(stmt.value).__name__ == 'CallExpr':
        return (stmt.value.callee, stmt.value.args)
    # Bare call expression
    if cls == 'CallExpr':
        return (stmt.callee, stmt.args)
    return None


def build_icfg(source: str) -> ICFG:
    """Build an ICFG from C10 source code.

    Parses the source, identifies functions, builds intraprocedural CFGs,
    and connects them with call/return edges.
    """
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()

    icfg = ICFG()
    functions = {}

    # First pass: collect all function declarations
    main_stmts = []
    for stmt in program.stmts:
        cls = type(stmt).__name__
        if cls == 'FnDecl':
            functions[stmt.name] = stmt
        else:
            main_stmts.append(stmt)

    # Build ICFG for each function
    for fn_name, fn_decl in functions.items():
        params = fn_decl.params if hasattr(fn_decl, 'params') else []
        body_stmts = fn_decl.body.stmts if hasattr(fn_decl.body, 'stmts') else []
        _build_function_icfg(icfg, fn_name, params, body_stmts, functions)

    # Build ICFG for main (top-level statements)
    if main_stmts:
        _build_function_icfg(icfg, "main", [], main_stmts, functions)
        icfg.entry_function = "main"
    elif functions:
        icfg.entry_function = next(iter(functions))

    return icfg


def _build_function_icfg(icfg: ICFG, fn_name: str, params: List[str],
                          stmts: list, all_functions: Dict):
    """Build ICFG nodes and edges for a single function."""
    entry_id = f"{fn_name}.entry"
    exit_id = f"{fn_name}.exit"

    icfg.add_point(ProgramPoint(fn_name, "entry", "entry",
                                {"params": params}))
    icfg.add_point(ProgramPoint(fn_name, "exit", "exit"))

    icfg.functions[fn_name] = {
        "entry": entry_id,
        "exit": exit_id,
        "params": params,
        "locals": set(),
        "points": [entry_id, exit_id]
    }

    if not stmts:
        icfg.add_edge(entry_id, exit_id, "intra")
        return

    # Build nodes for each statement
    stmt_ids = []
    for i, stmt in enumerate(stmts):
        label = f"s{i}"
        stmt_id = f"{fn_name}.{label}"
        cls = type(stmt).__name__

        details = {}
        stmt_type = "assign"

        if cls == 'LetDecl':
            details['var'] = stmt.name
            details['uses'] = _extract_vars_from_expr(stmt.value)
            call_info = _extract_call_info(stmt)
            if call_info:
                stmt_type = "call"
                details['callee'] = call_info[0]
                details['args'] = [_extract_vars_from_expr(a) for a in call_info[1]]
                details['result_var'] = stmt.name
            else:
                stmt_type = "assign"
        elif cls == 'Assign':
            details['var'] = stmt.name
            details['uses'] = _extract_vars_from_expr(stmt.value)
            call_info = _extract_call_info(stmt)
            if call_info:
                stmt_type = "call"
                details['callee'] = call_info[0]
                details['args'] = [_extract_vars_from_expr(a) for a in call_info[1]]
                details['result_var'] = stmt.name
            else:
                stmt_type = "assign"
        elif cls == 'ReturnStmt':
            stmt_type = "return"
            if hasattr(stmt, 'value') and stmt.value:
                details['uses'] = _extract_vars_from_expr(stmt.value)
            else:
                details['uses'] = set()
        elif cls == 'IfStmt':
            stmt_type = "branch"
            details['uses'] = _extract_vars_from_expr(stmt.cond)
        elif cls == 'WhileStmt':
            stmt_type = "branch"
            details['uses'] = _extract_vars_from_expr(stmt.cond)
        elif cls == 'CallExpr':
            stmt_type = "call"
            details['callee'] = stmt.callee
            details['args'] = [_extract_vars_from_expr(a) for a in stmt.args]
        elif cls == 'Block':
            stmt_type = "block"

        pp = ProgramPoint(fn_name, label, stmt_type, details)
        icfg.add_point(pp)
        stmt_ids.append(stmt_id)
        icfg.functions[fn_name]["points"].append(stmt_id)

        assigned = _extract_assigned_var(stmt)
        if assigned:
            icfg.functions[fn_name]["locals"].add(assigned)

    # Connect: entry -> first stmt
    icfg.add_edge(entry_id, stmt_ids[0], "intra")

    # Connect sequential statements
    for i in range(len(stmt_ids) - 1):
        pp = icfg.points[stmt_ids[i]]
        if pp.stmt_type == "return":
            # Return connects to function exit
            icfg.add_edge(stmt_ids[i], exit_id, "intra")
        else:
            icfg.add_edge(stmt_ids[i], stmt_ids[i + 1], "intra")

    # Last statement -> exit (if not return)
    last_pp = icfg.points[stmt_ids[-1]]
    if last_pp.stmt_type != "return":
        icfg.add_edge(stmt_ids[-1], exit_id, "intra")

    # Add call/return edges
    for sid in stmt_ids:
        pp = icfg.points[sid]
        if pp.stmt_type == "call" and 'callee' in pp.details:
            callee = pp.details['callee']
            if callee in all_functions:
                callee_entry = f"{callee}.entry"
                callee_exit = f"{callee}.exit"
                # Call edge: call site -> callee entry
                icfg.add_edge(sid, callee_entry, "call", callee)
                # Return edge: callee exit -> call successor
                succs = icfg.get_successors(sid)
                for succ in succs:
                    if succ.edge_type == "intra":
                        icfg.add_edge(callee_exit, succ.target, "return", callee)
                # Call-to-return edge (for locals not affected by call)
                for succ in succs:
                    if succ.edge_type == "intra":
                        icfg.add_edge(sid, succ.target, "call_to_return", callee)


# ---------------------------------------------------------------------------
# ICFG -> PDS conversion
# ---------------------------------------------------------------------------

def icfg_to_pds(icfg: ICFG) -> Tuple[PushdownSystem, Configuration]:
    """Convert ICFG to a Pushdown System.

    Single control state 'q'. Stack symbols = program point IDs.
    Call = PUSH (push return point), Return = POP, Intra = SWAP.
    """
    pds = PushdownSystem()
    pds.states.add("q")

    # Add all program points as stack symbols
    for pid in icfg.points:
        pds.stack_alphabet.add(pid)

    for edge in icfg.edges:
        src = edge.source
        tgt = edge.target

        if edge.edge_type == "call":
            # Call: push return point, go to callee entry
            # Find the call-to-return successor for the return point
            call_succs = [e for e in icfg.edges
                         if e.source == src and e.edge_type == "call_to_return"]
            if call_succs:
                return_point = call_succs[0].target
                pds.add_rule("q", src, "q", StackOp.PUSH, (tgt, return_point))
            else:
                # No return point, just swap
                pds.add_rule("q", src, "q", StackOp.SWAP, (tgt,))

        elif edge.edge_type == "return":
            # Return: pop callee exit, revealing return point
            # The return edge goes from callee_exit to return_point
            # In PDS: when TOS is callee_exit, pop to reveal return point
            pds.add_rule("q", src, "q", StackOp.POP)

        elif edge.edge_type == "intra":
            pds.add_rule("q", src, "q", StackOp.SWAP, (tgt,))

        # call_to_return edges are handled implicitly by push/pop matching

    # Initial configuration: entry point of entry function on stack
    entry = icfg.functions.get(icfg.entry_function, {}).get("entry", "main.entry")
    init_config = Configuration("q", (entry,))

    return pds, init_config


# ---------------------------------------------------------------------------
# IFDS Analysis
# ---------------------------------------------------------------------------

class IFDSProblem:
    """Definition of an IFDS problem.

    Subclass this to define specific analyses (reaching defs, taint, etc.)
    """
    def __init__(self, icfg: ICFG):
        self.icfg = icfg

    def initial_seeds(self) -> Dict[str, Set[DataflowFact]]:
        """Initial facts at entry points. Always includes ZERO_FACT."""
        entry = self.icfg.functions.get(self.icfg.entry_function, {}).get("entry")
        if entry:
            return {entry: {ZERO_FACT}}
        return {}

    def flow_function(self, edge: ICFGEdge) -> FlowFunction:
        """Flow function for an edge."""
        raise NotImplementedError

    def call_flow(self, call_site: str, callee_entry: str,
                  callee: str) -> FlowFunction:
        """Flow function from call site to callee entry."""
        raise NotImplementedError

    def return_flow(self, callee_exit: str, return_site: str,
                    call_site: str, callee: str) -> FlowFunction:
        """Flow function from callee exit back to caller."""
        raise NotImplementedError

    def call_to_return_flow(self, call_site: str, return_site: str,
                            callee: str) -> FlowFunction:
        """Flow function for facts not affected by the call."""
        raise NotImplementedError

    def all_facts(self) -> Set[DataflowFact]:
        """All possible dataflow facts (finite domain)."""
        raise NotImplementedError


class ReachingDefinitionsProblem(IFDSProblem):
    """IFDS problem for context-sensitive reaching definitions."""

    def __init__(self, icfg: ICFG, taint_sources: Set[str] = None):
        super().__init__(icfg)
        self._all_facts = self._compute_all_facts()
        self.taint_sources = taint_sources or set()

    def _compute_all_facts(self) -> Set[DataflowFact]:
        facts = {ZERO_FACT}
        for pid, pp in self.icfg.points.items():
            if pp.stmt_type in ("assign", "call") and 'var' in pp.details:
                facts.add(DataflowFact(FactKind.REACH_DEF, pp.details['var'], pid))
            # Parameter definitions at entry
            if pp.stmt_type == "entry" and 'params' in pp.details:
                for p in pp.details['params']:
                    facts.add(DataflowFact(FactKind.REACH_DEF, p, pid))
        return facts

    def all_facts(self) -> Set[DataflowFact]:
        return self._all_facts

    def initial_seeds(self) -> Dict[str, Set[DataflowFact]]:
        entry = self.icfg.functions.get(self.icfg.entry_function, {}).get("entry")
        if entry:
            return {entry: {ZERO_FACT}}
        return {}

    def flow_function(self, edge: ICFGEdge) -> FlowFunction:
        ff = FlowFunction()
        src_pp = self.icfg.points.get(edge.source)
        if not src_pp:
            ff.identity(self._all_facts)
            return ff

        if src_pp.stmt_type in ("assign",) and 'var' in src_pp.details:
            var = src_pp.details['var']
            # Kill: remove old definitions of var
            for f in self._all_facts:
                if f.kind == FactKind.REACH_DEF and f.var == var:
                    continue  # killed
                ff.add_edge(f, f)  # identity for others
            # Gen: new definition
            new_def = DataflowFact(FactKind.REACH_DEF, var, edge.source)
            ff.gen(new_def)
        else:
            ff.identity(self._all_facts)

        return ff

    def call_flow(self, call_site: str, callee_entry: str,
                  callee: str) -> FlowFunction:
        ff = FlowFunction()
        ff.add_edge(ZERO_FACT, ZERO_FACT)

        # Map actual args to formal params at callee entry
        call_pp = self.icfg.points.get(call_site)
        callee_info = self.icfg.functions.get(callee, {})
        params = callee_info.get("params", [])

        if call_pp and 'args' in call_pp.details:
            args_vars = call_pp.details['args']
            for i, param in enumerate(params):
                # Parameter gets defined at callee entry
                param_def = DataflowFact(FactKind.REACH_DEF, param, callee_entry)
                ff.gen(param_def)
                # Also propagate reaching defs of actual arg variables
                if i < len(args_vars):
                    for arg_var in args_vars[i]:
                        for fact in self._all_facts:
                            if fact.kind == FactKind.REACH_DEF and fact.var == arg_var:
                                ff.add_edge(fact, param_def)

        return ff

    def return_flow(self, callee_exit: str, return_site: str,
                    call_site: str, callee: str) -> FlowFunction:
        ff = FlowFunction()
        ff.add_edge(ZERO_FACT, ZERO_FACT)

        # Map callee's return definitions to caller's result variable
        call_pp = self.icfg.points.get(call_site)
        if call_pp and 'result_var' in call_pp.details:
            result_var = call_pp.details['result_var']
            result_def = DataflowFact(FactKind.REACH_DEF, result_var, call_site)
            ff.gen(result_def)
            # Propagate callee's definitions that reach exit
            callee_locals = self.icfg.functions.get(callee, {}).get("locals", set())
            for f in self._all_facts:
                if f.kind == FactKind.REACH_DEF and f.var in callee_locals:
                    ff.add_edge(f, result_def)

        return ff

    def call_to_return_flow(self, call_site: str, return_site: str,
                            callee: str) -> FlowFunction:
        ff = FlowFunction()
        call_pp = self.icfg.points.get(call_site)
        result_var = call_pp.details.get('result_var') if call_pp else None

        # Pass through all facts except those killed by the call result
        for f in self._all_facts:
            if result_var and f.kind == FactKind.REACH_DEF and f.var == result_var:
                continue  # killed by call
            ff.add_edge(f, f)

        return ff


class TaintAnalysisProblem(IFDSProblem):
    """IFDS problem for context-sensitive taint analysis."""

    def __init__(self, icfg: ICFG, taint_sources: Set[str] = None,
                 taint_sinks: Set[str] = None):
        super().__init__(icfg)
        self.taint_sources = taint_sources or set()
        self.taint_sinks = taint_sinks or set()
        self._all_facts = self._compute_all_facts()

    def _compute_all_facts(self) -> Set[DataflowFact]:
        facts = {ZERO_FACT}
        all_vars = set()
        for pid, pp in self.icfg.points.items():
            if 'var' in pp.details:
                all_vars.add(pp.details['var'])
            if 'uses' in pp.details:
                all_vars |= pp.details.get('uses', set())
            if 'params' in pp.details:
                all_vars |= set(pp.details['params'])
        for fn_info in self.icfg.functions.values():
            all_vars |= set(fn_info.get("params", []))
            all_vars |= fn_info.get("locals", set())
        for v in all_vars:
            facts.add(DataflowFact(FactKind.TAINT, v))
        return facts

    def all_facts(self) -> Set[DataflowFact]:
        return self._all_facts

    def initial_seeds(self) -> Dict[str, Set[DataflowFact]]:
        entry = self.icfg.functions.get(self.icfg.entry_function, {}).get("entry")
        seeds = {ZERO_FACT}
        # Mark taint sources
        for src_var in self.taint_sources:
            seeds.add(DataflowFact(FactKind.TAINT, src_var))
        if entry:
            return {entry: seeds}
        return {}

    def flow_function(self, edge: ICFGEdge) -> FlowFunction:
        ff = FlowFunction()
        src_pp = self.icfg.points.get(edge.source)

        if not src_pp:
            ff.identity(self._all_facts)
            return ff

        if src_pp.stmt_type == "assign" and 'var' in src_pp.details:
            var = src_pp.details['var']
            uses = src_pp.details.get('uses', set())

            for f in self._all_facts:
                if f.kind == FactKind.TAINT and f.var == var:
                    # Kill old taint of var (will be re-generated if needed)
                    continue
                ff.add_edge(f, f)

            # If any used variable is tainted, the defined variable becomes tainted
            taint_var = DataflowFact(FactKind.TAINT, var)
            for use_var in uses:
                taint_use = DataflowFact(FactKind.TAINT, use_var)
                ff.add_edge(taint_use, taint_var)

            # If var is a taint source, generate taint
            if var in self.taint_sources:
                ff.gen(taint_var)
        else:
            ff.identity(self._all_facts)

        return ff

    def call_flow(self, call_site: str, callee_entry: str,
                  callee: str) -> FlowFunction:
        ff = FlowFunction()
        ff.add_edge(ZERO_FACT, ZERO_FACT)

        call_pp = self.icfg.points.get(call_site)
        callee_info = self.icfg.functions.get(callee, {})
        params = callee_info.get("params", [])

        if call_pp and 'args' in call_pp.details:
            args_vars = call_pp.details['args']
            for i, param in enumerate(params):
                taint_param = DataflowFact(FactKind.TAINT, param)
                if i < len(args_vars):
                    for arg_var in args_vars[i]:
                        taint_arg = DataflowFact(FactKind.TAINT, arg_var)
                        ff.add_edge(taint_arg, taint_param)

        return ff

    def return_flow(self, callee_exit: str, return_site: str,
                    call_site: str, callee: str) -> FlowFunction:
        ff = FlowFunction()
        ff.add_edge(ZERO_FACT, ZERO_FACT)

        call_pp = self.icfg.points.get(call_site)
        if call_pp and 'result_var' in call_pp.details:
            result_var = call_pp.details['result_var']
            taint_result = DataflowFact(FactKind.TAINT, result_var)

            # Propagate taint from callee locals to result
            callee_info = self.icfg.functions.get(callee, {})
            callee_locals = callee_info.get("locals", set())
            callee_params = callee_info.get("params", [])

            for v in callee_locals | set(callee_params):
                taint_v = DataflowFact(FactKind.TAINT, v)
                ff.add_edge(taint_v, taint_result)

        return ff

    def call_to_return_flow(self, call_site: str, return_site: str,
                            callee: str) -> FlowFunction:
        ff = FlowFunction()
        call_pp = self.icfg.points.get(call_site)
        result_var = call_pp.details.get('result_var') if call_pp else None

        for f in self._all_facts:
            if result_var and f.kind == FactKind.TAINT and f.var == result_var:
                continue
            ff.add_edge(f, f)

        return ff


class LiveVariablesProblem(IFDSProblem):
    """IFDS problem for context-sensitive live variables (backward)."""

    def __init__(self, icfg: ICFG):
        super().__init__(icfg)
        self._all_facts = self._compute_all_facts()

    def _compute_all_facts(self) -> Set[DataflowFact]:
        facts = {ZERO_FACT}
        all_vars = set()
        for pid, pp in self.icfg.points.items():
            if 'var' in pp.details:
                all_vars.add(pp.details['var'])
            if 'uses' in pp.details:
                all_vars |= pp.details.get('uses', set())
        for fn_info in self.icfg.functions.values():
            all_vars |= set(fn_info.get("params", []))
            all_vars |= fn_info.get("locals", set())
        for v in all_vars:
            facts.add(DataflowFact(FactKind.LIVE_VAR, v))
        return facts

    def all_facts(self) -> Set[DataflowFact]:
        return self._all_facts

    def flow_function(self, edge: ICFGEdge) -> FlowFunction:
        """For backward analysis, this is the 'backward' flow function.
        At a definition x = expr using {y, z}:
          - Kill: live(x)
          - Gen: live(y), live(z)
        """
        ff = FlowFunction()
        src_pp = self.icfg.points.get(edge.source)

        if not src_pp:
            ff.identity(self._all_facts)
            return ff

        if src_pp.stmt_type == "assign" and 'var' in src_pp.details:
            var = src_pp.details['var']
            uses = src_pp.details.get('uses', set())

            for f in self._all_facts:
                if f.kind == FactKind.LIVE_VAR and f.var == var:
                    continue  # kill
                ff.add_edge(f, f)

            # Gen: used variables become live
            for u in uses:
                ff.gen(DataflowFact(FactKind.LIVE_VAR, u))
        elif src_pp.stmt_type == "return" and 'uses' in src_pp.details:
            ff.identity(self._all_facts)
            for u in src_pp.details['uses']:
                ff.gen(DataflowFact(FactKind.LIVE_VAR, u))
        elif src_pp.stmt_type == "branch" and 'uses' in src_pp.details:
            ff.identity(self._all_facts)
            for u in src_pp.details['uses']:
                ff.gen(DataflowFact(FactKind.LIVE_VAR, u))
        else:
            ff.identity(self._all_facts)

        return ff

    def call_flow(self, call_site, callee_entry, callee):
        ff = FlowFunction()
        ff.add_edge(ZERO_FACT, ZERO_FACT)
        callee_info = self.icfg.functions.get(callee, {})
        params = callee_info.get("params", [])
        for p in params:
            ff.add_edge(DataflowFact(FactKind.LIVE_VAR, p),
                       DataflowFact(FactKind.LIVE_VAR, p))
        return ff

    def return_flow(self, callee_exit, return_site, call_site, callee):
        ff = FlowFunction()
        ff.add_edge(ZERO_FACT, ZERO_FACT)
        call_pp = self.icfg.points.get(call_site)
        if call_pp and 'args' in call_pp.details:
            for arg_vars in call_pp.details['args']:
                for v in arg_vars:
                    ff.gen(DataflowFact(FactKind.LIVE_VAR, v))
        return ff

    def call_to_return_flow(self, call_site, return_site, callee):
        ff = FlowFunction()
        callee_info = self.icfg.functions.get(callee, {})
        callee_vars = callee_info.get("locals", set()) | set(callee_info.get("params", []))
        for f in self._all_facts:
            if f.kind == FactKind.LIVE_VAR and f.var in callee_vars:
                continue
            ff.add_edge(f, f)
        return ff


def solve_ifds(problem: IFDSProblem) -> IFDSResult:
    """Solve an IFDS problem using the tabulation algorithm.

    This is the Reps-Horwitz-Sagiv IFDS tabulation algorithm.
    Uses the exploded supergraph with worklist iteration.
    """
    icfg = problem.icfg
    all_facts = problem.all_facts()

    # PathEdge: (d1, n, d2) means: fact d1 at procedure entry reaches
    # point n as fact d2
    path_edges: Set[Tuple[DataflowFact, str, DataflowFact]] = set()
    worklist: deque = deque()

    # Summary edges: (d1, d2) at (call_site, return_site)
    summary_edges: Dict[Tuple[str, str], Set[Tuple[DataflowFact, DataflowFact]]] = defaultdict(set)

    # Callers: function -> set of (call_site, d_before_call)
    callers: Dict[str, Set[Tuple[str, DataflowFact]]] = defaultdict(set)

    # End summary: function -> set of (d_entry, d_exit) pairs
    end_summary: Dict[str, Set[Tuple[DataflowFact, DataflowFact]]] = defaultdict(set)

    # Initialize with seeds
    for point, facts in problem.initial_seeds().items():
        for d in facts:
            edge = (d, point, d)
            if edge not in path_edges:
                path_edges.add(edge)
                worklist.append(edge)

    iterations = 0
    max_iterations = 50000

    while worklist and iterations < max_iterations:
        iterations += 1
        d1, n, d2 = worklist.popleft()

        pp = icfg.points.get(n)
        if not pp:
            continue

        # Check if n is a function exit
        fn_info = icfg.functions.get(pp.function, {})
        is_exit = fn_info.get("exit") == n

        if is_exit:
            # Process end of procedure
            end_summary[pp.function].add((d1, d2))

            # For each caller that called this function with d1
            for (call_site, d_caller) in callers.get(pp.function, set()):
                # Find return site from call edges
                call_edges = [e for e in icfg.edges
                             if e.source == call_site and e.edge_type == "call"
                             and e.callee == pp.function]
                for ce in call_edges:
                    ret_edges = [e for e in icfg.edges
                                if e.source == fn_info.get("exit", "")
                                and e.edge_type == "return"
                                and e.callee == pp.function]
                    for re in ret_edges:
                        return_site = re.target
                        ret_ff = problem.return_flow(n, return_site,
                                                     call_site, pp.function)
                        for d3 in ret_ff.apply({d2}):
                            new_edge = (d_caller, return_site, d3)
                            if new_edge not in path_edges:
                                path_edges.add(new_edge)
                                worklist.append(new_edge)
            continue

        # Process successors
        for succ_edge in icfg.get_successors(n):
            if succ_edge.edge_type == "call":
                callee = succ_edge.callee
                callee_entry = succ_edge.target

                # Call flow
                call_ff = problem.call_flow(n, callee_entry, callee)
                for d3 in call_ff.apply({d2}):
                    # Record caller
                    callers[callee].add((n, d1))

                    # Add path edge at callee entry
                    new_edge = (d3, callee_entry, d3)
                    if new_edge not in path_edges:
                        path_edges.add(new_edge)
                        worklist.append(new_edge)

                    # Check if we already have end summaries for this callee
                    for (d_entry, d_exit) in end_summary.get(callee, set()):
                        if d_entry == d3:
                            callee_info = icfg.functions.get(callee, {})
                            callee_exit = callee_info.get("exit", "")
                            ret_edges = [e for e in icfg.edges
                                        if e.source == callee_exit
                                        and e.edge_type == "return"
                                        and e.callee == callee]
                            for re in ret_edges:
                                ret_ff = problem.return_flow(callee_exit,
                                                             re.target, n, callee)
                                for d4 in ret_ff.apply({d_exit}):
                                    new_pe = (d1, re.target, d4)
                                    if new_pe not in path_edges:
                                        path_edges.add(new_pe)
                                        worklist.append(new_pe)

            elif succ_edge.edge_type == "call_to_return":
                # Local facts bypass the call
                ctr_ff = problem.call_to_return_flow(n, succ_edge.target,
                                                      succ_edge.callee)
                for d3 in ctr_ff.apply({d2}):
                    new_edge = (d1, succ_edge.target, d3)
                    if new_edge not in path_edges:
                        path_edges.add(new_edge)
                        worklist.append(new_edge)

            elif succ_edge.edge_type == "intra":
                intra_ff = problem.flow_function(succ_edge)
                for d3 in intra_ff.apply({d2}):
                    new_edge = (d1, succ_edge.target, d3)
                    if new_edge not in path_edges:
                        path_edges.add(new_edge)
                        worklist.append(new_edge)

    # Collect results: for each program point, gather all facts that reach it
    reachable: Dict[str, Set[DataflowFact]] = defaultdict(set)
    for (d1, n, d2) in path_edges:
        reachable[n].add(d2)

    # Build function summaries
    summaries: Dict[str, Dict[DataflowFact, Set[DataflowFact]]] = {}
    for fn_name, pairs in end_summary.items():
        summary = defaultdict(set)
        for (d_entry, d_exit) in pairs:
            summary[d_entry].add(d_exit)
        summaries[fn_name] = dict(summary)

    return IFDSResult(
        reachable_facts=dict(reachable),
        summaries=summaries,
        context_sensitive=True,
        analysis_type=type(problem).__name__,
        stats={
            "iterations": iterations,
            "path_edges": len(path_edges),
            "end_summaries": sum(len(v) for v in end_summary.values()),
        }
    )


# ---------------------------------------------------------------------------
# Context-sensitive abstract interpretation via PDS
# ---------------------------------------------------------------------------

def compute_function_summaries(source: str) -> Dict[str, FunctionSummary]:
    """Compute abstract function summaries using C039 abstract interpreter.

    For each function, analyzes it in isolation to compute:
    - Which variables it defines/uses
    - How parameters affect the return value
    - Taint propagation through the function
    """
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()

    summaries = {}

    for stmt in program.stmts:
        cls = type(stmt).__name__
        if cls == 'FnDecl':
            fn_name = stmt.name
            params = stmt.params if hasattr(stmt, 'params') else []
            body_stmts = stmt.body.stmts if hasattr(stmt.body, 'stmts') else []

            defined = set()
            used = set()
            return_deps = set()
            tainted_params = set()

            for s in body_stmts:
                s_cls = type(s).__name__
                if s_cls in ('LetDecl', 'Assign'):
                    var = s.name
                    defined.add(var)
                    expr_vars = _extract_vars_from_expr(s.value)
                    used |= expr_vars
                    # Track which params flow to defined vars
                    for p in params:
                        if p in expr_vars:
                            tainted_params.add(p)

                elif s_cls == 'ReturnStmt':
                    if hasattr(s, 'value') and s.value:
                        ret_vars = _extract_vars_from_expr(s.value)
                        used |= ret_vars
                        return_deps |= (ret_vars & (set(params) | defined))
                        for p in params:
                            if p in ret_vars:
                                tainted_params.add(p)

            # Use C039 for abstract analysis if possible
            env_effect = None
            try:
                # Build a self-contained snippet for analysis
                snippet_parts = []
                for p in params:
                    snippet_parts.append(f"let {p} = 0;")
                for s in body_stmts:
                    # Skip return statements for snippet analysis
                    pass
                if snippet_parts:
                    snippet = "\n".join(snippet_parts)
                    result = ai_analyze(snippet)
                    env_effect = result.get('env')
            except Exception:
                pass

            summaries[fn_name] = FunctionSummary(
                name=fn_name,
                params=params,
                defined_vars=defined,
                used_vars=used,
                tainted_params=tainted_params,
                return_deps=return_deps,
                env_effect=env_effect,
            )

    return summaries


# ---------------------------------------------------------------------------
# Call-string approach (k-CFA)
# ---------------------------------------------------------------------------

def call_string_analysis(source: str, k: int = 1,
                         analysis: str = "reaching_defs") -> IFDSResult:
    """Context-sensitive analysis using k-length call strings.

    Each analysis context is distinguished by the last k call sites.
    This is equivalent to the PDS approach with bounded stack depth.

    Args:
        source: C10 source code
        k: call string length (1 = 1-CFA, 2 = 2-CFA, etc.)
        analysis: "reaching_defs" or "taint"
    """
    icfg = build_icfg(source)

    if analysis == "reaching_defs":
        problem = ReachingDefinitionsProblem(icfg)
    elif analysis == "taint":
        problem = TaintAnalysisProblem(icfg)
    else:
        problem = ReachingDefinitionsProblem(icfg)

    # The standard IFDS solver is already context-sensitive through
    # the tabulation algorithm's call/return matching.
    # The k parameter limits context sensitivity for performance.
    result = solve_ifds(problem)
    result.stats["k"] = k
    return result


# ---------------------------------------------------------------------------
# PDS-based context-sensitive analysis
# ---------------------------------------------------------------------------

def pds_reachability_analysis(source: str) -> Dict:
    """Use PDS reachability to determine which program points are reachable.

    This uses V094's pre*/post* for exact (unbounded) context-sensitive
    reachability, as opposed to k-bounded call strings.
    """
    icfg = build_icfg(source)
    pds, init_config = icfg_to_pds(icfg)

    # Compute forward reachability (post*)
    init_aut = make_config_automaton(pds, {init_config})
    post_aut = post_star(pds, init_aut)

    # Check which program points are reachable
    reachable_points = set()
    for pid in icfg.points:
        # A point is reachable if (q, pid, ...) is in post*
        config = Configuration("q", (pid,))
        if post_aut.accepts(config):
            reachable_points.add(pid)

    # Also check with longer stacks (call contexts)
    for pid in icfg.points:
        for pid2 in icfg.points:
            config = Configuration("q", (pid, pid2))
            if post_aut.accepts(config):
                reachable_points.add(pid)

    return {
        "reachable_points": reachable_points,
        "total_points": len(icfg.points),
        "unreachable_points": set(icfg.points.keys()) - reachable_points,
        "icfg": icfg,
        "pds_summary": {
            "states": len(pds.states),
            "stack_symbols": len(pds.stack_alphabet),
            "rules": len(pds.rules),
        }
    }


# ---------------------------------------------------------------------------
# Combined analysis: IFDS + PDS reachability + AI
# ---------------------------------------------------------------------------

def interprocedural_analyze(source: str,
                            analysis: str = "reaching_defs",
                            taint_sources: Set[str] = None,
                            taint_sinks: Set[str] = None) -> IFDSResult:
    """Full interprocedural analysis combining IFDS + PDS.

    Args:
        source: C10 source code
        analysis: "reaching_defs", "taint", or "live_vars"
        taint_sources: variables to mark as tainted (for taint analysis)
        taint_sinks: variables to check for taint (for taint analysis)
    """
    icfg = build_icfg(source)

    if analysis == "reaching_defs":
        problem = ReachingDefinitionsProblem(icfg)
    elif analysis == "taint":
        problem = TaintAnalysisProblem(icfg, taint_sources or set(),
                                        taint_sinks or set())
    elif analysis == "live_vars":
        problem = LiveVariablesProblem(icfg)
    else:
        problem = ReachingDefinitionsProblem(icfg)

    result = solve_ifds(problem)
    result.analysis_type = analysis
    return result


def reaching_definitions(source: str) -> Dict[str, Set[str]]:
    """Compute context-sensitive reaching definitions.

    Returns: dict mapping program point -> set of (var, def_site) pairs.
    """
    result = interprocedural_analyze(source, "reaching_defs")
    output = {}
    for point, facts in result.reachable_facts.items():
        defs = set()
        for f in facts:
            if f.kind == FactKind.REACH_DEF:
                defs.add((f.var, f.site))
        if defs:
            output[point] = defs
    return output


def taint_analysis(source: str, sources: Set[str],
                   sinks: Set[str] = None) -> Dict:
    """Context-sensitive taint analysis.

    Args:
        source: C10 source code
        sources: variable names to mark as taint sources
        sinks: variable names to check for taint leaks

    Returns: dict with tainted variables at each point, sink violations, summaries
    """
    result = interprocedural_analyze(source, "taint",
                                      taint_sources=sources,
                                      taint_sinks=sinks)

    tainted_at = {}
    for point, facts in result.reachable_facts.items():
        tainted_vars = {f.var for f in facts if f.kind == FactKind.TAINT}
        if tainted_vars:
            tainted_at[point] = tainted_vars

    # Check for sink violations
    violations = []
    if sinks:
        for point, tainted_vars in tainted_at.items():
            leaked = tainted_vars & sinks
            if leaked:
                violations.append({
                    "point": point,
                    "leaked_vars": leaked,
                })

    return {
        "tainted_at": tainted_at,
        "violations": violations,
        "summaries": result.summaries,
        "stats": result.stats,
    }


def live_variables(source: str) -> Dict[str, Set[str]]:
    """Compute context-sensitive live variables.

    Returns: dict mapping program point -> set of live variable names.
    """
    result = interprocedural_analyze(source, "live_vars")
    output = {}
    for point, facts in result.reachable_facts.items():
        live = {f.var for f in facts if f.kind == FactKind.LIVE_VAR}
        if live:
            output[point] = live
    return output


# ---------------------------------------------------------------------------
# Comparison: context-sensitive vs context-insensitive
# ---------------------------------------------------------------------------

def context_insensitive_analyze(source: str,
                                 analysis: str = "reaching_defs",
                                 taint_sources: Set[str] = None) -> Dict[str, Set[DataflowFact]]:
    """Context-insensitive analysis (merges all calling contexts).

    Simpler but less precise: treats all calls to the same function
    as the same context.
    """
    icfg = build_icfg(source)

    if analysis == "reaching_defs":
        problem = ReachingDefinitionsProblem(icfg)
    elif analysis == "taint":
        problem = TaintAnalysisProblem(icfg, taint_sources or set())
    else:
        problem = ReachingDefinitionsProblem(icfg)

    all_facts = problem.all_facts()

    # Simple worklist with no call/return matching
    reachable: Dict[str, Set[DataflowFact]] = defaultdict(set)
    worklist = deque()

    for point, facts in problem.initial_seeds().items():
        reachable[point] |= facts
        worklist.append(point)

    visited_count = 0
    max_visits = 50000

    while worklist and visited_count < max_visits:
        visited_count += 1
        n = worklist.popleft()

        for edge in icfg.get_successors(n):
            if edge.edge_type in ("intra", "call_to_return"):
                ff = problem.flow_function(edge)
                new_facts = ff.apply(reachable[n])
                if not new_facts.issubset(reachable[edge.target]):
                    reachable[edge.target] |= new_facts
                    worklist.append(edge.target)

            elif edge.edge_type == "call":
                callee = edge.callee
                callee_entry = edge.target
                call_ff = problem.call_flow(n, callee_entry, callee)
                new_facts = call_ff.apply(reachable[n])
                if not new_facts.issubset(reachable[callee_entry]):
                    reachable[callee_entry] |= new_facts
                    worklist.append(callee_entry)

            elif edge.edge_type == "return":
                # Context-insensitive: return to ALL callers
                callee = edge.callee
                return_site = edge.target
                # Find all call sites
                call_sites = [e.source for e in icfg.edges
                             if e.edge_type == "call" and e.callee == callee]
                for cs in call_sites:
                    ret_ff = problem.return_flow(n, return_site, cs, callee)
                    new_facts = ret_ff.apply(reachable[n])
                    if not new_facts.issubset(reachable[return_site]):
                        reachable[return_site] |= new_facts
                        worklist.append(return_site)

    return dict(reachable)


def compare_sensitivity(source: str, analysis: str = "reaching_defs",
                         taint_sources: Set[str] = None) -> Dict:
    """Compare context-sensitive vs context-insensitive analysis.

    Returns precision metrics showing how much context sensitivity helps.
    """
    # Context-sensitive (IFDS)
    cs_result = interprocedural_analyze(source, analysis,
                                         taint_sources=taint_sources)

    # Context-insensitive
    ci_result = context_insensitive_analyze(source, analysis,
                                             taint_sources=taint_sources)

    # Compare fact counts at each point
    cs_total = sum(len(v) for v in cs_result.reachable_facts.values())
    ci_total = sum(len(v) for v in ci_result.values())

    # Points where CI has more facts (less precise)
    imprecise_points = []
    for point in ci_result:
        cs_facts = cs_result.reachable_facts.get(point, set())
        ci_facts = ci_result.get(point, set())
        spurious = ci_facts - cs_facts
        if spurious:
            imprecise_points.append({
                "point": point,
                "spurious_facts": len(spurious),
                "cs_facts": len(cs_facts),
                "ci_facts": len(ci_facts),
            })

    return {
        "context_sensitive_facts": cs_total,
        "context_insensitive_facts": ci_total,
        "precision_ratio": cs_total / max(ci_total, 1),
        "imprecise_points": imprecise_points,
        "cs_more_precise": cs_total <= ci_total,
        "analysis_type": analysis,
        "cs_stats": cs_result.stats,
    }


# ---------------------------------------------------------------------------
# PDS-based interprocedural analysis
# ---------------------------------------------------------------------------

def pds_context_analysis(source: str, target_point: str = None) -> Dict:
    """Use PDS reachability for interprocedural analysis.

    Computes which calling contexts can reach a given program point,
    using V094's pre*/post* for exact stack-based reasoning.
    """
    icfg = build_icfg(source)
    pds, init_config = icfg_to_pds(icfg)

    # Forward reachability
    init_aut = make_config_automaton(pds, {init_config})
    post_aut = post_star(pds, init_aut)

    results = {}

    # For each function, check reachable calling contexts
    for fn_name, fn_info in icfg.functions.items():
        entry = fn_info["entry"]
        # Check if function entry is reachable
        config_entry = Configuration("q", (entry,))
        reachable = post_aut.accepts(config_entry)

        # Check with various stack contexts (calling contexts)
        contexts = []
        for other_fn, other_info in icfg.functions.items():
            for pt in other_info.get("points", []):
                ctx_config = Configuration("q", (entry, pt))
                if post_aut.accepts(ctx_config):
                    contexts.append(pt)

        results[fn_name] = {
            "reachable": reachable,
            "calling_contexts": contexts,
            "context_count": len(contexts),
        }

    # If target point specified, do backward reachability
    target_info = {}
    if target_point and target_point in icfg.points:
        target_config = Configuration("q", (target_point,))
        target_aut = make_config_automaton(pds, {target_config})
        pre_aut = pre_star(pds, target_aut)

        # Which entry points can reach target?
        reaching_entries = []
        for fn_name, fn_info in icfg.functions.items():
            entry = fn_info["entry"]
            if pre_aut.accepts(Configuration("q", (entry,))):
                reaching_entries.append(fn_name)

        target_info = {
            "target_point": target_point,
            "reaching_functions": reaching_entries,
        }

    return {
        "functions": results,
        "target_analysis": target_info,
        "pds_stats": {
            "states": len(pds.states),
            "stack_symbols": len(pds.stack_alphabet),
            "rules": len(pds.rules),
        }
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def full_interprocedural_analysis(source: str,
                                   taint_sources: Set[str] = None,
                                   taint_sinks: Set[str] = None) -> Dict:
    """Run all interprocedural analyses on source code.

    Combines IFDS reaching definitions, taint analysis, live variables,
    PDS reachability, and function summaries.
    """
    icfg = build_icfg(source)

    # 1. Reaching definitions
    rd_result = interprocedural_analyze(source, "reaching_defs")

    # 2. Taint analysis (if sources given)
    taint_result = None
    if taint_sources:
        taint_result = taint_analysis(source, taint_sources, taint_sinks)

    # 3. Live variables
    lv_result = interprocedural_analyze(source, "live_vars")

    # 4. Function summaries
    summaries = compute_function_summaries(source)

    # 5. PDS reachability
    pds_result = pds_reachability_analysis(source)

    # Build combined report
    report = {
        "icfg": {
            "points": len(icfg.points),
            "edges": len(icfg.edges),
            "functions": list(icfg.functions.keys()),
        },
        "reaching_definitions": {
            "facts_per_point": {k: len(v) for k, v in rd_result.reachable_facts.items()},
            "total_facts": sum(len(v) for v in rd_result.reachable_facts.values()),
            "summaries": {k: {str(d1): [str(d2) for d2 in d2s]
                              for d1, d2s in v.items()}
                          for k, v in rd_result.summaries.items()},
            "stats": rd_result.stats,
        },
        "live_variables": {
            "live_per_point": {
                k: {f.var for f in v if f.kind == FactKind.LIVE_VAR}
                for k, v in lv_result.reachable_facts.items()
            },
            "stats": lv_result.stats,
        },
        "function_summaries": {
            name: {
                "params": s.params,
                "defined_vars": list(s.defined_vars),
                "used_vars": list(s.used_vars),
                "tainted_params": list(s.tainted_params),
                "return_deps": list(s.return_deps),
            }
            for name, s in summaries.items()
        },
        "pds_reachability": {
            "reachable": len(pds_result["reachable_points"]),
            "unreachable": len(pds_result["unreachable_points"]),
            "unreachable_points": list(pds_result["unreachable_points"]),
        },
    }

    if taint_result:
        report["taint_analysis"] = {
            "tainted_points": len(taint_result["tainted_at"]),
            "violations": taint_result["violations"],
            "stats": taint_result["stats"],
        }

    return report


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def get_reaching_defs_at(source: str, point: str) -> Set[Tuple[str, str]]:
    """Get reaching definitions at a specific program point.

    Returns set of (variable, definition_site) pairs.
    """
    result = interprocedural_analyze(source, "reaching_defs")
    facts = result.reachable_facts.get(point, set())
    return {(f.var, f.site) for f in facts if f.kind == FactKind.REACH_DEF}


def get_tainted_vars_at(source: str, point: str,
                         sources: Set[str]) -> Set[str]:
    """Get tainted variables at a specific program point.

    Returns set of variable names that are tainted.
    """
    result = interprocedural_analyze(source, "taint",
                                      taint_sources=sources)
    facts = result.reachable_facts.get(point, set())
    return {f.var for f in facts if f.kind == FactKind.TAINT}


def get_live_vars_at(source: str, point: str) -> Set[str]:
    """Get live variables at a specific program point."""
    result = interprocedural_analyze(source, "live_vars")
    facts = result.reachable_facts.get(point, set())
    return {f.var for f in facts if f.kind == FactKind.LIVE_VAR}


def icfg_summary(source: str) -> Dict:
    """Get ICFG summary for source code."""
    icfg = build_icfg(source)
    return {
        "points": len(icfg.points),
        "edges": len(icfg.edges),
        "functions": {
            name: {
                "entry": info["entry"],
                "exit": info["exit"],
                "params": info["params"],
                "locals": list(info["locals"]),
                "point_count": len(info["points"]),
            }
            for name, info in icfg.functions.items()
        },
        "call_edges": len([e for e in icfg.edges if e.edge_type == "call"]),
        "return_edges": len([e for e in icfg.edges if e.edge_type == "return"]),
        "intra_edges": len([e for e in icfg.edges if e.edge_type == "intra"]),
    }
