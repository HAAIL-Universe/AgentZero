"""
V015: k-Induction Model Checking

k-Induction is a bounded model checking technique that combines:
1. Base case: Property holds for the first k steps (BMC)
2. Inductive step: If property holds for k consecutive steps, it holds for step k+1

If both checks pass for some k, the property holds universally (unbounded).

Composes: C037 (SMT solver) + V002 (TransitionSystem)

Also provides:
- k-induction with invariant strengthening (using V007 or manual invariants)
- Incremental k: automatically finds the minimal k that proves the property
- Comparison with PDR (V002) for benchmarking
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))

from smt_solver import (
    SMTSolver, SMTResult, Op, Var, IntConst, BoolConst, App, Sort,
    INT, BOOL
)
from pdr import TransitionSystem


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class KIndResult:
    """Result of k-induction check."""
    def __init__(self, result, k=None, counterexample=None, invariant=None, stats=None):
        self.result = result            # "SAFE", "UNSAFE", "UNKNOWN"
        self.k = k                      # k value used
        self.counterexample = counterexample  # list of states if UNSAFE
        self.invariant = invariant       # strengthening invariant if used
        self.stats = stats or {}         # timing, solver calls, etc.

    def __repr__(self):
        return f"KIndResult({self.result}, k={self.k})"


# ---------------------------------------------------------------------------
# Core k-induction engine
# ---------------------------------------------------------------------------

def _make_solver(ts):
    """Create a fresh SMT solver with all state variables registered."""
    s = SMTSolver()
    for name, sort in ts.state_vars:
        if sort == INT:
            s.Int(name)
        else:
            s.Bool(name)
    return s


def _step_vars(ts, solver, step):
    """Get or create step-indexed variables: x_0, x_1, etc."""
    var_map = {}
    for name, sort in ts.state_vars:
        step_name = f"{name}_{step}"
        if sort == INT:
            var_map[name] = solver.Int(step_name)
        else:
            var_map[name] = solver.Bool(step_name)
    return var_map


def _substitute(formula, var_map):
    """Substitute variables in a formula using a name->Term mapping."""
    if isinstance(formula, Var):
        return var_map.get(formula.name, formula)
    elif isinstance(formula, (IntConst, BoolConst)):
        return formula
    elif isinstance(formula, App):
        new_args = [_substitute(a, var_map) for a in formula.args]
        return App(formula.op, new_args, formula.sort)
    else:
        return formula


def _apply_formula_at_step(ts, solver, formula, step):
    """Apply a formula at a given step index (substitute state vars with step-indexed vars)."""
    var_map = _step_vars(ts, solver, step)
    return _substitute(formula, var_map)


def _apply_trans_at_step(ts, solver, step):
    """Apply transition relation from step to step+1.
    Replaces x with x_step and x' with x_{step+1}."""
    curr_vars = _step_vars(ts, solver, step)
    next_vars = _step_vars(ts, solver, step + 1)
    # Build combined mapping: x -> x_step, x' -> x_{step+1}
    var_map = {}
    for name, _ in ts.state_vars:
        var_map[name] = curr_vars[name]
        var_map[name + "'"] = next_vars[name]
    return _substitute(ts.trans_formula, var_map)


def _negate(term):
    """Negate a term using complement operators (workaround for NOT(EQ) bug)."""
    if isinstance(term, BoolConst):
        return BoolConst(not term.value)
    if isinstance(term, App):
        complement = {
            Op.EQ: Op.NEQ, Op.NEQ: Op.EQ,
            Op.LT: Op.GE, Op.GE: Op.LT,
            Op.LE: Op.GT, Op.GT: Op.LE,
        }
        if term.op in complement:
            return App(complement[term.op], term.args, BOOL)
        if term.op == Op.AND:
            return App(Op.OR, [_negate(a) for a in term.args], BOOL)
        if term.op == Op.OR:
            return App(Op.AND, [_negate(a) for a in term.args], BOOL)
        if term.op == Op.NOT:
            return term.args[0]
        if term.op == Op.IMPLIES:
            # NOT(a => b) = a AND NOT(b)
            return App(Op.AND, [term.args[0], _negate(term.args[1])], BOOL)
    return App(Op.NOT, [term], BOOL)


def _extract_trace(ts, solver, k):
    """Extract counterexample trace from SAT model."""
    model = solver.model()
    if model is None:
        return None
    trace = []
    for step in range(k + 1):
        state = {}
        for name, sort in ts.state_vars:
            step_name = f"{name}_{step}"
            if step_name in model:
                state[name] = model[step_name]
            else:
                state[name] = 0
        trace.append(state)
    return trace


# ---------------------------------------------------------------------------
# Base case check (BMC)
# ---------------------------------------------------------------------------

def check_base_case(ts, k, extra_invariant=None):
    """Check base case: property holds for steps 0..k.

    Encodes: Init(s0) AND Trans(s0,s1) AND ... AND Trans(s_{k-1},s_k) AND NOT(P(s0) AND ... AND P(s_k))
    If UNSAT, base case passes (property holds for first k+1 states).
    If SAT, found a counterexample within k steps.
    """
    s = SMTSolver()

    # Register all step-indexed variables upfront
    for step in range(k + 1):
        for name, sort in ts.state_vars:
            step_name = f"{name}_{step}"
            if sort == INT:
                s.Int(step_name)
            else:
                s.Bool(step_name)

    # Init at step 0
    s.add(_apply_formula_at_step(ts, s, ts.init_formula, 0))

    # Transition from step i to step i+1
    for i in range(k):
        s.add(_apply_trans_at_step(ts, s, i))

    # Extra invariant at each step (strengthening)
    if extra_invariant is not None:
        for step in range(k + 1):
            s.add(_apply_formula_at_step(ts, s, extra_invariant, step))

    # Negate: NOT(P(s0) AND P(s1) AND ... AND P(s_k))
    # = NOT(P(s0)) OR NOT(P(s1)) OR ... OR NOT(P(s_k))
    neg_props = []
    for step in range(k + 1):
        neg_props.append(_negate(_apply_formula_at_step(ts, s, ts.prop_formula, step)))

    if len(neg_props) == 1:
        s.add(neg_props[0])
    else:
        s.add(App(Op.OR, neg_props, BOOL))

    result = s.check()

    if result == SMTResult.SAT:
        trace = _extract_trace(ts, s, k)
        return False, trace  # Base case fails -- counterexample found
    elif result == SMTResult.UNSAT:
        return True, None    # Base case passes
    else:
        return None, None    # Unknown


# ---------------------------------------------------------------------------
# Inductive step check
# ---------------------------------------------------------------------------

def check_inductive_step(ts, k, extra_invariant=None):
    """Check inductive step: if property holds for k consecutive steps,
    it holds for step k+1.

    Encodes: P(s0) AND ... AND P(s_k) AND Trans(s0,s1) AND ... AND Trans(s_k,s_{k+1}) AND NOT(P(s_{k+1}))
    If UNSAT, inductive step passes.
    If SAT, induction fails at this k (need stronger invariant or larger k).
    """
    s = SMTSolver()

    # Register all step-indexed variables
    for step in range(k + 2):  # k+2 because we need steps 0..k+1
        for name, sort in ts.state_vars:
            step_name = f"{name}_{step}"
            if sort == INT:
                s.Int(step_name)
            else:
                s.Bool(step_name)

    # Property holds at steps 0..k (the k+1 consecutive states)
    for step in range(k + 1):
        s.add(_apply_formula_at_step(ts, s, ts.prop_formula, step))

    # Transitions from step i to step i+1 for i = 0..k
    for i in range(k + 1):
        s.add(_apply_trans_at_step(ts, s, i))

    # Extra invariant at each step (strengthening)
    if extra_invariant is not None:
        for step in range(k + 2):
            s.add(_apply_formula_at_step(ts, s, extra_invariant, step))

    # NOT(P(s_{k+1}))
    s.add(_negate(_apply_formula_at_step(ts, s, ts.prop_formula, k + 1)))

    result = s.check()

    if result == SMTResult.UNSAT:
        return True, None    # Inductive step passes
    elif result == SMTResult.SAT:
        return False, None   # Induction fails
    else:
        return None, None    # Unknown


# ---------------------------------------------------------------------------
# Uniqueness constraint (optional, for convergence)
# ---------------------------------------------------------------------------

def check_inductive_step_with_uniqueness(ts, k, extra_invariant=None):
    """Inductive step with path uniqueness constraint.

    Adds: all k+1 states in the assumption are pairwise distinct.
    This helps prove liveness-like properties where repeated states
    indicate the system is stuck.
    """
    s = SMTSolver()

    for step in range(k + 2):
        for name, sort in ts.state_vars:
            step_name = f"{name}_{step}"
            if sort == INT:
                s.Int(step_name)
            else:
                s.Bool(step_name)

    # Property at steps 0..k
    for step in range(k + 1):
        s.add(_apply_formula_at_step(ts, s, ts.prop_formula, step))

    # Transitions 0..k
    for i in range(k + 1):
        s.add(_apply_trans_at_step(ts, s, i))

    # Extra invariant
    if extra_invariant is not None:
        for step in range(k + 2):
            s.add(_apply_formula_at_step(ts, s, extra_invariant, step))

    # Uniqueness: pairwise distinct states
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            # At least one variable differs between step i and step j
            diffs = []
            for name, sort in ts.state_vars:
                vi = s.Int(f"{name}_{i}") if sort == INT else s.Bool(f"{name}_{i}")
                vj = s.Int(f"{name}_{j}") if sort == INT else s.Bool(f"{name}_{j}")
                diffs.append(App(Op.NEQ, [vi, vj], BOOL))
            if len(diffs) == 1:
                s.add(diffs[0])
            elif diffs:
                s.add(App(Op.OR, diffs, BOOL))

    # NOT(P(s_{k+1}))
    s.add(_negate(_apply_formula_at_step(ts, s, ts.prop_formula, k + 1)))

    result = s.check()

    if result == SMTResult.UNSAT:
        return True, None
    elif result == SMTResult.SAT:
        return False, None
    else:
        return None, None


# ---------------------------------------------------------------------------
# Main k-induction check
# ---------------------------------------------------------------------------

def k_induction_check(ts, k, extra_invariant=None, use_uniqueness=False):
    """Run k-induction for a specific k value.

    Args:
        ts: TransitionSystem
        k: induction depth
        extra_invariant: optional strengthening invariant (Term)
        use_uniqueness: if True, add path uniqueness constraints

    Returns:
        KIndResult with result SAFE/UNSAFE/UNKNOWN
    """
    start = time.time()
    stats = {"k": k, "base_checks": 0, "ind_checks": 0}

    # Base case
    stats["base_checks"] += 1
    base_ok, trace = check_base_case(ts, k, extra_invariant)

    if base_ok is False:
        # Counterexample found
        elapsed = time.time() - start
        stats["time"] = elapsed
        return KIndResult("UNSAFE", k=k, counterexample=trace, stats=stats)

    if base_ok is None:
        elapsed = time.time() - start
        stats["time"] = elapsed
        return KIndResult("UNKNOWN", k=k, stats=stats)

    # Inductive step
    stats["ind_checks"] += 1
    if use_uniqueness:
        ind_ok, _ = check_inductive_step_with_uniqueness(ts, k, extra_invariant)
    else:
        ind_ok, _ = check_inductive_step(ts, k, extra_invariant)

    elapsed = time.time() - start
    stats["time"] = elapsed

    if ind_ok is True:
        return KIndResult("SAFE", k=k, invariant=extra_invariant, stats=stats)
    elif ind_ok is False:
        return KIndResult("UNKNOWN", k=k, stats=stats)  # Need larger k
    else:
        return KIndResult("UNKNOWN", k=k, stats=stats)


# ---------------------------------------------------------------------------
# Incremental k-induction (finds minimal k)
# ---------------------------------------------------------------------------

def incremental_k_induction(ts, max_k=20, extra_invariant=None, use_uniqueness=False):
    """Try k-induction for k=0, 1, 2, ... up to max_k.

    Returns as soon as:
    - UNSAFE: base case fails (counterexample found at some depth)
    - SAFE: both base and inductive step pass at some k
    - UNKNOWN: max_k reached without conclusion
    """
    start = time.time()
    total_stats = {"base_checks": 0, "ind_checks": 0, "max_k_tried": 0}

    for k in range(max_k + 1):
        total_stats["max_k_tried"] = k

        # Base case for this k
        total_stats["base_checks"] += 1
        base_ok, trace = check_base_case(ts, k, extra_invariant)

        if base_ok is False:
            elapsed = time.time() - start
            total_stats["time"] = elapsed
            return KIndResult("UNSAFE", k=k, counterexample=trace, stats=total_stats)

        if base_ok is None:
            continue  # SMT returned UNKNOWN, try next k

        # Inductive step for this k
        total_stats["ind_checks"] += 1
        if use_uniqueness:
            ind_ok, _ = check_inductive_step_with_uniqueness(ts, k, extra_invariant)
        else:
            ind_ok, _ = check_inductive_step(ts, k, extra_invariant)

        if ind_ok is True:
            elapsed = time.time() - start
            total_stats["time"] = elapsed
            return KIndResult("SAFE", k=k, invariant=extra_invariant, stats=total_stats)

    elapsed = time.time() - start
    total_stats["time"] = elapsed
    return KIndResult("UNKNOWN", k=max_k, stats=total_stats)


# ---------------------------------------------------------------------------
# k-Induction with invariant strengthening
# ---------------------------------------------------------------------------

def k_induction_with_strengthening(ts, max_k=20, invariants=None):
    """k-Induction with auxiliary invariant strengthening.

    Invariant strengthening makes the inductive step stronger by adding
    known invariants to both the assumption and the conclusion. This can
    make the induction go through at a smaller k.

    Args:
        ts: TransitionSystem
        max_k: maximum k to try
        invariants: list of Term formulas that are (believed to be) invariants

    Returns:
        KIndResult
    """
    if not invariants:
        return incremental_k_induction(ts, max_k)

    # Combine invariants into a conjunction
    s = SMTSolver()
    if len(invariants) == 1:
        combined = invariants[0]
    else:
        combined = s.And(*invariants)

    return incremental_k_induction(ts, max_k, extra_invariant=combined)


# ---------------------------------------------------------------------------
# Source-level API (compose with C010 parser)
# ---------------------------------------------------------------------------

def _parse_source(source):
    """Parse C010 source into AST statements."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
    from stack_vm import lex, Parser
    tokens = lex(source)
    program = Parser(tokens).parse()
    return program.stmts


def _extract_loop_ts(source):
    """Extract a transition system from a while loop in source code.
    Returns (TransitionSystem, dict of var_name -> Var in TS)."""
    stmts = _parse_source(source)

    from stack_vm import LetDecl, Assign, WhileStmt, IfStmt, Block, Var as ASTVar, BinOp, IntLit

    ts = TransitionSystem()
    inits = {}  # name -> value

    # Collect pre-loop initializations
    loop = None
    for stmt in stmts:
        if isinstance(stmt, WhileStmt):
            loop = stmt
            break
        elif isinstance(stmt, LetDecl):
            inits[stmt.name] = stmt.value

    if loop is None:
        raise ValueError("No while loop found in source")

    # Determine state variables from initializations
    ts_vars = {}
    for name in inits:
        v = ts.add_int_var(name)
        ts_vars[name] = v

    # Also scan loop body for assigned variables not yet declared
    def scan_assigns(body_stmts):
        assigned = set()
        for s in body_stmts:
            if isinstance(s, Assign):
                assigned.add(s.name)
            elif isinstance(s, IfStmt):
                then_stmts = s.then_body.stmts if isinstance(s.then_body, Block) else [s.then_body]
                assigned |= scan_assigns(then_stmts)
                if s.else_body:
                    else_stmts = s.else_body.stmts if isinstance(s.else_body, Block) else [s.else_body]
                    assigned |= scan_assigns(else_stmts)
        return assigned

    body_stmts = loop.body.stmts if isinstance(loop.body, Block) else [loop.body]
    for name in scan_assigns(body_stmts):
        if name not in ts_vars:
            inits[name] = IntLit(0)
            v = ts.add_int_var(name)
            ts_vars[name] = v

    # Build init formula
    def ast_to_smt(expr, vars_map):
        if isinstance(expr, IntLit):
            return IntConst(expr.value)
        elif isinstance(expr, ASTVar):
            if expr.name in vars_map:
                return vars_map[expr.name]
            return IntConst(0)
        elif isinstance(expr, BinOp):
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                      '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
                      '==': Op.EQ, '!=': Op.NEQ}
            l = ast_to_smt(expr.left, vars_map)
            r = ast_to_smt(expr.right, vars_map)
            op = op_map.get(expr.op, None)
            if op is None:
                raise ValueError(f"Unknown operator: {expr.op}")
            sort = BOOL if op in (Op.LT, Op.GT, Op.LE, Op.GE, Op.EQ, Op.NEQ) else INT
            return App(op, [l, r], sort)
        else:
            return IntConst(0)

    init_parts = []
    for name, val_expr in inits.items():
        v = ts_vars[name]
        val = ast_to_smt(val_expr, ts_vars)
        init_parts.append(App(Op.EQ, [v, val], BOOL))

    if len(init_parts) == 1:
        ts.set_init(init_parts[0])
    elif init_parts:
        ts.set_init(App(Op.AND, init_parts, BOOL))
    else:
        ts.set_init(BoolConst(True))

    # Build transition relation (guarded)
    cond_smt = ast_to_smt(loop.cond, ts_vars)

    # Compute body assignments
    primed = {name: ts.prime(name) for name in ts_vars}

    def build_body_trans(body_stmts, curr_map):
        """Build transition for body, returning list of (var_name, next_value) pairs."""
        assigns = {}  # name -> smt_expr
        for s in body_stmts:
            if isinstance(s, Assign):
                # Use current values (with any prior assigns in this block)
                lookup = dict(curr_map)
                for n, e in assigns.items():
                    lookup[n] = e
                assigns[s.name] = ast_to_smt(s.value, lookup)
            elif isinstance(s, IfStmt):
                c = ast_to_smt(s.cond, curr_map)
                then_s = s.then_body.stmts if isinstance(s.then_body, Block) else [s.then_body]
                then_assigns = build_body_trans(then_s, curr_map)
                else_assigns = {}
                if s.else_body:
                    else_s = s.else_body.stmts if isinstance(s.else_body, Block) else [s.else_body]
                    else_assigns = build_body_trans(else_s, curr_map)
                # Merge with ITE
                all_vars = set(then_assigns.keys()) | set(else_assigns.keys())
                for n in all_vars:
                    then_val = then_assigns.get(n, curr_map.get(n, IntConst(0)))
                    else_val = else_assigns.get(n, curr_map.get(n, IntConst(0)))
                    assigns[n] = App(Op.ITE, [c, then_val, else_val], INT)
        return assigns

    body_assigns = build_body_trans(body_stmts, ts_vars)

    # Guarded transition: (cond AND body_trans) OR (NOT cond AND frame)
    body_parts = []
    frame_parts = []
    for name in ts_vars:
        p = primed[name]
        if name in body_assigns:
            body_parts.append(App(Op.EQ, [p, body_assigns[name]], BOOL))
        else:
            body_parts.append(App(Op.EQ, [p, ts_vars[name]], BOOL))
        frame_parts.append(App(Op.EQ, [p, ts_vars[name]], BOOL))

    def _and(parts):
        if len(parts) == 1:
            return parts[0]
        return App(Op.AND, parts, BOOL)

    body_trans = _and([cond_smt] + body_parts)
    frame_trans = _and([_negate(cond_smt)] + frame_parts)

    ts.set_trans(App(Op.OR, [body_trans, frame_trans], BOOL))

    return ts, ts_vars


def verify_loop(source, property_source, max_k=20, use_uniqueness=False):
    """Verify a property about a while loop using k-induction.

    Args:
        source: C010 source with a while loop
        property_source: property as a C10 expression string (e.g., "x >= 0")
        max_k: maximum induction depth
        use_uniqueness: add path uniqueness constraints

    Returns:
        KIndResult
    """
    ts, ts_vars = _extract_loop_ts(source)

    # Parse property
    from stack_vm import lex, Parser
    prop_tokens = lex(f"let __p = ({property_source});")
    prop_stmts = Parser(prop_tokens).parse().stmts

    from stack_vm import LetDecl, BinOp, IntLit, Var as ASTVar

    def ast_to_smt_prop(expr):
        if isinstance(expr, IntLit):
            return IntConst(expr.value)
        elif isinstance(expr, ASTVar):
            if expr.name in ts_vars:
                return ts_vars[expr.name]
            return IntConst(0)
        elif isinstance(expr, BinOp):
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                      '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
                      '==': Op.EQ, '!=': Op.NEQ}
            l = ast_to_smt_prop(expr.left)
            r = ast_to_smt_prop(expr.right)
            op = op_map.get(expr.op, None)
            if op is None:
                raise ValueError(f"Unknown op: {expr.op}")
            sort = BOOL if op in (Op.LT, Op.GT, Op.LE, Op.GE, Op.EQ, Op.NEQ) else INT
            return App(op, [l, r], sort)
        return IntConst(0)

    prop_expr = prop_stmts[0].value if hasattr(prop_stmts[0], 'value') else None
    if prop_expr is None:
        raise ValueError("Could not parse property")

    prop_smt = ast_to_smt_prop(prop_expr)
    ts.set_property(prop_smt)

    return incremental_k_induction(ts, max_k, use_uniqueness=use_uniqueness)


def verify_loop_with_invariants(source, property_source, invariant_sources, max_k=20):
    """Verify with user-provided auxiliary invariants for strengthening.

    Args:
        source: C10 source with a while loop
        property_source: property expression
        invariant_sources: list of invariant expression strings
        max_k: maximum k

    Returns:
        KIndResult
    """
    ts, ts_vars = _extract_loop_ts(source)

    from stack_vm import lex, Parser, LetDecl, BinOp, IntLit, Var as ASTVar

    def parse_expr(expr_str):
        tokens = lex(f"let __p = ({expr_str});")
        stmts = Parser(tokens).parse().stmts
        expr = stmts[0].value
        return _ast_to_smt_with_vars(expr, ts_vars)

    def _ast_to_smt_with_vars(expr, vars_map):
        if isinstance(expr, IntLit):
            return IntConst(expr.value)
        elif isinstance(expr, ASTVar):
            if expr.name in vars_map:
                return vars_map[expr.name]
            return IntConst(0)
        elif isinstance(expr, BinOp):
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                      '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
                      '==': Op.EQ, '!=': Op.NEQ}
            l = _ast_to_smt_with_vars(expr.left, vars_map)
            r = _ast_to_smt_with_vars(expr.right, vars_map)
            op = op_map.get(expr.op, None)
            if op is None:
                raise ValueError(f"Unknown op: {expr.op}")
            sort = BOOL if op in (Op.LT, Op.GT, Op.LE, Op.GE, Op.EQ, Op.NEQ) else INT
            return App(op, [l, r], sort)
        return IntConst(0)

    prop_smt = parse_expr(property_source)
    ts.set_property(prop_smt)

    inv_terms = [parse_expr(inv) for inv in invariant_sources]

    return k_induction_with_strengthening(ts, max_k, invariants=inv_terms)


# ---------------------------------------------------------------------------
# Comparison with PDR
# ---------------------------------------------------------------------------

def compare_with_pdr(ts):
    """Compare k-induction result with PDR (V002) on the same system.

    Returns dict with both results and timing.
    """
    from pdr import check_ts

    # k-induction
    start = time.time()
    kind_result = incremental_k_induction(ts, max_k=20)
    kind_time = time.time() - start

    # PDR
    start = time.time()
    pdr_result = check_ts(ts)
    pdr_time = time.time() - start

    return {
        "k_induction": {
            "result": kind_result.result,
            "k": kind_result.k,
            "time": kind_time,
        },
        "pdr": {
            "result": pdr_result.result.value.upper(),
            "time": pdr_time,
        },
    }


# ---------------------------------------------------------------------------
# BMC-only mode (bounded model checking without induction)
# ---------------------------------------------------------------------------

def bmc_check(ts, max_depth=20):
    """Pure bounded model checking: check if property can be violated within max_depth steps.

    Unlike k-induction, BMC alone cannot prove safety -- it can only find bugs.
    Returns UNSAFE with counterexample, or UNKNOWN if no bug found within bounds.
    """
    start = time.time()

    for k in range(max_depth + 1):
        ok, trace = check_base_case(ts, k)
        if ok is False:
            elapsed = time.time() - start
            return KIndResult("UNSAFE", k=k, counterexample=trace,
                            stats={"time": elapsed, "depth": k})

    elapsed = time.time() - start
    return KIndResult("UNKNOWN", k=max_depth,
                     stats={"time": elapsed, "depth": max_depth})
