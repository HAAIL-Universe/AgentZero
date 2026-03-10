"""
V005: Abstract-Interpretation-Strengthened PDR

Composes:
  - C039 Abstract Interpreter (interval/sign/constant analysis)
  - V002 PDR/IC3 (unbounded model checking)
  - C010 Parser (AST)

The key insight: abstract interpretation is cheap (polynomial) and produces
variable bounds that serve as candidate inductive invariants. PDR is expensive
(SMT-backed) but produces exact results. By seeding PDR frames with abstract-
derived candidates, we can:
  1. Skip expensive clause-discovery work when abstract analysis already found it
  2. Accelerate convergence by starting with stronger frames
  3. Bridge program-level analysis (C039) and transition-system verification (V002)

Pipeline:
  Source (C010) -> Abstract Interpretation (C039) -> Candidate invariants
                -> Loop extraction -> Transition system
                -> PDR with seeded frames (V002) -> Verified / Counterexample
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import math

# Import C010 parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Program,
    IntLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr, ReturnStmt
)

# Import C039 abstract interpreter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, Interval, INTERVAL_TOP, INTERVAL_BOT,
    Sign, INF, NEG_INF, analyze as ai_analyze
)

# Import V002 PDR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
from pdr import (
    TransitionSystem, PDREngine, PDROutput, PDRResult, PDRStats,
    Counterexample, check_ts,
    _and, _or, _negate, _eq, _implies, _substitute, _collect_conjuncts,
    Var as SMTVar, IntConst, BoolConst, App, Op, BOOL, INT, SMTSolver
)


# ============================================================
# Result Types
# ============================================================

class VerifyResult(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class AIPDRStats:
    """Statistics about the AI-strengthened PDR run."""
    abstract_candidates: int = 0       # Candidates from abstract interp
    candidates_accepted: int = 0       # Candidates that were valid invariants
    candidates_rejected: int = 0       # Candidates that failed validity checks
    candidates_seeded: int = 0         # Candidates added to PDR frames
    pdr_stats: Optional[PDRStats] = None
    abstract_intervals: dict = field(default_factory=dict)  # var -> Interval


@dataclass
class AIPDROutput:
    """Result of AI-strengthened PDR verification."""
    result: VerifyResult
    invariant: Optional[list] = None
    counterexample: Optional[Counterexample] = None
    stats: AIPDRStats = field(default_factory=AIPDRStats)
    num_frames: int = 0
    abstract_warnings: list = field(default_factory=list)


# ============================================================
# AST -> Transition System Extraction
# ============================================================

def parse(source):
    """Parse C010 source into AST."""
    tokens = lex(source)
    parser = Parser(tokens)
    return parser.parse()


def _collect_vars_in_expr(expr):
    """Collect all variable names referenced in an expression."""
    if isinstance(expr, Var):
        return {expr.name}
    elif isinstance(expr, BinOp):
        return _collect_vars_in_expr(expr.left) | _collect_vars_in_expr(expr.right)
    elif isinstance(expr, UnaryOp):
        return _collect_vars_in_expr(expr.operand)
    elif isinstance(expr, CallExpr):
        result = set()
        for arg in expr.args:
            result |= _collect_vars_in_expr(arg)
        return result
    return set()


def _collect_vars_in_stmt(stmt):
    """Collect all variable names in a statement."""
    if isinstance(stmt, (LetDecl, Assign)):
        return {stmt.name} | _collect_vars_in_expr(stmt.value)
    elif isinstance(stmt, IfStmt):
        result = _collect_vars_in_expr(stmt.cond)
        result |= _collect_vars_in_stmt(stmt.then_body)
        if stmt.else_body:
            result |= _collect_vars_in_stmt(stmt.else_body)
        return result
    elif isinstance(stmt, WhileStmt):
        return _collect_vars_in_expr(stmt.cond) | _collect_vars_in_stmt(stmt.body)
    elif isinstance(stmt, Block):
        result = set()
        for s in stmt.stmts:
            result |= _collect_vars_in_stmt(s)
        return result
    elif isinstance(stmt, ReturnStmt):
        if stmt.value:
            return _collect_vars_in_expr(stmt.value)
        return set()
    return set()


def _collect_assigned_vars(stmt):
    """Collect variables that are assigned in a statement."""
    if isinstance(stmt, (LetDecl, Assign)):
        return {stmt.name}
    elif isinstance(stmt, IfStmt):
        result = _collect_assigned_vars(stmt.then_body)
        if stmt.else_body:
            result |= _collect_assigned_vars(stmt.else_body)
        return result
    elif isinstance(stmt, Block):
        result = set()
        for s in stmt.stmts:
            result |= _collect_assigned_vars(s)
        return result
    return set()


def _expr_to_smt(expr, var_lookup):
    """Convert a C010 expression AST to SMT term."""
    if isinstance(expr, IntLit):
        return IntConst(expr.value)
    elif isinstance(expr, BoolLit):
        return BoolConst(expr.value)
    elif isinstance(expr, Var):
        if expr.name in var_lookup:
            return var_lookup[expr.name]
        return SMTVar(expr.name, INT)
    elif isinstance(expr, UnaryOp):
        operand = _expr_to_smt(expr.operand, var_lookup)
        if expr.op == '-':
            return App(Op.SUB, [IntConst(0), operand], INT)
        elif expr.op == 'not':
            return _negate(operand)
        return operand
    elif isinstance(expr, BinOp):
        left = _expr_to_smt(expr.left, var_lookup)
        right = _expr_to_smt(expr.right, var_lookup)
        op_map = {
            '+': (Op.ADD, INT), '-': (Op.SUB, INT),
            '*': (Op.MUL, INT),
            '<': (Op.LT, BOOL), '>': (Op.GT, BOOL),
            '<=': (Op.LE, BOOL), '>=': (Op.GE, BOOL),
            '==': (Op.EQ, BOOL), '!=': (Op.NEQ, BOOL),
            'and': (Op.AND, BOOL), 'or': (Op.OR, BOOL),
        }
        if expr.op in op_map:
            op, sort = op_map[expr.op]
            return App(op, [left, right], sort)
        return IntConst(0)
    return IntConst(0)


def _body_to_transition(body, state_vars, ts):
    """
    Convert a loop body to a transition relation.
    Returns a formula relating current-state vars to primed vars.

    Handles assignments (x = expr) and conditionals (if-then-else).
    Variables not assigned keep their value (x' = x).
    """
    # Build variable lookup: current state vars
    var_lookup = {}
    for name in state_vars:
        var_lookup[name] = ts.var(name)

    # Process body to get next-state expressions
    next_state = {}  # var_name -> SMT expression for next state
    for name in state_vars:
        next_state[name] = ts.var(name)  # default: unchanged

    _process_body_stmts(body, var_lookup, next_state, state_vars, ts)

    # Build transition: conjunction of x' == next_state[x]
    conjuncts = []
    for name in state_vars:
        prime_var = ts.prime(name)
        conjuncts.append(_eq(prime_var, next_state[name]))

    return _and(*conjuncts)


def _process_body_stmts(body, var_lookup, next_state, state_vars, ts):
    """Process body statements to build next-state expressions."""
    if isinstance(body, Block):
        stmts = body.stmts
    elif isinstance(body, list):
        stmts = body
    else:
        stmts = [body]

    for stmt in stmts:
        if isinstance(stmt, (LetDecl, Assign)):
            if stmt.name in state_vars:
                # Build expression using CURRENT var_lookup
                expr_smt = _expr_to_smt(stmt.value, var_lookup)
                next_state[stmt.name] = expr_smt
                # Update var_lookup so subsequent assignments see this value
                var_lookup[stmt.name] = expr_smt
        elif isinstance(stmt, IfStmt):
            cond_smt = _expr_to_smt(stmt.cond, var_lookup)

            # Save current state
            saved_lookup = dict(var_lookup)
            saved_next = dict(next_state)

            # Process then branch
            then_lookup = dict(var_lookup)
            then_next = dict(next_state)
            _process_body_stmts(stmt.then_body, then_lookup, then_next, state_vars, ts)

            # Process else branch
            else_lookup = dict(saved_lookup)
            else_next = dict(saved_next)
            if stmt.else_body:
                _process_body_stmts(stmt.else_body, else_lookup, else_next, state_vars, ts)

            # Merge: ITE for each variable that differs
            for name in state_vars:
                if str(then_next[name]) != str(else_next[name]):
                    ite = App(Op.ITE, [cond_smt, then_next[name], else_next[name]], INT)
                    next_state[name] = ite
                    var_lookup[name] = ite
                else:
                    next_state[name] = then_next[name]
                    var_lookup[name] = then_next[name]


def extract_loop_ts(source, loop_index=0, property_expr=None):
    """
    Extract a transition system from a while loop in C10 source.

    Steps:
    1. Parse source, find the loop
    2. Collect pre-loop variable initializations as init formula
    3. Convert loop body to transition relation
    4. Use loop condition as implicit property (loop stays in valid states),
       or use provided property_expr

    Returns: (TransitionSystem, list_of_state_var_names)
    """
    program = parse(source)
    stmts = program.stmts

    # Find pre-loop assignments and the while loop
    pre_assignments = {}  # var_name -> init value (IntLit)
    loop_stmt = None
    loop_count = 0

    for stmt in stmts:
        if isinstance(stmt, WhileStmt):
            if loop_count == loop_index:
                loop_stmt = stmt
                break
            loop_count += 1
        elif isinstance(stmt, LetDecl):
            if isinstance(stmt.value, IntLit):
                pre_assignments[stmt.name] = stmt.value.value
            elif isinstance(stmt.value, BoolLit):
                pre_assignments[stmt.name] = 1 if stmt.value.value else 0
            elif isinstance(stmt.value, UnaryOp) and stmt.op == '-' and isinstance(stmt.value.operand, IntLit):
                pre_assignments[stmt.name] = -stmt.value.operand.value
            else:
                pre_assignments[stmt.name] = None  # non-constant init

    if loop_stmt is None:
        raise ValueError(f"No while loop found at index {loop_index}")

    # Determine state variables: all vars assigned in loop body + condition vars
    body_vars = _collect_assigned_vars(loop_stmt.body)
    cond_vars = _collect_vars_in_expr(loop_stmt.cond)
    state_var_names = sorted(body_vars | cond_vars)

    # Build transition system
    ts = TransitionSystem()
    for name in state_var_names:
        ts.add_int_var(name)

    # Build init formula from pre-assignments
    init_conjuncts = []
    for name in state_var_names:
        v = ts.var(name)
        if name in pre_assignments and pre_assignments[name] is not None:
            init_conjuncts.append(_eq(v, IntConst(pre_assignments[name])))

    if init_conjuncts:
        ts.set_init(_and(*init_conjuncts))
    else:
        ts.set_init(BoolConst(True))

    # Build transition relation from loop body
    trans = _body_to_transition(loop_stmt.body, state_var_names, ts)
    ts.set_trans(trans)

    # Property: use provided expression, or derive from loop condition
    if property_expr is not None:
        var_lookup = {name: ts.var(name) for name in state_var_names}
        prop = _expr_to_smt(property_expr, var_lookup)
        ts.set_property(prop)
    else:
        ts.set_property(BoolConst(True))

    return ts, state_var_names


# ============================================================
# Abstract Interpretation -> Candidate Invariants
# ============================================================

def extract_abstract_candidates(source, state_vars):
    """
    Run abstract interpretation on source and extract candidate invariants
    as SMT formulas over the state variables.

    Candidates come from:
    1. Finite interval bounds -> x >= lo AND x <= hi
    2. Sign information -> x >= 0, x <= 0, x > 0, x < 0
    3. Constant values -> x == c
    """
    result = ai_analyze(source)
    env = result['env']
    candidates = []
    intervals_found = {}

    for name in state_vars:
        interval = env.get_interval(name)
        sign = env.get_sign(name)
        const = env.get_const(name)

        intervals_found[name] = interval
        v = SMTVar(name, INT)

        # Constant propagation: strongest possible candidate
        from abstract_interpreter import ConstVal
        if isinstance(const, ConstVal) and isinstance(const.value, (int, float)):
            val = int(const.value)
            candidates.append(('const', name, _eq(v, IntConst(val))))

        # Finite interval bounds
        if not interval.is_bot() and not interval.is_top():
            if interval.lo != NEG_INF and not math.isinf(interval.lo):
                lo = int(interval.lo)
                candidates.append(('lower', name,
                                   App(Op.GE, [v, IntConst(lo)], BOOL)))
            if interval.hi != INF and not math.isinf(interval.hi):
                hi = int(interval.hi)
                candidates.append(('upper', name,
                                   App(Op.LE, [v, IntConst(hi)], BOOL)))

        # Sign-derived bounds (weaker, but sometimes interval is TOP while sign isn't)
        if sign == Sign.NON_NEG:
            candidates.append(('sign', name, App(Op.GE, [v, IntConst(0)], BOOL)))
        elif sign == Sign.POS:
            candidates.append(('sign', name, App(Op.GT, [v, IntConst(0)], BOOL)))
            candidates.append(('sign', name, App(Op.GE, [v, IntConst(0)], BOOL)))
        elif sign == Sign.NON_POS:
            candidates.append(('sign', name, App(Op.LE, [v, IntConst(0)], BOOL)))
        elif sign == Sign.NEG:
            candidates.append(('sign', name, App(Op.LT, [v, IntConst(0)], BOOL)))
            candidates.append(('sign', name, App(Op.LE, [v, IntConst(0)], BOOL)))
        elif sign == Sign.ZERO:
            candidates.append(('sign', name, _eq(v, IntConst(0))))

    return candidates, intervals_found, result.get('warnings', [])


def validate_candidate(ts, candidate_formula):
    """
    Check if a candidate formula is a valid inductive invariant for the
    transition system:
    1. Init => candidate  (holds initially)
    2. candidate AND Trans => candidate'  (preserved by transitions)

    Returns True if valid, False otherwise.
    """
    # Check 1: Init => candidate
    s = SMTSolver()
    for name, sort in ts.state_vars:
        if sort == INT:
            s.Int(name)
            s.Int(name + "'")
        else:
            s.Bool(name)
            s.Bool(name + "'")

    s.add(ts.init_formula)
    s.add(_negate(candidate_formula))
    if s.check().name == 'SAT':
        return False  # Init doesn't imply candidate

    # Check 2: candidate AND Trans => candidate'
    # Equivalent to: candidate AND Trans AND NOT(candidate') is UNSAT
    s2 = SMTSolver()
    for name, sort in ts.state_vars:
        if sort == INT:
            s2.Int(name)
            s2.Int(name + "'")
        else:
            s2.Bool(name)
            s2.Bool(name + "'")

    s2.add(candidate_formula)
    s2.add(ts.trans_formula)

    # Prime the candidate
    prime_map = {}
    for name, sort in ts.state_vars:
        prime_map[name] = SMTVar(name + "'", sort)
    primed_candidate = _substitute(candidate_formula, prime_map)
    s2.add(_negate(primed_candidate))

    if s2.check().name == 'SAT':
        return False  # Not inductive

    return True


def filter_and_validate_candidates(ts, candidates):
    """
    Filter candidates: keep only those that are valid inductive invariants.
    Returns (accepted, rejected) lists.
    """
    accepted = []
    rejected = []
    # Deduplicate by string representation
    seen = set()

    for kind, var_name, formula in candidates:
        formula_str = str(formula)
        if formula_str in seen:
            continue
        seen.add(formula_str)

        if validate_candidate(ts, formula):
            accepted.append((kind, var_name, formula))
        else:
            rejected.append((kind, var_name, formula))

    return accepted, rejected


# ============================================================
# Strengthened PDR Engine
# ============================================================

class AIPDREngine(PDREngine):
    """
    PDR engine with abstract-interpretation-derived frame seeding.

    Overrides PDR initialization to seed frames with validated
    candidate invariants from abstract interpretation.
    """

    def __init__(self, ts, seed_clauses=None, max_frames=100):
        super().__init__(ts, max_frames)
        self.seed_clauses = seed_clauses or []
        self.ai_stats = AIPDRStats()

    def check(self):
        """Run PDR with seeded frames."""
        ts = self.ts

        if ts.init_formula is None:
            raise ValueError("No initial state formula set")
        if ts.trans_formula is None:
            raise ValueError("No transition relation set")
        if ts.prop_formula is None:
            raise ValueError("No property formula set")

        # Step 0: Check if Init => Property
        sat, model = self._check_sat(ts.init_formula, _negate(ts.prop_formula))
        if sat:
            state = self._extract_model_state(model)
            return PDROutput(
                result=PDRResult.UNSAFE,
                counterexample=Counterexample(trace=[state], length=0),
                stats=self.stats,
                num_frames=0
            )

        # Initialize frames
        self.frames = [[], []]
        self.stats.frames_created = 2

        # SEED: add validated abstract-derived clauses to frame 1
        for clause in self.seed_clauses:
            clause_str = str(clause)
            if not any(str(c) == clause_str for c in self.frames[1]):
                self.frames[1].append(clause)
                self.ai_stats.candidates_seeded += 1

        # Run standard PDR loop
        for iteration in range(self.max_frames):
            k = len(self.frames) - 1

            fk = self._frame_formula(k)
            sat, model = self._check_sat(fk, _negate(ts.prop_formula))

            if sat:
                result = self._block_bad_states(k)
                if result is not None:
                    return PDROutput(
                        result=PDRResult.UNSAFE,
                        counterexample=result,
                        stats=self.stats,
                        num_frames=len(self.frames)
                    )
                continue

            # Propagation phase
            self.frames.append([])
            self.stats.frames_created += 1

            # Also seed new frame with abstract candidates
            for clause in self.seed_clauses:
                clause_str = str(clause)
                if not any(str(c) == clause_str for c in self.frames[-1]):
                    self.frames[-1].append(clause)

            fixpoint = self._propagate()
            if fixpoint is not None:
                invariant = list(self.frames[fixpoint])
                return PDROutput(
                    result=PDRResult.SAFE,
                    invariant=invariant,
                    stats=self.stats,
                    num_frames=len(self.frames)
                )

        return PDROutput(
            result=PDRResult.UNKNOWN,
            stats=self.stats,
            num_frames=len(self.frames)
        )


# ============================================================
# Main API
# ============================================================

def ai_pdr_check(ts, source=None, state_vars=None, max_frames=100):
    """
    Run PDR with abstract-interpretation strengthening on a transition system.

    If source and state_vars are provided, runs abstract interpretation to
    extract candidate invariants and seeds PDR frames with them.

    Args:
        ts: TransitionSystem to verify
        source: Optional C10 source code for abstract analysis
        state_vars: Optional list of state variable names
        max_frames: Maximum PDR iterations

    Returns: AIPDROutput
    """
    stats = AIPDRStats()
    abstract_warnings = []
    seed_clauses = []

    if source is not None and state_vars is not None:
        # Run abstract interpretation
        candidates, intervals, warnings = extract_abstract_candidates(source, state_vars)
        abstract_warnings = warnings
        stats.abstract_intervals = intervals
        stats.abstract_candidates = len(candidates)

        # Validate candidates against the transition system
        accepted, rejected = filter_and_validate_candidates(ts, candidates)
        stats.candidates_accepted = len(accepted)
        stats.candidates_rejected = len(rejected)

        seed_clauses = [formula for _, _, formula in accepted]

    # Run strengthened PDR
    engine = AIPDREngine(ts, seed_clauses=seed_clauses, max_frames=max_frames)
    pdr_output = engine.check()

    stats.pdr_stats = pdr_output.stats
    stats.candidates_seeded = engine.ai_stats.candidates_seeded

    result_map = {
        PDRResult.SAFE: VerifyResult.SAFE,
        PDRResult.UNSAFE: VerifyResult.UNSAFE,
        PDRResult.UNKNOWN: VerifyResult.UNKNOWN,
    }

    return AIPDROutput(
        result=result_map[pdr_output.result],
        invariant=pdr_output.invariant,
        counterexample=pdr_output.counterexample,
        stats=stats,
        num_frames=pdr_output.num_frames,
        abstract_warnings=abstract_warnings,
    )


def verify_loop(source, property_source=None, loop_index=0, max_frames=100):
    """
    High-level API: verify a property about a while loop in C10 source.

    Extracts the loop as a transition system, runs abstract interpretation
    for candidate invariants, then runs strengthened PDR.

    Args:
        source: C10 source code containing a while loop
        property_source: Optional property as C10 expression string.
                        If None, just checks that the loop doesn't violate
                        any internally-derived invariants.
        loop_index: Which loop to verify (0-indexed)
        max_frames: Maximum PDR iterations

    Returns: AIPDROutput
    """
    # Parse property expression if provided
    property_expr = None
    if property_source is not None:
        prop_tokens = lex(property_source + ";")
        prop_parser = Parser(prop_tokens)
        prop_program = prop_parser.parse()
        if prop_program.stmts:
            property_expr = prop_program.stmts[0]

    # Extract transition system
    ts, state_vars = extract_loop_ts(source, loop_index, property_expr)

    # Run AI-strengthened PDR
    return ai_pdr_check(ts, source=source, state_vars=state_vars,
                        max_frames=max_frames)


def verify_ts_with_hints(ts, hints=None, max_frames=100):
    """
    Verify a transition system with manually-provided invariant hints.

    Hints are SMT formulas that might be inductive invariants.
    They are validated and used to seed PDR frames.

    Args:
        ts: TransitionSystem to verify
        hints: List of SMT formula hints
        max_frames: Maximum PDR iterations

    Returns: AIPDROutput
    """
    stats = AIPDRStats()
    seed_clauses = []

    if hints:
        stats.abstract_candidates = len(hints)
        for hint in hints:
            if validate_candidate(ts, hint):
                seed_clauses.append(hint)
                stats.candidates_accepted += 1
            else:
                stats.candidates_rejected += 1

    engine = AIPDREngine(ts, seed_clauses=seed_clauses, max_frames=max_frames)
    pdr_output = engine.check()

    stats.pdr_stats = pdr_output.stats
    stats.candidates_seeded = engine.ai_stats.candidates_seeded

    result_map = {
        PDRResult.SAFE: VerifyResult.SAFE,
        PDRResult.UNSAFE: VerifyResult.UNSAFE,
        PDRResult.UNKNOWN: VerifyResult.UNKNOWN,
    }

    return AIPDROutput(
        result=result_map[pdr_output.result],
        invariant=pdr_output.invariant,
        counterexample=pdr_output.counterexample,
        stats=stats,
        num_frames=pdr_output.num_frames,
    )


def compare_pdr_performance(ts, source=None, state_vars=None, max_frames=100):
    """
    Run both standard PDR and AI-strengthened PDR, returning both results
    for comparison.

    Returns: (standard_output, strengthened_output)
    """
    # Standard PDR
    std_output = check_ts(ts, max_frames=max_frames)

    # AI-strengthened PDR
    ai_output = ai_pdr_check(ts, source=source, state_vars=state_vars,
                             max_frames=max_frames)

    return std_output, ai_output
