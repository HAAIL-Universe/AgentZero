"""V066: Markov Chain Verification

Formal verification of Markov chain properties using SMT solving
and machine-checkable proof certificates.

Composes:
  - V065 (Markov chain analysis) for chain construction and numerical analysis
  - C037 (SMT solver) for formal property verification
  - V044 (proof certificates) for machine-checkable proofs

Key capabilities:
  1. SMT-verified steady-state bounds (prove pi_i >= p or pi_i <= p)
  2. SMT-verified absorption probability bounds
  3. SMT-verified hitting time bounds
  4. SMT-verified transition matrix properties (stochasticity, irreducibility)
  5. Proof certificates for all verified properties
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from markov_chain import (
    MarkovChain, make_chain, classify_states, StateType,
    communication_classes, steady_state, steady_state_exact,
    absorption_probabilities, expected_hitting_time,
    analyze_chain, ChainAnalysis, verify_absorption,
    verify_hitting_time_bound, verify_steady_state_bound,
    gambler_ruin_chain, random_walk_chain
)
from proof_certificates import (
    ProofCertificate, ProofObligation, ProofKind, CertStatus,
    check_certificate, combine_certificates, save_certificate, load_certificate
)
from smt_solver import SMTSolver, SMTResult, Op, App, Var, IntConst, BoolConst, INT, BOOL


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class Verdict(Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of verifying a single Markov chain property."""
    verdict: Verdict
    property_name: str
    description: str
    numerical_value: Optional[float] = None
    bound: Optional[float] = None
    counterexample: Optional[dict] = None
    certificate: Optional[ProofCertificate] = None
    details: dict = field(default_factory=dict)


@dataclass
class ChainVerificationResult:
    """Result of verifying multiple properties of a Markov chain."""
    chain: MarkovChain
    results: List[VerificationResult] = field(default_factory=list)
    certificate: Optional[ProofCertificate] = None

    @property
    def all_verified(self) -> bool:
        return all(r.verdict == Verdict.VERIFIED for r in self.results)

    @property
    def summary(self) -> str:
        lines = [f"Chain Verification ({len(self.results)} properties):"]
        for r in self.results:
            status = r.verdict.value.upper()
            lines.append(f"  [{status}] {r.property_name}: {r.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rational arithmetic helpers (exact, no floating-point)
# ---------------------------------------------------------------------------

def _to_fraction(val: float, max_denom: int = 10000) -> Fraction:
    """Convert float to exact fraction for SMT encoding."""
    return Fraction(val).limit_denominator(max_denom)


def _encode_fraction_constraint(solver, num_var, frac: Fraction) -> 'Term':
    """Encode num_var >= frac as num_var * denom >= numer (integer arithmetic)."""
    # num_var represents a probability scaled by a common denominator
    numer = IntConst(frac.numerator)
    denom = IntConst(frac.denominator)
    return App(Op.GE, [App(Op.MUL, [num_var, denom], INT), numer], BOOL)


# ---------------------------------------------------------------------------
# Core: Stochasticity verification
# ---------------------------------------------------------------------------

def verify_stochastic(mc: MarkovChain) -> VerificationResult:
    """Verify that the transition matrix is stochastic (rows sum to 1, all >= 0).

    Uses exact rational arithmetic to check without floating-point error.
    """
    obligations = []
    all_valid = True

    for i in range(mc.n_states):
        # Check non-negativity
        for j in range(mc.n_states):
            if mc.transition[i][j] < 0:
                all_valid = False
                obligations.append(ProofObligation(
                    name=f"nonneg_{i}_{j}",
                    description=f"P[{i}][{j}] >= 0",
                    formula_str=f"P[{i}][{j}] = {mc.transition[i][j]} >= 0",
                    formula_smt="",
                    status=CertStatus.INVALID,
                    counterexample={"row": i, "col": j, "value": mc.transition[i][j]}
                ))

        # Check row sum = 1 (exact rational)
        row_sum = sum(Fraction(v).limit_denominator(10000) for v in mc.transition[i])
        row_ok = row_sum == Fraction(1)
        obligations.append(ProofObligation(
            name=f"row_sum_{i}",
            description=f"Row {i} sums to 1",
            formula_str=f"sum(P[{i}]) = {float(row_sum)} == 1",
            formula_smt="",
            status=CertStatus.VALID if row_ok else CertStatus.INVALID,
            counterexample=None if row_ok else {"row": i, "sum": float(row_sum)}
        ))
        if not row_ok:
            all_valid = False

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim="Transition matrix is stochastic",
        source=None,
        obligations=obligations,
        metadata={"n_states": mc.n_states},
        status=CertStatus.VALID if all_valid else CertStatus.INVALID,
        timestamp=""
    )

    return VerificationResult(
        verdict=Verdict.VERIFIED if all_valid else Verdict.REFUTED,
        property_name="stochasticity",
        description="Transition matrix is row-stochastic (rows sum to 1, all entries >= 0)",
        certificate=cert,
        details={"n_states": mc.n_states}
    )


# ---------------------------------------------------------------------------
# Core: Steady-state bound verification via SMT
# ---------------------------------------------------------------------------

def verify_steady_state_smt(mc: MarkovChain, state: int,
                            lower_bound: Optional[float] = None,
                            upper_bound: Optional[float] = None) -> VerificationResult:
    """Verify steady-state probability bounds using SMT.

    Encodes the steady-state equations pi * P = pi, sum(pi) = 1
    in integer arithmetic (scaled by common denominator) and checks
    whether the bound holds.

    Args:
        mc: Markov chain
        state: State index to check
        lower_bound: If set, verify pi[state] >= lower_bound
        upper_bound: If set, verify pi[state] <= upper_bound
    """
    if lower_bound is None and upper_bound is None:
        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            property_name="steady_state_bound",
            description="No bound specified",
        )

    n = mc.n_states

    # Convert transition matrix to exact fractions
    P_frac = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(_to_fraction(mc.transition[i][j]))
        P_frac.append(row)

    # Find common denominator for all transition probabilities
    common_denom = 1
    for i in range(n):
        for j in range(n):
            common_denom = _lcm(common_denom, P_frac[i][j].denominator)

    # Scale: P_int[i][j] = P_frac[i][j] * common_denom (integer)
    P_int = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(P_frac[i][j] * common_denom))
        P_int.append(row)

    # SMT encoding: pi_i are non-negative integers representing
    # the steady-state distribution scaled by some factor S.
    # Equations: for each j, sum_i(pi_i * P_int[i][j]) = pi_j * common_denom
    # Normalization: sum(pi_i) = S (S is a free positive integer)
    solver = SMTSolver()
    pi = [solver.Int(f"pi_{i}") for i in range(n)]
    S = solver.Int("S")

    # pi_i >= 0
    for i in range(n):
        solver.add(pi[i] >= IntConst(0))

    # S > 0 (non-trivial solution)
    solver.add(S > IntConst(0))

    # sum(pi_i) = S
    total = pi[0]
    for i in range(1, n):
        total = total + pi[i]
    solver.add(total == S)

    # Steady-state equations: pi * P = pi (in scaled integers)
    # For each column j: sum_i(pi_i * P_int[i][j]) = pi_j * common_denom
    for j in range(n):
        lhs = App(Op.MUL, [pi[0], IntConst(P_int[0][j])], INT)
        for i in range(1, n):
            term = App(Op.MUL, [pi[i], IntConst(P_int[i][j])], INT)
            lhs = lhs + term
        rhs = App(Op.MUL, [pi[j], IntConst(common_denom)], INT)
        solver.add(lhs == rhs)

    # Now add the NEGATION of the property to check
    # If lower_bound: we want to prove pi[state]/S >= lower_bound
    # i.e., pi[state] * lb_denom >= lb_numer * S
    # Negate: pi[state] * lb_denom < lb_numer * S
    obligations = []

    if lower_bound is not None:
        lb_frac = _to_fraction(lower_bound)
        # Property: pi[state] * lb_denom >= lb_numer * S
        # Negate: pi[state] * lb_denom < lb_numer * S
        prop_lhs = App(Op.MUL, [pi[state], IntConst(lb_frac.denominator)], INT)
        prop_rhs = App(Op.MUL, [S, IntConst(lb_frac.numerator)], INT)

        solver.push()
        solver.add(App(Op.LT, [prop_lhs, prop_rhs], BOOL))
        result = solver.check()
        model = solver.model() if result == SMTResult.SAT else None
        solver.pop()

        if result == SMTResult.UNSAT:
            # No counterexample -> property holds
            ob_status = CertStatus.VALID
        elif result == SMTResult.SAT:
            ob_status = CertStatus.INVALID
        else:
            ob_status = CertStatus.UNKNOWN

        obligations.append(ProofObligation(
            name="steady_state_lower_bound",
            description=f"pi[{state}] >= {lower_bound}",
            formula_str=f"pi[{state}] / sum(pi) >= {lower_bound}",
            formula_smt="",
            status=ob_status,
            counterexample=model if ob_status == CertStatus.INVALID else None
        ))

    if upper_bound is not None:
        ub_frac = _to_fraction(upper_bound)
        # Property: pi[state] * ub_denom <= ub_numer * S
        # Negate: pi[state] * ub_denom > ub_numer * S
        prop_lhs = App(Op.MUL, [pi[state], IntConst(ub_frac.denominator)], INT)
        prop_rhs = App(Op.MUL, [S, IntConst(ub_frac.numerator)], INT)

        solver.push()
        solver.add(App(Op.GT, [prop_lhs, prop_rhs], BOOL))
        result = solver.check()
        model = solver.model() if result == SMTResult.SAT else None
        solver.pop()

        if result == SMTResult.UNSAT:
            ob_status = CertStatus.VALID
        elif result == SMTResult.SAT:
            ob_status = CertStatus.INVALID
        else:
            ob_status = CertStatus.UNKNOWN

        obligations.append(ProofObligation(
            name="steady_state_upper_bound",
            description=f"pi[{state}] <= {upper_bound}",
            formula_str=f"pi[{state}] / sum(pi) <= {upper_bound}",
            formula_smt="",
            status=ob_status,
            counterexample=model if ob_status == CertStatus.INVALID else None
        ))

    all_valid = all(ob.status == CertStatus.VALID for ob in obligations)
    any_invalid = any(ob.status == CertStatus.INVALID for ob in obligations)

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"Steady-state bound for state {state}",
        source=None,
        obligations=obligations,
        metadata={"state": state, "lower_bound": lower_bound, "upper_bound": upper_bound,
                  "n_states": n, "common_denom": common_denom},
        status=CertStatus.VALID if all_valid else (CertStatus.INVALID if any_invalid else CertStatus.UNKNOWN),
        timestamp=""
    )

    # Get numerical value for reference
    ss = steady_state_exact(mc)
    num_val = ss[state] if ss else None

    if all_valid:
        verdict = Verdict.VERIFIED
    elif any_invalid:
        verdict = Verdict.REFUTED
    else:
        verdict = Verdict.UNKNOWN

    desc_parts = []
    if lower_bound is not None:
        desc_parts.append(f"pi[{state}] >= {lower_bound}")
    if upper_bound is not None:
        desc_parts.append(f"pi[{state}] <= {upper_bound}")

    return VerificationResult(
        verdict=verdict,
        property_name="steady_state_bound",
        description=" AND ".join(desc_parts),
        numerical_value=num_val,
        bound=lower_bound if lower_bound is not None else upper_bound,
        certificate=cert,
        details={"common_denom": common_denom}
    )


# ---------------------------------------------------------------------------
# Core: Absorption probability verification via SMT
# ---------------------------------------------------------------------------

def verify_absorption_smt(mc: MarkovChain, start: int, target: int,
                          lower_bound: Optional[float] = None,
                          upper_bound: Optional[float] = None) -> VerificationResult:
    """Verify absorption probability bounds using SMT.

    Encodes the absorption equations:
      b[i] = P[i][target] + sum_{j in transient} P[i][j] * b[j]
      b[target] = 1, b[other_absorbing] = 0
    in integer arithmetic and checks bounds.
    """
    if lower_bound is None and upper_bound is None:
        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            property_name="absorption_bound",
            description="No bound specified",
        )

    n = mc.n_states
    state_types = classify_states(mc)
    absorbing = [i for i in range(n) if state_types[i] == StateType.ABSORBING]
    transient = [i for i in range(n) if state_types[i] == StateType.TRANSIENT]

    if target not in absorbing:
        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            property_name="absorption_bound",
            description=f"State {target} is not absorbing",
        )

    # Convert to exact fractions
    P_frac = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(_to_fraction(mc.transition[i][j]))
        P_frac.append(row)

    common_denom = 1
    for i in range(n):
        for j in range(n):
            common_denom = _lcm(common_denom, P_frac[i][j].denominator)

    P_int = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(P_frac[i][j] * common_denom))
        P_int.append(row)

    # SMT encoding: b_i are non-negative integers, D is the common scale
    # b[target] = D, b[other_absorbing] = 0
    # For transient i: b[i] * common_denom = P_int[i][target] * D + sum_{j in transient} P_int[i][j] * b[j]
    solver = SMTSolver()
    b = {}
    for i in transient:
        b[i] = solver.Int(f"b_{i}")
        solver.add(b[i] >= IntConst(0))

    D = solver.Int("D")
    solver.add(D > IntConst(0))

    # For absorbing states
    b[target] = D
    for a in absorbing:
        if a != target:
            b[a] = IntConst(0)

    # Absorption equations for transient states
    for i in transient:
        # b[i] * common_denom = sum_j P_int[i][j] * b[j]
        lhs = App(Op.MUL, [b[i], IntConst(common_denom)], INT)

        # RHS: sum over all states j of P_int[i][j] * b[j]
        rhs_terms = []
        for j in range(n):
            if P_int[i][j] == 0:
                continue
            if j == target:
                # P_int[i][target] * D
                rhs_terms.append(App(Op.MUL, [IntConst(P_int[i][j]), D], INT))
            elif j in b:
                rhs_terms.append(App(Op.MUL, [IntConst(P_int[i][j]), b[j]], INT))
            # else: absorbing state != target -> b[j] = 0, skip

        if not rhs_terms:
            rhs = IntConst(0)
        else:
            rhs = rhs_terms[0]
            for t in rhs_terms[1:]:
                rhs = rhs + t

        solver.add(lhs == rhs)

    # Check bounds on b[start] / D
    obligations = []

    if start in absorbing:
        # Absorbing state: absorption prob is 1 if start == target, else 0
        actual = 1.0 if start == target else 0.0
        lb_ok = lower_bound is None or actual >= lower_bound - 1e-9
        ub_ok = upper_bound is None or actual <= upper_bound + 1e-9

        if lower_bound is not None:
            obligations.append(ProofObligation(
                name="absorption_lower_bound",
                description=f"b[{start}->{target}] >= {lower_bound}",
                formula_str=f"b[{start}] = {actual} >= {lower_bound}",
                formula_smt="",
                status=CertStatus.VALID if lb_ok else CertStatus.INVALID,
            ))
        if upper_bound is not None:
            obligations.append(ProofObligation(
                name="absorption_upper_bound",
                description=f"b[{start}->{target}] <= {upper_bound}",
                formula_str=f"b[{start}] = {actual} <= {upper_bound}",
                formula_smt="",
                status=CertStatus.VALID if ub_ok else CertStatus.INVALID,
            ))
    else:
        # Transient start state -- use SMT
        if lower_bound is not None:
            lb_frac = _to_fraction(lower_bound)
            # b[start] / D >= lb => b[start] * lb_denom >= lb_numer * D
            prop_lhs = App(Op.MUL, [b[start], IntConst(lb_frac.denominator)], INT)
            prop_rhs = App(Op.MUL, [D, IntConst(lb_frac.numerator)], INT)

            solver.push()
            solver.add(App(Op.LT, [prop_lhs, prop_rhs], BOOL))
            result = solver.check()
            model = solver.model() if result == SMTResult.SAT else None
            solver.pop()

            ob_status = CertStatus.VALID if result == SMTResult.UNSAT else (
                CertStatus.INVALID if result == SMTResult.SAT else CertStatus.UNKNOWN)
            obligations.append(ProofObligation(
                name="absorption_lower_bound",
                description=f"b[{start}->{target}] >= {lower_bound}",
                formula_str=f"b[{start}] / D >= {lower_bound}",
                formula_smt="",
                status=ob_status,
                counterexample=model if ob_status == CertStatus.INVALID else None
            ))

        if upper_bound is not None:
            ub_frac = _to_fraction(upper_bound)
            prop_lhs = App(Op.MUL, [b[start], IntConst(ub_frac.denominator)], INT)
            prop_rhs = App(Op.MUL, [D, IntConst(ub_frac.numerator)], INT)

            solver.push()
            solver.add(App(Op.GT, [prop_lhs, prop_rhs], BOOL))
            result = solver.check()
            model = solver.model() if result == SMTResult.SAT else None
            solver.pop()

            ob_status = CertStatus.VALID if result == SMTResult.UNSAT else (
                CertStatus.INVALID if result == SMTResult.SAT else CertStatus.UNKNOWN)
            obligations.append(ProofObligation(
                name="absorption_upper_bound",
                description=f"b[{start}->{target}] <= {upper_bound}",
                formula_str=f"b[{start}] / D <= {upper_bound}",
                formula_smt="",
                status=ob_status,
                counterexample=model if ob_status == CertStatus.INVALID else None
            ))

    all_valid = all(ob.status == CertStatus.VALID for ob in obligations)
    any_invalid = any(ob.status == CertStatus.INVALID for ob in obligations)

    # Numerical reference
    abs_probs = absorption_probabilities(mc)
    num_val = abs_probs.get(target, [0.0] * n)[start] if abs_probs else None

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"Absorption probability bound: {start} -> {target}",
        source=None,
        obligations=obligations,
        metadata={"start": start, "target": target, "lower_bound": lower_bound,
                  "upper_bound": upper_bound, "n_states": n},
        status=CertStatus.VALID if all_valid else (CertStatus.INVALID if any_invalid else CertStatus.UNKNOWN),
        timestamp=""
    )

    if all_valid:
        verdict = Verdict.VERIFIED
    elif any_invalid:
        verdict = Verdict.REFUTED
    else:
        verdict = Verdict.UNKNOWN

    desc_parts = []
    if lower_bound is not None:
        desc_parts.append(f"P(absorb {start}->{target}) >= {lower_bound}")
    if upper_bound is not None:
        desc_parts.append(f"P(absorb {start}->{target}) <= {upper_bound}")

    return VerificationResult(
        verdict=verdict,
        property_name="absorption_bound",
        description=" AND ".join(desc_parts),
        numerical_value=num_val,
        bound=lower_bound if lower_bound is not None else upper_bound,
        certificate=cert,
    )


# ---------------------------------------------------------------------------
# Core: Hitting time bound verification via SMT
# ---------------------------------------------------------------------------

def verify_hitting_time_smt(mc: MarkovChain, start: int, target: int,
                            max_steps: float) -> VerificationResult:
    """Verify expected hitting time bound using SMT.

    Encodes h[i] = 1 + sum_j P[i][j] * h[j] for i != target, h[target] = 0
    in integer arithmetic and checks h[start] / D <= max_steps.
    """
    n = mc.n_states

    # Convert to fractions
    P_frac = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(_to_fraction(mc.transition[i][j]))
        P_frac.append(row)

    common_denom = 1
    for i in range(n):
        for j in range(n):
            common_denom = _lcm(common_denom, P_frac[i][j].denominator)

    P_int = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(int(P_frac[i][j] * common_denom))
        P_int.append(row)

    # SMT: h_i are non-negative integers, D is scale
    # h[target] = 0
    # For i != target: h[i] * common_denom = D * common_denom + sum_j P_int[i][j] * h[j]
    # (this encodes h[i] = 1 + sum_j P[i][j] * h[j] scaled by D)
    solver = SMTSolver()
    h = {}
    non_target = [i for i in range(n) if i != target]

    D = solver.Int("D")
    solver.add(D > IntConst(0))

    for i in non_target:
        h[i] = solver.Int(f"h_{i}")
        solver.add(h[i] >= IntConst(0))

    h[target] = IntConst(0)

    for i in non_target:
        # h[i] * common_denom = D * common_denom + sum_j P_int[i][j] * h[j]
        lhs = App(Op.MUL, [h[i], IntConst(common_denom)], INT)

        rhs = App(Op.MUL, [D, IntConst(common_denom)], INT)  # the "1" term scaled
        for j in range(n):
            if P_int[i][j] == 0:
                continue
            term = App(Op.MUL, [IntConst(P_int[i][j]), h[j]], INT)
            rhs = rhs + term

        solver.add(lhs == rhs)

    # Check h[start] / D <= max_steps
    # i.e., h[start] <= max_steps * D
    # Negate: h[start] > max_steps * D
    ms_frac = _to_fraction(max_steps)

    if start == target:
        # h[target] = 0 <= anything positive
        obligations = [ProofObligation(
            name="hitting_time_bound",
            description=f"E[T({start}->{target})] <= {max_steps}",
            formula_str=f"h[{start}] = 0 <= {max_steps}",
            formula_smt="",
            status=CertStatus.VALID if max_steps >= 0 else CertStatus.INVALID,
        )]
    else:
        # h[start] * ms_denom <= ms_numer * D
        # Negate: h[start] * ms_denom > ms_numer * D
        prop_lhs = App(Op.MUL, [h[start], IntConst(ms_frac.denominator)], INT)
        prop_rhs = App(Op.MUL, [D, IntConst(ms_frac.numerator)], INT)

        solver.push()
        solver.add(App(Op.GT, [prop_lhs, prop_rhs], BOOL))
        result = solver.check()
        model = solver.model() if result == SMTResult.SAT else None
        solver.pop()

        ob_status = CertStatus.VALID if result == SMTResult.UNSAT else (
            CertStatus.INVALID if result == SMTResult.SAT else CertStatus.UNKNOWN)
        obligations = [ProofObligation(
            name="hitting_time_bound",
            description=f"E[T({start}->{target})] <= {max_steps}",
            formula_str=f"h[{start}] / D <= {max_steps}",
            formula_smt="",
            status=ob_status,
            counterexample=model if ob_status == CertStatus.INVALID else None
        )]

    all_valid = all(ob.status == CertStatus.VALID for ob in obligations)
    any_invalid = any(ob.status == CertStatus.INVALID for ob in obligations)

    # Numerical reference
    ht = expected_hitting_time(mc, target)
    num_val = ht[start] if ht else None

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"Hitting time bound: {start} -> {target} <= {max_steps}",
        source=None,
        obligations=obligations,
        metadata={"start": start, "target": target, "max_steps": max_steps, "n_states": n},
        status=CertStatus.VALID if all_valid else (CertStatus.INVALID if any_invalid else CertStatus.UNKNOWN),
        timestamp=""
    )

    verdict = Verdict.VERIFIED if all_valid else (Verdict.REFUTED if any_invalid else Verdict.UNKNOWN)

    return VerificationResult(
        verdict=verdict,
        property_name="hitting_time_bound",
        description=f"E[T({start}->{target})] <= {max_steps}",
        numerical_value=num_val,
        bound=max_steps,
        certificate=cert,
    )


# ---------------------------------------------------------------------------
# Core: Irreducibility verification
# ---------------------------------------------------------------------------

def verify_irreducible(mc: MarkovChain) -> VerificationResult:
    """Verify that the chain is irreducible (single communication class)."""
    classes = communication_classes(mc)
    is_irr = len(classes) == 1

    obligations = [ProofObligation(
        name="irreducibility",
        description="Single communication class",
        formula_str=f"|communication_classes| = {len(classes)} == 1",
        formula_smt="",
        status=CertStatus.VALID if is_irr else CertStatus.INVALID,
        counterexample=None if is_irr else {"classes": [sorted(c) for c in classes]}
    )]

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim="Chain is irreducible",
        source=None,
        obligations=obligations,
        metadata={"n_classes": len(classes)},
        status=CertStatus.VALID if is_irr else CertStatus.INVALID,
        timestamp=""
    )

    return VerificationResult(
        verdict=Verdict.VERIFIED if is_irr else Verdict.REFUTED,
        property_name="irreducibility",
        description="Chain is irreducible (single communication class)",
        certificate=cert,
        details={"classes": [sorted(c) for c in classes]}
    )


# ---------------------------------------------------------------------------
# Core: State classification verification
# ---------------------------------------------------------------------------

def verify_state_type(mc: MarkovChain, state: int,
                      expected_type: StateType) -> VerificationResult:
    """Verify that a state has the expected type (transient/recurrent/absorbing)."""
    types = classify_states(mc)
    actual = types[state]
    ok = actual == expected_type

    obligations = [ProofObligation(
        name=f"state_type_{state}",
        description=f"State {state} is {expected_type.value}",
        formula_str=f"type({state}) = {actual.value} == {expected_type.value}",
        formula_smt="",
        status=CertStatus.VALID if ok else CertStatus.INVALID,
        counterexample=None if ok else {"actual": actual.value, "expected": expected_type.value}
    )]

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"State {state} is {expected_type.value}",
        source=None,
        obligations=obligations,
        metadata={"state": state, "expected": expected_type.value, "actual": actual.value},
        status=CertStatus.VALID if ok else CertStatus.INVALID,
        timestamp=""
    )

    return VerificationResult(
        verdict=Verdict.VERIFIED if ok else Verdict.REFUTED,
        property_name="state_type",
        description=f"State {state} is {expected_type.value}",
        certificate=cert,
    )


# ---------------------------------------------------------------------------
# Core: Steady-state uniqueness/existence verification via SMT
# ---------------------------------------------------------------------------

def verify_steady_state_unique(mc: MarkovChain) -> VerificationResult:
    """Verify that the steady-state distribution is unique.

    Uses structural proof: irreducible chain => unique steady-state distribution.
    Also checks via rank of (P^T - I) system: if nullity is 1, unique up to scaling.

    Note: SMT-based uniqueness requires nonlinear arithmetic (pi_i * S2 vs sigma_i * S1)
    which C037's LIA solver cannot handle. We use algebraic rank analysis instead.
    """
    n = mc.n_states
    classes = communication_classes(mc)
    is_irr = len(classes) == 1

    obligations = []

    if is_irr:
        # Irreducible => unique stationary distribution (Perron-Frobenius theorem)
        obligations.append(ProofObligation(
            name="irreducibility_check",
            description="Chain is irreducible (single communication class)",
            formula_str=f"|communication_classes| = {len(classes)} == 1",
            formula_smt="",
            status=CertStatus.VALID,
        ))
        obligations.append(ProofObligation(
            name="uniqueness_by_perron_frobenius",
            description="Irreducible => unique stationary distribution (Perron-Frobenius)",
            formula_str="irreducible(P) => unique(pi: pi*P = pi, sum(pi)=1)",
            formula_smt="",
            status=CertStatus.VALID,
        ))
        verdict = Verdict.VERIFIED
        cert_status = CertStatus.VALID
    else:
        # Check rank of (P^T - I): if nullity > 1, multiple solutions exist
        # Build (P^T - I) matrix and compute rank via Gaussian elimination
        # with exact rational arithmetic
        matrix = []
        for j in range(n):
            row = []
            for i in range(n):
                val = Fraction(mc.transition[i][j]).limit_denominator(10000)
                if i == j:
                    val -= 1
                row.append(val)
            matrix.append(row)

        rank = _fraction_rank(matrix, n)
        nullity = n - rank

        if nullity == 1:
            # Exactly one-dimensional null space => unique (up to normalization)
            obligations.append(ProofObligation(
                name="rank_analysis",
                description=f"rank(P^T - I) = {rank}, nullity = {nullity}",
                formula_str=f"nullity(P^T - I) = {nullity} == 1",
                formula_smt="",
                status=CertStatus.VALID,
            ))
            verdict = Verdict.VERIFIED
            cert_status = CertStatus.VALID
        else:
            # nullity > 1 => multiple independent stationary distributions
            obligations.append(ProofObligation(
                name="rank_analysis",
                description=f"rank(P^T - I) = {rank}, nullity = {nullity} > 1",
                formula_str=f"nullity(P^T - I) = {nullity} > 1",
                formula_smt="",
                status=CertStatus.INVALID,
                counterexample={"rank": rank, "nullity": nullity,
                                "classes": [sorted(c) for c in classes]}
            ))
            verdict = Verdict.REFUTED
            cert_status = CertStatus.INVALID

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim="Steady-state distribution is unique",
        source=None,
        obligations=obligations,
        metadata={"n_states": n, "n_classes": len(classes)},
        status=cert_status,
        timestamp=""
    )

    return VerificationResult(
        verdict=verdict,
        property_name="steady_state_uniqueness",
        description="Steady-state distribution is unique",
        certificate=cert,
    )


def _fraction_rank(matrix: List[List[Fraction]], n: int) -> int:
    """Compute rank of an n x n Fraction matrix via Gaussian elimination."""
    # Work on a copy
    m = [row[:] for row in matrix]
    rank = 0
    for col in range(n):
        # Find pivot
        pivot_row = None
        for row in range(rank, n):
            if m[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap
        m[rank], m[pivot_row] = m[pivot_row], m[rank]
        # Eliminate
        pivot_val = m[rank][col]
        for row in range(n):
            if row == rank:
                continue
            if m[row][col] != 0:
                factor = m[row][col] / pivot_val
                for c in range(n):
                    m[row][c] -= factor * m[rank][c]
        rank += 1
    return rank


# ---------------------------------------------------------------------------
# Core: Reachability verification
# ---------------------------------------------------------------------------

def verify_reachability(mc: MarkovChain, source: int, target: int,
                        max_steps: Optional[int] = None) -> VerificationResult:
    """Verify that target is reachable from source (optionally within max_steps)."""
    n = mc.n_states

    if max_steps is None:
        max_steps = n  # BFS can reach any state in at most n-1 steps

    # BFS reachability
    visited = {source}
    frontier = {source}
    step = 0

    while frontier and step < max_steps:
        next_frontier = set()
        for s in frontier:
            for j in range(n):
                if mc.transition[s][j] > 0 and j not in visited:
                    visited.add(j)
                    next_frontier.add(j)
        frontier = next_frontier
        step += 1

    reachable = target in visited

    obligations = [ProofObligation(
        name=f"reachability_{source}_{target}",
        description=f"State {target} reachable from {source}" +
                    (f" within {max_steps} steps" if max_steps < n else ""),
        formula_str=f"reachable({source}, {target}) = {reachable}",
        formula_smt="",
        status=CertStatus.VALID if reachable else CertStatus.INVALID,
    )]

    cert = ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"State {target} reachable from {source}",
        source=None,
        obligations=obligations,
        metadata={"source": source, "target": target, "max_steps": max_steps},
        status=CertStatus.VALID if reachable else CertStatus.INVALID,
        timestamp=""
    )

    return VerificationResult(
        verdict=Verdict.VERIFIED if reachable else Verdict.REFUTED,
        property_name="reachability",
        description=f"State {target} reachable from {source}",
        certificate=cert,
    )


# ---------------------------------------------------------------------------
# High-level: Full chain verification
# ---------------------------------------------------------------------------

def verify_chain(mc: MarkovChain,
                 properties: Optional[List[dict]] = None) -> ChainVerificationResult:
    """Verify multiple properties of a Markov chain.

    Args:
        mc: The Markov chain
        properties: List of property dicts. Each dict has:
            - "type": one of "stochastic", "irreducible", "state_type",
                      "steady_state", "absorption", "hitting_time",
                      "reachability", "uniqueness"
            - Additional keys depending on type (state, lower_bound, etc.)

    If properties is None, runs default checks (stochasticity + irreducibility).
    """
    result = ChainVerificationResult(chain=mc)

    if properties is None:
        properties = [
            {"type": "stochastic"},
            {"type": "irreducible"},
        ]

    certs = []
    for prop in properties:
        ptype = prop["type"]

        if ptype == "stochastic":
            r = verify_stochastic(mc)
        elif ptype == "irreducible":
            r = verify_irreducible(mc)
        elif ptype == "state_type":
            r = verify_state_type(mc, prop["state"], StateType(prop["expected"]))
        elif ptype == "steady_state":
            r = verify_steady_state_smt(mc, prop["state"],
                                        prop.get("lower_bound"), prop.get("upper_bound"))
        elif ptype == "absorption":
            r = verify_absorption_smt(mc, prop["start"], prop["target"],
                                      prop.get("lower_bound"), prop.get("upper_bound"))
        elif ptype == "hitting_time":
            r = verify_hitting_time_smt(mc, prop["start"], prop["target"], prop["max_steps"])
        elif ptype == "reachability":
            r = verify_reachability(mc, prop["source"], prop["target"],
                                    prop.get("max_steps"))
        elif ptype == "uniqueness":
            r = verify_steady_state_unique(mc)
        else:
            r = VerificationResult(
                verdict=Verdict.UNKNOWN,
                property_name=ptype,
                description=f"Unknown property type: {ptype}",
            )

        result.results.append(r)
        if r.certificate:
            certs.append(r.certificate)

    # Combine all certificates
    if certs:
        result.certificate = combine_certificates(*certs,
            claim=f"Chain verification ({len(certs)} properties)")

    return result


# ---------------------------------------------------------------------------
# Convenience: Certified steady-state analysis
# ---------------------------------------------------------------------------

def certified_steady_state(mc: MarkovChain,
                           tolerance: float = 0.01) -> ChainVerificationResult:
    """Compute steady-state distribution and certify each probability
    within the given tolerance.

    Returns a ChainVerificationResult with one result per state,
    each verifying pi[i] is within [computed - tol, computed + tol].
    """
    ss = steady_state_exact(mc)
    if ss is None:
        ss = steady_state(mc)
    if ss is None:
        return ChainVerificationResult(chain=mc)

    properties = []
    for i in range(mc.n_states):
        lb = max(0.0, ss[i] - tolerance)
        ub = min(1.0, ss[i] + tolerance)
        properties.append({
            "type": "steady_state",
            "state": i,
            "lower_bound": lb,
            "upper_bound": ub,
        })

    return verify_chain(mc, properties)


# ---------------------------------------------------------------------------
# Convenience: Certified absorption analysis
# ---------------------------------------------------------------------------

def certified_absorption(mc: MarkovChain,
                         tolerance: float = 0.01) -> ChainVerificationResult:
    """Compute absorption probabilities and certify them within tolerance."""
    abs_probs = absorption_probabilities(mc)
    if not abs_probs:
        return ChainVerificationResult(chain=mc)

    state_types = classify_states(mc)
    transient = [i for i in range(mc.n_states) if state_types[i] == StateType.TRANSIENT]
    absorbing = [i for i in range(mc.n_states) if state_types[i] == StateType.ABSORBING]

    properties = []
    for target in absorbing:
        probs = abs_probs.get(target, [])
        for start in transient:
            if start < len(probs):
                p = probs[start]
                lb = max(0.0, p - tolerance)
                ub = min(1.0, p + tolerance)
                properties.append({
                    "type": "absorption",
                    "start": start,
                    "target": target,
                    "lower_bound": lb,
                    "upper_bound": ub,
                })

    return verify_chain(mc, properties)


# ---------------------------------------------------------------------------
# Convenience: Compare numerical vs SMT-verified
# ---------------------------------------------------------------------------

def compare_numerical_vs_smt(mc: MarkovChain) -> dict:
    """Compare V065 numerical analysis with V066 SMT verification.

    Returns a dict summarizing agreement/disagreement.
    """
    analysis = analyze_chain(mc)
    results = {"chain_size": mc.n_states, "numerical": {}, "smt": {}, "agreement": True}

    # Stochasticity
    stoch_r = verify_stochastic(mc)
    results["smt"]["stochastic"] = stoch_r.verdict.value
    results["numerical"]["stochastic"] = len(mc.validate()) == 0

    # Irreducibility
    irr_r = verify_irreducible(mc)
    results["smt"]["irreducible"] = irr_r.verdict.value
    results["numerical"]["irreducible"] = analysis.is_irreducible

    # Steady state (if exists)
    if analysis.steady_state:
        ss = analysis.steady_state
        results["numerical"]["steady_state"] = ss
        ss_results = []
        for i in range(mc.n_states):
            r = verify_steady_state_smt(mc, i, lower_bound=max(0, ss[i] - 0.01),
                                        upper_bound=min(1, ss[i] + 0.01))
            ss_results.append(r.verdict.value)
            if r.verdict != Verdict.VERIFIED:
                results["agreement"] = False
        results["smt"]["steady_state_bounds"] = ss_results

    return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _lcm(a: int, b: int) -> int:
    return a * b // _gcd(a, b) if a and b else 0
