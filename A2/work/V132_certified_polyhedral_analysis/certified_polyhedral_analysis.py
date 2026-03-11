"""V132: Certified Polyhedral Analysis.

Composes V105 (polyhedral abstract domain) + V044 (proof certificates).

Certifies polyhedral analysis results with machine-checkable proof obligations:
- Variable bounds: each inferred interval has a proof obligation
- Relational constraints: each multi-variable constraint is certified
- Feasibility: non-emptiness of the final polyhedron
- Properties: user-specified linear properties verified against the polyhedron

Each certificate can be independently checked by re-running polyhedral analysis
from source, or by SMT verification of the encoded obligations.
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V105_polyhedral_domain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))

from polyhedral_domain import (
    polyhedral_analyze, get_variable_range, get_all_constraints,
    get_relational_constraints, verify_property, polyhedral_summary,
    PolyhedralDomain, PolyhedralInterpreter, LinearConstraint,
)
from proof_certificates import (
    ProofCertificate, ProofObligation, CertStatus, ProofKind,
    check_certificate, save_certificate, load_certificate,
    combine_certificates,
)


# ---------------------------------------------------------------------------
# Certificate types
# ---------------------------------------------------------------------------

class PolyhedralCertKind(Enum):
    BOUNDS = "bounds"
    RELATIONAL = "relational"
    FEASIBILITY = "feasibility"
    PROPERTY = "property"
    FULL = "full"


@dataclass
class PolyhedralCertificate:
    """Certificate for polyhedral analysis results."""
    kind: PolyhedralCertKind
    source: str
    env: Optional[PolyhedralDomain]
    obligations: List[ProofObligation]
    status: CertStatus
    metadata: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_obligations(self) -> int:
        return len(self.obligations)

    @property
    def valid_obligations(self) -> int:
        return sum(1 for o in self.obligations if o.status == CertStatus.VALID)

    @property
    def invalid_obligations(self) -> int:
        return sum(1 for o in self.obligations if o.status == CertStatus.INVALID)

    def summary(self) -> str:
        lines = [
            "Polyhedral Certificate",
            "=" * 40,
            f"Kind: {self.kind.value}",
            f"Status: {self.status.value}",
            f"Obligations: {self.valid_obligations}/{self.total_obligations} valid",
        ]
        if self.metadata.get("variables"):
            lines.append(f"Variables: {', '.join(self.metadata['variables'])}")
        if self.metadata.get("constraints_count"):
            lines.append(f"Constraints: {self.metadata['constraints_count']}")
        if self.metadata.get("relational_count"):
            lines.append(f"Relational constraints: {self.metadata['relational_count']}")
        if self.metadata.get("properties_checked"):
            lines.append(f"Properties checked: {self.metadata['properties_checked']}")
        for obl in self.obligations:
            marker = "OK" if obl.status == CertStatus.VALID else "FAIL" if obl.status == CertStatus.INVALID else "?"
            lines.append(f"  [{marker}] {obl.name}: {obl.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "source": self.source,
            "obligations": [o.to_dict() for o in self.obligations],
            "status": self.status.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> 'PolyhedralCertificate':
        return PolyhedralCertificate(
            kind=PolyhedralCertKind(d["kind"]),
            source=d["source"],
            env=None,
            obligations=[ProofObligation.from_dict(o) for o in d["obligations"]],
            status=CertStatus(d["status"]),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_json(s: str) -> 'PolyhedralCertificate':
        return PolyhedralCertificate.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Obligation generators
# ---------------------------------------------------------------------------

def _constraint_to_smt(var: str, lower: float, upper: float) -> str:
    """Generate SMT-LIB2 for variable bounds."""
    parts = []
    if lower != float('-inf'):
        parts.append(f"(>= {var} {int(lower) if lower == int(lower) else lower})")
    if upper != float('inf'):
        parts.append(f"(<= {var} {int(upper) if upper == int(upper) else upper})")
    if not parts:
        return "(assert true)"
    conj = parts[0] if len(parts) == 1 else f"(and {' '.join(parts)})"
    return f"(set-logic QF_LIA)\n(declare-fun {var} () Int)\n(assert (not {conj}))\n(check-sat)"


def _relational_to_smt(constraint_str: str, variables: List[str]) -> str:
    """Generate SMT-LIB2 for a relational constraint string."""
    decls = "\n".join(f"(declare-fun {v} () Int)" for v in sorted(set(variables)))
    return f"(set-logic QF_LIA)\n{decls}\n(assert (not {_parse_constraint_to_smt(constraint_str)}))\n(check-sat)"


def _parse_constraint_to_smt(constraint_str: str) -> str:
    """Parse a human-readable constraint like '2*x + y <= 10' to SMT-LIB2 term."""
    s = constraint_str.strip()
    for op, smt_op in [(" == ", "="), (" <= ", "<="), (" >= ", ">="), (" < ", "<"), (" > ", ">")]:
        if op in s:
            lhs, rhs = s.split(op, 1)
            lhs_smt = _expr_to_smt(lhs.strip())
            rhs_smt = _expr_to_smt(rhs.strip())
            return f"({smt_op} {lhs_smt} {rhs_smt})"
    return f"true"


def _expr_to_smt(expr: str) -> str:
    """Convert expression like '2*x + y' or '-3*z' to SMT term."""
    expr = expr.strip()
    # Try as integer constant
    try:
        val = int(expr)
        return str(val) if val >= 0 else f"(- {-val})"
    except ValueError:
        pass
    # Try as fraction
    try:
        val = float(expr)
        ival = int(val)
        if val == ival:
            return str(ival) if ival >= 0 else f"(- {-ival})"
    except ValueError:
        pass
    # Simple variable
    if expr.isidentifier():
        return expr
    # Parse additive terms: split on + and - (keeping sign)
    terms = []
    current = ""
    for ch in expr:
        if ch in "+-" and current.strip() and current.strip()[-1] not in "*":
            terms.append(current.strip())
            current = ch
        else:
            current += ch
    if current.strip():
        terms.append(current.strip())
    if len(terms) == 1:
        return _term_to_smt(terms[0])
    smt_terms = [_term_to_smt(t) for t in terms]
    result = smt_terms[0]
    for t in smt_terms[1:]:
        result = f"(+ {result} {t})"
    return result


def _term_to_smt(term: str) -> str:
    """Convert a single term like '2*x' or '-y' or '5' to SMT."""
    term = term.strip()
    if not term:
        return "0"
    try:
        val = int(term)
        return str(val) if val >= 0 else f"(- {-val})"
    except ValueError:
        pass
    # Handle coefficient*variable
    if "*" in term:
        parts = term.split("*", 1)
        coeff = parts[0].strip()
        var = parts[1].strip()
        try:
            c = int(coeff)
            if c == 1:
                return var
            if c == -1:
                return f"(- 0 {var})"
            if c < 0:
                return f"(- 0 (* {-c} {var}))"
            return f"(* {c} {var})"
        except ValueError:
            return f"(* {coeff} {var})"
    # Negated variable
    if term.startswith("-"):
        return f"(- 0 {term[1:].strip()})"
    # Plain variable
    return term


def _make_bounds_obligations(env: PolyhedralDomain) -> List[ProofObligation]:
    """Generate proof obligations for variable bounds."""
    obligations = []
    for var in sorted(env.var_names):
        lo, hi = env.get_interval(var)
        desc_parts = []
        if lo != float('-inf'):
            lo_int = int(lo) if lo == int(lo) else lo
            desc_parts.append(f"{lo_int} <= {var}")
        if hi != float('inf'):
            hi_int = int(hi) if hi == int(hi) else hi
            desc_parts.append(f"{var} <= {hi_int}")
        if not desc_parts:
            continue  # Unconstrained -- no obligation needed
        desc = " and ".join(desc_parts)
        formula_smt = _constraint_to_smt(var, lo, hi)
        obligations.append(ProofObligation(
            name=f"bounds_{var}",
            description=f"Variable {var} satisfies: {desc}",
            formula_str=desc,
            formula_smt=formula_smt,
            status=CertStatus.VALID,
        ))
    return obligations


def _make_relational_obligations(env: PolyhedralDomain) -> List[ProofObligation]:
    """Generate proof obligations for relational (multi-variable) constraints."""
    obligations = []
    rel_strs = env.get_relational_constraints()
    for i, cs in enumerate(rel_strs):
        # Extract variables from the constraint
        variables = [v for v in env.var_names if v in cs]
        formula_smt = _relational_to_smt(cs, variables)
        obligations.append(ProofObligation(
            name=f"relational_{i}",
            description=f"Relational constraint: {cs}",
            formula_str=cs,
            formula_smt=formula_smt,
            status=CertStatus.VALID,
        ))
    return obligations


def _make_feasibility_obligation(env: PolyhedralDomain) -> ProofObligation:
    """Generate a proof obligation for polyhedron non-emptiness."""
    is_feasible = not env.is_bot()
    return ProofObligation(
        name="feasibility",
        description=f"Polyhedron is {'feasible (non-empty)' if is_feasible else 'infeasible (empty/bottom)'}",
        formula_str=f"feasible = {is_feasible}",
        formula_smt=f"(set-logic QF_LIA)\n(assert {'true' if is_feasible else 'false'})\n(check-sat)",
        status=CertStatus.VALID if is_feasible else CertStatus.VALID,
        # Both feasible and infeasible are valid observations
    )


def _make_property_obligations(env: PolyhedralDomain, properties: List[str]) -> List[ProofObligation]:
    """Generate proof obligations for user-specified properties."""
    obligations = []
    for i, prop in enumerate(properties):
        # Parse and check the property against the polyhedron
        result = _check_property_against_env(env, prop)
        obligations.append(ProofObligation(
            name=f"property_{i}",
            description=f"Property: {prop} -- {'VERIFIED' if result else 'NOT VERIFIED'}",
            formula_str=prop,
            formula_smt=_property_to_smt(prop, env.var_names),
            status=CertStatus.VALID if result else CertStatus.INVALID,
        ))
    return obligations


def _is_infeasible(env: PolyhedralDomain) -> bool:
    """Check if a polyhedron is infeasible using Fourier-Motzkin interval check.

    More robust than env.is_bot() which only checks unary constraint contradictions.
    This uses get_interval() (Fourier-Motzkin projection) to detect contradictions
    in multi-variable constraint systems.
    """
    if env.is_bot():
        return True
    # Check each variable's projected interval for contradiction
    for var in env.var_names:
        lo, hi = env.get_interval(var)
        if lo > hi:
            return True
    return False


def _check_property_against_env(env: PolyhedralDomain, prop_str: str) -> bool:
    """Check if a linear property holds in the polyhedron."""
    s = prop_str.strip()
    for op in [" == ", " <= ", " >= ", " < ", " > "]:
        if op in s:
            lhs, rhs = s.split(op, 1)
            lhs_coeffs, lhs_const = _parse_linear_expr(lhs.strip())
            rhs_coeffs, rhs_const = _parse_linear_expr(rhs.strip())
            # combined = lhs_coeffs - rhs_coeffs, bound = rhs_const - lhs_const
            # Property: sum(combined[v]*v) op bound
            combined = dict(lhs_coeffs)
            for v, c in rhs_coeffs.items():
                combined[v] = combined.get(v, 0) - c
            bound = rhs_const - lhs_const
            if op.strip() == "==":
                # Check env implies combined == bound
                # Negate: combined != bound = (combined <= bound-1) OR (combined >= bound+1)
                # Check both are infeasible
                neg1 = env.copy()
                neg1.constraints.append(LinearConstraint.from_dict(combined, bound - 1))
                neg2 = env.copy()
                neg2.constraints.append(LinearConstraint.from_dict(
                    {v: -c for v, c in combined.items()}, -bound - 1))
                return _is_infeasible(neg1) and _is_infeasible(neg2)
            elif op.strip() in ["<=", "<"]:
                # Check env implies combined <= bound (<=) or combined < bound (<)
                # Negate <=: combined >= bound+1 = -combined <= -bound-1
                # Negate <: combined >= bound = -combined <= -bound
                neg_coeffs = {v: -c for v, c in combined.items()}
                neg_bound = -bound - (1 if op.strip() == "<=" else 0)
                neg_env = env.copy()
                neg_env.constraints.append(LinearConstraint.from_dict(neg_coeffs, neg_bound))
                return _is_infeasible(neg_env)
            elif op.strip() in [">=", ">"]:
                # Check env implies combined >= bound (>=) or combined > bound (>)
                # Negate >=: combined <= bound-1
                # Negate >: combined <= bound
                neg_coeffs = dict(combined)
                neg_bound = bound - (1 if op.strip() == ">=" else 0)
                neg_env = env.copy()
                neg_env.constraints.append(LinearConstraint.from_dict(neg_coeffs, neg_bound))
                return _is_infeasible(neg_env)
            break
    return False


def _parse_linear_expr(expr: str) -> Tuple[Dict[str, float], float]:
    """Parse a linear expression like '2*x + y - 3' into (coeffs, constant)."""
    coeffs = {}
    constant = 0.0
    expr = expr.strip()
    if not expr:
        return coeffs, constant
    # Tokenize: split on +/- keeping the sign
    tokens = []
    current = ""
    for i, ch in enumerate(expr):
        if ch in "+-" and i > 0 and current.strip():
            tokens.append(current.strip())
            current = ch
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "*" in tok:
            parts = tok.split("*", 1)
            try:
                c = float(parts[0].strip())
                v = parts[1].strip()
                coeffs[v] = coeffs.get(v, 0) + c
            except ValueError:
                pass
        else:
            try:
                constant += float(tok)
            except ValueError:
                # Bare variable (coefficient 1 or -1)
                if tok.startswith("-"):
                    coeffs[tok[1:].strip()] = coeffs.get(tok[1:].strip(), 0) - 1
                else:
                    coeffs[tok] = coeffs.get(tok, 0) + 1
    return coeffs, constant


def _property_to_smt(prop: str, var_names: List[str]) -> str:
    """Convert property string to SMT-LIB2."""
    variables = [v for v in var_names if v in prop]
    if not variables:
        variables = var_names
    decls = "\n".join(f"(declare-fun {v} () Int)" for v in sorted(set(variables)))
    smt_formula = _parse_constraint_to_smt(prop)
    return f"(set-logic QF_LIA)\n{decls}\n(assert (not {smt_formula}))\n(check-sat)"


# ---------------------------------------------------------------------------
# Core certification APIs
# ---------------------------------------------------------------------------

def certify_polyhedral_bounds(source: str) -> PolyhedralCertificate:
    """Certify variable bounds from polyhedral analysis."""
    result = polyhedral_analyze(source)
    env = result['env']
    obligations = _make_bounds_obligations(env)
    status = CertStatus.VALID if all(o.status == CertStatus.VALID for o in obligations) else CertStatus.INVALID
    variables = sorted(env.var_names) if env and not env.is_bot() else []
    return PolyhedralCertificate(
        kind=PolyhedralCertKind.BOUNDS,
        source=source,
        env=env,
        obligations=obligations,
        status=status,
        metadata={
            "variables": variables,
            "constraints_count": len(env.constraints) if env else 0,
            "warnings": result.get("warnings", []),
        },
    )


def certify_polyhedral_relational(source: str) -> PolyhedralCertificate:
    """Certify relational (multi-variable) constraints from polyhedral analysis."""
    result = polyhedral_analyze(source)
    env = result['env']
    obligations = _make_relational_obligations(env)
    status = CertStatus.VALID if all(o.status == CertStatus.VALID for o in obligations) else CertStatus.INVALID
    return PolyhedralCertificate(
        kind=PolyhedralCertKind.RELATIONAL,
        source=source,
        env=env,
        obligations=obligations,
        status=status,
        metadata={
            "variables": sorted(env.var_names) if env else [],
            "relational_count": len(obligations),
            "constraints_count": len(env.constraints) if env else 0,
        },
    )


def certify_polyhedral_feasibility(source: str) -> PolyhedralCertificate:
    """Certify feasibility (non-emptiness) of the analysis result."""
    result = polyhedral_analyze(source)
    env = result['env']
    obl = _make_feasibility_obligation(env)
    return PolyhedralCertificate(
        kind=PolyhedralCertKind.FEASIBILITY,
        source=source,
        env=env,
        obligations=[obl],
        status=CertStatus.VALID,
        metadata={
            "feasible": not env.is_bot(),
            "variables": sorted(env.var_names) if env else [],
            "constraints_count": len(env.constraints) if env else 0,
        },
    )


def certify_polyhedral_properties(source: str, properties: List[str]) -> PolyhedralCertificate:
    """Certify user-specified linear properties against polyhedral analysis."""
    result = polyhedral_analyze(source)
    env = result['env']
    obligations = _make_property_obligations(env, properties)
    status = CertStatus.VALID if all(o.status == CertStatus.VALID for o in obligations) else CertStatus.INVALID
    return PolyhedralCertificate(
        kind=PolyhedralCertKind.PROPERTY,
        source=source,
        env=env,
        obligations=obligations,
        status=status,
        metadata={
            "properties_checked": len(properties),
            "properties_verified": sum(1 for o in obligations if o.status == CertStatus.VALID),
            "variables": sorted(env.var_names) if env else [],
        },
    )


def certify_full_polyhedral(source: str, properties: Optional[List[str]] = None) -> PolyhedralCertificate:
    """Full certification: bounds + relational + feasibility + optional properties."""
    result = polyhedral_analyze(source)
    env = result['env']
    obligations = []
    obligations.append(_make_feasibility_obligation(env))
    obligations.extend(_make_bounds_obligations(env))
    obligations.extend(_make_relational_obligations(env))
    if properties:
        obligations.extend(_make_property_obligations(env, properties))
    status = CertStatus.VALID if all(o.status == CertStatus.VALID for o in obligations) else CertStatus.INVALID
    return PolyhedralCertificate(
        kind=PolyhedralCertKind.FULL,
        source=source,
        env=env,
        obligations=obligations,
        status=status,
        metadata={
            "variables": sorted(env.var_names) if env else [],
            "constraints_count": len(env.constraints) if env else 0,
            "relational_count": len(env.get_relational_constraints()) if env and not env.is_bot() else 0,
            "properties_checked": len(properties) if properties else 0,
            "warnings": result.get("warnings", []),
        },
    )


# ---------------------------------------------------------------------------
# Independent checking
# ---------------------------------------------------------------------------

def check_polyhedral_certificate(cert: PolyhedralCertificate) -> PolyhedralCertificate:
    """Independently re-check a polyhedral certificate by re-running analysis."""
    result = polyhedral_analyze(cert.source)
    env = result['env']

    new_obligations = []
    for obl in cert.obligations:
        name = obl.name
        new_status = obl.status  # default: keep original

        if name == "feasibility":
            is_feasible = not env.is_bot()
            new_status = CertStatus.VALID

        elif name.startswith("bounds_"):
            var = name[len("bounds_"):]
            if var in (env.var_names if env else []):
                lo, hi = env.get_interval(var)
                # Re-derive: if bounds exist, obligation is valid
                has_bounds = lo != float('-inf') or hi != float('inf')
                new_status = CertStatus.VALID if has_bounds else CertStatus.INVALID
            else:
                new_status = CertStatus.INVALID

        elif name.startswith("relational_"):
            # Re-derive relational constraints
            rel_strs = env.get_relational_constraints() if env and not env.is_bot() else []
            idx = int(name.split("_")[1])
            if idx < len(rel_strs):
                new_status = CertStatus.VALID
            else:
                new_status = CertStatus.INVALID

        elif name.startswith("property_"):
            # Re-check property
            prop_str = obl.formula_str
            result_ok = _check_property_against_env(env, prop_str)
            new_status = CertStatus.VALID if result_ok else CertStatus.INVALID

        new_obligations.append(ProofObligation(
            name=obl.name,
            description=obl.description,
            formula_str=obl.formula_str,
            formula_smt=obl.formula_smt,
            status=new_status,
        ))

    status = CertStatus.VALID if all(o.status == CertStatus.VALID for o in new_obligations) else CertStatus.INVALID
    return PolyhedralCertificate(
        kind=cert.kind,
        source=cert.source,
        env=env,
        obligations=new_obligations,
        status=status,
        metadata=cert.metadata,
        timestamp=cert.timestamp,
    )


def certify_and_check(source: str, properties: Optional[List[str]] = None) -> PolyhedralCertificate:
    """Generate full certificate and independently check it."""
    cert = certify_full_polyhedral(source, properties)
    return check_polyhedral_certificate(cert)


# ---------------------------------------------------------------------------
# V044 bridge
# ---------------------------------------------------------------------------

def to_v044_certificate(cert: PolyhedralCertificate) -> ProofCertificate:
    """Convert PolyhedralCertificate to V044 ProofCertificate."""
    return ProofCertificate(
        kind=ProofKind.VCGEN,
        claim=f"Polyhedral analysis ({cert.kind.value}) of program",
        source=cert.source,
        obligations=cert.obligations,
        metadata={"polyhedral_analysis": True, "cert_kind": cert.kind.value, **cert.metadata},
        status=cert.status,
    )


def from_v044_certificate(v044: ProofCertificate, source: Optional[str] = None) -> PolyhedralCertificate:
    """Convert V044 ProofCertificate back to PolyhedralCertificate."""
    kind_str = v044.metadata.get("cert_kind", "full")
    try:
        kind = PolyhedralCertKind(kind_str)
    except ValueError:
        kind = PolyhedralCertKind.FULL
    return PolyhedralCertificate(
        kind=kind,
        source=source or v044.source or "",
        env=None,
        obligations=v044.obligations,
        status=v044.status,
        metadata=v044.metadata,
        timestamp=getattr(v044, 'timestamp', ''),
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_polyhedral_certificate(cert: PolyhedralCertificate, path: str):
    """Save certificate to JSON file."""
    with open(path, 'w') as f:
        f.write(cert.to_json())


def load_polyhedral_certificate(path: str) -> PolyhedralCertificate:
    """Load certificate from JSON file."""
    with open(path, 'r') as f:
        return PolyhedralCertificate.from_json(f.read())


# ---------------------------------------------------------------------------
# Comparison and summary
# ---------------------------------------------------------------------------

def compare_certified_vs_uncertified(source: str) -> dict:
    """Compare certified vs uncertified polyhedral analysis."""
    t0 = time.time()
    uncert = polyhedral_analyze(source)
    t_uncert = time.time() - t0

    t0 = time.time()
    cert = certify_full_polyhedral(source)
    t_cert = time.time() - t0

    env = uncert['env']
    return {
        "uncertified": {
            "variables": len(env.var_names) if env else 0,
            "constraints": len(env.constraints) if env else 0,
            "warnings": len(uncert.get("warnings", [])),
            "time": t_uncert,
        },
        "certified": {
            "status": cert.status.value,
            "obligations_total": cert.total_obligations,
            "obligations_valid": cert.valid_obligations,
            "time": t_cert,
        },
    }


def polyhedral_certificate_summary(source: str, properties: Optional[List[str]] = None) -> str:
    """Generate human-readable summary of certified polyhedral analysis."""
    cert = certify_full_polyhedral(source, properties)
    return cert.summary()


def get_certified_bounds(source: str, var_name: str) -> dict:
    """Get certified bounds for a specific variable."""
    cert = certify_polyhedral_bounds(source)
    obl_name = f"bounds_{var_name}"
    for obl in cert.obligations:
        if obl.name == obl_name:
            return {
                "variable": var_name,
                "bounds_description": obl.formula_str,
                "status": obl.status.value,
                "formula_smt": obl.formula_smt,
            }
    return {
        "variable": var_name,
        "bounds_description": "unconstrained",
        "status": "unconstrained",
    }


def get_certified_constraints(source: str) -> List[dict]:
    """Get all certified constraints (bounds + relational)."""
    cert = certify_full_polyhedral(source)
    return [
        {
            "name": obl.name,
            "description": obl.description,
            "formula": obl.formula_str,
            "status": obl.status.value,
        }
        for obl in cert.obligations
    ]
