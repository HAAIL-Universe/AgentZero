"""V136: Certified k-Induction -- Machine-checkable certificates for k-induction proofs.

Composes V015 (k-induction model checking) + V044 (proof certificates).

For each k-induction proof, generates proof obligations:
- Base case: Init(s0) AND Trans^k => Prop(s0..sk)
- Inductive step: Prop(s0..sk) AND Trans^(k+1) => Prop(s_{k+1})
- Strengthening: invariant initiation and consecution (if used)

Each obligation includes SMT-LIB2 formula for independent machine checking.
"""

import sys, os, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V015_k_induction'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))

from k_induction import (
    k_induction_check, incremental_k_induction, k_induction_with_strengthening,
    verify_loop, verify_loop_with_invariants, bmc_check,
    _apply_formula_at_step, _apply_trans_at_step, _negate, _step_vars,
)
from proof_certificates import (
    ProofCertificate, ProofObligation, ProofKind, CertStatus,
    check_certificate, save_certificate, load_certificate,
    combine_certificates,
)
from pdr import TransitionSystem
from smt_solver import SMTSolver, Var, App, Op, IntConst, BoolConst, INT, BOOL, SMTResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _and(*args):
    if len(args) == 1:
        return args[0]
    return App(Op.AND, list(args), BOOL)


def _or(*args):
    if len(args) == 1:
        return args[0]
    return App(Op.OR, list(args), BOOL)


def _not(x):
    return App(Op.NOT, [x], BOOL)


def _implies(a, b):
    return App(Op.IMPLIES, [a, b], BOOL)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class KIndCertKind(Enum):
    BASIC = "basic"
    STRENGTHENED = "strengthened"


@dataclass
class KIndCertificate:
    """Certificate for a k-induction proof."""
    kind: KIndCertKind
    claim: str
    k: int
    result: str  # "safe", "unsafe", "unknown"
    obligations: List[ProofObligation] = field(default_factory=list)
    counterexample: Optional[List[Dict]] = None
    invariants_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: CertStatus = CertStatus.UNCHECKED
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        valid = sum(1 for o in self.obligations if o.status == CertStatus.VALID)
        invalid = sum(1 for o in self.obligations if o.status == CertStatus.INVALID)
        lines = [
            f"KIndCertificate: {self.claim}",
            f"  Kind: {self.kind.value}, k={self.k}, Result: {self.result}",
            f"  Status: {self.status.value}",
            f"  Obligations: {len(self.obligations)} total",
            f"  Valid: {valid}, Invalid: {invalid}",
        ]
        if self.invariants_used:
            lines.append(f"  Invariants: {len(self.invariants_used)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "claim": self.claim,
            "k": self.k,
            "result": self.result,
            "obligations": [_obligation_to_dict(o) for o in self.obligations],
            "counterexample": self.counterexample,
            "invariants_used": self.invariants_used,
            "metadata": self.metadata,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> 'KIndCertificate':
        return KIndCertificate(
            kind=KIndCertKind(d["kind"]),
            claim=d["claim"],
            k=d["k"],
            result=d["result"],
            obligations=[_obligation_from_dict(o) for o in d["obligations"]],
            counterexample=d.get("counterexample"),
            invariants_used=d.get("invariants_used", []),
            metadata=d.get("metadata", {}),
            status=CertStatus(d["status"]),
            timestamp=d.get("timestamp", ""),
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_json(s: str) -> 'KIndCertificate':
        import json
        return KIndCertificate.from_dict(json.loads(s))


def _obligation_to_dict(o: ProofObligation) -> dict:
    d = {
        "name": o.name,
        "description": o.description,
        "formula_str": o.formula_str,
        "formula_smt": o.formula_smt,
        "status": o.status.value,
    }
    if o.counterexample:
        d["counterexample"] = o.counterexample
    return d


def _obligation_from_dict(d: dict) -> ProofObligation:
    return ProofObligation(
        name=d["name"],
        description=d["description"],
        formula_str=d["formula_str"],
        formula_smt=d["formula_smt"],
        status=CertStatus(d["status"]),
        counterexample=d.get("counterexample"),
    )


# ---------------------------------------------------------------------------
# SMT-LIB2 serialization
# ---------------------------------------------------------------------------

def _term_to_smtlib(term):
    """Convert SMT term to SMT-LIB2 string."""
    if isinstance(term, (int, float)):
        return f"(- {abs(term)})" if term < 0 else str(term)
    if isinstance(term, bool):
        return "true" if term else "false"
    if isinstance(term, Var):
        return term.name
    if isinstance(term, IntConst):
        v = term.value
        return f"(- {abs(v)})" if v < 0 else str(v)
    if isinstance(term, BoolConst):
        return "true" if term.value else "false"
    if isinstance(term, App):
        op_map = {
            Op.ADD: "+", Op.SUB: "-", Op.MUL: "*",
            Op.EQ: "=", Op.NEQ: "distinct",
            Op.LT: "<", Op.LE: "<=", Op.GT: ">", Op.GE: ">=",
            Op.AND: "and", Op.OR: "or", Op.NOT: "not",
            Op.ITE: "ite", Op.IMPLIES: "=>",
        }
        op_str = op_map.get(term.op, str(term.op))
        args_str = " ".join(_term_to_smtlib(a) for a in term.args)
        return f"({op_str} {args_str})"
    return str(term)


def _collect_vars(term, result=None):
    """Collect all Var objects from a term."""
    if result is None:
        result = {}
    if isinstance(term, Var):
        result[term.name] = term.sort
    elif isinstance(term, App):
        for a in term.args:
            _collect_vars(a, result)
    return result


def _build_smtlib_script(formula, description=""):
    """Build SMT-LIB2 validity check script: UNSAT of NOT(formula) means VALID."""
    var_info = _collect_vars(formula)
    lines = [f"; {description}" if description else "; k-induction obligation"]
    lines.append("(set-logic LIA)")
    for name in sorted(var_info.keys()):
        sort_str = "Int" if var_info[name] == INT else "Bool"
        lines.append(f"(declare-const {name} {sort_str})")
    lines.append(f"(assert (not {_term_to_smtlib(formula)}))")
    lines.append("(check-sat)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Obligation building using V015's stepping functions
# ---------------------------------------------------------------------------

def _build_obligations(ts, k, invariant=None):
    """Build all proof obligations for a k-induction proof at depth k."""
    obligations = []

    # Use a temporary solver just for creating stepped variable names
    solver = SMTSolver()
    # Pre-register all variables for steps 0..k+1
    for step in range(k + 2):
        for name, sort in ts.state_vars:
            step_name = f"{name}_{step}"
            if sort == INT:
                solver.Int(step_name)
            else:
                solver.Bool(step_name)

    # --- Base case obligations ---
    # For each depth i (0..k): Init(s0) AND Trans(s0..si) => Prop(s0..si)
    for depth in range(k + 1):
        premises = []
        # Init
        premises.append(_apply_formula_at_step(ts, solver, ts.init_formula, 0))
        # Transitions 0->1, ..., (depth-1)->depth
        for s in range(depth):
            premises.append(_apply_trans_at_step(ts, solver, s))
        # Invariant at each step (if used)
        if invariant is not None:
            for s in range(depth + 1):
                premises.append(_apply_formula_at_step(ts, solver, invariant, s))

        # Properties at all steps 0..depth
        props = []
        for s in range(depth + 1):
            props.append(_apply_formula_at_step(ts, solver, ts.prop_formula, s))

        premise = _and(*premises) if len(premises) > 1 else premises[0]
        conclusion = _and(*props) if len(props) > 1 else props[0]
        formula = _implies(premise, conclusion)

        desc = f"Base case: property holds at steps 0..{depth} from initial state"
        obligations.append(ProofObligation(
            name=f"base_{depth}",
            description=desc,
            formula_str=f"Init AND Trans^{depth} => Prop(0..{depth})",
            formula_smt=_build_smtlib_script(formula, desc),
        ))

    # --- Inductive step obligation ---
    # Prop(s0..sk) AND Trans(s0..sk->sk+1) => Prop(sk+1)
    ind_premises = []
    for s in range(k + 1):
        ind_premises.append(_apply_formula_at_step(ts, solver, ts.prop_formula, s))
    for s in range(k + 1):
        ind_premises.append(_apply_trans_at_step(ts, solver, s))
    if invariant is not None:
        for s in range(k + 2):
            ind_premises.append(_apply_formula_at_step(ts, solver, invariant, s))

    ind_conclusion = _apply_formula_at_step(ts, solver, ts.prop_formula, k + 1)
    ind_formula = _implies(_and(*ind_premises), ind_conclusion)

    desc = f"Inductive step: Prop(0..{k}) AND Trans^{k+1} => Prop({k+1})"
    obligations.append(ProofObligation(
        name=f"inductive_k{k}",
        description=desc,
        formula_str=f"Prop(0..{k}) AND Trans^{k+1} => Prop({k+1})",
        formula_smt=_build_smtlib_script(ind_formula, desc),
    ))

    # --- Strengthening obligations (if invariant provided) ---
    if invariant is not None:
        # Inv initiation: Init => Inv(0)
        init_f = _apply_formula_at_step(ts, solver, ts.init_formula, 0)
        inv_0 = _apply_formula_at_step(ts, solver, invariant, 0)
        init_formula = _implies(init_f, inv_0)
        obligations.append(ProofObligation(
            name="inv_init",
            description="Strengthening: invariant holds at initial state",
            formula_str="Init => Inv(0)",
            formula_smt=_build_smtlib_script(init_formula, "Invariant initiation"),
        ))

        # Inv consecution: Inv(0..k) AND Prop(0..k) AND Trans^(k+1) => Inv(k+1)
        cons_premises = []
        for s in range(k + 1):
            cons_premises.append(_apply_formula_at_step(ts, solver, invariant, s))
            cons_premises.append(_apply_formula_at_step(ts, solver, ts.prop_formula, s))
        for s in range(k + 1):
            cons_premises.append(_apply_trans_at_step(ts, solver, s))
        inv_kp1 = _apply_formula_at_step(ts, solver, invariant, k + 1)
        cons_formula = _implies(_and(*cons_premises), inv_kp1)
        obligations.append(ProofObligation(
            name="inv_consecution",
            description=f"Strengthening: invariant preserved under {k+1} transitions",
            formula_str=f"Inv(0..{k}) AND Prop(0..{k}) AND Trans^{k+1} => Inv({k+1})",
            formula_smt=_build_smtlib_script(cons_formula, "Invariant consecution"),
        ))

    return obligations


def _compute_cert_status(obligations):
    """Compute overall status from obligations."""
    if not obligations:
        return CertStatus.UNKNOWN
    if all(o.status == CertStatus.VALID for o in obligations):
        return CertStatus.VALID
    if any(o.status == CertStatus.INVALID for o in obligations):
        return CertStatus.INVALID
    if all(o.status == CertStatus.UNCHECKED for o in obligations):
        return CertStatus.UNCHECKED
    return CertStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Independent checking (SMT-LIB2 parsing + re-verification)
# ---------------------------------------------------------------------------

def _tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in ' \t\n\r':
            i += 1
        elif c == '(':
            tokens.append('(')
            i += 1
        elif c == ')':
            tokens.append(')')
            i += 1
        elif c == ';':
            while i < len(text) and text[i] != '\n':
                i += 1
        else:
            j = i
            while j < len(text) and text[j] not in ' \t\n\r()':
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_tokens(tokens, pos, var_map):
    if pos >= len(tokens):
        return BoolConst(True), pos

    tok = tokens[pos]
    if tok == '(':
        pos += 1
        if pos >= len(tokens):
            return BoolConst(True), pos

        op_tok = tokens[pos]
        pos += 1

        # Skip commands
        if op_tok in ('declare-const', 'set-logic', 'check-sat'):
            depth = 0
            while pos < len(tokens):
                if tokens[pos] == '(':
                    depth += 1
                elif tokens[pos] == ')':
                    if depth == 0:
                        return None, pos + 1
                    depth -= 1
                pos += 1
            return None, pos

        if op_tok == 'assert':
            result, pos = _parse_tokens(tokens, pos, var_map)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return result, pos

        if op_tok == 'not':
            arg, pos = _parse_tokens(tokens, pos, var_map)
            if pos < len(tokens) and tokens[pos] == ')':
                pos += 1
            return _not(arg), pos

        if op_tok == '-':
            args = []
            while pos < len(tokens) and tokens[pos] != ')':
                arg, pos = _parse_tokens(tokens, pos, var_map)
                if arg is not None:
                    args.append(arg)
            if pos < len(tokens):
                pos += 1
            if len(args) == 1:
                if isinstance(args[0], IntConst):
                    return IntConst(-args[0].value), pos
                return App(Op.SUB, [IntConst(0), args[0]], INT), pos
            return App(Op.SUB, args, INT), pos

        op_map = {
            '+': Op.ADD, '*': Op.MUL,
            '=': Op.EQ, 'distinct': Op.NEQ,
            '<': Op.LT, '<=': Op.LE, '>': Op.GT, '>=': Op.GE,
            'and': Op.AND, 'or': Op.OR,
            '=>': Op.IMPLIES, 'ite': Op.ITE,
        }
        op = op_map.get(op_tok)
        if op is None:
            depth = 0
            while pos < len(tokens):
                if tokens[pos] == '(':
                    depth += 1
                elif tokens[pos] == ')':
                    if depth == 0:
                        return BoolConst(True), pos + 1
                    depth -= 1
                pos += 1
            return BoolConst(True), pos

        args = []
        while pos < len(tokens) and tokens[pos] != ')':
            arg, pos = _parse_tokens(tokens, pos, var_map)
            if arg is not None:
                args.append(arg)
        if pos < len(tokens):
            pos += 1

        sort = INT if op in (Op.ADD, Op.SUB, Op.MUL, Op.ITE) else BOOL

        # Handle n-ary and/or
        if op in (Op.AND, Op.OR) and len(args) > 2:
            result = args[0]
            for a in args[1:]:
                result = App(op, [result, a], BOOL)
            return result, pos

        if len(args) >= 2:
            return App(op, args, sort), pos
        if len(args) == 1:
            return args[0], pos
        return BoolConst(True), pos

    # Atoms
    if tok == 'true':
        return BoolConst(True), pos + 1
    if tok == 'false':
        return BoolConst(False), pos + 1
    try:
        return IntConst(int(tok)), pos + 1
    except ValueError:
        pass
    if tok in var_map:
        return var_map[tok], pos + 1
    return Var(tok, INT), pos + 1


def _recheck_obligation(obligation):
    """Re-check an obligation by parsing its SMT-LIB2 script."""
    smt_script = obligation.formula_smt

    # Extract declarations and assertion
    var_info = {}
    lines = smt_script.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('(declare-const'):
            parts = line.replace('(', ' ').replace(')', ' ').split()
            if len(parts) >= 3:
                var_info[parts[1]] = parts[2]

    # Find the assertion
    start = smt_script.find('(assert')
    if start < 0:
        obligation.status = CertStatus.UNKNOWN
        return

    # Extract assertion text
    depth = 0
    end = start
    for i in range(start, len(smt_script)):
        if smt_script[i] == '(':
            depth += 1
        elif smt_script[i] == ')':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assertion_text = smt_script[start:end]

    # Create solver and variable map
    solver = SMTSolver()
    var_map = {}
    for name, sort in var_info.items():
        if sort == "Int":
            var_map[name] = solver.Int(name)
        else:
            var_map[name] = solver.Bool(name)

    # Parse and check
    tokens = _tokenize(assertion_text)
    term, _ = _parse_tokens(tokens, 0, var_map)
    if term is not None:
        solver.add(term)
        result = solver.check()
        if result == SMTResult.UNSAT:
            obligation.status = CertStatus.VALID
        elif result == SMTResult.SAT:
            obligation.status = CertStatus.INVALID
            try:
                model = solver.model()
                obligation.counterexample = {str(k): v for k, v in model.items()}
            except Exception:
                pass
        else:
            obligation.status = CertStatus.UNKNOWN
    else:
        obligation.status = CertStatus.UNKNOWN


def check_kind_certificate(cert):
    """Independently verify all obligations in a certificate."""
    for obligation in cert.obligations:
        _recheck_obligation(obligation)
    cert.status = _compute_cert_status(cert.obligations)
    return cert


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def certify_k_induction(ts, k=None, max_k=20):
    """Run k-induction and generate a certificate."""
    t0 = time.time()

    if k is not None:
        result = k_induction_check(ts, k)
    else:
        result = incremental_k_induction(ts, max_k)

    found_k = result.k if result.k is not None else 0

    if result.result == "SAFE":
        obligations = _build_obligations(ts, found_k)
        cert = KIndCertificate(
            kind=KIndCertKind.BASIC,
            claim=f"Property holds (k-induction, k={found_k})",
            k=found_k,
            result="safe",
            obligations=obligations,
            metadata={"method": "k-induction", "k": found_k, "duration": time.time() - t0},
        )
    elif result.result == "UNSAFE":
        cert = KIndCertificate(
            kind=KIndCertKind.BASIC,
            claim=f"Property violated at step {found_k}",
            k=found_k,
            result="unsafe",
            counterexample=result.counterexample,
            metadata={"method": "k-induction", "k": found_k, "duration": time.time() - t0},
        )
    else:
        cert = KIndCertificate(
            kind=KIndCertKind.BASIC,
            claim=f"k-induction inconclusive up to k={max_k}",
            k=found_k,
            result="unknown",
            metadata={"method": "k-induction", "max_k": max_k, "duration": time.time() - t0},
        )

    return cert


def certify_strengthened_k_induction(ts, invariants, max_k=20):
    """Run k-induction with strengthening invariants and generate certificate."""
    t0 = time.time()

    result = k_induction_with_strengthening(ts, max_k, invariants)
    found_k = result.k if result.k is not None else 0

    # Build combined invariant
    if len(invariants) == 1:
        combined_inv = invariants[0]
    else:
        combined_inv = _and(*invariants)

    inv_strs = [_term_to_smtlib(inv) for inv in invariants]

    if result.result == "SAFE":
        obligations = _build_obligations(ts, found_k, invariant=combined_inv)
        cert = KIndCertificate(
            kind=KIndCertKind.STRENGTHENED,
            claim=f"Property holds (strengthened k-induction, k={found_k})",
            k=found_k,
            result="safe",
            obligations=obligations,
            invariants_used=inv_strs,
            metadata={"method": "strengthened-k-induction", "k": found_k,
                       "num_invariants": len(invariants), "duration": time.time() - t0},
        )
    elif result.result == "UNSAFE":
        cert = KIndCertificate(
            kind=KIndCertKind.STRENGTHENED,
            claim=f"Property violated at step {found_k}",
            k=found_k,
            result="unsafe",
            counterexample=result.counterexample,
            invariants_used=inv_strs,
            metadata={"method": "strengthened-k-induction", "k": found_k, "duration": time.time() - t0},
        )
    else:
        cert = KIndCertificate(
            kind=KIndCertKind.STRENGTHENED,
            claim=f"Strengthened k-induction inconclusive up to k={max_k}",
            k=found_k,
            result="unknown",
            invariants_used=inv_strs,
            metadata={"method": "strengthened-k-induction", "max_k": max_k, "duration": time.time() - t0},
        )

    return cert


def certify_and_check(ts, k=None, max_k=20, invariants=None):
    """Generate certificate and immediately verify it."""
    if invariants:
        cert = certify_strengthened_k_induction(ts, invariants, max_k)
    else:
        cert = certify_k_induction(ts, k, max_k)

    if cert.result == "safe":
        check_kind_certificate(cert)
    return cert


def certify_loop(source, property_source, max_k=20):
    """Source-level certified k-induction."""
    from k_induction import _extract_loop_ts

    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_expr_to_smt(property_source, ts_vars)
    ts.set_property(prop_smt)
    return certify_and_check(ts, max_k=max_k)


def certify_loop_with_invariants(source, property_source, invariant_sources, max_k=20):
    """Source-level certified k-induction with invariants."""
    from k_induction import _extract_loop_ts

    ts, ts_vars = _extract_loop_ts(source)
    prop_smt = _parse_expr_to_smt(property_source, ts_vars)
    ts.set_property(prop_smt)
    invariants = [_parse_expr_to_smt(inv, ts_vars) for inv in invariant_sources]
    return certify_and_check(ts, max_k=max_k, invariants=invariants)


def _parse_expr_to_smt(expr_str, ts_vars):
    """Parse a C10 expression string to SMT term using ts_vars."""
    from stack_vm import lex, Parser, IntLit, Var as ASTVar, BinOp

    tokens = lex(f"let __p = ({expr_str});")
    stmts = Parser(tokens).parse().stmts
    expr = stmts[0].value

    def convert(e):
        if isinstance(e, IntLit):
            return IntConst(e.value)
        elif isinstance(e, ASTVar):
            if e.name in ts_vars:
                return ts_vars[e.name]
            return IntConst(0)
        elif isinstance(e, BinOp):
            op_map = {'+': Op.ADD, '-': Op.SUB, '*': Op.MUL,
                      '<': Op.LT, '>': Op.GT, '<=': Op.LE, '>=': Op.GE,
                      '==': Op.EQ, '!=': Op.NEQ}
            l = convert(e.left)
            r = convert(e.right)
            op = op_map.get(e.op)
            if op is None:
                raise ValueError(f"Unknown op: {e.op}")
            sort = BOOL if op in (Op.LT, Op.GT, Op.LE, Op.GE, Op.EQ, Op.NEQ) else INT
            return App(op, [l, r], sort)
        return IntConst(0)

    return convert(expr)


# ---------------------------------------------------------------------------
# V044 bridge
# ---------------------------------------------------------------------------

def to_v044_certificate(cert):
    """Convert KIndCertificate to V044 ProofCertificate."""
    return ProofCertificate(
        kind=ProofKind.COMPOSITE,
        claim=cert.claim,
        source=None,
        obligations=list(cert.obligations),
        metadata={
            **cert.metadata,
            "cert_kind": cert.kind.value,
            "result": cert.result,
            "k": cert.k,
            "invariants_used": cert.invariants_used,
        },
        status=cert.status,
        timestamp=cert.timestamp,
    )


# ---------------------------------------------------------------------------
# Comparison and utilities
# ---------------------------------------------------------------------------

def compare_certified_vs_uncertified(ts, max_k=20):
    """Compare certified k-induction with plain k-induction."""
    t0 = time.time()
    plain_result = incremental_k_induction(ts, max_k)
    plain_time = time.time() - t0

    t0 = time.time()
    cert = certify_and_check(ts, max_k=max_k)
    cert_time = time.time() - t0

    return {
        "plain_result": plain_result.result,
        "plain_k": plain_result.k,
        "plain_time": plain_time,
        "certified_result": cert.result,
        "certified_k": cert.k,
        "certified_status": cert.status.value,
        "certified_obligations": len(cert.obligations),
        "certified_time": cert_time,
        "overhead_ratio": cert_time / plain_time if plain_time > 0 else 0,
    }


def kind_certificate_summary(cert):
    """Get human-readable summary."""
    return cert.summary()
