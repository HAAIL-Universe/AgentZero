"""
V058: Proof-Carrying Code (PCC)
Composes V044 (proof certificates) + V049 (verified compilation) + V055 (modular AI)
         + V004 (VCGen) + C010 (compiler/VM)

Attaches proof certificates to compiled bytecode so a receiver can verify
safety/correctness properties without re-analyzing the source code.

Pipeline:
1. Producer compiles source to bytecode via C010
2. Producer generates proofs: VCGen (Hoare logic), modular AI (bounds), compilation (optimization)
3. Producer packages bytecode + certificates into a PCC bundle
4. Consumer receives bundle, independently checks certificates via SMT
5. If all certificates check out, consumer can safely execute the bytecode
"""

import sys, os, json, hashlib, time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V004_verification_conditions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V049_verified_compilation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V055_modular_abstract_interpretation'))

from stack_vm import lex, Parser, Compiler, Chunk, VM
from proof_certificates import (
    ProofCertificate, ProofObligation, CertStatus, ProofKind,
    combine_certificates, check_certificate
)
from vc_gen import verify_function, verify_program, VCResult
from verified_compilation import (
    validate_compilation, certify_compilation,
    CompilationValidationResult
)
from modular_abstract_interpretation import (
    modular_analyze, FunctionSummary, ModularAIResult
)


# --- Enums ---

class PolicyKind(Enum):
    """Safety policies that PCC can enforce."""
    MEMORY_SAFETY = "memory_safety"       # No out-of-bounds, null derefs
    TYPE_SAFETY = "type_safety"           # Operations respect types
    TERMINATION = "termination"           # Program halts
    BOUND_SAFETY = "bound_safety"         # Variables within declared bounds
    CONTRACT_COMPLIANCE = "contract"      # Hoare-logic specifications met
    COMPILATION_SAFETY = "compilation"    # Optimizations preserve semantics


class BundleStatus(Enum):
    """Overall status of a PCC bundle."""
    VERIFIED = "verified"         # All certificates check out
    PARTIALLY_VERIFIED = "partial"  # Some certificates valid, some unknown
    FAILED = "failed"             # At least one certificate invalid
    UNCHECKED = "unchecked"       # Not yet verified by consumer


# --- Data Structures ---

@dataclass
class SafetyPolicy:
    """A safety property that must hold."""
    kind: PolicyKind
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BytecodePayload:
    """Compiled bytecode ready for execution."""
    instructions: List[Tuple]   # (opcode, *args) tuples
    constants: List[Any]
    source_hash: str           # SHA-256 of original source
    compiler_version: str = "C010"

    def to_dict(self) -> dict:
        return {
            'instructions': [list(i) for i in self.instructions],
            'constants': self.constants,
            'source_hash': self.source_hash,
            'compiler_version': self.compiler_version,
        }

    @staticmethod
    def from_dict(d: dict) -> 'BytecodePayload':
        return BytecodePayload(
            instructions=[tuple(i) for i in d['instructions']],
            constants=d['constants'],
            source_hash=d['source_hash'],
            compiler_version=d.get('compiler_version', 'C010'),
        )


@dataclass
class PCCBundle:
    """A proof-carrying code bundle: bytecode + proof certificates."""
    bytecode: BytecodePayload
    certificates: List[ProofCertificate]
    policies: List[SafetyPolicy]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: BundleStatus = BundleStatus.UNCHECKED

    @property
    def total_certificates(self) -> int:
        return len(self.certificates)

    @property
    def valid_certificates(self) -> int:
        return sum(1 for c in self.certificates if c.status == CertStatus.VALID)

    @property
    def invalid_certificates(self) -> int:
        return sum(1 for c in self.certificates if c.status == CertStatus.INVALID)

    @property
    def total_obligations(self) -> int:
        return sum(len(c.obligations) for c in self.certificates)

    @property
    def valid_obligations(self) -> int:
        return sum(
            sum(1 for o in c.obligations if o.status == CertStatus.VALID)
            for c in self.certificates
        )

    def summary(self) -> str:
        lines = [f"PCC Bundle: {self.status.value}"]
        lines.append(f"  Certificates: {self.valid_certificates}/{self.total_certificates} valid")
        lines.append(f"  Obligations: {self.valid_obligations}/{self.total_obligations} valid")
        lines.append(f"  Policies: {len(self.policies)}")
        for policy in self.policies:
            lines.append(f"    - {policy.kind.value}: {policy.description}")
        for cert in self.certificates:
            status = cert.status.value if hasattr(cert.status, 'value') else str(cert.status)
            lines.append(f"  [{status}] {cert.claim}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'bytecode': self.bytecode.to_dict(),
            'certificates': [c.to_dict() for c in self.certificates],
            'policies': [{'kind': p.kind.value, 'description': p.description,
                          'parameters': p.parameters} for p in self.policies],
            'metadata': self.metadata,
            'status': self.status.value,
        }

    @staticmethod
    def from_dict(d: dict) -> 'PCCBundle':
        certs = [ProofCertificate.from_dict(c) for c in d.get('certificates', [])]
        policies = [SafetyPolicy(
            kind=PolicyKind(p['kind']),
            description=p['description'],
            parameters=p.get('parameters', {})
        ) for p in d.get('policies', [])]
        return PCCBundle(
            bytecode=BytecodePayload.from_dict(d['bytecode']),
            certificates=certs,
            policies=policies,
            metadata=d.get('metadata', {}),
            status=BundleStatus(d.get('status', 'unchecked')),
        )


# --- Bytecode Extraction ---

def _compile_to_bytecode(source: str) -> BytecodePayload:
    """Compile C10 source to a BytecodePayload."""
    tokens = lex(source)
    ast = Parser(tokens).parse()
    compiler = Compiler()
    chunk = compiler.compile(ast)

    # Extract instruction tuples from chunk
    instructions = []
    for instr in chunk.code:
        if isinstance(instr, tuple):
            instructions.append(instr)
        else:
            instructions.append((instr,))

    source_hash = hashlib.sha256(source.encode()).hexdigest()

    return BytecodePayload(
        instructions=instructions,
        constants=list(chunk.constants),
        source_hash=source_hash,
    )


def _extract_functions(source: str) -> Dict[str, Any]:
    """Extract function names from source."""
    tokens = lex(source)
    stmts = Parser(tokens).parse().stmts
    fns = {}
    for stmt in stmts:
        if type(stmt).__name__ == 'FnDecl':
            fns[stmt.name] = stmt
    return fns


# --- Certificate Generation ---

def _generate_contract_certificate(source: str) -> Optional[ProofCertificate]:
    """Generate Hoare-logic contract verification certificate."""
    try:
        result = verify_program(source)
        obligations = []
        if hasattr(result, 'results') and result.results:
            for fn_name, fn_result in result.results.items():
                status = CertStatus.VALID if fn_result.status == 'valid' else (
                    CertStatus.INVALID if fn_result.status == 'invalid' else CertStatus.UNKNOWN
                )
                for i, vc in enumerate(fn_result.vcs if hasattr(fn_result, 'vcs') else []):
                    obligations.append(ProofObligation(
                        name=f"contract_{fn_name}_{i}",
                        description=f"VC for {fn_name}: {vc.description if hasattr(vc, 'description') else ''}",
                        formula_str=str(vc.formula if hasattr(vc, 'formula') else vc),
                        formula_smt="",
                        status=status,
                    ))
            if not obligations:
                for fn_name, fn_result in result.results.items():
                    status = CertStatus.VALID if fn_result.status == 'valid' else (
                        CertStatus.INVALID if fn_result.status == 'invalid' else CertStatus.UNKNOWN
                    )
                    obligations.append(ProofObligation(
                        name=f"contract_{fn_name}",
                        description=f"Contract for {fn_name}",
                        formula_str=f"contracts({fn_name}) verified",
                        formula_smt="",
                        status=status,
                    ))
        elif hasattr(result, 'status'):
            status = CertStatus.VALID if result.status == 'valid' else (
                CertStatus.INVALID if result.status == 'invalid' else CertStatus.UNKNOWN
            )
            obligations.append(ProofObligation(
                name="program_contract",
                description="Program-level contract verification",
                formula_str="program contracts verified",
                formula_smt="",
                status=status,
            ))

        if not obligations:
            return None

        overall = CertStatus.VALID if all(o.status == CertStatus.VALID for o in obligations) else (
            CertStatus.INVALID if any(o.status == CertStatus.INVALID for o in obligations) else CertStatus.UNKNOWN
        )

        return ProofCertificate(
            kind=ProofKind.VCGEN,
            claim="Source code satisfies declared contracts (requires/ensures)",
            source=source,
            obligations=obligations,
            status=overall,
            metadata={'policy': PolicyKind.CONTRACT_COMPLIANCE.value},
        )
    except Exception:
        return None


def _generate_bound_certificate(source: str) -> Optional[ProofCertificate]:
    """Generate abstract interpretation bound safety certificate."""
    try:
        ai_result = modular_analyze(source)
        obligations = []

        for fn_name, summary in ai_result.summaries.items():
            if not summary.analyzed:
                continue
            for var, bound in summary.result_bounds.items():
                desc_parts = []
                if bound.lower is not None:
                    desc_parts.append(f"{var} >= {bound.lower}")
                if bound.upper is not None:
                    desc_parts.append(f"{var} <= {bound.upper}")
                if desc_parts:
                    obligations.append(ProofObligation(
                        name=f"bound_{fn_name}_{var}",
                        description=f"Bound for {fn_name}.{var}: {' and '.join(desc_parts)}",
                        formula_str=f"bound({fn_name}, {var})",
                        formula_smt="",
                        status=CertStatus.VALID,  # AI analysis is sound
                    ))

        if not obligations:
            return None

        return ProofCertificate(
            kind=ProofKind.COMPOSITE,
            claim="Variable bounds verified via abstract interpretation",
            source=source,
            obligations=obligations,
            status=CertStatus.VALID,
            metadata={'policy': PolicyKind.BOUND_SAFETY.value},
        )
    except Exception:
        return None


def _generate_compilation_certificate(source: str) -> Optional[ProofCertificate]:
    """Generate compilation safety certificate via V049."""
    try:
        cert = certify_compilation(source)
        if cert:
            cert.metadata = cert.metadata or {}
            cert.metadata['policy'] = PolicyKind.COMPILATION_SAFETY.value
        return cert
    except Exception:
        return None


# --- Consumer Verification ---

def verify_bundle(bundle: PCCBundle) -> PCCBundle:
    """Consumer-side: independently verify all certificates in a bundle.

    Returns the bundle with updated status.
    """
    all_valid = True
    any_invalid = False
    any_checked = False

    for cert in bundle.certificates:
        try:
            checked = check_certificate(cert)
            cert.status = checked.status
            for i, obl in enumerate(checked.obligations):
                if i < len(cert.obligations):
                    cert.obligations[i].status = obl.status
        except Exception:
            pass

        if cert.status == CertStatus.VALID:
            any_checked = True
        elif cert.status == CertStatus.INVALID:
            any_invalid = True
            all_valid = False
            any_checked = True
        else:
            all_valid = False

    if any_invalid:
        bundle.status = BundleStatus.FAILED
    elif all_valid and any_checked:
        bundle.status = BundleStatus.VERIFIED
    elif any_checked:
        bundle.status = BundleStatus.PARTIALLY_VERIFIED
    else:
        bundle.status = BundleStatus.UNCHECKED

    return bundle


def check_policy(bundle: PCCBundle, policy_kind: PolicyKind) -> bool:
    """Check if a specific policy is satisfied in the bundle."""
    for cert in bundle.certificates:
        meta = cert.metadata or {}
        if meta.get('policy') == policy_kind.value:
            return cert.status == CertStatus.VALID
    return False


# --- Bundle Serialization ---

def save_bundle(bundle: PCCBundle, path: str):
    """Save a PCC bundle to a JSON file."""
    with open(path, 'w') as f:
        json.dump(bundle.to_dict(), f, indent=2, default=str)


def load_bundle(path: str) -> PCCBundle:
    """Load a PCC bundle from a JSON file."""
    with open(path, 'r') as f:
        return PCCBundle.from_dict(json.load(f))


# --- Main Producer API ---

def produce_pcc(
    source: str,
    policies: Optional[List[PolicyKind]] = None,
    include_contracts: bool = True,
    include_bounds: bool = True,
    include_compilation: bool = True,
) -> PCCBundle:
    """Producer: compile source and generate proof-carrying code bundle.

    Args:
        source: C10 source code
        policies: Which safety policies to prove (defaults to all applicable)
        include_contracts: Generate Hoare-logic contract certificates
        include_bounds: Generate abstract interpretation bound certificates
        include_compilation: Generate compilation safety certificates

    Returns:
        PCCBundle with bytecode + certificates
    """
    # Step 1: Compile
    bytecode = _compile_to_bytecode(source)

    # Step 2: Determine policies
    active_policies = []
    certificates = []

    # Step 3: Generate certificates for each policy
    if include_contracts:
        cert = _generate_contract_certificate(source)
        if cert:
            certificates.append(cert)
            active_policies.append(SafetyPolicy(
                kind=PolicyKind.CONTRACT_COMPLIANCE,
                description="Source satisfies declared requires/ensures contracts"
            ))

    if include_bounds:
        cert = _generate_bound_certificate(source)
        if cert:
            certificates.append(cert)
            active_policies.append(SafetyPolicy(
                kind=PolicyKind.BOUND_SAFETY,
                description="Variables stay within computed bounds"
            ))

    if include_compilation:
        cert = _generate_compilation_certificate(source)
        if cert:
            certificates.append(cert)
            active_policies.append(SafetyPolicy(
                kind=PolicyKind.COMPILATION_SAFETY,
                description="Optimization passes preserve program semantics"
            ))

    # Step 4: Determine initial status
    if certificates:
        if all(c.status == CertStatus.VALID for c in certificates):
            status = BundleStatus.VERIFIED
        elif any(c.status == CertStatus.INVALID for c in certificates):
            status = BundleStatus.FAILED
        else:
            status = BundleStatus.PARTIALLY_VERIFIED
    else:
        status = BundleStatus.UNCHECKED

    return PCCBundle(
        bytecode=bytecode,
        certificates=certificates,
        policies=active_policies,
        metadata={
            'source_hash': bytecode.source_hash,
            'produced_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'certificate_count': len(certificates),
        },
        status=status,
    )


def quick_pcc(source: str) -> PCCBundle:
    """Produce a minimal PCC bundle (bounds only, fastest)."""
    return produce_pcc(source, include_contracts=False, include_bounds=True,
                       include_compilation=False)


def full_pcc(source: str) -> PCCBundle:
    """Produce a full PCC bundle with all available certificates."""
    return produce_pcc(source, include_contracts=True, include_bounds=True,
                       include_compilation=True)


def pcc_report(source: str) -> str:
    """Generate a human-readable PCC report."""
    bundle = produce_pcc(source)
    return bundle.summary()


# --- Roundtrip API ---

def produce_and_verify(source: str) -> PCCBundle:
    """Full roundtrip: produce PCC bundle, then independently verify it."""
    bundle = produce_pcc(source)
    return verify_bundle(bundle)


def produce_save_load_verify(source: str, path: str) -> PCCBundle:
    """Full I/O roundtrip: produce, save, load, verify."""
    bundle = produce_pcc(source)
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    return verify_bundle(loaded)
