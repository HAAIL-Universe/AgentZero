"""Tests for V066: Markov Chain Verification."""

import sys
import os
import pytest
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V044_proof_certificates'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from markov_verify import (
    Verdict, VerificationResult, ChainVerificationResult,
    verify_stochastic, verify_steady_state_smt, verify_absorption_smt,
    verify_hitting_time_smt, verify_irreducible, verify_state_type,
    verify_steady_state_unique, verify_reachability, verify_chain,
    certified_steady_state, certified_absorption, compare_numerical_vs_smt,
)
from markov_chain import (
    MarkovChain, make_chain, StateType, gambler_ruin_chain,
    random_walk_chain, steady_state_exact, absorption_probabilities,
    expected_hitting_time,
)
from proof_certificates import CertStatus, ProofKind, combine_certificates


# ===================================================================
# Section 1: Stochasticity verification
# ===================================================================

class TestStochasticity:
    def test_valid_stochastic_matrix(self):
        mc = make_chain([[0.5, 0.5], [0.3, 0.7]])
        r = verify_stochastic(mc)
        assert r.verdict == Verdict.VERIFIED
        assert r.certificate.status == CertStatus.VALID

    def test_three_state_stochastic(self):
        mc = make_chain([
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
            [0.4, 0.4, 0.2],
        ])
        r = verify_stochastic(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_absorbing_chain_stochastic(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_stochastic(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_identity_matrix_stochastic(self):
        mc = make_chain([[1, 0], [0, 1]])
        r = verify_stochastic(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_certificate_has_obligations(self):
        mc = make_chain([[0.5, 0.5], [0.3, 0.7]])
        r = verify_stochastic(mc)
        assert len(r.certificate.obligations) > 0
        assert all(ob.status == CertStatus.VALID for ob in r.certificate.obligations)


# ===================================================================
# Section 2: Steady-state bound verification (SMT)
# ===================================================================

class TestSteadyStateSMT:
    def test_symmetric_chain_equal_probs(self):
        """Symmetric 2-state chain: pi = [0.5, 0.5]."""
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.5, upper_bound=0.5)
        assert r.verdict == Verdict.VERIFIED

    def test_biased_chain_lower_bound(self):
        """Biased chain: pi[0] should be > 0.3."""
        mc = make_chain([[0.6, 0.4], [0.4, 0.6]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.49)
        assert r.verdict == Verdict.VERIFIED

    def test_biased_chain_upper_bound(self):
        mc = make_chain([[0.6, 0.4], [0.4, 0.6]])
        r = verify_steady_state_smt(mc, 0, upper_bound=0.51)
        assert r.verdict == Verdict.VERIFIED

    def test_biased_chain_tight_bounds(self):
        """pi = [0.5, 0.5] for symmetric chain."""
        mc = make_chain([[0.6, 0.4], [0.4, 0.6]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.5, upper_bound=0.5)
        assert r.verdict == Verdict.VERIFIED

    def test_refuted_lower_bound(self):
        """pi[0] = 0.5, so pi[0] >= 0.7 should be refuted."""
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.7)
        assert r.verdict == Verdict.REFUTED

    def test_refuted_upper_bound(self):
        """pi[0] = 0.5, so pi[0] <= 0.3 should be refuted."""
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0, upper_bound=0.3)
        assert r.verdict == Verdict.REFUTED

    def test_three_state_chain(self):
        """3-state chain: pi = [1/3, 1/3, 1/3] for doubly stochastic."""
        mc = make_chain([
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.33, upper_bound=0.34)
        assert r.verdict == Verdict.VERIFIED

    def test_numerical_value_included(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.4)
        assert r.numerical_value is not None
        assert abs(r.numerical_value - 0.5) < 0.01

    def test_no_bound_returns_unknown(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0)
        assert r.verdict == Verdict.UNKNOWN

    def test_certificate_generated(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.4)
        assert r.certificate is not None
        assert r.certificate.status == CertStatus.VALID

    def test_asymmetric_chain(self):
        """mc with pi = [3/7, 4/7]: verify pi[0] in [0.42, 0.44]."""
        mc = make_chain([[0.6, 0.4], [0.3, 0.7]])
        # pi[0] = 0.3/(0.4+0.3) = 3/7 ~= 0.4286
        r = verify_steady_state_smt(mc, 0, lower_bound=0.42, upper_bound=0.44)
        assert r.verdict == Verdict.VERIFIED


# ===================================================================
# Section 3: Absorption probability verification (SMT)
# ===================================================================

class TestAbsorptionSMT:
    def test_simple_absorbing_chain(self):
        """State 0 absorbing, state 2 absorbing, state 1 transient."""
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        # From state 1, absorption into state 0 has some probability
        r = verify_absorption_smt(mc, start=1, target=0, lower_bound=0.4)
        assert r.verdict == Verdict.VERIFIED

    def test_absorption_upper_bound(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_absorption_smt(mc, start=1, target=0, upper_bound=0.6)
        assert r.verdict == Verdict.VERIFIED

    def test_absorption_refuted(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        # b[1->0] = 0.5, so >= 0.8 should be refuted
        r = verify_absorption_smt(mc, start=1, target=0, lower_bound=0.8)
        assert r.verdict == Verdict.REFUTED

    def test_absorbing_start_is_target(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_absorption_smt(mc, start=0, target=0, lower_bound=1.0)
        assert r.verdict == Verdict.VERIFIED

    def test_absorbing_start_not_target(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_absorption_smt(mc, start=0, target=2, upper_bound=0.0)
        assert r.verdict == Verdict.VERIFIED

    def test_gambler_ruin_absorption(self):
        """Gambler's ruin: 4 states, fair coin."""
        mc = gambler_ruin_chain(3, p=0.5)
        # State 0 and 3 absorbing. From state 1:
        # P(ruin at 0 | start=1) = 2/3
        r = verify_absorption_smt(mc, start=1, target=0, lower_bound=0.6, upper_bound=0.7)
        assert r.verdict == Verdict.VERIFIED

    def test_non_absorbing_target_returns_unknown(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_absorption_smt(mc, start=0, target=1, lower_bound=0.5)
        assert r.verdict == Verdict.UNKNOWN

    def test_certificate_generated(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_absorption_smt(mc, start=1, target=0, lower_bound=0.4)
        assert r.certificate is not None
        assert r.certificate.status == CertStatus.VALID


# ===================================================================
# Section 4: Hitting time bound verification (SMT)
# ===================================================================

class TestHittingTimeSMT:
    def test_self_hitting_time_zero(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_hitting_time_smt(mc, start=0, target=0, max_steps=0)
        assert r.verdict == Verdict.VERIFIED

    def test_two_state_hitting_time(self):
        """E[T(0->1)] for [[0.5, 0.5], [0.5, 0.5]] = 2."""
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_hitting_time_smt(mc, start=0, target=1, max_steps=2)
        assert r.verdict == Verdict.VERIFIED

    def test_hitting_time_tight_bound(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        # E[T(0->1)] = 2, so bound of 1 should fail
        r = verify_hitting_time_smt(mc, start=0, target=1, max_steps=1)
        assert r.verdict == Verdict.REFUTED

    def test_three_state_hitting_time(self):
        """Chain: 0->1->2 with some probabilities."""
        mc = make_chain([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        # From 0, takes exactly 2 steps to reach 2
        r = verify_hitting_time_smt(mc, start=0, target=2, max_steps=2)
        assert r.verdict == Verdict.VERIFIED

    def test_hitting_time_certificate(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_hitting_time_smt(mc, start=0, target=1, max_steps=3)
        assert r.certificate is not None
        assert r.certificate.status == CertStatus.VALID

    def test_numerical_value(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_hitting_time_smt(mc, start=0, target=1, max_steps=3)
        assert r.numerical_value is not None
        assert abs(r.numerical_value - 2.0) < 0.1


# ===================================================================
# Section 5: Irreducibility verification
# ===================================================================

class TestIrreducibility:
    def test_irreducible_chain(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_irreducible(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_reducible_chain(self):
        mc = make_chain([[1, 0], [0, 1]])
        r = verify_irreducible(mc)
        assert r.verdict == Verdict.REFUTED

    def test_absorbing_chain_not_irreducible(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_irreducible(mc)
        assert r.verdict == Verdict.REFUTED

    def test_certificate_shows_classes(self):
        mc = make_chain([[1, 0], [0, 1]])
        r = verify_irreducible(mc)
        assert r.certificate is not None
        assert r.certificate.status == CertStatus.INVALID


# ===================================================================
# Section 6: State type verification
# ===================================================================

class TestStateType:
    def test_absorbing_state(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_state_type(mc, 0, StateType.ABSORBING)
        assert r.verdict == Verdict.VERIFIED

    def test_transient_state(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_state_type(mc, 1, StateType.TRANSIENT)
        assert r.verdict == Verdict.VERIFIED

    def test_wrong_type_refuted(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = verify_state_type(mc, 0, StateType.TRANSIENT)
        assert r.verdict == Verdict.REFUTED

    def test_recurrent_state(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_state_type(mc, 0, StateType.RECURRENT)
        assert r.verdict == Verdict.VERIFIED


# ===================================================================
# Section 7: Steady-state uniqueness verification
# ===================================================================

class TestUniqueness:
    def test_irreducible_chain_unique(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_unique(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_three_state_unique(self):
        mc = make_chain([
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
            [0.4, 0.4, 0.2],
        ])
        r = verify_steady_state_unique(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_reducible_chain_not_unique(self):
        """Two absorbing states: multiple steady states."""
        mc = make_chain([[1, 0], [0, 1]])
        r = verify_steady_state_unique(mc)
        assert r.verdict == Verdict.REFUTED

    def test_uniqueness_certificate(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_unique(mc)
        assert r.certificate is not None
        assert r.certificate.status == CertStatus.VALID


# ===================================================================
# Section 8: Reachability verification
# ===================================================================

class TestReachability:
    def test_direct_reachability(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_reachability(mc, source=0, target=1)
        assert r.verdict == Verdict.VERIFIED

    def test_indirect_reachability(self):
        mc = make_chain([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        r = verify_reachability(mc, source=0, target=2)
        assert r.verdict == Verdict.VERIFIED

    def test_unreachable(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.5],
        ])
        r = verify_reachability(mc, source=0, target=2)
        assert r.verdict == Verdict.REFUTED

    def test_self_reachable(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_reachability(mc, source=0, target=0)
        assert r.verdict == Verdict.VERIFIED


# ===================================================================
# Section 9: Full chain verification
# ===================================================================

class TestChainVerification:
    def test_default_checks(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_chain(mc)
        assert len(r.results) == 2
        assert r.all_verified

    def test_multiple_properties(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        props = [
            {"type": "stochastic"},
            {"type": "irreducible"},
            {"type": "uniqueness"},
            {"type": "steady_state", "state": 0, "lower_bound": 0.4, "upper_bound": 0.6},
        ]
        r = verify_chain(mc, props)
        assert len(r.results) == 4
        assert r.all_verified

    def test_composite_certificate(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_chain(mc)
        assert r.certificate is not None
        assert r.certificate.kind == ProofKind.COMPOSITE

    def test_summary(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_chain(mc)
        s = r.summary
        assert "VERIFIED" in s

    def test_mixed_verdicts(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        props = [
            {"type": "stochastic"},
            {"type": "steady_state", "state": 0, "lower_bound": 0.9},
        ]
        r = verify_chain(mc, props)
        assert not r.all_verified
        assert r.results[0].verdict == Verdict.VERIFIED
        assert r.results[1].verdict == Verdict.REFUTED


# ===================================================================
# Section 10: Certified steady-state analysis
# ===================================================================

class TestCertifiedSteadyState:
    def test_symmetric_chain(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = certified_steady_state(mc)
        assert r.all_verified
        assert len(r.results) == 2

    def test_three_state_chain(self):
        mc = make_chain([
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
            [0.4, 0.4, 0.2],
        ])
        r = certified_steady_state(mc, tolerance=0.05)
        assert r.all_verified

    def test_tight_tolerance(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = certified_steady_state(mc, tolerance=0.001)
        assert r.all_verified


# ===================================================================
# Section 11: Certified absorption analysis
# ===================================================================

class TestCertifiedAbsorption:
    def test_simple_absorbing(self):
        mc = make_chain([
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.0, 0.0, 1.0],
        ])
        r = certified_absorption(mc, tolerance=0.02)
        assert r.all_verified
        assert len(r.results) > 0

    def test_gambler_ruin(self):
        mc = gambler_ruin_chain(3, p=0.5)
        r = certified_absorption(mc, tolerance=0.05)
        assert r.all_verified


# ===================================================================
# Section 12: Numerical vs SMT comparison
# ===================================================================

class TestCompareNumericalVsSMT:
    def test_comparison_basic(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = compare_numerical_vs_smt(mc)
        assert r["agreement"]
        assert r["smt"]["stochastic"] == "verified"
        assert r["smt"]["irreducible"] == "verified"

    def test_comparison_three_state(self):
        mc = make_chain([
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
            [0.4, 0.4, 0.2],
        ])
        r = compare_numerical_vs_smt(mc)
        assert "steady_state_bounds" in r["smt"]


# ===================================================================
# Section 13: Certificate serialization roundtrip
# ===================================================================

class TestCertificateSerialization:
    def test_steady_state_cert_serializable(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_steady_state_smt(mc, 0, lower_bound=0.4)
        cert = r.certificate
        j = cert.to_json()
        loaded = type(cert).from_json(j)
        assert loaded.status == cert.status
        assert len(loaded.obligations) == len(cert.obligations)

    def test_chain_verification_cert_serializable(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_chain(mc)
        j = r.certificate.to_json()
        assert len(j) > 0


# ===================================================================
# Section 14: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_state_chain(self):
        mc = make_chain([[1.0]])
        r = verify_stochastic(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_single_state_steady_state(self):
        mc = make_chain([[1.0]])
        r = verify_steady_state_smt(mc, 0, lower_bound=1.0, upper_bound=1.0)
        assert r.verdict == Verdict.VERIFIED

    def test_single_state_uniqueness(self):
        mc = make_chain([[1.0]])
        r = verify_steady_state_unique(mc)
        assert r.verdict == Verdict.VERIFIED

    def test_unknown_property_type(self):
        mc = make_chain([[0.5, 0.5], [0.5, 0.5]])
        r = verify_chain(mc, [{"type": "nonexistent"}])
        assert r.results[0].verdict == Verdict.UNKNOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
