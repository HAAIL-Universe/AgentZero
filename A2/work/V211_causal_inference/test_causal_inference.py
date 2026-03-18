"""Tests for V211: Causal Inference."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from causal_inference import (
    CausalModel, build_smoking_cancer_model, build_frontdoor_model,
    build_instrument_model,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))
from bayesian_networks import BayesianNetwork, variable_elimination, Factor


# ---------------------------------------------------------------------------
# Helper: build simple networks for testing
# ---------------------------------------------------------------------------

def build_chain_xy():
    """Simple X -> Y (no confounding)."""
    bn = BayesianNetwork()
    bn.add_node('X', [0, 1])
    bn.add_node('Y', [0, 1])
    bn.add_edge('X', 'Y')
    bn.set_cpt('X', {(0,): 0.6, (1,): 0.4})
    bn.set_cpt('Y', {(0, 0): 0.9, (0, 1): 0.1,
                      (1, 0): 0.3, (1, 1): 0.7})
    return CausalModel(bn)


def build_confounded_xy():
    """X -> Y with confounder Z: Z -> X, Z -> Y, X -> Y."""
    bn = BayesianNetwork()
    bn.add_node('Z', [0, 1])
    bn.add_node('X', [0, 1])
    bn.add_node('Y', [0, 1])
    bn.add_edge('Z', 'X')
    bn.add_edge('Z', 'Y')
    bn.add_edge('X', 'Y')
    bn.set_cpt('Z', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('X', {(0, 0): 0.8, (0, 1): 0.2,
                      (1, 0): 0.3, (1, 1): 0.7})
    bn.set_cpt('Y', {(0, 0, 0): 0.9, (0, 0, 1): 0.1,
                      (0, 1, 0): 0.6, (0, 1, 1): 0.4,
                      (1, 0, 0): 0.7, (1, 0, 1): 0.3,
                      (1, 1, 0): 0.3, (1, 1, 1): 0.7})
    return CausalModel(bn)


def build_collider():
    """Collider: X -> Z <- Y (no direct X-Y edge)."""
    bn = BayesianNetwork()
    bn.add_node('X', [0, 1])
    bn.add_node('Y', [0, 1])
    bn.add_node('Z', [0, 1])
    bn.add_edge('X', 'Z')
    bn.add_edge('Y', 'Z')
    bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('Y', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('Z', {(0, 0, 0): 0.9, (0, 0, 1): 0.1,
                      (0, 1, 0): 0.5, (0, 1, 1): 0.5,
                      (1, 0, 0): 0.5, (1, 0, 1): 0.5,
                      (1, 1, 0): 0.1, (1, 1, 1): 0.9})
    return CausalModel(bn)


def build_mediator():
    """X -> M -> Y (mediation)."""
    bn = BayesianNetwork()
    bn.add_node('X', [0, 1])
    bn.add_node('M', [0, 1])
    bn.add_node('Y', [0, 1])
    bn.add_edge('X', 'M')
    bn.add_edge('M', 'Y')
    bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('M', {(0, 0): 0.8, (0, 1): 0.2,
                      (1, 0): 0.3, (1, 1): 0.7})
    bn.set_cpt('Y', {(0, 0): 0.9, (0, 1): 0.1,
                      (1, 0): 0.4, (1, 1): 0.6})
    return CausalModel(bn)


# ===========================================================================
# D-Separation Tests
# ===========================================================================

class TestDSeparation:

    def test_chain_blocked_by_middle(self):
        """X -> Z -> Y: X _||_ Y | Z."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Z', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Z')
        bn.add_edge('Z', 'Y')
        cm = CausalModel(bn)
        assert cm.d_separated({'X'}, {'Y'}, {'Z'}) is True

    def test_chain_not_blocked_unconditional(self):
        """X -> Z -> Y: X NOT _||_ Y | {}."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Z', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Z')
        bn.add_edge('Z', 'Y')
        cm = CausalModel(bn)
        assert cm.d_separated({'X'}, {'Y'}, set()) is False

    def test_fork_blocked_by_middle(self):
        """X <- Z -> Y: X _||_ Y | Z."""
        bn = BayesianNetwork()
        bn.add_node('Z', [0, 1])
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('Z', 'X')
        bn.add_edge('Z', 'Y')
        cm = CausalModel(bn)
        assert cm.d_separated({'X'}, {'Y'}, {'Z'}) is True

    def test_fork_not_blocked_unconditional(self):
        """X <- Z -> Y: X NOT _||_ Y | {}."""
        bn = BayesianNetwork()
        bn.add_node('Z', [0, 1])
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('Z', 'X')
        bn.add_edge('Z', 'Y')
        cm = CausalModel(bn)
        assert cm.d_separated({'X'}, {'Y'}, set()) is False

    def test_collider_blocked_unconditional(self):
        """X -> Z <- Y: X _||_ Y | {} (collider blocks)."""
        cm = build_collider()
        assert cm.d_separated({'X'}, {'Y'}, set()) is True

    def test_collider_opened_by_conditioning(self):
        """X -> Z <- Y: X NOT _||_ Y | Z (conditioning opens collider)."""
        cm = build_collider()
        assert cm.d_separated({'X'}, {'Y'}, {'Z'}) is False

    def test_collider_opened_by_descendant(self):
        """Conditioning on descendant of collider also opens it."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_node('Z', [0, 1])
        bn.add_node('W', [0, 1])
        bn.add_edge('X', 'Z')
        bn.add_edge('Y', 'Z')
        bn.add_edge('Z', 'W')
        cm = CausalModel(bn)
        # Conditioning on W (descendant of collider Z) opens the path
        assert cm.d_separated({'X'}, {'Y'}, {'W'}) is False

    def test_self_dsep(self):
        """A node is not d-separated from itself."""
        cm = build_chain_xy()
        assert cm.d_separated({'X'}, {'X'}, set()) is False

    def test_disconnected_nodes(self):
        """Disconnected nodes are d-separated."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        cm = CausalModel(bn)
        assert cm.d_separated({'X'}, {'Y'}, set()) is True


# ===========================================================================
# Intervention (do-operator) Tests
# ===========================================================================

class TestIntervention:

    def test_do_removes_incoming_edges(self):
        """do(X=x) should remove all edges into X."""
        cm = build_confounded_xy()
        mutilated = cm.do({'X': 1})
        assert mutilated.parents['X'] == []
        assert 'Z' not in mutilated.parents['X']

    def test_do_preserves_outgoing_edges(self):
        """do(X=x) keeps edges from X to children."""
        cm = build_confounded_xy()
        mutilated = cm.do({'X': 1})
        assert 'X' in mutilated.parents['Y']

    def test_do_sets_deterministic_cpt(self):
        """Intervened variable has deterministic CPT."""
        cm = build_confounded_xy()
        mutilated = cm.do({'X': 1})
        x_cpt = mutilated.cpts['X']
        assert x_cpt.get({'X': 1}) == 1.0
        assert x_cpt.get({'X': 0}) == 0.0

    def test_do_preserves_other_cpts(self):
        """Non-intervened CPTs remain unchanged."""
        cm = build_confounded_xy()
        mutilated = cm.do({'X': 1})
        # Z's CPT should be unchanged
        for assignment, prob in cm.bn.cpts['Z'].table.items():
            assert mutilated.cpts['Z'].table[assignment] == prob

    def test_do_no_confounding_same_as_conditioning(self):
        """Without confounders, P(Y|do(X)) = P(Y|X)."""
        cm = build_chain_xy()
        # P(Y|do(X=1))
        do_result = cm.interventional_query(['Y'], {'X': 1}).normalize()
        # P(Y|X=1)
        cond_result = variable_elimination(cm.bn, ['Y'], {'X': 1}).normalize()
        for y_val in [0, 1]:
            assert abs(do_result.get({'Y': y_val}) - cond_result.get({'Y': y_val})) < 1e-10

    def test_do_with_confounding_differs_from_conditioning(self):
        """With confounders, P(Y|do(X)) != P(Y|X) in general."""
        cm = build_smoking_cancer_model()
        do_result = cm.interventional_query(['Y'], {'Smoking': 1}).normalize()
        cond_result = variable_elimination(cm.bn, ['Cancer'], {'Smoking': 1}).normalize()
        # These should differ due to confounding
        do_p = do_result.get({'Y': 1})
        cond_p = cond_result.get({'Cancer': 1})
        assert abs(do_p - cond_p) > 0.01  # They must differ

    def test_do_multiple_interventions(self):
        """Multiple simultaneous interventions."""
        cm = build_confounded_xy()
        mutilated = cm.do({'X': 1, 'Z': 0})
        assert mutilated.parents['X'] == []
        assert mutilated.parents['Z'] == []
        assert mutilated.cpts['X'].get({'X': 1}) == 1.0
        assert mutilated.cpts['Z'].get({'Z': 0}) == 1.0

    def test_interventional_query_probabilities_sum_to_one(self):
        """P(Y | do(X=x)) should sum to 1."""
        cm = build_smoking_cancer_model()
        result = cm.interventional_query(['Cancer'], {'Smoking': 1}).normalize()
        total = sum(result.get({'Cancer': v}) for v in [0, 1])
        assert abs(total - 1.0) < 1e-10

    def test_do_on_root_node(self):
        """Intervening on a root node is equivalent to changing its prior."""
        cm = build_chain_xy()
        do_result = cm.interventional_query(['Y'], {'X': 1}).normalize()
        # Same as P(Y | X=1) since X has no parents
        cond_result = variable_elimination(cm.bn, ['Y'], {'X': 1}).normalize()
        for y_val in [0, 1]:
            assert abs(do_result.get({'Y': y_val}) - cond_result.get({'Y': y_val})) < 1e-10


# ===========================================================================
# Backdoor Criterion Tests
# ===========================================================================

class TestBackdoorCriterion:

    def test_empty_set_unconfounded(self):
        """No confounders: empty set is valid backdoor adjustment."""
        cm = build_chain_xy()
        assert cm.backdoor_criterion('X', 'Y', set()) is True

    def test_confounder_blocks_empty_set(self):
        """Z -> X, Z -> Y: empty set is not valid (Z confounds)."""
        cm = build_confounded_xy()
        assert cm.backdoor_criterion('X', 'Y', set()) is False

    def test_confounder_adjusted(self):
        """Z -> X, Z -> Y: {Z} is valid backdoor adjustment."""
        cm = build_confounded_xy()
        assert cm.backdoor_criterion('X', 'Y', {'Z'}) is True

    def test_descendant_invalid(self):
        """Descendant of X cannot be in adjustment set."""
        cm = build_mediator()
        # M is a descendant of X, so {M} fails condition 1
        assert cm.backdoor_criterion('X', 'Y', {'M'}) is False

    def test_collider_not_needed(self):
        """Collider Z not needed in adjustment set."""
        cm = build_collider()
        # X -> Z <- Y: X and Y are d-separated unconditionally
        assert cm.backdoor_criterion('X', 'Y', set()) is True

    def test_collider_in_set_opens_path(self):
        """Adding collider to adjustment set opens spurious path."""
        cm = build_collider()
        # Conditioning on Z opens X -> Z <- Y
        assert cm.backdoor_criterion('X', 'Y', {'Z'}) is False

    def test_find_backdoor_set_confounded(self):
        """Automatically find {Z} for confounded model."""
        cm = build_confounded_xy()
        z_set = cm.find_backdoor_set('X', 'Y')
        assert z_set is not None
        assert cm.backdoor_criterion('X', 'Y', z_set) is True

    def test_find_backdoor_set_unconfounded(self):
        """Find empty set for unconfounded model."""
        cm = build_chain_xy()
        z_set = cm.find_backdoor_set('X', 'Y')
        assert z_set is not None
        assert z_set == set()

    def test_smoking_model_backdoor(self):
        """Smoking model: U is valid adjustment set."""
        cm = build_smoking_cancer_model()
        assert cm.backdoor_criterion('Smoking', 'Cancer', {'U'}) is True


# ===========================================================================
# Frontdoor Criterion Tests
# ===========================================================================

class TestFrontdoorCriterion:

    def test_frontdoor_valid(self):
        """Frontdoor model: M is valid frontdoor variable."""
        cm = build_frontdoor_model()
        assert cm.frontdoor_criterion('X', 'Y', {'M'}) is True

    def test_frontdoor_invalid_direct_path(self):
        """If there's a direct X -> Y path, M doesn't intercept all paths."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('M', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'M')
        bn.add_edge('M', 'Y')
        bn.add_edge('X', 'Y')  # Direct path bypasses M
        cm = CausalModel(bn)
        assert cm.frontdoor_criterion('X', 'Y', {'M'}) is False


# ===========================================================================
# Backdoor Adjustment Formula Tests
# ===========================================================================

class TestBackdoorAdjustment:

    def test_no_confounder_matches_conditioning(self):
        """Without confounders, adjustment = conditioning."""
        cm = build_chain_xy()
        adj = cm.backdoor_adjustment('X', 'Y', 1)
        cond = variable_elimination(cm.bn, ['Y'], {'X': 1}).normalize()
        for y_val in [0, 1]:
            assert abs(adj[y_val] - cond.get({'Y': y_val})) < 1e-10

    def test_adjustment_sums_to_one(self):
        """Adjustment formula output should sum to 1."""
        cm = build_confounded_xy()
        adj = cm.backdoor_adjustment('X', 'Y', 1, z={'Z'})
        total = sum(adj.values())
        assert abs(total - 1.0) < 1e-10

    def test_adjustment_matches_do_calculus(self):
        """Backdoor adjustment should match graph surgery result."""
        cm = build_confounded_xy()
        adj = cm.backdoor_adjustment('X', 'Y', 1, z={'Z'})
        do_result = cm.causal_effect('X', 'Y', 1)
        for y_val in [0, 1]:
            assert abs(adj[y_val] - do_result[y_val]) < 1e-10

    def test_auto_find_adjustment_set(self):
        """Backdoor adjustment auto-finds valid Z."""
        cm = build_confounded_xy()
        adj = cm.backdoor_adjustment('X', 'Y', 1)
        assert abs(sum(adj.values()) - 1.0) < 1e-10


# ===========================================================================
# Frontdoor Adjustment Tests
# ===========================================================================

class TestFrontdoorAdjustment:

    def test_frontdoor_sums_to_one(self):
        """Frontdoor adjustment output sums to 1."""
        cm = build_frontdoor_model()
        adj = cm.frontdoor_adjustment('X', 'Y', 'M', 0)
        total = sum(adj.values())
        assert abs(total - 1.0) < 1e-10

    def test_frontdoor_matches_do(self):
        """Frontdoor adjustment should approximate graph surgery.

        Note: frontdoor works even when U is unobserved, but in our model
        U is explicitly present so we can compare with do-calculus directly.
        """
        cm = build_frontdoor_model()
        fd_0 = cm.frontdoor_adjustment('X', 'Y', 'M', 0)
        fd_1 = cm.frontdoor_adjustment('X', 'Y', 'M', 1)
        do_0 = cm.causal_effect('X', 'Y', 0)
        do_1 = cm.causal_effect('X', 'Y', 1)
        # They should be close (exact if U is marginalized out correctly)
        for y_val in [0, 1]:
            assert abs(fd_0[y_val] - do_0[y_val]) < 0.05
            assert abs(fd_1[y_val] - do_1[y_val]) < 0.05

    def test_frontdoor_different_interventions(self):
        """Different X values should give different P(Y|do(X))."""
        cm = build_frontdoor_model()
        fd_0 = cm.frontdoor_adjustment('X', 'Y', 'M', 0)
        fd_1 = cm.frontdoor_adjustment('X', 'Y', 'M', 1)
        # Should not be identical
        assert abs(fd_0[0] - fd_1[0]) > 0.01 or abs(fd_0[1] - fd_1[1]) > 0.01


# ===========================================================================
# Causal Effect Estimation Tests
# ===========================================================================

class TestCausalEffects:

    def test_ate_positive(self):
        """Smoking has positive ATE on cancer."""
        cm = build_smoking_cancer_model()
        ate = cm.average_treatment_effect('Smoking', 'Cancer', 1, 0)
        assert ate > 0  # Smoking increases cancer

    def test_ate_zero_no_effect(self):
        """ATE is 0 when X has no effect on Y."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
        bn.set_cpt('Y', {(0,): 0.6, (1,): 0.4})  # Y independent of X
        cm = CausalModel(bn)
        ate = cm.average_treatment_effect('X', 'Y', 1, 0)
        assert abs(ate) < 1e-10

    def test_cde(self):
        """Controlled direct effect with mediator fixed."""
        cm = build_mediator()
        cde = cm.controlled_direct_effect('X', 'Y', {'M'}, 1, 0, {'M': 0})
        # With M fixed, X has no direct path to Y, so CDE should be 0
        assert abs(cde) < 1e-10

    def test_nde_plus_nie_equals_ate(self):
        """NDE + NIE = ATE (total effect decomposition)."""
        cm = build_mediator()
        ate = cm.average_treatment_effect('X', 'Y', 1, 0)
        nde = cm.natural_direct_effect('X', 'Y', 'M', 1, 0)
        nie = cm.natural_indirect_effect('X', 'Y', 'M', 1, 0)
        assert abs(nde + nie - ate) < 1e-10

    def test_pure_mediation_nde_zero(self):
        """When X -> M -> Y only (no direct X -> Y), NDE should be 0."""
        cm = build_mediator()
        nde = cm.natural_direct_effect('X', 'Y', 'M', 1, 0)
        assert abs(nde) < 1e-10

    def test_pure_mediation_nie_equals_ate(self):
        """When X -> M -> Y only, NIE should equal ATE."""
        cm = build_mediator()
        ate = cm.average_treatment_effect('X', 'Y', 1, 0)
        nie = cm.natural_indirect_effect('X', 'Y', 'M', 1, 0)
        assert abs(nie - ate) < 1e-10

    def test_causal_effect_all(self):
        """causal_effect_all returns effects for all X values."""
        cm = build_chain_xy()
        effects = cm.causal_effect_all('X', 'Y')
        assert 0 in effects
        assert 1 in effects
        for x_val in [0, 1]:
            total = sum(effects[x_val].values())
            assert abs(total - 1.0) < 1e-10


# ===========================================================================
# Counterfactual Tests
# ===========================================================================

class TestCounterfactuals:

    def test_counterfactual_deterministic(self):
        """If Y = f(X) deterministically, counterfactual is exact."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Y')
        bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
        # Y = X (deterministic)
        bn.set_cpt('Y', {(0, 0): 1.0, (0, 1): 0.0,
                          (1, 0): 0.0, (1, 1): 1.0})
        cm = CausalModel(bn)
        # "If X had been 1, Y would have been 1" (given X=0, Y=0 observed)
        result = cm.counterfactual_query('Y', {'X': 1}, {'X': 0, 'Y': 0})
        assert abs(result[1] - 1.0) < 1e-10

    def test_counterfactual_probabilistic(self):
        """Counterfactual with probabilistic mechanism gives valid distribution."""
        cm = build_chain_xy()
        result = cm.counterfactual_query('Y', {'X': 1}, {'X': 0})
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-10
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_counterfactual_with_confounder(self):
        """Counterfactual reasoning with confounders."""
        cm = build_confounded_xy()
        result = cm.counterfactual_query('Y', {'X': 1}, {'X': 0, 'Y': 0})
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-10

    def test_factual_agrees_with_observation(self):
        """If counterfactual matches factual intervention, should agree with evidence.

        If we observe X=1, Y=1, and ask "what if X had been 1?" (same as factual),
        answer should be P(Y=1) = 1.
        """
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Y')
        bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
        bn.set_cpt('Y', {(0, 0): 1.0, (0, 1): 0.0,
                          (1, 0): 0.0, (1, 1): 1.0})
        cm = CausalModel(bn)
        # Factual: X=1, Y=1. Counterfactual: do(X=1). Should give Y=1.
        result = cm.counterfactual_query('Y', {'X': 1}, {'X': 1, 'Y': 1})
        assert abs(result[1] - 1.0) < 1e-10


# ===========================================================================
# Instrumental Variable Tests
# ===========================================================================

class TestInstrumentalVariables:

    def test_valid_instrument(self):
        """Z is a valid instrument in the IV model."""
        cm = build_instrument_model()
        assert cm.is_instrument('Z', 'X', 'Y') is True

    def test_confounder_not_instrument(self):
        """A confounder of X and Y is not a valid instrument."""
        cm = build_instrument_model()
        assert cm.is_instrument('U', 'X', 'Y') is False

    def test_x_not_instrument_for_itself(self):
        """X is not an instrument for X -> Y."""
        cm = build_instrument_model()
        assert cm.is_instrument('X', 'X', 'Y') is False

    def test_direct_cause_not_instrument(self):
        """A direct cause of Y (through non-X path) is not an instrument."""
        bn = BayesianNetwork()
        bn.add_node('Z', [0, 1])
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('Z', 'X')
        bn.add_edge('Z', 'Y')  # Z directly affects Y
        bn.add_edge('X', 'Y')
        cm = CausalModel(bn)
        assert cm.is_instrument('Z', 'X', 'Y') is False


# ===========================================================================
# do-Calculus Rules Tests
# ===========================================================================

class TestDoCalculusRules:

    def test_rule1_disconnected(self):
        """Rule 1: disconnected Z can be ignored in P(Y|do(X), Z)."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Z', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Y')
        # Z is disconnected from Y
        cm = CausalModel(bn)
        # Rule 1: P(Y|do(X), Z) = P(Y|do(X)) iff Y _||_ Z | X in G_overbar_X
        # In G_overbar_X, Z is disconnected from Y, so they're d-separated
        assert cm.rule1_holds({'Y'}, {'X'}, {'Z'}, set()) is True

    def test_rule1_fails_when_connected(self):
        """Rule 1 fails when Z is informative about Y given do(X)."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Z', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Z')
        bn.add_edge('Z', 'Y')
        cm = CausalModel(bn)
        # Z is on the path X -> Z -> Y, so Z is informative about Y
        assert cm.rule1_holds({'Y'}, {'X'}, {'Z'}, set()) is False

    def test_rule2_fork(self):
        """Rule 2 in confounded model: can exchange do(Z) with observation Z."""
        cm = build_confounded_xy()
        # Rule 2: P(Y|do(X),do(Z)) = P(Y|do(X),Z) in certain graphs
        # Here Z -> X, Z -> Y, X -> Y
        # G_overbar_X_underbar_Z: remove incoming to X, remove outgoing from Z
        # => no edges involving Z or into X
        # Check: Y _||_ Z | X in this graph
        r2 = cm.rule2_holds({'Y'}, {'X'}, {'Z'}, set())
        # After removing X's incoming and Z's outgoing, Z is isolated
        assert r2 is True

    def test_rule3_action_deletion(self):
        """Rule 3: can delete do(Z) when appropriate."""
        cm = build_chain_xy()
        # In X -> Y, check if do on a disconnected variable can be deleted
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_node('Z', [0, 1])
        bn.add_edge('X', 'Y')
        cm2 = CausalModel(bn)
        # Z is disconnected from everything
        assert cm2.rule3_holds({'Y'}, {'X'}, {'Z'}, set()) is True


# ===========================================================================
# Builder Model Tests
# ===========================================================================

class TestBuilderModels:

    def test_smoking_model_valid(self):
        """Smoking model is a valid BN."""
        cm = build_smoking_cancer_model()
        assert 'Smoking' in cm.bn.nodes
        assert 'Cancer' in cm.bn.nodes
        assert 'U' in cm.bn.nodes

    def test_smoking_confounding_effect(self):
        """Demonstrate confounding: P(Cancer|Smoking=1) != P(Cancer|do(Smoking=1))."""
        cm = build_smoking_cancer_model()
        # Observational
        cond = variable_elimination(cm.bn, ['Cancer'], {'Smoking': 1}).normalize()
        p_cond = cond.get({'Cancer': 1})
        # Interventional
        do_result = cm.causal_effect('Smoking', 'Cancer', 1)
        p_do = do_result[1]
        # These should differ
        assert abs(p_cond - p_do) > 0.01

    def test_frontdoor_model_valid(self):
        """Frontdoor model structure is correct."""
        cm = build_frontdoor_model()
        assert cm.bn.parents['M'] == ['X']
        assert 'M' in cm.bn.parents['Y']

    def test_instrument_model_valid(self):
        """Instrument model structure is correct."""
        cm = build_instrument_model()
        assert 'Z' in cm.bn.parents['X']
        assert 'U' in cm.bn.parents['X']
        assert 'X' in cm.bn.parents['Y']

    def test_frontdoor_model_identifies_frontdoor(self):
        """M is a valid frontdoor variable in the frontdoor model."""
        cm = build_frontdoor_model()
        assert cm.frontdoor_criterion('X', 'Y', {'M'}) is True

    def test_instrument_model_identifies_instrument(self):
        """Z is a valid instrument in the IV model."""
        cm = build_instrument_model()
        assert cm.is_instrument('Z', 'X', 'Y') is True


# ===========================================================================
# Edge cases and integration tests
# ===========================================================================

class TestEdgeCases:

    def test_do_on_leaf_node(self):
        """Intervening on a leaf node."""
        cm = build_chain_xy()
        mutilated = cm.do({'Y': 1})
        assert mutilated.parents['Y'] == []
        assert mutilated.cpts['Y'].get({'Y': 1}) == 1.0

    def test_causal_effect_probabilities_valid(self):
        """All causal effects produce valid distributions."""
        cm = build_smoking_cancer_model()
        for x_val in [0, 1]:
            effect = cm.causal_effect('Smoking', 'Cancer', x_val)
            total = sum(effect.values())
            assert abs(total - 1.0) < 1e-10
            for p in effect.values():
                assert 0.0 <= p <= 1.0

    def test_empty_do_is_observational(self):
        """do({}) should give same result as observational query."""
        cm = build_chain_xy()
        mutilated = cm.do({})
        # Should be identical to original
        result = variable_elimination(mutilated, ['Y']).normalize()
        orig = variable_elimination(cm.bn, ['Y']).normalize()
        for y_val in [0, 1]:
            assert abs(result.get({'Y': y_val}) - orig.get({'Y': y_val})) < 1e-10

    def test_backdoor_adjustment_auto_vs_manual(self):
        """Auto-found adjustment set gives same result as manual."""
        cm = build_confounded_xy()
        auto = cm.backdoor_adjustment('X', 'Y', 1)
        manual = cm.backdoor_adjustment('X', 'Y', 1, z={'Z'})
        for y_val in [0, 1]:
            assert abs(auto[y_val] - manual[y_val]) < 1e-10

    def test_four_node_diamond(self):
        """Diamond: X -> A, X -> B, A -> Y, B -> Y."""
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('A', [0, 1])
        bn.add_node('B', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'A')
        bn.add_edge('X', 'B')
        bn.add_edge('A', 'Y')
        bn.add_edge('B', 'Y')
        bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
        bn.set_cpt('A', {(0, 0): 0.8, (0, 1): 0.2,
                          (1, 0): 0.3, (1, 1): 0.7})
        bn.set_cpt('B', {(0, 0): 0.7, (0, 1): 0.3,
                          (1, 0): 0.4, (1, 1): 0.6})
        bn.set_cpt('Y', {(0, 0, 0): 0.9, (0, 0, 1): 0.1,
                          (0, 1, 0): 0.6, (0, 1, 1): 0.4,
                          (1, 0, 0): 0.5, (1, 0, 1): 0.5,
                          (1, 1, 0): 0.2, (1, 1, 1): 0.8})
        cm = CausalModel(bn)
        # No confounders, so do(X) = conditioning
        do_result = cm.causal_effect('X', 'Y', 1)
        cond = variable_elimination(cm.bn, ['Y'], {'X': 1}).normalize()
        for y_val in [0, 1]:
            assert abs(do_result[y_val] - cond.get({'Y': y_val})) < 1e-10

    def test_three_variable_causal_chain(self):
        """X -> M -> Y: do(X) propagates through M."""
        cm = build_mediator()
        do_0 = cm.causal_effect('X', 'Y', 0)
        do_1 = cm.causal_effect('X', 'Y', 1)
        # Different interventions should give different results
        assert abs(do_0[1] - do_1[1]) > 0.01

    def test_smoking_ate_sign(self):
        """Smoking ATE on cancer should be positive."""
        cm = build_smoking_cancer_model()
        ate = cm.average_treatment_effect('Smoking', 'Cancer', 1, 0)
        assert ate > 0

    def test_descendants_computation(self):
        """Descendants are computed correctly."""
        cm = build_mediator()
        desc = cm._descendants({'X'})
        assert 'M' in desc
        assert 'Y' in desc
        assert 'X' not in desc

    def test_ancestors_computation(self):
        """Ancestors are computed correctly."""
        cm = build_mediator()
        anc = cm._ancestors({'Y'}, cm.bn.parents)
        assert 'M' in anc
        assert 'X' in anc
        assert 'Y' not in anc


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
