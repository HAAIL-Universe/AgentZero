"""Tests for C163: Structural Equation Model."""

import pytest
import math
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sem import (
    StructuralEquation, LinearSEM, NonlinearSEM,
    SEMIdentification, SEMIntervention, SEMCounterfactual,
    SEMEstimation, SEMFitMetrics, SEMSimulator, SEMAnalyzer
)


# ===== StructuralEquation =====

class TestStructuralEquation:
    def test_linear_equation_basic(self):
        eq = StructuralEquation('Y', parents=['X'], coefficients={'X': 2.0}, intercept=1.0, noise_std=0.0)
        assert eq.variable == 'Y'
        assert eq.is_linear
        val = eq.evaluate({'X': 3.0}, noise=0.0)
        assert val == 7.0  # 1 + 2*3

    def test_linear_equation_multiple_parents(self):
        eq = StructuralEquation('Y', parents=['X1', 'X2'],
                                coefficients={'X1': 0.5, 'X2': -1.0}, intercept=2.0, noise_std=0.0)
        val = eq.evaluate({'X1': 4.0, 'X2': 1.0}, noise=0.0)
        assert val == 3.0  # 2 + 0.5*4 + (-1)*1

    def test_linear_equation_with_noise(self):
        eq = StructuralEquation('Y', parents=['X'], coefficients={'X': 1.0}, noise_std=0.0)
        val = eq.evaluate({'X': 5.0}, noise=2.0)
        assert val == 7.0  # 0 + 1*5 + 2

    def test_nonlinear_equation(self):
        eq = StructuralEquation('Y', parents=['X'],
                                func=lambda pv, n: pv['X']**2 + n, noise_std=0.0)
        assert not eq.is_linear
        val = eq.evaluate({'X': 3.0}, noise=0.0)
        assert val == 9.0

    def test_exogenous_variable(self):
        eq = StructuralEquation('U', noise_std=1.0)
        # Should just return noise
        val = eq.evaluate({}, noise=5.0)
        assert val == 5.0

    def test_repr_linear(self):
        eq = StructuralEquation('Y', parents=['X'], coefficients={'X': 2.0}, intercept=1.0)
        r = repr(eq)
        assert 'Y' in r
        assert 'X' in r

    def test_repr_nonlinear(self):
        eq = StructuralEquation('Y', parents=['X'], func=lambda pv, n: 0)
        r = repr(eq)
        assert 'f(' in r

    def test_missing_parent_defaults_zero(self):
        eq = StructuralEquation('Y', parents=['X', 'Z'], coefficients={'X': 1.0}, noise_std=0.0)
        val = eq.evaluate({'X': 2.0}, noise=0.0)
        assert val == 2.0  # Z coefficient defaults to 0

    def test_noise_sampling(self):
        eq = StructuralEquation('Y', noise_std=1.0)
        random.seed(42)
        vals = [eq.evaluate({}) for _ in range(100)]
        assert abs(np.mean(vals)) < 0.5
        assert np.std(vals) > 0.5

    def test_zero_noise_std(self):
        eq = StructuralEquation('Y', noise_std=0.0)
        val = eq.evaluate({})
        assert val == 0.0


# ===== LinearSEM =====

class TestLinearSEM:
    def _simple_sem(self):
        """X -> M -> Y with X -> Y"""
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('M', ['X'], {'X': 0.5}, noise_std=0.5)
        sem.add_equation('Y', ['X', 'M'], {'X': 0.3, 'M': 0.8}, noise_std=0.5)
        return sem

    def test_add_variable(self):
        sem = LinearSEM()
        sem.add_variable('X')
        assert 'X' in sem.variables
        assert 'X' in sem.equations

    def test_add_equation(self):
        sem = LinearSEM()
        sem.add_equation('Y', ['X'], {'X': 2.0})
        assert 'Y' in sem.variables
        assert sem.equations['Y'].coefficients['X'] == 2.0

    def test_add_equation_list_coefficients(self):
        sem = LinearSEM()
        sem.add_equation('Y', ['X1', 'X2'], [1.0, 2.0])
        assert sem.equations['Y'].coefficients['X1'] == 1.0
        assert sem.equations['Y'].coefficients['X2'] == 2.0

    def test_path_matrix(self):
        sem = self._simple_sem()
        B = sem.get_path_matrix()
        xi, mi, yi = sem._var_index['X'], sem._var_index['M'], sem._var_index['Y']
        assert B[mi, xi] == 0.5
        assert B[yi, xi] == 0.3
        assert B[yi, mi] == 0.8
        assert B[xi, mi] == 0.0

    def test_noise_covariance(self):
        sem = self._simple_sem()
        Sigma = sem.get_noise_covariance()
        xi = sem._var_index['X']
        assert Sigma[xi, xi] == 1.0

    def test_topological_order(self):
        sem = self._simple_sem()
        order = sem.topological_order()
        assert order.index('X') < order.index('M')
        assert order.index('M') < order.index('Y')

    def test_sample_basic(self):
        sem = self._simple_sem()
        np.random.seed(42)
        data = sem.sample(1000)
        assert len(data['X']) == 1000
        assert len(data['M']) == 1000
        assert len(data['Y']) == 1000

    def test_sample_intervention(self):
        sem = self._simple_sem()
        np.random.seed(42)
        data = sem.sample(1000, interventions={'X': 2.0})
        assert np.all(data['X'] == 2.0)
        assert abs(np.mean(data['M']) - 1.0) < 0.1  # 0.5 * 2

    def test_sample_with_noise_values(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 1.0}, noise_std=1.0)
        data = sem.sample(3, noise_values={'X': [1, 2, 3], 'Y': [0, 0, 0]})
        np.testing.assert_array_almost_equal(data['X'], [1, 2, 3])
        np.testing.assert_array_almost_equal(data['Y'], [1, 2, 3])  # 1*X + 0

    def test_total_effect_direct(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('Y', ['X'], {'X': 3.0})
        assert abs(sem.total_effect('X', 'Y') - 3.0) < 1e-10

    def test_total_effect_indirect(self):
        sem = self._simple_sem()
        # Total = direct + indirect = 0.3 + 0.5*0.8 = 0.7
        total = sem.total_effect('X', 'Y')
        assert abs(total - 0.7) < 1e-10

    def test_direct_effect(self):
        sem = self._simple_sem()
        assert abs(sem.direct_effect('X', 'Y') - 0.3) < 1e-10
        assert abs(sem.direct_effect('X', 'M') - 0.5) < 1e-10

    def test_indirect_effect(self):
        sem = self._simple_sem()
        indirect = sem.indirect_effect('X', 'Y')
        assert abs(indirect - 0.4) < 1e-10  # 0.5 * 0.8

    def test_implied_covariance(self):
        sem = self._simple_sem()
        cov = sem.implied_covariance()
        assert cov.shape == (3, 3)
        # Diagonal should be positive
        for i in range(3):
            assert cov[i, i] > 0

    def test_to_causal_graph(self):
        sem = self._simple_sem()
        g = sem.to_causal_graph()
        assert 'X' in g.nodes
        assert 'M' in g.nodes
        assert 'Y' in g.nodes
        assert ('X', 'M') in g.edges
        assert ('X', 'Y') in g.edges

    def test_no_effect(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_variable('Y')
        assert sem.direct_effect('X', 'Y') == 0.0
        assert abs(sem.total_effect('X', 'Y')) < 1e-10

    def test_chain_total_effect(self):
        sem = LinearSEM()
        sem.add_variable('A')
        sem.add_equation('B', ['A'], {'A': 2.0}, noise_std=0.0)
        sem.add_equation('C', ['B'], {'B': 3.0}, noise_std=0.0)
        assert abs(sem.total_effect('A', 'C') - 6.0) < 1e-10

    def test_diamond_total_effect(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('M1', ['X'], {'X': 1.0}, noise_std=0.0)
        sem.add_equation('M2', ['X'], {'X': 2.0}, noise_std=0.0)
        sem.add_equation('Y', ['M1', 'M2'], {'M1': 1.0, 'M2': 1.0}, noise_std=0.0)
        # Total = 1*1 + 2*1 = 3
        assert abs(sem.total_effect('X', 'Y') - 3.0) < 1e-10


# ===== NonlinearSEM =====

class TestNonlinearSEM:
    def test_basic_nonlinear(self):
        sem = NonlinearSEM()
        sem.add_equation('X', noise_std=1.0)
        sem.add_equation('Y', parents=['X'],
                         func=lambda pv, n: pv['X']**2 + n, noise_std=0.0)
        data = sem.sample(100, noise_values={'X': np.ones(100) * 3.0, 'Y': np.zeros(100)})
        np.testing.assert_array_almost_equal(data['Y'], np.ones(100) * 9.0)

    def test_nonlinear_intervention(self):
        sem = NonlinearSEM()
        sem.add_equation('X', noise_std=1.0)
        sem.add_equation('Y', parents=['X'],
                         func=lambda pv, n: pv['X']**2 + n, noise_std=0.0)
        data = sem.sample(100, interventions={'X': 4.0})
        np.testing.assert_array_almost_equal(data['Y'], np.ones(100) * 16.0)

    def test_nonlinear_multiple_parents(self):
        sem = NonlinearSEM()
        sem.add_equation('X1', noise_std=0.0)
        sem.add_equation('X2', noise_std=0.0)
        sem.add_equation('Y', parents=['X1', 'X2'],
                         func=lambda pv, n: pv['X1'] * pv['X2'] + n, noise_std=0.0)
        data = sem.sample(1, noise_values={'X1': [3.0], 'X2': [4.0], 'Y': [0.0]})
        assert abs(data['Y'][0] - 12.0) < 1e-10

    def test_nonlinear_topological_order(self):
        sem = NonlinearSEM()
        sem.add_equation('Y', parents=['X'], func=lambda pv, n: pv['X'] + n)
        sem.add_equation('X', noise_std=1.0)
        order = sem.topological_order()
        assert order.index('X') < order.index('Y')

    def test_nonlinear_counterfactual(self):
        sem = NonlinearSEM()
        sem.add_equation('X', noise_std=0.0)
        sem.add_equation('Y', parents=['X'],
                         func=lambda pv, n: 2 * pv['X'] + n, noise_std=0.0)
        cf = sem.counterfactual(evidence={'X': 3, 'Y': 6}, intervention={'X': 5}, target='Y')
        assert abs(cf - 10.0) < 1e-10

    def test_exogenous_only(self):
        sem = NonlinearSEM()
        sem.add_equation('X', noise_std=1.0)
        np.random.seed(42)
        data = sem.sample(100)
        assert len(data['X']) == 100

    def test_callable_intervention(self):
        sem = NonlinearSEM()
        sem.add_equation('X', noise_std=1.0)
        sem.add_equation('Y', parents=['X'],
                         func=lambda pv, n: pv['X'] + n, noise_std=0.0)
        random.seed(42)
        data = sem.sample(10, interventions={'X': lambda: 5.0})
        np.testing.assert_array_almost_equal(data['X'], np.ones(10) * 5.0)


# ===== SEMIdentification =====

class TestSEMIdentification:
    def _confounded_sem(self):
        """X -> Y with confounder U -> X, U -> Y"""
        sem = LinearSEM()
        sem.add_variable('U', noise_std=1.0)
        sem.add_equation('X', ['U'], {'U': 1.0}, noise_std=0.5)
        sem.add_equation('Y', ['X', 'U'], {'X': 2.0, 'U': 0.5}, noise_std=0.5)
        return sem

    def test_all_directed_paths(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('M', ['X'], {'X': 1.0})
        sem.add_equation('Y', ['X', 'M'], {'X': 1.0, 'M': 1.0})
        ident = SEMIdentification(sem)
        paths = ident.all_directed_paths('X', 'Y')
        assert len(paths) == 2
        assert ['X', 'Y'] in paths
        assert ['X', 'M', 'Y'] in paths

    def test_wright_path_tracing(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('M', ['X'], {'X': 0.5})
        sem.add_equation('Y', ['X', 'M'], {'X': 0.3, 'M': 0.8})
        ident = SEMIdentification(sem)
        total = ident.wright_path_tracing('X', 'Y')
        assert abs(total - 0.7) < 1e-10

    def test_identification_with_backdoor(self):
        sem = self._confounded_sem()
        ident = SEMIdentification(sem)
        identified, adj_set = ident.is_identified('X', 'Y')
        assert identified
        assert adj_set is not None

    def test_identification_simple(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('Y', ['X'], {'X': 1.0})
        ident = SEMIdentification(sem)
        identified, adj_set = ident.is_identified('X', 'Y')
        assert identified

    def test_instrumental_variables(self):
        sem = LinearSEM()
        sem.add_variable('Z', noise_std=1.0)
        sem.add_variable('U', noise_std=1.0)
        sem.add_equation('X', ['Z', 'U'], {'Z': 1.0, 'U': 1.0})
        sem.add_equation('Y', ['X', 'U'], {'X': 2.0, 'U': 0.5})
        ident = SEMIdentification(sem)
        ivs = ident.instrumental_variables('X', 'Y')
        assert 'Z' in ivs

    def test_no_instrument(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('Y', ['X'], {'X': 1.0})
        ident = SEMIdentification(sem)
        ivs = ident.instrumental_variables('X', 'Y')
        # No variable qualifies as an instrument for a simple X->Y
        assert len(ivs) == 0

    def test_ancestors(self):
        sem = LinearSEM()
        sem.add_variable('A')
        sem.add_equation('B', ['A'], {'A': 1.0})
        sem.add_equation('C', ['B'], {'B': 1.0})
        ident = SEMIdentification(sem)
        anc = ident._ancestors('C')
        assert 'A' in anc
        assert 'B' in anc

    def test_descendants(self):
        sem = LinearSEM()
        sem.add_variable('A')
        sem.add_equation('B', ['A'], {'A': 1.0})
        sem.add_equation('C', ['B'], {'B': 1.0})
        ident = SEMIdentification(sem)
        desc = ident._descendants('A')
        assert 'B' in desc
        assert 'C' in desc

    def test_path_max_length(self):
        sem = LinearSEM()
        sem.add_variable('A')
        sem.add_equation('B', ['A'], {'A': 1.0})
        sem.add_equation('C', ['B'], {'B': 1.0})
        sem.add_equation('D', ['C'], {'C': 1.0})
        ident = SEMIdentification(sem)
        paths = ident.all_directed_paths('A', 'D', max_length=2)
        # Only path is A->B->C->D (length 3), so max_length=2 should exclude it
        assert len(paths) == 0


# ===== SEMIntervention =====

class TestSEMIntervention:
    def test_do_linear(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('M', ['X'], {'X': 0.5}, noise_std=0.5)
        sem.add_equation('Y', ['M'], {'M': 2.0}, noise_std=0.5)

        new_sem = SEMIntervention.do(sem, {'X': 3.0})
        assert new_sem.equations['X'].intercept == 3.0
        assert new_sem.equations['X'].noise_std == 0.0
        assert len(new_sem.equations['X'].parents) == 0

    def test_do_preserves_other_equations(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=1.0)

        new_sem = SEMIntervention.do(sem, {'X': 5.0})
        assert new_sem.equations['Y'].coefficients['X'] == 2.0
        assert new_sem.equations['Y'].noise_std == 1.0

    def test_do_nonlinear(self):
        sem = NonlinearSEM()
        sem.add_equation('X', noise_std=1.0)
        sem.add_equation('Y', parents=['X'],
                         func=lambda pv, n: pv['X']**2 + n, noise_std=0.5)

        new_sem = SEMIntervention.do(sem, {'X': 4.0})
        data = new_sem.sample(10)
        np.testing.assert_array_almost_equal(data['X'], np.ones(10) * 4.0)

    def test_do_multiple_interventions(self):
        sem = LinearSEM()
        sem.add_variable('X')
        sem.add_variable('Z')
        sem.add_equation('Y', ['X', 'Z'], {'X': 1.0, 'Z': 2.0}, noise_std=0.0)

        new_sem = SEMIntervention.do(sem, {'X': 1.0, 'Z': 2.0})
        data = new_sem.sample(1)
        assert abs(data['Y'][0] - 5.0) < 1e-10  # 1*1 + 2*2

    def test_conditional_intervention(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.5)
        sem.add_equation('Z', ['Y'], {'Y': 1.0}, noise_std=0.5)

        np.random.seed(42)
        result = SEMIntervention.conditional_intervention(
            sem, interventions={'X': 3.0}, conditions={}, n_samples=5000
        )
        # Y should be ~6.0 under do(X=3), Z should be ~6.0
        assert abs(result['Y'] - 6.0) < 0.5
        assert abs(result['Z'] - 6.0) < 0.5


# ===== SEMCounterfactual =====

class TestSEMCounterfactual:
    def test_linear_counterfactual_simple(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.0)

        # Observed X=3, Y=6. What if X=5?
        cf = SEMCounterfactual.query(sem, {'X': 3, 'Y': 6}, {'X': 5}, 'Y')
        assert abs(cf - 10.0) < 1e-10

    def test_counterfactual_with_noise_recovery(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, intercept=1.0, noise_std=1.0)

        # Y = 1 + 2*X + noise. If X=3 and Y=8, noise = 8 - 7 = 1
        # Counterfactual: X=5 -> Y = 1 + 2*5 + 1 = 12
        cf = SEMCounterfactual.query(sem, {'X': 3, 'Y': 8}, {'X': 5}, 'Y')
        assert abs(cf - 12.0) < 1e-10

    def test_counterfactual_chain(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('M', ['X'], {'X': 2.0}, noise_std=0.0)
        sem.add_equation('Y', ['M'], {'M': 3.0}, noise_std=0.0)

        # X=1, M=2, Y=6. What if X=2?
        cf = SEMCounterfactual.query(sem, {'X': 1, 'M': 2, 'Y': 6}, {'X': 2}, 'Y')
        assert abs(cf - 12.0) < 1e-10

    def test_counterfactual_partial_evidence(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 1.0}, noise_std=0.0)

        # Only X observed
        cf = SEMCounterfactual.query(sem, {'X': 5}, {'X': 10}, 'Y')
        assert abs(cf - 10.0) < 1e-10

    def test_ett(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 3.0}, noise_std=0.0)

        ett = SEMCounterfactual.effect_of_treatment_on_treated(
            sem, evidence={'Y': 9.0}, cause='X', effect='Y',
            treatment_value=3.0, control_value=0.0
        )
        # Y(3)=9, Y(0)=0+noise. noise=9-9=0, so Y(0)=0. ETT=9-0=9
        assert abs(ett - 9.0) < 1e-10

    def test_probability_of_necessity(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.0)

        pn = SEMCounterfactual.probability_of_necessity(
            sem, {'X': 1, 'Y': 2}, 'X', 'Y'
        )
        # Y observed=2, Y(X=0) counterfactual=0, difference=2
        assert abs(pn - 2.0) < 1e-10

    def test_probability_of_sufficiency(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.0)

        ps = SEMCounterfactual.probability_of_sufficiency(
            sem, {'X': 0, 'Y': 0}, 'X', 'Y'
        )
        # Y observed=0, Y(X=1) counterfactual=2, difference=2
        assert abs(ps - 2.0) < 1e-10


# ===== SEMEstimation =====

class TestSEMEstimation:
    def test_ols_basic(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 1000)
        result = SEMEstimation.ols({'X': x, 'Y': y}, 'Y', ['X'])
        assert abs(result['coefficients']['X'] - 2.0) < 0.1
        assert abs(result['intercept'] - 1.0) < 0.1
        assert result['r_squared'] > 0.95

    def test_ols_multiple_predictors(self):
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 1000)
        x2 = np.random.normal(0, 1, 1000)
        y = 1.5 * x1 - 0.5 * x2 + 0.5 + np.random.normal(0, 0.1, 1000)
        result = SEMEstimation.ols({'X1': x1, 'X2': x2, 'Y': y}, 'Y', ['X1', 'X2'])
        assert abs(result['coefficients']['X1'] - 1.5) < 0.1
        assert abs(result['coefficients']['X2'] - (-0.5)) < 0.1

    def test_ols_no_predictors(self):
        y = np.array([1.0, 2.0, 3.0])
        result = SEMEstimation.ols({'Y': y}, 'Y', [])
        assert abs(result['intercept'] - 2.0) < 1e-10
        assert result['r_squared'] == 0.0

    def test_two_stage_ls(self):
        np.random.seed(42)
        n = 2000
        z = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, n)
        x = 0.8 * z + 0.5 * u + np.random.normal(0, 0.3, n)
        y = 2.0 * x + 0.5 * u + np.random.normal(0, 0.3, n)

        result = SEMEstimation.two_stage_ls(
            {'X': x, 'Y': y, 'Z': z, 'U': u},
            target='Y', endogenous=['X'], instruments=['Z']
        )
        assert abs(result['coefficients']['X'] - 2.0) < 0.3

    def test_path_coefficients(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('M', ['X'], {'X': 0.5}, noise_std=0.3)
        sem.add_equation('Y', ['M'], {'M': 2.0}, noise_std=0.3)

        np.random.seed(42)
        data = sem.sample(5000)
        estimated = SEMEstimation.path_coefficients(data, sem)
        assert abs(estimated[('X', 'M')] - 0.5) < 0.1
        assert abs(estimated[('M', 'Y')] - 2.0) < 0.1

    def test_ols_r_squared(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 3.0 * x  # Perfect linear relationship
        result = SEMEstimation.ols({'X': x, 'Y': y}, 'Y', ['X'])
        assert result['r_squared'] > 0.99


# ===== SEMFitMetrics =====

class TestSEMFitMetrics:
    def _sem_and_data(self, seed=42, n=5000):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.5)
        np.random.seed(seed)
        data = sem.sample(n)
        return sem, data

    def test_equation_r_squared(self):
        sem, data = self._sem_and_data()
        r2 = SEMFitMetrics.equation_r_squared(data, sem)
        assert r2['X'] == 0.0  # Exogenous
        assert r2['Y'] > 0.7  # Should be high since noise is small

    def test_residuals(self):
        sem, data = self._sem_and_data()
        resid = SEMFitMetrics.residuals(data, sem)
        assert 'X' in resid
        assert 'Y' in resid
        assert len(resid['Y']) == 5000
        # Residuals should be ~noise
        assert abs(np.mean(resid['Y'])) < 0.1

    def test_implied_vs_observed(self):
        sem, data = self._sem_and_data()
        result = SEMFitMetrics.implied_vs_observed_covariance(data, sem)
        assert result['rmse'] < 0.5  # Should be reasonably close
        assert result['implied'].shape == (2, 2)
        assert result['observed'].shape == (2, 2)

    def test_aic_bic(self):
        sem, data = self._sem_and_data()
        result = SEMFitMetrics.aic_bic(data, sem)
        assert 'aic' in result
        assert 'bic' in result
        assert result['n_params'] > 0
        assert result['bic'] >= result['aic']  # BIC penalizes more with large n


# ===== SEMSimulator =====

class TestSEMSimulator:
    def test_simulate_basic(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 1.0}, noise_std=0.5)
        data = SEMSimulator.simulate(sem, n_samples=100, seed=42)
        assert len(data['X']) == 100
        assert len(data['Y']) == 100

    def test_simulate_intervention(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.5)
        data = SEMSimulator.simulate(sem, n_samples=100, interventions={'X': 3.0}, seed=42)
        assert np.all(data['X'] == 3.0)

    def test_interventional_distribution(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.1)
        result = SEMSimulator.interventional_distribution(
            sem, 'X', 'Y', [0.0, 1.0, 2.0], n_samples=5000, seed=42
        )
        assert len(result['values']) == 3
        assert len(result['expectations']) == 3
        # E[Y|do(X=0)] ~= 0, E[Y|do(X=1)] ~= 2, E[Y|do(X=2)] ~= 4
        assert abs(result['expectations'][0]) < 0.2
        assert abs(result['expectations'][1] - 2.0) < 0.2
        assert abs(result['expectations'][2] - 4.0) < 0.2

    def test_sensitivity_analysis(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=1.0)
        result = SEMSimulator.sensitivity_analysis(
            sem, 'X', 'Y', [0.5, 1.0, 2.0], n_samples=5000, seed=42
        )
        assert len(result['multipliers']) == 3
        # Effect should stay ~2.0 regardless of noise
        for e in result['effects']:
            assert abs(e - 2.0) < 0.5
        # Variance should increase with noise multiplier
        assert result['variances'][2] > result['variances'][0]

    def test_seed_reproducibility(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 1.0}, noise_std=1.0)
        d1 = SEMSimulator.simulate(sem, 10, seed=123)
        d2 = SEMSimulator.simulate(sem, 10, seed=123)
        np.testing.assert_array_equal(d1['X'], d2['X'])
        np.testing.assert_array_equal(d1['Y'], d2['Y'])


# ===== SEMAnalyzer =====

class TestSEMAnalyzer:
    def _mediation_sem(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('M', ['X'], {'X': 0.5}, noise_std=0.5)
        sem.add_equation('Y', ['X', 'M'], {'X': 0.3, 'M': 0.8}, noise_std=0.5)
        return sem

    def test_decompose_effect(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        result = analyzer.decompose_effect('X', 'Y')
        assert abs(result['total'] - 0.7) < 1e-10
        assert abs(result['direct'] - 0.3) < 1e-10
        assert abs(result['indirect'] - 0.4) < 1e-10
        assert len(result['paths']) == 2

    def test_mediation_analysis(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        result = analyzer.mediation_analysis('X', 'M', 'Y')
        assert abs(result['a_path'] - 0.5) < 1e-10
        assert abs(result['b_path'] - 0.8) < 1e-10
        assert abs(result['indirect_via_mediator'] - 0.4) < 1e-10
        assert abs(result['proportion_mediated'] - 0.4/0.7) < 1e-5

    def test_full_report(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        report = analyzer.full_report(n_samples=5000, seed=42)
        assert 'variables' in report
        assert 'r_squared' in report
        assert 'aic_bic' in report
        assert 'estimated_coefficients' in report
        assert 'true_coefficients' in report
        assert report['covariance_rmse'] < 1.0

    def test_causal_query_ate(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        ate = analyzer.causal_query('ate', cause='X', effect='Y')
        assert abs(ate - 0.7) < 1e-10

    def test_causal_query_intervention(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        np.random.seed(42)
        result = analyzer.causal_query('intervention',
                                       interventions={'X': 2.0}, target='Y',
                                       n_samples=10000)
        # E[Y|do(X=2)] = 0.3*2 + 0.8*(0.5*2) = 0.6 + 0.8 = 1.4
        assert abs(result - 1.4) < 0.2

    def test_causal_query_counterfactual(self):
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.0)
        analyzer = SEMAnalyzer(sem)
        cf = analyzer.causal_query('counterfactual',
                                   evidence={'X': 1, 'Y': 2},
                                   intervention={'X': 3},
                                   target='Y')
        assert abs(cf - 6.0) < 1e-10

    def test_causal_query_mediation(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        result = analyzer.causal_query('mediation', cause='X', mediator='M', effect='Y')
        assert abs(result['indirect_via_mediator'] - 0.4) < 1e-10

    def test_causal_query_invalid(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        with pytest.raises(ValueError, match="Unknown query type"):
            analyzer.causal_query('invalid')

    def test_decompose_nonlinear_raises(self):
        sem = NonlinearSEM()
        sem.add_equation('X')
        sem.add_equation('Y', parents=['X'], func=lambda pv, n: pv['X'])
        analyzer = SEMAnalyzer(sem)
        with pytest.raises(ValueError):
            analyzer.decompose_effect('X', 'Y')

    def test_full_report_with_data(self):
        sem = self._mediation_sem()
        np.random.seed(42)
        data = sem.sample(2000)
        analyzer = SEMAnalyzer(sem)
        report = analyzer.full_report(data=data)
        # Estimated coefficients should be close to true
        for edge, true_val in report['true_coefficients'].items():
            est_val = report['estimated_coefficients'].get(edge, 0)
            assert abs(est_val - true_val) < 0.2

    def test_identification_attribute(self):
        sem = self._mediation_sem()
        analyzer = SEMAnalyzer(sem)
        assert analyzer.identification is not None

    def test_nonlinear_no_identification(self):
        sem = NonlinearSEM()
        sem.add_equation('X')
        analyzer = SEMAnalyzer(sem)
        assert analyzer.identification is None


# ===== Integration tests =====

class TestIntegration:
    def test_full_pipeline_linear(self):
        """End-to-end: define SEM, simulate, estimate, compare."""
        sem = LinearSEM()
        sem.add_variable('Z', noise_std=1.0)
        sem.add_equation('X', ['Z'], {'Z': 0.8}, noise_std=0.5)
        sem.add_equation('Y', ['X'], {'X': 1.5}, noise_std=0.5)

        np.random.seed(42)
        data = sem.sample(10000)
        estimated = SEMEstimation.path_coefficients(data, sem)
        assert abs(estimated[('Z', 'X')] - 0.8) < 0.1
        assert abs(estimated[('X', 'Y')] - 1.5) < 0.1

    def test_intervention_vs_observation(self):
        """Verify do(X=x) differs from P(Y|X=x) when confounded."""
        sem = LinearSEM()
        sem.add_variable('U', noise_std=1.0)
        sem.add_equation('X', ['U'], {'U': 1.0}, noise_std=0.5)
        sem.add_equation('Y', ['X', 'U'], {'X': 2.0, 'U': 3.0}, noise_std=0.5)

        np.random.seed(42)
        data = sem.sample(50000)

        # Observational: P(Y|X~=2) -- biased by U
        mask = np.abs(data['X'] - 2.0) < 0.2
        obs_mean = np.mean(data['Y'][mask])

        # Interventional: E[Y|do(X=2)]
        data_do = sem.sample(50000, interventions={'X': 2.0})
        int_mean = np.mean(data_do['Y'])

        # Should differ because confounding
        # do(X=2): E[Y] = 2*2 + 3*E[U] = 4 + 0 = 4
        assert abs(int_mean - 4.0) < 0.2
        # Obs: when X~=2, U is likely ~2 (since X = U + noise), so Y ~= 2*2 + 3*2 = 10
        assert obs_mean > 5.0  # Much higher due to confounding

    def test_counterfactual_with_mediation(self):
        """Test counterfactual through a mediator."""
        sem = LinearSEM()
        sem.add_variable('X', noise_std=0.0)
        sem.add_equation('M', ['X'], {'X': 2.0}, noise_std=0.0)
        sem.add_equation('Y', ['X', 'M'], {'X': 1.0, 'M': 3.0}, noise_std=0.0)

        # Observed: X=1, M=2, Y=7 (1 + 6)
        cf = SEMCounterfactual.query(
            sem, {'X': 1, 'M': 2, 'Y': 7}, {'X': 2}, 'Y'
        )
        # CF: X=2, M=4, Y = 2 + 12 = 14
        assert abs(cf - 14.0) < 1e-10

    def test_nonlinear_interaction(self):
        """Test nonlinear SEM with interaction effects."""
        sem = NonlinearSEM()
        sem.add_equation('X1', noise_std=0.0)
        sem.add_equation('X2', noise_std=0.0)
        sem.add_equation('Y', parents=['X1', 'X2'],
                         func=lambda pv, n: pv['X1'] * pv['X2'] + pv['X1']**2 + n,
                         noise_std=0.0)

        data = sem.sample(1, noise_values={'X1': [3.0], 'X2': [2.0], 'Y': [0.0]})
        # Y = 3*2 + 9 = 15
        assert abs(data['Y'][0] - 15.0) < 1e-10

    def test_estimation_recovery(self):
        """Verify 2SLS recovers true coefficient under confounding."""
        np.random.seed(42)
        n = 10000
        z = np.random.normal(0, 1, n)
        u = np.random.normal(0, 1, n)
        x = 0.7 * z + 0.6 * u + np.random.normal(0, 0.2, n)
        y = 1.5 * x + 0.8 * u + np.random.normal(0, 0.2, n)

        # OLS is biased
        ols = SEMEstimation.ols({'X': x, 'Y': y}, 'Y', ['X'])
        ols_coeff = ols['coefficients']['X']

        # 2SLS should be closer to truth
        tsls = SEMEstimation.two_stage_ls(
            {'X': x, 'Y': y, 'Z': z},
            target='Y', endogenous=['X'], instruments=['Z']
        )
        tsls_coeff = tsls['coefficients']['X']

        assert abs(tsls_coeff - 1.5) < abs(ols_coeff - 1.5)

    def test_sensitivity_effect_stable(self):
        """Total effect should be stable regardless of noise levels."""
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 3.0}, noise_std=1.0)

        result = SEMSimulator.sensitivity_analysis(
            sem, 'X', 'Y', [0.1, 1.0, 5.0], n_samples=10000, seed=42
        )
        for e in result['effects']:
            assert abs(e - 3.0) < 0.5

    def test_large_sem(self):
        """Test with a larger SEM (5 variables)."""
        sem = LinearSEM()
        sem.add_variable('X1', noise_std=1.0)
        sem.add_variable('X2', noise_std=1.0)
        sem.add_equation('M1', ['X1', 'X2'], {'X1': 0.5, 'X2': 0.3}, noise_std=0.5)
        sem.add_equation('M2', ['X1'], {'X1': 0.8}, noise_std=0.5)
        sem.add_equation('Y', ['M1', 'M2'], {'M1': 1.0, 'M2': 0.5}, noise_std=0.5)

        total_x1_y = sem.total_effect('X1', 'Y')
        # Direct paths: X1->M1->Y (0.5*1.0=0.5) and X1->M2->Y (0.8*0.5=0.4) = 0.9
        assert abs(total_x1_y - 0.9) < 1e-10

        total_x2_y = sem.total_effect('X2', 'Y')
        # X2->M1->Y = 0.3*1.0 = 0.3
        assert abs(total_x2_y - 0.3) < 1e-10

    def test_wright_matches_total_effect(self):
        """Wright's path tracing should match total_effect."""
        sem = LinearSEM()
        sem.add_variable('A', noise_std=1.0)
        sem.add_equation('B', ['A'], {'A': 0.5}, noise_std=0.5)
        sem.add_equation('C', ['A', 'B'], {'A': 0.3, 'B': 0.7}, noise_std=0.5)
        sem.add_equation('D', ['B', 'C'], {'B': 0.2, 'C': 0.4}, noise_std=0.5)

        ident = SEMIdentification(sem)
        wright = ident.wright_path_tracing('A', 'D')
        total = sem.total_effect('A', 'D')
        assert abs(wright - total) < 1e-10

    def test_implied_covariance_accuracy(self):
        """Implied covariance should match sample covariance with large N."""
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('Y', ['X'], {'X': 2.0}, noise_std=0.5)

        np.random.seed(42)
        data = sem.sample(100000)

        result = SEMFitMetrics.implied_vs_observed_covariance(data, sem)
        assert result['rmse'] < 0.1

    def test_do_then_sample_consistency(self):
        """SEMIntervention.do() + sample should match direct sample with interventions."""
        sem = LinearSEM()
        sem.add_variable('X', noise_std=1.0)
        sem.add_equation('M', ['X'], {'X': 0.5}, noise_std=0.5)
        sem.add_equation('Y', ['M'], {'M': 2.0}, noise_std=0.5)

        np.random.seed(42)
        d1 = sem.sample(10000, interventions={'X': 3.0})

        new_sem = SEMIntervention.do(sem, {'X': 3.0})
        np.random.seed(42)
        d2 = new_sem.sample(10000)

        assert abs(np.mean(d1['Y']) - np.mean(d2['Y'])) < 0.2
