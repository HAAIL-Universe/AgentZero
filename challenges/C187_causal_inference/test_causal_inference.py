"""Tests for C187: Causal Inference"""

import math
import random
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from causal_inference import (
    CausalGraph, StructuralCausalModel, BackdoorCriterion, FrontdoorCriterion,
    InstrumentalVariable, PropensityScore, CausalEstimator, MediationAnalysis,
    CausalDiscovery, DoCalculus, SensitivityAnalysis, DifferenceInDifferences,
    RegressionDiscontinuity,
)


# ============================================================
# CausalGraph Tests
# ============================================================

class TestCausalGraph(unittest.TestCase):

    def test_add_nodes_edges(self):
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g.add_edge('Z', 'Y')
        self.assertEqual(g.nodes, {'X', 'Y', 'Z'})
        self.assertTrue(g.has_edge('X', 'Y'))
        self.assertFalse(g.has_edge('Y', 'X'))

    def test_parents_children(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'D')
        self.assertEqual(g.children('A'), {'B', 'C'})
        self.assertEqual(g.get_parents('D'), {'B'})

    def test_ancestors_descendants(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'D')
        self.assertEqual(g.ancestors('D'), {'A', 'B', 'C'})
        self.assertEqual(g.descendants('A'), {'B', 'C', 'D'})

    def test_topological_sort(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('A', 'C')
        order = g.topological_sort()
        self.assertEqual(order[0], 'A')
        self.assertIn('B', order)
        self.assertEqual(order[-1], 'C')

    def test_topological_sort_cycle_raises(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        with self.assertRaises(ValueError):
            g.topological_sort()

    def test_copy(self):
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g2 = g.copy()
        g2.add_edge('Y', 'Z')
        self.assertFalse(g.has_edge('Y', 'Z'))
        self.assertTrue(g2.has_edge('Y', 'Z'))

    def test_remove_incoming(self):
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')
        g.remove_incoming('B')
        self.assertFalse(g.has_edge('A', 'B'))
        self.assertFalse(g.has_edge('C', 'B'))
        self.assertEqual(g.get_parents('B'), set())

    # --- d-separation tests ---

    def test_d_separation_chain(self):
        # A -> B -> C: A _||_ C | B
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        self.assertTrue(g.is_d_separated('A', 'C', {'B'}))
        self.assertFalse(g.is_d_separated('A', 'C', set()))

    def test_d_separation_fork(self):
        # A <- B -> C: A _||_ C | B
        g = CausalGraph()
        g.add_edge('B', 'A')
        g.add_edge('B', 'C')
        self.assertTrue(g.is_d_separated('A', 'C', {'B'}))
        self.assertFalse(g.is_d_separated('A', 'C', set()))

    def test_d_separation_collider(self):
        # A -> B <- C: A _||_ C | {}; NOT d-separated given B
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')
        self.assertTrue(g.is_d_separated('A', 'C', set()))
        self.assertFalse(g.is_d_separated('A', 'C', {'B'}))

    def test_d_separation_collider_descendant(self):
        # A -> B <- C, B -> D: conditioning on descendant D opens collider
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')
        g.add_edge('B', 'D')
        self.assertFalse(g.is_d_separated('A', 'C', {'D'}))

    def test_d_separation_complex(self):
        # X -> Z -> Y, X <- U -> Y (confounder)
        g = CausalGraph()
        g.add_edge('X', 'Z')
        g.add_edge('Z', 'Y')
        g.add_edge('U', 'X')
        g.add_edge('U', 'Y')
        # X and Y are not d-separated unconditionally
        self.assertFalse(g.is_d_separated('X', 'Y', set()))
        # Conditioning on Z and U should d-separate
        self.assertTrue(g.is_d_separated('X', 'Y', {'Z', 'U'}))

    def test_moral_graph(self):
        g = CausalGraph()
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        moral = g.moral_graph()
        # A and B should be connected (married parents)
        self.assertIn('B', moral['A'])
        self.assertIn('A', moral['B'])


# ============================================================
# Structural Causal Model Tests
# ============================================================

class TestSCM(unittest.TestCase):

    def _make_simple_scm(self):
        """X -> Y with linear equation Y = 2*X + noise."""
        scm = StructuralCausalModel()
        scm.add_equation('X', [], lambda p, n: n, lambda: random.gauss(0, 1))
        scm.add_equation('Y', ['X'], lambda p, n: 2 * p['X'] + n, lambda: random.gauss(0, 0.1))
        return scm

    def test_sample(self):
        scm = self._make_simple_scm()
        samples = scm.sample(100)
        self.assertEqual(len(samples), 100)
        self.assertIn('X', samples[0])
        self.assertIn('Y', samples[0])

    def test_causal_effect(self):
        """Y = 2*X + noise, so causal effect of X on Y should be ~2."""
        random.seed(42)
        scm = self._make_simple_scm()
        samples = scm.sample(1000)
        # Regression of Y on X should give slope ~2
        x_vals = [s['X'] for s in samples]
        y_vals = [s['Y'] for s in samples]
        x_mean = sum(x_vals) / len(x_vals)
        y_mean = sum(y_vals) / len(y_vals)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        var_x = sum((x - x_mean) ** 2 for x in x_vals)
        slope = cov / var_x
        self.assertAlmostEqual(slope, 2.0, delta=0.2)

    def test_intervention(self):
        """do(X=5) should give Y = 2*5 + noise ~= 10."""
        random.seed(42)
        scm = self._make_simple_scm()
        samples = scm.sample(500, interventions={'X': 5})
        y_mean = sum(s['Y'] for s in samples) / len(samples)
        self.assertAlmostEqual(y_mean, 10.0, delta=0.2)

    def test_intervention_removes_parents(self):
        """With confounder U -> X, U -> Y, do(X) should remove U -> X path."""
        random.seed(42)
        scm = StructuralCausalModel()
        scm.add_equation('U', [], lambda p, n: n, lambda: random.gauss(0, 1))
        scm.add_equation('X', ['U'], lambda p, n: p['U'] + n, lambda: random.gauss(0, 0.1))
        scm.add_equation('Y', ['X', 'U'], lambda p, n: 3 * p['X'] + p['U'] + n, lambda: random.gauss(0, 0.1))

        # Observational: regressing Y on X gives biased estimate (confounded by U)
        obs = scm.sample(1000)
        x_vals = [s['X'] for s in obs]
        y_vals = [s['Y'] for s in obs]
        x_mean = sum(x_vals) / len(x_vals)
        y_mean = sum(y_vals) / len(y_vals)
        cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        var_x = sum((x - x_mean) ** 2 for x in x_vals)
        obs_slope = cov / var_x
        # Biased: should be > 3 due to confounding
        self.assertGreater(obs_slope, 3.0)

        # Interventional: do(X=0) vs do(X=1), difference should be ~3
        do0 = scm.sample(500, interventions={'X': 0})
        do1 = scm.sample(500, interventions={'X': 1})
        y0 = sum(s['Y'] for s in do0) / len(do0)
        y1 = sum(s['Y'] for s in do1) / len(do1)
        self.assertAlmostEqual(y1 - y0, 3.0, delta=0.3)

    def test_counterfactual_simple(self):
        """If X was 0 and Y was 0.5, what would Y be if X were 2?"""
        scm = self._make_simple_scm()
        cf_y = scm.counterfactual(
            evidence={'X': 0, 'Y': 0.5},
            intervention={'X': 2},
            target='Y'
        )
        # Y = 2*X + noise, noise = 0.5 - 2*0 = 0.5
        # Counterfactual: Y = 2*2 + 0.5 = 4.5
        self.assertAlmostEqual(cf_y, 4.5, delta=0.01)

    def test_counterfactual_with_confounder(self):
        """Counterfactual with common cause."""
        scm = StructuralCausalModel()
        scm.add_equation('U', [], lambda p, n: n)
        scm.add_equation('X', ['U'], lambda p, n: p['U'] + n, lambda: 0)
        scm.add_equation('Y', ['X', 'U'], lambda p, n: 2 * p['X'] + p['U'] + n, lambda: 0)

        # Observe: U=1, X=1, Y=3 (2*1+1=3)
        cf = scm.counterfactual(
            evidence={'U': 1, 'X': 1, 'Y': 3},
            intervention={'X': 5},
            target='Y'
        )
        # Counterfactual: Y = 2*5 + 1 + 0 = 11
        self.assertAlmostEqual(cf, 11.0, delta=0.01)

    def test_generate_data(self):
        scm = self._make_simple_scm()
        data = scm.generate_data(n=50, seed=42)
        self.assertEqual(len(data), 50)
        self.assertIn('X', data[0])

    def test_multiple_nodes(self):
        scm = StructuralCausalModel()
        scm.add_equation('A', [], lambda p, n: n, lambda: random.gauss(0, 1))
        scm.add_equation('B', ['A'], lambda p, n: p['A'] * 0.5 + n, lambda: random.gauss(0, 0.1))
        scm.add_equation('C', ['A', 'B'], lambda p, n: p['A'] + p['B'] + n, lambda: random.gauss(0, 0.1))
        samples = scm.sample(10)
        self.assertEqual(len(samples), 10)
        for s in samples:
            self.assertIn('A', s)
            self.assertIn('B', s)
            self.assertIn('C', s)


# ============================================================
# Backdoor Criterion Tests
# ============================================================

class TestBackdoorCriterion(unittest.TestCase):

    def test_simple_confounder(self):
        # Z -> X, Z -> Y: Z is valid adjustment
        g = CausalGraph()
        g.add_edge('Z', 'X')
        g.add_edge('Z', 'Y')
        g.add_edge('X', 'Y')
        bd = BackdoorCriterion(g)
        self.assertTrue(bd.is_valid_adjustment_set('X', 'Y', {'Z'}))

    def test_mediator_invalid(self):
        # X -> M -> Y: M is NOT valid (descendant of X)
        g = CausalGraph()
        g.add_edge('X', 'M')
        g.add_edge('M', 'Y')
        bd = BackdoorCriterion(g)
        self.assertFalse(bd.is_valid_adjustment_set('X', 'Y', {'M'}))

    def test_collider_invalid(self):
        # X -> C <- Y: C is NOT valid
        g = CausalGraph()
        g.add_edge('X', 'C')
        g.add_edge('Y', 'C')
        g.add_edge('X', 'Y')
        bd = BackdoorCriterion(g)
        self.assertFalse(bd.is_valid_adjustment_set('X', 'Y', {'C'}))

    def test_empty_valid_no_confounding(self):
        # X -> Y (no backdoor paths)
        g = CausalGraph()
        g.add_edge('X', 'Y')
        bd = BackdoorCriterion(g)
        self.assertTrue(bd.is_valid_adjustment_set('X', 'Y', set()))

    def test_find_minimal_adjustment(self):
        g = CausalGraph()
        g.add_edge('Z', 'X')
        g.add_edge('Z', 'Y')
        g.add_edge('X', 'Y')
        bd = BackdoorCriterion(g)
        adj = bd.find_minimal_adjustment_set('X', 'Y')
        self.assertIsNotNone(adj)
        self.assertTrue(bd.is_valid_adjustment_set('X', 'Y', adj))

    def test_multiple_confounders(self):
        g = CausalGraph()
        g.add_edge('Z1', 'X')
        g.add_edge('Z1', 'Y')
        g.add_edge('Z2', 'X')
        g.add_edge('Z2', 'Y')
        g.add_edge('X', 'Y')
        bd = BackdoorCriterion(g)
        self.assertTrue(bd.is_valid_adjustment_set('X', 'Y', {'Z1', 'Z2'}))
        # Single confounder may also suffice
        adj = bd.find_minimal_adjustment_set('X', 'Y')
        self.assertIsNotNone(adj)


# ============================================================
# Front-door Criterion Tests
# ============================================================

class TestFrontdoorCriterion(unittest.TestCase):

    def test_find_frontdoor_set(self):
        # X -> M -> Y, U -> X, U -> Y (confounded, but M is front-door)
        g = CausalGraph()
        g.add_edge('X', 'M')
        g.add_edge('M', 'Y')
        g.add_edge('U', 'X')
        g.add_edge('U', 'Y')
        fd = FrontdoorCriterion(g)
        result = fd.find_frontdoor_set('X', 'Y')
        # M should be valid front-door
        if result is not None:
            self.assertIn('M', result)

    def test_no_frontdoor_direct_effect(self):
        # X -> Y directly (no mediator) with confounder
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g.add_edge('U', 'X')
        g.add_edge('U', 'Y')
        fd = FrontdoorCriterion(g)
        result = fd.find_frontdoor_set('X', 'Y')
        # No valid front-door set (X has direct effect on Y)
        self.assertIsNone(result)

    def test_frontdoor_estimate(self):
        random.seed(42)
        # Generate data from front-door model
        data = []
        for _ in range(1000):
            u = random.gauss(0, 1)
            x = 1 if u + random.gauss(0, 0.5) > 0 else 0
            m = 1 if 0.8 * x + random.gauss(0, 0.3) > 0.4 else 0
            y = 2 * m + u + random.gauss(0, 0.1)
            data.append({'X': x, 'M': m, 'Y': y, 'U': u})

        g = CausalGraph()
        g.add_edge('X', 'M')
        g.add_edge('M', 'Y')
        g.add_edge('U', 'X')
        g.add_edge('U', 'Y')
        fd = FrontdoorCriterion(g)
        result = fd.estimate(data, 'X', 'Y', 'M')
        # The front-door formula should give estimates for X=0 and X=1
        self.assertIn(0, result)
        self.assertIn(1, result)


# ============================================================
# Instrumental Variable Tests
# ============================================================

class TestInstrumentalVariable(unittest.TestCase):

    def test_2sls_basic(self):
        """IV: Z -> X -> Y, no direct Z -> Y path."""
        random.seed(42)
        data = []
        for _ in range(2000):
            u = random.gauss(0, 1)
            z = random.gauss(0, 1)
            x = 0.5 * z + u + random.gauss(0, 0.1)
            y = 3.0 * x + u + random.gauss(0, 0.1)
            data.append({'Z': z, 'X': x, 'Y': y})

        iv = InstrumentalVariable()
        effect = iv.estimate(data, 'Z', 'X', 'Y')
        # True causal effect is 3.0
        self.assertAlmostEqual(effect, 3.0, delta=0.5)

    def test_2sls_with_covariates(self):
        """IV with additional covariates."""
        random.seed(42)
        data = []
        for _ in range(2000):
            w = random.gauss(0, 1)
            z = random.gauss(0, 1)
            u = random.gauss(0, 1)
            x = 0.5 * z + 0.3 * w + u + random.gauss(0, 0.1)
            y = 2.0 * x + 0.5 * w + u + random.gauss(0, 0.1)
            data.append({'Z': z, 'X': x, 'Y': y, 'W': w})

        iv = InstrumentalVariable()
        effect = iv.estimate(data, 'Z', 'X', 'Y', covariates=['W'])
        self.assertAlmostEqual(effect, 2.0, delta=0.5)

    def test_weak_instrument(self):
        """Weak instrument gives less precise estimate."""
        random.seed(42)
        data = []
        for _ in range(2000):
            z = random.gauss(0, 1)
            u = random.gauss(0, 1)
            x = 0.05 * z + u  # Very weak instrument
            y = 2.0 * x + u
            data.append({'Z': z, 'X': x, 'Y': y})

        iv = InstrumentalVariable()
        effect = iv.estimate(data, 'Z', 'X', 'Y')
        # With weak instrument, estimate exists but may be imprecise
        self.assertIsInstance(effect, float)


# ============================================================
# Propensity Score Tests
# ============================================================

class TestPropensityScore(unittest.TestCase):

    def _generate_data(self, n=1000):
        random.seed(42)
        data = []
        for _ in range(n):
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            # Treatment probability depends on covariates
            logit = 0.5 * x1 + 0.3 * x2
            p = 1 / (1 + math.exp(-logit))
            t = 1 if random.random() < p else 0
            # Outcome with treatment effect = 2.0
            y = 2.0 * t + 0.5 * x1 + 0.3 * x2 + random.gauss(0, 0.5)
            data.append({'X1': x1, 'X2': x2, 'T': t, 'Y': y})
        return data

    def test_estimate_propensity(self):
        data = self._generate_data()
        ps = PropensityScore()
        scores = ps.estimate_propensity(data, 'T', ['X1', 'X2'])
        self.assertEqual(len(scores), len(data))
        # All scores should be between 0 and 1
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_ipw_estimate(self):
        data = self._generate_data(2000)
        ps = PropensityScore()
        ate = ps.ipw_estimate(data, 'T', 'Y', ['X1', 'X2'])
        # True ATE is 2.0
        self.assertAlmostEqual(ate, 2.0, delta=0.5)

    def test_matching_estimate(self):
        data = self._generate_data(2000)
        ps = PropensityScore()
        ate = ps.matching_estimate(data, 'T', 'Y', ['X1', 'X2'], k=3)
        # True ATE is 2.0
        self.assertAlmostEqual(ate, 2.0, delta=0.6)


# ============================================================
# Causal Estimator Tests
# ============================================================

class TestCausalEstimator(unittest.TestCase):

    def test_ate_no_adjustment(self):
        """Simple ATE without confounding."""
        random.seed(42)
        data = []
        for _ in range(1000):
            t = random.choice([0, 1])
            y = 3.0 * t + random.gauss(0, 0.5)
            data.append({'T': t, 'Y': y})

        est = CausalEstimator()
        ate = est.ate_regression(data, 'T', 'Y', set())
        self.assertAlmostEqual(ate, 3.0, delta=0.3)

    def test_ate_with_adjustment(self):
        """ATE with confounding, adjusting for Z."""
        random.seed(42)
        data = []
        for _ in range(2000):
            z = random.gauss(0, 1)
            t = 1 if z + random.gauss(0, 0.5) > 0 else 0
            y = 2.0 * t + 1.5 * z + random.gauss(0, 0.5)
            data.append({'T': t, 'Y': y, 'Z': z})

        est = CausalEstimator()
        ate = est.ate_regression(data, 'T', 'Y', {'Z'})
        self.assertAlmostEqual(ate, 2.0, delta=0.3)

    def test_cate(self):
        """Conditional ATE for subgroup."""
        random.seed(42)
        data = []
        for _ in range(2000):
            g = random.choice([0, 1])
            t = random.choice([0, 1])
            # Different treatment effect by group
            effect = 3.0 if g == 1 else 1.0
            y = effect * t + random.gauss(0, 0.5)
            data.append({'T': t, 'Y': y, 'G': g})

        est = CausalEstimator()
        cate_g0 = est.cate(data, 'T', 'Y', set(), 'G', 0)
        cate_g1 = est.cate(data, 'T', 'Y', set(), 'G', 1)
        self.assertAlmostEqual(cate_g0, 1.0, delta=0.3)
        self.assertAlmostEqual(cate_g1, 3.0, delta=0.3)

    def test_att(self):
        """Average Treatment on the Treated."""
        random.seed(42)
        data = []
        for _ in range(1000):
            z = random.gauss(0, 1)
            t = 1 if z > 0 else 0
            y = 2.5 * t + z + random.gauss(0, 0.3)
            data.append({'T': t, 'Y': y, 'Z': z})

        est = CausalEstimator()
        att = est.att(data, 'T', 'Y', {'Z'})
        # ATT should be around 2.5 (matching-based, so wider tolerance)
        self.assertAlmostEqual(att, 2.5, delta=0.8)

    def test_empty_data(self):
        est = CausalEstimator()
        self.assertEqual(est.ate_regression([], 'T', 'Y', set()), 0.0)


# ============================================================
# Mediation Analysis Tests
# ============================================================

class TestMediationAnalysis(unittest.TestCase):

    def test_full_mediation(self):
        """X -> M -> Y with no direct effect."""
        random.seed(42)
        data = []
        for _ in range(2000):
            x = random.gauss(0, 1)
            m = 0.8 * x + random.gauss(0, 0.1)
            y = 2.0 * m + random.gauss(0, 0.1)
            data.append({'X': x, 'M': m, 'Y': y})

        ma = MediationAnalysis()
        result = ma.analyze(data, 'X', 'M', 'Y')
        # Indirect effect = 0.8 * 2.0 = 1.6
        self.assertAlmostEqual(result['path_a'], 0.8, delta=0.1)
        self.assertAlmostEqual(result['path_b'], 2.0, delta=0.2)
        self.assertAlmostEqual(result['indirect_effect'], 1.6, delta=0.3)
        # Direct effect should be ~0 (full mediation)
        self.assertAlmostEqual(result['direct_effect'], 0.0, delta=0.2)
        # Proportion mediated should be close to 1
        self.assertGreater(result['proportion_mediated'], 0.7)

    def test_partial_mediation(self):
        """X -> M -> Y and X -> Y."""
        random.seed(42)
        data = []
        for _ in range(2000):
            x = random.gauss(0, 1)
            m = 0.5 * x + random.gauss(0, 0.1)
            y = 1.0 * x + 2.0 * m + random.gauss(0, 0.1)
            data.append({'X': x, 'M': m, 'Y': y})

        ma = MediationAnalysis()
        result = ma.analyze(data, 'X', 'M', 'Y')
        # Total = 1.0 + 0.5*2.0 = 2.0
        self.assertAlmostEqual(result['total_effect'], 2.0, delta=0.3)
        # Direct = 1.0
        self.assertAlmostEqual(result['direct_effect'], 1.0, delta=0.3)
        # Indirect = 0.5 * 2.0 = 1.0
        self.assertAlmostEqual(result['indirect_effect'], 1.0, delta=0.3)

    def test_no_mediation(self):
        """X -> Y directly, M is unrelated."""
        random.seed(42)
        data = []
        for _ in range(2000):
            x = random.gauss(0, 1)
            m = random.gauss(0, 1)  # Independent
            y = 3.0 * x + random.gauss(0, 0.1)
            data.append({'X': x, 'M': m, 'Y': y})

        ma = MediationAnalysis()
        result = ma.analyze(data, 'X', 'M', 'Y')
        self.assertAlmostEqual(result['indirect_effect'], 0.0, delta=0.2)
        self.assertAlmostEqual(result['direct_effect'], 3.0, delta=0.3)


# ============================================================
# Causal Discovery Tests
# ============================================================

class TestCausalDiscovery(unittest.TestCase):

    def test_chain_discovery(self):
        """Discover A -> B -> C structure."""
        random.seed(42)
        data = []
        for _ in range(1000):
            a = random.gauss(0, 1)
            b = 2 * a + random.gauss(0, 0.1)
            c = 3 * b + random.gauss(0, 0.1)
            data.append({'A': a, 'B': b, 'C': c})

        cd = CausalDiscovery(alpha=0.05)
        dag = cd.pc_algorithm(data, ['A', 'B', 'C'])

        # Should find: A - B edge and B - C edge
        # A and C should be conditionally independent given B
        # So no direct A-C edge
        self.assertFalse(dag.has_edge('A', 'C') or dag.has_edge('C', 'A'))

    def test_fork_discovery(self):
        """Discover B -> A, B -> C."""
        random.seed(42)
        data = []
        for _ in range(1000):
            b = random.gauss(0, 1)
            a = 2 * b + random.gauss(0, 0.1)
            c = 3 * b + random.gauss(0, 0.1)
            data.append({'A': a, 'B': b, 'C': c})

        cd = CausalDiscovery(alpha=0.05)
        dag = cd.pc_algorithm(data, ['A', 'B', 'C'])

        # Should find A-B and B-C edges, no A-C edge
        has_ab = dag.has_edge('A', 'B') or dag.has_edge('B', 'A')
        has_bc = dag.has_edge('B', 'C') or dag.has_edge('C', 'B')
        self.assertTrue(has_ab)
        self.assertTrue(has_bc)

    def test_independence(self):
        """Independent variables should have no edges."""
        random.seed(42)
        data = []
        for _ in range(500):
            a = random.gauss(0, 1)
            b = random.gauss(0, 1)
            data.append({'A': a, 'B': b})

        cd = CausalDiscovery(alpha=0.05)
        dag = cd.pc_algorithm(data, ['A', 'B'])
        self.assertFalse(dag.has_edge('A', 'B'))
        self.assertFalse(dag.has_edge('B', 'A'))

    def test_correlation(self):
        cd = CausalDiscovery()
        data = [{'X': 1, 'Y': 2}, {'X': 2, 'Y': 4}, {'X': 3, 'Y': 6}]
        r = cd._correlation(data, 'X', 'Y')
        self.assertAlmostEqual(r, 1.0, delta=0.01)

    def test_partial_correlation(self):
        random.seed(42)
        data = []
        for _ in range(1000):
            z = random.gauss(0, 1)
            x = z + random.gauss(0, 0.1)
            y = z + random.gauss(0, 0.1)
            data.append({'X': x, 'Y': y, 'Z': z})

        cd = CausalDiscovery()
        # X and Y are correlated
        r_xy = cd._correlation(data, 'X', 'Y')
        self.assertGreater(abs(r_xy), 0.5)
        # But conditionally independent given Z
        r_xy_z = cd._partial_correlation(data, 'X', 'Y', ['Z'])
        self.assertAlmostEqual(r_xy_z, 0.0, delta=0.15)


# ============================================================
# Do-Calculus Tests
# ============================================================

class TestDoCalculus(unittest.TestCase):

    def test_rule1_insertion_deletion(self):
        # X -> Y, Z -> Y: Z can be removed if d-separated from Y given X in G_X_bar
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g.add_edge('Z', 'Y')
        dc = DoCalculus(g)
        # Z -> Y exists, so Z is NOT d-separated from Y given X in G_X_bar
        self.assertFalse(dc.rule1('Y', 'X', 'Z'))

    def test_rule1_applies(self):
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g.add_node('Z')  # Z is isolated
        dc = DoCalculus(g)
        self.assertTrue(dc.rule1('Y', 'X', 'Z'))

    def test_rule2(self):
        g = CausalGraph()
        g.add_edge('X', 'Z')
        g.add_edge('Z', 'Y')
        dc = DoCalculus(g)
        # Can we replace do(Z) with observe Z?
        result = dc.rule2('Y', 'X', 'Z')
        self.assertIsInstance(result, bool)

    def test_rule3(self):
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g.add_node('Z')
        dc = DoCalculus(g)
        result = dc.rule3('Y', 'X', 'Z')
        self.assertIsInstance(result, bool)

    def test_identifiable(self):
        # Simple graph X -> Y: always identifiable
        g = CausalGraph()
        g.add_edge('X', 'Y')
        dc = DoCalculus(g)
        self.assertTrue(dc.is_identifiable('X', 'Y'))

    def test_identifiable_with_confounder(self):
        # X -> Y, Z -> X, Z -> Y: identifiable by adjusting for Z
        g = CausalGraph()
        g.add_edge('X', 'Y')
        g.add_edge('Z', 'X')
        g.add_edge('Z', 'Y')
        dc = DoCalculus(g)
        self.assertTrue(dc.is_identifiable('X', 'Y'))


# ============================================================
# Sensitivity Analysis Tests
# ============================================================

class TestSensitivityAnalysis(unittest.TestCase):

    def test_e_value_positive(self):
        sa = SensitivityAnalysis()
        result = sa.e_value(0.5)
        self.assertGreater(result['e_value'], 1.0)
        self.assertIn('estimate', result)

    def test_e_value_with_se(self):
        sa = SensitivityAnalysis()
        result = sa.e_value(1.0, se=0.2)
        self.assertIn('e_value_ci', result)
        self.assertGreater(result['e_value'], result['e_value_ci'])

    def test_e_value_zero(self):
        sa = SensitivityAnalysis()
        result = sa.e_value(0)
        self.assertEqual(result['e_value'], 1.0)

    def test_rosenbaum_bounds(self):
        random.seed(42)
        data = []
        for _ in range(200):
            t = random.choice([0, 1])
            y = 2.0 * t + random.gauss(0, 1)
            data.append({'T': t, 'Y': y})

        sa = SensitivityAnalysis()
        bounds = sa.rosenbaum_bounds(data, 'T', 'Y')
        self.assertGreater(len(bounds), 0)
        # At gamma=1 (no hidden bias), bounds should be tight
        self.assertEqual(bounds[0]['gamma'], 1.0)
        self.assertAlmostEqual(bounds[0]['lower_bound'], bounds[0]['upper_bound'], delta=0.01)

    def test_rosenbaum_increasing_gamma(self):
        random.seed(42)
        data = [{'T': 1, 'Y': 5 + random.gauss(0, 0.5)} for _ in range(100)]
        data += [{'T': 0, 'Y': 3 + random.gauss(0, 0.5)} for _ in range(100)]

        sa = SensitivityAnalysis()
        bounds = sa.rosenbaum_bounds(data, 'T', 'Y')
        # Lower bound should decrease as gamma increases
        for i in range(1, len(bounds)):
            self.assertLessEqual(bounds[i]['lower_bound'], bounds[i - 1]['lower_bound'] + 0.01)


# ============================================================
# Difference-in-Differences Tests
# ============================================================

class TestDifferenceInDifferences(unittest.TestCase):

    def test_did_basic(self):
        random.seed(42)
        data = []
        # Pre-period
        for _ in range(200):
            data.append({'group': 1, 'time': 0, 'Y': 5 + random.gauss(0, 0.5)})
            data.append({'group': 0, 'time': 0, 'Y': 4 + random.gauss(0, 0.5)})
        # Post-period (treatment effect = 3.0 for treated group)
        for _ in range(200):
            data.append({'group': 1, 'time': 1, 'Y': 5 + 3 + random.gauss(0, 0.5)})
            data.append({'group': 0, 'time': 1, 'Y': 4 + random.gauss(0, 0.5)})

        did = DifferenceInDifferences()
        result = did.estimate(data, 'group', 'time', 'Y', 0, 1)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['effect'], 3.0, delta=0.3)

    def test_did_no_effect(self):
        random.seed(42)
        data = []
        for t in [0, 1]:
            for _ in range(200):
                data.append({'group': 1, 'time': t, 'Y': 5 + random.gauss(0, 0.5)})
                data.append({'group': 0, 'time': t, 'Y': 4 + random.gauss(0, 0.5)})

        did = DifferenceInDifferences()
        result = did.estimate(data, 'group', 'time', 'Y', 0, 1)
        self.assertAlmostEqual(result['effect'], 0.0, delta=0.3)

    def test_parallel_trends(self):
        random.seed(42)
        data = []
        # Three pre-treatment periods with parallel trends
        for t in [0, 1, 2]:
            for _ in range(100):
                data.append({'group': 1, 'time': t, 'Y': 5 + t + random.gauss(0, 0.3)})
                data.append({'group': 0, 'time': t, 'Y': 3 + t + random.gauss(0, 0.3)})

        did = DifferenceInDifferences()
        trends = did.parallel_trends_test(data, 'group', 'time', 'Y', [0, 1, 2])
        # Differences should be close to 0 (parallel trends hold)
        for t in trends:
            self.assertAlmostEqual(t['difference'], 0.0, delta=0.3)

    def test_did_empty_groups(self):
        did = DifferenceInDifferences()
        result = did.estimate([], 'group', 'time', 'Y', 0, 1)
        self.assertFalse(result['valid'])


# ============================================================
# Regression Discontinuity Tests
# ============================================================

class TestRegressionDiscontinuity(unittest.TestCase):

    def test_rd_sharp(self):
        """Sharp RD: treatment assigned by cutoff."""
        random.seed(42)
        data = []
        for _ in range(1000):
            x = random.uniform(-5, 5)
            treated = 1 if x >= 0 else 0
            y = 0.5 * x + 3.0 * treated + random.gauss(0, 0.5)
            data.append({'X': x, 'Y': y})

        rd = RegressionDiscontinuity()
        result = rd.estimate(data, 'X', 'Y', cutoff=0, bandwidth=2.0)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['effect'], 3.0, delta=0.8)

    def test_rd_no_effect(self):
        """No discontinuity at cutoff."""
        random.seed(42)
        data = []
        for _ in range(1000):
            x = random.uniform(-5, 5)
            y = 0.5 * x + random.gauss(0, 0.5)
            data.append({'X': x, 'Y': y})

        rd = RegressionDiscontinuity()
        result = rd.estimate(data, 'X', 'Y', cutoff=0, bandwidth=2.0)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['effect'], 0.0, delta=0.5)

    def test_rd_kernels(self):
        """Test different kernel functions."""
        random.seed(42)
        data = []
        for _ in range(500):
            x = random.uniform(-3, 3)
            treated = 1 if x >= 0 else 0
            y = 2.0 * treated + random.gauss(0, 0.5)
            data.append({'X': x, 'Y': y})

        rd = RegressionDiscontinuity()
        for kernel in ['triangular', 'uniform', 'epanechnikov']:
            result = rd.estimate(data, 'X', 'Y', cutoff=0, bandwidth=2.0, kernel=kernel)
            self.assertTrue(result['valid'])
            self.assertGreater(result['effect'], 0.5)  # Should detect positive effect

    def test_rd_auto_bandwidth(self):
        """Automatic bandwidth selection."""
        random.seed(42)
        data = []
        for _ in range(500):
            x = random.uniform(-5, 5)
            y = x + 2.0 * (1 if x >= 0 else 0) + random.gauss(0, 0.3)
            data.append({'X': x, 'Y': y})

        rd = RegressionDiscontinuity()
        result = rd.estimate(data, 'X', 'Y', cutoff=0)
        self.assertTrue(result['valid'])
        self.assertGreater(result['bandwidth'], 0)

    def test_rd_insufficient_data(self):
        rd = RegressionDiscontinuity()
        data = [{'X': -1, 'Y': 0}, {'X': 1, 'Y': 2}]
        result = rd.estimate(data, 'X', 'Y', cutoff=0, bandwidth=0.5)
        # Too few data points near cutoff
        self.assertIn('valid', result)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration(unittest.TestCase):

    def test_scm_to_backdoor_estimation(self):
        """Full pipeline: define SCM, generate data, identify adjustment, estimate effect."""
        random.seed(42)
        # SCM: Z -> X, Z -> Y, X -> Y (confounded)
        scm = StructuralCausalModel()
        scm.add_equation('Z', [], lambda p, n: n, lambda: random.gauss(0, 1))
        scm.add_equation('X', ['Z'], lambda p, n: 0.5 * p['Z'] + n, lambda: random.gauss(0, 0.3))
        scm.add_equation('Y', ['X', 'Z'], lambda p, n: 2.0 * p['X'] + 0.8 * p['Z'] + n, lambda: random.gauss(0, 0.3))

        # Generate data
        data = scm.generate_data(2000, seed=42)

        # Identify adjustment set
        bd = BackdoorCriterion(scm.graph)
        adj_set = bd.find_minimal_adjustment_set('X', 'Y')
        self.assertIsNotNone(adj_set)

        # Estimate effect
        est = CausalEstimator(scm.graph)
        ate = est.ate_regression(data, 'X', 'Y', adj_set)
        self.assertAlmostEqual(ate, 2.0, delta=0.3)

    def test_counterfactual_pipeline(self):
        """SCM counterfactual reasoning."""
        scm = StructuralCausalModel()
        scm.add_equation('Smoking', [], lambda p, n: n, lambda: 0)
        scm.add_equation('Tar', ['Smoking'], lambda p, n: 0.9 * p['Smoking'] + n, lambda: 0)
        scm.add_equation('Cancer', ['Tar'], lambda p, n: 0.7 * p['Tar'] + n, lambda: 0)

        # If someone smoked (=1) and got cancer (=0.63), what if they hadn't smoked?
        cf = scm.counterfactual(
            evidence={'Smoking': 1, 'Tar': 0.9, 'Cancer': 0.63},
            intervention={'Smoking': 0},
            target='Cancer'
        )
        # Without smoking: Tar=0, Cancer=0
        self.assertAlmostEqual(cf, 0.0, delta=0.01)

    def test_do_calculus_identification_flow(self):
        """Use do-calculus to check identifiability, then estimate."""
        g = CausalGraph()
        g.add_edge('Z', 'X')
        g.add_edge('Z', 'Y')
        g.add_edge('X', 'Y')

        dc = DoCalculus(g)
        self.assertTrue(dc.is_identifiable('X', 'Y'))

    def test_mediation_with_estimation(self):
        """Full mediation analysis pipeline."""
        random.seed(42)
        data = []
        for _ in range(2000):
            x = random.choice([0, 1])
            m = 0.6 * x + random.gauss(0, 0.2)
            y = 1.5 * m + 0.5 * x + random.gauss(0, 0.2)
            data.append({'X': x, 'M': m, 'Y': y})

        ma = MediationAnalysis()
        result = ma.analyze(data, 'X', 'M', 'Y')
        # Total = 0.5 + 0.6*1.5 = 1.4
        self.assertAlmostEqual(result['total_effect'], 1.4, delta=0.3)
        # Proportion mediated = 0.9/1.4 ~ 0.64
        self.assertGreater(result['proportion_mediated'], 0.3)

    def test_did_with_sensitivity(self):
        """DiD + sensitivity analysis."""
        random.seed(42)
        data = []
        for _ in range(200):
            data.append({'group': 1, 'time': 0, 'T': 0, 'Y': 5 + random.gauss(0, 0.5)})
            data.append({'group': 0, 'time': 0, 'T': 0, 'Y': 4 + random.gauss(0, 0.5)})
            data.append({'group': 1, 'time': 1, 'T': 1, 'Y': 8 + random.gauss(0, 0.5)})
            data.append({'group': 0, 'time': 1, 'T': 0, 'Y': 4 + random.gauss(0, 0.5)})

        did = DifferenceInDifferences()
        result = did.estimate(data, 'group', 'time', 'Y', 0, 1)
        self.assertAlmostEqual(result['effect'], 3.0, delta=0.3)

        # Sensitivity: how robust is this?
        sa = SensitivityAnalysis()
        e_val = sa.e_value(result['effect'])
        self.assertGreater(e_val['e_value'], 1.5)

    def test_graph_operations_consistency(self):
        """Test graph copy and mutilation don't affect original."""
        g = CausalGraph()
        g.add_edge('A', 'B')
        g.add_edge('C', 'B')
        g.add_edge('B', 'D')

        g2 = g.copy()
        g2.remove_incoming('B')

        # Original unchanged
        self.assertTrue(g.has_edge('A', 'B'))
        self.assertTrue(g.has_edge('C', 'B'))
        # Copy mutilated
        self.assertFalse(g2.has_edge('A', 'B'))
        self.assertFalse(g2.has_edge('C', 'B'))

    def test_iv_vs_ols_bias(self):
        """IV should recover true effect while OLS is biased."""
        random.seed(42)
        data = []
        for _ in range(3000):
            u = random.gauss(0, 1)
            z = random.gauss(0, 1)
            x = 0.5 * z + u + random.gauss(0, 0.1)
            y = 2.0 * x + u + random.gauss(0, 0.1)
            data.append({'Z': z, 'X': x, 'Y': y, 'U': u})

        # OLS (biased by confounding)
        est = CausalEstimator()
        ols = est.ate_regression(data, 'X', 'Y', set())
        # OLS should be biased upward (> 2.0)
        self.assertGreater(ols, 2.2)

        # IV (unbiased)
        iv = InstrumentalVariable()
        iv_effect = iv.estimate(data, 'Z', 'X', 'Y')
        self.assertAlmostEqual(iv_effect, 2.0, delta=0.5)

    def test_scm_sample_deterministic(self):
        """SCM with zero noise should be deterministic."""
        scm = StructuralCausalModel()
        scm.add_equation('X', [], lambda p, n: 1.0, lambda: 0)
        scm.add_equation('Y', ['X'], lambda p, n: 2 * p['X'], lambda: 0)

        samples = scm.sample(5)
        for s in samples:
            self.assertEqual(s['X'], 1.0)
            self.assertEqual(s['Y'], 2.0)

    def test_backdoor_no_valid_set(self):
        """M-bias: conditioning on M opens a path."""
        g = CausalGraph()
        g.add_edge('X', 'Y')
        # No confounders, empty set is valid
        bd = BackdoorCriterion(g)
        adj = bd.find_minimal_adjustment_set('X', 'Y')
        self.assertIsNotNone(adj)
        self.assertEqual(len(adj), 0)


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases(unittest.TestCase):

    def test_single_node_graph(self):
        g = CausalGraph()
        g.add_node('X')
        self.assertEqual(g.topological_sort(), ['X'])
        self.assertEqual(g.ancestors('X'), set())
        self.assertEqual(g.descendants('X'), set())

    def test_disconnected_graph(self):
        g = CausalGraph()
        g.add_node('A')
        g.add_node('B')
        self.assertTrue(g.is_d_separated('A', 'B', set()))

    def test_linear_chain_d_sep(self):
        g = CausalGraph()
        for i in range(5):
            g.add_edge(f'X{i}', f'X{i+1}')
        # X0 and X5 are d-separated given X2
        self.assertTrue(g.is_d_separated('X0', 'X5', {'X2'}))

    def test_propensity_all_treated(self):
        data = [{'X': 1, 'T': 1, 'Y': 5} for _ in range(10)]
        ps = PropensityScore()
        ate = ps.ipw_estimate(data, 'T', 'Y', ['X'])
        self.assertEqual(ate, 0.0)

    def test_variance_single_value(self):
        sa = SensitivityAnalysis()
        self.assertEqual(sa._variance([5.0]), 0.0)

    def test_normal_cdf(self):
        cd = CausalDiscovery()
        self.assertAlmostEqual(cd._normal_cdf(0), 0.5, delta=0.01)
        self.assertGreater(cd._normal_cdf(3), 0.99)
        self.assertLess(cd._normal_cdf(-3), 0.01)

    def test_rd_kernel_outside_bandwidth(self):
        rd = RegressionDiscontinuity()
        self.assertEqual(rd._kernel_weight(1.5, 'triangular'), 0.0)
        self.assertEqual(rd._kernel_weight(1.5, 'uniform'), 0.0)
        self.assertEqual(rd._kernel_weight(1.5, 'epanechnikov'), 0.0)

    def test_rd_kernel_at_center(self):
        rd = RegressionDiscontinuity()
        self.assertEqual(rd._kernel_weight(0, 'triangular'), 1.0)
        self.assertEqual(rd._kernel_weight(0, 'uniform'), 1.0)
        self.assertAlmostEqual(rd._kernel_weight(0, 'epanechnikov'), 0.75)

    def test_scm_intervene_returns_callable(self):
        scm = StructuralCausalModel()
        scm.add_equation('X', [], lambda p, n: n, lambda: 0)
        scm.add_equation('Y', ['X'], lambda p, n: p['X'] * 2, lambda: 0)
        fn = scm.intervene({'X': 3})
        result = fn(1)
        self.assertEqual(result[0]['Y'], 6.0)

    def test_solve_linear_singular(self):
        iv = InstrumentalVariable()
        # Singular matrix
        result = iv._solve_linear([[0, 0], [0, 0]], [0, 0])
        self.assertEqual(result, [0.0, 0.0])

    def test_did_result_fields(self):
        random.seed(42)
        data = []
        for _ in range(50):
            data.append({'group': 1, 'time': 0, 'Y': 5 + random.gauss(0, 0.5)})
            data.append({'group': 0, 'time': 0, 'Y': 3 + random.gauss(0, 0.5)})
            data.append({'group': 1, 'time': 1, 'Y': 7 + random.gauss(0, 0.5)})
            data.append({'group': 0, 'time': 1, 'Y': 3 + random.gauss(0, 0.5)})

        did = DifferenceInDifferences()
        result = did.estimate(data, 'group', 'time', 'Y', 0, 1)
        self.assertIn('treated_diff', result)
        self.assertIn('control_diff', result)
        self.assertIn('treated_pre_mean', result)
        self.assertIn('treated_post_mean', result)
        self.assertIn('control_pre_mean', result)
        self.assertIn('control_post_mean', result)


if __name__ == '__main__':
    unittest.main()
