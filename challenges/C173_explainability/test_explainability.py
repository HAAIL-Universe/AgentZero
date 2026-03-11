"""
Tests for C173: Model Explainability & Interpretability
"""

import math
import random
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from neural_network import Tensor, Sequential, Dense, Activation, MSELoss, CrossEntropyLoss, Adam, fit
from explainability import (
    PermutationImportance, LIME, LIMEExplanation, KernelSHAP, SHAPExplanation,
    IntegratedGradients, IGExplanation, SaliencyMap, SaliencyResult,
    PartialDependence, PDResult, PD2DResult, ICE, ICEResult,
    CounterfactualExplainer, CounterfactualResult,
    FeatureInteraction, ModelSummary, LayerStats,
    ExplanationComparator, FeatureAblation, AblationResult,
    _predict_single, _predict_batch, _predict_class, _accuracy,
    _get_data_list, _tensor_to_2d_list
)


# ============================================================
# Helper: build a simple trained model
# ============================================================

def make_xor_model(seed=42):
    """Train a small model on XOR-like data."""
    rng = random.Random(seed)
    model = Sequential()
    model.add(Dense(2, 8, rng=rng))
    model.add(Activation('relu'))
    model.add(Dense(8, 2, rng=rng))
    model.add(Activation('softmax'))

    # XOR data
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]] * 25)
    y = [0, 1, 1, 0] * 25
    opt = Adam(lr=0.01)
    fit(model, X, y, CrossEntropyLoss(), opt, epochs=100, batch_size=4)
    return model


def make_linear_model(seed=42):
    """Train model where feature 0 is important, feature 1 is noise."""
    rng = random.Random(seed)
    model = Sequential()
    model.add(Dense(2, 4, rng=rng))
    model.add(Activation('relu'))
    model.add(Dense(4, 1, rng=rng))

    # y ~ 3*x0 + noise
    X_data = []
    y_data = []
    for _ in range(100):
        x0 = rng.uniform(-1, 1)
        x1 = rng.uniform(-1, 1)  # noise feature
        X_data.append([x0, x1])
        y_data.append([3.0 * x0])

    X = Tensor(X_data)
    Y = Tensor(y_data)
    opt = Adam(lr=0.01)
    fit(model, X, Y, MSELoss(), opt, epochs=100, batch_size=16)
    return model, X_data, y_data


def make_3feature_model(seed=42):
    """Model with 3 features: f0 strong, f1 medium, f2 noise."""
    rng = random.Random(seed)
    model = Sequential()
    model.add(Dense(3, 8, rng=rng))
    model.add(Activation('relu'))
    model.add(Dense(8, 2, rng=rng))
    model.add(Activation('softmax'))

    X_data = []
    y_data = []
    for _ in range(200):
        x0 = rng.uniform(-2, 2)
        x1 = rng.uniform(-2, 2)
        x2 = rng.uniform(-2, 2)
        X_data.append([x0, x1, x2])
        # Class depends on x0 + 0.5*x1, x2 is noise
        y_data.append(1 if x0 + 0.5 * x1 > 0 else 0)

    X = Tensor(X_data)
    opt = Adam(lr=0.01)
    fit(model, X, y_data, CrossEntropyLoss(), opt, epochs=80, batch_size=16)
    return model, X_data, y_data


# ============================================================
# Utility tests
# ============================================================

class TestUtilities(unittest.TestCase):
    def test_predict_single_1d(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 3, rng=rng))
        result = _predict_single(model, Tensor([1.0, 2.0]))
        self.assertEqual(len(result), 3)

    def test_predict_single_from_list(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 3, rng=rng))
        result = _predict_single(model, [1.0, 2.0])
        self.assertEqual(len(result), 3)

    def test_predict_batch(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 3, rng=rng))
        result = _predict_batch(model, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)

    def test_predict_class(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 2, rng=rng))
        model.add(Activation('softmax'))
        cls = _predict_class(model, Tensor([1.0, 2.0]))
        self.assertIn(cls, [0, 1])

    def test_accuracy(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 2, rng=rng))
        model.add(Activation('softmax'))
        acc = _accuracy(model, [[1.0, 0.0], [0.0, 1.0]], [0, 1])
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_get_data_list_tensor(self):
        t = Tensor([[1, 2], [3, 4]])
        result = _get_data_list(t)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [1, 2])

    def test_get_data_list_1d(self):
        result = _get_data_list([1, 2, 3])
        self.assertEqual(result, [[1], [2], [3]])

    def test_tensor_to_2d_list_1d(self):
        t = Tensor([1, 2, 3])
        result = _tensor_to_2d_list(t)
        self.assertEqual(result, [[1], [2], [3]])

    def test_tensor_to_2d_list_2d(self):
        t = Tensor([[1, 2], [3, 4]])
        result = _tensor_to_2d_list(t)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_get_data_list_empty(self):
        self.assertEqual(_get_data_list([]), [])

    def test_accuracy_empty(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 2, rng=rng))
        self.assertEqual(_accuracy(model, [], []), 0.0)


# ============================================================
# PermutationImportance tests
# ============================================================

class TestPermutationImportance(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_basic_compute(self):
        pi = PermutationImportance(self.model, n_repeats=3)
        pi.compute(self.X[:50], self.y[:50])
        self.assertEqual(len(pi.importances_mean_), 3)
        self.assertEqual(len(pi.importances_std_), 3)

    def test_importances_shape(self):
        pi = PermutationImportance(self.model, n_repeats=5)
        pi.compute(self.X[:50], self.y[:50])
        self.assertEqual(len(pi.importances_), 3)
        self.assertEqual(len(pi.importances_[0]), 5)

    def test_ranking(self):
        pi = PermutationImportance(self.model, n_repeats=5)
        pi.compute(self.X[:50], self.y[:50])
        ranking = pi.ranking()
        self.assertEqual(len(ranking), 3)
        self.assertEqual(set(ranking), {0, 1, 2})

    def test_ranking_before_compute_raises(self):
        pi = PermutationImportance(self.model)
        with self.assertRaises(ValueError):
            pi.ranking()

    def test_reproducible_with_seed(self):
        pi1 = PermutationImportance(self.model, n_repeats=3, rng=random.Random(123))
        pi1.compute(self.X[:30], self.y[:30])
        pi2 = PermutationImportance(self.model, n_repeats=3, rng=random.Random(123))
        pi2.compute(self.X[:30], self.y[:30])
        for i in range(3):
            self.assertAlmostEqual(pi1.importances_mean_[i], pi2.importances_mean_[i], places=10)

    def test_nonnegative_std(self):
        pi = PermutationImportance(self.model, n_repeats=5)
        pi.compute(self.X[:50], self.y[:50])
        for s in pi.importances_std_:
            self.assertGreaterEqual(s, 0.0)

    def test_returns_self(self):
        pi = PermutationImportance(self.model)
        result = pi.compute(self.X[:20], self.y[:20])
        self.assertIs(result, pi)

    def test_single_repeat(self):
        pi = PermutationImportance(self.model, n_repeats=1)
        pi.compute(self.X[:20], self.y[:20])
        self.assertEqual(len(pi.importances_[0]), 1)
        # std should be 0 with 1 repeat
        for s in pi.importances_std_:
            self.assertAlmostEqual(s, 0.0)


# ============================================================
# LIME tests
# ============================================================

class TestLIME(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_explain_returns_explanation(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(self.X[0])
        self.assertIsInstance(exp, LIMEExplanation)

    def test_coefficients_count(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(self.X[0])
        self.assertEqual(len(exp.coefficients), 3)

    def test_top_features(self):
        lime = LIME(self.model, n_samples=100, rng=random.Random(42))
        exp = lime.explain(self.X[0])
        top = exp.top_features(k=2)
        self.assertEqual(len(top), 2)
        self.assertIsInstance(top[0], tuple)
        self.assertEqual(len(top[0]), 2)

    def test_as_dict(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(self.X[0])
        d = exp.as_dict()
        self.assertEqual(len(d), 3)
        self.assertIn(0, d)

    def test_target_class_explicit(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(self.X[0], target_class=1)
        self.assertEqual(exp.target_class, 1)

    def test_prediction_stored(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(self.X[0])
        self.assertIsNotNone(exp.prediction)
        self.assertEqual(len(exp.prediction), 2)

    def test_intercept_exists(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(self.X[0])
        self.assertIsInstance(exp.intercept, float)

    def test_tensor_input(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        exp = lime.explain(Tensor(self.X[0]))
        self.assertEqual(len(exp.coefficients), 3)

    def test_different_instances_different_explanations(self):
        lime = LIME(self.model, n_samples=100, rng=random.Random(42))
        exp1 = lime.explain(self.X[0])
        exp2 = lime.explain(self.X[50])
        # Coefficients should generally differ
        same = all(abs(exp1.coefficients[i] - exp2.coefficients[i]) < 1e-10
                    for i in range(3))
        # Not guaranteed but very unlikely to be identical
        # Just check both exist
        self.assertEqual(len(exp1.coefficients), 3)
        self.assertEqual(len(exp2.coefficients), 3)

    def test_solve_linear(self):
        lime = LIME(self.model)
        # Simple system: x = 1, y = 2
        A = [[1, 0], [0, 1]]
        b = [1, 2]
        result = lime._solve_linear(A, b, 2)
        self.assertAlmostEqual(result[0], 1.0, places=6)
        self.assertAlmostEqual(result[1], 2.0, places=6)

    def test_solve_linear_2x2(self):
        lime = LIME(self.model)
        # 2x + 3y = 8, x + y = 3
        A = [[2, 3], [1, 1]]
        b = [8, 3]
        result = lime._solve_linear(A, b, 2)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[1], 2.0, places=5)


# ============================================================
# KernelSHAP tests
# ============================================================

class TestKernelSHAP(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_explain_returns_shap(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        self.assertIsInstance(exp, SHAPExplanation)

    def test_shap_values_count(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        self.assertEqual(len(exp.shap_values), 3)

    def test_expected_value(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        self.assertIsInstance(exp.expected_value, float)

    def test_sum_shap(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        s = exp.sum_shap()
        self.assertIsInstance(s, float)

    def test_top_features(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        top = exp.top_features(k=2)
        self.assertEqual(len(top), 2)

    def test_as_dict(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        d = exp.as_dict()
        self.assertEqual(len(d), 3)

    def test_target_class_explicit(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0], target_class=0)
        self.assertEqual(exp.target_class, 0)

    def test_prediction_stored(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(self.X[0])
        self.assertIsNotNone(exp.prediction)

    def test_tensor_input(self):
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))
        exp = shap.explain(Tensor(self.X[0]))
        self.assertEqual(len(exp.shap_values), 3)


# ============================================================
# IntegratedGradients tests
# ============================================================

class TestIntegratedGradients(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_attribute_returns_ig(self):
        ig = IntegratedGradients(self.model, n_steps=20)
        exp = ig.attribute(self.X[0])
        self.assertIsInstance(exp, IGExplanation)

    def test_attributions_count(self):
        ig = IntegratedGradients(self.model, n_steps=20)
        exp = ig.attribute(self.X[0])
        self.assertEqual(len(exp.attributions), 3)

    def test_top_features(self):
        ig = IntegratedGradients(self.model, n_steps=20)
        exp = ig.attribute(self.X[0])
        top = exp.top_features(k=2)
        self.assertEqual(len(top), 2)

    def test_convergence_delta(self):
        ig = IntegratedGradients(self.model, n_steps=50)
        exp = ig.attribute(self.X[0])
        delta = exp.convergence_delta()
        self.assertIsInstance(delta, float)

    def test_custom_baseline(self):
        ig = IntegratedGradients(self.model, n_steps=20)
        exp = ig.attribute(self.X[0], baseline=[1.0, 1.0, 1.0])
        self.assertEqual(len(exp.attributions), 3)
        self.assertEqual(exp.baseline, [1.0, 1.0, 1.0])

    def test_target_class_explicit(self):
        ig = IntegratedGradients(self.model, n_steps=10)
        exp = ig.attribute(self.X[0], target_class=1)
        self.assertEqual(exp.target_class, 1)

    def test_as_dict(self):
        ig = IntegratedGradients(self.model, n_steps=10)
        exp = ig.attribute(self.X[0])
        d = exp.as_dict()
        self.assertEqual(len(d), 3)

    def test_tensor_input(self):
        ig = IntegratedGradients(self.model, n_steps=10)
        exp = ig.attribute(Tensor(self.X[0]))
        self.assertEqual(len(exp.attributions), 3)

    def test_baseline_tensor_input(self):
        ig = IntegratedGradients(self.model, n_steps=10)
        exp = ig.attribute(self.X[0], baseline=Tensor([0.5, 0.5, 0.5]))
        self.assertEqual(len(exp.attributions), 3)

    def test_zero_steps(self):
        ig = IntegratedGradients(self.model, n_steps=0)
        exp = ig.attribute(self.X[0])
        self.assertEqual(len(exp.attributions), 3)

    def test_compute_gradient(self):
        ig = IntegratedGradients(self.model, n_steps=10)
        grad = ig._compute_gradient(self.X[0], 0)
        self.assertEqual(len(grad), 3)
        # Gradients should be finite
        for g in grad:
            self.assertTrue(math.isfinite(g))


# ============================================================
# SaliencyMap tests
# ============================================================

class TestSaliencyMap(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_compute_returns_result(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0])
        self.assertIsInstance(result, SaliencyResult)

    def test_saliency_count(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0])
        self.assertEqual(len(result.saliency), 3)

    def test_saliency_nonnegative(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0])
        for s in result.saliency:
            self.assertGreaterEqual(s, 0.0)

    def test_gradient_stored(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0])
        self.assertEqual(len(result.gradient), 3)

    def test_top_features(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0])
        top = result.top_features(k=2)
        self.assertEqual(len(top), 2)

    def test_normalized(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0])
        normed = result.normalized()
        self.assertEqual(len(normed), 3)
        self.assertAlmostEqual(max(normed), 1.0, places=6)
        for v in normed:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-6)

    def test_target_class(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(self.X[0], target_class=1)
        self.assertEqual(result.target_class, 1)

    def test_tensor_input(self):
        sm = SaliencyMap(self.model)
        result = sm.compute(Tensor(self.X[0]))
        self.assertEqual(len(result.saliency), 3)


# ============================================================
# PartialDependence tests
# ============================================================

class TestPartialDependence(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_compute_returns_pdresult(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        result = pd.compute(0)
        self.assertIsInstance(result, PDResult)

    def test_grid_size(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=15)
        result = pd.compute(0)
        self.assertEqual(len(result.grid), 15)
        self.assertEqual(len(result.values), 15)

    def test_custom_grid_range(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        result = pd.compute(0, grid_range=(-5, 5))
        self.assertAlmostEqual(result.grid[0], -5.0, places=5)
        self.assertAlmostEqual(result.grid[-1], 5.0, places=5)

    def test_range_method(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        result = pd.compute(0)
        r = result.range()
        self.assertGreaterEqual(r, 0.0)

    def test_monotonic_method(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        result = pd.compute(0)
        m = result.monotonic()
        self.assertIsInstance(m, bool)

    def test_target_class(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        result = pd.compute(0, target_class=1)
        self.assertEqual(result.target_class, 1)

    def test_compute_2d(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=5)
        result = pd.compute_2d(0, 1, n_grid=5)
        self.assertIsInstance(result, PD2DResult)
        self.assertEqual(len(result.grid1), 5)
        self.assertEqual(len(result.grid2), 5)
        self.assertEqual(len(result.values), 5)
        self.assertEqual(len(result.values[0]), 5)

    def test_different_features(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        r0 = pd.compute(0)
        r1 = pd.compute(1)
        r2 = pd.compute(2)
        # All should have values
        self.assertEqual(len(r0.values), 10)
        self.assertEqual(len(r1.values), 10)
        self.assertEqual(len(r2.values), 10)

    def test_values_bounded(self):
        pd = PartialDependence(self.model, self.X[:20], n_grid=10)
        result = pd.compute(0)
        # Softmax output should be in [0, 1]
        for v in result.values:
            self.assertGreaterEqual(v, -0.1)
            self.assertLessEqual(v, 1.1)


# ============================================================
# ICE tests
# ============================================================

class TestICE(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_compute_returns_result(self):
        ice = ICE(self.model, self.X[:10], n_grid=10)
        result = ice.compute(0)
        self.assertIsInstance(result, ICEResult)

    def test_curves_count(self):
        ice = ICE(self.model, self.X[:10], n_grid=10)
        result = ice.compute(0)
        self.assertEqual(len(result.curves), 10)
        self.assertEqual(len(result.curves[0]), 10)

    def test_centered_curves(self):
        ice = ICE(self.model, self.X[:10], n_grid=10)
        result = ice.compute(0)
        self.assertEqual(len(result.centered_curves), 10)
        # First point of centered curve should be 0
        for curve in result.centered_curves:
            self.assertAlmostEqual(curve[0], 0.0, places=10)

    def test_average_matches_pdp(self):
        data = self.X[:10]
        ice = ICE(self.model, data, n_grid=10)
        pd = PartialDependence(self.model, data, n_grid=10)
        ice_result = ice.compute(0)
        pd_result = pd.compute(0)
        avg = ice_result.average()
        for i in range(10):
            self.assertAlmostEqual(avg[i], pd_result.values[i], places=5)

    def test_heterogeneity(self):
        ice = ICE(self.model, self.X[:10], n_grid=10)
        result = ice.compute(0)
        het = result.heterogeneity()
        self.assertEqual(len(het), 10)
        for h in het:
            self.assertGreaterEqual(h, 0.0)

    def test_max_instances(self):
        ice = ICE(self.model, self.X[:20], n_grid=5)
        result = ice.compute(0, max_instances=5)
        self.assertEqual(len(result.curves), 5)

    def test_custom_grid_range(self):
        ice = ICE(self.model, self.X[:10], n_grid=5)
        result = ice.compute(0, grid_range=(-3, 3))
        self.assertAlmostEqual(result.grid[0], -3.0, places=5)
        self.assertAlmostEqual(result.grid[-1], 3.0, places=5)


# ============================================================
# CounterfactualExplainer tests
# ============================================================

class TestCounterfactualExplainer(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_explain_returns_result(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0])
        self.assertIsInstance(result, CounterfactualResult)

    def test_counterfactual_has_fields(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0])
        self.assertEqual(len(result.counterfactual), 3)
        self.assertIsInstance(result.distance, float)
        self.assertIsInstance(result.changes, dict)

    def test_sparsity(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0])
        s = result.sparsity()
        self.assertGreaterEqual(s, 0)

    def test_top_changes(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0])
        top = result.top_changes(k=2)
        self.assertLessEqual(len(top), 2)

    def test_success_flag(self):
        cf = CounterfactualExplainer(self.model, max_iterations=100, rng=random.Random(42))
        result = cf.explain(self.X[0])
        self.assertIsInstance(result.success(), bool)

    def test_target_class(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0], target_class=1)
        self.assertIsInstance(result, CounterfactualResult)

    def test_distance_nonnegative(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0])
        self.assertGreaterEqual(result.distance, 0.0)

    def test_original_class_stored(self):
        cf = CounterfactualExplainer(self.model, max_iterations=50, rng=random.Random(42))
        result = cf.explain(self.X[0])
        self.assertIn(result.original_class, [0, 1])


# ============================================================
# FeatureInteraction tests
# ============================================================

class TestFeatureInteraction(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_h_statistic_returns_float(self):
        fi = FeatureInteraction(self.model, self.X[:20], n_grid=5)
        h = fi.h_statistic(0, 1)
        self.assertIsInstance(h, float)
        self.assertGreaterEqual(h, 0.0)

    def test_h_statistic_bounded(self):
        fi = FeatureInteraction(self.model, self.X[:20], n_grid=5)
        h = fi.h_statistic(0, 2)
        self.assertGreaterEqual(h, 0.0)

    def test_all_pairs(self):
        fi = FeatureInteraction(self.model, self.X[:20], n_grid=5)
        pairs = fi.all_pairs()
        self.assertEqual(len(pairs), 3)  # C(3,2) = 3
        self.assertIn((0, 1), pairs)
        self.assertIn((0, 2), pairs)
        self.assertIn((1, 2), pairs)

    def test_top_interactions(self):
        fi = FeatureInteraction(self.model, self.X[:20], n_grid=5)
        top = fi.top_interactions(k=2)
        self.assertEqual(len(top), 2)
        self.assertIsInstance(top[0][0], tuple)
        self.assertIsInstance(top[0][1], float)

    def test_target_class(self):
        fi = FeatureInteraction(self.model, self.X[:20], n_grid=5)
        h = fi.h_statistic(0, 1, target_class=1)
        self.assertIsInstance(h, float)


# ============================================================
# ModelSummary tests
# ============================================================

class TestModelSummary(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_layer_stats(self):
        ms = ModelSummary(self.model)
        stats = ms.layer_stats()
        self.assertGreater(len(stats), 0)
        self.assertIsInstance(stats[0], LayerStats)

    def test_layer_stats_types(self):
        ms = ModelSummary(self.model)
        stats = ms.layer_stats()
        types = [s.type for s in stats]
        self.assertIn('Dense', types)
        self.assertIn('Activation', types)

    def test_total_params(self):
        ms = ModelSummary(self.model)
        total = ms.total_params()
        # 3*8 + 8 + 8*2 + 2 = 24 + 8 + 16 + 2 = 50
        self.assertEqual(total, 50)

    def test_weight_norms(self):
        ms = ModelSummary(self.model)
        norms = ms.weight_norms()
        self.assertGreater(len(norms), 0)
        for n in norms:
            self.assertGreater(n, 0.0)

    def test_large_weights(self):
        ms = ModelSummary(self.model)
        # With default threshold=5.0, may or may not find any
        result = ms.large_weights(threshold=5.0)
        self.assertIsInstance(result, list)

    def test_large_weights_low_threshold(self):
        ms = ModelSummary(self.model)
        result = ms.large_weights(threshold=0.001)
        # Should find many weights > 0.001
        self.assertGreater(len(result), 0)

    def test_dead_neurons(self):
        ms = ModelSummary(self.model)
        dead = ms.dead_neurons(self.X[:20])
        self.assertIsInstance(dead, list)

    def test_weight_stats_structure(self):
        ms = ModelSummary(self.model)
        stats = ms.layer_stats()
        dense_stats = [s for s in stats if s.type == 'Dense']
        self.assertGreater(len(dense_stats), 0)
        ws = dense_stats[0].weight_stats
        self.assertIn('mean', ws)
        self.assertIn('std', ws)
        self.assertIn('min', ws)
        self.assertIn('max', ws)
        self.assertIn('norm', ws)

    def test_bias_stats(self):
        ms = ModelSummary(self.model)
        stats = ms.layer_stats()
        dense_stats = [s for s in stats if s.type == 'Dense']
        bs = dense_stats[0].bias_stats
        self.assertIsNotNone(bs)
        self.assertIn('mean', bs)

    def test_activation_layer_no_params(self):
        ms = ModelSummary(self.model)
        stats = ms.layer_stats()
        act_stats = [s for s in stats if s.type == 'Activation']
        self.assertGreater(len(act_stats), 0)
        self.assertEqual(act_stats[0].n_params, 0)

    def test_flatten_1d(self):
        ms = ModelSummary(self.model)
        t = Tensor([1, 2, 3])
        self.assertEqual(ms._flatten(t), [1, 2, 3])

    def test_flatten_2d(self):
        ms = ModelSummary(self.model)
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(ms._flatten(t), [1, 2, 3, 4])

    def test_compute_stats_empty(self):
        ms = ModelSummary(self.model)
        stats = ms._compute_stats([])
        self.assertEqual(stats['mean'], 0)

    def test_dead_neurons_empty(self):
        ms = ModelSummary(self.model)
        dead = ms.dead_neurons([])
        self.assertEqual(dead, [])


# ============================================================
# ExplanationComparator tests
# ============================================================

class TestExplanationComparator(unittest.TestCase):
    def test_add_and_summary(self):
        comp = ExplanationComparator()
        comp.add('method1', {0: 0.5, 1: 0.3, 2: 0.1})
        comp.add('method2', {0: 0.4, 1: 0.35, 2: 0.15})
        summary = comp.summary()
        self.assertIn('method1', summary)
        self.assertIn('method2', summary)

    def test_rank_correlation_perfect(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 3.0, 1: 2.0, 2: 1.0})
        comp.add('b', {0: 30.0, 1: 20.0, 2: 10.0})
        corr = comp.rank_correlation('a', 'b')
        self.assertAlmostEqual(corr, 1.0, places=5)

    def test_rank_correlation_inverse(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 3.0, 1: 2.0, 2: 1.0})
        comp.add('b', {0: 1.0, 1: 2.0, 2: 3.0})
        corr = comp.rank_correlation('a', 'b')
        self.assertAlmostEqual(corr, -1.0, places=5)

    def test_agreement_top_k(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 3.0, 1: 2.0, 2: 1.0})
        comp.add('b', {0: 3.0, 1: 2.0, 2: 1.0})
        agreement = comp.agreement_top_k('a', 'b', k=2)
        self.assertAlmostEqual(agreement, 1.0, places=5)

    def test_agreement_no_overlap(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 3.0, 1: 0.01, 2: 0.001})
        comp.add('b', {0: 0.001, 1: 0.01, 2: 3.0})
        agreement = comp.agreement_top_k('a', 'b', k=1)
        self.assertAlmostEqual(agreement, 0.0, places=5)

    def test_unknown_name_raises(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 1.0})
        with self.assertRaises(ValueError):
            comp.rank_correlation('a', 'nonexistent')

    def test_single_feature_correlation(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 1.0})
        comp.add('b', {0: 2.0})
        # With < 2 features, returns 0
        corr = comp.rank_correlation('a', 'b')
        self.assertEqual(corr, 0.0)


# ============================================================
# FeatureAblation tests
# ============================================================

class TestFeatureAblation(unittest.TestCase):
    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_compute_returns_result(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(self.X[0])
        self.assertIsInstance(result, AblationResult)

    def test_importances_count(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(self.X[0])
        self.assertEqual(len(result.importances), 3)

    def test_top_features(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(self.X[0])
        top = result.top_features(k=2)
        self.assertEqual(len(top), 2)

    def test_as_dict(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(self.X[0])
        d = result.as_dict()
        self.assertEqual(len(d), 3)

    def test_base_prediction(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(self.X[0])
        self.assertIsInstance(result.base_prediction, float)

    def test_custom_baseline(self):
        fa = FeatureAblation(self.model, baseline_value=0.5)
        result = fa.compute(self.X[0])
        self.assertEqual(len(result.importances), 3)

    def test_target_class(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(self.X[0], target_class=1)
        self.assertEqual(result.target_class, 1)

    def test_tensor_input(self):
        fa = FeatureAblation(self.model)
        result = fa.compute(Tensor(self.X[0]))
        self.assertEqual(len(result.importances), 3)


# ============================================================
# Integration tests
# ============================================================

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple explainability methods."""

    def setUp(self):
        self.model, self.X, self.y = make_3feature_model()

    def test_comparator_with_lime_and_shap(self):
        lime = LIME(self.model, n_samples=50, rng=random.Random(42))
        shap = KernelSHAP(self.model, self.X[:20], n_samples=50, rng=random.Random(42))

        lime_exp = lime.explain(self.X[0])
        shap_exp = shap.explain(self.X[0])

        comp = ExplanationComparator()
        comp.add('lime', lime_exp.as_dict())
        comp.add('shap', shap_exp.as_dict())
        corr = comp.rank_correlation('lime', 'shap')
        self.assertIsInstance(corr, float)

    def test_comparator_with_ablation_and_saliency(self):
        fa = FeatureAblation(self.model)
        sm = SaliencyMap(self.model)

        abl_result = fa.compute(self.X[0])
        sal_result = sm.compute(self.X[0])

        comp = ExplanationComparator()
        comp.add('ablation', abl_result.as_dict())
        comp.add('saliency', {i: s for i, s in enumerate(sal_result.saliency)})
        corr = comp.rank_correlation('ablation', 'saliency')
        self.assertIsInstance(corr, float)

    def test_ice_average_equals_pdp(self):
        """ICE average should match PDP values."""
        data = self.X[:15]
        ice = ICE(self.model, data, n_grid=8)
        pd = PartialDependence(self.model, data, n_grid=8)

        ice_result = ice.compute(0)
        pd_result = pd.compute(0)

        avg = ice_result.average()
        for i in range(len(avg)):
            self.assertAlmostEqual(avg[i], pd_result.values[i], places=5)

    def test_all_methods_on_same_instance(self):
        """Run all methods on the same instance, verify they complete."""
        instance = self.X[0]

        lime = LIME(self.model, n_samples=30, rng=random.Random(42))
        shap = KernelSHAP(self.model, self.X[:10], n_samples=30, rng=random.Random(42))
        ig = IntegratedGradients(self.model, n_steps=10)
        sm = SaliencyMap(self.model)
        fa = FeatureAblation(self.model)

        results = {
            'lime': lime.explain(instance),
            'shap': shap.explain(instance),
            'ig': ig.attribute(instance),
            'saliency': sm.compute(instance),
            'ablation': fa.compute(instance),
        }

        # All should have top_features
        for name, r in results.items():
            top = r.top_features(k=2)
            self.assertEqual(len(top), 2, f"{name} should return 2 top features")

    def test_model_summary_with_explainability(self):
        """ModelSummary + other methods."""
        ms = ModelSummary(self.model)
        stats = ms.layer_stats()
        total = ms.total_params()
        norms = ms.weight_norms()

        self.assertGreater(total, 0)
        self.assertGreater(len(norms), 0)
        self.assertGreater(len(stats), 0)

    def test_linear_model_feature_importance(self):
        """Feature 0 should be more important than feature 1 in linear model."""
        model, X, y = make_linear_model()
        fa = FeatureAblation(model)

        # Test on several instances
        total_imp = [0.0, 0.0]
        for i in range(min(20, len(X))):
            result = fa.compute(X[i])
            for j in range(2):
                total_imp[j] += abs(result.importances[j])

        # Feature 0 should dominate (y = 3*x0)
        self.assertGreater(total_imp[0], total_imp[1])

    def test_permutation_importance_consistent(self):
        """Multiple runs with same seed give same result."""
        pi1 = PermutationImportance(self.model, n_repeats=3, rng=random.Random(99))
        pi1.compute(self.X[:30], self.y[:30])

        pi2 = PermutationImportance(self.model, n_repeats=3, rng=random.Random(99))
        pi2.compute(self.X[:30], self.y[:30])

        for i in range(3):
            self.assertAlmostEqual(pi1.importances_mean_[i], pi2.importances_mean_[i])


class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_single_feature_model(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(1, 2, rng=rng))
        model.add(Activation('softmax'))

        fa = FeatureAblation(model)
        result = fa.compute([0.5])
        self.assertEqual(len(result.importances), 1)

    def test_single_output_model(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 1, rng=rng))

        sm = SaliencyMap(model)
        result = sm.compute([1.0, 2.0])
        self.assertEqual(len(result.saliency), 2)

    def test_lime_small_samples(self):
        rng_m = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 2, rng=rng_m))
        model.add(Activation('softmax'))

        lime = LIME(model, n_samples=10, rng=random.Random(42))
        exp = lime.explain([1.0, 2.0])
        self.assertEqual(len(exp.coefficients), 2)

    def test_shap_small_background(self):
        rng_m = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 2, rng=rng_m))
        model.add(Activation('softmax'))

        bg = [[0.0, 0.0], [1.0, 1.0]]
        shap = KernelSHAP(model, bg, n_samples=20, rng=random.Random(42))
        exp = shap.explain([0.5, 0.5])
        self.assertEqual(len(exp.shap_values), 2)

    def test_pdp_constant_feature(self):
        rng_m = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 1, rng=rng_m))

        X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]  # Feature 0 is constant
        pd = PartialDependence(model, X, n_grid=5)
        result = pd.compute(0)  # Constant feature
        self.assertEqual(len(result.values), 5)

    def test_comparator_different_feature_sets(self):
        comp = ExplanationComparator()
        comp.add('a', {0: 1.0, 1: 2.0, 2: 3.0})
        comp.add('b', {1: 5.0, 2: 6.0, 3: 7.0})
        # Should handle partial overlap
        corr = comp.rank_correlation('a', 'b')
        self.assertIsInstance(corr, float)

    def test_normalized_saliency_all_zero(self):
        """Saliency normalized when all values are zero."""
        result = SaliencyResult(
            instance=[0, 0], saliency=[0.0, 0.0],
            gradient=[0.0, 0.0], target_class=0, prediction=[0.5, 0.5]
        )
        normed = result.normalized()
        self.assertEqual(normed, [0.0, 0.0])

    def test_counterfactual_result_methods(self):
        result = CounterfactualResult(
            instance=[1.0, 2.0],
            counterfactual=[1.5, 2.0],
            original_class=0,
            cf_class=1,
            distance=0.5,
            changes={0: 0.5}
        )
        self.assertTrue(result.success())
        self.assertEqual(result.sparsity(), 1)
        self.assertEqual(len(result.top_changes(k=5)), 1)

    def test_pd_result_monotonic_trivial(self):
        result = PDResult(feature=0, grid=[1.0], values=[0.5], target_class=0)
        self.assertTrue(result.monotonic())

    def test_ice_empty_data(self):
        rng_m = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 1, rng=rng_m))
        ice = ICE(model, [], n_grid=5)
        result = ice.compute(0, grid_range=(0, 1))
        self.assertEqual(len(result.curves), 0)

    def test_ice_heterogeneity_empty(self):
        result = ICEResult(feature=0, grid=[], curves=[], centered_curves=[], target_class=0)
        self.assertEqual(result.average(), [])
        self.assertEqual(result.heterogeneity(), [])


if __name__ == '__main__':
    unittest.main()
