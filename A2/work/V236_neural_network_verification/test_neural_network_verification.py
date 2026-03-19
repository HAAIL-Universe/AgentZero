"""Tests for V236: Neural Network Verification."""

import numpy as np
import pytest
from neural_network_verification import (
    NeuralNetwork, Layer, Activation, HyperRectangle,
    linf_ball, box_spec,
    IntervalBounds, ZonotopeBounds,
    ibp_verify, ibp_propagate_layer,
    deeppoly_verify,
    zonotope_verify, zonotope_from_box, zonotope_propagate_layer,
    verify_output_bounds, verify_robustness, verify_monotonicity,
    compute_output_range, estimate_lipschitz, lipschitz_upper_bound,
    compare_methods, tightness_analysis,
    build_simple_relu_net, build_classifier, build_monotone_net,
    VerificationResult,
    _relu_linear_bounds, _sigmoid_linear_bounds, _tanh_linear_bounds,
    _apply_activation,
)


# ---------------------------------------------------------------------------
# Neural Network basics
# ---------------------------------------------------------------------------

class TestNeuralNetwork:
    def test_single_layer_linear(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1, 0], [0, 1]]), np.array([1, 2]), Activation.LINEAR)
        y = net.forward(np.array([3.0, 4.0]))
        np.testing.assert_allclose(y, [4.0, 6.0])

    def test_single_layer_relu(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1, -1], [-1, 1]]), np.array([0, 0]), Activation.RELU)
        y = net.forward(np.array([3.0, 1.0]))
        np.testing.assert_allclose(y, [2.0, 0.0])  # [3-1=2, -3+1=-2 -> 0]

    def test_two_layer_relu(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0], [-1.0]]), np.array([0.0, 0.0]), Activation.RELU)
        net.add_layer(np.array([[1.0, 1.0]]), np.array([0.0]), Activation.LINEAR)
        # input 2: layer1=[relu(2), relu(-2)]=[2,0], layer2=[2+0]=2
        y = net.forward(np.array([2.0]))
        np.testing.assert_allclose(y, [2.0])
        # input -3: layer1=[relu(-3), relu(3)]=[0,3], layer2=[0+3]=3
        y = net.forward(np.array([-3.0]))
        np.testing.assert_allclose(y, [3.0])

    def test_forward_all(self):
        net = NeuralNetwork()
        net.add_layer(np.eye(2), np.array([1, 1]), Activation.RELU)
        acts = net.forward_all(np.array([-2.0, 3.0]))
        assert len(acts) == 2
        np.testing.assert_allclose(acts[0], [-2, 3])
        np.testing.assert_allclose(acts[1], [0, 4])  # relu([-1, 4])

    def test_properties(self):
        net = NeuralNetwork()
        net.add_layer(np.ones((3, 2)), np.zeros(3), Activation.RELU)
        net.add_layer(np.ones((1, 3)), np.zeros(1), Activation.LINEAR)
        assert net.input_dim == 2
        assert net.output_dim == 1
        assert net.depth == 2

    def test_sigmoid_activation(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0]]), np.array([0.0]), Activation.SIGMOID)
        y = net.forward(np.array([0.0]))
        np.testing.assert_allclose(y, [0.5], atol=1e-10)

    def test_tanh_activation(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0]]), np.array([0.0]), Activation.TANH)
        y = net.forward(np.array([0.0]))
        np.testing.assert_allclose(y, [0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Input Specifications
# ---------------------------------------------------------------------------

class TestInputSpec:
    def test_hyperrectangle(self):
        box = HyperRectangle(np.array([0, 0]), np.array([1, 1]))
        assert box.dim == 2
        assert box.contains(np.array([0.5, 0.5]))
        assert not box.contains(np.array([1.5, 0.5]))
        np.testing.assert_allclose(box.center, [0.5, 0.5])
        np.testing.assert_allclose(box.radius, [0.5, 0.5])

    def test_volume(self):
        box = HyperRectangle(np.array([0, 0, 0]), np.array([2, 3, 4]))
        assert box.volume() == pytest.approx(24.0)

    def test_linf_ball(self):
        ball = linf_ball(np.array([1.0, 2.0]), 0.5)
        np.testing.assert_allclose(ball.lower, [0.5, 1.5])
        np.testing.assert_allclose(ball.upper, [1.5, 2.5])

    def test_box_spec(self):
        box = box_spec([0, 1], [2, 3])
        np.testing.assert_allclose(box.lower, [0, 1])
        np.testing.assert_allclose(box.upper, [2, 3])


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu(self):
        x = np.array([-2, -1, 0, 1, 2])
        np.testing.assert_allclose(_apply_activation(x, Activation.RELU), [0, 0, 0, 1, 2])

    def test_sigmoid(self):
        x = np.array([0.0])
        np.testing.assert_allclose(_apply_activation(x, Activation.SIGMOID), [0.5])

    def test_tanh(self):
        x = np.array([0.0])
        np.testing.assert_allclose(_apply_activation(x, Activation.TANH), [0.0])

    def test_linear(self):
        x = np.array([-1, 0, 1])
        np.testing.assert_allclose(_apply_activation(x, Activation.LINEAR), [-1, 0, 1])


# ---------------------------------------------------------------------------
# ReLU linear relaxation
# ---------------------------------------------------------------------------

class TestReLURelaxation:
    def test_strictly_positive(self):
        alpha_l, beta_l, alpha_u, beta_u = _relu_linear_bounds(
            np.array([1.0]), np.array([3.0])
        )
        np.testing.assert_allclose(alpha_l, [1.0])
        np.testing.assert_allclose(alpha_u, [1.0])
        np.testing.assert_allclose(beta_l, [0.0])
        np.testing.assert_allclose(beta_u, [0.0])

    def test_strictly_negative(self):
        alpha_l, beta_l, alpha_u, beta_u = _relu_linear_bounds(
            np.array([-3.0]), np.array([-1.0])
        )
        np.testing.assert_allclose(alpha_l, [0.0])
        np.testing.assert_allclose(alpha_u, [0.0])

    def test_crossing(self):
        alpha_l, beta_l, alpha_u, beta_u = _relu_linear_bounds(
            np.array([-2.0]), np.array([3.0])
        )
        # Upper bound: line from (-2,0) to (3,3), slope=3/5
        assert alpha_u[0] == pytest.approx(3.0 / 5.0)
        assert beta_u[0] == pytest.approx(6.0 / 5.0)
        # Lower bound: since |3| > |-2|, use slope 1
        assert alpha_l[0] == pytest.approx(1.0)

    def test_soundness(self):
        """Linear relaxation must always bound relu from below and above."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            lb = rng.uniform(-5, 0, size=5)
            ub = rng.uniform(0, 5, size=5)
            alpha_l, beta_l, alpha_u, beta_u = _relu_linear_bounds(lb, ub)
            # Check at sample points per dimension
            for j in range(5):
                for x_val in np.linspace(lb[j], ub[j], 20):
                    relu_val = max(0, x_val)
                    lower = alpha_l[j] * x_val + beta_l[j]
                    upper = alpha_u[j] * x_val + beta_u[j]
                    assert lower <= relu_val + 1e-10, f"Lower bound violated at x={x_val}"
                    assert upper >= relu_val - 1e-10, f"Upper bound violated at x={x_val}"


# ---------------------------------------------------------------------------
# Sigmoid/Tanh relaxation
# ---------------------------------------------------------------------------

class TestSigmoidRelaxation:
    def test_soundness(self):
        rng = np.random.default_rng(1)
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))
        for _ in range(50):
            lb = rng.uniform(-3, 0, size=3)
            ub = rng.uniform(0, 3, size=3)
            alpha_l, beta_l, alpha_u, beta_u = _sigmoid_linear_bounds(lb, ub)
            for j in range(3):
                for x_val in np.linspace(lb[j], ub[j], 20):
                    sv = sig(x_val)
                    lower = alpha_l[j] * x_val + beta_l[j]
                    upper = alpha_u[j] * x_val + beta_u[j]
                    assert lower <= sv + 1e-6
                    assert upper >= sv - 1e-6


class TestTanhRelaxation:
    def test_soundness(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            lb = rng.uniform(-3, 0, size=3)
            ub = rng.uniform(0, 3, size=3)
            alpha_l, beta_l, alpha_u, beta_u = _tanh_linear_bounds(lb, ub)
            for j in range(3):
                for x_val in np.linspace(lb[j], ub[j], 20):
                    tv = np.tanh(x_val)
                    lower = alpha_l[j] * x_val + beta_l[j]
                    upper = alpha_u[j] * x_val + beta_u[j]
                    assert lower <= tv + 1e-6
                    assert upper >= tv - 1e-6


# ---------------------------------------------------------------------------
# Zonotope basics
# ---------------------------------------------------------------------------

class TestZonotope:
    def test_from_box(self):
        box = box_spec([0, 0], [2, 4])
        z = zonotope_from_box(box)
        np.testing.assert_allclose(z.center, [1, 2])
        assert z.n_generators == 2
        # Generators should be diag([1, 2])
        np.testing.assert_allclose(z.generators, [[1, 0], [0, 2]])

    def test_to_interval(self):
        z = ZonotopeBounds(
            center=np.array([1.0, 2.0]),
            generators=np.array([[1.0, 0.0], [0.0, 2.0]])
        )
        iv = z.to_interval()
        np.testing.assert_allclose(iv.lower, [0, 0])
        np.testing.assert_allclose(iv.upper, [2, 4])

    def test_zonotope_contains_box(self):
        box = box_spec([-1, -1], [1, 1])
        z = zonotope_from_box(box)
        iv = z.to_interval()
        # Zonotope interval should contain the original box
        assert np.all(iv.lower <= box.lower + 1e-10)
        assert np.all(iv.upper >= box.upper - 1e-10)


# ---------------------------------------------------------------------------
# IBP Verification
# ---------------------------------------------------------------------------

class TestIBP:
    def test_identity_net(self):
        net = NeuralNetwork()
        net.add_layer(np.eye(2), np.zeros(2), Activation.LINEAR)
        box = box_spec([1, 2], [3, 4])
        bounds = ibp_verify(net, box)
        np.testing.assert_allclose(bounds.lower, [1, 2])
        np.testing.assert_allclose(bounds.upper, [3, 4])

    def test_relu_net(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1, -1], [-1, 1]]), np.zeros(2), Activation.RELU)
        box = box_spec([1, 0], [2, 1])
        bounds = ibp_verify(net, box)
        # [1-1, 2-0] = [0, 2] -> relu -> [0, 2]
        # [-2+0, -1+1] = [-2, 0] -> relu -> [0, 0]
        assert bounds.lower[0] >= 0
        assert bounds.upper[0] >= 0

    def test_soundness_random(self):
        """IBP bounds must contain all concrete outputs."""
        rng = np.random.default_rng(10)
        net = build_simple_relu_net([3, 4, 2], rng)
        box = box_spec([-1, -1, -1], [1, 1, 1])
        bounds = ibp_verify(net, box)

        for _ in range(1000):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-10), f"IBP lower violated: {y} < {bounds.lower}"
            assert np.all(y <= bounds.upper + 1e-10), f"IBP upper violated: {y} > {bounds.upper}"

    def test_scaling(self):
        """Doubling weights should double the output bounds."""
        net1 = NeuralNetwork()
        net1.add_layer(np.array([[2.0]]), np.array([0.0]), Activation.LINEAR)
        net2 = NeuralNetwork()
        net2.add_layer(np.array([[4.0]]), np.array([0.0]), Activation.LINEAR)

        box = box_spec([1.0], [3.0])
        b1 = ibp_verify(net1, box)
        b2 = ibp_verify(net2, box)
        np.testing.assert_allclose(b2.lower, 2 * b1.lower)
        np.testing.assert_allclose(b2.upper, 2 * b1.upper)

    def test_bias_shift(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0]]), np.array([5.0]), Activation.LINEAR)
        box = box_spec([0.0], [1.0])
        bounds = ibp_verify(net, box)
        np.testing.assert_allclose(bounds.lower, [5.0])
        np.testing.assert_allclose(bounds.upper, [6.0])


# ---------------------------------------------------------------------------
# DeepPoly Verification
# ---------------------------------------------------------------------------

class TestDeepPoly:
    def test_identity(self):
        net = NeuralNetwork()
        net.add_layer(np.eye(2), np.zeros(2), Activation.LINEAR)
        box = box_spec([1, 2], [3, 4])
        bounds = deeppoly_verify(net, box)
        np.testing.assert_allclose(bounds.lower, [1, 2])
        np.testing.assert_allclose(bounds.upper, [3, 4])

    def test_soundness_random(self):
        rng = np.random.default_rng(20)
        net = build_simple_relu_net([3, 5, 3, 2], rng)
        box = box_spec([-1, -1, -1], [1, 1, 1])
        bounds = deeppoly_verify(net, box)

        for _ in range(1000):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-9)
            assert np.all(y <= bounds.upper + 1e-9)

    def test_linear_network_exact(self):
        """DeepPoly should be exact for linear networks (no activations)."""
        net = NeuralNetwork()
        W1 = np.array([[1, 2], [3, -1]])
        W2 = np.array([[2, 1]])
        net.add_layer(W1, np.array([0, 0]), Activation.LINEAR)
        net.add_layer(W2, np.array([0]), Activation.LINEAR)
        box = box_spec([0, 0], [1, 1])
        dp_bounds = deeppoly_verify(net, box)
        # Combined: y = [5, 3] @ x. On [0,1]^2: min=0, max=8
        np.testing.assert_allclose(dp_bounds.lower, [0.0], atol=1e-10)
        np.testing.assert_allclose(dp_bounds.upper, [8.0], atol=1e-10)
        # DeepPoly should be at least as tight as IBP
        ibp_bounds = ibp_verify(net, box)
        assert dp_bounds.lower[0] >= ibp_bounds.lower[0] - 1e-10
        assert dp_bounds.upper[0] <= ibp_bounds.upper[0] + 1e-10

    def test_sigmoid_soundness(self):
        rng = np.random.default_rng(35)
        net = NeuralNetwork()
        net.add_layer(rng.standard_normal((3, 2)), np.zeros(3), Activation.SIGMOID)
        net.add_layer(rng.standard_normal((1, 3)), np.zeros(1), Activation.LINEAR)
        box = box_spec([-1, -1], [1, 1])
        bounds = deeppoly_verify(net, box)
        for _ in range(500):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-6)
            assert np.all(y <= bounds.upper + 1e-6)

    def test_tanh_soundness(self):
        rng = np.random.default_rng(36)
        net = NeuralNetwork()
        net.add_layer(rng.standard_normal((3, 2)), np.zeros(3), Activation.TANH)
        net.add_layer(rng.standard_normal((1, 3)), np.zeros(1), Activation.LINEAR)
        box = box_spec([-1, -1], [1, 1])
        bounds = deeppoly_verify(net, box)
        for _ in range(500):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-6)
            assert np.all(y <= bounds.upper + 1e-6)


# ---------------------------------------------------------------------------
# Zonotope Verification
# ---------------------------------------------------------------------------

class TestZonotopeVerification:
    def test_identity(self):
        net = NeuralNetwork()
        net.add_layer(np.eye(2), np.zeros(2), Activation.LINEAR)
        box = box_spec([1, 2], [3, 4])
        bounds = zonotope_verify(net, box)
        np.testing.assert_allclose(bounds.lower, [1, 2], atol=1e-10)
        np.testing.assert_allclose(bounds.upper, [3, 4], atol=1e-10)

    def test_soundness_random(self):
        rng = np.random.default_rng(40)
        net = build_simple_relu_net([3, 4, 2], rng)
        box = box_spec([-1, -1, -1], [1, 1, 1])
        bounds = zonotope_verify(net, box)

        for _ in range(1000):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-9)
            assert np.all(y <= bounds.upper + 1e-9)

    def test_propagation_linear_exact(self):
        """Zonotope should be exact for linear networks."""
        net = NeuralNetwork()
        W = np.array([[1, 2], [3, 4]])
        net.add_layer(W, np.array([1, 0]), Activation.LINEAR)
        box = box_spec([0, 0], [1, 1])
        bounds = zonotope_verify(net, box)
        # W @ [0,0] + b = [1, 0], W @ [1,1] + b = [4, 7]
        # But with combinations: min/max over corners
        # All 4 corners: (0,0)->[1,0], (1,0)->[2,3], (0,1)->[3,4], (1,1)->[4,7]
        np.testing.assert_allclose(bounds.lower, [1, 0], atol=1e-10)
        np.testing.assert_allclose(bounds.upper, [4, 7], atol=1e-10)


# ---------------------------------------------------------------------------
# Robustness Verification
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_trivially_robust(self):
        """A network with large margin should be verified robust for small epsilon."""
        net = NeuralNetwork()
        # Net that always gives [10, 0] regardless of input perturbation
        net.add_layer(np.zeros((2, 2)), np.array([10.0, 0.0]), Activation.LINEAR)
        x = np.array([0.5, 0.5])
        report = verify_robustness(net, x, epsilon=1.0, true_label=0)
        assert report.result == VerificationResult.VERIFIED

    def test_not_robust(self):
        """A sensitive network should fail robustness for large epsilon."""
        net = NeuralNetwork()
        # Identity-like: output = input, so class depends entirely on input
        net.add_layer(np.array([[10, 0], [0, 10]]), np.zeros(2), Activation.LINEAR)
        x = np.array([0.1, 0.0])  # Barely class 0
        report = verify_robustness(net, x, epsilon=0.2, true_label=0)
        # Should find counterexample or be unknown
        assert report.result != VerificationResult.VERIFIED

    def test_robustness_methods_agree(self):
        """All methods should agree on trivially robust cases."""
        net = NeuralNetwork()
        net.add_layer(np.zeros((2, 2)), np.array([10.0, 0.0]), Activation.LINEAR)
        x = np.array([0.5, 0.5])
        for method in ["ibp", "deeppoly", "zonotope"]:
            report = verify_robustness(net, x, epsilon=1.0, true_label=0, method=method)
            assert report.result == VerificationResult.VERIFIED

    def test_robustness_small_eps(self):
        """With very small epsilon, most nets should be robust at their prediction."""
        rng = np.random.default_rng(50)
        net = build_classifier(3, [8, 4], 3, rng)
        x = rng.standard_normal(3)
        true_label = int(np.argmax(net.forward(x)))
        report = verify_robustness(net, x, epsilon=1e-6, true_label=true_label)
        assert report.result == VerificationResult.VERIFIED


# ---------------------------------------------------------------------------
# Output Bounds Verification
# ---------------------------------------------------------------------------

class TestOutputBounds:
    def test_verified(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0]]), np.array([0.0]), Activation.RELU)
        box = box_spec([0.0], [1.0])
        report = verify_output_bounds(net, box,
                                       output_lower=np.array([-1.0]),
                                       output_upper=np.array([2.0]))
        assert report.result == VerificationResult.VERIFIED

    def test_lower_violated(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[-1.0]]), np.array([0.0]), Activation.LINEAR)
        box = box_spec([0.0], [1.0])
        # Output range: [-1, 0]. Lower bound 0.5 should fail.
        report = verify_output_bounds(net, box, output_lower=np.array([0.5]))
        assert report.result == VerificationResult.UNKNOWN

    def test_upper_violated(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[2.0]]), np.array([0.0]), Activation.LINEAR)
        box = box_spec([0.0], [5.0])
        # Output range: [0, 10]. Upper bound 5 should fail.
        report = verify_output_bounds(net, box, output_upper=np.array([5.0]))
        assert report.result == VerificationResult.UNKNOWN

    def test_all_methods(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0]]), np.array([0.0]), Activation.RELU)
        box = box_spec([0.0], [1.0])
        for method in ["ibp", "deeppoly", "zonotope"]:
            report = verify_output_bounds(net, box,
                                           output_lower=np.array([-1.0]),
                                           output_upper=np.array([2.0]),
                                           method=method)
            assert report.result == VerificationResult.VERIFIED


# ---------------------------------------------------------------------------
# Monotonicity Verification
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_positive_weights_monotone(self):
        net = build_monotone_net(input_dim=1, hidden_dim=4, output_dim=1)
        box = box_spec([0.0], [5.0])
        report = verify_monotonicity(net, box, input_dim=0, output_dim=0, increasing=True)
        assert report.result == VerificationResult.VERIFIED

    def test_negative_weights_decreasing(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[-2.0]]), np.array([10.0]), Activation.LINEAR)
        box = box_spec([0.0], [5.0])
        report = verify_monotonicity(net, box, input_dim=0, output_dim=0, increasing=False)
        assert report.result == VerificationResult.VERIFIED

    def test_non_monotone_unknown(self):
        # f(x) = relu(x) - 2*relu(x-1): increases on [0,1] then decreases for x>1
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0], [1.0]]), np.array([0.0, -1.0]), Activation.RELU)
        net.add_layer(np.array([[1.0, -2.0]]), np.array([0.0]), Activation.LINEAR)
        # x=0: [0, 0] -> 0; x=0.5: [0.5, 0] -> 0.5; x=1: [1, 0] -> 1
        # x=1.5: [1.5, 0.5] -> 0.5; x=2: [2, 1] -> 0 (decreasing after x=1)
        box = box_spec([0.0], [3.0])
        report = verify_monotonicity(net, box, input_dim=0, output_dim=0, increasing=True)
        assert report.result == VerificationResult.UNKNOWN


# ---------------------------------------------------------------------------
# Lipschitz Analysis
# ---------------------------------------------------------------------------

class TestLipschitz:
    def test_identity_lipschitz(self):
        net = NeuralNetwork()
        net.add_layer(np.eye(2), np.zeros(2), Activation.LINEAR)
        lip = lipschitz_upper_bound(net)
        assert lip == pytest.approx(1.0)

    def test_scaling_lipschitz(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[3.0, 0], [0, 3.0]]), np.zeros(2), Activation.LINEAR)
        lip = lipschitz_upper_bound(net)
        assert lip == pytest.approx(3.0)

    def test_estimate_vs_bound(self):
        """Estimated Lipschitz should be <= upper bound."""
        rng = np.random.default_rng(60)
        net = build_simple_relu_net([2, 4, 2], rng)
        box = box_spec([-1, -1], [1, 1])
        estimated = estimate_lipschitz(net, box, n_samples=2000, rng=rng)
        upper = lipschitz_upper_bound(net)
        assert estimated <= upper + 1e-6

    def test_sigmoid_lipschitz(self):
        net = NeuralNetwork()
        net.add_layer(np.eye(2), np.zeros(2), Activation.SIGMOID)
        lip = lipschitz_upper_bound(net)
        # Sigmoid Lipschitz = 0.25, times spectral norm 1
        assert lip == pytest.approx(0.25)

    def test_multi_layer_lipschitz(self):
        net = NeuralNetwork()
        net.add_layer(2 * np.eye(2), np.zeros(2), Activation.RELU)
        net.add_layer(3 * np.eye(2), np.zeros(2), Activation.LINEAR)
        lip = lipschitz_upper_bound(net)
        assert lip == pytest.approx(6.0)  # 2 * 1 * 3 * 1


# ---------------------------------------------------------------------------
# Method Comparison
# ---------------------------------------------------------------------------

class TestComparison:
    def test_compare_methods(self):
        rng = np.random.default_rng(70)
        net = build_simple_relu_net([2, 4, 2], rng)
        box = box_spec([-0.5, -0.5], [0.5, 0.5])
        results = compare_methods(net, box)
        assert "ibp" in results
        assert "deeppoly" in results
        assert "zonotope" in results
        # All should be sound
        for name, bounds in results.items():
            for _ in range(200):
                x = rng.uniform(box.lower, box.upper)
                y = net.forward(x)
                assert np.all(y >= bounds.lower - 1e-9), f"{name} lower violated"
                assert np.all(y <= bounds.upper + 1e-9), f"{name} upper violated"

    def test_tightness_analysis(self):
        rng = np.random.default_rng(71)
        net = build_simple_relu_net([2, 4, 1], rng)
        box = box_spec([-0.5, -0.5], [0.5, 0.5])
        analysis = tightness_analysis(net, box, n_samples=5000, rng=rng)
        assert "actual" in analysis
        assert "ibp" in analysis
        assert "deeppoly" in analysis
        assert "zonotope" in analysis
        # Tightness should be between 0 and 1
        for method in ["ibp", "deeppoly", "zonotope"]:
            assert 0 <= analysis[method]["mean_tightness"] <= 1.0 + 1e-6

    def test_all_methods_sound_on_deep_net(self):
        """All three methods produce sound bounds on a deep network."""
        rng = np.random.default_rng(72)
        net = build_simple_relu_net([3, 8, 8, 4, 2], rng)
        box = box_spec([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])
        results = compare_methods(net, box)
        for name, bounds in results.items():
            for _ in range(500):
                x = rng.uniform(box.lower, box.upper)
                y = net.forward(x)
                assert np.all(y >= bounds.lower - 1e-9), f"{name} lower violated"
                assert np.all(y <= bounds.upper + 1e-9), f"{name} upper violated"


# ---------------------------------------------------------------------------
# Network Builders
# ---------------------------------------------------------------------------

class TestBuilders:
    def test_build_simple(self):
        net = build_simple_relu_net([3, 4, 2])
        assert net.input_dim == 3
        assert net.output_dim == 2
        assert net.depth == 2
        y = net.forward(np.zeros(3))
        assert len(y) == 2

    def test_build_classifier(self):
        net = build_classifier(5, [10, 8], 3)
        assert net.input_dim == 5
        assert net.output_dim == 3
        y = net.forward(np.ones(5))
        assert len(y) == 3

    def test_build_monotone(self):
        net = build_monotone_net()
        assert net.input_dim == 1
        assert net.output_dim == 1
        # Should be monotonically increasing
        prev = net.forward(np.array([0.0]))[0]
        for x_val in np.linspace(0, 5, 50):
            curr = net.forward(np.array([x_val]))[0]
            assert curr >= prev - 1e-10
            prev = curr

    def test_different_seeds(self):
        rng1 = np.random.default_rng(100)
        rng2 = np.random.default_rng(200)
        net1 = build_simple_relu_net([2, 3], rng1)
        net2 = build_simple_relu_net([2, 3], rng2)
        x = np.array([1.0, 1.0])
        # Different seeds should give different networks
        assert not np.allclose(net1.forward(x), net2.forward(x))


# ---------------------------------------------------------------------------
# Compute Output Range
# ---------------------------------------------------------------------------

class TestOutputRange:
    def test_linear_exact(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[2.0, 1.0]]), np.array([0.0]), Activation.LINEAR)
        box = box_spec([0, 0], [1, 1])
        bounds = compute_output_range(net, box, method="ibp")
        np.testing.assert_allclose(bounds.lower, [0.0])
        np.testing.assert_allclose(bounds.upper, [3.0])

    def test_relu_positive_region(self):
        net = NeuralNetwork()
        net.add_layer(np.array([[1.0]]), np.array([0.0]), Activation.RELU)
        box = box_spec([1.0], [3.0])  # All positive
        bounds = compute_output_range(net, box)
        np.testing.assert_allclose(bounds.lower, [1.0], atol=1e-6)
        np.testing.assert_allclose(bounds.upper, [3.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Edge Cases and Integration
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_point_input(self):
        """When input is a single point, output should be a single point."""
        net = NeuralNetwork()
        net.add_layer(np.array([[2.0]]), np.array([1.0]), Activation.RELU)
        x = np.array([3.0])
        box = box_spec(x, x)
        for method in ["ibp", "deeppoly", "zonotope"]:
            bounds = compute_output_range(net, box, method=method)
            expected = net.forward(x)
            np.testing.assert_allclose(bounds.lower, expected, atol=1e-8)
            np.testing.assert_allclose(bounds.upper, expected, atol=1e-8)

    def test_empty_network(self):
        net = NeuralNetwork()
        assert net.input_dim == 0
        assert net.output_dim == 0
        assert net.depth == 0

    def test_large_network_soundness(self):
        """Soundness on a larger network."""
        rng = np.random.default_rng(80)
        net = build_simple_relu_net([10, 20, 15, 10, 5], rng)
        box = box_spec(-np.ones(10) * 0.1, np.ones(10) * 0.1)
        bounds = ibp_verify(net, box)
        for _ in range(500):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-8)
            assert np.all(y <= bounds.upper + 1e-8)

    def test_mixed_activation_soundness(self):
        """Network with mixed activations."""
        rng = np.random.default_rng(81)
        net = NeuralNetwork()
        net.add_layer(rng.standard_normal((4, 3)), np.zeros(4), Activation.RELU)
        net.add_layer(rng.standard_normal((3, 4)), np.zeros(3), Activation.SIGMOID)
        net.add_layer(rng.standard_normal((2, 3)), np.zeros(2), Activation.TANH)
        net.add_layer(rng.standard_normal((1, 2)), np.zeros(1), Activation.LINEAR)
        box = box_spec([-1, -1, -1], [1, 1, 1])
        for method in ["ibp", "deeppoly", "zonotope"]:
            bounds = compute_output_range(net, box, method=method)
            for _ in range(300):
                x = rng.uniform(box.lower, box.upper)
                y = net.forward(x)
                assert np.all(y >= bounds.lower - 1e-5), f"{method} lower violated"
                assert np.all(y <= bounds.upper + 1e-5), f"{method} upper violated"

    def test_wide_input_range(self):
        rng = np.random.default_rng(82)
        net = build_simple_relu_net([2, 4, 1], rng)
        box = box_spec([-100, -100], [100, 100])
        bounds = ibp_verify(net, box)
        # Should still be sound
        for _ in range(200):
            x = rng.uniform(box.lower, box.upper)
            y = net.forward(x)
            assert np.all(y >= bounds.lower - 1e-6)
            assert np.all(y <= bounds.upper + 1e-6)

    def test_narrow_input_range(self):
        rng = np.random.default_rng(83)
        net = build_simple_relu_net([2, 4, 1], rng)
        box = box_spec([0.499, 0.499], [0.501, 0.501])
        bounds = deeppoly_verify(net, box)
        # Bounds should be very tight
        width = bounds.upper - bounds.lower
        assert np.all(width < 0.1)


# ---------------------------------------------------------------------------
# Verification Report
# ---------------------------------------------------------------------------

class TestVerificationReport:
    def test_report_fields(self):
        report = verify_output_bounds(
            NeuralNetwork().add_layer(np.eye(1), np.zeros(1), Activation.LINEAR),
            box_spec([0], [1]),
            output_lower=np.array([-1.0]),
        )
        assert report.property_name == "output_bounds"
        assert report.bounds is not None
        assert isinstance(report.message, str)

    def test_robustness_report_with_counterexample(self):
        """When violation is found, report should include counterexample."""
        net = NeuralNetwork()
        net.add_layer(np.array([[10, 0], [0, 10]]), np.zeros(2), Activation.LINEAR)
        x = np.array([0.01, 0.0])
        report = verify_robustness(net, x, epsilon=0.1, true_label=0)
        if report.result == VerificationResult.VIOLATED:
            assert report.counterexample is not None
