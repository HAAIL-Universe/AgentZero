"""Tests for C183: Normalizing Flows."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from normalizing_flows import (
    BaseDistribution, AffineFlow, PlanarFlow, RadialFlow, ActNorm,
    Permutation, RealNVPCoupling, NormalizingFlow, FlowTrainer,
    InvertibleLinear, BatchNormFlow, RQSplineFlow,
    FlowImportanceSampler, build_planar_flow, build_realnvp_flow,
    build_mixed_flow, check_invertibility, check_log_det_numerical,
    estimate_kl_divergence, compute_effective_sample_size,
    _simple_net, _forward_net,
)


# ============================================================
# Base Distribution
# ============================================================

class TestBaseDistribution:
    def test_sample_shape(self):
        base = BaseDistribution(3)
        rng = np.random.RandomState(42)
        samples = base.sample(100, rng)
        assert samples.shape == (100, 3)

    def test_log_prob_shape(self):
        base = BaseDistribution(4)
        z = np.random.randn(50, 4)
        lp = base.log_prob(z)
        assert lp.shape == (50,)

    def test_log_prob_values(self):
        base = BaseDistribution(2)
        z = np.zeros((1, 2))
        lp = base.log_prob(z)
        expected = -0.5 * 2 * np.log(2 * np.pi)
        assert abs(lp[0] - expected) < 1e-10

    def test_log_prob_decreases_away_from_origin(self):
        base = BaseDistribution(2)
        z_near = np.array([[0.1, 0.1]])
        z_far = np.array([[3.0, 3.0]])
        assert base.log_prob(z_near)[0] > base.log_prob(z_far)[0]

    def test_sample_statistics(self):
        base = BaseDistribution(5)
        rng = np.random.RandomState(42)
        samples = base.sample(10000, rng)
        assert np.abs(np.mean(samples)) < 0.1
        assert np.abs(np.std(samples) - 1.0) < 0.1


# ============================================================
# Affine Flow
# ============================================================

class TestAffineFlow:
    def test_identity(self):
        layer = AffineFlow(3)
        z = np.random.randn(10, 3)
        x, ld = layer.forward(z)
        np.testing.assert_allclose(x, z, atol=1e-10)
        np.testing.assert_allclose(ld, 0.0, atol=1e-10)

    def test_forward_inverse(self):
        layer = AffineFlow(4)
        layer.log_scale = np.array([0.5, -0.3, 0.1, 0.8])
        layer.shift = np.array([1.0, -2.0, 0.5, 0.0])
        z = np.random.randn(20, 4)
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-10)
        np.testing.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-10)

    def test_log_det(self):
        layer = AffineFlow(3)
        layer.log_scale = np.array([1.0, 2.0, 3.0])
        z = np.random.randn(5, 3)
        _, ld = layer.forward(z)
        assert np.allclose(ld, 6.0)

    def test_parameters(self):
        layer = AffineFlow(3)
        params = layer.parameters()
        assert len(params) == 2

    def test_set_parameters(self):
        layer = AffineFlow(2)
        new_params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        layer.set_parameters(new_params)
        np.testing.assert_allclose(layer.log_scale, [1.0, 2.0])
        np.testing.assert_allclose(layer.shift, [3.0, 4.0])

    def test_numerical_log_det(self):
        layer = AffineFlow(3)
        layer.log_scale = np.array([0.5, -0.3, 0.7])
        layer.shift = np.array([1.0, -1.0, 0.5])
        z = np.random.RandomState(42).randn(5, 3)
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-5


# ============================================================
# Planar Flow
# ============================================================

class TestPlanarFlow:
    def test_output_shape(self):
        rng = np.random.RandomState(42)
        layer = PlanarFlow(3, rng)
        z = rng.randn(10, 3)
        x, ld = layer.forward(z)
        assert x.shape == (10, 3)
        assert ld.shape == (10,)

    def test_invertibility(self):
        rng = np.random.RandomState(42)
        layer = PlanarFlow(2, rng)
        z = rng.randn(20, 2) * 0.5
        x, _ = layer.forward(z)
        z_rec, _ = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-5)

    def test_u_hat_constraint(self):
        rng = np.random.RandomState(42)
        layer = PlanarFlow(3, rng)
        layer.w = np.array([1.0, 0.0, 0.0])
        layer.u = np.array([-5.0, 0.0, 0.0])  # violates w^T u >= -1
        u_hat = layer._get_u_hat()
        assert np.dot(layer.w, u_hat) >= -1.0 - 1e-10

    def test_log_det_numerical(self):
        rng = np.random.RandomState(42)
        layer = PlanarFlow(3, rng)
        z = rng.randn(5, 3) * 0.5
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-4

    def test_parameters(self):
        rng = np.random.RandomState(42)
        layer = PlanarFlow(3, rng)
        params = layer.parameters()
        assert len(params) == 3

    def test_set_parameters(self):
        rng = np.random.RandomState(42)
        layer = PlanarFlow(2, rng)
        old_params = layer.parameters()
        new_params = [p * 2 for p in old_params]
        layer.set_parameters(new_params)
        for p, n in zip(layer.parameters(), new_params):
            np.testing.assert_allclose(p, n, atol=1e-10)


# ============================================================
# Radial Flow
# ============================================================

class TestRadialFlow:
    def test_output_shape(self):
        rng = np.random.RandomState(42)
        layer = RadialFlow(3, rng)
        z = rng.randn(10, 3)
        x, ld = layer.forward(z)
        assert x.shape == (10, 3)
        assert ld.shape == (10,)

    def test_invertibility(self):
        rng = np.random.RandomState(42)
        layer = RadialFlow(2, rng)
        z = rng.randn(20, 2) * 0.5
        x, _ = layer.forward(z)
        z_rec, _ = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-4)

    def test_log_det_numerical(self):
        rng = np.random.RandomState(42)
        layer = RadialFlow(3, rng)
        z = rng.randn(5, 3) * 0.5
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-3

    def test_parameters(self):
        rng = np.random.RandomState(42)
        layer = RadialFlow(3, rng)
        params = layer.parameters()
        assert len(params) == 3


# ============================================================
# ActNorm
# ============================================================

class TestActNorm:
    def test_initialization(self):
        layer = ActNorm(3)
        assert not layer.initialized
        z = np.random.RandomState(42).randn(100, 3) * 3 + 5
        x, _ = layer.forward(z)
        assert layer.initialized
        # After init, output should be roughly zero mean, unit var
        assert np.abs(np.mean(x)) < 0.5
        assert np.abs(np.std(x) - 1.0) < 0.5

    def test_forward_inverse(self):
        layer = ActNorm(4)
        z = np.random.RandomState(42).randn(50, 4) * 2 + 1
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-10)
        np.testing.assert_allclose(ld_fwd + ld_inv, 0.0, atol=1e-10)

    def test_log_det(self):
        layer = ActNorm(3)
        layer.log_scale = np.array([1.0, 0.5, -0.5])
        layer.initialized = True
        z = np.random.randn(10, 3)
        _, ld = layer.forward(z)
        assert np.allclose(ld, 1.0)  # sum of log_scale

    def test_parameters(self):
        layer = ActNorm(3)
        params = layer.parameters()
        assert len(params) == 2

    def test_set_parameters_sets_initialized(self):
        layer = ActNorm(3)
        assert not layer.initialized
        layer.set_parameters([np.ones(3), np.zeros(3)])
        assert layer.initialized


# ============================================================
# Permutation
# ============================================================

class TestPermutation:
    def test_forward_inverse(self):
        rng = np.random.RandomState(42)
        layer = Permutation(5, rng)
        z = rng.randn(10, 5)
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-10)

    def test_volume_preserving(self):
        rng = np.random.RandomState(42)
        layer = Permutation(4, rng)
        z = rng.randn(10, 4)
        _, ld = layer.forward(z)
        np.testing.assert_allclose(ld, 0.0, atol=1e-10)

    def test_actually_permutes(self):
        rng = np.random.RandomState(42)
        layer = Permutation(4, rng)
        z = np.array([[1.0, 2.0, 3.0, 4.0]])
        x, _ = layer.forward(z)
        # Should be a permutation of [1,2,3,4]
        assert set(x[0].tolist()) == {1.0, 2.0, 3.0, 4.0}


# ============================================================
# RealNVP Coupling
# ============================================================

class TestRealNVPCoupling:
    def test_output_shape(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, rng=rng)
        z = rng.randn(10, 4)
        x, ld = layer.forward(z)
        assert x.shape == (10, 4)
        assert ld.shape == (10,)

    def test_forward_inverse(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, rng=rng)
        z = rng.randn(20, 4)
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-10)

    def test_log_det_numerical(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, rng=rng)
        z = rng.randn(5, 4)
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-4

    def test_masked_dims_unchanged(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, mask_even=True, rng=rng)
        z = rng.randn(10, 4)
        x, _ = layer.forward(z)
        # Even-indexed dims (0, 2) should be unchanged
        np.testing.assert_allclose(x[:, 0], z[:, 0])
        np.testing.assert_allclose(x[:, 2], z[:, 2])

    def test_alternating_masks(self):
        rng = np.random.RandomState(42)
        layer_even = RealNVPCoupling(4, 16, mask_even=True, rng=rng)
        layer_odd = RealNVPCoupling(4, 16, mask_even=False, rng=rng)
        assert np.sum(layer_even.mask) == 2
        assert np.sum(layer_odd.mask) == 2
        assert not np.allclose(layer_even.mask, layer_odd.mask)

    def test_parameters(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, rng=rng)
        params = layer.parameters()
        assert len(params) == 8  # 4 per net * 2 nets

    def test_set_parameters(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, rng=rng)
        params = layer.parameters()
        new_params = [p * 0.5 for p in params]
        layer.set_parameters(new_params)
        for got, expected in zip(layer.parameters(), new_params):
            np.testing.assert_allclose(got, expected, atol=1e-10)


# ============================================================
# Simple Net
# ============================================================

class TestSimpleNet:
    def test_net_creation(self):
        net = _simple_net(3, 8, 2)
        assert net['W1'].shape == (3, 8)
        assert net['b1'].shape == (8,)
        assert net['W2'].shape == (8, 2)
        assert net['b2'].shape == (2,)

    def test_forward_net(self):
        rng = np.random.RandomState(42)
        net = _simple_net(3, 8, 2, rng)
        x = rng.randn(5, 3)
        out = _forward_net(net, x)
        assert out.shape == (5, 2)


# ============================================================
# InvertibleLinear
# ============================================================

class TestInvertibleLinear:
    def test_forward_inverse(self):
        rng = np.random.RandomState(42)
        layer = InvertibleLinear(3, rng)
        z = rng.randn(20, 3)
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-8)

    def test_log_det_numerical(self):
        rng = np.random.RandomState(42)
        layer = InvertibleLinear(3, rng)
        z = rng.randn(5, 3)
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-5

    def test_orthogonal_init(self):
        rng = np.random.RandomState(42)
        layer = InvertibleLinear(4, rng)
        # Should be approximately orthogonal
        WtW = layer.W.T @ layer.W
        np.testing.assert_allclose(WtW, np.eye(4), atol=1e-10)

    def test_parameters(self):
        rng = np.random.RandomState(42)
        layer = InvertibleLinear(3, rng)
        params = layer.parameters()
        assert len(params) == 1
        assert params[0].shape == (3, 3)


# ============================================================
# BatchNormFlow
# ============================================================

class TestBatchNormFlow:
    def test_forward_shape(self):
        layer = BatchNormFlow(3)
        z = np.random.randn(20, 3)
        x, ld = layer.forward(z)
        assert x.shape == (20, 3)
        assert ld.shape == (20,)

    def test_forward_inverse_eval(self):
        layer = BatchNormFlow(3)
        z = np.random.RandomState(42).randn(50, 3) * 2 + 1
        # Training forward to set running stats
        layer.forward(z)
        layer.training = False
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-8)

    def test_running_stats_update(self):
        layer = BatchNormFlow(3)
        z = np.random.RandomState(42).randn(100, 3) * 3 + 5
        layer.forward(z)
        assert not np.allclose(layer.running_mean, 0.0)

    def test_parameters(self):
        layer = BatchNormFlow(3)
        params = layer.parameters()
        assert len(params) == 2


# ============================================================
# RQ Spline Flow
# ============================================================

class TestRQSplineFlow:
    def test_output_shape(self):
        rng = np.random.RandomState(42)
        layer = RQSplineFlow(2, n_bins=4, rng=rng)
        z = rng.randn(10, 2) * 0.5
        x, ld = layer.forward(z)
        assert x.shape == (10, 2)
        assert ld.shape == (10,)

    def test_identity_at_init(self):
        # Near identity when params are small
        rng = np.random.RandomState(42)
        layer = RQSplineFlow(2, n_bins=4, rng=rng)
        layer.raw_widths = np.zeros((2, 4))
        layer.raw_heights = np.zeros((2, 4))
        layer.raw_derivatives = np.zeros((2, 5))
        z = np.array([[0.5, -0.5]])
        x, _ = layer.forward(z)
        # Should be close to identity for inputs within bounds
        np.testing.assert_allclose(x, z, atol=0.5)

    def test_invertibility(self):
        rng = np.random.RandomState(42)
        layer = RQSplineFlow(2, n_bins=4, rng=rng)
        z = rng.randn(10, 2) * 0.5  # stay within bounds
        x, ld_fwd = layer.forward(z)
        z_rec, ld_inv = layer.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-4)

    def test_outside_bounds_identity(self):
        rng = np.random.RandomState(42)
        layer = RQSplineFlow(2, n_bins=4, bound=3.0, rng=rng)
        z = np.array([[5.0, -5.0]])  # outside bounds
        x, ld = layer.forward(z)
        np.testing.assert_allclose(x, z, atol=1e-10)
        np.testing.assert_allclose(ld, 0.0, atol=1e-10)

    def test_parameters(self):
        rng = np.random.RandomState(42)
        layer = RQSplineFlow(3, n_bins=4, rng=rng)
        params = layer.parameters()
        assert len(params) == 3


# ============================================================
# NormalizingFlow (Composition)
# ============================================================

class TestNormalizingFlow:
    def test_forward_shape(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(3, 2, rng)
        z = rng.randn(10, 3)
        x, ld = flow.forward(z)
        assert x.shape == (10, 3)
        assert ld.shape == (10,)

    def test_inverse_shape(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(3, 2, rng)
        x = rng.randn(10, 3)
        z, ld = flow.inverse(x)
        assert z.shape == (10, 3)
        assert ld.shape == (10,)

    def test_sample(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        samples = flow.sample(100, rng)
        assert samples.shape == (100, 2)

    def test_log_prob_shape(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        x = rng.randn(10, 2)
        lp = flow.log_prob(x)
        assert lp.shape == (10,)

    def test_log_prob_finite(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        x = rng.randn(10, 2)
        lp = flow.log_prob(x)
        assert np.all(np.isfinite(lp))

    def test_affine_flow_composition(self):
        base = BaseDistribution(2)
        layer1 = AffineFlow(2)
        layer1.log_scale = np.array([0.5, 0.5])
        layer1.shift = np.array([1.0, 1.0])
        layer2 = AffineFlow(2)
        layer2.log_scale = np.array([-0.5, -0.5])
        layer2.shift = np.array([-1.0, -1.0])
        flow = NormalizingFlow(base, [layer1, layer2])
        # These two layers approximately cancel
        z = np.random.RandomState(42).randn(20, 2)
        x, ld = flow.forward(z)
        # Should be close to identity (not exact due to shift ordering)
        # log_det should be 0 (scales cancel)
        np.testing.assert_allclose(ld, 0.0, atol=1e-10)

    def test_parameters_collection(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 3, rng)
        params = flow.parameters()
        # 3 planar layers, each with 3 params (w, u, b_array)
        assert len(params) == 9

    def test_set_parameters(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        params = flow.parameters()
        new_params = [p * 0.0 for p in params]
        flow.set_parameters(new_params)
        for p in flow.parameters():
            np.testing.assert_allclose(p, 0.0, atol=1e-10)


# ============================================================
# RealNVP Flow
# ============================================================

class TestRealNVPFlow:
    def test_build(self):
        rng = np.random.RandomState(42)
        flow = build_realnvp_flow(4, 2, 16, rng)
        assert len(flow.layers) == 4  # 2 * (ActNorm + Coupling)

    def test_invertibility(self):
        rng = np.random.RandomState(42)
        flow = build_realnvp_flow(4, 2, 16, rng)
        result = check_invertibility(flow, 50, rng)
        assert result['reconstruction_error'] < 1e-8

    def test_sample_and_log_prob(self):
        rng = np.random.RandomState(42)
        flow = build_realnvp_flow(4, 2, 16, rng)
        samples = flow.sample(50, rng)
        lp = flow.log_prob(samples)
        assert samples.shape == (50, 4)
        assert lp.shape == (50,)
        assert np.all(np.isfinite(lp))


# ============================================================
# Mixed Flow
# ============================================================

class TestMixedFlow:
    def test_build(self):
        rng = np.random.RandomState(42)
        flow = build_mixed_flow(4, rng)
        assert len(flow.layers) == 5

    def test_invertibility(self):
        rng = np.random.RandomState(42)
        flow = build_mixed_flow(4, rng)
        result = check_invertibility(flow, 50, rng)
        assert result['reconstruction_error'] < 1e-6

    def test_sample(self):
        rng = np.random.RandomState(42)
        flow = build_mixed_flow(4, rng)
        samples = flow.sample(100, rng)
        assert samples.shape == (100, 4)


# ============================================================
# Training
# ============================================================

class TestFlowTrainer:
    def test_train_step(self):
        rng = np.random.RandomState(42)
        flow = NormalizingFlow(BaseDistribution(2), [AffineFlow(2)])
        trainer = FlowTrainer(flow, lr=0.01)
        data = rng.randn(20, 2) * 0.5
        loss = trainer.train_step(data)
        assert np.isfinite(loss)

    def test_train_reduces_loss(self):
        rng = np.random.RandomState(42)
        flow = NormalizingFlow(BaseDistribution(2), [AffineFlow(2)])
        trainer = FlowTrainer(flow, lr=0.01)
        data = rng.randn(30, 2) * 0.5 + 1.0  # shifted from origin
        losses = trainer.train(data, epochs=5, batch_size=30, rng=rng)
        assert len(losses) == 5
        # Loss should generally decrease
        assert losses[-1] <= losses[0] + 1.0  # allow some noise

    def test_train_returns_losses(self):
        rng = np.random.RandomState(42)
        flow = NormalizingFlow(BaseDistribution(2), [AffineFlow(2)])
        trainer = FlowTrainer(flow, lr=0.01)
        data = rng.randn(20, 2)
        losses = trainer.train(data, epochs=3, rng=rng)
        assert len(losses) == 3
        assert all(np.isfinite(l) for l in losses)


# ============================================================
# Importance Sampling
# ============================================================

class TestFlowImportanceSampler:
    def test_estimate_shape(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        sampler = FlowImportanceSampler(flow)

        def target_log_prob(x):
            return -0.5 * np.sum(x ** 2, axis=-1)

        def f(x):
            return x[:, 0]

        result = sampler.estimate_expectation(target_log_prob, f, 100, rng)
        assert 'estimate' in result
        assert 'ess' in result
        assert 'log_normalizer_estimate' in result

    def test_ess_positive(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        sampler = FlowImportanceSampler(flow)

        def target_log_prob(x):
            return -0.5 * np.sum(x ** 2, axis=-1)

        def f(x):
            return np.ones(x.shape[0])

        result = sampler.estimate_expectation(target_log_prob, f, 200, rng)
        assert result['ess'] > 0


# ============================================================
# Diagnostics
# ============================================================

class TestDiagnostics:
    def test_check_invertibility_affine(self):
        base = BaseDistribution(3)
        layer = AffineFlow(3)
        layer.log_scale = np.array([0.5, -0.3, 0.1])
        flow = NormalizingFlow(base, [layer])
        rng = np.random.RandomState(42)
        result = check_invertibility(flow, 50, rng)
        assert result['reconstruction_error'] < 1e-10
        assert result['log_det_consistency'] < 1e-10

    def test_check_invertibility_realnvp(self):
        rng = np.random.RandomState(42)
        flow = build_realnvp_flow(4, 2, 16, rng)
        result = check_invertibility(flow, 50, rng)
        assert result['reconstruction_error'] < 1e-8

    def test_check_log_det_numerical_affine(self):
        layer = AffineFlow(3)
        layer.log_scale = np.array([0.5, -0.3, 0.7])
        z = np.random.RandomState(42).randn(5, 3)
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-5

    def test_check_log_det_numerical_realnvp(self):
        rng = np.random.RandomState(42)
        layer = RealNVPCoupling(4, 16, rng=rng)
        z = rng.randn(5, 4)
        result = check_log_det_numerical(layer, z)
        assert result['error'] < 1e-4


# ============================================================
# Utilities
# ============================================================

class TestUtilities:
    def test_estimate_kl(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        data = rng.randn(50, 2)
        kl = estimate_kl_divergence(flow, data)
        assert np.isfinite(kl)

    def test_effective_sample_size(self):
        # Uniform weights -> ESS = n
        log_w = np.zeros(100)
        ess = compute_effective_sample_size(log_w)
        np.testing.assert_allclose(ess, 100.0, atol=0.1)

    def test_effective_sample_size_degenerate(self):
        # One dominant weight -> ESS ~ 1
        log_w = np.array([100.0, 0.0, 0.0, 0.0])
        ess = compute_effective_sample_size(log_w)
        assert ess < 2.0

    def test_effective_sample_size_moderate(self):
        log_w = np.array([0.0, 0.0, -1.0, -2.0])
        ess = compute_effective_sample_size(log_w)
        assert 1.0 < ess < 4.0


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_sample(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        x = rng.randn(1, 2)
        lp = flow.log_prob(x)
        assert lp.shape == (1,)
        assert np.isfinite(lp[0])

    def test_high_dim(self):
        rng = np.random.RandomState(42)
        base = BaseDistribution(20)
        layer = AffineFlow(20)
        flow = NormalizingFlow(base, [layer])
        samples = flow.sample(10, rng)
        assert samples.shape == (10, 20)
        lp = flow.log_prob(samples)
        assert lp.shape == (10,)

    def test_empty_layers(self):
        base = BaseDistribution(3)
        flow = NormalizingFlow(base, [])
        rng = np.random.RandomState(42)
        z = base.sample(10, rng)
        x, ld = flow.forward(z)
        np.testing.assert_allclose(x, z)
        np.testing.assert_allclose(ld, 0.0)

    def test_many_layers(self):
        rng = np.random.RandomState(42)
        base = BaseDistribution(2)
        layers = [AffineFlow(2) for _ in range(10)]
        flow = NormalizingFlow(base, layers)
        samples = flow.sample(10, rng)
        assert samples.shape == (10, 2)

    def test_large_batch(self):
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        x = rng.randn(1000, 2)
        lp = flow.log_prob(x)
        assert lp.shape == (1000,)
        assert np.all(np.isfinite(lp))

    def test_flow_with_invertible_linear(self):
        rng = np.random.RandomState(42)
        base = BaseDistribution(3)
        layers = [
            InvertibleLinear(3, rng),
            AffineFlow(3),
            InvertibleLinear(3, rng),
        ]
        flow = NormalizingFlow(base, layers)
        result = check_invertibility(flow, 50, rng)
        assert result['reconstruction_error'] < 1e-8

    def test_flow_with_batch_norm(self):
        rng = np.random.RandomState(42)
        base = BaseDistribution(3)
        layers = [
            BatchNormFlow(3),
            AffineFlow(3),
        ]
        flow = NormalizingFlow(base, layers)
        # Do a forward pass to initialize
        z = base.sample(50, rng)
        x, ld = flow.forward(z)
        assert x.shape == (50, 3)

    def test_dim_1(self):
        base = BaseDistribution(1)
        layer = AffineFlow(1)
        layer.log_scale = np.array([1.0])
        layer.shift = np.array([2.0])
        flow = NormalizingFlow(base, [layer])
        rng = np.random.RandomState(42)
        z = base.sample(10, rng)
        x, ld = flow.forward(z)
        z_rec, _ = flow.inverse(x)
        np.testing.assert_allclose(z, z_rec, atol=1e-10)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_planar_flow_density_estimation(self):
        """Test that a planar flow can compute log-probs on data."""
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 4, rng)
        data = rng.randn(100, 2)
        lp = flow.log_prob(data)
        assert lp.shape == (100,)
        assert np.all(np.isfinite(lp))

    def test_realnvp_training(self):
        """Test that RealNVP can train without errors."""
        rng = np.random.RandomState(42)
        base = BaseDistribution(2)
        layers = [
            RealNVPCoupling(2, 8, mask_even=True, rng=rng),
            RealNVPCoupling(2, 8, mask_even=False, rng=rng),
        ]
        flow = NormalizingFlow(base, layers)
        trainer = FlowTrainer(flow, lr=0.001)
        data = rng.randn(20, 2) * 0.5 + np.array([1.0, -1.0])
        losses = trainer.train(data, epochs=2, batch_size=20, rng=rng)
        assert len(losses) == 2
        assert all(np.isfinite(l) for l in losses)

    def test_mixed_flow_sample_and_eval(self):
        """Test mixed flow can sample and evaluate."""
        rng = np.random.RandomState(42)
        flow = build_mixed_flow(4, rng)
        samples = flow.sample(50, rng)
        lp = flow.log_prob(samples)
        assert samples.shape == (50, 4)
        assert lp.shape == (50,)
        assert np.all(np.isfinite(lp))

    def test_end_to_end_pipeline(self):
        """Full pipeline: build, train, sample, evaluate."""
        rng = np.random.RandomState(42)
        # Target: shifted Gaussian
        data = rng.randn(50, 2) * 0.5 + np.array([2.0, -1.0])
        # Build flow
        flow = NormalizingFlow(BaseDistribution(2), [AffineFlow(2)])
        # Train
        trainer = FlowTrainer(flow, lr=0.01)
        losses = trainer.train(data, epochs=3, batch_size=50, rng=rng)
        # Sample
        samples = flow.sample(100, rng)
        # Evaluate
        lp = flow.log_prob(samples)
        assert samples.shape == (100, 2)
        assert np.all(np.isfinite(lp))

    def test_importance_sampling_integration(self):
        """Test importance sampling with a trained flow."""
        rng = np.random.RandomState(42)
        flow = build_planar_flow(2, 2, rng)
        sampler = FlowImportanceSampler(flow)

        def target(x):
            return -0.5 * np.sum(x ** 2, axis=-1)

        def f(x):
            return np.ones(x.shape[0])

        result = sampler.estimate_expectation(target, f, 500, rng)
        assert np.isfinite(result['estimate'])
        assert result['ess'] > 1

    def test_serialization_roundtrip(self):
        """Test parameter save/restore."""
        rng = np.random.RandomState(42)
        flow = build_realnvp_flow(4, 2, 16, rng)
        params = [p.copy() for p in flow.parameters()]
        x = rng.randn(10, 4)
        lp_before = flow.log_prob(x)

        # Perturb
        flow.set_parameters([p + 1.0 for p in params])
        lp_perturbed = flow.log_prob(x)
        assert not np.allclose(lp_before, lp_perturbed)

        # Restore
        flow.set_parameters(params)
        lp_after = flow.log_prob(x)
        np.testing.assert_allclose(lp_before, lp_after, atol=1e-10)

    def test_flow_with_spline(self):
        """Test flow with RQ spline layer."""
        rng = np.random.RandomState(42)
        base = BaseDistribution(2)
        layers = [
            RQSplineFlow(2, n_bins=4, rng=rng),
            AffineFlow(2),
        ]
        flow = NormalizingFlow(base, layers)
        samples = flow.sample(20, rng)
        lp = flow.log_prob(samples)
        assert samples.shape == (20, 2)
        assert np.all(np.isfinite(lp))

    def test_deep_flow_stability(self):
        """Test a deeper flow chain for numerical stability."""
        rng = np.random.RandomState(42)
        base = BaseDistribution(4)
        layers = []
        for i in range(6):
            layers.append(ActNorm(4))
            layers.append(RealNVPCoupling(4, 16, mask_even=(i % 2 == 0), rng=rng))
        flow = NormalizingFlow(base, layers)
        samples = flow.sample(50, rng)
        lp = flow.log_prob(samples)
        assert np.all(np.isfinite(lp))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
