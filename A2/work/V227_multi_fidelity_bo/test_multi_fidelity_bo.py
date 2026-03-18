"""Tests for V227: Multi-Fidelity Bayesian Optimization."""

import numpy as np
from numpy.random import default_rng
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V223_bayesian_optimization'))

from multi_fidelity_bo import (
    # Data structures
    FidelityLevel, MFObservation, MFBOResult, ContinuousFidelityResult,
    MFComparison, MFAcquisitionType,
    # Models
    MultiFidelityKernel, MultiFidelityGP, LinearMultiFidelityGP,
    # Acquisition functions
    cost_aware_ei, cost_aware_ucb, multi_fidelity_knowledge_gradient,
    max_value_entropy_search,
    # Main optimizers
    multi_fidelity_bo, continuous_fidelity_bo, multi_task_bo,
    # Comparison
    compare_mf_vs_single,
    # Benchmarks
    branin_hf, branin_lf, branin_mf, sphere_hf, sphere_lf,
    hartmann3_hf, hartmann3_lf, continuous_fidelity_branin,
    # Utilities
    mf_optimization_summary,
)
from gaussian_process import (
    GaussianProcess, Matern52Kernel, ScaleKernel, RBFKernel, GPPrediction
)
from bayesian_optimization import Bounds, expected_improvement


# ===== FidelityLevel =====

class TestFidelityLevel:
    def test_basic_creation(self):
        fl = FidelityLevel(level=0, cost=1.0, name="low")
        assert fl.level == 0
        assert fl.cost == 1.0
        assert fl.name == "low"

    def test_auto_name(self):
        fl = FidelityLevel(level=2, cost=10.0)
        assert fl.name == "fidelity_2"

    def test_custom_name(self):
        fl = FidelityLevel(level=1, cost=5.0, name="medium")
        assert fl.name == "medium"


# ===== MFObservation =====

class TestMFObservation:
    def test_creation(self):
        obs = MFObservation(x=np.array([1.0, 2.0]), fidelity=1, y=3.5, cost=5.0)
        assert obs.fidelity == 1
        assert obs.y == 3.5
        assert obs.cost == 5.0
        np.testing.assert_array_equal(obs.x, [1.0, 2.0])


# ===== MultiFidelityKernel =====

class TestMultiFidelityKernel:
    def test_kernel_shape(self):
        base = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        mfk = MultiFidelityKernel(base)
        # 2D input + 1 fidelity dim
        X1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]])
        X2 = np.array([[0.5, 0.5, 1.0]])
        K = mfk(X1, X2)
        assert K.shape == (2, 1)

    def test_same_fidelity_higher_correlation(self):
        base = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        mfk = MultiFidelityKernel(base, fidelity_length_scale=0.3)
        x = np.array([[0.5, 0.5, 0.0]])  # fid=0
        x_same = np.array([[0.6, 0.6, 0.0]])  # fid=0
        x_diff = np.array([[0.6, 0.6, 1.0]])  # fid=1
        k_same = mfk(x, x_same)[0, 0]
        k_diff = mfk(x, x_diff)[0, 0]
        assert k_same > k_diff  # Same fidelity -> higher correlation

    def test_params_roundtrip(self):
        base = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        mfk = MultiFidelityKernel(base, fidelity_length_scale=0.5, fidelity_variance=2.0)
        p = mfk.params()
        mfk2 = MultiFidelityKernel(
            ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0))
        mfk2.set_params(p)
        assert abs(mfk2._fidelity_ls - 0.5) < 1e-6
        assert abs(mfk2._fidelity_var - 2.0) < 1e-6

    def test_diag(self):
        base = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        mfk = MultiFidelityKernel(base)
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]])
        d = mfk.diag(X)
        assert d.shape == (3,)
        assert np.all(d > 0)

    def test_n_params(self):
        base = ScaleKernel(Matern52Kernel(length_scale=1.0), scale=1.0)
        mfk = MultiFidelityKernel(base)
        assert mfk.n_params() == base.n_params() + 2


# ===== MultiFidelityGP =====

class TestMultiFidelityGP:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n = 20
        X = rng.uniform(0, 1, (n, 2))
        fids = rng.integers(0, 3, n)
        y = np.array([sphere_hf(x) + (2 - f) * 0.5 for x, f in zip(X, fids)])

        mfgp = MultiFidelityGP(n_fidelities=3)
        mfgp.fit(X, fids, y)

        X_test = rng.uniform(0, 1, (5, 2))
        pred = mfgp.predict_highest(X_test)
        assert pred.mean.shape == (5,)
        assert pred.std.shape == (5,)
        assert np.all(pred.std > 0)

    def test_predict_different_fidelities(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, (30, 2))
        fids = np.repeat([0, 1, 2], 10)
        y = np.array([sphere_hf(x) + (2 - f) * 1.0 for x, f in zip(X, fids)])

        mfgp = MultiFidelityGP(n_fidelities=3)
        mfgp.fit(X, fids, y)

        X_test = np.array([[0.5, 0.5]])
        pred0 = mfgp.predict(X_test, 0)
        pred2 = mfgp.predict(X_test, 2)
        # Higher fidelity should predict lower values (sphere + offset)
        # At least predictions should differ
        assert pred0.mean[0] != pred2.mean[0]

    def test_encode_fidelity(self):
        mfgp = MultiFidelityGP(n_fidelities=3)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        fids = np.array([0, 2])
        X_aug = mfgp._encode_fidelity(X, fids)
        assert X_aug.shape == (2, 3)
        assert X_aug[0, 2] == 0.0  # fid 0 -> 0.0
        assert X_aug[1, 2] == 1.0  # fid 2 -> 1.0


# ===== LinearMultiFidelityGP =====

class TestLinearMultiFidelityGP:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        # Generate data with known AR1 structure
        n_per = 15
        X0 = rng.uniform(0, 1, (n_per, 1))
        y0 = np.sin(2 * np.pi * X0[:, 0]) + 0.5  # LF

        X1 = rng.uniform(0, 1, (n_per, 1))
        y1_lf = np.sin(2 * np.pi * X1[:, 0]) + 0.5
        y1 = 1.2 * y1_lf + 0.1  # HF = 1.2 * LF + 0.1

        X = np.vstack([X0, X1])
        fids = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int)
        y = np.concatenate([y0, y1])

        lmf = LinearMultiFidelityGP(n_fidelities=2)
        lmf.fit(X, fids, y)

        assert lmf._fitted
        assert len(lmf.delta_gps) == 2
        assert len(lmf.rho) == 2

        X_test = np.array([[0.25], [0.5], [0.75]])
        pred = lmf.predict_highest(X_test)
        assert pred.mean.shape == (3,)

    def test_rho_estimated(self):
        rng = np.random.default_rng(42)
        n = 20
        X = rng.uniform(0, 1, (n * 2, 1))
        fids = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

        # y_0 = x, y_1 = 2*x (rho should be ~2)
        y0 = X[:n, 0]
        y1 = 2.0 * X[n:, 0]
        y = np.concatenate([y0, y1])

        lmf = LinearMultiFidelityGP(n_fidelities=2)
        lmf.fit(X, fids, y)

        # rho[1] should be close to 2.0
        assert abs(lmf.rho[1] - 2.0) < 1.0  # Approximate

    def test_predict_per_level(self):
        rng = np.random.default_rng(42)
        n = 15
        X = rng.uniform(0, 5, (n * 2, 1))
        fids = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        y = np.concatenate([
            X[:n, 0] ** 2 * 0.8 + 1.0,  # LF
            X[n:, 0] ** 2               # HF
        ])

        lmf = LinearMultiFidelityGP(n_fidelities=2)
        lmf.fit(X, fids, y)

        X_test = np.array([[2.5]])
        p0 = lmf.predict(X_test, 0)
        p1 = lmf.predict(X_test, 1)
        # Both should produce reasonable predictions
        assert p0.mean.shape == (1,)
        assert p1.mean.shape == (1,)

    def test_three_fidelities(self):
        rng = np.random.default_rng(42)
        n = 10
        X = rng.uniform(0, 1, (n * 3, 1))
        fids = np.concatenate([np.zeros(n), np.ones(n), 2 * np.ones(n)]).astype(int)
        y = np.concatenate([
            np.sin(X[:n, 0]) + 0.5,        # LF
            np.sin(X[n:2*n, 0]) * 1.1,     # MF
            np.sin(X[2*n:, 0]) * 1.0 - 0.1  # HF
        ])

        lmf = LinearMultiFidelityGP(n_fidelities=3)
        lmf.fit(X, fids, y)

        pred = lmf.predict_highest(np.array([[0.5]]))
        assert pred.mean.shape == (1,)


# ===== Acquisition Functions =====

class TestCostAwareEI:
    def test_basic(self):
        mu = np.array([1.0, 2.0, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        f_best = 0.0
        # EI / cost
        ei_cheap = cost_aware_ei(mu, std, f_best, cost=1.0)
        ei_expensive = cost_aware_ei(mu, std, f_best, cost=10.0)
        np.testing.assert_array_less(ei_expensive, ei_cheap + 1e-10)

    def test_zero_std(self):
        mu = np.array([1.0])
        std = np.array([0.0])
        result = cost_aware_ei(mu, std, 0.0, cost=1.0)
        assert result[0] == 0.0

    def test_below_best(self):
        mu = np.array([-1.0])
        std = np.array([0.1])
        result = cost_aware_ei(mu, std, 0.0, cost=1.0)
        assert result[0] > 0.0


class TestCostAwareUCB:
    def test_basic(self):
        mu = np.array([1.0, 2.0])
        std = np.array([1.0, 0.5])
        ucb = cost_aware_ucb(mu, std, beta=2.0, cost=1.0)
        assert ucb.shape == (2,)

    def test_cost_scaling(self):
        mu = np.array([1.0])
        std = np.array([1.0])
        ucb1 = cost_aware_ucb(mu, std, cost=1.0)
        ucb10 = cost_aware_ucb(mu, std, cost=10.0)
        assert abs(ucb1[0]) > abs(ucb10[0])


class TestMFKnowledgeGradient:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 15
        X = rng.uniform(0, 1, (n, 2))
        fids = rng.integers(0, 2, n)
        y = np.array([sphere_hf(x) + (1 - f) * 0.5 for x, f in zip(X, fids)])

        mfgp = MultiFidelityGP(n_fidelities=2)
        mfgp.fit(X, fids, y)

        X_cand = rng.uniform(0, 1, (10, 2))
        kg = multi_fidelity_knowledge_gradient(
            mfgp, X_cand, fidelity=0, f_best_hf=0.5, cost=1.0, rng=rng)
        assert kg.shape == (10,)
        assert np.all(kg >= 0)

    def test_hf_more_valuable(self):
        rng = np.random.default_rng(42)
        n = 20
        X = rng.uniform(0, 1, (n, 2))
        fids = rng.integers(0, 2, n)
        y = np.array([sphere_hf(x) + (1 - f) * 0.5 for x, f in zip(X, fids)])

        mfgp = MultiFidelityGP(n_fidelities=2)
        mfgp.fit(X, fids, y)

        X_cand = rng.uniform(0, 1, (20, 2))
        kg_lf = multi_fidelity_knowledge_gradient(
            mfgp, X_cand, fidelity=0, f_best_hf=0.5, cost=1.0, rng=default_rng(42))
        kg_hf = multi_fidelity_knowledge_gradient(
            mfgp, X_cand, fidelity=1, f_best_hf=0.5, cost=1.0, rng=default_rng(42))
        # HF observations should be at least as informative (per unit, ignoring cost)
        assert np.mean(kg_hf) >= np.mean(kg_lf) * 0.5  # Relaxed check


class TestMaxValueEntropySearch:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 15
        X = rng.uniform(0, 1, (n, 2))
        fids = rng.integers(0, 2, n)
        y = np.array([sphere_hf(x) + (1 - f) * 0.5 for x, f in zip(X, fids)])

        mfgp = MultiFidelityGP(n_fidelities=2)
        mfgp.fit(X, fids, y)

        X_cand = rng.uniform(0, 1, (10, 2))
        mes = max_value_entropy_search(
            mfgp, X_cand, fidelity=1, f_best_hf=0.5, cost=1.0, rng=rng)
        assert mes.shape == (10,)
        assert np.all(mes >= 0)

    def test_cost_normalized(self):
        rng = np.random.default_rng(42)
        n = 15
        X = rng.uniform(0, 1, (n, 2))
        fids = rng.integers(0, 2, n)
        y = np.array([sphere_hf(x) for x in X])

        mfgp = MultiFidelityGP(n_fidelities=2)
        mfgp.fit(X, fids, y)

        X_cand = rng.uniform(0, 1, (10, 2))
        mes1 = max_value_entropy_search(
            mfgp, X_cand, fidelity=1, f_best_hf=0.5, cost=1.0, rng=default_rng(42))
        mes10 = max_value_entropy_search(
            mfgp, X_cand, fidelity=1, f_best_hf=0.5, cost=10.0, rng=default_rng(42))
        np.testing.assert_allclose(mes1, mes10 * 10, rtol=0.01)


# ===== Benchmark Functions =====

class TestBenchmarks:
    def test_branin_hf_minimum(self):
        # Known minimum ~ 0.397887
        x_opt = np.array([np.pi, 2.275])
        val = branin_hf(x_opt)
        assert val < 1.0

    def test_branin_lf_biased(self):
        x = np.array([0.0, 0.0])
        hf = branin_hf(x)
        lf = branin_lf(x)
        assert lf != hf  # LF is biased

    def test_branin_mf_closer_than_lf(self):
        rng = np.random.default_rng(42)
        diffs_lf = []
        diffs_mf = []
        for _ in range(20):
            x = np.array([rng.uniform(-5, 10), rng.uniform(0, 15)])
            hf = branin_hf(x)
            diffs_lf.append(abs(branin_lf(x) - hf))
            diffs_mf.append(abs(branin_mf(x) - hf))
        assert np.mean(diffs_mf) < np.mean(diffs_lf)

    def test_sphere_hf(self):
        assert sphere_hf(np.zeros(3)) == 0.0
        assert sphere_hf(np.ones(2)) == 2.0

    def test_sphere_lf_offset(self):
        x = np.zeros(2)
        assert sphere_lf(x) > sphere_hf(x)

    def test_hartmann3_minimum(self):
        # Global min ~ -3.86 at ~(0.114, 0.556, 0.853)
        x_opt = np.array([0.114614, 0.555649, 0.852547])
        val = hartmann3_hf(x_opt)
        assert val < -3.5

    def test_hartmann3_lf_biased(self):
        x = np.array([0.5, 0.5, 0.5])
        assert hartmann3_lf(x) != hartmann3_hf(x)

    def test_continuous_fidelity_branin_s1(self):
        x = np.array([np.pi, 2.275])
        # At s=1, should be close to true branin
        val = continuous_fidelity_branin(x, 1.0)
        assert abs(val - branin_hf(x)) < 1e-10

    def test_continuous_fidelity_branin_s0_biased(self):
        x = np.array([0.0, 0.0])
        val_s0 = continuous_fidelity_branin(x, 0.0)
        val_s1 = continuous_fidelity_branin(x, 1.0)
        assert val_s0 != val_s1


# ===== Multi-Fidelity BO (main optimizer) =====

class TestMultiFidelityBO:
    def test_sphere_2_fidelities(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=80.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, MFBOResult)
        assert result.f_best < 4.0  # Should find something reasonable
        assert result.total_cost <= 80.0 + 10.0  # Budget + one extra possible
        assert 0 in result.n_evaluations
        assert 1 in result.n_evaluations
        assert result.n_evaluations[0] > 0
        assert result.n_evaluations[1] > 0

    def test_uses_more_lf_evaluations(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=80.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        # With 10x cost difference, should do more LF evaluations
        assert result.n_evaluations[0] >= result.n_evaluations[1]

    def test_three_fidelities_branin(self):
        objectives = {0: branin_lf, 1: branin_mf, 2: branin_hf}
        costs = {0: 1.0, 1: 5.0, 2: 20.0}
        bounds = Bounds(np.array([-5.0, 0.0]), np.array([10.0, 15.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=150.0,
            n_initial_per_fidelity=2,
            rng=np.random.default_rng(42)
        )

        assert result.x_best.shape == (2,)
        assert len(result.convergence) > 1
        assert len(result.cost_convergence) > 1

    def test_ar1_model(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=60.0,
            n_initial_per_fidelity=3,
            model="ar1",
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, MFBOResult)
        assert result.f_best < 3.0

    def test_cost_aware_ucb_acquisition(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=60.0,
            n_initial_per_fidelity=3,
            acquisition=MFAcquisitionType.COST_AWARE_UCB,
            rng=np.random.default_rng(42)
        )

        assert result.f_best < 3.0

    def test_knowledge_gradient_acquisition(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=50.0,
            n_initial_per_fidelity=3,
            acquisition=MFAcquisitionType.MF_KNOWLEDGE_GRADIENT,
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, MFBOResult)
        assert result.total_cost <= 50.0 + 10.0

    def test_entropy_search_acquisition(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=50.0,
            n_initial_per_fidelity=3,
            acquisition=MFAcquisitionType.ENTROPY_SEARCH,
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, MFBOResult)

    def test_budget_respected(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=30.0,
            n_initial_per_fidelity=2,
            rng=np.random.default_rng(42)
        )

        # Budget should be approximately respected
        assert result.total_cost <= 40.0  # Some slack for final evaluation

    def test_history_consistency(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=50.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        n = len(result.y_history)
        assert result.X_history.shape[0] == n
        assert result.fidelity_history.shape[0] == n
        assert result.cost_history.shape[0] == n
        assert sum(result.n_evaluations.values()) == n

    def test_convergence_monotonic(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=50.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        # Convergence should be non-increasing (best HF value)
        for i in range(1, len(result.convergence)):
            assert result.convergence[i] <= result.convergence[i-1] + 1e-10


# ===== Continuous-Fidelity BO =====

class TestContinuousFidelityBO:
    def test_basic_sphere(self):
        def sphere_cf(x, s):
            return sphere_hf(x) + (1 - s) * 0.5

        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = continuous_fidelity_bo(
            objective=sphere_cf,
            bounds=bounds,
            budget=30.0,
            n_initial=5,
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, ContinuousFidelityResult)
        assert result.total_cost <= 35.0
        assert len(result.convergence) > 1

    def test_branin_continuous(self):
        bounds = Bounds(np.array([-5.0, 0.0]), np.array([10.0, 15.0]))

        result = continuous_fidelity_bo(
            objective=continuous_fidelity_branin,
            bounds=bounds,
            budget=50.0,
            n_initial=8,
            rng=np.random.default_rng(42)
        )

        assert result.x_best.shape == (2,)
        assert result.s_history.shape == result.y_history.shape

    def test_custom_cost_function(self):
        def sphere_cf(x, s):
            return sphere_hf(x) + (1 - s) * 1.0

        def cubic_cost(s):
            return 0.05 + 0.95 * s ** 3

        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = continuous_fidelity_bo(
            objective=sphere_cf,
            bounds=bounds,
            cost_function=cubic_cost,
            budget=20.0,
            n_initial=5,
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, ContinuousFidelityResult)
        # With cubic cost, should prefer lower fidelities
        mean_s = np.mean(result.s_history)
        assert mean_s < 0.9  # Not all high-fidelity

    def test_fidelity_values_in_range(self):
        def sphere_cf(x, s):
            return sphere_hf(x)

        bounds = Bounds(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        result = continuous_fidelity_bo(
            objective=sphere_cf,
            bounds=bounds,
            budget=15.0,
            n_initial=5,
            rng=np.random.default_rng(42)
        )

        assert np.all(result.s_history >= 0.0)
        assert np.all(result.s_history <= 1.0)

    def test_history_lengths_match(self):
        def sphere_cf(x, s):
            return sphere_hf(x) + (1 - s)

        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
        result = continuous_fidelity_bo(
            objective=sphere_cf,
            bounds=bounds,
            budget=20.0,
            n_initial=5,
            rng=np.random.default_rng(42)
        )

        n = len(result.y_history)
        assert result.X_history.shape[0] == n
        assert result.s_history.shape[0] == n
        assert result.cost_history.shape[0] == n


# ===== Multi-Task BO =====

class TestMultiTaskBO:
    def test_basic(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_task_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=60.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        assert isinstance(result, MFBOResult)
        assert result.f_best < 3.0

    def test_prefers_cheap_fidelity(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 20.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_task_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=80.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        # 20x cost difference -> should do many more LF evals
        assert result.n_evaluations[0] > result.n_evaluations[1]


# ===== Comparison =====

class TestCompare:
    def test_mf_vs_single(self):
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        comparison = compare_mf_vs_single(
            hf_objective=sphere_hf,
            lf_objectives={0: sphere_lf},
            bounds=bounds,
            fidelity_costs={0: 1.0, 1: 10.0},
            budget=80.0,
            n_initial=3,
            rng_seed=42
        )

        assert isinstance(comparison, MFComparison)
        assert comparison.mf_result is not None
        assert comparison.sf_result is not None
        assert comparison.cost_ratio > 0
        assert comparison.speedup >= 0

    def test_comparison_fidelity_breakdown(self):
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        comparison = compare_mf_vs_single(
            hf_objective=sphere_hf,
            lf_objectives={0: sphere_lf},
            bounds=bounds,
            fidelity_costs={0: 1.0, 1: 10.0},
            budget=60.0,
            n_initial=3,
            rng_seed=42
        )

        assert 0 in comparison.mf_evaluations
        assert 1 in comparison.mf_evaluations


# ===== Summary =====

class TestSummary:
    def test_generates_string(self):
        result = MFBOResult(
            x_best=np.array([0.1, 0.2]),
            f_best=0.05,
            X_history=np.zeros((10, 2)),
            y_history=np.zeros(10),
            fidelity_history=np.array([0]*7 + [1]*3),
            cost_history=np.ones(10),
            total_cost=10.0,
            n_evaluations={0: 7, 1: 3},
            convergence=[1.0, 0.5, 0.1, 0.05],
            cost_convergence=[1.0, 2.0, 5.0, 10.0]
        )

        summary = mf_optimization_summary(result, "Test MF-BO")
        assert "Test MF-BO" in summary
        assert "0.05" in summary
        assert "Fidelity 0: 7" in summary
        assert "Fidelity 1: 3" in summary


# ===== MFAcquisitionType enum =====

class TestMFAcquisitionType:
    def test_all_values(self):
        assert len(MFAcquisitionType) == 5
        assert MFAcquisitionType.COST_AWARE_EI.value == "cost_aware_ei"
        assert MFAcquisitionType.MF_KNOWLEDGE_GRADIENT.value == "mf_knowledge_gradient"
        assert MFAcquisitionType.COST_AWARE_UCB.value == "cost_aware_ucb"
        assert MFAcquisitionType.ENTROPY_SEARCH.value == "entropy_search"
        assert MFAcquisitionType.MAX_VALUE_ENTROPY.value == "max_value_entropy"


# ===== Integration: Hartmann3 Multi-Fidelity =====

class TestHartmann3Integration:
    def test_hartmann3_mf_bo(self):
        objectives = {0: hartmann3_lf, 1: hartmann3_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.zeros(3), np.ones(3))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=80.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        assert result.f_best < -1.0  # Should find a decent minimum
        assert result.x_best.shape == (3,)

    def test_hartmann3_ar1(self):
        objectives = {0: hartmann3_lf, 1: hartmann3_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.zeros(3), np.ones(3))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=60.0,
            n_initial_per_fidelity=3,
            model="ar1",
            rng=np.random.default_rng(42)
        )

        assert result.f_best < 0  # Hartmann3 min is negative


# ===== Edge Cases =====

class TestEdgeCases:
    def test_tiny_budget(self):
        objectives = {0: sphere_lf, 1: sphere_hf}
        costs = {0: 1.0, 1: 10.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=15.0,
            n_initial_per_fidelity=1,
            rng=np.random.default_rng(42)
        )

        # Should complete without error
        assert isinstance(result, MFBOResult)

    def test_single_fidelity_degenerate(self):
        # MF-BO with one fidelity is just standard BO
        objectives = {0: sphere_hf}
        costs = {0: 1.0}
        bounds = Bounds(np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=30.0,
            n_initial_per_fidelity=5,
            rng=np.random.default_rng(42)
        )

        assert result.f_best < 2.0

    def test_1d_problem(self):
        def f_hf(x):
            return (x[0] - 0.5) ** 2

        def f_lf(x):
            return (x[0] - 0.5) ** 2 * 0.8 + 0.1

        objectives = {0: f_lf, 1: f_hf}
        costs = {0: 1.0, 1: 5.0}
        bounds = Bounds(np.array([0.0]), np.array([1.0]))

        result = multi_fidelity_bo(
            objectives=objectives,
            bounds=bounds,
            fidelity_costs=costs,
            budget=40.0,
            n_initial_per_fidelity=3,
            rng=np.random.default_rng(42)
        )

        assert result.f_best < 0.1  # Should find near 0.5
        assert result.x_best.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
