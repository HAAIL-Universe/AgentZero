"""Tests for C157: Hidden Markov Model"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from hidden_markov_model import (
    HMM, GaussianHMM, GMMHMM, BayesianHMM,
    HMMClassifier, HMMSegmenter, ARHiddenMarkovModel,
    FactorialHMM, HMMUtils
)


# ===========================================================================
# HMM (Discrete)
# ===========================================================================

class TestHMMBasic:
    """Basic discrete HMM tests."""

    def test_init(self):
        hmm = HMM(3, 4, seed=42)
        assert hmm.n_states == 3
        assert hmm.n_obs == 4
        assert hmm.pi.shape == (3,)
        assert hmm.A.shape == (3, 3)
        assert hmm.B.shape == (3, 4)
        assert np.isclose(hmm.pi.sum(), 1.0)

    def test_set_params(self):
        hmm = HMM(2, 2)
        hmm.set_params(pi=[0.6, 0.4], A=[[0.7, 0.3], [0.4, 0.6]], B=[[0.5, 0.5], [0.1, 0.9]])
        assert np.isclose(hmm.pi[0], 0.6)
        assert np.isclose(hmm.A[0, 1], 0.3)
        assert np.isclose(hmm.B[1, 1], 0.9)

    def test_stochastic_constraints(self):
        hmm = HMM(5, 8, seed=42)
        assert np.isclose(hmm.pi.sum(), 1.0)
        for i in range(5):
            assert np.isclose(hmm.A[i].sum(), 1.0)
            assert np.isclose(hmm.B[i].sum(), 1.0)

    def test_sample(self):
        hmm = HMM(2, 3, seed=42)
        states, obs = hmm.sample(100, seed=0)
        assert len(states) == 100
        assert len(obs) == 100
        assert all(0 <= s < 2 for s in states)
        assert all(0 <= o < 3 for o in obs)

    def test_sample_deterministic_seed(self):
        hmm = HMM(2, 3, seed=42)
        s1, o1 = hmm.sample(50, seed=123)
        s2, o2 = hmm.sample(50, seed=123)
        assert s1 == s2
        assert o1 == o2


class TestHMMForward:
    """Forward algorithm tests."""

    def test_forward_probabilities_sum_to_one(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(20, seed=0)
        alpha, log_lik = hmm.forward(obs)
        # Scaled alpha should sum to 1 at each step
        for t in range(len(obs)):
            assert np.isclose(alpha[t].sum(), 1.0, atol=1e-6)

    def test_forward_log_likelihood_finite(self):
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(50, seed=0)
        _, log_lik = hmm.forward(obs)
        assert np.isfinite(log_lik)
        assert log_lik < 0  # log prob is negative

    def test_forward_shape(self):
        hmm = HMM(3, 5, seed=42)
        _, obs = hmm.sample(30, seed=0)
        alpha, _ = hmm.forward(obs)
        assert alpha.shape == (30, 3)

    def test_forward_known_model(self):
        """Test with a known simple model."""
        hmm = HMM(2, 2)
        hmm.set_params(
            pi=[0.5, 0.5],
            A=[[0.7, 0.3], [0.4, 0.6]],
            B=[[0.9, 0.1], [0.2, 0.8]]
        )
        alpha, log_lik = hmm.forward([0, 0, 1])
        assert np.isfinite(log_lik)
        assert alpha.shape == (3, 2)


class TestHMMForwardBackward:
    """Forward-backward algorithm tests."""

    def test_gamma_sums_to_one(self):
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(30, seed=0)
        gamma, xi, log_lik = hmm.forward_backward(obs)
        for t in range(len(obs)):
            assert np.isclose(gamma[t].sum(), 1.0, atol=1e-6)

    def test_xi_shape(self):
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(20, seed=0)
        gamma, xi, _ = hmm.forward_backward(obs)
        assert xi.shape == (19, 3, 3)

    def test_xi_marginal_equals_gamma(self):
        """Sum of xi over j should give gamma (approximately)."""
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(30, seed=0)
        gamma, xi, _ = hmm.forward_backward(obs)
        for t in range(len(obs) - 1):
            xi_marginal = xi[t].sum(axis=1)
            assert np.allclose(xi_marginal, gamma[t], atol=1e-4)

    def test_gamma_nonnegative(self):
        hmm = HMM(4, 5, seed=42)
        _, obs = hmm.sample(40, seed=0)
        gamma, _, _ = hmm.forward_backward(obs)
        assert np.all(gamma >= 0)


class TestHMMViterbi:
    """Viterbi algorithm tests."""

    def test_viterbi_returns_valid_path(self):
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(20, seed=0)
        path, log_prob = hmm.viterbi(obs)
        assert len(path) == 20
        assert all(0 <= s < 3 for s in path)

    def test_viterbi_log_prob_finite(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(30, seed=0)
        _, log_prob = hmm.viterbi(obs)
        assert np.isfinite(log_prob)
        assert log_prob < 0

    def test_viterbi_deterministic_model(self):
        """With very strong transitions, Viterbi should recover true states."""
        hmm = HMM(2, 2)
        hmm.set_params(
            pi=[1.0, 0.0],
            A=[[0.99, 0.01], [0.01, 0.99]],
            B=[[0.99, 0.01], [0.01, 0.99]]
        )
        # State 0 emits 0, state 1 emits 1
        obs = [0, 0, 0, 1, 1, 1]
        path, _ = hmm.viterbi(obs)
        assert path[:3] == [0, 0, 0]
        assert path[3:] == [1, 1, 1]

    def test_viterbi_single_observation(self):
        hmm = HMM(3, 4, seed=42)
        path, log_prob = hmm.viterbi([0])
        assert len(path) == 1
        assert 0 <= path[0] < 3


class TestHMMBaumWelch:
    """Baum-Welch (EM) tests."""

    def test_baum_welch_improves_likelihood(self):
        hmm_true = HMM(2, 3, seed=42)
        hmm_true.set_params(
            pi=[0.6, 0.4],
            A=[[0.7, 0.3], [0.4, 0.6]],
            B=[[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]]
        )
        sequences = [hmm_true.sample(50, seed=i)[1] for i in range(10)]

        hmm_learn = HMM(2, 3, seed=0)
        initial_ll = sum(hmm_learn.log_likelihood(s) for s in sequences)
        log_liks = hmm_learn.baum_welch(sequences, n_iter=30)
        final_ll = sum(hmm_learn.log_likelihood(s) for s in sequences)

        assert final_ll > initial_ll

    def test_baum_welch_convergence(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(100, seed=0)
        log_liks = hmm.baum_welch([obs], n_iter=50)
        # Should be non-decreasing (within numerical tolerance)
        for i in range(1, len(log_liks)):
            assert log_liks[i] >= log_liks[i - 1] - 1e-6

    def test_baum_welch_multiple_sequences(self):
        hmm_true = HMM(2, 4, seed=42)
        sequences = [hmm_true.sample(30, seed=i)[1] for i in range(5)]
        hmm_learn = HMM(2, 4, seed=0)
        log_liks = hmm_learn.baum_welch(sequences, n_iter=20)
        assert len(log_liks) >= 1
        assert log_liks[-1] >= log_liks[0]

    def test_baum_welch_maintains_stochastic(self):
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(50, seed=0)
        hmm.baum_welch([obs], n_iter=10)
        assert np.isclose(hmm.pi.sum(), 1.0, atol=1e-6)
        for i in range(3):
            assert np.isclose(hmm.A[i].sum(), 1.0, atol=1e-6)
            assert np.isclose(hmm.B[i].sum(), 1.0, atol=1e-6)

    def test_baum_welch_recovers_structure(self):
        """BW should recover approximate structure from generated data."""
        hmm_true = HMM(2, 3, seed=42)
        hmm_true.set_params(
            pi=[0.8, 0.2],
            A=[[0.9, 0.1], [0.2, 0.8]],
            B=[[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
        )
        sequences = [hmm_true.sample(100, seed=i)[1] for i in range(20)]

        hmm_learn = HMM(2, 3, seed=0)
        hmm_learn.baum_welch(sequences, n_iter=50)

        # Check that diagonal of A is dominant (states are sticky)
        assert hmm_learn.A[0, 0] > 0.5 or hmm_learn.A[1, 1] > 0.5


class TestHMMOther:
    """Additional HMM tests."""

    def test_log_likelihood(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(20, seed=0)
        ll = hmm.log_likelihood(obs)
        assert np.isfinite(ll)
        assert ll < 0

    def test_predict(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(20, seed=0)
        preds = hmm.predict(obs, n_steps=3)
        assert len(preds) == 3
        for state, probs in preds:
            assert 0 <= state < 2
            assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_stationary_distribution(self):
        hmm = HMM(3, 4, seed=42)
        stat = hmm.stationary_distribution()
        assert len(stat) == 3
        assert np.isclose(stat.sum(), 1.0, atol=1e-6)
        assert np.all(stat >= 0)

    def test_n_params(self):
        hmm = HMM(3, 4)
        # pi: 2, A: 3*2=6, B: 3*3=9 -> 17
        assert hmm.n_params() == 17

    def test_backward(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(15, seed=0)
        beta = hmm._backward_with_forward(obs)
        assert beta.shape == (15, 2)
        assert np.all(np.isfinite(beta))


# ===========================================================================
# Gaussian HMM
# ===========================================================================

class TestGaussianHMM:
    """Gaussian HMM tests."""

    def test_init(self):
        ghmm = GaussianHMM(3, n_features=2, seed=42)
        assert ghmm.n_states == 3
        assert ghmm.n_features == 2
        assert ghmm.means.shape == (3, 2)

    def test_sample(self):
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        states, obs = ghmm.sample(50, seed=0)
        assert len(states) == 50
        assert obs.shape == (50, 1)

    def test_forward_continuous(self):
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        ghmm.set_params(
            means=[[0.0], [5.0]],
            covars=[[0.5], [0.5]]
        )
        states, obs = ghmm.sample(30, seed=0)
        alpha, log_lik = ghmm.forward(obs)
        assert np.isfinite(log_lik)
        assert alpha.shape == (30, 2)

    def test_viterbi_continuous(self):
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        ghmm.set_params(
            pi=[0.5, 0.5],
            A=[[0.9, 0.1], [0.1, 0.9]],
            means=[[-3.0], [3.0]],
            covars=[[0.1], [0.1]]
        )
        states, obs = ghmm.sample(40, seed=0)
        path, log_prob = ghmm.viterbi(obs)
        assert len(path) == 40
        assert np.isfinite(log_prob)
        # With well-separated means and low variance, should mostly recover states
        agreement = sum(1 for a, b in zip(states, path) if a == b) / len(states)
        assert agreement > 0.7

    def test_fit_improves(self):
        ghmm_true = GaussianHMM(2, n_features=1, seed=42)
        ghmm_true.set_params(
            pi=[0.5, 0.5],
            A=[[0.8, 0.2], [0.3, 0.7]],
            means=[[-2.0], [2.0]],
            covars=[[0.5], [0.5]]
        )
        sequences = [ghmm_true.sample(60, seed=i)[1] for i in range(10)]

        ghmm_learn = GaussianHMM(2, n_features=1, seed=0)
        log_liks = ghmm_learn.fit(sequences, n_iter=30)
        assert log_liks[-1] >= log_liks[0]

    def test_fit_recovers_means(self):
        ghmm_true = GaussianHMM(2, n_features=1, seed=42)
        ghmm_true.set_params(
            pi=[0.5, 0.5],
            A=[[0.9, 0.1], [0.1, 0.9]],
            means=[[-5.0], [5.0]],
            covars=[[0.3], [0.3]]
        )
        sequences = [ghmm_true.sample(100, seed=i)[1] for i in range(15)]

        ghmm_learn = GaussianHMM(2, n_features=1, seed=0)
        ghmm_learn.fit(sequences, n_iter=50)

        learned_means = sorted(ghmm_learn.means[:, 0])
        assert abs(learned_means[0] - (-5.0)) < 2.0
        assert abs(learned_means[1] - 5.0) < 2.0

    def test_full_covariance(self):
        ghmm = GaussianHMM(2, n_features=2, covariance_type='full', seed=42)
        assert ghmm.covars.shape == (2, 2, 2)
        states, obs = ghmm.sample(30, seed=0)
        alpha, ll = ghmm.forward(obs)
        assert np.isfinite(ll)

    def test_spherical_covariance(self):
        ghmm = GaussianHMM(2, n_features=2, covariance_type='spherical', seed=42)
        states, obs = ghmm.sample(30, seed=0)
        path, lp = ghmm.viterbi(obs)
        assert len(path) == 30

    def test_log_likelihood(self):
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        _, obs = ghmm.sample(20, seed=0)
        ll = ghmm.log_likelihood(obs)
        assert np.isfinite(ll)

    def test_score(self):
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        sequences = [ghmm.sample(20, seed=i)[1] for i in range(5)]
        avg_ll = ghmm.score(sequences)
        assert np.isfinite(avg_ll)

    def test_set_params_1d(self):
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        ghmm.set_params(means=[1.0, 2.0])
        assert ghmm.means.shape == (2, 1)


# ===========================================================================
# GMM-HMM
# ===========================================================================

class TestGMMHMM:
    """GMM-HMM tests."""

    def test_init(self):
        model = GMMHMM(3, n_mix=2, n_features=1, seed=42)
        assert model.n_states == 3
        assert model.n_mix == 2
        assert model.weights.shape == (3, 2)
        assert model.means.shape == (3, 2, 1)

    def test_sample(self):
        model = GMMHMM(2, n_mix=2, n_features=1, seed=42)
        states, obs = model.sample(50, seed=0)
        assert len(states) == 50
        assert obs.shape == (50, 1)

    def test_forward(self):
        model = GMMHMM(2, n_mix=2, n_features=1, seed=42)
        _, obs = model.sample(30, seed=0)
        alpha, ll = model.forward(obs)
        assert np.isfinite(ll)
        assert alpha.shape == (30, 2)

    def test_viterbi(self):
        model = GMMHMM(2, n_mix=2, n_features=1, seed=42)
        model.set_params(
            means=np.array([[[-5.0], [-3.0]], [[3.0], [5.0]]]),
            covars=np.array([[[0.3], [0.3]], [[0.3], [0.3]]])
        )
        _, obs = model.sample(40, seed=0)
        path, lp = model.viterbi(obs)
        assert len(path) == 40
        assert np.isfinite(lp)

    def test_fit(self):
        model_true = GMMHMM(2, n_mix=2, n_features=1, seed=42)
        model_true.set_params(
            means=np.array([[[-3.0], [-1.0]], [[1.0], [3.0]]]),
            covars=np.array([[[0.2], [0.2]], [[0.2], [0.2]]])
        )
        sequences = [model_true.sample(50, seed=i)[1] for i in range(8)]

        model_learn = GMMHMM(2, n_mix=2, n_features=1, seed=0)
        log_liks = model_learn.fit(sequences, n_iter=20)
        assert log_liks[-1] >= log_liks[0]

    def test_set_params(self):
        model = GMMHMM(2, n_mix=2, n_features=1, seed=42)
        model.set_params(pi=[0.3, 0.7])
        assert np.isclose(model.pi[0], 0.3)


# ===========================================================================
# Bayesian HMM
# ===========================================================================

class TestBayesianHMM:
    """Bayesian HMM tests."""

    def test_fit(self):
        hmm_true = HMM(2, 3, seed=42)
        hmm_true.set_params(
            pi=[0.6, 0.4],
            A=[[0.8, 0.2], [0.3, 0.7]],
            B=[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
        )
        sequences = [hmm_true.sample(30, seed=i)[1] for i in range(5)]

        bhmm = BayesianHMM(2, 3, seed=42)
        bhmm.fit(sequences, n_samples=200)
        assert bhmm.trace is not None
        assert len(bhmm.trace) > 0

    def test_get_model(self):
        hmm_true = HMM(2, 3, seed=42)
        sequences = [hmm_true.sample(30, seed=i)[1] for i in range(3)]

        bhmm = BayesianHMM(2, 3, seed=42)
        bhmm.fit(sequences, n_samples=100)
        model = bhmm.get_model()
        assert isinstance(model, HMM)
        assert np.isclose(model.pi.sum(), 1.0, atol=1e-4)

    def test_posterior_predictive(self):
        hmm_true = HMM(2, 3, seed=42)
        sequences = [hmm_true.sample(30, seed=i)[1] for i in range(3)]

        bhmm = BayesianHMM(2, 3, seed=42)
        bhmm.fit(sequences, n_samples=100)
        pp = bhmm.posterior_predictive(length=10, n_samples=5)
        assert len(pp) == 5
        assert len(pp[0]['states']) == 10
        assert len(pp[0]['observations']) == 10

    def test_not_fit_raises(self):
        bhmm = BayesianHMM(2, 3)
        with pytest.raises(ValueError):
            bhmm.get_model()
        with pytest.raises(ValueError):
            bhmm.posterior_predictive()


# ===========================================================================
# HMM Classifier
# ===========================================================================

class TestHMMClassifier:
    """HMM Classifier tests."""

    def _make_data(self):
        hmm_a = HMM(2, 4, seed=10)
        hmm_a.set_params(
            pi=[0.8, 0.2],
            A=[[0.9, 0.1], [0.2, 0.8]],
            B=[[0.7, 0.2, 0.05, 0.05], [0.05, 0.05, 0.2, 0.7]]
        )
        hmm_b = HMM(2, 4, seed=20)
        hmm_b.set_params(
            pi=[0.2, 0.8],
            A=[[0.6, 0.4], [0.4, 0.6]],
            B=[[0.05, 0.7, 0.2, 0.05], [0.2, 0.05, 0.05, 0.7]]
        )
        train = {
            'A': [hmm_a.sample(40, seed=i)[1] for i in range(10)],
            'B': [hmm_b.sample(40, seed=i + 100)[1] for i in range(10)]
        }
        test = [
            (hmm_a.sample(40, seed=50)[1], 'A'),
            (hmm_a.sample(40, seed=51)[1], 'A'),
            (hmm_b.sample(40, seed=52)[1], 'B'),
            (hmm_b.sample(40, seed=53)[1], 'B'),
        ]
        return train, test

    def test_fit_and_predict(self):
        train, test = self._make_data()
        clf = HMMClassifier(n_states=2, n_obs=4, n_iter=30, seed=42)
        clf.fit(train)
        for obs, true_cls in test:
            pred = clf.predict(obs)
            assert pred in ['A', 'B']

    def test_predict_proba(self):
        train, test = self._make_data()
        clf = HMMClassifier(n_states=2, n_obs=4, n_iter=30, seed=42)
        clf.fit(train)
        probs = clf.predict_proba(test[0][0])
        assert 'A' in probs and 'B' in probs
        assert np.isclose(sum(probs.values()), 1.0, atol=1e-6)

    def test_score(self):
        train, test = self._make_data()
        clf = HMMClassifier(n_states=2, n_obs=4, n_iter=30, seed=42)
        clf.fit(train)
        acc = clf.score(test)
        assert 0.0 <= acc <= 1.0

    def test_auto_n_obs(self):
        train, _ = self._make_data()
        clf = HMMClassifier(n_states=2, n_iter=10, seed=42)
        clf.fit(train)
        assert clf.n_obs == 4


# ===========================================================================
# HMM Segmenter
# ===========================================================================

class TestHMMSegmenter:
    """HMM Segmenter tests."""

    def _make_segmented_data(self, seed=42):
        rng = np.random.RandomState(seed)
        # Three segments with different means
        seg1 = rng.randn(30) * 0.3 + 0.0
        seg2 = rng.randn(30) * 0.3 + 5.0
        seg3 = rng.randn(30) * 0.3 + 0.0
        return np.concatenate([seg1, seg2, seg3])

    def test_fit_and_segment(self):
        data = self._make_segmented_data()
        seg = HMMSegmenter(n_segments=2, seed=42)
        seg.fit(data.reshape(-1, 1), n_iter=50)
        labels = seg.segment(data.reshape(-1, 1))
        assert len(labels) == 90

    def test_change_points(self):
        data = self._make_segmented_data()
        seg = HMMSegmenter(n_segments=2, seed=42)
        seg.fit(data.reshape(-1, 1), n_iter=50)
        cps = seg.change_points(data.reshape(-1, 1))
        assert len(cps) >= 1  # Should detect at least one change

    def test_segment_summary(self):
        data = self._make_segmented_data()
        seg = HMMSegmenter(n_segments=2, seed=42)
        seg.fit(data.reshape(-1, 1), n_iter=50)
        summary = seg.segment_summary(data.reshape(-1, 1))
        assert len(summary) >= 2
        for s in summary:
            assert 'start' in s
            assert 'end' in s
            assert 'label' in s
            assert 'length' in s
            assert 'mean' in s

    def test_detects_shift(self):
        """Should detect the big mean shift."""
        data = self._make_segmented_data()
        seg = HMMSegmenter(n_segments=2, seed=42)
        seg.fit(data.reshape(-1, 1), n_iter=50)
        cps = seg.change_points(data.reshape(-1, 1))
        # Should have change points near 30 and 60
        has_near_30 = any(25 <= cp <= 35 for cp in cps)
        has_near_60 = any(55 <= cp <= 65 for cp in cps)
        assert has_near_30 or has_near_60


# ===========================================================================
# AR-HMM
# ===========================================================================

class TestARHMM:
    """Autoregressive HMM tests."""

    def test_init(self):
        model = ARHiddenMarkovModel(3, seed=42)
        assert model.n_states == 3
        assert len(model.ar_coeffs) == 3

    def test_sample(self):
        model = ARHiddenMarkovModel(2, seed=42)
        states, obs = model.sample(50, seed=0)
        assert len(states) == 50
        assert len(obs) == 50
        assert all(np.isfinite(obs))

    def test_forward(self):
        model = ARHiddenMarkovModel(2, seed=42)
        _, obs = model.sample(30, seed=0)
        alpha, ll = model.forward(obs)
        assert np.isfinite(ll)
        assert alpha.shape == (30, 2)

    def test_viterbi(self):
        model = ARHiddenMarkovModel(2, seed=42)
        model.set_params(
            pi=[0.5, 0.5],
            A=[[0.9, 0.1], [0.1, 0.9]],
            ar_coeffs=[0.9, -0.9],
            ar_intercepts=[5.0, -5.0],
            ar_variances=[0.1, 0.1]
        )
        states, obs = model.sample(30, seed=0)
        path, lp = model.viterbi(obs)
        assert len(path) == 30
        assert np.isfinite(lp)

    def test_log_likelihood(self):
        model = ARHiddenMarkovModel(2, seed=42)
        _, obs = model.sample(20, seed=0)
        ll = model.log_likelihood(obs)
        assert np.isfinite(ll)
        assert ll < 0

    def test_set_params(self):
        model = ARHiddenMarkovModel(2, seed=42)
        model.set_params(ar_coeffs=[0.5, -0.5], ar_intercepts=[1.0, -1.0])
        assert np.isclose(model.ar_coeffs[0], 0.5)
        assert np.isclose(model.ar_intercepts[1], -1.0)


# ===========================================================================
# Factorial HMM
# ===========================================================================

class TestFactorialHMM:
    """Factorial HMM tests."""

    def test_init(self):
        model = FactorialHMM(n_chains=2, states_per_chain=2, n_features=1, seed=42)
        assert model.n_chains == 2
        assert model.states_per_chain == 2
        assert len(model.pis) == 2
        assert len(model.As) == 2

    def test_sample(self):
        model = FactorialHMM(2, 2, n_features=1, seed=42)
        chain_states, obs = model.sample(30, seed=0)
        assert len(chain_states) == 30
        assert obs.shape == (30, 1)
        assert all(len(cs) == 2 for cs in chain_states)

    def test_forward(self):
        model = FactorialHMM(2, 2, n_features=1, seed=42)
        _, obs = model.sample(20, seed=0)
        alpha, ll, joint_states = model.forward(obs)
        assert np.isfinite(ll)
        assert alpha.shape == (20, 4)  # 2^2 = 4 joint states
        assert len(joint_states) == 4

    def test_viterbi(self):
        model = FactorialHMM(2, 2, n_features=1, seed=42)
        # Set well-separated means
        model.chain_means[0] = np.array([[0.0], [5.0]])
        model.chain_means[1] = np.array([[0.0], [10.0]])
        model.noise_var = np.array([0.1])

        _, obs = model.sample(20, seed=0)
        path, lp = model.viterbi(obs)
        assert len(path) == 20
        assert np.isfinite(lp)
        assert all(len(p) == 2 for p in path)

    def test_joint_state_count(self):
        model = FactorialHMM(3, 2, n_features=1, seed=42)
        js = model._all_joint_states()
        assert len(js) == 8  # 2^3


# ===========================================================================
# HMM Utils
# ===========================================================================

class TestHMMUtils:
    """HMM utility function tests."""

    def test_aic(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(50, seed=0)
        aic_val = HMMUtils.aic(hmm, [obs])
        assert np.isfinite(aic_val)

    def test_bic(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(50, seed=0)
        bic_val = HMMUtils.bic(hmm, [obs])
        assert np.isfinite(bic_val)

    def test_select_n_states(self):
        hmm_true = HMM(2, 3, seed=42)
        hmm_true.set_params(
            pi=[0.5, 0.5],
            A=[[0.9, 0.1], [0.1, 0.9]],
            B=[[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]
        )
        sequences = [hmm_true.sample(60, seed=i)[1] for i in range(10)]
        best_n, scores = HMMUtils.select_n_states(sequences, n_obs=3, max_states=4, seed=42, n_iter=20)
        assert 1 <= best_n <= 4
        assert len(scores) == 4

    def test_sequence_entropy(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(20, seed=0)
        ent = HMMUtils.sequence_entropy(hmm, obs)
        assert len(ent) == 20
        assert all(e >= 0 for e in ent)

    def test_state_occupancy(self):
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(50, seed=0)
        occ = HMMUtils.state_occupancy(hmm, obs)
        assert len(occ) == 3
        assert np.isclose(occ.sum(), 1.0, atol=1e-6)

    def test_most_likely_state_sequence(self):
        hmm = HMM(2, 3, seed=42)
        _, obs = hmm.sample(20, seed=0)
        path = HMMUtils.most_likely_state_sequence(hmm, obs)
        assert len(path) == 20
        assert all(0 <= s < 2 for s in path)

    def test_compare_models(self):
        hmm1 = HMM(2, 3, seed=42)
        hmm2 = HMM(3, 3, seed=42)
        _, obs = hmm1.sample(50, seed=0)
        results = HMMUtils.compare_models([('2-state', hmm1), ('3-state', hmm2)], [obs])
        assert len(results) == 2
        assert results[0][1] <= results[1][1]  # Sorted by score

    def test_kl_divergence(self):
        hmm1 = HMM(2, 3, seed=42)
        hmm2 = HMM(2, 3, seed=0)
        kl = HMMUtils.kl_divergence_hmm(hmm1, hmm2, n_sequences=20, seq_length=30)
        assert np.isfinite(kl)

    def test_kl_same_model_near_zero(self):
        hmm = HMM(2, 3, seed=42)
        kl = HMMUtils.kl_divergence_hmm(hmm, hmm, n_sequences=20, seq_length=30)
        assert abs(kl) < 1.0  # Should be close to 0

    def test_merge_states(self):
        hmm = HMM(3, 4, seed=42)
        merged = HMMUtils.merge_states(hmm, 0, 2)
        assert merged.n_states == 2
        assert merged.n_obs == 4
        assert np.isclose(merged.pi.sum(), 1.0, atol=1e-6)
        for i in range(2):
            assert np.isclose(merged.A[i].sum(), 1.0, atol=1e-6)
            assert np.isclose(merged.B[i].sum(), 1.0, atol=1e-6)

    def test_merge_states_reversed(self):
        hmm = HMM(3, 4, seed=42)
        m1 = HMMUtils.merge_states(hmm, 0, 2)
        m2 = HMMUtils.merge_states(hmm, 2, 0)
        assert np.allclose(m1.pi, m2.pi)


# ===========================================================================
# Edge cases & integration
# ===========================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_state_hmm(self):
        hmm = HMM(1, 3, seed=42)
        hmm.set_params(pi=[1.0], A=[[1.0]], B=[[0.5, 0.3, 0.2]])
        states, obs = hmm.sample(10, seed=0)
        assert all(s == 0 for s in states)
        path, lp = hmm.viterbi(obs)
        assert all(s == 0 for s in path)

    def test_single_obs_symbol(self):
        hmm = HMM(2, 1, seed=42)
        hmm.set_params(pi=[0.5, 0.5], A=[[0.7, 0.3], [0.4, 0.6]], B=[[1.0], [1.0]])
        states, obs = hmm.sample(10, seed=0)
        assert all(o == 0 for o in obs)

    def test_long_sequence(self):
        hmm = HMM(3, 5, seed=42)
        _, obs = hmm.sample(500, seed=0)
        alpha, ll = hmm.forward(obs)
        assert np.isfinite(ll)
        path, lp = hmm.viterbi(obs)
        assert len(path) == 500

    def test_short_sequence_length_1(self):
        hmm = HMM(2, 3, seed=42)
        gamma, xi, ll = hmm.forward_backward([1])
        assert gamma.shape == (1, 2)
        assert xi.shape == (0, 2, 2)

    def test_gaussian_hmm_multivariate(self):
        ghmm = GaussianHMM(2, n_features=3, covariance_type='diag', seed=42)
        states, obs = ghmm.sample(30, seed=0)
        assert obs.shape == (30, 3)
        alpha, ll = ghmm.forward(obs)
        assert np.isfinite(ll)

    def test_gaussian_hmm_fit_multivariate(self):
        ghmm = GaussianHMM(2, n_features=2, covariance_type='diag', seed=42)
        ghmm.set_params(
            means=[[-3.0, -3.0], [3.0, 3.0]],
            covars=[[0.3, 0.3], [0.3, 0.3]]
        )
        sequences = [ghmm.sample(40, seed=i)[1] for i in range(5)]
        ghmm_learn = GaussianHMM(2, n_features=2, seed=0)
        log_liks = ghmm_learn.fit(sequences, n_iter=20)
        assert log_liks[-1] >= log_liks[0]


class TestIntegration:
    """Integration tests composing multiple components."""

    def test_train_classify_pipeline(self):
        """Train HMMs, classify sequences, compute metrics."""
        hmm_a = HMM(2, 4, seed=10)
        hmm_a.set_params(
            pi=[0.9, 0.1], A=[[0.9, 0.1], [0.1, 0.9]],
            B=[[0.7, 0.2, 0.05, 0.05], [0.05, 0.05, 0.2, 0.7]]
        )
        hmm_b = HMM(2, 4, seed=20)
        hmm_b.set_params(
            pi=[0.1, 0.9], A=[[0.7, 0.3], [0.3, 0.7]],
            B=[[0.05, 0.7, 0.2, 0.05], [0.2, 0.05, 0.05, 0.7]]
        )

        train = {
            'A': [hmm_a.sample(50, seed=i)[1] for i in range(8)],
            'B': [hmm_b.sample(50, seed=i + 100)[1] for i in range(8)]
        }
        clf = HMMClassifier(n_states=2, n_obs=4, n_iter=30, seed=42)
        clf.fit(train)

        # Score should be reasonable
        test = [(hmm_a.sample(50, seed=80)[1], 'A'), (hmm_b.sample(50, seed=81)[1], 'B')]
        acc = clf.score(test)
        assert acc >= 0.5

    def test_sample_train_viterbi_pipeline(self):
        """Generate data, train, decode."""
        hmm_true = HMM(3, 5, seed=42)
        hmm_true.set_params(
            pi=[0.5, 0.3, 0.2],
            A=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            B=[[0.6, 0.2, 0.1, 0.05, 0.05],
               [0.05, 0.1, 0.6, 0.2, 0.05],
               [0.05, 0.05, 0.1, 0.2, 0.6]]
        )
        sequences = [hmm_true.sample(80, seed=i)[1] for i in range(10)]

        hmm_learn = HMM(3, 5, seed=0)
        log_liks = hmm_learn.baum_welch(sequences, n_iter=30)
        assert log_liks[-1] > log_liks[0]

        path, lp = hmm_learn.viterbi(sequences[0])
        assert len(path) == 80

    def test_segmenter_with_gaussian_hmm(self):
        """Segment a time series with clear regime changes."""
        rng = np.random.RandomState(42)
        seg1 = rng.randn(40) * 0.5 + 0.0
        seg2 = rng.randn(40) * 0.5 + 8.0
        data = np.concatenate([seg1, seg2])

        segmenter = HMMSegmenter(n_segments=2, seed=42)
        segmenter.fit(data.reshape(-1, 1), n_iter=50)
        labels = segmenter.segment(data.reshape(-1, 1))

        # First half and second half should have different labels
        first_label = labels[0]
        second_label = labels[-1]
        assert first_label != second_label

    def test_model_selection_and_evaluation(self):
        """Full pipeline: generate, select model, evaluate."""
        hmm_true = HMM(2, 3, seed=42)
        hmm_true.set_params(
            pi=[0.5, 0.5],
            A=[[0.85, 0.15], [0.2, 0.8]],
            B=[[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
        )
        sequences = [hmm_true.sample(60, seed=i)[1] for i in range(8)]

        best_n, scores = HMMUtils.select_n_states(sequences, n_obs=3, max_states=4, seed=42, n_iter=20)
        assert 1 <= best_n <= 4

        hmm_final = HMM(best_n, 3, seed=42)
        hmm_final.baum_welch(sequences, n_iter=30)

        occupancy = HMMUtils.state_occupancy(hmm_final, sequences[0])
        assert np.isclose(occupancy.sum(), 1.0, atol=1e-6)

    def test_ar_hmm_regime_switching(self):
        """AR-HMM for regime switching time series."""
        model = ARHiddenMarkovModel(2, seed=42)
        model.set_params(
            pi=[0.5, 0.5],
            A=[[0.95, 0.05], [0.05, 0.95]],
            ar_coeffs=[0.8, -0.5],
            ar_intercepts=[2.0, -2.0],
            ar_variances=[0.3, 0.3]
        )
        states, obs = model.sample(100, seed=0)
        path, lp = model.viterbi(obs)

        # Viterbi should recover some regime structure
        assert len(path) == 100
        assert np.isfinite(lp)

    def test_factorial_hmm_decomposition(self):
        """Factorial HMM should decompose additive signals."""
        model = FactorialHMM(2, 2, n_features=1, seed=42)
        model.chain_means[0] = np.array([[0.0], [10.0]])
        model.chain_means[1] = np.array([[0.0], [5.0]])
        model.noise_var = np.array([0.01])

        chain_states, obs = model.sample(20, seed=0)
        path, lp = model.viterbi(obs)
        assert len(path) == 20
        assert np.isfinite(lp)

    def test_bayesian_hmm_posterior_consistency(self):
        """Bayesian HMM posterior should be consistent with data."""
        hmm_true = HMM(2, 3, seed=42)
        hmm_true.set_params(
            pi=[0.7, 0.3],
            A=[[0.9, 0.1], [0.2, 0.8]],
            B=[[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]]
        )
        sequences = [hmm_true.sample(40, seed=i)[1] for i in range(5)]

        bhmm = BayesianHMM(2, 3, seed=42)
        bhmm.fit(sequences, n_samples=100)
        model = bhmm.get_model()

        # Posterior mean model should assign reasonable likelihood
        ll = model.log_likelihood(sequences[0])
        assert np.isfinite(ll)


# ===========================================================================
# Numerical stability tests
# ===========================================================================

class TestNumericalStability:
    """Numerical stability edge cases."""

    def test_forward_long_sequence(self):
        hmm = HMM(5, 10, seed=42)
        _, obs = hmm.sample(1000, seed=0)
        alpha, ll = hmm.forward(obs)
        assert np.isfinite(ll)
        assert np.all(np.isfinite(alpha))

    def test_viterbi_many_states(self):
        hmm = HMM(10, 5, seed=42)
        _, obs = hmm.sample(100, seed=0)
        path, lp = hmm.viterbi(obs)
        assert len(path) == 100
        assert np.isfinite(lp)

    def test_baum_welch_sparse_data(self):
        """BW with observations that rarely occur."""
        hmm = HMM(2, 10, seed=42)
        # Only observe symbols 0 and 1
        sequences = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
        log_liks = hmm.baum_welch(sequences, n_iter=10)
        assert all(np.isfinite(ll) for ll in log_liks)

    def test_gaussian_hmm_zero_variance_safeguard(self):
        """After fitting, variances should stay positive."""
        ghmm = GaussianHMM(2, n_features=1, seed=42)
        # Degenerate data -- all same value
        data = np.ones((20, 1))
        ghmm.fit([data], n_iter=10)
        if ghmm.covariance_type == 'diag':
            assert np.all(ghmm.covars > 0)

    def test_forward_backward_consistency(self):
        """forward() and forward_backward() should give same log_lik."""
        hmm = HMM(3, 4, seed=42)
        _, obs = hmm.sample(30, seed=0)
        _, ll_fwd = hmm.forward(obs)
        _, _, ll_fb = hmm.forward_backward(obs)
        assert np.isclose(ll_fwd, ll_fb, atol=1e-8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
