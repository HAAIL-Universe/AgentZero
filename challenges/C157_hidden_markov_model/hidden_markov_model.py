"""
C157: Hidden Markov Model
Composing C153 (Monte Carlo) + C156 (Probabilistic Programming)

Components:
- HMM: Core discrete HMM (forward, backward, Viterbi, Baum-Welch, sampling)
- GaussianHMM: Continuous observations with Gaussian emissions
- GMMHMM: Gaussian Mixture Model emissions
- BayesianHMM: Bayesian HMM using C156 PPL for posterior inference
- HMMClassifier: Sequence classification with multiple HMMs
- HMMSegmenter: Change-point / segmentation detection
- ARHiddenMarkovModel: Autoregressive HMM
- FactorialHMM: Multiple parallel hidden chains
- HMMUtils: Sequence generation, scoring, model selection
"""

import numpy as np
from numpy.linalg import inv, det
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C153_monte_carlo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C156_probabilistic_programming'))

from monte_carlo import MetropolisHastings, MCMCDiagnostics
from probabilistic_programming import (
    ProbModel, Normal, Dirichlet, Categorical, Distribution,
    MCMCInference, Trace
)


# ---------------------------------------------------------------------------
# Core Discrete HMM
# ---------------------------------------------------------------------------

class HMM:
    """Discrete Hidden Markov Model with N states and M observation symbols."""

    def __init__(self, n_states, n_obs, seed=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.rng = np.random.RandomState(seed)

        # Initialize with uniform + noise
        self.pi = self._random_stochastic_vector(n_states)
        self.A = self._random_stochastic_matrix(n_states, n_states)
        self.B = self._random_stochastic_matrix(n_states, n_obs)

    def _random_stochastic_vector(self, n):
        v = self.rng.dirichlet(np.ones(n))
        return v

    def _random_stochastic_matrix(self, rows, cols):
        M = np.zeros((rows, cols))
        for i in range(rows):
            M[i] = self.rng.dirichlet(np.ones(cols))
        return M

    def set_params(self, pi=None, A=None, B=None):
        """Set model parameters directly."""
        if pi is not None:
            self.pi = np.array(pi, dtype=float)
        if A is not None:
            self.A = np.array(A, dtype=float)
        if B is not None:
            self.B = np.array(B, dtype=float)

    def forward(self, obs):
        """Forward algorithm. Returns (alpha, log_likelihood).
        alpha[t, i] = P(o1..ot, qt=i | model), scaled for numerical stability.
        """
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        # Init
        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        # Induction
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        log_lik = np.sum(np.log(scales))
        return alpha, log_lik

    def backward(self, obs, scales=None):
        """Backward algorithm. Returns beta (scaled)."""
        T = len(obs)
        N = self.n_states
        beta = np.zeros((T, N))
        beta[T - 1] = 1.0

        if scales is None:
            _, log_lik = self.forward(obs)
            alpha, _ = self.forward(obs)
            # Recompute scales
            scales = np.zeros(T)
            raw_alpha = np.zeros((T, N))
            raw_alpha[0] = self.pi * self.B[:, obs[0]]
            scales[0] = raw_alpha[0].sum()
            if scales[0] == 0:
                scales[0] = 1e-300
            for t in range(1, T):
                raw_alpha[t] = (raw_alpha[t - 1] / scales[t - 1]) @ self.A * self.B[:, obs[t]]
                # Actually just redo forward properly
            alpha, _ = self.forward(obs)
            # Derive scales from forward pass
            return self._backward_with_forward(obs)

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            if scales[t + 1] > 0:
                beta[t] /= scales[t + 1]

        return beta

    def _backward_with_forward(self, obs):
        """Backward pass using scales from forward pass."""
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        beta = np.zeros((T, N))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            if scales[t + 1] > 0:
                beta[t] /= scales[t + 1]

        return beta

    def forward_backward(self, obs):
        """Full forward-backward. Returns (gamma, xi, log_lik).
        gamma[t, i] = P(qt=i | obs, model)
        xi[t, i, j] = P(qt=i, qt+1=j | obs, model)
        """
        T = len(obs)
        N = self.n_states

        # Forward with scales
        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        alpha[0] = self.pi * self.B[:, obs[0]]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        log_lik = np.sum(np.log(scales))

        # Backward
        beta = np.zeros((T, N))
        beta[T - 1] = 1.0

        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])
            if scales[t + 1] > 0:
                beta[t] /= scales[t + 1]

        # Gamma
        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=1, keepdims=True)
        gamma_sums[gamma_sums == 0] = 1e-300
        gamma = gamma / gamma_sums

        # Xi
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            mat = np.outer(alpha[t], self.B[:, obs[t + 1]] * beta[t + 1]) * self.A
            s = mat.sum()
            if s > 0:
                xi[t] = mat / s

        return gamma, xi, log_lik

    def viterbi(self, obs):
        """Viterbi algorithm. Returns (best_path, log_prob)."""
        T = len(obs)
        N = self.n_states

        # Log domain for numerical stability
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_B = np.log(np.maximum(self.B, 1e-300))

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = log_pi + log_B[:, obs[0]]

        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[j, obs[t]]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1])
        log_prob = delta[T - 1, path[T - 1]]

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path.tolist(), log_prob

    def baum_welch(self, sequences, n_iter=100, tol=1e-6):
        """Baum-Welch (EM) algorithm for parameter estimation.
        sequences: list of observation sequences (each a list of ints).
        Returns list of log-likelihoods per iteration.
        """
        log_liks = []

        for iteration in range(n_iter):
            total_log_lik = 0.0

            # Accumulators
            pi_acc = np.zeros(self.n_states)
            A_num = np.zeros((self.n_states, self.n_states))
            A_den = np.zeros(self.n_states)
            B_num = np.zeros((self.n_states, self.n_obs))
            B_den = np.zeros(self.n_states)

            for obs in sequences:
                obs = list(obs)
                gamma, xi, log_lik = self.forward_backward(obs)
                total_log_lik += log_lik

                # Accumulate
                pi_acc += gamma[0]

                if len(obs) > 1:
                    A_num += xi.sum(axis=0)
                    A_den += gamma[:-1].sum(axis=0)

                for t, o in enumerate(obs):
                    B_num[:, o] += gamma[t]
                B_den += gamma.sum(axis=0)

            # Update
            pi_new = pi_acc / pi_acc.sum()
            A_den_safe = np.maximum(A_den, 1e-300)
            A_new = A_num / A_den_safe[:, None]
            B_den_safe = np.maximum(B_den, 1e-300)
            B_new = B_num / B_den_safe[:, None]

            # Normalize rows
            for i in range(self.n_states):
                a_sum = A_new[i].sum()
                if a_sum > 0:
                    A_new[i] /= a_sum
                b_sum = B_new[i].sum()
                if b_sum > 0:
                    B_new[i] /= b_sum

            self.pi = pi_new
            self.A = A_new
            self.B = B_new

            log_liks.append(total_log_lik)

            if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
                break

        return log_liks

    def sample(self, length, seed=None):
        """Generate a sequence of (states, observations) from the model."""
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        states = []
        observations = []

        # Initial state
        state = rng.choice(self.n_states, p=self.pi)
        obs = rng.choice(self.n_obs, p=self.B[state])
        states.append(state)
        observations.append(obs)

        for _ in range(length - 1):
            state = rng.choice(self.n_states, p=self.A[state])
            obs = rng.choice(self.n_obs, p=self.B[state])
            states.append(state)
            observations.append(obs)

        return states, observations

    def log_likelihood(self, obs):
        """Compute log P(obs | model)."""
        _, log_lik = self.forward(obs)
        return log_lik

    def predict(self, obs, n_steps=1):
        """Predict next n_steps observations given obs.
        Returns list of (most_likely_state, observation_probs) per step.
        """
        alpha, _ = self.forward(obs)
        state_dist = alpha[-1]  # Already normalized

        predictions = []
        for _ in range(n_steps):
            state_dist = state_dist @ self.A
            obs_probs = state_dist @ self.B
            best_state = np.argmax(state_dist)
            predictions.append((best_state, obs_probs))

        return predictions

    def stationary_distribution(self):
        """Compute the stationary distribution of the Markov chain."""
        eigenvalues, eigenvectors = np.linalg.eig(self.A.T)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)

    def n_params(self):
        """Number of free parameters."""
        return (self.n_states - 1) + self.n_states * (self.n_states - 1) + self.n_states * (self.n_obs - 1)


# ---------------------------------------------------------------------------
# Gaussian HMM (continuous observations)
# ---------------------------------------------------------------------------

class GaussianHMM:
    """HMM with Gaussian emission distributions."""

    def __init__(self, n_states, n_features=1, covariance_type='diag', seed=None):
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.rng = np.random.RandomState(seed)

        self.pi = self.rng.dirichlet(np.ones(n_states))
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            self.A[i] = self.rng.dirichlet(np.ones(n_states))

        self.means = self.rng.randn(n_states, n_features)
        if covariance_type == 'diag':
            self.covars = np.ones((n_states, n_features))
        elif covariance_type == 'full':
            self.covars = np.array([np.eye(n_features) for _ in range(n_states)])
        else:  # spherical
            self.covars = np.ones(n_states)

    def set_params(self, pi=None, A=None, means=None, covars=None):
        if pi is not None:
            self.pi = np.array(pi, dtype=float)
        if A is not None:
            self.A = np.array(A, dtype=float)
        if means is not None:
            self.means = np.array(means, dtype=float)
            if self.means.ndim == 1:
                self.means = self.means.reshape(-1, 1)
        if covars is not None:
            self.covars = np.array(covars, dtype=float)

    def _emission_prob(self, x, state):
        """P(x | state) under Gaussian emission."""
        x = np.atleast_1d(x)
        mu = self.means[state]
        d = self.n_features

        if self.covariance_type == 'diag':
            var = self.covars[state]
            diff = x - mu
            log_p = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff ** 2 / var))
        elif self.covariance_type == 'full':
            cov = self.covars[state]
            diff = x - mu
            sign, logdet = np.linalg.slogdet(cov)
            log_p = -0.5 * (d * np.log(2 * np.pi) + logdet + diff @ np.linalg.solve(cov, diff))
        else:  # spherical
            var = self.covars[state]
            diff = x - mu
            log_p = -0.5 * (d * np.log(2 * np.pi) + d * np.log(var) + np.sum(diff ** 2) / var)

        return np.exp(log_p)

    def _emission_matrix(self, obs):
        """Compute B[t, i] = P(obs[t] | state=i) for all t, i."""
        T = len(obs)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            for i in range(self.n_states):
                B[t, i] = self._emission_prob(obs[t], i)
        return B

    def forward(self, obs):
        """Forward algorithm for continuous observations."""
        T = len(obs)
        N = self.n_states
        B = self._emission_matrix(obs)

        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        alpha[0] = self.pi * B[0]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        return alpha, np.sum(np.log(scales))

    def forward_backward(self, obs):
        """Forward-backward for continuous obs."""
        T = len(obs)
        N = self.n_states
        B = self._emission_matrix(obs)

        alpha = np.zeros((T, N))
        scales = np.zeros(T)
        alpha[0] = self.pi * B[0]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        log_lik = np.sum(np.log(scales))

        beta = np.zeros((T, N))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1])
            if scales[t + 1] > 0:
                beta[t] /= scales[t + 1]

        gamma = alpha * beta
        gamma_sums = gamma.sum(axis=1, keepdims=True)
        gamma_sums[gamma_sums == 0] = 1e-300
        gamma = gamma / gamma_sums

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            mat = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * self.A
            s = mat.sum()
            if s > 0:
                xi[t] = mat / s

        return gamma, xi, log_lik

    def viterbi(self, obs):
        """Viterbi for continuous observations."""
        T = len(obs)
        N = self.n_states
        B = self._emission_matrix(obs)

        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = log_pi + np.log(np.maximum(B[0], 1e-300))

        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + np.log(max(B[t, j], 1e-300))

        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1])
        log_prob = delta[T - 1, path[T - 1]]

        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path.tolist(), log_prob

    def fit(self, sequences, n_iter=100, tol=1e-6):
        """Baum-Welch EM for Gaussian HMM.
        sequences: list of arrays, each shape (T, n_features).
        """
        log_liks = []

        for iteration in range(n_iter):
            total_log_lik = 0.0
            pi_acc = np.zeros(self.n_states)
            A_num = np.zeros_like(self.A)
            A_den = np.zeros(self.n_states)

            # For Gaussian updates
            mean_num = np.zeros_like(self.means)
            mean_den = np.zeros(self.n_states)
            cov_acc = [np.zeros_like(self.means[0]) if self.covariance_type == 'diag'
                       else np.zeros((self.n_features, self.n_features))
                       for _ in range(self.n_states)]

            for seq in sequences:
                obs = np.atleast_2d(seq)
                if obs.shape[1] != self.n_features and obs.shape[0] == self.n_features:
                    obs = obs.T

                gamma, xi, log_lik = self.forward_backward(obs)
                total_log_lik += log_lik
                T = len(obs)

                pi_acc += gamma[0]
                if T > 1:
                    A_num += xi.sum(axis=0)
                    A_den += gamma[:-1].sum(axis=0)

                for i in range(self.n_states):
                    w = gamma[:, i]
                    w_sum = w.sum()
                    mean_den[i] += w_sum
                    mean_num[i] += (w[:, None] * obs).sum(axis=0)

                    if w_sum > 0:
                        diff = obs - self.means[i]
                        if self.covariance_type == 'diag':
                            cov_acc[i] += (w[:, None] * diff ** 2).sum(axis=0)
                        elif self.covariance_type == 'full':
                            for t in range(T):
                                cov_acc[i] += w[t] * np.outer(diff[t], diff[t])
                        else:  # spherical
                            cov_acc[i] += (w * np.sum(diff ** 2, axis=1)).sum()

            # M-step
            self.pi = pi_acc / pi_acc.sum()
            A_den_safe = np.maximum(A_den, 1e-300)
            self.A = A_num / A_den_safe[:, None]
            for i in range(self.n_states):
                a_sum = self.A[i].sum()
                if a_sum > 0:
                    self.A[i] /= a_sum

            mean_den_safe = np.maximum(mean_den, 1e-300)
            self.means = mean_num / mean_den_safe[:, None]

            for i in range(self.n_states):
                if mean_den[i] > 0:
                    if self.covariance_type == 'diag':
                        self.covars[i] = cov_acc[i] / mean_den[i] + 1e-6
                    elif self.covariance_type == 'full':
                        self.covars[i] = cov_acc[i] / mean_den[i] + 1e-6 * np.eye(self.n_features)
                    else:
                        if isinstance(cov_acc[i], np.ndarray):
                            self.covars[i] = cov_acc[i].sum() / (mean_den[i] * self.n_features) + 1e-6
                        else:
                            self.covars[i] = cov_acc[i] / (mean_den[i] * self.n_features) + 1e-6

            log_liks.append(total_log_lik)
            if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
                break

        return log_liks

    def sample(self, length, seed=None):
        """Generate (states, observations) from Gaussian HMM."""
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        states = []
        observations = []

        state = rng.choice(self.n_states, p=self.pi)

        for _ in range(length):
            states.append(state)
            mu = self.means[state]
            if self.covariance_type == 'diag':
                obs = mu + rng.randn(self.n_features) * np.sqrt(self.covars[state])
            elif self.covariance_type == 'full':
                obs = rng.multivariate_normal(mu, self.covars[state])
            else:
                obs = mu + rng.randn(self.n_features) * np.sqrt(self.covars[state])
            observations.append(obs)

            state = rng.choice(self.n_states, p=self.A[state])

        return states, np.array(observations)

    def log_likelihood(self, obs):
        _, ll = self.forward(obs)
        return ll

    def score(self, sequences):
        """Average log-likelihood per sequence."""
        total = 0.0
        for seq in sequences:
            _, ll = self.forward(np.atleast_2d(seq))
            total += ll
        return total / len(sequences)


# ---------------------------------------------------------------------------
# GMM-HMM (Gaussian Mixture emissions)
# ---------------------------------------------------------------------------

class GMMHMM:
    """HMM with Gaussian Mixture Model emissions."""

    def __init__(self, n_states, n_mix, n_features=1, seed=None):
        self.n_states = n_states
        self.n_mix = n_mix
        self.n_features = n_features
        self.rng = np.random.RandomState(seed)

        self.pi = self.rng.dirichlet(np.ones(n_states))
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            self.A[i] = self.rng.dirichlet(np.ones(n_states))

        # Per-state GMM params
        self.weights = np.zeros((n_states, n_mix))
        self.means = np.zeros((n_states, n_mix, n_features))
        self.covars = np.zeros((n_states, n_mix, n_features))

        for i in range(n_states):
            self.weights[i] = self.rng.dirichlet(np.ones(n_mix))
            self.means[i] = self.rng.randn(n_mix, n_features) * (i + 1)
            self.covars[i] = np.ones((n_mix, n_features))

    def set_params(self, pi=None, A=None, weights=None, means=None, covars=None):
        if pi is not None:
            self.pi = np.array(pi, dtype=float)
        if A is not None:
            self.A = np.array(A, dtype=float)
        if weights is not None:
            self.weights = np.array(weights, dtype=float)
        if means is not None:
            self.means = np.array(means, dtype=float)
        if covars is not None:
            self.covars = np.array(covars, dtype=float)

    def _gmm_prob(self, x, state):
        """P(x | state) under GMM emission."""
        x = np.atleast_1d(x)
        prob = 0.0
        for k in range(self.n_mix):
            mu = self.means[state, k]
            var = self.covars[state, k]
            diff = x - mu
            d = self.n_features
            log_p = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff ** 2 / var))
            prob += self.weights[state, k] * np.exp(log_p)
        return prob

    def _emission_matrix(self, obs):
        T = len(obs)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            for i in range(self.n_states):
                B[t, i] = self._gmm_prob(obs[t], i)
        return B

    def forward(self, obs):
        T = len(obs)
        N = self.n_states
        B = self._emission_matrix(obs)

        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        alpha[0] = self.pi * B[0]
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        return alpha, np.sum(np.log(scales))

    def viterbi(self, obs):
        T = len(obs)
        N = self.n_states
        B = self._emission_matrix(obs)

        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        delta[0] = log_pi + np.log(np.maximum(B[0], 1e-300))
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + np.log(max(B[t, j], 1e-300))

        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1])
        log_prob = delta[T - 1, path[T - 1]]
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path.tolist(), log_prob

    def fit(self, sequences, n_iter=100, tol=1e-6):
        """EM for GMM-HMM."""
        log_liks = []

        for iteration in range(n_iter):
            total_log_lik = 0.0
            pi_acc = np.zeros(self.n_states)
            A_num = np.zeros_like(self.A)
            A_den = np.zeros(self.n_states)

            # GMM accumulators per state, per mixture
            w_acc = np.zeros_like(self.weights)
            m_num = np.zeros_like(self.means)
            m_den = np.zeros_like(self.weights)
            c_acc = np.zeros_like(self.covars)

            for seq in sequences:
                obs = np.atleast_2d(seq)
                if obs.ndim == 1:
                    obs = obs.reshape(-1, 1)
                T = len(obs)
                B = self._emission_matrix(obs)

                # Forward
                alpha = np.zeros((T, self.n_states))
                scales = np.zeros(T)
                alpha[0] = self.pi * B[0]
                scales[0] = alpha[0].sum()
                if scales[0] == 0:
                    scales[0] = 1e-300
                alpha[0] /= scales[0]
                for t in range(1, T):
                    alpha[t] = (alpha[t - 1] @ self.A) * B[t]
                    scales[t] = alpha[t].sum()
                    if scales[t] == 0:
                        scales[t] = 1e-300
                    alpha[t] /= scales[t]
                total_log_lik += np.sum(np.log(scales))

                # Backward
                beta = np.zeros((T, self.n_states))
                beta[T - 1] = 1.0
                for t in range(T - 2, -1, -1):
                    beta[t] = self.A @ (B[t + 1] * beta[t + 1])
                    if scales[t + 1] > 0:
                        beta[t] /= scales[t + 1]

                gamma = alpha * beta
                gs = gamma.sum(axis=1, keepdims=True)
                gs[gs == 0] = 1e-300
                gamma /= gs

                xi = np.zeros((T - 1, self.n_states, self.n_states))
                for t in range(T - 1):
                    mat = np.outer(alpha[t], B[t + 1] * beta[t + 1]) * self.A
                    s = mat.sum()
                    if s > 0:
                        xi[t] = mat / s

                pi_acc += gamma[0]
                if T > 1:
                    A_num += xi.sum(axis=0)
                    A_den += gamma[:-1].sum(axis=0)

                # Component responsibilities
                for t in range(T):
                    x = obs[t]
                    for i in range(self.n_states):
                        comp_probs = np.zeros(self.n_mix)
                        for k in range(self.n_mix):
                            mu = self.means[i, k]
                            var = self.covars[i, k]
                            diff = x - mu
                            d = self.n_features
                            log_p = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff ** 2 / var))
                            comp_probs[k] = self.weights[i, k] * np.exp(log_p)
                        cp_sum = comp_probs.sum()
                        if cp_sum > 0:
                            comp_probs /= cp_sum

                        for k in range(self.n_mix):
                            resp = gamma[t, i] * comp_probs[k]
                            w_acc[i, k] += resp
                            m_num[i, k] += resp * x
                            m_den[i, k] += resp
                            c_acc[i, k] += resp * (x - self.means[i, k]) ** 2

            # M-step
            self.pi = pi_acc / pi_acc.sum()
            A_den_safe = np.maximum(A_den, 1e-300)
            self.A = A_num / A_den_safe[:, None]
            for i in range(self.n_states):
                a_sum = self.A[i].sum()
                if a_sum > 0:
                    self.A[i] /= a_sum

            for i in range(self.n_states):
                ws = w_acc[i].sum()
                if ws > 0:
                    self.weights[i] = w_acc[i] / ws
                for k in range(self.n_mix):
                    if m_den[i, k] > 0:
                        self.means[i, k] = m_num[i, k] / m_den[i, k]
                        self.covars[i, k] = c_acc[i, k] / m_den[i, k] + 1e-6

            log_liks.append(total_log_lik)
            if len(log_liks) > 1 and abs(log_liks[-1] - log_liks[-2]) < tol:
                break

        return log_liks

    def sample(self, length, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        states = []
        observations = []
        state = rng.choice(self.n_states, p=self.pi)

        for _ in range(length):
            states.append(state)
            k = rng.choice(self.n_mix, p=self.weights[state])
            mu = self.means[state, k]
            var = self.covars[state, k]
            obs = mu + rng.randn(self.n_features) * np.sqrt(var)
            observations.append(obs)
            state = rng.choice(self.n_states, p=self.A[state])

        return states, np.array(observations)


# ---------------------------------------------------------------------------
# Bayesian HMM (composing C156 PPL)
# ---------------------------------------------------------------------------

class BayesianHMM:
    """Bayesian HMM with posterior inference over parameters via C156 PPL."""

    def __init__(self, n_states, n_obs, prior_alpha=1.0, seed=42):
        self.n_states = n_states
        self.n_obs = n_obs
        self.prior_alpha = prior_alpha
        self.seed = seed
        self.trace = None
        self.point_estimate = None

    def fit(self, sequences, n_samples=500, method='mcmc'):
        """Fit Bayesian HMM using MCMC or VI.
        Returns posterior trace.
        """
        # Use Gibbs-style updates: sample parameters, then run forward-backward
        # For simplicity, use MH over the flattened parameter space
        rng = np.random.RandomState(self.seed)
        N = self.n_states
        M = self.n_obs
        alpha = self.prior_alpha

        # Flatten: pi(N) + A(N*N) + B(N*M)
        def unflatten(theta):
            idx = 0
            pi_raw = theta[idx:idx + N]
            pi = np.exp(pi_raw) / np.exp(pi_raw).sum()
            idx += N
            A = np.zeros((N, N))
            for i in range(N):
                row_raw = theta[idx:idx + N]
                A[i] = np.exp(row_raw) / np.exp(row_raw).sum()
                idx += N
            B = np.zeros((N, M))
            for i in range(N):
                row_raw = theta[idx:idx + M]
                B[i] = np.exp(row_raw) / np.exp(row_raw).sum()
                idx += M
            return pi, A, B

        dim = N + N * N + N * M

        def log_prob(theta):
            pi, A, B = unflatten(theta)

            # Dirichlet prior (uniform)
            lp = 0.0
            lp += (alpha - 1) * np.sum(np.log(np.maximum(pi, 1e-300)))
            for i in range(N):
                lp += (alpha - 1) * np.sum(np.log(np.maximum(A[i], 1e-300)))
                lp += (alpha - 1) * np.sum(np.log(np.maximum(B[i], 1e-300)))

            # Likelihood via forward algorithm
            hmm = HMM(N, M)
            hmm.set_params(pi=pi, A=A, B=B)
            for seq in sequences:
                _, ll = hmm.forward(list(seq))
                lp += ll

            return lp

        # Use MH sampler from C153
        sampler = MetropolisHastings(log_prob, dim, seed=self.seed)
        samples, accept_rate = sampler.sample(
            n_samples=n_samples, burn_in=200, thin=2,
            proposal='random_walk'
        )

        # Extract posterior statistics
        posterior_params = []
        for s in samples:
            pi, A, B = unflatten(s)
            posterior_params.append({'pi': pi, 'A': A, 'B': B})

        # Point estimate (posterior mean)
        pi_mean = np.mean([p['pi'] for p in posterior_params], axis=0)
        A_mean = np.mean([p['A'] for p in posterior_params], axis=0)
        B_mean = np.mean([p['B'] for p in posterior_params], axis=0)

        # Normalize
        pi_mean /= pi_mean.sum()
        for i in range(N):
            A_mean[i] /= A_mean[i].sum()
            B_mean[i] /= B_mean[i].sum()

        self.point_estimate = {'pi': pi_mean, 'A': A_mean, 'B': B_mean}
        self.trace = posterior_params
        self.accept_rate = accept_rate

        return self

    def get_model(self):
        """Return an HMM with posterior mean parameters."""
        if self.point_estimate is None:
            raise ValueError("Must call fit() first")
        hmm = HMM(self.n_states, self.n_obs)
        hmm.set_params(**self.point_estimate)
        return hmm

    def posterior_predictive(self, length=20, n_samples=50):
        """Generate sequences from posterior predictive distribution."""
        if self.trace is None:
            raise ValueError("Must call fit() first")
        rng = np.random.RandomState(self.seed + 1)
        results = []
        indices = rng.choice(len(self.trace), size=min(n_samples, len(self.trace)), replace=False)
        for idx in indices:
            params = self.trace[idx]
            hmm = HMM(self.n_states, self.n_obs, seed=int(rng.randint(100000)))
            hmm.set_params(**params)
            states, obs = hmm.sample(length)
            results.append({'states': states, 'observations': obs})
        return results


# ---------------------------------------------------------------------------
# HMM Classifier (sequence classification)
# ---------------------------------------------------------------------------

class HMMClassifier:
    """Classify sequences by training one HMM per class."""

    def __init__(self, n_states=3, n_obs=None, n_iter=50, seed=None):
        self.n_states = n_states
        self.n_obs = n_obs
        self.n_iter = n_iter
        self.seed = seed
        self.models = {}
        self.classes = []

    def fit(self, sequences_by_class):
        """Train one HMM per class.
        sequences_by_class: dict {class_label: [list of observation sequences]}
        """
        self.classes = sorted(sequences_by_class.keys())

        if self.n_obs is None:
            self.n_obs = max(max(max(s) for s in seqs) for seqs in sequences_by_class.values()) + 1

        for cls in self.classes:
            hmm = HMM(self.n_states, self.n_obs, seed=self.seed)
            hmm.baum_welch(sequences_by_class[cls], n_iter=self.n_iter)
            self.models[cls] = hmm

        return self

    def predict(self, obs):
        """Classify a single observation sequence."""
        best_class = None
        best_ll = -np.inf
        for cls in self.classes:
            ll = self.models[cls].log_likelihood(obs)
            if ll > best_ll:
                best_ll = ll
                best_class = cls
        return best_class

    def predict_proba(self, obs):
        """Return class probabilities for a sequence."""
        log_liks = {}
        for cls in self.classes:
            log_liks[cls] = self.models[cls].log_likelihood(obs)

        # Softmax
        max_ll = max(log_liks.values())
        probs = {cls: np.exp(ll - max_ll) for cls, ll in log_liks.items()}
        total = sum(probs.values())
        return {cls: p / total for cls, p in probs.items()}

    def score(self, test_data):
        """Accuracy on test data. test_data: list of (obs_seq, true_class)."""
        correct = sum(1 for obs, cls in test_data if self.predict(obs) == cls)
        return correct / len(test_data)


# ---------------------------------------------------------------------------
# HMM Segmenter (change-point detection)
# ---------------------------------------------------------------------------

class HMMSegmenter:
    """Segment a sequence into regions using HMM state assignments."""

    def __init__(self, n_segments, n_features=1, seed=None):
        self.n_segments = n_segments
        self.n_features = n_features
        self.seed = seed
        self.model = GaussianHMM(n_segments, n_features, covariance_type='diag', seed=seed)

    def fit(self, data, n_iter=100):
        """Fit the segmentation model.
        data: array of shape (T, n_features) or (T,).
        """
        data = np.atleast_2d(data)
        if data.shape[0] == self.n_features and data.shape[1] != self.n_features:
            data = data.T
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Initialize means spread across data range
        for i in range(self.n_segments):
            frac = (i + 0.5) / self.n_segments
            idx = int(frac * len(data))
            self.model.means[i] = data[min(idx, len(data) - 1)]

        # Encourage staying in same state (left-to-right bias)
        self.model.A = np.full((self.n_segments, self.n_segments), 0.05 / (self.n_segments - 1) if self.n_segments > 1 else 1.0)
        for i in range(self.n_segments):
            self.model.A[i, i] = 0.95

        self.model.fit([data], n_iter=n_iter)
        return self

    def segment(self, data):
        """Return segment labels for each time step."""
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[0] == self.n_features and data.shape[1] != self.n_features:
            data = data.T

        path, _ = self.model.viterbi(data)
        return path

    def change_points(self, data):
        """Return indices where segment changes occur."""
        labels = self.segment(data)
        changes = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
        return changes

    def segment_summary(self, data):
        """Return summary of each segment."""
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[0] == self.n_features and data.shape[1] != self.n_features:
            data = data.T

        labels = self.segment(data)
        segments = []
        start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                seg_data = data[start:i]
                segments.append({
                    'start': start,
                    'end': i - 1,
                    'label': labels[start],
                    'length': i - start,
                    'mean': seg_data.mean(axis=0).tolist(),
                    'std': seg_data.std(axis=0).tolist() if len(seg_data) > 1 else [0.0] * self.n_features
                })
                start = i

        return segments


# ---------------------------------------------------------------------------
# Autoregressive HMM
# ---------------------------------------------------------------------------

class ARHiddenMarkovModel:
    """Autoregressive HMM: emission depends on previous observation.
    P(x_t | s_t, x_{t-1}) = N(a[s_t] * x_{t-1} + b[s_t], sigma[s_t]^2)
    """

    def __init__(self, n_states, order=1, seed=None):
        self.n_states = n_states
        self.order = order  # AR order (we implement order 1)
        self.rng = np.random.RandomState(seed)

        self.pi = self.rng.dirichlet(np.ones(n_states))
        self.A = np.zeros((n_states, n_states))
        for i in range(n_states):
            self.A[i] = self.rng.dirichlet(np.ones(n_states))

        # AR(1) params per state: x_t = a[s]*x_{t-1} + b[s] + noise
        self.ar_coeffs = self.rng.randn(n_states) * 0.5  # a
        self.ar_intercepts = self.rng.randn(n_states)  # b
        self.ar_variances = np.ones(n_states)  # sigma^2

    def set_params(self, pi=None, A=None, ar_coeffs=None, ar_intercepts=None, ar_variances=None):
        if pi is not None:
            self.pi = np.array(pi, dtype=float)
        if A is not None:
            self.A = np.array(A, dtype=float)
        if ar_coeffs is not None:
            self.ar_coeffs = np.array(ar_coeffs, dtype=float)
        if ar_intercepts is not None:
            self.ar_intercepts = np.array(ar_intercepts, dtype=float)
        if ar_variances is not None:
            self.ar_variances = np.array(ar_variances, dtype=float)

    def _emission_prob(self, x_t, x_prev, state):
        """P(x_t | state, x_prev) under AR(1) Gaussian."""
        mu = self.ar_coeffs[state] * x_prev + self.ar_intercepts[state]
        var = self.ar_variances[state]
        diff = x_t - mu
        return np.exp(-0.5 * diff ** 2 / var) / np.sqrt(2 * np.pi * var)

    def forward(self, obs):
        """Forward algorithm for AR-HMM."""
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N))
        scales = np.zeros(T)

        # t=0: use unconditional emission (just intercept)
        for i in range(N):
            mu = self.ar_intercepts[i]
            var = self.ar_variances[i]
            alpha[0, i] = self.pi[i] * np.exp(-0.5 * (obs[0] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = sum(alpha[t - 1, i] * self.A[i, j] for i in range(N)) * self._emission_prob(obs[t], obs[t - 1], j)
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        return alpha, np.sum(np.log(scales))

    def viterbi(self, obs):
        """Viterbi for AR-HMM."""
        T = len(obs)
        N = self.n_states

        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))

        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        for i in range(N):
            mu = self.ar_intercepts[i]
            var = self.ar_variances[i]
            delta[0, i] = log_pi[i] + (-0.5 * (obs[0] - mu) ** 2 / var - 0.5 * np.log(2 * np.pi * var))

        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                ep = self._emission_prob(obs[t], obs[t - 1], j)
                delta[t, j] = candidates[psi[t, j]] + np.log(max(ep, 1e-300))

        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1])
        log_prob = delta[T - 1, path[T - 1]]
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path.tolist(), log_prob

    def sample(self, length, seed=None):
        """Generate from AR-HMM."""
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        states = []
        observations = []

        state = rng.choice(self.n_states, p=self.pi)
        x = self.ar_intercepts[state] + rng.randn() * np.sqrt(self.ar_variances[state])
        states.append(state)
        observations.append(x)

        for _ in range(length - 1):
            state = rng.choice(self.n_states, p=self.A[state])
            mu = self.ar_coeffs[state] * x + self.ar_intercepts[state]
            x = mu + rng.randn() * np.sqrt(self.ar_variances[state])
            states.append(state)
            observations.append(x)

        return states, np.array(observations)

    def log_likelihood(self, obs):
        _, ll = self.forward(obs)
        return ll


# ---------------------------------------------------------------------------
# Factorial HMM (multiple parallel chains)
# ---------------------------------------------------------------------------

class FactorialHMM:
    """Factorial HMM with K parallel hidden chains.
    Each chain has its own transition matrix.
    Observation depends on all chains combined.
    """

    def __init__(self, n_chains, states_per_chain, n_features=1, seed=None):
        self.n_chains = n_chains
        self.states_per_chain = states_per_chain
        self.n_features = n_features
        self.rng = np.random.RandomState(seed)

        # Per-chain initial distributions and transitions
        self.pis = []
        self.As = []
        for _ in range(n_chains):
            self.pis.append(self.rng.dirichlet(np.ones(states_per_chain)))
            A = np.zeros((states_per_chain, states_per_chain))
            for i in range(states_per_chain):
                A[i] = self.rng.dirichlet(np.ones(states_per_chain))
            self.As.append(A)

        # Emission: weighted combination of chain-specific means
        # W[k] = weight matrix for chain k, shape (states_per_chain, n_features)
        self.chain_means = [self.rng.randn(states_per_chain, n_features) for _ in range(n_chains)]
        self.noise_var = np.ones(n_features)

    def _all_joint_states(self):
        """Enumerate all joint states (Cartesian product)."""
        import itertools
        return list(itertools.product(range(self.states_per_chain), repeat=self.n_chains))

    def _emission_prob(self, x, joint_state):
        """P(x | joint_state) = N(sum of chain means, noise_var)."""
        x = np.atleast_1d(x)
        mu = np.zeros(self.n_features)
        for k, s in enumerate(joint_state):
            mu += self.chain_means[k][s]
        diff = x - mu
        d = self.n_features
        log_p = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(self.noise_var)) + np.sum(diff ** 2 / self.noise_var))
        return np.exp(log_p)

    def forward(self, obs):
        """Exact forward pass over joint state space.
        Note: exponential in n_chains -- practical for small n_chains.
        """
        joint_states = self._all_joint_states()
        n_joint = len(joint_states)
        T = len(obs)

        alpha = np.zeros((T, n_joint))
        scales = np.zeros(T)

        # Joint initial distribution
        for j_idx, js in enumerate(joint_states):
            p = 1.0
            for k, s in enumerate(js):
                p *= self.pis[k][s]
            alpha[0, j_idx] = p * self._emission_prob(obs[0], js)

        scales[0] = alpha[0].sum()
        if scales[0] == 0:
            scales[0] = 1e-300
        alpha[0] /= scales[0]

        # Precompute joint transition matrix
        joint_A = np.zeros((n_joint, n_joint))
        for i_idx, is_ in enumerate(joint_states):
            for j_idx, js in enumerate(joint_states):
                p = 1.0
                for k in range(self.n_chains):
                    p *= self.As[k][is_[k], js[k]]
                joint_A[i_idx, j_idx] = p

        for t in range(1, T):
            for j_idx, js in enumerate(joint_states):
                alpha[t, j_idx] = sum(alpha[t - 1, i_idx] * joint_A[i_idx, j_idx] for i_idx in range(n_joint))
                alpha[t, j_idx] *= self._emission_prob(obs[t], js)
            scales[t] = alpha[t].sum()
            if scales[t] == 0:
                scales[t] = 1e-300
            alpha[t] /= scales[t]

        return alpha, np.sum(np.log(scales)), joint_states

    def viterbi(self, obs):
        """Viterbi over joint state space."""
        joint_states = self._all_joint_states()
        n_joint = len(joint_states)
        T = len(obs)

        # Precompute joint transition log probs
        log_joint_A = np.zeros((n_joint, n_joint))
        for i_idx, is_ in enumerate(joint_states):
            for j_idx, js in enumerate(joint_states):
                p = 0.0
                for k in range(self.n_chains):
                    p += np.log(max(self.As[k][is_[k], js[k]], 1e-300))
                log_joint_A[i_idx, j_idx] = p

        delta = np.zeros((T, n_joint))
        psi = np.zeros((T, n_joint), dtype=int)

        for j_idx, js in enumerate(joint_states):
            log_pi = sum(np.log(max(self.pis[k][js[k]], 1e-300)) for k in range(self.n_chains))
            ep = self._emission_prob(obs[0], js)
            delta[0, j_idx] = log_pi + np.log(max(ep, 1e-300))

        for t in range(1, T):
            for j_idx, js in enumerate(joint_states):
                candidates = delta[t - 1] + log_joint_A[:, j_idx]
                psi[t, j_idx] = np.argmax(candidates)
                ep = self._emission_prob(obs[t], js)
                delta[t, j_idx] = candidates[psi[t, j_idx]] + np.log(max(ep, 1e-300))

        path_idx = np.zeros(T, dtype=int)
        path_idx[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            path_idx[t] = psi[t + 1, path_idx[t + 1]]

        path = [joint_states[i] for i in path_idx]
        log_prob = delta[T - 1, path_idx[T - 1]]
        return path, log_prob

    def sample(self, length, seed=None):
        """Generate from Factorial HMM."""
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        chain_states = []
        observations = []

        # Initial states
        current = []
        for k in range(self.n_chains):
            s = rng.choice(self.states_per_chain, p=self.pis[k])
            current.append(s)

        for _ in range(length):
            chain_states.append(tuple(current))
            mu = np.zeros(self.n_features)
            for k, s in enumerate(current):
                mu += self.chain_means[k][s]
            obs = mu + rng.randn(self.n_features) * np.sqrt(self.noise_var)
            observations.append(obs)

            # Transition each chain independently
            new_current = []
            for k in range(self.n_chains):
                s = rng.choice(self.states_per_chain, p=self.As[k][current[k]])
                new_current.append(s)
            current = new_current

        return chain_states, np.array(observations)


# ---------------------------------------------------------------------------
# HMM Utils (model selection, sequence utilities)
# ---------------------------------------------------------------------------

class HMMUtils:
    """Utility functions for HMM model selection and analysis."""

    @staticmethod
    def aic(hmm, sequences):
        """Akaike Information Criterion."""
        ll = sum(hmm.log_likelihood(seq) for seq in sequences)
        k = hmm.n_params() if hasattr(hmm, 'n_params') else 0
        return 2 * k - 2 * ll

    @staticmethod
    def bic(hmm, sequences):
        """Bayesian Information Criterion."""
        ll = sum(hmm.log_likelihood(seq) for seq in sequences)
        k = hmm.n_params() if hasattr(hmm, 'n_params') else 0
        n = sum(len(s) for s in sequences)
        return k * np.log(n) - 2 * ll

    @staticmethod
    def select_n_states(sequences, n_obs, max_states=10, criterion='bic', seed=None, n_iter=50):
        """Select optimal number of states using information criterion.
        Returns (best_n_states, scores_dict).
        """
        scores = {}
        for n in range(1, max_states + 1):
            hmm = HMM(n, n_obs, seed=seed)
            hmm.baum_welch(sequences, n_iter=n_iter)
            if criterion == 'bic':
                scores[n] = HMMUtils.bic(hmm, sequences)
            else:
                scores[n] = HMMUtils.aic(hmm, sequences)
        best = min(scores, key=scores.get)
        return best, scores

    @staticmethod
    def sequence_entropy(hmm, obs):
        """Entropy of state posterior distribution at each time step."""
        gamma, _, _ = hmm.forward_backward(obs)
        entropy = np.zeros(len(obs))
        for t in range(len(obs)):
            p = gamma[t]
            p = p[p > 0]
            entropy[t] = -np.sum(p * np.log(p))
        return entropy

    @staticmethod
    def state_occupancy(hmm, obs):
        """Expected fraction of time spent in each state."""
        gamma, _, _ = hmm.forward_backward(obs)
        return gamma.mean(axis=0)

    @staticmethod
    def most_likely_state_sequence(hmm, obs):
        """Viterbi decoding."""
        path, _ = hmm.viterbi(obs)
        return path

    @staticmethod
    def compare_models(models, sequences, criterion='bic'):
        """Compare multiple HMMs on the same data.
        models: list of (name, hmm) tuples.
        Returns sorted list of (name, score).
        """
        results = []
        for name, hmm in models:
            if criterion == 'bic':
                score = HMMUtils.bic(hmm, sequences)
            else:
                score = HMMUtils.aic(hmm, sequences)
            results.append((name, score))
        results.sort(key=lambda x: x[1])
        return results

    @staticmethod
    def kl_divergence_hmm(hmm1, hmm2, n_sequences=100, seq_length=50, seed=42):
        """Estimate KL(hmm1 || hmm2) via Monte Carlo sampling."""
        rng = np.random.RandomState(seed)
        kl = 0.0
        for _ in range(n_sequences):
            _, obs = hmm1.sample(seq_length, seed=int(rng.randint(100000)))
            ll1 = hmm1.log_likelihood(obs)
            ll2 = hmm2.log_likelihood(obs)
            kl += (ll1 - ll2)
        return kl / n_sequences

    @staticmethod
    def merge_states(hmm, state1, state2):
        """Merge two states in an HMM, returning a new HMM with n_states-1 states."""
        if state1 > state2:
            state1, state2 = state2, state1

        N = hmm.n_states
        M = hmm.n_obs
        new_hmm = HMM(N - 1, M)

        # Merge pi
        new_pi = np.delete(hmm.pi, state2)
        new_pi[state1] += hmm.pi[state2]
        new_hmm.pi = new_pi / new_pi.sum()

        # Map old states to new
        mapping = list(range(N))
        mapping[state2] = state1
        for i in range(state2 + 1, N):
            mapping[i] = i - 1

        # Merge A and B
        new_A = np.zeros((N - 1, N - 1))
        new_B = np.zeros((N - 1, M))
        w1 = hmm.pi[state1] / (hmm.pi[state1] + hmm.pi[state2] + 1e-300)
        w2 = 1 - w1

        for i in range(N):
            ni = mapping[i]
            if i == state2:
                continue
            for j in range(N):
                nj = mapping[j]
                if i == state1:
                    new_A[ni, nj] += w1 * hmm.A[state1, j] + w2 * hmm.A[state2, j]
                else:
                    new_A[ni, nj] += hmm.A[i, j]

        for i in range(N):
            ni = mapping[i]
            if i == state2:
                continue
            if i == state1:
                new_B[ni] = w1 * hmm.B[state1] + w2 * hmm.B[state2]
            else:
                new_B[ni] = hmm.B[i]

        # Normalize
        for i in range(N - 1):
            a_sum = new_A[i].sum()
            if a_sum > 0:
                new_A[i] /= a_sum
            b_sum = new_B[i].sum()
            if b_sum > 0:
                new_B[i] /= b_sum

        new_hmm.set_params(pi=new_hmm.pi, A=new_A, B=new_B)
        return new_hmm
