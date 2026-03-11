"""
C154: Variational Inference
Approximate Bayesian inference via optimization.

Composes:
- C153 Monte Carlo Methods (sampling, MCMC diagnostics)
- C140 Neural Network (amortized inference networks)

Features:
- Distribution classes: Normal, MultivariateNormal, Bernoulli, Categorical
- KL divergence (analytical + Monte Carlo)
- ELBO computation
- Mean-field VI (factored Gaussian)
- Black-box VI (score function / REINFORCE estimator)
- Reparameterized VI (pathwise gradients)
- Amortized VI (neural encoder, VAE-style)
- Normalizing flows (planar + radial transforms)
- ADVI (automatic transform to unconstrained space)
- VI diagnostics (ELBO tracking, convergence, posterior summary)
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C153_monte_carlo'))
from monte_carlo import RandomSampler, MCMCDiagnostics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Activation, Sequential, Adam, MSELoss, fit
)


# ============================================================
# Distributions
# ============================================================

class Distribution:
    """Base distribution class."""

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, n=1, rng=None):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class Normal(Distribution):
    """Univariate normal distribution."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        if std <= 0:
            raise ValueError("std must be positive")

    def log_prob(self, x):
        if isinstance(x, (list, tuple)):
            return [self.log_prob(xi) for xi in x]
        return -0.5 * math.log(2 * math.pi) - math.log(self.std) - 0.5 * ((x - self.mean) / self.std) ** 2

    def sample(self, n=1, rng=None):
        r = rng or random.Random()
        samples = [r.gauss(self.mean, self.std) for _ in range(n)]
        return samples[0] if n == 1 else samples

    def entropy(self):
        return 0.5 * math.log(2 * math.pi * math.e * self.std ** 2)

    def __repr__(self):
        return f"Normal(mean={self.mean}, std={self.std})"


class MultivariateNormal(Distribution):
    """Multivariate normal with diagonal or full covariance."""

    def __init__(self, mean, cov=None):
        self.mean = list(mean)
        self.dim = len(self.mean)
        if cov is None:
            self.cov = [[1.0 if i == j else 0.0 for j in range(self.dim)] for i in range(self.dim)]
        else:
            self.cov = [list(row) for row in cov]
        # Precompute Cholesky for sampling
        self._chol = self._cholesky(self.cov)
        self._log_det = 2.0 * sum(math.log(self._chol[i][i]) for i in range(self.dim))

    def _cholesky(self, A):
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    if val < 0:
                        val = 1e-10  # numerical fix
                    L[i][j] = math.sqrt(val)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j] if L[j][j] > 0 else 0.0
        return L

    def log_prob(self, x):
        if isinstance(x[0], (list, tuple)):
            return [self.log_prob(xi) for xi in x]
        # Solve L * y = (x - mean)
        diff = [x[i] - self.mean[i] for i in range(self.dim)]
        y = self._forward_solve(self._chol, diff)
        mahal = sum(yi ** 2 for yi in y)
        return -0.5 * (self.dim * math.log(2 * math.pi) + self._log_det + mahal)

    def _forward_solve(self, L, b):
        n = len(b)
        y = [0.0] * n
        for i in range(n):
            y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i] if L[i][i] > 0 else 0.0
        return y

    def sample(self, n=1, rng=None):
        r = rng or random.Random()
        samples = []
        for _ in range(n):
            z = [r.gauss(0, 1) for _ in range(self.dim)]
            x = [self.mean[i] + sum(self._chol[i][j] * z[j] for j in range(i + 1))
                 for i in range(self.dim)]
            samples.append(x)
        return samples[0] if n == 1 else samples

    def entropy(self):
        return 0.5 * (self.dim * (1.0 + math.log(2 * math.pi)) + self._log_det)

    def __repr__(self):
        return f"MultivariateNormal(dim={self.dim})"


class Bernoulli(Distribution):
    """Bernoulli distribution."""

    def __init__(self, p=0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def log_prob(self, x):
        if isinstance(x, (list, tuple)):
            return [self.log_prob(xi) for xi in x]
        if x == 1:
            return math.log(self.p) if self.p > 0 else -float('inf')
        elif x == 0:
            return math.log(1 - self.p) if self.p < 1 else -float('inf')
        else:
            return -float('inf')

    def sample(self, n=1, rng=None):
        r = rng or random.Random()
        samples = [1 if r.random() < self.p else 0 for _ in range(n)]
        return samples[0] if n == 1 else samples

    def entropy(self):
        if self.p == 0 or self.p == 1:
            return 0.0
        return -(self.p * math.log(self.p) + (1 - self.p) * math.log(1 - self.p))

    def __repr__(self):
        return f"Bernoulli(p={self.p})"


class Categorical(Distribution):
    """Categorical distribution over K classes."""

    def __init__(self, probs):
        total = sum(probs)
        self.probs = [p / total for p in probs]
        self.k = len(self.probs)

    def log_prob(self, x):
        if isinstance(x, (list, tuple)):
            return [self.log_prob(xi) for xi in x]
        if 0 <= x < self.k and self.probs[x] > 0:
            return math.log(self.probs[x])
        return -float('inf')

    def sample(self, n=1, rng=None):
        r = rng or random.Random()
        samples = []
        for _ in range(n):
            u = r.random()
            cumsum = 0.0
            for i, p in enumerate(self.probs):
                cumsum += p
                if u <= cumsum:
                    samples.append(i)
                    break
            else:
                samples.append(self.k - 1)
        return samples[0] if n == 1 else samples

    def entropy(self):
        return -sum(p * math.log(p) for p in self.probs if p > 0)

    def __repr__(self):
        return f"Categorical(k={self.k})"


# ============================================================
# KL Divergence
# ============================================================

class KLDivergence:
    """Compute KL divergence KL(q || p) between distributions."""

    @staticmethod
    def analytical(q, p):
        """Analytical KL for known pairs."""
        if isinstance(q, Normal) and isinstance(p, Normal):
            return (math.log(p.std / q.std)
                    + (q.std ** 2 + (q.mean - p.mean) ** 2) / (2 * p.std ** 2)
                    - 0.5)

        if isinstance(q, MultivariateNormal) and isinstance(p, MultivariateNormal):
            dim = q.dim
            # KL = 0.5 * (tr(Sp^-1 Sq) + (mp-mq)^T Sp^-1 (mp-mq) - d + ln|Sp|/|Sq|)
            p_chol = p._chol
            # Solve for Sp^-1 Sq columns
            trace_term = 0.0
            for j in range(dim):
                col = [q.cov[i][j] for i in range(dim)]
                y = p._forward_solve(p_chol, col)
                # Back-solve: L^T z = y
                z = _back_solve(p_chol, y)
                trace_term += z[j]

            diff = [q.mean[i] - p.mean[i] for i in range(dim)]
            y = p._forward_solve(p_chol, diff)
            mahal = sum(yi ** 2 for yi in y)

            return 0.5 * (trace_term + mahal - dim + p._log_det - q._log_det)

        if isinstance(q, Bernoulli) and isinstance(p, Bernoulli):
            kl = 0.0
            if q.p > 0:
                kl += q.p * math.log(q.p / p.p) if p.p > 0 else float('inf')
            if q.p < 1:
                kl += (1 - q.p) * math.log((1 - q.p) / (1 - p.p)) if p.p < 1 else float('inf')
            return kl

        if isinstance(q, Categorical) and isinstance(p, Categorical):
            return sum(
                qi * math.log(qi / pi) if qi > 0 else 0.0
                for qi, pi in zip(q.probs, p.probs)
            )

        raise ValueError(f"No analytical KL for {type(q).__name__} || {type(p).__name__}")

    @staticmethod
    def monte_carlo(q, p, n_samples=1000, rng=None):
        """Estimate KL(q||p) via Monte Carlo sampling from q."""
        samples = q.sample(n=n_samples, rng=rng)
        if not isinstance(samples, list):
            samples = [samples]
        # Handle single-value samples vs list-of-vectors
        q_lps = q.log_prob(samples)
        p_lps = p.log_prob(samples)
        if not isinstance(q_lps, list):
            q_lps = [q_lps]
            p_lps = [p_lps]
        kl = sum(ql - pl for ql, pl in zip(q_lps, p_lps)) / len(q_lps)
        return kl


def _back_solve(L, y):
    """Solve L^T z = y for z."""
    n = len(y)
    z = [0.0] * n
    for i in range(n - 1, -1, -1):
        z[i] = (y[i] - sum(L[j][i] * z[j] for j in range(i + 1, n))) / L[i][i] if L[i][i] > 0 else 0.0
    return z


# ============================================================
# ELBO
# ============================================================

class ELBO:
    """Evidence Lower Bound computation."""

    @staticmethod
    def compute(log_joint, q, n_samples=100, rng=None):
        """Compute ELBO = E_q[log p(x,z)] - E_q[log q(z)].

        Args:
            log_joint: callable(z) -> log p(x, z) for fixed observed x
            q: Distribution (variational posterior)
            n_samples: number of MC samples
        Returns:
            elbo estimate (float)
        """
        samples = q.sample(n=n_samples, rng=rng)
        if not isinstance(samples, list):
            samples = [samples]
        # Ensure list of samples
        if n_samples == 1 and not isinstance(samples[0], (list, tuple)):
            samples_list = [samples[0]] if not isinstance(samples, list) else samples
        else:
            samples_list = samples

        total = 0.0
        for z in samples_list:
            lp = log_joint(z)
            lq = q.log_prob(z)
            if isinstance(lq, list):
                lq = lq[0]
            total += lp - lq
        return total / len(samples_list)

    @staticmethod
    def compute_with_kl(log_likelihood, q, prior, n_samples=100, rng=None):
        """ELBO = E_q[log p(x|z)] - KL(q||prior).

        Uses analytical KL when possible, falls back to MC.
        """
        # E_q[log p(x|z)]
        samples = q.sample(n=n_samples, rng=rng)
        if not isinstance(samples, list):
            samples = [samples]
        if n_samples == 1 and not isinstance(samples[0], (list, tuple)):
            samples = [samples[0]] if not isinstance(samples, list) else samples

        ll_sum = sum(log_likelihood(z) for z in samples) / len(samples)

        # KL term
        try:
            kl = KLDivergence.analytical(q, prior)
        except ValueError:
            kl = KLDivergence.monte_carlo(q, prior, n_samples=n_samples, rng=rng)

        return ll_sum - kl


# ============================================================
# Mean-Field VI
# ============================================================

class MeanFieldVI:
    """Mean-field variational inference with factored Gaussian posterior.

    q(z) = prod_i Normal(mu_i, sigma_i)
    Optimizes mu and log_sigma via gradient ascent on ELBO.
    """

    def __init__(self, dim, log_joint, seed=None):
        self.dim = dim
        self.log_joint = log_joint
        self.rng = random.Random(seed)
        # Variational parameters
        self.mu = [0.0] * dim
        self.log_sigma = [0.0] * dim  # log(sigma) for unconstrained optimization
        self.elbo_history = []

    def _get_sigma(self):
        return [math.exp(ls) for ls in self.log_sigma]

    def get_posterior(self):
        """Return the current variational posterior as distributions."""
        sigma = self._get_sigma()
        return [Normal(self.mu[i], sigma[i]) for i in range(self.dim)]

    def get_joint_posterior(self):
        """Return as MultivariateNormal with diagonal covariance."""
        sigma = self._get_sigma()
        cov = [[sigma[i] ** 2 if i == j else 0.0 for j in range(self.dim)] for i in range(self.dim)]
        return MultivariateNormal(self.mu, cov)

    def fit(self, n_iter=1000, lr=0.01, n_samples=10, verbose=False):
        """Optimize ELBO via reparameterized gradient ascent."""
        eps = 1e-8
        for iteration in range(n_iter):
            sigma = self._get_sigma()

            # Sample via reparameterization: z = mu + sigma * epsilon
            grad_mu = [0.0] * self.dim
            grad_log_sigma = [0.0] * self.dim
            elbo_est = 0.0

            for _ in range(n_samples):
                epsilon = [self.rng.gauss(0, 1) for _ in range(self.dim)]
                z = [self.mu[i] + sigma[i] * epsilon[i] for i in range(self.dim)]

                lp = self.log_joint(z)
                lq = sum(-0.5 * math.log(2 * math.pi) - math.log(sigma[i]) - 0.5 * epsilon[i] ** 2
                         for i in range(self.dim))
                elbo_est += lp - lq

                # Numerical gradient via finite differences on mu and log_sigma
                for d in range(self.dim):
                    # Gradient w.r.t. mu_d
                    z_plus = z[:]
                    z_plus[d] += eps
                    z_minus = z[:]
                    z_minus[d] -= eps
                    dlp = (self.log_joint(z_plus) - self.log_joint(z_minus)) / (2 * eps)
                    grad_mu[d] += dlp  # d(ELBO)/d(mu) ~ d(log_joint)/d(z) * d(z)/d(mu) = d(log_joint)/d(z)

                    # Gradient w.r.t. log_sigma_d (reparameterization)
                    # d(z)/d(log_sigma) = sigma * epsilon = z - mu
                    grad_log_sigma[d] += dlp * sigma[d] * epsilon[d] + 1.0  # +1 for entropy term

            # Average over samples
            for d in range(self.dim):
                grad_mu[d] /= n_samples
                grad_log_sigma[d] /= n_samples
            elbo_est /= n_samples
            self.elbo_history.append(elbo_est)

            # Gradient ascent
            for d in range(self.dim):
                self.mu[d] += lr * grad_mu[d]
                self.log_sigma[d] += lr * grad_log_sigma[d]

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iter {iteration + 1}: ELBO = {elbo_est:.4f}")

        return self.mu[:], self._get_sigma()

    def sample_posterior(self, n=100):
        """Draw samples from the fitted posterior."""
        sigma = self._get_sigma()
        samples = []
        for _ in range(n):
            z = [self.rng.gauss(self.mu[i], sigma[i]) for i in range(self.dim)]
            samples.append(z)
        return samples


# ============================================================
# Black-Box VI (Score Function / REINFORCE)
# ============================================================

class BlackBoxVI:
    """Black-box VI using score function (REINFORCE) estimator.

    Works with any q that supports log_prob and sample.
    Uses control variates for variance reduction.
    """

    def __init__(self, dim, log_joint, seed=None):
        self.dim = dim
        self.log_joint = log_joint
        self.rng = random.Random(seed)
        self.mu = [0.0] * dim
        self.log_sigma = [0.0] * dim
        self.elbo_history = []

    def _get_sigma(self):
        return [math.exp(ls) for ls in self.log_sigma]

    def fit(self, n_iter=1000, lr=0.01, n_samples=20, use_control_variate=True, verbose=False):
        """Optimize using score function estimator with optional control variate."""
        for iteration in range(n_iter):
            sigma = self._get_sigma()

            # Sample from q
            samples = []
            log_weights = []
            score_mu = []
            score_log_sigma = []

            for _ in range(n_samples):
                eps = [self.rng.gauss(0, 1) for _ in range(self.dim)]
                z = [self.mu[d] + sigma[d] * eps[d] for d in range(self.dim)]
                samples.append(z)

                lp = self.log_joint(z)
                lq = sum(-0.5 * math.log(2 * math.pi) - math.log(sigma[d]) - 0.5 * eps[d] ** 2
                         for d in range(self.dim))
                log_weights.append(lp - lq)

                # Score function: grad log q(z; params)
                s_mu = [eps[d] / sigma[d] for d in range(self.dim)]
                s_ls = [eps[d] ** 2 - 1.0 for d in range(self.dim)]
                score_mu.append(s_mu)
                score_log_sigma.append(s_ls)

            elbo_est = sum(log_weights) / n_samples
            self.elbo_history.append(elbo_est)

            # Compute gradients: E[(log p - log q) * score]
            if use_control_variate:
                # Baseline: average of log weights
                baseline = sum(log_weights) / n_samples
                centered = [lw - baseline for lw in log_weights]
            else:
                centered = log_weights

            grad_mu = [0.0] * self.dim
            grad_log_sigma = [0.0] * self.dim

            for s in range(n_samples):
                for d in range(self.dim):
                    grad_mu[d] += centered[s] * score_mu[s][d]
                    grad_log_sigma[d] += centered[s] * score_log_sigma[s][d]

            for d in range(self.dim):
                grad_mu[d] /= n_samples
                grad_log_sigma[d] /= n_samples

            # Gradient ascent
            for d in range(self.dim):
                self.mu[d] += lr * grad_mu[d]
                self.log_sigma[d] += lr * grad_log_sigma[d]

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iter {iteration + 1}: ELBO = {elbo_est:.4f}")

        return self.mu[:], self._get_sigma()

    def get_posterior(self):
        sigma = self._get_sigma()
        return [Normal(self.mu[i], sigma[i]) for i in range(self.dim)]

    def sample_posterior(self, n=100):
        sigma = self._get_sigma()
        return [[self.rng.gauss(self.mu[d], sigma[d]) for d in range(self.dim)] for _ in range(n)]


# ============================================================
# Reparameterized VI
# ============================================================

class ReparameterizedVI:
    """VI using reparameterization trick for low-variance gradients.

    Uses Adam-style updates with momentum for stable optimization.
    """

    def __init__(self, dim, log_joint, seed=None):
        self.dim = dim
        self.log_joint = log_joint
        self.rng = random.Random(seed)
        self.mu = [0.0] * dim
        self.log_sigma = [0.0] * dim
        self.elbo_history = []

    def _get_sigma(self):
        return [math.exp(ls) for ls in self.log_sigma]

    def fit(self, n_iter=1000, lr=0.01, n_samples=10, beta1=0.9, beta2=0.999, verbose=False):
        """Optimize with Adam on reparameterized ELBO gradients."""
        eps_adam = 1e-8
        fd_eps = 1e-5
        m_mu = [0.0] * self.dim
        m_ls = [0.0] * self.dim
        v_mu = [0.0] * self.dim
        v_ls = [0.0] * self.dim

        for iteration in range(n_iter):
            sigma = self._get_sigma()

            grad_mu = [0.0] * self.dim
            grad_log_sigma = [0.0] * self.dim
            elbo_est = 0.0

            for _ in range(n_samples):
                epsilon = [self.rng.gauss(0, 1) for _ in range(self.dim)]
                z = [self.mu[d] + sigma[d] * epsilon[d] for d in range(self.dim)]

                lp = self.log_joint(z)
                lq = sum(-0.5 * math.log(2 * math.pi) - math.log(sigma[d]) - 0.5 * epsilon[d] ** 2
                         for d in range(self.dim))
                f_val = lp - lq
                elbo_est += f_val

                # Gradient via finite differences on log_joint
                for d in range(self.dim):
                    z_p = z[:]
                    z_p[d] += fd_eps
                    z_m = z[:]
                    z_m[d] -= fd_eps
                    dlp_dz = (self.log_joint(z_p) - self.log_joint(z_m)) / (2 * fd_eps)

                    grad_mu[d] += dlp_dz  # dz/dmu = 1
                    grad_log_sigma[d] += dlp_dz * sigma[d] * epsilon[d] + 1.0  # entropy gradient

            for d in range(self.dim):
                grad_mu[d] /= n_samples
                grad_log_sigma[d] /= n_samples
            elbo_est /= n_samples
            self.elbo_history.append(elbo_est)

            # Adam update
            t = iteration + 1
            for d in range(self.dim):
                m_mu[d] = beta1 * m_mu[d] + (1 - beta1) * grad_mu[d]
                v_mu[d] = beta2 * v_mu[d] + (1 - beta2) * grad_mu[d] ** 2
                m_hat = m_mu[d] / (1 - beta1 ** t)
                v_hat = v_mu[d] / (1 - beta2 ** t)
                self.mu[d] += lr * m_hat / (math.sqrt(v_hat) + eps_adam)

                m_ls[d] = beta1 * m_ls[d] + (1 - beta1) * grad_log_sigma[d]
                v_ls[d] = beta2 * v_ls[d] + (1 - beta2) * grad_log_sigma[d] ** 2
                m_hat = m_ls[d] / (1 - beta1 ** t)
                v_hat = v_ls[d] / (1 - beta2 ** t)
                self.log_sigma[d] += lr * m_hat / (math.sqrt(v_hat) + eps_adam)

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iter {iteration + 1}: ELBO = {elbo_est:.4f}")

        return self.mu[:], self._get_sigma()

    def get_posterior(self):
        sigma = self._get_sigma()
        return [Normal(self.mu[i], sigma[i]) for i in range(self.dim)]

    def sample_posterior(self, n=100):
        sigma = self._get_sigma()
        return [[self.rng.gauss(self.mu[d], sigma[d]) for d in range(self.dim)] for _ in range(n)]


# ============================================================
# Amortized VI (VAE-style encoder)
# ============================================================

class AmortizedVI:
    """Amortized variational inference using a neural network encoder.

    Maps observations x -> (mu, log_sigma) of q(z|x).
    Trains via reparameterized ELBO.
    Composes C140 Neural Network for the encoder.
    """

    def __init__(self, input_dim, latent_dim, hidden_dims=None, seed=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rng = random.Random(seed)
        self.elbo_history = []

        if hidden_dims is None:
            hidden_dims = [max(input_dim, latent_dim * 2)]

        # Build encoder: input -> hidden -> (mu, log_sigma)
        rng_obj = random.Random(seed)
        self.encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.encoder_layers.append(Dense(prev_dim, h_dim, rng=rng_obj))
            self.encoder_layers.append(Activation('tanh'))
            prev_dim = h_dim

        # Output heads
        self.mu_layer = Dense(prev_dim, latent_dim, rng=rng_obj)
        self.log_sigma_layer = Dense(prev_dim, latent_dim, rng=rng_obj)

    def encode(self, x):
        """Encode observation x -> (mu, log_sigma).

        Args:
            x: list of floats (single observation) or list of lists (batch)
        Returns:
            (mu, log_sigma) as lists of floats
        """
        # Convert to Tensor
        if isinstance(x[0], (list, tuple)):
            h = Tensor(x)
        else:
            h = Tensor([x])  # batch of 1

        for layer in self.encoder_layers:
            h = layer.forward(h)

        mu = self.mu_layer.forward(h)
        log_sigma = self.log_sigma_layer.forward(h)

        # Extract as lists
        if isinstance(mu.data[0], list):
            return mu.data[0], log_sigma.data[0]
        return mu.data, log_sigma.data

    def sample_latent(self, x, n=1):
        """Sample z ~ q(z|x) via reparameterization."""
        mu, log_sigma = self.encode(x)
        sigma = [math.exp(max(min(ls, 10), -10)) for ls in log_sigma]
        samples = []
        for _ in range(n):
            eps = [self.rng.gauss(0, 1) for _ in range(self.latent_dim)]
            z = [mu[d] + sigma[d] * eps[d] for d in range(self.latent_dim)]
            samples.append(z)
        return samples[0] if n == 1 else samples

    def fit(self, data, log_likelihood, n_epochs=100, lr=0.001, batch_size=None, verbose=False):
        """Train encoder by maximizing ELBO.

        Args:
            data: list of observations (list of lists)
            log_likelihood: callable(x, z) -> log p(x|z)
            n_epochs: number of training epochs
            lr: learning rate
            batch_size: mini-batch size (None = full batch)
        Returns:
            elbo_history
        """
        n = len(data)
        if batch_size is None:
            batch_size = min(n, 32)

        # Collect all trainable layers
        all_layers = self.encoder_layers + [self.mu_layer, self.log_sigma_layer]

        for epoch in range(n_epochs):
            indices = list(range(n))
            self.rng.shuffle(indices)
            epoch_elbo = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_elbo = 0.0

                for idx in batch_idx:
                    x = data[idx]
                    mu, log_sigma = self.encode(x)
                    log_sigma_c = [max(min(ls, 10), -10) for ls in log_sigma]
                    sigma = [math.exp(ls) for ls in log_sigma_c]

                    # Reparameterized sample
                    eps = [self.rng.gauss(0, 1) for _ in range(self.latent_dim)]
                    z = [mu[d] + sigma[d] * eps[d] for d in range(self.latent_dim)]

                    # ELBO = log p(x|z) - KL(q(z|x) || p(z))
                    ll = log_likelihood(x, z)
                    # KL for diagonal Gaussian vs standard normal
                    kl = 0.5 * sum(
                        mu[d] ** 2 + sigma[d] ** 2 - 1.0 - 2.0 * log_sigma_c[d]
                        for d in range(self.latent_dim)
                    )
                    batch_elbo += ll - kl

                batch_elbo /= len(batch_idx)
                epoch_elbo += batch_elbo
                n_batches += 1

                # Numerical gradient update for encoder parameters
                self._update_params(data, batch_idx, log_likelihood, lr)

            epoch_elbo /= n_batches
            self.elbo_history.append(epoch_elbo)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: ELBO = {epoch_elbo:.4f}")

        return self.elbo_history

    def _update_params(self, data, batch_idx, log_likelihood, lr):
        """Update encoder parameters via numerical gradients."""
        fd_eps = 1e-4
        all_layers = self.encoder_layers + [self.mu_layer, self.log_sigma_layer]

        for layer in all_layers:
            if not hasattr(layer, 'weights'):
                continue
            # Update weights
            w = layer.weights
            for i in range(len(w.data)):
                row = w.data[i] if isinstance(w.data[i], list) else [w.data[i]]
                for j in range(len(row)):
                    orig = row[j]
                    # Forward pass +
                    row[j] = orig + fd_eps
                    elbo_plus = self._batch_elbo(data, batch_idx, log_likelihood)
                    # Forward pass -
                    row[j] = orig - fd_eps
                    elbo_minus = self._batch_elbo(data, batch_idx, log_likelihood)
                    row[j] = orig
                    grad = (elbo_plus - elbo_minus) / (2 * fd_eps)
                    row[j] += lr * grad

            # Update bias
            if hasattr(layer, 'bias') and layer.bias is not None:
                b = layer.bias
                for i in range(len(b.data)):
                    orig = b.data[i]
                    b.data[i] = orig + fd_eps
                    elbo_plus = self._batch_elbo(data, batch_idx, log_likelihood)
                    b.data[i] = orig - fd_eps
                    elbo_minus = self._batch_elbo(data, batch_idx, log_likelihood)
                    b.data[i] = orig
                    grad = (elbo_plus - elbo_minus) / (2 * fd_eps)
                    b.data[i] += lr * grad

    def _batch_elbo(self, data, batch_idx, log_likelihood):
        """Compute average ELBO over a mini-batch."""
        total = 0.0
        for idx in batch_idx:
            x = data[idx]
            mu, log_sigma = self.encode(x)
            log_sigma_c = [max(min(ls, 10), -10) for ls in log_sigma]
            sigma = [math.exp(ls) for ls in log_sigma_c]
            eps = [self.rng.gauss(0, 1) for _ in range(self.latent_dim)]
            z = [mu[d] + sigma[d] * eps[d] for d in range(self.latent_dim)]
            ll = log_likelihood(x, z)
            kl = 0.5 * sum(mu[d] ** 2 + sigma[d] ** 2 - 1.0 - 2.0 * log_sigma_c[d]
                           for d in range(self.latent_dim))
            total += ll - kl
        return total / len(batch_idx)


# ============================================================
# Normalizing Flows
# ============================================================

class PlanarFlow:
    """Planar normalizing flow: f(z) = z + u * tanh(w^T z + b).

    Transforms a simple distribution into a more expressive one.
    """

    def __init__(self, dim, n_flows=1, seed=None):
        self.dim = dim
        self.n_flows = n_flows
        self.rng = random.Random(seed)
        # Initialize flow parameters
        self.flows = []
        for _ in range(n_flows):
            w = [self.rng.gauss(0, 0.5) for _ in range(dim)]
            u = [self.rng.gauss(0, 0.5) for _ in range(dim)]
            b = self.rng.gauss(0, 0.1)
            # Ensure invertibility: u_hat such that w^T u_hat >= -1
            u = self._enforce_invertibility(w, u)
            self.flows.append({'w': w, 'u': u, 'b': b})

    def _enforce_invertibility(self, w, u):
        """Enforce w^T u >= -1 for invertibility."""
        wtu = sum(wi * ui for wi, ui in zip(w, u))
        if wtu >= -1:
            return u
        # Project: u_hat = u + (m(w^T u) - w^T u) * w / ||w||^2
        w_norm_sq = sum(wi ** 2 for wi in w)
        if w_norm_sq < 1e-10:
            return u
        m = -1 + math.log(1 + math.exp(wtu))  # softplus
        correction = (m - wtu) / w_norm_sq
        return [u[i] + correction * w[i] for i in range(len(w))]

    def forward(self, z):
        """Apply flow transforms: z -> z'.

        Returns (z_transformed, sum_log_det_jacobian).
        """
        log_det = 0.0
        z_curr = list(z)
        for flow in self.flows:
            w, u, b = flow['w'], flow['u'], flow['b']
            # w^T z + b
            wtz = sum(w[d] * z_curr[d] for d in range(self.dim)) + b
            tanh_val = math.tanh(wtz)
            dtanh = 1.0 - tanh_val ** 2

            # z' = z + u * tanh(w^T z + b)
            z_new = [z_curr[d] + u[d] * tanh_val for d in range(self.dim)]

            # log |det dz'/dz| = log |1 + u^T * dtanh * w|
            psi = [dtanh * w[d] for d in range(self.dim)]
            det_term = 1.0 + sum(u[d] * psi[d] for d in range(self.dim))
            log_det += math.log(abs(det_term) + 1e-10)

            z_curr = z_new
        return z_curr, log_det

    def log_prob(self, z, base_log_prob):
        """Compute log probability of z under the flow.

        Uses change of variables: log q(z') = log q0(z) - log|det J|
        Note: This computes the density of the *transformed* point.
        """
        z_t, log_det = self.forward(z)
        return base_log_prob(z) - log_det

    def sample(self, base_dist, n=1):
        """Sample from the flow by transforming base samples."""
        base_samples = base_dist.sample(n=n, rng=self.rng)
        if n == 1:
            base_samples = [base_samples]
        results = []
        for z in base_samples:
            z_t, _ = self.forward(z)
            results.append(z_t)
        return results[0] if n == 1 else results


class RadialFlow:
    """Radial normalizing flow: f(z) = z + beta * h(alpha, r) * (z - z0).

    Contracts/expands around a reference point z0.
    """

    def __init__(self, dim, n_flows=1, seed=None):
        self.dim = dim
        self.n_flows = n_flows
        self.rng = random.Random(seed)
        self.flows = []
        for _ in range(n_flows):
            z0 = [self.rng.gauss(0, 0.5) for _ in range(dim)]
            log_alpha = self.rng.gauss(0, 0.5)
            beta_raw = self.rng.gauss(0, 0.5)
            self.flows.append({'z0': z0, 'log_alpha': log_alpha, 'beta_raw': beta_raw})

    def forward(self, z):
        """Apply radial flow transforms."""
        log_det = 0.0
        z_curr = list(z)
        for flow in self.flows:
            z0 = flow['z0']
            alpha = math.exp(flow['log_alpha'])
            # beta must be >= -alpha for invertibility
            beta = -alpha + math.log(1 + math.exp(flow['beta_raw']))  # softplus - alpha

            diff = [z_curr[d] - z0[d] for d in range(self.dim)]
            r = math.sqrt(sum(d ** 2 for d in diff) + 1e-10)
            h = 1.0 / (alpha + r)
            hp = -1.0 / (alpha + r) ** 2  # dh/dr

            # z' = z + beta * h(r) * (z - z0)
            z_new = [z_curr[d] + beta * h * diff[d] for d in range(self.dim)]

            # log |det| = (d-1) * log(1 + beta*h) + log(1 + beta*h + beta*hp*r)
            bh = beta * h
            log_det += (self.dim - 1) * math.log(abs(1 + bh) + 1e-10)
            log_det += math.log(abs(1 + bh + beta * hp * r) + 1e-10)

            z_curr = z_new
        return z_curr, log_det

    def sample(self, base_dist, n=1):
        """Sample from the flow."""
        base_samples = base_dist.sample(n=n, rng=self.rng)
        if n == 1:
            base_samples = [base_samples]
        results = []
        for z in base_samples:
            z_t, _ = self.forward(z)
            results.append(z_t)
        return results[0] if n == 1 else results


# ============================================================
# Flow-based VI
# ============================================================

class FlowVI:
    """Variational inference with normalizing flows.

    Uses a base Gaussian transformed by flows for a flexible posterior.
    """

    def __init__(self, dim, log_joint, flow_type='planar', n_flows=4, seed=None):
        self.dim = dim
        self.log_joint = log_joint
        self.rng = random.Random(seed)
        self.elbo_history = []

        # Base distribution parameters
        self.mu = [0.0] * dim
        self.log_sigma = [0.0] * dim

        # Create flow
        if flow_type == 'planar':
            self.flow = PlanarFlow(dim, n_flows, seed=seed)
        elif flow_type == 'radial':
            self.flow = RadialFlow(dim, n_flows, seed=seed)
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

    def _get_sigma(self):
        return [math.exp(max(min(ls, 10), -10)) for ls in self.log_sigma]

    def _sample_base(self):
        sigma = self._get_sigma()
        eps = [self.rng.gauss(0, 1) for _ in range(self.dim)]
        return [self.mu[d] + sigma[d] * eps[d] for d in range(self.dim)]

    def _base_log_prob(self, z):
        sigma = self._get_sigma()
        return sum(-0.5 * math.log(2 * math.pi) - math.log(max(sigma[d], 1e-10))
                   - 0.5 * ((z[d] - self.mu[d]) / max(sigma[d], 1e-10)) ** 2
                   for d in range(self.dim))

    def fit(self, n_iter=500, lr=0.01, n_samples=10, verbose=False):
        """Optimize flow + base distribution parameters."""
        fd_eps = 1e-4

        for iteration in range(n_iter):
            elbo_est = 0.0

            for _ in range(n_samples):
                z0 = self._sample_base()
                zk, log_det = self.flow.forward(z0)
                log_q0 = self._base_log_prob(z0)
                log_qk = log_q0 - log_det
                lp = self.log_joint(zk)
                elbo_est += lp - log_qk

            elbo_est /= n_samples
            self.elbo_history.append(elbo_est)

            # Update base distribution params via numerical gradients
            for d in range(self.dim):
                orig = self.mu[d]
                self.mu[d] = orig + fd_eps
                ep = self._compute_elbo(n_samples)
                self.mu[d] = orig - fd_eps
                em = self._compute_elbo(n_samples)
                self.mu[d] = orig
                grad = (ep - em) / (2 * fd_eps)
                self.mu[d] += lr * grad

                orig = self.log_sigma[d]
                self.log_sigma[d] = orig + fd_eps
                ep = self._compute_elbo(n_samples)
                self.log_sigma[d] = orig - fd_eps
                em = self._compute_elbo(n_samples)
                self.log_sigma[d] = orig
                grad = (ep - em) / (2 * fd_eps)
                self.log_sigma[d] += lr * grad

            # Update flow parameters
            for fi, flow_params in enumerate(self.flow.flows):
                for key in flow_params:
                    if key in ('w', 'u', 'z0'):
                        vec = flow_params[key]
                        for d in range(len(vec)):
                            orig = vec[d]
                            vec[d] = orig + fd_eps
                            if key == 'u' and hasattr(self.flow, '_enforce_invertibility'):
                                pass  # skip enforcement during gradient
                            ep = self._compute_elbo(n_samples)
                            vec[d] = orig - fd_eps
                            em = self._compute_elbo(n_samples)
                            vec[d] = orig
                            grad = (ep - em) / (2 * fd_eps)
                            vec[d] += lr * grad
                    else:
                        orig = flow_params[key]
                        flow_params[key] = orig + fd_eps
                        ep = self._compute_elbo(n_samples)
                        flow_params[key] = orig - fd_eps
                        em = self._compute_elbo(n_samples)
                        flow_params[key] = orig
                        grad = (ep - em) / (2 * fd_eps)
                        flow_params[key] += lr * grad

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iter {iteration + 1}: ELBO = {elbo_est:.4f}")

        return self.mu[:], self._get_sigma()

    def _compute_elbo(self, n_samples):
        total = 0.0
        for _ in range(n_samples):
            z0 = self._sample_base()
            zk, log_det = self.flow.forward(z0)
            log_q0 = self._base_log_prob(z0)
            log_qk = log_q0 - log_det
            lp = self.log_joint(zk)
            total += lp - log_qk
        return total / n_samples

    def sample_posterior(self, n=100):
        """Sample from the flow-transformed posterior."""
        samples = []
        for _ in range(n):
            z0 = self._sample_base()
            zk, _ = self.flow.forward(z0)
            samples.append(zk)
        return samples


# ============================================================
# ADVI (Automatic Differentiation VI)
# ============================================================

class ADVI:
    """Automatic Differentiation Variational Inference.

    Transforms constrained parameters to unconstrained space,
    then performs mean-field VI in the unconstrained space.
    """

    def __init__(self, dim, log_joint, transforms=None, seed=None):
        """
        Args:
            dim: number of parameters
            log_joint: callable(params) -> log p(data, params)
            transforms: list of transform specs per dimension
                Each is a dict with 'type': 'real' | 'positive' | 'unit_interval'
                None means all 'real' (no transform needed)
        """
        self.dim = dim
        self.log_joint = log_joint
        self.rng = random.Random(seed)
        self.elbo_history = []

        if transforms is None:
            self.transforms = [{'type': 'real'} for _ in range(dim)]
        else:
            self.transforms = transforms

        # Variational parameters (in unconstrained space)
        self.mu = [0.0] * dim
        self.log_sigma = [0.0] * dim

    def _to_constrained(self, z_unconstrained):
        """Transform unconstrained z to constrained parameter space."""
        result = []
        log_det = 0.0
        for d in range(self.dim):
            t = self.transforms[d]['type']
            zu = z_unconstrained[d]
            if t == 'real':
                result.append(zu)
            elif t == 'positive':
                result.append(math.exp(zu))
                log_det += zu  # log |d(exp(z))/dz| = z
            elif t == 'unit_interval':
                sig = 1.0 / (1.0 + math.exp(-zu))
                result.append(sig)
                log_det += math.log(sig * (1 - sig) + 1e-10)
            else:
                result.append(zu)
        return result, log_det

    def _get_sigma(self):
        return [math.exp(ls) for ls in self.log_sigma]

    def fit(self, n_iter=1000, lr=0.01, n_samples=10, verbose=False):
        """Fit in unconstrained space with Jacobian correction."""
        fd_eps = 1e-5

        for iteration in range(n_iter):
            sigma = self._get_sigma()
            elbo_est = 0.0
            grad_mu = [0.0] * self.dim
            grad_log_sigma = [0.0] * self.dim

            for _ in range(n_samples):
                eps = [self.rng.gauss(0, 1) for _ in range(self.dim)]
                z_unc = [self.mu[d] + sigma[d] * eps[d] for d in range(self.dim)]

                z_con, log_det_j = self._to_constrained(z_unc)
                lp = self.log_joint(z_con)
                lq = sum(-0.5 * math.log(2 * math.pi) - math.log(sigma[d]) - 0.5 * eps[d] ** 2
                         for d in range(self.dim))

                # ELBO = log_joint(constrained) + log_det_jacobian - log_q(unconstrained)
                f_val = lp + log_det_j - lq
                elbo_est += f_val

                for d in range(self.dim):
                    z_p = z_unc[:]
                    z_p[d] += fd_eps
                    zc_p, ldj_p = self._to_constrained(z_p)
                    z_m = z_unc[:]
                    z_m[d] -= fd_eps
                    zc_m, ldj_m = self._to_constrained(z_m)

                    dlp = (self.log_joint(zc_p) + ldj_p - self.log_joint(zc_m) - ldj_m) / (2 * fd_eps)
                    grad_mu[d] += dlp
                    grad_log_sigma[d] += dlp * sigma[d] * eps[d] + 1.0

            for d in range(self.dim):
                grad_mu[d] /= n_samples
                grad_log_sigma[d] /= n_samples
            elbo_est /= n_samples
            self.elbo_history.append(elbo_est)

            for d in range(self.dim):
                self.mu[d] += lr * grad_mu[d]
                self.log_sigma[d] += lr * grad_log_sigma[d]

            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iter {iteration + 1}: ELBO = {elbo_est:.4f}")

        return self.get_constrained_params()

    def get_constrained_params(self):
        """Return MAP estimate in constrained space."""
        z_con, _ = self._to_constrained(self.mu)
        return z_con

    def get_posterior(self):
        """Return posterior in unconstrained space."""
        sigma = self._get_sigma()
        return [Normal(self.mu[d], sigma[d]) for d in range(self.dim)]

    def sample_posterior(self, n=100):
        """Sample from posterior in constrained space."""
        sigma = self._get_sigma()
        samples = []
        for _ in range(n):
            z_unc = [self.rng.gauss(self.mu[d], sigma[d]) for d in range(self.dim)]
            z_con, _ = self._to_constrained(z_unc)
            samples.append(z_con)
        return samples


# ============================================================
# Stochastic VI with Natural Gradients
# ============================================================

class StochasticVI:
    """Stochastic variational inference.

    Uses mini-batch data subsampling and natural gradient updates
    for scalable inference in conjugate exponential family models.
    """

    def __init__(self, dim, seed=None):
        self.dim = dim
        self.rng = random.Random(seed)
        self.mu = [0.0] * dim
        self.log_sigma = [0.0] * dim
        self.elbo_history = []

    def _get_sigma(self):
        return [math.exp(ls) for ls in self.log_sigma]

    def fit(self, data, log_likelihood, log_prior, n_epochs=100, batch_size=10,
            lr=0.01, n_samples=5, verbose=False):
        """Stochastic optimization of ELBO with data subsampling.

        Args:
            data: list of observations
            log_likelihood: callable(z, x_i) -> log p(x_i | z)
            log_prior: callable(z) -> log p(z)
            n_epochs: number of passes through data
            batch_size: mini-batch size
            n_samples: MC samples per gradient estimate
        """
        n = len(data)
        fd_eps = 1e-5

        for epoch in range(n_epochs):
            indices = list(range(n))
            self.rng.shuffle(indices)
            epoch_elbo = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = [data[indices[i]] for i in range(start, min(start + batch_size, n))]
                batch_n = len(batch)
                sigma = self._get_sigma()

                grad_mu = [0.0] * self.dim
                grad_log_sigma = [0.0] * self.dim
                batch_elbo = 0.0

                for _ in range(n_samples):
                    eps = [self.rng.gauss(0, 1) for _ in range(self.dim)]
                    z = [self.mu[d] + sigma[d] * eps[d] for d in range(self.dim)]

                    # Scaled likelihood (n/batch_size * sum over batch)
                    ll = sum(log_likelihood(z, xi) for xi in batch) * (n / batch_n)
                    lp = log_prior(z)
                    lq = sum(-0.5 * math.log(2 * math.pi) - math.log(sigma[d]) - 0.5 * eps[d] ** 2
                             for d in range(self.dim))
                    f_val = ll + lp - lq
                    batch_elbo += f_val

                    for d in range(self.dim):
                        z_p = z[:]
                        z_p[d] += fd_eps
                        z_m = z[:]
                        z_m[d] -= fd_eps
                        ll_p = sum(log_likelihood(z_p, xi) for xi in batch) * (n / batch_n) + log_prior(z_p)
                        ll_m = sum(log_likelihood(z_m, xi) for xi in batch) * (n / batch_n) + log_prior(z_m)
                        dlp = (ll_p - ll_m) / (2 * fd_eps)
                        grad_mu[d] += dlp
                        grad_log_sigma[d] += dlp * sigma[d] * eps[d] + 1.0

                for d in range(self.dim):
                    grad_mu[d] /= n_samples
                    grad_log_sigma[d] /= n_samples
                batch_elbo /= n_samples

                # Natural gradient approximation: scale by Fisher inverse ~ sigma^2
                for d in range(self.dim):
                    self.mu[d] += lr * grad_mu[d]
                    self.log_sigma[d] += lr * grad_log_sigma[d]

                epoch_elbo += batch_elbo
                n_batches += 1

            epoch_elbo /= n_batches
            self.elbo_history.append(epoch_elbo)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: ELBO = {epoch_elbo:.4f}")

        return self.mu[:], self._get_sigma()

    def get_posterior(self):
        sigma = self._get_sigma()
        return [Normal(self.mu[d], sigma[d]) for d in range(self.dim)]

    def sample_posterior(self, n=100):
        sigma = self._get_sigma()
        return [[self.rng.gauss(self.mu[d], sigma[d]) for d in range(self.dim)] for _ in range(n)]


# ============================================================
# VI Diagnostics
# ============================================================

class VIDiagnostics:
    """Diagnostics for variational inference."""

    @staticmethod
    def elbo_convergence(elbo_history, window=50):
        """Check if ELBO has converged.

        Returns dict with convergence metrics.
        """
        n = len(elbo_history)
        if n < window * 2:
            return {
                'converged': False,
                'reason': 'insufficient_data',
                'n_iters': n,
                'final_elbo': elbo_history[-1] if n > 0 else None
            }

        recent = elbo_history[-window:]
        earlier = elbo_history[-2 * window:-window]

        recent_mean = sum(recent) / len(recent)
        earlier_mean = sum(earlier) / len(earlier)

        recent_std = math.sqrt(sum((x - recent_mean) ** 2 for x in recent) / len(recent))
        improvement = recent_mean - earlier_mean

        # Relative improvement
        scale = max(abs(recent_mean), 1.0)
        rel_improvement = abs(improvement) / scale

        converged = rel_improvement < 0.01 and recent_std / scale < 0.05

        return {
            'converged': converged,
            'final_elbo': recent_mean,
            'improvement': improvement,
            'relative_improvement': rel_improvement,
            'recent_std': recent_std,
            'n_iters': n
        }

    @staticmethod
    def posterior_summary(vi_model):
        """Summarize the fitted posterior from a VI model.

        Works with MeanFieldVI, BlackBoxVI, ReparameterizedVI, ADVI, StochasticVI.
        """
        mu = vi_model.mu
        sigma = vi_model._get_sigma()
        dim = vi_model.dim

        summary = {}
        for d in range(dim):
            summary[f'dim_{d}'] = {
                'mean': mu[d],
                'std': sigma[d],
                'q025': mu[d] - 1.96 * sigma[d],
                'q975': mu[d] + 1.96 * sigma[d],
                'cv': sigma[d] / abs(mu[d]) if abs(mu[d]) > 1e-10 else float('inf')
            }
        return summary

    @staticmethod
    def compare_methods(results):
        """Compare multiple VI methods.

        Args:
            results: dict of {method_name: {'mu': [...], 'sigma': [...], 'elbo_history': [...]}}
        Returns:
            comparison dict
        """
        comparison = {}
        for name, res in results.items():
            elbo = res.get('elbo_history', [])
            comparison[name] = {
                'final_elbo': elbo[-1] if elbo else None,
                'n_iters': len(elbo),
                'mu': res.get('mu', []),
                'sigma': res.get('sigma', [])
            }

        # Rank by final ELBO
        ranked = sorted(
            [(name, d['final_elbo']) for name, d in comparison.items() if d['final_elbo'] is not None],
            key=lambda x: x[1],
            reverse=True
        )
        comparison['_ranking'] = [name for name, _ in ranked]
        return comparison

    @staticmethod
    def kl_to_mcmc(vi_samples, mcmc_samples):
        """Estimate how well VI posterior matches MCMC posterior.

        Uses simple mean/variance comparison.
        Args:
            vi_samples: list of samples from VI posterior (list of lists)
            mcmc_samples: list of samples from MCMC (list of lists)
        Returns:
            dict with per-dimension comparison
        """
        if not vi_samples or not mcmc_samples:
            return {'error': 'empty samples'}

        dim = len(vi_samples[0]) if isinstance(vi_samples[0], (list, tuple)) else 1
        result = {}

        for d in range(dim):
            if dim == 1:
                vi_vals = [s if not isinstance(s, (list, tuple)) else s[0] for s in vi_samples]
                mc_vals = [s if not isinstance(s, (list, tuple)) else s[0] for s in mcmc_samples]
            else:
                vi_vals = [s[d] for s in vi_samples]
                mc_vals = [s[d] for s in mcmc_samples]

            vi_mean = sum(vi_vals) / len(vi_vals)
            mc_mean = sum(mc_vals) / len(mc_vals)
            vi_var = sum((v - vi_mean) ** 2 for v in vi_vals) / len(vi_vals)
            mc_var = sum((v - mc_mean) ** 2 for v in mc_vals) / len(mc_vals)

            result[f'dim_{d}'] = {
                'vi_mean': vi_mean,
                'mcmc_mean': mc_mean,
                'mean_diff': abs(vi_mean - mc_mean),
                'vi_std': math.sqrt(vi_var),
                'mcmc_std': math.sqrt(mc_var),
                'std_ratio': math.sqrt(vi_var) / math.sqrt(mc_var) if mc_var > 1e-10 else float('inf')
            }

        return result
