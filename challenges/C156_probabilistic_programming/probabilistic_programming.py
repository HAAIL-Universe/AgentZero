"""
C156: Probabilistic Programming Language
Composing C153 (Monte Carlo) + C154 (Variational Inference)

A declarative PPL: define models with sample/observe, run inference
with MCMC or VI backends, analyze results with traces and diagnostics.
"""

import math
import random
import numpy as np
from collections import OrderedDict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C153_monte_carlo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C154_variational_inference'))

from monte_carlo import (
    RandomSampler, MetropolisHastings, HamiltonianMC, NUTS,
    SliceSampler, MCMCDiagnostics
)
from variational_inference import (
    MeanFieldVI, BlackBoxVI, ReparameterizedVI, ADVI,
    Normal as VINormal, MultivariateNormal as VIMvNormal,
    KLDivergence, VIDiagnostics
)


# ============================================================
# Distributions
# ============================================================

class Distribution:
    """Base distribution class for the PPL."""

    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, rng=None, size=None):
        raise NotImplementedError

    @property
    def support(self):
        """Return (low, high) bounds for the support."""
        return (-np.inf, np.inf)

    @property
    def transform_type(self):
        """Return ADVI transform type: 'real', 'positive', 'unit'."""
        return 'real'


class Normal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        return -0.5 * math.log(2 * math.pi) - math.log(self.sigma) - 0.5 * ((x - self.mu) / self.sigma) ** 2

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.normal(self.mu, self.sigma, size=size)


class HalfNormal(Distribution):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def log_prob(self, x):
        if x < 0:
            return -np.inf
        return math.log(2) - 0.5 * math.log(2 * math.pi) - math.log(self.sigma) - 0.5 * (x / self.sigma) ** 2

    def sample(self, rng=None, size=None):
        r = rng or np.random
        s = np.abs(r.normal(0, self.sigma, size=size))
        return s

    @property
    def support(self):
        return (0, np.inf)

    @property
    def transform_type(self):
        return 'positive'


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def log_prob(self, x):
        if x < self.low or x > self.high:
            return -np.inf
        return -math.log(self.high - self.low)

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.uniform(self.low, self.high, size=size)

    @property
    def support(self):
        return (self.low, self.high)

    @property
    def transform_type(self):
        return 'unit'


class Beta(Distribution):
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def log_prob(self, x):
        if x <= 0 or x >= 1:
            return -np.inf
        from math import lgamma
        return (lgamma(self.alpha + self.beta) - lgamma(self.alpha) - lgamma(self.beta)
                + (self.alpha - 1) * math.log(x) + (self.beta - 1) * math.log(1 - x))

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.beta(self.alpha, self.beta, size=size)

    @property
    def support(self):
        return (0, 1)

    @property
    def transform_type(self):
        return 'unit'


class Gamma(Distribution):
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # shape
        self.beta = beta    # rate

    def log_prob(self, x):
        if x <= 0:
            return -np.inf
        from math import lgamma
        return (self.alpha * math.log(self.beta) - lgamma(self.alpha)
                + (self.alpha - 1) * math.log(x) - self.beta * x)

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.gamma(self.alpha, 1.0 / self.beta, size=size)

    @property
    def support(self):
        return (0, np.inf)

    @property
    def transform_type(self):
        return 'positive'


class Exponential(Distribution):
    def __init__(self, rate=1.0):
        self.rate = rate

    def log_prob(self, x):
        if x < 0:
            return -np.inf
        return math.log(self.rate) - self.rate * x

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.exponential(1.0 / self.rate, size=size)

    @property
    def support(self):
        return (0, np.inf)

    @property
    def transform_type(self):
        return 'positive'


class Bernoulli(Distribution):
    def __init__(self, p=0.5):
        self.p = p

    def log_prob(self, x):
        if x == 1:
            return math.log(self.p) if self.p > 0 else -np.inf
        elif x == 0:
            return math.log(1 - self.p) if self.p < 1 else -np.inf
        return -np.inf

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return (r.uniform(size=size) < self.p).astype(float)

    @property
    def support(self):
        return (0, 1)


class Poisson(Distribution):
    def __init__(self, rate=1.0):
        self.rate = rate

    def log_prob(self, x):
        if x < 0 or x != int(x):
            return -np.inf
        k = int(x)
        from math import lgamma
        return k * math.log(self.rate) - self.rate - lgamma(k + 1)

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.poisson(self.rate, size=size).astype(float)


class StudentT(Distribution):
    def __init__(self, nu=1.0, mu=0.0, sigma=1.0):
        self.nu = nu
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        from math import lgamma
        z = (x - self.mu) / self.sigma
        return (lgamma((self.nu + 1) / 2) - lgamma(self.nu / 2)
                - 0.5 * math.log(self.nu * math.pi) - math.log(self.sigma)
                - (self.nu + 1) / 2 * math.log(1 + z * z / self.nu))

    def sample(self, rng=None, size=None):
        r = rng or np.random
        t = r.standard_t(self.nu, size=size)
        return self.mu + self.sigma * t


class LogNormal(Distribution):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x):
        if x <= 0:
            return -np.inf
        return (-0.5 * math.log(2 * math.pi) - math.log(self.sigma) - math.log(x)
                - 0.5 * ((math.log(x) - self.mu) / self.sigma) ** 2)

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.lognormal(self.mu, self.sigma, size=size)

    @property
    def support(self):
        return (0, np.inf)

    @property
    def transform_type(self):
        return 'positive'


class HalfCauchy(Distribution):
    def __init__(self, beta=1.0):
        self.beta = beta

    def log_prob(self, x):
        if x < 0:
            return -np.inf
        return math.log(2) - math.log(math.pi) - math.log(self.beta) - math.log(1 + (x / self.beta) ** 2)

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return np.abs(r.standard_cauchy(size=size)) * self.beta

    @property
    def support(self):
        return (0, np.inf)

    @property
    def transform_type(self):
        return 'positive'


class Categorical(Distribution):
    def __init__(self, probs):
        probs = np.array(probs, dtype=float)
        self.probs = probs / probs.sum()
        self.k = len(probs)

    def log_prob(self, x):
        idx = int(x)
        if idx < 0 or idx >= self.k:
            return -np.inf
        p = self.probs[idx]
        return math.log(p) if p > 0 else -np.inf

    def sample(self, rng=None, size=None):
        r = rng or np.random
        if size is None:
            return float(r.choice(self.k, p=self.probs))
        return r.choice(self.k, p=self.probs, size=size).astype(float)


class Dirichlet(Distribution):
    def __init__(self, alpha):
        self.alpha = np.array(alpha, dtype=float)

    def log_prob(self, x):
        x = np.array(x)
        if np.any(x <= 0) or abs(x.sum() - 1.0) > 1e-6:
            return -np.inf
        from math import lgamma
        return (lgamma(self.alpha.sum()) - sum(lgamma(a) for a in self.alpha)
                + sum((a - 1) * math.log(xi) for a, xi in zip(self.alpha, x)))

    def sample(self, rng=None, size=None):
        r = rng or np.random
        return r.dirichlet(self.alpha, size=size)


class MultivariateNormal(Distribution):
    def __init__(self, mu, cov):
        self.mu = np.array(mu, dtype=float)
        self.cov = np.array(cov, dtype=float)
        self.dim = len(self.mu)

    def log_prob(self, x):
        x = np.array(x)
        diff = x - self.mu
        try:
            L = np.linalg.cholesky(self.cov)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            solve = np.linalg.solve(self.cov, diff)
            return -0.5 * (self.dim * math.log(2 * math.pi) + log_det + diff @ solve)
        except np.linalg.LinAlgError:
            return -np.inf

    def sample(self, rng=None, size=None):
        r = rng or np.random
        if size is None:
            return r.multivariate_normal(self.mu, self.cov)
        return r.multivariate_normal(self.mu, self.cov, size=size)


# ============================================================
# Probabilistic Model
# ============================================================

class ProbModel:
    """Declarative probabilistic model container."""

    def __init__(self, name="model"):
        self.name = name
        self._latent_sites = OrderedDict()   # name -> (dist, shape)
        self._observed_sites = OrderedDict()  # name -> (dist_fn, observed_value)
        self._deterministic_sites = OrderedDict()  # name -> fn(latents_dict)
        self._param_order = []  # ordered list of latent param names
        self._current_values = {}  # current latent values during log_prob eval

    def sample(self, name, dist, shape=None):
        """Register a latent random variable."""
        if name in self._latent_sites or name in self._observed_sites:
            raise ValueError(f"Site '{name}' already registered")
        self._latent_sites[name] = (dist, shape)
        self._param_order.append(name)
        return self

    def observe(self, name, dist_or_fn, value):
        """Register an observed random variable.

        dist_or_fn: either a Distribution, or a callable(latents_dict) -> Distribution
        value: observed data (scalar or array)
        """
        if name in self._latent_sites or name in self._observed_sites:
            raise ValueError(f"Site '{name}' already registered")
        self._observed_sites[name] = (dist_or_fn, np.atleast_1d(np.array(value, dtype=float)))
        return self

    def deterministic(self, name, fn):
        """Register a deterministic transformation of latent variables."""
        self._deterministic_sites[name] = fn
        return self

    @property
    def num_latent(self):
        """Total number of scalar latent parameters."""
        n = 0
        for name in self._param_order:
            dist, shape = self._latent_sites[name]
            if shape is None:
                n += 1
            else:
                n += int(np.prod(shape))
        return n

    def _unpack_params(self, theta):
        """Unpack flat parameter vector into named dict."""
        theta = np.array(theta, dtype=float)
        values = {}
        idx = 0
        for name in self._param_order:
            dist, shape = self._latent_sites[name]
            if shape is None:
                values[name] = float(theta[idx])
                idx += 1
            else:
                size = int(np.prod(shape))
                values[name] = theta[idx:idx + size].reshape(shape)
                idx += size
        return values

    def _pack_params(self, values):
        """Pack named dict into flat parameter vector."""
        parts = []
        for name in self._param_order:
            v = values[name]
            if isinstance(v, np.ndarray):
                parts.append(v.ravel())
            else:
                parts.append(np.array([float(v)]))
        return np.concatenate(parts)

    def log_prior(self, theta):
        """Compute log prior: sum of log_prob for all latent sites."""
        values = self._unpack_params(theta)
        lp = 0.0
        for name in self._param_order:
            dist, shape = self._latent_sites[name]
            v = values[name]
            if shape is None:
                lp += dist.log_prob(v)
            else:
                flat = np.atleast_1d(v).ravel()
                for xi in flat:
                    lp += dist.log_prob(xi)
            if lp == -np.inf:
                return -np.inf
        return lp

    def log_likelihood(self, theta):
        """Compute log likelihood: sum of log_prob for all observed sites."""
        values = self._unpack_params(theta)
        self._current_values = values

        # Compute deterministic sites
        for dname, fn in self._deterministic_sites.items():
            values[dname] = fn(values)

        ll = 0.0
        for name, (dist_or_fn, obs) in self._observed_sites.items():
            if callable(dist_or_fn) and not isinstance(dist_or_fn, Distribution):
                dist = dist_or_fn(values)
            else:
                dist = dist_or_fn
            for y in obs.ravel():
                ll += dist.log_prob(y)
            if ll == -np.inf:
                return -np.inf
        return ll

    def log_joint(self, theta):
        """Compute log joint = log prior + log likelihood."""
        lp = self.log_prior(theta)
        if lp == -np.inf:
            return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll

    def prior_sample(self, rng=None):
        """Draw a sample from the prior."""
        values = {}
        for name in self._param_order:
            dist, shape = self._latent_sites[name]
            values[name] = dist.sample(rng=rng, size=shape)
            if shape is None:
                values[name] = float(values[name])
        return values


# ============================================================
# Trace: Posterior samples container
# ============================================================

class Trace:
    """Container for posterior samples with diagnostics."""

    def __init__(self, samples_dict, model=None):
        """
        samples_dict: {param_name: np.array of shape (n_samples,) or (n_samples, *shape)}
        """
        self.samples = samples_dict
        self.model = model
        self._n_samples = len(next(iter(samples_dict.values())))

    def __getitem__(self, name):
        return self.samples[name]

    def __len__(self):
        return self._n_samples

    def __contains__(self, name):
        return name in self.samples

    @property
    def param_names(self):
        return list(self.samples.keys())

    def mean(self, name=None):
        if name:
            return np.mean(self.samples[name], axis=0)
        return {k: np.mean(v, axis=0) for k, v in self.samples.items()}

    def std(self, name=None):
        if name:
            return np.std(self.samples[name], axis=0)
        return {k: np.std(v, axis=0) for k, v in self.samples.items()}

    def median(self, name=None):
        if name:
            return np.median(self.samples[name], axis=0)
        return {k: np.median(v, axis=0) for k, v in self.samples.items()}

    def quantile(self, q, name=None):
        if name:
            return np.quantile(self.samples[name], q, axis=0)
        return {k: np.quantile(v, q, axis=0) for k, v in self.samples.items()}

    def hdi(self, prob=0.94, name=None):
        """Highest density interval."""
        def _hdi(arr, prob):
            arr = np.sort(arr)
            n = len(arr)
            interval_size = int(np.ceil(prob * n))
            if interval_size >= n:
                return (arr[0], arr[-1])
            widths = arr[interval_size:] - arr[:n - interval_size]
            best = np.argmin(widths)
            return (arr[best], arr[best + interval_size])

        if name:
            return _hdi(self.samples[name], prob)
        return {k: _hdi(v, prob) for k, v in self.samples.items()}

    def summary(self):
        """Return summary statistics for all parameters."""
        result = {}
        for name, samples in self.samples.items():
            s = samples.ravel() if samples.ndim > 1 else samples
            result[name] = {
                'mean': float(np.mean(s)),
                'std': float(np.std(s)),
                'median': float(np.median(s)),
                'q2.5': float(np.quantile(s, 0.025)),
                'q97.5': float(np.quantile(s, 0.975)),
                'hdi_low': float(self.hdi(0.94, name)[0]) if s.ndim <= 1 else None,
                'hdi_high': float(self.hdi(0.94, name)[1]) if s.ndim <= 1 else None,
            }
        return result

    def effective_sample_size(self, name):
        """ESS using MCMCDiagnostics from C153."""
        return MCMCDiagnostics.effective_sample_size(self.samples[name])

    def r_hat(self, name, n_chains=4):
        """Split R-hat diagnostic."""
        s = self.samples[name]
        n = len(s)
        chain_len = n // n_chains
        if chain_len < 4:
            return float('nan')
        chains = np.array([s[i * chain_len:(i + 1) * chain_len] for i in range(n_chains)])
        return MCMCDiagnostics.r_hat(chains)


# ============================================================
# MCMC Inference
# ============================================================

class MCMCInference:
    """MCMC inference backend using C153 samplers."""

    SAMPLERS = {'metropolis', 'hmc', 'nuts', 'slice'}

    def __init__(self, model, sampler='nuts', seed=42):
        self.model = model
        self.sampler_name = sampler
        self.seed = seed

    def run(self, n_samples=1000, n_warmup=500, n_chains=1, step_size=0.05,
            n_leapfrog=20, proposal_scale=0.1, **kwargs):
        """Run MCMC inference, return Trace."""
        all_chain_samples = []

        for chain in range(n_chains):
            chain_seed = self.seed + chain * 1000
            rng = np.random.RandomState(chain_seed)

            dim = self.model.num_latent

            def log_prob(theta):
                return self.model.log_joint(theta)

            # Initialize from prior (with bounds checking)
            init = self._get_init(rng)

            if self.sampler_name == 'metropolis':
                sampler = MetropolisHastings(
                    log_prob=log_prob, dim=dim, seed=chain_seed
                )
                samples, accept_rate = sampler.sample(
                    n_samples=n_samples, initial=init, burn_in=n_warmup
                )

            elif self.sampler_name == 'hmc':
                def grad_log_prob(theta):
                    return self._numerical_gradient(theta)

                sampler = HamiltonianMC(
                    log_prob=log_prob, grad_log_prob=grad_log_prob,
                    dim=dim, seed=chain_seed
                )
                samples, accept_rate = sampler.sample(
                    n_samples=n_samples, initial=init, burn_in=n_warmup
                )

            elif self.sampler_name == 'nuts':
                def grad_log_prob(theta):
                    return self._numerical_gradient(theta)

                sampler = NUTS(
                    log_prob=log_prob, grad_log_prob=grad_log_prob,
                    dim=dim, seed=chain_seed
                )
                samples = sampler.sample(
                    n_samples=n_samples, initial=init, burn_in=n_warmup
                )

            elif self.sampler_name == 'slice':
                sampler = SliceSampler(
                    log_prob=log_prob, dim=dim, seed=chain_seed
                )
                samples = sampler.sample(
                    n_samples=n_samples, initial=init, burn_in=n_warmup
                )

            else:
                raise ValueError(f"Unknown sampler: {self.sampler_name}")

            all_chain_samples.append(samples)

        # Combine chains
        combined = np.vstack(all_chain_samples)

        # Unpack into named trace
        samples_dict = {}
        idx = 0
        for name in self.model._param_order:
            dist, shape = self.model._latent_sites[name]
            if shape is None:
                samples_dict[name] = combined[:, idx]
                idx += 1
            else:
                size = int(np.prod(shape))
                samples_dict[name] = combined[:, idx:idx + size].reshape(-1, *shape)
                idx += size

        return Trace(samples_dict, self.model)

    def _get_init(self, rng):
        """Get initialization point."""
        dim = self.model.num_latent
        # Try prior samples, fall back to zeros
        for attempt in range(10):
            values = self.model.prior_sample(rng=rng)
            theta = self.model._pack_params(values)
            lp = self.model.log_joint(theta)
            if lp > -np.inf:
                return theta
        return np.zeros(dim)

    def _numerical_gradient(self, theta, eps=1e-5):
        """Numerical gradient of log_joint."""
        grad = np.zeros_like(theta)
        f0 = self.model.log_joint(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_p[i] += eps
            fp = self.model.log_joint(theta_p)
            if fp == -np.inf and f0 == -np.inf:
                grad[i] = 0.0
            elif fp == -np.inf:
                grad[i] = -1e6  # push away from boundary
            else:
                grad[i] = (fp - f0) / eps
        return grad


# ============================================================
# VI Inference
# ============================================================

class VIInference:
    """Variational inference backend using C154 methods."""

    METHODS = {'meanfield', 'blackbox', 'reparameterized', 'advi'}

    def __init__(self, model, method='meanfield', seed=42):
        self.model = model
        self.method_name = method
        self.seed = seed

    def run(self, n_iter=2000, lr=0.01, n_samples=10, **kwargs):
        """Run VI inference, return Trace with approximate posterior samples."""
        dim = self.model.num_latent

        def log_joint(z):
            return self.model.log_joint(np.array(z, dtype=float))

        if self.method_name == 'meanfield':
            vi = MeanFieldVI(dim=dim, log_joint=log_joint, seed=self.seed)
            vi.fit(n_iter=n_iter, lr=lr, n_samples=n_samples)
            posterior_samples = vi.sample_posterior(n=kwargs.get('n_posterior', 1000))
            elbo_history = vi.elbo_history

        elif self.method_name == 'blackbox':
            vi = BlackBoxVI(dim=dim, log_joint=log_joint, seed=self.seed)
            vi.fit(n_iter=n_iter, lr=lr, n_samples=max(n_samples, 20))
            posterior_samples = vi.sample_posterior(n=kwargs.get('n_posterior', 1000))
            elbo_history = vi.elbo_history

        elif self.method_name == 'reparameterized':
            vi = ReparameterizedVI(dim=dim, log_joint=log_joint, seed=self.seed)
            vi.fit(n_iter=n_iter, lr=lr, n_samples=n_samples)
            posterior_samples = vi.sample_posterior(n=kwargs.get('n_posterior', 1000))
            elbo_history = vi.elbo_history

        elif self.method_name == 'advi':
            transforms = []
            for name in self.model._param_order:
                dist, shape = self.model._latent_sites[name]
                t = dist.transform_type
                # ADVI expects 'unit_interval' not 'unit'
                if t == 'unit':
                    t = 'unit_interval'
                td = {'type': t}
                if shape is None:
                    transforms.append(td)
                else:
                    transforms.extend([td] * int(np.prod(shape)))

            vi = ADVI(dim=dim, log_joint=log_joint, transforms=transforms, seed=self.seed)
            vi.fit(n_iter=n_iter, lr=lr, n_samples=n_samples)
            posterior_samples = vi.sample_posterior(n=kwargs.get('n_posterior', 1000))
            elbo_history = vi.elbo_history

        else:
            raise ValueError(f"Unknown method: {self.method_name}")

        # Convert to ndarray if needed (some VI methods return list of lists)
        posterior_samples = np.array(posterior_samples, dtype=float)

        # Unpack into named trace
        samples_dict = {}
        idx = 0
        for name in self.model._param_order:
            dist, shape = self.model._latent_sites[name]
            if shape is None:
                samples_dict[name] = posterior_samples[:, idx]
                idx += 1
            else:
                size = int(np.prod(shape))
                samples_dict[name] = posterior_samples[:, idx:idx + size].reshape(-1, *shape)
                idx += size

        trace = Trace(samples_dict, self.model)
        trace.elbo_history = elbo_history
        return trace


# ============================================================
# Prior & Posterior Predictive
# ============================================================

class PriorPredictive:
    """Generate samples from the prior predictive distribution."""

    def __init__(self, model, seed=42):
        self.model = model
        self.rng = np.random.RandomState(seed)

    def sample(self, n=100):
        """Draw n samples from the prior predictive.

        Returns dict of {obs_name: array of samples per observation}.
        """
        predictions = {name: [] for name in self.model._observed_sites}

        for _ in range(n):
            values = self.model.prior_sample(rng=self.rng)

            # Compute deterministic sites
            for dname, fn in self.model._deterministic_sites.items():
                values[dname] = fn(values)

            for obs_name, (dist_or_fn, obs) in self.model._observed_sites.items():
                if callable(dist_or_fn) and not isinstance(dist_or_fn, Distribution):
                    dist = dist_or_fn(values)
                else:
                    dist = dist_or_fn
                pred = dist.sample(rng=self.rng, size=len(obs))
                predictions[obs_name].append(pred)

        return {k: np.array(v) for k, v in predictions.items()}


class PosteriorPredictive:
    """Generate samples from the posterior predictive distribution."""

    def __init__(self, model, trace, seed=42):
        self.model = model
        self.trace = trace
        self.rng = np.random.RandomState(seed)

    def sample(self, n=None):
        """Draw posterior predictive samples.

        Uses n posterior samples (default: all in trace).
        Returns dict of {obs_name: array of shape (n, n_obs)}.
        """
        if n is None:
            n = len(self.trace)
        n = min(n, len(self.trace))

        indices = self.rng.choice(len(self.trace), size=n, replace=False) if n < len(self.trace) else range(n)

        predictions = {name: [] for name in self.model._observed_sites}

        for i in indices:
            # Get posterior sample
            values = {}
            for name in self.model._param_order:
                s = self.trace.samples[name]
                values[name] = s[i]

            # Compute deterministic sites
            for dname, fn in self.model._deterministic_sites.items():
                values[dname] = fn(values)

            for obs_name, (dist_or_fn, obs) in self.model._observed_sites.items():
                if callable(dist_or_fn) and not isinstance(dist_or_fn, Distribution):
                    dist = dist_or_fn(values)
                else:
                    dist = dist_or_fn
                pred = dist.sample(rng=self.rng, size=len(obs))
                predictions[obs_name].append(pred)

        return {k: np.array(v) for k, v in predictions.items()}

    def mean_prediction(self, n=None):
        """Mean of posterior predictive samples."""
        preds = self.sample(n)
        return {k: np.mean(v, axis=0) for k, v in preds.items()}


# ============================================================
# Model Comparison
# ============================================================

class ModelComparison:
    """Model comparison metrics: WAIC, LOO-CV approximation, BIC."""

    @staticmethod
    def waic(model, trace):
        """Widely Applicable Information Criterion.

        WAIC = -2 * (lppd - p_waic)
        lppd = sum_i log(mean_s p(y_i | theta_s))
        p_waic = sum_i var_s(log p(y_i | theta_s))
        """
        n_samples = len(trace)

        # Collect all observed data points
        all_obs = []
        obs_sites = []
        for obs_name, (dist_or_fn, obs) in model._observed_sites.items():
            for y in obs.ravel():
                all_obs.append((obs_name, dist_or_fn, y))

        if not all_obs:
            return {'waic': 0.0, 'p_waic': 0.0, 'lppd': 0.0}

        # Compute pointwise log-likelihoods
        n_obs = len(all_obs)
        log_lik = np.zeros((n_samples, n_obs))

        for s in range(n_samples):
            values = {}
            for name in model._param_order:
                values[name] = trace.samples[name][s]

            for dname, fn in model._deterministic_sites.items():
                values[dname] = fn(values)

            for j, (obs_name, dist_or_fn, y) in enumerate(all_obs):
                if callable(dist_or_fn) and not isinstance(dist_or_fn, Distribution):
                    dist = dist_or_fn(values)
                else:
                    dist = dist_or_fn
                log_lik[s, j] = dist.log_prob(y)

        # lppd: log pointwise predictive density
        max_ll = np.max(log_lik, axis=0)
        lppd = np.sum(max_ll + np.log(np.mean(np.exp(log_lik - max_ll), axis=0)))

        # p_waic: effective number of parameters
        p_waic = np.sum(np.var(log_lik, axis=0))

        waic = -2 * (lppd - p_waic)

        return {'waic': float(waic), 'p_waic': float(p_waic), 'lppd': float(lppd)}

    @staticmethod
    def loo(model, trace):
        """Pareto-smoothed importance sampling LOO-CV (simplified).

        Uses importance weights from posterior for leave-one-out estimation.
        """
        n_samples = len(trace)

        all_obs = []
        for obs_name, (dist_or_fn, obs) in model._observed_sites.items():
            for y in obs.ravel():
                all_obs.append((obs_name, dist_or_fn, y))

        if not all_obs:
            return {'loo': 0.0, 'p_loo': 0.0}

        n_obs = len(all_obs)
        log_lik = np.zeros((n_samples, n_obs))

        for s in range(n_samples):
            values = {}
            for name in model._param_order:
                values[name] = trace.samples[name][s]

            for dname, fn in model._deterministic_sites.items():
                values[dname] = fn(values)

            for j, (obs_name, dist_or_fn, y) in enumerate(all_obs):
                if callable(dist_or_fn) and not isinstance(dist_or_fn, Distribution):
                    dist = dist_or_fn(values)
                else:
                    dist = dist_or_fn
                log_lik[s, j] = dist.log_prob(y)

        # LOO via importance sampling
        # log p(y_i | y_{-i}) approx -log(mean(1/p(y_i | theta_s)))
        loo_i = np.zeros(n_obs)
        for j in range(n_obs):
            # Negative log-lik as importance weights
            neg_ll = -log_lik[:, j]
            max_neg = np.max(neg_ll)
            # log(mean(exp(neg_ll))) = max_neg + log(mean(exp(neg_ll - max_neg)))
            log_mean_inv = max_neg + np.log(np.mean(np.exp(neg_ll - max_neg)))
            loo_i[j] = -log_mean_inv

        loo_total = np.sum(loo_i)

        # Full log predictive density for p_loo
        max_ll = np.max(log_lik, axis=0)
        lppd = np.sum(max_ll + np.log(np.mean(np.exp(log_lik - max_ll), axis=0)))
        p_loo = lppd - loo_total

        return {'loo': float(-2 * loo_total), 'p_loo': float(p_loo)}

    @staticmethod
    def bic(model, trace):
        """Bayesian Information Criterion.

        BIC = k * log(n) - 2 * log_lik_max
        """
        n_samples = len(trace)
        k = model.num_latent

        # Count observations
        n_obs = sum(len(obs.ravel()) for _, (_, obs) in model._observed_sites.items())

        # Find MAP (max log likelihood from trace)
        best_ll = -np.inf
        for s in range(n_samples):
            values = {}
            for name in model._param_order:
                values[name] = trace.samples[name][s]
            theta = model._pack_params(values)
            ll = model.log_likelihood(theta)
            if ll > best_ll:
                best_ll = ll

        bic = k * math.log(max(n_obs, 1)) - 2 * best_ll
        return {'bic': float(bic), 'k': k, 'n': n_obs, 'max_log_lik': float(best_ll)}

    @staticmethod
    def compare(models_traces):
        """Compare multiple models.

        models_traces: list of (name, model, trace) tuples
        Returns ranking by WAIC.
        """
        results = []
        for name, model, trace in models_traces:
            w = ModelComparison.waic(model, trace)
            results.append({
                'name': name,
                'waic': w['waic'],
                'p_waic': w['p_waic'],
                'lppd': w['lppd'],
            })

        results.sort(key=lambda r: r['waic'])
        for i, r in enumerate(results):
            r['rank'] = i + 1
            if i > 0:
                r['delta_waic'] = r['waic'] - results[0]['waic']
            else:
                r['delta_waic'] = 0.0

        return results


# ============================================================
# Convergence Diagnostics
# ============================================================

class ConvergenceDiagnostics:
    """Diagnostics for MCMC traces."""

    @staticmethod
    def summary(trace):
        """Full diagnostic summary."""
        return trace.summary()

    @staticmethod
    def check_convergence(trace, n_chains=4):
        """Check if chains have converged using R-hat and ESS."""
        results = {}
        for name in trace.param_names:
            ess = trace.effective_sample_size(name)
            rhat = trace.r_hat(name, n_chains=n_chains)
            converged = True
            if not np.isnan(rhat) and rhat > 1.1:
                converged = False
            ess_val = float(np.min(ess)) if isinstance(ess, np.ndarray) else float(ess)
            if ess_val < 100:
                converged = False
            results[name] = {
                'ess': float(ess_val),
                'r_hat': float(rhat),
                'converged': converged
            }
        return results

    @staticmethod
    def autocorrelation(trace, name, max_lag=50):
        """Compute autocorrelation for a parameter."""
        samples = trace.samples[name]
        n = len(samples)
        mean = np.mean(samples)
        var = np.var(samples)
        if var == 0:
            return np.zeros(min(max_lag, n))
        acf = []
        for lag in range(min(max_lag, n)):
            c = np.mean((samples[:n - lag] - mean) * (samples[lag:] - mean)) / var
            acf.append(c)
        return np.array(acf)


# ============================================================
# Model Builder Helpers
# ============================================================

def linear_regression(x, y, seed=42):
    """Build a Bayesian linear regression model.

    y ~ Normal(alpha + beta * x, sigma)
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.atleast_1d(np.array(y, dtype=float))

    model = ProbModel("linear_regression")
    model.sample("alpha", Normal(0, 10))
    model.sample("beta", Normal(0, 10))
    model.sample("sigma", HalfNormal(5))
    model.observe("y", lambda v: Normal(v["alpha"] + v["beta"] * x.mean(), v["sigma"]), y)

    return model


def logistic_regression(x, y, seed=42):
    """Build a Bayesian logistic regression model.

    p = sigmoid(alpha + beta * x)
    y ~ Bernoulli(p)
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.atleast_1d(np.array(y, dtype=float))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    model = ProbModel("logistic_regression")
    model.sample("alpha", Normal(0, 5))
    model.sample("beta", Normal(0, 5))

    # For logistic regression, each data point has its own probability
    # We need to observe each point separately
    for i in range(len(y)):
        xi = x[i]
        model.observe(f"y_{i}",
                      lambda v, xi=xi: Bernoulli(sigmoid(v["alpha"] + v["beta"] * xi)),
                      y[i])

    return model


def hierarchical_model(group_data, seed=42):
    """Build a hierarchical (random effects) model.

    group_data: dict of {group_name: array of observations}

    mu ~ Normal(0, 10)
    tau ~ HalfNormal(5)
    group_mean_i ~ Normal(mu, tau)
    sigma ~ HalfNormal(5)
    y_ij ~ Normal(group_mean_i, sigma)
    """
    model = ProbModel("hierarchical")
    model.sample("mu", Normal(0, 10))
    model.sample("tau", HalfNormal(5))
    model.sample("sigma", HalfNormal(5))

    for gname, gdata in group_data.items():
        gdata = np.atleast_1d(np.array(gdata, dtype=float))
        safe_name = str(gname).replace(' ', '_')
        model.sample(f"group_{safe_name}", Normal(0, 10))

        # Observe with group-specific mean (offset by global mu)
        model.observe(
            f"obs_{safe_name}",
            lambda v, sn=safe_name: Normal(v["mu"] + v[f"group_{sn}"] * v["tau"], v["sigma"]),
            gdata
        )

    return model


def mixture_model(data, k=2, seed=42):
    """Build a Gaussian mixture model.

    weights ~ Dirichlet(1, ..., 1)
    mu_i ~ Normal(0, 10)
    sigma_i ~ HalfNormal(5)
    y ~ sum_i weights_i * Normal(mu_i, sigma_i)
    """
    data = np.atleast_1d(np.array(data, dtype=float))

    model = ProbModel("mixture")

    # Component parameters
    for i in range(k):
        model.sample(f"mu_{i}", Normal(0, 10))
        model.sample(f"log_sigma_{i}", Normal(0, 1))

    # Mixture weight parameter (logit space)
    if k == 2:
        model.sample("logit_w", Normal(0, 1))

    # Observe data under mixture likelihood
    def mixture_dist(v):
        if k == 2:
            w = 1.0 / (1.0 + math.exp(-v["logit_w"]))
            weights = [w, 1 - w]
        else:
            weights = [1.0 / k] * k

        components = []
        for i in range(k):
            mu = v[f"mu_{i}"]
            sigma = math.exp(v[f"log_sigma_{i}"])
            components.append(Normal(mu, max(sigma, 1e-6)))

        return MixtureDistribution(weights, components)

    model.observe("y", mixture_dist, data)

    return model


class MixtureDistribution(Distribution):
    """Mixture of distributions."""

    def __init__(self, weights, components):
        self.weights = weights
        self.components = components

    def log_prob(self, x):
        log_probs = []
        for w, comp in zip(self.weights, self.components):
            if w > 0:
                lp = math.log(w) + comp.log_prob(x)
                log_probs.append(lp)
        if not log_probs:
            return -np.inf
        # Log-sum-exp for numerical stability
        max_lp = max(log_probs)
        if max_lp == -np.inf:
            return -np.inf
        return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

    def sample(self, rng=None, size=None):
        r = rng or np.random
        n = 1 if size is None else (size if isinstance(size, int) else int(np.prod(size)))
        samples = np.zeros(n)
        for i in range(n):
            k = r.choice(len(self.weights), p=np.array(self.weights) / sum(self.weights))
            samples[i] = self.components[k].sample(rng=r)
        if size is None:
            return float(samples[0])
        return samples


# ============================================================
# Plate (vectorized observations)
# ============================================================

class Plate:
    """Plate notation for repeated observations with shared parameters."""

    def __init__(self, model, name, data):
        self.model = model
        self.name = name
        self.data = np.atleast_1d(np.array(data, dtype=float))

    def observe(self, dist_fn):
        """Register all data points under this plate.

        dist_fn: callable(values_dict) -> Distribution
        """
        self.model.observe(self.name, dist_fn, self.data)
        return self


# ============================================================
# Guide (custom variational family)
# ============================================================

class Guide:
    """Custom variational guide for more flexible inference."""

    def __init__(self, model):
        self.model = model
        self._guide_dists = OrderedDict()

    def register(self, name, dist):
        """Register a guide distribution for a latent variable."""
        if name not in self.model._latent_sites:
            raise ValueError(f"'{name}' is not a latent site in the model")
        self._guide_dists[name] = dist
        return self

    def log_prob(self, theta):
        """Compute guide log probability."""
        values = self.model._unpack_params(theta)
        lp = 0.0
        for name in self.model._param_order:
            if name in self._guide_dists:
                dist = self._guide_dists[name]
            else:
                # Default: standard normal
                dist = Normal(0, 1)
            v = values[name]
            lp += dist.log_prob(v)
        return lp

    def sample(self, rng=None):
        """Sample from the guide."""
        values = {}
        for name in self.model._param_order:
            if name in self._guide_dists:
                dist = self._guide_dists[name]
            else:
                dist = Normal(0, 1)
            _, shape = self.model._latent_sites[name]
            values[name] = dist.sample(rng=rng, size=shape)
            if shape is None:
                values[name] = float(values[name])
        return values
