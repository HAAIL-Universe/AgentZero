"""
V219: Particle Filter -- Sequential Monte Carlo for nonlinear/non-Gaussian estimation.

Where the Kalman filter (V218) is optimal for linear-Gaussian systems, the
particle filter handles arbitrary nonlinear dynamics and non-Gaussian noise
by representing the posterior as a weighted set of samples (particles).

Components:
1. ParticleFilter (SIR/Bootstrap) -- Standard importance sampling with resampling
2. AuxiliaryParticleFilter -- Look-ahead auxiliary variable method
3. RegularizedParticleFilter -- Kernel-smoothed resampling (avoids sample impoverishment)
4. RaoBlackwellizedPF -- Marginalize linear sub-state analytically (Kalman + particles)
5. ParticleSmoother -- Fixed-lag and fixed-interval smoothing
6. ResamplingMethods -- Multinomial, systematic, stratified, residual

Uses NumPy for numerical computation (permitted by CLAUDE.md).

Composes with:
- V218 (Kalman Filter): Rao-Blackwellization, benchmark comparison
- V215 (HMM): discrete-state particle filter comparison
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
from enum import Enum


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Particle:
    """A single weighted particle."""
    state: np.ndarray   # (n,)
    weight: float        # unnormalized log-weight


@dataclass
class ParticleSet:
    """Weighted particle set representing a distribution."""
    states: np.ndarray   # (N, n) -- N particles, n dimensions
    weights: np.ndarray  # (N,) -- normalized weights (sum to 1)

    @property
    def n_particles(self) -> int:
        return self.states.shape[0]

    @property
    def dim(self) -> int:
        return self.states.shape[1]

    def mean(self) -> np.ndarray:
        """Weighted mean."""
        return self.weights @ self.states

    def covariance(self) -> np.ndarray:
        """Weighted covariance."""
        mu = self.mean()
        diff = self.states - mu  # (N, n)
        return (diff * self.weights[:, None]).T @ diff

    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2). Measures weight degeneracy."""
        return 1.0 / np.sum(self.weights ** 2)

    def percentile(self, q: float, dim: int = 0) -> float:
        """Weighted percentile along dimension dim."""
        idx = np.argsort(self.states[:, dim])
        sorted_vals = self.states[idx, dim]
        sorted_w = self.weights[idx]
        cumw = np.cumsum(sorted_w)
        return float(sorted_vals[np.searchsorted(cumw, q / 100.0)])

    def map_estimate(self) -> np.ndarray:
        """Maximum a posteriori (highest weight particle)."""
        return self.states[np.argmax(self.weights)].copy()


@dataclass
class PFResult:
    """Result of running a particle filter over a sequence of observations."""
    filtered_sets: list       # list of ParticleSet (after update at each step)
    predicted_sets: list      # list of ParticleSet (after predict at each step)
    log_likelihood: float     # total marginal log-likelihood estimate
    ess_history: list         # ESS at each step
    resample_count: int       # number of resampling events


# ---------------------------------------------------------------------------
# Resampling methods
# ---------------------------------------------------------------------------

class ResampleMethod(Enum):
    MULTINOMIAL = "multinomial"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    RESIDUAL = "residual"


def _resample_multinomial(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multinomial resampling. Returns indices."""
    N = len(weights)
    return rng.choice(N, size=N, replace=True, p=weights)


def _resample_systematic(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling -- single uniform draw, evenly spaced."""
    N = len(weights)
    positions = (rng.uniform() + np.arange(N)) / N
    cumw = np.cumsum(weights)
    indices = np.searchsorted(cumw, positions)
    return np.clip(indices, 0, N - 1)


def _resample_stratified(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stratified resampling -- one uniform draw per stratum."""
    N = len(weights)
    positions = (rng.uniform(size=N) + np.arange(N)) / N
    cumw = np.cumsum(weights)
    indices = np.searchsorted(cumw, positions)
    return np.clip(indices, 0, N - 1)


def _resample_residual(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Residual resampling -- deterministic + multinomial on residuals."""
    N = len(weights)
    nw = N * weights
    floor_nw = np.floor(nw).astype(int)
    residuals = nw - floor_nw
    n_deterministic = int(np.sum(floor_nw))

    indices = []
    for i, count in enumerate(floor_nw):
        indices.extend([i] * count)

    n_remaining = N - n_deterministic
    if n_remaining > 0:
        res_sum = np.sum(residuals)
        if res_sum > 0:
            res_weights = residuals / res_sum
            extra = rng.choice(N, size=n_remaining, replace=True, p=res_weights)
            indices.extend(extra.tolist())

    return np.array(indices[:N])


def resample(weights: np.ndarray, method: ResampleMethod,
             rng: np.random.Generator) -> np.ndarray:
    """Dispatch to resampling method. Returns indices."""
    dispatch = {
        ResampleMethod.MULTINOMIAL: _resample_multinomial,
        ResampleMethod.SYSTEMATIC: _resample_systematic,
        ResampleMethod.STRATIFIED: _resample_stratified,
        ResampleMethod.RESIDUAL: _resample_residual,
    }
    return dispatch[method](weights, rng)


# ---------------------------------------------------------------------------
# Log-weight utilities
# ---------------------------------------------------------------------------

def _log_norm_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log of multivariate normal PDF."""
    k = len(mean)
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -1e300
    inv_cov = np.linalg.inv(cov)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + float(diff @ inv_cov @ diff))


def _logsumexp(log_w: np.ndarray) -> float:
    """Log-sum-exp for numerical stability."""
    max_w = np.max(log_w)
    if max_w == -np.inf:
        return -np.inf
    return max_w + np.log(np.sum(np.exp(log_w - max_w)))


def _normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    """Convert log-weights to normalized weights."""
    lse = _logsumexp(log_w)
    if lse == -np.inf:
        return np.ones(len(log_w)) / len(log_w)
    w = np.exp(log_w - lse)
    w = np.maximum(w, 0)
    s = np.sum(w)
    if s == 0:
        return np.ones(len(log_w)) / len(log_w)
    return w / s


# ---------------------------------------------------------------------------
# Standard Particle Filter (SIR / Bootstrap)
# ---------------------------------------------------------------------------

class ParticleFilter:
    """
    Sequential Importance Resampling (SIR) / Bootstrap Particle Filter.

    transition_fn(x, rng) -> x_next    -- sample from p(x_t | x_{t-1})
    log_likelihood_fn(x, y) -> float   -- log p(y_t | x_t)
    """

    def __init__(self,
                 transition_fn: Callable,
                 log_likelihood_fn: Callable,
                 n_particles: int = 1000,
                 resample_method: ResampleMethod = ResampleMethod.SYSTEMATIC,
                 ess_threshold: float = 0.5,
                 seed: int = 42):
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold
        self.rng = np.random.default_rng(seed)

    def initialize(self, prior_sampler: Callable) -> ParticleSet:
        """Draw initial particles from prior."""
        states = np.array([prior_sampler(self.rng) for _ in range(self.n_particles)])
        weights = np.ones(self.n_particles) / self.n_particles
        return ParticleSet(states=states, weights=weights)

    def predict(self, ps: ParticleSet) -> ParticleSet:
        """Propagate particles through transition model."""
        new_states = np.array([
            self.transition_fn(ps.states[i], self.rng)
            for i in range(ps.n_particles)
        ])
        return ParticleSet(states=new_states, weights=ps.weights.copy())

    def update(self, ps: ParticleSet, observation: np.ndarray) -> Tuple[ParticleSet, float]:
        """Weight particles by observation likelihood. Returns (updated, log_marginal)."""
        log_w = np.array([
            np.log(ps.weights[i] + 1e-300) + self.log_likelihood_fn(ps.states[i], observation)
            for i in range(ps.n_particles)
        ])
        log_marginal = _logsumexp(log_w)
        weights = _normalize_log_weights(log_w)
        return ParticleSet(states=ps.states.copy(), weights=weights), log_marginal

    def resample_if_needed(self, ps: ParticleSet) -> Tuple[ParticleSet, bool]:
        """Resample if ESS drops below threshold."""
        ess = ps.effective_sample_size()
        if ess < self.ess_threshold * ps.n_particles:
            indices = resample(ps.weights, self.resample_method, self.rng)
            new_states = ps.states[indices].copy()
            new_weights = np.ones(ps.n_particles) / ps.n_particles
            return ParticleSet(states=new_states, weights=new_weights), True
        return ps, False

    def step(self, ps: ParticleSet, observation: np.ndarray) -> Tuple[ParticleSet, float, bool]:
        """One predict-update-resample cycle."""
        predicted = self.predict(ps)
        updated, log_marginal = self.update(predicted, observation)
        resampled, did_resample = self.resample_if_needed(updated)
        return resampled, log_marginal, did_resample

    def filter(self, observations: np.ndarray,
               prior_sampler: Callable) -> PFResult:
        """Run filter over full observation sequence."""
        ps = self.initialize(prior_sampler)
        filtered_sets = []
        predicted_sets = []
        ess_history = []
        total_ll = 0.0
        resample_count = 0

        for t in range(len(observations)):
            predicted = self.predict(ps)
            predicted_sets.append(ParticleSet(
                states=predicted.states.copy(),
                weights=predicted.weights.copy()))

            updated, log_marginal = self.update(predicted, observations[t])
            total_ll += log_marginal

            resampled, did_resample = self.resample_if_needed(updated)
            if did_resample:
                resample_count += 1

            ess_history.append(updated.effective_sample_size())
            filtered_sets.append(ParticleSet(
                states=resampled.states.copy(),
                weights=resampled.weights.copy()))
            ps = resampled

        return PFResult(
            filtered_sets=filtered_sets,
            predicted_sets=predicted_sets,
            log_likelihood=total_ll,
            ess_history=ess_history,
            resample_count=resample_count)


# ---------------------------------------------------------------------------
# Auxiliary Particle Filter
# ---------------------------------------------------------------------------

class AuxiliaryParticleFilter:
    """
    APF with first-stage weights based on predictive likelihood.

    Uses a point estimate (e.g., transition mean) to look ahead and
    pre-select particles likely to match the next observation.
    """

    def __init__(self,
                 transition_fn: Callable,
                 transition_mean_fn: Callable,
                 log_likelihood_fn: Callable,
                 n_particles: int = 1000,
                 resample_method: ResampleMethod = ResampleMethod.SYSTEMATIC,
                 seed: int = 42):
        self.transition_fn = transition_fn
        self.transition_mean_fn = transition_mean_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.rng = np.random.default_rng(seed)

    def filter(self, observations: np.ndarray,
               prior_sampler: Callable) -> PFResult:
        """Run APF over observation sequence."""
        N = self.n_particles
        states = np.array([prior_sampler(self.rng) for _ in range(N)])
        weights = np.ones(N) / N

        filtered_sets = []
        predicted_sets = []
        ess_history = []
        total_ll = 0.0
        resample_count = 0

        for t in range(len(observations)):
            y = observations[t]

            # First stage: compute predictive likelihoods using transition mean
            mu = np.array([self.transition_mean_fn(states[i]) for i in range(N)])
            first_stage_log_w = np.array([
                np.log(weights[i] + 1e-300) + self.log_likelihood_fn(mu[i], y)
                for i in range(N)
            ])
            first_stage_w = _normalize_log_weights(first_stage_log_w)

            # Resample using first-stage weights
            indices = resample(first_stage_w, self.resample_method, self.rng)
            resample_count += 1

            # Propagate resampled particles
            new_states = np.array([
                self.transition_fn(states[indices[i]], self.rng)
                for i in range(N)
            ])

            predicted_sets.append(ParticleSet(
                states=new_states.copy(),
                weights=np.ones(N) / N))

            # Second stage: correct weights
            log_w2 = np.array([
                self.log_likelihood_fn(new_states[i], y)
                - self.log_likelihood_fn(mu[indices[i]], y)
                for i in range(N)
            ])
            log_marginal = _logsumexp(first_stage_log_w) + _logsumexp(log_w2) - np.log(N)
            total_ll += log_marginal

            weights = _normalize_log_weights(log_w2)
            states = new_states

            ess_history.append(1.0 / np.sum(weights ** 2))
            filtered_sets.append(ParticleSet(
                states=states.copy(), weights=weights.copy()))

        return PFResult(
            filtered_sets=filtered_sets,
            predicted_sets=predicted_sets,
            log_likelihood=total_ll,
            ess_history=ess_history,
            resample_count=resample_count)


# ---------------------------------------------------------------------------
# Regularized Particle Filter
# ---------------------------------------------------------------------------

class RegularizedParticleFilter:
    """
    RPF -- kernel-smoothed resampling to combat sample impoverishment.

    After resampling, each particle is jittered by a kernel (Gaussian)
    scaled by the optimal bandwidth (Silverman's rule).
    """

    def __init__(self,
                 transition_fn: Callable,
                 log_likelihood_fn: Callable,
                 n_particles: int = 1000,
                 resample_method: ResampleMethod = ResampleMethod.SYSTEMATIC,
                 ess_threshold: float = 0.5,
                 bandwidth_scale: float = 1.0,
                 seed: int = 42):
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold
        self.bandwidth_scale = bandwidth_scale
        self.rng = np.random.default_rng(seed)

    def _optimal_bandwidth(self, ps: ParticleSet) -> np.ndarray:
        """Silverman's rule for kernel bandwidth matrix."""
        n = ps.dim
        N_eff = ps.effective_sample_size()
        cov = ps.covariance()
        # Silverman's rule: h = (4/(n+2))^(1/(n+4)) * N_eff^(-1/(n+4))
        h = ((4.0 / (n + 2)) ** (1.0 / (n + 4))) * (max(N_eff, 1) ** (-1.0 / (n + 4)))
        return h * self.bandwidth_scale * cov

    def filter(self, observations: np.ndarray,
               prior_sampler: Callable) -> PFResult:
        """Run regularized PF."""
        N = self.n_particles
        states = np.array([prior_sampler(self.rng) for _ in range(N)])
        weights = np.ones(N) / N

        filtered_sets = []
        predicted_sets = []
        ess_history = []
        total_ll = 0.0
        resample_count = 0

        for t in range(len(observations)):
            # Predict
            new_states = np.array([
                self.transition_fn(states[i], self.rng) for i in range(N)
            ])
            predicted_sets.append(ParticleSet(
                states=new_states.copy(), weights=weights.copy()))

            # Update
            log_w = np.array([
                np.log(weights[i] + 1e-300) +
                self.log_likelihood_fn(new_states[i], observations[t])
                for i in range(N)
            ])
            log_marginal = _logsumexp(log_w)
            total_ll += log_marginal
            weights = _normalize_log_weights(log_w)
            states = new_states

            ps = ParticleSet(states=states, weights=weights)
            ess = ps.effective_sample_size()
            ess_history.append(ess)

            # Regularized resampling
            if ess < self.ess_threshold * N:
                resample_count += 1
                kernel_cov = self._optimal_bandwidth(ps)
                indices = resample(weights, self.resample_method, self.rng)
                states = states[indices].copy()

                # Add kernel jitter
                try:
                    L = np.linalg.cholesky(kernel_cov)
                    noise = self.rng.standard_normal((N, ps.dim)) @ L.T
                except np.linalg.LinAlgError:
                    # Fallback: diagonal jitter
                    std = np.sqrt(np.maximum(np.diag(kernel_cov), 1e-10))
                    noise = self.rng.standard_normal((N, ps.dim)) * std

                states = states + noise
                weights = np.ones(N) / N

            filtered_sets.append(ParticleSet(
                states=states.copy(), weights=weights.copy()))

        return PFResult(
            filtered_sets=filtered_sets,
            predicted_sets=predicted_sets,
            log_likelihood=total_ll,
            ess_history=ess_history,
            resample_count=resample_count)


# ---------------------------------------------------------------------------
# Rao-Blackwellized Particle Filter
# ---------------------------------------------------------------------------

@dataclass
class RBParticle:
    """Particle with nonlinear state + linear sub-state (Kalman)."""
    nonlinear: np.ndarray    # nonlinear state component
    kf_mean: np.ndarray      # linear sub-state mean (Kalman)
    kf_cov: np.ndarray       # linear sub-state covariance


class RaoBlackwellizedPF:
    """
    RBPF -- marginalize linear sub-state analytically via Kalman filter.

    State = [nonlinear_part, linear_part].
    The nonlinear part is tracked by particles.
    The linear part (conditioned on nonlinear) is tracked by per-particle Kalman filters.

    Requires:
    - nonlinear_transition_fn(x_nl, rng) -> x_nl_next
    - linear_dynamics_fn(x_nl) -> (F, B, Q) for x_l_next = F*x_l + B*x_nl + noise(Q)
    - observation_fn(x_nl) -> (H, R) for y = H*x_l + noise(R)
    - log_nonlinear_obs_fn(x_nl, y) -> float  (optional direct nonlinear obs contribution)
    """

    def __init__(self,
                 nonlinear_transition_fn: Callable,
                 linear_dynamics_fn: Callable,
                 observation_fn: Callable,
                 n_particles: int = 500,
                 resample_method: ResampleMethod = ResampleMethod.SYSTEMATIC,
                 ess_threshold: float = 0.5,
                 log_nonlinear_obs_fn: Optional[Callable] = None,
                 seed: int = 42):
        self.nonlinear_transition_fn = nonlinear_transition_fn
        self.linear_dynamics_fn = linear_dynamics_fn
        self.observation_fn = observation_fn
        self.n_particles = n_particles
        self.resample_method = resample_method
        self.ess_threshold = ess_threshold
        self.log_nonlinear_obs_fn = log_nonlinear_obs_fn
        self.rng = np.random.default_rng(seed)

    def filter(self, observations: np.ndarray,
               nl_prior_sampler: Callable,
               linear_dim: int,
               linear_prior_mean: Optional[np.ndarray] = None,
               linear_prior_cov: Optional[np.ndarray] = None) -> PFResult:
        """Run RBPF."""
        N = self.n_particles

        # Initialize
        particles = []
        for i in range(N):
            x_nl = nl_prior_sampler(self.rng)
            kf_mean = linear_prior_mean if linear_prior_mean is not None else np.zeros(linear_dim)
            kf_cov = linear_prior_cov if linear_prior_cov is not None else np.eye(linear_dim) * 10.0
            particles.append(RBParticle(
                nonlinear=x_nl.copy(),
                kf_mean=kf_mean.copy(),
                kf_cov=kf_cov.copy()))

        weights = np.ones(N) / N

        filtered_sets = []
        predicted_sets = []
        ess_history = []
        total_ll = 0.0
        resample_count = 0

        for t in range(len(observations)):
            y = observations[t]
            log_w = np.zeros(N)

            new_particles = []
            for i in range(N):
                p = particles[i]

                # Propagate nonlinear state
                x_nl_new = self.nonlinear_transition_fn(p.nonlinear, self.rng)

                # Kalman predict for linear sub-state
                F, B, Q = self.linear_dynamics_fn(x_nl_new)
                kf_mean_pred = F @ p.kf_mean + B @ x_nl_new
                kf_cov_pred = F @ p.kf_cov @ F.T + Q

                # Kalman update with observation
                H, R = self.observation_fn(x_nl_new)
                innov = y - H @ kf_mean_pred
                S = H @ kf_cov_pred @ H.T + R
                K = kf_cov_pred @ H.T @ np.linalg.inv(S)
                kf_mean_upd = kf_mean_pred + K @ innov
                kf_cov_upd = (np.eye(len(kf_mean_pred)) - K @ H) @ kf_cov_pred

                # Weight: p(y | x_nl) using innovation likelihood
                log_w[i] = np.log(weights[i] + 1e-300) + _log_norm_pdf(y, H @ kf_mean_pred, S)

                if self.log_nonlinear_obs_fn is not None:
                    log_w[i] += self.log_nonlinear_obs_fn(x_nl_new, y)

                new_particles.append(RBParticle(
                    nonlinear=x_nl_new,
                    kf_mean=kf_mean_upd,
                    kf_cov=kf_cov_upd))

            log_marginal = _logsumexp(log_w)
            total_ll += log_marginal
            weights = _normalize_log_weights(log_w)
            particles = new_particles

            # Store as ParticleSet (nonlinear states only for visualization)
            nl_states = np.array([p.nonlinear for p in particles])
            predicted_sets.append(ParticleSet(
                states=nl_states.copy(), weights=weights.copy()))

            ess = 1.0 / np.sum(weights ** 2)
            ess_history.append(ess)

            # Resample
            if ess < self.ess_threshold * N:
                resample_count += 1
                indices = resample(weights, self.resample_method, self.rng)
                particles = [RBParticle(
                    nonlinear=particles[j].nonlinear.copy(),
                    kf_mean=particles[j].kf_mean.copy(),
                    kf_cov=particles[j].kf_cov.copy()
                ) for j in indices]
                weights = np.ones(N) / N

            nl_states = np.array([p.nonlinear for p in particles])
            filtered_sets.append(ParticleSet(
                states=nl_states.copy(), weights=weights.copy()))

        return PFResult(
            filtered_sets=filtered_sets,
            predicted_sets=predicted_sets,
            log_likelihood=total_ll,
            ess_history=ess_history,
            resample_count=resample_count)


# ---------------------------------------------------------------------------
# Particle Smoother
# ---------------------------------------------------------------------------

class ParticleSmoother:
    """
    Fixed-lag smoother using backward weighting.

    Given filtered particle sets, compute smoothed weights by
    backward reweighting: w_smooth(t) ~ sum_j w(t+1) * p(x_{t+1}^j | x_t^i).
    """

    def __init__(self, transition_log_density: Callable, lag: int = 5):
        """
        transition_log_density(x_next, x_curr) -> float
            Log transition density p(x_next | x_curr).
        lag: smoothing lag (0 = filter, larger = smoother but costlier).
        """
        self.transition_log_density = transition_log_density
        self.lag = lag

    def smooth(self, filtered_sets: List[ParticleSet]) -> List[ParticleSet]:
        """Apply fixed-lag smoothing to filtered particle sets."""
        T = len(filtered_sets)
        smoothed = [None] * T

        # Last time step = filtered
        smoothed[T - 1] = ParticleSet(
            states=filtered_sets[T - 1].states.copy(),
            weights=filtered_sets[T - 1].weights.copy())

        for t in range(T - 2, -1, -1):
            ps_t = filtered_sets[t]
            # How far to look ahead (bounded by lag and remaining steps)
            look_ahead = min(self.lag, T - 1 - t)
            ps_next = filtered_sets[t + 1] if look_ahead > 0 else ps_t

            N = ps_t.n_particles
            N_next = ps_next.n_particles

            # Backward weights: for each particle at t, sum contributions from t+1
            log_smooth_w = np.zeros(N)
            for i in range(N):
                log_contributions = np.zeros(N_next)
                for j in range(N_next):
                    log_contributions[j] = (
                        np.log(ps_next.weights[j] + 1e-300) +
                        self.transition_log_density(ps_next.states[j], ps_t.states[i])
                    )
                # Normalize by sum over all source particles (denominator)
                log_denom_parts = np.zeros(N)
                for k in range(N):
                    log_denom_parts[k] = (
                        np.log(ps_t.weights[k] + 1e-300) +
                        self.transition_log_density(ps_next.states[0], ps_t.states[k])
                    )

                log_smooth_w[i] = np.log(ps_t.weights[i] + 1e-300) + _logsumexp(log_contributions)

            weights = _normalize_log_weights(log_smooth_w)
            smoothed[t] = ParticleSet(
                states=ps_t.states.copy(),
                weights=weights)

        return smoothed


# ---------------------------------------------------------------------------
# Utility: simulate nonlinear system
# ---------------------------------------------------------------------------

def simulate_nonlinear_system(transition_fn: Callable,
                               observation_fn: Callable,
                               x0: np.ndarray,
                               T: int,
                               rng: np.random.Generator
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a nonlinear state-space model.
    transition_fn(x, rng) -> x_next
    observation_fn(x, rng) -> y
    Returns (states, observations) arrays.
    """
    n_state = len(x0)
    states = np.zeros((T + 1, n_state))
    states[0] = x0

    obs_list = []
    for t in range(T):
        states[t + 1] = transition_fn(states[t], rng)
        obs_list.append(observation_fn(states[t + 1], rng))

    observations = np.array(obs_list)
    return states, observations


# ---------------------------------------------------------------------------
# Comparison with Kalman (V218 bridge)
# ---------------------------------------------------------------------------

def compare_with_kalman(pf_result: PFResult,
                        kf_means: np.ndarray,
                        true_states: np.ndarray) -> dict:
    """
    Compare particle filter vs Kalman filter estimates against ground truth.
    Returns dict with RMSE for each method.
    """
    T = len(pf_result.filtered_sets)
    pf_means = np.array([ps.mean() for ps in pf_result.filtered_sets])

    n = min(T, len(kf_means), len(true_states) - 1)
    pf_rmse = np.sqrt(np.mean((pf_means[:n] - true_states[1:n+1]) ** 2))
    kf_rmse = np.sqrt(np.mean((kf_means[:n] - true_states[1:n+1]) ** 2))

    return {
        "pf_rmse": float(pf_rmse),
        "kf_rmse": float(kf_rmse),
        "pf_mean_ess": float(np.mean(pf_result.ess_history)),
        "pf_resample_rate": pf_result.resample_count / max(T, 1),
        "pf_log_likelihood": pf_result.log_likelihood,
    }


# ---------------------------------------------------------------------------
# Example models
# ---------------------------------------------------------------------------

def make_linear_gaussian_model(F: np.ndarray, H: np.ndarray,
                                Q: np.ndarray, R: np.ndarray):
    """Create transition/observation functions for a linear-Gaussian model."""
    def transition_fn(x, rng):
        return F @ x + rng.multivariate_normal(np.zeros(F.shape[0]), Q)

    def log_likelihood_fn(x, y):
        return _log_norm_pdf(y, H @ x, R)

    def observation_fn(x, rng):
        return H @ x + rng.multivariate_normal(np.zeros(H.shape[0]), R)

    return transition_fn, log_likelihood_fn, observation_fn


def make_bearings_only_model(process_noise: float = 0.5,
                              obs_noise: float = 0.1,
                              sensor_pos: np.ndarray = None):
    """
    Bearings-only tracking: observe angle to target, not distance.
    Classic nonlinear problem where Kalman fails but PF excels.

    State: [x, y, vx, vy]
    Observation: atan2(y - sy, x - sx) + noise
    """
    if sensor_pos is None:
        sensor_pos = np.array([0.0, 0.0])

    dt = 1.0
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Q = process_noise ** 2 * np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ])

    def transition_fn(x, rng):
        noise = rng.multivariate_normal(np.zeros(4), Q)
        return F @ x + noise

    def observation_fn(x, rng):
        angle = np.arctan2(x[1] - sensor_pos[1], x[0] - sensor_pos[0])
        return np.array([angle + rng.normal(0, obs_noise)])

    def log_likelihood_fn(x, y):
        angle = np.arctan2(x[1] - sensor_pos[1], x[0] - sensor_pos[0])
        diff = y[0] - angle
        # Wrap to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return -0.5 * (diff ** 2) / (obs_noise ** 2) - 0.5 * np.log(2 * np.pi * obs_noise ** 2)

    return transition_fn, log_likelihood_fn, observation_fn, F, Q


def make_stochastic_volatility_model(phi: float = 0.98,
                                      sigma_v: float = 0.16,
                                      beta: float = 0.65):
    """
    Stochastic volatility model for financial time series.

    State: x_t = phi * x_{t-1} + sigma_v * v_t  (log-volatility)
    Observation: y_t = beta * exp(x_t/2) * e_t  (returns)

    Non-Gaussian observation (heavy-tailed through stochastic variance).
    """
    def transition_fn(x, rng):
        return np.array([phi * x[0] + sigma_v * rng.standard_normal()])

    def observation_fn(x, rng):
        vol = beta * np.exp(x[0] / 2)
        return np.array([vol * rng.standard_normal()])

    def log_likelihood_fn(x, y):
        vol = beta * np.exp(x[0] / 2)
        return -0.5 * np.log(2 * np.pi) - np.log(vol + 1e-300) - 0.5 * (y[0] / (vol + 1e-300)) ** 2

    return transition_fn, log_likelihood_fn, observation_fn
