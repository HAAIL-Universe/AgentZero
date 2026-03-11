"""
C159: Particle Filter (Sequential Monte Carlo)
Composing C158 (Kalman Filter) + C153 (Monte Carlo)

Sequential Monte Carlo methods for nonlinear, non-Gaussian state estimation.

Components:
- ParticleFilter: Bootstrap/SIR particle filter (predict/update/resample)
- AuxiliaryParticleFilter: APF with first-stage weights
- RegularizedParticleFilter: RPF with kernel smoothing
- RaoBlackwellizedPF: RBPF combining particles with Kalman filters
- ParticleSmoother: Fixed-lag and FFBSi smoothing
- AdaptiveParticleFilter: Adaptive resampling with ESS threshold
- MultipleModelPF: Particle filter for switching/jump systems
- ParticleFilterUtils: ESS, resampling schemes, diagnostics
"""

import numpy as np
from numpy.linalg import inv, det, cholesky
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C158_kalman_filter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C153_monte_carlo'))
from kalman_filter import KalmanFilter
from monte_carlo import RandomSampler


# ---------------------------------------------------------------------------
# Utility: resampling schemes
# ---------------------------------------------------------------------------

class ParticleFilterUtils:
    """Resampling schemes and diagnostics for particle filters."""

    @staticmethod
    def effective_sample_size(weights):
        """Compute ESS = 1 / sum(w_i^2). Max = N when uniform."""
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return 1.0 / np.sum(w ** 2)

    @staticmethod
    def multinomial_resample(weights, rng=None):
        """Multinomial resampling. Returns indices."""
        rng = rng or np.random.RandomState()
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        N = len(w)
        return rng.choice(N, size=N, replace=True, p=w)

    @staticmethod
    def systematic_resample(weights, rng=None):
        """Systematic resampling (low variance). Returns indices."""
        rng = rng or np.random.RandomState()
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        N = len(w)
        positions = (np.arange(N) + rng.uniform()) / N
        cumsum = np.cumsum(w)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)
        return indices

    @staticmethod
    def stratified_resample(weights, rng=None):
        """Stratified resampling. Returns indices."""
        rng = rng or np.random.RandomState()
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        N = len(w)
        positions = (np.arange(N) + rng.uniform(size=N)) / N
        cumsum = np.cumsum(w)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)
        return indices

    @staticmethod
    def residual_resample(weights, rng=None):
        """Residual resampling. Returns indices."""
        rng = rng or np.random.RandomState()
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        N = len(w)
        # Deterministic part
        counts = np.floor(N * w).astype(int)
        residuals = N * w - counts
        # Stochastic part for residuals
        remainder = N - counts.sum()
        if remainder > 0:
            residuals = residuals / residuals.sum()
            extra = rng.choice(N, size=remainder, replace=True, p=residuals)
            for idx in extra:
                counts[idx] += 1
        indices = []
        for i, c in enumerate(counts):
            indices.extend([i] * c)
        return np.array(indices)

    @staticmethod
    def neff_history(weight_history):
        """Compute ESS over time from weight history."""
        return [ParticleFilterUtils.effective_sample_size(w) for w in weight_history]

    @staticmethod
    def log_marginal_likelihood(weight_history):
        """Estimate log p(y_{1:T}) from unnormalized weight history."""
        total = 0.0
        for w in weight_history:
            w = np.asarray(w, dtype=float)
            avg = np.mean(w)
            if avg > 0:
                total += np.log(avg)
        return total

    @staticmethod
    def particle_diversity(particles):
        """Fraction of unique particles."""
        N = len(particles)
        if N == 0:
            return 0.0
        unique = len(set(map(lambda p: tuple(np.asarray(p).flatten()), particles)))
        return unique / N


# ---------------------------------------------------------------------------
# Bootstrap / SIR Particle Filter
# ---------------------------------------------------------------------------

class ParticleFilter:
    """Bootstrap (SIR) particle filter.

    State model:  x_k = f(x_{k-1}) + process_noise
    Obs model:    z_k ~ p(z|x_k)

    Args:
        num_particles: Number of particles
        dim_x: State dimension
        transition_fn: f(particles, k) -> new_particles (vectorized, N x dim_x)
        log_likelihood_fn: log p(z_k | x_k) for each particle -> (N,)
        process_noise_fn: sample noise -> (N, dim_x), or None for transition_fn that includes noise
        initial_fn: sample initial particles -> (N, dim_x)
        resample_method: 'systematic', 'multinomial', 'stratified', 'residual'
        resample_threshold: ESS/N threshold for adaptive resampling (1.0 = always)
        seed: random seed
    """

    def __init__(self, num_particles, dim_x, transition_fn, log_likelihood_fn,
                 process_noise_fn=None, initial_fn=None,
                 resample_method='systematic', resample_threshold=0.5, seed=None):
        self.N = num_particles
        self.dim_x = dim_x
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.process_noise_fn = process_noise_fn
        self.initial_fn = initial_fn
        self.resample_method = resample_method
        self.resample_threshold = resample_threshold
        self.rng = np.random.RandomState(seed)

        # State
        self.particles = None  # (N, dim_x)
        self.weights = np.ones(num_particles) / num_particles
        self.log_weights = np.full(num_particles, -np.log(num_particles))

        # History
        self.particle_history = []
        self.weight_history = []
        self.estimate_history = []

    def initialize(self, particles=None):
        """Initialize particles."""
        if particles is not None:
            self.particles = np.array(particles, dtype=float)
        elif self.initial_fn is not None:
            self.particles = self.initial_fn(self.N, self.rng)
        else:
            self.particles = self.rng.randn(self.N, self.dim_x)
        self.weights = np.ones(self.N) / self.N
        self.log_weights = np.full(self.N, -np.log(self.N))
        self.particle_history = []
        self.weight_history = []
        self.estimate_history = []

    def predict(self, k=0):
        """Propagate particles through transition model."""
        self.particles = self.transition_fn(self.particles, k)
        if self.process_noise_fn is not None:
            noise = self.process_noise_fn(self.N, self.rng)
            self.particles = self.particles + noise

    def update(self, z, k=0):
        """Update weights with measurement z."""
        log_liks = self.log_likelihood_fn(self.particles, z, k)
        self.log_weights = self.log_weights + log_liks
        # Normalize
        max_lw = np.max(self.log_weights)
        self.log_weights -= max_lw
        self.weights = np.exp(self.log_weights)
        self.weights /= self.weights.sum()
        self.log_weights = np.log(np.maximum(self.weights, 1e-300))

    def resample(self, force=False):
        """Resample if ESS below threshold."""
        ess = ParticleFilterUtils.effective_sample_size(self.weights)
        if not force and ess >= self.resample_threshold * self.N:
            return False  # no resampling needed

        methods = {
            'systematic': ParticleFilterUtils.systematic_resample,
            'multinomial': ParticleFilterUtils.multinomial_resample,
            'stratified': ParticleFilterUtils.stratified_resample,
            'residual': ParticleFilterUtils.residual_resample,
        }
        resample_fn = methods.get(self.resample_method,
                                   ParticleFilterUtils.systematic_resample)
        indices = resample_fn(self.weights, self.rng)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.N) / self.N
        self.log_weights = np.full(self.N, -np.log(self.N))
        return True

    def estimate(self):
        """Weighted mean and covariance."""
        mean = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - mean
        cov = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.N):
            cov += self.weights[i] * np.outer(diff[i], diff[i])
        return mean, cov

    def step(self, z, k=0):
        """One full predict-update-resample cycle."""
        self.predict(k)
        self.update(z, k)
        self.resample()
        mean, cov = self.estimate()
        # Store history
        self.particle_history.append(self.particles.copy())
        self.weight_history.append(self.weights.copy())
        self.estimate_history.append(mean.copy())
        return mean, cov

    def filter(self, measurements):
        """Run filter over sequence. Returns (estimates, covariances)."""
        if self.particles is None:
            self.initialize()
        estimates = []
        covariances = []
        for k, z in enumerate(measurements):
            mean, cov = self.step(z, k)
            estimates.append(mean)
            covariances.append(cov)
        return np.array(estimates), np.array(covariances)

    def map_estimate(self):
        """Maximum a posteriori (highest weight) particle."""
        idx = np.argmax(self.weights)
        return self.particles[idx].copy()


# ---------------------------------------------------------------------------
# Auxiliary Particle Filter
# ---------------------------------------------------------------------------

class AuxiliaryParticleFilter:
    """Auxiliary particle filter (APF).

    Uses first-stage weights based on predicted measurement likelihood
    to improve sampling in the tails.
    """

    def __init__(self, num_particles, dim_x, transition_fn, log_likelihood_fn,
                 process_noise_fn=None, initial_fn=None,
                 predictive_fn=None, seed=None):
        """
        predictive_fn: predict a "characteristic" state for each particle
                       (e.g., the mean of the transition). (N, dim_x) -> (N, dim_x)
                       If None, uses transition_fn without noise.
        """
        self.N = num_particles
        self.dim_x = dim_x
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.process_noise_fn = process_noise_fn
        self.initial_fn = initial_fn
        self.predictive_fn = predictive_fn
        self.rng = np.random.RandomState(seed)

        self.particles = None
        self.weights = np.ones(num_particles) / num_particles

        self.particle_history = []
        self.weight_history = []
        self.estimate_history = []

    def initialize(self, particles=None):
        if particles is not None:
            self.particles = np.array(particles, dtype=float)
        elif self.initial_fn is not None:
            self.particles = self.initial_fn(self.N, self.rng)
        else:
            self.particles = self.rng.randn(self.N, self.dim_x)
        self.weights = np.ones(self.N) / self.N
        self.particle_history = []
        self.weight_history = []
        self.estimate_history = []

    def step(self, z, k=0):
        """One APF step."""
        # First stage: compute predictive likelihood
        if self.predictive_fn is not None:
            predicted = self.predictive_fn(self.particles, k)
        else:
            predicted = self.transition_fn(self.particles, k)

        first_stage_ll = self.log_likelihood_fn(predicted, z, k)
        first_stage_w = np.log(np.maximum(self.weights, 1e-300)) + first_stage_ll
        max_w = np.max(first_stage_w)
        first_stage_w = np.exp(first_stage_w - max_w)
        first_stage_w /= first_stage_w.sum()

        # Resample according to first-stage weights
        indices = ParticleFilterUtils.systematic_resample(first_stage_w, self.rng)

        # Propagate resampled particles
        resampled = self.particles[indices]
        propagated = self.transition_fn(resampled, k)
        if self.process_noise_fn is not None:
            propagated = propagated + self.process_noise_fn(self.N, self.rng)

        # Second stage: compute weights = p(z|x_k) / p(z|mu_k)
        log_lik_prop = self.log_likelihood_fn(propagated, z, k)
        log_lik_pred = first_stage_ll[indices]
        log_w = log_lik_prop - log_lik_pred
        max_lw = np.max(log_w)
        self.weights = np.exp(log_w - max_lw)
        self.weights /= self.weights.sum()
        self.particles = propagated

        mean = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - mean
        cov = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.N):
            cov += self.weights[i] * np.outer(diff[i], diff[i])

        self.particle_history.append(self.particles.copy())
        self.weight_history.append(self.weights.copy())
        self.estimate_history.append(mean.copy())
        return mean, cov

    def filter(self, measurements):
        if self.particles is None:
            self.initialize()
        estimates, covariances = [], []
        for k, z in enumerate(measurements):
            mean, cov = self.step(z, k)
            estimates.append(mean)
            covariances.append(cov)
        return np.array(estimates), np.array(covariances)


# ---------------------------------------------------------------------------
# Regularized Particle Filter
# ---------------------------------------------------------------------------

class RegularizedParticleFilter:
    """Regularized particle filter (RPF) with kernel smoothing.

    After resampling, adds continuous jitter from a kernel to prevent
    sample impoverishment.
    """

    def __init__(self, num_particles, dim_x, transition_fn, log_likelihood_fn,
                 process_noise_fn=None, initial_fn=None,
                 kernel_bandwidth=None, seed=None):
        self.N = num_particles
        self.dim_x = dim_x
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.process_noise_fn = process_noise_fn
        self.initial_fn = initial_fn
        self.rng = np.random.RandomState(seed)

        self.particles = None
        self.weights = np.ones(num_particles) / num_particles

        # Optimal bandwidth for Gaussian kernel (Silverman's rule)
        self.bandwidth = kernel_bandwidth

        self.particle_history = []
        self.weight_history = []
        self.estimate_history = []

    def initialize(self, particles=None):
        if particles is not None:
            self.particles = np.array(particles, dtype=float)
        elif self.initial_fn is not None:
            self.particles = self.initial_fn(self.N, self.rng)
        else:
            self.particles = self.rng.randn(self.N, self.dim_x)
        self.weights = np.ones(self.N) / self.N
        self.particle_history = []
        self.weight_history = []
        self.estimate_history = []

    def _compute_bandwidth(self):
        """Silverman's rule of thumb bandwidth."""
        if self.bandwidth is not None:
            return self.bandwidth
        # h_opt = (4 / (N * (d+2)))^(1/(d+4)) * std
        d = self.dim_x
        N = self.N
        factor = (4.0 / (N * (d + 2))) ** (1.0 / (d + 4))
        return factor

    def step(self, z, k=0):
        # Predict
        self.particles = self.transition_fn(self.particles, k)
        if self.process_noise_fn is not None:
            self.particles += self.process_noise_fn(self.N, self.rng)

        # Update weights
        log_liks = self.log_likelihood_fn(self.particles, z, k)
        log_w = np.log(np.maximum(self.weights, 1e-300)) + log_liks
        max_lw = np.max(log_w)
        self.weights = np.exp(log_w - max_lw)
        self.weights /= self.weights.sum()

        # Resample
        ess = ParticleFilterUtils.effective_sample_size(self.weights)
        if ess < 0.5 * self.N:
            indices = ParticleFilterUtils.systematic_resample(self.weights, self.rng)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N

            # Regularization: add kernel jitter
            h = self._compute_bandwidth()
            # Compute empirical covariance
            mean_p = np.mean(self.particles, axis=0)
            diff = self.particles - mean_p
            cov_p = (diff.T @ diff) / self.N
            # Add small regularization for stability
            cov_p += 1e-8 * np.eye(self.dim_x)
            try:
                L = cholesky(cov_p)
                noise = self.rng.randn(self.N, self.dim_x)
                self.particles += h * (noise @ L.T)
            except np.linalg.LinAlgError:
                # Fallback: isotropic jitter
                std = np.std(self.particles, axis=0)
                self.particles += h * std * self.rng.randn(self.N, self.dim_x)

        mean = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - mean
        cov = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.N):
            cov += self.weights[i] * np.outer(diff[i], diff[i])

        self.particle_history.append(self.particles.copy())
        self.weight_history.append(self.weights.copy())
        self.estimate_history.append(mean.copy())
        return mean, cov

    def filter(self, measurements):
        if self.particles is None:
            self.initialize()
        estimates, covariances = [], []
        for k, z in enumerate(measurements):
            mean, cov = self.step(z, k)
            estimates.append(mean)
            covariances.append(cov)
        return np.array(estimates), np.array(covariances)


# ---------------------------------------------------------------------------
# Rao-Blackwellized Particle Filter
# ---------------------------------------------------------------------------

class RaoBlackwellizedPF:
    """Rao-Blackwellized particle filter (RBPF).

    For models with mixed linear/nonlinear structure:
        x_k = [x_nl_k, x_l_k]
    where x_l evolves linearly conditioned on x_nl.

    Each particle carries a nonlinear state + a Kalman filter for the linear part.
    Composes C158 KalmanFilter.
    """

    def __init__(self, num_particles, dim_nl, dim_l, dim_z,
                 nl_transition_fn, nl_noise_fn,
                 F_l, H_l, Q_l, R,
                 initial_nl_fn=None, coupling_fn=None, seed=None):
        """
        dim_nl: dimension of nonlinear state
        dim_l: dimension of linear state
        dim_z: measurement dimension
        nl_transition_fn: f(x_nl, k) -> x_nl_new (per particle)
        nl_noise_fn: noise(N, rng) -> (N, dim_nl)
        F_l: linear state transition (dim_l, dim_l)
        H_l: linear observation model (dim_z, dim_l)
        Q_l: linear process noise (dim_l, dim_l)
        R: measurement noise (dim_z, dim_z)
        coupling_fn: optional g(x_nl) -> H_coupling contribution to measurement
        """
        self.N = num_particles
        self.dim_nl = dim_nl
        self.dim_l = dim_l
        self.dim_z = dim_z
        self.nl_transition_fn = nl_transition_fn
        self.nl_noise_fn = nl_noise_fn
        self.F_l = np.array(F_l, dtype=float)
        self.H_l = np.array(H_l, dtype=float)
        self.Q_l = np.array(Q_l, dtype=float)
        self.R = np.array(R, dtype=float)
        self.initial_nl_fn = initial_nl_fn
        self.coupling_fn = coupling_fn
        self.rng = np.random.RandomState(seed)

        # Each particle: (nl_state, KalmanFilter)
        self.nl_particles = None
        self.kf_particles = None  # list of KalmanFilter instances
        self.weights = np.ones(num_particles) / num_particles

        self.estimate_history = []
        self.weight_history = []

    def initialize(self, nl_particles=None, linear_mean=None, linear_cov=None):
        if nl_particles is not None:
            self.nl_particles = np.array(nl_particles, dtype=float)
        elif self.initial_nl_fn is not None:
            self.nl_particles = self.initial_nl_fn(self.N, self.rng)
        else:
            self.nl_particles = self.rng.randn(self.N, self.dim_nl)

        lm = linear_mean if linear_mean is not None else np.zeros(self.dim_l)
        lc = linear_cov if linear_cov is not None else np.eye(self.dim_l)

        self.kf_particles = []
        for _ in range(self.N):
            kf = KalmanFilter(self.dim_l, self.dim_z)
            kf.x = np.array(lm, dtype=float)
            kf.P = np.array(lc, dtype=float)
            kf.F = self.F_l.copy()
            kf.H = self.H_l.copy()
            kf.Q = self.Q_l.copy()
            kf.R = self.R.copy()
            self.kf_particles.append(kf)

        self.weights = np.ones(self.N) / self.N
        self.estimate_history = []
        self.weight_history = []

    def step(self, z, k=0):
        z = np.asarray(z, dtype=float)

        # Propagate nonlinear particles
        new_nl = np.zeros_like(self.nl_particles)
        for i in range(self.N):
            new_nl[i] = self.nl_transition_fn(self.nl_particles[i], k)
        noise = self.nl_noise_fn(self.N, self.rng)
        self.nl_particles = new_nl + noise

        # Update each Kalman filter and compute weights
        log_weights = np.zeros(self.N)
        for i in range(self.N):
            kf = self.kf_particles[i]
            kf.predict()

            # Measurement residual considering coupling
            if self.coupling_fn is not None:
                z_eff = z - self.coupling_fn(self.nl_particles[i])
            else:
                z_eff = z

            # Compute innovation likelihood for weight
            innovation = z_eff - kf.H @ kf.x
            S = kf.H @ kf.P @ kf.H.T + kf.R
            n = len(innovation)
            log_lik = -0.5 * (n * np.log(2 * np.pi) + np.log(max(det(S), 1e-300))
                              + innovation @ inv(S) @ innovation)
            log_weights[i] = log_lik

            kf.update(z_eff)

        # Normalize weights
        max_lw = np.max(log_weights)
        self.weights = np.exp(log_weights - max_lw)
        self.weights /= self.weights.sum()

        # Resample if needed
        ess = ParticleFilterUtils.effective_sample_size(self.weights)
        if ess < 0.5 * self.N:
            indices = ParticleFilterUtils.systematic_resample(self.weights, self.rng)
            self.nl_particles = self.nl_particles[indices]
            new_kfs = []
            for idx in indices:
                kf_old = self.kf_particles[idx]
                kf_new = KalmanFilter(self.dim_l, self.dim_z)
                kf_new.x = kf_old.x.copy()
                kf_new.P = kf_old.P.copy()
                kf_new.F = kf_old.F.copy()
                kf_new.H = kf_old.H.copy()
                kf_new.Q = kf_old.Q.copy()
                kf_new.R = kf_old.R.copy()
                new_kfs.append(kf_new)
            self.kf_particles = new_kfs
            self.weights = np.ones(self.N) / self.N

        # Estimate: combine nonlinear + linear parts
        nl_mean = np.average(self.nl_particles, weights=self.weights, axis=0)
        l_mean = np.average([kf.x for kf in self.kf_particles],
                            weights=self.weights, axis=0)
        full_mean = np.concatenate([nl_mean, l_mean])

        self.estimate_history.append(full_mean.copy())
        self.weight_history.append(self.weights.copy())
        return full_mean

    def filter(self, measurements):
        if self.nl_particles is None:
            self.initialize()
        estimates = []
        for k, z in enumerate(measurements):
            est = self.step(z, k)
            estimates.append(est)
        return np.array(estimates)


# ---------------------------------------------------------------------------
# Particle Smoother
# ---------------------------------------------------------------------------

class ParticleSmoother:
    """Particle smoothing algorithms.

    - Fixed-lag smoother (delays estimate by L steps)
    - Forward-filtering backward-simulation (FFBSi)
    """

    def __init__(self, particle_filter, lag=None):
        """
        particle_filter: a ParticleFilter instance (already run or will be run)
        lag: fixed-lag window size. If None, full backward simulation.
        """
        self.pf = particle_filter
        self.lag = lag

    def fixed_lag_smooth(self, measurements, lag=None):
        """Fixed-lag smoothing.

        For each time t, the estimate at t-L is refined using info up to t.
        Returns smoothed estimates for all times where lag is available.
        """
        L = lag if lag is not None else (self.lag or 5)
        if self.pf.particles is None:
            self.pf.initialize()

        all_particles = []
        all_weights = []
        smoothed = []

        for k, z in enumerate(measurements):
            self.pf.step(z, k)
            all_particles.append(self.pf.particles.copy())
            all_weights.append(self.pf.weights.copy())

            if k >= L:
                # Smoothed estimate at time k-L using current weights
                idx = k - L
                sm_mean = np.average(all_particles[idx], weights=all_weights[idx], axis=0)
                smoothed.append(sm_mean)

        # Add remaining (un-lagged) estimates
        for idx in range(max(0, len(measurements) - L), len(measurements)):
            sm_mean = np.average(all_particles[idx], weights=all_weights[idx], axis=0)
            if len(smoothed) < len(measurements):
                smoothed.append(sm_mean)

        return np.array(smoothed)

    def ffbsi(self, num_trajectories=None):
        """Forward-filtering backward-simulation.

        Requires that the particle filter has already been run (has history).
        Returns smoothed trajectories of shape (num_trajectories, T, dim_x).
        """
        if not self.pf.particle_history:
            raise ValueError("Must run particle filter first")

        T = len(self.pf.particle_history)
        N = self.pf.N
        M = num_trajectories or N
        rng = self.pf.rng

        trajectories = np.zeros((M, T, self.pf.dim_x))

        # Sample final time from final weights
        final_weights = self.pf.weight_history[-1]
        final_indices = rng.choice(N, size=M, replace=True, p=final_weights)
        trajectories[:, T - 1, :] = self.pf.particle_history[T - 1][final_indices]

        # Backward simulation
        for t in range(T - 2, -1, -1):
            particles_t = self.pf.particle_history[t]
            weights_t = self.pf.weight_history[t]

            for m in range(M):
                x_next = trajectories[m, t + 1, :]
                # Compute backward weights: w_t * p(x_{t+1} | x_t)
                # For simplicity, use distance-based kernel
                diffs = x_next - particles_t
                dists_sq = np.sum(diffs ** 2, axis=1)
                # Use Gaussian kernel as transition approximation
                bw = np.mean(np.std(particles_t, axis=0)) + 1e-8
                log_trans = -0.5 * dists_sq / (bw ** 2)
                log_bw = np.log(np.maximum(weights_t, 1e-300)) + log_trans
                max_lbw = np.max(log_bw)
                bw_weights = np.exp(log_bw - max_lbw)
                bw_weights /= bw_weights.sum()

                idx = rng.choice(N, p=bw_weights)
                trajectories[m, t, :] = particles_t[idx]

        return trajectories


# ---------------------------------------------------------------------------
# Adaptive Particle Filter
# ---------------------------------------------------------------------------

class AdaptiveParticleFilter:
    """Particle filter with adaptive number of particles.

    Adjusts particle count based on KLD (Kullback-Leibler divergence)
    bound to maintain approximation quality.
    """

    def __init__(self, dim_x, transition_fn, log_likelihood_fn,
                 process_noise_fn=None, initial_fn=None,
                 min_particles=50, max_particles=5000,
                 kld_epsilon=0.05, kld_delta=0.01,
                 bin_size=None, seed=None):
        self.dim_x = dim_x
        self.transition_fn = transition_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.process_noise_fn = process_noise_fn
        self.initial_fn = initial_fn
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.kld_epsilon = kld_epsilon
        self.kld_delta = kld_delta
        self.bin_size = bin_size
        self.rng = np.random.RandomState(seed)

        self.particles = None
        self.weights = None
        self.N = min_particles

        self.particle_count_history = []
        self.estimate_history = []

    def _kld_bound(self, k_bins):
        """Wilson-Hilferty approximation for KLD sample size."""
        if k_bins <= 1:
            return self.min_particles
        z = 2.33  # ~99% quantile for delta=0.01
        # KLD bound: n = (k-1)/(2*eps) * (1 - 2/(9*(k-1)) + sqrt(2/(9*(k-1)))*z)^3
        term = 1.0 - 2.0 / (9.0 * (k_bins - 1)) + np.sqrt(2.0 / (9.0 * (k_bins - 1))) * z
        n = (k_bins - 1) / (2.0 * self.kld_epsilon) * (term ** 3)
        return int(np.clip(n, self.min_particles, self.max_particles))

    def initialize(self, particles=None):
        self.N = self.min_particles
        if particles is not None:
            self.particles = np.array(particles, dtype=float)
            self.N = len(self.particles)
        elif self.initial_fn is not None:
            self.particles = self.initial_fn(self.N, self.rng)
        else:
            self.particles = self.rng.randn(self.N, self.dim_x)
        self.weights = np.ones(self.N) / self.N
        self.particle_count_history = []
        self.estimate_history = []

    def step(self, z, k=0):
        # Predict
        self.particles = self.transition_fn(self.particles, k)
        if self.process_noise_fn is not None:
            self.particles += self.process_noise_fn(self.N, self.rng)

        # Update weights
        log_liks = self.log_likelihood_fn(self.particles, z, k)
        log_w = np.log(np.maximum(self.weights, 1e-300)) + log_liks
        max_lw = np.max(log_w)
        self.weights = np.exp(log_w - max_lw)
        self.weights /= self.weights.sum()

        # Estimate
        mean = np.average(self.particles, weights=self.weights, axis=0)

        # Resample and adapt particle count
        ess = ParticleFilterUtils.effective_sample_size(self.weights)
        if ess < 0.5 * self.N:
            # Count occupied bins for KLD
            bs = self.bin_size if self.bin_size is not None else np.std(self.particles, axis=0).mean() * 0.5
            if bs < 1e-10:
                bs = 1.0
            bin_indices = np.floor(self.particles / bs).astype(int)
            unique_bins = len(set(map(tuple, bin_indices)))

            new_N = self._kld_bound(unique_bins)
            new_N = max(self.min_particles, min(self.max_particles, new_N))

            # Resample to new_N particles
            indices = self.rng.choice(self.N, size=new_N, replace=True, p=self.weights)
            self.particles = self.particles[indices]
            self.N = new_N
            self.weights = np.ones(self.N) / self.N

        self.particle_count_history.append(self.N)
        self.estimate_history.append(mean.copy())

        diff = self.particles - mean
        cov = np.zeros((self.dim_x, self.dim_x))
        for i in range(self.N):
            cov += self.weights[i] * np.outer(diff[i], diff[i])
        return mean, cov

    def filter(self, measurements):
        if self.particles is None:
            self.initialize()
        estimates, covariances = [], []
        for k, z in enumerate(measurements):
            mean, cov = self.step(z, k)
            estimates.append(mean)
            covariances.append(cov)
        return np.array(estimates), np.array(covariances)


# ---------------------------------------------------------------------------
# Multiple Model Particle Filter
# ---------------------------------------------------------------------------

class MultipleModelPF:
    """Particle filter for jump/switching systems.

    Each particle carries a discrete mode and a continuous state.
    Mode transitions follow a Markov chain (transition matrix).
    """

    def __init__(self, num_particles, dim_x, num_modes,
                 transition_fns, log_likelihood_fns,
                 process_noise_fns, mode_transition_matrix,
                 initial_fn=None, initial_mode_probs=None, seed=None):
        """
        transition_fns: dict {mode: fn(particles, k) -> particles}
        log_likelihood_fns: dict {mode: fn(particles, z, k) -> log_liks}
        process_noise_fns: dict {mode: fn(N, rng) -> noise}
        mode_transition_matrix: (num_modes, num_modes) row-stochastic
        """
        self.N = num_particles
        self.dim_x = dim_x
        self.num_modes = num_modes
        self.transition_fns = transition_fns
        self.log_likelihood_fns = log_likelihood_fns
        self.process_noise_fns = process_noise_fns
        self.mode_trans = np.array(mode_transition_matrix, dtype=float)
        self.initial_fn = initial_fn
        self.initial_mode_probs = initial_mode_probs
        self.rng = np.random.RandomState(seed)

        self.particles = None
        self.modes = None
        self.weights = np.ones(num_particles) / num_particles

        self.estimate_history = []
        self.mode_history = []
        self.weight_history = []

    def initialize(self, particles=None, modes=None):
        if particles is not None:
            self.particles = np.array(particles, dtype=float)
        elif self.initial_fn is not None:
            self.particles = self.initial_fn(self.N, self.rng)
        else:
            self.particles = self.rng.randn(self.N, self.dim_x)

        if modes is not None:
            self.modes = np.array(modes, dtype=int)
        elif self.initial_mode_probs is not None:
            probs = np.array(self.initial_mode_probs, dtype=float)
            probs /= probs.sum()
            self.modes = self.rng.choice(self.num_modes, size=self.N, p=probs)
        else:
            self.modes = self.rng.choice(self.num_modes, size=self.N)

        self.weights = np.ones(self.N) / self.N
        self.estimate_history = []
        self.mode_history = []
        self.weight_history = []

    def step(self, z, k=0):
        # Transition modes
        new_modes = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            probs = self.mode_trans[self.modes[i]]
            new_modes[i] = self.rng.choice(self.num_modes, p=probs)
        self.modes = new_modes

        # Propagate particles per mode
        new_particles = np.zeros_like(self.particles)
        for mode in range(self.num_modes):
            mask = self.modes == mode
            if not np.any(mask):
                continue
            count = np.sum(mask)
            mode_particles = self.particles[mask]
            propagated = self.transition_fns[mode](mode_particles, k)
            if mode in self.process_noise_fns:
                propagated = propagated + self.process_noise_fns[mode](count, self.rng)
            new_particles[mask] = propagated
        self.particles = new_particles

        # Update weights
        log_w = np.log(np.maximum(self.weights, 1e-300))
        for mode in range(self.num_modes):
            mask = self.modes == mode
            if not np.any(mask):
                continue
            ll = self.log_likelihood_fns[mode](self.particles[mask], z, k)
            log_w[mask] += ll

        max_lw = np.max(log_w)
        self.weights = np.exp(log_w - max_lw)
        self.weights /= self.weights.sum()

        # Resample
        ess = ParticleFilterUtils.effective_sample_size(self.weights)
        if ess < 0.5 * self.N:
            indices = ParticleFilterUtils.systematic_resample(self.weights, self.rng)
            self.particles = self.particles[indices]
            self.modes = self.modes[indices]
            self.weights = np.ones(self.N) / self.N

        mean = np.average(self.particles, weights=self.weights, axis=0)

        # Mode probabilities
        mode_probs = np.zeros(self.num_modes)
        for mode in range(self.num_modes):
            mode_probs[mode] = np.sum(self.weights[self.modes == mode])

        self.estimate_history.append(mean.copy())
        self.mode_history.append(mode_probs.copy())
        self.weight_history.append(self.weights.copy())
        return mean, mode_probs

    def filter(self, measurements):
        if self.particles is None:
            self.initialize()
        estimates, mode_probs_list = [], []
        for k, z in enumerate(measurements):
            mean, mp = self.step(z, k)
            estimates.append(mean)
            mode_probs_list.append(mp)
        return np.array(estimates), np.array(mode_probs_list)

    def mode_estimate(self):
        """Most likely current mode."""
        if not self.mode_history:
            return None
        return np.argmax(self.mode_history[-1])
