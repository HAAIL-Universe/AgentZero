"""
C153: Monte Carlo Methods
MCMC sampling, integration, and diagnostics -- standalone probabilistic computing.

Components:
- RandomSampler: Base sampling utilities
- MonteCarloIntegrator: MC integration (importance, stratified, control variates)
- MetropolisHastings: MH sampler (random walk, independent, adaptive)
- GibbsSampler: Full conditional sampling
- HamiltonianMC: HMC with leapfrog integration
- ParallelTempering: Replica exchange MCMC
- NUTS: No-U-Turn Sampler (adaptive HMC)
- MCMCDiagnostics: Convergence diagnostics (R-hat, ESS, autocorrelation)
"""

import math
import random
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any


# ============================================================
# RandomSampler -- base sampling utilities
# ============================================================

class RandomSampler:
    """Base sampling from various distributions."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return self.np_rng.uniform(low, high)
        return self.np_rng.uniform(low, high, size=size)

    def normal(self, mean=0.0, std=1.0, size=None):
        if size is None:
            return self.np_rng.normal(mean, std)
        return self.np_rng.normal(mean, std, size=size)

    def exponential(self, scale=1.0, size=None):
        if size is None:
            return self.np_rng.exponential(scale)
        return self.np_rng.exponential(scale, size=size)

    def multivariate_normal(self, mean, cov, size=None):
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if size is None:
            return self.np_rng.multivariate_normal(mean, cov)
        return self.np_rng.multivariate_normal(mean, cov, size=size)

    def categorical(self, probs):
        """Sample from categorical distribution."""
        probs = np.asarray(probs, dtype=float)
        probs = probs / probs.sum()
        return self.np_rng.choice(len(probs), p=probs)

    def beta(self, a, b, size=None):
        if size is None:
            return self.np_rng.beta(a, b)
        return self.np_rng.beta(a, b, size=size)

    def gamma(self, shape, scale=1.0, size=None):
        if size is None:
            return self.np_rng.gamma(shape, scale)
        return self.np_rng.gamma(shape, scale, size=size)

    def dirichlet(self, alpha, size=None):
        alpha = np.asarray(alpha, dtype=float)
        if size is None:
            return self.np_rng.dirichlet(alpha)
        return self.np_rng.dirichlet(alpha, size=size)


# ============================================================
# MonteCarloIntegrator -- MC integration methods
# ============================================================

class MonteCarloIntegrator:
    """Monte Carlo integration with variance reduction techniques."""

    def __init__(self, seed=None):
        self.sampler = RandomSampler(seed)

    def basic_integrate(self, f, bounds, n_samples=10000):
        """Basic MC integration: integral ~ volume * mean(f(x))."""
        dim = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        volume = np.prod(highs - lows)

        samples = self.sampler.np_rng.uniform(lows, highs, size=(n_samples, dim))
        values = np.array([f(x) for x in samples])
        estimate = volume * np.mean(values)
        std_err = volume * np.std(values) / math.sqrt(n_samples)
        return estimate, std_err

    def importance_sampling(self, f, proposal_sampler, proposal_density,
                            target_density=None, n_samples=10000):
        """Importance sampling integration.

        Args:
            f: function to integrate
            proposal_sampler: callable returning samples from proposal
            proposal_density: density of proposal distribution
            target_density: density of target (uniform if None)
            n_samples: number of samples
        """
        samples = [proposal_sampler() for _ in range(n_samples)]
        weights = []
        values = []
        for x in samples:
            q = proposal_density(x)
            if q < 1e-300:
                continue
            if target_density is not None:
                w = target_density(x) / q
            else:
                w = 1.0 / q
            weights.append(w)
            values.append(f(x) * w)

        if not values:
            return 0.0, float('inf')

        values = np.array(values)
        weights = np.array(weights)
        estimate = np.mean(values)
        std_err = np.std(values) / math.sqrt(len(values))
        return estimate, std_err

    def stratified_sampling(self, f, bounds, n_strata_per_dim=10, samples_per_stratum=10):
        """Stratified sampling -- divide domain into strata, sample each."""
        dim = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])

        # Build strata grid
        strata_edges = []
        for d in range(dim):
            edges = np.linspace(lows[d], highs[d], n_strata_per_dim + 1)
            strata_edges.append(edges)

        # Generate all strata indices
        import itertools
        strata_indices = list(itertools.product(range(n_strata_per_dim), repeat=dim))

        total_strata = len(strata_indices)
        stratum_volume = np.prod(highs - lows) / total_strata

        all_values = []
        for idx in strata_indices:
            s_low = np.array([strata_edges[d][idx[d]] for d in range(dim)])
            s_high = np.array([strata_edges[d][idx[d] + 1] for d in range(dim)])
            samples = self.sampler.np_rng.uniform(s_low, s_high,
                                                   size=(samples_per_stratum, dim))
            for x in samples:
                all_values.append(f(x))

        all_values = np.array(all_values)
        volume = np.prod(highs - lows)
        estimate = volume * np.mean(all_values)
        std_err = volume * np.std(all_values) / math.sqrt(len(all_values))
        return estimate, std_err

    def control_variate(self, f, g, g_expected, bounds, n_samples=10000):
        """Control variate method: use correlated function g with known mean.

        Estimate: mean(f) - c*(mean(g) - E[g])
        where c = cov(f,g)/var(g)
        """
        dim = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        volume = np.prod(highs - lows)

        samples = self.sampler.np_rng.uniform(lows, highs, size=(n_samples, dim))
        f_vals = np.array([f(x) for x in samples])
        g_vals = np.array([g(x) for x in samples])

        cov_fg = np.cov(f_vals, g_vals)[0, 1]
        var_g = np.var(g_vals)
        if var_g < 1e-300:
            c = 0.0
        else:
            c = cov_fg / var_g

        adjusted = f_vals - c * (g_vals - g_expected / volume)
        estimate = volume * np.mean(adjusted)
        std_err = volume * np.std(adjusted) / math.sqrt(n_samples)
        return estimate, std_err

    def antithetic_variates(self, f, bounds, n_samples=10000):
        """Antithetic variates -- use (x, 1-x) pairs for variance reduction."""
        dim = len(bounds)
        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        volume = np.prod(highs - lows)

        half_n = n_samples // 2
        u = self.sampler.np_rng.uniform(0, 1, size=(half_n, dim))
        samples = lows + u * (highs - lows)
        anti_samples = lows + (1 - u) * (highs - lows)

        f_vals = np.array([f(x) for x in samples])
        anti_vals = np.array([f(x) for x in anti_samples])

        paired_means = (f_vals + anti_vals) / 2.0
        estimate = volume * np.mean(paired_means)
        std_err = volume * np.std(paired_means) / math.sqrt(half_n)
        return estimate, std_err


# ============================================================
# MetropolisHastings -- MH MCMC sampler
# ============================================================

class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler."""

    def __init__(self, log_prob, dim, seed=None):
        """
        Args:
            log_prob: log probability function (unnormalized OK)
            dim: dimensionality
            seed: random seed
        """
        self.log_prob = log_prob
        self.dim = dim
        self.sampler = RandomSampler(seed)
        self.proposal_scale = 1.0
        self.adaptive = False
        self.target_accept_rate = 0.234  # Optimal for d>1

    def _propose_random_walk(self, current):
        """Random walk proposal: N(current, scale*I)."""
        return current + self.sampler.normal(0, self.proposal_scale, size=self.dim)

    def sample(self, n_samples, initial=None, burn_in=0, thin=1,
               proposal='random_walk', proposal_fn=None):
        """Run MH sampler.

        Args:
            n_samples: number of samples to collect
            initial: starting point (zeros if None)
            burn_in: discard first burn_in samples
            thin: keep every thin-th sample
            proposal: 'random_walk' or 'custom'
            proposal_fn: custom proposal function(current) -> (proposed, log_q_forward, log_q_reverse)
        """
        if initial is None:
            current = np.zeros(self.dim)
        else:
            current = np.asarray(initial, dtype=float).copy()

        current_lp = self.log_prob(current)
        samples = []
        accepts = 0
        total = 0

        total_needed = burn_in + n_samples * thin

        for i in range(total_needed):
            if proposal == 'random_walk':
                proposed = self._propose_random_walk(current)
                # Symmetric proposal: log_q ratio = 0
                log_alpha = self.log_prob(proposed) - current_lp
            elif proposal == 'custom' and proposal_fn is not None:
                proposed, log_q_fwd, log_q_rev = proposal_fn(current)
                proposed_lp = self.log_prob(proposed)
                log_alpha = proposed_lp - current_lp + log_q_rev - log_q_fwd
            else:
                raise ValueError(f"Unknown proposal: {proposal}")

            if math.log(self.sampler.rng.random() + 1e-300) < log_alpha:
                current = proposed
                current_lp = self.log_prob(proposed) if proposal == 'random_walk' else proposed_lp
                if i >= burn_in:
                    accepts += 1
            if i >= burn_in:
                total += 1

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current.copy())

            # Adaptive scaling during burn-in
            if self.adaptive and i < burn_in and i > 0 and i % 50 == 0:
                recent_rate = accepts / max(total, 1)
                if recent_rate < self.target_accept_rate - 0.05:
                    self.proposal_scale *= 0.9
                elif recent_rate > self.target_accept_rate + 0.05:
                    self.proposal_scale *= 1.1

        accept_rate = accepts / max(total, 1)
        return np.array(samples), accept_rate

    def adaptive_sample(self, n_samples, initial=None, burn_in=500, thin=1):
        """MH with adaptive proposal scaling."""
        self.adaptive = True
        result = self.sample(n_samples, initial=initial, burn_in=burn_in, thin=thin)
        self.adaptive = False
        return result


# ============================================================
# GibbsSampler -- full conditional sampling
# ============================================================

class GibbsSampler:
    """Gibbs sampler -- sample each dimension from its full conditional."""

    def __init__(self, conditionals, dim, seed=None):
        """
        Args:
            conditionals: list of callables, one per dimension.
                conditionals[i](state) -> sampled value for dimension i
            dim: dimensionality
        """
        self.conditionals = conditionals
        self.dim = dim
        self.sampler = RandomSampler(seed)

    def sample(self, n_samples, initial=None, burn_in=0, thin=1):
        """Run Gibbs sampler.

        Args:
            n_samples: number of samples
            initial: starting state
            burn_in: discard first burn_in
            thin: keep every thin-th
        """
        if initial is None:
            state = np.zeros(self.dim)
        else:
            state = np.asarray(initial, dtype=float).copy()

        samples = []
        total_needed = burn_in + n_samples * thin

        for i in range(total_needed):
            for d in range(self.dim):
                state[d] = self.conditionals[d](state)

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(state.copy())

        return np.array(samples)

    def block_sample(self, n_samples, blocks, block_conditionals,
                     initial=None, burn_in=0, thin=1):
        """Block Gibbs -- sample groups of variables together.

        Args:
            blocks: list of index lists, e.g. [[0,1], [2,3]]
            block_conditionals: list of callables, one per block.
                block_conditionals[i](state) -> array of values for block i
        """
        if initial is None:
            state = np.zeros(self.dim)
        else:
            state = np.asarray(initial, dtype=float).copy()

        samples = []
        total_needed = burn_in + n_samples * thin

        for i in range(total_needed):
            for b_idx, block in enumerate(blocks):
                values = block_conditionals[b_idx](state)
                for j, dim_idx in enumerate(block):
                    state[dim_idx] = values[j]

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(state.copy())

        return np.array(samples)


# ============================================================
# HamiltonianMC -- Hamiltonian Monte Carlo
# ============================================================

class HamiltonianMC:
    """Hamiltonian Monte Carlo with leapfrog integration."""

    def __init__(self, log_prob, grad_log_prob, dim, seed=None):
        """
        Args:
            log_prob: log probability (unnormalized OK)
            grad_log_prob: gradient of log probability
            dim: dimensionality
        """
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.dim = dim
        self.sampler = RandomSampler(seed)
        self.step_size = 0.1
        self.n_leapfrog = 10

    def _leapfrog(self, q, p):
        """Leapfrog integration."""
        q = q.copy()
        p = p.copy()

        # Half step for momentum
        p += 0.5 * self.step_size * self.grad_log_prob(q)

        # Full steps for position and momentum
        for _ in range(self.n_leapfrog - 1):
            q += self.step_size * p
            p += self.step_size * self.grad_log_prob(q)

        # Final full step for position
        q += self.step_size * p

        # Half step for momentum
        p += 0.5 * self.step_size * self.grad_log_prob(q)

        return q, -p  # Negate momentum for reversibility

    def _hamiltonian(self, q, p):
        """Compute Hamiltonian H(q,p) = -log_prob(q) + 0.5*p^T*p."""
        return -self.log_prob(q) + 0.5 * np.dot(p, p)

    def sample(self, n_samples, initial=None, burn_in=0, thin=1):
        """Run HMC sampler."""
        if initial is None:
            q = np.zeros(self.dim)
        else:
            q = np.asarray(initial, dtype=float).copy()

        samples = []
        accepts = 0
        total = 0
        total_needed = burn_in + n_samples * thin

        for i in range(total_needed):
            p = self.sampler.normal(size=self.dim)
            current_H = self._hamiltonian(q, p)

            q_new, p_new = self._leapfrog(q, p)
            proposed_H = self._hamiltonian(q_new, p_new)

            log_alpha = current_H - proposed_H

            if math.log(self.sampler.rng.random() + 1e-300) < log_alpha:
                q = q_new
                if i >= burn_in:
                    accepts += 1
            if i >= burn_in:
                total += 1

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(q.copy())

        accept_rate = accepts / max(total, 1)
        return np.array(samples), accept_rate


# ============================================================
# NUTS -- No-U-Turn Sampler
# ============================================================

class NUTS:
    """No-U-Turn Sampler -- adaptive HMC that auto-tunes trajectory length."""

    def __init__(self, log_prob, grad_log_prob, dim, seed=None):
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.dim = dim
        self.sampler = RandomSampler(seed)
        self.step_size = 0.1
        self.max_depth = 10
        self.delta_max = 1000.0

    def _leapfrog(self, q, p):
        """Single leapfrog step."""
        p = p + 0.5 * self.step_size * self.grad_log_prob(q)
        q = q + self.step_size * p
        p = p + 0.5 * self.step_size * self.grad_log_prob(q)
        return q, p

    def _uturn(self, q_minus, q_plus, p_minus, p_plus):
        """Check U-turn criterion."""
        dq = q_plus - q_minus
        return (np.dot(dq, p_minus) < 0) or (np.dot(dq, p_plus) < 0)

    def _build_tree(self, q, p, u, direction, depth, joint0):
        """Recursively build tree for NUTS."""
        if depth == 0:
            q_prime, p_prime = self._leapfrog(q, direction * np.abs(p) / np.abs(p) * p if False else p)
            # Actually: just do leapfrog in the given direction
            q_prime = q + direction * self.step_size * p + 0.5 * direction * self.step_size**2 * self.grad_log_prob(q)
            p_prime = p + 0.5 * self.step_size * (self.grad_log_prob(q) + self.grad_log_prob(q_prime))
            # Simplified: use standard leapfrog with direction
            q_prime = q.copy()
            p_prime = p.copy()
            if direction == 1:
                q_prime, p_prime = self._leapfrog(q_prime, p_prime)
            else:
                # Reverse leapfrog
                p_prime = p_prime - 0.5 * self.step_size * self.grad_log_prob(q_prime)
                q_prime = q_prime - self.step_size * p_prime
                p_prime = p_prime - 0.5 * self.step_size * self.grad_log_prob(q_prime)

            joint = self.log_prob(q_prime) - 0.5 * np.dot(p_prime, p_prime)
            n_prime = 1 if u <= math.exp(min(joint, 700)) else 0
            s_prime = 1 if joint - joint0 > -self.delta_max else 0
            return q_prime, p_prime, q_prime, p_prime, q_prime, n_prime, s_prime

        # Recursion
        (q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime
         ) = self._build_tree(q, p, u, direction, depth - 1, joint0)

        if s_prime == 1:
            if direction == -1:
                (q_minus, p_minus, _, _, q_pp, n_pp, s_pp
                 ) = self._build_tree(q_minus, p_minus, u, direction, depth - 1, joint0)
            else:
                (_, _, q_plus, p_plus, q_pp, n_pp, s_pp
                 ) = self._build_tree(q_plus, p_plus, u, direction, depth - 1, joint0)

            total = n_prime + n_pp
            if total > 0 and self.sampler.rng.random() < n_pp / total:
                q_prime = q_pp
            n_prime = total
            s_prime = s_pp * (1 if not self._uturn(q_minus, q_plus, p_minus, p_plus) else 0)

        return q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime

    def sample(self, n_samples, initial=None, burn_in=0, thin=1):
        """Run NUTS sampler."""
        if initial is None:
            q = np.zeros(self.dim)
        else:
            q = np.asarray(initial, dtype=float).copy()

        samples = []
        total_needed = burn_in + n_samples * thin

        for i in range(total_needed):
            p = self.sampler.normal(size=self.dim)
            joint0 = self.log_prob(q) - 0.5 * np.dot(p, p)
            u = self.sampler.rng.random() * math.exp(min(joint0, 700))

            q_minus = q.copy()
            q_plus = q.copy()
            p_minus = p.copy()
            p_plus = p.copy()
            q_candidate = q.copy()
            depth = 0
            n = 1
            s = 1

            while s == 1 and depth < self.max_depth:
                direction = 1 if self.sampler.rng.random() < 0.5 else -1
                if direction == -1:
                    (q_minus, p_minus, _, _, q_prime, n_prime, s_prime
                     ) = self._build_tree(q_minus, p_minus, u, direction, depth, joint0)
                else:
                    (_, _, q_plus, p_plus, q_prime, n_prime, s_prime
                     ) = self._build_tree(q_plus, p_plus, u, direction, depth, joint0)

                if s_prime == 1 and n + n_prime > 0:
                    if self.sampler.rng.random() < n_prime / (n + n_prime):
                        q_candidate = q_prime
                n += n_prime
                s = s_prime * (1 if not self._uturn(q_minus, q_plus, p_minus, p_plus) else 0)
                depth += 1

            q = q_candidate

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(q.copy())

        return np.array(samples)


# ============================================================
# ParallelTempering -- Replica Exchange MCMC
# ============================================================

class ParallelTempering:
    """Parallel tempering (replica exchange) for multimodal distributions."""

    def __init__(self, log_prob, dim, temperatures=None, n_chains=4, seed=None):
        """
        Args:
            log_prob: log probability at temperature 1
            dim: dimensionality
            temperatures: list of temperatures (ascending, first=1.0)
            n_chains: number of chains if temperatures not specified
        """
        self.log_prob = log_prob
        self.dim = dim
        self.sampler = RandomSampler(seed)

        if temperatures is not None:
            self.temperatures = list(temperatures)
        else:
            self.temperatures = [1.0 * (2.0 ** i) for i in range(n_chains)]

        self.n_chains = len(self.temperatures)
        self.proposal_scale = 1.0

    def _log_prob_tempered(self, x, temp):
        """Tempered log probability: log_prob(x) / temp."""
        return self.log_prob(x) / temp

    def sample(self, n_samples, initial=None, burn_in=0, thin=1, swap_interval=1):
        """Run parallel tempering.

        Args:
            n_samples: samples to collect from cold chain
            initial: starting points (n_chains x dim), or single point
            burn_in: burn-in iterations
            thin: thinning factor
            swap_interval: attempt swaps every N iterations
        """
        # Initialize chains
        if initial is None:
            chains = [np.zeros(self.dim) for _ in range(self.n_chains)]
        elif np.asarray(initial).ndim == 1:
            chains = [np.asarray(initial, dtype=float).copy() for _ in range(self.n_chains)]
        else:
            chains = [np.asarray(initial[i], dtype=float).copy() for i in range(self.n_chains)]

        log_probs = [self._log_prob_tempered(chains[i], self.temperatures[i])
                     for i in range(self.n_chains)]

        samples = []
        swaps_attempted = 0
        swaps_accepted = 0
        total_needed = burn_in + n_samples * thin

        for it in range(total_needed):
            # MH step for each chain
            for c in range(self.n_chains):
                proposed = chains[c] + self.sampler.normal(
                    0, self.proposal_scale * math.sqrt(self.temperatures[c]),
                    size=self.dim)
                proposed_lp = self._log_prob_tempered(proposed, self.temperatures[c])
                log_alpha = proposed_lp - log_probs[c]

                if math.log(self.sampler.rng.random() + 1e-300) < log_alpha:
                    chains[c] = proposed
                    log_probs[c] = proposed_lp

            # Swap attempts
            if it % swap_interval == 0 and self.n_chains > 1:
                i = self.sampler.rng.randint(0, self.n_chains - 2)
                j = i + 1
                swaps_attempted += 1

                # Swap criterion
                log_swap = ((1.0 / self.temperatures[i] - 1.0 / self.temperatures[j]) *
                            (self.log_prob(chains[j]) - self.log_prob(chains[i])))

                if math.log(self.sampler.rng.random() + 1e-300) < log_swap:
                    chains[i], chains[j] = chains[j], chains[i]
                    log_probs[i] = self._log_prob_tempered(chains[i], self.temperatures[i])
                    log_probs[j] = self._log_prob_tempered(chains[j], self.temperatures[j])
                    swaps_accepted += 1

            if it >= burn_in and (it - burn_in) % thin == 0:
                samples.append(chains[0].copy())

        swap_rate = swaps_accepted / max(swaps_attempted, 1)
        return np.array(samples), swap_rate


# ============================================================
# SliceSampler -- Slice sampling
# ============================================================

class SliceSampler:
    """Slice sampler -- auxiliary variable method."""

    def __init__(self, log_prob, dim, seed=None):
        self.log_prob = log_prob
        self.dim = dim
        self.sampler = RandomSampler(seed)
        self.width = 1.0
        self.max_steps_out = 10

    def _find_interval(self, x, d, y):
        """Find interval [L, R] around x in direction d at height y using stepping out."""
        L = -self.width * self.sampler.rng.random()
        R = L + self.width

        # Step out
        for _ in range(self.max_steps_out):
            x_L = x.copy()
            x_L[d] = L
            if self.log_prob(x_L) <= y:
                break
            L -= self.width

        for _ in range(self.max_steps_out):
            x_R = x.copy()
            x_R[d] = R
            if self.log_prob(x_R) <= y:
                break
            R += self.width

        return L, R

    def sample(self, n_samples, initial=None, burn_in=0, thin=1):
        """Run slice sampler."""
        if initial is None:
            x = np.zeros(self.dim)
        else:
            x = np.asarray(initial, dtype=float).copy()

        samples = []
        total_needed = burn_in + n_samples * thin

        for i in range(total_needed):
            for d in range(self.dim):
                lp = self.log_prob(x)
                y = lp - self.sampler.exponential(1.0)

                L, R = self._find_interval(x, d, y)

                # Shrink interval
                for _ in range(100):
                    x_new = x.copy()
                    x_new[d] = L + self.sampler.rng.random() * (R - L)
                    if self.log_prob(x_new) > y:
                        x = x_new
                        break
                    if x_new[d] < x[d]:
                        L = x_new[d]
                    else:
                        R = x_new[d]

            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(x.copy())

        return np.array(samples)


# ============================================================
# MCMCDiagnostics -- convergence diagnostics
# ============================================================

class MCMCDiagnostics:
    """MCMC convergence diagnostics."""

    @staticmethod
    def effective_sample_size(samples):
        """Compute effective sample size using autocorrelation."""
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        n, d = samples.shape
        ess = np.zeros(d)

        for dim in range(d):
            x = samples[:, dim]
            x = x - np.mean(x)
            var = np.var(x)
            if var < 1e-300:
                ess[dim] = n
                continue

            # Compute autocorrelation
            max_lag = min(n // 2, 1000)
            autocorr = np.correlate(x, x, mode='full')
            autocorr = autocorr[n - 1:] / (var * n)

            # Sum pairs until negative
            tau = 1.0
            for lag in range(1, max_lag):
                if autocorr[lag] < 0:
                    break
                tau += 2 * autocorr[lag]

            ess[dim] = n / tau

        return ess if d > 1 else ess[0]

    @staticmethod
    def r_hat(chains):
        """Gelman-Rubin R-hat diagnostic.

        Args:
            chains: list of arrays, each (n_samples, dim) or (n_samples,)
        """
        chains = [np.atleast_2d(c) if c.ndim == 1 else c for c in chains]
        # If 1D chains passed as column vectors, transpose
        chains = [c.reshape(-1, 1) if c.shape[0] == 1 and c.shape[1] > 1 else c for c in chains]
        # Actually: ensure each chain is (n, d)
        processed = []
        for c in chains:
            c = np.asarray(c)
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            processed.append(c)
        chains = processed

        m = len(chains)  # Number of chains
        n = chains[0].shape[0]  # Samples per chain
        d = chains[0].shape[1]  # Dimensions

        r_hats = np.zeros(d)

        for dim in range(d):
            chain_means = np.array([c[:, dim].mean() for c in chains])
            chain_vars = np.array([c[:, dim].var(ddof=1) for c in chains])

            grand_mean = chain_means.mean()
            B = n * np.var(chain_means, ddof=1)  # Between-chain variance
            W = np.mean(chain_vars)  # Within-chain variance

            if W < 1e-300:
                r_hats[dim] = 1.0
                continue

            var_hat = (1 - 1.0 / n) * W + (1.0 / n) * B
            r_hats[dim] = math.sqrt(var_hat / W)

        return r_hats if d > 1 else r_hats[0]

    @staticmethod
    def autocorrelation(samples, max_lag=None):
        """Compute autocorrelation function."""
        if samples.ndim > 1:
            samples = samples[:, 0]

        n = len(samples)
        if max_lag is None:
            max_lag = min(n // 2, 100)

        x = samples - np.mean(samples)
        var = np.var(samples)
        if var < 1e-300:
            return np.ones(max_lag + 1)

        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            acf[lag] = np.mean(x[:n - lag] * x[lag:]) / var if lag < n else 0

        return acf

    @staticmethod
    def split_r_hat(chains):
        """Split R-hat -- split each chain in half, then compute R-hat."""
        split_chains = []
        for c in chains:
            c = np.asarray(c)
            if c.ndim == 1:
                c = c.reshape(-1, 1)
            mid = len(c) // 2
            split_chains.append(c[:mid])
            split_chains.append(c[mid:])
        return MCMCDiagnostics.r_hat(split_chains)

    @staticmethod
    def geweke(samples, first_frac=0.1, last_frac=0.5):
        """Geweke convergence diagnostic -- z-score comparing first and last portions."""
        if samples.ndim > 1:
            samples = samples[:, 0]
        n = len(samples)
        n_a = int(n * first_frac)
        n_b = int(n * last_frac)

        a = samples[:n_a]
        b = samples[n - n_b:]

        mean_a = np.mean(a)
        mean_b = np.mean(b)
        var_a = np.var(a) / len(a)
        var_b = np.var(b) / len(b)

        denom = math.sqrt(var_a + var_b)
        if denom < 1e-300:
            return 0.0

        return (mean_a - mean_b) / denom

    @staticmethod
    def summary(samples):
        """Compute summary statistics for samples."""
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        d = samples.shape[1]
        result = {}
        for dim in range(d):
            s = samples[:, dim]
            result[f'dim_{dim}'] = {
                'mean': float(np.mean(s)),
                'std': float(np.std(s)),
                'median': float(np.median(s)),
                'q025': float(np.percentile(s, 2.5)),
                'q975': float(np.percentile(s, 97.5)),
                'ess': float(MCMCDiagnostics.effective_sample_size(s)),
            }
        return result
