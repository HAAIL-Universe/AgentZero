"""V221: Contextual Causal Bandits.

Composes V217 (Causal Bandits) + V214 (Causal Discovery) for subgroup-specific
interventions. The core idea: context variables (e.g., age, severity) determine
which intervention (arm) is optimal. Different subgroups may have different
optimal treatments.

Key components:
  - ContextualCausalEnv: causal bandit env with observable context variables
  - ContextualCausalPolicy: abstract policy mapping context -> arm
  - Six algorithms: binned UCB, binned Thompson, causal LinUCB, CATE-greedy,
    subgroup discovery, policy tree
  - CATE estimation: conditional average treatment effects per subgroup
  - Heterogeneous treatment effect analysis
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

import sys
import os

# Add parent paths for dependencies
_base = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_base, "V217_causal_bandits"))
sys.path.insert(0, os.path.join(_base, "V214_causal_discovery"))
sys.path.insert(0, os.path.join(_base, "V211_causal_inference"))
sys.path.insert(0, os.path.join(_base, "V209_bayesian_networks"))

from causal_bandits import (
    Arm,
    BanditResult,
    CausalBanditEnv,
    Intervention,
)
from causal_inference import CausalModel, variable_elimination
from bayesian_networks import BayesianNetwork, Factor
from causal_discovery import sample_from_bn


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Observable context before choosing an arm."""
    features: dict[str, object]  # variable_name -> observed_value

    def key(self, context_vars: list[str]) -> tuple:
        """Hashable key for binning."""
        return tuple(self.features.get(v) for v in sorted(context_vars))

    def __repr__(self) -> str:
        return f"Context({self.features})"


@dataclass
class ContextualResult:
    """Result from a contextual causal bandit run."""
    total_pulls: int = 0
    cumulative_regret: list[float] = field(default_factory=list)
    history: list[tuple[Context, str, float]] = field(default_factory=list)
    # Per-context statistics
    context_arm_pulls: dict[tuple, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    context_arm_rewards: dict[tuple, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))

    def regret(self) -> float:
        return self.cumulative_regret[-1] if self.cumulative_regret else 0.0

    def average_regret(self) -> float:
        return self.regret() / self.total_pulls if self.total_pulls else 0.0

    def learned_policy(self, context_vars: list[str]) -> dict[tuple, str]:
        """Extract best arm per context bin from empirical data."""
        policy = {}
        for ctx_key, arm_rews in self.context_arm_rewards.items():
            arm_pulls = self.context_arm_pulls[ctx_key]
            best_arm = None
            best_avg = float("-inf")
            for arm_name, total_rew in arm_rews.items():
                pulls = arm_pulls[arm_name]
                if pulls > 0:
                    avg = total_rew / pulls
                    if avg > best_avg:
                        best_avg = avg
                        best_arm = arm_name
            if best_arm is not None:
                policy[ctx_key] = best_arm
        return policy


@dataclass
class CATEEstimate:
    """Conditional Average Treatment Effect for a subgroup."""
    subgroup: dict[str, object]  # context conditions
    arm_a: str
    arm_b: str  # reference arm
    cate: float  # E[Y|do(a), context] - E[Y|do(b), context]
    n_samples: int
    std_err: float = 0.0


@dataclass
class SubgroupResult:
    """Result of subgroup analysis."""
    subgroups: list[dict]  # {context_key, optimal_arm, expected_reward, n_samples}
    heterogeneity_score: float  # variance of optimal rewards across subgroups
    cate_estimates: list[CATEEstimate]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ContextualCausalEnv:
    """Causal bandit environment with observable context variables.

    Some variables in the causal graph are 'context' (observed before action),
    some are 'action' (can be intervened on). The reward depends on both.
    """

    def __init__(
        self,
        causal_model: CausalModel,
        reward_var: str,
        context_vars: list[str],
        arms: list[Arm] | None = None,
        reward_mapping: dict[object, float] | None = None,
        seed: int | None = None,
    ):
        self.model = causal_model
        self.bn = causal_model.bn
        self.reward_var = reward_var
        self.context_vars = sorted(context_vars)
        self.reward_mapping = reward_mapping
        self.rng = random.Random(seed)

        # Build inner env for arm enumeration and reward computation
        self._inner_env = CausalBanditEnv(
            causal_model, reward_var, arms, reward_mapping, seed,
        )
        self.arms = self._inner_env.arms

        # Precompute context domains from BN
        self.context_domains: dict[str, list] = {}
        for v in self.context_vars:
            if v in self.bn.domains:
                self.context_domains[v] = list(self.bn.domains[v])
            else:
                self.context_domains[v] = [0, 1]  # default binary

        # Precompute true expected rewards per (context, arm)
        self._true_rewards: dict[tuple, dict[str, float]] = {}
        self._optimal_arms: dict[tuple, Arm] = {}
        self._optimal_rewards: dict[tuple, float] = {}
        self._precompute_contextual_rewards()

    def _precompute_contextual_rewards(self) -> None:
        """Compute E[Y | do(arm), context] for all (context, arm) pairs.

        For each arm, apply do-operator (graph surgery), then compute
        the joint P(context_vars, reward_var) and condition on context.
        """
        import itertools
        context_combos = list(itertools.product(
            *[self.context_domains[v] for v in self.context_vars]
        ))

        for arm in self.arms:
            interventions = arm.to_dict()
            mutilated_bn = self.model.do(interventions)

            # Query joint over context_vars + reward_var
            query_vars = self.context_vars + [self.reward_var]
            joint = variable_elimination(mutilated_bn, query_vars)
            joint = joint.normalize()

            for combo in context_combos:
                ctx_key = combo
                ctx_dict = dict(zip(self.context_vars, combo))

                # Compute P(Y=y, context) for each reward value
                reward_dist: dict[object, float] = {}
                p_ctx = 0.0
                for y_val in self.bn.domains[self.reward_var]:
                    assignment = {**ctx_dict, self.reward_var: y_val}
                    p = joint.get(assignment)
                    reward_dist[y_val] = p
                    p_ctx += p

                # P(Y|context) = P(Y, context) / P(context)
                if p_ctx > 0:
                    for y_val in reward_dist:
                        reward_dist[y_val] /= p_ctx

                if ctx_key not in self._true_rewards:
                    self._true_rewards[ctx_key] = {}
                self._true_rewards[ctx_key][arm.name] = self._expected_reward(reward_dist)

        # Compute optimal arms per context
        for combo in context_combos:
            ctx_key = combo
            arm_rewards = self._true_rewards.get(ctx_key, {})
            if arm_rewards:
                best_arm = max(self.arms, key=lambda a: arm_rewards.get(a.name, 0.0))
                self._optimal_arms[ctx_key] = best_arm
                self._optimal_rewards[ctx_key] = arm_rewards[best_arm.name]

    def _expected_reward(self, dist: dict[object, float]) -> float:
        """Expected reward from a distribution over reward variable values."""
        if self.reward_mapping:
            return sum(self.reward_mapping.get(v, 0) * p for v, p in dist.items())
        return sum(float(v) * p for v, p in dist.items())

    def sample_context(self) -> Context:
        """Sample a context from the observational distribution."""
        # Forward sample the causal model, extract context variables
        sample = sample_from_bn(self.bn, 1, seed=self.rng.randint(0, 2**31))[0]
        features = {v: sample[v] for v in self.context_vars}
        return Context(features=features)

    def _get_contextual_dist(self, arm: Arm, context: Context) -> dict[object, float]:
        """Get P(Y | do(arm), context) as a dict."""
        ctx_key = context.key(self.context_vars)
        if ctx_key in self._true_rewards:
            # Reconstruct dist from precomputed: use mutilated BN joint
            interventions = arm.to_dict()
            mutilated_bn = self.model.do(interventions)
            query_vars = self.context_vars + [self.reward_var]
            joint = variable_elimination(mutilated_bn, query_vars)
            joint = joint.normalize()
            ctx_dict = context.features
            reward_dist: dict[object, float] = {}
            p_ctx = 0.0
            for y_val in self.bn.domains[self.reward_var]:
                assignment = {**ctx_dict, self.reward_var: y_val}
                p = joint.get(assignment)
                reward_dist[y_val] = p
                p_ctx += p
            if p_ctx > 0:
                for y_val in reward_dist:
                    reward_dist[y_val] /= p_ctx
            return reward_dist
        return {v: 1.0 / len(self.bn.domains[self.reward_var])
                for v in self.bn.domains[self.reward_var]}

    def pull(self, arm: Arm, context: Context) -> float:
        """Pull an arm in a given context. Returns stochastic reward."""
        dist = self._get_contextual_dist(arm, context)
        # Sample from distribution
        r = self.rng.random()
        cumulative = 0.0
        for val, prob in dist.items():
            cumulative += prob
            if r <= cumulative:
                if self.reward_mapping:
                    return self.reward_mapping.get(val, 0.0)
                return float(val)
        # Fallback
        val = list(dist.keys())[-1]
        if self.reward_mapping:
            return self.reward_mapping.get(val, 0.0)
        return float(val)

    def true_expected_reward(self, arm: Arm, context: Context) -> float:
        """True expected reward E[Y | do(arm), context]."""
        ctx_key = context.key(self.context_vars)
        if ctx_key in self._true_rewards:
            return self._true_rewards[ctx_key].get(arm.name, 0.0)
        # Fallback: compute on the fly
        dist = self._get_contextual_dist(arm, context)
        return self._expected_reward(dist)

    def optimal_arm(self, context: Context) -> Arm:
        """Return the optimal arm for a given context."""
        ctx_key = context.key(self.context_vars)
        if ctx_key in self._optimal_arms:
            return self._optimal_arms[ctx_key]
        # Fallback
        best = max(self.arms, key=lambda a: self.true_expected_reward(a, context))
        return best

    def optimal_reward(self, context: Context) -> float:
        """Return the optimal expected reward for a given context."""
        ctx_key = context.key(self.context_vars)
        return self._optimal_rewards.get(ctx_key, 0.0)

    def is_heterogeneous(self) -> bool:
        """Check if different contexts have different optimal arms."""
        opt_arms = set()
        for ctx_key, arm in self._optimal_arms.items():
            opt_arms.add(arm.name)
        return len(opt_arms) > 1


# ---------------------------------------------------------------------------
# Algorithm 1: Binned UCB-Causal
# ---------------------------------------------------------------------------

def binned_ucb_causal(
    env: ContextualCausalEnv,
    rounds: int = 1000,
    causal_prior_weight: int = 5,
    exploration_constant: float = 2.0,
    seed: int | None = None,
) -> ContextualResult:
    """UCB1 with causal priors, maintaining separate statistics per context bin.

    Each context bin gets its own UCB statistics. Causal priors provide
    warm-start estimates per (context, arm).
    """
    rng = random.Random(seed)
    result = ContextualResult()

    # Per-context-bin statistics: counts and total rewards
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Initialize with causal priors per context
    if causal_prior_weight > 0:
        for ctx_key, arm_rews in env._true_rewards.items():
            for arm in env.arms:
                er = arm_rews.get(arm.name, 0.0)
                counts[ctx_key][arm.name] = causal_prior_weight
                totals[ctx_key][arm.name] = er * causal_prior_weight

    cum_regret = 0.0

    for t in range(1, rounds + 1):
        context = env.sample_context()
        ctx_key = context.key(env.context_vars)

        # UCB selection for this context
        total_ctx_pulls = sum(counts[ctx_key].values())
        if total_ctx_pulls == 0:
            # No data for this context -- explore randomly
            chosen = rng.choice(env.arms)
        else:
            best_ucb = float("-inf")
            chosen = env.arms[0]
            for arm in env.arms:
                n = counts[ctx_key][arm.name]
                if n == 0:
                    chosen = arm
                    break
                avg = totals[ctx_key][arm.name] / n
                ucb = avg + exploration_constant * math.sqrt(
                    math.log(total_ctx_pulls) / n
                )
                if ucb > best_ucb:
                    best_ucb = ucb
                    chosen = arm

        reward = env.pull(chosen, context)
        counts[ctx_key][chosen.name] += 1
        totals[ctx_key][chosen.name] += reward

        # Track regret
        opt_rew = env.optimal_reward(context)
        cum_regret += opt_rew - reward
        result.cumulative_regret.append(cum_regret)
        result.history.append((context, chosen.name, reward))
        result.context_arm_pulls[ctx_key][chosen.name] += 1
        result.context_arm_rewards[ctx_key][chosen.name] += reward
        result.total_pulls += 1

    return result


# ---------------------------------------------------------------------------
# Algorithm 2: Binned Thompson Sampling with Causal Priors
# ---------------------------------------------------------------------------

def binned_thompson_causal(
    env: ContextualCausalEnv,
    rounds: int = 1000,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    causal_prior_strength: int = 5,
    seed: int | None = None,
) -> ContextualResult:
    """Thompson Sampling with Beta posteriors, per context bin, causal priors."""
    rng = random.Random(seed)
    result = ContextualResult()

    # Per-context Beta parameters
    alphas: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: prior_alpha))
    betas: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: prior_beta))

    # Causal prior initialization
    if causal_prior_strength > 0:
        for ctx_key, arm_rews in env._true_rewards.items():
            # Find min/max for normalization
            vals = list(arm_rews.values())
            rmin, rmax = min(vals), max(vals)
            rng_span = rmax - rmin if rmax > rmin else 1.0
            for arm in env.arms:
                er = arm_rews.get(arm.name, 0.0)
                er_norm = (er - rmin) / rng_span if rng_span > 0 else 0.5
                er_norm = max(0.01, min(0.99, er_norm))
                alphas[ctx_key][arm.name] = prior_alpha + er_norm * causal_prior_strength
                betas[ctx_key][arm.name] = prior_beta + (1 - er_norm) * causal_prior_strength

    cum_regret = 0.0

    for t in range(1, rounds + 1):
        context = env.sample_context()
        ctx_key = context.key(env.context_vars)

        # Thompson sampling: draw from Beta posterior, pick highest
        best_sample = float("-inf")
        chosen = env.arms[0]
        for arm in env.arms:
            a = alphas[ctx_key][arm.name]
            b = betas[ctx_key][arm.name]
            sample = rng.betavariate(max(a, 0.01), max(b, 0.01))
            if sample > best_sample:
                best_sample = sample
                chosen = arm

        reward = env.pull(chosen, context)

        # Normalize reward to [0, 1] for Beta update
        # Use true reward range from causal model
        if ctx_key in env._true_rewards:
            vals = list(env._true_rewards[ctx_key].values())
            rmin, rmax = min(vals), max(vals)
            rng_span = rmax - rmin if rmax > rmin else 1.0
            r_norm = (reward - rmin) / rng_span
            r_norm = max(0.0, min(1.0, r_norm))
        else:
            r_norm = max(0.0, min(1.0, reward))

        alphas[ctx_key][chosen.name] += r_norm
        betas[ctx_key][chosen.name] += (1 - r_norm)

        opt_rew = env.optimal_reward(context)
        cum_regret += opt_rew - reward
        result.cumulative_regret.append(cum_regret)
        result.history.append((context, chosen.name, reward))
        result.context_arm_pulls[ctx_key][chosen.name] += 1
        result.context_arm_rewards[ctx_key][chosen.name] += reward
        result.total_pulls += 1

    return result


# ---------------------------------------------------------------------------
# Algorithm 3: Causal LinUCB
# ---------------------------------------------------------------------------

def _context_features(context: Context, context_vars: list[str],
                      context_domains: dict[str, list]) -> list[float]:
    """Convert context to a numeric feature vector.

    Binary/categorical values are one-hot encoded.
    """
    features = [1.0]  # bias term
    for v in context_vars:
        val = context.features.get(v)
        domain = context_domains.get(v, [0, 1])
        if len(domain) <= 2:
            # Binary: single feature
            features.append(1.0 if val == domain[-1] else 0.0)
        else:
            # One-hot for categorical
            for d in domain:
                features.append(1.0 if val == d else 0.0)
    return features


def causal_linucb(
    env: ContextualCausalEnv,
    rounds: int = 1000,
    alpha: float = 1.0,
    causal_prior_weight: float = 1.0,
    seed: int | None = None,
) -> ContextualResult:
    """Linear UCB with causal effect features.

    Features include context indicators and causal expected rewards.
    Uses ridge regression per arm.
    """
    rng = random.Random(seed)
    result = ContextualResult()

    # Feature dimension: context features + causal expected reward feature
    sample_ctx = Context(features={v: env.context_domains[v][0] for v in env.context_vars})
    base_dim = len(_context_features(sample_ctx, env.context_vars, env.context_domains))
    d = base_dim + 1  # +1 for causal expected reward feature

    # Per-arm LinUCB parameters
    A: dict[str, list[list[float]]] = {}  # d x d identity initially
    b: dict[str, list[float]] = {}  # d-dim zero vector

    for arm in env.arms:
        A[arm.name] = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        b[arm.name] = [0.0] * d

    cum_regret = 0.0

    for t in range(1, rounds + 1):
        context = env.sample_context()
        ctx_key = context.key(env.context_vars)

        # Build feature vector for each arm
        base_features = _context_features(context, env.context_vars, env.context_domains)

        best_ucb = float("-inf")
        chosen = env.arms[0]

        for arm in env.arms:
            # Add causal expected reward as a feature
            causal_er = 0.0
            if ctx_key in env._true_rewards:
                causal_er = env._true_rewards[ctx_key].get(arm.name, 0.0) * causal_prior_weight
            x = base_features + [causal_er]

            # Compute theta = A^{-1} b and UCB
            A_arm = A[arm.name]
            b_arm = b[arm.name]

            # Simple matrix inverse via Gaussian elimination
            theta = _solve_linear(A_arm, b_arm)
            if theta is None:
                ucb_val = rng.random()
            else:
                pred = sum(theta[i] * x[i] for i in range(d))
                # Confidence: sqrt(x^T A^{-1} x)
                A_inv_x = _solve_linear(A_arm, x)
                if A_inv_x is None:
                    conf = 1.0
                else:
                    conf = math.sqrt(max(0, sum(x[i] * A_inv_x[i] for i in range(d))))
                ucb_val = pred + alpha * conf

            if ucb_val > best_ucb:
                best_ucb = ucb_val
                chosen = arm

        reward = env.pull(chosen, context)

        # Update LinUCB parameters for chosen arm
        causal_er = 0.0
        if ctx_key in env._true_rewards:
            causal_er = env._true_rewards[ctx_key].get(chosen.name, 0.0) * causal_prior_weight
        x = base_features + [causal_er]

        # A += x x^T
        for i in range(d):
            for j in range(d):
                A[chosen.name][i][j] += x[i] * x[j]
        # b += reward * x
        for i in range(d):
            b[chosen.name][i] += reward * x[i]

        opt_rew = env.optimal_reward(context)
        cum_regret += opt_rew - reward
        result.cumulative_regret.append(cum_regret)
        result.history.append((context, chosen.name, reward))
        result.context_arm_pulls[ctx_key][chosen.name] += 1
        result.context_arm_rewards[ctx_key][chosen.name] += reward
        result.total_pulls += 1

    return result


def _solve_linear(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivot
        max_row = col
        max_val = abs(M[col][col])
        for row in range(col + 1, n):
            if abs(M[row][col]) > max_val:
                max_val = abs(M[row][col])
                max_row = row
        if max_val < 1e-12:
            return None
        M[col], M[max_row] = M[max_row], M[col]

        # Eliminate below
        pivot = M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-12:
            return None
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]

    return x


# ---------------------------------------------------------------------------
# Algorithm 4: CATE-Greedy
# ---------------------------------------------------------------------------

def cate_greedy(
    env: ContextualCausalEnv,
    rounds: int = 1000,
    explore_rounds: int = 200,
    seed: int | None = None,
) -> ContextualResult:
    """Explore then exploit using CATE (Conditional Average Treatment Effect).

    Phase 1 (explore): pull arms uniformly to estimate CATE per context.
    Phase 2 (exploit): use estimated CATEs to pick best arm per context.
    """
    rng = random.Random(seed)
    result = ContextualResult()

    # Per-context, per-arm: list of observed rewards
    observations: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cum_regret = 0.0

    # Learned policy (filled after exploration)
    learned: dict[tuple, Arm] = {}

    for t in range(1, rounds + 1):
        context = env.sample_context()
        ctx_key = context.key(env.context_vars)

        if t <= explore_rounds:
            # Explore: round-robin through arms
            chosen = env.arms[(t - 1) % len(env.arms)]
        else:
            # Exploit: use learned policy
            if ctx_key in learned:
                chosen = learned[ctx_key]
            else:
                # Never seen this context -- use global best
                chosen = _global_best_arm(observations, env.arms)

        reward = env.pull(chosen, context)
        observations[ctx_key][chosen.name].append(reward)

        # After exploration, compute policy
        if t == explore_rounds:
            learned = _compute_cate_policy(observations, env.arms)

        opt_rew = env.optimal_reward(context)
        cum_regret += opt_rew - reward
        result.cumulative_regret.append(cum_regret)
        result.history.append((context, chosen.name, reward))
        result.context_arm_pulls[ctx_key][chosen.name] += 1
        result.context_arm_rewards[ctx_key][chosen.name] += reward
        result.total_pulls += 1

    return result


def _compute_cate_policy(
    observations: dict[tuple, dict[str, list[float]]],
    arms: list[Arm],
) -> dict[tuple, Arm]:
    """Compute best arm per context from observed rewards."""
    policy = {}
    for ctx_key, arm_obs in observations.items():
        best_arm = arms[0]
        best_avg = float("-inf")
        for arm in arms:
            obs = arm_obs.get(arm.name, [])
            if obs:
                avg = sum(obs) / len(obs)
                if avg > best_avg:
                    best_avg = avg
                    best_arm = arm
        policy[ctx_key] = best_arm
    return policy


def _global_best_arm(
    observations: dict[tuple, dict[str, list[float]]],
    arms: list[Arm],
) -> Arm:
    """Find globally best arm across all contexts."""
    arm_totals: dict[str, float] = defaultdict(float)
    arm_counts: dict[str, int] = defaultdict(int)
    for ctx_key, arm_obs in observations.items():
        for arm_name, obs in arm_obs.items():
            arm_totals[arm_name] += sum(obs)
            arm_counts[arm_name] += len(obs)
    best = arms[0]
    best_avg = float("-inf")
    for arm in arms:
        n = arm_counts.get(arm.name, 0)
        if n > 0:
            avg = arm_totals[arm.name] / n
            if avg > best_avg:
                best_avg = avg
                best = arm
    return best


# ---------------------------------------------------------------------------
# Algorithm 5: Epsilon-Greedy with Causal Subgroup Decay
# ---------------------------------------------------------------------------

def epsilon_subgroup(
    env: ContextualCausalEnv,
    rounds: int = 1000,
    epsilon: float = 0.2,
    decay: float = 0.995,
    seed: int | None = None,
) -> ContextualResult:
    """Epsilon-greedy with per-subgroup exploration decay.

    Each context bin has its own epsilon that decays independently,
    allowing well-explored subgroups to exploit while rare subgroups
    keep exploring.
    """
    rng = random.Random(seed)
    result = ContextualResult()

    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    epsilons: dict[tuple, float] = defaultdict(lambda: epsilon)
    cum_regret = 0.0

    for t in range(1, rounds + 1):
        context = env.sample_context()
        ctx_key = context.key(env.context_vars)

        eps = epsilons[ctx_key]
        if rng.random() < eps:
            # Explore
            chosen = rng.choice(env.arms)
        else:
            # Exploit: best empirical arm for this context
            best_avg = float("-inf")
            chosen = env.arms[0]
            for arm in env.arms:
                n = counts[ctx_key][arm.name]
                if n == 0:
                    chosen = arm
                    break
                avg = totals[ctx_key][arm.name] / n
                if avg > best_avg:
                    best_avg = avg
                    chosen = arm

        reward = env.pull(chosen, context)
        counts[ctx_key][chosen.name] += 1
        totals[ctx_key][chosen.name] += reward
        epsilons[ctx_key] *= decay

        opt_rew = env.optimal_reward(context)
        cum_regret += opt_rew - reward
        result.cumulative_regret.append(cum_regret)
        result.history.append((context, chosen.name, reward))
        result.context_arm_pulls[ctx_key][chosen.name] += 1
        result.context_arm_rewards[ctx_key][chosen.name] += reward
        result.total_pulls += 1

    return result


# ---------------------------------------------------------------------------
# Algorithm 6: Causal Policy Tree
# ---------------------------------------------------------------------------

@dataclass
class PolicyTreeNode:
    """Node in a policy tree. Leaf = arm recommendation, internal = split."""
    is_leaf: bool = True
    arm: Arm | None = None  # for leaf nodes
    split_var: str | None = None  # for internal nodes
    split_val: object = None  # split threshold/value
    children: dict[object, "PolicyTreeNode"] = field(default_factory=dict)

    def predict(self, context: Context) -> Arm:
        if self.is_leaf:
            return self.arm
        val = context.features.get(self.split_var)
        if val in self.children:
            return self.children[val].predict(context)
        # Fallback: first child
        if self.children:
            return next(iter(self.children.values())).predict(context)
        return self.arm


def causal_policy_tree(
    env: ContextualCausalEnv,
    rounds: int = 1000,
    explore_rounds: int = 300,
    seed: int | None = None,
) -> ContextualResult:
    """Build a policy tree from exploration data, then exploit.

    Phase 1: Uniform exploration to collect (context, arm, reward) triples.
    Phase 2: Build a tree that splits on context variables to assign arms.
    Phase 3: Exploit the learned tree.
    """
    rng = random.Random(seed)
    result = ContextualResult()

    # Exploration data
    explore_data: list[tuple[Context, Arm, float]] = []
    cum_regret = 0.0
    tree = None

    for t in range(1, rounds + 1):
        context = env.sample_context()
        ctx_key = context.key(env.context_vars)

        if t <= explore_rounds:
            chosen = env.arms[(t - 1) % len(env.arms)]
            explore_data.append((context, chosen, 0.0))  # reward filled after pull
        elif tree is not None:
            chosen = tree.predict(context)
        else:
            chosen = rng.choice(env.arms)

        reward = env.pull(chosen, context)

        if t <= explore_rounds:
            explore_data[-1] = (context, chosen, reward)

        # Build tree after exploration
        if t == explore_rounds:
            tree = _build_policy_tree(explore_data, env)

        opt_rew = env.optimal_reward(context)
        cum_regret += opt_rew - reward
        result.cumulative_regret.append(cum_regret)
        result.history.append((context, chosen.name, reward))
        result.context_arm_pulls[ctx_key][chosen.name] += 1
        result.context_arm_rewards[ctx_key][chosen.name] += reward
        result.total_pulls += 1

    return result


def _build_policy_tree(
    data: list[tuple[Context, Arm, float]],
    env: ContextualCausalEnv,
    max_depth: int = 3,
    min_samples: int = 5,
) -> PolicyTreeNode:
    """Build a policy tree by greedy reward-maximizing splits."""
    return _build_tree_recursive(data, env, 0, max_depth, min_samples)


def _build_tree_recursive(
    data: list[tuple[Context, Arm, float]],
    env: ContextualCausalEnv,
    depth: int,
    max_depth: int,
    min_samples: int,
) -> PolicyTreeNode:
    if len(data) < min_samples or depth >= max_depth:
        # Leaf: assign best arm based on data
        return _make_leaf(data, env)

    # Try each context variable as a split
    best_gain = 0.0
    best_split = None
    best_children_data = None

    # Current best leaf reward
    leaf_reward = _leaf_expected_reward(data, env)

    for var in env.context_vars:
        domain = env.context_domains.get(var, [])
        if len(domain) < 2:
            continue

        # Split data by variable value
        splits: dict[object, list] = defaultdict(list)
        for ctx, arm, rew in data:
            val = ctx.features.get(var)
            splits[val].append((ctx, arm, rew))

        if len(splits) < 2:
            continue

        # Compute weighted child reward
        total_n = len(data)
        child_reward = 0.0
        for val, child_data in splits.items():
            if len(child_data) < min_samples:
                continue
            child_reward += len(child_data) / total_n * _leaf_expected_reward(child_data, env)

        gain = child_reward - leaf_reward
        if gain > best_gain:
            best_gain = gain
            best_split = var
            best_children_data = splits

    if best_split is None or best_children_data is None:
        return _make_leaf(data, env)

    node = PolicyTreeNode(is_leaf=False, split_var=best_split)
    for val, child_data in best_children_data.items():
        if len(child_data) >= min_samples:
            node.children[val] = _build_tree_recursive(
                child_data, env, depth + 1, max_depth, min_samples,
            )
        else:
            node.children[val] = _make_leaf(child_data, env)

    # Fallback arm for unseen values
    node.arm = _best_arm_from_data(data, env)
    return node


def _make_leaf(
    data: list[tuple[Context, Arm, float]],
    env: ContextualCausalEnv,
) -> PolicyTreeNode:
    arm = _best_arm_from_data(data, env)
    return PolicyTreeNode(is_leaf=True, arm=arm)


def _best_arm_from_data(
    data: list[tuple[Context, Arm, float]],
    env: ContextualCausalEnv,
) -> Arm:
    """Find arm with highest average reward in data."""
    arm_totals: dict[str, float] = defaultdict(float)
    arm_counts: dict[str, int] = defaultdict(int)
    for ctx, arm, rew in data:
        arm_totals[arm.name] += rew
        arm_counts[arm.name] += len([1])  # count
    # Fix: just count occurrences
    arm_counts2: dict[str, int] = defaultdict(int)
    arm_totals2: dict[str, float] = defaultdict(float)
    for ctx, arm, rew in data:
        arm_counts2[arm.name] += 1
        arm_totals2[arm.name] += rew

    best = env.arms[0]
    best_avg = float("-inf")
    for arm in env.arms:
        n = arm_counts2.get(arm.name, 0)
        if n > 0:
            avg = arm_totals2[arm.name] / n
            if avg > best_avg:
                best_avg = avg
                best = arm
    return best


def _leaf_expected_reward(
    data: list[tuple[Context, Arm, float]],
    env: ContextualCausalEnv,
) -> float:
    """Expected reward if we assign the best arm to all data points."""
    arm = _best_arm_from_data(data, env)
    arm_obs = [rew for ctx, a, rew in data if a.name == arm.name]
    if arm_obs:
        return sum(arm_obs) / len(arm_obs)
    return 0.0


# ---------------------------------------------------------------------------
# CATE Analysis
# ---------------------------------------------------------------------------

def estimate_cate(
    env: ContextualCausalEnv,
    arm_a: Arm,
    arm_b: Arm,
    n_samples: int = 500,
    seed: int | None = None,
) -> list[CATEEstimate]:
    """Estimate CATE = E[Y|do(a), ctx] - E[Y|do(b), ctx] for each context."""
    rng = random.Random(seed)
    estimates = []

    for ctx_key, arm_rews in env._true_rewards.items():
        ctx_dict = dict(zip(env.context_vars, ctx_key))
        er_a = arm_rews.get(arm_a.name, 0.0)
        er_b = arm_rews.get(arm_b.name, 0.0)
        cate = er_a - er_b

        # Estimate standard error via sampling
        diffs = []
        for _ in range(n_samples):
            ctx = Context(features=ctx_dict)
            r_a = env.pull(arm_a, ctx)
            r_b = env.pull(arm_b, ctx)
            diffs.append(r_a - r_b)

        mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
        if len(diffs) > 1:
            var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
            std_err = math.sqrt(var / len(diffs))
        else:
            std_err = 0.0

        estimates.append(CATEEstimate(
            subgroup=ctx_dict,
            arm_a=arm_a.name,
            arm_b=arm_b.name,
            cate=cate,
            n_samples=n_samples,
            std_err=std_err,
        ))

    return estimates


def subgroup_analysis(env: ContextualCausalEnv) -> SubgroupResult:
    """Analyze heterogeneous treatment effects across subgroups."""
    subgroups = []
    for ctx_key, arm_rews in env._true_rewards.items():
        ctx_dict = dict(zip(env.context_vars, ctx_key))
        optimal = env._optimal_arms[ctx_key]
        opt_rew = env._optimal_rewards[ctx_key]
        subgroups.append({
            "context_key": ctx_key,
            "context": ctx_dict,
            "optimal_arm": optimal.name,
            "expected_reward": opt_rew,
        })

    # Heterogeneity: variance of optimal rewards across subgroups
    rewards = [s["expected_reward"] for s in subgroups]
    if len(rewards) > 1:
        mean_r = sum(rewards) / len(rewards)
        heterogeneity = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    else:
        heterogeneity = 0.0

    # CATE estimates between all arm pairs for all contexts
    cate_estimates = []
    if len(env.arms) >= 2:
        ref_arm = env.arms[0]
        for arm in env.arms[1:]:
            for ctx_key, arm_rews in env._true_rewards.items():
                ctx_dict = dict(zip(env.context_vars, ctx_key))
                er_a = arm_rews.get(arm.name, 0.0)
                er_b = arm_rews.get(ref_arm.name, 0.0)
                cate_estimates.append(CATEEstimate(
                    subgroup=ctx_dict,
                    arm_a=arm.name,
                    arm_b=ref_arm.name,
                    cate=er_a - er_b,
                    n_samples=0,
                ))

    return SubgroupResult(
        subgroups=subgroups,
        heterogeneity_score=heterogeneity,
        cate_estimates=cate_estimates,
    )


# ---------------------------------------------------------------------------
# Algorithm Comparison
# ---------------------------------------------------------------------------

def compare_algorithms(
    env: ContextualCausalEnv,
    rounds: int = 500,
    seed: int = 42,
) -> dict[str, ContextualResult]:
    """Run all contextual causal bandit algorithms and compare."""
    return {
        "binned_ucb": binned_ucb_causal(env, rounds, seed=seed),
        "binned_thompson": binned_thompson_causal(env, rounds, seed=seed),
        "causal_linucb": causal_linucb(env, rounds, seed=seed),
        "cate_greedy": cate_greedy(env, rounds, explore_rounds=min(200, rounds // 3), seed=seed),
        "epsilon_subgroup": epsilon_subgroup(env, rounds, seed=seed),
        "policy_tree": causal_policy_tree(env, rounds, explore_rounds=min(200, rounds // 3), seed=seed),
    }


def regret_summary(results: dict[str, ContextualResult]) -> dict[str, dict]:
    """Summarize regret across algorithms."""
    summary = {}
    for name, res in results.items():
        policy = res.learned_policy([])  # empty if no context_vars passed
        summary[name] = {
            "total_regret": res.regret(),
            "average_regret": res.average_regret(),
            "total_pulls": res.total_pulls,
            "n_contexts_seen": len(res.context_arm_pulls),
        }
    return summary


# ---------------------------------------------------------------------------
# Example Environments
# ---------------------------------------------------------------------------

def build_treatment_env(seed: int | None = None) -> ContextualCausalEnv:
    """Medical treatment environment with age-dependent treatment effects.

    Causal graph: Age -> Treatment response, Age -> Recovery
    Treatment -> Recovery (but effect depends on Age).
    Young patients respond better to Treatment A.
    Old patients respond better to Treatment B.
    """
    bn = BayesianNetwork()
    bn.add_node("Age", domain=["young", "old"])
    bn.add_node("Treatment", domain=["A", "B"])
    bn.add_node("Recovery", domain=["good", "poor"])

    bn.add_edge("Age", "Recovery")
    bn.add_edge("Treatment", "Recovery")

    # P(Age)
    bn.set_cpt_dict("Age", {"young": 0.5, "old": 0.5})

    # P(Treatment) -- uniform prior (will be intervened on)
    bn.set_cpt_dict("Treatment", {"A": 0.5, "B": 0.5})

    # P(Recovery | Age, Treatment) -- heterogeneous effects
    bn.set_cpt_dict("Recovery", {
        ("young", "A"): {"good": 0.9, "poor": 0.1},  # Young + A -> good
        ("young", "B"): {"good": 0.4, "poor": 0.6},  # Young + B -> poor
        ("old", "A"): {"good": 0.3, "poor": 0.7},     # Old + A -> poor
        ("old", "B"): {"good": 0.8, "poor": 0.2},     # Old + B -> good
    })

    model = CausalModel(bn)
    arms = [
        Arm([Intervention("Treatment", "A")], name="Treatment_A"),
        Arm([Intervention("Treatment", "B")], name="Treatment_B"),
    ]

    return ContextualCausalEnv(
        causal_model=model,
        reward_var="Recovery",
        context_vars=["Age"],
        arms=arms,
        reward_mapping={"good": 1.0, "poor": 0.0},
        seed=seed,
    )


def build_advertising_env(seed: int | None = None) -> ContextualCausalEnv:
    """Advertising environment with user-type-dependent ad effectiveness.

    Causal graph: UserType -> Interest, UserType -> Purchase
    AdChannel -> Purchase (effect depends on UserType and Interest).
    """
    bn = BayesianNetwork()
    bn.add_node("UserType", domain=["tech", "casual", "senior"])
    bn.add_node("Interest", domain=["high", "low"])
    bn.add_node("AdChannel", domain=["social", "search", "email"])
    bn.add_node("Purchase", domain=["yes", "no"])

    bn.add_edge("UserType", "Interest")
    bn.add_edge("UserType", "Purchase")
    bn.add_edge("Interest", "Purchase")
    bn.add_edge("AdChannel", "Purchase")

    bn.set_cpt_dict("UserType", {"tech": 0.4, "casual": 0.4, "senior": 0.2})

    bn.set_cpt_dict("Interest", {
        ("tech",): {"high": 0.8, "low": 0.2},
        ("casual",): {"high": 0.5, "low": 0.5},
        ("senior",): {"high": 0.3, "low": 0.7},
    })

    bn.set_cpt_dict("AdChannel", {"social": 0.33, "search": 0.34, "email": 0.33})

    # Purchase depends on UserType, Interest, AdChannel
    # Tech users buy via search, casual via social, seniors via email
    purchase_cpt = {}
    for ut in ["tech", "casual", "senior"]:
        for interest in ["high", "low"]:
            for ad in ["social", "search", "email"]:
                base = 0.1
                if interest == "high":
                    base += 0.2
                # User-type-specific channel effects
                if ut == "tech" and ad == "search":
                    base += 0.5
                elif ut == "tech" and ad == "social":
                    base += 0.2
                elif ut == "casual" and ad == "social":
                    base += 0.5
                elif ut == "casual" and ad == "email":
                    base += 0.1
                elif ut == "senior" and ad == "email":
                    base += 0.5
                elif ut == "senior" and ad == "search":
                    base += 0.15
                else:
                    base += 0.1
                base = min(base, 0.95)
                purchase_cpt[(ut, interest, ad)] = {"yes": base, "no": 1 - base}

    bn.set_cpt_dict("Purchase", purchase_cpt)

    model = CausalModel(bn)
    arms = [
        Arm([Intervention("AdChannel", "social")], name="social"),
        Arm([Intervention("AdChannel", "search")], name="search"),
        Arm([Intervention("AdChannel", "email")], name="email"),
    ]

    return ContextualCausalEnv(
        causal_model=model,
        reward_var="Purchase",
        context_vars=["UserType", "Interest"],
        arms=arms,
        reward_mapping={"yes": 1.0, "no": 0.0},
        seed=seed,
    )


def build_simple_heterogeneous_env(seed: int | None = None) -> ContextualCausalEnv:
    """Simple 2-context, 2-arm env where optimal arm flips with context."""
    bn = BayesianNetwork()
    bn.add_node("X", domain=[0, 1])
    bn.add_node("A", domain=[0, 1])
    bn.add_node("Y", domain=[0, 1])

    bn.add_edge("X", "Y")
    bn.add_edge("A", "Y")

    bn.set_cpt("X", {(0,): 0.5, (1,): 0.5})
    bn.set_cpt("A", {(0,): 0.5, (1,): 0.5})

    # Y = 1 when X == A (context matches action)
    bn.set_cpt("Y", {
        (0, 0, 0): 0.1, (0, 0, 1): 0.9,  # X=0, A=0 -> Y=1
        (0, 1, 0): 0.8, (0, 1, 1): 0.2,  # X=0, A=1 -> Y=0
        (1, 0, 0): 0.8, (1, 0, 1): 0.2,  # X=1, A=0 -> Y=0
        (1, 1, 0): 0.1, (1, 1, 1): 0.9,  # X=1, A=1 -> Y=1
    })

    model = CausalModel(bn)
    arms = [
        Arm([Intervention("A", 0)], name="A0"),
        Arm([Intervention("A", 1)], name="A1"),
    ]

    return ContextualCausalEnv(
        causal_model=model,
        reward_var="Y",
        context_vars=["X"],
        arms=arms,
        seed=seed,
    )


def build_homogeneous_env(seed: int | None = None) -> ContextualCausalEnv:
    """Environment where the same arm is optimal regardless of context."""
    bn = BayesianNetwork()
    bn.add_node("X", domain=[0, 1])
    bn.add_node("A", domain=[0, 1])
    bn.add_node("Y", domain=[0, 1])

    bn.add_edge("X", "Y")
    bn.add_edge("A", "Y")

    bn.set_cpt("X", {(0,): 0.5, (1,): 0.5})
    bn.set_cpt("A", {(0,): 0.5, (1,): 0.5})

    # A=1 is always better regardless of X
    bn.set_cpt("Y", {
        (0, 0, 0): 0.7, (0, 0, 1): 0.3,
        (0, 1, 0): 0.2, (0, 1, 1): 0.8,
        (1, 0, 0): 0.6, (1, 0, 1): 0.4,
        (1, 1, 0): 0.1, (1, 1, 1): 0.9,
    })

    model = CausalModel(bn)
    arms = [
        Arm([Intervention("A", 0)], name="A0"),
        Arm([Intervention("A", 1)], name="A1"),
    ]

    return ContextualCausalEnv(
        causal_model=model,
        reward_var="Y",
        context_vars=["X"],
        arms=arms,
        seed=seed,
    )
