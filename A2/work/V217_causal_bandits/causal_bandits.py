"""V217: Causal Bandits -- Intervention Selection via Causal Reasoning.

Composes V214 (Causal Discovery) + V211 (Causal Inference) + V209 (Bayesian Networks).

A causal bandit is a multi-armed bandit where each arm corresponds to an
intervention do(X=x) on some variable in a causal graph. The agent aims to
find the intervention that maximizes a designated reward variable.

Key insight: causal knowledge (the graph structure) lets us compute
interventional distributions P(Y|do(X=x)) without actually pulling the arm,
dramatically reducing sample complexity compared to standard bandits.

Components:
- CausalBanditEnv: environment with a causal graph, interventional arms, reward variable
- UCB-Causal: UCB1 with causal prior initialization
- Thompson Sampling with causal posteriors
- Pure Causal: exploits known graph (no exploration needed when graph is known)
- Observational-Interventional: combines free observational data with costly interventions
- Causal structure learning bandit: learns the graph while optimizing interventions
"""

from __future__ import annotations

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V214_causal_discovery'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V211_causal_inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))

from causal_discovery import (
    pc_algorithm, hill_climbing, learn_bn_structure,
    sample_from_bn, structural_hamming_distance,
)
from causal_inference import CausalModel, variable_elimination
from bayesian_networks import BayesianNetwork, Factor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Intervention:
    """An intervention do(variable=value)."""

    def __init__(self, variable: str, value: object):
        self.variable = variable
        self.value = value

    def __repr__(self) -> str:
        return f"do({self.variable}={self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Intervention):
            return NotImplemented
        return self.variable == other.variable and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.variable, self.value))


class Arm:
    """A bandit arm = one or more simultaneous interventions."""

    def __init__(self, interventions: list[Intervention] | None = None,
                 name: str | None = None):
        self.interventions = interventions or []
        self.name = name or self._auto_name()

    def _auto_name(self) -> str:
        if not self.interventions:
            return "observe"
        return ", ".join(str(i) for i in self.interventions)

    def to_dict(self) -> dict[str, object]:
        """Convert to {variable: value} dict for CausalModel.do()."""
        return {i.variable: i.value for i in self.interventions}

    def __repr__(self) -> str:
        return f"Arm({self.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Arm):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class BanditResult:
    """Result of running a causal bandit algorithm."""

    def __init__(self):
        self.best_arm: Arm | None = None
        self.best_reward: float = 0.0
        self.arm_pulls: dict[str, int] = {}
        self.arm_rewards: dict[str, float] = {}
        self.cumulative_regret: list[float] = []
        self.total_pulls: int = 0
        self.history: list[tuple[str, float]] = []  # (arm_name, reward)

    def regret(self) -> float:
        """Total cumulative regret."""
        return self.cumulative_regret[-1] if self.cumulative_regret else 0.0

    def average_regret(self) -> float:
        """Average per-step regret."""
        if self.total_pulls == 0:
            return 0.0
        return self.regret() / self.total_pulls


# ---------------------------------------------------------------------------
# Causal Bandit Environment
# ---------------------------------------------------------------------------

class CausalBanditEnv:
    """A causal bandit environment.

    The environment is defined by a causal model (BN + graph) and:
    - A set of arms (interventions the agent can perform)
    - A reward variable (the variable we want to maximize)
    - An optional observation arm (no intervention, just observe)

    Pulling an arm:
    1. Applies the intervention to the causal model (graph surgery)
    2. Samples from the resulting mutilated distribution
    3. Returns the reward variable's value as numeric reward
    """

    def __init__(self, causal_model: CausalModel, reward_var: str,
                 arms: list[Arm] | None = None,
                 reward_mapping: dict[object, float] | None = None,
                 seed: int | None = None):
        """
        Args:
            causal_model: The underlying causal model.
            reward_var: Variable whose value is the reward.
            arms: List of arms (interventions). If None, auto-generates from
                  all single-variable interventions.
            reward_mapping: Maps reward_var values to numeric rewards.
                          If None, values are used directly (must be numeric).
            seed: Random seed.
        """
        self.model = causal_model
        self.bn = causal_model.bn
        self.reward_var = reward_var
        self.reward_mapping = reward_mapping
        self.rng = random.Random(seed)

        if arms is None:
            self.arms = self._auto_arms()
        else:
            self.arms = arms

        # Precompute true interventional distributions for each arm
        self._true_distributions: dict[str, dict[object, float]] = {}
        self._true_expected_rewards: dict[str, float] = {}
        self._compute_true_rewards()

        # Optimal arm
        self.optimal_arm = max(self.arms, key=lambda a: self._true_expected_rewards[a.name])
        self.optimal_reward = self._true_expected_rewards[self.optimal_arm.name]

    def _auto_arms(self) -> list[Arm]:
        """Generate arms from all single-variable interventions."""
        arms = []
        # Add observation arm
        arms.append(Arm([], name="observe"))
        # Add intervention arms for each non-reward variable
        for var in self.bn.nodes:
            if var == self.reward_var:
                continue
            for val in self.bn.domains[var]:
                arms.append(Arm([Intervention(var, val)]))
        return arms

    def _compute_true_rewards(self) -> None:
        """Precompute P(reward_var | do(arm)) for each arm."""
        for arm in self.arms:
            interventions = arm.to_dict()
            if interventions:
                dist = self.model.interventional_query(
                    [self.reward_var], interventions
                )
                dist = dist.normalize()
                p = {}
                for val in self.bn.domains[self.reward_var]:
                    p[val] = dist.get({self.reward_var: val})
            else:
                # Observational: marginal P(reward_var)
                dist = variable_elimination(self.bn, [self.reward_var])
                dist = dist.normalize()
                p = {}
                for val in self.bn.domains[self.reward_var]:
                    p[val] = dist.get({self.reward_var: val})

            self._true_distributions[arm.name] = p
            self._true_expected_rewards[arm.name] = self._expected_reward(p)

    def _expected_reward(self, dist: dict[object, float]) -> float:
        """Compute expected reward from a distribution over reward_var values."""
        total = 0.0
        for val, prob in dist.items():
            if self.reward_mapping:
                total += self.reward_mapping.get(val, 0.0) * prob
            else:
                total += float(val) * prob
        return total

    def pull(self, arm: Arm) -> float:
        """Pull an arm and get a stochastic reward.

        Applies the intervention, samples from the resulting distribution,
        returns the numeric reward.
        """
        dist = self._true_distributions[arm.name]
        # Sample from the distribution
        r = self.rng.random()
        cumulative = 0.0
        sampled_val = None
        for val, prob in dist.items():
            cumulative += prob
            if r <= cumulative:
                sampled_val = val
                break
        if sampled_val is None:
            sampled_val = list(dist.keys())[-1]

        if self.reward_mapping:
            return self.reward_mapping.get(sampled_val, 0.0)
        return float(sampled_val)

    def true_expected_reward(self, arm: Arm) -> float:
        """Return the true expected reward for an arm (for regret computation)."""
        return self._true_expected_rewards[arm.name]

    def observational_sample(self) -> dict[str, object]:
        """Draw one observational (non-interventional) sample from the BN."""
        samples = sample_from_bn(self.bn, 1, seed=self.rng.randint(0, 2**31))
        return samples[0]


# ---------------------------------------------------------------------------
# Algorithm 1: Pure Causal (known graph, no exploration)
# ---------------------------------------------------------------------------

def pure_causal(env: CausalBanditEnv) -> BanditResult:
    """When the causal graph is fully known, compute optimal arm analytically.

    No exploration needed -- just compute P(Y|do(X=x)) for each arm
    and pick the best one. This is the oracle baseline.
    """
    result = BanditResult()
    best_arm = None
    best_er = float('-inf')

    for arm in env.arms:
        er = env.true_expected_reward(arm)
        result.arm_rewards[arm.name] = er
        result.arm_pulls[arm.name] = 0
        if er > best_er:
            best_er = er
            best_arm = arm

    result.best_arm = best_arm
    result.best_reward = best_er
    result.total_pulls = 0
    return result


# ---------------------------------------------------------------------------
# Algorithm 2: UCB-Causal (UCB1 with causal prior initialization)
# ---------------------------------------------------------------------------

def ucb_causal(env: CausalBanditEnv, rounds: int = 1000,
               causal_prior_weight: int = 5,
               exploration_constant: float = 2.0,
               seed: int | None = None) -> BanditResult:
    """UCB1 with causal model providing prior reward estimates.

    The causal model initializes each arm's reward estimate as if
    we had `causal_prior_weight` virtual pulls with the expected
    interventional reward. This gives UCB a warm start.
    """
    rng = random.Random(seed)
    result = BanditResult()
    n_arms = len(env.arms)

    # Initialize with causal priors
    counts = {}
    totals = {}
    for arm in env.arms:
        er = env.true_expected_reward(arm)
        if causal_prior_weight > 0:
            counts[arm.name] = causal_prior_weight
            totals[arm.name] = er * causal_prior_weight
        else:
            counts[arm.name] = 0
            totals[arm.name] = 0.0
        result.arm_pulls[arm.name] = 0

    cumulative_regret = 0.0
    total_t = max(1, causal_prior_weight * n_arms)  # virtual time from priors

    for t in range(rounds):
        total_t += 1

        # UCB1 arm selection
        best_ucb = float('-inf')
        selected = None
        for arm in env.arms:
            if counts[arm.name] == 0:
                selected = arm
                break
            mean = totals[arm.name] / counts[arm.name]
            bonus = exploration_constant * math.sqrt(
                math.log(total_t) / counts[arm.name]
            )
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                selected = arm

        # Pull
        reward = env.pull(selected)
        counts[selected.name] += 1
        totals[selected.name] += reward
        result.arm_pulls[selected.name] += 1

        # Regret
        cumulative_regret += env.optimal_reward - reward
        result.cumulative_regret.append(cumulative_regret)
        result.history.append((selected.name, reward))

    # Final
    result.total_pulls = rounds
    best_arm = max(env.arms, key=lambda a: totals[a.name] / counts[a.name])
    result.best_arm = best_arm
    result.best_reward = totals[best_arm.name] / counts[best_arm.name]
    result.arm_rewards = {a.name: totals[a.name] / counts[a.name] for a in env.arms}
    return result


# ---------------------------------------------------------------------------
# Algorithm 3: Thompson Sampling with Causal Priors
# ---------------------------------------------------------------------------

def thompson_causal(env: CausalBanditEnv, rounds: int = 1000,
                    prior_alpha: float = 1.0, prior_beta: float = 1.0,
                    causal_prior_strength: int = 5,
                    seed: int | None = None) -> BanditResult:
    """Thompson Sampling with Beta posteriors initialized from causal model.

    Uses Beta(alpha, beta) conjugate priors for Bernoulli rewards.
    Causal model initializes the Beta parameters based on expected
    interventional reward.

    For non-Bernoulli rewards, rewards are normalized to [0, 1].
    """
    rng = random.Random(seed)
    result = BanditResult()

    # Determine reward range for normalization
    all_rewards = []
    for arm in env.arms:
        dist = env._true_distributions[arm.name]
        for val in dist:
            if env.reward_mapping:
                all_rewards.append(env.reward_mapping.get(val, 0.0))
            else:
                all_rewards.append(float(val))
    r_min = min(all_rewards)
    r_max = max(all_rewards)
    r_range = r_max - r_min if r_max > r_min else 1.0

    def normalize_reward(r: float) -> float:
        return (r - r_min) / r_range

    # Initialize Beta parameters with causal priors
    alphas = {}
    betas = {}
    for arm in env.arms:
        er = env.true_expected_reward(arm)
        er_norm = normalize_reward(er)
        # Virtual successes/failures from causal model
        alphas[arm.name] = prior_alpha + er_norm * causal_prior_strength
        betas[arm.name] = prior_beta + (1 - er_norm) * causal_prior_strength
        result.arm_pulls[arm.name] = 0

    cumulative_regret = 0.0

    for t in range(rounds):
        # Sample from each arm's Beta posterior
        best_sample = float('-inf')
        selected = None
        for arm in env.arms:
            sample = rng.betavariate(alphas[arm.name], betas[arm.name])
            if sample > best_sample:
                best_sample = sample
                selected = arm

        # Pull
        reward = env.pull(selected)
        norm_reward = normalize_reward(reward)
        result.arm_pulls[selected.name] += 1

        # Update Beta posterior
        alphas[selected.name] += norm_reward
        betas[selected.name] += (1 - norm_reward)

        # Regret
        cumulative_regret += env.optimal_reward - reward
        result.cumulative_regret.append(cumulative_regret)
        result.history.append((selected.name, reward))

    # Final
    result.total_pulls = rounds
    means = {a.name: alphas[a.name] / (alphas[a.name] + betas[a.name])
             for a in env.arms}
    best_arm = max(env.arms, key=lambda a: means[a.name])
    result.best_arm = best_arm
    result.best_reward = env.true_expected_reward(best_arm)
    result.arm_rewards = {a.name: means[a.name] for a in env.arms}
    return result


# ---------------------------------------------------------------------------
# Algorithm 4: Observational-Interventional Bandit
# ---------------------------------------------------------------------------

def obs_int_bandit(env: CausalBanditEnv, rounds: int = 1000,
                   obs_per_round: int = 10,
                   obs_cost: float = 0.0,
                   int_cost: float = 1.0,
                   exploration_constant: float = 2.0,
                   seed: int | None = None) -> BanditResult:
    """Bandit that uses cheap observational data alongside costly interventions.

    Each round the agent can either:
    1. Observe: collect obs_per_round observational samples (cost: obs_cost each)
    2. Intervene: pull an arm (cost: int_cost)

    Observational data updates reward estimates via the adjustment formula
    when a valid adjustment set exists.
    """
    rng = random.Random(seed)
    result = BanditResult()

    # Track per-arm statistics
    int_counts = {a.name: 0 for a in env.arms}
    int_totals = {a.name: 0.0 for a in env.arms}
    obs_estimates = {a.name: None for a in env.arms}
    obs_data: list[dict[str, object]] = []

    cumulative_regret = 0.0

    for arm in env.arms:
        result.arm_pulls[arm.name] = 0

    for t in range(rounds):
        # Decide: observe or intervene
        # Observe more early on, intervene more later (epsilon-decreasing)
        explore_prob = max(0.1, 1.0 - t / rounds)

        if t < len(env.arms):
            # Initial round-robin to get at least one pull per arm
            selected = env.arms[t % len(env.arms)]
            reward = env.pull(selected)
            int_counts[selected.name] += 1
            int_totals[selected.name] += reward
            result.arm_pulls[selected.name] += 1
        elif rng.random() < explore_prob and obs_cost < int_cost:
            # Collect observational data
            for _ in range(obs_per_round):
                obs_data.append(env.observational_sample())

            # Update observational estimates for each arm
            if len(obs_data) >= 50:
                obs_estimates = _estimate_from_observations(
                    env, obs_data
                )

            # Still need to select an arm for regret tracking
            # Use UCB on combined estimates
            selected = _ucb_select(env, int_counts, int_totals,
                                    obs_estimates, t + 1,
                                    exploration_constant)
            reward = env.pull(selected)
            int_counts[selected.name] += 1
            int_totals[selected.name] += reward
            result.arm_pulls[selected.name] += 1
        else:
            # Intervene using UCB
            selected = _ucb_select(env, int_counts, int_totals,
                                    obs_estimates, t + 1,
                                    exploration_constant)
            reward = env.pull(selected)
            int_counts[selected.name] += 1
            int_totals[selected.name] += reward
            result.arm_pulls[selected.name] += 1

        cumulative_regret += env.optimal_reward - reward
        result.cumulative_regret.append(cumulative_regret)
        result.history.append((selected.name, reward))

    result.total_pulls = rounds
    combined = _combine_estimates(env, int_counts, int_totals, obs_estimates)
    best_arm = max(env.arms, key=lambda a: combined.get(a.name, 0))
    result.best_arm = best_arm
    result.best_reward = env.true_expected_reward(best_arm)
    result.arm_rewards = combined
    return result


def _estimate_from_observations(env: CausalBanditEnv,
                                 obs_data: list[dict]) -> dict[str, float | None]:
    """Estimate interventional rewards from observational data.

    Uses the adjustment formula when possible:
    P(Y|do(X=x)) = sum_z P(Y|X=x,Z=z) P(Z=z)
    where Z is a valid adjustment set.
    """
    estimates: dict[str, float | None] = {}

    for arm in env.arms:
        if not arm.interventions:
            # Observation arm: just marginal reward
            vals = []
            for sample in obs_data:
                v = sample.get(env.reward_var)
                if v is not None:
                    if env.reward_mapping:
                        vals.append(env.reward_mapping.get(v, 0.0))
                    else:
                        vals.append(float(v))
            estimates[arm.name] = sum(vals) / len(vals) if vals else None
            continue

        # For intervention arms, try simple conditioning
        # (valid when no confounding -- approximation)
        iv = arm.interventions[0]
        matching = [s for s in obs_data
                    if s.get(iv.variable) == iv.value]
        if len(matching) >= 10:
            vals = []
            for s in matching:
                v = s.get(env.reward_var)
                if v is not None:
                    if env.reward_mapping:
                        vals.append(env.reward_mapping.get(v, 0.0))
                    else:
                        vals.append(float(v))
            estimates[arm.name] = sum(vals) / len(vals) if vals else None
        else:
            estimates[arm.name] = None

    return estimates


def _ucb_select(env: CausalBanditEnv,
                counts: dict[str, int],
                totals: dict[str, float],
                obs_estimates: dict[str, float | None],
                t: int, c: float) -> Arm:
    """Select arm using UCB with combined interventional + observational estimates."""
    best_ucb = float('-inf')
    selected = env.arms[0]

    for arm in env.arms:
        n = counts[arm.name]
        if n == 0:
            return arm  # Explore unvisited arms first

        mean = totals[arm.name] / n

        # Blend with observational estimate if available
        obs_est = obs_estimates.get(arm.name)
        if obs_est is not None:
            # Weighted combination: more weight to interventional with more pulls
            weight = n / (n + 10)  # 10 virtual obs samples
            mean = weight * mean + (1 - weight) * obs_est

        bonus = c * math.sqrt(math.log(t) / n)
        ucb = mean + bonus
        if ucb > best_ucb:
            best_ucb = ucb
            selected = arm

    return selected


def _combine_estimates(env: CausalBanditEnv,
                       counts: dict[str, int],
                       totals: dict[str, float],
                       obs_estimates: dict[str, float | None]) -> dict[str, float]:
    """Combine interventional and observational estimates."""
    combined = {}
    for arm in env.arms:
        n = counts[arm.name]
        if n > 0:
            int_est = totals[arm.name] / n
        else:
            int_est = 0.0

        obs_est = obs_estimates.get(arm.name)
        if obs_est is not None and n > 0:
            weight = n / (n + 10)
            combined[arm.name] = weight * int_est + (1 - weight) * obs_est
        elif obs_est is not None:
            combined[arm.name] = obs_est
        else:
            combined[arm.name] = int_est

    return combined


# ---------------------------------------------------------------------------
# Algorithm 5: Causal Structure Learning Bandit
# ---------------------------------------------------------------------------

def learning_bandit(env: CausalBanditEnv, rounds: int = 1000,
                    learn_interval: int = 100,
                    obs_per_learn: int = 200,
                    exploration_constant: float = 2.0,
                    seed: int | None = None) -> BanditResult:
    """Bandit that learns the causal structure while optimizing.

    Periodically collects observational data and runs causal discovery
    to learn/refine the causal graph. Uses the learned graph to compute
    causal reward estimates, and runs UCB on top.

    This is the hardest setting: unknown graph + unknown rewards.
    """
    rng = random.Random(seed)
    result = BanditResult()

    counts = {a.name: 0 for a in env.arms}
    totals = {a.name: 0.0 for a in env.arms}
    obs_data: list[dict[str, object]] = []
    learned_estimates: dict[str, float | None] = {a.name: None for a in env.arms}

    cumulative_regret = 0.0

    for arm in env.arms:
        result.arm_pulls[arm.name] = 0

    for t in range(rounds):
        # Periodically collect data and learn structure
        if t > 0 and t % learn_interval == 0:
            for _ in range(obs_per_learn):
                obs_data.append(env.observational_sample())

            if len(obs_data) >= 100:
                learned_estimates = _learn_and_estimate(
                    env, obs_data
                )

        # Select arm via UCB with learned estimates as priors
        if t < len(env.arms):
            selected = env.arms[t % len(env.arms)]
        else:
            selected = _ucb_with_learned(env, counts, totals,
                                          learned_estimates, t + 1,
                                          exploration_constant)

        reward = env.pull(selected)
        counts[selected.name] += 1
        totals[selected.name] += reward
        result.arm_pulls[selected.name] += 1

        cumulative_regret += env.optimal_reward - reward
        result.cumulative_regret.append(cumulative_regret)
        result.history.append((selected.name, reward))

    result.total_pulls = rounds
    combined = {}
    for arm in env.arms:
        n = counts[arm.name]
        combined[arm.name] = totals[arm.name] / n if n > 0 else 0.0
    best_arm = max(env.arms, key=lambda a: combined[a.name])
    result.best_arm = best_arm
    result.best_reward = env.true_expected_reward(best_arm)
    result.arm_rewards = combined
    return result


def _learn_and_estimate(env: CausalBanditEnv,
                         obs_data: list[dict]) -> dict[str, float | None]:
    """Learn causal structure from data and estimate interventional rewards."""
    estimates: dict[str, float | None] = {}

    try:
        # Learn BN structure via hill climbing
        bn = learn_bn_structure(obs_data, method="hc", max_parents=3,
                                max_iterations=200)
        model = CausalModel(bn)

        for arm in env.arms:
            interventions = arm.to_dict()
            if not interventions:
                # Observational arm
                dist = variable_elimination(bn, [env.reward_var])
                dist = dist.normalize()
                er = 0.0
                for val in bn.domains[env.reward_var]:
                    p = dist.get({env.reward_var: val})
                    if env.reward_mapping:
                        er += env.reward_mapping.get(val, 0.0) * p
                    else:
                        er += float(val) * p
                estimates[arm.name] = er
            else:
                try:
                    dist = model.interventional_query(
                        [env.reward_var], interventions
                    )
                    dist = dist.normalize()
                    er = 0.0
                    for val in bn.domains[env.reward_var]:
                        p = dist.get({env.reward_var: val})
                        if env.reward_mapping:
                            er += env.reward_mapping.get(val, 0.0) * p
                        else:
                            er += float(val) * p
                    estimates[arm.name] = er
                except Exception:
                    estimates[arm.name] = None

    except Exception:
        # Structure learning failed -- return None estimates
        for arm in env.arms:
            estimates[arm.name] = None

    return estimates


def _ucb_with_learned(env: CausalBanditEnv,
                       counts: dict[str, int],
                       totals: dict[str, float],
                       learned: dict[str, float | None],
                       t: int, c: float) -> Arm:
    """UCB selection with learned causal estimates as bonus."""
    best_ucb = float('-inf')
    selected = env.arms[0]

    for arm in env.arms:
        n = counts[arm.name]
        if n == 0:
            return arm

        mean = totals[arm.name] / n

        # Blend with learned estimate
        le = learned.get(arm.name)
        if le is not None:
            weight = n / (n + 5)
            mean = weight * mean + (1 - weight) * le

        bonus = c * math.sqrt(math.log(t) / n)
        ucb = mean + bonus
        if ucb > best_ucb:
            best_ucb = ucb
            selected = arm

    return selected


# ---------------------------------------------------------------------------
# Algorithm 6: Epsilon-Greedy with Causal Bounds
# ---------------------------------------------------------------------------

def epsilon_causal(env: CausalBanditEnv, rounds: int = 1000,
                   epsilon: float = 0.1, decay: float = 0.999,
                   use_causal_init: bool = True,
                   seed: int | None = None) -> BanditResult:
    """Simple epsilon-greedy with optional causal initialization.

    When use_causal_init=True, the initial reward estimates come from
    the causal model rather than being zero.
    """
    rng = random.Random(seed)
    result = BanditResult()

    counts = {a.name: 0 for a in env.arms}
    totals = {a.name: 0.0 for a in env.arms}

    if use_causal_init:
        for arm in env.arms:
            er = env.true_expected_reward(arm)
            counts[arm.name] = 1
            totals[arm.name] = er

    for arm in env.arms:
        result.arm_pulls[arm.name] = 0

    cumulative_regret = 0.0
    current_epsilon = epsilon

    for t in range(rounds):
        if rng.random() < current_epsilon:
            # Explore
            selected = rng.choice(env.arms)
        else:
            # Exploit
            means = {}
            for arm in env.arms:
                n = counts[arm.name]
                means[arm.name] = totals[arm.name] / n if n > 0 else 0.0
            selected = max(env.arms, key=lambda a: means[a.name])

        reward = env.pull(selected)
        counts[selected.name] += 1
        totals[selected.name] += reward
        result.arm_pulls[selected.name] += 1

        cumulative_regret += env.optimal_reward - reward
        result.cumulative_regret.append(cumulative_regret)
        result.history.append((selected.name, reward))

        current_epsilon *= decay

    result.total_pulls = rounds
    means = {a.name: totals[a.name] / counts[a.name] if counts[a.name] > 0
             else 0.0 for a in env.arms}
    best_arm = max(env.arms, key=lambda a: means[a.name])
    result.best_arm = best_arm
    result.best_reward = env.true_expected_reward(best_arm)
    result.arm_rewards = means
    return result


# ---------------------------------------------------------------------------
# Regret analysis and comparison
# ---------------------------------------------------------------------------

def compare_algorithms(env: CausalBanditEnv, rounds: int = 500,
                       seed: int = 42) -> dict[str, BanditResult]:
    """Run all algorithms on the same environment and compare."""
    results = {}

    results['pure_causal'] = pure_causal(env)
    results['ucb_causal'] = ucb_causal(env, rounds=rounds, seed=seed)
    results['thompson_causal'] = thompson_causal(env, rounds=rounds, seed=seed)
    results['epsilon_causal'] = epsilon_causal(env, rounds=rounds, seed=seed)
    results['epsilon_no_causal'] = epsilon_causal(
        env, rounds=rounds, seed=seed, use_causal_init=False
    )
    results['obs_int'] = obs_int_bandit(env, rounds=rounds, seed=seed)

    return results


def regret_summary(results: dict[str, BanditResult]) -> dict[str, dict]:
    """Summarize regret across algorithms."""
    summary = {}
    for name, res in results.items():
        summary[name] = {
            'best_arm': res.best_arm.name if res.best_arm else None,
            'total_regret': res.regret(),
            'avg_regret': res.average_regret(),
            'total_pulls': res.total_pulls,
            'arm_pulls': dict(res.arm_pulls),
        }
    return summary


# ---------------------------------------------------------------------------
# Causal bandit gap analysis
# ---------------------------------------------------------------------------

def interventional_gap(env: CausalBanditEnv) -> dict[str, float]:
    """Compute the gap between each arm and the optimal arm.

    The gap Delta_a = mu* - mu_a determines the regret contribution
    of pulling arm a. Arms with small gaps are harder to distinguish.
    """
    gaps = {}
    for arm in env.arms:
        gaps[arm.name] = env.optimal_reward - env.true_expected_reward(arm)
    return gaps


def confounding_analysis(env: CausalBanditEnv) -> dict[str, dict]:
    """Analyze confounding structure for each arm.

    For each intervention variable, check if there's confounding
    between the intervention and the reward (observational != interventional).
    """
    analysis = {}

    # Compute observational conditionals
    obs_data = sample_from_bn(env.bn, 5000, seed=42)

    for arm in env.arms:
        if not arm.interventions:
            analysis[arm.name] = {'type': 'observational', 'confounded': False}
            continue

        iv = arm.interventions[0]
        # Observational estimate: P(Y|X=x)
        matching = [s for s in obs_data if s.get(iv.variable) == iv.value]
        if len(matching) < 10:
            analysis[arm.name] = {
                'type': 'intervention',
                'confounded': None,
                'reason': 'insufficient data'
            }
            continue

        obs_reward = 0.0
        for s in matching:
            v = s[env.reward_var]
            if env.reward_mapping:
                obs_reward += env.reward_mapping.get(v, 0.0)
            else:
                obs_reward += float(v)
        obs_reward /= len(matching)

        # Interventional (true)
        int_reward = env.true_expected_reward(arm)

        # If they differ significantly, there's confounding
        diff = abs(obs_reward - int_reward)
        confounded = diff > 0.05  # threshold

        analysis[arm.name] = {
            'type': 'intervention',
            'observational_estimate': obs_reward,
            'interventional_truth': int_reward,
            'difference': diff,
            'confounded': confounded,
        }

    return analysis


# ---------------------------------------------------------------------------
# Example environments
# ---------------------------------------------------------------------------

def build_treatment_env(seed: int | None = None) -> CausalBanditEnv:
    """Medical treatment selection with confounding.

    Graph: Age -> Treatment, Age -> Recovery, Treatment -> Recovery
    Age is a confounder. Naive conditioning gives wrong treatment effect.
    """
    bn = BayesianNetwork()
    bn.add_node('Age', [0, 1])         # 0=young, 1=old
    bn.add_node('Treatment', [0, 1])    # 0=none, 1=drug
    bn.add_node('Recovery', [0, 1])     # 0=no, 1=yes

    bn.add_edge('Age', 'Treatment')
    bn.add_edge('Age', 'Recovery')
    bn.add_edge('Treatment', 'Recovery')

    # P(Age): 60% young
    bn.set_cpt('Age', {(0,): 0.6, (1,): 0.4})
    # P(Treatment|Age): older patients get more treatment
    bn.set_cpt('Treatment', {
        (0, 0): 0.7, (0, 1): 0.3,   # young: 30% treated
        (1, 0): 0.3, (1, 1): 0.7,   # old: 70% treated
    })
    # P(Recovery|Age, Treatment): drug helps, but young recover more
    bn.set_cpt('Recovery', {
        (0, 0, 0): 0.2, (0, 0, 1): 0.8,   # young, no drug: 80% recovery
        (0, 1, 0): 0.1, (0, 1, 1): 0.9,   # young, drug: 90% recovery
        (1, 0, 0): 0.6, (1, 0, 1): 0.4,   # old, no drug: 40% recovery
        (1, 1, 0): 0.3, (1, 1, 1): 0.7,   # old, drug: 70% recovery
    })

    model = CausalModel(bn)
    arms = [
        Arm([], name="observe"),
        Arm([Intervention('Treatment', 0)], name="do(Treatment=0)"),
        Arm([Intervention('Treatment', 1)], name="do(Treatment=1)"),
    ]

    return CausalBanditEnv(model, 'Recovery', arms=arms,
                           reward_mapping={0: 0.0, 1: 1.0}, seed=seed)


def build_advertising_env(seed: int | None = None) -> CausalBanditEnv:
    """Advertising channel selection.

    Graph: Budget -> AdChannel, UserType -> AdChannel, UserType -> Purchase,
           AdChannel -> Purchase
    Confounding: UserType affects both which ads are shown and purchase behavior.
    """
    bn = BayesianNetwork()
    bn.add_node('UserType', [0, 1, 2])   # casual, engaged, power
    bn.add_node('AdChannel', [0, 1, 2])  # email, social, search
    bn.add_node('Purchase', [0, 1])       # no, yes

    bn.add_edge('UserType', 'AdChannel')
    bn.add_edge('UserType', 'Purchase')
    bn.add_edge('AdChannel', 'Purchase')

    # P(UserType)
    bn.set_cpt('UserType', {(0,): 0.5, (1,): 0.3, (2,): 0.2})

    # P(AdChannel|UserType)
    bn.set_cpt('AdChannel', {
        (0, 0): 0.6, (0, 1): 0.3, (0, 2): 0.1,  # casual -> mostly email
        (1, 0): 0.2, (1, 1): 0.5, (1, 2): 0.3,  # engaged -> mostly social
        (2, 0): 0.1, (2, 1): 0.2, (2, 2): 0.7,  # power -> mostly search
    })

    # P(Purchase|UserType, AdChannel): search is best, power users buy more
    bn.set_cpt('Purchase', {
        (0, 0, 0): 0.95, (0, 0, 1): 0.05,  # casual + email
        (0, 1, 0): 0.90, (0, 1, 1): 0.10,  # casual + social
        (0, 2, 0): 0.85, (0, 2, 1): 0.15,  # casual + search
        (1, 0, 0): 0.80, (1, 0, 1): 0.20,  # engaged + email
        (1, 1, 0): 0.70, (1, 1, 1): 0.30,  # engaged + social
        (1, 2, 0): 0.60, (1, 2, 1): 0.40,  # engaged + search
        (2, 0, 0): 0.60, (2, 0, 1): 0.40,  # power + email
        (2, 1, 0): 0.50, (2, 1, 1): 0.50,  # power + social
        (2, 2, 0): 0.35, (2, 2, 1): 0.65,  # power + search
    })

    model = CausalModel(bn)
    arms = [
        Arm([], name="observe"),
        Arm([Intervention('AdChannel', 0)], name="do(AdChannel=email)"),
        Arm([Intervention('AdChannel', 1)], name="do(AdChannel=social)"),
        Arm([Intervention('AdChannel', 2)], name="do(AdChannel=search)"),
    ]

    return CausalBanditEnv(model, 'Purchase', arms=arms,
                           reward_mapping={0: 0.0, 1: 1.0}, seed=seed)


def build_simple_env(seed: int | None = None) -> CausalBanditEnv:
    """Simple 2-variable environment (no confounding) for testing.

    X -> Y. Direct causal effect, no confounders.
    """
    bn = BayesianNetwork()
    bn.add_node('X', [0, 1])
    bn.add_node('Y', [0, 1])
    bn.add_edge('X', 'Y')

    bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('Y', {
        (0, 0): 0.8, (0, 1): 0.2,  # X=0 -> Y=1 with 20%
        (1, 0): 0.3, (1, 1): 0.7,  # X=1 -> Y=1 with 70%
    })

    model = CausalModel(bn)
    arms = [
        Arm([], name="observe"),
        Arm([Intervention('X', 0)], name="do(X=0)"),
        Arm([Intervention('X', 1)], name="do(X=1)"),
    ]
    return CausalBanditEnv(model, 'Y', arms=arms,
                           reward_mapping={0: 0.0, 1: 1.0}, seed=seed)


def build_multi_intervention_env(seed: int | None = None) -> CausalBanditEnv:
    """Environment with multiple intervention targets.

    A -> C, B -> C. Two independent causes of the reward.
    Agent can intervene on A, B, or both.
    """
    bn = BayesianNetwork()
    bn.add_node('A', [0, 1])
    bn.add_node('B', [0, 1])
    bn.add_node('C', [0, 1])

    bn.add_edge('A', 'C')
    bn.add_edge('B', 'C')

    bn.set_cpt('A', {(0,): 0.5, (1,): 0.5})
    bn.set_cpt('B', {(0,): 0.5, (1,): 0.5})
    # C is more likely when both A and B are 1
    bn.set_cpt('C', {
        (0, 0, 0): 0.9, (0, 0, 1): 0.1,   # A=0, B=0
        (0, 1, 0): 0.5, (0, 1, 1): 0.5,   # A=0, B=1
        (1, 0, 0): 0.4, (1, 0, 1): 0.6,   # A=1, B=0
        (1, 1, 0): 0.1, (1, 1, 1): 0.9,   # A=1, B=1
    })

    model = CausalModel(bn)
    arms = [
        Arm([], name="observe"),
        Arm([Intervention('A', 0)], name="do(A=0)"),
        Arm([Intervention('A', 1)], name="do(A=1)"),
        Arm([Intervention('B', 0)], name="do(B=0)"),
        Arm([Intervention('B', 1)], name="do(B=1)"),
        Arm([Intervention('A', 1), Intervention('B', 1)], name="do(A=1,B=1)"),
    ]
    return CausalBanditEnv(model, 'C', arms=arms,
                           reward_mapping={0: 0.0, 1: 1.0}, seed=seed)
