"""V231: Causal Bandit -- Adaptive Intervention Selection for Causal Discovery.

Composes V228 (Causal Discovery from Interventions) + V225 (Causal RL).

The core idea: choosing which variable to intervene on in causal discovery
is a bandit problem. Each intervention target is an "arm", and the "reward"
is the information gained about the causal structure (edges oriented).

Strategies:
- UCB (Upper Confidence Bound): balance mean info gain + exploration bonus
- Thompson Sampling: sample from posterior over info gains, pick best
- Information-Directed Sampling: minimize cost per bit of information
- Causal UCB: UCB with causal structure-aware bonus terms
- Cost-Sensitive: incorporate variable intervention costs
- Batch: select multiple interventions per round

This bridges the gap between V228's greedy intervention selection and
principled exploration-exploitation tradeoffs from bandit theory.
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V228_causal_discovery_interventions'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V225_causal_reinforcement_learning'))

from causal_discovery_interventions import (
    CPDAG, InterventionalDataset, ActiveDiscoveryResult, InterventionResult,
    pc_result_to_cpdag, dag_to_cpdag, orient_edges_from_intervention,
    simulate_intervention, select_intervention, plan_interventions,
    minimum_intervention_set, discovery_summary,
    intervention_score_edge_count, intervention_score_entropy,
    intervention_score_separator, _apply_meek_rules,
)

# We need BayesianNetwork and PC from the causal discovery chain
v228_dir = os.path.join(os.path.dirname(__file__), '..', 'V228_causal_discovery_interventions')
sys.path.insert(0, v228_dir)

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class BanditStrategy(Enum):
    UCB = "ucb"
    THOMPSON = "thompson"
    IDS = "ids"  # Information-Directed Sampling
    CAUSAL_UCB = "causal_ucb"
    COST_SENSITIVE = "cost_sensitive"
    EPSILON_GREEDY = "epsilon_greedy"
    RANDOM = "random"


@dataclass
class ArmStats:
    """Statistics for a single bandit arm (intervention target)."""
    name: str
    pulls: int = 0
    total_reward: float = 0.0
    reward_sq_sum: float = 0.0
    rewards: list = field(default_factory=list)
    edges_oriented_history: list = field(default_factory=list)
    cost: float = 1.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls

    @property
    def reward_variance(self) -> float:
        if self.pulls < 2:
            return 1.0  # high uncertainty prior
        mean = self.mean_reward
        return self.reward_sq_sum / self.pulls - mean * mean

    def update(self, reward: float, edges_oriented: int = 0):
        self.pulls += 1
        self.total_reward += reward
        self.reward_sq_sum += reward * reward
        self.rewards.append(reward)
        self.edges_oriented_history.append(edges_oriented)


@dataclass
class BanditRound:
    """Record of a single bandit round."""
    round_num: int
    arm_selected: str
    reward: float
    edges_oriented: int
    cumulative_edges: int
    cpdag_undirected_remaining: int
    strategy_info: dict = field(default_factory=dict)


@dataclass
class CausalBanditResult:
    """Complete result of a causal bandit run."""
    final_cpdag: CPDAG
    rounds: list  # list[BanditRound]
    arm_stats: dict  # name -> ArmStats
    total_edges_oriented: int
    total_rounds: int
    total_cost: float
    strategy: str
    fully_oriented: bool
    regret_history: list = field(default_factory=list)  # per-round regret
    cumulative_regret: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def edge_count_reward(edges_oriented: int, cpdag_before: CPDAG,
                      cpdag_after: CPDAG) -> float:
    """Reward = number of edges oriented by this intervention."""
    return float(edges_oriented)


def normalized_edge_reward(edges_oriented: int, cpdag_before: CPDAG,
                           cpdag_after: CPDAG) -> float:
    """Reward = fraction of remaining undirected edges that were oriented."""
    remaining = cpdag_before.num_undirected()
    if remaining == 0:
        return 0.0
    return edges_oriented / remaining


def information_gain_reward(edges_oriented: int, cpdag_before: CPDAG,
                            cpdag_after: CPDAG) -> float:
    """Reward based on reduction in graph entropy (undirected edge count).

    Uses log ratio: log(remaining_before / remaining_after) to model
    information gain in bits about the true DAG.
    """
    before = cpdag_before.num_undirected()
    after = cpdag_after.num_undirected()
    if before == 0:
        return 0.0
    if after == 0:
        return math.log2(before + 1)
    return math.log2((before + 1) / (after + 1))


def cost_adjusted_reward(edges_oriented: int, cpdag_before: CPDAG,
                         cpdag_after: CPDAG, cost: float = 1.0) -> float:
    """Reward = edges oriented / cost of intervention."""
    if cost <= 0:
        cost = 1.0
    return float(edges_oriented) / cost


# ---------------------------------------------------------------------------
# Arm selection strategies
# ---------------------------------------------------------------------------

def ucb_select(arms: dict, t: int, cpdag: CPDAG,
               c: float = 2.0, **kwargs) -> str:
    """UCB1 selection: mean_reward + c * sqrt(ln(t) / n_i)."""
    candidates = [name for name in arms if name in _active_arms(arms, cpdag)]
    if not candidates:
        return None

    best_arm = None
    best_score = -float('inf')

    for name in candidates:
        arm = arms[name]
        if arm.pulls == 0:
            return name  # explore unvisited arms first

        exploration = c * math.sqrt(math.log(t + 1) / arm.pulls)
        score = arm.mean_reward + exploration
        if score > best_score:
            best_score = score
            best_arm = name

    return best_arm


def thompson_select(arms: dict, t: int, cpdag: CPDAG,
                    rng: np.random.Generator = None, **kwargs) -> str:
    """Thompson Sampling: sample from posterior, pick arm with highest sample.

    Uses Normal-Gamma posterior (conjugate for unknown mean + variance).
    Prior: mu0=0, kappa0=1, alpha0=1, beta0=1
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = _active_arms(arms, cpdag)
    if not candidates:
        return None

    best_arm = None
    best_sample = -float('inf')

    for name in candidates:
        arm = arms[name]
        # Normal-Gamma posterior parameters
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0

        if arm.pulls == 0:
            # Sample from prior
            tau = rng.gamma(alpha0, 1.0 / beta0)
            sample = rng.normal(mu0, 1.0 / math.sqrt(kappa0 * max(tau, 1e-10)))
        else:
            n = arm.pulls
            x_bar = arm.mean_reward
            # Update posterior
            kappa_n = kappa0 + n
            mu_n = (kappa0 * mu0 + n * x_bar) / kappa_n
            alpha_n = alpha0 + n / 2.0
            ss = sum((r - x_bar) ** 2 for r in arm.rewards)
            beta_n = beta0 + 0.5 * ss + (kappa0 * n * (x_bar - mu0) ** 2) / (2.0 * kappa_n)

            tau = rng.gamma(alpha_n, 1.0 / max(beta_n, 1e-10))
            sample = rng.normal(mu_n, 1.0 / math.sqrt(kappa_n * max(tau, 1e-10)))

        if sample > best_sample:
            best_sample = sample
            best_arm = name

    return best_arm


def ids_select(arms: dict, t: int, cpdag: CPDAG,
               rng: np.random.Generator = None, **kwargs) -> str:
    """Information-Directed Sampling: minimize (expected regret)^2 / info gain.

    Approximation: use variance as proxy for info gain, mean gap as regret.
    """
    candidates = _active_arms(arms, cpdag)
    if not candidates:
        return None

    # Need at least 1 pull each for estimates
    for name in candidates:
        if arms[name].pulls == 0:
            return name

    # Estimate best mean
    means = {name: arms[name].mean_reward for name in candidates}
    best_mean = max(means.values())

    best_arm = None
    best_ratio = float('inf')

    for name in candidates:
        arm = arms[name]
        delta = max(best_mean - arm.mean_reward, 1e-6)  # expected regret
        info = arm.reward_variance + 1e-6  # info gain proxy (variance)
        ratio = (delta ** 2) / info
        if ratio < best_ratio:
            best_ratio = ratio
            best_arm = name

    return best_arm


def causal_ucb_select(arms: dict, t: int, cpdag: CPDAG,
                      c: float = 2.0, causal_weight: float = 1.0,
                      obs_data: list = None, **kwargs) -> str:
    """Causal UCB: UCB + bonus from causal structure awareness.

    Adds a structure-based bonus proportional to the number of undirected
    edges adjacent to the intervention target (more edges = more potential info).
    """
    candidates = _active_arms(arms, cpdag)
    if not candidates:
        return None

    # Compute structural scores
    max_edges = max(
        len(cpdag.undirected_neighbors(name)) for name in candidates
    ) if candidates else 1
    max_edges = max(max_edges, 1)

    best_arm = None
    best_score = -float('inf')

    for name in candidates:
        arm = arms[name]
        if arm.pulls == 0:
            return name

        # Standard UCB
        exploration = c * math.sqrt(math.log(t + 1) / arm.pulls)
        ucb = arm.mean_reward + exploration

        # Causal structure bonus: normalized undirected degree
        struct_bonus = len(cpdag.undirected_neighbors(name)) / max_edges
        score = ucb + causal_weight * struct_bonus

        if score > best_score:
            best_score = score
            best_arm = name

    return best_arm


def cost_sensitive_select(arms: dict, t: int, cpdag: CPDAG,
                          c: float = 2.0, **kwargs) -> str:
    """Cost-sensitive UCB: (mean_reward / cost) + exploration bonus."""
    candidates = _active_arms(arms, cpdag)
    if not candidates:
        return None

    best_arm = None
    best_score = -float('inf')

    for name in candidates:
        arm = arms[name]
        if arm.pulls == 0:
            return name

        cost = max(arm.cost, 1e-6)
        exploration = c * math.sqrt(math.log(t + 1) / arm.pulls)
        score = (arm.mean_reward / cost) + exploration / math.sqrt(cost)

        if score > best_score:
            best_score = score
            best_arm = name

    return best_arm


def epsilon_greedy_select(arms: dict, t: int, cpdag: CPDAG,
                          epsilon: float = 0.1,
                          rng: np.random.Generator = None, **kwargs) -> str:
    """Epsilon-greedy: explore uniformly with prob epsilon, else exploit."""
    if rng is None:
        rng = np.random.default_rng()

    candidates = list(_active_arms(arms, cpdag))
    if not candidates:
        return None

    # Explore unvisited
    unvisited = [name for name in candidates if arms[name].pulls == 0]
    if unvisited:
        return rng.choice(unvisited)

    if rng.random() < epsilon:
        return rng.choice(candidates)
    else:
        return max(candidates, key=lambda n: arms[n].mean_reward)


def random_select(arms: dict, t: int, cpdag: CPDAG,
                  rng: np.random.Generator = None, **kwargs) -> str:
    """Uniform random selection."""
    if rng is None:
        rng = np.random.default_rng()

    candidates = list(_active_arms(arms, cpdag))
    if not candidates:
        return None
    return rng.choice(candidates)


def _active_arms(arms: dict, cpdag: CPDAG) -> list:
    """Return arms that can still orient edges (have undirected neighbors)."""
    active = []
    for name in arms:
        if name in cpdag.variables:
            # Arm is active if it has undirected neighbors or hasn't been tried
            if cpdag.undirected_neighbors(name) or arms[name].pulls == 0:
                active.append(name)
    # If no arms have undirected neighbors, return empty
    if all(len(cpdag.undirected_neighbors(name)) == 0 for name in active):
        return []
    # Filter to only those with undirected neighbors
    return [name for name in active if cpdag.undirected_neighbors(name)]


STRATEGY_MAP = {
    BanditStrategy.UCB: ucb_select,
    BanditStrategy.THOMPSON: thompson_select,
    BanditStrategy.IDS: ids_select,
    BanditStrategy.CAUSAL_UCB: causal_ucb_select,
    BanditStrategy.COST_SENSITIVE: cost_sensitive_select,
    BanditStrategy.EPSILON_GREEDY: epsilon_greedy_select,
    BanditStrategy.RANDOM: random_select,
}


# ---------------------------------------------------------------------------
# Core: CausalBandit
# ---------------------------------------------------------------------------

class CausalBandit:
    """Bandit-based adaptive intervention selection for causal discovery.

    Treats each possible intervention target as a bandit arm.
    Selects interventions to maximize information about causal structure
    while balancing exploration (try new targets) vs exploitation
    (revisit informative targets).
    """

    def __init__(self, variables: list, strategy: BanditStrategy = BanditStrategy.UCB,
                 reward_fn: Callable = None, costs: dict = None,
                 strategy_params: dict = None, rng: np.random.Generator = None):
        self.variables = list(variables)
        self.strategy = strategy
        self.reward_fn = reward_fn or edge_count_reward
        self.strategy_params = strategy_params or {}
        self.rng = rng or np.random.default_rng(42)

        # Initialize arms
        self.arms = {}
        for v in variables:
            cost = costs.get(v, 1.0) if costs else 1.0
            self.arms[v] = ArmStats(name=v, cost=cost)

        self.rounds = []
        self.total_edges_oriented = 0
        self.total_cost = 0.0

    def select_arm(self, cpdag: CPDAG, t: int, obs_data: list = None) -> str:
        """Select which variable to intervene on next."""
        select_fn = STRATEGY_MAP[self.strategy]
        params = dict(self.strategy_params)
        params['rng'] = self.rng
        params['obs_data'] = obs_data
        return select_fn(self.arms, t, cpdag, **params)

    def update(self, arm_name: str, edges_oriented: int,
               cpdag_before: CPDAG, cpdag_after: CPDAG, round_num: int):
        """Update arm statistics after an intervention."""
        reward = self.reward_fn(edges_oriented, cpdag_before, cpdag_after)

        self.arms[arm_name].update(reward, edges_oriented)
        self.total_edges_oriented += edges_oriented
        self.total_cost += self.arms[arm_name].cost

        round_record = BanditRound(
            round_num=round_num,
            arm_selected=arm_name,
            reward=reward,
            edges_oriented=edges_oriented,
            cumulative_edges=self.total_edges_oriented,
            cpdag_undirected_remaining=cpdag_after.num_undirected(),
            strategy_info={'mean_reward': self.arms[arm_name].mean_reward,
                           'pulls': self.arms[arm_name].pulls},
        )
        self.rounds.append(round_record)

    def result(self, cpdag: CPDAG) -> CausalBanditResult:
        """Package current state as result."""
        return CausalBanditResult(
            final_cpdag=cpdag,
            rounds=self.rounds,
            arm_stats=dict(self.arms),
            total_edges_oriented=self.total_edges_oriented,
            total_rounds=len(self.rounds),
            total_cost=self.total_cost,
            strategy=self.strategy.value,
            fully_oriented=cpdag.is_fully_oriented(),
        )


# ---------------------------------------------------------------------------
# Main discovery loops
# ---------------------------------------------------------------------------

def causal_bandit_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    strategy: BanditStrategy = BanditStrategy.UCB,
    max_rounds: int = 20,
    samples_per_intervention: int = 100,
    reward_fn: Callable = None,
    costs: dict = None,
    strategy_params: dict = None,
    alpha: float = 0.05,
    ground_truth: set = None,
    rng: np.random.Generator = None,
) -> CausalBanditResult:
    """Run causal bandit discovery: iteratively select interventions via bandit.

    Args:
        initial_data: Observational data samples (list of dicts)
        variables: List of variable names
        intervention_fn: Callable(target, value) -> list[dict] of samples
        strategy: Bandit strategy for arm selection
        max_rounds: Maximum number of intervention rounds
        samples_per_intervention: Samples per intervention call
        reward_fn: Reward function (edges_oriented, cpdag_before, cpdag_after) -> float
        costs: Dict mapping variable -> intervention cost
        strategy_params: Extra params for strategy function
        alpha: Significance level for independence tests
        ground_truth: True DAG edges for regret computation
        rng: Random number generator

    Returns:
        CausalBanditResult with final CPDAG, round history, arm stats
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build initial CPDAG from observational data via PC algorithm
    cpdag = _build_initial_cpdag(initial_data, variables, alpha)

    bandit = CausalBandit(
        variables=variables,
        strategy=strategy,
        reward_fn=reward_fn,
        costs=costs,
        strategy_params=strategy_params,
        rng=rng,
    )

    obs_data = list(initial_data)

    # If we have ground truth, compute oracle rewards for regret
    oracle_rewards = None
    if ground_truth is not None:
        oracle_rewards = _compute_oracle_rewards(
            cpdag, variables, intervention_fn, obs_data, alpha, ground_truth
        )

    for t in range(max_rounds):
        if cpdag.is_fully_oriented():
            break

        # Select arm
        arm = bandit.select_arm(cpdag, t + 1, obs_data)
        if arm is None:
            break

        # Perform intervention
        # Use domain value 1 as default intervention value
        int_data = intervention_fn(arm, 1)

        # Orient edges
        cpdag_before = cpdag.copy()
        cpdag, newly_oriented = orient_edges_from_intervention(
            cpdag, arm, int_data, obs_data, alpha
        )
        edges_oriented = len(newly_oriented)

        # Update bandit
        bandit.update(arm, edges_oriented, cpdag_before, cpdag, t)

        # Compute regret if ground truth available
        if oracle_rewards is not None:
            oracle_best = max(oracle_rewards.values()) if oracle_rewards else 0
            actual_reward = bandit.rounds[-1].reward
            bandit.rounds[-1].strategy_info['regret'] = oracle_best - actual_reward

    result = bandit.result(cpdag)

    # Compute regret history
    if oracle_rewards is not None:
        regret_hist = []
        cum_regret = 0.0
        for r in result.rounds:
            inst_regret = r.strategy_info.get('regret', 0.0)
            regret_hist.append(inst_regret)
            cum_regret += inst_regret
        result.regret_history = regret_hist
        result.cumulative_regret = _cumsum(regret_hist)

    return result


def batch_causal_bandit_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    strategy: BanditStrategy = BanditStrategy.UCB,
    max_batches: int = 5,
    batch_size: int = 3,
    samples_per_intervention: int = 100,
    reward_fn: Callable = None,
    costs: dict = None,
    strategy_params: dict = None,
    alpha: float = 0.05,
    diversity_weight: float = 0.5,
    rng: np.random.Generator = None,
) -> CausalBanditResult:
    """Batch causal bandit: select multiple interventions per round.

    Uses a diversity-aware selection to avoid redundant interventions
    within the same batch.

    Args:
        batch_size: Number of interventions per batch
        diversity_weight: Weight for diversity bonus (0=pure bandit, 1=max diversity)
        Other args same as causal_bandit_discovery
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cpdag = _build_initial_cpdag(initial_data, variables, alpha)

    bandit = CausalBandit(
        variables=variables,
        strategy=strategy,
        reward_fn=reward_fn,
        costs=costs,
        strategy_params=strategy_params,
        rng=rng,
    )

    obs_data = list(initial_data)
    round_counter = 0

    for batch_idx in range(max_batches):
        if cpdag.is_fully_oriented():
            break

        # Select batch of arms with diversity
        batch = _select_diverse_batch(
            bandit, cpdag, batch_size, round_counter + 1,
            obs_data, diversity_weight
        )

        if not batch:
            break

        # Execute all interventions in batch
        for arm in batch:
            if cpdag.is_fully_oriented():
                break

            int_data = intervention_fn(arm, 1)
            cpdag_before = cpdag.copy()
            cpdag, newly_oriented = orient_edges_from_intervention(
                cpdag, arm, int_data, obs_data, alpha
            )
            edges_oriented = len(newly_oriented)

            bandit.update(arm, edges_oriented, cpdag_before, cpdag, round_counter)
            round_counter += 1

    return bandit.result(cpdag)


def _select_diverse_batch(bandit: CausalBandit, cpdag: CPDAG,
                          batch_size: int, t: int,
                          obs_data: list, diversity_weight: float) -> list:
    """Select a diverse batch of arms.

    Uses iterative selection: after picking each arm, reduce scores for
    arms that are graph-neighbors (to avoid redundant interventions).
    """
    candidates = list(_active_arms(bandit.arms, cpdag))
    if not candidates:
        return []

    selected = []
    used = set()

    for i in range(min(batch_size, len(candidates))):
        remaining = [c for c in candidates if c not in used]
        if not remaining:
            break

        # Score each remaining arm
        scores = {}
        for name in remaining:
            # Base score from bandit strategy
            base = _arm_score(bandit, name, cpdag, t + i)

            # Diversity penalty: reduce score if neighbors already selected
            penalty = 0.0
            for sel in selected:
                if sel in cpdag.neighbors(name):
                    penalty += diversity_weight

            scores[name] = base - penalty

        best = max(scores, key=scores.get)
        selected.append(best)
        used.add(best)

    return selected


def _arm_score(bandit: CausalBandit, name: str, cpdag: CPDAG, t: int) -> float:
    """Compute a score for an arm (higher = more desirable)."""
    arm = bandit.arms[name]
    if arm.pulls == 0:
        return 10.0  # high priority for unexplored

    c = bandit.strategy_params.get('c', 2.0)
    exploration = c * math.sqrt(math.log(t + 1) / arm.pulls)
    return arm.mean_reward + exploration


# ---------------------------------------------------------------------------
# Regret analysis
# ---------------------------------------------------------------------------

def compute_regret(result: CausalBanditResult,
                   oracle_result: CausalBanditResult = None) -> dict:
    """Compute various regret metrics for a causal bandit run.

    Returns dict with:
        - cumulative_regret: total regret over all rounds
        - per_round_regret: list of per-round regret values
        - average_regret: mean per-round regret
        - efficiency: edges oriented per unit cost
    """
    if result.total_rounds == 0:
        return {
            'cumulative_regret': 0.0,
            'per_round_regret': [],
            'average_regret': 0.0,
            'efficiency': 0.0,
        }

    efficiency = result.total_edges_oriented / max(result.total_cost, 1e-6)

    if result.cumulative_regret:
        cum_regret = result.cumulative_regret[-1] if result.cumulative_regret else 0.0
        avg_regret = cum_regret / result.total_rounds
    else:
        cum_regret = 0.0
        avg_regret = 0.0

    return {
        'cumulative_regret': cum_regret,
        'per_round_regret': result.regret_history,
        'average_regret': avg_regret,
        'efficiency': efficiency,
    }


def compare_strategies(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    strategies: list = None,
    max_rounds: int = 20,
    samples_per_intervention: int = 100,
    reward_fn: Callable = None,
    alpha: float = 0.05,
    n_trials: int = 5,
    rng_seed: int = 42,
) -> dict:
    """Compare multiple bandit strategies on the same causal discovery problem.

    Returns dict mapping strategy name -> {
        mean_rounds, mean_edges, mean_cost, mean_efficiency, results: list
    }
    """
    if strategies is None:
        strategies = [
            BanditStrategy.UCB,
            BanditStrategy.THOMPSON,
            BanditStrategy.CAUSAL_UCB,
            BanditStrategy.EPSILON_GREEDY,
            BanditStrategy.RANDOM,
        ]

    comparison = {}

    for strat in strategies:
        results = []
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_seed + trial)
            result = causal_bandit_discovery(
                initial_data=initial_data,
                variables=variables,
                intervention_fn=intervention_fn,
                strategy=strat,
                max_rounds=max_rounds,
                samples_per_intervention=samples_per_intervention,
                reward_fn=reward_fn,
                alpha=alpha,
                rng=rng,
            )
            results.append(result)

        rounds = [r.total_rounds for r in results]
        edges = [r.total_edges_oriented for r in results]
        costs = [r.total_cost for r in results]
        efficiencies = [r.total_edges_oriented / max(r.total_cost, 1e-6)
                        for r in results]
        fully = sum(1 for r in results if r.fully_oriented)

        comparison[strat.value] = {
            'mean_rounds': np.mean(rounds),
            'mean_edges': np.mean(edges),
            'mean_cost': np.mean(costs),
            'mean_efficiency': np.mean(efficiencies),
            'fully_oriented_count': fully,
            'results': results,
        }

    return comparison


# ---------------------------------------------------------------------------
# Adaptive strategies
# ---------------------------------------------------------------------------

class AdaptiveCausalBandit:
    """Adaptive causal bandit that switches strategies based on performance.

    Starts with exploration-heavy strategy, transitions to exploitation
    as the CPDAG becomes more oriented.
    """

    def __init__(self, variables: list, costs: dict = None,
                 exploration_strategy: BanditStrategy = BanditStrategy.THOMPSON,
                 exploitation_strategy: BanditStrategy = BanditStrategy.CAUSAL_UCB,
                 switch_threshold: float = 0.5,
                 rng: np.random.Generator = None):
        self.variables = list(variables)
        self.exploration_strategy = exploration_strategy
        self.exploitation_strategy = exploitation_strategy
        self.switch_threshold = switch_threshold
        self.rng = rng or np.random.default_rng(42)
        self.costs = costs

        # Internal bandits
        self.explore_bandit = CausalBandit(
            variables, exploration_strategy, costs=costs, rng=self.rng
        )
        self.exploit_bandit = CausalBandit(
            variables, exploitation_strategy, costs=costs, rng=self.rng
        )
        self.current_phase = "explore"
        self.phase_history = []

    def select_arm(self, cpdag: CPDAG, t: int, obs_data: list = None) -> str:
        """Select arm, choosing strategy based on orientation progress."""
        total_edges = len(cpdag.directed) + cpdag.num_undirected()
        if total_edges == 0:
            return None
        orientation_ratio = len(cpdag.directed) / total_edges

        if orientation_ratio >= self.switch_threshold:
            self.current_phase = "exploit"
            arm = self.exploit_bandit.select_arm(cpdag, t, obs_data)
        else:
            self.current_phase = "explore"
            arm = self.explore_bandit.select_arm(cpdag, t, obs_data)

        self.phase_history.append(self.current_phase)
        return arm

    def update(self, arm_name: str, edges_oriented: int,
               cpdag_before: CPDAG, cpdag_after: CPDAG, round_num: int):
        """Update both bandits (shared arm stats)."""
        self.explore_bandit.update(arm_name, edges_oriented, cpdag_before, cpdag_after, round_num)
        self.exploit_bandit.update(arm_name, edges_oriented, cpdag_before, cpdag_after, round_num)


def adaptive_causal_bandit_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    max_rounds: int = 20,
    reward_fn: Callable = None,
    costs: dict = None,
    alpha: float = 0.05,
    switch_threshold: float = 0.5,
    rng: np.random.Generator = None,
) -> CausalBanditResult:
    """Run adaptive causal bandit that switches explore->exploit."""
    if rng is None:
        rng = np.random.default_rng(42)

    cpdag = _build_initial_cpdag(initial_data, variables, alpha)

    adaptive = AdaptiveCausalBandit(
        variables=variables, costs=costs,
        switch_threshold=switch_threshold, rng=rng,
    )

    # Use explore_bandit for result tracking
    if reward_fn:
        adaptive.explore_bandit.reward_fn = reward_fn
        adaptive.exploit_bandit.reward_fn = reward_fn

    obs_data = list(initial_data)

    for t in range(max_rounds):
        if cpdag.is_fully_oriented():
            break

        arm = adaptive.select_arm(cpdag, t + 1, obs_data)
        if arm is None:
            break

        int_data = intervention_fn(arm, 1)
        cpdag_before = cpdag.copy()
        cpdag, newly_oriented = orient_edges_from_intervention(
            cpdag, arm, int_data, obs_data, alpha
        )
        edges_oriented = len(newly_oriented)

        adaptive.update(arm, edges_oriented, cpdag_before, cpdag, t)

    result = adaptive.explore_bandit.result(cpdag)
    result.strategy = "adaptive"
    return result


# ---------------------------------------------------------------------------
# Budget-constrained discovery
# ---------------------------------------------------------------------------

def budget_constrained_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    budget: float,
    strategy: BanditStrategy = BanditStrategy.COST_SENSITIVE,
    costs: dict = None,
    reward_fn: Callable = None,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> CausalBanditResult:
    """Causal bandit with a fixed budget for intervention costs.

    Selects interventions until budget is exhausted. Cost-sensitive
    strategies are recommended to maximize edges oriented per unit cost.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if costs is None:
        costs = {v: 1.0 for v in variables}

    cpdag = _build_initial_cpdag(initial_data, variables, alpha)

    bandit = CausalBandit(
        variables=variables,
        strategy=strategy,
        reward_fn=reward_fn,
        costs=costs,
        strategy_params={'c': 2.0},
        rng=rng,
    )

    obs_data = list(initial_data)
    remaining_budget = budget
    t = 0

    while remaining_budget > 0 and not cpdag.is_fully_oriented():
        # Filter to affordable arms
        affordable = [v for v in _active_arms(bandit.arms, cpdag)
                       if costs.get(v, 1.0) <= remaining_budget]
        if not affordable:
            break

        arm = bandit.select_arm(cpdag, t + 1, obs_data)
        if arm is None or costs.get(arm, 1.0) > remaining_budget:
            # Try to find any affordable arm
            arm = min(affordable, key=lambda v: costs.get(v, 1.0))

        int_data = intervention_fn(arm, 1)
        cpdag_before = cpdag.copy()
        cpdag, newly_oriented = orient_edges_from_intervention(
            cpdag, arm, int_data, obs_data, alpha
        )
        edges_oriented = len(newly_oriented)

        bandit.update(arm, edges_oriented, cpdag_before, cpdag, t)
        remaining_budget -= costs.get(arm, 1.0)
        t += 1

    result = bandit.result(cpdag)
    return result


# ---------------------------------------------------------------------------
# Contextual causal bandit
# ---------------------------------------------------------------------------

class ContextualCausalBandit:
    """Contextual bandit where the context is the current CPDAG state.

    Features extracted from the CPDAG (e.g., undirected degree, neighbor
    structure) inform arm selection. This allows the bandit to generalize
    across different graph states rather than treating each round independently.
    """

    def __init__(self, variables: list, costs: dict = None,
                 rng: np.random.Generator = None):
        self.variables = list(variables)
        self.costs = costs or {v: 1.0 for v in variables}
        self.rng = rng or np.random.default_rng(42)

        # Linear model weights per arm: w^T * context -> expected reward
        self.feature_dim = 5  # undirected_degree, directed_in, directed_out, pulls, mean_reward
        self.weights = {v: np.zeros(self.feature_dim) for v in variables}
        self.covariance = {v: np.eye(self.feature_dim) for v in variables}
        self.reward_vec = {v: np.zeros(self.feature_dim) for v in variables}
        self.arms = {v: ArmStats(name=v, cost=self.costs.get(v, 1.0))
                     for v in variables}
        self.rounds = []
        self.total_edges_oriented = 0
        self.total_cost = 0.0

    def _extract_features(self, cpdag: CPDAG, arm_name: str) -> np.ndarray:
        """Extract context features for an arm given current CPDAG."""
        undirected_deg = len(cpdag.undirected_neighbors(arm_name))
        directed_in = len(cpdag.parents(arm_name))
        directed_out = len(cpdag.children(arm_name))
        pulls = self.arms[arm_name].pulls
        mean_r = self.arms[arm_name].mean_reward
        return np.array([undirected_deg, directed_in, directed_out,
                         pulls, mean_r], dtype=float)

    def select_arm(self, cpdag: CPDAG, t: int, obs_data: list = None) -> str:
        """LinUCB selection: argmax w^T x + alpha * sqrt(x^T A^{-1} x)."""
        candidates = list(_active_arms(self.arms, cpdag))
        if not candidates:
            return None

        alpha_param = 1.0
        best_arm = None
        best_score = -float('inf')

        for name in candidates:
            x = self._extract_features(cpdag, name)
            A_inv = np.linalg.solve(self.covariance[name], np.eye(self.feature_dim))
            theta = A_inv @ self.reward_vec[name]
            ucb = theta @ x + alpha_param * math.sqrt(x @ A_inv @ x)

            if ucb > best_score:
                best_score = ucb
                best_arm = name

        return best_arm

    def update(self, arm_name: str, reward: float, features: np.ndarray,
               edges_oriented: int, cpdag_before: CPDAG, cpdag_after: CPDAG,
               round_num: int):
        """Update LinUCB model for the selected arm."""
        self.covariance[arm_name] += np.outer(features, features)
        self.reward_vec[arm_name] += reward * features
        self.arms[arm_name].update(reward, edges_oriented)
        self.total_edges_oriented += edges_oriented
        self.total_cost += self.arms[arm_name].cost

        self.rounds.append(BanditRound(
            round_num=round_num,
            arm_selected=arm_name,
            reward=reward,
            edges_oriented=edges_oriented,
            cumulative_edges=self.total_edges_oriented,
            cpdag_undirected_remaining=cpdag_after.num_undirected(),
        ))

    def result(self, cpdag: CPDAG) -> CausalBanditResult:
        return CausalBanditResult(
            final_cpdag=cpdag,
            rounds=self.rounds,
            arm_stats=dict(self.arms),
            total_edges_oriented=self.total_edges_oriented,
            total_rounds=len(self.rounds),
            total_cost=self.total_cost,
            strategy="contextual_linucb",
            fully_oriented=cpdag.is_fully_oriented(),
        )


def contextual_causal_bandit_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    max_rounds: int = 20,
    costs: dict = None,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> CausalBanditResult:
    """Run contextual (LinUCB) causal bandit discovery."""
    if rng is None:
        rng = np.random.default_rng(42)

    cpdag = _build_initial_cpdag(initial_data, variables, alpha)
    obs_data = list(initial_data)

    bandit = ContextualCausalBandit(variables=variables, costs=costs, rng=rng)

    for t in range(max_rounds):
        if cpdag.is_fully_oriented():
            break

        arm = bandit.select_arm(cpdag, t + 1, obs_data)
        if arm is None:
            break

        features = bandit._extract_features(cpdag, arm)

        int_data = intervention_fn(arm, 1)
        cpdag_before = cpdag.copy()
        cpdag, newly_oriented = orient_edges_from_intervention(
            cpdag, arm, int_data, obs_data, alpha
        )
        edges_oriented = len(newly_oriented)

        reward = float(edges_oriented)
        bandit.update(arm, reward, features, edges_oriented, cpdag_before, cpdag, t)

    return bandit.result(cpdag)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_initial_cpdag(data: list, variables: list, alpha: float) -> CPDAG:
    """Build CPDAG from observational data using PC algorithm.

    Simplified: run pairwise independence tests to find edges,
    then identify v-structures and apply Meek rules.
    """
    n = len(data)
    # Build adjacency from pairwise correlations
    cpdag = CPDAG(variables=variables)

    # Pairwise marginal independence test (chi-squared)
    for i, v1 in enumerate(variables):
        for v2 in variables[i + 1:]:
            # Extract values
            vals1 = [s.get(v1) for s in data if v1 in s]
            vals2 = [s.get(v2) for s in data if v2 in s]
            if not vals1 or not vals2:
                continue

            min_len = min(len(vals1), len(vals2))
            vals1 = vals1[:min_len]
            vals2 = vals2[:min_len]

            # Chi-squared independence test
            if not _is_independent(vals1, vals2, alpha):
                cpdag.undirected.add(frozenset([v1, v2]))

    # Identify v-structures: for each X - Z - Y where X and Z not adjacent,
    # orient as X -> Z <- Y
    _orient_v_structures(cpdag, data, variables, alpha)

    # Apply Meek rules
    cpdag = _apply_meek_rules(cpdag)

    return cpdag


def _is_independent(vals1: list, vals2: list, alpha: float) -> bool:
    """Chi-squared test of independence between two discrete variables."""
    n = len(vals1)
    if n < 5:
        return True

    # Build contingency table
    domain1 = sorted(set(vals1))
    domain2 = sorted(set(vals2))

    if len(domain1) < 2 or len(domain2) < 2:
        return True

    idx1 = {v: i for i, v in enumerate(domain1)}
    idx2 = {v: i for i, v in enumerate(domain2)}

    table = np.zeros((len(domain1), len(domain2)))
    for v1, v2 in zip(vals1, vals2):
        table[idx1[v1]][idx2[v2]] += 1

    # Chi-squared statistic
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    total = table.sum()

    if total == 0:
        return True

    chi2 = 0.0
    for i in range(len(domain1)):
        for j in range(len(domain2)):
            expected = row_sums[i] * col_sums[j] / total
            if expected > 0:
                chi2 += (table[i][j] - expected) ** 2 / expected

    # Degrees of freedom
    df = (len(domain1) - 1) * (len(domain2) - 1)
    if df == 0:
        return True

    # p-value approximation using chi-squared CDF
    p_value = _chi2_survival(chi2, df)
    return p_value > alpha


def _chi2_survival(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Uses regularized incomplete gamma function approximation.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0

    a = df / 2.0
    z = x / 2.0

    # Use series expansion of regularized lower incomplete gamma
    gamma_val = _regularized_gamma_lower(a, z)
    return 1.0 - gamma_val


def _regularized_gamma_lower(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a)."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < a + 1:
        # Series expansion
        return _gamma_series(a, x)
    else:
        # Continued fraction
        return 1.0 - _gamma_cf(a, x)


def _gamma_series(a: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Series expansion for lower incomplete gamma."""
    if x == 0:
        return 0.0
    ap = a
    s = 1.0 / a
    delta = s
    for _ in range(max_iter):
        ap += 1
        delta *= x / ap
        s += delta
        if abs(delta) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    """Continued fraction for upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    b = x + 1 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _orient_v_structures(cpdag: CPDAG, data: list, variables: list,
                         alpha: float):
    """Orient v-structures: X - Z - Y where X and Y not adjacent -> X -> Z <- Y."""
    for z in variables:
        undirected_nbrs = list(cpdag.undirected_neighbors(z))
        for i, x in enumerate(undirected_nbrs):
            for y in undirected_nbrs[i + 1:]:
                # Check if x and y are NOT adjacent
                if frozenset([x, y]) in cpdag.undirected or (x, y) in cpdag.directed or (y, x) in cpdag.directed:
                    continue
                # Check if z is NOT in the separating set of x and y
                # (approximate: test if x _||_ y | z)
                vals_x = [s.get(x) for s in data if x in s and z in s and y in s]
                vals_y = [s.get(y) for s in data if x in s and z in s and y in s]
                vals_z = [s.get(z) for s in data if x in s and z in s and y in s]
                if not vals_x or len(vals_x) < 5:
                    continue

                # If x NOT independent of y given z -> v-structure
                if not _conditional_independent(vals_x, vals_y, vals_z, alpha):
                    # Orient as x -> z <- y
                    cpdag.undirected.discard(frozenset([x, z]))
                    cpdag.undirected.discard(frozenset([y, z]))
                    cpdag.directed.add((x, z))
                    cpdag.directed.add((y, z))


def _conditional_independent(vals_x: list, vals_y: list, vals_z: list,
                             alpha: float) -> bool:
    """Test X _||_ Y | Z using stratified chi-squared test."""
    z_domain = sorted(set(vals_z))
    total_chi2 = 0.0
    total_df = 0

    for z_val in z_domain:
        xs = [vals_x[i] for i in range(len(vals_z)) if vals_z[i] == z_val]
        ys = [vals_y[i] for i in range(len(vals_z)) if vals_z[i] == z_val]
        if len(xs) < 5:
            continue

        x_dom = sorted(set(xs))
        y_dom = sorted(set(ys))
        if len(x_dom) < 2 or len(y_dom) < 2:
            continue

        idx_x = {v: i for i, v in enumerate(x_dom)}
        idx_y = {v: i for i, v in enumerate(y_dom)}
        table = np.zeros((len(x_dom), len(y_dom)))
        for xv, yv in zip(xs, ys):
            table[idx_x[xv]][idx_y[yv]] += 1

        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        total = table.sum()
        if total == 0:
            continue

        chi2 = 0.0
        for i in range(len(x_dom)):
            for j in range(len(y_dom)):
                expected = row_sums[i] * col_sums[j] / total
                if expected > 0:
                    chi2 += (table[i][j] - expected) ** 2 / expected

        df = (len(x_dom) - 1) * (len(y_dom) - 1)
        total_chi2 += chi2
        total_df += df

    if total_df == 0:
        return True

    p_value = _chi2_survival(total_chi2, total_df)
    return p_value > alpha


def _compute_oracle_rewards(cpdag: CPDAG, variables: list,
                            intervention_fn: Callable,
                            obs_data: list, alpha: float,
                            ground_truth: set) -> dict:
    """Compute the oracle (best possible) reward for each arm.

    Runs each intervention once to see how many edges it would orient.
    Used for regret computation.
    """
    rewards = {}
    for v in variables:
        if not cpdag.undirected_neighbors(v):
            rewards[v] = 0.0
            continue
        int_data = intervention_fn(v, 1)
        cpdag_copy = cpdag.copy()
        _, newly_oriented = orient_edges_from_intervention(
            cpdag_copy, v, int_data, obs_data, alpha
        )
        rewards[v] = float(len(newly_oriented))
    return rewards


def _cumsum(values: list) -> list:
    """Cumulative sum."""
    result = []
    total = 0.0
    for v in values:
        total += v
        result.append(total)
    return result


# ---------------------------------------------------------------------------
# Benchmark environments
# ---------------------------------------------------------------------------

def build_chain_environment(n: int = 4, domain: list = None,
                            n_obs: int = 500, seed: int = 42):
    """Chain graph benchmark: X1 -> X2 -> ... -> Xn.

    Returns (obs_data, variables, intervention_fn, true_edges).
    """
    if domain is None:
        domain = [0, 1]
    rng = np.random.default_rng(seed)
    variables = [f"X{i}" for i in range(1, n + 1)]
    true_edges = set()
    for i in range(n - 1):
        true_edges.add((variables[i], variables[i + 1]))

    # Generate observational data
    obs_data = []
    for _ in range(n_obs):
        sample = {}
        for i, v in enumerate(variables):
            if i == 0:
                sample[v] = rng.choice(domain)
            else:
                parent_val = sample[variables[i - 1]]
                # Noisy copy: 80% copy parent, 20% random
                if rng.random() < 0.8:
                    sample[v] = parent_val
                else:
                    sample[v] = rng.choice(domain)
            sample[v] = int(sample[v])
        obs_data.append(sample)

    # Intervention function
    def intervention_fn(target, value, n_samples=200):
        return _simulate_chain_intervention(
            target, value, variables, domain, n_samples, rng
        )

    return obs_data, variables, intervention_fn, true_edges


def _simulate_chain_intervention(target, value, variables, domain,
                                 n_samples, rng):
    """Simulate do(target=value) on a chain graph."""
    target_idx = variables.index(target)
    samples = []
    for _ in range(n_samples):
        sample = {}
        for i, v in enumerate(variables):
            if v == target:
                sample[v] = int(value)
            elif i == 0:
                sample[v] = int(rng.choice(domain))
            elif i <= target_idx:
                # Before target: normal mechanism
                parent_val = sample[variables[i - 1]]
                sample[v] = int(parent_val if rng.random() < 0.8 else rng.choice(domain))
            else:
                # After target: normal mechanism
                parent_val = sample[variables[i - 1]]
                sample[v] = int(parent_val if rng.random() < 0.8 else rng.choice(domain))
        samples.append(sample)
    return samples


def build_diamond_environment(domain: list = None, n_obs: int = 500, seed: int = 42):
    """Diamond graph: X -> A, X -> B, A -> Y, B -> Y.

    Returns (obs_data, variables, intervention_fn, true_edges).
    """
    if domain is None:
        domain = [0, 1]
    rng = np.random.default_rng(seed)
    variables = ["X", "A", "B", "Y"]
    true_edges = {("X", "A"), ("X", "B"), ("A", "Y"), ("B", "Y")}

    obs_data = []
    for _ in range(n_obs):
        x = int(rng.choice(domain))
        a = int(x if rng.random() < 0.8 else rng.choice(domain))
        b = int(x if rng.random() < 0.7 else rng.choice(domain))
        y = int((a or b) if rng.random() < 0.8 else rng.choice(domain))
        obs_data.append({"X": x, "A": a, "B": b, "Y": y})

    def intervention_fn(target, value, n_samples=200):
        samples = []
        for _ in range(n_samples):
            x = int(rng.choice(domain))
            if target == "X":
                x = int(value)
            a = int(x if rng.random() < 0.8 else rng.choice(domain))
            if target == "A":
                a = int(value)
            b = int(x if rng.random() < 0.7 else rng.choice(domain))
            if target == "B":
                b = int(value)
            y = int((a or b) if rng.random() < 0.8 else rng.choice(domain))
            if target == "Y":
                y = int(value)
            samples.append({"X": x, "A": a, "B": b, "Y": y})
        return samples

    return obs_data, variables, intervention_fn, true_edges


def build_confounded_environment(domain: list = None, n_obs: int = 500,
                                  seed: int = 42):
    """Confounded graph: U -> X, U -> Y, X -> Y (U unobserved).

    X and Y are correlated both through X->Y and the confounder U.
    Intervention on X breaks the U->X path, revealing the true causal effect.

    Returns (obs_data, variables, intervention_fn, true_edges).
    """
    if domain is None:
        domain = [0, 1]
    rng = np.random.default_rng(seed)
    variables = ["X", "Y"]
    true_edges = {("X", "Y")}

    obs_data = []
    for _ in range(n_obs):
        u = int(rng.choice(domain))  # unobserved confounder
        x = int(u if rng.random() < 0.7 else rng.choice(domain))
        y = int((x or u) if rng.random() < 0.8 else rng.choice(domain))
        obs_data.append({"X": x, "Y": y})

    def intervention_fn(target, value, n_samples=200):
        samples = []
        for _ in range(n_samples):
            u = int(rng.choice(domain))
            if target == "X":
                x = int(value)
            else:
                x = int(u if rng.random() < 0.7 else rng.choice(domain))
            if target == "Y":
                y = int(value)
            else:
                y = int((x or u) if rng.random() < 0.8 else rng.choice(domain))
            samples.append({"X": x, "Y": y})
        return samples

    return obs_data, variables, intervention_fn, true_edges


def build_large_environment(n_vars: int = 8, edge_prob: float = 0.3,
                            domain: list = None, n_obs: int = 1000,
                            seed: int = 42):
    """Random DAG with n_vars variables.

    Returns (obs_data, variables, intervention_fn, true_edges).
    """
    if domain is None:
        domain = [0, 1]
    rng = np.random.default_rng(seed)
    variables = [f"V{i}" for i in range(n_vars)]

    # Random DAG: for each pair (i < j), add edge i->j with probability edge_prob
    true_edges = set()
    parents = {v: [] for v in variables}
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.random() < edge_prob:
                true_edges.add((variables[i], variables[j]))
                parents[variables[j]].append(variables[i])

    # Generate data using noisy-OR model
    obs_data = []
    for _ in range(n_obs):
        sample = {}
        for v in variables:
            if not parents[v]:
                sample[v] = int(rng.choice(domain))
            else:
                # Noisy-OR: 1 if any parent is 1 (with noise)
                parent_active = any(sample[p] == 1 for p in parents[v])
                if parent_active:
                    sample[v] = int(1 if rng.random() < 0.85 else 0)
                else:
                    sample[v] = int(1 if rng.random() < 0.15 else 0)
        obs_data.append(sample)

    def intervention_fn(target, value, n_samples=200):
        samples = []
        for _ in range(n_samples):
            sample = {}
            for v in variables:
                if v == target:
                    sample[v] = int(value)
                elif not parents[v]:
                    sample[v] = int(rng.choice(domain))
                else:
                    parent_active = any(sample[p] == 1 for p in parents[v])
                    if parent_active:
                        sample[v] = int(1 if rng.random() < 0.85 else 0)
                    else:
                        sample[v] = int(1 if rng.random() < 0.15 else 0)
            samples.append(sample)
        return samples

    return obs_data, variables, intervention_fn, true_edges


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------

def causal_bandit_summary(result: CausalBanditResult) -> dict:
    """Generate summary report for a causal bandit run."""
    arm_summaries = {}
    for name, arm in result.arm_stats.items():
        arm_summaries[name] = {
            'pulls': arm.pulls,
            'mean_reward': round(arm.mean_reward, 4),
            'total_edges_oriented': sum(arm.edges_oriented_history),
        }

    return {
        'strategy': result.strategy,
        'total_rounds': result.total_rounds,
        'total_edges_oriented': result.total_edges_oriented,
        'total_cost': result.total_cost,
        'fully_oriented': result.fully_oriented,
        'undirected_remaining': result.final_cpdag.num_undirected(),
        'efficiency': result.total_edges_oriented / max(result.total_cost, 1e-6),
        'arm_summaries': arm_summaries,
    }
