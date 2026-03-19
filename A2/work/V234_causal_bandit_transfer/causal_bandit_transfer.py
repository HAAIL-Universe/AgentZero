"""V234: Causal Bandit Transfer -- Cross-Graph Transfer Learning for Causal Discovery.

Composes V231 (Causal Bandit) + V230 (Transfer BO).

The core idea: when discovering the causal structure of a new graph, leverage
experience from previously solved causal discovery tasks (source graphs) to
accelerate the bandit's intervention selection.

Transfer mechanisms:
1. Arm prior transfer: Initialize bandit arm priors from source graph statistics
   (which variables were most informative to intervene on, given structural features).
2. Structural feature transfer: Source graphs provide learned feature-reward
   relationships for the contextual bandit -- "nodes with high undirected degree
   tend to be informative" transfers across graphs.
3. Strategy selection transfer: Past task performance informs which bandit strategy
   works best for graphs with similar structural properties.
4. Graph embedding similarity: Compute structural similarity between source and
   target CPDAGs to weight transfer contributions.

This bridges V231's within-graph optimization with V230's cross-task transfer,
creating a system that gets better at causal discovery as it solves more problems.
"""

import sys
import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V231_causal_bandit'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V228_causal_discovery_interventions'))

from causal_bandit import (
    CausalBandit, CausalBanditResult, BanditStrategy, BanditRound,
    ArmStats, STRATEGY_MAP,
    causal_bandit_discovery, batch_causal_bandit_discovery,
    contextual_causal_bandit_discovery, adaptive_causal_bandit_discovery,
    budget_constrained_discovery, ContextualCausalBandit,
    edge_count_reward, normalized_edge_reward, information_gain_reward,
    _build_initial_cpdag, _active_arms, _cumsum,
    build_chain_environment, build_diamond_environment,
    build_confounded_environment, build_large_environment,
    causal_bandit_summary,
)
from causal_discovery_interventions import (
    CPDAG, orient_edges_from_intervention, _apply_meek_rules,
)


# ---------------------------------------------------------------------------
# Data structures for transfer
# ---------------------------------------------------------------------------

@dataclass
class GraphFeatures:
    """Structural features of a CPDAG for similarity computation."""
    n_variables: int
    n_directed: int
    n_undirected: int
    mean_degree: float
    max_degree: float
    density: float  # edges / max_possible_edges
    orientation_ratio: float  # directed / total edges

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.n_variables, self.n_directed, self.n_undirected,
            self.mean_degree, self.max_degree, self.density,
            self.orientation_ratio
        ])


@dataclass
class ArmPrior:
    """Transferred prior for a bandit arm based on source task experience."""
    feature_name: str  # which structural feature this prior is based on
    mean_reward: float
    confidence: float  # how much to trust this prior (0-1)
    n_source_observations: int


@dataclass
class SourceTask:
    """A completed causal discovery task that can be used for transfer."""
    task_id: int
    variables: List[str]
    initial_cpdag: CPDAG
    final_cpdag: CPDAG
    result: CausalBanditResult
    graph_features: GraphFeatures
    arm_feature_rewards: Dict[str, List[Tuple[np.ndarray, float]]]
    # Maps arm structural feature vectors to observed rewards
    strategy_used: str
    name: str = ""

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    @property
    def was_fully_oriented(self) -> bool:
        return self.result.fully_oriented


@dataclass
class TransferDB:
    """Database of completed causal discovery tasks for transfer."""
    tasks: List[SourceTask] = field(default_factory=list)

    def add_task(self, task: SourceTask):
        self.tasks.append(task)

    def n_tasks(self) -> int:
        return len(self.tasks)


@dataclass
class TransferBanditResult:
    """Result of transfer-augmented causal bandit discovery."""
    bandit_result: CausalBanditResult
    source_tasks_used: List[int]  # IDs of source tasks that contributed
    transfer_weights: List[float]  # per-source transfer weight
    strategy_selected: str
    prior_strength: float  # how much prior influenced early decisions
    speedup_vs_cold: Optional[float] = None


# ---------------------------------------------------------------------------
# Graph feature extraction
# ---------------------------------------------------------------------------

def extract_graph_features(cpdag: CPDAG) -> GraphFeatures:
    """Extract structural features from a CPDAG."""
    n_vars = len(cpdag.variables)
    n_dir = len(cpdag.directed)
    n_undir = cpdag.num_undirected()
    total_edges = n_dir + n_undir

    # Compute degrees
    degrees = []
    for v in cpdag.variables:
        deg = (len(cpdag.undirected_neighbors(v)) +
               len(cpdag.parents(v)) + len(cpdag.children(v)))
        degrees.append(deg)

    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = float(max(degrees)) if degrees else 0.0

    max_possible = n_vars * (n_vars - 1) / 2 if n_vars > 1 else 1
    density = total_edges / max_possible if max_possible > 0 else 0.0
    orientation = n_dir / total_edges if total_edges > 0 else 0.0

    return GraphFeatures(
        n_variables=n_vars,
        n_directed=n_dir,
        n_undirected=n_undir,
        mean_degree=mean_deg,
        max_degree=max_deg,
        density=density,
        orientation_ratio=orientation,
    )


def extract_arm_features(cpdag: CPDAG, arm_name: str) -> np.ndarray:
    """Extract per-arm structural features from a CPDAG.

    Returns 6-dimensional feature vector:
    [undirected_degree, directed_in_degree, directed_out_degree,
     total_degree, neighbor_mean_degree, fraction_of_undirected]
    """
    undir_deg = len(cpdag.undirected_neighbors(arm_name))
    in_deg = len(cpdag.parents(arm_name))
    out_deg = len(cpdag.children(arm_name))
    total_deg = undir_deg + in_deg + out_deg

    # Mean degree of neighbors
    neighbors = (list(cpdag.undirected_neighbors(arm_name)) +
                 list(cpdag.parents(arm_name)) +
                 list(cpdag.children(arm_name)))
    if neighbors:
        nbr_degs = []
        for n in neighbors:
            nd = (len(cpdag.undirected_neighbors(n)) +
                  len(cpdag.parents(n)) + len(cpdag.children(n)))
            nbr_degs.append(nd)
        nbr_mean = float(np.mean(nbr_degs))
    else:
        nbr_mean = 0.0

    total_edges = len(cpdag.directed) + cpdag.num_undirected()
    frac_undir = undir_deg / total_edges if total_edges > 0 else 0.0

    return np.array([undir_deg, in_deg, out_deg, total_deg, nbr_mean, frac_undir])


# ---------------------------------------------------------------------------
# Graph similarity
# ---------------------------------------------------------------------------

def graph_similarity(feat1: GraphFeatures, feat2: GraphFeatures) -> float:
    """Compute similarity between two graphs based on structural features.

    Uses RBF kernel on normalized feature vectors.
    """
    v1 = feat1.to_vector()
    v2 = feat2.to_vector()

    # Normalize by feature-specific scales
    scales = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 1.0, 1.0])
    d1 = v1 / (scales + 1e-10)
    d2 = v2 / (scales + 1e-10)

    dist_sq = float(np.sum((d1 - d2) ** 2))
    return float(np.exp(-0.5 * dist_sq))


def find_similar_sources(db: TransferDB, target_features: GraphFeatures,
                         top_k: int = 3, min_similarity: float = 0.1
                         ) -> List[Tuple[int, float]]:
    """Find the most similar source tasks for a target graph.

    Returns list of (task_index, similarity_score) sorted by similarity desc.
    """
    if db.n_tasks() == 0:
        return []

    similarities = []
    for i, task in enumerate(db.tasks):
        sim = graph_similarity(task.graph_features, target_features)
        if sim >= min_similarity:
            similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


# ---------------------------------------------------------------------------
# Arm prior transfer
# ---------------------------------------------------------------------------

def compute_arm_priors(db: TransferDB, target_cpdag: CPDAG,
                       target_features: GraphFeatures,
                       top_k: int = 3) -> Dict[str, ArmPrior]:
    """Compute transferred priors for target arms from source experience.

    For each arm in the target graph, finds structurally similar arms in
    source graphs and transfers their reward statistics.
    """
    similar_sources = find_similar_sources(db, target_features, top_k=top_k)
    if not similar_sources:
        return {}

    priors = {}
    for arm_name in target_cpdag.variables:
        target_arm_feat = extract_arm_features(target_cpdag, arm_name)

        weighted_reward_sum = 0.0
        weight_sum = 0.0
        n_obs = 0

        for task_idx, graph_sim in similar_sources:
            source_task = db.tasks[task_idx]

            # Find structurally similar arms in source
            for src_arm_name, feat_reward_pairs in source_task.arm_feature_rewards.items():
                for src_feat, src_reward in feat_reward_pairs:
                    # Arm feature similarity (RBF)
                    arm_dist_sq = float(np.sum((target_arm_feat - src_feat) ** 2))
                    arm_sim = float(np.exp(-0.5 * arm_dist_sq / (len(target_arm_feat) + 1e-10)))

                    # Combined weight: graph similarity * arm feature similarity
                    w = graph_sim * arm_sim
                    weighted_reward_sum += w * src_reward
                    weight_sum += w
                    n_obs += 1

        if weight_sum > 0 and n_obs > 0:
            mean_reward = weighted_reward_sum / weight_sum
            # Confidence based on number of source observations and graph similarity
            max_sim = similar_sources[0][1] if similar_sources else 0.0
            confidence = min(0.8, max_sim * min(1.0, n_obs / 5.0))

            priors[arm_name] = ArmPrior(
                feature_name="structural",
                mean_reward=mean_reward,
                confidence=confidence,
                n_source_observations=n_obs,
            )

    return priors


# ---------------------------------------------------------------------------
# Strategy selection transfer
# ---------------------------------------------------------------------------

def select_strategy_from_sources(db: TransferDB, target_features: GraphFeatures,
                                  top_k: int = 5) -> Tuple[BanditStrategy, Dict]:
    """Select the best bandit strategy based on source task performance.

    Looks at which strategies worked best on similar graphs and recommends
    the one with highest weighted efficiency.
    """
    similar = find_similar_sources(db, target_features, top_k=top_k)
    if not similar:
        return BanditStrategy.THOMPSON, {"reason": "no_sources", "confidence": 0.0}

    # Aggregate strategy performance weighted by graph similarity
    strategy_scores = {}  # strategy -> weighted_efficiency_sum
    strategy_weights = {}  # strategy -> total_weight

    for task_idx, sim in similar:
        task = db.tasks[task_idx]
        strat = task.strategy_used
        efficiency = task.result.total_edges_oriented / max(task.result.total_cost, 1e-6)
        bonus = 1.0 if task.was_fully_oriented else 0.5

        score = efficiency * bonus

        if strat not in strategy_scores:
            strategy_scores[strat] = 0.0
            strategy_weights[strat] = 0.0

        strategy_scores[strat] += sim * score
        strategy_weights[strat] += sim

    # Normalize and pick best
    best_strat = None
    best_score = -1.0
    for strat, total_score in strategy_scores.items():
        w = strategy_weights[strat]
        normalized = total_score / w if w > 0 else 0.0
        if normalized > best_score:
            best_score = normalized
            best_strat = strat

    # Map string back to BanditStrategy enum
    strat_map = {s.value: s for s in BanditStrategy}
    selected = strat_map.get(best_strat, BanditStrategy.THOMPSON)

    confidence = min(1.0, sum(s for _, s in similar) / len(similar))

    return selected, {
        "reason": "transfer",
        "best_strategy": best_strat,
        "score": best_score,
        "confidence": confidence,
        "n_sources": len(similar),
    }


# ---------------------------------------------------------------------------
# Transfer-augmented contextual bandit
# ---------------------------------------------------------------------------

class TransferContextualBandit(ContextualCausalBandit):
    """Contextual causal bandit with transferred priors from source tasks.

    Extends LinUCB with:
    1. Prior weight initialization from source arm feature-reward mappings
    2. Warm-started covariance matrix to reflect prior confidence
    """

    def __init__(self, variables: list, priors: Dict[str, ArmPrior],
                 prior_strength: float = 1.0,
                 costs: dict = None, rng: np.random.Generator = None):
        super().__init__(variables=variables, costs=costs, rng=rng)
        self.priors = priors
        self.prior_strength = prior_strength
        self._apply_priors()

    def _apply_priors(self):
        """Apply transferred priors to LinUCB weights."""
        for v in self.variables:
            if v in self.priors:
                prior = self.priors[v]
                strength = self.prior_strength * prior.confidence

                # Pseudo-observations: set reward vector as if we had
                # observed prior.mean_reward with n_pseudo features
                n_pseudo = max(1, int(prior.n_source_observations * strength))

                # Create a feature vector that represents the prior
                # Use a simple unit vector scaled by mean reward
                pseudo_feat = np.ones(self.feature_dim) / np.sqrt(self.feature_dim)
                self.covariance[v] += n_pseudo * np.outer(pseudo_feat, pseudo_feat)
                self.reward_vec[v] += n_pseudo * prior.mean_reward * pseudo_feat


# ---------------------------------------------------------------------------
# Source task construction
# ---------------------------------------------------------------------------

def record_source_task(task_id: int, variables: List[str],
                       initial_cpdag: CPDAG, final_cpdag: CPDAG,
                       result: CausalBanditResult,
                       cpdag_snapshots: List[CPDAG] = None,
                       name: str = "") -> SourceTask:
    """Record a completed causal discovery task for future transfer.

    Extracts structural features and arm feature-reward mappings from
    the completed task.
    """
    graph_features = extract_graph_features(initial_cpdag)

    # Build arm feature-reward mapping from round history
    arm_feature_rewards = {v: [] for v in variables}

    if cpdag_snapshots and len(cpdag_snapshots) >= len(result.rounds):
        for i, round_rec in enumerate(result.rounds):
            arm = round_rec.arm_selected
            cpdag_at = cpdag_snapshots[i]
            features = extract_arm_features(cpdag_at, arm)
            reward = round_rec.reward
            arm_feature_rewards[arm].append((features, reward))
    else:
        # Fall back: use arm stats summary with initial CPDAG features
        for v in variables:
            if v in result.arm_stats:
                arm_stat = result.arm_stats[v]
                features = extract_arm_features(initial_cpdag, v)
                if arm_stat.pulls > 0:
                    arm_feature_rewards[v].append((features, arm_stat.mean_reward))

    return SourceTask(
        task_id=task_id,
        variables=variables,
        initial_cpdag=initial_cpdag,
        final_cpdag=final_cpdag,
        result=result,
        graph_features=graph_features,
        arm_feature_rewards=arm_feature_rewards,
        strategy_used=result.strategy,
        name=name,
    )


# ---------------------------------------------------------------------------
# Core: Transfer Causal Bandit Discovery
# ---------------------------------------------------------------------------

def transfer_causal_bandit_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    db: TransferDB,
    strategy: BanditStrategy = None,  # None = auto-select from sources
    max_rounds: int = 20,
    reward_fn: Callable = None,
    costs: dict = None,
    alpha: float = 0.05,
    prior_strength: float = 1.0,
    auto_strategy: bool = True,
    rng: np.random.Generator = None,
) -> TransferBanditResult:
    """Run causal bandit discovery with transfer from source tasks.

    Args:
        initial_data: Observational data samples
        variables: Variable names
        intervention_fn: Callable(target, value) -> list[dict]
        db: Database of completed source tasks
        strategy: Bandit strategy (None for auto-selection)
        max_rounds: Maximum intervention rounds
        reward_fn: Reward function
        costs: Per-variable intervention costs
        alpha: Significance level
        prior_strength: How much to trust transferred priors (0-2)
        auto_strategy: If True and strategy is None, auto-select from sources
        rng: Random number generator

    Returns:
        TransferBanditResult
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build initial CPDAG
    cpdag = _build_initial_cpdag(initial_data, variables, alpha)
    initial_cpdag = cpdag.copy()
    target_features = extract_graph_features(cpdag)

    # Find similar source tasks
    similar_sources = find_similar_sources(db, target_features, top_k=5)
    source_ids = [db.tasks[idx].task_id for idx, _ in similar_sources]
    source_weights = [w for _, w in similar_sources]

    # Auto-select strategy if not specified
    strategy_info = {}
    if strategy is None and auto_strategy and db.n_tasks() > 0:
        strategy, strategy_info = select_strategy_from_sources(db, target_features)
    elif strategy is None:
        strategy = BanditStrategy.THOMPSON

    # Compute arm priors from source experience
    priors = compute_arm_priors(db, cpdag, target_features)

    # Initialize bandit with transferred priors
    if priors and prior_strength > 0:
        # Use transfer-augmented selection
        bandit = CausalBandit(
            variables=variables,
            strategy=strategy,
            reward_fn=reward_fn,
            costs=costs,
            rng=rng,
        )
        # Apply priors as virtual pulls
        _apply_priors_to_bandit(bandit, priors, prior_strength)
    else:
        bandit = CausalBandit(
            variables=variables,
            strategy=strategy,
            reward_fn=reward_fn,
            costs=costs,
            rng=rng,
        )

    obs_data = list(initial_data)
    cpdag_snapshots = []

    for t in range(max_rounds):
        if cpdag.is_fully_oriented():
            break

        cpdag_snapshots.append(cpdag.copy())

        arm = bandit.select_arm(cpdag, t + 1, obs_data)
        if arm is None:
            break

        int_data = intervention_fn(arm, 1)
        cpdag_before = cpdag.copy()
        cpdag, newly_oriented = orient_edges_from_intervention(
            cpdag, arm, int_data, obs_data, alpha
        )
        edges_oriented = len(newly_oriented)

        bandit.update(arm, edges_oriented, cpdag_before, cpdag, t)

    bandit_result = bandit.result(cpdag)

    return TransferBanditResult(
        bandit_result=bandit_result,
        source_tasks_used=source_ids,
        transfer_weights=source_weights,
        strategy_selected=strategy.value,
        prior_strength=prior_strength,
    )


def _apply_priors_to_bandit(bandit: CausalBandit, priors: Dict[str, ArmPrior],
                             strength: float):
    """Apply transferred priors as virtual pulls to a standard bandit."""
    for arm_name, prior in priors.items():
        if arm_name not in bandit.arms:
            continue
        arm = bandit.arms[arm_name]
        # Add virtual observations weighted by confidence
        n_virtual = max(1, int(strength * prior.confidence * 3))
        for _ in range(n_virtual):
            arm.update(prior.mean_reward, edges_oriented=0)


# ---------------------------------------------------------------------------
# Transfer contextual bandit discovery
# ---------------------------------------------------------------------------

def transfer_contextual_discovery(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    db: TransferDB,
    max_rounds: int = 20,
    costs: dict = None,
    alpha: float = 0.05,
    prior_strength: float = 1.0,
    rng: np.random.Generator = None,
) -> TransferBanditResult:
    """Run contextual bandit discovery with LinUCB + transferred priors.

    The contextual bandit uses CPDAG structural features to select arms.
    Source task experience warm-starts the LinUCB model weights.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cpdag = _build_initial_cpdag(initial_data, variables, alpha)
    initial_cpdag = cpdag.copy()
    target_features = extract_graph_features(cpdag)

    # Compute priors
    priors = compute_arm_priors(db, cpdag, target_features)
    similar_sources = find_similar_sources(db, target_features, top_k=5)
    source_ids = [db.tasks[idx].task_id for idx, _ in similar_sources]
    source_weights = [w for _, w in similar_sources]

    # Create transfer-augmented contextual bandit
    bandit = TransferContextualBandit(
        variables=variables,
        priors=priors,
        prior_strength=prior_strength,
        costs=costs,
        rng=rng,
    )

    obs_data = list(initial_data)

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

    bandit_result = bandit.result(cpdag)

    return TransferBanditResult(
        bandit_result=bandit_result,
        source_tasks_used=source_ids,
        transfer_weights=source_weights,
        strategy_selected="contextual_transfer",
        prior_strength=prior_strength,
    )


# ---------------------------------------------------------------------------
# Run-and-store: discover + record for future transfer
# ---------------------------------------------------------------------------

def discover_and_store(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    db: TransferDB,
    task_name: str = "",
    strategy: BanditStrategy = None,
    max_rounds: int = 20,
    reward_fn: Callable = None,
    costs: dict = None,
    alpha: float = 0.05,
    prior_strength: float = 1.0,
    rng: np.random.Generator = None,
) -> TransferBanditResult:
    """Discover causal structure with transfer, then store result for future use.

    Main entry point for sequential causal discovery with accumulating knowledge.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build initial CPDAG for recording
    initial_cpdag = _build_initial_cpdag(initial_data, variables, alpha)

    # Run transfer discovery
    result = transfer_causal_bandit_discovery(
        initial_data=initial_data,
        variables=variables,
        intervention_fn=intervention_fn,
        db=db,
        strategy=strategy,
        max_rounds=max_rounds,
        reward_fn=reward_fn,
        costs=costs,
        alpha=alpha,
        prior_strength=prior_strength,
        rng=rng,
    )

    # Record source task
    source_task = record_source_task(
        task_id=db.n_tasks(),
        variables=variables,
        initial_cpdag=initial_cpdag,
        final_cpdag=result.bandit_result.final_cpdag,
        result=result.bandit_result,
        name=task_name,
    )
    db.add_task(source_task)

    return result


# ---------------------------------------------------------------------------
# Negative transfer detection
# ---------------------------------------------------------------------------

def detect_negative_transfer(transfer_result: TransferBanditResult,
                              cold_result: CausalBanditResult,
                              threshold: float = 0.1) -> Dict:
    """Detect if transfer hurt discovery performance.

    Compares transfer-augmented vs cold-start on:
    - Total edges oriented
    - Rounds needed for full orientation
    - Efficiency (edges per round)
    """
    tr = transfer_result.bandit_result
    cr = cold_result

    # Efficiency comparison
    transfer_eff = tr.total_edges_oriented / max(tr.total_rounds, 1)
    cold_eff = cr.total_edges_oriented / max(cr.total_rounds, 1)

    improvement = (transfer_eff - cold_eff) / max(cold_eff, 1e-6)
    is_negative = improvement < -threshold

    # Orientation comparison
    transfer_oriented = tr.fully_oriented
    cold_oriented = cr.fully_oriented

    return {
        "is_negative_transfer": is_negative,
        "improvement": float(improvement),
        "transfer_efficiency": float(transfer_eff),
        "cold_efficiency": float(cold_eff),
        "transfer_edges": tr.total_edges_oriented,
        "cold_edges": cr.total_edges_oriented,
        "transfer_rounds": tr.total_rounds,
        "cold_rounds": cr.total_rounds,
        "transfer_fully_oriented": transfer_oriented,
        "cold_fully_oriented": cold_oriented,
    }


# ---------------------------------------------------------------------------
# Compare transfer vs cold-start
# ---------------------------------------------------------------------------

def compare_transfer_vs_cold(
    initial_data: list,
    variables: list,
    intervention_fn: Callable,
    db: TransferDB,
    strategies: List[BanditStrategy] = None,
    max_rounds: int = 20,
    n_trials: int = 3,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> Dict[str, Dict]:
    """Compare transfer-augmented vs cold-start across strategies.

    Returns dict mapping label -> {
        mean_edges, mean_rounds, mean_efficiency, fully_oriented_count,
        results: list
    }
    """
    if strategies is None:
        strategies = [BanditStrategy.UCB, BanditStrategy.THOMPSON]

    comparison = {}

    # Cold-start runs
    for strat in strategies:
        results = []
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_seed + trial)
            r = causal_bandit_discovery(
                initial_data=initial_data,
                variables=variables,
                intervention_fn=intervention_fn,
                strategy=strat,
                max_rounds=max_rounds,
                alpha=alpha,
                rng=rng,
            )
            results.append(r)

        comparison[f"cold_{strat.value}"] = _summarize_results(results)

    # Transfer runs
    for strat in strategies:
        results = []
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_seed + trial)
            r = transfer_causal_bandit_discovery(
                initial_data=initial_data,
                variables=variables,
                intervention_fn=intervention_fn,
                db=db,
                strategy=strat,
                max_rounds=max_rounds,
                alpha=alpha,
                rng=rng,
            )
            results.append(r.bandit_result)

        comparison[f"transfer_{strat.value}"] = _summarize_results(results)

    # Auto-strategy transfer
    results = []
    for trial in range(n_trials):
        rng = np.random.default_rng(rng_seed + trial)
        r = transfer_causal_bandit_discovery(
            initial_data=initial_data,
            variables=variables,
            intervention_fn=intervention_fn,
            db=db,
            strategy=None,
            max_rounds=max_rounds,
            alpha=alpha,
            rng=rng,
        )
        results.append(r.bandit_result)

    comparison["transfer_auto"] = _summarize_results(results)

    return comparison


def _summarize_results(results: List[CausalBanditResult]) -> Dict:
    """Summarize a list of causal bandit results."""
    edges = [r.total_edges_oriented for r in results]
    rounds = [r.total_rounds for r in results]
    efficiencies = [r.total_edges_oriented / max(r.total_rounds, 1) for r in results]
    fully = sum(1 for r in results if r.fully_oriented)

    return {
        "mean_edges": float(np.mean(edges)),
        "mean_rounds": float(np.mean(rounds)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "fully_oriented_count": fully,
        "n_trials": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Multi-graph sequential discovery
# ---------------------------------------------------------------------------

def sequential_discovery(
    environments: List[Tuple],
    strategy: BanditStrategy = None,
    max_rounds: int = 20,
    alpha: float = 0.05,
    prior_strength: float = 1.0,
    rng_seed: int = 42,
) -> Tuple[TransferDB, List[TransferBanditResult]]:
    """Discover causal structure in a sequence of graphs, accumulating knowledge.

    Each environment is (initial_data, variables, intervention_fn).
    The first graph is solved cold-start; subsequent graphs benefit from transfer.

    Returns (database, list_of_results).
    """
    db = TransferDB()
    results = []

    for i, env in enumerate(environments):
        initial_data, variables, intervention_fn = env
        rng = np.random.default_rng(rng_seed + i)

        result = discover_and_store(
            initial_data=initial_data,
            variables=variables,
            intervention_fn=intervention_fn,
            db=db,
            task_name=f"task_{i}",
            strategy=strategy,
            max_rounds=max_rounds,
            alpha=alpha,
            prior_strength=prior_strength,
            rng=rng,
        )
        results.append(result)

    return db, results


# ---------------------------------------------------------------------------
# Learning curve: how transfer improves with more source tasks
# ---------------------------------------------------------------------------

def transfer_learning_curve(
    target_env: Tuple,
    source_envs: List[Tuple],
    max_rounds: int = 20,
    alpha: float = 0.05,
    n_trials: int = 3,
    rng_seed: int = 42,
) -> Dict:
    """Measure how discovery performance improves as more source tasks are added.

    Returns dict with:
        n_sources: list of source counts
        mean_edges: mean edges oriented for each count
        mean_rounds: mean rounds used for each count
        mean_efficiency: mean efficiency for each count
    """
    target_data, target_vars, target_int_fn = target_env

    n_sources_list = []
    mean_edges_list = []
    mean_rounds_list = []
    mean_eff_list = []

    for n_sources in range(len(source_envs) + 1):
        db = TransferDB()

        # Add source tasks cold-start
        for i in range(n_sources):
            src_data, src_vars, src_int_fn = source_envs[i]
            rng = np.random.default_rng(rng_seed + 1000 + i)

            src_cpdag = _build_initial_cpdag(src_data, src_vars, alpha)
            src_initial = src_cpdag.copy()

            src_result = causal_bandit_discovery(
                initial_data=src_data,
                variables=src_vars,
                intervention_fn=src_int_fn,
                strategy=BanditStrategy.THOMPSON,
                max_rounds=max_rounds,
                alpha=alpha,
                rng=rng,
            )

            source_task = record_source_task(
                task_id=i,
                variables=src_vars,
                initial_cpdag=src_initial,
                final_cpdag=src_result.final_cpdag,
                result=src_result,
                name=f"source_{i}",
            )
            db.add_task(source_task)

        # Run target with current number of sources
        trial_edges = []
        trial_rounds = []
        trial_eff = []

        for trial in range(n_trials):
            rng = np.random.default_rng(rng_seed + trial)
            result = transfer_causal_bandit_discovery(
                initial_data=target_data,
                variables=target_vars,
                intervention_fn=target_int_fn,
                db=db,
                max_rounds=max_rounds,
                alpha=alpha,
                rng=rng,
            )
            br = result.bandit_result
            trial_edges.append(br.total_edges_oriented)
            trial_rounds.append(br.total_rounds)
            trial_eff.append(br.total_edges_oriented / max(br.total_rounds, 1))

        n_sources_list.append(n_sources)
        mean_edges_list.append(float(np.mean(trial_edges)))
        mean_rounds_list.append(float(np.mean(trial_rounds)))
        mean_eff_list.append(float(np.mean(trial_eff)))

    return {
        "n_sources": n_sources_list,
        "mean_edges": mean_edges_list,
        "mean_rounds": mean_rounds_list,
        "mean_efficiency": mean_eff_list,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def transfer_bandit_summary(result: TransferBanditResult, name: str = "") -> str:
    """Human-readable summary of transfer bandit result."""
    br = result.bandit_result
    lines = [f"=== Transfer Causal Bandit{' (' + name + ')' if name else ''} ==="]
    lines.append(f"  Strategy: {result.strategy_selected}")
    lines.append(f"  Edges oriented: {br.total_edges_oriented}")
    lines.append(f"  Rounds: {br.total_rounds}")
    lines.append(f"  Fully oriented: {br.fully_oriented}")
    lines.append(f"  Cost: {br.total_cost:.1f}")
    lines.append(f"  Sources used: {len(result.source_tasks_used)}")
    if result.transfer_weights:
        lines.append(f"  Transfer weights: {[f'{w:.3f}' for w in result.transfer_weights]}")
    lines.append(f"  Prior strength: {result.prior_strength:.2f}")
    eff = br.total_edges_oriented / max(br.total_rounds, 1)
    lines.append(f"  Efficiency: {eff:.2f} edges/round")
    return "\n".join(lines)
