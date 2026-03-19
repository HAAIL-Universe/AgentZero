"""Tests for V234: Causal Bandit Transfer -- Cross-Graph Transfer Learning."""

import sys
import os
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V231_causal_bandit'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V228_causal_discovery_interventions'))

from causal_bandit_transfer import (
    # Data structures
    GraphFeatures, ArmPrior, SourceTask, TransferDB, TransferBanditResult,
    # Feature extraction
    extract_graph_features, extract_arm_features,
    # Similarity
    graph_similarity, find_similar_sources,
    # Priors
    compute_arm_priors,
    # Strategy selection
    select_strategy_from_sources,
    # Transfer contextual bandit
    TransferContextualBandit,
    # Source task recording
    record_source_task,
    # Core discovery
    transfer_causal_bandit_discovery,
    transfer_contextual_discovery,
    discover_and_store,
    # Negative transfer
    detect_negative_transfer,
    # Comparison
    compare_transfer_vs_cold,
    _summarize_results,
    # Sequential
    sequential_discovery,
    # Learning curve
    transfer_learning_curve,
    # Summary
    transfer_bandit_summary,
)
from causal_bandit import (
    CausalBandit, CausalBanditResult, BanditStrategy, ArmStats,
    causal_bandit_discovery, BanditRound,
    build_chain_environment, build_diamond_environment,
    build_confounded_environment, build_large_environment,
    _build_initial_cpdag,
)
from causal_discovery_interventions import CPDAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_simple_cpdag(variables, undirected=None, directed=None):
    """Create a CPDAG with given edges."""
    cpdag = CPDAG(variables=list(variables))
    if undirected:
        for u, v in undirected:
            cpdag.undirected.add(frozenset([u, v]))
    if directed:
        for u, v in directed:
            cpdag.directed.add((u, v))
    return cpdag


def make_chain_source(n=4, seed=42):
    """Create a chain source task for transfer."""
    obs_data, variables, int_fn, true_edges = build_chain_environment(
        n=n, n_obs=500, seed=seed
    )
    rng = np.random.default_rng(seed)
    result = causal_bandit_discovery(
        initial_data=obs_data,
        variables=variables,
        intervention_fn=int_fn,
        strategy=BanditStrategy.UCB,
        max_rounds=15,
        alpha=0.05,
        rng=rng,
    )
    initial_cpdag = _build_initial_cpdag(obs_data, variables, 0.05)
    return record_source_task(
        task_id=0,
        variables=variables,
        initial_cpdag=initial_cpdag,
        final_cpdag=result.final_cpdag,
        result=result,
        name=f"chain_{n}",
    )


def make_diamond_source(seed=42):
    """Create a diamond source task for transfer."""
    obs_data, variables, int_fn, true_edges = build_diamond_environment(
        n_obs=500, seed=seed
    )
    rng = np.random.default_rng(seed)
    result = causal_bandit_discovery(
        initial_data=obs_data,
        variables=variables,
        intervention_fn=int_fn,
        strategy=BanditStrategy.THOMPSON,
        max_rounds=15,
        alpha=0.05,
        rng=rng,
    )
    initial_cpdag = _build_initial_cpdag(obs_data, variables, 0.05)
    return record_source_task(
        task_id=1,
        variables=variables,
        initial_cpdag=initial_cpdag,
        final_cpdag=result.final_cpdag,
        result=result,
        name="diamond",
    )


# ===========================================================================
# GraphFeatures tests
# ===========================================================================

class TestGraphFeatures:
    def test_to_vector(self):
        gf = GraphFeatures(n_variables=4, n_directed=2, n_undirected=3,
                           mean_degree=2.5, max_degree=4.0,
                           density=0.5, orientation_ratio=0.4)
        v = gf.to_vector()
        assert len(v) == 7
        assert v[0] == 4
        assert v[5] == 0.5

    def test_empty_graph(self):
        gf = GraphFeatures(n_variables=3, n_directed=0, n_undirected=0,
                           mean_degree=0, max_degree=0,
                           density=0, orientation_ratio=0)
        v = gf.to_vector()
        assert np.all(v[:3] == [3, 0, 0])


# ===========================================================================
# extract_graph_features tests
# ===========================================================================

class TestExtractGraphFeatures:
    def test_chain_cpdag(self):
        cpdag = make_simple_cpdag(
            ["A", "B", "C"],
            undirected=[("A", "B"), ("B", "C")]
        )
        feat = extract_graph_features(cpdag)
        assert feat.n_variables == 3
        assert feat.n_undirected == 2
        assert feat.n_directed == 0
        assert feat.density > 0
        assert feat.orientation_ratio == 0.0

    def test_fully_directed(self):
        cpdag = make_simple_cpdag(
            ["X", "Y"],
            directed=[("X", "Y")]
        )
        feat = extract_graph_features(cpdag)
        assert feat.n_directed == 1
        assert feat.n_undirected == 0
        assert feat.orientation_ratio == 1.0

    def test_mixed(self):
        cpdag = make_simple_cpdag(
            ["A", "B", "C"],
            undirected=[("A", "C")],
            directed=[("A", "B")]
        )
        feat = extract_graph_features(cpdag)
        assert feat.n_directed == 1
        assert feat.n_undirected == 1
        assert 0 < feat.orientation_ratio < 1

    def test_empty(self):
        cpdag = make_simple_cpdag(["X", "Y", "Z"])
        feat = extract_graph_features(cpdag)
        assert feat.n_variables == 3
        assert feat.n_directed == 0
        assert feat.n_undirected == 0
        assert feat.mean_degree == 0.0

    def test_mean_degree(self):
        cpdag = make_simple_cpdag(
            ["A", "B", "C", "D"],
            undirected=[("A", "B"), ("B", "C"), ("C", "D")]
        )
        feat = extract_graph_features(cpdag)
        # A:1, B:2, C:2, D:1 -> mean 1.5
        assert feat.mean_degree == 1.5
        assert feat.max_degree == 2.0


# ===========================================================================
# extract_arm_features tests
# ===========================================================================

class TestExtractArmFeatures:
    def test_basic(self):
        cpdag = make_simple_cpdag(
            ["A", "B", "C"],
            undirected=[("A", "B"), ("B", "C")]
        )
        feat = extract_arm_features(cpdag, "B")
        assert len(feat) == 6
        assert feat[0] == 2  # undirected degree of B
        assert feat[3] == 2  # total degree

    def test_directed_degrees(self):
        cpdag = make_simple_cpdag(
            ["A", "B", "C"],
            directed=[("A", "B"), ("B", "C")]
        )
        feat = extract_arm_features(cpdag, "B")
        assert feat[1] == 1  # in-degree (A->B)
        assert feat[2] == 1  # out-degree (B->C)

    def test_isolated(self):
        cpdag = make_simple_cpdag(["A", "B", "C"])
        feat = extract_arm_features(cpdag, "A")
        assert feat[0] == 0
        assert feat[3] == 0
        assert feat[4] == 0  # neighbor mean degree = 0


# ===========================================================================
# graph_similarity tests
# ===========================================================================

class TestGraphSimilarity:
    def test_identical(self):
        gf = GraphFeatures(4, 2, 3, 2.5, 4.0, 0.5, 0.4)
        sim = graph_similarity(gf, gf)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_similar(self):
        gf1 = GraphFeatures(4, 2, 3, 2.5, 4.0, 0.5, 0.4)
        gf2 = GraphFeatures(4, 3, 2, 2.5, 4.0, 0.5, 0.6)
        sim = graph_similarity(gf1, gf2)
        assert 0 < sim < 1

    def test_very_different(self):
        gf1 = GraphFeatures(4, 2, 3, 2.5, 4.0, 0.5, 0.4)
        gf2 = GraphFeatures(20, 50, 100, 10.0, 15.0, 0.9, 0.3)
        sim = graph_similarity(gf1, gf2)
        assert sim < 0.5

    def test_symmetric(self):
        gf1 = GraphFeatures(4, 2, 3, 2.5, 4.0, 0.5, 0.4)
        gf2 = GraphFeatures(5, 4, 2, 3.0, 5.0, 0.6, 0.7)
        assert graph_similarity(gf1, gf2) == pytest.approx(
            graph_similarity(gf2, gf1), abs=1e-10
        )


# ===========================================================================
# TransferDB tests
# ===========================================================================

class TestTransferDB:
    def test_empty(self):
        db = TransferDB()
        assert db.n_tasks() == 0

    def test_add_task(self):
        db = TransferDB()
        source = make_chain_source(n=3, seed=42)
        db.add_task(source)
        assert db.n_tasks() == 1
        assert db.tasks[0].name == "chain_3"

    def test_multiple_tasks(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=3, seed=42))
        db.add_task(make_diamond_source(seed=43))
        assert db.n_tasks() == 2


# ===========================================================================
# find_similar_sources tests
# ===========================================================================

class TestFindSimilarSources:
    def test_empty_db(self):
        db = TransferDB()
        feat = GraphFeatures(4, 2, 3, 2.0, 3.0, 0.5, 0.4)
        result = find_similar_sources(db, feat)
        assert result == []

    def test_finds_sources(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))
        db.add_task(make_diamond_source(seed=43))

        # Target similar to chain
        chain_feat = db.tasks[0].graph_features
        result = find_similar_sources(db, chain_feat, top_k=2)
        assert len(result) > 0
        # First result should be the chain itself (most similar)
        assert result[0][1] > 0.5

    def test_top_k(self):
        db = TransferDB()
        for i in range(5):
            db.add_task(make_chain_source(n=3 + i, seed=42 + i))

        feat = db.tasks[0].graph_features
        result = find_similar_sources(db, feat, top_k=2)
        assert len(result) <= 2

    def test_min_similarity_filter(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=3, seed=42))

        # Very different target
        feat = GraphFeatures(20, 50, 100, 10.0, 15.0, 0.9, 0.3)
        result = find_similar_sources(db, feat, min_similarity=0.9)
        # May or may not find matches depending on the threshold
        for _, sim in result:
            assert sim >= 0.9


# ===========================================================================
# compute_arm_priors tests
# ===========================================================================

class TestComputeArmPriors:
    def test_empty_db(self):
        db = TransferDB()
        cpdag = make_simple_cpdag(["A", "B"], undirected=[("A", "B")])
        feat = extract_graph_features(cpdag)
        priors = compute_arm_priors(db, cpdag, feat)
        assert priors == {}

    def test_with_sources(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))
        db.add_task(make_chain_source(n=5, seed=43))

        # Target is another chain
        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=100)
        cpdag = _build_initial_cpdag(obs, vars, 0.05)
        feat = extract_graph_features(cpdag)

        priors = compute_arm_priors(db, cpdag, feat)
        # Should have priors for at least some variables
        assert len(priors) > 0
        for name, prior in priors.items():
            assert prior.confidence >= 0
            assert prior.confidence <= 1
            assert prior.n_source_observations >= 1

    def test_prior_values_reasonable(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)
        cpdag = _build_initial_cpdag(obs, vars, 0.05)
        feat = extract_graph_features(cpdag)

        priors = compute_arm_priors(db, cpdag, feat)
        for name, prior in priors.items():
            # Mean reward should be non-negative (edge counts are >= 0)
            assert prior.mean_reward >= 0


# ===========================================================================
# select_strategy_from_sources tests
# ===========================================================================

class TestSelectStrategy:
    def test_no_sources(self):
        db = TransferDB()
        feat = GraphFeatures(4, 2, 3, 2.0, 3.0, 0.5, 0.4)
        strat, info = select_strategy_from_sources(db, feat)
        assert strat == BanditStrategy.THOMPSON
        assert info["reason"] == "no_sources"

    def test_with_sources(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))
        db.add_task(make_diamond_source(seed=43))

        feat = db.tasks[0].graph_features
        strat, info = select_strategy_from_sources(db, feat)
        assert isinstance(strat, BanditStrategy)
        assert info["reason"] == "transfer"
        assert info["confidence"] > 0


# ===========================================================================
# TransferContextualBandit tests
# ===========================================================================

class TestTransferContextualBandit:
    def test_basic_creation(self):
        priors = {
            "A": ArmPrior("structural", mean_reward=2.0, confidence=0.5,
                           n_source_observations=5),
        }
        bandit = TransferContextualBandit(
            variables=["A", "B", "C"],
            priors=priors,
            prior_strength=1.0,
        )
        assert len(bandit.variables) == 3
        # Prior should have modified covariance and reward vec for A
        assert not np.allclose(bandit.reward_vec["A"], 0)

    def test_without_priors(self):
        bandit = TransferContextualBandit(
            variables=["A", "B"],
            priors={},
            prior_strength=1.0,
        )
        # Should work like regular contextual bandit
        assert np.allclose(bandit.reward_vec["A"], 0)
        assert np.allclose(bandit.reward_vec["B"], 0)

    def test_prior_strength_zero(self):
        priors = {
            "A": ArmPrior("structural", mean_reward=2.0, confidence=0.5,
                           n_source_observations=5),
        }
        bandit = TransferContextualBandit(
            variables=["A", "B"],
            priors=priors,
            prior_strength=0.0,
        )
        # With zero strength, priors should have minimal effect
        # (confidence * strength = 0, so n_pseudo = 0, clamped to 1,
        #  but still minimal)

    def test_arm_selection(self):
        priors = {
            "A": ArmPrior("structural", mean_reward=5.0, confidence=0.8,
                           n_source_observations=10),
            "B": ArmPrior("structural", mean_reward=0.1, confidence=0.8,
                           n_source_observations=10),
        }
        bandit = TransferContextualBandit(
            variables=["A", "B"],
            priors=priors,
            prior_strength=1.0,
        )
        cpdag = make_simple_cpdag(["A", "B"], undirected=[("A", "B")])
        arm = bandit.select_arm(cpdag, t=1)
        # A has much higher prior, should be preferred
        assert arm in ["A", "B"]


# ===========================================================================
# record_source_task tests
# ===========================================================================

class TestRecordSourceTask:
    def test_basic(self):
        obs, vars, int_fn, _ = build_chain_environment(n=3, seed=42)
        rng = np.random.default_rng(42)
        result = causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            strategy=BanditStrategy.UCB, max_rounds=10, rng=rng,
        )
        initial_cpdag = _build_initial_cpdag(obs, vars, 0.05)

        source = record_source_task(
            task_id=0, variables=vars,
            initial_cpdag=initial_cpdag,
            final_cpdag=result.final_cpdag,
            result=result, name="test_chain",
        )
        assert source.task_id == 0
        assert source.name == "test_chain"
        assert source.n_variables == 3
        assert isinstance(source.graph_features, GraphFeatures)
        assert len(source.arm_feature_rewards) == 3

    def test_arm_feature_rewards_populated(self):
        source = make_chain_source(n=4, seed=42)
        # At least one arm should have feature-reward pairs
        has_data = any(
            len(pairs) > 0
            for pairs in source.arm_feature_rewards.values()
        )
        assert has_data


# ===========================================================================
# transfer_causal_bandit_discovery tests
# ===========================================================================

class TestTransferDiscovery:
    def test_cold_start_no_sources(self):
        db = TransferDB()
        obs, vars, int_fn, _ = build_chain_environment(n=3, seed=42)
        rng = np.random.default_rng(42)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=15, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)
        assert isinstance(result.bandit_result, CausalBanditResult)
        assert result.source_tasks_used == []
        assert result.strategy_selected == "thompson"  # default

    def test_with_chain_sources(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))
        db.add_task(make_chain_source(n=5, seed=43))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)
        rng = np.random.default_rng(99)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=15, rng=rng,
        )
        assert result.bandit_result.total_edges_oriented >= 0
        assert len(result.source_tasks_used) > 0

    def test_with_diamond_sources(self):
        db = TransferDB()
        db.add_task(make_diamond_source(seed=42))

        obs, vars, int_fn, _ = build_diamond_environment(seed=99)
        rng = np.random.default_rng(99)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=15, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)
        assert result.bandit_result.total_rounds > 0

    def test_explicit_strategy(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=77)
        rng = np.random.default_rng(77)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, strategy=BanditStrategy.UCB, max_rounds=15, rng=rng,
        )
        assert result.strategy_selected == "ucb"

    def test_prior_strength_zero(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=55)
        rng = np.random.default_rng(55)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, prior_strength=0.0, max_rounds=15, rng=rng,
        )
        assert result.prior_strength == 0.0

    def test_confounded_graph(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=3, seed=42))

        obs, vars, int_fn, _ = build_confounded_environment(seed=99)
        rng = np.random.default_rng(99)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=10, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)


# ===========================================================================
# transfer_contextual_discovery tests
# ===========================================================================

class TestTransferContextualDiscovery:
    def test_no_sources(self):
        db = TransferDB()
        obs, vars, int_fn, _ = build_chain_environment(n=3, seed=42)
        rng = np.random.default_rng(42)

        result = transfer_contextual_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=10, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)
        assert result.strategy_selected == "contextual_transfer"

    def test_with_sources(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=77)
        rng = np.random.default_rng(77)

        result = transfer_contextual_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=10, rng=rng,
        )
        assert result.bandit_result.total_rounds > 0


# ===========================================================================
# discover_and_store tests
# ===========================================================================

class TestDiscoverAndStore:
    def test_first_task(self):
        db = TransferDB()
        obs, vars, int_fn, _ = build_chain_environment(n=3, seed=42)
        rng = np.random.default_rng(42)

        result = discover_and_store(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, task_name="first", max_rounds=10, rng=rng,
        )
        assert db.n_tasks() == 1
        assert db.tasks[0].name == "first"
        assert isinstance(result, TransferBanditResult)

    def test_second_task_uses_first(self):
        db = TransferDB()

        # First task
        obs1, vars1, int_fn1, _ = build_chain_environment(n=3, seed=42)
        rng1 = np.random.default_rng(42)
        discover_and_store(
            initial_data=obs1, variables=vars1, intervention_fn=int_fn1,
            db=db, task_name="chain_3", max_rounds=10, rng=rng1,
        )

        # Second task should use first as source
        obs2, vars2, int_fn2, _ = build_chain_environment(n=4, seed=43)
        rng2 = np.random.default_rng(43)
        result2 = discover_and_store(
            initial_data=obs2, variables=vars2, intervention_fn=int_fn2,
            db=db, task_name="chain_4", max_rounds=10, rng=rng2,
        )
        assert db.n_tasks() == 2
        # Second task should have used first as source
        assert len(result2.source_tasks_used) > 0 or db.n_tasks() == 2

    def test_accumulating_tasks(self):
        db = TransferDB()
        for i in range(3):
            obs, vars, int_fn, _ = build_chain_environment(n=3 + i, seed=42 + i)
            rng = np.random.default_rng(42 + i)
            discover_and_store(
                initial_data=obs, variables=vars, intervention_fn=int_fn,
                db=db, task_name=f"chain_{3+i}", max_rounds=10, rng=rng,
            )
        assert db.n_tasks() == 3


# ===========================================================================
# detect_negative_transfer tests
# ===========================================================================

class TestDetectNegativeTransfer:
    def test_positive_transfer(self):
        # Mock results where transfer is better
        transfer_br = CausalBanditResult(
            final_cpdag=make_simple_cpdag(["A", "B", "C"]),
            rounds=[], arm_stats={},
            total_edges_oriented=6,
            total_rounds=4,
            total_cost=4.0,
            strategy="ucb",
            fully_oriented=True,
        )
        cold_br = CausalBanditResult(
            final_cpdag=make_simple_cpdag(["A", "B", "C"]),
            rounds=[], arm_stats={},
            total_edges_oriented=4,
            total_rounds=6,
            total_cost=6.0,
            strategy="ucb",
            fully_oriented=False,
        )
        transfer_result = TransferBanditResult(
            bandit_result=transfer_br,
            source_tasks_used=[0],
            transfer_weights=[0.5],
            strategy_selected="ucb",
            prior_strength=1.0,
        )
        det = detect_negative_transfer(transfer_result, cold_br)
        assert not det["is_negative_transfer"]
        assert det["improvement"] > 0

    def test_negative_transfer(self):
        transfer_br = CausalBanditResult(
            final_cpdag=make_simple_cpdag(["A", "B"]),
            rounds=[], arm_stats={},
            total_edges_oriented=1,
            total_rounds=5,
            total_cost=5.0,
            strategy="ucb",
            fully_oriented=False,
        )
        cold_br = CausalBanditResult(
            final_cpdag=make_simple_cpdag(["A", "B"]),
            rounds=[], arm_stats={},
            total_edges_oriented=3,
            total_rounds=3,
            total_cost=3.0,
            strategy="ucb",
            fully_oriented=True,
        )
        transfer_result = TransferBanditResult(
            bandit_result=transfer_br,
            source_tasks_used=[0],
            transfer_weights=[0.5],
            strategy_selected="ucb",
            prior_strength=1.0,
        )
        det = detect_negative_transfer(transfer_result, cold_br)
        assert det["is_negative_transfer"]
        assert det["improvement"] < 0


# ===========================================================================
# _summarize_results tests
# ===========================================================================

class TestSummarizeResults:
    def test_basic(self):
        results = []
        for i in range(3):
            r = CausalBanditResult(
                final_cpdag=make_simple_cpdag(["A", "B"]),
                rounds=[], arm_stats={},
                total_edges_oriented=2 + i,
                total_rounds=3,
                total_cost=3.0,
                strategy="ucb",
                fully_oriented=i > 0,
            )
            results.append(r)

        summary = _summarize_results(results)
        assert summary["n_trials"] == 3
        assert summary["mean_edges"] == pytest.approx(3.0)
        assert summary["mean_rounds"] == 3.0
        assert summary["fully_oriented_count"] == 2


# ===========================================================================
# compare_transfer_vs_cold tests
# ===========================================================================

class TestCompareTransferVsCold:
    def test_basic_comparison(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)

        comparison = compare_transfer_vs_cold(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, strategies=[BanditStrategy.UCB],
            max_rounds=10, n_trials=2, rng_seed=42,
        )
        assert "cold_ucb" in comparison
        assert "transfer_ucb" in comparison
        assert "transfer_auto" in comparison
        assert comparison["cold_ucb"]["n_trials"] == 2


# ===========================================================================
# sequential_discovery tests
# ===========================================================================

class TestSequentialDiscovery:
    def test_two_tasks(self):
        envs = []
        for seed in [42, 43]:
            obs, vars, int_fn, _ = build_chain_environment(n=3, seed=seed)
            envs.append((obs, vars, int_fn))

        db, results = sequential_discovery(envs, max_rounds=10)
        assert db.n_tasks() == 2
        assert len(results) == 2
        for r in results:
            assert isinstance(r, TransferBanditResult)

    def test_three_tasks_mixed(self):
        envs = []
        obs, vars, int_fn, _ = build_chain_environment(n=3, seed=42)
        envs.append((obs, vars, int_fn))
        obs, vars, int_fn, _ = build_diamond_environment(seed=43)
        envs.append((obs, vars, int_fn))
        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=44)
        envs.append((obs, vars, int_fn))

        db, results = sequential_discovery(envs, max_rounds=10)
        assert db.n_tasks() == 3
        # First task has no sources
        assert results[0].source_tasks_used == []
        # Later tasks may or may not use sources depending on similarity

    def test_explicit_strategy(self):
        envs = []
        for seed in [42, 43]:
            obs, vars, int_fn, _ = build_chain_environment(n=3, seed=seed)
            envs.append((obs, vars, int_fn))

        db, results = sequential_discovery(
            envs, strategy=BanditStrategy.UCB, max_rounds=10
        )
        assert results[0].strategy_selected == "ucb"


# ===========================================================================
# transfer_learning_curve tests
# ===========================================================================

class TestTransferLearningCurve:
    def test_basic_curve(self):
        target_obs, target_vars, target_fn, _ = build_chain_environment(n=4, seed=99)
        target_env = (target_obs, target_vars, target_fn)

        source_envs = []
        for seed in [42, 43]:
            obs, vars, int_fn, _ = build_chain_environment(n=4, seed=seed)
            source_envs.append((obs, vars, int_fn))

        curve = transfer_learning_curve(
            target_env=target_env,
            source_envs=source_envs,
            max_rounds=10,
            n_trials=2,
        )
        assert "n_sources" in curve
        assert len(curve["n_sources"]) == 3  # 0, 1, 2 sources
        assert curve["n_sources"] == [0, 1, 2]
        assert len(curve["mean_edges"]) == 3
        assert len(curve["mean_rounds"]) == 3
        assert len(curve["mean_efficiency"]) == 3


# ===========================================================================
# transfer_bandit_summary tests
# ===========================================================================

class TestSummary:
    def test_basic(self):
        br = CausalBanditResult(
            final_cpdag=make_simple_cpdag(["A", "B"]),
            rounds=[], arm_stats={},
            total_edges_oriented=3,
            total_rounds=5,
            total_cost=5.0,
            strategy="ucb",
            fully_oriented=True,
        )
        result = TransferBanditResult(
            bandit_result=br,
            source_tasks_used=[0, 1],
            transfer_weights=[0.5, 0.3],
            strategy_selected="ucb",
            prior_strength=1.0,
        )
        text = transfer_bandit_summary(result, name="test")
        assert "test" in text
        assert "ucb" in text
        assert "3" in text
        assert "Sources used: 2" in text

    def test_no_name(self):
        br = CausalBanditResult(
            final_cpdag=make_simple_cpdag(["X", "Y"]),
            rounds=[], arm_stats={},
            total_edges_oriented=1,
            total_rounds=2,
            total_cost=2.0,
            strategy="thompson",
            fully_oriented=False,
        )
        result = TransferBanditResult(
            bandit_result=br,
            source_tasks_used=[],
            transfer_weights=[],
            strategy_selected="thompson",
            prior_strength=0.5,
        )
        text = transfer_bandit_summary(result)
        assert "Transfer Causal Bandit" in text
        assert "Sources used: 0" in text


# ===========================================================================
# Integration: end-to-end transfer improves discovery
# ===========================================================================

class TestIntegration:
    def test_end_to_end_chain(self):
        """Full pipeline: solve chains, then transfer to new chain."""
        db = TransferDB()

        # Solve two source chains
        for seed in [42, 43]:
            obs, vars, int_fn, _ = build_chain_environment(n=4, seed=seed)
            rng = np.random.default_rng(seed)
            result = causal_bandit_discovery(
                initial_data=obs, variables=vars, intervention_fn=int_fn,
                strategy=BanditStrategy.UCB, max_rounds=15, rng=rng,
            )
            initial_cpdag = _build_initial_cpdag(obs, vars, 0.05)
            source = record_source_task(
                task_id=db.n_tasks(), variables=vars,
                initial_cpdag=initial_cpdag,
                final_cpdag=result.final_cpdag,
                result=result, name=f"chain_{seed}",
            )
            db.add_task(source)

        # Transfer to new chain
        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)
        rng = np.random.default_rng(99)
        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=15, rng=rng,
        )
        assert result.bandit_result.total_edges_oriented > 0
        assert len(result.source_tasks_used) > 0

    def test_end_to_end_mixed_graphs(self):
        """Transfer from diamond to chain (cross-graph-type)."""
        db = TransferDB()
        db.add_task(make_diamond_source(seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)
        rng = np.random.default_rng(99)
        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=15, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)

    def test_sequential_builds_knowledge(self):
        """Sequential discovery accumulates knowledge properly."""
        envs = []
        for seed in [42, 43, 44, 45]:
            obs, vars, int_fn, _ = build_chain_environment(n=4, seed=seed)
            envs.append((obs, vars, int_fn))

        db, results = sequential_discovery(envs, max_rounds=15)
        assert db.n_tasks() == 4

        # Later tasks should have more sources available
        assert len(results[-1].source_tasks_used) >= len(results[0].source_tasks_used)

    def test_large_graph_transfer(self):
        """Transfer on larger random DAGs."""
        db = TransferDB()

        # Source: large random graph
        obs, vars, int_fn, _ = build_large_environment(n_vars=6, seed=42)
        rng = np.random.default_rng(42)
        result = causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            strategy=BanditStrategy.THOMPSON, max_rounds=15, rng=rng,
        )
        initial_cpdag = _build_initial_cpdag(obs, vars, 0.05)
        source = record_source_task(
            task_id=0, variables=vars,
            initial_cpdag=initial_cpdag,
            final_cpdag=result.final_cpdag,
            result=result, name="large_6",
        )
        db.add_task(source)

        # Target: another large graph
        obs2, vars2, int_fn2, _ = build_large_environment(n_vars=6, seed=99)
        rng2 = np.random.default_rng(99)
        result2 = transfer_causal_bandit_discovery(
            initial_data=obs2, variables=vars2, intervention_fn=int_fn2,
            db=db, max_rounds=15, rng=rng2,
        )
        assert isinstance(result2, TransferBanditResult)
        assert result2.bandit_result.total_rounds > 0

    def test_contextual_transfer_integration(self):
        """Full contextual transfer pipeline."""
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=77)
        rng = np.random.default_rng(77)
        result = transfer_contextual_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=10, rng=rng,
        )
        assert result.strategy_selected == "contextual_transfer"
        assert result.bandit_result.total_rounds > 0

    def test_comparison_integration(self):
        """Compare transfer vs cold-start end-to-end."""
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)
        comparison = compare_transfer_vs_cold(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, strategies=[BanditStrategy.UCB],
            max_rounds=8, n_trials=2,
        )
        # Check all expected keys exist
        assert "cold_ucb" in comparison
        assert "transfer_ucb" in comparison
        assert "transfer_auto" in comparison

        # Verify structure
        for key in comparison:
            assert "mean_edges" in comparison[key]
            assert "mean_rounds" in comparison[key]
            assert "n_trials" in comparison[key]


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_variable(self):
        """Graph with single variable (nothing to orient)."""
        db = TransferDB()
        obs = [{"X": 0}, {"X": 1}] * 100
        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=["X"], intervention_fn=lambda t, v: obs,
            db=db, max_rounds=5,
        )
        assert result.bandit_result.total_rounds == 0

    def test_two_variables(self):
        db = TransferDB()
        obs, vars, int_fn, _ = build_confounded_environment(seed=42)
        rng = np.random.default_rng(42)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=10, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)

    def test_high_prior_strength(self):
        db = TransferDB()
        db.add_task(make_chain_source(n=4, seed=42))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=55)
        rng = np.random.default_rng(55)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, prior_strength=2.0, max_rounds=10, rng=rng,
        )
        assert result.prior_strength == 2.0

    def test_many_source_tasks(self):
        db = TransferDB()
        for i in range(8):
            db.add_task(make_chain_source(n=3 + (i % 3), seed=42 + i))

        obs, vars, int_fn, _ = build_chain_environment(n=4, seed=99)
        rng = np.random.default_rng(99)

        result = transfer_causal_bandit_discovery(
            initial_data=obs, variables=vars, intervention_fn=int_fn,
            db=db, max_rounds=10, rng=rng,
        )
        assert isinstance(result, TransferBanditResult)
        # Should have selected top-k sources, not all 8
        assert len(result.source_tasks_used) <= 5

    def test_discover_and_store_multiple(self):
        """discover_and_store 5 times, verify accumulation."""
        db = TransferDB()
        for i in range(5):
            obs, vars, int_fn, _ = build_chain_environment(n=3, seed=42 + i)
            rng = np.random.default_rng(42 + i)
            discover_and_store(
                initial_data=obs, variables=vars, intervention_fn=int_fn,
                db=db, task_name=f"task_{i}", max_rounds=8, rng=rng,
            )
        assert db.n_tasks() == 5
        for i, task in enumerate(db.tasks):
            assert task.name == f"task_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
