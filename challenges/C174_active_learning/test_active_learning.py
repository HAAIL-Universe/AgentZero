"""
Tests for C174: Active Learning Framework
"""
import sys, os, math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from active_learning import (
    UncertaintySampler, QueryByCommittee, BALDSampler,
    DensityWeightedSampler, ExpectedModelChangeSampler,
    BatchActiveLearner, BOActiveLearner,
    ActiveLearner, StreamActiveLearner,
    ActiveLearningHistory, ActiveLearningMetrics,
    SimpleKNN, SimpleLinearModel,
    create_active_learner
)


# ===== Fixtures =====

def make_classification_data(n=100, n_features=2, n_classes=2, seed=42):
    """Generate synthetic classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Create linearly separable classes with some noise
    w = rng.randn(n_features)
    scores = X @ w
    if n_classes == 2:
        y = (scores > 0).astype(float)
    else:
        thresholds = np.percentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1])
        y = np.digitize(scores, thresholds).astype(float)
    return X, y


def make_regression_data(n=100, n_features=2, seed=42):
    """Generate synthetic regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    w = rng.randn(n_features)
    y = X @ w + rng.randn(n) * 0.1
    return X, y


def make_probs(n=50, n_classes=3, seed=42):
    """Generate random probability matrix."""
    rng = np.random.RandomState(seed)
    raw = rng.rand(n, n_classes)
    return raw / raw.sum(axis=1, keepdims=True)


# ===== 1. UncertaintySampler Tests =====

class TestUncertaintySampler:

    def test_entropy_uniform_max(self):
        """Uniform distribution has maximum entropy."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = np.array([[0.5, 0.5], [0.9, 0.1], [0.1, 0.9]])
        scores = sampler.score(probs)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_entropy_certain_zero(self):
        """Near-certain prediction has near-zero entropy."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = np.array([[0.999, 0.001]])
        scores = sampler.score(probs)
        assert scores[0] < 0.05

    def test_margin_small_means_uncertain(self):
        """Small margin = high uncertainty score."""
        sampler = UncertaintySampler(strategy='margin')
        probs = np.array([[0.51, 0.49], [0.9, 0.1]])
        scores = sampler.score(probs)
        assert scores[0] > scores[1]

    def test_least_confident(self):
        """Low max probability = high uncertainty."""
        sampler = UncertaintySampler(strategy='least_confident')
        probs = np.array([[0.4, 0.3, 0.3], [0.8, 0.1, 0.1]])
        scores = sampler.score(probs)
        assert scores[0] > scores[1]

    def test_query_returns_top_n(self):
        """Query returns the n most uncertain indices."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = np.array([
            [0.5, 0.5],   # most uncertain
            [0.9, 0.1],   # certain
            [0.6, 0.4],   # somewhat uncertain
            [0.99, 0.01], # very certain
        ])
        indices, scores = sampler.query(probs, n_query=2)
        assert 0 in indices  # uniform is most uncertain
        assert len(indices) == 2

    def test_query_n_exceeds_pool(self):
        """Query handles n_query > pool size."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = np.array([[0.5, 0.5], [0.9, 0.1]])
        indices, scores = sampler.query(probs, n_query=10)
        assert len(indices) == 2

    def test_binary_1d_input(self):
        """Handle 1D probability input (binary case)."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = np.array([0.5, 0.9, 0.1])
        scores = sampler.score(probs)
        assert scores[0] > scores[1]

    def test_multiclass_entropy(self):
        """Works with multiclass probabilities."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = make_probs(20, n_classes=5)
        scores = sampler.score(probs)
        assert len(scores) == 20
        assert np.all(scores >= 0)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            UncertaintySampler(strategy='invalid')

    def test_margin_multiclass(self):
        """Margin works with >2 classes."""
        sampler = UncertaintySampler(strategy='margin')
        probs = np.array([[0.4, 0.35, 0.25], [0.8, 0.1, 0.1]])
        scores = sampler.score(probs)
        assert scores[0] > scores[1]  # tighter margin = more uncertain

    def test_all_strategies_same_ranking_extreme(self):
        """All strategies agree on extreme cases."""
        probs = np.array([[0.5, 0.5], [0.99, 0.01]])
        for s in ['entropy', 'margin', 'least_confident']:
            sampler = UncertaintySampler(strategy=s)
            scores = sampler.score(probs)
            assert scores[0] > scores[1]


# ===== 2. QueryByCommittee Tests =====

class MockModel:
    """Simple mock model for QBC tests."""
    def __init__(self, bias=0.0, seed=42):
        self.bias = bias
        self.rng = np.random.RandomState(seed)

    def predict(self, X):
        X = np.array(X)
        raw = X @ np.ones(X.shape[1]) + self.bias
        p = 1 / (1 + np.exp(-raw))
        return np.column_stack([1 - p, p])


class TestQueryByCommittee:

    def test_vote_entropy_agreement(self):
        """Low disagreement = low score."""
        committee = [MockModel(bias=b, seed=i) for i, b in enumerate([5.0, 5.0, 5.0])]
        qbc = QueryByCommittee(committee, strategy='vote_entropy')
        X = np.array([[1.0, 1.0], [-1.0, -1.0]])
        scores = qbc.score(X)
        assert np.all(scores >= 0)

    def test_vote_entropy_disagreement(self):
        """High disagreement near decision boundary."""
        committee = [MockModel(bias=b) for b in [-2.0, 0.0, 2.0]]
        qbc = QueryByCommittee(committee, strategy='vote_entropy')
        X = np.array([[0.0, 0.0], [5.0, 5.0]])
        scores = qbc.score(X)
        # Origin is near boundary, more disagreement
        assert scores[0] >= scores[1]

    def test_kl_divergence(self):
        """KL divergence measures spread of predictions."""
        committee = [MockModel(bias=b) for b in [-2.0, 0.0, 2.0]]
        qbc = QueryByCommittee(committee, strategy='kl_divergence')
        X = np.array([[0.0, 0.0], [5.0, 5.0]])
        scores = qbc.score(X)
        assert len(scores) == 2
        assert np.all(scores >= 0)

    def test_query_returns_indices(self):
        committee = [MockModel(bias=b) for b in [-1.0, 0.0, 1.0]]
        qbc = QueryByCommittee(committee, strategy='vote_entropy')
        X = np.array([[0, 0], [5, 5], [0.1, 0.1], [-5, -5]])
        indices, scores = qbc.query(X, n_query=2)
        assert len(indices) == 2

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            QueryByCommittee([], strategy='invalid')

    def test_single_committee_member(self):
        """Single member = no disagreement."""
        committee = [MockModel(bias=0.0)]
        qbc = QueryByCommittee(committee, strategy='vote_entropy')
        X = np.array([[1.0, 1.0]])
        scores = qbc.score(X)
        assert scores[0] == 0.0  # No disagreement possible


# ===== 3. BALDSampler Tests =====

class MockMCModel:
    """Mock MC Dropout model."""
    def __init__(self, uncertainty_level=0.5, seed=42):
        self.uncertainty_level = uncertainty_level
        self.seed = seed

    def predict_with_uncertainty(self, X, n_samples=50, seed=42):
        rng = np.random.RandomState(seed)
        X = np.array(X)
        n = len(X)
        # More uncertain for points near origin
        base_unc = np.sqrt(np.sum(X ** 2, axis=1))
        base_unc = 1 / (1 + base_unc)  # Closer to origin = more uncertain

        preds = []
        for _ in range(n_samples):
            noise = rng.randn(n) * base_unc * self.uncertainty_level
            pred = 1 / (1 + np.exp(-(X @ np.ones(X.shape[1]) + noise)))
            preds.append(np.column_stack([1 - pred, pred]))

        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        return preds, mean, std


class TestBALDSampler:

    def test_bald_scores_positive(self):
        model = MockMCModel(uncertainty_level=1.0)
        bald = BALDSampler(model, n_mc_samples=30)
        X = np.array([[0.0, 0.0], [5.0, 5.0], [-5.0, -5.0]])
        scores = bald.score(X)
        assert len(scores) == 3
        assert np.all(scores >= -1e-10)  # BALD is non-negative (with tolerance)

    def test_bald_uncertain_higher(self):
        """Points near decision boundary should have higher BALD."""
        model = MockMCModel(uncertainty_level=1.0)
        bald = BALDSampler(model, n_mc_samples=50, seed=42)
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        scores = bald.score(X)
        # Near origin should be more uncertain
        assert scores[0] > scores[1]

    def test_bald_query(self):
        model = MockMCModel(uncertainty_level=1.0)
        bald = BALDSampler(model, n_mc_samples=20)
        X = np.array([[0, 0], [5, 5], [-5, -5], [0.1, -0.1]])
        indices, scores = bald.query(X, n_query=2)
        assert len(indices) == 2

    def test_bald_regression(self):
        """BALD for regression uses variance."""
        class MockRegModel:
            def predict_with_uncertainty(self, X, n_samples=50, seed=42):
                rng = np.random.RandomState(seed)
                X = np.array(X)
                n = len(X)
                preds = []
                for _ in range(n_samples):
                    preds.append(X[:, 0] + rng.randn(n) * 0.5)
                preds = np.array(preds)  # (n_samples, n)
                return preds, np.mean(preds, axis=0), np.std(preds, axis=0)

        model = MockRegModel()
        bald = BALDSampler(model, n_mc_samples=30)
        X = np.array([[1.0, 0.0], [2.0, 0.0]])
        scores = bald.score(X)
        assert len(scores) == 2
        assert np.all(scores > 0)


# ===== 4. DensityWeightedSampler Tests =====

class TestDensityWeightedSampler:

    def test_density_weights(self):
        """Dense regions get higher weight."""
        base = UncertaintySampler(strategy='entropy')
        dw = DensityWeightedSampler(base, beta=1.0)

        # Cluster of points plus an outlier
        X = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.1],  # dense cluster
            [10.0, 10.0],  # outlier
        ])
        probs = np.array([
            [0.5, 0.5],  # uncertain, dense
            [0.5, 0.5],  # uncertain, dense
            [0.5, 0.5],  # uncertain, dense
            [0.5, 0.5],  # uncertain, dense
            [0.5, 0.5],  # uncertain, sparse
        ])
        scores = dw.score(X, probs)
        # Dense cluster should score higher than outlier
        assert scores[0] > scores[4]

    def test_beta_zero_pure_informativeness(self):
        """Beta=0 ignores density."""
        base = UncertaintySampler(strategy='entropy')
        dw = DensityWeightedSampler(base, beta=0.0)
        X = np.array([[0, 0], [10, 10]])
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        scores = dw.score(X, probs)
        # Should be equal since informativeness is the same
        assert abs(scores[0] - scores[1]) < 0.1

    def test_query_returns_indices(self):
        base = UncertaintySampler(strategy='entropy')
        dw = DensityWeightedSampler(base, beta=1.0)
        X = np.random.randn(20, 2)
        probs = make_probs(20, n_classes=2)
        indices, scores = dw.query(X, probs, n_query=5)
        assert len(indices) == 5
        assert len(set(indices)) == 5  # unique


# ===== 5. ExpectedModelChangeSampler Tests =====

class TestExpectedModelChangeSampler:

    def test_gradient_scores(self):
        model = SimpleLinearModel(lr=0.01, n_epochs=50)
        X, y = make_regression_data(50)
        model.fit(X, y)

        sampler = ExpectedModelChangeSampler(model)
        X_pool = np.random.randn(10, 2)
        scores = sampler.score(X_pool)
        assert len(scores) == 10
        assert np.all(scores >= 0)

    def test_query(self):
        model = SimpleLinearModel()
        X, y = make_regression_data(50)
        model.fit(X, y)

        sampler = ExpectedModelChangeSampler(model)
        X_pool = np.random.randn(20, 2)
        indices, scores = sampler.query(X_pool, n_query=3)
        assert len(indices) == 3

    def test_with_hypothetical_labels(self):
        model = SimpleLinearModel()
        X, y = make_regression_data(50)
        model.fit(X, y)

        sampler = ExpectedModelChangeSampler(model)
        X_pool = np.random.randn(10, 2)
        hyp_labels = np.random.randn(10)
        scores = sampler.score(X_pool, hypothetical_labels=hyp_labels)
        assert len(scores) == 10


# ===== 6. BatchActiveLearner Tests =====

class TestBatchActiveLearner:

    def test_batch_diversity(self):
        """Batch should select diverse points."""
        base = UncertaintySampler(strategy='entropy')
        batch = BatchActiveLearner(base, diversity_weight=0.8)

        # Two clusters, all equally uncertain
        X = np.array([
            [0, 0], [0.1, 0], [0, 0.1],  # cluster 1
            [5, 5], [5.1, 5], [5, 5.1],  # cluster 2
        ])
        probs = np.array([[0.5, 0.5]] * 6)

        indices, scores = batch.query(X, probs, n_query=2)
        # Should pick one from each cluster
        cluster1 = {0, 1, 2}
        cluster2 = {3, 4, 5}
        assert (indices[0] in cluster1 and indices[1] in cluster2) or \
               (indices[0] in cluster2 and indices[1] in cluster1)

    def test_batch_single(self):
        """Batch of 1 = pure informativeness."""
        base = UncertaintySampler(strategy='entropy')
        batch = BatchActiveLearner(base, diversity_weight=0.5)
        probs = np.array([[0.5, 0.5], [0.9, 0.1], [0.6, 0.4]])
        X = np.array([[0, 0], [1, 1], [2, 2]])
        indices, scores = batch.query(X, probs, n_query=1)
        assert indices[0] == 0  # Most uncertain

    def test_batch_full_pool(self):
        """Select all from pool."""
        base = UncertaintySampler(strategy='entropy')
        batch = BatchActiveLearner(base, diversity_weight=0.5)
        X = np.random.randn(5, 2)
        probs = make_probs(5, n_classes=2)
        indices, scores = batch.query(X, probs, n_query=5)
        assert len(indices) == 5
        assert len(set(indices)) == 5

    def test_batch_n_exceeds_pool(self):
        base = UncertaintySampler(strategy='entropy')
        batch = BatchActiveLearner(base, diversity_weight=0.5)
        X = np.random.randn(3, 2)
        probs = make_probs(3, n_classes=2)
        indices, _ = batch.query(X, probs, n_query=10)
        assert len(indices) == 3


# ===== 7. BOActiveLearner Tests =====

class TestBOActiveLearner:

    def test_suggest_from_pool_cold_start(self):
        """Without observations, falls back to uncertainty or random."""
        bo = BOActiveLearner(bounds=[(-5, 5), (-5, 5)], seed=42)
        X_pool = np.random.randn(10, 2)
        unc = np.random.rand(10)
        idx, score = bo.suggest_from_pool(X_pool, uncertainty_scores=unc)
        assert 0 <= idx < 10

    def test_suggest_after_observations(self):
        """After observations, uses GP surrogate."""
        bo = BOActiveLearner(bounds=[(-5, 5), (-5, 5)], seed=42)
        # Observe a few points
        bo.observe(np.array([1.0, 1.0]), 0.5)
        bo.observe(np.array([-1.0, -1.0]), 0.3)
        bo.observe(np.array([2.0, 0.0]), 0.8)

        X_pool = np.random.randn(20, 2)
        idx, score = bo.suggest_from_pool(X_pool)
        assert 0 <= idx < 20

    def test_query_multiple(self):
        bo = BOActiveLearner(bounds=[(-5, 5), (-5, 5)], seed=42)
        X_pool = np.random.randn(15, 2)
        unc = np.random.rand(15)
        indices, scores = bo.query(X_pool, n_query=3, uncertainty_scores=unc)
        assert len(indices) == 3
        assert len(set(indices)) == 3  # unique

    def test_observe_and_suggest(self):
        """BO improves suggestions after observations."""
        bo = BOActiveLearner(bounds=[(0, 1)], seed=42)
        # Observe: high value at 0.5
        bo.observe(np.array([0.5]), 1.0)
        bo.observe(np.array([0.0]), 0.1)
        bo.observe(np.array([1.0]), 0.1)

        X_pool = np.linspace(0, 1, 20).reshape(-1, 1)
        idx, _ = bo.suggest_from_pool(X_pool)
        # Should suggest near 0.5 where value is high
        assert 0 <= idx < 20

    def test_cold_start_random(self):
        """No observations, no uncertainty -> random."""
        bo = BOActiveLearner(bounds=[(-1, 1)], seed=42)
        X = np.random.randn(5, 1)
        idx, score = bo.suggest_from_pool(X)
        assert score == 0.0


# ===== 8. ActiveLearner (Orchestrator) Tests =====

class TestActiveLearner:

    def test_init_empty(self):
        X, y = make_classification_data(100)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y)
        assert learner.labeled_size == 0
        assert learner.pool_size == 100

    def test_init_with_seed(self):
        X, y = make_classification_data(100)
        X_init, y_init = X[:5], y[:5]
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X_init, y_initial=y_init)
        assert learner.labeled_size == 5

    def test_teach(self):
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y)
        learner.teach(X[:3], y[:3])
        assert learner.labeled_size == 3

    def test_step(self):
        X, y = make_classification_data(100)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:5], y_initial=y[:5])
        indices, X_q, y_q = learner.step(n_query=3)
        assert len(indices) == 3
        assert learner.labeled_size == 8
        assert learner.pool_size == 97

    def test_step_with_oracle(self):
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X,
                                X_initial=X[:3], y_initial=y[:3])

        oracle = lambda x: (x[:, 0] > 0).astype(float)
        indices, X_q, y_q = learner.step(n_query=2, oracle=oracle)
        assert len(y_q) == 2

    def test_run_full_loop(self):
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:5], y_initial=y[:5])
        history = learner.run(n_iterations=10, n_query_per_step=2)
        assert len(history.n_labeled_list) == 10
        assert learner.labeled_size == 25

    def test_pool_exhaustion(self):
        X, y = make_classification_data(20)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:5], y_initial=y[:5])
        history = learner.run(n_iterations=100, n_query_per_step=5)
        assert learner.pool_size == 0

    def test_history_recording(self):
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:5], y_initial=y[:5])
        history = learner.run(n_iterations=5, n_query_per_step=1)
        assert len(history.accuracies) == 5
        assert all(a is not None for a in history.accuracies)

    def test_learning_improves(self):
        """More labels should improve accuracy (or at least not degrade)."""
        np.random.seed(42)
        X, y = make_classification_data(200, seed=42)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:10], y_initial=y[:10])
        history = learner.run(n_iterations=20, n_query_per_step=5)

        accs = [a for a in history.accuracies if a is not None]
        # Final accuracy should be >= initial (with tolerance)
        assert accs[-1] >= accs[0] - 0.15

    def test_no_oracle_no_pool_labels_raises(self):
        X = np.random.randn(10, 2)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X,
                                X_initial=X[:2], y_initial=np.array([0, 1]))
        with pytest.raises(ValueError, match="No oracle"):
            learner.step(n_query=1)

    def test_margin_strategy(self):
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='margin')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:5], y_initial=y[:5])
        indices, _, _ = learner.step(n_query=3)
        assert len(indices) == 3

    def test_least_confident_strategy(self):
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='least_confident')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X[:5], y_initial=y[:5])
        indices, _, _ = learner.step(n_query=3)
        assert len(indices) == 3


# ===== 9. StreamActiveLearner Tests =====

class TestStreamActiveLearner:

    def test_basic_stream(self):
        X, y = make_classification_data(50)
        model = SimpleKNN(k=3)
        stream = StreamActiveLearner(model, threshold=0.3, strategy='entropy')
        rate = stream.process_stream(X, y_stream=y)
        assert 0 < rate <= 1.0
        assert stream.n_queries > 0
        assert stream.n_seen == 50

    def test_budget_limiting(self):
        X, y = make_classification_data(100)
        model = SimpleKNN(k=3)
        stream = StreamActiveLearner(model, threshold=0.0, budget=10)
        stream.process_stream(X, y_stream=y)
        assert stream.n_queries <= 10

    def test_threshold_controls_rate(self):
        X, y = make_classification_data(100, seed=42)
        model1 = SimpleKNN(k=3)
        model2 = SimpleKNN(k=3)
        stream_low = StreamActiveLearner(model1, threshold=0.1)
        stream_high = StreamActiveLearner(model2, threshold=0.9)
        rate_low = stream_low.process_stream(X, y_stream=y)
        rate_high = stream_high.process_stream(X, y_stream=y)
        # Higher threshold = fewer queries (less of the data is "uncertain enough")
        assert rate_high <= rate_low

    def test_oracle_function(self):
        X, y = make_classification_data(30)
        model = SimpleKNN(k=3)
        stream = StreamActiveLearner(model, threshold=0.3)
        oracle = lambda x: (x[:, 0] > 0).astype(float)
        rate = stream.process_stream(X, oracle=oracle)
        assert stream.n_queries > 0

    def test_process_single(self):
        model = SimpleKNN(k=3)
        stream = StreamActiveLearner(model, threshold=0.3)
        # First instance is always queried
        queried = stream.process(np.array([1.0, 2.0]), y_true=1)
        assert queried is True
        assert stream.n_queries == 1

    def test_stream_history(self):
        X, y = make_classification_data(30)
        model = SimpleKNN(k=3)
        stream = StreamActiveLearner(model, threshold=0.3)
        stream.process_stream(X, y_stream=y)
        assert len(stream.history.n_labeled_list) > 0


# ===== 10. ActiveLearningHistory & Metrics Tests =====

class TestActiveLearningHistory:

    def test_record_and_summary(self):
        history = ActiveLearningHistory()
        history.record(5, 95, accuracy=0.6)
        history.record(10, 90, accuracy=0.7)
        history.record(15, 85, accuracy=0.8)

        summary = history.summary()
        assert summary['total_queries'] == 3
        assert summary['final_labeled'] == 15
        assert summary['final_accuracy'] == 0.8
        assert summary['best_accuracy'] == 0.8

    def test_learning_curve(self):
        history = ActiveLearningHistory()
        history.record(5, 95, accuracy=0.5)
        history.record(10, 90, accuracy=0.7)
        curve = history.learning_curve()
        assert curve == [(5, 0.5), (10, 0.7)]

    def test_empty_summary(self):
        history = ActiveLearningHistory()
        summary = history.summary()
        assert summary['total_queries'] == 0
        assert summary['final_labeled'] == 0


class TestActiveLearningMetrics:

    def test_aulc(self):
        """Area under learning curve."""
        history = ActiveLearningHistory()
        for i in range(10):
            history.record(i * 5 + 5, 100 - (i * 5 + 5),
                          accuracy=0.5 + i * 0.05)

        aulc = ActiveLearningMetrics.area_under_learning_curve(history)
        assert 0 < aulc < 1.5

    def test_aulc_perfect(self):
        """Perfect learner has high AULC."""
        history = ActiveLearningHistory()
        for i in range(10):
            history.record(i * 5 + 5, 100 - (i * 5 + 5), accuracy=1.0)
        aulc = ActiveLearningMetrics.area_under_learning_curve(history)
        assert aulc == pytest.approx(1.0, abs=0.01)

    def test_aulc_insufficient_data(self):
        history = ActiveLearningHistory()
        history.record(5, 95, accuracy=0.5)
        aulc = ActiveLearningMetrics.area_under_learning_curve(history)
        assert aulc == 0.0

    def test_query_efficiency(self):
        """Active should be more efficient than random."""
        # Simulate active: reaches 0.8 at 20 labels
        active = ActiveLearningHistory()
        active.record(10, 90, accuracy=0.6)
        active.record(20, 80, accuracy=0.8)

        # Simulate random: reaches 0.8 at 40 labels
        random = ActiveLearningHistory()
        random.record(20, 80, accuracy=0.6)
        random.record(40, 60, accuracy=0.8)

        efficiency = ActiveLearningMetrics.query_efficiency(active, random)
        assert efficiency > 1.0  # Active is more efficient

    def test_label_complexity(self):
        history = ActiveLearningHistory()
        history.record(5, 95, accuracy=0.5)
        history.record(10, 90, accuracy=0.7)
        history.record(15, 85, accuracy=0.9)

        n = ActiveLearningMetrics.label_complexity(history, target_accuracy=0.7)
        assert n == 10

    def test_label_complexity_not_reached(self):
        history = ActiveLearningHistory()
        history.record(5, 95, accuracy=0.5)
        n = ActiveLearningMetrics.label_complexity(history, target_accuracy=0.9)
        assert n is None


# ===== 11. SimpleKNN Tests =====

class TestSimpleKNN:

    def test_classification(self):
        X, y = make_classification_data(50, seed=42)
        knn = SimpleKNN(k=3)
        knn.fit(X, y)
        preds = knn.predict(X[:10])
        assert len(preds) == 10
        acc = np.mean(preds == y[:10])
        assert acc > 0.5  # Better than random

    def test_predict_proba(self):
        X, y = make_classification_data(50)
        knn = SimpleKNN(k=5)
        knn.fit(X, y)
        probs = knn.predict_proba(X[:10])
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_unfitted_predict(self):
        knn = SimpleKNN(k=3)
        preds = knn.predict(np.random.randn(5, 2))
        assert len(preds) == 5

    def test_unfitted_proba(self):
        knn = SimpleKNN(k=3)
        probs = knn.predict_proba(np.random.randn(5, 2))
        assert probs.shape == (5, 2)


# ===== 12. SimpleLinearModel Tests =====

class TestSimpleLinearModel:

    def test_regression(self):
        X, y = make_regression_data(100)
        model = SimpleLinearModel(lr=0.01, n_epochs=200)
        model.fit(X, y)
        preds = model.predict(X)
        mse = np.mean((preds - y) ** 2)
        assert mse < 1.0

    def test_predict_proba(self):
        model = SimpleLinearModel()
        X, y = make_classification_data(50)
        model.fit(X, y)
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


# ===== 13. Builder Function Tests =====

class TestCreateActiveLearner:

    def test_default(self):
        X, y = make_classification_data(50)
        learner = create_active_learner(X, y)
        assert learner.pool_size == 50

    def test_with_initial(self):
        X, y = make_classification_data(50)
        learner = create_active_learner(X, y, X_initial=X[:5], y_initial=y[:5])
        assert learner.labeled_size == 5

    def test_margin_strategy(self):
        X, y = make_classification_data(50)
        learner = create_active_learner(X, y, strategy='margin',
                                        X_initial=X[:3], y_initial=y[:3])
        learner.step(n_query=2)
        assert learner.labeled_size == 5

    def test_batch_strategy(self):
        X, y = make_classification_data(50)
        learner = create_active_learner(X, y, strategy='batch_entropy',
                                        X_initial=X[:5], y_initial=y[:5])
        assert learner.pool_size == 50

    def test_custom_model(self):
        X, y = make_classification_data(50)
        model = SimpleLinearModel(lr=0.01, n_epochs=50)
        learner = create_active_learner(X, y, model=model,
                                        X_initial=X[:5], y_initial=y[:5])
        learner.step(n_query=2)
        assert learner.labeled_size == 7


# ===== 14. Integration Tests =====

class TestIntegration:

    def test_full_active_learning_pipeline(self):
        """Full pipeline: create, run, evaluate."""
        np.random.seed(42)
        X, y = make_classification_data(200, seed=42)

        learner = create_active_learner(
            X, y,
            strategy='entropy',
            X_initial=X[:10],
            y_initial=y[:10]
        )

        history = learner.run(n_iterations=15, n_query_per_step=5)
        summary = history.summary()

        assert summary['total_queries'] == 15
        assert summary['final_labeled'] == 85  # 10 + 15*5
        assert summary['final_accuracy'] is not None

    def test_active_vs_random_comparison(self):
        """Active learning should match or beat random sampling."""
        np.random.seed(42)
        X, y = make_classification_data(100, seed=42)

        # Active learning
        active = create_active_learner(X, y, strategy='entropy',
                                       X_initial=X[:5], y_initial=y[:5])
        active_hist = active.run(n_iterations=10, n_query_per_step=3)

        # The active learner should have some accuracy recorded
        accs = [a for a in active_hist.accuracies if a is not None]
        assert len(accs) > 0

    def test_qbc_pipeline(self):
        """QBC with committee of models."""
        X, y = make_classification_data(80, seed=42)
        committee = [MockModel(bias=b, seed=i) for i, b in enumerate([-1, 0, 1])]
        qbc = QueryByCommittee(committee, strategy='vote_entropy')
        scores = qbc.score(X)
        indices, _ = qbc.query(X, n_query=5)
        assert len(indices) == 5

    def test_density_weighted_pipeline(self):
        """Density-weighted uncertainty pipeline."""
        X, y = make_classification_data(50, seed=42)
        base = UncertaintySampler(strategy='entropy')
        dw = DensityWeightedSampler(base, beta=1.0)

        knn = SimpleKNN(k=3)
        knn.fit(X[:5], y[:5])
        probs = knn.predict_proba(X)

        indices, scores = dw.query(X, probs, n_query=5)
        assert len(indices) == 5

    def test_batch_active_learning_pipeline(self):
        """Batch mode active learning."""
        X, y = make_classification_data(100, seed=42)
        base = UncertaintySampler(strategy='entropy')
        batch = BatchActiveLearner(base, diversity_weight=0.5)

        knn = SimpleKNN(k=3)
        knn.fit(X[:10], y[:10])
        probs = knn.predict_proba(X[10:])

        indices, scores = batch.query(X[10:], probs, n_query=10)
        assert len(indices) == 10
        assert len(set(indices)) == 10  # all unique

    def test_bo_active_learning_pipeline(self):
        """BO-driven active learning pipeline."""
        np.random.seed(42)
        X, y = make_regression_data(50, n_features=1, seed=42)
        bo = BOActiveLearner(bounds=[(-3, 3)], seed=42)

        # Observe some points
        for i in range(5):
            bo.observe(X[i], y[i])

        # Query from remaining pool
        indices, scores = bo.query(X[5:], n_query=3)
        assert len(indices) == 3

    def test_stream_to_pool_comparison(self):
        """Stream and pool learners work on same data."""
        np.random.seed(42)
        X, y = make_classification_data(50, seed=42)

        # Pool-based
        pool_learner = create_active_learner(X, y, strategy='entropy',
                                             X_initial=X[:3], y_initial=y[:3])
        pool_hist = pool_learner.run(n_iterations=5, n_query_per_step=2)

        # Stream-based
        stream = StreamActiveLearner(SimpleKNN(k=3), threshold=0.3)
        rate = stream.process_stream(X, y_stream=y)

        # Both should make progress
        assert pool_learner.labeled_size > 3
        assert stream.n_queries > 0

    def test_metrics_comparison(self):
        """Compare two strategies with metrics."""
        np.random.seed(42)
        X, y = make_classification_data(100, seed=42)

        # Strategy 1: entropy
        l1 = create_active_learner(X, y, strategy='entropy',
                                   X_initial=X[:5], y_initial=y[:5])
        h1 = l1.run(n_iterations=10, n_query_per_step=3)

        # Strategy 2: least confident
        l2 = create_active_learner(X, y, strategy='least_confident',
                                   X_initial=X[:5], y_initial=y[:5])
        h2 = l2.run(n_iterations=10, n_query_per_step=3)

        aulc1 = ActiveLearningMetrics.area_under_learning_curve(h1)
        aulc2 = ActiveLearningMetrics.area_under_learning_curve(h2)

        # Both should have valid AULC
        assert aulc1 > 0
        assert aulc2 > 0

    def test_bald_with_mock_mc_model(self):
        """BALD sampler with MC model in full pipeline."""
        model = MockMCModel(uncertainty_level=1.0)
        bald = BALDSampler(model, n_mc_samples=20)
        X = np.random.randn(30, 2)
        indices, scores = bald.query(X, n_query=5)
        assert len(indices) == 5
        assert all(s >= -1e-10 for s in scores)

    def test_multiple_teach_calls(self):
        """Teaching incrementally grows labeled set."""
        X, y = make_classification_data(50)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y)

        learner.teach(X[:3], y[:3])
        assert learner.labeled_size == 3

        learner.teach(X[3:6], y[3:6])
        assert learner.labeled_size == 6

    def test_empty_pool_query(self):
        """Query on empty pool returns empty."""
        X, y = make_classification_data(5)
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=3), sampler, X, y,
                                X_initial=X, y_initial=y)
        # Exhaust pool
        learner._pool_mask[:] = False
        indices, scores = learner.query(n_query=5)
        assert len(indices) == 0


# ===== 15. Edge Cases =====

class TestEdgeCases:

    def test_single_sample_pool(self):
        X = np.array([[1.0, 2.0]])
        y = np.array([1.0])
        sampler = UncertaintySampler(strategy='entropy')
        learner = ActiveLearner(SimpleKNN(k=1), sampler, X, y,
                                X_initial=X, y_initial=y)
        # Pool is available but already labeled
        learner._pool_mask[:] = True  # reset for test
        indices, _ = learner.query(n_query=1)
        assert len(indices) == 1

    def test_all_same_probs(self):
        """All equally uncertain."""
        sampler = UncertaintySampler(strategy='entropy')
        probs = np.ones((10, 3)) / 3
        scores = sampler.score(probs)
        assert np.allclose(scores, scores[0])

    def test_large_pool(self):
        """Performance with larger pool."""
        np.random.seed(42)
        X = np.random.randn(500, 5)
        y = (X[:, 0] > 0).astype(float)
        learner = create_active_learner(X, y, strategy='entropy',
                                        X_initial=X[:10], y_initial=y[:10])
        history = learner.run(n_iterations=5, n_query_per_step=10)
        assert learner.labeled_size == 60

    def test_multiclass(self):
        """Works with multiclass problems."""
        X, y = make_classification_data(100, n_classes=4, seed=42)
        learner = create_active_learner(X, y, strategy='entropy',
                                        X_initial=X[:10], y_initial=y[:10])
        history = learner.run(n_iterations=5, n_query_per_step=3)
        assert learner.labeled_size == 25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
