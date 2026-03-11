"""
C174: Active Learning Framework
Composing C166 (Bayesian Neural Network) + C167 (Bayesian Optimization)

Components:
1. UncertaintySampler -- entropy, margin, least confident query strategies
2. QueryByCommittee -- committee disagreement (vote entropy, KL divergence)
3. BALDSampler -- Bayesian Active Learning by Disagreement (uses C166)
4. DensityWeightedSampler -- informativeness x representativeness
5. ExpectedModelChangeSampler -- expected gradient length
6. BatchActiveLearner -- batch mode with diversity (k-medoids)
7. BOActiveLearner -- Bayesian optimization-driven (uses C167)
8. ActiveLearner -- main pool-based orchestrator
9. StreamActiveLearner -- stream-based variant
10. ActiveLearningMetrics -- tracking and evaluation
"""

import sys, os, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C166_bayesian_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C167_bayesian_optimization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from bayesian_neural_network import (
    MCDropoutNetwork, BayesianNetwork, BayesByBackprop,
    BNNPredictive, build_bnn, build_mc_dropout_model
)
from bayesian_optimization import (
    BayesianOptimizer, ExpectedImprovement, UpperConfidenceBound,
    create_optimizer
)


# ---------------------------------------------------------------------------
# 1. UncertaintySampler
# ---------------------------------------------------------------------------
class UncertaintySampler:
    """Query strategies based on prediction uncertainty."""

    def __init__(self, strategy='entropy'):
        """
        Args:
            strategy: 'entropy', 'margin', 'least_confident'
        """
        if strategy not in ('entropy', 'margin', 'least_confident'):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

    def score(self, probs):
        """Score unlabeled samples by uncertainty.

        Args:
            probs: (n_samples, n_classes) probability matrix

        Returns:
            scores: (n_samples,) higher = more uncertain
        """
        probs = np.array(probs)
        if probs.ndim == 1:
            # Binary: convert to 2-class
            probs = np.column_stack([1 - probs, probs])

        # Clip for numerical stability
        probs = np.clip(probs, 1e-10, 1.0)

        if self.strategy == 'entropy':
            return -np.sum(probs * np.log(probs), axis=1)

        elif self.strategy == 'margin':
            sorted_probs = np.sort(probs, axis=1)
            # Small margin = high uncertainty
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            return 1.0 - margin

        elif self.strategy == 'least_confident':
            return 1.0 - np.max(probs, axis=1)

    def query(self, probs, n_query=1):
        """Select indices of most uncertain samples.

        Args:
            probs: (n_samples, n_classes) probability matrix
            n_query: number of samples to select

        Returns:
            indices: array of selected indices
            scores: uncertainty scores for selected
        """
        scores = self.score(probs)
        n_query = min(n_query, len(scores))
        indices = np.argsort(scores)[-n_query:][::-1]
        return indices, scores[indices]


# ---------------------------------------------------------------------------
# 2. QueryByCommittee
# ---------------------------------------------------------------------------
class QueryByCommittee:
    """Committee-based active learning using model disagreement."""

    def __init__(self, committee, strategy='vote_entropy'):
        """
        Args:
            committee: list of models (each has predict(x) returning class probs)
            strategy: 'vote_entropy' or 'kl_divergence'
        """
        self.committee = committee
        if strategy not in ('vote_entropy', 'kl_divergence'):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy

    def score(self, X):
        """Score samples by committee disagreement.

        Args:
            X: input data (n_samples, n_features)

        Returns:
            scores: (n_samples,) higher = more disagreement
        """
        predictions = []
        for model in self.committee:
            pred = model.predict(X)
            pred = np.array(pred)
            if pred.ndim == 1:
                pred = np.column_stack([1 - pred, pred])
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_committee, n_samples, n_classes)

        if self.strategy == 'vote_entropy':
            return self._vote_entropy(predictions)
        else:
            return self._kl_divergence(predictions)

    def _vote_entropy(self, predictions):
        """Entropy of the vote distribution."""
        n_committee = len(predictions)
        # Hard votes: argmax for each committee member
        votes = np.argmax(predictions, axis=2)  # (n_committee, n_samples)
        n_samples = votes.shape[1]
        n_classes = predictions.shape[2]

        scores = np.zeros(n_samples)
        for i in range(n_samples):
            counts = np.bincount(votes[:, i], minlength=n_classes)
            probs = counts / n_committee
            probs = probs[probs > 0]
            scores[i] = -np.sum(probs * np.log(probs))
        return scores

    def _kl_divergence(self, predictions):
        """Average KL divergence from consensus."""
        # Consensus = mean prediction
        consensus = np.mean(predictions, axis=0)  # (n_samples, n_classes)
        consensus = np.clip(consensus, 1e-10, 1.0)

        n_committee = len(predictions)
        n_samples = predictions.shape[1]
        scores = np.zeros(n_samples)

        for m in range(n_committee):
            member_pred = np.clip(predictions[m], 1e-10, 1.0)
            kl = np.sum(member_pred * np.log(member_pred / consensus), axis=1)
            scores += kl

        return scores / n_committee

    def query(self, X, n_query=1):
        """Select indices of samples with most disagreement."""
        scores = self.score(X)
        n_query = min(n_query, len(scores))
        indices = np.argsort(scores)[-n_query:][::-1]
        return indices, scores[indices]


# ---------------------------------------------------------------------------
# 3. BALDSampler (Bayesian Active Learning by Disagreement)
# ---------------------------------------------------------------------------
class BALDSampler:
    """BALD: mutual information between predictions and model params.
    Uses MC Dropout or BNN for uncertainty estimation.
    """

    def __init__(self, model, n_mc_samples=50, seed=42):
        """
        Args:
            model: MCDropoutNetwork or BNN with stochastic forward passes
            n_mc_samples: number of MC forward passes
            seed: random seed
        """
        self.model = model
        self.n_mc_samples = n_mc_samples
        self.seed = seed

    def score(self, X):
        """Compute BALD score (mutual information).

        For classification: H[y|x] - E_w[H[y|x,w]]
        For regression: uses predictive variance decomposition

        Args:
            X: input data

        Returns:
            scores: (n_samples,) BALD scores
        """
        X = np.array(X)

        if hasattr(self.model, 'predict_with_uncertainty'):
            # MCDropoutNetwork or LaplaceApproximation
            preds, mean, std = self.model.predict_with_uncertainty(
                X, n_samples=self.n_mc_samples, seed=self.seed
            )
            # preds shape: (n_mc_samples, n_samples, n_outputs) or (n_mc_samples, n_samples)
            preds = np.array(preds)

            if preds.ndim == 2:
                # Regression: use variance as score
                return np.var(preds, axis=0)

            # Classification: BALD = H[E[p]] - E[H[p]]
            mean_pred = np.mean(preds, axis=0)  # (n_samples, n_classes)
            mean_pred = np.clip(mean_pred, 1e-10, 1.0)

            # Total entropy
            total_entropy = -np.sum(mean_pred * np.log(mean_pred), axis=-1)

            # Expected entropy
            preds_clipped = np.clip(preds, 1e-10, 1.0)
            per_sample_entropy = -np.sum(preds_clipped * np.log(preds_clipped), axis=-1)
            expected_entropy = np.mean(per_sample_entropy, axis=0)

            return total_entropy - expected_entropy

        elif hasattr(self.model, 'predict') and isinstance(self.model, BNNPredictive):
            # BNNPredictive
            result = self.model.predict(X, n_samples=self.n_mc_samples, seed=self.seed)
            # Epistemic uncertainty as BALD proxy
            return result['epistemic'] if 'epistemic' in result else result['std']

        else:
            raise ValueError("Model must support predict_with_uncertainty or be BNNPredictive")

    def query(self, X, n_query=1):
        """Select indices with highest BALD scores."""
        scores = self.score(X)
        n_query = min(n_query, len(scores))
        indices = np.argsort(scores)[-n_query:][::-1]
        return indices, scores[indices]


# ---------------------------------------------------------------------------
# 4. DensityWeightedSampler
# ---------------------------------------------------------------------------
class DensityWeightedSampler:
    """Combines informativeness with representativeness via density weighting."""

    def __init__(self, base_sampler, beta=1.0):
        """
        Args:
            base_sampler: any sampler with a score(X_or_probs) method
            beta: density weight exponent (0=pure informativeness, high=more density)
        """
        self.base_sampler = base_sampler
        self.beta = beta

    def _estimate_density(self, X, X_pool):
        """KDE-like density estimation using average similarity."""
        X = np.array(X)
        X_pool = np.array(X_pool)
        n = len(X_pool)

        # Use RBF kernel similarity
        # Bandwidth: median distance heuristic
        if n > 100:
            sample_idx = np.random.choice(n, 100, replace=False)
            sample = X_pool[sample_idx]
        else:
            sample = X_pool

        dists = np.sqrt(np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1))
        bandwidth = np.median(dists[dists > 0]) + 1e-10

        # Density for each point in X relative to X_pool
        densities = np.zeros(len(X))
        for i in range(len(X)):
            d = np.sqrt(np.sum((X_pool - X[i]) ** 2, axis=-1))
            densities[i] = np.mean(np.exp(-d ** 2 / (2 * bandwidth ** 2)))

        return densities

    def score(self, X_pool, informativeness_input):
        """Score combining informativeness and density.

        Args:
            X_pool: pool features for density estimation
            informativeness_input: input to base_sampler.score()
                (probs for UncertaintySampler, X for QBC/BALD)

        Returns:
            scores: combined scores
        """
        info_scores = self.base_sampler.score(informativeness_input)
        densities = self._estimate_density(X_pool, X_pool)

        # Normalize both
        info_range = info_scores.max() - info_scores.min()
        if info_range > 0:
            info_norm = (info_scores - info_scores.min()) / info_range
        else:
            info_norm = np.ones_like(info_scores)

        dens_range = densities.max() - densities.min()
        if dens_range > 0:
            dens_norm = (densities - densities.min()) / dens_range
        else:
            dens_norm = np.ones_like(densities)

        return info_norm * (dens_norm ** self.beta)

    def query(self, X_pool, informativeness_input, n_query=1):
        """Select indices combining informativeness and density."""
        scores = self.score(X_pool, informativeness_input)
        n_query = min(n_query, len(scores))
        indices = np.argsort(scores)[-n_query:][::-1]
        return indices, scores[indices]


# ---------------------------------------------------------------------------
# 5. ExpectedModelChangeSampler
# ---------------------------------------------------------------------------
class ExpectedModelChangeSampler:
    """Select samples that would cause the largest model change (gradient length)."""

    def __init__(self, model, loss_fn=None):
        """
        Args:
            model: neural network with forward/backward
            loss_fn: loss function (default: MSE)
        """
        self.model = model
        self.loss_fn = loss_fn

    def score(self, X, hypothetical_labels=None):
        """Estimate expected gradient length for each sample.

        Args:
            X: input data (n_samples, n_features)
            hypothetical_labels: if None, uses model predictions as labels

        Returns:
            scores: (n_samples,) gradient lengths
        """
        X = np.array(X)
        n = len(X)
        scores = np.zeros(n)

        for i in range(n):
            xi = X[i:i+1]

            # Forward pass
            if hasattr(self.model, 'forward'):
                pred = self.model.forward(xi)
            else:
                pred = self.model.predict(xi)
            pred = np.array(pred)

            if hypothetical_labels is not None:
                target = np.array(hypothetical_labels[i:i+1])
            else:
                target = pred.copy()
                # Perturb slightly to get non-zero gradient
                target = target + np.random.randn(*target.shape) * 0.1

            # Compute gradient of loss
            grad = 2 * (pred - target) / pred.size  # MSE gradient
            grad_norm = np.sqrt(np.sum(grad ** 2))
            scores[i] = grad_norm

        return scores

    def query(self, X, n_query=1, hypothetical_labels=None):
        """Select indices with largest expected model change."""
        scores = self.score(X, hypothetical_labels)
        n_query = min(n_query, len(scores))
        indices = np.argsort(scores)[-n_query:][::-1]
        return indices, scores[indices]


# ---------------------------------------------------------------------------
# 6. BatchActiveLearner
# ---------------------------------------------------------------------------
class BatchActiveLearner:
    """Batch mode active learning with diversity enforcement."""

    def __init__(self, base_sampler, diversity_weight=0.5):
        """
        Args:
            base_sampler: any sampler with query(input, n_query) method
            diversity_weight: trade-off between informativeness and diversity
        """
        self.base_sampler = base_sampler
        self.diversity_weight = diversity_weight

    def query(self, X_pool, sampler_input, n_query=1):
        """Select a diverse batch of informative samples.

        Uses greedy k-center-like approach:
        1. Score all samples with base sampler
        2. Pick top candidate
        3. Down-weight similar samples, repeat

        Args:
            X_pool: pool features for diversity (n_samples, n_features)
            sampler_input: input to base_sampler.score()
            n_query: batch size

        Returns:
            indices: selected indices
            scores: final combined scores
        """
        X_pool = np.array(X_pool)
        n = len(X_pool)
        n_query = min(n_query, n)

        # Get informativeness scores
        info_scores = self.base_sampler.score(sampler_input)

        # Normalize
        info_range = info_scores.max() - info_scores.min()
        if info_range > 0:
            info_norm = (info_scores - info_scores.min()) / info_range
        else:
            info_norm = np.ones(n)

        selected = []
        combined_scores = []
        remaining = set(range(n))

        for _ in range(n_query):
            if not remaining:
                break

            remaining_list = sorted(remaining)

            if not selected:
                # First pick: pure informativeness
                best_idx = remaining_list[np.argmax(info_norm[remaining_list])]
            else:
                # Compute min distance to already-selected
                selected_pts = X_pool[selected]
                remaining_pts = X_pool[remaining_list]

                dists = np.sqrt(np.sum(
                    (remaining_pts[:, None] - selected_pts[None, :]) ** 2, axis=-1
                ))
                min_dists = np.min(dists, axis=1)

                # Normalize distances
                dist_range = min_dists.max() - min_dists.min()
                if dist_range > 0:
                    dist_norm = (min_dists - min_dists.min()) / dist_range
                else:
                    dist_norm = np.ones(len(remaining_list))

                # Combined score
                combined = (1 - self.diversity_weight) * info_norm[remaining_list] + \
                           self.diversity_weight * dist_norm

                best_local_idx = np.argmax(combined)
                best_idx = remaining_list[best_local_idx]

            selected.append(best_idx)
            combined_scores.append(info_norm[best_idx])
            remaining.discard(best_idx)

        return np.array(selected), np.array(combined_scores)


# ---------------------------------------------------------------------------
# 7. BOActiveLearner
# ---------------------------------------------------------------------------
class BOActiveLearner:
    """Bayesian optimization-driven active learning.
    Uses BO to select samples that maximize an acquisition function
    over the input space, treating label acquisition as expensive evaluation.
    """

    def __init__(self, bounds, model=None, acquisition=None,
                 noise_variance=1e-4, seed=42):
        """
        Args:
            bounds: list of (low, high) for each feature dimension
            model: optional pre-trained model for warm-starting
            acquisition: acquisition function (default: EI)
            noise_variance: GP noise
            seed: random seed
        """
        self.bounds = bounds
        self.acquisition = acquisition or ExpectedImprovement(xi=0.01)
        self.noise_variance = noise_variance
        self.seed = seed
        self.optimizer = None
        self.user_model = model

    def _init_optimizer(self):
        """Lazy initialization of BO."""
        self.optimizer = create_optimizer(
            self.bounds,
            method='gp',
            acquisition='ei',
            noise_variance=self.noise_variance,
            n_initial=0,
            seed=self.seed
        )

    def suggest_from_pool(self, X_pool, uncertainty_scores=None):
        """Suggest which pool sample to query using BO principles.

        If we have past observations, uses GP surrogate to predict
        which regions are most promising. Otherwise falls back to
        uncertainty scores.

        Args:
            X_pool: available samples
            uncertainty_scores: optional pre-computed uncertainty

        Returns:
            index: best sample index
            score: acquisition value
        """
        X_pool = np.array(X_pool)

        if self.optimizer is None:
            self._init_optimizer()

        if len(self.optimizer.history.y) >= 2:
            # Use GP to predict and score pool
            X_data = np.array(self.optimizer.history.X)
            y_data = np.array(self.optimizer.history.y)
            self.optimizer.gp.fit(X_data, y_data)
            mean, std = self.optimizer.gp.predict(X_pool, return_std=True)
            best_y = max(self.optimizer.history.y)
            acq_values = self.acquisition.evaluate(mean, std, best_y)
            best_idx = np.argmax(acq_values)
            return best_idx, float(acq_values[best_idx])

        elif uncertainty_scores is not None:
            best_idx = np.argmax(uncertainty_scores)
            return best_idx, uncertainty_scores[best_idx]

        else:
            # Random
            rng = np.random.RandomState(self.seed)
            idx = rng.randint(len(X_pool))
            return idx, 0.0

    def observe(self, x, y_val):
        """Record an observation (label acquisition result)."""
        if self.optimizer is None:
            self._init_optimizer()
        self.optimizer.observe(np.array(x), y_val)

    def query(self, X_pool, n_query=1, uncertainty_scores=None):
        """Select multiple samples using sequential BO.

        Args:
            X_pool: pool data
            n_query: number to select
            uncertainty_scores: optional uncertainty from a classifier

        Returns:
            indices: selected indices
            scores: acquisition values
        """
        X_pool = np.array(X_pool)
        n_query = min(n_query, len(X_pool))
        available = set(range(len(X_pool)))
        selected = []
        scores = []

        for _ in range(n_query):
            if not available:
                break
            avail_list = sorted(available)
            avail_X = X_pool[avail_list]

            if uncertainty_scores is not None:
                avail_unc = uncertainty_scores[avail_list]
            else:
                avail_unc = None

            local_idx, score = self.suggest_from_pool(avail_X, avail_unc)
            global_idx = avail_list[local_idx]

            selected.append(global_idx)
            scores.append(score)
            available.discard(global_idx)

        return np.array(selected), np.array(scores)


# ---------------------------------------------------------------------------
# 8. ActiveLearner (Pool-based orchestrator)
# ---------------------------------------------------------------------------
class ActiveLearner:
    """Pool-based active learning orchestrator.

    Manages the learning loop:
    1. Train model on labeled data
    2. Query strategy selects samples from pool
    3. Oracle labels selected samples
    4. Repeat
    """

    def __init__(self, model, query_strategy, X_pool, y_pool=None,
                 X_initial=None, y_initial=None):
        """
        Args:
            model: learner model (must have fit(X,y) and predict(X)/predict_proba(X))
            query_strategy: sampler object with query() method
            X_pool: unlabeled pool
            y_pool: hidden labels (for simulation; oracle uses these)
            X_initial: initial labeled data
            y_initial: initial labels
        """
        self.model = model
        self.query_strategy = query_strategy
        self.X_pool = np.array(X_pool)
        self.y_pool = np.array(y_pool) if y_pool is not None else None

        if X_initial is not None and y_initial is not None:
            self.X_labeled = np.array(X_initial)
            self.y_labeled = np.array(y_initial)
        else:
            self.X_labeled = np.empty((0, self.X_pool.shape[1]))
            self.y_labeled = np.empty(0)

        self.history = ActiveLearningHistory()
        self._pool_mask = np.ones(len(self.X_pool), dtype=bool)

    def _get_pool(self):
        """Get currently available pool samples."""
        return self.X_pool[self._pool_mask]

    def _get_pool_indices(self):
        """Get original indices of available pool samples."""
        return np.where(self._pool_mask)[0]

    def teach(self, X, y):
        """Add labeled examples and retrain."""
        X = np.array(X).reshape(-1, self.X_pool.shape[1])
        y = np.array(y).ravel()

        if len(self.X_labeled) == 0:
            self.X_labeled = X
            self.y_labeled = y
        else:
            self.X_labeled = np.vstack([self.X_labeled, X])
            self.y_labeled = np.concatenate([self.y_labeled, y])

        self._train()

    def _train(self):
        """Train model on current labeled set."""
        if len(self.X_labeled) > 0:
            self.model.fit(self.X_labeled, self.y_labeled)

    def query(self, n_query=1):
        """Query the strategy for samples to label.

        Returns:
            pool_indices: indices into the original pool
            query_indices: indices into the current available pool
        """
        pool = self._get_pool()
        if len(pool) == 0:
            return np.array([]), np.array([])

        # Get input for query strategy
        sampler_input = self._get_sampler_input(pool)
        query_idx, scores = self.query_strategy.query(sampler_input, n_query)

        # Map back to original pool indices
        available_indices = self._get_pool_indices()
        original_indices = available_indices[query_idx]

        return original_indices, scores

    def _get_sampler_input(self, pool):
        """Get appropriate input for the query strategy."""
        if isinstance(self.query_strategy, UncertaintySampler):
            # Needs probability predictions
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(pool)
            else:
                return self.model.predict(pool)
        elif isinstance(self.query_strategy, (BALDSampler, QueryByCommittee)):
            return pool
        elif isinstance(self.query_strategy, BatchActiveLearner):
            # BatchActiveLearner needs both pool and sampler input
            return pool
        else:
            return pool

    def step(self, n_query=1, oracle=None):
        """Execute one active learning step.

        Args:
            n_query: number of samples to query
            oracle: function(X) -> y, or None to use y_pool

        Returns:
            queried_indices: original pool indices queried
            X_queried: queried feature vectors
            y_queried: labels obtained
        """
        if len(self.X_labeled) > 0:
            self._train()

        indices, scores = self.query(n_query)
        if len(indices) == 0:
            return np.array([]), np.empty((0,)), np.empty(0)

        X_queried = self.X_pool[indices]

        if oracle is not None:
            y_queried = oracle(X_queried)
        elif self.y_pool is not None:
            y_queried = self.y_pool[indices]
        else:
            raise ValueError("No oracle or pool labels available")

        y_queried = np.array(y_queried).ravel()

        # Add to labeled set
        self.teach(X_queried, y_queried)

        # Remove from pool
        self._pool_mask[indices] = False

        # Record history
        accuracy = self._evaluate() if self.y_pool is not None else None
        self.history.record(
            n_labeled=len(self.y_labeled),
            n_pool=np.sum(self._pool_mask),
            accuracy=accuracy,
            queried_indices=indices
        )

        return indices, X_queried, y_queried

    def _evaluate(self):
        """Evaluate model on remaining pool (proxy for test accuracy)."""
        if self.y_pool is None or len(self.X_labeled) < 2:
            return None

        # Use all pool data as test
        preds = self.model.predict(self.X_pool)
        preds = np.array(preds)

        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)

        y_true = self.y_pool
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)

        # Handle regression vs classification
        if len(np.unique(y_true)) <= 20:
            # Classification: accuracy
            return np.mean(np.round(preds) == y_true)
        else:
            # Regression: negative MSE
            return -np.mean((preds - y_true) ** 2)

    def run(self, n_iterations=10, n_query_per_step=1, oracle=None, verbose=False):
        """Run the full active learning loop.

        Args:
            n_iterations: number of AL steps
            n_query_per_step: samples per step
            oracle: label function
            verbose: print progress

        Returns:
            history: ActiveLearningHistory
        """
        for i in range(n_iterations):
            if np.sum(self._pool_mask) == 0:
                if verbose:
                    print(f"Pool exhausted at iteration {i}")
                break

            indices, X_q, y_q = self.step(n_query_per_step, oracle)

            if verbose and self.history.accuracies[-1] is not None:
                print(f"Iter {i+1}: labeled={len(self.y_labeled)}, "
                      f"pool={np.sum(self._pool_mask)}, "
                      f"acc={self.history.accuracies[-1]:.4f}")

        return self.history

    @property
    def labeled_size(self):
        return len(self.y_labeled)

    @property
    def pool_size(self):
        return int(np.sum(self._pool_mask))


# ---------------------------------------------------------------------------
# 9. StreamActiveLearner
# ---------------------------------------------------------------------------
class StreamActiveLearner:
    """Stream-based active learning -- decide per-instance whether to query."""

    def __init__(self, model, threshold=0.5, strategy='entropy', budget=None):
        """
        Args:
            model: learner with fit/predict/predict_proba
            threshold: uncertainty threshold for querying
            strategy: uncertainty strategy ('entropy', 'margin', 'least_confident')
            budget: max number of queries (None = unlimited)
        """
        self.model = model
        self.threshold = threshold
        self.sampler = UncertaintySampler(strategy=strategy)
        self.budget = budget
        self.n_queries = 0
        self.n_seen = 0

        self.X_labeled = None
        self.y_labeled = None
        self.history = ActiveLearningHistory()

    def _should_query(self, x):
        """Decide whether to query this instance."""
        if self.budget is not None and self.n_queries >= self.budget:
            return False

        if self.X_labeled is None or len(self.X_labeled) < 2:
            return True  # Always query initially

        x = np.array(x).reshape(1, -1)
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(x)
        else:
            probs = self.model.predict(x)
        probs = np.array(probs)

        score = self.sampler.score(probs)[0]
        return score >= self.threshold

    def process(self, x, oracle=None, y_true=None):
        """Process a single streaming instance.

        Args:
            x: feature vector
            oracle: function(x) -> y
            y_true: true label (alternative to oracle)

        Returns:
            queried: bool, whether this instance was queried
        """
        self.n_seen += 1
        x = np.array(x).reshape(1, -1)

        if self._should_query(x):
            if oracle is not None:
                y = oracle(x)
            elif y_true is not None:
                y = y_true
            else:
                raise ValueError("Need oracle or y_true")

            y = np.array([y]).ravel()

            if self.X_labeled is None:
                self.X_labeled = x
                self.y_labeled = y
            else:
                self.X_labeled = np.vstack([self.X_labeled, x])
                self.y_labeled = np.concatenate([self.y_labeled, y])

            self.model.fit(self.X_labeled, self.y_labeled)
            self.n_queries += 1

            self.history.record(
                n_labeled=len(self.y_labeled),
                n_pool=0,
                accuracy=None,
                queried_indices=np.array([self.n_seen - 1])
            )
            return True

        return False

    def process_stream(self, X_stream, y_stream=None, oracle=None):
        """Process a full stream.

        Args:
            X_stream: iterable of feature vectors
            y_stream: iterable of labels (for simulation)
            oracle: function(x) -> y

        Returns:
            query_rate: fraction of instances queried
        """
        for i, x in enumerate(X_stream):
            y = y_stream[i] if y_stream is not None else None
            self.process(x, oracle=oracle, y_true=y)

        return self.n_queries / max(self.n_seen, 1)


# ---------------------------------------------------------------------------
# 10. ActiveLearningHistory & Metrics
# ---------------------------------------------------------------------------
class ActiveLearningHistory:
    """Track active learning progress."""

    def __init__(self):
        self.n_labeled_list = []
        self.n_pool_list = []
        self.accuracies = []
        self.queried_indices_list = []

    def record(self, n_labeled, n_pool, accuracy=None, queried_indices=None):
        self.n_labeled_list.append(n_labeled)
        self.n_pool_list.append(n_pool)
        self.accuracies.append(accuracy)
        self.queried_indices_list.append(queried_indices)

    def learning_curve(self):
        """Return (n_labeled, accuracy) pairs."""
        return list(zip(self.n_labeled_list, self.accuracies))

    def summary(self):
        return {
            'total_queries': len(self.n_labeled_list),
            'final_labeled': self.n_labeled_list[-1] if self.n_labeled_list else 0,
            'final_accuracy': self.accuracies[-1] if self.accuracies else None,
            'best_accuracy': max(
                (a for a in self.accuracies if a is not None), default=None
            ),
        }


class ActiveLearningMetrics:
    """Evaluation metrics for active learning."""

    @staticmethod
    def area_under_learning_curve(history, normalize=True):
        """Compute AULC -- area under the learning curve.

        Higher is better (faster learning with fewer labels).
        """
        valid = [(n, a) for n, a in zip(history.n_labeled_list, history.accuracies)
                 if a is not None]
        if len(valid) < 2:
            return 0.0

        n_vals, acc_vals = zip(*valid)
        # Trapezoidal integration
        area = 0.0
        for i in range(1, len(n_vals)):
            area += (n_vals[i] - n_vals[i-1]) * (acc_vals[i] + acc_vals[i-1]) / 2

        if normalize and n_vals[-1] > n_vals[0]:
            area /= (n_vals[-1] - n_vals[0])

        return area

    @staticmethod
    def query_efficiency(history_active, history_random):
        """Compare active vs random learning.

        Returns ratio: how many fewer labels active learning needs
        to reach the same accuracy as random.
        """
        if not history_active.accuracies or not history_random.accuracies:
            return 1.0

        # Find max accuracy of random
        random_valid = [(n, a) for n, a in zip(
            history_random.n_labeled_list, history_random.accuracies
        ) if a is not None]
        if not random_valid:
            return 1.0

        random_final_acc = random_valid[-1][1]
        random_final_n = random_valid[-1][0]

        # Find when active reached that accuracy
        for n, a in zip(history_active.n_labeled_list, history_active.accuracies):
            if a is not None and a >= random_final_acc:
                return random_final_n / max(n, 1)

        return 1.0

    @staticmethod
    def label_complexity(history, target_accuracy):
        """How many labels needed to reach target accuracy."""
        for n, a in zip(history.n_labeled_list, history.accuracies):
            if a is not None and a >= target_accuracy:
                return n
        return None  # Never reached


# ---------------------------------------------------------------------------
# Simple model wrappers for testing
# ---------------------------------------------------------------------------
class SimpleKNN:
    """Simple K-Nearest Neighbors for active learning tests."""

    def __init__(self, k=3):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        X = np.array(X)
        if self.X is None:
            return np.zeros(len(X))

        preds = []
        for x in X:
            dists = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            k = min(self.k, len(self.X))
            nn_idx = np.argsort(dists)[:k]
            nn_labels = self.y[nn_idx]

            if len(np.unique(self.y)) <= 10:
                # Classification: majority vote
                values, counts = np.unique(nn_labels, return_counts=True)
                preds.append(values[np.argmax(counts)])
            else:
                preds.append(np.mean(nn_labels))

        return np.array(preds)

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        if self.X is None:
            return np.ones((len(X), 2)) * 0.5

        n_classes = len(np.unique(self.y))
        if n_classes < 2:
            n_classes = 2

        probs = np.zeros((len(X), n_classes))
        for i, x in enumerate(X):
            dists = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            k = min(self.k, len(self.X))
            nn_idx = np.argsort(dists)[:k]
            nn_labels = self.y[nn_idx].astype(int)

            for label in nn_labels:
                if label < n_classes:
                    probs[i, label] += 1
            probs[i] /= k

        return probs


class SimpleLinearModel:
    """Simple linear model for active learning tests."""

    def __init__(self, lr=0.01, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).ravel()
        n_features = X.shape[1]

        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

        for _ in range(self.n_epochs):
            pred = X @ self.weights + self.bias
            error = pred - y
            self.weights -= self.lr * (2 / len(X)) * X.T @ error
            self.bias -= self.lr * (2 / len(X)) * np.sum(error)

    def predict(self, X):
        X = np.array(X)
        if self.weights is None:
            return np.zeros(len(X))
        return X @ self.weights + self.bias

    def predict_proba(self, X):
        """Sigmoid for binary classification."""
        raw = self.predict(X)
        p = 1 / (1 + np.exp(-np.clip(raw, -500, 500)))
        return np.column_stack([1 - p, p])


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------
def create_active_learner(X_pool, y_pool=None, strategy='entropy',
                          model=None, X_initial=None, y_initial=None,
                          batch_size=1, diversity_weight=0.5):
    """Create an ActiveLearner with specified strategy.

    Args:
        X_pool: unlabeled pool
        y_pool: hidden labels for simulation
        strategy: 'entropy', 'margin', 'least_confident', 'bald', 'qbc', 'batch_entropy'
        model: learner model (default: SimpleKNN)
        X_initial: initial labeled data
        y_initial: initial labels
        batch_size: samples per query (>1 uses BatchActiveLearner)
        diversity_weight: for batch mode

    Returns:
        ActiveLearner instance
    """
    if model is None:
        model = SimpleKNN(k=3)

    if strategy in ('entropy', 'margin', 'least_confident'):
        sampler = UncertaintySampler(strategy=strategy)
        if batch_size > 1:
            sampler = BatchActiveLearner(sampler, diversity_weight=diversity_weight)
    elif strategy == 'batch_entropy':
        base = UncertaintySampler(strategy='entropy')
        sampler = BatchActiveLearner(base, diversity_weight=diversity_weight)
    else:
        sampler = UncertaintySampler(strategy='entropy')

    return ActiveLearner(
        model=model,
        query_strategy=sampler,
        X_pool=X_pool,
        y_pool=y_pool,
        X_initial=X_initial,
        y_initial=y_initial
    )
