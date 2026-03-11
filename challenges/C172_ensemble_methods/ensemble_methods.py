"""
C172: Ensemble Methods
Composing C140 (Neural Network) + C169 (Hyperparameter Tuning)

Ensemble learning: combine multiple models for better predictions.
- Bagging (Bootstrap Aggregating) with parallel model training
- Random Subspace Method (feature bagging)
- Boosting (AdaBoost, Gradient Boosting)
- Stacking (meta-learner over base learners)
- Voting (hard/soft voting for classification)
- Blending (holdout-based stacking variant)
- Ensemble Selection (greedy forward selection from library)
- Diversity metrics (disagreement, correlation, Q-statistic)
"""

import sys, os, math, random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C169_hyperparameter_tuning'))

from neural_network import (
    Tensor, Dense, Activation, Dropout, BatchNorm,
    Sequential, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    SGD, Adam, fit, evaluate
)
from hyperparameter_tuning import (
    HyperparameterSpace, ParamType, HPConfig, RandomSearchTuner
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_list(t):
    """Extract raw list from Tensor or return list as-is."""
    if isinstance(t, Tensor):
        return t.data
    return t


def _argmax(row):
    """Index of maximum value in a list."""
    best_i, best_v = 0, row[0]
    for i in range(1, len(row)):
        if row[i] > best_v:
            best_v = row[i]
            best_i = i
    return best_i


def _bootstrap_sample(X, Y, rng):
    """Draw bootstrap sample with replacement. Returns (X_boot, Y_boot, oob_indices)."""
    n = len(_to_list(X))
    X_data = _to_list(X)
    Y_data = _to_list(Y)
    indices = [rng.randint(0, n - 1) for _ in range(n)]
    selected = set(indices)
    oob = [i for i in range(n) if i not in selected]
    X_boot = [X_data[i] for i in indices]
    Y_boot = [Y_data[i] for i in indices]
    return Tensor(X_boot), Tensor(Y_boot), oob


def _feature_subsample(X, feature_indices):
    """Select subset of features from X."""
    X_data = _to_list(X)
    return Tensor([[row[j] for j in feature_indices] for row in X_data])


def _predict_class(model, X):
    """Get class predictions from a model (argmax of output)."""
    out = model.predict(X)
    out_data = _to_list(out)
    if isinstance(out_data[0], list):
        return [_argmax(row) for row in out_data]
    else:
        return [1 if v > 0.5 else 0 for v in out_data]


def _predict_proba(model, X):
    """Get probability predictions from a model."""
    out = model.predict(X)
    out_data = _to_list(out)
    if isinstance(out_data[0], list):
        return out_data
    else:
        return [[1 - v, v] for v in out_data]


def _accuracy(preds, targets):
    """Compute classification accuracy."""
    Y_data = _to_list(targets) if isinstance(targets, Tensor) else targets
    # Handle one-hot targets
    if isinstance(Y_data[0], list):
        Y_labels = [_argmax(row) for row in Y_data]
    else:
        Y_labels = [int(y) for y in Y_data]
    correct = sum(1 for p, y in zip(preds, Y_labels) if p == y)
    return correct / len(preds) if preds else 0.0


def _mse(preds, targets):
    """Compute mean squared error for regression."""
    Y_data = _to_list(targets) if isinstance(targets, Tensor) else targets
    if isinstance(Y_data[0], list):
        Y_flat = [row[0] for row in Y_data]
    else:
        Y_flat = list(Y_data)
    if isinstance(preds[0], list):
        P_flat = [row[0] for row in preds]
    else:
        P_flat = list(preds)
    return sum((p - y) ** 2 for p, y in zip(P_flat, Y_flat)) / len(P_flat)


def _softmax(logits):
    """Softmax over a list of values."""
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# Base Learner Factory
# ---------------------------------------------------------------------------

class BaseLearnerFactory:
    """Creates base learners (Sequential models) with configurable architecture."""

    def __init__(self, input_size, output_size, hidden_sizes=None,
                 activation='relu', loss_fn=None, optimizer_cls=None,
                 lr=0.01, epochs=50, batch_size=None, seed=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes or [16]
        self.activation = activation
        self.loss_fn = loss_fn or MSELoss()
        self.optimizer_cls = optimizer_cls or SGD
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self._counter = 0

    def build(self, rng=None):
        """Build and return a new untrained model."""
        model = Sequential()
        in_size = self.input_size
        seed_val = rng.randint(0, 100000) if rng else (self.seed or 42) + self._counter
        model_rng = random.Random(seed_val)
        self._counter += 1
        for h in self.hidden_sizes:
            model.add(Dense(in_size, h, rng=model_rng))
            model.add(Activation(self.activation))
            in_size = h
        model.add(Dense(in_size, self.output_size, rng=model_rng))
        if self.output_size > 1:
            model.add(Activation('softmax'))
        return model

    def train(self, model, X, Y):
        """Train a model on data."""
        optimizer = self.optimizer_cls(lr=self.lr)
        fit(model, X, Y, self.loss_fn, optimizer,
            epochs=self.epochs, batch_size=self.batch_size)
        return model

    def build_and_train(self, X, Y, rng=None):
        """Build, train, and return a model."""
        model = self.build(rng=rng)
        self.train(model, X, Y)
        return model


# ---------------------------------------------------------------------------
# Bagging Ensemble
# ---------------------------------------------------------------------------

class BaggingEnsemble:
    """Bootstrap Aggregating -- train models on bootstrap samples, aggregate predictions."""

    def __init__(self, factory, n_estimators=5, seed=42):
        self.factory = factory
        self.n_estimators = n_estimators
        self.seed = seed
        self.models = []
        self.oob_indices = []

    def fit(self, X, Y):
        """Train ensemble on bootstrap samples."""
        rng = random.Random(self.seed)
        self.models = []
        self.oob_indices = []
        for _ in range(self.n_estimators):
            X_boot, Y_boot, oob = _bootstrap_sample(X, Y, rng)
            model = self.factory.build_and_train(X_boot, Y_boot, rng=rng)
            self.models.append(model)
            self.oob_indices.append(oob)
        return self

    def predict(self, X):
        """Aggregate predictions by averaging."""
        all_preds = [_predict_proba(m, X) for m in self.models]
        n = len(_to_list(X))
        n_classes = len(all_preds[0][0])
        result = []
        for i in range(n):
            avg = [0.0] * n_classes
            for preds in all_preds:
                for c in range(n_classes):
                    avg[c] += preds[i][c]
            result.append([v / len(self.models) for v in avg])
        return result

    def predict_classes(self, X):
        """Predict class labels."""
        proba = self.predict(X)
        return [_argmax(row) for row in proba]

    def oob_score(self, X, Y):
        """Compute out-of-bag accuracy."""
        X_data = _to_list(X)
        Y_data = _to_list(Y)
        n = len(X_data)
        votes = [[] for _ in range(n)]
        for model, oob in zip(self.models, self.oob_indices):
            if not oob:
                continue
            X_oob = Tensor([X_data[i] for i in oob])
            preds = _predict_class(model, X_oob)
            for idx, pred in zip(oob, preds):
                votes[idx].append(pred)
        correct = 0
        counted = 0
        if isinstance(Y_data[0], list):
            Y_labels = [_argmax(row) for row in Y_data]
        else:
            Y_labels = [int(y) for y in Y_data]
        for i in range(n):
            if votes[i]:
                # Majority vote
                from collections import Counter
                majority = Counter(votes[i]).most_common(1)[0][0]
                if majority == Y_labels[i]:
                    correct += 1
                counted += 1
        return correct / counted if counted > 0 else 0.0


# ---------------------------------------------------------------------------
# Random Subspace Method
# ---------------------------------------------------------------------------

class RandomSubspaceEnsemble:
    """Feature bagging -- each model sees a random subset of features."""

    def __init__(self, factory, n_estimators=5, max_features=0.7, seed=42):
        self.factory = factory
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.seed = seed
        self.models = []
        self.feature_sets = []

    def fit(self, X, Y):
        """Train ensemble with random feature subsets."""
        rng = random.Random(self.seed)
        X_data = _to_list(X)
        n_features = len(X_data[0])
        if isinstance(self.max_features, float):
            k = max(1, int(n_features * self.max_features))
        else:
            k = min(self.max_features, n_features)

        self.models = []
        self.feature_sets = []
        for _ in range(self.n_estimators):
            features = sorted(rng.sample(range(n_features), k))
            self.feature_sets.append(features)
            X_sub = _feature_subsample(X, features)
            # Build factory with correct input size
            sub_factory = BaseLearnerFactory(
                input_size=k,
                output_size=self.factory.output_size,
                hidden_sizes=self.factory.hidden_sizes,
                activation=self.factory.activation,
                loss_fn=self.factory.loss_fn,
                optimizer_cls=self.factory.optimizer_cls,
                lr=self.factory.lr,
                epochs=self.factory.epochs,
                batch_size=self.factory.batch_size,
                seed=self.factory.seed
            )
            model = sub_factory.build_and_train(X_sub, Y, rng=rng)
            self.models.append(model)
        return self

    def predict(self, X):
        """Aggregate predictions by averaging."""
        n = len(_to_list(X))
        all_preds = []
        for model, features in zip(self.models, self.feature_sets):
            X_sub = _feature_subsample(X, features)
            all_preds.append(_predict_proba(model, X_sub))
        n_classes = len(all_preds[0][0])
        result = []
        for i in range(n):
            avg = [0.0] * n_classes
            for preds in all_preds:
                for c in range(n_classes):
                    avg[c] += preds[i][c]
            result.append([v / len(self.models) for v in avg])
        return result

    def predict_classes(self, X):
        """Predict class labels."""
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# AdaBoost
# ---------------------------------------------------------------------------

class AdaBoostEnsemble:
    """Adaptive Boosting -- weight samples by error, combine weighted models."""

    def __init__(self, factory, n_estimators=5, learning_rate=1.0, seed=42):
        self.factory = factory
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self.models = []
        self.alphas = []

    def fit(self, X, Y):
        """Train AdaBoost ensemble."""
        rng = random.Random(self.seed)
        X_data = _to_list(X)
        Y_data = _to_list(Y)
        n = len(X_data)

        if isinstance(Y_data[0], list):
            Y_labels = [_argmax(row) for row in Y_data]
        else:
            Y_labels = [int(y) for y in Y_data]

        weights = [1.0 / n] * n
        self.models = []
        self.alphas = []

        for _ in range(self.n_estimators):
            # Weighted sampling
            indices = rng.choices(range(n), weights=weights, k=n)
            X_boot = Tensor([X_data[i] for i in indices])
            Y_boot = Tensor([Y_data[i] for i in indices])

            model = self.factory.build_and_train(X_boot, Y_boot, rng=rng)
            preds = _predict_class(model, X)

            # Compute weighted error
            err = sum(weights[i] for i in range(n) if preds[i] != Y_labels[i])
            err = max(err, 1e-10)
            err = min(err, 1.0 - 1e-10)

            # Model weight
            alpha = self.learning_rate * 0.5 * math.log((1 - err) / err)
            self.models.append(model)
            self.alphas.append(alpha)

            # Update sample weights
            for i in range(n):
                if preds[i] != Y_labels[i]:
                    weights[i] *= math.exp(alpha)
                else:
                    weights[i] *= math.exp(-alpha)

            # Normalize
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

        return self

    def predict(self, X):
        """Weighted vote predictions."""
        n = len(_to_list(X))
        # Determine n_classes from first model
        first_proba = _predict_proba(self.models[0], X)
        n_classes = len(first_proba[0])

        scores = [[0.0] * n_classes for _ in range(n)]
        for model, alpha in zip(self.models, self.alphas):
            preds = _predict_class(model, X)
            for i in range(n):
                scores[i][preds[i]] += alpha

        # Normalize to probabilities
        result = []
        for row in scores:
            total = sum(row)
            if total > 0:
                result.append([v / total for v in row])
            else:
                result.append([1.0 / n_classes] * n_classes)
        return result

    def predict_classes(self, X):
        """Predict class labels."""
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# Gradient Boosting (for regression)
# ---------------------------------------------------------------------------

class GradientBoostingEnsemble:
    """Gradient Boosting -- sequential additive models fitting residuals."""

    def __init__(self, factory, n_estimators=5, learning_rate=0.1, seed=42):
        self.factory = factory
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.seed = seed
        self.models = []
        self.initial_prediction = 0.0

    def fit(self, X, Y):
        """Train gradient boosting ensemble."""
        rng = random.Random(self.seed)
        Y_data = _to_list(Y)
        if isinstance(Y_data[0], list):
            Y_flat = [row[0] for row in Y_data]
        else:
            Y_flat = list(Y_data)

        n = len(Y_flat)
        self.initial_prediction = sum(Y_flat) / n
        residuals = [Y_flat[i] - self.initial_prediction for i in range(n)]

        self.models = []
        # Build regression factory (output_size=1)
        reg_factory = BaseLearnerFactory(
            input_size=self.factory.input_size,
            output_size=1,
            hidden_sizes=self.factory.hidden_sizes,
            activation=self.factory.activation,
            loss_fn=MSELoss(),
            optimizer_cls=self.factory.optimizer_cls,
            lr=self.factory.lr,
            epochs=self.factory.epochs,
            batch_size=self.factory.batch_size,
            seed=self.factory.seed
        )

        for _ in range(self.n_estimators):
            Y_res = Tensor([[r] for r in residuals])
            model = reg_factory.build_and_train(X, Y_res, rng=rng)
            self.models.append(model)

            # Update residuals
            preds = _to_list(model.predict(X))
            for i in range(n):
                p = preds[i] if not isinstance(preds[i], list) else preds[i][0]
                residuals[i] -= self.learning_rate * p

        return self

    def predict(self, X):
        """Predict regression values."""
        n = len(_to_list(X))
        result = [self.initial_prediction] * n
        for model in self.models:
            preds = _to_list(model.predict(X))
            for i in range(n):
                p = preds[i] if not isinstance(preds[i], list) else preds[i][0]
                result[i] += self.learning_rate * p
        return result


# ---------------------------------------------------------------------------
# Voting Ensemble
# ---------------------------------------------------------------------------

class VotingEnsemble:
    """Hard/soft voting over pre-trained models."""

    def __init__(self, models, voting='soft', weights=None):
        self.models = models
        self.voting = voting
        self.weights = weights or [1.0] * len(models)

    def predict(self, X):
        """Predict using voting."""
        n = len(_to_list(X))
        if self.voting == 'hard':
            return self._hard_vote(X, n)
        else:
            return self._soft_vote(X, n)

    def _hard_vote(self, X, n):
        """Weighted majority vote."""
        all_preds = [_predict_class(m, X) for m in self.models]
        first_proba = _predict_proba(self.models[0], X)
        n_classes = len(first_proba[0])
        result = []
        for i in range(n):
            scores = [0.0] * n_classes
            for j, preds in enumerate(all_preds):
                scores[preds[i]] += self.weights[j]
            result.append(_argmax(scores))
        return result

    def _soft_vote(self, X, n):
        """Weighted probability averaging."""
        all_proba = [_predict_proba(m, X) for m in self.models]
        n_classes = len(all_proba[0][0])
        result = []
        w_total = sum(self.weights)
        for i in range(n):
            avg = [0.0] * n_classes
            for j, proba in enumerate(all_proba):
                for c in range(n_classes):
                    avg[c] += self.weights[j] * proba[i][c]
            result.append([v / w_total for v in avg])
        return result

    def predict_classes(self, X):
        """Predict class labels."""
        if self.voting == 'hard':
            return self.predict(X)
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """Stacking -- meta-learner trained on base learner predictions."""

    def __init__(self, base_factories, meta_factory, use_original_features=False, seed=42):
        self.base_factories = base_factories
        self.meta_factory = meta_factory
        self.use_original_features = use_original_features
        self.seed = seed
        self.base_models = []
        self.meta_model = None

    def fit(self, X, Y):
        """Train base models, then meta-model on their predictions."""
        rng = random.Random(self.seed)
        X_data = _to_list(X)
        Y_data = _to_list(Y)
        n = len(X_data)

        # Train base models on full data
        self.base_models = []
        for factory in self.base_factories:
            model = factory.build_and_train(X, Y, rng=rng)
            self.base_models.append(model)

        # Generate meta-features using cross-predictions
        # Simple approach: use predictions from models trained on full data
        meta_features = self._generate_meta_features(X)

        if self.use_original_features:
            # Concatenate original features with meta-features
            for i in range(n):
                if isinstance(X_data[i], list):
                    meta_features[i] = list(X_data[i]) + meta_features[i]
                else:
                    meta_features[i] = [X_data[i]] + meta_features[i]

        X_meta = Tensor(meta_features)
        self.meta_model = self.meta_factory.build_and_train(X_meta, Y, rng=rng)
        return self

    def _generate_meta_features(self, X):
        """Generate meta-features from base model predictions."""
        all_proba = [_predict_proba(m, X) for m in self.base_models]
        n = len(_to_list(X))
        meta = []
        for i in range(n):
            row = []
            for proba in all_proba:
                row.extend(proba[i])
            meta.append(row)
        return meta

    def predict(self, X):
        """Predict using meta-model on base model predictions."""
        X_data = _to_list(X)
        meta_features = self._generate_meta_features(X)
        if self.use_original_features:
            for i in range(len(X_data)):
                if isinstance(X_data[i], list):
                    meta_features[i] = list(X_data[i]) + meta_features[i]
                else:
                    meta_features[i] = [X_data[i]] + meta_features[i]
        X_meta = Tensor(meta_features)
        return _predict_proba(self.meta_model, X_meta)

    def predict_classes(self, X):
        """Predict class labels."""
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# Blending Ensemble (holdout-based stacking)
# ---------------------------------------------------------------------------

class BlendingEnsemble:
    """Blending -- stacking variant using holdout set instead of cross-validation."""

    def __init__(self, base_factories, meta_factory, holdout_ratio=0.3, seed=42):
        self.base_factories = base_factories
        self.meta_factory = meta_factory
        self.holdout_ratio = holdout_ratio
        self.seed = seed
        self.base_models = []
        self.meta_model = None

    def fit(self, X, Y):
        """Train with holdout split."""
        rng = random.Random(self.seed)
        X_data = _to_list(X)
        Y_data = _to_list(Y)
        n = len(X_data)

        # Split into train and holdout
        indices = list(range(n))
        rng.shuffle(indices)
        split = int(n * (1 - self.holdout_ratio))
        train_idx = indices[:split]
        hold_idx = indices[split:]

        X_train = Tensor([X_data[i] for i in train_idx])
        Y_train = Tensor([Y_data[i] for i in train_idx])
        X_hold = Tensor([X_data[i] for i in hold_idx])
        Y_hold = Tensor([Y_data[i] for i in hold_idx])

        # Train base models on train set
        self.base_models = []
        for factory in self.base_factories:
            model = factory.build_and_train(X_train, Y_train, rng=rng)
            self.base_models.append(model)

        # Generate meta-features on holdout set
        meta_features = []
        all_proba = [_predict_proba(m, X_hold) for m in self.base_models]
        for i in range(len(hold_idx)):
            row = []
            for proba in all_proba:
                row.extend(proba[i])
            meta_features.append(row)

        X_meta = Tensor(meta_features)
        self.meta_model = self.meta_factory.build_and_train(X_meta, Y_hold, rng=rng)
        return self

    def predict(self, X):
        """Predict using blended meta-model."""
        all_proba = [_predict_proba(m, X) for m in self.base_models]
        n = len(_to_list(X))
        meta_features = []
        for i in range(n):
            row = []
            for proba in all_proba:
                row.extend(proba[i])
            meta_features.append(row)
        X_meta = Tensor(meta_features)
        return _predict_proba(self.meta_model, X_meta)

    def predict_classes(self, X):
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# Ensemble Selection (Greedy Forward Selection)
# ---------------------------------------------------------------------------

class EnsembleSelection:
    """Greedy forward selection from a library of models."""

    def __init__(self, models, ensemble_size=5, metric='accuracy',
                 with_replacement=True, seed=42):
        self.library = models
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.with_replacement = with_replacement
        self.seed = seed
        self.selected_models = []
        self.selected_weights = []

    def fit(self, X_val, Y_val):
        """Select best ensemble from model library using validation set."""
        rng = random.Random(self.seed)
        Y_data = _to_list(Y_val)
        if isinstance(Y_data[0], list):
            Y_labels = [_argmax(row) for row in Y_data]
        else:
            Y_labels = [int(y) for y in Y_data]

        n = len(Y_labels)
        n_classes = max(Y_labels) + 1

        # Pre-compute all predictions
        lib_proba = [_predict_proba(m, X_val) for m in self.library]

        selected_indices = []
        current_sum = [[0.0] * n_classes for _ in range(n)]

        for step in range(self.ensemble_size):
            best_score = -1
            best_idx = 0

            candidates = range(len(self.library)) if self.with_replacement else \
                [i for i in range(len(self.library)) if i not in selected_indices]

            if not candidates:
                break

            for idx in candidates:
                # Try adding this model
                trial_sum = [row[:] for row in current_sum]
                for i in range(n):
                    for c in range(n_classes):
                        trial_sum[i][c] += lib_proba[idx][i][c]

                # Evaluate
                count = step + 1
                preds = [_argmax([trial_sum[i][c] / count for c in range(n_classes)]) for i in range(n)]
                score = sum(1 for p, y in zip(preds, Y_labels) if p == y) / n

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected_indices.append(best_idx)
            for i in range(n):
                for c in range(n_classes):
                    current_sum[i][c] += lib_proba[best_idx][i][c]

        # Build selected models and weights
        from collections import Counter
        counts = Counter(selected_indices)
        unique_indices = sorted(counts.keys())
        self.selected_models = [self.library[i] for i in unique_indices]
        total = sum(counts.values())
        self.selected_weights = [counts[i] / total for i in unique_indices]
        return self

    def predict(self, X):
        """Predict using selected ensemble."""
        n = len(_to_list(X))
        all_proba = [_predict_proba(m, X) for m in self.selected_models]
        n_classes = len(all_proba[0][0])
        result = []
        for i in range(n):
            avg = [0.0] * n_classes
            for j, proba in enumerate(all_proba):
                for c in range(n_classes):
                    avg[c] += self.selected_weights[j] * proba[i][c]
            result.append(avg)
        return result

    def predict_classes(self, X):
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# Diversity Metrics
# ---------------------------------------------------------------------------

class DiversityMetrics:
    """Measure diversity among ensemble members."""

    @staticmethod
    def disagreement(preds_a, preds_b):
        """Fraction of samples where two classifiers disagree."""
        n = len(preds_a)
        return sum(1 for a, b in zip(preds_a, preds_b) if a != b) / n if n > 0 else 0.0

    @staticmethod
    def q_statistic(preds_a, preds_b, labels):
        """Yule's Q statistic between two classifiers. Q=1 means identical, Q=-1 opposite."""
        n11 = n10 = n01 = n00 = 0
        for a, b, y in zip(preds_a, preds_b, labels):
            ca = (a == y)
            cb = (b == y)
            if ca and cb:
                n11 += 1
            elif ca and not cb:
                n10 += 1
            elif not ca and cb:
                n01 += 1
            else:
                n00 += 1
        denom = n11 * n00 + n10 * n01
        if denom == 0:
            return 0.0
        return (n11 * n00 - n10 * n01) / denom

    @staticmethod
    def correlation_diversity(preds_a, preds_b, labels):
        """Correlation coefficient between classifier correctness vectors."""
        n = len(labels)
        ca = [1.0 if a == y else 0.0 for a, y in zip(preds_a, labels)]
        cb = [1.0 if b == y else 0.0 for b, y in zip(preds_b, labels)]
        mean_a = sum(ca) / n
        mean_b = sum(cb) / n
        cov = sum((ca[i] - mean_a) * (cb[i] - mean_b) for i in range(n)) / n
        std_a = math.sqrt(sum((v - mean_a) ** 2 for v in ca) / n)
        std_b = math.sqrt(sum((v - mean_b) ** 2 for v in cb) / n)
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        return cov / (std_a * std_b)

    @staticmethod
    def ensemble_disagreement(models, X):
        """Average pairwise disagreement across all model pairs."""
        all_preds = [_predict_class(m, X) for m in models]
        n_models = len(all_preds)
        total = 0.0
        count = 0
        for i in range(n_models):
            for j in range(i + 1, n_models):
                total += DiversityMetrics.disagreement(all_preds[i], all_preds[j])
                count += 1
        return total / count if count > 0 else 0.0

    @staticmethod
    def double_fault(preds_a, preds_b, labels):
        """Proportion of samples where both classifiers are wrong."""
        n = len(labels)
        both_wrong = sum(1 for a, b, y in zip(preds_a, preds_b, labels)
                         if a != y and b != y)
        return both_wrong / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Auto-Ensemble (composes C169 HP tuning)
# ---------------------------------------------------------------------------

class AutoEnsemble:
    """Automatically tune ensemble composition using hyperparameter search."""

    def __init__(self, input_size, output_size, n_estimators=5,
                 method='bagging', seed=42):
        self.input_size = input_size
        self.output_size = output_size
        self.n_estimators = n_estimators
        self.method = method
        self.seed = seed
        self.best_ensemble = None
        self.best_config = None
        self.search_history = []

    def fit(self, X, Y, n_trials=10, val_split=0.2):
        """Auto-tune ensemble hyperparameters."""
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)
        X_data = _to_list(X)
        Y_data = _to_list(Y)
        n = len(X_data)

        # Split train/val
        indices = list(range(n))
        rng.shuffle(indices)
        split = int(n * (1 - val_split))
        train_idx = indices[:split]
        val_idx = indices[split:]

        X_train = Tensor([X_data[i] for i in train_idx])
        Y_train = Tensor([Y_data[i] for i in train_idx])
        X_val = Tensor([X_data[i] for i in val_idx])
        Y_val = Tensor([Y_data[i] for i in val_idx])

        # Define search space
        space = HyperparameterSpace()
        space.add_continuous('lr', low=0.001, high=0.1, log_scale=True)
        space.add_integer('hidden_size', low=8, high=32)
        space.add_integer('epochs', low=20, high=100)

        best_score = -1
        best_cfg = None
        best_ens = None

        for trial in range(n_trials):
            cfg = space.sample(np_rng)
            lr = cfg.values['lr']
            hidden = int(cfg.values['hidden_size'])
            epochs = int(cfg.values['epochs'])

            factory = BaseLearnerFactory(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_sizes=[hidden],
                lr=lr,
                epochs=epochs,
                seed=self.seed + trial
            )

            if self.method == 'bagging':
                ens = BaggingEnsemble(factory, n_estimators=self.n_estimators,
                                       seed=self.seed + trial)
            elif self.method == 'adaboost':
                ens = AdaBoostEnsemble(factory, n_estimators=self.n_estimators,
                                        seed=self.seed + trial)
            else:
                ens = BaggingEnsemble(factory, n_estimators=self.n_estimators,
                                       seed=self.seed + trial)

            ens.fit(X_train, Y_train)
            preds = ens.predict_classes(X_val)

            Y_val_data = _to_list(Y_val)
            if isinstance(Y_val_data[0], list):
                Y_val_labels = [_argmax(row) for row in Y_val_data]
            else:
                Y_val_labels = [int(y) for y in Y_val_data]

            score = sum(1 for p, y in zip(preds, Y_val_labels) if p == y) / len(preds)
            self.search_history.append({'config': cfg.values, 'score': score})

            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_ens = ens

        self.best_ensemble = best_ens
        self.best_config = best_cfg
        self.best_score = best_score

        # Retrain best on full data
        if best_cfg:
            lr = best_cfg.values['lr']
            hidden = int(best_cfg.values['hidden_size'])
            epochs = int(best_cfg.values['epochs'])
            factory = BaseLearnerFactory(
                input_size=self.input_size,
                output_size=self.output_size,
                hidden_sizes=[hidden],
                lr=lr,
                epochs=epochs,
                seed=self.seed
            )
            if self.method == 'bagging':
                self.best_ensemble = BaggingEnsemble(factory, self.n_estimators, self.seed)
            elif self.method == 'adaboost':
                self.best_ensemble = AdaBoostEnsemble(factory, self.n_estimators, self.seed)
            self.best_ensemble.fit(X, Y)

        return self

    def predict(self, X):
        return self.best_ensemble.predict(X)

    def predict_classes(self, X):
        return self.best_ensemble.predict_classes(X)


# ---------------------------------------------------------------------------
# Snapshot Ensemble (save models at different training stages)
# ---------------------------------------------------------------------------

class SnapshotEnsemble:
    """Collect model snapshots during training using cyclic learning rate."""

    def __init__(self, input_size, output_size, hidden_sizes=None,
                 n_cycles=5, epochs_per_cycle=20, max_lr=0.05, seed=42):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes or [16]
        self.n_cycles = n_cycles
        self.epochs_per_cycle = epochs_per_cycle
        self.max_lr = max_lr
        self.seed = seed
        self.snapshots = []

    def _cosine_lr(self, epoch, cycle_epochs):
        """Cosine annealing within cycle."""
        return self.max_lr * 0.5 * (1 + math.cos(math.pi * epoch / cycle_epochs))

    def fit(self, X, Y):
        """Train with cyclic LR, saving snapshots at minima."""
        model_rng = random.Random(self.seed)
        model = Sequential()
        in_size = self.input_size
        for h in self.hidden_sizes:
            model.add(Dense(in_size, h, rng=model_rng))
            model.add(Activation('relu'))
            in_size = h
        model.add(Dense(in_size, self.output_size, rng=model_rng))
        if self.output_size > 1:
            model.add(Activation('softmax'))

        loss_fn = CrossEntropyLoss() if self.output_size > 1 else MSELoss()
        self.snapshots = []

        for cycle in range(self.n_cycles):
            for epoch in range(self.epochs_per_cycle):
                lr = self._cosine_lr(epoch, self.epochs_per_cycle)
                optimizer = SGD(lr=lr)
                fit(model, X, Y, loss_fn, optimizer, epochs=1)

            # Save snapshot (deep copy weights)
            snapshot = self._clone_model(model, model_rng)
            self.snapshots.append(snapshot)

        return self

    def _clone_model(self, model, rng):
        """Clone model by rebuilding and copying weights."""
        clone = Sequential()
        in_size = self.input_size
        for h in self.hidden_sizes:
            clone.add(Dense(in_size, h, rng=rng))
            clone.add(Activation('relu'))
            in_size = h
        clone.add(Dense(in_size, self.output_size, rng=rng))
        if self.output_size > 1:
            clone.add(Activation('softmax'))

        # Copy weights
        orig_layers = [l for l in model.layers if hasattr(l, 'weights')]
        clone_layers = [l for l in clone.layers if hasattr(l, 'weights')]
        for orig, cl in zip(orig_layers, clone_layers):
            cl.weights = Tensor([row[:] for row in _to_list(orig.weights)])
            cl.bias = Tensor([v for v in _to_list(orig.bias)])

        return clone

    def predict(self, X):
        """Average predictions from all snapshots."""
        n = len(_to_list(X))
        all_proba = [_predict_proba(m, X) for m in self.snapshots]
        n_classes = len(all_proba[0][0])
        result = []
        for i in range(n):
            avg = [0.0] * n_classes
            for proba in all_proba:
                for c in range(n_classes):
                    avg[c] += proba[i][c]
            result.append([v / len(self.snapshots) for v in avg])
        return result

    def predict_classes(self, X):
        proba = self.predict(X)
        return [_argmax(row) for row in proba]


# ---------------------------------------------------------------------------
# Ensemble Comparison Report
# ---------------------------------------------------------------------------

class EnsembleComparison:
    """Compare multiple ensemble methods on the same data."""

    def __init__(self, ensembles, names=None):
        self.ensembles = ensembles
        self.names = names or [f"Ensemble_{i}" for i in range(len(ensembles))]
        self.results = {}

    def evaluate(self, X_test, Y_test):
        """Evaluate all ensembles on test data."""
        Y_data = _to_list(Y_test)
        if isinstance(Y_data[0], list):
            Y_labels = [_argmax(row) for row in Y_data]
        else:
            Y_labels = [int(y) for y in Y_data]

        self.results = {}
        for name, ens in zip(self.names, self.ensembles):
            preds = ens.predict_classes(X_test)
            acc = sum(1 for p, y in zip(preds, Y_labels) if p == y) / len(Y_labels)
            self.results[name] = {
                'accuracy': acc,
                'predictions': preds
            }
        return self.results

    def best(self):
        """Return name of best ensemble."""
        if not self.results:
            return None
        return max(self.results, key=lambda k: self.results[k]['accuracy'])

    def summary(self):
        """Generate comparison summary."""
        lines = ["Ensemble Comparison Results:", "-" * 40]
        for name in self.names:
            if name in self.results:
                acc = self.results[name]['accuracy']
                lines.append(f"  {name}: {acc:.4f}")
        if self.results:
            lines.append(f"  Best: {self.best()}")
        return "\n".join(lines)
