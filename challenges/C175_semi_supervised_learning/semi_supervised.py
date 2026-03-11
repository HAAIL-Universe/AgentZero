"""
C175: Semi-Supervised Learning
Composing C174 (Active Learning) + C140 (Tensor/Neural Network)

Semi-supervised learning leverages both labeled and unlabeled data.
10 components: SelfTrainer, CoTrainer, LabelPropagation, LabelSpreading,
ConsistencyRegularizer, MixUp, MixMatchTrainer, FixMatchTrainer,
SemiSupervisedTrainer, SemiSupervisedMetrics.
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C174_active_learning'))

from neural_network import (
    Tensor, Sequential, Dense, Activation, Dropout, BatchNorm,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    SGD, Adam, fit, train_step, evaluate, predict_classes, accuracy,
    one_hot, normalize, softmax, softmax_batch, build_model
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_list(t):
    """Convert Tensor or nested list to flat/nested Python list."""
    if isinstance(t, Tensor):
        return t.tolist()
    return t


def _predict_proba(model, X):
    """Get probability predictions from a model. Returns list of lists."""
    model.eval()
    out = model.forward(X)
    probs = softmax_batch(out)
    return _to_list(probs)


def _argmax(row):
    """Argmax of a list."""
    best_i, best_v = 0, row[0]
    for i in range(1, len(row)):
        if row[i] > best_v:
            best_v = row[i]
            best_i = i
    return best_i


def _max_val(row):
    """Max value in a list."""
    return max(row)


def _entropy(probs):
    """Entropy of a probability distribution."""
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p + 1e-12)
    return h


def _euclidean_dist_sq(a, b):
    """Squared Euclidean distance between two vectors."""
    s = 0.0
    for i in range(len(a)):
        d = a[i] - b[i]
        s += d * d
    return s


def _rbf_kernel(a, b, gamma):
    """RBF (Gaussian) kernel."""
    return math.exp(-gamma * _euclidean_dist_sq(a, b))


def _tensor_row(t, i):
    """Get row i from a Tensor as a list."""
    if isinstance(t, Tensor):
        row = t[i]
        if isinstance(row, Tensor):
            return row.tolist()
        return row if isinstance(row, list) else [row]
    return t[i]


def _num_rows(t):
    """Number of rows in a Tensor or list."""
    return len(t)


def _make_tensor(data):
    """Create Tensor from list data."""
    if isinstance(data, Tensor):
        return data
    return Tensor(data)


def _subset_tensor(t, indices):
    """Extract rows from Tensor by indices."""
    rows = [_tensor_row(t, i) for i in indices]
    if not rows:
        return Tensor([])
    return Tensor(rows)


def _concat_tensors(t1, t2):
    """Concatenate two Tensors (row-wise)."""
    rows1 = [_tensor_row(t1, i) for i in range(_num_rows(t1))]
    rows2 = [_tensor_row(t2, i) for i in range(_num_rows(t2))]
    return Tensor(rows1 + rows2)


def _concat_labels(l1, l2):
    """Concatenate two label lists."""
    a = list(l1) if not isinstance(l1, list) else l1
    b = list(l2) if not isinstance(l2, list) else l2
    return a + b


# ---------------------------------------------------------------------------
# 1. SelfTrainer
# ---------------------------------------------------------------------------

class SelfTrainer:
    """Iterative self-training with pseudo-labels.

    Train on labeled data, predict unlabeled, add high-confidence
    predictions as pseudo-labels, repeat.
    """

    def __init__(self, model, threshold=0.9, max_iter=10, batch_size=None,
                 loss_fn=None, optimizer=None, epochs_per_iter=10, seed=42):
        self.model = model
        self.threshold = threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.optimizer = optimizer or Adam(lr=0.01)
        self.epochs_per_iter = epochs_per_iter
        self.seed = seed
        self.history = []  # (iteration, n_labeled, n_pseudo, accuracy_if_available)

    def fit(self, X_labeled, y_labeled, X_unlabeled, num_classes=None,
            X_val=None, y_val=None):
        """Run self-training loop."""
        rng = random.Random(self.seed)

        if num_classes is None:
            num_classes = max(y_labeled) + 1

        # Working copies
        X_l = [_tensor_row(X_labeled, i) for i in range(_num_rows(X_labeled))]
        y_l = list(y_labeled)
        X_u = [_tensor_row(X_unlabeled, i) for i in range(_num_rows(X_unlabeled))]

        for iteration in range(self.max_iter):
            if len(X_u) == 0:
                break

            # Train on current labeled set
            X_t = Tensor(X_l)
            y_t = one_hot(y_l, num_classes)
            fit(self.model, X_t, y_t, self.loss_fn, self.optimizer,
                epochs=self.epochs_per_iter, batch_size=self.batch_size,
                shuffle=True, verbose=False)

            # Predict on unlabeled
            X_u_tensor = Tensor(X_u)
            probs = _predict_proba(self.model, X_u_tensor)

            # Select high-confidence predictions
            new_labeled_indices = []
            new_labels = []
            for i, p in enumerate(probs):
                conf = _max_val(p)
                if conf >= self.threshold:
                    new_labeled_indices.append(i)
                    new_labels.append(_argmax(p))

            # Record history
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = accuracy(self.model, X_val, y_val)
            self.history.append((iteration, len(X_l), len(new_labeled_indices), val_acc))

            if len(new_labeled_indices) == 0:
                break

            # Add pseudo-labeled to labeled set
            for idx, label in zip(new_labeled_indices, new_labels):
                X_l.append(X_u[idx])
                y_l.append(label)

            # Remove from unlabeled (reverse order to preserve indices)
            for idx in sorted(new_labeled_indices, reverse=True):
                X_u.pop(idx)

        return self.history

    def predict(self, X):
        """Predict class labels."""
        return predict_classes(self.model, X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return _predict_proba(self.model, X)


# ---------------------------------------------------------------------------
# 2. CoTrainer
# ---------------------------------------------------------------------------

class CoTrainer:
    """Co-training with two models on different feature views.

    Each model labels unlabeled data for the other model based on
    its most confident predictions.
    """

    def __init__(self, model1, model2, view1_indices, view2_indices,
                 threshold=0.85, max_iter=10, n_per_iter=5,
                 loss_fn=None, optimizer1=None, optimizer2=None,
                 epochs_per_iter=10, num_classes=None, seed=42):
        self.model1 = model1
        self.model2 = model2
        self.view1_indices = list(view1_indices)
        self.view2_indices = list(view2_indices)
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_per_iter = n_per_iter
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.optimizer1 = optimizer1 or Adam(lr=0.01)
        self.optimizer2 = optimizer2 or Adam(lr=0.01)
        self.epochs_per_iter = epochs_per_iter
        self.num_classes = num_classes
        self.seed = seed
        self.history = []

    def _extract_view(self, X, indices):
        """Extract feature subset for a view."""
        rows = []
        for i in range(_num_rows(X)):
            row = _tensor_row(X, i)
            rows.append([row[j] for j in indices])
        return Tensor(rows)

    def fit(self, X_labeled, y_labeled, X_unlabeled, X_val=None, y_val=None):
        """Run co-training loop."""
        if self.num_classes is None:
            self.num_classes = max(y_labeled) + 1

        # Split into views
        X_l1 = [_tensor_row(self._extract_view(X_labeled, self.view1_indices), i)
                 for i in range(_num_rows(X_labeled))]
        X_l2 = [_tensor_row(self._extract_view(X_labeled, self.view2_indices), i)
                 for i in range(_num_rows(X_labeled))]
        y_l1 = list(y_labeled)
        y_l2 = list(y_labeled)

        X_u_full = [_tensor_row(X_unlabeled, i) for i in range(_num_rows(X_unlabeled))]
        X_u1 = [_tensor_row(self._extract_view(X_unlabeled, self.view1_indices), i)
                 for i in range(_num_rows(X_unlabeled))]
        X_u2 = [_tensor_row(self._extract_view(X_unlabeled, self.view2_indices), i)
                 for i in range(_num_rows(X_unlabeled))]

        for iteration in range(self.max_iter):
            if len(X_u1) == 0:
                break

            # Train both models
            y_oh1 = one_hot(y_l1, self.num_classes)
            y_oh2 = one_hot(y_l2, self.num_classes)
            fit(self.model1, Tensor(X_l1), y_oh1, self.loss_fn, self.optimizer1,
                epochs=self.epochs_per_iter, verbose=False)
            fit(self.model2, Tensor(X_l2), y_oh2, self.loss_fn, self.optimizer2,
                epochs=self.epochs_per_iter, verbose=False)

            # Model1 labels for Model2
            probs1 = _predict_proba(self.model1, Tensor(X_u1))
            # Model2 labels for Model1
            probs2 = _predict_proba(self.model2, Tensor(X_u2))

            # Collect confident predictions
            indices_to_remove = set()

            # Top confident from model1 -> add to model2's labeled set
            scored1 = [(i, _max_val(probs1[i]), _argmax(probs1[i]))
                       for i in range(len(probs1))]
            scored1.sort(key=lambda x: -x[1])
            added1 = 0
            for idx, conf, label in scored1:
                if added1 >= self.n_per_iter:
                    break
                if conf >= self.threshold:
                    X_l2.append(X_u2[idx])
                    y_l2.append(label)
                    indices_to_remove.add(idx)
                    added1 += 1

            # Top confident from model2 -> add to model1's labeled set
            scored2 = [(i, _max_val(probs2[i]), _argmax(probs2[i]))
                       for i in range(len(probs2))]
            scored2.sort(key=lambda x: -x[1])
            added2 = 0
            for idx, conf, label in scored2:
                if added2 >= self.n_per_iter:
                    break
                if conf >= self.threshold:
                    X_l1.append(X_u1[idx])
                    y_l1.append(label)
                    indices_to_remove.add(idx)
                    added2 += 1

            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = self._combined_accuracy(X_val, y_val)
            self.history.append((iteration, len(y_l1), len(y_l2),
                                 added1 + added2, val_acc))

            if len(indices_to_remove) == 0:
                break

            # Remove labeled samples from unlabeled pool
            for idx in sorted(indices_to_remove, reverse=True):
                X_u1.pop(idx)
                X_u2.pop(idx)
                X_u_full.pop(idx)

        return self.history

    def _combined_accuracy(self, X_val, y_val):
        """Combined prediction accuracy (average of both models' predictions)."""
        X_v1 = self._extract_view(X_val, self.view1_indices)
        X_v2 = self._extract_view(X_val, self.view2_indices)
        probs1 = _predict_proba(self.model1, X_v1)
        probs2 = _predict_proba(self.model2, X_v2)
        correct = 0
        n = _num_rows(X_val)
        for i in range(n):
            # Average probabilities
            avg = [(probs1[i][j] + probs2[i][j]) / 2 for j in range(len(probs1[i]))]
            pred = _argmax(avg)
            y_true = y_val[i] if isinstance(y_val, list) else y_val[i]
            if pred == y_true:
                correct += 1
        return correct / n if n > 0 else 0.0

    def predict(self, X):
        """Predict using combined model votes."""
        X_v1 = self._extract_view(X, self.view1_indices)
        X_v2 = self._extract_view(X, self.view2_indices)
        probs1 = _predict_proba(self.model1, X_v1)
        probs2 = _predict_proba(self.model2, X_v2)
        preds = []
        for i in range(len(probs1)):
            avg = [(probs1[i][j] + probs2[i][j]) / 2 for j in range(len(probs1[i]))]
            preds.append(_argmax(avg))
        return preds

    def predict_proba(self, X):
        """Predict probabilities using averaged model outputs."""
        X_v1 = self._extract_view(X, self.view1_indices)
        X_v2 = self._extract_view(X, self.view2_indices)
        probs1 = _predict_proba(self.model1, X_v1)
        probs2 = _predict_proba(self.model2, X_v2)
        return [[(probs1[i][j] + probs2[i][j]) / 2 for j in range(len(probs1[i]))]
                for i in range(len(probs1))]


# ---------------------------------------------------------------------------
# 3. LabelPropagation
# ---------------------------------------------------------------------------

class LabelPropagation:
    """Graph-based label propagation.

    Build a similarity graph, then iteratively propagate labels from
    labeled to unlabeled nodes. Labels are clamped for labeled data.
    """

    def __init__(self, kernel='rbf', gamma=1.0, n_neighbors=None,
                 max_iter=100, tol=1e-4):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self._labels = None
        self._label_distributions = None

    def _build_affinity(self, X):
        """Build affinity matrix."""
        n = _num_rows(X)
        rows = [_tensor_row(X, i) for i in range(n)]
        W = [[0.0] * n for _ in range(n)]

        if self.kernel == 'rbf':
            for i in range(n):
                for j in range(i + 1, n):
                    w = _rbf_kernel(rows[i], rows[j], self.gamma)
                    W[i][j] = w
                    W[j][i] = w
        elif self.kernel == 'knn':
            k = self.n_neighbors or 7
            # Compute all distances
            for i in range(n):
                dists = []
                for j in range(n):
                    if i == j:
                        dists.append((float('inf'), j))
                    else:
                        dists.append((_euclidean_dist_sq(rows[i], rows[j]), j))
                dists.sort()
                for d, j in dists[:k]:
                    W[i][j] = 1.0
                    W[j][i] = 1.0  # Symmetrize

        return W

    def _build_transition(self, W):
        """Build row-normalized transition matrix T = D^{-1} W."""
        n = len(W)
        T = [[0.0] * n for _ in range(n)]
        for i in range(n):
            d = sum(W[i])
            if d > 0:
                for j in range(n):
                    T[i][j] = W[i][j] / d
        return T

    def fit(self, X, y, num_classes=None):
        """Fit label propagation.

        Args:
            X: All data (labeled + unlabeled)
            y: Labels (-1 for unlabeled)
            num_classes: Number of classes (inferred if None)
        """
        n = _num_rows(X)
        if num_classes is None:
            num_classes = max(yi for yi in y if yi >= 0) + 1

        # Build affinity and transition matrices
        W = self._build_affinity(X)
        T = self._build_transition(W)

        # Initialize label distributions
        Y = [[0.0] * num_classes for _ in range(n)]
        labeled_mask = [False] * n
        for i in range(n):
            if y[i] >= 0:
                Y[i][y[i]] = 1.0
                labeled_mask[i] = True

        # Initial distributions for clamping
        Y_init = [row[:] for row in Y]

        # Iterate
        for iteration in range(self.max_iter):
            Y_new = [[0.0] * num_classes for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if T[i][j] > 0:
                        for c in range(num_classes):
                            Y_new[i][c] += T[i][j] * Y[j][c]

            # Clamp labeled nodes
            for i in range(n):
                if labeled_mask[i]:
                    Y_new[i] = Y_init[i][:]

            # Check convergence
            diff = 0.0
            for i in range(n):
                for c in range(num_classes):
                    diff += abs(Y_new[i][c] - Y[i][c])
            if diff < self.tol:
                Y = Y_new
                break
            Y = Y_new

        # Normalize rows to distributions
        for i in range(n):
            s = sum(Y[i])
            if s > 0:
                Y[i] = [v / s for v in Y[i]]

        self._label_distributions = Y
        self._labels = [_argmax(row) for row in Y]
        return self

    def predict(self, X=None):
        """Return predicted labels for all nodes."""
        return self._labels[:]

    def predict_proba(self, X=None):
        """Return label distributions for all nodes."""
        return [row[:] for row in self._label_distributions]


# ---------------------------------------------------------------------------
# 4. LabelSpreading
# ---------------------------------------------------------------------------

class LabelSpreading:
    """Label spreading with normalized graph Laplacian.

    Uses alpha parameter to control clamping strength.
    alpha=0 means full clamping (like LabelPropagation),
    alpha=1 means no clamping.
    """

    def __init__(self, kernel='rbf', gamma=1.0, alpha=0.2,
                 max_iter=100, tol=1e-4):
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha  # spreading factor
        self.max_iter = max_iter
        self.tol = tol
        self._labels = None
        self._label_distributions = None

    def _build_affinity(self, X):
        """Build affinity matrix (same as LabelPropagation)."""
        n = _num_rows(X)
        rows = [_tensor_row(X, i) for i in range(n)]
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                w = _rbf_kernel(rows[i], rows[j], self.gamma)
                W[i][j] = w
                W[j][i] = w
        return W

    def _build_normalized_laplacian(self, W):
        """Build S = D^{-1/2} W D^{-1/2}."""
        n = len(W)
        # Degree
        D_inv_sqrt = [0.0] * n
        for i in range(n):
            d = sum(W[i])
            if d > 0:
                D_inv_sqrt[i] = 1.0 / math.sqrt(d)

        S = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                S[i][j] = D_inv_sqrt[i] * W[i][j] * D_inv_sqrt[j]
        return S

    def fit(self, X, y, num_classes=None):
        """Fit label spreading.

        Args:
            X: All data (labeled + unlabeled)
            y: Labels (-1 for unlabeled)
            num_classes: Number of classes (inferred if None)
        """
        n = _num_rows(X)
        if num_classes is None:
            num_classes = max(yi for yi in y if yi >= 0) + 1

        W = self._build_affinity(X)
        S = self._build_normalized_laplacian(W)

        # Initialize label distributions
        Y_init = [[0.0] * num_classes for _ in range(n)]
        for i in range(n):
            if y[i] >= 0:
                Y_init[i][y[i]] = 1.0

        Y = [row[:] for row in Y_init]
        alpha = self.alpha

        for iteration in range(self.max_iter):
            # Y_new = alpha * S * Y + (1 - alpha) * Y_init
            Y_new = [[0.0] * num_classes for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if S[i][j] != 0:
                        for c in range(num_classes):
                            Y_new[i][c] += alpha * S[i][j] * Y[j][c]
                for c in range(num_classes):
                    Y_new[i][c] += (1 - alpha) * Y_init[i][c]

            # Check convergence
            diff = 0.0
            for i in range(n):
                for c in range(num_classes):
                    diff += abs(Y_new[i][c] - Y[i][c])
            if diff < self.tol:
                Y = Y_new
                break
            Y = Y_new

        # Normalize
        for i in range(n):
            s = sum(Y[i])
            if s > 0:
                Y[i] = [v / s for v in Y[i]]

        self._label_distributions = Y
        self._labels = [_argmax(row) for row in Y]
        return self

    def predict(self, X=None):
        """Return predicted labels."""
        return self._labels[:]

    def predict_proba(self, X=None):
        """Return label distributions."""
        return [row[:] for row in self._label_distributions]


# ---------------------------------------------------------------------------
# 5. ConsistencyRegularizer (Pi-Model style)
# ---------------------------------------------------------------------------

class ConsistencyRegularizer:
    """Consistency regularization (Pi-Model).

    Enforces that model predictions should be consistent under
    random perturbations of the input.
    """

    def __init__(self, noise_std=0.1, consistency_weight=1.0,
                 ramp_up_epochs=10, seed=42):
        self.noise_std = noise_std
        self.consistency_weight = consistency_weight
        self.ramp_up_epochs = ramp_up_epochs
        self.seed = seed
        self.rng = random.Random(seed)

    def _add_noise(self, X):
        """Add Gaussian noise to input."""
        rows = []
        for i in range(_num_rows(X)):
            row = _tensor_row(X, i)
            noisy = [v + self.rng.gauss(0, self.noise_std) for v in row]
            rows.append(noisy)
        return Tensor(rows)

    def _ramp_weight(self, epoch):
        """Ramp up consistency weight over time."""
        if epoch >= self.ramp_up_epochs:
            return self.consistency_weight
        # Gaussian ramp-up
        p = epoch / self.ramp_up_epochs
        return self.consistency_weight * math.exp(-5 * (1 - p) ** 2)

    def consistency_loss(self, model, X, epoch=0):
        """Compute consistency loss: MSE between predictions on clean and noisy inputs."""
        model.eval()
        clean_out = model.forward(X)
        clean_probs = softmax_batch(clean_out)

        X_noisy = self._add_noise(X)
        noisy_out = model.forward(X_noisy)
        noisy_probs = softmax_batch(noisy_out)

        # MSE between probability distributions
        n = _num_rows(clean_probs)
        total = 0.0
        clean_list = _to_list(clean_probs)
        noisy_list = _to_list(noisy_probs)
        for i in range(n):
            for j in range(len(clean_list[i])):
                d = clean_list[i][j] - noisy_list[i][j]
                total += d * d

        weight = self._ramp_weight(epoch)
        return weight * total / n

    def train_step(self, model, X_labeled, y_labeled, X_unlabeled,
                   loss_fn, optimizer, epoch=0):
        """One training step with supervised + consistency loss."""
        model.train()

        # Supervised forward pass
        out = model.forward(X_labeled)
        sup_loss = loss_fn.forward(out, y_labeled)
        grad = loss_fn.backward(out, y_labeled)
        model.backward(grad)

        # Consistency on unlabeled (compute loss for reporting)
        cons_loss = self.consistency_loss(model, X_unlabeled, epoch)

        # Consistency backward pass on unlabeled
        model.train()
        weight = self._ramp_weight(epoch)
        if weight > 0 and _num_rows(X_unlabeled) > 0:
            clean_out = model.forward(X_unlabeled)
            clean_probs = softmax_batch(clean_out)
            X_noisy = self._add_noise(X_unlabeled)
            noisy_out = model.forward(X_noisy)
            noisy_probs = softmax_batch(noisy_out)

            # Gradient: 2 * weight * (noisy - clean) / n
            n = _num_rows(X_unlabeled)
            noisy_list = _to_list(noisy_probs)
            clean_list = _to_list(clean_probs)
            grad_rows = []
            for i in range(n):
                row = [2 * weight * (noisy_list[i][j] - clean_list[i][j]) / n
                       for j in range(len(noisy_list[i]))]
                grad_rows.append(row)
            cons_grad = Tensor(grad_rows)
            model.backward(cons_grad)

        optimizer.step(model.get_trainable_layers())
        return sup_loss, cons_loss


# ---------------------------------------------------------------------------
# 6. MixUp
# ---------------------------------------------------------------------------

class MixUp:
    """MixUp data augmentation.

    Creates virtual training examples by linear interpolation
    between pairs of examples and their labels.
    """

    def __init__(self, alpha=0.75, seed=42):
        self.alpha = alpha
        self.seed = seed
        self.rng = random.Random(seed)

    def _beta_sample(self, alpha):
        """Sample from Beta(alpha, alpha) using gamma approximation."""
        if alpha <= 0:
            return 0.5
        # Use the Marsaglia and Tsang method for gamma
        g1 = self._gamma_sample(alpha)
        g2 = self._gamma_sample(alpha)
        if g1 + g2 == 0:
            return 0.5
        return g1 / (g1 + g2)

    def _gamma_sample(self, alpha):
        """Sample from Gamma(alpha, 1) distribution."""
        if alpha >= 1:
            d = alpha - 1.0/3.0
            c = 1.0 / math.sqrt(9.0 * d)
            while True:
                x = self.rng.gauss(0, 1)
                v = (1 + c * x) ** 3
                if v > 0:
                    u = self.rng.random()
                    if u < 1 - 0.0331 * x * x * x * x:
                        return d * v
                    if math.log(u + 1e-12) < 0.5 * x * x + d * (1 - v + math.log(v + 1e-12)):
                        return d * v
        else:
            # For alpha < 1: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
            g = self._gamma_sample(alpha + 1)
            u = self.rng.random()
            return g * (u ** (1.0 / alpha))

    def mix(self, X1, y1, X2=None, y2=None):
        """Mix two batches. If X2/y2 not given, shuffles X1/y1.

        Args:
            X1, y1: First batch (Tensor, list/Tensor)
            X2, y2: Second batch (optional, defaults to shuffled X1/y1)

        Returns:
            (X_mixed, y_mixed, lam) where lam is the mixing coefficient
        """
        n = _num_rows(X1)

        if X2 is None:
            # Shuffle X1, y1
            indices = list(range(n))
            self.rng.shuffle(indices)
            X2_rows = [_tensor_row(X1, i) for i in indices]
            y2_list = _to_list(y1)
            y2_list = [y2_list[i] for i in indices]
            X2 = Tensor(X2_rows)
            y2 = y2_list

        lam = self._beta_sample(self.alpha)
        # Ensure lam >= 0.5 for stability
        lam = max(lam, 1 - lam)

        # Mix features
        X1_list = [_tensor_row(X1, i) for i in range(n)]
        X2_list = [_tensor_row(X2, i) for i in range(n)]
        X_mixed = []
        for i in range(n):
            row = [lam * X1_list[i][j] + (1 - lam) * X2_list[i][j]
                   for j in range(len(X1_list[i]))]
            X_mixed.append(row)

        # Mix labels
        y1_list = _to_list(y1)
        y2_list = _to_list(y2) if not isinstance(y2, list) else y2

        if isinstance(y1_list[0], list):
            # One-hot encoded
            y_mixed = []
            for i in range(n):
                row = [lam * y1_list[i][j] + (1 - lam) * y2_list[i][j]
                       for j in range(len(y1_list[i]))]
                y_mixed.append(row)
        else:
            # Integer labels -> soft labels
            y_mixed = (y1_list, y2_list, lam)

        return Tensor(X_mixed), y_mixed, lam


# ---------------------------------------------------------------------------
# 7. MixMatchTrainer
# ---------------------------------------------------------------------------

class MixMatchTrainer:
    """MixMatch: combines consistency regularization, entropy minimization, and MixUp.

    For unlabeled data: average predictions across augmentations,
    sharpen the distribution, then MixUp labeled and unlabeled together.
    """

    def __init__(self, model, num_classes, T=0.5, K=2, alpha=0.75,
                 lambda_u=1.0, noise_std=0.1, loss_fn=None, optimizer=None,
                 seed=42):
        self.model = model
        self.num_classes = num_classes
        self.T = T  # Sharpening temperature
        self.K = K  # Number of augmentations
        self.alpha = alpha  # MixUp alpha
        self.lambda_u = lambda_u  # Unsupervised loss weight
        self.noise_std = noise_std
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.optimizer = optimizer or Adam(lr=0.01)
        self.seed = seed
        self.rng = random.Random(seed)
        self.mixup = MixUp(alpha=alpha, seed=seed)
        self.history = []

    def _augment(self, X):
        """Apply random noise augmentation."""
        rows = []
        for i in range(_num_rows(X)):
            row = _tensor_row(X, i)
            noisy = [v + self.rng.gauss(0, self.noise_std) for v in row]
            rows.append(noisy)
        return Tensor(rows)

    def _sharpen(self, probs, T):
        """Sharpen probability distribution by raising to 1/T power."""
        sharpened = []
        for p in probs:
            powered = [v ** (1.0 / T) for v in p]
            s = sum(powered)
            if s > 0:
                sharpened.append([v / s for v in powered])
            else:
                sharpened.append(p)
        return sharpened

    def train_step(self, X_labeled, y_labeled, X_unlabeled, epoch=0):
        """One MixMatch training step."""
        n_l = _num_rows(X_labeled)
        n_u = _num_rows(X_unlabeled)

        # 1. Augment unlabeled K times and average predictions
        self.model.eval()
        avg_probs = [[0.0] * self.num_classes for _ in range(n_u)]
        for k in range(self.K):
            X_aug = self._augment(X_unlabeled)
            probs = _predict_proba(self.model, X_aug)
            for i in range(n_u):
                for c in range(self.num_classes):
                    avg_probs[i][c] += probs[i][c] / self.K

        # 2. Sharpen
        pseudo_labels = self._sharpen(avg_probs, self.T)

        # 3. Combine labeled + unlabeled with MixUp
        # Augment labeled
        X_l_aug = self._augment(X_labeled)
        y_l_list = _to_list(y_labeled)
        if not isinstance(y_l_list[0], list):
            # Convert to one-hot
            y_l_oh = [[0.0] * self.num_classes for _ in range(n_l)]
            for i in range(n_l):
                y_l_oh[i][y_l_list[i]] = 1.0
            y_l_list = y_l_oh

        X_u_aug = self._augment(X_unlabeled)

        # Combine all
        all_X = [_tensor_row(X_l_aug, i) for i in range(n_l)] + \
                [_tensor_row(X_u_aug, i) for i in range(n_u)]
        all_y = y_l_list + pseudo_labels

        # Shuffle
        indices = list(range(len(all_X)))
        self.rng.shuffle(indices)
        shuffled_X = [all_X[i] for i in indices]
        shuffled_y = [all_y[i] for i in indices]

        # MixUp: labeled with shuffled
        lam = self.mixup._beta_sample(self.alpha)
        lam = max(lam, 1 - lam)

        # Mix labeled portion
        X_l_mixed = []
        y_l_mixed = []
        for i in range(n_l):
            x_row = [lam * _tensor_row(X_l_aug, i)[j] + (1 - lam) * shuffled_X[i][j]
                     for j in range(len(all_X[0]))]
            X_l_mixed.append(x_row)
            y_row = [lam * y_l_list[i][c] + (1 - lam) * shuffled_y[i][c]
                     for c in range(self.num_classes)]
            y_l_mixed.append(y_row)

        # Mix unlabeled portion
        X_u_mixed = []
        y_u_mixed = []
        for i in range(n_u):
            si = n_l + i
            if si < len(shuffled_X):
                x_row = [lam * _tensor_row(X_u_aug, i)[j] + (1 - lam) * shuffled_X[si][j]
                         for j in range(len(all_X[0]))]
                y_row = [lam * pseudo_labels[i][c] + (1 - lam) * shuffled_y[si][c]
                         for c in range(self.num_classes)]
            else:
                x_row = _tensor_row(X_u_aug, i)
                y_row = pseudo_labels[i]
            X_u_mixed.append(x_row)
            y_u_mixed.append(y_row)

        # 4. Supervised loss on mixed labeled
        self.model.train()
        X_l_t = Tensor(X_l_mixed)
        y_l_t = Tensor(y_l_mixed)
        out_l = self.model.forward(X_l_t)
        sup_loss = self.loss_fn.forward(out_l, y_l_t)
        grad_l = self.loss_fn.backward(out_l, y_l_t)
        self.model.backward(grad_l)

        # 5. Unsupervised MSE loss on mixed unlabeled
        unsup_loss = 0.0
        if n_u > 0:
            X_u_t = Tensor(X_u_mixed)
            out_u = self.model.forward(X_u_t)
            probs_u = softmax_batch(out_u)
            y_u_t = Tensor(y_u_mixed)

            # MSE loss
            probs_list = _to_list(probs_u)
            total = 0.0
            for i in range(n_u):
                for c in range(self.num_classes):
                    d = probs_list[i][c] - y_u_mixed[i][c]
                    total += d * d
            unsup_loss = self.lambda_u * total / n_u

            # Gradient for unsupervised
            grad_rows = []
            for i in range(n_u):
                row = [2 * self.lambda_u * (probs_list[i][c] - y_u_mixed[i][c]) / n_u
                       for c in range(self.num_classes)]
                grad_rows.append(row)
            self.model.backward(Tensor(grad_rows))

        self.optimizer.step(self.model.get_trainable_layers())
        return sup_loss, unsup_loss

    def fit(self, X_labeled, y_labeled, X_unlabeled, epochs=50,
            X_val=None, y_val=None, verbose=False):
        """Run MixMatch training."""
        for epoch in range(epochs):
            sup_loss, unsup_loss = self.train_step(
                X_labeled, y_labeled, X_unlabeled, epoch)
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = accuracy(self.model, X_val, y_val)
            self.history.append((epoch, sup_loss, unsup_loss, val_acc))
            if verbose:
                print(f"Epoch {epoch}: sup={sup_loss:.4f} unsup={unsup_loss:.4f}"
                      + (f" val_acc={val_acc:.4f}" if val_acc else ""))
        return self.history

    def predict(self, X):
        """Predict class labels."""
        return predict_classes(self.model, X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return _predict_proba(self.model, X)


# ---------------------------------------------------------------------------
# 8. FixMatchTrainer
# ---------------------------------------------------------------------------

class FixMatchTrainer:
    """FixMatch: strong/weak augmentation with pseudo-labeling.

    Weak augmentation for pseudo-label generation, strong augmentation
    for training. Only uses pseudo-labels above confidence threshold.
    """

    def __init__(self, model, num_classes, threshold=0.95,
                 lambda_u=1.0, weak_noise=0.05, strong_noise=0.2,
                 loss_fn=None, optimizer=None, seed=42):
        self.model = model
        self.num_classes = num_classes
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.weak_noise = weak_noise
        self.strong_noise = strong_noise
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.optimizer = optimizer or Adam(lr=0.01)
        self.seed = seed
        self.rng = random.Random(seed)
        self.history = []

    def _augment(self, X, noise_std):
        """Apply noise augmentation."""
        rows = []
        for i in range(_num_rows(X)):
            row = _tensor_row(X, i)
            noisy = [v + self.rng.gauss(0, noise_std) for v in row]
            rows.append(noisy)
        return Tensor(rows)

    def train_step(self, X_labeled, y_labeled, X_unlabeled, epoch=0):
        """One FixMatch training step."""
        n_u = _num_rows(X_unlabeled)

        # 1. Weak augmentation on unlabeled -> pseudo-labels
        self.model.eval()
        X_weak = self._augment(X_unlabeled, self.weak_noise)
        probs_weak = _predict_proba(self.model, X_weak)

        # Select confident pseudo-labels
        mask = []
        pseudo_y = []
        for i in range(n_u):
            conf = _max_val(probs_weak[i])
            if conf >= self.threshold:
                mask.append(i)
                pseudo_y.append(_argmax(probs_weak[i]))

        # 2. Supervised loss
        self.model.train()
        out_l = self.model.forward(X_labeled)
        sup_loss = self.loss_fn.forward(out_l, y_labeled)
        grad_l = self.loss_fn.backward(out_l, y_labeled)
        self.model.backward(grad_l)

        # 3. Unsupervised loss on strongly augmented confident samples
        unsup_loss = 0.0
        n_pseudo = len(mask)
        if n_pseudo > 0:
            # Strong augmentation on selected unlabeled
            X_strong_rows = [_tensor_row(X_unlabeled, i) for i in mask]
            X_strong = self._augment(Tensor(X_strong_rows), self.strong_noise)
            y_pseudo_oh = one_hot(pseudo_y, self.num_classes)

            out_u = self.model.forward(X_strong)
            unsup_loss = self.lambda_u * self.loss_fn.forward(out_u, y_pseudo_oh)
            grad_u = self.loss_fn.backward(out_u, y_pseudo_oh)

            # Scale gradient
            grad_list = _to_list(grad_u)
            scaled = [[v * self.lambda_u for v in row] for row in grad_list]
            self.model.backward(Tensor(scaled))

        self.optimizer.step(self.model.get_trainable_layers())
        return sup_loss, unsup_loss, n_pseudo

    def fit(self, X_labeled, y_labeled, X_unlabeled, epochs=50,
            X_val=None, y_val=None, verbose=False):
        """Run FixMatch training."""
        # Convert labels to one-hot for training
        y_oh = one_hot(y_labeled, self.num_classes)
        for epoch in range(epochs):
            sup_loss, unsup_loss, n_pseudo = self.train_step(
                X_labeled, y_oh, X_unlabeled, epoch)
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = accuracy(self.model, X_val, y_val)
            self.history.append((epoch, sup_loss, unsup_loss, n_pseudo, val_acc))
            if verbose:
                print(f"Epoch {epoch}: sup={sup_loss:.4f} unsup={unsup_loss:.4f} "
                      f"pseudo={n_pseudo}" +
                      (f" val_acc={val_acc:.4f}" if val_acc else ""))
        return self.history

    def predict(self, X):
        """Predict class labels."""
        return predict_classes(self.model, X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return _predict_proba(self.model, X)


# ---------------------------------------------------------------------------
# 9. SemiSupervisedTrainer (General orchestrator)
# ---------------------------------------------------------------------------

class SemiSupervisedTrainer:
    """General semi-supervised training orchestrator.

    Combines supervised loss with an unsupervised consistency/pseudo-label
    component. Supports multiple strategies.
    """

    def __init__(self, model, strategy='self_training', num_classes=None,
                 loss_fn=None, optimizer=None, seed=42, **kwargs):
        self.model = model
        self.strategy = strategy
        self.num_classes = num_classes
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.optimizer = optimizer or Adam(lr=0.01)
        self.seed = seed
        self.kwargs = kwargs
        self.history = []
        self._trainer = None

    def _create_trainer(self, X_labeled, y_labeled):
        """Create the appropriate trainer based on strategy."""
        if self.num_classes is None:
            self.num_classes = max(y_labeled) + 1

        if self.strategy == 'self_training':
            self._trainer = SelfTrainer(
                self.model,
                threshold=self.kwargs.get('threshold', 0.9),
                max_iter=self.kwargs.get('max_iter', 10),
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                epochs_per_iter=self.kwargs.get('epochs_per_iter', 10),
                seed=self.seed
            )
        elif self.strategy == 'fixmatch':
            self._trainer = FixMatchTrainer(
                self.model, self.num_classes,
                threshold=self.kwargs.get('threshold', 0.95),
                lambda_u=self.kwargs.get('lambda_u', 1.0),
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                seed=self.seed
            )
        elif self.strategy == 'mixmatch':
            self._trainer = MixMatchTrainer(
                self.model, self.num_classes,
                T=self.kwargs.get('T', 0.5),
                K=self.kwargs.get('K', 2),
                lambda_u=self.kwargs.get('lambda_u', 1.0),
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                seed=self.seed
            )
        elif self.strategy == 'consistency':
            self._trainer = ConsistencyRegularizer(
                noise_std=self.kwargs.get('noise_std', 0.1),
                consistency_weight=self.kwargs.get('consistency_weight', 1.0),
                seed=self.seed
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def fit(self, X_labeled, y_labeled, X_unlabeled, epochs=50,
            X_val=None, y_val=None, verbose=False):
        """Run semi-supervised training."""
        self._create_trainer(X_labeled, y_labeled)

        if self.strategy == 'self_training':
            hist = self._trainer.fit(X_labeled, y_labeled, X_unlabeled,
                                     num_classes=self.num_classes,
                                     X_val=X_val, y_val=y_val)
            self.history = hist
        elif self.strategy in ('fixmatch', 'mixmatch'):
            hist = self._trainer.fit(X_labeled, y_labeled, X_unlabeled,
                                     epochs=epochs, X_val=X_val, y_val=y_val,
                                     verbose=verbose)
            self.history = hist
        elif self.strategy == 'consistency':
            y_oh = one_hot(y_labeled, self.num_classes)
            for epoch in range(epochs):
                sup_loss, cons_loss = self._trainer.train_step(
                    self.model, X_labeled, y_oh, X_unlabeled,
                    self.loss_fn, self.optimizer, epoch)
                val_acc = None
                if X_val is not None and y_val is not None:
                    val_acc = accuracy(self.model, X_val, y_val)
                self.history.append((epoch, sup_loss, cons_loss, val_acc))

        return self.history

    def predict(self, X):
        """Predict class labels."""
        if self._trainer and hasattr(self._trainer, 'predict'):
            return self._trainer.predict(X)
        return predict_classes(self.model, X)

    def predict_proba(self, X):
        """Predict probabilities."""
        if self._trainer and hasattr(self._trainer, 'predict_proba'):
            return self._trainer.predict_proba(X)
        return _predict_proba(self.model, X)


# ---------------------------------------------------------------------------
# 10. SemiSupervisedMetrics
# ---------------------------------------------------------------------------

class SemiSupervisedMetrics:
    """Evaluation metrics for semi-supervised learning."""

    @staticmethod
    def label_utilization(n_labeled, n_total):
        """Fraction of data that is labeled."""
        return n_labeled / n_total if n_total > 0 else 0.0

    @staticmethod
    def pseudo_label_accuracy(pseudo_labels, true_labels):
        """Accuracy of pseudo-labels compared to ground truth."""
        correct = 0
        n = len(pseudo_labels)
        for i in range(n):
            if pseudo_labels[i] == true_labels[i]:
                correct += 1
        return correct / n if n > 0 else 0.0

    @staticmethod
    def ssl_gain(acc_supervised, acc_ssl):
        """Improvement from SSL over purely supervised baseline."""
        return acc_ssl - acc_supervised

    @staticmethod
    def effective_label_ratio(acc_ssl, acc_curve):
        """Estimate how many labels would be needed for same accuracy with supervised only.

        acc_curve: list of (n_labels, accuracy) from supervised learning curve
        Returns: ratio of effective labels to actual labels used
        """
        if not acc_curve:
            return 1.0
        # Find where supervised achieves same accuracy
        for n_labels, acc in acc_curve:
            if acc >= acc_ssl:
                return n_labels
        # Never reached -- return last point
        return acc_curve[-1][0]

    @staticmethod
    def confidence_histogram(probs, bins=10):
        """Histogram of prediction confidences."""
        confidences = [_max_val(p) for p in probs]
        hist = [0] * bins
        for c in confidences:
            b = min(int(c * bins), bins - 1)
            hist[b] += 1
        return hist

    @staticmethod
    def class_balance(labels, num_classes=None):
        """Check class balance in predicted/pseudo labels."""
        if num_classes is None:
            num_classes = max(labels) + 1
        counts = [0] * num_classes
        for l in labels:
            if 0 <= l < num_classes:
                counts[l] += 1
        total = sum(counts)
        if total == 0:
            return [0.0] * num_classes
        return [c / total for c in counts]

    @staticmethod
    def entropy_score(probs):
        """Average entropy of predictions (lower = more confident)."""
        total = 0.0
        n = len(probs)
        for p in probs:
            total += _entropy(p)
        return total / n if n > 0 else 0.0

    @staticmethod
    def summary(history, strategy='generic'):
        """Generate summary dict from training history."""
        if not history:
            return {'iterations': 0}
        result = {'iterations': len(history)}
        if strategy in ('fixmatch', 'mixmatch'):
            result['final_sup_loss'] = history[-1][1]
            result['final_unsup_loss'] = history[-1][2]
            if len(history[-1]) > 3 and history[-1][-1] is not None:
                result['final_val_acc'] = history[-1][-1]
        elif strategy == 'self_training':
            result['final_n_labeled'] = history[-1][1]
            result['total_pseudo'] = sum(h[2] for h in history)
            if history[-1][3] is not None:
                result['final_val_acc'] = history[-1][3]
        return result
