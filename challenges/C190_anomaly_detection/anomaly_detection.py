"""
C190: Anomaly Detection
========================
From-scratch anomaly detection algorithms using only NumPy.

Components:
- ZScoreDetector: Statistical z-score based detection
- IQRDetector: Interquartile range method
- EWMADetector: Exponentially Weighted Moving Average control charts
- IsolationForest: Isolation Forest ensemble method
- LOF: Local Outlier Factor (density-based)
- DBSCAN: Density-based clustering for anomaly detection
- RobustZScore: Modified z-score using median absolute deviation
- MultivariateDetector: Mahalanobis distance based
- OneClassSVM: Simple one-class SVM (linear kernel)
- EnsembleDetector: Combine multiple detectors with voting
"""

import numpy as np
from collections import namedtuple

AnomalyResult = namedtuple('AnomalyResult', ['scores', 'labels', 'threshold'])


# ============================================================
# Z-Score Detector
# ============================================================
class ZScoreDetector:
    """Detect anomalies using z-score (standard deviations from mean)."""

    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        z = np.abs((X - self.mean_) / self.std_)
        return np.max(z, axis=1)

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > self.threshold).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=self.threshold)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# Robust Z-Score (MAD-based)
# ============================================================
class RobustZScoreDetector:
    """Modified z-score using median and MAD (median absolute deviation)."""

    def __init__(self, threshold=3.5):
        self.threshold = threshold
        self.median_ = None
        self.mad_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)
        self.mad_[self.mad_ == 0] = 1e-10
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # 0.6745 is the 0.75th quantile of the standard normal distribution
        modified_z = 0.6745 * np.abs(X - self.median_) / self.mad_
        return np.max(modified_z, axis=1)

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > self.threshold).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=self.threshold)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# IQR Detector
# ============================================================
class IQRDetector:
    """Detect anomalies using interquartile range (Tukey's fences)."""

    def __init__(self, factor=1.5):
        self.factor = factor
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.lower_ = None
        self.upper_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        self.iqr_ = self.q3_ - self.q1_
        self.iqr_[self.iqr_ == 0] = 1e-10
        self.lower_ = self.q1_ - self.factor * self.iqr_
        self.upper_ = self.q3_ + self.factor * self.iqr_
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # Distance outside fences, normalized by IQR
        below = np.maximum(0, self.lower_ - X) / self.iqr_
        above = np.maximum(0, X - self.upper_) / self.iqr_
        return np.max(below + above, axis=1)

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > 0).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=0.0)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# EWMA Control Chart Detector
# ============================================================
class EWMADetector:
    """Exponentially Weighted Moving Average control chart for streaming anomaly detection."""

    def __init__(self, alpha=0.3, n_sigma=3.0):
        self.alpha = alpha
        self.n_sigma = n_sigma
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10
        return self

    def score(self, X):
        """Compute EWMA deviation scores."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        scores = np.zeros(n)
        ewma = self.mean_.copy()
        for i in range(n):
            ewma = self.alpha * X[i] + (1 - self.alpha) * ewma
            # EWMA variance: sigma^2 * (alpha/(2-alpha)) * (1 - (1-alpha)^(2*(i+1)))
            factor = (self.alpha / (2 - self.alpha)) * (1 - (1 - self.alpha) ** (2 * (i + 1)))
            ewma_std = self.std_ * np.sqrt(factor)
            ewma_std[ewma_std == 0] = 1e-10
            deviation = np.abs(ewma - self.mean_) / ewma_std
            scores[i] = np.max(deviation)
        return scores

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > self.n_sigma).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=self.n_sigma)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def score_online(self, X):
        """Return per-step EWMA values and control limits for charting."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        ewma_values = np.zeros((n, X.shape[1]))
        ucl = np.zeros((n, X.shape[1]))
        lcl = np.zeros((n, X.shape[1]))
        ewma = self.mean_.copy()
        for i in range(n):
            ewma = self.alpha * X[i] + (1 - self.alpha) * ewma
            ewma_values[i] = ewma
            factor = (self.alpha / (2 - self.alpha)) * (1 - (1 - self.alpha) ** (2 * (i + 1)))
            ewma_std = self.std_ * np.sqrt(factor)
            ucl[i] = self.mean_ + self.n_sigma * ewma_std
            lcl[i] = self.mean_ - self.n_sigma * ewma_std
        return {'ewma': ewma_values, 'ucl': ucl, 'lcl': lcl}


# ============================================================
# Isolation Forest
# ============================================================
class IsolationTree:
    """Single isolation tree."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.is_leaf = False

    def fit(self, X, depth=0, rng=None):
        n, d = X.shape
        self.size = n
        if n <= 1 or depth >= self.max_depth:
            self.is_leaf = True
            return self
        # Random feature and split
        self.split_feature = rng.integers(0, d)
        col = X[:, self.split_feature]
        min_val, max_val = col.min(), col.max()
        if min_val == max_val:
            self.is_leaf = True
            return self
        self.split_value = rng.uniform(min_val, max_val)
        mask = col < self.split_value
        self.left = IsolationTree(self.max_depth).fit(X[mask], depth + 1, rng)
        self.right = IsolationTree(self.max_depth).fit(X[~mask], depth + 1, rng)
        self.is_leaf = False
        return self

    def path_length(self, x):
        if self.is_leaf:
            return _c(self.size)
        if x[self.split_feature] < self.split_value:
            return 1 + self.left.path_length(x)
        else:
            return 1 + self.right.path_length(x)


def _c(n):
    """Average path length of unsuccessful search in BST."""
    if n <= 1:
        return 0
    if n == 2:
        return 1
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationForest:
    """Isolation Forest for anomaly detection.

    Anomalies have shorter average path lengths in random trees.
    """

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees_ = []
        self.threshold_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        sample_size = min(self.max_samples, n)
        max_depth = int(np.ceil(np.log2(max(sample_size, 2))))
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=sample_size, replace=False) if n > sample_size else np.arange(n)
            tree = IsolationTree(max_depth).fit(X[idx], rng=rng)
            self.trees_.append(tree)
        # Set threshold based on contamination
        scores = self.score(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def score(self, X):
        """Anomaly score: higher = more anomalous. Range roughly [0, 1]."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        avg_path = np.zeros(n)
        for tree in self.trees_:
            for i in range(n):
                avg_path[i] += tree.path_length(X[i])
        avg_path /= len(self.trees_)
        # Anomaly score: 2^(-avg_path / c(sample_size))
        c_n = _c(min(self.max_samples, n))
        if c_n == 0:
            c_n = 1
        scores = 2 ** (-avg_path / c_n)
        return scores

    def predict(self, X):
        scores = self.score(X)
        if self.threshold_ is None:
            threshold = 0.5
        else:
            threshold = self.threshold_
        labels = (scores > threshold).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=threshold)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# Local Outlier Factor (LOF)
# ============================================================
class LOF:
    """Local Outlier Factor -- density-based anomaly detection.

    Compares local density of a point to its neighbors.
    LOF >> 1 indicates anomaly.
    """

    def __init__(self, k=20, contamination=0.1):
        self.k = k
        self.contamination = contamination
        self.X_train_ = None
        self.lrd_ = None
        self.threshold_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train_ = X
        n = X.shape[0]
        k = min(self.k, n - 1)
        if k < 1:
            self.lrd_ = np.ones(n)
            return self

        # Compute distance matrix
        dists = self._pairwise_distances(X, X)
        # k-distance and k-nearest neighbors
        self.k_distances_ = np.zeros(n)
        self.neighbors_ = []
        for i in range(n):
            sorted_idx = np.argsort(dists[i])
            # Skip self (index 0)
            nn_idx = sorted_idx[1:k + 1]
            self.neighbors_.append(nn_idx)
            self.k_distances_[i] = dists[i, nn_idx[-1]] if len(nn_idx) > 0 else 0

        # Reachability distance
        # reach_dist(A, B) = max(k_distance(B), dist(A, B))
        # Local reachability density
        self.lrd_ = np.zeros(n)
        for i in range(n):
            nn = self.neighbors_[i]
            if len(nn) == 0:
                self.lrd_[i] = 1.0
                continue
            reach_dists = np.maximum(self.k_distances_[nn], dists[i, nn])
            avg_reach = np.mean(reach_dists)
            self.lrd_[i] = 1.0 / max(avg_reach, 1e-10)

        # Compute LOF scores for training data to set threshold
        scores = self._compute_lof(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._compute_lof(X)

    def _compute_lof(self, X):
        n = X.shape[0]
        n_train = self.X_train_.shape[0]
        k = min(self.k, n_train - 1)
        if k < 1:
            return np.ones(n)

        dists = self._pairwise_distances(X, self.X_train_)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(dists[i])
            # If X is training data, skip self
            nn_idx = sorted_idx[:k]
            if len(nn_idx) == 0:
                scores[i] = 1.0
                continue
            # Reachability distances
            reach_dists = np.maximum(self.k_distances_[nn_idx], dists[i, nn_idx])
            avg_reach = np.mean(reach_dists)
            lrd_point = 1.0 / max(avg_reach, 1e-10)
            # LOF = mean(lrd(neighbors)) / lrd(point)
            avg_lrd_nn = np.mean(self.lrd_[nn_idx])
            scores[i] = avg_lrd_nn / max(lrd_point, 1e-10)
        return scores

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > self.threshold_).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=self.threshold_)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    @staticmethod
    def _pairwise_distances(A, B):
        """Compute Euclidean distance matrix between rows of A and B."""
        A_sq = np.sum(A ** 2, axis=1, keepdims=True)
        B_sq = np.sum(B ** 2, axis=1, keepdims=True)
        dist_sq = A_sq + B_sq.T - 2 * A @ B.T
        dist_sq = np.maximum(dist_sq, 0)
        return np.sqrt(dist_sq)


# ============================================================
# DBSCAN-based Anomaly Detection
# ============================================================
class DBSCANAnomaly:
    """DBSCAN clustering where noise points are anomalies."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.X_train_ = X
        n = X.shape[0]
        dists = LOF._pairwise_distances(X, X)
        # DBSCAN
        labels = np.full(n, -1)
        cluster_id = 0
        visited = np.zeros(n, dtype=bool)
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = np.where(dists[i] <= self.eps)[0]
            if len(neighbors) < self.min_samples:
                continue  # noise
            # Expand cluster
            labels[i] = cluster_id
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = np.where(dists[q] <= self.eps)[0]
                    if len(q_neighbors) >= self.min_samples:
                        seed_set.extend(q_neighbors.tolist())
                if labels[q] == -1:
                    labels[q] = cluster_id
                j += 1
            cluster_id += 1
        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        dists = LOF._pairwise_distances(X, self.X_train_)
        scores = np.zeros(n)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            # Find nearest core point distance
            min_dist = np.min(dists[i])
            neighbors = np.where(dists[i] <= self.eps)[0]
            core_neighbors = [j for j in neighbors if self.labels_[j] != -1]
            if len(core_neighbors) == 0:
                labels[i] = 1  # anomaly
                scores[i] = min_dist / max(self.eps, 1e-10)
            else:
                labels[i] = 0
                scores[i] = 0
        return AnomalyResult(scores=scores, labels=labels, threshold=self.eps)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# Multivariate Detector (Mahalanobis Distance)
# ============================================================
class MultivariateDetector:
    """Detect multivariate anomalies using Mahalanobis distance."""

    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.mean_ = None
        self.cov_inv_ = None
        self.threshold_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mean_ = np.mean(X, axis=0)
        cov = np.cov(X.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Regularize
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv_ = np.linalg.inv(cov)
        scores = self.score(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        diff = X - self.mean_
        # Mahalanobis: sqrt(diff @ cov_inv @ diff.T)
        left = diff @ self.cov_inv_
        mahal = np.sqrt(np.sum(left * diff, axis=1))
        return mahal

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > self.threshold_).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=self.threshold_)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# One-Class SVM (simplified, linear kernel)
# ============================================================
class OneClassSVM:
    """Simplified one-class SVM using SGD with RBF kernel approximation."""

    def __init__(self, nu=0.1, kernel='rbf', gamma='scale', max_iter=1000, random_state=None):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self.w_ = None
        self.rho_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape

        # RBF kernel approximation using random Fourier features
        if self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1.0 / (d * np.var(X)) if np.var(X) > 0 else 1.0
            else:
                gamma = self.gamma
            n_features = min(100, 2 * d + 10)
            self.random_weights_ = rng.normal(0, np.sqrt(2 * gamma), (d, n_features))
            self.random_offset_ = rng.uniform(0, 2 * np.pi, n_features)
            Z = np.sqrt(2.0 / n_features) * np.cos(X @ self.random_weights_ + self.random_offset_)
        else:
            Z = X

        # SGD to find w, rho such that w.T @ z >= rho for normal points
        d_z = Z.shape[1]
        w = rng.normal(0, 0.01, d_z)
        rho = 0.0
        lr = 0.01
        for epoch in range(self.max_iter):
            perm = rng.permutation(n)
            for i in perm:
                z = Z[i]
                decision = np.dot(w, z) - rho
                if decision < 0:
                    # Misclassified: push w toward z
                    w += lr * z
                    rho -= lr * self.nu
                else:
                    # Correctly classified: regularize
                    w *= (1 - lr * 0.001)
                    rho += lr * (1 - self.nu) * 0.001
            lr *= 0.999  # decay

        self.w_ = w
        self.rho_ = rho
        # Set threshold based on training scores
        scores = self._decision_function(Z)
        self.threshold_ = np.percentile(-scores, 100 * (1 - self.nu))
        return self

    def _transform(self, X):
        if self.kernel == 'rbf':
            n_features = self.random_weights_.shape[1]
            return np.sqrt(2.0 / n_features) * np.cos(X @ self.random_weights_ + self.random_offset_)
        return X

    def _decision_function(self, Z):
        return Z @ self.w_ - self.rho_

    def score(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Z = self._transform(X)
        # Negative decision = more anomalous
        return -self._decision_function(Z)

    def predict(self, X):
        scores = self.score(X)
        labels = (scores > self.threshold_).astype(int)
        return AnomalyResult(scores=scores, labels=labels, threshold=self.threshold_)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# Ensemble Detector
# ============================================================
class EnsembleDetector:
    """Combine multiple anomaly detectors with voting or score averaging."""

    def __init__(self, detectors, method='vote', weights=None):
        """
        Args:
            detectors: list of (name, detector) tuples
            method: 'vote' for majority voting, 'average' for score averaging
            weights: optional weights for each detector
        """
        self.detectors = detectors
        self.method = method
        self.weights = weights if weights is not None else np.ones(len(detectors))
        self.weights = np.asarray(self.weights, dtype=float)
        self.weights /= self.weights.sum()

    def fit(self, X):
        for name, det in self.detectors:
            det.fit(X)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]

        if self.method == 'vote':
            votes = np.zeros(n)
            for i, (name, det) in enumerate(self.detectors):
                result = det.predict(X)
                votes += self.weights[i] * result.labels
            labels = (votes > 0.5).astype(int)
            return AnomalyResult(scores=votes, labels=labels, threshold=0.5)
        else:  # average
            total_scores = np.zeros(n)
            for i, (name, det) in enumerate(self.detectors):
                scores = det.score(X)
                # Normalize scores to [0, 1]
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    scores = (scores - s_min) / (s_max - s_min)
                total_scores += self.weights[i] * scores
            labels = (total_scores > 0.5).astype(int)
            return AnomalyResult(scores=total_scores, labels=labels, threshold=0.5)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


# ============================================================
# Grubbs' Test (single outlier test)
# ============================================================
class GrubbsTest:
    """Grubbs' test for detecting a single outlier in univariate data."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def test(self, X):
        """Test for single outlier. Returns dict with outlier info."""
        X = np.asarray(X, dtype=float).ravel()
        n = len(X)
        if n < 3:
            return {'outlier_index': None, 'outlier_value': None, 'statistic': 0, 'is_outlier': False}
        mean = np.mean(X)
        std = np.std(X, ddof=1)
        if std == 0:
            return {'outlier_index': None, 'outlier_value': None, 'statistic': 0, 'is_outlier': False}
        # Grubbs statistic: max |x_i - mean| / std
        abs_dev = np.abs(X - mean)
        idx = np.argmax(abs_dev)
        G = abs_dev[idx] / std
        # Critical value approximation using t-distribution
        # For simplicity, use a threshold based on n
        # G_critical ≈ ((n-1)/sqrt(n)) * sqrt(t^2 / (n - 2 + t^2))
        # We approximate with a simpler formula
        from scipy.stats import t as t_dist
        t_val = t_dist.ppf(1 - self.alpha / (2 * n), n - 2)
        G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_val ** 2 / (n - 2 + t_val ** 2))
        return {
            'outlier_index': int(idx),
            'outlier_value': float(X[idx]),
            'statistic': float(G),
            'critical_value': float(G_critical),
            'is_outlier': G > G_critical
        }

    def detect_all(self, X, max_outliers=None):
        """Iteratively apply Grubbs' test to find multiple outliers."""
        X = np.asarray(X, dtype=float).ravel()
        outliers = []
        remaining = X.copy()
        remaining_indices = np.arange(len(X))
        if max_outliers is None:
            max_outliers = len(X) // 2
        for _ in range(max_outliers):
            if len(remaining) < 3:
                break
            result = self.test(remaining)
            if not result['is_outlier']:
                break
            orig_idx = remaining_indices[result['outlier_index']]
            outliers.append({'index': int(orig_idx), 'value': result['outlier_value']})
            mask = np.ones(len(remaining), dtype=bool)
            mask[result['outlier_index']] = False
            remaining = remaining[mask]
            remaining_indices = remaining_indices[mask]
        return outliers


# ============================================================
# CUSUM (Cumulative Sum) Change Detection
# ============================================================
class CUSUM:
    """Cumulative Sum control chart for detecting mean shifts in sequential data."""

    def __init__(self, threshold=5.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float).ravel()
        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        if self.std_ == 0:
            self.std_ = 1e-10
        return self

    def detect(self, X):
        """Detect change points using CUSUM."""
        X = np.asarray(X, dtype=float).ravel()
        n = len(X)
        normalized = (X - self.mean_) / self.std_
        # Upper and lower CUSUM
        s_pos = np.zeros(n)
        s_neg = np.zeros(n)
        alarms = []
        for i in range(n):
            s_pos[i] = max(0, (s_pos[i - 1] if i > 0 else 0) + normalized[i] - self.drift)
            s_neg[i] = max(0, (s_neg[i - 1] if i > 0 else 0) - normalized[i] - self.drift)
            if s_pos[i] > self.threshold or s_neg[i] > self.threshold:
                alarms.append(i)
        return {
            's_pos': s_pos,
            's_neg': s_neg,
            'alarms': alarms,
            'threshold': self.threshold
        }

    def fit_detect(self, X):
        self.fit(X)
        return self.detect(X)


# ============================================================
# Streaming Anomaly Detector
# ============================================================
class StreamingDetector:
    """Online anomaly detection with sliding window statistics."""

    def __init__(self, window_size=100, n_sigma=3.0):
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.buffer = []
        self.mean_ = 0.0
        self.var_ = 0.0
        self.n_ = 0

    def update(self, value):
        """Add a new value and return anomaly score."""
        value = float(value)
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Welford's online algorithm
        self.n_ += 1
        if self.n_ == 1:
            self.mean_ = value
            self.var_ = 0.0
            return {'value': value, 'score': 0.0, 'is_anomaly': False, 'mean': self.mean_, 'std': 0.0}

        old_mean = self.mean_
        self.mean_ += (value - self.mean_) / min(self.n_, self.window_size)
        self.var_ += (value - old_mean) * (value - self.mean_)
        if self.n_ > 1:
            std = np.sqrt(self.var_ / min(self.n_ - 1, self.window_size))
        else:
            std = 0.0

        # Use window stats if buffer full
        if len(self.buffer) >= self.window_size:
            buf = np.array(self.buffer)
            self.mean_ = np.mean(buf)
            std = np.std(buf)

        if std == 0:
            score = 0.0
        else:
            score = abs(value - self.mean_) / std

        return {
            'value': value,
            'score': score,
            'is_anomaly': score > self.n_sigma,
            'mean': self.mean_,
            'std': std
        }

    def update_batch(self, values):
        """Process a batch of values."""
        return [self.update(v) for v in values]
