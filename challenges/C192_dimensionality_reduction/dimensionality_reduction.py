"""
C192: Dimensionality Reduction
================================
From-scratch dimensionality reduction algorithms using only NumPy.

Components:
- PCA: Principal Component Analysis (eigendecomposition + SVD)
- KernelPCA: Kernel PCA (RBF, polynomial, linear kernels)
- IncrementalPCA: Online/streaming PCA
- tSNE: t-Distributed Stochastic Neighbor Embedding
- UMAP: Uniform Manifold Approximation (simplified from-scratch)
- LDA: Linear Discriminant Analysis (supervised)
- FactorAnalysis: Factor Analysis with EM
- MDS: Multidimensional Scaling (classical + metric)
- Isomap: Isometric Mapping (geodesic distances + MDS)
- LLE: Locally Linear Embedding
- TruncatedSVD: Randomized truncated SVD for sparse-like data
- NMF: Non-negative Matrix Factorization (multiplicative updates)
"""

import numpy as np
from collections import defaultdict


# ============================================================
# PCA
# ============================================================
class PCA:
    """Principal Component Analysis via eigendecomposition."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        if self.n_components is None:
            self.n_components = min(n_samples, n_features)

        if n_samples <= 1:
            # Degenerate case: no variance
            self.components_ = np.eye(n_features)[:self.n_components]
            self.explained_variance_ = np.zeros(self.n_components)
            self.explained_variance_ratio_ = np.zeros(self.n_components)
            self.singular_values_ = np.zeros(self.n_components)
            return self

        cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var if total_var > 0 else self.explained_variance_
        self.singular_values_ = np.sqrt(self.explained_variance_ * (n_samples - 1))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X - self.mean_, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        X_reduced = np.asarray(X_reduced, dtype=np.float64)
        return np.dot(X_reduced, self.components_) + self.mean_

    def get_covariance(self):
        return np.dot(self.components_.T * self.explained_variance_, self.components_)


# ============================================================
# KernelPCA
# ============================================================
class KernelPCA:
    """Kernel PCA for non-linear dimensionality reduction."""

    def __init__(self, n_components=2, kernel='rbf', gamma=1.0, degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.X_fit_ = None
        self.alphas_ = None
        self.lambdas_ = None
        self.K_fit_centered_ = None

    def _compute_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == 'rbf':
            sq_X = np.sum(X ** 2, axis=1)[:, None]
            sq_Y = np.sum(Y ** 2, axis=1)[None, :]
            dist = sq_X + sq_Y - 2 * np.dot(X, Y.T)
            return np.exp(-self.gamma * np.maximum(dist, 0))
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(X, Y.T) + self.coef0) ** self.degree
        elif self.kernel == 'linear':
            return np.dot(X, Y.T)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _center_kernel(self, K, K_fit=None):
        n = K.shape[0]
        if K_fit is None:
            # Training: center in feature space
            row_mean = K.mean(axis=1)
            col_mean = K.mean(axis=0)
            total_mean = K.mean()
            return K - row_mean[:, None] - col_mean[None, :] + total_mean
        else:
            # Transform: center using training statistics
            n_train = K_fit.shape[0]
            K_train_mean = K_fit.mean(axis=0)
            K_new_row_mean = K.mean(axis=1)
            K_train_total_mean = K_fit.mean()
            return K - K_new_row_mean[:, None] - K_train_mean[None, :] + K_train_total_mean

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.X_fit_ = X.copy()
        K = self._compute_kernel(X)
        K_centered = self._center_kernel(K)
        self.K_fit_centered_ = K

        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep positive eigenvalues
        n = min(self.n_components, np.sum(eigenvalues > 1e-10))
        if n == 0:
            n = self.n_components
        self.lambdas_ = eigenvalues[:n]
        self.alphas_ = eigenvectors[:, :n]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        K = self._compute_kernel(X, self.X_fit_)
        K_centered = self._center_kernel(K, self._compute_kernel(self.X_fit_))
        scaling = np.where(self.lambdas_ > 1e-10, 1.0 / np.sqrt(self.lambdas_), 0)
        return np.dot(K_centered, self.alphas_ * scaling)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# IncrementalPCA
# ============================================================
class IncrementalPCA:
    """Incremental PCA for streaming/batch data."""

    def __init__(self, n_components=2, batch_size=None):
        self.n_components = n_components
        self.batch_size = batch_size
        self.components_ = None
        self.mean_ = None
        self.var_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_samples_seen_ = 0
        self.singular_values_ = None

    def partial_fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.n_samples_seen_ == 0:
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)

        # Update mean and variance (Welford)
        old_mean = self.mean_.copy()
        old_n = self.n_samples_seen_
        new_n = old_n + n_samples
        new_mean = (old_n * old_mean + n_samples * X.mean(axis=0)) / new_n

        new_var_batch = X.var(axis=0) if n_samples > 1 else np.zeros(n_features)
        if old_n > 0:
            old_var = self.var_
            new_var = (old_n * (old_var + (old_mean - new_mean) ** 2) +
                       n_samples * (new_var_batch + (X.mean(axis=0) - new_mean) ** 2)) / new_n
        else:
            new_var = new_var_batch

        self.mean_ = new_mean
        self.var_ = new_var
        self.n_samples_seen_ = new_n

        # Combine old components with new data
        X_centered = X - new_mean
        if self.components_ is not None:
            # Build matrix from old components weighted by singular values and new data
            mean_correction = np.sqrt(old_n * n_samples / new_n) * (old_mean - X.mean(axis=0))
            X_combined = np.vstack([
                self.singular_values_[:, None] * self.components_,
                X_centered,
                mean_correction[None, :]
            ])
        else:
            X_combined = X_centered

        U, S, Vt = np.linalg.svd(X_combined, full_matrices=False)
        nc = min(self.n_components, Vt.shape[0], n_features)
        self.components_ = Vt[:nc]
        self.singular_values_ = S[:nc]
        self.explained_variance_ = S[:nc] ** 2 / (new_n - 1)
        total_var = new_var.sum() if new_var.sum() > 0 else 1.0
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.n_samples_seen_ = 0
        self.components_ = None
        bs = self.batch_size or X.shape[0]
        for i in range(0, X.shape[0], bs):
            self.partial_fit(X[i:i + bs])
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X - self.mean_, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# t-SNE
# ============================================================
class TSNE:
    """t-Distributed Stochastic Neighbor Embedding."""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, momentum=0.8, final_momentum=0.8,
                 early_exaggeration=4.0, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.embedding_ = None
        self.kl_divergence_ = None

    def _compute_pairwise_distances(self, X):
        sq = np.sum(X ** 2, axis=1)
        D = sq[:, None] + sq[None, :] - 2 * np.dot(X, X.T)
        return np.maximum(D, 0)

    def _binary_search_perplexity(self, distances, target_perplexity, tol=1e-5, max_iter=50):
        n = distances.shape[0]
        P = np.zeros((n, n))
        target_entropy = np.log(target_perplexity)

        for i in range(n):
            lo, hi = -1000.0, 1000.0
            beta = 1.0  # 1 / (2 * sigma^2)

            for _ in range(max_iter):
                d_i = distances[i].copy()
                d_i[i] = np.inf
                exp_d = np.exp(-d_i * beta)
                sum_exp = exp_d.sum()
                if sum_exp == 0:
                    sum_exp = 1e-10
                p_i = exp_d / sum_exp
                entropy = -np.sum(p_i * np.log(np.maximum(p_i, 1e-12)))

                diff = entropy - target_entropy
                if abs(diff) < tol:
                    break
                if diff > 0:
                    lo = beta
                    beta = (beta + hi) / 2 if hi < 900 else beta * 2
                else:
                    hi = beta
                    beta = (beta + lo) / 2 if lo > -900 else beta / 2

            P[i] = p_i

        return P

    def _compute_joint_probabilities(self, X):
        D = self._compute_pairwise_distances(X)
        P = self._binary_search_perplexity(D, self.perplexity)
        P = (P + P.T) / (2 * P.shape[0])
        P = np.maximum(P, 1e-12)
        return P

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]

        # Compute joint probabilities
        P = self._compute_joint_probabilities(X)

        # Initialize embedding
        Y = rng.randn(n, self.n_components) * 0.0001
        Y_prev = Y.copy()
        gains = np.ones_like(Y)

        for iteration in range(self.n_iter):
            # Early exaggeration
            if iteration < 250:
                P_used = P * self.early_exaggeration
                mom = self.momentum
            else:
                P_used = P
                mom = self.final_momentum

            # Compute Q (student-t distribution)
            D_y = self._compute_pairwise_distances(Y)
            num = 1.0 / (1.0 + D_y)
            np.fill_diagonal(num, 0)
            Q = num / np.maximum(num.sum(), 1e-12)
            Q = np.maximum(Q, 1e-12)

            # Gradient
            PQ_diff = P_used - Q
            grad = np.zeros_like(Y)
            for i in range(n):
                diff = Y[i] - Y
                grad[i] = 4.0 * np.sum((PQ_diff[i] * num[i])[:, None] * diff, axis=0)

            # Adaptive gains
            gains = (gains + 0.2) * ((grad > 0) != (Y - Y_prev > 0)) + \
                    (gains * 0.8) * ((grad > 0) == (Y - Y_prev > 0))
            gains = np.maximum(gains, 0.01)

            Y_new = Y - self.learning_rate * gains * grad + mom * (Y - Y_prev)
            Y_prev = Y.copy()
            Y = Y_new
            Y -= Y.mean(axis=0)

        # KL divergence
        D_y = self._compute_pairwise_distances(Y)
        num = 1.0 / (1.0 + D_y)
        np.fill_diagonal(num, 0)
        Q = num / np.maximum(num.sum(), 1e-12)
        Q = np.maximum(Q, 1e-12)
        self.kl_divergence_ = np.sum(P * np.log(P / Q))
        self.embedding_ = Y
        return Y


# ============================================================
# UMAP (simplified)
# ============================================================
class UMAP:
    """Simplified UMAP implementation from scratch."""

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1,
                 n_epochs=200, learning_rate=1.0, random_state=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.embedding_ = None

    def _knn_graph(self, X):
        """Compute k-nearest neighbor graph."""
        n = X.shape[0]
        sq = np.sum(X ** 2, axis=1)
        D = sq[:, None] + sq[None, :] - 2 * np.dot(X, X.T)
        D = np.maximum(D, 0)
        np.fill_diagonal(D, np.inf)

        k = min(self.n_neighbors, n - 1)
        indices = np.zeros((n, k), dtype=int)
        distances = np.zeros((n, k))

        for i in range(n):
            idx = np.argpartition(D[i], k)[:k]
            idx = idx[np.argsort(D[i, idx])]
            indices[i] = idx
            distances[i] = np.sqrt(np.maximum(D[i, idx], 0))

        return indices, distances

    def _smooth_knn_dist(self, distances):
        """Compute smooth k-NN distances (sigma values)."""
        n = distances.shape[0]
        target = np.log2(self.n_neighbors)
        sigmas = np.ones(n)

        for i in range(n):
            lo, hi = 0.0, 100.0
            for _ in range(64):
                mid = (lo + hi) / 2
                d = distances[i]
                rho = d[0] if len(d) > 0 else 0
                vals = np.exp(-(np.maximum(d - rho, 0)) / max(mid, 1e-10))
                vals[0] = 1.0  # nearest neighbor
                s = vals.sum()
                if abs(s - target) < 1e-5:
                    break
                if s > target:
                    hi = mid
                else:
                    lo = mid
            sigmas[i] = (lo + hi) / 2

        return sigmas

    def _compute_graph(self, X):
        """Compute fuzzy simplicial set (UMAP graph)."""
        indices, distances = self._knn_graph(X)
        sigmas = self._smooth_knn_dist(distances)
        n = X.shape[0]

        # Build weighted adjacency
        rows, cols, vals = [], [], []
        for i in range(n):
            rho = distances[i, 0] if distances.shape[1] > 0 else 0
            for j_idx in range(indices.shape[1]):
                j = indices[i, j_idx]
                d = distances[i, j_idx]
                w = np.exp(-(max(d - rho, 0)) / max(sigmas[i], 1e-10))
                rows.append(i)
                cols.append(j)
                vals.append(w)

        # Symmetrize: P(a|b) + P(b|a) - P(a|b)*P(b|a)
        graph = np.zeros((n, n))
        for r, c, v in zip(rows, cols, vals):
            graph[r, c] = v

        sym = graph + graph.T - graph * graph.T
        return sym

    def _find_ab_params(self, min_dist):
        """Find a, b parameters for the low-dimensional curve."""
        # Approximate: for min_dist in [0, 1], use a fixed approximation
        # These are reasonable approximations
        if min_dist <= 0:
            return 1.929, 0.7915
        elif min_dist >= 1:
            return 1.0, 1.0
        else:
            a = 1.929 * (1 - min_dist) + 1.0 * min_dist
            b = 0.7915 * (1 - min_dist) + 1.0 * min_dist
            return a, b

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]

        graph = self._compute_graph(X)
        a, b = self._find_ab_params(self.min_dist)

        # Collect edges with weights
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if graph[i, j] > 0:
                    edges.append((i, j, graph[i, j]))

        if not edges:
            self.embedding_ = rng.randn(n, self.n_components) * 0.01
            return self.embedding_

        edges_arr = np.array(edges)
        weights = edges_arr[:, 2]
        epochs_per_sample = np.maximum(1, (weights.max() / np.maximum(weights, 1e-10)))
        epochs_per_sample = np.minimum(epochs_per_sample, self.n_epochs)

        # Initialize with PCA
        pca = PCA(n_components=self.n_components)
        Y = pca.fit_transform(X)
        Y = Y / (np.std(Y) + 1e-10) * 0.01

        for epoch in range(self.n_epochs):
            alpha = self.learning_rate * (1.0 - epoch / self.n_epochs)

            for e_idx in range(len(edges)):
                if epoch % max(1, int(epochs_per_sample[e_idx])) != 0:
                    continue

                i, j = int(edges_arr[e_idx, 0]), int(edges_arr[e_idx, 1])
                diff = Y[i] - Y[j]
                dist_sq = np.sum(diff ** 2) + 1e-10
                dist = np.sqrt(dist_sq)

                # Attractive force
                grad_coeff = -2.0 * a * b * dist ** (b - 1) / (1.0 + a * dist ** b + 1e-10)
                grad = np.clip(grad_coeff * diff, -4, 4)
                Y[i] += alpha * grad
                Y[j] -= alpha * grad

                # Repulsive (negative sampling)
                for _ in range(5):
                    k = rng.randint(n)
                    if k == i:
                        continue
                    diff_k = Y[i] - Y[k]
                    dist_sq_k = np.sum(diff_k ** 2) + 1e-10
                    dist_k = np.sqrt(dist_sq_k)
                    grad_coeff_k = 2.0 * b / ((0.001 + dist_sq_k) * (1.0 + a * dist_k ** b + 1e-10))
                    grad_k = np.clip(grad_coeff_k * diff_k, -4, 4)
                    Y[i] += alpha * grad_k

        self.embedding_ = Y
        return Y


# ============================================================
# LDA (Linear Discriminant Analysis)
# ============================================================
class LDA:
    """Linear Discriminant Analysis for supervised dimensionality reduction."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scalings_ = None
        self.mean_ = None
        self.class_means_ = None
        self.classes_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)

        self.mean_ = X.mean(axis=0)
        self.class_means_ = np.zeros((n_classes, n_features))

        # Within-class scatter
        Sw = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_means_[idx] = X_c.mean(axis=0)
            diff = X_c - self.class_means_[idx]
            Sw += np.dot(diff.T, diff)

        # Between-class scatter
        Sb = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes_):
            n_c = np.sum(y == c)
            diff = (self.class_means_[idx] - self.mean_)[:, None]
            Sb += n_c * np.dot(diff, diff.T)

        # Solve generalized eigenvalue problem
        # Sw^-1 Sb w = lambda w
        Sw_reg = Sw + np.eye(n_features) * 1e-6
        try:
            Sw_inv = np.linalg.inv(Sw_reg)
            M = np.dot(Sw_inv, Sb)
            eigenvalues, eigenvectors = np.linalg.eig(M)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(Sb)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.scalings_ = eigenvectors[:, :self.n_components]
        total = np.sum(np.abs(eigenvalues[eigenvalues > 0])) if np.any(eigenvalues > 0) else 1.0
        self.explained_variance_ratio_ = np.abs(eigenvalues[:self.n_components]) / total
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X - self.mean_, self.scalings_)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Classify by nearest class centroid in projected space."""
        X_proj = self.transform(X)
        centroids = np.dot(self.class_means_ - self.mean_, self.scalings_)
        dists = np.array([np.sum((X_proj - c) ** 2, axis=1) for c in centroids])
        return self.classes_[np.argmin(dists, axis=0)]


# ============================================================
# FactorAnalysis
# ============================================================
class FactorAnalysis:
    """Factor Analysis with EM algorithm."""

    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None  # Loading matrix W (n_components x n_features)
        self.noise_variance_ = None
        self.mean_ = None
        self.loglike_ = []

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Initialize with PCA
        pca = PCA(n_components=self.n_components)
        pca.fit(X_centered)
        W = pca.components_.T * np.sqrt(pca.explained_variance_)  # (features x components)
        psi = np.var(X_centered, axis=0)
        psi = np.maximum(psi, 1e-6)

        for iteration in range(self.max_iter):
            # E-step: E[z|x] = (W^T Psi^-1 W + I)^-1 W^T Psi^-1 x
            Psi_inv_W = W / psi[:, None]
            G = np.linalg.inv(np.eye(self.n_components) + np.dot(W.T, Psi_inv_W))
            Ez = np.dot(X_centered, np.dot(Psi_inv_W, G))  # (n x k)
            Ezz = n_samples * G + np.dot(Ez.T, Ez)  # (k x k)

            # M-step
            W_new = np.dot(X_centered.T, Ez).dot(np.linalg.inv(Ezz))
            psi_new = np.diag(np.dot(X_centered.T, X_centered) - np.dot(W_new, np.dot(Ez.T, X_centered))) / n_samples
            psi_new = np.maximum(psi_new, 1e-6)

            # Check convergence
            diff = np.max(np.abs(W_new - W))
            W = W_new
            psi = psi_new

            if diff < self.tol:
                break

        self.components_ = W.T  # (n_components x n_features)
        self.noise_variance_ = psi
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        W = self.components_.T  # (features x components)
        Psi_inv_W = W / self.noise_variance_[:, None]
        G = np.linalg.inv(np.eye(self.n_components) + np.dot(W.T, Psi_inv_W))
        return np.dot(X_centered, np.dot(Psi_inv_W, G))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# MDS (Multidimensional Scaling)
# ============================================================
class MDS:
    """Classical and metric Multidimensional Scaling."""

    def __init__(self, n_components=2, metric=True, max_iter=300, random_state=None):
        self.n_components = n_components
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state
        self.embedding_ = None
        self.stress_ = None

    def _classical_mds(self, D):
        """Double-centering approach (Torgerson MDS)."""
        n = D.shape[0]
        D_sq = D ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H.dot(D_sq).dot(H)

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        nc = self.n_components
        pos_vals = np.maximum(eigenvalues[:nc], 0)
        return eigenvectors[:, :nc] * np.sqrt(pos_vals)

    def _stress(self, D, Y):
        D_y = np.sqrt(np.maximum(
            np.sum(Y ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * np.dot(Y, Y.T),
            0
        ))
        mask = np.triu(np.ones_like(D, dtype=bool), k=1)
        return np.sqrt(np.sum((D[mask] - D_y[mask]) ** 2) / np.maximum(np.sum(D[mask] ** 2), 1e-10))

    def fit_transform(self, X=None, dissimilarity_matrix=None):
        if dissimilarity_matrix is not None:
            D = np.asarray(dissimilarity_matrix, dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)
            sq = np.sum(X ** 2, axis=1)
            D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * np.dot(X, X.T), 0))

        if not self.metric:
            # Classical MDS
            self.embedding_ = self._classical_mds(D)
            self.stress_ = self._stress(D, self.embedding_)
            return self.embedding_

        # Metric MDS with SMACOF
        rng = np.random.RandomState(self.random_state)
        n = D.shape[0]
        Y = self._classical_mds(D)  # Initialize with classical

        for _ in range(self.max_iter):
            D_y = np.sqrt(np.maximum(
                np.sum(Y ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * np.dot(Y, Y.T),
                0
            ))
            D_y_safe = np.where(D_y < 1e-10, 1e-10, D_y)
            B = -D / D_y_safe
            np.fill_diagonal(B, 0)
            np.fill_diagonal(B, -B.sum(axis=1))
            Y_new = B.dot(Y) / n

            if np.max(np.abs(Y_new - Y)) < 1e-6:
                Y = Y_new
                break
            Y = Y_new

        self.embedding_ = Y
        self.stress_ = self._stress(D, Y)
        return Y


# ============================================================
# Isomap
# ============================================================
class Isomap:
    """Isometric Mapping (geodesic distances + MDS)."""

    def __init__(self, n_components=2, n_neighbors=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_ = None

    def _shortest_paths(self, graph, n):
        """Floyd-Warshall shortest paths."""
        dist = graph.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        # Compute pairwise distances
        sq = np.sum(X ** 2, axis=1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * np.dot(X, X.T), 0))

        # Build k-NN graph
        k = min(self.n_neighbors, n - 1)
        graph = np.full((n, n), np.inf)
        np.fill_diagonal(graph, 0)

        for i in range(n):
            dists_i = D[i].copy()
            dists_i[i] = np.inf
            idx = np.argpartition(dists_i, k)[:k]
            for j in idx:
                graph[i, j] = D[i, j]
                graph[j, i] = D[j, i]

        # Shortest paths (geodesic distances)
        geo = self._shortest_paths(graph, n)

        # Replace inf with large value
        max_finite = geo[np.isfinite(geo)].max() if np.any(np.isfinite(geo) & (geo > 0)) else 1.0
        geo = np.where(np.isinf(geo), max_finite * 2, geo)

        # Apply classical MDS
        mds = MDS(n_components=self.n_components, metric=False)
        self.embedding_ = mds.fit_transform(dissimilarity_matrix=geo)
        return self.embedding_


# ============================================================
# LLE (Locally Linear Embedding)
# ============================================================
class LLE:
    """Locally Linear Embedding."""

    def __init__(self, n_components=2, n_neighbors=5, reg=1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.embedding_ = None

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = min(self.n_neighbors, n - 1)

        # Find k-nearest neighbors
        sq = np.sum(X ** 2, axis=1)
        D = sq[:, None] + sq[None, :] - 2 * np.dot(X, X.T)
        D = np.maximum(D, 0)
        np.fill_diagonal(D, np.inf)

        neighbors = np.zeros((n, k), dtype=int)
        for i in range(n):
            neighbors[i] = np.argpartition(D[i], k)[:k]

        # Compute reconstruction weights
        W = np.zeros((n, n))
        for i in range(n):
            Z = X[neighbors[i]] - X[i]
            C = np.dot(Z, Z.T)
            C += self.reg * np.eye(k) * np.trace(C)
            try:
                w = np.linalg.solve(C, np.ones(k))
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(C, np.ones(k), rcond=None)[0]
            w /= w.sum() if w.sum() != 0 else 1
            W[i, neighbors[i]] = w

        # Compute embedding
        M = np.eye(n) - W
        M = np.dot(M.T, M)

        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # Skip first eigenvalue (near zero)
        idx = np.argsort(np.abs(eigenvalues))
        self.embedding_ = eigenvectors[:, idx[1:self.n_components + 1]]
        return self.embedding_


# ============================================================
# TruncatedSVD
# ============================================================
class TruncatedSVD:
    """Randomized truncated SVD (useful for sparse-like data)."""

    def __init__(self, n_components=2, n_iter=5, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = self.n_components

        # Randomized SVD via power iteration
        Omega = rng.randn(n_features, k + 10)
        Y = np.dot(X, Omega)

        for _ in range(self.n_iter):
            Y = np.dot(X, np.dot(X.T, Y))

        Q, _ = np.linalg.qr(Y)
        B = np.dot(Q.T, X)
        U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = S[:k] ** 2 / (n_samples - 1)
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var if total_var > 0 else self.explained_variance_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# NMF (Non-negative Matrix Factorization)
# ============================================================
class NMF:
    """Non-negative Matrix Factorization with multiplicative updates."""

    def __init__(self, n_components=2, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_ = None  # H: (n_components, n_features)
        self.reconstruction_err_ = None
        self.n_iter_ = 0

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.maximum(X, 0)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        k = self.n_components

        # Initialize
        avg = np.sqrt(X.mean() / k)
        W = np.abs(rng.randn(n_samples, k)) * avg + 0.01
        H = np.abs(rng.randn(k, n_features)) * avg + 0.01

        for i in range(self.max_iter):
            # Update W
            numerator = np.dot(X, H.T)
            denominator = np.dot(W, np.dot(H, H.T)) + 1e-10
            W *= numerator / denominator

            # Update H
            numerator = np.dot(W.T, X)
            denominator = np.dot(np.dot(W.T, W), H) + 1e-10
            H *= numerator / denominator

            # Check convergence
            err = np.sqrt(np.sum((X - np.dot(W, H)) ** 2))
            if i > 0 and abs(prev_err - err) / max(prev_err, 1e-10) < self.tol:
                self.n_iter_ = i + 1
                break
            prev_err = err

        self.components_ = H
        self.reconstruction_err_ = np.sqrt(np.sum((X - np.dot(W, H)) ** 2))
        self.n_iter_ = self.n_iter_ or self.max_iter
        self._W = W
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.maximum(X, 0)
        H = self.components_
        # Solve for W using multiplicative updates
        rng = np.random.RandomState(self.random_state)
        W = np.abs(rng.randn(X.shape[0], self.n_components)) * 0.1 + 0.01
        for _ in range(100):
            numerator = np.dot(X, H.T)
            denominator = np.dot(W, np.dot(H, H.T)) + 1e-10
            W *= numerator / denominator
        return W

    def fit_transform(self, X):
        self.fit(X)
        return self._W.copy()

    def inverse_transform(self, W):
        return np.dot(W, self.components_)
