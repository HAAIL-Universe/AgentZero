"""C194: Clustering Algorithms -- built from scratch with NumPy.

Algorithms:
- KMeans (Lloyd's algorithm with k-means++ init)
- MiniBatchKMeans (streaming variant)
- DBSCAN (density-based, core/border/noise)
- OPTICS (ordering-based density, variable epsilon)
- Agglomerative (single/complete/average/ward linkage)
- SpectralClustering (graph Laplacian + k-means)
- GaussianMixture (EM algorithm, full/diag/spherical covariance)
- BisectingKMeans (divisive hierarchical)
- MeanShift (kernel density mode-seeking)
- AffinityPropagation (message passing)

All use NumPy only. No scikit-learn.
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Distance / utility functions
# ============================================================

def euclidean_distances(X, Y=None):
    """Pairwise Euclidean distances between rows of X and Y."""
    if Y is None:
        Y = X
    XX = np.sum(X ** 2, axis=1)[:, None]
    YY = np.sum(Y ** 2, axis=1)[None, :]
    D2 = XX + YY - 2.0 * X @ Y.T
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)


def rbf_kernel(X, gamma=1.0):
    """RBF (Gaussian) kernel matrix."""
    D2 = np.sum(X ** 2, axis=1)[:, None] + np.sum(X ** 2, axis=1)[None, :] - 2.0 * X @ X.T
    np.maximum(D2, 0, out=D2)
    return np.exp(-gamma * D2)


# ============================================================
# KMeans
# ============================================================

class KMeans:
    """K-Means clustering with k-means++ initialization.

    Parameters:
        n_clusters: number of clusters
        max_iter: max iterations
        tol: convergence tolerance on centroid movement
        init: 'kmeans++' or 'random'
        n_init: number of random restarts (best inertia wins)
        random_state: seed
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6,
                 init='kmeans++', n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X, rng):
        n = X.shape[0]
        if self.init == 'random':
            idx = rng.choice(n, self.n_clusters, replace=False)
            return X[idx].copy()
        # k-means++
        centroids = [X[rng.randint(n)]]
        for _ in range(1, self.n_clusters):
            D = np.min(euclidean_distances(X, np.array(centroids)) ** 2, axis=1)
            probs = D / D.sum()
            cum = np.cumsum(probs)
            r = rng.random()
            idx = np.searchsorted(cum, r)
            idx = min(idx, n - 1)
            centroids.append(X[idx])
        return np.array(centroids)

    def _single_run(self, X, rng):
        centroids = self._init_centroids(X, rng)
        for _ in range(self.max_iter):
            dists = euclidean_distances(X, centroids)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()
            centroids = new_centroids
            if shift < self.tol:
                break
        dists = euclidean_distances(X, centroids)
        labels = np.argmin(dists, axis=1)
        inertia = sum(np.sum((X[labels == k] - centroids[k]) ** 2)
                       for k in range(self.n_clusters) if (labels == k).any())
        return centroids, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(self.n_init):
            centroids, labels, inertia = self._single_run(X, rng)
            if best is None or inertia < best[2]:
                best = (centroids, labels, inertia)
        self.centroids, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = euclidean_distances(X, self.centroids)
        return np.argmin(dists, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ============================================================
# MiniBatchKMeans
# ============================================================

class MiniBatchKMeans:
    """Mini-Batch K-Means for large datasets.

    Uses small random batches to update centroids incrementally.
    """

    def __init__(self, n_clusters=3, batch_size=100, max_iter=100,
                 tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # k-means++ init
        km = KMeans(self.n_clusters, max_iter=1, n_init=1, random_state=rng.randint(1 << 30))
        km.fit(X)
        centroids = km.centroids.copy()
        counts = np.ones(self.n_clusters)

        for _ in range(self.max_iter):
            idx = rng.choice(n, min(self.batch_size, n), replace=False)
            batch = X[idx]
            dists = euclidean_distances(batch, centroids)
            labels = np.argmin(dists, axis=1)
            old_centroids = centroids.copy()
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    points = batch[mask]
                    for p in points:
                        counts[k] += 1
                        lr = 1.0 / counts[k]
                        centroids[k] = (1 - lr) * centroids[k] + lr * p
            shift = np.sqrt(np.sum((centroids - old_centroids) ** 2, axis=1)).max()
            if shift < self.tol:
                break

        self.centroids = centroids
        dists = euclidean_distances(X, centroids)
        self.labels_ = np.argmin(dists, axis=1)
        self.inertia_ = sum(np.sum((X[self.labels_ == k] - centroids[k]) ** 2)
                            for k in range(self.n_clusters) if (self.labels_ == k).any())
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.argmin(euclidean_distances(X, self.centroids), axis=1)


# ============================================================
# DBSCAN
# ============================================================

class DBSCAN:
    """DBSCAN density-based clustering.

    Parameters:
        eps: neighborhood radius
        min_samples: minimum points to form a core point
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        D = euclidean_distances(X)
        neighbors = [np.where(D[i] <= self.eps)[0] for i in range(n)]
        core_mask = np.array([len(nb) >= self.min_samples for nb in neighbors])
        self.core_sample_indices_ = np.where(core_mask)[0]

        labels = -np.ones(n, dtype=int)
        cluster_id = 0

        for i in range(n):
            if labels[i] != -1 or not core_mask[i]:
                continue
            # BFS from core point i
            queue = [i]
            labels[i] = cluster_id
            head = 0
            while head < len(queue):
                q = queue[head]
                head += 1
                if core_mask[q]:
                    for nb in neighbors[q]:
                        if labels[nb] == -1:
                            labels[nb] = cluster_id
                            queue.append(nb)
            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ============================================================
# OPTICS
# ============================================================

class OPTICS:
    """OPTICS ordering-based density clustering.

    Produces a reachability ordering that can be cut at different eps
    to extract clusters at varying densities.
    """

    def __init__(self, min_samples=5, max_eps=np.inf, xi=0.05):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.xi = xi
        self.reachability_ = None
        self.ordering_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        D = euclidean_distances(X)

        # Core distances
        core_dist = np.full(n, np.inf)
        for i in range(n):
            sorted_d = np.sort(D[i])
            if len(sorted_d) > self.min_samples:
                core_dist[i] = sorted_d[self.min_samples - 1]
            elif len(sorted_d) == self.min_samples:
                core_dist[i] = sorted_d[-1]

        processed = np.zeros(n, dtype=bool)
        reachability = np.full(n, np.inf)
        ordering = []

        # Seeds priority queue (simple list-based)
        def get_next_seed(seeds):
            best_idx = min(range(len(seeds)), key=lambda i: seeds[i][1])
            return seeds.pop(best_idx)

        # Start from point with smallest core distance among unprocessed
        while len(ordering) < n:
            # Pick unprocessed point with smallest reachability, or first unprocessed
            unprocessed = [i for i in range(n) if not processed[i]]
            if not unprocessed:
                break
            # Prefer point with smallest reachability among unprocessed
            p = min(unprocessed, key=lambda i: reachability[i])
            processed[p] = True
            ordering.append(p)

            if core_dist[p] <= self.max_eps:
                seeds = []
                for nb in range(n):
                    if processed[nb]:
                        continue
                    new_reach = max(core_dist[p], D[p, nb])
                    if new_reach <= self.max_eps:
                        if new_reach < reachability[nb]:
                            reachability[nb] = new_reach
                        seeds.append((nb, reachability[nb]))

                while seeds:
                    q, _ = get_next_seed(seeds)
                    if processed[q]:
                        continue
                    processed[q] = True
                    ordering.append(q)

                    if core_dist[q] <= self.max_eps:
                        new_seeds = []
                        for nb in range(n):
                            if processed[nb]:
                                continue
                            new_reach = max(core_dist[q], D[q, nb])
                            if new_reach <= self.max_eps:
                                if new_reach < reachability[nb]:
                                    reachability[nb] = new_reach
                                new_seeds.append((nb, reachability[nb]))
                        seeds.extend(new_seeds)

        self.ordering_ = np.array(ordering)
        self.reachability_ = reachability[self.ordering_]

        # Extract clusters using xi method
        self._extract_clusters_xi(X)
        return self

    def _extract_clusters_xi(self, X):
        """Simple cluster extraction: cut reachability at threshold."""
        n = len(self.ordering_)
        labels = -np.ones(n, dtype=int)
        reach = self.reachability_.copy()
        reach[0] = 0  # first point has no reachability

        # Find a good threshold from the reachability plot
        finite_reach = reach[np.isfinite(reach)]
        if len(finite_reach) == 0:
            self.labels_ = -np.ones(len(self.ordering_), dtype=int)
            # Map back to original indices
            result = -np.ones(len(self.ordering_), dtype=int)
            for i, idx in enumerate(self.ordering_):
                result[idx] = labels[i]
            self.labels_ = result
            return

        threshold = np.median(finite_reach) * (1 + self.xi)

        cluster_id = 0
        in_cluster = False
        for i in range(n):
            if reach[i] <= threshold:
                if not in_cluster:
                    in_cluster = True
                labels[i] = cluster_id
            else:
                if in_cluster:
                    cluster_id += 1
                    in_cluster = False
                labels[i] = -1

        # Map back to original point indices
        result = -np.ones(len(self.ordering_), dtype=int)
        for i, idx in enumerate(self.ordering_):
            result[idx] = labels[i]
        self.labels_ = result

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ============================================================
# Agglomerative Clustering
# ============================================================

class AgglomerativeClustering:
    """Agglomerative (bottom-up) hierarchical clustering.

    Linkage: 'single', 'complete', 'average', 'ward'.
    """

    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.dendrogram_ = None  # list of (i, j, distance, size) merges

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Each cluster starts as a single point
        clusters = {i: [i] for i in range(n)}
        centroids = {i: X[i].copy() for i in range(n)}
        sizes = {i: 1 for i in range(n)}

        # Distance matrix (upper triangle)
        D = euclidean_distances(X)
        np.fill_diagonal(D, np.inf)

        # Build full distance dict for efficient updates
        dist = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist[(i, j)] = D[i, j]

        active = set(range(n))
        merges = []
        next_id = n

        while len(active) > self.n_clusters:
            # Find closest pair
            best_pair = None
            best_dist = np.inf
            for (i, j), d in dist.items():
                if i in active and j in active and d < best_dist:
                    best_dist = d
                    best_pair = (i, j)

            if best_pair is None:
                break

            ci, cj = best_pair
            new_size = sizes[ci] + sizes[cj]
            merges.append((ci, cj, best_dist, new_size))

            # Create merged cluster
            new_members = clusters[ci] + clusters[cj]
            new_centroid = (centroids[ci] * sizes[ci] + centroids[cj] * sizes[cj]) / new_size

            # Compute distances from new cluster to all remaining
            active.discard(ci)
            active.discard(cj)

            for k in list(active):
                dk_ci = self._get_dist(dist, ci, k)
                dk_cj = self._get_dist(dist, cj, k)

                if self.linkage == 'single':
                    new_d = min(dk_ci, dk_cj)
                elif self.linkage == 'complete':
                    new_d = max(dk_ci, dk_cj)
                elif self.linkage == 'average':
                    new_d = (dk_ci * sizes[ci] + dk_cj * sizes[cj]) / new_size
                elif self.linkage == 'ward':
                    # Lance-Williams formula for Ward
                    sk = sizes[k]
                    total = sizes[ci] + sizes[cj] + sk
                    new_d = np.sqrt(
                        ((sizes[ci] + sk) * dk_ci ** 2 +
                         (sizes[cj] + sk) * dk_cj ** 2 -
                         sk * best_dist ** 2) / total
                    )
                else:
                    raise ValueError(f"Unknown linkage: {self.linkage}")

                key = (min(next_id, k), max(next_id, k))
                dist[key] = new_d

            # Clean up old distances
            keys_to_remove = [key for key in dist if ci in key or cj in key]
            for key in keys_to_remove:
                if key not in dist:
                    continue
                # Only remove if both endpoints aren't the new cluster
                if next_id not in key:
                    del dist[key]

            clusters[next_id] = new_members
            centroids[next_id] = new_centroid
            sizes[next_id] = new_size
            active.add(next_id)
            next_id += 1

        self.dendrogram_ = merges

        # Assign labels
        labels = np.zeros(n, dtype=int)
        for label, cid in enumerate(sorted(active)):
            for member in clusters[cid]:
                labels[member] = label
        self.labels_ = labels
        return self

    @staticmethod
    def _get_dist(dist, i, j):
        key = (min(i, j), max(i, j))
        return dist.get(key, np.inf)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ============================================================
# Spectral Clustering
# ============================================================

class SpectralClustering:
    """Spectral clustering using graph Laplacian eigenvectors.

    Parameters:
        n_clusters: number of clusters
        gamma: RBF kernel parameter
        n_neighbors: if set, use k-nearest-neighbor graph instead of full RBF
        random_state: seed for k-means
    """

    def __init__(self, n_clusters=3, gamma=1.0, n_neighbors=None, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Build affinity matrix
        if self.n_neighbors is not None:
            D = euclidean_distances(X)
            W = np.zeros((n, n))
            for i in range(n):
                nn = np.argsort(D[i])[1:self.n_neighbors + 1]
                for j in nn:
                    W[i, j] = np.exp(-self.gamma * D[i, j] ** 2)
                    W[j, i] = W[i, j]  # symmetrize
        else:
            W = rbf_kernel(X, self.gamma)
            np.fill_diagonal(W, 0)

        # Normalized Laplacian: D^{-1/2} W D^{-1/2}
        degree = W.sum(axis=1)
        degree = np.maximum(degree, 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
        L_norm = D_inv_sqrt @ W @ D_inv_sqrt

        # Eigendecomposition -- take top k eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        # eigh returns ascending order; we want the largest
        idx = np.argsort(-eigenvalues)[:self.n_clusters]
        V = eigenvectors[:, idx]

        # Normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        V = V / norms

        # K-means on the embedding
        km = KMeans(self.n_clusters, random_state=self.random_state, n_init=10)
        km.fit(V)
        self.labels_ = km.labels_
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ============================================================
# Gaussian Mixture Model (EM)
# ============================================================

class GaussianMixture:
    """Gaussian Mixture Model with EM algorithm.

    Parameters:
        n_components: number of mixture components
        covariance_type: 'full', 'diag', 'spherical'
        max_iter: max EM iterations
        tol: convergence tolerance on log-likelihood
        random_state: seed
    """

    def __init__(self, n_components=3, covariance_type='full',
                 max_iter=100, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)
        k = self.n_components

        # Initialize with k-means
        km = KMeans(k, random_state=rng.randint(1 << 30), n_init=5)
        km.fit(X)
        means = km.centroids.copy()
        weights = np.ones(k) / k

        # Initialize covariances
        covs = []
        for j in range(k):
            mask = km.labels_ == j
            if mask.sum() > 1:
                c = np.cov(X[mask].T) + 1e-6 * np.eye(d)
            else:
                c = np.eye(d)
            covs.append(self._regularize_cov(c, d))

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            resp = self._e_step(X, weights, means, covs)

            # M-step
            Nk = resp.sum(axis=0)
            Nk = np.maximum(Nk, 1e-10)

            weights = Nk / n
            means = (resp.T @ X) / Nk[:, None]

            covs = []
            for j in range(k):
                diff = X - means[j]
                if self.covariance_type == 'full':
                    c = (resp[:, j:j+1] * diff).T @ diff / Nk[j]
                elif self.covariance_type == 'diag':
                    c = np.diag(np.sum(resp[:, j:j+1] * diff ** 2, axis=0) / Nk[j])
                elif self.covariance_type == 'spherical':
                    var = np.sum(resp[:, j] * np.sum(diff ** 2, axis=1)) / (Nk[j] * d)
                    c = var * np.eye(d)
                else:
                    raise ValueError(f"Unknown covariance type: {self.covariance_type}")
                covs.append(self._regularize_cov(c, d))

            # Log-likelihood
            ll = self._log_likelihood(X, weights, means, covs)
            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                break
            prev_ll = ll

        if not self.converged_:
            self.n_iter_ = self.max_iter

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covs
        return self

    def _regularize_cov(self, c, d):
        return c + 1e-6 * np.eye(d)

    def _log_gaussian(self, X, mean, cov):
        """Log probability density of multivariate Gaussian."""
        d = X.shape[1]
        diff = X - mean
        try:
            L = np.linalg.cholesky(cov)
            solve = np.linalg.solve(L, diff.T).T
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            maha = np.sum(solve ** 2, axis=1)
        except np.linalg.LinAlgError:
            # Fallback for non-PD
            inv_cov = np.linalg.pinv(cov)
            log_det = np.log(max(np.linalg.det(cov), 1e-300))
            maha = np.sum(diff @ inv_cov * diff, axis=1)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)

    def _e_step(self, X, weights, means, covs):
        n = X.shape[0]
        k = len(weights)
        log_resp = np.zeros((n, k))
        for j in range(k):
            log_resp[:, j] = np.log(weights[j] + 1e-300) + self._log_gaussian(X, means[j], covs[j])
        # Log-sum-exp for numerical stability
        log_max = log_resp.max(axis=1, keepdims=True)
        log_sum = log_max + np.log(np.sum(np.exp(log_resp - log_max), axis=1, keepdims=True))
        log_resp -= log_sum
        return np.exp(log_resp)

    def _log_likelihood(self, X, weights, means, covs):
        n = X.shape[0]
        k = len(weights)
        log_probs = np.zeros((n, k))
        for j in range(k):
            log_probs[:, j] = np.log(weights[j] + 1e-300) + self._log_gaussian(X, means[j], covs[j])
        log_max = log_probs.max(axis=1)
        return np.sum(log_max + np.log(np.sum(np.exp(log_probs - log_max[:, None]), axis=1)))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        resp = self._e_step(X, self.weights_, self.means_, self.covariances_)
        return np.argmax(resp, axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._e_step(X, self.weights_, self.means_, self.covariances_)

    def score(self, X):
        """Average log-likelihood per sample."""
        X = np.asarray(X, dtype=float)
        return self._log_likelihood(X, self.weights_, self.means_, self.covariances_) / X.shape[0]

    def bic(self, X):
        """Bayesian Information Criterion."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_components
        if self.covariance_type == 'full':
            n_params = k * (d + d * (d + 1) / 2) + k - 1
        elif self.covariance_type == 'diag':
            n_params = k * (d + d) + k - 1
        else:
            n_params = k * (d + 1) + k - 1
        ll = self._log_likelihood(X, self.weights_, self.means_, self.covariances_)
        return -2 * ll + n_params * np.log(n)


# ============================================================
# Bisecting KMeans
# ============================================================

class BisectingKMeans:
    """Bisecting K-Means: divisive hierarchical clustering.

    Repeatedly splits the cluster with highest SSE.
    """

    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.centroids = None
        self.inertia_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        # Start with everything in one cluster
        clusters = {0: np.arange(n)}
        centroids_map = {0: X.mean(axis=0)}

        while len(clusters) < self.n_clusters:
            # Find cluster with highest SSE
            best_cid = None
            best_sse = -1
            for cid, members in clusters.items():
                if len(members) < 2:
                    continue
                sse = np.sum((X[members] - centroids_map[cid]) ** 2)
                if sse > best_sse:
                    best_sse = sse
                    best_cid = cid

            if best_cid is None:
                break

            # Bisect the chosen cluster
            members = clusters[best_cid]
            sub_X = X[members]
            km = KMeans(2, max_iter=self.max_iter, random_state=rng.randint(1 << 30), n_init=5)
            km.fit(sub_X)

            new_id = max(clusters.keys()) + 1
            mask0 = km.labels_ == 0
            mask1 = km.labels_ == 1

            clusters[best_cid] = members[mask0]
            centroids_map[best_cid] = km.centroids[0]
            clusters[new_id] = members[mask1]
            centroids_map[new_id] = km.centroids[1]

        # Assign final labels
        labels = np.zeros(n, dtype=int)
        centroid_list = []
        for label, (cid, members) in enumerate(sorted(clusters.items())):
            labels[members] = label
            centroid_list.append(centroids_map[cid])

        self.labels_ = labels
        self.centroids = np.array(centroid_list)
        dists = euclidean_distances(X, self.centroids)
        self.inertia_ = sum(np.sum((X[labels == k] - self.centroids[k]) ** 2)
                            for k in range(len(centroid_list)) if (labels == k).any())
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.argmin(euclidean_distances(X, self.centroids), axis=1)


# ============================================================
# Mean Shift
# ============================================================

class MeanShift:
    """Mean Shift clustering -- kernel density mode-seeking.

    Parameters:
        bandwidth: kernel bandwidth (if None, estimated)
        max_iter: max iterations per seed
        tol: convergence tolerance
    """

    def __init__(self, bandwidth=None, max_iter=300, tol=1e-4):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None

    def _estimate_bandwidth(self, X):
        """Estimate bandwidth using median of pairwise distances."""
        D = euclidean_distances(X)
        n = D.shape[0]
        # Use median of nearest-neighbor distances
        nn_dists = []
        for i in range(n):
            sorted_d = np.sort(D[i])[1:]  # exclude self
            if len(sorted_d) > 0:
                nn_dists.append(sorted_d[min(len(sorted_d) - 1, max(1, n // 4))])
        return np.median(nn_dists) if nn_dists else 1.0

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if self.bandwidth is None:
            bw = self._estimate_bandwidth(X)
        else:
            bw = self.bandwidth

        # Use all points as seeds
        centers = []
        for seed in X:
            center = seed.copy()
            for _ in range(self.max_iter):
                dists = np.sqrt(np.sum((X - center) ** 2, axis=1))
                within = dists <= bw
                if not within.any():
                    break
                new_center = X[within].mean(axis=0)
                shift = np.sqrt(np.sum((new_center - center) ** 2))
                center = new_center
                if shift < self.tol:
                    break
            centers.append(center)

        # Merge nearby centers
        centers = np.array(centers)
        merged = []
        used = np.zeros(len(centers), dtype=bool)
        for i in range(len(centers)):
            if used[i]:
                continue
            group = [centers[i]]
            used[i] = True
            for j in range(i + 1, len(centers)):
                if used[j]:
                    continue
                if np.sqrt(np.sum((centers[i] - centers[j]) ** 2)) < bw:
                    group.append(centers[j])
                    used[j] = True
            merged.append(np.mean(group, axis=0))

        self.cluster_centers_ = np.array(merged)

        # Assign labels
        dists = euclidean_distances(X, self.cluster_centers_)
        self.labels_ = np.argmin(dists, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.argmin(euclidean_distances(X, self.cluster_centers_), axis=1)


# ============================================================
# Affinity Propagation
# ============================================================

class AffinityPropagation:
    """Affinity Propagation clustering -- message passing.

    Parameters:
        damping: damping factor (0.5 to 1.0)
        max_iter: max iterations
        preference: preference for each point being an exemplar (if None, uses median similarity)
        convergence_iter: number of iterations with no change to declare convergence
    """

    def __init__(self, damping=0.5, max_iter=200, preference=None, convergence_iter=15):
        self.damping = damping
        self.max_iter = max_iter
        self.preference = preference
        self.convergence_iter = convergence_iter
        self.cluster_centers_indices_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Similarity matrix (negative squared Euclidean distance)
        S = -euclidean_distances(X) ** 2

        if self.preference is None:
            pref = np.median(S[np.triu_indices(n, k=1)])
        else:
            pref = self.preference
        np.fill_diagonal(S, pref)

        # Initialize messages
        A = np.zeros((n, n))  # availability
        R = np.zeros((n, n))  # responsibility

        prev_exemplars = None
        stable_count = 0

        for _ in range(self.max_iter):
            # Update responsibilities (vectorized)
            AS = A + S
            # For each row, find the two largest values
            sorted_AS = np.partition(AS, -2, axis=1)
            max1 = sorted_AS[:, -1]  # largest
            max2_val = sorted_AS[:, -2]  # second largest
            idx1 = np.argmax(AS, axis=1)

            R_new = S.copy()
            # Subtract max1 from all columns
            R_new -= max1[:, None]
            # For the column that had the max, subtract max2 instead
            for i in range(n):
                R_new[i, idx1[i]] = S[i, idx1[i]] - max2_val[i]

            R = self.damping * R + (1 - self.damping) * R_new

            # Update availabilities (vectorized)
            Rp = np.maximum(R, 0)
            np.fill_diagonal(Rp, np.diag(R))
            col_sums = Rp.sum(axis=0)  # sum of each column

            # a(i,k) for i != k: min(0, r(k,k) + sum_{j not i,k} max(0,r(j,k)))
            # = min(0, col_sums[k] - Rp[i,k])
            # But we need r(k,k) not max(0,r(k,k)) in the self-responsibility term
            # Actually: a(i,k) = min(0, r(k,k) + sum_{j!=i,k} max(0, r(j,k)))
            # = min(0, col_sums[k] - Rp[i,k])  since col_sums includes Rp[k,k] = r(k,k)
            A_new = col_sums[None, :] - Rp  # subtract self from column sum
            np.minimum(A_new, 0, out=A_new)

            # Self-availability: a(k,k) = sum_{j!=k} max(0, r(j,k))
            # = col_sums[k] - Rp[k,k] = col_sums[k] - r(k,k)
            # But col_sums[k] already has Rp[k,k] = r(k,k), so:
            diag_vals = col_sums - np.diag(Rp)
            np.fill_diagonal(A_new, diag_vals)

            A = self.damping * A + (1 - self.damping) * A_new

            # Check convergence
            exemplars = set(np.where(np.diag(A + R) > 0)[0])
            if exemplars == prev_exemplars:
                stable_count += 1
                if stable_count >= self.convergence_iter:
                    break
            else:
                stable_count = 0
            prev_exemplars = exemplars

        # Extract exemplars
        exemplar_mask = np.diag(A + R) > 0
        exemplar_indices = np.where(exemplar_mask)[0]

        if len(exemplar_indices) == 0:
            # No exemplars found -- put everything in one cluster
            self.cluster_centers_indices_ = np.array([0])
            self.labels_ = np.zeros(n, dtype=int)
            return self

        self.cluster_centers_indices_ = exemplar_indices

        # Assign labels based on nearest exemplar
        S_exemplars = S[:, exemplar_indices]
        self.labels_ = np.argmax(S_exemplars, axis=1)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


# ============================================================
# Cluster evaluation metrics
# ============================================================

def silhouette_score(X, labels):
    """Mean silhouette coefficient over all samples."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 0.0

    D = euclidean_distances(X)
    scores = np.zeros(n)

    for i in range(n):
        # a(i) = mean distance to same-cluster points
        same = labels == labels[i]
        same[i] = False
        if same.sum() == 0:
            scores[i] = 0
            continue
        a_i = D[i, same].mean()

        # b(i) = min mean distance to any other cluster
        b_i = np.inf
        for label in unique_labels:
            if label == labels[i]:
                continue
            other = labels == label
            if other.sum() > 0:
                b_i = min(b_i, D[i, other].mean())

        if b_i == np.inf:
            scores[i] = 0
        else:
            scores[i] = (b_i - a_i) / max(a_i, b_i)

    return scores.mean()


def calinski_harabasz_score(X, labels):
    """Calinski-Harabasz index (variance ratio criterion)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2 or k >= n:
        return 0.0

    overall_mean = X.mean(axis=0)

    # Between-group dispersion
    bgd = 0.0
    # Within-group dispersion
    wgd = 0.0

    for label in unique_labels:
        mask = labels == label
        nk = mask.sum()
        cluster_mean = X[mask].mean(axis=0)
        bgd += nk * np.sum((cluster_mean - overall_mean) ** 2)
        wgd += np.sum((X[mask] - cluster_mean) ** 2)

    if wgd == 0:
        return float('inf')

    return (bgd / (k - 1)) / (wgd / (n - k))


def davies_bouldin_score(X, labels):
    """Davies-Bouldin index (lower is better)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        return 0.0

    centroids = []
    dispersions = []
    for label in unique_labels:
        mask = labels == label
        centroid = X[mask].mean(axis=0)
        centroids.append(centroid)
        disp = np.mean(np.sqrt(np.sum((X[mask] - centroid) ** 2, axis=1)))
        dispersions.append(disp)

    centroids = np.array(centroids)
    db = 0.0

    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i == j:
                continue
            d_ij = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
            if d_ij == 0:
                ratio = float('inf')
            else:
                ratio = (dispersions[i] + dispersions[j]) / d_ij
            max_ratio = max(max_ratio, ratio)
        db += max_ratio

    return db / k


def adjusted_rand_index(labels_true, labels_pred):
    """Adjusted Rand Index between two clusterings."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)

    # Build contingency table
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    contingency = np.zeros((len(classes), len(clusters)), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}
    cluster_map = {c: i for i, c in enumerate(clusters)}

    for i in range(n):
        contingency[class_map[labels_true[i]], cluster_map[labels_pred[i]]] += 1

    # Sum of C(n_ij, 2)
    sum_comb = sum(int(v) * (int(v) - 1) // 2 for v in contingency.flatten())

    # Row sums and column sums
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)

    sum_row = sum(int(s) * (int(s) - 1) // 2 for s in row_sums)
    sum_col = sum(int(s) * (int(s) - 1) // 2 for s in col_sums)

    n_comb = n * (n - 1) // 2
    expected = sum_row * sum_col / n_comb if n_comb > 0 else 0
    max_index = (sum_row + sum_col) / 2.0
    denom = max_index - expected

    if denom == 0:
        return 1.0 if sum_comb == expected else 0.0

    return (sum_comb - expected) / denom
