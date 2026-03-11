"""Tests for C194: Clustering Algorithms."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from clustering import (
    KMeans, MiniBatchKMeans, DBSCAN, OPTICS,
    AgglomerativeClustering, SpectralClustering, GaussianMixture,
    BisectingKMeans, MeanShift, AffinityPropagation,
    euclidean_distances, rbf_kernel,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_index,
)


# ============================================================
# Test data generators
# ============================================================

def make_blobs(n_per_cluster=50, centers=None, random_state=42):
    """Generate well-separated Gaussian blobs."""
    rng = np.random.RandomState(random_state)
    if centers is None:
        centers = np.array([[0, 0], [5, 5], [10, 0]])
    points = []
    labels = []
    for i, c in enumerate(centers):
        pts = rng.randn(n_per_cluster, len(c)) * 0.5 + c
        points.append(pts)
        labels.extend([i] * n_per_cluster)
    return np.vstack(points), np.array(labels)


def make_moons(n=200, noise=0.05, random_state=42):
    """Two interleaving half circles."""
    rng = np.random.RandomState(random_state)
    n_half = n // 2
    t1 = np.linspace(0, np.pi, n_half)
    t2 = np.linspace(0, np.pi, n - n_half)
    x1 = np.column_stack([np.cos(t1), np.sin(t1)])
    x2 = np.column_stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5])
    X = np.vstack([x1, x2]) + rng.randn(n, 2) * noise
    y = np.array([0] * n_half + [1] * (n - n_half))
    return X, y


def make_circles(n=200, noise=0.05, factor=0.5, random_state=42):
    """Concentric circles."""
    rng = np.random.RandomState(random_state)
    n_half = n // 2
    t1 = np.linspace(0, 2 * np.pi, n_half, endpoint=False)
    t2 = np.linspace(0, 2 * np.pi, n - n_half, endpoint=False)
    outer = np.column_stack([np.cos(t1), np.sin(t1)])
    inner = factor * np.column_stack([np.cos(t2), np.sin(t2)])
    X = np.vstack([outer, inner]) + rng.randn(n, 2) * noise
    y = np.array([0] * n_half + [1] * (n - n_half))
    return X, y


# ============================================================
# Utility functions
# ============================================================

class TestDistances:
    def test_euclidean_self(self):
        X = np.array([[0, 0], [1, 0], [0, 1]])
        D = euclidean_distances(X)
        assert D.shape == (3, 3)
        np.testing.assert_allclose(D[0, 0], 0)
        np.testing.assert_allclose(D[0, 1], 1.0)
        np.testing.assert_allclose(D[0, 2], 1.0)
        np.testing.assert_allclose(D[1, 2], np.sqrt(2))

    def test_euclidean_xy(self):
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([[2, 2], [3, 3]])
        D = euclidean_distances(X, Y)
        assert D.shape == (2, 2)
        np.testing.assert_allclose(D[0, 0], np.sqrt(8))

    def test_rbf_kernel(self):
        X = np.array([[0, 0], [1, 0]])
        K = rbf_kernel(X, gamma=1.0)
        assert K.shape == (2, 2)
        np.testing.assert_allclose(K[0, 0], 1.0)
        assert 0 < K[0, 1] < 1

    def test_euclidean_single_point(self):
        X = np.array([[3.0, 4.0]])
        D = euclidean_distances(X)
        np.testing.assert_allclose(D, [[0.0]])

    def test_euclidean_symmetry(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 3)
        D = euclidean_distances(X)
        np.testing.assert_allclose(D, D.T)


# ============================================================
# KMeans
# ============================================================

class TestKMeans:
    def test_basic_clustering(self):
        X, y = make_blobs(random_state=0)
        km = KMeans(n_clusters=3, random_state=0)
        km.fit(X)
        assert len(np.unique(km.labels_)) == 3
        assert km.centroids.shape == (3, 2)
        assert km.inertia_ >= 0

    def test_perfect_separation(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100], [200, 0]]), random_state=0)
        km = KMeans(n_clusters=3, random_state=0)
        km.fit(X)
        ari = adjusted_rand_index(y, km.labels_)
        assert ari > 0.9

    def test_predict(self):
        X, _ = make_blobs(random_state=0)
        km = KMeans(n_clusters=3, random_state=0).fit(X)
        preds = km.predict(X)
        np.testing.assert_array_equal(preds, km.labels_)

    def test_fit_predict(self):
        X, _ = make_blobs(random_state=0)
        km = KMeans(n_clusters=3, random_state=0)
        labels = km.fit_predict(X)
        np.testing.assert_array_equal(labels, km.labels_)

    def test_random_init(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        km = KMeans(n_clusters=2, init='random', random_state=0)
        km.fit(X)
        ari = adjusted_rand_index(y, km.labels_)
        assert ari > 0.9

    def test_one_cluster(self):
        X, _ = make_blobs(random_state=0)
        km = KMeans(n_clusters=1, random_state=0).fit(X)
        assert np.all(km.labels_ == 0)

    def test_n_init(self):
        X, _ = make_blobs(random_state=0)
        km1 = KMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
        km10 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        assert km10.inertia_ <= km1.inertia_ + 1e-6

    def test_convergence_tolerance(self):
        X, _ = make_blobs(random_state=0)
        km = KMeans(n_clusters=3, tol=1e-10, random_state=0).fit(X)
        assert km.inertia_ >= 0

    def test_high_dim(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(30, 10) + i * 10 for i in range(3)])
        km = KMeans(n_clusters=3, random_state=0).fit(X)
        assert len(np.unique(km.labels_)) == 3

    def test_two_points_two_clusters(self):
        X = np.array([[0, 0], [10, 10]])
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        assert km.labels_[0] != km.labels_[1]


# ============================================================
# MiniBatchKMeans
# ============================================================

class TestMiniBatchKMeans:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100], [200, 0]]), random_state=0)
        mb = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=30)
        mb.fit(X)
        assert len(np.unique(mb.labels_)) == 3
        ari = adjusted_rand_index(y, mb.labels_)
        assert ari > 0.8

    def test_predict(self):
        X, _ = make_blobs(random_state=0)
        mb = MiniBatchKMeans(n_clusters=3, random_state=0).fit(X)
        preds = mb.predict(X[:5])
        assert len(preds) == 5

    def test_inertia(self):
        X, _ = make_blobs(random_state=0)
        mb = MiniBatchKMeans(n_clusters=3, random_state=0).fit(X)
        assert mb.inertia_ >= 0


# ============================================================
# DBSCAN
# ============================================================

class TestDBSCAN:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [5, 5]]), random_state=0)
        db = DBSCAN(eps=1.5, min_samples=3).fit(X)
        unique = np.unique(db.labels_)
        n_clusters = len(unique[unique >= 0])
        assert n_clusters == 2

    def test_noise_points(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [5, 5]]), random_state=0)
        # Add isolated noise points
        noise = np.array([[50, 50], [60, 60]])
        X = np.vstack([X, noise])
        db = DBSCAN(eps=1.5, min_samples=3).fit(X)
        # Noise should be labeled -1
        assert db.labels_[-1] == -1
        assert db.labels_[-2] == -1

    def test_core_samples(self):
        X, _ = make_blobs(centers=np.array([[0, 0]]), n_per_cluster=20, random_state=0)
        db = DBSCAN(eps=2.0, min_samples=3).fit(X)
        assert len(db.core_sample_indices_) > 0

    def test_all_noise(self):
        X = np.array([[0, 0], [100, 100], [200, 200]])
        db = DBSCAN(eps=0.1, min_samples=3).fit(X)
        assert np.all(db.labels_ == -1)

    def test_single_cluster(self):
        rng = np.random.RandomState(0)
        X = rng.randn(30, 2) * 0.3
        db = DBSCAN(eps=1.0, min_samples=3).fit(X)
        assert len(np.unique(db.labels_[db.labels_ >= 0])) == 1

    def test_fit_predict(self):
        X, _ = make_blobs(random_state=0)
        db = DBSCAN(eps=1.5, min_samples=3)
        labels = db.fit_predict(X)
        np.testing.assert_array_equal(labels, db.labels_)

    def test_well_separated(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 0]]), random_state=0)
        db = DBSCAN(eps=2.0, min_samples=3).fit(X)
        n_clusters = len(np.unique(db.labels_[db.labels_ >= 0]))
        assert n_clusters == 2
        ari = adjusted_rand_index(y, db.labels_)
        assert ari > 0.9


# ============================================================
# OPTICS
# ============================================================

class TestOPTICS:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [10, 10]]), random_state=0)
        op = OPTICS(min_samples=5)
        op.fit(X)
        assert op.reachability_ is not None
        assert op.ordering_ is not None
        assert len(op.ordering_) == len(X)

    def test_reachability_order(self):
        X, _ = make_blobs(random_state=0)
        op = OPTICS(min_samples=5).fit(X)
        # Ordering should be a permutation
        assert set(op.ordering_) == set(range(len(X)))

    def test_labels_assigned(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [20, 20]]), random_state=0)
        op = OPTICS(min_samples=3).fit(X)
        assert len(op.labels_) == len(X)

    def test_fit_predict(self):
        X, _ = make_blobs(random_state=0)
        op = OPTICS(min_samples=5)
        labels = op.fit_predict(X)
        np.testing.assert_array_equal(labels, op.labels_)


# ============================================================
# Agglomerative Clustering
# ============================================================

class TestAgglomerative:
    def test_single_linkage(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        ac = AgglomerativeClustering(n_clusters=2, linkage='single').fit(X)
        ari = adjusted_rand_index(y, ac.labels_)
        assert ari > 0.9

    def test_complete_linkage(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        ac = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(X)
        ari = adjusted_rand_index(y, ac.labels_)
        assert ari > 0.9

    def test_average_linkage(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        ac = AgglomerativeClustering(n_clusters=2, linkage='average').fit(X)
        ari = adjusted_rand_index(y, ac.labels_)
        assert ari > 0.9

    def test_ward_linkage(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        ac = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
        ari = adjusted_rand_index(y, ac.labels_)
        assert ari > 0.9

    def test_three_clusters(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100], [200, 0]]), random_state=0)
        ac = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
        assert len(np.unique(ac.labels_)) == 3

    def test_dendrogram(self):
        X, _ = make_blobs(n_per_cluster=10, random_state=0)
        ac = AgglomerativeClustering(n_clusters=2).fit(X)
        assert len(ac.dendrogram_) > 0
        # Each merge is (i, j, dist, size)
        for merge in ac.dendrogram_:
            assert len(merge) == 4
            assert merge[2] >= 0  # distance non-negative
            assert merge[3] >= 2  # merged cluster has at least 2 points

    def test_fit_predict(self):
        X, _ = make_blobs(random_state=0)
        ac = AgglomerativeClustering(n_clusters=3)
        labels = ac.fit_predict(X)
        np.testing.assert_array_equal(labels, ac.labels_)

    def test_single_point_clusters(self):
        X = np.array([[0, 0], [1, 1], [100, 100]])
        ac = AgglomerativeClustering(n_clusters=2).fit(X)
        # [0,0] and [1,1] should be in same cluster
        assert ac.labels_[0] == ac.labels_[1]
        assert ac.labels_[0] != ac.labels_[2]


# ============================================================
# Spectral Clustering
# ============================================================

class TestSpectralClustering:
    def test_blobs(self):
        X, y = make_blobs(centers=np.array([[0, 0], [10, 10]]), random_state=0)
        sc = SpectralClustering(n_clusters=2, gamma=0.1, random_state=0).fit(X)
        ari = adjusted_rand_index(y, sc.labels_)
        assert ari > 0.8

    def test_moons(self):
        X, y = make_moons(n=100, noise=0.05, random_state=0)
        sc = SpectralClustering(n_clusters=2, gamma=10.0, n_neighbors=10, random_state=0).fit(X)
        ari = adjusted_rand_index(y, sc.labels_)
        assert ari > 0.3  # spectral with kNN should handle non-convex

    def test_knn_affinity(self):
        X, y = make_blobs(centers=np.array([[0, 0], [10, 10]]), random_state=0)
        sc = SpectralClustering(n_clusters=2, gamma=0.5, n_neighbors=10, random_state=0).fit(X)
        ari = adjusted_rand_index(y, sc.labels_)
        assert ari > 0.5

    def test_fit_predict(self):
        X, _ = make_blobs(random_state=0)
        sc = SpectralClustering(n_clusters=3, random_state=0)
        labels = sc.fit_predict(X)
        np.testing.assert_array_equal(labels, sc.labels_)

    def test_three_clusters(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [10, 10], [20, 0]]), random_state=0)
        sc = SpectralClustering(n_clusters=3, gamma=0.05, random_state=0).fit(X)
        assert len(np.unique(sc.labels_)) == 3


# ============================================================
# Gaussian Mixture
# ============================================================

class TestGaussianMixture:
    def test_basic_fit(self):
        X, y = make_blobs(centers=np.array([[0, 0], [10, 10]]), random_state=0)
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        assert gm.weights_ is not None
        assert len(gm.weights_) == 2
        np.testing.assert_allclose(gm.weights_.sum(), 1.0, atol=1e-6)

    def test_predict(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        preds = gm.predict(X)
        ari = adjusted_rand_index(y, preds)
        assert ari > 0.9

    def test_predict_proba(self):
        X, _ = make_blobs(random_state=0)
        gm = GaussianMixture(n_components=3, random_state=0).fit(X)
        proba = gm.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_score(self):
        X, _ = make_blobs(random_state=0)
        gm = GaussianMixture(n_components=3, random_state=0).fit(X)
        s = gm.score(X)
        assert np.isfinite(s)

    def test_bic(self):
        X, _ = make_blobs(random_state=0)
        gm = GaussianMixture(n_components=3, random_state=0).fit(X)
        b = gm.bic(X)
        assert np.isfinite(b)

    def test_diag_covariance(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        gm = GaussianMixture(n_components=2, covariance_type='diag', random_state=0).fit(X)
        preds = gm.predict(X)
        ari = adjusted_rand_index(y, preds)
        assert ari > 0.9

    def test_spherical_covariance(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        gm = GaussianMixture(n_components=2, covariance_type='spherical', random_state=0).fit(X)
        preds = gm.predict(X)
        ari = adjusted_rand_index(y, preds)
        assert ari > 0.9

    def test_convergence(self):
        X, _ = make_blobs(random_state=0)
        gm = GaussianMixture(n_components=3, max_iter=500, random_state=0).fit(X)
        assert gm.converged_ or gm.n_iter_ > 0

    def test_means_near_true(self):
        centers = np.array([[0, 0], [10, 10]])
        X, _ = make_blobs(centers=centers, random_state=0)
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        # Check means are close to true centers (order may differ)
        dists = euclidean_distances(gm.means_, centers)
        assert min(dists[0]) < 2.0
        assert min(dists[1]) < 2.0

    def test_bic_model_selection(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        bics = []
        for k in [1, 2, 3, 4]:
            gm = GaussianMixture(n_components=k, random_state=0).fit(X)
            bics.append(gm.bic(X))
        # BIC should favor k=2
        assert np.argmin(bics) == 1  # index 1 -> k=2


# ============================================================
# BisectingKMeans
# ============================================================

class TestBisectingKMeans:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100], [200, 0]]), random_state=0)
        bk = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
        assert len(np.unique(bk.labels_)) == 3
        ari = adjusted_rand_index(y, bk.labels_)
        assert ari > 0.8

    def test_predict(self):
        X, _ = make_blobs(random_state=0)
        bk = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
        preds = bk.predict(X[:5])
        assert len(preds) == 5

    def test_inertia(self):
        X, _ = make_blobs(random_state=0)
        bk = BisectingKMeans(n_clusters=3, random_state=0).fit(X)
        assert bk.inertia_ >= 0

    def test_two_clusters(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 0]]), random_state=0)
        bk = BisectingKMeans(n_clusters=2, random_state=0).fit(X)
        ari = adjusted_rand_index(y, bk.labels_)
        assert ari > 0.9


# ============================================================
# MeanShift
# ============================================================

class TestMeanShift:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [10, 10]]), random_state=0)
        ms = MeanShift(bandwidth=2.0).fit(X)
        assert ms.cluster_centers_ is not None
        assert len(ms.cluster_centers_) >= 1

    def test_well_separated(self):
        X, y = make_blobs(centers=np.array([[0, 0], [20, 20]]), random_state=0)
        ms = MeanShift(bandwidth=3.0).fit(X)
        assert len(ms.cluster_centers_) == 2

    def test_auto_bandwidth(self):
        X, _ = make_blobs(random_state=0)
        ms = MeanShift().fit(X)
        assert ms.cluster_centers_ is not None
        assert len(ms.labels_) == len(X)

    def test_predict(self):
        X, _ = make_blobs(random_state=0)
        ms = MeanShift(bandwidth=3.0).fit(X)
        preds = ms.predict(X[:5])
        assert len(preds) == 5

    def test_single_cluster(self):
        rng = np.random.RandomState(0)
        X = rng.randn(30, 2) * 0.3
        ms = MeanShift(bandwidth=5.0).fit(X)
        assert len(ms.cluster_centers_) == 1


# ============================================================
# Affinity Propagation
# ============================================================

class TestAffinityPropagation:
    def test_basic(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [20, 20]]), n_per_cluster=20, random_state=0)
        ap = AffinityPropagation(damping=0.5, max_iter=200).fit(X)
        assert ap.cluster_centers_indices_ is not None
        assert len(ap.labels_) == len(X)

    def test_well_separated(self):
        X, y = make_blobs(centers=np.array([[0, 0], [50, 50]]), n_per_cluster=20, random_state=0)
        ap = AffinityPropagation(damping=0.5, max_iter=300).fit(X)
        n_clusters = len(np.unique(ap.labels_))
        assert n_clusters >= 2

    def test_damping(self):
        X, _ = make_blobs(n_per_cluster=15, random_state=0)
        ap = AffinityPropagation(damping=0.9).fit(X)
        assert len(ap.labels_) == len(X)

    def test_preference(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [20, 20]]), n_per_cluster=15, random_state=0)
        # Low preference -> fewer clusters, high preference -> more
        ap_low = AffinityPropagation(preference=-5000, damping=0.9, max_iter=300).fit(X)
        ap_high = AffinityPropagation(preference=-10, damping=0.9, max_iter=300).fit(X)
        assert len(np.unique(ap_low.labels_)) <= len(np.unique(ap_high.labels_))

    def test_fit_predict(self):
        X, _ = make_blobs(n_per_cluster=15, random_state=0)
        ap = AffinityPropagation(damping=0.5)
        labels = ap.fit_predict(X)
        np.testing.assert_array_equal(labels, ap.labels_)


# ============================================================
# Evaluation Metrics
# ============================================================

class TestSilhouette:
    def test_perfect_clusters(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        s = silhouette_score(X, y)
        assert s > 0.9

    def test_bad_clusters(self):
        X, _ = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        rng = np.random.RandomState(0)
        random_labels = rng.randint(0, 2, len(X))
        s = silhouette_score(X, random_labels)
        assert s < 0.5

    def test_single_cluster(self):
        X, _ = make_blobs(random_state=0)
        labels = np.zeros(len(X), dtype=int)
        s = silhouette_score(X, labels)
        assert s == 0.0

    def test_range(self):
        X, y = make_blobs(random_state=0)
        s = silhouette_score(X, y)
        assert -1 <= s <= 1


class TestCalinskiHarabasz:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        ch = calinski_harabasz_score(X, y)
        assert ch > 0

    def test_well_separated_higher(self):
        X_good, y_good = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        X_bad, _ = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        rng = np.random.RandomState(0)
        y_bad = rng.randint(0, 2, len(X_bad))
        ch_good = calinski_harabasz_score(X_good, y_good)
        ch_bad = calinski_harabasz_score(X_bad, y_bad)
        assert ch_good > ch_bad

    def test_single_cluster(self):
        X, _ = make_blobs(random_state=0)
        ch = calinski_harabasz_score(X, np.zeros(len(X), dtype=int))
        assert ch == 0.0


class TestDaviesBouldin:
    def test_basic(self):
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        db = davies_bouldin_score(X, y)
        assert db >= 0

    def test_well_separated_lower(self):
        X_good, y_good = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        X_bad, _ = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        rng = np.random.RandomState(0)
        y_bad = rng.randint(0, 2, len(X_bad))
        db_good = davies_bouldin_score(X_good, y_good)
        db_bad = davies_bouldin_score(X_bad, y_bad)
        assert db_good < db_bad

    def test_single_cluster(self):
        X, _ = make_blobs(random_state=0)
        db = davies_bouldin_score(X, np.zeros(len(X), dtype=int))
        assert db == 0.0


class TestAdjustedRandIndex:
    def test_perfect_match(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(labels, labels) == 1.0

    def test_permuted_labels(self):
        true = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([2, 2, 0, 0, 1, 1])  # same partition, different labels
        ari = adjusted_rand_index(true, pred)
        assert ari > 0.99

    def test_random_close_to_zero(self):
        rng = np.random.RandomState(0)
        true = rng.randint(0, 3, 100)
        pred = rng.randint(0, 3, 100)
        ari = adjusted_rand_index(true, pred)
        assert abs(ari) < 0.3  # should be near zero

    def test_all_same(self):
        true = np.array([0, 0, 0, 0])
        pred = np.array([1, 1, 1, 1])
        ari = adjusted_rand_index(true, pred)
        assert ari == 1.0  # both have single cluster

    def test_symmetry(self):
        true = np.array([0, 0, 1, 1, 2])
        pred = np.array([1, 1, 0, 0, 2])
        assert abs(adjusted_rand_index(true, pred) - adjusted_rand_index(pred, true)) < 1e-10


# ============================================================
# Cross-algorithm comparisons
# ============================================================

class TestCrossAlgorithm:
    def test_all_find_two_clusters(self):
        """All algorithms should find 2 well-separated clusters."""
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        algorithms = [
            KMeans(n_clusters=2, random_state=0),
            MiniBatchKMeans(n_clusters=2, random_state=0),
            DBSCAN(eps=3.0, min_samples=3),
            AgglomerativeClustering(n_clusters=2, linkage='ward'),
            BisectingKMeans(n_clusters=2, random_state=0),
            GaussianMixture(n_components=2, random_state=0),
        ]
        for algo in algorithms:
            algo.fit(X)
            labels = algo.labels_ if hasattr(algo, 'labels_') else algo.predict(X)
            # Filter noise for DBSCAN
            non_noise = labels >= 0
            if non_noise.sum() < len(X) * 0.5:
                continue
            ari = adjusted_rand_index(y[non_noise], labels[non_noise])
            assert ari > 0.8, f"{algo.__class__.__name__} failed with ARI={ari}"

    def test_metrics_consistent(self):
        """Good clustering should score well on all metrics."""
        X, y = make_blobs(centers=np.array([[0, 0], [100, 100]]), random_state=0)
        s = silhouette_score(X, y)
        ch = calinski_harabasz_score(X, y)
        db = davies_bouldin_score(X, y)
        assert s > 0.8
        assert ch > 100
        assert db < 0.5


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_duplicate_points(self):
        X = np.array([[0, 0]] * 10 + [[10, 10]] * 10, dtype=float)
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        assert len(np.unique(km.labels_)) == 2

    def test_single_feature(self):
        X = np.array([[0], [0.1], [0.2], [10], [10.1], [10.2]])
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        assert km.labels_[0] == km.labels_[1]
        assert km.labels_[3] == km.labels_[4]

    def test_three_dim(self):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(20, 3), rng.randn(20, 3) + 10])
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        assert len(np.unique(km.labels_)) == 2

    def test_small_dataset(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
        km = KMeans(n_clusters=2, random_state=0).fit(X)
        assert km.labels_[0] == km.labels_[1] == km.labels_[2]
        assert km.labels_[3] == km.labels_[4] == km.labels_[5]

    def test_gmm_single_component(self):
        X, _ = make_blobs(random_state=0)
        gm = GaussianMixture(n_components=1, random_state=0).fit(X)
        assert np.all(gm.predict(X) == 0)

    def test_agglomerative_one_cluster(self):
        X, _ = make_blobs(random_state=0)
        ac = AgglomerativeClustering(n_clusters=1).fit(X)
        assert np.all(ac.labels_ == 0)

    def test_dbscan_min_samples_1(self):
        X = np.array([[0, 0], [100, 100]])
        db = DBSCAN(eps=1.0, min_samples=1).fit(X)
        assert len(np.unique(db.labels_[db.labels_ >= 0])) == 2


# ============================================================
# Integration: clustering pipeline
# ============================================================

class TestPipeline:
    def test_kmeans_elbow(self):
        """Elbow method: inertia should decrease with more clusters."""
        X, _ = make_blobs(random_state=0)
        inertias = []
        for k in [1, 2, 3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=0).fit(X)
            inertias.append(km.inertia_)
        # Inertia should be monotonically decreasing
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_silhouette_selection(self):
        """Silhouette score should peak at correct k."""
        X, _ = make_blobs(centers=np.array([[0, 0], [10, 10], [20, 0]]), random_state=0)
        scores = {}
        for k in [2, 3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=0).fit(X)
            scores[k] = silhouette_score(X, km.labels_)
        best_k = max(scores, key=scores.get)
        assert best_k == 3

    def test_gmm_bic_selection(self):
        """BIC should favor the correct number of components."""
        X, _ = make_blobs(centers=np.array([[0, 0], [10, 10]]), random_state=0)
        bics = {}
        for k in [1, 2, 3, 4]:
            gm = GaussianMixture(n_components=k, random_state=0).fit(X)
            bics[k] = gm.bic(X)
        best_k = min(bics, key=bics.get)
        assert best_k == 2

    def test_fit_then_evaluate(self):
        """Full pipeline: fit, predict, evaluate."""
        X, y = make_blobs(random_state=0)
        km = KMeans(n_clusters=3, random_state=0).fit(X)
        s = silhouette_score(X, km.labels_)
        ch = calinski_harabasz_score(X, km.labels_)
        db = davies_bouldin_score(X, km.labels_)
        ari = adjusted_rand_index(y, km.labels_)
        assert s > 0
        assert ch > 0
        assert db >= 0
        assert ari > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
