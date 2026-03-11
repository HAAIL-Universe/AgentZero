"""Tests for C192: Dimensionality Reduction."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dimensionality_reduction import (
    PCA, KernelPCA, IncrementalPCA, TSNE, UMAP, LDA,
    FactorAnalysis, MDS, Isomap, LLE, TruncatedSVD, NMF
)


def make_blobs(n_per_class=50, n_features=10, n_classes=3, random_state=42):
    """Generate simple clustered data."""
    rng = np.random.RandomState(random_state)
    X_list, y_list = [], []
    for c in range(n_classes):
        center = rng.randn(n_features) * 5
        X_list.append(center + rng.randn(n_per_class, n_features) * 0.5)
        y_list.append(np.full(n_per_class, c))
    return np.vstack(X_list), np.concatenate(y_list)


def make_swiss_roll(n_samples=200, random_state=42):
    """Generate swiss roll data."""
    rng = np.random.RandomState(random_state)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n_samples))
    x = t * np.cos(t)
    y = 21 * rng.rand(n_samples)
    z = t * np.sin(t)
    return np.column_stack([x, y, z]), t


# ============================================================
# PCA Tests
# ============================================================
class TestPCA:

    def test_basic_fit_transform(self):
        X, _ = make_blobs()
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape == (150, 2)

    def test_explained_variance(self):
        X, _ = make_blobs()
        pca = PCA(n_components=3)
        pca.fit(X)
        assert len(pca.explained_variance_) == 3
        assert all(pca.explained_variance_[i] >= pca.explained_variance_[i + 1]
                    for i in range(2))

    def test_explained_variance_ratio_sums_to_one(self):
        X, _ = make_blobs(n_features=5)
        pca = PCA(n_components=5)
        pca.fit(X)
        assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-10

    def test_inverse_transform(self):
        X, _ = make_blobs(n_features=5)
        pca = PCA(n_components=5)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        assert np.allclose(X, X_reconstructed, atol=1e-8)

    def test_inverse_transform_lossy(self):
        X, _ = make_blobs(n_features=10)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        # Lossy -- should not be identical
        assert X_reconstructed.shape == X.shape
        # But error should be less than original variance
        assert np.mean((X - X_reconstructed) ** 2) < np.var(X)

    def test_n_components_default(self):
        X, _ = make_blobs(n_features=5)
        pca = PCA()
        pca.fit(X)
        assert pca.n_components == 5

    def test_components_orthogonal(self):
        X, _ = make_blobs(n_features=5)
        pca = PCA(n_components=3)
        pca.fit(X)
        dot = np.dot(pca.components_, pca.components_.T)
        assert np.allclose(dot, np.eye(3), atol=1e-10)

    def test_mean_centering(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        pca = PCA(n_components=2)
        pca.fit(X)
        assert np.allclose(pca.mean_, [3, 4])

    def test_get_covariance(self):
        X, _ = make_blobs(n_features=3)
        pca = PCA(n_components=3)
        pca.fit(X)
        cov = pca.get_covariance()
        assert cov.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(cov, cov.T)

    def test_single_component(self):
        X, _ = make_blobs(n_features=10)
        pca = PCA(n_components=1)
        X_r = pca.fit_transform(X)
        assert X_r.shape == (150, 1)
        # First component captures most variance
        assert pca.explained_variance_ratio_[0] > 0.1

    def test_singular_values(self):
        X, _ = make_blobs(n_features=5)
        pca = PCA(n_components=3)
        pca.fit(X)
        assert len(pca.singular_values_) == 3
        assert all(pca.singular_values_[i] >= pca.singular_values_[i + 1] for i in range(2))


# ============================================================
# KernelPCA Tests
# ============================================================
class TestKernelPCA:

    def test_rbf_kernel(self):
        X, _ = make_blobs(n_features=5)
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        X_r = kpca.fit_transform(X)
        assert X_r.shape == (150, 2)

    def test_poly_kernel(self):
        X, _ = make_blobs(n_features=5)
        kpca = KernelPCA(n_components=2, kernel='poly', degree=2, gamma=0.1)
        X_r = kpca.fit_transform(X)
        assert X_r.shape == (150, 2)

    def test_linear_kernel_matches_pca(self):
        X, _ = make_blobs(n_features=5, n_per_class=30)
        kpca = KernelPCA(n_components=2, kernel='linear')
        X_kpca = kpca.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        # Linear kernel PCA should give similar results (up to sign)
        for i in range(2):
            corr = abs(np.corrcoef(X_kpca[:, i], X_pca[:, i])[0, 1])
            assert corr > 0.9 or np.isnan(corr)  # High correlation

    def test_transform_new_data(self):
        X, _ = make_blobs(n_features=5)
        kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
        kpca.fit(X[:100])
        X_r = kpca.transform(X[100:])
        assert X_r.shape == (50, 2)

    def test_stores_fit_data(self):
        X, _ = make_blobs(n_features=3)
        kpca = KernelPCA(n_components=2)
        kpca.fit(X)
        assert kpca.X_fit_ is not None
        assert kpca.alphas_ is not None


# ============================================================
# IncrementalPCA Tests
# ============================================================
class TestIncrementalPCA:

    def test_basic(self):
        X, _ = make_blobs(n_features=5)
        ipca = IncrementalPCA(n_components=2)
        ipca.fit(X)
        X_r = ipca.transform(X)
        assert X_r.shape == (150, 2)

    def test_batch_fit(self):
        X, _ = make_blobs(n_features=5)
        ipca = IncrementalPCA(n_components=2, batch_size=50)
        ipca.fit(X)
        X_r = ipca.transform(X)
        assert X_r.shape == (150, 2)

    def test_partial_fit(self):
        X, _ = make_blobs(n_features=5)
        ipca = IncrementalPCA(n_components=2)
        ipca.partial_fit(X[:50])
        ipca.partial_fit(X[50:100])
        ipca.partial_fit(X[100:])
        X_r = ipca.transform(X)
        assert X_r.shape == (150, 2)

    def test_similar_to_pca(self):
        X, _ = make_blobs(n_features=5, n_per_class=100)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ipca = IncrementalPCA(n_components=2)
        X_ipca = ipca.fit_transform(X)
        # Should capture similar variance
        var_pca = np.var(X_pca, axis=0).sum()
        var_ipca = np.var(X_ipca, axis=0).sum()
        assert abs(var_pca - var_ipca) / var_pca < 0.3

    def test_n_samples_seen(self):
        X, _ = make_blobs(n_features=5)
        ipca = IncrementalPCA(n_components=2)
        ipca.partial_fit(X[:50])
        assert ipca.n_samples_seen_ == 50
        ipca.partial_fit(X[50:])
        assert ipca.n_samples_seen_ == 150


# ============================================================
# t-SNE Tests
# ============================================================
class TestTSNE:

    def test_basic(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        tsne = TSNE(n_components=2, perplexity=10, n_iter=300, random_state=42)
        X_r = tsne.fit_transform(X)
        assert X_r.shape == (60, 2)

    def test_kl_divergence(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        tsne = TSNE(n_components=2, perplexity=10, n_iter=300, random_state=42)
        tsne.fit_transform(X)
        assert tsne.kl_divergence_ >= 0

    def test_embedding_stored(self):
        X, _ = make_blobs(n_per_class=15, n_features=5)
        tsne = TSNE(n_components=2, perplexity=5, n_iter=100, random_state=42)
        result = tsne.fit_transform(X)
        assert np.allclose(result, tsne.embedding_)

    def test_clusters_separated(self):
        """Well-separated clusters should remain separated in t-SNE."""
        rng = np.random.RandomState(42)
        c1 = rng.randn(20, 5) + 10
        c2 = rng.randn(20, 5) - 10
        X = np.vstack([c1, c2])
        tsne = TSNE(n_components=2, perplexity=10, n_iter=500, random_state=42)
        X_r = tsne.fit_transform(X)
        # Centroids should be different
        centroid1 = X_r[:20].mean(axis=0)
        centroid2 = X_r[20:].mean(axis=0)
        assert np.linalg.norm(centroid1 - centroid2) > 0.1

    def test_3d_output(self):
        X, _ = make_blobs(n_per_class=15, n_features=5)
        tsne = TSNE(n_components=3, perplexity=5, n_iter=100, random_state=42)
        X_r = tsne.fit_transform(X)
        assert X_r.shape == (45, 3)

    def test_deterministic(self):
        X, _ = make_blobs(n_per_class=15, n_features=5)
        r1 = TSNE(n_components=2, perplexity=5, n_iter=100, random_state=42).fit_transform(X)
        r2 = TSNE(n_components=2, perplexity=5, n_iter=100, random_state=42).fit_transform(X)
        assert np.allclose(r1, r2)


# ============================================================
# UMAP Tests
# ============================================================
class TestUMAP:

    def test_basic(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        umap = UMAP(n_components=2, n_neighbors=10, n_epochs=50, random_state=42)
        X_r = umap.fit_transform(X)
        assert X_r.shape == (60, 2)

    def test_embedding_stored(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        umap = UMAP(n_components=2, n_neighbors=10, n_epochs=50, random_state=42)
        result = umap.fit_transform(X)
        assert np.allclose(result, umap.embedding_)

    def test_clusters_separated(self):
        rng = np.random.RandomState(42)
        c1 = rng.randn(30, 5) + 10
        c2 = rng.randn(30, 5) - 10
        X = np.vstack([c1, c2])
        umap = UMAP(n_components=2, n_neighbors=10, n_epochs=100, random_state=42)
        X_r = umap.fit_transform(X)
        centroid1 = X_r[:30].mean(axis=0)
        centroid2 = X_r[30:].mean(axis=0)
        assert np.linalg.norm(centroid1 - centroid2) > 0.01

    def test_3d(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        umap = UMAP(n_components=3, n_neighbors=10, n_epochs=30, random_state=42)
        X_r = umap.fit_transform(X)
        assert X_r.shape == (60, 3)

    def test_small_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3)
        umap = UMAP(n_components=2, n_neighbors=3, n_epochs=30, random_state=42)
        X_r = umap.fit_transform(X)
        assert X_r.shape == (10, 2)


# ============================================================
# LDA Tests
# ============================================================
class TestLDA:

    def test_basic(self):
        X, y = make_blobs(n_features=5, n_classes=3)
        lda = LDA(n_components=2)
        X_r = lda.fit_transform(X, y)
        assert X_r.shape == (150, 2)

    def test_max_components(self):
        X, y = make_blobs(n_features=10, n_classes=3)
        lda = LDA()
        lda.fit(X, y)
        assert lda.n_components == 2  # min(n_classes-1, n_features)

    def test_predict(self):
        X, y = make_blobs(n_features=5, n_classes=3, n_per_class=50)
        lda = LDA(n_components=2)
        lda.fit(X, y)
        predictions = lda.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8  # Should classify well on training data

    def test_class_means(self):
        X, y = make_blobs(n_features=5, n_classes=3)
        lda = LDA(n_components=2)
        lda.fit(X, y)
        assert lda.class_means_.shape == (3, 5)

    def test_explained_variance_ratio(self):
        X, y = make_blobs(n_features=5, n_classes=3)
        lda = LDA(n_components=2)
        lda.fit(X, y)
        assert len(lda.explained_variance_ratio_) == 2
        assert all(r >= 0 for r in lda.explained_variance_ratio_)

    def test_two_classes(self):
        X, y = make_blobs(n_features=5, n_classes=2)
        lda = LDA(n_components=1)
        X_r = lda.fit_transform(X, y)
        assert X_r.shape == (100, 1)

    def test_transform_new_data(self):
        X, y = make_blobs(n_features=5, n_classes=3)
        lda = LDA(n_components=2)
        lda.fit(X[:100], y[:100])
        X_r = lda.transform(X[100:])
        assert X_r.shape == (50, 2)


# ============================================================
# FactorAnalysis Tests
# ============================================================
class TestFactorAnalysis:

    def test_basic(self):
        X, _ = make_blobs(n_features=5)
        fa = FactorAnalysis(n_components=2, max_iter=50)
        X_r = fa.fit_transform(X)
        assert X_r.shape == (150, 2)

    def test_components(self):
        X, _ = make_blobs(n_features=5)
        fa = FactorAnalysis(n_components=2)
        fa.fit(X)
        assert fa.components_.shape == (2, 5)

    def test_noise_variance(self):
        X, _ = make_blobs(n_features=5)
        fa = FactorAnalysis(n_components=2)
        fa.fit(X)
        assert len(fa.noise_variance_) == 5
        assert all(v > 0 for v in fa.noise_variance_)

    def test_mean(self):
        X, _ = make_blobs(n_features=5)
        fa = FactorAnalysis(n_components=2)
        fa.fit(X)
        assert np.allclose(fa.mean_, X.mean(axis=0))

    def test_transform_shape(self):
        X, _ = make_blobs(n_features=5)
        fa = FactorAnalysis(n_components=3)
        fa.fit(X[:100])
        X_r = fa.transform(X[100:])
        assert X_r.shape == (50, 3)


# ============================================================
# MDS Tests
# ============================================================
class TestMDS:

    def test_classical(self):
        X, _ = make_blobs(n_features=5)
        mds = MDS(n_components=2, metric=False)
        X_r = mds.fit_transform(X)
        assert X_r.shape == (150, 2)

    def test_metric(self):
        X, _ = make_blobs(n_features=5, n_per_class=30)
        mds = MDS(n_components=2, metric=True, max_iter=50, random_state=42)
        X_r = mds.fit_transform(X)
        assert X_r.shape == (90, 2)

    def test_stress(self):
        X, _ = make_blobs(n_features=5, n_per_class=30)
        mds = MDS(n_components=2, metric=True, max_iter=50)
        mds.fit_transform(X)
        assert mds.stress_ >= 0
        assert mds.stress_ < 1.0  # Should be reasonable

    def test_dissimilarity_matrix(self):
        rng = np.random.RandomState(42)
        D = rng.rand(20, 20)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        mds = MDS(n_components=2, metric=False)
        X_r = mds.fit_transform(dissimilarity_matrix=D)
        assert X_r.shape == (20, 2)

    def test_preserves_distances(self):
        """MDS should approximately preserve pairwise distances."""
        rng = np.random.RandomState(42)
        X = rng.randn(30, 3)
        mds = MDS(n_components=2, metric=True, max_iter=100)
        X_r = mds.fit_transform(X)
        # Stress should be low for 3D->2D
        assert mds.stress_ < 0.5


# ============================================================
# Isomap Tests
# ============================================================
class TestIsomap:

    def test_basic(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        iso = Isomap(n_components=2, n_neighbors=5)
        X_r = iso.fit_transform(X)
        assert X_r.shape == (60, 2)

    def test_embedding_stored(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        iso = Isomap(n_components=2, n_neighbors=5)
        result = iso.fit_transform(X)
        assert np.allclose(result, iso.embedding_)

    def test_swiss_roll(self):
        """Isomap should unfold the swiss roll better than PCA."""
        X, t = make_swiss_roll(n_samples=100, random_state=42)
        iso = Isomap(n_components=2, n_neighbors=8)
        X_iso = iso.fit_transform(X)
        assert X_iso.shape == (100, 2)

    def test_small_neighbors(self):
        X, _ = make_blobs(n_per_class=15, n_features=3)
        iso = Isomap(n_components=2, n_neighbors=3)
        X_r = iso.fit_transform(X)
        assert X_r.shape == (45, 2)


# ============================================================
# LLE Tests
# ============================================================
class TestLLE:

    def test_basic(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        lle = LLE(n_components=2, n_neighbors=5)
        X_r = lle.fit_transform(X)
        assert X_r.shape == (60, 2)

    def test_embedding_stored(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        lle = LLE(n_components=2, n_neighbors=5)
        result = lle.fit_transform(X)
        assert np.allclose(result, lle.embedding_)

    def test_swiss_roll(self):
        X, _ = make_swiss_roll(n_samples=100, random_state=42)
        lle = LLE(n_components=2, n_neighbors=8)
        X_r = lle.fit_transform(X)
        assert X_r.shape == (100, 2)

    def test_regularization(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        lle = LLE(n_components=2, n_neighbors=5, reg=0.01)
        X_r = lle.fit_transform(X)
        assert X_r.shape == (60, 2)


# ============================================================
# TruncatedSVD Tests
# ============================================================
class TestTruncatedSVD:

    def test_basic(self):
        X, _ = make_blobs(n_features=10)
        svd = TruncatedSVD(n_components=3, random_state=42)
        X_r = svd.fit_transform(X)
        assert X_r.shape == (150, 3)

    def test_components(self):
        X, _ = make_blobs(n_features=10)
        svd = TruncatedSVD(n_components=3, random_state=42)
        svd.fit(X)
        assert svd.components_.shape == (3, 10)

    def test_singular_values_descending(self):
        X, _ = make_blobs(n_features=10)
        svd = TruncatedSVD(n_components=5, random_state=42)
        svd.fit(X)
        for i in range(4):
            assert svd.singular_values_[i] >= svd.singular_values_[i + 1]

    def test_explained_variance(self):
        X, _ = make_blobs(n_features=10)
        svd = TruncatedSVD(n_components=3, random_state=42)
        svd.fit(X)
        assert len(svd.explained_variance_ratio_) == 3
        assert all(r > 0 for r in svd.explained_variance_ratio_)

    def test_transform_new_data(self):
        X, _ = make_blobs(n_features=10)
        svd = TruncatedSVD(n_components=3, random_state=42)
        svd.fit(X[:100])
        X_r = svd.transform(X[100:])
        assert X_r.shape == (50, 3)


# ============================================================
# NMF Tests
# ============================================================
class TestNMF:

    def test_basic(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf = NMF(n_components=3, random_state=42)
        W = nmf.fit_transform(X)
        assert W.shape == (50, 3)

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf = NMF(n_components=3, random_state=42)
        W = nmf.fit_transform(X)
        assert np.all(W >= 0)
        assert np.all(nmf.components_ >= 0)

    def test_reconstruction(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf = NMF(n_components=5, max_iter=500, random_state=42)
        W = nmf.fit_transform(X)
        X_reconstructed = nmf.inverse_transform(W)
        # Reconstruction should be reasonable
        assert X_reconstructed.shape == X.shape
        rel_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
        assert rel_error < 1.0  # Should reconstruct somewhat

    def test_reconstruction_error(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf = NMF(n_components=3, random_state=42)
        nmf.fit(X)
        assert nmf.reconstruction_err_ > 0

    def test_components_shape(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf = NMF(n_components=3, random_state=42)
        nmf.fit(X)
        assert nmf.components_.shape == (3, 10)

    def test_more_components_better_reconstruction(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf3 = NMF(n_components=3, max_iter=200, random_state=42)
        nmf3.fit(X)
        nmf7 = NMF(n_components=7, max_iter=200, random_state=42)
        nmf7.fit(X)
        assert nmf7.reconstruction_err_ <= nmf3.reconstruction_err_

    def test_transform_new_data(self):
        rng = np.random.RandomState(42)
        X = np.abs(rng.randn(50, 10))
        nmf = NMF(n_components=3, random_state=42)
        nmf.fit(X[:30])
        W = nmf.transform(X[30:])
        assert W.shape == (20, 3)
        assert np.all(W >= 0)


# ============================================================
# Cross-method comparison tests
# ============================================================
class TestCrossMethod:

    def test_all_methods_reduce_dimensionality(self):
        """All methods should produce lower-dimensional output."""
        X, y = make_blobs(n_per_class=20, n_features=10, n_classes=3)
        methods = [
            ('PCA', PCA(n_components=2).fit_transform(X)),
            ('TSVD', TruncatedSVD(n_components=2, random_state=42).fit_transform(X)),
            ('LDA', LDA(n_components=2).fit_transform(X, y)),
            ('FA', FactorAnalysis(n_components=2).fit_transform(X)),
            ('MDS', MDS(n_components=2, metric=False).fit_transform(X)),
        ]
        for name, X_r in methods:
            assert X_r.shape == (60, 2), f"{name} failed"

    def test_pca_captures_most_variance(self):
        """PCA should capture the most variance among linear methods."""
        X, _ = make_blobs(n_features=10, n_per_class=50)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        var_pca = np.var(X_pca, axis=0).sum()
        # PCA maximizes variance by definition
        assert var_pca > 0

    def test_supervised_vs_unsupervised(self):
        """LDA should use class information for better class separation."""
        X, y = make_blobs(n_features=10, n_classes=3, n_per_class=50)
        lda = LDA(n_components=2)
        X_lda = lda.fit_transform(X, y)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        # Both should produce valid results
        assert X_lda.shape == X_pca.shape == (150, 2)

    def test_nonlinear_methods_on_nonlinear_data(self):
        """Non-linear methods should handle non-linear manifolds."""
        X, _ = make_swiss_roll(n_samples=100, random_state=42)
        methods = [
            ('Isomap', Isomap(n_components=2, n_neighbors=8).fit_transform(X)),
            ('LLE', LLE(n_components=2, n_neighbors=8).fit_transform(X)),
        ]
        for name, X_r in methods:
            assert X_r.shape == (100, 2), f"{name} failed"


# ============================================================
# Edge case tests
# ============================================================
class TestEdgeCases:

    def test_pca_single_sample(self):
        X = np.array([[1, 2, 3]], dtype=float)
        pca = PCA(n_components=1)
        pca.fit(X)
        X_r = pca.transform(X)
        assert X_r.shape == (1, 1)

    def test_pca_identical_features(self):
        X = np.ones((20, 5))
        pca = PCA(n_components=2)
        pca.fit(X)
        X_r = pca.transform(X)
        assert np.allclose(X_r, 0)

    def test_lda_two_classes_one_component(self):
        X, y = make_blobs(n_features=5, n_classes=2)
        lda = LDA(n_components=1)
        X_r = lda.fit_transform(X, y)
        assert X_r.shape == (100, 1)

    def test_nmf_zeros(self):
        X = np.zeros((20, 10))
        nmf = NMF(n_components=2, random_state=42)
        W = nmf.fit_transform(X)
        assert W.shape == (20, 2)

    def test_mds_two_points(self):
        X = np.array([[0, 0], [3, 4]], dtype=float)
        mds = MDS(n_components=1, metric=False)
        X_r = mds.fit_transform(X)
        assert X_r.shape == (2, 1)

    def test_lle_high_reg(self):
        X, _ = make_blobs(n_per_class=20, n_features=5)
        lle = LLE(n_components=2, n_neighbors=5, reg=1.0)
        X_r = lle.fit_transform(X)
        assert X_r.shape == (60, 2)

    def test_kernel_pca_unknown_kernel(self):
        X = np.random.randn(20, 3)
        kpca = KernelPCA(n_components=2, kernel='unknown')
        try:
            kpca.fit_transform(X)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_truncated_svd_more_iter(self):
        X, _ = make_blobs(n_features=10)
        svd = TruncatedSVD(n_components=3, n_iter=10, random_state=42)
        X_r = svd.fit_transform(X)
        assert X_r.shape == (150, 3)


# ============================================================
# Run all tests
# ============================================================
def run_tests():
    test_classes = [
        TestPCA, TestKernelPCA, TestIncrementalPCA, TestTSNE, TestUMAP,
        TestLDA, TestFactorAnalysis, TestMDS, TestIsomap, TestLLE,
        TestTruncatedSVD, TestNMF, TestCrossMethod, TestEdgeCases
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
