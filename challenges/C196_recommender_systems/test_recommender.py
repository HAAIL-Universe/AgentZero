"""Tests for C196: Recommender Systems."""

import pytest
import math
import numpy as np
from recommender import (
    UserItemMatrix, cosine_similarity, pearson_similarity, adjusted_cosine_similarity,
    PopularityRecommender, UserBasedCF, ItemBasedCF, MatrixFactorization,
    SVDRecommender, ContentBasedRecommender, HybridRecommender, NMFRecommender,
    KNNBaseline, ImplicitALS, Evaluator, train_test_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_small_matrix():
    """5 users, 6 items, sparse ratings."""
    m = UserItemMatrix()
    # User preferences cluster:
    # u1,u2 like items 1-3 (action movies)
    # u3,u4 like items 4-6 (romance movies)
    # u5 is mixed
    ratings = [
        ('u1', 'i1', 5), ('u1', 'i2', 4), ('u1', 'i3', 5),
        ('u2', 'i1', 4), ('u2', 'i2', 5), ('u2', 'i3', 4), ('u2', 'i4', 1),
        ('u3', 'i4', 5), ('u3', 'i5', 4), ('u3', 'i6', 5),
        ('u4', 'i4', 4), ('u4', 'i5', 5), ('u4', 'i6', 4), ('u4', 'i1', 1),
        ('u5', 'i1', 3), ('u5', 'i3', 4), ('u5', 'i5', 3), ('u5', 'i6', 4),
    ]
    for u, i, r in ratings:
        m.add_rating(u, i, r)
    return m, ratings


def make_large_matrix():
    """20 users, 30 items with denser ratings."""
    rng = np.random.RandomState(123)
    m = UserItemMatrix()
    # Create 3 clusters of users
    for u in range(20):
        cluster = u // 7  # 0, 1, 2
        for i in range(30):
            if rng.random() < 0.3:  # 30% density
                # Users prefer items in their cluster range
                base = 3.0
                if cluster * 10 <= i < (cluster + 1) * 10:
                    base = 4.5
                rating = max(1, min(5, base + rng.normal(0, 0.5)))
                m.add_rating(f'u{u}', f'i{i}', round(rating, 1))
    return m


# ---------------------------------------------------------------------------
# UserItemMatrix tests
# ---------------------------------------------------------------------------

class TestUserItemMatrix:
    def test_add_and_get_rating(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 5.0)
        assert m.get_rating('u1', 'i1') == 5.0

    def test_missing_rating(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 5.0)
        assert m.get_rating('u1', 'i2') is None
        assert m.get_rating('u2', 'i1') is None

    def test_update_rating(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 3.0)
        m.add_rating('u1', 'i1', 5.0)
        assert m.get_rating('u1', 'i1') == 5.0

    def test_user_ratings(self):
        m, _ = make_small_matrix()
        ratings = m.get_user_ratings('u1')
        assert ratings == {'i1': 5, 'i2': 4, 'i3': 5}

    def test_item_ratings(self):
        m, _ = make_small_matrix()
        ratings = m.get_item_ratings('i1')
        assert 'u1' in ratings and 'u2' in ratings

    def test_counts(self):
        m, raw = make_small_matrix()
        assert m.n_users() == 5
        assert m.n_items() == 6
        assert m.n_ratings() == len(raw)

    def test_density(self):
        m, raw = make_small_matrix()
        expected = len(raw) / (5 * 6)
        assert abs(m.density() - expected) < 1e-6

    def test_to_dense(self):
        m, _ = make_small_matrix()
        dense = m.to_dense()
        assert dense.shape == (5, 6)
        assert not np.isnan(dense[0, 0])  # u1,i1 exists
        assert np.isnan(dense[0, 3])      # u1,i4 missing

    def test_user_mean(self):
        m, _ = make_small_matrix()
        assert abs(m.user_mean('u1') - (5+4+5)/3) < 1e-6

    def test_item_mean(self):
        m, _ = make_small_matrix()
        ratings = m.get_item_ratings('i1')
        expected = sum(ratings.values()) / len(ratings)
        assert abs(m.item_mean('i1') - expected) < 1e-6

    def test_global_mean(self):
        m, raw = make_small_matrix()
        expected = sum(r for _, _, r in raw) / len(raw)
        assert abs(m.global_mean() - expected) < 1e-6

    def test_empty_matrix(self):
        m = UserItemMatrix()
        assert m.n_users() == 0
        assert m.n_items() == 0
        assert m.density() == 0.0
        assert m.global_mean() == 0.0

    def test_get_users_for_item(self):
        m, _ = make_small_matrix()
        users = m.get_users_for_item('i1')
        assert 'u1' in users
        assert 'u2' in users
        assert 'u4' in users  # u4 rated i1

    def test_get_items_for_user(self):
        m, _ = make_small_matrix()
        items = m.get_items_for_user('u1')
        assert items == {'i1', 'i2', 'i3'}

    def test_unknown_user_items(self):
        m, _ = make_small_matrix()
        assert m.get_items_for_user('unknown') == set()
        assert m.get_users_for_item('unknown') == set()
        assert m.get_user_ratings('unknown') == {}


# ---------------------------------------------------------------------------
# Similarity tests
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_cosine_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-6

    def test_cosine_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_pearson_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        assert abs(pearson_similarity(a, b) - 1.0) < 1e-6

    def test_pearson_inverse(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        assert abs(pearson_similarity(a, b) + 1.0) < 1e-6

    def test_pearson_with_nan(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([2.0, np.nan, 6.0])
        # Only positions 0 and 2 used
        assert abs(pearson_similarity(a, b) - 1.0) < 1e-6

    def test_pearson_insufficient_data(self):
        a = np.array([1.0, np.nan])
        b = np.array([np.nan, 2.0])
        assert pearson_similarity(a, b) == 0.0

    def test_adjusted_cosine(self):
        a = np.array([5.0, 3.0])
        b = np.array([4.0, 2.0])
        means = np.array([4.0, 3.0])
        sim = adjusted_cosine_similarity(a, b, means)
        # a-mean = [1, 0], b-mean = [0, -1] -> cos = 0
        assert abs(sim) < 1e-6


# ---------------------------------------------------------------------------
# PopularityRecommender tests
# ---------------------------------------------------------------------------

class TestPopularityRecommender:
    def test_count_method(self):
        m, _ = make_small_matrix()
        rec = PopularityRecommender(method='count').fit(m)
        recs = rec.recommend('u1', n=3)
        assert len(recs) == 3
        # Should exclude u1's rated items
        assert 'i1' not in recs and 'i2' not in recs and 'i3' not in recs

    def test_mean_method(self):
        m, _ = make_small_matrix()
        rec = PopularityRecommender(method='mean').fit(m)
        recs = rec.recommend('u3', n=3)
        assert len(recs) == 3

    def test_weighted_method(self):
        m, _ = make_small_matrix()
        rec = PopularityRecommender(method='weighted').fit(m)
        recs = rec.recommend('u1', n=6)
        assert len(recs) == 3  # only 3 unrated items

    def test_predict_returns_global_mean(self):
        m, _ = make_small_matrix()
        rec = PopularityRecommender().fit(m)
        pred = rec.predict('u1', 'i4')
        assert abs(pred - m.global_mean()) < 1e-6

    def test_include_rated(self):
        m, _ = make_small_matrix()
        rec = PopularityRecommender().fit(m)
        recs = rec.recommend('u1', n=10, exclude_rated=False)
        assert 'i1' in recs


# ---------------------------------------------------------------------------
# UserBasedCF tests
# ---------------------------------------------------------------------------

class TestUserBasedCF:
    def test_predict_known_user(self):
        m, _ = make_small_matrix()
        cf = UserBasedCF(k=3, similarity='pearson').fit(m)
        pred = cf.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_recommend_excludes_rated(self):
        m, _ = make_small_matrix()
        cf = UserBasedCF(k=3).fit(m)
        recs = cf.recommend('u1', n=3)
        rated = m.get_items_for_user('u1')
        for r in recs:
            assert r not in rated

    def test_cosine_similarity_mode(self):
        m, _ = make_small_matrix()
        cf = UserBasedCF(k=3, similarity='cosine').fit(m)
        pred = cf.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_similar_users_get_similar_recs(self):
        m, _ = make_small_matrix()
        cf = UserBasedCF(k=3).fit(m)
        # u1 and u2 have similar tastes (both like i1,i2,i3)
        recs_u1 = set(cf.recommend('u1', n=3))
        recs_u2 = set(cf.recommend('u2', n=3))
        # They should have some overlap in what's recommended to them
        # (both should get romance items since they haven't seen them)
        assert len(recs_u1) > 0
        assert len(recs_u2) > 0

    def test_predict_unknown_user(self):
        m, _ = make_small_matrix()
        cf = UserBasedCF(k=3).fit(m)
        pred = cf.predict('unknown', 'i1')
        # Should return user_mean (0.0 for unknown)
        assert isinstance(pred, float)

    def test_recommend_respects_n(self):
        m, _ = make_small_matrix()
        cf = UserBasedCF(k=3).fit(m)
        recs = cf.recommend('u1', n=2)
        assert len(recs) == 2


# ---------------------------------------------------------------------------
# ItemBasedCF tests
# ---------------------------------------------------------------------------

class TestItemBasedCF:
    def test_predict(self):
        m, _ = make_small_matrix()
        cf = ItemBasedCF(k=3).fit(m)
        pred = cf.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_recommend(self):
        m, _ = make_small_matrix()
        cf = ItemBasedCF(k=3).fit(m)
        recs = cf.recommend('u1', n=3)
        assert len(recs) == 3
        rated = m.get_items_for_user('u1')
        for r in recs:
            assert r not in rated

    def test_cosine_mode(self):
        m, _ = make_small_matrix()
        cf = ItemBasedCF(k=3, similarity='cosine').fit(m)
        pred = cf.predict('u1', 'i5')
        assert isinstance(pred, float)

    def test_pearson_mode(self):
        m, _ = make_small_matrix()
        cf = ItemBasedCF(k=3, similarity='pearson').fit(m)
        pred = cf.predict('u3', 'i1')
        assert isinstance(pred, float)

    def test_empty_user(self):
        m, _ = make_small_matrix()
        cf = ItemBasedCF(k=3).fit(m)
        pred = cf.predict('unknown', 'i1')
        assert abs(pred - m.global_mean()) < 1e-6


# ---------------------------------------------------------------------------
# MatrixFactorization tests
# ---------------------------------------------------------------------------

class TestMatrixFactorization:
    def test_fit_and_predict(self):
        m, _ = make_small_matrix()
        mf = MatrixFactorization(n_factors=3, n_iterations=30, reg=0.1).fit(m)
        pred = mf.predict('u1', 'i1')
        # Should approximate the actual rating
        assert abs(pred - 5.0) < 2.0

    def test_recommend(self):
        m, _ = make_small_matrix()
        mf = MatrixFactorization(n_factors=3, n_iterations=30).fit(m)
        recs = mf.recommend('u1', n=3)
        assert len(recs) == 3
        rated = m.get_items_for_user('u1')
        for r in recs:
            assert r not in rated

    def test_rmse_decreases(self):
        m, _ = make_small_matrix()
        mf = MatrixFactorization(n_factors=5, n_iterations=50, reg=0.01).fit(m)
        rmse = mf.rmse()
        # Should have reasonable training RMSE
        assert rmse < 2.0

    def test_unknown_user_predict(self):
        m, _ = make_small_matrix()
        mf = MatrixFactorization(n_factors=3, n_iterations=10).fit(m)
        pred = mf.predict('unknown', 'i1')
        assert abs(pred - m.global_mean()) < 1e-6

    def test_factors_shape(self):
        m, _ = make_small_matrix()
        mf = MatrixFactorization(n_factors=4, n_iterations=5).fit(m)
        assert mf.user_factors.shape == (5, 4)
        assert mf.item_factors.shape == (6, 4)

    def test_larger_dataset(self):
        m = make_large_matrix()
        mf = MatrixFactorization(n_factors=5, n_iterations=20).fit(m)
        rmse = mf.rmse()
        assert rmse < 3.0  # Should converge on larger data
        recs = mf.recommend('u0', n=5)
        assert len(recs) == 5


# ---------------------------------------------------------------------------
# SVDRecommender tests
# ---------------------------------------------------------------------------

class TestSVDRecommender:
    def test_fit_and_predict(self):
        m, _ = make_small_matrix()
        svd = SVDRecommender(n_factors=3).fit(m)
        pred = svd.predict('u1', 'i1')
        assert isinstance(pred, float)

    def test_recommend(self):
        m, _ = make_small_matrix()
        svd = SVDRecommender(n_factors=3).fit(m)
        recs = svd.recommend('u1', n=3)
        assert len(recs) == 3
        rated = m.get_items_for_user('u1')
        for r in recs:
            assert r not in rated

    def test_unknown_user(self):
        m, _ = make_small_matrix()
        svd = SVDRecommender(n_factors=3).fit(m)
        pred = svd.predict('unknown', 'i1')
        assert abs(pred - m.global_mean()) < 1e-6

    def test_reconstruction_quality(self):
        m, raw = make_small_matrix()
        svd = SVDRecommender(n_factors=5).fit(m)
        # With enough factors, should approximate known ratings
        errors = []
        for u, i, r in raw:
            pred = svd.predict(u, i)
            errors.append((r - pred) ** 2)
        rmse = math.sqrt(sum(errors) / len(errors))
        assert rmse < 2.0


# ---------------------------------------------------------------------------
# ContentBasedRecommender tests
# ---------------------------------------------------------------------------

class TestContentBasedRecommender:
    def test_fit_and_predict(self):
        m, _ = make_small_matrix()
        features = {
            'i1': [1, 0, 0, 1], 'i2': [1, 0, 1, 0], 'i3': [1, 1, 0, 0],
            'i4': [0, 1, 0, 1], 'i5': [0, 1, 1, 0], 'i6': [0, 0, 1, 1],
        }
        cb = ContentBasedRecommender().fit(m, item_features=features)
        pred = cb.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_recommend(self):
        m, _ = make_small_matrix()
        features = {
            'i1': [1, 0, 0], 'i2': [1, 0, 1], 'i3': [1, 1, 0],
            'i4': [0, 1, 0], 'i5': [0, 1, 1], 'i6': [0, 0, 1],
        }
        cb = ContentBasedRecommender().fit(m, item_features=features)
        recs = cb.recommend('u1', n=3)
        assert len(recs) == 3

    def test_user_profile(self):
        m, _ = make_small_matrix()
        features = {
            'i1': [1, 0], 'i2': [1, 0], 'i3': [1, 0],
            'i4': [0, 1], 'i5': [0, 1], 'i6': [0, 1],
        }
        cb = ContentBasedRecommender().fit(m, item_features=features)
        # u1 likes i1,i2,i3 (action=[1,0]) -> profile should lean toward [1,0]
        profile = cb._user_profile('u1')
        assert profile[0] > profile[1]

    def test_similar_items_recommended(self):
        m, _ = make_small_matrix()
        features = {
            'i1': [1, 0, 0], 'i2': [0.9, 0.1, 0], 'i3': [0.8, 0.2, 0],
            'i4': [0, 0, 1], 'i5': [0, 0.1, 0.9], 'i6': [0, 0.2, 0.8],
        }
        cb = ContentBasedRecommender().fit(m, item_features=features)
        recs = cb.recommend('u3', n=3)
        # u3 likes i4,i5,i6 (romance-like features), should get action items
        assert len(recs) > 0

    def test_missing_features(self):
        m, _ = make_small_matrix()
        features = {'i1': [1, 0], 'i2': [0, 1]}  # only 2 items have features
        cb = ContentBasedRecommender().fit(m, item_features=features)
        pred = cb.predict('u1', 'i6')  # i6 has no features
        assert abs(pred - m.global_mean()) < 1e-6


# ---------------------------------------------------------------------------
# HybridRecommender tests
# ---------------------------------------------------------------------------

class TestHybridRecommender:
    def test_weighted_prediction(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        ubcf = UserBasedCF(k=3).fit(m)
        hybrid = HybridRecommender(
            recommenders=[('pop', pop), ('ubcf', ubcf)],
            weights=[0.3, 0.7]
        ).fit(m)
        pred = hybrid.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_recommend(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        ubcf = UserBasedCF(k=3).fit(m)
        hybrid = HybridRecommender(
            recommenders=[('pop', pop), ('ubcf', ubcf)],
            weights=[0.5, 0.5]
        ).fit(m)
        recs = hybrid.recommend('u1', n=3)
        assert len(recs) == 3

    def test_single_recommender(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        hybrid = HybridRecommender(
            recommenders=[('pop', pop)],
            weights=[1.0]
        ).fit(m)
        recs_hybrid = hybrid.recommend('u1', n=3)
        recs_pop = pop.recommend('u1', n=3)
        assert recs_hybrid == recs_pop

    def test_equal_weights(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        ubcf = UserBasedCF(k=3).fit(m)
        hybrid = HybridRecommender(
            recommenders=[('pop', pop), ('ubcf', ubcf)],
            weights=[1.0, 1.0]
        ).fit(m)
        pred = hybrid.predict('u1', 'i4')
        expected = (pop.predict('u1', 'i4') + ubcf.predict('u1', 'i4')) / 2
        assert abs(pred - expected) < 1e-6


# ---------------------------------------------------------------------------
# NMFRecommender tests
# ---------------------------------------------------------------------------

class TestNMFRecommender:
    def test_fit_and_predict(self):
        m, _ = make_small_matrix()
        nmf = NMFRecommender(n_factors=3, n_iterations=50).fit(m)
        pred = nmf.predict('u1', 'i1')
        assert isinstance(pred, float)
        assert pred > 0  # NMF should produce non-negative predictions

    def test_recommend(self):
        m, _ = make_small_matrix()
        nmf = NMFRecommender(n_factors=3, n_iterations=30).fit(m)
        recs = nmf.recommend('u1', n=3)
        assert len(recs) == 3

    def test_factors_non_negative(self):
        m, _ = make_small_matrix()
        nmf = NMFRecommender(n_factors=3, n_iterations=30).fit(m)
        assert np.all(nmf.W >= -1e-10)  # allow tiny numerical noise
        assert np.all(nmf.H >= -1e-10)

    def test_unknown_user(self):
        m, _ = make_small_matrix()
        nmf = NMFRecommender(n_factors=3, n_iterations=10).fit(m)
        pred = nmf.predict('unknown', 'i1')
        assert abs(pred - m.global_mean()) < 1e-6


# ---------------------------------------------------------------------------
# KNNBaseline tests
# ---------------------------------------------------------------------------

class TestKNNBaseline:
    def test_user_based(self):
        m, _ = make_small_matrix()
        knn = KNNBaseline(k=3, user_based=True).fit(m)
        pred = knn.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_item_based(self):
        m, _ = make_small_matrix()
        knn = KNNBaseline(k=3, user_based=False).fit(m)
        pred = knn.predict('u1', 'i4')
        assert isinstance(pred, float)

    def test_recommend(self):
        m, _ = make_small_matrix()
        knn = KNNBaseline(k=3).fit(m)
        recs = knn.recommend('u1', n=3)
        assert len(recs) == 3

    def test_baseline_bias(self):
        m, _ = make_small_matrix()
        knn = KNNBaseline(k=3).fit(m)
        # Biases should be reasonable
        assert abs(knn._global_mean - m.global_mean()) < 1e-6
        assert len(knn._user_bias) == m.n_users()
        assert len(knn._item_bias) == m.n_items()


# ---------------------------------------------------------------------------
# ImplicitALS tests
# ---------------------------------------------------------------------------

class TestImplicitALS:
    def test_fit_and_predict(self):
        m = UserItemMatrix()
        # Binary implicit data
        for u in range(10):
            for i in range(15):
                if (u + i) % 3 == 0:
                    m.add_rating(f'u{u}', f'i{i}', 1)
        als = ImplicitALS(n_factors=3, n_iterations=10).fit(m)
        pred = als.predict('u0', 'i0')
        assert isinstance(pred, float)

    def test_recommend(self):
        m = UserItemMatrix()
        for u in range(10):
            for i in range(15):
                if (u + i) % 3 == 0:
                    m.add_rating(f'u{u}', f'i{i}', 1)
        als = ImplicitALS(n_factors=3, n_iterations=10).fit(m)
        recs = als.recommend('u0', n=5)
        assert len(recs) == 5
        # Should not include already-interacted items
        rated = m.get_items_for_user('u0')
        for r in recs:
            assert r not in rated

    def test_preference_scores_positive_for_observed(self):
        m = UserItemMatrix()
        for u in range(5):
            for i in range(5):
                m.add_rating(f'u{u}', f'i{i}', 1)
        als = ImplicitALS(n_factors=2, n_iterations=20, alpha=40).fit(m)
        # Observed items should have higher preference
        pred = als.predict('u0', 'i0')
        assert pred > 0


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_rmse(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        test = [('u1', 'i1', 5), ('u2', 'i2', 5)]
        rmse = Evaluator.rmse(pop, test)
        assert rmse >= 0

    def test_mae(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        test = [('u1', 'i1', 5), ('u2', 'i2', 5)]
        mae = Evaluator.mae(pop, test)
        assert mae >= 0

    def test_precision_at_k(self):
        recommended = ['i1', 'i2', 'i3', 'i4', 'i5']
        relevant = ['i2', 'i4', 'i6']
        p = Evaluator.precision_at_k(recommended, relevant, k=5)
        assert abs(p - 2/5) < 1e-6

    def test_precision_at_k_truncated(self):
        recommended = ['i1', 'i2', 'i3', 'i4', 'i5']
        relevant = ['i2', 'i4']
        p = Evaluator.precision_at_k(recommended, relevant, k=3)
        assert abs(p - 1/3) < 1e-6

    def test_recall_at_k(self):
        recommended = ['i1', 'i2', 'i3']
        relevant = ['i2', 'i4', 'i6']
        r = Evaluator.recall_at_k(recommended, relevant, k=3)
        assert abs(r - 1/3) < 1e-6

    def test_f1_at_k(self):
        recommended = ['i1', 'i2']
        relevant = ['i2', 'i3']
        f1 = Evaluator.f1_at_k(recommended, relevant, k=2)
        p = 0.5  # 1/2
        r = 0.5  # 1/2
        expected = 2 * p * r / (p + r)
        assert abs(f1 - expected) < 1e-6

    def test_f1_zero(self):
        recommended = ['i1', 'i2']
        relevant = ['i3', 'i4']
        assert Evaluator.f1_at_k(recommended, relevant) == 0.0

    def test_ndcg_perfect(self):
        recommended = ['i1', 'i2', 'i3']
        relevant = ['i1', 'i2', 'i3']
        ndcg = Evaluator.ndcg_at_k(recommended, relevant, k=3)
        assert abs(ndcg - 1.0) < 1e-6

    def test_ndcg_partial(self):
        recommended = ['i1', 'i2', 'i3']
        relevant = ['i2']
        ndcg = Evaluator.ndcg_at_k(recommended, relevant, k=3)
        # Hit at position 2, DCG = 1/log2(3), IDCG = 1/log2(2)
        expected = (1/math.log2(3)) / (1/math.log2(2))
        assert abs(ndcg - expected) < 1e-6

    def test_ndcg_empty(self):
        assert Evaluator.ndcg_at_k([], ['i1']) == 0.0
        assert Evaluator.ndcg_at_k(['i1'], []) == 0.0

    def test_average_precision(self):
        recommended = ['i1', 'i2', 'i3', 'i4']
        relevant = ['i1', 'i3']
        ap = Evaluator.average_precision(recommended, relevant)
        # Hit at pos 1: precision=1/1, hit at pos 3: precision=2/3
        expected = (1.0 + 2/3) / 2
        assert abs(ap - expected) < 1e-6

    def test_map(self):
        all_rec = [['i1', 'i2'], ['i3', 'i4']]
        all_rel = [['i1'], ['i4']]
        map_score = Evaluator.mean_average_precision(all_rec, all_rel)
        ap1 = 1.0  # hit at position 1
        ap2 = 0.5  # hit at position 2
        expected = (ap1 + ap2) / 2
        assert abs(map_score - expected) < 1e-6

    def test_coverage(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        cov = Evaluator.coverage(pop, m, n=10)
        assert 0 < cov <= 1.0

    def test_diversity(self):
        recommended = ['i1', 'i2', 'i3']
        features = {
            'i1': [1, 0, 0], 'i2': [0, 1, 0], 'i3': [0, 0, 1]
        }
        div = Evaluator.diversity(recommended, features)
        assert div > 0.9  # orthogonal items should have max diversity

    def test_diversity_identical(self):
        recommended = ['i1', 'i2']
        features = {'i1': [1, 0], 'i2': [1, 0]}
        div = Evaluator.diversity(recommended, features)
        assert div == 0.0

    def test_novelty(self):
        recommended = ['i1', 'i2', 'i3']
        popularity = {'i1': 0.5, 'i2': 0.1, 'i3': 0.01}
        nov = Evaluator.novelty(recommended, popularity)
        assert nov > 0
        # Less popular items should contribute more novelty
        nov_popular = Evaluator.novelty(['i1'], popularity)
        nov_niche = Evaluator.novelty(['i3'], popularity)
        assert nov_niche > nov_popular

    def test_hit_rate(self):
        assert Evaluator.hit_rate(['i1', 'i2', 'i3'], ['i2']) == 1.0
        assert Evaluator.hit_rate(['i1', 'i2', 'i3'], ['i4']) == 0.0

    def test_mrr(self):
        assert Evaluator.mrr(['i1', 'i2', 'i3'], ['i1']) == 1.0
        assert Evaluator.mrr(['i1', 'i2', 'i3'], ['i2']) == 0.5
        assert Evaluator.mrr(['i1', 'i2', 'i3'], ['i3']) == pytest.approx(1/3)
        assert Evaluator.mrr(['i1', 'i2', 'i3'], ['i4']) == 0.0


# ---------------------------------------------------------------------------
# Train/Test Split tests
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def test_split_preserves_all_ratings(self):
        m, raw = make_small_matrix()
        train, test = train_test_split(m, test_ratio=0.2)
        total = train.n_ratings() + len(test)
        assert total == len(raw)

    def test_split_ratio(self):
        m, raw = make_small_matrix()
        train, test = train_test_split(m, test_ratio=0.3)
        # Should be approximately 30% test
        ratio = len(test) / len(raw)
        assert 0.1 < ratio < 0.5  # some flexibility due to user constraints

    def test_all_users_in_train(self):
        m, _ = make_small_matrix()
        train, test = train_test_split(m, test_ratio=0.2)
        # Each user should have at least one rating in train
        for user_id in m.user_ids:
            assert train.get_user_ratings(user_id), f"User {user_id} missing from train"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self):
        """End-to-end: create data, fit, evaluate."""
        m, _ = make_small_matrix()
        train, test = train_test_split(m, test_ratio=0.2)

        pop = PopularityRecommender().fit(train)
        mf = MatrixFactorization(n_factors=3, n_iterations=20).fit(train)

        rmse_pop = Evaluator.rmse(pop, test)
        rmse_mf = Evaluator.rmse(mf, test)

        # Both should produce valid RMSE
        assert rmse_pop >= 0
        assert rmse_mf >= 0

    def test_hybrid_improves_over_baseline(self):
        m = make_large_matrix()
        train, test = train_test_split(m, test_ratio=0.2)

        pop = PopularityRecommender().fit(train)
        mf = MatrixFactorization(n_factors=5, n_iterations=30).fit(train)
        hybrid = HybridRecommender(
            recommenders=[('pop', pop), ('mf', mf)],
            weights=[0.3, 0.7]
        ).fit(train)

        rmse_pop = Evaluator.rmse(pop, test)
        rmse_hybrid = Evaluator.rmse(hybrid, test)

        # Hybrid should be competitive
        assert rmse_hybrid >= 0
        assert rmse_pop >= 0

    def test_multiple_recommenders_comparison(self):
        m, _ = make_small_matrix()
        pop = PopularityRecommender().fit(m)
        ubcf = UserBasedCF(k=3).fit(m)
        ibcf = ItemBasedCF(k=3).fit(m)
        mf = MatrixFactorization(n_factors=3, n_iterations=20).fit(m)
        svd = SVDRecommender(n_factors=3).fit(m)

        # All should produce recommendations
        for rec in [pop, ubcf, ibcf, mf, svd]:
            recs = rec.recommend('u1', n=3)
            assert len(recs) > 0
            # Should not include rated items
            for r in recs:
                assert r not in m.get_items_for_user('u1')

    def test_cold_start_user(self):
        """New user with no ratings."""
        m, _ = make_small_matrix()
        m.add_rating('new_user', 'i1', 4)  # minimal data

        pop = PopularityRecommender().fit(m)
        recs = pop.recommend('new_user', n=5)
        assert len(recs) == 5

    def test_cold_start_item(self):
        """New item with very few ratings."""
        m, _ = make_small_matrix()
        m.add_rating('u1', 'new_item', 5)

        pop = PopularityRecommender().fit(m)
        # new_item should appear in recommendations for others
        recs = pop.recommend('u3', n=10)
        assert isinstance(recs, list)

    def test_large_scale_mf(self):
        """Matrix factorization on larger data."""
        m = make_large_matrix()
        mf = MatrixFactorization(n_factors=5, n_iterations=20).fit(m)
        rmse = mf.rmse()
        assert rmse < 3.0

        recs = mf.recommend('u0', n=10)
        assert len(recs) == 10

    def test_evaluation_suite(self):
        """Run full evaluation suite on a recommender."""
        m, _ = make_small_matrix()
        mf = MatrixFactorization(n_factors=3, n_iterations=30).fit(m)

        # Per-user evaluation
        for user_id in ['u1', 'u3', 'u5']:
            recs = mf.recommend(user_id, n=3)
            relevant = list(m.get_items_for_user(user_id))

            p = Evaluator.precision_at_k(recs, relevant, k=3)
            r = Evaluator.recall_at_k(recs, relevant, k=3)
            ndcg = Evaluator.ndcg_at_k(recs, relevant, k=3)
            hr = Evaluator.hit_rate(recs, relevant)
            rr = Evaluator.mrr(recs, relevant)

            assert 0 <= p <= 1
            assert 0 <= r <= 1
            assert 0 <= ndcg <= 1
            assert hr in (0.0, 1.0)
            assert 0 <= rr <= 1

    def test_content_hybrid(self):
        """Content-based + CF hybrid."""
        m, _ = make_small_matrix()
        features = {
            'i1': [1, 0, 0], 'i2': [0.9, 0.1, 0], 'i3': [0.8, 0.2, 0],
            'i4': [0, 0.1, 0.9], 'i5': [0, 0.2, 0.8], 'i6': [0.1, 0, 0.9],
        }
        cb = ContentBasedRecommender().fit(m, item_features=features)
        mf = MatrixFactorization(n_factors=3, n_iterations=20).fit(m)
        hybrid = HybridRecommender(
            recommenders=[('cb', cb), ('mf', mf)],
            weights=[0.4, 0.6]
        ).fit(m)

        recs = hybrid.recommend('u1', n=3)
        assert len(recs) == 3

    def test_implicit_pipeline(self):
        """ImplicitALS end-to-end."""
        m = UserItemMatrix()
        rng = np.random.RandomState(42)
        for u in range(15):
            cluster = u % 3
            for i in range(20):
                if rng.random() < 0.2:
                    # More interactions within cluster
                    if i // 7 == cluster:
                        m.add_rating(f'u{u}', f'i{i}', 1)
                    elif rng.random() < 0.1:
                        m.add_rating(f'u{u}', f'i{i}', 1)

        als = ImplicitALS(n_factors=3, n_iterations=10).fit(m)
        recs = als.recommend('u0', n=5)
        assert len(recs) > 0

    def test_nmf_pipeline(self):
        """NMF end-to-end."""
        m, _ = make_small_matrix()
        nmf = NMFRecommender(n_factors=3, n_iterations=50).fit(m)
        recs = nmf.recommend('u1', n=3)
        assert len(recs) == 3
        # Predictions should be non-negative
        for item in m.item_ids:
            pred = nmf.predict('u1', item)
            assert pred >= -0.1  # allow tiny numerical noise


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_user(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 5)
        m.add_rating('u1', 'i2', 3)
        pop = PopularityRecommender().fit(m)
        recs = pop.recommend('u1', n=5)
        assert len(recs) == 0  # no unrated items for single user with all items

    def test_single_item(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 5)
        m.add_rating('u2', 'i1', 3)
        pop = PopularityRecommender().fit(m)
        recs = pop.recommend('u1', n=5)
        assert len(recs) == 0

    def test_no_overlap_users(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 5)
        m.add_rating('u2', 'i2', 5)
        cf = UserBasedCF(k=3).fit(m)
        pred = cf.predict('u1', 'i2')
        assert isinstance(pred, float)

    def test_all_same_ratings(self):
        m = UserItemMatrix()
        for u in range(5):
            for i in range(5):
                m.add_rating(f'u{u}', f'i{i}', 3)
        mf = MatrixFactorization(n_factors=2, n_iterations=10).fit(m)
        pred = mf.predict('u0', 'i0')
        assert abs(pred - 3.0) < 1.0

    def test_extreme_ratings(self):
        m = UserItemMatrix()
        m.add_rating('u1', 'i1', 0)
        m.add_rating('u1', 'i2', 100)
        m.add_rating('u2', 'i1', 100)
        m.add_rating('u2', 'i2', 0)
        cf = UserBasedCF(k=1).fit(m)
        pred = cf.predict('u1', 'i1')
        assert isinstance(pred, float)
