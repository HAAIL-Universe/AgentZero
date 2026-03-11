"""
C196: Recommender Systems
Built from scratch using only NumPy.

Components:
1. UserItemMatrix -- sparse user-item interaction matrix
2. CosineSimilarity -- similarity computation
3. UserBasedCF -- user-based collaborative filtering
4. ItemBasedCF -- item-based collaborative filtering
5. MatrixFactorization -- ALS-based matrix factorization
6. SVDRecommender -- truncated SVD via power iteration
7. ContentBasedRecommender -- TF-IDF content features
8. HybridRecommender -- weighted hybrid of multiple recommenders
9. PopularityRecommender -- baseline popularity recommender
10. Evaluator -- precision, recall, NDCG, MAP, coverage, diversity
"""

import numpy as np
from collections import defaultdict
import math


# ---------------------------------------------------------------------------
# UserItemMatrix -- sparse representation
# ---------------------------------------------------------------------------

class UserItemMatrix:
    """Sparse user-item interaction matrix with efficient lookups."""

    def __init__(self):
        self.user_ids = []
        self.item_ids = []
        self._user_idx = {}   # user_id -> index
        self._item_idx = {}   # item_id -> index
        self._ratings = {}    # (user_idx, item_idx) -> rating
        self._user_items = defaultdict(dict)  # user_idx -> {item_idx: rating}
        self._item_users = defaultdict(dict)  # item_idx -> {user_idx: rating}

    def add_rating(self, user_id, item_id, rating):
        """Add or update a rating."""
        if user_id not in self._user_idx:
            self._user_idx[user_id] = len(self.user_ids)
            self.user_ids.append(user_id)
        if item_id not in self._item_idx:
            self._item_idx[item_id] = len(self.item_ids)
            self.item_ids.append(item_id)

        ui = self._user_idx[user_id]
        ii = self._item_idx[item_id]
        self._ratings[(ui, ii)] = rating
        self._user_items[ui][ii] = rating
        self._item_users[ii][ui] = rating

    def get_rating(self, user_id, item_id):
        """Get rating or None."""
        ui = self._user_idx.get(user_id)
        ii = self._item_idx.get(item_id)
        if ui is None or ii is None:
            return None
        return self._ratings.get((ui, ii))

    def get_user_ratings(self, user_id):
        """Get {item_id: rating} for a user."""
        ui = self._user_idx.get(user_id)
        if ui is None:
            return {}
        return {self.item_ids[ii]: r for ii, r in self._user_items[ui].items()}

    def get_item_ratings(self, item_id):
        """Get {user_id: rating} for an item."""
        ii = self._item_idx.get(item_id)
        if ii is None:
            return {}
        return {self.user_ids[ui]: r for ui, r in self._item_users[ii].items()}

    def to_dense(self):
        """Convert to dense numpy array (users x items), NaN for missing."""
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        mat = np.full((n_users, n_items), np.nan)
        for (ui, ii), r in self._ratings.items():
            mat[ui, ii] = r
        return mat

    def n_users(self):
        return len(self.user_ids)

    def n_items(self):
        return len(self.item_ids)

    def n_ratings(self):
        return len(self._ratings)

    def density(self):
        total = self.n_users() * self.n_items()
        if total == 0:
            return 0.0
        return self.n_ratings() / total

    def user_mean(self, user_id):
        """Mean rating for a user."""
        ratings = self.get_user_ratings(user_id)
        if not ratings:
            return 0.0
        return sum(ratings.values()) / len(ratings)

    def item_mean(self, item_id):
        """Mean rating for an item."""
        ratings = self.get_item_ratings(item_id)
        if not ratings:
            return 0.0
        return sum(ratings.values()) / len(ratings)

    def global_mean(self):
        """Global mean rating."""
        if not self._ratings:
            return 0.0
        return sum(self._ratings.values()) / len(self._ratings)

    def get_users_for_item(self, item_id):
        """Get set of user_ids who rated this item."""
        ii = self._item_idx.get(item_id)
        if ii is None:
            return set()
        return {self.user_ids[ui] for ui in self._item_users[ii]}

    def get_items_for_user(self, user_id):
        """Get set of item_ids rated by this user."""
        ui = self._user_idx.get(user_id)
        if ui is None:
            return set()
        return {self.item_ids[ii] for ii in self._user_items[ui]}


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two numpy vectors."""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def pearson_similarity(vec_a, vec_b):
    """Pearson correlation between two vectors (ignore NaN)."""
    mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
    if mask.sum() < 2:
        return 0.0
    a = vec_a[mask]
    b = vec_b[mask]
    a_mean = a.mean()
    b_mean = b.mean()
    a_centered = a - a_mean
    b_centered = b - b_mean
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def adjusted_cosine_similarity(vec_a, vec_b, user_means):
    """Adjusted cosine: subtract user means before computing cosine."""
    mask = ~(np.isnan(vec_a) | np.isnan(vec_b))
    if mask.sum() == 0:
        return 0.0
    a = vec_a[mask] - user_means[mask]
    b = vec_b[mask] - user_means[mask]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# PopularityRecommender -- baseline
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """Recommends most popular items (by rating count or mean rating)."""

    def __init__(self, method='count'):
        self.method = method  # 'count' or 'mean' or 'weighted'
        self._item_scores = {}
        self._matrix = None

    def fit(self, matrix):
        self._matrix = matrix
        self._item_scores = {}
        for item_id in matrix.item_ids:
            ratings = matrix.get_item_ratings(item_id)
            if not ratings:
                self._item_scores[item_id] = 0.0
                continue
            count = len(ratings)
            mean = sum(ratings.values()) / count
            if self.method == 'count':
                self._item_scores[item_id] = count
            elif self.method == 'mean':
                self._item_scores[item_id] = mean
            else:  # weighted
                self._item_scores[item_id] = mean * math.log1p(count)
        return self

    def recommend(self, user_id, n=10, exclude_rated=True):
        """Return top-n item_ids."""
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        candidates = [(item, score) for item, score in self._item_scores.items()
                       if item not in rated]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]

    def predict(self, user_id, item_id):
        """Predict rating as global mean (baseline)."""
        return self._matrix.global_mean()


# ---------------------------------------------------------------------------
# UserBasedCF
# ---------------------------------------------------------------------------

class UserBasedCF:
    """User-based collaborative filtering with k-nearest neighbors."""

    def __init__(self, k=20, similarity='pearson', min_common=1):
        self.k = k
        self.similarity = similarity
        self.min_common = min_common
        self._matrix = None
        self._dense = None
        self._sim_cache = {}

    def fit(self, matrix):
        self._matrix = matrix
        self._dense = matrix.to_dense()
        self._sim_cache = {}
        return self

    def _compute_sim(self, ui_a, ui_b):
        key = (min(ui_a, ui_b), max(ui_a, ui_b))
        if key in self._sim_cache:
            return self._sim_cache[key]
        va = self._dense[ui_a]
        vb = self._dense[ui_b]
        mask = ~(np.isnan(va) | np.isnan(vb))
        if mask.sum() < self.min_common:
            sim = 0.0
        elif self.similarity == 'cosine':
            a = va[mask]
            b = vb[mask]
            sim = cosine_similarity(a, b)
        else:
            sim = pearson_similarity(va, vb)
        self._sim_cache[key] = sim
        return sim

    def _get_neighbors(self, user_id, item_id):
        """Find k most similar users who rated the item."""
        ui = self._matrix._user_idx.get(user_id)
        ii = self._matrix._item_idx.get(item_id)
        if ui is None or ii is None:
            return []
        neighbors = []
        for other_ui in self._matrix._item_users.get(ii, {}):
            if other_ui == ui:
                continue
            sim = self._compute_sim(ui, other_ui)
            if sim != 0:
                neighbors.append((other_ui, sim))
        neighbors.sort(key=lambda x: -abs(x[1]))
        return neighbors[:self.k]

    def predict(self, user_id, item_id):
        """Predict rating using weighted mean deviation."""
        user_mean = self._matrix.user_mean(user_id)
        neighbors = self._get_neighbors(user_id, item_id)
        if not neighbors:
            return user_mean

        ii = self._matrix._item_idx[item_id]
        num = 0.0
        denom = 0.0
        for other_ui, sim in neighbors:
            other_mean = self._dense[other_ui][~np.isnan(self._dense[other_ui])].mean()
            rating = self._ratings_at(other_ui, ii)
            num += sim * (rating - other_mean)
            denom += abs(sim)
        if denom == 0:
            return user_mean
        return user_mean + num / denom

    def _ratings_at(self, ui, ii):
        return self._matrix._ratings.get((ui, ii), 0.0)

    def recommend(self, user_id, n=10, exclude_rated=True):
        """Recommend top-n items."""
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        candidates = []
        for item_id in self._matrix.item_ids:
            if item_id in rated:
                continue
            score = self.predict(user_id, item_id)
            candidates.append((item_id, score))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# ItemBasedCF
# ---------------------------------------------------------------------------

class ItemBasedCF:
    """Item-based collaborative filtering."""

    def __init__(self, k=20, similarity='adjusted_cosine', min_common=1):
        self.k = k
        self.similarity = similarity
        self.min_common = min_common
        self._matrix = None
        self._dense = None
        self._sim_cache = {}

    def fit(self, matrix):
        self._matrix = matrix
        self._dense = matrix.to_dense()
        self._sim_cache = {}
        return self

    def _compute_item_sim(self, ii_a, ii_b):
        key = (min(ii_a, ii_b), max(ii_a, ii_b))
        if key in self._sim_cache:
            return self._sim_cache[key]

        va = self._dense[:, ii_a]
        vb = self._dense[:, ii_b]
        mask = ~(np.isnan(va) | np.isnan(vb))
        if mask.sum() < self.min_common:
            sim = 0.0
        elif self.similarity == 'adjusted_cosine':
            # Compute user means for co-rated users
            user_means = np.nanmean(self._dense[mask], axis=1)
            a = va[mask] - user_means
            b = vb[mask] - user_means
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            sim = float(np.dot(a, b) / denom) if denom != 0 else 0.0
        elif self.similarity == 'cosine':
            sim = cosine_similarity(va[mask], vb[mask])
        else:
            sim = pearson_similarity(va, vb)
        self._sim_cache[key] = sim
        return sim

    def _get_similar_items(self, item_id, user_items):
        """Find k most similar items to item_id among user's rated items."""
        ii = self._matrix._item_idx.get(item_id)
        if ii is None:
            return []
        neighbors = []
        for other_item in user_items:
            other_ii = self._matrix._item_idx.get(other_item)
            if other_ii is None or other_ii == ii:
                continue
            sim = self._compute_item_sim(ii, other_ii)
            if sim != 0:
                neighbors.append((other_item, sim))
        neighbors.sort(key=lambda x: -abs(x[1]))
        return neighbors[:self.k]

    def predict(self, user_id, item_id):
        """Predict rating using weighted sum of similar items."""
        user_ratings = self._matrix.get_user_ratings(user_id)
        if not user_ratings:
            return self._matrix.global_mean()

        neighbors = self._get_similar_items(item_id, user_ratings.keys())
        if not neighbors:
            return self._matrix.user_mean(user_id)

        num = 0.0
        denom = 0.0
        for other_item, sim in neighbors:
            num += sim * user_ratings[other_item]
            denom += abs(sim)
        if denom == 0:
            return self._matrix.user_mean(user_id)
        return num / denom

    def recommend(self, user_id, n=10, exclude_rated=True):
        """Recommend top-n items."""
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        candidates = []
        for item_id in self._matrix.item_ids:
            if item_id in rated:
                continue
            score = self.predict(user_id, item_id)
            candidates.append((item_id, score))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# MatrixFactorization -- ALS
# ---------------------------------------------------------------------------

class MatrixFactorization:
    """Matrix Factorization using Alternating Least Squares (ALS)."""

    def __init__(self, n_factors=10, n_iterations=20, reg=0.1, learning_rate=None):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self._matrix = None
        self.user_factors = None  # (n_users, n_factors)
        self.item_factors = None  # (n_items, n_factors)
        self.user_bias = None
        self.item_bias = None
        self.global_bias = 0.0

    def fit(self, matrix, verbose=False):
        self._matrix = matrix
        n_users = matrix.n_users()
        n_items = matrix.n_items()

        rng = np.random.RandomState(42)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = matrix.global_mean()

        # Build rating list for efficiency
        ratings_list = [(ui, ii, r) for (ui, ii), r in matrix._ratings.items()]

        for iteration in range(self.n_iterations):
            # Fix items, solve for users
            for ui in range(n_users):
                items_rated = matrix._user_items.get(ui, {})
                if not items_rated:
                    continue
                ii_list = list(items_rated.keys())
                V = self.item_factors[ii_list]  # (n_rated, n_factors)
                r_vec = np.array([items_rated[ii] - self.global_bias - self.item_bias[ii]
                                  - self.user_bias[ui] for ii in ii_list])
                A = V.T @ V + self.reg * np.eye(self.n_factors)
                b = V.T @ r_vec
                self.user_factors[ui] = np.linalg.solve(A, b)

                # Update user bias
                preds = self.user_factors[ui] @ self.item_factors[ii_list].T
                residuals = np.array([items_rated[ii] for ii in ii_list]) - self.global_bias - self.item_bias[ii_list] - preds
                self.user_bias[ui] = residuals.sum() / (len(ii_list) + self.reg)

            # Fix users, solve for items
            for ii in range(n_items):
                users_rated = matrix._item_users.get(ii, {})
                if not users_rated:
                    continue
                ui_list = list(users_rated.keys())
                U = self.user_factors[ui_list]  # (n_rated, n_factors)
                r_vec = np.array([users_rated[ui] - self.global_bias - self.user_bias[ui]
                                  - self.item_bias[ii] for ui in ui_list])
                A = U.T @ U + self.reg * np.eye(self.n_factors)
                b = U.T @ r_vec
                self.item_factors[ii] = np.linalg.solve(A, b)

                # Update item bias
                preds = self.item_factors[ii] @ self.user_factors[ui_list].T
                residuals = np.array([users_rated[ui] for ui in ui_list]) - self.global_bias - self.user_bias[ui_list] - preds
                self.item_bias[ii] = residuals.sum() / (len(ui_list) + self.reg)

        return self

    def predict(self, user_id, item_id):
        """Predict rating."""
        ui = self._matrix._user_idx.get(user_id)
        ii = self._matrix._item_idx.get(item_id)
        if ui is None or ii is None:
            return self.global_bias
        return (self.global_bias + self.user_bias[ui] + self.item_bias[ii] +
                float(self.user_factors[ui] @ self.item_factors[ii]))

    def recommend(self, user_id, n=10, exclude_rated=True):
        """Recommend top-n items."""
        ui = self._matrix._user_idx.get(user_id)
        if ui is None:
            return []
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        scores = (self.global_bias + self.user_bias[ui] + self.item_bias +
                  self.user_factors[ui] @ self.item_factors.T)
        candidates = []
        for ii, score in enumerate(scores):
            item_id = self._matrix.item_ids[ii]
            if item_id in rated:
                continue
            candidates.append((item_id, float(score)))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]

    def rmse(self):
        """Compute RMSE on training data."""
        errors = []
        for (ui, ii), r in self._matrix._ratings.items():
            pred = (self.global_bias + self.user_bias[ui] + self.item_bias[ii] +
                    float(self.user_factors[ui] @ self.item_factors[ii]))
            errors.append((r - pred) ** 2)
        return math.sqrt(sum(errors) / len(errors)) if errors else 0.0


# ---------------------------------------------------------------------------
# SVDRecommender -- truncated SVD via numpy
# ---------------------------------------------------------------------------

class SVDRecommender:
    """SVD-based recommender using truncated SVD."""

    def __init__(self, n_factors=10):
        self.n_factors = n_factors
        self._matrix = None
        self._dense_filled = None
        self._user_means = None
        self._U = None
        self._sigma = None
        self._Vt = None

    def fit(self, matrix):
        self._matrix = matrix
        dense = matrix.to_dense()
        self._user_means = np.nanmean(dense, axis=1)
        # Fill NaN with user means
        filled = dense.copy()
        for i in range(filled.shape[0]):
            mask = np.isnan(filled[i])
            filled[i, mask] = self._user_means[i]
        # Center
        centered = filled - self._user_means[:, np.newaxis]
        # Full SVD then truncate
        U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self.n_factors, len(sigma))
        self._U = U[:, :k]
        self._sigma = sigma[:k]
        self._Vt = Vt[:k, :]
        self._dense_filled = filled
        return self

    def predict(self, user_id, item_id):
        ui = self._matrix._user_idx.get(user_id)
        ii = self._matrix._item_idx.get(item_id)
        if ui is None or ii is None:
            return self._matrix.global_mean()
        # Reconstruct: U * diag(sigma) * Vt + user_mean
        pred = float(self._U[ui] @ np.diag(self._sigma) @ self._Vt[:, ii]) + self._user_means[ui]
        return pred

    def recommend(self, user_id, n=10, exclude_rated=True):
        ui = self._matrix._user_idx.get(user_id)
        if ui is None:
            return []
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        # Reconstruct all scores for this user
        scores = self._U[ui] @ np.diag(self._sigma) @ self._Vt + self._user_means[ui]
        candidates = []
        for ii, score in enumerate(scores):
            item_id = self._matrix.item_ids[ii]
            if item_id in rated:
                continue
            candidates.append((item_id, float(score)))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# ContentBasedRecommender
# ---------------------------------------------------------------------------

class ContentBasedRecommender:
    """Content-based recommender using item feature vectors."""

    def __init__(self, similarity='cosine'):
        self.similarity = similarity
        self._matrix = None
        self._item_features = {}  # item_id -> np.array feature vector
        self._feature_dim = 0

    def fit(self, matrix, item_features=None):
        """
        item_features: dict of item_id -> feature_vector (list or np.array)
        """
        self._matrix = matrix
        if item_features:
            self._item_features = {}
            for item_id, feat in item_features.items():
                self._item_features[item_id] = np.array(feat, dtype=float)
            if self._item_features:
                self._feature_dim = len(next(iter(self._item_features.values())))
        return self

    def _user_profile(self, user_id):
        """Build user profile as weighted average of item features."""
        ratings = self._matrix.get_user_ratings(user_id)
        if not ratings:
            return np.zeros(self._feature_dim)
        profile = np.zeros(self._feature_dim)
        total_weight = 0.0
        for item_id, rating in ratings.items():
            feat = self._item_features.get(item_id)
            if feat is not None:
                profile += rating * feat
                total_weight += abs(rating)
        if total_weight > 0:
            profile /= total_weight
        return profile

    def predict(self, user_id, item_id):
        """Predict as cosine similarity between user profile and item features."""
        profile = self._user_profile(user_id)
        feat = self._item_features.get(item_id)
        if feat is None:
            return self._matrix.global_mean()
        sim = cosine_similarity(profile, feat)
        # Scale similarity to rating range
        user_mean = self._matrix.user_mean(user_id)
        return user_mean + sim * 2  # rough scaling

    def recommend(self, user_id, n=10, exclude_rated=True):
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        profile = self._user_profile(user_id)
        candidates = []
        for item_id in self._matrix.item_ids:
            if item_id in rated:
                continue
            feat = self._item_features.get(item_id)
            if feat is None:
                continue
            sim = cosine_similarity(profile, feat)
            candidates.append((item_id, sim))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# HybridRecommender
# ---------------------------------------------------------------------------

class HybridRecommender:
    """Weighted hybrid of multiple recommenders."""

    def __init__(self, recommenders=None, weights=None):
        """
        recommenders: list of (name, recommender) tuples
        weights: list of float weights (same length as recommenders)
        """
        self.recommenders = recommenders or []
        self.weights = weights or [1.0] * len(self.recommenders)
        self._matrix = None

    def fit(self, matrix, **kwargs):
        """All sub-recommenders should already be fitted."""
        self._matrix = matrix
        return self

    def predict(self, user_id, item_id):
        total = 0.0
        w_sum = 0.0
        for (name, rec), w in zip(self.recommenders, self.weights):
            pred = rec.predict(user_id, item_id)
            total += w * pred
            w_sum += w
        if w_sum == 0:
            return self._matrix.global_mean() if self._matrix else 0.0
        return total / w_sum

    def recommend(self, user_id, n=10, exclude_rated=True):
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        # Score each candidate item
        candidates = {}
        for item_id in self._matrix.item_ids:
            if item_id in rated:
                continue
            candidates[item_id] = self.predict(user_id, item_id)
        sorted_items = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
        return [item for item, _ in sorted_items[:n]]


# ---------------------------------------------------------------------------
# NMF Recommender -- Non-negative Matrix Factorization
# ---------------------------------------------------------------------------

class NMFRecommender:
    """Non-negative Matrix Factorization using multiplicative update rules."""

    def __init__(self, n_factors=10, n_iterations=50, reg=0.01):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self._matrix = None
        self.W = None  # (n_users, n_factors)
        self.H = None  # (n_factors, n_items)
        self.global_bias = 0.0

    def fit(self, matrix):
        self._matrix = matrix
        n_users = matrix.n_users()
        n_items = matrix.n_items()
        self.global_bias = matrix.global_mean()

        dense = matrix.to_dense()
        # Replace NaN with 0 for NMF and create mask
        mask = ~np.isnan(dense)
        R = np.where(mask, dense, 0.0)
        R = np.maximum(R, 0)  # Ensure non-negative

        rng = np.random.RandomState(42)
        self.W = rng.uniform(0.01, 1.0, (n_users, self.n_factors))
        self.H = rng.uniform(0.01, 1.0, (self.n_factors, n_items))

        for _ in range(self.n_iterations):
            # Update W
            WH = self.W @ self.H
            numerator = (R * mask) @ self.H.T
            denominator = (WH * mask) @ self.H.T + self.reg * self.W + 1e-10
            self.W *= numerator / denominator

            # Update H
            WH = self.W @ self.H
            numerator = self.W.T @ (R * mask)
            denominator = self.W.T @ (WH * mask) + self.reg * self.H + 1e-10
            self.H *= numerator / denominator

        return self

    def predict(self, user_id, item_id):
        ui = self._matrix._user_idx.get(user_id)
        ii = self._matrix._item_idx.get(item_id)
        if ui is None or ii is None:
            return self.global_bias
        return float(self.W[ui] @ self.H[:, ii])

    def recommend(self, user_id, n=10, exclude_rated=True):
        ui = self._matrix._user_idx.get(user_id)
        if ui is None:
            return []
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        scores = self.W[ui] @ self.H
        candidates = []
        for ii, score in enumerate(scores):
            item_id = self._matrix.item_ids[ii]
            if item_id in rated:
                continue
            candidates.append((item_id, float(score)))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# KNNBaseline -- baseline-adjusted kNN
# ---------------------------------------------------------------------------

class KNNBaseline:
    """KNN with baseline estimates (global mean + user bias + item bias)."""

    def __init__(self, k=20, user_based=True, reg_u=10, reg_i=25):
        self.k = k
        self.user_based = user_based
        self.reg_u = reg_u
        self.reg_i = reg_i
        self._matrix = None
        self._dense = None
        self._global_mean = 0.0
        self._user_bias = None
        self._item_bias = None

    def fit(self, matrix):
        self._matrix = matrix
        self._dense = matrix.to_dense()
        self._global_mean = matrix.global_mean()

        n_users = matrix.n_users()
        n_items = matrix.n_items()

        # Compute biases with regularization
        self._user_bias = np.zeros(n_users)
        self._item_bias = np.zeros(n_items)

        for _ in range(10):  # iterate to convergence
            for ii in range(n_items):
                users = matrix._item_users.get(ii, {})
                if users:
                    residuals = [r - self._global_mean - self._user_bias[ui] for ui, r in users.items()]
                    self._item_bias[ii] = sum(residuals) / (len(residuals) + self.reg_i)

            for ui in range(n_users):
                items = matrix._user_items.get(ui, {})
                if items:
                    residuals = [r - self._global_mean - self._item_bias[ii] for ii, r in items.items()]
                    self._user_bias[ui] = sum(residuals) / (len(residuals) + self.reg_u)

        return self

    def _baseline(self, ui, ii):
        return self._global_mean + self._user_bias[ui] + self._item_bias[ii]

    def predict(self, user_id, item_id):
        ui = self._matrix._user_idx.get(user_id)
        ii = self._matrix._item_idx.get(item_id)
        if ui is None or ii is None:
            return self._global_mean

        bui = self._baseline(ui, ii)

        if self.user_based:
            # Find similar users who rated this item
            neighbors = []
            for other_ui in self._matrix._item_users.get(ii, {}):
                if other_ui == ui:
                    continue
                sim = pearson_similarity(self._dense[ui], self._dense[other_ui])
                if sim != 0:
                    neighbors.append((other_ui, sim))
            neighbors.sort(key=lambda x: -abs(x[1]))
            neighbors = neighbors[:self.k]

            if not neighbors:
                return bui

            num = 0.0
            denom = 0.0
            for other_ui, sim in neighbors:
                b_other = self._baseline(other_ui, ii)
                r_other = self._matrix._ratings.get((other_ui, ii), 0)
                num += sim * (r_other - b_other)
                denom += abs(sim)
            if denom == 0:
                return bui
            return bui + num / denom
        else:
            # Item-based
            user_items = self._matrix._user_items.get(ui, {})
            neighbors = []
            for other_ii, r in user_items.items():
                if other_ii == ii:
                    continue
                sim = pearson_similarity(self._dense[:, ii], self._dense[:, other_ii])
                if sim != 0:
                    neighbors.append((other_ii, sim, r))
            neighbors.sort(key=lambda x: -abs(x[1]))
            neighbors = neighbors[:self.k]

            if not neighbors:
                return bui

            num = 0.0
            denom = 0.0
            for other_ii, sim, r in neighbors:
                b_other = self._baseline(ui, other_ii)
                num += sim * (r - b_other)
                denom += abs(sim)
            if denom == 0:
                return bui
            return bui + num / denom

    def recommend(self, user_id, n=10, exclude_rated=True):
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        candidates = []
        for item_id in self._matrix.item_ids:
            if item_id in rated:
                continue
            score = self.predict(user_id, item_id)
            candidates.append((item_id, score))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# ImplicitFeedback -- for binary interaction data (views, clicks)
# ---------------------------------------------------------------------------

class ImplicitALS:
    """ALS for implicit feedback (Hu, Koren, Volinsky 2008)."""

    def __init__(self, n_factors=10, n_iterations=15, reg=0.1, alpha=40):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.alpha = alpha  # confidence scaling
        self._matrix = None
        self.user_factors = None
        self.item_factors = None

    def fit(self, matrix):
        self._matrix = matrix
        n_users = matrix.n_users()
        n_items = matrix.n_items()

        rng = np.random.RandomState(42)
        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))

        # Build preference and confidence matrices (sparse)
        # p_ui = 1 if r_ui > 0, c_ui = 1 + alpha * r_ui
        for iteration in range(self.n_iterations):
            # Update users
            YtY = self.item_factors.T @ self.item_factors
            for ui in range(n_users):
                items = matrix._user_items.get(ui, {})
                A = YtY + self.reg * np.eye(self.n_factors)
                b = np.zeros(self.n_factors)
                for ii, r in items.items():
                    c = 1 + self.alpha * r
                    y = self.item_factors[ii]
                    A += (c - 1) * np.outer(y, y)
                    b += c * y  # p_ui = 1 for observed
                self.user_factors[ui] = np.linalg.solve(A, b)

            # Update items
            XtX = self.user_factors.T @ self.user_factors
            for ii in range(n_items):
                users = matrix._item_users.get(ii, {})
                A = XtX + self.reg * np.eye(self.n_factors)
                b = np.zeros(self.n_factors)
                for ui, r in users.items():
                    c = 1 + self.alpha * r
                    x = self.user_factors[ui]
                    A += (c - 1) * np.outer(x, x)
                    b += c * x
                self.item_factors[ii] = np.linalg.solve(A, b)

        return self

    def predict(self, user_id, item_id):
        """Predict preference score (not rating)."""
        ui = self._matrix._user_idx.get(user_id)
        ii = self._matrix._item_idx.get(item_id)
        if ui is None or ii is None:
            return 0.0
        return float(self.user_factors[ui] @ self.item_factors[ii])

    def recommend(self, user_id, n=10, exclude_rated=True):
        ui = self._matrix._user_idx.get(user_id)
        if ui is None:
            return []
        rated = self._matrix.get_items_for_user(user_id) if exclude_rated else set()
        scores = self.user_factors[ui] @ self.item_factors.T
        candidates = []
        for ii, score in enumerate(scores):
            item_id = self._matrix.item_ids[ii]
            if item_id in rated:
                continue
            candidates.append((item_id, float(score)))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item for item, _ in candidates[:n]]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluate recommender systems with standard metrics."""

    @staticmethod
    def rmse(recommender, test_ratings):
        """RMSE on (user_id, item_id, rating) triples."""
        errors = []
        for user_id, item_id, actual in test_ratings:
            pred = recommender.predict(user_id, item_id)
            errors.append((actual - pred) ** 2)
        if not errors:
            return 0.0
        return math.sqrt(sum(errors) / len(errors))

    @staticmethod
    def mae(recommender, test_ratings):
        """MAE on (user_id, item_id, rating) triples."""
        errors = []
        for user_id, item_id, actual in test_ratings:
            pred = recommender.predict(user_id, item_id)
            errors.append(abs(actual - pred))
        if not errors:
            return 0.0
        return sum(errors) / len(errors)

    @staticmethod
    def precision_at_k(recommended, relevant, k=None):
        """Precision@k: fraction of recommended items that are relevant."""
        if k is not None:
            recommended = recommended[:k]
        if not recommended:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for item in recommended if item in relevant_set)
        return hits / len(recommended)

    @staticmethod
    def recall_at_k(recommended, relevant, k=None):
        """Recall@k: fraction of relevant items that are recommended."""
        if k is not None:
            recommended = recommended[:k]
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        hits = sum(1 for item in recommended if item in relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def f1_at_k(recommended, relevant, k=None):
        """F1@k: harmonic mean of precision and recall."""
        p = Evaluator.precision_at_k(recommended, relevant, k)
        r = Evaluator.recall_at_k(recommended, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def ndcg_at_k(recommended, relevant, k=None):
        """Normalized Discounted Cumulative Gain@k."""
        if k is not None:
            recommended = recommended[:k]
        if not relevant or not recommended:
            return 0.0
        relevant_set = set(relevant)

        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended):
            if item in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # position 1-indexed

        # Ideal DCG
        ideal_hits = min(len(relevant_set), len(recommended))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def average_precision(recommended, relevant):
        """Average Precision for a single user."""
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        hits = 0
        sum_precision = 0.0
        for i, item in enumerate(recommended):
            if item in relevant_set:
                hits += 1
                sum_precision += hits / (i + 1)
        if not relevant_set:
            return 0.0
        return sum_precision / len(relevant_set)

    @staticmethod
    def mean_average_precision(all_recommended, all_relevant):
        """MAP across all users.
        all_recommended: list of recommended lists
        all_relevant: list of relevant lists
        """
        if not all_recommended:
            return 0.0
        aps = [Evaluator.average_precision(rec, rel)
               for rec, rel in zip(all_recommended, all_relevant)]
        return sum(aps) / len(aps)

    @staticmethod
    def coverage(recommender, matrix, n=10):
        """Catalog coverage: fraction of items that appear in any recommendation."""
        all_items = set(matrix.item_ids)
        recommended_items = set()
        for user_id in matrix.user_ids:
            recs = recommender.recommend(user_id, n=n)
            recommended_items.update(recs)
        if not all_items:
            return 0.0
        return len(recommended_items) / len(all_items)

    @staticmethod
    def diversity(recommended, item_features):
        """Intra-list diversity: average pairwise distance of recommended items."""
        if len(recommended) < 2:
            return 0.0
        sims = []
        for i in range(len(recommended)):
            feat_i = item_features.get(recommended[i])
            if feat_i is None:
                continue
            for j in range(i + 1, len(recommended)):
                feat_j = item_features.get(recommended[j])
                if feat_j is None:
                    continue
                sims.append(1 - cosine_similarity(np.array(feat_i), np.array(feat_j)))
        if not sims:
            return 0.0
        return sum(sims) / len(sims)

    @staticmethod
    def novelty(recommended, item_popularity):
        """Average self-information of recommended items.
        item_popularity: dict of item_id -> popularity (0-1)
        """
        if not recommended:
            return 0.0
        scores = []
        for item in recommended:
            pop = item_popularity.get(item, 0.001)
            pop = max(pop, 0.001)  # avoid log(0)
            scores.append(-math.log2(pop))
        return sum(scores) / len(scores)

    @staticmethod
    def hit_rate(recommended, relevant):
        """1 if any relevant item is in recommended, else 0."""
        relevant_set = set(relevant)
        for item in recommended:
            if item in relevant_set:
                return 1.0
        return 0.0

    @staticmethod
    def mrr(recommended, relevant):
        """Mean Reciprocal Rank: 1/rank of first relevant item."""
        relevant_set = set(relevant)
        for i, item in enumerate(recommended):
            if item in relevant_set:
                return 1.0 / (i + 1)
        return 0.0


# ---------------------------------------------------------------------------
# Train/Test Split utility
# ---------------------------------------------------------------------------

def train_test_split(matrix, test_ratio=0.2, seed=42):
    """Split a UserItemMatrix into train and test sets.
    Returns (train_matrix, test_ratings) where test_ratings is list of (user, item, rating).
    """
    rng = np.random.RandomState(seed)
    train = UserItemMatrix()
    test = []

    all_ratings = list(matrix._ratings.items())
    rng.shuffle(all_ratings)

    n_test = int(len(all_ratings) * test_ratio)
    test_set = set()

    # Ensure each user has at least one rating in train
    user_counts = defaultdict(int)
    for (ui, ii), r in all_ratings:
        user_counts[ui] += 1

    assigned_test = 0
    for (ui, ii), r in all_ratings:
        user_id = matrix.user_ids[ui]
        item_id = matrix.item_ids[ii]
        if assigned_test < n_test and user_counts[ui] > 1:
            test.append((user_id, item_id, r))
            user_counts[ui] -= 1
            assigned_test += 1
        else:
            train.add_rating(user_id, item_id, r)

    return train, test
