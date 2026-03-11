"""
C165: Causal Effect Estimation
Extends C161 (Causal Inference) + composes C163 (SEM)

Components:
1. PropensityScoreModel -- Logistic regression for P(T=1|X)
2. IPWEstimator -- Inverse Probability Weighting (Horvitz-Thompson, Hajek)
3. OutcomeModel -- Regression adjustment for potential outcomes
4. DoublyRobustEstimator -- AIPW combining propensity + outcome models
5. MatchingEstimator -- Propensity score matching (nearest neighbor, caliper)
6. StratificationEstimator -- Subclassification on propensity scores
7. TreatmentEffectAnalyzer -- High-level ATE/ATT/ATU/CATE + bootstrap CIs
"""

import math
import numpy as np
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C161_causal_inference'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C163_structural_equation_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C160_probabilistic_graphical_models'))

from causal_inference import CausalGraph, BackdoorCriterion, CausalUtils
from sem import LinearSEM, SEMEstimation


# ---------------------------------------------------------------------------
# 1. PropensityScoreModel
# ---------------------------------------------------------------------------

class PropensityScoreModel:
    """Logistic regression model for estimating propensity scores P(T=1|X)."""

    def __init__(self, max_iter=100, lr=0.1, regularization=0.0):
        self.max_iter = max_iter
        self.lr = lr
        self.regularization = regularization
        self.weights = None
        self.bias = 0.0
        self._fitted = False

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, T):
        """Fit logistic regression. X: (n, d) covariates, T: (n,) binary treatment."""
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0

        for _ in range(self.max_iter):
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)
            error = p - T
            grad_w = (X.T @ error) / n + self.regularization * self.weights
            grad_b = error.mean()
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        self._fitted = True
        return self

    def predict_proba(self, X):
        """Return P(T=1|X) for each sample."""
        X = np.asarray(X, dtype=float)
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Binary prediction."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @staticmethod
    def from_data(data, treatment, covariates, **kwargs):
        """Convenience: fit from list-of-dicts data format."""
        n = len(data)
        d = len(covariates)
        X = np.zeros((n, d))
        T = np.zeros(n)
        for i, row in enumerate(data):
            for j, c in enumerate(covariates):
                X[i, j] = row[c]
            T[i] = row[treatment]
        model = PropensityScoreModel(**kwargs)
        model.fit(X, T)
        model._covariates = covariates
        model._treatment = treatment
        return model

    def scores_from_data(self, data):
        """Get propensity scores from list-of-dicts using stored covariate names."""
        n = len(data)
        d = len(self._covariates)
        X = np.zeros((n, d))
        for i, row in enumerate(data):
            for j, c in enumerate(self._covariates):
                X[i, j] = row[c]
        return self.predict_proba(X)


# ---------------------------------------------------------------------------
# 2. IPWEstimator
# ---------------------------------------------------------------------------

class IPWEstimator:
    """Inverse Probability Weighting estimator for causal effects."""

    @staticmethod
    def horvitz_thompson(Y, T, ps, estimand='ate'):
        """
        Horvitz-Thompson IPW estimator.
        Y: outcomes, T: treatment (0/1), ps: propensity scores
        estimand: 'ate', 'att', or 'atu'
        """
        Y = np.asarray(Y, dtype=float)
        T = np.asarray(T, dtype=float)
        ps = np.asarray(ps, dtype=float)
        ps = np.clip(ps, 0.01, 0.99)
        n = len(Y)

        if estimand == 'ate':
            # E[Y(1)] - E[Y(0)]
            ey1 = np.sum(T * Y / ps) / n
            ey0 = np.sum((1 - T) * Y / (1 - ps)) / n
            return ey1 - ey0

        elif estimand == 'att':
            # ATT: E[Y(1) - Y(0) | T=1]
            n1 = T.sum()
            if n1 == 0:
                return 0.0
            ey1_t = np.sum(T * Y) / n1
            # Weight controls by odds: ps/(1-ps) for treated
            w = ps / (1 - ps)
            ey0_t = np.sum((1 - T) * Y * w) / np.sum((1 - T) * w)
            return ey1_t - ey0_t

        elif estimand == 'atu':
            # ATU: E[Y(1) - Y(0) | T=0]
            n0 = (1 - T).sum()
            if n0 == 0:
                return 0.0
            ey0_c = np.sum((1 - T) * Y) / n0
            w = (1 - ps) / ps
            ey1_c = np.sum(T * Y * w) / np.sum(T * w)
            return ey1_c - ey0_c

        else:
            raise ValueError(f"Unknown estimand: {estimand}")

    @staticmethod
    def hajek(Y, T, ps, estimand='ate'):
        """
        Hajek (normalized) IPW estimator -- more stable than Horvitz-Thompson.
        """
        Y = np.asarray(Y, dtype=float)
        T = np.asarray(T, dtype=float)
        ps = np.asarray(ps, dtype=float)
        ps = np.clip(ps, 0.01, 0.99)

        if estimand == 'ate':
            w1 = T / ps
            w0 = (1 - T) / (1 - ps)
            ey1 = np.sum(w1 * Y) / np.sum(w1)
            ey0 = np.sum(w0 * Y) / np.sum(w0)
            return ey1 - ey0

        elif estimand == 'att':
            n1 = T.sum()
            if n1 == 0:
                return 0.0
            ey1 = np.sum(T * Y) / n1
            w = ps / (1 - ps)
            ey0 = np.sum((1 - T) * Y * w) / np.sum((1 - T) * w)
            return ey1 - ey0

        elif estimand == 'atu':
            n0 = (1 - T).sum()
            if n0 == 0:
                return 0.0
            ey0 = np.sum((1 - T) * Y) / n0
            w = (1 - ps) / ps
            ey1 = np.sum(T * Y * w) / np.sum(T * w)
            return ey1 - ey0

        else:
            raise ValueError(f"Unknown estimand: {estimand}")

    @staticmethod
    def from_data(data, treatment, outcome, covariates, method='hajek',
                  estimand='ate', ps_model=None, **ps_kwargs):
        """Convenience: estimate effect from list-of-dicts."""
        n = len(data)
        Y = np.array([row[outcome] for row in data], dtype=float)
        T = np.array([row[treatment] for row in data], dtype=float)

        if ps_model is None:
            ps_model = PropensityScoreModel.from_data(
                data, treatment, covariates, **ps_kwargs)
        ps = ps_model.scores_from_data(data)

        if method == 'hajek':
            return IPWEstimator.hajek(Y, T, ps, estimand)
        else:
            return IPWEstimator.horvitz_thompson(Y, T, ps, estimand)


# ---------------------------------------------------------------------------
# 3. OutcomeModel
# ---------------------------------------------------------------------------

class OutcomeModel:
    """Linear regression outcome model for potential outcomes E[Y|X,T]."""

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self._fitted = False

    def fit(self, X, T, Y):
        """Fit E[Y|X,T] via OLS. X: (n,d), T: (n,), Y: (n,)."""
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float).reshape(-1, 1)
        Y = np.asarray(Y, dtype=float)

        # Design matrix: [X, T]
        Z = np.hstack([X, T])
        n, p = Z.shape

        # OLS: (Z'Z)^{-1} Z'Y
        ZtZ = Z.T @ Z + 1e-8 * np.eye(p)  # Ridge for stability
        ZtY = Z.T @ Y
        # Add intercept via centering
        Z_mean = Z.mean(axis=0)
        Y_mean = Y.mean()
        Zc = Z - Z_mean
        ZcZc = Zc.T @ Zc + 1e-8 * np.eye(p)
        ZcY = Zc.T @ (Y - Y_mean)
        self.weights = np.linalg.solve(ZcZc, ZcY)
        self.bias = Y_mean - Z_mean @ self.weights
        self._fitted = True
        return self

    def predict(self, X, T):
        """Predict E[Y|X,T]."""
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float).reshape(-1, 1)
        Z = np.hstack([X, T])
        return Z @ self.weights + self.bias

    def ate(self, X):
        """Compute ATE via regression adjustment: E[mu(X,1) - mu(X,0)]."""
        n = X.shape[0]
        mu1 = self.predict(X, np.ones(n))
        mu0 = self.predict(X, np.zeros(n))
        return np.mean(mu1 - mu0)

    def att(self, X, T):
        """ATT: E[mu(X,1) - mu(X,0) | T=1]."""
        T = np.asarray(T, dtype=float)
        treated = T == 1
        if not np.any(treated):
            return 0.0
        X_t = X[treated]
        n_t = X_t.shape[0]
        mu1 = self.predict(X_t, np.ones(n_t))
        mu0 = self.predict(X_t, np.zeros(n_t))
        return np.mean(mu1 - mu0)

    @staticmethod
    def from_data(data, treatment, outcome, covariates):
        """Convenience: fit from list-of-dicts."""
        n = len(data)
        d = len(covariates)
        X = np.zeros((n, d))
        T = np.zeros(n)
        Y = np.zeros(n)
        for i, row in enumerate(data):
            for j, c in enumerate(covariates):
                X[i, j] = row[c]
            T[i] = row[treatment]
            Y[i] = row[outcome]
        model = OutcomeModel()
        model.fit(X, T, Y)
        model._covariates = covariates
        return model, X, T, Y


# ---------------------------------------------------------------------------
# 4. DoublyRobustEstimator
# ---------------------------------------------------------------------------

class DoublyRobustEstimator:
    """
    Augmented IPW (AIPW / Doubly Robust) estimator.
    Consistent if EITHER propensity OR outcome model is correct.
    """

    @staticmethod
    def estimate(Y, T, ps, mu0, mu1, estimand='ate'):
        """
        AIPW estimator.
        Y: outcomes, T: treatment, ps: propensity scores
        mu0: E[Y|X,T=0] predictions, mu1: E[Y|X,T=1] predictions
        """
        Y = np.asarray(Y, dtype=float)
        T = np.asarray(T, dtype=float)
        ps = np.clip(np.asarray(ps, dtype=float), 0.01, 0.99)
        mu0 = np.asarray(mu0, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        n = len(Y)

        if estimand == 'ate':
            # AIPW: E[mu1(X) + T(Y-mu1(X))/ps - mu0(X) - (1-T)(Y-mu0(X))/(1-ps)]
            phi1 = mu1 + T * (Y - mu1) / ps
            phi0 = mu0 + (1 - T) * (Y - mu0) / (1 - ps)
            return np.mean(phi1 - phi0)

        elif estimand == 'att':
            n1 = T.sum()
            if n1 == 0:
                return 0.0
            # ATT AIPW
            w = ps / (1 - ps)
            phi1 = T * Y
            phi0 = T * mu0 + (1 - T) * w * (Y - mu0)
            return (phi1.sum() - phi0.sum()) / n1

        elif estimand == 'atu':
            n0 = (1 - T).sum()
            if n0 == 0:
                return 0.0
            w = (1 - ps) / ps
            phi1 = (1 - T) * mu1 + T * w * (Y - mu1)
            phi0 = (1 - T) * Y
            return (phi1.sum() - phi0.sum()) / n0

        else:
            raise ValueError(f"Unknown estimand: {estimand}")

    @staticmethod
    def from_data(data, treatment, outcome, covariates, estimand='ate',
                  ps_model=None, outcome_model=None, **ps_kwargs):
        """Convenience: full AIPW from list-of-dicts."""
        n = len(data)
        d = len(covariates)
        X = np.zeros((n, d))
        T = np.zeros(n)
        Y = np.zeros(n)
        for i, row in enumerate(data):
            for j, c in enumerate(covariates):
                X[i, j] = row[c]
            T[i] = row[treatment]
            Y[i] = row[outcome]

        # Propensity model
        if ps_model is None:
            ps_model = PropensityScoreModel(**ps_kwargs)
            ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)

        # Outcome model
        if outcome_model is None:
            outcome_model = OutcomeModel()
            outcome_model.fit(X, T, Y)
        mu0 = outcome_model.predict(X, np.zeros(n))
        mu1 = outcome_model.predict(X, np.ones(n))

        return DoublyRobustEstimator.estimate(Y, T, ps, mu0, mu1, estimand)


# ---------------------------------------------------------------------------
# 5. MatchingEstimator
# ---------------------------------------------------------------------------

class MatchingEstimator:
    """Propensity score matching estimator."""

    @staticmethod
    def nearest_neighbor(Y, T, ps, n_matches=1, caliper=None, estimand='ate',
                         with_replacement=True):
        """
        Nearest-neighbor matching on propensity scores.
        Returns estimated treatment effect.
        """
        Y = np.asarray(Y, dtype=float)
        T = np.asarray(T, dtype=float)
        ps = np.asarray(ps, dtype=float)
        n = len(Y)

        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            return 0.0

        def _match(source_idx, pool_idx, pool_ps):
            """Match each source unit to nearest pool unit(s)."""
            matches = []
            used = set()
            for si in source_idx:
                dists = np.abs(ps[si] - pool_ps)
                sorted_pool = np.argsort(dists)
                matched = []
                for pi_local in sorted_pool:
                    pi = pool_idx[pi_local]
                    if not with_replacement and pi in used:
                        continue
                    if caliper is not None and dists[pi_local] > caliper:
                        break
                    matched.append(pi)
                    used.add(pi)
                    if len(matched) >= n_matches:
                        break
                if matched:
                    matches.append((si, matched))
            return matches

        if estimand in ('ate', 'att'):
            # Match treated to controls
            matches_tc = _match(treated_idx, control_idx, ps[control_idx])
            att_effects = []
            for ti, cis in matches_tc:
                y_c = np.mean(Y[cis])
                att_effects.append(Y[ti] - y_c)

            if estimand == 'att':
                return np.mean(att_effects) if att_effects else 0.0

        if estimand in ('ate', 'atu'):
            # Match controls to treated
            matches_ct = _match(control_idx, treated_idx, ps[treated_idx])
            atu_effects = []
            for ci, tis in matches_ct:
                y_t = np.mean(Y[tis])
                atu_effects.append(y_t - Y[ci])

            if estimand == 'atu':
                return np.mean(atu_effects) if atu_effects else 0.0

        # ATE: weighted average of ATT and ATU
        n1 = len(treated_idx)
        n0 = len(control_idx)
        att = np.mean(att_effects) if att_effects else 0.0
        atu = np.mean(atu_effects) if atu_effects else 0.0
        return (n1 * att + n0 * atu) / n

    @staticmethod
    def from_data(data, treatment, outcome, covariates, n_matches=1,
                  caliper=None, estimand='ate', ps_model=None, **ps_kwargs):
        """Convenience: matching from list-of-dicts."""
        n = len(data)
        Y = np.array([row[outcome] for row in data], dtype=float)
        T = np.array([row[treatment] for row in data], dtype=float)

        if ps_model is None:
            ps_model = PropensityScoreModel.from_data(
                data, treatment, covariates, **ps_kwargs)
        ps = ps_model.scores_from_data(data)

        return MatchingEstimator.nearest_neighbor(
            Y, T, ps, n_matches=n_matches, caliper=caliper, estimand=estimand)


# ---------------------------------------------------------------------------
# 6. StratificationEstimator
# ---------------------------------------------------------------------------

class StratificationEstimator:
    """Propensity score stratification (subclassification) estimator."""

    @staticmethod
    def estimate(Y, T, ps, n_strata=5, estimand='ate'):
        """
        Stratify on propensity scores and compute within-stratum effects.
        """
        Y = np.asarray(Y, dtype=float)
        T = np.asarray(T, dtype=float)
        ps = np.asarray(ps, dtype=float)
        n = len(Y)

        # Create strata based on propensity score quantiles
        boundaries = np.linspace(0, 1, n_strata + 1)
        # Use percentile-based boundaries for better balance
        percentiles = np.linspace(0, 100, n_strata + 1)
        boundaries = np.percentile(ps, percentiles)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

        stratum_effects = []
        stratum_sizes = []

        for s in range(n_strata):
            mask = (ps >= boundaries[s]) & (ps < boundaries[s + 1])
            if s == n_strata - 1:
                mask = (ps >= boundaries[s]) & (ps <= boundaries[s + 1])

            Y_s = Y[mask]
            T_s = T[mask]

            treated = T_s == 1
            control = T_s == 0

            if np.sum(treated) == 0 or np.sum(control) == 0:
                continue

            ey1 = Y_s[treated].mean()
            ey0 = Y_s[control].mean()
            effect = ey1 - ey0

            if estimand == 'ate':
                stratum_sizes.append(np.sum(mask))
            elif estimand == 'att':
                stratum_sizes.append(np.sum(mask & (T == 1)))
            elif estimand == 'atu':
                stratum_sizes.append(np.sum(mask & (T == 0)))
            else:
                raise ValueError(f"Unknown estimand: {estimand}")

            stratum_effects.append(effect)

        if not stratum_effects:
            return 0.0

        sizes = np.array(stratum_sizes, dtype=float)
        effects = np.array(stratum_effects)
        return np.sum(sizes * effects) / np.sum(sizes)

    @staticmethod
    def from_data(data, treatment, outcome, covariates, n_strata=5,
                  estimand='ate', ps_model=None, **ps_kwargs):
        """Convenience: stratification from list-of-dicts."""
        n = len(data)
        Y = np.array([row[outcome] for row in data], dtype=float)
        T = np.array([row[treatment] for row in data], dtype=float)

        if ps_model is None:
            ps_model = PropensityScoreModel.from_data(
                data, treatment, covariates, **ps_kwargs)
        ps = ps_model.scores_from_data(data)

        return StratificationEstimator.estimate(
            Y, T, ps, n_strata=n_strata, estimand=estimand)


# ---------------------------------------------------------------------------
# 7. TreatmentEffectAnalyzer
# ---------------------------------------------------------------------------

class TreatmentEffectAnalyzer:
    """High-level analyzer combining all estimation methods with bootstrap CIs."""

    def __init__(self, data, treatment, outcome, covariates):
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = covariates
        self._n = len(data)

        # Extract arrays
        d = len(covariates)
        self.X = np.zeros((self._n, d))
        self.T = np.zeros(self._n)
        self.Y = np.zeros(self._n)
        for i, row in enumerate(data):
            for j, c in enumerate(covariates):
                self.X[i, j] = row[c]
            self.T[i] = row[treatment]
            self.Y[i] = row[outcome]

        # Fit models
        self.ps_model = PropensityScoreModel(max_iter=200, lr=0.1)
        self.ps_model.fit(self.X, self.T)
        self.ps = self.ps_model.predict_proba(self.X)

        self.outcome_model = OutcomeModel()
        self.outcome_model.fit(self.X, self.T, self.Y)

    def naive_difference(self):
        """Unadjusted difference in means (biased if confounders present)."""
        treated = self.T == 1
        if not np.any(treated) or not np.any(~treated):
            return 0.0
        return self.Y[treated].mean() - self.Y[~treated].mean()

    def ipw(self, method='hajek', estimand='ate'):
        """IPW estimate."""
        if method == 'hajek':
            return IPWEstimator.hajek(self.Y, self.T, self.ps, estimand)
        return IPWEstimator.horvitz_thompson(self.Y, self.T, self.ps, estimand)

    def regression_adjustment(self, estimand='ate'):
        """Outcome model regression adjustment."""
        if estimand == 'ate':
            return self.outcome_model.ate(self.X)
        elif estimand == 'att':
            return self.outcome_model.att(self.X, self.T)
        elif estimand == 'atu':
            # ATU: E[mu(1,X) - mu(0,X) | T=0]
            control = self.T == 0
            if not np.any(control):
                return 0.0
            X_c = self.X[control]
            n_c = X_c.shape[0]
            mu1 = self.outcome_model.predict(X_c, np.ones(n_c))
            mu0 = self.outcome_model.predict(X_c, np.zeros(n_c))
            return np.mean(mu1 - mu0)
        raise ValueError(f"Unknown estimand: {estimand}")

    def doubly_robust(self, estimand='ate'):
        """AIPW estimate."""
        mu0 = self.outcome_model.predict(self.X, np.zeros(self._n))
        mu1 = self.outcome_model.predict(self.X, np.ones(self._n))
        return DoublyRobustEstimator.estimate(
            self.Y, self.T, self.ps, mu0, mu1, estimand)

    def matching(self, n_matches=1, caliper=None, estimand='ate'):
        """Matching estimate."""
        return MatchingEstimator.nearest_neighbor(
            self.Y, self.T, self.ps, n_matches=n_matches,
            caliper=caliper, estimand=estimand)

    def stratification(self, n_strata=5, estimand='ate'):
        """Stratification estimate."""
        return StratificationEstimator.estimate(
            self.Y, self.T, self.ps, n_strata=n_strata, estimand=estimand)

    def cate(self, subgroup_var, subgroup_val, method='doubly_robust'):
        """
        Conditional ATE for a subgroup defined by subgroup_var == subgroup_val.
        """
        sub_data = [row for row in self.data if row[subgroup_var] == subgroup_val]
        if len(sub_data) < 4:
            return None
        sub_analyzer = TreatmentEffectAnalyzer(
            sub_data, self.treatment, self.outcome, self.covariates)
        if method == 'doubly_robust':
            return sub_analyzer.doubly_robust()
        elif method == 'ipw':
            return sub_analyzer.ipw()
        elif method == 'regression':
            return sub_analyzer.regression_adjustment()
        elif method == 'matching':
            return sub_analyzer.matching()
        return sub_analyzer.doubly_robust()

    def bootstrap_ci(self, method='doubly_robust', estimand='ate',
                     n_bootstrap=200, confidence=0.95, seed=None):
        """
        Bootstrap confidence interval for any estimator.
        Returns (estimate, lower, upper, se).
        """
        rng = np.random.RandomState(seed)
        estimates = []

        for _ in range(n_bootstrap):
            idx = rng.choice(self._n, size=self._n, replace=True)
            boot_data = [self.data[i] for i in idx]
            try:
                ba = TreatmentEffectAnalyzer(
                    boot_data, self.treatment, self.outcome, self.covariates)
                if method == 'doubly_robust':
                    est = ba.doubly_robust(estimand)
                elif method == 'ipw':
                    est = ba.ipw(estimand=estimand)
                elif method == 'regression':
                    est = ba.regression_adjustment(estimand)
                elif method == 'matching':
                    est = ba.matching(estimand=estimand)
                elif method == 'stratification':
                    est = ba.stratification(estimand=estimand)
                else:
                    est = ba.doubly_robust(estimand)
                estimates.append(est)
            except Exception:
                continue

        if not estimates:
            return (0.0, 0.0, 0.0, 0.0)

        estimates = np.array(estimates)
        alpha = (1 - confidence) / 2
        lower = np.percentile(estimates, 100 * alpha)
        upper = np.percentile(estimates, 100 * (1 - alpha))
        if method == 'doubly_robust':
            point_est = self.doubly_robust(estimand)
        elif method == 'ipw':
            point_est = self.ipw(estimand=estimand)
        elif method == 'regression':
            point_est = self.regression_adjustment(estimand)
        elif method == 'matching':
            point_est = self.matching(estimand=estimand)
        elif method == 'stratification':
            point_est = self.stratification(estimand=estimand)
        else:
            point_est = self.doubly_robust(estimand)

        se = np.std(estimates, ddof=1)
        return (point_est, lower, upper, se)

    def compare_methods(self, estimand='ate'):
        """Compare all estimation methods side-by-side."""
        results = {}
        results['naive'] = self.naive_difference()
        results['ipw_ht'] = IPWEstimator.horvitz_thompson(
            self.Y, self.T, self.ps, estimand)
        results['ipw_hajek'] = self.ipw('hajek', estimand)
        results['regression'] = self.regression_adjustment(estimand)
        results['doubly_robust'] = self.doubly_robust(estimand)
        results['matching'] = self.matching(estimand=estimand)
        results['stratification'] = self.stratification(estimand=estimand)
        return results

    def sensitivity_analysis(self, gamma_values=None):
        """
        Rosenbaum-style sensitivity analysis.
        How much unmeasured confounding (gamma) would change conclusions?
        Returns list of (gamma, lower_bound, upper_bound).
        """
        if gamma_values is None:
            gamma_values = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

        results = []
        base_ate = self.doubly_robust()

        for gamma in gamma_values:
            # Under Rosenbaum bounds, with confounding strength gamma,
            # propensity scores could be inflated/deflated
            lower_ates = []
            upper_ates = []

            # Adjust propensity scores for sensitivity
            ps_lower = self.ps / (self.ps + gamma * (1 - self.ps))
            ps_upper = gamma * self.ps / (gamma * self.ps + (1 - self.ps))
            ps_lower = np.clip(ps_lower, 0.01, 0.99)
            ps_upper = np.clip(ps_upper, 0.01, 0.99)

            ate_lo = IPWEstimator.hajek(self.Y, self.T, ps_lower, 'ate')
            ate_hi = IPWEstimator.hajek(self.Y, self.T, ps_upper, 'ate')

            results.append((gamma, min(ate_lo, ate_hi), max(ate_lo, ate_hi)))

        return results

    def full_report(self, n_bootstrap=100, seed=42):
        """Generate comprehensive analysis report."""
        report = {
            'n': self._n,
            'n_treated': int(self.T.sum()),
            'n_control': int((1 - self.T).sum()),
            'propensity_summary': {
                'mean': float(self.ps.mean()),
                'std': float(self.ps.std()),
                'min': float(self.ps.min()),
                'max': float(self.ps.max()),
            },
            'estimates': self.compare_methods(),
            'bootstrap_ci': {},
            'sensitivity': self.sensitivity_analysis(),
        }

        # Bootstrap CIs for key methods
        for method in ['doubly_robust', 'ipw', 'regression']:
            est, lo, hi, se = self.bootstrap_ci(
                method=method, n_bootstrap=n_bootstrap, seed=seed)
            report['bootstrap_ci'][method] = {
                'estimate': est, 'lower': lo, 'upper': hi, 'se': se
            }

        return report


# ---------------------------------------------------------------------------
# Data Generation Utilities
# ---------------------------------------------------------------------------

def generate_observational_data(n_samples=1000, true_ate=2.0, confounding=1.0,
                                 n_covariates=3, seed=None):
    """
    Generate observational data with known confounding for testing.
    Returns (data, true_ate) where data is list of dicts.
    """
    rng = np.random.RandomState(seed)
    d = n_covariates

    # Covariates
    X = rng.randn(n_samples, d)

    # Confounding: covariates affect both treatment and outcome
    beta_t = confounding * rng.randn(d)  # covariate -> treatment
    beta_y = confounding * rng.randn(d)  # covariate -> outcome

    # Treatment assignment (logistic)
    logit_t = X @ beta_t
    ps_true = 1.0 / (1.0 + np.exp(-logit_t))
    T = (rng.rand(n_samples) < ps_true).astype(float)

    # Outcome: Y = true_ate * T + X @ beta_y + noise
    Y = true_ate * T + X @ beta_y + rng.randn(n_samples) * 0.5

    data = []
    cov_names = [f'X{i}' for i in range(d)]
    for i in range(n_samples):
        row = {'T': int(T[i]), 'Y': float(Y[i])}
        for j in range(d):
            row[cov_names[j]] = float(X[i, j])
        data.append(row)

    return data, true_ate, cov_names


def generate_heterogeneous_data(n_samples=1000, seed=None):
    """
    Generate data with heterogeneous treatment effects.
    Treatment effect depends on X0: high when X0 > 0, low when X0 <= 0.
    Returns (data, cate_high, cate_low, covariates).
    """
    rng = np.random.RandomState(seed)

    X0 = rng.randn(n_samples)
    X1 = rng.randn(n_samples)

    # Treatment depends on X1 (confounder)
    logit = 0.5 * X1
    ps = 1.0 / (1.0 + np.exp(-logit))
    T = (rng.rand(n_samples) < ps).astype(float)

    # Heterogeneous effect: 3.0 for X0 > 0, 1.0 for X0 <= 0
    cate_high = 3.0
    cate_low = 1.0
    tau = np.where(X0 > 0, cate_high, cate_low)

    Y = tau * T + 0.5 * X1 + rng.randn(n_samples) * 0.3

    data = []
    for i in range(n_samples):
        data.append({
            'T': int(T[i]), 'Y': float(Y[i]),
            'X0': float(X0[i]), 'X1': float(X1[i])
        })

    return data, cate_high, cate_low, ['X0', 'X1']
