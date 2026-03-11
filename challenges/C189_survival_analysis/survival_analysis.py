"""
C189: Survival Analysis
Built from scratch using only NumPy.

Components:
- SurvivalData: container for time-to-event data with censoring
- KaplanMeier: non-parametric survival estimator
- NelsonAalen: cumulative hazard estimator
- LogRankTest: compare survival between groups
- CoxPH: Cox proportional hazards regression
- WeibullModel: parametric survival model
- ExponentialModel: parametric (constant hazard)
- LifeTable: actuarial life table analysis
- AcceleratedFailureTime: AFT models (log-normal, log-logistic, Weibull)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# SurvivalData container
# ---------------------------------------------------------------------------

@dataclass
class SurvivalData:
    """Container for survival/time-to-event data.

    times: array of event/censoring times
    events: array of event indicators (1=event, 0=censored)
    covariates: optional matrix of covariates (n_samples x n_features)
    groups: optional group labels for multi-group comparisons
    """
    times: np.ndarray
    events: np.ndarray
    covariates: Optional[np.ndarray] = None
    groups: Optional[np.ndarray] = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.events = np.asarray(self.events, dtype=float)
        if len(self.times) != len(self.events):
            raise ValueError("times and events must have same length")
        if np.any(self.times < 0):
            raise ValueError("times must be non-negative")
        if self.covariates is not None:
            self.covariates = np.asarray(self.covariates, dtype=float)
            if self.covariates.ndim == 1:
                self.covariates = self.covariates.reshape(-1, 1)
            if len(self.covariates) != len(self.times):
                raise ValueError("covariates must have same number of rows as times")
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
            if len(self.groups) != len(self.times):
                raise ValueError("groups must have same length as times")

    @property
    def n_samples(self):
        return len(self.times)

    @property
    def n_events(self):
        return int(np.sum(self.events))

    @property
    def n_censored(self):
        return self.n_samples - self.n_events

    def subset(self, mask):
        """Return SurvivalData for a subset."""
        cov = self.covariates[mask] if self.covariates is not None else None
        grp = self.groups[mask] if self.groups is not None else None
        return SurvivalData(self.times[mask], self.events[mask], cov, grp)


# ---------------------------------------------------------------------------
# Kaplan-Meier Estimator
# ---------------------------------------------------------------------------

class KaplanMeier:
    """Non-parametric survival function estimator (product-limit)."""

    def __init__(self):
        self.event_times = None
        self.survival_prob = None
        self.n_at_risk = None
        self.n_events_at = None
        self.n_censored_at = None
        self.variance = None  # Greenwood's formula
        self._fitted = False

    def fit(self, data: SurvivalData):
        """Fit KM estimator to survival data."""
        times = data.times
        events = data.events

        # Get unique event times (only where events occurred)
        unique_times = np.unique(times[events == 1])
        unique_times.sort()

        n = len(times)
        event_times = []
        surv_probs = []
        n_at_risk_list = []
        n_events_list = []
        n_censored_list = []
        var_terms = []

        cum_surv = 1.0
        greenwood_sum = 0.0

        for t in unique_times:
            # Number at risk just before time t
            at_risk = np.sum(times >= t)
            # Number of events at time t
            d = np.sum((times == t) & (events == 1))
            # Number censored at time t (but before next event time)
            c = np.sum((times == t) & (events == 0))

            if at_risk > 0 and d > 0:
                cum_surv *= (1.0 - d / at_risk)
                if at_risk > d:
                    greenwood_sum += d / (at_risk * (at_risk - d))

                event_times.append(t)
                surv_probs.append(cum_surv)
                n_at_risk_list.append(at_risk)
                n_events_list.append(d)
                n_censored_list.append(c)
                var_terms.append(cum_surv ** 2 * greenwood_sum)

        self.event_times = np.array(event_times)
        self.survival_prob = np.array(surv_probs)
        self.n_at_risk = np.array(n_at_risk_list)
        self.n_events_at = np.array(n_events_list)
        self.n_censored_at = np.array(n_censored_list)
        self.variance = np.array(var_terms)
        self._fitted = True
        return self

    def survival_function(self, t):
        """Evaluate S(t) at given time(s)."""
        if not self._fitted:
            raise RuntimeError("Must fit before evaluating")
        t = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.ones(len(t))
        for i, ti in enumerate(t):
            if len(self.event_times) == 0 or ti < self.event_times[0]:
                result[i] = 1.0
            else:
                idx = np.searchsorted(self.event_times, ti, side='right') - 1
                result[i] = self.survival_prob[idx]
        return result if len(result) > 1 else result[0]

    def confidence_interval(self, alpha=0.05):
        """Pointwise confidence intervals using Greenwood's formula."""
        if not self._fitted:
            raise RuntimeError("Must fit before computing CI")
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        se = np.sqrt(self.variance)
        lower = np.maximum(0, self.survival_prob - z * se)
        upper = np.minimum(1, self.survival_prob + z * se)
        return lower, upper

    def median_survival(self):
        """Median survival time (time when S(t) = 0.5)."""
        if not self._fitted:
            raise RuntimeError("Must fit before computing median")
        if len(self.survival_prob) == 0:
            return np.nan
        idx = np.where(self.survival_prob <= 0.5)[0]
        if len(idx) == 0:
            return np.nan  # Survival never drops below 0.5
        return self.event_times[idx[0]]

    def restricted_mean(self, tau=None):
        """Restricted mean survival time (area under KM curve up to tau)."""
        if not self._fitted:
            raise RuntimeError("Must fit before computing RMST")
        if len(self.event_times) == 0:
            return 0.0
        if tau is None:
            tau = self.event_times[-1]

        # Area under step function
        rmst = 0.0
        prev_t = 0.0
        prev_s = 1.0
        for i, t in enumerate(self.event_times):
            if t > tau:
                break
            rmst += prev_s * (t - prev_t)
            prev_t = t
            prev_s = self.survival_prob[i]
        # Add remaining area up to tau
        rmst += prev_s * (tau - prev_t)
        return rmst


# ---------------------------------------------------------------------------
# Nelson-Aalen Cumulative Hazard Estimator
# ---------------------------------------------------------------------------

class NelsonAalen:
    """Non-parametric cumulative hazard estimator."""

    def __init__(self):
        self.event_times = None
        self.cumulative_hazard = None
        self.variance = None
        self._fitted = False

    def fit(self, data: SurvivalData):
        """Fit Nelson-Aalen estimator."""
        times = data.times
        events = data.events

        unique_times = np.unique(times[events == 1])
        unique_times.sort()

        event_times = []
        cum_hazard_list = []
        var_list = []

        cum_h = 0.0
        cum_var = 0.0

        for t in unique_times:
            at_risk = np.sum(times >= t)
            d = np.sum((times == t) & (events == 1))

            if at_risk > 0 and d > 0:
                cum_h += d / at_risk
                cum_var += d / (at_risk ** 2)
                event_times.append(t)
                cum_hazard_list.append(cum_h)
                var_list.append(cum_var)

        self.event_times = np.array(event_times)
        self.cumulative_hazard = np.array(cum_hazard_list)
        self.variance = np.array(var_list)
        self._fitted = True
        return self

    def hazard_at(self, t):
        """Evaluate cumulative hazard H(t) at given time(s)."""
        if not self._fitted:
            raise RuntimeError("Must fit before evaluating")
        t = np.atleast_1d(np.asarray(t, dtype=float))
        result = np.zeros(len(t))
        for i, ti in enumerate(t):
            if len(self.event_times) == 0 or ti < self.event_times[0]:
                result[i] = 0.0
            else:
                idx = np.searchsorted(self.event_times, ti, side='right') - 1
                result[i] = self.cumulative_hazard[idx]
        return result if len(result) > 1 else result[0]

    def survival_function(self, t):
        """Convert to survival: S(t) = exp(-H(t))."""
        h = self.hazard_at(t)
        return np.exp(-h)


# ---------------------------------------------------------------------------
# Log-Rank Test
# ---------------------------------------------------------------------------

@dataclass
class LogRankResult:
    """Result of a log-rank test."""
    test_statistic: float
    p_value: float
    n_groups: int
    observed: np.ndarray  # observed events per group
    expected: np.ndarray  # expected events per group

    @property
    def significant(self):
        return self.p_value < 0.05


class LogRankTest:
    """Log-rank test for comparing survival curves between groups."""

    @staticmethod
    def test(data: SurvivalData, weights=None):
        """Perform log-rank test.

        weights: None for standard log-rank, 'wilcoxon' for Wilcoxon (Breslow),
                 'tarone-ware' for Tarone-Ware, 'peto' for Peto-Peto
        """
        if data.groups is None:
            raise ValueError("SurvivalData must have groups for log-rank test")

        unique_groups = np.unique(data.groups)
        n_groups = len(unique_groups)
        if n_groups < 2:
            raise ValueError("Need at least 2 groups")

        times = data.times
        events = data.events
        groups = data.groups

        # All unique event times
        all_event_times = np.unique(times[events == 1])
        all_event_times.sort()

        # Compute observed and expected for each group
        observed = np.zeros(n_groups)
        expected = np.zeros(n_groups)
        var_matrix = np.zeros((n_groups, n_groups))

        for t in all_event_times:
            # Total at risk and events
            n_total = np.sum(times >= t)
            d_total = np.sum((times == t) & (events == 1))

            if n_total <= 1 or d_total == 0:
                continue

            # Weight
            if weights is None:
                w = 1.0
            elif weights == 'wilcoxon':
                w = n_total
            elif weights == 'tarone-ware':
                w = np.sqrt(n_total)
            elif weights == 'peto':
                # Peto-Peto uses KM estimate
                km_est = 1.0
                earlier = all_event_times[all_event_times < t]
                for et in earlier:
                    n_r = np.sum(times >= et)
                    d_r = np.sum((times == et) & (events == 1))
                    if n_r > 0:
                        km_est *= (1 - d_r / n_r)
                w = km_est
            else:
                w = 1.0

            for j, g in enumerate(unique_groups):
                mask_g = groups == g
                n_j = np.sum((times >= t) & mask_g)
                d_j = np.sum((times == t) & (events == 1) & mask_g)

                e_j = n_j * d_total / n_total

                observed[j] += w * d_j
                expected[j] += w * e_j

            # Variance contribution (hypergeometric)
            for j in range(n_groups):
                mask_j = groups == unique_groups[j]
                n_j = np.sum((times >= t) & mask_j)

                factor = d_total * (n_total - d_total) / (n_total ** 2 * max(n_total - 1, 1))

                for k in range(n_groups):
                    mask_k = groups == unique_groups[k]
                    n_k = np.sum((times >= t) & mask_k)

                    if j == k:
                        var_matrix[j, k] += w ** 2 * n_j * (n_total - n_j) * factor
                    else:
                        var_matrix[j, k] -= w ** 2 * n_j * n_k * factor

        # Chi-squared statistic (use K-1 groups to avoid singularity)
        Z = (observed - expected)[:-1]
        V = var_matrix[:-1, :-1]

        try:
            V_inv = np.linalg.inv(V)
            chi2 = float(Z @ V_inv @ Z)
        except np.linalg.LinAlgError:
            # Fallback: sum of (O-E)^2/E
            chi2 = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-10)))

        df = n_groups - 1
        # Chi-squared p-value using survival function approximation
        p_value = _chi2_sf(chi2, df)

        return LogRankResult(
            test_statistic=chi2,
            p_value=p_value,
            n_groups=n_groups,
            observed=observed,
            expected=expected
        )


# ---------------------------------------------------------------------------
# Cox Proportional Hazards Model
# ---------------------------------------------------------------------------

@dataclass
class CoxPHResult:
    """Result of Cox PH model fitting."""
    coefficients: np.ndarray
    standard_errors: np.ndarray
    hazard_ratios: np.ndarray
    z_scores: np.ndarray
    p_values: np.ndarray
    log_likelihood: float
    concordance: float
    n_samples: int
    n_events: int

    def summary(self):
        """Return summary dict for each covariate."""
        results = []
        for i in range(len(self.coefficients)):
            results.append({
                'coef': self.coefficients[i],
                'se': self.standard_errors[i],
                'hr': self.hazard_ratios[i],
                'z': self.z_scores[i],
                'p': self.p_values[i],
            })
        return results


class CoxPH:
    """Cox Proportional Hazards regression model.

    Fitted via Newton-Raphson on the partial likelihood.
    Handles ties using Breslow's method.
    """

    def __init__(self, max_iter=100, tol=1e-9, l2_penalty=0.0):
        self.max_iter = max_iter
        self.tol = tol
        self.l2_penalty = l2_penalty
        self.coefficients = None
        self._fitted = False
        self._baseline_hazard_times = None
        self._baseline_hazard = None
        self._baseline_survival = None
        self._data = None

    def fit(self, data: SurvivalData):
        """Fit Cox PH model using Newton-Raphson."""
        if data.covariates is None:
            raise ValueError("CoxPH requires covariates")

        X = data.covariates
        times = data.times
        events = data.events
        n, p = X.shape

        # Sort by time (descending for efficient risk set computation)
        order = np.argsort(-times)
        X = X[order]
        times = times[order]
        events = events[order]

        # Initialize coefficients
        beta = np.zeros(p)

        for iteration in range(self.max_iter):
            # Compute exp(X @ beta)
            eta = X @ beta
            eta = np.clip(eta, -500, 500)
            exp_eta = np.exp(eta)

            # Compute gradient and Hessian using Breslow method
            # For descending-sorted data, risk set at time t_i includes all j with t_j >= t_i
            # Since sorted descending, risk set for index i = indices 0..i (all with time >= t_i)
            # But we need to handle ties properly

            grad = np.zeros(p)
            hess = np.zeros((p, p))
            log_lik = 0.0

            # Forward pass: accumulate risk sums
            # Since times are descending, we process from end (smallest time) to start (largest)
            # Actually, let's re-sort ascending for clearer logic
            asc_order = np.argsort(times)
            X_asc = X[asc_order]
            times_asc = times[asc_order]
            events_asc = events[asc_order]
            eta_asc = eta[asc_order]
            exp_eta_asc = exp_eta[asc_order]

            # Risk set sums (cumulative from right = from largest time)
            S0 = 0.0  # sum of exp(eta) in risk set
            S1 = np.zeros(p)  # sum of X * exp(eta) in risk set
            S2 = np.zeros((p, p))  # sum of X X^T exp(eta) in risk set

            # Process from largest time to smallest
            for i in range(n - 1, -1, -1):
                # Add individual i to risk set
                S0 += exp_eta_asc[i]
                S1 += X_asc[i] * exp_eta_asc[i]
                S2 += np.outer(X_asc[i], X_asc[i]) * exp_eta_asc[i]

                if events_asc[i] == 1:
                    if S0 > 0:
                        log_lik += eta_asc[i] - np.log(S0)
                        grad += X_asc[i] - S1 / S0
                        hess -= (S2 / S0 - np.outer(S1, S1) / S0 ** 2)

            # L2 penalty
            if self.l2_penalty > 0:
                log_lik -= 0.5 * self.l2_penalty * np.sum(beta ** 2)
                grad -= self.l2_penalty * beta
                hess -= self.l2_penalty * np.eye(p)

            # Newton-Raphson update
            try:
                delta = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                delta = grad * 0.01  # Fallback: gradient descent

            beta -= delta

            if np.max(np.abs(delta)) < self.tol:
                break

        # Compute final quantities
        self.coefficients = beta

        # Information matrix (negative Hessian)
        info_matrix = -hess
        try:
            cov_matrix = np.linalg.inv(info_matrix)
            se = np.sqrt(np.maximum(np.diag(cov_matrix), 0))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)

        hr = np.exp(beta)
        z = beta / np.maximum(se, 1e-10)
        p_vals = 2 * _normal_sf(np.abs(z))

        # Concordance index
        concordance = self._concordance(data.times, data.events, data.covariates @ beta)

        # Baseline hazard (Breslow estimator)
        self._compute_baseline_hazard(data)
        self._data = data
        self._fitted = True

        return CoxPHResult(
            coefficients=beta,
            standard_errors=se,
            hazard_ratios=hr,
            z_scores=z,
            p_values=p_vals,
            log_likelihood=log_lik,
            concordance=concordance,
            n_samples=n,
            n_events=int(np.sum(data.events))
        )

    def _compute_baseline_hazard(self, data):
        """Compute Breslow baseline hazard."""
        times = data.times
        events = data.events
        X = data.covariates

        eta = X @ self.coefficients
        eta = np.clip(eta, -500, 500)
        exp_eta = np.exp(eta)

        unique_event_times = np.unique(times[events == 1])
        unique_event_times.sort()

        h0 = []
        for t in unique_event_times:
            d = np.sum((times == t) & (events == 1))
            risk_sum = np.sum(exp_eta[times >= t])
            if risk_sum > 0:
                h0.append(d / risk_sum)
            else:
                h0.append(0.0)

        self._baseline_hazard_times = unique_event_times
        self._baseline_hazard = np.array(h0)
        self._baseline_survival = np.exp(-np.cumsum(self._baseline_hazard))

    def predict_survival(self, X_new, t=None):
        """Predict survival function for new covariates."""
        if not self._fitted:
            raise RuntimeError("Must fit before predicting")
        X_new = np.atleast_2d(X_new)
        eta = X_new @ self.coefficients

        if t is not None:
            t = np.atleast_1d(t)
            results = np.ones((len(X_new), len(t)))
            cum_h0 = np.cumsum(self._baseline_hazard)
            for i, ti in enumerate(t):
                idx = np.searchsorted(self._baseline_hazard_times, ti, side='right') - 1
                if idx < 0:
                    h0_t = 0.0
                else:
                    h0_t = cum_h0[idx]
                for j in range(len(X_new)):
                    results[j, i] = np.exp(-h0_t * np.exp(eta[j]))
            return results

        # Return baseline survival adjusted for covariates
        results = np.zeros((len(X_new), len(self._baseline_survival)))
        for j in range(len(X_new)):
            results[j] = self._baseline_survival ** np.exp(eta[j])
        return results

    def predict_hazard_ratio(self, X_new):
        """Predict hazard ratio relative to baseline."""
        X_new = np.atleast_2d(X_new)
        return np.exp(X_new @ self.coefficients)

    @staticmethod
    def _concordance(times, events, risk_scores):
        """Compute Harrell's concordance index."""
        concordant = 0
        discordant = 0
        tied = 0

        event_idx = np.where(events == 1)[0]
        for i in event_idx:
            # Compare with all individuals who survived longer
            longer = np.where(times > times[i])[0]
            for j in longer:
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied += 1

        total = concordant + discordant + tied
        if total == 0:
            return 0.5
        return (concordant + 0.5 * tied) / total


# ---------------------------------------------------------------------------
# Parametric Models
# ---------------------------------------------------------------------------

class ExponentialModel:
    """Exponential survival model: S(t) = exp(-lambda * t).

    Constant hazard rate.
    """

    def __init__(self):
        self.rate = None  # lambda
        self._fitted = False

    def fit(self, data: SurvivalData):
        """MLE: lambda = d / sum(t) where d = total events."""
        total_time = np.sum(data.times)
        n_events = np.sum(data.events)
        if total_time == 0:
            raise ValueError("Total observed time is zero")
        self.rate = n_events / total_time
        self._fitted = True
        return self

    def survival(self, t):
        """S(t) = exp(-lambda * t)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-self.rate * t)

    def hazard(self, t):
        """h(t) = lambda (constant)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.full_like(t, self.rate)

    def cumulative_hazard(self, t):
        """H(t) = lambda * t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.rate * t

    def mean_survival(self):
        """E[T] = 1/lambda."""
        return 1.0 / self.rate

    def median_survival(self):
        """Median = ln(2)/lambda."""
        return np.log(2) / self.rate

    def log_likelihood(self, data: SurvivalData):
        """Log-likelihood for censored data."""
        ll = 0.0
        for i in range(data.n_samples):
            if data.events[i] == 1:
                ll += np.log(self.rate) - self.rate * data.times[i]
            else:
                ll += -self.rate * data.times[i]
        return ll

    def aic(self, data: SurvivalData):
        """Akaike Information Criterion."""
        return -2 * self.log_likelihood(data) + 2  # 1 parameter


class WeibullModel:
    """Weibull survival model: S(t) = exp(-(t/lambda)^k).

    k (shape): k<1 decreasing hazard, k=1 exponential, k>1 increasing hazard
    lambda (scale): characteristic life
    """

    def __init__(self):
        self.shape = None  # k
        self.scale = None  # lambda
        self._fitted = False

    def fit(self, data: SurvivalData, max_iter=200, tol=1e-8):
        """MLE via Newton-Raphson for shape, closed-form for scale."""
        times = data.times
        events = data.events
        d = np.sum(events)

        if d == 0:
            raise ValueError("No events observed")

        # Avoid zero times
        t_safe = np.maximum(times, 1e-10)
        log_t = np.log(t_safe)

        # Initialize shape parameter
        k = 1.0

        for _ in range(max_iter):
            t_k = t_safe ** k
            sum_tk = np.sum(t_k)
            sum_tk_logt = np.sum(t_k * log_t)
            sum_d_logt = np.sum(events * log_t)

            if sum_tk == 0:
                break

            # Score equation for k
            f = d / k + sum_d_logt - d * sum_tk_logt / sum_tk

            # Derivative
            sum_tk_logt2 = np.sum(t_k * log_t ** 2)
            f_prime = -d / k ** 2 - d * (sum_tk_logt2 * sum_tk - sum_tk_logt ** 2) / sum_tk ** 2

            if abs(f_prime) < 1e-15:
                break

            delta = f / f_prime
            k_new = k - delta

            if k_new <= 0:
                k = k / 2  # Step back
            else:
                k = k_new

            if abs(delta) < tol:
                break

        # Scale parameter (closed form given k)
        t_k = t_safe ** k
        lam = (np.sum(t_k) / d) ** (1.0 / k)

        self.shape = k
        self.scale = lam
        self._fitted = True
        return self

    def survival(self, t):
        """S(t) = exp(-(t/lambda)^k)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.exp(-(t / self.scale) ** self.shape)

    def hazard(self, t):
        """h(t) = (k/lambda)(t/lambda)^(k-1)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_safe = np.maximum(t, 1e-10)
        return (self.shape / self.scale) * (t_safe / self.scale) ** (self.shape - 1)

    def cumulative_hazard(self, t):
        """H(t) = (t/lambda)^k."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return (t / self.scale) ** self.shape

    def mean_survival(self):
        """E[T] = lambda * Gamma(1 + 1/k)."""
        from math import gamma
        return self.scale * gamma(1 + 1.0 / self.shape)

    def median_survival(self):
        """Median = lambda * (ln 2)^(1/k)."""
        return self.scale * np.log(2) ** (1.0 / self.shape)

    def log_likelihood(self, data: SurvivalData):
        """Log-likelihood for censored data."""
        k, lam = self.shape, self.scale
        t_safe = np.maximum(data.times, 1e-10)
        ll = 0.0
        for i in range(data.n_samples):
            if data.events[i] == 1:
                ll += np.log(k / lam) + (k - 1) * np.log(t_safe[i] / lam) - (t_safe[i] / lam) ** k
            else:
                ll += -(t_safe[i] / lam) ** k
        return ll

    def aic(self, data: SurvivalData):
        """Akaike Information Criterion."""
        return -2 * self.log_likelihood(data) + 4  # 2 parameters


class LogNormalModel:
    """Log-normal survival model: log(T) ~ Normal(mu, sigma^2)."""

    def __init__(self):
        self.mu = None
        self.sigma = None
        self._fitted = False

    def fit(self, data: SurvivalData, max_iter=100, tol=1e-8):
        """MLE for log-normal with censoring (EM-style)."""
        events = data.events
        t_safe = np.maximum(data.times, 1e-10)
        log_t = np.log(t_safe)

        # If no censoring, simple MLE
        if np.all(events == 1):
            self.mu = np.mean(log_t)
            self.sigma = np.std(log_t, ddof=0)
            if self.sigma == 0:
                self.sigma = 1e-6
            self._fitted = True
            return self

        # Initial estimates from uncensored observations
        uncens = log_t[events == 1]
        if len(uncens) > 0:
            self.mu = np.mean(uncens)
            self.sigma = max(np.std(uncens, ddof=0), 1e-6)
        else:
            self.mu = np.mean(log_t)
            self.sigma = max(np.std(log_t, ddof=0), 1e-6)

        # Newton-Raphson on log-likelihood
        for _ in range(max_iter):
            z = (log_t - self.mu) / self.sigma

            # Gradient
            d_mu = 0.0
            d_sigma = 0.0

            for i in range(len(log_t)):
                if events[i] == 1:
                    d_mu += (log_t[i] - self.mu) / self.sigma ** 2
                    d_sigma += ((log_t[i] - self.mu) ** 2 / self.sigma ** 3 - 1.0 / self.sigma)
                else:
                    # Censored: use hazard function of standard normal
                    zi = z[i]
                    sf = _normal_sf_scalar(zi)
                    if sf > 1e-15:
                        phi = _normal_pdf(zi)
                        ratio = phi / sf  # inverse Mills ratio
                        d_mu -= ratio / self.sigma
                        d_sigma -= ratio * zi / self.sigma

            step_mu = 0.01 * d_mu
            step_sigma = 0.01 * d_sigma

            self.mu += step_mu
            self.sigma = max(self.sigma + step_sigma, 1e-6)

            if abs(step_mu) < tol and abs(step_sigma) < tol:
                break

        self._fitted = True
        return self

    def survival(self, t):
        """S(t) = 1 - Phi((log(t) - mu) / sigma)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_safe = np.maximum(t, 1e-10)
        z = (np.log(t_safe) - self.mu) / self.sigma
        return np.array([_normal_sf_scalar(zi) for zi in z])

    def hazard(self, t):
        """h(t) = f(t) / S(t)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_safe = np.maximum(t, 1e-10)
        z = (np.log(t_safe) - self.mu) / self.sigma
        pdf_vals = np.array([_normal_pdf(zi) for zi in z])
        sf_vals = np.array([_normal_sf_scalar(zi) for zi in z])
        f_t = pdf_vals / (t_safe * self.sigma)
        return f_t / np.maximum(sf_vals, 1e-15)

    def median_survival(self):
        """Median = exp(mu)."""
        return np.exp(self.mu)

    def mean_survival(self):
        """E[T] = exp(mu + sigma^2/2)."""
        return np.exp(self.mu + self.sigma ** 2 / 2)


class LogLogisticModel:
    """Log-logistic survival model: S(t) = 1 / (1 + (t/alpha)^beta)."""

    def __init__(self):
        self.alpha = None  # scale
        self.beta = None   # shape
        self._fitted = False

    def fit(self, data: SurvivalData, max_iter=1000, tol=1e-8):
        """MLE using logistic regression on log-transformed data.

        log(T) ~ Logistic(mu, s) where alpha=exp(mu), beta=1/s.
        Uses Newton-Raphson on (mu, log_s) to avoid constraints.
        """
        events = data.events
        t_safe = np.maximum(data.times, 1e-10)
        log_t = np.log(t_safe)
        n = len(t_safe)
        d = np.sum(events)

        if d == 0:
            raise ValueError("No events observed")

        # Initialize from uncensored log-times
        uncens = log_t[events == 1]
        if len(uncens) > 1:
            mu = np.mean(uncens)
            s = max(np.std(uncens, ddof=0) * np.sqrt(3) / np.pi, 0.1)
        else:
            mu = np.mean(log_t)
            s = max(np.std(log_t, ddof=0) * np.sqrt(3) / np.pi, 0.1)

        log_s = np.log(s)

        for iteration in range(max_iter):
            s = np.exp(log_s)

            # Compute log-likelihood and its gradient
            # w_i = (log_t_i - mu) / s
            w = (log_t - mu) / s
            w = np.clip(w, -500, 500)

            # Logistic CDF: F(w) = 1/(1+exp(-w)), S(w) = 1/(1+exp(w))
            # log f(t) for event i: -log(s) - log(t_i) + w_i - 2*log(1+exp(w_i))
            # log S(t) for censored i: -log(1+exp(w_i))

            # Stable computation of sigmoid
            # sigma(w) = 1/(1+exp(-w)), 1-sigma(w) = 1/(1+exp(w))
            sigma = np.where(w >= 0, 1.0 / (1.0 + np.exp(-w)), np.exp(w) / (1.0 + np.exp(w)))

            # Gradient w.r.t mu:
            # event: d/dmu = (2*sigma(w_i) - 1) / s
            # censored: d/dmu = sigma(w_i) / s
            # Gradient w.r.t log_s (chain rule: d/d(log_s) = s * d/ds):
            # event: d/d(log_s) = -1 + w_i*(2*sigma(w_i) - 1)
            # censored: d/d(log_s) = w_i*sigma(w_i)

            g_mu = 0.0
            g_logs = 0.0
            for i in range(n):
                if events[i] == 1:
                    g_mu += (2 * sigma[i] - 1) / s
                    g_logs += -1 + w[i] * (2 * sigma[i] - 1)
                else:
                    g_mu += sigma[i] / s
                    g_logs += w[i] * sigma[i]

            # Simple gradient ascent with adaptive step
            step = 0.1 / (1 + iteration * 0.01)
            mu_step = step * g_mu
            logs_step = step * g_logs

            # Clip steps
            mu_step = np.clip(mu_step, -1, 1)
            logs_step = np.clip(logs_step, -0.5, 0.5)

            mu += mu_step
            log_s += logs_step

            if abs(mu_step) < tol and abs(logs_step) < tol:
                break

        s = np.exp(log_s)
        self.alpha = np.exp(mu)
        self.beta = 1.0 / s
        self._fitted = True
        return self

    def survival(self, t):
        """S(t) = 1 / (1 + (t/alpha)^beta)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return 1.0 / (1.0 + (t / self.alpha) ** self.beta)

    def hazard(self, t):
        """h(t) = (beta/alpha)(t/alpha)^(beta-1) / (1 + (t/alpha)^beta)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        t_safe = np.maximum(t, 1e-10)
        u = (t_safe / self.alpha) ** self.beta
        return (self.beta / self.alpha) * (t_safe / self.alpha) ** (self.beta - 1) / (1 + u)

    def median_survival(self):
        """Median = alpha."""
        return self.alpha


# ---------------------------------------------------------------------------
# Life Table
# ---------------------------------------------------------------------------

@dataclass
class LifeTableRow:
    """Single row of a life table."""
    interval_start: float
    interval_end: float
    n_entering: int
    n_events: int
    n_censored: int
    n_at_risk: float  # adjusted: entering - censored/2
    mortality_rate: float  # q_x
    survival_prob: float  # p_x = 1 - q_x
    cumulative_survival: float  # S(x)
    hazard: float  # h(x)


class LifeTable:
    """Actuarial life table analysis."""

    def __init__(self, intervals=None, n_intervals=10):
        self.intervals = intervals
        self.n_intervals = n_intervals
        self.table = None
        self._fitted = False

    def fit(self, data: SurvivalData):
        """Build life table from survival data."""
        times = data.times
        events = data.events

        if self.intervals is None:
            max_t = np.max(times)
            self.intervals = np.linspace(0, max_t, self.n_intervals + 1)

        rows = []
        cum_surv = 1.0

        for i in range(len(self.intervals) - 1):
            t_start = self.intervals[i]
            t_end = self.intervals[i + 1]

            # In interval [t_start, t_end)
            in_interval = (times >= t_start) & (times < t_end)
            n_entering = int(np.sum(times >= t_start))
            n_events = int(np.sum(in_interval & (events == 1)))
            n_censored = int(np.sum(in_interval & (events == 0)))

            # Adjusted number at risk (actuarial adjustment)
            n_at_risk = n_entering - n_censored / 2.0

            if n_at_risk > 0:
                q_x = n_events / n_at_risk
            else:
                q_x = 0.0

            p_x = 1.0 - q_x
            cum_surv *= p_x

            width = t_end - t_start
            if width > 0 and q_x < 1:
                h_x = 2 * q_x / (width * (1 + p_x))
            else:
                h_x = 0.0

            rows.append(LifeTableRow(
                interval_start=t_start,
                interval_end=t_end,
                n_entering=n_entering,
                n_events=n_events,
                n_censored=n_censored,
                n_at_risk=n_at_risk,
                mortality_rate=q_x,
                survival_prob=p_x,
                cumulative_survival=cum_surv,
                hazard=h_x
            ))

        self.table = rows
        self._fitted = True
        return self

    def get_table(self):
        """Return life table as list of dicts."""
        if not self._fitted:
            raise RuntimeError("Must fit before getting table")
        return [
            {
                'interval': (r.interval_start, r.interval_end),
                'n_entering': r.n_entering,
                'n_events': r.n_events,
                'n_censored': r.n_censored,
                'n_at_risk': r.n_at_risk,
                'q_x': r.mortality_rate,
                'p_x': r.survival_prob,
                'S_x': r.cumulative_survival,
                'h_x': r.hazard,
            }
            for r in self.table
        ]


# ---------------------------------------------------------------------------
# Schoenfeld Residuals (PH assumption test)
# ---------------------------------------------------------------------------

class SchoenfeldTest:
    """Test proportional hazards assumption using Schoenfeld residuals."""

    @staticmethod
    def test(data: SurvivalData, cox_result: CoxPHResult):
        """Compute Schoenfeld residuals and test PH assumption.

        Returns dict with residuals and test results per covariate.
        """
        if data.covariates is None:
            raise ValueError("Need covariates")

        X = data.covariates
        times = data.times
        events = data.events
        beta = cox_result.coefficients
        n, p = X.shape

        eta = X @ beta
        eta = np.clip(eta, -500, 500)
        exp_eta = np.exp(eta)

        # Compute Schoenfeld residuals for each event
        event_idx = np.where(events == 1)[0]
        event_times_sorted = times[event_idx]
        order = np.argsort(event_times_sorted)
        event_idx = event_idx[order]

        residuals = []
        for i in event_idx:
            t_i = times[i]
            risk_set = times >= t_i

            # Expected covariate value under null
            weights = exp_eta[risk_set]
            if np.sum(weights) > 0:
                expected_x = np.average(X[risk_set], axis=0, weights=weights)
            else:
                expected_x = np.zeros(p)

            resid = X[i] - expected_x
            residuals.append(resid)

        residuals = np.array(residuals)
        event_times = times[event_idx]

        # Test: correlation of residuals with time
        results = {}
        for j in range(p):
            r = residuals[:, j]
            # Rank correlation with time
            rho = _spearman_correlation(event_times, r)
            n_events = len(r)
            # Test statistic
            if n_events > 2:
                t_stat = rho * np.sqrt((n_events - 2) / max(1 - rho ** 2, 1e-10))
                # Two-sided p-value from t-distribution approximation
                p_val = 2 * _t_sf(abs(t_stat), n_events - 2)
            else:
                t_stat = 0.0
                p_val = 1.0

            results[f'covariate_{j}'] = {
                'rho': rho,
                'test_statistic': t_stat,
                'p_value': p_val,
                'ph_holds': p_val > 0.05,
                'residuals': r,
            }

        return results


# ---------------------------------------------------------------------------
# Competing Risks (Cumulative Incidence Function)
# ---------------------------------------------------------------------------

class CumulativeIncidence:
    """Cumulative incidence function for competing risks.

    Estimates the probability of a specific event type in the
    presence of competing risks.
    """

    def __init__(self):
        self.event_times = None
        self.cif = None  # dict: event_type -> cumulative incidence
        self._fitted = False

    def fit(self, times, events, event_types=None):
        """Fit CIF.

        times: event/censoring times
        events: event type (0=censored, 1,2,...=different event types)
        event_types: which event types to compute CIF for (default: all non-zero)
        """
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)

        if event_types is None:
            event_types = sorted(set(events) - {0})

        # Overall KM estimate (all events combined)
        all_events = (events > 0).astype(float)
        km_data = SurvivalData(times, all_events)
        km = KaplanMeier().fit(km_data)

        unique_times = np.unique(times[events > 0])
        unique_times.sort()

        self.cif = {}
        for etype in event_types:
            cif_vals = []
            cum_inc = 0.0

            for t in unique_times:
                # KM estimate just before t
                if len(km.event_times) == 0:
                    s_prev = 1.0
                else:
                    idx = np.searchsorted(km.event_times, t, side='left') - 1
                    s_prev = km.survival_prob[idx] if idx >= 0 else 1.0
                    # Adjust: S just before t
                    if idx >= 0 and km.event_times[idx] == t:
                        # Use S from before this event time
                        if idx > 0:
                            s_prev = km.survival_prob[idx - 1]
                        else:
                            s_prev = 1.0

                # Cause-specific hazard at t
                at_risk = np.sum(times >= t)
                d_cause = np.sum((times == t) & (events == etype))

                if at_risk > 0:
                    h_cause = d_cause / at_risk
                else:
                    h_cause = 0.0

                cum_inc += s_prev * h_cause
                cif_vals.append(cum_inc)

            self.cif[etype] = np.array(cif_vals)

        self.event_times = unique_times
        self._fitted = True
        return self

    def incidence_at(self, t, event_type):
        """Get cumulative incidence for event_type at time t."""
        if not self._fitted:
            raise RuntimeError("Must fit first")
        if event_type not in self.cif:
            raise ValueError(f"Unknown event type: {event_type}")

        t = float(t)
        if t < self.event_times[0]:
            return 0.0
        idx = np.searchsorted(self.event_times, t, side='right') - 1
        return self.cif[event_type][idx]


# ---------------------------------------------------------------------------
# Utility functions (no scipy dependency for core)
# ---------------------------------------------------------------------------

def _normal_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def _normal_sf_scalar(x):
    """Standard normal survival function (1 - CDF) for scalar x."""
    # Use error function approximation
    return 0.5 * _erfc(x / np.sqrt(2))


def _normal_sf(x):
    """Standard normal survival function for array."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.array([_normal_sf_scalar(xi) for xi in x])


def _erfc(x):
    """Complementary error function approximation (Abramowitz & Stegun)."""
    # Handle sign
    if x < 0:
        return 2.0 - _erfc(-x)

    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return poly * np.exp(-x * x)


def _chi2_sf(x, df):
    """Chi-squared survival function approximation."""
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0

    # Wilson-Hilferty approximation
    if df > 2:
        z = ((x / df) ** (1.0 / 3) - (1 - 2.0 / (9 * df))) / np.sqrt(2.0 / (9 * df))
        return _normal_sf_scalar(z)

    # For df=1,2 use exact forms
    if df == 1:
        return 2 * _normal_sf_scalar(np.sqrt(x))
    if df == 2:
        return np.exp(-x / 2)

    return _normal_sf_scalar(np.sqrt(2 * x) - np.sqrt(2 * df - 1))


def _t_sf(x, df):
    """Student's t survival function approximation."""
    if df <= 0:
        return 0.5
    # For large df, approximate with normal
    if df > 30:
        return _normal_sf_scalar(x)
    # Approximation via normal with correction
    z = x * (1 - 1.0 / (4 * df)) / np.sqrt(1 + x ** 2 / (2 * df))
    return _normal_sf_scalar(z)


def _spearman_correlation(x, y):
    """Spearman rank correlation."""
    n = len(x)
    if n < 2:
        return 0.0

    def _rank(arr):
        order = np.argsort(arr)
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        # Handle ties (average rank)
        vals = arr[order]
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[j + 1] == vals[j]:
                j += 1
            if j > i:
                avg_rank = np.mean(ranks[order[i:j + 1]])
                for k in range(i, j + 1):
                    ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    # Pearson correlation on ranks
    mx, my = np.mean(rx), np.mean(ry)
    dx, dy = rx - mx, ry - my
    denom = np.sqrt(np.sum(dx ** 2) * np.sum(dy ** 2))
    if denom == 0:
        return 0.0
    return np.sum(dx * dy) / denom
