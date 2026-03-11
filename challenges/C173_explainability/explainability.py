"""
C173: Model Explainability & Interpretability

Composes:
- C140 Neural Network (model forward/backward, Tensor)

Features:
- PermutationImportance: feature importance via shuffling
- LIME: local interpretable model-agnostic explanations
- KernelSHAP: Shapley value approximation via weighted regression
- IntegratedGradients: gradient-based attribution (baseline -> input)
- SaliencyMap: raw input gradient magnitude
- PartialDependence: marginal effect of features
- ICE: individual conditional expectation curves
- CounterfactualExplainer: minimal perturbation to change prediction
- FeatureInteraction: H-statistic for interaction strength
- ModelSummary: weight statistics, layer analysis
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import Tensor, Sequential, Dense, Activation


# ============================================================
# Utility helpers
# ============================================================

def _predict_single(model, x):
    """Run model.predict on a single sample, return list of output values."""
    if isinstance(x, list):
        x = Tensor(x)
    was_2d = len(x.shape) == 2
    if not was_2d:
        x = Tensor([x.data])
    out = model.predict(x)
    if len(out.shape) == 2:
        return out.data[0] if isinstance(out.data[0], list) else out.data
    return out.data


def _predict_batch(model, X):
    """Run model.predict on batch, return 2D list [[output_vals], ...]."""
    if isinstance(X, list):
        X = Tensor(X)
    out = model.predict(X)
    if len(out.shape) == 1:
        return [[v] for v in out.data]
    return [row[:] for row in out.data]


def _predict_class(model, x):
    """Return predicted class index for a single sample."""
    vals = _predict_single(model, x)
    if len(vals) == 1:
        return 1 if vals[0] > 0.5 else 0
    return vals.index(max(vals))


def _accuracy(model, X_data, y_data):
    """Compute accuracy over dataset."""
    correct = 0
    for i in range(len(X_data)):
        pred = _predict_class(model, Tensor(X_data[i]))
        target = y_data[i] if isinstance(y_data[i], int) else (
            y_data[i].index(max(y_data[i])) if isinstance(y_data[i], list) else int(y_data[i])
        )
        if pred == target:
            correct += 1
    return correct / len(X_data) if X_data else 0.0


def _tensor_to_2d_list(t):
    """Convert Tensor to 2D list."""
    if isinstance(t, list):
        if len(t) == 0:
            return []
        if isinstance(t[0], list):
            return t
        return [[v] for v in t]
    if len(t.shape) == 1:
        return [[v] for v in t.data]
    return [row[:] for row in t.data]


def _get_data_list(X):
    """Get X as a 2D list."""
    if isinstance(X, Tensor):
        return _tensor_to_2d_list(X)
    if isinstance(X, list):
        if len(X) == 0:
            return []
        if isinstance(X[0], list):
            return X
        return [[v] for v in X]
    return X


# ============================================================
# PermutationImportance
# ============================================================

class PermutationImportance:
    """Feature importance via permutation.

    Shuffles each feature column, measures accuracy drop.
    Higher drop = more important feature.
    """

    def __init__(self, model, n_repeats=5, rng=None):
        self.model = model
        self.n_repeats = n_repeats
        self.rng = rng or random.Random(42)
        self.importances_ = None
        self.importances_mean_ = None
        self.importances_std_ = None

    def compute(self, X, y):
        """Compute permutation importance.

        X: 2D list or Tensor (n_samples, n_features)
        y: list of class labels (int)
        Returns self.
        """
        X_data = _get_data_list(X)
        n_samples = len(X_data)
        n_features = len(X_data[0]) if n_samples > 0 else 0

        baseline_acc = _accuracy(self.model, X_data, y)

        self.importances_ = []
        for f in range(n_features):
            drops = []
            for _ in range(self.n_repeats):
                # Shuffle feature f
                X_perm = [row[:] for row in X_data]
                col = [X_perm[i][f] for i in range(n_samples)]
                self.rng.shuffle(col)
                for i in range(n_samples):
                    X_perm[i][f] = col[i]
                perm_acc = _accuracy(self.model, X_perm, y)
                drops.append(baseline_acc - perm_acc)
            self.importances_.append(drops)

        self.importances_mean_ = [sum(d) / len(d) for d in self.importances_]
        self.importances_std_ = [
            math.sqrt(sum((v - m) ** 2 for v in d) / len(d))
            for d, m in zip(self.importances_, self.importances_mean_)
        ]
        return self

    def ranking(self):
        """Return feature indices sorted by importance (most important first)."""
        if self.importances_mean_ is None:
            raise ValueError("Call compute() first")
        indexed = list(enumerate(self.importances_mean_))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in indexed]


# ============================================================
# LIME: Local Interpretable Model-agnostic Explanations
# ============================================================

class LIME:
    """Local linear approximation around a data point.

    Generates perturbations around an instance, gets model predictions,
    fits a weighted linear model to explain the local behavior.
    """

    def __init__(self, model, n_samples=100, kernel_width=0.75, rng=None):
        self.model = model
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.rng = rng or random.Random(42)

    def explain(self, instance, target_class=None):
        """Explain prediction for a single instance.

        instance: list of feature values
        target_class: which output index to explain (default: predicted class)
        Returns: LIMEExplanation
        """
        if isinstance(instance, Tensor):
            instance = instance.data if len(instance.shape) == 1 else instance.data[0]

        n_features = len(instance)
        pred = _predict_single(self.model, Tensor(instance))
        if target_class is None:
            target_class = pred.index(max(pred)) if len(pred) > 1 else 0

        # Generate perturbations
        perturbations = []
        predictions = []
        weights = []

        for _ in range(self.n_samples):
            # Binary mask: which features to keep
            mask = [1 if self.rng.random() > 0.5 else 0 for _ in range(n_features)]
            # Create perturbed instance (0 = replace with noise)
            perturbed = []
            for j in range(n_features):
                if mask[j]:
                    perturbed.append(instance[j])
                else:
                    perturbed.append(instance[j] + self.rng.gauss(0, 1))
            perturbations.append(mask)

            p = _predict_single(self.model, Tensor(perturbed))
            target_val = p[target_class] if len(p) > 1 else p[0]
            predictions.append(target_val)

            # Kernel weight: exp(-d^2 / kernel_width^2)
            dist_sq = sum((1 - m) for m in mask)
            w = math.exp(-dist_sq / (self.kernel_width ** 2))
            weights.append(w)

        # Fit weighted linear regression: predictions ~ perturbations
        coefficients = self._weighted_linear_regression(
            perturbations, predictions, weights, n_features
        )

        return LIMEExplanation(
            instance=instance,
            coefficients=coefficients,
            target_class=target_class,
            prediction=pred,
            intercept=coefficients[-1] if len(coefficients) > n_features else 0.0
        )

    def _weighted_linear_regression(self, X, y, weights, n_features):
        """Solve weighted least squares: (X^T W X)^-1 X^T W y.

        Returns coefficients + intercept.
        """
        n = len(X)
        # Add intercept column
        X_aug = [row + [1.0] for row in X]
        p = n_features + 1

        # X^T W X
        XtWX = [[0.0] * p for _ in range(p)]
        XtWy = [0.0] * p

        for i in range(n):
            w = weights[i]
            for j in range(p):
                for k in range(p):
                    XtWX[j][k] += w * X_aug[i][j] * X_aug[i][k]
                XtWy[j] += w * X_aug[i][j] * y[i]

        # Regularize
        for j in range(p):
            XtWX[j][j] += 1e-6

        # Solve via Gaussian elimination
        coeffs = self._solve_linear(XtWX, XtWy, p)
        return coeffs

    def _solve_linear(self, A, b, n):
        """Gaussian elimination with partial pivoting."""
        # Augmented matrix
        M = [A[i][:] + [b[i]] for i in range(n)]

        for col in range(n):
            # Pivot
            max_row = col
            max_val = abs(M[col][col])
            for row in range(col + 1, n):
                if abs(M[row][col]) > max_val:
                    max_val = abs(M[row][col])
                    max_row = row
            M[col], M[max_row] = M[max_row], M[col]

            if abs(M[col][col]) < 1e-12:
                continue

            # Eliminate
            for row in range(col + 1, n):
                factor = M[row][col] / M[col][col]
                for k in range(col, n + 1):
                    M[row][k] -= factor * M[col][k]

        # Back-substitute
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(M[i][i]) < 1e-12:
                x[i] = 0.0
                continue
            x[i] = M[i][n]
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]

        return x


class LIMEExplanation:
    """Result of LIME explanation."""

    def __init__(self, instance, coefficients, target_class, prediction, intercept=0.0):
        self.instance = instance
        self.coefficients = coefficients[:-1] if len(coefficients) > len(instance) else coefficients
        self.intercept = intercept
        self.target_class = target_class
        self.prediction = prediction

    def top_features(self, k=5):
        """Return top-k features by absolute coefficient magnitude.

        Returns list of (feature_index, coefficient).
        """
        indexed = list(enumerate(self.coefficients))
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:k]

    def as_dict(self):
        """Return feature_index -> coefficient mapping."""
        return {i: c for i, c in enumerate(self.coefficients)}


# ============================================================
# KernelSHAP
# ============================================================

class KernelSHAP:
    """Kernel SHAP: Shapley value approximation via weighted linear regression.

    Uses the SHAP kernel to weight coalition samples, then solves
    weighted least squares to get approximate Shapley values.
    """

    def __init__(self, model, background_data, n_samples=100, rng=None):
        """
        model: trained model with predict
        background_data: 2D list, reference dataset for marginalizing
        """
        self.model = model
        self.background = _get_data_list(background_data)
        self.n_samples = n_samples
        self.rng = rng or random.Random(42)
        self.n_features = len(self.background[0]) if self.background else 0

    def explain(self, instance, target_class=None):
        """Compute SHAP values for a single instance.

        Returns: SHAPExplanation
        """
        if isinstance(instance, Tensor):
            instance = instance.data if len(instance.shape) == 1 else instance.data[0]

        M = self.n_features
        pred = _predict_single(self.model, Tensor(instance))
        if target_class is None:
            target_class = pred.index(max(pred)) if len(pred) > 1 else 0

        # Expected value (average prediction on background)
        bg_preds = []
        for bg in self.background:
            p = _predict_single(self.model, Tensor(bg))
            val = p[target_class] if len(p) > 1 else p[0]
            bg_preds.append(val)
        expected_value = sum(bg_preds) / len(bg_preds)

        # Sample coalitions and compute SHAP kernel weights
        coalitions = []
        coalition_preds = []
        kernel_weights = []

        # Always include empty and full coalitions
        for _ in range(self.n_samples):
            # Random coalition size
            size = self.rng.randint(1, M - 1) if M > 1 else 1
            # Random subset of features
            features = list(range(M))
            self.rng.shuffle(features)
            S = set(features[:size])

            mask = [1 if j in S else 0 for j in range(M)]
            coalitions.append(mask)

            # Compute E[f(x) | x_S] by averaging over background for missing features
            val = self._coalition_value(instance, mask, target_class)
            coalition_preds.append(val)

            # SHAP kernel weight: (M-1) / (C(M, |S|) * |S| * (M - |S|))
            s = sum(mask)
            if s == 0 or s == M:
                w = 1e6  # Large weight for edge cases
            else:
                # C(M, s) = M! / (s! * (M-s)!)
                comb = 1.0
                for i in range(1, min(s, M - s) + 1):
                    comb = comb * (M - i + 1) / i
                w = (M - 1) / (comb * s * (M - s))
            kernel_weights.append(w)

        # Solve weighted least squares for SHAP values
        shap_values = self._solve_shap(coalitions, coalition_preds, kernel_weights,
                                       expected_value, M)

        return SHAPExplanation(
            instance=instance,
            shap_values=shap_values,
            expected_value=expected_value,
            target_class=target_class,
            prediction=pred
        )

    def _coalition_value(self, instance, mask, target_class):
        """Compute expected prediction for coalition mask."""
        total = 0.0
        count = 0
        # Sample from background (or use all if small)
        bg_sample = self.background
        if len(bg_sample) > 10:
            indices = list(range(len(bg_sample)))
            self.rng.shuffle(indices)
            bg_sample = [self.background[i] for i in indices[:10]]

        for bg in bg_sample:
            # Replace missing features with background values
            x = [instance[j] if mask[j] else bg[j] for j in range(len(instance))]
            p = _predict_single(self.model, Tensor(x))
            val = p[target_class] if len(p) > 1 else p[0]
            total += val
            count += 1
        return total / count if count > 0 else 0.0

    def _solve_shap(self, coalitions, preds, weights, expected_value, M):
        """Solve for SHAP values via weighted linear regression."""
        n = len(coalitions)
        # Adjust predictions: subtract expected value
        y_adj = [preds[i] - expected_value for i in range(n)]

        # Weighted least squares: X^T W X beta = X^T W y
        XtWX = [[0.0] * M for _ in range(M)]
        XtWy = [0.0] * M

        for i in range(n):
            w = weights[i]
            for j in range(M):
                for k in range(M):
                    XtWX[j][k] += w * coalitions[i][j] * coalitions[i][k]
                XtWy[j] += w * coalitions[i][j] * y_adj[i]

        # Regularize
        for j in range(M):
            XtWX[j][j] += 1e-6

        # Solve
        lime = LIME.__new__(LIME)
        shap_vals = lime._solve_linear(XtWX, XtWy, M)
        return shap_vals


class SHAPExplanation:
    """Result of SHAP explanation."""

    def __init__(self, instance, shap_values, expected_value, target_class, prediction):
        self.instance = instance
        self.shap_values = shap_values
        self.expected_value = expected_value
        self.target_class = target_class
        self.prediction = prediction

    def top_features(self, k=5):
        """Top-k features by absolute SHAP value."""
        indexed = list(enumerate(self.shap_values))
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:k]

    def sum_shap(self):
        """Sum of SHAP values (should approximately equal prediction - expected)."""
        return sum(self.shap_values)

    def as_dict(self):
        return {i: v for i, v in enumerate(self.shap_values)}


# ============================================================
# IntegratedGradients
# ============================================================

class IntegratedGradients:
    """Integrated Gradients: attribution via path integral from baseline to input.

    Computes gradients at interpolated points between baseline and input,
    then averages and scales by (input - baseline).
    """

    def __init__(self, model, n_steps=50):
        self.model = model
        self.n_steps = n_steps

    def attribute(self, instance, baseline=None, target_class=None):
        """Compute integrated gradient attributions.

        instance: list of feature values
        baseline: reference point (default: zeros)
        target_class: which output to attribute (default: predicted)
        Returns: IGExplanation
        """
        if isinstance(instance, Tensor):
            instance = instance.data if len(instance.shape) == 1 else instance.data[0]

        n_features = len(instance)
        if baseline is None:
            baseline = [0.0] * n_features
        if isinstance(baseline, Tensor):
            baseline = baseline.data if len(baseline.shape) == 1 else baseline.data[0]

        pred = _predict_single(self.model, Tensor(instance))
        if target_class is None:
            target_class = pred.index(max(pred)) if len(pred) > 1 else 0

        # Compute gradients at interpolated points
        accumulated_grads = [0.0] * n_features

        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps if self.n_steps > 0 else 1.0
            # Interpolated point
            interp = [baseline[j] + alpha * (instance[j] - baseline[j])
                       for j in range(n_features)]

            # Get gradient via forward + backward
            grad = self._compute_gradient(interp, target_class)
            for j in range(n_features):
                accumulated_grads[j] += grad[j]

        # Average and scale by (input - baseline)
        attributions = [
            (accumulated_grads[j] / (self.n_steps + 1)) * (instance[j] - baseline[j])
            for j in range(n_features)
        ]

        return IGExplanation(
            instance=instance,
            baseline=baseline,
            attributions=attributions,
            target_class=target_class,
            prediction=pred
        )

    def _compute_gradient(self, x, target_class):
        """Compute gradient of output[target_class] w.r.t. input x."""
        x_tensor = Tensor([x])  # Batch of 1
        self.model.train()
        out = self.model.forward(x_tensor)
        self.model.eval()

        # Create gradient: 1.0 for target class, 0.0 elsewhere
        if len(out.shape) == 2:
            n_out = out.shape[1]
            grad = Tensor([[1.0 if j == target_class else 0.0 for j in range(n_out)]])
        else:
            grad = Tensor([1.0])

        # Backward pass to get input gradient
        input_grad = self.model.backward(grad)

        if len(input_grad.shape) == 2:
            return input_grad.data[0]
        return input_grad.data


class IGExplanation:
    """Result of Integrated Gradients."""

    def __init__(self, instance, baseline, attributions, target_class, prediction):
        self.instance = instance
        self.baseline = baseline
        self.attributions = attributions
        self.target_class = target_class
        self.prediction = prediction

    def top_features(self, k=5):
        indexed = list(enumerate(self.attributions))
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:k]

    def convergence_delta(self):
        """Check completeness axiom: sum of attributions ~ prediction diff."""
        return sum(self.attributions)

    def as_dict(self):
        return {i: v for i, v in enumerate(self.attributions)}


# ============================================================
# SaliencyMap
# ============================================================

class SaliencyMap:
    """Saliency map: raw input gradient magnitude.

    Simple but fast -- just one forward + backward pass.
    """

    def __init__(self, model):
        self.model = model

    def compute(self, instance, target_class=None):
        """Compute saliency map for an instance.

        Returns: SaliencyResult
        """
        if isinstance(instance, Tensor):
            instance = instance.data if len(instance.shape) == 1 else instance.data[0]

        pred = _predict_single(self.model, Tensor(instance))
        if target_class is None:
            target_class = pred.index(max(pred)) if len(pred) > 1 else 0

        # Forward + backward
        ig = IntegratedGradients(self.model, n_steps=0)
        grad = ig._compute_gradient(instance, target_class)

        # Saliency = absolute gradient
        saliency = [abs(g) for g in grad]

        return SaliencyResult(
            instance=instance,
            saliency=saliency,
            gradient=grad,
            target_class=target_class,
            prediction=pred
        )


class SaliencyResult:
    """Result of saliency map computation."""

    def __init__(self, instance, saliency, gradient, target_class, prediction):
        self.instance = instance
        self.saliency = saliency
        self.gradient = gradient
        self.target_class = target_class
        self.prediction = prediction

    def top_features(self, k=5):
        indexed = list(enumerate(self.saliency))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]

    def normalized(self):
        """Return saliency normalized to [0, 1]."""
        max_val = max(self.saliency) if self.saliency else 1.0
        if max_val == 0:
            return [0.0] * len(self.saliency)
        return [s / max_val for s in self.saliency]


# ============================================================
# PartialDependence
# ============================================================

class PartialDependence:
    """Partial Dependence Plot data: marginal effect of a feature.

    For feature j, varies its value over a grid while averaging
    predictions over the dataset for other features.
    """

    def __init__(self, model, X, n_grid=20):
        """
        model: trained model
        X: background dataset (2D list or Tensor)
        n_grid: number of grid points per feature
        """
        self.model = model
        self.X = _get_data_list(X)
        self.n_grid = n_grid

    def compute(self, feature, target_class=None, grid_range=None):
        """Compute PD for a single feature.

        feature: feature index
        target_class: output index (default 0)
        grid_range: (min, max) for grid, default from data
        Returns: PDResult
        """
        if target_class is None:
            target_class = 0

        # Determine grid
        col = [row[feature] for row in self.X]
        if grid_range is None:
            lo, hi = min(col), max(col)
        else:
            lo, hi = grid_range

        if lo == hi:
            hi = lo + 1.0

        step = (hi - lo) / (self.n_grid - 1) if self.n_grid > 1 else 0
        grid = [lo + i * step for i in range(self.n_grid)]

        # For each grid value, average prediction over background
        pdp_values = []
        for val in grid:
            total = 0.0
            for row in self.X:
                x = row[:]
                x[feature] = val
                p = _predict_single(self.model, Tensor(x))
                v = p[target_class] if len(p) > 1 else p[0]
                total += v
            pdp_values.append(total / len(self.X))

        return PDResult(
            feature=feature,
            grid=grid,
            values=pdp_values,
            target_class=target_class
        )

    def compute_2d(self, feature1, feature2, target_class=None, n_grid=10):
        """Compute 2D PD for two features.

        Returns: PD2DResult with grid1, grid2, values (2D).
        """
        if target_class is None:
            target_class = 0

        col1 = [row[feature1] for row in self.X]
        col2 = [row[feature2] for row in self.X]
        lo1, hi1 = min(col1), max(col1)
        lo2, hi2 = min(col2), max(col2)
        if lo1 == hi1:
            hi1 = lo1 + 1.0
        if lo2 == hi2:
            hi2 = lo2 + 1.0

        step1 = (hi1 - lo1) / (n_grid - 1) if n_grid > 1 else 0
        step2 = (hi2 - lo2) / (n_grid - 1) if n_grid > 1 else 0
        grid1 = [lo1 + i * step1 for i in range(n_grid)]
        grid2 = [lo2 + i * step2 for i in range(n_grid)]

        values = []
        for v1 in grid1:
            row_vals = []
            for v2 in grid2:
                total = 0.0
                for row in self.X:
                    x = row[:]
                    x[feature1] = v1
                    x[feature2] = v2
                    p = _predict_single(self.model, Tensor(x))
                    v = p[target_class] if len(p) > 1 else p[0]
                    total += v
                row_vals.append(total / len(self.X))
            values.append(row_vals)

        return PD2DResult(
            feature1=feature1,
            feature2=feature2,
            grid1=grid1,
            grid2=grid2,
            values=values,
            target_class=target_class
        )


class PDResult:
    """Partial dependence result for one feature."""

    def __init__(self, feature, grid, values, target_class):
        self.feature = feature
        self.grid = grid
        self.values = values
        self.target_class = target_class

    def range(self):
        """Range of PD values."""
        return max(self.values) - min(self.values)

    def monotonic(self):
        """Check if PD is monotonically increasing or decreasing."""
        if len(self.values) < 2:
            return True
        diffs = [self.values[i + 1] - self.values[i] for i in range(len(self.values) - 1)]
        return all(d >= -1e-10 for d in diffs) or all(d <= 1e-10 for d in diffs)


class PD2DResult:
    """2D partial dependence result."""

    def __init__(self, feature1, feature2, grid1, grid2, values, target_class):
        self.feature1 = feature1
        self.feature2 = feature2
        self.grid1 = grid1
        self.grid2 = grid2
        self.values = values
        self.target_class = target_class


# ============================================================
# ICE: Individual Conditional Expectation
# ============================================================

class ICE:
    """Individual Conditional Expectation curves.

    Like PDP but per-instance -- shows how each sample's prediction
    changes as a feature varies.
    """

    def __init__(self, model, X, n_grid=20):
        self.model = model
        self.X = _get_data_list(X)
        self.n_grid = n_grid

    def compute(self, feature, target_class=None, grid_range=None, max_instances=None):
        """Compute ICE curves for all instances.

        Returns: ICEResult
        """
        if target_class is None:
            target_class = 0

        col = [row[feature] for row in self.X]
        if grid_range is None:
            lo, hi = min(col), max(col)
        else:
            lo, hi = grid_range
        if lo == hi:
            hi = lo + 1.0

        step = (hi - lo) / (self.n_grid - 1) if self.n_grid > 1 else 0
        grid = [lo + i * step for i in range(self.n_grid)]

        instances = self.X
        if max_instances and len(instances) > max_instances:
            instances = instances[:max_instances]

        curves = []
        for row in instances:
            curve = []
            for val in grid:
                x = row[:]
                x[feature] = val
                p = _predict_single(self.model, Tensor(x))
                v = p[target_class] if len(p) > 1 else p[0]
                curve.append(v)
            curves.append(curve)

        # Centered ICE (c-ICE): subtract baseline
        centered = []
        for curve in curves:
            base = curve[0]
            centered.append([v - base for v in curve])

        return ICEResult(
            feature=feature,
            grid=grid,
            curves=curves,
            centered_curves=centered,
            target_class=target_class
        )


class ICEResult:
    """ICE computation result."""

    def __init__(self, feature, grid, curves, centered_curves, target_class):
        self.feature = feature
        self.grid = grid
        self.curves = curves
        self.centered_curves = centered_curves
        self.target_class = target_class

    def average(self):
        """Average of ICE curves (= PDP)."""
        n = len(self.curves)
        if n == 0:
            return []
        return [sum(self.curves[i][j] for i in range(n)) / n
                for j in range(len(self.grid))]

    def heterogeneity(self):
        """Standard deviation of ICE curves at each grid point."""
        n = len(self.curves)
        if n == 0:
            return []
        avg = self.average()
        return [
            math.sqrt(sum((self.curves[i][j] - avg[j]) ** 2 for i in range(n)) / n)
            for j in range(len(self.grid))
        ]


# ============================================================
# CounterfactualExplainer
# ============================================================

class CounterfactualExplainer:
    """Find minimal perturbation to change prediction.

    Uses gradient-guided search to find the closest input that
    produces a different prediction.
    """

    def __init__(self, model, max_iterations=100, step_size=0.1, rng=None):
        self.model = model
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.rng = rng or random.Random(42)

    def explain(self, instance, target_class=None):
        """Find counterfactual for instance.

        instance: list of feature values
        target_class: desired class for counterfactual (default: any different)
        Returns: CounterfactualResult
        """
        if isinstance(instance, Tensor):
            instance = instance.data if len(instance.shape) == 1 else instance.data[0]

        original_class = _predict_class(self.model, Tensor(instance))
        n_features = len(instance)

        # Start from instance, perturb toward target
        cf = instance[:]
        best_cf = None
        best_dist = float('inf')

        for iteration in range(self.max_iterations):
            current_class = _predict_class(self.model, Tensor(cf))

            if target_class is not None:
                found = (current_class == target_class)
            else:
                found = (current_class != original_class)

            if found:
                dist = sum((cf[j] - instance[j]) ** 2 for j in range(n_features))
                if dist < best_dist:
                    best_dist = dist
                    best_cf = cf[:]

            # Gradient-guided perturbation
            ig = IntegratedGradients(self.model, n_steps=0)
            desired = target_class if target_class is not None else (
                1 - original_class if original_class <= 1 else 0
            )
            grad = ig._compute_gradient(cf, desired)

            # Move in gradient direction + some noise
            for j in range(n_features):
                cf[j] += self.step_size * grad[j]
                cf[j] += self.rng.gauss(0, 0.01)

            # Elastic net toward original (sparsity + proximity)
            lam = 0.01
            for j in range(n_features):
                cf[j] -= lam * (cf[j] - instance[j])

        if best_cf is None:
            # Last resort: return current state
            best_cf = cf[:]
            best_dist = sum((cf[j] - instance[j]) ** 2 for j in range(n_features))

        changes = {j: best_cf[j] - instance[j]
                   for j in range(n_features) if abs(best_cf[j] - instance[j]) > 1e-6}

        return CounterfactualResult(
            instance=instance,
            counterfactual=best_cf,
            original_class=original_class,
            cf_class=_predict_class(self.model, Tensor(best_cf)),
            distance=math.sqrt(best_dist),
            changes=changes
        )


class CounterfactualResult:
    """Result of counterfactual search."""

    def __init__(self, instance, counterfactual, original_class, cf_class, distance, changes):
        self.instance = instance
        self.counterfactual = counterfactual
        self.original_class = original_class
        self.cf_class = cf_class
        self.distance = distance
        self.changes = changes  # feature_index -> delta

    def success(self):
        """Whether counterfactual changed the class."""
        return self.cf_class != self.original_class

    def sparsity(self):
        """Number of features changed."""
        return len(self.changes)

    def top_changes(self, k=5):
        """Top-k changes by absolute magnitude."""
        items = sorted(self.changes.items(), key=lambda x: abs(x[1]), reverse=True)
        return items[:k]


# ============================================================
# FeatureInteraction
# ============================================================

class FeatureInteraction:
    """H-statistic for measuring feature interaction strength.

    Friedman's H-statistic measures the fraction of variance in
    the joint partial dependence not captured by individual PDPs.
    """

    def __init__(self, model, X, n_grid=10):
        self.model = model
        self.X = _get_data_list(X)
        self.n_grid = n_grid
        self._pd_cache = {}

    def h_statistic(self, feature1, feature2, target_class=None):
        """Compute H-statistic for two features.

        Returns float in [0, 1]: 0 = no interaction, 1 = pure interaction.
        """
        if target_class is None:
            target_class = 0

        pd = PartialDependence(self.model, self.X, n_grid=self.n_grid)

        # Individual PDPs
        pd1 = pd.compute(feature1, target_class)
        pd2 = pd.compute(feature2, target_class)

        # 2D PDP
        pd12 = pd.compute_2d(feature1, feature2, target_class, n_grid=self.n_grid)

        # H = var(PD_12 - PD_1 - PD_2) / var(PD_12)
        # Average pd1 and pd2 over their grids for centering
        mean1 = sum(pd1.values) / len(pd1.values)
        mean2 = sum(pd2.values) / len(pd2.values)

        # Build lookup for individual PD values at closest grid points
        def interp1(val, grid, values):
            # Nearest grid point
            best_idx = 0
            best_dist = abs(val - grid[0])
            for i in range(1, len(grid)):
                d = abs(val - grid[i])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            return values[best_idx]

        numerator = 0.0
        denominator = 0.0
        count = 0

        for i, v1 in enumerate(pd12.grid1):
            for j, v2 in enumerate(pd12.grid2):
                f12 = pd12.values[i][j]
                f1 = interp1(v1, pd1.grid, pd1.values)
                f2 = interp1(v2, pd2.grid, pd2.values)

                interaction = f12 - f1 - f2 + mean1 + mean2 - f12
                # Simplified: interaction = -(f1 + f2 - mean1 - mean2)
                # Actually: H measures departure from additivity
                # Correct: interaction_ij = PD_12(i,j) - PD_1(i) - PD_2(j)
                interaction = f12 - f1 - f2
                numerator += interaction ** 2
                denominator += f12 ** 2
                count += 1

        if denominator < 1e-12:
            return 0.0

        return numerator / denominator

    def all_pairs(self, target_class=None):
        """Compute H-statistic for all feature pairs.

        Returns: dict of (i, j) -> H value
        """
        n_features = len(self.X[0]) if self.X else 0
        results = {}
        for i in range(n_features):
            for j in range(i + 1, n_features):
                results[(i, j)] = self.h_statistic(i, j, target_class)
        return results

    def top_interactions(self, k=5, target_class=None):
        """Top-k feature interactions by H-statistic."""
        pairs = self.all_pairs(target_class)
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:k]


# ============================================================
# ModelSummary
# ============================================================

class ModelSummary:
    """Analyze model weights and structure.

    Provides weight statistics, layer-by-layer analysis, and
    identifies potential issues (dead neurons, large weights).
    """

    def __init__(self, model):
        self.model = model

    def layer_stats(self):
        """Get statistics for each layer's parameters.

        Returns: list of LayerStats
        """
        results = []
        if not hasattr(self.model, 'layers'):
            return results

        for i, layer in enumerate(self.model.layers):
            params = layer.get_params()
            if not params:
                results.append(LayerStats(
                    index=i,
                    type=type(layer).__name__,
                    weight_stats=None,
                    bias_stats=None,
                    n_params=0
                ))
                continue

            weight_stats = None
            bias_stats = None
            n_params = 0

            for param_tuple in params:
                tensor = param_tuple[0]
                label = param_tuple[2] if len(param_tuple) > 2 else 'weight'

                flat = self._flatten(tensor)
                n_params += len(flat)
                stats = self._compute_stats(flat)

                if label == 'weight':
                    weight_stats = stats
                elif label == 'bias':
                    bias_stats = stats

            results.append(LayerStats(
                index=i,
                type=type(layer).__name__,
                weight_stats=weight_stats,
                bias_stats=bias_stats,
                n_params=n_params
            ))

        return results

    def dead_neurons(self, X, threshold=0.0):
        """Find neurons that are always inactive (output <= threshold).

        X: input data to test
        Returns: list of (layer_index, neuron_index) for dead neurons
        """
        X_data = _get_data_list(X)
        if not X_data:
            return []

        dead = []
        if not hasattr(self.model, 'layers'):
            return dead

        # Run forward to get activations
        x = Tensor(X_data)
        activations = []

        self.model.eval()
        current = x
        for i, layer in enumerate(self.model.layers):
            current = layer.forward(current)
            activations.append(current)

        # Check each activation layer output
        for i, act in enumerate(activations):
            if not isinstance(self.model.layers[i], Activation):
                continue
            if self.model.layers[i].name not in ('relu', 'leaky_relu'):
                continue

            if len(act.shape) == 2:
                n_neurons = act.shape[1]
                for j in range(n_neurons):
                    col = [act.data[r][j] for r in range(act.shape[0])]
                    if all(v <= threshold for v in col):
                        dead.append((i, j))
            elif len(act.shape) == 1:
                for j in range(len(act.data)):
                    if act.data[j] <= threshold:
                        dead.append((i, j))

        return dead

    def weight_norms(self):
        """L2 norm of weights per layer."""
        norms = []
        if not hasattr(self.model, 'layers'):
            return norms
        for layer in self.model.layers:
            params = layer.get_params()
            for param_tuple in params:
                tensor = param_tuple[0]
                label = param_tuple[2] if len(param_tuple) > 2 else 'weight'
                if label == 'weight':
                    flat = self._flatten(tensor)
                    norm = math.sqrt(sum(v ** 2 for v in flat))
                    norms.append(norm)
        return norms

    def large_weights(self, threshold=5.0):
        """Find weights with absolute value > threshold.

        Returns: list of (layer_index, weight_value)
        """
        results = []
        if not hasattr(self.model, 'layers'):
            return results
        for i, layer in enumerate(self.model.layers):
            params = layer.get_params()
            for param_tuple in params:
                tensor = param_tuple[0]
                flat = self._flatten(tensor)
                for v in flat:
                    if abs(v) > threshold:
                        results.append((i, v))
        return results

    def total_params(self):
        """Total number of trainable parameters."""
        total = 0
        if not hasattr(self.model, 'layers'):
            return total
        for layer in self.model.layers:
            for param_tuple in layer.get_params():
                total += len(self._flatten(param_tuple[0]))
        return total

    def _flatten(self, tensor):
        """Flatten tensor to 1D list."""
        if len(tensor.shape) == 1:
            return tensor.data
        flat = []
        for row in tensor.data:
            flat.extend(row)
        return flat

    def _compute_stats(self, values):
        """Compute basic statistics."""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'norm': 0}
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        return {
            'mean': mean,
            'std': std,
            'min': min(values),
            'max': max(values),
            'norm': math.sqrt(sum(v ** 2 for v in values)),
            'n': len(values)
        }


class LayerStats:
    """Statistics for a single layer."""

    def __init__(self, index, type, weight_stats, bias_stats, n_params):
        self.index = index
        self.type = type
        self.weight_stats = weight_stats
        self.bias_stats = bias_stats
        self.n_params = n_params


# ============================================================
# ExplanationComparator
# ============================================================

class ExplanationComparator:
    """Compare explanations from different methods.

    Useful for validating that different explainability methods
    agree on feature importance ranking.
    """

    def __init__(self):
        self.explanations = {}

    def add(self, name, feature_importances):
        """Add an explanation.

        feature_importances: dict of feature_index -> importance value
        """
        self.explanations[name] = feature_importances

    def rank_correlation(self, name1, name2):
        """Spearman rank correlation between two explanations.

        Returns: correlation coefficient in [-1, 1].
        """
        if name1 not in self.explanations or name2 not in self.explanations:
            raise ValueError("Unknown explanation name")

        imp1 = self.explanations[name1]
        imp2 = self.explanations[name2]

        # Common features
        features = sorted(set(imp1.keys()) & set(imp2.keys()))
        if len(features) < 2:
            return 0.0

        # Rank by absolute importance
        vals1 = [abs(imp1[f]) for f in features]
        vals2 = [abs(imp2[f]) for f in features]

        rank1 = self._rank(vals1)
        rank2 = self._rank(vals2)

        # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
        n = len(features)
        d_sq = sum((rank1[i] - rank2[i]) ** 2 for i in range(n))
        return 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))

    def agreement_top_k(self, name1, name2, k=3):
        """Fraction of top-k features that agree between two methods."""
        imp1 = self.explanations[name1]
        imp2 = self.explanations[name2]

        top1 = sorted(imp1.keys(), key=lambda f: abs(imp1[f]), reverse=True)[:k]
        top2 = sorted(imp2.keys(), key=lambda f: abs(imp2[f]), reverse=True)[:k]

        overlap = len(set(top1) & set(top2))
        return overlap / k

    def summary(self):
        """Summary of all explanations' top features."""
        result = {}
        for name, imp in self.explanations.items():
            top = sorted(imp.keys(), key=lambda f: abs(imp[f]), reverse=True)[:5]
            result[name] = [(f, imp[f]) for f in top]
        return result

    def _rank(self, values):
        """Assign ranks (1-based, average for ties)."""
        n = len(values)
        indexed = sorted(range(n), key=lambda i: values[i], reverse=True)
        ranks = [0.0] * n
        for rank_pos, idx in enumerate(indexed):
            ranks[idx] = rank_pos + 1.0
        return ranks


# ============================================================
# FeatureAblation
# ============================================================

class FeatureAblation:
    """Feature importance via ablation (zeroing out features).

    Simpler than permutation -- just sets features to a reference value
    and measures prediction change.
    """

    def __init__(self, model, baseline_value=0.0):
        self.model = model
        self.baseline_value = baseline_value

    def compute(self, instance, target_class=None):
        """Compute ablation importance for each feature.

        Returns: AblationResult
        """
        if isinstance(instance, Tensor):
            instance = instance.data if len(instance.shape) == 1 else instance.data[0]

        pred = _predict_single(self.model, Tensor(instance))
        if target_class is None:
            target_class = pred.index(max(pred)) if len(pred) > 1 else 0

        base_val = pred[target_class] if len(pred) > 1 else pred[0]
        importances = []

        for j in range(len(instance)):
            ablated = instance[:]
            ablated[j] = self.baseline_value
            p = _predict_single(self.model, Tensor(ablated))
            val = p[target_class] if len(p) > 1 else p[0]
            importances.append(base_val - val)

        return AblationResult(
            instance=instance,
            importances=importances,
            target_class=target_class,
            base_prediction=base_val
        )


class AblationResult:
    """Result of feature ablation."""

    def __init__(self, instance, importances, target_class, base_prediction):
        self.instance = instance
        self.importances = importances
        self.target_class = target_class
        self.base_prediction = base_prediction

    def top_features(self, k=5):
        indexed = list(enumerate(self.importances))
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed[:k]

    def as_dict(self):
        return {i: v for i, v in enumerate(self.importances)}
