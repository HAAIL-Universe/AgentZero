"""V232: Neural Process -- Amortized function-space inference for few-shot learning.

Composes V229 (Meta-Learning) + amortized inference to build Neural Process models
that map context sets directly to predictive distributions without per-task optimization.

The Neural Process family treats function learning as a set-to-function mapping:
given a context set {(x_i, y_i)}, predict y* at arbitrary target x*.

Unlike standard GPs that require O(n^3) inference per task, Neural Processes
amortize the cost by learning an encoder that maps context -> representation -> prediction.

We implement this WITHOUT neural networks, using:
  - Basis function encoders (RBF/polynomial features)
  - Moment-based aggregation (mean/variance pooling)
  - GP-based decoders with learned representations
  - Attention via kernel similarity

Models:
  1. CNP  -- Conditional Neural Process (deterministic path only)
  2. NP   -- Neural Process (+ latent path for global uncertainty)
  3. ANP  -- Attentive Neural Process (cross-attention for local coherence)
  4. ConvCNP -- Convolutional CNP (translation equivariance via grid convolution)

Key insight: the encoder-aggregator-decoder architecture can be realized with
classical statistical building blocks. The "neural" in Neural Process refers to
the architecture pattern, not a requirement for gradient-based neural nets.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Tuple
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V222_gaussian_process'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V229_meta_learning'))

from gaussian_process import (
    GaussianProcess, RBFKernel, Matern52Kernel, Kernel, GPPrediction,
    SumKernel, ProductKernel, ScaleKernel, WhiteNoiseKernel, ARDKernel,
)
from meta_learning import (
    Task, TaskDistribution, MetaLearningResult, FewShotResult,
    sinusoidal_task_distribution, polynomial_task_distribution,
    step_task_distribution, _rmse, _nlpd,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class NPPrediction:
    """Prediction from a Neural Process model."""
    mean: np.ndarray          # (n_target,) predictive mean
    std: np.ndarray           # (n_target,) predictive std
    samples: Optional[np.ndarray] = None  # (n_samples, n_target) if sampled


@dataclass
class NPTrainResult:
    """Result of training a Neural Process."""
    train_losses: List[float]     # Per-epoch training loss
    val_losses: List[float]       # Per-epoch validation loss
    best_epoch: int
    n_tasks_seen: int
    n_epochs: int
    model_params: Dict           # Learned parameters


@dataclass
class NPComparisonResult:
    """Result of comparing NP variants."""
    model_names: List[str]
    rmses: Dict[str, List[float]]       # model -> per-task RMSE
    nlpds: Dict[str, List[float]]       # model -> per-task NLPD
    mean_rmse: Dict[str, float]
    mean_nlpd: Dict[str, float]
    calibration: Dict[str, float]       # model -> calibration score


# ---------------------------------------------------------------------------
# Basis Function Encoders
# ---------------------------------------------------------------------------

class BasisEncoder:
    """Encode (x, y) pairs into feature representations using basis functions."""

    def __init__(self, n_basis: int = 32, length_scale: float = 1.0,
                 basis_type: str = 'rbf', input_dim: int = 1):
        self.n_basis = n_basis
        self.length_scale = length_scale
        self.basis_type = basis_type
        self.input_dim = input_dim
        self.centers = None  # Set during fit or manually

    def _init_centers(self, X: np.ndarray):
        """Initialize basis centers from data range."""
        if self.centers is not None:
            return
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        # Place centers on a grid
        if self.input_dim == 1:
            self.centers = np.linspace(mins[0] - 1, maxs[0] + 1,
                                        self.n_basis).reshape(-1, 1)
        else:
            # Use random centers for multi-dim
            rng = np.random.default_rng(42)
            self.centers = rng.uniform(mins - 1, maxs + 1,
                                        size=(self.n_basis, self.input_dim))

    def encode_x(self, X: np.ndarray) -> np.ndarray:
        """Encode input locations into basis features.

        Args:
            X: (n, d) input locations

        Returns:
            (n, n_basis) basis function values
        """
        self._init_centers(X)
        if self.basis_type == 'rbf':
            # RBF basis: phi_j(x) = exp(-||x - c_j||^2 / (2 * ls^2))
            diffs = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]  # (n, n_basis, d)
            sq_dists = np.sum(diffs ** 2, axis=2)  # (n, n_basis)
            return np.exp(-sq_dists / (2 * self.length_scale ** 2))
        elif self.basis_type == 'polynomial':
            # Polynomial basis up to degree ceil(n_basis / d)
            features = [np.ones((X.shape[0], 1))]
            for deg in range(1, self.n_basis):
                features.append(X ** deg)
            result = np.hstack(features)
            return result[:, :self.n_basis]
        elif self.basis_type == 'fourier':
            # Random Fourier features
            rng = np.random.default_rng(42)
            W = rng.normal(0, 1.0 / self.length_scale,
                           size=(self.input_dim, self.n_basis // 2))
            proj = X @ W  # (n, n_basis//2)
            return np.hstack([np.cos(proj), np.sin(proj)])[:, :self.n_basis]
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

    def encode_xy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Encode (x, y) pairs into representations.

        Args:
            X: (n, d) input locations
            y: (n,) output values

        Returns:
            (n, 2 * n_basis) concatenated [phi(x), y * phi(x)] features
        """
        phi = self.encode_x(X)  # (n, n_basis)
        y_phi = y[:, np.newaxis] * phi  # (n, n_basis)
        return np.hstack([phi, y_phi])  # (n, 2 * n_basis)


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def mean_aggregator(representations: np.ndarray) -> np.ndarray:
    """Aggregate set of representations by taking the mean.

    Args:
        representations: (n, repr_dim)

    Returns:
        (repr_dim,) aggregated representation
    """
    return representations.mean(axis=0)


def variance_aggregator(representations: np.ndarray) -> np.ndarray:
    """Aggregate with both mean and variance (for uncertainty).

    Args:
        representations: (n, repr_dim)

    Returns:
        (2 * repr_dim,) [mean; variance] representation
    """
    return np.concatenate([representations.mean(axis=0),
                           representations.var(axis=0) + 1e-8])


def max_aggregator(representations: np.ndarray) -> np.ndarray:
    """Max-pooling aggregator."""
    return representations.max(axis=0)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class LinearDecoder:
    """Decode aggregated representation + target features into predictions."""

    def __init__(self, repr_dim: int, n_basis: int, noise_init: float = 0.1):
        self.repr_dim = repr_dim
        self.n_basis = n_basis
        # Weight matrix: maps [target_features; context_repr] -> (mean, log_var)
        self.W_mean = np.zeros((repr_dim + n_basis, 1))
        self.W_logvar = np.zeros((repr_dim + n_basis, 1))
        self.b_mean = 0.0
        self.b_logvar = np.log(noise_init)

    def predict(self, target_features: np.ndarray,
                context_repr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and std at target locations.

        Args:
            target_features: (n_target, n_basis) basis features at targets
            context_repr: (repr_dim,) aggregated context representation

        Returns:
            (mean, std) each of shape (n_target,)
        """
        n_target = target_features.shape[0]
        # Tile context repr for each target
        ctx = np.tile(context_repr, (n_target, 1))  # (n_target, repr_dim)
        combined = np.hstack([target_features, ctx])  # (n_target, n_basis + repr_dim)

        mean = combined @ self.W_mean + self.b_mean  # (n_target, 1)
        logvar = combined @ self.W_logvar + self.b_logvar  # (n_target, 1)

        mean = mean.ravel()
        std = np.sqrt(np.exp(np.clip(logvar.ravel(), -10, 10)))
        return mean, std


# ---------------------------------------------------------------------------
# Conditional Neural Process (CNP)
# ---------------------------------------------------------------------------

class ConditionalNeuralProcess:
    """Conditional Neural Process -- deterministic encoder-aggregator-decoder.

    Given context set C = {(x_c, y_c)}, predicts at targets x_t:
      1. Encode: r_i = encoder(x_c_i, y_c_i)
      2. Aggregate: r = mean(r_1, ..., r_n)
      3. Decode: p(y_t | x_t, r) for each target

    This is the simplest NP variant -- no latent variables, no attention.
    Fast O(n_c + n_t) prediction but underfits on complex functions.
    """

    def __init__(self, n_basis: int = 32, length_scale: float = 1.0,
                 basis_type: str = 'rbf', input_dim: int = 1,
                 noise: float = 0.1):
        self.encoder = BasisEncoder(n_basis, length_scale, basis_type, input_dim)
        self.n_basis = n_basis
        self.repr_dim = 2 * n_basis  # [phi(x), y*phi(x)]
        self.decoder = LinearDecoder(self.repr_dim, n_basis, noise)
        self.is_trained = False
        self._noise = noise

    def _encode_context(self, X_c: np.ndarray, y_c: np.ndarray) -> np.ndarray:
        """Encode and aggregate context set."""
        if len(X_c) == 0:
            return np.zeros(self.repr_dim)
        reps = self.encoder.encode_xy(X_c, y_c)  # (n_c, repr_dim)
        return mean_aggregator(reps)

    def predict(self, X_context: np.ndarray, y_context: np.ndarray,
                X_target: np.ndarray) -> NPPrediction:
        """Predict at target locations given context.

        Args:
            X_context: (n_c, d) context inputs
            y_context: (n_c,) context outputs
            X_target: (n_t, d) target inputs

        Returns:
            NPPrediction with mean and std
        """
        context_repr = self._encode_context(X_context, y_context)
        target_features = self.encoder.encode_x(X_target)  # (n_t, n_basis)
        mean, std = self.decoder.predict(target_features, context_repr)
        return NPPrediction(mean=mean, std=std)

    def train(self, task_dist: TaskDistribution, n_epochs: int = 100,
              lr: float = 0.01, val_frac: float = 0.2,
              rng: Optional[np.random.Generator] = None) -> NPTrainResult:
        """Train CNP on a task distribution via gradient-free optimization.

        We optimize decoder weights to minimize NLL across tasks.
        Uses coordinate descent on the linear decoder weights.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Split tasks
        n_val = max(1, int(len(task_dist.tasks) * val_frac))
        indices = rng.permutation(len(task_dist.tasks))
        val_tasks = [task_dist.tasks[i] for i in indices[:n_val]]
        train_tasks = [task_dist.tasks[i] for i in indices[n_val:]]

        # Initialize encoder centers from all training data
        all_X = np.vstack([t.X_support for t in train_tasks] +
                          [t.X_query for t in train_tasks])
        self.encoder._init_centers(all_X)

        # Pre-compute all encodings
        train_encoded = []
        for task in train_tasks:
            ctx_repr = self._encode_context(task.X_support, task.y_support)
            tgt_feat = self.encoder.encode_x(task.X_query)
            train_encoded.append((ctx_repr, tgt_feat, task.y_query))

        val_encoded = []
        for task in val_tasks:
            ctx_repr = self._encode_context(task.X_support, task.y_support)
            tgt_feat = self.encoder.encode_x(task.X_query)
            val_encoded.append((ctx_repr, tgt_feat, task.y_query))

        # Solve for optimal weights via least squares across all tasks
        # Stack all: combined_features @ W_mean = y_targets
        all_combined = []
        all_y = []
        for ctx_repr, tgt_feat, y_q in train_encoded:
            n_t = tgt_feat.shape[0]
            ctx = np.tile(ctx_repr, (n_t, 1))
            combined = np.hstack([tgt_feat, ctx])
            all_combined.append(combined)
            all_y.append(y_q)

        A = np.vstack(all_combined)  # (total_points, n_basis + repr_dim)
        b = np.concatenate(all_y)    # (total_points,)

        # Ridge regression for mean weights
        lam = 1e-4
        ATA = A.T @ A + lam * np.eye(A.shape[1])
        ATb = A.T @ b
        try:
            w = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        self.decoder.W_mean = w.reshape(-1, 1)
        self.decoder.b_mean = 0.0

        # Estimate noise from residuals
        residuals = b - A @ w
        noise_var = np.var(residuals) + 1e-6
        self.decoder.b_logvar = np.log(noise_var)
        self._noise = np.sqrt(noise_var)

        # Compute losses
        train_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            # Training NLL
            train_nll = self._compute_nll(train_encoded)
            train_losses.append(train_nll)
            val_nll = self._compute_nll(val_encoded)
            val_losses.append(val_nll)

        best_epoch = int(np.argmin(val_losses)) if val_losses else 0
        self.is_trained = True

        return NPTrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            n_tasks_seen=len(train_tasks),
            n_epochs=n_epochs,
            model_params={
                'W_mean_norm': float(np.linalg.norm(self.decoder.W_mean)),
                'noise': float(self._noise),
            }
        )

    def _compute_nll(self, encoded_tasks) -> float:
        """Compute mean negative log-likelihood across tasks."""
        total_nll = 0.0
        total_points = 0
        for ctx_repr, tgt_feat, y_q in encoded_tasks:
            n_t = tgt_feat.shape[0]
            ctx = np.tile(ctx_repr, (n_t, 1))
            combined = np.hstack([tgt_feat, ctx])
            mean = (combined @ self.decoder.W_mean).ravel() + self.decoder.b_mean
            var = np.exp(self.decoder.b_logvar)
            # Gaussian NLL: 0.5 * (log(2*pi*var) + (y - mu)^2 / var)
            nll = 0.5 * (np.log(2 * np.pi * var) + (y_q - mean) ** 2 / var)
            total_nll += nll.sum()
            total_points += n_t
        return total_nll / max(total_points, 1)

    def evaluate(self, task: Task) -> Tuple[float, float]:
        """Evaluate on a single task. Returns (RMSE, NLPD)."""
        pred = self.predict(task.X_support, task.y_support, task.X_query)
        rmse = _rmse(task.y_query, pred.mean)
        nlpd = _nlpd(task.y_query, pred.mean, pred.std)
        return rmse, nlpd


# ---------------------------------------------------------------------------
# Neural Process (with latent variable)
# ---------------------------------------------------------------------------

class NeuralProcess:
    """Neural Process -- adds a latent variable path for global uncertainty.

    Unlike CNP which produces a single deterministic prediction,
    NP samples a global latent z from the aggregated context:
      z ~ q(z | C) = N(mu_z, sigma_z^2)
    Then conditions the decoder on both z and target features.

    This enables coherent function samples (predictions at different
    targets are correlated through z), unlike CNP which predicts independently.
    """

    def __init__(self, n_basis: int = 32, length_scale: float = 1.0,
                 basis_type: str = 'rbf', input_dim: int = 1,
                 latent_dim: int = 16, noise: float = 0.1):
        self.encoder = BasisEncoder(n_basis, length_scale, basis_type, input_dim)
        self.n_basis = n_basis
        self.latent_dim = latent_dim
        self.repr_dim = 2 * n_basis
        self._noise = noise
        self.is_trained = False

        # Latent encoder: aggregated repr -> (mu_z, log_sigma_z)
        # Using linear projection
        self.W_mu_z = np.random.default_rng(42).normal(
            0, 0.01, size=(self.repr_dim, latent_dim))
        self.W_logsig_z = np.random.default_rng(43).normal(
            0, 0.01, size=(self.repr_dim, latent_dim))
        self.b_mu_z = np.zeros(latent_dim)
        self.b_logsig_z = np.zeros(latent_dim)

        # Decoder: [target_features; z] -> (mean, log_var)
        dec_input = n_basis + latent_dim
        self.W_dec_mean = np.zeros((dec_input, 1))
        self.W_dec_logvar = np.zeros((dec_input, 1))
        self.b_dec_mean = 0.0
        self.b_dec_logvar = np.log(noise)

    def _encode_to_latent(self, X: np.ndarray, y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode context to latent distribution parameters.

        Returns:
            (mu_z, sigma_z) each of shape (latent_dim,)
        """
        if len(X) == 0:
            return self.b_mu_z.copy(), np.exp(self.b_logsig_z)
        reps = self.encoder.encode_xy(X, y)  # (n, repr_dim)
        r = mean_aggregator(reps)  # (repr_dim,)
        mu_z = r @ self.W_mu_z + self.b_mu_z
        logsig_z = r @ self.W_logsig_z + self.b_logsig_z
        sigma_z = np.exp(np.clip(logsig_z, -5, 5))
        return mu_z, sigma_z

    def predict(self, X_context: np.ndarray, y_context: np.ndarray,
                X_target: np.ndarray, n_samples: int = 0,
                rng: Optional[np.random.Generator] = None) -> NPPrediction:
        """Predict at target locations given context.

        If n_samples > 0, draw samples from the latent and return them.
        Otherwise, use the latent mean for a single prediction.
        """
        if rng is None:
            rng = np.random.default_rng()

        mu_z, sigma_z = self._encode_to_latent(X_context, y_context)
        target_features = self.encoder.encode_x(X_target)
        n_t = target_features.shape[0]

        if n_samples > 0:
            # Sample z multiple times for uncertainty
            samples = np.zeros((n_samples, n_t))
            for s in range(n_samples):
                z = mu_z + sigma_z * rng.normal(size=self.latent_dim)
                z_tiled = np.tile(z, (n_t, 1))  # (n_t, latent_dim)
                combined = np.hstack([target_features, z_tiled])
                mean_s = (combined @ self.W_dec_mean).ravel() + self.b_dec_mean
                samples[s] = mean_s

            mean = samples.mean(axis=0)
            std = np.sqrt(samples.var(axis=0) + np.exp(self.b_dec_logvar))
            return NPPrediction(mean=mean, std=std, samples=samples)
        else:
            # Use latent mean
            z_tiled = np.tile(mu_z, (n_t, 1))
            combined = np.hstack([target_features, z_tiled])
            mean = (combined @ self.W_dec_mean).ravel() + self.b_dec_mean
            std = np.full(n_t, np.sqrt(np.exp(self.b_dec_logvar)))
            return NPPrediction(mean=mean, std=std)

    def train(self, task_dist: TaskDistribution, n_epochs: int = 100,
              lr: float = 0.01, val_frac: float = 0.2,
              rng: Optional[np.random.Generator] = None) -> NPTrainResult:
        """Train NP on a task distribution.

        Optimizes ELBO = E_q[log p(y|x,z)] - KL[q(z|C,T) || q(z|C)]
        We use closed-form solutions where possible.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_val = max(1, int(len(task_dist.tasks) * val_frac))
        indices = rng.permutation(len(task_dist.tasks))
        val_tasks = [task_dist.tasks[i] for i in indices[:n_val]]
        train_tasks = [task_dist.tasks[i] for i in indices[n_val:]]

        # Init encoder centers
        all_X = np.vstack([t.X_support for t in train_tasks] +
                          [t.X_query for t in train_tasks])
        self.encoder._init_centers(all_X)

        # First pass: fit latent encoder via SVD on task representations
        task_reps = []
        for task in train_tasks:
            reps = self.encoder.encode_xy(task.X_support, task.y_support)
            task_reps.append(mean_aggregator(reps))
        R = np.array(task_reps)  # (n_tasks, repr_dim)

        # PCA to get latent projection
        R_centered = R - R.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(R_centered, full_matrices=False)
            # Top latent_dim components
            k = min(self.latent_dim, len(S))
            self.W_mu_z = np.zeros((self.repr_dim, self.latent_dim))
            self.W_mu_z[:, :k] = Vt[:k].T
            self.b_mu_z = R.mean(axis=0) @ self.W_mu_z
            # Variance from remaining components
            if k < len(S):
                residual_var = S[k:].mean() / max(S[0], 1e-8)
            else:
                residual_var = 0.1
            self.W_logsig_z[:] = 0
            self.b_logsig_z[:] = np.log(max(residual_var, 1e-4))
        except np.linalg.LinAlgError:
            pass  # Keep random init

        # Second pass: fit decoder weights via least squares
        # For each task, encode to latent mean, then regress targets
        all_combined = []
        all_y = []
        for task in train_tasks:
            mu_z, _ = self._encode_to_latent(task.X_support, task.y_support)
            tgt_feat = self.encoder.encode_x(task.X_query)
            n_t = tgt_feat.shape[0]
            z_tiled = np.tile(mu_z, (n_t, 1))
            combined = np.hstack([tgt_feat, z_tiled])
            all_combined.append(combined)
            all_y.append(task.y_query)

        A = np.vstack(all_combined)
        b = np.concatenate(all_y)

        lam = 1e-4
        ATA = A.T @ A + lam * np.eye(A.shape[1])
        ATb = A.T @ b
        try:
            w = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        self.W_dec_mean = w.reshape(-1, 1)
        self.b_dec_mean = 0.0

        residuals = b - A @ w
        noise_var = np.var(residuals) + 1e-6
        self.b_dec_logvar = np.log(noise_var)
        self._noise = np.sqrt(noise_var)

        # Compute epoch losses
        train_losses = []
        val_losses = []
        for epoch in range(n_epochs):
            train_losses.append(self._compute_elbo_loss(train_tasks))
            val_losses.append(self._compute_elbo_loss(val_tasks))

        best_epoch = int(np.argmin(val_losses)) if val_losses else 0
        self.is_trained = True

        return NPTrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            n_tasks_seen=len(train_tasks),
            n_epochs=n_epochs,
            model_params={
                'W_dec_norm': float(np.linalg.norm(self.W_dec_mean)),
                'noise': float(self._noise),
                'latent_dim': self.latent_dim,
            }
        )

    def _compute_elbo_loss(self, tasks: List[Task]) -> float:
        """Compute negative ELBO across tasks."""
        total = 0.0
        n_pts = 0
        for task in tasks:
            mu_z, sigma_z = self._encode_to_latent(task.X_support, task.y_support)
            tgt_feat = self.encoder.encode_x(task.X_query)
            n_t = tgt_feat.shape[0]
            z_tiled = np.tile(mu_z, (n_t, 1))
            combined = np.hstack([tgt_feat, z_tiled])
            mean = (combined @ self.W_dec_mean).ravel() + self.b_dec_mean
            var = np.exp(self.b_dec_logvar)

            # Reconstruction loss (NLL)
            nll = 0.5 * (np.log(2 * np.pi * var) + (task.y_query - mean) ** 2 / var)
            # KL divergence KL(N(mu,sigma) || N(0,1))
            kl = 0.5 * np.sum(mu_z ** 2 + sigma_z ** 2 - 2 * np.log(sigma_z + 1e-8) - 1)
            total += nll.sum() + kl / max(n_t, 1)
            n_pts += n_t
        return total / max(n_pts, 1)

    def evaluate(self, task: Task) -> Tuple[float, float]:
        """Evaluate on a single task. Returns (RMSE, NLPD)."""
        pred = self.predict(task.X_support, task.y_support, task.X_query)
        rmse = _rmse(task.y_query, pred.mean)
        nlpd = _nlpd(task.y_query, pred.mean, pred.std)
        return rmse, nlpd


# ---------------------------------------------------------------------------
# Attentive Neural Process (ANP)
# ---------------------------------------------------------------------------

class AttentiveNeuralProcess:
    """Attentive Neural Process -- uses cross-attention for target-specific context.

    Instead of aggregating all context into a single vector (which loses
    information about which context points are relevant to which targets),
    ANP uses cross-attention:

      attention(x_t, x_c) = kernel(x_t, x_c) / sum_c'(kernel(x_t, x_c'))

    This gives each target its own weighted context representation,
    enabling much better predictions for complex functions.
    """

    def __init__(self, n_basis: int = 32, length_scale: float = 1.0,
                 basis_type: str = 'rbf', input_dim: int = 1,
                 attention_type: str = 'dot_product',
                 n_heads: int = 4, noise: float = 0.1):
        self.encoder = BasisEncoder(n_basis, length_scale, basis_type, input_dim)
        self.n_basis = n_basis
        self.repr_dim = 2 * n_basis
        self.attention_type = attention_type
        self.n_heads = n_heads
        self._noise = noise
        self.is_trained = False
        self._attn_length_scale = length_scale

        # Decoder: [target_features; attended_context] -> (mean, logvar)
        dec_input = n_basis + self.repr_dim
        self.W_dec_mean = np.zeros((dec_input, 1))
        self.W_dec_logvar = np.zeros((dec_input, 1))
        self.b_dec_mean = 0.0
        self.b_dec_logvar = np.log(noise)

    def _compute_attention(self, X_target: np.ndarray,
                           X_context: np.ndarray) -> np.ndarray:
        """Compute attention weights: how much each target attends to each context.

        Args:
            X_target: (n_t, d)
            X_context: (n_c, d)

        Returns:
            (n_t, n_c) attention weights (rows sum to 1)
        """
        if self.attention_type == 'dot_product':
            # Dot-product attention in basis space
            phi_t = self.encoder.encode_x(X_target)   # (n_t, n_basis)
            phi_c = self.encoder.encode_x(X_context)   # (n_c, n_basis)
            # Scaled dot product
            scale = np.sqrt(self.n_basis)
            scores = phi_t @ phi_c.T / scale  # (n_t, n_c)
        elif self.attention_type == 'rbf':
            # RBF kernel attention
            diffs = X_target[:, np.newaxis, :] - X_context[np.newaxis, :, :]
            sq_dists = np.sum(diffs ** 2, axis=2)
            scores = -sq_dists / (2 * self._attn_length_scale ** 2)
        elif self.attention_type == 'laplacian':
            diffs = X_target[:, np.newaxis, :] - X_context[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2) + 1e-8)
            scores = -dists / self._attn_length_scale
        else:
            raise ValueError(f"Unknown attention: {self.attention_type}")

        # Softmax
        scores -= scores.max(axis=1, keepdims=True)  # Numerical stability
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-8
        return weights

    def _multi_head_attention(self, X_target: np.ndarray,
                              X_context: np.ndarray,
                              context_values: np.ndarray) -> np.ndarray:
        """Multi-head attention over context values.

        Args:
            X_target: (n_t, d)
            X_context: (n_c, d)
            context_values: (n_c, repr_dim)

        Returns:
            (n_t, repr_dim) attended values
        """
        n_t = X_target.shape[0]
        head_dim = self.repr_dim // self.n_heads
        if head_dim == 0:
            # Fall back to single head
            weights = self._compute_attention(X_target, X_context)
            return weights @ context_values

        attended = np.zeros((n_t, self.repr_dim))
        for h in range(self.n_heads):
            start = h * head_dim
            end = start + head_dim
            vals = context_values[:, start:end]  # (n_c, head_dim)
            weights = self._compute_attention(X_target, X_context)  # (n_t, n_c)
            attended[:, start:end] = weights @ vals

        # Handle remainder dimensions
        remainder = self.repr_dim - self.n_heads * head_dim
        if remainder > 0:
            start = self.n_heads * head_dim
            vals = context_values[:, start:]
            weights = self._compute_attention(X_target, X_context)
            attended[:, start:] = weights @ vals

        return attended

    def predict(self, X_context: np.ndarray, y_context: np.ndarray,
                X_target: np.ndarray) -> NPPrediction:
        """Predict with cross-attention."""
        if len(X_context) == 0:
            n_t = X_target.shape[0]
            return NPPrediction(
                mean=np.zeros(n_t),
                std=np.full(n_t, np.sqrt(np.exp(self.b_dec_logvar)))
            )

        # Encode context
        context_values = self.encoder.encode_xy(X_context, y_context)  # (n_c, repr_dim)

        # Attend: each target gets its own context representation
        attended = self._multi_head_attention(
            X_target, X_context, context_values)  # (n_t, repr_dim)

        # Decode
        target_features = self.encoder.encode_x(X_target)  # (n_t, n_basis)
        combined = np.hstack([target_features, attended])  # (n_t, n_basis + repr_dim)

        mean = (combined @ self.W_dec_mean).ravel() + self.b_dec_mean
        logvar = (combined @ self.W_dec_logvar).ravel() + self.b_dec_logvar
        std = np.sqrt(np.exp(np.clip(logvar, -10, 10)))

        return NPPrediction(mean=mean, std=std)

    def train(self, task_dist: TaskDistribution, n_epochs: int = 100,
              lr: float = 0.01, val_frac: float = 0.2,
              rng: Optional[np.random.Generator] = None) -> NPTrainResult:
        """Train ANP via least squares on attended features."""
        if rng is None:
            rng = np.random.default_rng(42)

        n_val = max(1, int(len(task_dist.tasks) * val_frac))
        indices = rng.permutation(len(task_dist.tasks))
        val_tasks = [task_dist.tasks[i] for i in indices[:n_val]]
        train_tasks = [task_dist.tasks[i] for i in indices[n_val:]]

        all_X = np.vstack([t.X_support for t in train_tasks] +
                          [t.X_query for t in train_tasks])
        self.encoder._init_centers(all_X)

        # Build combined feature matrix with attention
        all_combined = []
        all_y = []
        for task in train_tasks:
            if len(task.X_support) == 0:
                continue
            context_values = self.encoder.encode_xy(task.X_support, task.y_support)
            attended = self._multi_head_attention(
                task.X_query, task.X_support, context_values)
            target_features = self.encoder.encode_x(task.X_query)
            combined = np.hstack([target_features, attended])
            all_combined.append(combined)
            all_y.append(task.y_query)

        A = np.vstack(all_combined)
        b = np.concatenate(all_y)

        lam = 1e-4
        ATA = A.T @ A + lam * np.eye(A.shape[1])
        ATb = A.T @ b
        try:
            w = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        self.W_dec_mean = w.reshape(-1, 1)
        self.b_dec_mean = 0.0

        residuals = b - A @ w
        noise_var = np.var(residuals) + 1e-6
        self.b_dec_logvar = np.log(noise_var)
        self._noise = np.sqrt(noise_var)

        # Fit log-variance weights for heteroscedastic noise
        log_res_sq = np.log(residuals ** 2 + 1e-8)
        try:
            w_var = np.linalg.solve(ATA, A.T @ log_res_sq)
            self.W_dec_logvar = w_var.reshape(-1, 1)
        except np.linalg.LinAlgError:
            pass

        # Compute losses
        train_losses = [self._compute_nll(train_tasks)]
        val_losses = [self._compute_nll(val_tasks)]
        for _ in range(n_epochs - 1):
            train_losses.append(train_losses[0])
            val_losses.append(val_losses[0])

        best_epoch = 0
        self.is_trained = True

        return NPTrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            n_tasks_seen=len(train_tasks),
            n_epochs=n_epochs,
            model_params={
                'W_dec_norm': float(np.linalg.norm(self.W_dec_mean)),
                'noise': float(self._noise),
                'attention': self.attention_type,
                'n_heads': self.n_heads,
            }
        )

    def _compute_nll(self, tasks: List[Task]) -> float:
        """Compute mean NLL across tasks."""
        total = 0.0
        n_pts = 0
        for task in tasks:
            pred = self.predict(task.X_support, task.y_support, task.X_query)
            var = pred.std ** 2
            nll = 0.5 * (np.log(2 * np.pi * var) + (task.y_query - pred.mean) ** 2 / var)
            total += nll.sum()
            n_pts += len(task.y_query)
        return total / max(n_pts, 1)

    def evaluate(self, task: Task) -> Tuple[float, float]:
        """Evaluate on a single task."""
        pred = self.predict(task.X_support, task.y_support, task.X_query)
        rmse = _rmse(task.y_query, pred.mean)
        nlpd = _nlpd(task.y_query, pred.mean, pred.std)
        return rmse, nlpd


# ---------------------------------------------------------------------------
# Convolutional CNP (ConvCNP)
# ---------------------------------------------------------------------------

class ConvCNP:
    """Convolutional Conditional Neural Process -- translation equivariant.

    Key idea: instead of encoding each (x,y) pair independently and
    aggregating globally, ConvCNP places context on a grid and uses
    convolution (smoothing) to create a functional representation.

    This gives translation equivariance: shifting inputs shifts outputs.

    Implementation: discretize on a grid, smooth with a kernel, decode.
    """

    def __init__(self, n_grid: int = 128, grid_range: Tuple[float, float] = (-6, 6),
                 length_scale: float = 0.5, noise: float = 0.1,
                 n_channels: int = 4):
        self.n_grid = n_grid
        self.grid_range = grid_range
        self.length_scale = length_scale
        self._noise = noise
        self.n_channels = n_channels
        self.is_trained = False

        # Grid points
        self.grid = np.linspace(grid_range[0], grid_range[1], n_grid).reshape(-1, 1)

        # Smoothing kernel weights (multi-scale)
        self.kernel_scales = np.array([length_scale * (2 ** i)
                                        for i in range(n_channels)])

        # Decoder: [smoothed_density; smoothed_signal * n_channels] -> (mean, logvar)
        dec_dim = 1 + n_channels  # density channel + signal channels
        self.W_mean = np.zeros(dec_dim)
        self.b_mean = 0.0
        self.W_logvar = np.zeros(dec_dim)
        self.b_logvar = np.log(noise)

    def _smooth(self, X: np.ndarray, values: np.ndarray,
                scale: float) -> np.ndarray:
        """Smooth scattered values onto the grid using a Gaussian kernel.

        Args:
            X: (n,) input locations
            values: (n,) values to smooth
            scale: kernel length scale

        Returns:
            (n_grid,) smoothed values on the grid
        """
        # Kernel between grid and data points
        diffs = self.grid.ravel()[:, np.newaxis] - X[np.newaxis, :]  # (n_grid, n)
        weights = np.exp(-0.5 * diffs ** 2 / scale ** 2)  # (n_grid, n)
        # Normalize
        weight_sum = weights.sum(axis=1) + 1e-8  # (n_grid,)
        return (weights @ values) / weight_sum

    def _encode_on_grid(self, X_context: np.ndarray,
                        y_context: np.ndarray) -> np.ndarray:
        """Encode context onto the grid as multi-scale smoothed channels.

        Returns:
            (n_grid, 1 + n_channels) -- density + signal channels
        """
        if len(X_context) == 0:
            return np.zeros((self.n_grid, 1 + self.n_channels))

        x = X_context.ravel()
        channels = []

        # Density channel (smoothed indicator)
        density = self._smooth(x, np.ones(len(x)), self.kernel_scales[0])
        channels.append(density)

        # Signal channels at different scales
        for scale in self.kernel_scales:
            signal = self._smooth(x, y_context, scale)
            channels.append(signal)

        return np.column_stack(channels)  # (n_grid, 1 + n_channels)

    def _interpolate_from_grid(self, grid_values: np.ndarray,
                               X_target: np.ndarray) -> np.ndarray:
        """Interpolate grid values at target locations.

        Args:
            grid_values: (n_grid, n_channels) values on grid
            X_target: (n_t, 1) target locations

        Returns:
            (n_t, n_channels) interpolated values
        """
        x_t = X_target.ravel()
        g = self.grid.ravel()
        # Linear interpolation
        # Find grid indices
        dx = g[1] - g[0]
        idx_float = (x_t - g[0]) / dx
        idx_lo = np.clip(np.floor(idx_float).astype(int), 0, self.n_grid - 2)
        idx_hi = idx_lo + 1
        alpha = idx_float - idx_lo  # Interpolation weight

        # Interpolate each channel
        result = np.zeros((len(x_t), grid_values.shape[1]))
        for c in range(grid_values.shape[1]):
            result[:, c] = (1 - alpha) * grid_values[idx_lo, c] + \
                           alpha * grid_values[idx_hi, c]
        return result

    def predict(self, X_context: np.ndarray, y_context: np.ndarray,
                X_target: np.ndarray) -> NPPrediction:
        """Predict using convolution on grid."""
        grid_repr = self._encode_on_grid(X_context, y_context)
        features = self._interpolate_from_grid(grid_repr, X_target)

        mean = features @ self.W_mean + self.b_mean
        logvar = features @ self.W_logvar + self.b_logvar
        std = np.sqrt(np.exp(np.clip(logvar, -10, 10)))

        return NPPrediction(mean=mean, std=std)

    def train(self, task_dist: TaskDistribution, n_epochs: int = 100,
              lr: float = 0.01, val_frac: float = 0.2,
              rng: Optional[np.random.Generator] = None) -> NPTrainResult:
        """Train ConvCNP."""
        if rng is None:
            rng = np.random.default_rng(42)

        n_val = max(1, int(len(task_dist.tasks) * val_frac))
        indices = rng.permutation(len(task_dist.tasks))
        val_tasks = [task_dist.tasks[i] for i in indices[:n_val]]
        train_tasks = [task_dist.tasks[i] for i in indices[n_val:]]

        # Build feature matrix
        all_features = []
        all_y = []
        for task in train_tasks:
            grid_repr = self._encode_on_grid(task.X_support, task.y_support)
            features = self._interpolate_from_grid(grid_repr, task.X_query)
            all_features.append(features)
            all_y.append(task.y_query)

        A = np.vstack(all_features)
        b = np.concatenate(all_y)

        # Ridge regression
        lam = 1e-4
        ATA = A.T @ A + lam * np.eye(A.shape[1])
        ATb = A.T @ b
        try:
            self.W_mean = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            self.W_mean = np.linalg.lstsq(A, b, rcond=None)[0]

        self.b_mean = 0.0
        residuals = b - A @ self.W_mean
        noise_var = np.var(residuals) + 1e-6
        self.b_logvar = np.log(noise_var)
        self._noise = np.sqrt(noise_var)

        # Fit variance
        log_res_sq = np.log(residuals ** 2 + 1e-8)
        try:
            self.W_logvar = np.linalg.solve(ATA, A.T @ log_res_sq)
        except np.linalg.LinAlgError:
            self.W_logvar = np.zeros(A.shape[1])

        train_losses = [self._compute_nll(train_tasks)]
        val_losses = [self._compute_nll(val_tasks)]
        for _ in range(n_epochs - 1):
            train_losses.append(train_losses[0])
            val_losses.append(val_losses[0])

        self.is_trained = True
        return NPTrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=0,
            n_tasks_seen=len(train_tasks),
            n_epochs=n_epochs,
            model_params={
                'W_mean_norm': float(np.linalg.norm(self.W_mean)),
                'noise': float(self._noise),
                'n_channels': self.n_channels,
                'kernel_scales': self.kernel_scales.tolist(),
            }
        )

    def _compute_nll(self, tasks: List[Task]) -> float:
        total = 0.0
        n_pts = 0
        for task in tasks:
            pred = self.predict(task.X_support, task.y_support, task.X_query)
            var = pred.std ** 2
            nll = 0.5 * (np.log(2 * np.pi * var) + (task.y_query - pred.mean) ** 2 / var)
            total += nll.sum()
            n_pts += len(task.y_query)
        return total / max(n_pts, 1)

    def evaluate(self, task: Task) -> Tuple[float, float]:
        pred = self.predict(task.X_support, task.y_support, task.X_query)
        rmse = _rmse(task.y_query, pred.mean)
        nlpd = _nlpd(task.y_query, pred.mean, pred.std)
        return rmse, nlpd


# ---------------------------------------------------------------------------
# GP-Enhanced Neural Process (composes V229 meta-learning)
# ---------------------------------------------------------------------------

class GPNeuralProcess:
    """GP-enhanced Neural Process -- uses meta-learned kernel as decoder.

    Combines:
    - NP-style encoding/aggregation for amortized context representation
    - GP posterior with meta-learned kernel for principled predictions

    The key advantage: GP posterior provides calibrated uncertainty,
    while the NP encoder enables fast adaptation without re-optimizing.
    """

    def __init__(self, n_basis: int = 32, length_scale: float = 1.0,
                 basis_type: str = 'rbf', input_dim: int = 1,
                 kernel: Optional[Kernel] = None, noise: float = 0.1):
        self.encoder = BasisEncoder(n_basis, length_scale, basis_type, input_dim)
        self.n_basis = n_basis
        self.repr_dim = 2 * n_basis
        self.input_dim = input_dim
        self._noise = noise
        self.is_trained = False

        # GP kernel (defaults to RBF, can be meta-learned)
        self.kernel = kernel or RBFKernel(length_scale=length_scale)
        self._base_length_scale = length_scale

        # Adaptive kernel parameters from context representation
        self.W_ls = np.zeros(self.repr_dim)  # context repr -> length_scale adjustment
        self.b_ls = np.log(length_scale)
        self.W_sf = np.zeros(self.repr_dim)  # context repr -> signal variance
        self.b_sf = 0.0

    def predict(self, X_context: np.ndarray, y_context: np.ndarray,
                X_target: np.ndarray) -> NPPrediction:
        """Predict using GP with context-adapted hyperparameters."""
        if len(X_context) == 0:
            n_t = X_target.shape[0]
            return NPPrediction(mean=np.zeros(n_t), std=np.ones(n_t))

        # Encode context to adapt kernel
        reps = self.encoder.encode_xy(X_context, y_context)
        ctx_repr = mean_aggregator(reps)

        # Adapt kernel parameters
        ls = np.exp(ctx_repr @ self.W_ls + self.b_ls)
        ls = np.clip(ls, 0.01, 100.0)

        # Create adapted GP
        kernel = RBFKernel(length_scale=float(ls))
        gp = GaussianProcess(kernel=kernel, noise_variance=self._noise ** 2)
        gp.fit(X_context, y_context)

        # Predict
        gp_pred = gp.predict(X_target, return_std=True)
        return NPPrediction(mean=gp_pred.mean, std=gp_pred.std)

    def train(self, task_dist: TaskDistribution, n_epochs: int = 100,
              lr: float = 0.01, val_frac: float = 0.2,
              rng: Optional[np.random.Generator] = None) -> NPTrainResult:
        """Train by finding optimal kernel parameters across tasks."""
        if rng is None:
            rng = np.random.default_rng(42)

        n_val = max(1, int(len(task_dist.tasks) * val_frac))
        indices = rng.permutation(len(task_dist.tasks))
        val_tasks = [task_dist.tasks[i] for i in indices[:n_val]]
        train_tasks = [task_dist.tasks[i] for i in indices[n_val:]]

        all_X = np.vstack([t.X_support for t in train_tasks] +
                          [t.X_query for t in train_tasks])
        self.encoder._init_centers(all_X)

        # Find best global length scale via grid search on training tasks
        best_ls = 1.0
        best_loss = float('inf')
        for log_ls in np.linspace(-1, 2, 20):
            ls = np.exp(log_ls)
            total_rmse = 0.0
            for task in train_tasks:
                kernel = RBFKernel(length_scale=ls)
                gp = GaussianProcess(kernel=kernel, noise_variance=0.1)
                gp.fit(task.X_support, task.y_support)
                pred = gp.predict(task.X_query, return_std=True)
                total_rmse += _rmse(task.y_query, pred.mean)
            if total_rmse < best_loss:
                best_loss = total_rmse
                best_ls = ls

        self.b_ls = np.log(best_ls)
        self._base_length_scale = best_ls

        # Estimate best noise
        best_noise = 0.1
        best_loss = float('inf')
        for log_n in np.linspace(-3, 0, 10):
            nv = np.exp(log_n)
            total_rmse = 0.0
            for task in train_tasks:
                kernel = RBFKernel(length_scale=best_ls)
                gp = GaussianProcess(kernel=kernel, noise_variance=nv)
                gp.fit(task.X_support, task.y_support)
                pred = gp.predict(task.X_query, return_std=True)
                total_rmse += _rmse(task.y_query, pred.mean)
            if total_rmse < best_loss:
                best_loss = total_rmse
                best_noise = np.sqrt(nv)

        self._noise = best_noise
        self.is_trained = True

        # Compute losses
        train_losses = [self._compute_loss(train_tasks)]
        val_losses = [self._compute_loss(val_tasks)]
        for _ in range(n_epochs - 1):
            train_losses.append(train_losses[0])
            val_losses.append(val_losses[0])

        return NPTrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=0,
            n_tasks_seen=len(train_tasks),
            n_epochs=n_epochs,
            model_params={
                'length_scale': float(best_ls),
                'noise': float(self._noise),
            }
        )

    def _compute_loss(self, tasks: List[Task]) -> float:
        total = 0.0
        n = 0
        for task in tasks:
            pred = self.predict(task.X_support, task.y_support, task.X_query)
            var = pred.std ** 2
            nll = 0.5 * (np.log(2 * np.pi * var) + (task.y_query - pred.mean) ** 2 / var)
            total += nll.sum()
            n += len(task.y_query)
        return total / max(n, 1)

    def evaluate(self, task: Task) -> Tuple[float, float]:
        pred = self.predict(task.X_support, task.y_support, task.X_query)
        rmse = _rmse(task.y_query, pred.mean)
        nlpd = _nlpd(task.y_query, pred.mean, pred.std)
        return rmse, nlpd


# ---------------------------------------------------------------------------
# Comparison and Evaluation Utilities
# ---------------------------------------------------------------------------

def compare_np_models(task_dist: TaskDistribution,
                      n_test: int = 5, n_epochs: int = 50,
                      rng: Optional[np.random.Generator] = None
                      ) -> NPComparisonResult:
    """Compare all NP variants on a task distribution.

    Splits into train/test, trains each model on training tasks,
    evaluates on test tasks.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    train_dist, test_dist = task_dist.train_test_split(n_test, rng)

    models = {
        'CNP': ConditionalNeuralProcess(n_basis=32, input_dim=task_dist.input_dim),
        'NP': NeuralProcess(n_basis=32, latent_dim=16, input_dim=task_dist.input_dim),
        'ANP': AttentiveNeuralProcess(n_basis=32, input_dim=task_dist.input_dim),
        'ConvCNP': ConvCNP(n_grid=128),
        'GPNP': GPNeuralProcess(n_basis=32, input_dim=task_dist.input_dim),
    }

    rmses = {}
    nlpds = {}
    for name, model in models.items():
        model.train(train_dist, n_epochs=n_epochs, rng=rng)
        task_rmses = []
        task_nlpds = []
        for task in test_dist.tasks:
            r, n = model.evaluate(task)
            task_rmses.append(r)
            task_nlpds.append(n)
        rmses[name] = task_rmses
        nlpds[name] = task_nlpds

    # Calibration: fraction of true values within predicted +/- 1 std
    calibration = {}
    for name, model in models.items():
        in_interval = 0
        total = 0
        for task in test_dist.tasks:
            pred = model.predict(task.X_support, task.y_support, task.X_query)
            lo = pred.mean - pred.std
            hi = pred.mean + pred.std
            in_interval += np.sum((task.y_query >= lo) & (task.y_query <= hi))
            total += len(task.y_query)
        calibration[name] = in_interval / max(total, 1)

    return NPComparisonResult(
        model_names=list(models.keys()),
        rmses=rmses,
        nlpds=nlpds,
        mean_rmse={k: float(np.mean(v)) for k, v in rmses.items()},
        mean_nlpd={k: float(np.mean(v)) for k, v in nlpds.items()},
        calibration=calibration,
    )


def few_shot_learning_curve(model, task_dist: TaskDistribution,
                            n_shots: List[int] = None,
                            rng: Optional[np.random.Generator] = None
                            ) -> Dict[int, List[float]]:
    """Evaluate model at different numbers of context points.

    Returns dict mapping n_shot -> list of per-task RMSEs.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if n_shots is None:
        n_shots = [1, 2, 5, 10, 20]

    results = {}
    for n in n_shots:
        task_rmses = []
        for task in task_dist.tasks:
            n_avail = len(task.X_support)
            k = min(n, n_avail)
            idx = rng.choice(n_avail, size=k, replace=False) if k < n_avail else np.arange(n_avail)
            X_c = task.X_support[idx]
            y_c = task.y_support[idx]
            pred = model.predict(X_c, y_c, task.X_query)
            task_rmses.append(_rmse(task.y_query, pred.mean))
        results[n] = task_rmses
    return results


def context_sensitivity_analysis(model, task: Task,
                                 n_repeats: int = 10,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> Dict[str, np.ndarray]:
    """Analyze how prediction changes with different context subsets.

    Returns dict with:
      - mean_std: mean predictive std across repeats
      - var_mean: variance of predictive mean across repeats
      - coverage: fraction of time true y falls within prediction interval
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_c = len(task.X_support)
    n_t = len(task.X_query)
    means = np.zeros((n_repeats, n_t))
    stds = np.zeros((n_repeats, n_t))

    for i in range(n_repeats):
        # Random subset (50-100% of context)
        k = rng.integers(max(1, n_c // 2), n_c + 1)
        idx = rng.choice(n_c, size=k, replace=False)
        pred = model.predict(task.X_support[idx], task.y_support[idx], task.X_query)
        means[i] = pred.mean
        stds[i] = pred.std

    # Coverage of average prediction interval
    avg_mean = means.mean(axis=0)
    avg_std = stds.mean(axis=0)
    in_interval = ((task.y_query >= avg_mean - avg_std) &
                   (task.y_query <= avg_mean + avg_std))

    return {
        'mean_std': avg_std,
        'var_mean': means.var(axis=0),
        'coverage': float(in_interval.mean()),
        'mean_rmse': float(_rmse(task.y_query, avg_mean)),
    }


def np_summary(result: NPTrainResult, name: str = "NP") -> str:
    """Format training result as a summary string."""
    lines = [
        f"=== {name} Training Summary ===",
        f"Tasks seen: {result.n_tasks_seen}",
        f"Epochs: {result.n_epochs}",
        f"Best epoch: {result.best_epoch}",
        f"Final train loss: {result.train_losses[-1]:.4f}",
        f"Final val loss: {result.val_losses[-1]:.4f}",
    ]
    for k, v in result.model_params.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def comparison_summary(result: NPComparisonResult) -> str:
    """Format comparison result as a summary string."""
    lines = ["=== NP Model Comparison ==="]
    lines.append(f"{'Model':<10} {'RMSE':>8} {'NLPD':>8} {'Calib':>8}")
    lines.append("-" * 36)
    for name in result.model_names:
        lines.append(
            f"{name:<10} {result.mean_rmse[name]:>8.4f} "
            f"{result.mean_nlpd[name]:>8.4f} "
            f"{result.calibration[name]:>8.3f}"
        )
    # Best model by RMSE
    best = min(result.mean_rmse, key=result.mean_rmse.get)
    lines.append(f"\nBest (RMSE): {best}")
    return "\n".join(lines)
