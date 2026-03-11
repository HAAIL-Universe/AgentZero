"""
C183: Normalizing Flows
=======================
Invertible generative models for density estimation and sampling.

Core idea: transform a simple base distribution (e.g., Gaussian) through
a chain of invertible, differentiable transformations. The change-of-variables
formula gives exact log-likelihoods.

Components:
- Flow layers: Affine, Planar, Radial, RealNVP coupling, ActNorm
- Composition into flow chains
- Forward (sampling) and inverse (density estimation) passes
- Log-determinant of Jacobian tracking
- Training via maximum likelihood

Built with NumPy only. No external ML libraries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


# ============================================================
# Base Distribution
# ============================================================

class BaseDistribution:
    """Base distribution for normalizing flows (standard Gaussian)."""

    def __init__(self, dim: int):
        self.dim = dim

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample n points from base distribution."""
        if rng is None:
            rng = np.random.RandomState()
        return rng.randn(n, self.dim)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """Log probability of z under standard Gaussian."""
        # z: (batch, dim)
        return -0.5 * (self.dim * np.log(2 * np.pi) + np.sum(z ** 2, axis=-1))


# ============================================================
# Flow Layers (Abstract)
# ============================================================

class FlowLayer:
    """Abstract base for an invertible transformation."""

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """z -> x, returns (x, log_det_jacobian) where log_det is per-sample."""
        raise NotImplementedError

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x -> z, returns (z, log_det_jacobian) of the inverse."""
        raise NotImplementedError

    def parameters(self) -> List[np.ndarray]:
        """Return list of parameter arrays for optimization."""
        return []

    def set_parameters(self, params: List[np.ndarray]):
        """Set parameters from list."""
        pass


# ============================================================
# Affine Flow
# ============================================================

class AffineFlow(FlowLayer):
    """Element-wise affine transformation: x = z * exp(log_scale) + shift."""

    def __init__(self, dim: int):
        self.dim = dim
        self.log_scale = np.zeros(dim)
        self.shift = np.zeros(dim)

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = z * np.exp(self.log_scale) + self.shift
        log_det = np.sum(self.log_scale) * np.ones(z.shape[0])
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = (x - self.shift) * np.exp(-self.log_scale)
        log_det = -np.sum(self.log_scale) * np.ones(x.shape[0])
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.log_scale, self.shift]

    def set_parameters(self, params: List[np.ndarray]):
        self.log_scale = params[0].copy()
        self.shift = params[1].copy()


# ============================================================
# Planar Flow
# ============================================================

class PlanarFlow(FlowLayer):
    """Planar flow: x = z + u * tanh(w^T z + b).

    Invertibility constraint: w^T u >= -1 (enforced via reparameterization).
    """

    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        self.dim = dim
        self.w = rng.randn(dim) * 0.1
        self.u = rng.randn(dim) * 0.1
        self.b = np.array(0.0)

    def _get_u_hat(self) -> np.ndarray:
        """Enforce invertibility: w^T u_hat >= -1."""
        wtu = np.dot(self.w, self.u)
        if wtu >= -1:
            return self.u.copy()
        # Project u onto the constraint surface
        m = -1 + np.log(1 + np.exp(wtu))  # softplus(wtu) - 1
        w_norm_sq = np.dot(self.w, self.w) + 1e-15
        return self.u + (m - wtu) * self.w / w_norm_sq

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_hat = self._get_u_hat()
        # z: (batch, dim)
        linear = z @ self.w + self.b  # (batch,)
        tanh_val = np.tanh(linear)  # (batch,)
        x = z + np.outer(tanh_val, u_hat)  # (batch, dim)

        # log |det J| = log |1 + u_hat^T * dtanh/dz * w|
        dtanh = 1 - tanh_val ** 2  # (batch,)
        psi = np.outer(dtanh, self.w)  # (batch, dim)
        det_term = 1 + psi @ u_hat  # (batch,)
        log_det = np.log(np.abs(det_term) + 1e-15)
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate inverse using fixed-point iteration."""
        u_hat = self._get_u_hat()
        z = x.copy()
        for _ in range(50):
            linear = z @ self.w + self.b
            tanh_val = np.tanh(linear)
            z_new = x - np.outer(tanh_val, u_hat)
            if np.max(np.abs(z_new - z)) < 1e-10:
                break
            z = z_new

        # Compute log_det of inverse (negative of forward)
        linear = z @ self.w + self.b
        tanh_val = np.tanh(linear)
        dtanh = 1 - tanh_val ** 2
        psi = np.outer(dtanh, self.w)
        det_term = 1 + psi @ u_hat
        log_det = -np.log(np.abs(det_term) + 1e-15)
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.w, self.u, np.array([self.b])]

    def set_parameters(self, params: List[np.ndarray]):
        self.w = params[0].copy()
        self.u = params[1].copy()
        self.b = params[2][0]


# ============================================================
# Radial Flow
# ============================================================

class RadialFlow(FlowLayer):
    """Radial flow: x = z + beta * h(alpha, r) * (z - z0).

    Where r = |z - z0|, h(alpha, r) = 1 / (alpha + r).
    """

    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        self.dim = dim
        self.z0 = rng.randn(dim) * 0.1
        self.log_alpha = np.array(0.0)  # alpha = exp(log_alpha) > 0
        self.beta_raw = np.array(0.0)

    def _get_params(self):
        alpha = np.exp(self.log_alpha)
        # Enforce beta >= -alpha for invertibility
        beta = -alpha + np.log(1 + np.exp(self.beta_raw))  # softplus - alpha
        return alpha, beta

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha, beta = self._get_params()
        diff = z - self.z0  # (batch, dim)
        r = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True) + 1e-15)  # (batch, 1)
        h = 1.0 / (alpha + r)  # (batch, 1)
        x = z + beta * h * diff  # (batch, dim)

        # log |det J|
        # det = (1 + beta*h)^(d-1) * (1 + beta*h + beta*h'*r)
        # h' = -1/(alpha+r)^2
        h_prime = -1.0 / (alpha + r) ** 2
        term1 = 1 + beta * h  # (batch, 1)
        term2 = 1 + beta * h + beta * h_prime * r  # (batch, 1)
        log_det = (self.dim - 1) * np.log(np.abs(term1.ravel()) + 1e-15) + \
                  np.log(np.abs(term2.ravel()) + 1e-15)
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate inverse via fixed-point iteration."""
        alpha, beta = self._get_params()
        z = x.copy()
        for _ in range(50):
            diff = z - self.z0
            r = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True) + 1e-15)
            h = 1.0 / (alpha + r)
            z_new = x - beta * h * diff
            if np.max(np.abs(z_new - z)) < 1e-10:
                break
            z = z_new

        # Log det of inverse
        diff = z - self.z0
        r = np.sqrt(np.sum(diff ** 2, axis=-1, keepdims=True) + 1e-15)
        h = 1.0 / (alpha + r)
        h_prime = -1.0 / (alpha + r) ** 2
        term1 = 1 + beta * h
        term2 = 1 + beta * h + beta * h_prime * r
        log_det = -((self.dim - 1) * np.log(np.abs(term1.ravel()) + 1e-15) +
                     np.log(np.abs(term2.ravel()) + 1e-15))
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.z0, np.array([self.log_alpha]), np.array([self.beta_raw])]

    def set_parameters(self, params: List[np.ndarray]):
        self.z0 = params[0].copy()
        self.log_alpha = params[1][0]
        self.beta_raw = params[2][0]


# ============================================================
# ActNorm (Activation Normalization)
# ============================================================

class ActNorm(FlowLayer):
    """Activation normalization: data-dependent initialization of affine params.

    On first forward pass, initializes scale and bias so output has zero mean
    and unit variance per dimension.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.log_scale = np.zeros(dim)
        self.bias = np.zeros(dim)
        self.initialized = False

    def _initialize(self, z: np.ndarray):
        """Data-dependent initialization."""
        mean = np.mean(z, axis=0)
        std = np.std(z, axis=0) + 1e-6
        self.bias = -mean / std
        self.log_scale = -np.log(std)
        self.initialized = True

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            self._initialize(z)
        x = z * np.exp(self.log_scale) + self.bias
        log_det = np.sum(self.log_scale) * np.ones(z.shape[0])
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = (x - self.bias) * np.exp(-self.log_scale)
        log_det = -np.sum(self.log_scale) * np.ones(x.shape[0])
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.log_scale, self.bias]

    def set_parameters(self, params: List[np.ndarray]):
        self.log_scale = params[0].copy()
        self.bias = params[1].copy()
        self.initialized = True


# ============================================================
# Permutation Layer
# ============================================================

class Permutation(FlowLayer):
    """Fixed random permutation of dimensions (volume-preserving)."""

    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        self.dim = dim
        self.perm = rng.permutation(dim)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return z[:, self.perm], np.zeros(z.shape[0])

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x[:, self.inv_perm], np.zeros(x.shape[0])


# ============================================================
# RealNVP Coupling Layer
# ============================================================

def _simple_net(input_dim: int, hidden_dim: int, output_dim: int,
                rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
    """Create a simple 2-layer neural network (weights only)."""
    if rng is None:
        rng = np.random.RandomState()
    scale = 0.01
    return {
        'W1': rng.randn(input_dim, hidden_dim) * scale,
        'b1': np.zeros(hidden_dim),
        'W2': rng.randn(hidden_dim, output_dim) * scale,
        'b2': np.zeros(output_dim),
    }


def _forward_net(net: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Forward pass through 2-layer net with tanh activation."""
    h = np.tanh(x @ net['W1'] + net['b1'])
    return h @ net['W2'] + net['b2']


class RealNVPCoupling(FlowLayer):
    """RealNVP affine coupling layer.

    Splits input into two halves. First half passes through unchanged,
    second half is affinely transformed conditioned on the first half.
    """

    def __init__(self, dim: int, hidden_dim: int = 32, mask_even: bool = True,
                 rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        self.dim = dim
        # Create alternating mask
        self.mask = np.zeros(dim)
        if mask_even:
            self.mask[::2] = 1
        else:
            self.mask[1::2] = 1

        input_dim = int(np.sum(self.mask))
        output_dim = dim - input_dim
        self.scale_net = _simple_net(input_dim, hidden_dim, output_dim, rng)
        self.shift_net = _simple_net(input_dim, hidden_dim, output_dim, rng)

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_masked = z[:, self.mask == 1]  # conditioner
        log_s = _forward_net(self.scale_net, z_masked)
        # Clamp log_s for stability
        log_s = np.clip(log_s, -5, 5)
        t = _forward_net(self.shift_net, z_masked)

        x = z.copy()
        x[:, self.mask == 0] = z[:, self.mask == 0] * np.exp(log_s) + t
        log_det = np.sum(log_s, axis=-1)
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_masked = x[:, self.mask == 1]
        log_s = _forward_net(self.scale_net, x_masked)
        log_s = np.clip(log_s, -5, 5)
        t = _forward_net(self.shift_net, x_masked)

        z = x.copy()
        z[:, self.mask == 0] = (x[:, self.mask == 0] - t) * np.exp(-log_s)
        log_det = -np.sum(log_s, axis=-1)
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [
            self.scale_net['W1'], self.scale_net['b1'],
            self.scale_net['W2'], self.scale_net['b2'],
            self.shift_net['W1'], self.shift_net['b1'],
            self.shift_net['W2'], self.shift_net['b2'],
        ]

    def set_parameters(self, params: List[np.ndarray]):
        self.scale_net['W1'] = params[0].copy()
        self.scale_net['b1'] = params[1].copy()
        self.scale_net['W2'] = params[2].copy()
        self.scale_net['b2'] = params[3].copy()
        self.shift_net['W1'] = params[4].copy()
        self.shift_net['b1'] = params[5].copy()
        self.shift_net['W2'] = params[6].copy()
        self.shift_net['b2'] = params[7].copy()


# ============================================================
# Normalizing Flow (Composition)
# ============================================================

class NormalizingFlow:
    """Composes multiple flow layers with a base distribution.

    Forward direction: z ~ base -> x through layers (sampling)
    Inverse direction: x -> z through inverse layers (density estimation)
    """

    def __init__(self, base: BaseDistribution, layers: List[FlowLayer]):
        self.base = base
        self.layers = layers

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform from base space to data space. Returns (x, total_log_det)."""
        total_log_det = np.zeros(z.shape[0])
        x = z
        for layer in self.layers:
            x, log_det = layer.forward(x)
            total_log_det += log_det
        return x, total_log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform from data space to base space. Returns (z, total_log_det)."""
        total_log_det = np.zeros(x.shape[0])
        z = x
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            total_log_det += log_det
        return z, total_log_det

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample n points from the flow model."""
        z = self.base.sample(n, rng)
        x, _ = self.forward(z)
        return x

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability of x under the flow model.

        log p(x) = log p_base(z) + log |det dz/dx|
                  = log p_base(z) + sum of log_det from inverse layers
        """
        z, log_det = self.inverse(x)
        log_pz = self.base.log_prob(z)
        return log_pz + log_det

    def parameters(self) -> List[np.ndarray]:
        """Collect all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def set_parameters(self, params: List[np.ndarray]):
        """Set all parameters across all layers."""
        idx = 0
        for layer in self.layers:
            n = len(layer.parameters())
            layer.set_parameters(params[idx:idx + n])
            idx += n


# ============================================================
# Training
# ============================================================

class FlowTrainer:
    """Train normalizing flows via maximum likelihood with SGD."""

    def __init__(self, flow: NormalizingFlow, lr: float = 0.001):
        self.flow = flow
        self.lr = lr

    def _numerical_grad(self, data: np.ndarray, param: np.ndarray,
                        eps: float = 1e-5) -> np.ndarray:
        """Compute gradient of negative log-likelihood w.r.t. param."""
        grad = np.zeros_like(param)
        original = param.copy()
        flat = param.ravel()
        grad_flat = grad.ravel()

        for i in range(len(flat)):
            old_val = flat[i]

            flat[i] = old_val + eps
            np.copyto(param, flat.reshape(param.shape))
            loss_plus = -np.mean(self.flow.log_prob(data))

            flat[i] = old_val - eps
            np.copyto(param, flat.reshape(param.shape))
            loss_minus = -np.mean(self.flow.log_prob(data))

            grad_flat[i] = (loss_plus - loss_minus) / (2 * eps)
            flat[i] = old_val

        np.copyto(param, original)
        return grad

    def train_step(self, data: np.ndarray) -> float:
        """One step of gradient descent. Returns loss (negative log-likelihood)."""
        params = self.flow.parameters()
        loss = -np.mean(self.flow.log_prob(data))

        # Compute gradients and update
        grads = []
        for param in params:
            g = self._numerical_grad(data, param)
            grads.append(g)

        # SGD update
        new_params = []
        for param, grad in zip(params, grads):
            new_params.append(param - self.lr * grad)

        self.flow.set_parameters(new_params)
        return loss

    def train(self, data: np.ndarray, epochs: int = 10,
              batch_size: Optional[int] = None,
              rng: Optional[np.random.RandomState] = None,
              verbose: bool = False) -> List[float]:
        """Train for multiple epochs. Returns loss history."""
        if rng is None:
            rng = np.random.RandomState()
        if batch_size is None:
            batch_size = min(len(data), 32)

        losses = []
        n = len(data)
        for epoch in range(epochs):
            # Shuffle
            idx = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = data[idx[start:start + batch_size]]
                if len(batch) < 2:
                    continue
                loss = self.train_step(batch)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses


# ============================================================
# Flow Builders (Convenience)
# ============================================================

def build_planar_flow(dim: int, n_layers: int = 4,
                      rng: Optional[np.random.RandomState] = None) -> NormalizingFlow:
    """Build a normalizing flow with planar layers."""
    if rng is None:
        rng = np.random.RandomState()
    base = BaseDistribution(dim)
    layers = [PlanarFlow(dim, rng) for _ in range(n_layers)]
    return NormalizingFlow(base, layers)


def build_realnvp_flow(dim: int, n_coupling: int = 4, hidden_dim: int = 32,
                        rng: Optional[np.random.RandomState] = None) -> NormalizingFlow:
    """Build a RealNVP flow with alternating coupling layers."""
    if rng is None:
        rng = np.random.RandomState()
    base = BaseDistribution(dim)
    layers = []
    for i in range(n_coupling):
        layers.append(ActNorm(dim))
        layers.append(RealNVPCoupling(dim, hidden_dim, mask_even=(i % 2 == 0), rng=rng))
    return NormalizingFlow(base, layers)


def build_mixed_flow(dim: int, rng: Optional[np.random.RandomState] = None) -> NormalizingFlow:
    """Build a flow with mixed layer types."""
    if rng is None:
        rng = np.random.RandomState()
    base = BaseDistribution(dim)
    layers = [
        ActNorm(dim),
        RealNVPCoupling(dim, 16, mask_even=True, rng=rng),
        Permutation(dim, rng),
        RealNVPCoupling(dim, 16, mask_even=True, rng=rng),
        AffineFlow(dim),
    ]
    return NormalizingFlow(base, layers)


# ============================================================
# Density Estimation Utilities
# ============================================================

def estimate_kl_divergence(flow: NormalizingFlow, target_samples: np.ndarray,
                           n_flow_samples: int = 1000,
                           rng: Optional[np.random.RandomState] = None) -> float:
    """Estimate KL divergence from flow to empirical target (approximate).

    Uses the fact that KL(p_flow || p_target) ~ -E_flow[log p_target(x)] + E_flow[log p_flow(x)]
    We approximate with: -mean(log_prob(target_samples)) as a proxy.
    """
    return -np.mean(flow.log_prob(target_samples))


def compute_effective_sample_size(log_weights: np.ndarray) -> float:
    """Compute ESS from log importance weights."""
    log_w = log_weights - np.max(log_weights)  # for numerical stability
    w = np.exp(log_w)
    w = w / np.sum(w)
    return 1.0 / np.sum(w ** 2)


# ============================================================
# 1x1 Invertible Convolution (GLOW-style)
# ============================================================

class InvertibleLinear(FlowLayer):
    """Invertible 1x1 linear transformation (dense, not convolution).

    Uses LU decomposition for efficient log-det computation.
    """

    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        self.dim = dim
        # Initialize with random orthogonal matrix
        q, _ = np.linalg.qr(rng.randn(dim, dim))
        self.W = q

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = z @ self.W.T
        log_det = np.log(np.abs(np.linalg.det(self.W)) + 1e-15) * np.ones(z.shape[0])
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W_inv = np.linalg.inv(self.W)
        z = x @ W_inv.T
        log_det = -np.log(np.abs(np.linalg.det(self.W)) + 1e-15) * np.ones(x.shape[0])
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.W]

    def set_parameters(self, params: List[np.ndarray]):
        self.W = params[0].copy()


# ============================================================
# Batch Normalization Flow
# ============================================================

class BatchNormFlow(FlowLayer):
    """Batch normalization as a flow layer.

    Uses running statistics for inverse pass.
    """

    def __init__(self, dim: int, momentum: float = 0.1):
        self.dim = dim
        self.momentum = momentum
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.log_gamma = np.zeros(dim)
        self.beta = np.zeros(dim)
        self.eps = 1e-5
        self.training = True

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.training and z.shape[0] > 1:
            mean = np.mean(z, axis=0)
            var = np.var(z, axis=0) + self.eps
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var + self.eps

        z_norm = (z - mean) / np.sqrt(var)
        x = z_norm * np.exp(self.log_gamma) + self.beta
        log_det = np.sum(self.log_gamma - 0.5 * np.log(var)) * np.ones(z.shape[0])
        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        var = self.running_var + self.eps
        z_norm = (x - self.beta) * np.exp(-self.log_gamma)
        z = z_norm * np.sqrt(var) + self.running_mean
        log_det = np.sum(-self.log_gamma + 0.5 * np.log(var)) * np.ones(x.shape[0])
        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.log_gamma, self.beta]

    def set_parameters(self, params: List[np.ndarray]):
        self.log_gamma = params[0].copy()
        self.beta = params[1].copy()


# ============================================================
# Spline Flow (Rational Quadratic)
# ============================================================

class RQSplineFlow(FlowLayer):
    """Rational Quadratic Spline Flow (Neural Spline Flow).

    Element-wise transformation using rational quadratic splines.
    More expressive than affine coupling.
    """

    def __init__(self, dim: int, n_bins: int = 8, bound: float = 3.0,
                 rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        self.dim = dim
        self.n_bins = n_bins
        self.bound = bound
        # Parameters: widths, heights, derivatives for each dimension
        # (K widths, K heights, K+1 derivatives) per dimension
        self.raw_widths = rng.randn(dim, n_bins) * 0.01
        self.raw_heights = rng.randn(dim, n_bins) * 0.01
        self.raw_derivatives = rng.randn(dim, n_bins + 1) * 0.01

    def _get_spline_params(self):
        """Convert raw parameters to valid spline parameters."""
        # Widths and heights via softmax (sum to 2*bound)
        widths = np.exp(self.raw_widths)
        widths = widths / np.sum(widths, axis=-1, keepdims=True) * 2 * self.bound

        heights = np.exp(self.raw_heights)
        heights = heights / np.sum(heights, axis=-1, keepdims=True) * 2 * self.bound

        # Derivatives via softplus (positive)
        derivatives = np.log(1 + np.exp(self.raw_derivatives)) + 1e-3

        return widths, heights, derivatives

    def _get_knots(self, w, h):
        """Compute knot positions from widths and heights."""
        # Knot x-positions: [-bound, -bound+w[0], -bound+w[0]+w[1], ..., bound]
        cum_w = np.concatenate([[0], np.cumsum(w)])
        knots_x = -self.bound + cum_w  # length K+1

        cum_h = np.concatenate([[0], np.cumsum(h)])
        knots_y = -self.bound + cum_h  # length K+1

        return knots_x, knots_y

    def _rqs_forward_element(self, x_val: float, w: np.ndarray, h: np.ndarray,
                              d: np.ndarray) -> Tuple[float, float]:
        """Forward pass for single element through RQ spline."""
        if x_val <= -self.bound or x_val >= self.bound:
            return x_val, 0.0

        knots_x, knots_y = self._get_knots(w, h)

        # Find bin: knots_x[k] <= x_val < knots_x[k+1]
        bin_idx = int(np.searchsorted(knots_x[1:], x_val, side='right'))
        bin_idx = max(0, min(bin_idx, self.n_bins - 1))

        x_lo = knots_x[bin_idx]
        x_hi = knots_x[bin_idx + 1]
        y_lo = knots_y[bin_idx]
        y_hi = knots_y[bin_idx + 1]
        w_k = w[bin_idx]
        h_k = h[bin_idx]
        d_lo = d[bin_idx]
        d_hi = d[bin_idx + 1]
        delta = h_k / (w_k + 1e-15)

        theta = (x_val - x_lo) / (w_k + 1e-15)
        theta = np.clip(theta, 0, 1)

        # RQ spline formula
        num = h_k * (delta * theta ** 2 + d_lo * theta * (1 - theta))
        den = delta + (d_lo + d_hi - 2 * delta) * theta * (1 - theta)
        y = y_lo + num / (den + 1e-15)

        # Log derivative
        dnum = delta ** 2 * (d_hi * theta ** 2 + 2 * delta * theta * (1 - theta) + d_lo * (1 - theta) ** 2)
        log_deriv = np.log(dnum + 1e-15) - 2 * np.log(np.abs(den) + 1e-15)

        return y, log_deriv

    def _rqs_inverse_element(self, y_val: float, w: np.ndarray, h: np.ndarray,
                              d: np.ndarray) -> Tuple[float, float]:
        """Inverse pass for single element through RQ spline."""
        if y_val <= -self.bound or y_val >= self.bound:
            return y_val, 0.0

        knots_x, knots_y = self._get_knots(w, h)

        # Find bin in output space: knots_y[k] <= y_val < knots_y[k+1]
        bin_idx = int(np.searchsorted(knots_y[1:], y_val, side='right'))
        bin_idx = max(0, min(bin_idx, self.n_bins - 1))

        x_lo = knots_x[bin_idx]
        y_lo = knots_y[bin_idx]
        w_k = w[bin_idx]
        h_k = h[bin_idx]
        d_lo = d[bin_idx]
        d_hi = d[bin_idx + 1]
        delta = h_k / (w_k + 1e-15)

        # Solve quadratic for theta
        a_coeff = h_k * (delta - d_lo) + (y_val - y_lo) * (d_lo + d_hi - 2 * delta)
        b_coeff = h_k * d_lo - (y_val - y_lo) * (d_lo + d_hi - 2 * delta)
        c_coeff = -delta * (y_val - y_lo)

        discriminant = b_coeff ** 2 - 4 * a_coeff * c_coeff
        if discriminant < 0:
            discriminant = 0
        sqrt_disc = np.sqrt(discriminant)
        theta = (2 * c_coeff) / (-b_coeff - sqrt_disc + 1e-15)
        theta = np.clip(theta, 0, 1)

        x = x_lo + theta * w_k

        # Log derivative (negative of forward's log_det)
        den = delta + (d_lo + d_hi - 2 * delta) * theta * (1 - theta)
        dnum = delta ** 2 * (d_hi * theta ** 2 + 2 * delta * theta * (1 - theta) + d_lo * (1 - theta) ** 2)
        log_deriv = -(np.log(dnum + 1e-15) - 2 * np.log(np.abs(den) + 1e-15))

        return x, log_deriv

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        widths, heights, derivatives = self._get_spline_params()
        batch_size = z.shape[0]
        x = np.zeros_like(z)
        log_det = np.zeros(batch_size)

        for i in range(batch_size):
            for d in range(self.dim):
                val, ld = self._rqs_forward_element(
                    z[i, d], widths[d], heights[d], derivatives[d])
                x[i, d] = val
                log_det[i] += ld

        return x, log_det

    def inverse(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        widths, heights, derivatives = self._get_spline_params()
        batch_size = x.shape[0]
        z = np.zeros_like(x)
        log_det = np.zeros(batch_size)

        for i in range(batch_size):
            for d in range(self.dim):
                val, ld = self._rqs_inverse_element(
                    x[i, d], widths[d], heights[d], derivatives[d])
                z[i, d] = val
                log_det[i] += ld

        return z, log_det

    def parameters(self) -> List[np.ndarray]:
        return [self.raw_widths, self.raw_heights, self.raw_derivatives]

    def set_parameters(self, params: List[np.ndarray]):
        self.raw_widths = params[0].copy()
        self.raw_heights = params[1].copy()
        self.raw_derivatives = params[2].copy()


# ============================================================
# Importance Sampling with Flows
# ============================================================

class FlowImportanceSampler:
    """Use a normalizing flow as a proposal distribution for importance sampling."""

    def __init__(self, flow: NormalizingFlow):
        self.flow = flow

    def estimate_expectation(self, target_log_prob_fn, f_fn, n_samples: int = 1000,
                              rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        """Estimate E_target[f(x)] using importance sampling with flow as proposal.

        Args:
            target_log_prob_fn: log p_target(x), the target distribution
            f_fn: function to compute expectation of
            n_samples: number of samples
            rng: random state

        Returns:
            Dict with 'estimate', 'ess', 'log_normalizer_estimate'
        """
        if rng is None:
            rng = np.random.RandomState()

        # Sample from flow
        z = self.flow.base.sample(n_samples, rng)
        x, log_det_forward = self.flow.forward(z)
        log_q = self.flow.base.log_prob(z) - log_det_forward  # proposal density

        # Compute importance weights
        log_p = target_log_prob_fn(x)  # target density (unnormalized OK)
        log_w = log_p - log_q

        # Normalized weights
        max_log_w = np.max(log_w)
        w = np.exp(log_w - max_log_w)
        w_normalized = w / (np.sum(w) + 1e-15)

        # Estimate
        f_vals = f_fn(x)
        estimate = np.sum(w_normalized * f_vals)
        ess = compute_effective_sample_size(log_w)
        log_z = max_log_w + np.log(np.mean(np.exp(log_w - max_log_w)) + 1e-15)

        return {
            'estimate': estimate,
            'ess': ess,
            'log_normalizer_estimate': log_z,
        }


# ============================================================
# Flow Diagnostics
# ============================================================

def check_invertibility(flow: NormalizingFlow, n_samples: int = 100,
                        rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
    """Check how well the flow's inverse recovers the original input."""
    if rng is None:
        rng = np.random.RandomState()
    z = flow.base.sample(n_samples, rng)
    x, ld_fwd = flow.forward(z)
    z_recovered, ld_inv = flow.inverse(x)

    reconstruction_error = np.mean(np.abs(z - z_recovered))
    log_det_consistency = np.mean(np.abs(ld_fwd + ld_inv))

    return {
        'reconstruction_error': reconstruction_error,
        'log_det_consistency': log_det_consistency,
        'max_error': np.max(np.abs(z - z_recovered)),
    }


def check_log_det_numerical(layer: FlowLayer, z: np.ndarray,
                             eps: float = 1e-5) -> Dict[str, float]:
    """Numerically verify the log determinant of a layer's Jacobian."""
    dim = z.shape[1]
    batch_size = z.shape[0]

    _, analytic_log_det = layer.forward(z)

    # Compute numerical Jacobian for first sample
    z0 = z[0:1]
    J = np.zeros((dim, dim))
    for i in range(dim):
        z_plus = z0.copy()
        z_plus[0, i] += eps
        z_minus = z0.copy()
        z_minus[0, i] -= eps
        x_plus, _ = layer.forward(z_plus)
        x_minus, _ = layer.forward(z_minus)
        J[:, i] = (x_plus[0] - x_minus[0]) / (2 * eps)

    numerical_log_det = np.log(np.abs(np.linalg.det(J)) + 1e-15)
    error = np.abs(analytic_log_det[0] - numerical_log_det)

    return {
        'analytic_log_det': analytic_log_det[0],
        'numerical_log_det': numerical_log_det,
        'error': error,
    }
