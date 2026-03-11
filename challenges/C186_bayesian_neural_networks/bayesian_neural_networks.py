"""
C186: Bayesian Neural Networks
===============================
Neural networks with uncertainty quantification -- instead of point estimates
for weights, maintain probability distributions over weights.

Components:
- BayesianLinear: Layer with weight distributions (mean + log_var), local reparameterization
- BayesianNetwork: Multi-layer BNN with forward sampling
- KLDivergence: KL(q||p) for Gaussian variational posterior vs prior
- ELBO: Evidence Lower Bound = likelihood - KL (variational inference objective)
- MCDropout: Monte Carlo Dropout for approximate Bayesian inference
- BayesianEnsemble: Deep ensemble for uncertainty (epistemic via disagreement)
- VariationalInference: Training loop with KL annealing
- UncertaintyEstimator: Aleatoric + epistemic uncertainty decomposition
- BayesianOptimizer: SGD/Adam for variational parameters
- PredictiveDistribution: Full predictive posterior via MC sampling

Built with NumPy only. No external ML libraries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import copy
import json


# ============================================================
# Activation Functions
# ============================================================

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def softplus(x):
    """Smooth approximation to ReLU: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))


ACTIVATIONS = {
    'relu': (relu, relu_deriv),
    'sigmoid': (sigmoid, sigmoid_deriv),
    'tanh': (tanh_act, tanh_deriv),
    'softmax': (softmax, None),
    'none': (lambda x: x, lambda x: np.ones_like(x)),
}


# ============================================================
# Bayesian Linear Layer
# ============================================================

class BayesianLinear:
    """Linear layer with Gaussian weight distributions.

    Each weight w ~ N(mu_w, sigma_w^2) where sigma_w = softplus(rho_w).
    Uses local reparameterization trick for efficient sampling.
    """

    def __init__(self, in_features: int, out_features: int,
                 prior_mu: float = 0.0, prior_sigma: float = 1.0,
                 init_rho: float = -3.0):
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters for weights
        scale = np.sqrt(2.0 / in_features)
        self.weight_mu = np.random.randn(in_features, out_features) * scale
        self.weight_rho = np.full((in_features, out_features), init_rho)

        # Variational parameters for bias
        self.bias_mu = np.zeros(out_features)
        self.bias_rho = np.full(out_features, init_rho)

        # Prior
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        # Gradients
        self.grad_weight_mu = None
        self.grad_weight_rho = None
        self.grad_bias_mu = None
        self.grad_bias_rho = None

        # Cache for backward
        self._input = None
        self._weight_eps = None
        self._bias_eps = None
        self._weight_sigma = None
        self._bias_sigma = None
        self._sampled_weight = None
        self._sampled_bias = None

    @property
    def weight_sigma(self):
        return softplus(self.weight_rho)

    @property
    def bias_sigma(self):
        return softplus(self.bias_rho)

    def sample_weights(self):
        """Sample weights using reparameterization trick."""
        w_sigma = self.weight_sigma
        b_sigma = self.bias_sigma

        w_eps = np.random.randn(*self.weight_mu.shape)
        b_eps = np.random.randn(*self.bias_mu.shape)

        w = self.weight_mu + w_sigma * w_eps
        b = self.bias_mu + b_sigma * b_eps

        self._weight_eps = w_eps
        self._bias_eps = b_eps
        self._weight_sigma = w_sigma
        self._bias_sigma = b_sigma
        self._sampled_weight = w
        self._sampled_bias = b

        return w, b

    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """Forward pass with weight sampling."""
        self._input = x.copy()

        if sample:
            w, b = self.sample_weights()
        else:
            w = self.weight_mu
            b = self.bias_mu
            self._sampled_weight = w
            self._sampled_bias = b

        return x @ w + b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass computing gradients for variational parameters."""
        batch_size = grad_output.shape[0]

        # Gradient w.r.t. sampled weight and bias
        grad_w = self._input.T @ grad_output / batch_size
        grad_b = np.mean(grad_output, axis=0)

        # Gradients for mu (same as standard)
        self.grad_weight_mu = grad_w
        self.grad_bias_mu = grad_b

        # Gradients for rho (through reparameterization)
        # d/d_rho = d/d_sigma * d_sigma/d_rho
        # d_sigma/d_rho = sigmoid(rho) (derivative of softplus)
        sigmoid_weight_rho = sigmoid(self.weight_rho)
        sigmoid_bias_rho = sigmoid(self.bias_rho)

        if self._weight_eps is not None:
            self.grad_weight_rho = grad_w * self._weight_eps * sigmoid_weight_rho
            self.grad_bias_rho = grad_b * self._bias_eps * sigmoid_bias_rho
        else:
            self.grad_weight_rho = np.zeros_like(self.weight_rho)
            self.grad_bias_rho = np.zeros_like(self.bias_rho)

        # Gradient to propagate
        grad_input = grad_output @ self._sampled_weight.T
        return grad_input

    def kl_divergence(self) -> float:
        """KL(q(w|theta) || p(w)) for Gaussian q and p."""
        w_sigma = self.weight_sigma
        b_sigma = self.bias_sigma

        # KL for weights
        kl_w = _gaussian_kl(self.weight_mu, w_sigma,
                            self.prior_mu, self.prior_sigma)
        # KL for bias
        kl_b = _gaussian_kl(self.bias_mu, b_sigma,
                            self.prior_mu, self.prior_sigma)
        return kl_w + kl_b

    def get_params(self) -> Dict[str, np.ndarray]:
        return {
            'weight_mu': self.weight_mu.copy(),
            'weight_rho': self.weight_rho.copy(),
            'bias_mu': self.bias_mu.copy(),
            'bias_rho': self.bias_rho.copy(),
        }

    def set_params(self, params: Dict[str, np.ndarray]):
        self.weight_mu = params['weight_mu'].copy()
        self.weight_rho = params['weight_rho'].copy()
        self.bias_mu = params['bias_mu'].copy()
        self.bias_rho = params['bias_rho'].copy()

    def param_count(self) -> int:
        """Total variational parameters (mu + rho for weights and biases)."""
        return (self.in_features * self.out_features * 2 +
                self.out_features * 2)


def _gaussian_kl(mu_q, sigma_q, mu_p, sigma_p) -> float:
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))."""
    sigma_q = np.clip(sigma_q, 1e-8, None)
    if isinstance(sigma_p, (int, float)):
        sigma_p = float(sigma_p)
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    kl = np.sum(
        np.log(sigma_p / sigma_q) +
        (var_q + (mu_q - mu_p) ** 2) / (2 * var_p) - 0.5
    )
    return float(kl)


# ============================================================
# Bayesian Network
# ============================================================

class BayesianNetwork:
    """Multi-layer Bayesian neural network.

    Architecture: input -> [BayesianLinear + activation] * N -> output
    Training via variational inference (ELBO maximization).
    """

    def __init__(self, layer_sizes: List[int],
                 activations: Optional[List[str]] = None,
                 prior_sigma: float = 1.0,
                 init_rho: float = -3.0):
        self.layer_sizes = layer_sizes
        self.layers = []
        self.activation_fns = []
        self.activation_derivs = []

        n_layers = len(layer_sizes) - 1
        if activations is None:
            activations = ['relu'] * (n_layers - 1) + ['none']

        for i in range(n_layers):
            layer = BayesianLinear(
                layer_sizes[i], layer_sizes[i + 1],
                prior_sigma=prior_sigma, init_rho=init_rho
            )
            self.layers.append(layer)
            fn, deriv = ACTIVATIONS[activations[i]]
            self.activation_fns.append(fn)
            self.activation_derivs.append(deriv)

        # Cache for backward
        self._pre_activations = []
        self._post_activations = []

    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        """Forward pass through all layers."""
        self._pre_activations = []
        self._post_activations = [x.copy()]

        h = x
        for i, (layer, act_fn) in enumerate(zip(self.layers, self.activation_fns)):
            z = layer.forward(h, sample=sample)
            self._pre_activations.append(z)
            h = act_fn(z)
            self._post_activations.append(h)

        return h

    def backward(self, grad_output: np.ndarray):
        """Backward pass through all layers."""
        grad = grad_output

        for i in reversed(range(len(self.layers))):
            # Apply activation derivative
            if self.activation_derivs[i] is not None:
                pre_act = self._pre_activations[i]
                grad = grad * self.activation_derivs[i](pre_act)

            grad = self.layers[i].backward(grad)

    def kl_divergence(self) -> float:
        """Total KL divergence across all layers."""
        return sum(layer.kl_divergence() for layer in self.layers)

    def predict(self, x: np.ndarray, n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive distribution via MC sampling.

        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x, sample=True)
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_samples, batch, output)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std

    def get_params(self) -> List[Dict[str, np.ndarray]]:
        return [layer.get_params() for layer in self.layers]

    def set_params(self, params: List[Dict[str, np.ndarray]]):
        for layer, p in zip(self.layers, params):
            layer.set_params(p)

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)


# ============================================================
# Loss Functions
# ============================================================

class GaussianNLLLoss:
    """Negative log-likelihood for regression with learned aleatoric uncertainty.

    -log N(y | mu, sigma^2) = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
    """

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray,
                 log_var: Optional[np.ndarray] = None) -> float:
        if log_var is None:
            # Simple MSE when no aleatoric uncertainty
            return float(0.5 * np.mean((y_pred - y_true) ** 2))

        var = np.exp(log_var)
        nll = 0.5 * np.mean(log_var + (y_pred - y_true) ** 2 / var)
        return float(nll)

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray,
             log_var: Optional[np.ndarray] = None) -> np.ndarray:
        n = y_pred.shape[0]
        if log_var is None:
            return (y_pred - y_true) / n
        var = np.exp(log_var)
        return (y_pred - y_true) / (var * n)


class CrossEntropyLoss:
    """Cross entropy loss for classification."""

    def __call__(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        probs = softmax(logits)
        n = logits.shape[0]
        if y_true.ndim == 1:
            # Integer labels
            log_probs = np.log(np.clip(probs[np.arange(n), y_true.astype(int)], 1e-12, None))
        else:
            # One-hot
            log_probs = np.sum(y_true * np.log(np.clip(probs, 1e-12, None)), axis=-1)
        return float(-np.mean(log_probs))

    def grad(self, logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        probs = softmax(logits)
        n = logits.shape[0]
        if y_true.ndim == 1:
            grad = probs.copy()
            grad[np.arange(n), y_true.astype(int)] -= 1
        else:
            grad = probs - y_true
        return grad / n


# ============================================================
# ELBO (Evidence Lower Bound)
# ============================================================

class ELBO:
    """ELBO = E_q[log p(D|w)] - KL(q(w) || p(w))

    Maximizing ELBO = minimizing (NLL + beta * KL).
    """

    def __init__(self, model: BayesianNetwork, loss_fn,
                 n_batches: int = 1, beta: float = 1.0):
        self.model = model
        self.loss_fn = loss_fn
        self.n_batches = n_batches
        self.beta = beta

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray,
                 **loss_kwargs) -> Tuple[float, Dict[str, float]]:
        """Compute negative ELBO (to minimize)."""
        nll = self.loss_fn(y_pred, y_true, **loss_kwargs)
        kl = self.model.kl_divergence()

        # Scale KL by 1/n_batches (Blundell et al., 2015)
        kl_scaled = (self.beta / self.n_batches) * kl
        neg_elbo = nll + kl_scaled

        return neg_elbo, {
            'nll': nll,
            'kl': kl,
            'kl_scaled': kl_scaled,
            'neg_elbo': neg_elbo,
        }


# ============================================================
# Optimizers for Variational Parameters
# ============================================================

class BayesianSGD:
    """SGD optimizer for Bayesian layers (updates mu and rho)."""

    def __init__(self, model: BayesianNetwork, lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = None

    def _init_velocity(self):
        self._velocity = []
        for layer in self.model.layers:
            v = {
                'weight_mu': np.zeros_like(layer.weight_mu),
                'weight_rho': np.zeros_like(layer.weight_rho),
                'bias_mu': np.zeros_like(layer.bias_mu),
                'bias_rho': np.zeros_like(layer.bias_rho),
            }
            self._velocity.append(v)

    def step(self):
        if self._velocity is None and self.momentum > 0:
            self._init_velocity()

        for i, layer in enumerate(self.model.layers):
            for pname in ['weight_mu', 'weight_rho', 'bias_mu', 'bias_rho']:
                grad = getattr(layer, f'grad_{pname}')
                if grad is None:
                    continue

                param = getattr(layer, pname)

                if self.weight_decay > 0 and 'mu' in pname:
                    grad = grad + self.weight_decay * param

                if self.momentum > 0:
                    self._velocity[i][pname] = (
                        self.momentum * self._velocity[i][pname] + grad
                    )
                    update = self._velocity[i][pname]
                else:
                    update = grad

                setattr(layer, pname, param - self.lr * update)


class BayesianAdam:
    """Adam optimizer for Bayesian layers."""

    def __init__(self, model: BayesianNetwork, lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = None
        self._v = None

    def _init_moments(self):
        self._m = []
        self._v = []
        for layer in self.model.layers:
            m = {
                'weight_mu': np.zeros_like(layer.weight_mu),
                'weight_rho': np.zeros_like(layer.weight_rho),
                'bias_mu': np.zeros_like(layer.bias_mu),
                'bias_rho': np.zeros_like(layer.bias_rho),
            }
            v = {
                'weight_mu': np.zeros_like(layer.weight_mu),
                'weight_rho': np.zeros_like(layer.weight_rho),
                'bias_mu': np.zeros_like(layer.bias_mu),
                'bias_rho': np.zeros_like(layer.bias_rho),
            }
            self._m.append(m)
            self._v.append(v)

    def step(self):
        if self._m is None:
            self._init_moments()

        self.t += 1

        for i, layer in enumerate(self.model.layers):
            for pname in ['weight_mu', 'weight_rho', 'bias_mu', 'bias_rho']:
                grad = getattr(layer, f'grad_{pname}')
                if grad is None:
                    continue

                param = getattr(layer, pname)

                if self.weight_decay > 0 and 'mu' in pname:
                    grad = grad + self.weight_decay * param

                self._m[i][pname] = self.beta1 * self._m[i][pname] + (1 - self.beta1) * grad
                self._v[i][pname] = self.beta2 * self._v[i][pname] + (1 - self.beta2) * grad ** 2

                m_hat = self._m[i][pname] / (1 - self.beta1 ** self.t)
                v_hat = self._v[i][pname] / (1 - self.beta2 ** self.t)

                setattr(layer, pname, param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))


# ============================================================
# MC Dropout (Approximate Bayesian Inference)
# ============================================================

class MCDropoutLayer:
    """Dropout layer that stays active at test time for MC sampling."""

    def __init__(self, p: float = 0.5):
        self.p = p
        self._mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.p == 0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
        return x * self._mask / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.p == 0:
            return grad
        return grad * self._mask / (1 - self.p)


class DenseLayer:
    """Standard dense layer for MC Dropout network."""

    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)
        self.grad_weight = None
        self.grad_bias = None
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x.copy()
        return x @ self.weight + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size = grad.shape[0]
        self.grad_weight = self._input.T @ grad / batch_size
        self.grad_bias = np.mean(grad, axis=0)
        return grad @ self.weight.T


class MCDropoutNetwork:
    """Neural network using MC Dropout for approximate Bayesian inference.

    At test time, run forward pass multiple times with dropout active
    to get a distribution over predictions.
    """

    def __init__(self, layer_sizes: List[int],
                 activations: Optional[List[str]] = None,
                 dropout_rate: float = 0.1):
        self.layer_sizes = layer_sizes
        self.dense_layers = []
        self.dropout_layers = []
        self.activation_fns = []
        self.activation_derivs = []

        n_layers = len(layer_sizes) - 1
        if activations is None:
            activations = ['relu'] * (n_layers - 1) + ['none']

        for i in range(n_layers):
            self.dense_layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1]))
            # Dropout after each hidden layer, not the output
            if i < n_layers - 1:
                self.dropout_layers.append(MCDropoutLayer(dropout_rate))
            else:
                self.dropout_layers.append(MCDropoutLayer(0.0))
            fn, deriv = ACTIVATIONS[activations[i]]
            self.activation_fns.append(fn)
            self.activation_derivs.append(deriv)

        self._pre_activations = []
        self._post_activations = []

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._pre_activations = []
        self._post_activations = [x.copy()]

        h = x
        for dense, dropout, act_fn in zip(self.dense_layers, self.dropout_layers, self.activation_fns):
            z = dense.forward(h)
            self._pre_activations.append(z)
            a = act_fn(z)
            h = dropout.forward(a, training=training)
            self._post_activations.append(h)
        return h

    def backward(self, grad: np.ndarray):
        for i in reversed(range(len(self.dense_layers))):
            grad = self.dropout_layers[i].backward(grad)
            if self.activation_derivs[i] is not None:
                grad = grad * self.activation_derivs[i](self._pre_activations[i])
            grad = self.dense_layers[i].backward(grad)

    def predict(self, x: np.ndarray, n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout prediction with uncertainty."""
        predictions = []
        for _ in range(n_samples):
            pred = self.forward(x, training=True)  # Dropout active
            predictions.append(pred)

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std


# ============================================================
# Deep Ensemble
# ============================================================

class BayesianEnsemble:
    """Deep ensemble for uncertainty estimation.

    Trains multiple networks independently; uncertainty comes from
    disagreement between ensemble members.
    """

    def __init__(self, n_members: int, layer_sizes: List[int],
                 activations: Optional[List[str]] = None):
        self.n_members = n_members
        self.members = []
        for _ in range(n_members):
            net = MCDropoutNetwork(layer_sizes, activations, dropout_rate=0.0)
            self.members.append(net)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble prediction with uncertainty from member disagreement."""
        predictions = []
        for member in self.members:
            pred = member.forward(x, training=False)
            predictions.append(pred)

        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std

    def train_member(self, idx: int, X: np.ndarray, y: np.ndarray,
                     epochs: int = 50, lr: float = 0.01, batch_size: int = 32):
        """Train a single ensemble member."""
        member = self.members[idx]
        n = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                xb = X[batch_idx]
                yb = y[batch_idx]

                pred = member.forward(xb, training=False)
                grad = (pred - yb) / xb.shape[0]
                member.backward(grad)

                for dense in member.dense_layers:
                    if dense.grad_weight is not None:
                        dense.weight -= lr * dense.grad_weight
                        dense.bias -= lr * dense.grad_bias


# ============================================================
# Variational Inference Trainer
# ============================================================

class VariationalInference:
    """Training loop for BNN using ELBO optimization.

    Features:
    - KL annealing (warmup)
    - Mini-batch training
    - Multiple MC samples per step
    - Convergence tracking
    """

    def __init__(self, model: BayesianNetwork, loss_fn,
                 optimizer, n_mc_samples: int = 1,
                 kl_annealing_epochs: int = 0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.n_mc_samples = n_mc_samples
        self.kl_annealing_epochs = kl_annealing_epochs
        self.history = {'neg_elbo': [], 'nll': [], 'kl': []}

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              n_batches: Optional[int] = None,
              verbose: bool = False) -> Dict[str, List[float]]:
        """Train the BNN via variational inference."""
        n = X.shape[0]
        if n_batches is None:
            n_batches = max(1, n // batch_size)

        for epoch in range(epochs):
            # KL annealing
            if self.kl_annealing_epochs > 0 and epoch < self.kl_annealing_epochs:
                beta = (epoch + 1) / self.kl_annealing_epochs
            else:
                beta = 1.0

            elbo_obj = ELBO(self.model, self.loss_fn,
                            n_batches=n_batches, beta=beta)

            epoch_nll = 0.0
            epoch_kl = 0.0
            epoch_elbo = 0.0

            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                xb = X[batch_idx]
                yb = y[batch_idx]

                # MC sampling
                total_grad = None
                batch_nll = 0.0

                for _ in range(self.n_mc_samples):
                    pred = self.model.forward(xb, sample=True)
                    nll = self.loss_fn(pred, yb)
                    batch_nll += nll
                    grad = self.loss_fn.grad(pred, yb)

                    if total_grad is None:
                        total_grad = grad
                    else:
                        total_grad = total_grad + grad

                total_grad /= self.n_mc_samples
                batch_nll /= self.n_mc_samples

                # Add KL gradients
                self.model.backward(total_grad)
                self._add_kl_gradients(beta, n_batches)

                self.optimizer.step()

                kl = self.model.kl_divergence()
                neg_elbo = batch_nll + (beta / n_batches) * kl
                epoch_nll += batch_nll
                epoch_kl += kl
                epoch_elbo += neg_elbo

            n_steps = max(1, n // batch_size)
            self.history['neg_elbo'].append(epoch_elbo / n_steps)
            self.history['nll'].append(epoch_nll / n_steps)
            self.history['kl'].append(epoch_kl / n_steps)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}: ELBO={epoch_elbo / n_steps:.4f}, "
                      f"NLL={epoch_nll / n_steps:.4f}, KL={epoch_kl / n_steps:.4f}")

        return self.history

    def _add_kl_gradients(self, beta: float, n_batches: int):
        """Add KL divergence gradients to the variational parameters."""
        scale = beta / n_batches

        for layer in self.model.layers:
            w_sigma = layer.weight_sigma
            b_sigma = layer.bias_sigma

            # d/d_mu KL = (mu_q - mu_p) / sigma_p^2
            kl_grad_w_mu = (layer.weight_mu - layer.prior_mu) / (layer.prior_sigma ** 2)
            kl_grad_b_mu = (layer.bias_mu - layer.prior_mu) / (layer.prior_sigma ** 2)

            # d/d_sigma KL = sigma_q / sigma_p^2 - 1 / sigma_q
            kl_grad_w_sigma = w_sigma / (layer.prior_sigma ** 2) - 1.0 / w_sigma
            kl_grad_b_sigma = b_sigma / (layer.prior_sigma ** 2) - 1.0 / b_sigma

            # Chain rule: d/d_rho = d/d_sigma * sigmoid(rho)
            sig_w = sigmoid(layer.weight_rho)
            sig_b = sigmoid(layer.bias_rho)

            if layer.grad_weight_mu is not None:
                layer.grad_weight_mu += scale * kl_grad_w_mu
            if layer.grad_bias_mu is not None:
                layer.grad_bias_mu += scale * kl_grad_b_mu
            if layer.grad_weight_rho is not None:
                layer.grad_weight_rho += scale * kl_grad_w_sigma * sig_w
            if layer.grad_bias_rho is not None:
                layer.grad_bias_rho += scale * kl_grad_b_sigma * sig_b


# ============================================================
# Uncertainty Estimator
# ============================================================

class UncertaintyEstimator:
    """Decompose uncertainty into aleatoric and epistemic components.

    Epistemic: uncertainty about the model (reducible with more data)
    Aleatoric: inherent noise in the data (irreducible)
    """

    def __init__(self, model, n_samples: int = 50):
        self.model = model
        self.n_samples = n_samples

    def estimate(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate aleatoric and epistemic uncertainty."""
        predictions = []
        for _ in range(self.n_samples):
            if isinstance(self.model, BayesianNetwork):
                pred = self.model.forward(x, sample=True)
            elif isinstance(self.model, MCDropoutNetwork):
                pred = self.model.forward(x, training=True)
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            predictions.append(pred)

        predictions = np.array(predictions)  # (n_samples, batch, output)

        # Mean prediction
        mean = np.mean(predictions, axis=0)

        # Total uncertainty (variance of predictions)
        total_var = np.var(predictions, axis=0)

        # Epistemic uncertainty: variance of the means
        # (how much the model disagrees with itself)
        epistemic_var = np.var(predictions, axis=0)

        # For BNN with separate aleatoric head, we'd decompose further
        # Here epistemic ~ total since we don't model aleatoric separately
        aleatoric_var = np.zeros_like(total_var)

        return {
            'mean': mean,
            'total_uncertainty': np.sqrt(total_var),
            'epistemic_uncertainty': np.sqrt(epistemic_var),
            'aleatoric_uncertainty': np.sqrt(aleatoric_var),
            'predictions': predictions,
        }

    def calibration_score(self, x: np.ndarray, y_true: np.ndarray,
                          intervals: Optional[List[float]] = None) -> Dict[str, float]:
        """Check if predicted uncertainty is well-calibrated.

        For well-calibrated uncertainty, p% of true values should fall
        within p% prediction intervals.
        """
        if intervals is None:
            intervals = [0.5, 0.8, 0.9, 0.95]

        result = self.estimate(x)
        mean = result['mean']
        std = result['total_uncertainty']

        calibration = {}
        for p in intervals:
            # z-score for this interval
            from scipy.stats import norm as _norm_placeholder
            # Avoid scipy dependency -- use hardcoded z-scores
            z_scores = {0.5: 0.6745, 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
            z = z_scores.get(p, 1.96)

            lower = mean - z * std
            upper = mean + z * std
            in_interval = np.mean((y_true >= lower) & (y_true <= upper))
            calibration[f'p{int(p * 100)}'] = float(in_interval)

        return calibration


# ============================================================
# Predictive Distribution
# ============================================================

class PredictiveDistribution:
    """Full predictive posterior via MC sampling.

    Provides percentiles, credible intervals, and sampling.
    """

    def __init__(self, predictions: np.ndarray):
        """predictions: (n_samples, batch_size, output_dim)"""
        self.predictions = predictions
        self.n_samples = predictions.shape[0]

    @classmethod
    def from_model(cls, model, x: np.ndarray, n_samples: int = 100):
        preds = []
        for _ in range(n_samples):
            if isinstance(model, BayesianNetwork):
                pred = model.forward(x, sample=True)
            elif isinstance(model, MCDropoutNetwork):
                pred = model.forward(x, training=True)
            else:
                raise ValueError(f"Unsupported model type")
            preds.append(pred)
        return cls(np.array(preds))

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.predictions, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self.predictions, axis=0)

    @property
    def variance(self) -> np.ndarray:
        return np.var(self.predictions, axis=0)

    def percentile(self, q: float) -> np.ndarray:
        """Get q-th percentile (0-100)."""
        return np.percentile(self.predictions, q, axis=0)

    def credible_interval(self, p: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Central credible interval."""
        alpha = (1 - p) / 2
        lower = self.percentile(alpha * 100)
        upper = self.percentile((1 - alpha) * 100)
        return lower, upper

    def entropy(self) -> np.ndarray:
        """Differential entropy of predictive distribution (Gaussian approx)."""
        var = self.variance
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2)
        return 0.5 * np.log(2 * np.pi * np.e * np.clip(var, 1e-10, None))

    def mutual_information(self) -> np.ndarray:
        """Mutual information I(y; w | x, D) -- measures epistemic uncertainty.

        MI = H[y|x,D] - E_w[H[y|x,w]]
        For point predictions per sample, E_w[H] = 0, so MI ~ H[y|x,D].
        """
        return self.entropy()

    def sample(self, n: int = 1) -> np.ndarray:
        """Random samples from the predictive distribution."""
        indices = np.random.choice(self.n_samples, size=n, replace=True)
        return self.predictions[indices]


# ============================================================
# Spike-and-Slab Prior
# ============================================================

class SpikeAndSlabLayer:
    """Bayesian linear layer with spike-and-slab prior for sparsity.

    Spike: N(0, sigma_spike^2) -- near-zero weights (pruned)
    Slab:  N(0, sigma_slab^2)  -- active weights
    pi: mixing proportion (probability of slab)
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_spike: float = 0.01, sigma_slab: float = 1.0,
                 pi: float = 0.5):
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_spike = sigma_spike
        self.sigma_slab = sigma_slab
        self.pi = pi

        # Variational parameters
        scale = np.sqrt(2.0 / in_features)
        self.weight_mu = np.random.randn(in_features, out_features) * scale
        self.weight_rho = np.full((in_features, out_features), -3.0)
        self.bias_mu = np.zeros(out_features)
        self.bias_rho = np.full(out_features, -3.0)

        # Inclusion probability (logit space)
        self.alpha_logit = np.zeros((in_features, out_features))

        self._input = None
        self._sampled_weight = None
        self._sampled_bias = None
        self._z_mask = None

        self.grad_weight_mu = None
        self.grad_weight_rho = None
        self.grad_bias_mu = None
        self.grad_bias_rho = None
        self.grad_alpha_logit = None

    @property
    def inclusion_prob(self):
        return sigmoid(self.alpha_logit)

    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        self._input = x.copy()

        w_sigma = softplus(self.weight_rho)
        b_sigma = softplus(self.bias_rho)

        if sample:
            w_eps = np.random.randn(*self.weight_mu.shape)
            b_eps = np.random.randn(*self.bias_mu.shape)
            w = self.weight_mu + w_sigma * w_eps
            b = self.bias_mu + b_sigma * b_eps

            # Sample inclusion mask
            z = (np.random.rand(*self.alpha_logit.shape) < self.inclusion_prob).astype(float)
            self._z_mask = z
            w = w * z
        else:
            w = self.weight_mu * self.inclusion_prob
            b = self.bias_mu

        self._sampled_weight = w
        self._sampled_bias = b
        return x @ w + b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size = grad_output.shape[0]

        grad_w = self._input.T @ grad_output / batch_size
        grad_b = np.mean(grad_output, axis=0)

        if self._z_mask is not None:
            self.grad_weight_mu = grad_w * self._z_mask
        else:
            self.grad_weight_mu = grad_w
        self.grad_bias_mu = grad_b
        self.grad_weight_rho = np.zeros_like(self.weight_rho)
        self.grad_bias_rho = np.zeros_like(self.bias_rho)
        self.grad_alpha_logit = np.zeros_like(self.alpha_logit)

        return grad_output @ self._sampled_weight.T

    def kl_divergence(self) -> float:
        """KL for spike-and-slab prior."""
        alpha = self.inclusion_prob
        w_sigma = softplus(self.weight_rho)

        # KL for continuous part (Gaussian)
        kl_gauss = _gaussian_kl(self.weight_mu, w_sigma, 0.0, self.sigma_slab)

        # KL for discrete part (Bernoulli)
        alpha_clip = np.clip(alpha, 1e-8, 1 - 1e-8)
        kl_bern = np.sum(
            alpha_clip * np.log(alpha_clip / self.pi) +
            (1 - alpha_clip) * np.log((1 - alpha_clip) / (1 - self.pi))
        )

        # Bias KL
        b_sigma = softplus(self.bias_rho)
        kl_b = _gaussian_kl(self.bias_mu, b_sigma, 0.0, self.sigma_slab)

        return float(kl_gauss + kl_bern + kl_b)

    def effective_params(self) -> int:
        """Number of 'active' weights (inclusion prob > 0.5)."""
        return int(np.sum(self.inclusion_prob > 0.5))

    def sparsity(self) -> float:
        """Fraction of weights pruned."""
        total = self.in_features * self.out_features
        active = self.effective_params()
        return 1.0 - active / total if total > 0 else 0.0


# ============================================================
# Bayes By Backprop (Blundell et al., 2015)
# ============================================================

class BayesByBackprop:
    """Convenience wrapper implementing the full Bayes by Backprop algorithm.

    1. Sample weights from variational posterior
    2. Forward pass
    3. Compute negative ELBO
    4. Backward pass
    5. Update variational parameters
    """

    def __init__(self, layer_sizes: List[int],
                 activations: Optional[List[str]] = None,
                 prior_sigma: float = 1.0,
                 lr: float = 0.001,
                 n_mc_samples: int = 1,
                 kl_annealing_epochs: int = 10):
        self.model = BayesianNetwork(
            layer_sizes, activations, prior_sigma=prior_sigma
        )
        self.loss_fn = GaussianNLLLoss()
        self.optimizer = BayesianAdam(self.model, lr=lr)
        self.trainer = VariationalInference(
            self.model, self.loss_fn, self.optimizer,
            n_mc_samples=n_mc_samples,
            kl_annealing_epochs=kl_annealing_epochs
        )

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = False) -> Dict[str, List[float]]:
        return self.trainer.train(X, y, epochs=epochs,
                                  batch_size=batch_size, verbose=verbose)

    def predict(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X, n_samples=n_samples)

    def predictive_distribution(self, X: np.ndarray,
                                n_samples: int = 100) -> PredictiveDistribution:
        return PredictiveDistribution.from_model(self.model, X, n_samples)


# ============================================================
# Prior Types
# ============================================================

class ScaleMixturePrior:
    """Scale mixture of two Gaussians prior (Blundell et al., 2015).

    p(w) = pi * N(0, sigma1^2) + (1-pi) * N(0, sigma2^2)
    """

    def __init__(self, pi: float = 0.5, sigma1: float = 1.0, sigma2: float = 0.002):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def log_prob(self, w: np.ndarray) -> float:
        """Log probability under the mixture prior."""
        log_p1 = -0.5 * np.log(2 * np.pi * self.sigma1 ** 2) - w ** 2 / (2 * self.sigma1 ** 2)
        log_p2 = -0.5 * np.log(2 * np.pi * self.sigma2 ** 2) - w ** 2 / (2 * self.sigma2 ** 2)

        # Log-sum-exp for numerical stability
        log_pi = np.log(self.pi)
        log_1_pi = np.log(1 - self.pi)
        max_log = np.maximum(log_pi + log_p1, log_1_pi + log_p2)
        log_prob = max_log + np.log(
            np.exp(log_pi + log_p1 - max_log) +
            np.exp(log_1_pi + log_p2 - max_log)
        )
        return float(np.sum(log_prob))


# ============================================================
# Utilities
# ============================================================

def bnn_regression_demo(n_points: int = 100, noise: float = 0.3,
                        seed: int = 42) -> Dict[str, Any]:
    """Demo: BNN regression with uncertainty on a sine wave."""
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, (n_points, 1))
    y = np.sin(X) + np.random.randn(n_points, 1) * noise

    bbb = BayesByBackprop(
        layer_sizes=[1, 32, 32, 1],
        activations=['relu', 'relu', 'none'],
        lr=0.001, n_mc_samples=2,
        kl_annealing_epochs=10
    )
    history = bbb.fit(X, y, epochs=50, batch_size=32)

    # Predict on test range
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    mean, std = bbb.predict(X_test, n_samples=50)

    return {
        'model': bbb,
        'X_train': X, 'y_train': y,
        'X_test': X_test, 'mean': mean, 'std': std,
        'history': history,
    }


def expected_calibration_error(y_true: np.ndarray, y_pred_mean: np.ndarray,
                               y_pred_std: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error for regression.

    Measures how well predicted uncertainty matches observed error.
    """
    errors = np.abs(y_true - y_pred_mean).flatten()
    uncertainties = y_pred_std.flatten()

    # Sort by predicted uncertainty
    sort_idx = np.argsort(uncertainties)
    errors = errors[sort_idx]
    uncertainties = uncertainties[sort_idx]

    bin_size = len(errors) // n_bins
    if bin_size == 0:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(errors)
        bin_err = np.mean(errors[start:end])
        bin_unc = np.mean(uncertainties[start:end])
        ece += abs(bin_err - bin_unc) * (end - start) / len(errors)

    return float(ece)
