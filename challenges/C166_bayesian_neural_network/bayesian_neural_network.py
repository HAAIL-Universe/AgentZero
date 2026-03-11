"""
C166: Bayesian Neural Network
Composing C156 (Probabilistic Programming) + C140 (Neural Network)

Places probability distributions over neural network weights for:
- Epistemic uncertainty quantification (model uncertainty)
- Aleatoric uncertainty estimation (data noise)
- Automatic regularization through priors
- Predictive uncertainty decomposition

Components:
1. BayesianLinear -- Dense layer with variational weight distributions
2. BayesianNetwork -- Container for Bayesian layers
3. BayesByBackprop -- Variational inference training (Blundell et al. 2015)
4. MCDropoutNetwork -- MC Dropout approximate inference (Gal & Ghahramani 2016)
5. LaplaceApproximation -- Post-hoc Gaussian approximation around MAP
6. BNNPredictive -- Predictive distributions with uncertainty decomposition
7. UncertaintyMetrics -- Calibration, sharpness, coverage analysis
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C156_probabilistic_programming'))

from neural_network import (
    Tensor, Dense, Activation, Dropout, Sequential, MSELoss, CrossEntropyLoss,
    BinaryCrossEntropyLoss, SGD, Adam, fit, train_step, evaluate, predict_classes,
    accuracy, one_hot, normalize, xavier_init, he_init
)
from probabilistic_programming import (
    Normal, HalfNormal, HalfCauchy, Gamma,
    ProbModel, Trace, MCMCInference, VIInference
)


# ---------------------------------------------------------------------------
# BayesianLinear -- Variational dense layer
# ---------------------------------------------------------------------------

class BayesianLinear:
    """Dense layer with variational weight distributions.

    Each weight has a variational posterior q(w) = N(mu, softplus(rho)^2).
    Prior is N(0, prior_sigma^2) or scale mixture prior.
    """

    def __init__(self, input_size, output_size, prior_sigma=1.0,
                 prior_mix=False, seed=42):
        self.input_size = input_size
        self.output_size = output_size
        self.prior_sigma = prior_sigma
        self.prior_mix = prior_mix  # scale mixture of two Gaussians
        self.rng = np.random.RandomState(seed)

        # Variational parameters: mu and rho (sigma = softplus(rho))
        scale = 1.0 / np.sqrt(input_size)
        self.weight_mu = self.rng.randn(input_size, output_size) * scale
        self.weight_rho = np.full((input_size, output_size), -3.0)
        self.bias_mu = np.zeros(output_size)
        self.bias_rho = np.full(output_size, -3.0)

        # Gradients
        self.grad_weight_mu = np.zeros_like(self.weight_mu)
        self.grad_weight_rho = np.zeros_like(self.weight_rho)
        self.grad_bias_mu = np.zeros_like(self.bias_mu)
        self.grad_bias_rho = np.zeros_like(self.bias_rho)

        # Cached values for backward pass
        self._input = None
        self._weight_sample = None
        self._bias_sample = None
        self._epsilon_w = None
        self._epsilon_b = None
        self.training = True

    @staticmethod
    def softplus(x):
        """Numerically stable softplus: log(1 + exp(x))."""
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -20, 20))))

    @staticmethod
    def softplus_deriv(x):
        """Derivative of softplus = sigmoid."""
        clipped = np.clip(x, -20, 20)
        return 1.0 / (1.0 + np.exp(-clipped))

    def weight_sigma(self):
        return self.softplus(self.weight_rho)

    def bias_sigma(self):
        return self.softplus(self.bias_rho)

    def sample_weights(self):
        """Sample weights using reparameterization trick."""
        w_sigma = self.weight_sigma()
        b_sigma = self.bias_sigma()
        self._epsilon_w = self.rng.randn(*self.weight_mu.shape)
        self._epsilon_b = self.rng.randn(*self.bias_mu.shape)
        self._weight_sample = self.weight_mu + w_sigma * self._epsilon_w
        self._bias_sample = self.bias_mu + b_sigma * self._epsilon_b
        return self._weight_sample, self._bias_sample

    def forward(self, x):
        """Forward pass with sampled weights."""
        if isinstance(x, Tensor):
            x = np.array(x.data)
        self._input = x

        if self.training:
            w, b = self.sample_weights()
        else:
            w, b = self.weight_mu, self.bias_mu

        out = x @ w + b
        return out

    def backward(self, grad_output):
        """Backward pass computing gradients for variational parameters."""
        # grad w.r.t. weights (from likelihood)
        if self._input.ndim == 1:
            inp = self._input.reshape(1, -1)
            grad = grad_output.reshape(1, -1)
        else:
            inp = self._input
            grad = grad_output

        # Gradient for weight_mu: same as standard layer
        self.grad_weight_mu = inp.T @ grad
        self.grad_bias_mu = grad.sum(axis=0)

        # Gradient for weight_rho: chain through reparameterization
        # d/d_rho = d/d_sigma * d_sigma/d_rho
        # d/d_sigma = grad_output * epsilon (from reparam trick)
        if self._epsilon_w is not None:
            sigma_grad_w = inp.T @ grad * self._epsilon_w
            self.grad_weight_rho = sigma_grad_w * self.softplus_deriv(self.weight_rho)
            sigma_grad_b = grad.sum(axis=0) * self._epsilon_b
            self.grad_bias_rho = sigma_grad_b * self.softplus_deriv(self.bias_rho)

        # Propagate gradient to previous layer
        if self._weight_sample is not None:
            grad_input = grad @ self._weight_sample.T
        else:
            grad_input = grad @ self.weight_mu.T

        if self._input.ndim == 1:
            grad_input = grad_input.reshape(-1)
        return grad_input

    def log_variational_posterior(self):
        """Log q(w|theta) -- variational posterior density at sampled weights."""
        if self._weight_sample is None:
            return 0.0
        w_sigma = self.weight_sigma()
        b_sigma = self.bias_sigma()
        log_q_w = _log_normal(self._weight_sample, self.weight_mu, w_sigma)
        log_q_b = _log_normal(self._bias_sample, self.bias_mu, b_sigma)
        return log_q_w + log_q_b

    def log_prior(self):
        """Log p(w) -- prior density at sampled weights."""
        if self._weight_sample is None:
            return 0.0
        if self.prior_mix:
            log_p_w = _log_scale_mixture(self._weight_sample,
                                          sigma1=self.prior_sigma,
                                          sigma2=self.prior_sigma * 0.01)
            log_p_b = _log_scale_mixture(self._bias_sample,
                                          sigma1=self.prior_sigma,
                                          sigma2=self.prior_sigma * 0.01)
        else:
            log_p_w = _log_normal(self._weight_sample, 0.0, self.prior_sigma)
            log_p_b = _log_normal(self._bias_sample, 0.0, self.prior_sigma)
        return log_p_w + log_p_b

    def kl_divergence(self):
        """KL(q||p) = log q - log p."""
        return self.log_variational_posterior() - self.log_prior()

    def get_params(self):
        """Return variational parameters and their gradients."""
        return [
            (self.weight_mu, self.grad_weight_mu, 'weight_mu'),
            (self.weight_rho, self.grad_weight_rho, 'weight_rho'),
            (self.bias_mu, self.grad_bias_mu, 'bias_mu'),
            (self.bias_rho, self.grad_bias_rho, 'bias_rho'),
        ]

    def num_params(self):
        return self.weight_mu.size + self.bias_mu.size

    def set_training(self, mode):
        self.training = mode


# ---------------------------------------------------------------------------
# Activation wrapper for BNN (reuses C140 functions)
# ---------------------------------------------------------------------------

class BayesianActivation:
    """Activation function for Bayesian network (numpy-based)."""

    def __init__(self, name='relu', alpha=0.01):
        self.name = name
        self.alpha = alpha
        self._input = None
        self.training = True

    def forward(self, x):
        self._input = x
        if self.name == 'relu':
            return np.maximum(0, x)
        elif self.name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif self.name == 'tanh':
            return np.tanh(x)
        elif self.name == 'leaky_relu':
            return np.where(x > 0, x, self.alpha * x)
        elif self.name in ('linear', 'none'):
            return x
        elif self.name == 'softmax':
            shifted = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(shifted)
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return x

    def backward(self, grad_output):
        x = self._input
        if self.name == 'relu':
            return grad_output * (x > 0).astype(float)
        elif self.name == 'sigmoid':
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return grad_output * s * (1 - s)
        elif self.name == 'tanh':
            return grad_output * (1 - np.tanh(x) ** 2)
        elif self.name == 'leaky_relu':
            return grad_output * np.where(x > 0, 1.0, self.alpha)
        elif self.name in ('linear', 'none'):
            return grad_output
        elif self.name == 'softmax':
            # Simplified: used with cross-entropy which handles Jacobian
            return grad_output
        return grad_output

    def get_params(self):
        return []

    def set_training(self, mode):
        self.training = mode


# ---------------------------------------------------------------------------
# BayesianNetwork -- Container
# ---------------------------------------------------------------------------

class BayesianNetwork:
    """Container for Bayesian neural network layers."""

    def __init__(self, layers=None):
        self.layers = layers or []
        self.training = True

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        self.train(False)
        out = self.forward(x)
        self.train(True)
        return out

    def train(self, mode=True):
        self.training = mode
        for layer in self.layers:
            layer.set_training(mode)

    def eval(self):
        self.train(False)

    def kl_divergence(self):
        """Total KL divergence across all Bayesian layers."""
        kl = 0.0
        for layer in self.layers:
            if hasattr(layer, 'kl_divergence'):
                kl += layer.kl_divergence()
        return kl

    def log_prior(self):
        total = 0.0
        for layer in self.layers:
            if hasattr(layer, 'log_prior'):
                total += layer.log_prior()
        return total

    def log_variational_posterior(self):
        total = 0.0
        for layer in self.layers:
            if hasattr(layer, 'log_variational_posterior'):
                total += layer.log_variational_posterior()
        return total

    def bayesian_layers(self):
        return [l for l in self.layers if isinstance(l, BayesianLinear)]

    def get_params(self):
        params = []
        for i, layer in enumerate(self.layers):
            for param, grad, name in layer.get_params():
                params.append((param, grad, f'layer_{i}_{name}'))
        return params

    def num_params(self):
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'num_params'):
                total += layer.num_params()
        return total

    def sample_weights(self):
        """Sample all Bayesian layer weights."""
        for layer in self.layers:
            if hasattr(layer, 'sample_weights'):
                layer.sample_weights()

    def get_weight_snapshot(self):
        """Get current weight means (for Laplace approximation)."""
        snapshot = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BayesianLinear):
                snapshot[f'layer_{i}_weight'] = layer.weight_mu.copy()
                snapshot[f'layer_{i}_bias'] = layer.bias_mu.copy()
        return snapshot

    def set_weights_from_snapshot(self, snapshot):
        """Set weight means from snapshot."""
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BayesianLinear):
                key_w = f'layer_{i}_weight'
                key_b = f'layer_{i}_bias'
                if key_w in snapshot:
                    layer.weight_mu = snapshot[key_w].copy()
                if key_b in snapshot:
                    layer.bias_mu = snapshot[key_b].copy()


# ---------------------------------------------------------------------------
# BayesByBackprop -- Variational training
# ---------------------------------------------------------------------------

class BayesByBackprop:
    """Bayes by Backprop (Blundell et al. 2015).

    Minimizes the variational free energy:
    F(theta) = KL(q(w|theta) || p(w)) - E_q[log p(D|w)]

    Uses the ELBO loss with KL cost annealing.
    """

    def __init__(self, network, loss_fn=None, lr=0.001, n_samples=1,
                 kl_weight=1.0, seed=42):
        self.network = network
        self.loss_fn = loss_fn or _NumpyMSELoss()
        self.lr = lr
        self.n_samples = n_samples
        self.kl_weight = kl_weight
        self.rng = np.random.RandomState(seed)

        # Adam state for each parameter
        self._m = {}
        self._v = {}
        self._t = 0

    def train_step(self, x, y, kl_weight=None):
        """Single training step. Returns (elbo_loss, nll, kl)."""
        if isinstance(x, Tensor):
            x = np.array(x.data)
        if isinstance(y, Tensor):
            y = np.array(y.data)

        kl_w = kl_weight if kl_weight is not None else self.kl_weight

        total_nll = 0.0
        total_kl = 0.0

        # Zero gradients
        for layer in self.network.layers:
            if isinstance(layer, BayesianLinear):
                layer.grad_weight_mu[:] = 0
                layer.grad_weight_rho[:] = 0
                layer.grad_bias_mu[:] = 0
                layer.grad_bias_rho[:] = 0

        for _ in range(self.n_samples):
            # Forward with sampled weights
            output = self.network.forward(x)
            nll = self.loss_fn.forward(output, y)
            total_nll += nll

            kl = self.network.kl_divergence()
            total_kl += kl

            # Backward for likelihood
            grad = self.loss_fn.backward(output, y)
            self.network.backward(grad)

            # Add KL gradients for variational parameters
            for layer in self.network.layers:
                if isinstance(layer, BayesianLinear):
                    # KL gradient for mu: d/dmu [log q - log p]
                    w_sigma = layer.weight_sigma()
                    b_sigma = layer.bias_sigma()

                    # d(log q)/d(mu) = (w - mu) / sigma^2 -- but negative
                    # d(log p)/d(mu) = -w / prior_sigma^2
                    # Combined KL grad for mu:
                    if layer._weight_sample is not None:
                        kl_grad_w_mu = (layer._weight_sample - layer.weight_mu) / (w_sigma ** 2 + 1e-8)
                        kl_grad_w_mu -= layer._weight_sample / (layer.prior_sigma ** 2)
                        layer.grad_weight_mu += kl_w * (-kl_grad_w_mu) / self.n_samples

                        kl_grad_b_mu = (layer._bias_sample - layer.bias_mu) / (b_sigma ** 2 + 1e-8)
                        kl_grad_b_mu -= layer._bias_sample / (layer.prior_sigma ** 2)
                        layer.grad_bias_mu += kl_w * (-kl_grad_b_mu) / self.n_samples

        total_nll /= self.n_samples
        total_kl /= self.n_samples
        elbo_loss = total_nll + kl_w * total_kl

        # Update parameters with Adam
        self._adam_update()

        return elbo_loss, total_nll, total_kl

    def fit(self, X, Y, epochs=100, batch_size=None, kl_annealing=False,
            verbose=False):
        """Train the BNN. Returns training history."""
        if isinstance(X, Tensor):
            X = np.array(X.data)
        if isinstance(Y, Tensor):
            Y = np.array(Y.data)

        n = len(X)
        if batch_size is None:
            batch_size = n

        history = {'elbo': [], 'nll': [], 'kl': []}

        for epoch in range(epochs):
            # KL annealing: linearly increase from 0 to kl_weight
            if kl_annealing:
                kl_w = self.kl_weight * min(1.0, (epoch + 1) / (epochs * 0.5))
            else:
                kl_w = self.kl_weight

            indices = self.rng.permutation(n)
            epoch_elbo = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]
                x_batch = X[idx]
                y_batch = Y[idx]

                elbo, nll, kl = self.train_step(x_batch, y_batch, kl_weight=kl_w)
                epoch_elbo += elbo
                epoch_nll += nll
                epoch_kl += kl
                n_batches += 1

            history['elbo'].append(epoch_elbo / n_batches)
            history['nll'].append(epoch_nll / n_batches)
            history['kl'].append(epoch_kl / n_batches)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}: ELBO={history['elbo'][-1]:.4f} "
                      f"NLL={history['nll'][-1]:.4f} KL={history['kl'][-1]:.4f}")

        return history

    def _adam_update(self, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer step for all variational parameters."""
        self._t += 1
        for param, grad, name in self.network.get_params():
            if name not in self._m:
                self._m[name] = np.zeros_like(param)
                self._v[name] = np.zeros_like(param)

            self._m[name] = beta1 * self._m[name] + (1 - beta1) * grad
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * grad ** 2

            m_hat = self._m[name] / (1 - beta1 ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


# ---------------------------------------------------------------------------
# MCDropoutNetwork -- MC Dropout approximate BNN
# ---------------------------------------------------------------------------

class MCDropoutNetwork:
    """MC Dropout for approximate Bayesian inference (Gal & Ghahramani 2016).

    Wraps a standard C140 Sequential network. At test time, runs multiple
    forward passes with dropout enabled to estimate predictive uncertainty.
    """

    def __init__(self, model, drop_rate=0.1):
        """
        Args:
            model: C140 Sequential model (must contain Dropout layers)
            drop_rate: dropout rate to use
        """
        self.model = model
        self.drop_rate = drop_rate

    def predict_with_uncertainty(self, x, n_samples=50, seed=42):
        """Run n_samples forward passes with dropout to estimate uncertainty.

        Returns:
            predictions: (n_samples, batch_size, output_dim)
            mean: (batch_size, output_dim)
            std: (batch_size, output_dim) -- epistemic uncertainty
        """
        if isinstance(x, Tensor):
            x_np = np.array(x.data)
        else:
            x_np = np.array(x)

        rng = np.random.RandomState(seed)

        # Enable training mode (keeps dropout active)
        self.model.train()

        predictions = []
        for i in range(n_samples):
            # Set different RNG for each dropout sample
            for layer in self.model.layers:
                if isinstance(layer, Dropout):
                    layer.rng = np.random.RandomState(rng.randint(0, 2**31))

            x_tensor = Tensor(x_np.tolist())
            out = self.model.forward(x_tensor)
            predictions.append(np.array(out.data))

        self.model.eval()

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        return predictions, mean, std

    def predictive_entropy(self, x, n_samples=50, seed=42):
        """Compute predictive entropy (for classification)."""
        predictions, mean, _ = self.predict_with_uncertainty(x, n_samples, seed)
        # For classification: entropy of mean predictions
        mean_clipped = np.clip(mean, 1e-10, 1.0)
        entropy = -np.sum(mean_clipped * np.log(mean_clipped), axis=-1)
        return entropy

    def mutual_information(self, x, n_samples=50, seed=42):
        """BALD: mutual information between predictions and model parameters.

        MI = H[y|x,D] - E_w[H[y|x,w]]
        High MI = high epistemic uncertainty = informative to label.
        """
        predictions, mean, _ = self.predict_with_uncertainty(x, n_samples, seed)

        # Total entropy H[y|x,D]
        mean_clipped = np.clip(mean, 1e-10, 1.0)
        total_entropy = -np.sum(mean_clipped * np.log(mean_clipped), axis=-1)

        # Expected entropy E_w[H[y|x,w]]
        pred_clipped = np.clip(predictions, 1e-10, 1.0)
        per_sample_entropy = -np.sum(pred_clipped * np.log(pred_clipped), axis=-1)
        expected_entropy = per_sample_entropy.mean(axis=0)

        mi = total_entropy - expected_entropy
        return mi


# ---------------------------------------------------------------------------
# LaplaceApproximation -- Post-hoc BNN
# ---------------------------------------------------------------------------

class LaplaceApproximation:
    """Laplace approximation for post-hoc BNN uncertainty.

    Fits a Gaussian N(theta_MAP, H^{-1}) around the MAP estimate,
    where H is the (approximate) Hessian of the loss.
    """

    def __init__(self, network, loss_fn=None, prior_precision=1.0):
        self.network = network
        self.loss_fn = loss_fn or _NumpyMSELoss()
        self.prior_precision = prior_precision
        self._map_weights = None
        self._hessian_diag = None
        self._posterior_var = None

    def fit(self, X, Y, n_epochs=100, lr=0.01, batch_size=None):
        """Train to MAP estimate, then compute Hessian approximation."""
        if isinstance(X, Tensor):
            X = np.array(X.data)
        if isinstance(Y, Tensor):
            Y = np.array(Y.data)

        # Phase 1: Train to MAP estimate
        self.network.train(True)
        n = len(X)
        if batch_size is None:
            batch_size = n

        rng = np.random.RandomState(42)
        optimizer_state = {}

        for epoch in range(n_epochs):
            indices = rng.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                output = self.network.forward(X[idx])
                loss = self.loss_fn.forward(output, Y[idx])
                grad = self.loss_fn.backward(output, Y[idx])
                self.network.backward(grad)

                # SGD with L2 regularization (MAP = MLE + prior)
                for param, grad_p, name in self.network.get_params():
                    # Add prior gradient (L2 regularization)
                    total_grad = grad_p + self.prior_precision * param
                    param -= lr * total_grad

        # Save MAP weights
        self._map_weights = {}
        for param, _, name in self.network.get_params():
            self._map_weights[name] = param.copy()

        # Phase 2: Approximate diagonal Hessian via Gauss-Newton
        self._compute_diagonal_hessian(X, Y)

    def _compute_diagonal_hessian(self, X, Y):
        """Compute diagonal Hessian approximation using gradient squares."""
        n = len(X)
        hessian = {}

        for param, _, name in self.network.get_params():
            hessian[name] = np.zeros_like(param)

        # Accumulate squared gradients as Hessian approximation
        for i in range(n):
            x_i = X[i:i+1]
            y_i = Y[i:i+1]

            output = self.network.forward(x_i)
            grad = self.loss_fn.backward(output, y_i)
            self.network.backward(grad)

            for param, grad_p, name in self.network.get_params():
                hessian[name] += grad_p ** 2

        # Add prior precision to diagonal
        for name in hessian:
            hessian[name] = hessian[name] + self.prior_precision

        self._hessian_diag = hessian
        self._posterior_var = {name: 1.0 / (h + 1e-8)
                               for name, h in hessian.items()}

    def predict_with_uncertainty(self, x, n_samples=50, seed=42):
        """Sample from Laplace posterior and predict."""
        if isinstance(x, Tensor):
            x = np.array(x.data)

        rng = np.random.RandomState(seed)
        predictions = []

        for _ in range(n_samples):
            # Sample weights from Laplace posterior
            for param, _, name in self.network.get_params():
                if name in self._map_weights:
                    std = np.sqrt(self._posterior_var[name])
                    param[:] = self._map_weights[name] + rng.randn(*param.shape) * std

            self.network.train(False)
            output = self.network.forward(x)
            predictions.append(output.copy())

        # Restore MAP weights
        for param, _, name in self.network.get_params():
            if name in self._map_weights:
                param[:] = self._map_weights[name]

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return predictions, mean, std

    def log_marginal_likelihood(self, X, Y):
        """Approximate log marginal likelihood (Laplace evidence)."""
        if isinstance(X, Tensor):
            X = np.array(X.data)
        if isinstance(Y, Tensor):
            Y = np.array(Y.data)

        # Restore MAP weights
        for param, _, name in self.network.get_params():
            if name in self._map_weights:
                param[:] = self._map_weights[name]

        output = self.network.forward(X)
        nll = self.loss_fn.forward(output, Y)

        # Log prior at MAP
        log_prior = 0.0
        for param, _, name in self.network.get_params():
            log_prior -= 0.5 * self.prior_precision * np.sum(param ** 2)

        # Log determinant of posterior precision (Hessian)
        log_det_H = 0.0
        for name, h in self._hessian_diag.items():
            log_det_H += np.sum(np.log(h + 1e-8))

        # n_params
        n_params = sum(p.size for p, _, _ in self.network.get_params())

        # Laplace evidence: log p(D) ≈ log p(D|w_MAP) + log p(w_MAP) + (d/2)log(2pi) - 0.5 log|H|
        lml = -nll + log_prior + 0.5 * n_params * np.log(2 * np.pi) - 0.5 * log_det_H

        return lml


# ---------------------------------------------------------------------------
# BNNPredictive -- Predictive distribution with uncertainty decomposition
# ---------------------------------------------------------------------------

class BNNPredictive:
    """Predictive distribution for BNNs with uncertainty decomposition.

    Decomposes total predictive uncertainty into:
    - Epistemic uncertainty: model uncertainty (reducible with more data)
    - Aleatoric uncertainty: data noise (irreducible)
    """

    def __init__(self, network, method='bbb'):
        """
        Args:
            network: BayesianNetwork
            method: 'bbb' for Bayes by Backprop, 'map' for point estimate
        """
        self.network = network
        self.method = method

    def predict(self, x, n_samples=50, seed=42):
        """Generate predictive distribution.

        Returns dict with:
            mean: predictive mean
            std: total predictive std
            epistemic: epistemic uncertainty (std of means)
            aleatoric: aleatoric uncertainty (mean of stds) -- if heteroscedastic
            samples: raw prediction samples
        """
        if isinstance(x, Tensor):
            x = np.array(x.data)

        rng = np.random.RandomState(seed)
        self.network.train(True)

        predictions = []
        for i in range(n_samples):
            # Reseed each Bayesian layer
            for layer in self.network.layers:
                if isinstance(layer, BayesianLinear):
                    layer.rng = np.random.RandomState(rng.randint(0, 2**31))

            output = self.network.forward(x)
            predictions.append(output.copy())

        self.network.train(False)

        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        total_std = predictions.std(axis=0)

        # Epistemic uncertainty = variance of predictions across weight samples
        epistemic = total_std  # For homoscedastic models, this is the epistemic part

        result = {
            'mean': mean,
            'std': total_std,
            'epistemic': epistemic,
            'samples': predictions,
        }
        return result

    def credible_interval(self, x, prob=0.95, n_samples=100, seed=42):
        """Compute credible interval from predictive distribution."""
        pred = self.predict(x, n_samples, seed)
        samples = pred['samples']

        alpha = (1 - prob) / 2
        lower = np.percentile(samples, 100 * alpha, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha), axis=0)

        return {
            'mean': pred['mean'],
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
        }

    def expected_improvement(self, x, best_y, n_samples=100, seed=42,
                              maximize=True):
        """Expected improvement acquisition function (for Bayesian optimization)."""
        pred = self.predict(x, n_samples, seed)
        samples = pred['samples']

        if maximize:
            improvements = np.maximum(0, samples - best_y)
        else:
            improvements = np.maximum(0, best_y - samples)

        ei = improvements.mean(axis=0)
        return ei


# ---------------------------------------------------------------------------
# UncertaintyMetrics -- Calibration and analysis
# ---------------------------------------------------------------------------

class UncertaintyMetrics:
    """Calibration and uncertainty quality metrics for BNNs."""

    @staticmethod
    def calibration_curve(y_true, y_mean, y_std, n_bins=10):
        """Compute calibration curve for regression.

        For each confidence level, check if the observed coverage matches.

        Returns:
            expected: expected coverage levels
            observed: observed coverage at each level
            calibration_error: mean absolute calibration error
        """
        expected = np.linspace(0.1, 0.99, n_bins)
        observed = []

        for p in expected:
            from scipy.stats import norm
            z = norm.ppf((1 + p) / 2)
            lower = y_mean - z * y_std
            upper = y_mean + z * y_std
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            observed.append(coverage)

        observed = np.array(observed)
        calibration_error = np.mean(np.abs(expected - observed))

        return {
            'expected': expected,
            'observed': observed,
            'calibration_error': calibration_error,
        }

    @staticmethod
    def sharpness(y_std):
        """Sharpness: average predictive standard deviation.
        Lower is better (more confident), but must be calibrated.
        """
        return float(np.mean(y_std))

    @staticmethod
    def coverage(y_true, y_mean, y_std, level=0.95):
        """Coverage: fraction of true values within credible interval."""
        from scipy.stats import norm
        z = norm.ppf((1 + level) / 2)
        lower = y_mean - z * y_std
        upper = y_mean + z * y_std
        cov = np.mean((y_true >= lower) & (y_true <= upper))
        return float(cov)

    @staticmethod
    def interval_score(y_true, y_mean, y_std, alpha=0.05):
        """Interval score (Gneiting & Raftery 2007).
        Proper scoring rule for interval forecasts.
        """
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        lower = y_mean - z * y_std
        upper = y_mean + z * y_std
        width = upper - lower

        penalty_lower = (2 / alpha) * np.maximum(0, lower - y_true)
        penalty_upper = (2 / alpha) * np.maximum(0, y_true - upper)

        score = width + penalty_lower + penalty_upper
        return float(np.mean(score))

    @staticmethod
    def crps(y_true, y_mean, y_std):
        """Continuous Ranked Probability Score.
        Proper scoring rule for probabilistic forecasts.
        CRPS for Gaussian: sigma * [z*Phi(z) + phi(z) - 1/sqrt(pi)]
        where z = (y - mu) / sigma
        """
        from scipy.stats import norm
        z = (y_true - y_mean) / (y_std + 1e-8)
        crps_values = y_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        return float(np.mean(np.abs(crps_values)))

    @staticmethod
    def nll(y_true, y_mean, y_std):
        """Negative log-likelihood under Gaussian predictive distribution."""
        var = y_std ** 2 + 1e-8
        nll_values = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - y_mean) ** 2 / var
        return float(np.mean(nll_values))

    @staticmethod
    def ece(y_true_classes, predicted_probs, n_bins=10):
        """Expected Calibration Error for classification.

        Args:
            y_true_classes: true class labels (integers)
            predicted_probs: predicted probability of predicted class
            n_bins: number of confidence bins
        """
        confidences = np.max(predicted_probs, axis=-1) if predicted_probs.ndim > 1 else predicted_probs
        if predicted_probs.ndim > 1:
            predicted_classes = np.argmax(predicted_probs, axis=-1)
        else:
            predicted_classes = (predicted_probs > 0.5).astype(int)

        correct = (predicted_classes == y_true_classes).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0

        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = confidences[mask].mean()
                ece_val += mask.sum() * np.abs(bin_acc - bin_conf)

        return float(ece_val / len(y_true_classes))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _log_normal(x, mu, sigma):
    """Log density of Normal(mu, sigma) at x. Summed over all elements."""
    return np.sum(-0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-8)
                  - 0.5 * ((x - mu) / (sigma + 1e-8)) ** 2)


def _log_scale_mixture(x, sigma1=1.0, sigma2=0.01, pi_mix=0.5):
    """Log density of scale mixture prior: pi * N(0,s1^2) + (1-pi) * N(0,s2^2)."""
    log_p1 = -0.5 * np.log(2 * np.pi) - np.log(sigma1) - 0.5 * (x / sigma1) ** 2
    log_p2 = -0.5 * np.log(2 * np.pi) - np.log(sigma2) - 0.5 * (x / sigma2) ** 2
    log_mix = np.logaddexp(np.log(pi_mix) + log_p1, np.log(1 - pi_mix) + log_p2)
    return np.sum(log_mix)


class _NumpyMSELoss:
    """MSE loss operating on numpy arrays (not Tensors)."""

    def forward(self, predicted, target):
        self._predicted = predicted
        self._target = target
        diff = predicted - target
        return float(np.mean(diff ** 2))

    def backward(self, predicted, target):
        n = predicted.shape[0] if predicted.ndim > 0 else 1
        return 2.0 * (predicted - target) / max(n, 1)


class _NumpyCrossEntropyLoss:
    """Cross-entropy loss operating on numpy arrays."""

    def forward(self, logits, targets):
        """targets: integer class labels."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
        if targets.ndim == 1:
            # Integer labels
            n = len(targets)
            correct_logits = shifted[np.arange(n), targets.astype(int)]
        else:
            correct_logits = np.sum(shifted * targets, axis=-1)
            n = len(targets)
        loss = -np.mean(correct_logits - log_sum_exp)
        return float(loss)

    def backward(self, logits, targets):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
        n = logits.shape[0]
        if targets.ndim == 1:
            grad = probs.copy()
            grad[np.arange(n), targets.astype(int)] -= 1
        else:
            grad = probs - targets
        return grad / n


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_bnn(layer_sizes, activation='relu', prior_sigma=1.0,
              prior_mix=False, seed=42):
    """Build a BayesianNetwork from a list of layer sizes.

    Args:
        layer_sizes: list of ints, e.g. [2, 50, 50, 1]
        activation: activation function name
        prior_sigma: prior std for weights
        prior_mix: use scale mixture prior
        seed: random seed

    Returns:
        BayesianNetwork
    """
    network = BayesianNetwork()
    rng = np.random.RandomState(seed)

    for i in range(len(layer_sizes) - 1):
        network.add(BayesianLinear(
            layer_sizes[i], layer_sizes[i + 1],
            prior_sigma=prior_sigma,
            prior_mix=prior_mix,
            seed=rng.randint(0, 2**31)
        ))
        # Add activation after all layers except the last
        if i < len(layer_sizes) - 2:
            network.add(BayesianActivation(activation))

    return network


def build_mc_dropout_model(layer_sizes, activation='relu', drop_rate=0.1,
                            seed=42):
    """Build a standard Sequential model with Dropout for MC Dropout inference.

    Returns:
        MCDropoutNetwork wrapping a Sequential model
    """
    model = Sequential()
    rng = np.random.RandomState(seed)

    for i in range(len(layer_sizes) - 1):
        model.add(Dense(layer_sizes[i], layer_sizes[i + 1],
                        rng=rng))
        if i < len(layer_sizes) - 2:
            model.add(Activation(activation))
            model.add(Dropout(rate=drop_rate, rng=np.random.RandomState(rng.randint(0, 2**31))))

    return MCDropoutNetwork(model, drop_rate=drop_rate)


def active_learning_query(network, X_pool, n_query=10, strategy='bald',
                          n_samples=50, seed=42):
    """Select most informative points from pool for labeling.

    Strategies:
        'entropy': maximum predictive entropy
        'bald': maximum mutual information (BALD)
        'variation_ratio': proportion of non-modal predictions
        'std': maximum predictive standard deviation

    Returns:
        indices: indices into X_pool of selected points
        scores: acquisition scores for all pool points
    """
    if isinstance(X_pool, Tensor):
        X_pool = np.array(X_pool.data)

    predictive = BNNPredictive(network)
    pred = predictive.predict(X_pool, n_samples=n_samples, seed=seed)
    samples = pred['samples']

    if strategy == 'std':
        scores = pred['std'].mean(axis=-1) if pred['std'].ndim > 1 else pred['std'].ravel()
    elif strategy == 'entropy':
        # For regression: use predictive variance as proxy
        scores = pred['std'].mean(axis=-1) if pred['std'].ndim > 1 else pred['std'].ravel()
    elif strategy == 'bald':
        # BALD for regression: variance of means (epistemic)
        scores = pred['epistemic'].mean(axis=-1) if pred['epistemic'].ndim > 1 else pred['epistemic'].ravel()
    elif strategy == 'variation_ratio':
        # For classification: 1 - max prediction frequency
        if samples.ndim == 3 and samples.shape[-1] > 1:
            classes = np.argmax(samples, axis=-1)  # (n_samples, n_points)
            n_points = classes.shape[1]
            scores = np.zeros(n_points)
            for j in range(n_points):
                vals, counts = np.unique(classes[:, j], return_counts=True)
                scores[j] = 1 - counts.max() / n_samples
        else:
            scores = pred['std'].ravel()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    scores = scores.ravel()
    indices = np.argsort(-scores)[:n_query]
    return indices, scores
