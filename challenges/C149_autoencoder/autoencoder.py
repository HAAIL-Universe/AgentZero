"""
C149: Autoencoder / Variational Autoencoder (VAE)
Composing C140 (Neural Networks)

Implements:
- Autoencoder: encoder compresses input to latent space, decoder reconstructs
- Denoising Autoencoder: learns to reconstruct from corrupted input
- Sparse Autoencoder: KL-divergence sparsity penalty on activations
- Variational Autoencoder (VAE): probabilistic latent space with reparameterization trick
- Conditional VAE (CVAE): condition on labels for controlled generation
- Beta-VAE: tunable KL weight for disentangled representations

Key concepts:
- Encoder/Decoder architecture
- Latent space representation
- Reconstruction loss (MSE / BCE)
- KL divergence (VAE regularization)
- Reparameterization trick (backprop through stochastic sampling)
- Anomaly detection via reconstruction error
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Activation, Dropout, BatchNorm, Sequential,
    MSELoss, BinaryCrossEntropyLoss, SGD, Adam, RMSProp,
    xavier_init, he_init, sigmoid, relu, relu_deriv, sigmoid_deriv,
    softmax_batch, train_test_split, normalize
)


# ============================================================
# Utility functions
# ============================================================

def _safe_log(x, eps=1e-8):
    """Numerically stable log."""
    return math.log(max(x, eps))


def _clip(x, lo=1e-8, hi=1.0 - 1e-8):
    return max(lo, min(hi, x))


def _tensor_mse(a, b):
    """Mean squared error between two tensors (flattened)."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    flat_a = a.flatten().data
    flat_b = b.flatten().data
    n = len(flat_a)
    return sum((flat_a[i] - flat_b[i]) ** 2 for i in range(n)) / n


def _tensor_bce(pred, target):
    """Binary cross-entropy between pred and target tensors."""
    flat_p = pred.flatten().data
    flat_t = target.flatten().data
    n = len(flat_p)
    total = 0.0
    for i in range(n):
        p = _clip(flat_p[i])
        t = flat_t[i]
        total += -(t * _safe_log(p) + (1 - t) * _safe_log(1 - p))
    return total / n


def _tensor_bce_grad(pred, target):
    """Gradient of BCE w.r.t. pred."""
    flat_p = pred.flatten().data
    flat_t = target.flatten().data
    n = len(flat_p)
    grad = []
    for i in range(n):
        p = _clip(flat_p[i])
        t = flat_t[i]
        g = (-t / p + (1 - t) / (1 - p)) / n
        grad.append(g)
    if len(pred.shape) == 2:
        rows, cols = pred.shape
        data = [grad[r * cols:(r + 1) * cols] for r in range(rows)]
        return Tensor(data)
    return Tensor(grad)


def _tensor_mse_grad(pred, target):
    """Gradient of MSE w.r.t. pred."""
    flat_p = pred.flatten().data
    flat_t = target.flatten().data
    n = len(flat_p)
    grad = [2.0 * (flat_p[i] - flat_t[i]) / n for i in range(n)]
    if len(pred.shape) == 2:
        rows, cols = pred.shape
        data = [grad[r * cols:(r + 1) * cols] for r in range(rows)]
        return Tensor(data)
    return Tensor(grad)


def add_noise(tensor, noise_factor=0.3, rng=None):
    """Add Gaussian noise to tensor, clip to [0, 1]."""
    r = rng or random.Random()
    flat = tensor.flatten().data
    noisy = [max(0.0, min(1.0, v + noise_factor * r.gauss(0, 1))) for v in flat]
    if len(tensor.shape) == 2:
        rows, cols = tensor.shape
        data = [noisy[r_ * cols:(r_ + 1) * cols] for r_ in range(rows)]
        return Tensor(data)
    return Tensor(noisy)


# ============================================================
# Encoder / Decoder building blocks
# ============================================================

class Encoder:
    """Feedforward encoder: input_dim -> hidden layers -> latent_dim."""

    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu',
                 init='he', dropout=None, rng=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.network = Sequential()
        self.rng = rng

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.network.add(Dense(prev_dim, h_dim, init=init, rng=rng))
            self.network.add(Activation(activation))
            if dropout and dropout > 0:
                self.network.add(Dropout(dropout, rng=rng))
            prev_dim = h_dim

        # Final projection to latent space
        self.output_layer = Dense(prev_dim, latent_dim, init=init, rng=rng)

    def forward(self, x):
        h = self.network.forward(x)
        z = self.output_layer.forward(h)
        return z

    def backward(self, grad):
        g = self.output_layer.backward(grad)
        g = self.network.backward(g)
        return g

    def get_trainable_layers(self):
        layers = self.network.get_trainable_layers()
        layers.append(self.output_layer)
        return layers

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


class Decoder:
    """Feedforward decoder: latent_dim -> hidden layers -> output_dim."""

    def __init__(self, latent_dim, hidden_dims, output_dim, activation='relu',
                 output_activation='sigmoid', init='he', dropout=None, rng=None):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.network = Sequential()
        self.rng = rng

        prev_dim = latent_dim
        for h_dim in hidden_dims:
            self.network.add(Dense(prev_dim, h_dim, init=init, rng=rng))
            self.network.add(Activation(activation))
            if dropout and dropout > 0:
                self.network.add(Dropout(dropout, rng=rng))
            prev_dim = h_dim

        self.output_layer = Dense(prev_dim, output_dim, init=init, rng=rng)
        self.output_act = Activation(output_activation) if output_activation != 'linear' else None

    def forward(self, z):
        h = self.network.forward(z)
        out = self.output_layer.forward(h)
        if self.output_act:
            out = self.output_act.forward(out)
        return out

    def backward(self, grad):
        if self.output_act:
            grad = self.output_act.backward(grad)
        g = self.output_layer.backward(grad)
        g = self.network.backward(g)
        return g

    def get_trainable_layers(self):
        layers = self.network.get_trainable_layers()
        layers.append(self.output_layer)
        return layers

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


# ============================================================
# Autoencoder
# ============================================================

class Autoencoder:
    """Standard autoencoder: minimize reconstruction loss."""

    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu',
                 output_activation='sigmoid', loss_type='mse', init='he',
                 dropout=None, rng=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.encoder = Encoder(input_dim, hidden_dims, latent_dim,
                               activation=activation, init=init, dropout=dropout, rng=rng)
        # Decoder hidden dims are reversed
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(latent_dim, dec_hidden, input_dim,
                               activation=activation, output_activation=output_activation,
                               init=init, dropout=dropout, rng=rng)

    def forward(self, x):
        """Encode then decode."""
        self.last_input = x
        z = self.encoder.forward(x)
        self.last_latent = z
        reconstruction = self.decoder.forward(z)
        self.last_output = reconstruction
        return reconstruction

    def encode(self, x):
        """Get latent representation."""
        return self.encoder.forward(x)

    def decode(self, z):
        """Reconstruct from latent."""
        return self.decoder.forward(z)

    def compute_loss(self, x, reconstruction):
        """Compute reconstruction loss."""
        if self.loss_type == 'bce':
            return _tensor_bce(reconstruction, x)
        return _tensor_mse(reconstruction, x)

    def compute_loss_grad(self, x, reconstruction):
        """Gradient of reconstruction loss w.r.t. reconstruction."""
        if self.loss_type == 'bce':
            return _tensor_bce_grad(reconstruction, x)
        return _tensor_mse_grad(reconstruction, x)

    def backward(self, grad):
        """Backprop through decoder then encoder."""
        g = self.decoder.backward(grad)
        g = self.encoder.backward(g)
        return g

    def train_step(self, x_batch, optimizer):
        """Single training step. Returns loss."""
        self.train()
        reconstruction = self.forward(x_batch)
        loss = self.compute_loss(x_batch, reconstruction)
        grad = self.compute_loss_grad(x_batch, reconstruction)
        self.backward(grad)
        optimizer.step(self.get_trainable_layers())
        return loss

    def get_trainable_layers(self):
        return self.encoder.get_trainable_layers() + self.decoder.get_trainable_layers()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def reconstruction_error(self, x):
        """Compute per-sample reconstruction error (for anomaly detection)."""
        self.eval()
        if len(x.shape) == 1:
            x = Tensor([x.data])
        recon = self.forward(x)
        # Per-sample MSE
        errors = []
        if len(x.shape) == 2:
            for i in range(x.shape[0]):
                row_x = x.data[i]
                row_r = recon.data[i]
                err = sum((row_x[j] - row_r[j]) ** 2 for j in range(len(row_x))) / len(row_x)
                errors.append(err)
        else:
            errors.append(_tensor_mse(recon, x))
        return errors

    def count_params(self):
        """Count total parameters."""
        total = 0
        for layer in self.get_trainable_layers():
            for param, _, _ in layer.get_params():
                flat = param.flatten().data
                total += len(flat)
        return total


# ============================================================
# Denoising Autoencoder
# ============================================================

class DenoisingAutoencoder(Autoencoder):
    """Learns to reconstruct clean input from noisy input."""

    def __init__(self, input_dim, hidden_dims, latent_dim, noise_factor=0.3,
                 activation='relu', output_activation='sigmoid', loss_type='mse',
                 init='he', dropout=None, rng=None):
        super().__init__(input_dim, hidden_dims, latent_dim,
                         activation=activation, output_activation=output_activation,
                         loss_type=loss_type, init=init, dropout=dropout, rng=rng)
        self.noise_factor = noise_factor
        self.noise_rng = rng or random.Random(42)

    def train_step(self, x_batch, optimizer):
        """Train with noisy input, reconstruct clean."""
        self.train()
        noisy_x = add_noise(x_batch, self.noise_factor, self.noise_rng)
        reconstruction = self.forward(noisy_x)
        # Loss is against CLEAN input
        loss = self.compute_loss(x_batch, reconstruction)
        grad = self.compute_loss_grad(x_batch, reconstruction)
        self.backward(grad)
        optimizer.step(self.get_trainable_layers())
        return loss


# ============================================================
# Sparse Autoencoder
# ============================================================

class SparseAutoencoder(Autoencoder):
    """Autoencoder with KL-divergence sparsity penalty on latent activations."""

    def __init__(self, input_dim, hidden_dims, latent_dim, sparsity_target=0.05,
                 sparsity_weight=1.0, activation='relu', output_activation='sigmoid',
                 loss_type='mse', init='he', dropout=None, rng=None):
        super().__init__(input_dim, hidden_dims, latent_dim,
                         activation=activation, output_activation=output_activation,
                         loss_type=loss_type, init=init, dropout=dropout, rng=rng)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def _kl_divergence(self, rho, rho_hat):
        """KL(rho || rho_hat) for sparsity penalty."""
        rho_hat = _clip(rho_hat, 1e-6, 1 - 1e-6)
        rho = _clip(rho, 1e-6, 1 - 1e-6)
        return rho * _safe_log(rho / rho_hat) + (1 - rho) * _safe_log((1 - rho) / (1 - rho_hat))

    def _kl_grad(self, rho, rho_hat):
        """Gradient of KL divergence w.r.t. rho_hat."""
        rho_hat = _clip(rho_hat, 1e-6, 1 - 1e-6)
        rho = _clip(rho, 1e-6, 1 - 1e-6)
        return -rho / rho_hat + (1 - rho) / (1 - rho_hat)

    def train_step(self, x_batch, optimizer):
        """Training step with sparsity penalty."""
        self.train()
        reconstruction = self.forward(x_batch)
        recon_loss = self.compute_loss(x_batch, reconstruction)

        # Compute average activation of latent units
        z = self.last_latent
        if len(z.shape) == 2:
            # Average over batch
            avg_act = [0.0] * z.shape[1]
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    avg_act[j] += abs(z.data[i][j])
                avg_act = [a / z.shape[0] for a in avg_act]
        else:
            avg_act = [abs(v) for v in z.data]

        # KL divergence sparsity loss
        sparsity_loss = sum(
            self._kl_divergence(self.sparsity_target, rho_hat)
            for rho_hat in avg_act
        )
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        # Reconstruction gradient
        recon_grad = self.compute_loss_grad(x_batch, reconstruction)
        self.decoder.backward(recon_grad)

        # Sparsity gradient on latent
        if len(z.shape) == 2:
            sparsity_grad_data = []
            for i in range(z.shape[0]):
                row = []
                for j in range(z.shape[1]):
                    sign = 1.0 if z.data[i][j] >= 0 else -1.0
                    row.append(self.sparsity_weight * self._kl_grad(self.sparsity_target, avg_act[j]) * sign / z.shape[0])
                sparsity_grad_data.append(row)
            sparsity_grad = Tensor(sparsity_grad_data)
        else:
            sparsity_grad = Tensor([
                self.sparsity_weight * self._kl_grad(self.sparsity_target, avg_act[j]) * (1.0 if z.data[j] >= 0 else -1.0)
                for j in range(len(avg_act))
            ])

        # Backprop through encoder with combined gradient
        # The decoder backward already happened, now get the gradient from decoder's backward
        # that flows into latent space, and add sparsity gradient
        # Actually, we need to redo this properly:
        dec_input_grad = self.decoder.backward(recon_grad)
        combined_grad = Tensor(dec_input_grad.data)
        # Add sparsity gradient
        if len(combined_grad.shape) == 2:
            for i in range(combined_grad.shape[0]):
                for j in range(combined_grad.shape[1]):
                    combined_grad.data[i][j] += sparsity_grad.data[i][j]
        else:
            for j in range(len(combined_grad.data)):
                combined_grad.data[j] += sparsity_grad.data[j]

        self.encoder.backward(combined_grad)
        optimizer.step(self.get_trainable_layers())
        return total_loss


# ============================================================
# VAE Encoder (outputs mu and log_var)
# ============================================================

class VAEEncoder:
    """Encoder that outputs mean and log-variance for VAE."""

    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu',
                 init='he', dropout=None, rng=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rng = rng or random.Random(42)

        # Shared hidden layers
        self.network = Sequential()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.network.add(Dense(prev_dim, h_dim, init=init, rng=rng))
            self.network.add(Activation(activation))
            if dropout and dropout > 0:
                self.network.add(Dropout(dropout, rng=rng))
            prev_dim = h_dim

        # Two heads: mu and log_var
        self.mu_layer = Dense(prev_dim, latent_dim, init='xavier', rng=rng)
        self.logvar_layer = Dense(prev_dim, latent_dim, init='xavier', rng=rng)

    def forward(self, x):
        h = self.network.forward(x)
        self.last_h = h
        mu = self.mu_layer.forward(h)
        log_var = self.logvar_layer.forward(h)
        return mu, log_var

    def backward(self, grad_mu, grad_logvar):
        """Backprop through both heads."""
        g_mu = self.mu_layer.backward(grad_mu)
        g_logvar = self.logvar_layer.backward(grad_logvar)
        # Sum gradients from both heads
        if len(g_mu.shape) == 2:
            combined = [[g_mu.data[i][j] + g_logvar.data[i][j]
                         for j in range(g_mu.shape[1])]
                        for i in range(g_mu.shape[0])]
        else:
            combined = [g_mu.data[j] + g_logvar.data[j] for j in range(len(g_mu.data))]
        combined_tensor = Tensor(combined)
        return self.network.backward(combined_tensor)

    def get_trainable_layers(self):
        layers = self.network.get_trainable_layers()
        layers.append(self.mu_layer)
        layers.append(self.logvar_layer)
        return layers

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


# ============================================================
# Reparameterization
# ============================================================

def reparameterize(mu, log_var, rng=None):
    """Sample z = mu + std * epsilon (reparameterization trick).

    Returns z, epsilon (needed for backward pass).
    """
    r = rng or random.Random()

    if len(mu.shape) == 2:
        batch_size, latent_dim = mu.shape
        eps_data = [[r.gauss(0, 1) for _ in range(latent_dim)]
                    for _ in range(batch_size)]
        z_data = []
        for i in range(batch_size):
            row = []
            for j in range(latent_dim):
                std = math.exp(0.5 * log_var.data[i][j])
                row.append(mu.data[i][j] + std * eps_data[i][j])
            z_data.append(row)
        return Tensor(z_data), Tensor(eps_data)
    else:
        latent_dim = len(mu.data)
        eps_data = [r.gauss(0, 1) for _ in range(latent_dim)]
        z_data = []
        for j in range(latent_dim):
            std = math.exp(0.5 * log_var.data[j])
            z_data.append(mu.data[j] + std * eps_data[j])
        return Tensor(z_data), Tensor(eps_data)


# ============================================================
# KL Divergence for VAE
# ============================================================

def kl_divergence(mu, log_var):
    """KL(q(z|x) || p(z)) where p(z) = N(0,1).

    KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    Returns scalar.
    """
    if len(mu.shape) == 2:
        batch_size = mu.shape[0]
        total = 0.0
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                m = mu.data[i][j]
                lv = log_var.data[i][j]
                total += 1 + lv - m * m - math.exp(lv)
        return -0.5 * total / batch_size
    else:
        total = 0.0
        for j in range(len(mu.data)):
            m = mu.data[j]
            lv = log_var.data[j]
            total += 1 + lv - m * m - math.exp(lv)
        return -0.5 * total


def kl_divergence_grad(mu, log_var):
    """Gradients of KL divergence w.r.t. mu and log_var."""
    if len(mu.shape) == 2:
        batch_size = mu.shape[0]
        grad_mu_data = []
        grad_logvar_data = []
        for i in range(mu.shape[0]):
            mu_row = []
            lv_row = []
            for j in range(mu.shape[1]):
                m = mu.data[i][j]
                lv = log_var.data[i][j]
                # d(KL)/d(mu) = mu / batch_size
                mu_row.append(m / batch_size)
                # d(KL)/d(log_var) = 0.5 * (exp(log_var) - 1) / batch_size
                lv_row.append(0.5 * (math.exp(lv) - 1) / batch_size)
            grad_mu_data.append(mu_row)
            grad_logvar_data.append(lv_row)
        return Tensor(grad_mu_data), Tensor(grad_logvar_data)
    else:
        grad_mu = Tensor([mu.data[j] for j in range(len(mu.data))])
        grad_logvar = Tensor([0.5 * (math.exp(log_var.data[j]) - 1)
                              for j in range(len(log_var.data))])
        return grad_mu, grad_logvar


# ============================================================
# Variational Autoencoder (VAE)
# ============================================================

class VAE:
    """Variational Autoencoder with reparameterization trick.

    Loss = reconstruction_loss + beta * KL_divergence
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, beta=1.0,
                 activation='relu', output_activation='sigmoid', loss_type='mse',
                 init='he', dropout=None, rng=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_type = loss_type
        self.rng = rng or random.Random(42)

        self.encoder = VAEEncoder(input_dim, hidden_dims, latent_dim,
                                  activation=activation, init=init, dropout=dropout, rng=rng)
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(latent_dim, dec_hidden, input_dim,
                               activation=activation, output_activation=output_activation,
                               init=init, dropout=dropout, rng=rng)

    def forward(self, x):
        """Full forward pass: encode -> sample -> decode."""
        mu, log_var = self.encoder.forward(x)
        self.last_mu = mu
        self.last_log_var = log_var
        z, eps = reparameterize(mu, log_var, self.rng)
        self.last_z = z
        self.last_eps = eps
        reconstruction = self.decoder.forward(z)
        self.last_output = reconstruction
        self.last_input = x
        return reconstruction, mu, log_var

    def encode(self, x):
        """Encode to latent parameters (mu, log_var)."""
        return self.encoder.forward(x)

    def sample(self, mu, log_var):
        """Sample from latent distribution."""
        z, _ = reparameterize(mu, log_var, self.rng)
        return z

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder.forward(z)

    def generate(self, n_samples=1):
        """Generate samples from prior p(z) = N(0, 1)."""
        self.eval()
        z_data = [[self.rng.gauss(0, 1) for _ in range(self.latent_dim)]
                   for _ in range(n_samples)]
        z = Tensor(z_data) if n_samples > 1 else Tensor(z_data[0])
        return self.decode(z)

    def compute_loss(self, x, reconstruction, mu, log_var):
        """ELBO loss = reconstruction + beta * KL."""
        if self.loss_type == 'bce':
            recon_loss = _tensor_bce(reconstruction, x)
        else:
            recon_loss = _tensor_mse(reconstruction, x)
        kl_loss = kl_divergence(mu, log_var)
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def train_step(self, x_batch, optimizer):
        """Single training step. Returns (total_loss, recon_loss, kl_loss)."""
        self.train()
        reconstruction, mu, log_var = self.forward(x_batch)
        total_loss, recon_loss, kl_loss = self.compute_loss(
            x_batch, reconstruction, mu, log_var)

        # Reconstruction gradient
        if self.loss_type == 'bce':
            recon_grad = _tensor_bce_grad(reconstruction, x_batch)
        else:
            recon_grad = _tensor_mse_grad(reconstruction, x_batch)

        # Backprop through decoder
        dz = self.decoder.backward(recon_grad)

        # KL gradients
        grad_mu_kl, grad_logvar_kl = kl_divergence_grad(mu, log_var)

        # Reparameterization gradients: z = mu + exp(0.5*logvar) * eps
        # dL/dmu = dL/dz * dz/dmu + beta * dKL/dmu = dL/dz + beta * mu
        # dL/dlogvar = dL/dz * dz/dlogvar + beta * dKL/dlogvar
        #            = dL/dz * 0.5 * exp(0.5*logvar) * eps + beta * 0.5*(exp(logvar) - 1)
        eps = self.last_eps

        if len(mu.shape) == 2:
            grad_mu_data = []
            grad_logvar_data = []
            for i in range(mu.shape[0]):
                mu_row = []
                lv_row = []
                for j in range(mu.shape[1]):
                    # dz/dmu = 1
                    g_mu = dz.data[i][j] + self.beta * grad_mu_kl.data[i][j]
                    # dz/dlogvar = 0.5 * exp(0.5 * logvar) * eps
                    std = math.exp(0.5 * log_var.data[i][j])
                    g_lv = dz.data[i][j] * 0.5 * std * eps.data[i][j] + self.beta * grad_logvar_kl.data[i][j]
                    mu_row.append(g_mu)
                    lv_row.append(g_lv)
                grad_mu_data.append(mu_row)
                grad_logvar_data.append(lv_row)
            grad_mu = Tensor(grad_mu_data)
            grad_logvar = Tensor(grad_logvar_data)
        else:
            grad_mu = Tensor([
                dz.data[j] + self.beta * grad_mu_kl.data[j]
                for j in range(len(mu.data))
            ])
            grad_logvar = Tensor([
                dz.data[j] * 0.5 * math.exp(0.5 * log_var.data[j]) * eps.data[j] + self.beta * grad_logvar_kl.data[j]
                for j in range(len(log_var.data))
            ])

        # Backprop through encoder
        self.encoder.backward(grad_mu, grad_logvar)
        optimizer.step(self.get_trainable_layers())
        return total_loss, recon_loss, kl_loss

    def get_trainable_layers(self):
        return self.encoder.get_trainable_layers() + self.decoder.get_trainable_layers()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def reconstruction_error(self, x):
        """Per-sample reconstruction error for anomaly detection."""
        self.eval()
        if len(x.shape) == 1:
            x = Tensor([x.data])
        reconstruction, _, _ = self.forward(x)
        errors = []
        if len(x.shape) == 2:
            for i in range(x.shape[0]):
                row_x = x.data[i]
                row_r = reconstruction.data[i]
                err = sum((row_x[j] - row_r[j]) ** 2 for j in range(len(row_x))) / len(row_x)
                errors.append(err)
        return errors

    def count_params(self):
        total = 0
        for layer in self.get_trainable_layers():
            for param, _, _ in layer.get_params():
                flat = param.flatten().data
                total += len(flat)
        return total


# ============================================================
# Conditional VAE (CVAE)
# ============================================================

class ConditionalVAE:
    """VAE conditioned on labels: encoder gets (x, label), decoder gets (z, label).

    Labels are one-hot encoded and concatenated to input.
    """

    def __init__(self, input_dim, num_classes, hidden_dims, latent_dim, beta=1.0,
                 activation='relu', output_activation='sigmoid', loss_type='mse',
                 init='he', dropout=None, rng=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.beta = beta
        self.loss_type = loss_type
        self.rng = rng or random.Random(42)

        # Encoder: input + one_hot(label) -> latent
        enc_input_dim = input_dim + num_classes
        self.encoder = VAEEncoder(enc_input_dim, hidden_dims, latent_dim,
                                  activation=activation, init=init, dropout=dropout, rng=rng)

        # Decoder: latent + one_hot(label) -> output
        dec_input_dim = latent_dim + num_classes
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(dec_input_dim, dec_hidden, input_dim,
                               activation=activation, output_activation=output_activation,
                               init=init, dropout=dropout, rng=rng)

    def _one_hot(self, labels):
        """Convert labels (int or list) to one-hot tensor."""
        if isinstance(labels, int):
            oh = [0.0] * self.num_classes
            oh[labels] = 1.0
            return Tensor(oh)
        # list of ints
        data = []
        for l in labels:
            oh = [0.0] * self.num_classes
            oh[l] = 1.0
            data.append(oh)
        return Tensor(data)

    def _concat(self, a, b):
        """Concatenate two tensors along feature dimension."""
        if len(a.shape) == 2 and len(b.shape) == 2:
            return Tensor([a.data[i] + b.data[i] for i in range(a.shape[0])])
        elif len(a.shape) == 1 and len(b.shape) == 1:
            return Tensor(a.data + b.data)
        # mixed: promote 1D to 2D if needed
        if len(a.shape) == 1:
            a = Tensor([a.data])
        if len(b.shape) == 1:
            b = Tensor([b.data])
        return Tensor([a.data[i] + b.data[i] for i in range(a.shape[0])])

    def forward(self, x, labels):
        """Forward pass with conditioning."""
        label_oh = self._one_hot(labels)
        self.last_label_oh = label_oh

        # Encode
        enc_input = self._concat(x, label_oh)
        mu, log_var = self.encoder.forward(enc_input)
        self.last_mu = mu
        self.last_log_var = log_var

        # Sample
        z, eps = reparameterize(mu, log_var, self.rng)
        self.last_z = z
        self.last_eps = eps

        # Decode
        dec_input = self._concat(z, label_oh)
        self.last_dec_input = dec_input
        reconstruction = self.decoder.forward(dec_input)
        self.last_output = reconstruction
        self.last_input = x
        return reconstruction, mu, log_var

    def generate(self, labels, n_samples=1):
        """Generate conditioned on labels."""
        self.eval()
        if isinstance(labels, int):
            labels = [labels] * n_samples
        z_data = [[self.rng.gauss(0, 1) for _ in range(self.latent_dim)]
                   for _ in range(len(labels))]
        z = Tensor(z_data) if len(labels) > 1 else Tensor(z_data[0])
        label_oh = self._one_hot(labels)
        dec_input = self._concat(z, label_oh)
        return self.decoder.forward(dec_input)

    def train_step(self, x_batch, labels, optimizer):
        """Training step with conditioning."""
        self.train()
        reconstruction, mu, log_var = self.forward(x_batch, labels)

        if self.loss_type == 'bce':
            recon_loss = _tensor_bce(reconstruction, x_batch)
            recon_grad = _tensor_bce_grad(reconstruction, x_batch)
        else:
            recon_loss = _tensor_mse(reconstruction, x_batch)
            recon_grad = _tensor_mse_grad(reconstruction, x_batch)

        kl_loss = kl_divergence(mu, log_var)
        total_loss = recon_loss + self.beta * kl_loss

        # Backprop through decoder
        dz_full = self.decoder.backward(recon_grad)
        # Extract gradient for z only (first latent_dim dims), ignore label gradient
        if len(dz_full.shape) == 2:
            dz = Tensor([row[:self.latent_dim] for row in dz_full.data])
        else:
            dz = Tensor(dz_full.data[:self.latent_dim])

        # KL gradients + reparameterization
        grad_mu_kl, grad_logvar_kl = kl_divergence_grad(mu, log_var)
        eps = self.last_eps
        log_var = self.last_log_var

        if len(mu.shape) == 2:
            grad_mu_data = []
            grad_logvar_data = []
            for i in range(mu.shape[0]):
                mu_row = []
                lv_row = []
                for j in range(mu.shape[1]):
                    g_mu = dz.data[i][j] + self.beta * grad_mu_kl.data[i][j]
                    std = math.exp(0.5 * log_var.data[i][j])
                    g_lv = dz.data[i][j] * 0.5 * std * eps.data[i][j] + self.beta * grad_logvar_kl.data[i][j]
                    mu_row.append(g_mu)
                    lv_row.append(g_lv)
                grad_mu_data.append(mu_row)
                grad_logvar_data.append(lv_row)
            grad_mu = Tensor(grad_mu_data)
            grad_logvar = Tensor(grad_logvar_data)
        else:
            grad_mu = Tensor([
                dz.data[j] + self.beta * grad_mu_kl.data[j]
                for j in range(len(mu.data))
            ])
            grad_logvar = Tensor([
                dz.data[j] * 0.5 * math.exp(0.5 * log_var.data[j]) * eps.data[j] + self.beta * grad_logvar_kl.data[j]
                for j in range(len(log_var.data))
            ])

        self.encoder.backward(grad_mu, grad_logvar)
        optimizer.step(self.get_trainable_layers())
        return total_loss, recon_loss, kl_loss

    def get_trainable_layers(self):
        return self.encoder.get_trainable_layers() + self.decoder.get_trainable_layers()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def count_params(self):
        total = 0
        for layer in self.get_trainable_layers():
            for param, _, _ in layer.get_params():
                total += len(param.flatten().data)
        return total


# ============================================================
# Training utilities
# ============================================================

def fit_autoencoder(model, X, epochs=100, optimizer=None, batch_size=None,
                    shuffle=True, verbose=False, rng=None):
    """Train an Autoencoder or DenoisingAutoencoder.

    Returns history dict with 'loss' key.
    """
    if optimizer is None:
        optimizer = Adam(lr=0.001)
    r = rng or random.Random(42)
    n = X.shape[0] if len(X.shape) == 2 else 1
    history = {'loss': []}

    for epoch in range(epochs):
        if len(X.shape) == 2 and shuffle:
            indices = list(range(n))
            r.shuffle(indices)
            X_shuffled = Tensor([X.data[i] for i in indices])
        else:
            X_shuffled = X

        if batch_size and len(X.shape) == 2 and batch_size < n:
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = Tensor(X_shuffled.data[start:end])
                loss = model.train_step(batch, optimizer)
                epoch_loss += loss
                n_batches += 1
            avg_loss = epoch_loss / n_batches
        else:
            avg_loss = model.train_step(X_shuffled, optimizer)

        history['loss'].append(avg_loss)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return history


def fit_vae(model, X, epochs=100, optimizer=None, batch_size=None,
            shuffle=True, verbose=False, labels=None, rng=None):
    """Train a VAE or ConditionalVAE.

    Returns history dict with 'loss', 'recon_loss', 'kl_loss' keys.
    """
    if optimizer is None:
        optimizer = Adam(lr=0.001)
    r = rng or random.Random(42)
    n = X.shape[0] if len(X.shape) == 2 else 1
    is_cvae = isinstance(model, ConditionalVAE)
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(epochs):
        if len(X.shape) == 2 and shuffle:
            indices = list(range(n))
            r.shuffle(indices)
            X_shuffled = Tensor([X.data[i] for i in indices])
            if is_cvae and labels is not None:
                labels_shuffled = [labels[i] for i in indices]
            else:
                labels_shuffled = labels
        else:
            X_shuffled = X
            labels_shuffled = labels

        if batch_size and len(X.shape) == 2 and batch_size < n:
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = Tensor(X_shuffled.data[start:end])
                if is_cvae:
                    batch_labels = labels_shuffled[start:end]
                    total, recon, kl = model.train_step(batch, batch_labels, optimizer)
                else:
                    total, recon, kl = model.train_step(batch, optimizer)
                epoch_loss += total
                epoch_recon += recon
                epoch_kl += kl
                n_batches += 1
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
        else:
            if is_cvae:
                avg_loss, avg_recon, avg_kl = model.train_step(X_shuffled, labels_shuffled, optimizer)
            else:
                avg_loss, avg_recon, avg_kl = model.train_step(X_shuffled, optimizer)

        history['loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, "
                  f"Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}")

    return history


# ============================================================
# Data generators
# ============================================================

def make_sine_data(n=200, dim=20, seed=42):
    """Generate sine wave patterns for autoencoder testing."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * math.pi)
        amplitude = rng.uniform(0.3, 1.0)
        row = []
        for j in range(dim):
            t = j / dim * 2 * math.pi
            val = 0.5 + 0.5 * amplitude * math.sin(freq * t + phase)
            row.append(max(0.0, min(1.0, val)))
        data.append(row)
    return Tensor(data)


def make_gaussian_mixture_data(n=200, dim=10, n_clusters=3, seed=42):
    """Generate Gaussian mixture data for clustering/reconstruction."""
    rng = random.Random(seed)
    data = []
    labels = []
    # Generate cluster centers
    centers = []
    for _ in range(n_clusters):
        center = [rng.uniform(0.2, 0.8) for _ in range(dim)]
        centers.append(center)

    for i in range(n):
        cluster = i % n_clusters
        labels.append(cluster)
        row = []
        for j in range(dim):
            val = centers[cluster][j] + rng.gauss(0, 0.05)
            row.append(max(0.0, min(1.0, val)))
        data.append(row)

    return Tensor(data), labels


def make_binary_pattern_data(n=200, dim=16, seed=42):
    """Generate binary-ish patterns (near 0 or 1) for BCE loss testing."""
    rng = random.Random(seed)
    patterns = []
    for _ in range(min(8, n)):
        p = [rng.choice([0.1, 0.9]) for _ in range(dim)]
        patterns.append(p)

    data = []
    for i in range(n):
        base = patterns[i % len(patterns)]
        row = [max(0.01, min(0.99, base[j] + rng.gauss(0, 0.05))) for j in range(dim)]
        data.append(row)
    return Tensor(data)


# ============================================================
# Anomaly detection
# ============================================================

class AnomalyDetector:
    """Anomaly detection using autoencoder reconstruction error."""

    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold

    def fit_threshold(self, X_normal, percentile=95):
        """Set threshold from normal data reconstruction errors."""
        errors = self.model.reconstruction_error(X_normal)
        errors_sorted = sorted(errors)
        idx = min(int(len(errors_sorted) * percentile / 100), len(errors_sorted) - 1)
        self.threshold = errors_sorted[idx]
        return self.threshold

    def predict(self, X):
        """Returns list of booleans: True = anomaly."""
        errors = self.model.reconstruction_error(X)
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold first.")
        return [e > self.threshold for e in errors]

    def score(self, X):
        """Returns reconstruction errors (anomaly scores)."""
        return self.model.reconstruction_error(X)


# ============================================================
# Latent space utilities
# ============================================================

def interpolate_latent(model, x1, x2, n_steps=10):
    """Linear interpolation in latent space between two inputs.

    Works for both Autoencoder and VAE.
    Returns list of decoded tensors.
    """
    model.eval()
    if isinstance(model, VAE):
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        z1 = mu1
        z2 = mu2
    else:
        z1 = model.encode(x1)
        z2 = model.encode(x2)

    results = []
    for step in range(n_steps + 1):
        alpha = step / n_steps
        if len(z1.shape) == 1:
            z_interp = Tensor([
                (1 - alpha) * z1.data[j] + alpha * z2.data[j]
                for j in range(len(z1.data))
            ])
        else:
            z_interp = Tensor([
                [(1 - alpha) * z1.data[i][j] + alpha * z2.data[i][j]
                 for j in range(z1.shape[1])]
                for i in range(z1.shape[0])
            ])
        decoded = model.decode(z_interp)
        results.append(decoded)
    return results


def latent_statistics(model, X):
    """Compute statistics of the latent space.

    Returns dict with mean, std, min, max per dimension.
    """
    model.eval()
    if isinstance(model, VAE):
        mu, log_var = model.encode(X)
        Z = mu  # Use mean as representation
    else:
        Z = model.encode(X)

    if len(Z.shape) == 1:
        Z = Tensor([Z.data])

    latent_dim = Z.shape[1]
    n = Z.shape[0]
    stats = {'mean': [], 'std': [], 'min': [], 'max': []}

    for j in range(latent_dim):
        vals = [Z.data[i][j] for i in range(n)]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        stats['mean'].append(mean)
        stats['std'].append(std)
        stats['min'].append(min(vals))
        stats['max'].append(max(vals))

    return stats
