"""C177: Variational Autoencoder -- composing C140 Neural Network.

Components:
1. Encoder -- maps input to latent distribution (mu, log_var)
2. Decoder -- maps latent samples to reconstructions
3. Reparameterization -- differentiable sampling z = mu + std * eps
4. VAELoss -- reconstruction + KL divergence
5. VAE -- full variational autoencoder
6. ConditionalVAE -- class-conditioned generation
7. BetaVAE -- disentangled representations via beta weighting
8. VQVAE -- vector quantized VAE (discrete latents)
9. LatentSpace -- analysis, interpolation, traversal
10. VAETrainer -- training loop with warmup, annealing
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Activation, BatchNorm, Dropout, Sequential,
    MSELoss, BinaryCrossEntropyLoss, SGD, Adam,
    relu, sigmoid, tanh_act, softmax,
    relu_deriv, sigmoid_deriv, tanh_deriv,
    xavier_init, he_init,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RNG:
    """Simple RNG for reproducibility. Compatible with C140's rng interface."""
    def __init__(self, seed=42):
        self._state = seed

    def _next(self):
        self._state = (self._state * 1103515245 + 12345) & 0x7FFFFFFF
        return self._state

    def random(self):
        """Return float in [0, 1). Compatible with C140's rng.random()."""
        return self._next() / 0x7FFFFFFF

    def uniform(self, low=0.0, high=1.0):
        return low + (high - low) * self.random()

    def normal(self, mean=0.0, std=1.0):
        # Box-Muller
        u1 = max(1e-10, self.random())
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    def normal_tensor(self, shape, mean=0.0, std=1.0):
        if len(shape) == 1:
            return Tensor([self.normal(mean, std) for _ in range(shape[0])])
        rows, cols = shape
        return Tensor([[self.normal(mean, std) for _ in range(cols)] for _ in range(rows)])


def _tensor_to_flat(t):
    """Flatten a Tensor to a list of floats."""
    if len(t.shape) == 1:
        return list(t.data)
    return [v for row in t.data for v in row]


def _flat_to_tensor(data, shape):
    """Reshape flat list to Tensor with given shape."""
    if len(shape) == 1:
        return Tensor(list(data[:shape[0]]))
    rows, cols = shape
    return Tensor([list(data[i * cols:(i + 1) * cols]) for i in range(rows)])


# ---------------------------------------------------------------------------
# 1. Encoder
# ---------------------------------------------------------------------------

class Encoder:
    """Maps input x to latent distribution parameters (mu, log_var).

    Architecture: input -> hidden layers -> (mu_layer, logvar_layer)
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, activation='relu',
                 use_batchnorm=False, dropout=0.0, rng=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.rng = rng

        # Build shared trunk
        self.trunk = Sequential()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.trunk.add(Dense(prev_dim, h_dim, init='he', rng=rng))
            if use_batchnorm:
                self.trunk.add(BatchNorm(h_dim))
            self.trunk.add(Activation(activation))
            if dropout > 0:
                self.trunk.add(Dropout(dropout, rng=rng))
            prev_dim = h_dim

        # Separate heads for mu and log_var
        last_hidden = hidden_dims[-1] if hidden_dims else input_dim
        self.mu_layer = Dense(last_hidden, latent_dim, init='xavier', rng=rng)
        self.logvar_layer = Dense(last_hidden, latent_dim, init='xavier', rng=rng)

        # Cache for backward
        self._h = None
        self._mu = None
        self._logvar = None

    def forward(self, x):
        """Forward pass. Returns (mu, log_var) as Tensors."""
        h = self.trunk.forward(x)
        self._h = h
        mu = self.mu_layer.forward(h)
        logvar = self.logvar_layer.forward(h)
        self._mu = mu
        self._logvar = logvar
        return mu, logvar

    def backward(self, grad_mu, grad_logvar):
        """Backward through both heads and trunk."""
        grad_h_mu = self.mu_layer.backward(grad_mu)
        grad_h_lv = self.logvar_layer.backward(grad_logvar)
        # Sum gradients from both heads
        grad_h = _add_tensors(grad_h_mu, grad_h_lv)
        return self.trunk.backward(grad_h)

    def get_trainable_layers(self):
        layers = self.trunk.get_trainable_layers()
        if self.mu_layer.get_params():
            layers.append(self.mu_layer)
        if self.logvar_layer.get_params():
            layers.append(self.logvar_layer)
        return layers

    def train(self):
        self.trunk.train()
        self.mu_layer.set_training(True)
        self.logvar_layer.set_training(True)

    def eval(self):
        self.trunk.eval()
        self.mu_layer.set_training(False)
        self.logvar_layer.set_training(False)


# ---------------------------------------------------------------------------
# 2. Decoder
# ---------------------------------------------------------------------------

class Decoder:
    """Maps latent z to reconstruction of input space."""

    def __init__(self, latent_dim, hidden_dims, output_dim, activation='relu',
                 output_activation='sigmoid', use_batchnorm=False, dropout=0.0, rng=None):
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.network = Sequential()
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            self.network.add(Dense(prev_dim, h_dim, init='he', rng=rng))
            if use_batchnorm:
                self.network.add(BatchNorm(h_dim))
            self.network.add(Activation(activation))
            if dropout > 0:
                self.network.add(Dropout(dropout, rng=rng))
            prev_dim = h_dim

        last_hidden = hidden_dims[-1] if hidden_dims else latent_dim
        self.network.add(Dense(last_hidden, output_dim, init='xavier', rng=rng))
        if output_activation and output_activation != 'none':
            self.network.add(Activation(output_activation))

    def forward(self, z):
        return self.network.forward(z)

    def backward(self, grad_output):
        return self.network.backward(grad_output)

    def get_trainable_layers(self):
        return self.network.get_trainable_layers()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


# ---------------------------------------------------------------------------
# Tensor arithmetic helpers
# ---------------------------------------------------------------------------

def _add_tensors(a, b):
    """Element-wise add two tensors."""
    if len(a.shape) == 1:
        return Tensor([a.data[i] + b.data[i] for i in range(len(a.data))])
    return Tensor([[a.data[r][c] + b.data[r][c]
                    for c in range(len(a.data[0]))]
                   for r in range(len(a.data))])


def _sub_tensors(a, b):
    """Element-wise subtract."""
    if len(a.shape) == 1:
        return Tensor([a.data[i] - b.data[i] for i in range(len(a.data))])
    return Tensor([[a.data[r][c] - b.data[r][c]
                    for c in range(len(a.data[0]))]
                   for r in range(len(a.data))])


def _mul_tensors(a, b):
    """Element-wise multiply."""
    if len(a.shape) == 1:
        return Tensor([a.data[i] * b.data[i] for i in range(len(a.data))])
    return Tensor([[a.data[r][c] * b.data[r][c]
                    for c in range(len(a.data[0]))]
                   for r in range(len(a.data))])


def _scale_tensor(t, s):
    """Multiply tensor by scalar."""
    if len(t.shape) == 1:
        return Tensor([v * s for v in t.data])
    return Tensor([[v * s for v in row] for row in t.data])


def _apply_tensor(t, fn):
    """Apply function element-wise."""
    if len(t.shape) == 1:
        return Tensor([fn(v) for v in t.data])
    return Tensor([[fn(v) for v in row] for row in t.data])


def _tensor_sum(t):
    """Sum all elements."""
    if len(t.shape) == 1:
        return sum(t.data)
    return sum(v for row in t.data for v in row)


def _tensor_mean(t):
    """Mean of all elements."""
    s = _tensor_sum(t)
    n = t.shape[0] if len(t.shape) == 1 else t.shape[0] * t.shape[1]
    return s / n


def _ones_like(t):
    """Tensor of ones with same shape."""
    if len(t.shape) == 1:
        return Tensor([1.0] * t.shape[0])
    return Tensor([[1.0] * t.shape[1] for _ in range(t.shape[0])])


def _zeros_like(t):
    """Tensor of zeros with same shape."""
    if len(t.shape) == 1:
        return Tensor([0.0] * t.shape[0])
    return Tensor([[0.0] * t.shape[1] for _ in range(t.shape[0])])


def _get_row(t, i):
    """Get row i of a 2D tensor as 1D tensor."""
    if len(t.shape) == 1:
        return t
    return Tensor(list(t.data[i]))


def _set_row(t, i, row):
    """Set row i of 2D tensor from 1D tensor."""
    if len(t.shape) == 1:
        return
    for j in range(len(row.data)):
        t.data[i][j] = row.data[j]


def _stack_rows(rows):
    """Stack list of 1D tensors into 2D tensor."""
    return Tensor([list(r.data) for r in rows])


def _concat_tensors(a, b):
    """Concatenate two 1D tensors."""
    return Tensor(list(a.data) + list(b.data))


def _batch_size(t):
    """Get batch size (first dim)."""
    return t.shape[0]


# ---------------------------------------------------------------------------
# 3. Reparameterization
# ---------------------------------------------------------------------------

class Reparameterize:
    """Reparameterization trick: z = mu + std * eps.

    Allows gradients to flow through sampling.
    """

    def __init__(self, rng=None):
        self.rng = rng or _RNG(42)
        self._mu = None
        self._logvar = None
        self._eps = None
        self._std = None

    def forward(self, mu, logvar):
        """Sample z from N(mu, exp(logvar)) using reparameterization."""
        self._mu = mu
        self._logvar = logvar

        # std = exp(0.5 * logvar)
        self._std = _apply_tensor(logvar, lambda v: math.exp(0.5 * v))

        # eps ~ N(0, 1)
        self._eps = self.rng.normal_tensor(mu.shape)

        # z = mu + std * eps
        z = _add_tensors(mu, _mul_tensors(self._std, self._eps))
        return z

    def backward(self, grad_z):
        """Compute gradients w.r.t. mu and logvar.

        dL/d_mu = dL/dz
        dL/d_logvar = dL/dz * eps * 0.5 * exp(0.5 * logvar)
        """
        grad_mu = grad_z  # dz/dmu = 1
        # dz/d_logvar = eps * 0.5 * std
        grad_logvar = _mul_tensors(
            _mul_tensors(grad_z, self._eps),
            _scale_tensor(self._std, 0.5)
        )
        return grad_mu, grad_logvar


# ---------------------------------------------------------------------------
# 4. VAELoss
# ---------------------------------------------------------------------------

class VAELoss:
    """Combined reconstruction + KL divergence loss.

    L = recon_loss + beta * KL(q(z|x) || p(z))

    KL for Gaussian: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(self, recon_type='mse', beta=1.0):
        self.beta = beta
        self.recon_type = recon_type
        if recon_type == 'mse':
            self._recon_fn = MSELoss()
        elif recon_type == 'bce':
            self._recon_fn = BinaryCrossEntropyLoss()
        else:
            raise ValueError(f"Unknown recon_type: {recon_type}")

        self._mu = None
        self._logvar = None
        self._recon_loss = None
        self._kl_loss = None

    def forward(self, x_recon, x_orig, mu, logvar):
        """Compute total VAE loss.

        Returns (total_loss, recon_loss, kl_loss).
        """
        self._mu = mu
        self._logvar = logvar

        recon = self._recon_fn.forward(x_recon, x_orig)
        self._recon_loss = recon

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = self._compute_kl(mu, logvar)
        self._kl_loss = kl

        total = recon + self.beta * kl
        return total, recon, kl

    def _compute_kl(self, mu, logvar):
        """KL divergence for Gaussian posterior."""
        # Per-element: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        kl_sum = 0.0
        if len(mu.shape) == 1:
            for i in range(len(mu.data)):
                m = mu.data[i]
                lv = logvar.data[i]
                kl_sum += -0.5 * (1.0 + lv - m * m - math.exp(lv))
        else:
            for r in range(len(mu.data)):
                for c in range(len(mu.data[0])):
                    m = mu.data[r][c]
                    lv = logvar.data[r][c]
                    kl_sum += -0.5 * (1.0 + lv - m * m - math.exp(lv))
            # Average over batch
            kl_sum /= len(mu.data)
        return kl_sum

    def backward_recon(self, x_recon, x_orig):
        """Gradient of reconstruction loss w.r.t. x_recon."""
        return self._recon_fn.backward(x_recon, x_orig)

    def backward_kl(self, mu, logvar):
        """Gradients of KL loss w.r.t. mu and logvar.

        dKL/dmu = mu (per element)
        dKL/dlogvar = 0.5 * (exp(logvar) - 1) (per element)
        """
        batch_div = 1.0
        if len(mu.shape) == 2:
            batch_div = mu.shape[0]

        grad_mu = _scale_tensor(mu, self.beta / batch_div)
        grad_logvar = _apply_tensor(
            logvar,
            lambda lv: self.beta * 0.5 * (math.exp(lv) - 1.0) / batch_div
        )
        return grad_mu, grad_logvar


# ---------------------------------------------------------------------------
# 5. VAE -- Standard Variational Autoencoder
# ---------------------------------------------------------------------------

class VAE:
    """Standard Variational Autoencoder.

    Composes Encoder, Reparameterize, Decoder, VAELoss.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim,
                 activation='relu', output_activation='sigmoid',
                 recon_type='mse', beta=1.0,
                 use_batchnorm=False, dropout=0.0, seed=42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        rng = _RNG(seed)

        self.encoder = Encoder(
            input_dim, hidden_dims, latent_dim,
            activation=activation, use_batchnorm=use_batchnorm,
            dropout=dropout, rng=rng
        )
        self.reparameterize = Reparameterize(rng=rng)
        # Decoder hidden dims are reversed
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(
            latent_dim, dec_hidden, input_dim,
            activation=activation, output_activation=output_activation,
            use_batchnorm=use_batchnorm, dropout=dropout, rng=rng
        )
        self.loss_fn = VAELoss(recon_type=recon_type, beta=beta)

        # Cache
        self._x = None
        self._mu = None
        self._logvar = None
        self._z = None
        self._x_recon = None

    def forward(self, x):
        """Forward pass: encode -> reparameterize -> decode.

        Returns (x_recon, mu, logvar, z).
        """
        self._x = x
        mu, logvar = self.encoder.forward(x)
        self._mu = mu
        self._logvar = logvar
        z = self.reparameterize.forward(mu, logvar)
        self._z = z
        x_recon = self.decoder.forward(z)
        self._x_recon = x_recon
        return x_recon, mu, logvar, z

    def compute_loss(self, x_recon, x_orig, mu, logvar):
        """Compute VAE loss. Returns (total, recon, kl)."""
        return self.loss_fn.forward(x_recon, x_orig, mu, logvar)

    def backward(self, x_recon, x_orig, mu, logvar):
        """Full backward pass through the VAE."""
        # Reconstruction gradient
        grad_recon = self.loss_fn.backward_recon(x_recon, x_orig)

        # Backward through decoder
        grad_z = self.decoder.backward(grad_recon)

        # KL gradients
        grad_kl_mu, grad_kl_logvar = self.loss_fn.backward_kl(mu, logvar)

        # Backward through reparameterization
        grad_mu_reparam, grad_logvar_reparam = self.reparameterize.backward(grad_z)

        # Combine gradients
        grad_mu_total = _add_tensors(grad_mu_reparam, grad_kl_mu)
        grad_logvar_total = _add_tensors(grad_logvar_reparam, grad_kl_logvar)

        # Backward through encoder
        self.encoder.backward(grad_mu_total, grad_logvar_total)

    def get_trainable_layers(self):
        return self.encoder.get_trainable_layers() + self.decoder.get_trainable_layers()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, x):
        """Encode input to latent distribution. Returns (mu, logvar)."""
        return self.encoder.forward(x)

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder.forward(z)

    def sample(self, n_samples=1, rng=None):
        """Generate samples from prior p(z) = N(0, I)."""
        rng = rng or _RNG(123)
        z = rng.normal_tensor((n_samples, self.latent_dim))
        self.eval()
        return self.decoder.forward(z)

    def reconstruct(self, x):
        """Reconstruct input through full pipeline."""
        self.eval()
        x_recon, mu, logvar, z = self.forward(x)
        return x_recon


# ---------------------------------------------------------------------------
# 6. ConditionalVAE
# ---------------------------------------------------------------------------

class ConditionalVAE:
    """Conditional VAE -- conditions on class labels.

    Encoder input: concat(x, one_hot(y))
    Decoder input: concat(z, one_hot(y))
    """

    def __init__(self, input_dim, num_classes, hidden_dims, latent_dim,
                 activation='relu', output_activation='sigmoid',
                 recon_type='mse', beta=1.0, seed=42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        rng = _RNG(seed)

        # Encoder takes input + one-hot label
        enc_input_dim = input_dim + num_classes
        self.encoder = Encoder(
            enc_input_dim, hidden_dims, latent_dim,
            activation=activation, rng=rng
        )
        self.reparameterize = Reparameterize(rng=rng)

        # Decoder takes latent + one-hot label
        dec_input_dim = latent_dim + num_classes
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(
            dec_input_dim, dec_hidden, input_dim,
            activation=activation, output_activation=output_activation, rng=rng
        )
        self.loss_fn = VAELoss(recon_type=recon_type, beta=beta)

        self._x = None
        self._y_onehot = None
        self._mu = None
        self._logvar = None
        self._z = None

    def _one_hot(self, y):
        """Convert label(s) to one-hot tensor."""
        if isinstance(y, int):
            oh = [0.0] * self.num_classes
            oh[y] = 1.0
            return Tensor(oh)
        # Batch of labels
        result = []
        labels = y if isinstance(y, list) else y.data
        for label in labels:
            oh = [0.0] * self.num_classes
            oh[int(label)] = 1.0
            result.append(oh)
        return Tensor(result)

    def forward(self, x, y):
        """Forward pass with conditioning on label y."""
        y_oh = self._one_hot(y)
        self._y_onehot = y_oh

        # Concat x and y for encoder
        enc_input = self._concat_input(x, y_oh)
        mu, logvar = self.encoder.forward(enc_input)
        self._mu = mu
        self._logvar = logvar

        z = self.reparameterize.forward(mu, logvar)
        self._z = z

        # Concat z and y for decoder
        dec_input = self._concat_input(z, y_oh)
        x_recon = self.decoder.forward(dec_input)
        return x_recon, mu, logvar, z

    def _concat_input(self, a, b):
        """Concatenate two tensors along feature dimension."""
        if len(a.shape) == 1:
            return _concat_tensors(a, b)
        # 2D: concat columns
        rows = []
        for i in range(len(a.data)):
            b_row = b.data[i] if len(b.shape) == 2 else b.data
            rows.append(list(a.data[i]) + list(b_row))
        return Tensor(rows)

    def compute_loss(self, x_recon, x_orig, mu, logvar):
        return self.loss_fn.forward(x_recon, x_orig, mu, logvar)

    def backward(self, x_recon, x_orig, mu, logvar):
        grad_recon = self.loss_fn.backward_recon(x_recon, x_orig)
        grad_dec_input = self.decoder.backward(grad_recon)

        # Split gradient: [grad_z, grad_y] -- discard grad_y
        if len(grad_dec_input.shape) == 1:
            grad_z = Tensor(list(grad_dec_input.data[:self.latent_dim]))
        else:
            grad_z = Tensor([list(row[:self.latent_dim]) for row in grad_dec_input.data])

        grad_kl_mu, grad_kl_logvar = self.loss_fn.backward_kl(mu, logvar)
        grad_mu_r, grad_logvar_r = self.reparameterize.backward(grad_z)
        grad_mu_total = _add_tensors(grad_mu_r, grad_kl_mu)
        grad_logvar_total = _add_tensors(grad_logvar_r, grad_kl_logvar)
        self.encoder.backward(grad_mu_total, grad_logvar_total)

    def get_trainable_layers(self):
        return self.encoder.get_trainable_layers() + self.decoder.get_trainable_layers()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def generate(self, y, n_samples=1, rng=None):
        """Generate samples conditioned on class label y."""
        rng = rng or _RNG(456)
        self.eval()
        z = rng.normal_tensor((n_samples, self.latent_dim))
        y_oh = self._one_hot(y if isinstance(y, int) else [y] * n_samples)
        if isinstance(y, int):
            # Repeat for batch
            y_oh = Tensor([list(y_oh.data) if len(y_oh.shape) == 1 else y_oh.data[0]] * n_samples)
        dec_input = self._concat_input(z, y_oh)
        return self.decoder.forward(dec_input)


# ---------------------------------------------------------------------------
# 7. BetaVAE
# ---------------------------------------------------------------------------

class BetaVAE(VAE):
    """Beta-VAE for disentangled representation learning.

    Same as VAE but with beta > 1 to encourage disentanglement.
    Supports cyclical annealing of beta.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim,
                 beta=4.0, beta_schedule=None,
                 activation='relu', output_activation='sigmoid',
                 recon_type='mse', seed=42):
        super().__init__(
            input_dim, hidden_dims, latent_dim,
            activation=activation, output_activation=output_activation,
            recon_type=recon_type, beta=beta, seed=seed
        )
        self.beta_schedule = beta_schedule  # 'cyclical', 'warmup', or None
        self.base_beta = beta
        self._step = 0

    def update_beta(self, epoch, total_epochs):
        """Update beta according to schedule."""
        if self.beta_schedule == 'warmup':
            # Linear warmup from 0 to base_beta
            progress = min(1.0, epoch / max(1, total_epochs * 0.5))
            self.loss_fn.beta = self.base_beta * progress
        elif self.beta_schedule == 'cyclical':
            # Cyclical annealing: cycle every total_epochs/4 epochs
            cycle_len = max(1, total_epochs // 4)
            cycle_pos = epoch % cycle_len
            progress = min(1.0, cycle_pos / max(1, cycle_len * 0.5))
            self.loss_fn.beta = self.base_beta * progress
        self._step = epoch

    def get_beta(self):
        return self.loss_fn.beta


# ---------------------------------------------------------------------------
# 8. VQ-VAE (Vector Quantized VAE)
# ---------------------------------------------------------------------------

class VectorQuantizer:
    """Vector Quantization layer for VQ-VAE.

    Maps continuous encoder output to nearest embedding vector.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, rng=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize codebook
        rng = rng or _RNG(42)
        self.embeddings = Tensor([
            [rng.normal(0.0, 0.1) for _ in range(embedding_dim)]
            for _ in range(num_embeddings)
        ])

        self._z_e = None
        self._z_q = None
        self._indices = None
        self._usage_count = [0] * num_embeddings

    def forward(self, z_e):
        """Quantize encoder output.

        Args:
            z_e: Tensor of shape (batch, embedding_dim) or (embedding_dim,)

        Returns:
            (z_q, indices, vq_loss) where z_q has same shape as z_e
        """
        self._z_e = z_e
        is_1d = len(z_e.shape) == 1

        if is_1d:
            z_e_2d = Tensor([list(z_e.data)])
        else:
            z_e_2d = z_e

        batch_size = len(z_e_2d.data)
        indices = []
        z_q_rows = []

        for i in range(batch_size):
            vec = z_e_2d.data[i]
            # Find nearest embedding
            min_dist = float('inf')
            min_idx = 0
            for j in range(self.num_embeddings):
                emb = self.embeddings.data[j]
                dist = sum((vec[k] - emb[k]) ** 2 for k in range(self.embedding_dim))
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            indices.append(min_idx)
            z_q_rows.append(list(self.embeddings.data[min_idx]))
            self._usage_count[min_idx] += 1

        z_q = Tensor(z_q_rows) if not is_1d else Tensor(z_q_rows[0])
        self._z_q = z_q
        self._indices = indices

        # VQ loss = ||z_e - sg(z_q)||^2 + commitment * ||sg(z_e) - z_q||^2
        vq_loss = self._compute_vq_loss(z_e_2d, Tensor(z_q_rows))

        return z_q, indices, vq_loss

    def _compute_vq_loss(self, z_e, z_q):
        """Compute VQ loss (codebook + commitment)."""
        # Codebook loss: ||sg(z_e) - z_q||^2 (gradient flows to embeddings)
        # Commitment loss: ||z_e - sg(z_q)||^2 (gradient flows to encoder)
        codebook_loss = 0.0
        commitment_loss = 0.0
        n = 0
        for i in range(len(z_e.data)):
            for j in range(self.embedding_dim):
                diff = z_e.data[i][j] - z_q.data[i][j]
                codebook_loss += diff * diff
                commitment_loss += diff * diff
                n += 1
        if n > 0:
            codebook_loss /= n
            commitment_loss /= n
        return codebook_loss + self.commitment_cost * commitment_loss

    def backward(self, grad_output):
        """Straight-through estimator: pass gradient directly to encoder."""
        # Update embeddings toward encoder output (EMA or gradient)
        self._update_embeddings()
        return grad_output  # Straight-through

    def _update_embeddings(self, lr=0.01):
        """Move embeddings toward assigned encoder outputs."""
        if self._z_e is None or self._indices is None:
            return
        z_e = self._z_e
        if len(z_e.shape) == 1:
            z_e = Tensor([list(z_e.data)])

        for i, idx in enumerate(self._indices):
            for j in range(self.embedding_dim):
                diff = z_e.data[i][j] - self.embeddings.data[idx][j]
                self.embeddings.data[idx][j] += lr * diff

    def get_codebook_usage(self):
        """Return fraction of codebook being used."""
        used = sum(1 for c in self._usage_count if c > 0)
        return used / self.num_embeddings

    def reset_usage(self):
        self._usage_count = [0] * self.num_embeddings


class VQVAE:
    """Vector Quantized VAE.

    Uses discrete latent codes instead of continuous.
    No KL divergence -- uses VQ loss instead.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim,
                 num_embeddings=64, commitment_cost=0.25,
                 activation='relu', output_activation='sigmoid',
                 recon_type='mse', seed=42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        rng = _RNG(seed)

        # Encoder maps to continuous space
        self.encoder_net = Sequential()
        prev = input_dim
        for h in hidden_dims:
            self.encoder_net.add(Dense(prev, h, init='he', rng=rng))
            self.encoder_net.add(Activation(activation))
            prev = h
        last_h = hidden_dims[-1] if hidden_dims else input_dim
        self.encoder_net.add(Dense(last_h, latent_dim, init='xavier', rng=rng))

        # Vector quantizer
        self.vq = VectorQuantizer(num_embeddings, latent_dim,
                                  commitment_cost=commitment_cost, rng=rng)

        # Decoder
        dec_hidden = list(reversed(hidden_dims))
        self.decoder = Decoder(
            latent_dim, dec_hidden, input_dim,
            activation=activation, output_activation=output_activation, rng=rng
        )

        self.recon_type = recon_type
        if recon_type == 'mse':
            self._recon_loss_fn = MSELoss()
        else:
            self._recon_loss_fn = BinaryCrossEntropyLoss()

        self._z_e = None
        self._z_q = None
        self._indices = None

    def forward(self, x):
        """Forward pass.

        Returns (x_recon, z_e, z_q, indices, vq_loss).
        """
        z_e = self.encoder_net.forward(x)
        self._z_e = z_e
        z_q, indices, vq_loss = self.vq.forward(z_e)
        self._z_q = z_q
        self._indices = indices

        # Straight-through: use z_q for forward, but gradient goes to z_e
        x_recon = self.decoder.forward(z_q)
        return x_recon, z_e, z_q, indices, vq_loss

    def compute_loss(self, x_recon, x_orig, vq_loss):
        """Total loss = recon_loss + vq_loss."""
        recon = self._recon_loss_fn.forward(x_recon, x_orig)
        return recon + vq_loss, recon, vq_loss

    def backward(self, x_recon, x_orig):
        """Backward pass with straight-through estimator."""
        grad_recon = self._recon_loss_fn.backward(x_recon, x_orig)
        grad_z_q = self.decoder.backward(grad_recon)
        # Straight-through: gradient passes through VQ to encoder
        grad_z_e = self.vq.backward(grad_z_q)
        self.encoder_net.backward(grad_z_e)

    def get_trainable_layers(self):
        return self.encoder_net.get_trainable_layers() + self.decoder.get_trainable_layers()

    def train(self):
        for layer in self.encoder_net.layers:
            layer.set_training(True)
        self.decoder.train()

    def eval(self):
        for layer in self.encoder_net.layers:
            layer.set_training(False)
        self.decoder.eval()

    def get_codebook_usage(self):
        return self.vq.get_codebook_usage()


# ---------------------------------------------------------------------------
# 9. LatentSpace -- Analysis, interpolation, traversal
# ---------------------------------------------------------------------------

class LatentSpace:
    """Tools for analyzing and navigating the latent space."""

    def __init__(self, vae):
        self.vae = vae

    def encode_to_latent(self, x):
        """Encode input to latent mean (no sampling noise)."""
        self.vae.eval()
        mu, logvar = self.vae.encoder.forward(x)
        return mu

    def interpolate(self, x1, x2, n_steps=10):
        """Linear interpolation between two inputs in latent space."""
        self.vae.eval()
        mu1, _ = self.vae.encoder.forward(x1)
        mu2, _ = self.vae.encoder.forward(x2)

        results = []
        for i in range(n_steps):
            alpha = i / max(1, n_steps - 1)
            z = _add_tensors(
                _scale_tensor(mu1, 1.0 - alpha),
                _scale_tensor(mu2, alpha)
            )
            recon = self.vae.decoder.forward(z)
            results.append(recon)
        return results

    def spherical_interpolate(self, x1, x2, n_steps=10):
        """Spherical (slerp) interpolation in latent space."""
        self.vae.eval()
        mu1, _ = self.vae.encoder.forward(x1)
        mu2, _ = self.vae.encoder.forward(x2)

        # Compute angle
        dot = _tensor_sum(_mul_tensors(mu1, mu2))
        norm1 = math.sqrt(max(1e-10, _tensor_sum(_mul_tensors(mu1, mu1))))
        norm2 = math.sqrt(max(1e-10, _tensor_sum(_mul_tensors(mu2, mu2))))
        cos_omega = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        omega = math.acos(cos_omega)

        results = []
        for i in range(n_steps):
            t = i / max(1, n_steps - 1)
            if abs(omega) < 1e-6:
                # Nearly parallel, use linear
                z = _add_tensors(
                    _scale_tensor(mu1, 1.0 - t),
                    _scale_tensor(mu2, t)
                )
            else:
                s1 = math.sin((1 - t) * omega) / math.sin(omega)
                s2 = math.sin(t * omega) / math.sin(omega)
                z = _add_tensors(
                    _scale_tensor(mu1, s1),
                    _scale_tensor(mu2, s2)
                )
            recon = self.vae.decoder.forward(z)
            results.append(recon)
        return results

    def traverse_dimension(self, x, dim, values=None):
        """Traverse a single latent dimension while keeping others fixed."""
        self.vae.eval()
        mu, _ = self.vae.encoder.forward(x)

        if values is None:
            values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

        results = []
        for val in values:
            z = Tensor(list(mu.data)) if len(mu.shape) == 1 else mu  # copy
            if len(z.shape) == 1:
                z_copy = Tensor(list(z.data))
                z_copy.data[dim] = val
            else:
                z_copy = Tensor([list(row) for row in z.data])
                z_copy.data[0][dim] = val
            recon = self.vae.decoder.forward(z_copy)
            results.append(recon)
        return results

    def compute_latent_stats(self, dataset):
        """Compute statistics of the latent space over a dataset.

        Args:
            dataset: list of Tensor inputs

        Returns dict with mean, std, and active_dims.
        """
        self.vae.eval()
        all_mu = []
        all_logvar = []
        for x in dataset:
            mu, logvar = self.vae.encoder.forward(x)
            if len(mu.shape) == 1:
                all_mu.append(list(mu.data))
                all_logvar.append(list(logvar.data))
            else:
                for r in range(len(mu.data)):
                    all_mu.append(list(mu.data[r]))
                    all_logvar.append(list(logvar.data[r]))

        n = len(all_mu)
        if n == 0:
            return {'mean': [], 'std': [], 'active_dims': 0}

        latent_dim = len(all_mu[0])
        means = [sum(all_mu[i][d] for i in range(n)) / n for d in range(latent_dim)]
        stds = [math.sqrt(sum((all_mu[i][d] - means[d]) ** 2 for i in range(n)) / max(1, n))
                for d in range(latent_dim)]

        # Active dimensions: those with std > threshold
        active = sum(1 for s in stds if s > 0.1)

        # Mean KL per dimension
        mean_kl = []
        for d in range(latent_dim):
            kl_d = 0.0
            for i in range(n):
                m = all_mu[i][d]
                lv = all_logvar[i][d]
                kl_d += -0.5 * (1.0 + lv - m * m - math.exp(lv))
            mean_kl.append(kl_d / n)

        return {
            'mean': means,
            'std': stds,
            'active_dims': active,
            'mean_kl_per_dim': mean_kl,
            'total_mean_kl': sum(mean_kl),
        }

    def nearest_neighbors(self, x, dataset, k=5):
        """Find k nearest neighbors in latent space."""
        self.vae.eval()
        mu_x, _ = self.vae.encoder.forward(x)
        if len(mu_x.shape) == 2:
            mu_x = Tensor(list(mu_x.data[0]))

        distances = []
        for i, sample in enumerate(dataset):
            mu_s, _ = self.vae.encoder.forward(sample)
            if len(mu_s.shape) == 2:
                mu_s = Tensor(list(mu_s.data[0]))
            dist = sum((mu_x.data[j] - mu_s.data[j]) ** 2
                       for j in range(len(mu_x.data)))
            distances.append((dist, i))

        distances.sort()
        return [(idx, dist) for dist, idx in distances[:k]]


# ---------------------------------------------------------------------------
# 10. VAETrainer -- Training with warmup, annealing, logging
# ---------------------------------------------------------------------------

class VAETrainer:
    """Training loop for VAE models with warmup and logging."""

    def __init__(self, vae, optimizer, kl_warmup_epochs=0, kl_anneal_epochs=0):
        self.vae = vae
        self.optimizer = optimizer
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_anneal_epochs = kl_anneal_epochs
        self.history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'beta': [],
        }

    def _get_beta_factor(self, epoch):
        """Compute KL annealing factor."""
        if self.kl_warmup_epochs <= 0:
            return 1.0
        if epoch < self.kl_warmup_epochs:
            return epoch / self.kl_warmup_epochs
        if self.kl_anneal_epochs > 0:
            cycle_pos = (epoch - self.kl_warmup_epochs) % self.kl_anneal_epochs
            return min(1.0, cycle_pos / max(1, self.kl_anneal_epochs * 0.5))
        return 1.0

    def train_epoch(self, data, batch_size=None, epoch=0):
        """Train one epoch. data is list of Tensor samples or single Tensor batch.

        Returns (avg_total, avg_recon, avg_kl).
        """
        self.vae.train()

        # KL warmup
        beta_factor = self._get_beta_factor(epoch)
        original_beta = self.vae.loss_fn.beta
        effective_beta = original_beta * beta_factor
        self.vae.loss_fn.beta = effective_beta

        # Handle data
        if isinstance(data, list):
            batches = self._make_batches(data, batch_size)
        else:
            batches = [data]

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_batches = 0

        for batch in batches:
            x_recon, mu, logvar, z = self.vae.forward(batch)
            total, recon, kl = self.vae.compute_loss(x_recon, batch, mu, logvar)
            self.vae.backward(x_recon, batch, mu, logvar)
            self.optimizer.step(self.vae.get_trainable_layers())

            total_loss_sum += total
            recon_loss_sum += recon
            kl_loss_sum += kl
            n_batches += 1

        # Restore original beta
        self.vae.loss_fn.beta = original_beta

        avg_total = total_loss_sum / max(1, n_batches)
        avg_recon = recon_loss_sum / max(1, n_batches)
        avg_kl = kl_loss_sum / max(1, n_batches)

        self.history['total_loss'].append(avg_total)
        self.history['recon_loss'].append(avg_recon)
        self.history['kl_loss'].append(avg_kl)
        self.history['beta'].append(effective_beta)

        return avg_total, avg_recon, avg_kl

    def train(self, data, epochs=10, batch_size=None, verbose=False):
        """Full training loop."""
        for epoch in range(epochs):
            total, recon, kl = self.train_epoch(data, batch_size, epoch)
            if verbose:
                print(f"Epoch {epoch}: total={total:.4f} recon={recon:.4f} kl={kl:.4f}")
        return self.history

    def _make_batches(self, data_list, batch_size):
        """Convert list of samples to batches."""
        if batch_size is None or batch_size >= len(data_list):
            # Stack all into one batch
            return [_stack_rows(data_list)]

        batches = []
        for i in range(0, len(data_list), batch_size):
            chunk = data_list[i:i + batch_size]
            if len(chunk) > 0:
                batches.append(_stack_rows(chunk))
        return batches

    def evaluate(self, data):
        """Evaluate without training. Returns (total, recon, kl)."""
        self.vae.eval()
        if isinstance(data, list):
            data = _stack_rows(data)
        x_recon, mu, logvar, z = self.vae.forward(data)
        return self.vae.compute_loss(x_recon, data, mu, logvar)


# ---------------------------------------------------------------------------
# VQ-VAE Trainer
# ---------------------------------------------------------------------------

class VQVAETrainer:
    """Training loop for VQ-VAE."""

    def __init__(self, vqvae, optimizer):
        self.vqvae = vqvae
        self.optimizer = optimizer
        self.history = {
            'total_loss': [],
            'recon_loss': [],
            'vq_loss': [],
            'codebook_usage': [],
        }

    def train_epoch(self, data):
        """Train one epoch."""
        self.vqvae.train()

        if isinstance(data, list):
            data = _stack_rows(data)

        x_recon, z_e, z_q, indices, vq_loss = self.vqvae.forward(data)
        total, recon, vq = self.vqvae.compute_loss(x_recon, data, vq_loss)
        self.vqvae.backward(x_recon, data)
        self.optimizer.step(self.vqvae.get_trainable_layers())

        usage = self.vqvae.get_codebook_usage()

        self.history['total_loss'].append(total)
        self.history['recon_loss'].append(recon)
        self.history['vq_loss'].append(vq)
        self.history['codebook_usage'].append(usage)

        return total, recon, vq

    def train(self, data, epochs=10, verbose=False):
        for epoch in range(epochs):
            total, recon, vq = self.train_epoch(data)
            if verbose:
                print(f"Epoch {epoch}: total={total:.4f} recon={recon:.4f} vq={vq:.4f}")
        return self.history


# ---------------------------------------------------------------------------
# Data generation utilities
# ---------------------------------------------------------------------------

def make_vae_data(n=100, dim=8, n_modes=3, seed=42):
    """Generate synthetic multi-modal data for VAE testing."""
    rng = _RNG(seed)
    samples = []
    for i in range(n):
        mode = i % n_modes
        center = [(mode * 2.0 + 1.0) / (n_modes * 2.0)] * dim
        point = [center[d] + rng.normal(0.0, 0.1) for d in range(dim)]
        # Clamp to [0, 1] for sigmoid output
        point = [max(0.01, min(0.99, p)) for p in point]
        samples.append(Tensor(point))
    return samples


def make_labeled_data(n=100, dim=8, n_classes=3, seed=42):
    """Generate synthetic labeled data for CVAE testing."""
    rng = _RNG(seed)
    samples = []
    labels = []
    for i in range(n):
        cls = i % n_classes
        center = [(cls * 2.0 + 1.0) / (n_classes * 2.0)] * dim
        point = [center[d] + rng.normal(0.0, 0.1) for d in range(dim)]
        point = [max(0.01, min(0.99, p)) for p in point]
        samples.append(Tensor(point))
        labels.append(cls)
    return samples, labels
