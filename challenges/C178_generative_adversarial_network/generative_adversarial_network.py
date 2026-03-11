"""
C178: Generative Adversarial Network (GAN)
Composing C140 (Neural Network)

Components:
- Generator: noise -> fake samples
- Discriminator: sample -> real/fake probability
- GAN: adversarial training framework
- WGAN: Wasserstein GAN with gradient penalty
- ConditionalGAN: class-conditioned generation
- GANTrainer: training loop with diagnostics
- WGANTrainer: Wasserstein training with critic iterations
- SpectralNorm: spectral normalization for discriminator stability
- ModeCollapseDetector: detects mode collapse during training
- GANEvaluator: generation quality metrics
"""

import sys, os, math, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Activation, BatchNorm, Dropout, Sequential,
    MSELoss, BinaryCrossEntropyLoss, CrossEntropyLoss,
    SGD, Adam, RMSProp,
    relu, sigmoid, tanh_act, leaky_relu,
    relu_deriv, sigmoid_deriv, tanh_deriv, leaky_relu_deriv,
    xavier_init, he_init,
    softmax, softmax_batch,
    one_hot, train_step,
)


# ---- Tensor helpers (C140-compatible: nested lists for 2D) ----

def _get_row(t, i):
    """Get row i from a 2D tensor as a 1D tensor."""
    if len(t.shape) == 1:
        return t
    return Tensor(list(t.data[i]))

def _set_row(t, i, row):
    """Set row i of a 2D tensor."""
    t.data[i] = list(row.data)

def _stack_rows(rows):
    """Stack 1D tensors into a 2D tensor."""
    if not rows:
        return Tensor([])
    return Tensor([list(r.data) for r in rows])

def _concat_tensors(a, b):
    """Concatenate two tensors along last axis."""
    if len(a.shape) == 1 and len(b.shape) == 1:
        return Tensor(list(a.data) + list(b.data))
    rows_a = a.shape[0] if len(a.shape) == 2 else 1
    a_data = a.data if len(a.shape) == 2 else [a.data]
    b_data = b.data if len(b.shape) == 2 else [b.data]
    result = []
    for i in range(rows_a):
        result.append(list(a_data[i]) + list(b_data[i]))
    return Tensor(result)

def _batch_size(t):
    return t.shape[0] if len(t.shape) == 2 else 1

def _ones_like(t):
    if len(t.shape) == 2:
        return Tensor([[1.0] * t.shape[1] for _ in range(t.shape[0])])
    return Tensor([1.0] * t.shape[0])

def _zeros_like(t):
    if len(t.shape) == 2:
        return Tensor([[0.0] * t.shape[1] for _ in range(t.shape[0])])
    return Tensor([0.0] * t.shape[0])

def _scale_tensor(t, s):
    if len(t.shape) == 2:
        return Tensor([[x * s for x in row] for row in t.data])
    return Tensor([x * s for x in t.data])

def _add_tensors(a, b):
    if len(a.shape) == 2:
        return Tensor([[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a.data, b.data)])
    return Tensor([x + y for x, y in zip(a.data, b.data)])

def _sub_tensors(a, b):
    if len(a.shape) == 2:
        return Tensor([[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a.data, b.data)])
    return Tensor([x - y for x, y in zip(a.data, b.data)])

def _mul_tensors(a, b):
    if len(a.shape) == 2:
        return Tensor([[x * y for x, y in zip(ra, rb)] for ra, rb in zip(a.data, b.data)])
    return Tensor([x * y for x, y in zip(a.data, b.data)])

def _apply_tensor(t, fn):
    if len(t.shape) == 2:
        return Tensor([[fn(x) for x in row] for row in t.data])
    return Tensor([fn(x) for x in t.data])

def _tensor_mean(t):
    if len(t.shape) == 2:
        total = sum(sum(row) for row in t.data)
        count = t.shape[0] * t.shape[1]
        return total / count
    return sum(t.data) / len(t.data)

def _tensor_sum(t):
    if len(t.shape) == 2:
        return sum(sum(row) for row in t.data)
    return sum(t.data)

def _flat_values(t):
    """Get all values as a flat list."""
    if len(t.shape) == 2:
        return [x for row in t.data for x in row]
    return list(t.data)

def _clip_tensor(t, lo, hi):
    if len(t.shape) == 2:
        return Tensor([[max(lo, min(hi, x)) for x in row] for row in t.data])
    return Tensor([max(lo, min(hi, x)) for x in t.data])

def _to_flat(t):
    """Convert (N,1) or (N,) tensor to flat list of floats."""
    if len(t.shape) == 2 and t.shape[1] == 1:
        return [row[0] for row in t.data]
    return list(t.data)

def _from_flat(vals, shape):
    """Convert flat list back to tensor with given shape."""
    if len(shape) == 2 and shape[1] == 1:
        return Tensor([[v] for v in vals])
    return Tensor(list(vals))


class _RNG:
    """Reproducible random number generator."""

    def __init__(self, seed=42):
        self._rng = random.Random(seed)

    def random(self):
        return self._rng.random()

    def uniform(self, lo, hi):
        return self._rng.uniform(lo, hi)

    def normal(self, mean=0.0, std=1.0):
        u1 = max(1e-10, self._rng.random())
        u2 = self._rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    def normal_tensor(self, shape, mean=0.0, std=1.0):
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) == 1:
            return Tensor([self.normal(mean, std) for _ in range(shape[0])])
        elif len(shape) == 2:
            return Tensor([[self.normal(mean, std) for _ in range(shape[1])]
                          for _ in range(shape[0])])
        else:
            raise ValueError("Only 1D and 2D tensors supported")

    def shuffle(self, lst):
        for i in range(len(lst) - 1, 0, -1):
            j = int(self._rng.random() * (i + 1))
            lst[i], lst[j] = lst[j], lst[i]


# ---- Generator ----

class Generator:
    """Maps noise vector z to generated samples."""

    def __init__(self, noise_dim, hidden_dims, output_dim,
                 activation='leaky_relu', output_activation='tanh',
                 use_batchnorm=True, dropout=0.0, seed=42):
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.rng = _RNG(seed)

        layers = []
        prev_dim = noise_dim
        for h in hidden_dims:
            layers.append(Dense(prev_dim, h, init='he'))
            if use_batchnorm:
                layers.append(BatchNorm(h))
            layers.append(Activation(activation))
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev_dim = h
        layers.append(Dense(prev_dim, output_dim, init='xavier'))
        if output_activation:
            layers.append(Activation(output_activation))

        self.network = Sequential(layers)

    def forward(self, z):
        self._input = z
        return self.network.forward(z)

    def backward(self, grad):
        return self.network.backward(grad)

    def get_trainable_layers(self):
        return self.network.get_trainable_layers()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def generate(self, n_samples):
        """Generate n samples from random noise."""
        z = self.rng.normal_tensor((n_samples, self.noise_dim))
        self.eval()
        out = self.forward(z)
        self.train()
        return out

    def count_params(self):
        return self.network.count_params()


# ---- Discriminator ----

class Discriminator:
    """Classifies samples as real (1) or fake (0)."""

    def __init__(self, input_dim, hidden_dims, activation='leaky_relu',
                 use_batchnorm=False, dropout=0.3, seed=42):
        self.input_dim = input_dim
        self.rng = _RNG(seed)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(Dense(prev_dim, h, init='he'))
            if use_batchnorm:
                layers.append(BatchNorm(h))
            layers.append(Activation(activation))
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev_dim = h
        layers.append(Dense(prev_dim, 1, init='xavier'))
        layers.append(Activation('sigmoid'))

        self.network = Sequential(layers)

    def forward(self, x):
        self._input = x
        return self.network.forward(x)

    def backward(self, grad):
        return self.network.backward(grad)

    def get_trainable_layers(self):
        return self.network.get_trainable_layers()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def count_params(self):
        return self.network.count_params()


# ---- Critic (for WGAN) ----

class Critic:
    """Wasserstein critic -- no sigmoid, outputs unbounded score."""

    def __init__(self, input_dim, hidden_dims, activation='leaky_relu',
                 use_batchnorm=False, dropout=0.0, seed=42):
        self.input_dim = input_dim
        self.rng = _RNG(seed)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(Dense(prev_dim, h, init='he'))
            if use_batchnorm:
                layers.append(BatchNorm(h))
            layers.append(Activation(activation))
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev_dim = h
        layers.append(Dense(prev_dim, 1, init='xavier'))
        # No sigmoid -- raw score

        self.network = Sequential(layers)

    def forward(self, x):
        self._input = x
        return self.network.forward(x)

    def backward(self, grad):
        return self.network.backward(grad)

    def get_trainable_layers(self):
        return self.network.get_trainable_layers()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def count_params(self):
        return self.network.count_params()

    def clip_weights(self, clip_value):
        """Weight clipping for original WGAN."""
        for layer in self.network.layers:
            if hasattr(layer, 'weights'):
                w = layer.weights
                if len(w.shape) == 2:
                    w.data = [[max(-clip_value, min(clip_value, x)) for x in row] for row in w.data]
                else:
                    w.data = [max(-clip_value, min(clip_value, x)) for x in w.data]
            if hasattr(layer, 'bias') and layer.bias is not None:
                b = layer.bias
                b.data = [max(-clip_value, min(clip_value, x)) for x in b.data]


# ---- GAN Loss Functions ----

class GANLoss:
    """Standard GAN loss (minimax / non-saturating)."""

    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing
        self.bce = BinaryCrossEntropyLoss()
        self.eps = 1e-7

    def discriminator_loss(self, d_real, d_fake):
        """D loss: -[log(D(x)) + log(1 - D(G(z)))]"""
        real_target = 1.0 - self.label_smoothing
        real_vals = _to_flat(d_real)
        fake_vals = _to_flat(d_fake)
        n = len(real_vals)
        loss_real = 0.0
        loss_fake = 0.0
        for i in range(n):
            dr = max(self.eps, min(1.0 - self.eps, real_vals[i]))
            df = max(self.eps, min(1.0 - self.eps, fake_vals[i]))
            loss_real -= math.log(dr) * real_target + math.log(1.0 - dr) * (1.0 - real_target)
            loss_fake -= math.log(1.0 - df)
        return (loss_real + loss_fake) / n

    def discriminator_grad_real(self, d_real):
        """Gradient of D loss w.r.t. D(real)."""
        real_target = 1.0 - self.label_smoothing
        vals = _to_flat(d_real)
        n = len(vals)
        grad = []
        for i in range(n):
            dr = max(self.eps, min(1.0 - self.eps, vals[i]))
            g = (-real_target / dr + (1.0 - real_target) / (1.0 - dr)) / n
            grad.append(g)
        return _from_flat(grad, d_real.shape)

    def discriminator_grad_fake(self, d_fake):
        """Gradient of D loss w.r.t. D(fake)."""
        vals = _to_flat(d_fake)
        n = len(vals)
        grad = []
        for i in range(n):
            df = max(self.eps, min(1.0 - self.eps, vals[i]))
            g = (1.0 / (1.0 - df)) / n
            grad.append(g)
        return _from_flat(grad, d_fake.shape)

    def generator_loss(self, d_fake):
        """Non-saturating G loss: -log(D(G(z)))"""
        vals = _to_flat(d_fake)
        n = len(vals)
        loss = 0.0
        for i in range(n):
            df = max(self.eps, min(1.0 - self.eps, vals[i]))
            loss -= math.log(df)
        return loss / n

    def generator_grad(self, d_fake):
        """Gradient of G loss w.r.t. D(fake)."""
        vals = _to_flat(d_fake)
        n = len(vals)
        grad = []
        for i in range(n):
            df = max(self.eps, min(1.0 - self.eps, vals[i]))
            g = (-1.0 / df) / n
            grad.append(g)
        return _from_flat(grad, d_fake.shape)


class WassersteinLoss:
    """Wasserstein loss for WGAN."""

    def critic_loss(self, c_real, c_fake):
        """W loss: E[C(fake)] - E[C(real)] (minimize -> maximize E[C(real)] - E[C(fake)])"""
        real_vals = _to_flat(c_real)
        fake_vals = _to_flat(c_fake)
        mean_real = sum(real_vals) / len(real_vals)
        mean_fake = sum(fake_vals) / len(fake_vals)
        return mean_fake - mean_real

    def critic_grad_real(self, c_real):
        """Gradient w.r.t. C(real): -1/n"""
        vals = _to_flat(c_real)
        n = len(vals)
        return _from_flat([-1.0 / n] * n, c_real.shape)

    def critic_grad_fake(self, c_fake):
        """Gradient w.r.t. C(fake): 1/n"""
        vals = _to_flat(c_fake)
        n = len(vals)
        return _from_flat([1.0 / n] * n, c_fake.shape)

    def generator_loss(self, c_fake):
        """G loss: -E[C(G(z))]"""
        vals = _to_flat(c_fake)
        return -sum(vals) / len(vals)

    def generator_grad(self, c_fake):
        """Gradient of G loss w.r.t. C(fake): -1/n"""
        vals = _to_flat(c_fake)
        n = len(vals)
        return _from_flat([-1.0 / n] * n, c_fake.shape)


class GradientPenalty:
    """Gradient penalty for WGAN-GP."""

    def __init__(self, lambda_gp=10.0):
        self.lambda_gp = lambda_gp

    def compute(self, critic, real, fake, rng):
        """Compute gradient penalty via finite differences."""
        batch = _batch_size(real)
        total_penalty = 0.0

        for i in range(batch):
            eps = rng.random()
            real_row = _get_row(real, i)
            fake_row = _get_row(fake, i)
            # Interpolate: x_hat = eps * real + (1 - eps) * fake
            interp_data = [eps * r + (1 - eps) * f
                          for r, f in zip(real_row.data, fake_row.data)]
            x_hat = _stack_rows([Tensor(interp_data)])

            # Estimate gradient norm via finite differences
            delta = 1e-4
            base_out = critic.forward(x_hat)
            base_val = _to_flat(base_out)[0]

            grad_sq_sum = 0.0
            dim = len(interp_data)
            for d in range(dim):
                perturbed = list(interp_data)
                perturbed[d] += delta
                p_tensor = _stack_rows([Tensor(perturbed)])
                p_out = critic.forward(p_tensor)
                grad_d = (_to_flat(p_out)[0] - base_val) / delta
                grad_sq_sum += grad_d * grad_d

            grad_norm = math.sqrt(grad_sq_sum + 1e-12)
            total_penalty += (grad_norm - 1.0) ** 2

        return self.lambda_gp * total_penalty / batch


# ---- Spectral Normalization ----

class SpectralNorm:
    """Spectral normalization for weight matrices."""

    def __init__(self, n_power_iterations=1, seed=42):
        self.n_iterations = n_power_iterations
        self.rng = _RNG(seed)
        self._u_vectors = {}

    def normalize(self, layer):
        """Apply spectral normalization to a Dense layer's weights."""
        if not hasattr(layer, 'weights'):
            return
        w = layer.weights
        if len(w.shape) != 2:
            return
        key = id(layer)
        rows, cols = w.shape
        if key not in self._u_vectors:
            u = [self.rng.normal() for _ in range(rows)]
            norm = math.sqrt(sum(x * x for x in u) + 1e-12)
            u = [x / norm for x in u]
            self._u_vectors[key] = u

        u = list(self._u_vectors[key])

        for _ in range(self.n_iterations):
            # v = W^T u
            v = [0.0] * cols
            for i in range(rows):
                for j in range(cols):
                    v[j] += w.data[i][j] * u[i]
            norm_v = math.sqrt(sum(x * x for x in v) + 1e-12)
            v = [x / norm_v for x in v]

            # u = W v
            u = [0.0] * rows
            for i in range(rows):
                for j in range(cols):
                    u[i] += w.data[i][j] * v[j]
            norm_u = math.sqrt(sum(x * x for x in u) + 1e-12)
            u = [x / norm_u for x in u]

        self._u_vectors[key] = u

        # sigma = u^T W v
        sigma = 0.0
        for i in range(rows):
            for j in range(cols):
                sigma += u[i] * w.data[i][j] * v[j]
        sigma = max(sigma, 1e-12)

        # Normalize: W = W / sigma
        w.data = [[x / sigma for x in row] for row in w.data]


# ---- Standard GAN ----

class GAN:
    """Standard Generative Adversarial Network."""

    def __init__(self, noise_dim, gen_hidden, disc_hidden, data_dim,
                 gen_activation='leaky_relu', disc_activation='leaky_relu',
                 gen_output_activation='tanh',
                 label_smoothing=0.0, seed=42):
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.rng = _RNG(seed)

        self.generator = Generator(
            noise_dim, gen_hidden, data_dim,
            activation=gen_activation,
            output_activation=gen_output_activation,
            use_batchnorm=True, seed=seed
        )
        self.discriminator = Discriminator(
            data_dim, disc_hidden,
            activation=disc_activation,
            use_batchnorm=False, dropout=0.3, seed=seed + 1
        )
        self.loss_fn = GANLoss(label_smoothing=label_smoothing)

    def sample_noise(self, n):
        return self.rng.normal_tensor((n, self.noise_dim))

    def generate(self, n):
        z = self.sample_noise(n)
        self.generator.eval()
        out = self.generator.forward(z)
        self.generator.train()
        return out

    def discriminator_step(self, real_data, g_optimizer=None, d_optimizer=None):
        """One discriminator training step. Returns d_loss."""
        batch = _batch_size(real_data)
        z = self.sample_noise(batch)

        # Forward
        self.generator.eval()
        fake_data = self.generator.forward(z)
        self.generator.train()

        self.discriminator.train()
        d_real = self.discriminator.forward(real_data)
        d_fake = self.discriminator.forward(fake_data)

        d_loss = self.loss_fn.discriminator_loss(d_real, d_fake)

        # Backward through discriminator on fake
        grad_fake = self.loss_fn.discriminator_grad_fake(d_fake)
        self.discriminator.forward(fake_data)
        self.discriminator.backward(grad_fake)
        if d_optimizer:
            d_optimizer.step(self.discriminator.get_trainable_layers())

        # Backward through discriminator on real
        grad_real = self.loss_fn.discriminator_grad_real(d_real)
        self.discriminator.forward(real_data)
        self.discriminator.backward(grad_real)
        if d_optimizer:
            d_optimizer.step(self.discriminator.get_trainable_layers())

        return d_loss

    def generator_step(self, batch_size, g_optimizer=None):
        """One generator training step. Returns g_loss."""
        z = self.sample_noise(batch_size)

        self.generator.train()
        self.discriminator.train()

        fake_data = self.generator.forward(z)
        d_fake = self.discriminator.forward(fake_data)

        g_loss = self.loss_fn.generator_loss(d_fake)

        # Backward: through discriminator then generator
        grad_d = self.loss_fn.generator_grad(d_fake)
        grad_gen_output = self.discriminator.backward(grad_d)
        self.generator.backward(grad_gen_output)

        if g_optimizer:
            g_optimizer.step(self.generator.get_trainable_layers())

        return g_loss


# ---- Wasserstein GAN ----

class WGAN:
    """Wasserstein GAN with optional gradient penalty."""

    def __init__(self, noise_dim, gen_hidden, critic_hidden, data_dim,
                 gen_activation='leaky_relu', critic_activation='leaky_relu',
                 gen_output_activation='tanh',
                 clip_value=0.01, use_gp=True, lambda_gp=10.0, seed=42):
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.clip_value = clip_value
        self.use_gp = use_gp
        self.rng = _RNG(seed)

        self.generator = Generator(
            noise_dim, gen_hidden, data_dim,
            activation=gen_activation,
            output_activation=gen_output_activation,
            use_batchnorm=True, seed=seed
        )
        self.critic = Critic(
            data_dim, critic_hidden,
            activation=critic_activation,
            use_batchnorm=False, seed=seed + 1
        )
        self.loss_fn = WassersteinLoss()
        self.gp = GradientPenalty(lambda_gp) if use_gp else None

    def sample_noise(self, n):
        return self.rng.normal_tensor((n, self.noise_dim))

    def generate(self, n):
        z = self.sample_noise(n)
        self.generator.eval()
        out = self.generator.forward(z)
        self.generator.train()
        return out

    def critic_step(self, real_data, c_optimizer=None):
        """One critic training step. Returns w_distance estimate."""
        batch = _batch_size(real_data)
        z = self.sample_noise(batch)

        self.generator.eval()
        fake_data = self.generator.forward(z)
        self.generator.train()

        self.critic.train()
        c_real = self.critic.forward(real_data)
        c_fake = self.critic.forward(fake_data)

        w_loss = self.loss_fn.critic_loss(c_real, c_fake)

        # Backward on fake
        grad_fake = self.loss_fn.critic_grad_fake(c_fake)
        self.critic.forward(fake_data)
        self.critic.backward(grad_fake)
        if c_optimizer:
            c_optimizer.step(self.critic.get_trainable_layers())

        # Backward on real
        grad_real = self.loss_fn.critic_grad_real(c_real)
        self.critic.forward(real_data)
        self.critic.backward(grad_real)
        if c_optimizer:
            c_optimizer.step(self.critic.get_trainable_layers())

        # Weight clipping or GP
        if not self.use_gp:
            self.critic.clip_weights(self.clip_value)

        return w_loss

    def generator_step(self, batch_size, g_optimizer=None):
        """One generator training step."""
        z = self.sample_noise(batch_size)

        self.generator.train()
        self.critic.train()

        fake_data = self.generator.forward(z)
        c_fake = self.critic.forward(fake_data)

        g_loss = self.loss_fn.generator_loss(c_fake)

        grad_c = self.loss_fn.generator_grad(c_fake)
        grad_gen_output = self.critic.backward(grad_c)
        self.generator.backward(grad_gen_output)

        if g_optimizer:
            g_optimizer.step(self.generator.get_trainable_layers())

        return g_loss


# ---- Conditional GAN ----

class ConditionalGenerator:
    """Generator conditioned on class labels."""

    def __init__(self, noise_dim, num_classes, hidden_dims, output_dim,
                 activation='leaky_relu', output_activation='tanh',
                 use_batchnorm=True, seed=42):
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.rng = _RNG(seed)

        # Input: noise + one-hot label
        input_dim = noise_dim + num_classes
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(Dense(prev, h, init='he'))
            if use_batchnorm:
                layers.append(BatchNorm(h))
            layers.append(Activation(activation))
            prev = h
        layers.append(Dense(prev, output_dim, init='xavier'))
        if output_activation:
            layers.append(Activation(output_activation))

        self.network = Sequential(layers)

    def forward(self, z, labels):
        """Forward pass. labels: Tensor of int class indices or one-hot."""
        batch = _batch_size(z)
        rows = []
        for i in range(batch):
            z_row = _get_row(z, i)
            if len(labels.shape) == 1:
                # Integer labels -> one-hot
                oh = [0.0] * self.num_classes
                oh[int(labels.data[i])] = 1.0
                label_row = Tensor(oh)
            else:
                label_row = _get_row(labels, i)
            combined = Tensor(z_row.data + label_row.data)
            rows.append(combined)
        x = _stack_rows(rows)
        self._input = x
        return self.network.forward(x)

    def backward(self, grad):
        return self.network.backward(grad)

    def get_trainable_layers(self):
        return self.network.get_trainable_layers()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def count_params(self):
        return self.network.count_params()


class ConditionalDiscriminator:
    """Discriminator conditioned on class labels."""

    def __init__(self, input_dim, num_classes, hidden_dims,
                 activation='leaky_relu', dropout=0.3, seed=42):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.rng = _RNG(seed)

        cond_input_dim = input_dim + num_classes
        layers = []
        prev = cond_input_dim
        for h in hidden_dims:
            layers.append(Dense(prev, h, init='he'))
            layers.append(Activation(activation))
            if dropout > 0:
                layers.append(Dropout(dropout))
            prev = h
        layers.append(Dense(prev, 1, init='xavier'))
        layers.append(Activation('sigmoid'))

        self.network = Sequential(layers)

    def forward(self, x, labels):
        batch = _batch_size(x)
        rows = []
        for i in range(batch):
            x_row = _get_row(x, i)
            if len(labels.shape) == 1:
                oh = [0.0] * self.num_classes
                oh[int(labels.data[i])] = 1.0
                label_row = Tensor(oh)
            else:
                label_row = _get_row(labels, i)
            combined = Tensor(x_row.data + label_row.data)
            rows.append(combined)
        cond_x = _stack_rows(rows)
        self._input = cond_x
        return self.network.forward(cond_x)

    def backward(self, grad):
        return self.network.backward(grad)

    def get_trainable_layers(self):
        return self.network.get_trainable_layers()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def count_params(self):
        return self.network.count_params()


class ConditionalGAN:
    """Conditional GAN with class-conditioned generation."""

    def __init__(self, noise_dim, num_classes, gen_hidden, disc_hidden, data_dim,
                 label_smoothing=0.0, seed=42):
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.data_dim = data_dim
        self.rng = _RNG(seed)

        self.generator = ConditionalGenerator(
            noise_dim, num_classes, gen_hidden, data_dim, seed=seed
        )
        self.discriminator = ConditionalDiscriminator(
            data_dim, num_classes, disc_hidden, seed=seed + 1
        )
        self.loss_fn = GANLoss(label_smoothing=label_smoothing)

    def sample_noise(self, n):
        return self.rng.normal_tensor((n, self.noise_dim))

    def generate(self, labels, n_per_class=1):
        """Generate samples for given class labels."""
        if isinstance(labels, int):
            labels = [labels] * n_per_class
        if isinstance(labels, list):
            labels = Tensor([float(l) for l in labels])
        n = len(labels.data)
        z = self.sample_noise(n)
        self.generator.eval()
        out = self.generator.forward(z, labels)
        self.generator.train()
        return out

    def discriminator_step(self, real_data, labels, d_optimizer=None):
        batch = _batch_size(real_data)
        z = self.sample_noise(batch)

        self.generator.eval()
        fake_data = self.generator.forward(z, labels)
        self.generator.train()

        self.discriminator.train()
        d_real = self.discriminator.forward(real_data, labels)
        d_fake = self.discriminator.forward(fake_data, labels)

        d_loss = self.loss_fn.discriminator_loss(d_real, d_fake)

        grad_fake = self.loss_fn.discriminator_grad_fake(d_fake)
        self.discriminator.forward(fake_data, labels)
        self.discriminator.backward(grad_fake)
        if d_optimizer:
            d_optimizer.step(self.discriminator.get_trainable_layers())

        grad_real = self.loss_fn.discriminator_grad_real(d_real)
        self.discriminator.forward(real_data, labels)
        self.discriminator.backward(grad_real)
        if d_optimizer:
            d_optimizer.step(self.discriminator.get_trainable_layers())

        return d_loss

    def generator_step(self, batch_size, labels, g_optimizer=None):
        z = self.sample_noise(batch_size)

        self.generator.train()
        self.discriminator.train()

        fake_data = self.generator.forward(z, labels)
        d_fake = self.discriminator.forward(fake_data, labels)

        g_loss = self.loss_fn.generator_loss(d_fake)

        grad_d = self.loss_fn.generator_grad(d_fake)
        grad_full = self.discriminator.backward(grad_d)
        # grad_full has shape (batch, data_dim + num_classes)
        # Extract only the data part for generator backward
        data_dim = self.data_dim
        if len(grad_full.shape) == 2:
            grad_gen_out = Tensor([row[:data_dim] for row in grad_full.data])
        else:
            grad_gen_out = Tensor(grad_full.data[:data_dim])
        self.generator.backward(grad_gen_out)

        if g_optimizer:
            g_optimizer.step(self.generator.get_trainable_layers())

        return g_loss


# ---- Mode Collapse Detector ----

class ModeCollapseDetector:
    """Detects mode collapse by analyzing generated sample diversity."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.diversity_history = []

    def compute_diversity(self, samples):
        """Compute pairwise distance diversity of samples."""
        batch = _batch_size(samples)
        if batch < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(min(batch, 20)):  # Cap for performance
            for j in range(i + 1, min(batch, 20)):
                row_i = _get_row(samples, i)
                row_j = _get_row(samples, j)
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row_i.data, row_j.data)) + 1e-12)
                total_dist += dist
                count += 1

        return total_dist / count if count > 0 else 0.0

    def update(self, samples):
        """Update with new batch of generated samples."""
        div = self.compute_diversity(samples)
        self.diversity_history.append(div)
        if len(self.diversity_history) > self.window_size * 3:
            self.diversity_history = self.diversity_history[-self.window_size * 3:]
        return div

    def is_collapsed(self, threshold=0.1):
        """Check if mode collapse is detected."""
        if len(self.diversity_history) < self.window_size:
            return False
        recent = self.diversity_history[-self.window_size:]
        avg_div = sum(recent) / len(recent)
        return avg_div < threshold

    def get_trend(self):
        """Get diversity trend (positive = increasing diversity)."""
        if len(self.diversity_history) < 2:
            return 0.0
        n = min(len(self.diversity_history), self.window_size)
        recent = self.diversity_history[-n:]
        if n < 2:
            return 0.0
        # Simple linear trend
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0


# ---- GAN Evaluator ----

class GANEvaluator:
    """Evaluates GAN generation quality."""

    def __init__(self):
        pass

    def reconstruction_error(self, real, generated):
        """Mean squared error between real and generated distributions."""
        # Compare distribution statistics
        real_mean = [0.0] * (real.shape[1] if len(real.shape) == 2 else len(real.data))
        gen_mean = [0.0] * (generated.shape[1] if len(generated.shape) == 2 else len(generated.data))
        n_real = _batch_size(real)
        n_gen = _batch_size(generated)
        dim = len(real_mean)

        for i in range(n_real):
            row = _get_row(real, i)
            for d in range(dim):
                real_mean[d] += row.data[d]
        real_mean = [x / n_real for x in real_mean]

        for i in range(n_gen):
            row = _get_row(generated, i)
            for d in range(dim):
                gen_mean[d] += row.data[d]
        gen_mean = [x / n_gen for x in gen_mean]

        mse = sum((r - g) ** 2 for r, g in zip(real_mean, gen_mean)) / dim
        return mse

    def discriminator_accuracy(self, discriminator, real, fake):
        """How well discriminator separates real from fake."""
        discriminator.eval()
        d_real = discriminator.forward(real)
        d_fake = discriminator.forward(fake)
        discriminator.train()

        real_vals = _to_flat(d_real)
        fake_vals = _to_flat(d_fake)
        correct = 0
        total = len(real_vals) + len(fake_vals)
        for v in real_vals:
            if v > 0.5:
                correct += 1
        for v in fake_vals:
            if v <= 0.5:
                correct += 1
        return correct / total if total > 0 else 0.0

    def sample_statistics(self, samples):
        """Compute mean, std, min, max of generated samples."""
        batch = _batch_size(samples)
        dim = samples.shape[1] if len(samples.shape) == 2 else len(samples.data)

        means = [0.0] * dim
        for i in range(batch):
            row = _get_row(samples, i)
            for d in range(dim):
                means[d] += row.data[d]
        means = [m / batch for m in means]

        variances = [0.0] * dim
        for i in range(batch):
            row = _get_row(samples, i)
            for d in range(dim):
                variances[d] += (row.data[d] - means[d]) ** 2
        stds = [math.sqrt(v / batch) for v in variances]

        all_vals = _flat_values(samples)
        return {
            'mean': means,
            'std': stds,
            'min': min(all_vals),
            'max': max(all_vals),
            'overall_mean': sum(means) / dim,
            'overall_std': sum(stds) / dim,
        }


# ---- Training Utilities ----

class GANTrainer:
    """Training loop for standard GAN."""

    def __init__(self, gan, g_optimizer, d_optimizer,
                 d_steps=1, g_steps=1, rng=None):
        self.gan = gan
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.d_steps = d_steps
        self.g_steps = g_steps
        self.rng = rng or _RNG(42)
        self.history = {'d_loss': [], 'g_loss': []}
        self.collapse_detector = ModeCollapseDetector()

    def _get_batch(self, X, batch_size, indices, start):
        end = min(start + batch_size, len(indices))
        rows = []
        for idx in indices[start:end]:
            rows.append(_get_row(X, idx))
        return _stack_rows(rows)

    def train_epoch(self, X, batch_size=32):
        """Train one epoch. Returns (avg_d_loss, avg_g_loss)."""
        n = _batch_size(X)
        indices = list(range(n))
        self.rng.shuffle(indices)

        total_d = 0.0
        total_g = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            real_batch = self._get_batch(X, batch_size, indices, start)
            bs = _batch_size(real_batch)
            if bs < 2:
                continue

            # Train discriminator
            for _ in range(self.d_steps):
                d_loss = self.gan.discriminator_step(
                    real_batch, d_optimizer=self.d_optimizer
                )

            # Train generator
            for _ in range(self.g_steps):
                g_loss = self.gan.generator_step(
                    bs, g_optimizer=self.g_optimizer
                )

            total_d += d_loss
            total_g += g_loss
            steps += 1

        avg_d = total_d / max(steps, 1)
        avg_g = total_g / max(steps, 1)
        self.history['d_loss'].append(avg_d)
        self.history['g_loss'].append(avg_g)
        return avg_d, avg_g

    def train(self, X, epochs=100, batch_size=32, verbose=False):
        """Full training loop."""
        for epoch in range(epochs):
            d_loss, g_loss = self.train_epoch(X, batch_size)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - D: {d_loss:.4f}, G: {g_loss:.4f}")

            # Check mode collapse periodically
            if (epoch + 1) % 5 == 0:
                samples = self.gan.generate(min(20, _batch_size(X)))
                self.collapse_detector.update(samples)

        return self.history


class WGANTrainer:
    """Training loop for WGAN."""

    def __init__(self, wgan, g_optimizer, c_optimizer,
                 n_critic=5, rng=None):
        self.wgan = wgan
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.n_critic = n_critic
        self.rng = rng or _RNG(42)
        self.history = {'w_distance': [], 'g_loss': []}

    def _get_batch(self, X, batch_size, indices, start):
        end = min(start + batch_size, len(indices))
        rows = []
        for idx in indices[start:end]:
            rows.append(_get_row(X, idx))
        return _stack_rows(rows)

    def train_epoch(self, X, batch_size=32):
        n = _batch_size(X)
        indices = list(range(n))
        self.rng.shuffle(indices)

        total_w = 0.0
        total_g = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            real_batch = self._get_batch(X, batch_size, indices, start)
            bs = _batch_size(real_batch)
            if bs < 2:
                continue

            # Train critic multiple times
            for _ in range(self.n_critic):
                w_loss = self.wgan.critic_step(
                    real_batch, c_optimizer=self.c_optimizer
                )

            # Train generator
            g_loss = self.wgan.generator_step(
                bs, g_optimizer=self.g_optimizer
            )

            total_w += w_loss
            total_g += g_loss
            steps += 1

        avg_w = total_w / max(steps, 1)
        avg_g = total_g / max(steps, 1)
        self.history['w_distance'].append(avg_w)
        self.history['g_loss'].append(avg_g)
        return avg_w, avg_g

    def train(self, X, epochs=100, batch_size=32, verbose=False):
        for epoch in range(epochs):
            w_dist, g_loss = self.train_epoch(X, batch_size)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - W: {w_dist:.4f}, G: {g_loss:.4f}")
        return self.history


class ConditionalGANTrainer:
    """Training loop for Conditional GAN."""

    def __init__(self, cgan, g_optimizer, d_optimizer,
                 d_steps=1, rng=None):
        self.cgan = cgan
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.d_steps = d_steps
        self.rng = rng or _RNG(42)
        self.history = {'d_loss': [], 'g_loss': []}

    def _get_batch(self, X, Y, batch_size, indices, start):
        end = min(start + batch_size, len(indices))
        x_rows = []
        y_vals = []
        for idx in indices[start:end]:
            x_rows.append(_get_row(X, idx))
            if len(Y.shape) == 1:
                y_vals.append(Y.data[idx])
            else:
                y_vals.append(_get_row(Y, idx).data)
        x_batch = _stack_rows(x_rows)
        if isinstance(y_vals[0], (int, float)):
            y_batch = Tensor(y_vals)
        else:
            y_batch = _stack_rows([Tensor(v) for v in y_vals])
        return x_batch, y_batch

    def train_epoch(self, X, Y, batch_size=32):
        n = _batch_size(X)
        indices = list(range(n))
        self.rng.shuffle(indices)

        total_d = 0.0
        total_g = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            x_batch, y_batch = self._get_batch(X, Y, batch_size, indices, start)
            bs = _batch_size(x_batch)
            if bs < 2:
                continue

            for _ in range(self.d_steps):
                d_loss = self.cgan.discriminator_step(
                    x_batch, y_batch, d_optimizer=self.d_optimizer
                )

            g_loss = self.cgan.generator_step(
                bs, y_batch, g_optimizer=self.g_optimizer
            )

            total_d += d_loss
            total_g += g_loss
            steps += 1

        avg_d = total_d / max(steps, 1)
        avg_g = total_g / max(steps, 1)
        self.history['d_loss'].append(avg_d)
        self.history['g_loss'].append(avg_g)
        return avg_d, avg_g

    def train(self, X, Y, epochs=100, batch_size=32, verbose=False):
        for epoch in range(epochs):
            d_loss, g_loss = self.train_epoch(X, Y, batch_size)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} - D: {d_loss:.4f}, G: {g_loss:.4f}")
        return self.history


# ---- Noise Interpolation ----

class NoiseInterpolator:
    """Interpolation in latent noise space."""

    def __init__(self, generator, rng=None):
        self.generator = generator
        self.rng = rng or _RNG(42)

    def linear_interpolate(self, z1, z2, steps=10):
        """Linear interpolation between two noise vectors."""
        results = []
        for i in range(steps):
            alpha = i / max(steps - 1, 1)
            z = Tensor([a * (1 - alpha) + b * alpha for a, b in zip(z1.data, z2.data)])
            z_batch = _stack_rows([z])
            self.generator.eval()
            out = self.generator.forward(z_batch)
            self.generator.train()
            results.append(_get_row(out, 0))
        return results

    def slerp(self, z1, z2, steps=10):
        """Spherical linear interpolation."""
        # Normalize
        n1 = math.sqrt(sum(x * x for x in z1.data) + 1e-12)
        n2 = math.sqrt(sum(x * x for x in z2.data) + 1e-12)
        z1_norm = [x / n1 for x in z1.data]
        z2_norm = [x / n2 for x in z2.data]

        dot = sum(a * b for a, b in zip(z1_norm, z2_norm))
        dot = max(-1.0, min(1.0, dot))
        omega = math.acos(dot)

        results = []
        for i in range(steps):
            t = i / max(steps - 1, 1)
            if abs(omega) < 1e-6:
                z = [a * (1 - t) + b * t for a, b in zip(z1.data, z2.data)]
            else:
                s1 = math.sin((1 - t) * omega) / math.sin(omega)
                s2 = math.sin(t * omega) / math.sin(omega)
                z = [a * s1 + b * s2 for a, b in zip(z1.data, z2.data)]
            z_batch = _stack_rows([Tensor(z)])
            self.generator.eval()
            out = self.generator.forward(z_batch)
            self.generator.train()
            results.append(_get_row(out, 0))
        return results
