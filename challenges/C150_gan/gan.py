"""
C150: Generative Adversarial Network (GAN)
Composing C140 (Neural Networks) + C149 (Autoencoder)

Implements:
- GAN: vanilla GAN with minimax training
- WGAN: Wasserstein GAN with weight clipping for stable training
- LSGAN: Least Squares GAN (replaces log loss with L2)
- ConditionalGAN: class-conditional generation
- InfoGAN: information-maximizing GAN (disentangled latent codes)
- BiGAN: bidirectional GAN with encoder (inference network)

Key concepts:
- Generator: maps noise z -> fake data (decoder architecture)
- Discriminator: classifies real vs fake (encoder-like)
- Adversarial training: G and D play minimax game
- Mode collapse mitigation (label smoothing, noise injection)
- Wasserstein distance for training stability
- Conditional generation with class labels
- Mutual information maximization (InfoGAN)
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Activation, Sequential, Dropout, BatchNorm,
    MSELoss, BinaryCrossEntropyLoss, SGD, Adam, RMSProp,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C149_autoencoder'))
from autoencoder import (
    Encoder, Decoder, _tensor_bce, _tensor_bce_grad,
    _tensor_mse, _tensor_mse_grad, _safe_log, _clip,
)


# ============================================================
# Utility functions
# ============================================================

def _ones_tensor(shape):
    """Create tensor of ones."""
    if isinstance(shape, int):
        return Tensor([1.0] * shape)
    rows, cols = shape
    return Tensor([[1.0] * cols for _ in range(rows)])


def _zeros_tensor(shape):
    """Create tensor of zeros."""
    if isinstance(shape, int):
        return Tensor([0.0] * shape)
    rows, cols = shape
    return Tensor([[0.0] * cols for _ in range(rows)])


def _random_noise(batch_size, noise_dim, rng=None):
    """Generate random noise from N(0, 1)."""
    r = rng or random.Random()
    return Tensor([[r.gauss(0, 1) for _ in range(noise_dim)]
                   for _ in range(batch_size)])


def _concat_tensors(a, b):
    """Concatenate two 2D tensors along columns (axis=1)."""
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("Both tensors must be 2D")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Batch size mismatch: {a.shape[0]} vs {b.shape[0]}")
    result = []
    for i in range(a.shape[0]):
        result.append(a.data[i] + b.data[i])
    return Tensor(result)


def _one_hot(labels, num_classes):
    """Convert labels to one-hot encoding as Tensor."""
    result = []
    for label in labels:
        row = [0.0] * num_classes
        row[label] = 1.0
        result.append(row)
    return Tensor(result)


def _slice_tensor(t, start, end):
    """Slice rows from a 2D tensor."""
    if len(t.shape) == 2:
        return Tensor(t.data[start:end])
    return Tensor(t.data[start:end])


def _sigmoid_scalar(x):
    """Sigmoid for a scalar value."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def _apply_sigmoid(t):
    """Apply sigmoid element-wise to tensor."""
    if len(t.shape) == 2:
        return Tensor([[_sigmoid_scalar(t.data[i][j])
                        for j in range(t.shape[1])]
                       for i in range(t.shape[0])])
    return Tensor([_sigmoid_scalar(v) for v in t.data])


# ============================================================
# Discriminator
# ============================================================

class Discriminator:
    """Discriminator network: data -> [0, 1] (real/fake probability).

    Uses feedforward network with sigmoid output.
    """

    def __init__(self, input_dim, hidden_dims, activation='leaky_relu',
                 use_sigmoid=True, init='he', dropout=None, rng=None):
        self.input_dim = input_dim
        self.use_sigmoid = use_sigmoid
        self.network = Sequential()
        self.rng = rng

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.network.add(Dense(prev_dim, h_dim, init=init, rng=rng))
            self.network.add(Activation(activation))
            if dropout and dropout > 0:
                self.network.add(Dropout(dropout, rng=rng))
            prev_dim = h_dim

        # Output layer: single logit
        self.output_layer = Dense(prev_dim, 1, init=init, rng=rng)
        if use_sigmoid:
            self.output_act = Activation('sigmoid')
        else:
            self.output_act = None

    def forward(self, x):
        """Forward pass. Returns probabilities (or raw logits if no sigmoid)."""
        h = self.network.forward(x)
        out = self.output_layer.forward(h)
        if self.output_act:
            out = self.output_act.forward(out)
        self.last_output = out
        return out

    def backward(self, grad):
        """Backward pass through discriminator."""
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

    def clip_weights(self, clip_value):
        """Clip all weights to [-clip_value, clip_value] (for WGAN)."""
        for layer in self.get_trainable_layers():
            for param_tuple in layer.get_params():
                tensor = param_tuple[0]
                if len(tensor.shape) == 2:
                    for i in range(tensor.shape[0]):
                        for j in range(tensor.shape[1]):
                            tensor.data[i][j] = max(-clip_value,
                                                     min(clip_value, tensor.data[i][j]))
                else:
                    for i in range(len(tensor.data)):
                        tensor.data[i] = max(-clip_value, min(clip_value, tensor.data[i]))

    def count_params(self):
        total = 0
        for layer in self.get_trainable_layers():
            for param_tuple in layer.get_params():
                t = param_tuple[0]
                flat = t.flatten().data
                total += len(flat)
        return total


# ============================================================
# Generator
# ============================================================

class Generator:
    """Generator network: noise_dim -> data_dim.

    Uses feedforward network. Reuses Decoder architecture from C149.
    """

    def __init__(self, noise_dim, hidden_dims, output_dim, activation='relu',
                 output_activation='sigmoid', init='he', dropout=None, rng=None):
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.decoder = Decoder(noise_dim, hidden_dims, output_dim,
                               activation=activation,
                               output_activation=output_activation,
                               init=init, dropout=dropout, rng=rng)

    def forward(self, z):
        """Generate data from noise."""
        return self.decoder.forward(z)

    def backward(self, grad):
        """Backward pass through generator."""
        return self.decoder.backward(grad)

    def get_trainable_layers(self):
        return self.decoder.get_trainable_layers()

    def train(self):
        self.decoder.train()

    def eval(self):
        self.decoder.eval()

    def count_params(self):
        total = 0
        for layer in self.get_trainable_layers():
            for param_tuple in layer.get_params():
                t = param_tuple[0]
                flat = t.flatten().data
                total += len(flat)
        return total


# ============================================================
# GAN (Vanilla)
# ============================================================

class GAN:
    """Vanilla GAN with minimax training.

    min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

    Features:
    - Label smoothing for discriminator targets
    - Separate optimizers for G and D
    - Training history tracking
    - Sample generation
    """

    def __init__(self, data_dim, noise_dim, gen_hidden, disc_hidden,
                 gen_activation='relu', gen_output='sigmoid',
                 disc_activation='leaky_relu', label_smoothing=0.0,
                 gen_lr=0.0002, disc_lr=0.0002, optimizer='adam',
                 dropout=None, rng=None):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.label_smoothing = label_smoothing
        self.rng = rng or random.Random(42)

        self.generator = Generator(
            noise_dim, gen_hidden, data_dim,
            activation=gen_activation, output_activation=gen_output,
            init='he', dropout=dropout, rng=self.rng
        )

        self.discriminator = Discriminator(
            data_dim, disc_hidden,
            activation=disc_activation, use_sigmoid=True,
            init='he', dropout=dropout, rng=self.rng
        )

        # Create optimizers
        if optimizer == 'adam':
            self.gen_optimizer = Adam(lr=gen_lr, beta1=0.5, beta2=0.999)
            self.disc_optimizer = Adam(lr=disc_lr, beta1=0.5, beta2=0.999)
        elif optimizer == 'rmsprop':
            self.gen_optimizer = RMSProp(lr=gen_lr)
            self.disc_optimizer = RMSProp(lr=disc_lr)
        else:
            self.gen_optimizer = SGD(lr=gen_lr)
            self.disc_optimizer = SGD(lr=disc_lr)

        self.history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}

    def _bce_loss(self, pred, target):
        """Binary cross-entropy loss."""
        return _tensor_bce(pred, target)

    def _bce_grad(self, pred, target):
        """Gradient of BCE w.r.t. pred."""
        return _tensor_bce_grad(pred, target)

    def train_discriminator(self, real_data):
        """Train discriminator on real and fake data. Returns d_loss."""
        self.discriminator.train()
        batch_size = real_data.shape[0] if len(real_data.shape) == 2 else 1

        # Real data targets (with optional label smoothing)
        real_target_val = 1.0 - self.label_smoothing
        real_targets = Tensor([[real_target_val]] * batch_size)

        # Forward on real data
        d_real = self.discriminator.forward(real_data)
        d_real_loss = self._bce_loss(d_real, real_targets)
        d_real_grad = self._bce_grad(d_real, real_targets)

        # Backward on real data
        self.discriminator.backward(d_real_grad)

        # Generate fake data
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        self.generator.eval()
        fake_data = self.generator.forward(z)
        self.generator.train()

        # Fake data targets
        fake_targets = Tensor([[0.0]] * batch_size)

        # Forward on fake data
        d_fake = self.discriminator.forward(fake_data)
        d_fake_loss = self._bce_loss(d_fake, fake_targets)
        d_fake_grad = self._bce_grad(d_fake, fake_targets)

        # Backward on fake data
        self.discriminator.backward(d_fake_grad)

        # Update discriminator
        self.disc_optimizer.step(self.discriminator.get_trainable_layers())

        d_loss = d_real_loss + d_fake_loss

        # Track D's accuracy on real/fake
        d_real_mean = sum(d_real.flatten().data) / len(d_real.flatten().data)
        d_fake_mean = sum(d_fake.flatten().data) / len(d_fake.flatten().data)

        return d_loss, d_real_mean, d_fake_mean

    def train_generator(self, batch_size):
        """Train generator to fool discriminator. Returns g_loss."""
        self.generator.train()
        self.discriminator.train()

        # Generate fake data
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        fake_data = self.generator.forward(z)

        # We want D(G(z)) -> 1 (fool discriminator)
        real_targets = Tensor([[1.0]] * batch_size)

        # Forward through discriminator (but only update generator)
        d_fake = self.discriminator.forward(fake_data)
        g_loss = self._bce_loss(d_fake, real_targets)
        g_grad = self._bce_grad(d_fake, real_targets)

        # Backward through discriminator (don't update it)
        d_grad_to_gen = self.discriminator.backward(g_grad)

        # Backward through generator
        self.generator.backward(d_grad_to_gen)

        # Update generator only
        self.gen_optimizer.step(self.generator.get_trainable_layers())

        return g_loss

    def train_step(self, real_data, n_disc=1):
        """One training step: train D n_disc times, then train G once."""
        batch_size = real_data.shape[0] if len(real_data.shape) == 2 else 1

        # Train discriminator
        d_loss = 0.0
        d_real_mean = 0.0
        d_fake_mean = 0.0
        for _ in range(n_disc):
            dl, dr, df = self.train_discriminator(real_data)
            d_loss += dl
            d_real_mean += dr
            d_fake_mean += df
        d_loss /= n_disc
        d_real_mean /= n_disc
        d_fake_mean /= n_disc

        # Train generator
        g_loss = self.train_generator(batch_size)

        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)
        self.history['d_real'].append(d_real_mean)
        self.history['d_fake'].append(d_fake_mean)

        return {'d_loss': d_loss, 'g_loss': g_loss,
                'd_real': d_real_mean, 'd_fake': d_fake_mean}

    def fit(self, data, epochs=100, batch_size=32, n_disc=1, verbose=False):
        """Train GAN on data.

        Args:
            data: Tensor (num_samples, data_dim)
            epochs: number of training epochs
            batch_size: mini-batch size
            n_disc: discriminator updates per generator update
            verbose: print progress
        """
        num_samples = data.shape[0] if len(data.shape) == 2 else 1
        indices = list(range(num_samples))

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                if len(batch_idx) < 2:
                    continue

                x_batch = Tensor([data.data[i] for i in batch_idx])
                metrics = self.train_step(x_batch, n_disc=n_disc)
                epoch_d_loss += metrics['d_loss']
                epoch_g_loss += metrics['g_loss']
                n_batches += 1

            if n_batches > 0 and verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - d_loss: {epoch_d_loss/n_batches:.4f} "
                      f"g_loss: {epoch_g_loss/n_batches:.4f}")

        return self.history

    def generate(self, n_samples=1):
        """Generate samples from the generator."""
        self.generator.eval()
        z = _random_noise(n_samples, self.noise_dim, self.rng)
        samples = self.generator.forward(z)
        self.generator.train()
        return samples

    def discriminate(self, data):
        """Get discriminator scores for data."""
        self.discriminator.eval()
        scores = self.discriminator.forward(data)
        self.discriminator.train()
        return scores


# ============================================================
# WGAN (Wasserstein GAN)
# ============================================================

class WGAN:
    """Wasserstein GAN with weight clipping.

    Uses Wasserstein distance instead of JS divergence.
    Critic (discriminator) outputs unbounded score, not probability.
    Loss: E[D(x)] - E[D(G(z))]  (maximize for D, minimize for G)
    Weight clipping on critic for Lipschitz constraint.
    """

    def __init__(self, data_dim, noise_dim, gen_hidden, critic_hidden,
                 gen_activation='relu', gen_output='tanh',
                 critic_activation='leaky_relu', clip_value=0.01,
                 gen_lr=0.00005, critic_lr=0.00005, n_critic=5,
                 dropout=None, rng=None):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.clip_value = clip_value
        self.n_critic = n_critic
        self.rng = rng or random.Random(42)

        self.generator = Generator(
            noise_dim, gen_hidden, data_dim,
            activation=gen_activation, output_activation=gen_output,
            init='he', dropout=dropout, rng=self.rng
        )

        # Critic has no sigmoid -- outputs raw score
        self.critic = Discriminator(
            data_dim, critic_hidden,
            activation=critic_activation, use_sigmoid=False,
            init='he', dropout=dropout, rng=self.rng
        )

        self.gen_optimizer = RMSProp(lr=gen_lr)
        self.critic_optimizer = RMSProp(lr=critic_lr)

        self.history = {'w_dist': [], 'g_loss': [], 'c_real': [], 'c_fake': []}

    def train_critic(self, real_data):
        """Train critic: maximize E[C(x)] - E[C(G(z))]."""
        self.critic.train()
        batch_size = real_data.shape[0] if len(real_data.shape) == 2 else 1

        # Score real data
        c_real = self.critic.forward(real_data)
        c_real_flat = c_real.flatten().data
        c_real_mean = sum(c_real_flat) / len(c_real_flat)

        # Gradient: maximize E[C(x)] -> grad = -1/N for each output
        # (We minimize -E[C(x)], so grad = -1/N)
        grad_real = Tensor([[-1.0 / batch_size]] * batch_size)
        self.critic.backward(grad_real)

        # Generate fake data
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        self.generator.eval()
        fake_data = self.generator.forward(z)
        self.generator.train()

        # Score fake data
        c_fake = self.critic.forward(fake_data)
        c_fake_flat = c_fake.flatten().data
        c_fake_mean = sum(c_fake_flat) / len(c_fake_flat)

        # Gradient: minimize E[C(G(z))] -> grad = 1/N for each output
        grad_fake = Tensor([[1.0 / batch_size]] * batch_size)
        self.critic.backward(grad_fake)

        # Update critic
        self.critic_optimizer.step(self.critic.get_trainable_layers())

        # Clip weights
        self.critic.clip_weights(self.clip_value)

        w_dist = c_real_mean - c_fake_mean
        return w_dist, c_real_mean, c_fake_mean

    def train_generator(self, batch_size):
        """Train generator: minimize -E[C(G(z))]."""
        self.generator.train()
        self.critic.train()

        z = _random_noise(batch_size, self.noise_dim, self.rng)
        fake_data = self.generator.forward(z)

        c_fake = self.critic.forward(fake_data)

        # Gradient: minimize -E[C(G(z))], so grad = -1/N
        grad = Tensor([[-1.0 / batch_size]] * batch_size)
        d_grad = self.critic.backward(grad)
        self.generator.backward(d_grad)

        self.gen_optimizer.step(self.generator.get_trainable_layers())

        c_fake_flat = c_fake.flatten().data
        g_loss = -sum(c_fake_flat) / len(c_fake_flat)
        return g_loss

    def train_step(self, real_data):
        """One training step: train critic n_critic times, then generator once."""
        batch_size = real_data.shape[0] if len(real_data.shape) == 2 else 1

        # Train critic multiple times
        w_dist = 0.0
        c_real = 0.0
        c_fake = 0.0
        for _ in range(self.n_critic):
            wd, cr, cf = self.train_critic(real_data)
            w_dist += wd
            c_real += cr
            c_fake += cf
        w_dist /= self.n_critic
        c_real /= self.n_critic
        c_fake /= self.n_critic

        # Train generator once
        g_loss = self.train_generator(batch_size)

        self.history['w_dist'].append(w_dist)
        self.history['g_loss'].append(g_loss)
        self.history['c_real'].append(c_real)
        self.history['c_fake'].append(c_fake)

        return {'w_dist': w_dist, 'g_loss': g_loss,
                'c_real': c_real, 'c_fake': c_fake}

    def fit(self, data, epochs=100, batch_size=32, verbose=False):
        """Train WGAN on data."""
        num_samples = data.shape[0] if len(data.shape) == 2 else 1
        indices = list(range(num_samples))

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            epoch_w = 0.0
            epoch_g = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                if len(batch_idx) < 2:
                    continue

                x_batch = Tensor([data.data[i] for i in batch_idx])
                metrics = self.train_step(x_batch)
                epoch_w += metrics['w_dist']
                epoch_g += metrics['g_loss']
                n_batches += 1

            if n_batches > 0 and verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - w_dist: {epoch_w/n_batches:.4f} "
                      f"g_loss: {epoch_g/n_batches:.4f}")

        return self.history

    def generate(self, n_samples=1):
        """Generate samples."""
        self.generator.eval()
        z = _random_noise(n_samples, self.noise_dim, self.rng)
        samples = self.generator.forward(z)
        self.generator.train()
        return samples


# ============================================================
# LSGAN (Least Squares GAN)
# ============================================================

class LSGAN:
    """Least Squares GAN.

    Uses MSE loss instead of cross-entropy:
    D loss: 0.5 * E[(D(x) - 1)^2] + 0.5 * E[D(G(z))^2]
    G loss: 0.5 * E[(D(G(z)) - 1)^2]

    More stable gradients than vanilla GAN.
    """

    def __init__(self, data_dim, noise_dim, gen_hidden, disc_hidden,
                 gen_activation='relu', gen_output='sigmoid',
                 disc_activation='leaky_relu',
                 gen_lr=0.0002, disc_lr=0.0002,
                 dropout=None, rng=None):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.rng = rng or random.Random(42)

        self.generator = Generator(
            noise_dim, gen_hidden, data_dim,
            activation=gen_activation, output_activation=gen_output,
            init='he', dropout=dropout, rng=self.rng
        )

        # No sigmoid -- LSGAN uses raw output
        self.discriminator = Discriminator(
            data_dim, disc_hidden,
            activation=disc_activation, use_sigmoid=False,
            init='he', dropout=dropout, rng=self.rng
        )

        self.gen_optimizer = Adam(lr=gen_lr, beta1=0.5, beta2=0.999)
        self.disc_optimizer = Adam(lr=disc_lr, beta1=0.5, beta2=0.999)

        self.history = {'d_loss': [], 'g_loss': []}

    def _mse_loss_and_grad(self, pred, target_val):
        """Compute 0.5 * E[(pred - target)^2] and its gradient."""
        flat = pred.flatten().data
        n = len(flat)
        loss = 0.0
        grad = []
        for v in flat:
            diff = v - target_val
            loss += diff * diff
            grad.append(diff / n)  # d/dv of 0.5*(v-t)^2 = (v-t), avg over batch
        loss = 0.5 * loss / n

        if len(pred.shape) == 2:
            rows, cols = pred.shape
            grad_data = [grad[r * cols:(r + 1) * cols] for r in range(rows)]
            return loss, Tensor(grad_data)
        return loss, Tensor(grad)

    def train_discriminator(self, real_data):
        """Train discriminator with LS loss."""
        self.discriminator.train()
        batch_size = real_data.shape[0] if len(real_data.shape) == 2 else 1

        # Real data: D(x) -> 1
        d_real = self.discriminator.forward(real_data)
        d_real_loss, d_real_grad = self._mse_loss_and_grad(d_real, 1.0)
        self.discriminator.backward(d_real_grad)

        # Fake data: D(G(z)) -> 0
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        self.generator.eval()
        fake_data = self.generator.forward(z)
        self.generator.train()

        d_fake = self.discriminator.forward(fake_data)
        d_fake_loss, d_fake_grad = self._mse_loss_and_grad(d_fake, 0.0)
        self.discriminator.backward(d_fake_grad)

        self.disc_optimizer.step(self.discriminator.get_trainable_layers())

        return d_real_loss + d_fake_loss

    def train_generator(self, batch_size):
        """Train generator with LS loss: D(G(z)) -> 1."""
        self.generator.train()
        self.discriminator.train()

        z = _random_noise(batch_size, self.noise_dim, self.rng)
        fake_data = self.generator.forward(z)

        d_fake = self.discriminator.forward(fake_data)
        g_loss, g_grad = self._mse_loss_and_grad(d_fake, 1.0)

        d_grad = self.discriminator.backward(g_grad)
        self.generator.backward(d_grad)

        self.gen_optimizer.step(self.generator.get_trainable_layers())
        return g_loss

    def train_step(self, real_data, n_disc=1):
        """One training step."""
        batch_size = real_data.shape[0] if len(real_data.shape) == 2 else 1

        d_loss = 0.0
        for _ in range(n_disc):
            d_loss += self.train_discriminator(real_data)
        d_loss /= n_disc

        g_loss = self.train_generator(batch_size)

        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def fit(self, data, epochs=100, batch_size=32, n_disc=1, verbose=False):
        """Train LSGAN."""
        num_samples = data.shape[0] if len(data.shape) == 2 else 1
        indices = list(range(num_samples))

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            epoch_d = 0.0
            epoch_g = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                if len(batch_idx) < 2:
                    continue

                x_batch = Tensor([data.data[i] for i in batch_idx])
                m = self.train_step(x_batch, n_disc=n_disc)
                epoch_d += m['d_loss']
                epoch_g += m['g_loss']
                n_batches += 1

            if n_batches > 0 and verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - d_loss: {epoch_d/n_batches:.4f} "
                      f"g_loss: {epoch_g/n_batches:.4f}")

        return self.history

    def generate(self, n_samples=1):
        self.generator.eval()
        z = _random_noise(n_samples, self.noise_dim, self.rng)
        samples = self.generator.forward(z)
        self.generator.train()
        return samples


# ============================================================
# Conditional GAN (cGAN)
# ============================================================

class ConditionalGAN:
    """Conditional GAN: condition on class labels.

    Generator: (z, y) -> x  (noise + label -> data)
    Discriminator: (x, y) -> [0,1]  (data + label -> real/fake)

    Labels are one-hot encoded and concatenated to inputs.
    """

    def __init__(self, data_dim, noise_dim, num_classes, gen_hidden, disc_hidden,
                 gen_activation='relu', gen_output='sigmoid',
                 disc_activation='leaky_relu', label_smoothing=0.0,
                 gen_lr=0.0002, disc_lr=0.0002,
                 dropout=None, rng=None):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.rng = rng or random.Random(42)

        # Generator input: noise + one-hot label
        gen_input_dim = noise_dim + num_classes
        self.generator = Generator(
            gen_input_dim, gen_hidden, data_dim,
            activation=gen_activation, output_activation=gen_output,
            init='he', dropout=dropout, rng=self.rng
        )

        # Discriminator input: data + one-hot label
        disc_input_dim = data_dim + num_classes
        self.discriminator = Discriminator(
            disc_input_dim, disc_hidden,
            activation=disc_activation, use_sigmoid=True,
            init='he', dropout=dropout, rng=self.rng
        )

        self.gen_optimizer = Adam(lr=gen_lr, beta1=0.5, beta2=0.999)
        self.disc_optimizer = Adam(lr=disc_lr, beta1=0.5, beta2=0.999)

        self.history = {'d_loss': [], 'g_loss': []}

    def _make_gen_input(self, z, labels):
        """Concatenate noise with one-hot labels."""
        one_hot = _one_hot(labels, self.num_classes)
        return _concat_tensors(z, one_hot)

    def _make_disc_input(self, data, labels):
        """Concatenate data with one-hot labels."""
        one_hot = _one_hot(labels, self.num_classes)
        return _concat_tensors(data, one_hot)

    def train_discriminator(self, real_data, labels):
        """Train discriminator on real and fake data with labels."""
        self.discriminator.train()
        batch_size = real_data.shape[0]

        # Real data
        real_target_val = 1.0 - self.label_smoothing
        real_targets = Tensor([[real_target_val]] * batch_size)
        disc_real_input = self._make_disc_input(real_data, labels)
        d_real = self.discriminator.forward(disc_real_input)
        d_real_loss = _tensor_bce(d_real, real_targets)
        d_real_grad = _tensor_bce_grad(d_real, real_targets)
        self.discriminator.backward(d_real_grad)

        # Fake data
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        gen_input = self._make_gen_input(z, labels)
        self.generator.eval()
        fake_data = self.generator.forward(gen_input)
        self.generator.train()

        fake_targets = Tensor([[0.0]] * batch_size)
        disc_fake_input = self._make_disc_input(fake_data, labels)
        d_fake = self.discriminator.forward(disc_fake_input)
        d_fake_loss = _tensor_bce(d_fake, fake_targets)
        d_fake_grad = _tensor_bce_grad(d_fake, fake_targets)
        self.discriminator.backward(d_fake_grad)

        self.disc_optimizer.step(self.discriminator.get_trainable_layers())

        return d_real_loss + d_fake_loss

    def train_generator(self, batch_size, labels):
        """Train generator to fool discriminator, conditioned on labels."""
        self.generator.train()
        self.discriminator.train()

        z = _random_noise(batch_size, self.noise_dim, self.rng)
        gen_input = self._make_gen_input(z, labels)
        fake_data = self.generator.forward(gen_input)

        real_targets = Tensor([[1.0]] * batch_size)
        disc_input = self._make_disc_input(fake_data, labels)
        d_fake = self.discriminator.forward(disc_input)
        g_loss = _tensor_bce(d_fake, real_targets)
        g_grad = _tensor_bce_grad(d_fake, real_targets)

        # Backward through D to get gradient for G's output
        d_grad_full = self.discriminator.backward(g_grad)

        # d_grad_full has shape (batch, data_dim + num_classes)
        # We only need the first data_dim columns for the generator
        if len(d_grad_full.shape) == 2:
            gen_grad = Tensor([row[:self.data_dim] for row in d_grad_full.data])
        else:
            gen_grad = Tensor(d_grad_full.data[:self.data_dim])

        self.generator.backward(gen_grad)
        self.gen_optimizer.step(self.generator.get_trainable_layers())

        return g_loss

    def train_step(self, real_data, labels, n_disc=1):
        """One training step."""
        batch_size = real_data.shape[0]

        d_loss = 0.0
        for _ in range(n_disc):
            d_loss += self.train_discriminator(real_data, labels)
        d_loss /= n_disc

        g_loss = self.train_generator(batch_size, labels)

        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(g_loss)

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def fit(self, data, labels, epochs=100, batch_size=32, n_disc=1, verbose=False):
        """Train conditional GAN."""
        num_samples = data.shape[0]
        indices = list(range(num_samples))

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            epoch_d = 0.0
            epoch_g = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                if len(batch_idx) < 2:
                    continue

                x_batch = Tensor([data.data[i] for i in batch_idx])
                y_batch = [labels[i] for i in batch_idx]
                m = self.train_step(x_batch, y_batch, n_disc=n_disc)
                epoch_d += m['d_loss']
                epoch_g += m['g_loss']
                n_batches += 1

            if n_batches > 0 and verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - d_loss: {epoch_d/n_batches:.4f} "
                      f"g_loss: {epoch_g/n_batches:.4f}")

        return self.history

    def generate(self, labels, n_per_class=1):
        """Generate samples conditioned on labels."""
        self.generator.eval()
        all_labels = []
        for label in labels:
            all_labels.extend([label] * n_per_class)
        n = len(all_labels)
        z = _random_noise(n, self.noise_dim, self.rng)
        gen_input = self._make_gen_input(z, all_labels)
        samples = self.generator.forward(gen_input)
        self.generator.train()
        return samples, all_labels


# ============================================================
# InfoGAN
# ============================================================

class InfoGAN:
    """Information-maximizing GAN.

    Adds a Q-network (shares layers with D) that predicts latent codes.
    Maximizes mutual information I(c; G(z, c)) via auxiliary distribution Q(c|x).

    Latent input: z (noise) + c_cat (categorical codes) + c_cont (continuous codes)
    Q-network outputs: predicted categorical probs + predicted continuous params (mu, sigma)
    """

    def __init__(self, data_dim, noise_dim, n_cat_codes, cat_dim,
                 n_cont_codes, gen_hidden, disc_hidden,
                 gen_activation='relu', gen_output='sigmoid',
                 disc_activation='leaky_relu',
                 info_weight=1.0, gen_lr=0.0002, disc_lr=0.0002,
                 dropout=None, rng=None):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.n_cat_codes = n_cat_codes    # number of categorical code variables
        self.cat_dim = cat_dim            # dimension of each categorical code
        self.n_cont_codes = n_cont_codes  # number of continuous code variables
        self.info_weight = info_weight
        self.rng = rng or random.Random(42)

        total_cat = n_cat_codes * cat_dim
        total_code_dim = total_cat + n_cont_codes

        # Generator input: noise + codes
        gen_input_dim = noise_dim + total_code_dim
        self.generator = Generator(
            gen_input_dim, gen_hidden, data_dim,
            activation=gen_activation, output_activation=gen_output,
            init='he', dropout=dropout, rng=self.rng
        )

        # Discriminator
        self.discriminator = Discriminator(
            data_dim, disc_hidden,
            activation=disc_activation, use_sigmoid=True,
            init='he', dropout=dropout, rng=self.rng
        )

        # Q-network: predicts codes from D's intermediate representation
        # For simplicity, Q takes data directly (like D but with different output)
        q_hidden = disc_hidden
        self.q_network = Sequential()
        prev_dim = data_dim
        for h_dim in q_hidden:
            self.q_network.add(Dense(prev_dim, h_dim, init='he', rng=self.rng))
            self.q_network.add(Activation(disc_activation))
            prev_dim = h_dim

        # Q outputs: categorical logits + continuous mu + continuous sigma
        q_output_dim = total_cat + n_cont_codes * 2  # mu and sigma for each cont code
        self.q_output_layer = Dense(prev_dim, q_output_dim, init='he', rng=self.rng)

        self.gen_optimizer = Adam(lr=gen_lr, beta1=0.5, beta2=0.999)
        self.disc_optimizer = Adam(lr=disc_lr, beta1=0.5, beta2=0.999)
        self.q_optimizer = Adam(lr=gen_lr, beta1=0.5, beta2=0.999)

        self.history = {'d_loss': [], 'g_loss': [], 'info_loss': []}

    def _sample_codes(self, batch_size):
        """Sample latent codes: categorical (uniform) + continuous (N(0,1))."""
        total_cat = self.n_cat_codes * self.cat_dim
        codes_data = []
        cat_indices = []  # for computing categorical loss

        for _ in range(batch_size):
            row = []
            sample_cats = []
            for _ in range(self.n_cat_codes):
                idx = self.rng.randint(0, self.cat_dim - 1)
                sample_cats.append(idx)
                one_hot = [0.0] * self.cat_dim
                one_hot[idx] = 1.0
                row.extend(one_hot)
            for _ in range(self.n_cont_codes):
                row.append(self.rng.gauss(0, 1))
            codes_data.append(row)
            cat_indices.append(sample_cats)

        return Tensor(codes_data), cat_indices

    def _make_gen_input(self, z, codes):
        """Concatenate noise with codes."""
        return _concat_tensors(z, codes)

    def _q_forward(self, data):
        """Forward through Q-network."""
        h = self.q_network.forward(data)
        out = self.q_output_layer.forward(h)
        return out

    def _q_backward(self, grad):
        """Backward through Q-network."""
        g = self.q_output_layer.backward(grad)
        g = self.q_network.backward(g)
        return g

    def _q_trainable_layers(self):
        layers = self.q_network.get_trainable_layers()
        layers.append(self.q_output_layer)
        return layers

    def _compute_info_loss(self, q_out, cat_indices, codes):
        """Compute mutual information loss.

        Categorical: cross-entropy between Q(c|x) and true c
        Continuous: Gaussian NLL between Q params and true c
        """
        total_cat = self.n_cat_codes * self.cat_dim
        batch_size = q_out.shape[0]

        info_loss = 0.0
        grad_data = [[0.0] * q_out.shape[1] for _ in range(batch_size)]

        # Categorical loss: cross-entropy
        for b in range(batch_size):
            for k in range(self.n_cat_codes):
                start = k * self.cat_dim
                # Softmax over logits
                logits = [q_out.data[b][start + j] for j in range(self.cat_dim)]
                max_l = max(logits)
                exp_l = [math.exp(l - max_l) for l in logits]
                sum_exp = sum(exp_l)
                probs = [e / sum_exp for e in exp_l]

                true_idx = cat_indices[b][k]
                info_loss -= _safe_log(probs[true_idx]) / batch_size

                # Gradient of cross-entropy w.r.t. logits
                for j in range(self.cat_dim):
                    target = 1.0 if j == true_idx else 0.0
                    grad_data[b][start + j] = (probs[j] - target) / batch_size

        # Continuous loss: treat as Gaussian NLL
        for b in range(batch_size):
            for k in range(self.n_cont_codes):
                mu_idx = total_cat + k * 2
                sigma_idx = total_cat + k * 2 + 1
                mu_pred = q_out.data[b][mu_idx]
                sigma_raw = q_out.data[b][sigma_idx]
                sigma = math.exp(sigma_raw)  # ensure positive

                true_val = codes.data[b][total_cat + k]
                diff = true_val - mu_pred
                nll = 0.5 * (diff * diff) / (sigma * sigma) + sigma_raw + 0.5 * math.log(2 * math.pi)
                info_loss += nll / batch_size

                # Gradients
                grad_data[b][mu_idx] = -(diff / (sigma * sigma)) / batch_size
                grad_data[b][sigma_idx] = (1.0 - (diff * diff) / (sigma * sigma)) / batch_size

        return info_loss, Tensor(grad_data)

    def train_step(self, real_data):
        """One training step for InfoGAN."""
        batch_size = real_data.shape[0]

        # === Train Discriminator ===
        self.discriminator.train()

        # Real
        real_targets = Tensor([[1.0]] * batch_size)
        d_real = self.discriminator.forward(real_data)
        d_real_loss = _tensor_bce(d_real, real_targets)
        d_real_grad = _tensor_bce_grad(d_real, real_targets)
        self.discriminator.backward(d_real_grad)

        # Fake
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        codes, cat_indices = self._sample_codes(batch_size)
        gen_input = self._make_gen_input(z, codes)
        self.generator.eval()
        fake_data = self.generator.forward(gen_input)
        self.generator.train()

        fake_targets = Tensor([[0.0]] * batch_size)
        d_fake = self.discriminator.forward(fake_data)
        d_fake_loss = _tensor_bce(d_fake, fake_targets)
        d_fake_grad = _tensor_bce_grad(d_fake, fake_targets)
        self.discriminator.backward(d_fake_grad)

        self.disc_optimizer.step(self.discriminator.get_trainable_layers())
        d_loss = d_real_loss + d_fake_loss

        # === Train Generator + Q ===
        self.generator.train()
        self.discriminator.train()

        z = _random_noise(batch_size, self.noise_dim, self.rng)
        codes, cat_indices = self._sample_codes(batch_size)
        gen_input = self._make_gen_input(z, codes)
        fake_data = self.generator.forward(gen_input)

        # GAN loss
        real_targets = Tensor([[1.0]] * batch_size)
        d_fake = self.discriminator.forward(fake_data)
        g_loss = _tensor_bce(d_fake, real_targets)
        g_grad = _tensor_bce_grad(d_fake, real_targets)
        d_grad = self.discriminator.backward(g_grad)

        # Info loss
        q_out = self._q_forward(fake_data)
        info_loss, info_grad = self._compute_info_loss(q_out, cat_indices, codes)

        # Scale info grad
        info_grad_scaled = Tensor([[v * self.info_weight for v in row]
                                   for row in info_grad.data])
        self._q_backward(info_grad_scaled)

        # Combine D grad and info grad for generator update
        # d_grad goes back to generator
        self.generator.backward(d_grad)

        self.gen_optimizer.step(self.generator.get_trainable_layers())
        self.q_optimizer.step(self._q_trainable_layers())

        total_g = g_loss + self.info_weight * info_loss

        self.history['d_loss'].append(d_loss)
        self.history['g_loss'].append(total_g)
        self.history['info_loss'].append(info_loss)

        return {'d_loss': d_loss, 'g_loss': total_g, 'info_loss': info_loss}

    def fit(self, data, epochs=100, batch_size=32, verbose=False):
        """Train InfoGAN."""
        num_samples = data.shape[0]
        indices = list(range(num_samples))

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            epoch_d = 0.0
            epoch_g = 0.0
            epoch_i = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                if len(batch_idx) < 2:
                    continue

                x_batch = Tensor([data.data[i] for i in batch_idx])
                m = self.train_step(x_batch)
                epoch_d += m['d_loss']
                epoch_g += m['g_loss']
                epoch_i += m['info_loss']
                n_batches += 1

            if n_batches > 0 and verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - d: {epoch_d/n_batches:.4f} "
                      f"g: {epoch_g/n_batches:.4f} info: {epoch_i/n_batches:.4f}")

        return self.history

    def generate(self, n_samples=1, cat_codes=None, cont_codes=None):
        """Generate with specific codes.

        Args:
            cat_codes: list of lists of category indices, or None for random
            cont_codes: list of lists of continuous values, or None for random
        """
        self.generator.eval()
        total_cat = self.n_cat_codes * self.cat_dim

        codes_data = []
        for i in range(n_samples):
            row = []
            # Categorical
            for k in range(self.n_cat_codes):
                if cat_codes is not None and i < len(cat_codes):
                    idx = cat_codes[i][k] if k < len(cat_codes[i]) else 0
                else:
                    idx = self.rng.randint(0, self.cat_dim - 1)
                one_hot = [0.0] * self.cat_dim
                one_hot[idx] = 1.0
                row.extend(one_hot)
            # Continuous
            for k in range(self.n_cont_codes):
                if cont_codes is not None and i < len(cont_codes):
                    row.append(cont_codes[i][k] if k < len(cont_codes[i]) else 0.0)
                else:
                    row.append(self.rng.gauss(0, 1))
            codes_data.append(row)

        codes = Tensor(codes_data)
        z = _random_noise(n_samples, self.noise_dim, self.rng)
        gen_input = self._make_gen_input(z, codes)
        samples = self.generator.forward(gen_input)
        self.generator.train()
        return samples


# ============================================================
# BiGAN (Bidirectional GAN)
# ============================================================

class BiGAN:
    """Bidirectional GAN with encoder.

    Adds an encoder E that maps x -> z (inference).
    Discriminator sees (x, E(x)) for real and (G(z), z) for fake.
    After training, encoder provides a learned representation.
    """

    def __init__(self, data_dim, noise_dim, gen_hidden, enc_hidden, disc_hidden,
                 gen_activation='relu', gen_output='sigmoid',
                 enc_activation='relu', disc_activation='leaky_relu',
                 gen_lr=0.0002, disc_lr=0.0002, enc_lr=0.0002,
                 dropout=None, rng=None):
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.rng = rng or random.Random(42)

        self.generator = Generator(
            noise_dim, gen_hidden, data_dim,
            activation=gen_activation, output_activation=gen_output,
            init='he', dropout=dropout, rng=self.rng
        )

        self.encoder = Encoder(
            data_dim, enc_hidden, noise_dim,
            activation=enc_activation, init='he', dropout=dropout, rng=self.rng
        )

        # Discriminator takes (data, latent) pairs
        disc_input_dim = data_dim + noise_dim
        self.discriminator = Discriminator(
            disc_input_dim, disc_hidden,
            activation=disc_activation, use_sigmoid=True,
            init='he', dropout=dropout, rng=self.rng
        )

        self.gen_optimizer = Adam(lr=gen_lr, beta1=0.5, beta2=0.999)
        self.enc_optimizer = Adam(lr=enc_lr, beta1=0.5, beta2=0.999)
        self.disc_optimizer = Adam(lr=disc_lr, beta1=0.5, beta2=0.999)

        self.history = {'d_loss': [], 'ge_loss': []}

    def train_discriminator(self, real_data):
        """Train D on (x, E(x))=real, (G(z), z)=fake."""
        self.discriminator.train()
        batch_size = real_data.shape[0]

        # Real pair: (x, E(x))
        self.encoder.eval()
        encoded = self.encoder.forward(real_data)
        self.encoder.train()
        real_pair = _concat_tensors(real_data, encoded)

        real_targets = Tensor([[1.0]] * batch_size)
        d_real = self.discriminator.forward(real_pair)
        d_real_loss = _tensor_bce(d_real, real_targets)
        d_real_grad = _tensor_bce_grad(d_real, real_targets)
        self.discriminator.backward(d_real_grad)

        # Fake pair: (G(z), z)
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        self.generator.eval()
        fake_data = self.generator.forward(z)
        self.generator.train()
        fake_pair = _concat_tensors(fake_data, z)

        fake_targets = Tensor([[0.0]] * batch_size)
        d_fake = self.discriminator.forward(fake_pair)
        d_fake_loss = _tensor_bce(d_fake, fake_targets)
        d_fake_grad = _tensor_bce_grad(d_fake, fake_targets)
        self.discriminator.backward(d_fake_grad)

        self.disc_optimizer.step(self.discriminator.get_trainable_layers())

        return d_real_loss + d_fake_loss

    def train_generator_encoder(self, real_data):
        """Train G and E together (adversarial against D).

        G wants (G(z), z) to be classified as real.
        E wants (x, E(x)) to be classified as fake.
        """
        batch_size = real_data.shape[0]
        self.generator.train()
        self.encoder.train()
        self.discriminator.train()

        # Generator loss: D(G(z), z) -> 1
        z = _random_noise(batch_size, self.noise_dim, self.rng)
        fake_data = self.generator.forward(z)
        fake_pair = _concat_tensors(fake_data, z)

        real_targets = Tensor([[1.0]] * batch_size)
        d_fake = self.discriminator.forward(fake_pair)
        g_loss = _tensor_bce(d_fake, real_targets)
        g_grad = _tensor_bce_grad(d_fake, real_targets)

        d_grad_full = self.discriminator.backward(g_grad)
        # Only first data_dim columns are for generator
        if len(d_grad_full.shape) == 2:
            gen_grad = Tensor([row[:self.data_dim] for row in d_grad_full.data])
        else:
            gen_grad = Tensor(d_grad_full.data[:self.data_dim])
        self.generator.backward(gen_grad)

        # Encoder loss: D(x, E(x)) -> 0
        encoded = self.encoder.forward(real_data)
        real_pair = _concat_tensors(real_data, encoded)

        fake_targets = Tensor([[0.0]] * batch_size)
        d_real = self.discriminator.forward(real_pair)
        e_loss = _tensor_bce(d_real, fake_targets)
        e_grad = _tensor_bce_grad(d_real, fake_targets)

        d_grad_full = self.discriminator.backward(e_grad)
        # Last noise_dim columns are for encoder
        if len(d_grad_full.shape) == 2:
            enc_grad = Tensor([row[self.data_dim:] for row in d_grad_full.data])
        else:
            enc_grad = Tensor(d_grad_full.data[self.data_dim:])
        self.encoder.backward(enc_grad)

        self.gen_optimizer.step(self.generator.get_trainable_layers())
        self.enc_optimizer.step(self.encoder.get_trainable_layers())

        return g_loss + e_loss

    def train_step(self, real_data, n_disc=1):
        """One training step."""
        d_loss = 0.0
        for _ in range(n_disc):
            d_loss += self.train_discriminator(real_data)
        d_loss /= n_disc

        ge_loss = self.train_generator_encoder(real_data)

        self.history['d_loss'].append(d_loss)
        self.history['ge_loss'].append(ge_loss)

        return {'d_loss': d_loss, 'ge_loss': ge_loss}

    def fit(self, data, epochs=100, batch_size=32, n_disc=1, verbose=False):
        """Train BiGAN."""
        num_samples = data.shape[0]
        indices = list(range(num_samples))

        for epoch in range(epochs):
            self.rng.shuffle(indices)
            epoch_d = 0.0
            epoch_ge = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]
                if len(batch_idx) < 2:
                    continue

                x_batch = Tensor([data.data[i] for i in batch_idx])
                m = self.train_step(x_batch, n_disc=n_disc)
                epoch_d += m['d_loss']
                epoch_ge += m['ge_loss']
                n_batches += 1

            if n_batches > 0 and verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - d_loss: {epoch_d/n_batches:.4f} "
                      f"ge_loss: {epoch_ge/n_batches:.4f}")

        return self.history

    def generate(self, n_samples=1):
        """Generate samples."""
        self.generator.eval()
        z = _random_noise(n_samples, self.noise_dim, self.rng)
        samples = self.generator.forward(z)
        self.generator.train()
        return samples

    def encode(self, data):
        """Encode data to latent space."""
        self.encoder.eval()
        encoded = self.encoder.forward(data)
        self.encoder.train()
        return encoded


# ============================================================
# GAN Evaluation Metrics
# ============================================================

class GANMetrics:
    """Evaluation metrics for GANs.

    - Distribution statistics comparison
    - Mode coverage estimation
    - Discriminator score analysis
    """

    @staticmethod
    def mean_std(tensor):
        """Compute mean and std of each feature."""
        if len(tensor.shape) == 1:
            mean = sum(tensor.data) / len(tensor.data)
            var = sum((v - mean) ** 2 for v in tensor.data) / len(tensor.data)
            return [mean], [math.sqrt(var + 1e-8)]

        rows, cols = tensor.shape
        means = [0.0] * cols
        for i in range(rows):
            for j in range(cols):
                means[j] += tensor.data[i][j]
        means = [m / rows for m in means]

        stds = [0.0] * cols
        for i in range(rows):
            for j in range(cols):
                stds[j] += (tensor.data[i][j] - means[j]) ** 2
        stds = [math.sqrt(s / rows + 1e-8) for s in stds]

        return means, stds

    @staticmethod
    def distribution_distance(real, fake):
        """Simple distribution distance: sum of squared differences in mean and std."""
        r_mean, r_std = GANMetrics.mean_std(real)
        f_mean, f_std = GANMetrics.mean_std(fake)

        mean_dist = sum((rm - fm) ** 2 for rm, fm in zip(r_mean, f_mean))
        std_dist = sum((rs - fs) ** 2 for rs, fs in zip(r_std, f_std))

        return math.sqrt(mean_dist + std_dist)

    @staticmethod
    def mode_coverage(real, fake, n_bins=10):
        """Estimate mode coverage by comparing histograms per feature.

        Returns fraction of real bins covered by fake samples.
        """
        if len(real.shape) == 1 or len(fake.shape) == 1:
            return 1.0

        cols = real.shape[1]
        total_covered = 0
        total_bins = 0

        for j in range(cols):
            r_col = [real.data[i][j] for i in range(real.shape[0])]
            f_col = [fake.data[i][j] for i in range(fake.shape[0])]

            lo = min(min(r_col), min(f_col))
            hi = max(max(r_col), max(f_col))
            if hi == lo:
                total_covered += n_bins
                total_bins += n_bins
                continue

            bin_width = (hi - lo) / n_bins

            # Count real bins that have samples
            real_bins = set()
            for v in r_col:
                b = min(int((v - lo) / bin_width), n_bins - 1)
                real_bins.add(b)

            # Count how many real bins are covered by fake
            fake_bins = set()
            for v in f_col:
                b = min(int((v - lo) / bin_width), n_bins - 1)
                fake_bins.add(b)

            covered = len(real_bins & fake_bins)
            total_covered += covered
            total_bins += len(real_bins)

        return total_covered / max(total_bins, 1)

    @staticmethod
    def discriminator_score_stats(discriminator, real, fake):
        """Compute D's scores on real and fake data."""
        discriminator.eval()
        d_real = discriminator.forward(real)
        d_fake = discriminator.forward(fake)
        discriminator.train()

        r_scores = d_real.flatten().data
        f_scores = d_fake.flatten().data

        r_mean = sum(r_scores) / len(r_scores)
        f_mean = sum(f_scores) / len(f_scores)

        return {'real_mean': r_mean, 'fake_mean': f_mean,
                'real_min': min(r_scores), 'real_max': max(r_scores),
                'fake_min': min(f_scores), 'fake_max': max(f_scores)}


# ============================================================
# Data generators for GAN testing
# ============================================================

def make_gaussian_mixture(n_samples=200, n_modes=4, std=0.1, seed=42):
    """Generate data from a Gaussian mixture model (2D).

    Modes arranged in a circle for clear separation.
    """
    rng = random.Random(seed)
    data = []
    labels = []

    samples_per_mode = n_samples // n_modes
    for mode in range(n_modes):
        angle = 2.0 * math.pi * mode / n_modes
        cx = math.cos(angle) * 0.5 + 0.5  # center in [0, 1]
        cy = math.sin(angle) * 0.5 + 0.5
        for _ in range(samples_per_mode):
            x = cx + rng.gauss(0, std)
            y = cy + rng.gauss(0, std)
            data.append([x, y])
            labels.append(mode)

    # Fill remaining
    for i in range(n_samples - len(data)):
        mode = i % n_modes
        angle = 2.0 * math.pi * mode / n_modes
        cx = math.cos(angle) * 0.5 + 0.5
        cy = math.sin(angle) * 0.5 + 0.5
        data.append([cx + rng.gauss(0, std), cy + rng.gauss(0, std)])
        labels.append(mode)

    return Tensor(data), labels


def make_line_data(n_samples=200, noise=0.05, seed=42):
    """Generate data along a line y = x with noise."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        x = rng.random()
        y = x + rng.gauss(0, noise)
        data.append([x, y])
    return Tensor(data)


def make_grid_data(n_samples=200, grid_size=3, std=0.05, seed=42):
    """Generate data from a grid of Gaussians."""
    rng = random.Random(seed)
    data = []
    samples_per_point = n_samples // (grid_size * grid_size)

    for i in range(grid_size):
        for j in range(grid_size):
            cx = (i + 0.5) / grid_size
            cy = (j + 0.5) / grid_size
            for _ in range(samples_per_point):
                data.append([cx + rng.gauss(0, std), cy + rng.gauss(0, std)])

    # Fill remaining
    while len(data) < n_samples:
        i = rng.randint(0, grid_size - 1)
        j = rng.randint(0, grid_size - 1)
        cx = (i + 0.5) / grid_size
        cy = (j + 0.5) / grid_size
        data.append([cx + rng.gauss(0, std), cy + rng.gauss(0, std)])

    return Tensor(data[:n_samples])


def make_ring_data(n_samples=200, noise=0.05, seed=42):
    """Generate data on a ring."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        angle = rng.uniform(0, 2 * math.pi)
        r = 0.5 + rng.gauss(0, noise)
        data.append([0.5 + r * math.cos(angle), 0.5 + r * math.sin(angle)])
    return Tensor(data)
