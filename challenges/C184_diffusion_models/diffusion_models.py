"""
C184: Diffusion Models
======================
Denoising Diffusion Probabilistic Models (DDPM) and score-based generative
models, built from scratch with NumPy.

Core idea: gradually add noise to data over T timesteps (forward process),
then learn to reverse the process (reverse/denoising process). At inference,
start from pure noise and iteratively denoise to generate samples.

Components:
- Noise schedules: linear, cosine, quadratic, sigmoid
- Forward diffusion process (q): closed-form noising at arbitrary timestep
- Reverse process: learned denoising with simple neural networks
- Training: predict noise (epsilon-prediction) or x0 (x0-prediction)
- Sampling: DDPM, DDIM (deterministic), ancestral
- Conditional generation with classifier-free guidance
- Diagnostics: FID proxy, noise prediction quality, schedule analysis

Built with NumPy only. No external ML libraries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable


# ============================================================
# Noise Schedules
# ============================================================

class NoiseSchedule:
    """Manages the variance schedule for diffusion."""

    def __init__(self, num_timesteps: int, schedule_type: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = self._make_schedule()
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.alpha_cumprod_prev = np.concatenate([[1.0], self.alpha_cumprod[:-1]])

        # Precompute useful quantities
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = 1.0 / np.sqrt(self.alphas)

        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_cumprod_{t-1}) / (1 - alpha_cumprod_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )
        # Clip first element to avoid log(0)
        self.posterior_log_variance = np.log(
            np.clip(self.posterior_variance, a_min=1e-20, a_max=None)
        )

        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_cumprod_prev) * np.sqrt(self.alphas) / (1.0 - self.alpha_cumprod)
        )

    def _make_schedule(self) -> np.ndarray:
        T = self.T
        if self.schedule_type == "linear":
            return np.linspace(self.beta_start, self.beta_end, T)
        elif self.schedule_type == "cosine":
            # Cosine schedule from Nichol & Dhariwal
            steps = np.arange(T + 1, dtype=np.float64)
            s = 0.008
            f = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            return np.clip(betas, 0, 0.999)
        elif self.schedule_type == "quadratic":
            return np.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, T) ** 2
        elif self.schedule_type == "sigmoid":
            x = np.linspace(-6, 6, T)
            betas = 1.0 / (1.0 + np.exp(-x))
            return betas * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def snr(self, t: int) -> float:
        """Signal-to-noise ratio at timestep t."""
        return self.alpha_cumprod[t] / (1.0 - self.alpha_cumprod[t])


# ============================================================
# Simple Neural Network (MLP denoiser)
# ============================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return x * sig

def silu_grad(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return sig + x * sig * (1 - sig)


class TimeEmbedding:
    """Sinusoidal timestep embedding."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """t: (batch,) integer timesteps -> (batch, embed_dim) embeddings."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        half = self.embed_dim // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(half) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        if self.embed_dim % 2 == 1:
            emb = np.concatenate([emb, np.zeros((len(t), 1))], axis=-1)
        return emb


class DenoisingMLP:
    """Simple MLP for noise prediction: (x_t, t) -> predicted noise."""

    def __init__(self, data_dim: int, hidden_dim: int = 128, time_embed_dim: int = 32,
                 num_layers: int = 3, cond_dim: int = 0):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.num_layers = num_layers
        self.cond_dim = cond_dim
        self.time_embed = TimeEmbedding(time_embed_dim)

        # Input: data_dim + time_embed_dim + cond_dim
        input_dim = data_dim + time_embed_dim + cond_dim

        self.weights = []
        self.biases = []
        rng = np.random.RandomState(42)

        prev_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else data_dim
            # He initialization
            scale = np.sqrt(2.0 / prev_dim)
            self.weights.append(rng.randn(prev_dim, out_dim) * scale)
            self.biases.append(np.zeros(out_dim))
            prev_dim = out_dim

        # Cache for backprop
        self._cache = {}

    def forward(self, x_t: np.ndarray, t: np.ndarray,
                cond: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict noise. x_t: (batch, data_dim), t: (batch,)."""
        t_emb = self.time_embed(t)
        if cond is not None:
            h = np.concatenate([x_t, t_emb, cond], axis=-1)
        else:
            h = np.concatenate([x_t, t_emb], axis=-1)

        self._cache['input'] = h
        self._cache['pre_activations'] = []
        self._cache['activations'] = [h]

        for i in range(self.num_layers):
            pre = h @ self.weights[i] + self.biases[i]
            self._cache['pre_activations'].append(pre)
            if i < self.num_layers - 1:
                h = silu(pre)
            else:
                h = pre  # No activation on output
            self._cache['activations'].append(h)

        return h

    def backward(self, grad_output: np.ndarray, lr: float = 1e-3) -> None:
        """Backprop and update weights."""
        batch_size = grad_output.shape[0]
        grad = grad_output

        for i in range(self.num_layers - 1, -1, -1):
            # Gradient through activation
            if i < self.num_layers - 1:
                grad = grad * silu_grad(self._cache['pre_activations'][i])

            # Weight and bias gradients
            act = self._cache['activations'][i]
            grad_w = act.T @ grad / batch_size
            grad_b = np.mean(grad, axis=0)

            # Propagate gradient
            if i > 0:
                grad = grad @ self.weights[i].T

            # Update
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    def set_parameters(self, params: List[np.ndarray]):
        n = self.num_layers
        self.weights = [p.copy() for p in params[:n]]
        self.biases = [p.copy() for p in params[n:]]

    def copy(self) -> 'DenoisingMLP':
        """Create a copy of this network."""
        net = DenoisingMLP(self.data_dim, self.hidden_dim, self.time_embed_dim,
                           self.num_layers, self.cond_dim)
        net.weights = [w.copy() for w in self.weights]
        net.biases = [b.copy() for b in self.biases]
        return net


# ============================================================
# DDPM: Denoising Diffusion Probabilistic Model
# ============================================================

class DDPM:
    """Core DDPM implementation."""

    def __init__(self, data_dim: int, num_timesteps: int = 100,
                 schedule_type: str = "linear", hidden_dim: int = 128,
                 prediction_type: str = "epsilon", cond_dim: int = 0):
        """
        prediction_type: 'epsilon' (predict noise) or 'x0' (predict clean data)
        """
        self.data_dim = data_dim
        self.schedule = NoiseSchedule(num_timesteps, schedule_type)
        self.T = num_timesteps
        self.prediction_type = prediction_type
        self.cond_dim = cond_dim

        self.model = DenoisingMLP(data_dim, hidden_dim, cond_dim=cond_dim)

        # EMA model for better sampling
        self.ema_model = None
        self.ema_decay = 0.999

    def q_sample(self, x_0: np.ndarray, t: np.ndarray,
                 noise: Optional[np.ndarray] = None,
                 rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward process: sample x_t given x_0 and t.
        Returns (x_t, noise)."""
        if noise is None:
            if rng is None:
                rng = np.random.RandomState()
            noise = rng.randn(*x_0.shape)

        sqrt_alpha = self.schedule.sqrt_alpha_cumprod[t][:, None]
        sqrt_one_minus = self.schedule.sqrt_one_minus_alpha_cumprod[t][:, None]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def p_mean_variance(self, x_t: np.ndarray, t: np.ndarray,
                        cond: Optional[np.ndarray] = None,
                        use_ema: bool = False) -> Dict[str, np.ndarray]:
        """Compute predicted mean and variance of p(x_{t-1} | x_t)."""
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        pred = model.forward(x_t, t, cond)

        if self.prediction_type == "epsilon":
            # Reconstruct x_0 from predicted noise
            sqrt_recip = self.schedule.sqrt_recip_alpha[t][:, None]
            beta = self.schedule.betas[t][:, None]
            sqrt_one_minus = self.schedule.sqrt_one_minus_alpha_cumprod[t][:, None]

            # Predict x_0
            sqrt_alpha_cum = self.schedule.sqrt_alpha_cumprod[t][:, None]
            pred_x0 = (x_t - sqrt_one_minus * pred) / np.clip(sqrt_alpha_cum, 1e-10, None)

            # Compute posterior mean
            coef1 = self.schedule.posterior_mean_coef1[t][:, None]
            coef2 = self.schedule.posterior_mean_coef2[t][:, None]
            mean = coef1 * pred_x0 + coef2 * x_t
        else:
            # x0 prediction
            pred_x0 = pred
            coef1 = self.schedule.posterior_mean_coef1[t][:, None]
            coef2 = self.schedule.posterior_mean_coef2[t][:, None]
            mean = coef1 * pred_x0 + coef2 * x_t

        variance = self.schedule.posterior_variance[t][:, None]
        log_variance = self.schedule.posterior_log_variance[t][:, None]

        return {
            'mean': mean,
            'variance': variance,
            'log_variance': log_variance,
            'pred_x0': pred_x0,
            'pred': pred,
        }

    def training_step(self, x_0: np.ndarray, lr: float = 1e-3,
                      cond: Optional[np.ndarray] = None,
                      rng: Optional[np.random.RandomState] = None) -> float:
        """Single training step. Returns loss."""
        if rng is None:
            rng = np.random.RandomState()

        batch_size = x_0.shape[0]
        t = rng.randint(0, self.T, size=batch_size)

        x_t, noise = self.q_sample(x_0, t, rng=rng)

        pred = self.model.forward(x_t, t, cond)

        if self.prediction_type == "epsilon":
            target = noise
        else:
            target = x_0

        # MSE loss
        loss = np.mean((pred - target) ** 2)

        # Backward
        grad = 2.0 * (pred - target) / pred.size
        self.model.backward(grad, lr=lr)

        # Update EMA
        if self.ema_model is not None:
            self._update_ema()

        return loss

    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
              lr: float = 1e-3, cond: Optional[np.ndarray] = None,
              rng: Optional[np.random.RandomState] = None,
              verbose: bool = False) -> List[float]:
        """Train the model. Returns loss history."""
        if rng is None:
            rng = np.random.RandomState(42)

        n = data.shape[0]
        losses = []

        for epoch in range(epochs):
            indices = rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[indices[start:end]]
                batch_cond = cond[indices[start:end]] if cond is not None else None

                loss = self.training_step(batch, lr=lr, cond=batch_cond, rng=rng)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def enable_ema(self, decay: float = 0.999):
        """Enable exponential moving average of model weights."""
        self.ema_decay = decay
        self.ema_model = self.model.copy()

    def _update_ema(self):
        """Update EMA model parameters."""
        for i in range(len(self.model.weights)):
            self.ema_model.weights[i] = (
                self.ema_decay * self.ema_model.weights[i] +
                (1 - self.ema_decay) * self.model.weights[i]
            )
            self.ema_model.biases[i] = (
                self.ema_decay * self.ema_model.biases[i] +
                (1 - self.ema_decay) * self.model.biases[i]
            )

    # --------------------------------------------------------
    # Sampling Methods
    # --------------------------------------------------------

    def sample_ddpm(self, n_samples: int, cond: Optional[np.ndarray] = None,
                    use_ema: bool = False,
                    rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Standard DDPM sampling (ancestral)."""
        if rng is None:
            rng = np.random.RandomState()

        x = rng.randn(n_samples, self.data_dim)

        for t_val in range(self.T - 1, -1, -1):
            t = np.full(n_samples, t_val, dtype=int)
            result = self.p_mean_variance(x, t, cond, use_ema=use_ema)

            if t_val > 0:
                noise = rng.randn(*x.shape)
                x = result['mean'] + np.sqrt(result['variance']) * noise
            else:
                x = result['mean']

        return x

    def sample_ddim(self, n_samples: int, num_steps: int = 50,
                    eta: float = 0.0, cond: Optional[np.ndarray] = None,
                    use_ema: bool = False,
                    rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """DDIM sampling (Song et al. 2020). eta=0 is deterministic."""
        if rng is None:
            rng = np.random.RandomState()

        # Create subsequence of timesteps
        step_size = max(1, self.T // num_steps)
        timesteps = list(range(0, self.T, step_size))
        if timesteps[-1] != self.T - 1:
            timesteps.append(self.T - 1)
        timesteps = list(reversed(timesteps))

        x = rng.randn(n_samples, self.data_dim)
        model = self.ema_model if (use_ema and self.ema_model is not None) else self.model

        for i in range(len(timesteps)):
            t_val = timesteps[i]
            t = np.full(n_samples, t_val, dtype=int)

            pred = model.forward(x, t, cond)

            # Predict x_0
            alpha_t = self.schedule.alpha_cumprod[t_val]
            if self.prediction_type == "epsilon":
                pred_x0 = (x - np.sqrt(1 - alpha_t) * pred) / np.sqrt(alpha_t)
            else:
                pred_x0 = pred

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.schedule.alpha_cumprod[t_prev]
            else:
                alpha_prev = 1.0  # t=0

            # DDIM update
            sigma = eta * np.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            dir_xt = np.sqrt(1 - alpha_prev - sigma ** 2)

            if self.prediction_type == "epsilon":
                pred_noise = pred
            else:
                pred_noise = (x - np.sqrt(alpha_t) * pred_x0) / np.sqrt(1 - alpha_t)

            x = np.sqrt(alpha_prev) * pred_x0 + dir_xt * pred_noise

            if sigma > 0 and i < len(timesteps) - 1:
                x += sigma * rng.randn(*x.shape)

        return x

    def sample_progressive(self, n_samples: int, save_every: int = 10,
                           cond: Optional[np.ndarray] = None,
                           use_ema: bool = False,
                           rng: Optional[np.random.RandomState] = None) -> List[np.ndarray]:
        """Sample and save intermediate states for visualization."""
        if rng is None:
            rng = np.random.RandomState()

        x = rng.randn(n_samples, self.data_dim)
        trajectory = [x.copy()]

        for t_val in range(self.T - 1, -1, -1):
            t = np.full(n_samples, t_val, dtype=int)
            result = self.p_mean_variance(x, t, cond, use_ema=use_ema)

            if t_val > 0:
                noise = rng.randn(*x.shape)
                x = result['mean'] + np.sqrt(result['variance']) * noise
            else:
                x = result['mean']

            if t_val % save_every == 0:
                trajectory.append(x.copy())

        return trajectory


# ============================================================
# Classifier-Free Guidance
# ============================================================

class ClassifierFreeGuidance:
    """Classifier-free guidance for conditional diffusion."""

    def __init__(self, ddpm: DDPM, guidance_scale: float = 3.0,
                 uncond_prob: float = 0.1):
        self.ddpm = ddpm
        self.guidance_scale = guidance_scale
        self.uncond_prob = uncond_prob

    def training_step(self, x_0: np.ndarray, cond: np.ndarray,
                      lr: float = 1e-3,
                      rng: Optional[np.random.RandomState] = None) -> float:
        """Training with random conditioning dropout."""
        if rng is None:
            rng = np.random.RandomState()

        batch_size = x_0.shape[0]

        # Randomly drop conditioning
        mask = rng.random(batch_size) < self.uncond_prob
        cond_masked = cond.copy()
        cond_masked[mask] = 0.0

        return self.ddpm.training_step(x_0, lr=lr, cond=cond_masked, rng=rng)

    def train(self, data: np.ndarray, cond: np.ndarray,
              epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
              rng: Optional[np.random.RandomState] = None) -> List[float]:
        """Train with classifier-free guidance."""
        if rng is None:
            rng = np.random.RandomState(42)

        n = data.shape[0]
        losses = []

        for epoch in range(epochs):
            indices = rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[indices[start:end]]
                batch_cond = cond[indices[start:end]]

                loss = self.training_step(batch, batch_cond, lr=lr, rng=rng)
                epoch_loss += loss
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        return losses

    def sample(self, n_samples: int, cond: np.ndarray,
               method: str = "ddpm",
               rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample with classifier-free guidance."""
        if rng is None:
            rng = np.random.RandomState()

        x = rng.randn(n_samples, self.ddpm.data_dim)
        uncond = np.zeros_like(cond)

        T = self.ddpm.T
        schedule = self.ddpm.schedule
        model = self.ddpm.model

        for t_val in range(T - 1, -1, -1):
            t = np.full(n_samples, t_val, dtype=int)

            # Conditional prediction
            pred_cond = model.forward(x, t, cond)
            # Unconditional prediction
            pred_uncond = model.forward(x, t, uncond)

            # Guided prediction
            pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)

            # Compute mean using guided prediction
            if self.ddpm.prediction_type == "epsilon":
                sqrt_alpha_cum = schedule.sqrt_alpha_cumprod[t][:, None]
                sqrt_one_minus = schedule.sqrt_one_minus_alpha_cumprod[t][:, None]
                pred_x0 = (x - sqrt_one_minus * pred) / np.clip(sqrt_alpha_cum, 1e-10, None)
            else:
                pred_x0 = pred

            coef1 = schedule.posterior_mean_coef1[t][:, None]
            coef2 = schedule.posterior_mean_coef2[t][:, None]
            mean = coef1 * pred_x0 + coef2 * x

            if t_val > 0:
                variance = schedule.posterior_variance[t][:, None]
                noise = rng.randn(*x.shape)
                x = mean + np.sqrt(variance) * noise
            else:
                x = mean

        return x


# ============================================================
# Noise Prediction Quality Metrics
# ============================================================

class DiffusionDiagnostics:
    """Diagnostics for evaluating diffusion model quality."""

    @staticmethod
    def noise_prediction_mse(ddpm: DDPM, data: np.ndarray,
                             rng: Optional[np.random.RandomState] = None) -> Dict[str, float]:
        """Evaluate noise prediction MSE at different timesteps."""
        if rng is None:
            rng = np.random.RandomState(0)

        n = data.shape[0]
        early_t = np.full(n, ddpm.T // 10, dtype=int)
        mid_t = np.full(n, ddpm.T // 2, dtype=int)
        late_t = np.full(n, ddpm.T - 1, dtype=int)

        results = {}
        for label, t in [("early", early_t), ("mid", mid_t), ("late", late_t)]:
            x_t, noise = ddpm.q_sample(data, t, rng=rng)
            pred = ddpm.model.forward(x_t, t)
            if ddpm.prediction_type == "epsilon":
                mse = np.mean((pred - noise) ** 2)
            else:
                mse = np.mean((pred - data) ** 2)
            results[f"mse_{label}"] = float(mse)

        results["mse_avg"] = np.mean(list(results.values()))
        return results

    @staticmethod
    def sample_statistics(samples: np.ndarray) -> Dict[str, Any]:
        """Compute statistics of generated samples."""
        return {
            'mean': np.mean(samples, axis=0).tolist(),
            'std': np.std(samples, axis=0).tolist(),
            'min': np.min(samples, axis=0).tolist(),
            'max': np.max(samples, axis=0).tolist(),
            'global_mean': float(np.mean(samples)),
            'global_std': float(np.std(samples)),
        }

    @staticmethod
    def fid_proxy(real: np.ndarray, fake: np.ndarray) -> float:
        """Simplified FID-like metric using mean/covariance comparison.
        Not true FID (no Inception), but captures distribution similarity."""
        mu_real = np.mean(real, axis=0)
        mu_fake = np.mean(fake, axis=0)
        cov_real = np.cov(real, rowvar=False)
        cov_fake = np.cov(fake, rowvar=False)

        # Mean difference
        diff = mu_real - mu_fake
        mean_term = np.dot(diff, diff)

        # Covariance trace term (simplified -- no matrix sqrt)
        cov_term = np.trace(cov_real + cov_fake - 2 * np.sqrt(
            np.abs(cov_real * cov_fake) + 1e-10
        ))

        return float(mean_term + cov_term)

    @staticmethod
    def schedule_analysis(schedule: NoiseSchedule) -> Dict[str, Any]:
        """Analyze properties of a noise schedule."""
        return {
            'type': schedule.schedule_type,
            'T': schedule.T,
            'beta_min': float(np.min(schedule.betas)),
            'beta_max': float(np.max(schedule.betas)),
            'beta_mean': float(np.mean(schedule.betas)),
            'alpha_cumprod_min': float(np.min(schedule.alpha_cumprod)),
            'alpha_cumprod_max': float(np.max(schedule.alpha_cumprod)),
            'snr_first': float(schedule.snr(0)),
            'snr_last': float(schedule.snr(schedule.T - 1)),
            'snr_mid': float(schedule.snr(schedule.T // 2)),
        }

    @staticmethod
    def reconstruction_quality(ddpm: DDPM, data: np.ndarray,
                               timestep: int = 10,
                               rng: Optional[np.random.RandomState] = None) -> float:
        """Test reconstruction: noise to small t, then denoise. Lower = better."""
        if rng is None:
            rng = np.random.RandomState(0)

        n = data.shape[0]
        t = np.full(n, timestep, dtype=int)
        x_t, noise = ddpm.q_sample(data, t, rng=rng)

        # Single-step denoise
        result = ddpm.p_mean_variance(x_t, t)
        pred_x0 = result['pred_x0']

        return float(np.mean((pred_x0 - data) ** 2))


# ============================================================
# Score-Based Model (Score Matching)
# ============================================================

class ScoreBasedModel:
    """Score-based diffusion (Langevin dynamics sampling)."""

    def __init__(self, data_dim: int, num_scales: int = 10,
                 sigma_min: float = 0.01, sigma_max: float = 10.0,
                 hidden_dim: int = 128):
        self.data_dim = data_dim
        self.num_scales = num_scales
        self.sigmas = np.geomspace(sigma_max, sigma_min, num_scales)
        self.model = DenoisingMLP(data_dim, hidden_dim, time_embed_dim=32)

    def score_matching_loss(self, data: np.ndarray, lr: float = 1e-3,
                            rng: Optional[np.random.RandomState] = None) -> float:
        """Denoising score matching loss."""
        if rng is None:
            rng = np.random.RandomState()

        batch_size = data.shape[0]
        # Random noise level
        idx = rng.randint(0, self.num_scales, size=batch_size)
        sigma = self.sigmas[idx]

        noise = rng.randn(*data.shape)
        perturbed = data + sigma[:, None] * noise

        # Target score: -noise / sigma
        target = -noise / sigma[:, None]

        # Predict score (use sigma index as "timestep")
        pred = self.model.forward(perturbed, idx)

        # Weighted MSE
        weight = sigma[:, None] ** 2
        loss = np.mean(weight * (pred - target) ** 2)

        grad = 2.0 * weight * (pred - target) / pred.size
        self.model.backward(grad, lr=lr)

        return float(loss)

    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
              lr: float = 1e-3, rng: Optional[np.random.RandomState] = None) -> List[float]:
        """Train score model."""
        if rng is None:
            rng = np.random.RandomState(42)

        n = data.shape[0]
        losses = []

        for epoch in range(epochs):
            indices = rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[indices[start:end]]
                loss = self.score_matching_loss(batch, lr=lr, rng=rng)
                epoch_loss += loss
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        return losses

    def sample_langevin(self, n_samples: int, num_steps: int = 100,
                        step_size: float = 0.01,
                        rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Annealed Langevin dynamics sampling."""
        if rng is None:
            rng = np.random.RandomState()

        x = rng.randn(n_samples, self.data_dim) * self.sigmas[0]

        for i, sigma in enumerate(self.sigmas):
            alpha = step_size * (sigma / self.sigmas[-1]) ** 2
            t = np.full(n_samples, i, dtype=int)

            for _ in range(num_steps):
                score = self.model.forward(x, t)
                noise = rng.randn(*x.shape)
                x = x + alpha * score + np.sqrt(2 * alpha) * noise

        return x


# ============================================================
# Variance-Preserving SDE (VP-SDE)
# ============================================================

class VPSDE:
    """Variance-Preserving SDE formulation of diffusion."""

    def __init__(self, data_dim: int, beta_min: float = 0.1,
                 beta_max: float = 20.0, hidden_dim: int = 128):
        self.data_dim = data_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.model = DenoisingMLP(data_dim, hidden_dim, time_embed_dim=32)

    def beta(self, t: np.ndarray) -> np.ndarray:
        """Beta(t) for continuous time t in [0, 1]."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def mean_coeff(self, t: np.ndarray) -> np.ndarray:
        """Mean coefficient for q(x_t | x_0)."""
        log_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return np.exp(log_coeff)

    def variance(self, t: np.ndarray) -> np.ndarray:
        """Variance of q(x_t | x_0)."""
        return 1.0 - self.mean_coeff(t) ** 2

    def training_step(self, data: np.ndarray, lr: float = 1e-3,
                      rng: Optional[np.random.RandomState] = None) -> float:
        """Single training step for VP-SDE score model."""
        if rng is None:
            rng = np.random.RandomState()

        batch_size = data.shape[0]
        # Continuous time
        t = rng.uniform(1e-5, 1.0, size=batch_size)

        mean_c = self.mean_coeff(t)[:, None]
        var = self.variance(t)[:, None]
        std = np.sqrt(var)

        noise = rng.randn(*data.shape)
        x_t = mean_c * data + std * noise

        # Target: -noise / std (score)
        # But predict noise directly for simplicity
        # Use t * 100 as discrete-like timestep for embedding
        t_discrete = (t * 100).astype(int)
        pred = self.model.forward(x_t, t_discrete)

        loss = np.mean((pred - noise) ** 2)
        grad = 2.0 * (pred - noise) / pred.size
        self.model.backward(grad, lr=lr)

        return float(loss)

    def train(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
              lr: float = 1e-3, rng: Optional[np.random.RandomState] = None) -> List[float]:
        """Train VP-SDE model."""
        if rng is None:
            rng = np.random.RandomState(42)

        n = data.shape[0]
        losses = []

        for epoch in range(epochs):
            indices = rng.permutation(n)
            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[indices[start:end]]
                loss = self.training_step(batch, lr=lr, rng=rng)
                epoch_loss += loss
                num_batches += 1

            losses.append(epoch_loss / num_batches)

        return losses

    def sample(self, n_samples: int, num_steps: int = 100,
               rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Sample using Euler-Maruyama SDE solver."""
        if rng is None:
            rng = np.random.RandomState()

        dt = 1.0 / num_steps
        x = rng.randn(n_samples, self.data_dim)

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t = np.full(n_samples, t_val)
            t_discrete = (t * 100).astype(int)

            score_pred = self.model.forward(x, t_discrete)
            beta_t = self.beta(t)[:, None]

            # Reverse SDE drift
            drift = -0.5 * beta_t * x + beta_t * score_pred
            diffusion = np.sqrt(beta_t)

            noise = rng.randn(*x.shape) if i < num_steps - 1 else 0
            x = x + drift * dt + diffusion * np.sqrt(dt) * noise

        return x


# ============================================================
# Data Generators (for testing)
# ============================================================

def make_moons(n: int, noise: float = 0.1,
               rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate 2D moons dataset."""
    if rng is None:
        rng = np.random.RandomState(42)

    n_each = n // 2
    # Outer moon
    theta1 = np.linspace(0, np.pi, n_each)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    # Inner moon
    theta2 = np.linspace(0, np.pi, n - n_each)
    x2 = np.column_stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)])

    data = np.vstack([x1, x2])
    data += rng.randn(*data.shape) * noise
    return data


def make_circles(n: int, noise: float = 0.05,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Generate 2D concentric circles."""
    if rng is None:
        rng = np.random.RandomState(42)

    n_each = n // 2
    theta1 = rng.uniform(0, 2 * np.pi, n_each)
    r1 = 1.0
    x1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    theta2 = rng.uniform(0, 2 * np.pi, n - n_each)
    r2 = 0.5
    x2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])

    data = np.vstack([x1, x2])
    data += rng.randn(*data.shape) * noise
    return data


def make_gaussian_mixture(n: int, n_components: int = 4,
                          spread: float = 3.0, dim: int = 2,
                          rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian mixture data. Returns (data, labels)."""
    if rng is None:
        rng = np.random.RandomState(42)

    per_comp = n // n_components
    data_parts = []
    label_parts = []

    for i in range(n_components):
        angle = 2 * np.pi * i / n_components
        center = np.zeros(dim)
        center[0] = spread * np.cos(angle)
        center[1] = spread * np.sin(angle)

        count = per_comp if i < n_components - 1 else n - per_comp * (n_components - 1)
        points = rng.randn(count, dim) * 0.5 + center
        data_parts.append(points)
        label_parts.append(np.full(count, i))

    data = np.vstack(data_parts)
    labels = np.concatenate(label_parts)

    # Shuffle
    idx = rng.permutation(n)
    return data[idx], labels[idx]
