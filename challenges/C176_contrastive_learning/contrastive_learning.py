"""
C176: Contrastive Learning
Composing C140 (Neural Networks) + C175 (Semi-Supervised Learning)

10 components:
1. AugmentationPipeline -- creates augmented views for contrastive pairs
2. ContrastiveLoss -- NT-Xent, TripletLoss, InfoNCE
3. ProjectionHead -- MLP projection to contrastive embedding space
4. SimCLR -- SimCLR framework (encoder + projector + NT-Xent)
5. BYOL -- Bootstrap Your Own Latent (EMA target network, no negatives)
6. BarlowTwins -- Redundancy reduction via cross-correlation identity
7. ContrastiveTrainer -- Orchestrates contrastive pre-training
8. LinearEvaluator -- Linear probe protocol for representation quality
9. RepresentationAnalyzer -- Alignment, uniformity, clustering quality
10. ContrastiveMetrics -- Static evaluation metrics
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C175_semi_supervised_learning'))

from neural_network import (
    Tensor, Dense, Activation, Sequential, Adam, SGD,
    CrossEntropyLoss, MSELoss, train_step, fit, one_hot,
    build_model, accuracy, predict_classes, BatchNorm, Dropout
)


# ── Helpers ──────────────────────────────────────────────────────────

def _num_rows(t):
    if isinstance(t, Tensor):
        return t.shape[0]
    return len(t)


def _tensor_row(t, i):
    if isinstance(t, Tensor):
        if len(t.shape) == 1:
            return t.data[i]
        return t.data[i]
    return t[i]


def _make_tensor(data):
    if isinstance(data, Tensor):
        return data
    return Tensor(data)


def _dot(a, b):
    """Dot product of two flat lists."""
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    """L2 norm of a flat list."""
    return math.sqrt(sum(x * x for x in a))


def _cosine_sim(a, b):
    """Cosine similarity between two flat lists."""
    n1 = _norm(a)
    n2 = _norm(b)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return _dot(a, b) / (n1 * n2)


def _l2_normalize(vec):
    """L2 normalize a flat list."""
    n = _norm(vec)
    if n < 1e-12:
        return vec[:]
    return [x / n for x in vec]


def _softmax(logits):
    """Softmax over a flat list."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _log_softmax(logits):
    """Log-softmax over a flat list."""
    m = max(logits)
    log_sum = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - log_sum for x in logits]


def _subset_rows(t, indices):
    """Extract rows from a Tensor or list by indices."""
    if isinstance(t, Tensor):
        if len(t.shape) == 1:
            return Tensor([t.data[i] for i in indices])
        return Tensor([t.data[i][:] for i in indices])
    return [t[i] for i in indices]


def _forward_get_repr(model, x):
    """Forward pass returning representation (before last layer if projection)."""
    return model.forward(x)


# ── 1. AugmentationPipeline ─────────────────────────────────────────

class AugmentationPipeline:
    """Creates augmented views of data for contrastive learning.

    Supports: additive noise, feature masking, scaling, and shuffling.
    """

    def __init__(self, noise_std=0.1, mask_ratio=0.0, scale_range=None, seed=42):
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio
        self.scale_range = scale_range  # (low, high) or None
        self.rng = random.Random(seed)

    def augment(self, X):
        """Create an augmented view of X (Tensor or list of lists)."""
        n = _num_rows(X)
        result = []
        for i in range(n):
            row = list(_tensor_row(X, i))
            row = self._augment_row(row)
            result.append(row)
        return Tensor(result) if isinstance(X, Tensor) else result

    def _augment_row(self, row):
        """Apply augmentations to a single row."""
        d = len(row)

        # Additive noise
        if self.noise_std > 0:
            row = [x + self.rng.gauss(0, self.noise_std) for x in row]

        # Feature masking
        if self.mask_ratio > 0:
            n_mask = max(1, int(d * self.mask_ratio))
            mask_indices = self.rng.sample(range(d), min(n_mask, d))
            for idx in mask_indices:
                row[idx] = 0.0

        # Random scaling
        if self.scale_range is not None:
            lo, hi = self.scale_range
            scale = self.rng.uniform(lo, hi)
            row = [x * scale for x in row]

        return row

    def create_pair(self, X):
        """Create two different augmented views of X."""
        view1 = self.augment(X)
        view2 = self.augment(X)
        return view1, view2


# ── 2. ContrastiveLoss ──────────────────────────────────────────────

class NTXentLoss:
    """Normalized Temperature-scaled Cross-Entropy Loss (SimCLR).

    For a batch of N pairs (2N total), the loss for pair (i, i+N) is:
    -log(exp(sim(z_i, z_{i+N})/tau) / sum_k!=i(exp(sim(z_i, z_k)/tau)))
    """

    def __init__(self, temperature=0.5):
        self.temperature = temperature

    def forward(self, z1, z2):
        """Compute NT-Xent loss.

        Args:
            z1, z2: Tensor or list of lists, shape (N, D) -- projected embeddings
        Returns:
            scalar loss
        """
        n = _num_rows(z1)
        if n == 0:
            return 0.0

        # Normalize embeddings
        embs1 = [_l2_normalize(list(_tensor_row(z1, i))) for i in range(n)]
        embs2 = [_l2_normalize(list(_tensor_row(z2, i))) for i in range(n)]

        # Concatenate: [z1_0, ..., z1_{n-1}, z2_0, ..., z2_{n-1}]
        all_embs = embs1 + embs2
        total = 2 * n

        # Compute similarity matrix
        sim_matrix = [[0.0] * total for _ in range(total)]
        for i in range(total):
            for j in range(total):
                if i != j:
                    sim_matrix[i][j] = _dot(all_embs[i], all_embs[j]) / self.temperature

        # Loss: for each i, positive is i+n (mod 2n)
        total_loss = 0.0
        for i in range(total):
            pos_j = (i + n) % total
            # log-sum-exp over all j != i
            logits = []
            pos_idx = -1
            for j in range(total):
                if j != i:
                    logits.append(sim_matrix[i][j])
                    if j == pos_j:
                        pos_idx = len(logits) - 1

            log_probs = _log_softmax(logits)
            total_loss -= log_probs[pos_idx]

        return total_loss / total

    def backward(self, z1, z2):
        """Compute gradients w.r.t. z1 and z2.

        Returns (grad_z1, grad_z2) as Tensors.
        """
        n = _num_rows(z1)
        if n == 0:
            return z1, z2

        embs1 = [_l2_normalize(list(_tensor_row(z1, i))) for i in range(n)]
        embs2 = [_l2_normalize(list(_tensor_row(z2, i))) for i in range(n)]
        all_embs = embs1 + embs2
        total = 2 * n
        d = len(embs1[0])

        # Similarity matrix
        sim_matrix = [[0.0] * total for _ in range(total)]
        for i in range(total):
            for j in range(total):
                if i != j:
                    sim_matrix[i][j] = _dot(all_embs[i], all_embs[j]) / self.temperature

        # Gradient computation
        grad_all = [[0.0] * d for _ in range(total)]
        for i in range(total):
            pos_j = (i + n) % total
            logits = []
            indices = []
            for j in range(total):
                if j != i:
                    logits.append(sim_matrix[i][j])
                    indices.append(j)

            probs = _softmax(logits)
            # Gradient: for each j != i, (prob_j - 1{j==pos}) * z_j / (tau * total)
            for k_idx, j in enumerate(indices):
                target = 1.0 if j == pos_j else 0.0
                coeff = (probs[k_idx] - target) / (self.temperature * total)
                for dd in range(d):
                    grad_all[i][dd] += coeff * all_embs[j][dd]

        grad_z1 = Tensor(grad_all[:n])
        grad_z2 = Tensor(grad_all[n:])
        return grad_z1, grad_z2


class TripletLoss:
    """Triplet loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)."""

    def __init__(self, margin=1.0):
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """Compute triplet loss.

        Args:
            anchor, positive, negative: Tensor or list of lists (N, D)
        Returns:
            scalar loss
        """
        n = _num_rows(anchor)
        if n == 0:
            return 0.0

        total_loss = 0.0
        for i in range(n):
            a = list(_tensor_row(anchor, i))
            p = list(_tensor_row(positive, i))
            ne = list(_tensor_row(negative, i))

            d_pos = sum((ai - pi) ** 2 for ai, pi in zip(a, p))
            d_neg = sum((ai - ni) ** 2 for ai, ni in zip(a, ne))
            loss_i = max(0.0, d_pos - d_neg + self.margin)
            total_loss += loss_i

        return total_loss / n

    def backward(self, anchor, positive, negative):
        """Returns (grad_anchor, grad_positive, grad_negative)."""
        n = _num_rows(anchor)
        d = len(list(_tensor_row(anchor, 0)))

        grad_a = [[0.0] * d for _ in range(n)]
        grad_p = [[0.0] * d for _ in range(n)]
        grad_n = [[0.0] * d for _ in range(n)]

        for i in range(n):
            a = list(_tensor_row(anchor, i))
            p = list(_tensor_row(positive, i))
            ne = list(_tensor_row(negative, i))

            d_pos = sum((ai - pi) ** 2 for ai, pi in zip(a, p))
            d_neg = sum((ai - ni) ** 2 for ai, ni in zip(a, ne))

            if d_pos - d_neg + self.margin > 0:
                for dd in range(d):
                    grad_a[i][dd] = 2.0 * ((a[dd] - p[dd]) - (a[dd] - ne[dd])) / n
                    grad_p[i][dd] = 2.0 * (p[dd] - a[dd]) / n
                    grad_n[i][dd] = 2.0 * (a[dd] - ne[dd]) / (-n)

        return Tensor(grad_a), Tensor(grad_p), Tensor(grad_n)


class InfoNCELoss:
    """InfoNCE loss (contrastive predictive coding).

    Loss = -log(exp(sim(q, k+)/tau) / (exp(sim(q, k+)/tau) + sum(exp(sim(q, k-)/tau))))
    """

    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def forward(self, queries, keys_pos, keys_neg=None):
        """Compute InfoNCE loss.

        Args:
            queries: (N, D) query embeddings
            keys_pos: (N, D) positive key embeddings
            keys_neg: (M, D) negative key embeddings or None (use other positives)
        Returns:
            scalar loss
        """
        n = _num_rows(queries)
        if n == 0:
            return 0.0

        q_embs = [_l2_normalize(list(_tensor_row(queries, i))) for i in range(n)]
        kp_embs = [_l2_normalize(list(_tensor_row(keys_pos, i))) for i in range(n)]

        if keys_neg is not None:
            m = _num_rows(keys_neg)
            kn_embs = [_l2_normalize(list(_tensor_row(keys_neg, j))) for j in range(m)]
        else:
            # Use other positives as negatives (in-batch)
            kn_embs = None

        total_loss = 0.0
        for i in range(n):
            pos_sim = _dot(q_embs[i], kp_embs[i]) / self.temperature
            neg_sims = []

            if kn_embs is not None:
                for j in range(len(kn_embs)):
                    neg_sims.append(_dot(q_embs[i], kn_embs[j]) / self.temperature)
            else:
                # In-batch negatives: all other positive keys
                for j in range(n):
                    if j != i:
                        neg_sims.append(_dot(q_embs[i], kp_embs[j]) / self.temperature)

            all_logits = [pos_sim] + neg_sims
            log_probs = _log_softmax(all_logits)
            total_loss -= log_probs[0]

        return total_loss / n


# ── 3. ProjectionHead ───────────────────────────────────────────────

class ProjectionHead:
    """MLP projection head mapping representations to contrastive space.

    Typically 2-3 layers with ReLU, projects to lower dimension for loss computation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        rng = random.Random(seed)

        layers = []
        in_d = input_dim
        for i in range(num_layers - 1):
            layers.append(Dense(in_d, hidden_dim, init='xavier', rng=rng))
            layers.append(Activation('relu'))
            in_d = hidden_dim
        layers.append(Dense(in_d, output_dim, init='xavier', rng=rng))

        self.model = Sequential(layers)
        self.training = True

    def forward(self, x):
        return self.model.forward(x)

    def backward(self, grad):
        return self.model.backward(grad)

    def get_params(self):
        params = []
        for layer in self.model.layers:
            params.extend(layer.get_params())
        return params

    def train(self):
        self.training = True
        self.model.train()

    def eval(self):
        self.training = False
        self.model.eval()

    def copy_params_from(self, other):
        """Copy parameters from another ProjectionHead."""
        for my_layer, other_layer in zip(self.model.layers, other.model.layers):
            my_params = my_layer.get_params()
            other_params = other_layer.get_params()
            for (mp, _, _), (op, _, _) in zip(my_params, other_params):
                if isinstance(mp, Tensor):
                    if len(mp.shape) == 2:
                        for i in range(mp.shape[0]):
                            for j in range(mp.shape[1]):
                                mp.data[i][j] = op.data[i][j]
                    else:
                        for i in range(mp.shape[0]):
                            mp.data[i] = op.data[i]


# ── 4. SimCLR ───────────────────────────────────────────────────────

class SimCLR:
    """SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

    Architecture: encoder -> projection_head -> NT-Xent loss
    After training, discard projection head and use encoder representations.
    """

    def __init__(self, encoder, proj_input_dim, proj_hidden_dim=128,
                 proj_output_dim=64, temperature=0.5, seed=42):
        self.encoder = encoder
        self.projector = ProjectionHead(proj_input_dim, proj_hidden_dim,
                                         proj_output_dim, num_layers=2, seed=seed)
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.temperature = temperature

    def encode(self, x):
        """Get representations from encoder (without projection)."""
        return self.encoder.forward(x)

    def project(self, h):
        """Project representations to contrastive space."""
        return self.projector.forward(h)

    def forward(self, x):
        """Full forward: encode then project."""
        h = self.encode(x)
        z = self.project(h)
        return h, z

    def compute_loss(self, z1, z2):
        """Compute NT-Xent loss between two projected views."""
        return self.loss_fn.forward(z1, z2)

    def train_step(self, x1, x2, optimizer):
        """One training step with two augmented views.

        Returns scalar loss.
        """
        self.encoder.train()
        self.projector.train()

        # Forward
        h1, z1 = self.forward(x1)
        h2, z2 = self.forward(x2)

        loss = self.compute_loss(z1, z2)

        # Backward
        grad_z1, grad_z2 = self.loss_fn.backward(z1, z2)

        # Backprop through projector and encoder for view 1
        grad_h1 = self.projector.backward(grad_z1)
        self.encoder.backward(grad_h1)

        # Backprop through projector and encoder for view 2
        # Re-forward view 2 to set cached values
        self.encoder.forward(x2)
        self.projector.forward(h2)
        grad_h2 = self.projector.backward(grad_z2)
        self.encoder.backward(grad_h2)

        # Update all parameters
        all_layers = list(self.encoder.layers) + list(self.projector.model.layers)
        optimizer.step(all_layers)

        return loss

    def get_representations(self, X):
        """Get encoder representations (no projection)."""
        self.encoder.eval()
        return self.encoder.forward(X)

    def train(self):
        self.encoder.train()
        self.projector.train()

    def eval(self):
        self.encoder.eval()
        self.projector.eval()


# ── 5. BYOL ─────────────────────────────────────────────────────────

class BYOL:
    """Bootstrap Your Own Latent -- self-supervised without negative pairs.

    Uses online/target network architecture with EMA updates.
    Online: encoder -> projector -> predictor
    Target: encoder -> projector (EMA of online, no gradient)
    Loss: MSE between online predictor output and target projection (both L2-normalized).
    """

    def __init__(self, encoder, proj_input_dim, proj_hidden_dim=128,
                 proj_output_dim=64, pred_hidden_dim=64,
                 ema_decay=0.996, seed=42):
        self.ema_decay = ema_decay
        rng = random.Random(seed)

        # Online network
        self.online_encoder = encoder
        self.online_projector = ProjectionHead(
            proj_input_dim, proj_hidden_dim, proj_output_dim,
            num_layers=2, seed=rng.randint(0, 99999))
        self.predictor = ProjectionHead(
            proj_output_dim, pred_hidden_dim, proj_output_dim,
            num_layers=2, seed=rng.randint(0, 99999))

        # Target network (copy of online, updated via EMA)
        self.target_encoder = self._copy_sequential(encoder, rng)
        self.target_projector = ProjectionHead(
            proj_input_dim, proj_hidden_dim, proj_output_dim,
            num_layers=2, seed=rng.randint(0, 99999))
        # Initialize target with online params
        self._copy_params(self.online_encoder, self.target_encoder)
        self.target_projector.copy_params_from(self.online_projector)

    def _copy_sequential(self, model, rng):
        """Create a new Sequential with same architecture."""
        layers = []
        for layer in model.layers:
            if isinstance(layer, Dense):
                new_layer = Dense(layer.weights.shape[0], layer.weights.shape[1],
                                  init='xavier', rng=rng)
                layers.append(new_layer)
            elif isinstance(layer, Activation):
                layers.append(Activation(layer.name))
            elif isinstance(layer, BatchNorm):
                layers.append(BatchNorm(layer.gamma.shape[0]))
            elif isinstance(layer, Dropout):
                layers.append(Dropout(layer.rate))
        return Sequential(layers)

    def _copy_params(self, src, dst):
        """Copy parameters from src to dst Sequential."""
        for sl, dl in zip(src.layers, dst.layers):
            sp = sl.get_params()
            dp = dl.get_params()
            for (s_tensor, _, _), (d_tensor, _, _) in zip(sp, dp):
                if isinstance(s_tensor, Tensor):
                    if len(s_tensor.shape) == 2:
                        for i in range(s_tensor.shape[0]):
                            for j in range(s_tensor.shape[1]):
                                d_tensor.data[i][j] = s_tensor.data[i][j]
                    else:
                        for i in range(s_tensor.shape[0]):
                            d_tensor.data[i] = s_tensor.data[i]

    def _ema_update(self):
        """Update target network with EMA of online network."""
        tau = self.ema_decay
        # Update encoder
        for ol, tl in zip(self.online_encoder.layers, self.target_encoder.layers):
            op = ol.get_params()
            tp = tl.get_params()
            for (o_t, _, _), (t_t, _, _) in zip(op, tp):
                if isinstance(o_t, Tensor):
                    if len(o_t.shape) == 2:
                        for i in range(o_t.shape[0]):
                            for j in range(o_t.shape[1]):
                                t_t.data[i][j] = tau * t_t.data[i][j] + (1 - tau) * o_t.data[i][j]
                    else:
                        for i in range(o_t.shape[0]):
                            t_t.data[i] = tau * t_t.data[i] + (1 - tau) * o_t.data[i]
        # Update projector
        for ol, tl in zip(self.online_projector.model.layers,
                          self.target_projector.model.layers):
            op = ol.get_params()
            tp = tl.get_params()
            for (o_t, _, _), (t_t, _, _) in zip(op, tp):
                if isinstance(o_t, Tensor):
                    if len(o_t.shape) == 2:
                        for i in range(o_t.shape[0]):
                            for j in range(o_t.shape[1]):
                                t_t.data[i][j] = tau * t_t.data[i][j] + (1 - tau) * o_t.data[i][j]
                    else:
                        for i in range(o_t.shape[0]):
                            t_t.data[i] = tau * t_t.data[i] + (1 - tau) * o_t.data[i]

    def _byol_loss(self, p, z):
        """MSE between L2-normalized predictions and targets."""
        n = _num_rows(p)
        d = len(list(_tensor_row(p, 0)))
        total = 0.0
        for i in range(n):
            p_row = _l2_normalize(list(_tensor_row(p, i)))
            z_row = _l2_normalize(list(_tensor_row(z, i)))
            total += sum((a - b) ** 2 for a, b in zip(p_row, z_row))
        return total / n

    def _byol_loss_grad(self, p, z):
        """Gradient of BYOL loss w.r.t. p (predictions)."""
        n = _num_rows(p)
        d = len(list(_tensor_row(p, 0)))
        grad = [[0.0] * d for _ in range(n)]
        for i in range(n):
            p_raw = list(_tensor_row(p, i))
            z_raw = list(_tensor_row(z, i))
            p_n = _l2_normalize(p_raw)
            z_n = _l2_normalize(z_raw)
            p_norm = _norm(p_raw)
            if p_norm < 1e-12:
                continue
            # d/dp of ||p/||p|| - z/||z||||^2
            # = 2/||p|| * (p/||p|| - z/||z|| - (p/||p|| . (p/||p|| - z/||z||)) * p/||p||)
            diff = [a - b for a, b in zip(p_n, z_n)]
            dot_pn_diff = sum(a * b for a, b in zip(p_n, diff))
            for dd in range(d):
                grad[i][dd] = 2.0 * (diff[dd] - dot_pn_diff * p_n[dd]) / (p_norm * n)
        return Tensor(grad)

    def train_step(self, x1, x2, optimizer):
        """One BYOL training step.

        Returns scalar loss.
        """
        # Online forward for view 1
        h1_online = self.online_encoder.forward(x1)
        z1_online = self.online_projector.forward(h1_online)
        p1 = self.predictor.forward(z1_online)

        # Target forward for view 2 (no gradient)
        h2_target = self.target_encoder.forward(x2)
        z2_target = self.target_projector.forward(h2_target)

        # Loss: predict view2 from view1
        loss1 = self._byol_loss(p1, z2_target)

        # Online forward for view 2
        h2_online = self.online_encoder.forward(x2)
        z2_online = self.online_projector.forward(h2_online)
        p2 = self.predictor.forward(z2_online)

        # Target forward for view 1 (no gradient)
        h1_target = self.target_encoder.forward(x1)
        z1_target = self.target_projector.forward(h1_target)

        # Loss: predict view1 from view2
        loss2 = self._byol_loss(p2, z1_target)

        total_loss = (loss1 + loss2) / 2.0

        # Backward for loss1 path: grad through predictor -> projector -> encoder (view1)
        self.online_encoder.forward(x1)
        h1_online = self.online_encoder.forward(x1)
        z1_online = self.online_projector.forward(h1_online)
        p1 = self.predictor.forward(z1_online)

        grad_p1 = self._byol_loss_grad(p1, z2_target)
        grad_z1 = self.predictor.backward(grad_p1)
        grad_h1 = self.online_projector.backward(grad_z1)
        self.online_encoder.backward(grad_h1)

        # Update online network
        all_layers = (list(self.online_encoder.layers) +
                      list(self.online_projector.model.layers) +
                      list(self.predictor.model.layers))
        optimizer.step(all_layers)

        # EMA update target
        self._ema_update()

        return total_loss

    def get_representations(self, X):
        """Get encoder representations."""
        self.online_encoder.eval()
        return self.online_encoder.forward(X)

    def train(self):
        self.online_encoder.train()
        self.online_projector.train()
        self.predictor.train()

    def eval(self):
        self.online_encoder.eval()
        self.online_projector.eval()
        self.predictor.eval()


# ── 6. BarlowTwins ─────────────────────────────────────────────────

class BarlowTwins:
    """Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

    Pushes cross-correlation matrix between embeddings of two views toward identity.
    Loss = sum_i (1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2
    """

    def __init__(self, encoder, proj_input_dim, proj_hidden_dim=128,
                 proj_output_dim=64, lambd=0.005, seed=42):
        self.encoder = encoder
        self.projector = ProjectionHead(proj_input_dim, proj_hidden_dim,
                                         proj_output_dim, num_layers=2, seed=seed)
        self.lambd = lambd
        self.proj_output_dim = proj_output_dim

    def _standardize(self, z):
        """Standardize along batch dimension (zero mean, unit std)."""
        n = _num_rows(z)
        d = len(list(_tensor_row(z, 0)))
        rows = [list(_tensor_row(z, i)) for i in range(n)]

        # Compute mean
        mean = [0.0] * d
        for row in rows:
            for j in range(d):
                mean[j] += row[j]
        mean = [m / n for m in mean]

        # Compute std
        std = [0.0] * d
        for row in rows:
            for j in range(d):
                std[j] += (row[j] - mean[j]) ** 2
        std = [math.sqrt(s / max(n - 1, 1) + 1e-8) for s in std]

        # Standardize
        result = []
        for row in rows:
            result.append([(row[j] - mean[j]) / std[j] for j in range(d)])

        return result, mean, std

    def _cross_correlation(self, z1_std, z2_std):
        """Compute cross-correlation matrix C between standardized embeddings."""
        n = len(z1_std)
        d = len(z1_std[0])
        C = [[0.0] * d for _ in range(d)]

        for i in range(d):
            for j in range(d):
                for k in range(n):
                    C[i][j] += z1_std[k][i] * z2_std[k][j]
                C[i][j] /= max(n, 1)

        return C

    def compute_loss(self, z1, z2):
        """Compute Barlow Twins loss.

        Returns scalar loss.
        """
        z1_std, _, _ = self._standardize(z1)
        z2_std, _, _ = self._standardize(z2)
        C = self._cross_correlation(z1_std, z2_std)
        d = len(C)

        loss = 0.0
        for i in range(d):
            for j in range(d):
                if i == j:
                    loss += (1.0 - C[i][j]) ** 2
                else:
                    loss += self.lambd * C[i][j] ** 2

        return loss

    def train_step(self, x1, x2, optimizer):
        """One Barlow Twins training step. Returns scalar loss."""
        self.encoder.train()
        self.projector.train()

        # Forward
        h1 = self.encoder.forward(x1)
        z1 = self.projector.forward(h1)
        h2 = self.encoder.forward(x2)
        z2 = self.projector.forward(h2)

        loss = self.compute_loss(z1, z2)

        # Numerical gradient for simplicity (analytical is complex due to standardization)
        eps = 1e-4
        n = _num_rows(z1)
        d = len(list(_tensor_row(z1, 0)))

        grad_z1 = [[0.0] * d for _ in range(n)]
        for i in range(n):
            for j in range(d):
                old_val = z1.data[i][j]
                z1.data[i][j] = old_val + eps
                loss_plus = self.compute_loss(z1, z2)
                z1.data[i][j] = old_val - eps
                loss_minus = self.compute_loss(z1, z2)
                z1.data[i][j] = old_val
                grad_z1[i][j] = (loss_plus - loss_minus) / (2 * eps)

        grad_z1 = Tensor(grad_z1)

        # Backprop through projector and encoder for view 1
        self.encoder.forward(x1)
        self.projector.forward(h1)
        grad_h1 = self.projector.backward(grad_z1)
        self.encoder.backward(grad_h1)

        all_layers = list(self.encoder.layers) + list(self.projector.model.layers)
        optimizer.step(all_layers)

        return loss

    def get_representations(self, X):
        """Get encoder representations."""
        self.encoder.eval()
        return self.encoder.forward(X)

    def train(self):
        self.encoder.train()
        self.projector.train()

    def eval(self):
        self.encoder.eval()
        self.projector.eval()


# ── 7. ContrastiveTrainer ───────────────────────────────────────────

class ContrastiveTrainer:
    """Orchestrates contrastive pre-training with any framework."""

    def __init__(self, framework, augmenter, optimizer, seed=42):
        """
        Args:
            framework: SimCLR, BYOL, or BarlowTwins instance
            augmenter: AugmentationPipeline instance
            optimizer: Adam, SGD, etc.
        """
        self.framework = framework
        self.augmenter = augmenter
        self.optimizer = optimizer
        self.rng = random.Random(seed)
        self.history = []

    def train(self, X, epochs=10, batch_size=None, verbose=False):
        """Pre-train the framework on unlabeled data X.

        Returns list of per-epoch losses.
        """
        n = _num_rows(X)
        if batch_size is None:
            batch_size = n

        self.history = []
        for epoch in range(epochs):
            # Shuffle indices
            indices = list(range(n))
            self.rng.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_idx = indices[start:end]
                X_batch = _subset_rows(X, batch_idx)

                # Create two augmented views
                view1, view2 = self.augmenter.create_pair(X_batch)

                # Train step
                loss = self.framework.train_step(view1, view2, self.optimizer)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        return self.history

    def get_representations(self, X):
        """Get learned representations for data X."""
        return self.framework.get_representations(X)


# ── 8. LinearEvaluator ──────────────────────────────────────────────

class LinearEvaluator:
    """Linear evaluation protocol for representation quality.

    Freezes encoder, trains a linear classifier on top of learned representations.
    """

    def __init__(self, repr_dim, num_classes, lr=0.01, seed=42):
        self.repr_dim = repr_dim
        self.num_classes = num_classes
        rng = random.Random(seed)
        self.classifier = Sequential([
            Dense(repr_dim, num_classes, init='xavier', rng=rng)
        ])
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD(lr=lr)

    def fit(self, representations, labels, epochs=50, batch_size=None, verbose=False):
        """Train linear classifier on frozen representations.

        Args:
            representations: Tensor (N, repr_dim)
            labels: list of int class labels
        Returns:
            list of per-epoch losses
        """
        n = _num_rows(representations)
        num_classes = self.num_classes
        Y_oh = one_hot(labels, num_classes)

        history = fit(self.classifier, representations, Y_oh, self.loss_fn,
                      self.optimizer, epochs=epochs, batch_size=batch_size,
                      verbose=verbose)
        return history.get('loss', [])

    def evaluate(self, representations, labels):
        """Evaluate linear probe accuracy.

        Returns accuracy float in [0, 1].
        """
        self.classifier.eval()
        return accuracy(self.classifier, representations, labels)

    def predict(self, representations):
        """Predict class labels."""
        self.classifier.eval()
        return predict_classes(self.classifier, representations)


# ── 9. RepresentationAnalyzer ───────────────────────────────────────

class RepresentationAnalyzer:
    """Analyzes quality of learned representations."""

    @staticmethod
    def alignment(z, labels):
        """Alignment: average distance between same-class embeddings (lower = better).

        Measures how close positive pairs are in embedding space.
        """
        n = _num_rows(z)
        embs = [list(_tensor_row(z, i)) for i in range(n)]

        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    dist = sum((a - b) ** 2 for a, b in zip(embs[i], embs[j]))
                    total_dist += dist
                    count += 1

        return total_dist / max(count, 1)

    @staticmethod
    def uniformity(z, t=2.0):
        """Uniformity: log of average pairwise Gaussian potential (lower = more uniform).

        Measures how uniformly distributed embeddings are on hypersphere.
        """
        n = _num_rows(z)
        embs = [_l2_normalize(list(_tensor_row(z, i))) for i in range(n)]

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = sum((a - b) ** 2 for a, b in zip(embs[i], embs[j]))
                total += math.exp(-t * dist_sq)
                count += 1

        if count == 0:
            return 0.0
        return math.log(total / count + 1e-12)

    @staticmethod
    def silhouette_score(z, labels):
        """Simplified silhouette score for cluster quality.

        Returns score in [-1, 1], higher is better.
        """
        n = _num_rows(z)
        embs = [list(_tensor_row(z, i)) for i in range(n)]

        classes = sorted(set(labels))
        if len(classes) < 2:
            return 0.0

        scores = []
        for i in range(n):
            # a(i): mean distance to same-class points
            same_dists = []
            for j in range(n):
                if j != i and labels[j] == labels[i]:
                    d = math.sqrt(sum((a - b) ** 2 for a, b in zip(embs[i], embs[j])))
                    same_dists.append(d)
            a_i = sum(same_dists) / max(len(same_dists), 1)

            # b(i): min mean distance to other-class points
            b_i = float('inf')
            for c in classes:
                if c == labels[i]:
                    continue
                other_dists = []
                for j in range(n):
                    if labels[j] == c:
                        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(embs[i], embs[j])))
                        other_dists.append(d)
                if other_dists:
                    mean_d = sum(other_dists) / len(other_dists)
                    b_i = min(b_i, mean_d)

            if b_i == float('inf'):
                scores.append(0.0)
            else:
                scores.append((b_i - a_i) / max(a_i, b_i, 1e-12))

        return sum(scores) / len(scores)

    @staticmethod
    def cosine_similarity_matrix(z):
        """Compute pairwise cosine similarity matrix."""
        n = _num_rows(z)
        embs = [_l2_normalize(list(_tensor_row(z, i))) for i in range(n)]
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = _dot(embs[i], embs[j])
        return matrix

    @staticmethod
    def nearest_neighbors(z, labels, k=5):
        """Compute k-nearest-neighbor accuracy (same label among neighbors).

        Returns accuracy float in [0, 1].
        """
        n = _num_rows(z)
        embs = [list(_tensor_row(z, i)) for i in range(n)]

        correct = 0
        for i in range(n):
            dists = []
            for j in range(n):
                if j != i:
                    d = sum((a - b) ** 2 for a, b in zip(embs[i], embs[j]))
                    dists.append((d, labels[j]))
            dists.sort(key=lambda x: x[0])
            neighbors = [lab for _, lab in dists[:k]]
            # Majority vote
            counts = {}
            for lab in neighbors:
                counts[lab] = counts.get(lab, 0) + 1
            pred = max(counts, key=counts.get)
            if pred == labels[i]:
                correct += 1

        return correct / max(n, 1)

    @staticmethod
    def intra_inter_ratio(z, labels):
        """Ratio of intra-class to inter-class distances (lower = better separation)."""
        n = _num_rows(z)
        embs = [list(_tensor_row(z, i)) for i in range(n)]

        intra_sum = 0.0
        intra_count = 0
        inter_sum = 0.0
        inter_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                d = sum((a - b) ** 2 for a, b in zip(embs[i], embs[j]))
                if labels[i] == labels[j]:
                    intra_sum += d
                    intra_count += 1
                else:
                    inter_sum += d
                    inter_count += 1

        intra_avg = intra_sum / max(intra_count, 1)
        inter_avg = inter_sum / max(inter_count, 1)
        return intra_avg / max(inter_avg, 1e-12)


# ── 10. ContrastiveMetrics ──────────────────────────────────────────

class ContrastiveMetrics:
    """Static evaluation metrics for contrastive learning."""

    @staticmethod
    def linear_probe_accuracy(framework, X_train, y_train, X_test, y_test,
                               repr_dim, num_classes, epochs=50, lr=0.01, seed=42):
        """End-to-end linear probe evaluation.

        Pre-extracts representations, trains linear classifier, evaluates.
        Returns (train_acc, test_acc).
        """
        framework.eval() if hasattr(framework, 'eval') else None

        repr_train = framework.get_representations(X_train)
        repr_test = framework.get_representations(X_test)

        evaluator = LinearEvaluator(repr_dim, num_classes, lr=lr, seed=seed)
        evaluator.fit(repr_train, y_train, epochs=epochs)

        train_acc = evaluator.evaluate(repr_train, y_train)
        test_acc = evaluator.evaluate(repr_test, y_test)

        return train_acc, test_acc

    @staticmethod
    def representation_quality(framework, X, labels):
        """Comprehensive representation quality report.

        Returns dict with alignment, uniformity, silhouette, knn_accuracy, intra_inter.
        """
        framework.eval() if hasattr(framework, 'eval') else None
        z = framework.get_representations(X)

        return {
            'alignment': RepresentationAnalyzer.alignment(z, labels),
            'uniformity': RepresentationAnalyzer.uniformity(z),
            'silhouette': RepresentationAnalyzer.silhouette_score(z, labels),
            'knn_accuracy': RepresentationAnalyzer.nearest_neighbors(z, labels, k=3),
            'intra_inter_ratio': RepresentationAnalyzer.intra_inter_ratio(z, labels),
        }

    @staticmethod
    def training_summary(history):
        """Summarize training history.

        Returns dict with start_loss, end_loss, best_loss, improvement, converged.
        """
        if not history:
            return {'start_loss': 0, 'end_loss': 0, 'best_loss': 0,
                    'improvement': 0, 'converged': False}

        return {
            'start_loss': history[0],
            'end_loss': history[-1],
            'best_loss': min(history),
            'improvement': history[0] - history[-1],
            'converged': len(history) > 1 and abs(history[-1] - history[-2]) < 0.01,
        }

    @staticmethod
    def compare_frameworks(results):
        """Compare results from multiple frameworks.

        Args:
            results: dict of {name: quality_dict}
        Returns:
            dict with rankings per metric and overall winner.
        """
        metrics = ['alignment', 'uniformity', 'silhouette', 'knn_accuracy', 'intra_inter_ratio']
        # Lower is better for: alignment, uniformity, intra_inter_ratio
        # Higher is better for: silhouette, knn_accuracy
        lower_better = {'alignment', 'uniformity', 'intra_inter_ratio'}

        rankings = {}
        scores = {name: 0 for name in results}

        for metric in metrics:
            vals = [(name, results[name].get(metric, 0)) for name in results]
            if metric in lower_better:
                vals.sort(key=lambda x: x[1])
            else:
                vals.sort(key=lambda x: -x[1])

            rankings[metric] = [name for name, _ in vals]
            for rank, (name, _) in enumerate(vals):
                scores[name] += rank

        overall = sorted(scores.items(), key=lambda x: x[1])
        rankings['overall'] = [name for name, _ in overall]

        return rankings
