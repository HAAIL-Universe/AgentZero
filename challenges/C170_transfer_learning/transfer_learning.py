"""
C170: Transfer Learning
Composing C140 Neural Network for transfer learning workflows.

Components:
- PretrainedModel: Named layer groups (backbone/head) with freeze/unfreeze
- FeatureExtractor: Extract intermediate representations from any layer
- FineTuner: Discriminative learning rates, gradual unfreezing
- DomainAdapter: Domain adaptation via feature alignment (MMD, CORAL)
- KnowledgeDistiller: Teacher-student knowledge distillation
- ModelRegistry: Save/load/catalog pretrained models
- TransferTrainer: Orchestrates full transfer learning workflows
- DataAugmenter: Simple data augmentation for training
"""

import math
import random
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Sequential, Dense, Activation, Dropout, BatchNorm,
    Layer, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    SGD, Adam, fit, evaluate, predict_classes, accuracy,
    save_weights, load_weights, build_model, softmax, softmax_batch,
    xavier_init, he_init, train_step, one_hot, normalize, train_test_split
)


# ============================================================
# Layer Freezing Utilities
# ============================================================

def freeze_layer(layer):
    """Freeze a layer -- its parameters won't be updated."""
    layer._frozen = True


def unfreeze_layer(layer):
    """Unfreeze a layer -- its parameters will be updated."""
    layer._frozen = False


def is_frozen(layer):
    """Check if a layer is frozen."""
    return getattr(layer, '_frozen', False)


def count_frozen_params(model):
    """Count frozen parameters in a model."""
    total = 0
    for layer in model.layers:
        if is_frozen(layer):
            for pt in layer.get_params():
                t = pt[0]
                if len(t.shape) == 1:
                    total += t.shape[0]
                else:
                    total += t.shape[0] * t.shape[1]
    return total


def count_trainable_params(model):
    """Count trainable (unfrozen) parameters."""
    total = 0
    for layer in model.layers:
        if not is_frozen(layer):
            for pt in layer.get_params():
                t = pt[0]
                if len(t.shape) == 1:
                    total += t.shape[0]
                else:
                    total += t.shape[0] * t.shape[1]
    return total


# ============================================================
# PretrainedModel: Named layer groups with freeze control
# ============================================================

class PretrainedModel:
    """
    Wraps a Sequential model with named layer groups.
    Supports freezing/unfreezing by group name.
    """

    def __init__(self, model, groups=None):
        """
        model: Sequential model
        groups: dict mapping group name -> list of layer indices
                e.g. {'backbone': [0,1,2,3], 'head': [4,5]}
        """
        self.model = model
        self.groups = groups or {}
        self._metadata = {}

    @staticmethod
    def from_sequential(model, backbone_end=None):
        """
        Create PretrainedModel from Sequential, auto-splitting into
        backbone and head groups.
        backbone_end: index of last backbone layer (exclusive).
                      If None, all but last Dense+Activation are backbone.
        """
        layers = model.layers
        if backbone_end is None:
            # Find the last Dense layer -- everything before it is backbone
            last_dense = 0
            for i, layer in enumerate(layers):
                if isinstance(layer, Dense):
                    last_dense = i
            backbone_end = last_dense

        backbone_indices = list(range(backbone_end))
        head_indices = list(range(backbone_end, len(layers)))

        groups = {
            'backbone': backbone_indices,
            'head': head_indices
        }
        return PretrainedModel(model, groups)

    def freeze_group(self, group_name):
        """Freeze all layers in a named group."""
        if group_name not in self.groups:
            raise ValueError(f"Unknown group: {group_name}")
        for idx in self.groups[group_name]:
            freeze_layer(self.model.layers[idx])

    def unfreeze_group(self, group_name):
        """Unfreeze all layers in a named group."""
        if group_name not in self.groups:
            raise ValueError(f"Unknown group: {group_name}")
        for idx in self.groups[group_name]:
            unfreeze_layer(self.model.layers[idx])

    def freeze_all(self):
        """Freeze all layers."""
        for layer in self.model.layers:
            freeze_layer(layer)

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for layer in self.model.layers:
            unfreeze_layer(layer)

    def get_group_layers(self, group_name):
        """Get layers in a named group."""
        if group_name not in self.groups:
            raise ValueError(f"Unknown group: {group_name}")
        return [self.model.layers[i] for i in self.groups[group_name]]

    def add_group(self, name, indices):
        """Add a new named group."""
        self.groups[name] = indices

    def replace_head(self, new_layers):
        """
        Replace head layers with new ones.
        Returns new PretrainedModel with updated groups.
        """
        if 'head' not in self.groups:
            raise ValueError("No 'head' group defined")

        head_start = min(self.groups['head'])
        # Remove old head layers
        backbone_layers = self.model.layers[:head_start]
        # Add new head layers
        all_layers = backbone_layers + new_layers

        new_model = Sequential(all_layers)
        new_head_indices = list(range(head_start, len(all_layers)))
        new_groups = dict(self.groups)
        new_groups['head'] = new_head_indices

        result = PretrainedModel(new_model, new_groups)
        result._metadata = dict(self._metadata)
        return result

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, x):
        return self.model.predict(x)

    def summary(self):
        """Summary with freeze status."""
        lines = ["Layer | Group | Frozen | Params"]
        lines.append("-" * 50)

        # Invert groups for lookup
        idx_to_group = {}
        for gname, indices in self.groups.items():
            for idx in indices:
                idx_to_group[idx] = gname

        total = 0
        frozen = 0
        for i, layer in enumerate(self.model.layers):
            name = layer.__class__.__name__
            group = idx_to_group.get(i, '-')
            is_frz = is_frozen(layer)
            params = 0
            for pt in layer.get_params():
                t = pt[0]
                if len(t.shape) == 1:
                    params += t.shape[0]
                else:
                    params += t.shape[0] * t.shape[1]
            total += params
            if is_frz:
                frozen += params
            if isinstance(layer, Dense):
                lines.append(f"{name}({layer.input_size},{layer.output_size}) | {group} | {is_frz} | {params}")
            elif isinstance(layer, Activation):
                lines.append(f"{name}({layer.name}) | {group} | {is_frz} | 0")
            else:
                lines.append(f"{name} | {group} | {is_frz} | {params}")

        lines.append(f"Total: {total}, Frozen: {frozen}, Trainable: {total - frozen}")
        return "\n".join(lines)


# ============================================================
# FeatureExtractor: Extract intermediate representations
# ============================================================

class FeatureExtractor:
    """
    Extracts features from intermediate layers of a model.
    Useful for using pretrained models as feature extractors.
    """

    def __init__(self, model, extract_at=None):
        """
        model: Sequential or PretrainedModel
        extract_at: layer index or list of indices to extract from.
                    If None, extracts from last layer before head.
        """
        if isinstance(model, PretrainedModel):
            self.model = model.model
            self._pretrained = model
        else:
            self.model = model
            self._pretrained = None

        if extract_at is None:
            # Default: extract from last backbone layer
            if self._pretrained and 'backbone' in self._pretrained.groups:
                self.extract_at = [max(self._pretrained.groups['backbone'])]
            else:
                # Last layer with params before the final layer
                last_param = 0
                for i, layer in enumerate(self.model.layers[:-1]):
                    if layer.get_params():
                        last_param = i
                self.extract_at = [last_param]
        elif isinstance(extract_at, int):
            self.extract_at = [extract_at]
        else:
            self.extract_at = list(extract_at)

    def extract(self, x):
        """
        Run forward pass and return features at specified layers.
        Returns dict mapping layer_index -> Tensor.
        """
        self.model.eval()
        features = {}
        out = x
        for i, layer in enumerate(self.model.layers):
            out = layer.forward(out)
            if i in self.extract_at:
                features[i] = out.copy() if isinstance(out, Tensor) else out
        self.model.train()
        return features

    def extract_single(self, x):
        """Extract features from the primary extraction point."""
        features = self.extract(x)
        key = self.extract_at[-1]
        return features[key]

    def extract_up_to(self, x, layer_idx):
        """Run forward pass up to (inclusive) a given layer."""
        self.model.eval()
        out = x
        for i in range(layer_idx + 1):
            out = self.model.layers[i].forward(out)
        self.model.train()
        return out


# ============================================================
# FineTuner: Discriminative LR + Gradual Unfreezing
# ============================================================

class FineTuner:
    """
    Fine-tuning strategies for transfer learning:
    - Discriminative learning rates (lower LR for earlier layers)
    - Gradual unfreezing (unfreeze layers one at a time)
    - Warm-up scheduling
    """

    def __init__(self, pretrained_model, base_lr=0.001, lr_mult=0.1):
        """
        pretrained_model: PretrainedModel
        base_lr: learning rate for head layers
        lr_mult: multiplier for backbone layers (e.g., 0.1 = 10x lower)
        """
        self.pretrained = pretrained_model
        self.base_lr = base_lr
        self.lr_mult = lr_mult
        self._unfreeze_schedule = []
        self._current_epoch = 0

    def set_discriminative_lrs(self):
        """
        Assign per-layer learning rates.
        Head gets base_lr, backbone layers get progressively smaller LRs.
        """
        model = self.pretrained.model
        lrs = {}

        if 'backbone' in self.pretrained.groups:
            backbone_indices = self.pretrained.groups['backbone']
            head_indices = self.pretrained.groups.get('head', [])

            n_backbone = len(backbone_indices)
            for i, idx in enumerate(backbone_indices):
                # Earlier layers get smaller LR
                depth_factor = self.lr_mult ** (n_backbone - i)
                lrs[idx] = self.base_lr * depth_factor

            for idx in head_indices:
                lrs[idx] = self.base_lr
        else:
            n = len(model.layers)
            for i in range(n):
                depth_factor = self.lr_mult ** (n - 1 - i)
                lrs[i] = self.base_lr * depth_factor

        self._layer_lrs = lrs
        return lrs

    def setup_gradual_unfreeze(self, epochs_per_layer=1):
        """
        Set up gradual unfreezing schedule.
        Unfreezes from the head backward, one group/layer at a time.
        """
        model = self.pretrained.model

        # Freeze everything first
        self.pretrained.freeze_all()

        # Build schedule: unfreeze from end to beginning
        if 'head' in self.pretrained.groups:
            # First unfreeze head, then backbone layers in reverse
            schedule = [('head', 0)]
            if 'backbone' in self.pretrained.groups:
                backbone = self.pretrained.groups['backbone']
                # Group backbone into chunks
                param_layers = [i for i in backbone
                               if self.pretrained.model.layers[i].get_params()]
                for i, idx in enumerate(reversed(param_layers)):
                    epoch = (i + 1) * epochs_per_layer
                    schedule.append(((idx,), epoch))
        else:
            param_layers = [i for i, l in enumerate(model.layers)
                           if l.get_params()]
            schedule = []
            for i, idx in enumerate(reversed(param_layers)):
                epoch = i * epochs_per_layer
                schedule.append(((idx,), epoch))

        self._unfreeze_schedule = schedule
        return schedule

    def step_unfreeze(self, epoch):
        """
        Called each epoch to check if layers should be unfrozen.
        """
        self._current_epoch = epoch
        for item in self._unfreeze_schedule:
            target, trigger_epoch = item
            if epoch >= trigger_epoch:
                if isinstance(target, str):
                    # Group name
                    self.pretrained.unfreeze_group(target)
                else:
                    # Tuple of indices
                    for idx in target:
                        unfreeze_layer(self.pretrained.model.layers[idx])

    def discriminative_step(self, layers):
        """
        Update parameters with per-layer learning rates.
        Call this instead of optimizer.step() for discriminative LR.
        """
        if not hasattr(self, '_layer_lrs'):
            self.set_discriminative_lrs()

        model = self.pretrained.model
        for layer_idx, layer in enumerate(model.layers):
            if is_frozen(layer):
                continue
            lr = self._layer_lrs.get(layer_idx, self.base_lr)
            for param_tuple in layer.get_params():
                tensor, grad, name = param_tuple
                if grad is None:
                    continue
                # Simple SGD update with discriminative LR
                if len(tensor.shape) == 1:
                    for k in range(len(tensor.data)):
                        tensor.data[k] -= lr * grad.data[k]
                else:
                    for i in range(tensor.shape[0]):
                        for j in range(tensor.shape[1]):
                            tensor.data[i][j] -= lr * grad.data[i][j]


# ============================================================
# Frozen-aware optimizer wrapper
# ============================================================

class FrozenAwareOptimizer:
    """Wraps an optimizer to skip frozen layers."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, layers):
        """Update only unfrozen layers."""
        unfrozen = [l for l in layers if not is_frozen(l)]
        if unfrozen:
            self.optimizer.step(unfrozen)


# ============================================================
# DomainAdapter: Domain Adaptation
# ============================================================

class DomainAdapter:
    """
    Domain adaptation via feature distribution alignment.
    Supports:
    - MMD (Maximum Mean Discrepancy) -- aligns feature distributions
    - CORAL (Correlation Alignment) -- aligns covariance matrices
    - Domain-adversarial training (gradient reversal)
    """

    def __init__(self, method='mmd', lambda_adapt=1.0):
        """
        method: 'mmd' | 'coral' | 'adversarial'
        lambda_adapt: weight for adaptation loss
        """
        self.method = method
        self.lambda_adapt = lambda_adapt
        self._discriminator = None

    def compute_mmd(self, source_features, target_features):
        """
        Compute Maximum Mean Discrepancy between two feature sets.
        Uses linear kernel: MMD^2 = ||mean(source) - mean(target)||^2
        """
        if len(source_features.shape) == 1:
            s_mean = sum(source_features.data) / len(source_features.data)
            t_mean = sum(target_features.data) / len(target_features.data)
            return (s_mean - t_mean) ** 2

        # 2D: compute per-feature means
        s_rows, s_cols = source_features.shape
        t_rows, t_cols = target_features.shape

        mmd = 0.0
        for j in range(s_cols):
            s_mean = sum(source_features.data[i][j] for i in range(s_rows)) / s_rows
            t_mean = sum(target_features.data[i][j] for i in range(t_rows)) / t_rows
            mmd += (s_mean - t_mean) ** 2
        return mmd

    def compute_coral(self, source_features, target_features):
        """
        Compute CORAL loss: ||C_s - C_t||_F^2 / (4 * d^2)
        Where C_s, C_t are covariance matrices.
        """
        if len(source_features.shape) == 1:
            return 0.0

        s_rows, d = source_features.shape
        t_rows, _ = target_features.shape

        # Source covariance
        s_mean = [sum(source_features.data[i][j] for i in range(s_rows)) / s_rows
                  for j in range(d)]
        t_mean = [sum(target_features.data[i][j] for i in range(t_rows)) / t_rows
                  for j in range(d)]

        # Compute covariance matrices
        cs = [[0.0] * d for _ in range(d)]
        ct = [[0.0] * d for _ in range(d)]

        for i in range(s_rows):
            for j in range(d):
                for k in range(d):
                    cs[j][k] += (source_features.data[i][j] - s_mean[j]) * \
                                (source_features.data[i][k] - s_mean[k])
        for j in range(d):
            for k in range(d):
                cs[j][k] /= max(s_rows - 1, 1)

        for i in range(t_rows):
            for j in range(d):
                for k in range(d):
                    ct[j][k] += (target_features.data[i][j] - t_mean[j]) * \
                                (target_features.data[i][k] - t_mean[k])
        for j in range(d):
            for k in range(d):
                ct[j][k] /= max(t_rows - 1, 1)

        # Frobenius norm of difference
        frob = 0.0
        for j in range(d):
            for k in range(d):
                frob += (cs[j][k] - ct[j][k]) ** 2

        return frob / (4.0 * d * d)

    def adaptation_loss(self, source_features, target_features):
        """Compute adaptation loss using selected method."""
        if self.method == 'mmd':
            return self.lambda_adapt * self.compute_mmd(source_features, target_features)
        elif self.method == 'coral':
            return self.lambda_adapt * self.compute_coral(source_features, target_features)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def compute_mmd_gradient(self, source_features, target_features):
        """
        Gradient of MMD loss w.r.t. source features.
        d/d(s_i) ||mean(s) - mean(t)||^2 = 2*(mean(s) - mean(t)) / n_s
        """
        if len(source_features.shape) == 1:
            n_s = len(source_features.data)
            n_t = len(target_features.data)
            s_mean = sum(source_features.data) / n_s
            t_mean = sum(target_features.data) / n_t
            grad_val = 2.0 * (s_mean - t_mean) / n_s * self.lambda_adapt
            return Tensor([grad_val] * n_s)

        s_rows, d = source_features.shape
        t_rows, _ = target_features.shape

        s_mean = [sum(source_features.data[i][j] for i in range(s_rows)) / s_rows
                  for j in range(d)]
        t_mean = [sum(target_features.data[i][j] for i in range(t_rows)) / t_rows
                  for j in range(d)]

        grad = [[2.0 * (s_mean[j] - t_mean[j]) / s_rows * self.lambda_adapt
                 for j in range(d)]
                for _ in range(s_rows)]
        return Tensor(grad)


# ============================================================
# KnowledgeDistiller: Teacher-Student Training
# ============================================================

class KnowledgeDistiller:
    """
    Knowledge distillation: train a smaller student network
    to mimic a larger teacher network's outputs.

    Loss = alpha * hard_loss + (1 - alpha) * soft_loss
    where soft_loss uses temperature-scaled softmax.
    """

    def __init__(self, teacher, student, temperature=3.0, alpha=0.5):
        """
        teacher: trained model (will be in eval mode)
        student: model to train
        temperature: softmax temperature for soft targets
        alpha: weight for hard loss (1-alpha for soft loss)
        """
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self._hard_loss_fn = None
        self._history = {'loss': [], 'hard_loss': [], 'soft_loss': []}

    def soft_softmax(self, logits, temperature):
        """Temperature-scaled softmax."""
        if len(logits.shape) == 1:
            scaled = [v / temperature for v in logits.data]
            return Tensor(softmax(scaled))
        return Tensor([softmax([v / temperature for v in row])
                       for row in logits.data])

    def soft_cross_entropy(self, student_logits, teacher_logits):
        """
        KL divergence between teacher and student soft distributions.
        KL(teacher || student) with temperature scaling.
        """
        T = self.temperature
        teacher_soft = self.soft_softmax(teacher_logits, T)
        student_soft = self.soft_softmax(student_logits, T)

        if len(student_logits.shape) == 1:
            n = len(student_logits.data)
            loss = 0.0
            for j in range(n):
                t_val = max(teacher_soft.data[j], 1e-15)
                s_val = max(student_soft.data[j], 1e-15)
                loss += t_val * math.log(t_val / s_val)
            return loss * (T * T)

        batch_size = student_logits.shape[0]
        total = 0.0
        for i in range(batch_size):
            for j in range(student_logits.shape[1]):
                t_val = max(teacher_soft.data[i][j], 1e-15)
                s_val = max(student_soft.data[i][j], 1e-15)
                total += t_val * math.log(t_val / s_val)
        return total / batch_size * (T * T)

    def soft_cross_entropy_gradient(self, student_logits, teacher_logits):
        """
        Gradient of soft CE loss w.r.t. student logits.
        d/d(z_s) = T * (softmax(z_s/T) - softmax(z_t/T))
        """
        T = self.temperature
        teacher_soft = self.soft_softmax(teacher_logits, T)
        student_soft = self.soft_softmax(student_logits, T)

        if len(student_logits.shape) == 1:
            return Tensor([T * (student_soft.data[j] - teacher_soft.data[j])
                          for j in range(len(student_logits.data))])

        batch_size = student_logits.shape[0]
        cols = student_logits.shape[1]
        return Tensor([[T * (student_soft.data[i][j] - teacher_soft.data[i][j]) / batch_size
                       for j in range(cols)]
                      for i in range(batch_size)])

    def distill_step(self, x_batch, y_batch, hard_loss_fn, optimizer):
        """
        Single distillation training step.
        Returns (total_loss, hard_loss, soft_loss).
        """
        # Teacher forward (eval mode, no gradients needed)
        self.teacher.eval()
        teacher_output = self.teacher.forward(x_batch)

        # Student forward
        self.student.train()
        student_output = self.student.forward(x_batch)

        # Hard loss (standard task loss)
        hard_loss = hard_loss_fn.forward(student_output, y_batch)
        hard_grad = hard_loss_fn.backward(student_output, y_batch)

        # Soft loss (distillation loss)
        soft_loss = self.soft_cross_entropy(student_output, teacher_output)
        soft_grad = self.soft_cross_entropy_gradient(student_output, teacher_output)

        # Combined gradient
        alpha = self.alpha
        if len(hard_grad.shape) == 1:
            combined = Tensor([alpha * hard_grad.data[j] + (1 - alpha) * soft_grad.data[j]
                              for j in range(len(hard_grad.data))])
        else:
            rows, cols = hard_grad.shape
            combined = Tensor([[alpha * hard_grad.data[i][j] + (1 - alpha) * soft_grad.data[i][j]
                               for j in range(cols)]
                              for i in range(rows)])

        # Backward + update
        self.student.backward(combined)
        optimizer.step(self.student.get_trainable_layers())

        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
        return total_loss, hard_loss, soft_loss

    def distill(self, X, Y, hard_loss_fn, optimizer, epochs=50,
                batch_size=None, verbose=False):
        """
        Full distillation training loop.
        Returns training history dict.
        """
        num_samples = X.shape[0] if len(X.shape) == 2 else 1
        if batch_size is None:
            batch_size = num_samples

        rng = random.Random(42)

        for epoch in range(epochs):
            indices = list(range(num_samples))
            rng.shuffle(indices)

            epoch_loss = 0.0
            epoch_hard = 0.0
            epoch_soft = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]

                if len(X.shape) == 2:
                    x_batch = Tensor([X.data[i] for i in batch_idx])
                else:
                    x_batch = X

                if isinstance(Y, list):
                    y_batch = [Y[i] for i in batch_idx]
                elif isinstance(Y, Tensor) and len(Y.shape) == 2:
                    y_batch = Tensor([Y.data[i] for i in batch_idx])
                else:
                    y_batch = Y

                total, hard, soft = self.distill_step(
                    x_batch, y_batch, hard_loss_fn, optimizer
                )
                epoch_loss += total
                epoch_hard += hard
                epoch_soft += soft
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_hard = epoch_hard / n_batches
            avg_soft = epoch_soft / n_batches

            self._history['loss'].append(avg_loss)
            self._history['hard_loss'].append(avg_hard)
            self._history['soft_loss'].append(avg_soft)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f} "
                      f"(hard: {avg_hard:.6f}, soft: {avg_soft:.6f})")

        return self._history


# ============================================================
# ModelRegistry: Save/Load/Catalog Pretrained Models
# ============================================================

class ModelRegistry:
    """
    Registry for pretrained models.
    Stores model architectures and weights for reuse.
    """

    def __init__(self):
        self._models = {}  # name -> {architecture, weights, metadata}

    def register(self, name, model, metadata=None):
        """
        Register a pretrained model.
        model: Sequential or PretrainedModel
        """
        if isinstance(model, PretrainedModel):
            seq = model.model
            groups = model.groups
        else:
            seq = model
            groups = None

        # Save architecture description
        arch = []
        for layer in seq.layers:
            if isinstance(layer, Dense):
                arch.append({
                    'type': 'Dense',
                    'input_size': layer.input_size,
                    'output_size': layer.output_size,
                    'bias': layer.use_bias
                })
            elif isinstance(layer, Activation):
                arch.append({'type': 'Activation', 'name': layer.name})
            elif isinstance(layer, Dropout):
                arch.append({'type': 'Dropout', 'rate': layer.rate})
            elif isinstance(layer, BatchNorm):
                arch.append({
                    'type': 'BatchNorm',
                    'num_features': layer.num_features
                })

        weights = save_weights(seq)

        self._models[name] = {
            'architecture': arch,
            'weights': weights,
            'groups': groups,
            'metadata': metadata or {}
        }

    def load(self, name):
        """Load a registered model by name. Returns PretrainedModel."""
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")

        entry = self._models[name]
        arch = entry['architecture']

        # Rebuild model
        layers = []
        for spec in arch:
            if spec['type'] == 'Dense':
                layers.append(Dense(
                    spec['input_size'], spec['output_size'],
                    bias=spec.get('bias', True)
                ))
            elif spec['type'] == 'Activation':
                layers.append(Activation(spec['name']))
            elif spec['type'] == 'Dropout':
                layers.append(Dropout(spec['rate']))
            elif spec['type'] == 'BatchNorm':
                layers.append(BatchNorm(spec['num_features']))

        model = Sequential(layers)
        load_weights(model, entry['weights'])

        groups = entry.get('groups')
        if groups is None:
            pm = PretrainedModel.from_sequential(model)
        else:
            pm = PretrainedModel(model, groups)
        pm._metadata = entry.get('metadata', {})
        return pm

    def list_models(self):
        """List all registered model names."""
        return list(self._models.keys())

    def get_metadata(self, name):
        """Get metadata for a registered model."""
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        return self._models[name]['metadata']

    def remove(self, name):
        """Remove a model from the registry."""
        if name in self._models:
            del self._models[name]

    def export_weights(self, name):
        """Export weights dict for a registered model."""
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        return self._models[name]['weights']

    def import_weights(self, name, weights):
        """Import weights into an existing registered model."""
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        self._models[name]['weights'] = weights


# ============================================================
# TransferTrainer: Orchestrate Transfer Learning Workflows
# ============================================================

class TransferTrainer:
    """
    Orchestrates full transfer learning workflow:
    1. Load pretrained model
    2. Replace head for new task
    3. Freeze backbone
    4. Train head
    5. Optionally fine-tune with discriminative LR
    """

    def __init__(self, pretrained, new_head_layers=None):
        """
        pretrained: PretrainedModel
        new_head_layers: list of Layer objects for the new head
        """
        if new_head_layers is not None:
            self.model = pretrained.replace_head(new_head_layers)
        else:
            self.model = pretrained
        self.fine_tuner = None
        self._history = {
            'phase1_loss': [],
            'phase2_loss': []
        }

    def phase1_train_head(self, X, Y, loss_fn, optimizer, epochs=50,
                          batch_size=None, verbose=False):
        """
        Phase 1: Train only the head with frozen backbone.
        """
        # Freeze backbone, unfreeze head
        if 'backbone' in self.model.groups:
            self.model.freeze_group('backbone')
        if 'head' in self.model.groups:
            self.model.unfreeze_group('head')

        wrapped_opt = FrozenAwareOptimizer(optimizer)
        history = fit(self.model.model, X, Y, loss_fn, wrapped_opt,
                     epochs=epochs, batch_size=batch_size, verbose=verbose)

        self._history['phase1_loss'] = history['loss']
        return history

    def phase2_fine_tune(self, X, Y, loss_fn, base_lr=0.0001,
                         lr_mult=0.1, epochs=20, batch_size=None,
                         verbose=False, gradual=False, epochs_per_unfreeze=5):
        """
        Phase 2: Fine-tune with discriminative learning rates.
        Optionally uses gradual unfreezing.
        """
        self.fine_tuner = FineTuner(self.model, base_lr=base_lr, lr_mult=lr_mult)
        self.fine_tuner.set_discriminative_lrs()

        if gradual:
            self.fine_tuner.setup_gradual_unfreeze(epochs_per_layer=epochs_per_unfreeze)
        else:
            # Unfreeze all
            self.model.unfreeze_all()

        num_samples = X.shape[0] if len(X.shape) == 2 else 1
        if batch_size is None:
            batch_size = num_samples
        rng = random.Random(42)

        for epoch in range(epochs):
            if gradual:
                self.fine_tuner.step_unfreeze(epoch)

            self.model.model.train()
            indices = list(range(num_samples))
            rng.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = indices[start:end]

                if len(X.shape) == 2:
                    x_batch = Tensor([X.data[i] for i in batch_idx])
                else:
                    x_batch = X

                if isinstance(Y, list):
                    y_batch = [Y[i] for i in batch_idx]
                elif isinstance(Y, Tensor) and len(Y.shape) == 2:
                    y_batch = Tensor([Y.data[i] for i in batch_idx])
                else:
                    y_batch = Y

                # Forward
                output = self.model.model.forward(x_batch)
                loss = loss_fn.forward(output, y_batch)
                grad = loss_fn.backward(output, y_batch)
                self.model.model.backward(grad)

                # Discriminative update
                self.fine_tuner.discriminative_step(None)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self._history['phase2_loss'].append(avg_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Fine-tune epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}")

        return self._history

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y, loss_fn):
        return evaluate(self.model.model, X, Y, loss_fn)


# ============================================================
# DataAugmenter: Simple augmentation for training
# ============================================================

class DataAugmenter:
    """
    Data augmentation for tabular/vector data.
    Supports: noise injection, mixup, cutout, random scaling.
    """

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def add_noise(self, X, std=0.1):
        """Add Gaussian noise to features."""
        if len(X.shape) == 1:
            return Tensor([x + self.rng.gauss(0, std) for x in X.data])
        return Tensor([[x + self.rng.gauss(0, std) for x in row]
                       for row in X.data])

    def mixup(self, X, Y, alpha=0.2):
        """
        Mixup augmentation: blend pairs of samples.
        Returns (X_mixed, Y_mixed).
        Y must be Tensor (one-hot or regression targets).
        """
        if len(X.shape) == 1:
            return X.copy(), Y.copy() if isinstance(Y, Tensor) else Y

        n = X.shape[0]
        indices = list(range(n))
        self.rng.shuffle(indices)

        lam = self._beta_sample(alpha, alpha)

        x_mixed = []
        for i in range(n):
            j = indices[i]
            row = [lam * X.data[i][k] + (1 - lam) * X.data[j][k]
                   for k in range(X.shape[1])]
            x_mixed.append(row)

        if isinstance(Y, Tensor) and len(Y.shape) == 2:
            y_mixed = []
            for i in range(n):
                j = indices[i]
                row = [lam * Y.data[i][k] + (1 - lam) * Y.data[j][k]
                       for k in range(Y.shape[1])]
                y_mixed.append(row)
            return Tensor(x_mixed), Tensor(y_mixed)

        return Tensor(x_mixed), Y

    def cutout(self, X, n_features=1):
        """
        Cutout: zero out random features.
        n_features: number of features to zero out per sample.
        """
        if len(X.shape) == 1:
            data = X.data[:]
            d = len(data)
            for _ in range(min(n_features, d)):
                idx = self.rng.randint(0, d - 1)
                data[idx] = 0.0
            return Tensor(data)

        result = []
        d = X.shape[1]
        for row in X.data:
            new_row = row[:]
            for _ in range(min(n_features, d)):
                idx = self.rng.randint(0, d - 1)
                new_row[idx] = 0.0
            result.append(new_row)
        return Tensor(result)

    def random_scale(self, X, low=0.9, high=1.1):
        """Scale each feature by a random factor."""
        if len(X.shape) == 1:
            return Tensor([x * self.rng.uniform(low, high) for x in X.data])

        d = X.shape[1]
        scales = [self.rng.uniform(low, high) for _ in range(d)]
        return Tensor([[row[j] * scales[j] for j in range(d)]
                       for row in X.data])

    def augment_batch(self, X, Y, methods=None, **kwargs):
        """
        Apply multiple augmentation methods to a batch.
        methods: list of method names (default: ['noise', 'scale'])
        Returns (X_aug, Y_aug) with original + augmented data.
        """
        if methods is None:
            methods = ['noise']

        x_all = [row[:] if isinstance(row, list) else row
                 for row in (X.data if len(X.shape) == 2 else [X.data])]

        if isinstance(Y, Tensor) and len(Y.shape) == 2:
            y_all = [row[:] for row in Y.data]
        elif isinstance(Y, list):
            y_all = Y[:]
        else:
            y_all = Y

        for method in methods:
            if method == 'noise':
                x_aug = self.add_noise(X, std=kwargs.get('noise_std', 0.1))
            elif method == 'scale':
                x_aug = self.random_scale(X,
                                          low=kwargs.get('scale_low', 0.9),
                                          high=kwargs.get('scale_high', 1.1))
            elif method == 'cutout':
                x_aug = self.cutout(X, n_features=kwargs.get('n_cutout', 1))
            else:
                continue

            if len(x_aug.shape) == 2:
                x_all.extend(x_aug.data)
            else:
                x_all.append(x_aug.data)

            if isinstance(Y, Tensor) and len(Y.shape) == 2:
                y_all.extend(Y.data)
            elif isinstance(Y, list):
                y_all.extend(Y)

        if isinstance(y_all, list) and len(y_all) > 0 and isinstance(y_all[0], list):
            return Tensor(x_all), Tensor(y_all)
        return Tensor(x_all), y_all

    def _beta_sample(self, a, b):
        """Simple beta distribution sample using gamma."""
        # Use inverse CDF approximation for small alpha
        if a <= 0 or b <= 0:
            return 0.5
        # Gamma samples via Marsaglia-Tsang
        x = self._gamma_sample(a)
        y = self._gamma_sample(b)
        if x + y == 0:
            return 0.5
        return x / (x + y)

    def _gamma_sample(self, shape):
        """Gamma distribution sample (Marsaglia-Tsang for shape >= 1)."""
        if shape < 1:
            u = self.rng.random()
            return self._gamma_sample(shape + 1) * (u ** (1.0 / shape))

        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)

        while True:
            x = self.rng.gauss(0, 1)
            v = (1 + c * x) ** 3
            if v <= 0:
                continue
            u = self.rng.random()
            if u < 1 - 0.0331 * (x * x) * (x * x):
                return d * v
            if math.log(max(u, 1e-15)) < 0.5 * x * x + d * (1 - v + math.log(max(v, 1e-15))):
                return d * v


# ============================================================
# MultiTaskHead: Shared backbone with multiple task heads
# ============================================================

class MultiTaskHead:
    """
    Multi-task learning: shared backbone with multiple task-specific heads.
    Each head can have different output sizes and loss functions.
    """

    def __init__(self, backbone_model, backbone_output_size):
        """
        backbone_model: Sequential model for shared features
        backbone_output_size: output dimension of backbone
        """
        self.backbone = backbone_model
        self.backbone_output_size = backbone_output_size
        self.heads = {}  # task_name -> Sequential
        self.loss_fns = {}  # task_name -> loss function
        self.task_weights = {}  # task_name -> weight

    def add_task(self, name, head_layers, loss_fn, weight=1.0):
        """Add a task head."""
        self.heads[name] = Sequential(head_layers)
        self.loss_fns[name] = loss_fn
        self.task_weights[name] = weight

    def forward(self, x, task_name=None):
        """
        Forward pass. If task_name given, return that task's output.
        Otherwise return dict of all task outputs.
        """
        features = self.backbone.forward(x)

        if task_name is not None:
            if task_name not in self.heads:
                raise ValueError(f"Unknown task: {task_name}")
            return self.heads[task_name].forward(features)

        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head.forward(features)
        return outputs

    def train_step(self, x_batch, targets, optimizer):
        """
        Multi-task training step.
        targets: dict mapping task_name -> target tensor/labels
        Returns dict of per-task losses.
        """
        # Forward through backbone
        features = self.backbone.forward(x_batch)

        total_grad = None
        losses = {}

        for name, head in self.heads.items():
            if name not in targets:
                continue

            output = head.forward(features)
            loss = self.loss_fns[name].forward(output, targets[name])
            losses[name] = loss

            grad = self.loss_fns[name].backward(output, targets[name])
            head_grad = head.backward(grad)

            weight = self.task_weights.get(name, 1.0)
            if len(head_grad.shape) == 1:
                weighted = Tensor([weight * v for v in head_grad.data])
            else:
                weighted = Tensor([[weight * v for v in row] for row in head_grad.data])

            if total_grad is None:
                total_grad = weighted
            else:
                total_grad = total_grad + weighted

        # Backward through backbone
        if total_grad is not None:
            self.backbone.backward(total_grad)

        # Update all parameters
        all_layers = self.backbone.get_trainable_layers()
        for head in self.heads.values():
            all_layers.extend(head.get_trainable_layers())
        optimizer.step(all_layers)

        return losses

    def predict(self, x, task_name):
        """Predict for a specific task."""
        self.backbone.eval()
        self.heads[task_name].eval()
        features = self.backbone.forward(x)
        output = self.heads[task_name].forward(features)
        self.backbone.train()
        self.heads[task_name].train()
        return output


# ============================================================
# EWC: Elastic Weight Consolidation (catastrophic forgetting prevention)
# ============================================================

class EWC:
    """
    Elastic Weight Consolidation to prevent catastrophic forgetting
    when fine-tuning on a new task.

    Adds a penalty term: lambda/2 * sum_i F_i * (theta_i - theta_i*)^2
    where F_i is the Fisher information (approximated by squared gradients)
    and theta_i* are the optimal parameters for the previous task.
    """

    def __init__(self, model, lambda_ewc=1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self._saved_params = {}  # layer_idx -> [(param_data_copy, fisher)]
        self._computed = False

    def compute_fisher(self, X, Y, loss_fn, n_samples=None):
        """
        Compute Fisher Information Matrix (diagonal approximation)
        using the model's gradients on the original task data.
        """
        seq = self.model.model if isinstance(self.model, PretrainedModel) else self.model

        num_samples = X.shape[0] if len(X.shape) == 2 else 1
        if n_samples is None:
            n_samples = min(num_samples, 100)

        indices = list(range(num_samples))[:n_samples]

        # Initialize fisher accumulators and save params
        self._saved_params = {}
        for layer_idx, layer in enumerate(seq.layers):
            params = layer.get_params()
            if not params:
                continue
            layer_data = []
            for pt in params:
                tensor = pt[0]
                # Save parameter values
                if len(tensor.shape) == 1:
                    param_copy = tensor.data[:]
                    fisher = [0.0] * len(tensor.data)
                else:
                    param_copy = [row[:] for row in tensor.data]
                    fisher = [[0.0] * tensor.shape[1] for _ in range(tensor.shape[0])]
                layer_data.append((param_copy, fisher))
            self._saved_params[layer_idx] = layer_data

        # Accumulate squared gradients
        for idx in indices:
            if len(X.shape) == 2:
                x_sample = Tensor([X.data[idx]])
            else:
                x_sample = X

            if isinstance(Y, list):
                y_sample = [Y[idx]]
            elif isinstance(Y, Tensor) and len(Y.shape) == 2:
                y_sample = Tensor([Y.data[idx]])
            else:
                y_sample = Y

            output = seq.forward(x_sample)
            loss = loss_fn.forward(output, y_sample)
            grad = loss_fn.backward(output, y_sample)
            seq.backward(grad)

            # Accumulate squared gradients as Fisher approximation
            for layer_idx, layer in enumerate(seq.layers):
                if layer_idx not in self._saved_params:
                    continue
                params = layer.get_params()
                for p_idx, pt in enumerate(params):
                    _, grad_tensor, _ = pt
                    if grad_tensor is None:
                        continue
                    fisher = self._saved_params[layer_idx][p_idx][1]
                    if isinstance(fisher[0], list):
                        for i in range(len(fisher)):
                            for j in range(len(fisher[i])):
                                fisher[i][j] += grad_tensor.data[i][j] ** 2
                    else:
                        for i in range(len(fisher)):
                            fisher[i] += grad_tensor.data[i] ** 2

        # Normalize
        for layer_idx in self._saved_params:
            for p_idx in range(len(self._saved_params[layer_idx])):
                fisher = self._saved_params[layer_idx][p_idx][1]
                if isinstance(fisher[0], list):
                    for i in range(len(fisher)):
                        for j in range(len(fisher[i])):
                            fisher[i][j] /= n_samples
                else:
                    for i in range(len(fisher)):
                        fisher[i] /= n_samples

        self._computed = True

    def penalty(self):
        """Compute EWC penalty term."""
        if not self._computed:
            return 0.0

        seq = self.model.model if isinstance(self.model, PretrainedModel) else self.model
        total = 0.0

        for layer_idx, layer in enumerate(seq.layers):
            if layer_idx not in self._saved_params:
                continue
            params = layer.get_params()
            for p_idx, pt in enumerate(params):
                tensor = pt[0]
                saved, fisher = self._saved_params[layer_idx][p_idx]
                if isinstance(saved[0], list):
                    for i in range(len(saved)):
                        for j in range(len(saved[i])):
                            diff = tensor.data[i][j] - saved[i][j]
                            total += fisher[i][j] * diff * diff
                else:
                    for i in range(len(saved)):
                        diff = tensor.data[i] - saved[i]
                        total += fisher[i] * diff * diff

        return self.lambda_ewc / 2.0 * total

    def penalty_gradient(self):
        """
        Compute gradient of EWC penalty w.r.t. current parameters.
        Returns dict mapping layer_idx -> list of gradient tensors.
        """
        if not self._computed:
            return {}

        seq = self.model.model if isinstance(self.model, PretrainedModel) else self.model
        grads = {}

        for layer_idx, layer in enumerate(seq.layers):
            if layer_idx not in self._saved_params:
                continue
            params = layer.get_params()
            layer_grads = []
            for p_idx, pt in enumerate(params):
                tensor = pt[0]
                saved, fisher = self._saved_params[layer_idx][p_idx]
                if isinstance(saved[0], list):
                    grad = [[self.lambda_ewc * fisher[i][j] * (tensor.data[i][j] - saved[i][j])
                            for j in range(len(saved[i]))]
                           for i in range(len(saved))]
                    layer_grads.append(Tensor(grad))
                else:
                    grad = [self.lambda_ewc * fisher[i] * (tensor.data[i] - saved[i])
                           for i in range(len(saved))]
                    layer_grads.append(Tensor(grad))
            grads[layer_idx] = layer_grads

        return grads


# ============================================================
# ProgressiveNet: Progressive Neural Networks
# ============================================================

class ProgressiveNet:
    """
    Progressive neural networks: add new columns for new tasks
    with lateral connections to previous columns.
    Prevents catastrophic forgetting by keeping old columns frozen.
    """

    def __init__(self):
        self.columns = []  # list of Sequential models
        self.adapters = []  # adapters[col][layer] connects prev columns to current
        self._frozen_columns = set()

    def add_column(self, layer_sizes, activations=None, rng=None):
        """
        Add a new column (task-specific network).
        layer_sizes: list of ints [input, hidden..., output]
        """
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['linear']

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Dense(layer_sizes[i], layer_sizes[i + 1], rng=rng))
            if i < len(activations):
                layers.append(Activation(activations[i]))

        col = Sequential(layers)
        col_idx = len(self.columns)

        # Create lateral adapters from previous columns
        adapters = []
        if col_idx > 0:
            # For each Dense layer in new column, create adapter from each prev column
            for layer in layers:
                if isinstance(layer, Dense):
                    col_adapters = []
                    for prev_col in self.columns:
                        # Find matching Dense layer in previous column
                        prev_dense_layers = [l for l in prev_col.layers
                                            if isinstance(l, Dense)]
                        if prev_dense_layers:
                            # Adapter: simple linear projection
                            # from prev layer output to current layer input
                            prev_out = prev_dense_layers[-1].output_size if prev_dense_layers else layer.input_size
                            adapter = Dense(prev_out, layer.input_size, rng=rng)
                            col_adapters.append(adapter)
                    adapters.append(col_adapters)

        # Freeze all previous columns
        for i in range(col_idx):
            self._frozen_columns.add(i)
            for layer in self.columns[i].layers:
                freeze_layer(layer)

        self.columns.append(col)
        self.adapters.append(adapters)
        return col_idx

    def forward(self, x, column_idx=None):
        """
        Forward pass through specified column (default: latest).
        Includes lateral connections from previous columns.
        """
        if column_idx is None:
            column_idx = len(self.columns) - 1

        if column_idx == 0 or not self.adapters[column_idx]:
            return self.columns[column_idx].forward(x)

        # Get activations from previous columns
        prev_outputs = []
        for i in range(column_idx):
            self.columns[i].eval()
            prev_out = self.columns[i].forward(x)
            prev_outputs.append(prev_out)
            self.columns[i].train()

        # Forward through current column with lateral connections
        col = self.columns[column_idx]
        out = x
        adapter_idx = 0

        for layer in col.layers:
            if isinstance(layer, Dense) and adapter_idx < len(self.adapters[column_idx]):
                # Add lateral contributions
                lateral_sum = None
                adapters = self.adapters[column_idx][adapter_idx]
                for a_idx, adapter in enumerate(adapters):
                    if a_idx < len(prev_outputs):
                        lateral = adapter.forward(prev_outputs[a_idx])
                        if lateral_sum is None:
                            lateral_sum = lateral
                        else:
                            lateral_sum = lateral_sum + lateral

                out = layer.forward(out)
                if lateral_sum is not None:
                    # Add lateral to output (shape must match)
                    try:
                        out = out + lateral_sum
                    except (ValueError, IndexError):
                        pass  # Skip if shapes don't match
                adapter_idx += 1
            else:
                out = layer.forward(out)

        return out

    def get_column(self, idx):
        """Get a specific column."""
        return self.columns[idx]

    def num_columns(self):
        """Number of columns (tasks)."""
        return len(self.columns)
