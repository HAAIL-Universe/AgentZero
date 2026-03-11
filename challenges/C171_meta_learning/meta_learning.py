"""
C171: Meta-Learning -- composing C170 (Transfer Learning) + C140 (Neural Network)

Implements meta-learning algorithms for few-shot learning:
- MAML (Model-Agnostic Meta-Learning) with first/second-order variants
- Reptile (first-order meta-learning)
- Prototypical Networks (metric-based few-shot)
- Matching Networks (attention-based few-shot)
- Task distributions for N-way K-shot episodes
- Meta-trainer orchestration
- Few-shot evaluation

Built from scratch using only C140 neural network primitives.
"""

import sys, os, math, copy, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C170_transfer_learning'))

from neural_network import (
    Tensor, Dense, Activation, Sequential, MSELoss, CrossEntropyLoss,
    SGD, Adam, build_model, save_weights, load_weights, softmax, softmax_batch,
    xavier_init, he_init, relu, sigmoid, tanh_act
)
from transfer_learning import (
    PretrainedModel, FeatureExtractor, freeze_layer, unfreeze_layer
)


# ============================================================
# Task Distribution -- generates N-way K-shot episodes
# ============================================================

class Task:
    """A single few-shot task (episode) with support and query sets."""

    def __init__(self, support_x, support_y, query_x, query_y, classes):
        self.support_x = support_x  # Tensor (n_support, feat_dim)
        self.support_y = support_y  # list of int labels (0..N-1)
        self.query_x = query_x      # Tensor (n_query, feat_dim)
        self.query_y = query_y      # list of int labels (0..N-1)
        self.classes = classes       # list of original class IDs
        self.n_way = len(classes)
        self.k_shot = len(support_y) // len(classes) if classes else 0


class TaskDistribution:
    """Generates few-shot learning tasks from a dataset.

    Dataset format: list of (features, label) pairs, or (X_tensor, Y_labels).
    """

    def __init__(self, X, Y, seed=42):
        """
        X: Tensor (n_samples, feat_dim) or list of lists
        Y: list of int class labels
        """
        if isinstance(X, Tensor):
            self._data = {}
            for i in range(len(Y)):
                label = Y[i]
                if label not in self._data:
                    self._data[label] = []
                row = X.data[i] if len(X.shape) == 2 else [X.data[i]]
                self._data[label].append(row)
        else:
            self._data = {}
            for i in range(len(Y)):
                label = Y[i]
                if label not in self._data:
                    self._data[label] = []
                self._data[label].append(X[i])

        self._classes = sorted(self._data.keys())
        self._rng = random.Random(seed)
        self._feat_dim = len(self._data[self._classes[0]][0])

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def feat_dim(self):
        return self._feat_dim

    def sample_task(self, n_way, k_shot, q_queries=1):
        """Sample a single N-way K-shot task with q query samples per class."""
        if n_way > len(self._classes):
            raise ValueError(f"n_way={n_way} > available classes={len(self._classes)}")

        selected = self._rng.sample(self._classes, n_way)
        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, cls in enumerate(selected):
            examples = self._data[cls]
            needed = k_shot + q_queries
            if needed > len(examples):
                # Sample with replacement if not enough
                chosen = [examples[self._rng.randint(0, len(examples) - 1)]
                          for _ in range(needed)]
            else:
                chosen = self._rng.sample(examples, needed)

            for i in range(k_shot):
                support_x.append(chosen[i])
                support_y.append(new_label)
            for i in range(k_shot, k_shot + q_queries):
                query_x.append(chosen[i])
                query_y.append(new_label)

        return Task(
            support_x=Tensor(support_x),
            support_y=support_y,
            query_x=Tensor(query_x),
            query_y=query_y,
            classes=selected
        )

    def sample_tasks(self, n_tasks, n_way, k_shot, q_queries=1):
        """Sample multiple tasks."""
        return [self.sample_task(n_way, k_shot, q_queries) for _ in range(n_tasks)]


# ============================================================
# Parameter utilities for meta-learning
# ============================================================

def _clone_model(model):
    """Deep clone a Sequential model (architecture + weights)."""
    weights = save_weights(model)
    # Build identical architecture
    clone = Sequential()
    for layer in model.layers:
        if isinstance(layer, Dense):
            new_layer = Dense(layer.input_size, layer.output_size,
                              init='zeros', bias=layer.use_bias)
            clone.add(new_layer)
        elif isinstance(layer, Activation):
            clone.add(Activation(layer.name))
        else:
            clone.add(layer)
    load_weights(clone, weights)
    return clone


def _get_flat_params(model):
    """Extract all trainable parameters as a flat list of floats."""
    params = []
    for layer in model.layers:
        for param, grad, name in layer.get_params():
            if len(param.shape) == 2:
                for row in param.data:
                    params.extend(row)
            else:
                params.extend(param.data)
    return params


def _set_flat_params(model, flat_params):
    """Set all trainable parameters from a flat list of floats."""
    idx = 0
    for layer in model.layers:
        for param, grad, name in layer.get_params():
            if len(param.shape) == 2:
                for r in range(param.shape[0]):
                    for c in range(param.shape[1]):
                        param.data[r][c] = flat_params[idx]
                        idx += 1
            else:
                for i in range(param.shape[0]):
                    param.data[i] = flat_params[idx]
                    idx += 1


def _param_subtract(params_a, params_b):
    """Element-wise subtraction of two flat param lists."""
    return [a - b for a, b in zip(params_a, params_b)]


def _param_add(params_a, params_b):
    """Element-wise addition of two flat param lists."""
    return [a + b for a, b in zip(params_a, params_b)]


def _param_scale(params, scalar):
    """Scale flat param list by scalar."""
    return [p * scalar for p in params]


def _param_zeros_like(params):
    """Zero-filled list same length as params."""
    return [0.0] * len(params)


def _inner_train_step(model, x_batch, y_batch, loss_fn, lr):
    """Perform one gradient step on model (in-place). Returns loss."""
    model.train()
    output = model.forward(x_batch)
    loss = loss_fn.forward(output, y_batch)
    grad = loss_fn.backward(output, y_batch)
    model.backward(grad)

    # Manual SGD step
    for layer in model.layers:
        for param, grad_p, name in layer.get_params():
            if grad_p is None:
                continue
            if len(param.shape) == 2:
                for r in range(param.shape[0]):
                    for c in range(param.shape[1]):
                        param.data[r][c] -= lr * grad_p.data[r][c]
            else:
                for i in range(param.shape[0]):
                    param.data[i] -= lr * grad_p.data[i]
    return loss


# ============================================================
# MAML -- Model-Agnostic Meta-Learning
# ============================================================

class MAML:
    """Model-Agnostic Meta-Learning (Finn et al., 2017).

    Learns an initialization that can be quickly adapted to new tasks
    with just a few gradient steps.

    Supports first-order approximation (FOMAML) for efficiency.
    """

    def __init__(self, model, inner_lr=0.01, outer_lr=0.001,
                 inner_steps=1, first_order=False):
        """
        model: Sequential model (the meta-learner)
        inner_lr: learning rate for task-specific adaptation
        outer_lr: learning rate for meta-update
        inner_steps: number of gradient steps in inner loop
        first_order: if True, use FOMAML (ignore second-order gradients)
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self._history = {'meta_loss': [], 'meta_accuracy': []}

    def adapt(self, task, steps=None):
        """Adapt model to a single task. Returns adapted model clone."""
        if steps is None:
            steps = self.inner_steps

        adapted = _clone_model(self.model)
        loss_fn = CrossEntropyLoss()

        for _ in range(steps):
            _inner_train_step(adapted, task.support_x, task.support_y,
                              loss_fn, self.inner_lr)

        return adapted

    def meta_train_step(self, tasks):
        """One meta-training step over a batch of tasks.

        Returns average meta-loss across tasks.
        """
        loss_fn = CrossEntropyLoss()
        meta_params_before = _get_flat_params(self.model)
        meta_grad_accum = _param_zeros_like(meta_params_before)
        total_loss = 0.0
        total_correct = 0
        total_queries = 0

        for task in tasks:
            # Inner loop: adapt to support set
            adapted = self.adapt(task)

            # Outer loop: evaluate on query set
            adapted.eval()
            output = adapted.forward(task.query_x)
            loss = loss_fn.forward(output, task.query_y)
            total_loss += loss

            # Accuracy on query set
            preds = output.argmax(axis=1) if len(output.shape) == 2 else [output.argmax()]
            if not isinstance(preds, list):
                preds = [preds]
            for p, t in zip(preds, task.query_y):
                if p == t:
                    total_correct += 1
                total_queries += 1

            # Compute gradient of query loss w.r.t. adapted params
            adapted.train()
            output = adapted.forward(task.query_x)
            grad = loss_fn.backward(output, task.query_y)
            adapted.backward(grad)

            if self.first_order:
                # FOMAML: use gradient at adapted params directly
                adapted_grad = []
                for layer in adapted.layers:
                    for param, grad_p, name in layer.get_params():
                        if grad_p is None:
                            if len(param.shape) == 2:
                                for row in param.data:
                                    adapted_grad.extend([0.0] * len(row))
                            else:
                                adapted_grad.extend([0.0] * len(param.data))
                        elif len(grad_p.shape) == 2:
                            for row in grad_p.data:
                                adapted_grad.extend(row)
                        else:
                            adapted_grad.extend(grad_p.data)
                meta_grad_accum = _param_add(meta_grad_accum, adapted_grad)
            else:
                # Second-order: approximate by computing gradient at adapted params
                # and projecting back (finite-difference Hessian-vector product)
                adapted_params = _get_flat_params(adapted)
                adapted_grad = []
                for layer in adapted.layers:
                    for param, grad_p, name in layer.get_params():
                        if grad_p is None:
                            if len(param.shape) == 2:
                                for row in param.data:
                                    adapted_grad.extend([0.0] * len(row))
                            else:
                                adapted_grad.extend([0.0] * len(param.data))
                        elif len(grad_p.shape) == 2:
                            for row in grad_p.data:
                                adapted_grad.extend(row)
                        else:
                            adapted_grad.extend(grad_p.data)

                # For second-order MAML, we approximate the meta-gradient
                # using the adapted gradient projected through the inner loop
                # For simplicity, use the adapted gradient directly (this is
                # equivalent to FOMAML but with the full computation path)
                eps = 0.01 / (max(abs(g) for g in adapted_grad) + 1e-10)

                # Forward finite diff: f(theta + eps*g) - f(theta - eps*g)
                plus_params = _param_add(meta_params_before,
                                          _param_scale(adapted_grad, eps))
                minus_params = _param_add(meta_params_before,
                                           _param_scale(adapted_grad, -eps))

                _set_flat_params(self.model, plus_params)
                adapted_plus = self.adapt(task)
                adapted_plus.eval()
                out_plus = adapted_plus.forward(task.query_x)
                loss_plus = loss_fn.forward(out_plus, task.query_y)

                _set_flat_params(self.model, minus_params)
                adapted_minus = self.adapt(task)
                adapted_minus.eval()
                out_minus = adapted_minus.forward(task.query_x)
                loss_minus = loss_fn.forward(out_minus, task.query_y)

                # Restore original params
                _set_flat_params(self.model, meta_params_before)

                # Hessian-vector product approximation
                hvp = _param_scale(adapted_grad, (loss_plus - loss_minus) / (2 * eps))
                # Meta gradient = adapted_grad - inner_lr * hvp
                correction = _param_scale(hvp, self.inner_lr)
                task_meta_grad = _param_subtract(adapted_grad, correction)
                meta_grad_accum = _param_add(meta_grad_accum, task_meta_grad)

        # Average gradients over tasks
        n_tasks = len(tasks)
        avg_grad = _param_scale(meta_grad_accum, 1.0 / n_tasks)

        # Meta-update
        new_params = _param_subtract(meta_params_before,
                                      _param_scale(avg_grad, self.outer_lr))
        _set_flat_params(self.model, new_params)

        avg_loss = total_loss / n_tasks
        accuracy = total_correct / total_queries if total_queries > 0 else 0.0
        self._history['meta_loss'].append(avg_loss)
        self._history['meta_accuracy'].append(accuracy)
        return avg_loss

    def meta_train(self, task_distribution, n_way, k_shot, q_queries=1,
                   meta_epochs=100, tasks_per_epoch=4, verbose=False):
        """Full meta-training loop.

        Returns history dict with 'meta_loss' and 'meta_accuracy'.
        """
        for epoch in range(meta_epochs):
            tasks = task_distribution.sample_tasks(tasks_per_epoch, n_way,
                                                    k_shot, q_queries)
            loss = self.meta_train_step(tasks)
            if verbose and (epoch + 1) % 10 == 0:
                acc = self._history['meta_accuracy'][-1]
                print(f"Epoch {epoch+1}: meta_loss={loss:.4f}, accuracy={acc:.3f}")
        return dict(self._history)

    def evaluate(self, task_distribution, n_way, k_shot, q_queries=1,
                 n_tasks=100, adaptation_steps=None):
        """Evaluate meta-learner on held-out tasks.

        Returns dict with 'accuracy', 'loss', 'per_task_accuracy'.
        """
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        loss_fn = CrossEntropyLoss()
        total_correct = 0
        total_queries = 0
        total_loss = 0.0
        per_task_acc = []

        for _ in range(n_tasks):
            task = task_distribution.sample_task(n_way, k_shot, q_queries)
            adapted = self.adapt(task, steps=adaptation_steps)
            adapted.eval()
            output = adapted.forward(task.query_x)
            loss = loss_fn.forward(output, task.query_y)
            total_loss += loss

            preds = output.argmax(axis=1) if len(output.shape) == 2 else [output.argmax()]
            if not isinstance(preds, list):
                preds = [preds]
            correct = sum(1 for p, t in zip(preds, task.query_y) if p == t)
            total_correct += correct
            total_queries += len(task.query_y)
            per_task_acc.append(correct / len(task.query_y))

        return {
            'accuracy': total_correct / total_queries if total_queries > 0 else 0.0,
            'loss': total_loss / n_tasks,
            'per_task_accuracy': per_task_acc
        }


# ============================================================
# Reptile -- first-order meta-learning
# ============================================================

class Reptile:
    """Reptile meta-learning algorithm (Nichol et al., 2018).

    Simpler than MAML: no second derivatives needed.
    Meta-update moves initialization toward adapted parameters.
    """

    def __init__(self, model, inner_lr=0.01, outer_lr=0.1,
                 inner_steps=5):
        """
        model: Sequential model
        inner_lr: learning rate for inner adaptation
        outer_lr: step size for meta-update (interpolation factor)
        inner_steps: number of inner gradient steps per task
        """
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self._history = {'meta_loss': [], 'meta_accuracy': []}

    def adapt(self, task, steps=None):
        """Adapt to a task. Returns adapted model."""
        if steps is None:
            steps = self.inner_steps

        adapted = _clone_model(self.model)
        loss_fn = CrossEntropyLoss()

        for _ in range(steps):
            _inner_train_step(adapted, task.support_x, task.support_y,
                              loss_fn, self.inner_lr)
        return adapted

    def meta_train_step(self, tasks):
        """One Reptile meta-update step.

        For each task, adapt model and then move meta-params toward adapted params.
        Returns average query loss.
        """
        loss_fn = CrossEntropyLoss()
        meta_params = _get_flat_params(self.model)
        direction_accum = _param_zeros_like(meta_params)
        total_loss = 0.0
        total_correct = 0
        total_queries = 0

        for task in tasks:
            adapted = self.adapt(task)
            adapted_params = _get_flat_params(adapted)

            # Direction: adapted - original
            diff = _param_subtract(adapted_params, meta_params)
            direction_accum = _param_add(direction_accum, diff)

            # Evaluate on query set
            adapted.eval()
            output = adapted.forward(task.query_x)
            loss = loss_fn.forward(output, task.query_y)
            total_loss += loss

            preds = output.argmax(axis=1) if len(output.shape) == 2 else [output.argmax()]
            if not isinstance(preds, list):
                preds = [preds]
            for p, t in zip(preds, task.query_y):
                if p == t:
                    total_correct += 1
                total_queries += 1

        # Average direction and apply meta-update
        n_tasks = len(tasks)
        avg_direction = _param_scale(direction_accum, 1.0 / n_tasks)
        new_params = _param_add(meta_params,
                                 _param_scale(avg_direction, self.outer_lr))
        _set_flat_params(self.model, new_params)

        avg_loss = total_loss / n_tasks
        accuracy = total_correct / total_queries if total_queries > 0 else 0.0
        self._history['meta_loss'].append(avg_loss)
        self._history['meta_accuracy'].append(accuracy)
        return avg_loss

    def meta_train(self, task_distribution, n_way, k_shot, q_queries=1,
                   meta_epochs=100, tasks_per_epoch=4, verbose=False):
        """Full Reptile meta-training loop."""
        for epoch in range(meta_epochs):
            tasks = task_distribution.sample_tasks(tasks_per_epoch, n_way,
                                                    k_shot, q_queries)
            loss = self.meta_train_step(tasks)
            if verbose and (epoch + 1) % 10 == 0:
                acc = self._history['meta_accuracy'][-1]
                print(f"Epoch {epoch+1}: meta_loss={loss:.4f}, accuracy={acc:.3f}")
        return dict(self._history)

    def evaluate(self, task_distribution, n_way, k_shot, q_queries=1,
                 n_tasks=100, adaptation_steps=None):
        """Evaluate Reptile on held-out tasks."""
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps

        loss_fn = CrossEntropyLoss()
        total_correct = 0
        total_queries = 0
        total_loss = 0.0
        per_task_acc = []

        for _ in range(n_tasks):
            task = task_distribution.sample_task(n_way, k_shot, q_queries)
            adapted = self.adapt(task, steps=adaptation_steps)
            adapted.eval()
            output = adapted.forward(task.query_x)
            loss = loss_fn.forward(output, task.query_y)
            total_loss += loss

            preds = output.argmax(axis=1) if len(output.shape) == 2 else [output.argmax()]
            if not isinstance(preds, list):
                preds = [preds]
            correct = sum(1 for p, t in zip(preds, task.query_y) if p == t)
            total_correct += correct
            total_queries += len(task.query_y)
            per_task_acc.append(correct / len(task.query_y))

        return {
            'accuracy': total_correct / total_queries if total_queries > 0 else 0.0,
            'loss': total_loss / n_tasks,
            'per_task_accuracy': per_task_acc
        }


# ============================================================
# Prototypical Networks -- metric-based few-shot learning
# ============================================================

def _euclidean_distance(a, b):
    """Squared Euclidean distance between two 1D tensors."""
    total = 0.0
    for i in range(len(a.data)):
        d = a.data[i] - b.data[i]
        total += d * d
    return total


def _cosine_similarity(a, b):
    """Cosine similarity between two 1D tensors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(a.data)):
        dot += a.data[i] * b.data[i]
        norm_a += a.data[i] * a.data[i]
        norm_b += b.data[i] * b.data[i]
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class PrototypicalNetwork:
    """Prototypical Networks (Snell et al., 2017).

    Learns an embedding space where classification is performed
    by computing distances to class prototypes (mean embeddings).
    """

    def __init__(self, encoder, distance='euclidean', lr=0.001):
        """
        encoder: Sequential model mapping inputs to embeddings
        distance: 'euclidean' or 'cosine'
        lr: learning rate for encoder training
        """
        self.encoder = encoder
        self.distance = distance
        self.lr = lr
        self._history = {'loss': [], 'accuracy': []}

    def compute_prototypes(self, support_x, support_y, n_way):
        """Compute class prototypes from support set.

        Returns list of prototype tensors (one per class).
        """
        embeddings = self.encoder.forward(support_x)
        if len(embeddings.shape) == 1:
            embeddings = Tensor([embeddings.data])

        embed_dim = embeddings.shape[1] if len(embeddings.shape) == 2 else len(embeddings.data)
        prototypes = []

        for c in range(n_way):
            class_embeds = []
            for i in range(len(support_y)):
                if support_y[i] == c:
                    class_embeds.append(embeddings.data[i])

            # Mean embedding
            proto = [0.0] * embed_dim
            for emb in class_embeds:
                for d in range(embed_dim):
                    proto[d] += emb[d]
            n = len(class_embeds)
            if n > 0:
                proto = [p / n for p in proto]
            prototypes.append(Tensor(proto))

        return prototypes

    def classify(self, query_x, prototypes):
        """Classify query points by distance to prototypes.

        Returns (predictions, log_probabilities).
        predictions: list of predicted class indices
        log_probs: Tensor of negative distances (n_queries, n_way)
        """
        embeddings = self.encoder.forward(query_x)
        if len(embeddings.shape) == 1:
            embeddings = Tensor([embeddings.data])

        n_queries = embeddings.shape[0]
        n_way = len(prototypes)
        neg_distances = []

        for i in range(n_queries):
            query_embed = Tensor(embeddings.data[i])
            dists = []
            for proto in prototypes:
                if self.distance == 'euclidean':
                    d = -_euclidean_distance(query_embed, proto)
                else:  # cosine
                    d = _cosine_similarity(query_embed, proto)
                dists.append(d)
            neg_distances.append(dists)

        # Softmax over negative distances
        log_probs = Tensor(neg_distances)
        predictions = []
        for i in range(n_queries):
            best = 0
            best_val = neg_distances[i][0]
            for j in range(1, n_way):
                if neg_distances[i][j] > best_val:
                    best_val = neg_distances[i][j]
                    best = j
            predictions.append(best)

        return predictions, log_probs

    def episode_loss(self, task):
        """Compute loss for a single episode. Returns (loss, accuracy)."""
        prototypes = self.compute_prototypes(task.support_x, task.support_y,
                                              task.n_way)
        predictions, log_probs = self.classify(task.query_x, prototypes)

        # Cross-entropy over softmax of negative distances
        loss = 0.0
        n_queries = len(task.query_y)
        for i in range(n_queries):
            scores = log_probs.data[i]
            # Numerically stable softmax + cross-entropy
            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exp_scores)
            log_prob = (scores[task.query_y[i]] - max_s) - math.log(sum_exp)
            loss -= log_prob

        loss /= n_queries
        correct = sum(1 for p, t in zip(predictions, task.query_y) if p == t)
        accuracy = correct / n_queries

        return loss, accuracy

    def train_step(self, task):
        """One training step on a single episode. Returns loss."""
        self.encoder.train()

        # Forward pass for query embeddings with gradients
        prototypes = self.compute_prototypes(task.support_x, task.support_y,
                                              task.n_way)
        query_embeds = self.encoder.forward(task.query_x)
        if len(query_embeds.shape) == 1:
            query_embeds = Tensor([query_embeds.data])

        n_queries = query_embeds.shape[0]
        n_way = task.n_way
        embed_dim = query_embeds.shape[1]

        # Compute softmax probabilities and gradients
        all_probs = []
        loss = 0.0
        for i in range(n_queries):
            q = query_embeds.data[i]
            scores = []
            for proto in prototypes:
                if self.distance == 'euclidean':
                    d = sum((q[d] - proto.data[d]) ** 2 for d in range(embed_dim))
                    scores.append(-d)
                else:
                    scores.append(_cosine_similarity(Tensor(q), proto))

            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exp_scores)
            probs = [e / sum_exp for e in exp_scores]
            all_probs.append(probs)

            log_prob = (scores[task.query_y[i]] - max_s) - math.log(sum_exp)
            loss -= log_prob

        loss /= n_queries

        # Gradient of loss w.r.t query embeddings
        grad_embeds = [[0.0] * embed_dim for _ in range(n_queries)]
        for i in range(n_queries):
            q = query_embeds.data[i]
            target = task.query_y[i]
            probs = all_probs[i]

            for c in range(n_way):
                proto = prototypes[c].data
                # d(score_c)/d(q) for euclidean = -2*(q - proto_c)
                if self.distance == 'euclidean':
                    weight = probs[c] - (1.0 if c == target else 0.0)
                    for d in range(embed_dim):
                        grad_embeds[i][d] += weight * (-2.0) * (q[d] - proto[d])
                else:
                    # Cosine gradient is more complex, use euclidean-style approx
                    weight = probs[c] - (1.0 if c == target else 0.0)
                    for d in range(embed_dim):
                        grad_embeds[i][d] += weight * (-2.0) * (q[d] - proto[d])

        grad_embeds = Tensor(grad_embeds)
        grad_embeds = grad_embeds * (1.0 / n_queries)

        # Backward through encoder
        self.encoder.backward(grad_embeds)

        # SGD update
        for layer in self.encoder.layers:
            for param, grad_p, name in layer.get_params():
                if grad_p is None:
                    continue
                if len(param.shape) == 2:
                    for r in range(param.shape[0]):
                        for c in range(param.shape[1]):
                            param.data[r][c] -= self.lr * grad_p.data[r][c]
                else:
                    for j in range(param.shape[0]):
                        param.data[j] -= self.lr * grad_p.data[j]

        return loss

    def meta_train(self, task_distribution, n_way, k_shot, q_queries=1,
                   episodes=200, verbose=False):
        """Train prototypical network over episodes."""
        for ep in range(episodes):
            task = task_distribution.sample_task(n_way, k_shot, q_queries)
            loss = self.train_step(task)
            _, acc = self.episode_loss(task)
            self._history['loss'].append(loss)
            self._history['accuracy'].append(acc)

            if verbose and (ep + 1) % 20 == 0:
                print(f"Episode {ep+1}: loss={loss:.4f}, accuracy={acc:.3f}")

        return dict(self._history)

    def evaluate(self, task_distribution, n_way, k_shot, q_queries=1,
                 n_tasks=100):
        """Evaluate prototypical network."""
        self.encoder.eval()
        total_correct = 0
        total_queries = 0
        total_loss = 0.0
        per_task_acc = []

        for _ in range(n_tasks):
            task = task_distribution.sample_task(n_way, k_shot, q_queries)
            loss, acc = self.episode_loss(task)
            total_loss += loss
            per_task_acc.append(acc)
            n_q = len(task.query_y)
            total_correct += int(acc * n_q)
            total_queries += n_q

        return {
            'accuracy': total_correct / total_queries if total_queries > 0 else 0.0,
            'loss': total_loss / n_tasks,
            'per_task_accuracy': per_task_acc
        }


# ============================================================
# Matching Networks -- attention-based few-shot
# ============================================================

class MatchingNetwork:
    """Matching Networks (Vinyals et al., 2016).

    Uses attention over support set embeddings to classify queries.
    Attention weights are based on cosine similarity.
    """

    def __init__(self, encoder, lr=0.001):
        """
        encoder: Sequential model mapping inputs to embeddings
        lr: learning rate
        """
        self.encoder = encoder
        self.lr = lr
        self._history = {'loss': [], 'accuracy': []}

    def attention_classify(self, query_embed, support_embeds, support_y, n_way):
        """Classify a single query using attention over support set.

        Returns (predicted_class, class_probabilities).
        """
        # Compute cosine similarities
        sims = []
        for i in range(len(support_y)):
            s = _cosine_similarity(query_embed, Tensor(support_embeds.data[i]))
            sims.append(s)

        # Softmax attention
        max_s = max(sims) if sims else 0.0
        exp_sims = [math.exp(s - max_s) for s in sims]
        sum_exp = sum(exp_sims)
        attention = [e / sum_exp for e in exp_sims]

        # Weighted vote over classes
        class_probs = [0.0] * n_way
        for i, label in enumerate(support_y):
            class_probs[label] += attention[i]

        pred = max(range(n_way), key=lambda c: class_probs[c])
        return pred, class_probs

    def episode_loss(self, task):
        """Compute loss and accuracy for one episode."""
        self.encoder.eval()
        support_embeds = self.encoder.forward(task.support_x)
        query_embeds = self.encoder.forward(task.query_x)

        if len(support_embeds.shape) == 1:
            support_embeds = Tensor([support_embeds.data])
        if len(query_embeds.shape) == 1:
            query_embeds = Tensor([query_embeds.data])

        n_queries = len(task.query_y)
        loss = 0.0
        correct = 0

        for i in range(n_queries):
            q_embed = Tensor(query_embeds.data[i])
            pred, probs = self.attention_classify(q_embed, support_embeds,
                                                   task.support_y, task.n_way)
            if pred == task.query_y[i]:
                correct += 1
            # Cross-entropy
            p = max(probs[task.query_y[i]], 1e-12)
            loss -= math.log(p)

        loss /= n_queries
        accuracy = correct / n_queries
        return loss, accuracy

    def train_step(self, task):
        """One training step. Returns loss."""
        self.encoder.train()

        # Forward through encoder for support and query
        support_embeds = self.encoder.forward(task.support_x)
        if len(support_embeds.shape) == 1:
            support_embeds = Tensor([support_embeds.data])

        query_embeds = self.encoder.forward(task.query_x)
        if len(query_embeds.shape) == 1:
            query_embeds = Tensor([query_embeds.data])

        n_queries = len(task.query_y)
        n_support = len(task.support_y)
        embed_dim = query_embeds.shape[1]

        loss = 0.0
        grad_query = [[0.0] * embed_dim for _ in range(n_queries)]

        for i in range(n_queries):
            q = query_embeds.data[i]

            # Cosine similarities
            sims = []
            for j in range(n_support):
                s = _cosine_similarity(Tensor(q), Tensor(support_embeds.data[j]))
                sims.append(s)

            # Softmax
            max_s = max(sims) if sims else 0.0
            exp_sims = [math.exp(s - max_s) for s in sims]
            sum_exp = sum(exp_sims)
            attention = [e / sum_exp for e in exp_sims]

            # Class probabilities
            class_probs = [0.0] * task.n_way
            for j, label in enumerate(task.support_y):
                class_probs[label] += attention[j]

            p = max(class_probs[task.query_y[i]], 1e-12)
            loss -= math.log(p)

            # Gradient: use simplified Euclidean-based gradient for training stability
            for j in range(n_support):
                is_target = 1.0 if task.support_y[j] == task.query_y[i] else 0.0
                weight = attention[j] - is_target * attention[j] / max(class_probs[task.query_y[i]], 1e-12) * attention[j]
                for d in range(embed_dim):
                    grad_query[i][d] += weight * (q[d] - support_embeds.data[j][d])

        loss /= n_queries
        grad_query = Tensor(grad_query)
        grad_query = grad_query * (1.0 / n_queries)

        self.encoder.backward(grad_query)

        # SGD
        for layer in self.encoder.layers:
            for param, grad_p, name in layer.get_params():
                if grad_p is None:
                    continue
                if len(param.shape) == 2:
                    for r in range(param.shape[0]):
                        for c in range(param.shape[1]):
                            param.data[r][c] -= self.lr * grad_p.data[r][c]
                else:
                    for j in range(param.shape[0]):
                        param.data[j] -= self.lr * grad_p.data[j]

        return loss

    def meta_train(self, task_distribution, n_way, k_shot, q_queries=1,
                   episodes=200, verbose=False):
        """Train matching network."""
        for ep in range(episodes):
            task = task_distribution.sample_task(n_way, k_shot, q_queries)
            loss = self.train_step(task)
            self._history['loss'].append(loss)

            _, acc = self.episode_loss(task)
            self._history['accuracy'].append(acc)

            if verbose and (ep + 1) % 20 == 0:
                print(f"Episode {ep+1}: loss={loss:.4f}, accuracy={acc:.3f}")

        return dict(self._history)

    def evaluate(self, task_distribution, n_way, k_shot, q_queries=1,
                 n_tasks=100):
        """Evaluate matching network."""
        self.encoder.eval()
        total_correct = 0
        total_queries = 0
        total_loss = 0.0
        per_task_acc = []

        for _ in range(n_tasks):
            task = task_distribution.sample_task(n_way, k_shot, q_queries)
            loss, acc = self.episode_loss(task)
            total_loss += loss
            n_q = len(task.query_y)
            total_correct += int(acc * n_q)
            total_queries += n_q
            per_task_acc.append(acc)

        return {
            'accuracy': total_correct / total_queries if total_queries > 0 else 0.0,
            'loss': total_loss / n_tasks,
            'per_task_accuracy': per_task_acc
        }


# ============================================================
# Meta-Trainer -- orchestrates meta-training
# ============================================================

class MetaTrainer:
    """Orchestrates meta-training with support for multiple algorithms."""

    def __init__(self, algorithm, task_distribution):
        """
        algorithm: MAML, Reptile, PrototypicalNetwork, or MatchingNetwork
        task_distribution: TaskDistribution
        """
        self.algorithm = algorithm
        self.task_dist = task_distribution
        self._results = []

    def train(self, n_way, k_shot, q_queries=1, epochs=100,
              tasks_per_epoch=4, verbose=False):
        """Run meta-training."""
        algo = self.algorithm
        alg_type = type(algo).__name__

        if alg_type in ('MAML', 'Reptile'):
            history = algo.meta_train(self.task_dist, n_way, k_shot,
                                       q_queries, epochs, tasks_per_epoch,
                                       verbose)
        elif alg_type in ('PrototypicalNetwork', 'MatchingNetwork'):
            history = algo.meta_train(self.task_dist, n_way, k_shot,
                                       q_queries, epochs, verbose)
        else:
            raise ValueError(f"Unknown algorithm: {alg_type}")

        self._results.append({
            'algorithm': alg_type,
            'n_way': n_way,
            'k_shot': k_shot,
            'epochs': epochs,
            'history': history
        })
        return history

    def evaluate(self, n_way, k_shot, q_queries=1, n_tasks=50):
        """Evaluate current algorithm."""
        return self.algorithm.evaluate(self.task_dist, n_way, k_shot,
                                        q_queries, n_tasks)

    def compare_algorithms(self, algorithms, n_way, k_shot, q_queries=1,
                           train_epochs=50, eval_tasks=30, tasks_per_epoch=4):
        """Compare multiple meta-learning algorithms.

        algorithms: dict mapping name -> algorithm instance
        Returns dict mapping name -> evaluation results.
        """
        results = {}
        for name, algo in algorithms.items():
            alg_type = type(algo).__name__
            if alg_type in ('MAML', 'Reptile'):
                algo.meta_train(self.task_dist, n_way, k_shot, q_queries,
                                train_epochs, tasks_per_epoch)
            else:
                algo.meta_train(self.task_dist, n_way, k_shot, q_queries,
                                train_epochs)
            eval_result = algo.evaluate(self.task_dist, n_way, k_shot,
                                         q_queries, eval_tasks)
            results[name] = eval_result
        return results


# ============================================================
# Few-Shot Classifier -- wraps meta-learned model for inference
# ============================================================

class FewShotClassifier:
    """Wraps a meta-learned model for few-shot classification at inference time."""

    def __init__(self, algorithm):
        """
        algorithm: trained MAML, Reptile, PrototypicalNetwork, or MatchingNetwork
        """
        self.algorithm = algorithm
        self._adapted_model = None
        self._prototypes = None
        self._support_embeds = None
        self._support_y = None
        self._n_way = None

    def fit(self, support_x, support_y, n_way=None, adaptation_steps=None):
        """Adapt to a new task given support examples.

        support_x: Tensor of support features
        support_y: list of int labels
        n_way: number of classes (inferred if None)
        """
        if n_way is None:
            n_way = len(set(support_y))
        self._n_way = n_way

        algo = self.algorithm
        alg_type = type(algo).__name__

        if alg_type in ('MAML', 'Reptile'):
            task = Task(support_x, support_y,
                        support_x, support_y, list(range(n_way)))
            steps = adaptation_steps or getattr(algo, 'inner_steps', 5)
            self._adapted_model = algo.adapt(task, steps=steps)
            self._adapted_model.eval()

        elif alg_type == 'PrototypicalNetwork':
            algo.encoder.eval()
            self._prototypes = algo.compute_prototypes(support_x, support_y,
                                                        n_way)

        elif alg_type == 'MatchingNetwork':
            algo.encoder.eval()
            self._support_embeds = algo.encoder.forward(support_x)
            if len(self._support_embeds.shape) == 1:
                self._support_embeds = Tensor([self._support_embeds.data])
            self._support_y = support_y

    def predict(self, query_x):
        """Predict classes for query examples. Returns list of predicted class indices."""
        algo = self.algorithm
        alg_type = type(algo).__name__

        if alg_type in ('MAML', 'Reptile'):
            output = self._adapted_model.forward(query_x)
            if len(output.shape) == 1:
                return [output.argmax()]
            return output.argmax(axis=1)

        elif alg_type == 'PrototypicalNetwork':
            preds, _ = algo.classify(query_x, self._prototypes)
            return preds

        elif alg_type == 'MatchingNetwork':
            query_embeds = algo.encoder.forward(query_x)
            if len(query_embeds.shape) == 1:
                query_embeds = Tensor([query_embeds.data])

            predictions = []
            for i in range(len(query_embeds.data)):
                q = Tensor(query_embeds.data[i])
                pred, _ = algo.attention_classify(q, self._support_embeds,
                                                   self._support_y, self._n_way)
                predictions.append(pred)
            return predictions

        raise ValueError(f"Unknown algorithm: {alg_type}")

    def predict_proba(self, query_x):
        """Predict class probabilities. Returns list of probability lists."""
        algo = self.algorithm
        alg_type = type(algo).__name__

        if alg_type in ('MAML', 'Reptile'):
            output = self._adapted_model.forward(query_x)
            if len(output.shape) == 1:
                probs = softmax(output.data)
                return [probs]
            return [softmax(row) for row in output.data]

        elif alg_type == 'PrototypicalNetwork':
            _, log_probs = algo.classify(query_x, self._prototypes)
            result = []
            for row in log_probs.data:
                probs = softmax(row)
                result.append(probs)
            return result

        elif alg_type == 'MatchingNetwork':
            query_embeds = algo.encoder.forward(query_x)
            if len(query_embeds.shape) == 1:
                query_embeds = Tensor([query_embeds.data])
            result = []
            for i in range(len(query_embeds.data)):
                q = Tensor(query_embeds.data[i])
                _, probs = algo.attention_classify(q, self._support_embeds,
                                                    self._support_y, self._n_way)
                result.append(probs)
            return result

        raise ValueError(f"Unknown algorithm: {alg_type}")


# ============================================================
# Task Augmentation for Meta-Learning
# ============================================================

class TaskAugmenter:
    """Augments few-shot tasks by generating additional support examples."""

    def __init__(self, seed=42):
        self._rng = random.Random(seed)

    def add_noise(self, task, std=0.1):
        """Augment support set with noisy copies."""
        new_support_x = list(task.support_x.data)
        new_support_y = list(task.support_y)

        for i in range(len(task.support_y)):
            row = task.support_x.data[i]
            noisy = [v + self._rng.gauss(0, std) for v in row]
            new_support_x.append(noisy)
            new_support_y.append(task.support_y[i])

        return Task(
            Tensor(new_support_x), new_support_y,
            task.query_x, task.query_y, task.classes
        )

    def mixup_support(self, task, alpha=0.2, n_extra=None):
        """Generate mixup samples within each class in the support set."""
        new_support_x = list(task.support_x.data)
        new_support_y = list(task.support_y)

        if n_extra is None:
            n_extra = len(task.support_y)

        # Group by class
        class_indices = {}
        for i, label in enumerate(task.support_y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)

        for _ in range(n_extra):
            # Pick a random class
            label = self._rng.choice(list(class_indices.keys()))
            indices = class_indices[label]
            if len(indices) < 2:
                continue
            i, j = self._rng.sample(indices, 2)
            lam = self._rng.betavariate(alpha, alpha) if alpha > 0 else 0.5
            row_i = task.support_x.data[i]
            row_j = task.support_x.data[j]
            mixed = [lam * a + (1 - lam) * b for a, b in zip(row_i, row_j)]
            new_support_x.append(mixed)
            new_support_y.append(label)

        return Task(
            Tensor(new_support_x), new_support_y,
            task.query_x, task.query_y, task.classes
        )

    def random_scale(self, task, low=0.9, high=1.1):
        """Scale support features randomly."""
        new_support_x = []
        for row in task.support_x.data:
            scaled = [v * (low + self._rng.random() * (high - low)) for v in row]
            new_support_x.append(scaled)
        return Task(
            Tensor(new_support_x), list(task.support_y),
            task.query_x, task.query_y, task.classes
        )


# ============================================================
# Meta Learning Rate Schedule
# ============================================================

class MetaScheduler:
    """Learning rate scheduler for meta-learning outer loop."""

    def __init__(self, algorithm, schedule='cosine', T_max=100, eta_min=0.0001):
        """
        algorithm: MAML or Reptile (must have outer_lr attribute)
        schedule: 'cosine', 'step', or 'linear'
        T_max: total epochs for scheduling
        eta_min: minimum learning rate
        """
        self.algorithm = algorithm
        self.schedule = schedule
        self.T_max = T_max
        self.eta_min = eta_min
        self._initial_lr = algorithm.outer_lr
        self._step_count = 0

    def step(self):
        """Update the outer learning rate."""
        self._step_count += 1
        t = self._step_count

        if self.schedule == 'cosine':
            lr = self.eta_min + (self._initial_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * t / self.T_max)) / 2
        elif self.schedule == 'step':
            # Decay by 0.1 every T_max/3 steps
            decay_step = max(1, self.T_max // 3)
            factor = 0.1 ** (t // decay_step)
            lr = max(self._initial_lr * factor, self.eta_min)
        elif self.schedule == 'linear':
            lr = self._initial_lr * (1 - t / self.T_max)
            lr = max(lr, self.eta_min)
        else:
            lr = self._initial_lr

        self.algorithm.outer_lr = lr
        return lr

    def get_lr(self):
        return self.algorithm.outer_lr


# ============================================================
# Data generation utilities for testing
# ============================================================

def make_few_shot_data(n_classes=10, samples_per_class=20, feat_dim=8, seed=42):
    """Generate synthetic classification data suitable for few-shot learning.

    Each class has a distinct cluster center. Returns (X, Y).
    """
    rng = random.Random(seed)
    X_data = []
    Y_data = []

    for c in range(n_classes):
        # Random cluster center
        center = [rng.gauss(0, 3) for _ in range(feat_dim)]
        for _ in range(samples_per_class):
            point = [center[d] + rng.gauss(0, 0.5) for d in range(feat_dim)]
            X_data.append(point)
            Y_data.append(c)

    return Tensor(X_data), Y_data
