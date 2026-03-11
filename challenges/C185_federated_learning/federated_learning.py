"""
C185: Federated Learning
========================
Distributed machine learning where multiple clients train locally and a central
server aggregates model updates -- without sharing raw data.

Components:
- FederatedModel: Simple neural network with get/set params, train, evaluate
- FederatedClient: Local training on private data, gradient clipping, differential privacy
- FederatedServer: Orchestrates rounds, aggregates updates, evaluates global model
- Aggregation strategies: FedAvg, FedProx, FedSGD, weighted, trimmed mean, Krum
- Non-IID data partitioning: IID, label-skew, Dirichlet, quantity-skew
- Privacy: Gaussian noise (DP-SGD style), secure aggregation simulation
- Compression: top-k sparsification, quantization for communication efficiency
- Diagnostics: convergence tracking, client drift, participation stats

Built with NumPy only. No external ML libraries.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import copy
import json


# ============================================================
# Neural Network Layer
# ============================================================

class DenseLayer:
    """A single dense (fully connected) layer."""

    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        # He initialization
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        # Cache for backprop
        self._input = None
        self._pre_act = None
        self._output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        self._pre_act = x @ self.W + self.b
        if self.activation == "relu":
            self._output = np.maximum(0, self._pre_act)
        elif self.activation == "sigmoid":
            self._output = 1.0 / (1.0 + np.exp(-np.clip(self._pre_act, -500, 500)))
        elif self.activation == "softmax":
            shifted = self._pre_act - np.max(self._pre_act, axis=-1, keepdims=True)
            exp_x = np.exp(shifted)
            self._output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        elif self.activation == "none":
            self._output = self._pre_act
        else:
            self._output = self._pre_act
        return self._output

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (grad_input, grad_W, grad_b)."""
        if self.activation == "relu":
            grad_act = grad_output * (self._pre_act > 0).astype(float)
        elif self.activation == "sigmoid":
            s = self._output
            grad_act = grad_output * s * (1 - s)
        elif self.activation == "softmax":
            # For softmax + cross-entropy, grad_output is already d_loss/d_pre_act
            grad_act = grad_output
        elif self.activation == "none":
            grad_act = grad_output
        else:
            grad_act = grad_output

        grad_W = self._input.T @ grad_act
        grad_b = np.sum(grad_act, axis=0)
        grad_input = grad_act @ self.W.T
        return grad_input, grad_W, grad_b


# ============================================================
# Federated Model
# ============================================================

class FederatedModel:
    """Simple feedforward neural network for federated learning."""

    def __init__(self, layer_sizes: List[int], activations: Optional[List[str]] = None):
        self.layer_sizes = layer_sizes
        self.layers: List[DenseLayer] = []
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["none"]
        for i in range(len(layer_sizes) - 1):
            act = activations[i] if i < len(activations) else "relu"
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i + 1], act))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        if logits.shape[-1] > 1:
            return np.argmax(logits, axis=-1)
        return (logits > 0).astype(int).flatten()

    def get_params(self) -> List[np.ndarray]:
        """Get all model parameters as a flat list [W1, b1, W2, b2, ...]."""
        params = []
        for layer in self.layers:
            params.append(layer.W.copy())
            params.append(layer.b.copy())
        return params

    def set_params(self, params: List[np.ndarray]):
        """Set model parameters from flat list."""
        idx = 0
        for layer in self.layers:
            layer.W = params[idx].copy()
            layer.b = params[idx + 1].copy()
            idx += 2

    def copy(self) -> 'FederatedModel':
        """Create a deep copy of this model."""
        new_model = FederatedModel(self.layer_sizes,
                                    [l.activation for l in self.layers])
        new_model.set_params(self.get_params())
        return new_model

    def compute_loss(self, x: np.ndarray, y: np.ndarray,
                     loss_type: str = "cross_entropy") -> float:
        """Compute loss on data."""
        logits = self.forward(x)
        if loss_type == "cross_entropy":
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            n = x.shape[0]
            y_int = y.astype(int)
            loss = -np.mean(log_softmax[np.arange(n), y_int])
            return float(loss)
        elif loss_type == "mse":
            return float(np.mean((logits.flatten() - y.flatten()) ** 2))
        raise ValueError(f"Unknown loss type: {loss_type}")

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate accuracy and loss."""
        predictions = self.predict(x)
        y_int = y.astype(int).flatten()
        accuracy = float(np.mean(predictions == y_int))
        loss = self.compute_loss(x, y)
        return {"accuracy": accuracy, "loss": loss}

    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01,
                   loss_type: str = "cross_entropy",
                   proximal_term: float = 0.0,
                   global_params: Optional[List[np.ndarray]] = None) -> float:
        """One training step with backpropagation. Returns loss."""
        n = x.shape[0]
        logits = self.forward(x)

        # Compute gradient of loss w.r.t. logits
        if loss_type == "cross_entropy":
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
            y_int = y.astype(int)
            grad = probs.copy()
            grad[np.arange(n), y_int] -= 1.0
            grad /= n
            log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            loss = float(-np.mean(log_softmax[np.arange(n), y_int]))
        elif loss_type == "mse":
            diff = logits.flatten() - y.flatten()
            loss = float(np.mean(diff ** 2))
            grad = (2.0 * diff / n).reshape(logits.shape)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Backprop through layers
        grads = []
        g = grad
        for layer in reversed(self.layers):
            g, gW, gb = layer.backward(g)
            grads.append((gW, gb))
        grads.reverse()

        # Update with optional proximal term (FedProx)
        params = self.get_params()
        for i, layer in enumerate(self.layers):
            gW, gb = grads[i]
            if proximal_term > 0.0 and global_params is not None:
                gW = gW + proximal_term * (layer.W - global_params[2 * i])
                gb = gb + proximal_term * (layer.b - global_params[2 * i + 1])
            layer.W -= lr * gW
            layer.b -= lr * gb

        return loss


# ============================================================
# Data Partitioning (Non-IID simulation)
# ============================================================

def partition_iid(x: np.ndarray, y: np.ndarray, num_clients: int
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data IID across clients."""
    n = len(x)
    indices = np.random.permutation(n)
    splits = np.array_split(indices, num_clients)
    return [(x[s], y[s]) for s in splits]


def partition_label_skew(x: np.ndarray, y: np.ndarray, num_clients: int,
                         labels_per_client: int = 2
                         ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each client gets data from only a few label classes."""
    classes = np.unique(y)
    num_classes = len(classes)
    # Assign labels to clients in round-robin fashion
    client_data = [[] for _ in range(num_clients)]
    for c_idx in range(num_clients):
        assigned = []
        for j in range(labels_per_client):
            assigned.append(classes[(c_idx * labels_per_client + j) % num_classes])
        client_data[c_idx] = assigned

    result = []
    for c_idx in range(num_clients):
        mask = np.isin(y, client_data[c_idx])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            # Fallback: give some random data
            indices = np.random.choice(len(x), size=max(1, len(x) // num_clients), replace=False)
        result.append((x[indices], y[indices]))
    return result


def partition_dirichlet(x: np.ndarray, y: np.ndarray, num_clients: int,
                        alpha: float = 0.5
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Dirichlet distribution-based non-IID partition."""
    classes = np.unique(y)
    num_classes = len(classes)
    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(np.full(num_clients, alpha))
        # Scale to actual counts
        counts = (proportions * len(class_indices)).astype(int)
        # Fix rounding
        counts[-1] = len(class_indices) - np.sum(counts[:-1])
        counts = np.maximum(counts, 0)
        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(class_indices[start:end].tolist())
            start = end

    result = []
    for i in range(num_clients):
        indices = np.array(client_indices[i], dtype=int)
        if len(indices) == 0:
            indices = np.random.choice(len(x), size=1, replace=False)
        result.append((x[indices], y[indices]))
    return result


def partition_quantity_skew(x: np.ndarray, y: np.ndarray, num_clients: int,
                            min_ratio: float = 0.1
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Clients get different amounts of data (quantity skew)."""
    n = len(x)
    indices = np.random.permutation(n)
    # Generate random proportions with minimum
    raw = np.random.dirichlet(np.ones(num_clients))
    # Scale so minimum is at least min_ratio / num_clients
    floor = min_ratio / num_clients
    proportions = raw * (1.0 - floor * num_clients) + floor
    proportions /= proportions.sum()
    counts = (proportions * n).astype(int)
    counts[-1] = n - np.sum(counts[:-1])
    counts = np.maximum(counts, 1)

    result = []
    start = 0
    for i in range(num_clients):
        end = min(start + counts[i], n)
        if start >= n:
            # Wrap around
            idx = np.random.choice(n, size=1, replace=False)
            result.append((x[idx], y[idx]))
        else:
            result.append((x[indices[start:end]], y[indices[start:end]]))
        start = end
    return result


# ============================================================
# Federated Client
# ============================================================

class FederatedClient:
    """A client in federated learning -- trains locally on private data."""

    def __init__(self, client_id: int, x: np.ndarray, y: np.ndarray,
                 model: FederatedModel, lr: float = 0.01,
                 local_epochs: int = 5, batch_size: int = 32):
        self.client_id = client_id
        self.x = x
        self.y = y
        self.model = model.copy()
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.train_history: List[float] = []
        self.num_samples = len(x)

    def receive_global_model(self, params: List[np.ndarray]):
        """Receive updated global model parameters."""
        self.model.set_params(params)

    def train_local(self, proximal_term: float = 0.0,
                    global_params: Optional[List[np.ndarray]] = None,
                    max_grad_norm: Optional[float] = None,
                    dp_noise_scale: float = 0.0) -> Dict[str, Any]:
        """Train locally on private data. Returns update info."""
        initial_params = self.model.get_params()
        total_loss = 0.0
        num_steps = 0

        for epoch in range(self.local_epochs):
            indices = np.random.permutation(self.num_samples)
            for start in range(0, self.num_samples, self.batch_size):
                end = min(start + self.batch_size, self.num_samples)
                batch_idx = indices[start:end]
                bx, by = self.x[batch_idx], self.y[batch_idx]
                loss = self.model.train_step(bx, by, self.lr,
                                             proximal_term=proximal_term,
                                             global_params=global_params)
                total_loss += loss
                num_steps += 1

        # Compute update (delta)
        final_params = self.model.get_params()
        delta = [f - i for f, i in zip(final_params, initial_params)]

        # Gradient clipping
        if max_grad_norm is not None:
            total_norm = np.sqrt(sum(np.sum(d ** 2) for d in delta))
            if total_norm > max_grad_norm:
                scale = max_grad_norm / total_norm
                delta = [d * scale for d in delta]

        # Differential privacy noise
        if dp_noise_scale > 0.0:
            delta = [d + np.random.randn(*d.shape) * dp_noise_scale for d in delta]

        avg_loss = total_loss / max(num_steps, 1)
        self.train_history.append(avg_loss)

        return {
            "client_id": self.client_id,
            "delta": delta,
            "num_samples": self.num_samples,
            "avg_loss": avg_loss,
            "num_steps": num_steps,
        }

    def evaluate_local(self) -> Dict[str, float]:
        """Evaluate model on local data."""
        return self.model.evaluate(self.x, self.y)


# ============================================================
# Aggregation Strategies
# ============================================================

def aggregate_fedavg(updates: List[Dict[str, Any]],
                     global_params: List[np.ndarray]) -> List[np.ndarray]:
    """FedAvg: weighted average of client updates by number of samples."""
    total_samples = sum(u["num_samples"] for u in updates)
    new_params = [np.zeros_like(p) for p in global_params]
    for u in updates:
        weight = u["num_samples"] / total_samples
        delta = u["delta"]
        for i in range(len(new_params)):
            new_params[i] += weight * (global_params[i] + delta[i])
    return new_params


def aggregate_fedsgd(updates: List[Dict[str, Any]],
                     global_params: List[np.ndarray],
                     server_lr: float = 1.0) -> List[np.ndarray]:
    """FedSGD: average gradients and apply with server learning rate."""
    total_samples = sum(u["num_samples"] for u in updates)
    avg_delta = [np.zeros_like(p) for p in global_params]
    for u in updates:
        weight = u["num_samples"] / total_samples
        for i, d in enumerate(u["delta"]):
            avg_delta[i] += weight * d
    return [p + server_lr * d for p, d in zip(global_params, avg_delta)]


def aggregate_trimmed_mean(updates: List[Dict[str, Any]],
                           global_params: List[np.ndarray],
                           trim_ratio: float = 0.1) -> List[np.ndarray]:
    """Trimmed mean aggregation -- robust to Byzantine clients."""
    num_clients = len(updates)
    trim_count = max(1, int(num_clients * trim_ratio))
    if trim_count * 2 >= num_clients:
        trim_count = 0  # Can't trim if too few clients

    new_params = []
    for i in range(len(global_params)):
        # Stack all client param updates
        all_updates = np.stack([global_params[i] + u["delta"][i] for u in updates])
        if trim_count > 0:
            sorted_updates = np.sort(all_updates, axis=0)
            trimmed = sorted_updates[trim_count:-trim_count]
            new_params.append(np.mean(trimmed, axis=0))
        else:
            new_params.append(np.mean(all_updates, axis=0))
    return new_params


def aggregate_krum(updates: List[Dict[str, Any]],
                   global_params: List[np.ndarray],
                   num_byzantine: int = 0) -> List[np.ndarray]:
    """Krum aggregation -- selects the update closest to most others."""
    num_clients = len(updates)
    if num_clients <= 2 * num_byzantine + 2:
        # Fallback to simple average
        return aggregate_fedavg(updates, global_params)

    # Compute all param vectors
    vectors = []
    for u in updates:
        v = np.concatenate([d.flatten() for d in u["delta"]])
        vectors.append(v)
    vectors = np.array(vectors)

    # Compute pairwise distances
    n = len(vectors)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(vectors[i] - vectors[j])
            distances[i][j] = d
            distances[j][i] = d

    # For each client, sum distances to n - num_byzantine - 2 closest
    k = n - num_byzantine - 2
    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(distances[i])
        # sorted_dists[0] is 0 (self), take next k
        scores[i] = np.sum(sorted_dists[1:k + 1])

    # Select the client with smallest score
    best = int(np.argmin(scores))
    return [global_params[i] + updates[best]["delta"][i]
            for i in range(len(global_params))]


def aggregate_median(updates: List[Dict[str, Any]],
                     global_params: List[np.ndarray]) -> List[np.ndarray]:
    """Coordinate-wise median aggregation -- robust to outliers."""
    new_params = []
    for i in range(len(global_params)):
        all_updates = np.stack([global_params[i] + u["delta"][i] for u in updates])
        new_params.append(np.median(all_updates, axis=0))
    return new_params


# ============================================================
# Compression
# ============================================================

def compress_topk(delta: List[np.ndarray], k_ratio: float = 0.1
                  ) -> List[Tuple[np.ndarray, np.ndarray, Tuple]]:
    """Top-k sparsification: keep only k% largest magnitude values."""
    compressed = []
    for d in delta:
        flat = d.flatten()
        k = max(1, int(len(flat) * k_ratio))
        top_indices = np.argsort(np.abs(flat))[-k:]
        values = flat[top_indices]
        compressed.append((top_indices, values, d.shape))
    return compressed


def decompress_topk(compressed: List[Tuple[np.ndarray, np.ndarray, Tuple]]
                    ) -> List[np.ndarray]:
    """Decompress top-k sparsified updates."""
    result = []
    for indices, values, shape in compressed:
        flat = np.zeros(int(np.prod(shape)))
        flat[indices] = values
        result.append(flat.reshape(shape))
    return result


def compress_quantize(delta: List[np.ndarray], bits: int = 8
                      ) -> List[Tuple[np.ndarray, float, float, Tuple]]:
    """Uniform quantization to reduce communication."""
    compressed = []
    levels = 2 ** bits - 1
    for d in delta:
        d_min, d_max = float(np.min(d)), float(np.max(d))
        if d_max - d_min < 1e-10:
            quantized = np.zeros_like(d, dtype=np.uint8)
            compressed.append((quantized, d_min, d_max, d.shape))
        else:
            normalized = (d - d_min) / (d_max - d_min)
            quantized = np.round(normalized * levels).astype(np.uint8)
            compressed.append((quantized, d_min, d_max, d.shape))
    return compressed


def decompress_quantize(compressed: List[Tuple[np.ndarray, float, float, Tuple]],
                        bits: int = 8) -> List[np.ndarray]:
    """Decompress quantized updates."""
    levels = 2 ** bits - 1
    result = []
    for quantized, d_min, d_max, shape in compressed:
        if d_max - d_min < 1e-10:
            result.append(np.full(shape, d_min))
        else:
            normalized = quantized.astype(float) / levels
            result.append(normalized * (d_max - d_min) + d_min)
    return result


# ============================================================
# Secure Aggregation (Simulation)
# ============================================================

class SecureAggregator:
    """Simulates secure aggregation -- server only sees the sum of updates."""

    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.masks: Dict[int, List[np.ndarray]] = {}

    def generate_masks(self, param_shapes: List[Tuple]) -> Dict[int, List[np.ndarray]]:
        """Generate random masks that sum to zero (pairwise cancellation)."""
        self.masks = {i: [np.zeros(s) for s in param_shapes]
                      for i in range(self.num_clients)}
        # For each pair (i, j), generate random mask and assign +/- to each
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                for p in range(len(param_shapes)):
                    mask = np.random.randn(*param_shapes[p])
                    self.masks[i][p] += mask
                    self.masks[j][p] -= mask
        return self.masks

    def mask_update(self, client_id: int, delta: List[np.ndarray]) -> List[np.ndarray]:
        """Client masks their update before sending."""
        return [d + m for d, m in zip(delta, self.masks[client_id])]

    def aggregate_masked(self, masked_updates: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Sum masked updates -- masks cancel out."""
        result = [np.zeros_like(m) for m in masked_updates[0]]
        for update in masked_updates:
            for i in range(len(result)):
                result[i] += update[i]
        # Divide by number of clients to get average
        return [r / len(masked_updates) for r in result]


# ============================================================
# Federated Server
# ============================================================

class FederatedServer:
    """Central server that orchestrates federated learning."""

    def __init__(self, model: FederatedModel,
                 aggregation: str = "fedavg",
                 client_fraction: float = 1.0,
                 server_lr: float = 1.0,
                 trim_ratio: float = 0.1,
                 num_byzantine: int = 0):
        self.global_model = model
        self.aggregation = aggregation
        self.client_fraction = client_fraction
        self.server_lr = server_lr
        self.trim_ratio = trim_ratio
        self.num_byzantine = num_byzantine
        self.clients: List[FederatedClient] = []
        self.round_history: List[Dict[str, Any]] = []
        self.current_round = 0

    def add_client(self, client: FederatedClient):
        """Register a client."""
        self.clients.append(client)

    def select_clients(self) -> List[FederatedClient]:
        """Select a random subset of clients for this round."""
        num_selected = max(1, int(len(self.clients) * self.client_fraction))
        indices = np.random.choice(len(self.clients), size=num_selected, replace=False)
        return [self.clients[i] for i in indices]

    def aggregate(self, updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Aggregate client updates using the configured strategy."""
        global_params = self.global_model.get_params()
        if self.aggregation == "fedavg":
            return aggregate_fedavg(updates, global_params)
        elif self.aggregation == "fedsgd":
            return aggregate_fedsgd(updates, global_params, self.server_lr)
        elif self.aggregation == "trimmed_mean":
            return aggregate_trimmed_mean(updates, global_params, self.trim_ratio)
        elif self.aggregation == "krum":
            return aggregate_krum(updates, global_params, self.num_byzantine)
        elif self.aggregation == "median":
            return aggregate_median(updates, global_params)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def run_round(self, proximal_term: float = 0.0,
                  max_grad_norm: Optional[float] = None,
                  dp_noise_scale: float = 0.0,
                  compression: Optional[str] = None,
                  compression_ratio: float = 0.1) -> Dict[str, Any]:
        """Execute one round of federated learning."""
        self.current_round += 1
        global_params = self.global_model.get_params()

        # Select and distribute
        selected = self.select_clients()
        for client in selected:
            client.receive_global_model(global_params)

        # Local training
        updates = []
        for client in selected:
            update = client.train_local(
                proximal_term=proximal_term,
                global_params=global_params if proximal_term > 0 else None,
                max_grad_norm=max_grad_norm,
                dp_noise_scale=dp_noise_scale,
            )

            # Optional compression
            if compression == "topk":
                comp = compress_topk(update["delta"], compression_ratio)
                update["delta"] = decompress_topk(comp)
                update["compressed_size"] = sum(len(c[0]) for c in comp)
            elif compression == "quantize":
                comp = compress_quantize(update["delta"])
                update["delta"] = decompress_quantize(comp)

            updates.append(update)

        # Aggregate
        new_params = self.aggregate(updates)
        self.global_model.set_params(new_params)

        # Record round info
        avg_loss = float(np.mean([u["avg_loss"] for u in updates]))
        round_info = {
            "round": self.current_round,
            "num_clients_selected": len(selected),
            "avg_client_loss": avg_loss,
            "client_ids": [u["client_id"] for u in updates],
        }
        self.round_history.append(round_info)
        return round_info

    def evaluate_global(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the global model."""
        return self.global_model.evaluate(x, y)

    def run_training(self, num_rounds: int, test_x: Optional[np.ndarray] = None,
                     test_y: Optional[np.ndarray] = None,
                     eval_every: int = 1,
                     **kwargs) -> List[Dict[str, Any]]:
        """Run multiple rounds of federated learning."""
        results = []
        for r in range(num_rounds):
            round_info = self.run_round(**kwargs)
            if test_x is not None and test_y is not None and (r + 1) % eval_every == 0:
                eval_result = self.evaluate_global(test_x, test_y)
                round_info.update(eval_result)
            results.append(round_info)
        return results


# ============================================================
# Client Drift Analysis
# ============================================================

def compute_client_drift(clients: List[FederatedClient],
                         global_params: List[np.ndarray]) -> Dict[str, Any]:
    """Measure how far each client's model has drifted from global."""
    drifts = []
    for client in clients:
        client_params = client.model.get_params()
        drift = np.sqrt(sum(
            np.sum((cp - gp) ** 2) for cp, gp in zip(client_params, global_params)
        ))
        drifts.append({"client_id": client.client_id, "drift": float(drift)})

    all_drifts = [d["drift"] for d in drifts]
    return {
        "per_client": drifts,
        "mean_drift": float(np.mean(all_drifts)),
        "max_drift": float(np.max(all_drifts)),
        "std_drift": float(np.std(all_drifts)),
    }


# ============================================================
# Personalization: Local Fine-tuning
# ============================================================

class PersonalizedClient(FederatedClient):
    """Client with personalization -- maintains a local adaptation layer."""

    def __init__(self, client_id: int, x: np.ndarray, y: np.ndarray,
                 model: FederatedModel, lr: float = 0.01,
                 local_epochs: int = 5, batch_size: int = 32,
                 personalization_epochs: int = 3):
        super().__init__(client_id, x, y, model, lr, local_epochs, batch_size)
        self.personalization_epochs = personalization_epochs
        self.personalized_model: Optional[FederatedModel] = None

    def personalize(self):
        """Fine-tune on local data after receiving global model."""
        self.personalized_model = self.model.copy()
        n = self.num_samples
        for epoch in range(self.personalization_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = indices[start:end]
                self.personalized_model.train_step(
                    self.x[batch_idx], self.y[batch_idx], self.lr * 0.1
                )

    def evaluate_personalized(self) -> Dict[str, float]:
        """Evaluate the personalized model on local data."""
        if self.personalized_model is None:
            self.personalize()
        return self.personalized_model.evaluate(self.x, self.y)


# ============================================================
# Asynchronous Federated Learning
# ============================================================

class AsyncFederatedServer(FederatedServer):
    """Asynchronous FL -- clients update at different times/speeds."""

    def __init__(self, model: FederatedModel,
                 staleness_penalty: float = 0.5,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.staleness_penalty = staleness_penalty
        self.client_rounds: Dict[int, int] = {}  # Last round each client participated

    def apply_async_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single client's update with staleness weighting."""
        client_id = update["client_id"]
        last_round = self.client_rounds.get(client_id, 0)
        staleness = self.current_round - last_round

        # Weight decreases with staleness
        weight = 1.0 / (1.0 + self.staleness_penalty * staleness)

        global_params = self.global_model.get_params()
        new_params = [gp + weight * d for gp, d in zip(global_params, update["delta"])]
        self.global_model.set_params(new_params)

        self.client_rounds[client_id] = self.current_round
        self.current_round += 1

        return {
            "round": self.current_round,
            "client_id": client_id,
            "staleness": staleness,
            "weight": weight,
            "avg_loss": update["avg_loss"],
        }


# ============================================================
# Data Generation Utilities
# ============================================================

def make_classification_data(n_samples: int = 500, n_features: int = 10,
                             n_classes: int = 3, seed: Optional[int] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randn(n_samples, n_features)
    # Create class centers
    centers = np.random.randn(n_classes, n_features) * 2
    y = np.zeros(n_samples, dtype=int)
    per_class = n_samples // n_classes
    for c in range(n_classes):
        start = c * per_class
        end = start + per_class if c < n_classes - 1 else n_samples
        x[start:end] += centers[c]
        y[start:end] = c
    # Shuffle
    perm = np.random.permutation(n_samples)
    return x[perm], y[perm]


def make_regression_data(n_samples: int = 500, n_features: int = 5,
                         noise: float = 0.1, seed: Optional[int] = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    if seed is not None:
        np.random.seed(seed)
    x = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    y = x @ true_w + np.random.randn(n_samples) * noise
    return x, y


# ============================================================
# Diagnostics
# ============================================================

class FederatedDiagnostics:
    """Track and analyze federated learning metrics."""

    def __init__(self):
        self.round_data: List[Dict[str, Any]] = []

    def record_round(self, round_info: Dict[str, Any]):
        """Record metrics for a round."""
        self.round_data.append(round_info)

    def convergence_rate(self) -> Optional[float]:
        """Estimate convergence rate from loss history."""
        losses = [r.get("avg_client_loss", r.get("loss")) for r in self.round_data]
        losses = [l for l in losses if l is not None]
        if len(losses) < 2:
            return None
        # Average ratio of consecutive losses
        ratios = [losses[i + 1] / losses[i] for i in range(len(losses) - 1)
                  if losses[i] > 1e-10]
        if not ratios:
            return None
        return float(np.mean(ratios))

    def participation_stats(self) -> Dict[str, Any]:
        """Analyze client participation across rounds."""
        all_ids = []
        for r in self.round_data:
            ids = r.get("client_ids", [])
            all_ids.extend(ids)
        if not all_ids:
            return {"total_rounds": len(self.round_data)}
        unique_ids = set(all_ids)
        from collections import Counter
        counts = Counter(all_ids)
        return {
            "total_rounds": len(self.round_data),
            "unique_clients": len(unique_ids),
            "avg_participation": float(np.mean(list(counts.values()))),
            "min_participation": min(counts.values()),
            "max_participation": max(counts.values()),
        }

    def loss_summary(self) -> Dict[str, Any]:
        """Summarize loss trajectory."""
        losses = [r.get("avg_client_loss", r.get("loss")) for r in self.round_data]
        losses = [l for l in losses if l is not None]
        if not losses:
            return {}
        return {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "min_loss": float(np.min(losses)),
            "improvement": losses[0] - losses[-1] if len(losses) > 1 else 0,
            "num_rounds": len(losses),
        }

    def accuracy_summary(self) -> Dict[str, Any]:
        """Summarize accuracy trajectory."""
        accs = [r.get("accuracy") for r in self.round_data]
        accs = [a for a in accs if a is not None]
        if not accs:
            return {}
        return {
            "initial_accuracy": accs[0],
            "final_accuracy": accs[-1],
            "max_accuracy": float(np.max(accs)),
            "improvement": accs[-1] - accs[0] if len(accs) > 1 else 0,
        }
