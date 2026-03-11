"""
C168: Neural Architecture Search (NAS)
Composing: C167 (Bayesian Optimization) + C012 (Code Evolver) + C140 (Neural Network)

Searches for optimal neural network architectures using:
- Random search (baseline)
- Bayesian NAS (BO over architecture hyperparameters)
- Evolutionary NAS (genetic programming over architecture graphs)
- BOHB-style combined BO + evolutionary search
- Multi-objective NAS (accuracy vs complexity tradeoff)
- Performance prediction (surrogate models to skip expensive training)
"""

import numpy as np
import sys
import os
import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C167_bayesian_optimization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C012_code_evolver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from bayesian_optimization import (
    BayesianOptimizer, BatchBayesianOptimizer, MultiObjectiveBO,
    ExpectedImprovement, UpperConfidenceBound, BayesOptHistory
)
from evolver import (
    Node, NodeType, EvolutionConfig, Evolver, TestCase,
    make_const, make_var, make_binary, make_unary,
    tournament_select, crossover, mutate, evaluate as eval_tree,
    all_nodes, format_node
)
from neural_network import (
    Sequential, Dense, Activation, Dropout, BatchNorm,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    SGD, Adam, RMSProp, fit, evaluate as nn_evaluate, Tensor
)


# ---------------------------------------------------------------------------
# Search Space Definition
# ---------------------------------------------------------------------------

class LayerType(Enum):
    DENSE = 'dense'
    ACTIVATION = 'activation'
    DROPOUT = 'dropout'
    BATCHNORM = 'batchnorm'


@dataclass
class LayerSpec:
    """Specification for a single layer."""
    layer_type: LayerType
    params: dict = field(default_factory=dict)

    def build(self, input_size=None):
        """Build a concrete layer from this spec."""
        if self.layer_type == LayerType.DENSE:
            return Dense(
                self.params.get('input_size', input_size),
                self.params['output_size'],
                init=self.params.get('init', 'xavier')
            )
        elif self.layer_type == LayerType.ACTIVATION:
            return Activation(self.params.get('name', 'relu'))
        elif self.layer_type == LayerType.DROPOUT:
            return Dropout(self.params.get('rate', 0.5))
        elif self.layer_type == LayerType.BATCHNORM:
            return BatchNorm(self.params.get('num_features', input_size))
        raise ValueError(f"Unknown layer type: {self.layer_type}")


@dataclass
class ArchitectureSpec:
    """Full architecture specification."""
    layers: list  # list of LayerSpec
    optimizer: str = 'adam'
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 50

    def build(self, input_size):
        """Build a Sequential model from this spec."""
        model = Sequential()
        current_size = input_size
        for spec in self.layers:
            if spec.layer_type == LayerType.DENSE:
                spec.params['input_size'] = current_size
                current_size = spec.params['output_size']
            elif spec.layer_type == LayerType.BATCHNORM:
                spec.params['num_features'] = current_size
            model.add(spec.build(current_size))
        return model

    def param_count(self, input_size):
        """Estimate parameter count without building."""
        count = 0
        current = input_size
        for spec in self.layers:
            if spec.layer_type == LayerType.DENSE:
                out = spec.params['output_size']
                count += current * out + out  # weights + bias
                current = out
            elif spec.layer_type == LayerType.BATCHNORM:
                count += 2 * current  # gamma + beta
        return count

    def depth(self):
        """Number of dense layers."""
        return sum(1 for s in self.layers if s.layer_type == LayerType.DENSE)

    def summary(self):
        """Human-readable summary."""
        parts = []
        for s in self.layers:
            if s.layer_type == LayerType.DENSE:
                parts.append(f"Dense({s.params.get('output_size', '?')})")
            elif s.layer_type == LayerType.ACTIVATION:
                parts.append(s.params.get('name', 'relu'))
            elif s.layer_type == LayerType.DROPOUT:
                parts.append(f"Drop({s.params.get('rate', 0.5):.1f})")
            elif s.layer_type == LayerType.BATCHNORM:
                parts.append("BN")
        return " -> ".join(parts)


@dataclass
class SearchSpace:
    """Defines the space of possible architectures."""
    input_size: int
    output_size: int
    min_layers: int = 1
    max_layers: int = 5
    layer_sizes: list = field(default_factory=lambda: [16, 32, 64, 128, 256])
    activations: list = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid'])
    dropout_rates: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    use_batchnorm: bool = True
    optimizers: list = field(default_factory=lambda: ['adam', 'sgd'])
    lr_range: tuple = (1e-4, 1e-1)
    output_activation: str = None  # None means no final activation

    def sample_random(self, rng=None):
        """Sample a random architecture from this space."""
        if rng is None:
            rng = np.random.RandomState()

        n_layers = rng.randint(self.min_layers, self.max_layers + 1)
        layers = []

        for i in range(n_layers):
            size = int(rng.choice(self.layer_sizes))
            layers.append(LayerSpec(LayerType.DENSE, {'output_size': size}))

            act = rng.choice(self.activations)
            layers.append(LayerSpec(LayerType.ACTIVATION, {'name': act}))

            if self.use_batchnorm and rng.random() < 0.3:
                layers.append(LayerSpec(LayerType.BATCHNORM))

            drop = float(rng.choice(self.dropout_rates))
            if drop > 0:
                layers.append(LayerSpec(LayerType.DROPOUT, {'rate': drop}))

        # Output layer
        layers.append(LayerSpec(LayerType.DENSE, {'output_size': self.output_size}))
        if self.output_activation:
            layers.append(LayerSpec(LayerType.ACTIVATION, {'name': self.output_activation}))

        opt = rng.choice(self.optimizers)
        lr = float(np.exp(rng.uniform(np.log(self.lr_range[0]), np.log(self.lr_range[1]))))

        return ArchitectureSpec(layers=layers, optimizer=opt, lr=lr)

    def encode(self, arch):
        """Encode architecture as a fixed-length vector for BO."""
        # Vector: [n_layers, size_1..size_max, act_1..act_max, drop_1..drop_max, bn_1..bn_max, lr, opt]
        dense_layers = [s for s in arch.layers if s.layer_type == LayerType.DENSE][:-1]  # exclude output
        n = len(dense_layers)

        vec = [n / self.max_layers]  # normalized layer count

        # Encode each layer slot
        for i in range(self.max_layers):
            if i < n:
                dl = dense_layers[i]
                size_idx = self.layer_sizes.index(dl.params['output_size']) if dl.params['output_size'] in self.layer_sizes else 0
                vec.append(size_idx / max(1, len(self.layer_sizes) - 1))

                # Find activation after this dense layer
                dense_idx = arch.layers.index(dl)
                act_name = 'relu'
                for j in range(dense_idx + 1, min(dense_idx + 4, len(arch.layers))):
                    if arch.layers[j].layer_type == LayerType.ACTIVATION:
                        act_name = arch.layers[j].params.get('name', 'relu')
                        break
                act_idx = self.activations.index(act_name) if act_name in self.activations else 0
                vec.append(act_idx / max(1, len(self.activations) - 1))

                # Find dropout
                drop = 0.0
                for j in range(dense_idx + 1, min(dense_idx + 4, len(arch.layers))):
                    if arch.layers[j].layer_type == LayerType.DROPOUT:
                        drop = arch.layers[j].params.get('rate', 0.0)
                        break
                vec.append(drop)

                # Batchnorm
                has_bn = any(
                    arch.layers[j].layer_type == LayerType.BATCHNORM
                    for j in range(dense_idx + 1, min(dense_idx + 4, len(arch.layers)))
                )
                vec.append(1.0 if has_bn else 0.0)
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0])

        # Learning rate (log-scaled)
        lr_norm = (np.log(arch.lr) - np.log(self.lr_range[0])) / (np.log(self.lr_range[1]) - np.log(self.lr_range[0]))
        vec.append(float(np.clip(lr_norm, 0, 1)))

        # Optimizer
        opt_idx = self.optimizers.index(arch.optimizer) if arch.optimizer in self.optimizers else 0
        vec.append(opt_idx / max(1, len(self.optimizers) - 1))

        return np.array(vec)

    def decode(self, vec):
        """Decode a vector back into an ArchitectureSpec."""
        n_layers = max(1, min(self.max_layers, int(round(vec[0] * self.max_layers))))
        layers = []
        idx = 1

        for i in range(n_layers):
            # Size
            size_idx = int(round(np.clip(vec[idx], 0, 1) * (len(self.layer_sizes) - 1)))
            size = self.layer_sizes[size_idx]
            layers.append(LayerSpec(LayerType.DENSE, {'output_size': size}))
            idx += 1

            # Activation
            act_idx = int(round(np.clip(vec[idx], 0, 1) * (len(self.activations) - 1)))
            layers.append(LayerSpec(LayerType.ACTIVATION, {'name': self.activations[act_idx]}))
            idx += 1

            # Dropout
            drop = float(np.clip(vec[idx], 0, 1))
            if drop > 0.05:
                drop_idx = np.argmin([abs(d - drop) for d in self.dropout_rates if d > 0] or [0.1])
                actual_rates = [d for d in self.dropout_rates if d > 0]
                drop = actual_rates[drop_idx] if actual_rates else 0.1
                layers.append(LayerSpec(LayerType.DROPOUT, {'rate': drop}))
            idx += 1

            # Batchnorm
            if vec[idx] > 0.5 and self.use_batchnorm:
                layers.append(LayerSpec(LayerType.BATCHNORM))
            idx += 1

        # Skip unused slots
        for i in range(n_layers, self.max_layers):
            idx += 4

        # Output layer
        layers.append(LayerSpec(LayerType.DENSE, {'output_size': self.output_size}))
        if self.output_activation:
            layers.append(LayerSpec(LayerType.ACTIVATION, {'name': self.output_activation}))

        # Learning rate
        lr_norm = float(np.clip(vec[idx], 0, 1))
        lr = float(np.exp(np.log(self.lr_range[0]) + lr_norm * (np.log(self.lr_range[1]) - np.log(self.lr_range[0]))))
        idx += 1

        # Optimizer
        opt_idx = int(round(np.clip(vec[idx], 0, 1) * (len(self.optimizers) - 1)))
        optimizer = self.optimizers[opt_idx]

        return ArchitectureSpec(layers=layers, optimizer=optimizer, lr=lr)

    def vector_dim(self):
        """Dimensionality of encoded vector."""
        return 1 + self.max_layers * 4 + 2  # n_layers + 4 per slot + lr + opt


# ---------------------------------------------------------------------------
# Architecture Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating an architecture."""
    architecture: ArchitectureSpec
    loss: float
    accuracy: float = 0.0
    param_count: int = 0
    train_time: float = 0.0  # epochs actually run
    converged: bool = False


class ArchitectureEvaluator:
    """Evaluates architectures by training and testing."""

    def __init__(self, X_train, Y_train, X_val=None, Y_val=None,
                 loss_fn=None, max_epochs=50, early_stop=5, seed=42):
        # Store as numpy for shape info, convert to Tensor for training
        self._X_np = np.array(X_train.data if isinstance(X_train, Tensor) else X_train)
        self._Y_np = np.array(Y_train.data if isinstance(Y_train, Tensor) else Y_train)
        self.X_train = self._to_tensor(X_train)
        self.Y_train = self._to_tensor(Y_train)
        self.X_val = self._to_tensor(X_val) if X_val is not None else None
        self.Y_val = self._to_tensor(Y_val) if Y_val is not None else None
        self.loss_fn = loss_fn or MSELoss()
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.seed = seed
        self.eval_count = 0
        self.cache = {}

    @staticmethod
    def _to_tensor(data):
        if isinstance(data, Tensor):
            return data
        if hasattr(data, 'tolist'):
            return Tensor(data.tolist())
        return Tensor(data)

    def evaluate(self, arch, epochs=None):
        """Train and evaluate an architecture. Returns EvalResult."""
        key = arch.summary()
        if key in self.cache:
            return self.cache[key]

        epochs = epochs or min(arch.epochs, self.max_epochs)
        input_size = self._X_np.shape[1] if len(self._X_np.shape) > 1 else 1

        try:
            model = arch.build(input_size)
        except Exception:
            return EvalResult(architecture=arch, loss=float('inf'), param_count=0)

        # Build optimizer
        if arch.optimizer == 'adam':
            opt = Adam(lr=arch.lr)
        elif arch.optimizer == 'sgd':
            opt = SGD(lr=arch.lr, momentum=0.9)
        else:
            opt = RMSProp(lr=arch.lr)

        # Train with early stopping
        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            try:
                history = fit(model, self.X_train, self.Y_train, self.loss_fn, opt,
                              epochs=1, batch_size=arch.batch_size, verbose=False)
                current_loss = history['loss'][-1]
            except Exception:
                return EvalResult(architecture=arch, loss=float('inf'), param_count=0)

            if math.isnan(current_loss) or math.isinf(current_loss):
                return EvalResult(architecture=arch, loss=float('inf'), param_count=0,
                                  train_time=epoch + 1)

            if current_loss < best_loss:
                best_loss = current_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop:
                    break

        # Evaluate on validation if available
        if self.X_val is not None:
            try:
                val_loss = nn_evaluate(model, self.X_val, self.Y_val, self.loss_fn)
                final_loss = val_loss
            except Exception:
                final_loss = best_loss
        else:
            final_loss = best_loss

        param_count = arch.param_count(input_size)
        converged = patience < self.early_stop

        result = EvalResult(
            architecture=arch,
            loss=final_loss,
            param_count=param_count,
            train_time=epoch + 1,
            converged=converged
        )

        self.cache[key] = result
        self.eval_count += 1
        return result


# ---------------------------------------------------------------------------
# Performance Predictor (Surrogate)
# ---------------------------------------------------------------------------

class PerformancePredictor:
    """Surrogate model that predicts architecture performance without training."""

    def __init__(self, search_space, seed=42):
        self.space = search_space
        self.seed = seed
        self.X_observed = []
        self.y_observed = []
        self._weights = None
        self._bias = 0.0

    def observe(self, arch, loss):
        """Record an observation."""
        vec = self.space.encode(arch)
        self.X_observed.append(vec)
        self.y_observed.append(loss)
        self._weights = None  # invalidate

    def predict(self, arch):
        """Predict loss for an architecture."""
        if len(self.X_observed) < 3:
            return float('inf'), float('inf')  # mean, uncertainty

        self._fit()
        vec = self.space.encode(arch)
        pred = float(np.dot(vec, self._weights) + self._bias)

        # Estimate uncertainty from residuals
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        preds = X @ self._weights + self._bias
        residuals = y - preds
        std = float(np.std(residuals)) if len(residuals) > 1 else float('inf')

        # Distance-based uncertainty boost
        dists = np.sqrt(np.sum((X - vec) ** 2, axis=1))
        min_dist = float(np.min(dists))
        uncertainty = std * (1 + min_dist)

        return pred, uncertainty

    def _fit(self):
        """Fit linear model to observations."""
        if self._weights is not None:
            return

        X = np.array(self.X_observed)
        y = np.array(self.y_observed)

        # Ridge regression
        lam = 0.1
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        Xty = X.T @ y
        try:
            self._weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self._weights = np.zeros(X.shape[1])
        self._bias = float(np.mean(y) - np.mean(X @ self._weights))

    def rank_candidates(self, candidates, n_best=5):
        """Rank candidate architectures by predicted performance."""
        scored = []
        for arch in candidates:
            pred, unc = self.predict(arch)
            # Lower confidence bound (explore uncertain, exploit good predictions)
            lcb = pred - 0.5 * unc
            scored.append((lcb, pred, unc, arch))
        scored.sort(key=lambda x: x[0])
        return [(s[3], s[1], s[2]) for s in scored[:n_best]]


# ---------------------------------------------------------------------------
# Search History
# ---------------------------------------------------------------------------

@dataclass
class NASHistory:
    """Tracks the full NAS search history."""
    evaluations: list = field(default_factory=list)
    best_loss: float = float('inf')
    best_arch: ArchitectureSpec = None

    def record(self, result):
        """Record an evaluation result."""
        self.evaluations.append(result)
        if result.loss < self.best_loss:
            self.best_loss = result.loss
            self.best_arch = result.architecture

    def get_best(self):
        """Return best architecture and its loss."""
        return self.best_arch, self.best_loss

    def convergence_curve(self):
        """Return best loss at each evaluation."""
        curve = []
        best = float('inf')
        for r in self.evaluations:
            best = min(best, r.loss)
            curve.append(best)
        return curve

    def summary(self):
        """Summary statistics."""
        losses = [r.loss for r in self.evaluations if r.loss < float('inf')]
        return {
            'total_evaluations': len(self.evaluations),
            'valid_evaluations': len(losses),
            'best_loss': self.best_loss,
            'mean_loss': float(np.mean(losses)) if losses else float('inf'),
            'best_arch': self.best_arch.summary() if self.best_arch else None,
            'best_params': self.best_arch.param_count(0) if self.best_arch else 0,
        }


# ---------------------------------------------------------------------------
# NAS Strategy: Random Search
# ---------------------------------------------------------------------------

class RandomNAS:
    """Random architecture search (baseline)."""

    def __init__(self, search_space, evaluator, seed=42):
        self.space = search_space
        self.evaluator = evaluator
        self.rng = np.random.RandomState(seed)
        self.history = NASHistory()

    def search(self, n_trials=20, verbose=False):
        """Run random search."""
        for i in range(n_trials):
            arch = self.space.sample_random(self.rng)
            result = self.evaluator.evaluate(arch)
            self.history.record(result)

            if verbose:
                print(f"Trial {i+1}/{n_trials}: loss={result.loss:.4f} "
                      f"params={result.param_count} [{arch.summary()}]")

        return self.history.get_best()


# ---------------------------------------------------------------------------
# NAS Strategy: Bayesian NAS
# ---------------------------------------------------------------------------

class BayesianNAS:
    """Bayesian optimization over architecture search space."""

    def __init__(self, search_space, evaluator, acquisition=None,
                 n_initial=5, seed=42):
        self.space = search_space
        self.evaluator = evaluator
        self.n_initial = n_initial
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.history = NASHistory()

        dim = search_space.vector_dim()
        bounds = np.array([[0.0, 1.0]] * dim)
        self.optimizer = BayesianOptimizer(
            bounds=bounds,
            acquisition=acquisition or ExpectedImprovement(xi=0.01),
            noise_variance=1e-3,
            n_initial=0,
            seed=seed
        )

    def search(self, n_trials=20, verbose=False):
        """Run Bayesian NAS."""
        # Initial random sampling
        for i in range(min(self.n_initial, n_trials)):
            arch = self.space.sample_random(self.rng)
            result = self.evaluator.evaluate(arch)
            self.history.record(result)

            vec = self.space.encode(arch)
            self.optimizer.observe(vec, -result.loss)  # maximize negative loss

            if verbose:
                print(f"Initial {i+1}/{self.n_initial}: loss={result.loss:.4f} [{arch.summary()}]")

        # BO-guided search
        for i in range(self.n_initial, n_trials):
            try:
                vec_next, _ = self.optimizer.suggest()
                arch = self.space.decode(vec_next)
            except Exception:
                arch = self.space.sample_random(self.rng)

            result = self.evaluator.evaluate(arch)
            self.history.record(result)

            vec = self.space.encode(arch)
            self.optimizer.observe(vec, -result.loss)

            if verbose:
                best_arch, best_loss = self.history.get_best()
                print(f"Trial {i+1}/{n_trials}: loss={result.loss:.4f} "
                      f"best={best_loss:.4f} [{arch.summary()}]")

        return self.history.get_best()


# ---------------------------------------------------------------------------
# NAS Strategy: Evolutionary NAS
# ---------------------------------------------------------------------------

class ArchGenome:
    """Genome representation for evolutionary NAS."""

    def __init__(self, arch, fitness=None):
        self.arch = arch
        self.fitness = fitness

    def copy(self):
        layers_copy = [LayerSpec(s.layer_type, dict(s.params)) for s in self.arch.layers]
        arch_copy = ArchitectureSpec(
            layers=layers_copy,
            optimizer=self.arch.optimizer,
            lr=self.arch.lr,
            batch_size=self.arch.batch_size,
            epochs=self.arch.epochs
        )
        return ArchGenome(arch_copy, self.fitness)


class EvolutionaryNAS:
    """Evolutionary search over architecture space."""

    def __init__(self, search_space, evaluator, population_size=20,
                 tournament_size=3, mutation_rate=0.3, crossover_rate=0.5,
                 elitism=2, seed=42):
        self.space = search_space
        self.evaluator = evaluator
        self.pop_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.rng = np.random.RandomState(seed)
        self.history = NASHistory()
        self.population = []

    def _init_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.pop_size):
            arch = self.space.sample_random(self.rng)
            result = self.evaluator.evaluate(arch)
            genome = ArchGenome(arch, fitness=-result.loss)  # higher is better
            self.population.append(genome)
            self.history.record(result)

    def _tournament_select(self):
        """Select individual by tournament."""
        contestants = [self.population[i] for i in
                       self.rng.choice(len(self.population), self.tournament_size, replace=False)]
        return max(contestants, key=lambda g: g.fitness).copy()

    def _mutate(self, genome):
        """Mutate an architecture genome."""
        arch = genome.arch
        r = self.rng.random()

        if r < 0.25:
            # Add a layer
            if arch.depth() < self.space.max_layers:
                dense_indices = [i for i, s in enumerate(arch.layers)
                                 if s.layer_type == LayerType.DENSE]
                if dense_indices:
                    insert_at = self.rng.choice(dense_indices)
                    size = int(self.rng.choice(self.space.layer_sizes))
                    act = self.rng.choice(self.space.activations)
                    new_layers = [
                        LayerSpec(LayerType.DENSE, {'output_size': size}),
                        LayerSpec(LayerType.ACTIVATION, {'name': act})
                    ]
                    arch.layers = arch.layers[:insert_at] + new_layers + arch.layers[insert_at:]

        elif r < 0.5:
            # Remove a layer
            dense_layers = [(i, s) for i, s in enumerate(arch.layers)
                            if s.layer_type == LayerType.DENSE]
            if len(dense_layers) > 2:  # keep at least output + 1 hidden
                # Remove a non-output dense layer and its associated layers
                removable = dense_layers[:-1]  # don't remove output
                idx, _ = removable[self.rng.randint(len(removable))]
                # Remove dense and following activation/dropout/batchnorm
                end = idx + 1
                while end < len(arch.layers) and arch.layers[end].layer_type != LayerType.DENSE:
                    end += 1
                arch.layers = arch.layers[:idx] + arch.layers[end:]

        elif r < 0.7:
            # Change layer size
            dense_layers = [(i, s) for i, s in enumerate(arch.layers)
                            if s.layer_type == LayerType.DENSE]
            if len(dense_layers) > 1:
                idx, spec = dense_layers[self.rng.randint(len(dense_layers) - 1)]
                spec.params['output_size'] = int(self.rng.choice(self.space.layer_sizes))

        elif r < 0.85:
            # Change activation
            act_layers = [(i, s) for i, s in enumerate(arch.layers)
                          if s.layer_type == LayerType.ACTIVATION]
            if act_layers:
                idx, spec = act_layers[self.rng.randint(len(act_layers))]
                spec.params['name'] = self.rng.choice(self.space.activations)

        else:
            # Change learning rate
            arch.lr = float(np.exp(self.rng.uniform(
                np.log(self.space.lr_range[0]),
                np.log(self.space.lr_range[1])
            )))

        return genome

    def _crossover(self, parent1, parent2):
        """Crossover two architecture genomes."""
        # Take layers from parent1 up to a point, then from parent2
        dense1 = [s for s in parent1.arch.layers if s.layer_type == LayerType.DENSE]
        dense2 = [s for s in parent2.arch.layers if s.layer_type == LayerType.DENSE]

        if len(dense1) <= 1 or len(dense2) <= 1:
            return parent1.copy()

        # Split point (exclude output layer)
        split1 = self.rng.randint(1, len(dense1))
        split2 = self.rng.randint(1, len(dense2))

        # Rebuild: layers from p1[:split1] + layers from p2[split2:-1] + output
        child_layers = []

        # Gather layer blocks from parent1
        block_idx = 0
        for i, s in enumerate(parent1.arch.layers):
            if s.layer_type == LayerType.DENSE:
                block_idx += 1
            if block_idx <= split1:
                child_layers.append(LayerSpec(s.layer_type, dict(s.params)))

        # Gather layer blocks from parent2
        block_idx = 0
        for i, s in enumerate(parent2.arch.layers):
            if s.layer_type == LayerType.DENSE:
                block_idx += 1
            if block_idx > split2:
                child_layers.append(LayerSpec(s.layer_type, dict(s.params)))

        # Ensure output layer exists
        if not child_layers or child_layers[-1].layer_type != LayerType.DENSE:
            child_layers.append(LayerSpec(LayerType.DENSE, {'output_size': self.space.output_size}))
        else:
            # Make sure last dense has output_size
            child_layers[-1].params['output_size'] = self.space.output_size

        # Enforce depth limit
        dense_count = sum(1 for s in child_layers if s.layer_type == LayerType.DENSE)
        while dense_count > self.space.max_layers + 1:
            # Remove a non-output dense block
            for i, s in enumerate(child_layers[:-1]):
                if s.layer_type == LayerType.DENSE:
                    end = i + 1
                    while end < len(child_layers) - 1 and child_layers[end].layer_type != LayerType.DENSE:
                        end += 1
                    child_layers = child_layers[:i] + child_layers[end:]
                    break
            dense_count = sum(1 for s in child_layers if s.layer_type == LayerType.DENSE)

        # Inherit lr/optimizer from fitter parent
        better = parent1 if (parent1.fitness or 0) >= (parent2.fitness or 0) else parent2
        child_arch = ArchitectureSpec(
            layers=child_layers,
            optimizer=better.arch.optimizer,
            lr=better.arch.lr,
            batch_size=better.arch.batch_size,
            epochs=better.arch.epochs
        )
        return ArchGenome(child_arch)

    def search(self, n_generations=10, verbose=False):
        """Run evolutionary search."""
        self._init_population()

        for gen in range(n_generations):
            # Sort by fitness (higher is better)
            self.population.sort(key=lambda g: g.fitness or float('-inf'), reverse=True)

            new_pop = []
            # Elitism
            for i in range(self.elitism):
                new_pop.append(self.population[i].copy())

            # Fill rest
            while len(new_pop) < self.pop_size:
                if self.rng.random() < self.crossover_rate:
                    p1 = self._tournament_select()
                    p2 = self._tournament_select()
                    child = self._crossover(p1, p2)
                else:
                    child = self._tournament_select()

                if self.rng.random() < self.mutation_rate:
                    child = self._mutate(child)

                # Evaluate
                result = self.evaluator.evaluate(child.arch)
                child.fitness = -result.loss
                self.history.record(result)
                new_pop.append(child)

            self.population = new_pop

            if verbose:
                best_arch, best_loss = self.history.get_best()
                gen_best = max(self.population, key=lambda g: g.fitness or float('-inf'))
                print(f"Gen {gen+1}/{n_generations}: gen_best={-gen_best.fitness:.4f} "
                      f"overall_best={best_loss:.4f}")

        return self.history.get_best()


# ---------------------------------------------------------------------------
# NAS Strategy: BOHB (Bayesian Optimization + HyperBand)
# ---------------------------------------------------------------------------

class BOHBNAS:
    """Combined Bayesian optimization and evolutionary search (BOHB-style).

    Uses BO for exploitation (refining promising regions) and evolution
    for exploration (structural changes). Runs in alternating phases.
    """

    def __init__(self, search_space, evaluator, n_initial=5,
                 bo_fraction=0.5, seed=42):
        self.space = search_space
        self.evaluator = evaluator
        self.n_initial = n_initial
        self.bo_fraction = bo_fraction
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.history = NASHistory()
        self.predictor = PerformancePredictor(search_space, seed)

        dim = search_space.vector_dim()
        bounds = np.array([[0.0, 1.0]] * dim)
        self.bo = BayesianOptimizer(
            bounds=bounds,
            acquisition=ExpectedImprovement(xi=0.01),
            noise_variance=1e-3,
            n_initial=0,
            seed=seed
        )

        self.evo = EvolutionaryNAS(
            search_space, evaluator,
            population_size=10, tournament_size=3,
            mutation_rate=0.4, crossover_rate=0.5,
            elitism=2, seed=seed
        )

    def search(self, n_trials=30, verbose=False):
        """Run BOHB search."""
        # Phase 1: Initial random exploration
        for i in range(min(self.n_initial, n_trials)):
            arch = self.space.sample_random(self.rng)
            result = self.evaluator.evaluate(arch)
            self.history.record(result)
            self.predictor.observe(arch, result.loss)

            vec = self.space.encode(arch)
            self.bo.observe(vec, -result.loss)

            if verbose:
                print(f"Init {i+1}/{self.n_initial}: loss={result.loss:.4f}")

        # Phase 2: Alternating BO and evolutionary steps
        for i in range(self.n_initial, n_trials):
            use_bo = self.rng.random() < self.bo_fraction

            if use_bo:
                # BO-guided suggestion
                try:
                    vec_next, _ = self.bo.suggest()
                    arch = self.space.decode(vec_next)
                except Exception:
                    arch = self.space.sample_random(self.rng)
                tag = "BO"
            else:
                # Evolutionary mutation of best architectures
                if self.history.evaluations:
                    # Pick from top performers
                    sorted_evals = sorted(
                        [r for r in self.history.evaluations if r.loss < float('inf')],
                        key=lambda r: r.loss
                    )
                    if sorted_evals:
                        parent = sorted_evals[self.rng.randint(min(5, len(sorted_evals)))]
                        genome = ArchGenome(parent.architecture)
                        genome = self.evo._mutate(genome)
                        arch = genome.arch
                    else:
                        arch = self.space.sample_random(self.rng)
                else:
                    arch = self.space.sample_random(self.rng)
                tag = "EVO"

            # Use predictor to filter bad candidates
            pred_loss, pred_unc = self.predictor.predict(arch)
            if pred_loss > self.history.best_loss * 3 and pred_unc < pred_loss * 0.5:
                # Skip likely-bad architecture, try another
                arch = self.space.sample_random(self.rng)
                tag = "RAND"

            result = self.evaluator.evaluate(arch)
            self.history.record(result)
            self.predictor.observe(arch, result.loss)

            vec = self.space.encode(arch)
            self.bo.observe(vec, -result.loss)

            if verbose:
                best_arch, best_loss = self.history.get_best()
                print(f"Trial {i+1}/{n_trials} [{tag}]: loss={result.loss:.4f} "
                      f"best={best_loss:.4f}")

        return self.history.get_best()


# ---------------------------------------------------------------------------
# NAS Strategy: Multi-Objective NAS
# ---------------------------------------------------------------------------

class MultiObjectiveNAS:
    """Multi-objective NAS optimizing accuracy AND model complexity."""

    def __init__(self, search_space, evaluator, complexity_weight=1.0,
                 seed=42):
        self.space = search_space
        self.evaluator = evaluator
        self.complexity_weight = complexity_weight
        self.rng = np.random.RandomState(seed)
        self.history = NASHistory()
        self.pareto_front = []  # list of (loss, complexity, arch)

    def _dominates(self, a, b):
        """Does point a dominate point b? (both objectives minimized)"""
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    def _update_pareto(self, loss, complexity, arch):
        """Update Pareto front with new point."""
        point = (loss, complexity, arch)
        # Remove dominated points
        self.pareto_front = [p for p in self.pareto_front
                             if not self._dominates(point, p)]
        # Add if not dominated
        if not any(self._dominates(p, point) for p in self.pareto_front):
            self.pareto_front.append(point)

    def search(self, n_trials=20, verbose=False):
        """Run multi-objective NAS."""
        for i in range(n_trials):
            arch = self.space.sample_random(self.rng)
            result = self.evaluator.evaluate(arch)
            self.history.record(result)

            input_size = self.evaluator._X_np.shape[1] if len(self.evaluator._X_np.shape) > 1 else 1
            complexity = arch.param_count(input_size) / 1000.0  # normalize
            self._update_pareto(result.loss, complexity, arch)

            if verbose:
                print(f"Trial {i+1}/{n_trials}: loss={result.loss:.4f} "
                      f"params={result.param_count} pareto_size={len(self.pareto_front)}")

        return self.history.get_best()

    def get_pareto_front(self):
        """Return Pareto-optimal architectures."""
        return [(loss, comp, arch) for loss, comp, arch in self.pareto_front]

    def get_knee_point(self):
        """Find the knee point on the Pareto front (best tradeoff)."""
        if not self.pareto_front:
            return None

        if len(self.pareto_front) == 1:
            return self.pareto_front[0]

        # Normalize objectives
        losses = [p[0] for p in self.pareto_front]
        complexities = [p[1] for p in self.pareto_front]

        loss_range = max(losses) - min(losses) if max(losses) > min(losses) else 1
        comp_range = max(complexities) - min(complexities) if max(complexities) > min(complexities) else 1

        # Find point with maximum distance from line connecting extremes
        sorted_front = sorted(self.pareto_front, key=lambda p: p[0])
        p1 = np.array([(sorted_front[0][0] - min(losses)) / loss_range,
                        (sorted_front[0][1] - min(complexities)) / comp_range])
        p2 = np.array([(sorted_front[-1][0] - min(losses)) / loss_range,
                        (sorted_front[-1][1] - min(complexities)) / comp_range])

        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return sorted_front[0]

        line_unit = line_vec / line_len
        best_dist = -1
        best_point = sorted_front[0]

        for p in sorted_front:
            pn = np.array([(p[0] - min(losses)) / loss_range,
                           (p[1] - min(complexities)) / comp_range])
            proj = np.dot(pn - p1, line_unit)
            closest = p1 + proj * line_unit
            dist = np.linalg.norm(pn - closest)
            if dist > best_dist:
                best_dist = dist
                best_point = p

        return best_point


# ---------------------------------------------------------------------------
# Convenience: create_nas factory
# ---------------------------------------------------------------------------

def create_nas(search_space, evaluator, method='bayesian', **kwargs):
    """Factory function for NAS strategies.

    Args:
        search_space: SearchSpace instance
        evaluator: ArchitectureEvaluator instance
        method: 'random', 'bayesian', 'evolutionary', 'bohb', 'multi_objective'
        **kwargs: Strategy-specific parameters

    Returns:
        NAS strategy instance
    """
    strategies = {
        'random': RandomNAS,
        'bayesian': BayesianNAS,
        'evolutionary': EvolutionaryNAS,
        'bohb': BOHBNAS,
        'multi_objective': MultiObjectiveNAS,
    }

    if method not in strategies:
        raise ValueError(f"Unknown NAS method: {method}. Available: {list(strategies.keys())}")

    # Filter kwargs to only pass valid ones
    import inspect
    cls = strategies[method]
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(search_space, evaluator, **filtered)
