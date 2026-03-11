"""
C169: Hyperparameter Tuning
Composing C167 (Bayesian Optimization) + C140 (Neural Network)

Automated hyperparameter optimization with multiple strategies:
- Grid Search: exhaustive enumeration
- Random Search: random sampling with optional sobol-like coverage
- Bayesian Tuning: GP-based BO from C167
- Successive Halving: budget-aware early stopping
- Hyperband: multi-bracket successive halving
- Population-Based Training: exploit+explore during training

Supports continuous, discrete, categorical, and log-scale parameters.
"""

import sys
import os
import math
import time
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C167_bayesian_optimization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from bayesian_optimization import (
    BayesianOptimizer, ExpectedImprovement, UpperConfidenceBound,
    ProbabilityOfImprovement, BayesOptHistory
)
from neural_network import (
    Sequential, Dense, Activation, Dropout, BatchNorm,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    SGD, Adam, RMSProp, fit, evaluate as nn_evaluate, Tensor
)


# ---------------------------------------------------------------------------
# Parameter types
# ---------------------------------------------------------------------------

class ParamType(Enum):
    CONTINUOUS = 'continuous'
    DISCRETE = 'discrete'
    CATEGORICAL = 'categorical'
    LOG_CONTINUOUS = 'log_continuous'
    INTEGER = 'integer'


@dataclass
class HyperparameterDef:
    """Definition of a single hyperparameter."""
    name: str
    param_type: ParamType
    low: float = 0.0
    high: float = 1.0
    choices: list = field(default_factory=list)
    step: float = 1.0  # for discrete
    default: Any = None
    condition: Optional[dict] = None  # {'param': name, 'value': val} -- only active when condition met

    def sample(self, rng):
        """Sample a random value for this parameter."""
        if self.param_type == ParamType.CONTINUOUS:
            return rng.uniform(self.low, self.high)
        elif self.param_type == ParamType.LOG_CONTINUOUS:
            log_low = math.log(max(self.low, 1e-10))
            log_high = math.log(max(self.high, 1e-10))
            return math.exp(rng.uniform(log_low, log_high))
        elif self.param_type == ParamType.INTEGER:
            return int(rng.integers(int(self.low), int(self.high) + 1))
        elif self.param_type == ParamType.DISCRETE:
            n_steps = max(1, int(round((self.high - self.low) / self.step)))
            idx = int(rng.integers(0, n_steps + 1))
            return self.low + idx * self.step
        elif self.param_type == ParamType.CATEGORICAL:
            return self.choices[int(rng.integers(0, len(self.choices)))]
        return self.default

    def encode(self, value):
        """Encode value to [0, 1] for BO."""
        if self.param_type == ParamType.CONTINUOUS:
            if self.high == self.low:
                return 0.5
            return (value - self.low) / (self.high - self.low)
        elif self.param_type == ParamType.LOG_CONTINUOUS:
            log_low = math.log(max(self.low, 1e-10))
            log_high = math.log(max(self.high, 1e-10))
            if log_high == log_low:
                return 0.5
            return (math.log(max(value, 1e-10)) - log_low) / (log_high - log_low)
        elif self.param_type == ParamType.INTEGER:
            if self.high == self.low:
                return 0.5
            return (value - self.low) / (self.high - self.low)
        elif self.param_type == ParamType.DISCRETE:
            if self.high == self.low:
                return 0.5
            return (value - self.low) / (self.high - self.low)
        elif self.param_type == ParamType.CATEGORICAL:
            if len(self.choices) <= 1:
                return 0.5
            idx = self.choices.index(value) if value in self.choices else 0
            return idx / (len(self.choices) - 1)
        return 0.5

    def decode(self, normalized):
        """Decode [0, 1] value back to parameter space."""
        normalized = max(0.0, min(1.0, normalized))
        if self.param_type == ParamType.CONTINUOUS:
            return self.low + normalized * (self.high - self.low)
        elif self.param_type == ParamType.LOG_CONTINUOUS:
            log_low = math.log(max(self.low, 1e-10))
            log_high = math.log(max(self.high, 1e-10))
            return math.exp(log_low + normalized * (log_high - log_low))
        elif self.param_type == ParamType.INTEGER:
            return int(round(self.low + normalized * (self.high - self.low)))
        elif self.param_type == ParamType.DISCRETE:
            raw = self.low + normalized * (self.high - self.low)
            n_steps = max(1, int(round((self.high - self.low) / self.step)))
            step_idx = int(round((raw - self.low) / self.step))
            step_idx = max(0, min(n_steps, step_idx))
            return self.low + step_idx * self.step
        elif self.param_type == ParamType.CATEGORICAL:
            idx = int(round(normalized * (len(self.choices) - 1)))
            idx = max(0, min(len(self.choices) - 1, idx))
            return self.choices[idx]
        return self.default

    def grid_values(self, n_points=5):
        """Generate grid values for this parameter."""
        if self.param_type == ParamType.CATEGORICAL:
            return list(self.choices)
        elif self.param_type == ParamType.INTEGER:
            vals = list(range(int(self.low), int(self.high) + 1))
            if len(vals) > n_points:
                step = max(1, len(vals) // n_points)
                vals = vals[::step]
                if vals[-1] != int(self.high):
                    vals.append(int(self.high))
            return vals
        elif self.param_type == ParamType.DISCRETE:
            vals = []
            v = self.low
            while v <= self.high + 1e-10:
                vals.append(round(v, 10))
                v += self.step
            if len(vals) > n_points:
                step = max(1, len(vals) // n_points)
                vals = vals[::step]
            return vals
        elif self.param_type == ParamType.LOG_CONTINUOUS:
            log_low = math.log(max(self.low, 1e-10))
            log_high = math.log(max(self.high, 1e-10))
            return [math.exp(log_low + i * (log_high - log_low) / max(1, n_points - 1))
                    for i in range(n_points)]
        else:  # CONTINUOUS
            return [self.low + i * (self.high - self.low) / max(1, n_points - 1)
                    for i in range(n_points)]


# ---------------------------------------------------------------------------
# Hyperparameter space
# ---------------------------------------------------------------------------

class HyperparameterSpace:
    """Defines searchable hyperparameter space."""

    def __init__(self):
        self.params = {}  # name -> HyperparameterDef
        self._order = []  # insertion order

    def add_continuous(self, name, low, high, default=None, log_scale=False):
        ptype = ParamType.LOG_CONTINUOUS if log_scale else ParamType.CONTINUOUS
        self.params[name] = HyperparameterDef(
            name=name, param_type=ptype, low=low, high=high,
            default=default if default is not None else (low + high) / 2
        )
        if name not in self._order:
            self._order.append(name)
        return self

    def add_integer(self, name, low, high, default=None):
        self.params[name] = HyperparameterDef(
            name=name, param_type=ParamType.INTEGER, low=low, high=high,
            default=default if default is not None else (low + high) // 2
        )
        if name not in self._order:
            self._order.append(name)
        return self

    def add_discrete(self, name, low, high, step, default=None):
        self.params[name] = HyperparameterDef(
            name=name, param_type=ParamType.DISCRETE, low=low, high=high, step=step,
            default=default if default is not None else low
        )
        if name not in self._order:
            self._order.append(name)
        return self

    def add_categorical(self, name, choices, default=None):
        self.params[name] = HyperparameterDef(
            name=name, param_type=ParamType.CATEGORICAL, choices=list(choices),
            default=default if default is not None else choices[0]
        )
        if name not in self._order:
            self._order.append(name)
        return self

    def add_conditional(self, name, param_type, condition_param, condition_value, **kwargs):
        """Add a parameter that is only active when condition_param == condition_value."""
        pdef = HyperparameterDef(
            name=name, param_type=param_type,
            condition={'param': condition_param, 'value': condition_value},
            **kwargs
        )
        self.params[name] = pdef
        if name not in self._order:
            self._order.append(name)
        return self

    def sample(self, rng=None):
        """Sample a random configuration."""
        if rng is None:
            rng = np.random.default_rng()
        config = {}
        for name in self._order:
            pdef = self.params[name]
            if pdef.condition is not None:
                cond_param = pdef.condition['param']
                cond_val = pdef.condition['value']
                if config.get(cond_param) != cond_val:
                    continue  # skip inactive conditional param
            config[name] = pdef.sample(rng)
        return HPConfig(config)

    def encode(self, config):
        """Encode config to numpy vector for BO."""
        vec = []
        for name in self._order:
            pdef = self.params[name]
            if pdef.condition is not None:
                # Always encode conditional params (use default if inactive)
                val = config.values.get(name, pdef.default)
                if val is None:
                    val = pdef.default if pdef.default is not None else 0.5
                    vec.append(0.5)
                    continue
            else:
                val = config.values.get(name, pdef.default)
            vec.append(pdef.encode(val))
        return np.array(vec, dtype=np.float64)

    def decode(self, vec):
        """Decode numpy vector to HPConfig."""
        config = {}
        for i, name in enumerate(self._order):
            pdef = self.params[name]
            config[name] = pdef.decode(vec[i])
        # Apply conditions: remove inactive conditional params
        for name in list(config.keys()):
            pdef = self.params[name]
            if pdef.condition is not None:
                cond_param = pdef.condition['param']
                cond_val = pdef.condition['value']
                if config.get(cond_param) != cond_val:
                    del config[name]
        return HPConfig(config)

    @property
    def dim(self):
        return len(self._order)

    def get_defaults(self):
        return HPConfig({name: self.params[name].default for name in self._order
                         if self.params[name].default is not None})

    def grid(self, n_points=5):
        """Generate grid of all parameter combinations."""
        from itertools import product as iterproduct
        param_grids = []
        active_names = []
        for name in self._order:
            pdef = self.params[name]
            if pdef.condition is not None:
                continue  # skip conditional for grid
            active_names.append(name)
            param_grids.append(pdef.grid_values(n_points))
        configs = []
        for combo in iterproduct(*param_grids):
            config = dict(zip(active_names, combo))
            # Add conditional params if conditions met
            for name in self._order:
                pdef = self.params[name]
                if pdef.condition is not None:
                    cond_param = pdef.condition['param']
                    cond_val = pdef.condition['value']
                    if config.get(cond_param) == cond_val:
                        config[name] = pdef.default
            configs.append(HPConfig(config))
        return configs


# ---------------------------------------------------------------------------
# Config and Trial
# ---------------------------------------------------------------------------

@dataclass
class HPConfig:
    """A concrete hyperparameter configuration."""
    values: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key):
        return key in self.values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def copy(self):
        return HPConfig(dict(self.values))

    def summary(self):
        parts = []
        for k, v in sorted(self.values.items()):
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v}")
        return ', '.join(parts)

    def __repr__(self):
        return f"HPConfig({self.summary()})"


@dataclass
class Trial:
    """Result of evaluating one HP configuration."""
    trial_id: int
    config: HPConfig
    objective: float = float('inf')
    metrics: dict = field(default_factory=dict)
    budget: float = 1.0  # fraction of full budget used
    status: str = 'pending'  # pending, running, completed, pruned
    duration: float = 0.0

    def is_better_than(self, other, minimize=True):
        if minimize:
            return self.objective < other.objective
        return self.objective > other.objective


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

class HPTuningHistory:
    """Track all trials and compute statistics."""

    def __init__(self, minimize=True):
        self.trials = []
        self.minimize = minimize
        self._best_trial = None

    def record(self, trial):
        self.trials.append(trial)
        if trial.status == 'completed':
            if self._best_trial is None or trial.is_better_than(self._best_trial, self.minimize):
                self._best_trial = trial

    @property
    def best_trial(self):
        return self._best_trial

    @property
    def best_config(self):
        return self._best_trial.config if self._best_trial else None

    @property
    def best_objective(self):
        return self._best_trial.objective if self._best_trial else None

    def completed_trials(self):
        return [t for t in self.trials if t.status == 'completed']

    def convergence_curve(self):
        """Running best objective over completed trials."""
        curve = []
        best = float('inf') if self.minimize else float('-inf')
        for t in self.trials:
            if t.status != 'completed':
                continue
            if self.minimize:
                best = min(best, t.objective)
            else:
                best = max(best, t.objective)
            curve.append(best)
        return curve

    def top_k(self, k=5):
        """Return top k trials."""
        completed = self.completed_trials()
        completed.sort(key=lambda t: t.objective, reverse=not self.minimize)
        return completed[:k]

    def summary(self):
        completed = self.completed_trials()
        pruned = [t for t in self.trials if t.status == 'pruned']
        objectives = [t.objective for t in completed]
        return {
            'total_trials': len(self.trials),
            'completed': len(completed),
            'pruned': len(pruned),
            'best_objective': self.best_objective,
            'best_config': self.best_config.summary() if self.best_config else None,
            'mean_objective': float(np.mean(objectives)) if objectives else None,
            'std_objective': float(np.std(objectives)) if objectives else None,
        }


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback:
    """Monitor intermediate results and decide whether to prune a trial."""

    def __init__(self, patience=5, min_delta=1e-4, minimize=True):
        self.patience = patience
        self.min_delta = min_delta
        self.minimize = minimize
        self.best_value = float('inf') if minimize else float('-inf')
        self.wait = 0

    def report(self, value):
        """Report intermediate value. Returns True if should stop."""
        improved = False
        if self.minimize:
            if value < self.best_value - self.min_delta:
                improved = True
        else:
            if value > self.best_value + self.min_delta:
                improved = True

        if improved:
            self.best_value = value
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience

    def reset(self):
        self.best_value = float('inf') if self.minimize else float('-inf')
        self.wait = 0


# ---------------------------------------------------------------------------
# Objective wrapper for NN training
# ---------------------------------------------------------------------------

class NNObjective:
    """Wraps neural network training as a black-box objective for HP tuning.

    Maps HPConfig -> objective value (validation loss or negative accuracy).
    """

    def __init__(self, build_model_fn, X_train, Y_train, X_val=None, Y_val=None,
                 loss_fn=None, metric='val_loss', max_epochs=50, early_stop_patience=5,
                 seed=42):
        """
        Args:
            build_model_fn: callable(config) -> Sequential model
            X_train, Y_train: training data (Tensor or numpy)
            X_val, Y_val: validation data
            loss_fn: loss function (default MSELoss)
            metric: 'val_loss' or 'train_loss'
            max_epochs: maximum training epochs
            early_stop_patience: epochs without improvement before stopping
        """
        self.build_model_fn = build_model_fn
        self.X_train = self._to_tensor(X_train)
        self.Y_train = self._to_tensor(Y_train)
        self.X_val = self._to_tensor(X_val) if X_val is not None else None
        self.Y_val = self._to_tensor(Y_val) if Y_val is not None else None
        self.loss_fn = loss_fn or MSELoss()
        self.metric = metric
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.seed = seed
        self._eval_count = 0

    @staticmethod
    def _to_tensor(data):
        if data is None:
            return None
        if isinstance(data, Tensor):
            return data
        if isinstance(data, np.ndarray):
            return Tensor(data.tolist())
        return Tensor(data)

    def __call__(self, config, budget=1.0):
        """Evaluate config. budget in [0,1] scales max_epochs."""
        self._eval_count += 1
        epochs = max(1, int(self.max_epochs * budget))

        try:
            model = self.build_model_fn(config)
        except Exception:
            return float('inf'), {}

        # Create optimizer from config
        lr = config.get('lr', 0.001)
        optimizer_name = config.get('optimizer', 'adam')
        weight_decay = config.get('weight_decay', 0.0)

        if optimizer_name == 'sgd':
            momentum = config.get('momentum', 0.9)
            optimizer = SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSProp(lr=lr, weight_decay=weight_decay)
        else:
            optimizer = Adam(lr=lr, weight_decay=weight_decay)

        batch_size = config.get('batch_size', 32)
        if isinstance(batch_size, float):
            batch_size = int(batch_size)

        validation_data = None
        if self.X_val is not None and self.Y_val is not None:
            validation_data = (self.X_val, self.Y_val)

        try:
            history = fit(
                model, self.X_train, self.Y_train,
                self.loss_fn, optimizer,
                epochs=epochs, batch_size=batch_size,
                shuffle=True, validation_data=validation_data,
                verbose=False
            )
        except Exception:
            return float('inf'), {}

        metrics = {}
        if 'loss' in history and history['loss']:
            metrics['train_loss'] = history['loss'][-1]
        if 'val_loss' in history and history['val_loss']:
            metrics['val_loss'] = history['val_loss'][-1]
        metrics['epochs_trained'] = epochs

        if self.metric == 'val_loss' and 'val_loss' in metrics:
            objective = metrics['val_loss']
        elif 'train_loss' in metrics:
            objective = metrics['train_loss']
        else:
            objective = float('inf')

        return objective, metrics

    @property
    def eval_count(self):
        return self._eval_count


# ---------------------------------------------------------------------------
# Grid Search Tuner
# ---------------------------------------------------------------------------

class GridSearchTuner:
    """Exhaustive grid search over hyperparameter space."""

    def __init__(self, space, objective_fn, n_points=5, minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.n_points = n_points
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0

    def search(self, max_trials=None, verbose=False):
        configs = self.space.grid(self.n_points)
        if max_trials is not None:
            configs = configs[:max_trials]

        for config in configs:
            self._trial_id += 1
            trial = Trial(trial_id=self._trial_id, config=config, status='running')

            t0 = time.time()
            result = self.objective_fn(config)
            if isinstance(result, tuple):
                objective, metrics = result
            else:
                objective, metrics = result, {}
            trial.duration = time.time() - t0

            trial.objective = objective
            trial.metrics = metrics
            trial.status = 'completed'
            self.history.record(trial)

            if verbose:
                print(f"  Trial {trial.trial_id}: {config.summary()} -> {objective:.6f}")

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Random Search Tuner
# ---------------------------------------------------------------------------

class RandomSearchTuner:
    """Random search with optional Latin Hypercube sampling."""

    def __init__(self, space, objective_fn, minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0

    def search(self, n_trials=20, verbose=False):
        for i in range(n_trials):
            config = self.space.sample(self.rng)
            self._trial_id += 1
            trial = Trial(trial_id=self._trial_id, config=config, status='running')

            t0 = time.time()
            result = self.objective_fn(config)
            if isinstance(result, tuple):
                objective, metrics = result
            else:
                objective, metrics = result, {}
            trial.duration = time.time() - t0

            trial.objective = objective
            trial.metrics = metrics
            trial.status = 'completed'
            self.history.record(trial)

            if verbose:
                print(f"  Trial {trial.trial_id}: {config.summary()} -> {objective:.6f}")

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Bayesian Tuner (composes C167)
# ---------------------------------------------------------------------------

class BayesianTuner:
    """Bayesian optimization-based hyperparameter tuning using C167 GP-BO."""

    def __init__(self, space, objective_fn, acquisition=None, n_initial=5,
                 minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0
        self.n_initial = n_initial

        bounds = [[0.0, 1.0]] * space.dim
        acq = acquisition or ExpectedImprovement(xi=0.01)
        self.optimizer = BayesianOptimizer(
            bounds=bounds,
            acquisition=acq,
            noise_variance=1e-3,
            n_initial=0,  # we handle initial sampling
            seed=seed
        )

    def search(self, n_trials=20, verbose=False):
        for i in range(n_trials):
            self._trial_id += 1

            if i < self.n_initial:
                # Random exploration phase
                config = self.space.sample(self.rng)
            else:
                # BO-guided phase
                try:
                    vec_next, _ = self.optimizer.suggest()
                    config = self.space.decode(vec_next)
                except Exception:
                    config = self.space.sample(self.rng)

            trial = Trial(trial_id=self._trial_id, config=config, status='running')

            t0 = time.time()
            result = self.objective_fn(config)
            if isinstance(result, tuple):
                objective, metrics = result
            else:
                objective, metrics = result, {}
            trial.duration = time.time() - t0

            trial.objective = objective
            trial.metrics = metrics
            trial.status = 'completed'
            self.history.record(trial)

            # Observe in BO (negate if minimizing since BO maximizes)
            vec = self.space.encode(config)
            y_val = -objective if self.minimize else objective
            self.optimizer.observe(vec, y_val)

            if verbose:
                best_so_far = self.history.best_objective
                print(f"  Trial {trial.trial_id}: {config.summary()} -> {objective:.6f} (best: {best_so_far:.6f})")

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Successive Halving Tuner
# ---------------------------------------------------------------------------

class SuccessiveHalvingTuner:
    """Successive Halving: budget-aware early stopping.

    Starts many configs at low budget, progressively doubles budget
    for the top fraction, pruning the rest.
    """

    def __init__(self, space, objective_fn, n_configs=27, eta=3,
                 min_budget=1/27, max_budget=1.0, minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.n_configs = n_configs
        self.eta = eta  # reduction factor
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0

    def search(self, verbose=False):
        # Generate initial configs
        configs = [self.space.sample(self.rng) for _ in range(self.n_configs)]

        # Calculate number of rounds
        s_max = max(1, int(math.floor(math.log(self.n_configs) / math.log(self.eta))))
        budget = self.min_budget

        active_configs = list(configs)
        results = {}  # config_idx -> best objective

        for rung in range(s_max + 1):
            if not active_configs:
                break

            budget = min(self.max_budget, self.min_budget * (self.eta ** rung))

            if verbose:
                print(f"  Rung {rung}: {len(active_configs)} configs at budget {budget:.3f}")

            # Evaluate all active configs at current budget
            rung_results = []
            for config in active_configs:
                self._trial_id += 1
                trial = Trial(trial_id=self._trial_id, config=config,
                              budget=budget, status='running')

                t0 = time.time()
                result = self.objective_fn(config, budget=budget)
                if isinstance(result, tuple):
                    objective, metrics = result
                else:
                    objective, metrics = result, {}
                trial.duration = time.time() - t0

                trial.objective = objective
                trial.metrics = metrics
                trial.status = 'completed'
                self.history.record(trial)
                rung_results.append((config, objective))

            # Sort and keep top 1/eta
            if self.minimize:
                rung_results.sort(key=lambda x: x[1])
            else:
                rung_results.sort(key=lambda x: x[1], reverse=True)

            n_keep = max(1, int(math.ceil(len(rung_results) / self.eta)))
            active_configs = [r[0] for r in rung_results[:n_keep]]

            # Mark pruned trials
            for config, _ in rung_results[n_keep:]:
                pruned_trial = Trial(
                    trial_id=self._trial_id + 1,
                    config=config, budget=budget, status='pruned'
                )
                self.history.record(pruned_trial)

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Hyperband Tuner
# ---------------------------------------------------------------------------

class HyperbandTuner:
    """Hyperband: multi-bracket successive halving.

    Runs multiple brackets of successive halving with different
    exploration/exploitation tradeoffs.
    """

    def __init__(self, space, objective_fn, max_budget=1.0, eta=3,
                 minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.max_budget = max_budget
        self.eta = eta
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0

    def search(self, verbose=False):
        s_max = max(1, int(math.floor(math.log(self.max_budget / (1.0 / 27.0)) / math.log(self.eta))))
        B = (s_max + 1) * self.max_budget

        for s in range(s_max, -1, -1):
            n = int(math.ceil((B / self.max_budget) * (self.eta ** s) / (s + 1)))
            r = self.max_budget * self.eta ** (-s)

            if verbose:
                print(f"  Bracket s={s}: n={n} configs, initial budget={r:.4f}")

            # Sample n configs
            configs = [self.space.sample(self.rng) for _ in range(n)]

            for i in range(s + 1):
                n_i = max(1, int(math.floor(n * self.eta ** (-i))))
                r_i = min(self.max_budget, r * self.eta ** i)

                # Evaluate configs at budget r_i
                results = []
                for config in configs[:n_i]:
                    self._trial_id += 1
                    trial = Trial(trial_id=self._trial_id, config=config,
                                  budget=r_i, status='running')

                    t0 = time.time()
                    result = self.objective_fn(config, budget=r_i)
                    if isinstance(result, tuple):
                        objective, metrics = result
                    else:
                        objective, metrics = result, {}
                    trial.duration = time.time() - t0

                    trial.objective = objective
                    trial.metrics = metrics
                    trial.status = 'completed'
                    self.history.record(trial)
                    results.append((config, objective))

                # Sort and keep top 1/eta
                if self.minimize:
                    results.sort(key=lambda x: x[1])
                else:
                    results.sort(key=lambda x: x[1], reverse=True)

                n_keep = max(1, int(math.floor(len(results) / self.eta)))
                configs = [r[0] for r in results[:n_keep]]

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Population-Based Training
# ---------------------------------------------------------------------------

class PopulationBasedTraining:
    """PBT: exploit + explore during training.

    Maintains a population of workers. Periodically:
    - Exploit: copy weights from better-performing members
    - Explore: perturb hyperparameters of copied configs
    """

    def __init__(self, space, objective_fn, population_size=10,
                 n_generations=5, exploit_fraction=0.2, perturb_factor=0.2,
                 minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.population_size = population_size
        self.n_generations = n_generations
        self.exploit_fraction = exploit_fraction
        self.perturb_factor = perturb_factor
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0

    def _perturb(self, config):
        """Perturb a configuration by exploring nearby values."""
        new_values = dict(config.values)
        for name in new_values:
            if name not in self.space.params:
                continue
            pdef = self.space.params[name]
            if pdef.param_type in (ParamType.CONTINUOUS, ParamType.LOG_CONTINUOUS):
                # Multiply by random factor
                factor = self.rng.choice([1.0 - self.perturb_factor, 1.0 + self.perturb_factor])
                new_val = new_values[name] * factor
                new_val = max(pdef.low, min(pdef.high, new_val))
                new_values[name] = new_val
            elif pdef.param_type == ParamType.INTEGER:
                delta = max(1, int((pdef.high - pdef.low) * self.perturb_factor))
                new_val = new_values[name] + self.rng.integers(-delta, delta + 1)
                new_val = max(int(pdef.low), min(int(pdef.high), int(new_val)))
                new_values[name] = new_val
            elif pdef.param_type == ParamType.CATEGORICAL:
                if self.rng.random() < self.perturb_factor:
                    new_values[name] = pdef.sample(self.rng)
            elif pdef.param_type == ParamType.DISCRETE:
                if self.rng.random() < 0.5:
                    new_val = new_values[name] + pdef.step * self.rng.choice([-1, 1])
                    new_val = max(pdef.low, min(pdef.high, new_val))
                    new_values[name] = new_val
        return HPConfig(new_values)

    def search(self, verbose=False):
        # Initialize population
        population = []
        for _ in range(self.population_size):
            config = self.space.sample(self.rng)
            population.append({'config': config, 'objective': float('inf'), 'metrics': {}})

        budget_per_gen = 1.0 / max(1, self.n_generations)

        for gen in range(self.n_generations):
            budget = min(1.0, budget_per_gen * (gen + 1))

            if verbose:
                print(f"  Generation {gen}: budget={budget:.3f}")

            # Evaluate all members
            for member in population:
                self._trial_id += 1
                config = member['config']
                trial = Trial(trial_id=self._trial_id, config=config,
                              budget=budget, status='running')

                t0 = time.time()
                result = self.objective_fn(config, budget=budget)
                if isinstance(result, tuple):
                    objective, metrics = result
                else:
                    objective, metrics = result, {}
                trial.duration = time.time() - t0

                trial.objective = objective
                trial.metrics = metrics
                trial.status = 'completed'
                self.history.record(trial)

                member['objective'] = objective
                member['metrics'] = metrics

            # Sort population
            if self.minimize:
                population.sort(key=lambda m: m['objective'])
            else:
                population.sort(key=lambda m: m['objective'], reverse=True)

            if verbose:
                best = population[0]
                print(f"    Best: {best['config'].summary()} -> {best['objective']:.6f}")

            # Exploit + Explore (skip last generation)
            if gen < self.n_generations - 1:
                n_top = max(1, int(self.population_size * self.exploit_fraction))
                n_bottom = max(1, int(self.population_size * self.exploit_fraction))

                for i in range(self.population_size - n_top, self.population_size):
                    # Bottom performers copy from top
                    top_idx = self.rng.integers(0, n_top)
                    population[i]['config'] = self._perturb(population[top_idx]['config'])

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Median Pruner
# ---------------------------------------------------------------------------

class MedianPruner:
    """Prune trials that are worse than the median at the same step.

    Used with multi-fidelity tuning to stop unpromising trials early.
    """

    def __init__(self, n_startup_trials=5, n_warmup_steps=3, minimize=True):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.minimize = minimize
        self._step_values = {}  # step -> list of values from completed trials

    def report(self, trial_id, step, value):
        """Report intermediate value. Returns True if should prune."""
        if step not in self._step_values:
            self._step_values[step] = []

        # Don't prune during warmup
        if step < self.n_warmup_steps:
            return False

        # Don't prune until enough trials reported at this step
        if len(self._step_values.get(step, [])) < self.n_startup_trials:
            self._step_values.setdefault(step, []).append(value)
            return False

        median = float(np.median(self._step_values[step]))
        self._step_values[step].append(value)

        if self.minimize:
            return value > median
        return value < median


# ---------------------------------------------------------------------------
# HP Importance Analysis
# ---------------------------------------------------------------------------

class HPImportanceAnalyzer:
    """Analyze which hyperparameters matter most based on trial results."""

    def __init__(self, space, history):
        self.space = space
        self.history = history

    def fanova_importance(self):
        """Functional ANOVA-inspired importance estimation.

        Measures variance in objective explained by each parameter.
        """
        completed = self.history.completed_trials()
        if len(completed) < 3:
            return {}

        objectives = np.array([t.objective for t in completed])
        total_var = np.var(objectives)
        if total_var < 1e-10:
            return {name: 0.0 for name in self.space._order}

        importances = {}
        for name in self.space._order:
            pdef = self.space.params[name]
            if pdef.condition is not None:
                continue

            # Get values for this param
            values = []
            for t in completed:
                val = t.config.get(name)
                if val is not None:
                    if pdef.param_type == ParamType.CATEGORICAL:
                        values.append(pdef.encode(val))
                    else:
                        values.append(float(val))
                else:
                    values.append(0.0)

            values = np.array(values)
            if np.std(values) < 1e-10:
                importances[name] = 0.0
                continue

            # Correlation-based importance
            corr = np.corrcoef(values, objectives)[0, 1]
            if np.isnan(corr):
                importances[name] = 0.0
            else:
                importances[name] = abs(corr)

        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def marginal_effects(self, n_bins=5):
        """Compute marginal effect of each parameter on objective."""
        completed = self.history.completed_trials()
        if len(completed) < 3:
            return {}

        effects = {}
        for name in self.space._order:
            pdef = self.space.params[name]
            if pdef.condition is not None:
                continue

            # Group trials by parameter value bins
            val_obj_pairs = []
            for t in completed:
                val = t.config.get(name)
                if val is not None:
                    if pdef.param_type == ParamType.CATEGORICAL:
                        val_obj_pairs.append((str(val), t.objective))
                    else:
                        val_obj_pairs.append((float(val), t.objective))

            if not val_obj_pairs:
                effects[name] = []
                continue

            if pdef.param_type == ParamType.CATEGORICAL:
                # Group by category
                groups = {}
                for v, obj in val_obj_pairs:
                    groups.setdefault(v, []).append(obj)
                effects[name] = [(k, float(np.mean(v))) for k, v in sorted(groups.items())]
            else:
                # Bin continuous values
                values = [v for v, _ in val_obj_pairs]
                objectives = [o for _, o in val_obj_pairs]
                min_v, max_v = min(values), max(values)
                if max_v - min_v < 1e-10:
                    effects[name] = [(min_v, float(np.mean(objectives)))]
                    continue
                bin_edges = np.linspace(min_v, max_v, n_bins + 1)
                bins = []
                for b in range(n_bins):
                    low, high = bin_edges[b], bin_edges[b + 1]
                    bin_objs = [o for v, o in zip(values, objectives)
                                if low <= v <= high + 1e-10]
                    if bin_objs:
                        center = (low + high) / 2
                        bins.append((center, float(np.mean(bin_objs))))
                effects[name] = bins

        return effects


# ---------------------------------------------------------------------------
# Warm-starting from previous trials
# ---------------------------------------------------------------------------

class WarmStartMixin:
    """Mixin to support warm-starting from previous tuning runs."""

    def warm_start(self, previous_history):
        """Import trials from a previous tuning run."""
        for trial in previous_history.completed_trials():
            self.history.record(trial)
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                vec = self.space.encode(trial.config)
                y_val = -trial.objective if self.minimize else trial.objective
                self.optimizer.observe(vec, y_val)


class WarmBayesianTuner(BayesianTuner, WarmStartMixin):
    """Bayesian tuner with warm-start support."""
    pass


# ---------------------------------------------------------------------------
# Multi-fidelity Bayesian Tuner
# ---------------------------------------------------------------------------

class MultiFidelityBayesianTuner:
    """Combines Bayesian optimization with multi-fidelity evaluation.

    Uses low-budget evaluations to quickly prune bad configs,
    then full-budget evaluation for promising ones.
    """

    def __init__(self, space, objective_fn, n_initial=5, low_budget=0.2,
                 promotion_fraction=0.5, minimize=True, seed=42):
        self.space = space
        self.objective_fn = objective_fn
        self.n_initial = n_initial
        self.low_budget = low_budget
        self.promotion_fraction = promotion_fraction
        self.minimize = minimize
        self.history = HPTuningHistory(minimize=minimize)
        self.rng = np.random.default_rng(seed)
        self._trial_id = 0

        bounds = [[0.0, 1.0]] * space.dim
        self.optimizer = BayesianOptimizer(
            bounds=bounds,
            acquisition=ExpectedImprovement(xi=0.01),
            noise_variance=1e-3,
            n_initial=0,
            seed=seed
        )

    def search(self, n_trials=20, verbose=False):
        # Phase 1: Low-budget screening
        n_screen = max(n_trials, int(n_trials / self.promotion_fraction))
        low_results = []

        for i in range(n_screen):
            self._trial_id += 1

            if i < self.n_initial:
                config = self.space.sample(self.rng)
            else:
                try:
                    vec_next, _ = self.optimizer.suggest()
                    config = self.space.decode(vec_next)
                except Exception:
                    config = self.space.sample(self.rng)

            trial = Trial(trial_id=self._trial_id, config=config,
                          budget=self.low_budget, status='running')

            t0 = time.time()
            result = self.objective_fn(config, budget=self.low_budget)
            if isinstance(result, tuple):
                objective, metrics = result
            else:
                objective, metrics = result, {}
            trial.duration = time.time() - t0

            trial.objective = objective
            trial.metrics = metrics
            trial.status = 'completed'
            self.history.record(trial)
            low_results.append((config, objective))

            vec = self.space.encode(config)
            y_val = -objective if self.minimize else objective
            self.optimizer.observe(vec, y_val)

        # Phase 2: Full-budget evaluation of top configs
        if self.minimize:
            low_results.sort(key=lambda x: x[1])
        else:
            low_results.sort(key=lambda x: x[1], reverse=True)

        n_promote = max(1, int(len(low_results) * self.promotion_fraction))
        promoted = low_results[:n_promote]

        if verbose:
            print(f"  Screened {len(low_results)} at budget {self.low_budget}, promoting {n_promote}")

        for config, _ in promoted:
            self._trial_id += 1
            trial = Trial(trial_id=self._trial_id, config=config,
                          budget=1.0, status='running')

            t0 = time.time()
            result = self.objective_fn(config, budget=1.0)
            if isinstance(result, tuple):
                objective, metrics = result
            else:
                objective, metrics = result, {}
            trial.duration = time.time() - t0

            trial.objective = objective
            trial.metrics = metrics
            trial.status = 'completed'
            self.history.record(trial)

        return self.history.best_config, self.history.best_objective


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tuner(space, objective_fn, method='bayesian', minimize=True, seed=42, **kwargs):
    """Factory function to create a tuner.

    Args:
        space: HyperparameterSpace
        objective_fn: callable(config, budget=1.0) -> (objective, metrics) or objective
        method: 'grid', 'random', 'bayesian', 'successive_halving', 'hyperband', 'pbt', 'multi_fidelity'
        minimize: whether to minimize objective
        seed: random seed
        **kwargs: method-specific arguments

    Returns:
        Tuner instance
    """
    if method == 'grid':
        return GridSearchTuner(space, objective_fn, minimize=minimize, seed=seed,
                               n_points=kwargs.get('n_points', 5))
    elif method == 'random':
        return RandomSearchTuner(space, objective_fn, minimize=minimize, seed=seed)
    elif method == 'bayesian':
        return BayesianTuner(space, objective_fn, minimize=minimize, seed=seed,
                             acquisition=kwargs.get('acquisition'),
                             n_initial=kwargs.get('n_initial', 5))
    elif method == 'successive_halving':
        return SuccessiveHalvingTuner(
            space, objective_fn, minimize=minimize, seed=seed,
            n_configs=kwargs.get('n_configs', 27),
            eta=kwargs.get('eta', 3),
            min_budget=kwargs.get('min_budget', 1/27),
            max_budget=kwargs.get('max_budget', 1.0)
        )
    elif method == 'hyperband':
        return HyperbandTuner(
            space, objective_fn, minimize=minimize, seed=seed,
            max_budget=kwargs.get('max_budget', 1.0),
            eta=kwargs.get('eta', 3)
        )
    elif method == 'pbt':
        return PopulationBasedTraining(
            space, objective_fn, minimize=minimize, seed=seed,
            population_size=kwargs.get('population_size', 10),
            n_generations=kwargs.get('n_generations', 5),
            exploit_fraction=kwargs.get('exploit_fraction', 0.2),
            perturb_factor=kwargs.get('perturb_factor', 0.2)
        )
    elif method == 'multi_fidelity':
        return MultiFidelityBayesianTuner(
            space, objective_fn, minimize=minimize, seed=seed,
            n_initial=kwargs.get('n_initial', 5),
            low_budget=kwargs.get('low_budget', 0.2),
            promotion_fraction=kwargs.get('promotion_fraction', 0.5)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
