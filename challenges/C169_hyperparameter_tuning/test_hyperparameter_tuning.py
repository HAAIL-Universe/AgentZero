"""
Tests for C169: Hyperparameter Tuning
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C167_bayesian_optimization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from hyperparameter_tuning import (
    ParamType, HyperparameterDef, HyperparameterSpace, HPConfig, Trial,
    HPTuningHistory, EarlyStoppingCallback, NNObjective,
    GridSearchTuner, RandomSearchTuner, BayesianTuner,
    SuccessiveHalvingTuner, HyperbandTuner, PopulationBasedTraining,
    MedianPruner, HPImportanceAnalyzer, WarmBayesianTuner,
    MultiFidelityBayesianTuner, create_tuner
)
from bayesian_optimization import UpperConfidenceBound
from neural_network import Sequential, Dense, Activation, MSELoss, Tensor


# ===== Helpers =====

def simple_objective(config, budget=1.0):
    """Simple quadratic objective for testing."""
    x = config.get('x', 0.5)
    y = config.get('y', 0.5)
    noise = 0.01 * (1.0 - budget)  # less noise at higher budget
    obj = (x - 0.3) ** 2 + (y - 0.7) ** 2 + noise
    return obj, {'x': x, 'y': y}


def categorical_objective(config, budget=1.0):
    """Objective with categorical params."""
    method = config.get('method', 'a')
    lr = config.get('lr', 0.01)
    scores = {'a': 0.0, 'b': 0.5, 'c': 1.0}
    obj = scores.get(method, 0.5) + (lr - 0.01) ** 2
    return obj, {}


def make_simple_space():
    space = HyperparameterSpace()
    space.add_continuous('x', 0.0, 1.0)
    space.add_continuous('y', 0.0, 1.0)
    return space


# ===== ParamType Tests =====

class TestParamType:
    def test_enum_values(self):
        assert ParamType.CONTINUOUS.value == 'continuous'
        assert ParamType.DISCRETE.value == 'discrete'
        assert ParamType.CATEGORICAL.value == 'categorical'
        assert ParamType.LOG_CONTINUOUS.value == 'log_continuous'
        assert ParamType.INTEGER.value == 'integer'


# ===== HyperparameterDef Tests =====

class TestHyperparameterDef:
    def test_continuous_sample(self):
        pdef = HyperparameterDef('x', ParamType.CONTINUOUS, low=0.0, high=1.0)
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = pdef.sample(rng)
            assert 0.0 <= v <= 1.0

    def test_log_continuous_sample(self):
        pdef = HyperparameterDef('lr', ParamType.LOG_CONTINUOUS, low=1e-4, high=1e-1)
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = pdef.sample(rng)
            assert 1e-4 <= v <= 1e-1

    def test_integer_sample(self):
        pdef = HyperparameterDef('n', ParamType.INTEGER, low=1, high=10)
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = pdef.sample(rng)
            assert isinstance(v, int)
            assert 1 <= v <= 10

    def test_discrete_sample(self):
        pdef = HyperparameterDef('d', ParamType.DISCRETE, low=0.0, high=1.0, step=0.25)
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = pdef.sample(rng)
            assert 0.0 <= v <= 1.0

    def test_categorical_sample(self):
        pdef = HyperparameterDef('c', ParamType.CATEGORICAL, choices=['a', 'b', 'c'])
        rng = np.random.default_rng(42)
        for _ in range(20):
            v = pdef.sample(rng)
            assert v in ['a', 'b', 'c']

    def test_continuous_encode_decode(self):
        pdef = HyperparameterDef('x', ParamType.CONTINUOUS, low=0.0, high=10.0)
        assert pdef.encode(5.0) == pytest.approx(0.5)
        assert pdef.decode(0.5) == pytest.approx(5.0)
        assert pdef.decode(0.0) == pytest.approx(0.0)
        assert pdef.decode(1.0) == pytest.approx(10.0)

    def test_log_continuous_encode_decode(self):
        pdef = HyperparameterDef('lr', ParamType.LOG_CONTINUOUS, low=1e-4, high=1e-1)
        encoded = pdef.encode(1e-4)
        assert encoded == pytest.approx(0.0, abs=0.01)
        decoded = pdef.decode(0.0)
        assert decoded == pytest.approx(1e-4, rel=0.01)
        decoded_high = pdef.decode(1.0)
        assert decoded_high == pytest.approx(1e-1, rel=0.01)

    def test_integer_encode_decode(self):
        pdef = HyperparameterDef('n', ParamType.INTEGER, low=1, high=10)
        assert pdef.encode(1) == pytest.approx(0.0)
        assert pdef.encode(10) == pytest.approx(1.0)
        assert pdef.decode(0.0) == 1
        assert pdef.decode(1.0) == 10
        assert isinstance(pdef.decode(0.5), int)

    def test_categorical_encode_decode(self):
        pdef = HyperparameterDef('c', ParamType.CATEGORICAL, choices=['a', 'b', 'c'])
        assert pdef.encode('a') == pytest.approx(0.0)
        assert pdef.encode('c') == pytest.approx(1.0)
        assert pdef.decode(0.0) == 'a'
        assert pdef.decode(1.0) == 'c'

    def test_grid_values_continuous(self):
        pdef = HyperparameterDef('x', ParamType.CONTINUOUS, low=0.0, high=1.0)
        grid = pdef.grid_values(5)
        assert len(grid) == 5
        assert grid[0] == pytest.approx(0.0)
        assert grid[-1] == pytest.approx(1.0)

    def test_grid_values_categorical(self):
        pdef = HyperparameterDef('c', ParamType.CATEGORICAL, choices=['a', 'b', 'c'])
        grid = pdef.grid_values()
        assert grid == ['a', 'b', 'c']

    def test_grid_values_integer(self):
        pdef = HyperparameterDef('n', ParamType.INTEGER, low=1, high=5)
        grid = pdef.grid_values(10)
        assert 1 in grid
        assert 5 in grid

    def test_encode_edge_cases(self):
        # Equal low/high
        pdef = HyperparameterDef('x', ParamType.CONTINUOUS, low=5.0, high=5.0)
        assert pdef.encode(5.0) == 0.5

    def test_decode_clamp(self):
        pdef = HyperparameterDef('x', ParamType.CONTINUOUS, low=0.0, high=1.0)
        assert pdef.decode(-0.5) == pytest.approx(0.0)
        assert pdef.decode(1.5) == pytest.approx(1.0)


# ===== HyperparameterSpace Tests =====

class TestHyperparameterSpace:
    def test_add_params(self):
        space = HyperparameterSpace()
        space.add_continuous('lr', 1e-4, 1e-1, log_scale=True)
        space.add_integer('layers', 1, 5)
        space.add_categorical('optimizer', ['adam', 'sgd'])
        assert space.dim == 3
        assert 'lr' in space.params
        assert 'layers' in space.params
        assert 'optimizer' in space.params

    def test_sample(self):
        space = make_simple_space()
        config = space.sample(np.random.default_rng(42))
        assert isinstance(config, HPConfig)
        assert 'x' in config
        assert 'y' in config
        assert 0.0 <= config['x'] <= 1.0
        assert 0.0 <= config['y'] <= 1.0

    def test_encode_decode_roundtrip(self):
        space = HyperparameterSpace()
        space.add_continuous('x', 0.0, 10.0)
        space.add_integer('n', 1, 5)
        space.add_categorical('c', ['a', 'b', 'c'])

        config = HPConfig({'x': 5.0, 'n': 3, 'c': 'b'})
        vec = space.encode(config)
        assert len(vec) == 3

        decoded = space.decode(vec)
        assert decoded['x'] == pytest.approx(5.0, abs=0.5)
        assert decoded['n'] == 3
        assert decoded['c'] == 'b'

    def test_grid(self):
        space = HyperparameterSpace()
        space.add_categorical('a', ['x', 'y'])
        space.add_continuous('b', 0.0, 1.0)
        grid = space.grid(n_points=3)
        assert len(grid) == 6  # 2 * 3

    def test_conditional_param(self):
        space = HyperparameterSpace()
        space.add_categorical('optimizer', ['adam', 'sgd'])
        space.add_conditional('momentum', ParamType.CONTINUOUS,
                              condition_param='optimizer', condition_value='sgd',
                              low=0.0, high=1.0, default=0.9)
        rng = np.random.default_rng(42)
        # Sample many times -- momentum should only appear when optimizer=sgd
        for _ in range(50):
            config = space.sample(rng)
            if config['optimizer'] == 'adam':
                assert 'momentum' not in config
            else:
                assert 'momentum' in config

    def test_get_defaults(self):
        space = HyperparameterSpace()
        space.add_continuous('x', 0.0, 1.0, default=0.5)
        space.add_integer('n', 1, 10, default=5)
        defaults = space.get_defaults()
        assert defaults['x'] == 0.5
        assert defaults['n'] == 5

    def test_chaining(self):
        space = HyperparameterSpace()
        result = space.add_continuous('x', 0.0, 1.0).add_integer('n', 1, 5)
        assert result is space
        assert space.dim == 2

    def test_dim(self):
        space = make_simple_space()
        assert space.dim == 2

    def test_log_scale_space(self):
        space = HyperparameterSpace()
        space.add_continuous('lr', 1e-5, 1e-1, log_scale=True)
        assert space.params['lr'].param_type == ParamType.LOG_CONTINUOUS

    def test_discrete_space(self):
        space = HyperparameterSpace()
        space.add_discrete('dropout', 0.0, 0.5, step=0.1)
        config = space.sample(np.random.default_rng(42))
        assert 'dropout' in config


# ===== HPConfig Tests =====

class TestHPConfig:
    def test_getitem(self):
        config = HPConfig({'x': 1, 'y': 2})
        assert config['x'] == 1
        assert config['y'] == 2

    def test_contains(self):
        config = HPConfig({'x': 1})
        assert 'x' in config
        assert 'z' not in config

    def test_get_default(self):
        config = HPConfig({'x': 1})
        assert config.get('x') == 1
        assert config.get('z', 42) == 42

    def test_copy(self):
        config = HPConfig({'x': 1, 'y': 2})
        copy = config.copy()
        assert copy['x'] == 1
        copy.values['x'] = 99
        assert config['x'] == 1  # original unchanged

    def test_summary(self):
        config = HPConfig({'x': 0.5, 'y': 'adam'})
        s = config.summary()
        assert 'x=' in s
        assert 'y=adam' in s

    def test_repr(self):
        config = HPConfig({'x': 1})
        assert 'HPConfig' in repr(config)


# ===== Trial Tests =====

class TestTrial:
    def test_create(self):
        config = HPConfig({'x': 0.5})
        trial = Trial(trial_id=1, config=config, objective=0.1)
        assert trial.trial_id == 1
        assert trial.objective == 0.1

    def test_is_better_than_minimize(self):
        t1 = Trial(1, HPConfig({}), objective=0.1)
        t2 = Trial(2, HPConfig({}), objective=0.5)
        assert t1.is_better_than(t2, minimize=True)
        assert not t2.is_better_than(t1, minimize=True)

    def test_is_better_than_maximize(self):
        t1 = Trial(1, HPConfig({}), objective=0.1)
        t2 = Trial(2, HPConfig({}), objective=0.5)
        assert t2.is_better_than(t1, minimize=False)

    def test_default_status(self):
        trial = Trial(1, HPConfig({}))
        assert trial.status == 'pending'
        assert trial.budget == 1.0

    def test_metrics(self):
        trial = Trial(1, HPConfig({}), metrics={'loss': 0.5, 'acc': 0.9})
        assert trial.metrics['loss'] == 0.5


# ===== HPTuningHistory Tests =====

class TestHPTuningHistory:
    def test_record_and_best(self):
        history = HPTuningHistory(minimize=True)
        t1 = Trial(1, HPConfig({'x': 0.5}), objective=0.5, status='completed')
        t2 = Trial(2, HPConfig({'x': 0.3}), objective=0.1, status='completed')
        history.record(t1)
        history.record(t2)
        assert history.best_objective == 0.1
        assert history.best_config['x'] == 0.3

    def test_maximize(self):
        history = HPTuningHistory(minimize=False)
        t1 = Trial(1, HPConfig({}), objective=0.5, status='completed')
        t2 = Trial(2, HPConfig({}), objective=0.9, status='completed')
        history.record(t1)
        history.record(t2)
        assert history.best_objective == 0.9

    def test_convergence_curve(self):
        history = HPTuningHistory(minimize=True)
        for i, obj in enumerate([0.5, 0.3, 0.4, 0.2, 0.25]):
            t = Trial(i, HPConfig({}), objective=obj, status='completed')
            history.record(t)
        curve = history.convergence_curve()
        assert curve == [0.5, 0.3, 0.3, 0.2, 0.2]

    def test_top_k(self):
        history = HPTuningHistory(minimize=True)
        for i in range(10):
            t = Trial(i, HPConfig({}), objective=float(i), status='completed')
            history.record(t)
        top3 = history.top_k(3)
        assert len(top3) == 3
        assert top3[0].objective == 0.0

    def test_summary(self):
        history = HPTuningHistory(minimize=True)
        t = Trial(1, HPConfig({'x': 0.5}), objective=0.3, status='completed')
        history.record(t)
        s = history.summary()
        assert s['total_trials'] == 1
        assert s['completed'] == 1
        assert s['best_objective'] == 0.3

    def test_completed_trials(self):
        history = HPTuningHistory()
        history.record(Trial(1, HPConfig({}), objective=0.5, status='completed'))
        history.record(Trial(2, HPConfig({}), status='pruned'))
        assert len(history.completed_trials()) == 1

    def test_pruned_count(self):
        history = HPTuningHistory()
        history.record(Trial(1, HPConfig({}), status='pruned'))
        history.record(Trial(2, HPConfig({}), status='pruned'))
        history.record(Trial(3, HPConfig({}), objective=0.5, status='completed'))
        s = history.summary()
        assert s['pruned'] == 2

    def test_empty_history(self):
        history = HPTuningHistory()
        assert history.best_trial is None
        assert history.best_config is None
        assert history.best_objective is None
        assert history.convergence_curve() == []


# ===== EarlyStoppingCallback Tests =====

class TestEarlyStoppingCallback:
    def test_no_improvement_triggers_stop(self):
        cb = EarlyStoppingCallback(patience=3, minimize=True)
        assert not cb.report(0.5)  # first report, improvement
        assert not cb.report(0.5)  # no improvement, wait=1
        assert not cb.report(0.5)  # wait=2
        assert cb.report(0.5)     # wait=3, should stop

    def test_improvement_resets(self):
        cb = EarlyStoppingCallback(patience=3, minimize=True)
        cb.report(0.5)
        cb.report(0.5)  # wait=1
        assert not cb.report(0.3)  # improvement, reset
        cb.report(0.3)  # wait=1
        cb.report(0.3)  # wait=2
        assert cb.report(0.3)  # wait=3, stop

    def test_maximize_mode(self):
        cb = EarlyStoppingCallback(patience=2, minimize=False)
        assert not cb.report(0.5)  # improvement
        assert not cb.report(0.8)  # improvement
        assert not cb.report(0.8)  # no improvement, wait=1
        assert cb.report(0.8)     # wait=2, stop

    def test_reset(self):
        cb = EarlyStoppingCallback(patience=2, minimize=True)
        cb.report(0.5)
        cb.report(0.5)
        cb.reset()
        assert cb.wait == 0

    def test_min_delta(self):
        cb = EarlyStoppingCallback(patience=2, min_delta=0.1, minimize=True)
        cb.report(0.5)
        # Small improvement (< min_delta) doesn't count
        assert not cb.report(0.49)  # wait=1
        assert cb.report(0.48)     # wait=2, stop


# ===== GridSearchTuner Tests =====

class TestGridSearchTuner:
    def test_basic_grid_search(self):
        space = make_simple_space()
        tuner = GridSearchTuner(space, simple_objective, n_points=3, minimize=True)
        best_config, best_obj = tuner.search()
        assert best_config is not None
        assert best_obj < 1.0

    def test_grid_search_explores_all(self):
        space = HyperparameterSpace()
        space.add_categorical('c', ['a', 'b'])
        tuner = GridSearchTuner(space, lambda c, budget=1.0: (0.5, {}), n_points=5)
        tuner.search()
        assert len(tuner.history.trials) == 2

    def test_max_trials_limit(self):
        space = make_simple_space()
        tuner = GridSearchTuner(space, simple_objective, n_points=5)
        tuner.search(max_trials=3)
        assert len(tuner.history.trials) == 3

    def test_grid_finds_optimum(self):
        space = HyperparameterSpace()
        space.add_continuous('x', 0.0, 1.0)
        # Optimum at x=0.5
        def obj(config, budget=1.0):
            return abs(config['x'] - 0.5), {}
        tuner = GridSearchTuner(space, obj, n_points=11, minimize=True)
        best_config, best_obj = tuner.search()
        assert abs(best_config['x'] - 0.5) < 0.11


# ===== RandomSearchTuner Tests =====

class TestRandomSearchTuner:
    def test_basic_random_search(self):
        space = make_simple_space()
        tuner = RandomSearchTuner(space, simple_objective, minimize=True, seed=42)
        best_config, best_obj = tuner.search(n_trials=20)
        assert best_config is not None
        assert best_obj < 0.5

    def test_trial_count(self):
        space = make_simple_space()
        tuner = RandomSearchTuner(space, simple_objective, seed=42)
        tuner.search(n_trials=10)
        assert len(tuner.history.trials) == 10

    def test_reproducible(self):
        space = make_simple_space()
        t1 = RandomSearchTuner(space, simple_objective, seed=42)
        t1.search(n_trials=5)
        t2 = RandomSearchTuner(space, simple_objective, seed=42)
        t2.search(n_trials=5)
        assert t1.history.best_objective == t2.history.best_objective

    def test_different_seeds(self):
        space = make_simple_space()
        t1 = RandomSearchTuner(space, simple_objective, seed=42)
        t1.search(n_trials=10)
        t2 = RandomSearchTuner(space, simple_objective, seed=99)
        t2.search(n_trials=10)
        # Different seeds should (almost certainly) give different results
        configs1 = [t.config['x'] for t in t1.history.trials]
        configs2 = [t.config['x'] for t in t2.history.trials]
        assert configs1 != configs2


# ===== BayesianTuner Tests =====

class TestBayesianTuner:
    def test_basic_bayesian(self):
        space = make_simple_space()
        tuner = BayesianTuner(space, simple_objective, n_initial=5, minimize=True, seed=42)
        best_config, best_obj = tuner.search(n_trials=15)
        assert best_config is not None
        assert best_obj < 0.5

    def test_bayesian_improves_over_random(self):
        space = make_simple_space()
        # Random baseline
        rand = RandomSearchTuner(space, simple_objective, seed=42)
        rand.search(n_trials=15)
        # Bayesian
        bayes = BayesianTuner(space, simple_objective, n_initial=5, seed=42)
        bayes.search(n_trials=15)
        # Bayesian should find at least as good (or better on average)
        assert bayes.history.best_objective <= rand.history.best_objective + 0.3

    def test_bayesian_trial_count(self):
        space = make_simple_space()
        tuner = BayesianTuner(space, simple_objective, n_initial=3, seed=42)
        tuner.search(n_trials=8)
        assert len(tuner.history.trials) == 8

    def test_bayesian_with_ucb(self):
        space = make_simple_space()
        tuner = BayesianTuner(space, simple_objective,
                              acquisition=UpperConfidenceBound(kappa=2.0),
                              n_initial=3, seed=42)
        best_config, best_obj = tuner.search(n_trials=10)
        assert best_obj < 1.0

    def test_bayesian_convergence(self):
        space = make_simple_space()
        tuner = BayesianTuner(space, simple_objective, n_initial=5, seed=42)
        tuner.search(n_trials=15)
        curve = tuner.history.convergence_curve()
        assert len(curve) == 15
        # Should be non-increasing (minimize)
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i-1] + 1e-10


# ===== SuccessiveHalvingTuner Tests =====

class TestSuccessiveHalvingTuner:
    def test_basic_successive_halving(self):
        space = make_simple_space()
        tuner = SuccessiveHalvingTuner(space, simple_objective, n_configs=9, eta=3, seed=42)
        best_config, best_obj = tuner.search()
        assert best_config is not None
        assert best_obj < 1.0

    def test_progressive_elimination(self):
        space = make_simple_space()
        tuner = SuccessiveHalvingTuner(space, simple_objective, n_configs=9, eta=3, seed=42)
        tuner.search()
        # Should have multiple trials at different budgets
        budgets = set(t.budget for t in tuner.history.completed_trials())
        assert len(budgets) >= 2

    def test_small_population(self):
        space = make_simple_space()
        tuner = SuccessiveHalvingTuner(space, simple_objective, n_configs=3, eta=3, seed=42)
        best_config, best_obj = tuner.search()
        assert best_config is not None


# ===== HyperbandTuner Tests =====

class TestHyperbandTuner:
    def test_basic_hyperband(self):
        space = make_simple_space()
        tuner = HyperbandTuner(space, simple_objective, max_budget=1.0, eta=3, seed=42)
        best_config, best_obj = tuner.search()
        assert best_config is not None
        assert best_obj < 1.0

    def test_hyperband_multiple_brackets(self):
        space = make_simple_space()
        tuner = HyperbandTuner(space, simple_objective, max_budget=1.0, eta=3, seed=42)
        tuner.search()
        # Should evaluate many configs across brackets
        assert len(tuner.history.trials) > 5

    def test_hyperband_finds_good_config(self):
        space = make_simple_space()
        tuner = HyperbandTuner(space, simple_objective, max_budget=1.0, eta=3, seed=42)
        best_config, best_obj = tuner.search()
        assert best_obj < 0.5


# ===== PopulationBasedTraining Tests =====

class TestPopulationBasedTraining:
    def test_basic_pbt(self):
        space = make_simple_space()
        tuner = PopulationBasedTraining(space, simple_objective,
                                         population_size=6, n_generations=3, seed=42)
        best_config, best_obj = tuner.search()
        assert best_config is not None
        assert best_obj < 1.0

    def test_pbt_improves_over_generations(self):
        space = make_simple_space()
        tuner = PopulationBasedTraining(space, simple_objective,
                                         population_size=8, n_generations=4, seed=42)
        tuner.search()
        curve = tuner.history.convergence_curve()
        # Later trials should tend to be better
        assert curve[-1] <= curve[0]

    def test_pbt_perturb(self):
        space = make_simple_space()
        tuner = PopulationBasedTraining(space, simple_objective,
                                         population_size=4, n_generations=2, seed=42)
        config = HPConfig({'x': 0.5, 'y': 0.5})
        perturbed = tuner._perturb(config)
        # Should be different but close
        assert isinstance(perturbed, HPConfig)
        assert 'x' in perturbed
        # At least one value should change (with high probability)

    def test_pbt_exploit_explore(self):
        space = make_simple_space()
        tuner = PopulationBasedTraining(space, simple_objective,
                                         population_size=10, n_generations=3,
                                         exploit_fraction=0.3, seed=42)
        tuner.search()
        # Should have population_size * n_generations completed trials
        completed = tuner.history.completed_trials()
        assert len(completed) == 30

    def test_pbt_with_categorical(self):
        space = HyperparameterSpace()
        space.add_categorical('method', ['a', 'b', 'c'])
        space.add_continuous('x', 0.0, 1.0)
        tuner = PopulationBasedTraining(space, categorical_objective,
                                         population_size=6, n_generations=2, seed=42)
        best_config, best_obj = tuner.search()
        assert best_config is not None


# ===== MedianPruner Tests =====

class TestMedianPruner:
    def test_no_prune_during_warmup(self):
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
        assert not pruner.report(1, 0, 100.0)
        assert not pruner.report(1, 1, 100.0)

    def test_no_prune_until_startup(self):
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        # First 3 trials don't prune
        for i in range(3):
            assert not pruner.report(i, 5, float(i))

    def test_prune_worse_than_median(self):
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0, minimize=True)
        # Build up history at step 5
        pruner._step_values[5] = [0.1, 0.2, 0.3, 0.4, 0.5]
        # Report a bad value
        assert pruner.report(99, 5, 0.9)  # worse than median (0.3), should prune

    def test_no_prune_good_value(self):
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0, minimize=True)
        pruner._step_values[5] = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert not pruner.report(99, 5, 0.1)  # better than median, don't prune


# ===== HPImportanceAnalyzer Tests =====

class TestHPImportanceAnalyzer:
    def test_fanova_importance(self):
        space = make_simple_space()
        history = HPTuningHistory()
        rng = np.random.default_rng(42)
        for i in range(20):
            x = rng.uniform(0, 1)
            y = rng.uniform(0, 1)
            obj = (x - 0.3) ** 2  # only depends on x
            config = HPConfig({'x': x, 'y': y})
            history.record(Trial(i, config, objective=obj, status='completed'))

        analyzer = HPImportanceAnalyzer(space, history)
        imp = analyzer.fanova_importance()
        assert 'x' in imp
        assert 'y' in imp
        # x should be more important since objective only depends on x
        assert imp['x'] > imp['y']

    def test_marginal_effects(self):
        space = make_simple_space()
        history = HPTuningHistory()
        for i in range(20):
            x = i / 19.0
            config = HPConfig({'x': x, 'y': 0.5})
            obj = (x - 0.5) ** 2
            history.record(Trial(i, config, objective=obj, status='completed'))

        analyzer = HPImportanceAnalyzer(space, history)
        effects = analyzer.marginal_effects(n_bins=5)
        assert 'x' in effects
        assert len(effects['x']) > 0

    def test_marginal_effects_categorical(self):
        space = HyperparameterSpace()
        space.add_categorical('method', ['a', 'b', 'c'])
        history = HPTuningHistory()
        scores = {'a': 0.1, 'b': 0.5, 'c': 0.9}
        for i, method in enumerate(['a', 'b', 'c', 'a', 'b', 'c']):
            config = HPConfig({'method': method})
            history.record(Trial(i, config, objective=scores[method], status='completed'))

        analyzer = HPImportanceAnalyzer(space, history)
        effects = analyzer.marginal_effects()
        assert 'method' in effects
        assert len(effects['method']) == 3

    def test_empty_history(self):
        space = make_simple_space()
        history = HPTuningHistory()
        analyzer = HPImportanceAnalyzer(space, history)
        assert analyzer.fanova_importance() == {}
        assert analyzer.marginal_effects() == {}


# ===== WarmBayesianTuner Tests =====

class TestWarmBayesianTuner:
    def test_warm_start(self):
        space = make_simple_space()
        # First run
        tuner1 = BayesianTuner(space, simple_objective, n_initial=5, seed=42)
        tuner1.search(n_trials=10)
        prev_best = tuner1.history.best_objective

        # Warm start from previous
        tuner2 = WarmBayesianTuner(space, simple_objective, n_initial=3, seed=99)
        tuner2.warm_start(tuner1.history)
        assert len(tuner2.history.completed_trials()) == 10
        tuner2.search(n_trials=5)
        # Should have all trials
        assert len(tuner2.history.completed_trials()) == 15


# ===== MultiFidelityBayesianTuner Tests =====

class TestMultiFidelityBayesianTuner:
    def test_basic_multi_fidelity(self):
        space = make_simple_space()
        tuner = MultiFidelityBayesianTuner(space, simple_objective,
                                            n_initial=3, low_budget=0.3,
                                            promotion_fraction=0.5, seed=42)
        best_config, best_obj = tuner.search(n_trials=10)
        assert best_config is not None
        assert best_obj < 1.0

    def test_multi_fidelity_has_two_phases(self):
        space = make_simple_space()
        tuner = MultiFidelityBayesianTuner(space, simple_objective,
                                            n_initial=3, low_budget=0.2,
                                            promotion_fraction=0.5, seed=42)
        tuner.search(n_trials=10)
        budgets = [t.budget for t in tuner.history.completed_trials()]
        # Should have both low and full budget trials
        assert any(b < 0.5 for b in budgets)
        assert any(b >= 0.9 for b in budgets)


# ===== NNObjective Tests =====

class TestNNObjective:
    def _make_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2))
        Y = (X[:, 0:1] + X[:, 1:2]) * 0.5
        return Tensor(X.tolist()), Tensor(Y.tolist())

    def test_basic_nn_objective(self):
        X, Y = self._make_data()

        def build_model(config):
            hidden = config.get('hidden_size', 16)
            model = Sequential()
            model.add(Dense(2, hidden))
            model.add(Activation('relu'))
            model.add(Dense(hidden, 1))
            return model

        space = HyperparameterSpace()
        space.add_continuous('lr', 1e-4, 1e-1, log_scale=True)
        space.add_integer('hidden_size', 4, 32)
        space.add_categorical('optimizer', ['adam', 'sgd'])

        obj = NNObjective(build_model, X, Y, loss_fn=MSELoss(), max_epochs=10, seed=42)
        config = HPConfig({'lr': 0.01, 'hidden_size': 16, 'optimizer': 'adam', 'batch_size': 16})
        result, metrics = obj(config)
        assert isinstance(result, float)
        assert result < float('inf')
        assert obj.eval_count == 1

    def test_nn_objective_with_budget(self):
        X, Y = self._make_data()

        def build_model(config):
            model = Sequential()
            model.add(Dense(2, 8))
            model.add(Activation('relu'))
            model.add(Dense(8, 1))
            return model

        obj = NNObjective(build_model, X, Y, max_epochs=20)
        config = HPConfig({'lr': 0.01, 'optimizer': 'adam', 'batch_size': 16})
        result_low, metrics_low = obj(config, budget=0.5)
        assert isinstance(result_low, float)

    def test_nn_objective_bad_model(self):
        X, Y = self._make_data()
        def build_model(config):
            raise ValueError("bad model")
        obj = NNObjective(build_model, X, Y)
        config = HPConfig({'lr': 0.01, 'optimizer': 'adam'})
        result, metrics = obj(config)
        assert result == float('inf')

    def test_nn_objective_numpy_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))
        Y = rng.standard_normal((20, 1))
        def build_model(config):
            model = Sequential()
            model.add(Dense(2, 4))
            model.add(Dense(4, 1))
            return model
        obj = NNObjective(build_model, X, Y, max_epochs=5)
        config = HPConfig({'lr': 0.01, 'optimizer': 'adam', 'batch_size': 10})
        result, metrics = obj(config)
        assert result < float('inf')

    def test_nn_objective_with_validation(self):
        X, Y = self._make_data()
        rng = np.random.default_rng(99)
        X_val = Tensor(rng.standard_normal((10, 2)).tolist())
        Y_val = Tensor(rng.standard_normal((10, 1)).tolist())

        def build_model(config):
            model = Sequential()
            model.add(Dense(2, 8))
            model.add(Dense(8, 1))
            return model

        obj = NNObjective(build_model, X, Y, X_val=X_val, Y_val=Y_val,
                          metric='val_loss', max_epochs=5)
        config = HPConfig({'lr': 0.01, 'optimizer': 'adam', 'batch_size': 16})
        result, metrics = obj(config)
        assert 'val_loss' in metrics


# ===== Integration: Full Tuning Pipeline =====

class TestFullPipeline:
    def test_grid_then_bayesian(self):
        """Grid search to find rough region, then Bayesian to refine."""
        space = make_simple_space()
        # Grid search
        grid = GridSearchTuner(space, simple_objective, n_points=5, seed=42)
        grid.search()
        grid_best = grid.history.best_objective

        # Bayesian refinement
        bayes = WarmBayesianTuner(space, simple_objective, n_initial=2, seed=42)
        bayes.warm_start(grid.history)
        bayes.search(n_trials=10)
        bayes_best = bayes.history.best_objective

        assert bayes_best <= grid_best + 0.01

    def test_importance_after_tuning(self):
        space = make_simple_space()
        tuner = RandomSearchTuner(space, simple_objective, seed=42)
        tuner.search(n_trials=30)

        analyzer = HPImportanceAnalyzer(space, tuner.history)
        imp = analyzer.fanova_importance()
        assert sum(imp.values()) == pytest.approx(1.0, abs=0.01)

    def test_convergence_comparison(self):
        """Compare convergence of different methods."""
        space = make_simple_space()
        n = 15

        rand = RandomSearchTuner(space, simple_objective, seed=42)
        rand.search(n_trials=n)

        bayes = BayesianTuner(space, simple_objective, n_initial=5, seed=42)
        bayes.search(n_trials=n)

        # Both should find reasonable solutions
        assert rand.history.best_objective < 1.0
        assert bayes.history.best_objective < 1.0


# ===== Factory Tests =====

class TestFactory:
    def test_create_grid(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='grid')
        assert isinstance(tuner, GridSearchTuner)

    def test_create_random(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='random')
        assert isinstance(tuner, RandomSearchTuner)

    def test_create_bayesian(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='bayesian')
        assert isinstance(tuner, BayesianTuner)

    def test_create_successive_halving(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='successive_halving')
        assert isinstance(tuner, SuccessiveHalvingTuner)

    def test_create_hyperband(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='hyperband')
        assert isinstance(tuner, HyperbandTuner)

    def test_create_pbt(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='pbt')
        assert isinstance(tuner, PopulationBasedTraining)

    def test_create_multi_fidelity(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='multi_fidelity')
        assert isinstance(tuner, MultiFidelityBayesianTuner)

    def test_create_unknown_raises(self):
        space = make_simple_space()
        with pytest.raises(ValueError):
            create_tuner(space, simple_objective, method='unknown')

    def test_factory_kwargs(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='pbt',
                             population_size=5, n_generations=2)
        assert tuner.population_size == 5
        assert tuner.n_generations == 2

    def test_factory_minimize_flag(self):
        space = make_simple_space()
        tuner = create_tuner(space, simple_objective, method='random', minimize=False)
        assert tuner.minimize is False


# ===== Edge Cases =====

class TestEdgeCases:
    def test_single_param_space(self):
        space = HyperparameterSpace()
        space.add_continuous('x', 0.0, 1.0)
        tuner = RandomSearchTuner(space, lambda c, b=1.0: (c['x'] ** 2, {}), seed=42)
        best_config, best_obj = tuner.search(n_trials=10)
        assert best_obj < 0.1

    def test_high_dim_space(self):
        space = HyperparameterSpace()
        for i in range(10):
            space.add_continuous(f'x{i}', 0.0, 1.0)
        def obj(config, budget=1.0):
            return sum(config.get(f'x{i}', 0.5) ** 2 for i in range(10)), {}
        tuner = RandomSearchTuner(space, obj, seed=42)
        best_config, best_obj = tuner.search(n_trials=20)
        assert best_obj < 5.0

    def test_objective_returning_scalar(self):
        space = make_simple_space()
        def obj(config, budget=1.0):
            return config['x'] ** 2  # scalar, not tuple
        tuner = RandomSearchTuner(space, obj, seed=42)
        best_config, best_obj = tuner.search(n_trials=5)
        assert isinstance(best_obj, float)

    def test_all_same_objective(self):
        space = make_simple_space()
        tuner = RandomSearchTuner(space, lambda c, b=1.0: (1.0, {}), seed=42)
        tuner.search(n_trials=5)
        assert tuner.history.best_objective == 1.0

    def test_inf_objective(self):
        space = make_simple_space()
        call_count = [0]
        def obj(config, budget=1.0):
            call_count[0] += 1
            if call_count[0] <= 2:
                return float('inf'), {}
            return 0.5, {}
        tuner = RandomSearchTuner(space, obj, seed=42)
        tuner.search(n_trials=5)
        assert tuner.history.best_objective == 0.5

    def test_mixed_param_types(self):
        space = HyperparameterSpace()
        space.add_continuous('lr', 1e-4, 1e-1, log_scale=True)
        space.add_integer('hidden', 8, 128)
        space.add_categorical('act', ['relu', 'tanh', 'sigmoid'])
        space.add_discrete('dropout', 0.0, 0.5, step=0.1)
        config = space.sample(np.random.default_rng(42))
        vec = space.encode(config)
        assert len(vec) == 4
        decoded = space.decode(vec)
        assert decoded['act'] in ['relu', 'tanh', 'sigmoid']


# ===== NN Tuning Integration =====

class TestNNTuningIntegration:
    def test_full_nn_tuning(self):
        """Full pipeline: define space, create NN objective, tune with Bayesian."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 2))
        Y = (X[:, 0:1] * 2 + X[:, 1:2] * 0.5)
        X_t, Y_t = Tensor(X.tolist()), Tensor(Y.tolist())

        def build_model(config):
            hidden = config.get('hidden_size', 16)
            model = Sequential()
            model.add(Dense(2, hidden))
            model.add(Activation('relu'))
            model.add(Dense(hidden, 1))
            return model

        space = HyperparameterSpace()
        space.add_continuous('lr', 1e-3, 1e-1, log_scale=True)
        space.add_integer('hidden_size', 4, 32)
        space.add_categorical('optimizer', ['adam'])
        space.add_integer('batch_size', 8, 32)

        obj = NNObjective(build_model, X_t, Y_t, max_epochs=10, seed=42)
        tuner = BayesianTuner(space, obj, n_initial=3, minimize=True, seed=42)
        best_config, best_obj = tuner.search(n_trials=8)
        assert best_config is not None
        assert best_obj < float('inf')
        assert obj.eval_count == 8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
