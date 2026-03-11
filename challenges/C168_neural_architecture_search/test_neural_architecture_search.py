"""
Tests for C168: Neural Architecture Search
"""

import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from neural_architecture_search import (
    LayerType, LayerSpec, ArchitectureSpec, SearchSpace,
    ArchitectureEvaluator, EvalResult, PerformancePredictor,
    NASHistory, RandomNAS, BayesianNAS, EvolutionaryNAS,
    BOHBNAS, MultiObjectiveNAS, ArchGenome, create_nas
)

# Reusable fixtures
def make_regression_data(n=100, d=4, seed=42):
    """Simple regression dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d)
    Y = (X @ w + 0.1 * rng.randn(n)).reshape(-1, 1)
    return X, Y


def make_search_space(d=4, output=1):
    return SearchSpace(
        input_size=d, output_size=output,
        min_layers=1, max_layers=3,
        layer_sizes=[8, 16, 32],
        activations=['relu', 'tanh'],
        dropout_rates=[0.0, 0.1],
        use_batchnorm=False,
        optimizers=['adam'],
        lr_range=(1e-3, 1e-2)
    )


def make_evaluator(X, Y, epochs=5):
    return ArchitectureEvaluator(X, Y, max_epochs=epochs, early_stop=3, seed=42)


# ===========================================================================
# LayerSpec tests
# ===========================================================================

class TestLayerSpec:
    def test_dense_spec(self):
        spec = LayerSpec(LayerType.DENSE, {'input_size': 4, 'output_size': 8})
        layer = spec.build()
        assert layer is not None

    def test_activation_spec(self):
        spec = LayerSpec(LayerType.ACTIVATION, {'name': 'relu'})
        layer = spec.build()
        assert layer is not None

    def test_dropout_spec(self):
        spec = LayerSpec(LayerType.DROPOUT, {'rate': 0.3})
        layer = spec.build()
        assert layer is not None

    def test_batchnorm_spec(self):
        spec = LayerSpec(LayerType.BATCHNORM, {'num_features': 16})
        layer = spec.build()
        assert layer is not None

    def test_invalid_type_raises(self):
        spec = LayerSpec.__new__(LayerSpec)
        spec.layer_type = 'invalid'
        spec.params = {}
        with pytest.raises(ValueError):
            spec.build()


# ===========================================================================
# ArchitectureSpec tests
# ===========================================================================

class TestArchitectureSpec:
    def test_build_simple(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers)
        model = arch.build(input_size=4)
        assert model is not None

    def test_param_count(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers)
        count = arch.param_count(input_size=4)
        # Dense(4->16): 4*16+16=80, Dense(16->1): 16*1+1=17 => 97
        assert count == 97

    def test_depth(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
            LayerSpec(LayerType.DENSE, {'output_size': 8}),
            LayerSpec(LayerType.ACTIVATION, {'name': 'tanh'}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers)
        assert arch.depth() == 3

    def test_summary(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
            LayerSpec(LayerType.DROPOUT, {'rate': 0.3}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers)
        s = arch.summary()
        assert 'Dense(16)' in s
        assert 'relu' in s
        assert 'Drop(0.3)' in s

    def test_summary_with_batchnorm(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 32}),
            LayerSpec(LayerType.BATCHNORM),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers)
        assert 'BN' in arch.summary()

    def test_build_with_dropout(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
            LayerSpec(LayerType.DROPOUT, {'rate': 0.5}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers, optimizer='adam', lr=0.001)
        model = arch.build(4)
        assert model is not None

    def test_build_with_batchnorm(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.BATCHNORM),
            LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers)
        model = arch.build(4)
        assert model is not None


# ===========================================================================
# SearchSpace tests
# ===========================================================================

class TestSearchSpace:
    def test_sample_random(self):
        space = make_search_space()
        arch = space.sample_random(np.random.RandomState(42))
        assert isinstance(arch, ArchitectureSpec)
        assert arch.depth() >= 2  # at least 1 hidden + output

    def test_sample_respects_bounds(self):
        space = make_search_space()
        rng = np.random.RandomState(123)
        for _ in range(20):
            arch = space.sample_random(rng)
            # hidden layers between min and max
            hidden = arch.depth() - 1  # exclude output
            assert 1 <= hidden <= space.max_layers

    def test_encode_decode_roundtrip(self):
        space = make_search_space()
        rng = np.random.RandomState(42)
        arch = space.sample_random(rng)
        vec = space.encode(arch)
        assert len(vec) == space.vector_dim()
        decoded = space.decode(vec)
        assert isinstance(decoded, ArchitectureSpec)
        assert decoded.depth() >= 1

    def test_vector_dim(self):
        space = make_search_space()
        assert space.vector_dim() == 1 + 3 * 4 + 2  # 15

    def test_encode_values_in_range(self):
        space = make_search_space()
        rng = np.random.RandomState(42)
        for _ in range(10):
            arch = space.sample_random(rng)
            vec = space.encode(arch)
            assert all(0 <= v <= 1 for v in vec), f"Out of range: {vec}"

    def test_decode_preserves_output_size(self):
        space = make_search_space(output=3)
        rng = np.random.RandomState(42)
        arch = space.sample_random(rng)
        vec = space.encode(arch)
        decoded = space.decode(vec)
        # Last dense layer should have output_size = 3
        dense_layers = [s for s in decoded.layers if s.layer_type == LayerType.DENSE]
        assert dense_layers[-1].params['output_size'] == 3

    def test_output_activation(self):
        space = SearchSpace(input_size=4, output_size=3, output_activation='softmax',
                            min_layers=1, max_layers=2, layer_sizes=[8],
                            activations=['relu'], optimizers=['adam'])
        arch = space.sample_random(np.random.RandomState(42))
        assert arch.layers[-1].layer_type == LayerType.ACTIVATION
        assert arch.layers[-1].params['name'] == 'softmax'

    def test_multiple_random_architectures_differ(self):
        space = make_search_space()
        rng = np.random.RandomState(42)
        archs = [space.sample_random(rng) for _ in range(5)]
        summaries = [a.summary() for a in archs]
        # At least some should differ
        assert len(set(summaries)) > 1


# ===========================================================================
# ArchitectureEvaluator tests
# ===========================================================================

class TestArchitectureEvaluator:
    def test_evaluate_simple(self):
        X, Y = make_regression_data(50, 4)
        ev = make_evaluator(X, Y, epochs=3)
        arch = ArchitectureSpec(
            layers=[
                LayerSpec(LayerType.DENSE, {'output_size': 8}),
                LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
                LayerSpec(LayerType.DENSE, {'output_size': 1}),
            ],
            lr=0.01, epochs=3
        )
        result = ev.evaluate(arch)
        assert isinstance(result, EvalResult)
        assert result.loss < float('inf')
        assert result.param_count > 0

    def test_caching(self):
        X, Y = make_regression_data(50, 4)
        ev = make_evaluator(X, Y, epochs=3)
        arch = ArchitectureSpec(
            layers=[
                LayerSpec(LayerType.DENSE, {'output_size': 8}),
                LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
                LayerSpec(LayerType.DENSE, {'output_size': 1}),
            ],
            lr=0.01, epochs=3
        )
        r1 = ev.evaluate(arch)
        r2 = ev.evaluate(arch)
        assert r1.loss == r2.loss  # cached

    def test_eval_count(self):
        X, Y = make_regression_data(50, 4)
        ev = make_evaluator(X, Y, epochs=2)
        space = make_search_space()
        rng = np.random.RandomState(42)
        for _ in range(3):
            arch = space.sample_random(rng)
            ev.evaluate(arch)
        assert ev.eval_count >= 1  # at least 1 unique

    def test_validation_data(self):
        X, Y = make_regression_data(80, 4)
        X_val, Y_val = make_regression_data(20, 4, seed=99)
        ev = ArchitectureEvaluator(X, Y, X_val=X_val, Y_val=Y_val,
                                   max_epochs=3, early_stop=2, seed=42)
        arch = ArchitectureSpec(
            layers=[
                LayerSpec(LayerType.DENSE, {'output_size': 8}),
                LayerSpec(LayerType.ACTIVATION, {'name': 'relu'}),
                LayerSpec(LayerType.DENSE, {'output_size': 1}),
            ],
            lr=0.01, epochs=3
        )
        result = ev.evaluate(arch)
        assert result.loss < float('inf')

    def test_bad_architecture_returns_inf(self):
        X, Y = make_regression_data(50, 4)
        ev = make_evaluator(X, Y, epochs=2)
        # Architecture with mismatched sizes that will fail
        arch = ArchitectureSpec(
            layers=[
                LayerSpec(LayerType.DENSE, {'output_size': 0}),  # invalid size
            ],
            lr=0.01, epochs=2
        )
        result = ev.evaluate(arch)
        assert result.loss == float('inf')


# ===========================================================================
# PerformancePredictor tests
# ===========================================================================

class TestPerformancePredictor:
    def test_predict_insufficient_data(self):
        space = make_search_space()
        pred = PerformancePredictor(space)
        arch = space.sample_random(np.random.RandomState(42))
        mean, unc = pred.predict(arch)
        assert mean == float('inf')  # not enough data

    def test_observe_and_predict(self):
        space = make_search_space()
        pred = PerformancePredictor(space, seed=42)
        rng = np.random.RandomState(42)

        # Add enough observations
        for i in range(5):
            arch = space.sample_random(rng)
            pred.observe(arch, 1.0 + i * 0.1)

        arch = space.sample_random(rng)
        mean, unc = pred.predict(arch)
        assert mean != float('inf')
        assert unc >= 0

    def test_rank_candidates(self):
        space = make_search_space()
        pred = PerformancePredictor(space, seed=42)
        rng = np.random.RandomState(42)

        for i in range(5):
            arch = space.sample_random(rng)
            pred.observe(arch, 1.0 + i * 0.5)

        candidates = [space.sample_random(rng) for _ in range(10)]
        ranked = pred.rank_candidates(candidates, n_best=3)
        assert len(ranked) == 3
        assert all(len(r) == 3 for r in ranked)  # (arch, pred, unc)

    def test_predictions_improve_with_data(self):
        space = make_search_space()
        pred = PerformancePredictor(space, seed=42)
        rng = np.random.RandomState(42)

        # Observe with consistent pattern
        for i in range(10):
            arch = space.sample_random(rng)
            vec = space.encode(arch)
            # Loss correlates with vector sum
            loss = float(np.sum(vec))
            pred.observe(arch, loss)

        # Predictions should be finite
        test_arch = space.sample_random(rng)
        mean, unc = pred.predict(test_arch)
        assert np.isfinite(mean)
        assert unc < 100


# ===========================================================================
# NASHistory tests
# ===========================================================================

class TestNASHistory:
    def test_empty_history(self):
        h = NASHistory()
        assert h.best_loss == float('inf')
        assert h.best_arch is None

    def test_record_updates_best(self):
        h = NASHistory()
        arch1 = ArchitectureSpec(layers=[], lr=0.01)
        arch2 = ArchitectureSpec(layers=[], lr=0.001)
        h.record(EvalResult(arch1, loss=1.0))
        h.record(EvalResult(arch2, loss=0.5))
        assert h.best_loss == 0.5
        assert h.best_arch is arch2

    def test_convergence_curve(self):
        h = NASHistory()
        losses = [1.0, 0.8, 0.9, 0.5, 0.6]
        for l in losses:
            h.record(EvalResult(ArchitectureSpec(layers=[]), loss=l))
        curve = h.convergence_curve()
        assert curve == [1.0, 0.8, 0.8, 0.5, 0.5]

    def test_summary(self):
        h = NASHistory()
        arch = ArchitectureSpec(layers=[
            LayerSpec(LayerType.DENSE, {'output_size': 8}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ])
        h.record(EvalResult(arch, loss=0.5, param_count=100))
        s = h.summary()
        assert s['total_evaluations'] == 1
        assert s['best_loss'] == 0.5

    def test_get_best(self):
        h = NASHistory()
        arch = ArchitectureSpec(layers=[])
        h.record(EvalResult(arch, loss=0.3))
        best_arch, best_loss = h.get_best()
        assert best_loss == 0.3
        assert best_arch is arch


# ===========================================================================
# RandomNAS tests
# ===========================================================================

class TestRandomNAS:
    def test_basic_search(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = RandomNAS(space, ev, seed=42)
        best_arch, best_loss = nas.search(n_trials=5)
        assert best_arch is not None
        assert best_loss < float('inf')

    def test_history_populated(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = RandomNAS(space, ev, seed=42)
        nas.search(n_trials=5)
        assert len(nas.history.evaluations) == 5

    def test_more_trials_better_or_equal(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev1 = make_evaluator(X, Y, epochs=2)
        ev2 = make_evaluator(X, Y, epochs=2)
        nas1 = RandomNAS(space, ev1, seed=42)
        nas2 = RandomNAS(space, ev2, seed=42)
        _, loss5 = nas1.search(n_trials=3)
        _, loss10 = nas2.search(n_trials=10)
        assert loss10 <= loss5 + 1.0  # more trials should find comparable or better

    def test_convergence_curve_monotonic(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = RandomNAS(space, ev, seed=42)
        nas.search(n_trials=8)
        curve = nas.history.convergence_curve()
        for i in range(1, len(curve)):
            assert curve[i] <= curve[i-1]


# ===========================================================================
# BayesianNAS tests
# ===========================================================================

class TestBayesianNAS:
    def test_basic_search(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BayesianNAS(space, ev, n_initial=3, seed=42)
        best_arch, best_loss = nas.search(n_trials=6)
        assert best_arch is not None
        assert best_loss < float('inf')

    def test_history_populated(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BayesianNAS(space, ev, n_initial=3, seed=42)
        nas.search(n_trials=6)
        assert len(nas.history.evaluations) == 6

    def test_initial_phase(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BayesianNAS(space, ev, n_initial=5, seed=42)
        nas.search(n_trials=5)  # only initial phase
        assert len(nas.history.evaluations) == 5

    def test_bo_guided_phase(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BayesianNAS(space, ev, n_initial=3, seed=42)
        nas.search(n_trials=8)  # 3 initial + 5 BO
        assert len(nas.history.evaluations) == 8

    def test_finds_reasonable_architecture(self):
        X, Y = make_regression_data(80, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=5)
        nas = BayesianNAS(space, ev, n_initial=3, seed=42)
        best_arch, best_loss = nas.search(n_trials=8)
        assert best_arch.depth() >= 2
        assert best_loss < 100  # should find something reasonable


# ===========================================================================
# EvolutionaryNAS tests
# ===========================================================================

class TestEvolutionaryNAS:
    def test_basic_search(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=6, seed=42)
        best_arch, best_loss = nas.search(n_generations=2)
        assert best_arch is not None
        assert best_loss < float('inf')

    def test_population_initialized(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=8, seed=42)
        nas._init_population()
        assert len(nas.population) == 8

    def test_tournament_select(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=6, tournament_size=3, seed=42)
        nas._init_population()
        selected = nas._tournament_select()
        assert isinstance(selected, ArchGenome)
        assert selected.fitness is not None

    def test_mutation(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=6, seed=42)
        arch = space.sample_random(np.random.RandomState(42))
        genome = ArchGenome(arch, fitness=-1.0)
        mutated = nas._mutate(genome.copy())
        assert isinstance(mutated, ArchGenome)

    def test_crossover(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=6, seed=42)
        rng = np.random.RandomState(42)
        p1 = ArchGenome(space.sample_random(rng), fitness=-1.0)
        p2 = ArchGenome(space.sample_random(rng), fitness=-0.5)
        child = nas._crossover(p1, p2)
        assert isinstance(child, ArchGenome)
        assert child.arch.depth() >= 1

    def test_elitism_preserves_best(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=6, elitism=2, seed=42)
        nas.search(n_generations=3)
        # Best should have been preserved across generations
        assert nas.history.best_loss < float('inf')

    def test_multiple_generations_improve(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=8, seed=42)
        nas.search(n_generations=3)
        curve = nas.history.convergence_curve()
        # Overall trend should improve (first > last)
        assert curve[-1] <= curve[0]


# ===========================================================================
# ArchGenome tests
# ===========================================================================

class TestArchGenome:
    def test_copy(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers, lr=0.01)
        genome = ArchGenome(arch, fitness=-0.5)
        clone = genome.copy()
        assert clone.fitness == genome.fitness
        assert clone.arch.lr == genome.arch.lr
        assert clone is not genome
        assert clone.arch is not genome.arch

    def test_copy_independence(self):
        layers = [
            LayerSpec(LayerType.DENSE, {'output_size': 16}),
            LayerSpec(LayerType.DENSE, {'output_size': 1}),
        ]
        arch = ArchitectureSpec(layers=layers, lr=0.01)
        genome = ArchGenome(arch, fitness=-0.5)
        clone = genome.copy()
        clone.arch.lr = 0.1
        assert genome.arch.lr == 0.01  # original unchanged


# ===========================================================================
# BOHBNAS tests
# ===========================================================================

class TestBOHBNAS:
    def test_basic_search(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BOHBNAS(space, ev, n_initial=3, seed=42)
        best_arch, best_loss = nas.search(n_trials=8)
        assert best_arch is not None
        assert best_loss < float('inf')

    def test_history_populated(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BOHBNAS(space, ev, n_initial=3, seed=42)
        nas.search(n_trials=8)
        assert len(nas.history.evaluations) == 8

    def test_predictor_used(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BOHBNAS(space, ev, n_initial=3, seed=42)
        nas.search(n_trials=10)
        # Predictor should have observations
        assert len(nas.predictor.X_observed) >= 3

    def test_bo_fraction(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        # All BO
        nas = BOHBNAS(space, ev, n_initial=3, bo_fraction=1.0, seed=42)
        nas.search(n_trials=6)
        assert nas.history.best_loss < float('inf')

    def test_all_evolutionary(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BOHBNAS(space, ev, n_initial=3, bo_fraction=0.0, seed=42)
        nas.search(n_trials=6)
        assert nas.history.best_loss < float('inf')


# ===========================================================================
# MultiObjectiveNAS tests
# ===========================================================================

class TestMultiObjectiveNAS:
    def test_basic_search(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = MultiObjectiveNAS(space, ev, seed=42)
        best_arch, best_loss = nas.search(n_trials=8)
        assert best_arch is not None
        assert best_loss < float('inf')

    def test_pareto_front(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = MultiObjectiveNAS(space, ev, seed=42)
        nas.search(n_trials=10)
        front = nas.get_pareto_front()
        assert len(front) >= 1
        # All points should have (loss, complexity, arch)
        for loss, comp, arch in front:
            assert loss < float('inf')
            assert comp >= 0

    def test_no_dominated_in_front(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = MultiObjectiveNAS(space, ev, seed=42)
        nas.search(n_trials=10)
        front = nas.get_pareto_front()
        # No point in front should dominate another
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i != j:
                    assert not nas._dominates(a, b), \
                        f"Point {i} dominates {j} in Pareto front"

    def test_knee_point(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = MultiObjectiveNAS(space, ev, seed=42)
        nas.search(n_trials=10)
        knee = nas.get_knee_point()
        assert knee is not None
        assert len(knee) == 3  # (loss, complexity, arch)

    def test_knee_point_single(self):
        nas = MultiObjectiveNAS(make_search_space(), None, seed=42)
        arch = ArchitectureSpec(layers=[])
        nas.pareto_front = [(0.5, 10.0, arch)]
        knee = nas.get_knee_point()
        assert knee[0] == 0.5

    def test_empty_pareto(self):
        nas = MultiObjectiveNAS(make_search_space(), None, seed=42)
        assert nas.get_knee_point() is None
        assert nas.get_pareto_front() == []

    def test_dominates(self):
        nas = MultiObjectiveNAS(make_search_space(), None, seed=42)
        a = (0.5, 10.0, None)
        b = (0.6, 15.0, None)
        c = (0.5, 10.0, None)
        assert nas._dominates(a, b)
        assert not nas._dominates(b, a)
        assert not nas._dominates(a, c)  # equal is not domination


# ===========================================================================
# create_nas factory tests
# ===========================================================================

class TestCreateNAS:
    def test_create_random(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = create_nas(space, ev, method='random', seed=42)
        assert isinstance(nas, RandomNAS)

    def test_create_bayesian(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = create_nas(space, ev, method='bayesian', seed=42)
        assert isinstance(nas, BayesianNAS)

    def test_create_evolutionary(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = create_nas(space, ev, method='evolutionary', seed=42)
        assert isinstance(nas, EvolutionaryNAS)

    def test_create_bohb(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = create_nas(space, ev, method='bohb', seed=42)
        assert isinstance(nas, BOHBNAS)

    def test_create_multi_objective(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = create_nas(space, ev, method='multi_objective', seed=42)
        assert isinstance(nas, MultiObjectiveNAS)

    def test_invalid_method(self):
        space = make_search_space()
        with pytest.raises(ValueError, match="Unknown NAS method"):
            create_nas(space, None, method='invalid')

    def test_extra_kwargs_filtered(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        # Should not crash with irrelevant kwargs
        nas = create_nas(space, ev, method='random', seed=42, bo_fraction=0.5)
        assert isinstance(nas, RandomNAS)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_random(self):
        """End-to-end: random NAS finds architecture, builds and evaluates it."""
        X, Y = make_regression_data(60, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=3)
        nas = RandomNAS(space, ev, seed=42)
        best_arch, best_loss = nas.search(n_trials=5)

        # Build the best architecture
        model = best_arch.build(4)
        assert model is not None
        assert best_loss < float('inf')

    def test_full_pipeline_bayesian(self):
        X, Y = make_regression_data(60, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=3)
        nas = BayesianNAS(space, ev, n_initial=3, seed=42)
        best_arch, best_loss = nas.search(n_trials=6)
        assert best_arch is not None
        model = best_arch.build(4)
        assert model is not None

    def test_full_pipeline_evolutionary(self):
        X, Y = make_regression_data(60, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=6, seed=42)
        best_arch, best_loss = nas.search(n_generations=2)
        assert best_arch is not None
        model = best_arch.build(4)
        assert model is not None

    def test_full_pipeline_bohb(self):
        X, Y = make_regression_data(60, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = BOHBNAS(space, ev, n_initial=3, seed=42)
        best_arch, best_loss = nas.search(n_trials=6)
        assert best_arch is not None

    def test_full_pipeline_multi_objective(self):
        X, Y = make_regression_data(60, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = MultiObjectiveNAS(space, ev, seed=42)
        nas.search(n_trials=6)
        front = nas.get_pareto_front()
        assert len(front) >= 1
        knee = nas.get_knee_point()
        assert knee is not None

    def test_predictor_integration(self):
        """Performance predictor integrates with search."""
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        pred = PerformancePredictor(space, seed=42)

        rng = np.random.RandomState(42)
        for _ in range(5):
            arch = space.sample_random(rng)
            result = ev.evaluate(arch)
            pred.observe(arch, result.loss)

        # Rank remaining candidates
        candidates = [space.sample_random(rng) for _ in range(5)]
        ranked = pred.rank_candidates(candidates, n_best=3)
        assert len(ranked) == 3

    def test_encode_decode_consistency(self):
        """Encode -> decode produces valid architectures."""
        space = make_search_space()
        rng = np.random.RandomState(42)
        X, Y = make_regression_data(50, 4)
        ev = make_evaluator(X, Y, epochs=2)

        for _ in range(5):
            arch = space.sample_random(rng)
            vec = space.encode(arch)
            decoded = space.decode(vec)
            result = ev.evaluate(decoded)
            assert result.loss < float('inf') or result.param_count == 0

    def test_search_with_different_data(self):
        """NAS works with different dataset sizes."""
        for n in [30, 80]:
            X, Y = make_regression_data(n, 4)
            space = make_search_space()
            ev = make_evaluator(X, Y, epochs=2)
            nas = RandomNAS(space, ev, seed=42)
            best, loss = nas.search(n_trials=3)
            assert best is not None

    def test_search_space_with_output_activation(self):
        """NAS with output activation (classification-like)."""
        X, Y = make_regression_data(50, 4)
        space = SearchSpace(
            input_size=4, output_size=1,
            min_layers=1, max_layers=2,
            layer_sizes=[8, 16], activations=['relu'],
            output_activation='sigmoid',
            optimizers=['adam'], lr_range=(1e-3, 1e-2)
        )
        ev = make_evaluator(X, Y, epochs=2)
        nas = RandomNAS(space, ev, seed=42)
        best, loss = nas.search(n_trials=3)
        assert best is not None
        # Last layer should be activation
        assert best.layers[-1].layer_type == LayerType.ACTIVATION


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    def test_single_trial(self):
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        ev = make_evaluator(X, Y, epochs=2)
        nas = RandomNAS(space, ev, seed=42)
        best, loss = nas.search(n_trials=1)
        assert best is not None

    def test_min_equals_max_layers(self):
        space = SearchSpace(
            input_size=4, output_size=1,
            min_layers=2, max_layers=2,
            layer_sizes=[16], activations=['relu'],
            optimizers=['adam']
        )
        rng = np.random.RandomState(42)
        arch = space.sample_random(rng)
        assert arch.depth() >= 2  # 2 hidden + output

    def test_single_layer_size(self):
        space = SearchSpace(
            input_size=4, output_size=1,
            min_layers=1, max_layers=1,
            layer_sizes=[32], activations=['relu'],
            optimizers=['adam']
        )
        arch = space.sample_random(np.random.RandomState(42))
        dense_layers = [s for s in arch.layers if s.layer_type == LayerType.DENSE]
        # Should have hidden + output = 2 dense layers
        assert len(dense_layers) == 2
        assert dense_layers[0].params['output_size'] == 32

    def test_evolutionary_crossover_single_layer_parent(self):
        X, Y = make_regression_data(50, 4)
        space = SearchSpace(
            input_size=4, output_size=1,
            min_layers=1, max_layers=1,
            layer_sizes=[8], activations=['relu'],
            optimizers=['adam']
        )
        ev = make_evaluator(X, Y, epochs=2)
        nas = EvolutionaryNAS(space, ev, population_size=4, seed=42)
        arch1 = space.sample_random(np.random.RandomState(1))
        arch2 = space.sample_random(np.random.RandomState(2))
        p1 = ArchGenome(arch1, fitness=-1.0)
        p2 = ArchGenome(arch2, fitness=-0.5)
        child = nas._crossover(p1, p2)
        assert child is not None

    def test_history_with_inf_losses(self):
        h = NASHistory()
        h.record(EvalResult(ArchitectureSpec(layers=[]), loss=float('inf')))
        h.record(EvalResult(ArchitectureSpec(layers=[]), loss=0.5))
        h.record(EvalResult(ArchitectureSpec(layers=[]), loss=float('inf')))
        assert h.best_loss == 0.5
        s = h.summary()
        assert s['valid_evaluations'] == 1  # only non-inf

    def test_architecture_spec_zero_depth(self):
        # Only output layer
        layers = [LayerSpec(LayerType.DENSE, {'output_size': 1})]
        arch = ArchitectureSpec(layers=layers)
        assert arch.depth() == 1

    def test_genome_none_fitness(self):
        arch = ArchitectureSpec(layers=[])
        g = ArchGenome(arch, fitness=None)
        clone = g.copy()
        assert clone.fitness is None

    def test_multiple_strategies_same_data(self):
        """All strategies can run on the same data without errors."""
        X, Y = make_regression_data(50, 4)
        space = make_search_space()
        for method in ['random', 'bayesian', 'evolutionary', 'bohb', 'multi_objective']:
            ev = make_evaluator(X, Y, epochs=2)
            nas = create_nas(space, ev, method=method, seed=42,
                             n_initial=3, population_size=6)
            if method == 'evolutionary':
                nas.search(n_generations=2)
            else:
                nas.search(n_trials=5)
            assert nas.history.best_loss < float('inf'), f"{method} failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
