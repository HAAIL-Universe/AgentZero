"""
Tests for C012: Code Evolver -- Genetic Programming System

Tests organized by component:
1. Node construction and properties
2. Formatting
3. Evaluation
4. Random tree generation
5. Tree traversal utilities
6. Genetic operators (mutation, crossover)
7. Fitness functions
8. Selection
9. Bloat control
10. Evolution engine
11. Problem generators
12. Integration: symbolic regression
"""

import math
import random
import pytest
from evolver import (
    Node, NodeType, make_const, make_var, make_unary, make_binary, make_ternary,
    format_node, evaluate, random_tree, all_nodes, get_node_at, replace_node_at,
    mutate_point, mutate_subtree, mutate_hoist, mutate, crossover,
    TestCase, fitness_mse, fitness_with_parsimony,
    tournament_select, enforce_depth_limit,
    EvolutionConfig, EvolutionResult, Evolver,
    symbolic_regression, make_samples_1d, make_samples_2d,
    UNARY_OPS, BINARY_OPS,
)


# ============================================================
# 1. Node Construction and Properties
# ============================================================

class TestNodeConstruction:
    def test_const_node(self):
        n = make_const(3.14)
        assert n.type == NodeType.CONST
        assert n.value == 3.14
        assert n.children == []
        assert n.depth() == 1
        assert n.size() == 1

    def test_var_node(self):
        n = make_var('x')
        assert n.type == NodeType.VAR
        assert n.value == 'x'
        assert n.depth() == 1
        assert n.size() == 1

    def test_unary_node(self):
        n = make_unary('neg', make_const(5))
        assert n.type == NodeType.UNARY
        assert n.value == 'neg'
        assert len(n.children) == 1
        assert n.depth() == 2
        assert n.size() == 2

    def test_binary_node(self):
        n = make_binary('add', make_const(1), make_var('x'))
        assert n.type == NodeType.BINARY
        assert len(n.children) == 2
        assert n.depth() == 2
        assert n.size() == 3

    def test_ternary_node(self):
        n = make_ternary(make_var('x'), make_const(1), make_const(0))
        assert n.type == NodeType.TERNARY
        assert len(n.children) == 3
        assert n.depth() == 2
        assert n.size() == 4

    def test_deep_tree_depth(self):
        # add(add(add(1, 2), 3), 4) -> depth 4
        n = make_binary('add',
                make_binary('add',
                    make_binary('add', make_const(1), make_const(2)),
                    make_const(3)),
                make_const(4))
        assert n.depth() == 4
        assert n.size() == 7

    def test_copy_independence(self):
        orig = make_binary('add', make_const(1), make_const(2))
        copy = orig.copy()
        copy.children[0].value = 99
        assert orig.children[0].value == 1


# ============================================================
# 2. Formatting
# ============================================================

class TestFormatting:
    def test_format_const_int(self):
        assert format_node(make_const(5.0)) == "5"

    def test_format_const_float(self):
        assert format_node(make_const(3.14159)) == "3.142"

    def test_format_var(self):
        assert format_node(make_var('x')) == "x"

    def test_format_unary(self):
        n = make_unary('neg', make_var('x'))
        assert format_node(n) == "neg(x)"

    def test_format_binary(self):
        n = make_binary('add', make_const(1.0), make_var('x'))
        assert format_node(n) == "(1 add x)"

    def test_format_ternary(self):
        n = make_ternary(make_var('x'), make_const(1.0), make_const(0.0))
        assert format_node(n) == "(if x then 1 else 0)"

    def test_format_nested(self):
        n = make_binary('mul', make_var('x'), make_unary('square', make_var('x')))
        assert format_node(n) == "(x mul square(x))"

    def test_repr_uses_format(self):
        n = make_var('y')
        assert repr(n) == "y"


# ============================================================
# 3. Evaluation
# ============================================================

class TestEvaluation:
    def test_const(self):
        assert evaluate(make_const(42), {}) == 42.0

    def test_var(self):
        assert evaluate(make_var('x'), {'x': 7.0}) == 7.0

    def test_var_missing(self):
        assert evaluate(make_var('z'), {'x': 1}) == 0.0

    def test_add(self):
        n = make_binary('add', make_const(3), make_const(4))
        assert evaluate(n, {}) == 7.0

    def test_sub(self):
        n = make_binary('sub', make_const(10), make_const(3))
        assert evaluate(n, {}) == 7.0

    def test_mul(self):
        n = make_binary('mul', make_var('x'), make_var('x'))
        assert evaluate(n, {'x': 5}) == 25.0

    def test_safe_div(self):
        n = make_binary('safe_div', make_const(10), make_const(3))
        assert abs(evaluate(n, {}) - 10/3) < 1e-10

    def test_safe_div_by_zero(self):
        n = make_binary('safe_div', make_const(10), make_const(0))
        assert evaluate(n, {}) == 1.0

    def test_neg(self):
        n = make_unary('neg', make_const(5))
        assert evaluate(n, {}) == -5.0

    def test_abs(self):
        n = make_unary('abs', make_const(-5))
        assert evaluate(n, {}) == 5.0

    def test_square(self):
        n = make_unary('square', make_const(4))
        assert evaluate(n, {}) == 16.0

    def test_safe_sqrt(self):
        n = make_unary('safe_sqrt', make_const(9))
        assert evaluate(n, {}) == 3.0

    def test_safe_sqrt_negative(self):
        n = make_unary('safe_sqrt', make_const(-9))
        assert evaluate(n, {}) == 3.0  # sqrt(abs(-9))

    def test_safe_inv(self):
        n = make_unary('safe_inv', make_const(4))
        assert evaluate(n, {}) == 0.25

    def test_safe_inv_zero(self):
        n = make_unary('safe_inv', make_const(0))
        assert evaluate(n, {}) == 1.0

    def test_double(self):
        n = make_unary('double', make_const(7))
        assert evaluate(n, {}) == 14.0

    def test_half(self):
        n = make_unary('half', make_const(10))
        assert evaluate(n, {}) == 5.0

    def test_sin(self):
        n = make_unary('sin', make_const(0))
        assert abs(evaluate(n, {})) < 1e-10

    def test_cos(self):
        n = make_unary('cos', make_const(0))
        assert abs(evaluate(n, {}) - 1.0) < 1e-10

    def test_max(self):
        n = make_binary('max', make_const(3), make_const(7))
        assert evaluate(n, {}) == 7.0

    def test_min(self):
        n = make_binary('min', make_const(3), make_const(7))
        assert evaluate(n, {}) == 3.0

    def test_mod(self):
        n = make_binary('mod', make_const(7), make_const(3))
        assert evaluate(n, {}) == 1.0

    def test_mod_zero(self):
        n = make_binary('mod', make_const(7), make_const(0))
        assert evaluate(n, {}) == 0.0

    def test_pow(self):
        n = make_binary('pow', make_const(2), make_const(3))
        assert evaluate(n, {}) == 8.0

    def test_pow_overflow_clamped(self):
        n = make_binary('pow', make_const(2), make_const(1000))
        # Should not crash, b is clamped to 10
        result = evaluate(n, {})
        assert not math.isinf(result)

    def test_ternary_true(self):
        n = make_ternary(make_const(1), make_const(10), make_const(20))
        assert evaluate(n, {}) == 10.0

    def test_ternary_false(self):
        n = make_ternary(make_const(-1), make_const(10), make_const(20))
        assert evaluate(n, {}) == 20.0

    def test_ternary_zero_is_false(self):
        n = make_ternary(make_const(0), make_const(10), make_const(20))
        assert evaluate(n, {}) == 20.0

    def test_complex_expression(self):
        # (x + 1) * (x - 1) = x^2 - 1
        n = make_binary('mul',
                make_binary('add', make_var('x'), make_const(1)),
                make_binary('sub', make_var('x'), make_const(1)))
        assert evaluate(n, {'x': 5}) == 24.0  # 6 * 4

    def test_unknown_unary_raises(self):
        n = Node(NodeType.UNARY, value='bogus', children=[make_const(1)])
        with pytest.raises(ValueError, match="Unknown unary"):
            evaluate(n, {})

    def test_unknown_binary_raises(self):
        n = Node(NodeType.BINARY, value='bogus', children=[make_const(1), make_const(2)])
        with pytest.raises(ValueError, match="Unknown binary"):
            evaluate(n, {})


# ============================================================
# 4. Random Tree Generation
# ============================================================

class TestRandomTree:
    def test_returns_node(self):
        tree = random_tree(['x'], max_depth=3, rng=random.Random(42))
        assert isinstance(tree, Node)

    def test_respects_max_depth(self):
        for seed in range(20):
            tree = random_tree(['x', 'y'], max_depth=4, rng=random.Random(seed))
            assert tree.depth() <= 4

    def test_uses_variables(self):
        found_var = False
        for seed in range(50):
            tree = random_tree(['x'], max_depth=3, rng=random.Random(seed))
            nodes = all_nodes(tree)
            if any(n.type == NodeType.VAR for n, _ in nodes):
                found_var = True
                break
        assert found_var

    def test_uses_constants(self):
        found_const = False
        for seed in range(50):
            tree = random_tree(['x'], max_depth=3, rng=random.Random(seed))
            nodes = all_nodes(tree)
            if any(n.type == NodeType.CONST for n, _ in nodes):
                found_const = True
                break
        assert found_const

    def test_evaluates_without_error(self):
        for seed in range(30):
            tree = random_tree(['x', 'y'], max_depth=4, rng=random.Random(seed))
            result = evaluate(tree, {'x': 1.0, 'y': 2.0})
            assert isinstance(result, float)

    def test_deterministic_with_seed(self):
        t1 = random_tree(['x'], max_depth=3, rng=random.Random(123))
        t2 = random_tree(['x'], max_depth=3, rng=random.Random(123))
        assert format_node(t1) == format_node(t2)


# ============================================================
# 5. Tree Traversal Utilities
# ============================================================

class TestTraversal:
    def test_all_nodes_leaf(self):
        n = make_const(5)
        nodes = all_nodes(n)
        assert len(nodes) == 1
        assert nodes[0][1] == []  # empty path for root

    def test_all_nodes_count(self):
        n = make_binary('add', make_const(1), make_var('x'))
        nodes = all_nodes(n)
        assert len(nodes) == 3

    def test_get_node_at_root(self):
        n = make_const(5)
        assert get_node_at(n, []).value == 5

    def test_get_node_at_child(self):
        n = make_binary('add', make_const(1), make_var('x'))
        left = get_node_at(n, [0])
        right = get_node_at(n, [1])
        assert left.value == 1
        assert right.value == 'x'

    def test_replace_root(self):
        n = make_const(5)
        new = replace_node_at(n, [], make_const(10))
        assert new.value == 10
        assert n.value == 5  # original unchanged

    def test_replace_child(self):
        n = make_binary('add', make_const(1), make_const(2))
        new = replace_node_at(n, [0], make_const(99))
        assert new.children[0].value == 99
        assert new.children[1].value == 2
        assert n.children[0].value == 1  # original unchanged

    def test_replace_deep(self):
        n = make_binary('add',
                make_unary('neg', make_const(5)),
                make_const(3))
        new = replace_node_at(n, [0, 0], make_var('x'))
        assert get_node_at(new, [0, 0]).value == 'x'
        assert get_node_at(n, [0, 0]).value == 5  # original unchanged


# ============================================================
# 6. Genetic Operators
# ============================================================

class TestMutation:
    def test_point_mutation_changes_something(self):
        tree = make_binary('add', make_const(1), make_var('x'))
        changed = False
        for seed in range(50):
            mutated = mutate_point(tree, ['x', 'y'], rng=random.Random(seed))
            if format_node(mutated) != format_node(tree):
                changed = True
                break
        assert changed

    def test_point_mutation_preserves_structure(self):
        tree = make_binary('add', make_const(1), make_var('x'))
        for seed in range(20):
            mutated = mutate_point(tree, ['x'], rng=random.Random(seed))
            # Size should remain the same (point mutation doesn't change structure)
            assert mutated.size() == tree.size()

    def test_subtree_mutation_returns_valid_tree(self):
        tree = make_binary('mul', make_var('x'), make_const(2))
        mutated = mutate_subtree(tree, ['x'], rng=random.Random(42))
        result = evaluate(mutated, {'x': 3.0})
        assert isinstance(result, float)

    def test_hoist_mutation_simplifies(self):
        tree = make_binary('add',
                make_unary('neg', make_const(5)),
                make_binary('mul', make_var('x'), make_const(3)))
        hoisted = mutate_hoist(tree, rng=random.Random(42))
        assert hoisted.size() < tree.size() or hoisted.size() == tree.size()

    def test_hoist_on_leaf(self):
        tree = make_const(5)
        hoisted = mutate_hoist(tree, rng=random.Random(42))
        assert hoisted.value == 5

    def test_mutate_returns_valid(self):
        tree = make_binary('add', make_var('x'), make_const(1))
        for seed in range(20):
            mutated = mutate(tree, ['x'], rng=random.Random(seed))
            result = evaluate(mutated, {'x': 5.0})
            assert isinstance(result, float)

    def test_original_unchanged_after_mutation(self):
        tree = make_binary('add', make_const(1), make_const(2))
        original_fmt = format_node(tree)
        mutate(tree, ['x'], rng=random.Random(42))
        assert format_node(tree) == original_fmt


class TestCrossover:
    def test_crossover_returns_two_children(self):
        p1 = make_binary('add', make_var('x'), make_const(1))
        p2 = make_binary('mul', make_var('x'), make_const(2))
        c1, c2 = crossover(p1, p2, rng=random.Random(42))
        assert isinstance(c1, Node)
        assert isinstance(c2, Node)

    def test_crossover_children_evaluate(self):
        p1 = make_binary('add', make_var('x'), make_const(1))
        p2 = make_unary('square', make_var('x'))
        c1, c2 = crossover(p1, p2, rng=random.Random(42))
        assert isinstance(evaluate(c1, {'x': 3}), float)
        assert isinstance(evaluate(c2, {'x': 3}), float)

    def test_crossover_preserves_parents(self):
        p1 = make_binary('add', make_var('x'), make_const(1))
        p2 = make_binary('mul', make_var('x'), make_const(2))
        f1 = format_node(p1)
        f2 = format_node(p2)
        crossover(p1, p2, rng=random.Random(42))
        assert format_node(p1) == f1
        assert format_node(p2) == f2

    def test_crossover_can_produce_different_offspring(self):
        p1 = make_binary('add', make_var('x'), make_const(1))
        p2 = make_binary('mul', make_const(3), make_unary('neg', make_var('x')))
        different = False
        for seed in range(30):
            c1, c2 = crossover(p1, p2, rng=random.Random(seed))
            if format_node(c1) != format_node(p1):
                different = True
                break
        assert different


# ============================================================
# 7. Fitness Functions
# ============================================================

class TestFitness:
    def test_perfect_fitness(self):
        # f(x) = x + 1, test with x+1 tree
        tree = make_binary('add', make_var('x'), make_const(1))
        cases = [
            TestCase({'x': 0}, 1),
            TestCase({'x': 1}, 2),
            TestCase({'x': 5}, 6),
        ]
        assert fitness_mse(tree, cases) < 1e-10

    def test_imperfect_fitness(self):
        # f(x) = x, but tree is x+1
        tree = make_binary('add', make_var('x'), make_const(1))
        cases = [
            TestCase({'x': 0}, 0),
            TestCase({'x': 1}, 1),
            TestCase({'x': 2}, 2),
        ]
        assert abs(fitness_mse(tree, cases) - 1.0) < 1e-10  # MSE = 1

    def test_empty_cases(self):
        tree = make_const(0)
        assert fitness_mse(tree, []) == float('inf')

    def test_parsimony_prefers_smaller(self):
        simple = make_var('x')
        complex_tree = make_binary('add', make_var('x'), make_const(0))
        cases = [TestCase({'x': i}, float(i)) for i in range(5)]
        f1 = fitness_with_parsimony(simple, cases, parsimony_weight=0.1)
        f2 = fitness_with_parsimony(complex_tree, cases, parsimony_weight=0.1)
        assert f1 < f2  # simpler is better when accuracy is equal

    def test_fitness_with_inf(self):
        # A tree that produces inf should get inf fitness
        tree = make_binary('safe_div', make_const(1), make_const(0))
        cases = [TestCase({'x': 0}, 0)]
        fit = fitness_mse(tree, cases)
        assert isinstance(fit, float)  # should not crash


# ============================================================
# 8. Selection
# ============================================================

class TestSelection:
    def test_tournament_select_returns_node(self):
        pop = [
            (make_const(i), float(i)) for i in range(10)
        ]
        winner = tournament_select(pop, tournament_size=3, rng=random.Random(42))
        assert isinstance(winner, Node)

    def test_tournament_prefers_fitter(self):
        # With 5/10 best individuals, tournament size 3 should pick best often
        best = make_const(0)
        worst = make_const(100)
        pop = [(best, 0.0)] * 5 + [(worst, 100.0)] * 5
        win_count = 0
        rng = random.Random(42)
        for _ in range(100):
            winner = tournament_select(pop, tournament_size=3, rng=rng)
            if winner.value == 0:
                win_count += 1
        assert win_count > 70  # best should win most tournaments


# ============================================================
# 9. Bloat Control
# ============================================================

class TestBloatControl:
    def test_within_limit_unchanged(self):
        tree = make_binary('add', make_const(1), make_const(2))
        result = enforce_depth_limit(tree, 5, ['x'])
        assert result.depth() == tree.depth()

    def test_exceeding_limit_trimmed(self):
        # Build a deep tree
        tree = make_unary('neg',
                make_unary('abs',
                    make_unary('square',
                        make_unary('neg',
                            make_unary('abs', make_var('x'))))))
        assert tree.depth() == 6
        result = enforce_depth_limit(tree, 3, ['x'], rng=random.Random(42))
        assert result.depth() <= 3

    def test_trimmed_still_evaluates(self):
        tree = make_unary('neg',
                make_unary('abs',
                    make_unary('square',
                        make_unary('neg', make_var('x')))))
        result = enforce_depth_limit(tree, 2, ['x'], rng=random.Random(42))
        val = evaluate(result, {'x': 5})
        assert isinstance(val, float)


# ============================================================
# 10. Evolution Engine
# ============================================================

class TestEvolver:
    def _make_simple_evolver(self, seed=42):
        """Evolver for x + 1."""
        cases = [TestCase({'x': float(i)}, float(i + 1)) for i in range(-5, 6)]
        config = EvolutionConfig(
            population_size=50,
            max_generations=50,
            tournament_size=3,
            elitism=3,
            fitness_threshold=0.01,
        )
        return Evolver(['x'], cases, config, seed=seed)

    def test_initialize(self):
        ev = self._make_simple_evolver()
        ev.initialize()
        assert len(ev.population) == 50
        assert all(isinstance(t, Node) and isinstance(f, float)
                    for t, f in ev.population)

    def test_step(self):
        ev = self._make_simple_evolver()
        ev.initialize()
        fit = ev.step()
        assert isinstance(fit, float)
        assert ev.generation == 1
        assert len(ev.fitness_history) == 1

    def test_population_size_maintained(self):
        ev = self._make_simple_evolver()
        ev.initialize()
        for _ in range(5):
            ev.step()
        assert len(ev.population) == 50

    def test_best(self):
        ev = self._make_simple_evolver()
        ev.initialize()
        best_tree, best_fit = ev.best()
        assert isinstance(best_tree, Node)
        assert isinstance(best_fit, float)

    def test_run_completes(self):
        ev = self._make_simple_evolver()
        result = ev.run()
        assert isinstance(result, EvolutionResult)
        assert result.generations_run > 0
        assert result.generations_run <= 50
        assert len(result.fitness_history) == result.generations_run

    def test_fitness_improves(self):
        ev = self._make_simple_evolver()
        result = ev.run()
        # First fitness should be >= last fitness (lower is better)
        assert result.fitness_history[-1] <= result.fitness_history[0] + 1e-10

    def test_convergence_on_easy_problem(self):
        """Test that evolver can solve x + 1 (a very easy problem)."""
        # Try multiple seeds since GP is stochastic
        solved = False
        for seed in range(10):
            cases = [TestCase({'x': float(i)}, float(i + 1)) for i in range(-5, 6)]
            config = EvolutionConfig(
                population_size=200,
                max_generations=100,
                tournament_size=5,
                elitism=5,
                fitness_threshold=0.05,
                parsimony_weight=0.001,
            )
            ev = Evolver(['x'], cases, config, seed=seed)
            result = ev.run()
            if result.converged:
                solved = True
                break
        assert solved, "Failed to solve x+1 in 10 attempts"

    def test_diversity_tracking(self):
        ev = self._make_simple_evolver()
        result = ev.run()
        assert len(result.population_diversity) == result.generations_run
        assert all(0 <= d <= 1 for d in result.population_diversity)

    def test_stagnation_injection(self):
        """Force stagnation and verify diversity injection happens."""
        # Use a very hard problem with tiny population to trigger stagnation
        cases = [TestCase({'x': float(i)}, math.sin(i) * i**2) for i in range(-5, 6)]
        config = EvolutionConfig(
            population_size=20,
            max_generations=50,
            stagnation_limit=5,
            fitness_threshold=0.0001,
        )
        ev = Evolver(['x'], cases, config, seed=42)
        ev.run()
        # Just verify it completes without error
        assert ev.generation > 0

    def test_elitism_preserves_best(self):
        ev = self._make_simple_evolver()
        ev.initialize()
        ev.step()
        best_before = ev.best()[1]
        ev.step()
        best_after = ev.best()[1]
        # With elitism, best should never get worse
        assert best_after <= best_before + 1e-10


# ============================================================
# 11. Problem Generators
# ============================================================

class TestProblemGenerators:
    def test_symbolic_regression(self):
        cases = symbolic_regression(
            lambda x: x * x,
            ['x'],
            [{'x': 0}, {'x': 1}, {'x': 2}, {'x': 3}]
        )
        assert len(cases) == 4
        assert cases[0].expected == 0
        assert cases[2].expected == 4
        assert cases[3].expected == 9

    def test_make_samples_1d(self):
        samples = make_samples_1d((-1, 1), 5)
        assert len(samples) == 5
        assert samples[0] == {'x': -1.0}
        assert abs(samples[-1]['x'] - 1.0) < 1e-10

    def test_make_samples_2d(self):
        samples = make_samples_2d((-1, 1), (-1, 1), 3)
        assert len(samples) == 9  # 3x3 grid

    def test_make_samples_1d_single(self):
        samples = make_samples_1d((0, 10), 1)
        assert len(samples) == 1
        assert samples[0] == {'x': 0.0}


# ============================================================
# 12. Integration: Symbolic Regression
# ============================================================

class TestIntegration:
    def test_evolve_identity(self):
        """Can we evolve f(x) = x?"""
        samples = make_samples_1d((-5, 5), 11)
        cases = symbolic_regression(lambda x: x, ['x'], samples)
        config = EvolutionConfig(
            population_size=150,
            max_generations=80,
            fitness_threshold=0.01,
            parsimony_weight=0.001,
        )
        solved = False
        for seed in range(5):
            ev = Evolver(['x'], cases, config, seed=seed)
            result = ev.run()
            if result.best_fitness < 0.1:
                solved = True
                break
        assert solved, "Failed to evolve identity function"

    def test_evolve_constant(self):
        """Can we evolve f(x) = 5?"""
        samples = make_samples_1d((-5, 5), 11)
        cases = symbolic_regression(lambda x: 5.0, ['x'], samples)
        config = EvolutionConfig(
            population_size=100,
            max_generations=50,
            fitness_threshold=0.1,
        )
        solved = False
        for seed in range(5):
            ev = Evolver(['x'], cases, config, seed=seed)
            result = ev.run()
            if result.best_fitness < 0.5:
                solved = True
                break
        assert solved, "Failed to evolve constant function"

    def test_evolve_quadratic(self):
        """Can we evolve f(x) = x^2? (harder but achievable)."""
        samples = make_samples_1d((-3, 3), 13)
        cases = symbolic_regression(lambda x: x * x, ['x'], samples)
        config = EvolutionConfig(
            population_size=300,
            max_generations=150,
            fitness_threshold=0.1,
            parsimony_weight=0.0005,
            stagnation_limit=25,
        )
        solved = False
        for seed in range(8):
            ev = Evolver(['x'], cases, config, seed=seed)
            result = ev.run()
            if result.best_fitness < 1.0:
                solved = True
                break
        assert solved, "Failed to evolve x^2"

    def test_evolved_program_generalizes(self):
        """Test that evolved program works on unseen data."""
        # Train on integers, test on half-integers
        train_cases = [TestCase({'x': float(i)}, float(i + 1)) for i in range(-5, 6)]
        test_cases = [TestCase({'x': i + 0.5}, i + 1.5) for i in range(-5, 5)]
        config = EvolutionConfig(
            population_size=200,
            max_generations=100,
            fitness_threshold=0.01,
        )
        for seed in range(10):
            ev = Evolver(['x'], train_cases, config, seed=seed)
            result = ev.run()
            if result.converged:
                # Test generalization
                test_fit = fitness_mse(result.best_program, test_cases)
                assert test_fit < 1.0, f"Evolved program doesn't generalize: {test_fit}"
                return
        pytest.skip("Could not converge on training set")

    def test_two_variable_problem(self):
        """Can we evolve f(x, y) = x + y?"""
        samples = make_samples_2d((-3, 3), (-3, 3), 5)
        cases = symbolic_regression(lambda x, y: x + y, ['x', 'y'], samples)
        config = EvolutionConfig(
            population_size=300,
            max_generations=100,
            fitness_threshold=0.1,
            parsimony_weight=0.001,
        )
        solved = False
        for seed in range(8):
            ev = Evolver(['x', 'y'], cases, config, seed=seed)
            result = ev.run()
            if result.best_fitness < 1.0:
                solved = True
                break
        assert solved, "Failed to evolve x + y"

    def test_all_operations_are_safe(self):
        """Verify all ops handle edge cases without crashing."""
        for name, fn in UNARY_OPS.items():
            for val in [0, 1, -1, 1e10, -1e10, 0.001]:
                try:
                    result = fn(val)
                    assert not (math.isinf(result) and name.startswith('safe'))
                except (OverflowError, ValueError):
                    pass  # acceptable for non-safe ops

        for name, fn in BINARY_OPS.items():
            for a in [0, 1, -1, 100]:
                for b in [0, 1, -1, 100]:
                    try:
                        result = fn(a, b)
                    except (OverflowError, ValueError, ZeroDivisionError):
                        pass  # acceptable
