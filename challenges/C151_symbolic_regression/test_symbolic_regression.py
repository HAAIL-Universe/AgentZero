"""Tests for C151: Symbolic Regression (GP + AD)"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from symbolic_regression import (
    ExprNode, Op, UNARY_OPS, BINARY_OPS,
    evaluate, _eval_float, const, var, unary, binary,
    random_tree, full_tree,
    ConstantOptimizer, OptimizeResult,
    SymbolicRegressor, SRConfig, SRResult,
    Simplifier,
    MultiObjectiveRegressor, ParetoSolution,
    FeatureSelector,
    make_dataset, symbolic_regression
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))
from autodiff import Var, Dual


# ===================================================================
# 1. ExprTree tests
# ===================================================================

class TestExprNode:
    def test_const_node(self):
        n = const(3.14)
        assert n.op == Op.CONST
        assert n.value == 3.14
        assert n.size() == 1
        assert n.depth() == 0

    def test_var_node(self):
        n = var('x')
        assert n.op == Op.VAR
        assert n.var_name == 'x'

    def test_unary_node(self):
        n = unary(Op.NEG, const(5))
        assert n.op == Op.NEG
        assert len(n.children) == 1
        assert n.size() == 2
        assert n.depth() == 1

    def test_binary_node(self):
        n = binary(Op.ADD, var('x'), const(1))
        assert n.op == Op.ADD
        assert len(n.children) == 2
        assert n.size() == 3
        assert n.depth() == 1

    def test_nested_tree(self):
        # (x + 1) * sin(x)
        tree = binary(Op.MUL,
                       binary(Op.ADD, var('x'), const(1)),
                       unary(Op.SIN, var('x')))
        assert tree.size() == 6
        assert tree.depth() == 2

    def test_copy_deep(self):
        tree = binary(Op.ADD, var('x'), const(1))
        copied = tree.copy()
        copied.children[1].value = 99
        assert tree.children[1].value == 1

    def test_constants_collection(self):
        tree = binary(Op.ADD, const(3), binary(Op.MUL, const(2), var('x')))
        consts = tree.constants()
        assert len(consts) == 2
        assert consts[0].value == 3
        assert consts[1].value == 2

    def test_format_const(self):
        assert repr(const(3)) == '3'
        assert '3.14' in repr(const(3.14))

    def test_format_var(self):
        assert repr(var('x')) == 'x'

    def test_format_binary(self):
        tree = binary(Op.ADD, var('x'), const(1))
        assert repr(tree) == '(x + 1)'

    def test_format_unary(self):
        tree = unary(Op.SIN, var('x'))
        assert repr(tree) == 'sin(x)'

    def test_format_square(self):
        tree = unary(Op.SQUARE, var('x'))
        assert repr(tree) == '(x)^2'


# ===================================================================
# 2. Evaluation tests
# ===================================================================

class TestEvaluation:
    def test_eval_const(self):
        assert evaluate(const(42), {}) == 42

    def test_eval_var(self):
        assert evaluate(var('x'), {'x': 7.0}) == 7.0

    def test_eval_add(self):
        tree = binary(Op.ADD, var('x'), const(3))
        assert evaluate(tree, {'x': 2.0}) == 5.0

    def test_eval_sub(self):
        tree = binary(Op.SUB, var('x'), const(1))
        assert evaluate(tree, {'x': 5.0}) == 4.0

    def test_eval_mul(self):
        tree = binary(Op.MUL, var('x'), const(3))
        assert evaluate(tree, {'x': 4.0}) == 12.0

    def test_eval_div_normal(self):
        tree = binary(Op.DIV, var('x'), const(2))
        assert evaluate(tree, {'x': 6.0}) == 3.0

    def test_eval_div_zero(self):
        """Protected division returns 1.0 for zero denominator."""
        tree = binary(Op.DIV, var('x'), const(0))
        assert evaluate(tree, {'x': 5.0}) == 1.0

    def test_eval_neg(self):
        tree = unary(Op.NEG, var('x'))
        assert evaluate(tree, {'x': 3.0}) == -3.0

    def test_eval_abs(self):
        tree = unary(Op.ABS, var('x'))
        assert evaluate(tree, {'x': -4.0}) == 4.0

    def test_eval_sin(self):
        tree = unary(Op.SIN, const(0))
        assert abs(evaluate(tree, {}) - 0.0) < 1e-10

    def test_eval_cos(self):
        tree = unary(Op.COS, const(0))
        assert abs(evaluate(tree, {}) - 1.0) < 1e-10

    def test_eval_exp(self):
        tree = unary(Op.EXP, const(1))
        assert abs(evaluate(tree, {}) - math.e) < 1e-6

    def test_eval_exp_overflow_protection(self):
        """Large inputs should be clamped."""
        tree = unary(Op.EXP, const(1000))
        result = evaluate(tree, {})
        assert math.isfinite(result)

    def test_eval_log(self):
        tree = unary(Op.LOG, const(math.e))
        assert abs(evaluate(tree, {}) - 1.0) < 1e-6

    def test_eval_log_negative(self):
        """Protected log returns 0 for non-positive."""
        tree = unary(Op.LOG, const(-1))
        assert evaluate(tree, {}) == 0.0

    def test_eval_sqrt(self):
        tree = unary(Op.SQRT, const(9))
        assert abs(evaluate(tree, {}) - 3.0) < 1e-10

    def test_eval_sqrt_negative(self):
        tree = unary(Op.SQRT, const(-4))
        assert evaluate(tree, {}) == 0.0

    def test_eval_tanh(self):
        tree = unary(Op.TANH, const(0))
        assert abs(evaluate(tree, {}) - 0.0) < 1e-10

    def test_eval_square(self):
        tree = unary(Op.SQUARE, var('x'))
        assert evaluate(tree, {'x': 3.0}) == 9.0

    def test_eval_cube(self):
        tree = unary(Op.CUBE, var('x'))
        assert evaluate(tree, {'x': 2.0}) == 8.0

    def test_eval_nested(self):
        # x^2 + 2*x + 1
        tree = binary(Op.ADD,
                       binary(Op.ADD,
                              unary(Op.SQUARE, var('x')),
                              binary(Op.MUL, const(2), var('x'))),
                       const(1))
        assert evaluate(tree, {'x': 3.0}) == 16.0  # 9 + 6 + 1

    def test_eval_pow(self):
        tree = binary(Op.POW, const(2), const(3))
        result = evaluate(tree, {})
        assert abs(result - 8.0) < 1e-10

    def test_eval_with_var_ad(self):
        """Evaluation with Var objects for AD."""
        tree = binary(Op.MUL, var('x'), const(3))
        result = evaluate(tree, {'x': Var(2.0)})
        assert isinstance(result, Var)
        assert abs(result.val - 6.0) < 1e-10

    def test_eval_with_dual_ad(self):
        """Evaluation with Dual objects."""
        tree = binary(Op.ADD, var('x'), const(1))
        result = evaluate(tree, {'x': Dual(3.0, 1.0)})
        assert isinstance(result, Dual)
        assert abs(result.val - 4.0) < 1e-10
        assert abs(result.der - 1.0) < 1e-10

    def test_eval_float_wrapper(self):
        tree = binary(Op.ADD, var('x'), const(1))
        assert _eval_float(tree, {'x': 5.0}) == 6.0

    def test_eval_multivar(self):
        # x + y
        tree = binary(Op.ADD, var('x'), var('y'))
        assert evaluate(tree, {'x': 3.0, 'y': 4.0}) == 7.0


# ===================================================================
# 3. Tree generation tests
# ===================================================================

class TestTreeGeneration:
    def test_random_tree_creates_valid(self):
        import random
        rng = random.Random(42)
        tree = random_tree(['x'], max_depth=3, rng=rng)
        assert isinstance(tree, ExprNode)
        assert tree.depth() <= 3

    def test_random_tree_respects_depth(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            tree = random_tree(['x', 'y'], max_depth=2, rng=rng)
            assert tree.depth() <= 2

    def test_random_tree_evaluatable(self):
        import random
        rng = random.Random(42)
        for _ in range(20):
            tree = random_tree(['x'], max_depth=3, rng=rng)
            result = _eval_float(tree, {'x': 1.0})
            assert isinstance(result, float)

    def test_full_tree_reaches_depth(self):
        import random
        rng = random.Random(42)
        tree = full_tree(['x'], depth=3, rng=rng)
        assert tree.depth() >= 2  # Full trees are mostly deep

    def test_full_tree_evaluatable(self):
        import random
        rng = random.Random(42)
        tree = full_tree(['x', 'y'], depth=2, rng=rng)
        result = _eval_float(tree, {'x': 1.0, 'y': 2.0})
        assert isinstance(result, float)

    def test_random_tree_uses_vars(self):
        import random
        rng = random.Random(42)
        found_var = False
        for _ in range(50):
            tree = random_tree(['x'], max_depth=3, rng=rng)
            if 'x' in repr(tree):
                found_var = True
                break
        assert found_var

    def test_random_tree_deterministic(self):
        import random
        t1 = random_tree(['x'], max_depth=3, rng=random.Random(42))
        t2 = random_tree(['x'], max_depth=3, rng=random.Random(42))
        assert repr(t1) == repr(t2)


# ===================================================================
# 4. ConstantOptimizer tests
# ===================================================================

class TestConstantOptimizer:
    def test_optimize_linear(self):
        """Optimize a*x to match y=2x."""
        tree = binary(Op.MUL, const(1.0), var('x'))
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [2.0 * i for i in range(1, 6)]
        opt = ConstantOptimizer(lr=0.1, max_iter=100)
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.final_loss < result.initial_loss
        assert result.final_loss < 0.5

    def test_optimize_constant_offset(self):
        """Optimize x + c to match y=x+3."""
        tree = binary(Op.ADD, var('x'), const(0.0))
        data_x = [{'x': float(i)} for i in range(1, 11)]
        data_y = [float(i) + 3.0 for i in range(1, 11)]
        opt = ConstantOptimizer(lr=0.1, max_iter=200)
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.final_loss < 1.0

    def test_optimize_no_constants(self):
        """Tree with no constants is unchanged."""
        tree = var('x')
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i) for i in range(1, 6)]
        opt = ConstantOptimizer()
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.iterations == 0
        assert result.initial_loss == result.final_loss

    def test_optimize_returns_best(self):
        """Even if optimization diverges, returns best found."""
        tree = binary(Op.MUL, const(5.0), var('x'))
        data_x = [{'x': 1.0}, {'x': 2.0}]
        data_y = [3.0, 6.0]
        opt = ConstantOptimizer(lr=0.01, max_iter=50)
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.final_loss <= result.initial_loss + 1e-6

    def test_optimize_preserves_structure(self):
        """Optimization changes constants, not structure."""
        tree = binary(Op.ADD, const(1.0), binary(Op.MUL, const(1.0), var('x')))
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [2.0 * float(i) + 5.0 for i in range(1, 6)]
        opt = ConstantOptimizer(lr=0.05, max_iter=100)
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.tree.op == Op.ADD
        assert result.tree.children[1].op == Op.MUL

    def test_optimize_multiple_constants(self):
        """Optimize a*x + b."""
        tree = binary(Op.ADD, binary(Op.MUL, const(0.5), var('x')), const(0.5))
        data_x = [{'x': float(i)} for i in range(-5, 6)]
        data_y = [3.0 * float(i) + 7.0 for i in range(-5, 6)]
        opt = ConstantOptimizer(lr=0.01, max_iter=200)
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.final_loss < result.initial_loss


# ===================================================================
# 5. Simplifier tests
# ===================================================================

class TestSimplifier:
    def setup_method(self):
        self.s = Simplifier()

    def test_add_zero_left(self):
        tree = binary(Op.ADD, const(0), var('x'))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR
        assert result.var_name == 'x'

    def test_add_zero_right(self):
        tree = binary(Op.ADD, var('x'), const(0))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_sub_zero(self):
        tree = binary(Op.SUB, var('x'), const(0))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_sub_self(self):
        tree = binary(Op.SUB, var('x'), var('x'))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 0

    def test_mul_zero(self):
        tree = binary(Op.MUL, const(0), var('x'))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 0

    def test_mul_one(self):
        tree = binary(Op.MUL, const(1), var('x'))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_mul_one_right(self):
        tree = binary(Op.MUL, var('x'), const(1))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_div_one(self):
        tree = binary(Op.DIV, var('x'), const(1))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_div_self(self):
        tree = binary(Op.DIV, var('x'), var('x'))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 1

    def test_div_zero_numerator(self):
        tree = binary(Op.DIV, const(0), var('x'))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 0

    def test_pow_zero(self):
        tree = binary(Op.POW, var('x'), const(0))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 1

    def test_pow_one(self):
        tree = binary(Op.POW, var('x'), const(1))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_double_neg(self):
        tree = unary(Op.NEG, unary(Op.NEG, var('x')))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR

    def test_constant_folding(self):
        tree = binary(Op.ADD, const(3), const(4))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 7

    def test_nested_constant_folding(self):
        tree = binary(Op.MUL, const(2), binary(Op.ADD, const(3), const(4)))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 14

    def test_neg_const(self):
        tree = unary(Op.NEG, const(5))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == -5

    def test_square_const(self):
        tree = unary(Op.SQUARE, const(3))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 9

    def test_cube_const(self):
        tree = unary(Op.CUBE, const(2))
        result = self.s.simplify(tree)
        assert result.op == Op.CONST
        assert result.value == 8

    def test_simplify_complex(self):
        # (x + 0) * 1 -> x
        tree = binary(Op.MUL, binary(Op.ADD, var('x'), const(0)), const(1))
        result = self.s.simplify(tree)
        assert result.op == Op.VAR
        assert result.var_name == 'x'

    def test_simplify_preserves_non_trivial(self):
        # x + y is already simple
        tree = binary(Op.ADD, var('x'), var('y'))
        result = self.s.simplify(tree)
        assert result.op == Op.ADD

    def test_structural_equality(self):
        assert self.s._equal(const(3), const(3))
        assert not self.s._equal(const(3), const(4))
        assert self.s._equal(var('x'), var('x'))
        assert not self.s._equal(var('x'), var('y'))


# ===================================================================
# 6. SymbolicRegressor tests
# ===================================================================

class TestSymbolicRegressor:
    def test_init(self):
        sr = SymbolicRegressor(['x'])
        assert sr.var_names == ['x']
        assert sr.config.population_size == 300

    def test_init_custom_config(self):
        cfg = SRConfig(population_size=50, max_generations=10)
        sr = SymbolicRegressor(['x'], config=cfg)
        assert sr.config.population_size == 50

    def test_fitness_perfect(self):
        """Tree that perfectly matches data should have low fitness."""
        sr = SymbolicRegressor(['x'], seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i) for i in range(1, 6)]
        sr.data_x = data_x
        sr.data_y = data_y
        tree = var('x')
        fitness = sr._fitness(tree)
        assert fitness < 0.1  # near zero MSE plus tiny parsimony

    def test_fitness_with_parsimony(self):
        """Larger trees get higher fitness penalty."""
        sr = SymbolicRegressor(['x'], config=SRConfig(parsimony_weight=1.0), seed=42)
        sr.data_x = [{'x': 1.0}]
        sr.data_y = [1.0]

        small = var('x')
        big = binary(Op.ADD, var('x'), binary(Op.SUB, const(0), const(0)))
        f_small = sr._fitness(small)
        f_big = sr._fitness(big)
        assert f_big > f_small  # bigger tree penalized more

    def test_fit_constant_function(self):
        """Should find constant for constant data."""
        cfg = SRConfig(population_size=100, max_generations=30, optimize_constants=True,
                       const_opt_frequency=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(-5, 6)]
        data_y = [5.0] * 11
        result = sr.fit(data_x, data_y)
        assert result.best_fitness < 1.0

    def test_fit_linear(self):
        """Should find something close to y=x."""
        cfg = SRConfig(population_size=200, max_generations=50, optimize_constants=True,
                       const_opt_frequency=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(-5, 6)]
        data_y = [float(i) for i in range(-5, 6)]
        result = sr.fit(data_x, data_y)
        assert result.best_fitness < 5.0  # Should be close

    def test_fit_returns_result(self):
        cfg = SRConfig(population_size=50, max_generations=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': 1.0}, {'x': 2.0}]
        data_y = [1.0, 4.0]
        result = sr.fit(data_x, data_y)
        assert isinstance(result, SRResult)
        assert result.generations_run > 0
        assert len(result.fitness_history) > 0
        assert isinstance(result.best_expr, ExprNode)

    def test_fit_hall_of_fame(self):
        cfg = SRConfig(population_size=100, max_generations=10)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i)**2 for i in range(1, 6)]
        result = sr.fit(data_x, data_y)
        assert len(result.hall_of_fame) > 0

    def test_fit_fitness_decreases(self):
        cfg = SRConfig(population_size=100, max_generations=20)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [2.0 * float(i) for i in range(1, 6)]
        result = sr.fit(data_x, data_y)
        # First fitness should be >= last fitness (improvement or stagnation)
        assert result.fitness_history[-1] <= result.fitness_history[0] + 1e-6

    def test_tournament_select(self):
        cfg = SRConfig(population_size=20, max_generations=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        sr.data_x = [{'x': 1.0}]
        sr.data_y = [1.0]
        sr._initialize()
        selected = sr._tournament_select()
        assert isinstance(selected, ExprNode)

    def test_crossover(self):
        sr = SymbolicRegressor(['x'], seed=42)
        p1 = binary(Op.ADD, var('x'), const(1))
        p2 = binary(Op.MUL, var('x'), const(2))
        c1, c2 = sr._crossover(p1, p2)
        assert isinstance(c1, ExprNode)
        assert isinstance(c2, ExprNode)

    def test_mutate(self):
        sr = SymbolicRegressor(['x'], seed=42)
        tree = binary(Op.ADD, var('x'), const(1))
        mutated = sr._mutate(tree)
        assert isinstance(mutated, ExprNode)

    def test_mutate_point(self):
        sr = SymbolicRegressor(['x'], seed=42)
        tree = binary(Op.ADD, var('x'), const(1))
        mutated = sr._mutate_point(tree.copy())
        assert isinstance(mutated, ExprNode)

    def test_mutate_subtree(self):
        sr = SymbolicRegressor(['x'], seed=42)
        tree = binary(Op.ADD, var('x'), const(1))
        mutated = sr._mutate_subtree(tree.copy())
        assert isinstance(mutated, ExprNode)

    def test_mutate_hoist(self):
        sr = SymbolicRegressor(['x'], seed=42)
        tree = binary(Op.ADD, var('x'), binary(Op.MUL, var('x'), const(2)))
        mutated = sr._mutate_hoist(tree.copy())
        assert isinstance(mutated, ExprNode)
        assert mutated.size() <= tree.size()

    def test_enforce_depth(self):
        sr = SymbolicRegressor(['x'], config=SRConfig(max_tree_depth=3), seed=42)
        # Create deep tree
        tree = var('x')
        for _ in range(10):
            tree = unary(Op.NEG, tree)
        assert tree.depth() == 10
        trimmed = sr._enforce_depth(tree)
        assert trimmed.depth() <= 3

    def test_inject_diversity(self):
        cfg = SRConfig(population_size=20, max_generations=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        sr.data_x = [{'x': 1.0}]
        sr.data_y = [1.0]
        sr._initialize()
        old_worst = sr.population[-1][1]
        sr._inject_diversity()
        # Population should still be valid
        assert len(sr.population) == 20

    def test_multivar_regression(self):
        """Regression with multiple variables."""
        cfg = SRConfig(population_size=100, max_generations=20)
        sr = SymbolicRegressor(['x', 'y'], config=cfg, seed=42)
        data_x = [{'x': float(i), 'y': float(j)} for i in range(1, 4) for j in range(1, 4)]
        data_y = [xi['x'] + xi['y'] for xi in data_x]
        result = sr.fit(data_x, data_y)
        assert isinstance(result, SRResult)
        assert result.best_fitness < 50.0

    def test_stagnation_triggers_diversity(self):
        cfg = SRConfig(population_size=50, max_generations=30, stagnation_limit=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [math.sin(float(i)) for i in range(1, 6)]
        result = sr.fit(data_x, data_y)
        # Should run and not crash even with stagnation
        assert result.generations_run > 0

    def test_custom_ops(self):
        cfg = SRConfig(
            population_size=50, max_generations=5,
            unary_ops=[Op.NEG, Op.SQUARE],
            binary_ops=[Op.ADD, Op.MUL]
        )
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i)**2 for i in range(1, 6)]
        result = sr.fit(data_x, data_y)
        assert isinstance(result, SRResult)


# ===================================================================
# 7. MultiObjectiveRegressor tests
# ===================================================================

class TestMultiObjectiveRegressor:
    def test_pareto_front(self):
        cfg = SRConfig(population_size=100, max_generations=15)
        mor = MultiObjectiveRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i)**2 for i in range(1, 6)]
        front = mor.fit(data_x, data_y)
        assert len(front) > 0
        for sol in front:
            assert isinstance(sol, ParetoSolution)
            assert sol.complexity >= 1

    def test_pareto_front_sorted_by_complexity(self):
        cfg = SRConfig(population_size=100, max_generations=15)
        mor = MultiObjectiveRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [2.0 * float(i) for i in range(1, 6)]
        front = mor.fit(data_x, data_y)
        for i in range(len(front) - 1):
            assert front[i].complexity <= front[i+1].complexity

    def test_pareto_dominance(self):
        """No solution on the front should be dominated by another."""
        cfg = SRConfig(population_size=100, max_generations=15)
        mor = MultiObjectiveRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i) for i in range(1, 6)]
        front = mor.fit(data_x, data_y)
        for i, a in enumerate(front):
            for j, b in enumerate(front):
                if i == j:
                    continue
                # b should not dominate a
                assert not (b.fitness <= a.fitness and b.complexity <= a.complexity and
                            (b.fitness < a.fitness or b.complexity < a.complexity))

    def test_pareto_with_simplification(self):
        cfg = SRConfig(population_size=50, max_generations=10)
        mor = MultiObjectiveRegressor(['x'], config=cfg, seed=42)
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [float(i) for i in range(1, 6)]
        front = mor.fit(data_x, data_y)
        # All expressions should be simplified
        for sol in front:
            assert isinstance(sol.expr, ExprNode)


# ===================================================================
# 8. FeatureSelector tests
# ===================================================================

class TestFeatureSelector:
    def test_select_relevant_feature(self):
        """Should identify that x is relevant, z is not, for y=x^2."""
        cfg = SRConfig(population_size=100, max_generations=20)
        fs = FeatureSelector(['x', 'z'], config=cfg, seed=42)
        data_x = [{'x': float(i), 'z': float(i) * 0.1} for i in range(-5, 6)]
        data_y = [float(i)**2 for i in range(-5, 6)]
        result = fs.select(data_x, data_y)
        assert 'selected' in result
        assert 'importance' in result
        assert 'best_expr' in result
        assert isinstance(result['importance'], dict)

    def test_select_returns_importance(self):
        cfg = SRConfig(population_size=50, max_generations=10)
        fs = FeatureSelector(['x', 'y'], config=cfg, seed=42)
        data_x = [{'x': float(i), 'y': float(i)} for i in range(1, 6)]
        data_y = [float(i) for i in range(1, 6)]
        result = fs.select(data_x, data_y)
        assert 'x' in result['importance']
        assert 'y' in result['importance']

    def test_select_max_features(self):
        cfg = SRConfig(population_size=50, max_generations=10)
        fs = FeatureSelector(['x', 'y', 'z'], config=cfg, seed=42)
        data_x = [{'x': float(i), 'y': 0.0, 'z': 0.0} for i in range(1, 6)]
        data_y = [float(i) for i in range(1, 6)]
        result = fs.select(data_x, data_y, max_features=2)
        assert len(result['selected']) <= 2

    def test_variables_used(self):
        fs = FeatureSelector(['x', 'y'])
        tree = binary(Op.ADD, var('x'), unary(Op.SIN, var('y')))
        used = fs._variables_used(tree)
        assert used == {'x', 'y'}

    def test_variables_used_const_only(self):
        fs = FeatureSelector(['x'])
        tree = const(5)
        used = fs._variables_used(tree)
        assert used == set()


# ===================================================================
# 9. Convenience function tests
# ===================================================================

class TestConvenience:
    def test_make_dataset(self):
        data_x, data_y = make_dataset(lambda x: x**2, ['x'], n_samples=20, seed=42)
        assert len(data_x) == 20
        assert len(data_y) == 20
        assert all('x' in d for d in data_x)

    def test_make_dataset_multivar(self):
        data_x, data_y = make_dataset(lambda x, y: x + y, ['x', 'y'], n_samples=10, seed=42)
        assert len(data_x) == 10
        assert all('x' in d and 'y' in d for d in data_x)

    def test_make_dataset_handles_errors(self):
        def bad_func(x):
            if x < 0:
                raise ValueError("negative")
            return x
        data_x, data_y = make_dataset(bad_func, ['x'], n_samples=20,
                                       x_range=(-5, 5), seed=42)
        assert len(data_y) == 20  # errors produce 0.0

    def test_symbolic_regression_convenience(self):
        result = symbolic_regression(
            lambda x: 2 * x + 1, ['x'], n_samples=30,
            x_range=(-3, 3), seed=42,
            config=SRConfig(population_size=100, max_generations=20)
        )
        assert isinstance(result, SRResult)
        assert result.generations_run > 0

    def test_make_dataset_range(self):
        data_x, data_y = make_dataset(lambda x: x, ['x'], n_samples=100,
                                       x_range=(0, 10), seed=42)
        for d in data_x:
            assert 0 <= d['x'] <= 10


# ===================================================================
# 10. AD Integration tests
# ===================================================================

class TestADIntegration:
    def test_gradient_through_add(self):
        """Gradient of (c + x) w.r.t. c should be 1."""
        tree = binary(Op.ADD, const(2.0), var('x'))
        c_var = Var(2.0)
        tree.children[0]._ad_var = c_var
        opt = ConstantOptimizer()
        result = opt._eval_with_ad(tree, {'x': Var(3.0)})
        result.backward()
        assert abs(c_var.grad - 1.0) < 1e-6

    def test_gradient_through_mul(self):
        """Gradient of (c * x) w.r.t. c is x."""
        tree = binary(Op.MUL, const(2.0), var('x'))
        c_var = Var(2.0)
        tree.children[0]._ad_var = c_var
        opt = ConstantOptimizer()
        result = opt._eval_with_ad(tree, {'x': Var(3.0)})
        result.backward()
        assert abs(c_var.grad - 3.0) < 1e-6

    def test_gradient_through_sin(self):
        """Gradient of sin(c) w.r.t. c is cos(c)."""
        tree = unary(Op.SIN, const(0.0))
        c_var = Var(0.0)
        tree.children[0]._ad_var = c_var
        opt = ConstantOptimizer()
        result = opt._eval_with_ad(tree, {})
        result.backward()
        assert abs(c_var.grad - 1.0) < 1e-6  # cos(0) = 1

    def test_gradient_through_square(self):
        """Gradient of c^2 w.r.t. c is 2c."""
        tree = unary(Op.SQUARE, const(3.0))
        c_var = Var(3.0)
        tree.children[0]._ad_var = c_var
        opt = ConstantOptimizer()
        result = opt._eval_with_ad(tree, {})
        result.backward()
        assert abs(c_var.grad - 6.0) < 1e-6

    def test_forward_mode_dual(self):
        """Forward-mode AD with Dual numbers."""
        tree = binary(Op.MUL, var('x'), var('x'))  # x^2
        result = evaluate(tree, {'x': Dual(3.0, 1.0)})
        assert abs(result.val - 9.0) < 1e-10
        assert abs(result.der - 6.0) < 1e-10  # d/dx(x^2) = 2x = 6

    def test_reverse_mode_var(self):
        """Reverse-mode AD with Var objects."""
        tree = binary(Op.ADD, binary(Op.MUL, var('x'), var('x')), var('x'))  # x^2 + x
        x = Var(2.0)
        result = evaluate(tree, {'x': x})
        result.backward()
        assert abs(result.val - 6.0) < 1e-10  # 4 + 2
        assert abs(x.grad - 5.0) < 1e-10  # 2x + 1 = 5

    def test_ad_with_nested_ops(self):
        """AD through nested operations."""
        tree = unary(Op.EXP, binary(Op.MUL, const(-1), unary(Op.SQUARE, var('x'))))
        x = Var(0.0)
        result = evaluate(tree, {'x': x})
        assert abs(result.val - 1.0) < 1e-6  # exp(-0^2) = 1

    def test_constant_optimization_improves_ad_fitness(self):
        """Constant optimization via AD should reduce MSE."""
        # Target: y = 3x + 7
        # Initial: y = 1*x + 1
        tree = binary(Op.ADD, binary(Op.MUL, const(1.0), var('x')), const(1.0))
        data_x = [{'x': float(i)} for i in range(-5, 6)]
        data_y = [3.0 * i + 7.0 for i in range(-5, 6)]
        opt = ConstantOptimizer(lr=0.05, max_iter=200)
        result = opt.optimize(tree, data_x, data_y, ['x'])
        assert result.final_loss < result.initial_loss


# ===================================================================
# 11. Edge cases and robustness
# ===================================================================

class TestEdgeCases:
    def test_empty_tree_eval(self):
        """Const evaluates fine with empty env."""
        assert evaluate(const(5), {}) == 5

    def test_single_point_regression(self):
        cfg = SRConfig(population_size=20, max_generations=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        result = sr.fit([{'x': 1.0}], [5.0])
        assert isinstance(result, SRResult)

    def test_large_values_dont_crash(self):
        tree = unary(Op.EXP, const(100))
        result = _eval_float(tree, {})
        assert math.isfinite(result)

    def test_negative_sqrt_protected(self):
        tree = unary(Op.SQRT, const(-100))
        assert _eval_float(tree, {}) == 0.0

    def test_negative_log_protected(self):
        tree = unary(Op.LOG, const(-5))
        assert _eval_float(tree, {}) == 0.0

    def test_zero_division_protected(self):
        tree = binary(Op.DIV, const(1), const(0))
        assert _eval_float(tree, {}) == 1.0

    def test_pow_overflow_protected(self):
        tree = binary(Op.POW, const(1000), const(1000))
        result = _eval_float(tree, {})
        assert math.isfinite(result)

    def test_all_unary_ops(self):
        for op in UNARY_OPS:
            tree = unary(op, const(1.0))
            result = _eval_float(tree, {})
            assert math.isfinite(result), f"Op {op} produced non-finite"

    def test_all_binary_ops(self):
        for op in BINARY_OPS:
            tree = binary(op, const(2.0), const(3.0))
            result = _eval_float(tree, {})
            assert math.isfinite(result), f"Op {op} produced non-finite"

    def test_deep_tree_no_stack_overflow(self):
        tree = var('x')
        for _ in range(50):
            tree = unary(Op.NEG, tree)
        result = _eval_float(tree, {'x': 1.0})
        assert result == 1.0  # 50 negations (even) = positive

    def test_copy_independence(self):
        tree = binary(Op.ADD, const(1), const(2))
        cp = tree.copy()
        cp.children[0].value = 99
        assert tree.children[0].value == 1

    def test_srconfig_defaults(self):
        cfg = SRConfig()
        assert cfg.population_size == 300
        assert cfg.optimize_constants is True
        assert len(cfg.unary_ops) > 0
        assert len(cfg.binary_ops) > 0

    def test_srresult_fields(self):
        cfg = SRConfig(population_size=20, max_generations=3)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        result = sr.fit([{'x': 1.0}, {'x': 2.0}], [1.0, 4.0])
        assert hasattr(result, 'best_expr')
        assert hasattr(result, 'best_fitness')
        assert hasattr(result, 'generations_run')
        assert hasattr(result, 'fitness_history')
        assert hasattr(result, 'converged')
        assert hasattr(result, 'hall_of_fame')

    def test_optimize_result_fields(self):
        tree = binary(Op.MUL, const(1.0), var('x'))
        opt = ConstantOptimizer()
        result = opt.optimize(tree, [{'x': 1.0}], [2.0], ['x'])
        assert hasattr(result, 'tree')
        assert hasattr(result, 'initial_loss')
        assert hasattr(result, 'final_loss')
        assert hasattr(result, 'iterations')


# ===================================================================
# 12. Integration tests -- full pipeline
# ===================================================================

class TestIntegration:
    def test_full_pipeline_quadratic(self):
        """Full pipeline: generate data, run SR, simplify, evaluate."""
        data_x, data_y = make_dataset(lambda x: x**2, ['x'],
                                       n_samples=30, x_range=(-3, 3), seed=42)
        cfg = SRConfig(population_size=200, max_generations=30,
                       optimize_constants=True, const_opt_frequency=5)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        result = sr.fit(data_x, data_y)

        # Simplify
        s = Simplifier()
        simplified = s.simplify(result.best_expr)

        # Evaluate on new data
        test_x = [{'x': 1.5}, {'x': -2.0}]
        for xi in test_x:
            pred = _eval_float(simplified, xi)
            assert math.isfinite(pred)

    def test_full_pipeline_with_ad_refinement(self):
        """Full pipeline: SR + AD constant optimization."""
        data_x, data_y = make_dataset(lambda x: 3*x + 5, ['x'],
                                       n_samples=30, x_range=(-5, 5), seed=42)
        cfg = SRConfig(population_size=150, max_generations=20,
                       optimize_constants=True, const_opt_frequency=3)
        sr = SymbolicRegressor(['x'], config=cfg, seed=42)
        result = sr.fit(data_x, data_y)
        assert result.best_fitness < 50.0  # Reasonable fit

    def test_multiobjective_pipeline(self):
        data_x, data_y = make_dataset(lambda x: x**2 + x, ['x'],
                                       n_samples=30, x_range=(-3, 3), seed=42)
        cfg = SRConfig(population_size=100, max_generations=15)
        mor = MultiObjectiveRegressor(['x'], config=cfg, seed=42)
        front = mor.fit(data_x, data_y)
        assert len(front) > 0
        # All solutions should be evaluatable
        for sol in front:
            val = _eval_float(sol.expr, {'x': 1.0})
            assert math.isfinite(val)

    def test_feature_selection_pipeline(self):
        # y = x^2, z is irrelevant
        data_x = [{'x': float(i), 'z': float(i) * 100} for i in range(-5, 6)]
        data_y = [float(i)**2 for i in range(-5, 6)]
        cfg = SRConfig(population_size=100, max_generations=15)
        fs = FeatureSelector(['x', 'z'], config=cfg, seed=42)
        result = fs.select(data_x, data_y)
        assert 'selected' in result
        assert isinstance(result['best_expr'], ExprNode)

    def test_constant_optimization_then_simplify(self):
        """Optimize constants, then simplify."""
        tree = binary(Op.ADD,
                       binary(Op.MUL, const(0.5), var('x')),
                       binary(Op.MUL, const(0), var('x')))
        data_x = [{'x': float(i)} for i in range(1, 6)]
        data_y = [2.0 * float(i) for i in range(1, 6)]

        opt = ConstantOptimizer(lr=0.1, max_iter=100)
        result = opt.optimize(tree, data_x, data_y, ['x'])

        s = Simplifier()
        simplified = s.simplify(result.tree)
        # The 0*x term should simplify away
        assert simplified.size() <= result.tree.size()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
