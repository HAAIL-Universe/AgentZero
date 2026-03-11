"""
C151: Symbolic Regression -- Composing C012 (Code Evolver) + C128 (Automatic Differentiation)

Genetic programming discovers expression STRUCTURE while gradient-based
optimization refines numeric CONSTANTS. This separation of concerns lets
evolution focus on topology and AD handle parameter tuning.

6 components:
  1. ExprTree     -- Expression tree with AD-compatible evaluation
  2. ConstantOptimizer -- Gradient-based constant refinement using C128
  3. SymbolicRegressor -- Full GP + AD symbolic regression engine
  4. Simplifier   -- Algebraic simplification of expression trees
  5. MultiObjectiveRegressor -- Pareto front of accuracy vs complexity
  6. FeatureSelector -- Automatic input variable selection
"""

import math
import random
import sys
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

# Import C128 Automatic Differentiation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))
from autodiff import Var, var_sin, var_cos, var_exp, var_log, var_sqrt, var_tanh, Dual, dual_sin, dual_cos, dual_exp, dual_log, dual_sqrt, dual_tanh


# ---------------------------------------------------------------------------
# 1. ExprTree -- Expression trees with AD-compatible evaluation
# ---------------------------------------------------------------------------

class Op(Enum):
    CONST = auto()
    VAR = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()   # protected division
    POW = auto()   # protected power
    NEG = auto()
    ABS = auto()
    SIN = auto()
    COS = auto()
    EXP = auto()   # protected exp
    LOG = auto()   # protected log
    SQRT = auto()  # protected sqrt
    TANH = auto()
    SQUARE = auto()
    CUBE = auto()


UNARY_OPS = [Op.NEG, Op.ABS, Op.SIN, Op.COS, Op.EXP, Op.LOG, Op.SQRT, Op.TANH, Op.SQUARE, Op.CUBE]
BINARY_OPS = [Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.POW]


class ExprNode:
    """Expression tree node."""

    __slots__ = ['op', 'value', 'var_name', 'children', '_ad_var']

    def __init__(self, op: Op, value: float = 0.0, var_name: str = '',
                 children: list = None):
        self.op = op
        self.value = value
        self.var_name = var_name
        self.children = children or []

    def copy(self) -> 'ExprNode':
        return ExprNode(
            self.op, self.value, self.var_name,
            [c.copy() for c in self.children]
        )

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def constants(self) -> list['ExprNode']:
        """Collect all CONST nodes (for gradient optimization)."""
        result = []
        if self.op == Op.CONST:
            result.append(self)
        for c in self.children:
            result.extend(c.constants())
        return result

    def _format(self) -> str:
        if self.op == Op.CONST:
            v = self.value
            if v == int(v) and abs(v) < 1e10:
                return str(int(v))
            return f"{v:.4g}"
        if self.op == Op.VAR:
            return self.var_name
        if self.op == Op.NEG:
            return f"(-{self.children[0]._format()})"
        if self.op == Op.ABS:
            return f"abs({self.children[0]._format()})"
        if self.op == Op.SIN:
            return f"sin({self.children[0]._format()})"
        if self.op == Op.COS:
            return f"cos({self.children[0]._format()})"
        if self.op == Op.EXP:
            return f"exp({self.children[0]._format()})"
        if self.op == Op.LOG:
            return f"log({self.children[0]._format()})"
        if self.op == Op.SQRT:
            return f"sqrt({self.children[0]._format()})"
        if self.op == Op.TANH:
            return f"tanh({self.children[0]._format()})"
        if self.op == Op.SQUARE:
            return f"({self.children[0]._format()})^2"
        if self.op == Op.CUBE:
            return f"({self.children[0]._format()})^3"
        if self.op == Op.ADD:
            return f"({self.children[0]._format()} + {self.children[1]._format()})"
        if self.op == Op.SUB:
            return f"({self.children[0]._format()} - {self.children[1]._format()})"
        if self.op == Op.MUL:
            return f"({self.children[0]._format()} * {self.children[1]._format()})"
        if self.op == Op.DIV:
            return f"({self.children[0]._format()} / {self.children[1]._format()})"
        if self.op == Op.POW:
            return f"({self.children[0]._format()} ** {self.children[1]._format()})"
        return "?"

    def __repr__(self):
        return self._format()


def _safe_div(a, b):
    """Protected division -- returns 1.0 for near-zero denominator."""
    if isinstance(b, (Var, Dual)):
        # For AD types, check the value
        bv = b.val if isinstance(b, Var) else b.val
        if abs(bv) < 1e-10:
            return a * 0.0 + 1.0  # Keep in computation graph
        return a / b
    if abs(b) < 1e-10:
        return 1.0
    return a / b


def _safe_exp(x):
    """Protected exp -- clamps input to avoid overflow."""
    if isinstance(x, Var):
        clamped_val = max(-20.0, min(20.0, x.val))
        if clamped_val != x.val:
            return var_exp(Var(clamped_val))
        return var_exp(x)
    if isinstance(x, Dual):
        clamped_val = max(-20.0, min(20.0, x.val))
        if clamped_val != x.val:
            return dual_exp(Dual(clamped_val, 0.0))
        return dual_exp(x)
    return math.exp(max(-20.0, min(20.0, x)))


def _safe_log(x):
    """Protected log -- returns 0.0 for non-positive inputs."""
    if isinstance(x, Var):
        if x.val <= 1e-10:
            return Var(0.0)
        return var_log(x)
    if isinstance(x, Dual):
        if x.val <= 1e-10:
            return Dual(0.0, 0.0)
        return dual_log(x)
    if x <= 1e-10:
        return 0.0
    return math.log(x)


def _safe_sqrt(x):
    """Protected sqrt -- returns 0.0 for negative inputs."""
    if isinstance(x, Var):
        if x.val < 0:
            return Var(0.0)
        return var_sqrt(x)
    if isinstance(x, Dual):
        if x.val < 0:
            return Dual(0.0, 0.0)
        return dual_sqrt(x)
    if x < 0:
        return 0.0
    return math.sqrt(x)


def _safe_pow(a, b):
    """Protected power -- handles edge cases."""
    if isinstance(a, (Var, Dual)):
        av = a.val
    else:
        av = a
    if isinstance(b, (Var, Dual)):
        bv = b.val
    else:
        bv = b

    # Clamp exponent
    bv_clamped = max(-5.0, min(5.0, bv))

    if abs(av) < 1e-10:
        if bv_clamped <= 0:
            return 1.0 if not isinstance(a, (Var, Dual)) else a * 0.0 + 1.0
        return 0.0 if not isinstance(a, (Var, Dual)) else a * 0.0

    if av < 0:
        # Negative base: use abs, integer powers only
        int_b = round(bv_clamped)
        result = abs(av) ** abs(int_b)
        if result > 1e15:
            return 1.0 if not isinstance(a, (Var, Dual)) else a * 0.0 + 1.0
        if int_b < 0:
            result = 1.0 / result if result > 1e-10 else 1.0
        if int(int_b) % 2 == 1:
            result = -result
        return result if not isinstance(a, (Var, Dual)) else a * 0.0 + result

    # Positive base
    try:
        result = av ** bv_clamped
        if not math.isfinite(result) or abs(result) > 1e15:
            return 1.0 if not isinstance(a, (Var, Dual)) else a * 0.0 + 1.0
        if isinstance(a, Var) and isinstance(b, Var):
            return a ** Var(bv_clamped)
        if isinstance(a, Var):
            return a ** bv_clamped
        if isinstance(b, Var):
            return Var(av) ** b
        if isinstance(a, Dual):
            return a ** bv_clamped
        return result
    except (OverflowError, ValueError):
        return 1.0


def evaluate(node: ExprNode, env: dict) -> Any:
    """Evaluate expression tree. env values can be float, Var, or Dual."""
    if node.op == Op.CONST:
        v = node.value
        # Check if any env value is Var/Dual to match types
        sample = next(iter(env.values()), None) if env else None
        if isinstance(sample, Var):
            return Var(v)
        if isinstance(sample, Dual):
            return Dual(v, 0.0)
        return v
    if node.op == Op.VAR:
        return env[node.var_name]

    # Evaluate children
    if len(node.children) == 1:
        a = evaluate(node.children[0], env)
        if node.op == Op.NEG:
            return -a
        if node.op == Op.ABS:
            return abs(a)
        if node.op == Op.SIN:
            if isinstance(a, Var): return var_sin(a)
            if isinstance(a, Dual): return dual_sin(a)
            return math.sin(a)
        if node.op == Op.COS:
            if isinstance(a, Var): return var_cos(a)
            if isinstance(a, Dual): return dual_cos(a)
            return math.cos(a)
        if node.op == Op.EXP:
            return _safe_exp(a)
        if node.op == Op.LOG:
            return _safe_log(a)
        if node.op == Op.SQRT:
            return _safe_sqrt(a)
        if node.op == Op.TANH:
            if isinstance(a, Var): return var_tanh(a)
            if isinstance(a, Dual): return dual_tanh(a)
            return math.tanh(a)
        if node.op == Op.SQUARE:
            return a * a
        if node.op == Op.CUBE:
            return a * a * a
    elif len(node.children) == 2:
        a = evaluate(node.children[0], env)
        b = evaluate(node.children[1], env)
        if node.op == Op.ADD:
            return a + b
        if node.op == Op.SUB:
            return a - b
        if node.op == Op.MUL:
            return a * b
        if node.op == Op.DIV:
            return _safe_div(a, b)
        if node.op == Op.POW:
            return _safe_pow(a, b)

    return 0.0


def _eval_float(node: ExprNode, env: dict) -> float:
    """Evaluate to plain float (no AD)."""
    result = evaluate(node, env)
    if isinstance(result, (Var, Dual)):
        return result.val
    return float(result)


# ---------------------------------------------------------------------------
# Tree construction helpers
# ---------------------------------------------------------------------------

def const(v: float) -> ExprNode:
    return ExprNode(Op.CONST, value=v)

def var(name: str) -> ExprNode:
    return ExprNode(Op.VAR, var_name=name)

def unary(op: Op, child: ExprNode) -> ExprNode:
    return ExprNode(op, children=[child])

def binary(op: Op, left: ExprNode, right: ExprNode) -> ExprNode:
    return ExprNode(op, children=[left, right])


# ---------------------------------------------------------------------------
# Tree generation
# ---------------------------------------------------------------------------

def random_tree(var_names: list, max_depth: int = 4, const_range=(-5.0, 5.0),
                rng: random.Random = None) -> ExprNode:
    """Generate a random expression tree (grow method)."""
    rng = rng or random.Random()

    if max_depth <= 0:
        # Terminal
        if rng.random() < 0.5 and var_names:
            return var(rng.choice(var_names))
        return const(round(rng.uniform(*const_range), 2))

    # Choose: terminal, unary, or binary
    r = rng.random()
    if r < 0.3 and max_depth > 0:
        # Unary
        op = rng.choice(UNARY_OPS)
        child = random_tree(var_names, max_depth - 1, const_range, rng)
        return unary(op, child)
    elif r < 0.7 and max_depth > 0:
        # Binary
        op = rng.choice(BINARY_OPS)
        left = random_tree(var_names, max_depth - 1, const_range, rng)
        right = random_tree(var_names, max_depth - 1, const_range, rng)
        return binary(op, left, right)
    else:
        # Terminal
        if rng.random() < 0.5 and var_names:
            return var(rng.choice(var_names))
        return const(round(rng.uniform(*const_range), 2))


def full_tree(var_names: list, depth: int = 3, const_range=(-5.0, 5.0),
              rng: random.Random = None) -> ExprNode:
    """Generate a full tree (all branches reach max depth)."""
    rng = rng or random.Random()

    if depth <= 0:
        if rng.random() < 0.5 and var_names:
            return var(rng.choice(var_names))
        return const(round(rng.uniform(*const_range), 2))

    if rng.random() < 0.35:
        op = rng.choice(UNARY_OPS)
        return unary(op, full_tree(var_names, depth - 1, const_range, rng))
    else:
        op = rng.choice(BINARY_OPS)
        left = full_tree(var_names, depth - 1, const_range, rng)
        right = full_tree(var_names, depth - 1, const_range, rng)
        return binary(op, left, right)


# ---------------------------------------------------------------------------
# 2. ConstantOptimizer -- Gradient-based constant refinement via C128
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    tree: ExprNode
    initial_loss: float
    final_loss: float
    iterations: int


class ConstantOptimizer:
    """Optimize numeric constants in expression trees using AD (C128)."""

    def __init__(self, lr: float = 0.01, max_iter: int = 50, tol: float = 1e-8):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, tree: ExprNode, data_x: list, data_y: list,
                 var_names: list) -> OptimizeResult:
        """Optimize constants in tree to minimize MSE on data.

        data_x: list of dicts {var_name: float_value}
        data_y: list of float target values
        """
        tree = tree.copy()
        const_nodes = tree.constants()
        if not const_nodes:
            loss = self._mse_float(tree, data_x, data_y, var_names)
            return OptimizeResult(tree, loss, loss, 0)

        initial_loss = self._mse_float(tree, data_x, data_y, var_names)
        best_loss = initial_loss
        best_values = [c.value for c in const_nodes]

        for iteration in range(self.max_iter):
            # Create Var objects for constants
            const_vars = [Var(c.value) for c in const_nodes]

            # Compute loss as sum of squared errors using AD
            total_loss = Var(0.0)
            for xi, yi in zip(data_x, data_y):
                # Set const node values to Var objects temporarily
                for cn, cv in zip(const_nodes, const_vars):
                    cn._ad_var = cv

                env = {}
                for vn in var_names:
                    env[vn] = Var(xi[vn])
                pred = self._eval_with_ad(tree, env)
                diff = pred - Var(yi)
                total_loss = total_loss + diff * diff

            n = len(data_x)
            total_loss = total_loss * Var(1.0 / n)

            # Backward pass
            total_loss.backward()

            # Gradient descent update
            loss_val = total_loss.val
            if math.isnan(loss_val) or math.isinf(loss_val):
                # Restore best
                for cn, bv in zip(const_nodes, best_values):
                    cn.value = bv
                break

            if loss_val < best_loss:
                best_loss = loss_val
                best_values = [c.value for c in const_nodes]

            max_grad = 0.0
            for cn, cv in zip(const_nodes, const_vars):
                grad = cv.grad
                if math.isnan(grad) or math.isinf(grad):
                    grad = 0.0
                # Clip gradient
                grad = max(-10.0, min(10.0, grad))
                max_grad = max(max_grad, abs(grad))
                cn.value -= self.lr * grad

            # Clean up ad vars
            for cn in const_nodes:
                if hasattr(cn, '_ad_var'):
                    delattr(cn, '_ad_var')

            if max_grad < self.tol:
                break

        # Ensure best values
        final_loss = self._mse_float(tree, data_x, data_y, var_names)
        if final_loss > best_loss:
            for cn, bv in zip(const_nodes, best_values):
                cn.value = bv
            final_loss = best_loss

        return OptimizeResult(tree, initial_loss, final_loss, iteration + 1 if const_nodes else 0)

    def _eval_with_ad(self, node: ExprNode, env: dict) -> Var:
        """Evaluate using Var objects, routing constants through their _ad_var."""
        if node.op == Op.CONST:
            if hasattr(node, '_ad_var'):
                return node._ad_var
            return Var(node.value)
        if node.op == Op.VAR:
            return env[node.var_name]

        if len(node.children) == 1:
            a = self._eval_with_ad(node.children[0], env)
            if node.op == Op.NEG: return -a
            if node.op == Op.ABS: return abs(a)
            if node.op == Op.SIN: return var_sin(a)
            if node.op == Op.COS: return var_cos(a)
            if node.op == Op.EXP: return _safe_exp(a)
            if node.op == Op.LOG: return _safe_log(a)
            if node.op == Op.SQRT: return _safe_sqrt(a)
            if node.op == Op.TANH: return var_tanh(a)
            if node.op == Op.SQUARE: return a * a
            if node.op == Op.CUBE: return a * a * a
        elif len(node.children) == 2:
            a = self._eval_with_ad(node.children[0], env)
            b = self._eval_with_ad(node.children[1], env)
            if node.op == Op.ADD: return a + b
            if node.op == Op.SUB: return a - b
            if node.op == Op.MUL: return a * b
            if node.op == Op.DIV: return _safe_div(a, b)
            if node.op == Op.POW: return _safe_pow(a, b)

        return Var(0.0)

    def _mse_float(self, tree, data_x, data_y, var_names):
        """MSE using plain floats."""
        total = 0.0
        for xi, yi in zip(data_x, data_y):
            env = {vn: xi[vn] for vn in var_names}
            pred = _eval_float(tree, env)
            if math.isnan(pred) or math.isinf(pred):
                return 1e15
            total += (pred - yi) ** 2
        return total / len(data_x)


# ---------------------------------------------------------------------------
# 3. SymbolicRegressor -- Full GP + AD symbolic regression
# ---------------------------------------------------------------------------

@dataclass
class SRConfig:
    """Symbolic regression configuration."""
    population_size: int = 300
    max_generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    reproduction_rate: float = 0.1
    max_tree_depth: int = 8
    initial_max_depth: int = 4
    const_range: tuple = (-5.0, 5.0)
    parsimony_weight: float = 0.001
    elitism: int = 5
    fitness_threshold: float = 1e-6
    stagnation_limit: int = 20
    # AD constant optimization
    optimize_constants: bool = True
    const_opt_lr: float = 0.01
    const_opt_iterations: int = 30
    const_opt_frequency: int = 5  # Optimize every N generations
    # Allowed ops
    unary_ops: list = None
    binary_ops: list = None

    def __post_init__(self):
        if self.unary_ops is None:
            self.unary_ops = [Op.NEG, Op.SIN, Op.COS, Op.SQUARE, Op.SQRT, Op.EXP, Op.LOG]
        if self.binary_ops is None:
            self.binary_ops = [Op.ADD, Op.SUB, Op.MUL, Op.DIV]


@dataclass
class SRResult:
    """Symbolic regression result."""
    best_expr: ExprNode
    best_fitness: float
    generations_run: int
    fitness_history: list
    converged: bool
    hall_of_fame: list  # top-k expressions


class SymbolicRegressor:
    """Genetic programming + AD constant optimization for symbolic regression."""

    def __init__(self, var_names: list, config: SRConfig = None, seed: int = None):
        self.var_names = var_names
        self.config = config or SRConfig()
        self.rng = random.Random(seed)
        self.population = []  # list of (tree, fitness)
        self.generation = 0
        self.best = None  # (tree, fitness)
        self.fitness_history = []
        self.hall_of_fame = []  # (tree, fitness) sorted by fitness
        self.hof_size = 10
        self.stagnation_counter = 0
        self.optimizer = ConstantOptimizer(
            lr=self.config.const_opt_lr,
            max_iter=self.config.const_opt_iterations
        )

    def fit(self, data_x: list, data_y: list) -> SRResult:
        """Run symbolic regression.

        data_x: list of dicts {var_name: value}
        data_y: list of target values
        """
        self.data_x = data_x
        self.data_y = data_y

        # Initialize population
        self._initialize()

        converged = False
        for gen in range(self.config.max_generations):
            self.generation = gen

            # Constant optimization step
            if (self.config.optimize_constants and
                gen > 0 and gen % self.config.const_opt_frequency == 0):
                self._optimize_constants()

            # Evolve one generation
            best_fitness = self._step()
            self.fitness_history.append(best_fitness)

            # Check convergence
            if best_fitness <= self.config.fitness_threshold:
                converged = True
                break

            # Stagnation detection
            if len(self.fitness_history) >= 2:
                improvement = self.fitness_history[-2] - self.fitness_history[-1]
                if improvement < 1e-10:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

            if self.stagnation_counter >= self.config.stagnation_limit:
                self._inject_diversity()
                self.stagnation_counter = 0

        # Final constant optimization
        if self.config.optimize_constants and self.best:
            result = self.optimizer.optimize(
                self.best[0], data_x, data_y, self.var_names
            )
            if result.final_loss < self.best[1]:
                self.best = (result.tree, result.final_loss)
                self._update_hof(result.tree, result.final_loss)

        return SRResult(
            best_expr=self.best[0].copy() if self.best else const(0),
            best_fitness=self.best[1] if self.best else float('inf'),
            generations_run=self.generation + 1,
            fitness_history=self.fitness_history,
            converged=converged,
            hall_of_fame=[(t.copy(), f) for t, f in self.hall_of_fame[:self.hof_size]]
        )

    def _initialize(self):
        """Ramped half-and-half initialization."""
        self.population = []
        pop_size = self.config.population_size
        max_d = self.config.initial_max_depth

        for i in range(pop_size):
            depth = 1 + (i % max_d)
            if i % 2 == 0:
                tree = random_tree(self.var_names, depth, self.config.const_range, self.rng)
            else:
                tree = full_tree(self.var_names, depth, self.config.const_range, self.rng)
            fitness = self._fitness(tree)
            self.population.append((tree, fitness))
            self._update_hof(tree, fitness)

        self.population.sort(key=lambda x: x[1])
        self.best = self.population[0]

    def _fitness(self, tree: ExprNode) -> float:
        """MSE + parsimony penalty."""
        mse = 0.0
        for xi, yi in zip(self.data_x, self.data_y):
            env = {vn: xi[vn] for vn in self.var_names}
            try:
                pred = _eval_float(tree, env)
            except Exception:
                return 1e15
            if math.isnan(pred) or math.isinf(pred):
                return 1e15
            mse += (pred - yi) ** 2
        mse /= len(self.data_x)
        return mse + self.config.parsimony_weight * tree.size()

    def _step(self) -> float:
        """One generation of evolution."""
        new_pop = []
        cfg = self.config

        # Elitism
        self.population.sort(key=lambda x: x[1])
        for i in range(min(cfg.elitism, len(self.population))):
            new_pop.append(self.population[i])

        # Fill rest
        while len(new_pop) < cfg.population_size:
            r = self.rng.random()
            if r < cfg.crossover_rate:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._crossover(p1, p2)
                c1 = self._enforce_depth(c1)
                c2 = self._enforce_depth(c2)
                f1 = self._fitness(c1)
                f2 = self._fitness(c2)
                new_pop.append((c1, f1))
                self._update_hof(c1, f1)
                if len(new_pop) < cfg.population_size:
                    new_pop.append((c2, f2))
                    self._update_hof(c2, f2)
            elif r < cfg.crossover_rate + cfg.mutation_rate:
                parent = self._tournament_select()
                child = self._mutate(parent)
                child = self._enforce_depth(child)
                f = self._fitness(child)
                new_pop.append((child, f))
                self._update_hof(child, f)
            else:
                parent = self._tournament_select()
                new_pop.append((parent.copy(), self._fitness(parent)))

        self.population = new_pop[:cfg.population_size]
        self.population.sort(key=lambda x: x[1])
        if self.population[0][1] < self.best[1]:
            self.best = (self.population[0][0].copy(), self.population[0][1])

        return self.best[1]

    def _tournament_select(self) -> ExprNode:
        """Tournament selection."""
        candidates = self.rng.sample(self.population, min(self.config.tournament_size, len(self.population)))
        winner = min(candidates, key=lambda x: x[1])
        return winner[0].copy()

    def _crossover(self, p1: ExprNode, p2: ExprNode) -> tuple:
        """Subtree crossover."""
        c1 = p1.copy()
        c2 = p2.copy()

        # Collect all nodes with parent references
        nodes1 = self._collect_nodes(c1)
        nodes2 = self._collect_nodes(c2)

        if len(nodes1) < 2 or len(nodes2) < 2:
            return c1, c2

        # Pick crossover points (skip root)
        idx1 = self.rng.randint(1, len(nodes1) - 1)
        idx2 = self.rng.randint(1, len(nodes2) - 1)

        n1_parent, n1_child_idx = nodes1[idx1]
        n2_parent, n2_child_idx = nodes2[idx2]

        # Swap subtrees
        n1_parent.children[n1_child_idx], n2_parent.children[n2_child_idx] = \
            n2_parent.children[n2_child_idx], n1_parent.children[n1_child_idx]

        return c1, c2

    def _collect_nodes(self, tree: ExprNode) -> list:
        """Collect (parent, child_index) pairs for all nodes."""
        result = [(None, 0)]  # root
        stack = [tree]
        while stack:
            node = stack.pop()
            for i, child in enumerate(node.children):
                result.append((node, i))
                stack.append(child)
        return result

    def _mutate(self, tree: ExprNode) -> ExprNode:
        """Mutation: point, subtree, or hoist."""
        tree = tree.copy()
        r = self.rng.random()

        if r < 0.4:
            # Point mutation
            return self._mutate_point(tree)
        elif r < 0.8:
            # Subtree mutation
            return self._mutate_subtree(tree)
        else:
            # Hoist mutation
            return self._mutate_hoist(tree)

    def _mutate_point(self, tree: ExprNode) -> ExprNode:
        """Change a single node's operation or value."""
        nodes = self._collect_all_nodes(tree)
        if not nodes:
            return tree
        node = self.rng.choice(nodes)

        if node.op == Op.CONST:
            # Perturb constant
            node.value += self.rng.gauss(0, 1.0)
        elif node.op == Op.VAR:
            if self.var_names:
                node.var_name = self.rng.choice(self.var_names)
        elif node.op in UNARY_OPS:
            available = [op for op in self.config.unary_ops if op != node.op]
            if available:
                node.op = self.rng.choice(available)
        elif node.op in BINARY_OPS:
            available = [op for op in self.config.binary_ops if op != node.op]
            if available:
                node.op = self.rng.choice(available)

        return tree

    def _mutate_subtree(self, tree: ExprNode) -> ExprNode:
        """Replace a random subtree."""
        nodes = self._collect_nodes(tree)
        if len(nodes) < 2:
            return random_tree(self.var_names, 2, self.config.const_range, self.rng)

        idx = self.rng.randint(1, len(nodes) - 1)
        parent, child_idx = nodes[idx]
        new_subtree = random_tree(self.var_names, 3, self.config.const_range, self.rng)
        parent.children[child_idx] = new_subtree
        return tree

    def _mutate_hoist(self, tree: ExprNode) -> ExprNode:
        """Replace tree with a subtree (simplification)."""
        if not tree.children:
            return tree
        all_nodes = self._collect_all_nodes(tree)
        internal = [n for n in all_nodes if n.children]
        if not internal:
            return tree
        return self.rng.choice(internal).copy()

    def _collect_all_nodes(self, tree: ExprNode) -> list:
        """Flat list of all nodes."""
        result = [tree]
        for c in tree.children:
            result.extend(self._collect_all_nodes(c))
        return result

    def _enforce_depth(self, tree: ExprNode) -> ExprNode:
        """Trim tree if it exceeds max depth."""
        if tree.depth() <= self.config.max_tree_depth:
            return tree
        return self._trim(tree, self.config.max_tree_depth)

    def _trim(self, node: ExprNode, remaining: int) -> ExprNode:
        if remaining <= 0 or not node.children:
            # Replace with terminal
            if self.rng.random() < 0.5 and self.var_names:
                return var(self.rng.choice(self.var_names))
            return const(round(self.rng.uniform(*self.config.const_range), 2))
        node.children = [self._trim(c, remaining - 1) for c in node.children]
        return node

    def _optimize_constants(self):
        """Optimize constants for top individuals."""
        n = min(10, len(self.population))
        for i in range(n):
            tree, fitness = self.population[i]
            if not tree.constants():
                continue
            result = self.optimizer.optimize(tree, self.data_x, self.data_y, self.var_names)
            if result.final_loss < fitness:
                self.population[i] = (result.tree, result.final_loss + self.config.parsimony_weight * result.tree.size())
                self._update_hof(result.tree, result.final_loss)
        self.population.sort(key=lambda x: x[1])
        if self.population[0][1] < self.best[1]:
            self.best = (self.population[0][0].copy(), self.population[0][1])

    def _inject_diversity(self):
        """Replace worst individuals with fresh random trees."""
        n_replace = self.config.population_size // 4
        for i in range(n_replace):
            idx = len(self.population) - 1 - i
            if idx <= self.config.elitism:
                break
            tree = random_tree(self.var_names, self.config.initial_max_depth, self.config.const_range, self.rng)
            fitness = self._fitness(tree)
            self.population[idx] = (tree, fitness)

    def _update_hof(self, tree: ExprNode, fitness: float):
        """Update hall of fame."""
        if fitness >= 1e14:
            return
        self.hall_of_fame.append((tree.copy(), fitness))
        self.hall_of_fame.sort(key=lambda x: x[1])
        self.hall_of_fame = self.hall_of_fame[:self.hof_size]


# ---------------------------------------------------------------------------
# 4. Simplifier -- Algebraic simplification
# ---------------------------------------------------------------------------

class Simplifier:
    """Simplify expression trees algebraically."""

    def simplify(self, tree: ExprNode) -> ExprNode:
        """Apply simplification rules until fixpoint."""
        tree = tree.copy()
        for _ in range(20):
            new_tree = self._simplify_once(tree)
            if repr(new_tree) == repr(tree):
                break
            tree = new_tree
        return tree

    def _simplify_once(self, node: ExprNode) -> ExprNode:
        # Simplify children first
        node = ExprNode(node.op, node.value, node.var_name,
                        [self._simplify_once(c) for c in node.children])

        # Constant folding
        if node.children and all(c.op == Op.CONST for c in node.children):
            try:
                env = {}
                result = _eval_float(node, env)
                if math.isfinite(result):
                    return const(result)
            except Exception:
                pass

        # Identity rules
        if node.op == Op.ADD:
            l, r = node.children
            if l.op == Op.CONST and l.value == 0:
                return r
            if r.op == Op.CONST and r.value == 0:
                return l
        elif node.op == Op.SUB:
            l, r = node.children
            if r.op == Op.CONST and r.value == 0:
                return l
            if self._equal(l, r):
                return const(0)
        elif node.op == Op.MUL:
            l, r = node.children
            if l.op == Op.CONST and l.value == 0:
                return const(0)
            if r.op == Op.CONST and r.value == 0:
                return const(0)
            if l.op == Op.CONST and l.value == 1:
                return r
            if r.op == Op.CONST and r.value == 1:
                return l
        elif node.op == Op.DIV:
            l, r = node.children
            if l.op == Op.CONST and l.value == 0:
                return const(0)
            if r.op == Op.CONST and r.value == 1:
                return l
            if self._equal(l, r):
                return const(1)
        elif node.op == Op.POW:
            l, r = node.children
            if r.op == Op.CONST and r.value == 0:
                return const(1)
            if r.op == Op.CONST and r.value == 1:
                return l
        elif node.op == Op.NEG:
            child = node.children[0]
            if child.op == Op.NEG:
                return child.children[0]
            if child.op == Op.CONST:
                return const(-child.value)
        elif node.op == Op.SQUARE:
            child = node.children[0]
            if child.op == Op.CONST:
                return const(child.value ** 2)
        elif node.op == Op.CUBE:
            child = node.children[0]
            if child.op == Op.CONST:
                return const(child.value ** 3)

        return node

    def _equal(self, a: ExprNode, b: ExprNode) -> bool:
        """Structural equality."""
        if a.op != b.op:
            return False
        if a.op == Op.CONST:
            return abs(a.value - b.value) < 1e-10
        if a.op == Op.VAR:
            return a.var_name == b.var_name
        if len(a.children) != len(b.children):
            return False
        return all(self._equal(ac, bc) for ac, bc in zip(a.children, b.children))


# ---------------------------------------------------------------------------
# 5. MultiObjectiveRegressor -- Pareto front (accuracy vs complexity)
# ---------------------------------------------------------------------------

@dataclass
class ParetoSolution:
    """A solution on the Pareto front."""
    expr: ExprNode
    fitness: float    # MSE
    complexity: int   # tree size
    score: float      # combined score


class MultiObjectiveRegressor:
    """Find Pareto-optimal expressions trading off accuracy and complexity."""

    def __init__(self, var_names: list, config: SRConfig = None, seed: int = None):
        self.var_names = var_names
        self.config = config or SRConfig()
        self.seed = seed
        self.simplifier = Simplifier()

    def fit(self, data_x: list, data_y: list) -> list:
        """Run multi-objective regression, return Pareto front."""
        # Run standard SR
        sr = SymbolicRegressor(self.var_names, self.config, self.seed)
        result = sr.fit(data_x, data_y)

        # Collect candidates from hall of fame
        candidates = []
        for tree, fitness in result.hall_of_fame:
            simplified = self.simplifier.simplify(tree)
            mse = self._mse(simplified, data_x, data_y)
            candidates.append(ParetoSolution(
                expr=simplified,
                fitness=mse,
                complexity=simplified.size(),
                score=mse
            ))

        # Also add the best
        simplified_best = self.simplifier.simplify(result.best_expr)
        mse_best = self._mse(simplified_best, data_x, data_y)
        candidates.append(ParetoSolution(
            expr=simplified_best,
            fitness=mse_best,
            complexity=simplified_best.size(),
            score=mse_best
        ))

        # Extract Pareto front
        front = self._pareto_front(candidates)
        front.sort(key=lambda s: s.complexity)
        return front

    def _mse(self, tree, data_x, data_y):
        total = 0.0
        for xi, yi in zip(data_x, data_y):
            env = {vn: xi[vn] for vn in self.var_names}
            try:
                pred = _eval_float(tree, env)
            except Exception:
                return 1e15
            if math.isnan(pred) or math.isinf(pred):
                return 1e15
            total += (pred - yi) ** 2
        return total / len(data_x)

    def _pareto_front(self, solutions: list) -> list:
        """Extract non-dominated solutions (minimize fitness AND complexity)."""
        front = []
        for s in solutions:
            dominated = False
            for other in solutions:
                if other is s:
                    continue
                # other dominates s if better or equal on both, strictly better on at least one
                if (other.fitness <= s.fitness and other.complexity <= s.complexity and
                    (other.fitness < s.fitness or other.complexity < s.complexity)):
                    dominated = True
                    break
            if not dominated:
                # Check no duplicate in front
                dup = False
                for f in front:
                    if abs(f.fitness - s.fitness) < 1e-12 and f.complexity == s.complexity:
                        dup = True
                        break
                if not dup:
                    front.append(s)
        return front


# ---------------------------------------------------------------------------
# 6. FeatureSelector -- Automatic input variable selection
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Identify which input variables are most relevant via symbolic regression."""

    def __init__(self, var_names: list, config: SRConfig = None, seed: int = None):
        self.var_names = var_names
        self.config = config or SRConfig()
        self.seed = seed

    def select(self, data_x: list, data_y: list, max_features: int = None) -> dict:
        """Run SR and analyze which variables appear in best expressions.

        Returns dict with:
          - selected: list of variable names that appear in best expressions
          - importance: dict of var_name -> frequency in hall of fame
          - best_expr: best expression found
        """
        sr = SymbolicRegressor(self.var_names, self.config, self.seed)
        result = sr.fit(data_x, data_y)

        # Count variable usage across hall of fame
        importance = {vn: 0 for vn in self.var_names}
        total_exprs = len(result.hall_of_fame)
        if total_exprs == 0:
            total_exprs = 1

        for tree, _ in result.hall_of_fame:
            used = self._variables_used(tree)
            for vn in used:
                if vn in importance:
                    importance[vn] += 1

        # Normalize
        for vn in importance:
            importance[vn] /= total_exprs

        # Select top features
        sorted_vars = sorted(importance.items(), key=lambda x: -x[1])
        if max_features is None:
            selected = [vn for vn, score in sorted_vars if score > 0]
        else:
            selected = [vn for vn, _ in sorted_vars[:max_features]]

        return {
            'selected': selected,
            'importance': importance,
            'best_expr': result.best_expr,
            'best_fitness': result.best_fitness
        }

    def _variables_used(self, tree: ExprNode) -> set:
        """Collect all variable names used in tree."""
        result = set()
        if tree.op == Op.VAR:
            result.add(tree.var_name)
        for c in tree.children:
            result.update(self._variables_used(c))
        return result


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def make_dataset(func: Callable, var_names: list, n_samples: int = 50,
                 x_range: tuple = (-5.0, 5.0), seed: int = None) -> tuple:
    """Generate dataset from a target function.

    Returns (data_x, data_y) where data_x is list of dicts and data_y is list of floats.
    """
    rng = random.Random(seed)
    data_x = []
    data_y = []
    for _ in range(n_samples):
        point = {vn: rng.uniform(*x_range) for vn in var_names}
        data_x.append(point)
        try:
            y = func(**point)
            if math.isfinite(y):
                data_y.append(y)
            else:
                data_y.append(0.0)
        except Exception:
            data_y.append(0.0)
    return data_x, data_y


def symbolic_regression(func: Callable, var_names: list, n_samples: int = 50,
                        x_range: tuple = (-5.0, 5.0), seed: int = None,
                        config: SRConfig = None) -> SRResult:
    """One-call symbolic regression: generate data and find expression."""
    data_x, data_y = make_dataset(func, var_names, n_samples, x_range, seed)
    sr = SymbolicRegressor(var_names, config, seed)
    return sr.fit(data_x, data_y)
