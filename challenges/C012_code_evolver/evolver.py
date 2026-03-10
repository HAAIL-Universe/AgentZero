"""
C012: Code Evolver -- Genetic Programming System

Evolves programs (expression trees) to solve problems via natural selection.
Programs are represented as trees of operations, mutated and crossed over,
then selected by fitness against test cases.

This is what AgentZero does across sessions, encoded as a program:
variation -> evaluation -> selection -> repeat.
"""

import random
import math
import copy
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum, auto


# --- Node Types ---

class NodeType(Enum):
    CONST = auto()      # literal number
    VAR = auto()        # input variable (x, y, z...)
    UNARY = auto()      # one-child operation
    BINARY = auto()     # two-child operation
    TERNARY = auto()    # if-then-else


@dataclass
class Node:
    """A node in a program tree."""
    type: NodeType
    value: Any = None       # for CONST: number, VAR: var name, ops: function name
    children: list = field(default_factory=list)

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> 'Node':
        return copy.deepcopy(self)

    def __repr__(self):
        return format_node(self)


# --- Operations ---

UNARY_OPS = {
    'neg': lambda x: -x,
    'abs': lambda x: abs(x),
    'square': lambda x: x * x,
    'safe_sqrt': lambda x: math.sqrt(abs(x)),
    'safe_inv': lambda x: 1.0 / x if x != 0 else 1.0,
    'double': lambda x: x * 2,
    'half': lambda x: x / 2,
    'sin': lambda x: math.sin(x),
    'cos': lambda x: math.cos(x),
}

BINARY_OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'safe_div': lambda a, b: a / b if b != 0 else 1.0,
    'max': lambda a, b: max(a, b),
    'min': lambda a, b: min(a, b),
    'mod': lambda a, b: a % b if b != 0 else 0.0,
    'pow': lambda a, b: _safe_pow(a, b),
}

def _safe_pow(a, b):
    """Power that won't overflow."""
    try:
        b_clamped = max(-10, min(10, b))
        result = math.pow(abs(a), b_clamped)
        if a < 0 and int(b_clamped) == b_clamped and int(b_clamped) % 2 == 1:
            result = -result
        if math.isinf(result) or math.isnan(result):
            return 1.0
        return result
    except (OverflowError, ValueError):
        return 1.0

UNARY_NAMES = list(UNARY_OPS.keys())
BINARY_NAMES = list(BINARY_OPS.keys())


# --- Program Construction ---

def make_const(value: float) -> Node:
    return Node(NodeType.CONST, value=value)

def make_var(name: str) -> Node:
    return Node(NodeType.VAR, value=name)

def make_unary(op: str, child: Node) -> Node:
    return Node(NodeType.UNARY, value=op, children=[child])

def make_binary(op: str, left: Node, right: Node) -> Node:
    return Node(NodeType.BINARY, value=op, children=[left, right])

def make_ternary(cond: Node, then_branch: Node, else_branch: Node) -> Node:
    return Node(NodeType.TERNARY, value='if', children=[cond, then_branch, else_branch])


# --- Formatting ---

def format_node(node: Node) -> str:
    """Human-readable representation."""
    if node.type == NodeType.CONST:
        v = node.value
        if isinstance(v, float) and v == int(v) and abs(v) < 1e6:
            return str(int(v))
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    elif node.type == NodeType.VAR:
        return str(node.value)
    elif node.type == NodeType.UNARY:
        return f"{node.value}({format_node(node.children[0])})"
    elif node.type == NodeType.BINARY:
        return f"({format_node(node.children[0])} {node.value} {format_node(node.children[1])})"
    elif node.type == NodeType.TERNARY:
        return f"(if {format_node(node.children[0])} then {format_node(node.children[1])} else {format_node(node.children[2])})"
    return "?"


# --- Evaluation ---

def evaluate(node: Node, env: dict[str, float]) -> float:
    """Evaluate a program tree with given variable bindings."""
    if node.type == NodeType.CONST:
        return float(node.value)
    elif node.type == NodeType.VAR:
        return float(env.get(node.value, 0.0))
    elif node.type == NodeType.UNARY:
        child_val = evaluate(node.children[0], env)
        fn = UNARY_OPS.get(node.value)
        if fn is None:
            raise ValueError(f"Unknown unary op: {node.value}")
        try:
            result = fn(child_val)
            if math.isinf(result) or math.isnan(result):
                return 0.0
            return result
        except (OverflowError, ValueError, ZeroDivisionError):
            return 0.0
    elif node.type == NodeType.BINARY:
        left_val = evaluate(node.children[0], env)
        right_val = evaluate(node.children[1], env)
        fn = BINARY_OPS.get(node.value)
        if fn is None:
            raise ValueError(f"Unknown binary op: {node.value}")
        try:
            result = fn(left_val, right_val)
            if math.isinf(result) or math.isnan(result):
                return 0.0
            return result
        except (OverflowError, ValueError, ZeroDivisionError):
            return 0.0
    elif node.type == NodeType.TERNARY:
        cond_val = evaluate(node.children[0], env)
        if cond_val > 0:
            return evaluate(node.children[1], env)
        else:
            return evaluate(node.children[2], env)
    raise ValueError(f"Unknown node type: {node.type}")


# --- Random Tree Generation ---

def random_tree(var_names: list[str], max_depth: int = 4,
                const_range: tuple[float, float] = (-5.0, 5.0),
                rng: random.Random = None) -> Node:
    """Generate a random program tree."""
    if rng is None:
        rng = random.Random()
    return _grow(var_names, max_depth, const_range, rng, depth=0)


def _grow(var_names, max_depth, const_range, rng, depth):
    """Grow method: mix of full and grow."""
    # Force terminal at max depth (depth is 0-indexed, so >= max_depth-1)
    if depth >= max_depth - 1:
        return _random_terminal(var_names, const_range, rng)

    # Bias toward terminals as depth increases
    terminal_prob = 0.3 + 0.15 * depth
    if rng.random() < terminal_prob:
        return _random_terminal(var_names, const_range, rng)

    # Pick a function node
    choice = rng.random()
    if choice < 0.1 and max_depth - depth >= 3:
        # Ternary (rare)
        cond = _grow(var_names, max_depth, const_range, rng, depth + 1)
        then = _grow(var_names, max_depth, const_range, rng, depth + 1)
        els = _grow(var_names, max_depth, const_range, rng, depth + 1)
        return make_ternary(cond, then, els)
    elif choice < 0.45:
        # Unary
        op = rng.choice(UNARY_NAMES)
        child = _grow(var_names, max_depth, const_range, rng, depth + 1)
        return make_unary(op, child)
    else:
        # Binary
        op = rng.choice(BINARY_NAMES)
        left = _grow(var_names, max_depth, const_range, rng, depth + 1)
        right = _grow(var_names, max_depth, const_range, rng, depth + 1)
        return make_binary(op, left, right)


def _random_terminal(var_names, const_range, rng):
    if var_names and rng.random() < 0.6:
        return make_var(rng.choice(var_names))
    return make_const(round(rng.uniform(*const_range), 2))


# --- Tree Traversal Utilities ---

def all_nodes(node: Node) -> list[tuple[Node, list[int]]]:
    """Return all nodes with their path from root."""
    result = [(node, [])]
    for i, child in enumerate(node.children):
        for n, path in all_nodes(child):
            result.append((n, [i] + path))
    return result


def get_node_at(root: Node, path: list[int]) -> Node:
    """Get node at a given path."""
    current = root
    for idx in path:
        current = current.children[idx]
    return current


def replace_node_at(root: Node, path: list[int], replacement: Node) -> Node:
    """Replace node at path, returning new root."""
    root = root.copy()
    if not path:
        return replacement.copy()
    parent = root
    for idx in path[:-1]:
        parent = parent.children[idx]
    parent.children[path[-1]] = replacement.copy()
    return root


# --- Genetic Operators ---

def mutate_point(tree: Node, var_names: list[str],
                 const_range: tuple[float, float] = (-5.0, 5.0),
                 rng: random.Random = None) -> Node:
    """Mutate a single random node in place (same arity)."""
    if rng is None:
        rng = random.Random()
    tree = tree.copy()
    nodes = all_nodes(tree)
    node, path = rng.choice(nodes)
    target = get_node_at(tree, path) if path else tree

    if target.type == NodeType.CONST:
        # Perturb or replace constant
        if rng.random() < 0.5:
            target.value = round(target.value + rng.gauss(0, 1), 2)
        else:
            target.value = round(rng.uniform(*const_range), 2)
    elif target.type == NodeType.VAR:
        if var_names:
            target.value = rng.choice(var_names)
    elif target.type == NodeType.UNARY:
        target.value = rng.choice(UNARY_NAMES)
    elif target.type == NodeType.BINARY:
        target.value = rng.choice(BINARY_NAMES)
    # Ternary: nothing to mutate (the 'if' operator is fixed)
    return tree


def mutate_subtree(tree: Node, var_names: list[str], max_depth: int = 3,
                   const_range: tuple[float, float] = (-5.0, 5.0),
                   rng: random.Random = None) -> Node:
    """Replace a random subtree with a new random tree."""
    if rng is None:
        rng = random.Random()
    nodes = all_nodes(tree)
    _, path = rng.choice(nodes)
    new_subtree = random_tree(var_names, max_depth, const_range, rng)
    return replace_node_at(tree, path, new_subtree)


def mutate_hoist(tree: Node, rng: random.Random = None) -> Node:
    """Replace tree with one of its subtrees (simplification)."""
    if rng is None:
        rng = random.Random()
    nodes = all_nodes(tree)
    if len(nodes) <= 1:
        return tree.copy()
    # Pick a non-root node
    _, path = rng.choice(nodes[1:])
    return get_node_at(tree, path).copy()


def mutate(tree: Node, var_names: list[str], max_depth: int = 3,
           const_range: tuple[float, float] = (-5.0, 5.0),
           rng: random.Random = None) -> Node:
    """Apply a random mutation."""
    if rng is None:
        rng = random.Random()
    r = rng.random()
    if r < 0.4:
        return mutate_point(tree, var_names, const_range, rng)
    elif r < 0.8:
        return mutate_subtree(tree, var_names, max_depth, const_range, rng)
    else:
        return mutate_hoist(tree, rng)


def crossover(parent1: Node, parent2: Node,
              rng: random.Random = None) -> tuple[Node, Node]:
    """Subtree crossover: swap random subtrees between two parents."""
    if rng is None:
        rng = random.Random()

    nodes1 = all_nodes(parent1)
    nodes2 = all_nodes(parent2)

    _, path1 = rng.choice(nodes1)
    _, path2 = rng.choice(nodes2)

    subtree1 = get_node_at(parent1, path1)
    subtree2 = get_node_at(parent2, path2)

    child1 = replace_node_at(parent1, path1, subtree2)
    child2 = replace_node_at(parent2, path2, subtree1)

    return child1, child2


# --- Fitness ---

@dataclass
class TestCase:
    """A single input->output test case."""
    inputs: dict[str, float]
    expected: float


def fitness_mse(tree: Node, test_cases: list[TestCase]) -> float:
    """Mean squared error fitness (lower is better)."""
    if not test_cases:
        return float('inf')
    total_error = 0.0
    for tc in test_cases:
        try:
            result = evaluate(tree, tc.inputs)
            error = (result - tc.expected) ** 2
            if math.isinf(error) or math.isnan(error):
                return float('inf')
            total_error += error
        except Exception:
            return float('inf')
    return total_error / len(test_cases)


def fitness_with_parsimony(tree: Node, test_cases: list[TestCase],
                           parsimony_weight: float = 0.01) -> float:
    """MSE fitness with size penalty to prefer simpler programs."""
    mse = fitness_mse(tree, test_cases)
    if math.isinf(mse):
        return float('inf')
    return mse + parsimony_weight * tree.size()


# --- Selection ---

def tournament_select(population: list[tuple[Node, float]],
                      tournament_size: int = 3,
                      rng: random.Random = None) -> Node:
    """Select individual via tournament selection."""
    if rng is None:
        rng = random.Random()
    tournament = rng.sample(population, min(tournament_size, len(population)))
    winner = min(tournament, key=lambda x: x[1])
    return winner[0]


# --- Bloat Control ---

def enforce_depth_limit(tree: Node, max_depth: int, var_names: list[str],
                        rng: random.Random = None) -> Node:
    """If tree exceeds max depth, replace deep subtrees with terminals."""
    if tree.depth() <= max_depth:
        return tree
    if rng is None:
        rng = random.Random()
    tree = tree.copy()
    _trim(tree, max_depth, var_names, rng, current_depth=1)
    return tree


def _trim(node: Node, max_depth: int, var_names: list[str],
          rng: random.Random, current_depth: int):
    """Recursively trim tree to max depth."""
    for i, child in enumerate(node.children):
        if current_depth + 1 >= max_depth:
            # Replace with terminal
            node.children[i] = _random_terminal(var_names, (-5.0, 5.0), rng)
        else:
            _trim(child, max_depth, var_names, rng, current_depth + 1)


# --- Evolution Engine ---

@dataclass
class EvolutionConfig:
    """Configuration for the evolution run."""
    population_size: int = 200
    max_generations: int = 100
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    reproduction_rate: float = 0.1
    max_tree_depth: int = 8
    initial_max_depth: int = 4
    const_range: tuple[float, float] = (-5.0, 5.0)
    parsimony_weight: float = 0.001
    elitism: int = 5
    fitness_threshold: float = 0.001  # stop if best fitness below this
    stagnation_limit: int = 20  # inject diversity if no improvement for N gens


@dataclass
class EvolutionResult:
    """Result of an evolution run."""
    best_program: Node
    best_fitness: float
    generations_run: int
    fitness_history: list[float]  # best fitness per generation
    population_diversity: list[float]  # unique fitness values per gen
    converged: bool


class Evolver:
    """The evolution engine. Variation -> Evaluation -> Selection -> Repeat."""

    def __init__(self, var_names: list[str], test_cases: list[TestCase],
                 config: EvolutionConfig = None, seed: int = None):
        self.var_names = var_names
        self.test_cases = test_cases
        self.config = config or EvolutionConfig()
        self.rng = random.Random(seed)
        self.population: list[tuple[Node, float]] = []
        self.generation = 0
        self.fitness_history: list[float] = []
        self.diversity_history: list[float] = []

    def initialize(self):
        """Create initial population using ramped half-and-half."""
        self.population = []
        cfg = self.config
        for i in range(cfg.population_size):
            # Ramp max depth from 2 to initial_max_depth
            depth = 2 + (i % (cfg.initial_max_depth - 1))
            tree = random_tree(self.var_names, depth, cfg.const_range, self.rng)
            fit = self._evaluate(tree)
            self.population.append((tree, fit))
        self.population.sort(key=lambda x: x[1])
        self.generation = 0

    def _evaluate(self, tree: Node) -> float:
        """Evaluate a program's fitness."""
        return fitness_with_parsimony(
            tree, self.test_cases, self.config.parsimony_weight
        )

    def step(self) -> float:
        """Run one generation. Returns best fitness."""
        cfg = self.config
        new_pop = []

        # Elitism: keep best individuals
        self.population.sort(key=lambda x: x[1])
        for i in range(min(cfg.elitism, len(self.population))):
            new_pop.append(self.population[i])

        # Fill rest of population
        while len(new_pop) < cfg.population_size:
            r = self.rng.random()
            if r < cfg.crossover_rate:
                p1 = tournament_select(self.population, cfg.tournament_size, self.rng)
                p2 = tournament_select(self.population, cfg.tournament_size, self.rng)
                c1, c2 = crossover(p1, p2, self.rng)
                c1 = enforce_depth_limit(c1, cfg.max_tree_depth, self.var_names, self.rng)
                c2 = enforce_depth_limit(c2, cfg.max_tree_depth, self.var_names, self.rng)
                new_pop.append((c1, self._evaluate(c1)))
                if len(new_pop) < cfg.population_size:
                    new_pop.append((c2, self._evaluate(c2)))
            elif r < cfg.crossover_rate + cfg.mutation_rate:
                parent = tournament_select(self.population, cfg.tournament_size, self.rng)
                child = mutate(parent, self.var_names, cfg.initial_max_depth, cfg.const_range, self.rng)
                child = enforce_depth_limit(child, cfg.max_tree_depth, self.var_names, self.rng)
                new_pop.append((child, self._evaluate(child)))
            else:
                # Reproduction (copy)
                parent = tournament_select(self.population, cfg.tournament_size, self.rng)
                new_pop.append((parent.copy(), self._evaluate(parent)))

        self.population = new_pop
        self.population.sort(key=lambda x: x[1])
        self.generation += 1

        best_fit = self.population[0][1]
        self.fitness_history.append(best_fit)

        # Track diversity
        unique_fits = len(set(round(f, 6) for _, f in self.population))
        diversity = unique_fits / len(self.population)
        self.diversity_history.append(diversity)

        return best_fit

    def _inject_diversity(self):
        """Replace worst half of population with fresh random individuals."""
        cfg = self.config
        self.population.sort(key=lambda x: x[1])
        half = cfg.population_size // 2
        keep = self.population[:half]
        for _ in range(cfg.population_size - half):
            depth = 2 + (self.rng.randint(0, cfg.initial_max_depth - 2))
            tree = random_tree(self.var_names, depth, cfg.const_range, self.rng)
            fit = self._evaluate(tree)
            keep.append((tree, fit))
        self.population = keep

    def run(self) -> EvolutionResult:
        """Run evolution to completion."""
        self.initialize()
        cfg = self.config
        stagnation_counter = 0
        best_ever = float('inf')

        for gen in range(cfg.max_generations):
            best_fit = self.step()

            if best_fit < best_ever - 1e-10:
                best_ever = best_fit
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Convergence check
            if best_fit <= cfg.fitness_threshold:
                return EvolutionResult(
                    best_program=self.population[0][0].copy(),
                    best_fitness=best_fit,
                    generations_run=gen + 1,
                    fitness_history=self.fitness_history,
                    population_diversity=self.diversity_history,
                    converged=True,
                )

            # Stagnation handling
            if stagnation_counter >= cfg.stagnation_limit:
                self._inject_diversity()
                stagnation_counter = 0

        return EvolutionResult(
            best_program=self.population[0][0].copy(),
            best_fitness=self.population[0][1],
            generations_run=cfg.max_generations,
            fitness_history=self.fitness_history,
            population_diversity=self.diversity_history,
            converged=False,
        )

    def best(self) -> tuple[Node, float]:
        """Return current best individual."""
        self.population.sort(key=lambda x: x[1])
        return self.population[0]


# --- Problem Generators ---

def symbolic_regression(func: Callable, var_names: list[str],
                        samples: list[dict[str, float]]) -> list[TestCase]:
    """Generate test cases for symbolic regression from a target function."""
    cases = []
    for sample in samples:
        expected = func(**{k: sample[k] for k in var_names})
        cases.append(TestCase(inputs=sample, expected=expected))
    return cases


def make_samples_1d(x_range: tuple[float, float], n: int,
                    rng: random.Random = None) -> list[dict[str, float]]:
    """Generate n evenly-spaced 1D samples."""
    if rng is None:
        rng = random.Random()
    lo, hi = x_range
    step = (hi - lo) / (n - 1) if n > 1 else 0
    return [{'x': lo + i * step} for i in range(n)]


def make_samples_2d(x_range: tuple[float, float], y_range: tuple[float, float],
                    n_per_dim: int, rng: random.Random = None) -> list[dict[str, float]]:
    """Generate grid of 2D samples."""
    if rng is None:
        rng = random.Random()
    samples = []
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    x_step = (x_hi - x_lo) / (n_per_dim - 1) if n_per_dim > 1 else 0
    y_step = (y_hi - y_lo) / (n_per_dim - 1) if n_per_dim > 1 else 0
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            samples.append({'x': x_lo + i * x_step, 'y': y_lo + j * y_step})
    return samples
