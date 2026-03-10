"""
C023 -- Typed Evolutionary Optimizer
Composes: C022 (Meta-Evolver) + C013 (Type Checker) + C014 (Bytecode Optimizer) + C010 (Stack VM)

Evolves programs that must type-check, compiles and optimizes bytecode,
then evaluates fitness on optimized code. 4-system deep composition.
"""

import sys, os, copy, random, math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

# -- Import composed systems --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C013_type_checker'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C014_bytecode_optimizer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C022_meta_evolver'))

from stack_vm import (
    lex, Parser, Compiler, VM, Chunk, Op, Token, TokenType,
    compile_source, execute, FnObject,
    IntLit, FloatLit, BoolLit, StringLit, Var, BinOp, UnaryOp,
    LetDecl, Assign, PrintStmt, IfStmt, WhileStmt, Block,
    FnDecl, ReturnStmt, CallExpr, Program
)
from type_checker import (
    check_source, check_program, TypeChecker, TypeEnv,
    TInt, TFloat, TBool, TString, TVoid, TFunc, TVar as TypeVar,
    INT, FLOAT, STRING, BOOL, VOID,
    TypeError_ as TypeCheckError, format_errors, unify, resolve, types_compatible
)
from optimizer import (
    optimize_chunk, optimize_all, optimize_source, execute_optimized,
    OptimizationStats, Instr
)
from meta_evolver import (
    EvolvableProgram, FitnessResult, TestCase, EvolutionConfig, EvolutionResult,
    MetaEvolver, ProgramType,
    random_program, random_expr, compile_program, execute_program,
    program_to_source, evaluate_fitness, ast_depth, ast_size,
    mutate, crossover, tournament_select, simplify_program,
    enforce_depth, enforce_stmt_count,
    regression_problem, classification_problem, sequence_problem,
    make_samples_1d, make_samples_2d, Island
)


# ============================================================
# Type Annotations for Evolution
# ============================================================

class TypeSignature(Enum):
    """Expected type signatures for evolved programs."""
    NUMERIC = auto()       # int/float inputs -> numeric output
    BOOLEAN = auto()       # inputs -> bool output
    MIXED = auto()         # mixed types
    POLYMORPHIC = auto()   # type-variable inputs


@dataclass
class TypeConstraint:
    """Type constraint for a variable."""
    name: str
    expected_type: object  # TInt, TFloat, TBool, etc.


@dataclass
class TypedFitnessResult:
    """Extended fitness result with type checking and optimization info."""
    # Base fitness
    correctness: float = float('inf')
    efficiency: float = 0.0
    simplicity: float = 0.0
    compiled: bool = False
    executed: bool = False
    raw_score: float = float('inf')
    # Type checking
    type_checked: bool = False
    type_errors: int = 0
    type_error_messages: list = field(default_factory=list)
    inferred_type: object = None
    # Optimization
    optimized: bool = False
    original_size: int = 0
    optimized_size: int = 0
    optimization_ratio: float = 0.0  # percent reduction
    steps_before_opt: int = 0
    steps_after_opt: int = 0
    step_reduction: float = 0.0
    optimization_passes: int = 0

    @property
    def type_penalty(self):
        """Penalty for type errors."""
        return self.type_errors * 10.0

    @property
    def optimization_bonus(self):
        """Bonus for optimization effectiveness (lower is better)."""
        if not self.optimized:
            return 0.0
        return -self.step_reduction * 0.01  # negative = bonus


@dataclass
class TypedEvolutionConfig:
    """Configuration for typed evolution."""
    # Base evolution config
    population_size: int = 80
    max_generations: int = 50
    tournament_size: int = 5
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    reproduction_rate: float = 0.1
    max_tree_depth: int = 6
    initial_max_depth: int = 4
    max_stmts: int = 8
    fitness_threshold: float = 0.01
    stagnation_limit: int = 10
    elitism: int = 2
    # Island model
    num_islands: int = 3
    migration_interval: int = 5
    migration_count: int = 2
    # Typed evolution specific
    type_error_weight: float = 10.0     # penalty per type error
    optimization_weight: float = 0.005  # weight for optimization benefit
    efficiency_weight: float = 0.001
    simplicity_weight: float = 0.01
    require_type_check: bool = False    # if True, programs MUST type-check
    type_guided_mutation: bool = True   # use type info to guide mutations
    optimize_before_eval: bool = True   # optimize bytecode before fitness eval
    # Seed
    seed: Optional[int] = None

    def to_base_config(self):
        """Convert to base EvolutionConfig."""
        return EvolutionConfig(
            population_size=self.population_size,
            max_generations=self.max_generations,
            tournament_size=self.tournament_size,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            reproduction_rate=self.reproduction_rate,
            max_tree_depth=self.max_tree_depth,
            initial_max_depth=self.initial_max_depth,
            max_stmts=self.max_stmts,
            fitness_threshold=self.fitness_threshold,
            stagnation_limit=self.stagnation_limit,
            elitism=self.elitism,
            num_islands=self.num_islands,
            migration_interval=self.migration_interval,
            migration_count=self.migration_count,
            efficiency_weight=self.efficiency_weight,
            simplicity_weight=self.simplicity_weight,
        )


@dataclass
class TypedEvolutionResult:
    """Result of typed evolution."""
    best_program: Optional[EvolvableProgram] = None
    best_fitness: Optional[TypedFitnessResult] = None
    best_source: str = ""
    generations_run: int = 0
    fitness_history: list = field(default_factory=list)
    type_error_history: list = field(default_factory=list)
    optimization_history: list = field(default_factory=list)
    diversity_history: list = field(default_factory=list)
    converged: bool = False
    total_type_checks: int = 0
    total_optimizations: int = 0
    type_check_pass_rate: float = 0.0


# ============================================================
# Type-Aware Program Analysis
# ============================================================

def type_check_program(source):
    """Type-check a source string. Returns (errors, checker) or (errors, None)."""
    try:
        errors, checker = check_source(source)
        return errors, checker
    except Exception:
        return [TypeCheckError("Failed to parse for type checking", 0)], None


def infer_result_type(source, checker):
    """Infer the result type of a program from the type checker."""
    if checker is None:
        return None
    # Try to look up 'result' variable (common in evolver output)
    env = checker.env
    for name in ['result', '_result', 'r']:
        t = env.lookup(name)
        if t is not None:
            return resolve(t)
    return None


def type_compatible_value(value, expected_type):
    """Check if a runtime value matches an expected type."""
    if expected_type is None:
        return True
    if isinstance(expected_type, type(INT)):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected_type, type(FLOAT)):
        return isinstance(value, (int, float))
    if isinstance(expected_type, type(BOOL)):
        return isinstance(value, bool)
    if isinstance(expected_type, type(STRING)):
        return isinstance(value, str)
    return True


# ============================================================
# Type-Guided Mutation
# ============================================================

def type_guided_mutate(prog, rng, max_depth=6, const_range=(-10, 10), type_hints=None):
    """Mutate a program with type awareness.

    type_hints: dict mapping variable names to expected types (TInt, TFloat, etc.)
    Uses type info to prefer mutations that maintain type correctness.
    """
    new_prog = prog.copy()

    if prog.program_type == ProgramType.EXPRESSION:
        return _mutate_expr_typed(new_prog, rng, max_depth, const_range, type_hints)
    else:
        return _mutate_imperative_typed(new_prog, rng, max_depth, const_range, type_hints)


def _mutate_expr_typed(prog, rng, max_depth, const_range, type_hints):
    """Type-aware expression mutation."""
    roll = rng.random()
    if roll < 0.5:
        # Point mutation preserving numeric type
        prog.ast = _point_mutate_typed(prog.ast, rng, const_range, type_hints)
    elif roll < 0.8:
        # Subtree replacement with type-compatible subtree
        var_names = prog.input_vars[:]
        new_subtree = random_expr(var_names, rng, max_depth=max(2, max_depth - 2),
                                  const_range=const_range)
        prog.ast = _replace_random_subtree(prog.ast, new_subtree, rng)
    else:
        # Operator swap (preserves structure and types better)
        prog.ast = _swap_operator(prog.ast, rng)
    return prog


def _mutate_imperative_typed(prog, rng, max_depth, const_range, type_hints):
    """Type-aware imperative mutation."""
    roll = rng.random()
    if roll < 0.4:
        # Point mutation in a random statement
        if isinstance(prog.ast, list) and len(prog.ast) > 0:
            idx = rng.randint(0, len(prog.ast) - 1)
            stmt = prog.ast[idx]
            if isinstance(stmt, (LetDecl, Assign)):
                # Mutate the value expression
                if hasattr(stmt, 'value') and stmt.value is not None:
                    new_val = _point_mutate_typed(stmt.value, rng, const_range, type_hints)
                    if isinstance(stmt, LetDecl):
                        prog.ast[idx] = LetDecl(stmt.name, new_val)
                    else:
                        prog.ast[idx] = Assign(stmt.name, new_val)
    elif roll < 0.7:
        # Standard mutation fallback
        prog = mutate(prog, rng, max_depth, const_range)
    else:
        # Insert a type-compatible assignment
        if isinstance(prog.ast, list) and type_hints:
            var_names = list(type_hints.keys())
            if var_names:
                var = rng.choice(var_names)
                expr = _generate_typed_expr(type_hints.get(var), prog.input_vars, rng, const_range)
                new_stmt = Assign(var, expr)
                idx = rng.randint(0, len(prog.ast))
                prog.ast.insert(idx, new_stmt)
    return prog


def _point_mutate_typed(node, rng, const_range, type_hints):
    """Point mutation that tries to preserve types."""
    if isinstance(node, IntLit):
        return IntLit(node.value + rng.randint(-3, 3))
    elif isinstance(node, FloatLit):
        return FloatLit(round(node.value + rng.uniform(-2.0, 2.0), 4))
    elif isinstance(node, BoolLit):
        return BoolLit(not node.value)
    elif isinstance(node, BinOp):
        # Swap to type-compatible operator
        numeric_ops = ['+', '-', '*', '/']
        compare_ops = ['<', '>', '<=', '>=', '==', '!=']
        if node.op in numeric_ops:
            new_op = rng.choice(numeric_ops)
            return BinOp(new_op, node.left, node.right)
        elif node.op in compare_ops:
            new_op = rng.choice(compare_ops)
            return BinOp(new_op, node.left, node.right)
        return node
    elif isinstance(node, UnaryOp):
        return node
    elif isinstance(node, Var):
        # Replace with same-type variable if type hints available
        if type_hints and node.name in type_hints:
            expected = type_hints[node.name]
            same_type_vars = [v for v, t in type_hints.items() if t == expected]
            if same_type_vars:
                return Var(rng.choice(same_type_vars))
        return node
    return node


def _swap_operator(node, rng):
    """Swap an operator in the AST (preserves structure)."""
    if isinstance(node, BinOp):
        numeric_ops = ['+', '-', '*']
        compare_ops = ['<', '>', '==', '!=']
        roll = rng.random()
        if roll < 0.5 and isinstance(node.left, BinOp):
            return BinOp(node.op, _swap_operator(node.left, rng), node.right)
        elif roll < 0.8 and isinstance(node.right, BinOp):
            return BinOp(node.op, node.left, _swap_operator(node.right, rng))
        else:
            if node.op in numeric_ops:
                return BinOp(rng.choice(numeric_ops), node.left, node.right)
            elif node.op in compare_ops:
                return BinOp(rng.choice(compare_ops), node.left, node.right)
    return node


def _replace_random_subtree(node, replacement, rng):
    """Replace a random subtree in the AST."""
    if rng.random() < 0.3:
        return replacement
    if isinstance(node, BinOp):
        if rng.random() < 0.5:
            return BinOp(node.op,
                         _replace_random_subtree(node.left, replacement, rng),
                         node.right)
        else:
            return BinOp(node.op, node.left,
                         _replace_random_subtree(node.right, replacement, rng))
    return node


def _generate_typed_expr(expected_type, var_names, rng, const_range):
    """Generate an expression matching an expected type.
    const_range can be tuple (lo, hi) or int (treated as (-val, val)).
    """
    if isinstance(const_range, (list, tuple)):
        lo, hi = const_range
    else:
        lo, hi = -abs(const_range), abs(const_range)

    if expected_type is not None and isinstance(expected_type, type(BOOL)):
        if var_names and rng.random() < 0.5:
            return BinOp(rng.choice(['<', '>', '==']),
                         Var(rng.choice(var_names)),
                         IntLit(rng.randint(lo, hi)))
        return BoolLit(rng.choice([True, False]))
    elif expected_type is not None and isinstance(expected_type, type(FLOAT)):
        if var_names and rng.random() < 0.7:
            return BinOp(rng.choice(['+', '-', '*', '/']),
                         Var(rng.choice(var_names)),
                         FloatLit(round(rng.uniform(0.1, float(hi)), 2)))
        return FloatLit(round(rng.uniform(float(lo), float(hi)), 2))
    else:
        if var_names and rng.random() < 0.7:
            return BinOp(rng.choice(['+', '-', '*']),
                         Var(rng.choice(var_names)),
                         IntLit(rng.randint(max(1, lo), hi)))
        return IntLit(rng.randint(lo, hi))


# ============================================================
# Optimization Pipeline
# ============================================================

def compile_and_optimize(source, optimize=True):
    """Compile source to bytecode and optionally optimize.

    Returns dict with:
        chunk, compiler, optimized_chunk, stats, functions, opt_functions
    Or None on failure.
    """
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        chunk = compiler.compile(ast)
    except Exception:
        return None

    result = {
        'chunk': chunk,
        'compiler': compiler,
        'optimized_chunk': chunk,
        'stats': None,
        'functions': compiler.functions if hasattr(compiler, 'functions') else {},
        'opt_functions': {},
    }

    if optimize:
        try:
            opt_chunk, opt_funcs, stats = optimize_all(chunk, compiler)
            result['optimized_chunk'] = opt_chunk
            result['opt_functions'] = opt_funcs
            result['stats'] = stats
        except Exception:
            # Optimization failed, use unoptimized
            pass

    return result


def run_on_vm(chunk, functions=None, max_steps=10000, trace=False):
    """Run a chunk on the VM with optional function support.

    Returns dict with result, output, env, steps or None on failure.
    """
    try:
        vm = VM(chunk, trace=trace)
        # Load functions into VM constants if available
        if functions:
            for name, fn_obj in functions.items():
                if isinstance(fn_obj, FnObject):
                    idx = chunk.add_constant(fn_obj)
        result = vm.run(max_steps=max_steps) if hasattr(VM.run, '__code__') and 'max_steps' in VM.run.__code__.co_varnames else vm.run()
        return {
            'result': result,
            'output': vm.output if hasattr(vm, 'output') else [],
            'env': vm.env if hasattr(vm, 'env') else {},
            'steps': vm.step_count if hasattr(vm, 'step_count') else 0,
        }
    except Exception:
        return None


# ============================================================
# Typed Fitness Evaluation
# ============================================================

def evaluate_typed_fitness(prog, test_cases, config=None, max_steps=10000):
    """Evaluate fitness with type checking and optimization.

    Pipeline: source -> type-check -> compile -> optimize -> execute -> score
    """
    if config is None:
        config = TypedEvolutionConfig()

    result = TypedFitnessResult()
    source = program_to_source(prog)

    if not source or not source.strip():
        return result

    # Build full source with input declarations for type checking and compilation
    input_decls = []
    for v in prog.input_vars:
        input_decls.append(f"let {v} = 0;")
    full_source = '\n'.join(input_decls) + '\n' + source if input_decls else source

    # Step 1: Type check (with input declarations)
    type_errors, checker = type_check_program(full_source)
    result.type_errors = len(type_errors)
    result.type_error_messages = [e.message for e in type_errors]
    result.type_checked = (len(type_errors) == 0)
    result.inferred_type = infer_result_type(full_source, checker)

    # If type checking required and failed, return penalty
    if config.require_type_check and not result.type_checked:
        result.raw_score = float('inf')
        return result

    # Step 2: Compile (with input declarations)
    try:
        tokens = lex(full_source)
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        chunk = compiler.compile(ast)
        result.compiled = True
    except Exception:
        result.raw_score = float('inf')
        return result

    # Step 3: Optimize (if enabled)
    opt_chunk = chunk
    if config.optimize_before_eval:
        try:
            opt_chunk, stats = optimize_chunk(chunk)
            result.optimized = True
            result.original_size = stats.original_size
            result.optimized_size = stats.optimized_size
            result.optimization_ratio = stats.size_reduction * 100
            result.optimization_passes = stats.rounds
        except Exception:
            # Optimization failed, use unoptimized
            opt_chunk = chunk

    # Step 4: Execute against test cases (both optimized and unoptimized for comparison)
    total_error = 0.0
    total_steps_opt = 0
    total_steps_orig = 0
    cases_run = 0

    for tc in test_cases:
        # Build source with input values
        input_source = _build_input_source(prog, tc.inputs)
        if input_source is None:
            total_error += 1000.0
            continue

        # Execute original (for step comparison)
        orig_result = execute_program(prog, tc.inputs, max_steps=max_steps)
        if orig_result:
            total_steps_orig += orig_result.get('steps', 0)

        # Execute optimized
        opt_result = _execute_optimized_program(prog, tc.inputs, config, max_steps)
        if opt_result is None:
            total_error += 1000.0
            continue

        result.executed = True
        cases_run += 1
        total_steps_opt += opt_result.get('steps', 0)

        # Compare result to expected
        actual = opt_result.get('result')
        expected = tc.expected
        if actual is None:
            # Try to get from env
            if prog.result_var and prog.result_var in opt_result.get('env', {}):
                actual = opt_result['env'][prog.result_var]
            elif opt_result.get('output'):
                try:
                    actual = float(opt_result['output'][0])
                except (ValueError, IndexError):
                    pass

        if actual is None:
            total_error += 1000.0
        elif isinstance(expected, bool):
            total_error += 0.0 if (bool(actual) == expected) else 1.0
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            err = (actual - expected) ** 2
            if err <= tc.tolerance ** 2:
                err = 0.0
            total_error += err
        else:
            total_error += 0.0 if actual == expected else 1.0

    # Step 5: Compute final score
    result.correctness = total_error
    result.steps_before_opt = total_steps_orig
    result.steps_after_opt = total_steps_opt
    if total_steps_orig > 0:
        result.step_reduction = (total_steps_orig - total_steps_opt) / total_steps_orig * 100.0
    result.efficiency = total_steps_opt
    result.simplicity = ast_size(prog.ast) if prog.ast else 0

    # Multi-objective score
    score = result.correctness
    score += config.type_error_weight * result.type_errors
    score += config.efficiency_weight * result.efficiency
    score += config.simplicity_weight * result.simplicity
    # Optimization bonus (reward step reduction)
    if result.optimized and result.step_reduction > 0:
        score -= config.optimization_weight * result.step_reduction
    result.raw_score = score

    return result


def _build_input_source(prog, inputs):
    """Build source code with input variable declarations."""
    lines = []
    for name, value in inputs.items():
        if isinstance(value, bool):
            lines.append(f"let {name} = {'true' if value else 'false'};")
        elif isinstance(value, float):
            lines.append(f"let {name} = {value};")
        elif isinstance(value, int):
            lines.append(f"let {name} = {value};")
        elif isinstance(value, str):
            lines.append(f'let {name} = "{value}";')
        else:
            return None
    return '\n'.join(lines)


def _execute_optimized_program(prog, inputs, config, max_steps):
    """Execute a program with optimization pipeline."""
    source = program_to_source(prog)
    if not source:
        return None

    # Prepend input declarations
    input_lines = []
    for name, value in inputs.items():
        if isinstance(value, bool):
            input_lines.append(f"let {name} = {'true' if value else 'false'};")
        elif isinstance(value, float):
            input_lines.append(f"let {name} = {value};")
        elif isinstance(value, int):
            input_lines.append(f"let {name} = {value};")
        else:
            input_lines.append(f'let {name} = "{value}";')

    full_source = '\n'.join(input_lines) + '\n' + source

    if config.optimize_before_eval:
        try:
            tokens = lex(full_source)
            parser = Parser(tokens)
            ast = parser.parse()
            compiler = Compiler()
            chunk = compiler.compile(ast)
            opt_chunk, _ = optimize_chunk(chunk)
            vm = VM(opt_chunk)
            result = vm.run()
            return {
                'result': result,
                'output': vm.output if hasattr(vm, 'output') else [],
                'env': vm.env if hasattr(vm, 'env') else {},
                'steps': vm.step_count if hasattr(vm, 'step_count') else 0,
            }
        except Exception:
            return None
    else:
        return execute_program(prog, inputs, max_steps=max_steps)


# ============================================================
# Typed Island Model
# ============================================================

@dataclass
class TypedIsland:
    """Island with type checking statistics."""
    population: list = field(default_factory=list)  # [(EvolvableProgram, TypedFitnessResult)]
    config: TypedEvolutionConfig = None
    best_fitness: Optional[TypedFitnessResult] = None
    generations: int = 0
    type_check_passes: int = 0
    type_check_total: int = 0
    optimizations_run: int = 0


# ============================================================
# Typed Meta-Evolver (Main Engine)
# ============================================================

class TypedMetaEvolver:
    """Evolves programs with type checking and bytecode optimization.

    Composes:
    - C022 MetaEvolver: genetic programming engine
    - C013 TypeChecker: static type analysis
    - C014 Optimizer: bytecode optimization
    - C010 StackVM: execution substrate
    """

    def __init__(self, input_vars, test_cases, program_type=ProgramType.EXPRESSION,
                 config=None, seed=None):
        self.input_vars = input_vars
        self.test_cases = test_cases
        self.program_type = program_type
        self.config = config or TypedEvolutionConfig()
        if seed is not None:
            self.config.seed = seed
        self.rng = random.Random(self.config.seed)
        self.islands = []
        self.generation = 0
        self.total_type_checks = 0
        self.total_optimizations = 0
        self.type_check_passes = 0
        self.fitness_history = []
        self.type_error_history = []
        self.optimization_history = []
        self.diversity_history = []
        self._type_hint_cache = {}

    def initialize(self):
        """Create initial island populations."""
        pop_per_island = self.config.population_size // max(1, self.config.num_islands)
        for i in range(self.config.num_islands):
            island = TypedIsland(config=self.config)
            population = []
            attempts = 0
            while len(population) < pop_per_island and attempts < pop_per_island * 5:
                attempts += 1
                prog = random_program(
                    self.input_vars, self.rng, self.program_type,
                    max_depth=self.config.initial_max_depth,
                    const_range=(-10, 10),
                    num_statements=self.rng.randint(2, max(2, self.config.max_stmts // 2))
                )
                fitness = evaluate_typed_fitness(prog, self.test_cases, self.config)
                self.total_type_checks += 1
                if fitness.compiled:
                    self.total_optimizations += 1
                if fitness.type_checked:
                    self.type_check_passes += 1
                    island.type_check_passes += 1
                island.type_check_total += 1
                population.append((prog, fitness))

            # Fill remainder if needed
            while len(population) < pop_per_island:
                prog = random_program(self.input_vars, self.rng, self.program_type,
                                      max_depth=3, const_range=(-5, 5))
                fitness = evaluate_typed_fitness(prog, self.test_cases, self.config)
                self.total_type_checks += 1
                population.append((prog, fitness))

            # Sort by fitness
            population.sort(key=lambda x: x[1].raw_score)
            island.population = population
            island.best_fitness = population[0][1] if population else None
            self.islands.append(island)

    def step(self):
        """Run one generation across all islands."""
        self.generation += 1
        best_overall = None

        gen_type_errors = 0
        gen_opt_ratios = []

        for island in self.islands:
            self._evolve_island(island)
            if island.best_fitness:
                gen_type_errors += sum(1 for _, f in island.population if f.type_errors > 0)
                gen_opt_ratios.extend([f.optimization_ratio for _, f in island.population if f.optimized])
                if best_overall is None or island.best_fitness.raw_score < best_overall.raw_score:
                    best_overall = island.best_fitness

        # Migration
        if self.generation % self.config.migration_interval == 0:
            self._migrate()

        # Track history
        self.fitness_history.append(best_overall.raw_score if best_overall else float('inf'))
        total_pop = sum(len(isl.population) for isl in self.islands)
        self.type_error_history.append(gen_type_errors / max(1, total_pop) * 100)
        avg_opt = sum(gen_opt_ratios) / max(1, len(gen_opt_ratios)) if gen_opt_ratios else 0
        self.optimization_history.append(avg_opt)
        self.diversity_history.append(self._measure_diversity())

        return best_overall

    def _evolve_island(self, island):
        """Evolve one island for one generation."""
        pop = island.population
        if not pop:
            return

        new_pop = []

        # Elitism
        pop.sort(key=lambda x: x[1].raw_score)
        for i in range(min(self.config.elitism, len(pop))):
            new_pop.append(pop[i])

        # Build type hints from best programs
        type_hints = self._extract_type_hints(pop[:5])

        # Fill rest of population
        while len(new_pop) < len(pop):
            roll = self.rng.random()
            if roll < self.config.crossover_rate:
                # Crossover
                p1 = self._typed_tournament(pop)
                p2 = self._typed_tournament(pop)
                c1, c2 = crossover(p1, p2, self.rng)
                c1 = enforce_depth(c1, self.config.max_tree_depth, self.rng)
                c2 = enforce_depth(c2, self.config.max_tree_depth, self.rng)
                if self.program_type == ProgramType.IMPERATIVE:
                    c1 = enforce_stmt_count(c1, self.config.max_stmts)
                    c2 = enforce_stmt_count(c2, self.config.max_stmts)
                f1 = evaluate_typed_fitness(c1, self.test_cases, self.config)
                f2 = evaluate_typed_fitness(c2, self.test_cases, self.config)
                self.total_type_checks += 2
                self.total_optimizations += sum(1 for f in [f1, f2] if f.compiled)
                self.type_check_passes += sum(1 for f in [f1, f2] if f.type_checked)
                new_pop.append((c1, f1))
                if len(new_pop) < len(pop):
                    new_pop.append((c2, f2))
            elif roll < self.config.crossover_rate + self.config.mutation_rate:
                # Mutation (type-guided if enabled)
                parent = self._typed_tournament(pop)
                if self.config.type_guided_mutation and type_hints:
                    child = type_guided_mutate(parent, self.rng,
                                               self.config.max_tree_depth,
                                               type_hints=type_hints)
                else:
                    child = mutate(parent, self.rng, self.config.max_tree_depth)
                child = enforce_depth(child, self.config.max_tree_depth, self.rng)
                if self.program_type == ProgramType.IMPERATIVE:
                    child = enforce_stmt_count(child, self.config.max_stmts)
                fitness = evaluate_typed_fitness(child, self.test_cases, self.config)
                self.total_type_checks += 1
                if fitness.compiled:
                    self.total_optimizations += 1
                if fitness.type_checked:
                    self.type_check_passes += 1
                new_pop.append((child, fitness))
            else:
                # Reproduction
                parent = self._typed_tournament(pop)
                new_pop.append((parent.copy(), evaluate_typed_fitness(
                    parent, self.test_cases, self.config)))
                self.total_type_checks += 1

        island.population = new_pop[:len(pop)]
        island.population.sort(key=lambda x: x[1].raw_score)
        island.best_fitness = island.population[0][1]
        island.generations += 1

    def _typed_tournament(self, pop):
        """Tournament selection with type-check preference."""
        candidates = [self.rng.choice(pop) for _ in range(self.config.tournament_size)]
        # Sort by raw_score (type errors already factored in)
        candidates.sort(key=lambda x: x[1].raw_score)
        return candidates[0][0]

    def _migrate(self):
        """Ring topology migration between islands."""
        if len(self.islands) < 2:
            return
        migrants = []
        for island in self.islands:
            island.population.sort(key=lambda x: x[1].raw_score)
            best = island.population[:self.config.migration_count]
            migrants.append([(p.copy(), f) for p, f in best])

        for i in range(len(self.islands)):
            target = (i + 1) % len(self.islands)
            for prog, fitness in migrants[i]:
                # Replace worst in target
                if self.islands[target].population:
                    self.islands[target].population[-1] = (prog, fitness)
                    self.islands[target].population.sort(key=lambda x: x[1].raw_score)

    def _extract_type_hints(self, top_programs):
        """Extract type hints from top-performing programs."""
        hints = {}
        for prog, fitness in top_programs:
            if fitness.inferred_type is not None:
                for var in prog.input_vars:
                    if var not in hints:
                        hints[var] = INT  # default to numeric
        return hints if hints else None

    def _measure_diversity(self):
        """Measure population diversity as unique fitness values."""
        scores = set()
        for island in self.islands:
            for _, f in island.population:
                scores.add(round(f.raw_score, 4))
        total_pop = sum(len(isl.population) for isl in self.islands)
        return len(scores) / max(1, total_pop) * 100

    def best(self):
        """Return best (program, fitness) across all islands."""
        best_prog = None
        best_fit = None
        for island in self.islands:
            if island.population:
                prog, fit = island.population[0]
                if best_fit is None or fit.raw_score < best_fit.raw_score:
                    best_prog = prog
                    best_fit = fit
        return best_prog, best_fit

    def run(self):
        """Run full typed evolution. Returns TypedEvolutionResult."""
        self.initialize()
        stagnation = 0
        prev_best = float('inf')

        for gen in range(self.config.max_generations):
            best_fitness = self.step()
            if best_fitness is None:
                continue

            # Check convergence
            if best_fitness.raw_score <= self.config.fitness_threshold:
                best_prog, best_fit = self.best()
                return self._build_result(best_prog, best_fit, gen + 1, converged=True)

            # Stagnation detection
            if abs(best_fitness.raw_score - prev_best) < 1e-8:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = best_fitness.raw_score

            if stagnation >= self.config.stagnation_limit:
                # Inject diversity
                self._inject_diversity()
                stagnation = 0

        best_prog, best_fit = self.best()
        return self._build_result(best_prog, best_fit, self.config.max_generations, converged=False)

    def _inject_diversity(self):
        """Replace worst individuals with fresh random programs."""
        for island in self.islands:
            n_replace = max(1, len(island.population) // 4)
            for i in range(n_replace):
                prog = random_program(
                    self.input_vars, self.rng, self.program_type,
                    max_depth=self.config.initial_max_depth,
                    const_range=(-10, 10),
                    num_statements=self.rng.randint(2, max(2, self.config.max_stmts // 2))
                )
                fitness = evaluate_typed_fitness(prog, self.test_cases, self.config)
                self.total_type_checks += 1
                idx = len(island.population) - 1 - i
                if idx >= 0:
                    island.population[idx] = (prog, fitness)
            island.population.sort(key=lambda x: x[1].raw_score)

    def _build_result(self, prog, fitness, gens, converged):
        """Build TypedEvolutionResult."""
        source = program_to_source(prog) if prog else ""
        total_tc = self.total_type_checks
        return TypedEvolutionResult(
            best_program=prog,
            best_fitness=fitness,
            best_source=source,
            generations_run=gens,
            fitness_history=self.fitness_history[:],
            type_error_history=self.type_error_history[:],
            optimization_history=self.optimization_history[:],
            diversity_history=self.diversity_history[:],
            converged=converged,
            total_type_checks=total_tc,
            total_optimizations=self.total_optimizations,
            type_check_pass_rate=self.type_check_passes / max(1, total_tc) * 100,
        )


# ============================================================
# Problem Generators (Typed Variants)
# ============================================================

def typed_regression_problem(func, input_vars, n_samples=20, tolerance=0.1,
                             input_type=FLOAT, output_type=FLOAT):
    """Generate a regression problem with type annotations."""
    # Generate sample dicts
    samples = [{v: random.uniform(-10, 10) for v in input_vars} for _ in range(n_samples)]
    test_cases = regression_problem(func, input_vars, samples, tolerance)
    type_constraints = [TypeConstraint(v, input_type) for v in input_vars]
    return test_cases, type_constraints


def typed_classification_problem(func, input_vars, n_samples=20):
    """Generate a classification problem with type annotations."""
    samples = [{v: random.uniform(-10, 10) for v in input_vars} for _ in range(n_samples)]
    test_cases = classification_problem(func, input_vars, samples)
    type_constraints = [TypeConstraint(v, FLOAT) for v in input_vars]
    return test_cases, type_constraints


def typed_sequence_problem(func, n_values=15):
    """Generate a sequence problem with integer types."""
    test_cases = sequence_problem(func, n_values)
    type_constraints = [TypeConstraint('x', INT)]
    return test_cases, type_constraints


# ============================================================
# Analysis Utilities
# ============================================================

def analyze_evolution(result):
    """Analyze a TypedEvolutionResult and return summary dict."""
    summary = {
        'converged': result.converged,
        'generations': result.generations_run,
        'best_score': result.best_fitness.raw_score if result.best_fitness else None,
        'best_correctness': result.best_fitness.correctness if result.best_fitness else None,
        'type_checked': result.best_fitness.type_checked if result.best_fitness else False,
        'type_errors': result.best_fitness.type_errors if result.best_fitness else 0,
        'optimized': result.best_fitness.optimized if result.best_fitness else False,
        'optimization_ratio': result.best_fitness.optimization_ratio if result.best_fitness else 0,
        'step_reduction': result.best_fitness.step_reduction if result.best_fitness else 0,
        'total_type_checks': result.total_type_checks,
        'type_check_pass_rate': result.type_check_pass_rate,
        'source': result.best_source,
    }
    return summary


def compare_typed_vs_untyped(input_vars, test_cases, program_type=ProgramType.EXPRESSION,
                              config=None, seed=42):
    """Run evolution with and without type checking, compare results."""
    # Typed evolution
    typed_config = config or TypedEvolutionConfig(seed=seed, max_generations=30)
    typed_evo = TypedMetaEvolver(input_vars, test_cases, program_type, typed_config, seed)
    typed_result = typed_evo.run()

    # Untyped evolution (use base MetaEvolver)
    base_config = typed_config.to_base_config()
    base_config.max_generations = typed_config.max_generations
    untyped_evo = MetaEvolver(input_vars, test_cases, program_type, base_config, seed)
    untyped_result = untyped_evo.run()

    return {
        'typed': analyze_evolution(typed_result),
        'untyped': {
            'converged': untyped_result.converged,
            'generations': untyped_result.generations_run,
            'best_score': untyped_result.best_fitness.raw_score if untyped_result.best_fitness else None,
            'best_correctness': untyped_result.best_fitness.correctness if untyped_result.best_fitness else None,
            'source': untyped_result.best_source,
        },
        'typed_better': (typed_result.best_fitness.raw_score if typed_result.best_fitness else float('inf')) <
                        (untyped_result.best_fitness.raw_score if untyped_result.best_fitness else float('inf')),
    }


def optimization_analysis(prog, test_cases):
    """Analyze optimization impact on a specific program."""
    source = program_to_source(prog)
    if not source:
        return None

    # Check types
    errors, checker = type_check_program(source)

    # Compile
    try:
        tokens = lex(source)
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        chunk = compiler.compile(ast)
    except Exception:
        return None

    original_size = len(chunk.code)

    # Optimize
    try:
        opt_chunk, stats = optimize_chunk(chunk)
        optimized_size = len(opt_chunk.code)
    except Exception:
        return {'type_errors': len(errors), 'optimizable': False}

    return {
        'type_errors': len(errors),
        'type_clean': len(errors) == 0,
        'original_size': original_size,
        'optimized_size': optimized_size,
        'reduction_percent': stats.size_reduction * 100,
        'rounds': stats.rounds,
        'optimizable': True,
    }
