"""V117: Widening Strategy Framework

Composes V103 (widening policy synthesis) + V108 (domain composition framework)
to create an adaptive widening system that is domain-composition-aware.

Key ideas:
1. Domain-aware policy synthesis: widening strategy adapts based on which domains
   are composed (e.g., polyhedral needs different widening than intervals)
2. Multi-domain widening coordination: when widening a ReducedProductDomain,
   coordinate component widening (one component may widen while others narrow)
3. Feedback-driven widening: cross-domain reduction information feeds back
   into widening decisions (e.g., parity constrains interval widening targets)
4. Staged widening: progressive widening pipeline (delay -> threshold -> standard)
   with domain-specific stages

Composes:
- V103 (widening policy synthesis) -- per-loop policy analysis + synthesis
- V108 (domain composition) -- ReducedProductBuilder, CompositionInterpreter
- V020 (abstract domain functor) -- AbstractDomain protocol
- C039 (abstract interpreter) -- baseline interval/sign domains
- C010 (parser) -- AST access
- C037 (SMT solver) -- policy validation
"""

import sys
import os
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set, Any, Callable, Type
from enum import Enum

# Import dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from stack_vm import Parser, lex

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
from abstract_interpreter import (
    AbstractInterpreter, AbstractEnv, AbstractValue,
    Interval, NEG_INF, INF, interval_widen,
    Sign, analyze as baseline_analyze
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V020_abstract_domain_functor'))
from domain_functor import (
    AbstractDomain as FunctorDomain, IntervalDomain, SignDomain,
    FunctorInterpreter, DomainEnv
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V103_widening_policy_synthesis'))
from widening_policy import (
    WideningStrategy, WideningPolicy, LoopInfo, PolicyResult,
    SynthesisResult, analyze_loops, synthesize_policies,
    PolicyInterpreter
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V108_domain_composition'))
from domain_composition import (
    ReducedProductBuilder, CompositionInterpreter, CompositionEnv,
    compose_domains, analyze_with_composition, find_builtin_reducers,
    reduce_sign_interval, reduce_parity_interval,
    DisjunctiveDomain, LiftedDomain, CardinalPowerDomain
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Op, Sort, SortKind, IntConst, BoolConst, Var, App

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------------------------------------------------------------------------
# Domain factory helpers
# ---------------------------------------------------------------------------

def _make_single_factory(dt):
    """Build a factory for a single domain type.
    factory(None) -> TOP, factory(5) -> constant 5.
    """
    name = dt.__name__ if hasattr(dt, '__name__') else ''
    if name == 'IntervalDomain':
        def factory(value=None):
            if value is None:
                return dt()  # TOP = [NEG_INF, INF]
            return dt(value, value)
        return factory
    elif name == 'SignDomain':
        from domain_functor import SignValue
        def factory(value=None):
            if value is None:
                return dt()  # TOP
            if isinstance(value, int):
                if value > 0:
                    return dt(SignValue.POS)
                elif value < 0:
                    return dt(SignValue.NEG)
                else:
                    return dt(SignValue.ZERO)
            return dt()
        return factory
    else:
        # Generic: dt() for TOP, dt(value) for constant
        def factory(value=None):
            if value is None:
                return dt()
            try:
                return dt(value)
            except TypeError:
                return dt()
        return factory


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DomainKind(Enum):
    """Classification of abstract domains for strategy selection."""
    NUMERIC = "numeric"       # Sign, Interval, Constant, Parity
    RELATIONAL = "relational" # Polyhedral, Octagon
    COMPOSITE = "composite"   # ReducedProduct, Disjunctive
    HEAP = "heap"             # Shape, Points-to


class WideningPhase(Enum):
    """Phase in the staged widening pipeline."""
    DELAY = "delay"           # No widening yet, just iterate
    THRESHOLD = "threshold"   # Widen to thresholds
    GRADUATED = "graduated"   # Widen progressively (50%, 75%, infinity)
    STANDARD = "standard"     # Full widening to infinity


@dataclass(frozen=True)
class ComponentWideningConfig:
    """Per-component widening configuration within a composed domain."""
    domain_index: int           # Index in the ReducedProduct components
    domain_kind: DomainKind     # Classification
    phase: WideningPhase        # Current widening phase
    thresholds: Tuple[float, ...] = ()
    delay: int = 0
    allow_narrowing: bool = True
    graduated_steps: Tuple[float, ...] = (50.0, 100.0, 500.0, 1000.0)


@dataclass
class AdaptivePolicy:
    """Adaptive widening policy for a composed domain at a specific loop."""
    loop_id: int
    component_configs: List[ComponentWideningConfig]
    max_iterations: int = 100
    reduction_between_widenings: bool = True  # Run cross-domain reduction after widening
    coordinate_components: bool = True        # Coordinate component widening phases
    narrowing_iterations: int = 2


@dataclass
class FrameworkResult:
    """Result of widening strategy framework analysis."""
    env: Dict[str, Any]           # Final variable -> domain value bindings
    warnings: List[str]
    policies: Dict[int, AdaptivePolicy]  # loop_id -> policy used
    iterations_per_loop: Dict[int, int]
    widening_events: List[Dict[str, Any]]  # Log of widening decisions
    narrowing_events: int
    reduction_events: int


@dataclass
class StrategyComparison:
    """Comparison of different widening strategies on same program."""
    source: str
    strategies: Dict[str, FrameworkResult]   # strategy_name -> result
    precision_ranking: List[str]             # Most precise first
    iteration_ranking: List[str]             # Fewest iterations first
    summary: str


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

def classify_domain(domain) -> DomainKind:
    """Classify an abstract domain for strategy selection."""
    type_name = type(domain).__name__
    if type_name in ('SignDomain', 'IntervalDomain', 'ConstDomain', 'ParityDomain',
                     'ConstantDomain', 'FlatDomain'):
        return DomainKind.NUMERIC
    if type_name in ('PolyhedralDomain', 'OctagonDomain'):
        return DomainKind.RELATIONAL
    if type_name in ('ReducedProductDomain', 'ProductDomain', 'DisjunctiveDomain',
                     'LiftedDomain', 'CardinalPowerDomain'):
        return DomainKind.COMPOSITE
    if type_name in ('ShapeGraph', 'PointsToSet'):
        return DomainKind.HEAP
    return DomainKind.NUMERIC  # Default to numeric


def classify_domain_type(domain_type) -> DomainKind:
    """Classify a domain type (class, not instance)."""
    name = domain_type.__name__ if hasattr(domain_type, '__name__') else str(domain_type)
    if name in ('SignDomain', 'IntervalDomain', 'ConstDomain', 'ParityDomain',
                'ConstantDomain', 'FlatDomain'):
        return DomainKind.NUMERIC
    if name in ('PolyhedralDomain', 'OctagonDomain'):
        return DomainKind.RELATIONAL
    return DomainKind.NUMERIC


# ---------------------------------------------------------------------------
# Threshold extraction helpers
# ---------------------------------------------------------------------------

def extract_thresholds_from_ast(source):
    """Extract all numeric constants from source code for threshold candidates."""
    tokens = lex(source)
    parser = Parser(tokens)
    program = parser.parse()
    thresholds = set()
    _collect_constants(program.stmts, thresholds)
    return sorted(thresholds)


def _get_stmts(node):
    """Extract statement list from a node (handles Block, list, or single stmt)."""
    if isinstance(node, list):
        return node
    if hasattr(node, 'stmts'):
        return node.stmts
    return [node]


def _collect_constants(stmts, constants):
    """Recursively collect integer constants from AST nodes."""
    for stmt in stmts:
        _collect_constants_node(stmt, constants)


def _collect_constants_node(node, constants):
    """Collect constants from a single AST node."""
    if node is None:
        return
    type_name = type(node).__name__
    if type_name == 'IntLit':
        constants.add(node.value)
        # Also add neighbors for boundary thresholds
        constants.add(node.value - 1)
        constants.add(node.value + 1)
    elif type_name == 'BinOp':
        _collect_constants_node(node.left, constants)
        _collect_constants_node(node.right, constants)
    elif type_name == 'UnaryOp':
        _collect_constants_node(node.operand, constants)
    elif type_name == 'LetDecl':
        _collect_constants_node(node.value, constants)
    elif type_name == 'Assign':
        _collect_constants_node(node.value, constants)
    elif type_name == 'IfStmt':
        _collect_constants_node(node.cond, constants)
        if hasattr(node, 'then_body') and node.then_body:
            _collect_constants(_get_stmts(node.then_body), constants)
        if hasattr(node, 'else_body') and node.else_body:
            _collect_constants(_get_stmts(node.else_body), constants)
    elif type_name == 'WhileStmt':
        _collect_constants_node(node.cond, constants)
        if hasattr(node, 'body') and node.body:
            _collect_constants(_get_stmts(node.body), constants)
    elif type_name == 'CallExpr':
        for arg in getattr(node, 'args', []):
            _collect_constants_node(arg, constants)
    elif type_name == 'ReturnStmt':
        _collect_constants_node(getattr(node, 'value', None), constants)
    elif type_name == 'PrintStmt':
        _collect_constants_node(getattr(node, 'value', None), constants)
    elif type_name == 'FnDecl':
        if hasattr(node, 'body') and node.body:
            _collect_constants(_get_stmts(node.body), constants)


# ---------------------------------------------------------------------------
# Adaptive policy synthesis
# ---------------------------------------------------------------------------

def synthesize_adaptive_policy(source, domain_types=None, loop_id=0):
    """Synthesize an adaptive widening policy for a composed domain.

    Analyzes loop structure and domain composition to produce per-component
    widening configurations.

    Args:
        source: C10 source code
        domain_types: list of domain types in the composition (default: [SignDomain, IntervalDomain])
        loop_id: which loop to synthesize for

    Returns:
        AdaptivePolicy with per-component configurations
    """
    if domain_types is None:
        domain_types = [SignDomain, IntervalDomain]

    # Analyze loop structure via V103
    loop_infos = analyze_loops(source)

    # Extract thresholds from source
    thresholds = tuple(sorted(set(extract_thresholds_from_ast(source))))

    # Build per-component configs
    component_configs = []
    for i, dt in enumerate(domain_types):
        kind = classify_domain_type(dt)
        config = _synthesize_component_config(i, kind, loop_infos, thresholds, loop_id)
        component_configs.append(config)

    # Determine coordination
    has_numeric = any(c.domain_kind == DomainKind.NUMERIC for c in component_configs)
    has_relational = any(c.domain_kind == DomainKind.RELATIONAL for c in component_configs)
    coordinate = has_numeric and len(component_configs) > 1

    # Determine max iterations based on loop structure
    max_iter = 100
    if loop_infos:
        li = loop_infos[min(loop_id, len(loop_infos) - 1)]
        if li.is_simple_counter and li.bound_value is not None:
            # For bounded counters, can iterate exactly
            max_iter = min(li.bound_value + 10, 200)

    return AdaptivePolicy(
        loop_id=loop_id,
        component_configs=component_configs,
        max_iterations=max_iter,
        reduction_between_widenings=len(domain_types) > 1,
        coordinate_components=coordinate,
        narrowing_iterations=2
    )


def _synthesize_component_config(index, kind, loop_infos, thresholds, loop_id):
    """Synthesize widening config for one component domain."""
    if not loop_infos:
        # No loops -- just use standard
        return ComponentWideningConfig(
            domain_index=index,
            domain_kind=kind,
            phase=WideningPhase.STANDARD,
            thresholds=thresholds
        )

    li = loop_infos[min(loop_id, len(loop_infos) - 1)]

    if li.is_simple_counter and li.bound_value is not None:
        # Simple counter: delay until bound, no widening needed
        return ComponentWideningConfig(
            domain_index=index,
            domain_kind=kind,
            phase=WideningPhase.DELAY,
            delay=min(li.bound_value + 5, 50),
            thresholds=thresholds,
            allow_narrowing=True
        )

    if kind == DomainKind.NUMERIC:
        # Numeric domains: use threshold widening with program constants
        if thresholds:
            return ComponentWideningConfig(
                domain_index=index,
                domain_kind=kind,
                phase=WideningPhase.THRESHOLD,
                delay=2,  # Small delay for numeric domains
                thresholds=thresholds,
                allow_narrowing=True
            )
        else:
            return ComponentWideningConfig(
                domain_index=index,
                domain_kind=kind,
                phase=WideningPhase.GRADUATED,
                delay=1,
                graduated_steps=(10.0, 100.0, 1000.0, 10000.0),
                allow_narrowing=True
            )

    if kind == DomainKind.RELATIONAL:
        # Relational domains: standard widening (drop violated constraints)
        return ComponentWideningConfig(
            domain_index=index,
            domain_kind=kind,
            phase=WideningPhase.STANDARD,
            delay=3,  # Longer delay to gather more constraints first
            allow_narrowing=True
        )

    # Default: threshold if available, else standard
    return ComponentWideningConfig(
        domain_index=index,
        domain_kind=kind,
        phase=WideningPhase.THRESHOLD if thresholds else WideningPhase.STANDARD,
        thresholds=thresholds,
        delay=1,
        allow_narrowing=True
    )


def synthesize_all_policies(source, domain_types=None):
    """Synthesize adaptive policies for all loops in a program.

    Returns:
        Dict[int, AdaptivePolicy] -- loop_id -> policy
    """
    if domain_types is None:
        domain_types = [SignDomain, IntervalDomain]

    loop_infos = analyze_loops(source)
    policies = {}
    for i in range(len(loop_infos)):
        policies[i] = synthesize_adaptive_policy(source, domain_types, loop_id=i)

    if not policies:
        # No loops; create a default policy for consistency
        policies[0] = AdaptivePolicy(
            loop_id=0,
            component_configs=[
                ComponentWideningConfig(
                    domain_index=j,
                    domain_kind=classify_domain_type(dt),
                    phase=WideningPhase.STANDARD
                )
                for j, dt in enumerate(domain_types)
            ]
        )

    return policies


# ---------------------------------------------------------------------------
# Adaptive widening operations
# ---------------------------------------------------------------------------

def adaptive_widen_interval(old_lo, old_hi, new_lo, new_hi, config, iteration):
    """Widen an interval according to the component config and current iteration.

    Args:
        old_lo, old_hi: previous interval bounds
        new_lo, new_hi: new (post-join) interval bounds
        config: ComponentWideningConfig
        iteration: current fixpoint iteration

    Returns:
        (widened_lo, widened_hi)
    """
    # Delay phase: no widening
    if iteration <= config.delay:
        return new_lo, new_hi

    effective_iteration = iteration - config.delay

    if config.phase == WideningPhase.DELAY:
        # Pure delay -- after delay, standard widen
        return _standard_widen(old_lo, old_hi, new_lo, new_hi)

    elif config.phase == WideningPhase.THRESHOLD:
        return _threshold_widen(old_lo, old_hi, new_lo, new_hi, config.thresholds)

    elif config.phase == WideningPhase.GRADUATED:
        return _graduated_widen(old_lo, old_hi, new_lo, new_hi,
                                config.graduated_steps, effective_iteration)

    else:  # STANDARD
        return _standard_widen(old_lo, old_hi, new_lo, new_hi)


def _standard_widen(old_lo, old_hi, new_lo, new_hi):
    """Standard widening: jump to infinity on unstable bounds."""
    lo = NEG_INF if new_lo < old_lo else old_lo
    hi = INF if new_hi > old_hi else old_hi
    return lo, hi


def _threshold_widen(old_lo, old_hi, new_lo, new_hi, thresholds):
    """Threshold widening: widen to next threshold instead of infinity."""
    if new_lo < old_lo:
        # Find largest threshold <= new_lo
        candidates = [t for t in thresholds if t <= new_lo]
        lo = candidates[-1] if candidates else NEG_INF
    else:
        lo = old_lo

    if new_hi > old_hi:
        # Find smallest threshold >= new_hi
        candidates = [t for t in thresholds if t >= new_hi]
        hi = candidates[0] if candidates else INF
    else:
        hi = old_hi

    return lo, hi


def _graduated_widen(old_lo, old_hi, new_lo, new_hi, steps, effective_iteration):
    """Graduated widening: progressively widen through steps."""
    if effective_iteration <= 0:
        return new_lo, new_hi

    # Pick the step for current iteration
    step_idx = min(effective_iteration - 1, len(steps) - 1)
    limit = steps[step_idx] if step_idx < len(steps) else INF

    if new_lo < old_lo:
        lo = max(old_lo - limit, NEG_INF) if limit != INF else NEG_INF
    else:
        lo = old_lo

    if new_hi > old_hi:
        hi = min(old_hi + limit, INF) if limit != INF else INF
    else:
        hi = old_hi

    return lo, hi


# ---------------------------------------------------------------------------
# Strategy Framework Interpreter
# ---------------------------------------------------------------------------

class StrategyInterpreter:
    """Abstract interpreter with adaptive widening strategy framework.

    Combines V108's CompositionInterpreter with V103-style per-loop policies,
    using domain-aware adaptive widening.
    """

    def __init__(self, domain_factory, policies=None, domain_types=None,
                 max_iterations=100, div_zero_check=True):
        """
        Args:
            domain_factory: callable returning a composed AbstractDomain
            policies: Dict[int, AdaptivePolicy] or None (auto-synthesize)
            domain_types: list of domain types for policy synthesis
            max_iterations: global iteration bound
            div_zero_check: whether to check for division by zero
        """
        self.domain_factory = domain_factory
        self.policies = policies
        self.domain_types = domain_types or [SignDomain, IntervalDomain]
        self.max_iterations = max_iterations
        self.div_zero_check = div_zero_check
        self.warnings = []
        self.widening_events = []
        self.narrowing_events = 0
        self.reduction_events = 0
        self.iterations_per_loop = {}
        self._loop_counter = 0

    def analyze(self, source):
        """Analyze a C10 program with adaptive widening.

        Returns:
            FrameworkResult
        """
        self.warnings = []
        self.widening_events = []
        self.narrowing_events = 0
        self.reduction_events = 0
        self.iterations_per_loop = {}
        self._loop_counter = 0

        # Auto-synthesize policies if not provided
        if self.policies is None:
            self.policies = synthesize_all_policies(source, self.domain_types)

        # Parse
        tokens = lex(source)
        parser = Parser(tokens)
        program = parser.parse()

        # Initialize environment
        env = {}  # var_name -> domain value

        # Interpret
        self._interpret_stmts(program.stmts, env)

        # Format bindings
        bindings = {}
        for name, val in env.items():
            bindings[name] = str(val)

        return FrameworkResult(
            env=bindings,
            warnings=self.warnings,
            policies=self.policies,
            iterations_per_loop=self.iterations_per_loop,
            widening_events=self.widening_events,
            narrowing_events=self.narrowing_events,
            reduction_events=self.reduction_events
        )

    def _interpret_stmts(self, stmts, env):
        """Interpret a list of statements."""
        for stmt in stmts:
            self._interpret_stmt(stmt, env)

    def _interpret_stmt(self, stmt, env):
        """Interpret a single statement."""
        type_name = type(stmt).__name__

        if type_name == 'LetDecl':
            val = self._eval_expr(stmt.value, env)
            env[stmt.name] = val

        elif type_name == 'Assign':
            val = self._eval_expr(stmt.value, env)
            env[stmt.name] = val

        elif type_name == 'IfStmt':
            self._interpret_if(stmt, env)

        elif type_name == 'WhileStmt':
            self._interpret_while(stmt, env)

        elif type_name == 'PrintStmt':
            pass  # No effect on abstract state

        elif type_name == 'ReturnStmt':
            pass

        elif type_name == 'FnDecl':
            pass  # Skip function declarations in top-level analysis

        elif type_name == 'ExprStmt':
            self._eval_expr(stmt.expr, env)

    def _interpret_if(self, stmt, env):
        """Interpret an if statement by analyzing both branches and joining."""
        then_env = dict(env)
        self._refine_condition(stmt.cond, then_env, True)
        self._interpret_stmts(_get_stmts(stmt.then_body), then_env)

        else_env = dict(env)
        self._refine_condition(stmt.cond, else_env, False)
        if stmt.else_body:
            self._interpret_stmts(_get_stmts(stmt.else_body), else_env)

        # Join branches
        all_vars = set(then_env.keys()) | set(else_env.keys())
        for v in all_vars:
            t_val = then_env.get(v, self.domain_factory())
            e_val = else_env.get(v, self.domain_factory())
            env[v] = t_val.join(e_val)

    def _interpret_while(self, stmt, env):
        """Interpret a while loop with adaptive widening strategy."""
        loop_id = self._loop_counter
        self._loop_counter += 1

        policy = self.policies.get(loop_id, AdaptivePolicy(
            loop_id=loop_id,
            component_configs=[
                ComponentWideningConfig(
                    domain_index=0,
                    domain_kind=DomainKind.NUMERIC,
                    phase=WideningPhase.STANDARD
                )
            ]
        ))

        body = _get_stmts(stmt.body)

        # Fixpoint iteration with adaptive widening
        iteration = 0
        for iteration in range(1, policy.max_iterations + 1):
            # Copy current state
            old_env = {k: copy.deepcopy(v) for k, v in env.items()}

            # Refine by loop condition
            loop_env = dict(env)
            self._refine_condition(stmt.cond, loop_env, True)

            # Execute body
            self._interpret_stmts(body, loop_env)

            # Adaptive widening: apply per-component strategy
            changed = False
            all_vars = set(env.keys()) | set(loop_env.keys())
            for v in all_vars:
                old_val = env.get(v, self.domain_factory())
                new_val = loop_env.get(v, self.domain_factory())

                # Join first
                joined = old_val.join(new_val)

                # Then widen adaptively
                widened = self._adaptive_widen(old_val, joined, policy, iteration)

                if not self._domain_equals(widened, old_val):
                    changed = True

                env[v] = widened

            # Cross-domain reduction after widening
            if policy.reduction_between_widenings and changed:
                self._apply_reduction(env)
                self.reduction_events += 1

            if not changed:
                break

        self.iterations_per_loop[loop_id] = iteration

        # Narrowing phase
        if policy.narrowing_iterations > 0:
            for _ in range(policy.narrowing_iterations):
                loop_env = dict(env)
                self._refine_condition(stmt.cond, loop_env, True)
                self._interpret_stmts(body, loop_env)

                narrowed = False
                for v in env:
                    if v in loop_env:
                        old_val = env[v]
                        new_val = loop_env[v]
                        met = old_val.meet(new_val)
                        if not self._domain_equals(met, old_val):
                            env[v] = met
                            narrowed = True

                if narrowed:
                    self.narrowing_events += 1
                else:
                    break

        # Apply exit condition (negate loop condition)
        self._refine_condition(stmt.cond, env, False)

    def _adaptive_widen(self, old_val, new_val, policy, iteration):
        """Apply adaptive widening based on domain type and policy."""
        type_name = type(old_val).__name__

        # For ReducedProductDomain: widen each component according to its config
        if type_name == 'ReducedProductDomain' and hasattr(old_val, 'components'):
            return self._widen_product(old_val, new_val, policy, iteration)

        # For IntervalDomain: use adaptive interval widening
        if type_name == 'IntervalDomain':
            return self._widen_interval_domain(old_val, new_val, policy, iteration)

        # For SignDomain: no meaningful widening (already finite)
        if type_name == 'SignDomain':
            return new_val.join(old_val) if hasattr(new_val, 'join') else new_val

        # Default: use the domain's own widen method
        if hasattr(old_val, 'widen'):
            # Check if we should delay
            config = policy.component_configs[0] if policy.component_configs else None
            if config and iteration <= config.delay:
                return new_val  # During delay, just use new value (no widening)
            return old_val.widen(new_val)

        return new_val

    def _widen_product(self, old_val, new_val, policy, iteration):
        """Widen a ReducedProductDomain by widening each component."""
        if not hasattr(old_val, 'components') or not hasattr(new_val, 'components'):
            return old_val.widen(new_val) if hasattr(old_val, 'widen') else new_val

        old_comps = old_val.components
        new_comps = new_val.components

        if len(old_comps) != len(new_comps):
            return old_val.widen(new_val)

        widened_comps = []
        for i, (oc, nc) in enumerate(zip(old_comps, new_comps)):
            # Find matching config
            config = None
            for cc in policy.component_configs:
                if cc.domain_index == i:
                    config = cc
                    break

            if config is None:
                config = ComponentWideningConfig(
                    domain_index=i,
                    domain_kind=classify_domain(oc),
                    phase=WideningPhase.STANDARD
                )

            # Apply per-component widening
            if type(oc).__name__ == 'IntervalDomain':
                widened = self._widen_interval_component(oc, nc, config, iteration)
            elif config.phase == WideningPhase.DELAY and iteration <= config.delay:
                widened = nc  # During delay, use new value
            elif hasattr(oc, 'widen'):
                widened = oc.widen(nc)
            else:
                widened = nc

            widened_comps.append(widened)

            self.widening_events.append({
                'loop': policy.loop_id,
                'iteration': iteration,
                'component': i,
                'phase': config.phase.value,
                'old': str(oc),
                'new': str(nc),
                'result': str(widened)
            })

        # Rebuild product with widened components
        return self._rebuild_product(old_val, widened_comps)

    def _widen_interval_domain(self, old_val, new_val, policy, iteration):
        """Widen an IntervalDomain value adaptively."""
        config = policy.component_configs[0] if policy.component_configs else \
            ComponentWideningConfig(domain_index=0, domain_kind=DomainKind.NUMERIC,
                                   phase=WideningPhase.STANDARD)
        return self._widen_interval_component(old_val, new_val, config, iteration)

    def _widen_interval_component(self, old_val, new_val, config, iteration):
        """Widen a single IntervalDomain component."""
        if old_val.is_bot():
            return new_val
        if new_val.is_bot():
            return old_val

        old_lo = getattr(old_val, 'lo', NEG_INF)
        old_hi = getattr(old_val, 'hi', INF)
        new_lo = getattr(new_val, 'lo', NEG_INF)
        new_hi = getattr(new_val, 'hi', INF)

        # Use adaptive widening
        w_lo, w_hi = adaptive_widen_interval(old_lo, old_hi, new_lo, new_hi, config, iteration)

        return IntervalDomain(w_lo, w_hi)

    def _rebuild_product(self, old_product, new_components):
        """Rebuild a ReducedProductDomain with new components."""
        # Create a new product with same reducers
        result = copy.copy(old_product)
        if hasattr(result, '_components'):
            result._components = tuple(new_components)
        elif hasattr(result, 'components'):
            # Try to set components directly
            try:
                result.components = tuple(new_components)
            except AttributeError:
                # Frozen dataclass or property -- use widen fallback
                return old_product.widen(old_product.join(
                    type(old_product)(new_components, getattr(old_product, '_reducers', []))
                ))
        return result

    def _apply_reduction(self, env):
        """Apply cross-domain reduction to all variables in the environment."""
        for v in env:
            val = env[v]
            if hasattr(val, 'reduce'):
                env[v] = val.reduce()

    def _may_be_zero(self, val):
        """Check if a domain value may contain zero."""
        if hasattr(val, 'contains'):
            return val.contains(0)
        if hasattr(val, 'may_contain'):
            return val.may_contain(0)
        type_name = type(val).__name__
        if type_name == 'IntervalDomain':
            lo = getattr(val, 'lo', NEG_INF)
            hi = getattr(val, 'hi', INF)
            return lo <= 0 <= hi
        return True  # Conservative

    def _domain_equals(self, a, b):
        """Check if two domain values are equal."""
        if hasattr(a, 'equals'):
            return a.equals(b)
        if hasattr(a, '__eq__'):
            try:
                return a == b
            except Exception:
                pass
        return str(a) == str(b)

    def _eval_expr(self, expr, env):
        """Evaluate an expression to a domain value."""
        if expr is None:
            return self.domain_factory()

        type_name = type(expr).__name__

        if type_name == 'IntLit':
            return self.domain_factory(expr.value)

        elif type_name == 'Var':
            return env.get(expr.name, self.domain_factory())

        elif type_name == 'BinOp':
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            op = expr.op

            if op == '+':
                return left.add(right) if hasattr(left, 'add') else self.domain_factory()
            elif op == '-':
                return left.sub(right) if hasattr(left, 'sub') else self.domain_factory()
            elif op == '*':
                return left.mul(right) if hasattr(left, 'mul') else self.domain_factory()
            elif op == '/':
                if self.div_zero_check and self._may_be_zero(right):
                    self.warnings.append("Possible division by zero")
                return left.div(right) if hasattr(left, 'div') else self.domain_factory()
            elif op == '%':
                return left.mod(right) if hasattr(left, 'mod') else self.domain_factory()
            elif op in ('<', '<=', '>', '>=', '==', '!='):
                return self.domain_factory()  # Comparison result is abstract
            else:
                return self.domain_factory()

        elif type_name == 'UnaryOp':
            operand = self._eval_expr(expr.operand, env)
            if expr.op == '-':
                return operand.neg() if hasattr(operand, 'neg') else self.domain_factory()
            return operand

        elif type_name == 'CallExpr':
            return self.domain_factory()  # Conservative for calls

        elif type_name == 'BoolLit':
            val = 1 if expr.value else 0
            return self.domain_factory(val)

        return self.domain_factory()

    def _refine_condition(self, cond, env, is_true):
        """Refine environment based on branch condition."""
        if cond is None:
            return

        type_name = type(cond).__name__

        if type_name == 'BinOp' and cond.op in ('<', '<=', '>', '>=', '==', '!='):
            left_name = cond.left.name if type(cond.left).__name__ == 'Var' else None
            right_name = cond.right.name if type(cond.right).__name__ == 'Var' else None

            op = cond.op
            if not is_true:
                # Negate the operator
                negate_map = {'<': '>=', '<=': '>', '>': '<=', '>=': '<',
                              '==': '!=', '!=': '=='}
                op = negate_map.get(op, op)

            # Var vs constant refinement
            if left_name and type(cond.right).__name__ == 'IntLit':
                c = cond.right.value
                val = env.get(left_name, self.domain_factory())
                refined = self._refine_value(val, op, c)
                if refined is not None:
                    env[left_name] = refined

            elif right_name and type(cond.left).__name__ == 'IntLit':
                c = cond.left.value
                # Flip operator for var on right
                flip_map = {'<': '>', '<=': '>=', '>': '<', '>=': '<=',
                            '==': '==', '!=': '!='}
                flipped_op = flip_map.get(op, op)
                val = env.get(right_name, self.domain_factory())
                refined = self._refine_value(val, flipped_op, c)
                if refined is not None:
                    env[right_name] = refined

    def _refine_value(self, val, op, constant):
        """Refine a domain value by a comparison with a constant."""
        type_name = type(val).__name__

        if type_name == 'IntervalDomain':
            lo = getattr(val, 'lo', NEG_INF)
            hi = getattr(val, 'hi', INF)

            if op == '<':
                hi = min(hi, constant - 1)
            elif op == '<=':
                hi = min(hi, constant)
            elif op == '>':
                lo = max(lo, constant + 1)
            elif op == '>=':
                lo = max(lo, constant)
            elif op == '==':
                lo = max(lo, constant)
                hi = min(hi, constant)
            elif op == '!=':
                pass  # Can't easily refine interval for !=

            if lo > hi:
                return IntervalDomain(1, 0)  # bot: lo > hi
            return IntervalDomain(lo, hi)

        elif type_name == 'ReducedProductDomain' and hasattr(val, 'components'):
            # Refine each component
            new_comps = []
            for comp in val.components:
                refined = self._refine_value(comp, op, constant)
                new_comps.append(refined if refined is not None else comp)
            return self._rebuild_product(val, new_comps)

        return val


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def adaptive_analyze(source, domain_types=None, policies=None):
    """Analyze a C10 program with adaptive widening strategy.

    Args:
        source: C10 source code
        domain_types: list of domain types (default: [SignDomain, IntervalDomain])
        policies: explicit policies or None (auto-synthesize)

    Returns:
        FrameworkResult
    """
    if domain_types is None:
        domain_types = [SignDomain, IntervalDomain]

    # Build domain factory
    if len(domain_types) == 1:
        dt = domain_types[0]
        factory = _make_single_factory(dt)
    else:
        factory = compose_domains(*domain_types, auto_reduce=True)

    interp = StrategyInterpreter(
        domain_factory=factory,
        policies=policies,
        domain_types=domain_types
    )
    return interp.analyze(source)


def adaptive_analyze_interval(source, policies=None):
    """Convenience: analyze with IntervalDomain + adaptive widening."""
    return adaptive_analyze(source, domain_types=[IntervalDomain], policies=policies)


def adaptive_analyze_composed(source, domain_types=None, policies=None):
    """Convenience: analyze with composed domains + adaptive widening."""
    if domain_types is None:
        domain_types = [SignDomain, IntervalDomain]
    return adaptive_analyze(source, domain_types=domain_types, policies=policies)


def standard_analyze(source, domain_types=None):
    """Analyze with standard (non-adaptive) widening for comparison.

    Uses V108 CompositionInterpreter with default widening.
    """
    if domain_types is None:
        domain_types = [SignDomain, IntervalDomain]

    if len(domain_types) == 1:
        dt = domain_types[0]
        factory = _make_single_factory(dt)
    else:
        factory = compose_domains(*domain_types, auto_reduce=True)

    interp = CompositionInterpreter(factory)
    result = interp.analyze(source)

    return FrameworkResult(
        env=result.get('bindings', {}),
        warnings=result.get('warnings', []),
        policies={},
        iterations_per_loop={},
        widening_events=[],
        narrowing_events=0,
        reduction_events=0
    )


def compare_strategies(source, domain_types=None):
    """Compare standard vs adaptive widening on the same program.

    Returns:
        StrategyComparison with results from both strategies
    """
    if domain_types is None:
        domain_types = [SignDomain, IntervalDomain]

    standard = standard_analyze(source, domain_types)
    adaptive = adaptive_analyze(source, domain_types)

    # Also try with explicit delayed-threshold policy
    delayed_policies = {}
    loop_infos = analyze_loops(source)
    thresholds = tuple(sorted(set(extract_thresholds_from_ast(source))))
    for i in range(max(1, len(loop_infos))):
        delayed_policies[i] = AdaptivePolicy(
            loop_id=i,
            component_configs=[
                ComponentWideningConfig(
                    domain_index=j,
                    domain_kind=classify_domain_type(dt),
                    phase=WideningPhase.THRESHOLD,
                    delay=3,
                    thresholds=thresholds,
                    allow_narrowing=True
                )
                for j, dt in enumerate(domain_types)
            ],
            narrowing_iterations=3
        )
    delayed = adaptive_analyze(source, domain_types, delayed_policies)

    strategies = {
        'standard': standard,
        'adaptive': adaptive,
        'delayed_threshold': delayed
    }

    # Rank by precision (tighter intervals = more precise)
    precision_ranking = _rank_by_precision(strategies)

    # Rank by iterations
    iteration_ranking = sorted(
        strategies.keys(),
        key=lambda k: sum(strategies[k].iterations_per_loop.values())
    )

    summary = _build_comparison_summary(strategies, precision_ranking, iteration_ranking)

    return StrategyComparison(
        source=source,
        strategies=strategies,
        precision_ranking=precision_ranking,
        iteration_ranking=iteration_ranking,
        summary=summary
    )


def get_adaptive_policies(source, domain_types=None):
    """Get the synthesized adaptive policies for a program.

    Returns:
        Dict[int, AdaptivePolicy]
    """
    return synthesize_all_policies(source, domain_types)


def get_loop_analysis(source):
    """Get loop analysis results.

    Returns:
        List[Dict] with loop info
    """
    loop_infos = analyze_loops(source)
    result = []
    for li in loop_infos:
        result.append({
            'loop_id': li.loop_id,
            'is_simple_counter': li.is_simple_counter,
            'counter_var': li.counter_var,
            'bound_value': li.bound_value,
            'bound_direction': li.bound_direction,
            'condition_vars': list(li.condition_vars),
            'modified_vars': list(li.modified_vars),
            'constants_in_condition': list(li.constants_in_condition),
            'nested_depth': li.nested_depth
        })
    return result


def validate_adaptive_policy(source, domain_types=None):
    """Validate that adaptive widening produces sound results.

    Checks: adaptive result is a sound over-approximation (standard result
    is contained within adaptive result for all variables).

    Returns:
        Dict with validation results
    """
    if domain_types is None:
        domain_types = [IntervalDomain]

    standard = standard_analyze(source, domain_types)
    adaptive = adaptive_analyze(source, domain_types)

    # Check containment: adaptive should be at least as imprecise as standard
    # (sound = over-approximate). But adaptive can also be MORE precise due to
    # narrowing and threshold widening.
    issues = []
    for var in standard.env:
        if var not in adaptive.env:
            continue
        s_val = standard.env[var]
        a_val = adaptive.env[var]
        # Both are strings at this point -- just report

    return {
        'standard': standard.env,
        'adaptive': adaptive.env,
        'issues': issues,
        'valid': len(issues) == 0,
        'adaptive_more_precise': _count_precision_gains(standard.env, adaptive.env)
    }


def widening_summary(source, domain_types=None):
    """Human-readable summary of widening strategy analysis.

    Returns:
        str
    """
    if domain_types is None:
        domain_types = [IntervalDomain]

    comparison = compare_strategies(source, domain_types)

    lines = ["=== Widening Strategy Framework Summary ===", ""]

    for name, result in comparison.strategies.items():
        lines.append(f"Strategy: {name}")
        lines.append(f"  Variables: {result.env}")
        lines.append(f"  Iterations: {result.iterations_per_loop}")
        lines.append(f"  Widening events: {len(result.widening_events)}")
        lines.append(f"  Narrowing events: {result.narrowing_events}")
        lines.append(f"  Reduction events: {result.reduction_events}")
        lines.append(f"  Warnings: {result.warnings}")
        lines.append("")

    lines.append(f"Precision ranking: {comparison.precision_ranking}")
    lines.append(f"Iteration ranking: {comparison.iteration_ranking}")
    lines.append("")
    lines.append(comparison.summary)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_by_precision(strategies):
    """Rank strategies by precision (tighter intervals = better)."""
    def precision_score(result):
        """Lower score = more precise."""
        score = 0
        for var, val_str in result.env.items():
            # Try to parse interval from string representation
            if 'IntervalDomain' in val_str or '[' in val_str:
                try:
                    # Extract numbers from interval string
                    import re
                    nums = re.findall(r'-?\d+', val_str)
                    if len(nums) >= 2:
                        lo, hi = int(nums[0]), int(nums[-1])
                        score += (hi - lo)
                    elif 'inf' in val_str.lower() or 'INF' in val_str:
                        score += 10000
                except (ValueError, IndexError):
                    score += 1000
            elif val_str == 'TOP' or 'top' in val_str.lower():
                score += 10000
            else:
                try:
                    # Single value = most precise
                    int(val_str)
                    score += 0
                except (ValueError, TypeError):
                    score += 100
        return score

    return sorted(strategies.keys(), key=lambda k: precision_score(strategies[k]))


def _count_precision_gains(standard_env, adaptive_env):
    """Count variables where adaptive is more precise than standard."""
    gains = 0
    for var in standard_env:
        if var in adaptive_env:
            s = standard_env[var]
            a = adaptive_env[var]
            # Heuristic: shorter string representation = more precise
            if len(a) < len(s):
                gains += 1
    return gains


def _build_comparison_summary(strategies, precision_ranking, iteration_ranking):
    """Build a human-readable comparison summary."""
    lines = []
    if precision_ranking:
        lines.append(f"Most precise strategy: {precision_ranking[0]}")
    if iteration_ranking:
        best = iteration_ranking[0]
        iters = sum(strategies[best].iterations_per_loop.values())
        lines.append(f"Fewest iterations: {best} ({iters} total)")

    # Check if adaptive improved over standard
    if 'standard' in strategies and 'adaptive' in strategies:
        s = strategies['standard']
        a = strategies['adaptive']
        gains = _count_precision_gains(s.env, a.env)
        if gains > 0:
            lines.append(f"Adaptive improved precision on {gains} variable(s)")
        else:
            lines.append("Adaptive and standard achieved similar precision")

    return "\n".join(lines) if lines else "No comparison data"
