"""V135: Effect-Typed Program Synthesis

Composes V040 (effect systems) + C097 (program synthesis) to synthesize
programs that satisfy both I/O specifications and effect constraints.

Key idea: effect constraints prune the synthesis search space by filtering
out components/operators that would violate effect requirements.
Post-synthesis verification confirms the synthesized program has correct effects.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from enum import Enum

# C097: Program Synthesis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C097_program_synthesis'))
from synthesis import (
    synthesize, SynthesisResult, SynthesisSpec, IOExample,
    Expr, IntConst, BoolConst, VarExpr, UnaryOp, BinOp, IfExpr,
    evaluate, expr_size, expr_depth, simplify, pretty_print,
    EnumerativeSynthesizer, ConstraintSynthesizer, CEGISSynthesizer,
    ComponentSynthesizer, ConditionalSynthesizer,
    Type,
)

# V040: Effect Systems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V040_effect_systems'))
from effect_systems import (
    EffectKind, Effect, EffectSet,
    PURE, IO, DIV, NONDET, State, Exn,
    infer_effects, effect_subtype,
)

# C010: Parser (for effect inference on generated code)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EffectConstraint(str, Enum):
    """Effect constraints for synthesis."""
    PURE = "pure"           # No effects at all
    NO_STATE = "no_state"   # No state mutation
    NO_IO = "no_io"         # No I/O
    NO_EXN = "no_exn"       # No exceptions
    NO_DIV = "no_div"       # No divergence
    ALLOW_STATE = "allow_state"  # State mutation permitted
    ALLOW_EXN = "allow_exn"      # Exceptions permitted


@dataclass
class EffectSpec:
    """Effect specification for synthesis."""
    allowed: EffectSet = field(default_factory=lambda: EffectSet.pure())
    forbidden_kinds: Set[EffectKind] = field(default_factory=set)

    @staticmethod
    def pure():
        return EffectSpec(allowed=EffectSet.pure(), forbidden_kinds={
            EffectKind.STATE, EffectKind.IO, EffectKind.EXN,
            EffectKind.DIV, EffectKind.NONDET
        })

    @staticmethod
    def no_io():
        return EffectSpec(
            allowed=EffectSet.of(State("x"), Exn(), Effect(EffectKind.DIV)),
            forbidden_kinds={EffectKind.IO}
        )

    @staticmethod
    def no_state():
        return EffectSpec(
            allowed=EffectSet.of(IO, Exn(), Effect(EffectKind.DIV)),
            forbidden_kinds={EffectKind.STATE}
        )

    @staticmethod
    def no_exn():
        return EffectSpec(
            allowed=EffectSet.of(IO, State("x"), Effect(EffectKind.DIV)),
            forbidden_kinds={EffectKind.EXN}
        )

    @staticmethod
    def total():
        """No divergence allowed (must terminate)."""
        return EffectSpec(
            allowed=EffectSet.of(IO, State("x"), Exn()),
            forbidden_kinds={EffectKind.DIV}
        )

    @staticmethod
    def unrestricted():
        return EffectSpec(allowed=EffectSet.of(
            IO, State("x"), Exn(), Effect(EffectKind.DIV), NONDET
        ), forbidden_kinds=set())

    def is_allowed(self, effect_kind: EffectKind) -> bool:
        return effect_kind not in self.forbidden_kinds

    def check_effects(self, effects: EffectSet) -> bool:
        """Check if an effect set satisfies this spec."""
        for e in effects.effects:
            if e.kind in self.forbidden_kinds:
                return False
        return True


@dataclass
class EffectSynthesisResult:
    """Result of effect-typed synthesis."""
    success: bool
    program: Optional[Expr] = None
    program_str: Optional[str] = None
    inferred_effects: Optional[EffectSet] = None
    effect_spec: Optional[EffectSpec] = None
    effect_satisfied: bool = False
    io_satisfied: bool = False
    method: str = ""
    candidates_explored: int = 0
    candidates_filtered: int = 0  # Filtered by effect constraints
    iterations: int = 0


# ---------------------------------------------------------------------------
# Effect analysis of DSL expressions
# ---------------------------------------------------------------------------

def _expr_effects(expr: Expr) -> EffectSet:
    """Infer effects of a DSL expression.

    Pure expressions: constants, variables, arithmetic (+, -, *, max, min, abs, neg),
    comparisons, boolean ops, conditionals (if-then-else).

    Potentially effectful:
    - Division (/) or modulo (%): may raise DivByZero -> Exn
    """
    if isinstance(expr, (IntConst, BoolConst, VarExpr)):
        return EffectSet.pure()
    elif isinstance(expr, UnaryOp):
        return _expr_effects(expr.arg)
    elif isinstance(expr, BinOp):
        left_eff = _expr_effects(expr.left)
        right_eff = _expr_effects(expr.right)
        combined = left_eff.union(right_eff)
        if expr.op in ('/', '%'):
            # Division may raise exception
            combined = combined.union(EffectSet.of(Exn("DivByZero")))
        return combined
    elif isinstance(expr, IfExpr):
        cond_eff = _expr_effects(expr.cond)
        then_eff = _expr_effects(expr.then_expr)
        else_eff = _expr_effects(expr.else_expr)
        return cond_eff.union(then_eff).union(else_eff)
    return EffectSet.pure()


def _check_expr_effect_constraint(expr: Expr, spec: EffectSpec) -> bool:
    """Check if an expression satisfies an effect constraint."""
    effects = _expr_effects(expr)
    return spec.check_effects(effects)


# ---------------------------------------------------------------------------
# Effect-aware component filtering
# ---------------------------------------------------------------------------

# Standard component sets
PURE_ARITHMETIC = ['+', '-', '*', 'max', 'min']
EFFECTFUL_ARITHMETIC = ['+', '-', '*', '/', '%', 'max', 'min']
COMPARISON_OPS = ['==', '!=', '<', '<=', '>', '>=']
BOOLEAN_OPS = ['and', 'or']
UNARY_OPS = ['neg', 'abs', 'not']

def _filter_components(components: List[str], spec: EffectSpec) -> List[str]:
    """Filter components that would violate effect constraints."""
    if spec.is_allowed(EffectKind.EXN):
        return components  # No filtering needed

    # Remove division/modulo if exceptions forbidden
    filtered = [c for c in components if c not in ('/', '%')]
    return filtered


def _get_default_components(spec: EffectSpec) -> List[str]:
    """Get default components respecting effect constraints."""
    if EffectKind.EXN in spec.forbidden_kinds:
        return PURE_ARITHMETIC
    return EFFECTFUL_ARITHMETIC


# ---------------------------------------------------------------------------
# Core: Effect-typed synthesis
# ---------------------------------------------------------------------------

def effect_typed_synthesize(examples, input_vars, effect_spec=None,
                             method="enumerative", components=None,
                             constants=None, max_size=10, max_depth=4):
    """Synthesize a program satisfying both I/O examples and effect constraints.

    Args:
        examples: list of (inputs_dict, output) or IOExample objects
        input_vars: list of variable names
        effect_spec: EffectSpec (default: pure)
        method: synthesis method ("enumerative", "constraint", "cegis", etc.)
        components: list of operators to use (filtered by effect spec)
        constants: list of constants
        max_size: max AST size
        max_depth: max AST depth

    Returns:
        EffectSynthesisResult
    """
    if effect_spec is None:
        effect_spec = EffectSpec.pure()

    # Filter components by effect constraints
    if components is None:
        components = _get_default_components(effect_spec)
    else:
        components = _filter_components(components, effect_spec)

    # Normalize examples
    io_examples = []
    for ex in examples:
        if isinstance(ex, IOExample):
            io_examples.append(ex)
        elif isinstance(ex, (tuple, list)) and len(ex) == 2:
            inputs, output = ex
            if isinstance(inputs, dict):
                io_examples.append(IOExample(inputs=inputs, output=output))
            else:
                # Single input
                io_examples.append(IOExample(
                    inputs={input_vars[0]: inputs}, output=output
                ))
        else:
            raise ValueError(f"Invalid example format: {ex}")

    # Run synthesis
    result = synthesize(
        examples=io_examples,
        input_vars=input_vars,
        method=method,
        components=components,
        constants=constants,
        max_size=max_size,
        max_depth=max_depth,
    )

    # Build effect synthesis result
    eff_result = EffectSynthesisResult(
        success=result.success,
        program=result.program if result.success else None,
        method=result.method,
        candidates_explored=result.candidates_explored,
        iterations=result.iterations,
        effect_spec=effect_spec,
    )

    if result.success and result.program is not None:
        # Verify I/O correctness
        eff_result.io_satisfied = _verify_io(result.program, io_examples, input_vars)

        # Infer effects of synthesized program
        eff_result.inferred_effects = _expr_effects(result.program)
        eff_result.effect_satisfied = effect_spec.check_effects(eff_result.inferred_effects)

        # Convert to readable string
        eff_result.program_str = _expr_to_source(result.program)

        # If effect constraint violated, try to find an alternative
        if not eff_result.effect_satisfied:
            eff_result = _retry_with_effect_filter(
                io_examples, input_vars, effect_spec, components,
                constants, max_size, max_depth, eff_result
            )

    return eff_result


def _retry_with_effect_filter(io_examples, input_vars, effect_spec,
                               components, constants, max_size, max_depth,
                               original_result):
    """Retry synthesis with stricter component filtering if effect check failed."""
    # Try enumerative with smaller size to find effect-safe variant
    strict_components = _filter_components(components, effect_spec)

    # Use enumerative to find ALL candidates and pick effect-safe one
    spec = SynthesisSpec(
        examples=io_examples,
        input_vars=input_vars,
        components=strict_components,
        constants=constants or [0, 1],
    )

    synth = EnumerativeSynthesizer(spec, max_size=max_size, max_depth=max_depth)
    result = synth.synthesize()

    if result.success and result.program is not None:
        effects = _expr_effects(result.program)
        if effect_spec.check_effects(effects):
            return EffectSynthesisResult(
                success=True,
                program=result.program,
                program_str=_expr_to_source(result.program),
                inferred_effects=effects,
                effect_spec=effect_spec,
                effect_satisfied=True,
                io_satisfied=True,
                method=f"{result.method}+effect_retry",
                candidates_explored=original_result.candidates_explored + result.candidates_explored,
                candidates_filtered=original_result.candidates_filtered + 1,
                iterations=original_result.iterations + result.iterations,
            )

    # Return original (with effect_satisfied=False)
    return original_result


def _verify_io(program: Expr, examples: List[IOExample], input_vars: List[str]) -> bool:
    """Verify that a program satisfies all I/O examples."""
    for ex in examples:
        try:
            result = evaluate(program, ex.inputs)
            if result != ex.output:
                return False
        except Exception:
            return False
    return True


def _expr_to_source(expr: Expr) -> str:
    """Convert DSL Expr to human-readable source string."""
    if isinstance(expr, IntConst):
        return str(expr.value)
    elif isinstance(expr, BoolConst):
        return "true" if expr.value else "false"
    elif isinstance(expr, VarExpr):
        return expr.name
    elif isinstance(expr, UnaryOp):
        if expr.op == 'neg':
            return f"(0 - {_expr_to_source(expr.arg)})"
        elif expr.op == 'abs':
            return f"abs({_expr_to_source(expr.arg)})"
        elif expr.op == 'not':
            return f"!{_expr_to_source(expr.arg)}"
        return f"{expr.op}({_expr_to_source(expr.arg)})"
    elif isinstance(expr, BinOp):
        left = _expr_to_source(expr.left)
        right = _expr_to_source(expr.right)
        if expr.op in ('max', 'min'):
            return f"{expr.op}({left}, {right})"
        return f"({left} {expr.op} {right})"
    elif isinstance(expr, IfExpr):
        cond = _expr_to_source(expr.cond)
        then = _expr_to_source(expr.then_expr)
        els = _expr_to_source(expr.else_expr)
        return f"if ({cond}) {{ {then} }} else {{ {els} }}"
    return str(expr)


# ---------------------------------------------------------------------------
# Effect-aware synthesis variants
# ---------------------------------------------------------------------------

def synthesize_pure(examples, input_vars, method="enumerative",
                     constants=None, max_size=10, max_depth=4):
    """Synthesize a pure (no effects) program."""
    return effect_typed_synthesize(
        examples, input_vars,
        effect_spec=EffectSpec.pure(),
        method=method,
        components=PURE_ARITHMETIC,
        constants=constants,
        max_size=max_size,
        max_depth=max_depth,
    )


def synthesize_total(examples, input_vars, method="enumerative",
                      constants=None, max_size=10, max_depth=4):
    """Synthesize a total (must terminate, no divergence) program."""
    return effect_typed_synthesize(
        examples, input_vars,
        effect_spec=EffectSpec.total(),
        method=method,
        constants=constants,
        max_size=max_size,
        max_depth=max_depth,
    )


def synthesize_safe(examples, input_vars, method="enumerative",
                     constants=None, max_size=10, max_depth=4):
    """Synthesize a safe (no exceptions) program."""
    return effect_typed_synthesize(
        examples, input_vars,
        effect_spec=EffectSpec.no_exn(),
        method=method,
        constants=constants,
        max_size=max_size,
        max_depth=max_depth,
    )


def synthesize_with_effects(examples, input_vars, allowed_effects,
                              method="enumerative", constants=None,
                              max_size=10, max_depth=4):
    """Synthesize with explicit allowed effect set."""
    forbidden = set()
    all_kinds = {EffectKind.STATE, EffectKind.IO, EffectKind.EXN,
                 EffectKind.DIV, EffectKind.NONDET}
    for kind in all_kinds:
        if not any(e.kind == kind for e in allowed_effects.effects):
            forbidden.add(kind)

    spec = EffectSpec(allowed=allowed_effects, forbidden_kinds=forbidden)
    return effect_typed_synthesize(
        examples, input_vars,
        effect_spec=spec,
        method=method,
        constants=constants,
        max_size=max_size,
        max_depth=max_depth,
    )


# ---------------------------------------------------------------------------
# Source-level effect verification of synthesized programs
# ---------------------------------------------------------------------------

def verify_synthesized_effects(program: Expr, effect_spec: EffectSpec) -> Dict[str, Any]:
    """Verify the effects of a synthesized program expression.

    Returns a dict with effect analysis results.
    """
    effects = _expr_effects(program)
    satisfied = effect_spec.check_effects(effects)

    effect_list = []
    for e in effects.effects:
        if e.kind != EffectKind.PURE:
            effect_list.append({"kind": e.kind.value, "detail": e.detail})

    violations = []
    for e in effects.effects:
        if e.kind in effect_spec.forbidden_kinds:
            violations.append({"kind": e.kind.value, "detail": e.detail})

    return {
        "satisfied": satisfied,
        "effects": effect_list,
        "violations": violations,
        "is_pure": effects.is_pure,
        "effect_count": len(effect_list),
    }


def source_level_effect_check(source: str, effect_spec: EffectSpec) -> Dict[str, Any]:
    """Run V040 effect inference on C10 source and check against spec."""
    try:
        sigs = infer_effects(source)
    except Exception as e:
        return {"error": str(e), "satisfied": False}

    results = {}
    all_satisfied = True
    for fn_name, sig in sigs.items():
        fn_satisfied = effect_spec.check_effects(sig.effects)
        if not fn_satisfied:
            all_satisfied = False
        results[fn_name] = {
            "effects": [{"kind": e.kind.value, "detail": e.detail} for e in sig.effects.effects if e.kind != EffectKind.PURE],
            "satisfied": fn_satisfied,
        }

    return {"functions": results, "all_satisfied": all_satisfied}


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

def compare_with_unrestricted(examples, input_vars, effect_spec=None,
                                constants=None, max_size=10):
    """Compare effect-typed synthesis vs unrestricted synthesis."""
    if effect_spec is None:
        effect_spec = EffectSpec.pure()

    # Unrestricted synthesis
    unrestricted = effect_typed_synthesize(
        examples, input_vars,
        effect_spec=EffectSpec.unrestricted(),
        components=EFFECTFUL_ARITHMETIC,
        constants=constants,
        max_size=max_size,
    )

    # Effect-constrained synthesis
    constrained = effect_typed_synthesize(
        examples, input_vars,
        effect_spec=effect_spec,
        constants=constants,
        max_size=max_size,
    )

    return {
        "unrestricted": {
            "success": unrestricted.success,
            "program": unrestricted.program_str,
            "effects": [e.kind.value for e in (unrestricted.inferred_effects.effects if unrestricted.inferred_effects else []) if e.kind != EffectKind.PURE],
            "size": expr_size(unrestricted.program) if unrestricted.program else 0,
            "candidates": unrestricted.candidates_explored,
        },
        "constrained": {
            "success": constrained.success,
            "program": constrained.program_str,
            "effects": [e.kind.value for e in (constrained.inferred_effects.effects if constrained.inferred_effects else []) if e.kind != EffectKind.PURE],
            "effect_satisfied": constrained.effect_satisfied,
            "size": expr_size(constrained.program) if constrained.program else 0,
            "candidates": constrained.candidates_explored,
        },
        "both_succeeded": unrestricted.success and constrained.success,
        "constraint_spec": [k.value for k in effect_spec.forbidden_kinds],
    }


def effect_synthesis_summary(result: EffectSynthesisResult) -> Dict[str, Any]:
    """Get a concise summary of an effect synthesis result."""
    effects = []
    if result.inferred_effects:
        effects = [e.kind.value for e in result.inferred_effects.effects if e.kind != EffectKind.PURE]

    return {
        "success": result.success,
        "program": result.program_str,
        "effects": effects,
        "is_pure": result.inferred_effects.is_pure if result.inferred_effects else None,
        "effect_satisfied": result.effect_satisfied,
        "io_satisfied": result.io_satisfied,
        "method": result.method,
        "candidates_explored": result.candidates_explored,
        "candidates_filtered": result.candidates_filtered,
        "program_size": expr_size(result.program) if result.program else 0,
    }
