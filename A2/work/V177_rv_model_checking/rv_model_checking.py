"""V177: Runtime Verification + LTL Model Checking Composition

Bridges V176 (runtime verification monitors) with V023 (LTL model checking)
to create a unified verification system that can:

1. Translate formulas between runtime monitor and model checker representations
2. Verify properties both online (monitoring traces) and offline (model checking)
3. Extract system models from execution traces for model checking
4. Use model checking counterexamples to create targeted monitors
5. Mine specifications from traces, then verify them exhaustively
6. Check trace conformance against system models

Composes: V176 (runtime_monitor) + V023 (ltl_model_checker)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable, FrozenSet
from enum import Enum, auto
from collections import defaultdict
from functools import reduce

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V176_runtime_verification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))

import runtime_monitor as rv
import ltl_model_checker as mc


# ============================================================
# Section 1: Formula Bridge (V176 <-> V023 translation)
# ============================================================

class FormulaTranslationError(Exception):
    """Raised when a formula cannot be translated between representations."""
    pass


def rv_to_mc(formula: rv.Formula) -> mc.LTL:
    """Translate V176 runtime monitor formula to V023 model checker LTL.

    Past-time operators (Y, O, H, S) cannot be translated -- model checking
    only supports future-time LTL.

    Bounded operators (F[k], G[k]) are expanded to nested Next/And/Or.
    """
    if isinstance(formula, rv.TrueF):
        return mc.LTLTrue()
    if isinstance(formula, rv.FalseF):
        return mc.LTLFalse()
    if isinstance(formula, rv.Atom):
        return mc.Atom(formula.predicate)
    if isinstance(formula, rv.Not):
        return mc.Not(rv_to_mc(formula.sub))
    if isinstance(formula, rv.And):
        return mc.And(rv_to_mc(formula.left), rv_to_mc(formula.right))
    if isinstance(formula, rv.Or):
        return mc.Or(rv_to_mc(formula.left), rv_to_mc(formula.right))
    if isinstance(formula, rv.Implies):
        return mc.Implies(rv_to_mc(formula.left), rv_to_mc(formula.right))
    if isinstance(formula, rv.Next):
        return mc.Next(rv_to_mc(formula.sub))
    if isinstance(formula, rv.Eventually):
        return mc.Finally(rv_to_mc(formula.sub))
    if isinstance(formula, rv.Always):
        return mc.Globally(rv_to_mc(formula.sub))
    if isinstance(formula, rv.Until):
        return mc.Until(rv_to_mc(formula.left), rv_to_mc(formula.right))
    if isinstance(formula, rv.Release):
        return mc.Release(rv_to_mc(formula.left), rv_to_mc(formula.right))
    if isinstance(formula, rv.BoundedEventually):
        return _expand_bounded_eventually(rv_to_mc(formula.sub), formula.bound)
    if isinstance(formula, rv.BoundedAlways):
        return _expand_bounded_always(rv_to_mc(formula.sub), formula.bound)
    if isinstance(formula, (rv.Previous, rv.Once, rv.Historically, rv.Since)):
        op_name = type(formula).__name__
        raise FormulaTranslationError(
            f"Past-time operator '{op_name}' cannot be translated to LTL model checker. "
            f"Model checking supports only future-time LTL (X, F, G, U, R)."
        )
    raise FormulaTranslationError(f"Unknown formula type: {type(formula)}")


def _expand_bounded_eventually(phi: mc.LTL, k: int) -> mc.LTL:
    """F[k] phi = phi || X(phi || X(phi || ...)) with k levels."""
    if k <= 0:
        return phi
    result = phi
    for _ in range(k):
        result = mc.Or(phi, mc.Next(result))
    return result


def _expand_bounded_always(phi: mc.LTL, k: int) -> mc.LTL:
    """G[k] phi = phi && X(phi && X(phi && ...)) with k levels."""
    if k <= 0:
        return phi
    result = phi
    for _ in range(k):
        result = mc.And(phi, mc.Next(result))
    return result


def mc_to_rv(formula: mc.LTL) -> rv.Formula:
    """Translate V023 model checker LTL to V176 runtime monitor formula."""
    op = formula.op
    if op == mc.LTLOp.TRUE:
        return rv.TrueF()
    if op == mc.LTLOp.FALSE:
        return rv.FalseF()
    if op == mc.LTLOp.ATOM:
        return rv.Atom(formula.name)
    if op == mc.LTLOp.NOT:
        return rv.Not(mc_to_rv(formula.left))
    if op == mc.LTLOp.AND:
        return rv.And(mc_to_rv(formula.left), mc_to_rv(formula.right))
    if op == mc.LTLOp.OR:
        return rv.Or(mc_to_rv(formula.left), mc_to_rv(formula.right))
    if op == mc.LTLOp.IMPLIES:
        return rv.Implies(mc_to_rv(formula.left), mc_to_rv(formula.right))
    if op == mc.LTLOp.IFF:
        a = mc_to_rv(formula.left)
        b = mc_to_rv(formula.right)
        return rv.And(rv.Implies(a, b), rv.Implies(b, a))
    if op == mc.LTLOp.X:
        return rv.Next(mc_to_rv(formula.left))
    if op == mc.LTLOp.F:
        return rv.Eventually(mc_to_rv(formula.left))
    if op == mc.LTLOp.G:
        return rv.Always(mc_to_rv(formula.left))
    if op == mc.LTLOp.U:
        return rv.Until(mc_to_rv(formula.left), mc_to_rv(formula.right))
    if op == mc.LTLOp.R:
        return rv.Release(mc_to_rv(formula.left), mc_to_rv(formula.right))
    if op == mc.LTLOp.W:
        a = mc_to_rv(formula.left)
        b = mc_to_rv(formula.right)
        return rv.Or(rv.Until(a, b), rv.Always(a))
    raise FormulaTranslationError(f"Unknown MC formula op: {op}")


def formulas_equivalent(f1: rv.Formula, f2: rv.Formula) -> bool:
    """Check structural equivalence of two RV formulas."""
    if type(f1) != type(f2):
        return False
    if isinstance(f1, rv.Atom):
        return f1.predicate == f2.predicate
    if isinstance(f1, (rv.TrueF, rv.FalseF)):
        return True
    if isinstance(f1, rv.Not):
        return formulas_equivalent(f1.sub, f2.sub)
    if isinstance(f1, (rv.And, rv.Or, rv.Implies, rv.Until, rv.Release, rv.Since)):
        return (formulas_equivalent(f1.left, f2.left) and
                formulas_equivalent(f1.right, f2.right))
    if isinstance(f1, (rv.Next, rv.Eventually, rv.Always, rv.Once,
                        rv.Historically, rv.Previous)):
        return formulas_equivalent(f1.sub, f2.sub)
    if isinstance(f1, (rv.BoundedEventually, rv.BoundedAlways)):
        return f1.bound == f2.bound and formulas_equivalent(f1.sub, f2.sub)
    return False


# ============================================================
# Section 2: Trace-to-Model Extraction
# ============================================================

@dataclass
class TraceState:
    """A state observed in a trace, represented as boolean propositions."""
    propositions: FrozenSet[str]
    step: int = 0

    def to_dict(self, all_props: Set[str]) -> Dict[str, bool]:
        return {p: (p in self.propositions) for p in all_props}


@dataclass
class ExtractedModel:
    """A boolean transition system extracted from traces."""
    state_vars: List[str]
    initial_states: List[Dict[str, bool]]
    transitions: List[Tuple[Dict[str, bool], Dict[str, bool]]]
    num_traces: int = 0
    num_events: int = 0


def events_to_propositions(event: rv.Event,
                           prop_extractors: Optional[Dict[str, Callable]] = None
                           ) -> FrozenSet[str]:
    """Convert a runtime event to a set of boolean propositions."""
    props = set()
    props.add(event.name if isinstance(event, rv.Event) else str(event))
    if prop_extractors:
        for prop_name, extractor in prop_extractors.items():
            evt = event if isinstance(event, rv.Event) else rv.Event(str(event))
            if extractor(evt):
                props.add(prop_name)
    return frozenset(props)


def extract_model_from_traces(
    traces: List[List[rv.Event]],
    prop_extractors: Optional[Dict[str, Callable]] = None,
    window_size: int = 1
) -> ExtractedModel:
    """Extract a boolean transition system from observed execution traces."""
    all_props = set()
    all_initial = []
    all_transitions = []

    for trace in traces:
        if not trace:
            continue

        prop_sets = []
        for evt in trace:
            if isinstance(evt, str):
                evt = rv.Event(evt)
            props = events_to_propositions(evt, prop_extractors)
            all_props.update(props)
            prop_sets.append(props)

        states = []
        for i in range(0, len(prop_sets), window_size):
            window = prop_sets[i:i + window_size]
            merged = frozenset().union(*window)
            states.append(merged)

        if states:
            all_initial.append(states[0])
            for i in range(len(states) - 1):
                all_transitions.append((states[i], states[i + 1]))

    state_vars = sorted(all_props)

    def props_to_dict(props):
        return {v: (v in props) for v in state_vars}

    initial_dicts = [props_to_dict(s) for s in all_initial]
    transition_dicts = [(props_to_dict(s1), props_to_dict(s2))
                        for s1, s2 in all_transitions]

    seen_init = set()
    unique_init = []
    for d in initial_dicts:
        key = frozenset(d.items())
        if key not in seen_init:
            seen_init.add(key)
            unique_init.append(d)

    seen_trans = set()
    unique_trans = []
    for s1, s2 in transition_dicts:
        key = (frozenset(s1.items()), frozenset(s2.items()))
        if key not in seen_trans:
            seen_trans.add(key)
            unique_trans.append((s1, s2))

    return ExtractedModel(
        state_vars=state_vars,
        initial_states=unique_init,
        transitions=unique_trans,
        num_traces=len(traces),
        num_events=sum(len(t) for t in traces)
    )


# ============================================================
# Section 3: BDD Model Builder (bridge dict-based models to V023)
# ============================================================

def _build_init_fn(init_map: Dict[str, bool]):
    """Build BDD init function from a boolean assignment dict."""
    def init_fn(bdd):
        result = bdd.TRUE
        for var, val in init_map.items():
            idx = bdd.var_index(var)
            v = bdd.var(idx)
            if val:
                result = bdd.AND(result, v)
            else:
                result = bdd.AND(result, bdd.NOT(v))
        return result
    return init_fn


def _build_trans_fn(state_vars: List[str], transitions: List[Dict]):
    """Build BDD trans function from condition/update dicts.

    Each transition: {'condition': {var: bool}, 'update': {var: bool}}
    Semantics: if condition matches current state, next state = update.
    Multiple transitions are OR'd (nondeterministic choice).
    """
    def trans_fn(bdd, current_vars, next_vars):
        if not transitions:
            return bdd.FALSE

        all_trans = bdd.FALSE
        for t in transitions:
            cond = t.get('condition', {})
            update = t.get('update', {})

            # Build condition on current vars
            clause = bdd.TRUE
            for var in state_vars:
                if var in cond:
                    idx = current_vars[var]
                    v = bdd.var(idx)
                    if cond[var]:
                        clause = bdd.AND(clause, v)
                    else:
                        clause = bdd.AND(clause, bdd.NOT(v))

            # Build update on next vars
            for var in state_vars:
                if var in update:
                    nv_name = f"{var}'"
                    idx = next_vars[nv_name]
                    v = bdd.var(idx)
                    if update[var]:
                        clause = bdd.AND(clause, v)
                    else:
                        clause = bdd.AND(clause, bdd.NOT(v))

            all_trans = bdd.OR(all_trans, clause)

        return all_trans
    return trans_fn


# ============================================================
# Section 4: Dual-Mode Verifier
# ============================================================

class VerificationMode(Enum):
    MODEL_CHECK = auto()
    MONITOR = auto()
    DUAL = auto()


@dataclass
class DualResult:
    """Result from dual-mode verification."""
    property_str: str
    mc_result: Optional[mc.LTLResult] = None
    rv_verdicts: Optional[List[rv.Verdict]] = None
    rv_final: Optional[rv.Verdict] = None
    mode: VerificationMode = VerificationMode.DUAL
    consistent: Optional[bool] = None
    diagnosis: str = ""

    @property
    def mc_satisfied(self) -> Optional[bool]:
        if self.mc_result is None:
            return None
        return self.mc_result.holds

    @property
    def rv_satisfied(self) -> Optional[bool]:
        if self.rv_final is None:
            return None
        return self.rv_final == rv.Verdict.TRUE


class DualVerifier:
    """Verify properties both via model checking and runtime monitoring."""

    def __init__(self, state_vars: List[str],
                 init_map: Optional[Dict[str, bool]] = None,
                 transitions: Optional[List] = None,
                 init_expr: Optional[Callable] = None,
                 trans_expr: Optional[Callable] = None):
        self.state_vars = state_vars
        self.init_map = init_map
        self.transitions = transitions
        self.init_expr = init_expr
        self.trans_expr = trans_expr
        self._has_model = (init_map is not None or init_expr is not None)

    @classmethod
    def from_extracted_model(cls, model: ExtractedModel) -> 'DualVerifier':
        """Create verifier from a trace-extracted model."""
        if not model.initial_states or not model.transitions:
            raise ValueError("Extracted model has no states or transitions")

        transitions = []
        for src, dst in model.transitions:
            cond = {v: src[v] for v in model.state_vars}
            update = {v: dst[v] for v in model.state_vars}
            transitions.append({'condition': cond, 'update': update})

        return cls(
            state_vars=model.state_vars,
            init_map=model.initial_states[0],
            transitions=transitions
        )

    def model_check(self, prop: rv.Formula, max_steps: int = 100) -> mc.LTLResult:
        """Check property via V023 model checking."""
        if not self._has_model:
            raise ValueError("No system model available for model checking")

        mc_formula = rv_to_mc(prop)

        if self.init_expr is not None and self.trans_expr is not None:
            return mc.check_ltl(
                self.state_vars, self.init_expr, self.trans_expr,
                mc_formula, max_steps=max_steps
            )
        else:
            # Build BDD functions from dict-based model
            init_fn = _build_init_fn(self.init_map)
            trans_fn = _build_trans_fn(self.state_vars, self.transitions)
            return mc.check_ltl(
                self.state_vars, init_fn, trans_fn,
                mc_formula, max_steps=max_steps
            )

    def monitor_trace(self, prop: rv.Formula,
                      trace: List[rv.Event]) -> Tuple[List[rv.Verdict], rv.Verdict]:
        """Check property via V176 runtime monitoring."""
        monitor = rv.FutureTimeMonitor(prop)
        verdicts = []
        for event in trace:
            v = monitor.process(event)
            verdicts.append(v)
        final = monitor.finalize()
        return verdicts, final

    def verify(self, prop: rv.Formula,
               trace: Optional[List[rv.Event]] = None,
               mode: VerificationMode = VerificationMode.DUAL,
               max_mc_steps: int = 100) -> DualResult:
        """Verify property in specified mode."""
        result = DualResult(
            property_str=str(prop),
            mode=mode
        )

        if mode in (VerificationMode.MODEL_CHECK, VerificationMode.DUAL):
            if self._has_model:
                try:
                    result.mc_result = self.model_check(prop, max_mc_steps)
                except FormulaTranslationError as e:
                    result.diagnosis = f"MC skipped: {e}"
                except Exception as e:
                    result.diagnosis = f"MC error: {e}"

        if mode in (VerificationMode.MONITOR, VerificationMode.DUAL):
            if trace is not None:
                result.rv_verdicts, result.rv_final = self.monitor_trace(prop, trace)

        # Consistency check
        if result.mc_result is not None and result.rv_final is not None:
            mc_sat = result.mc_satisfied
            rv_sat = result.rv_satisfied
            if mc_sat is not None and rv_sat is not None:
                if mc_sat and not rv_sat:
                    result.consistent = False
                    result.diagnosis = (
                        "INCONSISTENCY: Model checker says SATISFIED but "
                        "monitor says VIOLATED on trace."
                    )
                elif not mc_sat and rv_sat:
                    result.consistent = True
                    result.diagnosis = (
                        "MC found counterexample but trace satisfies property. "
                        "The trace avoids the violating path."
                    )
                else:
                    result.consistent = True
            elif rv_sat is None and result.rv_final == rv.Verdict.UNKNOWN:
                result.consistent = True
                result.diagnosis = "Monitor verdict inconclusive (UNKNOWN)"

        return result


# ============================================================
# Section 5: Counterexample-Guided Monitor
# ============================================================

@dataclass
class TargetedMonitor:
    """A monitor configured from model checking counterexamples."""
    property_formula: rv.Formula
    counterexample_prefix: List[Dict[str, bool]]
    counterexample_cycle: List[Dict[str, bool]]
    monitors: List[rv.FutureTimeMonitor]
    prefix_patterns: List[List[str]]

    def check_trace(self, trace: List[rv.Event]) -> Dict[str, Any]:
        """Check if trace follows a counterexample pattern."""
        results = {
            'property_verdict': None,
            'matches_counterexample': False,
            'matching_prefix_length': 0,
            'monitor_verdicts': []
        }

        prop_monitor = rv.FutureTimeMonitor(self.property_formula)
        verdicts = []
        for evt in trace:
            v = prop_monitor.process(evt)
            verdicts.append(v)
        results['property_verdict'] = prop_monitor.finalize()
        results['monitor_verdicts'] = verdicts

        if self.prefix_patterns:
            for pattern in self.prefix_patterns:
                match_len = 0
                for i, expected in enumerate(pattern):
                    if i >= len(trace):
                        break
                    evt_name = trace[i].name if isinstance(trace[i], rv.Event) else str(trace[i])
                    if evt_name == expected:
                        match_len += 1
                    else:
                        break
                if match_len > results['matching_prefix_length']:
                    results['matching_prefix_length'] = match_len
                if match_len == len(pattern):
                    results['matches_counterexample'] = True

        return results


def counterexample_to_monitor(
    prop: rv.Formula,
    mc_result: mc.LTLResult,
    state_vars: List[str]
) -> Optional[TargetedMonitor]:
    """Create targeted monitor from model checking counterexample.

    mc_result.counterexample is (prefix, cycle) where each is a list of
    {var: bool} dicts.
    """
    if mc_result.holds:
        return None
    if not mc_result.counterexample:
        return None

    prefix, cycle = mc_result.counterexample

    # Extract event patterns from counterexample prefix states
    prefix_patterns = []
    pattern = []
    for state in prefix:
        true_props = [v for v in state_vars if state.get(v, False)]
        if true_props:
            pattern.append(true_props[0])
    if pattern:
        prefix_patterns.append(pattern)

    monitors = [rv.FutureTimeMonitor(prop)]

    return TargetedMonitor(
        property_formula=prop,
        counterexample_prefix=prefix,
        counterexample_cycle=cycle,
        monitors=monitors,
        prefix_patterns=prefix_patterns
    )


# ============================================================
# Section 6: Specification Mining
# ============================================================

class PropertyPattern(Enum):
    """Common temporal property patterns."""
    RESPONSE = auto()
    ABSENCE = auto()
    EXISTENCE = auto()
    PRECEDENCE = auto()
    ALTERNATION = auto()


@dataclass
class MinedProperty:
    """A property discovered from trace analysis."""
    pattern: PropertyPattern
    formula: rv.Formula
    confidence: float
    support: int
    description: str
    mc_result: Optional[mc.LTLResult] = None


def mine_response_patterns(traces: List[List[rv.Event]],
                           min_support: int = 1) -> List[MinedProperty]:
    """Mine response patterns: G(a -> F(b))."""
    results = []
    all_names = set()
    for trace in traces:
        for evt in trace:
            name = evt.name if isinstance(evt, rv.Event) else str(evt)
            all_names.add(name)

    for a in sorted(all_names):
        for b in sorted(all_names):
            if a == b:
                continue
            support = 0
            total_triggers = 0
            responded = 0
            for trace in traces:
                names = [e.name if isinstance(e, rv.Event) else str(e)
                         for e in trace]
                triggered = False
                for i, n in enumerate(names):
                    if n == a:
                        triggered = True
                        total_triggers += 1
                        if b in names[i + 1:]:
                            responded += 1
                if triggered:
                    support += 1

            if support >= min_support and total_triggers > 0:
                confidence = responded / total_triggers
                if confidence >= 0.9:
                    formula = rv.Always(rv.Implies(rv.Atom(a), rv.Eventually(rv.Atom(b))))
                    results.append(MinedProperty(
                        pattern=PropertyPattern.RESPONSE,
                        formula=formula,
                        confidence=confidence,
                        support=support,
                        description=f"G({a} -> F({b}))"
                    ))
    return results


def mine_absence_patterns(traces: List[List[rv.Event]],
                          min_support: int = 1) -> List[MinedProperty]:
    """Mine absence patterns: G(!bad)."""
    results = []
    all_names = set()
    per_trace_names = []
    for trace in traces:
        names = set()
        for evt in trace:
            name = evt.name if isinstance(evt, rv.Event) else str(evt)
            names.add(name)
            all_names.add(name)
        per_trace_names.append(names)

    for name in sorted(all_names):
        absent_count = sum(1 for names in per_trace_names if name not in names)
        if absent_count > 0 and absent_count >= min_support:
            confidence = absent_count / len(traces)
            formula = rv.Always(rv.Not(rv.Atom(name)))
            results.append(MinedProperty(
                pattern=PropertyPattern.ABSENCE,
                formula=formula,
                confidence=confidence,
                support=absent_count,
                description=f"G(!{name})"
            ))
    return results


def mine_precedence_patterns(traces: List[List[rv.Event]],
                             min_support: int = 1) -> List[MinedProperty]:
    """Mine precedence patterns: !b W a (a must precede b)."""
    results = []
    all_names = set()
    for trace in traces:
        for evt in trace:
            name = evt.name if isinstance(evt, rv.Event) else str(evt)
            all_names.add(name)

    for a in sorted(all_names):
        for b in sorted(all_names):
            if a == b:
                continue
            support = 0
            total_with_both = 0
            a_precedes = 0
            for trace in traces:
                names = [e.name if isinstance(e, rv.Event) else str(e)
                         for e in trace]
                if a in names and b in names:
                    total_with_both += 1
                    first_a = names.index(a)
                    first_b = names.index(b)
                    if first_a < first_b:
                        a_precedes += 1
                        support += 1

            if total_with_both >= min_support and total_with_both > 0:
                confidence = a_precedes / total_with_both
                if confidence >= 0.9:
                    formula = rv.Or(
                        rv.Until(rv.Not(rv.Atom(b)), rv.Atom(a)),
                        rv.Always(rv.Not(rv.Atom(b)))
                    )
                    results.append(MinedProperty(
                        pattern=PropertyPattern.PRECEDENCE,
                        formula=formula,
                        confidence=confidence,
                        support=support,
                        description=f"!{b} W {a} ({a} precedes {b})"
                    ))
    return results


def mine_existence_patterns(traces: List[List[rv.Event]],
                            min_support: int = 1) -> List[MinedProperty]:
    """Mine existence patterns: F(event)."""
    results = []
    all_names = set()
    per_trace_names = []
    for trace in traces:
        names = set()
        for evt in trace:
            name = evt.name if isinstance(evt, rv.Event) else str(evt)
            names.add(name)
            all_names.add(name)
        per_trace_names.append(names)

    for name in sorted(all_names):
        present_count = sum(1 for names in per_trace_names if name in names)
        if present_count >= min_support:
            confidence = present_count / len(traces)
            if confidence >= 0.9:
                formula = rv.Eventually(rv.Atom(name))
                results.append(MinedProperty(
                    pattern=PropertyPattern.EXISTENCE,
                    formula=formula,
                    confidence=confidence,
                    support=present_count,
                    description=f"F({name})"
                ))
    return results


def mine_specifications(
    traces: List[List[rv.Event]],
    patterns: Optional[List[PropertyPattern]] = None,
    min_support: int = 1
) -> List[MinedProperty]:
    """Mine temporal specifications from execution traces."""
    if patterns is None:
        patterns = list(PropertyPattern)

    results = []
    if PropertyPattern.RESPONSE in patterns:
        results.extend(mine_response_patterns(traces, min_support))
    if PropertyPattern.ABSENCE in patterns:
        results.extend(mine_absence_patterns(traces, min_support))
    if PropertyPattern.PRECEDENCE in patterns:
        results.extend(mine_precedence_patterns(traces, min_support))
    if PropertyPattern.EXISTENCE in patterns:
        results.extend(mine_existence_patterns(traces, min_support))

    results.sort(key=lambda p: (-p.confidence, p.description))
    return results


def mine_and_verify(
    traces: List[List[rv.Event]],
    verifier: DualVerifier,
    patterns: Optional[List[PropertyPattern]] = None,
    min_confidence: float = 0.9,
    max_mc_steps: int = 100
) -> List[MinedProperty]:
    """Mine specifications from traces, then verify via model checking."""
    candidates = mine_specifications(traces, patterns)
    verified = []
    for candidate in candidates:
        if candidate.confidence < min_confidence:
            continue
        try:
            mc_result = verifier.model_check(candidate.formula, max_mc_steps)
            candidate.mc_result = mc_result
            verified.append(candidate)
        except (FormulaTranslationError, Exception):
            verified.append(candidate)
    return verified


# ============================================================
# Section 7: Trace Conformance Checking
# ============================================================

@dataclass
class ConformanceResult:
    """Result of checking trace conformance against a model."""
    conforms: bool
    violation_step: int = -1
    violation_state: Optional[Dict[str, bool]] = None
    expected_transitions: List[Dict[str, bool]] = field(default_factory=list)
    actual_state: Optional[FrozenSet[str]] = None
    diagnosis: str = ""


def check_trace_conformance(
    trace: List[rv.Event],
    model: ExtractedModel,
    prop_extractors: Optional[Dict[str, Callable]] = None,
    strict: bool = False
) -> ConformanceResult:
    """Check if an execution trace conforms to an extracted model."""
    if not trace:
        return ConformanceResult(conforms=True, diagnosis="Empty trace")

    trans_set = set()
    state_set = set()
    for s1, s2 in model.transitions:
        k1 = frozenset(s1.items())
        k2 = frozenset(s2.items())
        trans_set.add((k1, k2))
        state_set.add(k1)
        state_set.add(k2)

    init_set = set()
    for s in model.initial_states:
        init_set.add(frozenset(s.items()))

    trace_states = []
    for evt in trace:
        if isinstance(evt, str):
            evt = rv.Event(evt)
        props = events_to_propositions(evt, prop_extractors)
        state_dict = {v: (v in props) for v in model.state_vars}
        trace_states.append(frozenset(state_dict.items()))

    if trace_states and trace_states[0] not in init_set:
        if strict:
            return ConformanceResult(
                conforms=False,
                violation_step=0,
                actual_state=trace_states[0],
                diagnosis="Initial state not in model's initial states"
            )

    for i in range(len(trace_states) - 1):
        s1 = trace_states[i]
        s2 = trace_states[i + 1]
        if strict:
            if (s1, s2) not in trans_set:
                return ConformanceResult(
                    conforms=False,
                    violation_step=i + 1,
                    actual_state=s2,
                    diagnosis=f"Transition at step {i}->{i+1} not in model"
                )
        else:
            if s2 not in state_set and s1 not in state_set:
                return ConformanceResult(
                    conforms=False,
                    violation_step=i + 1,
                    actual_state=s2,
                    diagnosis=f"State at step {i+1} never observed in model"
                )

    return ConformanceResult(conforms=True, diagnosis="Trace conforms to model")


# ============================================================
# Section 8: Integrated Pipeline
# ============================================================

class RVModelChecker:
    """Integrated runtime verification + model checking pipeline."""

    def __init__(self):
        self.traces: List[List[rv.Event]] = []
        self.model: Optional[ExtractedModel] = None
        self.verifier: Optional[DualVerifier] = None
        self.mined_properties: List[MinedProperty] = []
        self.monitors: Dict[str, rv.FutureTimeMonitor] = {}

    def add_trace(self, trace: List[rv.Event],
                  prop_extractors: Optional[Dict[str, Callable]] = None):
        """Add an execution trace. Rebuilds model if needed."""
        self.traces.append(trace)
        self.model = extract_model_from_traces(self.traces, prop_extractors)
        if self.model.initial_states and self.model.transitions:
            self.verifier = DualVerifier.from_extracted_model(self.model)

    def set_model(self, state_vars: List[str],
                  init_map: Dict[str, bool],
                  transitions: List):
        """Set an explicit system model for model checking."""
        self.verifier = DualVerifier(state_vars, init_map, transitions)

    def mine(self, patterns: Optional[List[PropertyPattern]] = None,
             min_confidence: float = 0.9) -> List[MinedProperty]:
        """Mine specifications from collected traces."""
        self.mined_properties = mine_specifications(
            self.traces, patterns, min_support=1
        )
        return [p for p in self.mined_properties if p.confidence >= min_confidence]

    def verify_property(self, prop: rv.Formula,
                        trace: Optional[List[rv.Event]] = None,
                        mode: VerificationMode = VerificationMode.DUAL,
                        max_mc_steps: int = 100) -> DualResult:
        """Verify a property in the specified mode."""
        if self.verifier is None and mode != VerificationMode.MONITOR:
            raise ValueError("No model available. Add traces or set model first.")
        if self.verifier is None:
            v = DualVerifier([], None, None)
            return v.verify(prop, trace, VerificationMode.MONITOR)
        return self.verifier.verify(prop, trace, mode, max_mc_steps)

    def verify_mined(self, max_mc_steps: int = 100) -> List[MinedProperty]:
        """Verify all mined properties via model checking."""
        if not self.mined_properties:
            self.mine()
        if self.verifier is None:
            return self.mined_properties
        for prop in self.mined_properties:
            try:
                prop.mc_result = self.verifier.model_check(
                    prop.formula, max_mc_steps
                )
            except (FormulaTranslationError, Exception):
                pass
        return self.mined_properties

    def create_targeted_monitor(self, prop: rv.Formula,
                                max_mc_steps: int = 100
                                ) -> Optional[TargetedMonitor]:
        """Model-check property and create targeted monitor from counterexample."""
        if self.verifier is None:
            return None
        try:
            mc_result = self.verifier.model_check(prop, max_mc_steps)
        except Exception:
            return None

        if mc_result.holds:
            return None

        sv = self.model.state_vars if self.model else (
            self.verifier.state_vars if self.verifier else []
        )
        return counterexample_to_monitor(prop, mc_result, sv)

    def check_conformance(self, trace: List[rv.Event],
                          prop_extractors: Optional[Dict[str, Callable]] = None,
                          strict: bool = False) -> ConformanceResult:
        """Check if a trace conforms to the learned model."""
        if self.model is None:
            raise ValueError("No model available. Add traces first.")
        return check_trace_conformance(trace, self.model, prop_extractors, strict)

    def full_pipeline(self, max_mc_steps: int = 100,
                      min_confidence: float = 0.9
                      ) -> Dict[str, Any]:
        """Run the full pipeline: mine -> verify -> report."""
        mined = self.mine(min_confidence=min_confidence)

        verified_props = []
        for prop in mined:
            try:
                prop.mc_result = self.verifier.model_check(
                    prop.formula, max_mc_steps
                ) if self.verifier else None
                verified_props.append(prop)
            except (FormulaTranslationError, Exception):
                verified_props.append(prop)

        mc_confirmed = [p for p in verified_props
                        if p.mc_result and p.mc_result.holds]
        mc_refuted = [p for p in verified_props
                      if p.mc_result and not p.mc_result.holds]
        mc_unknown = [p for p in verified_props
                      if p.mc_result is None]

        return {
            'total_mined': len(mined),
            'mc_confirmed': len(mc_confirmed),
            'mc_refuted': len(mc_refuted),
            'mc_unknown': len(mc_unknown),
            'confirmed_properties': [p.description for p in mc_confirmed],
            'refuted_properties': [p.description for p in mc_refuted],
            'all_properties': verified_props
        }


# ============================================================
# Section 9: Convenience Functions
# ============================================================

def verify_with_traces(
    prop: rv.Formula,
    traces: List[List[rv.Event]],
    prop_extractors: Optional[Dict[str, Callable]] = None,
    max_mc_steps: int = 100
) -> DualResult:
    """One-shot: extract model from traces, verify property both ways."""
    model = extract_model_from_traces(traces, prop_extractors)
    if not model.initial_states or not model.transitions:
        all_verdicts = []
        for trace in traces:
            mon = rv.FutureTimeMonitor(prop)
            for evt in trace:
                mon.process(evt)
            all_verdicts.append(mon.finalize())
        final = rv.Verdict.TRUE
        for v in all_verdicts:
            if v == rv.Verdict.FALSE:
                final = rv.Verdict.FALSE
                break
            if v == rv.Verdict.UNKNOWN:
                final = rv.Verdict.UNKNOWN
        return DualResult(
            property_str=str(prop),
            rv_final=final,
            mode=VerificationMode.MONITOR
        )

    verifier = DualVerifier.from_extracted_model(model)
    return verifier.verify(prop, traces[0], VerificationMode.DUAL, max_mc_steps)


def mine_and_check(
    traces: List[List[rv.Event]],
    prop_extractors: Optional[Dict[str, Callable]] = None,
    max_mc_steps: int = 100,
    min_confidence: float = 0.9
) -> Dict[str, Any]:
    """One-shot: mine specs from traces, verify via model checking."""
    pipeline = RVModelChecker()
    for trace in traces:
        pipeline.add_trace(trace, prop_extractors)
    return pipeline.full_pipeline(max_mc_steps, min_confidence)
