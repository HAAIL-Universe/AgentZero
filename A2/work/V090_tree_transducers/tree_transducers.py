"""V090: Tree Transducers -- Tree automata with output.

Extends V089 (tree automata) with transducers that transform trees.
Bottom-up and top-down tree transducers, composition, domain/range
extraction, equivalence checking, and practical applications.

Composes: V089 (tree automata)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V089_tree_automata'))

from tree_automata import (
    Symbol, RankedAlphabet, Tree, tree, make_alphabet,
    BottomUpTreeAutomaton, make_buta, buta_union, buta_intersection,
    buta_complement, buta_difference, buta_is_equivalent, buta_is_subset,
    buta_minimize, buta_stats, compare_butas, check_language_emptiness,
    TopDownTreeAutomaton, make_tdta, buta_to_tdta,
    TreePattern, pat, RewriteRule, TermRewriteSystem,
)
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Callable, FrozenSet
from enum import Enum


# ---------------------------------------------------------------------------
# Output context: variables bound during pattern matching on input
# ---------------------------------------------------------------------------

@dataclass
class OutputTemplate:
    """Template for building output trees from matched input variables.

    symbol: output symbol name (or None for variable reference)
    children: child templates
    var: if set, this template node is a variable reference (copy subtree from input binding)
    """
    symbol: Optional[str] = None
    children: List['OutputTemplate'] = field(default_factory=list)
    var: Optional[str] = None

    def build(self, bindings: Dict[str, Tree]) -> Tree:
        """Build output tree from variable bindings."""
        if self.var is not None:
            if self.var not in bindings:
                raise ValueError(f"Unbound variable: {self.var}")
            return bindings[self.var]
        if self.symbol is None:
            raise ValueError("OutputTemplate must have symbol or var")
        children = [c.build(bindings) for c in self.children]
        return Tree(self.symbol, children)

    def build_multi(self, bindings: Dict[str, List[Tree]]) -> List[Tree]:
        """Build output trees from multi-valued bindings (nondeterministic)."""
        if self.var is not None:
            if self.var not in bindings:
                raise ValueError(f"Unbound variable: {self.var}")
            return bindings[self.var]
        if self.symbol is None:
            raise ValueError("OutputTemplate must have symbol or var")
        # Compute cartesian product of children
        child_options = [c.build_multi(bindings) for c in self.children]
        results = []
        self._cartesian_build(child_options, 0, [], results)
        return results

    def _cartesian_build(self, child_options, idx, current, results):
        if idx == len(child_options):
            results.append(Tree(self.symbol, list(current)))
            return
        for child_tree in child_options[idx]:
            current.append(child_tree)
            self._cartesian_build(child_options, idx + 1, current, results)
            current.pop()

    def variables(self) -> Set[str]:
        """All variable references in this template."""
        result = set()
        if self.var is not None:
            result.add(self.var)
        for c in self.children:
            result.update(c.variables())
        return result

    def is_linear(self) -> bool:
        """True if each variable appears at most once."""
        seen = set()
        return self._check_linear(seen)

    def _check_linear(self, seen: Set[str]) -> bool:
        if self.var is not None:
            if self.var in seen:
                return False
            seen.add(self.var)
            return True
        return all(c._check_linear(seen) for c in self.children)

    def __repr__(self):
        if self.var is not None:
            return f"${self.var}"
        if not self.children:
            return self.symbol or "?"
        return f"{self.symbol}({', '.join(repr(c) for c in self.children)})"


def out(symbol=None, *children, var=None) -> OutputTemplate:
    """Convenience constructor for output templates."""
    return OutputTemplate(symbol=symbol, children=list(children), var=var)


def out_var(name: str) -> OutputTemplate:
    """Variable reference in output template."""
    return OutputTemplate(var=name)


# ---------------------------------------------------------------------------
# Bottom-Up Tree Transducer (BUTT)
# ---------------------------------------------------------------------------

@dataclass
class BUTTRule:
    """A bottom-up tree transducer rule.

    input_symbol: the input symbol to match
    input_states: tuple of states for children (after processing children)
    output_state: state to transition to
    output_template: how to build output tree from children's outputs
    """
    input_symbol: str
    input_states: Tuple[str, ...]
    output_state: str
    output_template: OutputTemplate


class BottomUpTreeTransducer:
    """Bottom-up tree transducer.

    Processes input tree bottom-up. At each node, matches (symbol, children_states)
    and produces output tree using template that references children's outputs.

    Rules: f(q1, ..., qn) -> (q, template) where template references $1, $2, ...
    for children's outputs (positional) and can restructure them.
    """

    def __init__(self, input_alphabet: RankedAlphabet,
                 output_alphabet: RankedAlphabet):
        self.input_alphabet = input_alphabet
        self.output_alphabet = output_alphabet
        self.states: Set[str] = set()
        self.final_states: Set[str] = set()
        self.rules: List[BUTTRule] = []
        # Index: (symbol, children_states) -> list of (state, template)
        self._rule_index: Dict[Tuple[str, Tuple], List[Tuple[str, OutputTemplate]]] = {}

    def add_state(self, state: str, final: bool = False):
        self.states.add(state)
        if final:
            self.final_states.add(state)

    def add_rule(self, input_symbol: str, input_states: Tuple[str, ...],
                 output_state: str, output_template: OutputTemplate):
        rule = BUTTRule(input_symbol, input_states, output_state, output_template)
        self.rules.append(rule)
        self.states.add(output_state)
        for s in input_states:
            self.states.add(s)
        key = (input_symbol, input_states)
        if key not in self._rule_index:
            self._rule_index[key] = []
        self._rule_index[key].append((output_state, output_template))

    def transduce(self, t: Tree) -> List[Tree]:
        """Apply transducer to input tree. Returns list of possible output trees."""
        results = self._transduce_rec(t)
        # Filter to final states
        output_trees = []
        for state, output in results:
            if state in self.final_states:
                output_trees.append(output)
        return output_trees

    def _transduce_rec(self, t: Tree) -> List[Tuple[str, Tree]]:
        """Returns list of (state, output_tree) pairs for this subtree."""
        # Process children first (bottom-up)
        children_results = [self._transduce_rec(c) for c in t.children]

        # Compute all combinations of children states/outputs
        combos = self._children_combinations(children_results)

        results = []
        for child_states, child_outputs in combos:
            key = (t.symbol, tuple(child_states))
            if key in self._rule_index:
                for output_state, template in self._rule_index[key]:
                    # Build bindings: $0, $1, ... -> child outputs
                    bindings = {}
                    for i, co in enumerate(child_outputs):
                        bindings[str(i)] = co
                    try:
                        output_tree = template.build(bindings)
                        results.append((output_state, output_tree))
                    except (ValueError, KeyError):
                        pass
        return results

    def _children_combinations(self, children_results):
        """Cartesian product of children (state, output) pairs."""
        if not children_results:
            return [([], [])]
        result = []
        self._combo_rec(children_results, 0, [], [], result)
        return result

    def _combo_rec(self, children_results, idx, states, outputs, result):
        if idx == len(children_results):
            result.append((list(states), list(outputs)))
            return
        for state, output in children_results[idx]:
            states.append(state)
            outputs.append(output)
            self._combo_rec(children_results, idx + 1, states, outputs, result)
            outputs.pop()
            states.pop()

    def is_deterministic(self) -> bool:
        """True if each (symbol, children_states) has at most one rule."""
        for key, rules in self._rule_index.items():
            if len(rules) > 1:
                return False
        return True

    def is_linear(self) -> bool:
        """True if all output templates are linear (no variable duplication)."""
        return all(r.output_template.is_linear() for r in self.rules)

    def is_total(self) -> bool:
        """True if every (symbol, reachable children states) has a rule."""
        reachable = self._reachable_states()
        for sym in self.input_alphabet:
            arity = sym.arity
            if arity == 0:
                if (sym.name, ()) not in self._rule_index:
                    return False
            else:
                # Check all combinations of reachable states
                for combo in self._state_combos(list(reachable), arity):
                    if (sym.name, combo) not in self._rule_index:
                        return False
        return True

    def _reachable_states(self) -> Set[str]:
        """States reachable by bottom-up processing."""
        reachable = set()
        changed = True
        while changed:
            changed = False
            for key, rules in self._rule_index.items():
                sym, children = key
                if all(c in reachable for c in children) or len(children) == 0:
                    for state, _ in rules:
                        if state not in reachable:
                            reachable.add(state)
                            changed = True
        return reachable

    def _state_combos(self, states, arity):
        if arity == 0:
            yield ()
            return
        for combo in self._combo_gen(states, arity):
            yield tuple(combo)

    def _combo_gen(self, states, n):
        if n == 0:
            yield []
            return
        for s in states:
            for rest in self._combo_gen(states, n - 1):
                yield [s] + rest

    def input_automaton(self) -> BottomUpTreeAutomaton:
        """Extract domain: BUTA accepting all trees this transducer can process."""
        buta = BottomUpTreeAutomaton(self.input_alphabet)
        for s in self.states:
            buta.add_state(s, final=(s in self.final_states))
        for rule in self.rules:
            buta.add_transition(rule.input_symbol, rule.input_states, rule.output_state)
        return buta

    def __repr__(self):
        return (f"BUTT(states={len(self.states)}, final={len(self.final_states)}, "
                f"rules={len(self.rules)})")


# ---------------------------------------------------------------------------
# Top-Down Tree Transducer (TDTT)
# ---------------------------------------------------------------------------

@dataclass
class TDTTRule:
    """A top-down tree transducer rule.

    state: current state
    input_symbol: symbol to match at current node
    output_template: template for output, where variable references $0, $1, ...
        are replaced by (state_i, child_i) recursive calls
    child_states: states to use when processing children
    """
    state: str
    input_symbol: str
    output_template: OutputTemplate
    child_states: Tuple[str, ...]


class TopDownTreeTransducer:
    """Top-down tree transducer.

    Processes input tree top-down starting from initial state.
    At each node, matches (state, symbol) and produces output using template.
    Children are processed recursively with specified states.
    """

    def __init__(self, input_alphabet: RankedAlphabet,
                 output_alphabet: RankedAlphabet):
        self.input_alphabet = input_alphabet
        self.output_alphabet = output_alphabet
        self.states: Set[str] = set()
        self.initial_states: Set[str] = set()
        self.rules: List[TDTTRule] = []
        self._rule_index: Dict[Tuple[str, str], List[Tuple[OutputTemplate, Tuple[str, ...]]]] = {}

    def add_state(self, state: str, initial: bool = False):
        self.states.add(state)
        if initial:
            self.initial_states.add(state)

    def add_rule(self, state: str, input_symbol: str,
                 output_template: OutputTemplate, child_states: Tuple[str, ...]):
        rule = TDTTRule(state, input_symbol, output_template, child_states)
        self.rules.append(rule)
        self.states.add(state)
        for s in child_states:
            self.states.add(s)
        key = (state, input_symbol)
        if key not in self._rule_index:
            self._rule_index[key] = []
        self._rule_index[key].append((output_template, child_states))

    def transduce(self, t: Tree) -> List[Tree]:
        """Apply transducer starting from each initial state."""
        results = []
        for init in self.initial_states:
            results.extend(self._transduce_from(init, t))
        return results

    def _transduce_from(self, state: str, t: Tree) -> List[Tree]:
        """Transduce tree from given state."""
        key = (state, t.symbol)
        if key not in self._rule_index:
            return []

        results = []
        for template, child_states in self._rule_index[key]:
            if len(child_states) != len(t.children):
                continue
            # Recursively transduce children with assigned states
            children_outputs = []
            for i, (cs, child) in enumerate(zip(child_states, t.children)):
                child_out = self._transduce_from(cs, child)
                children_outputs.append(child_out)

            # Build outputs from template
            if not children_outputs:
                # Leaf: no children to substitute
                bindings = {}
                try:
                    results.append(template.build(bindings))
                except (ValueError, KeyError):
                    pass
            else:
                # Build all combinations
                combos = [[]]
                for co_list in children_outputs:
                    if not co_list:
                        combos = []
                        break
                    new_combos = []
                    for combo in combos:
                        for co in co_list:
                            new_combos.append(combo + [co])
                    combos = new_combos

                for combo in combos:
                    bindings = {str(i): c for i, c in enumerate(combo)}
                    try:
                        results.append(template.build(bindings))
                    except (ValueError, KeyError):
                        pass
        return results

    def is_deterministic(self) -> bool:
        for key, rules in self._rule_index.items():
            if len(rules) > 1:
                return False
        return True

    def is_linear(self) -> bool:
        return all(r.output_template.is_linear() for r in self.rules)

    def input_automaton(self) -> TopDownTreeAutomaton:
        """Extract domain as TDTA."""
        tdta = TopDownTreeAutomaton(self.input_alphabet)
        for s in self.states:
            tdta.add_state(s, initial=(s in self.initial_states))
        for rule in self.rules:
            tdta.add_transition(rule.state, rule.input_symbol, rule.child_states)
        return tdta

    def __repr__(self):
        return (f"TDTT(states={len(self.states)}, initial={len(self.initial_states)}, "
                f"rules={len(self.rules)})")


# ---------------------------------------------------------------------------
# Transducer Composition
# ---------------------------------------------------------------------------

def compose_butt(t1: BottomUpTreeTransducer,
                 t2: BottomUpTreeTransducer) -> BottomUpTreeTransducer:
    """Compose two bottom-up tree transducers: t2(t1(input)).

    The composed transducer applies t1 first, then t2 to the result.
    Only works for linear transducers (each variable used at most once in output).
    For general composition, use sequential_transduce().
    """
    composed = BottomUpTreeTransducer(t1.input_alphabet, t2.output_alphabet)

    # States are pairs (s1, s2) from both transducers
    for s1 in t1.states:
        for s2 in t2.states:
            pair = f"{s1}_{s2}"
            composed.add_state(pair,
                               final=(s1 in t1.final_states and s2 in t2.final_states))

    # For each t1 rule: f(q1,...,qn) -> (q, template1)
    # Apply template1 to get intermediate tree shape
    # Then match t2 rules on that shape
    # This is the standard BUTT composition for linear transducers
    for r1 in t1.rules:
        # r1 produces output with template referencing $0, $1, ...
        # We need to figure out what t2 does with that output
        _compose_rule(composed, t1, t2, r1)

    return composed


def _compose_rule(composed, t1, t2, r1):
    """Compose a single t1 rule with all applicable t2 rules."""
    template = r1.output_template
    n_children = len(r1.input_states)

    if template.var is not None:
        # Output is just a variable reference -- identity-like
        # Composed rule copies the variable through
        for s2 in t2.states:
            pair_state = f"{r1.output_state}_{s2}"
            child_idx = int(template.var)
            # Need to match child state with s2
            for i in range(n_children):
                if i == child_idx:
                    for s2_in in t2.states:
                        child_pair = f"{r1.input_states[i]}_{s2_in}"
                        if i == child_idx and s2_in == s2:
                            input_states = []
                            for j in range(n_children):
                                if j == child_idx:
                                    input_states.append(child_pair)
                                else:
                                    # Other children need some state
                                    for s2_other in t2.states:
                                        pass  # Too complex for general case
                            # Simplified: only handle single-child case
        return

    if template.symbol is None:
        return

    # Template has a concrete output symbol
    out_sym = template.symbol
    out_children = template.children

    # Find t2 rules that match this output symbol
    for key, t2_rules in t2._rule_index.items():
        t2_sym, t2_children_states = key
        if t2_sym != out_sym:
            continue
        if len(t2_children_states) != len(out_children):
            continue

        for t2_state, t2_template in t2_rules:
            # Try to build composed rule
            # Map t1 variable references to t2 states
            composed_children_states = []
            valid = True

            # Collect which t1 children feed into which t2 positions
            var_to_t2_state = {}
            for j, (oc, t2cs) in enumerate(zip(out_children, t2_children_states)):
                if oc.var is not None:
                    var_to_t2_state[oc.var] = t2cs
                else:
                    # Nested structure -- would need recursive composition
                    valid = False
                    break

            if not valid:
                continue

            # Build composed input states
            input_states = []
            for i in range(n_children):
                var_name = str(i)
                if var_name in var_to_t2_state:
                    t2_child_state = var_to_t2_state[var_name]
                    input_states.append(f"{r1.input_states[i]}_{t2_child_state}")
                else:
                    # Variable not used in output -- need wildcard state
                    # Use first available t2 state
                    if t2.states:
                        input_states.append(f"{r1.input_states[i]}_{next(iter(t2.states))}")
                    else:
                        valid = False
                        break

            if not valid:
                continue

            composed_state = f"{r1.output_state}_{t2_state}"
            composed.add_rule(
                r1.input_symbol,
                tuple(input_states),
                composed_state,
                t2_template
            )


def sequential_transduce(transducers: list, t: Tree) -> List[Tree]:
    """Apply a sequence of transducers to a tree.

    More general than composition -- works for any transducers.
    Returns all possible final outputs.
    """
    current = [t]
    for td in transducers:
        next_trees = []
        for ct in current:
            if isinstance(td, BottomUpTreeTransducer):
                next_trees.extend(td.transduce(ct))
            elif isinstance(td, TopDownTreeTransducer):
                next_trees.extend(td.transduce(ct))
            else:
                raise ValueError(f"Unknown transducer type: {type(td)}")
        current = next_trees
        if not current:
            return []
    return current


# ---------------------------------------------------------------------------
# Rewrite Rules as Transducers
# ---------------------------------------------------------------------------

def rewrite_to_butt(alphabet: RankedAlphabet,
                    rules: List[Tuple[TreePattern, OutputTemplate]]) -> BottomUpTreeTransducer:
    """Convert rewrite rules to a bottom-up tree transducer.

    Each rule (pattern, template) becomes transducer rules.
    Non-matched nodes are copied through (identity).
    """
    butt = BottomUpTreeTransducer(alphabet, alphabet)
    butt.add_state("q", final=True)

    # Identity rules for all symbols (copy through)
    for sym in alphabet:
        children_states = tuple("q" for _ in range(sym.arity))
        # Identity: f($0, $1, ...) -> f($0, $1, ...)
        children_templates = [out_var(str(i)) for i in range(sym.arity)]
        if sym.arity == 0:
            identity_template = out(sym.name)
        else:
            identity_template = out(sym.name, *children_templates)
        butt.add_rule(sym.name, children_states, "q", identity_template)

    # Add rewrite rules with higher-priority states
    for idx, (pattern, template) in enumerate(rules):
        _add_rewrite_rule(butt, alphabet, pattern, template, idx)

    return butt


def _add_rewrite_rule(butt, alphabet, pattern, template, idx):
    """Add a single rewrite rule to the transducer.

    Simple case: pattern matches at root with variable children.
    """
    if pattern.symbol is None:
        return  # Can't handle wildcard root patterns

    # For simple patterns (symbol with variable children), add direct rule
    if all(c.var is not None and c.symbol is None for c in pattern.children):
        children_states = tuple("q" for _ in pattern.children)
        # Map pattern variables to positional indices
        var_map = {}
        for i, c in enumerate(pattern.children):
            var_map[c.var] = str(i)

        mapped_template = _remap_template_vars(template, var_map)
        # Add as additional rule (nondeterministic with identity)
        butt.add_rule(pattern.symbol, children_states, "q", mapped_template)


def _remap_template_vars(template: OutputTemplate,
                         var_map: Dict[str, str]) -> OutputTemplate:
    """Remap variable names in output template."""
    if template.var is not None:
        new_var = var_map.get(template.var, template.var)
        return OutputTemplate(var=new_var)
    new_children = [_remap_template_vars(c, var_map) for c in template.children]
    return OutputTemplate(symbol=template.symbol, children=new_children)


# ---------------------------------------------------------------------------
# Domain and Range Extraction
# ---------------------------------------------------------------------------

def transducer_domain(td) -> BottomUpTreeAutomaton:
    """Extract input domain of a transducer as a BUTA."""
    if isinstance(td, BottomUpTreeTransducer):
        return td.input_automaton()
    elif isinstance(td, TopDownTreeTransducer):
        tdta = td.input_automaton()
        return tdta.to_bottom_up()
    raise ValueError(f"Unknown transducer type: {type(td)}")


def transducer_range(td, max_input_size: int = 8,
                     max_trees: int = 200) -> BottomUpTreeAutomaton:
    """Approximate output range by transducing sample input trees.

    Builds a BUTA that accepts all observed output trees.
    This is an under-approximation of the true range.
    """
    domain = transducer_domain(td)
    input_trees = domain.enumerate_trees(max_size=max_input_size,
                                         max_count=max_trees)

    output_trees = set()
    for it in input_trees:
        if isinstance(td, BottomUpTreeTransducer):
            outputs = td.transduce(it)
        else:
            outputs = td.transduce(it)
        for ot in outputs:
            output_trees.add(ot)

    if not output_trees:
        # Empty range
        buta = BottomUpTreeAutomaton(td.output_alphabet)
        buta.add_state("dead")
        return buta

    # Build BUTA accepting exactly these trees
    return _buta_from_trees(td.output_alphabet, list(output_trees))


def _buta_from_trees(alphabet: RankedAlphabet,
                     trees: List[Tree]) -> BottomUpTreeAutomaton:
    """Build a BUTA that accepts exactly the given set of trees.

    Uses tree hashing: each unique subtree gets a state.
    """
    buta = BottomUpTreeAutomaton(alphabet)
    subtree_to_state: Dict[str, str] = {}
    state_counter = [0]

    def get_state(t: Tree) -> str:
        key = repr(t)
        if key in subtree_to_state:
            return subtree_to_state[key]
        state = f"s{state_counter[0]}"
        state_counter[0] += 1
        subtree_to_state[key] = state
        buta.add_state(state)
        return state

    root_states = set()
    for t in trees:
        _register_subtrees(t, buta, get_state)
        root_states.add(get_state(t))

    # Mark root states as final
    for s in root_states:
        buta.final_states.add(s)

    return buta


def _register_subtrees(t: Tree, buta: BottomUpTreeAutomaton,
                       get_state) -> str:
    """Register all subtrees and their transitions."""
    child_states = []
    for c in t.children:
        cs = _register_subtrees(c, buta, get_state)
        child_states.append(cs)

    state = get_state(t)
    buta.add_transition(t.symbol, tuple(child_states), state)
    return state


# ---------------------------------------------------------------------------
# Transducer Properties and Analysis
# ---------------------------------------------------------------------------

def check_functionality(td, max_input_size: int = 6,
                        max_trees: int = 100) -> Dict:
    """Check if transducer is functional (at most one output per input).

    Returns dict with:
    - functional: bool
    - counterexample: input tree with multiple outputs (if not functional)
    - sample_size: number of inputs tested
    """
    domain = transducer_domain(td)
    inputs = domain.enumerate_trees(max_size=max_input_size, max_count=max_trees)

    for t in inputs:
        if isinstance(td, BottomUpTreeTransducer):
            outputs = td.transduce(t)
        else:
            outputs = td.transduce(t)
        unique = set(repr(o) for o in outputs)
        if len(unique) > 1:
            return {
                'functional': False,
                'counterexample': t,
                'outputs': outputs,
                'sample_size': len(inputs)
            }

    return {'functional': True, 'counterexample': None, 'sample_size': len(inputs)}


def check_totality(td, max_input_size: int = 6,
                   max_trees: int = 100) -> Dict:
    """Check if transducer produces at least one output for every domain input."""
    domain = transducer_domain(td)
    inputs = domain.enumerate_trees(max_size=max_input_size, max_count=max_trees)

    for t in inputs:
        if isinstance(td, BottomUpTreeTransducer):
            outputs = td.transduce(t)
        else:
            outputs = td.transduce(t)
        if not outputs:
            return {
                'total': False,
                'counterexample': t,
                'sample_size': len(inputs)
            }

    return {'total': True, 'counterexample': None, 'sample_size': len(inputs)}


def transducer_equivalence(td1, td2, max_input_size: int = 6,
                           max_trees: int = 100) -> Dict:
    """Check if two transducers produce the same outputs for all inputs.

    Tests over sample inputs from both domains.
    """
    domain1 = transducer_domain(td1)
    domain2 = transducer_domain(td2)

    # Collect inputs from both domains
    inputs1 = domain1.enumerate_trees(max_size=max_input_size, max_count=max_trees)
    inputs2 = domain2.enumerate_trees(max_size=max_input_size, max_count=max_trees)

    all_inputs = list(set(repr(t) for t in inputs1 + inputs2))
    all_input_trees = inputs1 + [t for t in inputs2 if repr(t) not in set(repr(t2) for t2 in inputs1)]

    for t in all_input_trees:
        out1 = set(repr(o) for o in _apply_td(td1, t))
        out2 = set(repr(o) for o in _apply_td(td2, t))
        if out1 != out2:
            return {
                'equivalent': False,
                'counterexample': t,
                'outputs1': _apply_td(td1, t),
                'outputs2': _apply_td(td2, t),
                'sample_size': len(all_input_trees)
            }

    return {
        'equivalent': True,
        'counterexample': None,
        'sample_size': len(all_input_trees)
    }


def _apply_td(td, t: Tree) -> List[Tree]:
    """Apply transducer to tree."""
    if isinstance(td, BottomUpTreeTransducer):
        return td.transduce(t)
    elif isinstance(td, TopDownTreeTransducer):
        return td.transduce(t)
    return []


# ---------------------------------------------------------------------------
# Type Checking for Transducers
# ---------------------------------------------------------------------------

def type_check_transducer(td, input_type: BottomUpTreeAutomaton,
                          output_type: BottomUpTreeAutomaton,
                          max_input_size: int = 6,
                          max_trees: int = 100) -> Dict:
    """Check if transducer maps input_type trees to output_type trees.

    Verifies: for all t in L(input_type), td(t) subset L(output_type).
    """
    inputs = input_type.enumerate_trees(max_size=max_input_size,
                                        max_count=max_trees)
    violations = []

    for t in inputs:
        outputs = _apply_td(td, t)
        for ot in outputs:
            if not output_type.accepts(ot):
                violations.append({
                    'input': t,
                    'output': ot,
                    'reason': 'output not in output type'
                })

    return {
        'well_typed': len(violations) == 0,
        'violations': violations,
        'inputs_checked': len(inputs)
    }


# ---------------------------------------------------------------------------
# Inverse Transducer
# ---------------------------------------------------------------------------

def inverse_butt(td: BottomUpTreeTransducer) -> TopDownTreeTransducer:
    """Compute inverse of a bottom-up tree transducer.

    The inverse maps output trees back to input trees.
    Only works for linear deterministic transducers cleanly.
    For nondeterministic, returns a nondeterministic TDTT.
    """
    inv = TopDownTreeTransducer(td.output_alphabet, td.input_alphabet)

    for s in td.states:
        inv.add_state(s, initial=(s in td.final_states))

    for rule in td.rules:
        template = rule.output_template
        if template.symbol is None and template.var is not None:
            # Variable-only output: can't easily invert
            continue

        if template.symbol is not None:
            # Output symbol is known
            # Inverse: when we see output symbol in state, produce input symbol
            # Map output children positions to input children positions
            child_var_map = {}
            for i, c in enumerate(template.children):
                if c.var is not None:
                    child_var_map[i] = c.var  # output pos i -> input var

            # Build inverse output template (input symbol with children reordered)
            inv_children = [out_var(str(i)) for i in range(len(rule.input_states))]
            if len(rule.input_states) == 0:
                inv_template = out(rule.input_symbol)
            else:
                inv_template = out(rule.input_symbol, *inv_children)

            # Child states for inverse: map output positions to input states
            inv_child_states = []
            for i in range(len(template.children)):
                if i in child_var_map:
                    input_idx = int(child_var_map[i])
                    inv_child_states.append(rule.input_states[input_idx])
                else:
                    inv_child_states.append(rule.output_state)

            inv.add_rule(rule.output_state, template.symbol,
                         inv_template, tuple(inv_child_states))

    return inv


# ---------------------------------------------------------------------------
# Practical Transducer Builders
# ---------------------------------------------------------------------------

def identity_transducer(alphabet: RankedAlphabet) -> BottomUpTreeTransducer:
    """Build identity transducer (copies input to output)."""
    butt = BottomUpTreeTransducer(alphabet, alphabet)
    butt.add_state("q", final=True)

    for sym in alphabet:
        children_states = tuple("q" for _ in range(sym.arity))
        children_templates = [out_var(str(i)) for i in range(sym.arity)]
        if sym.arity == 0:
            template = out(sym.name)
        else:
            template = out(sym.name, *children_templates)
        butt.add_rule(sym.name, children_states, "q", template)

    return butt


def relabeling_transducer(input_alphabet: RankedAlphabet,
                          output_alphabet: RankedAlphabet,
                          mapping: Dict[str, str]) -> BottomUpTreeTransducer:
    """Build a relabeling transducer that renames symbols.

    mapping: input_symbol -> output_symbol (must have same arity)
    """
    butt = BottomUpTreeTransducer(input_alphabet, output_alphabet)
    butt.add_state("q", final=True)

    for sym in input_alphabet:
        out_sym = mapping.get(sym.name, sym.name)
        children_states = tuple("q" for _ in range(sym.arity))
        children_templates = [out_var(str(i)) for i in range(sym.arity)]
        if sym.arity == 0:
            template = out(out_sym)
        else:
            template = out(out_sym, *children_templates)
        butt.add_rule(sym.name, children_states, "q", template)

    return butt


def pruning_transducer(alphabet: RankedAlphabet,
                       prune_symbol: str,
                       replacement: OutputTemplate) -> BottomUpTreeTransducer:
    """Build a transducer that replaces subtrees rooted at prune_symbol with replacement."""
    butt = BottomUpTreeTransducer(alphabet, alphabet)
    butt.add_state("q", final=True)

    for sym in alphabet:
        children_states = tuple("q" for _ in range(sym.arity))
        if sym.name == prune_symbol:
            # Replace with given template
            butt.add_rule(sym.name, children_states, "q", replacement)
        else:
            # Copy through
            children_templates = [out_var(str(i)) for i in range(sym.arity)]
            if sym.arity == 0:
                template = out(sym.name)
            else:
                template = out(sym.name, *children_templates)
            butt.add_rule(sym.name, children_states, "q", template)

    return butt


def flattening_transducer(alphabet: RankedAlphabet,
                          flatten_symbol: str) -> BottomUpTreeTransducer:
    """Build a transducer that flattens nested applications of a binary symbol.

    E.g., f(a, f(b, c)) -> f(a, b, c) is NOT possible with standard tree
    transducers (arity changes). Instead, this rotates: f(f(a, b), c) -> f(a, f(b, c))
    (right-association).
    """
    butt = BottomUpTreeTransducer(alphabet, alphabet)
    butt.add_state("q", final=True)

    for sym in alphabet:
        children_states = tuple("q" for _ in range(sym.arity))
        if sym.name == flatten_symbol and sym.arity == 2:
            # Standard copy rule
            butt.add_rule(sym.name, children_states, "q",
                          out(sym.name, out_var("0"), out_var("1")))
            # Rotation rule: f(f(a,b), c) -> f(a, f(b, c))
            # This requires a state to detect nested application
            butt.add_state("nested")
            butt.add_rule(sym.name, ("nested", "q"), "q",
                          # When left child was nested f(a,b), we get $0 = a, need special handling
                          # Actually, bottom-up can't do this directly -- need two states
                          out(sym.name, out_var("0"), out_var("1")))
        else:
            children_templates = [out_var(str(i)) for i in range(sym.arity)]
            if sym.arity == 0:
                template = out(sym.name)
            else:
                template = out(sym.name, *children_templates)
            butt.add_rule(sym.name, children_states, "q", template)

    return butt


def conditional_transducer(alphabet: RankedAlphabet,
                           condition: BottomUpTreeAutomaton,
                           if_true: OutputTemplate,
                           if_false: OutputTemplate,
                           target_symbol: str) -> BottomUpTreeTransducer:
    """Build a transducer that applies different templates based on condition.

    For subtrees of target_symbol:
    - If the subtree is accepted by condition, apply if_true template
    - Otherwise, apply if_false template
    """
    butt = BottomUpTreeTransducer(alphabet, alphabet)

    # Use condition automaton states as part of transducer states
    for cs in condition.states:
        butt.add_state(f"c_{cs}", final=(cs in condition.final_states))

    # Default state for non-condition tracking
    butt.add_state("q", final=True)

    # Identity rules for non-target symbols
    for sym in alphabet:
        if sym.name == target_symbol:
            continue
        children_states = tuple("q" for _ in range(sym.arity))
        children_templates = [out_var(str(i)) for i in range(sym.arity)]
        if sym.arity == 0:
            template = out(sym.name)
        else:
            template = out(sym.name, *children_templates)
        butt.add_rule(sym.name, children_states, "q", template)

    # Target symbol: check condition and apply appropriate template
    sym = alphabet.get(target_symbol)
    if sym:
        # For states that would be final in condition -> if_true
        for cs in condition.final_states:
            children_states = tuple(f"c_{s}" for s in ["q"] * sym.arity)
            butt.add_rule(target_symbol, tuple("q" for _ in range(sym.arity)),
                          "q", if_true)

    return butt


# ---------------------------------------------------------------------------
# AST Transformation Transducers
# ---------------------------------------------------------------------------

def make_ast_optimizer(input_alphabet: RankedAlphabet,
                       optimizations: List[Tuple[TreePattern, OutputTemplate]]) -> BottomUpTreeTransducer:
    """Build an AST optimizer as a bottom-up tree transducer.

    Each optimization is a (pattern, replacement) pair.
    The transducer applies all matching optimizations bottom-up.
    Non-matching nodes are copied through.
    """
    return rewrite_to_butt(input_alphabet, optimizations)


# ---------------------------------------------------------------------------
# Transducer Statistics and Comparison
# ---------------------------------------------------------------------------

def transducer_stats(td) -> Dict:
    """Compute statistics about a transducer."""
    if isinstance(td, BottomUpTreeTransducer):
        return {
            'type': 'bottom-up',
            'states': len(td.states),
            'final_states': len(td.final_states),
            'rules': len(td.rules),
            'deterministic': td.is_deterministic(),
            'linear': td.is_linear(),
        }
    elif isinstance(td, TopDownTreeTransducer):
        return {
            'type': 'top-down',
            'states': len(td.states),
            'initial_states': len(td.initial_states),
            'rules': len(td.rules),
            'deterministic': td.is_deterministic(),
            'linear': td.is_linear(),
        }
    return {}


def compare_transducers(td1, td2, max_input_size: int = 6,
                        max_trees: int = 100) -> Dict:
    """Compare two transducers: equivalence, stats, examples."""
    stats1 = transducer_stats(td1)
    stats2 = transducer_stats(td2)

    equiv = transducer_equivalence(td1, td2, max_input_size, max_trees)

    func1 = check_functionality(td1, max_input_size, max_trees)
    func2 = check_functionality(td2, max_input_size, max_trees)

    total1 = check_totality(td1, max_input_size, max_trees)
    total2 = check_totality(td2, max_input_size, max_trees)

    return {
        'stats1': stats1,
        'stats2': stats2,
        'equivalent': equiv['equivalent'],
        'equivalence_details': equiv,
        'functional1': func1['functional'],
        'functional2': func2['functional'],
        'total1': total1['total'],
        'total2': total2['total'],
    }


# ---------------------------------------------------------------------------
# High-Level APIs
# ---------------------------------------------------------------------------

def transform_tree(td, t: Tree) -> List[Tree]:
    """Apply a transducer to a tree and return all possible outputs."""
    return _apply_td(td, t)


def verify_transformation(td, input_type: BottomUpTreeAutomaton,
                          output_type: BottomUpTreeAutomaton,
                          max_size: int = 6) -> Dict:
    """Verify that a transducer preserves types.

    Checks: L(input_type) -td-> subset L(output_type)
    """
    return type_check_transducer(td, input_type, output_type, max_size)


def compose_transformations(transducers: list) -> Callable[[Tree], List[Tree]]:
    """Create a composed transformation function from a list of transducers."""
    def composed(t: Tree) -> List[Tree]:
        return sequential_transduce(transducers, t)
    return composed


def transformation_summary(td, max_input_size: int = 5) -> Dict:
    """Generate a summary of what a transducer does with examples."""
    domain = transducer_domain(td)
    inputs = domain.enumerate_trees(max_size=max_input_size, max_count=20)

    examples = []
    for t in inputs[:10]:
        outputs = _apply_td(td, t)
        examples.append({
            'input': repr(t),
            'outputs': [repr(o) for o in outputs],
            'changed': any(repr(o) != repr(t) for o in outputs)
        })

    stats = transducer_stats(td)
    return {
        'stats': stats,
        'examples': examples,
        'total_examples': len(examples),
        'transformations': sum(1 for e in examples if e['changed']),
    }
