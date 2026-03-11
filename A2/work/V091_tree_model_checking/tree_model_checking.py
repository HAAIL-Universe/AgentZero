"""V091: Regular Tree Model Checking

Verifies properties of tree-transforming systems using tree automata
as state sets and tree transducers as transition relations.

Core idea: If states are regular tree languages (tree automata) and
transitions are regular tree relations (tree transducers), then
reachability analysis computes fixpoints of automata transformations.

Features:
- Forward reachability: compute post*(Init) via transducer image iteration
- Backward reachability: compute pre*(Bad) via inverse transducer
- Safety checking: Init intersect pre*(Bad) = empty?
- Invariant verification: Inv is inductive (tau(Inv) subset Inv)?
- Bounded model checking: k-step reachability
- Acceleration: widening for convergence of automata sequences
- Counterexample: witness tree trace from Init to Bad

Composes: V089 (tree automata) + V090 (tree transducers)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V090_tree_transducers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V089_tree_automata'))

from tree_automata import (
    Symbol, RankedAlphabet, Tree, tree, make_alphabet,
    BottomUpTreeAutomaton, make_buta, buta_union, buta_intersection,
    buta_complement, buta_difference, buta_is_equivalent, buta_is_subset,
    buta_minimize, buta_stats, check_language_emptiness,
    check_language_inclusion,
)
from tree_transducers import (
    OutputTemplate, out, out_var, BUTTRule, BottomUpTreeTransducer,
    TDTTRule, TopDownTreeTransducer,
    transducer_domain, transducer_range, check_functionality,
    identity_transducer, transform_tree, transducer_stats,
    compose_butt, inverse_butt,
)
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, FrozenSet
from enum import Enum


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class VerificationResult(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class CounterexampleTrace:
    """A concrete trace from initial to bad state."""
    trees: List[Tree]  # sequence of trees: t0 -> t1 -> ... -> tn
    length: int

    def __repr__(self):
        steps = " -> ".join(str(t) for t in self.trees)
        return f"Trace({self.length} steps): {steps}"


@dataclass
class ModelCheckResult:
    """Result of a tree model checking query."""
    result: VerificationResult
    invariant: Optional[BottomUpTreeAutomaton] = None
    counterexample: Optional[CounterexampleTrace] = None
    steps: int = 0
    stats: Dict = field(default_factory=dict)

    def __repr__(self):
        s = f"ModelCheckResult({self.result.value}, steps={self.steps})"
        if self.counterexample:
            s += f"\n  {self.counterexample}"
        return s


@dataclass
class TreeTransitionSystem:
    """A tree-transforming transition system.

    alphabet: the ranked alphabet for trees
    init: tree automaton recognizing initial configurations
    transition: tree transducer defining one-step transitions
    bad: tree automaton recognizing bad/error configurations (optional)
    property_aut: tree automaton recognizing property (good) configurations (optional)
    """
    alphabet: RankedAlphabet
    init: BottomUpTreeAutomaton
    transition: BottomUpTreeTransducer
    bad: Optional[BottomUpTreeAutomaton] = None
    property_aut: Optional[BottomUpTreeAutomaton] = None


# ---------------------------------------------------------------------------
# Transducer image: compute the set of trees reachable via one transducer step
# ---------------------------------------------------------------------------

def transducer_image(td: BottomUpTreeTransducer, lang: BottomUpTreeAutomaton,
                     max_input_size: int = 8, max_trees: int = 500) -> BottomUpTreeAutomaton:
    """Compute the image of a tree language under a transducer.

    Returns a BUTA recognizing {t' | exists t in L(lang), t' in td(t)}.
    Uses enumeration of input trees + transduction + automaton construction.
    """
    alphabet = lang.alphabet

    # Enumerate trees in the input language
    input_trees = lang.enumerate(max_size=max_input_size)
    if len(input_trees) > max_trees:
        input_trees = input_trees[:max_trees]

    # Apply transducer to each input tree
    output_trees = set()
    for t in input_trees:
        results = td.transduce(t)
        for r in results:
            output_trees.add(_tree_to_hashable(r))

    if not output_trees:
        # Empty image -- return empty automaton
        return _empty_automaton(alphabet)

    # Build BUTA accepting exactly the output trees
    return _buta_from_tree_set(alphabet, [_hashable_to_tree(h) for h in output_trees])


def _tree_to_hashable(t: Tree):
    """Convert tree to hashable representation."""
    children = tuple(_tree_to_hashable(c) for c in t.children)
    return (t.symbol, children)


def _hashable_to_tree(h) -> Tree:
    """Convert hashable back to Tree."""
    symbol, children = h
    return Tree(symbol, [_hashable_to_tree(c) for c in children])


def _empty_automaton(alphabet: RankedAlphabet) -> BottomUpTreeAutomaton:
    """Create automaton accepting empty language."""
    buta = BottomUpTreeAutomaton(alphabet)
    buta.states = {"qsink"}
    buta.final_states = set()  # no final states = empty language
    for sym in alphabet:
        if sym.arity == 0:
            buta.add_transition(sym.name, [], "qsink")
        else:
            for combo in _state_combos(["qsink"], sym.arity):
                buta.add_transition(sym.name, list(combo), "qsink")
    return buta


def _state_combos(states, arity):
    """Generate all combinations of states for given arity."""
    import itertools
    return itertools.product(states, repeat=arity)


def _buta_from_tree_set(alphabet: RankedAlphabet, trees: List[Tree]) -> BottomUpTreeAutomaton:
    """Build a BUTA that accepts exactly the given set of trees.

    Uses a trie-based construction: each unique subtree gets its own state.
    """
    buta = BottomUpTreeAutomaton(alphabet)
    subtree_to_state = {}
    state_counter = [0]

    def get_state(t: Tree) -> str:
        key = _tree_to_hashable(t)
        if key in subtree_to_state:
            return subtree_to_state[key]

        child_states = [get_state(c) for c in t.children]
        state_name = f"q{state_counter[0]}"
        state_counter[0] += 1
        subtree_to_state[key] = state_name
        buta.states.add(state_name)
        buta.add_transition(t.symbol, child_states, state_name)
        return state_name

    for t in trees:
        final = get_state(t)
        buta.final_states.add(final)

    return buta


# ---------------------------------------------------------------------------
# Forward reachability: compute post*(Init)
# ---------------------------------------------------------------------------

def forward_reachability(tts: TreeTransitionSystem, max_steps: int = 20,
                         max_input_size: int = 8) -> ModelCheckResult:
    """Compute forward reachable states from Init via repeated transducer application.

    Computes: R0 = Init, R_{i+1} = R_i union Image(td, R_i)
    Stops when R_{i+1} = R_i (fixpoint) or max_steps reached.

    If bad is specified, checks at each step whether R_i intersects Bad.
    """
    reach = tts.init
    stats = {"steps": [], "state_counts": []}

    for step in range(max_steps):
        # Check safety: does reachable set intersect bad?
        if tts.bad is not None:
            inter = buta_intersection(reach, tts.bad)
            emptiness = check_language_emptiness(inter)
            if not emptiness["is_empty"]:
                # Found counterexample -- try to extract trace
                witness = emptiness.get("witness")
                cex = _extract_counterexample(tts, step + 1, witness)
                return ModelCheckResult(
                    result=VerificationResult.UNSAFE,
                    counterexample=cex,
                    steps=step + 1,
                    stats=stats
                )

        # Compute one-step image
        img = transducer_image(tts.transition, reach, max_input_size=max_input_size)

        # Union with current reachable set
        new_reach = buta_union(reach, img)

        # Check fixpoint
        stats["steps"].append(step)
        stats["state_counts"].append(len(new_reach.states))

        is_subset, _ = buta_is_subset(new_reach, reach)
        if is_subset:
            # Fixpoint reached!
            if tts.bad is not None:
                # Final safety check
                inter = buta_intersection(new_reach, tts.bad)
                emptiness = check_language_emptiness(inter)
                if not emptiness["is_empty"]:
                    witness = emptiness.get("witness")
                    cex = _extract_counterexample(tts, step + 1, witness)
                    return ModelCheckResult(
                        result=VerificationResult.UNSAFE,
                        counterexample=cex,
                        steps=step + 1,
                        stats=stats
                    )
            return ModelCheckResult(
                result=VerificationResult.SAFE,
                invariant=new_reach,
                steps=step + 1,
                stats=stats
            )

        reach = new_reach

    # Didn't converge
    return ModelCheckResult(
        result=VerificationResult.UNKNOWN,
        steps=max_steps,
        stats=stats
    )


# ---------------------------------------------------------------------------
# Backward reachability: compute pre*(Bad)
# ---------------------------------------------------------------------------

def backward_reachability(tts: TreeTransitionSystem, max_steps: int = 20,
                          max_input_size: int = 8) -> ModelCheckResult:
    """Compute backward reachable states from Bad via inverse transducer.

    Computes: B0 = Bad, B_{i+1} = B_i union PreImage(td, B_i)
    Safe if B* intersect Init = empty.
    """
    if tts.bad is None:
        raise ValueError("Backward reachability requires bad states")

    bad_reach = tts.bad
    stats = {"steps": [], "state_counts": []}

    # Build inverse transducer
    inv_td = inverse_butt(tts.transition)

    for step in range(max_steps):
        # Check: does backward set reach init?
        inter = buta_intersection(bad_reach, tts.init)
        emptiness = check_language_emptiness(inter)
        if not emptiness["is_empty"]:
            witness = emptiness.get("witness")
            cex = _extract_backward_counterexample(tts, step, witness)
            return ModelCheckResult(
                result=VerificationResult.UNSAFE,
                counterexample=cex,
                steps=step + 1,
                stats=stats
            )

        # Compute pre-image via inverse transducer (TDTT)
        # The inverse maps output->input, so we enumerate bad_reach trees
        # and apply inverse to get pre-images
        pre_trees = _compute_preimage_via_inverse(inv_td, bad_reach, tts.alphabet,
                                                   max_input_size=max_input_size)

        if not pre_trees:
            # No pre-image -- fixpoint
            return ModelCheckResult(
                result=VerificationResult.SAFE,
                invariant=buta_complement(bad_reach),
                steps=step + 1,
                stats=stats
            )

        pre_aut = _buta_from_tree_set(tts.alphabet, pre_trees)
        new_bad = buta_union(bad_reach, pre_aut)

        stats["steps"].append(step)
        stats["state_counts"].append(len(new_bad.states))

        is_subset, _ = buta_is_subset(new_bad, bad_reach)
        if is_subset:
            # Fixpoint
            inter = buta_intersection(new_bad, tts.init)
            emptiness = check_language_emptiness(inter)
            if not emptiness["is_empty"]:
                witness = emptiness.get("witness")
                cex = _extract_backward_counterexample(tts, step + 1, witness)
                return ModelCheckResult(
                    result=VerificationResult.UNSAFE,
                    counterexample=cex,
                    steps=step + 1,
                    stats=stats
                )
            return ModelCheckResult(
                result=VerificationResult.SAFE,
                invariant=buta_complement(new_bad),
                steps=step + 1,
                stats=stats
            )

        bad_reach = new_bad

    return ModelCheckResult(
        result=VerificationResult.UNKNOWN,
        steps=max_steps,
        stats=stats
    )


def _compute_preimage_via_inverse(inv_td: TopDownTreeTransducer,
                                   lang: BottomUpTreeAutomaton,
                                   alphabet: RankedAlphabet,
                                   max_input_size: int = 8) -> List[Tree]:
    """Compute pre-image trees via inverse transducer."""
    input_trees = lang.enumerate(max_size=max_input_size)
    pre_trees = set()
    for t in input_trees[:500]:
        results = inv_td.transduce(t)
        for r in results:
            pre_trees.add(_tree_to_hashable(r))
    return [_hashable_to_tree(h) for h in pre_trees]


# ---------------------------------------------------------------------------
# Bounded model checking
# ---------------------------------------------------------------------------

def bounded_check(tts: TreeTransitionSystem, bound: int,
                  max_input_size: int = 8) -> ModelCheckResult:
    """Check safety up to a bounded number of transition steps.

    No fixpoint check -- just unrolls the transducer up to `bound` steps.
    """
    if tts.bad is None:
        raise ValueError("Bounded check requires bad states")

    reach = tts.init
    stats = {"step_details": []}

    for step in range(bound + 1):
        inter = buta_intersection(reach, tts.bad)
        emptiness = check_language_emptiness(inter)
        stats["step_details"].append({
            "step": step,
            "states": len(reach.states),
            "intersects_bad": not emptiness["is_empty"]
        })
        if not emptiness["is_empty"]:
            witness = emptiness.get("witness")
            cex = _extract_counterexample(tts, step, witness)
            return ModelCheckResult(
                result=VerificationResult.UNSAFE,
                counterexample=cex,
                steps=step,
                stats=stats
            )

        if step < bound:
            img = transducer_image(tts.transition, reach, max_input_size=max_input_size)
            reach = buta_union(reach, img)

    return ModelCheckResult(
        result=VerificationResult.SAFE,
        steps=bound,
        stats=stats
    )


# ---------------------------------------------------------------------------
# Invariant checking
# ---------------------------------------------------------------------------

def check_invariant(tts: TreeTransitionSystem, invariant: BottomUpTreeAutomaton,
                    max_input_size: int = 8) -> Dict:
    """Check if an automaton is an inductive invariant for the system.

    An invariant Inv must satisfy:
    1. Init subset Inv (initiation)
    2. Image(td, Inv) subset Inv (consecution / inductiveness)
    3. Inv intersect Bad = empty (safety, if bad is specified)
    """
    result = {"is_invariant": True, "violations": []}

    # 1. Initiation: Init subset Inv
    is_sub, witness = buta_is_subset(tts.init, invariant)
    if not is_sub:
        result["is_invariant"] = False
        result["violations"].append({
            "type": "initiation",
            "message": "Init not subset of Inv",
            "witness": witness
        })

    # 2. Consecution: Image(td, Inv) subset Inv
    img = transducer_image(tts.transition, invariant, max_input_size=max_input_size)
    is_sub2, witness2 = buta_is_subset(img, invariant)
    if not is_sub2:
        result["is_invariant"] = False
        result["violations"].append({
            "type": "consecution",
            "message": "Image(td, Inv) not subset of Inv",
            "witness": witness2
        })

    # 3. Safety: Inv intersect Bad = empty
    if tts.bad is not None:
        inter = buta_intersection(invariant, tts.bad)
        emptiness = check_language_emptiness(inter)
        if not emptiness["is_empty"]:
            result["is_invariant"] = False
            result["violations"].append({
                "type": "safety",
                "message": "Inv intersects Bad",
                "witness": emptiness.get("witness")
            })

    return result


# ---------------------------------------------------------------------------
# Counterexample extraction
# ---------------------------------------------------------------------------

def _extract_counterexample(tts: TreeTransitionSystem, steps: int,
                             bad_tree: Optional[Tree]) -> Optional[CounterexampleTrace]:
    """Try to reconstruct a concrete counterexample trace."""
    if bad_tree is None:
        return CounterexampleTrace(trees=[], length=steps)

    # For bounded traces: try to reconstruct step by step
    # Forward search: find init tree that reaches bad_tree in `steps` transitions
    trace = _find_trace_forward(tts, bad_tree, steps)
    if trace:
        return CounterexampleTrace(trees=trace, length=len(trace) - 1)

    return CounterexampleTrace(trees=[bad_tree], length=steps)


def _extract_backward_counterexample(tts: TreeTransitionSystem, steps: int,
                                      init_tree: Optional[Tree]) -> Optional[CounterexampleTrace]:
    """Extract counterexample from backward reachability."""
    if init_tree is None:
        return CounterexampleTrace(trees=[], length=steps)
    return CounterexampleTrace(trees=[init_tree], length=steps)


def _find_trace_forward(tts: TreeTransitionSystem, target: Tree,
                         max_steps: int) -> Optional[List[Tree]]:
    """Find a concrete trace from init to target."""
    if max_steps <= 0:
        if tts.init.accepts(target):
            return [target]
        return None

    # Enumerate initial trees and try to reach target
    init_trees = tts.init.enumerate(max_size=6)
    target_hash = _tree_to_hashable(target)

    for init_t in init_trees[:100]:
        trace = _bfs_trace(tts.transition, init_t, target_hash, max_steps)
        if trace:
            return trace

    return None


def _bfs_trace(td: BottomUpTreeTransducer, start: Tree,
               target_hash, max_steps: int) -> Optional[List[Tree]]:
    """BFS from start tree toward target via transducer steps."""
    from collections import deque

    queue = deque([(start, [start], 0)])
    visited = {_tree_to_hashable(start)}

    while queue:
        current, path, depth = queue.popleft()
        if depth >= max_steps:
            continue

        results = td.transduce(current)
        for r in results:
            r_hash = _tree_to_hashable(r)
            if r_hash == target_hash:
                return path + [r]
            if r_hash not in visited and depth + 1 < max_steps:
                visited.add(r_hash)
                queue.append((r, path + [r], depth + 1))
                if len(visited) > 1000:
                    return None

    return None


# ---------------------------------------------------------------------------
# Widening: approximate fixpoint acceleration for automata sequences
# ---------------------------------------------------------------------------

def widen_automata(a_prev: BottomUpTreeAutomaton,
                   a_curr: BottomUpTreeAutomaton) -> BottomUpTreeAutomaton:
    """Widen two automata in an ascending chain.

    Simple widening strategy: union + state merging heuristic.
    States in a_curr that have similar transition structure to states in a_prev
    are merged, over-approximating the language to force convergence.

    For tree model checking, a practical widening merges states that accept
    overlapping tree sets (collapse equivalence classes more aggressively).
    """
    # Strategy: union then aggressively minimize (collapse non-distinguishable pairs)
    union = buta_union(a_prev, a_curr)
    minimized = buta_minimize(union)
    return minimized


def accelerated_forward(tts: TreeTransitionSystem, max_steps: int = 30,
                        widen_after: int = 5,
                        max_input_size: int = 8) -> ModelCheckResult:
    """Forward reachability with widening acceleration.

    Runs exact iteration for `widen_after` steps, then applies widening
    to accelerate convergence. Trades precision for termination.
    """
    reach = tts.init
    stats = {"steps": [], "widened_at": [], "state_counts": []}

    for step in range(max_steps):
        # Safety check
        if tts.bad is not None:
            inter = buta_intersection(reach, tts.bad)
            emptiness = check_language_emptiness(inter)
            if not emptiness["is_empty"]:
                witness = emptiness.get("witness")
                cex = _extract_counterexample(tts, step + 1, witness)
                return ModelCheckResult(
                    result=VerificationResult.UNSAFE,
                    counterexample=cex,
                    steps=step + 1,
                    stats=stats
                )

        # One-step image
        img = transducer_image(tts.transition, reach, max_input_size=max_input_size)
        new_reach = buta_union(reach, img)

        # Apply widening after threshold
        if step >= widen_after:
            new_reach = widen_automata(reach, new_reach)
            stats["widened_at"].append(step)

        stats["steps"].append(step)
        stats["state_counts"].append(len(new_reach.states))

        # Fixpoint check
        is_subset, _ = buta_is_subset(new_reach, reach)
        if is_subset:
            if tts.bad is not None:
                inter = buta_intersection(new_reach, tts.bad)
                emptiness = check_language_emptiness(inter)
                if not emptiness["is_empty"]:
                    witness = emptiness.get("witness")
                    cex = _extract_counterexample(tts, step + 1, witness)
                    return ModelCheckResult(
                        result=VerificationResult.UNSAFE,
                        counterexample=cex,
                        steps=step + 1,
                        stats=stats
                    )
            return ModelCheckResult(
                result=VerificationResult.SAFE,
                invariant=new_reach,
                steps=step + 1,
                stats=stats
            )

        reach = new_reach

    return ModelCheckResult(
        result=VerificationResult.UNKNOWN,
        steps=max_steps,
        stats=stats
    )


# ---------------------------------------------------------------------------
# Property checking: temporal-like properties over tree sequences
# ---------------------------------------------------------------------------

def check_safety(tts: TreeTransitionSystem, max_steps: int = 20,
                 method: str = "forward",
                 max_input_size: int = 8) -> ModelCheckResult:
    """Check safety property: no reachable state is bad.

    Methods: "forward" (default), "backward", "bounded", "accelerated"
    """
    if method == "forward":
        return forward_reachability(tts, max_steps=max_steps,
                                    max_input_size=max_input_size)
    elif method == "backward":
        return backward_reachability(tts, max_steps=max_steps,
                                     max_input_size=max_input_size)
    elif method == "bounded":
        return bounded_check(tts, bound=max_steps, max_input_size=max_input_size)
    elif method == "accelerated":
        return accelerated_forward(tts, max_steps=max_steps,
                                   max_input_size=max_input_size)
    else:
        raise ValueError(f"Unknown method: {method}")


def check_reachability(tts: TreeTransitionSystem, target: BottomUpTreeAutomaton,
                       max_steps: int = 20,
                       max_input_size: int = 8) -> ModelCheckResult:
    """Check if any tree in target is reachable from Init.

    Returns UNSAFE if reachable (target is "bad"), SAFE if not.
    """
    tts_with_target = TreeTransitionSystem(
        alphabet=tts.alphabet,
        init=tts.init,
        transition=tts.transition,
        bad=target
    )
    return forward_reachability(tts_with_target, max_steps=max_steps,
                                max_input_size=max_input_size)


# ---------------------------------------------------------------------------
# System construction helpers
# ---------------------------------------------------------------------------

def make_tree_system(alphabet: RankedAlphabet,
                     init: BottomUpTreeAutomaton,
                     transition: BottomUpTreeTransducer,
                     bad: Optional[BottomUpTreeAutomaton] = None) -> TreeTransitionSystem:
    """Create a tree transition system."""
    return TreeTransitionSystem(
        alphabet=alphabet,
        init=init,
        transition=transition,
        bad=bad
    )


def make_init_from_trees(alphabet: RankedAlphabet, trees: List[Tree]) -> BottomUpTreeAutomaton:
    """Create an initial state automaton from a set of concrete trees."""
    return _buta_from_tree_set(alphabet, trees)


def make_bad_from_pattern(alphabet: RankedAlphabet,
                          pattern_transitions: List[Tuple],
                          states: List[str],
                          final_states: List[str]) -> BottomUpTreeAutomaton:
    """Create a bad-state automaton from transition specifications."""
    buta = BottomUpTreeAutomaton(alphabet)
    buta.states = set(states)
    buta.final_states = set(final_states)
    for sym_name, child_states, target in pattern_transitions:
        buta.add_transition(sym_name, child_states, target)
    return buta


# ---------------------------------------------------------------------------
# Comparison and analysis
# ---------------------------------------------------------------------------

def compare_methods(tts: TreeTransitionSystem, max_steps: int = 15,
                    max_input_size: int = 8) -> Dict:
    """Compare different model checking methods on the same system."""
    results = {}

    for method in ["forward", "bounded", "accelerated"]:
        res = check_safety(tts, max_steps=max_steps, method=method,
                           max_input_size=max_input_size)
        results[method] = {
            "result": res.result.value,
            "steps": res.steps,
            "has_counterexample": res.counterexample is not None,
            "has_invariant": res.invariant is not None
        }

    # Try backward if bad is specified
    if tts.bad is not None:
        try:
            res = check_safety(tts, max_steps=max_steps, method="backward",
                               max_input_size=max_input_size)
            results["backward"] = {
                "result": res.result.value,
                "steps": res.steps,
                "has_counterexample": res.counterexample is not None,
                "has_invariant": res.invariant is not None
            }
        except Exception:
            results["backward"] = {"result": "error"}

    return results


def system_stats(tts: TreeTransitionSystem) -> Dict:
    """Compute statistics about a tree transition system."""
    init_stats = buta_stats(tts.init)
    td_stats = transducer_stats(tts.transition)

    result = {
        "alphabet_size": len(tts.alphabet),
        "init_states": init_stats["num_states"],
        "init_transitions": init_stats["num_transitions"],
        "transducer_states": td_stats["num_states"],
        "transducer_rules": td_stats["num_rules"],
        "has_bad": tts.bad is not None,
        "has_property": tts.property_aut is not None,
    }

    if tts.bad is not None:
        bad_stats = buta_stats(tts.bad)
        result["bad_states"] = bad_stats["num_states"]
        result["bad_transitions"] = bad_stats["num_transitions"]

    return result


def model_check_summary(tts: TreeTransitionSystem,
                        max_steps: int = 15,
                        max_input_size: int = 8) -> Dict:
    """Full model checking summary: stats + results from all methods."""
    stats = system_stats(tts)
    comparison = compare_methods(tts, max_steps=max_steps,
                                 max_input_size=max_input_size)

    # Consensus
    verdicts = [v["result"] for v in comparison.values() if isinstance(v, dict) and "result" in v]
    safe_count = verdicts.count("safe")
    unsafe_count = verdicts.count("unsafe")

    if unsafe_count > 0:
        consensus = "unsafe"
    elif safe_count == len(verdicts):
        consensus = "safe"
    else:
        consensus = "unknown"

    return {
        "system": stats,
        "methods": comparison,
        "consensus": consensus,
        "methods_agree": len(set(verdicts)) <= 1
    }


# ---------------------------------------------------------------------------
# Application: list/tree manipulation verification
# ---------------------------------------------------------------------------

def verify_list_manipulation(alphabet: RankedAlphabet,
                             init_list: BottomUpTreeAutomaton,
                             transform: BottomUpTreeTransducer,
                             property_aut: BottomUpTreeAutomaton,
                             max_steps: int = 10) -> ModelCheckResult:
    """Verify that a list transformation preserves a property.

    Common patterns: sorted lists stay sorted, non-empty stays non-empty,
    length bounds are maintained.
    """
    bad = buta_complement(property_aut)
    tts = TreeTransitionSystem(
        alphabet=alphabet,
        init=init_list,
        transition=transform,
        bad=bad,
        property_aut=property_aut
    )
    return forward_reachability(tts, max_steps=max_steps)


def verify_tree_transform_preserves(
    alphabet: RankedAlphabet,
    initial: BottomUpTreeAutomaton,
    transform: BottomUpTreeTransducer,
    property_aut: BottomUpTreeAutomaton,
    max_steps: int = 10
) -> Dict:
    """High-level: verify that applying transform to initial trees preserves property.

    Returns dict with result, invariant check, and trace if unsafe.
    """
    bad = buta_complement(property_aut)
    tts = make_tree_system(alphabet, initial, transform, bad)

    result = forward_reachability(tts, max_steps=max_steps)

    return {
        "result": result.result.value,
        "steps": result.steps,
        "safe": result.result == VerificationResult.SAFE,
        "counterexample": result.counterexample,
        "invariant_found": result.invariant is not None,
    }
