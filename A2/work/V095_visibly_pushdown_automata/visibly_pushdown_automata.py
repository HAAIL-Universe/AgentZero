"""V095: Visibly Pushdown Automata (VPA)

Visibly pushdown automata are a subclass of pushdown automata where the stack
operation (push/pop/internal) is determined by the input symbol, not by the
automaton's state. This gives the class of visibly pushdown languages (VPL)
closure under union, intersection, complement, and concatenation -- properties
that general CFLs lack.

VPLs model well-matched call/return patterns in programs:
- Call symbols push onto the stack
- Return symbols pop from the stack
- Internal symbols don't touch the stack

Key capabilities:
- Determinization (unlike general PDA)
- Boolean closure (union, intersection, complement)
- Decidable emptiness, universality, inclusion, equivalence
- Language operations: concatenation, Kleene star
- Program verification: nested word model checking
- XML/HTML validation: well-nested tag structure

Composes: V094 (pushdown systems concepts)

References:
- Alur, Madhusudan: "Visibly pushdown languages" (STOC 2004)
- Alur: "Marrying words and trees" (PODS 2007)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import deque
from enum import Enum
from itertools import product as cartesian_product


# --- Alphabet Classification ---

@dataclass(frozen=True)
class VisibleAlphabet:
    """A visibly pushdown alphabet partitions symbols into three disjoint sets."""
    calls: FrozenSet[str]      # Call symbols: push onto stack
    returns: FrozenSet[str]    # Return symbols: pop from stack
    internals: FrozenSet[str]  # Internal symbols: no stack operation

    def __post_init__(self):
        assert not (self.calls & self.returns), "calls and returns must be disjoint"
        assert not (self.calls & self.internals), "calls and internals must be disjoint"
        assert not (self.returns & self.internals), "returns and internals must be disjoint"

    @property
    def all_symbols(self) -> FrozenSet[str]:
        return self.calls | self.returns | self.internals

    def classify(self, symbol: str) -> str:
        """Return 'call', 'return', or 'internal'."""
        if symbol in self.calls:
            return "call"
        elif symbol in self.returns:
            return "return"
        elif symbol in self.internals:
            return "internal"
        raise ValueError(f"Symbol '{symbol}' not in alphabet")


# --- VPA Data Structures ---

# Special bottom-of-stack marker
STACK_BOTTOM = "_bot_"


@dataclass(frozen=True)
class VPATransition:
    """A VPA transition."""
    source: str
    symbol: str
    target: str
    # For call: stack_push is the symbol pushed
    # For return: stack_pop is the symbol that must be on top
    # For internal: both are None
    stack_push: Optional[str] = None
    stack_pop: Optional[str] = None


@dataclass
class VPA:
    """Visibly Pushdown Automaton.

    A VPA = (Q, Sigma, Gamma, delta, Q0, QF, q_bot) where:
    - Q: finite set of states
    - Sigma: visibly pushdown alphabet (calls, returns, internals)
    - Gamma: stack alphabet (states x calls, plus bottom marker)
    - delta: transition relation
    - Q0: initial states
    - QF: accepting states
    """
    alphabet: VisibleAlphabet
    states: Set[str] = field(default_factory=set)
    initial_states: Set[str] = field(default_factory=set)
    accepting_states: Set[str] = field(default_factory=set)
    # call transitions: (state, call_symbol) -> set of (target_state, stack_symbol)
    call_transitions: Dict[Tuple[str, str], Set[Tuple[str, str]]] = field(default_factory=dict)
    # return transitions: (state, return_symbol, stack_symbol) -> set of target_states
    return_transitions: Dict[Tuple[str, str, str], Set[str]] = field(default_factory=dict)
    # internal transitions: (state, internal_symbol) -> set of target_states
    internal_transitions: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)

    def add_state(self, state: str, initial: bool = False, accepting: bool = False):
        """Add a state."""
        self.states.add(state)
        if initial:
            self.initial_states.add(state)
        if accepting:
            self.accepting_states.add(state)

    def add_call_transition(self, source: str, symbol: str, target: str, stack_sym: str):
        """Add call transition: on call symbol, push stack_sym and go to target."""
        assert symbol in self.alphabet.calls, f"'{symbol}' is not a call symbol"
        self.states.add(source)
        self.states.add(target)
        key = (source, symbol)
        if key not in self.call_transitions:
            self.call_transitions[key] = set()
        self.call_transitions[key].add((target, stack_sym))

    def add_return_transition(self, source: str, symbol: str, stack_sym: str, target: str):
        """Add return transition: on return symbol with stack_sym on top, pop and go to target."""
        assert symbol in self.alphabet.returns, f"'{symbol}' is not a return symbol"
        self.states.add(source)
        self.states.add(target)
        key = (source, symbol, stack_sym)
        if key not in self.return_transitions:
            self.return_transitions[key] = set()
        self.return_transitions[key].add(target)

    def add_internal_transition(self, source: str, symbol: str, target: str):
        """Add internal transition: on internal symbol, no stack change."""
        assert symbol in self.alphabet.internals, f"'{symbol}' is not an internal symbol"
        self.states.add(source)
        self.states.add(target)
        key = (source, symbol)
        if key not in self.internal_transitions:
            self.internal_transitions[key] = set()
        self.internal_transitions[key].add(target)

    def is_deterministic(self) -> bool:
        """Check if VPA is deterministic."""
        if len(self.initial_states) > 1:
            return False
        for targets in self.call_transitions.values():
            if len(targets) > 1:
                return False
        for targets in self.return_transitions.values():
            if len(targets) > 1:
                return False
        for targets in self.internal_transitions.values():
            if len(targets) > 1:
                return False
        return True


# --- Acceptance / Simulation ---

def run_vpa(vpa: VPA, word: List[str], empty_stack: bool = False) -> bool:
    """Run a VPA on a word. Returns True if accepted (nondeterministic).

    If empty_stack=True, acceptance requires both final state AND stack
    contains only the bottom marker (well-matched word acceptance).
    """
    # Configuration: (state, stack) where stack is a tuple
    # Stack grows to the right: stack[-1] is top
    initial_configs = set()
    for q0 in vpa.initial_states:
        initial_configs.add((q0, (STACK_BOTTOM,)))

    current = initial_configs

    for sym in word:
        kind = vpa.alphabet.classify(sym)
        next_configs = set()

        for (state, stack) in current:
            if kind == "call":
                key = (state, sym)
                if key in vpa.call_transitions:
                    for (target, push_sym) in vpa.call_transitions[key]:
                        next_configs.add((target, stack + (push_sym,)))

            elif kind == "return":
                if len(stack) > 0:
                    top = stack[-1]
                    if top == STACK_BOTTOM:
                        # Return on empty stack: try STACK_BOTTOM transition
                        key = (state, sym, STACK_BOTTOM)
                        if key in vpa.return_transitions:
                            for target in vpa.return_transitions[key]:
                                next_configs.add((target, (STACK_BOTTOM,)))
                    else:
                        key = (state, sym, top)
                        if key in vpa.return_transitions:
                            for target in vpa.return_transitions[key]:
                                next_configs.add((target, stack[:-1]))

            elif kind == "internal":
                key = (state, sym)
                if key in vpa.internal_transitions:
                    for target in vpa.internal_transitions[key]:
                        next_configs.add((target, stack))

        current = next_configs
        if not current:
            return False

    # Accept if any config is in accepting state
    for (state, stack) in current:
        if state in vpa.accepting_states:
            if empty_stack:
                if stack == (STACK_BOTTOM,):
                    return True
            else:
                return True
    return False


# --- Determinization ---

def determinize_vpa(vpa: VPA) -> VPA:
    """Determinize a VPA using the subset construction for VPAs.

    Unlike general PDA, VPAs can always be determinized.
    The key insight: on call symbols, we push the current macro-state onto the stack
    so that on matching returns, we know which macro-state to return to.
    """
    alpha = vpa.alphabet
    det = VPA(alphabet=alpha)

    # Macro-state = frozenset of original states
    init_macro = frozenset(vpa.initial_states)
    if not init_macro:
        return det

    def state_name(macro: FrozenSet[str]) -> str:
        return "{" + ",".join(sorted(macro)) + "}"

    init_name = state_name(init_macro)
    det.add_state(init_name, initial=True)
    if init_macro & vpa.accepting_states:
        det.accepting_states.add(init_name)

    # Stack alphabet for determinized VPA: (macro_state, call_symbol) pairs
    # We encode as strings for simplicity
    def stack_sym_name(macro: FrozenSet[str], call_sym: str) -> str:
        return f"[{state_name(macro)}|{call_sym}]"

    worklist = deque([init_macro])
    visited_macros = {init_macro}
    # For return transitions, we also need to explore (macro, stack_sym) pairs
    # We track which stack symbols can appear
    stack_syms_seen = set()  # frozenset pairs (macro, call_sym)

    # First pass: discover all reachable macro states via call and internal transitions
    # We need multiple passes because return transitions depend on stack content
    # which depends on call transitions

    # Actually, for VPA determinization, we need to track which (macro, stack_sym)
    # combinations are reachable. The stack symbol pushed is (calling_macro, call_sym).
    # On a return, we pop (calling_macro, call_sym) and need the return transition
    # for the current macro + return symbol + calling_macro.

    # Full construction: explore (macro_state) for call/internal,
    # and (macro_state, popped_macro, call_sym) for returns.

    all_macros = {}  # macro -> name
    all_macros[init_macro] = init_name

    def get_or_create(macro):
        if macro in all_macros:
            return all_macros[macro]
        name = state_name(macro)
        all_macros[macro] = name
        det.add_state(name)
        if macro & vpa.accepting_states:
            det.accepting_states.add(name)
        return name

    get_or_create(init_macro)

    while worklist:
        macro = worklist.popleft()
        macro_name = all_macros[macro]

        # Call transitions
        for c in alpha.calls:
            target_set = set()
            for q in macro:
                key = (q, c)
                if key in vpa.call_transitions:
                    for (t, _push) in vpa.call_transitions[key]:
                        target_set.add(t)
            if target_set:
                target_macro = frozenset(target_set)
                target_name = get_or_create(target_macro)
                # Push the calling macro-state paired with call symbol
                push = stack_sym_name(macro, c)
                det.add_call_transition(macro_name, c, target_name, push)
                pair = (macro, c)
                if pair not in stack_syms_seen:
                    stack_syms_seen.add(pair)
                if target_macro not in visited_macros:
                    visited_macros.add(target_macro)
                    worklist.append(target_macro)

        # Internal transitions
        for i in alpha.internals:
            target_set = set()
            for q in macro:
                key = (q, i)
                if key in vpa.internal_transitions:
                    for t in vpa.internal_transitions[key]:
                        target_set.add(t)
            if target_set:
                target_macro = frozenset(target_set)
                target_name = get_or_create(target_macro)
                det.add_internal_transition(macro_name, i, target_name)
                if target_macro not in visited_macros:
                    visited_macros.add(target_macro)
                    worklist.append(target_macro)

    # Return transitions: for each (current_macro, return_sym, calling_macro, call_sym)
    # The original VPA had transitions (q, r, gamma) -> q'
    # In the original, gamma is whatever was pushed by the call.
    # In the determinized version, we pushed (calling_macro, call_sym).
    # The return target depends on which original states were in the calling_macro
    # and which pushed what.
    for current_macro in list(all_macros.keys()):
        current_name = all_macros[current_macro]
        for r in alpha.returns:
            # For each possible stack symbol (calling_macro, call_sym)
            for (calling_macro, call_sym) in stack_syms_seen:
                target_set = set()
                for q in current_macro:
                    # Which original stack symbols could have been pushed when
                    # we were in calling_macro and read call_sym?
                    for q_caller in calling_macro:
                        key_call = (q_caller, call_sym)
                        if key_call in vpa.call_transitions:
                            for (_t, push_sym) in vpa.call_transitions[key_call]:
                                key_ret = (q, r, push_sym)
                                if key_ret in vpa.return_transitions:
                                    for t in vpa.return_transitions[key_ret]:
                                        target_set.add(t)
                if target_set:
                    target_macro = frozenset(target_set)
                    target_name = get_or_create(target_macro)
                    stack_sym = stack_sym_name(calling_macro, call_sym)
                    det.add_return_transition(current_name, r, stack_sym, target_name)
                    if target_macro not in visited_macros:
                        visited_macros.add(target_macro)
                        worklist.append(target_macro)

    # May need more iterations for newly discovered macros from returns
    while worklist:
        macro = worklist.popleft()
        macro_name = all_macros[macro]
        for c in alpha.calls:
            target_set = set()
            for q in macro:
                key = (q, c)
                if key in vpa.call_transitions:
                    for (t, _push) in vpa.call_transitions[key]:
                        target_set.add(t)
            if target_set:
                target_macro = frozenset(target_set)
                target_name = get_or_create(target_macro)
                push = stack_sym_name(macro, c)
                det.add_call_transition(macro_name, c, target_name, push)
                pair = (macro, c)
                if pair not in stack_syms_seen:
                    stack_syms_seen.add(pair)
                if target_macro not in visited_macros:
                    visited_macros.add(target_macro)
                    worklist.append(target_macro)
        for i in alpha.internals:
            target_set = set()
            for q in macro:
                key = (q, i)
                if key in vpa.internal_transitions:
                    for t in vpa.internal_transitions[key]:
                        target_set.add(t)
            if target_set:
                target_macro = frozenset(target_set)
                target_name = get_or_create(target_macro)
                det.add_internal_transition(macro_name, i, target_name)
                if target_macro not in visited_macros:
                    visited_macros.add(target_macro)
                    worklist.append(target_macro)
        for r in alpha.returns:
            for (calling_macro, call_sym) in stack_syms_seen:
                target_set = set()
                for q in macro:
                    for q_caller in calling_macro:
                        key_call = (q_caller, call_sym)
                        if key_call in vpa.call_transitions:
                            for (_t, push_sym) in vpa.call_transitions[key_call]:
                                key_ret = (q, r, push_sym)
                                if key_ret in vpa.return_transitions:
                                    for t in vpa.return_transitions[key_ret]:
                                        target_set.add(t)
                if target_set:
                    target_macro = frozenset(target_set)
                    target_name = get_or_create(target_macro)
                    stack_sym = stack_sym_name(calling_macro, call_sym)
                    det.add_return_transition(current_name, r, stack_sym, target_name)
                    if target_macro not in visited_macros:
                        visited_macros.add(target_macro)
                        worklist.append(target_macro)

    return det


# --- Complement ---

def complement_vpa(vpa: VPA) -> VPA:
    """Complement a VPA. First determinizes if needed, then swaps accepting/non-accepting."""
    if not vpa.is_deterministic():
        dvpa = determinize_vpa(vpa)
    else:
        dvpa = vpa

    # Need to make the automaton complete (add sink state for missing transitions)
    comp = VPA(alphabet=dvpa.alphabet)
    sink = "__sink__"
    need_sink = False

    for s in dvpa.states:
        comp.add_state(s, s in dvpa.initial_states, s not in dvpa.accepting_states)
    # sink is non-accepting (since we flip, it was "accepting" before = not in original accepting)

    alpha = dvpa.alphabet

    # Copy existing transitions
    for key, targets in dvpa.call_transitions.items():
        for (t, push) in targets:
            comp.add_call_transition(key[0], key[1], t, push)
    for key, targets in dvpa.return_transitions.items():
        for t in targets:
            comp.add_return_transition(key[0], key[1], key[2], t)
    for key, targets in dvpa.internal_transitions.items():
        for t in targets:
            comp.add_internal_transition(key[0], key[1], t)

    # Add sink transitions for completeness
    all_states = set(dvpa.states)
    all_stack_syms = set()
    for key in dvpa.return_transitions:
        all_stack_syms.add(key[2])
    # Also add STACK_BOTTOM
    all_stack_syms.add(STACK_BOTTOM)

    for s in all_states:
        for c in alpha.calls:
            if (s, c) not in dvpa.call_transitions:
                need_sink = True
                comp.add_call_transition(s, c, sink, "__sink_push__")
        for i in alpha.internals:
            if (s, i) not in dvpa.internal_transitions:
                need_sink = True
                comp.add_internal_transition(s, i, sink)
        for r in alpha.returns:
            for gamma in all_stack_syms:
                if (s, r, gamma) not in dvpa.return_transitions:
                    need_sink = True
                    comp.add_return_transition(s, r, gamma, sink)

    if need_sink:
        comp.add_state(sink, accepting=True)  # flipped: sink was non-accepting -> now accepting
        # Wait, we want complement. Original accepting = non-accepting in complement.
        # Sink state was not in original accepting, so in complement it IS accepting.
        # But sink means "reject" in original -> "accept" in complement? That's wrong for
        # words that go to sink on an unmatched return with empty stack.
        # Actually: in a complete DFA-like VPA, sink is a dead state. In original, words
        # ending in sink are rejected. In complement, they should be accepted.
        # This is correct: complement accepts exactly what the original rejects.

        # Self-loops on sink for all symbols
        for c in alpha.calls:
            comp.add_call_transition(sink, c, sink, "__sink_push__")
        for i in alpha.internals:
            comp.add_internal_transition(sink, i, sink)
        for r in alpha.returns:
            for gamma in all_stack_syms | {"__sink_push__"}:
                comp.add_return_transition(sink, r, gamma, sink)

    return comp


# --- Intersection ---

def intersect_vpa(vpa1: VPA, vpa2: VPA) -> VPA:
    """Intersect two VPAs using product construction.

    Unlike general PDA, VPAs are closed under intersection because
    the stack operations are determined by the input symbol, so both
    automata do the same stack operation on each symbol.
    """
    assert vpa1.alphabet == vpa2.alphabet, "VPAs must share alphabet"
    alpha = vpa1.alphabet

    result = VPA(alphabet=alpha)

    def pair_name(s1: str, s2: str) -> str:
        return f"({s1},{s2})"

    def stack_pair(g1: str, g2: str) -> str:
        return f"<{g1},{g2}>"

    # Initial states: product of initial states
    for q1 in vpa1.initial_states:
        for q2 in vpa2.initial_states:
            name = pair_name(q1, q2)
            result.add_state(name, initial=True)
            if q1 in vpa1.accepting_states and q2 in vpa2.accepting_states:
                result.accepting_states.add(name)

    # BFS exploration
    worklist = deque()
    visited = set()

    for q1 in vpa1.initial_states:
        for q2 in vpa2.initial_states:
            worklist.append((q1, q2))
            visited.add((q1, q2))

    while worklist:
        s1, s2 = worklist.popleft()
        src = pair_name(s1, s2)

        # Call transitions
        for c in alpha.calls:
            k1 = (s1, c)
            k2 = (s2, c)
            if k1 in vpa1.call_transitions and k2 in vpa2.call_transitions:
                for (t1, g1) in vpa1.call_transitions[k1]:
                    for (t2, g2) in vpa2.call_transitions[k2]:
                        tgt = pair_name(t1, t2)
                        result.add_state(tgt)
                        if t1 in vpa1.accepting_states and t2 in vpa2.accepting_states:
                            result.accepting_states.add(tgt)
                        result.add_call_transition(src, c, tgt, stack_pair(g1, g2))
                        if (t1, t2) not in visited:
                            visited.add((t1, t2))
                            worklist.append((t1, t2))

        # Internal transitions
        for i in alpha.internals:
            k1 = (s1, i)
            k2 = (s2, i)
            if k1 in vpa1.internal_transitions and k2 in vpa2.internal_transitions:
                for t1 in vpa1.internal_transitions[k1]:
                    for t2 in vpa2.internal_transitions[k2]:
                        tgt = pair_name(t1, t2)
                        result.add_state(tgt)
                        if t1 in vpa1.accepting_states and t2 in vpa2.accepting_states:
                            result.accepting_states.add(tgt)
                        result.add_internal_transition(src, i, tgt)
                        if (t1, t2) not in visited:
                            visited.add((t1, t2))
                            worklist.append((t1, t2))

        # Return transitions: need to match stack symbols
        for r in alpha.returns:
            # Collect all stack symbols used in both VPAs' return transitions
            gamma1_set = set()
            gamma2_set = set()
            for (st, sym, gamma) in vpa1.return_transitions:
                if st == s1 and sym == r:
                    gamma1_set.add(gamma)
            for (st, sym, gamma) in vpa2.return_transitions:
                if st == s2 and sym == r:
                    gamma2_set.add(gamma)

            for g1 in gamma1_set:
                for g2 in gamma2_set:
                    k1 = (s1, r, g1)
                    k2 = (s2, r, g2)
                    if k1 in vpa1.return_transitions and k2 in vpa2.return_transitions:
                        for t1 in vpa1.return_transitions[k1]:
                            for t2 in vpa2.return_transitions[k2]:
                                tgt = pair_name(t1, t2)
                                result.add_state(tgt)
                                if t1 in vpa1.accepting_states and t2 in vpa2.accepting_states:
                                    result.accepting_states.add(tgt)
                                result.add_return_transition(src, r, stack_pair(g1, g2), tgt)
                                if (t1, t2) not in visited:
                                    visited.add((t1, t2))
                                    worklist.append((t1, t2))

    return result


# --- Union ---

def union_vpa(vpa1: VPA, vpa2: VPA) -> VPA:
    """Union of two VPAs. Uses nondeterminism: initial states from both."""
    assert vpa1.alphabet == vpa2.alphabet, "VPAs must share alphabet"
    alpha = vpa1.alphabet

    result = VPA(alphabet=alpha)

    # Prefix states to avoid name collisions
    def name1(s): return f"L.{s}"
    def name2(s): return f"R.{s}"

    # Add all states from vpa1
    for s in vpa1.states:
        result.add_state(name1(s),
                         initial=(s in vpa1.initial_states),
                         accepting=(s in vpa1.accepting_states))

    # Add all states from vpa2
    for s in vpa2.states:
        result.add_state(name2(s),
                         initial=(s in vpa2.initial_states),
                         accepting=(s in vpa2.accepting_states))

    # Copy transitions from vpa1
    for (src, sym), targets in vpa1.call_transitions.items():
        for (tgt, push) in targets:
            result.add_call_transition(name1(src), sym, name1(tgt), f"L.{push}")
    for (src, sym, gamma), targets in vpa1.return_transitions.items():
        for tgt in targets:
            result.add_return_transition(name1(src), sym, f"L.{gamma}", name1(tgt))
    for (src, sym), targets in vpa1.internal_transitions.items():
        for tgt in targets:
            result.add_internal_transition(name1(src), sym, name1(tgt))

    # Copy transitions from vpa2
    for (src, sym), targets in vpa2.call_transitions.items():
        for (tgt, push) in targets:
            result.add_call_transition(name2(src), sym, name2(tgt), f"R.{push}")
    for (src, sym, gamma), targets in vpa2.return_transitions.items():
        for tgt in targets:
            result.add_return_transition(name2(src), sym, f"R.{gamma}", name2(tgt))
    for (src, sym), targets in vpa2.internal_transitions.items():
        for tgt in targets:
            result.add_internal_transition(name2(src), sym, name2(tgt))

    return result


# --- Emptiness Check ---

def check_emptiness(vpa: VPA) -> dict:
    """Check if the language of a VPA is empty.

    Returns {empty: bool, witness: Optional[List[str]]}.
    Uses BFS over configurations (state, stack_height_abstraction).
    For emptiness, we use a summary-based approach.
    """
    # Summary-based emptiness: find path from initial to accepting
    # that respects matched call/return structure.
    # A summary (p, q) means: there exists a word that takes state p
    # to state q with the same stack height.

    alpha = vpa.alphabet

    # Compute summaries bottom-up
    # summaries[p] = set of states q reachable from p with matching call/return
    summaries = {}  # (src, tgt) pairs representing same-level reachability

    # Base: internal transitions give immediate summaries
    changed = True
    summary_set = set()

    # Identity summaries (empty word)
    for q in vpa.states:
        summary_set.add((q, q))

    # Internal transitions
    for (src, sym), targets in vpa.internal_transitions.items():
        for tgt in targets:
            summary_set.add((src, tgt))

    # Iterate until fixpoint
    while changed:
        changed = False
        new_summaries = set()

        # Compose summaries: if (p, q) and (q, r) then (p, r)
        by_src = {}
        for (s, t) in summary_set:
            if s not in by_src:
                by_src[s] = set()
            by_src[s].add(t)

        for (s, mid) in list(summary_set):
            if mid in by_src:
                for t in by_src[mid]:
                    pair = (s, t)
                    if pair not in summary_set:
                        new_summaries.add(pair)

        # Call-return pairs: if (p, c) -> (q, gamma) and summary (q, r)
        # and (r, ret, gamma) -> s, then (p, s) is a summary
        for (src, call_sym), call_targets in vpa.call_transitions.items():
            for (call_tgt, gamma) in call_targets:
                # Find summaries from call_tgt
                if call_tgt in by_src:
                    for mid in by_src[call_tgt]:
                        # Find return transitions from mid
                        for ret_sym in alpha.returns:
                            key = (mid, ret_sym, gamma)
                            if key in vpa.return_transitions:
                                for ret_tgt in vpa.return_transitions[key]:
                                    pair = (src, ret_tgt)
                                    if pair not in summary_set:
                                        new_summaries.add(pair)

        if new_summaries:
            summary_set.update(new_summaries)
            changed = True

    # Check if any initial state has a summary to an accepting state
    for q0 in vpa.initial_states:
        for qf in vpa.accepting_states:
            if (q0, qf) in summary_set:
                # Find a witness word
                witness = _find_witness(vpa, q0, qf, summary_set)
                return {"empty": False, "witness": witness}

    return {"empty": True, "witness": None}


def _find_witness(vpa, start, end, summaries):
    """Find a witness word from start to end state using BFS."""
    alpha = vpa.alphabet
    # BFS over (state, stack, word)
    queue = deque()
    queue.append((start, (STACK_BOTTOM,), []))
    visited = set()
    visited.add((start, (STACK_BOTTOM,)))

    max_steps = 10000
    steps = 0

    while queue and steps < max_steps:
        steps += 1
        state, stack, word = queue.popleft()

        if state == end and len(word) > 0:
            return word
        if state == end and start == end and len(word) == 0:
            # Need at least empty word
            pass

        # Internal transitions
        for i in alpha.internals:
            key = (state, i)
            if key in vpa.internal_transitions:
                for tgt in vpa.internal_transitions[key]:
                    config = (tgt, stack)
                    if config not in visited:
                        visited.add(config)
                        queue.append((tgt, stack, word + [i]))

        # Call transitions
        for c in alpha.calls:
            key = (state, c)
            if key in vpa.call_transitions:
                for (tgt, push) in vpa.call_transitions[key]:
                    new_stack = stack + (push,)
                    if len(new_stack) <= 20:  # bounded stack depth
                        config = (tgt, new_stack)
                        if config not in visited:
                            visited.add(config)
                            queue.append((tgt, new_stack, word + [c]))

        # Return transitions
        if len(stack) > 0:
            top = stack[-1]
            for r in alpha.returns:
                key = (state, r, top)
                if key in vpa.return_transitions:
                    for tgt in vpa.return_transitions[key]:
                        new_stack = stack[:-1] if top != STACK_BOTTOM else stack
                        config = (tgt, new_stack)
                        if config not in visited:
                            visited.add(config)
                            queue.append((tgt, new_stack, word + [r]))

    # If start == end and we found no non-empty word, empty word is the witness
    if start == end:
        return []
    return None


# --- Equivalence and Inclusion ---

def check_inclusion(vpa1: VPA, vpa2: VPA) -> dict:
    """Check if L(vpa1) is a subset of L(vpa2).

    L(vpa1) subset L(vpa2) iff L(vpa1) intersect complement(L(vpa2)) is empty.
    """
    comp2 = complement_vpa(vpa2)
    intersection = intersect_vpa(vpa1, comp2)
    result = check_emptiness(intersection)
    return {
        "included": result["empty"],
        "counterexample": result["witness"]
    }


def check_equivalence(vpa1: VPA, vpa2: VPA) -> dict:
    """Check if L(vpa1) == L(vpa2).

    Equivalent iff L(vpa1) subset L(vpa2) and L(vpa2) subset L(vpa1).
    """
    inc1 = check_inclusion(vpa1, vpa2)
    if not inc1["included"]:
        return {
            "equivalent": False,
            "counterexample": inc1["counterexample"],
            "direction": "in L1 but not L2"
        }
    inc2 = check_inclusion(vpa2, vpa1)
    if not inc2["included"]:
        return {
            "equivalent": False,
            "counterexample": inc2["counterexample"],
            "direction": "in L2 but not L1"
        }
    return {"equivalent": True, "counterexample": None, "direction": None}


# --- Universality ---

def check_universality(vpa: VPA) -> dict:
    """Check if L(vpa) = Sigma*."""
    comp = complement_vpa(vpa)
    result = check_emptiness(comp)
    return {
        "universal": result["empty"],
        "counterexample": result["witness"]
    }


# --- Concatenation ---

def concatenate_vpa(vpa1: VPA, vpa2: VPA) -> VPA:
    """Concatenate two VPAs: L(result) = {uv | u in L(vpa1), v in L(vpa2)}.

    On accepting vpa1, epsilon-transition to initial states of vpa2.
    Only works for well-matched words (stack empty at concatenation point).
    """
    assert vpa1.alphabet == vpa2.alphabet
    alpha = vpa1.alphabet

    result = VPA(alphabet=alpha)

    def name1(s): return f"A.{s}"
    def name2(s): return f"B.{s}"

    # Copy vpa1 states (initial states of result = initial of vpa1)
    for s in vpa1.states:
        is_init = s in vpa1.initial_states
        result.add_state(name1(s), initial=is_init)
        # vpa1 accepting states are NOT accepting in result (they transition to vpa2)

    # Copy vpa2 states (accepting states of result = accepting of vpa2)
    for s in vpa2.states:
        is_accept = s in vpa2.accepting_states
        result.add_state(name2(s), accepting=is_accept)

    # Also: if vpa1 has an accepting initial state (epsilon in L(vpa1)),
    # then vpa2's initial states are also initial in result
    for q0 in vpa1.initial_states:
        if q0 in vpa1.accepting_states:
            for q2 in vpa2.initial_states:
                result.initial_states.add(name2(q2))

    # Copy vpa1 transitions
    for (src, sym), targets in vpa1.call_transitions.items():
        for (tgt, push) in targets:
            result.add_call_transition(name1(src), sym, name1(tgt), f"A.{push}")
    for (src, sym, gamma), targets in vpa1.return_transitions.items():
        for tgt in targets:
            result.add_return_transition(name1(src), sym, f"A.{gamma}", name1(tgt))
    for (src, sym), targets in vpa1.internal_transitions.items():
        for tgt in targets:
            result.add_internal_transition(name1(src), sym, name1(tgt))

    # Copy vpa2 transitions
    for (src, sym), targets in vpa2.call_transitions.items():
        for (tgt, push) in targets:
            result.add_call_transition(name2(src), sym, name2(tgt), f"B.{push}")
    for (src, sym, gamma), targets in vpa2.return_transitions.items():
        for tgt in targets:
            result.add_return_transition(name2(src), sym, f"B.{gamma}", name2(tgt))
    for (src, sym), targets in vpa2.internal_transitions.items():
        for tgt in targets:
            result.add_internal_transition(name2(src), sym, name2(tgt))

    # Epsilon transitions from vpa1 accepting to vpa2 initial
    # Simulated by duplicating transitions: for each (vpa1_accepting, symbol) transition,
    # also add transition from vpa2_initial states
    for qa in vpa1.accepting_states:
        for q2_init in vpa2.initial_states:
            # For every transition OUT of any vpa2 initial state, also make it
            # available from the vpa1 accepting state perspective
            # Actually, the standard approach: make vpa1 accepting states also
            # behave like vpa2 initial states by copying outgoing transitions
            for c in alpha.calls:
                key = (q2_init, c)
                if key in vpa2.call_transitions:
                    for (tgt, push) in vpa2.call_transitions[key]:
                        result.add_call_transition(name1(qa), c, name2(tgt), f"B.{push}")
            for i in alpha.internals:
                key = (q2_init, i)
                if key in vpa2.internal_transitions:
                    for tgt in vpa2.internal_transitions[key]:
                        result.add_internal_transition(name1(qa), i, name2(tgt))
            for r in alpha.returns:
                # Return from vpa1 accepting state: need vpa1's stack symbols
                for (src2, sym2, gamma2), targets in vpa2.return_transitions.items():
                    if src2 == q2_init and sym2 == r:
                        # We can't easily handle this because the stack has vpa1's symbols
                        # For well-matched concatenation, at the junction the stack is
                        # at the same level as the start, so this shouldn't happen
                        # (returns without matching calls = unmatched returns)
                        pass

    # If epsilon is in L(vpa2), then vpa1 accepting states are also result accepting
    for q2 in vpa2.initial_states:
        if q2 in vpa2.accepting_states:
            for qa in vpa1.accepting_states:
                result.accepting_states.add(name1(qa))

    return result


# --- Kleene Star ---

def kleene_star_vpa(vpa: VPA) -> VPA:
    """Kleene star of a VPA: L(result) = L(vpa)*.

    Only works correctly for well-matched words.
    """
    alpha = vpa.alphabet
    result = VPA(alphabet=alpha)

    # New initial/accepting state for epsilon
    start = "__star_init__"
    result.add_state(start, initial=True, accepting=True)

    # Copy all states
    for s in vpa.states:
        result.add_state(s)

    # Copy transitions
    for key, targets in vpa.call_transitions.items():
        for (tgt, push) in targets:
            result.add_call_transition(key[0], key[1], tgt, push)
    for key, targets in vpa.return_transitions.items():
        for tgt in targets:
            result.add_return_transition(key[0], key[1], key[2], tgt)
    for key, targets in vpa.internal_transitions.items():
        for tgt in targets:
            result.add_internal_transition(key[0], key[1], tgt)

    # Start state behaves like all initial states
    for q0 in vpa.initial_states:
        for c in alpha.calls:
            key = (q0, c)
            if key in vpa.call_transitions:
                for (tgt, push) in vpa.call_transitions[key]:
                    result.add_call_transition(start, c, tgt, push)
        for i in alpha.internals:
            key = (q0, i)
            if key in vpa.internal_transitions:
                for tgt in vpa.internal_transitions[key]:
                    result.add_internal_transition(start, i, tgt)
        for r in alpha.returns:
            for (src, sym, gamma), targets in vpa.return_transitions.items():
                if src == q0 and sym == r:
                    for tgt in targets:
                        result.add_return_transition(start, r, gamma, tgt)

    # From accepting states, epsilon-transition back to start
    # (simulate by copying start's outgoing transitions to accepting states)
    for qa in vpa.accepting_states:
        result.accepting_states.add(qa)
        for q0 in vpa.initial_states:
            for c in alpha.calls:
                key = (q0, c)
                if key in vpa.call_transitions:
                    for (tgt, push) in vpa.call_transitions[key]:
                        result.add_call_transition(qa, c, tgt, push)
            for i in alpha.internals:
                key = (q0, i)
                if key in vpa.internal_transitions:
                    for tgt in vpa.internal_transitions[key]:
                        result.add_internal_transition(qa, i, tgt)

    return result


# --- Nested Word Model ---

@dataclass
class NestedWord:
    """A nested word: linear word + nesting relation (matching call/return pairs)."""
    word: List[str]
    nesting: List[Tuple[int, int]]  # (call_position, return_position) pairs

    @staticmethod
    def from_word(word: List[str], alphabet: VisibleAlphabet) -> 'NestedWord':
        """Construct nested word from linear word + alphabet classification."""
        nesting = []
        stack = []  # stack of call positions
        for i, sym in enumerate(word):
            kind = alphabet.classify(sym)
            if kind == "call":
                stack.append(i)
            elif kind == "return":
                if stack:
                    call_pos = stack.pop()
                    nesting.append((call_pos, i))
        return NestedWord(word=word, nesting=nesting)

    def is_well_matched(self) -> bool:
        """Check if all calls have matching returns and vice versa."""
        call_positions = {c for c, r in self.nesting}
        return_positions = {r for c, r in self.nesting}
        # Every position should be in at most one pair
        return len(call_positions) == len(self.nesting) and len(return_positions) == len(self.nesting)

    @property
    def depth(self) -> int:
        """Maximum nesting depth."""
        if not self.word:
            return 0
        max_depth = 0
        current = 0
        call_set = {c for c, r in self.nesting}
        return_set = {r for c, r in self.nesting}
        for i in range(len(self.word)):
            if i in call_set:
                current += 1
                max_depth = max(max_depth, current)
            elif i in return_set:
                current -= 1
        return max_depth


# --- Program Verification Applications ---

def make_balanced_parens_vpa(alphabet: VisibleAlphabet) -> VPA:
    """Create a VPA that accepts all well-matched words (Dyck language)."""
    vpa = VPA(alphabet=alphabet)
    vpa.add_state("q", initial=True, accepting=True)

    # For each call symbol, push a marker and stay in q
    for c in alphabet.calls:
        vpa.add_call_transition("q", c, "q", f"mark_{c}")

    # For each return symbol, pop the corresponding marker
    for r in alphabet.returns:
        for c in alphabet.calls:
            vpa.add_return_transition("q", r, f"mark_{c}", "q")

    # Internal symbols: stay in q
    for i in alphabet.internals:
        vpa.add_internal_transition("q", i, "q")

    return vpa


def make_matched_call_return_vpa(call_sym: str, ret_sym: str,
                                  internals: FrozenSet[str] = frozenset()) -> VPA:
    """Create a VPA for matched call/return pairs with optional internals."""
    alpha = VisibleAlphabet(
        calls=frozenset({call_sym}),
        returns=frozenset({ret_sym}),
        internals=internals
    )
    return make_balanced_parens_vpa(alpha)


def make_bounded_depth_vpa(alphabet: VisibleAlphabet, max_depth: int) -> VPA:
    """Create a VPA that accepts well-matched words with nesting depth <= max_depth."""
    vpa = VPA(alphabet=alphabet)

    # States: d0, d1, ..., d_{max_depth} representing current nesting depth
    for d in range(max_depth + 1):
        name = f"d{d}"
        vpa.add_state(name, initial=(d == 0), accepting=(d == 0))

    # Call transitions: di -> d(i+1), push marker
    for d in range(max_depth):
        src = f"d{d}"
        tgt = f"d{d+1}"
        for c in alphabet.calls:
            vpa.add_call_transition(src, c, tgt, f"depth_{d}")

    # Return transitions: d(i+1) -> di, pop marker
    for d in range(max_depth):
        src = f"d{d+1}"
        tgt = f"d{d}"
        for r in alphabet.returns:
            vpa.add_return_transition(src, r, f"depth_{d}", tgt)

    # Internal transitions: stay at same depth
    for d in range(max_depth + 1):
        name = f"d{d}"
        for i in alphabet.internals:
            vpa.add_internal_transition(name, i, name)

    return vpa


def make_call_return_pattern_vpa(alphabet: VisibleAlphabet,
                                  pattern: List[str]) -> VPA:
    """Create a VPA that accepts words containing a specific sequence of internals.

    The pattern must consist only of internal symbols.
    """
    vpa = VPA(alphabet=alphabet)
    n = len(pattern)

    # States: p0, p1, ..., p_n where p_n is accepting
    for i in range(n + 1):
        vpa.add_state(f"p{i}", initial=(i == 0), accepting=(i == n))

    # Advance pattern on matching internal
    for i in range(n):
        target_sym = pattern[i]
        assert target_sym in alphabet.internals, f"Pattern symbol must be internal: {target_sym}"
        vpa.add_internal_transition(f"p{i}", target_sym, f"p{i+1}")
        # Non-matching internals: stay
        for other in alphabet.internals:
            if other != target_sym:
                vpa.add_internal_transition(f"p{i}", other, f"p{i}")

    # After pattern matched, stay in accepting state
    for i_sym in alphabet.internals:
        vpa.add_internal_transition(f"p{n}", i_sym, f"p{n}")

    # Call/return transitions: track nesting but keep pattern progress
    # Use push to save pattern state, restore on return
    for i in range(n + 1):
        for c in alphabet.calls:
            vpa.add_call_transition(f"p{i}", c, f"p{i}", f"save_p{i}")
        for r in alphabet.returns:
            for j in range(n + 1):
                vpa.add_return_transition(f"p{i}", r, f"save_p{j}", f"p{j}")

    return vpa


# --- XML/HTML Validation ---

def make_xml_validator(tags: List[str]) -> VPA:
    """Create a VPA that validates well-nested XML-like tags.

    Call symbols: <tag> (open tags)
    Return symbols: </tag> (close tags)
    Each open tag must be closed by its matching close tag.
    """
    calls = frozenset(f"<{t}>" for t in tags)
    returns = frozenset(f"</{t}>" for t in tags)
    alpha = VisibleAlphabet(calls=calls, returns=returns, internals=frozenset({"text"}))

    vpa = VPA(alphabet=alpha)
    vpa.add_state("q", initial=True, accepting=True)

    # Each open tag pushes its identity
    for t in tags:
        vpa.add_call_transition("q", f"<{t}>", "q", f"tag_{t}")

    # Each close tag pops only its matching open tag
    for t in tags:
        vpa.add_return_transition("q", f"</{t}>", f"tag_{t}", "q")
        # Mismatched close -> no transition (reject)

    # Text is internal
    vpa.add_internal_transition("q", "text", "q")

    return vpa


# --- Conversion to/from V094 PDS ---

def vpa_to_pds(vpa: VPA):
    """Convert a VPA to a pushdown system (V094 format).

    Each VPA configuration (state, stack) maps directly to a PDS configuration.
    """
    # Import V094
    import sys
    sys.path.insert(0, "Z:/AgentZero/A2/work/V094_pushdown_systems")
    from pushdown_systems import PushdownSystem, StackOp as PdsStackOp, Configuration

    pds = PushdownSystem()

    # Map VPA transitions to PDS rules
    for (src, sym), targets in vpa.call_transitions.items():
        for (tgt, push_sym) in targets:
            # Call: read sym, push push_sym (PUSH rule: replace top with push_sym then sym_marker)
            # Actually in PDS: we model the input by consuming stack symbols
            # VPA is input-driven; PDS doesn't read input. We encode input into transitions.
            # Simpler: create one PDS state per (VPA_state, input_position) -- but that's bounded.
            # Better: encode as labeled PDS rules and track input externally.
            pds.add_rule(src, f"read_{sym}", tgt, PdsStackOp.PUSH, (push_sym, f"read_{sym}"))

    for (src, sym, gamma), targets in vpa.return_transitions.items():
        for tgt in targets:
            pds.add_rule(src, gamma, tgt, PdsStackOp.POP)

    for (src, sym), targets in vpa.internal_transitions.items():
        for tgt in targets:
            pds.add_rule(src, f"read_{sym}", tgt, PdsStackOp.SWAP, (f"read_{sym}",))

    initial_configs = []
    for q0 in vpa.initial_states:
        initial_configs.append(Configuration(state=q0, stack=(STACK_BOTTOM,)))

    return pds, initial_configs


# --- Minimization ---

def minimize_vpa(vpa: VPA) -> VPA:
    """Minimize a deterministic VPA using partition refinement.

    Analogous to DFA minimization but accounts for stack-dependent transitions.
    First determinizes if needed.
    """
    if not vpa.is_deterministic():
        dvpa = determinize_vpa(vpa)
    else:
        dvpa = vpa

    alpha = dvpa.alphabet

    # Start with two partitions: accepting and non-accepting
    accepting = frozenset(dvpa.accepting_states)
    non_accepting = frozenset(dvpa.states - dvpa.accepting_states)

    partitions = set()
    if accepting:
        partitions.add(accepting)
    if non_accepting:
        partitions.add(non_accepting)

    # Refine until stable
    changed = True
    while changed:
        changed = False
        new_partitions = set()
        for part in partitions:
            # Try to split this partition
            split = _try_split(dvpa, part, partitions, alpha)
            if len(split) > 1:
                changed = True
                new_partitions.update(split)
            else:
                new_partitions.add(part)
        partitions = new_partitions

    # Build minimized VPA
    state_to_part = {}
    for part in partitions:
        rep = min(sorted(part))
        for s in part:
            state_to_part[s] = rep

    mini = VPA(alphabet=alpha)
    for part in partitions:
        rep = min(sorted(part))
        mini.add_state(rep,
                       initial=any(s in dvpa.initial_states for s in part),
                       accepting=any(s in dvpa.accepting_states for s in part))

    # Add transitions using representatives
    seen_call = set()
    seen_ret = set()
    seen_int = set()

    for (src, sym), targets in dvpa.call_transitions.items():
        for (tgt, push) in targets:
            key = (state_to_part[src], sym, state_to_part[tgt], push)
            if key not in seen_call:
                seen_call.add(key)
                mini.add_call_transition(state_to_part[src], sym, state_to_part[tgt], push)

    for (src, sym, gamma), targets in dvpa.return_transitions.items():
        for tgt in targets:
            key = (state_to_part[src], sym, gamma, state_to_part[tgt])
            if key not in seen_ret:
                seen_ret.add(key)
                mini.add_return_transition(state_to_part[src], sym, gamma, state_to_part[tgt])

    for (src, sym), targets in dvpa.internal_transitions.items():
        for tgt in targets:
            key = (state_to_part[src], sym, state_to_part[tgt])
            if key not in seen_int:
                seen_int.add(key)
                mini.add_internal_transition(state_to_part[src], sym, state_to_part[tgt])

    return mini


def _try_split(vpa, partition, all_partitions, alpha):
    """Try to split a partition based on transition targets."""
    if len(partition) <= 1:
        return {partition}

    states = sorted(partition)
    ref = states[0]

    def get_target_part(state, sym, kind, gamma=None):
        """Get which partition a transition leads to."""
        if kind == "call":
            key = (state, sym)
            if key in vpa.call_transitions:
                targets = vpa.call_transitions[key]
                if targets:
                    tgt, push = next(iter(targets))
                    for part in all_partitions:
                        if tgt in part:
                            return (part, push)
            return None
        elif kind == "return":
            key = (state, sym, gamma)
            if key in vpa.return_transitions:
                targets = vpa.return_transitions[key]
                if targets:
                    tgt = next(iter(targets))
                    for part in all_partitions:
                        if tgt in part:
                            return part
            return None
        elif kind == "internal":
            key = (state, sym)
            if key in vpa.internal_transitions:
                targets = vpa.internal_transitions[key]
                if targets:
                    tgt = next(iter(targets))
                    for part in all_partitions:
                        if tgt in part:
                            return part
            return None

    # Try each symbol to find a split
    for c in alpha.calls:
        ref_target = get_target_part(ref, c, "call")
        same = set()
        diff = set()
        for s in states:
            if get_target_part(s, c, "call") == ref_target:
                same.add(s)
            else:
                diff.add(s)
        if diff:
            return {frozenset(same), frozenset(diff)}

    for i_sym in alpha.internals:
        ref_target = get_target_part(ref, i_sym, "internal")
        same = set()
        diff = set()
        for s in states:
            if get_target_part(s, i_sym, "internal") == ref_target:
                same.add(s)
            else:
                diff.add(s)
        if diff:
            return {frozenset(same), frozenset(diff)}

    # For returns, we need to try each stack symbol
    all_gammas = set()
    for (src, sym, gamma) in vpa.return_transitions:
        if src in partition:
            all_gammas.add(gamma)
    for r in alpha.returns:
        for gamma in all_gammas:
            ref_target = get_target_part(ref, r, "return", gamma)
            same = set()
            diff = set()
            for s in states:
                if get_target_part(s, r, "return", gamma) == ref_target:
                    same.add(s)
                else:
                    diff.add(s)
            if diff:
                return {frozenset(same), frozenset(diff)}

    return {partition}


# --- Statistics ---

def vpa_summary(vpa: VPA) -> dict:
    """Get summary statistics for a VPA."""
    n_call = sum(len(t) for t in vpa.call_transitions.values())
    n_ret = sum(len(t) for t in vpa.return_transitions.values())
    n_int = sum(len(t) for t in vpa.internal_transitions.values())
    return {
        "states": len(vpa.states),
        "initial_states": len(vpa.initial_states),
        "accepting_states": len(vpa.accepting_states),
        "call_transitions": n_call,
        "return_transitions": n_ret,
        "internal_transitions": n_int,
        "total_transitions": n_call + n_ret + n_int,
        "deterministic": vpa.is_deterministic(),
        "alphabet": {
            "calls": len(vpa.alphabet.calls),
            "returns": len(vpa.alphabet.returns),
            "internals": len(vpa.alphabet.internals),
        }
    }


# --- High-Level Verification APIs ---

def verify_well_nestedness(word: List[str], alphabet: VisibleAlphabet) -> dict:
    """Verify that a word is well-nested (balanced calls/returns)."""
    vpa = make_balanced_parens_vpa(alphabet)
    accepted = run_vpa(vpa, word, empty_stack=True)
    nw = NestedWord.from_word(word, alphabet)
    return {
        "well_nested": accepted,
        "nesting_depth": nw.depth,
        "nesting_pairs": nw.nesting
    }


def verify_xml_structure(tag_sequence: List[str], tags: List[str]) -> dict:
    """Verify XML-like tag structure."""
    vpa = make_xml_validator(tags)
    accepted = run_vpa(vpa, tag_sequence, empty_stack=True)
    return {
        "valid": accepted,
        "tags": tags,
        "sequence_length": len(tag_sequence)
    }


def verify_bounded_recursion(word: List[str], alphabet: VisibleAlphabet,
                              max_depth: int) -> dict:
    """Verify that a call/return sequence respects a recursion depth bound."""
    balanced_vpa = make_balanced_parens_vpa(alphabet)
    bounded_vpa = make_bounded_depth_vpa(alphabet, max_depth)

    is_balanced = run_vpa(balanced_vpa, word, empty_stack=True)
    is_bounded = run_vpa(bounded_vpa, word, empty_stack=True)

    return {
        "well_nested": is_balanced,
        "within_bound": is_bounded,
        "max_depth": max_depth,
        "nesting_depth": NestedWord.from_word(word, alphabet).depth
    }


def compare_vpa(vpa1: VPA, vpa2: VPA, test_words: Optional[List[List[str]]] = None) -> dict:
    """Compare two VPAs: equivalence check + test word evaluation."""
    equiv = check_equivalence(vpa1, vpa2)
    result = {
        "equivalent": equiv["equivalent"],
        "counterexample": equiv.get("counterexample"),
        "vpa1_summary": vpa_summary(vpa1),
        "vpa2_summary": vpa_summary(vpa2),
    }
    if test_words:
        comparisons = []
        for w in test_words:
            r1 = run_vpa(vpa1, w)
            r2 = run_vpa(vpa2, w)
            comparisons.append({
                "word": w,
                "vpa1_accepts": r1,
                "vpa2_accepts": r2,
                "agree": r1 == r2
            })
        result["test_comparisons"] = comparisons
    return result
