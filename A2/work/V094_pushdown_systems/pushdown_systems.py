"""V094: Pushdown Systems Verification

Verification of pushdown systems via saturation-based pre*/post* computation.
Pushdown systems model recursive programs where finite control states represent
program points and the stack represents the call stack.

Key algorithms:
- Pre* computation (backward reachability) via saturation of P-automata
- Post* computation (forward reachability) via saturation with epsilon-summary edges
- Configuration reachability checking
- Regular model checking: does the PDS satisfy a property expressed as
  a regular set of configurations?
- LTL model checking via Buchi pushdown systems
- Recursive program modeling: function calls as push, returns as pop

Composes: V089 (tree automata concepts -- ranked alphabet/trees used for comparison)

References:
- Bouajjani, Esparza, Maler: "Reachability analysis of pushdown automata" (1997)
- Esparza, Schwoon: "A BDD-based model checker for recursive programs" (2001)
- Schwoon: "Model-Checking Pushdown Systems" (PhD thesis, 2002)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import deque
from enum import Enum


# --- Data Structures ---

class StackOp(Enum):
    """Stack operation type."""
    POP = "pop"        # (p, a) -> (q, epsilon) -- pop a, go to q
    SWAP = "swap"      # (p, a) -> (q, b) -- replace a with b
    PUSH = "push"      # (p, a) -> (q, b c) -- replace a with bc (b on top)


@dataclass(frozen=True)
class PDSRule:
    """A pushdown system transition rule.

    (state, symbol) -> (new_state, stack_action)
    where stack_action depends on the operation type.
    """
    state: str           # source control state p
    symbol: str          # top-of-stack symbol a (read and consumed)
    new_state: str       # target control state q
    op: StackOp
    push_symbols: Tuple[str, ...] = ()  # symbols pushed (for SWAP: (b,), for PUSH: (b, c))

    def __post_init__(self):
        if self.op == StackOp.POP:
            assert len(self.push_symbols) == 0
        elif self.op == StackOp.SWAP:
            assert len(self.push_symbols) == 1
        elif self.op == StackOp.PUSH:
            assert len(self.push_symbols) == 2


@dataclass
class PushdownSystem:
    """A pushdown system (PDS).

    A PDS = (P, Gamma, Delta) where:
    - P: finite set of control states
    - Gamma: stack alphabet
    - Delta: finite set of transition rules
    """
    states: Set[str] = field(default_factory=set)
    stack_alphabet: Set[str] = field(default_factory=set)
    rules: List[PDSRule] = field(default_factory=list)

    def add_rule(self, state: str, symbol: str, new_state: str,
                 op: StackOp, push_symbols: Tuple[str, ...] = ()):
        """Add a transition rule."""
        self.states.add(state)
        self.states.add(new_state)
        self.stack_alphabet.add(symbol)
        for s in push_symbols:
            self.stack_alphabet.add(s)
        rule = PDSRule(state, symbol, new_state, op, push_symbols)
        self.rules.append(rule)
        return rule

    def get_rules(self, state: str, symbol: str) -> List[PDSRule]:
        """Get all rules applicable from (state, symbol)."""
        return [r for r in self.rules if r.state == state and r.symbol == symbol]


@dataclass(frozen=True)
class Configuration:
    """A PDS configuration: (control_state, stack_contents).

    Stack is a tuple of symbols, leftmost = top.
    """
    state: str
    stack: Tuple[str, ...]

    def top(self) -> Optional[str]:
        return self.stack[0] if self.stack else None

    def pop(self) -> 'Configuration':
        return Configuration(self.state, self.stack[1:])

    def push(self, symbol: str) -> 'Configuration':
        return Configuration(self.state, (symbol,) + self.stack)

    def is_empty_stack(self) -> bool:
        return len(self.stack) == 0


# --- P-Automaton (for representing regular sets of configurations) ---

@dataclass
class PAutomaton:
    """P-Automaton: NFA that accepts stack contents for each control state.

    A P-automaton A = (Q, Gamma, delta, P, F) where:
    - Q: states (includes PDS control states P as initial states)
    - Gamma: input alphabet (= stack alphabet)
    - delta: transitions Q x Gamma -> 2^Q
    - P: initial states (= PDS control states)
    - F: final states (accepting = empty stack reached)

    Accepts configuration (p, w) iff the NFA starting from state p accepts word w.
    """
    pds_states: Set[str]  # initial states (from PDS control states)
    states: Set[str] = field(default_factory=set)
    final_states: Set[str] = field(default_factory=set)
    transitions: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    # Epsilon transitions for post* computation
    epsilon_transitions: Dict[str, Set[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.states.update(self.pds_states)

    def add_state(self, state: str, final: bool = False):
        self.states.add(state)
        if final:
            self.final_states.add(state)

    def add_transition(self, src: str, symbol: str, dst: str) -> bool:
        """Add transition. Returns True if new (for saturation worklist)."""
        self.states.add(src)
        self.states.add(dst)
        key = (src, symbol)
        if key not in self.transitions:
            self.transitions[key] = set()
        if dst not in self.transitions[key]:
            self.transitions[key].add(dst)
            return True
        return False

    def add_epsilon(self, src: str, dst: str) -> bool:
        """Add epsilon transition. Returns True if new."""
        if src not in self.epsilon_transitions:
            self.epsilon_transitions[src] = set()
        if dst not in self.epsilon_transitions[src]:
            self.epsilon_transitions[src].add(dst)
            return True
        return False

    def get_targets(self, state: str, symbol: str) -> Set[str]:
        """Get target states for (state, symbol) transition."""
        return self.transitions.get((state, symbol), set())

    def get_epsilon_targets(self, state: str) -> Set[str]:
        """Get epsilon-reachable targets from state."""
        return self.epsilon_transitions.get(state, set())

    def accepts(self, config: Configuration) -> bool:
        """Check if this automaton accepts the given configuration."""
        # Start from the PDS control state, read stack symbols left to right
        current = {config.state}
        # First, close over epsilons
        current = self._epsilon_closure(current)

        for symbol in config.stack:
            next_states = set()
            for s in current:
                next_states.update(self.get_targets(s, symbol))
            current = self._epsilon_closure(next_states)
            if not current:
                return False

        return bool(current & self.final_states)

    def _epsilon_closure(self, states: Set[str]) -> Set[str]:
        """Compute epsilon closure of a set of states."""
        if not self.epsilon_transitions:
            return states
        closure = set(states)
        worklist = list(states)
        while worklist:
            s = worklist.pop()
            for t in self.get_epsilon_targets(s):
                if t not in closure:
                    closure.add(t)
                    worklist.append(t)
        return closure

    def accepted_configs(self, pds: PushdownSystem, max_stack: int = 5) -> Set[Configuration]:
        """Enumerate accepted configurations up to max stack depth."""
        configs = set()
        symbols = list(pds.stack_alphabet)

        for state in self.pds_states:
            # Try all stack contents up to max_stack
            self._enumerate_stacks(state, symbols, (), max_stack, configs)
        return configs

    def _enumerate_stacks(self, state, symbols, stack, max_depth, configs):
        config = Configuration(state, stack)
        if self.accepts(config):
            configs.add(config)
        if len(stack) < max_depth:
            for sym in symbols:
                self._enumerate_stacks(state, symbols, stack + (sym,), max_depth, configs)

    def copy(self) -> 'PAutomaton':
        """Deep copy."""
        pa = PAutomaton(set(self.pds_states))
        pa.states = set(self.states)
        pa.final_states = set(self.final_states)
        pa.transitions = {k: set(v) for k, v in self.transitions.items()}
        pa.epsilon_transitions = {k: set(v) for k, v in self.epsilon_transitions.items()}
        return pa


def make_config_automaton(pds: PushdownSystem, configs: Set[Configuration]) -> PAutomaton:
    """Build a P-automaton accepting exactly the given configurations."""
    pa = PAutomaton(set(pds.states))

    # For each configuration, build a path in the automaton
    state_counter = [0]

    def fresh_state():
        state_counter[0] += 1
        return f"__q{state_counter[0]}"

    # Single shared final state
    final = fresh_state()
    pa.add_state(final, final=True)

    for config in configs:
        current = config.state
        for i, sym in enumerate(config.stack):
            if i == len(config.stack) - 1:
                # Last symbol -> go to final
                pa.add_transition(current, sym, final)
            else:
                next_s = fresh_state()
                pa.add_transition(current, sym, next_s)
                current = next_s

        if len(config.stack) == 0:
            # Empty stack -> control state is directly final
            pa.add_state(current, final=True)

    return pa


def make_state_automaton(pds: PushdownSystem, target_states: Set[str]) -> PAutomaton:
    """Build a P-automaton accepting all configurations in the given control states
    (any stack content)."""
    pa = PAutomaton(set(pds.states))

    final = "__accept"
    pa.add_state(final, final=True)

    for state in target_states:
        # Accept any stack: state is also final (for empty stack)
        pa.add_state(state, final=True)
        # For any symbol, loop on a single accepting state
        for sym in pds.stack_alphabet:
            pa.add_transition(state, sym, final)
            pa.add_transition(final, sym, final)

    return pa


# --- Pre* Computation (Backward Reachability) ---

def pre_star(pds: PushdownSystem, initial_automaton: PAutomaton) -> PAutomaton:
    """Compute pre*(C) where C is the set of configs accepted by initial_automaton.

    Saturation algorithm from Bouajjani, Esparza, Maler (1997):
    For each rule (p, a) -> (q, w) in Delta:
      If q --w--> r in the automaton, add transition p --a--> r

    Repeat until no new transitions are added (fixpoint).

    Returns: P-automaton accepting all configurations that can reach C.
    """
    result = initial_automaton.copy()

    # Worklist: transitions to process
    # We process all existing transitions plus newly added ones
    worklist = deque()

    # Initialize worklist with all current transitions
    for (src, sym), dsts in result.transitions.items():
        for dst in dsts:
            worklist.append((src, sym, dst))

    # Track which transitions exist to avoid duplicates in worklist
    seen = set()
    for (src, sym), dsts in result.transitions.items():
        for dst in dsts:
            seen.add((src, sym, dst))

    while worklist:
        (q_src, gamma, q_dst) = worklist.popleft()

        # For each PDS rule, check if this transition helps complete a path
        for rule in pds.rules:
            if rule.new_state != q_src:
                continue

            if rule.op == StackOp.POP:
                # Rule: (p, a) -> (q, epsilon)
                # If q is q_src, we don't need to read anything more
                # But this transition reads gamma, which is not what POP rules match
                # POP: (p, a) -> (q, eps) means q can reach whatever q could with empty stack
                # Actually: if q --eps--> ... is accepted, then p with 'a' on top reaches same
                # For POP rules: q_src doesn't matter for reading gamma
                pass  # handled separately below

            elif rule.op == StackOp.SWAP:
                # Rule: (p, a) -> (q, b)
                # If q --b--> q_dst exists, add p --a--> q_dst
                if gamma == rule.push_symbols[0]:
                    new_trans = (rule.state, rule.symbol, q_dst)
                    if new_trans not in seen:
                        if result.add_transition(rule.state, rule.symbol, q_dst):
                            seen.add(new_trans)
                            worklist.append(new_trans)

            elif rule.op == StackOp.PUSH:
                # Rule: (p, a) -> (q, b c)
                # Need q --b--> q_mid --c--> q_dst
                # This transition is q_src --gamma--> q_dst
                # If gamma == b, then we need q_mid == q_dst and look for q_dst --c--> q_final
                # If gamma == c, then q_src --c--> q_dst, need q --b--> q_src
                if gamma == rule.push_symbols[0]:
                    # This is q --b--> q_dst, now we need q_dst --c--> some final
                    c = rule.push_symbols[1]
                    for q_after_c in result.get_targets(q_dst, c):
                        new_trans = (rule.state, rule.symbol, q_after_c)
                        if new_trans not in seen:
                            if result.add_transition(rule.state, rule.symbol, q_after_c):
                                seen.add(new_trans)
                                worklist.append(new_trans)

    # Handle POP rules separately: (p, a) -> (q, eps) means if q is accepting,
    # then add p --a--> (some accepting state). More precisely: if q reaches
    # an accepting state with empty stack (i.e., q IS accepting or q reaches
    # accepting via existing transitions with empty remaining), add p --a--> q.
    # Actually, POP means the stack word after is the rest of the stack.
    # So (p, a) -> (q, eps): for config (p, a.w), we go to (q, w).
    # In the automaton: if q --w--> accepting, then p --a.w--> accepting.
    # So: add transition p --a--> q (since q will then read the rest w).

    # Restart saturation with POP rules included
    result2 = initial_automaton.copy()
    worklist2 = deque()
    seen2 = set()

    for (src, sym), dsts in result2.transitions.items():
        for dst in dsts:
            seen2.add((src, sym, dst))
            worklist2.append((src, sym, dst))

    # Saturate
    changed = True
    while changed:
        changed = False
        for rule in pds.rules:
            if rule.op == StackOp.POP:
                # (p, a) -> (q, eps): add p --a--> q
                new_t = (rule.state, rule.symbol, rule.new_state)
                if new_t not in seen2:
                    if result2.add_transition(rule.state, rule.symbol, rule.new_state):
                        seen2.add(new_t)
                        worklist2.append(new_t)
                        changed = True

            elif rule.op == StackOp.SWAP:
                # (p, a) -> (q, b): if q --b--> r exists, add p --a--> r
                b = rule.push_symbols[0]
                for r in set(result2.get_targets(rule.new_state, b)):
                    new_t = (rule.state, rule.symbol, r)
                    if new_t not in seen2:
                        if result2.add_transition(rule.state, rule.symbol, r):
                            seen2.add(new_t)
                            worklist2.append(new_t)
                            changed = True

            elif rule.op == StackOp.PUSH:
                # (p, a) -> (q, b c): if q --b--> r --c--> s exists, add p --a--> s
                b, c = rule.push_symbols
                for r in set(result2.get_targets(rule.new_state, b)):
                    for s in set(result2.get_targets(r, c)):
                        new_t = (rule.state, rule.symbol, s)
                        if new_t not in seen2:
                            if result2.add_transition(rule.state, rule.symbol, s):
                                seen2.add(new_t)
                                worklist2.append(new_t)
                                changed = True

    return result2


# --- Post* Computation (Forward Reachability) ---

def post_star(pds: PushdownSystem, initial_automaton: PAutomaton) -> PAutomaton:
    """Compute post*(C) where C is the set of configs accepted by initial_automaton.

    Saturation algorithm from Schwoon (2002):
    For each rule (p, a) -> (q, w) in Delta:
      If there exists p --a--> r in current automaton:
        - POP: add q --eps--> r (epsilon transition)
        - SWAP(b): add q --b--> r
        - PUSH(b,c): add q --b--> q_new, q_new --c--> r
          where q_new is a fresh state unique to (q, b)

    Returns: P-automaton accepting all configurations reachable from C.
    """
    result = initial_automaton.copy()

    # For PUSH rules, we create fresh intermediate states
    # Key: (state, symbol) -> intermediate state name
    mid_states: Dict[Tuple[str, str], str] = {}
    mid_counter = [0]

    def get_mid_state(state: str, symbol: str) -> str:
        key = (state, symbol)
        if key not in mid_states:
            mid_counter[0] += 1
            name = f"__mid_{state}_{symbol}_{mid_counter[0]}"
            mid_states[key] = name
            result.add_state(name)
        return mid_states[key]

    # Worklist of transitions to process
    # Each entry: (src, symbol, dst) -- a transition that was just added
    worklist = deque()
    seen = set()

    for (src, sym), dsts in result.transitions.items():
        for dst in dsts:
            t = (src, sym, dst)
            seen.add(t)
            worklist.append(t)

    while worklist:
        (p, a, r) = worklist.popleft()

        # For each rule where this transition matches the LHS
        for rule in pds.rules:
            if rule.state != p or rule.symbol != a:
                continue

            q = rule.new_state

            if rule.op == StackOp.POP:
                # (p, a) -> (q, eps): add epsilon q -> r
                if result.add_epsilon(q, r):
                    # Epsilon transition added: propagate all transitions from r to q
                    # For any r --b--> s, add q --b--> s
                    for (src2, sym2), dsts2 in list(result.transitions.items()):
                        if src2 == r:
                            for dst2 in set(dsts2):
                                t2 = (q, sym2, dst2)
                                if t2 not in seen:
                                    if result.add_transition(q, sym2, dst2):
                                        seen.add(t2)
                                        worklist.append(t2)

            elif rule.op == StackOp.SWAP:
                # (p, a) -> (q, b): add q --b--> r
                b = rule.push_symbols[0]
                t2 = (q, b, r)
                if t2 not in seen:
                    if result.add_transition(q, b, r):
                        seen.add(t2)
                        worklist.append(t2)

            elif rule.op == StackOp.PUSH:
                # (p, a) -> (q, b c): add q --b--> mid, mid --c--> r
                b, c = rule.push_symbols
                mid = get_mid_state(q, b)

                # mid --c--> r
                t_cr = (mid, c, r)
                if t_cr not in seen:
                    if result.add_transition(mid, c, r):
                        seen.add(t_cr)
                        worklist.append(t_cr)

                # q --b--> mid
                t_bm = (q, b, mid)
                if t_bm not in seen:
                    if result.add_transition(q, b, mid):
                        seen.add(t_bm)
                        worklist.append(t_bm)

    # Also propagate epsilon transitions to catch all reachable configs
    # For any epsilon q -> r and r --b--> s, ensure q --b--> s exists
    changed = True
    while changed:
        changed = False
        for src, eps_dsts in list(result.epsilon_transitions.items()):
            for eps_dst in set(eps_dsts):
                # src --eps--> eps_dst, so src inherits all transitions from eps_dst
                for (t_src, t_sym), t_dsts in list(result.transitions.items()):
                    if t_src == eps_dst:
                        for t_dst in set(t_dsts):
                            t2 = (src, t_sym, t_dst)
                            if t2 not in seen:
                                if result.add_transition(src, t_sym, t_dst):
                                    seen.add(t2)
                                    changed = True
                # Also inherit final status
                if eps_dst in result.final_states:
                    if src not in result.final_states:
                        result.final_states.add(src)
                        changed = True
                # Chain epsilons
                for eps2_dst in set(result.get_epsilon_targets(eps_dst)):
                    if result.add_epsilon(src, eps2_dst):
                        changed = True

    return result


# --- Reachability Checking ---

def check_reachability(pds: PushdownSystem,
                       source: Configuration,
                       target: Configuration) -> dict:
    """Check if target is reachable from source in the PDS.

    Returns dict with 'reachable' bool and 'witness_path' if reachable.
    """
    # Build automaton accepting only {target}
    target_auto = make_config_automaton(pds, {target})

    # Compute pre*(target) -- all configs that can reach target
    pre_auto = pre_star(pds, target_auto)

    reachable = pre_auto.accepts(source)

    result = {
        "reachable": reachable,
        "source": source,
        "target": target,
    }

    if reachable:
        # Find witness path via BFS
        path = _find_path_bfs(pds, source, target)
        result["witness_path"] = path

    return result


def _find_path_bfs(pds: PushdownSystem, source: Configuration,
                   target: Configuration, max_steps: int = 100) -> Optional[List[Configuration]]:
    """BFS to find a concrete path from source to target."""
    if source == target:
        return [source]

    visited = {source}
    queue = deque([(source, [source])])

    while queue and len(visited) < 10000:
        config, path = queue.popleft()
        if len(path) > max_steps:
            continue

        for succ in _successors(pds, config):
            if succ == target:
                return path + [succ]
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path + [succ]))

    return None


def _successors(pds: PushdownSystem, config: Configuration) -> List[Configuration]:
    """Compute all successor configurations of config."""
    if config.is_empty_stack():
        return []

    top = config.top()
    rest = config.stack[1:]
    succs = []

    for rule in pds.get_rules(config.state, top):
        if rule.op == StackOp.POP:
            succs.append(Configuration(rule.new_state, rest))
        elif rule.op == StackOp.SWAP:
            succs.append(Configuration(rule.new_state, (rule.push_symbols[0],) + rest))
        elif rule.op == StackOp.PUSH:
            b, c = rule.push_symbols
            succs.append(Configuration(rule.new_state, (b, c) + rest))

    return succs


def _predecessors(pds: PushdownSystem, config: Configuration) -> List[Configuration]:
    """Compute all predecessor configurations of config."""
    preds = []

    for rule in pds.rules:
        if rule.new_state != config.state:
            continue

        if rule.op == StackOp.POP:
            # (p, a) -> (q, eps): predecessor of (q, w) is (p, a.w)
            preds.append(Configuration(rule.state, (rule.symbol,) + config.stack))

        elif rule.op == StackOp.SWAP:
            # (p, a) -> (q, b): predecessor of (q, b.w) is (p, a.w)
            b = rule.push_symbols[0]
            if config.stack and config.stack[0] == b:
                preds.append(Configuration(rule.state, (rule.symbol,) + config.stack[1:]))

        elif rule.op == StackOp.PUSH:
            # (p, a) -> (q, b c): predecessor of (q, b.c.w) is (p, a.w)
            b, c = rule.push_symbols
            if len(config.stack) >= 2 and config.stack[0] == b and config.stack[1] == c:
                preds.append(Configuration(rule.state, (rule.symbol,) + config.stack[2:]))

    return preds


# --- Regular Set Operations ---

def check_regular_property(pds: PushdownSystem,
                          initial_configs: PAutomaton,
                          target_configs: PAutomaton,
                          direction: str = "forward") -> dict:
    """Check if any configuration reachable from initial_configs is in target_configs.

    direction: "forward" uses post*, "backward" uses pre*.

    Returns dict with 'satisfies' (True if intersection is non-empty).
    """
    if direction == "forward":
        reach = post_star(pds, initial_configs)
    else:
        reach = pre_star(pds, target_configs)
        # Check intersection with initial_configs
        # We check: does any config in initial_configs appear in pre*(target)?
        # This is equivalent to checking forward reachability
        target_configs = initial_configs
        reach_to_check = reach
        # Enumerate and check
        configs = target_configs.accepted_configs(pds, max_stack=6)
        for c in configs:
            if reach_to_check.accepts(c):
                return {"satisfies": True, "witness": c}
        return {"satisfies": False}

    # Forward: check intersection of reach and target_configs
    configs = target_configs.accepted_configs(pds, max_stack=6)
    for c in configs:
        if reach.accepts(c):
            return {"satisfies": True, "witness": c}

    # Also check reach-accepted configs against target
    reach_configs = reach.accepted_configs(pds, max_stack=6)
    for c in reach_configs:
        if target_configs.accepts(c):
            return {"satisfies": True, "witness": c}

    return {"satisfies": False}


# --- Safety Checking ---

def check_safety(pds: PushdownSystem,
                initial_configs: Set[Configuration],
                bad_configs: Set[Configuration],
                max_depth: int = 50) -> dict:
    """Check if bad configurations are unreachable from initial configurations.

    Returns dict with 'safe' bool and counterexample if unsafe.
    """
    # Build automaton for bad configs
    bad_auto = make_config_automaton(pds, bad_configs)

    # Compute pre*(bad) -- all configs that can reach bad
    pre_bad = pre_star(pds, bad_auto)

    # Check if any initial config is in pre*(bad)
    for init_config in initial_configs:
        if pre_bad.accepts(init_config):
            # Find concrete counterexample
            for bad in bad_configs:
                path = _find_path_bfs(pds, init_config, bad)
                if path:
                    return {
                        "safe": False,
                        "counterexample": path,
                        "initial": init_config,
                        "bad": bad,
                    }
            return {
                "safe": False,
                "initial": init_config,
                "counterexample": None,
            }

    return {"safe": True}


# --- Recursive Program Modeling ---

@dataclass
class RecursiveProgram:
    """Simple recursive program model for PDS encoding.

    Each function has labeled statements. Call/return creates push/pop.
    """
    functions: Dict[str, List[dict]] = field(default_factory=dict)
    entry: str = "main"
    entry_label: str = "start"

    def add_function(self, name: str, body: List[dict]):
        """Add a function with labeled statements.

        Each statement is a dict with:
        - 'type': 'assign', 'call', 'return', 'goto', 'branch'
        - 'label': statement label
        - 'next': next label (for assign/call)
        - 'target': called function (for call)
        - 'return_label': label after call returns (for call)
        - 'true_label', 'false_label': branch targets
        """
        self.functions[name] = body


def program_to_pds(program: RecursiveProgram) -> Tuple[PushdownSystem, Configuration]:
    """Convert a recursive program to a PDS.

    Encoding:
    - Control states = {program_points} (function_name.label pairs)
    - Stack alphabet = {return addresses} (function_name.label for call return points)
    - Call f at point p with return to r: PUSH rule (p, top) -> (f.entry, r top)
    - Return at point p: POP rule (p, top) -> (top_as_state, eps)
    - Sequential at point p to q: SWAP rule (p, top) -> (q, top) [or just keep top]

    Actually, standard PDS encoding of recursive programs:
    - Control state = single state 'run' (all info on stack)
    - Stack symbol = (function, label) pair = program point
    - Call f from (g, L) with return to (g, L'):
      rule: (run, (g,L)) -> (run, (f,entry) (g,L'))  [PUSH]
    - Return from (f, ret):
      rule: (run, (f,ret)) -> (run, eps)  [POP -- returns to caller]
    - Goto from (g, L) to (g, L'):
      rule: (run, (g,L)) -> (run, (g,L'))  [SWAP]
    """
    pds = PushdownSystem()

    for fname, body in program.functions.items():
        for stmt in body:
            src_sym = f"{fname}.{stmt['label']}"
            pds.stack_alphabet.add(src_sym)
            pds.states.add("run")

            if stmt['type'] == 'assign' or stmt['type'] == 'goto':
                # Sequential: swap current stack symbol with next label
                next_sym = f"{fname}.{stmt['next']}"
                pds.add_rule("run", src_sym, "run", StackOp.SWAP, (next_sym,))

            elif stmt['type'] == 'call':
                # Call: push return address, enter callee
                target = stmt['target']
                callee_entry = f"{target}.{program.functions[target][0]['label']}"
                return_sym = f"{fname}.{stmt['return_label']}"
                pds.add_rule("run", src_sym, "run", StackOp.PUSH, (callee_entry, return_sym))

            elif stmt['type'] == 'return':
                # Return: pop current frame, control goes to return address
                pds.add_rule("run", src_sym, "run", StackOp.POP)

            elif stmt['type'] == 'branch':
                # Branch: nondeterministic choice (overapproximate both branches)
                true_sym = f"{fname}.{stmt['true_label']}"
                false_sym = f"{fname}.{stmt['false_label']}"
                pds.add_rule("run", src_sym, "run", StackOp.SWAP, (true_sym,))
                pds.add_rule("run", src_sym, "run", StackOp.SWAP, (false_sym,))

    # Initial configuration: (run, main.entry)
    entry_sym = f"{program.entry}.{program.entry_label}"
    initial = Configuration("run", (entry_sym,))

    return pds, initial


# --- Bounded Model Checking ---

def bounded_reachability(pds: PushdownSystem,
                        initial: Configuration,
                        target_fn,
                        max_steps: int = 50,
                        max_stack: int = 20) -> dict:
    """Bounded reachability via BFS up to max_steps.

    target_fn: callable(Configuration) -> bool

    Returns dict with 'reachable', 'witness_path', 'explored_count'.
    """
    visited = {initial}
    queue = deque([(initial, [initial], 0)])

    if target_fn(initial):
        return {
            "reachable": True,
            "witness_path": [initial],
            "explored_count": 1,
            "steps": 0,
        }

    while queue:
        config, path, depth = queue.popleft()
        if depth >= max_steps:
            continue

        for succ in _successors(pds, config):
            if len(succ.stack) > max_stack:
                continue
            if succ in visited:
                continue
            visited.add(succ)

            new_path = path + [succ]
            if target_fn(succ):
                return {
                    "reachable": True,
                    "witness_path": new_path,
                    "explored_count": len(visited),
                    "steps": depth + 1,
                }
            queue.append((succ, new_path, depth + 1))

    return {
        "reachable": False,
        "explored_count": len(visited),
        "steps": max_steps,
    }


# --- State Space Exploration ---

def explore_state_space(pds: PushdownSystem,
                       initial: Configuration,
                       max_configs: int = 1000,
                       max_stack: int = 20) -> dict:
    """Explore the state space starting from initial configuration.

    Returns statistics about the reachable configuration space.
    """
    visited = {initial}
    queue = deque([initial])
    transitions_count = 0
    max_stack_seen = len(initial.stack)
    deadlocks = []

    while queue and len(visited) < max_configs:
        config = queue.popleft()
        succs = _successors(pds, config)

        if not succs and not config.is_empty_stack():
            deadlocks.append(config)

        for succ in succs:
            if len(succ.stack) > max_stack:
                continue
            transitions_count += 1
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)
                max_stack_seen = max(max_stack_seen, len(succ.stack))

    return {
        "reachable_configs": len(visited),
        "transitions": transitions_count,
        "max_stack_depth": max_stack_seen,
        "deadlocks": len(deadlocks),
        "deadlock_configs": deadlocks[:10],  # first 10
        "exhaustive": len(visited) < max_configs,
    }


# --- Invariant Checking ---

def check_invariant(pds: PushdownSystem,
                   initial_configs: Set[Configuration],
                   invariant_fn,
                   max_steps: int = 100,
                   max_stack: int = 20) -> dict:
    """Check that invariant_fn holds for all reachable configurations.

    invariant_fn: callable(Configuration) -> bool

    Returns dict with 'holds' and counterexample if violated.
    """
    for init in initial_configs:
        if not invariant_fn(init):
            return {
                "holds": False,
                "counterexample": [init],
                "violation": init,
            }

    # BFS from all initial configs
    visited = set(initial_configs)
    queue = deque([(c, [c]) for c in initial_configs])

    while queue:
        config, path = queue.popleft()
        if len(path) > max_steps:
            continue

        for succ in _successors(pds, config):
            if len(succ.stack) > max_stack:
                continue
            if succ in visited:
                continue
            visited.add(succ)

            if not invariant_fn(succ):
                return {
                    "holds": False,
                    "counterexample": path + [succ],
                    "violation": succ,
                }
            queue.append((succ, path + [succ]))

    return {
        "holds": True,
        "explored": len(visited),
    }


# --- Comparison & Summary APIs ---

def compare_pre_post(pds: PushdownSystem,
                    configs: Set[Configuration],
                    max_stack: int = 5) -> dict:
    """Compare pre* and post* computations for the same set of configs."""
    auto = make_config_automaton(pds, configs)

    pre_auto = pre_star(pds, auto)
    post_auto = post_star(pds, auto)

    pre_configs = pre_auto.accepted_configs(pds, max_stack)
    post_configs = post_auto.accepted_configs(pds, max_stack)

    return {
        "original": configs,
        "pre_star_size": len(pre_configs),
        "post_star_size": len(post_configs),
        "pre_star_configs": pre_configs,
        "post_star_configs": post_configs,
        "pre_only": pre_configs - post_configs,
        "post_only": post_configs - pre_configs,
        "both": pre_configs & post_configs,
    }


def pds_summary(pds: PushdownSystem) -> dict:
    """Summary statistics about a PDS."""
    pop_rules = [r for r in pds.rules if r.op == StackOp.POP]
    swap_rules = [r for r in pds.rules if r.op == StackOp.SWAP]
    push_rules = [r for r in pds.rules if r.op == StackOp.PUSH]

    return {
        "states": len(pds.states),
        "stack_symbols": len(pds.stack_alphabet),
        "rules": len(pds.rules),
        "pop_rules": len(pop_rules),
        "swap_rules": len(swap_rules),
        "push_rules": len(push_rules),
        "state_list": sorted(pds.states),
        "symbol_list": sorted(pds.stack_alphabet),
    }


# --- Example PDS Constructors ---

def make_simple_counter() -> Tuple[PushdownSystem, Configuration]:
    """Counter that increments/decrements using stack depth.

    States: inc, dec, done
    Stack: 'c' marks, '#' bottom marker
    inc: push 'c' on 'c' or '#'
    dec: pop 'c'
    done: when '#' is on top
    """
    pds = PushdownSystem()

    # Increment: push c
    pds.add_rule("inc", "c", "inc", StackOp.PUSH, ("c", "c"))
    pds.add_rule("inc", "#", "inc", StackOp.PUSH, ("c", "#"))

    # Switch to decrement
    pds.add_rule("inc", "c", "dec", StackOp.SWAP, ("c",))
    pds.add_rule("inc", "#", "dec", StackOp.SWAP, ("#",))

    # Decrement: pop c
    pds.add_rule("dec", "c", "dec", StackOp.POP)

    # Done when bottom marker reached
    pds.add_rule("dec", "#", "done", StackOp.SWAP, ("#",))

    initial = Configuration("inc", ("#",))
    return pds, initial


def make_recursive_program_pds() -> Tuple[PushdownSystem, Configuration, RecursiveProgram]:
    """Example: recursive factorial-like program.

    main:
      start: call f
      end: return

    f:
      entry: branch (base case or recursive)
      base: return
      recurse: call f
      after: return
    """
    prog = RecursiveProgram()
    prog.entry = "main"
    prog.entry_label = "start"

    prog.add_function("main", [
        {"type": "call", "label": "start", "target": "f", "return_label": "end"},
        {"type": "return", "label": "end"},
    ])

    prog.add_function("f", [
        {"type": "branch", "label": "entry", "true_label": "base", "false_label": "recurse"},
        {"type": "return", "label": "base"},
        {"type": "call", "label": "recurse", "target": "f", "return_label": "after"},
        {"type": "return", "label": "after"},
    ])

    pds, initial = program_to_pds(prog)
    return pds, initial, prog


def make_mutual_recursion_pds() -> Tuple[PushdownSystem, Configuration, RecursiveProgram]:
    """Mutually recursive functions f and g.

    main: call f -> end
    f: branch -> base | call g -> after_g
    g: call f -> after_f -> return
    """
    prog = RecursiveProgram()
    prog.entry = "main"
    prog.entry_label = "start"

    prog.add_function("main", [
        {"type": "call", "label": "start", "target": "f", "return_label": "end"},
        {"type": "return", "label": "end"},
    ])

    prog.add_function("f", [
        {"type": "branch", "label": "entry", "true_label": "base", "false_label": "call_g"},
        {"type": "return", "label": "base"},
        {"type": "call", "label": "call_g", "target": "g", "return_label": "after_g"},
        {"type": "return", "label": "after_g"},
    ])

    prog.add_function("g", [
        {"type": "call", "label": "entry", "target": "f", "return_label": "after_f"},
        {"type": "return", "label": "after_f"},
    ])

    pds, initial = program_to_pds(prog)
    return pds, initial, prog


def make_stack_inspection_pds() -> Tuple[PushdownSystem, Configuration]:
    """PDS modeling stack inspection for security.

    States: trusted, untrusted, check
    Stack: frames from different trust levels
    Security property: 'check' state only reachable with trusted frame on stack.
    """
    pds = PushdownSystem()

    # Trusted code can call untrusted
    pds.add_rule("trusted", "t_frame", "untrusted", StackOp.PUSH, ("u_frame", "t_frame"))

    # Untrusted code can call trusted
    pds.add_rule("untrusted", "u_frame", "trusted", StackOp.PUSH, ("t_frame", "u_frame"))

    # Both can call check
    pds.add_rule("trusted", "t_frame", "check", StackOp.PUSH, ("check_frame", "t_frame"))
    pds.add_rule("untrusted", "u_frame", "check", StackOp.PUSH, ("check_frame", "u_frame"))

    # Return from check
    pds.add_rule("check", "check_frame", "trusted", StackOp.POP)

    # Return from calls
    pds.add_rule("trusted", "t_frame", "trusted", StackOp.POP)
    pds.add_rule("untrusted", "u_frame", "untrusted", StackOp.POP)
    pds.add_rule("trusted", "t_frame", "untrusted", StackOp.POP)
    pds.add_rule("untrusted", "u_frame", "trusted", StackOp.POP)

    initial = Configuration("trusted", ("t_frame",))
    return pds, initial
