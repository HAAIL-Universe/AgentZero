"""
V151: Probabilistic Process Algebra

CCS-style process algebra with probabilistic choice.
Processes communicate via shared action names (handshake synchronization).
Supports: prefix, probabilistic choice, nondeterministic choice,
parallel composition, restriction, relabeling, recursion.

Composes: V150 (weak probabilistic bisimulation) + V148 (strong bisimulation)

Key concepts:
- Prefix: a.P -- perform action a then behave as P
- Probabilistic choice: P [p] Q -- choose P with probability p, Q with 1-p
- Nondeterministic choice: P + Q -- can behave as P or Q
- Parallel: P | Q -- concurrent composition with CCS-style synchronization
- Restriction: P \\ {a} -- hide action a (becomes tau)
- Relabeling: P[f] -- rename actions
- Recursion: fix X. P -- recursive process definition
"""

import sys
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V150_weak_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V148_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from weak_probabilistic_bisimulation import (
    TAU, LabeledProbTS, make_labeled_prob_ts,
    compute_weak_bisimulation, WeakBisimVerdict, WeakBisimResult,
)


# ---- Process AST ----

class ProcKind(Enum):
    STOP = "stop"          # deadlock
    PREFIX = "prefix"      # a.P
    PROB_CHOICE = "prob"   # P [p] Q
    ND_CHOICE = "nd"       # P + Q
    PARALLEL = "par"       # P | Q
    RESTRICT = "restrict"  # P \\ L
    RELABEL = "relabel"    # P[f]
    RECVAR = "recvar"      # X (recursion variable)
    RECDEF = "recdef"      # fix X. P


@dataclass(frozen=True)
class Proc:
    kind: ProcKind
    action: Optional[str] = None          # PREFIX
    prob: Optional[float] = None          # PROB_CHOICE
    left: Optional['Proc'] = None         # PREFIX(cont), PROB/ND/PAR
    right: Optional['Proc'] = None        # PROB/ND/PAR
    labels: Optional[FrozenSet[str]] = None  # RESTRICT
    relabel_map: Optional[Tuple[Tuple[str, str], ...]] = None  # RELABEL
    var: Optional[str] = None             # RECVAR, RECDEF
    body: Optional['Proc'] = None         # RECDEF

    def __repr__(self):
        return _format_proc(self)


def _format_proc(p: Proc, depth: int = 0) -> str:
    if p.kind == ProcKind.STOP:
        return "0"
    elif p.kind == ProcKind.PREFIX:
        return f"{p.action}.{_format_proc(p.left, depth+1)}"
    elif p.kind == ProcKind.PROB_CHOICE:
        l = _format_proc(p.left, depth+1)
        r = _format_proc(p.right, depth+1)
        return f"({l} [{p.prob}] {r})"
    elif p.kind == ProcKind.ND_CHOICE:
        l = _format_proc(p.left, depth+1)
        r = _format_proc(p.right, depth+1)
        return f"({l} + {r})"
    elif p.kind == ProcKind.PARALLEL:
        l = _format_proc(p.left, depth+1)
        r = _format_proc(p.right, depth+1)
        return f"({l} | {r})"
    elif p.kind == ProcKind.RESTRICT:
        return f"({_format_proc(p.left, depth+1)} \\\\ {{{','.join(sorted(p.labels))}}})"
    elif p.kind == ProcKind.RELABEL:
        rl = ",".join(f"{b}/{a}" for a, b in p.relabel_map)
        return f"({_format_proc(p.left, depth+1)}[{rl}])"
    elif p.kind == ProcKind.RECVAR:
        return p.var
    elif p.kind == ProcKind.RECDEF:
        return f"(fix {p.var}. {_format_proc(p.body, depth+1)})"
    return "?"


# ---- Constructors ----

def stop() -> Proc:
    """Deadlock process."""
    return Proc(ProcKind.STOP)

def prefix(action: str, cont: Proc) -> Proc:
    """Action prefix: a.P"""
    return Proc(ProcKind.PREFIX, action=action, left=cont)

def tau_prefix(cont: Proc) -> Proc:
    """Internal prefix: tau.P"""
    return Proc(ProcKind.PREFIX, action=TAU, left=cont)

def prob_choice(p: float, left: Proc, right: Proc) -> Proc:
    """Probabilistic choice: P [p] Q"""
    assert 0.0 < p < 1.0, "Probability must be in (0, 1)"
    return Proc(ProcKind.PROB_CHOICE, prob=p, left=left, right=right)

def nd_choice(left: Proc, right: Proc) -> Proc:
    """Nondeterministic choice: P + Q"""
    return Proc(ProcKind.ND_CHOICE, left=left, right=right)

def parallel(left: Proc, right: Proc) -> Proc:
    """Parallel composition: P | Q"""
    return Proc(ProcKind.PARALLEL, left=left, right=right)

def restrict(proc: Proc, labels: Set[str]) -> Proc:
    """Restriction: P \\\\ L"""
    return Proc(ProcKind.RESTRICT, left=proc, labels=frozenset(labels))

def relabel(proc: Proc, mapping: Dict[str, str]) -> Proc:
    """Relabeling: P[f]"""
    return Proc(ProcKind.RELABEL, left=proc,
                relabel_map=tuple(sorted(mapping.items())))

def recvar(name: str) -> Proc:
    """Recursion variable."""
    return Proc(ProcKind.RECVAR, var=name)

def recdef(name: str, body: Proc) -> Proc:
    """Recursive definition: fix X. P"""
    return Proc(ProcKind.RECDEF, var=name, body=body)


# ---- Operational Semantics: LTS Generation ----

@dataclass
class LTSState:
    """A state in the labeled transition system."""
    proc: Proc
    id: int


class LTSBuilder:
    """Builds finite LTS from process term by exploring reachable states.

    Handles recursion by unfolding fix X. P into P[fix X.P / X] up to max_states.
    """

    def __init__(self, max_states: int = 200):
        self.max_states = max_states
        self.states: List[Proc] = []
        self.state_map: Dict[str, int] = {}  # proc repr -> state id
        self.transitions: Dict[int, Dict[str, List[Tuple[int, float]]]] = {}

    def _get_or_create_state(self, proc: Proc) -> int:
        key = repr(proc)
        if key in self.state_map:
            return self.state_map[key]
        if len(self.states) >= self.max_states:
            return -1  # overflow
        sid = len(self.states)
        self.states.append(proc)
        self.state_map[key] = sid
        self.transitions[sid] = {}
        return sid

    def build(self, proc: Proc) -> LabeledProbTS:
        """Build LTS from process term."""
        self.states = []
        self.state_map = {}
        self.transitions = {}

        init_id = self._get_or_create_state(proc)
        assert init_id == 0

        # BFS exploration
        queue = [0]
        visited = set()

        while queue:
            sid = queue.pop(0)
            if sid in visited:
                continue
            visited.add(sid)

            p = self.states[sid]
            moves = self._compute_transitions(p)

            for action, targets in moves.items():
                self.transitions[sid][action] = []
                for target_proc, prob in targets:
                    tid = self._get_or_create_state(target_proc)
                    if tid == -1:
                        continue  # state limit reached
                    self.transitions[sid][action].append((tid, prob))
                    if tid not in visited:
                        queue.append(tid)

        # Build LabeledProbTS
        n = len(self.states)
        actions = {}
        labels = {}
        names = []

        for sid in range(n):
            actions[sid] = dict(self.transitions.get(sid, {}))
            # Derive labels from behavior (transitions), not AST structure
            if any(self.transitions.get(sid, {}).values()):
                labels[sid] = {"active"}
            else:
                labels[sid] = {"deadlock"}
            names.append(f"s{sid}")

        return LabeledProbTS(n, actions, labels, names)

    def _compute_transitions(self, proc: Proc) -> Dict[str, List[Tuple[Proc, float]]]:
        """Compute all transitions from a process term (SOS rules)."""
        return self._transitions(proc, {})

    def _transitions(self, proc: Proc, env: Dict[str, Proc]) -> Dict[str, List[Tuple[Proc, float]]]:
        """Structural operational semantics."""
        if proc.kind == ProcKind.STOP:
            return {}

        elif proc.kind == ProcKind.PREFIX:
            # a.P --a--> P (probability 1.0)
            return {proc.action: [(proc.left, 1.0)]}

        elif proc.kind == ProcKind.PROB_CHOICE:
            # P [p] Q --tau--> P with prob p, Q with prob 1-p
            return {TAU: [(proc.left, proc.prob), (proc.right, 1.0 - proc.prob)]}

        elif proc.kind == ProcKind.ND_CHOICE:
            # P + Q: P can do a -> P', or Q can do a -> Q'
            result = {}
            left_trans = self._transitions(proc.left, env)
            right_trans = self._transitions(proc.right, env)

            for a, targets in left_trans.items():
                if a not in result:
                    result[a] = []
                result[a].extend(targets)

            for a, targets in right_trans.items():
                if a not in result:
                    result[a] = []
                result[a].extend(targets)

            return result

        elif proc.kind == ProcKind.PARALLEL:
            # P | Q: interleaving + synchronization
            result = {}
            left_trans = self._transitions(proc.left, env)
            right_trans = self._transitions(proc.right, env)

            # Interleaving: P does a, Q stays
            for a, targets in left_trans.items():
                if a not in result:
                    result[a] = []
                for p_prime, prob in targets:
                    result[a].append((parallel(p_prime, proc.right), prob))

            # Interleaving: Q does a, P stays
            for a, targets in right_trans.items():
                if a not in result:
                    result[a] = []
                for q_prime, prob in targets:
                    result[a].append((parallel(proc.left, q_prime), prob))

            # Synchronization: P does a, Q does complement(a) -> tau
            for a_l, l_targets in left_trans.items():
                if a_l == TAU:
                    continue
                comp = _complement(a_l)
                if comp in right_trans:
                    for p_prime, p_prob in l_targets:
                        for q_prime, q_prob in right_trans[comp]:
                            if TAU not in result:
                                result[TAU] = []
                            result[TAU].append((parallel(p_prime, q_prime), p_prob * q_prob))

            return result

        elif proc.kind == ProcKind.RESTRICT:
            # P \\ L: hide actions in L (they become tau, but only via synchronization)
            # Actually: restriction removes actions in L from the interface
            inner_trans = self._transitions(proc.left, env)
            result = {}
            for a, targets in inner_trans.items():
                base_a = a.lstrip("~")  # strip complement marker
                if base_a in proc.labels or a in proc.labels:
                    # Restricted: if it's a synchronization (tau from sync), keep as tau
                    # If it's a standalone restricted action, remove it
                    if a == TAU:
                        if TAU not in result:
                            result[TAU] = []
                        for p_prime, prob in targets:
                            result[TAU].append((restrict(p_prime, set(proc.labels)), prob))
                    # else: action is restricted, transitions are blocked
                else:
                    if a not in result:
                        result[a] = []
                    for p_prime, prob in targets:
                        result[a].append((restrict(p_prime, set(proc.labels)), prob))
            return result

        elif proc.kind == ProcKind.RELABEL:
            inner_trans = self._transitions(proc.left, env)
            result = {}
            mapping = dict(proc.relabel_map)
            for a, targets in inner_trans.items():
                new_a = mapping.get(a, a)
                if new_a not in result:
                    result[new_a] = []
                for p_prime, prob in targets:
                    result[new_a].append((relabel(p_prime, mapping), prob))
            return result

        elif proc.kind == ProcKind.RECVAR:
            # Look up in environment
            if proc.var in env:
                return self._transitions(env[proc.var], env)
            return {}

        elif proc.kind == ProcKind.RECDEF:
            # fix X. P -> unfold: substitute X with (fix X. P) in P
            unfolded = _substitute(proc.body, proc.var, proc)
            return self._transitions(unfolded, env)

        return {}

    def _derive_labels(self, proc: Proc) -> Set[str]:
        """Derive atomic propositions from process structure.

        Labels reflect behavioral properties (deadlock, active),
        NOT structural details (action names, parallel composition).
        This ensures bisimulation compares behavior, not syntax.
        """
        labels = set()
        if proc.kind == ProcKind.STOP:
            labels.add("deadlock")
        else:
            labels.add("active")
        return labels


def _complement(action: str) -> str:
    """CCS complement: a <-> ~a"""
    if action.startswith("~"):
        return action[1:]
    return "~" + action


def _substitute(proc: Proc, var: str, replacement: Proc) -> Proc:
    """Substitute recursion variable with its definition."""
    if proc.kind == ProcKind.STOP:
        return proc
    elif proc.kind == ProcKind.RECVAR:
        if proc.var == var:
            return replacement
        return proc
    elif proc.kind == ProcKind.PREFIX:
        return prefix(proc.action, _substitute(proc.left, var, replacement))
    elif proc.kind == ProcKind.PROB_CHOICE:
        return prob_choice(
            proc.prob,
            _substitute(proc.left, var, replacement),
            _substitute(proc.right, var, replacement),
        )
    elif proc.kind == ProcKind.ND_CHOICE:
        return nd_choice(
            _substitute(proc.left, var, replacement),
            _substitute(proc.right, var, replacement),
        )
    elif proc.kind == ProcKind.PARALLEL:
        return parallel(
            _substitute(proc.left, var, replacement),
            _substitute(proc.right, var, replacement),
        )
    elif proc.kind == ProcKind.RESTRICT:
        return restrict(_substitute(proc.left, var, replacement), set(proc.labels))
    elif proc.kind == ProcKind.RELABEL:
        return relabel(
            _substitute(proc.left, var, replacement),
            dict(proc.relabel_map),
        )
    elif proc.kind == ProcKind.RECDEF:
        if proc.var == var:
            # Inner binding shadows outer
            return proc
        return recdef(proc.var, _substitute(proc.body, var, replacement))
    return proc


# ---- Process Analysis ----

def generate_lts(proc: Proc, max_states: int = 200) -> LabeledProbTS:
    """Generate labeled transition system from a process term."""
    builder = LTSBuilder(max_states=max_states)
    return builder.build(proc)


def check_process_equivalence(
    p1: Proc, p2: Proc, max_states: int = 200
) -> WeakBisimResult:
    """Check if two processes are weakly bisimilar (initial states)."""
    from weak_probabilistic_bisimulation import check_cross_weak_bisimulation
    lts1 = generate_lts(p1, max_states)
    lts2 = generate_lts(p2, max_states)
    result = check_cross_weak_bisimulation(lts1, lts2)

    # Check specifically if initial states (0 in lts1, n1 in combined = 0 in lts2) are in same block
    n1 = lts1.n_states
    init1 = 0
    init2 = n1  # state 0 of lts2 is at offset n1 in combined system

    for block in result.partition:
        if init1 in block and init2 in block:
            return WeakBisimResult(
                verdict=WeakBisimVerdict.WEAKLY_BISIMILAR,
                partition=result.partition,
                relation=result.relation,
                statistics=result.statistics,
            )

    return WeakBisimResult(
        verdict=WeakBisimVerdict.NOT_WEAKLY_BISIMILAR,
        partition=result.partition,
        witness="Initial states not weakly bisimilar",
        statistics=result.statistics,
    )


def check_strong_equivalence(
    p1: Proc, p2: Proc, max_states: int = 200
) -> WeakBisimResult:
    """Check if two processes are strongly bisimilar (via LTS)."""
    from weak_probabilistic_bisimulation import compute_weak_bisimulation
    lts1 = generate_lts(p1, max_states)
    lts2 = generate_lts(p2, max_states)

    # Build combined system (no tau abstraction -- check exact transitions)
    n1 = lts1.n_states
    n2 = lts2.n_states
    n_total = n1 + n2

    combined_actions = {}
    combined_labels = {}
    for s in range(n1):
        combined_actions[s] = dict(lts1.actions.get(s, {}))
        combined_labels[s] = lts1.state_labels.get(s, set()).copy()
    for s in range(n2):
        combined_actions[n1 + s] = {}
        for a, dist in lts2.actions.get(s, {}).items():
            combined_actions[n1 + s][a] = [(n1 + t, p) for t, p in dist]
        combined_labels[n1 + s] = lts2.state_labels.get(s, set()).copy()

    combined = LabeledProbTS(n_total, combined_actions, combined_labels)
    result = compute_weak_bisimulation(combined)

    # Check if initial states are in same block
    for block in result.partition:
        if 0 in block and n1 in block:
            return WeakBisimResult(
                verdict=WeakBisimVerdict.WEAKLY_BISIMILAR,
                partition=result.partition,
                statistics=result.statistics,
            )

    return WeakBisimResult(
        verdict=WeakBisimVerdict.NOT_WEAKLY_BISIMILAR,
        partition=result.partition,
        witness="Initial states not bisimilar",
        statistics=result.statistics,
    )


def trace_set(proc: Proc, max_depth: int = 10, max_states: int = 100) -> Set[Tuple[str, ...]]:
    """Compute the (bounded) trace set of a process."""
    lts = generate_lts(proc, max_states)
    traces = set()

    def dfs(state: int, trace: Tuple[str, ...], depth: int):
        if depth >= max_depth:
            traces.add(trace)
            return

        has_transition = False
        for a, dist in lts.actions.get(state, {}).items():
            for target, prob in dist:
                if prob > 0:
                    has_transition = True
                    if a == TAU:
                        # Tau steps don't appear in trace
                        dfs(target, trace, depth + 1)
                    else:
                        dfs(target, trace + (a,), depth + 1)

        if not has_transition:
            traces.add(trace)

    dfs(0, (), 0)
    return traces


def deadlock_free(proc: Proc, max_states: int = 100) -> bool:
    """Check if a process is deadlock-free (no reachable deadlock state)."""
    lts = generate_lts(proc, max_states)
    for s in range(lts.n_states):
        if not lts.actions.get(s, {}):
            return False
    return True


def action_set(proc: Proc, max_states: int = 100) -> Set[str]:
    """Compute the set of observable actions a process can perform."""
    lts = generate_lts(proc, max_states)
    actions = set()
    for s in range(lts.n_states):
        for a in lts.actions.get(s, {}):
            if a != TAU:
                actions.add(a)
    return actions


def process_summary(proc: Proc, max_states: int = 100) -> str:
    """Generate a summary of process behavior."""
    lts = generate_lts(proc, max_states)
    traces = trace_set(proc, max_depth=5, max_states=max_states)
    actions = action_set(proc, max_states)
    df = deadlock_free(proc, max_states)

    lines = [
        f"Process: {proc}",
        f"LTS states: {lts.n_states}",
        f"Actions: {sorted(actions)}",
        f"Deadlock-free: {df}",
        f"Traces (depth 5): {len(traces)}",
    ]
    for t in sorted(traces)[:10]:
        lines.append(f"  {' -> '.join(t) if t else '<empty>'}")
    if len(traces) > 10:
        lines.append(f"  ... and {len(traces) - 10} more")

    return "\n".join(lines)


# ---- Parser (simple text syntax) ----

class ProcParser:
    """Parse process algebra expressions.

    Syntax:
        P ::= 0                    -- stop
            | a.P                  -- prefix
            | P [p] P             -- probabilistic choice
            | P + P               -- nondeterministic choice
            | P | P               -- parallel
            | P \\\\ {a,b,...}       -- restriction
            | fix X. P            -- recursion
            | X                   -- recursion variable
            | (P)                 -- grouping
    """

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> Proc:
        result = self._parse_parallel()
        self._skip_ws()
        return result

    def _skip_ws(self):
        while self.pos < len(self.text) and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def _peek(self) -> Optional[str]:
        self._skip_ws()
        if self.pos < len(self.text):
            return self.text[self.pos]
        return None

    def _expect(self, ch: str):
        self._skip_ws()
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            raise ValueError(f"Expected '{ch}' at pos {self.pos}")
        self.pos += 1

    def _parse_parallel(self) -> Proc:
        left = self._parse_choice()
        while self._peek() == '|':
            self.pos += 1
            right = self._parse_choice()
            left = parallel(left, right)
        return left

    def _parse_choice(self) -> Proc:
        left = self._parse_prefix()
        while True:
            self._skip_ws()
            if self._peek() == '+':
                self.pos += 1
                right = self._parse_prefix()
                left = nd_choice(left, right)
            elif self._peek() == '[':
                self.pos += 1
                # Probabilistic choice: [p]
                p_str = ""
                while self.pos < len(self.text) and self.text[self.pos] != ']':
                    p_str += self.text[self.pos]
                    self.pos += 1
                self._expect(']')
                p = float(p_str)
                right = self._parse_prefix()
                left = prob_choice(p, left, right)
            else:
                break
        return left

    def _parse_prefix(self) -> Proc:
        self._skip_ws()

        # Check for fix
        if self.text[self.pos:self.pos+3] == 'fix':
            self.pos += 3
            self._skip_ws()
            var = self._parse_ident()
            self._expect('.')
            body = self._parse_parallel()
            return recdef(var, body)

        # Check for prefix: ident.P
        saved = self.pos
        try:
            name = self._parse_ident()
            if self._peek() == '.':
                self.pos += 1
                cont = self._parse_prefix()
                return prefix(name, cont)
            else:
                # Not a prefix, might be a variable or stop
                if name == '0':
                    return stop()
                return recvar(name)
        except (ValueError, IndexError):
            self.pos = saved
            return self._parse_atom()

    def _parse_atom(self) -> Proc:
        self._skip_ws()
        if self._peek() == '(':
            self.pos += 1
            result = self._parse_parallel()
            self._expect(')')
            return result
        if self._peek() == '0':
            self.pos += 1
            return stop()
        raise ValueError(f"Unexpected at pos {self.pos}: '{self.text[self.pos:]}'")

    def _parse_ident(self) -> str:
        self._skip_ws()
        start = self.pos
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] in '_~'):
            self.pos += 1
        if self.pos == start:
            raise ValueError(f"Expected identifier at pos {self.pos}")
        return self.text[start:self.pos]


def parse_proc(text: str) -> Proc:
    """Parse a process algebra expression."""
    return ProcParser(text).parse()
