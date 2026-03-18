"""
V208: Strategy Logic (SL) -- First-Class Strategy Quantification

Strategy Logic extends ATL* with explicit strategy variables and quantification.
Instead of asking "can coalition A enforce phi?", SL asks "does there exist a
strategy x for agent a and a strategy y for agent b such that phi holds?"

This enables reasoning about:
- Nash equilibria: "there exist strategies s.t. no agent benefits from deviating"
- Dominant strategies: "there exists a strategy that wins against all opponent strategies"
- Strategy sharing: multiple agents can use the SAME strategy variable
- Strategy composition: combine strategies for different sub-goals

Key concepts:
- SL formula: exists x. [a bind x] phi  (existential strategy quantification)
- SL formula: forall x. [a bind x] phi  (universal strategy quantification)
- [a bind x] means agent a plays according to strategy x
- A strategy is a function from histories (sequences of states) to actions
- Model checking SL is non-elementary in general, but decidable
- We implement the memoryless fragment for tractability

Self-contained module with lightweight CGS (string-based states/actions).
Can compose with V205 via adapter if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, FrozenSet, List, Optional, Set, Tuple, Any
)
from itertools import product as cartesian_product


# ============================================================
# Lightweight Concurrent Game Structure (string-based)
# ============================================================

@dataclass
class CGS:
    """Concurrent Game Structure with string-based states/actions.

    At each state, every agent simultaneously chooses an action.
    The joint action determines the successor state.
    """
    states: Set[str] = field(default_factory=set)
    agents: List[str] = field(default_factory=list)
    # agent -> state -> list of available actions
    _actions: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # (state, action_tuple) -> next_state
    _transition: Dict[Tuple[str, Tuple[str, ...]], str] = field(default_factory=dict)
    # state -> set of atomic propositions
    labeling: Dict[str, Set[str]] = field(default_factory=dict)

    def actions(self, state: str, agent: str) -> List[str]:
        """Get available actions for agent at state."""
        return self._actions.get(agent, {}).get(state, [])

    def transition(self, state: str, joint_action: Tuple[str, ...]) -> Optional[str]:
        """Get next state for a joint action."""
        return self._transition.get((state, joint_action))

    def successors(self, state: str) -> Set[str]:
        """Get all possible successor states."""
        result = set()
        action_lists = [self.actions(state, a) or [""] for a in self.agents]
        for combo in cartesian_product(*action_lists):
            ns = self.transition(state, combo)
            if ns is not None:
                result.add(ns)
        return result


class CGSBuilder:
    """Fluent builder for CGS."""

    def __init__(self, agents: List[str]):
        self._cgs = CGS(agents=list(agents))
        for a in agents:
            self._cgs._actions[a] = {}

    def add_state(self, name: str, labels: Optional[Set[str]] = None) -> CGSBuilder:
        self._cgs.states.add(name)
        self._cgs.labeling[name] = labels or set()
        return self

    def add_actions(self, state: str, agent: str, actions: List[str]) -> CGSBuilder:
        self._cgs._actions[agent][state] = actions
        return self

    def add_transition(self, state: str, joint_action: Tuple[str, ...],
                       next_state: str) -> CGSBuilder:
        self._cgs._transition[(state, joint_action)] = next_state
        return self

    def build(self) -> CGS:
        return self._cgs


# ============================================================
# Strategy Logic Formula AST
# ============================================================

class SLOp(Enum):
    """Strategy Logic formula operators."""
    ATOM = auto()
    TRUE = auto()
    FALSE = auto()
    NOT = auto()
    AND = auto()
    OR = auto()
    IMPLIES = auto()

    EXISTS_STRATEGY = auto()    # exists x. phi
    FORALL_STRATEGY = auto()    # forall x. phi
    BIND = auto()               # [a bind x] phi

    NEXT = auto()       # X phi
    GLOBALLY = auto()   # G phi
    FINALLY = auto()    # F phi
    UNTIL = auto()      # phi U psi


@dataclass(frozen=True)
class SL:
    """Strategy Logic formula."""
    op: SLOp
    name: str = ""
    left: Optional[SL] = None
    right: Optional[SL] = None
    strategy_var: str = ""
    agent: str = ""

    def __repr__(self):
        if self.op == SLOp.ATOM:
            return self.name
        if self.op == SLOp.TRUE:
            return "true"
        if self.op == SLOp.FALSE:
            return "false"
        if self.op == SLOp.NOT:
            return f"!{self.left}"
        if self.op == SLOp.AND:
            return f"({self.left} & {self.right})"
        if self.op == SLOp.OR:
            return f"({self.left} | {self.right})"
        if self.op == SLOp.IMPLIES:
            return f"({self.left} -> {self.right})"
        if self.op == SLOp.EXISTS_STRATEGY:
            return f"(Ex {self.strategy_var}. {self.left})"
        if self.op == SLOp.FORALL_STRATEGY:
            return f"(Ax {self.strategy_var}. {self.left})"
        if self.op == SLOp.BIND:
            return f"[{self.agent} bind {self.strategy_var}] {self.left}"
        if self.op == SLOp.NEXT:
            return f"X {self.left}"
        if self.op == SLOp.GLOBALLY:
            return f"G {self.left}"
        if self.op == SLOp.FINALLY:
            return f"F {self.left}"
        if self.op == SLOp.UNTIL:
            return f"({self.left} U {self.right})"
        return f"SL({self.op})"


# ============================================================
# Formula Constructors
# ============================================================

def sl_atom(name: str) -> SL:
    return SL(SLOp.ATOM, name=name)

def sl_true() -> SL:
    return SL(SLOp.TRUE)

def sl_false() -> SL:
    return SL(SLOp.FALSE)

def sl_not(phi: SL) -> SL:
    return SL(SLOp.NOT, left=phi)

def sl_and(phi: SL, psi: SL) -> SL:
    return SL(SLOp.AND, left=phi, right=psi)

def sl_or(phi: SL, psi: SL) -> SL:
    return SL(SLOp.OR, left=phi, right=psi)

def sl_implies(phi: SL, psi: SL) -> SL:
    return SL(SLOp.IMPLIES, left=phi, right=psi)

def exists_strategy(var: str, phi: SL) -> SL:
    return SL(SLOp.EXISTS_STRATEGY, strategy_var=var, left=phi)

def forall_strategy(var: str, phi: SL) -> SL:
    return SL(SLOp.FORALL_STRATEGY, strategy_var=var, left=phi)

def bind(agent: str, var: str, phi: SL) -> SL:
    return SL(SLOp.BIND, agent=agent, strategy_var=var, left=phi)

def sl_next(phi: SL) -> SL:
    return SL(SLOp.NEXT, left=phi)

def sl_globally(phi: SL) -> SL:
    return SL(SLOp.GLOBALLY, left=phi)

def sl_finally(phi: SL) -> SL:
    return SL(SLOp.FINALLY, left=phi)

def sl_until(phi: SL, psi: SL) -> SL:
    return SL(SLOp.UNTIL, left=phi, right=psi)


# ============================================================
# Formula Analysis
# ============================================================

def free_strategy_vars(phi: SL) -> Set[str]:
    """Return free (unquantified) strategy variables in phi."""
    if phi.op in (SLOp.ATOM, SLOp.TRUE, SLOp.FALSE):
        return set()
    if phi.op == SLOp.NOT:
        return free_strategy_vars(phi.left)
    if phi.op in (SLOp.AND, SLOp.OR, SLOp.IMPLIES, SLOp.UNTIL):
        return free_strategy_vars(phi.left) | free_strategy_vars(phi.right)
    if phi.op in (SLOp.NEXT, SLOp.GLOBALLY, SLOp.FINALLY):
        return free_strategy_vars(phi.left)
    if phi.op in (SLOp.EXISTS_STRATEGY, SLOp.FORALL_STRATEGY):
        return free_strategy_vars(phi.left) - {phi.strategy_var}
    if phi.op == SLOp.BIND:
        return free_strategy_vars(phi.left) | {phi.strategy_var}
    return set()


def bound_agents(phi: SL) -> Set[str]:
    """Return agents appearing in binding operators."""
    if phi.op in (SLOp.ATOM, SLOp.TRUE, SLOp.FALSE):
        return set()
    if phi.op == SLOp.NOT:
        return bound_agents(phi.left)
    if phi.op in (SLOp.AND, SLOp.OR, SLOp.IMPLIES, SLOp.UNTIL):
        return bound_agents(phi.left) | bound_agents(phi.right)
    if phi.op in (SLOp.NEXT, SLOp.GLOBALLY, SLOp.FINALLY):
        return bound_agents(phi.left)
    if phi.op in (SLOp.EXISTS_STRATEGY, SLOp.FORALL_STRATEGY):
        return bound_agents(phi.left)
    if phi.op == SLOp.BIND:
        return {phi.agent} | bound_agents(phi.left)
    return set()


def is_sentence(phi: SL) -> bool:
    """Check if phi is a sentence (no free strategy variables)."""
    return len(free_strategy_vars(phi)) == 0


def is_atl_star_fragment(phi: SL) -> bool:
    """Check if phi is in the ATL* fragment of SL.

    ATL* = exists x1...xk. [a1 bind x1]...[ak bind xk] psi
    where each variable is used by exactly one agent and all are existential.
    """
    current = phi
    quant_vars = []
    while current.op == SLOp.EXISTS_STRATEGY:
        quant_vars.append(current.strategy_var)
        current = current.left

    bindings = {}
    while current.op == SLOp.BIND:
        if current.strategy_var not in quant_vars:
            return False
        if current.strategy_var in bindings.values():
            return False
        bindings[current.agent] = current.strategy_var
        current = current.left

    if set(bindings.values()) != set(quant_vars):
        return False

    return _is_temporal_only(current)


def _is_temporal_only(phi: SL) -> bool:
    """Check formula contains only propositional and temporal operators."""
    if phi.op in (SLOp.ATOM, SLOp.TRUE, SLOp.FALSE):
        return True
    if phi.op == SLOp.NOT:
        return _is_temporal_only(phi.left)
    if phi.op in (SLOp.AND, SLOp.OR, SLOp.IMPLIES, SLOp.UNTIL):
        return _is_temporal_only(phi.left) and _is_temporal_only(phi.right)
    if phi.op in (SLOp.NEXT, SLOp.GLOBALLY, SLOp.FINALLY):
        return _is_temporal_only(phi.left)
    return False


def sl_subformulas(phi: SL) -> Set[SL]:
    """Return all sub-formulas of phi."""
    result = {phi}
    if phi.left:
        result |= sl_subformulas(phi.left)
    if phi.right:
        result |= sl_subformulas(phi.right)
    return result


# ============================================================
# Strategies
# ============================================================

@dataclass
class MemorylessStrategy:
    """Memoryless (positional) strategy: state -> action."""
    agent: str
    choices: Dict[str, str]

    def choose(self, state: str, _history: List[str] = None) -> str:
        return self.choices.get(state, "")


@dataclass
class BoundedMemoryStrategy:
    """Bounded-memory strategy with k memory states."""
    agent: str
    memory_size: int
    initial_memory: int = 0
    # (game_state, memory_state) -> (action, next_memory)
    transition: Dict[Tuple[str, int], Tuple[str, int]] = field(default_factory=dict)

    def choose_with_memory(self, state: str, memory: int) -> Tuple[str, int]:
        key = (state, memory)
        if key in self.transition:
            return self.transition[key]
        return ("", memory)


@dataclass
class HistoryStrategy:
    """History-dependent strategy (finite lookup table)."""
    agent: str
    table: Dict[Tuple[str, ...], str] = field(default_factory=dict)
    default_action: str = ""

    def choose(self, state: str, history: List[str] = None) -> str:
        key = tuple(history or []) + (state,)
        return self.table.get(key, self.default_action)


Strategy = MemorylessStrategy | BoundedMemoryStrategy | HistoryStrategy


# ============================================================
# Strategy Profile
# ============================================================

@dataclass
class StrategyProfile:
    """Assignment of strategies to agents (and strategy variables)."""
    strategies: Dict[str, Strategy] = field(default_factory=dict)

    def assign(self, agent: str, strategy: Strategy) -> StrategyProfile:
        new_strats = dict(self.strategies)
        new_strats[agent] = strategy
        return StrategyProfile(new_strats)

    def get(self, agent: str) -> Optional[Strategy]:
        return self.strategies.get(agent)

    def is_complete(self, agents: Set[str]) -> bool:
        return all(a in self.strategies for a in agents)


# ============================================================
# Outcome Computation
# ============================================================

def compute_outcome(
    cgs: CGS,
    profile: StrategyProfile,
    initial_state: str,
    max_steps: int = 100
) -> List[str]:
    """Compute outcome path under a strategy profile."""
    path = [initial_state]
    state = initial_state
    memory_states: Dict[str, int] = {}

    for agent, strat in profile.strategies.items():
        if isinstance(strat, BoundedMemoryStrategy):
            memory_states[agent] = strat.initial_memory

    for _ in range(max_steps):
        actions = {}
        for agent in cgs.agents:
            strat = profile.get(agent)
            if strat is None:
                avail = cgs.actions(state, agent)
                actions[agent] = avail[0] if avail else ""
            elif isinstance(strat, MemorylessStrategy):
                actions[agent] = strat.choose(state)
            elif isinstance(strat, BoundedMemoryStrategy):
                mem = memory_states.get(agent, strat.initial_memory)
                action, new_mem = strat.choose_with_memory(state, mem)
                actions[agent] = action
                memory_states[agent] = new_mem
            elif isinstance(strat, HistoryStrategy):
                actions[agent] = strat.choose(state, path[:-1])
            else:
                actions[agent] = ""

        action_tuple = tuple(actions.get(a, "") for a in cgs.agents)
        next_state = cgs.transition(state, action_tuple)
        if next_state is None:
            break
        path.append(next_state)
        state = next_state

        if len(path) > 2 and path[-1] == path[-2]:
            break

    return path


# ============================================================
# Path Checking
# ============================================================

def check_path(cgs: CGS, path: List[str], phi: SL) -> bool:
    """Check if a path satisfies a temporal SL formula."""
    return _check_path_at(cgs, path, 0, phi)


def _check_path_at(cgs: CGS, path: List[str], pos: int, phi: SL) -> bool:
    if pos >= len(path):
        if phi.op in (SLOp.TRUE, SLOp.GLOBALLY):
            return True
        if phi.op in (SLOp.FALSE, SLOp.FINALLY):
            return False
        if phi.op == SLOp.ATOM:
            return False
        if phi.op == SLOp.NOT:
            return not _check_path_at(cgs, path, pos, phi.left)
        return False

    state = path[pos]

    if phi.op == SLOp.TRUE:
        return True
    if phi.op == SLOp.FALSE:
        return False
    if phi.op == SLOp.ATOM:
        return phi.name in cgs.labeling.get(state, set())
    if phi.op == SLOp.NOT:
        return not _check_path_at(cgs, path, pos, phi.left)
    if phi.op == SLOp.AND:
        return (_check_path_at(cgs, path, pos, phi.left) and
                _check_path_at(cgs, path, pos, phi.right))
    if phi.op == SLOp.OR:
        return (_check_path_at(cgs, path, pos, phi.left) or
                _check_path_at(cgs, path, pos, phi.right))
    if phi.op == SLOp.IMPLIES:
        return (not _check_path_at(cgs, path, pos, phi.left) or
                _check_path_at(cgs, path, pos, phi.right))
    if phi.op == SLOp.NEXT:
        return _check_path_at(cgs, path, pos + 1, phi.left)
    if phi.op == SLOp.GLOBALLY:
        return all(_check_path_at(cgs, path, i, phi.left)
                   for i in range(pos, len(path)))
    if phi.op == SLOp.FINALLY:
        return any(_check_path_at(cgs, path, i, phi.left)
                   for i in range(pos, len(path)))
    if phi.op == SLOp.UNTIL:
        for i in range(pos, len(path)):
            if _check_path_at(cgs, path, i, phi.right):
                return True
            if not _check_path_at(cgs, path, i, phi.left):
                return False
        return False
    return False


# ============================================================
# Memoryless Strategy Enumeration
# ============================================================

def _enumerate_memoryless_strategies(
    cgs: CGS, agent: str
) -> List[MemorylessStrategy]:
    """Enumerate all memoryless strategies for an agent."""
    states = sorted(cgs.states)
    action_lists = []
    for s in states:
        avail = cgs.actions(s, agent)
        if not avail:
            avail = [""]
        action_lists.append(avail)

    strategies = []
    for combo in cartesian_product(*action_lists):
        choices = {s: a for s, a in zip(states, combo)}
        strategies.append(MemorylessStrategy(agent=agent, choices=choices))
    return strategies


# ============================================================
# SL Model Checking (Memoryless Fragment)
# ============================================================

def check_sl(
    cgs: CGS,
    phi: SL,
    state: str,
    max_steps: int = 50
) -> bool:
    """Model check an SL formula on a CGS at a given state.

    Supports the memoryless fragment: all strategies are positional.
    """
    return _check_sl(cgs, phi, state, StrategyProfile(), max_steps)


def _check_sl(
    cgs: CGS,
    phi: SL,
    state: str,
    profile: StrategyProfile,
    max_steps: int
) -> bool:
    if phi.op == SLOp.TRUE:
        return True
    if phi.op == SLOp.FALSE:
        return False
    if phi.op == SLOp.ATOM:
        return phi.name in cgs.labeling.get(state, set())

    if phi.op == SLOp.NOT:
        return not _check_sl(cgs, phi.left, state, profile, max_steps)
    if phi.op == SLOp.AND:
        return (_check_sl(cgs, phi.left, state, profile, max_steps) and
                _check_sl(cgs, phi.right, state, profile, max_steps))
    if phi.op == SLOp.OR:
        return (_check_sl(cgs, phi.left, state, profile, max_steps) or
                _check_sl(cgs, phi.right, state, profile, max_steps))
    if phi.op == SLOp.IMPLIES:
        return (not _check_sl(cgs, phi.left, state, profile, max_steps) or
                _check_sl(cgs, phi.right, state, profile, max_steps))

    if phi.op == SLOp.EXISTS_STRATEGY:
        var = phi.strategy_var
        binding_agent = _find_binding_agent(phi.left, var)
        if binding_agent is None:
            return _check_sl(cgs, phi.left, state, profile, max_steps)
        strategies = _enumerate_memoryless_strategies(cgs, binding_agent)
        return any(
            _check_sl(cgs, phi.left, state,
                      _bind_var(profile, var, strat), max_steps)
            for strat in strategies
        )

    if phi.op == SLOp.FORALL_STRATEGY:
        var = phi.strategy_var
        binding_agent = _find_binding_agent(phi.left, var)
        if binding_agent is None:
            return _check_sl(cgs, phi.left, state, profile, max_steps)
        strategies = _enumerate_memoryless_strategies(cgs, binding_agent)
        return all(
            _check_sl(cgs, phi.left, state,
                      _bind_var(profile, var, strat), max_steps)
            for strat in strategies
        )

    if phi.op == SLOp.BIND:
        agent = phi.agent
        var = phi.strategy_var
        strat = _lookup_var(profile, var)
        if strat is None:
            return False
        new_profile = profile.assign(agent, strat)
        return _check_sl(cgs, phi.left, state, new_profile, max_steps)

    if phi.op in (SLOp.NEXT, SLOp.GLOBALLY, SLOp.FINALLY, SLOp.UNTIL):
        path = compute_outcome(cgs, profile, state, max_steps)
        return check_path(cgs, path, phi)

    return False


def _find_binding_agent(phi: SL, var: str) -> Optional[str]:
    """Find which agent is bound to a given strategy variable."""
    if phi.op == SLOp.BIND and phi.strategy_var == var:
        return phi.agent
    if phi.left:
        result = _find_binding_agent(phi.left, var)
        if result:
            return result
    if phi.right:
        result = _find_binding_agent(phi.right, var)
        if result:
            return result
    return None


def _bind_var(profile: StrategyProfile, var: str, strat: Strategy) -> StrategyProfile:
    """Bind a strategy variable to a strategy."""
    new_strats = dict(profile.strategies)
    new_strats[f"__var_{var}"] = strat
    return StrategyProfile(new_strats)


def _lookup_var(profile: StrategyProfile, var: str) -> Optional[Strategy]:
    """Look up a strategy variable."""
    return profile.strategies.get(f"__var_{var}")


# ============================================================
# Nash Equilibrium
# ============================================================

def check_nash_equilibrium(
    cgs: CGS,
    profile: StrategyProfile,
    objectives: Dict[str, SL],
    state: str,
    max_steps: int = 50
) -> Dict[str, Any]:
    """Check if a strategy profile is a Nash equilibrium.

    No agent can improve their objective by unilaterally deviating.
    """
    outcome = compute_outcome(cgs, profile, state, max_steps)
    current_payoffs = {}
    for agent, obj in objectives.items():
        current_payoffs[agent] = check_path(cgs, outcome, obj)

    deviations = {}
    details = {}

    for agent, obj in objectives.items():
        agent_strategies = _enumerate_memoryless_strategies(cgs, agent)
        can_deviate = False
        deviation_strategy = None

        for alt_strat in agent_strategies:
            alt_profile = profile.assign(agent, alt_strat)
            alt_outcome = compute_outcome(cgs, alt_profile, state, max_steps)
            alt_satisfied = check_path(cgs, alt_outcome, obj)

            if alt_satisfied and not current_payoffs[agent]:
                can_deviate = True
                deviation_strategy = alt_strat
                break

        details[agent] = {
            "current_satisfied": current_payoffs[agent],
            "can_deviate": can_deviate,
        }
        if can_deviate:
            deviations[agent] = deviation_strategy

    return {
        "is_nash": len(deviations) == 0,
        "deviations": deviations,
        "details": details,
    }


def find_nash_equilibria(
    cgs: CGS,
    objectives: Dict[str, SL],
    state: str,
    max_steps: int = 50,
    max_results: int = 10
) -> List[StrategyProfile]:
    """Find Nash equilibria by exhaustive search."""
    agents = list(objectives.keys())
    agent_strategies = {a: _enumerate_memoryless_strategies(cgs, a) for a in agents}

    equilibria = []

    def search(idx: int, current: StrategyProfile):
        if len(equilibria) >= max_results:
            return
        if idx == len(agents):
            result = check_nash_equilibrium(cgs, current, objectives, state, max_steps)
            if result["is_nash"]:
                equilibria.append(StrategyProfile(dict(current.strategies)))
            return
        agent = agents[idx]
        for strat in agent_strategies[agent]:
            search(idx + 1, current.assign(agent, strat))

    search(0, StrategyProfile())
    return equilibria


# ============================================================
# Dominant Strategy
# ============================================================

def find_dominant_strategy(
    cgs: CGS,
    agent: str,
    objective: SL,
    opponents: List[str],
    state: str,
    max_steps: int = 50
) -> Optional[MemorylessStrategy]:
    """Find a dominant strategy: satisfies objective against ALL opponent strategies."""
    agent_strategies = _enumerate_memoryless_strategies(cgs, agent)
    opponent_strat_lists = {o: _enumerate_memoryless_strategies(cgs, o) for o in opponents}

    for strat in agent_strategies:
        is_dominant = True
        for combo in _opponent_combos(opponents, opponent_strat_lists):
            profile = StrategyProfile({agent: strat})
            for opp, opp_strat in combo.items():
                profile = profile.assign(opp, opp_strat)

            outcome = compute_outcome(cgs, profile, state, max_steps)
            if not check_path(cgs, outcome, objective):
                is_dominant = False
                break

        if is_dominant:
            return strat

    return None


def _opponent_combos(
    opponents: List[str],
    strat_lists: Dict[str, List[MemorylessStrategy]]
) -> List[Dict[str, MemorylessStrategy]]:
    if not opponents:
        return [{}]
    lists = [strat_lists[o] for o in opponents]
    combos = []
    for combo in cartesian_product(*lists):
        combos.append({opp: strat for opp, strat in zip(opponents, combo)})
    return combos


# ============================================================
# Strategy Sharing (SL-only)
# ============================================================

def check_with_shared_strategy(
    cgs: CGS,
    agents: List[str],
    objective: SL,
    state: str,
    max_steps: int = 50
) -> Optional[MemorylessStrategy]:
    """Check if agents can share a single strategy to achieve objective.

    Uniquely SL -- ATL* cannot express strategy sharing.
    """
    strategies = _enumerate_memoryless_strategies(cgs, agents[0])

    for strat in strategies:
        profile = StrategyProfile()
        for agent in agents:
            shared = MemorylessStrategy(agent=agent, choices=dict(strat.choices))
            profile = profile.assign(agent, shared)

        outcome = compute_outcome(cgs, profile, state, max_steps)
        if check_path(cgs, outcome, objective):
            return strat

    return None


# ============================================================
# Expressiveness Comparison
# ============================================================

def compare_sl_vs_atl_fragment(
    cgs: CGS,
    sl_formula: SL,
    state: str,
    max_steps: int = 50
) -> Dict[str, Any]:
    """Compare SL checking with ATL* fragment analysis."""
    sl_result = check_sl(cgs, sl_formula, state, max_steps)
    is_atl_frag = is_atl_star_fragment(sl_formula)

    return {
        "sl_result": sl_result,
        "is_atl_star_fragment": is_atl_frag,
        "sl_only": not is_atl_frag,
    }


# ============================================================
# Example Games
# ============================================================

def make_simple_game() -> CGS:
    """2-agent game: agent1 picks a/b, agent2 picks c/d.
    (a,c)->win1, (a,d)->draw, (b,c)->draw, (b,d)->win2. Absorbing terminals.
    """
    b = CGSBuilder(["agent1", "agent2"])
    b.add_state("s0", {"start"})
    b.add_state("s1", {"win1"})
    b.add_state("s2", {"win2"})
    b.add_state("s3", {"draw"})

    for s in ["s0", "s1", "s2", "s3"]:
        if s == "s0":
            b.add_actions(s, "agent1", ["a", "b"])
            b.add_actions(s, "agent2", ["c", "d"])
        else:
            b.add_actions(s, "agent1", ["stay"])
            b.add_actions(s, "agent2", ["stay"])

    b.add_transition("s0", ("a", "c"), "s1")
    b.add_transition("s0", ("a", "d"), "s3")
    b.add_transition("s0", ("b", "c"), "s3")
    b.add_transition("s0", ("b", "d"), "s2")
    for s in ["s1", "s2", "s3"]:
        b.add_transition(s, ("stay", "stay"), s)
    return b.build()


def make_coordination_game() -> CGS:
    """Coordination: alice/bob pick L/R. Match -> win, mismatch -> lose."""
    b = CGSBuilder(["alice", "bob"])
    b.add_state("s0", {"choosing"})
    b.add_state("s1", {"coordinated", "win"})
    b.add_state("s2", {"mismatched"})

    for agent in ["alice", "bob"]:
        b.add_actions("s0", agent, ["L", "R"])
        b.add_actions("s1", agent, ["stay"])
        b.add_actions("s2", agent, ["stay"])

    b.add_transition("s0", ("L", "L"), "s1")
    b.add_transition("s0", ("R", "R"), "s1")
    b.add_transition("s0", ("L", "R"), "s2")
    b.add_transition("s0", ("R", "L"), "s2")
    b.add_transition("s1", ("stay", "stay"), "s1")
    b.add_transition("s2", ("stay", "stay"), "s2")
    return b.build()


def make_prisoners_dilemma() -> CGS:
    """Iterated Prisoner's Dilemma. Actions: C/D. States encode last round."""
    b = CGSBuilder(["p1", "p2"])
    states_labels = {
        "s0": {"start"},
        "s_CC": {"mutual_cooperate"},
        "s_CD": {"p1_sucker", "p2_temptation"},
        "s_DC": {"p1_temptation", "p2_sucker"},
        "s_DD": {"mutual_defect"},
    }
    for s, labels in states_labels.items():
        b.add_state(s, labels)
        for agent in ["p1", "p2"]:
            b.add_actions(s, agent, ["C", "D"])

    for s in states_labels:
        b.add_transition(s, ("C", "C"), "s_CC")
        b.add_transition(s, ("C", "D"), "s_CD")
        b.add_transition(s, ("D", "C"), "s_DC")
        b.add_transition(s, ("D", "D"), "s_DD")
    return b.build()


def make_traffic_intersection() -> CGS:
    """Traffic: two cars, go/wait. Both go = crash."""
    b = CGSBuilder(["car1", "car2"])
    b.add_state("approach", {"approach"})
    b.add_state("car1_through", {"car1_ok"})
    b.add_state("car2_through", {"car2_ok"})
    b.add_state("crash", {"crash", "bad"})
    b.add_state("waiting", {"delay"})

    for agent in ["car1", "car2"]:
        b.add_actions("approach", agent, ["go", "wait"])
        for s in ["car1_through", "car2_through", "crash", "waiting"]:
            b.add_actions(s, agent, ["stay"])

    b.add_transition("approach", ("go", "go"), "crash")
    b.add_transition("approach", ("go", "wait"), "car1_through")
    b.add_transition("approach", ("wait", "go"), "car2_through")
    b.add_transition("approach", ("wait", "wait"), "waiting")
    for s in ["car1_through", "car2_through", "crash", "waiting"]:
        b.add_transition(s, ("stay", "stay"), s)
    return b.build()


def make_resource_sharing_game() -> CGS:
    """3 agents compete for 2 resources (A, B). Priority allocation."""
    b = CGSBuilder(["ag1", "ag2", "ag3"])
    b.add_state("idle", {"idle"})

    outcomes = []
    for a1 in ["A", "B", "N"]:
        for a2 in ["A", "B", "N"]:
            for a3 in ["A", "B", "N"]:
                name = f"o_{a1}{a2}{a3}"
                labels = set()
                requests = {"ag1": a1, "ag2": a2, "ag3": a3}
                allocated = set()
                for agent in ["ag1", "ag2", "ag3"]:
                    r = requests[agent]
                    if r != "N" and r not in allocated:
                        allocated.add(r)
                        labels.add(f"{agent}_got_{r}")
                if not labels:
                    labels.add("nobody_got")
                b.add_state(name, labels)
                outcomes.append(name)

    for agent in ["ag1", "ag2", "ag3"]:
        b.add_actions("idle", agent, ["A", "B", "N"])
        for o in outcomes:
            b.add_actions(o, agent, ["stay"])

    for a1 in ["A", "B", "N"]:
        for a2 in ["A", "B", "N"]:
            for a3 in ["A", "B", "N"]:
                b.add_transition("idle", (a1, a2, a3), f"o_{a1}{a2}{a3}")

    for o in outcomes:
        b.add_transition(o, ("stay", "stay", "stay"), o)
    return b.build()
