"""V201: Assume-Guarantee Games -- Compositional Game Solving.

Decomposes large parity and energy games into smaller component games
connected at interface vertices, solves each independently under
interface assumptions, then discharges assumptions to obtain a global solution.

Composes:
- V156 (parity games): ParityGame, Solution, zielonka, attractor
- V160 (energy games): EnergyGame, EnergyResult, solve_energy
- V147 (assume-guarantee reasoning): discharge patterns

Key ideas:
1. Game decomposition at interface vertices (shared between components)
2. Interface contracts: assumptions on opponent behavior at boundaries
3. Local solving: each component solved under optimistic/pessimistic assumptions
4. Assumption discharge: verify consistency of mutual assumptions
5. Strategy composition: combine local strategies into global winning strategy
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Dict, Set, List, Tuple, Optional, FrozenSet, Any
)
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))

from parity_games import (
    ParityGame, Player, Solution, zielonka, attractor, make_game,
    verify_solution, simulate_play as pg_simulate
)
from energy_games import (
    EnergyGame, EnergyResult, solve_energy, solve_fixed_energy,
    make_simple_energy_game, verify_energy_strategy,
    simulate_play as eg_simulate
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class GameType(Enum):
    PARITY = "parity"
    ENERGY = "energy"


class AssumptionKind(Enum):
    """How interface vertices are assumed to behave."""
    OPTIMISTIC = "optimistic"     # Assume interface favors us (Even wins)
    PESSIMISTIC = "pessimistic"   # Assume interface favors opponent (Odd wins)
    FIXED = "fixed"               # Interface vertices have fixed winner


class DischargeVerdict(Enum):
    SOUND = "sound"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


@dataclass
class InterfaceContract:
    """Contract for an interface vertex between two components."""
    vertex: int
    owner: Player
    # Which component "provides" this vertex (it's in that component's territory)
    provider: str
    # Which component "requires" assumptions about this vertex
    consumer: str
    # Assumed winner at this vertex
    assumed_winner: Player


@dataclass
class GameComponent:
    """A sub-game with interface assumptions."""
    name: str
    vertices: Set[int]
    # Interface vertices shared with other components
    interface_in: Set[int]   # Vertices entering from other components
    interface_out: Set[int]  # Vertices leaving to other components

    def all_vertices(self) -> Set[int]:
        return self.vertices | self.interface_in | self.interface_out


@dataclass
class ComponentResult:
    """Result of solving a single component game."""
    name: str
    game_type: GameType
    # Parity results
    win_even: Set[int] = field(default_factory=set)
    win_odd: Set[int] = field(default_factory=set)
    strategy_even: Dict[int, int] = field(default_factory=dict)
    strategy_odd: Dict[int, int] = field(default_factory=dict)
    # Energy results (if applicable)
    min_energy: Dict[int, Optional[int]] = field(default_factory=dict)
    # Assumptions used
    assumptions: List[InterfaceContract] = field(default_factory=list)


@dataclass
class AGGameResult:
    """Full result of compositional game solving."""
    verdict: DischargeVerdict
    game_type: GameType
    # Global solution
    win_even: Set[int] = field(default_factory=set)
    win_odd: Set[int] = field(default_factory=set)
    strategy_even: Dict[int, int] = field(default_factory=dict)
    strategy_odd: Dict[int, int] = field(default_factory=dict)
    # Per-component results
    component_results: Dict[str, ComponentResult] = field(default_factory=dict)
    # Discharge info
    discharge_strategy: str = ""
    discharge_details: Dict[str, Any] = field(default_factory=dict)
    # Energy-specific
    min_energy: Dict[int, Optional[int]] = field(default_factory=dict)


@dataclass
class GameDecomposition:
    """A decomposition of a game into components."""
    components: List[GameComponent]
    contracts: List[InterfaceContract]
    original_game_type: GameType


# ---------------------------------------------------------------------------
# Game decomposition
# ---------------------------------------------------------------------------

def decompose_parity_game(
    game: ParityGame,
    partition: Dict[str, Set[int]],
) -> Tuple[GameDecomposition, Dict[str, ParityGame]]:
    """Decompose a parity game into components based on vertex partition.

    Args:
        game: The original parity game
        partition: Map from component name to set of vertices in that component

    Returns:
        (decomposition, sub_games) where sub_games maps name -> restricted ParityGame
    """
    # Validate partition covers all vertices
    all_partitioned = set()
    for verts in partition.values():
        all_partitioned |= verts
    if all_partitioned != game.vertices:
        missing = game.vertices - all_partitioned
        extra = all_partitioned - game.vertices
        raise ValueError(
            f"Partition mismatch: missing={missing}, extra={extra}"
        )

    # Find interface vertices -- vertices with edges crossing components
    # Map vertex -> component name
    vertex_to_comp = {}
    for name, verts in partition.items():
        for v in verts:
            vertex_to_comp[v] = name

    components = []
    contracts = []
    sub_games = {}

    for name, verts in partition.items():
        interface_in = set()
        interface_out = set()

        for v in verts:
            for succ in game.successors(v):
                if succ not in verts:
                    interface_out.add(succ)
            for pred_v in game.vertices:
                if pred_v not in verts and v in game.successors(pred_v):
                    interface_in.add(pred_v)

        comp = GameComponent(
            name=name,
            vertices=verts,
            interface_in=interface_in,
            interface_out=interface_out,
        )
        components.append(comp)

        # Build sub-game including interface vertices
        sub_verts = verts | interface_in | interface_out
        sg = ParityGame()
        for v in sub_verts:
            if v in game.vertices:
                sg.add_vertex(v, game.owner[v], game.priority[v])

        # Add edges within the sub-game
        for v in sub_verts:
            if v in game.edges:
                for succ in game.edges[v]:
                    if succ in sub_verts:
                        sg.add_edge(v, succ)

        # Interface out vertices need self-loops as sinks
        for v in interface_out:
            if v in sg.vertices and not sg.successors(v) & sub_verts:
                sg.add_edge(v, v)

        sub_games[name] = sg

    # Create contracts for interface edges
    for comp in components:
        for v in comp.interface_out:
            target_comp = vertex_to_comp[v]
            contracts.append(InterfaceContract(
                vertex=v,
                owner=game.owner[v],
                provider=target_comp,
                consumer=comp.name,
                assumed_winner=Player.EVEN,  # default optimistic
            ))

    decomp = GameDecomposition(
        components=components,
        contracts=contracts,
        original_game_type=GameType.PARITY,
    )

    return decomp, sub_games


def decompose_energy_game(
    game: EnergyGame,
    partition: Dict[str, Set[int]],
) -> Tuple[GameDecomposition, Dict[str, EnergyGame]]:
    """Decompose an energy game into components based on vertex partition."""
    all_partitioned = set()
    for verts in partition.values():
        all_partitioned |= verts
    if all_partitioned != game.vertices:
        raise ValueError("Partition must cover all vertices exactly")

    vertex_to_comp = {}
    for name, verts in partition.items():
        for v in verts:
            vertex_to_comp[v] = name

    components = []
    contracts = []
    sub_games = {}

    for name, verts in partition.items():
        interface_in = set()
        interface_out = set()

        for v in verts:
            for succ, w in game.successors(v):
                if succ not in verts:
                    interface_out.add(succ)
        for v2 in game.vertices:
            if v2 not in verts:
                for succ, w in game.successors(v2):
                    if succ in verts:
                        interface_in.add(v2)

        comp = GameComponent(
            name=name,
            vertices=verts,
            interface_in=interface_in,
            interface_out=interface_out,
        )
        components.append(comp)

        # Build sub-game
        sub_verts = verts | interface_in | interface_out
        sg = EnergyGame()
        for v in sub_verts:
            if v in game.vertices:
                sg.add_vertex(v, game.owner[v])

        for v in sub_verts:
            if v in game.edges:
                for succ, w in game.edges[v]:
                    if succ in sub_verts:
                        sg.add_edge(v, succ, w)

        # Interface out vertices get zero-weight self-loops
        for v in interface_out:
            if v in sg.vertices:
                has_edge = any(s in sub_verts for s, _ in game.edges.get(v, []))
                if not has_edge:
                    sg.add_edge(v, v, 0)

        sub_games[name] = sg

    for comp in components:
        for v in comp.interface_out:
            target_comp = vertex_to_comp[v]
            contracts.append(InterfaceContract(
                vertex=v,
                owner=game.owner[v],
                provider=target_comp,
                consumer=comp.name,
                assumed_winner=Player.EVEN,
            ))

    decomp = GameDecomposition(
        components=components,
        contracts=contracts,
        original_game_type=GameType.ENERGY,
    )

    return decomp, sub_games


# ---------------------------------------------------------------------------
# Component solving under assumptions
# ---------------------------------------------------------------------------

def solve_component_parity(
    name: str,
    sub_game: ParityGame,
    component: GameComponent,
    assumptions: List[InterfaceContract],
) -> ComponentResult:
    """Solve a parity sub-game under interface assumptions.

    Interface vertices assumed won by Even get priority 0 (safe for Even).
    Interface vertices assumed won by Odd get priority 1 (bad for Even).
    """
    # Create a modified game reflecting assumptions
    modified = ParityGame()

    for v in sub_game.vertices:
        prio = sub_game.priority[v]
        owner = sub_game.owner[v]

        # Check if this vertex has an assumption
        for a in assumptions:
            if a.vertex == v:
                if a.assumed_winner == Player.EVEN:
                    # Even wins here -- make it a sink with even priority
                    prio = 0
                else:
                    # Odd wins here -- make it a sink with odd priority
                    prio = 1
                break

        modified.add_vertex(v, owner, prio)

    for v in sub_game.vertices:
        for succ in sub_game.edges.get(v, set()):
            if succ in modified.vertices:
                modified.add_edge(v, succ)

    sol = zielonka(modified)

    return ComponentResult(
        name=name,
        game_type=GameType.PARITY,
        win_even=sol.win_even & component.vertices,
        win_odd=sol.win_odd & component.vertices,
        strategy_even={v: s for v, s in sol.strategy_even.items()
                       if v in component.vertices or v in component.interface_in},
        strategy_odd={v: s for v, s in sol.strategy_odd.items()
                      if v in component.vertices or v in component.interface_in},
        assumptions=assumptions,
    )


def solve_component_energy(
    name: str,
    sub_game: EnergyGame,
    component: GameComponent,
    assumptions: List[InterfaceContract],
    energy_bound: Optional[int] = None,
) -> ComponentResult:
    """Solve an energy sub-game under interface assumptions.

    Interface vertices assumed won by Even get large positive self-loop (energy source).
    Interface vertices assumed won by Odd get large negative self-loop (energy drain).
    """
    modified = EnergyGame()

    assumed_vertices = {a.vertex for a in assumptions}
    bound = energy_bound or max(
        abs(w) for v in sub_game.edges for _, w in sub_game.edges[v]
    ) * len(sub_game.vertices) if sub_game.edges else 10

    for v in sub_game.vertices:
        modified.add_vertex(v, sub_game.owner[v])

    for v in sub_game.vertices:
        if v in assumed_vertices:
            # Find the assumption
            for a in assumptions:
                if a.vertex == v:
                    if a.assumed_winner == Player.EVEN:
                        modified.add_edge(v, v, bound)  # Energy source
                    else:
                        modified.add_edge(v, v, -bound)  # Energy drain
                    break
        else:
            for succ, w in sub_game.edges.get(v, []):
                if succ in modified.vertices:
                    modified.add_edge(v, succ, w)

    result = solve_energy(modified)

    return ComponentResult(
        name=name,
        game_type=GameType.ENERGY,
        win_even=result.win_energy & component.vertices,
        win_odd=result.win_opponent & component.vertices,
        strategy_even={v: s for v, s in result.strategy_energy.items()
                       if v in component.vertices},
        strategy_odd={v: s for v, s in result.strategy_opponent.items()
                      if v in component.vertices},
        min_energy={v: e for v, e in result.min_energy.items()
                    if v in component.vertices},
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Assumption discharge
# ---------------------------------------------------------------------------

def discharge_direct(
    decomp: GameDecomposition,
    results: Dict[str, ComponentResult],
) -> Tuple[DischargeVerdict, Dict[str, Any]]:
    """Direct discharge: check each assumption against provider's solution.

    For each contract (vertex v, assumed winner W by consumer C):
    - Check if provider P's solution has v in W's winning region
    - If all match: SOUND. If any mismatch: REFUTED.
    """
    mismatches = []
    matched = []

    for contract in decomp.contracts:
        provider_result = results.get(contract.provider)
        if provider_result is None:
            mismatches.append({
                'contract': contract,
                'reason': f'no result for provider {contract.provider}',
            })
            continue

        if contract.assumed_winner == Player.EVEN:
            if contract.vertex in provider_result.win_even:
                matched.append(contract)
            else:
                mismatches.append({
                    'contract': contract,
                    'reason': f'vertex {contract.vertex} not in Even winning region of {contract.provider}',
                })
        else:
            if contract.vertex in provider_result.win_odd:
                matched.append(contract)
            else:
                mismatches.append({
                    'contract': contract,
                    'reason': f'vertex {contract.vertex} not in Odd winning region of {contract.provider}',
                })

    if not mismatches:
        verdict = DischargeVerdict.SOUND
    else:
        verdict = DischargeVerdict.REFUTED

    return verdict, {
        'matched': len(matched),
        'mismatched': len(mismatches),
        'mismatches': mismatches,
    }


def discharge_iterative(
    decomp: GameDecomposition,
    sub_games: Dict[str, Any],  # ParityGame or EnergyGame
    components: Dict[str, GameComponent],
    game_type: GameType,
    max_iterations: int = 10,
) -> Tuple[DischargeVerdict, Dict[str, ComponentResult], Dict[str, Any]]:
    """Iterative discharge: start pessimistic, monotonically upgrade assumptions.

    Sound approach:
    1. Start with PESSIMISTIC assumptions (all interface = Odd wins)
       This is a sound under-approximation of Even's winning region.
    2. Solve each component under current assumptions.
    3. For each interface vertex, if the provider NOW shows it in Even's region,
       upgrade the assumption from Odd to Even.
    4. Re-solve. Repeat until fixpoint (no more upgrades).

    Monotonicity guarantees convergence and soundness:
    - Pessimistic start = sound under-approximation
    - Each upgrade only happens when justified by provider's solution
    - No circular self-justification possible (upgrades are one-directional)
    """
    # Initialize contracts with PESSIMISTIC assumptions (sound starting point)
    current_contracts = []
    for c in decomp.contracts:
        current_contracts.append(InterfaceContract(
            vertex=c.vertex,
            owner=c.owner,
            provider=c.provider,
            consumer=c.consumer,
            assumed_winner=Player.ODD,  # Pessimistic start
        ))

    results = {}
    iteration = 0
    history = []

    while iteration < max_iterations:
        iteration += 1

        # Solve each component under current assumptions
        new_results = {}
        for comp in decomp.components:
            comp_assumptions = [c for c in current_contracts if c.consumer == comp.name]

            if game_type == GameType.PARITY:
                new_results[comp.name] = solve_component_parity(
                    comp.name, sub_games[comp.name], comp, comp_assumptions
                )
            else:
                new_results[comp.name] = solve_component_energy(
                    comp.name, sub_games[comp.name], comp, comp_assumptions
                )

        # Check if any assumptions can be upgraded (Odd -> Even)
        changed = False
        new_contracts = []
        for contract in current_contracts:
            provider_result = new_results.get(contract.provider)
            if provider_result is None:
                new_contracts.append(contract)
                continue

            if contract.assumed_winner == Player.ODD:
                # Can we upgrade to Even?
                if contract.vertex in provider_result.win_even:
                    changed = True
                    new_contracts.append(InterfaceContract(
                        vertex=contract.vertex,
                        owner=contract.owner,
                        provider=contract.provider,
                        consumer=contract.consumer,
                        assumed_winner=Player.EVEN,
                    ))
                else:
                    new_contracts.append(contract)
            else:
                # Already Even -- keep (monotone: never downgrade)
                new_contracts.append(contract)

        history.append({
            'iteration': iteration,
            'changed': changed,
            'assumptions': [(c.vertex, c.assumed_winner.name) for c in new_contracts],
        })

        results = new_results
        current_contracts = new_contracts

        if not changed:
            break

    # Build updated decomposition with final contracts for discharge check
    final_decomp = GameDecomposition(
        decomp.components, current_contracts, decomp.original_game_type
    )
    verdict, discharge_info = discharge_direct(final_decomp, results)

    return verdict, results, {
        'iterations': iteration,
        'converged': not (iteration >= max_iterations),
        'history': history,
        'discharge_info': discharge_info,
    }


def discharge_pessimistic(
    decomp: GameDecomposition,
    sub_games: Dict[str, Any],
    components: Dict[str, GameComponent],
    game_type: GameType,
) -> Tuple[DischargeVerdict, Dict[str, ComponentResult], Dict[str, Any]]:
    """Pessimistic discharge: assume worst case for all interfaces.

    Even's winning region under pessimistic assumptions is a sound
    under-approximation of the true winning region.
    """
    pessimistic_contracts = []
    for contract in decomp.contracts:
        pessimistic_contracts.append(InterfaceContract(
            vertex=contract.vertex,
            owner=contract.owner,
            provider=contract.provider,
            consumer=contract.consumer,
            assumed_winner=Player.ODD,  # Worst case for Even
        ))

    # Override decomposition contracts
    pessimistic_decomp = GameDecomposition(
        components=decomp.components,
        contracts=pessimistic_contracts,
        original_game_type=decomp.original_game_type,
    )

    results = {}
    for comp in decomp.components:
        comp_assumptions = [c for c in pessimistic_contracts if c.consumer == comp.name]

        if game_type == GameType.PARITY:
            results[comp.name] = solve_component_parity(
                comp.name, sub_games[comp.name], comp, comp_assumptions
            )
        else:
            results[comp.name] = solve_component_energy(
                comp.name, sub_games[comp.name], comp, comp_assumptions
            )

    # Pessimistic is always sound (under-approximation)
    return DischargeVerdict.SOUND, results, {
        'strategy': 'pessimistic',
        'note': 'Under-approximation of Even winning region',
    }


# ---------------------------------------------------------------------------
# Main compositional solvers
# ---------------------------------------------------------------------------

def solve_parity_compositional(
    game: ParityGame,
    partition: Dict[str, Set[int]],
    strategy: str = "iterative",
) -> AGGameResult:
    """Solve a parity game compositionally.

    Args:
        game: The parity game to solve
        partition: Vertex partition into named components
        strategy: "iterative", "pessimistic", or "optimistic"

    Returns:
        AGGameResult with global solution
    """
    decomp, sub_games = decompose_parity_game(game, partition)
    comp_map = {c.name: c for c in decomp.components}

    if strategy == "pessimistic":
        verdict, results, details = discharge_pessimistic(
            decomp, sub_games, comp_map, GameType.PARITY
        )
    elif strategy == "optimistic":
        # Solve once with optimistic assumptions, check
        results = {}
        for comp in decomp.components:
            comp_assumptions = [c for c in decomp.contracts if c.consumer == comp.name]
            results[comp.name] = solve_component_parity(
                comp.name, sub_games[comp.name], comp, comp_assumptions
            )
        verdict, discharge_info = discharge_direct(decomp, results)
        details = {'strategy': 'optimistic', 'discharge_info': discharge_info}
    else:  # iterative
        verdict, results, details = discharge_iterative(
            decomp, sub_games, comp_map, GameType.PARITY
        )

    # Combine results
    global_even = set()
    global_odd = set()
    global_strat_even = {}
    global_strat_odd = {}

    for name, res in results.items():
        global_even |= res.win_even
        global_odd |= res.win_odd
        global_strat_even.update(res.strategy_even)
        global_strat_odd.update(res.strategy_odd)

    return AGGameResult(
        verdict=verdict,
        game_type=GameType.PARITY,
        win_even=global_even,
        win_odd=global_odd,
        strategy_even=global_strat_even,
        strategy_odd=global_strat_odd,
        component_results=results,
        discharge_strategy=strategy,
        discharge_details=details,
    )


def solve_energy_compositional(
    game: EnergyGame,
    partition: Dict[str, Set[int]],
    strategy: str = "iterative",
) -> AGGameResult:
    """Solve an energy game compositionally.

    Args:
        game: The energy game to solve
        partition: Vertex partition into named components
        strategy: "iterative", "pessimistic", or "optimistic"

    Returns:
        AGGameResult with global solution
    """
    decomp, sub_games = decompose_energy_game(game, partition)
    comp_map = {c.name: c for c in decomp.components}

    if strategy == "pessimistic":
        verdict, results, details = discharge_pessimistic(
            decomp, sub_games, comp_map, GameType.ENERGY
        )
    elif strategy == "optimistic":
        results = {}
        for comp in decomp.components:
            comp_assumptions = [c for c in decomp.contracts if c.consumer == comp.name]
            results[comp.name] = solve_component_energy(
                comp.name, sub_games[comp.name], comp, comp_assumptions
            )
        verdict, discharge_info = discharge_direct(decomp, results)
        details = {'strategy': 'optimistic', 'discharge_info': discharge_info}
    else:
        verdict, results, details = discharge_iterative(
            decomp, sub_games, comp_map, GameType.ENERGY
        )

    global_even = set()
    global_odd = set()
    global_strat_even = {}
    global_strat_odd = {}
    global_energy = {}

    for name, res in results.items():
        global_even |= res.win_even
        global_odd |= res.win_odd
        global_strat_even.update(res.strategy_even)
        global_strat_odd.update(res.strategy_odd)
        global_energy.update(res.min_energy)

    return AGGameResult(
        verdict=verdict,
        game_type=GameType.ENERGY,
        win_even=global_even,
        win_odd=global_odd,
        strategy_even=global_strat_even,
        strategy_odd=global_strat_odd,
        component_results=results,
        discharge_strategy=strategy,
        discharge_details=details,
        min_energy=global_energy,
    )


# ---------------------------------------------------------------------------
# Verification: compare compositional vs monolithic
# ---------------------------------------------------------------------------

def verify_against_monolithic_parity(
    game: ParityGame,
    ag_result: AGGameResult,
) -> Dict[str, Any]:
    """Compare compositional result against monolithic solution."""
    mono_sol = zielonka(game)

    # Under pessimistic: compositional Even region is subset of monolithic
    # Under iterative with SOUND verdict: should match exactly
    even_match = ag_result.win_even == mono_sol.win_even
    odd_match = ag_result.win_odd == mono_sol.win_odd

    even_subset = ag_result.win_even <= mono_sol.win_even
    even_superset = ag_result.win_even >= mono_sol.win_even

    return {
        'exact_match': even_match and odd_match,
        'even_subset': even_subset,
        'even_superset': even_superset,
        'mono_even': mono_sol.win_even,
        'mono_odd': mono_sol.win_odd,
        'comp_even': ag_result.win_even,
        'comp_odd': ag_result.win_odd,
        'extra_even': ag_result.win_even - mono_sol.win_even,
        'missing_even': mono_sol.win_even - ag_result.win_even,
    }


def verify_against_monolithic_energy(
    game: EnergyGame,
    ag_result: AGGameResult,
) -> Dict[str, Any]:
    """Compare compositional energy result against monolithic solution."""
    mono = solve_energy(game)

    even_match = ag_result.win_even == mono.win_energy
    odd_match = ag_result.win_odd == mono.win_opponent

    even_subset = ag_result.win_even <= mono.win_energy

    return {
        'exact_match': even_match and odd_match,
        'even_subset': even_subset,
        'mono_even': mono.win_energy,
        'mono_odd': mono.win_opponent,
        'comp_even': ag_result.win_even,
        'comp_odd': ag_result.win_odd,
    }


# ---------------------------------------------------------------------------
# Strategy composition and validation
# ---------------------------------------------------------------------------

def compose_strategies(
    game: ParityGame,
    ag_result: AGGameResult,
) -> Solution:
    """Compose component strategies into a global Solution for validation."""
    # Fill in missing strategy entries for interface vertices
    strat_even = dict(ag_result.strategy_even)
    strat_odd = dict(ag_result.strategy_odd)

    # For vertices in Even's winning region without a strategy, pick any successor
    for v in ag_result.win_even:
        if game.owner[v] == Player.EVEN and v not in strat_even:
            succs = game.successors(v)
            # Prefer successors in Even's winning region
            for s in succs:
                if s in ag_result.win_even:
                    strat_even[v] = s
                    break
            else:
                if succs:
                    strat_even[v] = next(iter(succs))

    for v in ag_result.win_odd:
        if game.owner[v] == Player.ODD and v not in strat_odd:
            succs = game.successors(v)
            for s in succs:
                if s in ag_result.win_odd:
                    strat_odd[v] = s
                    break
            else:
                if succs:
                    strat_odd[v] = next(iter(succs))

    return Solution(
        win_even=ag_result.win_even,
        win_odd=ag_result.win_odd,
        strategy_even=strat_even,
        strategy_odd=strat_odd,
    )


# ---------------------------------------------------------------------------
# Automatic partitioning heuristics
# ---------------------------------------------------------------------------

def partition_by_priority_bands(
    game: ParityGame,
    num_bands: int = 2,
) -> Dict[str, Set[int]]:
    """Partition vertices into bands by priority ranges."""
    if not game.vertices:
        return {}

    max_p = game.max_priority()
    band_size = max(1, (max_p + 1 + num_bands - 1) // num_bands)

    partition = defaultdict(set)
    for v in game.vertices:
        band = game.priority[v] // band_size
        partition[f"band_{band}"].add(v)

    return dict(partition)


def partition_by_owner(game: ParityGame) -> Dict[str, Set[int]]:
    """Partition vertices by owner (Even vs Odd)."""
    even_verts = {v for v in game.vertices if game.owner[v] == Player.EVEN}
    odd_verts = {v for v in game.vertices if game.owner[v] == Player.ODD}

    result = {}
    if even_verts:
        result["even_owned"] = even_verts
    if odd_verts:
        result["odd_owned"] = odd_verts
    return result


def partition_by_scc(game: ParityGame) -> Dict[str, Set[int]]:
    """Partition vertices by strongly connected components."""
    # Tarjan's SCC
    index_counter = [0]
    stack = []
    on_stack = set()
    index = {}
    lowlink = {}
    sccs = []

    def strongconnect(v):
        index[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in game.successors(v):
            if w not in index:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])

        if lowlink[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in sorted(game.vertices):
        if v not in index:
            strongconnect(v)

    # If only one SCC, fall back to priority bands
    if len(sccs) <= 1:
        return partition_by_priority_bands(game)

    partition = {}
    for i, scc in enumerate(sccs):
        partition[f"scc_{i}"] = scc

    return partition


def auto_partition_energy(
    game: EnergyGame,
    num_parts: int = 2,
) -> Dict[str, Set[int]]:
    """Simple partition of energy game vertices."""
    verts = sorted(game.vertices)
    if not verts:
        return {}

    part_size = max(1, (len(verts) + num_parts - 1) // num_parts)
    partition = {}
    for i in range(0, len(verts), part_size):
        chunk = set(verts[i:i + part_size])
        partition[f"part_{i // part_size}"] = chunk

    return partition


# ---------------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------------

def solve_parity_ag(
    game: ParityGame,
    partition: Optional[Dict[str, Set[int]]] = None,
    strategy: str = "iterative",
) -> AGGameResult:
    """Solve parity game with automatic partitioning if not provided."""
    if partition is None:
        partition = partition_by_scc(game)
    return solve_parity_compositional(game, partition, strategy)


def solve_energy_ag(
    game: EnergyGame,
    partition: Optional[Dict[str, Set[int]]] = None,
    strategy: str = "iterative",
) -> AGGameResult:
    """Solve energy game with automatic partitioning if not provided."""
    if partition is None:
        partition = auto_partition_energy(game)
    return solve_energy_compositional(game, partition, strategy)


def compare_strategies_parity(
    game: ParityGame,
    partition: Dict[str, Set[int]],
) -> Dict[str, Any]:
    """Compare all discharge strategies on the same game and partition."""
    results = {}
    for strat in ["optimistic", "pessimistic", "iterative"]:
        ag_result = solve_parity_compositional(game, partition, strat)
        mono_check = verify_against_monolithic_parity(game, ag_result)
        results[strat] = {
            'verdict': ag_result.verdict.value,
            'win_even': ag_result.win_even,
            'win_odd': ag_result.win_odd,
            'exact_match': mono_check['exact_match'],
            'even_subset': mono_check['even_subset'],
        }
    return results


def compare_strategies_energy(
    game: EnergyGame,
    partition: Dict[str, Set[int]],
) -> Dict[str, Any]:
    """Compare all discharge strategies on the same energy game."""
    results = {}
    for strat in ["optimistic", "pessimistic", "iterative"]:
        ag_result = solve_energy_compositional(game, partition, strat)
        mono_check = verify_against_monolithic_energy(game, ag_result)
        results[strat] = {
            'verdict': ag_result.verdict.value,
            'win_even': ag_result.win_even,
            'win_odd': ag_result.win_odd,
            'exact_match': mono_check['exact_match'],
            'even_subset': mono_check['even_subset'],
        }
    return results


def ag_game_summary(result: AGGameResult) -> str:
    """Generate a human-readable summary of an AG game result."""
    lines = [
        f"Assume-Guarantee {result.game_type.value.title()} Game Result",
        f"  Verdict: {result.verdict.value}",
        f"  Strategy: {result.discharge_strategy}",
        f"  Even wins: {sorted(result.win_even)}",
        f"  Odd wins: {sorted(result.win_odd)}",
        f"  Components: {len(result.component_results)}",
    ]

    for name, cr in sorted(result.component_results.items()):
        lines.append(f"    {name}: Even={sorted(cr.win_even)}, Odd={sorted(cr.win_odd)}")

    if result.game_type == GameType.ENERGY and result.min_energy:
        finite = {v: e for v, e in result.min_energy.items() if e is not None}
        lines.append(f"  Min energy: {dict(sorted(finite.items()))}")

    return "\n".join(lines)
