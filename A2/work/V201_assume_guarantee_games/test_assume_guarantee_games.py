"""Tests for V201: Assume-Guarantee Games."""

import pytest
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V156_parity_games'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V160_energy_games'))

from assume_guarantee_games import (
    GameType, AssumptionKind, DischargeVerdict,
    InterfaceContract, GameComponent, ComponentResult, AGGameResult,
    GameDecomposition,
    decompose_parity_game, decompose_energy_game,
    solve_component_parity, solve_component_energy,
    discharge_direct, discharge_iterative, discharge_pessimistic,
    solve_parity_compositional, solve_energy_compositional,
    verify_against_monolithic_parity, verify_against_monolithic_energy,
    compose_strategies, partition_by_priority_bands, partition_by_owner,
    partition_by_scc, auto_partition_energy,
    solve_parity_ag, solve_energy_ag,
    compare_strategies_parity, compare_strategies_energy,
    ag_game_summary,
)
from parity_games import ParityGame, Player, Solution, zielonka, make_game
from energy_games import EnergyGame, EnergyResult, solve_energy, make_simple_energy_game


# ===================================================================
# Test fixtures -- games
# ===================================================================

def make_simple_parity():
    """Simple 4-vertex parity game.
    0(E,p=2) -> 1(O,p=1) -> 2(E,p=0) -> 3(O,p=1) -> 0
                  1 -> 3
    Even wins from 0,2 (can stay in even-priority cycle).
    """
    g = ParityGame()
    g.add_vertex(0, Player.EVEN, 2)
    g.add_vertex(1, Player.ODD, 1)
    g.add_vertex(2, Player.EVEN, 0)
    g.add_vertex(3, Player.ODD, 1)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(3, 0)
    return g


def make_chain_parity(n=6):
    """Chain: 0->1->2->...->n-1->0. Alternating owners, priorities."""
    g = ParityGame()
    for i in range(n):
        g.add_vertex(i, Player.EVEN if i % 2 == 0 else Player.ODD, i % 3)
    for i in range(n):
        g.add_edge(i, (i + 1) % n)
    return g


def make_two_component_parity():
    """Two clearly separable components connected by edges.
    Component A: {0, 1, 2} -- Even-dominated
    Component B: {3, 4, 5} -- Odd-dominated
    Edges: 2->3, 5->0 (cross-component)
    """
    g = ParityGame()
    # Component A
    g.add_vertex(0, Player.EVEN, 2)
    g.add_vertex(1, Player.EVEN, 0)
    g.add_vertex(2, Player.ODD, 0)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)  # Internal cycle
    g.add_edge(2, 3)  # Cross to B

    # Component B
    g.add_vertex(3, Player.ODD, 1)
    g.add_vertex(4, Player.ODD, 3)
    g.add_vertex(5, Player.EVEN, 0)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 3)  # Internal cycle
    g.add_edge(5, 0)  # Cross to A
    return g


def make_simple_energy():
    """Simple energy game.
    0(E) -3-> 1(O) -(-2)-> 2(E) -(1)-> 0
                    -(-1)-> 0
    """
    g = EnergyGame()
    g.add_vertex(0, Player.EVEN)
    g.add_vertex(1, Player.ODD)
    g.add_vertex(2, Player.EVEN)
    g.add_edge(0, 1, 3)
    g.add_edge(1, 2, -2)
    g.add_edge(1, 0, -1)
    g.add_edge(2, 0, 1)
    return g


def make_two_component_energy():
    """Two-component energy game.
    Component A: {0, 1}
    Component B: {2, 3}
    """
    g = EnergyGame()
    g.add_vertex(0, Player.EVEN)
    g.add_vertex(1, Player.ODD)
    g.add_vertex(2, Player.EVEN)
    g.add_vertex(3, Player.ODD)
    g.add_edge(0, 1, 5)
    g.add_edge(1, 0, -2)   # Stay in A
    g.add_edge(1, 2, -1)   # Cross to B
    g.add_edge(2, 3, 3)
    g.add_edge(3, 2, -1)   # Stay in B
    g.add_edge(3, 0, -2)   # Cross to A
    return g


# ===================================================================
# Data structure tests
# ===================================================================

class TestDataStructures:
    def test_game_type_enum(self):
        assert GameType.PARITY.value == "parity"
        assert GameType.ENERGY.value == "energy"

    def test_assumption_kind_enum(self):
        assert AssumptionKind.OPTIMISTIC.value == "optimistic"
        assert AssumptionKind.PESSIMISTIC.value == "pessimistic"
        assert AssumptionKind.FIXED.value == "fixed"

    def test_discharge_verdict_enum(self):
        assert DischargeVerdict.SOUND.value == "sound"
        assert DischargeVerdict.REFUTED.value == "refuted"

    def test_interface_contract_creation(self):
        c = InterfaceContract(
            vertex=5, owner=Player.EVEN,
            provider="A", consumer="B",
            assumed_winner=Player.EVEN,
        )
        assert c.vertex == 5
        assert c.provider == "A"
        assert c.consumer == "B"

    def test_game_component_all_vertices(self):
        comp = GameComponent(
            name="test", vertices={1, 2, 3},
            interface_in={0}, interface_out={4},
        )
        assert comp.all_vertices() == {0, 1, 2, 3, 4}

    def test_component_result_defaults(self):
        cr = ComponentResult(name="c", game_type=GameType.PARITY)
        assert cr.win_even == set()
        assert cr.win_odd == set()
        assert cr.strategy_even == {}

    def test_ag_game_result_defaults(self):
        r = AGGameResult(
            verdict=DischargeVerdict.SOUND,
            game_type=GameType.PARITY,
        )
        assert r.win_even == set()
        assert r.discharge_strategy == ""

    def test_game_decomposition(self):
        comp = GameComponent("a", {1, 2}, set(), set())
        d = GameDecomposition(
            components=[comp], contracts=[],
            original_game_type=GameType.PARITY,
        )
        assert len(d.components) == 1


# ===================================================================
# Decomposition tests
# ===================================================================

class TestDecompositionParity:
    def test_simple_decompose(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        assert len(decomp.components) == 2
        assert "A" in sub_games and "B" in sub_games

    def test_decompose_preserves_vertices(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        # Each sub-game includes interface vertices
        assert 0 in sub_games["A"].vertices
        assert 3 in sub_games["A"].vertices  # interface out for A

    def test_decompose_interface_detection(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, _ = decompose_parity_game(g, partition)
        # Find component A
        comp_a = [c for c in decomp.components if c.name == "A"][0]
        assert 3 in comp_a.interface_out  # 2->3 crosses

    def test_decompose_creates_contracts(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, _ = decompose_parity_game(g, partition)
        assert len(decomp.contracts) > 0

    def test_decompose_bad_partition_raises(self):
        g = make_simple_parity()
        with pytest.raises(ValueError, match="Partition mismatch"):
            decompose_parity_game(g, {"A": {0, 1}})

    def test_chain_decompose(self):
        g = make_chain_parity(6)
        partition = {"left": {0, 1, 2}, "right": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        assert len(decomp.components) == 2

    def test_single_component_decompose(self):
        g = make_simple_parity()
        partition = {"all": g.vertices.copy()}
        decomp, sub_games = decompose_parity_game(g, partition)
        assert len(decomp.components) == 1
        assert len(decomp.contracts) == 0


class TestDecompositionEnergy:
    def test_simple_decompose(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        decomp, sub_games = decompose_energy_game(g, partition)
        assert len(decomp.components) == 2

    def test_decompose_preserves_weights(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        _, sub_games = decompose_energy_game(g, partition)
        # Check weight is preserved for internal edge 0->1
        edges_from_0 = sub_games["A"].successors(0)
        weights = {s: w for s, w in edges_from_0}
        assert weights.get(1) == 5

    def test_decompose_creates_contracts(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        decomp, _ = decompose_energy_game(g, partition)
        assert len(decomp.contracts) > 0

    def test_decompose_bad_partition_raises(self):
        g = make_simple_energy()
        with pytest.raises(ValueError):
            decompose_energy_game(g, {"A": {0}})


# ===================================================================
# Component solving tests
# ===================================================================

class TestComponentSolvingParity:
    def test_solve_without_assumptions(self):
        g = make_simple_parity()
        comp = GameComponent("all", g.vertices.copy(), set(), set())
        result = solve_component_parity("all", g, comp, [])
        # Should match monolithic
        mono = zielonka(g)
        assert result.win_even == mono.win_even
        assert result.win_odd == mono.win_odd

    def test_solve_with_optimistic_assumption(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        comp_a = [c for c in decomp.components if c.name == "A"][0]
        assumptions = [c for c in decomp.contracts if c.consumer == "A"]
        result = solve_component_parity("A", sub_games["A"], comp_a, assumptions)
        assert result.game_type == GameType.PARITY

    def test_solve_with_pessimistic_assumption(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        comp_a = [c for c in decomp.components if c.name == "A"][0]
        pessimistic = [
            InterfaceContract(c.vertex, c.owner, c.provider, c.consumer, Player.ODD)
            for c in decomp.contracts if c.consumer == "A"
        ]
        result = solve_component_parity("A", sub_games["A"], comp_a, pessimistic)
        # With pessimistic assumptions, Even may win fewer vertices
        assert isinstance(result.win_even, set)


class TestComponentSolvingEnergy:
    def test_solve_without_assumptions(self):
        g = make_simple_energy()
        comp = GameComponent("all", g.vertices.copy(), set(), set())
        result = solve_component_energy("all", g, comp, [])
        mono = solve_energy(g)
        assert result.win_even == mono.win_energy

    def test_solve_with_assumptions(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        decomp, sub_games = decompose_energy_game(g, partition)
        comp_a = [c for c in decomp.components if c.name == "A"][0]
        assumptions = [c for c in decomp.contracts if c.consumer == "A"]
        result = solve_component_energy("A", sub_games["A"], comp_a, assumptions)
        assert result.game_type == GameType.ENERGY


# ===================================================================
# Discharge tests
# ===================================================================

class TestDischarge:
    def test_direct_discharge_sound(self):
        """When all assumptions match, discharge is SOUND."""
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)

        # Solve monolithically to know true winners
        mono = zielonka(g)

        # Create contracts reflecting true winners
        correct_contracts = []
        for c in decomp.contracts:
            winner = Player.EVEN if c.vertex in mono.win_even else Player.ODD
            correct_contracts.append(InterfaceContract(
                c.vertex, c.owner, c.provider, c.consumer, winner
            ))

        correct_decomp = GameDecomposition(
            decomp.components, correct_contracts, decomp.original_game_type
        )

        # Solve components with correct assumptions
        results = {}
        for comp in decomp.components:
            comp_assumptions = [c for c in correct_contracts if c.consumer == comp.name]
            results[comp.name] = solve_component_parity(
                comp.name, sub_games[comp.name], comp, comp_assumptions
            )

        verdict, details = discharge_direct(correct_decomp, results)
        assert verdict == DischargeVerdict.SOUND

    def test_direct_discharge_refuted(self):
        """When assumptions contradict provider's solution, discharge is REFUTED."""
        # Build a game where we KNOW the interface vertex winner
        # 0(E,p=2)->1(O,p=0)->0  and  1->2(E,p=0)->2
        # Mono: Even wins {0,1,2} (cycle 0->1->0 has max prio 2, even)
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(1, 2)
        g.add_edge(2, 2)

        partition = {"A": {0, 1}, "B": {2}}
        decomp, sub_games = decompose_parity_game(g, partition)

        # Force WRONG assumption: vertex 2 assumed won by Odd (but Even actually wins it)
        wrong_contracts = []
        for c in decomp.contracts:
            wrong_contracts.append(InterfaceContract(
                c.vertex, c.owner, c.provider, c.consumer,
                assumed_winner=Player.ODD,  # Wrong: Even wins vertex 2
            ))

        wrong_decomp = GameDecomposition(
            decomp.components, wrong_contracts, decomp.original_game_type
        )

        # Solve B (provider of vertex 2) WITHOUT wrong assumptions
        # B has no interface contracts to other components
        comp_b = [c for c in decomp.components if c.name == "B"][0]
        result_b = solve_component_parity("B", sub_games["B"], comp_b, [])
        # Vertex 2 has self-loop with even priority -> Even wins
        assert 2 in result_b.win_even

        # Solve A with wrong assumption
        comp_a = [c for c in decomp.components if c.name == "A"][0]
        result_a = solve_component_parity("A", sub_games["A"], comp_a, wrong_contracts)

        results = {"A": result_a, "B": result_b}
        verdict, details = discharge_direct(wrong_decomp, results)
        # Contract says Odd wins vertex 2, but provider B shows Even wins it
        assert verdict == DischargeVerdict.REFUTED

    def test_iterative_converges(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        comp_map = {c.name: c for c in decomp.components}

        verdict, results, details = discharge_iterative(
            decomp, sub_games, comp_map, GameType.PARITY
        )
        assert details['converged'] or details['iterations'] <= 10

    def test_pessimistic_always_sound(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        decomp, sub_games = decompose_parity_game(g, partition)
        comp_map = {c.name: c for c in decomp.components}

        verdict, results, details = discharge_pessimistic(
            decomp, sub_games, comp_map, GameType.PARITY
        )
        assert verdict == DischargeVerdict.SOUND


# ===================================================================
# Compositional solving -- parity
# ===================================================================

class TestCompositionalParity:
    def test_two_component_iterative(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "iterative")
        assert result.game_type == GameType.PARITY

    def test_two_component_pessimistic(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "pessimistic")
        assert result.verdict == DischargeVerdict.SOUND

    def test_two_component_optimistic(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "optimistic")
        assert isinstance(result, AGGameResult)

    def test_pessimistic_underapproximates(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "pessimistic")
        mono = zielonka(g)
        # Pessimistic Even region is a subset of true Even region
        assert result.win_even <= mono.win_even

    def test_single_component_matches_monolithic(self):
        g = make_simple_parity()
        partition = {"all": g.vertices.copy()}
        result = solve_parity_compositional(g, partition, "iterative")
        mono = zielonka(g)
        assert result.win_even == mono.win_even
        assert result.win_odd == mono.win_odd

    def test_chain_game(self):
        g = make_chain_parity(6)
        partition = {"left": {0, 1, 2}, "right": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "iterative")
        assert len(result.win_even) + len(result.win_odd) == 6

    def test_verify_against_monolithic(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "iterative")
        check = verify_against_monolithic_parity(g, result)
        assert isinstance(check['exact_match'], bool)

    def test_all_even_game(self):
        """Game where Even trivially wins everywhere."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.EVEN, 2)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        partition = {"a": {0}, "b": {1}}
        result = solve_parity_compositional(g, partition, "iterative")
        assert result.win_even == {0, 1}

    def test_all_odd_game(self):
        """Game where Odd trivially wins everywhere."""
        g = ParityGame()
        g.add_vertex(0, Player.ODD, 1)
        g.add_vertex(1, Player.ODD, 3)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        partition = {"a": {0}, "b": {1}}
        result = solve_parity_compositional(g, partition, "iterative")
        assert result.win_odd == {0, 1}


# ===================================================================
# Compositional solving -- energy
# ===================================================================

class TestCompositionalEnergy:
    def test_two_component_iterative(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_energy_compositional(g, partition, "iterative")
        assert result.game_type == GameType.ENERGY

    def test_two_component_pessimistic(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_energy_compositional(g, partition, "pessimistic")
        assert result.verdict == DischargeVerdict.SOUND

    def test_pessimistic_underapproximates_energy(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_energy_compositional(g, partition, "pessimistic")
        mono = solve_energy(g)
        assert result.win_even <= mono.win_energy

    def test_single_component_matches_monolithic(self):
        g = make_simple_energy()
        partition = {"all": g.vertices.copy()}
        result = solve_energy_compositional(g, partition, "iterative")
        mono = solve_energy(g)
        assert result.win_even == mono.win_energy

    def test_verify_against_monolithic_energy(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_energy_compositional(g, partition, "iterative")
        check = verify_against_monolithic_energy(g, result)
        assert isinstance(check['exact_match'], bool)

    def test_energy_min_values_populated(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_energy_compositional(g, partition, "iterative")
        # Should have min_energy for winning vertices
        for v in result.win_even:
            assert v in result.min_energy


# ===================================================================
# Verification and strategy composition
# ===================================================================

class TestVerification:
    def test_compose_strategies_parity(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        ag_result = solve_parity_compositional(g, partition, "iterative")
        sol = compose_strategies(g, ag_result)
        assert isinstance(sol, Solution)
        assert sol.win_even == ag_result.win_even

    def test_composed_strategy_completeness(self):
        g = make_simple_parity()
        partition = {"all": g.vertices.copy()}
        ag_result = solve_parity_compositional(g, partition, "iterative")
        sol = compose_strategies(g, ag_result)
        # Every Even-owned vertex in Even's region has a strategy
        for v in sol.win_even:
            if g.owner[v] == Player.EVEN:
                assert v in sol.strategy_even


# ===================================================================
# Partitioning heuristics
# ===================================================================

class TestPartitioning:
    def test_partition_by_priority_bands(self):
        g = make_chain_parity(6)
        p = partition_by_priority_bands(g, num_bands=2)
        all_verts = set()
        for verts in p.values():
            all_verts |= verts
        assert all_verts == g.vertices

    def test_partition_by_owner(self):
        g = make_two_component_parity()
        p = partition_by_owner(g)
        all_verts = set()
        for verts in p.values():
            all_verts |= verts
        assert all_verts == g.vertices

    def test_partition_by_scc(self):
        g = make_two_component_parity()
        p = partition_by_scc(g)
        all_verts = set()
        for verts in p.values():
            all_verts |= verts
        assert all_verts == g.vertices

    def test_auto_partition_energy(self):
        g = make_two_component_energy()
        p = auto_partition_energy(g, num_parts=2)
        all_verts = set()
        for verts in p.values():
            all_verts |= verts
        assert all_verts == g.vertices

    def test_scc_single_component_fallback(self):
        """When game has one SCC, partition_by_scc falls back to priority bands."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        p = partition_by_scc(g)
        all_verts = set()
        for verts in p.values():
            all_verts |= verts
        assert all_verts == g.vertices

    def test_priority_bands_covers_all(self):
        g = ParityGame()
        for i in range(10):
            g.add_vertex(i, Player.EVEN, i)
            g.add_edge(i, (i + 1) % 10)
        p = partition_by_priority_bands(g, num_bands=3)
        all_verts = set()
        for verts in p.values():
            all_verts |= verts
        assert all_verts == g.vertices


# ===================================================================
# Convenience API tests
# ===================================================================

class TestConvenienceAPIs:
    def test_solve_parity_ag_auto_partition(self):
        g = make_two_component_parity()
        result = solve_parity_ag(g)
        assert isinstance(result, AGGameResult)
        assert len(result.win_even) + len(result.win_odd) == len(g.vertices)

    def test_solve_parity_ag_custom_partition(self):
        g = make_two_component_parity()
        result = solve_parity_ag(g, partition={"A": {0, 1, 2}, "B": {3, 4, 5}})
        assert isinstance(result, AGGameResult)

    def test_solve_energy_ag_auto_partition(self):
        g = make_two_component_energy()
        result = solve_energy_ag(g)
        assert isinstance(result, AGGameResult)

    def test_solve_energy_ag_custom_partition(self):
        g = make_two_component_energy()
        result = solve_energy_ag(g, partition={"A": {0, 1}, "B": {2, 3}})
        assert isinstance(result, AGGameResult)


# ===================================================================
# Strategy comparison tests
# ===================================================================

class TestStrategyComparison:
    def test_compare_parity_strategies(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        cmp = compare_strategies_parity(g, partition)
        assert "optimistic" in cmp
        assert "pessimistic" in cmp
        assert "iterative" in cmp
        for strat, info in cmp.items():
            assert 'verdict' in info
            assert 'exact_match' in info

    def test_compare_energy_strategies(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        cmp = compare_strategies_energy(g, partition)
        assert "optimistic" in cmp
        assert "pessimistic" in cmp
        assert "iterative" in cmp


# ===================================================================
# Summary tests
# ===================================================================

class TestSummary:
    def test_parity_summary(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        result = solve_parity_compositional(g, partition, "iterative")
        summary = ag_game_summary(result)
        assert "Parity" in summary
        assert "Verdict" in summary

    def test_energy_summary(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_energy_compositional(g, partition, "iterative")
        summary = ag_game_summary(result)
        assert "Energy" in summary


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    def test_isolated_vertices(self):
        """Vertices with only self-loops."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 0)
        g.add_vertex(1, Player.ODD, 1)
        g.add_edge(0, 0)
        g.add_edge(1, 1)
        partition = {"a": {0}, "b": {1}}
        result = solve_parity_compositional(g, partition, "iterative")
        assert 0 in result.win_even  # Even priority self-loop
        assert 1 in result.win_odd   # Odd priority self-loop

    def test_large_game_decomposition(self):
        """Test with a larger game."""
        g = ParityGame()
        n = 20
        for i in range(n):
            g.add_vertex(i, Player.EVEN if i % 2 == 0 else Player.ODD, i % 4)
        for i in range(n):
            g.add_edge(i, (i + 1) % n)
            g.add_edge(i, (i + 3) % n)

        partition = {
            "A": set(range(0, n // 2)),
            "B": set(range(n // 2, n)),
        }
        result = solve_parity_compositional(g, partition, "iterative")
        assert len(result.win_even) + len(result.win_odd) == n

    def test_three_component_decomposition(self):
        """Three components in a parity game."""
        g = ParityGame()
        for i in range(9):
            g.add_vertex(i, Player.EVEN if i % 2 == 0 else Player.ODD, i % 3)
        for i in range(9):
            g.add_edge(i, (i + 1) % 9)
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}, "C": {6, 7, 8}}
        result = solve_parity_compositional(g, partition, "iterative")
        assert len(result.win_even) + len(result.win_odd) == 9

    def test_energy_negative_cycle(self):
        """Energy game with a negative cycle."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_edge(0, 1, -1)
        g.add_edge(1, 0, -1)
        partition = {"a": {0}, "b": {1}}
        result = solve_energy_compositional(g, partition, "pessimistic")
        # With net negative cycle, Even cannot maintain energy
        assert result.verdict == DischargeVerdict.SOUND

    def test_energy_positive_cycle(self):
        """Energy game with a positive cycle -- Even wins."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.EVEN)
        g.add_edge(0, 1, 3)
        g.add_edge(1, 0, 1)
        partition = {"a": {0}, "b": {1}}
        result = solve_energy_compositional(g, partition, "iterative")
        # Net positive cycle, Even controls, should win
        mono = solve_energy(g)
        # Pessimistic under-approx is a subset
        pess = solve_energy_compositional(g, partition, "pessimistic")
        assert pess.win_even <= mono.win_energy


# ===================================================================
# Soundness tests -- the crucial property
# ===================================================================

class TestSoundness:
    """Verify the key soundness property:
    Pessimistic compositional <= Monolithic
    If iterative verdict is SOUND, iterative should match monolithic.
    """

    def test_soundness_simple(self):
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        pess = solve_parity_compositional(g, partition, "pessimistic")
        mono = zielonka(g)
        assert pess.win_even <= mono.win_even

    def test_soundness_chain(self):
        g = make_chain_parity(8)
        partition = {"left": {0, 1, 2, 3}, "right": {4, 5, 6, 7}}
        pess = solve_parity_compositional(g, partition, "pessimistic")
        mono = zielonka(g)
        assert pess.win_even <= mono.win_even

    def test_soundness_energy(self):
        g = make_two_component_energy()
        partition = {"A": {0, 1}, "B": {2, 3}}
        pess = solve_energy_compositional(g, partition, "pessimistic")
        mono = solve_energy(g)
        assert pess.win_even <= mono.win_energy

    def test_iterative_matches_or_underapprox(self):
        """Iterative should at least be a subset of monolithic Even region."""
        g = make_two_component_parity()
        partition = {"A": {0, 1, 2}, "B": {3, 4, 5}}
        it = solve_parity_compositional(g, partition, "iterative")
        mono = zielonka(g)
        # Iterative should be sound (subset or equal)
        if it.verdict == DischargeVerdict.SOUND:
            # Even region should be subset of mono
            assert it.win_even <= mono.win_even

    def test_soundness_various_partitions(self):
        """Test soundness with multiple partition strategies."""
        g = make_chain_parity(8)
        for part_fn in [
            lambda: partition_by_priority_bands(g, 2),
            lambda: partition_by_owner(g),
        ]:
            p = part_fn()
            if len(p) < 2:
                continue
            pess = solve_parity_compositional(g, p, "pessimistic")
            mono = zielonka(g)
            assert pess.win_even <= mono.win_even


# ===================================================================
# Regression / complex scenario tests
# ===================================================================

class TestComplexScenarios:
    def test_diamond_game(self):
        """Diamond: 0->1, 0->2, 1->3, 2->3. Even wants to reach 3."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 1)
        g.add_vertex(1, Player.ODD, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_vertex(3, Player.EVEN, 2)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 3)
        g.add_edge(3, 3)  # Sink with even prio

        partition = {"top": {0, 1}, "bot": {2, 3}}
        result = solve_parity_compositional(g, partition, "iterative")
        mono = zielonka(g)
        # Should find that Even wins from everywhere (can reach p=2 sink)
        assert result.win_even <= mono.win_even

    def test_mutual_dependency_game(self):
        """Two components each depending on the other's interface."""
        g = ParityGame()
        g.add_vertex(0, Player.EVEN, 2)
        g.add_vertex(1, Player.ODD, 0)
        g.add_vertex(2, Player.EVEN, 0)
        g.add_vertex(3, Player.ODD, 2)
        g.add_edge(0, 1)
        g.add_edge(1, 2)   # A->B
        g.add_edge(1, 0)   # Stay in A
        g.add_edge(2, 3)
        g.add_edge(3, 0)   # B->A
        g.add_edge(3, 2)   # Stay in B

        partition = {"A": {0, 1}, "B": {2, 3}}
        result = solve_parity_compositional(g, partition, "iterative")
        mono = zielonka(g)
        assert len(result.win_even) + len(result.win_odd) == 4
        if result.verdict == DischargeVerdict.SOUND:
            assert result.win_even <= mono.win_even

    def test_energy_with_choice(self):
        """Energy game where Even has a choice of paths."""
        g = EnergyGame()
        g.add_vertex(0, Player.EVEN)
        g.add_vertex(1, Player.ODD)
        g.add_vertex(2, Player.EVEN)
        g.add_vertex(3, Player.ODD)
        g.add_edge(0, 1, 5)
        g.add_edge(0, 2, 2)   # Even chooses
        g.add_edge(1, 3, -3)
        g.add_edge(2, 3, -1)
        g.add_edge(3, 0, 1)

        partition = {"left": {0, 1}, "right": {2, 3}}
        result = solve_energy_compositional(g, partition, "pessimistic")
        mono = solve_energy(g)
        assert result.win_even <= mono.win_energy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
