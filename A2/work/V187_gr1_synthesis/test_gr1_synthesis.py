"""Tests for V187: GR(1) Synthesis."""

import pytest
from gr1_synthesis import (
    GR1Game, GR1Verdict, GR1Result, GR1Strategy, BoolGR1Spec,
    gr1_solve, gr1_synthesize, build_bool_game,
    make_game, make_safety_game, make_reachability_game, make_response_game,
    sys_attractor, env_attractor, verify_strategy, game_statistics,
    gr1_summary, strategy_to_mealy, MealyMachine,
    _all_valuations,
)


# ===========================================================================
# Helper: build simple games
# ===========================================================================

def linear_game(n):
    """Linear chain: 0 -> 1 -> ... -> n-1 -> 0. Single env choice, single sys choice."""
    states = set(range(n))
    transitions = {i: [{(i + 1) % n}] for i in range(n)}
    return states, transitions


def two_choice_game():
    """States {0,1,2}. From each state, env picks left/right, sys has one successor each.
    0 -L-> 1, 0 -R-> 2
    1 -L-> 0, 1 -R-> 1
    2 -L-> 2, 2 -R-> 0
    """
    states = {0, 1, 2}
    transitions = {
        0: [{1}, {2}],      # env choice 0 -> {1}, env choice 1 -> {2}
        1: [{0}, {1}],
        2: [{2}, {0}],
    }
    return states, transitions


# ===========================================================================
# Test: Controllable predecessor
# ===========================================================================

class TestCpre:
    def test_cpre_all_successors_in_target(self):
        game = make_game([0, 1, 2], [0], {0: [{1, 2}], 1: [{0}], 2: [{0}]})
        # From 0, single env choice with {1,2}. Both in target {1,2} -> 0 in cpre
        assert 0 in game.cpre({1, 2})

    def test_cpre_not_all_env_covered(self):
        game = make_game([0, 1, 2], [0], {0: [{1}, {2}], 1: [{0}], 2: [{0}]})
        # From 0, env choice 0 -> {1}, env choice 1 -> {2}
        # target {1}: env choice 0 covered, env choice 1 not -> 0 not in cpre
        assert 0 not in game.cpre({1})

    def test_cpre_both_env_choices_covered(self):
        game = make_game([0, 1, 2], [0], {0: [{1}, {2}], 1: [{0}], 2: [{0}]})
        assert 0 in game.cpre({1, 2})

    def test_cpre_empty_target(self):
        game = make_game([0, 1], [0], {0: [{1}], 1: [{0}]})
        assert game.cpre(set()) == set()

    def test_cpre_dead_end(self):
        game = make_game([0, 1], [0], {0: [{1}], 1: []})
        # State 1 has no env choices (dead end) -> not in cpre
        assert 1 not in game.cpre({0, 1})

    def test_cpre_self_loop(self):
        game = make_game([0], [0], {0: [{0}]})
        assert game.cpre({0}) == {0}

    def test_upre_env_forces(self):
        game = make_game([0, 1, 2], [0], {0: [{1}, {2}], 1: [{0}], 2: [{0}]})
        # Env can force reaching {1} by picking env choice 0 (if sys has no choice)
        assert 0 in game.upre({1})

    def test_apre_exists_path(self):
        game = make_game([0, 1, 2], [0], {0: [{1}, {2}], 1: [{0}], 2: [{0}]})
        assert 0 in game.apre({1})
        assert 0 in game.apre({2})


# ===========================================================================
# Test: Simple reachability/safety
# ===========================================================================

class TestSimpleGames:
    def test_single_state_realizable(self):
        game = make_game([0], [0], {0: [{0}]}, sys_justice=[{0}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert 0 in result.winning_region

    def test_single_state_unrealizable(self):
        # sys_justice = {}, guarantee requires visiting empty set -> unrealizable
        game = make_game([0], [0], {0: [{0}]}, sys_justice=[set()])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_no_guarantees(self):
        game = make_game([0, 1], [0], {0: [{1}], 1: [{0}]})
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_linear_cycle_single_guarantee(self):
        states, trans = linear_game(3)
        # Guarantee: visit state 0. Cycle 0->1->2->0 naturally visits 0.
        game = make_game(states, [0], trans, sys_justice=[{0}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert result.winning_region == states

    def test_linear_cycle_multiple_guarantees(self):
        states, trans = linear_game(4)
        # Must visit both state 0 and state 2
        game = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert result.winning_region == states

    def test_unreachable_guarantee(self):
        # States 0,1 form cycle. State 2 is disconnected. Guarantee: visit 2.
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1}], 1: [{0}], 2: [{2}]},
            sys_justice=[{2}],
        )
        result = gr1_solve(game)
        # From initial state 0, can't reach 2
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_safety_game(self):
        states, trans = linear_game(3)
        game = make_safety_game(states, [0], trans, bad_states={2})
        result = gr1_solve(game)
        # Must avoid state 2, but cycle forces 0->1->2. From 0 and 1, can't avoid 2.
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_safety_game_avoidable(self):
        # 0 -> {1,2}, 1 -> {0}, 2 -> {2}. Sys can avoid 2 by always choosing 1.
        game = make_safety_game(
            [0, 1, 2], [0],
            {0: [{1, 2}], 1: [{0}], 2: [{2}]},
            bad_states={2},
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert 0 in result.winning_region

    def test_reachability_game(self):
        states, trans = linear_game(3)
        game = make_reachability_game(states, [0], trans, target={2})
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Environment choices (adversarial)
# ===========================================================================

class TestAdversarialEnv:
    def test_env_can_prevent_guarantee(self):
        # From 0, env picks {1} or {2}. From 1, only -> 0. From 2, only -> 0.
        # Guarantee: visit 1. Env can always pick {2} to avoid 1.
        # But wait: from 0, env picks which set of successors. Sys picks within the set.
        # If transitions = {0: [{1}, {2}]}, env picks to go to 1 or 2.
        # Guarantee: visit {1}. Env always picks {2} -> never reaches 1.
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1}, {2}], 1: [{0}], 2: [{0}]},
            sys_justice=[{1}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_sys_has_choice_within_env(self):
        # From 0, env picks {1,2} (single env choice). Sys picks 1 or 2.
        # Guarantee: visit 1. Sys can always pick 1.
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1, 2}], 1: [{0}], 2: [{0}]},
            sys_justice=[{1}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_env_two_choices_sys_can_still_win(self):
        # 0: env picks {1,3} or {2,3}. 1->0, 2->0, 3->0.
        # Guarantee: visit {1,2}. Regardless of env, sys can pick 1 or 2 (both available).
        game = make_game(
            [0, 1, 2, 3], [0],
            {0: [{1, 3}, {2, 3}], 1: [{0}], 2: [{0}], 3: [{0}]},
            sys_justice=[{1, 2}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_env_blocks_one_path_sys_uses_other(self):
        # 0: env picks {1} or {2}. 1->{0,3}. 2->{0,3}. 3->{0}.
        # Guarantee: visit 3. Sys can reach 3 from either 1 or 2.
        game = make_game(
            [0, 1, 2, 3], [0],
            {0: [{1}, {2}], 1: [{0, 3}], 2: [{0, 3}], 3: [{0}]},
            sys_justice=[{3}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Assumptions (env justice)
# ===========================================================================

class TestAssumptions:
    def test_assumption_makes_unrealizable_realizable(self):
        # 0: env picks {1} or {2}. 1->0, 2->0.
        # Guarantee: visit 1. Without assumption, env avoids 1 -> unrealizable.
        # With assumption GF({1,2}): env must visit 1 or 2 inf often.
        # Hmm, this doesn't help directly. Let me redesign.
        #
        # Better: 0: env picks {1} or {2}. 1->0, 2->0.
        # Guarantee: visit 1. Assumption: env visits 1 (picks {1}) inf often.
        # With assumption, env must sometimes pick {1}, so sys visits 1.
        # Wait, but env picks {1} means state goes to 1 (sys has no choice).
        # Assumption: GF({1}) = env must cause state 1 inf often.
        # But env controls which successor from 0... the assumption says
        # "state 1 is visited inf often" which means env picks {1} from 0 inf often.
        # Under this assumption, sys trivially visits 1.
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1}, {2}], 1: [{0}], 2: [{0}]},
            env_justice=[{1}],  # Assumption: visit state 1 inf often
            sys_justice=[{1}],  # Guarantee: visit state 1 inf often
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_assumption_env_visits_trigger(self):
        # Arbiter pattern: env requests, sys must eventually grant.
        # States: (no_req, req, grant). Env controls request.
        # 0(idle): env picks {0} (stay) or {1} (request)
        # 1(req): env picks {1} (keep requesting). Sys can go to 2 (grant).
        # 2(grant): env picks {0} (back to idle)
        # Assumption: GF(req) -- env keeps requesting
        # Guarantee: GF(grant) -- sys must grant
        game = make_game(
            [0, 1, 2], [0],
            {
                0: [{0}, {1}],       # env: stay idle or request
                1: [{1, 2}],         # env: stay in req, sys can grant
                2: [{0}],            # after grant, back to idle
            },
            env_justice=[{1}],  # Assumption: GF(request)
            sys_justice=[{2}],  # Guarantee: GF(grant)
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_without_assumption_fails(self):
        # Same as above but without assumption: env can stay in 0 forever
        game = make_game(
            [0, 1, 2], [0],
            {
                0: [{0}, {1}],
                1: [{1, 2}],
                2: [{0}],
            },
            sys_justice=[{2}],  # Must visit grant, but env may never request
        )
        result = gr1_solve(game)
        # Env can stay at 0 forever, never entering 1, so sys can't reach 2
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_multiple_assumptions(self):
        # Two assumptions, one guarantee
        game = make_game(
            [0, 1, 2, 3], [0],
            {
                0: [{1}, {2}],       # env picks direction
                1: [{0}, {3}],       # sys can go to 3
                2: [{0}, {3}],       # sys can go to 3
                3: [{0}],            # back to start
            },
            env_justice=[{1}, {2}],  # env must visit both 1 and 2
            sys_justice=[{3}],       # sys must visit 3
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_assumption_spoiling(self):
        # Sys can win by preventing env from satisfying assumption
        # 0: env picks {1} or {2}. 1->{0,1}. 2->{0}.
        # Guarantee: GF({2}). Assumption: GF({1}).
        # From 0, env picks {2} -> visits 2. Or env picks {1} -> sys can stay in 1 forever.
        # If sys stays in 1 forever: env assumption GF({1}) is satisfied,
        # but sys guarantee GF({2}) fails.
        # So sys must visit 2 when possible.
        # Actually from 1, sys picks {0} or {1}. If sys picks {0}, back to 0,
        # and env might pick {2} next. If sys picks {1}, stays in 1.
        # Hmm, this is realizable: from 1, sys goes to 0, from 0 env picks {2}.
        # Under GF({1}) assumption, env picks {1} inf often from 0.
        # Then sys must go back to 0 and hope env picks {2}. But env may not.
        # Without assumption on {2}, env can avoid 2 by always picking {1}.
        # Let's make it so env must visit 2: assumption GF({2}).
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1}, {2}], 1: [{0, 1}], 2: [{0}]},
            env_justice=[{2}],  # Env must visit 2
            sys_justice=[{2}],  # Sys must visit 2
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Strategy extraction and verification
# ===========================================================================

class TestStrategy:
    def test_strategy_exists_for_realizable(self):
        states, trans = linear_game(3)
        game = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        result = gr1_solve(game)
        assert result.strategy is not None
        assert result.strategy.initial_state == 0

    def test_strategy_simulation(self):
        states, trans = linear_game(4)
        game = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        result = gr1_solve(game)
        # Simulate: each state has 1 env choice (idx 0)
        trace = result.strategy.simulate(game, [0] * 12)
        assert len(trace) == 13  # initial + 12 steps
        # Check we cycle through states
        visited_states = {s for s, m in trace}
        assert visited_states == {0, 1, 2, 3}

    def test_strategy_modes_advance(self):
        states, trans = linear_game(3)
        game = make_game(states, [0], trans, sys_justice=[{0}, {1}])
        result = gr1_solve(game)
        trace = result.strategy.simulate(game, [0] * 10)
        # Modes should cycle between 0 and 1
        modes = [m for s, m in trace]
        assert 0 in modes
        assert 1 in modes

    def test_strategy_no_strategy_for_unrealizable(self):
        game = make_game([0], [0], {0: [{0}]}, sys_justice=[set()])
        result = gr1_solve(game)
        assert result.strategy is None

    def test_verify_strategy_valid(self):
        states, trans = linear_game(3)
        game = make_game(states, [0], trans, sys_justice=[{0}])
        result = gr1_solve(game)
        valid, msg = verify_strategy(game, result.strategy)
        assert valid, msg

    def test_verify_strategy_with_env_choices(self):
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1, 2}], 1: [{0}], 2: [{0}]},
            sys_justice=[{1}],
        )
        result = gr1_solve(game)
        valid, msg = verify_strategy(game, result.strategy)
        assert valid, msg


# ===========================================================================
# Test: Boolean variable games
# ===========================================================================

class TestBoolGame:
    def test_all_valuations_empty(self):
        vals = _all_valuations([])
        assert vals == [frozenset()]

    def test_all_valuations_one_var(self):
        vals = _all_valuations(['a'])
        assert len(vals) == 2
        assert frozenset() in vals
        assert frozenset({'a'}) in vals

    def test_all_valuations_two_vars(self):
        vals = _all_valuations(['a', 'b'])
        assert len(vals) == 4

    def test_build_simple_bool_game(self):
        spec = BoolGR1Spec(
            env_vars=['r'],
            sys_vars=['g'],
            sys_justice=[lambda s: 'g' in s],  # Must grant inf often
        )
        game = build_bool_game(spec)
        assert len(game.states) == 4  # 2^2
        assert len(game.sys_justice) == 1

    def test_bool_arbiter(self):
        """Boolean arbiter: env requests (r), sys grants (g).
        Assumption: GF(r). Guarantee: GF(g).
        Sys can always set g=true, so this is trivially realizable.
        """
        spec = BoolGR1Spec(
            env_vars=['r'],
            sys_vars=['g'],
            env_justice=[lambda s: 'r' in s],
            sys_justice=[lambda s: 'g' in s],
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_bool_mutex(self):
        """Two grants, must alternate. Safety: not both true. Liveness: both visited."""
        spec = BoolGR1Spec(
            env_vars=['r1', 'r2'],
            sys_vars=['g1', 'g2'],
            sys_init=lambda s: 'g1' not in s and 'g2' not in s,  # Start with no grants
            sys_trans=lambda s, ns: not ('g1' in ns and 'g2' in ns),  # Mutex
            env_justice=[lambda s: 'r1' in s, lambda s: 'r2' in s],  # Both request
            sys_justice=[lambda s: 'g1' in s, lambda s: 'g2' in s],  # Both get granted
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_bool_impossible_mutex(self):
        """Must grant both simultaneously -- contradicts mutex constraint."""
        spec = BoolGR1Spec(
            env_vars=[],
            sys_vars=['g1', 'g2'],
            sys_trans=lambda s, ns: not ('g1' in ns and 'g2' in ns),  # Mutex
            sys_justice=[lambda s: 'g1' in s and 'g2' in s],  # Must have both (impossible!)
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_bool_with_env_init(self):
        spec = BoolGR1Spec(
            env_vars=['e'],
            sys_vars=['s'],
            env_init=lambda s: 'e' not in s,  # Env starts false
            sys_init=lambda s: 's' not in s,  # Sys starts false
            sys_justice=[lambda s: 's' in s],  # Must eventually set s
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Attractor computation
# ===========================================================================

class TestAttractor:
    def test_sys_attractor_direct(self):
        game = make_game([0, 1, 2], [0], {0: [{1}], 1: [{2}], 2: [{0}]})
        attr = sys_attractor(game, {2})
        assert attr == {0, 1, 2}  # All states can reach 2

    def test_sys_attractor_env_blocks(self):
        game = make_game([0, 1, 2], [0], {0: [{1}, {2}], 1: [{1}], 2: [{0}]})
        attr = sys_attractor(game, {2})
        # From 0, env can pick {1} (self-loop at 1) -> can't guarantee reaching 2
        # Only state 2 and states where sys can force reaching 2
        # 0: env picks {1} (stuck at 1) or {2} (reaches target). Not all env covered.
        assert 0 not in attr
        assert 2 in attr

    def test_env_attractor(self):
        game = make_game([0, 1, 2], [0], {0: [{1}, {2}], 1: [{0}], 2: [{0}]})
        attr = env_attractor(game, {1})
        # Env can force reaching 1 from 0 (pick {1})
        assert 0 in attr
        assert 1 in attr


# ===========================================================================
# Test: Multiple guarantees (cycling behavior)
# ===========================================================================

class TestMultipleGuarantees:
    def test_two_guarantees_cycle(self):
        # Must visit both A and B regions
        game = make_game(
            [0, 1, 2, 3], [0],
            {0: [{1}], 1: [{2}], 2: [{3}], 3: [{0}]},
            sys_justice=[{0}, {2}],  # Visit 0 and 2
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert result.n_guarantees == 2

    def test_three_guarantees_cycle(self):
        states, trans = linear_game(6)
        game = make_game(states, [0], trans,
                         sys_justice=[{0}, {2}, {4}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert result.n_guarantees == 3

    def test_incompatible_guarantees(self):
        # Two absorbing states, must visit both -> impossible
        game = make_game(
            [0, 1], [0],
            {0: [{0}], 1: [{1}]},  # Absorbing states
            sys_justice=[{0}, {1}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_guarantees_order_independent(self):
        states, trans = linear_game(4)
        result1 = gr1_solve(make_game(states, [0], trans, sys_justice=[{0}, {2}]))
        result2 = gr1_solve(make_game(states, [0], trans, sys_justice=[{2}, {0}]))
        assert result1.verdict == result2.verdict
        assert result1.winning_region == result2.winning_region

    def test_guarantee_subset_of_states(self):
        # Guarantee set contains multiple states
        states, trans = linear_game(4)
        game = make_game(states, [0], trans, sys_justice=[{0, 1}, {2, 3}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Complex game scenarios
# ===========================================================================

class TestComplexGames:
    def test_traffic_light(self):
        """Traffic light controller:
        States: (ns_phase, ew_phase) where phase in {red, green}
        Env controls car arrival, sys controls lights.
        Safety: not both green. Liveness: both get green.
        """
        # Simplified: states are light configurations
        # 0: both red, 1: ns green, 2: ew green
        game = make_game(
            [0, 1, 2], [0],
            {
                0: [{0, 1, 2}],  # From both-red, sys picks any
                1: [{0, 1}],     # From ns-green, sys can stay or go to both-red
                2: [{0, 2}],     # From ew-green, sys can stay or go to both-red
            },
            sys_justice=[{1}, {2}],  # Both directions must get green
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        # Strategy: cycle 0->1->0->2->0->1->...

    def test_buffer_controller(self):
        """Buffer: env fills, sys empties. Must keep buffer from overflowing.
        States: buffer level 0,1,2,3 (3 = overflow = bad).
        Env: +1 or +0. Sys: -1 or -0.
        """
        states = {0, 1, 2, 3}
        transitions = {}
        for level in states:
            env_choices = []
            # Env choice 0: don't add
            sys_opts_no_add = set()
            sys_opts_no_add.add(max(0, level - 1))  # Sys drains
            sys_opts_no_add.add(level)               # Sys doesn't drain
            env_choices.append(sys_opts_no_add)
            # Env choice 1: add one
            new_level = min(3, level + 1)
            sys_opts_add = set()
            sys_opts_add.add(max(0, new_level - 1))  # Sys drains
            sys_opts_add.add(new_level)               # Sys doesn't drain
            env_choices.append(sys_opts_add)
            transitions[level] = env_choices

        game = make_game(states, [0], transitions,
                         sys_justice=[{0, 1, 2}])  # Stay non-overflow
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_producer_consumer(self):
        """Producer (env) produces items, consumer (sys) consumes.
        States: queue size 0-3.
        Assumption: env produces inf often.
        Guarantee: sys consumes inf often (queue reaches 0).
        """
        states = {0, 1, 2, 3}
        transitions = {}
        for q in states:
            env_choices = []
            # Env: produce (q+1 capped at 3)
            produced = min(3, q + 1)
            sys_opts_p = {produced, max(0, produced - 1)}
            env_choices.append(sys_opts_p)
            # Env: don't produce
            sys_opts_np = {q, max(0, q - 1)}
            env_choices.append(sys_opts_np)
            transitions[q] = env_choices

        game = make_game(states, [0], transitions,
                         env_justice=[{1, 2, 3}],  # Env produces (queue > 0)
                         sys_justice=[{0}])         # Queue empties
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_elevator(self):
        """Simple elevator: 3 floors (0,1,2). Env presses button, sys moves.
        Must visit all requested floors.
        """
        # States: (floor, requested_floor) -- simplified to just floor
        states = {0, 1, 2}
        transitions = {
            0: [{0, 1}],     # From floor 0, sys can stay or go up
            1: [{0, 1, 2}],  # From floor 1, sys can go anywhere
            2: [{1, 2}],     # From floor 2, sys can stay or go down
        }
        game = make_game(states, [0], transitions,
                         sys_justice=[{0}, {1}, {2}])  # Must visit all floors
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_game(self):
        game = make_game([], [], {})
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE  # Vacuously

    def test_no_initial_states(self):
        game = make_game([0, 1], [], {0: [{1}], 1: [{0}]}, sys_justice=[{0}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.UNREALIZABLE  # No initial state in winning

    def test_winning_region_not_including_initial(self):
        # Initial at 0, winning only from 1
        game = make_game(
            [0, 1, 2], [0],
            {0: [{2}], 1: [{1}], 2: [{2}]},
            sys_justice=[{1}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.UNREALIZABLE
        # But state 1 should be in winning region (self-loop satisfying guarantee)
        assert 1 in result.winning_region

    def test_all_states_initial(self):
        game = make_game([0, 1], [0, 1], {0: [{1}], 1: [{0}]}, sys_justice=[{0}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_large_env_branching(self):
        # State 0 with 10 env choices, each with single successor
        states = set(range(11))
        transitions = {0: [{i} for i in range(1, 11)]}
        for i in range(1, 11):
            transitions[i] = [{0}]
        game = make_game(states, [0], transitions, sys_justice=[{5}])
        result = gr1_solve(game)
        # Env can avoid 5 by picking any other successor
        assert result.verdict == GR1Verdict.UNREALIZABLE

    def test_large_sys_branching(self):
        # State 0 with 1 env choice and 10 sys successors
        states = set(range(11))
        transitions = {0: [{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}]}
        for i in range(1, 11):
            transitions[i] = [{0}]
        game = make_game(states, [0], transitions, sys_justice=[{5}])
        result = gr1_solve(game)
        # Sys can pick 5
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_self_loop_guarantee(self):
        game = make_game([0], [0], {0: [{0}]}, sys_justice=[{0}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_disconnected_components(self):
        # Two disconnected cycles
        game = make_game(
            [0, 1, 2, 3], [0],
            {0: [{1}], 1: [{0}], 2: [{3}], 3: [{2}]},
            sys_justice=[{0}, {2}],
        )
        result = gr1_solve(game)
        # Can't visit both 0 and 2 from initial state 0
        assert result.verdict == GR1Verdict.UNREALIZABLE


# ===========================================================================
# Test: Statistics and summary
# ===========================================================================

class TestStatistics:
    def test_game_statistics(self):
        states, trans = linear_game(5)
        game = make_game(states, [0], trans, sys_justice=[{0}])
        stats = game_statistics(game)
        assert stats['states'] == 5
        assert stats['initial'] == 1
        assert stats['env_justice'] == 0
        assert stats['sys_justice'] == 1
        assert stats['dead_ends'] == 0

    def test_statistics_with_dead_ends(self):
        game = make_game([0, 1], [0], {0: [{1}], 1: []})
        stats = game_statistics(game)
        assert stats['dead_ends'] == 1

    def test_gr1_summary(self):
        states, trans = linear_game(3)
        game = make_game(states, [0], trans, sys_justice=[{0}])
        result = gr1_solve(game)
        summary = gr1_summary(game, result)
        assert "REALIZABLE" in summary.upper()
        assert "3" in summary  # 3 states

    def test_result_fields(self):
        states, trans = linear_game(3)
        game = make_game(states, [0], trans,
                         env_justice=[{1}], sys_justice=[{0}, {2}])
        result = gr1_solve(game)
        assert result.n_states == 3
        assert result.n_guarantees == 2
        assert result.n_assumptions == 1
        assert result.iterations > 0


# ===========================================================================
# Test: Mealy machine conversion
# ===========================================================================

class TestMealyConversion:
    def test_mealy_from_bool_game(self):
        spec = BoolGR1Spec(
            env_vars=['r'],
            sys_vars=['g'],
            env_justice=[lambda s: 'r' in s],
            sys_justice=[lambda s: 'g' in s],
        )
        game = build_bool_game(spec)
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

        mealy = strategy_to_mealy(game, result.strategy,
                                  env_vars=['r'], sys_vars=['g'])
        assert mealy is not None
        assert len(mealy.states) > 0

    def test_mealy_simulation(self):
        mealy = MealyMachine(
            states={0, 1},
            initial=0,
            inputs={'r'},
            outputs={'g'},
            transitions={
                (0, frozenset()): (0, frozenset()),
                (0, frozenset({'r'})): (1, frozenset({'g'})),
                (1, frozenset()): (0, frozenset()),
                (1, frozenset({'r'})): (1, frozenset({'g'})),
            },
        )
        trace = mealy.simulate([frozenset({'r'}), frozenset(), frozenset({'r'})])
        assert len(trace) == 3


# ===========================================================================
# Test: Response game builder
# ===========================================================================

class TestResponseGame:
    def test_single_response(self):
        # trigger: state 1, response: state 2
        game = make_response_game(
            [0, 1, 2], [0],
            {0: [{1, 2}], 1: [{0, 2}], 2: [{0}]},
            triggers_responses=[({1}, {2})],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_multiple_responses(self):
        game = make_response_game(
            [0, 1, 2, 3], [0],
            {0: [{1, 2, 3}], 1: [{0}], 2: [{0}], 3: [{0}]},
            triggers_responses=[({1}, {2}), ({1}, {3})],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Fixpoint convergence properties
# ===========================================================================

class TestFixpointProperties:
    def test_winning_region_is_fixpoint(self):
        """Winning region should be a fixpoint of the operator."""
        states, trans = linear_game(4)
        game = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        result = gr1_solve(game)
        W = result.winning_region

        # For each guarantee, cpre(W) should cover W
        # (simplified check: every winning state has a path to each guarantee within W)
        for j in range(2):
            # States in guarantee
            gj = game.sys_justice[j] & W
            # All winning states should be able to reach gj via cpre iterations
            reachable = set(gj)
            for _ in range(len(W)):
                reachable = reachable | (game.cpre(reachable) & W)
            assert W <= reachable, f"Not all winning states can reach guarantee {j}"

    def test_winning_region_monotone(self):
        """Adding assumptions should only increase (or maintain) winning region."""
        states, trans = linear_game(4)
        game_no_assume = make_game(states, [0], trans, sys_justice=[{0}])
        game_with_assume = make_game(states, [0], trans,
                                     env_justice=[{2}], sys_justice=[{0}])
        r1 = gr1_solve(game_no_assume)
        r2 = gr1_solve(game_with_assume)
        assert r1.winning_region <= r2.winning_region

    def test_removing_guarantee_increases_winning(self):
        """Fewer guarantees -> larger or equal winning region."""
        states, trans = linear_game(4)
        game_2g = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        game_1g = make_game(states, [0], trans, sys_justice=[{0}])
        r2 = gr1_solve(game_2g)
        r1 = gr1_solve(game_1g)
        assert r2.winning_region <= r1.winning_region


# ===========================================================================
# Test: Classical GR(1) examples
# ===========================================================================

class TestClassicalExamples:
    def test_simple_arbiter_2_clients(self):
        """Two clients request access to shared resource.
        Env: r1, r2 (requests). Sys: g1, g2 (grants).
        Safety: not (g1 and g2).
        Assumptions: GF(r1), GF(r2).
        Guarantees: GF(g1), GF(g2).
        """
        spec = BoolGR1Spec(
            env_vars=['r1', 'r2'],
            sys_vars=['g1', 'g2'],
            sys_trans=lambda s, ns: not ('g1' in ns and 'g2' in ns),
            env_justice=[lambda s: 'r1' in s, lambda s: 'r2' in s],
            sys_justice=[lambda s: 'g1' in s, lambda s: 'g2' in s],
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_robot_grid(self):
        """Robot on 2x2 grid. Must visit all corners.
        States: (0,0), (0,1), (1,0), (1,1).
        Sys moves: adjacent cells. No env interference.
        """
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        states = set(positions)
        transitions = {}
        for x, y in positions:
            neighbors = set()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in states:
                    neighbors.add((nx, ny))
            transitions[(x, y)] = [neighbors]  # Single env choice

        game = make_game(states, [(0, 0)], transitions,
                         sys_justice=[{(0, 0)}, {(0, 1)}, {(1, 0)}, {(1, 1)}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_robot_grid_with_obstacle(self):
        """Robot must visit (0,0) and (1,1) but (0,1) is blocked."""
        positions = [(0, 0), (1, 0), (1, 1)]
        states = set(positions)
        transitions = {}
        for x, y in positions:
            neighbors = set()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in states:
                    neighbors.add((nx, ny))
            transitions[(x, y)] = [neighbors]

        game = make_game(states, [(0, 0)], transitions,
                         sys_justice=[{(0, 0)}, {(1, 1)}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_turn_based_game(self):
        """Turn-based game: env and sys alternate moves on a graph.
        Even states: sys moves. Odd states: env moves.
        """
        # 0(sys) -> {1,3}, 1(env) -> {0,2}, 2(sys) -> {1,3}, 3(env) -> {0,2}
        # Guarantee: visit state 0 (sys state)
        game = make_game(
            [0, 1, 2, 3], [0],
            {
                0: [{1, 3}],       # Sys picks (1 env choice)
                1: [{0}, {2}],     # Env picks (2 choices, 1 successor each)
                2: [{1, 3}],       # Sys picks
                3: [{0}, {2}],     # Env picks
            },
            sys_justice=[{0}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_philosophers_lite(self):
        """Simplified dining philosophers with 2 philosophers.
        Each philosopher needs 2 forks. Only 2 forks total.
        States: which philosopher holds which forks.
        """
        # States: (p1_forks, p2_forks) where each is 0, 1, or 2
        # Total forks <= 2
        states = set()
        for p1 in range(3):
            for p2 in range(3):
                if p1 + p2 <= 2:
                    states.add((p1, p2))

        transitions = {}
        for s in states:
            p1, p2 = s
            # Env chooses who tries to pick up (or release)
            choices = []
            # Env choice: p1 acts
            p1_options = set()
            if p1 < 2 and p1 + p2 < 2:  # p1 picks up (if fork available)
                p1_options.add((p1 + 1, p2))
            if p1 > 0:  # p1 releases
                p1_options.add((p1 - 1, p2))
            p1_options.add(s)  # No change
            choices.append(p1_options)
            # Env choice: p2 acts
            p2_options = set()
            if p2 < 2 and p1 + p2 < 2:
                p2_options.add((p1, p2 + 1))
            if p2 > 0:
                p2_options.add((p1, p2 - 1))
            p2_options.add(s)
            choices.append(p2_options)
            transitions[s] = choices

        # Both philosophers must eat (hold 2 forks) infinitely often
        game = make_game(states, [(0, 0)], transitions,
                         sys_justice=[{(2, 0)}, {(0, 2)}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Deterministic games (no env choice)
# ===========================================================================

class TestDeterministicGames:
    def test_deterministic_cycle(self):
        """No env choice: sys fully controls transitions."""
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1, 2}], 1: [{0, 2}], 2: [{0, 1}]},
            sys_justice=[{0}, {1}, {2}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE

    def test_deterministic_stuck(self):
        """Sys has no choice, stuck in loop that misses guarantee."""
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1}], 1: [{0}], 2: [{2}]},
            sys_justice=[{2}],
        )
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.UNREALIZABLE


# ===========================================================================
# Test: Performance (moderate-sized games)
# ===========================================================================

class TestPerformance:
    def test_grid_10x10(self):
        """10x10 grid with 4-connectivity. Must visit corners."""
        states = set()
        for x in range(10):
            for y in range(10):
                states.add((x, y))

        transitions = {}
        for x, y in states:
            neighbors = set()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 10 and 0 <= ny < 10:
                    neighbors.add((nx, ny))
            transitions[(x, y)] = [neighbors]

        game = make_game(states, [(0, 0)], transitions,
                         sys_justice=[{(0, 0)}, {(9, 9)}])
        result = gr1_solve(game)
        assert result.verdict == GR1Verdict.REALIZABLE
        assert len(result.winning_region) == 100  # All states winning

    def test_bool_game_3_vars(self):
        """3 env vars + 3 sys vars = 64 states."""
        spec = BoolGR1Spec(
            env_vars=['e1', 'e2', 'e3'],
            sys_vars=['s1', 's2', 's3'],
            env_justice=[lambda s: 'e1' in s],
            sys_justice=[lambda s: 's1' in s, lambda s: 's2' in s],
        )
        result = gr1_synthesize(spec)
        assert result.verdict == GR1Verdict.REALIZABLE


# ===========================================================================
# Test: Strategy simulation scenarios
# ===========================================================================

class TestStrategySimulation:
    def test_strategy_visits_all_guarantees(self):
        """Simulate strategy and check all guarantees are visited."""
        states, trans = linear_game(4)
        game = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        result = gr1_solve(game)
        trace = result.strategy.simulate(game, [0] * 20)

        visited_in_g0 = any(s in game.sys_justice[0] for s, m in trace)
        visited_in_g1 = any(s in game.sys_justice[1] for s, m in trace)
        assert visited_in_g0
        assert visited_in_g1

    def test_strategy_against_adversarial_env(self):
        """Strategy should work regardless of env choices."""
        game = make_game(
            [0, 1, 2], [0],
            {0: [{1, 2}], 1: [{0}], 2: [{0}]},
            sys_justice=[{1}],
        )
        result = gr1_solve(game)
        # Env always picks choice 0 (which has {1,2}), sys picks 1
        trace = result.strategy.simulate(game, [0] * 10)
        visited_1 = any(s == 1 for s, m in trace)
        assert visited_1

    def test_empty_env_choices_list(self):
        """Strategy handles states with no env choices gracefully."""
        strategy = GR1Strategy(n_modes=1, transitions={}, initial_state=None)
        trace = strategy.simulate(None, [0, 0, 0])
        assert trace == []


# ===========================================================================
# Test: Compositional patterns
# ===========================================================================

class TestCompositional:
    def test_solve_then_refine(self):
        """Solve a game, then add constraints and resolve."""
        states, trans = linear_game(4)
        game1 = make_game(states, [0], trans, sys_justice=[{0}])
        r1 = gr1_solve(game1)

        # Add second guarantee
        game2 = make_game(states, [0], trans, sys_justice=[{0}, {2}])
        r2 = gr1_solve(game2)

        assert r1.verdict == GR1Verdict.REALIZABLE
        assert r2.verdict == GR1Verdict.REALIZABLE
        assert r2.winning_region <= r1.winning_region

    def test_assumption_weakening(self):
        """Weaker assumptions should give smaller or equal winning region."""
        game_strong = make_game(
            [0, 1, 2], [0],
            {0: [{1}, {2}], 1: [{0}], 2: [{0}]},
            env_justice=[{1}, {2}],  # Strong: env visits both
            sys_justice=[{1}],
        )
        game_weak = make_game(
            [0, 1, 2], [0],
            {0: [{1}, {2}], 1: [{0}], 2: [{0}]},
            env_justice=[{1}],  # Weak: env visits 1 only
            sys_justice=[{1}],
        )
        r_strong = gr1_solve(game_strong)
        r_weak = gr1_solve(game_weak)
        # Both should be realizable since assumption covers guarantee
        assert r_strong.verdict == GR1Verdict.REALIZABLE
        assert r_weak.verdict == GR1Verdict.REALIZABLE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
