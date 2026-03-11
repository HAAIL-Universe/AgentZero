"""Tests for V071: MDP Model Checking (PCTL for MDPs)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from mdp_pctl import (
    LabeledMDP, make_labeled_mdp, MDPPCTLChecker, MDPPCTLResult,
    Quantification,
    check_mdp_pctl, check_mdp_pctl_state, mdp_pctl_quantitative,
    verify_mdp_property, compare_quantifications, batch_check_mdp,
    induced_mc_comparison, mdp_expected_reward,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
from pctl_model_check import (
    tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until,
    eventually, always, bounded_eventually,
    parse_pctl, FormulaKind,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V069_mdp_verification'))
from mdp_verification import MDP, Policy, make_mdp, Objective


# ---------------------------------------------------------------------------
# Test fixtures: canonical MDPs
# ---------------------------------------------------------------------------

def simple_mdp():
    """Simple 3-state MDP: s0 has choice of 'left' or 'right'.
    s0 --left--> s1 (prob 0.6), s2 (prob 0.4)
    s0 --right--> s1 (prob 0.3), s2 (prob 0.7)
    s1 (absorbing, labeled 'good')
    s2 (absorbing, labeled 'bad')
    """
    return make_labeled_mdp(
        n_states=3,
        action_transitions={
            0: {'left': [0.0, 0.6, 0.4], 'right': [0.0, 0.3, 0.7]},
            1: {'stay': [0.0, 1.0, 0.0]},
            2: {'stay': [0.0, 0.0, 1.0]},
        },
        labels={0: set(), 1: {'good'}, 2: {'bad'}},
        state_labels=['s0', 's1', 's2'],
    )


def diamond_mdp():
    """4-state diamond MDP.
    s0 --a--> s1(0.8), s2(0.2)  |  s0 --b--> s1(0.3), s2(0.7)
    s1 --c--> s3(1.0)           |  s1 labeled 'mid'
    s2 --d--> s3(1.0)
    s3 (absorbing, labeled 'target')
    """
    return make_labeled_mdp(
        n_states=4,
        action_transitions={
            0: {'a': [0.0, 0.8, 0.2, 0.0], 'b': [0.0, 0.3, 0.7, 0.0]},
            1: {'c': [0.0, 0.0, 0.0, 1.0]},
            2: {'d': [0.0, 0.0, 0.0, 1.0]},
            3: {'stay': [0.0, 0.0, 0.0, 1.0]},
        },
        labels={0: set(), 1: {'mid'}, 2: set(), 3: {'target'}},
        state_labels=['s0', 's1', 's2', 's3'],
    )


def coin_flip_mdp():
    """MDP modeling a coin flip game.
    s0: choose 'fair' (0.5/0.5) or 'biased' (0.8/0.2) to win/lose.
    s1: 'win' (absorbing)
    s2: 'lose' (absorbing)
    """
    return make_labeled_mdp(
        n_states=3,
        action_transitions={
            0: {'fair': [0.0, 0.5, 0.5], 'biased': [0.0, 0.8, 0.2]},
            1: {'stay': [0.0, 1.0, 0.0]},
            2: {'stay': [0.0, 0.0, 1.0]},
        },
        labels={0: {'start'}, 1: {'win'}, 2: {'lose'}},
        state_labels=['start', 'win', 'lose'],
    )


def multi_step_mdp():
    """4-state MDP with multi-step decisions.
    s0 --safe--> s1(1.0)     |  s0 --risky--> s2(0.5), s3(0.5)
    s1 --go--> s3(1.0)       |  s1 labeled 'safe_mid'
    s2 (absorbing, 'fail')   |  s3 (absorbing, 'goal')
    """
    return make_labeled_mdp(
        n_states=4,
        action_transitions={
            0: {'safe': [0.0, 1.0, 0.0, 0.0], 'risky': [0.0, 0.0, 0.5, 0.5]},
            1: {'go': [0.0, 0.0, 0.0, 1.0]},
            2: {'stay': [0.0, 0.0, 1.0, 0.0]},
            3: {'stay': [0.0, 0.0, 0.0, 1.0]},
        },
        labels={0: set(), 1: {'safe_mid'}, 2: {'fail'}, 3: {'goal'}},
        state_labels=['s0', 's1', 's2', 's3'],
    )


def cycle_mdp():
    """MDP with cycles: s0 can go to s1 or stay, s1 goes to s0 or s2.
    s0 --try-->  s1(0.7), s0(0.3)  |  s0 --wait--> s0(1.0)
    s1 --push--> s2(0.9), s0(0.1)  |  s1 --back--> s0(1.0)
    s2 (absorbing, 'done')
    """
    return make_labeled_mdp(
        n_states=3,
        action_transitions={
            0: {'try': [0.3, 0.7, 0.0], 'wait': [1.0, 0.0, 0.0]},
            1: {'push': [0.1, 0.0, 0.9], 'back': [1.0, 0.0, 0.0]},
            2: {'stay': [0.0, 0.0, 1.0]},
        },
        labels={0: {'init'}, 1: {'trying'}, 2: {'done'}},
        state_labels=['s0', 's1', 's2'],
    )


# ===========================================================================
# Tests: Basic state formula checking
# ===========================================================================

class TestBasicFormulas:
    def test_true_all_states(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        assert checker.check(tt()) == {0, 1, 2}

    def test_false_no_states(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        assert checker.check(ff()) == set()

    def test_atom(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        assert checker.check(atom('good')) == {1}
        assert checker.check(atom('bad')) == {2}

    def test_not(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        assert checker.check(pnot(atom('good'))) == {0, 2}

    def test_and(self):
        lmdp = diamond_mdp()
        checker = MDPPCTLChecker(lmdp)
        # mid AND NOT target
        result = checker.check(pand(atom('mid'), pnot(atom('target'))))
        assert result == {1}

    def test_or(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        result = checker.check(por(atom('good'), atom('bad')))
        assert result == {1, 2}


# ===========================================================================
# Tests: Next-state probabilities
# ===========================================================================

class TestNextState:
    def test_next_max(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(next_f(atom('good')))
        # Max prob of reaching 'good' next: left gives 0.6, right gives 0.3
        assert abs(probs[0] - 0.6) < 1e-9
        assert abs(probs[1] - 1.0) < 1e-9  # already good
        assert abs(probs[2] - 0.0) < 1e-9  # bad, stays bad

    def test_next_min(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_min(next_f(atom('good')))
        # Min prob: right gives 0.3
        assert abs(probs[0] - 0.3) < 1e-9

    def test_next_prob_geq_universal(self):
        lmdp = simple_mdp()
        # P>=0.5[X "good"] under universal: need Pmin >= 0.5
        # Pmin = 0.3 < 0.5, so s0 does NOT satisfy
        result = check_mdp_pctl(lmdp, prob_geq(0.5, next_f(atom('good'))),
                                Quantification.UNIVERSAL)
        assert 0 not in result.satisfying_states
        assert 1 in result.satisfying_states  # prob = 1.0

    def test_next_prob_geq_existential(self):
        lmdp = simple_mdp()
        # P>=0.5[X "good"] under existential: need Pmax >= 0.5
        # Pmax = 0.6 >= 0.5, so s0 satisfies
        result = check_mdp_pctl(lmdp, prob_geq(0.5, next_f(atom('good'))),
                                Quantification.EXISTENTIAL)
        assert 0 in result.satisfying_states

    def test_next_coin_flip(self):
        lmdp = coin_flip_mdp()
        checker = MDPPCTLChecker(lmdp)
        # Max prob of winning next
        probs_max = checker.check_quantitative_max(next_f(atom('win')))
        assert abs(probs_max[0] - 0.8) < 1e-9  # biased coin

        probs_min = checker.check_quantitative_min(next_f(atom('win')))
        assert abs(probs_min[0] - 0.5) < 1e-9  # fair coin


# ===========================================================================
# Tests: Unbounded Until
# ===========================================================================

class TestUnboundedUntil:
    def test_until_simple_max(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        # P(true U "good") = probability of eventually reaching good
        probs = checker.check_quantitative_max(eventually(atom('good')))
        assert abs(probs[0] - 0.6) < 1e-9  # left action
        assert abs(probs[1] - 1.0) < 1e-9
        assert abs(probs[2] - 0.0) < 1e-9  # absorbed in bad

    def test_until_simple_min(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_min(eventually(atom('good')))
        assert abs(probs[0] - 0.3) < 1e-9  # right action

    def test_until_diamond_all_reach_target(self):
        lmdp = diamond_mdp()
        checker = MDPPCTLChecker(lmdp)
        # All paths eventually reach target (s3)
        probs_max = checker.check_quantitative_max(eventually(atom('target')))
        probs_min = checker.check_quantitative_min(eventually(atom('target')))
        for s in range(4):
            assert abs(probs_max[s] - 1.0) < 1e-6
            assert abs(probs_min[s] - 1.0) < 1e-6

    def test_until_multi_step_max(self):
        lmdp = multi_step_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(eventually(atom('goal')))
        # Max: safe path gives 1.0
        assert abs(probs[0] - 1.0) < 1e-6
        assert abs(probs[1] - 1.0) < 1e-6  # s1 goes directly to goal

    def test_until_multi_step_min(self):
        lmdp = multi_step_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_min(eventually(atom('goal')))
        # Min: risky path gives 0.5
        assert abs(probs[0] - 0.5) < 1e-6

    def test_until_cycle_max(self):
        lmdp = cycle_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(eventually(atom('done')))
        # With try+push: s0->s1 (0.7), s1->s2 (0.9). Cycle back with (0.3 + 0.7*0.1).
        # Eventually reaches done with prob 1.0 (can always try again)
        assert abs(probs[0] - 1.0) < 1e-4
        assert abs(probs[1] - 1.0) < 1e-4

    def test_until_cycle_min(self):
        lmdp = cycle_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_min(eventually(atom('done')))
        # Minimizer: s0 can 'wait' forever (self-loop), prob = 0
        assert abs(probs[0] - 0.0) < 1e-6
        # s1: minimizer can 'back' to s0 which then waits forever -> prob = 0
        assert abs(probs[1] - 0.0) < 1e-6

    def test_until_with_phi_constraint(self):
        lmdp = multi_step_mdp()
        checker = MDPPCTLChecker(lmdp)
        # "safe_mid" U "goal": must go through safe_mid
        # Only the safe path goes through s1 (safe_mid), risky skips it
        probs_max = checker.check_quantitative_max(
            until(por(atom('safe_mid'), tt()), atom('goal'))
        )
        # tt() U goal = eventually goal, so both paths work
        assert abs(probs_max[0] - 1.0) < 1e-6

    def test_prob_geq_until(self):
        lmdp = simple_mdp()
        # P>=0.5[F "good"] existential: Pmax=0.6 >= 0.5 -> s0 satisfies
        result = check_mdp_pctl(lmdp, prob_geq(0.5, eventually(atom('good'))),
                                Quantification.EXISTENTIAL)
        assert 0 in result.satisfying_states

    def test_prob_leq_until(self):
        lmdp = simple_mdp()
        # P<=0.5[F "good"] universal: Pmax=0.6 > 0.5 -> s0 does NOT satisfy
        result = check_mdp_pctl(lmdp, prob_leq(0.5, eventually(atom('good'))),
                                Quantification.UNIVERSAL)
        assert 0 not in result.satisfying_states


# ===========================================================================
# Tests: Bounded Until
# ===========================================================================

class TestBoundedUntil:
    def test_bounded_until_1_step(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs_max = checker.check_quantitative_max(
            bounded_eventually(atom('good'), 1)
        )
        # 1 step: same as next
        assert abs(probs_max[0] - 0.6) < 1e-9

    def test_bounded_until_multi_step(self):
        lmdp = multi_step_mdp()
        checker = MDPPCTLChecker(lmdp)
        # Bounded 1 step: risky can reach goal in 1 step (prob 0.5)
        # safe goes to s1 in 1 step, not goal
        probs_max_1 = checker.check_quantitative_max(
            bounded_eventually(atom('goal'), 1)
        )
        assert abs(probs_max_1[0] - 0.5) < 1e-9  # risky: direct to goal

        # Bounded 2 steps: safe can reach goal in 2 (s0->s1->s3)
        probs_max_2 = checker.check_quantitative_max(
            bounded_eventually(atom('goal'), 2)
        )
        assert abs(probs_max_2[0] - 1.0) < 1e-9  # safe: 100% in 2 steps

    def test_bounded_until_min(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs_min = checker.check_quantitative_min(
            bounded_eventually(atom('good'), 1)
        )
        assert abs(probs_min[0] - 0.3) < 1e-9  # right: 0.3

    def test_bounded_until_convergence(self):
        """Bounded until converges to unbounded as k increases."""
        lmdp = cycle_mdp()
        checker = MDPPCTLChecker(lmdp)
        unbounded = checker.check_quantitative_max(eventually(atom('done')))

        prev_prob = 0.0
        for k in [1, 5, 10, 50, 100]:
            bounded = checker.check_quantitative_max(
                bounded_eventually(atom('done'), k)
            )
            assert bounded[0] >= prev_prob - 1e-9  # monotonically increasing
            prev_prob = bounded[0]

        # After enough steps, bounded should be close to unbounded
        bounded_100 = checker.check_quantitative_max(
            bounded_eventually(atom('done'), 100)
        )
        assert abs(bounded_100[0] - unbounded[0]) < 0.01


# ===========================================================================
# Tests: Policy extraction
# ===========================================================================

class TestPolicyExtraction:
    def test_max_policy_simple(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        policy = checker.extract_policy(eventually(atom('good')), maximize=True)
        # s0 should choose action 0 (left, prob 0.6)
        assert policy.get_action(0) == 0  # 'left'

    def test_min_policy_simple(self):
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        policy = checker.extract_policy(eventually(atom('good')), maximize=False)
        # s0 should choose action 1 (right, prob 0.3)
        assert policy.get_action(0) == 1  # 'right'

    def test_max_policy_multi_step(self):
        lmdp = multi_step_mdp()
        checker = MDPPCTLChecker(lmdp)
        policy = checker.extract_policy(eventually(atom('goal')), maximize=True)
        # s0 should choose safe (action 0) for guaranteed goal
        assert policy.get_action(0) == 0  # 'safe'

    def test_min_policy_multi_step(self):
        lmdp = multi_step_mdp()
        checker = MDPPCTLChecker(lmdp)
        policy = checker.extract_policy(eventually(atom('goal')), maximize=False)
        # s0 should choose risky (action 1) for minimum prob 0.5
        assert policy.get_action(0) == 1  # 'risky'

    def test_policy_result_included(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, prob_geq(0.5, eventually(atom('good'))),
                                Quantification.EXISTENTIAL)
        assert result.policy_max is not None
        assert result.policy_min is not None


# ===========================================================================
# Tests: High-level API
# ===========================================================================

class TestHighLevelAPI:
    def test_check_mdp_pctl_basic(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, prob_geq(0.25, next_f(atom('good'))),
                                Quantification.UNIVERSAL)
        # Pmin(X good) at s0 = 0.3 >= 0.25, so s0 satisfies
        assert 0 in result.satisfying_states

    def test_check_state(self):
        lmdp = simple_mdp()
        assert check_mdp_pctl_state(lmdp, 0,
                                     prob_geq(0.25, next_f(atom('good'))),
                                     Quantification.UNIVERSAL)
        assert not check_mdp_pctl_state(lmdp, 0,
                                         prob_geq(0.5, next_f(atom('good'))),
                                         Quantification.UNIVERSAL)

    def test_quantitative(self):
        lmdp = coin_flip_mdp()
        result = mdp_pctl_quantitative(lmdp, next_f(atom('win')))
        assert abs(result['max'][0] - 0.8) < 1e-9
        assert abs(result['min'][0] - 0.5) < 1e-9

    def test_verify_property(self):
        lmdp = multi_step_mdp()
        result = verify_mdp_property(lmdp,
                                      prob_geq(1.0, eventually(atom('goal'))),
                                      initial_state=0,
                                      quantification=Quantification.EXISTENTIAL)
        assert result['holds']
        assert abs(result['max_at_initial'] - 1.0) < 1e-6

    def test_verify_property_fails(self):
        lmdp = multi_step_mdp()
        result = verify_mdp_property(lmdp,
                                      prob_geq(1.0, eventually(atom('goal'))),
                                      initial_state=0,
                                      quantification=Quantification.UNIVERSAL)
        # Universal: min prob = 0.5 < 1.0
        assert not result['holds']

    def test_summary(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, prob_geq(0.5, next_f(atom('good'))),
                                Quantification.EXISTENTIAL)
        s = result.summary()
        assert 'Formula' in s
        assert 'Quantification' in s

    def test_all_satisfy_property(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, tt())
        assert result.all_satisfy

    def test_none_satisfy_property(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, ff())
        assert result.none_satisfy


# ===========================================================================
# Tests: Compare quantifications
# ===========================================================================

class TestCompareQuantifications:
    def test_universal_subset_existential(self):
        lmdp = simple_mdp()
        result = compare_quantifications(lmdp,
                                          prob_geq(0.5, next_f(atom('good'))))
        # Universal is always a subset of existential for P>=
        assert result['universal_subset_of_existential']

    def test_different_results(self):
        lmdp = simple_mdp()
        result = compare_quantifications(lmdp,
                                          prob_geq(0.5, next_f(atom('good'))))
        # Universal: Pmin=0.3 < 0.5, so s0 not in universal
        assert 0 not in result['universal_sat']
        # Existential: Pmax=0.6 >= 0.5, so s0 in existential
        assert 0 in result['existential_sat']

    def test_agree_when_deterministic(self):
        """When only one action, universal = existential."""
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {'go': [0.0, 1.0]},
                1: {'stay': [0.0, 1.0]},
            },
            labels={0: set(), 1: {'goal'}},
        )
        result = compare_quantifications(lmdp,
                                          prob_geq(1.0, next_f(atom('goal'))))
        assert set(result['universal_sat']) == set(result['existential_sat'])


# ===========================================================================
# Tests: Batch check
# ===========================================================================

class TestBatchCheck:
    def test_batch_check(self):
        lmdp = simple_mdp()
        formulas = [
            prob_geq(0.3, next_f(atom('good'))),
            prob_leq(0.5, next_f(atom('good'))),
            tt(),
        ]
        results = batch_check_mdp(lmdp, formulas, Quantification.UNIVERSAL)
        assert len(results) == 3
        # First: Pmin=0.3 >= 0.3, s0 satisfies
        assert 0 in results[0].satisfying_states
        # Second: Pmax=0.6 <= 0.5? No. s0 does NOT satisfy
        assert 0 not in results[1].satisfying_states


# ===========================================================================
# Tests: Induced MC comparison
# ===========================================================================

class TestInducedMCComparison:
    def test_induced_mc(self):
        lmdp = simple_mdp()
        result = induced_mc_comparison(lmdp,
                                        prob_geq(0.5, eventually(atom('good'))))
        assert 'mc_max_policy_sat' in result
        assert 'mc_min_policy_sat' in result
        assert 'mdp_prob_max' in result

    def test_mc_max_probs_match(self):
        lmdp = simple_mdp()
        result = induced_mc_comparison(lmdp,
                                        prob_geq(0.5, eventually(atom('good'))))
        # Under max policy (left), MC prob should match MDP Pmax
        mdp_max = result['mdp_prob_max']
        mc_max = result['mc_max_probs']
        for s in range(3):
            assert abs(mdp_max[s] - mc_max[s]) < 1e-6


# ===========================================================================
# Tests: Expected reward
# ===========================================================================

class TestExpectedReward:
    def test_expected_reward_max(self):
        lmdp = multi_step_mdp()
        rewards = [1.0, 1.0, 0.0, 0.0]  # reward per step
        values, policy = mdp_expected_reward(lmdp, rewards, atom('goal'),
                                              maximize=True)
        # Max reward: safe path gives 2.0 (1 for s0 + 1 for s1)
        assert abs(values[0] - 2.0) < 1e-6
        assert abs(values[1] - 1.0) < 1e-6

    def test_expected_reward_min(self):
        lmdp = multi_step_mdp()
        rewards = [1.0, 1.0, 0.0, 0.0]
        values, policy = mdp_expected_reward(lmdp, rewards, atom('goal'),
                                              maximize=False)
        # Min reward: risky path, 0.5 prob reach goal in 1 step (reward 1),
        # 0.5 prob stuck in fail (reward 0). Expected = 1 * 0.5 = 0.5? No...
        # Actually if fail is absorbing and can't reach goal, value is inf
        # for that path. But minimize avoids inf. Let's see:
        # risky: reward 1 + 0.5 * 0 (goal) + 0.5 * inf (fail) = inf
        # safe: reward 1 + 1 * (1 + 0) = 2
        # Min chooses safe (2.0) since risky leads to inf
        # Wait: minimize means minimize expected reward.
        # If fail can't reach goal, value[2] = inf for maximize.
        # For minimize, unreachable = 0 (not inf). Let me check the code...
        # In the code: `float('inf') if maximize else 0.0`
        # So for min: fail state has value 0. risky gives 1 + 0.5*0 + 0.5*0 = 1.0
        # safe gives 1 + 1*1 = 2.0. Min chooses risky (1.0)
        assert abs(values[0] - 1.0) < 1e-6

    def test_expected_reward_target_zero(self):
        lmdp = simple_mdp()
        rewards = [1.0, 0.0, 0.0]
        values, _ = mdp_expected_reward(lmdp, rewards, atom('good'),
                                         maximize=True)
        # s0 reaches good in 1 step, reward = 1.0
        assert abs(values[0] - 1.0) < 1e-6
        assert abs(values[1] - 0.0) < 1e-6  # already at target


# ===========================================================================
# Tests: Complex formulas
# ===========================================================================

class TestComplexFormulas:
    def test_nested_prob(self):
        """P>=0.5[X P>=0.8[F "done"]] -- nested probability operators."""
        lmdp = cycle_mdp()
        # Inner: P>=0.8[F "done"] -- which states satisfy this under universal?
        # Pmin(F done) at s0 = 0 (wait forever), at s1 = 0 (back forever)
        # So under universal, no state satisfies P>=0.8[F done] except s2
        # Under existential: Pmax(F done) at s0 = 1.0, s1 = 1.0
        # s0 and s1 satisfy under existential
        inner = prob_geq(0.8, eventually(atom('done')))
        outer = prob_geq(0.5, next_f(inner))

        # Under existential for both
        result = check_mdp_pctl(lmdp, outer, Quantification.EXISTENTIAL)
        # s2 already done, always satisfies inner.
        # Pmax(X inner_sat) at s0: best action 'try', prob to s1.
        # Under existential, inner_sat = {0, 1, 2} (Pmax(F done) >= 0.8 for all)
        # So Pmax(X inner_sat) = max action prob to {0,1,2} = 1.0
        assert 0 in result.satisfying_states

    def test_and_prob(self):
        lmdp = coin_flip_mdp()
        # P>=0.5[X "win"] AND P<=0.3[X "lose"]
        formula = pand(
            prob_geq(0.5, next_f(atom('win'))),
            prob_leq(0.3, next_f(atom('lose')))
        )
        result = check_mdp_pctl(lmdp, formula, Quantification.EXISTENTIAL)
        # Pmax(X win) = 0.8 >= 0.5? Yes
        # Pmin(X lose) = 0.2 <= 0.3? Yes
        # So s0 satisfies both under existential
        assert 0 in result.satisfying_states

    def test_prob_gt(self):
        lmdp = simple_mdp()
        # P>0.5[X "good"] existential: Pmax=0.6 > 0.5, s0 satisfies
        result = check_mdp_pctl(lmdp, prob_gt(0.5, next_f(atom('good'))),
                                Quantification.EXISTENTIAL)
        assert 0 in result.satisfying_states

    def test_prob_lt(self):
        lmdp = simple_mdp()
        # P<0.5[X "good"] existential: Pmin=0.3 < 0.5, s0 satisfies
        result = check_mdp_pctl(lmdp, prob_lt(0.5, next_f(atom('good'))),
                                Quantification.EXISTENTIAL)
        assert 0 in result.satisfying_states


# ===========================================================================
# Tests: Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_single_state(self):
        lmdp = make_labeled_mdp(
            n_states=1,
            action_transitions={0: {'stay': [1.0]}},
            labels={0: {'here'}},
        )
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(next_f(atom('here')))
        assert abs(probs[0] - 1.0) < 1e-9

    def test_all_absorbing(self):
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {'stay': [1.0, 0.0]},
                1: {'stay': [0.0, 1.0]},
            },
            labels={0: {'a'}, 1: {'b'}},
        )
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(eventually(atom('b')))
        assert abs(probs[0] - 0.0) < 1e-9
        assert abs(probs[1] - 1.0) < 1e-9

    def test_deterministic_mdp(self):
        """MDP with single action per state = Markov chain."""
        lmdp = make_labeled_mdp(
            n_states=3,
            action_transitions={
                0: {'go': [0.0, 0.7, 0.3]},
                1: {'stay': [0.0, 1.0, 0.0]},
                2: {'stay': [0.0, 0.0, 1.0]},
            },
            labels={1: {'target'}},
        )
        checker = MDPPCTLChecker(lmdp)
        probs_max = checker.check_quantitative_max(eventually(atom('target')))
        probs_min = checker.check_quantitative_min(eventually(atom('target')))
        # No choice -> max = min
        assert abs(probs_max[0] - probs_min[0]) < 1e-9
        assert abs(probs_max[0] - 0.7) < 1e-9

    def test_three_actions(self):
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {
                    'a': [0.0, 1.0],
                    'b': [0.5, 0.5],
                    'c': [0.9, 0.1],
                },
                1: {'stay': [0.0, 1.0]},
            },
            labels={1: {'goal'}},
        )
        checker = MDPPCTLChecker(lmdp)
        probs_max = checker.check_quantitative_max(next_f(atom('goal')))
        probs_min = checker.check_quantitative_min(next_f(atom('goal')))
        assert abs(probs_max[0] - 1.0) < 1e-9  # action 'a'
        assert abs(probs_min[0] - 0.1) < 1e-9  # action 'c'

    def test_prob_geq_zero_always_true(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, prob_geq(0.0, next_f(atom('good'))),
                                Quantification.UNIVERSAL)
        # Pmin >= 0 is always true
        assert result.all_satisfy

    def test_prob_leq_one_always_true(self):
        lmdp = simple_mdp()
        result = check_mdp_pctl(lmdp, prob_leq(1.0, next_f(atom('good'))),
                                Quantification.UNIVERSAL)
        assert result.all_satisfy


# ===========================================================================
# Tests: Larger MDP (5 states)
# ===========================================================================

class TestLargerMDP:
    def make_grid_mdp(self):
        """5-state grid: s0->s1/s2, s1->s3, s2->s3/s4, s3='win', s4='lose'."""
        return make_labeled_mdp(
            n_states=5,
            action_transitions={
                0: {
                    'up': [0.0, 0.9, 0.1, 0.0, 0.0],
                    'down': [0.0, 0.2, 0.8, 0.0, 0.0],
                },
                1: {'go': [0.0, 0.0, 0.0, 1.0, 0.0]},
                2: {
                    'try': [0.0, 0.0, 0.0, 0.6, 0.4],
                    'safe': [0.0, 0.0, 0.0, 0.3, 0.7],
                },
                3: {'stay': [0.0, 0.0, 0.0, 1.0, 0.0]},
                4: {'stay': [0.0, 0.0, 0.0, 0.0, 1.0]},
            },
            labels={3: {'win'}, 4: {'lose'}},
            state_labels=['s0', 's1', 's2', 's3', 's4'],
        )

    def test_grid_reach_max(self):
        lmdp = self.make_grid_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(eventually(atom('win')))
        # Max strategy: s0 goes up (0.9 to s1), s2 tries (0.6 to s3)
        # Pmax(s0) = 0.9 * 1.0 + 0.1 * 0.6 = 0.96
        assert abs(probs[0] - 0.96) < 1e-6

    def test_grid_reach_min(self):
        lmdp = self.make_grid_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_min(eventually(atom('win')))
        # Min strategy: s0 goes down (0.2 to s1, 0.8 to s2), s2 safe (0.3 to s3)
        # Pmin(s0) = 0.2 * 1.0 + 0.8 * 0.3 = 0.44
        assert abs(probs[0] - 0.44) < 1e-6

    def test_grid_bounded(self):
        lmdp = self.make_grid_mdp()
        checker = MDPPCTLChecker(lmdp)
        # Bounded 1: s0 can't reach win in 1 step (goes to s1 or s2 first)
        probs_1 = checker.check_quantitative_max(
            bounded_eventually(atom('win'), 1)
        )
        assert abs(probs_1[0] - 0.0) < 1e-9

        # Bounded 2: s0->s1->s3 or s0->s2->s3
        probs_2 = checker.check_quantitative_max(
            bounded_eventually(atom('win'), 2)
        )
        assert abs(probs_2[0] - 0.96) < 1e-6  # same as unbounded

    def test_grid_policy(self):
        lmdp = self.make_grid_mdp()
        checker = MDPPCTLChecker(lmdp)
        policy = checker.extract_policy(eventually(atom('win')), maximize=True)
        assert policy.get_action(0) == 0  # 'up'
        assert policy.get_action(2) == 0  # 'try'


# ===========================================================================
# Tests: Parse + check integration
# ===========================================================================

class TestParseIntegration:
    def test_parse_and_check(self):
        lmdp = simple_mdp()
        formula = parse_pctl('P>=0.25[X "good"]')
        result = check_mdp_pctl(lmdp, formula, Quantification.UNIVERSAL)
        assert 0 in result.satisfying_states

    def test_parse_eventually(self):
        lmdp = simple_mdp()
        formula = parse_pctl('P>=0.5[F "good"]')
        result = check_mdp_pctl(lmdp, formula, Quantification.EXISTENTIAL)
        assert 0 in result.satisfying_states


# ===========================================================================
# Tests: MDP-specific semantics
# ===========================================================================

class TestMDPSemantics:
    def test_min_max_gap(self):
        """The gap between Pmin and Pmax should be non-negative."""
        lmdp = simple_mdp()
        result = mdp_pctl_quantitative(lmdp, next_f(atom('good')))
        for s in range(3):
            assert result['max'][s] >= result['min'][s] - 1e-9

    def test_min_max_gap_until(self):
        lmdp = cycle_mdp()
        result = mdp_pctl_quantitative(lmdp, eventually(atom('done')))
        for s in range(3):
            assert result['max'][s] >= result['min'][s] - 1e-9

    def test_absorbing_states_prob_one(self):
        """Absorbing target states have prob 1 for eventually reaching themselves."""
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(eventually(atom('good')))
        assert abs(probs[1] - 1.0) < 1e-9  # good state reaches itself

    def test_unreachable_target(self):
        """States that can't reach target have prob 0."""
        lmdp = simple_mdp()
        checker = MDPPCTLChecker(lmdp)
        probs = checker.check_quantitative_max(eventually(atom('good')))
        assert abs(probs[2] - 0.0) < 1e-9  # bad can't reach good

    def test_prob_one_under_all_actions(self):
        """If all actions lead to target, both min and max are 1."""
        lmdp = make_labeled_mdp(
            n_states=2,
            action_transitions={
                0: {
                    'a': [0.3, 0.7],
                    'b': [0.1, 0.9],
                },
                1: {'stay': [0.0, 1.0]},
            },
            labels={1: {'target'}},
        )
        checker = MDPPCTLChecker(lmdp)
        probs_max = checker.check_quantitative_max(eventually(atom('target')))
        probs_min = checker.check_quantitative_min(eventually(atom('target')))
        assert abs(probs_max[0] - 1.0) < 1e-6
        assert abs(probs_min[0] - 1.0) < 1e-6
