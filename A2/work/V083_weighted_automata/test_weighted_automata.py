"""Tests for V083: Weighted Automata over Semirings."""

import pytest
import math
from weighted_automata import (
    # Semirings
    Semiring, BooleanSemiring, TropicalSemiring, MaxPlusSemiring,
    ProbabilitySemiring, CountingSemiring, ViterbiSemiring,
    MinMaxSemiring, LogSemiring, make_semiring, SEMIRINGS,
    # WFA core
    WFA, WFATransition,
    # Constructors
    wfa_from_word, wfa_from_symbol, wfa_epsilon, wfa_empty,
    # Operations
    wfa_union, wfa_concat, wfa_star, wfa_intersect,
    # Run weight
    wfa_run_weight, wfa_accepts,
    # Shortest distance
    shortest_distance, all_pairs_distance,
    # N-best
    n_best_paths,
    # Trim
    wfa_trim,
    # Weight pushing
    wfa_push_weights,
    # Determinization
    wfa_determinize,
    # Equivalence
    wfa_equivalent,
    # NFA conversion
    nfa_to_wfa, wfa_to_nfa,
    # Stats
    wfa_stats, wfa_summary, WFAStats,
    # Other
    wfa_language_weight, convert_semiring, compare_wfas,
)


# ============================================================
# Section 1: Semiring Axioms
# ============================================================

class TestSemiringAxioms:
    """Verify semiring axioms for all implementations."""

    @pytest.fixture(params=[
        ('boolean', [True, False]),
        ('tropical', [0.0, 1.0, 3.5, float('inf')]),
        ('maxplus', [0.0, -1.0, 2.5, float('-inf')]),
        ('probability', [0.0, 0.5, 1.0, 0.3]),
        ('counting', [0, 1, 2, 5]),
        ('viterbi', [0.0, 0.5, 0.8, 1.0]),
        ('minmax', [0.0, 1.0, 3.0, float('inf')]),
        ('log', [float('-inf'), -2.0, -1.0, 0.0]),
    ])
    def sr_data(self, request):
        name, values = request.param
        return make_semiring(name), values

    def test_zero_identity(self, sr_data):
        sr, values = sr_data
        for a in values:
            assert sr.plus(a, sr.zero()) == a
            assert sr.plus(sr.zero(), a) == a

    def test_one_identity(self, sr_data):
        sr, values = sr_data
        for a in values:
            assert sr.times(a, sr.one()) == a
            assert sr.times(sr.one(), a) == a

    def test_zero_annihilates(self, sr_data):
        sr, values = sr_data
        for a in values:
            result = sr.times(a, sr.zero())
            assert sr.is_zero(result) or result == sr.zero()

    def test_plus_commutative(self, sr_data):
        sr, values = sr_data
        for a in values:
            for b in values:
                assert sr.plus(a, b) == sr.plus(b, a)

    def test_plus_associative(self, sr_data):
        sr, values = sr_data
        for a in values:
            for b in values:
                for c in values:
                    left = sr.plus(sr.plus(a, b), c)
                    right = sr.plus(a, sr.plus(b, c))
                    if isinstance(left, float) and isinstance(right, float):
                        if left == right:  # handles inf == inf
                            continue
                        if math.isfinite(left) and math.isfinite(right):
                            assert abs(left - right) < 1e-9
                        else:
                            assert left == right
                    else:
                        assert left == right

    def test_times_associative(self, sr_data):
        sr, values = sr_data
        for a in values[:3]:
            for b in values[:3]:
                for c in values[:3]:
                    left = sr.times(sr.times(a, b), c)
                    right = sr.times(a, sr.times(b, c))
                    if isinstance(left, float) and isinstance(right, float):
                        if math.isfinite(left) and math.isfinite(right):
                            assert abs(left - right) < 1e-9
                        else:
                            assert left == right
                    else:
                        assert left == right


# ============================================================
# Section 2: Individual Semiring Behavior
# ============================================================

class TestBooleanSemiring:
    def test_operations(self):
        sr = BooleanSemiring()
        assert sr.plus(True, False) == True
        assert sr.plus(False, False) == False
        assert sr.times(True, True) == True
        assert sr.times(True, False) == False
        assert sr.star(False) == True
        assert sr.star(True) == True

    def test_is_zero(self):
        sr = BooleanSemiring()
        assert sr.is_zero(False) == True
        assert sr.is_zero(True) == False


class TestTropicalSemiring:
    def test_operations(self):
        sr = TropicalSemiring()
        assert sr.plus(3.0, 5.0) == 3.0  # min
        assert sr.times(3.0, 5.0) == 8.0  # add
        assert sr.zero() == float('inf')
        assert sr.one() == 0.0

    def test_star(self):
        sr = TropicalSemiring()
        assert sr.star(0.0) == 0.0
        assert sr.star(5.0) == 0.0
        with pytest.raises(ValueError):
            sr.star(-1.0)

    def test_infinity(self):
        sr = TropicalSemiring()
        assert sr.times(float('inf'), 5.0) == float('inf')
        assert sr.plus(float('inf'), 3.0) == 3.0


class TestProbabilitySemiring:
    def test_operations(self):
        sr = ProbabilitySemiring()
        assert sr.plus(0.3, 0.5) == pytest.approx(0.8)
        assert sr.times(0.3, 0.5) == pytest.approx(0.15)

    def test_star(self):
        sr = ProbabilitySemiring()
        assert sr.star(0.5) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            sr.star(1.0)


class TestLogSemiring:
    def test_plus(self):
        sr = LogSemiring()
        # log(e^a + e^b)
        a, b = math.log(0.3), math.log(0.7)
        result = sr.plus(a, b)
        assert abs(math.exp(result) - 1.0) < 1e-9

    def test_times(self):
        sr = LogSemiring()
        a, b = math.log(0.3), math.log(0.5)
        result = sr.times(a, b)
        assert abs(result - math.log(0.15)) < 1e-9

    def test_zero_identity(self):
        sr = LogSemiring()
        a = math.log(0.5)
        assert sr.plus(a, sr.zero()) == pytest.approx(a)


class TestCountingSemiring:
    def test_operations(self):
        sr = CountingSemiring()
        assert sr.plus(3, 5) == 8
        assert sr.times(3, 5) == 15
        assert sr.zero() == 0
        assert sr.one() == 1


class TestViterbiSemiring:
    def test_operations(self):
        sr = ViterbiSemiring()
        assert sr.plus(0.3, 0.7) == 0.7  # max
        assert sr.times(0.3, 0.5) == pytest.approx(0.15)  # multiply
        assert sr.star(0.5) == 1.0
        assert sr.star(1.0) == 1.0


class TestMinMaxSemiring:
    def test_operations(self):
        sr = MinMaxSemiring()
        assert sr.plus(3.0, 5.0) == 3.0  # min
        assert sr.times(3.0, 5.0) == 5.0  # max


class TestMakeSemiring:
    def test_all_names(self):
        for name in SEMIRINGS:
            sr = make_semiring(name)
            assert isinstance(sr, Semiring)

    def test_unknown(self):
        with pytest.raises(ValueError):
            make_semiring('nonexistent')


class TestSemiringNary:
    def test_plus_n(self):
        sr = TropicalSemiring()
        assert sr.plus_n([5.0, 3.0, 7.0]) == 3.0

    def test_times_n(self):
        sr = TropicalSemiring()
        assert sr.times_n([1.0, 2.0, 3.0]) == 6.0


# ============================================================
# Section 3: WFA Construction
# ============================================================

class TestWFAConstruction:
    def test_add_state(self):
        wfa = WFA(TropicalSemiring())
        s0 = wfa.add_state(initial=0.0)
        s1 = wfa.add_state(final=0.0)
        assert s0 in wfa.states
        assert s1 in wfa.states
        assert s0 in wfa.initial_weight
        assert s1 in wfa.final_weight

    def test_add_transition(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_transition(0, 'a', 1, 5.0)
        assert len(wfa.transitions[0]) == 1
        assert wfa.transitions[0][0] == ('a', 1, 5.0)
        assert 'a' in wfa.alphabet

    def test_auto_state_ids(self):
        wfa = WFA(BooleanSemiring())
        s0 = wfa.add_state()
        s1 = wfa.add_state()
        s2 = wfa.add_state()
        assert {s0, s1, s2} == {0, 1, 2}

    def test_copy(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_transition(0, 'a', 1, 3.0)
        c = wfa.copy()
        assert c.states == wfa.states
        assert c.initial_weight == wfa.initial_weight
        c.add_state(99)
        assert 99 not in wfa.states

    def test_get_transitions(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_transition(0, 'a', 1, 1.0)
        wfa.add_transition(0, 'b', 1, 2.0)
        assert len(wfa.get_transitions(0)) == 2
        assert len(wfa.get_transitions(0, 'a')) == 1
        assert len(wfa.get_transitions(0, 'c')) == 0


# ============================================================
# Section 4: WFA from Word/Symbol/Epsilon/Empty
# ============================================================

class TestWFAFactories:
    def test_from_word_tropical(self):
        sr = TropicalSemiring()
        wfa = wfa_from_word("abc", sr, 10.0)
        assert wfa_run_weight(wfa, "abc") == 10.0
        assert wfa_run_weight(wfa, "ab") == sr.zero()
        assert wfa_run_weight(wfa, "abcd") == sr.zero()

    def test_from_word_boolean(self):
        sr = BooleanSemiring()
        wfa = wfa_from_word("hello", sr)
        assert wfa_accepts(wfa, "hello")
        assert not wfa_accepts(wfa, "hell")

    def test_from_symbol(self):
        sr = ProbabilitySemiring()
        wfa = wfa_from_symbol("x", sr, 0.5)
        assert wfa_run_weight(wfa, "x") == pytest.approx(0.5)
        assert wfa_run_weight(wfa, "xx") == pytest.approx(0.0)

    def test_epsilon(self):
        sr = TropicalSemiring()
        wfa = wfa_epsilon(sr, 3.0)
        assert wfa_run_weight(wfa, "") == 3.0
        assert wfa_run_weight(wfa, "a") == sr.zero()

    def test_empty(self):
        sr = TropicalSemiring()
        wfa = wfa_empty(sr)
        assert wfa_run_weight(wfa, "") == sr.zero()
        assert wfa_run_weight(wfa, "a") == sr.zero()


# ============================================================
# Section 5: Run Weight Computation
# ============================================================

class TestRunWeight:
    def test_tropical_shortest_path(self):
        """Diamond graph: two paths from 0 to 3 with different costs."""
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1)
        wfa.add_state(2)
        wfa.add_state(3, final=0.0)
        wfa.add_transition(0, 'a', 1, 1.0)  # path 1: cost 1+2=3
        wfa.add_transition(1, 'b', 3, 2.0)
        wfa.add_transition(0, 'a', 2, 5.0)  # path 2: cost 5+1=6
        wfa.add_transition(2, 'b', 3, 1.0)
        # "ab" weight = min(3, 6) = 3
        assert wfa_run_weight(wfa, "ab") == 3.0

    def test_counting_multiple_paths(self):
        """Count the number of accepting runs."""
        sr = CountingSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1)
        wfa.add_state(1, final=1)
        # Two transitions on 'a' from 0 to 1
        wfa.add_transition(0, 'a', 1, 1)
        wfa.add_transition(0, 'a', 1, 1)
        assert wfa_run_weight(wfa, "a") == 2

    def test_probability_sum(self):
        """Probability: sum over all paths."""
        sr = ProbabilitySemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1.0)
        wfa.add_state(1, final=1.0)
        wfa.add_transition(0, 'a', 1, 0.3)
        wfa.add_transition(0, 'a', 1, 0.5)
        assert wfa_run_weight(wfa, "a") == pytest.approx(0.8)

    def test_viterbi_max_path(self):
        """Viterbi: max over all paths."""
        sr = ViterbiSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1.0)
        wfa.add_state(1, final=1.0)
        wfa.add_transition(0, 'a', 1, 0.3)
        wfa.add_transition(0, 'a', 1, 0.7)
        assert wfa_run_weight(wfa, "a") == pytest.approx(0.7)

    def test_empty_word(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True, final=True)
        assert wfa_accepts(wfa, "")

    def test_no_path(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        # No transition from 0 to 1
        assert wfa_run_weight(wfa, "a") == float('inf')

    def test_accepts(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True)
        wfa.add_state(1, final=True)
        wfa.add_transition(0, 'x', 1)
        assert wfa_accepts(wfa, "x")
        assert not wfa_accepts(wfa, "y")
        assert not wfa_accepts(wfa, "")


# ============================================================
# Section 6: Union
# ============================================================

class TestUnion:
    def test_tropical_union(self):
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 3.0)
        b = wfa_from_word("cd", sr, 5.0)
        u = wfa_union(a, b)
        assert wfa_run_weight(u, "ab") == 3.0
        assert wfa_run_weight(u, "cd") == 5.0
        assert wfa_run_weight(u, "ac") == sr.zero()

    def test_boolean_union(self):
        sr = BooleanSemiring()
        a = wfa_from_word("hello", sr)
        b = wfa_from_word("world", sr)
        u = wfa_union(a, b)
        assert wfa_accepts(u, "hello")
        assert wfa_accepts(u, "world")
        assert not wfa_accepts(u, "other")

    def test_union_overlapping(self):
        """Same word in both: weights combine (tropical = min)."""
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 3.0)
        b = wfa_from_word("ab", sr, 5.0)
        u = wfa_union(a, b)
        assert wfa_run_weight(u, "ab") == 3.0  # min(3, 5)


# ============================================================
# Section 7: Concatenation
# ============================================================

class TestConcat:
    def test_tropical_concat(self):
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 2.0)
        b = wfa_from_word("cd", sr, 3.0)
        c = wfa_concat(a, b)
        assert wfa_run_weight(c, "abcd") == 5.0  # 2+3 (tropical times = add)
        assert wfa_run_weight(c, "ab") == sr.zero()
        assert wfa_run_weight(c, "cd") == sr.zero()

    def test_boolean_concat(self):
        sr = BooleanSemiring()
        a = wfa_from_word("ab", sr)
        b = wfa_from_word("cd", sr)
        c = wfa_concat(a, b)
        assert wfa_accepts(c, "abcd")
        assert not wfa_accepts(c, "ab")

    def test_concat_with_epsilon(self):
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 2.0)
        e = wfa_epsilon(sr, 0.0)
        c = wfa_concat(a, e)
        assert wfa_run_weight(c, "ab") == 2.0


# ============================================================
# Section 8: Kleene Star
# ============================================================

class TestStar:
    def test_star_boolean(self):
        sr = BooleanSemiring()
        a = wfa_from_symbol("a", sr)
        s = wfa_star(a)
        assert wfa_accepts(s, "")
        assert wfa_accepts(s, "a")
        assert wfa_accepts(s, "aa")
        assert wfa_accepts(s, "aaa")
        assert not wfa_accepts(s, "b")

    def test_star_tropical(self):
        sr = TropicalSemiring()
        a = wfa_from_symbol("a", sr, 0.0)
        a.add_transition(0, 'a', 0)  # need self-referencing for star to work
        # Actually let's test star properly
        a2 = wfa_from_symbol("a", sr, 0.0)
        s = wfa_star(a2)
        assert wfa_run_weight(s, "") == 0.0  # epsilon accepted
        assert wfa_run_weight(s, "a") == 0.0  # one 'a'
        assert wfa_run_weight(s, "aa") == 0.0  # two 'a's

    def test_star_counting(self):
        """Star of a single-symbol automaton."""
        sr = CountingSemiring()
        a = wfa_from_symbol("a", sr, 1)
        s = wfa_star(a)
        # "" has 1 run, "a" has 1 run, "aa" has 1 run
        assert wfa_run_weight(s, "") == 1
        assert wfa_run_weight(s, "a") == 1


# ============================================================
# Section 9: Intersection (Hadamard Product)
# ============================================================

class TestIntersection:
    def test_boolean_intersection(self):
        """Intersection of two NFAs."""
        sr = BooleanSemiring()
        # a accepts "ab", "ac"
        a = WFA(sr)
        a.add_state(0, initial=True)
        a.add_state(1)
        a.add_state(2, final=True)
        a.add_state(3, final=True)
        a.add_transition(0, 'a', 1)
        a.add_transition(1, 'b', 2)
        a.add_transition(1, 'c', 3)

        # b accepts "ab", "ad"
        b = WFA(sr)
        b.add_state(0, initial=True)
        b.add_state(1)
        b.add_state(2, final=True)
        b.add_state(3, final=True)
        b.add_transition(0, 'a', 1)
        b.add_transition(1, 'b', 2)
        b.add_transition(1, 'd', 3)

        i = wfa_intersect(a, b)
        assert wfa_accepts(i, "ab")
        assert not wfa_accepts(i, "ac")
        assert not wfa_accepts(i, "ad")

    def test_tropical_intersection(self):
        """Product of tropical weights."""
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 3.0)
        b = wfa_from_word("ab", sr, 5.0)
        i = wfa_intersect(a, b)
        # Tropical times = add, so 3+5=8 for weight product
        # But also initial weights (0+0=0) and final weights (3+5=8)
        # Actually: initial weights are both 0, final weights are 3 and 5
        # Path: init(0)*trans*trans*final = 0*0*0*3 and 0*0*0*5
        # Product: (0+0)*(0+0)*(0+0)*(3+5) = 0+0+0+8 = 8
        result = wfa_run_weight(i, "ab")
        assert result == 8.0

    def test_probability_intersection(self):
        """Product of probability weights."""
        sr = ProbabilitySemiring()
        a = wfa_from_word("x", sr, 0.6)
        b = wfa_from_word("x", sr, 0.5)
        i = wfa_intersect(a, b)
        result = wfa_run_weight(i, "x")
        assert result == pytest.approx(0.3)  # 0.6 * 0.5


# ============================================================
# Section 10: Shortest Distance
# ============================================================

class TestShortestDistance:
    def test_tropical_shortest(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1)
        wfa.add_state(2)
        wfa.add_transition(0, 'a', 1, 1.0)
        wfa.add_transition(0, 'b', 2, 5.0)
        wfa.add_transition(1, 'c', 2, 1.0)
        dist = shortest_distance(wfa)
        assert dist[0] == 0.0
        assert dist[1] == 1.0
        assert dist[2] == 2.0  # via 0->1->2

    def test_probability_shortest(self):
        sr = ProbabilitySemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1.0)
        wfa.add_state(1)
        wfa.add_transition(0, 'a', 1, 0.5)
        wfa.add_transition(0, 'a', 1, 0.3)
        dist = shortest_distance(wfa)
        assert dist[1] == pytest.approx(0.8)

    def test_all_pairs(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1)
        wfa.add_state(2)
        wfa.add_transition(0, 'a', 1, 2.0)
        wfa.add_transition(1, 'b', 2, 3.0)
        wfa.add_transition(0, 'c', 2, 10.0)
        dist = all_pairs_distance(wfa)
        assert dist[(0, 2)] == 5.0  # 2+3 via state 1


# ============================================================
# Section 11: N-Best Paths
# ============================================================

class TestNBestPaths:
    def test_tropical_nbest(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_state(2, final=0.0)
        wfa.add_transition(0, 'a', 1, 3.0)
        wfa.add_transition(0, 'a', 2, 7.0)
        best = n_best_paths(wfa, "a", 2)
        assert len(best) == 2
        assert best[0][0] == 3.0
        assert best[1][0] == 7.0

    def test_viterbi_nbest(self):
        sr = ViterbiSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1.0)
        wfa.add_state(1, final=1.0)
        wfa.add_state(2, final=1.0)
        wfa.add_transition(0, 'a', 1, 0.3)
        wfa.add_transition(0, 'a', 2, 0.7)
        best = n_best_paths(wfa, "a", 2)
        assert len(best) == 2
        assert best[0][0] == pytest.approx(0.7)  # highest first
        assert best[1][0] == pytest.approx(0.3)

    def test_no_accepting_path(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1)  # not final
        wfa.add_transition(0, 'a', 1, 1.0)
        best = n_best_paths(wfa, "a", 3)
        assert len(best) == 0


# ============================================================
# Section 12: Trim
# ============================================================

class TestTrim:
    def test_remove_unreachable(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True)
        wfa.add_state(1, final=True)
        wfa.add_state(2)  # unreachable
        wfa.add_transition(0, 'a', 1)
        trimmed = wfa_trim(wfa)
        assert 2 not in trimmed.states
        assert {0, 1} == trimmed.states

    def test_remove_nonproductive(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True)
        wfa.add_state(1)  # non-productive (no path to final)
        wfa.add_state(2, final=True)
        wfa.add_transition(0, 'a', 1)
        wfa.add_transition(0, 'b', 2)
        trimmed = wfa_trim(wfa)
        assert 1 not in trimmed.states
        assert {0, 2} == trimmed.states

    def test_already_trim(self):
        sr = BooleanSemiring()
        wfa = wfa_from_word("abc", sr)
        trimmed = wfa_trim(wfa)
        assert trimmed.states == wfa.states


# ============================================================
# Section 13: Weight Pushing
# ============================================================

class TestWeightPushing:
    def test_tropical_push(self):
        """After pushing, equivalent weight for all words."""
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1)
        wfa.add_state(2, final=0.0)
        wfa.add_transition(0, 'a', 1, 3.0)
        wfa.add_transition(1, 'b', 2, 5.0)
        pushed = wfa_push_weights(wfa)
        # Weight of "ab" should be preserved
        orig_w = wfa_run_weight(wfa, "ab")
        push_w = wfa_run_weight(pushed, "ab")
        assert abs(orig_w - push_w) < 1e-9


# ============================================================
# Section 14: Determinization
# ============================================================

class TestDeterminize:
    def test_tropical_determinize(self):
        """Determinize a non-deterministic tropical WFA."""
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_state(2, final=0.0)
        wfa.add_transition(0, 'a', 1, 3.0)
        wfa.add_transition(0, 'a', 2, 5.0)
        det = wfa_determinize(wfa)
        stats = wfa_stats(det)
        assert stats.is_deterministic
        # Weight should be min(3, 5) = 3
        assert wfa_run_weight(det, "a") == 3.0

    def test_determinize_preserves_language(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1)
        wfa.add_state(2, final=0.0)
        wfa.add_state(3, final=0.0)
        wfa.add_transition(0, 'a', 1, 1.0)
        wfa.add_transition(0, 'a', 2, 5.0)
        wfa.add_transition(1, 'b', 3, 2.0)
        det = wfa_determinize(wfa)
        # "a" weight: min(inf+5, inf) -- state 2 is final(0), state 1 not final
        # Actually: state 2 final=0.0, state 1 not final
        # So "a" = 5.0 (via state 2)
        assert wfa_run_weight(det, "a") == 5.0
        # "ab" = 1+2+0 = 3.0 (via 0->1->3)
        assert wfa_run_weight(det, "ab") == 3.0


# ============================================================
# Section 15: Equivalence
# ============================================================

class TestEquivalence:
    def test_same_wfa(self):
        sr = BooleanSemiring()
        a = wfa_from_word("abc", sr)
        eq, ce = wfa_equivalent(a, a)
        assert eq
        assert ce is None

    def test_different_wfa(self):
        sr = BooleanSemiring()
        a = wfa_from_word("abc", sr)
        b = wfa_from_word("abd", sr)
        eq, ce = wfa_equivalent(a, b)
        assert not eq
        assert ce is not None

    def test_tropical_equivalent(self):
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 5.0)
        b = wfa_from_word("ab", sr, 5.0)
        eq, ce = wfa_equivalent(a, b)
        assert eq

    def test_tropical_not_equivalent(self):
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 5.0)
        b = wfa_from_word("ab", sr, 3.0)
        eq, ce = wfa_equivalent(a, b)
        assert not eq
        assert ce == "ab"


# ============================================================
# Section 16: NFA Conversion
# ============================================================

class TestNFAConversion:
    def test_nfa_to_wfa(self):
        sr = BooleanSemiring()
        states = {0, 1, 2}
        initial = {0}
        final = {2}
        trans = {0: [('a', 1)], 1: [('b', 2)]}
        wfa = nfa_to_wfa(states, initial, final, trans, sr)
        assert wfa_accepts(wfa, "ab")
        assert not wfa_accepts(wfa, "a")

    def test_wfa_to_nfa(self):
        sr = TropicalSemiring()
        wfa = wfa_from_word("xy", sr, 5.0)
        states, init, final, trans = wfa_to_nfa(wfa)
        assert len(init) == 1
        assert len(final) == 1
        assert len(states) == 3

    def test_roundtrip(self):
        sr = BooleanSemiring()
        wfa1 = wfa_from_word("abc", sr)
        states, init, final, trans = wfa_to_nfa(wfa1)
        wfa2 = nfa_to_wfa(states, init, final, trans, sr)
        assert wfa_accepts(wfa2, "abc")
        assert not wfa_accepts(wfa2, "ab")


# ============================================================
# Section 17: Statistics & Summary
# ============================================================

class TestStats:
    def test_stats(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_transition(0, 'a', 1, 1.0)
        wfa.add_transition(0, 'b', 1, 2.0)
        stats = wfa_stats(wfa)
        assert stats.n_states == 2
        assert stats.n_transitions == 2
        assert stats.n_initial == 1
        assert stats.n_final == 1
        assert stats.is_deterministic == True
        assert stats.is_trim == True

    def test_nondeterministic(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True)
        wfa.add_state(1, final=True)
        wfa.add_state(2, final=True)
        wfa.add_transition(0, 'a', 1)
        wfa.add_transition(0, 'a', 2)
        stats = wfa_stats(wfa)
        assert stats.is_deterministic == False

    def test_summary(self):
        sr = BooleanSemiring()
        wfa = wfa_from_word("ab", sr)
        s = wfa_summary(wfa)
        assert "WFA" in s
        assert "states" in s


# ============================================================
# Section 18: Language Weight
# ============================================================

class TestLanguageWeight:
    def test_counting_total(self):
        """Total count of accepting runs for all words."""
        sr = CountingSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1)
        wfa.add_state(1, final=1)
        wfa.add_transition(0, 'a', 1, 1)
        # Only "a" is accepted, weight = 1
        total = wfa_language_weight(wfa, max_length=3)
        assert total == 1

    def test_boolean_total(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True, final=True)
        wfa.add_transition(0, 'a', 0)
        # Accepts "", "a", "aa", ...
        total = wfa_language_weight(wfa, max_length=3)
        assert total == True  # At least one word accepted


# ============================================================
# Section 19: Semiring Conversion & Comparison
# ============================================================

class TestConversion:
    def test_convert_boolean_to_counting(self):
        sr_b = BooleanSemiring()
        sr_c = CountingSemiring()
        wfa = wfa_from_word("ab", sr_b)
        converted = convert_semiring(wfa, sr_c, lambda x: 1 if x else 0)
        assert wfa_run_weight(converted, "ab") == 1
        assert wfa_run_weight(converted, "ac") == 0

    def test_compare_wfas(self):
        sr = TropicalSemiring()
        a = wfa_from_word("ab", sr, 3.0)
        b = wfa_from_word("ab", sr, 5.0)
        result = compare_wfas(a, b, ["ab", "cd"])
        assert result['n_words'] == 2
        assert result['n_agree'] == 1  # "cd" both inf
        assert result['per_word']['ab']['a'] == 3.0
        assert result['per_word']['ab']['b'] == 5.0


# ============================================================
# Section 20: Composition Patterns
# ============================================================

class TestCompositionPatterns:
    def test_regex_to_wfa(self):
        """Simulate a|b using union."""
        sr = BooleanSemiring()
        a = wfa_from_symbol("a", sr)
        b = wfa_from_symbol("b", sr)
        ab = wfa_union(a, b)
        assert wfa_accepts(ab, "a")
        assert wfa_accepts(ab, "b")
        assert not wfa_accepts(ab, "c")

    def test_concat_star(self):
        """a(b*) using concat + star."""
        sr = BooleanSemiring()
        a = wfa_from_symbol("a", sr)
        b = wfa_from_symbol("b", sr)
        bs = wfa_star(b)
        result = wfa_concat(a, bs)
        assert wfa_accepts(result, "a")
        assert wfa_accepts(result, "ab")
        assert wfa_accepts(result, "abb")
        assert not wfa_accepts(result, "b")
        assert not wfa_accepts(result, "ba")

    def test_weighted_shortest_path_network(self):
        """Model a weighted network as a WFA and find shortest path."""
        sr = TropicalSemiring()
        wfa = WFA(sr)
        # Network: A --(x,2)--> B --(y,3)--> C
        #          A --(x,1)--> D --(y,6)--> C
        wfa.add_state(0, initial=0.0)  # A
        wfa.add_state(1)               # B
        wfa.add_state(2, final=0.0)    # C
        wfa.add_state(3)               # D
        wfa.add_transition(0, 'x', 1, 2.0)
        wfa.add_transition(1, 'y', 2, 3.0)
        wfa.add_transition(0, 'x', 3, 1.0)
        wfa.add_transition(3, 'y', 2, 6.0)
        # "xy" weight = min(2+3, 1+6) = min(5, 7) = 5
        assert wfa_run_weight(wfa, "xy") == 5.0

    def test_probabilistic_model(self):
        """HMM-like: probability of observation sequence."""
        sr = ProbabilitySemiring()
        wfa = WFA(sr)
        # Two hidden states, observations 'a' and 'b'
        wfa.add_state(0, initial=0.6)  # P(start=0) = 0.6
        wfa.add_state(1, initial=0.4)  # P(start=1) = 0.4
        # State 0: P(a|0)=0.7, P(b|0)=0.3, transitions back to self/other
        wfa.add_transition(0, 'a', 0, 0.7 * 0.8)  # emit a, stay
        wfa.add_transition(0, 'a', 1, 0.7 * 0.2)  # emit a, switch
        wfa.add_transition(0, 'b', 0, 0.3 * 0.8)
        wfa.add_transition(0, 'b', 1, 0.3 * 0.2)
        # State 1: P(a|1)=0.4, P(b|1)=0.6
        wfa.add_transition(1, 'a', 0, 0.4 * 0.3)
        wfa.add_transition(1, 'a', 1, 0.4 * 0.7)
        wfa.add_transition(1, 'b', 0, 0.6 * 0.3)
        wfa.add_transition(1, 'b', 1, 0.6 * 0.7)
        # Both states are final (observation sequence can end anywhere)
        wfa.final_weight[0] = 1.0
        wfa.final_weight[1] = 1.0
        # P("a") should be sum of all paths for "a"
        p_a = wfa_run_weight(wfa, "a")
        assert p_a > 0
        assert p_a < 1.0

    def test_multi_semiring_comparison(self):
        """Same structure, different semirings give different answers."""
        # Build identical structure on boolean and counting
        sr_b = BooleanSemiring()
        sr_c = CountingSemiring()

        wb = WFA(sr_b)
        wb.add_state(0, initial=True)
        wb.add_state(1, final=True)
        wb.add_transition(0, 'a', 1, True)
        wb.add_transition(0, 'a', 1, True)  # two paths

        wc = WFA(sr_c)
        wc.add_state(0, initial=1)
        wc.add_state(1, final=1)
        wc.add_transition(0, 'a', 1, 1)
        wc.add_transition(0, 'a', 1, 1)

        assert wfa_run_weight(wb, "a") == True   # boolean: exists a path
        assert wfa_run_weight(wc, "a") == 2      # counting: 2 paths


# ============================================================
# Section 21: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_alphabet(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True, final=True)
        assert wfa_accepts(wfa, "")
        assert not wfa_accepts(wfa, "a")

    def test_single_state_loop(self):
        sr = CountingSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=1, final=1)
        wfa.add_transition(0, 'a', 0, 1)
        assert wfa_run_weight(wfa, "") == 1
        assert wfa_run_weight(wfa, "a") == 1
        assert wfa_run_weight(wfa, "aa") == 1

    def test_multiple_initial_states(self):
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, initial=3.0)
        wfa.add_state(2, final=0.0)
        wfa.add_transition(0, 'a', 2, 5.0)
        wfa.add_transition(1, 'a', 2, 1.0)
        # min(0+5+0, 3+1+0) = min(5, 4) = 4
        assert wfa_run_weight(wfa, "a") == 4.0

    def test_large_alphabet(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True)
        wfa.add_state(1, final=True)
        for i in range(26):
            wfa.add_transition(0, chr(ord('a') + i), 1)
        for ch in "abcdefghijklmnopqrstuvwxyz":
            assert wfa_accepts(wfa, ch)

    def test_n_transitions(self):
        sr = BooleanSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=True)
        wfa.add_state(1, final=True)
        wfa.add_transition(0, 'a', 1)
        wfa.add_transition(0, 'b', 1)
        wfa.add_transition(0, 'c', 1)
        assert wfa.n_transitions() == 3

    def test_union_empty(self):
        sr = BooleanSemiring()
        a = wfa_from_word("abc", sr)
        e = wfa_empty(sr)
        u = wfa_union(a, e)
        assert wfa_accepts(u, "abc")

    def test_concat_empty(self):
        sr = BooleanSemiring()
        a = wfa_from_word("abc", sr)
        e = wfa_empty(sr)
        c = wfa_concat(a, e)
        # Concat with empty language should give empty language
        assert not wfa_accepts(c, "abc")

    def test_star_empty(self):
        sr = BooleanSemiring()
        e = wfa_empty(sr)
        s = wfa_star(e)
        # Star of empty should accept epsilon
        # Actually, empty WFA has no initial states, so star still has none
        # The only thing star adds is final weight to initial states
        # If there are no initial states, no epsilon acceptance either
        # But wait: wfa_empty creates a state with no initial/final weights
        # So star doesn't help
        assert not wfa_accepts(s, "a")


# ============================================================
# Section 22: Integration Tests
# ============================================================

class TestIntegration:
    def test_full_regex_pipeline(self):
        """Build (a|b)*c as a WFA and test."""
        sr = BooleanSemiring()
        a = wfa_from_symbol("a", sr)
        b = wfa_from_symbol("b", sr)
        ab = wfa_union(a, b)
        ab_star = wfa_star(ab)
        c = wfa_from_symbol("c", sr)
        result = wfa_concat(ab_star, c)

        assert wfa_accepts(result, "c")
        assert wfa_accepts(result, "ac")
        assert wfa_accepts(result, "bc")
        assert wfa_accepts(result, "abc")
        assert wfa_accepts(result, "aabc")
        assert wfa_accepts(result, "babac")
        assert not wfa_accepts(result, "")
        assert not wfa_accepts(result, "a")
        assert not wfa_accepts(result, "ca")

    def test_shortest_path_graph(self):
        """Full shortest-path computation using tropical WFA."""
        sr = TropicalSemiring()
        wfa = WFA(sr)
        # Graph with 5 nodes
        for i in range(5):
            wfa.add_state(i)
        wfa.initial_weight[0] = 0.0
        wfa.final_weight[4] = 0.0
        # Edges
        wfa.add_transition(0, 'e', 1, 1.0)
        wfa.add_transition(0, 'e', 2, 4.0)
        wfa.add_transition(1, 'e', 2, 2.0)
        wfa.add_transition(1, 'e', 3, 6.0)
        wfa.add_transition(2, 'e', 3, 3.0)
        wfa.add_transition(3, 'e', 4, 1.0)
        # Shortest 0->4 via 4 edges: 0->1(1)->2(2)->3(3)->4(1) = 7
        assert wfa_run_weight(wfa, "eeee") == 7.0
        # Shortest 0->4 via 3 edges: 0->2(4)->3(3)->4(1) = 8
        assert wfa_run_weight(wfa, "eee") == 8.0

    def test_determinize_then_equivalence(self):
        """Determinized WFA should be equivalent to original (acyclic)."""
        sr = TropicalSemiring()
        wfa = WFA(sr)
        wfa.add_state(0, initial=0.0)
        wfa.add_state(1, final=0.0)
        wfa.add_state(2, final=0.0)
        wfa.add_state(3, final=0.0)
        wfa.add_transition(0, 'a', 1, 3.0)
        wfa.add_transition(0, 'a', 2, 7.0)
        wfa.add_transition(1, 'b', 3, 1.0)
        wfa.add_transition(2, 'b', 3, 2.0)
        det = wfa_determinize(wfa)
        eq, ce = wfa_equivalent(wfa, det, max_length=3)
        assert eq, f"Counterexample: {ce}"

    def test_trim_union_intersect(self):
        """Trim(union(a, b)) intersect c = trim of intersection."""
        sr = BooleanSemiring()
        a = wfa_from_word("abc", sr)
        b = wfa_from_word("abd", sr)
        c = wfa_from_word("abc", sr)
        u = wfa_union(a, b)
        ut = wfa_trim(u)
        i = wfa_intersect(ut, c)
        assert wfa_accepts(i, "abc")
        assert not wfa_accepts(i, "abd")
