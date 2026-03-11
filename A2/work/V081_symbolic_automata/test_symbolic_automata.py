"""Tests for V081: Symbolic Automata."""

import pytest
from symbolic_automata import (
    # Predicates
    PredKind, Pred, PTrue, PFalse, PChar, PRange, PAnd, POr, PNot,
    # Algebras
    CharAlgebra, IntAlgebra,
    # SFA
    SFA, SFATransition,
    # Boolean operations
    sfa_intersection, sfa_union, sfa_complement, sfa_difference,
    sfa_is_equivalent, sfa_is_subset,
    # Constructors
    sfa_from_string, sfa_from_char_class, sfa_from_range, sfa_any_char,
    sfa_epsilon, sfa_empty, sfa_concat, sfa_star, sfa_plus, sfa_optional,
    # SFT
    SFT, SFTTransition,
    # Utilities
    sfa_stats, compare_sfas, shortest_accepted,
)


# ============================================================================
# 1. Predicate Construction and Simplification
# ============================================================================

class TestPredicates:
    def test_true_false(self):
        assert PTrue().kind == PredKind.TRUE
        assert PFalse().kind == PredKind.FALSE

    def test_char(self):
        p = PChar('a')
        assert p.kind == PredKind.CHAR
        assert p.char == 'a'

    def test_range(self):
        p = PRange('a', 'z')
        assert p.kind == PredKind.RANGE
        assert p.lo == 'a'
        assert p.hi == 'z'

    def test_range_single_char(self):
        p = PRange('a', 'a')
        assert p.kind == PredKind.CHAR  # Simplified to char

    def test_range_empty(self):
        p = PRange('z', 'a')
        assert p.kind == PredKind.FALSE  # Empty range

    def test_and_simplification(self):
        assert PAnd(PTrue(), PChar('x')) == PChar('x')
        assert PAnd(PChar('x'), PTrue()) == PChar('x')
        assert PAnd(PFalse(), PChar('x')).kind == PredKind.FALSE
        assert PAnd(PChar('x'), PFalse()).kind == PredKind.FALSE
        assert PAnd(PChar('x'), PChar('x')) == PChar('x')

    def test_or_simplification(self):
        assert POr(PFalse(), PChar('x')) == PChar('x')
        assert POr(PChar('x'), PFalse()) == PChar('x')
        assert POr(PTrue(), PChar('x')).kind == PredKind.TRUE
        assert POr(PChar('x'), PTrue()).kind == PredKind.TRUE
        assert POr(PChar('x'), PChar('x')) == PChar('x')

    def test_not_simplification(self):
        assert PNot(PTrue()).kind == PredKind.FALSE
        assert PNot(PFalse()).kind == PredKind.TRUE
        assert PNot(PNot(PChar('x'))) == PChar('x')

    def test_repr(self):
        assert repr(PTrue()) == "T"
        assert repr(PFalse()) == "F"
        assert repr(PChar('a')) == "'a'"
        assert "[a-z]" in repr(PRange('a', 'z'))


# ============================================================================
# 2. CharAlgebra
# ============================================================================

class TestCharAlgebra:
    def setup_method(self):
        self.alg = CharAlgebra("abcdefghijklmnopqrstuvwxyz")

    def test_evaluate_true(self):
        assert self.alg.evaluate(PTrue(), 'a')

    def test_evaluate_false(self):
        assert not self.alg.evaluate(PFalse(), 'a')

    def test_evaluate_char(self):
        assert self.alg.evaluate(PChar('a'), 'a')
        assert not self.alg.evaluate(PChar('a'), 'b')

    def test_evaluate_range(self):
        p = PRange('a', 'f')
        assert self.alg.evaluate(p, 'a')
        assert self.alg.evaluate(p, 'c')
        assert self.alg.evaluate(p, 'f')
        assert not self.alg.evaluate(p, 'g')

    def test_evaluate_and(self):
        p = PAnd(PRange('a', 'f'), PRange('d', 'z'))
        assert self.alg.evaluate(p, 'd')
        assert self.alg.evaluate(p, 'f')
        assert not self.alg.evaluate(p, 'a')
        assert not self.alg.evaluate(p, 'g')

    def test_evaluate_or(self):
        p = POr(PChar('a'), PChar('z'))
        assert self.alg.evaluate(p, 'a')
        assert self.alg.evaluate(p, 'z')
        assert not self.alg.evaluate(p, 'm')

    def test_evaluate_not(self):
        p = PNot(PChar('a'))
        assert not self.alg.evaluate(p, 'a')
        assert self.alg.evaluate(p, 'b')

    def test_satisfiable(self):
        assert self.alg.is_satisfiable(PTrue())
        assert not self.alg.is_satisfiable(PFalse())
        assert self.alg.is_satisfiable(PChar('a'))
        assert not self.alg.is_satisfiable(PChar('1'))  # Not in alphabet

    def test_witness(self):
        assert self.alg.witness(PTrue()) == 'a'
        assert self.alg.witness(PChar('z')) == 'z'
        assert self.alg.witness(PFalse()) is None

    def test_enumerate(self):
        chars = self.alg.enumerate(PRange('x', 'z'))
        assert chars == ['x', 'y', 'z']

    def test_equivalence(self):
        p1 = POr(PChar('a'), PChar('b'))
        p2 = PRange('a', 'b')
        assert self.alg.is_equivalent(p1, p2)

    def test_non_equivalence(self):
        p1 = PRange('a', 'c')
        p2 = PRange('a', 'b')
        assert not self.alg.is_equivalent(p1, p2)


# ============================================================================
# 3. IntAlgebra
# ============================================================================

class TestIntAlgebra:
    def setup_method(self):
        self.alg = IntAlgebra(0, 10)

    def test_evaluate_char(self):
        assert self.alg.evaluate(PChar(5), 5)
        assert not self.alg.evaluate(PChar(5), 6)

    def test_evaluate_range(self):
        p = PRange(3, 7)
        assert self.alg.evaluate(p, 3)
        assert self.alg.evaluate(p, 5)
        assert self.alg.evaluate(p, 7)
        assert not self.alg.evaluate(p, 2)
        assert not self.alg.evaluate(p, 8)

    def test_satisfiable_range(self):
        assert self.alg.is_satisfiable(PRange(0, 5))
        assert not self.alg.is_satisfiable(PRange(11, 20))

    def test_witness(self):
        assert self.alg.witness(PRange(5, 8)) == 5

    def test_enumerate(self):
        vals = self.alg.enumerate(PRange(3, 5))
        assert vals == [3, 4, 5]

    def test_equivalence(self):
        p1 = POr(PChar(1), PChar(2))
        p2 = PRange(1, 2)
        assert self.alg.is_equivalent(p1, p2)


# ============================================================================
# 4. SFA Basic Construction
# ============================================================================

class TestSFABasic:
    def setup_method(self):
        self.alg = CharAlgebra("abc")

    def test_from_string(self):
        sfa = sfa_from_string("abc", self.alg)
        assert sfa.accepts("abc")
        assert not sfa.accepts("ab")
        assert not sfa.accepts("abcd")
        assert not sfa.accepts("")

    def test_from_string_empty(self):
        sfa = sfa_from_string("", self.alg)
        assert sfa.accepts("")
        assert not sfa.accepts("a")

    def test_from_char_class(self):
        sfa = sfa_from_char_class("ab", self.alg)
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert not sfa.accepts("c")
        assert not sfa.accepts("ab")

    def test_from_range(self):
        sfa = sfa_from_range('a', 'b', self.alg)
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert not sfa.accepts("c")

    def test_any_char(self):
        sfa = sfa_any_char(self.alg)
        assert sfa.accepts("a")
        assert sfa.accepts("b")
        assert sfa.accepts("c")
        assert not sfa.accepts("")
        assert not sfa.accepts("ab")

    def test_epsilon(self):
        sfa = sfa_epsilon(self.alg)
        assert sfa.accepts("")
        assert not sfa.accepts("a")

    def test_empty(self):
        sfa = sfa_empty(self.alg)
        assert not sfa.accepts("")
        assert not sfa.accepts("a")


# ============================================================================
# 5. SFA Properties
# ============================================================================

class TestSFAProperties:
    def setup_method(self):
        self.alg = CharAlgebra("ab")

    def test_deterministic(self):
        sfa = sfa_from_string("ab", self.alg)
        assert sfa.is_deterministic()

    def test_nondeterministic(self):
        # Two transitions from state 0 on 'a' to different states
        sfa = SFA(
            states={0, 1, 2},
            initial=0,
            accepting={1, 2},
            transitions=[
                SFATransition(0, PChar('a'), 1),
                SFATransition(0, PChar('a'), 2),
            ],
            algebra=self.alg
        )
        assert not sfa.is_deterministic()

    def test_is_empty(self):
        assert sfa_empty(self.alg).is_empty()
        assert not sfa_epsilon(self.alg).is_empty()
        assert not sfa_from_string("a", self.alg).is_empty()

    def test_is_empty_no_path_to_accepting(self):
        sfa = SFA(
            states={0, 1, 2},
            initial=0,
            accepting={2},
            transitions=[SFATransition(0, PChar('a'), 1)],  # No path to 2
            algebra=self.alg
        )
        assert sfa.is_empty()

    def test_accepted_word(self):
        sfa = sfa_from_string("ab", self.alg)
        word = sfa.accepted_word()
        assert word == ['a', 'b']

    def test_accepted_word_empty_language(self):
        sfa = sfa_empty(self.alg)
        assert sfa.accepted_word() is None

    def test_reachable_states(self):
        sfa = SFA(
            states={0, 1, 2, 3},
            initial=0,
            accepting={1},
            transitions=[SFATransition(0, PChar('a'), 1)],
            algebra=self.alg
        )
        assert sfa.reachable_states() == {0, 1}

    def test_trim(self):
        sfa = SFA(
            states={0, 1, 2, 3},
            initial=0,
            accepting={2},
            transitions=[
                SFATransition(0, PChar('a'), 1),
                SFATransition(1, PChar('b'), 2),
                SFATransition(0, PChar('b'), 3),  # 3 is dead end
            ],
            algebra=self.alg
        )
        trimmed = sfa.trim()
        assert 3 not in trimmed.states or not any(
            t.dst == 3 for t in trimmed.transitions
        )


# ============================================================================
# 6. SFA Determinization
# ============================================================================

class TestSFADeterminization:
    def setup_method(self):
        self.alg = CharAlgebra("ab")

    def test_determinize_nfa(self):
        # NFA: 0 -a-> 1, 0 -a-> 2, 1 accept, 2 -b-> 3, 3 accept
        nfa = SFA(
            states={0, 1, 2, 3},
            initial=0,
            accepting={1, 3},
            transitions=[
                SFATransition(0, PChar('a'), 1),
                SFATransition(0, PChar('a'), 2),
                SFATransition(2, PChar('b'), 3),
            ],
            algebra=self.alg
        )
        dfa = nfa.determinize()
        assert dfa.is_deterministic()
        assert dfa.accepts("a")
        assert dfa.accepts("ab")
        assert not dfa.accepts("b")
        assert not dfa.accepts("")

    def test_determinize_preserves_language(self):
        nfa = SFA(
            states={0, 1, 2},
            initial=0,
            accepting={1, 2},
            transitions=[
                SFATransition(0, PRange('a', 'b'), 1),
                SFATransition(0, PChar('a'), 2),
            ],
            algebra=self.alg
        )
        dfa = nfa.determinize()
        assert dfa.is_deterministic()
        assert dfa.accepts("a")
        assert dfa.accepts("b")

    def test_determinize_already_deterministic(self):
        dfa = sfa_from_string("ab", self.alg)
        dfa2 = dfa.determinize()
        assert dfa2.is_deterministic()
        assert dfa2.accepts("ab")
        assert not dfa2.accepts("a")


# ============================================================================
# 7. SFA Minimization
# ============================================================================

class TestSFAMinimization:
    def setup_method(self):
        self.alg = CharAlgebra("ab")

    def test_minimize_reduces_states(self):
        # Build a DFA with equivalent states
        sfa = SFA(
            states={0, 1, 2, 3},
            initial=0,
            accepting={2, 3},
            transitions=[
                SFATransition(0, PChar('a'), 1),
                SFATransition(0, PChar('b'), 1),
                SFATransition(1, PChar('a'), 2),
                SFATransition(1, PChar('b'), 3),
                # 2 and 3 are both accepting with no outgoing -> equivalent
            ],
            algebra=self.alg
        )
        minimized = sfa.minimize()
        # 2 and 3 should merge
        assert len(minimized.states) <= len(sfa.states)

    def test_minimize_preserves_language(self):
        sfa = sfa_from_string("ab", self.alg)
        minimized = sfa.minimize()
        assert minimized.accepts("ab")
        assert not minimized.accepts("a")
        assert not minimized.accepts("ba")


# ============================================================================
# 8. Boolean Operations -- Intersection
# ============================================================================

class TestIntersection:
    def setup_method(self):
        self.alg = CharAlgebra("abc")

    def test_intersection_strings(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ab", self.alg)
        result = sfa_intersection(a, b)
        assert result.accepts("ab")
        assert not result.accepts("a")

    def test_intersection_disjoint(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ba", self.alg)
        result = sfa_intersection(a, b)
        assert result.is_empty()

    def test_intersection_char_classes(self):
        a = sfa_from_char_class("abc", self.alg)
        b = sfa_from_char_class("bc", self.alg)
        result = sfa_intersection(a, b)
        assert result.accepts("b")
        assert result.accepts("c")
        assert not result.accepts("a")

    def test_intersection_range_overlap(self):
        a = sfa_from_range('a', 'b', self.alg)
        b = sfa_from_range('b', 'c', self.alg)
        result = sfa_intersection(a, b)
        assert result.accepts("b")
        assert not result.accepts("a")
        assert not result.accepts("c")


# ============================================================================
# 9. Boolean Operations -- Union
# ============================================================================

class TestUnion:
    def setup_method(self):
        self.alg = CharAlgebra("abc")

    def test_union_strings(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ba", self.alg)
        result = sfa_union(a, b)
        assert result.accepts("ab")
        assert result.accepts("ba")
        assert not result.accepts("aa")

    def test_union_char_classes(self):
        a = sfa_from_char_class("a", self.alg)
        b = sfa_from_char_class("c", self.alg)
        result = sfa_union(a, b)
        assert result.accepts("a")
        assert result.accepts("c")
        assert not result.accepts("b")


# ============================================================================
# 10. Boolean Operations -- Complement
# ============================================================================

class TestComplement:
    def setup_method(self):
        self.alg = CharAlgebra("ab")

    def test_complement_string(self):
        sfa = sfa_from_string("ab", self.alg)
        comp = sfa_complement(sfa)
        assert not comp.accepts("ab")
        assert comp.accepts("")
        assert comp.accepts("a")
        assert comp.accepts("ba")
        assert comp.accepts("aa")

    def test_complement_empty(self):
        sfa = sfa_empty(self.alg)
        comp = sfa_complement(sfa)
        assert comp.accepts("")
        assert comp.accepts("a")

    def test_double_complement(self):
        sfa = sfa_from_string("a", self.alg)
        comp2 = sfa_complement(sfa_complement(sfa))
        assert sfa_is_equivalent(sfa, comp2)


# ============================================================================
# 11. Boolean Operations -- Difference
# ============================================================================

class TestDifference:
    def setup_method(self):
        self.alg = CharAlgebra("abc")

    def test_difference_subset(self):
        a = sfa_from_char_class("abc", self.alg)
        b = sfa_from_char_class("a", self.alg)
        diff = sfa_difference(a, b)
        assert not diff.accepts("a")
        assert diff.accepts("b")
        assert diff.accepts("c")

    def test_difference_disjoint(self):
        a = sfa_from_char_class("a", self.alg)
        b = sfa_from_char_class("b", self.alg)
        diff = sfa_difference(a, b)
        assert diff.accepts("a")
        assert not diff.accepts("b")

    def test_difference_self(self):
        a = sfa_from_string("abc", self.alg)
        diff = sfa_difference(a, a)
        assert diff.is_empty()


# ============================================================================
# 12. Equivalence and Subset
# ============================================================================

class TestEquivalenceSubset:
    def setup_method(self):
        self.alg = CharAlgebra("abc")

    def test_equivalent_same(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ab", self.alg)
        assert sfa_is_equivalent(a, b)

    def test_not_equivalent(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ba", self.alg)
        assert not sfa_is_equivalent(a, b)

    def test_subset(self):
        a = sfa_from_char_class("a", self.alg)
        b = sfa_from_char_class("abc", self.alg)
        assert sfa_is_subset(a, b)
        assert not sfa_is_subset(b, a)

    def test_subset_equal(self):
        a = sfa_from_string("abc", self.alg)
        assert sfa_is_subset(a, a)


# ============================================================================
# 13. Concatenation
# ============================================================================

class TestConcatenation:
    def setup_method(self):
        self.alg = CharAlgebra("abc")

    def test_concat_strings(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("c", self.alg)
        result = sfa_concat(a, b)
        assert result.accepts("abc")
        assert not result.accepts("ab")
        assert not result.accepts("c")

    def test_concat_with_epsilon(self):
        a = sfa_from_string("ab", self.alg)
        eps = sfa_epsilon(self.alg)
        result = sfa_concat(a, eps)
        assert result.accepts("ab")
        assert not result.accepts("a")

    def test_concat_epsilon_left(self):
        eps = sfa_epsilon(self.alg)
        b = sfa_from_string("ab", self.alg)
        result = sfa_concat(eps, b)
        assert result.accepts("ab")

    def test_concat_char_classes(self):
        a = sfa_from_char_class("ab", self.alg)
        b = sfa_from_char_class("c", self.alg)
        result = sfa_concat(a, b)
        assert result.accepts("ac")
        assert result.accepts("bc")
        assert not result.accepts("cc")
        assert not result.accepts("a")


# ============================================================================
# 14. Kleene Star and Plus
# ============================================================================

class TestKleeneStar:
    def setup_method(self):
        self.alg = CharAlgebra("ab")

    def test_star_accepts_epsilon(self):
        sfa = sfa_star(sfa_from_char_class("a", self.alg))
        assert sfa.accepts("")

    def test_star_accepts_one(self):
        sfa = sfa_star(sfa_from_char_class("a", self.alg))
        assert sfa.accepts("a")

    def test_star_accepts_many(self):
        sfa = sfa_star(sfa_from_char_class("a", self.alg))
        assert sfa.accepts("aaa")

    def test_star_rejects_other(self):
        sfa = sfa_star(sfa_from_char_class("a", self.alg))
        assert not sfa.accepts("b")
        assert not sfa.accepts("ab")

    def test_plus(self):
        sfa = sfa_plus(sfa_from_char_class("a", self.alg))
        assert not sfa.accepts("")
        assert sfa.accepts("a")
        assert sfa.accepts("aa")

    def test_optional(self):
        sfa = sfa_optional(sfa_from_string("ab", self.alg))
        assert sfa.accepts("")
        assert sfa.accepts("ab")
        assert not sfa.accepts("a")


# ============================================================================
# 15. Symbolic Finite Transducer
# ============================================================================

class TestSFT:
    def setup_method(self):
        self.alg = CharAlgebra("abcABC")

    def test_identity_transducer(self):
        sft = SFT(
            states={0},
            initial=0,
            accepting={0},
            transitions=[
                SFTTransition(0, PTrue(), [lambda c: c], 0),
            ],
            algebra=self.alg
        )
        assert sft.transduce("abc") == ['a', 'b', 'c']
        assert sft.transduce("") == []

    def test_uppercasing_transducer(self):
        sft = SFT(
            states={0},
            initial=0,
            accepting={0},
            transitions=[
                SFTTransition(0, PRange('a', 'c'), [lambda c: c.upper()], 0),
                SFTTransition(0, PRange('A', 'C'), [lambda c: c], 0),
            ],
            algebra=self.alg
        )
        assert sft.transduce("abc") == ['A', 'B', 'C']

    def test_deleting_transducer(self):
        sft = SFT(
            states={0},
            initial=0,
            accepting={0},
            transitions=[
                SFTTransition(0, PChar('a'), [], 0),  # Delete 'a'
                SFTTransition(0, PNot(PChar('a')), [lambda c: c], 0),
            ],
            algebra=self.alg
        )
        assert sft.transduce("abcabc") == ['b', 'c', 'b', 'c']

    def test_transducer_domain(self):
        sft = SFT(
            states={0, 1},
            initial=0,
            accepting={1},
            transitions=[
                SFTTransition(0, PChar('a'), ['x'], 1),
            ],
            algebra=self.alg
        )
        domain = sft.domain()
        assert domain.accepts("a")
        assert not domain.accepts("b")

    def test_transducer_no_match(self):
        sft = SFT(
            states={0, 1},
            initial=0,
            accepting={1},
            transitions=[
                SFTTransition(0, PChar('a'), ['x'], 1),
            ],
            algebra=self.alg
        )
        assert sft.transduce("b") is None


# ============================================================================
# 16. Integer Alphabet SFA
# ============================================================================

class TestIntSFA:
    def setup_method(self):
        self.alg = IntAlgebra(0, 10)

    def test_int_sfa_accepts(self):
        sfa = SFA(
            states={0, 1, 2},
            initial=0,
            accepting={2},
            transitions=[
                SFATransition(0, PRange(0, 5), 1),
                SFATransition(1, PRange(6, 10), 2),
            ],
            algebra=self.alg
        )
        assert sfa.accepts([3, 7])
        assert not sfa.accepts([7, 3])
        assert not sfa.accepts([3])

    def test_int_complement(self):
        sfa = SFA(
            states={0, 1},
            initial=0,
            accepting={1},
            transitions=[
                SFATransition(0, PChar(5), 1),
            ],
            algebra=self.alg
        )
        comp = sfa_complement(sfa)
        assert not comp.accepts([5])
        assert comp.accepts([3])
        assert comp.accepts([])

    def test_int_intersection(self):
        a = SFA(
            states={0, 1},
            initial=0,
            accepting={1},
            transitions=[SFATransition(0, PRange(0, 7), 1)],
            algebra=self.alg
        )
        b = SFA(
            states={0, 1},
            initial=0,
            accepting={1},
            transitions=[SFATransition(0, PRange(3, 10), 1)],
            algebra=self.alg
        )
        result = sfa_intersection(a, b)
        assert result.accepts([5])
        assert not result.accepts([1])
        assert not result.accepts([9])


# ============================================================================
# 17. Complex Patterns
# ============================================================================

class TestComplexPatterns:
    def setup_method(self):
        self.alg = CharAlgebra("abcde")

    def test_pattern_a_star_b(self):
        # a*b
        a_star = sfa_star(sfa_from_char_class("a", self.alg))
        b = sfa_from_string("b", self.alg)
        pattern = sfa_concat(a_star, b)
        assert pattern.accepts("b")
        assert pattern.accepts("ab")
        assert pattern.accepts("aab")
        assert not pattern.accepts("a")
        assert not pattern.accepts("")

    def test_pattern_a_or_b_star(self):
        # (a|b)*
        ab = sfa_from_char_class("ab", self.alg)
        pattern = sfa_star(ab)
        assert pattern.accepts("")
        assert pattern.accepts("a")
        assert pattern.accepts("b")
        assert pattern.accepts("abba")
        assert not pattern.accepts("c")

    def test_pattern_complement_of_star(self):
        # Complement of a* = strings containing non-'a' characters
        a_star = sfa_star(sfa_from_char_class("a", self.alg))
        comp = sfa_complement(a_star)
        assert not comp.accepts("")
        assert not comp.accepts("a")
        assert not comp.accepts("aaa")
        assert comp.accepts("b")
        assert comp.accepts("ab")

    def test_intersection_of_patterns(self):
        # L1 = strings starting with 'a', L2 = strings ending with 'b'
        # L1 & L2 = strings starting with 'a' and ending with 'b'
        a_start = sfa_concat(
            sfa_from_char_class("a", self.alg),
            sfa_star(sfa_any_char(self.alg))
        )
        b_end = sfa_concat(
            sfa_star(sfa_any_char(self.alg)),
            sfa_from_char_class("b", self.alg)
        )
        result = sfa_intersection(a_start, b_end)
        assert result.accepts("ab")
        assert result.accepts("acb")
        assert not result.accepts("ba")
        assert not result.accepts("a")


# ============================================================================
# 18. Statistics and Comparison
# ============================================================================

class TestStatsComparison:
    def setup_method(self):
        self.alg = CharAlgebra("ab")

    def test_stats(self):
        sfa = sfa_from_string("ab", self.alg)
        stats = sfa_stats(sfa)
        assert stats['states'] == 3
        assert stats['transitions'] == 2
        assert stats['accepting'] == 1
        assert stats['deterministic'] is True
        assert stats['empty'] is False

    def test_compare_equivalent(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ab", self.alg)
        result = compare_sfas(a, b)
        assert result['equivalent'] is True

    def test_compare_different(self):
        a = sfa_from_string("ab", self.alg)
        b = sfa_from_string("ba", self.alg)
        result = compare_sfas(a, b)
        assert result['equivalent'] is False
        assert 'witness_in_a_not_b' in result or 'witness_in_b_not_a' in result

    def test_shortest_accepted(self):
        sfa = sfa_from_string("abc", CharAlgebra("abc"))
        word = shortest_accepted(sfa)
        assert word == ['a', 'b', 'c']


# ============================================================================
# 19. Edge Cases
# ============================================================================

class TestEdgeCases:
    def setup_method(self):
        self.alg = CharAlgebra("a")

    def test_single_char_alphabet(self):
        sfa = sfa_from_char_class("a", self.alg)
        assert sfa.accepts("a")
        assert not sfa.accepts("")

    def test_star_of_empty(self):
        sfa = sfa_star(sfa_empty(self.alg))
        assert sfa.accepts("")  # Kleene star of empty always accepts epsilon

    def test_concat_with_empty(self):
        a = sfa_from_string("a", self.alg)
        empty = sfa_empty(self.alg)
        result = sfa_concat(a, empty)
        # a followed by nothing from empty = nothing (empty absorbs)
        # Actually: concat with empty language produces empty language
        assert not result.accepts("a")
        assert result.is_empty()

    def test_union_with_empty(self):
        a = sfa_from_string("a", self.alg)
        empty = sfa_empty(self.alg)
        result = sfa_union(a, empty)
        assert result.accepts("a")

    def test_complement_of_universal(self):
        # star of any char = all strings
        universal = sfa_star(sfa_any_char(self.alg))
        comp = sfa_complement(universal)
        assert comp.is_empty()

    def test_self_loop(self):
        sfa = SFA(
            states={0},
            initial=0,
            accepting={0},
            transitions=[SFATransition(0, PChar('a'), 0)],
            algebra=self.alg
        )
        assert sfa.accepts("")
        assert sfa.accepts("a")
        assert sfa.accepts("aaa")


# ============================================================================
# 20. Predicate Combinations
# ============================================================================

class TestPredicateCombinations:
    def setup_method(self):
        self.alg = CharAlgebra("abcdefghij")

    def test_complex_predicate(self):
        # (a-d) AND NOT(b) = {a, c, d}
        p = PAnd(PRange('a', 'd'), PNot(PChar('b')))
        assert self.alg.evaluate(p, 'a')
        assert not self.alg.evaluate(p, 'b')
        assert self.alg.evaluate(p, 'c')
        assert self.alg.evaluate(p, 'd')
        assert not self.alg.evaluate(p, 'e')

    def test_nested_or_and(self):
        # (a OR b) AND (b OR c) = {b}
        p = PAnd(POr(PChar('a'), PChar('b')), POr(PChar('b'), PChar('c')))
        assert not self.alg.evaluate(p, 'a')
        assert self.alg.evaluate(p, 'b')
        assert not self.alg.evaluate(p, 'c')

    def test_de_morgan(self):
        # NOT(a AND b) = NOT(a) OR NOT(b) -- but on characters, a AND b is
        # satisfiable only if a == b
        p1 = PNot(PAnd(PChar('a'), PChar('b')))
        # Since PChar('a') AND PChar('b') is never true for any single char,
        # NOT of that is always true
        assert self.alg.evaluate(p1, 'a')
        assert self.alg.evaluate(p1, 'b')
        assert self.alg.evaluate(p1, 'c')

    def test_sfa_with_complex_guards(self):
        # Accept strings of length 1 that are vowels
        vowels = POr(PChar('a'), POr(PChar('e'), PChar('i')))
        sfa = SFA(
            states={0, 1},
            initial=0,
            accepting={1},
            transitions=[SFATransition(0, vowels, 1)],
            algebra=self.alg
        )
        assert sfa.accepts("a")
        assert sfa.accepts("e")
        assert sfa.accepts("i")
        assert not sfa.accepts("b")
        assert not sfa.accepts("ae")
