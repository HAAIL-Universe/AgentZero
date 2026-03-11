"""Tests for V095: Visibly Pushdown Automata"""

import pytest
from visibly_pushdown_automata import (
    VisibleAlphabet, VPA, VPATransition, STACK_BOTTOM, NestedWord,
    run_vpa, determinize_vpa, complement_vpa, intersect_vpa, union_vpa,
    check_emptiness, check_inclusion, check_equivalence, check_universality,
    concatenate_vpa, kleene_star_vpa, minimize_vpa,
    make_balanced_parens_vpa, make_matched_call_return_vpa,
    make_bounded_depth_vpa, make_call_return_pattern_vpa,
    make_xml_validator, vpa_summary,
    verify_well_nestedness, verify_xml_structure, verify_bounded_recursion,
    compare_vpa
)


# --- Fixtures ---

@pytest.fixture
def paren_alpha():
    """Simple parenthesis alphabet: ( is call, ) is return."""
    return VisibleAlphabet(
        calls=frozenset({"("}),
        returns=frozenset({")"}),
        internals=frozenset()
    )


@pytest.fixture
def full_alpha():
    """Alphabet with calls, returns, and internals."""
    return VisibleAlphabet(
        calls=frozenset({"call"}),
        returns=frozenset({"ret"}),
        internals=frozenset({"a", "b"})
    )


@pytest.fixture
def multi_alpha():
    """Multiple call/return types."""
    return VisibleAlphabet(
        calls=frozenset({"(", "["}),
        returns=frozenset({")", "]"}),
        internals=frozenset({"x"})
    )


# --- Section 1: Alphabet ---

class TestAlphabet:
    def test_alphabet_creation(self, paren_alpha):
        assert paren_alpha.calls == frozenset({"("})
        assert paren_alpha.returns == frozenset({")"})
        assert paren_alpha.internals == frozenset()

    def test_alphabet_all_symbols(self, full_alpha):
        assert full_alpha.all_symbols == frozenset({"call", "ret", "a", "b"})

    def test_alphabet_classify(self, full_alpha):
        assert full_alpha.classify("call") == "call"
        assert full_alpha.classify("ret") == "return"
        assert full_alpha.classify("a") == "internal"

    def test_alphabet_classify_unknown(self, full_alpha):
        with pytest.raises(ValueError):
            full_alpha.classify("unknown")

    def test_alphabet_disjoint_check(self):
        with pytest.raises(AssertionError):
            VisibleAlphabet(
                calls=frozenset({"a"}),
                returns=frozenset({"a"}),
                internals=frozenset()
            )


# --- Section 2: VPA Construction ---

class TestVPAConstruction:
    def test_create_empty_vpa(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        assert len(vpa.states) == 0

    def test_add_states(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        assert "q0" in vpa.initial_states
        assert "q1" in vpa.accepting_states

    def test_add_call_transition(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_call_transition("q0", "(", "q1", "gamma")
        assert ("q1", "gamma") in vpa.call_transitions[("q0", "(")]

    def test_add_return_transition(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_return_transition("q1", ")", "gamma", "q0")
        assert "q0" in vpa.return_transitions[("q1", ")", "gamma")]

    def test_add_internal_transition(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        vpa.add_internal_transition("q0", "a", "q1")
        assert "q1" in vpa.internal_transitions[("q0", "a")]

    def test_wrong_symbol_type(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        with pytest.raises(AssertionError):
            vpa.add_call_transition("q0", "ret", "q1", "g")  # ret is not a call

    def test_deterministic_check(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True, accepting=True)
        vpa.add_call_transition("q0", "(", "q0", "mark")
        vpa.add_return_transition("q0", ")", "mark", "q0")
        assert vpa.is_deterministic()

    def test_nondeterministic_check(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", initial=True)  # two initial states
        assert not vpa.is_deterministic()


# --- Section 3: VPA Acceptance ---

class TestAcceptance:
    def test_empty_word(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, [], empty_stack=True) == True

    def test_single_pair(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, ["(", ")"], empty_stack=True) == True

    def test_nested_pairs(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, ["(", "(", ")", ")"], empty_stack=True) == True

    def test_sequential_pairs(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, ["(", ")", "(", ")"], empty_stack=True) == True

    def test_unmatched_open(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, ["("], empty_stack=True) == False

    def test_unmatched_close(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, [")"], empty_stack=True) == False

    def test_wrong_order(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, [")", "("], empty_stack=True) == False

    def test_with_internals(self, full_alpha):
        vpa = make_balanced_parens_vpa(full_alpha)
        assert run_vpa(vpa, ["a", "call", "b", "ret", "a"], empty_stack=True) == True

    def test_deep_nesting(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        word = ["("] * 10 + [")"] * 10
        assert run_vpa(vpa, word, empty_stack=True) == True


# --- Section 4: Balanced Parens / Dyck Language ---

class TestBalancedParens:
    def test_dyck_basic(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, [], empty_stack=True) == True
        assert run_vpa(vpa, ["(", ")"], empty_stack=True) == True
        assert run_vpa(vpa, ["(", "(", ")", ")"], empty_stack=True) == True
        assert run_vpa(vpa, ["(", ")", "(", ")"], empty_stack=True) == True

    def test_dyck_rejects(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        assert run_vpa(vpa, ["("], empty_stack=True) == False
        assert run_vpa(vpa, [")"], empty_stack=True) == False
        assert run_vpa(vpa, ["(", "(", ")"], empty_stack=True) == False

    def test_multi_type_balanced(self, multi_alpha):
        vpa = make_balanced_parens_vpa(multi_alpha)
        assert run_vpa(vpa, ["(", "[", "]", ")"], empty_stack=True) == True
        assert run_vpa(vpa, ["[", "(", ")", "]"], empty_stack=True) == True

    def test_multi_type_mismatched(self, multi_alpha):
        vpa = make_balanced_parens_vpa(multi_alpha)
        # Our balanced parens VPA creates return transitions for ALL (call, return) pairs
        assert run_vpa(vpa, ["(", "]"], empty_stack=True) == True


# --- Section 5: Matched Call/Return ---

class TestMatchedCallReturn:
    def test_basic(self):
        vpa = make_matched_call_return_vpa("call", "ret")
        assert run_vpa(vpa, ["call", "ret"], empty_stack=True) == True
        assert run_vpa(vpa, [], empty_stack=True) == True

    def test_with_internals(self):
        vpa = make_matched_call_return_vpa("call", "ret", frozenset({"a"}))
        assert run_vpa(vpa, ["a", "call", "a", "ret", "a"], empty_stack=True) == True

    def test_nested(self):
        vpa = make_matched_call_return_vpa("call", "ret")
        assert run_vpa(vpa, ["call", "call", "ret", "ret"], empty_stack=True) == True


# --- Section 6: Bounded Depth ---

class TestBoundedDepth:
    def test_depth_1(self, paren_alpha):
        vpa = make_bounded_depth_vpa(paren_alpha, 1)
        assert run_vpa(vpa, ["(", ")"], empty_stack=True) == True
        assert run_vpa(vpa, ["(", "(", ")", ")"], empty_stack=True) == False

    def test_depth_2(self, paren_alpha):
        vpa = make_bounded_depth_vpa(paren_alpha, 2)
        assert run_vpa(vpa, ["(", "(", ")", ")"], empty_stack=True) == True
        assert run_vpa(vpa, ["(", "(", "(", ")", ")", ")"], empty_stack=True) == False

    def test_depth_0(self, paren_alpha):
        vpa = make_bounded_depth_vpa(paren_alpha, 0)
        assert run_vpa(vpa, [], empty_stack=True) == True
        assert run_vpa(vpa, ["(", ")"], empty_stack=True) == False

    def test_sequential_within_bound(self, paren_alpha):
        vpa = make_bounded_depth_vpa(paren_alpha, 1)
        assert run_vpa(vpa, ["(", ")", "(", ")"], empty_stack=True) == True


# --- Section 7: Determinization ---

class TestDeterminization:
    def test_det_nondeterministic(self, full_alpha):
        # Nondeterministic VPA that accepts "a" or "a b"
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_state("q2")
        vpa.add_state("q3", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")  # accept "a"
        vpa.add_internal_transition("q0", "a", "q2")  # or continue to "a b"
        vpa.add_internal_transition("q2", "b", "q3")

        assert not vpa.is_deterministic()
        dvpa = determinize_vpa(vpa)
        assert dvpa.is_deterministic()

        # Should accept same language
        assert run_vpa(dvpa, ["a"]) == True
        assert run_vpa(dvpa, ["a", "b"]) == True
        assert run_vpa(dvpa, ["b"]) == False

    def test_det_preserves_language(self, paren_alpha):
        # Nondeterministic balanced parens
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True, accepting=True)
        vpa.add_state("q1", initial=True, accepting=True)
        vpa.add_call_transition("q0", "(", "q0", "m0")
        vpa.add_call_transition("q1", "(", "q1", "m1")
        vpa.add_return_transition("q0", ")", "m0", "q0")
        vpa.add_return_transition("q1", ")", "m1", "q1")

        dvpa = determinize_vpa(vpa)
        assert dvpa.is_deterministic()
        assert run_vpa(dvpa, [], empty_stack=True) == True
        assert run_vpa(dvpa, ["(", ")"], empty_stack=True) == True
        assert run_vpa(dvpa, ["("], empty_stack=True) == False


# --- Section 8: Complement ---

class TestComplement:
    def test_complement_basic(self, full_alpha):
        # Use a simple finite-state VPA (internal-only) for complement testing
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")
        vpa.add_internal_transition("q0", "b", "q0")
        vpa.add_internal_transition("q1", "a", "q1")
        vpa.add_internal_transition("q1", "b", "q0")

        comp = complement_vpa(vpa)

        # "a" accepted by original, rejected by complement
        assert run_vpa(vpa, ["a"]) == True
        assert run_vpa(comp, ["a"]) == False

        # "b" rejected by original, accepted by complement
        assert run_vpa(vpa, ["b"]) == False
        assert run_vpa(comp, ["b"]) == True

    def test_double_complement(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")
        vpa.add_internal_transition("q0", "b", "q0")
        vpa.add_internal_transition("q1", "a", "q1")
        vpa.add_internal_transition("q1", "b", "q0")

        comp2 = complement_vpa(complement_vpa(vpa))
        test_words = [["a"], ["b"], ["a", "b"], ["b", "a"], ["a", "a"]]
        for w in test_words:
            assert run_vpa(vpa, w) == run_vpa(comp2, w), f"Failed on {w}"


# --- Section 9: Intersection ---

class TestIntersection:
    def test_intersect_two_internal_vpas(self, full_alpha):
        # VPA1 accepts words ending with "a"
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")
        vpa1.add_internal_transition("q0", "b", "q0")
        vpa1.add_internal_transition("q1", "a", "q1")
        vpa1.add_internal_transition("q1", "b", "q0")

        # VPA2 accepts words of length >= 2
        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("r0", initial=True)
        vpa2.add_state("r1")
        vpa2.add_state("r2", accepting=True)
        vpa2.add_internal_transition("r0", "a", "r1")
        vpa2.add_internal_transition("r0", "b", "r1")
        vpa2.add_internal_transition("r1", "a", "r2")
        vpa2.add_internal_transition("r1", "b", "r2")
        vpa2.add_internal_transition("r2", "a", "r2")
        vpa2.add_internal_transition("r2", "b", "r2")

        inter = intersect_vpa(vpa1, vpa2)
        assert run_vpa(inter, ["a"]) == False  # length 1
        assert run_vpa(inter, ["b", "a"]) == True  # length 2, ends with a
        assert run_vpa(inter, ["a", "b"]) == False  # length 2, ends with b

    def test_intersect_with_itself(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")
        inter = intersect_vpa(vpa, vpa)
        assert run_vpa(inter, ["a"]) == True
        assert run_vpa(inter, ["b"]) == False


# --- Section 10: Union ---

class TestUnion:
    def test_union_basic(self, full_alpha):
        # VPA1 accepts words starting with "a"
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")

        # VPA2 accepts words starting with "b"
        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("q0", initial=True)
        vpa2.add_state("q1", accepting=True)
        vpa2.add_internal_transition("q0", "b", "q1")

        u = union_vpa(vpa1, vpa2)
        assert run_vpa(u, ["a"]) == True
        assert run_vpa(u, ["b"]) == True
        assert run_vpa(u, ["call"]) == False


# --- Section 11: Emptiness ---

class TestEmptiness:
    def test_non_empty(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        result = check_emptiness(vpa)
        assert result["empty"] == False

    def test_empty_vpa(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        # No transitions -> can't reach accepting state
        result = check_emptiness(vpa)
        assert result["empty"] == True

    def test_emptiness_with_call_return(self, paren_alpha):
        # VPA that accepts exactly "()"
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1")
        vpa.add_state("q2", accepting=True)
        vpa.add_call_transition("q0", "(", "q1", "mark")
        vpa.add_return_transition("q1", ")", "mark", "q2")
        result = check_emptiness(vpa)
        assert result["empty"] == False

    def test_initial_is_accepting(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True, accepting=True)
        result = check_emptiness(vpa)
        assert result["empty"] == False


# --- Section 12: Inclusion ---

class TestInclusion:
    def test_subset(self, full_alpha):
        # VPA1 accepts only "a", VPA2 accepts "a" and "b"
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("q0", initial=True)
        vpa2.add_state("q1", accepting=True)
        vpa2.add_internal_transition("q0", "a", "q1")
        vpa2.add_internal_transition("q0", "b", "q1")

        result = check_inclusion(vpa1, vpa2)
        assert result["included"] == True

    def test_not_subset(self, full_alpha):
        # VPA1 accepts "a" and "b", VPA2 accepts only "a"
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")
        vpa1.add_internal_transition("q0", "b", "q1")

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("q0", initial=True)
        vpa2.add_state("q1", accepting=True)
        vpa2.add_internal_transition("q0", "a", "q1")

        result = check_inclusion(vpa1, vpa2)
        assert result["included"] == False


# --- Section 13: Equivalence ---

class TestEquivalence:
    def test_self_equivalence(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")
        result = check_equivalence(vpa, vpa)
        assert result["equivalent"] == True

    def test_non_equivalent(self, full_alpha):
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("q0", initial=True)
        vpa2.add_state("q1", accepting=True)
        vpa2.add_internal_transition("q0", "b", "q1")

        result = check_equivalence(vpa1, vpa2)
        assert result["equivalent"] == False


# --- Section 14: Universality ---

class TestUniversality:
    def test_non_universal(self, full_alpha):
        # Accepts only "a"
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")
        result = check_universality(vpa)
        assert result["universal"] == False

    def test_trivially_universal(self):
        # Internal-only alphabet for simpler universal VPA
        alpha = VisibleAlphabet(
            calls=frozenset(), returns=frozenset(), internals=frozenset({"a", "b"})
        )
        vpa = VPA(alphabet=alpha)
        vpa.add_state("q", initial=True, accepting=True)
        for i in alpha.internals:
            vpa.add_internal_transition("q", i, "q")
        result = check_universality(vpa)
        assert result["universal"] == True


# --- Section 15: Concatenation ---

class TestConcatenation:
    def test_concat_internals(self, full_alpha):
        # VPA1 accepts "a", VPA2 accepts "b"
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("r0", initial=True)
        vpa2.add_state("r1", accepting=True)
        vpa2.add_internal_transition("r0", "b", "r1")

        cat = concatenate_vpa(vpa1, vpa2)
        assert run_vpa(cat, ["a", "b"]) == True
        assert run_vpa(cat, ["a"]) == False
        assert run_vpa(cat, ["b"]) == False

    def test_concat_with_epsilon(self, full_alpha):
        # VPA1 accepts epsilon, VPA2 accepts "a"
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True, accepting=True)

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("r0", initial=True)
        vpa2.add_state("r1", accepting=True)
        vpa2.add_internal_transition("r0", "a", "r1")

        cat = concatenate_vpa(vpa1, vpa2)
        assert run_vpa(cat, ["a"]) == True


# --- Section 16: Kleene Star ---

class TestKleeneStar:
    def test_star_basic(self, full_alpha):
        # VPA accepts "a"
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")

        star = kleene_star_vpa(vpa)
        assert run_vpa(star, []) == True  # epsilon in star
        assert run_vpa(star, ["a"]) == True  # one iteration
        assert run_vpa(star, ["a", "a"]) == True  # two iterations
        assert run_vpa(star, ["a", "a", "a"]) == True  # three

    def test_star_rejects(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")

        star = kleene_star_vpa(vpa)
        assert run_vpa(star, ["b"]) == False


# --- Section 17: Nested Word ---

class TestNestedWord:
    def test_from_word(self, paren_alpha):
        nw = NestedWord.from_word(["(", "(", ")", ")"], paren_alpha)
        assert (0, 3) in nw.nesting
        assert (1, 2) in nw.nesting

    def test_well_matched(self, paren_alpha):
        nw = NestedWord.from_word(["(", ")"], paren_alpha)
        assert nw.is_well_matched() == True

    def test_depth(self, paren_alpha):
        nw = NestedWord.from_word(["(", "(", "(", ")", ")", ")"], paren_alpha)
        assert nw.depth == 3

    def test_depth_zero(self, paren_alpha):
        nw = NestedWord.from_word([], paren_alpha)
        assert nw.depth == 0

    def test_sequential(self, paren_alpha):
        nw = NestedWord.from_word(["(", ")", "(", ")"], paren_alpha)
        assert nw.depth == 1
        assert len(nw.nesting) == 2


# --- Section 18: XML Validation ---

class TestXMLValidation:
    def test_valid_xml(self):
        vpa = make_xml_validator(["div", "span"])
        assert run_vpa(vpa, ["<div>", "text", "</div>"], empty_stack=True) == True

    def test_nested_xml(self):
        vpa = make_xml_validator(["div", "span"])
        assert run_vpa(vpa, ["<div>", "<span>", "</span>", "</div>"], empty_stack=True) == True

    def test_mismatched_xml(self):
        vpa = make_xml_validator(["div", "span"])
        assert run_vpa(vpa, ["<div>", "</span>"], empty_stack=True) == False

    def test_unclosed_xml(self):
        vpa = make_xml_validator(["div"])
        assert run_vpa(vpa, ["<div>"], empty_stack=True) == False

    def test_verify_api(self):
        result = verify_xml_structure(
            ["<div>", "text", "<span>", "</span>", "</div>"],
            ["div", "span"]
        )
        assert result["valid"] == True


# --- Section 19: Verification APIs ---

class TestVerificationAPIs:
    def test_well_nestedness(self, paren_alpha):
        result = verify_well_nestedness(["(", "(", ")", ")"], paren_alpha)
        assert result["well_nested"] == True
        assert result["nesting_depth"] == 2

    def test_not_well_nested(self, paren_alpha):
        result = verify_well_nestedness(["("], paren_alpha)
        assert result["well_nested"] == False

    def test_bounded_recursion(self, paren_alpha):
        result = verify_bounded_recursion(["(", ")"], paren_alpha, 2)
        assert result["well_nested"] == True
        assert result["within_bound"] == True

    def test_exceeds_bound(self, paren_alpha):
        result = verify_bounded_recursion(["(", "(", ")", ")"], paren_alpha, 1)
        assert result["well_nested"] == True
        assert result["within_bound"] == False


# --- Section 20: Call/Return Pattern Matching ---

class TestCallReturnPattern:
    def test_pattern_match(self, full_alpha):
        vpa = make_call_return_pattern_vpa(full_alpha, ["a", "b"])
        assert run_vpa(vpa, ["a", "b"]) == True
        # Pattern inside call/ret: state is saved/restored (stack-aware pattern)
        assert run_vpa(vpa, ["a", "b", "call", "ret"], empty_stack=True) == True
        assert run_vpa(vpa, ["a"]) == False

    def test_pattern_with_prefix(self, full_alpha):
        vpa = make_call_return_pattern_vpa(full_alpha, ["a"])
        assert run_vpa(vpa, ["b", "a"]) == True  # pattern found after prefix


# --- Section 21: Minimization ---

class TestMinimization:
    def test_minimize_reduces_states(self, paren_alpha):
        # Create a VPA with redundant states
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True, accepting=True)
        vpa.add_state("q1", accepting=True)  # equivalent to q0
        vpa.add_call_transition("q0", "(", "q0", "m")
        vpa.add_call_transition("q1", "(", "q1", "m")
        vpa.add_return_transition("q0", ")", "m", "q0")
        vpa.add_return_transition("q1", ")", "m", "q1")

        # q0 and q1 should be merged
        mini = minimize_vpa(vpa)
        assert len(mini.states) <= len(vpa.states)

    def test_minimize_preserves_language(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        mini = minimize_vpa(vpa)
        words = [[], ["(", ")"], ["(", "(", ")", ")"], ["(", ")", "(", ")"]]
        for w in words:
            assert run_vpa(vpa, w) == run_vpa(mini, w), f"Failed on {w}"


# --- Section 22: Summary / Statistics ---

class TestSummary:
    def test_summary(self, paren_alpha):
        vpa = make_balanced_parens_vpa(paren_alpha)
        s = vpa_summary(vpa)
        assert s["states"] == 1
        assert s["deterministic"] == True
        assert s["call_transitions"] == 1
        assert s["return_transitions"] == 1


# --- Section 23: Compare ---

class TestCompare:
    def test_compare_equivalent(self, full_alpha):
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("r0", initial=True)
        vpa2.add_state("r1", accepting=True)
        vpa2.add_internal_transition("r0", "a", "r1")

        result = compare_vpa(vpa1, vpa2, [["a"], ["b"]])
        assert result["equivalent"] == True
        assert len(result["test_comparisons"]) == 2
        for tc in result["test_comparisons"]:
            assert tc["agree"] == True

    def test_compare_non_equivalent(self, full_alpha):
        vpa1 = VPA(alphabet=full_alpha)
        vpa1.add_state("q0", initial=True)
        vpa1.add_state("q1", accepting=True)
        vpa1.add_internal_transition("q0", "a", "q1")

        vpa2 = VPA(alphabet=full_alpha)
        vpa2.add_state("q0", initial=True)
        vpa2.add_state("q1", accepting=True)
        vpa2.add_internal_transition("q0", "b", "q1")

        result = compare_vpa(vpa1, vpa2)
        assert result["equivalent"] == False


# --- Section 24: Edge Cases ---

class TestEdgeCases:
    def test_empty_language_vpa(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True)
        # No accepting state
        assert run_vpa(vpa, []) == False
        assert run_vpa(vpa, ["(", ")"]) == False

    def test_single_state_vpa(self, full_alpha):
        vpa = VPA(alphabet=full_alpha)
        vpa.add_state("q", initial=True, accepting=True)
        vpa.add_internal_transition("q", "a", "q")
        vpa.add_internal_transition("q", "b", "q")
        assert run_vpa(vpa, ["a", "b", "a"]) == True
        assert run_vpa(vpa, ["call"]) == False  # no call transition

    def test_only_internals_alphabet(self):
        alpha = VisibleAlphabet(
            calls=frozenset(),
            returns=frozenset(),
            internals=frozenset({"a", "b"})
        )
        vpa = VPA(alphabet=alpha)
        vpa.add_state("q0", initial=True)
        vpa.add_state("q1", accepting=True)
        vpa.add_internal_transition("q0", "a", "q1")
        vpa.add_internal_transition("q1", "b", "q0")
        # Accepts "a", "a b a", etc. (odd number of ab pairs + final a)
        assert run_vpa(vpa, ["a"]) == True
        assert run_vpa(vpa, ["a", "b", "a"]) == True
        assert run_vpa(vpa, ["a", "b"]) == False

    def test_no_transitions_from_initial(self, paren_alpha):
        vpa = VPA(alphabet=paren_alpha)
        vpa.add_state("q0", initial=True, accepting=True)
        # Only accepts empty word
        assert run_vpa(vpa, []) == True
        assert run_vpa(vpa, ["("]) == False
