"""Tests for V091: Regular Tree Model Checking"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V090_tree_transducers'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V089_tree_automata'))

from tree_automata import (
    Symbol, RankedAlphabet, Tree, tree, make_alphabet, make_buta,
    BottomUpTreeAutomaton, buta_union, buta_intersection, buta_complement,
    buta_is_equivalent, check_language_emptiness,
)
from tree_transducers import (
    OutputTemplate, out, out_var, BUTTRule, BottomUpTreeTransducer,
    identity_transducer, rewrite_to_butt, transform_tree,
)
from tree_model_checking import (
    TreeTransitionSystem, VerificationResult, ModelCheckResult,
    CounterexampleTrace, make_tree_system, make_init_from_trees,
    make_bad_from_pattern,
    forward_reachability, backward_reachability, bounded_check,
    accelerated_forward, check_safety, check_reachability,
    check_invariant, transducer_image,
    compare_methods, system_stats, model_check_summary,
    verify_tree_transform_preserves, widen_automata,
    _buta_from_tree_set, _empty_automaton,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def make_nat_alphabet():
    """Natural numbers: z (zero), s(x) (successor)."""
    return make_alphabet(("z", 0), ("s", 1))


def nat(n):
    """Build the natural number n as a tree: s(s(...(z)...))"""
    t = tree("z")
    for _ in range(n):
        t = tree("s", t)
    return t


def make_increment_transducer(alphabet):
    """Transducer that increments a natural number: n -> s(n)."""
    td = BottomUpTreeTransducer(alphabet, alphabet)
    # State: q (just one state, wraps everything in s())
    td.states = {"q"}
    td.final_states = {"q"}
    # z -> q with output s(z)
    td.add_rule("z", (), "q", out("s", out("z")))
    # s(q) -> q with output s($0)
    td.add_rule("s", ("q",), "q", out("s", out_var("0")))
    return td


def make_double_transducer(alphabet):
    """Transducer that doubles: n -> 2n (adds n successors)."""
    td = BottomUpTreeTransducer(alphabet, alphabet)
    td.states = {"q"}
    td.final_states = {"q"}
    # z -> q with output z
    td.add_rule("z", (), "q", out("z"))
    # s(q) -> q with output s(s($0)) -- adds two s for each s
    td.add_rule("s", ("q",), "q", out("s", out("s", out_var("0"))))
    return td


def make_binary_alphabet():
    """Binary trees: leaf, node(l, r)."""
    return make_alphabet(("leaf", 0), ("node", 2))


# ---------------------------------------------------------------
# Section 1: Data types and construction
# ---------------------------------------------------------------

class TestDataTypes:
    def test_tree_transition_system_creation(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        tts = make_tree_system(alph, init, td)
        assert tts.alphabet is alph
        assert tts.init is init
        assert tts.transition is td
        assert tts.bad is None

    def test_verification_result_enum(self):
        assert VerificationResult.SAFE.value == "safe"
        assert VerificationResult.UNSAFE.value == "unsafe"
        assert VerificationResult.UNKNOWN.value == "unknown"

    def test_model_check_result(self):
        r = ModelCheckResult(result=VerificationResult.SAFE, steps=3)
        assert r.result == VerificationResult.SAFE
        assert r.steps == 3
        assert r.counterexample is None
        assert r.invariant is None

    def test_counterexample_trace(self):
        t0, t1 = nat(0), nat(1)
        cex = CounterexampleTrace(trees=[t0, t1], length=1)
        assert cex.length == 1
        assert len(cex.trees) == 2

    def test_make_init_from_trees(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0), nat(1)])
        assert init.accepts(nat(0))
        assert init.accepts(nat(1))
        assert not init.accepts(nat(2))

    def test_make_bad_from_pattern(self):
        alph = make_nat_alphabet()
        # Bad = exactly zero
        bad = make_bad_from_pattern(alph,
            [("z", [], "qz")],
            ["qz"],
            ["qz"])
        assert bad.accepts(nat(0))
        assert not bad.accepts(nat(1))


# ---------------------------------------------------------------
# Section 2: Transducer image computation
# ---------------------------------------------------------------

class TestTransducerImage:
    def test_image_of_singleton(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        img = transducer_image(td, init)
        assert img.accepts(nat(1))
        assert not img.accepts(nat(0))

    def test_image_of_set(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0), nat(1)])
        td = make_increment_transducer(alph)
        img = transducer_image(td, init)
        assert img.accepts(nat(1))
        assert img.accepts(nat(2))
        assert not img.accepts(nat(0))

    def test_image_double(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1), nat(2)])
        td = make_double_transducer(alph)
        img = transducer_image(td, init)
        assert img.accepts(nat(2))
        assert img.accepts(nat(4))

    def test_empty_image(self):
        alph = make_nat_alphabet()
        init = _empty_automaton(alph)
        td = make_increment_transducer(alph)
        img = transducer_image(td, init)
        emp = check_language_emptiness(img)
        assert emp["empty"]


# ---------------------------------------------------------------
# Section 3: Forward reachability -- safe systems
# ---------------------------------------------------------------

class TestForwardSafe:
    def test_identity_is_safe(self):
        """Identity transducer: init stays init."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        # Bad = zero
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        result = forward_reachability(tts, max_steps=5)
        assert result.result == VerificationResult.SAFE

    def test_increment_avoids_zero(self):
        """Starting from 1, incrementing never reaches 0."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        result = forward_reachability(tts, max_steps=10)
        assert result.result == VerificationResult.SAFE

    def test_increment_from_zero_bounded_safe(self):
        """Starting from 0, incrementing doesn't reach 100 within 5 steps."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        # Bad = nat(100) -- unreachable in 5 steps
        bad = make_init_from_trees(alph, [nat(100)])
        tts = make_tree_system(alph, init, td, bad)
        # Forward won't converge (diverges), so use bounded check
        result = bounded_check(tts, bound=5)
        assert result.result == VerificationResult.SAFE


# ---------------------------------------------------------------
# Section 4: Forward reachability -- unsafe systems
# ---------------------------------------------------------------

class TestForwardUnsafe:
    def test_increment_reaches_target(self):
        """Starting from 0, incrementing reaches 3."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(3)])
        tts = make_tree_system(alph, init, td, bad)
        result = forward_reachability(tts, max_steps=10)
        assert result.result == VerificationResult.UNSAFE

    def test_double_reaches_target(self):
        """Starting from 1, doubling reaches 2."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = make_double_transducer(alph)
        bad = make_init_from_trees(alph, [nat(2)])
        tts = make_tree_system(alph, init, td, bad)
        result = forward_reachability(tts, max_steps=5)
        assert result.result == VerificationResult.UNSAFE

    def test_init_is_bad(self):
        """Initial state is already bad."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        result = forward_reachability(tts, max_steps=5)
        assert result.result == VerificationResult.UNSAFE
        assert result.steps <= 1

    def test_counterexample_exists(self):
        """Unsafe result should have counterexample."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(2)])
        tts = make_tree_system(alph, init, td, bad)
        result = forward_reachability(tts, max_steps=10)
        assert result.result == VerificationResult.UNSAFE
        assert result.counterexample is not None


# ---------------------------------------------------------------
# Section 5: Bounded model checking
# ---------------------------------------------------------------

class TestBoundedCheck:
    def test_bounded_safe(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(10)])
        tts = make_tree_system(alph, init, td, bad)
        result = bounded_check(tts, bound=5)
        assert result.result == VerificationResult.SAFE

    def test_bounded_unsafe(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(3)])
        tts = make_tree_system(alph, init, td, bad)
        result = bounded_check(tts, bound=5)
        assert result.result == VerificationResult.UNSAFE

    def test_bounded_insufficient_depth(self):
        """Target reachable in 5 steps, but bound is 2."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(5)])
        tts = make_tree_system(alph, init, td, bad)
        result = bounded_check(tts, bound=2)
        assert result.result == VerificationResult.SAFE  # safe within bound

    def test_bounded_stats(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(2)])
        tts = make_tree_system(alph, init, td, bad)
        result = bounded_check(tts, bound=5)
        assert "step_details" in result.stats

    def test_bounded_requires_bad(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        tts = make_tree_system(alph, init, td)
        with pytest.raises(ValueError):
            bounded_check(tts, bound=5)


# ---------------------------------------------------------------
# Section 6: Backward reachability
# ---------------------------------------------------------------

class TestBackwardReachability:
    def test_backward_safe(self):
        """Bad state not reachable backward from init."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(100)])
        tts = make_tree_system(alph, init, td, bad)
        result = backward_reachability(tts, max_steps=3)
        # 100 is not reachable from 0 in small steps
        assert result.result in (VerificationResult.SAFE, VerificationResult.UNKNOWN)

    def test_backward_requires_bad(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        tts = make_tree_system(alph, init, td)
        with pytest.raises(ValueError):
            backward_reachability(tts)


# ---------------------------------------------------------------
# Section 7: Invariant checking
# ---------------------------------------------------------------

class TestInvariantChecking:
    def test_init_is_trivial_invariant_for_identity(self):
        """For identity transducer, init set is inductive."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        tts = make_tree_system(alph, init, td)
        result = check_invariant(tts, init)
        assert result["is_invariant"]
        assert len(result["violations"]) == 0

    def test_invariant_initiation_violation(self):
        """Invariant that doesn't contain init."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0), nat(1)])
        td = identity_transducer(alph)
        tts = make_tree_system(alph, init, td)
        # Invariant only contains nat(1)
        inv = make_init_from_trees(alph, [nat(1)])
        result = check_invariant(tts, inv)
        assert not result["is_invariant"]
        violations = [v["type"] for v in result["violations"]]
        assert "initiation" in violations

    def test_invariant_safety_violation(self):
        """Invariant that overlaps bad."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(1)])
        tts = make_tree_system(alph, init, td, bad)
        # Invariant = init = bad -> safety violation
        inv = make_init_from_trees(alph, [nat(1)])
        result = check_invariant(tts, inv)
        assert not result["is_invariant"]
        violations = [v["type"] for v in result["violations"]]
        assert "safety" in violations

    def test_invariant_consecution_violation(self):
        """Invariant not closed under transition."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        tts = make_tree_system(alph, init, td)
        # Invariant = just {0} -- increment produces 1 which is not in {0}
        inv = make_init_from_trees(alph, [nat(0)])
        result = check_invariant(tts, inv)
        assert not result["is_invariant"]
        violations = [v["type"] for v in result["violations"]]
        assert "consecution" in violations


# ---------------------------------------------------------------
# Section 8: Accelerated forward reachability (widening)
# ---------------------------------------------------------------

class TestAccelerated:
    def test_accelerated_safe(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        result = accelerated_forward(tts, max_steps=10, widen_after=3)
        assert result.result == VerificationResult.SAFE

    def test_accelerated_unsafe(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(2)])
        tts = make_tree_system(alph, init, td, bad)
        result = accelerated_forward(tts, max_steps=10, widen_after=3)
        assert result.result == VerificationResult.UNSAFE


# ---------------------------------------------------------------
# Section 9: check_safety dispatcher
# ---------------------------------------------------------------

class TestCheckSafety:
    def test_forward_method(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        r = check_safety(tts, method="forward")
        assert r.result == VerificationResult.SAFE

    def test_bounded_method(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(3)])
        tts = make_tree_system(alph, init, td, bad)
        r = check_safety(tts, max_steps=5, method="bounded")
        assert r.result == VerificationResult.UNSAFE

    def test_accelerated_method(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        r = check_safety(tts, method="accelerated")
        assert r.result == VerificationResult.SAFE

    def test_unknown_method_raises(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = identity_transducer(alph)
        tts = make_tree_system(alph, init, td)
        with pytest.raises(ValueError):
            check_safety(tts, method="nonexistent")


# ---------------------------------------------------------------
# Section 10: check_reachability
# ---------------------------------------------------------------

class TestCheckReachability:
    def test_reachable_target(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        target = make_init_from_trees(alph, [nat(2)])
        tts = make_tree_system(alph, init, td)
        r = check_reachability(tts, target, max_steps=10)
        # UNSAFE means target IS reachable
        assert r.result == VerificationResult.UNSAFE

    def test_unreachable_target(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(5)])
        td = make_increment_transducer(alph)
        target = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td)
        r = check_reachability(tts, target, max_steps=5)
        assert r.result == VerificationResult.SAFE


# ---------------------------------------------------------------
# Section 11: Binary tree examples
# ---------------------------------------------------------------

class TestBinaryTrees:
    def test_binary_tree_identity(self):
        """Identity transducer on binary trees is safe."""
        alph = make_binary_alphabet()
        t1 = tree("node", tree("leaf"), tree("leaf"))
        init = make_init_from_trees(alph, [t1])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [tree("leaf")])
        tts = make_tree_system(alph, init, td, bad)
        r = forward_reachability(tts, max_steps=5)
        assert r.result == VerificationResult.SAFE

    def test_binary_tree_rewrite(self):
        """Rewrite leaf -> node(leaf, leaf) makes tree grow."""
        alph = make_binary_alphabet()
        td = BottomUpTreeTransducer(alph, alph)
        td.states = {"q"}
        td.final_states = {"q"}
        # leaf -> q with output node(leaf, leaf)
        td.add_rule("leaf", (), "q", out("node", out("leaf"), out("leaf")))
        # node(q, q) -> q with output node($0, $1)
        td.add_rule("node", ("q", "q"), "q", out("node", out_var("0"), out_var("1")))

        init = make_init_from_trees(alph, [tree("leaf")])
        # After one step: node(leaf, leaf)
        target = make_init_from_trees(alph, [tree("node", tree("leaf"), tree("leaf"))])
        tts = make_tree_system(alph, init, td, target)
        r = forward_reachability(tts, max_steps=3)
        assert r.result == VerificationResult.UNSAFE

    def test_binary_tree_bounded_growth(self):
        """Bounded check catches growth within bound."""
        alph = make_binary_alphabet()
        td = BottomUpTreeTransducer(alph, alph)
        td.states = {"q"}
        td.final_states = {"q"}
        td.add_rule("leaf", (), "q", out("node", out("leaf"), out("leaf")))
        td.add_rule("node", ("q", "q"), "q", out("node", out_var("0"), out_var("1")))

        init = make_init_from_trees(alph, [tree("leaf")])
        # Full binary tree of depth 2
        target = make_init_from_trees(alph, [
            tree("node",
                 tree("node", tree("leaf"), tree("leaf")),
                 tree("node", tree("leaf"), tree("leaf")))
        ])
        tts = make_tree_system(alph, init, td, target)
        r = bounded_check(tts, bound=3)
        assert r.result == VerificationResult.UNSAFE


# ---------------------------------------------------------------
# Section 12: Widening
# ---------------------------------------------------------------

class TestWidening:
    def test_widen_union_minimizes(self):
        alph = make_nat_alphabet()
        a1 = make_init_from_trees(alph, [nat(0), nat(1)])
        a2 = make_init_from_trees(alph, [nat(0), nat(1), nat(2)])
        w = widen_automata(a1, a2)
        # Widened should accept at least everything in a2
        assert w.accepts(nat(0))
        assert w.accepts(nat(1))
        assert w.accepts(nat(2))

    def test_widen_monotone(self):
        """Widened language is at least as large as both inputs."""
        alph = make_nat_alphabet()
        a1 = make_init_from_trees(alph, [nat(0)])
        a2 = make_init_from_trees(alph, [nat(0), nat(1)])
        w = widen_automata(a1, a2)
        # w should accept at least a1 and a2
        is_sub1, _ = buta_is_equivalent(buta_intersection(a1, buta_complement(w)),
                                         _empty_automaton(alph))
        # Just check concrete acceptance
        assert w.accepts(nat(0))
        assert w.accepts(nat(1))


# ---------------------------------------------------------------
# Section 13: System stats and comparison
# ---------------------------------------------------------------

class TestStatsAndComparison:
    def test_system_stats(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(5)])
        tts = make_tree_system(alph, init, td, bad)
        stats = system_stats(tts)
        assert "alphabet_size" in stats
        assert stats["alphabet_size"] == 2
        assert stats["has_bad"]
        assert "bad_states" in stats

    def test_system_stats_no_bad(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        tts = make_tree_system(alph, init, td)
        stats = system_stats(tts)
        assert not stats["has_bad"]

    def test_compare_methods(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(3)])
        tts = make_tree_system(alph, init, td, bad)
        comp = compare_methods(tts, max_steps=10)
        assert "forward" in comp
        assert "bounded" in comp
        assert "accelerated" in comp
        # All should agree on unsafe
        for method in ["forward", "bounded", "accelerated"]:
            assert comp[method]["result"] == "unsafe"

    def test_model_check_summary(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        summary = model_check_summary(tts, max_steps=5)
        assert "system" in summary
        assert "methods" in summary
        assert "consensus" in summary
        assert summary["consensus"] == "safe"


# ---------------------------------------------------------------
# Section 14: Empty automaton / edge cases
# ---------------------------------------------------------------

class TestEdgeCases:
    def test_empty_init(self):
        """Empty initial set -- trivially safe."""
        alph = make_nat_alphabet()
        init = _empty_automaton(alph)
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        r = forward_reachability(tts, max_steps=5)
        assert r.result == VerificationResult.SAFE

    def test_buta_from_tree_set_deduplication(self):
        """Same tree twice should produce same result."""
        alph = make_nat_alphabet()
        a = _buta_from_tree_set(alph, [nat(2), nat(2)])
        assert a.accepts(nat(2))
        assert not a.accepts(nat(1))

    def test_single_step_fixpoint(self):
        """Identity transducer converges in one step."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = identity_transducer(alph)
        tts = make_tree_system(alph, init, td)
        r = forward_reachability(tts, max_steps=5)
        assert r.result == VerificationResult.SAFE
        assert r.steps <= 2  # should converge quickly


# ---------------------------------------------------------------
# Section 15: verify_tree_transform_preserves
# ---------------------------------------------------------------

class TestVerifyTransformPreserves:
    def test_identity_preserves_property(self):
        """Identity preserves any property."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1), nat(2)])
        td = identity_transducer(alph)
        # Property: not zero
        prop = make_init_from_trees(alph, [nat(1), nat(2), nat(3), nat(4), nat(5)])
        result = verify_tree_transform_preserves(alph, init, td, prop)
        assert result["safe"]

    def test_increment_violates_bounded_property(self):
        """Incrementing eventually leaves a bounded set."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        # Property: value <= 2
        prop = make_init_from_trees(alph, [nat(0), nat(1), nat(2)])
        result = verify_tree_transform_preserves(alph, init, td, prop, max_steps=10)
        assert not result["safe"]

    def test_result_has_expected_keys(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = identity_transducer(alph)
        prop = make_init_from_trees(alph, [nat(0)])
        result = verify_tree_transform_preserves(alph, init, td, prop)
        assert "result" in result
        assert "steps" in result
        assert "safe" in result
        assert "counterexample" in result
        assert "invariant_found" in result


# ---------------------------------------------------------------
# Section 16: Multi-step trace reconstruction
# ---------------------------------------------------------------

class TestTraceReconstruction:
    def test_trace_from_unsafe_result(self):
        """Unsafe forward check should produce a trace."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(2)])
        tts = make_tree_system(alph, init, td, bad)
        r = forward_reachability(tts, max_steps=10)
        assert r.result == VerificationResult.UNSAFE
        assert r.counterexample is not None
        assert r.counterexample.length > 0

    def test_immediate_violation_short_trace(self):
        """Init = Bad means step 0 or 1 violation."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = make_increment_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        r = forward_reachability(tts, max_steps=5)
        assert r.result == VerificationResult.UNSAFE
        assert r.steps <= 1


# ---------------------------------------------------------------
# Section 17: Complex multi-symbol systems
# ---------------------------------------------------------------

class TestMultiSymbol:
    def test_three_symbol_system(self):
        """System with a, b(x), c(x, y)."""
        alph = make_alphabet(("a", 0), ("b", 1), ("c", 2))
        init = make_init_from_trees(alph, [tree("a")])

        # Transducer: a -> b(a), b(x) -> c(x, a), c(x,y) -> c(x, y)
        td = BottomUpTreeTransducer(alph, alph)
        td.states = {"q"}
        td.final_states = {"q"}
        td.add_rule("a", (), "q", out("b", out("a")))
        td.add_rule("b", ("q",), "q", out("c", out_var("0"), out("a")))
        td.add_rule("c", ("q", "q"), "q", out("c", out_var("0"), out_var("1")))

        # a -> b(a) -> c(b(a), a)
        # Target: c(b(a), a) -- reachable after 2 steps
        target = make_init_from_trees(alph, [tree("c", tree("b", tree("a")), tree("a"))])
        tts = make_tree_system(alph, init, td, target)
        r = bounded_check(tts, bound=3)
        assert r.result == VerificationResult.UNSAFE

    def test_three_symbol_safe(self):
        """b(a) never becomes just a under the transducer."""
        alph = make_alphabet(("a", 0), ("b", 1), ("c", 2))
        init = make_init_from_trees(alph, [tree("b", tree("a"))])

        td = BottomUpTreeTransducer(alph, alph)
        td.states = {"q"}
        td.final_states = {"q"}
        td.add_rule("a", (), "q", out("a"))
        td.add_rule("b", ("q",), "q", out("c", out_var("0"), out("a")))
        td.add_rule("c", ("q", "q"), "q", out("c", out_var("0"), out_var("1")))

        bad = make_init_from_trees(alph, [tree("a")])
        tts = make_tree_system(alph, init, td, bad)
        r = bounded_check(tts, bound=3)
        assert r.result == VerificationResult.SAFE


# ---------------------------------------------------------------
# Section 18: Forward reachability invariant
# ---------------------------------------------------------------

class TestForwardInvariant:
    def test_safe_result_has_invariant(self):
        """Safe forward result should include computed invariant."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        r = forward_reachability(tts, max_steps=5)
        assert r.result == VerificationResult.SAFE
        assert r.invariant is not None
        # The invariant should contain all reachable states
        assert r.invariant.accepts(nat(1))

    def test_invariant_is_valid(self):
        """Computed invariant should pass check_invariant."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(1)])
        td = identity_transducer(alph)
        bad = make_init_from_trees(alph, [nat(0)])
        tts = make_tree_system(alph, init, td, bad)
        r = forward_reachability(tts, max_steps=5)
        assert r.invariant is not None
        check = check_invariant(tts, r.invariant)
        assert check["is_invariant"]


# ---------------------------------------------------------------
# Section 19: No bad states (reachability only)
# ---------------------------------------------------------------

class TestNoBadStates:
    def test_forward_no_bad_converges(self):
        """Forward reachability without bad states just computes fixpoint."""
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = identity_transducer(alph)
        tts = make_tree_system(alph, init, td)
        r = forward_reachability(tts, max_steps=5)
        assert r.result == VerificationResult.SAFE  # converges = safe
        assert r.invariant is not None

    def test_accelerated_no_bad(self):
        alph = make_nat_alphabet()
        init = make_init_from_trees(alph, [nat(0)])
        td = identity_transducer(alph)
        tts = make_tree_system(alph, init, td)
        r = accelerated_forward(tts, max_steps=5, widen_after=2)
        assert r.result == VerificationResult.SAFE


# ---------------------------------------------------------------
# Section 20: Repr and formatting
# ---------------------------------------------------------------

class TestRepr:
    def test_model_check_result_repr(self):
        r = ModelCheckResult(result=VerificationResult.SAFE, steps=3)
        s = repr(r)
        assert "safe" in s
        assert "3" in s

    def test_counterexample_repr(self):
        cex = CounterexampleTrace(trees=[nat(0), nat(1)], length=1)
        s = repr(cex)
        assert "Trace" in s
        assert "1 steps" in s

    def test_unsafe_result_repr(self):
        cex = CounterexampleTrace(trees=[nat(0)], length=1)
        r = ModelCheckResult(result=VerificationResult.UNSAFE,
                             counterexample=cex, steps=1)
        s = repr(r)
        assert "unsafe" in s
        assert "Trace" in s
