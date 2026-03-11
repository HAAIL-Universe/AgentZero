"""Tests for V158: Symbolic Mu-Calculus Model Checking"""

import sys, os, pytest
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V157_mu_calculus'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))

from symbolic_mu_calculus import (
    SymbolicLTS, SymbolicMuChecker, SymMCResult,
    lts_to_symbolic, make_symbolic_lts, boolean_ts_to_symbolic_lts,
    symbolic_check, symbolic_check_lts, check_state_symbolic,
    compare_with_explicit, check_ctl_symbolic, check_ctl_on_lts,
    batch_symbolic_check, symbolic_reachable, check_safety_symbolic,
    make_counter_lts, make_mutex_lts,
    symbolic_mu_summary, full_analysis,
)
from mu_calculus import (
    Prop, Var, TT, FF, Not, And, Or, Diamond, Box, Mu, Nu,
    LTS, make_lts, parse_mu, eval_formula,
    ctl_EF, ctl_AG, ctl_AF, ctl_EG, ctl_EU, ctl_AU, ctl_EX, ctl_AX,
)
from bdd_model_checker import BDD, make_boolean_ts


# ---------------------------------------------------------------------------
# Helper: small explicit LTS for testing
# ---------------------------------------------------------------------------

def traffic_light_lts():
    """3-state traffic light: red -> green -> yellow -> red."""
    return make_lts(3, [
        (0, "change", 1),  # red -> green
        (1, "change", 2),  # green -> yellow
        (2, "change", 0),  # yellow -> red
    ], {0: {"red"}, 1: {"green"}, 2: {"yellow"}})


def simple_lts():
    """Simple 4-state LTS with branching."""
    return make_lts(4, [
        (0, "a", 1), (0, "b", 2),
        (1, "a", 3), (2, "b", 3),
        (3, "a", 0),
    ], {0: {"start"}, 1: {"left"}, 2: {"right"}, 3: {"end"}})


def self_loop_lts():
    """2-state LTS with self-loop."""
    return make_lts(2, [
        (0, "a", 1), (1, "a", 1),
    ], {0: {"init"}, 1: {"loop"}})


# ===========================================================================
# 1. LTS Conversion
# ===========================================================================

class TestLTSConversion:
    def test_traffic_light_conversion(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        assert slts.n_bits == 2  # ceil(log2(3)) = 2
        assert len(slts.state_vars) == 2
        assert len(slts.next_vars) == 2
        assert "red" in slts.labels
        assert "green" in slts.labels
        assert "yellow" in slts.labels
        assert "change" in slts.trans

    def test_simple_conversion(self):
        lts = simple_lts()
        slts = lts_to_symbolic(lts)
        assert slts.n_bits == 2  # ceil(log2(4)) = 2
        assert "a" in slts.trans
        assert "b" in slts.trans

    def test_self_loop_conversion(self):
        lts = self_loop_lts()
        slts = lts_to_symbolic(lts)
        assert slts.n_bits == 1
        assert "a" in slts.trans

    def test_single_state(self):
        lts = make_lts(1, [(0, "a", 0)], {0: {"only"}})
        slts = lts_to_symbolic(lts)
        assert slts.n_bits == 1
        assert "only" in slts.labels

    def test_valid_states_mask(self):
        lts = traffic_light_lts()  # 3 states, 2 bits -> 1 phantom
        slts = lts_to_symbolic(lts)
        assert slts.valid_states is not None
        mc = SymbolicMuChecker(slts)
        valid_count = mc.sat_count(slts.valid_states)
        assert valid_count == 3


# ===========================================================================
# 2. Basic Formula Evaluation
# ===========================================================================

class TestBasicFormulas:
    def test_tt_all_states(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, TT())
        assert r.sat_count == 3

    def test_ff_no_states(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, FF())
        assert r.sat_count == 0

    def test_proposition(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, Prop("red"))
        assert r.sat_count == 1
        assert 0 in r.sat_states  # state 0 is red

    def test_proposition_green(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, Prop("green"))
        assert r.sat_count == 1

    def test_nonexistent_prop(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, Prop("blue"))
        assert r.sat_count == 0

    def test_negation(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, Not(Prop("red")))
        assert r.sat_count == 2  # green and yellow

    def test_conjunction(self):
        lts = simple_lts()
        r = symbolic_check_lts(lts, And(Prop("start"), Not(Prop("end"))))
        assert r.sat_count == 1

    def test_disjunction(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, Or(Prop("red"), Prop("green")))
        assert r.sat_count == 2


# ===========================================================================
# 3. Modal Operators (Diamond, Box)
# ===========================================================================

class TestModalOperators:
    def test_diamond_specific_action(self):
        lts = traffic_light_lts()
        # <change>green: states that can reach green in one step
        r = symbolic_check_lts(lts, Diamond("change", Prop("green")))
        assert r.sat_count == 1
        assert 0 in r.sat_states  # red -> green

    def test_diamond_any_action(self):
        lts = simple_lts()
        # <>end: states that can reach end in one step
        r = symbolic_check_lts(lts, Diamond(None, Prop("end")))
        assert 1 in r.sat_states  # left -> end via a
        assert 2 in r.sat_states  # right -> end via b

    def test_box_specific_action(self):
        lts = traffic_light_lts()
        # [change]green: all successors via change satisfy green
        r = symbolic_check_lts(lts, Box("change", Prop("green")))
        assert 0 in r.sat_states  # red's only change-successor is green

    def test_box_vacuous(self):
        lts = simple_lts()
        # [c]ff: vacuously true for states with no c-transitions
        r = symbolic_check_lts(lts, Box("c", FF()))
        assert r.sat_count == 4  # no state has c-transitions

    def test_diamond_no_successors(self):
        lts = make_lts(2, [(0, "a", 1)], {0: {"start"}, 1: {"end"}})
        # State 1 has no transitions -> <a>tt should not include state 1
        r = symbolic_check_lts(lts, Diamond("a", TT()))
        assert r.sat_count == 1
        assert 0 in r.sat_states

    def test_nested_diamond(self):
        lts = traffic_light_lts()
        # <change><change>red: can reach red in 2 steps = green
        r = symbolic_check_lts(lts, Diamond("change", Diamond("change", Prop("red"))))
        assert r.sat_count == 1
        assert 1 in r.sat_states  # green -> yellow -> red


# ===========================================================================
# 4. Fixpoint Operators (Mu, Nu)
# ===========================================================================

class TestFixpoints:
    def test_mu_ef_reachability(self):
        lts = traffic_light_lts()
        # mu X. (red | <change>X) = EF red = all states (cyclic)
        ef_red = Mu("X", Or(Prop("red"), Diamond("change", Var("X"))))
        r = symbolic_check_lts(lts, ef_red)
        assert r.sat_count == 3  # all states can reach red

    def test_nu_ag_safety(self):
        lts = traffic_light_lts()
        # nu X. (tt & [change]X) = AG tt = all states
        ag_tt = Nu("X", And(TT(), Box("change", Var("X"))))
        r = symbolic_check_lts(lts, ag_tt)
        assert r.sat_count == 3

    def test_mu_empty_start(self):
        lts = make_lts(2, [(0, "a", 1)], {1: {"target"}})
        # mu X. (target | <>X): reachable from target
        ef_target = Mu("X", Or(Prop("target"), Diamond(None, Var("X"))))
        r = symbolic_check_lts(lts, ef_target)
        assert 0 in r.sat_states  # can reach target from 0
        assert 1 in r.sat_states  # is target

    def test_nu_greatest_fixpoint(self):
        lts = self_loop_lts()
        # nu X. (<a>X): states where we can keep doing a forever
        eg = Nu("X", Diamond("a", Var("X")))
        r = symbolic_check_lts(lts, eg)
        assert 1 in r.sat_states  # self-loop: can do a forever
        # State 0 goes to 1 which can loop, so 0 is also in EG
        assert r.sat_count >= 1

    def test_mu_ef_unreachable(self):
        lts = make_lts(3, [(0, "a", 1), (2, "a", 2)], {0: {"start"}, 2: {"isolated"}})
        # mu X. (isolated | <>X): states that can reach isolated
        ef_iso = Mu("X", Or(Prop("isolated"), Diamond(None, Var("X"))))
        r = symbolic_check_lts(lts, ef_iso)
        assert 2 in r.sat_states
        assert 0 not in r.sat_states  # 0 -> 1 (dead end), can't reach 2


# ===========================================================================
# 5. CTL Encodings
# ===========================================================================

class TestCTLEncodings:
    def test_ef_via_mu(self):
        lts = traffic_light_lts()
        ef = ctl_EF(Prop("green"))
        r = symbolic_check_lts(lts, ef)
        assert r.sat_count == 3  # all can reach green

    def test_ag_via_nu(self):
        lts = traffic_light_lts()
        # AG(red | green | yellow) = always one of the three
        ag = ctl_AG(Or(Prop("red"), Or(Prop("green"), Prop("yellow"))))
        r = symbolic_check_lts(lts, ag)
        assert r.sat_count == 3

    def test_af_via_mu(self):
        lts = traffic_light_lts()
        af_red = ctl_AF(Prop("red"))
        r = symbolic_check_lts(lts, af_red)
        assert r.sat_count == 3  # all paths inevitably reach red

    def test_eg_via_nu(self):
        lts = self_loop_lts()
        eg_loop = ctl_EG(Prop("loop"))
        r = symbolic_check_lts(lts, eg_loop)
        assert 1 in r.sat_states  # self-loop on loop state

    def test_eu_via_mu(self):
        lts = traffic_light_lts()
        # E[NOT red U green]: path from non-red to green
        eu = ctl_EU(Not(Prop("red")), Prop("green"))
        r = symbolic_check_lts(lts, eu)
        assert 1 in r.sat_states  # green satisfies psi immediately

    def test_ex_via_diamond(self):
        lts = traffic_light_lts()
        ex_green = ctl_EX(Prop("green"))
        r = symbolic_check_lts(lts, ex_green)
        assert 0 in r.sat_states  # red -> green


# ===========================================================================
# 6. Comparison with Explicit Model Checking
# ===========================================================================

class TestComparison:
    def test_traffic_light_agree(self):
        lts = traffic_light_lts()
        comp = compare_with_explicit(lts, Prop("red"))
        assert comp["agree"]
        assert comp["explicit_count"] == comp["symbolic_count"]

    def test_ef_agree(self):
        lts = traffic_light_lts()
        ef = ctl_EF(Prop("green"))
        comp = compare_with_explicit(lts, ef)
        assert comp["agree"]

    def test_ag_agree(self):
        lts = simple_lts()
        ag = ctl_AG(Or(Prop("start"), Or(Prop("left"), Or(Prop("right"), Prop("end")))))
        comp = compare_with_explicit(lts, ag)
        assert comp["agree"]

    def test_diamond_agree(self):
        lts = simple_lts()
        comp = compare_with_explicit(lts, Diamond("a", Prop("end")))
        assert comp["agree"]

    def test_box_agree(self):
        lts = simple_lts()
        comp = compare_with_explicit(lts, Box("a", Prop("left")))
        assert comp["agree"]

    def test_nested_fixpoint_agree(self):
        lts = traffic_light_lts()
        # nu X. mu Y. (red & X) | (<>Y)
        f = Nu("X", Mu("Y", Or(And(Prop("red"), Var("X")), Diamond(None, Var("Y")))))
        comp = compare_with_explicit(lts, f)
        assert comp["agree"]


# ===========================================================================
# 7. Counter System
# ===========================================================================

class TestCounterSystem:
    def test_counter_creation(self):
        slts = make_counter_lts(3, {"zero": lambda x: x == 0, "even": lambda x: x % 2 == 0})
        assert slts.n_bits == 3
        assert "zero" in slts.labels
        assert "even" in slts.labels

    def test_counter_reachability(self):
        slts = make_counter_lts(3, {"zero": lambda x: x == 0})
        reached, iters = symbolic_reachable(slts)
        mc = SymbolicMuChecker(slts)
        count = mc.sat_count(reached)
        assert count == 8  # all 2^3 states reachable

    def test_counter_ef_zero(self):
        slts = make_counter_lts(3, {"zero": lambda x: x == 0})
        ef = ctl_EF(Prop("zero"))
        r = symbolic_check(slts, ef)
        assert r.sat_count == 8  # all states can reach zero

    def test_counter_ag_eventually_zero(self):
        slts = make_counter_lts(3, {"zero": lambda x: x == 0})
        af = ctl_AF(Prop("zero"))
        r = symbolic_check(slts, af)
        assert r.sat_count == 8  # all paths reach zero (cyclic counter)

    def test_counter_even_states(self):
        slts = make_counter_lts(3, {"even": lambda x: x % 2 == 0})
        r = symbolic_check(slts, Prop("even"))
        assert r.sat_count == 4  # 0, 2, 4, 6

    def test_counter_diamond_even(self):
        slts = make_counter_lts(3, {"even": lambda x: x % 2 == 0})
        # <tick>even: odd states (predecessor of even states)
        r = symbolic_check(slts, Diamond("tick", Prop("even")))
        assert r.sat_count == 4  # 1, 3, 5, 7 (odd numbers transition to even)


# ===========================================================================
# 8. Mutual Exclusion Protocol
# ===========================================================================

class TestMutex:
    def test_mutex_creation(self):
        slts = make_mutex_lts(2)
        assert slts.n_bits == 4  # 2 bits per process * 2 processes
        assert "critical0" in slts.labels
        assert "critical1" in slts.labels
        assert "mutex" in slts.labels

    def test_mutex_safety(self):
        slts = make_mutex_lts(2)
        # AG(mutex): mutual exclusion always holds
        r = symbolic_check(slts, ctl_AG(Prop("mutex")))
        # Check if mutex holds in all reachable states
        safety = check_safety_symbolic(slts, slts.labels["mutex"])
        assert safety["result"] == "safe"

    def test_mutex_reachability(self):
        slts = make_mutex_lts(2)
        reached, _ = symbolic_reachable(slts)
        mc = SymbolicMuChecker(slts)
        count = mc.sat_count(reached)
        # Both processes can be idle, trying, or critical (with mutex constraint)
        # So reachable states < 3*3=9
        assert count > 0

    def test_mutex_liveness_critical0(self):
        slts = make_mutex_lts(2)
        # EF(critical0): can eventually reach critical section
        ef = ctl_EF(Prop("critical0"))
        r = symbolic_check(slts, ef)
        assert r.sat_count > 0  # some states can reach critical


# ===========================================================================
# 9. Make Symbolic LTS Directly
# ===========================================================================

class TestDirectConstruction:
    def test_make_symbolic_lts(self):
        slts = make_symbolic_lts(
            state_var_names=["x"],
            init_fn=lambda bdd, v: bdd.NOT(v["x"]),  # x=0
            trans_fns={"flip": lambda bdd, c, n: bdd.XOR(c["x"], n["x"])},
            label_fns={"on": lambda bdd, v: v["x"]},
        )
        assert slts.n_bits == 1
        r = symbolic_check(slts, Prop("on"))
        assert r.sat_count == 1  # only state 1 has "on"

    def test_make_symmetric_2bit(self):
        slts = make_symbolic_lts(
            state_var_names=["a", "b"],
            init_fn=lambda bdd, v: bdd.AND(bdd.NOT(v["a"]), bdd.NOT(v["b"])),
            trans_fns={
                "set_a": lambda bdd, c, n: bdd.AND(n["a"], bdd.IFF(c["b"], n["b"])),
                "set_b": lambda bdd, c, n: bdd.AND(n["b"], bdd.IFF(c["a"], n["a"])),
            },
            label_fns={
                "both": lambda bdd, v: bdd.AND(v["a"], v["b"]),
                "none": lambda bdd, v: bdd.AND(bdd.NOT(v["a"]), bdd.NOT(v["b"])),
            },
        )
        # EF(both): can reach state where both are set
        r = symbolic_check(slts, ctl_EF(Prop("both")))
        assert 0 in r.sat_states  # from init (0,0) can reach (1,1)


# ===========================================================================
# 10. Check State
# ===========================================================================

class TestCheckState:
    def test_check_specific_state(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        assert check_state_symbolic(slts, Prop("red"), 0)
        assert not check_state_symbolic(slts, Prop("red"), 1)
        assert check_state_symbolic(slts, Prop("green"), 1)

    def test_check_ef_specific_state(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        ef = ctl_EF(Prop("red"))
        assert check_state_symbolic(slts, ef, 0)
        assert check_state_symbolic(slts, ef, 1)
        assert check_state_symbolic(slts, ef, 2)


# ===========================================================================
# 11. Batch Checking
# ===========================================================================

class TestBatch:
    def test_batch_formulas(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        results = batch_symbolic_check(slts, [
            Prop("red"), Prop("green"), Prop("yellow"), TT(), FF()
        ])
        assert len(results) == 5
        assert results[0].sat_count == 1  # red
        assert results[1].sat_count == 1  # green
        assert results[2].sat_count == 1  # yellow
        assert results[3].sat_count == 3  # TT
        assert results[4].sat_count == 0  # FF


# ===========================================================================
# 12. Safety Checking
# ===========================================================================

class TestSafety:
    def test_safety_holds(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        # All states have exactly one color
        prop = slts.bdd.OR(
            slts.labels["red"],
            slts.bdd.OR(slts.labels["green"], slts.labels["yellow"])
        )
        result = check_safety_symbolic(slts, prop)
        assert result["result"] == "safe"

    def test_safety_violated(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        # "always red" -- violated
        result = check_safety_symbolic(slts, slts.labels["red"])
        assert result["result"] == "unsafe"


# ===========================================================================
# 13. Parser Integration
# ===========================================================================

class TestParser:
    def test_parsed_formula(self):
        lts = traffic_light_lts()
        f = parse_mu("mu X. (red | <>X)")
        r = symbolic_check_lts(lts, f)
        assert r.sat_count == 3

    def test_parsed_nu(self):
        lts = traffic_light_lts()
        f = parse_mu("nu X. (red & []X)")
        r = symbolic_check_lts(lts, f)
        # nu X. (red & [change]X): states where red holds forever = none (cyclic)
        assert r.sat_count == 0

    def test_parsed_complex(self):
        lts = traffic_light_lts()
        f = parse_mu("mu X. (green | <>X)")
        r = symbolic_check_lts(lts, f)
        assert r.sat_count == 3


# ===========================================================================
# 14. Summary and Full Analysis
# ===========================================================================

class TestSummary:
    def test_summary_string(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        s = symbolic_mu_summary(slts, Prop("red"))
        assert "red" in s
        assert "1" in s

    def test_full_analysis(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        results = full_analysis(slts, [
            ("red_states", Prop("red")),
            ("ef_green", ctl_EF(Prop("green"))),
            ("ag_colored", ctl_AG(Or(Prop("red"), Or(Prop("green"), Prop("yellow"))))),
        ])
        assert results["red_states"]["sat_count"] == 1
        assert results["ef_green"]["holds_everywhere"]
        assert results["ag_colored"]["holds_everywhere"]


# ===========================================================================
# 15. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_single_state_self_loop(self):
        lts = make_lts(1, [(0, "a", 0)], {0: {"p"}})
        r = symbolic_check_lts(lts, Nu("X", And(Prop("p"), Diamond("a", Var("X")))))
        assert r.sat_count == 1  # state 0 satisfies: p holds and can loop forever

    def test_double_negation(self):
        lts = traffic_light_lts()
        r1 = symbolic_check_lts(lts, Prop("red"))
        r2 = symbolic_check_lts(lts, Not(Not(Prop("red"))))
        assert r1.sat_states == r2.sat_states

    def test_and_ff_absorbs(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, And(TT(), FF()))
        assert r.sat_count == 0

    def test_or_tt_absorbs(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, Or(FF(), TT()))
        assert r.sat_count == 3

    def test_holds_everywhere_property(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, TT())
        assert r.holds_everywhere

    def test_holds_nowhere_property(self):
        lts = traffic_light_lts()
        r = symbolic_check_lts(lts, FF())
        assert r.holds_nowhere

    def test_unbound_variable_raises(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        mc = SymbolicMuChecker(slts)
        with pytest.raises(ValueError, match="Unbound variable"):
            mc.check(Var("X"))

    def test_empty_lts_raises(self):
        with pytest.raises(ValueError):
            lts_to_symbolic(LTS(states=set(), transitions={}, labels={}))

    def test_stats_tracking(self):
        lts = traffic_light_lts()
        slts = lts_to_symbolic(lts)
        mc = SymbolicMuChecker(slts)
        mc.check(ctl_EF(Prop("red")))
        assert mc.stats["fixpoint_iterations"] > 0
        assert mc.stats["diamond_ops"] > 0


# ===========================================================================
# 16. Larger System (8 states)
# ===========================================================================

class TestLargerSystem:
    def test_8state_ring(self):
        """8-state ring: each state transitions to next."""
        trans = [(i, "next", (i + 1) % 8) for i in range(8)]
        labels = {0: {"origin"}, 4: {"halfway"}}
        lts = make_lts(8, trans, labels)
        r = symbolic_check_lts(lts, ctl_EF(Prop("origin")))
        assert r.sat_count == 8  # all can reach origin

    def test_8state_ef_halfway(self):
        trans = [(i, "next", (i + 1) % 8) for i in range(8)]
        labels = {0: {"origin"}, 4: {"halfway"}}
        lts = make_lts(8, trans, labels)
        r = symbolic_check_lts(lts, ctl_EF(Prop("halfway")))
        assert r.sat_count == 8

    def test_8state_ag(self):
        trans = [(i, "next", (i + 1) % 8) for i in range(8)]
        labels = {i: {"exists"} for i in range(8)}
        lts = make_lts(8, trans, labels)
        r = symbolic_check_lts(lts, ctl_AG(Prop("exists")))
        assert r.sat_count == 8

    def test_comparison_8state(self):
        trans = [(i, "next", (i + 1) % 8) for i in range(8)]
        labels = {0: {"origin"}, 4: {"halfway"}}
        lts = make_lts(8, trans, labels)
        comp = compare_with_explicit(lts, Diamond("next", Prop("origin")))
        assert comp["agree"]


# ===========================================================================
# 17. Boolean TS Conversion
# ===========================================================================

class TestBooleanTSConversion:
    def test_boolean_ts_to_slts(self):
        bdd = BDD(4)  # 2 curr + 2 next vars
        bts = make_boolean_ts(bdd, ["x", "y"])
        # Set up init: x=0, y=0
        bts.init = bdd.AND(bdd.NOT(bdd.var(bts.var_indices["x"])),
                           bdd.NOT(bdd.var(bts.var_indices["y"])))
        # trans: toggle x, keep y
        # next_indices uses unprimed keys in V021
        bts.trans = bdd.AND(
            bdd.XOR(bdd.var(bts.var_indices["x"]),
                     bdd.var(bts.next_indices["x"])),
            bdd.IFF(bdd.var(bts.var_indices["y"]),
                     bdd.var(bts.next_indices["y"]))
        )
        labels = {"x_on": bdd.var(bts.var_indices["x"])}
        slts = boolean_ts_to_symbolic_lts(bts, labels)
        assert slts.n_bits == 2
        assert "x_on" in slts.labels
        assert None in slts.trans  # wildcard action


# ===========================================================================
# 18. Nested Fixpoints
# ===========================================================================

class TestNestedFixpoints:
    def test_nu_mu_nesting(self):
        lts = traffic_light_lts()
        # nu X. mu Y. (red & X) | <>Y
        # = greatest fixpoint of: smallest set containing red&X or reachable from it
        # = states from which red is always reachable
        f = Nu("X", Mu("Y", Or(And(Prop("red"), Var("X")), Diamond(None, Var("Y")))))
        r = symbolic_check_lts(lts, f)
        comp = compare_with_explicit(lts, f)
        assert comp["agree"]

    def test_mu_nu_nesting(self):
        lts = traffic_light_lts()
        # mu X. nu Y. (green | <>Y) & <>X
        f = Mu("X", And(Nu("Y", Or(Prop("green"), Diamond(None, Var("Y")))),
                         Diamond(None, Var("X"))))
        r = symbolic_check_lts(lts, f)
        comp = compare_with_explicit(lts, f)
        assert comp["agree"]

    def test_alternation_depth_2(self):
        lts = simple_lts()
        # nu X. mu Y. (<>X | end) & []Y
        f = Nu("X", Mu("Y", And(Or(Diamond(None, Var("X")), Prop("end")),
                                  Box(None, Var("Y")))))
        comp = compare_with_explicit(lts, f)
        assert comp["agree"]


# ===========================================================================
# 19. Action-Specific Modalities
# ===========================================================================

class TestActionModalities:
    def test_different_actions(self):
        lts = simple_lts()  # has actions "a" and "b"
        # <a>left: states with a-successor satisfying left
        r = symbolic_check_lts(lts, Diamond("a", Prop("left")))
        assert 0 in r.sat_states  # state 0 --a--> 1 (left)

    def test_box_specific_action(self):
        lts = simple_lts()
        # [a]end: all a-successors satisfy end
        r = symbolic_check_lts(lts, Box("a", Prop("end")))
        assert 1 in r.sat_states  # state 1 --a--> 3 (end), only a-succ
        assert 3 not in r.sat_states  # state 3 --a--> 0 (start, not end)

    def test_no_action_transitions(self):
        lts = simple_lts()
        # <c>tt: no c-transitions exist
        r = symbolic_check_lts(lts, Diamond("c", TT()))
        assert r.sat_count == 0


# ===========================================================================
# 20. Reachability Analysis
# ===========================================================================

class TestReachability:
    def test_counter_reachable_all(self):
        slts = make_counter_lts(2)
        reached, iters = symbolic_reachable(slts)
        mc = SymbolicMuChecker(slts)
        assert mc.sat_count(reached) == 4  # all 4 states

    def test_reachable_disconnected(self):
        lts = make_lts(4, [
            (0, "a", 1), (2, "a", 3),
        ], {0: {"start"}, 2: {"isolated"}})
        slts = lts_to_symbolic(lts)
        # Init = all valid states, so all are "reachable" from init=all
        reached, _ = symbolic_reachable(slts)
        mc = SymbolicMuChecker(slts)
        assert mc.sat_count(reached) == 4

    def test_fixpoint_convergence(self):
        slts = make_counter_lts(3)
        _, iters = symbolic_reachable(slts)
        assert iters <= 8  # should converge within 8 steps for 8-state counter
