"""Tests for V094: Pushdown Systems Verification."""

import pytest
from pushdown_systems import (
    PushdownSystem, PDSRule, StackOp, Configuration, PAutomaton,
    make_config_automaton, make_state_automaton,
    pre_star, post_star,
    check_reachability, check_safety, check_regular_property,
    bounded_reachability, explore_state_space, check_invariant,
    compare_pre_post, pds_summary,
    RecursiveProgram, program_to_pds,
    make_simple_counter, make_recursive_program_pds,
    make_mutual_recursion_pds, make_stack_inspection_pds,
    _successors, _predecessors, _find_path_bfs,
)


# === Section 1: Data Structures ===

class TestDataStructures:
    def test_configuration_basic(self):
        c = Configuration("p", ("a", "b"))
        assert c.state == "p"
        assert c.stack == ("a", "b")
        assert c.top() == "a"
        assert not c.is_empty_stack()

    def test_configuration_empty_stack(self):
        c = Configuration("p", ())
        assert c.top() is None
        assert c.is_empty_stack()

    def test_configuration_push_pop(self):
        c = Configuration("p", ("a",))
        pushed = c.push("b")
        assert pushed.stack == ("b", "a")
        popped = pushed.pop()
        assert popped.stack == ("a",)

    def test_configuration_frozen(self):
        c1 = Configuration("p", ("a",))
        c2 = Configuration("p", ("a",))
        assert c1 == c2
        assert hash(c1) == hash(c2)
        s = {c1, c2}
        assert len(s) == 1

    def test_pds_rule_pop(self):
        r = PDSRule("p", "a", "q", StackOp.POP)
        assert r.op == StackOp.POP
        assert r.push_symbols == ()

    def test_pds_rule_swap(self):
        r = PDSRule("p", "a", "q", StackOp.SWAP, ("b",))
        assert r.push_symbols == ("b",)

    def test_pds_rule_push(self):
        r = PDSRule("p", "a", "q", StackOp.PUSH, ("b", "c"))
        assert r.push_symbols == ("b", "c")

    def test_pds_add_rule(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        assert "p" in pds.states
        assert "q" in pds.states
        assert "a" in pds.stack_alphabet
        assert "b" in pds.stack_alphabet
        assert len(pds.rules) == 1

    def test_pds_get_rules(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        pds.add_rule("p", "a", "r", StackOp.SWAP, ("b",))
        pds.add_rule("p", "b", "s", StackOp.POP)
        rules = pds.get_rules("p", "a")
        assert len(rules) == 2


# === Section 2: Successors and Predecessors ===

class TestSuccessorsPredecessors:
    def test_pop_successor(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        config = Configuration("p", ("a", "b"))
        succs = _successors(pds, config)
        assert Configuration("q", ("b",)) in succs

    def test_swap_successor(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("c",))
        config = Configuration("p", ("a", "b"))
        succs = _successors(pds, config)
        assert Configuration("q", ("c", "b")) in succs

    def test_push_successor(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("x", "y"))
        config = Configuration("p", ("a", "b"))
        succs = _successors(pds, config)
        assert Configuration("q", ("x", "y", "b")) in succs

    def test_empty_stack_no_successors(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        config = Configuration("p", ())
        succs = _successors(pds, config)
        assert len(succs) == 0

    def test_no_matching_rule(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        config = Configuration("p", ("b",))  # wrong symbol
        succs = _successors(pds, config)
        assert len(succs) == 0

    def test_pop_predecessor(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        config = Configuration("q", ("b",))
        preds = _predecessors(pds, config)
        assert Configuration("p", ("a", "b")) in preds

    def test_swap_predecessor(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("c",))
        config = Configuration("q", ("c", "b"))
        preds = _predecessors(pds, config)
        assert Configuration("p", ("a", "b")) in preds

    def test_push_predecessor(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("x", "y"))
        config = Configuration("q", ("x", "y", "b"))
        preds = _predecessors(pds, config)
        assert Configuration("p", ("a", "b")) in preds


# === Section 3: P-Automaton ===

class TestPAutomaton:
    def test_basic_acceptance(self):
        pa = PAutomaton({"p"})
        final = "f"
        pa.add_state(final, final=True)
        pa.add_transition("p", "a", final)
        assert pa.accepts(Configuration("p", ("a",)))
        assert not pa.accepts(Configuration("p", ("b",)))
        assert not pa.accepts(Configuration("p", ()))

    def test_multi_symbol_acceptance(self):
        pa = PAutomaton({"p"})
        mid = "m"
        final = "f"
        pa.add_state(mid)
        pa.add_state(final, final=True)
        pa.add_transition("p", "a", mid)
        pa.add_transition(mid, "b", final)
        assert pa.accepts(Configuration("p", ("a", "b")))
        assert not pa.accepts(Configuration("p", ("a",)))
        assert not pa.accepts(Configuration("p", ("b",)))

    def test_empty_stack_acceptance(self):
        pa = PAutomaton({"p"})
        pa.add_state("p", final=True)  # p is both initial and final
        assert pa.accepts(Configuration("p", ()))
        assert not pa.accepts(Configuration("p", ("a",)))

    def test_epsilon_transitions(self):
        pa = PAutomaton({"p"})
        final = "f"
        mid = "m"
        pa.add_state(mid)
        pa.add_state(final, final=True)
        pa.add_transition("p", "a", mid)
        pa.add_epsilon(mid, final)
        # p --a--> mid --eps--> final: accepts (p, "a")
        assert pa.accepts(Configuration("p", ("a",)))

    def test_make_config_automaton(self):
        pds = PushdownSystem()
        pds.states = {"p", "q"}
        pds.stack_alphabet = {"a", "b"}

        configs = {
            Configuration("p", ("a",)),
            Configuration("q", ("a", "b")),
        }
        pa = make_config_automaton(pds, configs)
        assert pa.accepts(Configuration("p", ("a",)))
        assert pa.accepts(Configuration("q", ("a", "b")))
        assert not pa.accepts(Configuration("p", ("b",)))

    def test_make_config_automaton_empty_stack(self):
        pds = PushdownSystem()
        pds.states = {"p"}
        pds.stack_alphabet = {"a"}

        configs = {Configuration("p", ())}
        pa = make_config_automaton(pds, configs)
        assert pa.accepts(Configuration("p", ()))

    def test_copy(self):
        pa = PAutomaton({"p"})
        pa.add_state("f", final=True)
        pa.add_transition("p", "a", "f")
        pa2 = pa.copy()
        pa2.add_transition("p", "b", "f")
        assert pa2.accepts(Configuration("p", ("b",)))
        assert not pa.accepts(Configuration("p", ("b",)))


# === Section 4: Pre* Computation ===

class TestPreStar:
    def test_pre_star_pop(self):
        """(p, a) -> (q, eps): pre* of (q, eps) should include (p, a)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)

        target = {Configuration("q", ())}
        target_auto = make_config_automaton(pds, target)
        pre = pre_star(pds, target_auto)

        assert pre.accepts(Configuration("q", ()))  # original
        assert pre.accepts(Configuration("p", ("a",)))  # predecessor via pop

    def test_pre_star_swap(self):
        """(p, a) -> (q, b): pre* of (q, b) should include (p, a)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))

        target = {Configuration("q", ("b",))}
        target_auto = make_config_automaton(pds, target)
        pre = pre_star(pds, target_auto)

        assert pre.accepts(Configuration("q", ("b",)))  # original
        assert pre.accepts(Configuration("p", ("a",)))  # predecessor via swap

    def test_pre_star_push(self):
        """(p, a) -> (q, b c): pre* of (q, b c) should include (p, a)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "c"))

        target = {Configuration("q", ("b", "c"))}
        target_auto = make_config_automaton(pds, target)
        pre = pre_star(pds, target_auto)

        assert pre.accepts(Configuration("q", ("b", "c")))  # original
        assert pre.accepts(Configuration("p", ("a",)))  # predecessor via push

    def test_pre_star_chain(self):
        """Chain: p --swap--> q --pop--> r. Pre* of r includes p."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("q", "b", "r", StackOp.POP)

        target = {Configuration("r", ())}
        target_auto = make_config_automaton(pds, target)
        pre = pre_star(pds, target_auto)

        assert pre.accepts(Configuration("r", ()))
        assert pre.accepts(Configuration("q", ("b",)))
        assert pre.accepts(Configuration("p", ("a",)))

    def test_pre_star_preserves_stack_suffix(self):
        """Pre* should handle arbitrary stack suffixes.
        (p, a) -> (q, eps): pre* of (q, x) should include (p, a x)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)

        target = {Configuration("q", ("x",))}
        target_auto = make_config_automaton(pds, target)
        pre = pre_star(pds, target_auto)

        assert pre.accepts(Configuration("q", ("x",)))
        assert pre.accepts(Configuration("p", ("a", "x")))


# === Section 5: Post* Computation ===

class TestPostStar:
    def test_post_star_pop(self):
        """(p, a) -> (q, eps): post* of (p, a) includes (q, eps)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)

        source = {Configuration("p", ("a",))}
        source_auto = make_config_automaton(pds, source)
        post = post_star(pds, source_auto)

        assert post.accepts(Configuration("p", ("a",)))  # original
        assert post.accepts(Configuration("q", ()))  # successor via pop

    def test_post_star_swap(self):
        """(p, a) -> (q, b): post* of (p, a) includes (q, b)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))

        source = {Configuration("p", ("a",))}
        source_auto = make_config_automaton(pds, source)
        post = post_star(pds, source_auto)

        assert post.accepts(Configuration("p", ("a",)))
        assert post.accepts(Configuration("q", ("b",)))

    def test_post_star_push(self):
        """(p, a) -> (q, b c): post* of (p, a) includes (q, b c)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "c"))

        source = {Configuration("p", ("a",))}
        source_auto = make_config_automaton(pds, source)
        post = post_star(pds, source_auto)

        assert post.accepts(Configuration("p", ("a",)))
        assert post.accepts(Configuration("q", ("b", "c")))

    def test_post_star_chain(self):
        """Chain: p --swap--> q --pop--> r. Post* of (p,a) includes (r, eps)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("q", "b", "r", StackOp.POP)

        source = {Configuration("p", ("a",))}
        source_auto = make_config_automaton(pds, source)
        post = post_star(pds, source_auto)

        assert post.accepts(Configuration("p", ("a",)))
        assert post.accepts(Configuration("q", ("b",)))
        assert post.accepts(Configuration("r", ()))

    def test_post_star_preserves_stack_suffix(self):
        """(p, a) -> (q, b): post* of (p, a x) includes (q, b x)."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))

        source = {Configuration("p", ("a", "x"))}
        source_auto = make_config_automaton(pds, source)
        post = post_star(pds, source_auto)

        assert post.accepts(Configuration("p", ("a", "x")))
        assert post.accepts(Configuration("q", ("b", "x")))


# === Section 6: Reachability Checking ===

class TestReachability:
    def test_self_reachable(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        config = Configuration("p", ("a",))
        result = check_reachability(pds, config, config)
        assert result["reachable"]

    def test_simple_reachable(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        src = Configuration("p", ("a",))
        tgt = Configuration("q", ())
        result = check_reachability(pds, src, tgt)
        assert result["reachable"]
        assert result["witness_path"] is not None

    def test_unreachable(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        src = Configuration("p", ("a",))
        tgt = Configuration("r", ())  # r is not reachable
        pds.states.add("r")
        result = check_reachability(pds, src, tgt)
        assert not result["reachable"]

    def test_multi_step_reachable(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("q", "b", "r", StackOp.SWAP, ("c",))
        pds.add_rule("r", "c", "s", StackOp.POP)
        src = Configuration("p", ("a",))
        tgt = Configuration("s", ())
        result = check_reachability(pds, src, tgt)
        assert result["reachable"]

    def test_push_pop_reachable(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "a"))
        pds.add_rule("q", "b", "r", StackOp.POP)
        # (p, a) -> (q, b a) -> (r, a)
        src = Configuration("p", ("a",))
        tgt = Configuration("r", ("a",))
        result = check_reachability(pds, src, tgt)
        assert result["reachable"]


# === Section 7: Safety Checking ===

class TestSafety:
    def test_safe_system(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = {Configuration("p", ("a",))}
        bad = {Configuration("r", ())}
        pds.states.add("r")
        result = check_safety(pds, initial, bad)
        assert result["safe"]

    def test_unsafe_system(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = {Configuration("p", ("a",))}
        bad = {Configuration("q", ())}
        result = check_safety(pds, initial, bad)
        assert not result["safe"]
        assert result["counterexample"] is not None

    def test_safe_with_multiple_paths(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("p", "a", "r", StackOp.SWAP, ("c",))
        pds.add_rule("q", "b", "s", StackOp.POP)
        pds.add_rule("r", "c", "s", StackOp.POP)
        initial = {Configuration("p", ("a",))}
        bad = {Configuration("bad", ())}
        pds.states.add("bad")
        result = check_safety(pds, initial, bad)
        assert result["safe"]


# === Section 8: Bounded Reachability ===

class TestBoundedReachability:
    def test_immediate_target(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = bounded_reachability(pds, initial, lambda c: c.state == "p")
        assert result["reachable"]
        assert result["steps"] == 0

    def test_one_step_target(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = bounded_reachability(pds, initial, lambda c: c.state == "q")
        assert result["reachable"]
        assert result["steps"] == 1

    def test_bounded_no_target(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = bounded_reachability(pds, initial, lambda c: c.state == "r", max_steps=10)
        assert not result["reachable"]

    def test_stack_depth_limit(self):
        pds = PushdownSystem()
        # Infinite push
        pds.add_rule("p", "a", "p", StackOp.PUSH, ("a", "a"))
        initial = Configuration("p", ("a",))
        result = bounded_reachability(pds, initial,
                                      lambda c: len(c.stack) > 100,
                                      max_steps=50, max_stack=10)
        assert not result["reachable"]


# === Section 9: State Space Exploration ===

class TestExploreStateSpace:
    def test_simple_exploration(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = explore_state_space(pds, initial)
        assert result["reachable_configs"] == 2  # (p, a) and (q, eps)
        assert result["exhaustive"]

    def test_branching_exploration(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        pds.add_rule("p", "a", "r", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = explore_state_space(pds, initial)
        assert result["reachable_configs"] == 3  # (p,a), (q,), (r,)

    def test_deadlock_detection(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        # q with b has no rules
        initial = Configuration("p", ("a",))
        result = explore_state_space(pds, initial)
        assert result["deadlocks"] == 1


# === Section 10: Invariant Checking ===

class TestInvariantChecking:
    def test_invariant_holds(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = {Configuration("p", ("a",))}
        result = check_invariant(pds, initial, lambda c: c.state in {"p", "q"})
        assert result["holds"]

    def test_invariant_violated(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        pds.add_rule("p", "a", "bad", StackOp.POP)
        initial = {Configuration("p", ("a",))}
        result = check_invariant(pds, initial, lambda c: c.state != "bad")
        assert not result["holds"]
        assert result["violation"].state == "bad"

    def test_stack_invariant(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "p", StackOp.PUSH, ("a", "a"))
        pds.add_rule("p", "a", "q", StackOp.POP)
        initial = {Configuration("p", ("a",))}
        # Stack only contains 'a's
        result = check_invariant(pds, initial,
                                 lambda c: all(s == "a" for s in c.stack),
                                 max_stack=5)
        assert result["holds"]


# === Section 11: Recursive Program Modeling ===

class TestRecursiveProgram:
    def test_simple_program(self):
        prog = RecursiveProgram()
        prog.entry = "main"
        prog.entry_label = "start"
        prog.add_function("main", [
            {"type": "assign", "label": "start", "next": "end"},
            {"type": "return", "label": "end"},
        ])
        pds, initial = program_to_pds(prog)
        assert initial.state == "run"
        assert initial.stack == ("main.start",)

    def test_call_return(self):
        prog = RecursiveProgram()
        prog.entry = "main"
        prog.entry_label = "start"
        prog.add_function("main", [
            {"type": "call", "label": "start", "target": "f", "return_label": "end"},
            {"type": "return", "label": "end"},
        ])
        prog.add_function("f", [
            {"type": "return", "label": "entry"},
        ])
        pds, initial = program_to_pds(prog)

        # Initial: (run, main.start)
        # After call: (run, f.entry main.end)
        # After f returns: (run, main.end)
        # After main returns: (run, eps)
        result = bounded_reachability(pds, initial,
                                      lambda c: c.is_empty_stack())
        assert result["reachable"]

    def test_recursive_function(self):
        pds, initial, prog = make_recursive_program_pds()
        # The recursive program can reach empty stack (main returns)
        result = bounded_reachability(pds, initial,
                                      lambda c: c.is_empty_stack(),
                                      max_steps=20, max_stack=10)
        assert result["reachable"]

    def test_recursive_function_stack_growth(self):
        pds, initial, prog = make_recursive_program_pds()
        # Recursive calls grow the stack
        result = bounded_reachability(pds, initial,
                                      lambda c: len(c.stack) >= 4,
                                      max_steps=20, max_stack=10)
        assert result["reachable"]

    def test_mutual_recursion(self):
        pds, initial, prog = make_mutual_recursion_pds()
        # Should be able to reach empty stack
        result = bounded_reachability(pds, initial,
                                      lambda c: c.is_empty_stack(),
                                      max_steps=30, max_stack=10)
        assert result["reachable"]

    def test_mutual_recursion_stack_depth(self):
        pds, initial, prog = make_mutual_recursion_pds()
        # Mutual recursion can build up stack
        result = bounded_reachability(pds, initial,
                                      lambda c: len(c.stack) >= 5,
                                      max_steps=30, max_stack=10)
        assert result["reachable"]


# === Section 12: Simple Counter PDS ===

class TestSimpleCounter:
    def test_counter_creation(self):
        pds, initial = make_simple_counter()
        assert initial == Configuration("inc", ("#",))
        assert len(pds.rules) > 0

    def test_counter_can_reach_done(self):
        pds, initial = make_simple_counter()
        result = bounded_reachability(pds, initial,
                                      lambda c: c.state == "done",
                                      max_steps=20, max_stack=10)
        assert result["reachable"]

    def test_counter_exploration(self):
        pds, initial = make_simple_counter()
        result = explore_state_space(pds, initial, max_configs=100, max_stack=5)
        assert result["reachable_configs"] > 1

    def test_counter_summary(self):
        pds, initial = make_simple_counter()
        summary = pds_summary(pds)
        assert summary["states"] >= 3
        assert summary["rules"] >= 5


# === Section 13: Stack Inspection ===

class TestStackInspection:
    def test_stack_inspection_creation(self):
        pds, initial = make_stack_inspection_pds()
        assert initial == Configuration("trusted", ("t_frame",))

    def test_check_state_reachable(self):
        pds, initial = make_stack_inspection_pds()
        # Check state is reachable from trusted
        result = bounded_reachability(pds, initial,
                                      lambda c: c.state == "check",
                                      max_steps=10, max_stack=10)
        assert result["reachable"]

    def test_untrusted_can_reach_check(self):
        pds, initial = make_stack_inspection_pds()
        # From trusted, we can go through untrusted, then to check
        result = bounded_reachability(pds, initial,
                                      lambda c: c.state == "check",
                                      max_steps=10, max_stack=10)
        assert result["reachable"]

    def test_stack_inspection_invariant(self):
        """Stack depth never exceeds max when bounded."""
        pds, initial = make_stack_inspection_pds()
        result = check_invariant(pds, {initial},
                                 lambda c: len(c.stack) <= 10,
                                 max_steps=20, max_stack=10)
        assert result["holds"]


# === Section 14: Compare Pre*/Post* ===

class TestComparePrePost:
    def test_compare_basic(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("q", "b", "r", StackOp.POP)
        configs = {Configuration("q", ("b",))}
        result = compare_pre_post(pds, configs, max_stack=3)
        assert result["pre_star_size"] >= 1
        assert result["post_star_size"] >= 1

    def test_pre_star_superset(self):
        """Pre* always contains the original set."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        configs = {Configuration("q", ())}
        result = compare_pre_post(pds, configs, max_stack=3)
        for c in configs:
            assert c in result["pre_star_configs"]

    def test_post_star_superset(self):
        """Post* always contains the original set."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        configs = {Configuration("p", ("a",))}
        result = compare_pre_post(pds, configs, max_stack=3)
        for c in configs:
            assert c in result["post_star_configs"]


# === Section 15: PDS Summary ===

class TestPDSSummary:
    def test_summary_counts(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        pds.add_rule("p", "a", "r", StackOp.SWAP, ("b",))
        pds.add_rule("r", "b", "s", StackOp.PUSH, ("c", "d"))
        summary = pds_summary(pds)
        assert summary["states"] == 4
        assert summary["rules"] == 3
        assert summary["pop_rules"] == 1
        assert summary["swap_rules"] == 1
        assert summary["push_rules"] == 1


# === Section 16: State Automaton ===

class TestStateAutomaton:
    def test_state_automaton_accepts_any_stack(self):
        pds = PushdownSystem()
        pds.states = {"p", "q"}
        pds.stack_alphabet = {"a", "b"}
        auto = make_state_automaton(pds, {"p"})
        assert auto.accepts(Configuration("p", ()))
        assert auto.accepts(Configuration("p", ("a",)))
        assert auto.accepts(Configuration("p", ("a", "b")))
        assert auto.accepts(Configuration("p", ("a", "b", "a")))
        assert not auto.accepts(Configuration("q", ("a",)))


# === Section 17: Regular Property Checking ===

class TestRegularProperty:
    def test_reachable_property(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)

        init_configs = make_config_automaton(pds, {Configuration("p", ("a",))})
        target_configs = make_config_automaton(pds, {Configuration("q", ())})

        result = check_regular_property(pds, init_configs, target_configs, "forward")
        assert result["satisfies"]

    def test_unreachable_property(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        pds.states.add("r")
        pds.stack_alphabet.add("c")

        init_configs = make_config_automaton(pds, {Configuration("p", ("a",))})
        target_configs = make_config_automaton(pds, {Configuration("r", ("c",))})

        result = check_regular_property(pds, init_configs, target_configs, "forward")
        assert not result["satisfies"]


# === Section 18: Complex PDS Scenarios ===

class TestComplexScenarios:
    def test_push_then_pop_identity(self):
        """Push then pop returns to equivalent configuration."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "a"))
        pds.add_rule("q", "b", "r", StackOp.POP)
        # (p, a) -> (q, b a) -> (r, a)
        src = Configuration("p", ("a",))
        tgt = Configuration("r", ("a",))
        result = check_reachability(pds, src, tgt)
        assert result["reachable"]

    def test_nondeterministic_choice(self):
        """PDS with nondeterministic choice can reach either target."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q1", StackOp.POP)
        pds.add_rule("p", "a", "q2", StackOp.POP)
        src = Configuration("p", ("a",))

        r1 = check_reachability(pds, src, Configuration("q1", ()))
        r2 = check_reachability(pds, src, Configuration("q2", ()))
        assert r1["reachable"]
        assert r2["reachable"]

    def test_loop_detection(self):
        """Swap loop doesn't cause infinite exploration."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("q", "b", "p", StackOp.SWAP, ("a",))
        pds.add_rule("p", "a", "done", StackOp.POP)

        initial = Configuration("p", ("a",))
        result = explore_state_space(pds, initial, max_configs=100)
        assert result["exhaustive"]  # finite state space

    def test_deep_recursion_reachability(self):
        """Deep push-pop chain."""
        pds = PushdownSystem()
        # Push chain: p->q pushing many symbols
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "a"))
        pds.add_rule("q", "b", "p", StackOp.SWAP, ("a",))
        # Exit: pop all
        pds.add_rule("p", "a", "r", StackOp.POP)

        initial = Configuration("p", ("a",))
        # Can reach r with any stack suffix
        result = bounded_reachability(pds, initial,
                                      lambda c: c.state == "r",
                                      max_steps=20, max_stack=10)
        assert result["reachable"]

    def test_stack_depth_safety(self):
        """Verify stack depth is bounded."""
        pds = PushdownSystem()
        # Push one, pop one -- stack stays bounded
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "a"))
        pds.add_rule("q", "b", "p", StackOp.POP)

        initial = {Configuration("p", ("a",))}
        result = check_invariant(pds, initial,
                                 lambda c: len(c.stack) <= 3,
                                 max_steps=20, max_stack=5)
        assert result["holds"]


# === Section 19: Pre*/Post* Soundness Cross-Validation ===

class TestCrossValidation:
    def test_pre_star_sound(self):
        """Every config in pre*(T) can reach T."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.POP)
        pds.add_rule("q", "b", "r", StackOp.SWAP, ("c",))

        target = {Configuration("r", ("c",))}
        target_auto = make_config_automaton(pds, target)
        pre = pre_star(pds, target_auto)

        # Enumerate pre* configs and verify each can reach target via BFS
        for config in pre.accepted_configs(pds, max_stack=3):
            for t in target:
                path = _find_path_bfs(pds, config, t, max_steps=20)
                if path:
                    break
            else:
                # Config might reach target with longer stack
                # Check via post* from config
                config_auto = make_config_automaton(pds, {config})
                post = post_star(pds, config_auto)
                found = any(post.accepts(t) for t in target)
                assert found, f"Config {config} in pre* but can't reach target"

    def test_post_star_sound(self):
        """Every config in post*(S) is reachable from S."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.SWAP, ("b",))
        pds.add_rule("q", "b", "r", StackOp.POP)

        source = {Configuration("p", ("a",))}
        source_auto = make_config_automaton(pds, source)
        post = post_star(pds, source_auto)

        for config in post.accepted_configs(pds, max_stack=3):
            found = False
            for s in source:
                path = _find_path_bfs(pds, s, config, max_steps=20)
                if path:
                    found = True
                    break
            if not found:
                # Check via pre*
                config_auto = make_config_automaton(pds, {config})
                pre = pre_star(pds, config_auto)
                found = any(pre.accepts(s) for s in source)
            assert found, f"Config {config} in post* but not reachable from source"


# === Section 20: Edge Cases ===

class TestEdgeCases:
    def test_empty_pds(self):
        pds = PushdownSystem()
        pds.states.add("p")
        pds.stack_alphabet.add("a")
        config = Configuration("p", ("a",))
        succs = _successors(pds, config)
        assert len(succs) == 0

    def test_single_state_pds(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "p", StackOp.POP)
        initial = Configuration("p", ("a", "a", "a"))
        result = bounded_reachability(pds, initial,
                                      lambda c: c.is_empty_stack())
        assert result["reachable"]
        assert result["steps"] == 3

    def test_multiple_stack_symbols(self):
        pds = PushdownSystem()
        pds.add_rule("p", "a", "p", StackOp.SWAP, ("b",))
        pds.add_rule("p", "b", "p", StackOp.SWAP, ("c",))
        pds.add_rule("p", "c", "q", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = bounded_reachability(pds, initial,
                                      lambda c: c.state == "q")
        assert result["reachable"]
        assert result["steps"] == 3

    def test_config_automaton_multiple_states(self):
        pds = PushdownSystem()
        pds.states = {"p", "q"}
        pds.stack_alphabet = {"a", "b"}
        configs = {
            Configuration("p", ("a",)),
            Configuration("p", ("b",)),
            Configuration("q", ("a", "b")),
        }
        pa = make_config_automaton(pds, configs)
        for c in configs:
            assert pa.accepts(c)
        assert not pa.accepts(Configuration("p", ("a", "b")))
        assert not pa.accepts(Configuration("q", ("a",)))


# === Section 21: Program-Level Verification ===

class TestProgramVerification:
    def test_program_termination(self):
        """Verify a simple program can terminate (reach empty stack)."""
        prog = RecursiveProgram()
        prog.entry = "main"
        prog.entry_label = "s0"
        prog.add_function("main", [
            {"type": "assign", "label": "s0", "next": "s1"},
            {"type": "return", "label": "s1"},
        ])
        pds, initial = program_to_pds(prog)
        result = bounded_reachability(pds, initial,
                                      lambda c: c.is_empty_stack(),
                                      max_steps=10)
        assert result["reachable"]

    def test_program_call_depth(self):
        """Verify max call depth is reachable."""
        pds, initial, prog = make_recursive_program_pds()
        # Recursive calls push frames
        result = explore_state_space(pds, initial, max_configs=500, max_stack=6)
        assert result["max_stack_depth"] >= 3  # at least 3 frames deep

    def test_program_all_points_reachable(self):
        """Verify all program points are reachable."""
        prog = RecursiveProgram()
        prog.entry = "main"
        prog.entry_label = "s0"
        prog.add_function("main", [
            {"type": "assign", "label": "s0", "next": "s1"},
            {"type": "assign", "label": "s1", "next": "s2"},
            {"type": "return", "label": "s2"},
        ])
        pds, initial = program_to_pds(prog)

        for label in ["s0", "s1", "s2"]:
            result = bounded_reachability(
                pds, initial,
                lambda c, l=label: c.stack and c.stack[0] == f"main.{l}",
                max_steps=10)
            assert result["reachable"], f"main.{label} not reachable"


# === Section 22: Stress Tests ===

class TestStressTests:
    def test_wide_branching(self):
        """PDS with many nondeterministic branches."""
        pds = PushdownSystem()
        for i in range(10):
            pds.add_rule("p", "a", f"q{i}", StackOp.POP)
        initial = Configuration("p", ("a",))
        result = explore_state_space(pds, initial)
        assert result["reachable_configs"] == 11  # p,a + 10 q_i

    def test_chain_of_swaps(self):
        """Long chain of swap transitions."""
        pds = PushdownSystem()
        n = 20
        for i in range(n):
            pds.add_rule(f"s{i}", f"a{i}", f"s{i+1}", StackOp.SWAP, (f"a{i+1}",))
        pds.add_rule(f"s{n}", f"a{n}", "done", StackOp.POP)

        initial = Configuration("s0", ("a0",))
        result = bounded_reachability(pds, initial,
                                      lambda c: c.state == "done",
                                      max_steps=n + 5)
        assert result["reachable"]
        assert result["steps"] == n + 1

    def test_repeated_push_pop(self):
        """Push and pop in cycle."""
        pds = PushdownSystem()
        pds.add_rule("p", "a", "q", StackOp.PUSH, ("b", "a"))
        pds.add_rule("q", "b", "p", StackOp.POP)
        # (p, a) -> (q, b a) -> (p, a) -- cycle
        initial = Configuration("p", ("a",))
        result = explore_state_space(pds, initial, max_stack=5)
        # Should find the cycle and stop
        assert result["reachable_configs"] == 2  # (p,a) and (q,ba)
        assert result["exhaustive"]
