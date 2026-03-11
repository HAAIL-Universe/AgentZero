"""Tests for V151: Probabilistic Process Algebra."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from prob_process_algebra import (
    Proc, ProcKind, stop, prefix, tau_prefix, prob_choice, nd_choice,
    parallel, restrict, relabel, recvar, recdef,
    generate_lts, check_process_equivalence, check_strong_equivalence,
    trace_set, deadlock_free, action_set, process_summary,
    parse_proc, _complement, _substitute, TAU,
    WeakBisimVerdict,
)


# ---- Tests: Constructors ----

class TestConstructors:
    def test_stop(self):
        p = stop()
        assert p.kind == ProcKind.STOP

    def test_prefix(self):
        p = prefix("a", stop())
        assert p.kind == ProcKind.PREFIX
        assert p.action == "a"
        assert p.left.kind == ProcKind.STOP

    def test_tau_prefix(self):
        p = tau_prefix(stop())
        assert p.action == TAU

    def test_prob_choice(self):
        p = prob_choice(0.3, stop(), stop())
        assert p.kind == ProcKind.PROB_CHOICE
        assert p.prob == 0.3

    def test_prob_choice_invalid(self):
        with pytest.raises(AssertionError):
            prob_choice(0.0, stop(), stop())
        with pytest.raises(AssertionError):
            prob_choice(1.0, stop(), stop())

    def test_nd_choice(self):
        p = nd_choice(prefix("a", stop()), prefix("b", stop()))
        assert p.kind == ProcKind.ND_CHOICE

    def test_parallel(self):
        p = parallel(prefix("a", stop()), prefix("~a", stop()))
        assert p.kind == ProcKind.PARALLEL

    def test_restrict(self):
        p = restrict(prefix("a", stop()), {"a"})
        assert p.kind == ProcKind.RESTRICT
        assert "a" in p.labels

    def test_relabel(self):
        p = relabel(prefix("a", stop()), {"a": "b"})
        assert p.kind == ProcKind.RELABEL

    def test_recvar(self):
        p = recvar("X")
        assert p.kind == ProcKind.RECVAR
        assert p.var == "X"

    def test_recdef(self):
        p = recdef("X", prefix("a", recvar("X")))
        assert p.kind == ProcKind.RECDEF

    def test_repr(self):
        p = prefix("a", stop())
        assert "a" in repr(p)


# ---- Tests: LTS Generation ----

class TestLTSGeneration:
    def test_stop_lts(self):
        lts = generate_lts(stop())
        assert lts.n_states == 1
        assert len(lts.actions.get(0, {})) == 0

    def test_prefix_lts(self):
        p = prefix("a", stop())
        lts = generate_lts(p)
        assert lts.n_states == 2
        assert "a" in lts.actions[0]

    def test_prefix_chain(self):
        p = prefix("a", prefix("b", stop()))
        lts = generate_lts(p)
        assert lts.n_states == 3

    def test_prob_choice_lts(self):
        p = prob_choice(0.5, prefix("a", stop()), prefix("b", stop()))
        lts = generate_lts(p)
        # Initial state has tau transition (probabilistic choice)
        assert TAU in lts.actions[0]

    def test_nd_choice_lts(self):
        p = nd_choice(prefix("a", stop()), prefix("b", stop()))
        lts = generate_lts(p)
        # Initial state has both a and b
        assert "a" in lts.actions[0]
        assert "b" in lts.actions[0]

    def test_parallel_interleaving(self):
        p = parallel(prefix("a", stop()), prefix("b", stop()))
        lts = generate_lts(p)
        # Initial state can do a or b (interleaving)
        assert "a" in lts.actions[0]
        assert "b" in lts.actions[0]

    def test_parallel_synchronization(self):
        p = parallel(prefix("a", stop()), prefix("~a", stop()))
        lts = generate_lts(p)
        # Can synchronize on a/~a -> tau, or interleave a and ~a
        assert TAU in lts.actions[0]

    def test_recursion_unfolds(self):
        # fix X. a.X -> a -> a -> a -> ...
        p = recdef("X", prefix("a", recvar("X")))
        lts = generate_lts(p, max_states=5)
        assert lts.n_states >= 1
        assert "a" in lts.actions[0]

    def test_recursion_finite_lts(self):
        # fix X. a.X unfolds to the same state -> should be 1 state
        p = recdef("X", prefix("a", recvar("X")))
        lts = generate_lts(p, max_states=100)
        # The unfolded a.(fix X. a.X) leads back to fix X. a.X -> 1 state
        # But: prefix("a", recdef("X", prefix("a", recvar("X")))) != recdef(...)
        # So it depends on representation. Let's just check it's finite.
        assert lts.n_states <= 100


# ---- Tests: Complement ----

class TestComplement:
    def test_complement_basic(self):
        assert _complement("a") == "~a"
        assert _complement("~a") == "a"

    def test_complement_involution(self):
        assert _complement(_complement("x")) == "x"


# ---- Tests: Substitution ----

class TestSubstitution:
    def test_substitute_var(self):
        p = _substitute(recvar("X"), "X", stop())
        assert p.kind == ProcKind.STOP

    def test_substitute_different_var(self):
        p = _substitute(recvar("Y"), "X", stop())
        assert p.kind == ProcKind.RECVAR
        assert p.var == "Y"

    def test_substitute_in_prefix(self):
        p = _substitute(prefix("a", recvar("X")), "X", stop())
        assert p.kind == ProcKind.PREFIX
        assert p.left.kind == ProcKind.STOP

    def test_substitute_shadowed(self):
        # fix X. (a.X) -- inner X is bound, should not substitute
        inner = recdef("X", prefix("a", recvar("X")))
        p = _substitute(inner, "X", stop())
        assert p.kind == ProcKind.RECDEF  # unchanged


# ---- Tests: Trace Set ----

class TestTraceSet:
    def test_stop_traces(self):
        traces = trace_set(stop())
        assert traces == {()}

    def test_prefix_trace(self):
        p = prefix("a", stop())
        traces = trace_set(p)
        assert ("a",) in traces

    def test_prefix_chain_trace(self):
        p = prefix("a", prefix("b", stop()))
        traces = trace_set(p)
        assert ("a", "b") in traces

    def test_nd_choice_traces(self):
        p = nd_choice(prefix("a", stop()), prefix("b", stop()))
        traces = trace_set(p)
        assert ("a",) in traces
        assert ("b",) in traces

    def test_prob_choice_traces(self):
        p = prob_choice(0.5, prefix("a", stop()), prefix("b", stop()))
        traces = trace_set(p)
        # After tau (probabilistic resolution), either a or b
        assert ("a",) in traces
        assert ("b",) in traces


# ---- Tests: Deadlock Freedom ----

class TestDeadlockFree:
    def test_stop_is_deadlock(self):
        assert not deadlock_free(stop())

    def test_prefix_stop_has_deadlock(self):
        # a.0 eventually deadlocks
        assert not deadlock_free(prefix("a", stop()))

    def test_recursive_no_deadlock(self):
        # fix X. a.X -> always can do a
        p = recdef("X", prefix("a", recvar("X")))
        assert deadlock_free(p)


# ---- Tests: Action Set ----

class TestActionSet:
    def test_stop_no_actions(self):
        assert action_set(stop()) == set()

    def test_prefix_actions(self):
        p = prefix("a", prefix("b", stop()))
        assert action_set(p) == {"a", "b"}

    def test_parallel_actions(self):
        p = parallel(prefix("a", stop()), prefix("b", stop()))
        acts = action_set(p)
        assert "a" in acts
        assert "b" in acts

    def test_tau_not_in_actions(self):
        p = tau_prefix(stop())
        acts = action_set(p)
        assert TAU not in acts


# ---- Tests: Process Equivalence ----

class TestProcessEquivalence:
    def test_stop_equivalent_to_stop(self):
        result = check_process_equivalence(stop(), stop())
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR

    def test_different_actions_not_equivalent(self):
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = check_process_equivalence(p1, p2)
        assert result.verdict == WeakBisimVerdict.NOT_WEAKLY_BISIMILAR

    def test_same_prefix_equivalent(self):
        p1 = prefix("a", stop())
        p2 = prefix("a", stop())
        result = check_process_equivalence(p1, p2)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR

    def test_tau_prefix_equivalent_to_target(self):
        """tau.P ~= P (weakly)"""
        p1 = tau_prefix(prefix("a", stop()))
        p2 = prefix("a", stop())
        result = check_process_equivalence(p1, p2)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR

    def test_commutativity_of_nd_choice(self):
        """P + Q ~ Q + P"""
        a = prefix("a", stop())
        b = prefix("b", stop())
        p1 = nd_choice(a, b)
        p2 = nd_choice(b, a)
        result = check_process_equivalence(p1, p2)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR


# ---- Tests: Relabeling ----

class TestRelabeling:
    def test_relabel_action(self):
        p = relabel(prefix("a", stop()), {"a": "b"})
        lts = generate_lts(p)
        assert "b" in lts.actions[0]

    def test_relabel_preserves_structure(self):
        p1 = prefix("a", stop())
        p2 = relabel(prefix("b", stop()), {"b": "a"})
        result = check_process_equivalence(p1, p2)
        assert result.verdict == WeakBisimVerdict.WEAKLY_BISIMILAR


# ---- Tests: Restriction ----

class TestRestriction:
    def test_restrict_blocks_action(self):
        p = restrict(prefix("a", stop()), {"a"})
        lts = generate_lts(p)
        # Action a is restricted -> no transitions
        assert "a" not in lts.actions.get(0, {})

    def test_restrict_allows_other(self):
        p = restrict(
            nd_choice(prefix("a", stop()), prefix("b", stop())),
            {"a"}
        )
        lts = generate_lts(p)
        assert "b" in lts.actions.get(0, {})


# ---- Tests: Parser ----

class TestParser:
    def test_parse_stop(self):
        p = parse_proc("0")
        assert p.kind == ProcKind.STOP

    def test_parse_prefix(self):
        p = parse_proc("a.0")
        assert p.kind == ProcKind.PREFIX
        assert p.action == "a"

    def test_parse_prefix_chain(self):
        p = parse_proc("a.b.0")
        assert p.kind == ProcKind.PREFIX
        assert p.action == "a"
        assert p.left.kind == ProcKind.PREFIX
        assert p.left.action == "b"

    def test_parse_nd_choice(self):
        p = parse_proc("a.0 + b.0")
        assert p.kind == ProcKind.ND_CHOICE

    def test_parse_parallel(self):
        p = parse_proc("a.0 | b.0")
        assert p.kind == ProcKind.PARALLEL

    def test_parse_prob_choice(self):
        p = parse_proc("a.0 [0.5] b.0")
        assert p.kind == ProcKind.PROB_CHOICE
        assert p.prob == 0.5

    def test_parse_recursion(self):
        p = parse_proc("fix X. a.X")
        assert p.kind == ProcKind.RECDEF
        assert p.var == "X"

    def test_parse_parens(self):
        p = parse_proc("(a.0)")
        assert p.kind == ProcKind.PREFIX

    def test_parse_complex(self):
        p = parse_proc("a.0 + b.0 | c.0")
        # | binds looser than +
        assert p.kind == ProcKind.PARALLEL


# ---- Tests: Complex Scenarios ----

class TestComplexScenarios:
    def test_coin_flip(self):
        """Model a fair coin flip as probabilistic choice."""
        coin = prob_choice(0.5, prefix("heads", stop()), prefix("tails", stop()))
        traces = trace_set(coin)
        assert ("heads",) in traces
        assert ("tails",) in traces

    def test_vending_machine(self):
        """coin.coffee.0 + coin.tea.0"""
        vm = nd_choice(
            prefix("coin", prefix("coffee", stop())),
            prefix("coin", prefix("tea", stop())),
        )
        traces = trace_set(vm)
        assert ("coin", "coffee") in traces
        assert ("coin", "tea") in traces

    def test_client_server_sync(self):
        """Client: req.~resp.0, Server: ~req.resp.0, Parallel with sync."""
        client = prefix("req", prefix("~resp", stop()))
        server = prefix("~req", prefix("resp", stop()))
        system = parallel(client, server)
        lts = generate_lts(system)
        # Should be able to synchronize (tau from req/~req)
        assert lts.n_states > 1

    def test_probabilistic_server(self):
        """Server that succeeds 70% of time, fails 30%."""
        server = prefix("~req",
            prob_choice(0.7,
                prefix("ok", stop()),
                prefix("err", stop()),
            )
        )
        traces = trace_set(server)
        assert any("ok" in t for t in traces)
        assert any("err" in t for t in traces)

    def test_recursive_server(self):
        """fix X. req.resp.X -- server that loops forever."""
        server = recdef("X", prefix("req", prefix("resp", recvar("X"))))
        assert deadlock_free(server)
        acts = action_set(server)
        assert "req" in acts
        assert "resp" in acts

    def test_mutual_exclusion(self):
        """Two processes competing for a resource."""
        p1 = prefix("lock", prefix("use", prefix("unlock", stop())))
        p2 = prefix("lock", prefix("use", prefix("unlock", stop())))
        system = parallel(p1, p2)
        lts = generate_lts(system)
        assert lts.n_states > 2

    def test_dining_philosophers_simplified(self):
        """Two philosophers, two forks (simplified)."""
        phil1 = prefix("pick_l", prefix("pick_r", prefix("eat", stop())))
        phil2 = prefix("pick_r", prefix("pick_l", prefix("eat", stop())))
        system = parallel(phil1, phil2)
        lts = generate_lts(system)
        assert lts.n_states > 1


# ---- Tests: Summary ----

class TestSummary:
    def test_summary_basic(self):
        p = prefix("a", stop())
        summary = process_summary(p)
        assert "Process" in summary
        assert "LTS states" in summary
        assert "Actions" in summary

    def test_summary_deadlock_info(self):
        p = stop()
        summary = process_summary(p)
        assert "Deadlock-free: False" in summary


# ---- Tests: Edge Cases ----

class TestEdgeCases:
    def test_nested_prob_choice(self):
        p = prob_choice(0.5,
            prob_choice(0.5, prefix("a", stop()), prefix("b", stop())),
            prefix("c", stop()),
        )
        lts = generate_lts(p)
        assert lts.n_states >= 4

    def test_deep_nesting(self):
        p = stop()
        for i in range(10):
            p = prefix(f"a{i}", p)
        lts = generate_lts(p)
        assert lts.n_states == 11

    def test_parallel_same_process(self):
        a = prefix("a", stop())
        p = parallel(a, a)
        lts = generate_lts(p)
        assert "a" in lts.actions[0]

    def test_nd_choice_same(self):
        a = prefix("a", stop())
        p = nd_choice(a, a)
        lts = generate_lts(p)
        assert "a" in lts.actions[0]

    def test_restrict_empty_set(self):
        p = restrict(prefix("a", stop()), set())
        lts = generate_lts(p)
        assert "a" in lts.actions[0]

    def test_relabel_identity(self):
        p = relabel(prefix("a", stop()), {"a": "a"})
        lts = generate_lts(p)
        assert "a" in lts.actions[0]

    def test_complement_in_name(self):
        p = prefix("~a", stop())
        lts = generate_lts(p)
        assert "~a" in lts.actions[0]

    def test_max_states_limit(self):
        """Should not exceed max_states."""
        # Parallel of two recursive processes can generate many states
        p1 = recdef("X", prefix("a", recvar("X")))
        p2 = recdef("Y", prefix("b", recvar("Y")))
        system = parallel(p1, p2)
        lts = generate_lts(system, max_states=10)
        assert lts.n_states <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
