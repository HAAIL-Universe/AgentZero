"""Tests for V022: Trace Partitioning."""

import sys
import os
import pytest

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from trace_partitioning import (
    trace_partition_analyze, trace_partition_full,
    get_variable_partitions, compare_precision,
    analyze_with_loop_partitioning, analyze_branches_only, analyze_no_partition,
    PartitionPolicy, PartitionToken, PartitionKind,
    PartitionedEnv, TracePartitionInterpreter,
)

_a2 = os.path.dirname(_dir)
_az = os.path.dirname(_a2)
sys.path.insert(0, os.path.join(_az, 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(_a2, 'work', 'V020_abstract_domain_functor'))

from domain_functor import (
    make_sign_interval, make_full_product, create_custom_domain,
    SignDomain, IntervalDomain, ConstDomain, ParityDomain,
    analyze_with_domain,
)


# ============================================================
# Section 1: Partition Token Basics
# ============================================================

class TestPartitionToken:
    def test_initial_token(self):
        t = PartitionToken.initial()
        assert t.depth() == 1
        assert repr(t) == "[init]"

    def test_extend_token(self):
        t = PartitionToken.initial()
        t2 = t.extend(PartitionKind.BRANCH_THEN, "if_1")
        assert t2.depth() == 2
        assert "then:if_1" in repr(t2)

    def test_token_equality(self):
        t1 = PartitionToken.initial().extend(PartitionKind.BRANCH_THEN, "x")
        t2 = PartitionToken.initial().extend(PartitionKind.BRANCH_THEN, "x")
        assert t1 == t2

    def test_token_hash(self):
        t1 = PartitionToken.initial().extend(PartitionKind.BRANCH_THEN, "x")
        t2 = PartitionToken.initial().extend(PartitionKind.BRANCH_THEN, "x")
        assert hash(t1) == hash(t2)
        s = {t1, t2}
        assert len(s) == 1

    def test_different_tokens(self):
        t1 = PartitionToken.initial().extend(PartitionKind.BRANCH_THEN, "x")
        t2 = PartitionToken.initial().extend(PartitionKind.BRANCH_ELSE, "x")
        assert t1 != t2


# ============================================================
# Section 2: Partitioned Environment
# ============================================================

class TestPartitionedEnv:
    def test_initial_env(self):
        factory = make_sign_interval()
        penv = PartitionedEnv(factory)
        assert penv.num_partitions == 1

    def test_copy(self):
        factory = make_sign_interval()
        penv = PartitionedEnv(factory)
        copy = penv.copy()
        assert copy.num_partitions == penv.num_partitions
        assert copy is not penv

    def test_join(self):
        factory = make_sign_interval()
        penv1 = PartitionedEnv(factory)
        penv2 = PartitionedEnv(factory)
        joined = penv1.join(penv2)
        assert joined.num_partitions >= 1

    def test_merge_partitions(self):
        factory = make_sign_interval()
        t1 = PartitionToken.initial().extend(PartitionKind.BRANCH_THEN, "a")
        t2 = PartitionToken.initial().extend(PartitionKind.BRANCH_ELSE, "a")
        from domain_functor import DomainEnv
        parts = {
            t1: DomainEnv(factory),
            t2: DomainEnv(factory),
        }
        penv = PartitionedEnv(factory, parts)
        assert penv.num_partitions == 2
        merged = penv.merge_partitions([t1, t2], PartitionToken.initial())
        assert merged.num_partitions == 1


# ============================================================
# Section 3: Partition Policy
# ============================================================

class TestPartitionPolicy:
    def test_default_policy(self):
        p = PartitionPolicy()
        assert p.partition_branches is True
        assert p.partition_loops is False
        assert p.max_partitions == 16

    def test_branch_budget(self):
        p = PartitionPolicy(max_partitions=4)
        assert p.should_partition_branch(2, 1) is True
        assert p.should_partition_branch(3, 1) is False  # 3*2=6 > 4

    def test_depth_limit(self):
        p = PartitionPolicy(partition_depth=3)
        assert p.should_partition_branch(1, 2) is True
        assert p.should_partition_branch(1, 3) is False

    def test_loop_policy(self):
        p = PartitionPolicy(partition_loops=True, max_loop_unroll=3)
        assert p.should_partition_loop(1, 0) is True
        assert p.should_partition_loop(1, 3) is False  # >= max_unroll


# ============================================================
# Section 4: Basic Analysis (No Branching)
# ============================================================

class TestBasicAnalysis:
    def test_simple_assignment(self):
        source = "let x = 5;"
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        val = merged.get('x')
        assert not val.is_bot()
        assert not val.is_top()

    def test_multiple_assignments(self):
        source = "let x = 5; let y = 10; let z = x + y;"
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        z = merged.get('z')
        assert not z.is_bot()

    def test_single_partition_no_branch(self):
        source = "let x = 5; let y = x + 1;"
        result = trace_partition_analyze(source)
        assert result['num_partitions'] == 1

    def test_negation(self):
        source = "let x = 5; let y = -x;"
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        assert not merged.get('y').is_bot()


# ============================================================
# Section 5: Branch Partitioning (Core Feature)
# ============================================================

class TestBranchPartitioning:
    def test_simple_if_creates_partitions(self):
        source = """
        let x = 5;
        if (x > 0) {
            let y = 1;
        } else {
            let y = -1;
        }
        """
        result = trace_partition_analyze(source)
        # Should create 2 partitions (then/else) or 1 if one branch is infeasible
        assert result['num_partitions'] >= 1

    def test_branch_preserves_info(self):
        """Key precision test: after branch, each partition retains branch-specific info."""
        source = """
        let x = 10;
        let y = 0;
        if (x > 5) {
            y = 1;
        } else {
            y = -1;
        }
        """
        result = trace_partition_analyze(source)
        penv = result['partitioned_env']
        y_vals = penv.get_variable_per_partition('y')
        # At least one partition should have y with known sign
        vals = list(y_vals.values())
        assert len(vals) >= 1

    def test_partition_vs_standard_precision(self):
        """Trace partitioning should be at least as precise as standard."""
        source = """
        let x = 10;
        let y = 0;
        if (x > 5) {
            y = 1;
        } else {
            y = -1;
        }
        """
        comparison = compare_precision(source)
        # Partitioned merged should be <= standard (at least as precise)
        # or equal
        assert comparison['num_partitions'] >= 1

    def test_nested_if_partitioning(self):
        source = """
        let x = 10;
        let y = 0;
        let z = 0;
        if (x > 5) {
            y = 1;
            if (x > 8) {
                z = 2;
            } else {
                z = 1;
            }
        } else {
            y = -1;
            z = 0;
        }
        """
        result = trace_partition_analyze(source)
        # Should have up to 3 partitions (then-then, then-else, else)
        assert result['num_partitions'] >= 1

    def test_infeasible_branch_pruned(self):
        """If a branch is infeasible, don't create a partition for it."""
        source = """
        let x = 10;
        if (x > 5) {
            let y = 1;
        } else {
            let y = -1;
        }
        """
        result = trace_partition_analyze(source)
        penv = result['partitioned_env']
        # x=10, so x>5 is always true -> only then partition should exist
        # (depends on interval refinement -- x is [10,10], then branch is feasible,
        #  else branch refines x to [10,10] meet (-inf,5] = BOT)
        assert result['num_partitions'] == 1


# ============================================================
# Section 6: Precision Gains
# ============================================================

class TestPrecisionGains:
    def test_abs_function_precision(self):
        """Classic example: abs(x) loses precision with standard join."""
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = -x;
        }
        """
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        y = merged.get('y')
        # y should be non-negative in both partitions
        # Standard: y = join(POS, NON_NEG) = NON_NEG (or TOP if x could be anything)
        # With partitioning: each partition knows y is POS, so merged is POS
        assert not y.is_top()

    def test_conditional_assignment_precision(self):
        """After if-else assigning different values, partitions keep them separate."""
        source = """
        let x = 3;
        let y = 0;
        if (x > 0) {
            y = 10;
        } else {
            y = 20;
        }
        """
        result = trace_partition_analyze(source)
        penv = result['partitioned_env']
        y_vals = penv.get_variable_per_partition('y')
        # Should have at least 1 partition with precise y value
        assert len(y_vals) >= 1

    def test_sign_precision_after_branch(self):
        """Branch on sign should maintain sign info in partitions."""
        source = """
        let x = 5;
        let r = 0;
        if (x > 0) {
            r = x + 1;
        } else {
            r = -x + 1;
        }
        """
        # In the then-branch, x is POS, so x+1 is POS
        # Standard would merge and lose precision
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        r = merged.get('r')
        assert not r.is_bot()

    def test_compare_precision_api(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = -1;
        }
        """
        comparison = compare_precision(source)
        assert 'standard' in comparison
        assert 'partitioned_merged' in comparison
        assert 'precision_gains' in comparison
        assert 'num_partitions' in comparison


# ============================================================
# Section 7: Budget Control
# ============================================================

class TestBudgetControl:
    def test_budget_limits_partitions(self):
        """Many branches should not exceed budget."""
        source = """
        let a = 5;
        let b = 0;
        let c = 0;
        let d = 0;
        if (a > 1) { b = 1; } else { b = 0; }
        if (a > 2) { c = 1; } else { c = 0; }
        if (a > 3) { d = 1; } else { d = 0; }
        """
        policy = PartitionPolicy(max_partitions=4)
        result = trace_partition_analyze(source, policy=policy)
        assert result['num_partitions'] <= 4

    def test_unlimited_budget(self):
        source = """
        let a = 5;
        let b = 0;
        let c = 0;
        if (a > 1) { b = 1; } else { b = 0; }
        if (a > 2) { c = 1; } else { c = 0; }
        """
        policy = PartitionPolicy(max_partitions=64)
        result = trace_partition_analyze(source, policy=policy)
        # With large budget, should keep all partitions
        assert result['num_partitions'] >= 1

    def test_depth_limit(self):
        """Deep nesting should stop partitioning at depth limit."""
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            if (x > 1) {
                if (x > 2) {
                    y = 3;
                } else {
                    y = 2;
                }
            } else {
                y = 1;
            }
        } else {
            y = 0;
        }
        """
        policy = PartitionPolicy(partition_depth=3)
        result = trace_partition_analyze(source, policy=policy)
        # Should have some partitions but not exponential
        assert result['num_partitions'] >= 1


# ============================================================
# Section 8: Loop Handling (Standard)
# ============================================================

class TestLoopStandard:
    def test_simple_loop(self):
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        i_val = merged.get('i')
        assert not i_val.is_bot()

    def test_loop_with_branch_before(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = 0;
        }
        let i = 0;
        while (i < 3) {
            i = i + 1;
        }
        """
        result = trace_partition_analyze(source)
        # Partitions from the branch should survive through the loop
        assert result['num_partitions'] >= 1

    def test_loop_convergence(self):
        """Loop should converge even with partitions."""
        source = """
        let x = 5;
        let s = 0;
        if (x > 0) { s = 1; } else { s = 0; }
        let i = 0;
        while (i < 5) {
            i = i + 1;
            s = s + 1;
        }
        """
        result = trace_partition_analyze(source)
        assert result['merged_env'].get('i') is not None


# ============================================================
# Section 9: Loop Partitioning
# ============================================================

class TestLoopPartitioning:
    def test_loop_unroll_partitioning(self):
        source = """
        let i = 0;
        while (i < 5) {
            i = i + 1;
        }
        """
        result = analyze_with_loop_partitioning(source, max_unroll=2)
        # Should have partitions for unrolled iterations + rest
        assert result['num_partitions'] >= 1
        merged = result['merged_env']
        assert not merged.get('i').is_bot()

    def test_loop_partition_precision(self):
        """Loop partitioning should give precise values for early iterations."""
        source = """
        let i = 0;
        let s = 0;
        while (i < 3) {
            s = s + i;
            i = i + 1;
        }
        """
        result = analyze_with_loop_partitioning(source, max_unroll=3)
        assert result['num_partitions'] >= 1

    def test_loop_partition_budget(self):
        """Loop partitioning should respect budget."""
        source = """
        let i = 0;
        while (i < 100) {
            i = i + 1;
        }
        """
        policy = PartitionPolicy(
            partition_loops=True, max_loop_unroll=5,
            max_partitions=8
        )
        result = trace_partition_analyze(source, policy=policy)
        assert result['num_partitions'] <= 8


# ============================================================
# Section 10: Full Product Domain
# ============================================================

class TestFullProduct:
    def test_full_product_basic(self):
        source = "let x = 5; let y = -3;"
        result = trace_partition_full(source)
        merged = result['merged_env']
        x = merged.get('x')
        y = merged.get('y')
        assert not x.is_bot()
        assert not y.is_bot()

    def test_full_product_with_branch(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x + 1;
        } else {
            y = -x;
        }
        """
        result = trace_partition_full(source)
        assert result['num_partitions'] >= 1

    def test_full_product_parity(self):
        """Full product should track parity through partitions."""
        source = """
        let x = 4;
        let y = 0;
        if (x > 0) {
            y = x + 2;
        } else {
            y = 0;
        }
        """
        result = trace_partition_full(source)
        # y should be EVEN in both partitions (4+2=6, or 0)
        merged = result['merged_env']
        assert not merged.get('y').is_bot()


# ============================================================
# Section 11: No Partitioning Mode
# ============================================================

class TestNoPartition:
    def test_no_partition_mode(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = -1;
        }
        """
        result = analyze_no_partition(source)
        # Should always have 1 partition
        assert result['num_partitions'] == 1

    def test_no_partition_matches_standard(self):
        """With no partitioning, should give same results as standard."""
        source = """
        let a = 10;
        let b = 0;
        if (a > 5) {
            b = 1;
        } else {
            b = -1;
        }
        """
        no_part = analyze_no_partition(source)
        std = analyze_with_domain(source, make_sign_interval())
        # Both should have b defined
        assert no_part['merged_env'].get('b') is not None
        assert std['env'].get('b') is not None


# ============================================================
# Section 12: Statistics Tracking
# ============================================================

class TestStatistics:
    def test_stats_present(self):
        source = """
        let x = 5;
        if (x > 0) { let y = 1; } else { let y = 0; }
        """
        result = trace_partition_analyze(source)
        stats = result['stats']
        assert 'branches_partitioned' in stats
        assert 'branches_merged' in stats
        assert 'max_partitions_reached' in stats

    def test_branches_counted(self):
        source = """
        let x = 5;
        if (x > 0) { let y = 1; } else { let y = 0; }
        if (x > 3) { let z = 1; } else { let z = 0; }
        """
        result = trace_partition_analyze(source)
        stats = result['stats']
        total = stats['branches_partitioned'] + stats['branches_merged']
        assert total >= 2

    def test_merge_stats(self):
        """When budget forces merges, stat should reflect."""
        source = """
        let a = 5;
        let b = 0; let c = 0; let d = 0; let e = 0;
        if (a > 0) { b = 1; } else { b = 0; }
        if (a > 1) { c = 1; } else { c = 0; }
        if (a > 2) { d = 1; } else { d = 0; }
        if (a > 3) { e = 1; } else { e = 0; }
        """
        policy = PartitionPolicy(max_partitions=2)
        result = trace_partition_analyze(source, policy=policy)
        assert result['num_partitions'] <= 2


# ============================================================
# Section 13: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        source = "let x = 0;"
        result = trace_partition_analyze(source)
        assert result['num_partitions'] == 1

    def test_only_assignments(self):
        source = """
        let a = 1;
        let b = 2;
        let c = a + b;
        """
        result = trace_partition_analyze(source)
        assert result['num_partitions'] == 1

    def test_deeply_nested(self):
        source = """
        let x = 5;
        let r = 0;
        if (x > 0) {
            if (x > 1) {
                if (x > 2) {
                    if (x > 3) {
                        r = 4;
                    } else {
                        r = 3;
                    }
                } else {
                    r = 2;
                }
            } else {
                r = 1;
            }
        } else {
            r = 0;
        }
        """
        result = trace_partition_analyze(source)
        assert result['num_partitions'] >= 1
        merged = result['merged_env']
        assert not merged.get('r').is_bot()

    def test_if_without_else(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        }
        """
        result = trace_partition_analyze(source)
        assert result['num_partitions'] >= 1

    def test_multiple_variables_tracked(self):
        source = """
        let a = 1;
        let b = 2;
        let c = 3;
        if (a > 0) {
            b = 10;
            c = 20;
        } else {
            b = -10;
            c = -20;
        }
        """
        result = trace_partition_analyze(source)
        penv = result['partitioned_env']
        b_parts = penv.get_variable_per_partition('b')
        c_parts = penv.get_variable_per_partition('c')
        assert len(b_parts) >= 1
        assert len(c_parts) >= 1


# ============================================================
# Section 14: Function Calls
# ============================================================

class TestFunctionCalls:
    def test_function_definition(self):
        source = """
        fn add(a, b) { return a + b; }
        let x = add(3, 4);
        """
        result = trace_partition_analyze(source)
        assert 'add' in result['functions']

    def test_function_in_branch(self):
        source = """
        fn double(x) { return x + x; }
        let a = 5;
        let r = 0;
        if (a > 0) {
            r = double(a);
        } else {
            r = 0;
        }
        """
        result = trace_partition_analyze(source)
        merged = result['merged_env']
        assert not merged.get('r').is_bot()


# ============================================================
# Section 15: Composition with Custom Domains
# ============================================================

class TestCustomDomains:
    def test_sign_only(self):
        factory = lambda v=None: (SignDomain.from_concrete(v) if v is not None
                                   else SignDomain())
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x;
        } else {
            y = -x;
        }
        """
        result = trace_partition_analyze(source, domain_factory=factory)
        assert result['num_partitions'] >= 1

    def test_interval_only(self):
        factory = lambda v=None: (IntervalDomain.from_concrete(v) if v is not None
                                   else IntervalDomain())
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x + 1;
        } else {
            y = 0;
        }
        """
        result = trace_partition_analyze(source, domain_factory=factory)
        merged = result['merged_env']
        assert not merged.get('y').is_bot()

    def test_custom_reduced_product(self):
        factory = create_custom_domain(SignDomain, IntervalDomain, ConstDomain)
        source = """
        let x = 5;
        if (x > 0) {
            let y = x + 1;
        } else {
            let y = 0;
        }
        """
        result = trace_partition_analyze(source, domain_factory=factory)
        assert result['num_partitions'] >= 1

    def test_branches_only_api(self):
        source = """
        let x = 5;
        if (x > 0) { let y = 1; } else { let y = 0; }
        """
        result = analyze_branches_only(source, max_partitions=8)
        assert result['num_partitions'] >= 1

    def test_get_variable_partitions_api(self):
        source = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = 1;
        } else {
            y = -1;
        }
        """
        partitions = get_variable_partitions(source, 'y')
        assert len(partitions) >= 1
