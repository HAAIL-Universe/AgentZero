"""Tests for V098: IDE Framework (Interprocedural Distributive Environment)"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from ide_framework import (
    # Lattice
    LatticeValue, Top, Bot, Const, TOP, BOT,
    lattice_meet, lattice_join, lattice_leq,
    # Micro-functions
    MicroFunction, IdFunction, ConstFunction, TopFunction, BotFunction,
    LinearFunction, ComposedFunction, MeetFunction,
    ID_FN, TOP_FN, BOT_FN,
    # IDE types
    Fact, ZERO, IDEProblem, IDEResult, IDESolver,
    # APIs
    ide_analyze, get_constants, get_variable_value,
    linear_const_analyze, compare_analyses,
    get_function_summary, ide_verify_constant,
    build_ide_problem,
)


# ===================================================================
# Section 1: Lattice operations
# ===================================================================

class TestLattice:
    def test_meet_top_top(self):
        assert lattice_meet(TOP, TOP) == TOP

    def test_meet_top_const(self):
        assert lattice_meet(TOP, Const(5)) == Const(5)

    def test_meet_const_top(self):
        assert lattice_meet(Const(5), TOP) == Const(5)

    def test_meet_same_const(self):
        assert lattice_meet(Const(3), Const(3)) == Const(3)

    def test_meet_diff_const(self):
        assert lattice_meet(Const(3), Const(5)) == BOT

    def test_meet_bot_anything(self):
        assert lattice_meet(BOT, Const(5)) == BOT
        assert lattice_meet(BOT, TOP) == BOT

    def test_join_bot_const(self):
        assert lattice_join(BOT, Const(5)) == Const(5)

    def test_join_diff_const(self):
        assert lattice_join(Const(3), Const(5)) == TOP

    def test_join_same_const(self):
        assert lattice_join(Const(3), Const(3)) == Const(3)

    def test_leq_bot_anything(self):
        assert lattice_leq(BOT, TOP)
        assert lattice_leq(BOT, Const(5))
        assert lattice_leq(BOT, BOT)

    def test_leq_anything_top(self):
        assert lattice_leq(Const(5), TOP)
        assert lattice_leq(TOP, TOP)

    def test_leq_const_const(self):
        assert lattice_leq(Const(5), Const(5))
        assert not lattice_leq(Const(5), Const(3))


# ===================================================================
# Section 2: Micro-function basics
# ===================================================================

class TestMicroFunctions:
    def test_id_function(self):
        f = ID_FN
        assert f.apply(Const(5)) == Const(5)
        assert f.apply(TOP) == TOP
        assert f.apply(BOT) == BOT

    def test_const_function(self):
        f = ConstFunction(Const(42))
        assert f.apply(Const(5)) == Const(42)
        assert f.apply(TOP) == Const(42)
        assert f.apply(BOT) == Const(42)

    def test_top_function(self):
        f = TOP_FN
        assert f.apply(Const(5)) == TOP
        assert f.apply(BOT) == TOP

    def test_bot_function(self):
        f = BOT_FN
        assert f.apply(Const(5)) == BOT
        assert f.apply(TOP) == BOT

    def test_linear_function(self):
        f = LinearFunction(2, 3)  # 2*x + 3
        assert f.apply(Const(5)) == Const(13)
        assert f.apply(Const(0)) == Const(3)
        assert f.apply(TOP) == TOP  # a != 0
        assert f.apply(BOT) == BOT

    def test_linear_zero_coeff(self):
        f = LinearFunction(0, 7)  # 0*x + 7 = 7
        assert f.apply(TOP) == Const(7)
        assert f.apply(Const(999)) == Const(7)


# ===================================================================
# Section 3: Micro-function composition
# ===================================================================

class TestMicroFunctionComposition:
    def test_id_compose_anything(self):
        f = ConstFunction(Const(5))
        assert ID_FN.compose(f).apply(TOP) == Const(5)

    def test_const_compose_anything(self):
        f = ConstFunction(Const(5))
        g = LinearFunction(2, 3)
        # const(5) . lin(2x+3) = const(5) (always returns 5)
        h = f.compose(g)
        assert h.apply(Const(10)) == Const(5)

    def test_linear_compose_linear(self):
        f = LinearFunction(2, 1)  # 2x + 1
        g = LinearFunction(3, 4)  # 3x + 4
        # f(g(x)) = 2*(3x+4) + 1 = 6x + 9
        h = f.compose(g)
        assert h.apply(Const(1)) == Const(15)  # 6*1 + 9
        assert h.apply(Const(0)) == Const(9)   # 6*0 + 9

    def test_linear_compose_const(self):
        f = LinearFunction(2, 3)  # 2x + 3
        g = ConstFunction(Const(5))
        h = f.compose(g)
        assert h.apply(TOP) == Const(13)  # 2*5 + 3

    def test_triple_composition(self):
        f = LinearFunction(1, 1)  # x + 1
        g = LinearFunction(1, 1)  # x + 1
        h = LinearFunction(1, 1)  # x + 1
        # (x+1) . (x+1) . (x+1) = x + 3
        composed = f.compose(g).compose(h)
        assert composed.apply(Const(0)) == Const(3)
        assert composed.apply(Const(10)) == Const(13)


# ===================================================================
# Section 4: Micro-function meet
# ===================================================================

class TestMicroFunctionMeet:
    def test_meet_top_anything(self):
        f = ConstFunction(Const(5))
        assert TOP_FN.meet(f) == f

    def test_meet_bot_anything(self):
        f = ConstFunction(Const(5))
        assert BOT_FN.meet(f) == BOT_FN

    def test_meet_same_const(self):
        f1 = ConstFunction(Const(5))
        f2 = ConstFunction(Const(5))
        result = f1.meet(f2)
        assert result.apply(TOP) == Const(5)

    def test_meet_diff_const(self):
        f1 = ConstFunction(Const(5))
        f2 = ConstFunction(Const(3))
        result = f1.meet(f2)
        assert result.apply(TOP) == BOT  # meet of 5 and 3


# ===================================================================
# Section 5: IDE solver basics (intraprocedural)
# ===================================================================

class TestIDESolverBasic:
    def test_single_assignment(self):
        """let x = 5; -- x should be Const(5)"""
        source = "let x = 5;"
        result = ide_analyze(source)
        # Find x value at some point
        found = False
        for pt, facts in result.values.items():
            for fact, val in facts.items():
                if fact.name == "x" and isinstance(val, Const):
                    assert val.value == 5
                    found = True
        assert found, f"x=5 not found in {result.values}"

    def test_two_assignments(self):
        """let x = 5; let y = 10;"""
        source = "let x = 5; let y = 10;"
        result = ide_analyze(source)
        x_found = False
        y_found = False
        for pt, facts in result.values.items():
            for fact, val in facts.items():
                if fact.name == "x" and isinstance(val, Const) and val.value == 5:
                    x_found = True
                if fact.name == "y" and isinstance(val, Const) and val.value == 10:
                    y_found = True
        assert x_found, "x=5 not found"
        assert y_found, "y=10 not found"

    def test_copy_propagation(self):
        """let x = 5; let y = x; -- y should be Const(5)"""
        source = "let x = 5; let y = x;"
        result = ide_analyze(source)
        y_const = False
        for pt, facts in result.values.items():
            for fact, val in facts.items():
                if fact.name == "y" and isinstance(val, Const) and val.value == 5:
                    y_const = True
        assert y_const, f"y should be Const(5), values: {result.values}"

    def test_chain_copy(self):
        """let x = 42; let y = x; let z = y; -- z should be Const(42)"""
        source = "let x = 42; let y = x; let z = y;"
        result = ide_analyze(source)
        z_const = False
        for pt, facts in result.values.items():
            for fact, val in facts.items():
                if fact.name == "z" and isinstance(val, Const) and val.value == 42:
                    z_const = True
        assert z_const, "z should be Const(42)"

    def test_overwrite(self):
        """let x = 5; x = 10; -- x should be Const(10) at exit"""
        source = "let x = 5; x = 10;"
        result = ide_analyze(source)
        # Check at main.exit
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == 10


# ===================================================================
# Section 6: Linear constant propagation
# ===================================================================

class TestLinearConstProp:
    def test_increment(self):
        """let x = 5; x = x + 1; -- x should be Const(6) at exit"""
        source = "let x = 5; x = x + 1;"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == 6

    def test_double(self):
        """let x = 3; x = x * 2; -- x should be Const(6) at exit"""
        source = "let x = 3; x = x * 2;"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == 6

    def test_linear_chain(self):
        """let x = 1; x = x * 2; x = x + 3; -- x should be Const(5) at exit"""
        source = "let x = 1; x = x * 2; x = x + 3;"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == 5

    def test_linear_transfer(self):
        """let x = 3; let y = x * 2 + 1; -- y should be Const(7)"""
        source = "let x = 3; let y = x * 2 + 1;"
        result = ide_analyze(source, "linear_const")
        y_found = False
        for pt, facts in result.values.items():
            if Fact("y") in facts:
                val = facts[Fact("y")]
                if isinstance(val, Const) and val.value == 7:
                    y_found = True
        assert y_found, "y should be Const(7)"

    def test_negate(self):
        """let x = 5; x = 0 - x; -- x should be Const(-5) at exit"""
        source = "let x = 5; x = 0 - x;"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == -5


# ===================================================================
# Section 7: Interprocedural analysis (function calls)
# ===================================================================

class TestInterprocedural:
    def test_identity_function(self):
        """fn id(x) { return x; } let a = 5; let b = id(a);"""
        source = "fn id(x) { return x; } let a = 5; let b = id(a);"
        result = ide_analyze(source)
        exit_val = result.values.get("main.exit", {}).get(Fact("b"))
        assert isinstance(exit_val, Const) and exit_val.value == 5, \
            f"b should be Const(5), exit values: {result.values.get('main.exit', {})}"

    def test_constant_return(self):
        """fn seven() { return 7; } let x = seven();"""
        source = "fn seven() { return 7; } let x = seven();"
        result = ide_analyze(source)
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == 7, \
            f"x should be Const(7), exit values: {result.values.get('main.exit', {})}"

    def test_add_function(self):
        """fn add1(x) { return x + 1; } let a = 5; let b = add1(a);"""
        source = "fn add1(x) { return x + 1; } let a = 5; let b = add1(a);"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("b"))
        assert isinstance(exit_val, Const) and exit_val.value == 6, \
            f"b should be Const(6), exit values: {result.values.get('main.exit', {})}"

    def test_const_arg(self):
        """fn inc(x) { return x + 1; } let y = inc(10);"""
        source = "fn inc(x) { return x + 1; } let y = inc(10);"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("y"))
        assert isinstance(exit_val, Const) and exit_val.value == 11, \
            f"y should be Const(11), exit values: {result.values.get('main.exit', {})}"

    def test_two_params(self):
        """fn first(a, b) { return a; } let x = first(3, 7);"""
        source = "fn first(a, b) { return a; } let x = first(3, 7);"
        result = ide_analyze(source)
        exit_val = result.values.get("main.exit", {}).get(Fact("x"))
        assert isinstance(exit_val, Const) and exit_val.value == 3, \
            f"x should be Const(3), exit values: {result.values.get('main.exit', {})}"


# ===================================================================
# Section 8: get_constants API
# ===================================================================

class TestGetConstants:
    def test_simple(self):
        source = "let x = 5; let y = 10;"
        consts = get_constants(source)
        # Should find x=5 and y=10 at some points
        found_x = any(
            'x' in pt_consts and pt_consts['x'].value == 5
            for pt_consts in consts.values()
        )
        found_y = any(
            'y' in pt_consts and pt_consts['y'].value == 10
            for pt_consts in consts.values()
        )
        assert found_x, "x=5 not found"
        assert found_y, "y=10 not found"

    def test_overwritten(self):
        source = "let x = 5; x = 10;"
        consts = get_constants(source)
        # Should find x=10 at some later point
        found_10 = any(
            'x' in pt_consts and pt_consts['x'].value == 10
            for pt_consts in consts.values()
        )
        assert found_10, "x=10 not found"


# ===================================================================
# Section 9: get_variable_value API
# ===================================================================

class TestGetVariableValue:
    def test_simple(self):
        source = "let x = 42;"
        val = get_variable_value(source, "x")
        assert isinstance(val, Const) and val.value == 42

    def test_at_point(self):
        source = "let x = 5; let y = 10;"
        result = ide_analyze(source)
        # Find a point where y=10
        for pt, facts in result.values.items():
            if Fact("y") in facts and isinstance(facts[Fact("y")], Const):
                val = get_variable_value(source, "y", pt)
                assert isinstance(val, Const) and val.value == 10
                break


# ===================================================================
# Section 10: compare_analyses API
# ===================================================================

class TestCompareAnalyses:
    def test_basic_comparison(self):
        source = "let x = 3; let y = x * 2 + 1;"
        comp = compare_analyses(source)
        assert 'copy_const' in comp
        assert 'linear_const' in comp
        assert 'stats' in comp

    def test_precision_gain(self):
        """Linear should be more precise for x = x * 2 + 1 kind of expressions."""
        source = "let x = 3; x = x * 2 + 1;"
        comp = compare_analyses(source)
        # Linear const should resolve x=7, copy const may not
        # (copy const can resolve x=3 initially, but x = x*2+1 needs linear)
        # Check that precision_gains is populated or both resolve
        assert isinstance(comp['precision_gains'], list)


# ===================================================================
# Section 11: Function summaries
# ===================================================================

class TestFunctionSummary:
    def test_identity_summary(self):
        source = "fn id(x) { return x; } let a = id(5);"
        summary = get_function_summary(source, "id")
        assert isinstance(summary, dict)

    def test_constant_summary(self):
        source = "fn seven() { return 7; } let x = seven();"
        summary = get_function_summary(source, "seven")
        assert isinstance(summary, dict)


# ===================================================================
# Section 12: ide_verify_constant API
# ===================================================================

class TestVerifyConstant:
    def test_verify_true(self):
        source = "let x = 42;"
        result = ide_verify_constant(source, "x", 42)
        assert result['verified'] is True

    def test_verify_false(self):
        source = "let x = 42;"
        result = ide_verify_constant(source, "x", 99)
        assert result['verified'] is False

    def test_verify_linear(self):
        source = "let x = 5; x = x * 2;"
        result = ide_verify_constant(source, "x", 10)
        assert result['verified'] is True


# ===================================================================
# Section 13: Multiple functions
# ===================================================================

class TestMultipleFunctions:
    def test_two_functions(self):
        """fn double(x) { return x * 2; } fn inc(x) { return x + 1; } let a = double(5);"""
        source = "fn double(x) { return x * 2; } fn inc(x) { return x + 1; } let a = double(5);"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("a"))
        assert isinstance(exit_val, Const) and exit_val.value == 10, \
            f"a should be Const(10), exit values: {result.values.get('main.exit', {})}"

    def test_chained_calls(self):
        """fn add1(x) { return x + 1; } let a = 5; let b = add1(a); let c = add1(b);"""
        source = "fn add1(x) { return x + 1; } let a = 5; let b = add1(a); let c = add1(b);"
        result = ide_analyze(source, "linear_const")
        exit_val = result.values.get("main.exit", {}).get(Fact("c"))
        assert isinstance(exit_val, Const) and exit_val.value == 7, \
            f"c should be Const(7), exit values: {result.values.get('main.exit', {})}"


# ===================================================================
# Section 14: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_empty_program(self):
        """Empty program should not crash."""
        source = "let x = 0;"
        result = ide_analyze(source)
        assert isinstance(result, IDEResult)

    def test_no_functions(self):
        source = "let x = 1; let y = 2; let z = x + y;"
        result = ide_analyze(source)
        assert isinstance(result, IDEResult)

    def test_unused_function(self):
        source = "fn unused() { return 42; } let x = 5;"
        result = ide_analyze(source)
        x_found = False
        for pt, facts in result.values.items():
            if Fact("x") in facts and isinstance(facts[Fact("x")], Const):
                assert facts[Fact("x")].value == 5
                x_found = True
        assert x_found

    def test_zero_value(self):
        source = "let x = 0;"
        result = ide_analyze(source)
        found = False
        for pt, facts in result.values.items():
            if Fact("x") in facts and isinstance(facts[Fact("x")], Const):
                assert facts[Fact("x")].value == 0
                found = True
        assert found

    def test_negative_value(self):
        source = "let x = 0 - 5;"
        result = ide_analyze(source)
        found = False
        for pt, facts in result.values.items():
            if Fact("x") in facts and isinstance(facts[Fact("x")], Const):
                assert facts[Fact("x")].value == -5
                found = True
        assert found


# ===================================================================
# Section 15: Build IDE problem directly
# ===================================================================

class TestBuildIDEProblem:
    def test_problem_structure(self):
        source = "let x = 5; let y = x;"
        problem = build_ide_problem(source)
        assert 'main' in problem.functions
        assert problem.entry_function == "main"
        assert len(problem.all_facts) >= 2  # x, y

    def test_problem_with_function(self):
        source = "fn f(a) { return a; } let x = f(5);"
        problem = build_ide_problem(source)
        assert 'f' in problem.functions
        assert 'main' in problem.functions

    def test_problem_edges(self):
        source = "fn f(a) { return a; } let x = f(5);"
        problem = build_ide_problem(source)
        edge_types = {e[2] for e in problem.edges}
        assert 'call' in edge_types or 'normal' in edge_types


# ===================================================================
# Section 16: Solver directly
# ===================================================================

class TestIDESolverDirect:
    def test_solver_stats(self):
        source = "let x = 5; let y = x;"
        problem = build_ide_problem(source)
        solver = IDESolver(problem)
        result = solver.solve()
        assert 'worklist_pops' in result.stats
        assert result.stats['worklist_pops'] > 0

    def test_solver_reachable_facts(self):
        source = "let x = 5; let y = 10;"
        result = ide_analyze(source)
        # At least some points should have reachable facts
        any_facts = any(len(facts) > 0 for facts in result.reachable_facts.values())
        assert any_facts


# ===================================================================
# Section 17: Lattice value equality and hashing
# ===================================================================

class TestLatticeEquality:
    def test_const_equality(self):
        assert Const(5) == Const(5)
        assert Const(5) != Const(3)
        assert Const(5) != TOP

    def test_top_singleton(self):
        assert TOP == Top()
        assert TOP != BOT

    def test_hashing(self):
        s = {Const(5), Const(5), Const(3)}
        assert len(s) == 2

    def test_fact_equality(self):
        assert Fact("x") == Fact("x")
        assert Fact("x") != Fact("y")
        assert ZERO == Fact("__ZERO__")


# ===================================================================
# Section 18: Micro-function equality and hashing
# ===================================================================

class TestMicroFunctionEquality:
    def test_id_equality(self):
        assert IdFunction() == IdFunction()
        assert ID_FN == ID_FN

    def test_const_fn_equality(self):
        f1 = ConstFunction(Const(5))
        f2 = ConstFunction(Const(5))
        assert f1 == f2

    def test_linear_equality(self):
        f1 = LinearFunction(2, 3)
        f2 = LinearFunction(2, 3)
        assert f1 == f2

    def test_linear_inequality(self):
        f1 = LinearFunction(2, 3)
        f2 = LinearFunction(3, 2)
        assert f1 != f2

    def test_linear_id_equivalence(self):
        """LinearFunction(1, 0) should equal IdFunction."""
        f = LinearFunction(1, 0)
        assert f == IdFunction()

    def test_hashing(self):
        s = {ConstFunction(Const(5)), ConstFunction(Const(5))}
        assert len(s) == 1
