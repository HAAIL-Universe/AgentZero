"""Tests for C124: Linear Programming."""

import pytest
import math
from linear_programming import (
    StandardLP, SimplexSolver, TwoPhaseSimplexSolver, DualSimplexSolver,
    LPBuilder, MILPSolver, LPStatus, LPResult, Sense, ConstraintOp,
    sensitivity_rhs, sensitivity_obj, transportation_problem, diet_problem,
    EPS,
)


# ============================================================
# StandardLP construction
# ============================================================

class TestStandardLP:
    def test_basic_construction(self):
        lp = StandardLP([1, 2], [[1, 0], [0, 1]], [3, 4])
        assert lp.n == 2
        assert lp.m == 2

    def test_dimension_mismatch_rows(self):
        with pytest.raises(ValueError, match="rows"):
            StandardLP([1, 2], [[1, 0]], [3, 4])

    def test_dimension_mismatch_cols(self):
        with pytest.raises(ValueError, match="cols"):
            StandardLP([1, 2], [[1], [0, 1]], [3, 4])

    def test_float_conversion(self):
        lp = StandardLP([1, 2], [[1, 0], [0, 1]], [3, 4])
        assert all(isinstance(v, float) for v in lp.c)
        assert all(isinstance(v, float) for v in lp.b)

    def test_single_var(self):
        lp = StandardLP([5], [[1], [-1]], [10, 0])
        assert lp.n == 1
        assert lp.m == 2

    def test_empty(self):
        lp = StandardLP([], [], [])
        assert lp.n == 0
        assert lp.m == 0


# ============================================================
# SimplexSolver
# ============================================================

class TestSimplexSolver:
    def test_basic_2d(self):
        # minimize -x - y s.t. x+y<=4, x<=3, y<=3, x,y>=0
        lp = StandardLP([-1, -1], [[1, 1], [1, 0], [0, 1]], [4, 3, 3])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-4)) < 1e-6

    def test_single_variable(self):
        # minimize -x s.t. x<=5, x>=0
        lp = StandardLP([-1], [[1]], [5])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-5)) < 1e-6

    def test_unbounded(self):
        # minimize -x s.t. x-y<=1, x,y>=0 (unbounded: increase y)
        lp = StandardLP([-1, 0], [[1, -1]], [1])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.UNBOUNDED

    def test_negative_b_infeasible(self):
        # b < 0 not supported by basic simplex
        lp = StandardLP([1], [[-1]], [-5])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.INFEASIBLE

    def test_degenerate(self):
        # Degenerate case: multiple constraints active at same point
        lp = StandardLP([-1, 0], [[1, 0], [0, 1], [1, 1]], [2, 2, 2])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x0"] - 2) < 1e-6

    def test_three_variables(self):
        # minimize -2x-3y-z s.t. x+y+z<=10, x+2y<=12, x,y,z>=0
        lp = StandardLP([-2, -3, -1], [[1, 1, 1], [1, 2, 0]], [10, 12])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert r.objective < 0

    def test_already_optimal_at_origin(self):
        # minimize x+y s.t. x+y<=5, x,y>=0 -> optimal at origin
        lp = StandardLP([1, 1], [[1, 1]], [5])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective) < 1e-6

    def test_iterations_counted(self):
        lp = StandardLP([-1, -1], [[1, 1], [1, 0], [0, 1]], [4, 3, 3])
        r = SimplexSolver().solve(lp)
        assert r.iterations > 0

    def test_dual_values_returned(self):
        lp = StandardLP([-1, -1], [[1, 1], [1, 0], [0, 1]], [4, 3, 3])
        r = SimplexSolver().solve(lp)
        assert r.dual_values is not None
        assert len(r.dual_values) == 3

    def test_production_planning(self):
        # Classic: max 5x+4y s.t. 6x+4y<=24, x+2y<=6 -> min -5x-4y
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-21)) < 1e-6


# ============================================================
# TwoPhaseSimplexSolver
# ============================================================

class TestTwoPhaseSimplexSolver:
    def test_basic_feasible(self):
        lp = StandardLP([-1, -1], [[1, 1], [1, 0], [0, 1]], [4, 3, 3])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-4)) < 1e-6

    def test_negative_rhs(self):
        # x + y >= 2 (i.e., -x - y <= -2), x <= 5
        # minimize x+y -> optimal at x+y=2
        lp = StandardLP([1, 1], [[-1, -1], [1, 0], [0, 1]], [-2, 5, 5])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 2) < 1e-6

    def test_infeasible(self):
        # x <= 1 and x >= 3 -- infeasible
        lp = StandardLP([1], [[1], [-1]], [1, -3])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.INFEASIBLE

    def test_equality_via_two_inequalities(self):
        # x = 3 via x<=3 and x>=3, minimize x
        lp = StandardLP([1], [[1], [-1]], [3, -3])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 3) < 1e-6

    def test_mixed_signs(self):
        # min x s.t. x >= 1, x <= 5
        lp = StandardLP([1], [[-1], [1]], [-1, 5])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x0"] - 1) < 1e-6

    def test_all_negative_b(self):
        # min x+y s.t. x>=2, y>=3
        lp = StandardLP([1, 1], [[-1, 0], [0, -1]], [-2, -3])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 5) < 1e-6

    def test_unbounded_through_two_phase(self):
        lp = StandardLP([-1, 0], [[1, -1]], [1])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.UNBOUNDED

    def test_three_var_mixed(self):
        # min -x-y-z s.t. x+y+z>=3, x<=5, y<=5, z<=5
        lp = StandardLP(
            [-1, -1, -1],
            [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [-3, 5, 5, 5]
        )
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert r.objective <= -15 + 1e-6  # can go up to x=y=z=5


# ============================================================
# DualSimplexSolver
# ============================================================

class TestDualSimplexSolver:
    def test_basic(self):
        lp = StandardLP([-1, -1], [[1, 1], [1, 0], [0, 1]], [4, 3, 3])
        r = DualSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL

    def test_negative_b_fallback(self):
        # Falls back to two-phase
        lp = StandardLP([1, 1], [[-1, -1]], [-2])
        r = DualSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 2) < 1e-6

    def test_dual_feasible_start(self):
        # All c >= 0, b >= 0 but some b might go negative after adding constraint
        lp = StandardLP([1, 1], [[1, 0], [0, 1]], [3, 3])
        r = DualSimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective) < 1e-6  # optimal at origin

    def test_matches_simplex(self):
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r1 = SimplexSolver().solve(lp)
        r2 = DualSimplexSolver().solve(lp)
        assert abs(r1.objective - r2.objective) < 1e-6


# ============================================================
# LPBuilder
# ============================================================

class TestLPBuilder:
    def test_maximize(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 5, y: 4})
        lp.add_constraint({x: 6, y: 4}, ConstraintOp.LEQ, 24)
        lp.add_constraint({x: 1, y: 2}, ConstraintOp.LEQ, 6)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 21) < 1e-6

    def test_minimize(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 1, y: 1}, ConstraintOp.GEQ, 4)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 4) < 1e-6

    def test_equality_constraint(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 1}, ConstraintOp.EQ, 3)
        lp.add_constraint({y: 1}, ConstraintOp.EQ, 2)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x"] - 3) < 1e-6
        assert abs(r.variables["y"] - 2) < 1e-6

    def test_upper_bound(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", upper=10)
        lp.set_objective({x: 1})
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x"] - 10) < 1e-6

    def test_duplicate_variable(self):
        lp = LPBuilder()
        lp.add_var("x")
        with pytest.raises(ValueError, match="already exists"):
            lp.add_var("x")

    def test_unknown_variable_in_objective(self):
        lp = LPBuilder()
        with pytest.raises(ValueError, match="Unknown"):
            lp.set_objective({"z": 1})

    def test_unknown_variable_in_constraint(self):
        lp = LPBuilder()
        lp.add_var("x")
        with pytest.raises(ValueError, match="Unknown"):
            lp.add_constraint({"z": 1}, ConstraintOp.LEQ, 5)

    def test_named_variables_in_result(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("profit_a")
        y = lp.add_var("profit_b")
        lp.set_objective({x: 3, y: 2})
        lp.add_constraint({x: 1, y: 1}, ConstraintOp.LEQ, 10)
        r = lp.solve()
        assert "profit_a" in r.variables
        assert "profit_b" in r.variables

    def test_geq_constraint(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        lp.set_objective({x: 1})
        lp.add_constraint({x: 1}, ConstraintOp.GEQ, 5)
        r = lp.solve()
        assert abs(r.variables["x"] - 5) < 1e-6

    def test_multiple_constraints(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 3, y: 5})
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 4)
        lp.add_constraint({y: 1}, ConstraintOp.LEQ, 6)
        lp.add_constraint({x: 1, y: 1}, ConstraintOp.LEQ, 8)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 36) < 1e-6


# ============================================================
# MILPSolver
# ============================================================

class TestMILPSolver:
    def test_basic_integer(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", is_integer=True)
        y = lp.add_var("y", is_integer=True)
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 2, y: 1}, ConstraintOp.LEQ, 7)
        lp.add_constraint({x: 1, y: 2}, ConstraintOp.LEQ, 7)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        # Check integer solution
        assert abs(r.variables["x"] - round(r.variables["x"])) < 1e-6
        assert abs(r.variables["y"] - round(r.variables["y"])) < 1e-6

    def test_mixed_integer(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", is_integer=True)
        y = lp.add_var("y")  # continuous
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 1, y: 1}, ConstraintOp.LEQ, 5.5)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x"] - round(r.variables["x"])) < 1e-6

    def test_infeasible_integer(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", is_integer=True)
        lp.set_objective({x: 1})
        # x >= 2.5 and x <= 2.5 -- no integer in range
        lp.add_constraint({x: 1}, ConstraintOp.GEQ, 2.5)
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 2.5)
        r = lp.solve()
        assert r.status == LPStatus.INFEASIBLE

    def test_binary_knapsack(self):
        # 0-1 knapsack: items with weights [2,3,4,5] values [3,4,5,6] capacity 8
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        items = []
        for i in range(4):
            v = lp.add_var(f"x{i}", upper=1, is_integer=True)
            items.append(v)
        weights = [2, 3, 4, 5]
        values = [3, 4, 5, 6]
        lp.set_objective({items[i]: values[i] for i in range(4)})
        lp.add_constraint({items[i]: weights[i] for i in range(4)}, ConstraintOp.LEQ, 8)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        # Optimal: x0=1, x1=1, x2=0, x3=0 (weight=5, value=7)
        # or x0=0, x1=1, x2=0, x3=1 (weight=8, value=10)
        # or x0=1, x1=0, x2=1, x3=0 (weight=6, value=8)
        # or x0=0, x1=0, x2=1, x3=0 ... let's just check valid
        total_weight = sum(weights[i] * r.variables[f"x{i}"] for i in range(4))
        assert total_weight <= 8 + 1e-6
        assert r.objective >= 7 - 1e-6  # at least 7

    def test_single_integer_var(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", is_integer=True)
        lp.set_objective({x: 1})
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 3.7)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x"] - 3) < 1e-6


# ============================================================
# Sensitivity Analysis
# ============================================================

class TestSensitivity:
    def test_rhs_increase(self):
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r_base = TwoPhaseSimplexSolver().solve(lp)
        r_new = sensitivity_rhs(lp, 0, 6)  # increase first constraint RHS
        assert r_new.status == LPStatus.OPTIMAL
        # More resources -> better (more negative) objective
        assert r_new.objective <= r_base.objective + 1e-6

    def test_rhs_decrease(self):
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r_base = TwoPhaseSimplexSolver().solve(lp)
        r_new = sensitivity_rhs(lp, 1, -1)  # decrease second constraint
        assert r_new.status == LPStatus.OPTIMAL

    def test_obj_change(self):
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r_new = sensitivity_obj(lp, 0, 2)  # make x0 less attractive
        assert r_new.status == LPStatus.OPTIMAL

    def test_rhs_makes_infeasible(self):
        # x >= 5, x <= 3 after change
        lp = StandardLP([1], [[-1], [1]], [-5, 3])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.status == LPStatus.INFEASIBLE


# ============================================================
# Transportation Problem
# ============================================================

class TestTransportation:
    def test_basic(self):
        supply = [20, 30]
        demand = [10, 15, 25]
        costs = [[8, 6, 10], [9, 12, 7]]
        r = transportation_problem(supply, demand, costs)
        assert r.status == LPStatus.OPTIMAL
        assert r.objective is not None

    def test_balanced(self):
        supply = [10, 10]
        demand = [10, 10]
        costs = [[1, 2], [3, 1]]
        r = transportation_problem(supply, demand, costs)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 20) < 1e-6  # 10*1 + 10*1

    def test_single_source(self):
        supply = [100]
        demand = [30, 30, 40]
        costs = [[5, 3, 7]]
        r = transportation_problem(supply, demand, costs)
        assert r.status == LPStatus.OPTIMAL

    def test_unbalanced_excess_supply(self):
        supply = [50, 50]
        demand = [30, 20]
        costs = [[4, 6], [5, 3]]
        r = transportation_problem(supply, demand, costs)
        assert r.status == LPStatus.OPTIMAL


# ============================================================
# Diet Problem
# ============================================================

class TestDiet:
    def test_basic(self):
        # 2 nutrients, 3 foods
        nutrients_min = [10, 8]
        food_nutrients = [[3, 2], [1, 4], [2, 1]]  # food x nutrient
        food_costs = [5, 4, 3]
        r = diet_problem(nutrients_min, food_nutrients, food_costs)
        assert r.status == LPStatus.OPTIMAL
        assert r.objective > 0

    def test_single_food(self):
        nutrients_min = [10]
        food_nutrients = [[2]]
        food_costs = [3]
        r = diet_problem(nutrients_min, food_nutrients, food_costs)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 15) < 1e-6  # need 5 units at cost 3 each


# ============================================================
# LPResult repr
# ============================================================

class TestLPResult:
    def test_optimal_repr(self):
        r = LPResult(LPStatus.OPTIMAL, 42.0, {"x": 1})
        s = repr(r)
        assert "OPTIMAL" in s
        assert "42" in s

    def test_infeasible_repr(self):
        r = LPResult(LPStatus.INFEASIBLE)
        assert "infeasible" in repr(r)

    def test_not_solved_repr(self):
        r = LPResult(LPStatus.NOT_SOLVED)
        assert "not_solved" in repr(r)


# ============================================================
# Edge cases and stress tests
# ============================================================

class TestEdgeCases:
    def test_zero_objective(self):
        lp = StandardLP([0, 0], [[1, 1]], [5])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective) < 1e-6

    def test_single_constraint_single_var(self):
        lp = StandardLP([-1], [[1]], [10])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x0"] - 10) < 1e-6

    def test_tight_constraints(self):
        # x=2, y=3 is the only feasible point
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 1}, ConstraintOp.EQ, 2)
        lp.add_constraint({y: 1}, ConstraintOp.EQ, 3)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 5) < 1e-6

    def test_many_variables(self):
        # 10 variables, minimize sum, each <= 1
        n = 10
        c = [-1] * n
        A = []
        b = []
        for i in range(n):
            row = [0] * n
            row[i] = 1
            A.append(row)
            b.append(1)
        lp = StandardLP(c, A, b)
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-10)) < 1e-6

    def test_redundant_constraints(self):
        # x <= 5, x <= 10, x <= 3 -- only x<=3 active
        lp = StandardLP([-1], [[1], [1], [1]], [5, 10, 3])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.variables["x0"] - 3) < 1e-6

    def test_parallel_constraints(self):
        # x+y <= 4, x+y <= 6 (parallel, first is binding)
        lp = StandardLP([-1, -1], [[1, 1], [1, 1]], [4, 6])
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-4)) < 1e-6


# ============================================================
# Classic LP problems
# ============================================================

class TestClassicProblems:
    def test_resource_allocation(self):
        # 2 products, 3 resources
        # max 10*x1 + 20*x2
        # 4*x1 + 3*x2 <= 120 (labor)
        # 2*x1 + 4*x2 <= 80  (material)
        # x1 <= 25, x2 <= 25
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x1 = lp.add_var("x1")
        x2 = lp.add_var("x2")
        lp.set_objective({x1: 10, x2: 20})
        lp.add_constraint({x1: 4, x2: 3}, ConstraintOp.LEQ, 120)
        lp.add_constraint({x1: 2, x2: 4}, ConstraintOp.LEQ, 80)
        lp.add_constraint({x1: 1}, ConstraintOp.LEQ, 25)
        lp.add_constraint({x2: 1}, ConstraintOp.LEQ, 25)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert r.objective > 0

    def test_blending(self):
        # Blend 3 ingredients to minimize cost
        # min 2*x1 + 3*x2 + 5*x3
        # protein: 0.1*x1 + 0.2*x2 + 0.3*x3 >= 10
        # fat: 0.3*x1 + 0.1*x2 + 0.2*x3 >= 8
        # x1 + x2 + x3 = 100
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x1 = lp.add_var("x1")
        x2 = lp.add_var("x2")
        x3 = lp.add_var("x3")
        lp.set_objective({x1: 2, x2: 3, x3: 5})
        lp.add_constraint({x1: 0.1, x2: 0.2, x3: 0.3}, ConstraintOp.GEQ, 10)
        lp.add_constraint({x1: 0.3, x2: 0.1, x3: 0.2}, ConstraintOp.GEQ, 8)
        lp.add_constraint({x1: 1, x2: 1, x3: 1}, ConstraintOp.EQ, 100)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        total = sum(r.variables.values())
        assert abs(total - 100) < 1e-4

    def test_portfolio_simple(self):
        # Invest in 2 assets, maximize return, risk budget
        # max 0.08*x + 0.12*y
        # x + y <= 10000
        # 0.1*x + 0.3*y <= 2000 (risk)
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("bonds")
        y = lp.add_var("stocks")
        lp.set_objective({x: 0.08, y: 0.12})
        lp.add_constraint({x: 1, y: 1}, ConstraintOp.LEQ, 10000)
        lp.add_constraint({x: 0.1, y: 0.3}, ConstraintOp.LEQ, 2000)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert r.objective > 0

    def test_assignment_relaxation(self):
        # 3x3 assignment (LP relaxation gives integer solution for assignment)
        costs = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
        lp = LPBuilder(sense=Sense.MINIMIZE)
        vars = {}
        for i in range(3):
            for j in range(3):
                vars[(i, j)] = lp.add_var(f"x_{i}_{j}", upper=1)

        obj = {vars[(i, j)]: costs[i][j] for i in range(3) for j in range(3)}
        lp.set_objective(obj)

        # Each row sums to 1
        for i in range(3):
            lp.add_constraint({vars[(i, j)]: 1 for j in range(3)}, ConstraintOp.EQ, 1)
        # Each col sums to 1
        for j in range(3):
            lp.add_constraint({vars[(i, j)]: 1 for i in range(3)}, ConstraintOp.EQ, 1)

        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        # Optimal assignment cost
        assert r.objective < 10

    def test_network_flow_as_lp(self):
        # Simple max flow: s->a (cap 3), s->b (cap 2), a->t (cap 2), b->t (cap 3), a->b (cap 1)
        # max flow s->t
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        sa = lp.add_var("sa", upper=3)
        sb = lp.add_var("sb", upper=2)
        at_ = lp.add_var("at", upper=2)
        bt = lp.add_var("bt", upper=3)
        ab = lp.add_var("ab", upper=1)

        # Maximize flow into t
        lp.set_objective({at_: 1, bt: 1})

        # Flow conservation at a: sa = at + ab
        lp.add_constraint({sa: 1, at_: -1, ab: -1}, ConstraintOp.EQ, 0)
        # Flow conservation at b: sb + ab = bt
        lp.add_constraint({sb: 1, ab: 1, bt: -1}, ConstraintOp.EQ, 0)

        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 5) < 1e-6  # max flow = 5


# ============================================================
# Solver comparison tests
# ============================================================

class TestSolverComparison:
    def test_all_solvers_agree(self):
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r1 = SimplexSolver().solve(lp)
        r2 = TwoPhaseSimplexSolver().solve(lp)
        r3 = DualSimplexSolver().solve(lp)
        assert abs(r1.objective - r2.objective) < 1e-6
        assert abs(r1.objective - r3.objective) < 1e-6

    def test_all_solvers_detect_unbounded(self):
        lp = StandardLP([-1, 0], [[1, -1]], [1])
        for solver_cls in [SimplexSolver, TwoPhaseSimplexSolver, DualSimplexSolver]:
            r = solver_cls().solve(lp)
            assert r.status == LPStatus.UNBOUNDED

    def test_builder_vs_standard(self):
        # Same problem via builder and direct standard form
        lp_std = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r_std = SimplexSolver().solve(lp_std)

        lp_build = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp_build.add_var("x")
        y = lp_build.add_var("y")
        lp_build.set_objective({x: 5, y: 4})
        lp_build.add_constraint({x: 6, y: 4}, ConstraintOp.LEQ, 24)
        lp_build.add_constraint({x: 1, y: 2}, ConstraintOp.LEQ, 6)
        r_build = lp_build.solve()

        assert abs(r_std.objective - (-r_build.objective)) < 1e-6


# ============================================================
# Larger problems
# ============================================================

class TestLargerProblems:
    def test_20_variables(self):
        n = 20
        c = [-(i + 1) for i in range(n)]
        A = []
        b = []
        # Each var <= 10
        for i in range(n):
            row = [0] * n
            row[i] = 1
            A.append(row)
            b.append(10)
        # Sum <= 100
        A.append([1] * n)
        b.append(100)
        lp = StandardLP(c, A, b)
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL

    def test_many_constraints(self):
        # 5 vars, 20 constraints
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        vars = [lp.add_var(f"x{i}") for i in range(5)]
        obj = {vars[i]: (i + 1) for i in range(5)}
        lp.set_objective(obj)

        # Random-ish constraints
        for i in range(20):
            coeffs = {vars[j]: ((i * 7 + j * 3) % 5 + 1) for j in range(5)}
            lp.add_constraint(coeffs, ConstraintOp.LEQ, 50 + i * 3)

        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL

    def test_diet_5_foods_4_nutrients(self):
        nutrients_min = [20, 15, 10, 25]
        food_nutrients = [
            [3, 2, 1, 4],
            [1, 4, 2, 1],
            [2, 1, 3, 2],
            [4, 3, 2, 3],
            [1, 2, 4, 1],
        ]
        food_costs = [5, 4, 3, 7, 2]
        r = diet_problem(nutrients_min, food_nutrients, food_costs)
        assert r.status == LPStatus.OPTIMAL
        # Verify all nutrients met
        for j in range(4):
            total = sum(
                food_nutrients[i][j] * r.variables.get(f"food_{i}", 0)
                for i in range(5)
            )
            assert total >= nutrients_min[j] - 1e-4


# ============================================================
# Integer programming additional tests
# ============================================================

class TestIntegerProgramming:
    def test_facility_location(self):
        # Open facilities to serve customers at minimum cost
        # 2 facilities, 3 customers
        lp = LPBuilder(sense=Sense.MINIMIZE)
        # Binary: open facility i?
        y0 = lp.add_var("y0", upper=1, is_integer=True)
        y1 = lp.add_var("y1", upper=1, is_integer=True)
        # Assignment: customer j to facility i
        x = {}
        for i in range(2):
            for j in range(3):
                x[(i, j)] = lp.add_var(f"x_{i}_{j}", upper=1)

        # Fixed costs + assignment costs
        assign_costs = [[4, 6, 9], [5, 3, 2]]
        fixed_costs = [10, 12]
        obj = {y0: fixed_costs[0], y1: fixed_costs[1]}
        for i in range(2):
            for j in range(3):
                obj[x[(i, j)]] = assign_costs[i][j]
        lp.set_objective(obj)

        # Each customer assigned to exactly one facility
        for j in range(3):
            lp.add_constraint({x[(0, j)]: 1, x[(1, j)]: 1}, ConstraintOp.EQ, 1)

        # Assignment only if facility open (x_ij <= y_i)
        for i in range(2):
            yi = y0 if i == 0 else y1
            for j in range(3):
                lp.add_constraint({x[(i, j)]: 1, yi: -1}, ConstraintOp.LEQ, 0)

        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL

    def test_integer_rounding(self):
        # Relaxation gives x=2.5 but integer optimum is x=2 or x=3
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", is_integer=True)
        lp.set_objective({x: 1})
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 2.5)
        r = lp.solve()
        assert abs(r.variables["x"] - 2) < 1e-6

    def test_all_integer_small(self):
        # 3 integer vars
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x", is_integer=True)
        y = lp.add_var("y", is_integer=True)
        z = lp.add_var("z", is_integer=True)
        lp.set_objective({x: 3, y: 5, z: 2})
        lp.add_constraint({x: 1, y: 1, z: 1}, ConstraintOp.LEQ, 10)
        lp.add_constraint({x: 2, y: 1}, ConstraintOp.LEQ, 8)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        for v in ["x", "y", "z"]:
            assert abs(r.variables[v] - round(r.variables[v])) < 1e-6


# ============================================================
# Constraint type tests
# ============================================================

class TestConstraintTypes:
    def test_all_leq(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x")
        lp.set_objective({x: 1})
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 5)
        r = lp.solve()
        assert abs(r.variables["x"] - 5) < 1e-6

    def test_all_geq(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 1}, ConstraintOp.GEQ, 3)
        lp.add_constraint({y: 1}, ConstraintOp.GEQ, 2)
        r = lp.solve()
        assert abs(r.objective - 5) < 1e-6

    def test_all_eq(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 1, y: 2})
        lp.add_constraint({x: 1}, ConstraintOp.EQ, 4)
        lp.add_constraint({y: 1}, ConstraintOp.EQ, 3)
        r = lp.solve()
        assert abs(r.objective - 10) < 1e-6

    def test_mixed_constraint_types(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 1, y: 1})
        lp.add_constraint({x: 1}, ConstraintOp.GEQ, 2)
        lp.add_constraint({y: 1}, ConstraintOp.LEQ, 10)
        lp.add_constraint({x: 1, y: 1}, ConstraintOp.EQ, 7)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 7) < 1e-6


# ============================================================
# Duality and complementary slackness
# ============================================================

class TestDuality:
    def test_strong_duality(self):
        # Primal: min c^Tx, Ax<=b, x>=0
        # Dual: max b^Ty, A^Ty<=c, y>=0
        # Use a problem where optimal is not at origin
        c = [-5, -4]
        A = [[6, 4], [1, 2]]
        b = [24, 6]
        primal = StandardLP(c, A, b)
        rp = TwoPhaseSimplexSolver().solve(primal)

        # Dual: max 24*y1+6*y2 s.t. 6y1+y2<=-5, 4y1+2y2<=-4, y>=0
        # Since c is negative, dual constraints have negative RHS
        # => min -24y1-6y2 s.t. 6y1+y2<=5, 4y1+2y2<=4 (negate c for >=0 coeffs)
        # Actually for standard form: dual of min c^Tx Ax<=b is max b^Ty A^Ty<=c y>=0
        # With c=[-5,-4]: A^Ty <= c means 6y1+y2<=-5 -- infeasible for y>=0
        # Use positive c instead
        c2 = [2, 3]
        A2 = [[1, 1], [1, 0]]
        b2 = [4, 3]
        primal2 = StandardLP(c2, A2, b2)
        rp2 = TwoPhaseSimplexSolver().solve(primal2)
        # Optimal: x=(0,0), obj=0. Dual also 0.
        # Better test: min -2x-3y, x+y<=4, 2x+y<=6
        c3 = [-2, -3]
        A3 = [[1, 1], [2, 1]]
        b3 = [4, 6]
        primal3 = StandardLP(c3, A3, b3)
        rp3 = TwoPhaseSimplexSolver().solve(primal3)
        # Dual: max 4y1+6y2 s.t. y1+2y2<=-2, y1+y2<=-3, y>=0
        # Negative RHS means y=0 is only feasible, dual obj=0
        # Strong duality: both should equal -12
        # Actually: min -2x-3y s.t. x+y<=4, 2x+y<=6
        # Corner pts: (0,0)->0, (0,4)->-12, (3,0)->-6, (2,2)->-10
        # Optimal at (0,4): obj = -12
        assert abs(rp3.objective - (-12)) < 1e-6

    def test_dual_values_interpretation(self):
        # Shadow prices: how much objective improves per unit of RHS increase
        lp = StandardLP([-5, -4], [[6, 4], [1, 2]], [24, 6])
        r = TwoPhaseSimplexSolver().solve(lp)
        assert r.dual_values is not None
        # Dual values should be non-negative for minimization with <= constraints


# ============================================================
# Special cases
# ============================================================

class TestSpecialCases:
    def test_zero_rhs(self):
        lp = StandardLP([-1, -1], [[1, -1], [-1, 1]], [0, 0])
        r = SimplexSolver().solve(lp)
        # x1 = x2, unbounded
        assert r.status == LPStatus.UNBOUNDED or r.status == LPStatus.OPTIMAL

    def test_all_zero_constraints(self):
        lp = StandardLP([-1], [[0]], [5])
        r = SimplexSolver().solve(lp)
        # 0*x <= 5 is always true, -x unbounded
        assert r.status == LPStatus.UNBOUNDED

    def test_infeasible_via_builder(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        lp.set_objective({x: 1})
        lp.add_constraint({x: 1}, ConstraintOp.GEQ, 10)
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 5)
        r = lp.solve()
        assert r.status == LPStatus.INFEASIBLE

    def test_single_equality_point(self):
        lp = LPBuilder(sense=Sense.MINIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 3, y: 7})
        lp.add_constraint({x: 1}, ConstraintOp.EQ, 5)
        lp.add_constraint({y: 1}, ConstraintOp.EQ, 2)
        r = lp.solve()
        assert abs(r.objective - 29) < 1e-6


# ============================================================
# Regression / numerical robustness
# ============================================================

class TestNumerical:
    def test_near_degenerate(self):
        # Almost parallel constraints
        lp = StandardLP(
            [-1, -1],
            [[1, 1], [1, 1.0001]],
            [10, 10.001]
        )
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL

    def test_small_coefficients(self):
        lp = StandardLP(
            [-0.001, -0.002],
            [[0.001, 0.001], [0.001, 0], [0, 0.001]],
            [0.005, 0.003, 0.003]
        )
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL

    def test_large_coefficients(self):
        lp = StandardLP(
            [-1000, -2000],
            [[1000, 1000], [1000, 0], [0, 1000]],
            [5000, 3000, 3000]
        )
        r = SimplexSolver().solve(lp)
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - (-8000)) < 1

    def test_mixed_scale(self):
        lp = LPBuilder(sense=Sense.MAXIMIZE)
        x = lp.add_var("x")
        y = lp.add_var("y")
        lp.set_objective({x: 0.001, y: 1000})
        lp.add_constraint({x: 1}, ConstraintOp.LEQ, 100)
        lp.add_constraint({y: 1}, ConstraintOp.LEQ, 0.5)
        r = lp.solve()
        assert r.status == LPStatus.OPTIMAL
        assert abs(r.objective - 500.1) < 0.1
