"""
Tests for V185: Octagon-Guided CEGAR
"""
import sys, os, pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V002_pdr_ic3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V010_predicate_abstraction_cegar'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V173_octagon_abstract_domain'))

from octagon_guided_cegar import (
    octagon_pre_analyze, octagon_predicates_from_source,
    octagon_predicates_from_constraints, octagon_guided_cegar,
    verify_loop_with_octagon_cegar, verify_ts_with_octagon_cegar,
    compare_cegar_approaches, compare_loop,
    classify_predicates, predicate_strength_analysis,
    octagon_invariant_candidates, verify_octagon_invariant,
    build_ts_with_octagon_hints,
    _ast_to_oct_program, _convert_stmt, _convert_expr, _convert_cond,
    _oct_constraint_to_smt, _oct_constraint_name,
    OctCEGARResult, OctCEGARStats, ComparisonResult, CEGARVerdict
)
from octagon import Octagon, OctConstraint, OctExpr
from pred_abs_cegar import (
    Predicate, ConcreteTS, CEGARResult, CEGARStats,
    verify_with_cegar, extract_loop_ts, _parse_property,
    auto_predicates_from_ts, _and, _or, _not, _eq
)
from smt_solver import Var, IntConst, BoolConst, App, Op, Sort, SortKind
from pdr import TransitionSystem

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ===========================================================================
# AST Conversion Tests
# ===========================================================================

class TestASTConversion:
    """Test C10 AST -> octagon tuple AST conversion."""

    def test_simple_assignment(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 5;")).parse()
        oct = _ast_to_oct_program(ast)
        assert oct == ('assign', 'x', ('const', 5))

    def test_sequence(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; let y = 1;")).parse()
        oct = _ast_to_oct_program(ast)
        assert oct[0] == 'seq'
        assert ('assign', 'x', ('const', 0)) in oct
        assert ('assign', 'y', ('const', 1)) in oct

    def test_while_loop(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; while (x < 10) { x = x + 1; }")).parse()
        oct = _ast_to_oct_program(ast)
        # Should contain assign and while
        assert oct[0] == 'seq'

    def test_if_stmt(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; if (x < 5) { x = 1; }")).parse()
        oct = _ast_to_oct_program(ast)
        assert oct[0] == 'seq'

    def test_convert_expr_intlit(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 42;")).parse()
        expr = ast.stmts[0].value
        result = _convert_expr(expr)
        assert result == ('const', 42)

    def test_convert_expr_var(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; x = x;")).parse()
        expr = ast.stmts[1].value
        result = _convert_expr(expr)
        assert result == ('var', 'x')

    def test_convert_expr_binop_add(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; x = x + 1;")).parse()
        expr = ast.stmts[1].value
        result = _convert_expr(expr)
        assert result[0] == 'add'

    def test_convert_expr_binop_sub(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; x = x - 1;")).parse()
        expr = ast.stmts[1].value
        result = _convert_expr(expr)
        assert result[0] == 'sub'

    def test_convert_cond_lt(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("while (x < 10) { x = x + 1; }")).parse()
        cond = ast.stmts[0].cond
        result = _convert_cond(cond)
        assert result[0] == 'lt'

    def test_convert_cond_ge(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("while (x >= 0) { x = x - 1; }")).parse()
        cond = ast.stmts[0].cond
        result = _convert_cond(cond)
        assert result[0] == 'ge'

    def test_convert_cond_eq(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("while (x == 5) { x = x + 1; }")).parse()
        cond = ast.stmts[0].cond
        result = _convert_cond(cond)
        assert result[0] == 'eq'

    def test_empty_program(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("")).parse()
        oct = _ast_to_oct_program(ast)
        assert oct == ('skip',)

    def test_negation_expr(self):
        from stack_vm import lex, Parser
        ast = Parser(lex("let x = 0; x = -x;")).parse()
        expr = ast.stmts[1].value
        result = _convert_expr(expr)
        assert result[0] == 'neg'


# ===========================================================================
# Octagon Constraint -> SMT Conversion Tests
# ===========================================================================

class TestConstraintConversion:
    """Test octagon constraint to SMT predicate conversion."""

    def test_var_upper_bound(self):
        c = OctConstraint.var_le('x', 10)
        cts = ConcreteTS(int_vars=['x'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None
        assert smt.op == Op.LE

    def test_var_lower_bound(self):
        c = OctConstraint.var_ge('x', 5)
        cts = ConcreteTS(int_vars=['x'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None
        assert smt.op == Op.GE

    def test_difference_bound(self):
        c = OctConstraint.diff_le('x', 'y', 3)
        cts = ConcreteTS(int_vars=['x', 'y'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None
        assert smt.op == Op.LE
        # Inner should be SUB
        assert smt.args[0].op == Op.SUB

    def test_reverse_difference_bound(self):
        c = OctConstraint.diff_ge('x', 'y', -2)
        cts = ConcreteTS(int_vars=['x', 'y'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None

    def test_sum_upper_bound(self):
        c = OctConstraint.sum_le('x', 'y', 20)
        cts = ConcreteTS(int_vars=['x', 'y'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None
        assert smt.op == Op.LE
        assert smt.args[0].op == Op.ADD

    def test_sum_lower_bound(self):
        c = OctConstraint.sum_ge('x', 'y', 5)
        cts = ConcreteTS(int_vars=['x', 'y'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None
        assert smt.op == Op.GE

    def test_constraint_name_upper(self):
        c = OctConstraint.var_le('x', 10)
        name = _oct_constraint_name(c)
        assert 'x' in name
        assert 'le' in name
        assert '10' in name

    def test_constraint_name_lower(self):
        c = OctConstraint.var_ge('x', 5)
        name = _oct_constraint_name(c)
        assert 'x' in name
        assert 'ge' in name

    def test_constraint_name_diff(self):
        c = OctConstraint.diff_le('x', 'y', 3)
        name = _oct_constraint_name(c)
        assert 'minus' in name

    def test_constraint_name_sum(self):
        c = OctConstraint.sum_le('x', 'y', 20)
        name = _oct_constraint_name(c)
        assert 'plus' in name

    def test_equality_constraints(self):
        constraints = OctConstraint.eq('x', 'y')
        cts = ConcreteTS(int_vars=['x', 'y'])
        # eq returns 2 constraints (diff_le and diff_ge with bound 0)
        assert len(constraints) == 2
        for c in constraints:
            smt = _oct_constraint_to_smt(c, cts)
            assert smt is not None

    def test_var_eq_constraints(self):
        constraints = OctConstraint.var_eq('x', 5)
        cts = ConcreteTS(int_vars=['x'])
        assert len(constraints) == 2
        for c in constraints:
            smt = _oct_constraint_to_smt(c, cts)
            assert smt is not None


# ===========================================================================
# Octagon Pre-Analysis Tests
# ===========================================================================

class TestOctagonPreAnalysis:
    """Test octagon pre-analysis of C10 source code."""

    def test_simple_assignment_analysis(self):
        source = "let x = 5;"
        result, constraints = octagon_pre_analyze(source)
        assert result.final_state is not None
        assert not result.final_state.is_bot()

    def test_two_var_analysis(self):
        source = "let x = 0; let y = 10;"
        result, constraints = octagon_pre_analyze(source)
        assert not result.final_state.is_bot()
        assert len(constraints) > 0

    def test_loop_analysis(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        result, constraints = octagon_pre_analyze(source)
        assert result.final_state is not None

    def test_two_var_loop(self):
        source = """
        let x = 0;
        let y = 10;
        while (x < y) {
            x = x + 1;
            y = y - 1;
        }
        """
        result, constraints = octagon_pre_analyze(source)
        assert result.final_state is not None
        # Should discover relational constraints between x and y

    def test_parallel_increment(self):
        source = """
        let x = 0;
        let y = 0;
        while (x < 5) {
            x = x + 1;
            y = y + 1;
        }
        """
        result, constraints = octagon_pre_analyze(source)
        assert result.final_state is not None

    def test_predicate_generation(self):
        source = "let x = 0; let y = 10;"
        cts = ConcreteTS(int_vars=['x', 'y'])
        preds = octagon_predicates_from_source(source, cts)
        # Should generate predicates for x and y bounds
        assert len(preds) > 0
        for p in preds:
            assert isinstance(p, Predicate)
            assert p.formula is not None

    def test_predicate_filters_unknown_vars(self):
        source = "let x = 0; let y = 10; let z = 20;"
        # cts only knows about x and y
        cts = ConcreteTS(int_vars=['x', 'y'])
        preds = octagon_predicates_from_source(source, cts)
        # Should not include predicates about z
        for p in preds:
            assert 'z' not in p.name

    def test_predicate_dedup(self):
        constraints = [
            OctConstraint.var_le('x', 10),
            OctConstraint.var_le('x', 10),  # duplicate
        ]
        cts = ConcreteTS(int_vars=['x'])
        preds = octagon_predicates_from_constraints(constraints, cts)
        assert len(preds) == 1

    def test_predicate_filters_huge_bounds(self):
        from fractions import Fraction
        constraints = [
            OctConstraint.var_le('x', Fraction(10**15)),
        ]
        cts = ConcreteTS(int_vars=['x'])
        preds = octagon_predicates_from_constraints(constraints, cts)
        assert len(preds) == 0  # Should be filtered as too large


# ===========================================================================
# Octagon-Guided CEGAR Verification Tests
# ===========================================================================

class TestOctagonGuidedCEGAR:
    """Test the full octagon-guided CEGAR verification pipeline."""

    def test_simple_counter_safe(self):
        """Counter x=0, x'=x+1, property x>=0. Should be SAFE."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_simple_counter_unsafe(self):
        """Init violates property: x=-1, prop x>=0. Immediately UNSAFE."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(-1))
        cts.trans_formula = _eq(Var("x'", INT), Var('x', INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.UNSAFE

    def test_two_var_safe(self):
        """x=0, y=10, x'=x+1, y'=y-1, property x+y==10. Should be SAFE."""
        cts = ConcreteTS(int_vars=['x', 'y'])
        x, y = Var('x', INT), Var('y', INT)
        xp, yp = Var("x'", INT), Var("y'", INT)

        cts.init_formula = _and(
            _eq(x, IntConst(0)),
            _eq(y, IntConst(10))
        )
        cts.trans_formula = _and(
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
            _eq(yp, App(Op.SUB, [y, IntConst(1)], INT))
        )
        cts.prop_formula = _eq(App(Op.ADD, [x, y], INT), IntConst(10))

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_with_source_octagon(self):
        """Provide source for octagon pre-analysis."""
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        cts = extract_loop_ts(source)
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts, source=source)
        assert result.verdict == CEGARVerdict.SAFE

    def test_octagon_predicates_in_result(self):
        """Check that octagon predicates are tracked in stats."""
        source = "let x = 0; let y = 10;"
        cts = ConcreteTS(int_vars=['x', 'y'])
        cts.init_formula = _and(
            _eq(Var('x', INT), IntConst(0)),
            _eq(Var('y', INT), IntConst(10))
        )
        cts.trans_formula = _and(
            _eq(Var("x'", INT), Var('x', INT)),
            _eq(Var("y'", INT), Var('y', INT))
        )
        cts.prop_formula = App(Op.GE, [Var('y', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts, source=source)
        assert result.stats.oct_constraints_found >= 0

    def test_without_oct_refinement(self):
        """Test with octagon refinement disabled."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts, use_oct_refinement=False)
        assert result.verdict == CEGARVerdict.SAFE

    def test_identity_system(self):
        """x=0, x'=x (no change), property x==0."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), Var('x', INT))
        cts.prop_formula = _eq(Var('x', INT), IntConst(0))

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_bounded_system(self):
        """x=0, x'=min(x+1,10), property x<=10."""
        cts = ConcreteTS(int_vars=['x'])
        x = Var('x', INT)
        xp = Var("x'", INT)

        # ITE: if x < 10 then x+1 else x
        from pred_abs_cegar import _ite
        cts.init_formula = _eq(x, IntConst(0))
        cts.trans_formula = _eq(xp, _ite(
            App(Op.LT, [x, IntConst(10)], BOOL),
            App(Op.ADD, [x, IntConst(1)], INT),
            x
        ))
        cts.prop_formula = App(Op.LE, [x, IntConst(10)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE


# ===========================================================================
# Source-Level Verification Tests
# ===========================================================================

class TestSourceLevelVerification:
    """Test verify_loop_with_octagon_cegar on C10 source."""

    def test_simple_counter_ge_0(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        result = verify_loop_with_octagon_cegar(source, "x >= 0")
        assert result.verdict == CEGARVerdict.SAFE

    def test_counter_bounded(self):
        source = """
        let x = 0;
        while (x < 5) {
            x = x + 1;
        }
        """
        result = verify_loop_with_octagon_cegar(source, "x >= 0")
        assert result.verdict == CEGARVerdict.SAFE

    def test_two_var_sum_invariant(self):
        """x+y should remain 10 when x++ and y-- together."""
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        result = verify_loop_with_octagon_cegar(source, "x + y == 10")
        assert result.verdict == CEGARVerdict.SAFE

    def test_y_nonneg(self):
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        result = verify_loop_with_octagon_cegar(source, "y >= 0")
        assert result.verdict == CEGARVerdict.SAFE

    def test_x_bounded_above(self):
        source = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        result = verify_loop_with_octagon_cegar(source, "x <= 100")
        assert result.verdict == CEGARVerdict.SAFE

    def test_verify_ts_wrapper(self):
        """Test verify_ts_with_octagon_cegar convenience wrapper."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = verify_ts_with_octagon_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE


# ===========================================================================
# Comparison Framework Tests
# ===========================================================================

class TestComparisonFramework:
    """Test standard vs octagon-guided CEGAR comparison."""

    def test_compare_simple_counter(self):
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        comp = compare_cegar_approaches(cts)
        assert comp.both_agree
        assert comp.standard.verdict == CEGARVerdict.SAFE
        assert comp.octagon_guided.verdict == CEGARVerdict.SAFE

    def test_compare_with_source(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        cts = extract_loop_ts(source)
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        comp = compare_cegar_approaches(cts, source=source)
        assert comp.both_agree
        assert comp.speedup_iterations > 0

    def test_compare_loop_convenience(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        comp = compare_loop(source, "x >= 0")
        assert comp.both_agree

    def test_comparison_result_fields(self):
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), Var('x', INT))
        cts.prop_formula = _eq(Var('x', INT), IntConst(0))

        comp = compare_cegar_approaches(cts)
        assert isinstance(comp.standard, CEGARResult)
        assert isinstance(comp.octagon_guided, OctCEGARResult)
        assert isinstance(comp.speedup_iterations, float)
        assert isinstance(comp.extra_predicates, int)


# ===========================================================================
# Predicate Analysis Tests
# ===========================================================================

class TestPredicateAnalysis:
    """Test predicate classification and analysis."""

    def test_classify_octagon_predicates(self):
        preds = [
            Predicate("oct_x_le_10", App(Op.LE, [Var('x', INT), IntConst(10)], BOOL)),
            Predicate("oct_x_minus_y_le_5", App(Op.LE, [App(Op.SUB, [Var('x', INT), Var('y', INT)], INT), IntConst(5)], BOOL)),
            Predicate("oct_x_plus_y_le_20", App(Op.LE, [App(Op.ADD, [Var('x', INT), Var('y', INT)], INT), IntConst(20)], BOOL)),
            Predicate("property", App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)),
        ]
        classified = classify_predicates(preds)
        assert len(classified['unary']) == 1
        assert len(classified['difference']) == 1
        assert len(classified['sum']) == 1
        assert len(classified['other']) == 1

    def test_classify_empty(self):
        classified = classify_predicates([])
        assert all(len(v) == 0 for v in classified.values())

    def test_strength_analysis(self):
        cts = ConcreteTS(int_vars=['x'])
        preds = [
            Predicate("x_ge0", App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)),
        ]
        results = predicate_strength_analysis(preds, cts)
        assert len(results) == 1
        pred, is_taut, is_contra = results[0]
        assert pred.name == "x_ge0"
        # x >= 0 is neither tautology nor contradiction
        assert not is_taut
        assert not is_contra

    def test_invariant_candidates(self):
        source = "let x = 0; let y = 10;"
        cts = ConcreteTS(int_vars=['x', 'y'])
        candidates = octagon_invariant_candidates(source, cts)
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, (App, Var, type(IntConst(0))))


# ===========================================================================
# Octagon Invariant Quick-Check Tests
# ===========================================================================

class TestOctagonInvariantCheck:
    """Test verify_octagon_invariant for quick octagon-only proofs."""

    def test_simple_bounds(self):
        source = "let x = 5;"
        proved, result, constraints = verify_octagon_invariant(source, "x >= 0")
        # Octagon should know x=5, so x>=0 should be provable
        assert proved is True

    def test_two_var_bounds(self):
        source = "let x = 0; let y = 10;"
        proved, result, constraints = verify_octagon_invariant(source, "y >= 0")
        assert proved is True

    def test_loop_bounds(self):
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        # Octagon with widening may overshoot, so this may or may not prove
        proved, result, constraints = verify_octagon_invariant(source, "x >= 0")
        # At minimum, result and constraints should be valid
        assert result.final_state is not None
        assert isinstance(constraints, list)

    def test_unprovable_property(self):
        source = "let x = 5;"
        proved, result, constraints = verify_octagon_invariant(source, "x >= 10")
        assert proved is False


# ===========================================================================
# Build TS With Hints Tests
# ===========================================================================

class TestBuildTSWithHints:
    """Test building transition systems with octagon constraint hints."""

    def test_basic_build(self):
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x'],
            init_map={'x': 0},
            trans_map={'x': App(Op.ADD, [Var('x', INT), IntConst(1)], INT)},
            prop_str=App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)
        )
        assert 'x' in cts.int_vars
        assert cts.init_formula is not None
        assert cts.trans_formula is not None
        assert cts.prop_formula is not None

    def test_build_with_hints(self):
        hints = [
            OctConstraint.var_le('x', 100),
            OctConstraint.var_ge('x', 0),
        ]
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x'],
            init_map={'x': 0},
            trans_map={'x': App(Op.ADD, [Var('x', INT), IntConst(1)], INT)},
            prop_str=App(Op.GE, [Var('x', INT), IntConst(0)], BOOL),
            oct_hints=hints
        )
        assert len(preds) == 2

    def test_build_two_vars(self):
        x, y = Var('x', INT), Var('y', INT)
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x', 'y'],
            init_map={'x': 0, 'y': 10},
            trans_map={
                'x': App(Op.ADD, [x, IntConst(1)], INT),
                'y': App(Op.SUB, [y, IntConst(1)], INT)
            },
            prop_str=App(Op.GE, [y, IntConst(0)], BOOL)
        )
        assert len(cts.int_vars) == 2

    def test_build_with_relational_hints(self):
        hints = [
            OctConstraint.diff_le('x', 'y', 0),  # x - y <= 0
            OctConstraint.sum_le('x', 'y', 10),   # x + y <= 10
        ]
        x, y = Var('x', INT), Var('y', INT)
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x', 'y'],
            init_map={'x': 0, 'y': 10},
            trans_map={
                'x': App(Op.ADD, [x, IntConst(1)], INT),
                'y': App(Op.SUB, [y, IntConst(1)], INT)
            },
            prop_str=App(Op.GE, [y, IntConst(0)], BOOL),
            oct_hints=hints
        )
        assert len(preds) == 2
        # Should have difference and sum predicates
        names = [p.name for p in preds]
        assert any('minus' in n for n in names)
        assert any('plus' in n for n in names)

    def test_build_frame_vars(self):
        """Variables not in trans_map should get frame condition (x' = x)."""
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x', 'y'],
            init_map={'x': 0, 'y': 5},
            trans_map={'x': App(Op.ADD, [Var('x', INT), IntConst(1)], INT)},
            prop_str=App(Op.GE, [Var('y', INT), IntConst(0)], BOOL)
        )
        # y should be unchanged in transition
        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_verify_with_hints(self):
        """Build TS with hints and verify."""
        hints = [OctConstraint.var_ge('x', 0)]
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x'],
            init_map={'x': 0},
            trans_map={'x': App(Op.ADD, [Var('x', INT), IntConst(1)], INT)},
            prop_str=App(Op.GE, [Var('x', INT), IntConst(0)], BOOL),
            oct_hints=hints
        )
        result = octagon_guided_cegar(cts, initial_predicates=preds)
        assert result.verdict == CEGARVerdict.SAFE


# ===========================================================================
# Stats & Result Structure Tests
# ===========================================================================

class TestStatsAndResults:
    """Test result data structures and stats tracking."""

    def test_oct_cegar_stats_defaults(self):
        s = OctCEGARStats()
        assert s.oct_constraints_found == 0
        assert s.oct_predicates_generated == 0
        assert s.oct_refinements == 0

    def test_oct_cegar_result_fields(self):
        r = OctCEGARResult(verdict=CEGARVerdict.SAFE)
        assert r.verdict == CEGARVerdict.SAFE
        assert r.invariant is None
        assert r.counterexample is None
        assert r.predicates == []
        assert r.oct_predicates == []

    def test_comparison_result_defaults(self):
        c = ComparisonResult()
        assert c.standard is None
        assert c.octagon_guided is None
        assert c.speedup_iterations == 0.0
        assert c.both_agree is True

    def test_stats_populated_after_run(self):
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.stats.cegar_stats.iterations >= 1
        assert result.stats.cegar_stats.pdr_calls >= 1

    def test_unsafe_has_counterexample(self):
        """Init violates property: x=-5, prop x>=0. Has counterexample."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(-5))
        cts.trans_formula = _eq(Var("x'", INT), Var('x', INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.UNSAFE
        assert result.counterexample is not None


# ===========================================================================
# Edge Cases & Robustness Tests
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_single_var_no_loop(self):
        source = "let x = 42;"
        result = verify_loop_with_octagon_cegar(source, "x >= 0")
        # No loop, so transition is trivial identity
        assert result.verdict in (CEGARVerdict.SAFE, CEGARVerdict.UNKNOWN)

    def test_empty_octagon_constraints(self):
        """When octagon produces no useful constraints."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        # No source => no octagon analysis
        result = octagon_guided_cegar(cts, source=None)
        assert result.verdict == CEGARVerdict.SAFE

    def test_max_iterations_reached(self):
        """Test that max_iterations is respected."""
        cts = ConcreteTS(int_vars=['x'])
        cts.init_formula = _eq(Var('x', INT), IntConst(0))
        cts.trans_formula = _eq(Var("x'", INT), App(Op.ADD, [Var('x', INT), IntConst(1)], INT))
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts, max_iterations=1)
        # With 1 iteration and good auto-predicates, may still prove SAFE
        assert result.verdict in (CEGARVerdict.SAFE, CEGARVerdict.UNKNOWN)

    def test_predicate_name_sanitization(self):
        """Octagon constraint names should be valid identifiers."""
        c = OctConstraint.var_le('x', -5)
        name = _oct_constraint_name(c)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_negative_bound_constraint(self):
        c = OctConstraint.var_ge('x', -10)
        cts = ConcreteTS(int_vars=['x'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None

    def test_zero_bound_constraint(self):
        c = OctConstraint.var_le('x', 0)
        cts = ConcreteTS(int_vars=['x'])
        smt = _oct_constraint_to_smt(c, cts)
        assert smt is not None
        assert smt.op == Op.LE

    def test_large_system_three_vars(self):
        """Three-variable system."""
        cts = ConcreteTS(int_vars=['x', 'y', 'z'])
        x, y, z = Var('x', INT), Var('y', INT), Var('z', INT)
        xp, yp, zp = Var("x'", INT), Var("y'", INT), Var("z'", INT)

        cts.init_formula = _and(
            _eq(x, IntConst(0)),
            _eq(y, IntConst(0)),
            _eq(z, IntConst(0))
        )
        cts.trans_formula = _and(
            _eq(xp, App(Op.ADD, [x, IntConst(1)], INT)),
            _eq(yp, App(Op.ADD, [y, IntConst(1)], INT)),
            _eq(zp, App(Op.ADD, [z, IntConst(1)], INT))
        )
        cts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_conditional_transition(self):
        """System with conditional transition."""
        from pred_abs_cegar import _ite
        cts = ConcreteTS(int_vars=['x'])
        x = Var('x', INT)
        xp = Var("x'", INT)

        cts.init_formula = _eq(x, IntConst(0))
        # if x < 5 then x+1 else 5
        cts.trans_formula = _eq(xp, _ite(
            App(Op.LT, [x, IntConst(5)], BOOL),
            App(Op.ADD, [x, IntConst(1)], INT),
            IntConst(5)
        ))
        cts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)

        result = octagon_guided_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    """Integration tests composing multiple features."""

    def test_full_pipeline_with_source(self):
        """Full pipeline: source -> octagon -> predicates -> CEGAR -> verdict."""
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        # Verify multiple properties
        r1 = verify_loop_with_octagon_cegar(source, "x >= 0")
        assert r1.verdict == CEGARVerdict.SAFE

        r2 = verify_loop_with_octagon_cegar(source, "x + y == 10")
        assert r2.verdict == CEGARVerdict.SAFE

    def test_octagon_then_compare(self):
        """Run octagon analysis, then compare standard vs guided."""
        source = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        # First get octagon info
        result, constraints = octagon_pre_analyze(source)
        assert result.final_state is not None

        # Then compare approaches
        comp = compare_loop(source, "x >= 0")
        assert comp.both_agree

    def test_hints_to_verification(self):
        """Build TS with hints, generate predicates, verify."""
        hints = [
            OctConstraint.var_ge('x', 0),
            OctConstraint.var_le('x', 100),
        ]
        cts, preds = build_ts_with_octagon_hints(
            int_vars=['x'],
            init_map={'x': 0},
            trans_map={'x': App(Op.ADD, [Var('x', INT), IntConst(1)], INT)},
            prop_str=App(Op.GE, [Var('x', INT), IntConst(0)], BOOL),
            oct_hints=hints
        )
        result = octagon_guided_cegar(cts, initial_predicates=preds)
        assert result.verdict == CEGARVerdict.SAFE

    def test_classify_after_verification(self):
        """Classify predicates after verification."""
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        cts = extract_loop_ts(source)
        cts.prop_formula = App(Op.GE, [Var('x', INT), IntConst(0)], BOOL)

        result = octagon_guided_cegar(cts, source=source)
        classified = classify_predicates(result.predicates)
        # Should have at least some classified predicates
        total = sum(len(v) for v in classified.values())
        assert total == len(result.predicates)

    def test_invariant_candidates_useful(self):
        """Octagon invariant candidates should be valid SMT formulas."""
        source = """
        let x = 0;
        let y = 10;
        while (x < 10) {
            x = x + 1;
            y = y - 1;
        }
        """
        cts = extract_loop_ts(source)
        candidates = octagon_invariant_candidates(source, cts)
        # All candidates should be SMT terms
        for c in candidates:
            assert c is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
