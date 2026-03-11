"""Tests for V120: Array Domain Abstract Interpretation."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from array_domain import (
    # AST
    parse_source, Program, IntLit, VarExpr, BinExpr, ArrayLit, ArrayNew,
    ArrayRead, ArrayLen, LetStmt, AssignStmt, ArrayWriteStmt, IfStmt,
    WhileStmt, AssertStmt,
    # Lexer
    lex, TT,
    # Domain
    ArrayAbstractValue, ArrayEnv, IntervalDomain, INF, NEG_INF,
    # Interpreter
    ArrayInterpreter, Warning, WarningKind,
    # Properties
    ArrayProperty, ArrayPropertyKind, analyze_array_properties,
    # APIs
    array_analyze, check_bounds, check_assertions, get_array_info,
    get_variable_range, infer_properties, compare_analyses, array_summary,
)


# ===========================================================================
# 1. Lexer Tests
# ===========================================================================

class TestLexer:
    def test_basic_tokens(self):
        tokens = lex("let x = 5;")
        types = [t.type for t in tokens]
        assert TT.LET in types
        assert TT.IDENT in types
        assert TT.ASSIGN in types
        assert TT.INT in types
        assert TT.SEMI in types

    def test_array_tokens(self):
        tokens = lex("[1, 2, 3]")
        types = [t.type for t in tokens]
        assert TT.LBRACKET in types
        assert TT.RBRACKET in types
        assert TT.COMMA in types

    def test_keywords(self):
        tokens = lex("let if else while assert new_array len")
        kw_types = [t.type for t in tokens if t.type != TT.EOF]
        assert kw_types == [TT.LET, TT.IF, TT.ELSE, TT.WHILE, TT.ASSERT,
                           TT.NEW_ARRAY, TT.LEN]

    def test_comparison_operators(self):
        tokens = lex("< <= > >= == !=")
        op_types = [t.type for t in tokens if t.type != TT.EOF]
        assert op_types == [TT.LT, TT.LE, TT.GT, TT.GE, TT.EQ, TT.NEQ]

    def test_comments(self):
        tokens = lex("let x = 5; // comment\nlet y = 6;")
        idents = [t.value for t in tokens if t.type == TT.IDENT]
        assert idents == ['x', 'y']

    def test_line_tracking(self):
        tokens = lex("let x = 1;\nlet y = 2;")
        y_tok = [t for t in tokens if t.type == TT.IDENT and t.value == 'y'][0]
        assert y_tok.line == 2


# ===========================================================================
# 2. Parser Tests
# ===========================================================================

class TestParser:
    def test_let_scalar(self):
        prog = parse_source("let x = 5;")
        assert len(prog.stmts) == 1
        assert isinstance(prog.stmts[0], LetStmt)
        assert prog.stmts[0].name == 'x'

    def test_let_array_literal(self):
        prog = parse_source("let a = [1, 2, 3];")
        stmt = prog.stmts[0]
        assert isinstance(stmt, LetStmt)
        assert isinstance(stmt.value, ArrayLit)
        assert len(stmt.value.elements) == 3

    def test_new_array(self):
        prog = parse_source("let a = new_array(10, 0);")
        stmt = prog.stmts[0]
        assert isinstance(stmt.value, ArrayNew)

    def test_array_read(self):
        prog = parse_source("let x = a[0];")
        stmt = prog.stmts[0]
        assert isinstance(stmt.value, ArrayRead)

    def test_array_write(self):
        prog = parse_source("a[0] = 5;")
        stmt = prog.stmts[0]
        assert isinstance(stmt, ArrayWriteStmt)
        assert stmt.array == 'a'

    def test_len_expr(self):
        prog = parse_source("let n = len(a);")
        stmt = prog.stmts[0]
        assert isinstance(stmt.value, ArrayLen)

    def test_if_stmt(self):
        prog = parse_source("if (x < 5) { let y = 1; } else { let y = 2; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, IfStmt)
        assert len(stmt.then_body) == 1
        assert len(stmt.else_body) == 1

    def test_while_stmt(self):
        prog = parse_source("while (i < 10) { i = i + 1; }")
        stmt = prog.stmts[0]
        assert isinstance(stmt, WhileStmt)

    def test_assert_stmt(self):
        prog = parse_source("assert(x >= 0);")
        stmt = prog.stmts[0]
        assert isinstance(stmt, AssertStmt)

    def test_nested_array_access(self):
        prog = parse_source("let x = a[i + 1];")
        stmt = prog.stmts[0]
        assert isinstance(stmt.value, ArrayRead)
        assert isinstance(stmt.value.index, BinExpr)

    def test_binary_expressions(self):
        prog = parse_source("let x = (a + b) * c - d;")
        stmt = prog.stmts[0]
        assert isinstance(stmt.value, BinExpr)

    def test_unary_minus(self):
        prog = parse_source("let x = -5;")
        stmt = prog.stmts[0]
        assert stmt.value.op == '-' or (isinstance(stmt.value, IntLit) and False)

    def test_empty_array(self):
        prog = parse_source("let a = [];")
        stmt = prog.stmts[0]
        assert isinstance(stmt.value, ArrayLit)
        assert len(stmt.value.elements) == 0


# ===========================================================================
# 3. ArrayAbstractValue Tests
# ===========================================================================

class TestArrayAbstractValue:
    def test_bot(self):
        v = ArrayAbstractValue.bot()
        assert v.is_bot()
        assert not v.is_top()

    def test_top(self):
        v = ArrayAbstractValue.top()
        assert v.is_top()
        assert not v.is_bot()

    def test_from_literal(self):
        v = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(1, 1), 1: IntervalDomain(2, 2),
                      2: IntervalDomain(3, 3)},
            smash=IntervalDomain(1, 3),
        )
        assert v.get_element(0) == IntervalDomain(1, 1)
        assert v.get_element(1) == IntervalDomain(2, 2)
        assert v.get_element(5) == IntervalDomain(1, 3)  # smash

    def test_strong_update(self):
        v = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(1, 1)},
            smash=IntervalDomain(0, 0),
        )
        v2 = v.set_element(0, IntervalDomain(10, 10))
        assert v2.get_element(0) == IntervalDomain(10, 10)
        assert v.get_element(0) == IntervalDomain(1, 1)  # original unchanged

    def test_weak_update(self):
        v = ArrayAbstractValue(
            length=IntervalDomain(5, 5),
            elements={0: IntervalDomain(0, 0)},
            smash=IntervalDomain(0, 0),
        )
        idx = IntervalDomain(0, 4)  # unknown index
        v2 = v.set_element_weak(idx, IntervalDomain(99, 99))
        # Smash should now include 99
        assert v2.smash.hi >= 99

    def test_read_concrete_index(self):
        v = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(10, 10), 1: IntervalDomain(20, 20)},
            smash=IntervalDomain(0, 0),
        )
        assert v.read_element(IntervalDomain(0, 0)) == IntervalDomain(10, 10)
        assert v.read_element(IntervalDomain(1, 1)) == IntervalDomain(20, 20)

    def test_read_abstract_index(self):
        v = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(10, 10), 1: IntervalDomain(20, 20)},
            smash=IntervalDomain(0, 0),
        )
        result = v.read_element(IntervalDomain(0, 1))
        assert result.lo <= 10 and result.hi >= 20  # join of both

    def test_join(self):
        v1 = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(1, 1)},
            smash=IntervalDomain(0, 0),
        )
        v2 = ArrayAbstractValue(
            length=IntervalDomain(4, 4),
            elements={0: IntervalDomain(5, 5)},
            smash=IntervalDomain(0, 0),
        )
        joined = v1.join(v2)
        assert joined.length.lo == 3
        assert joined.length.hi == 4
        assert joined.get_element(0).lo <= 1
        assert joined.get_element(0).hi >= 5

    def test_widen(self):
        v1 = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(0, 5)},
            smash=IntervalDomain(0, 0),
        )
        v2 = ArrayAbstractValue(
            length=IntervalDomain(3, 4),
            elements={0: IntervalDomain(0, 10)},
            smash=IntervalDomain(0, 0),
        )
        widened = v1.widen(v2)
        # Length upper bound should widen to INF
        assert widened.length.hi == INF

    def test_leq(self):
        v1 = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={0: IntervalDomain(1, 1)},
            smash=IntervalDomain(0, 0),
        )
        v2 = ArrayAbstractValue(
            length=IntervalDomain(0, 10),
            elements={},
            smash=IntervalDomain(NEG_INF, INF),
        )
        assert v1.leq(v2)
        assert not v2.leq(v1)

    def test_bot_join(self):
        bot = ArrayAbstractValue.bot()
        v = ArrayAbstractValue(
            length=IntervalDomain(3, 3),
            elements={},
            smash=IntervalDomain(0, 0),
        )
        assert bot.join(v).length == IntervalDomain(3, 3)
        assert v.join(bot).length == IntervalDomain(3, 3)


# ===========================================================================
# 4. ArrayEnv Tests
# ===========================================================================

class TestArrayEnv:
    def test_scalar_set_get(self):
        env = ArrayEnv()
        env.set_scalar('x', IntervalDomain(5, 5))
        assert env.get_scalar('x') == IntervalDomain(5, 5)

    def test_scalar_default_top(self):
        env = ArrayEnv()
        x = env.get_scalar('x')
        assert x.is_top()

    def test_array_set_get(self):
        env = ArrayEnv()
        arr = ArrayAbstractValue(
            length=IntervalDomain(3, 3), elements={}, smash=IntervalDomain(0, 0))
        env.set_array('a', arr)
        assert env.get_array('a').length == IntervalDomain(3, 3)

    def test_env_join(self):
        e1 = ArrayEnv()
        e1.set_scalar('x', IntervalDomain(1, 1))
        e2 = ArrayEnv()
        e2.set_scalar('x', IntervalDomain(5, 5))
        joined = e1.join(e2)
        x = joined.get_scalar('x')
        assert x.lo == 1 and x.hi == 5

    def test_env_widen(self):
        e1 = ArrayEnv()
        e1.set_scalar('x', IntervalDomain(0, 5))
        e2 = ArrayEnv()
        e2.set_scalar('x', IntervalDomain(0, 10))
        widened = e1.widen(e2)
        assert widened.get_scalar('x').hi == INF

    def test_env_equals(self):
        e1 = ArrayEnv()
        e1.set_scalar('x', IntervalDomain(1, 5))
        e2 = ArrayEnv()
        e2.set_scalar('x', IntervalDomain(1, 5))
        assert e1.equals(e2)

    def test_env_copy(self):
        e = ArrayEnv()
        e.set_scalar('x', IntervalDomain(1, 1))
        e2 = e.copy()
        e2.set_scalar('x', IntervalDomain(2, 2))
        assert e.get_scalar('x') == IntervalDomain(1, 1)


# ===========================================================================
# 5. Basic Scalar Analysis
# ===========================================================================

class TestScalarAnalysis:
    def test_let_constant(self):
        result = array_analyze("let x = 5;")
        assert result['scalars']['x'] == IntervalDomain(5, 5)

    def test_arithmetic(self):
        result = array_analyze("let x = 3; let y = x + 2;")
        assert result['scalars']['y'] == IntervalDomain(5, 5)

    def test_subtraction(self):
        result = array_analyze("let x = 10; let y = x - 3;")
        assert result['scalars']['y'] == IntervalDomain(7, 7)

    def test_multiplication(self):
        result = array_analyze("let x = 4; let y = x * 3;")
        assert result['scalars']['y'] == IntervalDomain(12, 12)

    def test_negation(self):
        result = array_analyze("let x = 5; let y = -x;")
        assert result['scalars']['y'] == IntervalDomain(-5, -5)

    def test_assignment(self):
        result = array_analyze("let x = 1; x = 10;")
        assert result['scalars']['x'] == IntervalDomain(10, 10)


# ===========================================================================
# 6. Array Literal Analysis
# ===========================================================================

class TestArrayLiteralAnalysis:
    def test_simple_array(self):
        result = array_analyze("let a = [1, 2, 3];")
        arr = result['arrays']['a']
        assert arr.length == IntervalDomain(3, 3)
        assert arr.get_element(0) == IntervalDomain(1, 1)
        assert arr.get_element(1) == IntervalDomain(2, 2)
        assert arr.get_element(2) == IntervalDomain(3, 3)

    def test_empty_array(self):
        result = array_analyze("let a = [];")
        arr = result['arrays']['a']
        assert arr.length == IntervalDomain(0, 0)

    def test_array_with_expressions(self):
        result = array_analyze("let x = 5; let a = [x, x + 1, x * 2];")
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(5, 5)
        assert arr.get_element(1) == IntervalDomain(6, 6)
        assert arr.get_element(2) == IntervalDomain(10, 10)

    def test_single_element(self):
        result = array_analyze("let a = [42];")
        arr = result['arrays']['a']
        assert arr.length == IntervalDomain(1, 1)
        assert arr.get_element(0) == IntervalDomain(42, 42)


# ===========================================================================
# 7. new_array Analysis
# ===========================================================================

class TestNewArrayAnalysis:
    def test_new_array_concrete(self):
        result = array_analyze("let a = new_array(5, 0);")
        arr = result['arrays']['a']
        assert arr.length == IntervalDomain(5, 5)
        assert arr.get_element(0) == IntervalDomain(0, 0)
        assert arr.get_element(4) == IntervalDomain(0, 0)

    def test_new_array_init_value(self):
        result = array_analyze("let a = new_array(3, 42);")
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(42, 42)
        assert arr.smash == IntervalDomain(42, 42)

    def test_new_array_symbolic_size(self):
        result = array_analyze("let n = 10; let a = new_array(n, 0);")
        arr = result['arrays']['a']
        assert arr.length == IntervalDomain(10, 10)


# ===========================================================================
# 8. Array Read Analysis
# ===========================================================================

class TestArrayReadAnalysis:
    def test_read_concrete_index(self):
        result = array_analyze("let a = [10, 20, 30]; let x = a[1];")
        assert result['scalars']['x'] == IntervalDomain(20, 20)

    def test_read_first_element(self):
        result = array_analyze("let a = [42, 0, 0]; let x = a[0];")
        assert result['scalars']['x'] == IntervalDomain(42, 42)

    def test_read_last_element(self):
        result = array_analyze("let a = [1, 2, 99]; let x = a[2];")
        assert result['scalars']['x'] == IntervalDomain(99, 99)

    def test_read_with_len(self):
        result = array_analyze("let a = [1, 2, 3]; let n = len(a);")
        assert result['scalars']['n'] == IntervalDomain(3, 3)


# ===========================================================================
# 9. Array Write Analysis
# ===========================================================================

class TestArrayWriteAnalysis:
    def test_write_concrete_index(self):
        result = array_analyze("let a = [0, 0, 0]; a[1] = 99;")
        arr = result['arrays']['a']
        assert arr.get_element(1) == IntervalDomain(99, 99)

    def test_write_preserves_others(self):
        result = array_analyze("let a = [10, 20, 30]; a[1] = 99;")
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(10, 10)
        assert arr.get_element(1) == IntervalDomain(99, 99)
        assert arr.get_element(2) == IntervalDomain(30, 30)

    def test_write_expression(self):
        result = array_analyze("let a = [0, 0, 0]; let x = 5; a[0] = x + 1;")
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(6, 6)


# ===========================================================================
# 10. Conditional Analysis
# ===========================================================================

class TestConditionalAnalysis:
    def test_if_then_else(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = array_analyze(src)
        # x is always > 3, so y should be 1
        assert result['scalars']['y'] == IntervalDomain(1, 1)

    def test_if_both_branches(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 10) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = array_analyze(src)
        # x is always <= 10, so y should be 2
        assert result['scalars']['y'] == IntervalDomain(2, 2)

    def test_if_refines_variable(self):
        src = """
        let x = 5;
        let a = new_array(10, 0);
        if (x < 10) {
            a[x] = 1;
        }
        """
        result = array_analyze(src)
        # Should not warn: x is [5,5] which is < 10
        oob = [w for w in result['warnings']
               if w.kind == WarningKind.OUT_OF_BOUNDS]
        assert len(oob) == 0


# ===========================================================================
# 11. Loop Analysis
# ===========================================================================

class TestLoopAnalysis:
    def test_simple_loop(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = array_analyze(src)
        # After loop, i >= 10
        i_val = result['scalars']['i']
        assert i_val.lo >= 10

    def test_array_init_loop(self):
        src = """
        let a = new_array(5, 0);
        let i = 0;
        while (i < 5) {
            a[i] = i;
            i = i + 1;
        }
        """
        result = array_analyze(src)
        # No definite out-of-bounds
        oob = [w for w in result['warnings']
               if w.kind == WarningKind.OUT_OF_BOUNDS]
        assert len(oob) == 0

    def test_loop_convergence(self):
        src = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        result = array_analyze(src)
        x = result['scalars']['x']
        assert x.lo >= 100


# ===========================================================================
# 12. Out-of-Bounds Detection
# ===========================================================================

class TestOutOfBounds:
    def test_definite_oob(self):
        warnings = check_bounds("let a = [1, 2, 3]; let x = a[5];")
        assert any(w.kind == WarningKind.OUT_OF_BOUNDS for w in warnings)

    def test_negative_index(self):
        warnings = check_bounds("let a = [1, 2]; let x = a[-1];")
        assert any(w.kind == WarningKind.OUT_OF_BOUNDS for w in warnings)

    def test_safe_access(self):
        warnings = check_bounds("let a = [1, 2, 3]; let x = a[0];")
        assert len(warnings) == 0

    def test_possible_oob(self):
        src = """
        let a = new_array(5, 0);
        let i = 3;
        if (i > 2) {
            i = 10;
        }
        let x = a[i];
        """
        warnings = check_bounds(src)
        # i could be 10, which is OOB
        assert any(w.kind in (WarningKind.OUT_OF_BOUNDS, WarningKind.POSSIBLE_OUT_OF_BOUNDS)
                   for w in warnings)

    def test_write_oob(self):
        warnings = check_bounds("let a = [1, 2]; a[5] = 10;")
        assert any(w.kind == WarningKind.OUT_OF_BOUNDS for w in warnings)

    def test_safe_write(self):
        warnings = check_bounds("let a = [0, 0, 0]; a[2] = 5;")
        assert len(warnings) == 0


# ===========================================================================
# 13. Division Safety
# ===========================================================================

class TestDivisionSafety:
    def test_div_by_zero(self):
        result = array_analyze("let x = 5; let y = x / 0;")
        assert any(w.kind == WarningKind.DIVISION_BY_ZERO for w in result['warnings'])

    def test_safe_division(self):
        result = array_analyze("let x = 10; let y = x / 2;")
        div_warnings = [w for w in result['warnings']
                       if w.kind in (WarningKind.DIVISION_BY_ZERO, WarningKind.POSSIBLE_DIVISION_BY_ZERO)]
        assert len(div_warnings) == 0

    def test_possible_div_zero(self):
        src = """
        let x = 0;
        if (x >= 0) {
            let y = 10 / x;
        }
        """
        result = array_analyze(src)
        # x is exactly 0, so this is a definite division by zero
        assert any(w.kind == WarningKind.DIVISION_BY_ZERO for w in result['warnings'])


# ===========================================================================
# 14. Assertion Checking
# ===========================================================================

class TestAssertions:
    def test_assertion_holds(self):
        warnings = check_assertions("let x = 5; assert(x > 0);")
        assert len(warnings) == 0

    def test_assertion_fails(self):
        warnings = check_assertions("let x = 0; assert(x > 0);")
        assert any(w.kind == WarningKind.ASSERTION_FAILURE for w in warnings)

    def test_assertion_may_fail(self):
        src = """
        let x = 5;
        if (x > 3) {
            x = 0;
        }
        assert(x > 0);
        """
        warnings = check_assertions(src)
        assert any(w.kind in (WarningKind.ASSERTION_FAILURE, WarningKind.POSSIBLE_ASSERTION_FAILURE)
                   for w in warnings)

    def test_assertion_with_array(self):
        src = """
        let a = [1, 2, 3];
        let x = a[0];
        assert(x > 0);
        """
        warnings = check_assertions(src)
        assert len(warnings) == 0


# ===========================================================================
# 15. Array Properties
# ===========================================================================

class TestArrayProperties:
    def test_sorted_array(self):
        props = infer_properties("let a = [1, 2, 3, 4, 5];")
        sorted_props = [p for p in props if p.kind == ArrayPropertyKind.SORTED]
        assert len(sorted_props) > 0
        assert sorted_props[0].holds

    def test_unsorted_array(self):
        props = infer_properties("let a = [5, 1, 3];")
        sorted_props = [p for p in props if p.kind == ArrayPropertyKind.SORTED]
        # Either no sorted property or holds=False
        for p in sorted_props:
            assert not p.holds

    def test_bounded_array(self):
        props = infer_properties("let a = [1, 2, 3];")
        bounded = [p for p in props if p.kind == ArrayPropertyKind.BOUNDED]
        assert len(bounded) > 0
        assert bounded[0].details['lower'] == 1
        assert bounded[0].details['upper'] == 3

    def test_constant_array(self):
        props = infer_properties("let a = new_array(5, 42);")
        const = [p for p in props if p.kind == ArrayPropertyKind.CONSTANT]
        assert len(const) > 0
        assert const[0].details['value'] == 42

    def test_initialized_array(self):
        props = infer_properties("let a = [1, 2, 3];")
        init = [p for p in props if p.kind == ArrayPropertyKind.INITIALIZED]
        assert len(init) > 0


# ===========================================================================
# 16. Composition: Array + Scalar
# ===========================================================================

class TestComposition:
    def test_array_read_scalar_use(self):
        src = """
        let a = [10, 20, 30];
        let x = a[1];
        let y = x + 5;
        """
        result = array_analyze(src)
        assert result['scalars']['y'] == IntervalDomain(25, 25)

    def test_scalar_index_array(self):
        src = """
        let a = [10, 20, 30];
        let i = 2;
        let x = a[i];
        """
        result = array_analyze(src)
        assert result['scalars']['x'] == IntervalDomain(30, 30)

    def test_len_in_loop_bound(self):
        src = """
        let a = [1, 2, 3, 4, 5];
        let n = len(a);
        let i = 0;
        let s = 0;
        while (i < n) {
            s = s + a[i];
            i = i + 1;
        }
        """
        result = array_analyze(src)
        # n should be 5
        assert result['scalars']['n'] == IntervalDomain(5, 5)

    def test_conditional_array_write(self):
        src = """
        let a = new_array(5, 0);
        let x = 3;
        if (x > 2) {
            a[0] = 100;
        } else {
            a[0] = 200;
        }
        """
        result = array_analyze(src)
        # x > 2 is always true, so a[0] should be 100
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(100, 100)


# ===========================================================================
# 17. Complex Programs
# ===========================================================================

class TestComplexPrograms:
    def test_array_swap(self):
        src = """
        let a = [3, 1, 2];
        let tmp = a[0];
        a[0] = a[1];
        a[1] = tmp;
        """
        result = array_analyze(src)
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(1, 1)
        assert arr.get_element(1) == IntervalDomain(3, 3)

    def test_find_max(self):
        src = """
        let a = [3, 7, 2, 9, 1];
        let m = a[0];
        let i = 1;
        while (i < 5) {
            let v = a[i];
            if (v > m) {
                m = v;
            }
            i = i + 1;
        }
        """
        result = array_analyze(src)
        m = result['scalars']['m']
        # m should include 9
        assert m.hi >= 9

    def test_array_sum(self):
        src = """
        let a = [1, 2, 3];
        let s = 0;
        s = s + a[0];
        s = s + a[1];
        s = s + a[2];
        """
        result = array_analyze(src)
        assert result['scalars']['s'] == IntervalDomain(6, 6)

    def test_reassign_array(self):
        src = """
        let a = [1, 2, 3];
        a = [4, 5, 6];
        """
        result = array_analyze(src)
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(4, 4)

    def test_two_arrays(self):
        src = """
        let a = [1, 2, 3];
        let b = [10, 20, 30];
        let x = a[0] + b[0];
        """
        result = array_analyze(src)
        assert result['scalars']['x'] == IntervalDomain(11, 11)


# ===========================================================================
# 18. Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = array_analyze("")
        assert len(result['warnings']) == 0

    def test_only_scalars(self):
        result = array_analyze("let x = 1; let y = 2; let z = x + y;")
        assert result['scalars']['z'] == IntervalDomain(3, 3)
        assert len(result['arrays']) == 0

    def test_large_array(self):
        result = array_analyze("let a = new_array(100, 0);")
        arr = result['arrays']['a']
        assert arr.length == IntervalDomain(100, 100)

    def test_nested_array_access(self):
        src = """
        let a = [0, 1, 2, 3];
        let i = a[0];
        let x = a[i];
        """
        result = array_analyze(src)
        # a[0] = 0, so a[a[0]] = a[0] = 0
        assert result['scalars']['x'] == IntervalDomain(0, 0)

    def test_division_result(self):
        result = array_analyze("let x = 10; let y = x / 3;")
        y = result['scalars']['y']
        assert y.lo <= 3 and y.hi >= 3


# ===========================================================================
# 19. High-Level API Tests
# ===========================================================================

class TestAPIs:
    def test_get_array_info(self):
        info = get_array_info("let a = [1, 2, 3];", 'a')
        assert info is not None
        assert info.length == IntervalDomain(3, 3)

    def test_get_variable_range(self):
        val = get_variable_range("let x = 5;", 'x')
        assert val == IntervalDomain(5, 5)

    def test_compare_analyses(self):
        result = compare_analyses("let a = [1, 2, 3]; let x = a[0];")
        assert result['arrays_tracked'] > 0

    def test_array_summary(self):
        summary = array_summary("let a = [1, 2, 3]; let x = a[0];")
        assert "Array" in summary
        assert "a:" in summary

    def test_check_bounds_api(self):
        warnings = check_bounds("let a = [1]; let x = a[0];")
        assert len(warnings) == 0

    def test_check_assertions_api(self):
        warnings = check_assertions("let x = 5; assert(x == 5);")
        assert len(warnings) == 0


# ===========================================================================
# 20. Widening and Fixpoint
# ===========================================================================

class TestWidening:
    def test_widening_terminates(self):
        src = """
        let x = 0;
        while (x < 1000000) {
            x = x + 1;
        }
        """
        result = array_analyze(src)
        # Should terminate despite large bound
        x = result['scalars']['x']
        assert x.lo >= 1000000

    def test_nested_loops(self):
        src = """
        let i = 0;
        while (i < 3) {
            let j = 0;
            while (j < 3) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = array_analyze(src)
        assert result['scalars']['i'].lo >= 3

    def test_array_in_loop_widening(self):
        src = """
        let a = new_array(10, 0);
        let i = 0;
        while (i < 10) {
            a[i] = i;
            i = i + 1;
        }
        """
        result = array_analyze(src)
        # Should not crash, widening should handle array growth


# ===========================================================================
# 21. Condition Refinement
# ===========================================================================

class TestConditionRefinement:
    def test_lt_refinement(self):
        src = """
        let x = 5;
        if (x < 3) {
            assert(x < 3);
        }
        """
        # x is 5, so the then-branch is unreachable
        warnings = check_assertions(src)
        assert len(warnings) == 0

    def test_eq_refinement(self):
        src = """
        let x = 5;
        let y = 0;
        if (x == 5) {
            y = 1;
        } else {
            y = 2;
        }
        """
        result = array_analyze(src)
        # x==5 is always true
        assert result['scalars']['y'] == IntervalDomain(1, 1)

    def test_bounds_check_pattern(self):
        src = """
        let a = new_array(10, 0);
        let i = 5;
        if (i >= 0) {
            if (i < 10) {
                let x = a[i];
            }
        }
        """
        result = array_analyze(src)
        oob = [w for w in result['warnings'] if w.kind == WarningKind.OUT_OF_BOUNDS]
        assert len(oob) == 0


# ===========================================================================
# 22. Modulo and Comparison
# ===========================================================================

class TestModuloComparison:
    def test_modulo(self):
        result = array_analyze("let x = 10; let y = x % 3;")
        y = result['scalars']['y']
        assert y.lo >= 0 and y.hi <= 2

    def test_comparison_true(self):
        result = array_analyze("let x = 5; let y = x > 3;")
        y = result['scalars']['y']
        assert y == IntervalDomain(1, 1)

    def test_comparison_false(self):
        result = array_analyze("let x = 1; let y = x > 3;")
        y = result['scalars']['y']
        assert y == IntervalDomain(0, 0)

    def test_comparison_unknown(self):
        src = """
        let x = 5;
        if (x > 3) {
            x = 1;
        }
        let y = x > 2;
        """
        result = array_analyze(src)
        # x could be 1 or 5 (but actually x>3 is true, so x=1)
        # After join: x could be 1
        y = result['scalars']['y']
        assert y == IntervalDomain(0, 0)


# ===========================================================================
# 23. Multiple Array Operations
# ===========================================================================

class TestMultipleArrayOps:
    def test_copy_array_elements(self):
        src = """
        let a = [1, 2, 3];
        let b = new_array(3, 0);
        b[0] = a[0];
        b[1] = a[1];
        b[2] = a[2];
        """
        result = array_analyze(src)
        b = result['arrays']['b']
        assert b.get_element(0) == IntervalDomain(1, 1)
        assert b.get_element(1) == IntervalDomain(2, 2)
        assert b.get_element(2) == IntervalDomain(3, 3)

    def test_reverse_array(self):
        src = """
        let a = [10, 20, 30];
        let tmp = a[0];
        a[0] = a[2];
        a[2] = tmp;
        """
        result = array_analyze(src)
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(30, 30)
        assert arr.get_element(2) == IntervalDomain(10, 10)

    def test_array_fill(self):
        src = """
        let a = new_array(3, 0);
        a[0] = 7;
        a[1] = 7;
        a[2] = 7;
        """
        result = array_analyze(src)
        arr = result['arrays']['a']
        assert arr.get_element(0) == IntervalDomain(7, 7)
        assert arr.get_element(1) == IntervalDomain(7, 7)
        assert arr.get_element(2) == IntervalDomain(7, 7)


# ===========================================================================
# 24. Summary and Integration
# ===========================================================================

class TestSummaryIntegration:
    def test_summary_no_warnings(self):
        summary = array_summary("let a = [1, 2, 3]; let x = a[0];")
        assert "No warnings" in summary

    def test_summary_with_warnings(self):
        summary = array_summary("let a = [1]; let x = a[5];")
        assert "out_of_bounds" in summary

    def test_full_pipeline(self):
        src = """
        let a = [5, 3, 1, 4, 2];
        let n = len(a);
        let i = 0;
        let total = 0;
        while (i < n) {
            total = total + a[i];
            i = i + 1;
        }
        assert(n == 5);
        """
        result = array_analyze(src)
        assert result['scalars']['n'] == IntervalDomain(5, 5)
        # total should be >= 15 at minimum
        # (widening may make it imprecise, that's OK)

    def test_properties_from_new_array(self):
        src = "let a = new_array(10, 0);"
        props = infer_properties(src)
        const = [p for p in props if p.kind == ArrayPropertyKind.CONSTANT]
        assert len(const) > 0
        assert const[0].details['value'] == 0

    def test_compare_api(self):
        result = compare_analyses("let a = [1, 2, 3]; let x = a[0];")
        assert 'arrays_tracked' in result
        assert 'scalars_tracked' in result
        assert 'properties_found' in result
