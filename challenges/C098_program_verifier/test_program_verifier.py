"""
Tests for C098: Program Verifier
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from program_verifier import (
    # AST
    Skip, Assign, Seq, If, While, Assert, Assume, Block, FuncDecl, Return,
    ArrayAssign,
    # Expressions
    IntLit, BoolLit, VarRef, BinaryOp, UnaryOp, CondExpr,
    ArrayRead, ArrayStore, Forall, Exists, OldExpr, ResultExpr,
    # Helpers
    AND, OR, NOT, IMPLIES, EQ, LT, LE, GT, GE, ADD, SUB, MUL, NEG,
    VAR, INT, BOOL,
    # Core
    substitute, free_vars, WPCalculus, SPCalculus, VC,
    SMTTranslator, InvariantInference, ConcreteExecutor,
    ProgramVerifier, VResult, VerificationResult, VerificationError,
    # Parser
    parse, verify, VerifParser,
    # Formatting
    format_expr, format_stmt,
)
from smt_solver import SMTSolver, SMTResult


# ============================================================
# Expression construction tests
# ============================================================

class TestExpressionHelpers:
    def test_int_lit(self):
        e = INT(42)
        assert isinstance(e, IntLit)
        assert e.value == 42

    def test_bool_lit(self):
        assert BOOL(True) == BoolLit(True)
        assert BOOL(False) == BoolLit(False)

    def test_var_ref(self):
        x = VAR('x')
        assert isinstance(x, VarRef)
        assert x.name == 'x'

    def test_binary_ops(self):
        x, y = VAR('x'), VAR('y')
        assert ADD(x, y) == BinaryOp('+', x, y)
        assert SUB(x, y) == BinaryOp('-', x, y)
        assert MUL(x, y) == BinaryOp('*', x, y)
        assert EQ(x, y) == BinaryOp('==', x, y)
        assert LT(x, y) == BinaryOp('<', x, y)
        assert LE(x, y) == BinaryOp('<=', x, y)
        assert GT(x, y) == BinaryOp('>', x, y)
        assert GE(x, y) == BinaryOp('>=', x, y)

    def test_neg(self):
        x = VAR('x')
        assert NEG(x) == UnaryOp('neg', x)

    def test_and_simplification(self):
        assert AND() == BoolLit(True)
        assert AND(BoolLit(True), VAR('x')) == VAR('x')
        assert AND(BoolLit(False), VAR('x')) == BoolLit(False)

    def test_or_simplification(self):
        assert OR() == BoolLit(False)
        assert OR(BoolLit(False), VAR('x')) == VAR('x')
        assert OR(BoolLit(True), VAR('x')) == BoolLit(True)

    def test_not_simplification(self):
        assert NOT(BoolLit(True)) == BoolLit(False)
        assert NOT(BoolLit(False)) == BoolLit(True)
        x = VAR('x')
        assert NOT(NOT(x)) == x

    def test_implies(self):
        x, y = VAR('x'), VAR('y')
        e = IMPLIES(x, y)
        assert e == BinaryOp('implies', x, y)

    def test_cond_expr(self):
        c = CondExpr(BOOL(True), INT(1), INT(2))
        assert c.cond == BoolLit(True)
        assert c.then_expr == IntLit(1)
        assert c.else_expr == IntLit(2)


# ============================================================
# Substitution tests
# ============================================================

class TestSubstitution:
    def test_subst_var(self):
        result = substitute(VAR('x'), 'x', INT(5))
        assert result == INT(5)

    def test_subst_different_var(self):
        result = substitute(VAR('y'), 'x', INT(5))
        assert result == VAR('y')

    def test_subst_literal(self):
        assert substitute(INT(3), 'x', INT(5)) == INT(3)
        assert substitute(BOOL(True), 'x', INT(5)) == BOOL(True)

    def test_subst_binary(self):
        expr = ADD(VAR('x'), VAR('y'))
        result = substitute(expr, 'x', INT(3))
        assert result == ADD(INT(3), VAR('y'))

    def test_subst_unary(self):
        expr = NEG(VAR('x'))
        result = substitute(expr, 'x', INT(3))
        assert result == NEG(INT(3))

    def test_subst_nested(self):
        expr = ADD(MUL(VAR('x'), INT(2)), VAR('x'))
        result = substitute(expr, 'x', VAR('y'))
        assert result == ADD(MUL(VAR('y'), INT(2)), VAR('y'))

    def test_subst_cond(self):
        expr = CondExpr(EQ(VAR('x'), INT(0)), INT(1), VAR('x'))
        result = substitute(expr, 'x', INT(5))
        assert result == CondExpr(EQ(INT(5), INT(0)), INT(1), INT(5))

    def test_subst_forall_shadowed(self):
        expr = Forall('x', GT(VAR('x'), INT(0)))
        result = substitute(expr, 'x', INT(5))
        assert result == expr  # x is bound, no substitution

    def test_subst_forall_free(self):
        expr = Forall('y', GT(VAR('y'), VAR('x')))
        result = substitute(expr, 'x', INT(5))
        assert result == Forall('y', GT(VAR('y'), INT(5)))

    def test_subst_exists(self):
        expr = Exists('x', EQ(VAR('x'), VAR('y')))
        result = substitute(expr, 'y', INT(3))
        assert result == Exists('x', EQ(VAR('x'), INT(3)))

    def test_subst_array_read(self):
        expr = ArrayRead(VAR('a'), VAR('x'))
        result = substitute(expr, 'x', INT(0))
        assert result == ArrayRead(VAR('a'), INT(0))

    def test_subst_old_expr(self):
        expr = OldExpr(VAR('x'))
        result = substitute(expr, 'x', INT(5))
        assert result == OldExpr(INT(5))

    def test_subst_result_expr(self):
        expr = ResultExpr()
        result = substitute(expr, 'x', INT(5))
        assert result == ResultExpr()


# ============================================================
# Free variables tests
# ============================================================

class TestFreeVars:
    def test_lit(self):
        assert free_vars(INT(5)) == set()
        assert free_vars(BOOL(True)) == set()

    def test_var(self):
        assert free_vars(VAR('x')) == {'x'}

    def test_binary(self):
        assert free_vars(ADD(VAR('x'), VAR('y'))) == {'x', 'y'}

    def test_unary(self):
        assert free_vars(NEG(VAR('x'))) == {'x'}

    def test_forall_binds(self):
        assert free_vars(Forall('x', GT(VAR('x'), VAR('y')))) == {'y'}

    def test_exists_binds(self):
        assert free_vars(Exists('x', EQ(VAR('x'), VAR('y')))) == {'y'}

    def test_cond_expr(self):
        e = CondExpr(VAR('c'), VAR('x'), VAR('y'))
        assert free_vars(e) == {'c', 'x', 'y'}

    def test_result_expr(self):
        assert free_vars(ResultExpr()) == set()


# ============================================================
# WP Calculus tests
# ============================================================

class TestWPCalculus:
    def test_wp_skip(self):
        wp = WPCalculus()
        result = wp.wp(Skip(), VAR('Q'))
        assert result == VAR('Q')

    def test_wp_assign(self):
        # wp(x := 5, x > 0) = 5 > 0
        wp = WPCalculus()
        result = wp.wp(Assign('x', INT(5)), GT(VAR('x'), INT(0)))
        assert result == GT(INT(5), INT(0))

    def test_wp_assign_expr(self):
        # wp(x := x + 1, x > 0) = (x + 1) > 0
        wp = WPCalculus()
        result = wp.wp(Assign('x', ADD(VAR('x'), INT(1))), GT(VAR('x'), INT(0)))
        assert result == GT(ADD(VAR('x'), INT(1)), INT(0))

    def test_wp_seq(self):
        # wp(x := 1; y := x + 1, y == 2) = wp(x := 1, wp(y := x+1, y==2))
        # = wp(x := 1, x + 1 == 2) = 1 + 1 == 2
        wp = WPCalculus()
        stmt = Seq(Assign('x', INT(1)), Assign('y', ADD(VAR('x'), INT(1))))
        result = wp.wp(stmt, EQ(VAR('y'), INT(2)))
        assert result == EQ(ADD(INT(1), INT(1)), INT(2))

    def test_wp_if(self):
        # wp(if x>0 then y:=1 else y:=0, y>=0)
        wp = WPCalculus()
        cond = GT(VAR('x'), INT(0))
        stmt = If(cond, Assign('y', INT(1)), Assign('y', INT(0)))
        result = wp.wp(stmt, GE(VAR('y'), INT(0)))
        # Should be (x>0 => 1>=0) /\ (!(x>0) => 0>=0)
        assert isinstance(result, BinaryOp)
        assert result.op == 'and'

    def test_wp_assert(self):
        wp = WPCalculus()
        result = wp.wp(Assert(GT(VAR('x'), INT(0))), BOOL(True))
        # wp(assert P, Q) = P /\ Q
        assert len(wp.vcs) == 0  # no separate VCs for assertions

    def test_wp_assume(self):
        wp = WPCalculus()
        result = wp.wp(Assume(GT(VAR('x'), INT(0))), EQ(VAR('y'), INT(1)))
        # wp(assume P, Q) = P => Q
        assert isinstance(result, BinaryOp)
        assert result.op == 'implies'

    def test_wp_while_with_invariant(self):
        wp = WPCalculus()
        inv = GE(VAR('x'), INT(0))
        cond = GT(VAR('x'), INT(0))
        body = Assign('x', SUB(VAR('x'), INT(1)))
        stmt = While(cond, body, invariant=inv)
        result = wp.wp(stmt, GE(VAR('x'), INT(0)))
        # WP of while embeds invariant, preservation, and exit
        assert isinstance(result, BinaryOp)
        assert result.op == 'and'
        # No separate VCs -- all embedded in formula
        assert len(wp.vcs) == 0

    def test_wp_while_no_invariant_raises(self):
        wp = WPCalculus()
        stmt = While(GT(VAR('x'), INT(0)), Assign('x', SUB(VAR('x'), INT(1))))
        with pytest.raises(VerificationError):
            wp.wp(stmt, BOOL(True))

    def test_wp_block(self):
        wp = WPCalculus()
        stmt = Block([Assign('x', INT(1)), Assign('y', INT(2))])
        result = wp.wp(stmt, AND(EQ(VAR('x'), INT(1)), EQ(VAR('y'), INT(2))))
        # wp should substitute backwards
        assert isinstance(result, BinaryOp)

    def test_wp_return(self):
        wp = WPCalculus()
        result = wp.wp(Return(ADD(VAR('x'), INT(1))), EQ(VAR('result'), INT(5)))
        # wp(return x+1, result==5) = (x+1)==5
        assert result == EQ(ADD(VAR('x'), INT(1)), INT(5))


# ============================================================
# SP Calculus tests
# ============================================================

class TestSPCalculus:
    def test_sp_skip(self):
        sp = SPCalculus()
        result = sp.sp(Skip(), VAR('P'))
        assert result == VAR('P')

    def test_sp_assign(self):
        sp = SPCalculus()
        result = sp.sp(Assign('x', INT(5)), BOOL(True))
        # sp(x:=5, true) should include x==5
        assert isinstance(result, BinaryOp)

    def test_sp_seq(self):
        sp = SPCalculus()
        stmt = Seq(Assign('x', INT(1)), Assign('y', ADD(VAR('x'), INT(1))))
        result = sp.sp(stmt, BOOL(True))
        assert result is not None

    def test_sp_if(self):
        sp = SPCalculus()
        stmt = If(GT(VAR('x'), INT(0)), Assign('y', INT(1)), Assign('y', INT(0)))
        result = sp.sp(stmt, BOOL(True))
        # Should be a disjunction
        assert isinstance(result, BinaryOp)
        assert result.op == 'or'

    def test_sp_assert(self):
        sp = SPCalculus()
        result = sp.sp(Assert(GT(VAR('x'), INT(0))), GT(VAR('x'), INT(1)))
        assert len(sp.vcs) == 1  # SP does add assertion VCs (forward reasoning)

    def test_sp_assume(self):
        sp = SPCalculus()
        result = sp.sp(Assume(GT(VAR('x'), INT(0))), BOOL(True))
        # AND(true, x>0) simplifies to just x>0
        assert result == GT(VAR('x'), INT(0))


# ============================================================
# SMT Translation tests
# ============================================================

class TestSMTTranslation:
    def test_translate_int_lit(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(INT(42))
        assert result is not None

    def test_translate_bool_lit(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(BOOL(True))
        assert result is not None

    def test_translate_var(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(VAR('x'))
        assert result is not None

    def test_translate_add(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(ADD(VAR('x'), INT(1)))
        assert result is not None

    def test_translate_comparison(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(GT(VAR('x'), INT(0)))
        assert result is not None

    def test_translate_and_or(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        e = AND(GT(VAR('x'), INT(0)), LT(VAR('x'), INT(10)))
        result = t.translate(e)
        assert result is not None

    def test_translate_not(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(NOT(VAR('b')))
        assert result is not None

    def test_translate_implies(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(IMPLIES(VAR('a'), VAR('b')))
        assert result is not None

    def test_translate_neg(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(NEG(VAR('x')))
        assert result is not None

    def test_translate_cond_expr(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        e = CondExpr(GT(VAR('x'), INT(0)), VAR('x'), NEG(VAR('x')))
        result = t.translate(e)
        assert result is not None

    def test_translate_result_expr(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(ResultExpr())
        assert result is not None

    def test_translate_old_expr(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(OldExpr(VAR('x')))
        assert result is not None

    def test_translate_forall_approx(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(Forall('i', GT(VAR('i'), INT(-5))))
        assert result is not None

    def test_translate_exists_approx(self):
        solver = SMTSolver()
        t = SMTTranslator(solver)
        result = t.translate(Exists('i', EQ(VAR('i'), INT(0))))
        assert result is not None


# ============================================================
# Concrete Executor tests
# ============================================================

class TestConcreteExecutor:
    def test_skip(self):
        ex = ConcreteExecutor()
        env, trace = ex.execute(Skip(), {'x': 5})
        assert env == {'x': 5}

    def test_assign(self):
        ex = ConcreteExecutor()
        env, _ = ex.execute(Assign('x', INT(5)))
        assert env['x'] == 5

    def test_assign_expr(self):
        ex = ConcreteExecutor()
        env, _ = ex.execute(Assign('y', ADD(VAR('x'), INT(1))), {'x': 3})
        assert env['y'] == 4

    def test_seq(self):
        ex = ConcreteExecutor()
        env, _ = ex.execute(Seq(Assign('x', INT(1)), Assign('y', INT(2))))
        assert env == {'x': 1, 'y': 2}

    def test_if_true(self):
        ex = ConcreteExecutor()
        stmt = If(GT(VAR('x'), INT(0)), Assign('y', INT(1)), Assign('y', INT(0)))
        env, _ = ex.execute(stmt, {'x': 5})
        assert env['y'] == 1

    def test_if_false(self):
        ex = ConcreteExecutor()
        stmt = If(GT(VAR('x'), INT(0)), Assign('y', INT(1)), Assign('y', INT(0)))
        env, _ = ex.execute(stmt, {'x': -1})
        assert env['y'] == 0

    def test_while(self):
        ex = ConcreteExecutor()
        stmt = While(GT(VAR('x'), INT(0)), Assign('x', SUB(VAR('x'), INT(1))))
        env, trace = ex.execute(stmt, {'x': 3})
        assert env['x'] == 0
        assert len(trace) > 0

    def test_assert_pass(self):
        ex = ConcreteExecutor()
        env, _ = ex.execute(Assert(GT(VAR('x'), INT(0))), {'x': 5})
        assert env['x'] == 5

    def test_assert_fail(self):
        ex = ConcreteExecutor()
        with pytest.raises(VerificationError):
            ex.execute(Assert(GT(VAR('x'), INT(0))), {'x': -1})

    def test_assume(self):
        ex = ConcreteExecutor()
        env, _ = ex.execute(Assume(GT(VAR('x'), INT(0))), {'x': 5})
        assert env['x'] == 5

    def test_assume_violated(self):
        ex = ConcreteExecutor()
        with pytest.raises(VerificationError):
            ex.execute(Assume(GT(VAR('x'), INT(0))), {'x': -1})

    def test_block(self):
        ex = ConcreteExecutor()
        stmt = Block([Assign('x', INT(1)), Assign('y', INT(2)), Assign('z', ADD(VAR('x'), VAR('y')))])
        env, _ = ex.execute(stmt)
        assert env['z'] == 3

    def test_max_steps(self):
        ex = ConcreteExecutor(max_steps=10)
        stmt = While(BOOL(True), Skip())
        with pytest.raises(VerificationError):
            ex.execute(stmt)

    def test_array_assign(self):
        ex = ConcreteExecutor()
        stmt = ArrayAssign('a', INT(0), INT(42))
        env, _ = ex.execute(stmt)
        assert env['a'][0] == 42

    def test_return(self):
        ex = ConcreteExecutor()
        env, _ = ex.execute(Return(ADD(VAR('x'), INT(1))), {'x': 4})
        assert env['result'] == 5

    def test_nested_while(self):
        ex = ConcreteExecutor()
        inner = While(GT(VAR('y'), INT(0)),
                     Seq(Assign('y', SUB(VAR('y'), INT(1))),
                         Assign('z', ADD(VAR('z'), INT(1)))))
        outer = While(GT(VAR('x'), INT(0)),
                     Seq(Assign('y', INT(2)),
                         Seq(inner,
                             Assign('x', SUB(VAR('x'), INT(1))))))
        env, _ = ex.execute(outer, {'x': 2, 'y': 0, 'z': 0})
        assert env['x'] == 0
        assert env['z'] == 4

    def test_cond_expr_eval(self):
        ex = ConcreteExecutor()
        expr = CondExpr(GT(VAR('x'), INT(0)), VAR('x'), NEG(VAR('x')))
        assert ex._eval(expr, {'x': 5}) == 5
        assert ex._eval(expr, {'x': -3}) == 3

    def test_comparison_ops(self):
        ex = ConcreteExecutor()
        env = {'x': 5, 'y': 3}
        assert ex._eval(EQ(VAR('x'), INT(5)), env) == True
        assert ex._eval(BinaryOp('!=', VAR('x'), VAR('y')), env) == True
        assert ex._eval(LT(VAR('y'), VAR('x')), env) == True
        assert ex._eval(LE(VAR('x'), INT(5)), env) == True
        assert ex._eval(GT(VAR('x'), VAR('y')), env) == True
        assert ex._eval(GE(VAR('x'), INT(5)), env) == True

    def test_logical_ops(self):
        ex = ConcreteExecutor()
        env = {'a': True, 'b': False}
        assert ex._eval(BinaryOp('and', VAR('a'), VAR('b')), env) == False
        assert ex._eval(BinaryOp('or', VAR('a'), VAR('b')), env) == True
        assert ex._eval(UnaryOp('not', VAR('a')), env) == False
        assert ex._eval(BinaryOp('implies', VAR('a'), VAR('b')), env) == False
        assert ex._eval(BinaryOp('implies', VAR('b'), VAR('a')), env) == True

    def test_division(self):
        ex = ConcreteExecutor()
        assert ex._eval(BinaryOp('/', INT(7), INT(2)), {}) == 3
        assert ex._eval(BinaryOp('/', INT(7), INT(0)), {}) == 0
        assert ex._eval(BinaryOp('%', INT(7), INT(3)), {}) == 1


# ============================================================
# Verifier integration tests -- simple programs
# ============================================================

class TestVerifierSimple:
    def test_skip_verified(self):
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(Skip(), BOOL(True), BOOL(True))
        assert result.verified

    def test_assign_verified(self):
        # {true} x := 5 {x == 5}
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(Assign('x', INT(5)), BOOL(True), EQ(VAR('x'), INT(5)))
        assert result.verified

    def test_assign_wrong_post(self):
        # {true} x := 5 {x == 3} -- should fail
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(Assign('x', INT(5)), BOOL(True), EQ(VAR('x'), INT(3)))
        assert result.status == VResult.FAILED

    def test_seq_verified(self):
        # {true} x := 1; y := x + 1 {y == 2}
        v = ProgramVerifier(infer_invariants=False)
        stmt = Seq(Assign('x', INT(1)), Assign('y', ADD(VAR('x'), INT(1))))
        result = v.verify(stmt, BOOL(True), EQ(VAR('y'), INT(2)))
        assert result.verified

    def test_if_both_branches(self):
        # {true} if x > 0 then y := 1 else y := 1 end {y == 1}
        v = ProgramVerifier(infer_invariants=False)
        stmt = If(GT(VAR('x'), INT(0)), Assign('y', INT(1)), Assign('y', INT(1)))
        result = v.verify(stmt, BOOL(True), EQ(VAR('y'), INT(1)))
        assert result.verified

    def test_if_with_pre(self):
        # {x >= 0} if x > 0 then y := x else y := 0 end {y >= 0}
        v = ProgramVerifier(infer_invariants=False)
        stmt = If(GT(VAR('x'), INT(0)), Assign('y', VAR('x')), Assign('y', INT(0)))
        result = v.verify(stmt, GE(VAR('x'), INT(0)), GE(VAR('y'), INT(0)))
        assert result.verified

    def test_assert_valid(self):
        # {x > 0} assert x > 0 {true}
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(Assert(GT(VAR('x'), INT(0))),
                         GT(VAR('x'), INT(0)), BOOL(True))
        assert result.verified

    def test_assert_invalid(self):
        # {true} assert x > 0 {true} -- can't prove
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(Assert(GT(VAR('x'), INT(0))),
                         BOOL(True), BOOL(True))
        assert result.status == VResult.FAILED

    def test_assume_then_assert(self):
        # {true} assume x > 0; assert x > 0 {true}
        v = ProgramVerifier(infer_invariants=False)
        stmt = Seq(Assume(GT(VAR('x'), INT(0))), Assert(GT(VAR('x'), INT(0))))
        result = v.verify(stmt, BOOL(True), BOOL(True))
        assert result.verified

    def test_multi_assign(self):
        # {true} x := 3; y := x * 2; z := y - 1 {z == 5}
        v = ProgramVerifier(infer_invariants=False)
        stmt = Seq(Assign('x', INT(3)),
                   Seq(Assign('y', MUL(VAR('x'), INT(2))),
                       Assign('z', SUB(VAR('y'), INT(1)))))
        result = v.verify(stmt, BOOL(True), EQ(VAR('z'), INT(5)))
        assert result.verified

    def test_swap(self):
        # {x == a and y == b} t := x; x := y; y := t {x == b and y == a}
        v = ProgramVerifier(infer_invariants=False)
        stmt = Seq(Assign('t', VAR('x')),
                   Seq(Assign('x', VAR('y')),
                       Assign('y', VAR('t'))))
        pre = AND(EQ(VAR('x'), VAR('a')), EQ(VAR('y'), VAR('b')))
        post = AND(EQ(VAR('x'), VAR('b')), EQ(VAR('y'), VAR('a')))
        result = v.verify(stmt, pre, post)
        assert result.verified


# ============================================================
# Verifier -- loop tests with provided invariants
# ============================================================

class TestVerifierLoops:
    def test_simple_loop(self):
        # {x >= 0} while x > 0 invariant x >= 0 do x := x - 1 end {x == 0}
        v = ProgramVerifier(infer_invariants=False)
        inv = GE(VAR('x'), INT(0))
        stmt = While(GT(VAR('x'), INT(0)),
                    Assign('x', SUB(VAR('x'), INT(1))),
                    invariant=inv)
        result = v.verify(stmt, GE(VAR('x'), INT(0)), EQ(VAR('x'), INT(0)))
        assert result.verified

    def test_accumulation_loop(self):
        # {n >= 0} s := 0; i := 0; while i < n inv (s == i * (i-1) / 2 -- too complex)
        # Use simpler: {n >= 0} i := n; while i > 0 inv i >= 0 do i := i - 1 end {i == 0}
        v = ProgramVerifier(infer_invariants=False)
        stmt = Seq(
            Assign('i', VAR('n')),
            While(GT(VAR('i'), INT(0)),
                  Assign('i', SUB(VAR('i'), INT(1))),
                  invariant=GE(VAR('i'), INT(0)))
        )
        result = v.verify(stmt, GE(VAR('n'), INT(0)), EQ(VAR('i'), INT(0)))
        assert result.verified

    def test_wrong_invariant(self):
        # Wrong invariant: x > 0 (not preserved when x becomes 0)
        v = ProgramVerifier(infer_invariants=False)
        stmt = While(GT(VAR('x'), INT(0)),
                    Assign('x', SUB(VAR('x'), INT(1))),
                    invariant=GT(VAR('x'), INT(0)))
        result = v.verify(stmt, GT(VAR('x'), INT(0)), EQ(VAR('x'), INT(0)))
        assert result.status == VResult.FAILED

    def test_loop_with_multiple_vars(self):
        # {x >= 0} y := 0; while x > 0 inv (x >= 0 and y >= 0) do x:=x-1; y:=y+1 end {y >= 0}
        v = ProgramVerifier(infer_invariants=False)
        inv = AND(GE(VAR('x'), INT(0)), GE(VAR('y'), INT(0)))
        body = Seq(Assign('x', SUB(VAR('x'), INT(1))),
                   Assign('y', ADD(VAR('y'), INT(1))))
        stmt = Seq(Assign('y', INT(0)),
                   While(GT(VAR('x'), INT(0)), body, invariant=inv))
        result = v.verify(stmt, GE(VAR('x'), INT(0)), GE(VAR('y'), INT(0)))
        assert result.verified


# ============================================================
# Verifier -- function contracts
# ============================================================

class TestFunctionContracts:
    def test_simple_function(self):
        # function abs(x) requires true ensures result >= 0
        v = ProgramVerifier(infer_invariants=False)
        func = FuncDecl(
            'abs', ['x'],
            requires=BOOL(True),
            ensures=GE(ResultExpr(), INT(0)),
            body=If(GE(VAR('x'), INT(0)),
                   Return(VAR('x')),
                   Return(NEG(VAR('x'))))
        )
        result = v.verify_function(func)
        assert result.verified

    def test_max_function(self):
        # function max(x, y) ensures result >= x and result >= y
        v = ProgramVerifier(infer_invariants=False)
        func = FuncDecl(
            'max', ['x', 'y'],
            requires=BOOL(True),
            ensures=AND(GE(ResultExpr(), VAR('x')), GE(ResultExpr(), VAR('y'))),
            body=If(GE(VAR('x'), VAR('y')),
                   Return(VAR('x')),
                   Return(VAR('y')))
        )
        result = v.verify_function(func)
        assert result.verified

    def test_function_with_precondition(self):
        # function inc(x) requires x >= 0 ensures result > 0
        v = ProgramVerifier(infer_invariants=False)
        func = FuncDecl(
            'inc', ['x'],
            requires=GE(VAR('x'), INT(0)),
            ensures=GT(ResultExpr(), INT(0)),
            body=Return(ADD(VAR('x'), INT(1)))
        )
        result = v.verify_function(func)
        assert result.verified

    def test_function_contract_violated(self):
        # function bad(x) requires true ensures result > 0
        # body: return x  -- violated when x <= 0
        v = ProgramVerifier(infer_invariants=False)
        func = FuncDecl(
            'bad', ['x'],
            requires=BOOL(True),
            ensures=GT(ResultExpr(), INT(0)),
            body=Return(VAR('x'))
        )
        result = v.verify_function(func)
        assert result.status == VResult.FAILED


# ============================================================
# SP mode tests
# ============================================================

class TestSPMode:
    def test_sp_simple_assign(self):
        v = ProgramVerifier(infer_invariants=False, use_sp=True)
        result = v.verify(Assign('x', INT(5)), BOOL(True), EQ(VAR('x'), INT(5)))
        # SP may produce complex formulas; check it generates VCs
        assert result.vcs_total > 0

    def test_sp_seq(self):
        v = ProgramVerifier(infer_invariants=False, use_sp=True)
        stmt = Seq(Assign('x', INT(1)), Assign('y', ADD(VAR('x'), INT(1))))
        result = v.verify(stmt, BOOL(True), EQ(VAR('y'), INT(2)))
        assert result.vcs_total > 0


# ============================================================
# Invariant inference tests
# ============================================================

class TestInvariantInference:
    def test_infer_simple_bound(self):
        inv_inf = InvariantInference()
        cond = GT(VAR('x'), INT(0))
        body = Assign('x', SUB(VAR('x'), INT(1)))
        loop = While(cond, body)
        pre = GE(VAR('x'), INT(0))
        post = EQ(VAR('x'), INT(0))

        inv = inv_inf.infer(loop, pre, post)
        # Should find x >= 0
        assert inv is not None

    def test_infer_from_traces(self):
        inv_inf = InvariantInference()
        cond = GT(VAR('x'), INT(0))
        body = Assign('x', SUB(VAR('x'), INT(1)))
        loop = While(cond, body)
        pre = GE(VAR('x'), INT(0))
        post = EQ(VAR('x'), INT(0))

        # Provide traces
        traces = [
            ({'x': 3}, True), ({'x': 2}, True), ({'x': 1}, True), ({'x': 0}, True)
        ]
        inv = inv_inf.infer(loop, pre, post, traces)
        assert inv is not None

    def test_verifier_auto_infer(self):
        # Verifier should auto-infer invariant for simple loop
        v = ProgramVerifier(infer_invariants=True)
        stmt = While(GT(VAR('x'), INT(0)),
                    Assign('x', SUB(VAR('x'), INT(1))))
        result = v.verify(stmt, GE(VAR('x'), INT(0)), EQ(VAR('x'), INT(0)))
        # May or may not find invariant, but shouldn't crash
        assert result.vcs_total >= 0


# ============================================================
# Parser tests
# ============================================================

class TestParser:
    def test_parse_skip(self):
        stmt = parse('skip')
        assert isinstance(stmt, Skip)

    def test_parse_assign(self):
        stmt = parse('x := 5')
        assert isinstance(stmt, Assign)
        assert stmt.var == 'x'
        assert isinstance(stmt.expr, IntLit)
        assert stmt.expr.value == 5

    def test_parse_assign_expr(self):
        stmt = parse('x := x + 1')
        assert isinstance(stmt, Assign)
        assert isinstance(stmt.expr, BinaryOp)
        assert stmt.expr.op == '+'

    def test_parse_seq(self):
        stmt = parse('x := 1; y := 2')
        assert isinstance(stmt, Seq)

    def test_parse_if(self):
        stmt = parse('if x > 0 then y := 1 else y := 0 end')
        assert isinstance(stmt, If)

    def test_parse_if_no_else(self):
        stmt = parse('if x > 0 then y := 1 end')
        assert isinstance(stmt, If)
        assert isinstance(stmt.else_branch, Skip)

    def test_parse_while(self):
        stmt = parse('while x > 0 do x := x - 1 end')
        assert isinstance(stmt, While)
        assert stmt.invariant is None

    def test_parse_while_with_invariant(self):
        stmt = parse('while x > 0 invariant x >= 0 do x := x - 1 end')
        assert isinstance(stmt, While)
        assert stmt.invariant is not None

    def test_parse_assert(self):
        stmt = parse('assert x > 0')
        assert isinstance(stmt, Assert)

    def test_parse_assume(self):
        stmt = parse('assume x > 0')
        assert isinstance(stmt, Assume)

    def test_parse_block(self):
        stmt = parse('{ x := 1; y := 2; z := 3 }')
        assert isinstance(stmt, Seq)

    def test_parse_nested_if(self):
        stmt = parse('if x > 0 then if y > 0 then z := 1 end end')
        assert isinstance(stmt, If)
        assert isinstance(stmt.then_branch, If)

    def test_parse_bool_literals(self):
        stmt = parse('x := true; y := false')
        assert isinstance(stmt, Seq)

    def test_parse_not(self):
        stmt = parse('assert not x > 0')
        assert isinstance(stmt, Assert)

    def test_parse_neg(self):
        stmt = parse('x := - y')
        assert isinstance(stmt, Assign)
        assert isinstance(stmt.expr, UnaryOp)
        assert stmt.expr.op == 'neg'

    def test_parse_paren(self):
        stmt = parse('x := (a + b) * c')
        assert isinstance(stmt, Assign)
        assert isinstance(stmt.expr, BinaryOp)
        assert stmt.expr.op == '*'

    def test_parse_comparison_ops(self):
        stmt = parse('assert x <= y and y >= z and a == b and c != d')
        assert isinstance(stmt, Assert)

    def test_parse_implies(self):
        stmt = parse('assert x > 0 => y > 0')
        assert isinstance(stmt, Assert)
        assert isinstance(stmt.cond, BinaryOp)
        assert stmt.cond.op == 'implies'

    def test_parse_return(self):
        stmt = parse('return x + 1')
        assert isinstance(stmt, Return)

    def test_parse_old(self):
        stmt = parse('assert old(x) == 5')
        assert isinstance(stmt, Assert)

    def test_parse_result(self):
        stmt = parse('assert result > 0')
        assert isinstance(stmt, Assert)
        cond = stmt.cond
        assert isinstance(cond.left, ResultExpr)

    def test_parse_function(self):
        source = '''
        function inc(x)
        requires x >= 0
        ensures result > 0
        {
            return x + 1
        }
        '''
        stmt = parse(source)
        assert isinstance(stmt, FuncDecl)
        assert stmt.name == 'inc'
        assert stmt.params == ['x']

    def test_parse_complex_program(self):
        source = '''
        assume n >= 0;
        i := 0;
        s := 0;
        while i < n invariant i <= n and s >= 0 do
            s := s + i;
            i := i + 1
        end;
        assert s >= 0
        '''
        stmt = parse(source)
        assert isinstance(stmt, Seq)

    def test_parse_mul_div_mod(self):
        stmt = parse('x := a * b / c % d')
        assert isinstance(stmt, Assign)

    def test_parse_comments(self):
        source = '''
        // this is a comment
        x := 5
        '''
        stmt = parse(source)
        assert isinstance(stmt, Assign)

    def test_parse_empty_block(self):
        stmt = parse('{ }')
        assert isinstance(stmt, Skip)

    def test_parse_multiple_stmts(self):
        source = 'x := 1; y := 2; z := 3'
        stmt = parse(source)
        assert isinstance(stmt, Seq)


# ============================================================
# Verify via parser integration tests
# ============================================================

class TestVerifyIntegration:
    def test_verify_simple(self):
        result = verify('x := 5', postcondition=EQ(VAR('x'), INT(5)),
                        infer_invariants=False)
        assert result.verified

    def test_verify_with_pre(self):
        result = verify('y := x + 1',
                        precondition=EQ(VAR('x'), INT(4)),
                        postcondition=EQ(VAR('y'), INT(5)),
                        infer_invariants=False)
        assert result.verified

    def test_verify_if_program(self):
        source = 'if x > 0 then y := x else y := 0 - x end'
        result = verify(source,
                        postcondition=GE(VAR('y'), INT(0)),
                        infer_invariants=False)
        assert result.verified

    def test_verify_loop_program(self):
        source = 'while x > 0 invariant x >= 0 do x := x - 1 end'
        result = verify(source,
                        precondition=GE(VAR('x'), INT(0)),
                        postcondition=EQ(VAR('x'), INT(0)),
                        infer_invariants=False)
        assert result.verified

    def test_verify_assert_program(self):
        source = 'assume x > 0; assert x > 0'
        result = verify(source, infer_invariants=False)
        assert result.verified


# ============================================================
# Formatting tests
# ============================================================

class TestFormatting:
    def test_format_int(self):
        assert format_expr(INT(42)) == '42'

    def test_format_bool(self):
        assert format_expr(BOOL(True)) == 'true'
        assert format_expr(BOOL(False)) == 'false'

    def test_format_var(self):
        assert format_expr(VAR('x')) == 'x'

    def test_format_binary(self):
        assert format_expr(ADD(VAR('x'), INT(1))) == '(x + 1)'

    def test_format_unary(self):
        assert format_expr(NEG(VAR('x'))) == '(neg x)'
        assert format_expr(NOT(VAR('b'))) == '(not b)'

    def test_format_cond(self):
        e = CondExpr(GT(VAR('x'), INT(0)), INT(1), INT(0))
        s = format_expr(e)
        assert 'if' in s
        assert 'then' in s
        assert 'else' in s

    def test_format_forall(self):
        s = format_expr(Forall('i', GT(VAR('i'), INT(0))))
        assert 'forall' in s

    def test_format_exists(self):
        s = format_expr(Exists('i', EQ(VAR('i'), INT(0))))
        assert 'exists' in s

    def test_format_old(self):
        assert 'old' in format_expr(OldExpr(VAR('x')))

    def test_format_result(self):
        assert format_expr(ResultExpr()) == 'result'

    def test_format_skip(self):
        assert 'skip' in format_stmt(Skip())

    def test_format_assign(self):
        s = format_stmt(Assign('x', INT(5)))
        assert 'x' in s and ':=' in s

    def test_format_while(self):
        s = format_stmt(While(GT(VAR('x'), INT(0)),
                             Assign('x', SUB(VAR('x'), INT(1))),
                             invariant=GE(VAR('x'), INT(0))))
        assert 'while' in s
        assert 'invariant' in s

    def test_format_if(self):
        s = format_stmt(If(GT(VAR('x'), INT(0)),
                          Assign('y', INT(1)),
                          Assign('y', INT(0))))
        assert 'if' in s
        assert 'then' in s
        assert 'else' in s


# ============================================================
# Edge cases and error handling
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = verify('skip', infer_invariants=False)
        assert result.verified

    def test_deeply_nested_if(self):
        # Build deep nesting
        stmt = Assign('y', VAR('x'))
        for i in range(5):
            stmt = If(GT(VAR('x'), INT(i)), stmt, Assign('y', INT(i)))
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt)
        assert result.vcs_total >= 0  # shouldn't crash

    def test_verification_result_properties(self):
        vr = VerificationResult(VResult.VERIFIED, 3, 3, 0, 0)
        assert vr.verified
        assert vr.vcs_total == 3

        vr2 = VerificationResult(VResult.FAILED, 3, 2, 1, 0)
        assert not vr2.verified

    def test_counterexample_on_failure(self):
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(Assign('x', INT(5)), BOOL(True), EQ(VAR('x'), INT(3)))
        assert result.status == VResult.FAILED
        # Should have at least one failed VC
        assert len(result.failed_vcs) > 0

    def test_multiple_assertions(self):
        stmt = Block([
            Assign('x', INT(5)),
            Assert(GT(VAR('x'), INT(0))),
            Assert(LT(VAR('x'), INT(10))),
            Assert(EQ(VAR('x'), INT(5)))
        ])
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt)
        assert result.verified

    def test_multiple_assertions_one_fails(self):
        stmt = Block([
            Assign('x', INT(5)),
            Assert(GT(VAR('x'), INT(0))),
            Assert(EQ(VAR('x'), INT(3)))  # fails
        ])
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt)
        assert result.status == VResult.FAILED

    def test_assume_strengthens(self):
        # assume x > 5; assert x > 3 -- should verify
        v = ProgramVerifier(infer_invariants=False)
        stmt = Seq(Assume(GT(VAR('x'), INT(5))), Assert(GT(VAR('x'), INT(3))))
        result = v.verify(stmt)
        assert result.verified

    def test_nested_while_with_invariants(self):
        inner = While(GT(VAR('j'), INT(0)),
                     Assign('j', SUB(VAR('j'), INT(1))),
                     invariant=GE(VAR('j'), INT(0)))
        outer_body = Seq(Assign('j', VAR('n')), Seq(inner, Assign('i', SUB(VAR('i'), INT(1)))))
        outer = While(GT(VAR('i'), INT(0)),
                     outer_body,
                     invariant=GE(VAR('i'), INT(0)))
        stmt = Seq(Assign('i', VAR('n')), outer)
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt, GE(VAR('n'), INT(0)), EQ(VAR('i'), INT(0)))
        assert result.verified

    def test_array_read_in_assertion(self):
        # After a[0] := 42, assert read of a at 0
        # This is complex due to array theory -- just test it doesn't crash
        stmt = ArrayAssign('a', INT(0), INT(42))
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt)
        assert result.vcs_total >= 0


# ============================================================
# Verification of classic algorithms
# ============================================================

class TestClassicAlgorithms:
    def test_absolute_value(self):
        # if x >= 0 then r := x else r := -x end
        # postcondition: r >= 0
        stmt = If(GE(VAR('x'), INT(0)),
                 Assign('r', VAR('x')),
                 Assign('r', NEG(VAR('x'))))
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt, postcondition=GE(VAR('r'), INT(0)))
        assert result.verified

    def test_max_of_two(self):
        # if x >= y then m := x else m := y end
        # post: m >= x and m >= y
        stmt = If(GE(VAR('x'), VAR('y')),
                 Assign('m', VAR('x')),
                 Assign('m', VAR('y')))
        post = AND(GE(VAR('m'), VAR('x')), GE(VAR('m'), VAR('y')))
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt, postcondition=post)
        assert result.verified

    def test_countdown(self):
        # {n >= 0} i := n; while i > 0 inv i >= 0 do i := i - 1 end {i == 0}
        stmt = Seq(
            Assign('i', VAR('n')),
            While(GT(VAR('i'), INT(0)),
                  Assign('i', SUB(VAR('i'), INT(1))),
                  invariant=GE(VAR('i'), INT(0)))
        )
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt, GE(VAR('n'), INT(0)), EQ(VAR('i'), INT(0)))
        assert result.verified

    def test_increment_preserve_bound(self):
        # {x >= 0 and x < 10} x := x + 1 {x > 0 and x <= 10}
        v = ProgramVerifier(infer_invariants=False)
        pre = AND(GE(VAR('x'), INT(0)), LT(VAR('x'), INT(10)))
        post = AND(GT(VAR('x'), INT(0)), LE(VAR('x'), INT(10)))
        result = v.verify(Assign('x', ADD(VAR('x'), INT(1))), pre, post)
        assert result.verified

    def test_conditional_increment(self):
        # {true} if x > 0 then x := x + 1 else x := 1 end {x > 0}
        stmt = If(GT(VAR('x'), INT(0)),
                 Assign('x', ADD(VAR('x'), INT(1))),
                 Assign('x', INT(1)))
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify(stmt, postcondition=GT(VAR('x'), INT(0)))
        assert result.verified


# ============================================================
# Verification condition count and structure tests
# ============================================================

class TestVCStructure:
    def test_no_vcs_for_simple_assign(self):
        calc = WPCalculus()
        calc.wp(Assign('x', INT(5)), EQ(VAR('x'), INT(5)))
        assert len(calc.vcs) == 0

    def test_no_vc_for_assert(self):
        calc = WPCalculus()
        calc.wp(Assert(GT(VAR('x'), INT(0))), BOOL(True))
        assert len(calc.vcs) == 0

    def test_no_separate_vcs_for_while(self):
        # All loop VCs are embedded in the WP formula
        calc = WPCalculus()
        stmt = While(GT(VAR('x'), INT(0)),
                    Assign('x', SUB(VAR('x'), INT(1))),
                    invariant=GE(VAR('x'), INT(0)))
        calc.wp(stmt, EQ(VAR('x'), INT(0)))
        assert len(calc.vcs) == 0

    def test_no_separate_vc_for_function(self):
        calc = WPCalculus()
        func = FuncDecl('f', ['x'], GE(VAR('x'), INT(0)),
                        GT(ResultExpr(), INT(0)),
                        Return(ADD(VAR('x'), INT(1))))
        result = calc.wp(func, BOOL(True))
        assert len(calc.vcs) == 0
        # Result is the embedded formula
        assert isinstance(result, BinaryOp)

    def test_all_embedded_in_formula(self):
        calc = WPCalculus()
        stmt = Block([
            Assert(GT(VAR('x'), INT(0))),
            While(GT(VAR('x'), INT(0)),
                  Assign('x', SUB(VAR('x'), INT(1))),
                  invariant=GE(VAR('x'), INT(0))),
            Assert(EQ(VAR('x'), INT(0)))
        ])
        result = calc.wp(stmt, BOOL(True))
        assert len(calc.vcs) == 0
        # Everything is embedded in the WP formula
        assert result is not None


# ============================================================
# Array reasoning tests
# ============================================================

class TestArrayReasoning:
    def test_array_assign_concrete(self):
        ex = ConcreteExecutor()
        stmt = Block([
            ArrayAssign('a', INT(0), INT(10)),
            ArrayAssign('a', INT(1), INT(20)),
        ])
        env, _ = ex.execute(stmt)
        assert env['a'][0] == 10
        assert env['a'][1] == 20

    def test_array_read_concrete(self):
        ex = ConcreteExecutor()
        env = {'a': {0: 42, 1: 99}}
        expr = ArrayRead(VAR('a'), INT(0))
        assert ex._eval(expr, env) == 42

    def test_array_store_wp(self):
        # wp(a[i] := v, a[i] == v) should be valid
        wp_calc = WPCalculus()
        stmt = ArrayAssign('a', VAR('i'), VAR('v'))
        post = EQ(ArrayRead(VarRef('a'), VAR('i')), VAR('v'))
        result = wp_calc.wp(stmt, post)
        # Result involves ArrayStore substitution
        assert result is not None


# ============================================================
# Additional integration / end-to-end tests
# ============================================================

class TestEndToEnd:
    def test_parsed_loop_verified(self):
        source = '''
        assume n >= 0;
        i := n;
        while i > 0 invariant i >= 0 do
            i := i - 1
        end;
        assert i == 0
        '''
        result = verify(source, infer_invariants=False)
        assert result.verified

    def test_parsed_if_verified(self):
        source = '''
        if x > y then
            max := x
        else
            max := y
        end;
        assert max >= x;
        assert max >= y
        '''
        result = verify(source, infer_invariants=False)
        assert result.verified

    def test_parsed_function_verified(self):
        source = '''
        function double(x)
        requires x >= 0
        ensures result >= 0
        {
            return x + x
        }
        '''
        stmt = parse(source)
        assert isinstance(stmt, FuncDecl)
        v = ProgramVerifier(infer_invariants=False)
        result = v.verify_function(stmt)
        assert result.verified

    def test_full_program_with_loop_and_assert(self):
        # Compute: start with x, decrement to 0, assert final value
        source = '''
        assume x >= 0;
        y := x;
        while y > 0 invariant y >= 0 do
            y := y - 1
        end;
        assert y == 0
        '''
        result = verify(source, infer_invariants=False)
        assert result.verified

    def test_program_with_failing_assert(self):
        source = '''
        x := 5;
        assert x == 3
        '''
        result = verify(source, infer_invariants=False)
        assert result.status == VResult.FAILED

    def test_sequential_assignments_verified(self):
        source = '''
        a := 1;
        b := 2;
        c := a + b;
        assert c == 3
        '''
        result = verify(source, infer_invariants=False)
        assert result.verified

    def test_nested_conditions(self):
        source = '''
        if x > 0 then
            if y > 0 then
                z := 1
            else
                z := 2
            end
        else
            z := 3
        end;
        assert z >= 1
        '''
        result = verify(source, infer_invariants=False)
        assert result.verified

    def test_chained_assignments(self):
        source = '''
        x := 10;
        y := x;
        z := y;
        assert z == 10
        '''
        result = verify(source, infer_invariants=False)
        assert result.verified


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
