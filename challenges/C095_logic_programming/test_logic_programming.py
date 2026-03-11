"""Tests for C095: Logic Programming Engine."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from logic_programming import (
    Atom, Number, Var, Compound, NIL, cons, make_list, list_to_python,
    Substitution, unify, occurs_check, eval_arith,
    Clause, Interpreter, Parser, Lexer, Token,
    parse, parse_term, parse_query, run, PrologError,
)


# ============================================================
# Terms
# ============================================================

class TestTerms:
    def test_atom_equality(self):
        assert Atom('foo') == Atom('foo')
        assert Atom('foo') != Atom('bar')

    def test_atom_hash(self):
        assert hash(Atom('a')) == hash(Atom('a'))
        s = {Atom('a'), Atom('b'), Atom('a')}
        assert len(s) == 2

    def test_number_equality(self):
        assert Number(42) == Number(42)
        assert Number(1) != Number(2)

    def test_number_repr(self):
        assert repr(Number(5)) == '5'
        assert repr(Number(3.14)) == '3.14'
        assert repr(Number(2.0)) == '2'  # float that equals int

    def test_var_equality(self):
        assert Var('X') == Var('X')
        assert Var('X') != Var('Y')

    def test_compound_equality(self):
        c1 = Compound('f', [Atom('a'), Number(1)])
        c2 = Compound('f', [Atom('a'), Number(1)])
        c3 = Compound('f', [Atom('b'), Number(1)])
        assert c1 == c2
        assert c1 != c3

    def test_compound_indicator(self):
        c = Compound('foo', [Atom('a'), Atom('b')])
        assert c.indicator == 'foo/2'

    def test_nil(self):
        assert NIL == Atom('[]')

    def test_cons(self):
        c = cons(Atom('a'), NIL)
        assert isinstance(c, Compound)
        assert c.functor == '.'
        assert c.args[0] == Atom('a')
        assert c.args[1] == NIL

    def test_make_list(self):
        lst = make_list([Atom('a'), Atom('b'), Atom('c')])
        assert list_to_python(lst) == [Atom('a'), Atom('b'), Atom('c')]

    def test_make_list_with_tail(self):
        lst = make_list([Atom('a')], Var('T'))
        assert isinstance(lst, Compound)
        assert lst.args[0] == Atom('a')
        assert lst.args[1] == Var('T')

    def test_list_to_python_nil(self):
        assert list_to_python(NIL) == []

    def test_list_to_python_improper(self):
        # improper list returns None
        lst = cons(Atom('a'), Atom('b'))
        assert list_to_python(lst) is None

    def test_list_repr(self):
        lst = make_list([Number(1), Number(2), Number(3)])
        assert '[' in repr(lst)


# ============================================================
# Substitution
# ============================================================

class TestSubstitution:
    def test_empty_subst(self):
        s = Substitution()
        assert s.walk(Var('X')) == Var('X')

    def test_bind_and_lookup(self):
        s = Substitution()
        s = s.bind('X', Atom('hello'))
        assert s.walk(Var('X')) == Atom('hello')

    def test_chain_binding(self):
        s = Substitution()
        s = s.bind('X', Var('Y'))
        s = s.bind('Y', Atom('z'))
        assert s.walk(Var('X')) == Atom('z')

    def test_walk_compound(self):
        s = Substitution({'X': Number(1), 'Y': Number(2)})
        t = Compound('f', [Var('X'), Var('Y')])
        result = s.walk(t)
        assert result == Compound('f', [Number(1), Number(2)])

    def test_copy(self):
        s = Substitution({'X': Atom('a')})
        s2 = s.copy()
        s2.bindings['Y'] = Atom('b')
        assert 'Y' not in s.bindings


# ============================================================
# Unification
# ============================================================

class TestUnification:
    def test_unify_atoms(self):
        s = unify(Atom('a'), Atom('a'))
        assert s is not None
        s = unify(Atom('a'), Atom('b'))
        assert s is None

    def test_unify_numbers(self):
        s = unify(Number(1), Number(1))
        assert s is not None
        s = unify(Number(1), Number(2))
        assert s is None

    def test_unify_var_atom(self):
        s = unify(Var('X'), Atom('hello'))
        assert s is not None
        assert s.walk(Var('X')) == Atom('hello')

    def test_unify_var_var(self):
        s = unify(Var('X'), Var('Y'))
        assert s is not None

    def test_unify_compound(self):
        t1 = Compound('f', [Var('X'), Atom('b')])
        t2 = Compound('f', [Atom('a'), Var('Y')])
        s = unify(t1, t2)
        assert s is not None
        assert s.walk(Var('X')) == Atom('a')
        assert s.walk(Var('Y')) == Atom('b')

    def test_unify_compound_mismatch_functor(self):
        t1 = Compound('f', [Atom('a')])
        t2 = Compound('g', [Atom('a')])
        assert unify(t1, t2) is None

    def test_unify_compound_mismatch_arity(self):
        t1 = Compound('f', [Atom('a')])
        t2 = Compound('f', [Atom('a'), Atom('b')])
        assert unify(t1, t2) is None

    def test_occurs_check(self):
        # X = f(X) should fail with occurs check
        t1 = Var('X')
        t2 = Compound('f', [Var('X')])
        s = unify(t1, t2, check_occurs=True)
        assert s is None

    def test_unify_without_occurs_check(self):
        t1 = Var('X')
        t2 = Compound('f', [Var('X')])
        s = unify(t1, t2, check_occurs=False)
        assert s is not None

    def test_unify_lists(self):
        l1 = make_list([Number(1), Number(2)])
        l2 = make_list([Var('X'), Var('Y')])
        s = unify(l1, l2)
        assert s is not None
        assert s.walk(Var('X')) == Number(1)
        assert s.walk(Var('Y')) == Number(2)

    def test_unify_list_head_tail(self):
        l = make_list([Number(1), Number(2), Number(3)])
        pattern = cons(Var('H'), Var('T'))
        s = unify(l, pattern)
        assert s is not None
        assert s.walk(Var('H')) == Number(1)
        tail = list_to_python(s.walk(Var('T')))
        assert tail == [Number(2), Number(3)]

    def test_unify_nested_compound(self):
        t1 = Compound('f', [Compound('g', [Var('X')]), Var('Y')])
        t2 = Compound('f', [Compound('g', [Number(1)]), Atom('a')])
        s = unify(t1, t2)
        assert s is not None
        assert s.walk(Var('X')) == Number(1)
        assert s.walk(Var('Y')) == Atom('a')


# ============================================================
# Parser
# ============================================================

class TestParser:
    def test_parse_atom(self):
        t = parse_term("hello")
        assert t == Atom('hello')

    def test_parse_number(self):
        t = parse_term("42")
        assert t == Number(42)

    def test_parse_negative_number(self):
        t = parse_term("-5")
        assert t == Number(-5)

    def test_parse_float(self):
        t = parse_term("3.14")
        assert t == Number(3.14)

    def test_parse_variable(self):
        t = parse_term("X")
        assert isinstance(t, Var)
        assert t.name == 'X'

    def test_parse_compound(self):
        t = parse_term("f(a, b)")
        assert isinstance(t, Compound)
        assert t.functor == 'f'
        assert t.args == (Atom('a'), Atom('b'))

    def test_parse_nested_compound(self):
        t = parse_term("f(g(X), h(Y, Z))")
        assert t.functor == 'f'
        assert t.args[0].functor == 'g'
        assert isinstance(t.args[0].args[0], Var)

    def test_parse_list_empty(self):
        t = parse_term("[]")
        assert t == NIL

    def test_parse_list(self):
        t = parse_term("[1, 2, 3]")
        elems = list_to_python(t)
        assert elems == [Number(1), Number(2), Number(3)]

    def test_parse_list_head_tail(self):
        t = parse_term("[H | T]")
        assert isinstance(t, Compound)
        assert t.functor == '.'
        assert isinstance(t.args[0], Var)
        assert isinstance(t.args[1], Var)

    def test_parse_clause_fact(self):
        clauses, _ = parse("parent(tom, bob).")
        assert len(clauses) == 1
        assert clauses[0].head.functor == 'parent'
        assert len(clauses[0].body) == 0

    def test_parse_clause_rule(self):
        clauses, _ = parse("grandparent(X, Z) :- parent(X, Y), parent(Y, Z).")
        assert len(clauses) == 1
        assert len(clauses[0].body) == 2

    def test_parse_arithmetic(self):
        t = parse_term("X + Y * Z")
        assert isinstance(t, Compound)
        assert t.functor == '+'

    def test_parse_is(self):
        t = parse_term("X is 2 + 3")
        assert isinstance(t, Compound)
        assert t.functor == 'is'

    def test_parse_comparison(self):
        t = parse_term("X > 5")
        assert isinstance(t, Compound)
        assert t.functor == '>'

    def test_parse_cut(self):
        clauses, _ = parse("foo(X) :- X > 0, !.")
        assert len(clauses[0].body) == 2

    def test_parse_negation(self):
        t = parse_term("\\+ member(X, L)")
        assert isinstance(t, Compound)
        assert t.functor == '\\+'

    def test_parse_quoted_atom(self):
        t = parse_term("'hello world'")
        assert isinstance(t, Atom)
        assert t.name == 'hello world'

    def test_parse_anonymous_variable(self):
        t = parse_term("f(_, _)")
        assert isinstance(t, Compound)
        # Anonymous vars get unique names
        assert t.args[0] != t.args[1]

    def test_parse_string(self):
        t = parse_term('"abc"')
        elems = list_to_python(t)
        assert elems == [Number(97), Number(98), Number(99)]

    def test_parse_univ(self):
        t = parse_term("X =.. Y")
        assert isinstance(t, Compound)
        assert t.functor == '=..'

    def test_parse_multiple_clauses(self):
        src = "a(1). a(2). a(3)."
        clauses, _ = parse(src)
        assert len(clauses) == 3

    def test_parse_query(self):
        _, queries = parse("?- parent(tom, X).")
        assert len(queries) == 1

    def test_parse_comment(self):
        clauses, _ = parse("% this is a comment\nfoo(a).")
        assert len(clauses) == 1


# ============================================================
# Arithmetic
# ============================================================

class TestArithmetic:
    def test_eval_number(self):
        assert eval_arith(Number(5), Substitution()) == 5

    def test_eval_addition(self):
        expr = Compound('+', [Number(3), Number(4)])
        assert eval_arith(expr, Substitution()) == 7

    def test_eval_subtraction(self):
        expr = Compound('-', [Number(10), Number(3)])
        assert eval_arith(expr, Substitution()) == 7

    def test_eval_multiplication(self):
        expr = Compound('*', [Number(3), Number(4)])
        assert eval_arith(expr, Substitution()) == 12

    def test_eval_division(self):
        expr = Compound('/', [Number(10), Number(2)])
        assert eval_arith(expr, Substitution()) == 5.0

    def test_eval_integer_division(self):
        expr = Compound('//', [Number(7), Number(2)])
        assert eval_arith(expr, Substitution()) == 3

    def test_eval_mod(self):
        expr = Compound('mod', [Number(7), Number(3)])
        assert eval_arith(expr, Substitution()) == 1

    def test_eval_nested(self):
        # (2 + 3) * 4
        expr = Compound('*', [Compound('+', [Number(2), Number(3)]), Number(4)])
        assert eval_arith(expr, Substitution()) == 20

    def test_eval_with_variable(self):
        expr = Compound('+', [Var('X'), Number(1)])
        s = Substitution({'X': Number(5)})
        assert eval_arith(expr, s) == 6

    def test_eval_unbound_variable_error(self):
        expr = Var('X')
        with pytest.raises(ValueError):
            eval_arith(expr, Substitution())

    def test_eval_abs(self):
        expr = Compound('abs', [Number(-5)])
        assert eval_arith(expr, Substitution()) == 5

    def test_eval_min_max(self):
        assert eval_arith(Compound('min', [Number(3), Number(5)]), Substitution()) == 3
        assert eval_arith(Compound('max', [Number(3), Number(5)]), Substitution()) == 5


# ============================================================
# Interpreter: Basic Queries
# ============================================================

class TestInterpreterBasic:
    def test_simple_fact(self):
        interp = Interpreter()
        interp.consult("likes(john, mary).")
        results = interp.query_string("likes(john, mary)")
        assert len(results) == 1

    def test_fact_no_match(self):
        interp = Interpreter()
        interp.consult("likes(john, mary).")
        results = interp.query_string("likes(john, bob)")
        assert len(results) == 0

    def test_variable_binding(self):
        interp = Interpreter()
        interp.consult("likes(john, mary). likes(john, pizza).")
        results = interp.query_string("likes(john, X)")
        assert len(results) == 2

    def test_rule(self):
        interp = Interpreter()
        interp.consult("""
            parent(tom, bob).
            parent(bob, ann).
            grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """)
        results = interp.query_string("grandparent(tom, ann)")
        assert len(results) == 1

    def test_multiple_solutions(self):
        interp = Interpreter()
        interp.consult("""
            color(red).
            color(green).
            color(blue).
        """)
        results = interp.query_string("color(X)")
        assert len(results) == 3

    def test_recursive_rule(self):
        interp = Interpreter()
        interp.consult("""
            parent(a, b).
            parent(b, c).
            parent(c, d).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        """)
        results = interp.query_string("ancestor(a, d)")
        assert len(results) == 1

    def test_consult_returns_query_results(self):
        interp = Interpreter()
        interp.consult("fact(a).")
        results = interp.query_string("fact(X)")
        assert len(results) == 1

    def test_true(self):
        interp = Interpreter()
        results = interp.query_string("true")
        assert len(results) == 1

    def test_fail(self):
        interp = Interpreter()
        results = interp.query_string("fail")
        assert len(results) == 0


# ============================================================
# Interpreter: Unification Built-ins
# ============================================================

class TestUnificationBuiltins:
    def test_unify_eq(self):
        interp = Interpreter()
        results = interp.query_string("X = hello")
        assert len(results) == 1

    def test_unify_neq(self):
        interp = Interpreter()
        results = interp.query_string("a \\= b")
        assert len(results) == 1

    def test_unify_neq_fail(self):
        interp = Interpreter()
        results = interp.query_string("a \\= a")
        assert len(results) == 0

    def test_structural_eq(self):
        interp = Interpreter()
        interp.consult("foo(a).")
        results = interp.query_string("X = a, X == a")
        assert len(results) == 1

    def test_structural_neq(self):
        interp = Interpreter()
        results = interp.query_string("a \\== b")
        assert len(results) == 1


# ============================================================
# Interpreter: Arithmetic
# ============================================================

class TestInterpreterArithmetic:
    def test_is(self):
        _, results = run("", "X is 2 + 3")
        assert len(results) == 1
        assert results[0]['X'] == Number(5)

    def test_is_multiplication(self):
        _, results = run("", "X is 3 * 4")
        assert results[0]['X'] == Number(12)

    def test_is_complex(self):
        _, results = run("", "X is (2 + 3) * 4")
        assert results[0]['X'] == Number(20)

    def test_arith_comparison_lt(self):
        interp = Interpreter()
        assert len(interp.query_string("3 < 5")) == 1
        assert len(interp.query_string("5 < 3")) == 0

    def test_arith_comparison_gt(self):
        interp = Interpreter()
        assert len(interp.query_string("5 > 3")) == 1
        assert len(interp.query_string("3 > 5")) == 0

    def test_arith_comparison_gte(self):
        interp = Interpreter()
        assert len(interp.query_string("5 >= 5")) == 1
        assert len(interp.query_string("5 >= 3")) == 1

    def test_arith_comparison_lte(self):
        interp = Interpreter()
        assert len(interp.query_string("3 =< 5")) == 1

    def test_arith_eq(self):
        interp = Interpreter()
        assert len(interp.query_string("5 =:= 2 + 3")) == 1
        assert len(interp.query_string("5 =:= 6")) == 0

    def test_arith_neq(self):
        interp = Interpreter()
        r = interp.query_string("3 =\\= 4")
        assert len(r) == 1

    def test_factorial(self):
        interp = Interpreter()
        interp.consult("""
            factorial(0, 1).
            factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
        """)
        results = interp.query_string("factorial(5, F)")
        assert len(results) == 1
        assert results[0].walk(Var('F')) == Number(120)

    def test_fibonacci(self):
        interp = Interpreter()
        interp.consult("""
            fib(0, 0).
            fib(1, 1).
            fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, fib(N1, F1), fib(N2, F2), F is F1 + F2.
        """)
        results = interp.query_string("fib(7, F)")
        assert len(results) == 1
        assert results[0].walk(Var('F')) == Number(13)


# ============================================================
# Interpreter: Cut
# ============================================================

class TestCut:
    def test_cut_prunes(self):
        interp = Interpreter()
        interp.consult("""
            foo(X) :- X = a, !.
            foo(X) :- X = b.
        """)
        results = interp.query_string("foo(X)")
        assert len(results) == 1
        assert results[0].walk(Var('X')) == Atom('a')

    def test_cut_in_max(self):
        interp = Interpreter()
        interp.consult("""
            mymax(X, Y, X) :- X >= Y, !.
            mymax(X, Y, Y).
        """)
        results = interp.query_string("mymax(5, 3, M)")
        assert len(results) == 1
        assert results[0].walk(Var('M')) == Number(5)
        results = interp.query_string("mymax(3, 5, M)")
        assert len(results) == 1
        assert results[0].walk(Var('M')) == Number(5)


# ============================================================
# Interpreter: Negation as Failure
# ============================================================

class TestNegation:
    def test_negation_success(self):
        interp = Interpreter()
        interp.consult("likes(john, mary).")
        results = interp.query_string("\\+ likes(john, bob)")
        assert len(results) == 1

    def test_negation_failure(self):
        interp = Interpreter()
        interp.consult("likes(john, mary).")
        results = interp.query_string("\\+ likes(john, mary)")
        assert len(results) == 0


# ============================================================
# Interpreter: Lists
# ============================================================

class TestLists:
    def test_member(self):
        interp = Interpreter()
        results = interp.query_string("member(X, [1, 2, 3])")
        assert len(results) == 3

    def test_member_specific(self):
        interp = Interpreter()
        results = interp.query_string("member(2, [1, 2, 3])")
        assert len(results) == 1

    def test_member_not_found(self):
        interp = Interpreter()
        results = interp.query_string("member(4, [1, 2, 3])")
        assert len(results) == 0

    def test_append(self):
        interp = Interpreter()
        results = interp.query_string("append([1, 2], [3, 4], X)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('X')))
        assert lst == [Number(1), Number(2), Number(3), Number(4)]

    def test_append_split(self):
        interp = Interpreter()
        results = interp.query_string("append(X, Y, [1, 2, 3])")
        assert len(results) == 4  # [], [1], [1,2], [1,2,3]

    def test_length(self):
        _, results = run("", "length([a, b, c], N)")
        assert results[0]['N'] == Number(3)

    def test_reverse(self):
        _, results = run("", "reverse([1, 2, 3], R)")
        lst = list_to_python(results[0]['R'])
        assert lst == [Number(3), Number(2), Number(1)]

    def test_sort(self):
        _, results = run("", "sort([3, 1, 2, 1], S)")
        lst = list_to_python(results[0]['S'])
        assert lst == [Number(1), Number(2), Number(3)]

    def test_msort(self):
        _, results = run("", "msort([3, 1, 2, 1], S)")
        lst = list_to_python(results[0]['S'])
        assert lst == [Number(1), Number(1), Number(2), Number(3)]

    def test_last(self):
        _, results = run("", "last([1, 2, 3], X)")
        assert results[0]['X'] == Number(3)

    def test_nth0(self):
        _, results = run("", "nth0(1, [a, b, c], X)")
        assert results[0]['X'] == Atom('b')

    def test_nth1(self):
        _, results = run("", "nth1(2, [a, b, c], X)")
        assert results[0]['X'] == Atom('b')

    def test_flatten(self):
        _, results = run("", "flatten([[1, 2], [3, [4]]], F)")
        lst = list_to_python(results[0]['F'])
        assert lst == [Number(1), Number(2), Number(3), Number(4)]

    def test_select(self):
        interp = Interpreter()
        results = interp.query_string("select(X, [a, b, c], R)")
        assert len(results) == 3

    def test_numlist(self):
        _, results = run("", "numlist(1, 5, L)")
        lst = list_to_python(results[0]['L'])
        assert lst == [Number(1), Number(2), Number(3), Number(4), Number(5)]

    def test_permutation(self):
        interp = Interpreter()
        results = interp.query_string("permutation([1, 2, 3], P)")
        assert len(results) == 6


# ============================================================
# Interpreter: findall / bagof / setof
# ============================================================

class TestCollecting:
    def test_findall(self):
        interp = Interpreter()
        interp.consult("age(peter, 7). age(ann, 11). age(pat, 8).")
        results = interp.query_string("findall(X, age(X, _), L)")
        assert len(results) == 1
        # L should be a list of 3 names

    def test_findall_empty(self):
        interp = Interpreter()
        results = interp.query_string("findall(X, fail, L)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('L')))
        assert lst == []

    def test_bagof(self):
        interp = Interpreter()
        interp.consult("age(peter, 7). age(ann, 11). age(pat, 8).")
        results = interp.query_string("bagof(X, age(X, 7), L)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('L')))
        assert lst == [Atom('peter')]

    def test_setof(self):
        interp = Interpreter()
        interp.consult("color(red). color(blue). color(red).")
        results = interp.query_string("setof(X, color(X), L)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('L')))
        # setof removes duplicates and sorts
        assert Atom('blue') in lst
        assert Atom('red') in lst
        assert len(lst) == 2

    def test_findall_with_computation(self):
        interp = Interpreter()
        interp.consult("num(1). num(2). num(3).")
        results = interp.query_string("findall(S, (num(X), S is X * X), Squares)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('Squares')))
        assert lst == [Number(1), Number(4), Number(9)]


# ============================================================
# Interpreter: Assert / Retract
# ============================================================

class TestDynamic:
    def test_assert(self):
        interp = Interpreter()
        interp.query_all(parse_query("assert(fact(hello))"))
        results = interp.query_string("fact(X)")
        assert len(results) == 1

    def test_assertz(self):
        interp = Interpreter()
        interp.consult("item(a).")
        interp.query_all(parse_query("assertz(item(b))"))
        results = interp.query_string("item(X)")
        assert len(results) == 2

    def test_asserta(self):
        interp = Interpreter()
        interp.consult("item(a).")
        interp.query_all(parse_query("asserta(item(b))"))
        results = interp.query_string("item(X)")
        assert len(results) == 2
        # b should come first
        assert results[0].walk(Var('X')) == Atom('b')

    def test_retract(self):
        interp = Interpreter()
        interp.consult("item(a). item(b). item(c).")
        interp.query_all(parse_query("retract(item(b))"))
        results = interp.query_string("item(X)")
        assert len(results) == 2

    def test_abolish(self):
        interp = Interpreter()
        interp.consult("foo(1). foo(2). foo(3).")
        interp.query_all(parse_query("abolish(foo/1)"))
        results = interp.query_string("foo(X)")
        assert len(results) == 0


# ============================================================
# Interpreter: Type Checking
# ============================================================

class TestTypeChecking:
    def test_atom_check(self):
        interp = Interpreter()
        assert len(interp.query_string("atom(hello)")) == 1
        assert len(interp.query_string("atom(42)")) == 0

    def test_number_check(self):
        interp = Interpreter()
        assert len(interp.query_string("number(42)")) == 1
        assert len(interp.query_string("number(hello)")) == 0

    def test_integer_check(self):
        interp = Interpreter()
        assert len(interp.query_string("integer(42)")) == 1

    def test_var_check(self):
        interp = Interpreter()
        assert len(interp.query_string("var(X)")) == 1
        assert len(interp.query_string("X = 1, var(X)")) == 0

    def test_nonvar_check(self):
        interp = Interpreter()
        assert len(interp.query_string("nonvar(hello)")) == 1

    def test_compound_check(self):
        interp = Interpreter()
        assert len(interp.query_string("compound(f(a))")) == 1
        assert len(interp.query_string("compound(hello)")) == 0

    def test_is_list_check(self):
        interp = Interpreter()
        assert len(interp.query_string("is_list([1, 2, 3])")) == 1
        assert len(interp.query_string("is_list([])")) == 1
        assert len(interp.query_string("is_list(hello)")) == 0

    def test_ground_check(self):
        interp = Interpreter()
        assert len(interp.query_string("ground(f(1, a))")) == 1
        assert len(interp.query_string("ground(f(X, a))")) == 0

    def test_callable_check(self):
        interp = Interpreter()
        assert len(interp.query_string("callable(foo)")) == 1
        assert len(interp.query_string("callable(f(x))")) == 1
        assert len(interp.query_string("callable(42)")) == 0


# ============================================================
# Interpreter: Term Manipulation
# ============================================================

class TestTermManipulation:
    def test_functor(self):
        _, results = run("", "functor(f(a, b), Name, Arity)")
        assert results[0]['Name'] == Atom('f')
        assert results[0]['Arity'] == Number(2)

    def test_functor_atom(self):
        _, results = run("", "functor(hello, Name, Arity)")
        assert results[0]['Name'] == Atom('hello')
        assert results[0]['Arity'] == Number(0)

    def test_functor_construct(self):
        _, results = run("", "functor(T, foo, 2)")
        assert len(results) == 1
        t = results[0]['T']
        assert isinstance(t, Compound)
        assert t.functor == 'foo'
        assert len(t.args) == 2

    def test_arg(self):
        _, results = run("", "arg(2, f(a, b, c), X)")
        assert results[0]['X'] == Atom('b')

    def test_univ(self):
        _, results = run("", "f(a, b) =.. L")
        lst = list_to_python(results[0]['L'])
        assert lst[0] == Atom('f')
        assert lst[1] == Atom('a')
        assert lst[2] == Atom('b')

    def test_univ_construct(self):
        _, results = run("", "T =.. [g, 1, 2]")
        t = results[0]['T']
        assert isinstance(t, Compound)
        assert t.functor == 'g'
        assert t.args == (Number(1), Number(2))

    def test_copy_term(self):
        _, results = run("", "copy_term(f(X, Y), Copy)")
        assert len(results) == 1
        copy = results[0]['Copy']
        assert isinstance(copy, Compound)
        assert copy.functor == 'f'
        # Variables should be different from originals


# ============================================================
# Interpreter: Control Flow
# ============================================================

class TestControlFlow:
    def test_disjunction(self):
        interp = Interpreter()
        results = interp.query_string("(X = a ; X = b)")
        assert len(results) == 2

    def test_if_then_else(self):
        interp = Interpreter()
        results = interp.query_string("(1 > 0 -> X = yes ; X = no)")
        assert len(results) == 1
        assert results[0].walk(Var('X')) == Atom('yes')

    def test_if_then_else_false(self):
        interp = Interpreter()
        results = interp.query_string("(0 > 1 -> X = yes ; X = no)")
        assert len(results) == 1
        assert results[0].walk(Var('X')) == Atom('no')

    def test_once(self):
        interp = Interpreter()
        interp.consult("a(1). a(2). a(3).")
        results = interp.query_string("once(a(X))")
        assert len(results) == 1
        assert results[0].walk(Var('X')) == Number(1)

    def test_ignore_success(self):
        interp = Interpreter()
        results = interp.query_string("ignore(true)")
        assert len(results) == 1

    def test_ignore_failure(self):
        interp = Interpreter()
        results = interp.query_string("ignore(fail)")
        assert len(results) == 1

    def test_forall(self):
        interp = Interpreter()
        interp.consult("even(2). even(4). even(6).")
        # all even numbers are > 0
        results = interp.query_string("forall(even(X), X > 0)")
        assert len(results) == 1

    def test_between(self):
        interp = Interpreter()
        results = interp.query_string("between(1, 5, X)")
        assert len(results) == 5

    def test_call(self):
        interp = Interpreter()
        interp.consult("foo(a). foo(b).")
        results = interp.query_string("call(foo, X)")
        assert len(results) == 2

    def test_call_atom(self):
        interp = Interpreter()
        results = interp.query_string("call(true)")
        assert len(results) == 1

    def test_succ(self):
        _, results = run("", "succ(3, X)")
        assert results[0]['X'] == Number(4)

    def test_succ_reverse(self):
        _, results = run("", "succ(X, 5)")
        assert results[0]['X'] == Number(4)

    def test_plus(self):
        _, results = run("", "plus(2, 3, X)")
        assert results[0]['X'] == Number(5)

    def test_plus_reverse(self):
        _, results = run("", "plus(X, 3, 5)")
        assert results[0]['X'] == Number(2)


# ============================================================
# Interpreter: I/O
# ============================================================

class TestIO:
    def test_write(self):
        interp = Interpreter()
        interp.query_all(parse_query("write(hello)"))
        assert 'hello' in ''.join(interp.output)

    def test_writeln(self):
        interp = Interpreter()
        interp.query_all(parse_query("writeln(hello)"))
        assert 'hello\n' in ''.join(interp.output)

    def test_nl(self):
        interp = Interpreter()
        interp.query_all(parse_query("nl"))
        assert '\n' in ''.join(interp.output)

    def test_write_number(self):
        interp = Interpreter()
        interp.query_all(parse_query("write(42)"))
        assert '42' in ''.join(interp.output)


# ============================================================
# Interpreter: String/Atom Operations
# ============================================================

class TestStringOps:
    def test_atom_chars(self):
        _, results = run("", "atom_chars(hello, X)")
        lst = list_to_python(results[0]['X'])
        assert lst == [Atom('h'), Atom('e'), Atom('l'), Atom('l'), Atom('o')]

    def test_atom_chars_construct(self):
        _, results = run("", "atom_chars(X, [a, b, c])")
        assert results[0]['X'] == Atom('abc')

    def test_atom_length(self):
        _, results = run("", "atom_length(hello, N)")
        assert results[0]['N'] == Number(5)

    def test_atom_concat(self):
        _, results = run("", "atom_concat(hello, world, X)")
        assert results[0]['X'] == Atom('helloworld')

    def test_char_code(self):
        _, results = run("", "char_code(a, X)")
        assert results[0]['X'] == Number(97)

    def test_char_code_reverse(self):
        _, results = run("", "char_code(X, 65)")
        assert results[0]['X'] == Atom('A')

    def test_sub_atom(self):
        interp = Interpreter()
        results = interp.query_string("sub_atom(abcdef, 2, 3, _, Sub)")
        found = False
        for r in results:
            sub = r.walk(Var('Sub'))
            if isinstance(sub, Atom) and sub.name == 'cde':
                found = True
        assert found

    def test_number_chars(self):
        _, results = run("", "number_chars(42, X)")
        lst = list_to_python(results[0]['X'])
        assert lst == [Atom('4'), Atom('2')]

    def test_atom_number(self):
        _, results = run("", "atom_number('42', X)")
        assert results[0]['X'] == Number(42)


# ============================================================
# Interpreter: Higher-Order
# ============================================================

class TestHigherOrder:
    def test_maplist(self):
        interp = Interpreter()
        interp.consult("double(X, Y) :- Y is X * 2.")
        results = interp.query_string("maplist(double, [1, 2, 3], Y)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('Y')))
        assert lst == [Number(2), Number(4), Number(6)]

    def test_include(self):
        interp = Interpreter()
        interp.consult("positive(X) :- X > 0.")
        results = interp.query_string("include(positive, [3, -1, 2, -5, 4], R)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('R')))
        assert lst == [Number(3), Number(2), Number(4)]

    def test_foldl(self):
        interp = Interpreter()
        interp.consult("add(X, Acc, R) :- R is Acc + X.")
        results = interp.query_string("foldl(add, [1, 2, 3, 4], 0, Sum)")
        assert len(results) == 1
        assert results[0].walk(Var('Sum')) == Number(10)

    def test_sum_list(self):
        _, results = run("", "sum_list([1, 2, 3, 4], S)")
        assert results[0]['S'] == Number(10)

    def test_max_list(self):
        _, results = run("", "max_list([3, 7, 1, 9, 2], M)")
        assert results[0]['M'] == Number(9)

    def test_min_list(self):
        _, results = run("", "min_list([3, 7, 1, 9, 2], M)")
        assert results[0]['M'] == Number(1)


# ============================================================
# Interpreter: Exception Handling
# ============================================================

class TestExceptions:
    def test_catch_throw(self):
        interp = Interpreter()
        interp.consult("""
            risky(X) :- X < 0, throw(error(negative, X)).
            risky(X) :- X >= 0.
        """)
        results = interp.query_string("catch(risky(-1), error(negative, V), V = -1)")
        assert len(results) == 1

    def test_throw_uncaught(self):
        interp = Interpreter()
        with pytest.raises(PrologError):
            interp.query_all(parse_query("throw(my_error)"))

    def test_catch_no_error(self):
        interp = Interpreter()
        results = interp.query_string("catch(true, _, fail)")
        assert len(results) == 1


# ============================================================
# Classic Prolog Programs
# ============================================================

class TestClassicPrograms:
    def test_natural_numbers(self):
        interp = Interpreter()
        interp.consult("""
            nat(0).
            nat(s(X)) :- nat(X).
        """)
        interp.max_depth = 20
        results = interp.query_string("nat(s(s(s(0))))")
        assert len(results) == 1

    def test_list_length(self):
        interp = Interpreter()
        interp.consult("""
            my_length([], 0).
            my_length([_|T], N) :- my_length(T, N1), N is N1 + 1.
        """)
        results = interp.query_string("my_length([a, b, c], N)")
        assert len(results) == 1
        assert results[0].walk(Var('N')) == Number(3)

    def test_quicksort(self):
        interp = Interpreter()
        interp.consult("""
            qsort([], []).
            qsort([H|T], Sorted) :-
                partition(H, T, Less, Greater),
                qsort(Less, SortedLess),
                qsort(Greater, SortedGreater),
                append(SortedLess, [H|SortedGreater], Sorted).
            partition(_, [], [], []).
            partition(Pivot, [H|T], [H|Less], Greater) :-
                H =< Pivot, partition(Pivot, T, Less, Greater).
            partition(Pivot, [H|T], Less, [H|Greater]) :-
                H > Pivot, partition(Pivot, T, Less, Greater).
        """)
        results = interp.query_string("qsort([3, 1, 4, 1, 5, 9, 2, 6], S)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('S')))
        vals = [n.value for n in lst]
        assert vals == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_hanoi(self):
        interp = Interpreter()
        interp.consult("""
            hanoi(1, From, To, _) :- write(From), write(' -> '), writeln(To).
            hanoi(N, From, To, Via) :-
                N > 1,
                N1 is N - 1,
                hanoi(N1, From, Via, To),
                write(From), write(' -> '), writeln(To),
                hanoi(N1, Via, To, From).
        """)
        results = interp.query_string("hanoi(3, left, right, center)")
        assert len(results) == 1
        assert len(interp.output) > 0

    def test_map_coloring(self):
        interp = Interpreter()
        interp.consult("""
            color(red). color(green). color(blue).
            adjacent(wa, nt). adjacent(wa, sa). adjacent(nt, sa).
            adjacent(nt, q). adjacent(sa, q). adjacent(sa, nsw).
            adjacent(sa, v). adjacent(q, nsw). adjacent(nsw, v).
            valid(X, Y) :- color(X), color(Y), X \\= Y.
            coloring(WA, NT, SA, Q, NSW, V) :-
                valid(WA, NT), valid(WA, SA), valid(NT, SA),
                valid(NT, Q), valid(SA, Q), valid(SA, NSW),
                valid(SA, V), valid(Q, NSW), valid(NSW, V).
        """)
        results = interp.query_string("coloring(WA, NT, SA, Q, NSW, V)")
        assert len(results) > 0

    def test_list_reverse_acc(self):
        interp = Interpreter()
        interp.consult("""
            rev([], Acc, Acc).
            rev([H|T], Acc, R) :- rev(T, [H|Acc], R).
            my_reverse(L, R) :- rev(L, [], R).
        """)
        results = interp.query_string("my_reverse([1, 2, 3, 4, 5], R)")
        assert len(results) == 1
        lst = list_to_python(results[0].walk(Var('R')))
        vals = [n.value for n in lst]
        assert vals == [5, 4, 3, 2, 1]

    def test_path_finding(self):
        interp = Interpreter()
        interp.consult("""
            edge(a, b). edge(b, c). edge(c, d). edge(a, d).
            path(X, X, [X]).
            path(X, Y, [X|P]) :- edge(X, Z), path(Z, Y, P).
        """)
        results = interp.query_string("path(a, d, P)")
        assert len(results) >= 1
        # At least the direct path a->d

    def test_eight_queens_4(self):
        """4-queens as a simpler version of 8-queens."""
        interp = Interpreter()
        interp.consult("""
            queens(N, Qs) :- numlist(1, N, Ns), permutation(Ns, Qs), safe(Qs).
            safe([]).
            safe([Q|Qs]) :- no_attack(Q, Qs, 1), safe(Qs).
            no_attack(_, [], _).
            no_attack(Q, [Q1|Qs], D) :-
                Q1 - Q =\\= D,
                Q - Q1 =\\= D,
                D1 is D + 1,
                no_attack(Q, Qs, D1).
        """)
        results = interp.query_string("queens(4, Q)")
        assert len(results) == 2  # 4-queens has 2 solutions


# ============================================================
# CLP(FD) Integration
# ============================================================

class TestCLP:
    def test_clp_simple(self):
        interp = Interpreter()
        interp.query_all(parse_query("clp_in(X, 1, 3)"))
        interp.query_all(parse_query("clp_in(Y, 1, 3)"))
        # X \= Y
        interp._clp_constraints.append({'type': 'alldiff', 'vars': [Var('X'), Var('Y')]})
        # This tests CLP setup; actual solving requires clp_label

    def test_clp_label(self):
        """Test basic CLP variable labeling."""
        interp = Interpreter()
        interp.consult("""
            solve(X, Y) :-
                clp_in(X, 1, 3),
                clp_in(Y, 1, 3),
                clp_all_different([X, Y]),
                clp_label([X, Y]).
        """)
        results = interp.query_string("solve(X, Y)")
        # Should find solutions where X != Y, both in 1..3
        assert len(results) > 0


# ============================================================
# Convenience API
# ============================================================

class TestConvenienceAPI:
    def test_run_basic(self):
        interp, results = run("foo(a). foo(b).", "foo(X)")
        assert len(results) == 2
        assert results[0]['X'] == Atom('a')
        assert results[1]['X'] == Atom('b')

    def test_run_no_query(self):
        interp, results = run("foo(a).")
        assert results == []

    def test_run_complex_query(self):
        interp, results = run(
            "parent(a, b). parent(b, c).",
            "parent(a, X), parent(X, Y)"
        )
        assert len(results) == 1
        assert results[0]['X'] == Atom('b')
        assert results[0]['Y'] == Atom('c')

    def test_run_arithmetic_query(self):
        _, results = run("", "X is 10 + 20")
        assert results[0]['X'] == Number(30)

    def test_run_with_underscore_vars(self):
        """Variables starting with _ should not appear in results."""
        _, results = run("p(1, a). p(2, b).", "p(X, _)")
        assert len(results) == 2
        assert 'X' in results[0]


# ============================================================
# Edge Cases and Robustness
# ============================================================

class TestEdgeCases:
    def test_empty_list_unification(self):
        _, results = run("", "X = []")
        assert results[0]['X'] == NIL

    def test_nested_list(self):
        _, results = run("", "X = [[1, 2], [3, 4]]")
        assert len(results) == 1

    def test_deep_recursion_limit(self):
        interp = Interpreter(max_depth=10)
        interp.consult("""
            loop(X) :- loop(X).
        """)
        results = interp.query_string("loop(1)")
        assert len(results) == 0  # Should terminate due to depth limit

    def test_multiple_solutions_with_backtracking(self):
        interp = Interpreter()
        interp.consult("""
            fruit(apple).
            fruit(banana).
            fruit(cherry).
            color(apple, red).
            color(banana, yellow).
            color(cherry, red).
            red_fruit(X) :- fruit(X), color(X, red).
        """)
        results = interp.query_string("red_fruit(X)")
        assert len(results) == 2

    def test_variable_renaming(self):
        """Ensure clause variables are properly renamed (standardized apart)."""
        interp = Interpreter()
        interp.consult("""
            parent(a, b).
            parent(b, c).
            gp(X, Z) :- parent(X, Y), parent(Y, Z).
        """)
        # Multiple calls shouldn't share variables
        r1 = interp.query_string("gp(a, Z)")
        r2 = interp.query_string("gp(a, Z)")
        assert len(r1) == len(r2) == 1

    def test_assert_rule(self):
        """Assert a rule, not just a fact."""
        interp = Interpreter()
        # We need to use the Prolog representation for rules
        interp.consult("""
            my_double(X, Y) :- Y is X * 2.
        """)
        # Now assert a new rule
        interp.query_all(parse_query("assert((triple(X, Y) :- Y is X * 3))"))
        results = interp.query_string("triple(4, Y)")
        assert len(results) == 1
        assert results[0].walk(Var('Y')) == Number(12)

    def test_conjunction_in_body(self):
        interp = Interpreter()
        interp.consult("""
            check(X) :- X > 0, X < 10.
        """)
        assert len(interp.query_string("check(5)")) == 1
        assert len(interp.query_string("check(15)")) == 0

    def test_apply(self):
        interp = Interpreter()
        interp.consult("greet(X) :- write(X).")
        results = interp.query_string("apply(greet, [hello])")
        assert len(results) == 1

    def test_term_to_atom(self):
        _, results = run("", "term_to_atom(f(1, 2), A)")
        assert len(results) == 1

    def test_number_zero(self):
        _, results = run("", "X is 0")
        assert results[0]['X'] == Number(0)

    def test_negative_arithmetic(self):
        _, results = run("", "X is 3 - 5")
        assert results[0]['X'] == Number(-2)


# ============================================================
# Peano Arithmetic
# ============================================================

class TestPeano:
    def test_peano_add(self):
        interp = Interpreter()
        interp.consult("""
            add(0, X, X).
            add(s(X), Y, s(Z)) :- add(X, Y, Z).
        """)
        results = interp.query_string("add(s(s(0)), s(s(s(0))), R)")
        assert len(results) == 1
        # R should be s(s(s(s(s(0))))) = 5 in Peano

    def test_peano_multiply(self):
        interp = Interpreter()
        interp.consult("""
            add(0, X, X).
            add(s(X), Y, s(Z)) :- add(X, Y, Z).
            mul(0, _, 0).
            mul(s(X), Y, Z) :- mul(X, Y, W), add(W, Y, Z).
        """)
        results = interp.query_string("mul(s(s(0)), s(s(s(0))), R)")
        assert len(results) == 1
        # 2 * 3 = 6 in Peano


# ============================================================
# Logic Puzzles
# ============================================================

class TestLogicPuzzles:
    def test_family_relations(self):
        interp = Interpreter()
        interp.consult("""
            male(tom). male(bob). male(jim).
            female(ann). female(pat).
            parent(tom, bob). parent(tom, ann).
            parent(bob, jim). parent(bob, pat).
            father(X, Y) :- parent(X, Y), male(X).
            mother(X, Y) :- parent(X, Y), female(X).
            sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \\= Y.
        """)
        results = interp.query_string("father(tom, X)")
        assert len(results) == 2
        results = interp.query_string("sibling(bob, ann)")
        assert len(results) == 1
        results = interp.query_string("sibling(jim, pat)")
        assert len(results) == 1

    def test_zebra_puzzle_simplified(self):
        """Simplified version of the zebra puzzle."""
        interp = Interpreter()
        interp.consult("""
            next_to(X, Y, List) :- append(_, [X, Y|_], List).
            next_to(X, Y, List) :- append(_, [Y, X|_], List).

            puzzle(Houses) :-
                Houses = [_, _, _],
                member(house(red, english), Houses),
                member(house(green, spanish), Houses),
                member(house(blue, japanese), Houses),
                next_to(house(red, _), house(blue, _), Houses).
        """)
        results = interp.query_string("puzzle(H)")
        assert len(results) > 0


# ============================================================
# Stress / Performance
# ============================================================

class TestPerformance:
    def test_many_facts(self):
        interp = Interpreter()
        facts = "\n".join(f"num({i})." for i in range(100))
        interp.consult(facts)
        results = interp.query_string("num(X)")
        assert len(results) == 100

    def test_deep_list(self):
        """Build and query a long list."""
        _, results = run("", "numlist(1, 50, L), length(L, N)")
        assert results[0]['N'] == Number(50)

    def test_findall_many(self):
        interp = Interpreter()
        facts = "\n".join(f"item({i})." for i in range(50))
        interp.consult(facts)
        results = interp.query_string("findall(X, item(X), L), length(L, N)")
        assert len(results) == 1
        assert results[0].walk(Var('N')) == Number(50)
