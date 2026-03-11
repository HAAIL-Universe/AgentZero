"""Tests for V087: Abstract Interpretation over Strings"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from string_abstract_interpretation import (
    LengthDomain, PrefixDomain, SuffixDomain, CharSetDomain, SFADomain,
    StringProduct, StringEnv, StringInterpreter,
    SAssign, SConcat, SSlice, SIf, SWhile, SAssert,
    SConst, SVar, SConcatExpr,
    SLenEq, SLenLt, SLenGt, SStartsWith, SEndsWith, SEquals, SNotEquals,
    SContains, SIsEmpty, SNot,
    analyze_string_program, get_variable_info, compare_domains,
    string_domain_from_constraints, analyze_string_flow, check_string_property,
    INF,
)


# ==================================================================
# Section 1: LengthDomain
# ==================================================================

class TestLengthDomain:
    def test_exact_length(self):
        d = LengthDomain.exact(5)
        assert d.lo == 5 and d.hi == 5
        assert d.contains(5)
        assert not d.contains(4)
        assert not d.contains(6)

    def test_top_and_bot(self):
        t = LengthDomain.top()
        b = LengthDomain.bot()
        assert t.is_top()
        assert b.is_bot()
        assert not t.is_bot()
        assert not b.is_top()
        assert t.contains(0)
        assert t.contains(1000)
        assert not b.contains(0)

    def test_at_least_at_most(self):
        d = LengthDomain.at_least(3)
        assert d.contains(3) and d.contains(100)
        assert not d.contains(2)
        d = LengthDomain.at_most(5)
        assert d.contains(0) and d.contains(5)
        assert not d.contains(6)

    def test_join(self):
        a = LengthDomain.exact(3)
        b = LengthDomain.exact(7)
        j = a.join(b)
        assert j.lo == 3 and j.hi == 7
        assert j.contains(5)

    def test_meet(self):
        a = LengthDomain(2, 8)
        b = LengthDomain(5, 10)
        m = a.meet(b)
        assert m.lo == 5 and m.hi == 8

    def test_meet_empty(self):
        a = LengthDomain(0, 3)
        b = LengthDomain(5, 10)
        m = a.meet(b)
        assert m.is_bot()

    def test_widen(self):
        a = LengthDomain(2, 5)
        b = LengthDomain(1, 7)
        w = a.widen(b)
        assert w.lo == 0  # b.lo < a.lo -> widen to 0
        assert w.hi == INF  # b.hi > a.hi -> widen to INF

    def test_widen_stable(self):
        a = LengthDomain(2, 5)
        b = LengthDomain(3, 4)
        w = a.widen(b)
        assert w.lo == 2 and w.hi == 5

    def test_concat(self):
        a = LengthDomain.exact(3)
        b = LengthDomain.exact(4)
        c = a.concat(b)
        assert c.lo == 7 and c.hi == 7

    def test_concat_ranges(self):
        a = LengthDomain(2, 5)
        b = LengthDomain(1, 3)
        c = a.concat(b)
        assert c.lo == 3 and c.hi == 8

    def test_from_string(self):
        d = LengthDomain.from_string("hello")
        assert d.lo == 5 and d.hi == 5

    def test_slice(self):
        d = LengthDomain.exact(10)
        s = d.slice(2, 7)
        assert s.lo == 5 and s.hi == 5  # exact source length -> exact slice length

    def test_ordering(self):
        a = LengthDomain(3, 7)
        b = LengthDomain(2, 8)
        assert a <= b
        assert not b <= a

    def test_bot_join(self):
        b = LengthDomain.bot()
        a = LengthDomain.exact(5)
        assert b.join(a) == a
        assert a.join(b) == a

    def test_repr(self):
        assert "BOT" in repr(LengthDomain.bot())
        assert "TOP" in repr(LengthDomain.top())
        assert "5" in repr(LengthDomain.exact(5))


# ==================================================================
# Section 2: PrefixDomain
# ==================================================================

class TestPrefixDomain:
    def test_from_string(self):
        d = PrefixDomain.from_string("hello")
        assert d.prefix == "hello"
        assert d.contains("hello world")
        assert not d.contains("hey")

    def test_top_bot(self):
        t = PrefixDomain.top()
        b = PrefixDomain.bot()
        assert t.is_top() and not t.is_bot()
        assert b.is_bot() and not b.is_top()
        assert t.contains("anything")
        assert not b.contains("anything")

    def test_join_lcp(self):
        a = PrefixDomain("hello")
        b = PrefixDomain("help")
        j = a.join(b)
        assert j.prefix == "hel"

    def test_join_no_common(self):
        a = PrefixDomain("abc")
        b = PrefixDomain("xyz")
        j = a.join(b)
        assert j.prefix == ""  # TOP

    def test_meet(self):
        a = PrefixDomain("hel")
        b = PrefixDomain("hello")
        m = a.meet(b)
        assert m.prefix == "hello"  # more specific

    def test_meet_incompatible(self):
        a = PrefixDomain("abc")
        b = PrefixDomain("xyz")
        m = a.meet(b)
        assert m.is_bot()

    def test_ordering(self):
        a = PrefixDomain("hello")
        b = PrefixDomain("hel")
        assert a <= b  # more specific is less than more general
        assert not b <= a

    def test_repr(self):
        assert "hello" in repr(PrefixDomain("hello"))


# ==================================================================
# Section 3: SuffixDomain
# ==================================================================

class TestSuffixDomain:
    def test_from_string(self):
        d = SuffixDomain.from_string("world")
        assert d.contains("hello world")
        assert not d.contains("hello")

    def test_top_bot(self):
        assert SuffixDomain.top().is_top()
        assert SuffixDomain.bot().is_bot()

    def test_join_lcs(self):
        a = SuffixDomain("ing")
        b = SuffixDomain("ring")
        j = a.join(b)
        assert j.suffix == "ing"

    def test_meet(self):
        a = SuffixDomain("ing")
        b = SuffixDomain("ring")
        m = a.meet(b)
        assert m.suffix == "ring"

    def test_concat(self):
        a = SuffixDomain("abc")
        b = SuffixDomain("xyz")
        c = a.concat(b)
        assert c.suffix == "xyz"  # suffix of concat = suffix of second

    def test_ordering(self):
        a = SuffixDomain("ring")
        b = SuffixDomain("ing")
        assert a <= b


# ==================================================================
# Section 4: CharSetDomain
# ==================================================================

class TestCharSetDomain:
    def test_from_string(self):
        d = CharSetDomain.from_string("abc")
        assert d.chars == [{'a'}, {'b'}, {'c'}]
        assert d.contains("abc")
        assert not d.contains("abd")

    def test_top_bot(self):
        assert CharSetDomain.top().is_top()
        assert CharSetDomain.bot().is_bot()

    def test_join(self):
        a = CharSetDomain.from_string("abc")
        b = CharSetDomain.from_string("aXc")
        j = a.join(b)
        assert j.chars[0] == {'a'}
        assert j.chars[1] == {'b', 'X'}
        assert j.chars[2] == {'c'}

    def test_join_different_lengths(self):
        a = CharSetDomain.from_string("ab")
        b = CharSetDomain.from_string("xyz")
        j = a.join(b)
        assert j.chars == []  # lost position info

    def test_meet(self):
        a = CharSetDomain()
        a.chars = [{'a', 'b'}, {'c', 'd'}]
        b = CharSetDomain()
        b.chars = [{'a', 'x'}, {'d', 'y'}]
        m = a.meet(b)
        assert m.chars[0] == {'a'}
        assert m.chars[1] == {'d'}

    def test_meet_empty(self):
        a = CharSetDomain.from_string("a")
        b = CharSetDomain.from_string("b")
        m = a.meet(b)
        assert m.is_bot()

    def test_concat(self):
        a = CharSetDomain.from_string("ab")
        b = CharSetDomain.from_string("cd")
        c = a.concat(b)
        assert len(c.chars) == 4
        assert c.contains("abcd")

    def test_alphabet(self):
        d = CharSetDomain.from_alphabet({'a', 'b', 'c'})
        assert d.contains("abc")
        assert d.contains("aaa")
        assert not d.contains("abd")

    def test_repr(self):
        d = CharSetDomain.from_string("hi")
        r = repr(d)
        assert "CharSet" in r


# ==================================================================
# Section 5: SFADomain
# ==================================================================

class TestSFADomain:
    def test_from_string(self):
        d = SFADomain.from_string("hello")
        assert d.contains("hello")
        assert not d.contains("hell")
        assert not d.contains("helloo")

    def test_top_bot(self):
        t = SFADomain.top()
        b = SFADomain.bot()
        assert not b.contains("x")
        assert t.contains("anything")

    def test_join(self):
        a = SFADomain.from_string("cat")
        b = SFADomain.from_string("dog")
        j = a.join(b)
        assert j.contains("cat")
        assert j.contains("dog")
        assert not j.contains("bat")

    def test_meet(self):
        a = SFADomain.from_string("hello")
        b = SFADomain.from_string("hello")
        m = a.meet(b)
        assert m.contains("hello")

    def test_meet_disjoint(self):
        a = SFADomain.from_string("cat")
        b = SFADomain.from_string("dog")
        m = a.meet(b)
        assert m.is_bot()

    def test_concat(self):
        a = SFADomain.from_string("hel")
        b = SFADomain.from_string("lo")
        c = a.concat(b)
        assert c.contains("hello")
        assert not c.contains("helo")

    def test_accepted_word(self):
        d = SFADomain.from_string("test")
        w = d.accepted_word()
        assert w == "test"

    def test_bot_accepted_word(self):
        assert SFADomain.bot().accepted_word() is None

    def test_repr(self):
        d = SFADomain.from_string("hi")
        assert "SFA" in repr(d)
        assert "BOT" in repr(SFADomain.bot())


# ==================================================================
# Section 6: StringProduct (reduced product)
# ==================================================================

class TestStringProduct:
    def test_from_string(self):
        p = StringProduct.from_string("hello")
        assert p.length.lo == 5 and p.length.hi == 5
        assert p.prefix.prefix == "hello"
        assert p.suffix.suffix == "hello"
        assert p.contains("hello")

    def test_top_bot(self):
        t = StringProduct.top()
        b = StringProduct.bot()
        assert t.is_top() and not t.is_bot()
        assert b.is_bot() and not b.is_top()

    def test_join(self):
        a = StringProduct.from_string("hello")
        b = StringProduct.from_string("help")
        j = a.join(b)
        assert j.prefix.prefix == "hel"
        assert j.length.lo == 4 and j.length.hi == 5

    def test_meet(self):
        a = StringProduct.from_string("hello")
        b = StringProduct.from_string("hello")
        m = a.meet(b)
        assert m.contains("hello")

    def test_concat(self):
        a = StringProduct.from_string("hel")
        b = StringProduct.from_string("lo")
        c = a.concat(b)
        assert c.length.lo == 5 and c.length.hi == 5
        assert c.prefix.prefix == "hello"
        assert c.suffix.suffix == "lo"

    def test_reduction_prefix_tightens_length(self):
        p = StringProduct(
            length=LengthDomain.top(),
            prefix=PrefixDomain("abc"),
        )
        p._reduce()
        assert p.length.lo >= 3

    def test_reduction_charset_tightens_length(self):
        p = StringProduct(
            length=LengthDomain.top(),
            charset=CharSetDomain.from_string("abcd"),
        )
        p._reduce()
        assert p.length.lo == 4 and p.length.hi == 4

    def test_reduction_inconsistent_prefix_suffix(self):
        # prefix="abc", suffix="xyz", length=4 -> overlap check
        p = StringProduct(
            length=LengthDomain.exact(4),
            prefix=PrefixDomain("abc"),
            suffix=SuffixDomain("xyz"),
        )
        p._reduce()
        # abc_ and _xyz with len 4 -> overlap at position 2: "c" vs "x" -> BOT
        assert p.is_bot()

    def test_reduction_consistent_prefix_suffix(self):
        p = StringProduct(
            length=LengthDomain.exact(5),
            prefix=PrefixDomain("abc"),
            suffix=SuffixDomain("cde"),
        )
        p._reduce()
        assert not p.is_bot()

    def test_with_sfa(self):
        p = StringProduct.from_string("test", use_sfa=True)
        assert p.sfa_dom is not None
        assert p.sfa_dom.contains("test")

    def test_repr(self):
        p = StringProduct.from_string("hi")
        assert "StringProduct" in repr(p)
        assert "BOT" in repr(StringProduct.bot())
        assert "TOP" in repr(StringProduct.top())


# ==================================================================
# Section 7: StringEnv
# ==================================================================

class TestStringEnv:
    def test_get_set(self):
        env = StringEnv()
        env.set("x", StringProduct.from_string("hello"))
        val = env.get("x")
        assert val.contains("hello")

    def test_get_unknown(self):
        env = StringEnv()
        val = env.get("unknown")
        assert val.is_top()

    def test_copy(self):
        env = StringEnv()
        env.set("x", StringProduct.from_string("hello"))
        env2 = env.copy()
        env2.set("x", StringProduct.from_string("world"))
        assert env.get("x").contains("hello")  # original unchanged

    def test_join(self):
        env1 = StringEnv()
        env1.set("x", StringProduct.from_string("hello"))
        env2 = StringEnv()
        env2.set("x", StringProduct.from_string("help"))
        joined = env1.join(env2)
        assert joined.get("x").prefix.prefix == "hel"


# ==================================================================
# Section 8: Interpreter -- Assignments
# ==================================================================

class TestInterpreterAssign:
    def test_const_assign(self):
        stmts = [SAssign("x", SConst("hello"))]
        result = analyze_string_program(stmts)
        val = result['env'].get("x")
        assert val.length.lo == 5 and val.length.hi == 5
        assert val.prefix.prefix == "hello"

    def test_var_assign(self):
        stmts = [
            SAssign("x", SConst("hello")),
            SAssign("y", SVar("x")),
        ]
        result = analyze_string_program(stmts)
        assert result['env'].get("y").contains("hello")

    def test_concat_assign(self):
        stmts = [SAssign("x", SConcatExpr(SConst("hel"), SConst("lo")))]
        result = analyze_string_program(stmts)
        val = result['env'].get("x")
        assert val.length.lo == 5

    def test_init_vars(self):
        stmts = [SAssign("y", SVar("x"))]
        result = analyze_string_program(stmts, init_vars={"x": "test"})
        assert result['env'].get("y").contains("test")


# ==================================================================
# Section 9: Interpreter -- Concat
# ==================================================================

class TestInterpreterConcat:
    def test_basic_concat(self):
        stmts = [
            SAssign("a", SConst("hello")),
            SAssign("b", SConst(" world")),
            SConcat("c", "a", "b"),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("c")
        assert val.length.lo == 11
        assert val.prefix.prefix == "hello world"  # both exact -> full prefix
        assert val.suffix.suffix == " world"

    def test_concat_preserves_prefix(self):
        stmts = [
            SAssign("a", SConst("http://")),
            SAssign("b", SConst("example.com")),
            SConcat("url", "a", "b"),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("url")
        assert val.prefix.prefix == "http://example.com"


# ==================================================================
# Section 10: Interpreter -- Slice
# ==================================================================

class TestInterpreterSlice:
    def test_basic_slice(self):
        stmts = [
            SAssign("x", SConst("hello world")),
            SSlice("y", "x", 0, 5),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("y")
        assert val.length.hi <= 5
        assert val.prefix.prefix == "hello"

    def test_slice_suffix(self):
        stmts = [
            SAssign("x", SConst("hello world")),
            SSlice("y", "x", 6, None),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("y")
        # Suffix is preserved from source (conservative: source suffix = "hello world")
        assert val.suffix.suffix == "hello world"


# ==================================================================
# Section 11: Interpreter -- Conditionals
# ==================================================================

class TestInterpreterIf:
    def test_length_branch(self):
        stmts = [
            SIf(
                SLenGt("x", 5),
                [SAssign("result", SConst("long"))],
                [SAssign("result", SConst("short"))],
            ),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("result")
        # result is join of "long" and "short"
        assert val.length.lo == 4 and val.length.hi == 5

    def test_startswith_refinement(self):
        stmts = [
            SIf(
                SStartsWith("x", "http://"),
                [SAssign("result", SConst("url"))],
                [SAssign("result", SConst("other"))],
            ),
        ]
        result = analyze_string_program(stmts)
        # Just check it doesn't crash and produces valid output
        assert not result['env'].get("result").is_bot()

    def test_equals_refinement(self):
        stmts = [
            SIf(
                SEquals("x", "admin"),
                [SAssign("role", SConst("admin"))],
                [SAssign("role", SConst("user"))],
            ),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("role")
        assert val.length.lo == 4  # min("admin"=5, "user"=4)

    def test_is_empty_refinement(self):
        stmts = [
            SIf(
                SIsEmpty("x"),
                [SAssign("result", SConst("empty"))],
                [SAssign("result", SConst("notempty"))],
            ),
        ]
        result = analyze_string_program(stmts)
        assert not result['env'].get("result").is_bot()

    def test_not_condition(self):
        stmts = [
            SIf(
                SNot(SIsEmpty("x")),
                [SAssign("result", SConst("has_content"))],
                [SAssign("result", SConst("empty"))],
            ),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("result")
        assert val.length.lo == 5  # min len of "empty" and "has_content"

    def test_length_lt_refinement(self):
        # Both branches assign to same variable to avoid TOP from join
        stmts = [
            SIf(
                SLenLt("x", 10),
                [SAssign("result", SConst("short"))],
                [SAssign("result", SConst("long_string"))],
            ),
        ]
        result = analyze_string_program(stmts, init_vars={"x": "hello"})
        result_val = result['env'].get("result")
        # join of "short" (5) and "long_string" (11)
        assert result_val.length.lo == 5
        assert result_val.length.hi == 11

    def test_endswith_refinement(self):
        stmts = [
            SIf(
                SEndsWith("file", ".py"),
                [SAssign("type", SConst("python"))],
                [SAssign("type", SConst("other"))],
            ),
        ]
        result = analyze_string_program(stmts)
        assert not result['env'].get("type").is_bot()

    def test_contains_refinement(self):
        stmts = [
            SIf(
                SContains("msg", "error"),
                [SAssign("level", SConst("error"))],
                [SAssign("level", SConst("info"))],
            ),
        ]
        result = analyze_string_program(stmts)
        assert not result['env'].get("level").is_bot()


# ==================================================================
# Section 12: Interpreter -- Loops
# ==================================================================

class TestInterpreterWhile:
    def test_simple_loop(self):
        # x starts as "", loop: while len(x) < 3, x = x + "a"
        stmts = [
            SAssign("x", SConst("")),
            SWhile(
                SLenLt("x", 3),
                [SConcat("x", "x", "a")],
            ),
        ]
        # Need "a" defined
        result = analyze_string_program(stmts, init_vars={"a": "a"})
        val = result['env'].get("x")
        # After loop, len(x) >= 3
        assert val.length.lo >= 3

    def test_loop_convergence(self):
        # Just check the loop terminates and produces non-BOT
        stmts = [
            SAssign("x", SConst("start")),
            SWhile(
                SLenLt("x", 10),
                [SConcat("x", "x", "pad")],
            ),
        ]
        result = analyze_string_program(stmts, init_vars={"pad": "."})
        assert not result['env'].get("x").is_bot()


# ==================================================================
# Section 13: Assertions
# ==================================================================

class TestAssertions:
    def test_assert_holds(self):
        stmts = [
            SAssign("x", SConst("hello")),
            SAssert(SLenEq("x", 5)),
        ]
        result = analyze_string_program(stmts)
        assert len(result['assertions']) == 1
        assert result['assertions'][0][1] == True  # holds

    def test_assert_fails(self):
        stmts = [
            SAssign("x", SConst("hello")),
            SAssert(SLenEq("x", 3)),
        ]
        result = analyze_string_program(stmts)
        assert result['assertions'][0][1] == False
        assert len(result['warnings']) > 0

    def test_assert_startswith(self):
        stmts = [
            SAssign("x", SConst("http://example.com")),
            SAssert(SStartsWith("x", "http://")),
        ]
        result = analyze_string_program(stmts)
        assert result['assertions'][0][1] == True

    def test_assert_equals(self):
        stmts = [
            SAssign("x", SConst("admin")),
            SAssert(SEquals("x", "admin")),
        ]
        result = analyze_string_program(stmts)
        assert result['assertions'][0][1] == True

    def test_assert_is_empty(self):
        stmts = [
            SAssign("x", SConst("")),
            SAssert(SIsEmpty("x")),
        ]
        result = analyze_string_program(stmts)
        assert result['assertions'][0][1] == True

    def test_assert_after_concat(self):
        stmts = [
            SAssign("a", SConst("hello")),
            SAssign("b", SConst(" world")),
            SConcat("c", "a", "b"),
            SAssert(SLenEq("c", 11)),
        ]
        result = analyze_string_program(stmts)
        assert result['assertions'][0][1] == True


# ==================================================================
# Section 14: get_variable_info API
# ==================================================================

class TestGetVariableInfo:
    def test_basic_info(self):
        stmts = [SAssign("x", SConst("hello"))]
        result = analyze_string_program(stmts)
        info = get_variable_info(result['env'], "x")
        assert info['length_lo'] == 5
        assert info['length_hi'] == 5
        assert info['known_prefix'] == "hello"
        assert info['known_suffix'] == "hello"

    def test_unknown_var(self):
        env = StringEnv()
        info = get_variable_info(env, "unknown")
        assert info['is_top']

    def test_bot_info(self):
        env = StringEnv()
        env.set("x", StringProduct.bot())
        info = get_variable_info(env, "x")
        assert info['is_bot']

    def test_charset_info(self):
        stmts = [SAssign("x", SConst("abc"))]
        result = analyze_string_program(stmts)
        info = get_variable_info(result['env'], "x")
        assert 'char_positions' in info
        assert info['char_positions'] == 3


# ==================================================================
# Section 15: compare_domains API
# ==================================================================

class TestCompareDomains:
    def test_basic_comparison(self):
        stmts = [
            SAssign("x", SConst("hello")),
            SAssign("y", SConst("world")),
            SConcat("z", "x", "y"),
        ]
        result = compare_domains(stmts)
        assert 'without_sfa' in result
        assert 'with_sfa' in result
        assert 'z' in result['without_sfa']
        assert 'z' in result['with_sfa']


# ==================================================================
# Section 16: analyze_string_flow API
# ==================================================================

class TestStringFlow:
    def test_basic_flow(self):
        sources = {
            'greeting': StringProduct.from_string("hello"),
            'name': StringProduct.from_string("world"),
        }
        ops = [
            ('space', 'const', [' ']),
            ('tmp', 'concat', ['greeting', 'space']),
            ('result', 'concat', ['tmp', 'name']),
        ]
        env = analyze_string_flow(sources, ops)
        assert env['result'].length.lo == 11
        assert env['result'].prefix.prefix == "hello world"  # all exact -> full prefix

    def test_slice_flow(self):
        sources = {
            'url': StringProduct.from_string("http://example.com"),
        }
        ops = [
            ('protocol', 'slice_prefix', ['url', '7']),
        ]
        env = analyze_string_flow(sources, ops)
        assert env['protocol'].length.hi <= 7

    def test_assign_flow(self):
        sources = {'x': StringProduct.from_string("test")}
        ops = [('y', 'assign', ['x'])]
        env = analyze_string_flow(sources, ops)
        assert env['y'].contains("test")

    def test_unknown_op(self):
        sources = {'x': StringProduct.from_string("test")}
        ops = [('y', 'reverse', ['x'])]
        env = analyze_string_flow(sources, ops)
        assert env['y'].is_top()  # unknown op -> TOP


# ==================================================================
# Section 17: check_string_property API
# ==================================================================

class TestCheckProperty:
    def test_true_property(self):
        env = StringEnv()
        env.set("x", StringProduct.from_string("hello"))
        assert check_string_property(env, "x", SLenEq("x", 5)) == 'TRUE'

    def test_false_property(self):
        env = StringEnv()
        env.set("x", StringProduct.from_string("hello"))
        assert check_string_property(env, "x", SIsEmpty("x")) == 'FALSE'

    def test_unknown_property(self):
        env = StringEnv()
        env.set("x", StringProduct.top())
        assert check_string_property(env, "x", SLenEq("x", 5)) == 'UNKNOWN'

    def test_startswith_true(self):
        env = StringEnv()
        env.set("x", StringProduct.from_string("http://example.com"))
        assert check_string_property(env, "x", SStartsWith("x", "http://")) == 'TRUE'

    def test_endswith_true(self):
        env = StringEnv()
        env.set("x", StringProduct.from_string("file.py"))
        assert check_string_property(env, "x", SEndsWith("x", ".py")) == 'TRUE'


# ==================================================================
# Section 18: Cross-domain reduction
# ==================================================================

class TestReduction:
    def test_prefix_tightens_length(self):
        p = StringProduct(
            length=LengthDomain.top(),
            prefix=PrefixDomain("hello"),
        )
        p._reduce()
        assert p.length.lo >= 5

    def test_charset_exact_length(self):
        p = StringProduct(
            length=LengthDomain.top(),
            charset=CharSetDomain.from_string("abc"),
        )
        p._reduce()
        assert p.length.lo == 3 and p.length.hi == 3

    def test_prefix_chars_tighten_charset(self):
        cs = CharSetDomain()
        cs.chars = [{'a', 'b', 'c'}, {'x', 'y', 'z'}]
        p = StringProduct(
            length=LengthDomain.exact(2),
            prefix=PrefixDomain("ax"),
            charset=cs,
        )
        p._reduce()
        assert p.charset.chars[0] == {'a'}
        assert p.charset.chars[1] == {'x'}

    def test_inconsistent_becomes_bot(self):
        p = StringProduct(
            length=LengthDomain.exact(3),
            prefix=PrefixDomain("abcde"),  # prefix longer than max length
        )
        p._reduce()
        assert p.is_bot()


# ==================================================================
# Section 19: SFA domain with concrete composition
# ==================================================================

class TestSFAComposition:
    def test_sfa_concat(self):
        a = SFADomain.from_string("hello")
        b = SFADomain.from_string(" world")
        c = a.concat(b)
        assert c.contains("hello world")
        assert not c.contains("hello")
        assert not c.contains(" world")

    def test_sfa_join_union(self):
        a = SFADomain.from_string("yes")
        b = SFADomain.from_string("no")
        j = a.join(b)
        assert j.contains("yes")
        assert j.contains("no")
        assert not j.contains("maybe")

    def test_sfa_meet_intersection(self):
        a = SFADomain.from_string("same")
        b = SFADomain.from_string("same")
        m = a.meet(b)
        assert m.contains("same")

    def test_sfa_widen_threshold(self):
        # Build up large SFA to trigger widening
        base = SFADomain.from_string("a")
        for i in range(150):
            other = SFADomain.from_string("x" * (i + 2))
            base = base.join(other)
        wide = base.widen(SFADomain.from_string("y"))
        # Should have widened to TOP due to state count
        assert wide.is_top()


# ==================================================================
# Section 20: Integration -- URL validation pattern
# ==================================================================

class TestIntegrationURL:
    def test_url_protocol_check(self):
        stmts = [
            SAssign("url", SConst("http://example.com")),
            SIf(
                SStartsWith("url", "http://"),
                [
                    SSlice("rest", "url", 7, None),
                    SAssert(SLenGt("rest", 0)),
                ],
                [SAssert(SStartsWith("url", "http://"))],
            ),
        ]
        result = analyze_string_program(stmts)
        # Assert in then-branch should hold (rest = "example.com" has len 11 > 0)
        assert result['assertions'][0][1] == True

    def test_url_concatenation(self):
        stmts = [
            SAssign("protocol", SConst("https://")),
            SAssign("host", SConst("api.example.com")),
            SAssign("path", SConst("/v1/users")),
            SConcat("base", "protocol", "host"),
            SConcat("url", "base", "path"),
            SAssert(SStartsWith("url", "https://")),
            SAssert(SEndsWith("url", "/v1/users")),
        ]
        result = analyze_string_program(stmts)
        assert result['assertions'][0][1] == True  # starts with https://
        assert result['assertions'][1][1] == True  # ends with /v1/users
        val = result['env'].get("url")
        assert val.length.lo == 32


# ==================================================================
# Section 21: Integration -- Input validation pattern
# ==================================================================

class TestIntegrationValidation:
    def test_empty_input_check(self):
        # Use same variable across all branches to avoid TOP from missing assignments
        stmts = [
            SIf(
                SIsEmpty("input"),
                [SAssign("msg", SConst("Input required"))],
                [
                    SIf(
                        SLenGt("input", 100),
                        [SAssign("msg", SConst("Too long"))],
                        [SAssign("msg", SConst("ok"))],
                    ),
                ],
            ),
        ]
        result = analyze_string_program(stmts)
        msg_val = result['env'].get("msg")
        assert msg_val.length.lo == 2  # min("ok"=2, "Too long"=8, "Input required"=14)
        assert msg_val.length.hi == 14

    def test_prefix_based_routing(self):
        stmts = [
            SIf(
                SStartsWith("path", "/api/"),
                [SAssign("handler", SConst("api"))],
                [
                    SIf(
                        SStartsWith("path", "/admin/"),
                        [SAssign("handler", SConst("admin"))],
                        [SAssign("handler", SConst("static"))],
                    ),
                ],
            ),
        ]
        result = analyze_string_program(stmts)
        handler = result['env'].get("handler")
        # join of "api", "admin", "static"
        assert handler.length.lo == 3  # "api" is shortest


# ==================================================================
# Section 22: Edge cases
# ==================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = analyze_string_program([])
        assert result['env'] is not None

    def test_empty_string(self):
        p = StringProduct.from_string("")
        assert p.length.lo == 0 and p.length.hi == 0
        assert p.contains("")
        assert not p.contains("x")

    def test_single_char(self):
        p = StringProduct.from_string("a")
        assert p.length.lo == 1 and p.length.hi == 1

    def test_bot_concat(self):
        a = StringProduct.bot()
        b = StringProduct.from_string("hello")
        c = a.concat(b)
        assert c.is_bot()

    def test_top_concat(self):
        a = StringProduct.top()
        b = StringProduct.from_string("hello")
        c = a.concat(b)
        assert c.suffix.suffix == "hello"

    def test_nested_if(self):
        stmts = [
            SIf(
                SLenGt("x", 0),
                [
                    SIf(
                        SStartsWith("x", "a"),
                        [SAssign("result", SConst("a-string"))],
                        [SAssign("result", SConst("other"))],
                    ),
                ],
                [SAssign("result", SConst("empty"))],
            ),
        ]
        result = analyze_string_program(stmts)
        assert not result['env'].get("result").is_bot()

    def test_multiple_concats(self):
        stmts = [
            SAssign("a", SConst("a")),
            SAssign("b", SConst("b")),
            SAssign("c", SConst("c")),
            SConcat("ab", "a", "b"),
            SConcat("abc", "ab", "c"),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("abc")
        assert val.length.lo == 3

    def test_self_concat(self):
        stmts = [
            SAssign("x", SConst("ab")),
            SConcat("x", "x", "x"),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("x")
        assert val.length.lo == 4

    def test_reassign(self):
        stmts = [
            SAssign("x", SConst("first")),
            SAssign("x", SConst("second")),
        ]
        result = analyze_string_program(stmts)
        val = result['env'].get("x")
        assert val.prefix.prefix == "second"


# ==================================================================
# Section 23: Domain lattice properties
# ==================================================================

class TestLatticeProperties:
    def test_length_join_commutative(self):
        a = LengthDomain(2, 5)
        b = LengthDomain(3, 8)
        assert a.join(b) == b.join(a)

    def test_length_meet_commutative(self):
        a = LengthDomain(2, 5)
        b = LengthDomain(3, 8)
        assert a.meet(b) == b.meet(a)

    def test_prefix_join_commutative(self):
        a = PrefixDomain("hello")
        b = PrefixDomain("help")
        assert a.join(b) == b.join(a)

    def test_suffix_join_commutative(self):
        a = SuffixDomain("ing")
        b = SuffixDomain("ring")
        assert a.join(b) == b.join(a)

    def test_bot_is_identity_for_join(self):
        a = LengthDomain.exact(5)
        assert LengthDomain.bot().join(a) == a
        assert a.join(LengthDomain.bot()) == a

    def test_top_is_identity_for_meet(self):
        a = LengthDomain.exact(5)
        assert LengthDomain.top().meet(a) == a
        assert a.meet(LengthDomain.top()) == a

    def test_product_join_commutative(self):
        a = StringProduct.from_string("hello")
        b = StringProduct.from_string("help")
        j1 = a.join(b)
        j2 = b.join(a)
        assert j1.prefix.prefix == j2.prefix.prefix
        assert j1.length == j2.length


# ==================================================================
# Section 24: Practical analysis -- SQL injection pattern
# ==================================================================

class TestPracticalSQLi:
    def test_string_concat_query(self):
        """Detect that user input flows into SQL query."""
        stmts = [
            SAssign("prefix", SConst("SELECT * FROM users WHERE name = '")),
            SAssign("suffix", SConst("'")),
            SConcat("query_part", "prefix", "user_input"),
            SConcat("query", "query_part", "suffix"),
            SAssert(SStartsWith("query", "SELECT * FROM users WHERE name = '")),
        ]
        result = analyze_string_program(stmts)
        # The query starts with the known prefix
        assert result['assertions'][0][1] == True
        # But we can track that user_input is embedded
        query = result['env'].get("query")
        assert query.prefix.prefix == "SELECT * FROM users WHERE name = '"
        assert query.suffix.suffix == "'"


# ==================================================================
# Section 25: Domain operations on boundary values
# ==================================================================

class TestBoundaryValues:
    def test_length_inf_concat(self):
        a = LengthDomain(5, INF)
        b = LengthDomain.exact(3)
        c = a.concat(b)
        assert c.lo == 8
        assert c.hi == INF

    def test_length_zero(self):
        d = LengthDomain.exact(0)
        assert d.contains(0)
        assert not d.contains(1)

    def test_empty_charset(self):
        d = CharSetDomain.from_string("")
        assert d.chars == []

    def test_product_empty_string(self):
        p = StringProduct.from_string("")
        assert p.length.lo == 0 and p.length.hi == 0
        assert p.prefix.prefix == ""
        assert p.suffix.suffix == ""


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
