"""Tests for C070: Trait Decorators (@decorator on trait methods)."""
import pytest
from trait_decorators import lex, Parser, Compiler, VM, run, execute, ParseError


# ============================================================
# Helper
# ============================================================

def run_code(source):
    """Run source, return (result, output_lines)."""
    return run(source)

def run_output(source):
    """Run source, return joined output."""
    _, output = run(source)
    return "\n".join(output)


# ============================================================
# Basic trait method decorators
# ============================================================

class TestBasicTraitDecorators:
    def test_simple_trait_decorator(self):
        """@decorator on a trait default method."""
        code = '''
        fn logger(f) {
            fn wrapper(this) {
                print "before";
                let r = f(this);
                print "after";
                return r;
            }
            return wrapper;
        }

        trait Greetable {
            @logger
            fn greet() {
                print "hello";
                return 42;
            }
        }

        class Dog implements Greetable {}

        let d = Dog();
        let r = d.greet();
        print r;
        '''
        out = run_output(code)
        assert "before" in out
        assert "hello" in out
        assert "after" in out
        assert "42" in out

    def test_identity_decorator(self):
        """Identity decorator preserves original method."""
        code = '''
        fn identity(f) { return f; }

        trait Calculable {
            @identity
            fn calc() { return 99; }
        }

        class Calc implements Calculable {}
        let c = Calc();
        print c.calc();
        '''
        out = run_output(code)
        assert "99" in out

    def test_undecorated_trait_method_still_works(self):
        """Undecorated methods in same trait work normally."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "wrapped"; return f(this); }
            return w;
        }

        trait T {
            fn plain() { return 1; }
            @wrap
            fn fancy() { return 2; }
        }

        class C implements T {}
        let c = C();
        print c.plain();
        print c.fancy();
        '''
        out = run_output(code)
        assert "1" in out
        assert "wrapped" in out
        assert "2" in out

    def test_required_methods_still_enforced(self):
        """Required methods are still checked even with decorated methods."""
        code = '''
        fn wrap(f) { return f; }

        trait T {
            fn required();
            @wrap
            fn default_method() { return 42; }
        }

        class C implements T {}
        '''
        with pytest.raises(Exception):
            run_code(code)

    def test_required_method_cannot_be_decorated(self):
        """@decorator on required method raises ParseError."""
        code = '''
        fn wrap(f) { return f; }
        trait T {
            @wrap
            fn required();
        }
        '''
        with pytest.raises(ParseError):
            run_code(code)


# ============================================================
# Multiple decorators
# ============================================================

class TestMultipleDecorators:
    def test_multiple_decorators_on_trait_method(self):
        """Multiple @decorators applied bottom-up."""
        code = '''
        fn add_a(f) {
            fn w(this) { print "a"; return f(this); }
            return w;
        }
        fn add_b(f) {
            fn w(this) { print "b"; return f(this); }
            return w;
        }

        trait T {
            @add_a
            @add_b
            fn run() {
                print "run";
                return 1;
            }
        }

        class C implements T {}
        let c = C();
        c.run();
        '''
        out = run_output(code)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        # @add_a is outermost, @add_b is inner: a -> b -> run
        assert lines[0] == "a"
        assert lines[1] == "b"
        assert lines[2] == "run"

    def test_three_decorators(self):
        """Three decorators stacked on trait method."""
        code = '''
        fn d1(f) { fn w(this) { print "1"; return f(this); } return w; }
        fn d2(f) { fn w(this) { print "2"; return f(this); } return w; }
        fn d3(f) { fn w(this) { print "3"; return f(this); } return w; }

        trait T {
            @d1
            @d2
            @d3
            fn go() { print "go"; return 0; }
        }

        class C implements T {}
        C().go();
        '''
        out = run_output(code)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        assert lines == ["1", "2", "3", "go"]


# ============================================================
# Decorator factories (decorators with arguments)
# ============================================================

class TestDecoratorFactories:
    def test_decorator_factory(self):
        """@factory(arg) on trait method."""
        code = '''
        fn tag(name) {
            fn decorator(f) {
                fn w(this) {
                    print name;
                    return f(this);
                }
                return w;
            }
            return decorator;
        }

        trait T {
            @tag("hello")
            fn greet() { return 42; }
        }

        class C implements T {}
        print C().greet();
        '''
        out = run_output(code)
        assert "hello" in out
        assert "42" in out

    def test_decorator_factory_with_multiple_args(self):
        """@factory(a, b) on trait method."""
        code = '''
        fn prefix(a, b) {
            fn dec(f) {
                fn w(this) {
                    print a;
                    print b;
                    return f(this);
                }
                return w;
            }
            return dec;
        }

        trait T {
            @prefix("x", "y")
            fn go() { return 1; }
        }

        class C implements T {}
        C().go();
        '''
        out = run_output(code)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        assert lines[0] == "x"
        assert lines[1] == "y"


# ============================================================
# Decorator with dot access
# ============================================================

class TestDotAccessDecorators:
    def test_decorator_dot_access(self):
        """@obj.method style decorator on trait method."""
        code = '''
        let decorators = {
            "log": fn(f) {
                fn w(this) {
                    print "logged";
                    return f(this);
                }
                return w;
            }
        };

        trait T {
            @decorators.log
            fn action() { return 5; }
        }

        class C implements T {}
        print C().action();
        '''
        out = run_output(code)
        assert "logged" in out
        assert "5" in out


# ============================================================
# Trait inheritance with decorators
# ============================================================

class TestTraitInheritance:
    def test_child_trait_inherits_decorated_method(self):
        """Child trait inherits decorated method from parent."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "wrapped"; return f(this); }
            return w;
        }

        trait Base {
            @wrap
            fn base_method() { return 10; }
        }

        trait Child < Base {
            fn child_method() { return 20; }
        }

        class C implements Child {}
        let c = C();
        print c.base_method();
        print c.child_method();
        '''
        out = run_output(code)
        assert "wrapped" in out
        assert "10" in out
        assert "20" in out

    def test_child_trait_overrides_decorated_method(self):
        """Child trait can override parent's decorated method."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "wrapped"; return f(this); }
            return w;
        }

        trait Base {
            @wrap
            fn method() { return 1; }
        }

        trait Child < Base {
            fn method() { return 2; }
        }

        class C implements Child {}
        print C().method();
        '''
        out = run_output(code)
        # Child overrides -- no wrapping
        assert "2" in out

    def test_child_trait_decorates_own_methods(self):
        """Child trait can have its own decorators."""
        code = '''
        fn tag(name) {
            fn dec(f) {
                fn w(this) { print name; return f(this); }
                return w;
            }
            return dec;
        }

        trait Base {
            @tag("base")
            fn base_fn() { return 1; }
        }

        trait Child < Base {
            @tag("child")
            fn child_fn() { return 2; }
        }

        class C implements Child {}
        let c = C();
        print c.base_fn();
        print c.child_fn();
        '''
        out = run_output(code)
        assert "base" in out
        assert "1" in out
        assert "child" in out
        assert "2" in out


# ============================================================
# Class overrides decorated trait method
# ============================================================

class TestClassOverrides:
    def test_class_overrides_decorated_trait_method(self):
        """Class method takes priority over decorated trait default."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "wrapped"; return f(this); }
            return w;
        }

        trait T {
            @wrap
            fn method() { return 1; }
        }

        class C implements T {
            method() { return 99; }
        }

        print C().method();
        '''
        out = run_output(code)
        # Class own method wins -- no wrapping
        assert "99" in out
        assert "wrapped" not in out

    def test_class_uses_decorated_trait_default_when_not_overridden(self):
        """Class gets decorated trait default if it doesn't override."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "decorated"; return f(this); }
            return w;
        }

        trait T {
            @wrap
            fn method() { return 42; }
        }

        class C implements T {
            other() { return 0; }
        }

        print C().method();
        '''
        out = run_output(code)
        assert "decorated" in out
        assert "42" in out


# ============================================================
# Multiple traits with decorators
# ============================================================

class TestMultipleTraits:
    def test_two_traits_with_decorators(self):
        """Class implements two traits, each with decorated methods."""
        code = '''
        fn tag(name) {
            fn dec(f) {
                fn w(this) { print name; return f(this); }
                return w;
            }
            return dec;
        }

        trait A {
            @tag("A")
            fn method_a() { return 1; }
        }

        trait B {
            @tag("B")
            fn method_b() { return 2; }
        }

        class C implements A, B {}
        let c = C();
        print c.method_a();
        print c.method_b();
        '''
        out = run_output(code)
        assert "A" in out
        assert "1" in out
        assert "B" in out
        assert "2" in out

    def test_mixed_decorated_and_plain_across_traits(self):
        """One trait decorated, one plain."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "wrap"; return f(this); }
            return w;
        }

        trait Decorated {
            @wrap
            fn dec_method() { return 10; }
        }

        trait Plain {
            fn plain_method() { return 20; }
        }

        class C implements Decorated, Plain {}
        let c = C();
        print c.dec_method();
        print c.plain_method();
        '''
        out = run_output(code)
        assert "wrap" in out
        assert "10" in out
        assert "20" in out


# ============================================================
# Decorator interacts with this/instance state
# ============================================================

class TestThisAccess:
    def test_decorated_trait_method_accesses_this(self):
        """Decorated trait method can access instance properties."""
        code = '''
        fn wrap(f) {
            fn w(this) {
                print "before";
                return f(this);
            }
            return w;
        }

        trait Named {
            @wrap
            fn get_name() {
                return this.name;
            }
        }

        class Animal implements Named {
            init(n) { this.name = n; }
        }

        let a = Animal("Rex");
        print a.get_name();
        '''
        out = run_output(code)
        assert "before" in out
        assert "Rex" in out

    def test_decorator_reads_this_property(self):
        """Decorator wrapper can read this properties."""
        code = '''
        fn check_name(f) {
            fn w(this) {
                if (this.name == "admin") {
                    print "admin access";
                }
                return f(this);
            }
            return w;
        }

        trait Securable {
            @check_name
            fn action() { return "done"; }
        }

        class User implements Securable {
            init(n) { this.name = n; }
        }

        let u = User("admin");
        print u.action();
        '''
        out = run_output(code)
        assert "admin access" in out
        assert "done" in out


# ============================================================
# Decorator with method parameters
# ============================================================

class TestMethodParams:
    def test_decorated_trait_method_with_params(self):
        """Decorated method with parameters."""
        code = '''
        fn wrap(f) {
            fn w(this, x) {
                print "pre";
                return f(this, x);
            }
            return w;
        }

        trait Addable {
            @wrap
            fn add(x) { return x + 1; }
        }

        class N implements Addable {}
        print N().add(10);
        '''
        out = run_output(code)
        assert "pre" in out
        assert "11" in out

    def test_decorated_trait_method_with_multiple_params(self):
        """Decorated method with multiple parameters."""
        code = '''
        fn wrap(f) {
            fn w(this, a, b) {
                print "call";
                return f(this, a, b);
            }
            return w;
        }

        trait Math {
            @wrap
            fn add(a, b) { return a + b; }
        }

        class Calc implements Math {}
        print Calc().add(3, 4);
        '''
        out = run_output(code)
        assert "call" in out
        assert "7" in out


# ============================================================
# Export decorated trait
# ============================================================

class TestExportTrait:
    def test_export_trait_with_decorated_method(self):
        """export trait with decorated methods compiles correctly."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "exp"; return f(this); }
            return w;
        }

        export trait T {
            @wrap
            fn method() { return 1; }
        }

        class C implements T {}
        print C().method();
        '''
        out = run_output(code)
        assert "exp" in out
        assert "1" in out


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_all_methods_decorated(self):
        """Every default method in trait is decorated."""
        code = '''
        fn d(f) {
            fn w(this) { print "d"; return f(this); }
            return w;
        }

        trait T {
            @d
            fn a() { return 1; }
            @d
            fn b() { return 2; }
            @d
            fn c() { return 3; }
        }

        class C implements T {}
        let c = C();
        print c.a();
        print c.b();
        print c.c();
        '''
        out = run_output(code)
        assert out.count("d") == 3

    def test_decorator_returns_different_fn(self):
        """Decorator replaces method entirely."""
        code = '''
        fn always_zero(f) {
            fn w(this) { return 0; }
            return w;
        }

        trait T {
            @always_zero
            fn method() { return 999; }
        }

        class C implements T {}
        print C().method();
        '''
        out = run_output(code)
        assert "0" in out
        assert "999" not in out

    def test_empty_trait_with_no_methods(self):
        """Empty trait still works (no decorators to apply)."""
        code = '''
        trait Empty {}
        class C implements Empty {}
        print 1;
        '''
        out = run_output(code)
        assert "1" in out

    def test_trait_with_only_required_methods(self):
        """Trait with only required methods (no decorators possible)."""
        code = '''
        trait T {
            fn required();
        }

        class C implements T {
            required() { return 42; }
        }

        print C().required();
        '''
        out = run_output(code)
        assert "42" in out

    def test_decorated_trait_method_works_on_instance(self):
        """Decorated trait method works when called on instance."""
        code = '''
        fn wrap(f) { return f; }

        trait T {
            @wrap
            fn method() { return 1; }
        }

        class C implements T {}
        let c = C();
        print c.method();
        '''
        out = run_output(code)
        assert "1" in out

    def test_mixed_required_and_decorated_methods(self):
        """Trait with both required and decorated default methods."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "w"; return f(this); }
            return w;
        }

        trait T {
            fn required_method();
            @wrap
            fn default_method() { return 10; }
        }

        class C implements T {
            required_method() { return 5; }
        }

        let c = C();
        print c.required_method();
        print c.default_method();
        '''
        out = run_output(code)
        assert "5" in out
        assert "w" in out
        assert "10" in out


# ============================================================
# Decorator ordering verification
# ============================================================

class TestDecoratorOrdering:
    def test_bottom_up_application(self):
        """Decorators apply bottom-up (closest to fn applied first)."""
        code = '''
        fn first(f) {
            fn w(this) {
                print "first-before";
                let r = f(this);
                print "first-after";
                return r;
            }
            return w;
        }
        fn second(f) {
            fn w(this) {
                print "second-before";
                let r = f(this);
                print "second-after";
                return r;
            }
            return w;
        }

        trait T {
            @first
            @second
            fn method() {
                print "body";
                return 0;
            }
        }

        class C implements T {}
        C().method();
        '''
        out = run_output(code)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        assert lines == ["first-before", "second-before", "body",
                         "second-after", "first-after"]


# ============================================================
# Class method decorators still work (regression)
# ============================================================

class TestClassDecoratorRegression:
    def test_class_method_decorator_still_works(self):
        """Class method decorators (C069) still function correctly."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "class-wrap"; return f(this); }
            return w;
        }

        class C {
            @wrap
            method() { return 77; }
        }

        print C().method();
        '''
        out = run_output(code)
        assert "class-wrap" in out
        assert "77" in out

    def test_both_trait_and_class_decorators(self):
        """Both trait and class can use decorators in same program."""
        code = '''
        fn t_wrap(f) {
            fn w(this) { print "trait"; return f(this); }
            return w;
        }
        fn c_wrap(f) {
            fn w(this) { print "class"; return f(this); }
            return w;
        }

        trait T {
            @t_wrap
            fn trait_method() { return 1; }
        }

        class C implements T {
            @c_wrap
            class_method() { return 2; }
        }

        let c = C();
        print c.trait_method();
        print c.class_method();
        '''
        out = run_output(code)
        assert "trait" in out
        assert "1" in out
        assert "class" in out
        assert "2" in out


# ============================================================
# Complex composition
# ============================================================

class TestComplexComposition:
    def test_decorator_with_closure_state(self):
        """Decorator that uses closure for counting."""
        code = '''
        fn counter(f) {
            let count = [0];
            fn w(this) {
                count[0] = count[0] + 1;
                print count[0];
                return f(this);
            }
            return w;
        }

        trait T {
            @counter
            fn tick() { return "ok"; }
        }

        class C implements T {}
        let c = C();
        c.tick();
        c.tick();
        c.tick();
        '''
        out = run_output(code)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        assert "1" in lines
        assert "2" in lines
        assert "3" in lines

    def test_trait_decorator_with_class_inheritance(self):
        """Decorated trait method works through class inheritance."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "wrap"; return f(this); }
            return w;
        }

        trait T {
            @wrap
            fn method() { return 42; }
        }

        class Base implements T {}
        class Child < Base {}

        print Child().method();
        '''
        out = run_output(code)
        assert "wrap" in out
        assert "42" in out

    def test_multiple_classes_same_decorated_trait(self):
        """Multiple classes implementing same decorated trait."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "w"; return f(this); }
            return w;
        }

        trait T {
            @wrap
            fn method() { return 1; }
        }

        class A implements T {}
        class B implements T {}

        print A().method();
        print B().method();
        '''
        out = run_output(code)
        assert out.count("w") == 2

    def test_decorator_factory_and_trait_inheritance(self):
        """Decorator factory on child trait method with parent trait."""
        code = '''
        fn tag(label) {
            fn dec(f) {
                fn w(this) { print label; return f(this); }
                return w;
            }
            return dec;
        }

        trait Base {
            fn base_method() { return 1; }
        }

        trait Child < Base {
            @tag("child-dec")
            fn child_method() { return 2; }
        }

        class C implements Child {}
        let c = C();
        print c.base_method();
        print c.child_method();
        '''
        out = run_output(code)
        assert "1" in out
        assert "child-dec" in out
        assert "2" in out


# ============================================================
# Static/getter/setter in traits (not supported yet, but
# ensure decorated regular methods don't break anything)
# ============================================================

class TestTraitMethodKinds:
    def test_decorated_method_without_fn_keyword(self):
        """Trait method can omit fn keyword (C067 feature)."""
        code = '''
        fn wrap(f) {
            fn w(this) { print "ok"; return f(this); }
            return w;
        }

        trait T {
            @wrap
            method() { return 1; }
        }

        class C implements T {}
        print C().method();
        '''
        out = run_output(code)
        assert "ok" in out
        assert "1" in out


# ============================================================
# Error cases
# ============================================================

class TestErrors:
    def test_decorator_on_required_method_parse_error(self):
        """Decorator on required (abstract) trait method is a parse error."""
        code = '''
        fn wrap(f) { return f; }
        trait T {
            @wrap
            fn required();
        }
        '''
        with pytest.raises(ParseError):
            run_code(code)

    def test_decorator_expression_evaluated_at_trait_creation(self):
        """Decorator expression is evaluated when trait is defined."""
        code = '''
        let calls = [0];
        fn counting_decorator(f) {
            calls[0] = calls[0] + 1;
            return f;
        }

        trait T {
            @counting_decorator
            fn method() { return 1; }
        }

        print calls[0];
        class C implements T {}
        print calls[0];
        '''
        out = run_output(code)
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        # Decorator is called once at trait creation, not at class creation
        assert lines[0] == "1"
        assert lines[1] == "1"
