"""Tests for C069: Method Decorators (@decorator on class methods)."""
import pytest
from method_decorators import lex, Parser, Compiler, VM, run, execute


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
# Basic method decorators
# ============================================================

class TestBasicMethodDecorators:
    def test_simple_method_decorator(self):
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

        class Greeter {
            @logger
            greet() {
                print "hello";
                return 42;
            }
        }

        let g = Greeter();
        let r = g.greet();
        print r;
        '''
        out = run_output(code)
        assert "before" in out
        assert "hello" in out
        assert "after" in out
        assert "42" in out

    def test_decorator_preserves_return_value(self):
        code = '''
        fn identity(f) {
            return f;
        }

        class Calc {
            @identity
            add(a, b) {
                return a + b;
            }
        }

        let c = Calc();
        print c.add(3, 4);
        '''
        out = run_output(code)
        assert "7" in out

    def test_multiple_decorators_on_method(self):
        code = '''
        fn dec_a(f) {
            fn wrapper(this) {
                print "A";
                return f(this);
            }
            return wrapper;
        }

        fn dec_b(f) {
            fn wrapper(this) {
                print "B";
                return f(this);
            }
            return wrapper;
        }

        class Foo {
            @dec_a
            @dec_b
            bar() {
                print "bar";
                return 1;
            }
        }

        let x = Foo();
        x.bar();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        # @dec_a @dec_b -> dec_a(dec_b(bar)) -> A first, then B, then bar
        assert lines[0] == "A"
        assert lines[1] == "B"
        assert lines[2] == "bar"

    def test_decorator_with_no_other_methods(self):
        code = '''
        fn noop(f) { return f; }

        class Single {
            @noop
            only() { return "only"; }
        }

        let s = Single();
        print s.only();
        '''
        out = run_output(code)
        assert "only" in out

    def test_undecorated_methods_still_work(self):
        code = '''
        fn tag(f) {
            fn wrapper(this) {
                print "tagged";
                return f(this);
            }
            return wrapper;
        }

        class Mixed {
            @tag
            decorated() { return "d"; }

            plain() { return "p"; }
        }

        let m = Mixed();
        m.decorated();
        print m.plain();
        '''
        out = run_output(code)
        assert "tagged" in out
        assert "p" in out


# ============================================================
# Decorator factories on methods
# ============================================================

class TestDecoratorFactories:
    def test_factory_decorator(self):
        code = '''
        fn repeat(n) {
            fn decorator(f) {
                fn wrapper(this) {
                    let i = 0;
                    let result = null;
                    while (i < n) {
                        result = f(this);
                        i = i + 1;
                    }
                    return result;
                }
                return wrapper;
            }
            return decorator;
        }

        class Printer {
            @repeat(3)
            say() {
                print "hi";
                return "done";
            }
        }

        let p = Printer();
        p.say();
        '''
        out = run_output(code)
        assert out.count("hi") == 3

    def test_factory_with_string_arg(self):
        code = '''
        fn prefix(tag) {
            fn decorator(f) {
                fn wrapper(this) {
                    print tag;
                    return f(this);
                }
                return wrapper;
            }
            return decorator;
        }

        class Logger {
            @prefix("INFO")
            log() {
                print "message";
                return null;
            }
        }

        let l = Logger();
        l.log();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "INFO"
        assert lines[1] == "message"


# ============================================================
# Dot-access decorators on methods
# ============================================================

class TestDotAccessDecorators:
    def test_dot_access_decorator(self):
        code = '''
        let decorators = {
            "log": fn(f) {
                fn wrapper(this) {
                    print "logged";
                    return f(this);
                }
                return wrapper;
            }
        };

        class Service {
            @decorators.log
            process() {
                print "processing";
                return true;
            }
        }

        let s = Service();
        s.process();
        '''
        out = run_output(code)
        assert "logged" in out
        assert "processing" in out


# ============================================================
# Static method decorators
# ============================================================

class TestStaticMethodDecorators:
    def test_decorated_static_method(self):
        code = '''
        fn trace(f) {
            fn wrapper() {
                print "trace";
                return f();
            }
            return wrapper;
        }

        class Utils {
            @trace
            static helper() {
                return 99;
            }
        }

        print Utils.helper();
        '''
        out = run_output(code)
        assert "trace" in out
        assert "99" in out

    def test_multiple_decorators_on_static(self):
        code = '''
        fn a(f) {
            fn w() { print "a"; return f(); }
            return w;
        }
        fn b(f) {
            fn w() { print "b"; return f(); }
            return w;
        }

        class X {
            @a
            @b
            static run() {
                print "run";
                return 1;
            }
        }

        X.run();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "a"
        assert lines[1] == "b"
        assert lines[2] == "run"


# ============================================================
# Getter/setter decorators
# ============================================================

class TestGetterSetterDecorators:
    def test_decorated_getter(self):
        code = '''
        fn traced(f) {
            fn wrapper(this) {
                print "get";
                return f(this);
            }
            return wrapper;
        }

        class Config {
            @traced
            get value() {
                return 42;
            }
        }

        let c = Config();
        print c.value;
        print c.value;
        '''
        out = run_output(code)
        assert out.count("get") == 2
        assert out.count("42") == 2

    def test_decorated_setter(self):
        code = '''
        fn logged(f) {
            fn wrapper(this, val) {
                print "setting";
                return f(this, val);
            }
            return wrapper;
        }

        class Account {
            init() {
                this.balance = 0;
            }

            @logged
            set amount(val) {
                this.balance = val;
            }

            get balance_val() {
                return this.balance;
            }
        }

        let a = Account();
        a.amount = 100;
        print a.balance_val;
        '''
        out = run_output(code)
        assert "setting" in out
        assert "100" in out


# ============================================================
# Decorator with method arguments
# ============================================================

class TestDecoratorWithMethodArgs:
    def test_decorator_passes_through_args(self):
        code = '''
        fn log_args(f) {
            fn wrapper(this, a, b) {
                print "args:";
                print a;
                print b;
                return f(this, a, b);
            }
            return wrapper;
        }

        class Math {
            @log_args
            add(a, b) {
                return a + b;
            }
        }

        let m = Math();
        print m.add(10, 20);
        '''
        out = run_output(code)
        assert "args:" in out
        assert "10" in out
        assert "20" in out
        assert "30" in out

    def test_decorator_modifies_args(self):
        code = '''
        fn double_args(f) {
            fn wrapper(this, a, b) {
                return f(this, a * 2, b * 2);
            }
            return wrapper;
        }

        class Calc {
            @double_args
            mul(a, b) {
                return a * b;
            }
        }

        let c = Calc();
        print c.mul(3, 4);
        '''
        out = run_output(code)
        # 3*2=6, 4*2=8, 6*8=48
        assert "48" in out


# ============================================================
# Inheritance with method decorators
# ============================================================

class TestInheritanceDecorators:
    def test_decorated_method_in_subclass(self):
        code = '''
        fn tag(f) {
            fn wrapper(this) {
                print "tagged";
                return f(this);
            }
            return wrapper;
        }

        class Base {
            base_method() { return "base"; }
        }

        class Child < Base {
            @tag
            child_method() {
                return "child";
            }
        }

        let c = Child();
        print c.base_method();
        c.child_method();
        '''
        out = run_output(code)
        assert "base" in out
        assert "tagged" in out

    def test_both_parent_and_child_decorated(self):
        code = '''
        fn wrap(f) {
            fn w(this) {
                print "wrap";
                return f(this);
            }
            return w;
        }

        class Parent {
            @wrap
            greet() { print "parent"; return 1; }
        }

        class Child < Parent {
            @wrap
            hello() { print "child"; return 2; }
        }

        let c = Child();
        c.greet();
        c.hello();
        '''
        out = run_output(code)
        assert out.count("wrap") == 2


# ============================================================
# Decorator modifying method behavior
# ============================================================

class TestDecoratorBehavior:
    def test_transform_decorator(self):
        """Decorator that transforms method return value."""
        code = '''
        fn stringify(f) {
            fn wrapper(this, n) {
                let result = f(this, n);
                return "result:" + string(result);
            }
            return wrapper;
        }

        class Math {
            @stringify
            square(n) {
                return n * n;
            }
        }

        let m = Math();
        print m.square(7);
        '''
        out = run_output(code)
        assert "result:49" in out

    def test_before_after_decorator(self):
        code = '''
        fn lifecycle(f) {
            fn wrapper(this) {
                print "start";
                let r = f(this);
                print "end";
                return r;
            }
            return wrapper;
        }

        class Worker {
            @lifecycle
            work() {
                print "working";
                return "done";
            }
        }

        let w = Worker();
        print w.work();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "start"
        assert lines[1] == "working"
        assert lines[2] == "end"
        assert lines[3] == "done"

    def test_decorator_can_skip_original(self):
        code = '''
        fn skip(f) {
            fn wrapper(this) {
                return "skipped";
            }
            return wrapper;
        }

        class Foo {
            @skip
            bar() {
                print "should not print";
                return "original";
            }
        }

        let x = Foo();
        print x.bar();
        '''
        out = run_output(code)
        assert "skipped" in out
        assert "should not print" not in out


# ============================================================
# Mixed decorated and undecorated methods
# ============================================================

class TestMixedMethods:
    def test_first_method_decorated(self):
        code = '''
        fn noop(f) { return f; }

        class A {
            @noop
            first() { return 1; }
            second() { return 2; }
            third() { return 3; }
        }

        let a = A();
        print a.first();
        print a.second();
        print a.third();
        '''
        out = run_output(code)
        assert "1" in out
        assert "2" in out
        assert "3" in out

    def test_last_method_decorated(self):
        code = '''
        fn noop(f) { return f; }

        class B {
            first() { return 1; }
            second() { return 2; }
            @noop
            third() { return 3; }
        }

        let b = B();
        print b.first();
        print b.second();
        print b.third();
        '''
        out = run_output(code)
        assert "1" in out
        assert "2" in out
        assert "3" in out

    def test_middle_method_decorated(self):
        code = '''
        fn noop(f) { return f; }

        class C {
            first() { return 1; }
            @noop
            second() { return 2; }
            third() { return 3; }
        }

        let c = C();
        print c.first();
        print c.second();
        print c.third();
        '''
        out = run_output(code)
        assert "1" in out
        assert "2" in out
        assert "3" in out

    def test_all_methods_decorated(self):
        code = '''
        fn noop(f) { return f; }

        class D {
            @noop
            first() { return 1; }
            @noop
            second() { return 2; }
            @noop
            third() { return 3; }
        }

        let d = D();
        print d.first();
        print d.second();
        print d.third();
        '''
        out = run_output(code)
        assert "1" in out
        assert "2" in out
        assert "3" in out


# ============================================================
# Constructor decorator (init)
# ============================================================

class TestConstructorDecorator:
    def test_decorated_init(self):
        code = '''
        fn log_init(f) {
            fn wrapper(this) {
                print "constructing";
                return f(this);
            }
            return wrapper;
        }

        class Widget {
            @log_init
            init() {
                this.ready = true;
            }
        }

        let w = Widget();
        print w.ready;
        '''
        out = run_output(code)
        assert "constructing" in out
        assert "true" in out


# ============================================================
# Decorator with class-level and method-level combined
# ============================================================

class TestCombinedDecorators:
    def test_class_and_method_decorators(self):
        code = '''
        fn class_dec(cls) {
            print "class decorated";
            return cls;
        }

        fn method_dec(f) {
            fn wrapper(this) {
                print "method decorated";
                return f(this);
            }
            return wrapper;
        }

        @class_dec
        class Example {
            @method_dec
            run() {
                return "running";
            }
        }

        let e = Example();
        print e.run();
        '''
        out = run_output(code)
        assert "class decorated" in out
        assert "method decorated" in out
        assert "running" in out


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_decorator_on_method_named_like_keyword_adjacent(self):
        """Methods can have names that look like keywords but are identifiers."""
        code = '''
        fn noop(f) { return f; }

        class Foo {
            @noop
            value() { return 10; }
        }

        let f = Foo();
        print f.value();
        '''
        out = run_output(code)
        assert "10" in out

    def test_decorator_factory_with_multiple_args(self):
        code = '''
        fn config(min, max) {
            fn decorator(f) {
                fn wrapper(this, val) {
                    if (val < min) { return min; }
                    if (val > max) { return max; }
                    return f(this, val);
                }
                return wrapper;
            }
            return decorator;
        }

        class Clamp {
            @config(0, 100)
            process(val) {
                return val;
            }
        }

        let c = Clamp();
        print c.process(-5);
        print c.process(50);
        print c.process(200);
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "0"
        assert lines[1] == "50"
        assert lines[2] == "100"

    def test_three_decorators_on_method(self):
        code = '''
        fn a(f) {
            fn w(this) { print "a"; return f(this); }
            return w;
        }
        fn b(f) {
            fn w(this) { print "b"; return f(this); }
            return w;
        }
        fn c(f) {
            fn w(this) { print "c"; return f(this); }
            return w;
        }

        class Foo {
            @a
            @b
            @c
            bar() { print "bar"; return 1; }
        }

        let x = Foo();
        x.bar();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines == ["a", "b", "c", "bar"]

    def test_empty_class_with_no_decorators(self):
        """Ensure classes without any methods still work."""
        code = '''
        class Empty {
            noop() { return null; }
        }
        let e = Empty();
        print e.noop();
        '''
        out = run_output(code)
        assert "null" in out

    def test_decorator_returns_different_function(self):
        code = '''
        fn replace(f) {
            fn totally_different(this, x) {
                return x * 100;
            }
            return totally_different;
        }

        class Transformer {
            @replace
            transform(x) {
                return x + 1;
            }
        }

        let t = Transformer();
        print t.transform(5);
        '''
        out = run_output(code)
        assert "500" in out


# ============================================================
# Interaction with traits
# ============================================================

class TestTraitInteraction:
    def test_decorated_method_satisfies_trait(self):
        code = '''
        fn noop(f) { return f; }

        trait Runnable {
            run();
        }

        class Runner implements Runnable {
            @noop
            run() {
                return "running";
            }
        }

        let r = Runner();
        print r.run();
        '''
        out = run_output(code)
        assert "running" in out


# ============================================================
# Interaction with export
# ============================================================

class TestExportInteraction:
    def test_export_class_with_decorated_methods(self):
        """Class with decorated methods can be exported."""
        code = '''
        fn noop(f) { return f; }

        class Exported {
            @noop
            method() { return "exported"; }
        }

        let e = Exported();
        print e.method();
        '''
        out = run_output(code)
        assert "exported" in out


# ============================================================
# Decorator with this access
# ============================================================

class TestDecoratorThisAccess:
    def test_decorator_wrapper_accesses_this(self):
        code = '''
        fn add_greeting(f) {
            fn wrapper(this) {
                this.greeted = true;
                return f(this);
            }
            return wrapper;
        }

        class Person {
            init() {
                this.greeted = false;
                this.name = "Alice";
            }

            @add_greeting
            introduce() {
                return this.name;
            }
        }

        let p = Person();
        print p.greeted;
        print p.introduce();
        print p.greeted;
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "false"
        assert lines[1] == "Alice"
        assert lines[2] == "true"

    def test_decorator_accesses_instance_properties(self):
        code = '''
        fn require_auth(f) {
            fn wrapper(this) {
                if (this.authenticated != true) {
                    return "unauthorized";
                }
                return f(this);
            }
            return wrapper;
        }

        class API {
            init() {
                this.authenticated = false;
            }

            @require_auth
            get_data() {
                return "secret data";
            }

            login() {
                this.authenticated = true;
            }
        }

        let api = API();
        print api.get_data();
        api.login();
        print api.get_data();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "unauthorized"
        assert lines[1] == "secret data"


# ============================================================
# Stress / composition tests
# ============================================================

class TestComposition:
    def test_decorator_chain_order(self):
        """Verify exact application order with 4 decorators."""
        code = '''
        fn d1(f) { fn w(this) { print 1; return f(this); } return w; }
        fn d2(f) { fn w(this) { print 2; return f(this); } return w; }
        fn d3(f) { fn w(this) { print 3; return f(this); } return w; }
        fn d4(f) { fn w(this) { print 4; return f(this); } return w; }

        class X {
            @d1
            @d2
            @d3
            @d4
            method() { print "end"; return null; }
        }

        let x = X();
        x.method();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines == ["1", "2", "3", "4", "end"]

    def test_multiple_methods_each_with_different_decorators(self):
        code = '''
        fn tag_a(f) {
            fn w(this) { print "A"; return f(this); }
            return w;
        }
        fn tag_b(f) {
            fn w(this) { print "B"; return f(this); }
            return w;
        }

        class Multi {
            @tag_a
            method_a() { print "ma"; return 1; }

            @tag_b
            method_b() { print "mb"; return 2; }
        }

        let m = Multi();
        m.method_a();
        m.method_b();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines == ["A", "ma", "B", "mb"]

    def test_decorated_method_in_complex_class(self):
        """Class with init, getters, setters, static, and decorated regular method."""
        code = '''
        fn noop(f) { return f; }

        class Complex {
            init() {
                this.x = 0;
            }

            get val() {
                return this.x;
            }

            set val(v) {
                this.x = v;
            }

            static create() {
                return Complex();
            }

            @noop
            process() {
                return this.x * 2;
            }
        }

        let c = Complex.create();
        c.val = 21;
        print c.process();
        '''
        out = run_output(code)
        assert "42" in out


# ============================================================
# Parse errors
# ============================================================

class TestParseErrors:
    def test_decorator_on_non_method(self):
        """Decorator before something that isn't a method in class body should error."""
        code = '''
        fn noop(f) { return f; }

        class Bad {
            @noop
            let x = 5;
        }
        '''
        with pytest.raises(Exception):
            run_code(code)


# ============================================================
# Regression: existing decorator tests still pass
# ============================================================

class TestExistingDecoratorRegression:
    def test_function_decorator_still_works(self):
        code = '''
        fn logger(f) {
            fn wrapper() {
                print "before";
                let r = f();
                print "after";
                return r;
            }
            return wrapper;
        }

        @logger
        fn greet() {
            print "hello";
            return 42;
        }

        let r = greet();
        print r;
        '''
        out = run_output(code)
        assert "before" in out
        assert "hello" in out
        assert "after" in out
        assert "42" in out

    def test_class_decorator_still_works(self):
        code = '''
        fn register(cls) {
            print "registered";
            return cls;
        }

        @register
        class MyClass {
            greet() { return "hi"; }
        }

        let m = MyClass();
        print m.greet();
        '''
        out = run_output(code)
        assert "registered" in out
        assert "hi" in out

    def test_multiple_function_decorators_still_work(self):
        code = '''
        fn a(f) {
            fn w() { print "a"; return f(); }
            return w;
        }
        fn b(f) {
            fn w() { print "b"; return f(); }
            return w;
        }

        @a
        @b
        fn target() { print "target"; return 1; }

        target();
        '''
        out = run_output(code)
        lines = out.strip().split("\n")
        assert lines[0] == "a"
        assert lines[1] == "b"
        assert lines[2] == "target"
