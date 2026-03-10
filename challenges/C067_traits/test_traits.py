"""
Tests for C067: Traits
Challenge C067 -- AgentZero Session 068

Tests trait declarations, required methods, default methods, trait inheritance,
multiple traits, instanceof with traits, export traits, and integration with
existing class features (inheritance, getters/setters, static methods, etc.).
"""

import pytest
from traits import run, execute, VMError, ParseError


# ============================================================
# Basic trait declaration
# ============================================================

class TestTraitDeclaration:
    def test_empty_trait(self):
        result, _ = run('trait Empty {} let x = 1; x;')
        assert result == 1

    def test_trait_with_required_method(self):
        _, output = run('''
        trait Greetable {
            greet();
        }
        class Person implements Greetable {
            greet() { return "hello"; }
        }
        let p = Person();
        print(p.greet());
        ''')
        assert output == ['hello']

    def test_trait_with_default_method(self):
        _, output = run('''
        trait HasDefault {
            value() { return 42; }
        }
        class Foo implements HasDefault {
            init() {}
        }
        print(Foo().value());
        ''')
        assert output == ['42']

    def test_trait_with_mixed_methods(self):
        _, output = run('''
        trait Mixed {
            required_fn();
            default_fn() { return "default"; }
        }
        class Bar implements Mixed {
            required_fn() { return "implemented"; }
        }
        let b = Bar();
        print(b.required_fn());
        print(b.default_fn());
        ''')
        assert output == ['implemented', 'default']

    def test_trait_stored_in_variable(self):
        result, _ = run('''
        trait T { x(); }
        type(T);
        ''')
        assert result == 'trait'

    def test_trait_with_fn_keyword(self):
        """Trait methods can optionally use fn keyword."""
        _, output = run('''
        trait WithFn {
            fn greet();
            fn hello() { return "hi"; }
        }
        class Impl implements WithFn {
            greet() { return "yo"; }
        }
        let i = Impl();
        print(i.greet());
        print(i.hello());
        ''')
        assert output == ['yo', 'hi']


# ============================================================
# Required method validation
# ============================================================

class TestRequiredMethods:
    def test_missing_required_method_error(self):
        with pytest.raises(VMError, match="does not implement required method"):
            run('''
            trait Foo { bar(); }
            class Baz implements Foo { init() {} }
            ''')

    def test_missing_one_of_multiple_required(self):
        with pytest.raises(VMError, match="does not implement required method"):
            run('''
            trait T { a(); b(); c(); }
            class C implements T { a() { return 1; } b() { return 2; } }
            ''')

    def test_all_required_methods_satisfied(self):
        _, output = run('''
        trait T { a(); b(); c(); }
        class C implements T {
            a() { return 1; }
            b() { return 2; }
            c() { return 3; }
        }
        let obj = C();
        print(obj.a());
        print(obj.b());
        print(obj.c());
        ''')
        assert output == ['1', '2', '3']

    def test_class_override_default_method(self):
        _, output = run('''
        trait T {
            value() { return "default"; }
        }
        class C implements T {
            value() { return "custom"; }
        }
        print(C().value());
        ''')
        assert output == ['custom']

    def test_required_method_with_params(self):
        _, output = run('''
        trait Math {
            add(a, b);
        }
        class Calculator implements Math {
            add(a, b) { return a + b; }
        }
        print(Calculator().add(3, 4));
        ''')
        assert output == ['7']


# ============================================================
# Trait inheritance
# ============================================================

class TestTraitInheritance:
    def test_basic_trait_inheritance(self):
        _, output = run('''
        trait Base {
            base_method() { return "base"; }
        }
        trait Child < Base {
            child_method() { return "child"; }
        }
        class Impl implements Child {
            init() {}
        }
        let obj = Impl();
        print(obj.base_method());
        print(obj.child_method());
        ''')
        assert output == ['base', 'child']

    def test_trait_inherits_required_methods(self):
        with pytest.raises(VMError, match="does not implement required method"):
            run('''
            trait Base { required(); }
            trait Child < Base {}
            class C implements Child { init() {} }
            ''')

    def test_trait_inherits_required_satisfied(self):
        _, output = run('''
        trait Base { required(); }
        trait Child < Base {
            extra() { return "extra"; }
        }
        class C implements Child {
            required() { return "done"; }
        }
        let obj = C();
        print(obj.required());
        print(obj.extra());
        ''')
        assert output == ['done', 'extra']

    def test_child_trait_overrides_parent_default(self):
        _, output = run('''
        trait Base {
            value() { return "base_default"; }
        }
        trait Child < Base {
            value() { return "child_default"; }
        }
        class C implements Child { init() {} }
        print(C().value());
        ''')
        assert output == ['child_default']

    def test_child_trait_provides_default_for_parent_required(self):
        _, output = run('''
        trait Base { required(); }
        trait Child < Base {
            required() { return "provided_by_child"; }
        }
        class C implements Child { init() {} }
        print(C().required());
        ''')
        assert output == ['provided_by_child']

    def test_deep_trait_inheritance(self):
        _, output = run('''
        trait A { a() { return "a"; } }
        trait B < A { b() { return "b"; } }
        trait C < B { c() { return "c"; } }
        class Impl implements C { init() {} }
        let obj = Impl();
        print(obj.a());
        print(obj.b());
        print(obj.c());
        ''')
        assert output == ['a', 'b', 'c']

    def test_parent_trait_not_a_trait_error(self):
        with pytest.raises(VMError, match="Parent of trait must be a trait"):
            run('''
            class Foo { init() {} }
            trait Bad < Foo {}
            ''')


# ============================================================
# Multiple traits
# ============================================================

class TestMultipleTraits:
    def test_two_traits(self):
        _, output = run('''
        trait A { a() { return 1; } }
        trait B { b() { return 2; } }
        class C implements A, B { init() {} }
        let c = C();
        print(c.a());
        print(c.b());
        ''')
        assert output == ['1', '2']

    def test_three_traits(self):
        _, output = run('''
        trait X { x(); }
        trait Y { y(); }
        trait Z { z(); }
        class Impl implements X, Y, Z {
            x() { return "x"; }
            y() { return "y"; }
            z() { return "z"; }
        }
        let obj = Impl();
        print(obj.x());
        print(obj.y());
        print(obj.z());
        ''')
        assert output == ['x', 'y', 'z']

    def test_multiple_traits_with_defaults(self):
        _, output = run('''
        trait Printable { to_string() { return "object"; } }
        trait Comparable { compare(other) { return 0; } }
        class Item implements Printable, Comparable { init() {} }
        let i = Item();
        print(i.to_string());
        print(i.compare(null));
        ''')
        assert output == ['object', '0']

    def test_multiple_traits_one_missing_required(self):
        with pytest.raises(VMError, match="does not implement required method"):
            run('''
            trait A { a(); }
            trait B { b(); }
            class C implements A, B { a() { return 1; } }
            ''')

    def test_later_trait_default_does_not_override_earlier(self):
        """When multiple traits define same default, first one wins (already set)."""
        _, output = run('''
        trait A { value() { return "a"; } }
        trait B { value() { return "b"; } }
        class C implements A, B { init() {} }
        print(C().value());
        ''')
        assert output == ['a']

    def test_class_method_overrides_all_trait_defaults(self):
        _, output = run('''
        trait A { value() { return "a"; } }
        trait B { value() { return "b"; } }
        class C implements A, B {
            value() { return "c"; }
        }
        print(C().value());
        ''')
        assert output == ['c']


# ============================================================
# instanceof with traits
# ============================================================

class TestInstanceofTraits:
    def test_instanceof_trait(self):
        result, _ = run('''
        trait T { x(); }
        class C implements T { x() { return 1; } }
        instanceof(C(), T);
        ''')
        assert result is True

    def test_instanceof_trait_false(self):
        result, _ = run('''
        trait T { x(); }
        class C { init() {} }
        instanceof(C(), T);
        ''')
        assert result is False

    def test_instanceof_parent_trait(self):
        result, _ = run('''
        trait Base { x(); }
        trait Child < Base { y(); }
        class C implements Child {
            x() { return 1; }
            y() { return 2; }
        }
        instanceof(C(), Base);
        ''')
        assert result is True

    def test_instanceof_multiple_traits(self):
        _, output = run('''
        trait A { a(); }
        trait B { b(); }
        class C implements A, B {
            a() { return 1; }
            b() { return 2; }
        }
        let c = C();
        print(instanceof(c, A));
        print(instanceof(c, B));
        ''')
        assert output == ['true', 'true']

    def test_instanceof_non_instance_with_trait(self):
        result, _ = run('''
        trait T { x(); }
        instanceof(42, T);
        ''')
        assert result is False

    def test_instanceof_null_with_trait(self):
        result, _ = run('''
        trait T { x(); }
        instanceof(null, T);
        ''')
        assert result is False

    def test_instanceof_with_class_inheritance_and_traits(self):
        """Subclass inherits parent's traits for instanceof."""
        _, output = run('''
        trait T { x(); }
        class Base implements T {
            x() { return "base"; }
        }
        class Child < Base {
            x() { return "child"; }
        }
        let c = Child();
        print(instanceof(c, T));
        ''')
        assert output == ['true']

    def test_instanceof_deep_class_hierarchy(self):
        result, _ = run('''
        trait T { x(); }
        class A implements T { x() { return 1; } }
        class B < A { init() {} }
        class C < B { init() {} }
        instanceof(C(), T);
        ''')
        assert result is True


# ============================================================
# Traits with class inheritance
# ============================================================

class TestTraitsWithClassInheritance:
    def test_class_with_parent_and_trait(self):
        _, output = run('''
        trait Greetable { greet(); }
        class Base {
            init(name) { this.name = name; }
        }
        class Person < Base implements Greetable {
            greet() { return f"hello, ${this.name}"; }
        }
        print(Person("world").greet());
        ''')
        assert output == ['hello, world']

    def test_parent_satisfies_trait_method(self):
        """Method from parent class satisfies trait requirement."""
        _, output = run('''
        trait Describable { describe(); }
        class Base {
            describe() { return "I am base"; }
        }
        class Child < Base implements Describable {
            init() {}
        }
        print(Child().describe());
        ''')
        assert output == ['I am base']

    def test_trait_default_available_to_subclass(self):
        _, output = run('''
        trait T { value() { return 99; } }
        class A implements T { init() {} }
        class B < A { init() {} }
        print(B().value());
        ''')
        assert output == ['99']

    def test_subclass_overrides_trait_default(self):
        _, output = run('''
        trait T { value() { return 99; } }
        class A implements T { init() {} }
        class B < A {
            value() { return 100; }
        }
        print(B().value());
        ''')
        assert output == ['100']


# ============================================================
# Trait default methods with this
# ============================================================

class TestTraitDefaultMethodsThis:
    def test_default_method_accesses_this(self):
        _, output = run('''
        trait Named {
            greet() { return f"Hi, ${this.name}"; }
        }
        class User implements Named {
            init(name) { this.name = name; }
        }
        print(User("Alice").greet());
        ''')
        assert output == ['Hi, Alice']

    def test_default_method_calls_other_method(self):
        _, output = run('''
        trait Formatted {
            raw();
            formatted() { return f"[${this.raw()}]"; }
        }
        class Item implements Formatted {
            init(v) { this.v = v; }
            raw() { return this.v; }
        }
        print(Item("test").formatted());
        ''')
        assert output == ['[test]']

    def test_default_method_calls_class_method(self):
        _, output = run('''
        trait Display {
            display() { return f"Display: ${this.to_string()}"; }
        }
        class Widget implements Display {
            init(id) { this.id = id; }
            to_string() { return f"Widget#${this.id}"; }
        }
        print(Widget(42).display());
        ''')
        assert output == ['Display: Widget#42']


# ============================================================
# Export traits
# ============================================================

class TestExportTraits:
    def test_export_trait(self):
        r = execute('''
        export trait Serializable {
            serialize();
            deserialize(data);
        }
        let x = 1;
        ''')
        assert 'Serializable' in r['exports']

    def test_export_trait_and_class(self):
        r = execute('''
        export trait Printable {
            to_string() { return "obj"; }
        }
        export class Item implements Printable {
            init() {}
        }
        ''')
        assert 'Printable' in r['exports']
        assert 'Item' in r['exports']


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_trait_no_methods(self):
        """Empty trait -- marker trait."""
        _, output = run('''
        trait Marker {}
        class C implements Marker { init() {} }
        print(instanceof(C(), Marker));
        ''')
        assert output == ['true']

    def test_multiple_classes_same_trait(self):
        _, output = run('''
        trait Speakable { speak(); }
        class Dog implements Speakable { speak() { return "woof"; } }
        class Cat implements Speakable { speak() { return "meow"; } }
        print(Dog().speak());
        print(Cat().speak());
        print(instanceof(Dog(), Speakable));
        print(instanceof(Cat(), Speakable));
        ''')
        assert output == ['woof', 'meow', 'true', 'true']

    def test_instanceof_wrong_trait(self):
        result, _ = run('''
        trait A {}
        trait B {}
        class C implements A { init() {} }
        instanceof(C(), B);
        ''')
        assert result is False

    def test_trait_method_with_no_params(self):
        _, output = run('''
        trait T { value() { return 42; } }
        class C implements T { init() {} }
        print(C().value());
        ''')
        assert output == ['42']

    def test_trait_method_with_multiple_params(self):
        _, output = run('''
        trait T { compute(a, b, c); }
        class C implements T {
            compute(a, b, c) { return a + b + c; }
        }
        print(C().compute(1, 2, 3));
        ''')
        assert output == ['6']

    def test_trait_with_class_that_has_init(self):
        _, output = run('''
        trait T { value() { return this.x * 2; } }
        class C implements T {
            init(x) { this.x = x; }
        }
        print(C(5).value());
        ''')
        assert output == ['10']

    def test_impl_traits_not_a_trait_error(self):
        with pytest.raises(VMError, match="Expected trait"):
            run('''
            class Foo { init() {} }
            class Bar implements Foo { init() {} }
            ''')

    def test_type_of_trait(self):
        result, _ = run('''
        trait T {}
        type(T);
        ''')
        assert result == 'trait'


# ============================================================
# Integration with existing features
# ============================================================

class TestIntegrationFeatures:
    def test_trait_with_closures(self):
        _, output = run('''
        trait T {
            make_adder(n) {
                return fn(x) { return x + n; };
            }
        }
        class C implements T { init() {} }
        let add5 = C().make_adder(5);
        print(add5(3));
        ''')
        assert output == ['8']

    def test_trait_with_arrays(self):
        _, output = run('''
        trait Listable {
            to_list();
        }
        class Range implements Listable {
            init(n) { this.n = n; }
            to_list() {
                let result = [];
                let i = 0;
                while (i < this.n) {
                    result.push(i);
                    i = i + 1;
                }
                return result;
            }
        }
        print(Range(3).to_list());
        ''')
        assert output == ['[0, 1, 2]']

    def test_trait_with_hash_maps(self):
        _, output = run('''
        trait Configurable {
            defaults() { return {a: 1, b: 2}; }
        }
        class App implements Configurable { init() {} }
        let cfg = App().defaults();
        print(cfg.a);
        print(cfg.b);
        ''')
        assert output == ['1', '2']

    def test_trait_with_error_handling(self):
        _, output = run('''
        trait Validator {
            validate(x) {
                if (x < 0) {
                    throw "negative";
                }
                return true;
            }
        }
        class Checker implements Validator { init() {} }
        try {
            Checker().validate(-1);
        } catch (e) {
            print(e);
        }
        ''')
        assert output == ['negative']

    def test_trait_with_string_interpolation(self):
        _, output = run('''
        trait Labeled {
            label() { return f"[${this.name}]"; }
        }
        class Item implements Labeled {
            init(name) { this.name = name; }
        }
        print(Item("test").label());
        ''')
        assert output == ['[test]']

    def test_trait_with_optional_chaining(self):
        _, output = run('''
        trait T { value() { return this.data?.name; } }
        class C implements T {
            init(data) { this.data = data; }
        }
        print(C({name: "hi"}).value());
        print(C(null).value());
        ''')
        assert output == ['hi', 'null']

    def test_trait_with_null_coalescing(self):
        _, output = run('''
        trait T {
            safe_name() { return this.name ?? "unknown"; }
        }
        class C implements T {
            init(name) { this.name = name; }
        }
        print(C("test").safe_name());
        print(C(null).safe_name());
        ''')
        assert output == ['test', 'unknown']

    def test_trait_with_destructuring(self):
        _, output = run('''
        trait T {
            first_two() {
                let [a, b] = this.items;
                return a + b;
            }
        }
        class C implements T {
            init() { this.items = [10, 20, 30]; }
        }
        print(C().first_two());
        ''')
        assert output == ['30']

    def test_trait_with_spread(self):
        _, output = run('''
        trait Mergeable {
            merge(other) {
                return {...this.data, ...other};
            }
        }
        class Config implements Mergeable {
            init(data) { this.data = data; }
        }
        let c = Config({a: 1, b: 2});
        let merged = c.merge({b: 3, c: 4});
        print(merged.a);
        print(merged.b);
        print(merged.c);
        ''')
        assert output == ['1', '3', '4']

    def test_trait_with_pipe_operator(self):
        _, output = run('''
        fn double(x) { return x * 2; }
        fn add1(x) { return x + 1; }
        trait T {
            transform(x) {
                return x |> double |> add1;
            }
        }
        class C implements T { init() {} }
        print(C().transform(5));
        ''')
        assert output == ['11']


# ============================================================
# Trait with getters/setters (C058)
# ============================================================

class TestTraitsWithClassGettersSetters:
    def test_class_getters_with_trait(self):
        _, output = run('''
        trait Sized { size(); }
        class Collection implements Sized {
            init() { this.items = [1, 2, 3]; }
            size() { return this.items.length; }
        }
        print(Collection().size());
        ''')
        assert output == ['3']

    def test_class_with_static_and_trait(self):
        _, output = run('''
        trait Identifiable { id(); }
        class Entity implements Identifiable {
            static create(n) { return Entity(n); }
            init(n) { this.n = n; }
            id() { return this.n; }
        }
        let e = Entity.create(42);
        print(e.id());
        print(instanceof(e, Identifiable));
        ''')
        assert output == ['42', 'true']


# ============================================================
# Complex composition patterns
# ============================================================

class TestComplexComposition:
    def test_diamond_traits(self):
        """Two traits inherit from same base, class implements both."""
        _, output = run('''
        trait Base { base_val() { return "base"; } }
        trait Left < Base { left_val() { return "left"; } }
        trait Right < Base { right_val() { return "right"; } }
        class Diamond implements Left, Right { init() {} }
        let d = Diamond();
        print(d.base_val());
        print(d.left_val());
        print(d.right_val());
        print(instanceof(d, Base));
        print(instanceof(d, Left));
        print(instanceof(d, Right));
        ''')
        assert output == ['base', 'left', 'right', 'true', 'true', 'true']

    def test_trait_and_class_hierarchy(self):
        _, output = run('''
        trait Renderable {
            render();
            visible() { return true; }
        }
        class Shape {
            init(name) { this.name = name; }
        }
        class Circle < Shape implements Renderable {
            init(r) {
                super.init(f"circle-${r}");
                this.r = r;
            }
            render() { return f"O(${this.r})"; }
        }
        let c = Circle(5);
        print(c.name);
        print(c.render());
        print(c.visible());
        print(instanceof(c, Renderable));
        print(instanceof(c, Shape));
        ''')
        assert output == ['circle-5', 'O(5)', 'true', 'true', 'true']

    def test_trait_method_returns_new_instance(self):
        _, output = run('''
        trait Clonable {
            clone() {
                return Pair(this.x, this.y);
            }
        }
        class Pair implements Clonable {
            init(x, y) {
                this.x = x;
                this.y = y;
            }
        }
        let p = Pair(1, 2);
        let p2 = p.clone();
        print(p2.x);
        print(p2.y);
        ''')
        assert output == ['1', '2']

    def test_many_traits(self):
        _, output = run('''
        trait A { a() { return 1; } }
        trait B { b() { return 2; } }
        trait C { c() { return 3; } }
        trait D { d() { return 4; } }
        class Multi implements A, B, C, D { init() {} }
        let m = Multi();
        print(m.a() + m.b() + m.c() + m.d());
        ''')
        assert output == ['10']


# ============================================================
# Regression: existing features still work
# ============================================================

class TestRegression:
    def test_class_without_traits(self):
        _, output = run('''
        class Simple {
            init(x) { this.x = x; }
            get_x() { return this.x; }
        }
        print(Simple(42).get_x());
        ''')
        assert output == ['42']

    def test_class_inheritance_without_traits(self):
        _, output = run('''
        class Base {
            init(x) { this.x = x; }
        }
        class Child < Base {
            double() { return this.x * 2; }
        }
        print(Child(5).double());
        ''')
        assert output == ['10']

    def test_enum_still_works(self):
        _, output = run('''
        enum Color { Red, Green, Blue }
        print(Color.Red.name);
        print(Color.Green.ordinal);
        ''')
        assert output == ['Red', '1']

    def test_closures_still_work(self):
        _, output = run('''
        fn make_counter() {
            let count = 0;
            return fn() {
                count = count + 1;
                return count;
            };
        }
        let c = make_counter();
        print(c());
        print(c());
        ''')
        assert output == ['1', '1']  # closures copy env

    def test_generators_still_work(self):
        _, output = run('''
        fn* range(n) {
            let i = 0;
            while (i < n) {
                yield i;
                i = i + 1;
            }
        }
        let g = range(3);
        print(next(g));
        print(next(g));
        print(next(g));
        ''')
        assert output == ['0', '1', '2']

    def test_async_await_still_works(self):
        _, output = run('''
        async fn compute() {
            return 42;
        }
        let p = compute();
        print(await p);
        ''')
        assert output == ['42']

    def test_for_in_still_works(self):
        _, output = run('''
        let arr = [10, 20, 30];
        for (x in arr) {
            print(x);
        }
        ''')
        assert output == ['10', '20', '30']

    def test_try_catch_finally_still_works(self):
        _, output = run('''
        try {
            throw "err";
        } catch (e) {
            print(e);
        } finally {
            print("done");
        }
        ''')
        assert output == ['err', 'done']

    def test_modules_still_work(self):
        r = execute('''
        export fn add(a, b) { return a + b; }
        export let PI = 3;
        ''')
        assert 'add' in r['exports']
        assert 'PI' in r['exports']

    def test_string_methods_still_work(self):
        _, output = run('''
        print("hello".toUpperCase());
        print("WORLD".toLowerCase());
        ''')
        assert output == ['HELLO', 'world']

    def test_optional_chaining_still_works(self):
        _, output = run('''
        let x = null;
        print(x?.foo);
        ''')
        assert output == ['null']

    def test_null_coalescing_still_works(self):
        _, output = run('''
        let x = null;
        print(x ?? "default");
        ''')
        assert output == ['default']

    def test_spread_still_works(self):
        _, output = run('''
        let a = [1, 2];
        let b = [...a, 3];
        print(b);
        ''')
        assert output == ['[1, 2, 3]']

    def test_pipe_still_works(self):
        _, output = run('''
        fn double(x) { return x * 2; }
        print(5 |> double);
        ''')
        assert output == ['10']

    def test_classes_still_work(self):
        _, output = run('''
        class Point {
            init(x, y) { this.x = x; this.y = y; }
            to_string() { return f"(${this.x}, ${this.y})"; }
        }
        print(Point(1, 2).to_string());
        ''')
        assert output == ['(1, 2)']

    def test_instanceof_class_still_works(self):
        result, _ = run('''
        class A { init() {} }
        class B < A { init() {} }
        instanceof(B(), A);
        ''')
        assert result is True

    def test_destructuring_still_works(self):
        _, output = run('''
        let [a, b, c] = [1, 2, 3];
        print(a + b + c);
        ''')
        assert output == ['6']


# ============================================================
# Trait as first-class value
# ============================================================

class TestTraitFirstClass:
    def test_trait_in_variable(self):
        _, output = run('''
        trait T { x() { return 1; } }
        let t = T;
        print(type(t));
        ''')
        assert output == ['trait']

    def test_trait_passed_to_function(self):
        _, output = run('''
        trait T { x(); }
        class C implements T { x() { return "yes"; } }
        fn check(obj, t) {
            return instanceof(obj, t);
        }
        print(check(C(), T));
        ''')
        assert output == ['true']

    def test_trait_in_array(self):
        _, output = run('''
        trait A {}
        trait B {}
        let traits = [A, B];
        print(type(traits[0]));
        print(type(traits[1]));
        ''')
        assert output == ['trait', 'trait']


# ============================================================
# Advanced trait patterns
# ============================================================

class TestAdvancedPatterns:
    def test_trait_method_mutates_this(self):
        _, output = run('''
        trait Counter {
            increment() {
                this.count = this.count + 1;
                return this.count;
            }
        }
        class C implements Counter {
            init() { this.count = 0; }
        }
        let c = C();
        print(c.increment());
        print(c.increment());
        print(c.increment());
        ''')
        assert output == ['1', '2', '3']

    def test_trait_with_while_loop(self):
        _, output = run('''
        trait Summable {
            sum_to(n) {
                let total = 0;
                let i = 1;
                while (i <= n) {
                    total = total + i;
                    i = i + 1;
                }
                return total;
            }
        }
        class Math implements Summable { init() {} }
        print(Math().sum_to(10));
        ''')
        assert output == ['55']

    def test_trait_with_recursion_via_class(self):
        _, output = run('''
        trait Factorial {
            fact(n) {
                if (n <= 1) { return 1; }
                return n * this.fact(n - 1);
            }
        }
        class M implements Factorial { init() {} }
        print(M().fact(5));
        ''')
        assert output == ['120']

    def test_trait_default_uses_other_trait_default(self):
        _, output = run('''
        trait T {
            base() { return 10; }
            derived() { return this.base() * 2; }
        }
        class C implements T { init() {} }
        print(C().derived());
        ''')
        assert output == ['20']

    def test_trait_with_array_methods(self):
        _, output = run('''
        trait Filterable {
            even_items() {
                return this.items.filter(fn(x) { return x % 2 == 0; });
            }
        }
        class Nums implements Filterable {
            init(items) { this.items = items; }
        }
        print(Nums([1, 2, 3, 4, 5, 6]).even_items());
        ''')
        assert output == ['[2, 4, 6]']

    def test_trait_with_for_in(self):
        _, output = run('''
        trait Printable {
            print_all() {
                for (item in this.items) {
                    print(item);
                }
            }
        }
        class Container implements Printable {
            init(items) { this.items = items; }
        }
        Container(["a", "b", "c"]).print_all();
        ''')
        assert output == ['a', 'b', 'c']

    def test_class_with_parent_trait_and_own_trait(self):
        """Parent has trait A, child has trait B -- instanceof both."""
        _, output = run('''
        trait A { a() { return "a"; } }
        trait B { b() { return "b"; } }
        class Parent implements A { init() {} }
        class Child < Parent implements B { init() {} }
        let c = Child();
        print(c.a());
        print(c.b());
        print(instanceof(c, A));
        print(instanceof(c, B));
        ''')
        assert output == ['a', 'b', 'true', 'true']

    def test_trait_with_enum(self):
        _, output = run('''
        enum Status { Active, Inactive }
        trait Statusable {
            is_active() { return this.status == Status.Active; }
        }
        class User implements Statusable {
            init(status) { this.status = status; }
        }
        print(User(Status.Active).is_active());
        print(User(Status.Inactive).is_active());
        ''')
        assert output == ['true', 'false']

    def test_trait_with_try_catch(self):
        _, output = run('''
        trait SafeDiv {
            safe_divide(a, b) {
                try {
                    if (b == 0) { throw "div by zero"; }
                    return a / b;
                } catch (e) {
                    return e;
                }
            }
        }
        class Calc implements SafeDiv { init() {} }
        print(Calc().safe_divide(10, 2));
        print(Calc().safe_divide(10, 0));
        ''')
        assert output == ['5', 'div by zero']

    def test_instanceof_with_hash_not_instance(self):
        """Plain hash (not instance) should not match trait."""
        result, _ = run('''
        trait T {}
        instanceof({a: 1}, T);
        ''')
        assert result is False

    def test_trait_default_with_conditional(self):
        _, output = run('''
        trait Classifiable {
            classify() {
                if (this.value > 0) { return "positive"; }
                else if (this.value < 0) { return "negative"; }
                else { return "zero"; }
            }
        }
        class Num implements Classifiable {
            init(value) { this.value = value; }
        }
        print(Num(5).classify());
        print(Num(-3).classify());
        print(Num(0).classify());
        ''')
        assert output == ['positive', 'negative', 'zero']

    def test_multiple_classes_multiple_traits(self):
        _, output = run('''
        trait Speakable { speak(); }
        trait Movable { move() { return "moving"; } }
        class Dog implements Speakable, Movable {
            speak() { return "woof"; }
        }
        class Robot implements Speakable, Movable {
            speak() { return "beep"; }
        }
        print(Dog().speak());
        print(Dog().move());
        print(Robot().speak());
        print(Robot().move());
        print(instanceof(Dog(), Speakable));
        print(instanceof(Robot(), Movable));
        ''')
        assert output == ['woof', 'moving', 'beep', 'moving', 'true', 'true']

    def test_trait_with_string_operations(self):
        _, output = run('''
        trait Formatter {
            upper() { return this.text.toUpperCase(); }
            lower() { return this.text.toLowerCase(); }
        }
        class Message implements Formatter {
            init(text) { this.text = text; }
        }
        let m = Message("Hello World");
        print(m.upper());
        print(m.lower());
        ''')
        assert output == ['HELLO WORLD', 'hello world']

    def test_trait_inheritance_chain_instanceof(self):
        """A < B < C, class implements C, instanceof A returns true."""
        result, _ = run('''
        trait A {}
        trait B < A {}
        trait C < B {}
        class Impl implements C { init() {} }
        instanceof(Impl(), A);
        ''')
        assert result is True

    def test_two_unrelated_traits_no_cross_instanceof(self):
        _, output = run('''
        trait X {}
        trait Y {}
        class OnlyX implements X { init() {} }
        class OnlyY implements Y { init() {} }
        print(instanceof(OnlyX(), Y));
        print(instanceof(OnlyY(), X));
        ''')
        assert output == ['false', 'false']


# ============================================================
# Run all tests
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
