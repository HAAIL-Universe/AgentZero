"""
Tests for C052: Classes
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from classes import (
    run, execute, parse, compile_source, lex, disassemble,
    ClassObject, BoundMethod, FnObject, ClosureObject,
    ModuleRegistry,
    VMError, ParseError, CompileError,
)


# ============================================================
# Basic class declaration
# ============================================================

class TestBasicClass:
    def test_empty_class(self):
        r, out = run("class Foo {} let f = Foo(); print type(f);")
        assert out == ["Foo"]

    def test_class_with_init(self):
        r, out = run("""
            class Point {
                init(x, y) {
                    this.x = x;
                    this.y = y;
                }
            }
            let p = Point(3, 4);
            print p.x;
            print p.y;
        """)
        assert out == ["3", "4"]

    def test_class_with_method(self):
        r, out = run("""
            class Greeter {
                init(name) { this.name = name; }
                greet() { return "Hello, " + this.name; }
            }
            let g = Greeter("World");
            print g.greet();
        """)
        assert out == ["Hello, World"]

    def test_class_multiple_methods(self):
        r, out = run("""
            class Calc {
                init(val) { this.val = val; }
                add(n) { return this.val + n; }
                mul(n) { return this.val * n; }
            }
            let c = Calc(10);
            print c.add(5);
            print c.mul(3);
        """)
        assert out == ["15", "30"]

    def test_class_no_init_no_args(self):
        r, out = run("""
            class Empty {}
            let e = Empty();
            print type(e);
        """)
        assert out == ["Empty"]

    def test_class_init_no_return(self):
        """init returns the instance, not None."""
        r, out = run("""
            class Foo {
                init(x) { this.x = x; }
            }
            let f = Foo(42);
            print f.x;
        """)
        assert out == ["42"]

    def test_class_stored_as_variable(self):
        r, out = run("""
            class Dog { init(n) { this.n = n; } }
            let cls = Dog;
            let d = cls("Rex");
            print d.n;
        """)
        assert out == ["Rex"]

    def test_class_type_function(self):
        r, out = run("""
            class Foo {}
            print type(Foo);
            let f = Foo();
            print type(f);
        """)
        assert out == ["class", "Foo"]


# ============================================================
# Property access and mutation
# ============================================================

class TestProperties:
    def test_set_property_in_init(self):
        r, out = run("""
            class Box {
                init(val) { this.val = val; }
            }
            let b = Box(99);
            print b.val;
        """)
        assert out == ["99"]

    def test_set_property_outside(self):
        r, out = run("""
            class Obj {}
            let o = Obj();
            o.x = 10;
            o.y = 20;
            print o.x + o.y;
        """)
        assert out == ["30"]

    def test_mutate_property(self):
        r, out = run("""
            class Counter {
                init() { this.count = 0; }
                inc() { this.count = this.count + 1; }
                get() { return this.count; }
            }
            let c = Counter();
            c.inc();
            c.inc();
            c.inc();
            print c.get();
        """)
        assert out == ["3"]

    def test_method_modifies_instance(self):
        r, out = run("""
            class List {
                init() { this.items = []; }
                add(item) { push(this.items, item); }
                size() { return len(this.items); }
            }
            let l = List();
            l.add(1);
            l.add(2);
            l.add(3);
            print l.size();
        """)
        assert out == ["3"]

    def test_property_not_found(self):
        with pytest.raises(VMError, match="has no property"):
            run("""
                class Foo { init() { this.x = 1; } }
                let f = Foo();
                print f.y;
            """)

    def test_multiple_instances_independent(self):
        r, out = run("""
            class Pair {
                init(a, b) { this.a = a; this.b = b; }
            }
            let p1 = Pair(1, 2);
            let p2 = Pair(3, 4);
            print p1.a;
            print p2.a;
            print p1.b;
            print p2.b;
        """)
        assert out == ["1", "3", "2", "4"]


# ============================================================
# Inheritance
# ============================================================

class TestInheritance:
    def test_basic_inheritance(self):
        r, out = run("""
            class Animal {
                init(name) { this.name = name; }
                speak() { return this.name + " speaks"; }
            }
            class Dog < Animal {
                bark() { return this.name + " barks"; }
            }
            let d = Dog("Rex");
            print d.speak();
            print d.bark();
        """)
        assert out == ["Rex speaks", "Rex barks"]

    def test_super_init(self):
        r, out = run("""
            class Animal {
                init(name, sound) {
                    this.name = name;
                    this.sound = sound;
                }
                speak() { return this.name + " says " + this.sound; }
            }
            class Dog < Animal {
                init(name) {
                    super.init(name, "woof");
                }
            }
            let d = Dog("Rex");
            print d.speak();
        """)
        assert out == ["Rex says woof"]

    def test_method_override(self):
        r, out = run("""
            class Base {
                greet() { return "Base"; }
            }
            class Child < Base {
                greet() { return "Child"; }
            }
            let c = Child();
            print c.greet();
        """)
        assert out == ["Child"]

    def test_super_method_call(self):
        r, out = run("""
            class Base {
                greet() { return "Hello"; }
            }
            class Child < Base {
                greet() { return super.greet() + " World"; }
            }
            let c = Child();
            print c.greet();
        """)
        assert out == ["Hello World"]

    def test_three_level_inheritance(self):
        r, out = run("""
            class A {
                init(x) { this.x = x; }
                who() { return "A"; }
            }
            class B < A {
                init(x, y) {
                    super.init(x);
                    this.y = y;
                }
                who() { return "B"; }
            }
            class C < B {
                init(x, y, z) {
                    super.init(x, y);
                    this.z = z;
                }
                who() { return "C"; }
            }
            let c = C(1, 2, 3);
            print c.x;
            print c.y;
            print c.z;
            print c.who();
        """)
        assert out == ["1", "2", "3", "C"]

    def test_inherit_method_from_grandparent(self):
        r, out = run("""
            class A {
                foo() { return "A.foo"; }
            }
            class B < A {}
            class C < B {}
            let c = C();
            print c.foo();
        """)
        assert out == ["A.foo"]

    def test_super_chain(self):
        r, out = run("""
            class A {
                greet() { return "A"; }
            }
            class B < A {
                greet() { return super.greet() + "B"; }
            }
            class C < B {
                greet() { return super.greet() + "C"; }
            }
            let c = C();
            print c.greet();
        """)
        assert out == ["ABC"]

    def test_instanceof(self):
        r, out = run("""
            class Animal {}
            class Dog < Animal {}
            let d = Dog();
            print instanceof(d, Dog);
            print instanceof(d, Animal);
            print instanceof(d, Dog);
            let a = Animal();
            print instanceof(a, Dog);
            print instanceof(a, Animal);
        """)
        assert out == ["true", "true", "true", "false", "true"]

    def test_instanceof_non_instance(self):
        r, out = run("""
            class Foo {}
            print instanceof(42, Foo);
            print instanceof("hello", Foo);
            print instanceof([], Foo);
        """)
        assert out == ["false", "false", "false"]

    def test_non_class_parent_error(self):
        with pytest.raises(VMError, match="Superclass must be a class"):
            run("""
                let x = 42;
                class Foo < x {}
            """)


# ============================================================
# Method binding
# ============================================================

class TestMethodBinding:
    def test_method_returns_value(self):
        r, out = run("""
            class Math {
                init(base) { this.base = base; }
                add(n) { return this.base + n; }
            }
            let m = Math(100);
            print m.add(42);
        """)
        assert out == ["142"]

    def test_method_as_value(self):
        """Bound methods can be stored and called later."""
        r, out = run("""
            class Foo {
                init(x) { this.x = x; }
                getX() { return this.x; }
            }
            let f = Foo(42);
            let getter = f.getX;
            print getter();
        """)
        assert out == ["42"]

    def test_method_type(self):
        r, out = run("""
            class Foo {
                bar() { return 1; }
            }
            let f = Foo();
            print type(f.bar);
        """)
        assert out == ["function"]

    def test_this_in_nested_call(self):
        r, out = run("""
            class Wrapper {
                init(val) { this.val = val; }
                getVal() { return this.val; }
                doubled() { return this.getVal() * 2; }
            }
            let w = Wrapper(21);
            print w.doubled();
        """)
        assert out == ["42"]

    def test_method_with_closure(self):
        r, out = run("""
            class Adder {
                init(base) { this.base = base; }
                makeAdd() {
                    let self = this;
                    return fn(n) { return self.base + n; };
                }
            }
            let a = Adder(100);
            let add = a.makeAdd();
            print add(42);
        """)
        assert out == ["142"]

    def test_no_init_args_error(self):
        with pytest.raises(VMError, match="has no init"):
            run("class Foo {} Foo(1);")


# ============================================================
# This keyword
# ============================================================

class TestThis:
    def test_this_property_access(self):
        r, out = run("""
            class Pair {
                init(a, b) { this.a = a; this.b = b; }
                sum() { return this.a + this.b; }
            }
            print Pair(3, 4).sum();
        """)
        assert out == ["7"]

    def test_this_mutation(self):
        r, out = run("""
            class Acc {
                init() { this.total = 0; }
                add(n) {
                    this.total = this.total + n;
                    return this;
                }
            }
            let a = Acc();
            a.add(1).add(2).add(3);
            print a.total;
        """)
        assert out == ["6"]

    def test_this_in_conditional(self):
        r, out = run("""
            class Checker {
                init(val) { this.val = val; }
                isPositive() {
                    if (this.val > 0) { return true; }
                    return false;
                }
            }
            print Checker(5).isPositive();
            print Checker(0).isPositive();
        """)
        # Note: isPositive on a temp object -- each Checker() creates a new instance
        assert out == ["true", "false"]


# ============================================================
# Complex scenarios
# ============================================================

class TestComplex:
    def test_class_with_arrays(self):
        r, out = run("""
            class Stack {
                init() { this.items = []; }
                push(val) { push(this.items, val); }
                pop() { return pop(this.items); }
                size() { return len(this.items); }
            }
            let s = Stack();
            s.push(1);
            s.push(2);
            s.push(3);
            print s.size();
            print s.pop();
            print s.size();
        """)
        assert out == ["3", "3", "2"]

    def test_class_with_hash(self):
        r, out = run("""
            class Config {
                init() { this.data = {}; }
                set(key, val) {
                    this.data[key] = val;
                }
                get(key) {
                    return this.data[key];
                }
            }
            let c = Config();
            c.set("name", "AgentZero");
            c.set("version", 1);
            print c.get("name");
            print c.get("version");
        """)
        assert out == ["AgentZero", "1"]

    def test_class_with_for_in(self):
        r, out = run("""
            class Numbers {
                init(nums) { this.nums = nums; }
                sum() {
                    let total = 0;
                    for (n in this.nums) {
                        total = total + n;
                    }
                    return total;
                }
            }
            print Numbers([1, 2, 3, 4, 5]).sum();
        """)
        assert out == ["15"]

    def test_class_with_try_catch(self):
        r, out = run("""
            class SafeDiv {
                init(val) { this.val = val; }
                divBy(n) {
                    try {
                        return this.val / n;
                    } catch (e) {
                        return "error";
                    }
                }
            }
            let s = SafeDiv(10);
            print s.divBy(2);
            print s.divBy(0);
        """)
        assert out == ["5", "error"]

    def test_class_with_string_interpolation(self):
        r, out = run("""
            class Person {
                init(name, age) {
                    this.name = name;
                    this.age = age;
                }
                describe() {
                    return f"${this.name} is ${this.age} years old";
                }
            }
            print Person("Alice", 30).describe();
        """)
        assert out == ["Alice is 30 years old"]

    def test_class_with_pipe(self):
        r, out = run("""
            fn double(x) { return x * 2; }
            fn addTen(x) { return x + 10; }
            class Num {
                init(val) { this.val = val; }
                transform() { return this.val |> double |> addTen; }
            }
            print Num(5).transform();
        """)
        assert out == ["20"]

    def test_class_with_spread(self):
        r, out = run("""
            class Merger {
                init(base) { this.base = base; }
                merge(extra) { return [...this.base, ...extra]; }
            }
            let m = Merger([1, 2]);
            print m.merge([3, 4]);
        """)
        assert out == ["[1, 2, 3, 4]"]

    def test_class_with_destructuring(self):
        r, out = run("""
            class Point {
                init(x, y) { this.x = x; this.y = y; }
                toArray() { return [this.x, this.y]; }
            }
            let p = Point(3, 4);
            let [x, y] = p.toArray();
            print x;
            print y;
        """)
        assert out == ["3", "4"]


# ============================================================
# Inheritance patterns
# ============================================================

class TestInheritancePatterns:
    def test_template_method_pattern(self):
        r, out = run("""
            class Shape {
                area() { return 0; }
                describe() { return f"Area: ${string(this.area())}"; }
            }
            class Circle < Shape {
                init(r) { this.r = r; }
                area() { return this.r * this.r * 3; }
            }
            class Rect < Shape {
                init(w, h) { this.w = w; this.h = h; }
                area() { return this.w * this.h; }
            }
            print Circle(5).describe();
            print Rect(3, 4).describe();
        """)
        assert out == ["Area: 75", "Area: 12"]

    def test_factory_pattern(self):
        r, out = run("""
            class Animal {
                init(name, sound) {
                    this.name = name;
                    this.sound = sound;
                }
                speak() { return this.name + ": " + this.sound; }
            }
            class Dog < Animal {
                init(name) { super.init(name, "woof"); }
            }
            class Cat < Animal {
                init(name) { super.init(name, "meow"); }
            }
            print Dog("Rex").speak();
            print Cat("Whiskers").speak();
        """)
        assert out == ["Rex: woof", "Whiskers: meow"]

    def test_mixin_like_pattern(self):
        """Classes can have methods that call other methods -- like mixins."""
        r, out = run("""
            class Printable {
                toString() { return "object"; }
                display() { print this.toString(); }
            }
            class Named < Printable {
                init(name) { this.name = name; }
                toString() { return this.name; }
            }
            Named("AgentZero").display();
        """)
        assert out == ["AgentZero"]

    def test_builder_pattern(self):
        r, out = run("""
            class Builder {
                init() {
                    this.parts = [];
                }
                add(part) {
                    push(this.parts, part);
                    return this;
                }
                build() {
                    let result = "";
                    for (p in this.parts) {
                        if (result != "") { result = result + ", "; }
                        result = result + p;
                    }
                    return result;
                }
            }
            let b = Builder();
            print b.add("A").add("B").add("C").build();
        """)
        assert out == ["A, B, C"]


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_class_is_expression(self):
        """Class declaration is a statement that stores in env."""
        r = execute("class Foo {} let f = Foo();")
        assert 'Foo' in r['env']
        assert isinstance(r['env']['Foo'], ClassObject)

    def test_class_instance_is_dict(self):
        r = execute("class Foo { init(x) { this.x = x; } } let f = Foo(42);")
        f = r['env']['f']
        assert isinstance(f, dict)
        assert '__class__' in f
        assert f['x'] == 42

    def test_class_method_arity_mismatch(self):
        with pytest.raises(VMError):
            run("""
                class Foo {
                    bar(a, b) { return a + b; }
                }
                Foo().bar(1);
            """)

    def test_init_arity_mismatch(self):
        with pytest.raises(VMError):
            run("class Foo { init(x) { this.x = x; } } Foo();")

    def test_call_non_method(self):
        """Accessing a property that isn't a method and calling it should work if it's callable."""
        r, out = run("""
            class Holder {
                init(fn_val) { this.fn_val = fn_val; }
            }
            let h = Holder(fn(x) { return x * 2; });
            print h.fn_val(21);
        """)
        assert out == ["42"]

    def test_class_with_no_methods(self):
        r, out = run("""
            class Empty {}
            let e = Empty();
            e.x = 42;
            print e.x;
        """)
        assert out == ["42"]

    def test_class_format_value(self):
        r, out = run("""
            class Foo { init(x) { this.x = x; } }
            let f = Foo(42);
            print string(f);
            print string(Foo);
        """)
        assert "<Foo" in out[0]
        assert "<class:Foo>" in out[1]

    def test_override_init(self):
        r, out = run("""
            class Base {
                init(x) { this.x = x; }
            }
            class Child < Base {
                init(x, y) {
                    super.init(x);
                    this.y = y;
                }
            }
            let c = Child(1, 2);
            print c.x;
            print c.y;
        """)
        assert out == ["1", "2"]


# ============================================================
# Modules + Classes
# ============================================================

class TestModules:
    def test_export_class(self):
        reg = ModuleRegistry()
        reg.register("shapes", """
            export class Circle {
                init(r) { this.r = r; }
                area() { return this.r * this.r * 3; }
            }
        """)
        r, out = run("""
            import { Circle } from "shapes";
            let c = Circle(5);
            print c.area();
        """, registry=reg)
        assert out == ["75"]

    def test_export_class_with_inheritance(self):
        reg = ModuleRegistry()
        reg.register("base", """
            export class Shape {
                area() { return 0; }
            }
        """)
        reg.register("circle", """
            import { Shape } from "base";
            export class Circle < Shape {
                init(r) { this.r = r; }
                area() { return this.r * this.r * 3; }
            }
        """)
        r, out = run("""
            import { Circle } from "circle";
            print Circle(10).area();
        """, registry=reg)
        assert out == ["300"]


# ============================================================
# Polymorphism
# ============================================================

class TestPolymorphism:
    def test_polymorphic_dispatch(self):
        r, out = run("""
            class Shape {
                area() { return 0; }
            }
            class Square < Shape {
                init(s) { this.s = s; }
                area() { return this.s * this.s; }
            }
            class Triangle < Shape {
                init(b, h) { this.b = b; this.h = h; }
                area() { return this.b * this.h / 2; }
            }
            let shapes = [Square(4), Triangle(6, 3)];
            for (s in shapes) {
                print s.area();
            }
        """)
        assert out == ["16", "9"]

    def test_polymorphic_array(self):
        r, out = run("""
            class Animal {
                speak() { return "..."; }
            }
            class Dog < Animal {
                speak() { return "woof"; }
            }
            class Cat < Animal {
                speak() { return "meow"; }
            }
            class Fish < Animal {}
            let animals = [Dog(), Cat(), Fish()];
            for (a in animals) {
                print a.speak();
            }
        """)
        assert out == ["woof", "meow", "..."]


# ============================================================
# Error handling with classes
# ============================================================

class TestErrorHandling:
    def test_throw_in_method(self):
        r, out = run("""
            class Validator {
                init(max) { this.max = max; }
                check(val) {
                    if (val > this.max) {
                        throw "too large";
                    }
                    return val;
                }
            }
            let v = Validator(10);
            try {
                print v.check(5);
                print v.check(15);
            } catch (e) {
                print e;
            }
        """)
        assert out == ["5", "too large"]

    def test_catch_in_method(self):
        r, out = run("""
            class Safe {
                run(fn_val) {
                    try {
                        return fn_val();
                    } catch (e) {
                        return "caught: " + e;
                    }
                }
            }
            let s = Safe();
            print s.run(fn() { return "ok"; });
            print s.run(fn() { throw "boom"; });
        """)
        assert out == ["ok", "caught: boom"]


# ============================================================
# Advanced features
# ============================================================

class TestAdvanced:
    def test_method_returning_new_instance(self):
        r, out = run("""
            class Vec {
                init(x, y) { this.x = x; this.y = y; }
                add(other) {
                    return Vec(this.x + other.x, this.y + other.y);
                }
                toString() {
                    return f"(${string(this.x)}, ${string(this.y)})";
                }
            }
            let v1 = Vec(1, 2);
            let v2 = Vec(3, 4);
            let v3 = v1.add(v2);
            print v3.toString();
        """)
        assert out == ["(4, 6)"]

    def test_recursive_class_method(self):
        r, out = run("""
            class Factorial {
                compute(n) {
                    if (n <= 1) { return 1; }
                    return n * this.compute(n - 1);
                }
            }
            print Factorial().compute(5);
        """)
        assert out == ["120"]

    def test_class_with_while_loop(self):
        r, out = run("""
            class Counter {
                init(n) { this.n = n; }
                countdown() {
                    let result = [];
                    let i = this.n;
                    while (i > 0) {
                        push(result, i);
                        i = i - 1;
                    }
                    return result;
                }
            }
            print Counter(5).countdown();
        """)
        assert out == ["[5, 4, 3, 2, 1]"]

    def test_class_instances_as_values(self):
        r, out = run("""
            class Node {
                init(val, nxt) {
                    this.val = val;
                    this.nxt = nxt;
                }
            }
            let list = Node(1, Node(2, Node(3, false)));
            let current = list;
            let result = [];
            while (current != false) {
                push(result, current.val);
                current = current.nxt;
            }
            print result;
        """)
        assert out == ["[1, 2, 3]"]

    def test_class_method_with_default_like_behavior(self):
        r, out = run("""
            class Config {
                init() { this.opts = {}; }
                set(key, val) { this.opts[key] = val; return this; }
                get(key, default_val) {
                    if (has(this.opts, key)) { return this.opts[key]; }
                    return default_val;
                }
            }
            let c = Config();
            c.set("a", 1);
            print c.get("a", 0);
            print c.get("b", 42);
        """)
        assert out == ["1", "42"]

    def test_two_classes_interacting(self):
        r, out = run("""
            class Engine {
                init(hp) { this.hp = hp; }
                describe() { return f"${string(this.hp)}hp"; }
            }
            class Car {
                init(name, engine) {
                    this.name = name;
                    this.engine = engine;
                }
                describe() {
                    return f"${this.name} with ${this.engine.describe()}";
                }
            }
            let e = Engine(200);
            let c = Car("Tesla", e);
            print c.describe();
        """)
        assert out == ["Tesla with 200hp"]

    def test_class_equality(self):
        """Instances with same __class__ and props compare equal (dict equality)."""
        r, out = run("""
            class Foo { init(x) { this.x = x; } }
            let a = Foo(1);
            let b = Foo(1);
            let c = a;
            print a == b;
            print a == c;
            let d = Foo(2);
            print a == d;
        """)
        # Instances are dicts, Python dict eq compares content
        assert out == ["true", "true", "false"]


# ============================================================
# Super edge cases
# ============================================================

class TestSuperEdgeCases:
    def test_super_in_non_init_method(self):
        r, out = run("""
            class Base {
                describe() { return "base"; }
            }
            class Child < Base {
                describe() { return "child:" + super.describe(); }
            }
            print Child().describe();
        """)
        assert out == ["child:base"]

    def test_super_with_args(self):
        r, out = run("""
            class Base {
                compute(a, b) { return a + b; }
            }
            class Child < Base {
                compute(a, b) { return super.compute(a, b) * 2; }
            }
            print Child().compute(3, 4);
        """)
        assert out == ["14"]

    def test_super_skips_to_parent(self):
        """Super always goes to the parent of the class defining the method."""
        r, out = run("""
            class A {
                who() { return "A"; }
            }
            class B < A {
                who() { return "B+" + super.who(); }
            }
            class C < B {}
            print C().who();
        """)
        # C has no who(), so B.who() is called. B.who's super is A.
        assert out == ["B+A"]


# ============================================================
# Backward compatibility -- all existing features still work
# ============================================================

class TestBackwardCompat:
    def test_closures(self):
        r, out = run("""
            fn apply(f, x) { return f(x); }
            let double = fn(x) { return x * 2; };
            print apply(double, 21);
        """)
        assert out == ["42"]

    def test_generators(self):
        r, out = run("""
            fn* count(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            let g = count(3);
            print next(g);
            print next(g);
            print next(g);
        """)
        assert out == ["0", "1", "2"]

    def test_destructuring(self):
        r, out = run("""
            let [a, b, ...rest] = [1, 2, 3, 4, 5];
            print a;
            print b;
            print rest;
        """)
        assert out == ["1", "2", "[3, 4, 5]"]

    def test_spread(self):
        r, out = run("""
            let a = [1, 2];
            let b = [0, ...a, 3];
            print b;
        """)
        assert out == ["[0, 1, 2, 3]"]

    def test_pipe(self):
        r, out = run("""
            fn double(x) { return x * 2; }
            fn inc(x) { return x + 1; }
            print 5 |> double |> inc;
        """)
        assert out == ["11"]

    def test_fstring(self):
        r, out = run("""
            let name = "World";
            print f"Hello ${name}!";
        """)
        assert out == ["Hello World!"]

    def test_for_in(self):
        r, out = run("""
            let sum = 0;
            for (x in [1, 2, 3, 4, 5]) {
                sum = sum + x;
            }
            print sum;
        """)
        assert out == ["15"]

    def test_modules(self):
        reg = ModuleRegistry()
        reg.register("math", """
            export fn add(a, b) { return a + b; }
        """)
        r, out = run("""
            import { add } from "math";
            print add(3, 4);
        """, registry=reg)
        assert out == ["7"]

    def test_error_handling(self):
        r, out = run("""
            try {
                throw "oops";
            } catch (e) {
                print e;
            }
        """)
        assert out == ["oops"]

    def test_hash_maps(self):
        r, out = run("""
            let h = {a: 1, b: 2};
            print h.a;
            print keys(h);
        """)
        assert out == ["1", "[a, b]"]

    def test_pattern_matching_keywords(self):
        """Ensure 'class' and 'super' are keywords, not identifiers."""
        with pytest.raises(ParseError):
            run("let class = 5;")


# ============================================================
# Misc integration
# ============================================================

class TestIntegration:
    def test_class_in_hash(self):
        r, out = run("""
            class Foo { init(x) { this.x = x; } }
            let h = {cls: Foo};
            let f = h.cls(42);
            print f.x;
        """)
        assert out == ["42"]

    def test_class_in_array(self):
        r, out = run("""
            class A { greet() { return "A"; } }
            class B { greet() { return "B"; } }
            let classes = [A, B];
            for (cls in classes) {
                print cls().greet();
            }
        """)
        assert out == ["A", "B"]

    def test_method_chaining(self):
        r, out = run("""
            class Query {
                init() { this.parts = []; }
                sel(col) { push(this.parts, "SELECT " + col); return this; }
                tbl(name) { push(this.parts, "FROM " + name); return this; }
                build() {
                    let result = "";
                    for (p in this.parts) {
                        if (result != "") { result = result + " "; }
                        result = result + p;
                    }
                    return result;
                }
            }
            print Query().sel("*").tbl("users").build();
        """)
        assert out == ["SELECT * FROM users"]

    def test_observer_pattern(self):
        r, out = run("""
            class EventEmitter {
                init() { this.listeners = []; }
                on(fn_val) { push(this.listeners, fn_val); }
                emit(data) {
                    for (listener in this.listeners) {
                        listener(data);
                    }
                }
            }
            let emitter = EventEmitter();
            emitter.on(fn(d) { print f"Got: ${d}"; });
            emitter.on(fn(d) { print f"Also: ${d}"; });
            emitter.emit("hello");
        """)
        assert out == ["Got: hello", "Also: hello"]

    def test_false_as_property(self):
        r, out = run("""
            class Node {
                init(val) {
                    this.val = val;
                    this.left = false;
                    this.right = false;
                }
            }
            let n = Node(42);
            print n.val;
            print n.left;
            print n.right;
        """)
        assert out == ["42", "false", "false"]

    def test_class_string_representation(self):
        r, out = run("""
            class Point {
                init(x, y) { this.x = x; this.y = y; }
            }
            let p = Point(1, 2);
            print string(p);
        """)
        assert "<Point" in out[0]


# ============================================================
# Stress tests / additional coverage
# ============================================================

class TestStress:
    def test_many_instances_in_loop(self):
        r, out = run("""
            class Counter {
                init(n) { this.n = n; }
                val() { return this.n; }
            }
            let total = 0;
            let i = 0;
            while (i < 10) {
                let c = Counter(i);
                total = total + c.val();
                i = i + 1;
            }
            print total;
        """)
        assert out == ["45"]

    def test_method_call_in_for_in(self):
        """The stack corruption bug fix must work."""
        r, out = run("""
            class Foo {
                init(x) { this.x = x; }
                get() { return this.x; }
            }
            let items = [Foo(10), Foo(20), Foo(30)];
            let total = 0;
            for (item in items) {
                total = total + item.get();
            }
            print total;
        """)
        assert out == ["60"]

    def test_inheritance_in_for_in(self):
        r, out = run("""
            class Shape { area() { return 0; } }
            class Rect < Shape {
                init(w, h) { this.w = w; this.h = h; }
                area() { return this.w * this.h; }
            }
            class Circle < Shape {
                init(r) { this.r = r; }
                area() { return this.r * this.r * 3; }
            }
            let shapes = [Rect(3, 4), Circle(5)];
            for (s in shapes) {
                print s.area();
            }
        """)
        assert out == ["12", "75"]

    def test_deep_super_chain(self):
        r, out = run("""
            class A { greet() { return "A"; } }
            class B < A { greet() { return super.greet() + "B"; } }
            class C < B { greet() { return super.greet() + "C"; } }
            class D < C { greet() { return super.greet() + "D"; } }
            print D().greet();
        """)
        assert out == ["ABCD"]

    def test_constructor_chain(self):
        r, out = run("""
            class A {
                init(x) { this.x = x; }
            }
            class B < A {
                init(x, y) { super.init(x); this.y = y; }
            }
            class C < B {
                init(x, y, z) { super.init(x, y); this.z = z; }
            }
            class D < C {
                init(x, y, z, w) { super.init(x, y, z); this.w = w; }
            }
            let d = D(1, 2, 3, 4);
            print d.x;
            print d.y;
            print d.z;
            print d.w;
        """)
        assert out == ["1", "2", "3", "4"]

    def test_method_returning_this(self):
        r, out = run("""
            class Fluent {
                init() { this.vals = []; }
                add(v) { push(this.vals, v); return this; }
                result() { return this.vals; }
            }
            print Fluent().add(1).add(2).add(3).result();
        """)
        assert out == ["[1, 2, 3]"]

    def test_class_with_spread_args(self):
        r, out = run("""
            class Pair {
                init(a, b) { this.a = a; this.b = b; }
                sum() { return this.a + this.b; }
            }
            let args = [3, 4];
            let p = Pair(...args);
            print p.sum();
        """)
        assert out == ["7"]

    def test_class_method_with_destructuring(self):
        r, out = run("""
            class Math {
                sumPair([a, b]) { return a + b; }
            }
            print Math().sumPair([3, 4]);
        """)
        assert out == ["7"]

    def test_instanceof_chain(self):
        r, out = run("""
            class A {}
            class B < A {}
            class C < B {}
            let c = C();
            print instanceof(c, A);
            print instanceof(c, B);
            print instanceof(c, C);
        """)
        assert out == ["true", "true", "true"]

    def test_class_with_map_filter(self):
        r, out = run("""
            class Box {
                init(val) { this.val = val; }
                get() { return this.val; }
            }
            let boxes = [Box(1), Box(2), Box(3)];
            let vals = map(boxes, fn(b) { return b.get(); });
            print vals;
        """)
        assert out == ["[1, 2, 3]"]

    def test_export_and_import_class(self):
        reg = ModuleRegistry()
        reg.register("animals", """
            export class Animal {
                init(name) { this.name = name; }
                speak() { return this.name + " speaks"; }
            }
            export class Dog < Animal {
                init(name) { super.init(name); }
                speak() { return this.name + " barks"; }
            }
        """)
        r, out = run("""
            import { Animal, Dog } from "animals";
            print Animal("Cat").speak();
            print Dog("Rex").speak();
        """, registry=reg)
        assert out == ["Cat speaks", "Rex barks"]


# ============================================================
# Run all
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
