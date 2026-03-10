"""Tests for C060: Enums -- AgentZero Session 061"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from enums import run, execute, ParseError, CompileError, VMError, EnumVariant, EnumObject


# ============================================================
# Basic enum declaration
# ============================================================

class TestBasicEnum:
    def test_simple_enum(self):
        r, out = run('enum Color { Red, Green, Blue } print Color;')
        assert out == ['<enum:Color>']

    def test_enum_variant_access(self):
        r, out = run('enum Color { Red, Green, Blue } print Color.Red;')
        assert out == ['Color.Red']

    def test_enum_multiple_variants(self):
        r, out = run('''
            enum Direction { Up, Down, Left, Right }
            print Direction.Up;
            print Direction.Down;
            print Direction.Left;
            print Direction.Right;
        ''')
        assert out == ['Direction.Up', 'Direction.Down', 'Direction.Left', 'Direction.Right']

    def test_enum_single_variant(self):
        r, out = run('enum Unit { Value } print Unit.Value;')
        assert out == ['Unit.Value']

    def test_enum_stored_in_variable(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.Red;
            print c;
        ''')
        assert out == ['Color.Red']

    def test_enum_trailing_comma(self):
        r, out = run('enum Color { Red, Green, Blue, } print Color.Blue;')
        assert out == ['Color.Blue']


# ============================================================
# Enum ordinals
# ============================================================

class TestEnumOrdinals:
    def test_auto_ordinals(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.Red.ordinal;
            print Color.Green.ordinal;
            print Color.Blue.ordinal;
        ''')
        assert out == ['0', '1', '2']

    def test_explicit_ordinals(self):
        r, out = run('''
            enum Status { Active = 1, Inactive = 0, Pending = 2 }
            print Status.Active.ordinal;
            print Status.Inactive.ordinal;
            print Status.Pending.ordinal;
        ''')
        assert out == ['1', '0', '2']

    def test_mixed_ordinals(self):
        r, out = run('''
            enum Level { Low = 10, Medium, High }
            print Level.Low.ordinal;
            print Level.Medium.ordinal;
            print Level.High.ordinal;
        ''')
        assert out == ['10', '11', '12']

    def test_negative_ordinal(self):
        r, out = run('''
            enum Temp { Cold = -10, Warm = 20, Hot = 40 }
            print Temp.Cold.ordinal;
        ''')
        assert out == ['-10']

    def test_ordinal_continuation_after_explicit(self):
        r, out = run('''
            enum Code { A = 100, B, C, D = 200, E }
            print Code.A.ordinal;
            print Code.B.ordinal;
            print Code.C.ordinal;
            print Code.D.ordinal;
            print Code.E.ordinal;
        ''')
        assert out == ['100', '101', '102', '200', '201']


# ============================================================
# Enum comparison
# ============================================================

class TestEnumComparison:
    def test_equality(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.Red == Color.Red;
            print Color.Red == Color.Blue;
        ''')
        assert out == ['true', 'false']

    def test_inequality(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.Red != Color.Blue;
            print Color.Red != Color.Red;
        ''')
        assert out == ['true', 'false']

    def test_variable_comparison(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let a = Color.Red;
            let b = Color.Red;
            let c = Color.Blue;
            print a == b;
            print a == c;
        ''')
        assert out == ['true', 'false']

    def test_different_enum_comparison(self):
        r, out = run('''
            enum Color { Red }
            enum Shape { Red }
            print Color.Red == Shape.Red;
        ''')
        assert out == ['false']

    def test_enum_not_equal_to_int(self):
        r, out = run('''
            enum Color { Red }
            print Color.Red == 0;
        ''')
        assert out == ['false']

    def test_enum_not_equal_to_string(self):
        r, out = run('''
            enum Color { Red }
            print Color.Red == "Red";
        ''')
        assert out == ['false']


# ============================================================
# Enum variant properties
# ============================================================

class TestEnumVariantProperties:
    def test_name_property(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.Red.name;
            print Color.Blue.name;
        ''')
        assert out == ['Red', 'Blue']

    def test_ordinal_property(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.Green.ordinal;
        ''')
        assert out == ['1']

    def test_enum_ref_via_type(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print type(Color.Red);
        ''')
        assert out == ['Color']

    def test_name_and_ordinal_in_expression(self):
        r, out = run('''
            enum Prio { Low = 1, Medium = 5, High = 10 }
            let p = Prio.High;
            print p.name;
            print p.ordinal;
        ''')
        assert out == ['High', '10']


# ============================================================
# Enum built-in methods
# ============================================================

class TestEnumBuiltinMethods:
    def test_values(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let vals = Color.values();
            print len(vals);
            print vals[0];
            print vals[1];
            print vals[2];
        ''')
        assert out == ['3', 'Color.Red', 'Color.Green', 'Color.Blue']

    def test_name_method(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.name(Color.Red);
            print Color.name(Color.Blue);
        ''')
        assert out == ['Red', 'Blue']

    def test_ordinal_method(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print Color.ordinal(Color.Green);
        ''')
        assert out == ['1']

    def test_from_ordinal(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.from_ordinal(1);
            print c;
            print c == Color.Green;
        ''')
        assert out == ['Color.Green', 'true']

    def test_from_ordinal_explicit(self):
        r, out = run('''
            enum Status { Active = 10, Inactive = 20 }
            print Status.from_ordinal(20);
        ''')
        assert out == ['Status.Inactive']

    def test_from_ordinal_not_found(self):
        with pytest.raises(VMError, match="No variant with ordinal"):
            run('enum Color { Red } Color.from_ordinal(99);')

    def test_name_method_wrong_enum(self):
        with pytest.raises(VMError, match="must be a Color variant"):
            run('''
                enum Color { Red }
                enum Shape { Circle }
                Color.name(Shape.Circle);
            ''')


# ============================================================
# Enum with methods
# ============================================================

class TestEnumMethods:
    def test_simple_method(self):
        r, out = run('''
            enum Direction {
                Up, Down, Left, Right
                fn isVertical() {
                    return this == Direction.Up or this == Direction.Down;
                }
            }
            print Direction.Up.isVertical();
            print Direction.Left.isVertical();
        ''')
        assert out == ['true', 'false']

    def test_method_with_params(self):
        r, out = run('''
            enum Color {
                Red, Green, Blue
                fn matches(other) {
                    return this == other;
                }
            }
            print Color.Red.matches(Color.Red);
            print Color.Red.matches(Color.Blue);
        ''')
        assert out == ['true', 'false']

    def test_method_using_ordinal(self):
        r, out = run('''
            enum Priority {
                Low = 1, Medium = 5, High = 10
                fn isHigherThan(other) {
                    return this.ordinal > other.ordinal;
                }
            }
            print Priority.High.isHigherThan(Priority.Low);
            print Priority.Low.isHigherThan(Priority.High);
        ''')
        assert out == ['true', 'false']

    def test_multiple_methods(self):
        r, out = run('''
            enum Season {
                Spring, Summer, Fall, Winter
                fn isWarm() {
                    return this == Season.Spring or this == Season.Summer;
                }
                fn display() {
                    return this.name;
                }
            }
            print Season.Summer.isWarm();
            print Season.Winter.isWarm();
            print Season.Fall.display();
        ''')
        assert out == ['true', 'false', 'Fall']


# ============================================================
# Enum with if/match patterns
# ============================================================

class TestEnumControlFlow:
    def test_enum_in_if(self):
        r, out = run('''
            enum Light { Red, Yellow, Green }
            let l = Light.Green;
            if (l == Light.Green) { print "go"; }
            if (l == Light.Red) { print "stop"; }
        ''')
        assert out == ['go']

    def test_enum_in_while(self):
        r, out = run('''
            enum State { Running, Stopped }
            let s = State.Running;
            let count = 0;
            while (s == State.Running) {
                count = count + 1;
                if (count == 3) { s = State.Stopped; }
            }
            print count;
        ''')
        assert out == ['3']

    def test_enum_in_function(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            fn describe(c) {
                if (c == Color.Red) { return "red"; }
                if (c == Color.Green) { return "green"; }
                return "blue";
            }
            print describe(Color.Red);
            print describe(Color.Green);
            print describe(Color.Blue);
        ''')
        assert out == ['red', 'green', 'blue']

    def test_enum_in_array(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let colors = [Color.Red, Color.Green, Color.Blue];
            for (c in colors) {
                print c;
            }
        ''')
        assert out == ['Color.Red', 'Color.Green', 'Color.Blue']

    def test_enum_in_hash(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let names = {};
            names[Color.Red] = "red";
            names[Color.Green] = "green";
            print names[Color.Red];
        ''')
        # EnumVariant is hashable so this should work
        assert out == ['red']

    def test_enum_for_in_values(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            for (c in Color.values()) {
                print c.name;
            }
        ''')
        assert out == ['Red', 'Green', 'Blue']


# ============================================================
# Type system integration
# ============================================================

class TestEnumTypeSystem:
    def test_type_of_variant(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print type(Color.Red);
        ''')
        assert out == ['Color']

    def test_type_of_enum(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print type(Color);
        ''')
        assert out == ['enum']

    def test_instanceof_enum(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print instanceof(Color.Red, Color);
            print instanceof(Color.Blue, Color);
        ''')
        assert out == ['true', 'true']

    def test_instanceof_wrong_enum(self):
        r, out = run('''
            enum Color { Red }
            enum Shape { Circle }
            print instanceof(Color.Red, Shape);
        ''')
        assert out == ['false']

    def test_instanceof_non_variant(self):
        r, out = run('''
            enum Color { Red }
            print instanceof(42, Color);
            print instanceof("Red", Color);
        ''')
        assert out == ['false', 'false']

    def test_string_conversion(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print string(Color.Red);
            print string(Color);
        ''')
        assert out == ['Color.Red', '<enum:Color>']


# ============================================================
# Enum in f-strings
# ============================================================

class TestEnumFStrings:
    def test_enum_in_fstring(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.Red;
            print f"color: ${c}";
        ''')
        assert out == ['color: Color.Red']

    def test_enum_property_in_fstring(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            print f"name: ${Color.Red.name}, ord: ${Color.Red.ordinal}";
        ''')
        assert out == ['name: Red, ord: 0']


# ============================================================
# Export enum
# ============================================================

class TestEnumExport:
    def test_export_enum(self):
        from enums import ModuleRegistry
        reg = ModuleRegistry()
        reg.register('colors', '''
            export enum Color { Red, Green, Blue }
        ''')
        r, out = run('''
            import { Color } from "colors";
            print Color.Red;
            print Color.Green;
        ''', registry=reg)
        assert out == ['Color.Red', 'Color.Green']


# ============================================================
# Edge cases and errors
# ============================================================

class TestEnumEdgeCases:
    def test_nonexistent_variant(self):
        with pytest.raises(VMError, match="has no member"):
            run('enum Color { Red } Color.Purple;')

    def test_enum_variant_no_such_property(self):
        with pytest.raises(VMError, match="has no property"):
            run('enum Color { Red } Color.Red.foo;')

    def test_enum_in_let_destructure(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let [a, b] = [Color.Red, Color.Blue];
            print a;
            print b;
        ''')
        assert out == ['Color.Red', 'Color.Blue']

    def test_enum_as_function_arg(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            fn show(c) { print c; }
            show(Color.Green);
        ''')
        assert out == ['Color.Green']

    def test_enum_in_closure(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.Red;
            let f = fn() { return c; };
            print f();
        ''')
        assert out == ['Color.Red']

    def test_multiple_enums(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            enum Shape { Circle, Square, Triangle }
            print Color.Red;
            print Shape.Circle;
        ''')
        assert out == ['Color.Red', 'Shape.Circle']

    def test_enum_reassignment(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.Red;
            c = Color.Blue;
            print c;
        ''')
        assert out == ['Color.Blue']


# ============================================================
# Enum with other language features
# ============================================================

class TestEnumIntegration:
    def test_enum_with_try_catch(self):
        r, out = run('''
            enum Color { Red }
            try {
                Color.from_ordinal(99);
            } catch (e) {
                print "caught";
            }
        ''')
        assert out == ['caught']

    def test_enum_with_spread(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let vals = Color.values();
            let more = [...vals, Color.Red];
            print len(more);
        ''')
        assert out == ['4']

    def test_enum_with_pipe(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            fn show(c) { return c.name; }
            let result = Color.Red |> show;
            print result;
        ''')
        assert out == ['Red']

    def test_enum_with_optional_chaining(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.Red;
            print c?.name;
            let n = null;
            print n?.name;
        ''')
        assert out == ['Red', 'null']

    def test_enum_with_null_coalescing(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = null;
            let result = c ?? Color.Red;
            print result;
        ''')
        assert out == ['Color.Red']

    def test_enum_values_with_map(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let names = Color.values().map(fn(c) { return c.name; });
            print names;
        ''')
        assert out == ['[Red, Green, Blue]']

    def test_enum_values_with_filter(self):
        r, out = run('''
            enum Priority { Low = 1, Medium = 5, High = 10 }
            let high = Priority.values().filter(fn(p) { return p.ordinal >= 5; });
            print len(high);
            print high[0];
            print high[1];
        ''')
        assert out == ['2', 'Priority.Medium', 'Priority.High']

    def test_enum_in_class(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            class Pixel {
                init(x, y, color) {
                    this.x = x;
                    this.y = y;
                    this.color = color;
                }
                describe() {
                    return f"(${this.x},${this.y}):${this.color}";
                }
            }
            let p = Pixel(1, 2, Color.Red);
            print p.describe();
        ''')
        assert out == ['(1,2):Color.Red']

    def test_enum_method_accesses_enum_via_this(self):
        r, out = run('''
            enum Direction {
                North, South, East, West
                fn opposite() {
                    if (this == Direction.North) { return Direction.South; }
                    if (this == Direction.South) { return Direction.North; }
                    if (this == Direction.East) { return Direction.West; }
                    return Direction.East;
                }
            }
            print Direction.North.opposite();
            print Direction.East.opposite();
        ''')
        assert out == ['Direction.South', 'Direction.West']

    def test_enum_as_hash_key(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let m = {};
            m[Color.Red] = "rouge";
            m[Color.Green] = "vert";
            m[Color.Blue] = "bleu";
            for (c in Color.values()) {
                print f"${c.name} = ${m[c]}";
            }
        ''')
        assert out == ['Red = rouge', 'Green = vert', 'Blue = bleu']

    def test_enum_method_chaining(self):
        r, out = run('''
            enum Direction {
                North, South, East, West
                fn opposite() {
                    if (this == Direction.North) { return Direction.South; }
                    if (this == Direction.South) { return Direction.North; }
                    if (this == Direction.East) { return Direction.West; }
                    return Direction.East;
                }
            }
            print Direction.North.opposite().opposite();
        ''')
        assert out == ['Direction.North']

    def test_enum_equality_after_from_ordinal(self):
        r, out = run('''
            enum Color { Red, Green, Blue }
            let c = Color.from_ordinal(0);
            print c == Color.Red;
        ''')
        assert out == ['true']


# ============================================================
# Previous features still work (regression)
# ============================================================

class TestRegression:
    def test_classes_still_work(self):
        r, out = run('''
            class Dog {
                init(name) { this.name = name; }
                bark() { return f"${this.name} says woof"; }
            }
            let d = Dog("Rex");
            print d.bark();
        ''')
        assert out == ['Rex says woof']

    def test_computed_properties_still_work(self):
        r, out = run('''
            let key = "hello";
            let obj = {[key]: 42};
            print obj.hello;
        ''')
        assert out == ['42']

    def test_static_methods_still_work(self):
        r, out = run('''
            class Math {
                static max(a, b) {
                    if (a > b) { return a; }
                    return b;
                }
            }
            print Math.max(3, 7);
        ''')
        assert out == ['7']

    def test_getters_setters_still_work(self):
        r, out = run('''
            class Circle {
                init(r) { this.r = r; }
                get area() { return 3 * this.r * this.r; }
            }
            let c = Circle(5);
            print c.area;
        ''')
        assert out == ['75']

    def test_async_still_works(self):
        r, out = run('''
            async fn greet() { return "hello"; }
            let p = greet();
            let v = await p;
            print v;
        ''')
        assert out == ['hello']

    def test_generators_still_work(self):
        r, out = run('''
            fn* count(n) {
                let i = 0;
                while (i < n) {
                    yield i;
                    i = i + 1;
                }
            }
            for (x in count(3)) { print x; }
        ''')
        assert out == ['0', '1', '2']

    def test_destructuring_still_works(self):
        r, out = run('''
            let [a, b, ...rest] = [1, 2, 3, 4, 5];
            print a;
            print b;
            print rest;
        ''')
        assert out == ['1', '2', '[3, 4, 5]']

    def test_pipe_still_works(self):
        r, out = run('''
            fn double(x) { return x * 2; }
            fn add1(x) { return x + 1; }
            print 5 |> double |> add1;
        ''')
        assert out == ['11']

    def test_optional_chaining_still_works(self):
        r, out = run('''
            let x = null;
            print x?.foo;
        ''')
        assert out == ['null']

    def test_finally_still_works(self):
        r, out = run('''
            try {
                print "try";
            } finally {
                print "finally";
            }
        ''')
        assert out == ['try', 'finally']
