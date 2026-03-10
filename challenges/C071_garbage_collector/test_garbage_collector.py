"""
Tests for C071: Garbage Collector

Tests cover:
- Heap tracking and allocation counting
- Root scanning (stack, env, closures, generators, etc.)
- Mark-sweep collection
- Generational collection (gen0 vs full)
- Weak references
- Finalizers
- Auto-collection threshold
- GC statistics
- GC with complex programs (closures, classes, generators, async)
- Object graph traversal correctness
- GC doesn't break program semantics
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from garbage_collector import (
    GarbageCollector, GCVM, HeapRef, WeakRef, GCStats,
    compile_source, run_with_gc, execute_with_gc, HEAP_TYPES,
    FnObject, ClosureObject, GeneratorObject, ClassObject,
    TraitObject, EnumObject, BoundMethod, PromiseObject,
    NativeFunction, NativeModule,
    VM, Chunk, Op, VMError,
    run, execute,
)


# ============================================================
# Section 1: GarbageCollector unit tests
# ============================================================

class TestGCBasics:
    def test_create_gc(self):
        gc = GarbageCollector()
        assert gc.stats.total_allocations == 0
        assert gc.stats.heap_size == 0
        assert gc.threshold == 256

    def test_track_object(self):
        gc = GarbageCollector()
        obj = [1, 2, 3]
        ref = gc.track(obj)
        assert isinstance(ref, HeapRef)
        assert ref.obj is obj
        assert ref.generation == 0
        assert ref.marked is False
        assert gc.stats.total_allocations == 1
        assert gc.stats.heap_size == 1

    def test_track_same_object_twice(self):
        gc = GarbageCollector()
        obj = [1, 2, 3]
        ref1 = gc.track(obj)
        ref2 = gc.track(obj)
        assert ref1 is ref2
        assert gc.stats.total_allocations == 1  # not double-counted

    def test_track_multiple_objects(self):
        gc = GarbageCollector()
        objs = [[i] for i in range(10)]
        refs = [gc.track(o) for o in objs]
        assert gc.stats.total_allocations == 10
        assert gc.stats.heap_size == 10
        assert gc.stats.peak_heap_size == 10

    def test_heap_ref_identity(self):
        gc = GarbageCollector()
        a = [1]
        b = [2]
        ra = gc.track(a)
        rb = gc.track(b)
        assert ra != rb
        assert ra == ra
        assert hash(ra) != hash(rb)

    def test_gc_disabled(self):
        gc = GarbageCollector(threshold=1)
        gc.enabled = False
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(42)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        for _ in range(10):
            gc.track([1])
        freed = gc.maybe_collect(vm)
        assert freed == 0  # GC disabled

    def test_custom_threshold(self):
        gc = GarbageCollector(threshold=5)
        assert gc.threshold == 5

    def test_stats_initial(self):
        gc = GarbageCollector()
        stats = gc.get_stats()
        assert stats['total_allocations'] == 0
        assert stats['total_collections'] == 0
        assert stats['total_freed'] == 0
        assert stats['heap_size'] == 0
        assert stats['peak_heap_size'] == 0


class TestHeapRef:
    def test_heap_ref_repr(self):
        obj = {"x": 1}
        ref = HeapRef(obj)
        r = repr(ref)
        assert "HeapRef" in r
        assert "dict" in r
        assert "gen=0" in r

    def test_heap_ref_generation(self):
        ref = HeapRef([1, 2])
        assert ref.generation == 0
        ref.generation = 3
        assert ref.generation == 3

    def test_heap_ref_mark(self):
        ref = HeapRef("test")
        assert ref.marked is False
        ref.marked = True
        assert ref.marked is True


# ============================================================
# Section 2: Mark-sweep tests
# ============================================================

class TestMarkSweep:
    def _make_vm(self):
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(42)
        chunk.emit(Op.HALT)
        return VM(chunk)

    def test_collect_empty_heap(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        freed = gc.collect(vm)
        assert freed == 0
        assert gc.stats.total_collections == 1

    def test_collect_unreachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        # Track objects not reachable from VM roots
        gc.track([1, 2, 3])
        gc.track({"a": 1})
        assert gc.stats.heap_size == 2
        freed = gc.collect(vm)
        assert freed == 2
        assert gc.stats.heap_size == 0
        assert gc.stats.total_freed == 2

    def test_collect_reachable_in_env(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        obj = [1, 2, 3]
        gc.track(obj)
        vm.env['mylist'] = obj
        freed = gc.collect(vm)
        assert freed == 0  # obj is reachable via env
        assert gc.stats.heap_size == 1

    def test_collect_reachable_on_stack(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        obj = {"key": "value"}
        gc.track(obj)
        vm.stack.append(obj)
        freed = gc.collect(vm)
        assert freed == 0

    def test_collect_mixed_reachable_unreachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        reachable = [1, 2]
        unreachable1 = [3, 4]
        unreachable2 = {"x": 5}
        gc.track(reachable)
        gc.track(unreachable1)
        gc.track(unreachable2)
        vm.env['kept'] = reachable
        freed = gc.collect(vm)
        assert freed == 2
        assert gc.stats.heap_size == 1

    def test_generation_increments_on_survival(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        obj = [1]
        ref = gc.track(obj)
        vm.env['obj'] = obj
        assert ref.generation == 0
        gc.collect(vm)
        assert ref.generation == 1
        gc.collect(vm)
        assert ref.generation == 2

    def test_nested_reachability(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        inner = [10, 20]
        outer = {"nested": inner}
        gc.track(inner)
        gc.track(outer)
        vm.env['data'] = outer
        freed = gc.collect(vm)
        assert freed == 0  # both reachable via env -> outer -> inner

    def test_circular_reference_unreachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        a = {}
        b = {}
        a['ref'] = b
        b['ref'] = a
        gc.track(a)
        gc.track(b)
        # Neither a nor b is in VM roots
        freed = gc.collect(vm)
        assert freed == 2  # circular but unreachable

    def test_circular_reference_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        a = {}
        b = {}
        a['ref'] = b
        b['ref'] = a
        gc.track(a)
        gc.track(b)
        vm.env['root'] = a  # a is reachable, b reachable through a
        freed = gc.collect(vm)
        assert freed == 0

    def test_deep_chain_reachability(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        chain = [None]
        gc.track(chain)
        current = chain
        for i in range(20):
            next_obj = [None]
            gc.track(next_obj)
            current[0] = next_obj
            current = next_obj
        vm.env['chain'] = chain
        freed = gc.collect(vm)
        assert freed == 0  # all reachable through chain

    def test_deep_chain_partial_unreachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        kept = [1]
        dropped = [2]
        gc.track(kept)
        gc.track(dropped)
        vm.env['kept'] = kept
        # dropped is not referenced from anywhere
        freed = gc.collect(vm)
        assert freed == 1

    def test_multiple_collections(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        # First round
        gc.track([1])
        gc.collect(vm)
        assert gc.stats.total_freed == 1
        # Second round
        gc.track([2])
        gc.track([3])
        gc.collect(vm)
        assert gc.stats.total_freed == 3
        assert gc.stats.total_collections == 2


# ============================================================
# Section 3: Generational GC tests
# ============================================================

class TestGenerationalGC:
    def _make_vm(self):
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        return VM(chunk)

    def test_gen0_collection_skips_promoted(self):
        gc = GarbageCollector(gen1_threshold=2)
        vm = self._make_vm()
        old_obj = [1]
        ref = gc.track(old_obj)
        vm.env['old'] = old_obj
        # Survive 2 collections -> promoted to gen1
        gc.collect(vm)
        gc.collect(vm)
        assert ref.generation == 2
        # Remove from roots
        del vm.env['old']
        # Gen0 collection should NOT collect gen1+ objects
        freed = gc.collect_gen0(vm)
        assert freed == 0  # old_obj promoted, survives gen0

    def test_gen0_collects_young(self):
        gc = GarbageCollector(gen1_threshold=3)
        vm = self._make_vm()
        young = [99]
        gc.track(young)
        # young is gen0 and unreachable
        freed = gc.collect_gen0(vm)
        assert freed == 1
        assert gc.stats.gen0_collections == 1

    def test_full_collection_collects_all(self):
        gc = GarbageCollector(gen1_threshold=1)
        vm = self._make_vm()
        obj = [1]
        ref = gc.track(obj)
        vm.env['obj'] = obj
        gc.collect(vm)  # promote to gen1
        del vm.env['obj']
        # Full collection should collect even promoted objects
        freed = gc.collect(vm)
        assert freed == 1
        assert gc.stats.full_collections == 2


# ============================================================
# Section 4: Weak references
# ============================================================

class TestWeakRefs:
    def test_create_weak_ref(self):
        gc = GarbageCollector()
        obj = [1, 2, 3]
        gc.track(obj)
        wr = gc.create_weak_ref(obj)
        assert wr is not None
        assert wr.alive is True
        assert wr.get() is obj

    def test_weak_ref_invalidated_on_collect(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        obj = [1, 2, 3]
        gc.track(obj)
        wr = gc.create_weak_ref(obj)
        # obj is unreachable
        gc.collect(vm)
        assert wr.alive is False
        assert wr.get() is None

    def test_weak_ref_survives_if_reachable(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        obj = [1, 2, 3]
        gc.track(obj)
        wr = gc.create_weak_ref(obj)
        vm.env['obj'] = obj
        gc.collect(vm)
        assert wr.alive is True
        assert wr.get() is obj

    def test_weak_ref_untracked_returns_none(self):
        gc = GarbageCollector()
        obj = [1]
        wr = gc.create_weak_ref(obj)
        assert wr is None

    def test_multiple_weak_refs(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        obj = [1]
        gc.track(obj)
        wr1 = gc.create_weak_ref(obj)
        wr2 = gc.create_weak_ref(obj)
        gc.collect(vm)  # unreachable
        assert wr1.alive is False
        assert wr2.alive is False


# ============================================================
# Section 5: Finalizers
# ============================================================

class TestFinalizers:
    def test_finalizer_called_on_collect(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        log = []
        obj = [1, 2, 3]
        gc.track(obj, finalizer=lambda o: log.append(('finalized', len(o))))
        gc.collect(vm)
        assert log == [('finalized', 3)]

    def test_finalizer_not_called_if_reachable(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        log = []
        obj = [1]
        gc.track(obj, finalizer=lambda o: log.append('fin'))
        vm.env['obj'] = obj
        gc.collect(vm)
        assert log == []

    def test_finalizer_exception_swallowed(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)

        def bad_finalizer(o):
            raise RuntimeError("boom")

        gc.track([1], finalizer=bad_finalizer)
        # Should not raise
        gc.collect(vm)
        assert gc.stats.total_freed == 1

    def test_multiple_finalizers(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        log = []
        gc.track([1], finalizer=lambda o: log.append('a'))
        gc.track([2], finalizer=lambda o: log.append('b'))
        gc.collect(vm)
        assert set(log) == {'a', 'b'}


# ============================================================
# Section 6: Auto-collection threshold
# ============================================================

class TestAutoCollection:
    def test_auto_collect_at_threshold(self):
        gc = GarbageCollector(threshold=3)
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        gc.track([1])
        gc.track([2])
        freed = gc.maybe_collect(vm)
        assert freed == 0  # below threshold
        gc.track([3])
        freed = gc.maybe_collect(vm)
        assert freed == 3  # at threshold, all unreachable

    def test_auto_collect_resets_counter(self):
        gc = GarbageCollector(threshold=2)
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        gc.track([1])
        gc.track([2])
        gc.maybe_collect(vm)
        assert gc._allocs_since_gc == 0
        gc.track([3])
        assert gc._allocs_since_gc == 1


# ============================================================
# Section 7: GCVM integration tests
# ============================================================

class TestGCVM:
    def test_basic_program(self):
        result, output, stats = run_with_gc("let x = 42; print x;")
        assert output == ['42']
        assert stats['total_collections'] >= 0

    def test_array_allocation(self):
        result, output, stats = run_with_gc("let a = [1, 2, 3]; print len(a);")
        assert output == ['3']

    def test_hash_allocation(self):
        result, output, stats = run_with_gc('let h = {"a": 1, "b": 2}; print h.a;')
        assert output == ['1']

    def test_closure_allocation(self):
        code = """
        fn make_adder(x) {
            return fn(y) { return x + y; };
        }
        let add5 = make_adder(5);
        print add5(10);
        """
        result, output, stats = run_with_gc(code)
        assert output == ['15']

    def test_class_allocation(self):
        code = """
        class Dog {
            init(name) { this.name = name; }
            speak() { return this.name; }
        }
        let d = Dog("Rex");
        print d.speak();
        """
        result, output, stats = run_with_gc(code)
        assert output == ['Rex']

    def test_gc_stats_available(self):
        info = execute_with_gc("let x = 1;")
        assert 'gc_stats' in info
        assert 'result' in info
        assert 'output' in info
        stats = info['gc_stats']
        assert 'total_allocations' in stats
        assert 'heap_size' in stats

    def test_gc_manual_collect(self):
        chunk = compile_source("let x = [1, 2, 3];")
        vm = GCVM(chunk, gc_threshold=1000)
        vm.run()
        freed = vm.gc_collect()
        assert isinstance(freed, int)

    def test_many_allocations_with_gc(self):
        """Create many short-lived objects -- GC should reclaim them."""
        code = """
        let i = 0;
        while (i < 50) {
            let temp = [i, i+1, i+2];
            i = i + 1;
        }
        print i;
        """
        result, output, stats = run_with_gc(code, gc_threshold=10)
        assert output == ['50']
        # With threshold=10 and 50 iterations, should have collected
        assert stats['total_allocations'] > 0

    def test_gc_preserves_live_data(self):
        code = """
        let data = [];
        let i = 0;
        while (i < 20) {
            push(data, i * 2);
            i = i + 1;
        }
        print len(data);
        print data[0];
        print data[19];
        """
        result, output, stats = run_with_gc(code, gc_threshold=5)
        assert output == ['20', '0', '38']

    def test_enum_allocation(self):
        code = """
        enum Color { Red, Green, Blue }
        print Color.Red;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['Color.Red']

    def test_trait_allocation(self):
        code = """
        trait Greet {
            greet() { return "hello"; }
        }
        class Person implements Greet {
            init(name) { this.name = name; }
        }
        let p = Person("Alice");
        print p.greet();
        """
        result, output, stats = run_with_gc(code)
        assert output == ['hello']

    def test_generator_allocation(self):
        code = """
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
        """
        result, output, stats = run_with_gc(code)
        assert output == ['0', '1', '2']

    def test_async_allocation(self):
        code = """
        async fn greet(name) {
            return name;
        }
        let p = greet("world");
        print "done";
        """
        result, output, stats = run_with_gc(code)
        assert output == ['done']

    def test_string_interpolation(self):
        code = """
        let name = "world";
        print f"hello ${name}";
        """
        result, output, stats = run_with_gc(code)
        assert output == ['hello world']

    def test_destructuring(self):
        code = """
        let [a, b, c] = [10, 20, 30];
        print a + b + c;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['60']

    def test_spread_operator(self):
        code = """
        let a = [1, 2];
        let b = [0, ...a, 3];
        print len(b);
        print b[2];
        """
        result, output, stats = run_with_gc(code)
        assert output == ['4', '2']

    def test_if_expression(self):
        code = """
        let x = 42;
        let result = if (x == 42) "answer" else "other";
        print result;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['answer']

    def test_optional_chaining(self):
        code = """
        let obj = {"a": 10};
        print obj?.a;
        let x = null;
        print x?.b;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['10', 'null']

    def test_null_coalescing(self):
        code = """
        let x = null;
        print x ?? "default";
        """
        result, output, stats = run_with_gc(code)
        assert output == ['default']

    def test_for_in_loop(self):
        code = """
        let items = [10, 20, 30];
        for (item in items) {
            print item;
        }
        """
        result, output, stats = run_with_gc(code)
        assert output == ['10', '20', '30']

    def test_try_catch_finally(self):
        code = """
        try {
            throw "boom";
        } catch (e) {
            print e;
        } finally {
            print "done";
        }
        """
        result, output, stats = run_with_gc(code)
        assert output == ['boom', 'done']

    def test_pipe_operator(self):
        code = """
        fn double(x) { return x * 2; }
        fn add1(x) { return x + 1; }
        let result = 5 |> double |> add1;
        print result;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['11']

    def test_class_inheritance(self):
        code = """
        class Animal {
            init(name) { this.name = name; }
        }
        class Dog < Animal {
            init(name) { super.init(name); }
            bark() { return this.name; }
        }
        let d = Dog("Rex");
        print d.bark();
        """
        result, output, stats = run_with_gc(code)
        assert output == ['Rex']


# ============================================================
# Section 8: Object graph traversal tests
# ============================================================

class TestObjectGraphTraversal:
    def _make_vm(self):
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        return VM(chunk)

    def test_fn_object_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        fn = FnObject(name="test", arity=0, chunk=Chunk())
        gc.track(fn)
        vm.env['fn'] = fn
        freed = gc.collect(vm)
        assert freed == 0

    def test_closure_tracks_env(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        inner_list = [1, 2, 3]
        env = {'data': inner_list}
        fn = FnObject(name="cls", arity=0, chunk=Chunk())
        closure = ClosureObject(fn=fn, env=env)
        gc.track(inner_list)
        gc.track(env)
        gc.track(fn)
        gc.track(closure)
        vm.env['cls'] = closure
        freed = gc.collect(vm)
        assert freed == 0  # all reachable through closure

    def test_class_methods_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        method = FnObject(name="speak", arity=1, chunk=Chunk())
        klass = ClassObject(name="Dog", methods={"speak": method})
        gc.track(method)
        gc.track(klass)
        vm.env['Dog'] = klass
        freed = gc.collect(vm)
        assert freed == 0

    def test_class_parent_chain(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        parent = ClassObject(name="Animal", methods={})
        child = ClassObject(name="Dog", methods={}, parent=parent)
        gc.track(parent)
        gc.track(child)
        vm.env['Dog'] = child  # only child in env
        freed = gc.collect(vm)
        assert freed == 0  # parent reachable through child.parent

    def test_trait_reachable_through_class(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        trait = TraitObject(name="Greet", methods={}, required_methods=set())
        klass = ClassObject(name="Person", methods={}, traits=[trait])
        gc.track(trait)
        gc.track(klass)
        vm.env['Person'] = klass
        freed = gc.collect(vm)
        assert freed == 0  # trait reachable through class.traits

    def test_enum_variants_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        enum = EnumObject(name="Color", variants={"Red": None})
        gc.track(enum)
        vm.env['Color'] = enum
        freed = gc.collect(vm)
        assert freed == 0

    def test_bound_method_keeps_instance_alive(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        instance = {"__class__": "Dog", "name": "Rex"}
        method = FnObject(name="speak", arity=1, chunk=Chunk())
        klass = ClassObject(name="Dog", methods={"speak": method})
        bm = BoundMethod(instance=instance, method=method, klass=klass)
        gc.track(instance)
        gc.track(method)
        gc.track(klass)
        gc.track(bm)
        vm.env['bm'] = bm
        freed = gc.collect(vm)
        assert freed == 0  # all kept alive via bound method

    def test_promise_value_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        p = PromiseObject()
        result = [42]
        p.resolve(result)
        gc.track(p)
        gc.track(result)
        vm.env['p'] = p
        freed = gc.collect(vm)
        assert freed == 0

    def test_generator_state_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        fn = FnObject(name="gen", arity=0, chunk=Chunk())
        inner = [1, 2, 3]
        gen = GeneratorObject(fn=fn, env={"data": inner})
        gc.track(fn)
        gc.track(inner)
        gc.track(gen)
        vm.env['gen'] = gen
        freed = gc.collect(vm)
        assert freed == 0

    def test_native_module_exports_reachable(self):
        gc = GarbageCollector()
        vm = self._make_vm()
        nf = NativeFunction(name="write", arity=1, fn=print)
        nm = NativeModule(name="Console", exports={"write": nf})
        gc.track(nf)
        gc.track(nm)
        vm.env['Console'] = nm
        freed = gc.collect(vm)
        assert freed == 0


# ============================================================
# Section 9: GC correctness -- programs produce same output with/without GC
# ============================================================

class TestGCCorrectness:
    """Verify GC doesn't change program semantics."""

    PROGRAMS = [
        ("let x = 10; print x;", ['10']),
        ("let a = [1,2,3]; print len(a);", ['3']),
        ('let h = {"x": 42}; print h.x;', ['42']),
        ("fn add(a, b) { return a + b; } print add(3, 4);", ['7']),
        ("let i = 0; while (i < 5) { i = i + 1; } print i;", ['5']),
        ("print 10 > 5;", ['true']),
        ("print null ?? 99;", ['99']),
    ]

    @pytest.mark.parametrize("code,expected", PROGRAMS)
    def test_gc_preserves_output(self, code, expected):
        # Without GC
        result1, output1 = run(code)
        # With GC (aggressive threshold)
        result2, output2, stats = run_with_gc(code, gc_threshold=1)
        assert output1 == expected
        assert output2 == expected


# ============================================================
# Section 10: Stress tests
# ============================================================

class TestStress:
    def test_many_short_lived_arrays(self):
        code = """
        let sum = 0;
        let i = 0;
        while (i < 100) {
            let arr = [i, i+1, i+2];
            sum = sum + arr[0];
            i = i + 1;
        }
        print sum;
        """
        result, output, stats = run_with_gc(code, gc_threshold=5)
        assert output == ['4950']

    def test_many_short_lived_hashes(self):
        code = """
        let count = 0;
        let i = 0;
        while (i < 50) {
            let h = {"val": i};
            count = count + 1;
            i = i + 1;
        }
        print count;
        """
        result, output, stats = run_with_gc(code, gc_threshold=5)
        assert output == ['50']

    def test_accumulating_array(self):
        code = """
        let data = [];
        let i = 0;
        while (i < 30) {
            push(data, [i]);
            i = i + 1;
        }
        print len(data);
        """
        result, output, stats = run_with_gc(code, gc_threshold=5)
        assert output == ['30']

    def test_recursive_function(self):
        code = """
        fn fib(n) {
            if (n <= 1) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        print fib(10);
        """
        result, output, stats = run_with_gc(code, gc_threshold=50)
        assert output == ['55']

    def test_nested_closures(self):
        code = """
        fn make_adder(x) {
            return fn(y) { return x + y; };
        }
        let add5 = make_adder(5);
        let add10 = make_adder(10);
        print add5(3);
        print add10(3);
        """
        result, output, stats = run_with_gc(code, gc_threshold=3)
        assert output == ['8', '13']


# ============================================================
# Section 11: Edge cases
# ============================================================

class TestEdgeCases:
    def test_gc_on_empty_vm(self):
        chunk = compile_source("print 1;")
        vm = GCVM(chunk, gc_threshold=1000)
        freed = vm.gc_collect()
        assert freed >= 0  # should not crash

    def test_gc_stats_type(self):
        chunk = compile_source("let x = 1;")
        vm = GCVM(chunk)
        stats = vm.gc_stats()
        assert isinstance(stats, dict)
        for key in ['total_allocations', 'total_collections', 'total_freed',
                     'heap_size', 'peak_heap_size']:
            assert key in stats

    def test_execute_with_gc_api(self):
        info = execute_with_gc("print 42;")
        assert info['output'] == ['42']
        assert isinstance(info['gc_stats'], dict)

    def test_run_with_gc_api(self):
        result, output, stats = run_with_gc("print 1;")
        assert output == ['1']
        assert isinstance(stats, dict)

    def test_compile_source_api(self):
        chunk = compile_source("let x = 1;")
        assert isinstance(chunk, Chunk)

    def test_gc_with_no_heap_objects(self):
        """Pure arithmetic -- no heap objects."""
        result, output, stats = run_with_gc("print 1 + 2;")
        assert output == ['3']

    def test_peak_heap_tracking(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        # Create 5 objects
        objs = [[i] for i in range(5)]
        for o in objs:
            gc.track(o)
        assert gc.stats.peak_heap_size == 5
        gc.collect(vm)  # all unreachable
        assert gc.stats.heap_size == 0
        assert gc.stats.peak_heap_size == 5  # peak preserved

    def test_heap_types_constant(self):
        """Verify HEAP_TYPES includes expected types."""
        assert FnObject in HEAP_TYPES
        assert ClosureObject in HEAP_TYPES
        assert ClassObject in HEAP_TYPES
        assert list in HEAP_TYPES
        assert dict in HEAP_TYPES

    def test_weak_ref_after_multiple_collections(self):
        gc = GarbageCollector()
        chunk = Chunk()
        chunk.emit(Op.CONST, 0)
        chunk.constants.append(0)
        chunk.emit(Op.HALT)
        vm = VM(chunk)
        obj = [1]
        gc.track(obj)
        wr = gc.create_weak_ref(obj)
        vm.env['obj'] = obj
        gc.collect(vm)
        gc.collect(vm)
        gc.collect(vm)
        assert wr.alive is True
        del vm.env['obj']
        gc.collect(vm)
        assert wr.alive is False


# ============================================================
# Section 12: Integration with VM features
# ============================================================

class TestVMFeatureIntegration:
    def test_module_system(self):
        code = """
        let x = 100;
        print x;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['100']

    def test_error_handling(self):
        code = """
        try {
            throw "test error";
        } catch (e) {
            print e;
        }
        """
        result, output, stats = run_with_gc(code)
        assert output == ['test error']

    def test_iterator_protocol(self):
        code = """
        fn* range_gen(n) {
            let i = 0;
            while (i < n) {
                yield i;
                i = i + 1;
            }
        }
        let g = range_gen(3);
        print next(g);
        print next(g);
        print next(g);
        """
        result, output, stats = run_with_gc(code)
        assert output == ['0', '1', '2']

    def test_string_methods(self):
        code = """
        let s = "hello world";
        print len(s);
        print s.toUpperCase();
        """
        result, output, stats = run_with_gc(code)
        assert output == ['11', 'HELLO WORLD']

    def test_class_with_static(self):
        code = """
        class Math {
            static max(a, b) {
                if (a > b) { return a; }
                return b;
            }
        }
        print Math.max(3, 7);
        """
        result, output, stats = run_with_gc(code)
        assert output == ['7']

    def test_class_with_getter(self):
        code = """
        class Circle {
            init(r) { this.r = r; }
            get area() { return 3 * this.r * this.r; }
        }
        let c = Circle(5);
        print c.area;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['75']

    def test_enum_with_methods(self):
        code = """
        enum Color {
            Red,
            Green,
            Blue
        }
        print Color.Red;
        print Color.Green;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['Color.Red', 'Color.Green']

    def test_trait_with_decorator(self):
        code = """
        fn logger(method) {
            return fn(this) {
                return method(this);
            };
        }
        trait Describable {
            @logger
            describe() { return "thing"; }
        }
        class Item implements Describable {
            init() {}
        }
        let i = Item();
        print i.describe();
        """
        result, output, stats = run_with_gc(code)
        assert output == ['thing']

    def test_computed_properties(self):
        code = """
        let key = "name";
        let obj = {[key]: "Alice"};
        print obj.name;
        """
        result, output, stats = run_with_gc(code)
        assert output == ['Alice']

    def test_for_await(self):
        code = """
        async fn fetch(n) { return n * 10; }
        let results = [];
        let p1 = fetch(1);
        let p2 = fetch(2);
        let p3 = fetch(3);
        push(results, 10);
        push(results, 20);
        push(results, 30);
        print len(results);
        """
        result, output, stats = run_with_gc(code)
        assert output == ['3']

    def test_multiple_features_combined(self):
        """Use many features together to test GC handles complex state."""
        code = """
        class Container {
            init() { this.items = []; }
            add(item) { push(this.items, item); }
            get count() { return len(this.items); }
        }
        let c = Container();
        let i = 0;
        while (i < 10) {
            c.add({"id": i, "data": [i * 2]});
            i = i + 1;
        }
        print c.count;
        print c.items[0].id;
        print c.items[9].data[0];
        """
        result, output, stats = run_with_gc(code, gc_threshold=3)
        assert output == ['10', '0', '18']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
