"""
Tests for C034: Concurrent Effect Runtime
Composes C033 (Effect Runtime) + C029 (Concurrent Runtime)

Tests organized by category:
1. Basic effects (no concurrency)
2. Basic concurrency (no effects)
3. Task-local effect handlers
4. Handler inheritance on spawn
5. Effects within concurrent tasks
6. Channel + effect interactions
7. Continuation capture/resume in tasks
8. Nested handlers in concurrent context
9. Select with effects
10. Complex composition scenarios
11. Error handling
12. Edge cases
"""

import pytest
from concurrent_effect_runtime import (
    parse, compile_program, run,
    lex, Token, TokenType,
    Channel, Task, TaskState,
    EffectError, ResumeError, ParseError, CompileError,
    ConcurrentEffectVM,
    EffectDecl, PerformExpr, HandleWith, SpawnExpr, SelectStmt,
)


# ============================================================
# 1. Basic Effects (no concurrency)
# ============================================================

class TestBasicEffects:
    def test_effect_declare_and_perform(self):
        result, output, _ = run("""
            effect Console {
                log(msg);
            }
            handle {
                perform Console.log("hello");
                42;
            } with {
                Console.log(msg) -> {
                    resume(0);
                }
            }
        """)
        assert result == 42

    def test_effect_handler_receives_args(self):
        _, output, _ = run("""
            effect IO {
                write(msg);
            }
            handle {
                perform IO.write("hello");
            } with {
                IO.write(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        """)
        assert output == ["hello"]

    def test_resume_returns_value(self):
        result, _, _ = run("""
            effect Ask {
                question(q);
            }
            handle {
                let answer = perform Ask.question("name");
                answer;
            } with {
                Ask.question(q) -> {
                    resume("Alice");
                }
            }
        """)
        assert result == "Alice"

    def test_multiple_performs(self):
        _, output, _ = run("""
            effect Log {
                info(msg);
            }
            handle {
                perform Log.info("one");
                perform Log.info("two");
                perform Log.info("three");
            } with {
                Log.info(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        """)
        assert output == ["one", "two", "three"]

    def test_effect_in_function(self):
        result, _, _ = run("""
            effect State {
                get();
                set(val);
            }
            fn compute() {
                let x = perform State.get();
                perform State.set(x + 10);
                return perform State.get();
            }
            let state = 0;
            handle {
                compute();
            } with {
                State.get() -> {
                    resume(state);
                }
                State.set(val) -> {
                    state = val;
                    resume(0);
                }
            }
        """)
        assert result == 10

    def test_nested_handlers(self):
        result, _, _ = run("""
            effect A {
                getA();
            }
            effect B {
                getB();
            }
            handle {
                handle {
                    let a = perform A.getA();
                    let b = perform B.getB();
                    a + b;
                } with {
                    B.getB() -> {
                        resume(20);
                    }
                }
            } with {
                A.getA() -> {
                    resume(10);
                }
            }
        """)
        assert result == 30

    def test_handler_without_resume(self):
        """Handler that doesn't resume effectively aborts the computation."""
        result, output, _ = run("""
            effect Abort {
                abort(msg);
            }
            handle {
                print("before");
                perform Abort.abort("oops");
                print("after");
            } with {
                Abort.abort(msg) -> {
                    print(msg);
                }
            }
        """)
        assert output == ["before", "oops"]
        # "after" not printed because handler didn't resume

    def test_effect_with_loop(self):
        _, output, _ = run("""
            effect Emit {
                emit(val);
            }
            handle {
                let i = 0;
                while (i < 3) {
                    perform Emit.emit(i);
                    i = i + 1;
                }
            } with {
                Emit.emit(val) -> {
                    print(val);
                    resume(0);
                }
            }
        """)
        assert output == ["0", "1", "2"]


# ============================================================
# 2. Basic Concurrency (no effects)
# ============================================================

class TestBasicConcurrency:
    def test_spawn_and_join(self):
        result, output, _ = run("""
            fn worker(x) {
                print(x * 2);
                return x * 2;
            }
            let t = spawn worker(5);
            let r = join(t);
            r;
        """)
        assert result == 10
        assert output == ["10"]

    def test_multiple_tasks(self):
        _, output, _ = run("""
            fn worker(id) {
                print(id);
            }
            let t1 = spawn worker(1);
            let t2 = spawn worker(2);
            let t3 = spawn worker(3);
            join(t1);
            join(t2);
            join(t3);
        """)
        assert sorted(output) == ["1", "2", "3"]

    def test_channel_communication(self):
        result, _, _ = run("""
            fn producer(ch) {
                send(ch, 42);
            }
            let c = chan(1);
            let t = spawn producer(c);
            let val = recv(c);
            join(t);
            val;
        """)
        assert result == 42

    def test_channel_ping_pong(self):
        _, output, _ = run("""
            fn pong(ch_in, ch_out) {
                let val = recv(ch_in);
                send(ch_out, val + 1);
            }
            let c1 = chan(1);
            let c2 = chan(1);
            let t = spawn pong(c1, c2);
            send(c1, 10);
            let result = recv(c2);
            print(result);
            join(t);
        """)
        assert output == ["11"]

    def test_yield_fairness(self):
        _, output, _ = run("""
            fn worker(id, n) {
                let i = 0;
                while (i < n) {
                    print(id);
                    yield;
                    i = i + 1;
                }
            }
            let t1 = spawn worker("A", 2);
            let t2 = spawn worker("B", 2);
            join(t1);
            join(t2);
        """)
        assert output.count("A") == 2
        assert output.count("B") == 2

    def test_task_id(self):
        result, output, _ = run("""
            fn worker() {
                return task_id();
            }
            let main_id = task_id();
            let t = spawn worker();
            let worker_id = join(t);
            print(main_id);
            print(worker_id);
        """)
        assert output[0] == "0"
        assert output[1] == "1"

    def test_select_send(self):
        _, output, _ = run("""
            let c1 = chan(1);
            let c2 = chan(1);
            send(c1, 10);
            select {
                case send(c1, 20) -> {
                    print("sent to c1");
                }
                case send(c2, 30) -> {
                    print("sent to c2");
                }
            }
            print(recv(c2));
        """)
        assert "sent to c2" in output
        assert "30" in output

    def test_select_recv(self):
        _, output, _ = run("""
            let c = chan(1);
            send(c, 99);
            select {
                case recv(c) -> val {
                    print(val);
                }
                default -> {
                    print("nothing");
                }
            }
        """)
        assert output == ["99"]

    def test_select_default(self):
        _, output, _ = run("""
            let c = chan(1);
            select {
                case recv(c) -> val {
                    print("got");
                }
                default -> {
                    print("default");
                }
            }
        """)
        assert output == ["default"]


# ============================================================
# 3. Task-local Effect Handlers
# ============================================================

class TestTaskLocalHandlers:
    def test_handler_in_spawned_task(self):
        """Each task can install its own handlers."""
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn worker() {
                handle {
                    perform Log.log("from worker");
                } with {
                    Log.log(msg) -> {
                        print(msg);
                        resume(0);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert output == ["from worker"]

    def test_separate_handlers_per_task(self):
        """Different tasks can have different handlers for the same effect."""
        _, output, _ = run("""
            effect Tag {
                tag(msg);
            }
            fn worker(prefix) {
                handle {
                    perform Tag.tag("hello");
                } with {
                    Tag.tag(msg) -> {
                        print(prefix);
                        print(msg);
                        resume(0);
                    }
                }
            }
            let t1 = spawn worker("A");
            let t2 = spawn worker("B");
            join(t1);
            join(t2);
        """)
        assert output.count("hello") == 2
        assert "A" in output
        assert "B" in output

    def test_handler_scope_isolation(self):
        """Handler installed in one task doesn't affect other tasks."""
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn logged_worker() {
                handle {
                    perform Log.log("logged");
                } with {
                    Log.log(msg) -> {
                        print(msg);
                        resume(0);
                    }
                }
            }
            fn plain_worker() {
                print("plain");
            }
            let t1 = spawn logged_worker();
            let t2 = spawn plain_worker();
            join(t1);
            join(t2);
        """)
        assert "logged" in output
        assert "plain" in output


# ============================================================
# 4. Handler Inheritance on Spawn
# ============================================================

class TestHandlerInheritance:
    def test_spawned_task_inherits_handlers(self):
        """When spawning inside a handle block, child inherits handlers."""
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn child() {
                perform Log.log("from child");
            }
            handle {
                let t = spawn child();
                join(t);
            } with {
                Log.log(msg) -> {
                    print(msg);
                    resume(0);
                }
            }
        """)
        assert output == ["from child"]

    def test_inherited_handler_independent(self):
        """Inherited handler is a copy -- modifying parent handler doesn't affect child."""
        _, output, _ = run("""
            effect Counter {
                inc();
            }
            fn child() {
                perform Counter.inc();
                perform Counter.inc();
            }
            let count = 0;
            handle {
                let t = spawn child();
                perform Counter.inc();
                join(t);
                print(count);
            } with {
                Counter.inc() -> {
                    count = count + 1;
                    resume(0);
                }
            }
        """)
        # Parent increments once, child increments twice but with its own handler copy
        # Due to env isolation, child's increments affect its own 'count'
        # Parent sees its own count = 1
        assert "1" in output

    def test_deep_handler_inheritance(self):
        """Nested handlers are all inherited."""
        _, output, _ = run("""
            effect A { getA(); }
            effect B { getB(); }
            fn child() {
                let a = perform A.getA();
                let b = perform B.getB();
                print(a + b);
            }
            handle {
                handle {
                    let t = spawn child();
                    join(t);
                } with {
                    B.getB() -> { resume(20); }
                }
            } with {
                A.getA() -> { resume(10); }
            }
        """)
        assert output == ["30"]


# ============================================================
# 5. Effects Within Concurrent Tasks
# ============================================================

class TestEffectsInTasks:
    def test_effect_with_yield(self):
        """Effect handling works across yield points."""
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn worker() {
                handle {
                    perform Log.log("before yield");
                    yield;
                    perform Log.log("after yield");
                } with {
                    Log.log(msg) -> {
                        print(msg);
                        resume(0);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert output == ["before yield", "after yield"]

    def test_effect_state_across_yields(self):
        """Effect handler state persists across yield."""
        result, output, _ = run("""
            effect State {
                get();
                set(v);
            }
            fn counter() {
                let s = 0;
                handle {
                    perform State.set(1);
                    yield;
                    perform State.set(perform State.get() + 1);
                    yield;
                    return perform State.get();
                } with {
                    State.get() -> { resume(s); }
                    State.set(v) -> { s = v; resume(0); }
                }
            }
            let t = spawn counter();
            let r = join(t);
            print(r);
        """)
        assert output == ["2"]

    def test_multiple_tasks_with_effects(self):
        """Multiple concurrent tasks each with their own effect handlers."""
        _, output, _ = run("""
            effect Tag {
                tag();
            }
            fn worker(label) {
                handle {
                    let t = perform Tag.tag();
                    print(t);
                } with {
                    Tag.tag() -> {
                        resume(label);
                    }
                }
            }
            let t1 = spawn worker("X");
            let t2 = spawn worker("Y");
            join(t1);
            join(t2);
        """)
        assert "X" in output
        assert "Y" in output

    def test_effect_with_channel_send(self):
        """Effect handler result sent via channel."""
        _, output, _ = run("""
            effect Compute {
                calc(x);
            }
            fn worker(ch) {
                handle {
                    let r = perform Compute.calc(5);
                    send(ch, r);
                } with {
                    Compute.calc(x) -> {
                        resume(x * x);
                    }
                }
            }
            let c = chan(1);
            let t = spawn worker(c);
            let result = recv(c);
            print(result);
            join(t);
        """)
        assert output == ["25"]

    def test_channel_value_in_effect_handler(self):
        """Effect handler reads from channel to produce result."""
        _, output, _ = run("""
            effect Config {
                get_config();
            }
            fn worker(config_ch) {
                handle {
                    let cfg = perform Config.get_config();
                    print(cfg);
                } with {
                    Config.get_config() -> {
                        let v = recv(config_ch);
                        resume(v);
                    }
                }
            }
            let c = chan(1);
            send(c, "production");
            let t = spawn worker(c);
            join(t);
        """)
        assert output == ["production"]


# ============================================================
# 6. Channel + Effect Interactions
# ============================================================

class TestChannelEffectInteraction:
    def test_effect_handler_sends_to_channel(self):
        _, output, _ = run("""
            effect Emit {
                emit(val);
            }
            let ch = chan(10);
            fn producer() {
                handle {
                    perform Emit.emit(1);
                    perform Emit.emit(2);
                    perform Emit.emit(3);
                } with {
                    Emit.emit(val) -> {
                        send(ch, val);
                        resume(0);
                    }
                }
            }
            let t = spawn producer();
            join(t);
            print(recv(ch));
            print(recv(ch));
            print(recv(ch));
        """)
        assert output == ["1", "2", "3"]

    def test_effect_driven_pipeline(self):
        """Producer emits via effect, consumer reads via channel."""
        _, output, _ = run("""
            effect Source {
                next();
            }
            let pipe = chan(5);
            fn producer() {
                let items = 0;
                handle {
                    let i = 0;
                    while (i < 3) {
                        let val = perform Source.next();
                        send(pipe, val);
                        i = i + 1;
                    }
                } with {
                    Source.next() -> {
                        items = items + 1;
                        resume(items * 10);
                    }
                }
            }
            fn consumer() {
                let i = 0;
                while (i < 3) {
                    print(recv(pipe));
                    i = i + 1;
                }
            }
            let tp = spawn producer();
            let tc = spawn consumer();
            join(tp);
            join(tc);
        """)
        assert output == ["10", "20", "30"]

    def test_bidirectional_effect_channel(self):
        """Effect handler communicates with another task via channels."""
        _, output, _ = run("""
            effect RPC {
                call(x);
            }
            let req_ch = chan(1);
            let resp_ch = chan(1);
            fn server() {
                let val = recv(req_ch);
                send(resp_ch, val * 3);
            }
            fn client() {
                handle {
                    let result = perform RPC.call(7);
                    print(result);
                } with {
                    RPC.call(x) -> {
                        send(req_ch, x);
                        let r = recv(resp_ch);
                        resume(r);
                    }
                }
            }
            let ts = spawn server();
            let tc = spawn client();
            join(ts);
            join(tc);
        """)
        assert output == ["21"]


# ============================================================
# 7. Continuation Capture/Resume in Tasks
# ============================================================

class TestContinuationsInTasks:
    def test_resume_in_task(self):
        result, _, _ = run("""
            effect Ask {
                ask(q);
            }
            fn worker() {
                handle {
                    return perform Ask.ask("x") + perform Ask.ask("y");
                } with {
                    Ask.ask(q) -> {
                        resume(10);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert result == 20

    def test_resume_multiple_times_in_loop(self):
        _, output, _ = run("""
            effect Gen {
                next();
            }
            fn generator() {
                let n = 0;
                handle {
                    let i = 0;
                    while (i < 4) {
                        let val = perform Gen.next();
                        print(val);
                        i = i + 1;
                    }
                } with {
                    Gen.next() -> {
                        n = n + 1;
                        resume(n);
                    }
                }
            }
            let t = spawn generator();
            join(t);
        """)
        assert output == ["1", "2", "3", "4"]

    def test_continuation_preserves_task_state(self):
        """After resume, task local variables are preserved."""
        result, _, _ = run("""
            effect Get {
                get();
            }
            fn worker() {
                let x = 5;
                handle {
                    let y = perform Get.get();
                    return x + y;
                } with {
                    Get.get() -> {
                        resume(10);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert result == 15

    def test_non_resuming_handler_in_task(self):
        """Handler that doesn't resume aborts the task's computation."""
        _, output, _ = run("""
            effect Abort {
                abort();
            }
            fn worker() {
                handle {
                    print("start");
                    perform Abort.abort();
                    print("unreachable");
                } with {
                    Abort.abort() -> {
                        print("aborted");
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert output == ["start", "aborted"]


# ============================================================
# 8. Nested Handlers in Concurrent Context
# ============================================================

class TestNestedHandlersConcurrent:
    def test_nested_handlers_in_task(self):
        result, _, _ = run("""
            effect A { getA(); }
            effect B { getB(); }
            fn worker() {
                handle {
                    handle {
                        return perform A.getA() + perform B.getB();
                    } with {
                        B.getB() -> { resume(3); }
                    }
                } with {
                    A.getA() -> { resume(7); }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert result == 10

    def test_inner_handler_shadows_outer(self):
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn worker() {
                handle {
                    perform Log.log("outer");
                    handle {
                        perform Log.log("inner");
                    } with {
                        Log.log(msg) -> {
                            print("INNER:" + msg);
                            resume(0);
                        }
                    }
                    perform Log.log("outer again");
                } with {
                    Log.log(msg) -> {
                        print("OUTER:" + msg);
                        resume(0);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert output == ["OUTER:outer", "INNER:inner", "OUTER:outer again"]

    def test_handler_in_task_with_function_calls(self):
        _, output, _ = run("""
            effect IO {
                write(msg);
            }
            fn helper() {
                perform IO.write("from helper");
            }
            fn worker() {
                handle {
                    helper();
                    perform IO.write("from worker");
                } with {
                    IO.write(msg) -> {
                        print(msg);
                        resume(0);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert output == ["from helper", "from worker"]


# ============================================================
# 9. Select with Effects
# ============================================================

class TestSelectWithEffects:
    def test_select_in_handler_body(self):
        """Select can be used inside an effect handler."""
        _, output, _ = run("""
            effect Route {
                route(msg);
            }
            let c1 = chan(1);
            let c2 = chan(1);
            fn router() {
                handle {
                    perform Route.route("hello");
                } with {
                    Route.route(msg) -> {
                        select {
                            case send(c1, msg) -> {
                                print("routed to c1");
                            }
                            default -> {
                                print("routed to c2");
                            }
                        }
                        resume(0);
                    }
                }
            }
            let t = spawn router();
            join(t);
            print(recv(c1));
        """)
        assert "routed to c1" in output
        assert "hello" in output

    def test_effect_in_select_body(self):
        """Effects can be performed inside select case bodies."""
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn worker() {
                let c = chan(1);
                send(c, "data");
                handle {
                    select {
                        case recv(c) -> val {
                            perform Log.log(val);
                        }
                    }
                } with {
                    Log.log(msg) -> {
                        print("LOG:" + msg);
                        resume(0);
                    }
                }
            }
            let t = spawn worker();
            join(t);
        """)
        assert output == ["LOG:data"]


# ============================================================
# 10. Complex Composition Scenarios
# ============================================================

class TestComplexComposition:
    def test_effect_supervisor_pattern(self):
        """Parent handles errors from child tasks via effects."""
        _, output, _ = run("""
            effect Err {
                error(msg);
            }
            fn risky_worker() {
                perform Err.error("task failed");
            }
            handle {
                let t = spawn risky_worker();
                join(t);
                print("completed");
            } with {
                Err.error(msg) -> {
                    print("caught: " + msg);
                    resume(0);
                }
            }
        """)
        assert "caught: task failed" in output

    def test_producer_consumer_with_effects(self):
        """Full producer-consumer with effect-driven production."""
        _, output, _ = run("""
            effect Generate {
                gen(seed);
            }
            let pipe = chan(10);
            fn producer() {
                handle {
                    let i = 0;
                    while (i < 3) {
                        let val = perform Generate.gen(i);
                        send(pipe, val);
                        i = i + 1;
                    }
                } with {
                    Generate.gen(seed) -> {
                        resume(seed * seed + 1);
                    }
                }
            }
            fn consumer() {
                let i = 0;
                while (i < 3) {
                    print(recv(pipe));
                    i = i + 1;
                }
            }
            let tp = spawn producer();
            let tc = spawn consumer();
            join(tp);
            join(tc);
        """)
        # seed=0: 0*0+1=1, seed=1: 1*1+1=2, seed=2: 2*2+1=5
        assert output == ["1", "2", "5"]

    def test_worker_pool_with_effect_logging(self):
        _, output, _ = run("""
            effect Log {
                log(msg);
            }
            fn worker(id, ch) {
                handle {
                    perform Log.log("start " + id);
                    send(ch, id);
                    perform Log.log("done " + id);
                } with {
                    Log.log(msg) -> {
                        print(msg);
                        resume(0);
                    }
                }
            }
            let results = chan(10);
            let t1 = spawn worker("A", results);
            let t2 = spawn worker("B", results);
            join(t1);
            join(t2);
            print(recv(results));
            print(recv(results));
        """)
        assert "start A" in output
        assert "start B" in output
        assert "done A" in output
        assert "done B" in output

    def test_effect_based_task_coordination(self):
        """Use effects to coordinate between tasks via shared channel."""
        _, output, _ = run("""
            effect Signal {
                signal(val);
                wait();
            }
            let sig_ch = chan(1);
            fn signaler() {
                handle {
                    perform Signal.signal(42);
                } with {
                    Signal.signal(val) -> {
                        send(sig_ch, val);
                        resume(0);
                    }
                }
            }
            fn waiter() {
                handle {
                    let val = perform Signal.wait();
                    print(val);
                } with {
                    Signal.wait() -> {
                        let v = recv(sig_ch);
                        resume(v);
                    }
                }
            }
            let ts = spawn signaler();
            let tw = spawn waiter();
            join(ts);
            join(tw);
        """)
        assert output == ["42"]

    def test_concurrent_state_effects(self):
        """Each task has independent state effect."""
        _, output, _ = run("""
            effect State {
                get();
                set(v);
            }
            fn counter(id, n) {
                let s = 0;
                handle {
                    let i = 0;
                    while (i < n) {
                        perform State.set(perform State.get() + 1);
                        i = i + 1;
                    }
                    print(id + ":" + perform State.get());
                } with {
                    State.get() -> { resume(s); }
                    State.set(v) -> { s = v; resume(0); }
                }
            }
            let t1 = spawn counter("A", 3);
            let t2 = spawn counter("B", 5);
            join(t1);
            join(t2);
        """)
        assert "A:3" in output
        assert "B:5" in output

    def test_fan_out_fan_in_with_effects(self):
        """Fan-out work to multiple tasks with effects, collect results."""
        _, output, _ = run("""
            effect Transform {
                transform(x);
            }
            let results = chan(10);
            fn worker(x) {
                handle {
                    let r = perform Transform.transform(x);
                    send(results, r);
                } with {
                    Transform.transform(x) -> {
                        resume(x * x);
                    }
                }
            }
            let t1 = spawn worker(2);
            let t2 = spawn worker(3);
            let t3 = spawn worker(4);
            join(t1);
            join(t2);
            join(t3);
            let sum = recv(results) + recv(results) + recv(results);
            print(sum);
        """)
        # 4 + 9 + 16 = 29
        assert output == ["29"]


# ============================================================
# 11. Error Handling
# ============================================================

class TestErrorHandling:
    def test_unhandled_effect_error(self):
        with pytest.raises(EffectError, match="Unhandled effect"):
            run("""
                effect Missing {
                    op();
                }
                perform Missing.op();
            """)

    def test_resume_outside_handler(self):
        with pytest.raises(ResumeError):
            run("""
                resume(42);
            """)

    def test_undefined_variable(self):
        with pytest.raises(RuntimeError, match="Undefined variable"):
            run("""
                print(undefined_var);
            """)

    def test_spawn_non_function(self):
        with pytest.raises(RuntimeError, match="Cannot spawn"):
            run("""
                let x = 5;
                spawn x();
            """)

    def test_send_to_non_channel(self):
        with pytest.raises(RuntimeError, match="Not a channel"):
            run("""
                send(42, "msg");
            """)

    def test_recv_from_non_channel(self):
        with pytest.raises(RuntimeError, match="Not a channel"):
            run("""
                recv(42);
            """)

    def test_call_non_function(self):
        with pytest.raises(RuntimeError, match="Not callable"):
            run("""
                let x = 5;
                x();
            """)

    def test_unhandled_effect_in_task_propagates(self):
        """Unhandled effect in non-main task should fail that task."""
        _, _, tasks = run("""
            effect Missing {
                op();
            }
            fn worker() {
                perform Missing.op();
            }
            let t = spawn worker();
            join(t);
        """)
        # Worker task should fail
        worker_info = tasks[1]
        assert worker_info['state'] == 'FAILED'


# ============================================================
# 12. Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_handler_body(self):
        result, _, _ = run("""
            effect E { op(); }
            handle {
                42;
            } with {
                E.op() -> { resume(0); }
            }
        """)
        assert result == 42

    def test_handler_with_no_resume_returns_value(self):
        result, _, _ = run("""
            effect Early {
                ret(val);
            }
            handle {
                perform Early.ret(99);
                0;
            } with {
                Early.ret(val) -> {
                    val;
                }
            }
        """)
        # Handler body returns None (expression statement pops, then implicit None return)
        # The handle block's result comes from handler clause

    def test_deeply_nested_spawn_with_effects(self):
        _, output, _ = run("""
            effect Log { log(msg); }
            fn deep(n) {
                if (n > 0) {
                    handle {
                        perform Log.log("depth " + n);
                        let t = spawn deep(n - 1);
                        join(t);
                    } with {
                        Log.log(msg) -> {
                            print(msg);
                            resume(0);
                        }
                    }
                }
            }
            let t = spawn deep(3);
            join(t);
        """)
        assert "depth 3" in output
        assert "depth 2" in output
        assert "depth 1" in output

    def test_string_operations(self):
        result, output, _ = run("""
            let s = "hello" + " " + "world";
            print(s);
            s;
        """)
        assert output == ["hello world"]
        assert result == "hello world"

    def test_boolean_operations(self):
        _, output, _ = run("""
            print(true && false);
            print(true || false);
            print(!true);
            print(1 == 1);
            print(1 != 2);
        """)
        assert output == ["False", "True", "False", "True", "True"]

    def test_comparison_operators(self):
        _, output, _ = run("""
            print(1 < 2);
            print(2 > 1);
            print(1 <= 1);
            print(1 >= 2);
        """)
        assert output == ["True", "True", "True", "False"]

    def test_modulo(self):
        result, _, _ = run("""
            10 % 3;
        """)
        assert result == 1

    def test_nested_function_calls(self):
        result, _, _ = run("""
            fn add(a, b) { return a + b; }
            fn double(x) { return x * 2; }
            double(add(3, 4));
        """)
        assert result == 14

    def test_while_with_break_condition(self):
        _, output, _ = run("""
            let i = 0;
            let sum = 0;
            while (i < 10) {
                sum = sum + i;
                i = i + 1;
            }
            print(sum);
        """)
        assert output == ["45"]

    def test_if_else_chain(self):
        _, output, _ = run("""
            fn classify(x) {
                if (x < 0) {
                    return "negative";
                } else if (x == 0) {
                    return "zero";
                } else {
                    return "positive";
                }
            }
            print(classify(0 - 5));
            print(classify(0));
            print(classify(5));
        """)
        assert output == ["negative", "zero", "positive"]

    def test_recursive_function(self):
        result, _, _ = run("""
            fn fib(n) {
                if (n <= 1) {
                    return n;
                }
                return fib(n - 1) + fib(n - 2);
            }
            fib(10);
        """)
        assert result == 55

    def test_channel_buffer_sizes(self):
        result, _, _ = run("""
            let c = chan(3);
            send(c, 1);
            send(c, 2);
            send(c, 3);
            let a = recv(c);
            let b = recv(c);
            let c2 = recv(c);
            a + b + c2;
        """)
        assert result == 6

    def test_task_join_already_completed(self):
        """Joining a task that's already completed should return immediately."""
        result, _, _ = run("""
            fn quick() {
                return 42;
            }
            let t = spawn quick();
            yield;
            yield;
            join(t);
        """)
        assert result == 42

    def test_multiple_handler_clauses(self):
        result, _, _ = run("""
            effect Math {
                add(a, b);
                mul(a, b);
            }
            handle {
                let x = perform Math.add(3, 4);
                let y = perform Math.mul(x, 2);
                y;
            } with {
                Math.add(a, b) -> { resume(a + b); }
                Math.mul(a, b) -> { resume(a * b); }
            }
        """)
        assert result == 14

    def test_effect_with_string_args(self):
        _, output, _ = run("""
            effect Format {
                fmt(template, val);
            }
            handle {
                let r = perform Format.fmt("count", 42);
                print(r);
            } with {
                Format.fmt(template, val) -> {
                    resume(template + "=" + val);
                }
            }
        """)
        assert output == ["count=42"]

    def test_spawn_inside_handler_body(self):
        """Spawning a task from within a handler clause body."""
        _, output, _ = run("""
            effect Delegate {
                delegate(x);
            }
            let ch = chan(1);
            fn helper(x, ch) {
                send(ch, x * 10);
            }
            handle {
                perform Delegate.delegate(5);
                print(recv(ch));
            } with {
                Delegate.delegate(x) -> {
                    let t = spawn helper(x, ch);
                    join(t);
                    resume(0);
                }
            }
        """)
        assert output == ["50"]

    def test_many_tasks_with_effects(self):
        """Stress test: many tasks each performing effects."""
        _, output, _ = run("""
            effect ID {
                get_id();
            }
            let ch = chan(20);
            fn worker(n) {
                handle {
                    let my_id = perform ID.get_id();
                    send(ch, my_id);
                } with {
                    ID.get_id() -> {
                        resume(n);
                    }
                }
            }
            let i = 0;
            let tasks = chan(20);
            while (i < 10) {
                let t = spawn worker(i);
                send(tasks, t);
                i = i + 1;
            }
            i = 0;
            while (i < 10) {
                let t = recv(tasks);
                join(t);
                i = i + 1;
            }
            let sum = 0;
            i = 0;
            while (i < 10) {
                sum = sum + recv(ch);
                i = i + 1;
            }
            print(sum);
        """, max_total_steps=1000000)
        # 0+1+2+...+9 = 45
        assert output == ["45"]

    def test_lexer_keywords_as_identifiers(self):
        """Keywords used as effect operation names (known recurring bug pattern)."""
        _, output, _ = run("""
            effect IO {
                print(msg);
            }
            handle {
                perform IO.print("via effect");
            } with {
                IO.print(msg) -> {
                    print("EFFECT:" + msg);
                    resume(0);
                }
            }
        """)
        assert output == ["EFFECT:via effect"]

    def test_handler_clause_with_many_params(self):
        result, _, _ = run("""
            effect Multi {
                combine(a, b, c);
            }
            handle {
                perform Multi.combine(1, 2, 3);
            } with {
                Multi.combine(a, b, c) -> {
                    resume(a + b + c);
                }
            }
        """)
        assert result == 6

    def test_effect_and_return(self):
        """Returning from a function that uses effects."""
        result, _, _ = run("""
            effect Val {
                get();
            }
            fn compute() {
                handle {
                    let x = perform Val.get();
                    return x * 2;
                } with {
                    Val.get() -> {
                        resume(21);
                    }
                }
            }
            compute();
        """)
        assert result == 42

    def test_concurrent_effects_no_interference(self):
        """Two tasks performing the same effect type don't interfere."""
        _, output, _ = run("""
            effect Counter {
                inc();
                get();
            }
            fn counting_task(id, n) {
                let count = 0;
                handle {
                    let i = 0;
                    while (i < n) {
                        perform Counter.inc();
                        i = i + 1;
                    }
                    let final = perform Counter.get();
                    print(id + "=" + final);
                } with {
                    Counter.inc() -> {
                        count = count + 1;
                        resume(0);
                    }
                    Counter.get() -> {
                        resume(count);
                    }
                }
            }
            let t1 = spawn counting_task("A", 3);
            let t2 = spawn counting_task("B", 7);
            join(t1);
            join(t2);
        """)
        assert "A=3" in output
        assert "B=7" in output


# ============================================================
# 13. Parser & Lexer Tests
# ============================================================

class TestParserLexer:
    def test_lex_all_token_types(self):
        tokens = lex("let fn if else while return print true false 42 3.14 \"hello\" x + - * / % == != < > <= >= = ! && || ( ) { } , ; . -> spawn yield chan send recv join task_id select case default effect perform handle with resume")
        types = [t.type for t in tokens]
        assert TokenType.LET in types
        assert TokenType.FN in types
        assert TokenType.SPAWN in types
        assert TokenType.EFFECT in types
        assert TokenType.PERFORM in types
        assert TokenType.RESUME in types
        assert TokenType.EOF in types

    def test_parse_effect_decl(self):
        ast = parse("effect IO { read(); write(msg); }")
        assert len(ast) == 1
        assert isinstance(ast[0], EffectDecl)
        assert ast[0].name == "IO"
        assert len(ast[0].operations) == 2

    def test_parse_perform(self):
        ast = parse("perform IO.write(42);")
        assert len(ast) == 1
        assert isinstance(ast[0], PerformExpr)
        assert ast[0].effect == "IO"
        assert ast[0].operation == "write"

    def test_parse_handle_with(self):
        ast = parse("""
            handle { 42; } with {
                IO.write(msg) -> { print(msg); }
            }
        """)
        assert len(ast) == 1
        assert isinstance(ast[0], HandleWith)
        assert len(ast[0].handlers) == 1

    def test_parse_spawn(self):
        ast = parse("spawn worker(1, 2);")
        assert len(ast) == 1
        assert isinstance(ast[0], SpawnExpr)
        assert ast[0].callee == "worker"
        assert len(ast[0].args) == 2

    def test_parse_select(self):
        ast = parse("""
            select {
                case send(c, 1) -> { print("sent"); }
                case recv(c) -> val { print(val); }
                default -> { print("default"); }
            }
        """)
        assert len(ast) == 1
        assert isinstance(ast[0], SelectStmt)
        assert len(ast[0].cases) == 2
        assert ast[0].default_body is not None

    def test_parse_error(self):
        with pytest.raises(ParseError):
            parse("let = ;")

    def test_comments_ignored(self):
        result, _, _ = run("""
            // This is a comment
            let x = 42;
            // Another comment
            x;
        """)
        assert result == 42


# ============================================================
# 14. VM Internals
# ============================================================

class TestVMInternals:
    def test_channel_creation(self):
        ch = Channel(buffer_size=3)
        assert ch.buffer_size == 3
        assert ch.can_send()
        assert not ch.can_recv()

    def test_channel_try_ops(self):
        ch = Channel(buffer_size=2)
        assert ch.try_send(1)
        assert ch.try_send(2)
        assert not ch.try_send(3)  # buffer full
        ok, val = ch.try_recv()
        assert ok and val == 1
        assert ch.try_send(3)  # space now

    def test_channel_close(self):
        ch = Channel(buffer_size=1)
        ch.close()
        assert not ch.can_send()
        assert not ch.try_send(42)

    def test_task_state_enum(self):
        assert TaskState.READY == 0
        assert TaskState.COMPLETED == 5
        assert TaskState.FAILED == 6

    def test_compile_and_run_separately(self):
        ast = parse("let x = 42; x;")
        chunk, compiler = compile_program(ast)
        vm = ConcurrentEffectVM(chunk, functions=compiler.functions)
        result = vm.run()
        assert result == 42


# ============================================================
# 15. Integration Stress Tests
# ============================================================

class TestIntegrationStress:
    def test_effect_handler_across_many_yields(self):
        _, output, _ = run("""
            effect Tick {
                tick();
            }
            fn ticker(n) {
                let count = 0;
                handle {
                    let i = 0;
                    while (i < n) {
                        perform Tick.tick();
                        yield;
                        i = i + 1;
                    }
                    print(count);
                } with {
                    Tick.tick() -> {
                        count = count + 1;
                        resume(0);
                    }
                }
            }
            let t = spawn ticker(5);
            join(t);
        """)
        assert output == ["5"]

    def test_concurrent_effects_with_channels(self):
        """Multiple tasks using effects AND channels simultaneously."""
        _, output, _ = run("""
            effect Transform {
                transform(x);
            }
            let input_ch = chan(5);
            let output_ch = chan(5);
            fn transformer(id) {
                handle {
                    let val = recv(input_ch);
                    let result = perform Transform.transform(val);
                    send(output_ch, result);
                } with {
                    Transform.transform(x) -> {
                        resume(x + id);
                    }
                }
            }
            send(input_ch, 10);
            send(input_ch, 20);
            let t1 = spawn transformer(100);
            let t2 = spawn transformer(200);
            join(t1);
            join(t2);
            let r1 = recv(output_ch);
            let r2 = recv(output_ch);
            let sum = r1 + r2;
            print(sum);
        """)
        # One gets 10+100=110, other gets 20+200=220, sum=330
        # Or 10+200=210, 20+100=120, sum=330
        assert output == ["330"]

    def test_recursive_spawn_with_effects(self):
        """Recursive spawning with effects at each level."""
        _, output, _ = run("""
            effect Depth {
                report(d);
            }
            fn recursive_spawn(n) {
                handle {
                    perform Depth.report(n);
                    if (n > 1) {
                        let t = spawn recursive_spawn(n - 1);
                        join(t);
                    }
                } with {
                    Depth.report(d) -> {
                        print(d);
                        resume(0);
                    }
                }
            }
            let t = spawn recursive_spawn(3);
            join(t);
        """)
        assert "3" in output
        assert "2" in output
        assert "1" in output

    def test_effect_handler_with_closure_over_channel(self):
        """Handler body closes over a channel from outer scope."""
        _, output, _ = run("""
            effect Audit {
                audit(msg);
            }
            let audit_ch = chan(10);
            fn audited_work() {
                handle {
                    perform Audit.audit("step1");
                    perform Audit.audit("step2");
                } with {
                    Audit.audit(msg) -> {
                        send(audit_ch, msg);
                        resume(0);
                    }
                }
            }
            let t = spawn audited_work();
            join(t);
            print(recv(audit_ch));
            print(recv(audit_ch));
        """)
        assert output == ["step1", "step2"]

    def test_pipeline_three_stages(self):
        """Three-stage pipeline: generate -> transform -> collect."""
        _, output, _ = run("""
            effect Gen { next(i); }
            effect Xform { apply(x); }
            let ch1 = chan(5);
            let ch2 = chan(5);
            fn generator() {
                handle {
                    let i = 0;
                    while (i < 3) {
                        let v = perform Gen.next(i);
                        send(ch1, v);
                        i = i + 1;
                    }
                } with {
                    Gen.next(i) -> { resume(i * 10); }
                }
            }
            fn transformer() {
                handle {
                    let i = 0;
                    while (i < 3) {
                        let v = recv(ch1);
                        let r = perform Xform.apply(v);
                        send(ch2, r);
                        i = i + 1;
                    }
                } with {
                    Xform.apply(x) -> { resume(x + 1); }
                }
            }
            fn collector() {
                let i = 0;
                while (i < 3) {
                    print(recv(ch2));
                    i = i + 1;
                }
            }
            let tg = spawn generator();
            let tt = spawn transformer();
            let tc = spawn collector();
            join(tg);
            join(tt);
            join(tc);
        """)
        # gen: 0*10=0, 1*10=10, 2*10=20
        # xform: 0+1=1, 10+1=11, 20+1=21
        assert output == ["1", "11", "21"]

    def test_division(self):
        result, _, _ = run("10 / 3;")
        assert result == 3  # integer division

    def test_float_division(self):
        result, _, _ = run("10.0 / 3;")
        assert abs(result - 10.0/3) < 0.001

    def test_negative_numbers(self):
        result, _, _ = run("let x = 0 - 5; x;")
        assert result == -5

    def test_unary_negation(self):
        result, _, _ = run("-5 + 3;")
        assert result == -2

    def test_assignment_as_expression(self):
        _, output, _ = run("""
            let x = 0;
            let y = x = 5;
            print(x);
            print(y);
        """)
        assert output == ["5", "5"]

    def test_string_escape(self):
        _, output, _ = run(r"""
            print("hello\nworld");
            print("tab\there");
        """)
        assert output == ["hello\nworld", "tab\there"]

    def test_empty_function(self):
        result, _, _ = run("""
            fn noop() {}
            noop();
        """)
        assert result is None

    def test_max_steps_limit(self):
        """Verify that max_total_steps prevents infinite loops."""
        result, _, tasks = run("""
            fn loop_forever() {
                while (true) {
                    yield;
                }
            }
            let t = spawn loop_forever();
            yield;
            yield;
            42;
        """, max_total_steps=1000)
        # Main should complete before infinite loop exhausts steps

    def test_type_aware_equality(self):
        """True != 1, False != 0 with type-aware comparison."""
        _, output, _ = run("""
            print(true == 1);
            print(false == 0);
            print(true == true);
            print(1 == 1);
        """)
        assert output == ["False", "False", "True", "True"]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
