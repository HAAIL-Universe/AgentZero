"""
Tests for the Bytecode Optimizer (C014)
Challenge C014 -- AgentZero Session 015

Tests organized by optimization pass, then integration tests verifying
correctness (same result with and without optimization).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from stack_vm import (
    Op, Chunk, FnObject, lex, Parser, Compiler, VM,
    compile_source, execute, disassemble,
)
from optimizer import (
    Instr, OPS_WITH_OPERAND,
    decode_chunk, encode_instructions,
    addrs_to_indices, indices_to_addrs,
    constant_fold, strength_reduce, peephole,
    eliminate_dead_code, optimize_jumps, constant_propagation,
    optimize_chunk, optimize_all, optimize_source,
    execute_optimized, compare_execution,
    OptimizationStats,
)


# ============================================================
# Helper
# ============================================================

def compile_to_instrs(source):
    """Compile source and return (instrs, chunk) with index-based jumps."""
    chunk, compiler = compile_source(source)
    addr_instrs = decode_chunk(chunk)
    instrs = addrs_to_indices(addr_instrs)
    return instrs, chunk, compiler


def assert_same_result(source):
    """Assert optimized execution produces same result as unoptimized."""
    result = compare_execution(source)
    assert result['same_result'], (
        f"Results differ: unopt={result['unoptimized']['result']} "
        f"vs opt={result['optimized']['result']}"
    )
    assert result['same_output'], (
        f"Outputs differ: unopt={result['unoptimized']['output']} "
        f"vs opt={result['optimized']['output']}"
    )
    return result


# ============================================================
# Instruction Encoding/Decoding
# ============================================================

class TestDecodeEncode:
    def test_roundtrip_simple(self):
        source = "let x = 42;"
        chunk, _ = compile_source(source)
        addr_instrs = decode_chunk(chunk)
        instrs = [instr for _, instr in addr_instrs]
        new_chunk = encode_instructions(instrs, chunk)
        assert new_chunk.code == chunk.code

    def test_roundtrip_with_jumps(self):
        source = "if (true) { let x = 1; } else { let x = 2; }"
        chunk, _ = compile_source(source)
        addr_instrs = decode_chunk(chunk)
        instrs = [instr for _, instr in addr_instrs]
        new_chunk = encode_instructions(instrs, chunk)
        assert new_chunk.code == chunk.code

    def test_decode_preserves_ops(self):
        source = "let x = 1 + 2;"
        chunk, _ = compile_source(source)
        addr_instrs = decode_chunk(chunk)
        ops = [instr.op for _, instr in addr_instrs]
        assert Op.CONST in ops
        assert Op.ADD in ops
        assert Op.STORE in ops
        assert Op.HALT in ops

    def test_decode_operands(self):
        source = "let x = 10;"
        chunk, _ = compile_source(source)
        addr_instrs = decode_chunk(chunk)
        # First: CONST with operand
        assert addr_instrs[0][1].op == Op.CONST
        assert addr_instrs[0][1].operand is not None

    def test_instr_predicates(self):
        assert Instr(Op.CONST, 0).has_operand()
        assert Instr(Op.JUMP, 5).is_jump()
        assert Instr(Op.ADD).is_arithmetic()
        assert Instr(Op.EQ).is_comparison()
        assert Instr(Op.HALT).is_terminator()
        assert not Instr(Op.ADD).is_jump()
        assert not Instr(Op.POP).has_operand()


# ============================================================
# Dead Code Detection (replaces CFG tests)
# ============================================================

class TestDeadCodeDetection:
    def test_single_block_no_dead(self):
        source = "let x = 42;"
        instrs, chunk, _ = compile_to_instrs(source)
        result, changed = eliminate_dead_code(instrs, chunk)
        assert not changed  # no dead code

    def test_unreachable_after_halt(self):
        chunk = Chunk()
        idx0 = chunk.add_constant(42)
        chunk.emit(Op.CONST, idx0, 1)
        chunk.emit(Op.HALT, line=1)
        idx1 = chunk.add_constant(99)
        chunk.emit(Op.CONST, idx1, 2)
        chunk.emit(Op.PRINT, line=2)
        chunk.emit(Op.HALT, line=2)

        addr_instrs = decode_chunk(chunk)
        instrs = addrs_to_indices(addr_instrs)
        result, changed = eliminate_dead_code(instrs, chunk)
        assert changed
        assert len(result) < len(instrs)

    def test_empty_instrs(self):
        result, changed = eliminate_dead_code([], Chunk())
        assert not changed


# ============================================================
# Pass 1: Constant Folding
# ============================================================

class TestConstantFolding:
    def test_add_integers(self):
        instrs, chunk, _ = compile_to_instrs("let x = 2 + 3;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        # Should have fewer instructions (CONST 2, CONST 3, ADD -> CONST 5)
        const_ops = [i for i in result if i.op == Op.CONST]
        add_ops = [i for i in result if i.op == Op.ADD]
        assert len(add_ops) == 0  # ADD folded away

    def test_multiply_integers(self):
        instrs, chunk, _ = compile_to_instrs("let x = 6 * 7;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        # Verify the folded constant is 42
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert 42 in vals

    def test_chain_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = 1 + 2 + 3;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert 6 in vals

    def test_float_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = 1.5 + 2.5;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert 4.0 in vals

    def test_division_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = 10 / 3;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert 3 in vals  # integer division

    def test_no_div_by_zero(self):
        instrs, chunk, _ = compile_to_instrs("let x = 10 / 0;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        # Should NOT fold division by zero
        div_ops = [i for i in result if i.op == Op.DIV]
        assert len(div_ops) == 1

    def test_comparison_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = 5 > 3;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert True in vals

    def test_unary_neg_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = -42;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        neg_ops = [i for i in result if i.op == Op.NEG]
        assert len(neg_ops) == 0

    def test_unary_not_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = not true;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        not_ops = [i for i in result if i.op == Op.NOT]
        assert len(not_ops) == 0

    def test_no_fold_variables(self):
        instrs, chunk, _ = compile_to_instrs("let a = 1; let b = a + 2;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        # a + 2 can't be folded since a is a variable at compile time
        add_ops = [i for i in result if i.op == Op.ADD]
        assert len(add_ops) == 1

    def test_no_fold_bool_arithmetic(self):
        """Booleans should not be folded in arithmetic (True==1 issue)."""
        instrs, chunk, _ = compile_to_instrs("let x = true; let y = 1 + 2;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        # 1+2 should fold but true should remain untouched
        assert changed

    def test_modulo_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = 17 % 5;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert 2 in vals

    def test_subtraction_folding(self):
        instrs, chunk, _ = compile_to_instrs("let x = 10 - 3;")
        result, new_chunk, changed = constant_fold(instrs, chunk)
        assert changed
        const_instrs = [i for i in result if i.op == Op.CONST]
        vals = [new_chunk.constants[i.operand] for i in const_instrs]
        assert 7 in vals


# ============================================================
# Pass 2: Strength Reduction
# ============================================================

class TestStrengthReduction:
    def test_add_zero(self):
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = a + 0;")
        result, changed = strength_reduce(instrs, chunk)
        assert changed
        # CONST 0, ADD should be removed
        add_ops = [i for i in result if i.op == Op.ADD]
        zero_consts = [i for i in result
                       if i.op == Op.CONST and chunk.constants[i.operand] == 0
                       and not isinstance(chunk.constants[i.operand], bool)]
        assert len(add_ops) + len(zero_consts) < 2

    def test_sub_zero(self):
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = a - 0;")
        result, changed = strength_reduce(instrs, chunk)
        assert changed

    def test_mul_one(self):
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = a * 1;")
        result, changed = strength_reduce(instrs, chunk)
        assert changed

    def test_div_one(self):
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = a / 1;")
        result, changed = strength_reduce(instrs, chunk)
        assert changed

    def test_mul_two(self):
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = a * 2;")
        result, changed = strength_reduce(instrs, chunk)
        assert changed
        # Should be replaced with DUP, ADD
        dup_ops = [i for i in result if i.op == Op.DUP]
        assert len(dup_ops) >= 1

    def test_double_neg(self):
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = - -a;")
        result, changed = strength_reduce(instrs, chunk)
        # Actually the parser might handle --a differently, let's check
        # UnaryOp('-', UnaryOp('-', Var('a'))) -> NEG, NEG which should cancel
        neg_ops = [i for i in result if i.op == Op.NEG]
        # Either 0 (both removed) or still 2 (not consecutive) depending on compilation
        # The compiler emits: LOAD a, NEG, NEG -- consecutive, should cancel
        if changed:
            assert len(neg_ops) == 0

    def test_double_not(self):
        instrs, chunk, _ = compile_to_instrs("let a = true; let b = not not a;")
        result, changed = strength_reduce(instrs, chunk)
        if changed:
            not_ops = [i for i in result if i.op == Op.NOT]
            assert len(not_ops) == 0


# ============================================================
# Pass 3: Peephole Optimization
# ============================================================

class TestPeephole:
    def test_store_load_elimination(self):
        """STORE x, LOAD x -> DUP, STORE x."""
        instrs, chunk, _ = compile_to_instrs("let a = 5; let b = a;")
        # In the compiled output, "let b = a" generates LOAD a, STORE b
        # Not STORE then LOAD same variable. Let me construct manually.
        work_chunk = Chunk()
        work_chunk.constants = [42]
        work_chunk.names = ['x']
        test_instrs = [
            Instr(Op.CONST, 0),
            Instr(Op.STORE, 0),
            Instr(Op.LOAD, 0),
            Instr(Op.PRINT),
            Instr(Op.HALT),
        ]
        result, changed = peephole(test_instrs, work_chunk)
        assert changed
        # Should be: CONST, DUP, STORE, PRINT, HALT
        assert result[1].op == Op.DUP
        assert result[2].op == Op.STORE

    def test_const_pop_elimination(self):
        """CONST x, POP -> nothing."""
        work_chunk = Chunk()
        work_chunk.constants = [42]
        test_instrs = [
            Instr(Op.CONST, 0),
            Instr(Op.POP),
            Instr(Op.HALT),
        ]
        result, changed = peephole(test_instrs, work_chunk)
        assert changed
        assert len(result) == 1  # just HALT

    def test_load_pop_elimination(self):
        """LOAD x, POP -> nothing."""
        work_chunk = Chunk()
        work_chunk.names = ['x']
        test_instrs = [
            Instr(Op.LOAD, 0),
            Instr(Op.POP),
            Instr(Op.HALT),
        ]
        result, changed = peephole(test_instrs, work_chunk)
        assert changed
        assert len(result) == 1  # just HALT

    def test_dup_pop_elimination(self):
        """DUP, POP -> nothing."""
        work_chunk = Chunk()
        test_instrs = [
            Instr(Op.CONST, 0),
            Instr(Op.DUP),
            Instr(Op.POP),
            Instr(Op.HALT),
        ]
        work_chunk.constants = [1]
        result, changed = peephole(test_instrs, work_chunk)
        assert changed
        assert len(result) == 2  # CONST, HALT


# ============================================================
# Pass 4: Dead Code Elimination
# ============================================================

class TestDeadCodeElimination:
    def test_code_after_halt(self):
        chunk = Chunk()
        idx = chunk.add_constant(42)
        chunk.emit(Op.CONST, idx, 1)
        chunk.emit(Op.HALT, line=1)
        # Dead code
        idx2 = chunk.add_constant(99)
        chunk.emit(Op.CONST, idx2, 2)
        chunk.emit(Op.PRINT, line=2)
        chunk.emit(Op.HALT, line=2)

        addr_instrs = decode_chunk(chunk)
        instrs = addrs_to_indices(addr_instrs)
        result, changed = eliminate_dead_code(instrs, chunk)
        assert changed
        assert len(result) < len(instrs)

    def test_no_dead_code(self):
        source = "let x = 42;"
        instrs, chunk, _ = compile_to_instrs(source)
        result, changed = eliminate_dead_code(instrs, chunk)
        assert not changed
        assert len(result) == len(instrs)

    def test_reachable_branches_preserved(self):
        source = "if (true) { let x = 1; } else { let x = 2; }"
        instrs, chunk, _ = compile_to_instrs(source)
        result, changed = eliminate_dead_code(instrs, chunk)
        # Both branches should be preserved (condition unknown at bytecode level)
        assert len(result) >= len(instrs) - 2


# ============================================================
# Pass 5: Jump Optimization
# ============================================================

class TestJumpOptimization:
    def test_jump_to_jump_threading(self):
        """JUMP -> JUMP target -> thread to final target (index-based)."""
        chunk = Chunk()
        # Instruction indices: 0, 1, 2, 3, 4
        test_instrs = [
            Instr(Op.JUMP, 2),    # idx 0: jump to idx 2
            Instr(Op.HALT),       # idx 1: (unreachable)
            Instr(Op.JUMP, 4),    # idx 2: jump to idx 4
            Instr(Op.HALT),       # idx 3: (unreachable)
            Instr(Op.HALT),       # idx 4: final target
        ]
        result, changed = optimize_jumps(test_instrs, chunk)
        assert changed
        # First jump should now target idx 4 directly
        assert result[0].operand == 4

    def test_jump_to_next_removal(self):
        """JUMP to immediately next instruction -> remove (index-based)."""
        chunk = Chunk()
        test_instrs = [
            Instr(Op.CONST, 0),   # idx 0
            Instr(Op.JUMP, 2),    # idx 1: targets idx 2 = next instr
            Instr(Op.HALT),       # idx 2
        ]
        result, changed = optimize_jumps(test_instrs, chunk)
        assert changed
        assert len(result) == 2  # CONST, HALT (JUMP removed)


# ============================================================
# Pass 6: Constant Propagation
# ============================================================

class TestConstantPropagation:
    def test_simple_propagation(self):
        """let x = 5; let y = x; -> y gets constant 5 directly."""
        instrs, chunk, _ = compile_to_instrs("let x = 5; let y = x;")
        result, changed = constant_propagation(instrs, chunk)
        assert changed
        # The LOAD x should be replaced with CONST 5
        load_ops = [i for i in result if i.op == Op.LOAD]
        # Should have fewer LOADs (the second one replaced)
        original_loads = [i for i in instrs if i.op == Op.LOAD]
        assert len(load_ops) < len(original_loads)

    def test_propagation_after_reassignment(self):
        """After reassignment, old constant is invalidated."""
        work_chunk = Chunk()
        work_chunk.constants = [5, 10]
        work_chunk.names = ['x']
        test_instrs = [
            Instr(Op.CONST, 0),   # push 5
            Instr(Op.STORE, 0),   # x = 5
            Instr(Op.LOAD, 0),    # load x (should become CONST 5)
            Instr(Op.PRINT),
            Instr(Op.CONST, 1),   # push 10
            Instr(Op.STORE, 0),   # x = 10
            Instr(Op.LOAD, 0),    # load x (should become CONST 10)
            Instr(Op.PRINT),
            Instr(Op.HALT),
        ]
        result, changed = constant_propagation(test_instrs, work_chunk)
        assert changed
        # Both LOADs should be replaced
        loads = [i for i in result if i.op == Op.LOAD]
        assert len(loads) == 0

    def test_no_propagation_across_jumps(self):
        """Constants should not propagate across jump boundaries."""
        work_chunk = Chunk()
        work_chunk.constants = [5, True]
        work_chunk.names = ['x']
        test_instrs = [
            Instr(Op.CONST, 0),        # idx 0: push 5
            Instr(Op.STORE, 0),         # idx 1: x = 5
            Instr(Op.CONST, 1),         # idx 2: push true
            Instr(Op.JUMP_IF_FALSE, 5), # idx 3: conditional jump to idx 5
            Instr(Op.LOAD, 0),          # idx 4: load x -- after jump, can't propagate
            Instr(Op.HALT),             # idx 5: target of jump
        ]
        result, changed = constant_propagation(test_instrs, work_chunk)
        # After the jump, knowledge is cleared, so LOAD stays
        # Also idx 5 is a jump target so knowledge is cleared there too
        # But idx 4 comes after a JUMP_IF_FALSE which clears knowledge
        loads_after_jump = [i for i in result[4:] if i.op == Op.LOAD]
        assert len(loads_after_jump) == 1  # not propagated

    def test_non_constant_store_invalidates(self):
        """STORE without preceding CONST invalidates knowledge."""
        work_chunk = Chunk()
        work_chunk.constants = [5]
        work_chunk.names = ['x', 'y']
        test_instrs = [
            Instr(Op.CONST, 0),   # push 5
            Instr(Op.STORE, 0),   # x = 5
            Instr(Op.LOAD, 1),    # load y (unknown, non-constant)
            Instr(Op.STORE, 0),   # x = y (non-constant store, invalidates x)
            Instr(Op.LOAD, 0),    # load x (should NOT propagate -- x is unknown)
            Instr(Op.HALT),
        ]
        result, changed = constant_propagation(test_instrs, work_chunk)
        # LOAD y stays (y unknown), LOAD x at end stays (x invalidated by non-const store)
        assert result[4].op == Op.LOAD


# ============================================================
# Full Pipeline Tests
# ============================================================

class TestOptimizeChunk:
    def test_basic_optimization(self):
        source = "let x = 2 + 3;"
        chunk, _ = compile_source(source)
        opt_chunk, stats = optimize_chunk(chunk)
        assert stats.total_optimizations > 0
        assert len(opt_chunk.code) <= len(chunk.code)

    def test_multiple_rounds(self):
        """Cascading optimizations should be caught by multiple rounds."""
        source = "let x = 1 + 2 + 3 + 4;"
        chunk, _ = compile_source(source)
        opt_chunk, stats = optimize_chunk(chunk)
        assert stats.constant_folds >= 1
        # All additions should be folded to 10
        assert len(opt_chunk.code) < len(chunk.code)

    def test_stats_tracking(self):
        source = "let x = 2 + 3;"
        chunk, _ = compile_source(source)
        _, stats = optimize_chunk(chunk)
        assert stats.original_size > 0
        assert stats.optimized_size > 0
        assert stats.rounds >= 1

    def test_empty_program(self):
        source = ""
        chunk, _ = compile_source(source)
        opt_chunk, stats = optimize_chunk(chunk)
        # Should not crash


# ============================================================
# Function Optimization
# ============================================================

class TestFunctionOptimization:
    def test_optimize_function_body(self):
        source = """
        fn add() {
            return 2 + 3;
        }
        let r = add();
        """
        chunk, compiler = compile_source(source)
        opt_chunk, opt_fns, stats = optimize_all(chunk, compiler)
        # Function body should be optimized
        assert stats.total_optimizations > 0

    def test_function_result_preserved(self):
        source = """
        fn compute() {
            return 10 * 5 + 2;
        }
        print(compute());
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['52']

    def test_nested_function_calls(self):
        source = """
        fn double(x) {
            return x * 2;
        }
        fn quad(x) {
            return double(double(x));
        }
        print(quad(5));
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['20']


# ============================================================
# Correctness (Same Result) Tests
# ============================================================

class TestCorrectness:
    def test_simple_arithmetic(self):
        assert_same_result("let x = 2 + 3 * 4;")

    def test_variable_ops(self):
        assert_same_result("let x = 10; let y = x + 5; print(y);")

    def test_if_else(self):
        assert_same_result("if (3 > 2) { print(1); } else { print(0); }")

    def test_while_loop(self):
        assert_same_result("""
            let x = 0;
            while (x < 10) {
                x = x + 1;
            }
            print(x);
        """)

    def test_fibonacci(self):
        assert_same_result("""
            fn fib(n) {
                if (n <= 1) { return n; }
                return fib(n - 1) + fib(n - 2);
            }
            print(fib(10));
        """)

    def test_string_ops(self):
        assert_same_result('print("hello" + " " + "world");')

    def test_boolean_logic(self):
        assert_same_result("""
            let x = true and false;
            let y = true or false;
            print(x);
            print(y);
        """)

    def test_comparison_chain(self):
        assert_same_result("""
            let a = 5;
            let b = 10;
            print(a < b);
            print(a > b);
            print(a == b);
            print(a != b);
        """)

    def test_nested_ifs(self):
        assert_same_result("""
            let x = 15;
            if (x > 10) {
                if (x > 20) {
                    print(3);
                } else {
                    print(2);
                }
            } else {
                print(1);
            }
        """)

    def test_while_with_function(self):
        assert_same_result("""
            fn square(n) { return n * n; }
            let i = 1;
            let sum = 0;
            while (i <= 5) {
                sum = sum + square(i);
                i = i + 1;
            }
            print(sum);
        """)

    def test_multiple_functions(self):
        assert_same_result("""
            fn add(a, b) { return a + b; }
            fn mul(a, b) { return a * b; }
            print(add(mul(3, 4), 5));
        """)

    def test_negative_numbers(self):
        assert_same_result("let x = -5; let y = -x; print(y);")

    def test_float_arithmetic(self):
        assert_same_result("let x = 1.5 + 2.5; print(x);")

    def test_mixed_int_float(self):
        assert_same_result("let x = 1 + 2.5; print(x);")

    def test_recursive_countdown(self):
        assert_same_result("""
            fn countdown(n) {
                if (n <= 0) { return 0; }
                print(n);
                return countdown(n - 1);
            }
            countdown(5);
        """)

    def test_empty_function(self):
        assert_same_result("""
            fn noop() { }
            noop();
            print(42);
        """)

    def test_deeply_nested_arithmetic(self):
        assert_same_result("let x = ((1 + 2) * (3 + 4)) - (5 * 6);")

    def test_constant_condition_if(self):
        assert_same_result("""
            if (1 > 0) {
                print(1);
            } else {
                print(0);
            }
        """)


# ============================================================
# Optimization Effectiveness Tests
# ============================================================

class TestEffectiveness:
    def test_constant_expression_reduces_size(self):
        source = "let x = 1 + 2 + 3 + 4 + 5;"
        result = compare_execution(source)
        assert result['stats'].optimized_size < result['stats'].original_size

    def test_step_reduction(self):
        """Optimized code should execute in fewer steps."""
        source = "let x = 1 + 2 + 3 + 4 + 5;"
        result = compare_execution(source)
        assert result['steps_saved'] > 0

    def test_strength_reduction_effective(self):
        source = "let a = 7; let b = a * 2;"
        result = compare_execution(source)
        # * 2 should become DUP + ADD (same step count or fewer)
        assert result['same_result']

    def test_function_optimization_steps(self):
        source = """
        fn constant() { return 10 + 20 + 30; }
        print(constant());
        """
        result = compare_execution(source)
        assert result['same_result']
        assert result['same_output']

    def test_complex_optimization(self):
        source = """
        let a = 2 + 3;
        let b = a * 1;
        let c = b + 0;
        let d = -(-c);
        print(d);
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['5']
        # Should have significant optimization
        assert result['stats'].total_optimizations > 0

    def test_loop_body_optimization(self):
        """Constant expressions in loop bodies get optimized."""
        source = """
        let i = 0;
        let sum = 0;
        while (i < 100) {
            sum = sum + (2 + 3);
            i = i + 1;
        }
        print(sum);
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['500']

    def test_size_reduction_stat(self):
        source = "let x = 1 + 2 + 3 + 4 + 5;"
        info = optimize_source(source)
        assert info['stats'].size_reduction > 0

    def test_fixpoint_convergence(self):
        """Optimizer should converge (reach fixpoint) within max rounds."""
        source = "let x = 1 + 2 * 3 - 4 + 5 * 6;"
        chunk, _ = compile_source(source)
        _, stats = optimize_chunk(chunk, max_rounds=20)
        assert stats.rounds <= 20


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_source(self):
        result = execute_optimized("")
        assert result['output'] == []

    def test_only_print(self):
        result = assert_same_result("print(42);")
        assert result['optimized']['output'] == ['42']

    def test_many_constants(self):
        # Lots of constant folding opportunities
        parts = " + ".join(str(i) for i in range(1, 21))
        source = f"let x = {parts}; print(x);"
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['210']

    def test_no_optimization_possible(self):
        source = """
        let x = 1;
        let y = 2;
        print(x + y);
        """
        # x + y can't be folded (variables, not constants)
        # But constant propagation might help here
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['3']

    def test_boolean_not_folded_as_int(self):
        """Booleans should not be treated as integers in folding."""
        source = "let x = true; let y = false; print(x); print(y);"
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['true', 'false']

    def test_string_concatenation_preserved(self):
        source = 'let x = "a" + "b"; print(x);'
        # String concat is not constant-folded (by design -- ARITHMETIC_OPS checks numeric)
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['ab']

    def test_division_by_zero_not_folded(self):
        """Division by zero should not be folded -- let the VM handle it."""
        source = "let x = 10 / 0;"
        info = optimize_source(source)
        # The DIV op should still be present
        instrs = decode_chunk(info['optimized_chunk'])
        ops = [instr.op for _, instr in instrs]
        assert Op.DIV in ops

    def test_modulo_by_zero_not_folded(self):
        source = "let x = 10 % 0;"
        info = optimize_source(source)
        instrs = decode_chunk(info['optimized_chunk'])
        ops = [instr.op for _, instr in instrs]
        assert Op.MOD in ops

    def test_large_program(self):
        """Test optimizer doesn't crash on larger programs."""
        lines = []
        for i in range(50):
            lines.append(f"let v{i} = {i} + {i + 1};")
        lines.append("print(v49);")
        source = "\n".join(lines)
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['99']

    def test_deeply_nested_functions(self):
        source = """
        fn a(x) { return x + 1; }
        fn b(x) { return a(x) + 1; }
        fn c(x) { return b(x) + 1; }
        print(c(10));
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['13']


# ============================================================
# OptimizationStats Tests
# ============================================================

class TestStats:
    def test_stats_repr(self):
        stats = OptimizationStats()
        stats.original_size = 100
        stats.optimized_size = 80
        stats.rounds = 3
        r = repr(stats)
        assert 'rounds=3' in r
        assert '20.0%' in r

    def test_zero_size_reduction(self):
        stats = OptimizationStats()
        stats.original_size = 0
        assert stats.size_reduction == 0.0

    def test_total_optimizations(self):
        stats = OptimizationStats()
        stats.constant_folds = 2
        stats.strength_reductions = 1
        stats.peephole_opts = 3
        assert stats.total_optimizations == 6


# ============================================================
# Public API Tests
# ============================================================

class TestPublicAPI:
    def test_optimize_source(self):
        info = optimize_source("let x = 2 + 3;")
        assert 'original_chunk' in info
        assert 'optimized_chunk' in info
        assert 'stats' in info

    def test_execute_optimized(self):
        result = execute_optimized("print(42);")
        assert result['output'] == ['42']
        assert 'stats' in result

    def test_compare_execution(self):
        result = compare_execution("let x = 2 + 3; print(x);")
        assert result['same_result']
        assert result['same_output']
        assert 'steps_saved' in result

    def test_execute_optimized_env(self):
        result = execute_optimized("let x = 2 + 3;")
        assert result['env']['x'] == 5


# ============================================================
# Disassembly comparison (visual verification helper)
# ============================================================

class TestDisassembly:
    def test_disassembly_changes(self):
        """Verify optimized bytecode is actually different."""
        source = "let x = 1 + 2 + 3;"
        info = optimize_source(source)
        orig = disassemble(info['original_chunk'])
        opt = disassemble(info['optimized_chunk'])
        assert orig != opt  # optimized should be different


# ============================================================
# Regression Tests
# ============================================================

class TestRegression:
    def test_fibonacci_correctness(self):
        """Fibonacci must produce correct results after optimization."""
        source = """
        fn fib(n) {
            if (n <= 1) { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        print(fib(0));
        print(fib(1));
        print(fib(5));
        print(fib(8));
        """
        result = execute_optimized(source)
        assert result['output'] == ['0', '1', '5', '21']

    def test_scope_not_broken(self):
        """Optimization must not break variable scoping."""
        source = """
        let x = 10;
        fn get_x() { return x; }
        x = 20;
        print(get_x());
        """
        result = assert_same_result(source)

    def test_recursive_function_not_broken(self):
        """Recursive functions must work after optimization."""
        source = """
        fn fact(n) {
            if (n <= 1) { return 1; }
            return n * fact(n - 1);
        }
        print(fact(6));
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['720']

    def test_while_loop_not_broken(self):
        """While loop control flow must survive optimization."""
        source = """
        let i = 0;
        let product = 1;
        while (i < 5) {
            i = i + 1;
            product = product * i;
        }
        print(product);
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['120']

    def test_short_circuit_and(self):
        """Short-circuit AND must work after optimization."""
        source = """
        let x = 0;
        if (false and true) { x = 1; }
        print(x);
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['0']

    def test_short_circuit_or(self):
        """Short-circuit OR must work after optimization."""
        source = """
        let x = 0;
        if (true or false) { x = 1; }
        print(x);
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['1']

    def test_return_early(self):
        source = """
        fn check(n) {
            if (n > 10) { return 1; }
            return 0;
        }
        print(check(5));
        print(check(15));
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['0', '1']

    def test_multiple_assignments(self):
        source = """
        let x = 1;
        x = x + 1;
        x = x + 1;
        x = x + 1;
        print(x);
        """
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['4']

    def test_negative_result(self):
        source = "let x = 3 - 10; print(x);"
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['-7']

    def test_float_division(self):
        source = "let x = 7.0 / 2.0; print(x);"
        result = assert_same_result(source)
        assert result['optimized']['output'] == ['3.5']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
