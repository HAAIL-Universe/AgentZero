"""
Bytecode Optimizer for the Stack VM Language
Challenge C014 -- AgentZero Session 015

Multi-pass optimization engine operating on compiled bytecode from C010.

Architecture:
  Source -> Lex -> Parse -> TypeCheck -> Compile -> **Optimize** -> VM

Optimization passes:
  1. Constant folding -- evaluate constant expressions at compile time
  2. Constant propagation -- replace loads with known constants
  3. Strength reduction -- replace expensive ops with cheaper equivalents
  4. Peephole optimization -- pattern-match and simplify instruction sequences
  5. Jump optimization -- thread jumps, eliminate redundant jumps
  6. Dead code elimination -- remove unreachable code

Each pass is independent and composable. The optimizer runs multiple rounds
until a fixpoint (no further changes).

Key design: Jump operands are converted to INSTRUCTION INDICES after decode
and back to BYTE ADDRESSES before encode. This ensures passes that add/remove
instructions don't corrupt jump targets.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Any, Optional

# Import the VM module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    Op, Chunk, FnObject, lex, Parser, Compiler, VM,
    compile_source, execute, disassemble,
)


# ============================================================
# Instruction Representation
# ============================================================

@dataclass
class Instr:
    """A single instruction with optional operand."""
    op: Op
    operand: Any = None
    line: int = 0

    def has_operand(self):
        return self.op in OPS_WITH_OPERAND

    def is_jump(self):
        return self.op in (Op.JUMP, Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE)

    def is_const(self):
        return self.op == Op.CONST

    def is_arithmetic(self):
        return self.op in (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD, Op.NEG)

    def is_comparison(self):
        return self.op in (Op.EQ, Op.NE, Op.LT, Op.GT, Op.LE, Op.GE)

    def is_terminator(self):
        return self.op in (Op.JUMP, Op.HALT, Op.RETURN)


OPS_WITH_OPERAND = frozenset({
    Op.CONST, Op.LOAD, Op.STORE,
    Op.JUMP, Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE,
    Op.CALL,
})


# ============================================================
# Bytecode <-> Instruction List conversion
# ============================================================

def decode_chunk(chunk: Chunk) -> list:
    """Convert flat bytecode to list of (addr, Instr) tuples."""
    instructions = []
    i = 0
    while i < len(chunk.code):
        op = Op(chunk.code[i])
        line = chunk.lines[i] if i < len(chunk.lines) else 0
        if op in OPS_WITH_OPERAND:
            operand = chunk.code[i + 1]
            instructions.append((i, Instr(op, operand, line)))
            i += 2
        else:
            instructions.append((i, Instr(op, None, line)))
            i += 1
    return instructions


def addrs_to_indices(addr_instrs: list) -> list:
    """Convert jump operands from byte addresses to instruction indices.
    Returns list of Instr with index-based jump targets."""
    # Build addr -> index map
    addr_to_idx = {}
    for idx, (addr, instr) in enumerate(addr_instrs):
        addr_to_idx[addr] = idx

    result = []
    for _, instr in addr_instrs:
        if instr.is_jump() and instr.operand is not None:
            target_idx = addr_to_idx.get(instr.operand, instr.operand)
            result.append(Instr(instr.op, target_idx, instr.line))
        else:
            result.append(Instr(instr.op, instr.operand, instr.line))
    return result


def indices_to_addrs(instrs: list) -> list:
    """Convert jump operands from instruction indices to byte addresses.
    Returns list of Instr with byte-address jump targets."""
    # Build index -> byte address map
    idx_to_addr = {}
    addr = 0
    for idx, instr in enumerate(instrs):
        idx_to_addr[idx] = addr
        addr += 2 if instr.has_operand() else 1

    result = []
    for instr in instrs:
        if instr.is_jump() and instr.operand is not None:
            target_addr = idx_to_addr.get(instr.operand, instr.operand)
            result.append(Instr(instr.op, target_addr, instr.line))
        else:
            result.append(Instr(instr.op, instr.operand, instr.line))
    return result


def encode_instructions(instrs: list, old_chunk: Chunk) -> Chunk:
    """Convert list of Instr objects back to a Chunk.
    Assumes jump operands are already byte addresses."""
    new_chunk = Chunk()
    new_chunk.constants = list(old_chunk.constants)
    new_chunk.names = list(old_chunk.names)

    for instr in instrs:
        new_chunk.code.append(instr.op)
        new_chunk.lines.append(instr.line)
        if instr.has_operand():
            new_chunk.code.append(instr.operand)
            new_chunk.lines.append(instr.line)

    return new_chunk


# ============================================================
# Pass 1: Constant Folding
# ============================================================

ARITHMETIC_OPS = {
    Op.ADD: lambda a, b: a + b,
    Op.SUB: lambda a, b: a - b,
    Op.MUL: lambda a, b: a * b,
    Op.DIV: lambda a, b: (a // b if isinstance(a, int) and isinstance(b, int) else a / b) if b != 0 else None,
    Op.MOD: lambda a, b: a % b if b != 0 else None,
}

COMPARISON_OPS = {
    Op.EQ: lambda a, b: a == b,
    Op.NE: lambda a, b: a != b,
    Op.LT: lambda a, b: a < b,
    Op.GT: lambda a, b: a > b,
    Op.LE: lambda a, b: a <= b,
    Op.GE: lambda a, b: a >= b,
}


def constant_fold(instrs: list, chunk: Chunk) -> tuple:
    """Fold constant expressions: CONST a, CONST b, OP -> CONST result.
    Also folds unary: CONST a, NEG -> CONST -a; CONST a, NOT -> CONST (not a).
    Jump operands are instruction indices -- must remap when removing instructions.
    Returns (new_instrs, new_chunk, changed)."""
    new_chunk = Chunk()
    new_chunk.constants = list(chunk.constants)
    new_chunk.names = list(chunk.names)
    result = list(instrs)
    changed = False

    progress = True
    while progress:
        progress = False
        i = 0
        new_result = []
        # Build old->new index remap
        old_to_new = {}

        while i < len(result):
            old_to_new[i] = len(new_result)

            # Binary fold: CONST a, CONST b, BINOP -> CONST result
            if (i + 2 < len(result)
                    and result[i].op == Op.CONST
                    and result[i + 1].op == Op.CONST
                    and result[i + 2].op in (ARITHMETIC_OPS | COMPARISON_OPS)):
                a_val = new_chunk.constants[result[i].operand]
                b_val = new_chunk.constants[result[i + 1].operand]
                op = result[i + 2].op

                if op in ARITHMETIC_OPS:
                    if (isinstance(a_val, (int, float)) and isinstance(b_val, (int, float))
                            and not isinstance(a_val, bool) and not isinstance(b_val, bool)):
                        fold_fn = ARITHMETIC_OPS[op]
                        folded = fold_fn(a_val, b_val)
                        if folded is not None:
                            idx = new_chunk.add_constant(folded)
                            new_result.append(Instr(Op.CONST, idx, result[i].line))
                            old_to_new[i + 1] = len(new_result)
                            old_to_new[i + 2] = len(new_result)
                            i += 3
                            progress = True
                            changed = True
                            continue
                elif op in COMPARISON_OPS:
                    if (type(a_val) is type(b_val) or (
                            isinstance(a_val, (int, float)) and isinstance(b_val, (int, float))
                            and not isinstance(a_val, bool) and not isinstance(b_val, bool))):
                        fold_fn = COMPARISON_OPS[op]
                        try:
                            folded = fold_fn(a_val, b_val)
                            idx = new_chunk.add_constant(folded)
                            new_result.append(Instr(Op.CONST, idx, result[i].line))
                            old_to_new[i + 1] = len(new_result)
                            old_to_new[i + 2] = len(new_result)
                            i += 3
                            progress = True
                            changed = True
                            continue
                        except TypeError:
                            pass

            # Unary fold: CONST a, NEG -> CONST -a
            if (i + 1 < len(result)
                    and result[i].op == Op.CONST
                    and result[i + 1].op == Op.NEG):
                val = new_chunk.constants[result[i].operand]
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    idx = new_chunk.add_constant(-val)
                    new_result.append(Instr(Op.CONST, idx, result[i].line))
                    old_to_new[i + 1] = len(new_result)
                    i += 2
                    progress = True
                    changed = True
                    continue

            # Unary fold: CONST a, NOT -> CONST (not a)
            if (i + 1 < len(result)
                    and result[i].op == Op.CONST
                    and result[i + 1].op == Op.NOT):
                val = new_chunk.constants[result[i].operand]
                if isinstance(val, bool):
                    idx = new_chunk.add_constant(not val)
                    new_result.append(Instr(Op.CONST, idx, result[i].line))
                    old_to_new[i + 1] = len(new_result)
                    i += 2
                    progress = True
                    changed = True
                    continue

            new_result.append(result[i])
            i += 1

        # Fill remaining mappings
        if i <= len(result):
            old_to_new[len(result)] = len(new_result)

        # Remap jump targets
        if progress:
            for j, instr in enumerate(new_result):
                if instr.is_jump() and instr.operand is not None:
                    if instr.operand in old_to_new:
                        new_result[j] = Instr(instr.op, old_to_new[instr.operand], instr.line)

        result = new_result

    return result, new_chunk, changed


# ============================================================
# Pass 2: Constant Propagation
# ============================================================

def constant_propagation(instrs: list, chunk: Chunk) -> tuple:
    """Track known constant values through STORE/LOAD sequences.
    If a variable is stored with a constant and loaded before any reassignment,
    replace the LOAD with the constant.
    Returns (new_instrs, changed).

    Conservative single-block analysis: any jump or jump target resets knowledge.
    STORE at a join point does NOT record -- stack value is path-dependent."""
    result = list(instrs)
    changed = False

    # Find all jump targets (instruction indices that are branch destinations)
    jump_targets = set()
    for instr in result:
        if instr.is_jump() and instr.operand is not None:
            jump_targets.add(instr.operand)

    # Track: name_idx -> const_idx
    known = {}

    for i, instr in enumerate(result):
        at_join = i in jump_targets
        if at_join:
            known.clear()
        if instr.is_jump() or instr.op in (Op.CALL, Op.RETURN, Op.HALT):
            known.clear()
            continue

        if instr.op == Op.STORE:
            # Only record CONST->STORE when not at a join point
            # At join points, the stack value depends on which path was taken
            if not at_join and i > 0 and result[i - 1].op == Op.CONST:
                known[instr.operand] = result[i - 1].operand
            else:
                if instr.operand in known:
                    del known[instr.operand]

        elif instr.op == Op.LOAD and instr.operand in known:
            const_idx = known[instr.operand]
            result[i] = Instr(Op.CONST, const_idx, instr.line)
            changed = True

    return result, changed


# ============================================================
# Pass 3: Strength Reduction
# ============================================================

def strength_reduce(instrs: list, chunk: Chunk) -> tuple:
    """Replace expensive operations with cheaper equivalents.
    Does NOT change instruction count (only replaces), so no jump remapping needed
    for identity removals that DO change count -- those use _remap_jumps.
    Returns (new_instrs, changed)."""
    result = []
    changed = False
    i = 0
    old_to_new = {}

    while i < len(instrs):
        old_to_new[i] = len(result)

        if (i + 1 < len(instrs) and instrs[i].op == Op.CONST):
            val = chunk.constants[instrs[i].operand]
            next_op = instrs[i + 1].op

            # a + 0 = a -> remove CONST 0 and ADD
            if val == 0 and not isinstance(val, bool) and next_op == Op.ADD:
                old_to_new[i + 1] = len(result)
                i += 2
                changed = True
                continue
            if val == 0 and not isinstance(val, bool) and next_op == Op.SUB:
                old_to_new[i + 1] = len(result)
                i += 2
                changed = True
                continue
            if val == 1 and not isinstance(val, bool) and next_op == Op.MUL:
                old_to_new[i + 1] = len(result)
                i += 2
                changed = True
                continue
            if val == 1 and not isinstance(val, bool) and next_op == Op.DIV:
                old_to_new[i + 1] = len(result)
                i += 2
                changed = True
                continue

            # a * 2 -> DUP, ADD (same count, just replacement)
            if val == 2 and not isinstance(val, bool) and next_op == Op.MUL:
                result.append(Instr(Op.DUP, None, instrs[i].line))
                old_to_new[i + 1] = len(result)
                result.append(Instr(Op.ADD, None, instrs[i + 1].line))
                i += 2
                changed = True
                continue

        # Double negation: NEG, NEG -> remove both
        if (i + 1 < len(instrs)
                and instrs[i].op == Op.NEG and instrs[i + 1].op == Op.NEG):
            old_to_new[i + 1] = len(result)
            i += 2
            changed = True
            continue

        # Double NOT
        if (i + 1 < len(instrs)
                and instrs[i].op == Op.NOT and instrs[i + 1].op == Op.NOT):
            old_to_new[i + 1] = len(result)
            i += 2
            changed = True
            continue

        result.append(instrs[i])
        i += 1

    old_to_new[len(instrs)] = len(result)

    # Remap jumps
    if changed:
        _remap_jumps(result, old_to_new)

    return result, changed


# ============================================================
# Pass 4: Peephole Optimization
# ============================================================

def peephole(instrs: list, chunk: Chunk) -> tuple:
    """Pattern-match and simplify instruction sequences.
    Returns (new_instrs, changed)."""
    # Find jump targets to avoid transforming at join points
    jump_targets = set()
    for instr in instrs:
        if instr.is_jump() and instr.operand is not None:
            jump_targets.add(instr.operand)

    result = []
    changed = False
    i = 0
    old_to_new = {}

    while i < len(instrs):
        old_to_new[i] = len(result)

        # STORE x, LOAD x -> DUP, STORE x
        # But NOT if the LOAD is a jump target (loop back-edges target it)
        if (i + 1 < len(instrs)
                and instrs[i].op == Op.STORE
                and instrs[i + 1].op == Op.LOAD
                and instrs[i].operand == instrs[i + 1].operand
                and (i + 1) not in jump_targets):
            result.append(Instr(Op.DUP, None, instrs[i].line))
            old_to_new[i + 1] = len(result)
            result.append(Instr(Op.STORE, instrs[i].operand, instrs[i].line))
            i += 2
            changed = True
            continue

        # CONST x, POP -> remove both
        if (i + 1 < len(instrs)
                and instrs[i].op == Op.CONST and instrs[i + 1].op == Op.POP):
            old_to_new[i + 1] = len(result)
            i += 2
            changed = True
            continue

        # LOAD x, POP -> remove both
        if (i + 1 < len(instrs)
                and instrs[i].op == Op.LOAD and instrs[i + 1].op == Op.POP):
            old_to_new[i + 1] = len(result)
            i += 2
            changed = True
            continue

        # DUP, POP -> remove both
        if (i + 1 < len(instrs)
                and instrs[i].op == Op.DUP and instrs[i + 1].op == Op.POP):
            old_to_new[i + 1] = len(result)
            i += 2
            changed = True
            continue

        result.append(instrs[i])
        i += 1

    old_to_new[len(instrs)] = len(result)

    if changed:
        _remap_jumps(result, old_to_new)

    return result, changed


# ============================================================
# Pass 5: Jump Optimization
# ============================================================

def optimize_jumps(instrs: list, chunk: Chunk) -> tuple:
    """Thread jumps and eliminate redundant patterns.
    Jump operands are instruction indices.
    Returns (new_instrs, changed)."""
    result = list(instrs)
    changed = False

    # Jump threading: if a jump targets another unconditional jump, thread through
    for i, instr in enumerate(result):
        if instr.is_jump() and instr.operand is not None:
            target_idx = instr.operand
            if 0 <= target_idx < len(result) and result[target_idx].op == Op.JUMP:
                final_target = result[target_idx].operand
                if final_target != instr.operand:
                    result[i] = Instr(instr.op, final_target, instr.line)
                    changed = True

    # Jump to next instruction -> remove
    new_result = []
    old_to_new = {}
    for i, instr in enumerate(result):
        old_to_new[i] = len(new_result)
        if instr.op == Op.JUMP and instr.operand == i + 1:
            changed = True
            continue
        new_result.append(instr)
    old_to_new[len(result)] = len(new_result)

    if len(new_result) < len(result):
        _remap_jumps(new_result, old_to_new)

    return new_result, changed


# ============================================================
# Pass 6: Dead Code Elimination
# ============================================================

def eliminate_dead_code(instrs: list, chunk: Chunk) -> tuple:
    """Remove unreachable instructions using reachability from instruction 0.
    Jump operands are instruction indices.
    Returns (new_instrs, changed)."""
    if not instrs:
        return instrs, False

    # BFS reachability from instruction 0
    reachable = set()
    queue = [0]
    while queue:
        idx = queue.pop(0)
        if idx in reachable or idx < 0 or idx >= len(instrs):
            continue
        reachable.add(idx)
        instr = instrs[idx]

        if instr.op == Op.HALT or instr.op == Op.RETURN:
            continue  # no successors

        if instr.op == Op.JUMP:
            # Only successor is the jump target
            if instr.operand is not None:
                queue.append(instr.operand)
            continue

        if instr.op in (Op.JUMP_IF_FALSE, Op.JUMP_IF_TRUE):
            # Two successors: jump target and fall-through
            if instr.operand is not None:
                queue.append(instr.operand)
            queue.append(idx + 1)
            continue

        # Normal instruction: fall through
        queue.append(idx + 1)

    if len(reachable) == len(instrs):
        return instrs, False

    # Remove unreachable instructions, remap indices
    old_to_new = {}
    new_result = []
    for i, instr in enumerate(instrs):
        old_to_new[i] = len(new_result)
        if i in reachable:
            new_result.append(instr)
    old_to_new[len(instrs)] = len(new_result)

    _remap_jumps(new_result, old_to_new)

    return new_result, True


# ============================================================
# Jump Remapping Helper
# ============================================================

def _remap_jumps(instrs: list, old_to_new: dict):
    """Remap jump operands in-place using old_to_new index mapping.
    For targets that map between entries, find the closest valid new index."""
    for i, instr in enumerate(instrs):
        if instr.is_jump() and instr.operand is not None:
            old_target = instr.operand
            if old_target in old_to_new:
                instrs[i] = Instr(instr.op, old_to_new[old_target], instr.line)
            else:
                # Find closest mapped index >= old_target
                candidates = [k for k in old_to_new if k >= old_target]
                if candidates:
                    instrs[i] = Instr(instr.op, old_to_new[min(candidates)], instr.line)


# ============================================================
# Optimizer Pipeline
# ============================================================

class OptimizationStats:
    """Track what optimizations were applied."""
    def __init__(self):
        self.rounds = 0
        self.constant_folds = 0
        self.strength_reductions = 0
        self.peephole_opts = 0
        self.dead_code_eliminations = 0
        self.jump_optimizations = 0
        self.constant_propagations = 0
        self.original_size = 0
        self.optimized_size = 0

    @property
    def total_optimizations(self):
        return (self.constant_folds + self.strength_reductions +
                self.peephole_opts + self.dead_code_eliminations +
                self.jump_optimizations + self.constant_propagations)

    @property
    def size_reduction(self):
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.optimized_size / self.original_size)

    def __repr__(self):
        return (f"OptStats(rounds={self.rounds}, opts={self.total_optimizations}, "
                f"size={self.original_size}->{self.optimized_size}, "
                f"reduction={self.size_reduction:.1%})")


def optimize_chunk(chunk: Chunk, max_rounds: int = 10) -> tuple:
    """Run all optimization passes on a chunk until fixpoint.
    Returns (optimized_chunk, stats)."""
    stats = OptimizationStats()
    stats.original_size = len(chunk.code)

    # Decode and convert to index-based jumps
    addr_instrs = decode_chunk(chunk)
    instrs = addrs_to_indices(addr_instrs)

    # Working chunk tracks constants/names
    work_chunk = Chunk()
    work_chunk.constants = list(chunk.constants)
    work_chunk.names = list(chunk.names)

    for round_num in range(max_rounds):
        any_changed = False

        # Pass 1: Constant folding
        instrs, work_chunk, changed = constant_fold(instrs, work_chunk)
        if changed:
            any_changed = True
            stats.constant_folds += 1

        # Pass 2: Constant propagation
        instrs, changed = constant_propagation(instrs, work_chunk)
        if changed:
            any_changed = True
            stats.constant_propagations += 1

        # Pass 3: Strength reduction
        instrs, changed = strength_reduce(instrs, work_chunk)
        if changed:
            any_changed = True
            stats.strength_reductions += 1

        # Pass 4: Peephole
        instrs, changed = peephole(instrs, work_chunk)
        if changed:
            any_changed = True
            stats.peephole_opts += 1

        # Pass 5: Jump optimization
        instrs, changed = optimize_jumps(instrs, work_chunk)
        if changed:
            any_changed = True
            stats.jump_optimizations += 1

        # Pass 6: Dead code elimination
        instrs, changed = eliminate_dead_code(instrs, work_chunk)
        if changed:
            any_changed = True
            stats.dead_code_eliminations += 1

        stats.rounds = round_num + 1
        if not any_changed:
            break

    # Convert back to byte-address jumps and encode
    instrs = indices_to_addrs(instrs)
    result = encode_instructions(instrs, work_chunk)
    stats.optimized_size = len(result.code)

    return result, stats


def optimize_all(chunk: Chunk, compiler: Compiler = None, max_rounds: int = 10) -> tuple:
    """Optimize main chunk and all function chunks.
    Returns (optimized_chunk, optimized_functions, total_stats)."""
    opt_chunk, main_stats = optimize_chunk(chunk, max_rounds)

    opt_functions = {}
    fn_stats_list = []
    if compiler:
        for name, fn_obj in compiler.functions.items():
            opt_fn_chunk, fn_stats = optimize_chunk(fn_obj.chunk, max_rounds)
            opt_fn = FnObject(fn_obj.name, fn_obj.arity, opt_fn_chunk)
            opt_functions[name] = opt_fn
            fn_stats_list.append(fn_stats)

            for i, c in enumerate(opt_chunk.constants):
                if isinstance(c, FnObject) and c.name == name:
                    opt_chunk.constants[i] = opt_fn

    # Aggregate stats
    total = OptimizationStats()
    total.rounds = main_stats.rounds
    total.original_size = main_stats.original_size
    total.optimized_size = main_stats.optimized_size
    total.constant_folds = main_stats.constant_folds
    total.strength_reductions = main_stats.strength_reductions
    total.peephole_opts = main_stats.peephole_opts
    total.dead_code_eliminations = main_stats.dead_code_eliminations
    total.jump_optimizations = main_stats.jump_optimizations
    total.constant_propagations = main_stats.constant_propagations

    for fs in fn_stats_list:
        total.original_size += fs.original_size
        total.optimized_size += fs.optimized_size
        total.constant_folds += fs.constant_folds
        total.strength_reductions += fs.strength_reductions
        total.peephole_opts += fs.peephole_opts
        total.dead_code_eliminations += fs.dead_code_eliminations
        total.jump_optimizations += fs.jump_optimizations
        total.constant_propagations += fs.constant_propagations
        total.rounds = max(total.rounds, fs.rounds)

    return opt_chunk, opt_functions, total


# ============================================================
# Public API
# ============================================================

def optimize_source(source: str) -> dict:
    """Compile, optimize, and return analysis."""
    chunk, compiler = compile_source(source)
    opt_chunk, opt_functions, stats = optimize_all(chunk, compiler)
    return {
        'original_chunk': chunk,
        'optimized_chunk': opt_chunk,
        'compiler': compiler,
        'optimized_functions': opt_functions,
        'stats': stats,
    }


def execute_optimized(source: str, trace=False) -> dict:
    """Compile, optimize, and execute source code."""
    chunk, compiler = compile_source(source)
    opt_chunk, opt_functions, stats = optimize_all(chunk, compiler)

    vm = VM(opt_chunk, trace=trace)
    result = vm.run()

    return {
        'result': result,
        'output': vm.output,
        'env': vm.env,
        'steps': vm.step_count,
        'stats': stats,
    }


def compare_execution(source: str) -> dict:
    """Execute source both with and without optimization. Compare results."""
    unopt = execute(source)
    opt = execute_optimized(source)

    return {
        'unoptimized': unopt,
        'optimized': opt,
        'steps_saved': unopt['steps'] - opt['steps'],
        'same_result': unopt['result'] == opt['result'],
        'same_output': unopt['output'] == opt['output'],
        'stats': opt['stats'],
    }
