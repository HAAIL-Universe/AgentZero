"""
C022: Meta-Evolver -- Genetic Programming on the Stack VM

Composes C012 (Code Evolver) + C010 (Stack VM).

Evolves programs represented as C010 AST nodes, compiles them to bytecode,
and evaluates fitness by executing on the VM. Supports:

- Expression evolution (numeric computation from inputs)
- Imperative evolution (programs with loops, variables, conditionals)
- Statement-level mutation and crossover
- Multi-objective fitness (correctness + efficiency + simplicity)
- Bloat control via depth limits and parsimony pressure
- Island model for population diversity
- Program simplification (constant folding, dead code removal)

Architecture:
  Random AST -> Mutate/Crossover -> Compile (C010) -> Execute (VM) -> Fitness
"""

import sys
import os
import random
import math
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum, auto

# Import C010 stack VM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))
from stack_vm import (
    lex, Parser, Compiler, VM, Chunk, Op, FnObject,
    IntLit, FloatLit, BoolLit, StringLit, Var, UnaryOp, BinOp,
    Assign, LetDecl, Block, IfStmt, WhileStmt, FnDecl, CallExpr,
    ReturnStmt, PrintStmt, Program, VMError, CompileError, ParseError,
)


# ============================================================
# Program Representation
# ============================================================

class ProgramType(Enum):
    """What kind of program are we evolving?"""
    EXPRESSION = auto()    # single expression, result is stack top
    IMPERATIVE = auto()    # statements with a designated result variable


@dataclass
class EvolvableProgram:
    """A program that can be evolved."""
    ast: Any                     # C010 AST node or list of statements
    program_type: ProgramType
    result_var: str = "result"   # for IMPERATIVE: which var holds the answer
    input_vars: list = field(default_factory=list)  # variable names for inputs

    def copy(self):
        return EvolvableProgram(
            ast=copy.deepcopy(self.ast),
            program_type=self.program_type,
            result_var=self.result_var,
            input_vars=list(self.input_vars),
        )


# ============================================================
# Random AST Generation
# ============================================================

EXPR_BINARY_OPS = ['+', '-', '*', '/', '%']
EXPR_COMPARE_OPS = ['<', '>', '<=', '>=', '==', '!=']
EXPR_UNARY_OPS = ['-']


def random_int(rng: random.Random, lo: int = -10, hi: int = 10) -> IntLit:
    return IntLit(rng.randint(lo, hi))


def random_float(rng: random.Random, lo: float = -5.0, hi: float = 5.0) -> FloatLit:
    return FloatLit(round(rng.uniform(lo, hi), 2))


def random_bool(rng: random.Random) -> BoolLit:
    return BoolLit(rng.choice([True, False]))


def random_terminal(var_names: list, rng: random.Random,
                    const_range: tuple = (-10, 10)) -> Any:
    """Generate a random terminal (literal or variable reference)."""
    if var_names and rng.random() < 0.6:
        return Var(rng.choice(var_names))
    r = rng.random()
    if r < 0.7:
        return IntLit(rng.randint(const_range[0], const_range[1]))
    elif r < 0.9:
        return FloatLit(round(rng.uniform(const_range[0], const_range[1]), 2))
    else:
        return BoolLit(rng.choice([True, False]))


def random_expr(var_names: list, rng: random.Random, max_depth: int = 4,
                depth: int = 0, const_range: tuple = (-10, 10)) -> Any:
    """Generate a random expression AST node."""
    if depth >= max_depth - 1:
        return random_terminal(var_names, rng, const_range)

    terminal_prob = 0.3 + 0.15 * depth
    if rng.random() < terminal_prob:
        return random_terminal(var_names, rng, const_range)

    choice = rng.random()
    if choice < 0.15:
        # Unary
        op = rng.choice(EXPR_UNARY_OPS)
        operand = random_expr(var_names, rng, max_depth, depth + 1, const_range)
        return UnaryOp(op, operand)
    elif choice < 0.85:
        # Binary arithmetic
        op = rng.choice(EXPR_BINARY_OPS)
        left = random_expr(var_names, rng, max_depth, depth + 1, const_range)
        right = random_expr(var_names, rng, max_depth, depth + 1, const_range)
        return BinOp(op, left, right)
    else:
        # Comparison
        op = rng.choice(EXPR_COMPARE_OPS)
        left = random_expr(var_names, rng, max_depth, depth + 1, const_range)
        right = random_expr(var_names, rng, max_depth, depth + 1, const_range)
        return BinOp(op, left, right)


def random_condition(var_names: list, rng: random.Random, max_depth: int = 3,
                     depth: int = 0) -> Any:
    """Generate a random boolean expression."""
    if depth >= max_depth - 1:
        # Simple comparison
        left = random_terminal(var_names, rng)
        right = random_terminal(var_names, rng)
        op = rng.choice(EXPR_COMPARE_OPS)
        return BinOp(op, left, right)

    choice = rng.random()
    if choice < 0.6:
        left = random_expr(var_names, rng, max_depth, depth + 1)
        right = random_expr(var_names, rng, max_depth, depth + 1)
        op = rng.choice(EXPR_COMPARE_OPS)
        return BinOp(op, left, right)
    elif choice < 0.8:
        # and/or
        left = random_condition(var_names, rng, max_depth, depth + 1)
        right = random_condition(var_names, rng, max_depth, depth + 1)
        op = rng.choice(['and', 'or'])
        return BinOp(op, left, right)
    else:
        # not
        operand = random_condition(var_names, rng, max_depth, depth + 1)
        return UnaryOp('not', operand)


def random_statement(var_names: list, all_vars: list, rng: random.Random,
                     max_depth: int = 3, depth: int = 0) -> Any:
    """Generate a random statement.
    var_names: variables available for reading
    all_vars: all declared variables (for assignment targets)
    """
    if depth >= max_depth - 1:
        # Simple assignment
        if all_vars:
            target = rng.choice(all_vars)
            value = random_expr(var_names, rng, max_depth=3)
            return Assign(target, value)
        return Assign("result", random_expr(var_names, rng, max_depth=3))

    choice = rng.random()

    if choice < 0.4:
        # Assignment
        if all_vars:
            target = rng.choice(all_vars)
        else:
            target = "result"
        value = random_expr(var_names, rng, max_depth=3)
        return Assign(target, value)

    elif choice < 0.7:
        # If statement
        cond = random_condition(var_names, rng, max_depth=3)
        then_stmts = [random_statement(var_names, all_vars, rng, max_depth, depth + 1)]
        then_body = Block(then_stmts)
        else_body = None
        if rng.random() < 0.5:
            else_stmts = [random_statement(var_names, all_vars, rng, max_depth, depth + 1)]
            else_body = Block(else_stmts)
        return IfStmt(cond, then_body, else_body)

    else:
        # While loop (with bounded iteration via a counter)
        counter = f"_i{depth}"
        if counter not in all_vars:
            all_vars.append(counter)
        limit = rng.randint(2, 10)
        cond = BinOp('<', Var(counter), IntLit(limit))
        inc = Assign(counter, BinOp('+', Var(counter), IntLit(1)))
        body_stmt = random_statement(var_names, all_vars, rng, max_depth, depth + 1)
        body = Block([body_stmt, inc])
        return WhileStmt(cond, body)


def random_program(input_vars: list, rng: random.Random,
                   program_type: ProgramType = ProgramType.EXPRESSION,
                   max_depth: int = 4, const_range: tuple = (-10, 10),
                   num_statements: int = 3) -> EvolvableProgram:
    """Generate a random evolvable program."""
    if program_type == ProgramType.EXPRESSION:
        expr = random_expr(input_vars, rng, max_depth, const_range=const_range)
        return EvolvableProgram(
            ast=expr,
            program_type=ProgramType.EXPRESSION,
            input_vars=list(input_vars),
        )
    else:
        # Imperative program
        result_var = "result"
        all_vars = [result_var] + list(input_vars)
        # Temporary variables
        num_temps = rng.randint(0, 2)
        for i in range(num_temps):
            all_vars.append(f"t{i}")

        stmts = []
        actual_count = rng.randint(1, num_statements)
        for _ in range(actual_count):
            stmt = random_statement(input_vars + [result_var] + [f"t{i}" for i in range(num_temps)],
                                    all_vars, rng, max_depth=max_depth)
            stmts.append(stmt)

        return EvolvableProgram(
            ast=stmts,
            program_type=ProgramType.IMPERATIVE,
            result_var=result_var,
            input_vars=list(input_vars),
        )


# ============================================================
# AST Utilities
# ============================================================

def ast_depth(node) -> int:
    """Calculate depth of an AST node."""
    if isinstance(node, (IntLit, FloatLit, BoolLit, StringLit, Var)):
        return 1
    elif isinstance(node, UnaryOp):
        return 1 + ast_depth(node.operand)
    elif isinstance(node, BinOp):
        return 1 + max(ast_depth(node.left), ast_depth(node.right))
    elif isinstance(node, Assign):
        return 1 + ast_depth(node.value)
    elif isinstance(node, LetDecl):
        return 1 + ast_depth(node.value)
    elif isinstance(node, IfStmt):
        d = 1 + ast_depth(node.cond)
        d = max(d, 1 + ast_depth(node.then_body))
        if node.else_body:
            d = max(d, 1 + ast_depth(node.else_body))
        return d
    elif isinstance(node, WhileStmt):
        return 1 + max(ast_depth(node.cond), ast_depth(node.body))
    elif isinstance(node, Block):
        if not node.stmts:
            return 1
        return 1 + max(ast_depth(s) for s in node.stmts)
    elif isinstance(node, PrintStmt):
        return 1 + ast_depth(node.value)
    elif isinstance(node, ReturnStmt):
        return 1 + (ast_depth(node.value) if node.value else 0)
    elif isinstance(node, list):
        if not node:
            return 0
        return max(ast_depth(s) for s in node)
    return 1


def ast_size(node) -> int:
    """Count nodes in AST."""
    if isinstance(node, (IntLit, FloatLit, BoolLit, StringLit, Var)):
        return 1
    elif isinstance(node, UnaryOp):
        return 1 + ast_size(node.operand)
    elif isinstance(node, BinOp):
        return 1 + ast_size(node.left) + ast_size(node.right)
    elif isinstance(node, Assign):
        return 1 + ast_size(node.value)
    elif isinstance(node, LetDecl):
        return 1 + ast_size(node.value)
    elif isinstance(node, IfStmt):
        s = 1 + ast_size(node.cond) + ast_size(node.then_body)
        if node.else_body:
            s += ast_size(node.else_body)
        return s
    elif isinstance(node, WhileStmt):
        return 1 + ast_size(node.cond) + ast_size(node.body)
    elif isinstance(node, Block):
        return 1 + sum(ast_size(s) for s in node.stmts)
    elif isinstance(node, PrintStmt):
        return 1 + ast_size(node.value)
    elif isinstance(node, ReturnStmt):
        return 1 + (ast_size(node.value) if node.value else 0)
    elif isinstance(node, list):
        return sum(ast_size(s) for s in node)
    return 1


def collect_expr_nodes(node, path=None) -> list:
    """Collect all expression sub-nodes with their paths (for mutation/crossover)."""
    if path is None:
        path = []
    result = [(node, list(path))]

    if isinstance(node, UnaryOp):
        result.extend(collect_expr_nodes(node.operand, path + ['operand']))
    elif isinstance(node, BinOp):
        result.extend(collect_expr_nodes(node.left, path + ['left']))
        result.extend(collect_expr_nodes(node.right, path + ['right']))
    elif isinstance(node, Assign):
        result.extend(collect_expr_nodes(node.value, path + ['value']))
    elif isinstance(node, LetDecl):
        result.extend(collect_expr_nodes(node.value, path + ['value']))
    elif isinstance(node, IfStmt):
        result.extend(collect_expr_nodes(node.cond, path + ['cond']))
        result.extend(collect_expr_nodes(node.then_body, path + ['then_body']))
        if node.else_body:
            result.extend(collect_expr_nodes(node.else_body, path + ['else_body']))
    elif isinstance(node, WhileStmt):
        result.extend(collect_expr_nodes(node.cond, path + ['cond']))
        result.extend(collect_expr_nodes(node.body, path + ['body']))
    elif isinstance(node, Block):
        for i, stmt in enumerate(node.stmts):
            result.extend(collect_expr_nodes(stmt, path + ['stmts', i]))
    elif isinstance(node, PrintStmt):
        result.extend(collect_expr_nodes(node.value, path + ['value']))
    elif isinstance(node, ReturnStmt):
        if node.value:
            result.extend(collect_expr_nodes(node.value, path + ['value']))

    return result


def get_at_path(root, path):
    """Get a node at a given path."""
    current = root
    for step in path:
        if isinstance(step, int):
            current = current[step]
        else:
            current = getattr(current, step)
    return current


def set_at_path(root, path, value):
    """Set a node at a given path (mutates in place)."""
    if not path:
        return value  # can't set root in place
    current = root
    for step in path[:-1]:
        if isinstance(step, int):
            current = current[step]
        else:
            current = getattr(current, step)
    last = path[-1]
    if isinstance(last, int):
        current[last] = value
    else:
        setattr(current, last, value)
    return root


def is_expression(node) -> bool:
    """Check if a node is a pure expression (not a statement)."""
    return isinstance(node, (IntLit, FloatLit, BoolLit, StringLit, Var, UnaryOp, BinOp))


# ============================================================
# Genetic Operators
# ============================================================

def mutate_point(program: EvolvableProgram, rng: random.Random,
                 const_range: tuple = (-10, 10)) -> EvolvableProgram:
    """Point mutation: change a single node's value (same structure)."""
    prog = program.copy()
    nodes = collect_expr_nodes(prog.ast)
    if not nodes:
        return prog

    node, path = rng.choice(nodes)

    if isinstance(node, IntLit):
        if rng.random() < 0.5:
            node.value = node.value + rng.randint(-3, 3)
        else:
            node.value = rng.randint(const_range[0], const_range[1])
        if path:
            set_at_path(prog.ast, path, node)
    elif isinstance(node, FloatLit):
        if rng.random() < 0.5:
            node.value = round(node.value + rng.gauss(0, 1), 2)
        else:
            node.value = round(rng.uniform(const_range[0], const_range[1]), 2)
        if path:
            set_at_path(prog.ast, path, node)
    elif isinstance(node, BoolLit):
        node.value = not node.value
        if path:
            set_at_path(prog.ast, path, node)
    elif isinstance(node, Var):
        if prog.input_vars:
            all_available = list(prog.input_vars)
            if prog.program_type == ProgramType.IMPERATIVE:
                all_available.append(prog.result_var)
            node.name = rng.choice(all_available) if all_available else node.name
            if path:
                set_at_path(prog.ast, path, node)
    elif isinstance(node, BinOp):
        if node.op in EXPR_BINARY_OPS:
            node.op = rng.choice(EXPR_BINARY_OPS)
        elif node.op in EXPR_COMPARE_OPS:
            node.op = rng.choice(EXPR_COMPARE_OPS)
        elif node.op in ('and', 'or'):
            node.op = rng.choice(['and', 'or'])
    elif isinstance(node, UnaryOp):
        if node.op == '-':
            pass  # only one arithmetic unary op
        elif node.op == 'not':
            pass  # only one logical unary op

    return prog


def mutate_subtree(program: EvolvableProgram, rng: random.Random,
                   max_depth: int = 3, const_range: tuple = (-10, 10)) -> EvolvableProgram:
    """Subtree mutation: replace a random subtree with a new random one."""
    prog = program.copy()
    nodes = collect_expr_nodes(prog.ast)

    # Filter to expression nodes only (safe to replace)
    expr_nodes = [(n, p) for n, p in nodes if is_expression(n) and p]
    if not expr_nodes:
        # Mutate entire expression if it's an expression program
        if prog.program_type == ProgramType.EXPRESSION:
            prog.ast = random_expr(prog.input_vars, rng, max_depth, const_range=const_range)
        return prog

    _, path = rng.choice(expr_nodes)
    new_subtree = random_expr(prog.input_vars, rng, max_depth, const_range=const_range)
    set_at_path(prog.ast, path, new_subtree)
    return prog


def mutate_hoist(program: EvolvableProgram, rng: random.Random) -> EvolvableProgram:
    """Hoist mutation: replace tree with a subtree (simplification)."""
    if program.program_type != ProgramType.EXPRESSION:
        return program.copy()

    prog = program.copy()
    nodes = collect_expr_nodes(prog.ast)
    expr_nodes = [(n, p) for n, p in nodes if is_expression(n) and p]
    if not expr_nodes:
        return prog

    node, _ = rng.choice(expr_nodes)
    prog.ast = copy.deepcopy(node)
    return prog


def mutate_statement(program: EvolvableProgram, rng: random.Random,
                     max_depth: int = 3) -> EvolvableProgram:
    """Statement-level mutation for imperative programs."""
    if program.program_type != ProgramType.IMPERATIVE:
        return mutate_point(program, rng)

    prog = program.copy()
    stmts = prog.ast
    if not isinstance(stmts, list):
        return prog

    choice = rng.random()

    if choice < 0.3 and len(stmts) > 1:
        # Delete a random statement
        idx = rng.randint(0, len(stmts) - 1)
        stmts.pop(idx)
    elif choice < 0.6:
        # Insert a new statement
        all_vars = list(prog.input_vars) + [prog.result_var]
        new_stmt = random_statement(prog.input_vars, all_vars, rng, max_depth=max_depth)
        idx = rng.randint(0, len(stmts))
        stmts.insert(idx, new_stmt)
    else:
        # Replace a random statement
        idx = rng.randint(0, len(stmts) - 1)
        all_vars = list(prog.input_vars) + [prog.result_var]
        stmts[idx] = random_statement(prog.input_vars, all_vars, rng, max_depth=max_depth)

    return prog


def mutate(program: EvolvableProgram, rng: random.Random,
           max_depth: int = 3, const_range: tuple = (-10, 10)) -> EvolvableProgram:
    """Apply a random mutation."""
    r = rng.random()
    if program.program_type == ProgramType.IMPERATIVE:
        if r < 0.3:
            return mutate_point(program, rng, const_range)
        elif r < 0.6:
            return mutate_subtree(program, rng, max_depth, const_range)
        else:
            return mutate_statement(program, rng, max_depth)
    else:
        if r < 0.4:
            return mutate_point(program, rng, const_range)
        elif r < 0.8:
            return mutate_subtree(program, rng, max_depth, const_range)
        else:
            return mutate_hoist(program, rng)


def crossover_expr(parent1: EvolvableProgram, parent2: EvolvableProgram,
                   rng: random.Random) -> tuple:
    """Expression crossover: swap random expression subtrees."""
    p1 = parent1.copy()
    p2 = parent2.copy()

    nodes1 = [(n, p) for n, p in collect_expr_nodes(p1.ast) if is_expression(n)]
    nodes2 = [(n, p) for n, p in collect_expr_nodes(p2.ast) if is_expression(n)]

    if not nodes1 or not nodes2:
        return p1, p2

    n1, path1 = rng.choice(nodes1)
    n2, path2 = rng.choice(nodes2)

    subtree1 = copy.deepcopy(n1)
    subtree2 = copy.deepcopy(n2)

    if path1:
        set_at_path(p1.ast, path1, subtree2)
    else:
        p1.ast = subtree2

    if path2:
        set_at_path(p2.ast, path2, subtree1)
    else:
        p2.ast = subtree1

    return p1, p2


def crossover_stmt(parent1: EvolvableProgram, parent2: EvolvableProgram,
                   rng: random.Random) -> tuple:
    """Statement crossover: exchange statements between imperative programs."""
    p1 = parent1.copy()
    p2 = parent2.copy()

    if not isinstance(p1.ast, list) or not isinstance(p2.ast, list):
        return crossover_expr(p1, p2, rng)

    if not p1.ast or not p2.ast:
        return p1, p2

    # Single-point crossover on statement lists
    point1 = rng.randint(0, len(p1.ast) - 1)
    point2 = rng.randint(0, len(p2.ast) - 1)

    # Swap tails
    tail1 = copy.deepcopy(p1.ast[point1:])
    tail2 = copy.deepcopy(p2.ast[point2:])

    child1_stmts = p1.ast[:point1] + tail2
    child2_stmts = p2.ast[:point2] + tail1

    # Limit statement count
    max_stmts = 10
    p1.ast = child1_stmts[:max_stmts] if len(child1_stmts) > max_stmts else child1_stmts
    p2.ast = child2_stmts[:max_stmts] if len(child2_stmts) > max_stmts else child2_stmts

    # Ensure at least one statement
    if not p1.ast:
        p1.ast = [Assign("result", IntLit(0))]
    if not p2.ast:
        p2.ast = [Assign("result", IntLit(0))]

    return p1, p2


def crossover(parent1: EvolvableProgram, parent2: EvolvableProgram,
              rng: random.Random) -> tuple:
    """Crossover appropriate to program type."""
    if parent1.program_type == ProgramType.IMPERATIVE:
        if rng.random() < 0.5:
            return crossover_stmt(parent1, parent2, rng)
        else:
            return crossover_expr(parent1, parent2, rng)
    else:
        return crossover_expr(parent1, parent2, rng)


# ============================================================
# Compilation and Execution
# ============================================================

def program_to_source(prog: EvolvableProgram) -> str:
    """Convert an EvolvableProgram to C010 source code."""
    if prog.program_type == ProgramType.EXPRESSION:
        # Wrap expression in: let result = <expr>; print(result);
        expr_src = expr_to_source(prog.ast)
        lines = []
        for v in prog.input_vars:
            lines.append(f"// input: {v}")
        lines.append(f"let result = {expr_src};")
        return '\n'.join(lines)
    else:
        # Imperative: declare vars, run statements
        lines = []
        declared = set()
        # Declare result var
        lines.append("let result = 0;")
        declared.add("result")
        # Declare all referenced and assigned vars
        stmts = prog.ast if isinstance(prog.ast, list) else [prog.ast]
        needed_vars = _collect_assigned_vars(stmts) | _collect_referenced_vars(stmts)
        for v in sorted(needed_vars):
            if v not in declared and v not in prog.input_vars:
                lines.append(f"let {v} = 0;")
                declared.add(v)
        # Statements
        for stmt in stmts:
            src = stmt_to_source(stmt)
            if isinstance(stmt, (IfStmt, WhileStmt, Block)):
                lines.append(src)
            else:
                lines.append(src + ";")
        return '\n'.join(lines)


def _collect_referenced_vars(stmts) -> set:
    """Collect all variable names referenced (read) in statements."""
    result = set()
    for stmt in stmts:
        _collect_vars_in_node(stmt, result)
    return result


def _collect_vars_in_node(node, result: set):
    """Recursively collect Var references."""
    if isinstance(node, Var):
        result.add(node.name)
    elif isinstance(node, UnaryOp):
        _collect_vars_in_node(node.operand, result)
    elif isinstance(node, BinOp):
        _collect_vars_in_node(node.left, result)
        _collect_vars_in_node(node.right, result)
    elif isinstance(node, Assign):
        result.add(node.name)
        _collect_vars_in_node(node.value, result)
    elif isinstance(node, LetDecl):
        result.add(node.name)
        _collect_vars_in_node(node.value, result)
    elif isinstance(node, IfStmt):
        _collect_vars_in_node(node.cond, result)
        _collect_vars_in_node(node.then_body, result)
        if node.else_body:
            _collect_vars_in_node(node.else_body, result)
    elif isinstance(node, WhileStmt):
        _collect_vars_in_node(node.cond, result)
        _collect_vars_in_node(node.body, result)
    elif isinstance(node, Block):
        for s in node.stmts:
            _collect_vars_in_node(s, result)
    elif isinstance(node, PrintStmt):
        _collect_vars_in_node(node.value, result)
    elif isinstance(node, ReturnStmt):
        if node.value:
            _collect_vars_in_node(node.value, result)


def _collect_assigned_vars(stmts) -> set:
    """Collect all variable names assigned in statements."""
    result = set()
    for stmt in stmts:
        if isinstance(stmt, Assign):
            result.add(stmt.name)
        elif isinstance(stmt, LetDecl):
            result.add(stmt.name)
        elif isinstance(stmt, IfStmt):
            if isinstance(stmt.then_body, Block):
                result.update(_collect_assigned_vars(stmt.then_body.stmts))
            if stmt.else_body and isinstance(stmt.else_body, Block):
                result.update(_collect_assigned_vars(stmt.else_body.stmts))
        elif isinstance(stmt, WhileStmt):
            if isinstance(stmt.body, Block):
                result.update(_collect_assigned_vars(stmt.body.stmts))
    return result


def expr_to_source(node) -> str:
    """Convert an expression AST node to source code."""
    if isinstance(node, IntLit):
        return str(node.value)
    elif isinstance(node, FloatLit):
        return f"{node.value}"
    elif isinstance(node, BoolLit):
        return "true" if node.value else "false"
    elif isinstance(node, StringLit):
        return f'"{node.value}"'
    elif isinstance(node, Var):
        return node.name
    elif isinstance(node, UnaryOp):
        if node.op == '-':
            return f"(-{expr_to_source(node.operand)})"
        elif node.op == 'not':
            return f"(not {expr_to_source(node.operand)})"
        return f"({node.op} {expr_to_source(node.operand)})"
    elif isinstance(node, BinOp):
        left = expr_to_source(node.left)
        right = expr_to_source(node.right)
        return f"({left} {node.op} {right})"
    return "0"


def _stmt_line(node, indent: int) -> str:
    """Convert a statement to a source line, adding semicolons where needed."""
    src = stmt_to_source(node, indent)
    if isinstance(node, (IfStmt, WhileStmt, Block)):
        return src
    return src + ";"


def stmt_to_source(node, indent: int = 0) -> str:
    """Convert a statement AST node to source code."""
    pad = "  " * indent
    if isinstance(node, Assign):
        return f"{pad}{node.name} = {expr_to_source(node.value)}"
    elif isinstance(node, LetDecl):
        return f"{pad}let {node.name} = {expr_to_source(node.value)}"
    elif isinstance(node, IfStmt):
        lines = [f"{pad}if ({expr_to_source(node.cond)}) {{"]
        if isinstance(node.then_body, Block):
            for s in node.then_body.stmts:
                lines.append(_stmt_line(s, indent + 1))
        else:
            lines.append(_stmt_line(node.then_body, indent + 1))
        if node.else_body:
            lines.append(f"{pad}}} else {{")
            if isinstance(node.else_body, Block):
                for s in node.else_body.stmts:
                    lines.append(_stmt_line(s, indent + 1))
            else:
                lines.append(_stmt_line(node.else_body, indent + 1))
        lines.append(f"{pad}}}")
        return '\n'.join(lines)
    elif isinstance(node, WhileStmt):
        lines = [f"{pad}while ({expr_to_source(node.cond)}) {{"]
        if isinstance(node.body, Block):
            for s in node.body.stmts:
                lines.append(_stmt_line(s, indent + 1))
        else:
            lines.append(_stmt_line(node.body, indent + 1))
        lines.append(f"{pad}}}")
        return '\n'.join(lines)
    elif isinstance(node, PrintStmt):
        return f"{pad}print({expr_to_source(node.value)})"
    elif isinstance(node, Block):
        lines = [f"{pad}{{"]
        for s in node.stmts:
            lines.append(stmt_to_source(s, indent + 1) + ";")
        lines.append(f"{pad}}}")
        return '\n'.join(lines)
    # Fallback for expressions used as statements
    if is_expression(node):
        return f"{pad}{expr_to_source(node)}"
    return f"{pad}0"


def compile_program(prog: EvolvableProgram, input_values: dict) -> Optional[tuple]:
    """Compile program to bytecode. Returns (chunk, compiler) or None on error."""
    source = program_to_source(prog)

    # Prepend input variable declarations
    input_lines = []
    for var_name in prog.input_vars:
        val = input_values.get(var_name, 0)
        if isinstance(val, float):
            input_lines.append(f"let {var_name} = {val};")
        elif isinstance(val, bool):
            input_lines.append(f"let {var_name} = {'true' if val else 'false'};")
        else:
            input_lines.append(f"let {var_name} = {val};")

    full_source = '\n'.join(input_lines) + '\n' + source

    try:
        tokens = lex(full_source)
        parser = Parser(tokens)
        ast = parser.parse()
        compiler = Compiler()
        chunk = compiler.compile(ast)
        return chunk, compiler
    except (ParseError, CompileError, Exception):
        return None


def execute_program(prog: EvolvableProgram, input_values: dict,
                    max_steps: int = 10000) -> Optional[dict]:
    """Compile and execute a program on the VM.

    Returns dict with 'result', 'output', 'env', 'steps', or None on error.
    """
    compiled = compile_program(prog, input_values)
    if compiled is None:
        return None

    chunk, compiler = compiled
    vm = VM(chunk)
    vm.max_steps = max_steps

    try:
        result = vm.run()
        return {
            'result': result,
            'output': vm.output,
            'env': vm.env,
            'steps': vm.step_count,
        }
    except (VMError, Exception):
        return None


# ============================================================
# Fitness Evaluation
# ============================================================

@dataclass
class TestCase:
    """A test case for program evolution."""
    inputs: dict          # variable name -> value
    expected: Any         # expected result value
    tolerance: float = 0.001  # for floating-point comparison


@dataclass
class FitnessResult:
    """Multi-objective fitness result."""
    correctness: float    # 0.0 = perfect, higher = worse
    efficiency: float     # step count (lower = better)
    simplicity: float     # AST size (lower = better)
    compiled: bool        # did it compile?
    executed: bool        # did it execute without error?
    raw_score: float      # combined fitness (lower = better)


def evaluate_fitness(prog: EvolvableProgram, test_cases: list,
                     max_steps: int = 10000,
                     efficiency_weight: float = 0.001,
                     simplicity_weight: float = 0.01) -> FitnessResult:
    """Evaluate a program's fitness against test cases.

    Returns FitnessResult with combined score (lower = better).
    """
    if not test_cases:
        return FitnessResult(float('inf'), 0, 0, False, False, float('inf'))

    total_error = 0.0
    total_steps = 0
    compiled_count = 0
    executed_count = 0

    for tc in test_cases:
        result = execute_program(prog, tc.inputs, max_steps)
        if result is None:
            total_error += 1e6
            continue

        compiled_count += 1

        # Get the result value
        if prog.program_type == ProgramType.EXPRESSION:
            value = result['env'].get('result', None)
        else:
            value = result['env'].get(prog.result_var, None)

        if value is None:
            total_error += 1e6
            continue

        executed_count += 1
        total_steps += result['steps']

        # Calculate error
        try:
            if isinstance(tc.expected, bool):
                error = 0.0 if (bool(value) == tc.expected) else 1.0
            elif isinstance(tc.expected, (int, float)):
                diff = float(value) - float(tc.expected)
                error = diff * diff
                if math.isinf(error) or math.isnan(error):
                    error = 1e6
            else:
                error = 0.0 if value == tc.expected else 1.0
        except (TypeError, ValueError):
            error = 1e6

        total_error += error

    correctness = total_error / len(test_cases)
    avg_steps = total_steps / max(executed_count, 1)
    size = ast_size(prog.ast)

    raw_score = correctness + efficiency_weight * avg_steps + simplicity_weight * size

    if math.isinf(raw_score) or math.isnan(raw_score):
        raw_score = float('inf')

    return FitnessResult(
        correctness=correctness,
        efficiency=avg_steps,
        simplicity=size,
        compiled=compiled_count == len(test_cases),
        executed=executed_count == len(test_cases),
        raw_score=raw_score,
    )


# ============================================================
# Selection
# ============================================================

def tournament_select(population: list, tournament_size: int = 3,
                      rng: random.Random = None) -> EvolvableProgram:
    """Tournament selection (lower fitness = better)."""
    if rng is None:
        rng = random.Random()
    tournament = rng.sample(population, min(tournament_size, len(population)))
    winner = min(tournament, key=lambda x: x[1].raw_score)
    return winner[0]


# ============================================================
# Bloat Control
# ============================================================

def enforce_depth(prog: EvolvableProgram, max_depth: int,
                  rng: random.Random) -> EvolvableProgram:
    """Enforce maximum AST depth by trimming deep subtrees."""
    if ast_depth(prog.ast) <= max_depth:
        return prog

    prog = prog.copy()
    if prog.program_type == ProgramType.EXPRESSION:
        prog.ast = _trim_expr(prog.ast, max_depth, prog.input_vars, rng, 0)
    elif isinstance(prog.ast, list):
        prog.ast = [_trim_stmt(s, max_depth, prog.input_vars, rng, 0) for s in prog.ast]
    return prog


def _trim_expr(node, max_depth, var_names, rng, depth):
    """Trim expression to max depth."""
    if depth >= max_depth - 1:
        return random_terminal(var_names, rng)

    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _trim_expr(node.operand, max_depth, var_names, rng, depth + 1))
    elif isinstance(node, BinOp):
        return BinOp(
            node.op,
            _trim_expr(node.left, max_depth, var_names, rng, depth + 1),
            _trim_expr(node.right, max_depth, var_names, rng, depth + 1),
        )
    return node


def _trim_stmt(node, max_depth, var_names, rng, depth):
    """Trim statement to max depth."""
    if depth >= max_depth - 1:
        return Assign("result", random_terminal(var_names, rng))

    if isinstance(node, Assign):
        return Assign(node.name, _trim_expr(node.value, max_depth, var_names, rng, depth + 1))
    elif isinstance(node, IfStmt):
        cond = _trim_expr(node.cond, max_depth, var_names, rng, depth + 1)
        then_body = _trim_stmt(node.then_body, max_depth, var_names, rng, depth + 1)
        else_body = _trim_stmt(node.else_body, max_depth, var_names, rng, depth + 1) if node.else_body else None
        return IfStmt(cond, then_body, else_body)
    elif isinstance(node, WhileStmt):
        cond = _trim_expr(node.cond, max_depth, var_names, rng, depth + 1)
        body = _trim_stmt(node.body, max_depth, var_names, rng, depth + 1)
        return WhileStmt(cond, body)
    elif isinstance(node, Block):
        return Block([_trim_stmt(s, max_depth, var_names, rng, depth + 1) for s in node.stmts])
    return node


def enforce_stmt_count(prog: EvolvableProgram, max_stmts: int = 10) -> EvolvableProgram:
    """Limit statement count for imperative programs."""
    if prog.program_type != ProgramType.IMPERATIVE:
        return prog
    if not isinstance(prog.ast, list) or len(prog.ast) <= max_stmts:
        return prog
    prog = prog.copy()
    prog.ast = prog.ast[:max_stmts]
    return prog


# ============================================================
# Program Simplification
# ============================================================

def simplify_expr(node) -> Any:
    """Constant fold and simplify an expression."""
    if isinstance(node, (IntLit, FloatLit, BoolLit, StringLit, Var)):
        return node

    if isinstance(node, UnaryOp):
        operand = simplify_expr(node.operand)
        if node.op == '-' and isinstance(operand, IntLit):
            return IntLit(-operand.value)
        if node.op == '-' and isinstance(operand, FloatLit):
            return FloatLit(-operand.value)
        if node.op == 'not' and isinstance(operand, BoolLit):
            return BoolLit(not operand.value)
        # Double negation
        if node.op == '-' and isinstance(operand, UnaryOp) and operand.op == '-':
            return simplify_expr(operand.operand)
        return UnaryOp(node.op, operand)

    if isinstance(node, BinOp):
        left = simplify_expr(node.left)
        right = simplify_expr(node.right)

        # Constant folding for ints
        if isinstance(left, IntLit) and isinstance(right, IntLit):
            try:
                if node.op == '+': return IntLit(left.value + right.value)
                if node.op == '-': return IntLit(left.value - right.value)
                if node.op == '*': return IntLit(left.value * right.value)
                if node.op == '/' and right.value != 0:
                    return IntLit(left.value // right.value)
                if node.op == '%' and right.value != 0:
                    return IntLit(left.value % right.value)
                if node.op == '<': return BoolLit(left.value < right.value)
                if node.op == '>': return BoolLit(left.value > right.value)
                if node.op == '<=': return BoolLit(left.value <= right.value)
                if node.op == '>=': return BoolLit(left.value >= right.value)
                if node.op == '==': return BoolLit(left.value == right.value)
                if node.op == '!=': return BoolLit(left.value != right.value)
            except (OverflowError, ZeroDivisionError):
                pass

        # Identity simplifications
        if node.op == '+' and isinstance(right, IntLit) and right.value == 0:
            return left
        if node.op == '+' and isinstance(left, IntLit) and left.value == 0:
            return right
        if node.op == '-' and isinstance(right, IntLit) and right.value == 0:
            return left
        if node.op == '*' and isinstance(right, IntLit) and right.value == 1:
            return left
        if node.op == '*' and isinstance(left, IntLit) and left.value == 1:
            return right
        if node.op == '*' and (
            (isinstance(left, IntLit) and left.value == 0) or
            (isinstance(right, IntLit) and right.value == 0)
        ):
            return IntLit(0)

        return BinOp(node.op, left, right)

    return node


def simplify_program(prog: EvolvableProgram) -> EvolvableProgram:
    """Simplify a program through constant folding."""
    prog = prog.copy()
    if prog.program_type == ProgramType.EXPRESSION:
        prog.ast = simplify_expr(prog.ast)
    elif isinstance(prog.ast, list):
        prog.ast = [_simplify_stmt(s) for s in prog.ast]
    return prog


def _simplify_stmt(stmt):
    """Simplify expressions within a statement."""
    if isinstance(stmt, Assign):
        return Assign(stmt.name, simplify_expr(stmt.value))
    elif isinstance(stmt, LetDecl):
        return LetDecl(stmt.name, simplify_expr(stmt.value))
    elif isinstance(stmt, IfStmt):
        cond = simplify_expr(stmt.cond)
        # Dead branch elimination
        if isinstance(cond, BoolLit):
            if cond.value:
                return stmt.then_body
            elif stmt.else_body:
                return stmt.else_body
            else:
                return Assign("result", IntLit(0))  # no-op
        then_body = _simplify_block(stmt.then_body)
        else_body = _simplify_block(stmt.else_body) if stmt.else_body else None
        return IfStmt(cond, then_body, else_body)
    elif isinstance(stmt, WhileStmt):
        cond = simplify_expr(stmt.cond)
        if isinstance(cond, BoolLit) and not cond.value:
            return Assign("result", IntLit(0))  # dead loop
        body = _simplify_block(stmt.body)
        return WhileStmt(cond, body)
    elif isinstance(stmt, Block):
        return _simplify_block(stmt)
    return stmt


def _simplify_block(node):
    """Simplify a block's contents."""
    if isinstance(node, Block):
        return Block([_simplify_stmt(s) for s in node.stmts])
    return _simplify_stmt(node)


# ============================================================
# Island Model
# ============================================================

@dataclass
class Island:
    """A sub-population with its own evolution dynamics."""
    population: list  # list of (EvolvableProgram, FitnessResult)
    config: 'EvolutionConfig'
    best_fitness: float = float('inf')
    generations: int = 0


# ============================================================
# Evolution Engine
# ============================================================

@dataclass
class EvolutionConfig:
    """Configuration for meta-evolution."""
    population_size: int = 100
    max_generations: int = 50
    tournament_size: int = 5
    crossover_rate: float = 0.6
    mutation_rate: float = 0.3
    reproduction_rate: float = 0.1
    max_tree_depth: int = 8
    initial_max_depth: int = 4
    const_range: tuple = (-10, 10)
    efficiency_weight: float = 0.001
    simplicity_weight: float = 0.01
    elitism: int = 3
    fitness_threshold: float = 0.001
    stagnation_limit: int = 10
    max_steps: int = 10000
    max_stmts: int = 10
    # Island model
    num_islands: int = 1
    migration_interval: int = 5
    migration_count: int = 2


@dataclass
class EvolutionResult:
    """Result of a meta-evolution run."""
    best_program: EvolvableProgram
    best_fitness: FitnessResult
    best_source: str
    generations_run: int
    fitness_history: list      # best raw_score per generation
    diversity_history: list    # unique fitness values per gen
    converged: bool


class MetaEvolver:
    """The meta-evolution engine.

    Evolves programs on the C010 stack VM using genetic programming.
    """

    def __init__(self, input_vars: list, test_cases: list,
                 program_type: ProgramType = ProgramType.EXPRESSION,
                 config: EvolutionConfig = None, seed: int = None):
        self.input_vars = input_vars
        self.test_cases = test_cases
        self.program_type = program_type
        self.config = config or EvolutionConfig()
        self.rng = random.Random(seed)
        self.islands: list[Island] = []
        self.generation = 0
        self.fitness_history: list = []
        self.diversity_history: list = []

    def initialize(self):
        """Create initial population(s)."""
        cfg = self.config
        island_size = cfg.population_size // max(cfg.num_islands, 1)

        self.islands = []
        for _ in range(max(cfg.num_islands, 1)):
            pop = []
            for i in range(island_size):
                depth = 2 + (i % max(cfg.initial_max_depth - 1, 1))
                prog = random_program(
                    self.input_vars, self.rng, self.program_type,
                    max_depth=depth, const_range=cfg.const_range,
                )
                fit = evaluate_fitness(
                    prog, self.test_cases, cfg.max_steps,
                    cfg.efficiency_weight, cfg.simplicity_weight,
                )
                pop.append((prog, fit))
            pop.sort(key=lambda x: x[1].raw_score)
            island = Island(population=pop, config=cfg)
            if pop:
                island.best_fitness = pop[0][1].raw_score
            self.islands.append(island)

        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []

    def _evolve_island(self, island: Island) -> float:
        """Run one generation on a single island."""
        cfg = island.config
        pop = island.population
        new_pop = []

        # Elitism
        pop.sort(key=lambda x: x[1].raw_score)
        for i in range(min(cfg.elitism, len(pop))):
            new_pop.append(pop[i])

        target_size = len(pop) if len(pop) > 0 else cfg.population_size // max(cfg.num_islands, 1)

        while len(new_pop) < target_size:
            r = self.rng.random()
            if r < cfg.crossover_rate:
                p1 = tournament_select(pop, cfg.tournament_size, self.rng)
                p2 = tournament_select(pop, cfg.tournament_size, self.rng)
                c1, c2 = crossover(p1, p2, self.rng)
                c1 = enforce_depth(c1, cfg.max_tree_depth, self.rng)
                c2 = enforce_depth(c2, cfg.max_tree_depth, self.rng)
                c1 = enforce_stmt_count(c1, cfg.max_stmts)
                c2 = enforce_stmt_count(c2, cfg.max_stmts)
                f1 = evaluate_fitness(c1, self.test_cases, cfg.max_steps,
                                      cfg.efficiency_weight, cfg.simplicity_weight)
                new_pop.append((c1, f1))
                if len(new_pop) < target_size:
                    f2 = evaluate_fitness(c2, self.test_cases, cfg.max_steps,
                                          cfg.efficiency_weight, cfg.simplicity_weight)
                    new_pop.append((c2, f2))
            elif r < cfg.crossover_rate + cfg.mutation_rate:
                parent = tournament_select(pop, cfg.tournament_size, self.rng)
                child = mutate(parent, self.rng, cfg.initial_max_depth, cfg.const_range)
                child = enforce_depth(child, cfg.max_tree_depth, self.rng)
                child = enforce_stmt_count(child, cfg.max_stmts)
                fit = evaluate_fitness(child, self.test_cases, cfg.max_steps,
                                       cfg.efficiency_weight, cfg.simplicity_weight)
                new_pop.append((child, fit))
            else:
                parent = tournament_select(pop, cfg.tournament_size, self.rng)
                fit = evaluate_fitness(parent, self.test_cases, cfg.max_steps,
                                       cfg.efficiency_weight, cfg.simplicity_weight)
                new_pop.append((parent.copy(), fit))

        island.population = new_pop
        island.population.sort(key=lambda x: x[1].raw_score)
        island.generations += 1

        best = island.population[0][1].raw_score
        island.best_fitness = best
        return best

    def _migrate(self):
        """Migrate best individuals between islands."""
        if len(self.islands) < 2:
            return
        cfg = self.config
        for i in range(len(self.islands)):
            src = self.islands[i]
            dst = self.islands[(i + 1) % len(self.islands)]
            # Send best individuals from src to dst
            src.population.sort(key=lambda x: x[1].raw_score)
            migrants = [src.population[j] for j in range(min(cfg.migration_count, len(src.population)))]
            # Replace worst in dst
            dst.population.sort(key=lambda x: x[1].raw_score)
            for j, m in enumerate(migrants):
                if j < len(dst.population):
                    dst.population[-(j + 1)] = (m[0].copy(), m[1])

    def step(self) -> float:
        """Run one generation across all islands."""
        best_overall = float('inf')

        for island in self.islands:
            best = self._evolve_island(island)
            best_overall = min(best_overall, best)

        # Migration
        if self.config.num_islands > 1 and self.generation % self.config.migration_interval == 0:
            self._migrate()

        self.generation += 1
        self.fitness_history.append(best_overall)

        # Track diversity
        all_fits = [f.raw_score for island in self.islands for _, f in island.population]
        unique = len(set(round(f, 6) for f in all_fits if not math.isinf(f)))
        total = len(all_fits)
        self.diversity_history.append(unique / max(total, 1))

        return best_overall

    def _inject_diversity(self):
        """Replace worst individuals with fresh random programs."""
        for island in self.islands:
            island.population.sort(key=lambda x: x[1].raw_score)
            half = len(island.population) // 2
            keep = island.population[:half]
            cfg = self.config
            for _ in range(len(island.population) - half):
                depth = 2 + self.rng.randint(0, max(cfg.initial_max_depth - 2, 1))
                prog = random_program(
                    self.input_vars, self.rng, self.program_type,
                    max_depth=depth, const_range=cfg.const_range,
                )
                fit = evaluate_fitness(prog, self.test_cases, cfg.max_steps,
                                       cfg.efficiency_weight, cfg.simplicity_weight)
                keep.append((prog, fit))
            island.population = keep

    def run(self) -> EvolutionResult:
        """Run evolution to completion."""
        self.initialize()
        cfg = self.config
        stagnation_counter = 0
        best_ever = float('inf')

        for gen in range(cfg.max_generations):
            best_fit = self.step()

            if best_fit < best_ever - 1e-10:
                best_ever = best_fit
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if best_fit <= cfg.fitness_threshold:
                best_prog, best_result = self.best()
                simplified = simplify_program(best_prog)
                return EvolutionResult(
                    best_program=simplified,
                    best_fitness=best_result,
                    best_source=program_to_source(simplified),
                    generations_run=gen + 1,
                    fitness_history=self.fitness_history,
                    diversity_history=self.diversity_history,
                    converged=True,
                )

            if stagnation_counter >= cfg.stagnation_limit:
                self._inject_diversity()
                stagnation_counter = 0

        best_prog, best_result = self.best()
        simplified = simplify_program(best_prog)
        return EvolutionResult(
            best_program=simplified,
            best_fitness=best_result,
            best_source=program_to_source(simplified),
            generations_run=cfg.max_generations,
            fitness_history=self.fitness_history,
            diversity_history=self.diversity_history,
            converged=False,
        )

    def best(self) -> tuple:
        """Return the best individual across all islands."""
        all_pop = [(p, f) for island in self.islands for p, f in island.population]
        if not all_pop:
            dummy = EvolvableProgram(IntLit(0), ProgramType.EXPRESSION, input_vars=self.input_vars)
            return dummy, FitnessResult(float('inf'), 0, 1, False, False, float('inf'))
        all_pop.sort(key=lambda x: x[1].raw_score)
        return all_pop[0]


# ============================================================
# Problem Generators
# ============================================================

def regression_problem(func: Callable, input_vars: list,
                       samples: list[dict], tolerance: float = 0.001) -> list[TestCase]:
    """Generate test cases for symbolic regression."""
    cases = []
    for sample in samples:
        expected = func(**{k: sample[k] for k in input_vars})
        cases.append(TestCase(inputs=sample, expected=expected, tolerance=tolerance))
    return cases


def classification_problem(func: Callable, input_vars: list,
                           samples: list[dict]) -> list[TestCase]:
    """Generate test cases for classification (boolean output)."""
    cases = []
    for sample in samples:
        expected = func(**{k: sample[k] for k in input_vars})
        cases.append(TestCase(inputs=sample, expected=expected))
    return cases


def sequence_problem(func: Callable, n_values: int = 10) -> list[TestCase]:
    """Generate test cases for integer sequence prediction."""
    cases = []
    for i in range(n_values):
        cases.append(TestCase(inputs={'x': i}, expected=func(i)))
    return cases


def make_samples_1d(x_range: tuple, n: int) -> list[dict]:
    """Generate evenly-spaced 1D samples."""
    lo, hi = x_range
    step = (hi - lo) / (n - 1) if n > 1 else 0
    return [{'x': lo + i * step} for i in range(n)]


def make_samples_2d(x_range: tuple, y_range: tuple, n_per_dim: int) -> list[dict]:
    """Generate grid of 2D samples."""
    samples = []
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    x_step = (x_hi - x_lo) / (n_per_dim - 1) if n_per_dim > 1 else 0
    y_step = (y_hi - y_lo) / (n_per_dim - 1) if n_per_dim > 1 else 0
    for i in range(n_per_dim):
        for j in range(n_per_dim):
            samples.append({'x': x_lo + i * x_step, 'y': y_lo + j * y_step})
    return samples
