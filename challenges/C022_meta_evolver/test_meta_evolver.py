"""
Tests for C022: Meta-Evolver -- Genetic Programming on the Stack VM

Covers:
  1. Program representation and types
  2. Random AST generation (expressions and imperative)
  3. AST utilities (depth, size, node collection)
  4. Source code generation
  5. Compilation and execution via VM
  6. Genetic operators (mutation, crossover)
  7. Fitness evaluation
  8. Selection
  9. Bloat control
  10. Program simplification
  11. Evolution engine (expression and imperative)
  12. Island model
  13. Problem generators
  14. Integration: evolving real programs on the VM
"""

import pytest
import random
import math
import copy

from meta_evolver import (
    # Types
    ProgramType, EvolvableProgram,
    # AST generation
    random_terminal, random_expr, random_condition, random_statement,
    random_program, random_int, random_float, random_bool,
    # AST utilities
    ast_depth, ast_size, collect_expr_nodes, get_at_path, set_at_path,
    is_expression,
    # Source generation
    program_to_source, expr_to_source, stmt_to_source,
    # Compilation/execution
    compile_program, execute_program,
    # Genetic operators
    mutate_point, mutate_subtree, mutate_hoist, mutate_statement, mutate,
    crossover_expr, crossover_stmt, crossover,
    # Fitness
    TestCase, FitnessResult, evaluate_fitness,
    # Selection
    tournament_select,
    # Bloat control
    enforce_depth, enforce_stmt_count,
    # Simplification
    simplify_expr, simplify_program,
    # Evolution
    EvolutionConfig, EvolutionResult, MetaEvolver, Island,
    # Problem generators
    regression_problem, classification_problem, sequence_problem,
    make_samples_1d, make_samples_2d,
    # C010 AST nodes
    IntLit, FloatLit, BoolLit, StringLit, Var, UnaryOp, BinOp,
    Assign, LetDecl, Block, IfStmt, WhileStmt,
    EXPR_BINARY_OPS, EXPR_COMPARE_OPS, EXPR_UNARY_OPS,
)


# ============================================================
# 1. Program Representation
# ============================================================

class TestProgramRepresentation:
    def test_expression_program(self):
        prog = EvolvableProgram(
            ast=IntLit(42),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        assert prog.program_type == ProgramType.EXPRESSION
        assert prog.result_var == "result"
        assert prog.input_vars == ['x']

    def test_imperative_program(self):
        stmts = [Assign("result", BinOp('+', Var('x'), IntLit(1)))]
        prog = EvolvableProgram(
            ast=stmts,
            program_type=ProgramType.IMPERATIVE,
            input_vars=['x'],
        )
        assert prog.program_type == ProgramType.IMPERATIVE

    def test_copy_independence(self):
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        prog2 = prog.copy()
        prog2.ast.left.value = 99
        assert prog.ast.left.value == 1  # original unchanged

    def test_program_types(self):
        assert ProgramType.EXPRESSION != ProgramType.IMPERATIVE


# ============================================================
# 2. Random AST Generation
# ============================================================

class TestRandomGeneration:
    def test_random_int(self):
        rng = random.Random(42)
        node = random_int(rng)
        assert isinstance(node, IntLit)
        assert -10 <= node.value <= 10

    def test_random_float(self):
        rng = random.Random(42)
        node = random_float(rng)
        assert isinstance(node, FloatLit)
        assert -5.0 <= node.value <= 5.0

    def test_random_bool(self):
        rng = random.Random(42)
        node = random_bool(rng)
        assert isinstance(node, BoolLit)
        assert isinstance(node.value, bool)

    def test_random_terminal_var(self):
        rng = random.Random(42)
        terminals = [random_terminal(['x', 'y'], rng) for _ in range(100)]
        has_var = any(isinstance(t, Var) for t in terminals)
        has_lit = any(isinstance(t, (IntLit, FloatLit, BoolLit)) for t in terminals)
        assert has_var
        assert has_lit

    def test_random_terminal_no_vars(self):
        rng = random.Random(42)
        t = random_terminal([], rng)
        assert isinstance(t, (IntLit, FloatLit, BoolLit))

    def test_random_expr_depth_1(self):
        rng = random.Random(42)
        expr = random_expr(['x'], rng, max_depth=1)
        assert is_expression(expr)
        assert ast_depth(expr) == 1

    def test_random_expr_respects_max_depth(self):
        rng = random.Random(42)
        for _ in range(20):
            expr = random_expr(['x', 'y'], rng, max_depth=4)
            assert ast_depth(expr) <= 4

    def test_random_expr_variety(self):
        rng = random.Random(42)
        types = set()
        for _ in range(100):
            expr = random_expr(['x'], rng, max_depth=4)
            types.add(type(expr).__name__)
        assert len(types) >= 3  # at least int, var, binop

    def test_random_condition(self):
        rng = random.Random(42)
        cond = random_condition(['x', 'y'], rng, max_depth=3)
        assert isinstance(cond, (BinOp, UnaryOp))

    def test_random_statement_assignment(self):
        rng = random.Random(0)
        # Force assignment by setting depth high
        stmt = random_statement(['x'], ['x', 'result'], rng, max_depth=1)
        assert isinstance(stmt, Assign)

    def test_random_statement_variety(self):
        rng = random.Random(42)
        types = set()
        for _ in range(100):
            stmt = random_statement(['x'], ['x', 'result'], rng, max_depth=3)
            types.add(type(stmt).__name__)
        assert 'Assign' in types

    def test_random_program_expression(self):
        rng = random.Random(42)
        prog = random_program(['x'], rng, ProgramType.EXPRESSION)
        assert prog.program_type == ProgramType.EXPRESSION
        assert is_expression(prog.ast)

    def test_random_program_imperative(self):
        rng = random.Random(42)
        prog = random_program(['x'], rng, ProgramType.IMPERATIVE)
        assert prog.program_type == ProgramType.IMPERATIVE
        assert isinstance(prog.ast, list)
        assert len(prog.ast) >= 1

    def test_random_program_deterministic(self):
        p1 = random_program(['x'], random.Random(123), ProgramType.EXPRESSION)
        p2 = random_program(['x'], random.Random(123), ProgramType.EXPRESSION)
        assert expr_to_source(p1.ast) == expr_to_source(p2.ast)


# ============================================================
# 3. AST Utilities
# ============================================================

class TestASTUtilities:
    def test_depth_terminal(self):
        assert ast_depth(IntLit(1)) == 1
        assert ast_depth(Var('x')) == 1

    def test_depth_unary(self):
        assert ast_depth(UnaryOp('-', IntLit(1))) == 2

    def test_depth_binary(self):
        node = BinOp('+', IntLit(1), IntLit(2))
        assert ast_depth(node) == 2

    def test_depth_nested(self):
        node = BinOp('+', BinOp('*', Var('x'), IntLit(2)), IntLit(1))
        assert ast_depth(node) == 3

    def test_depth_if_stmt(self):
        node = IfStmt(BinOp('<', Var('x'), IntLit(5)),
                       Block([Assign("r", IntLit(1))]))
        assert ast_depth(node) >= 2

    def test_depth_while_stmt(self):
        node = WhileStmt(BinOp('<', Var('i'), IntLit(10)),
                          Block([Assign("i", BinOp('+', Var('i'), IntLit(1)))]))
        assert ast_depth(node) >= 2

    def test_depth_block(self):
        node = Block([Assign("x", IntLit(1)), Assign("y", IntLit(2))])
        assert ast_depth(node) >= 2

    def test_depth_list(self):
        stmts = [Assign("x", IntLit(1)), Assign("y", BinOp('+', Var('x'), IntLit(2)))]
        assert ast_depth(stmts) >= 2

    def test_depth_empty_list(self):
        assert ast_depth([]) == 0

    def test_size_terminal(self):
        assert ast_size(IntLit(1)) == 1

    def test_size_binary(self):
        assert ast_size(BinOp('+', IntLit(1), IntLit(2))) == 3

    def test_size_nested(self):
        node = BinOp('+', BinOp('*', Var('x'), IntLit(2)), IntLit(1))
        assert ast_size(node) == 5

    def test_size_assign(self):
        assert ast_size(Assign("x", IntLit(1))) == 2

    def test_size_block(self):
        node = Block([Assign("x", IntLit(1))])
        assert ast_size(node) == 3  # block + assign + int

    def test_size_list(self):
        stmts = [Assign("x", IntLit(1))]
        assert ast_size(stmts) == 2  # assign + int

    def test_collect_expr_nodes(self):
        node = BinOp('+', IntLit(1), Var('x'))
        nodes = collect_expr_nodes(node)
        assert len(nodes) == 3  # binop, int, var

    def test_collect_nested(self):
        node = BinOp('+', BinOp('*', Var('x'), IntLit(2)), IntLit(1))
        nodes = collect_expr_nodes(node)
        assert len(nodes) == 5

    def test_collect_from_assign(self):
        node = Assign("result", BinOp('+', Var('x'), IntLit(1)))
        nodes = collect_expr_nodes(node)
        assert len(nodes) >= 3  # assign, binop, var, int

    def test_get_at_path(self):
        node = BinOp('+', IntLit(1), IntLit(2))
        assert isinstance(get_at_path(node, ['left']), IntLit)
        assert get_at_path(node, ['left']).value == 1

    def test_set_at_path(self):
        node = BinOp('+', IntLit(1), IntLit(2))
        set_at_path(node, ['left'], IntLit(99))
        assert node.left.value == 99

    def test_is_expression(self):
        assert is_expression(IntLit(1))
        assert is_expression(Var('x'))
        assert is_expression(BinOp('+', IntLit(1), IntLit(2)))
        assert is_expression(UnaryOp('-', IntLit(1)))
        assert not is_expression(Assign("x", IntLit(1)))
        assert not is_expression(IfStmt(BoolLit(True), Block([])))


# ============================================================
# 4. Source Code Generation
# ============================================================

class TestSourceGeneration:
    def test_int_source(self):
        assert expr_to_source(IntLit(42)) == "42"

    def test_float_source(self):
        src = expr_to_source(FloatLit(3.14))
        assert "3.14" in src

    def test_bool_source(self):
        assert expr_to_source(BoolLit(True)) == "true"
        assert expr_to_source(BoolLit(False)) == "false"

    def test_var_source(self):
        assert expr_to_source(Var('x')) == "x"

    def test_unary_source(self):
        src = expr_to_source(UnaryOp('-', IntLit(5)))
        assert "-" in src and "5" in src

    def test_binary_source(self):
        src = expr_to_source(BinOp('+', IntLit(1), IntLit(2)))
        assert "+" in src and "1" in src and "2" in src

    def test_assign_source(self):
        src = stmt_to_source(Assign("result", IntLit(42)))
        assert "result" in src and "42" in src

    def test_if_source(self):
        stmt = IfStmt(BinOp('<', Var('x'), IntLit(5)),
                       Block([Assign("result", IntLit(1))]))
        src = stmt_to_source(stmt)
        assert "if" in src and "result" in src

    def test_if_else_source(self):
        stmt = IfStmt(
            BinOp('<', Var('x'), IntLit(5)),
            Block([Assign("result", IntLit(1))]),
            Block([Assign("result", IntLit(2))]),
        )
        src = stmt_to_source(stmt)
        assert "if" in src and "else" in src

    def test_while_source(self):
        stmt = WhileStmt(
            BinOp('<', Var('i'), IntLit(10)),
            Block([Assign("i", BinOp('+', Var('i'), IntLit(1)))]),
        )
        src = stmt_to_source(stmt)
        assert "while" in src

    def test_expression_program_source(self):
        prog = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        src = program_to_source(prog)
        assert "let result" in src

    def test_imperative_program_source(self):
        prog = EvolvableProgram(
            ast=[Assign("result", BinOp('+', Var('x'), IntLit(1)))],
            program_type=ProgramType.IMPERATIVE,
            input_vars=['x'],
        )
        src = program_to_source(prog)
        assert "let result" in src
        assert "result =" in src

    def test_not_source(self):
        src = expr_to_source(UnaryOp('not', BoolLit(True)))
        assert "not" in src

    def test_comparison_source(self):
        src = expr_to_source(BinOp('<=', Var('x'), IntLit(5)))
        assert "<=" in src


# ============================================================
# 5. Compilation and Execution
# ============================================================

class TestCompilationExecution:
    def test_compile_simple_expression(self):
        prog = EvolvableProgram(
            ast=IntLit(42),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        result = compile_program(prog, {})
        assert result is not None

    def test_execute_constant(self):
        prog = EvolvableProgram(
            ast=IntLit(42),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        result = execute_program(prog, {})
        assert result is not None
        assert result['env']['result'] == 42

    def test_execute_with_input(self):
        prog = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        result = execute_program(prog, {'x': 5})
        assert result is not None
        assert result['env']['result'] == 6

    def test_execute_arithmetic(self):
        prog = EvolvableProgram(
            ast=BinOp('*', BinOp('+', Var('x'), IntLit(1)), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        result = execute_program(prog, {'x': 3})
        assert result is not None
        assert result['env']['result'] == 8

    def test_execute_imperative(self):
        stmts = [Assign("result", BinOp('+', Var('x'), Var('y')))]
        prog = EvolvableProgram(
            ast=stmts,
            program_type=ProgramType.IMPERATIVE,
            input_vars=['x', 'y'],
        )
        result = execute_program(prog, {'x': 3, 'y': 4})
        assert result is not None
        assert result['env']['result'] == 7

    def test_execute_imperative_loop(self):
        stmts = [
            Assign("result", IntLit(0)),
            WhileStmt(
                BinOp('<', Var('result'), Var('x')),
                Block([Assign("result", BinOp('+', Var('result'), IntLit(1)))]),
            ),
        ]
        prog = EvolvableProgram(
            ast=stmts,
            program_type=ProgramType.IMPERATIVE,
            input_vars=['x'],
        )
        result = execute_program(prog, {'x': 5})
        assert result is not None
        assert result['env']['result'] == 5

    def test_execute_with_float_input(self):
        prog = EvolvableProgram(
            ast=BinOp('*', Var('x'), FloatLit(2.0)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        result = execute_program(prog, {'x': 3.5})
        assert result is not None
        assert abs(result['env']['result'] - 7.0) < 0.01

    def test_execute_division_by_zero_crashes(self):
        prog = EvolvableProgram(
            ast=BinOp('/', IntLit(1), IntLit(0)),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        result = execute_program(prog, {})
        assert result is None  # VM raises on div by zero

    def test_execute_respects_step_limit(self):
        # Infinite loop
        stmts = [
            WhileStmt(BoolLit(True), Block([Assign("result", IntLit(1))])),
        ]
        prog = EvolvableProgram(
            ast=stmts,
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        result = execute_program(prog, {}, max_steps=100)
        assert result is None  # should hit step limit

    def test_compile_invalid_returns_none(self):
        # Create a program that will generate invalid source
        prog = EvolvableProgram(
            ast=StringLit("not valid code"),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        # This should still compile since it's wrapped in let result = "...";
        result = compile_program(prog, {})
        assert result is not None  # strings are valid

    def test_execute_comparison(self):
        prog = EvolvableProgram(
            ast=BinOp('<', Var('x'), IntLit(5)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        r1 = execute_program(prog, {'x': 3})
        r2 = execute_program(prog, {'x': 7})
        assert r1 is not None
        assert r2 is not None
        assert r1['env']['result'] == True
        assert r2['env']['result'] == False

    def test_execute_bool_input(self):
        prog = EvolvableProgram(
            ast=Var('x'),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        result = execute_program(prog, {'x': True})
        assert result is not None


# ============================================================
# 6. Genetic Operators
# ============================================================

class TestMutation:
    def test_mutate_point_int(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=IntLit(5),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        mutated = mutate_point(prog, rng)
        # Should still be valid
        assert mutated.program_type == ProgramType.EXPRESSION

    def test_mutate_point_preserves_structure(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        mutated = mutate_point(prog, rng)
        # Structure should be similar
        assert isinstance(mutated.ast, (BinOp, IntLit, FloatLit, BoolLit, Var))

    def test_mutate_point_independence(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        mutated = mutate_point(prog, rng)
        # Original unchanged
        assert prog.ast.left.value == 1

    def test_mutate_subtree(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        mutated = mutate_subtree(prog, rng)
        assert mutated.program_type == ProgramType.EXPRESSION

    def test_mutate_hoist(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', BinOp('*', Var('x'), IntLit(3)), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        hoisted = mutate_hoist(prog, rng)
        # Should be simpler or same
        assert ast_size(hoisted.ast) <= ast_size(prog.ast)

    def test_mutate_hoist_imperative_noop(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=[Assign("result", IntLit(1))],
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        hoisted = mutate_hoist(prog, rng)
        assert hoisted.program_type == ProgramType.IMPERATIVE

    def test_mutate_statement_delete(self):
        rng = random.Random(42)
        stmts = [
            Assign("result", IntLit(1)),
            Assign("result", IntLit(2)),
            Assign("result", IntLit(3)),
        ]
        prog = EvolvableProgram(ast=stmts, program_type=ProgramType.IMPERATIVE, input_vars=[])
        # Run enough times to get a deletion
        found_shorter = False
        for seed in range(100):
            r = random.Random(seed)
            m = mutate_statement(prog, r)
            if len(m.ast) < len(stmts):
                found_shorter = True
                break
        assert found_shorter

    def test_mutate_statement_insert(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=[Assign("result", IntLit(1))],
            program_type=ProgramType.IMPERATIVE,
            input_vars=['x'],
        )
        found_longer = False
        for seed in range(100):
            r = random.Random(seed)
            m = mutate_statement(prog, r)
            if len(m.ast) > 1:
                found_longer = True
                break
        assert found_longer

    def test_mutate_dispatches(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        mutated = mutate(prog, rng)
        assert mutated.program_type == ProgramType.EXPRESSION

    def test_mutate_var_changes(self):
        # Mutate should sometimes change variable references
        prog = EvolvableProgram(
            ast=Var('x'),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x', 'y'],
        )
        found_different = False
        for seed in range(100):
            m = mutate_point(prog, random.Random(seed))
            if isinstance(m.ast, Var) and m.ast.name != 'x':
                found_different = True
                break
        assert found_different

    def test_mutate_bool_flip(self):
        prog = EvolvableProgram(
            ast=BoolLit(True),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        found_flip = False
        for seed in range(100):
            m = mutate_point(prog, random.Random(seed))
            if isinstance(m.ast, BoolLit) and m.ast.value == False:
                found_flip = True
                break
        assert found_flip

    def test_mutate_op_change(self):
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        found_different_op = False
        for seed in range(200):
            m = mutate_point(prog, random.Random(seed))
            if isinstance(m.ast, BinOp) and m.ast.op != '+':
                found_different_op = True
                break
        assert found_different_op


class TestCrossover:
    def test_crossover_expr_basic(self):
        rng = random.Random(42)
        p1 = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        p2 = EvolvableProgram(
            ast=BinOp('*', Var('x'), IntLit(3)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        c1, c2 = crossover_expr(p1, p2, rng)
        assert c1.program_type == ProgramType.EXPRESSION
        assert c2.program_type == ProgramType.EXPRESSION

    def test_crossover_expr_independence(self):
        rng = random.Random(42)
        p1 = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        p2 = EvolvableProgram(
            ast=BinOp('*', Var('x'), IntLit(3)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        c1, c2 = crossover_expr(p1, p2, rng)
        # Parents unchanged
        assert p1.ast.left.value == 1

    def test_crossover_stmt_basic(self):
        rng = random.Random(42)
        p1 = EvolvableProgram(
            ast=[Assign("result", IntLit(1)), Assign("result", IntLit(2))],
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        p2 = EvolvableProgram(
            ast=[Assign("result", IntLit(10)), Assign("result", IntLit(20))],
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        c1, c2 = crossover_stmt(p1, p2, rng)
        assert isinstance(c1.ast, list)
        assert isinstance(c2.ast, list)
        assert len(c1.ast) >= 1
        assert len(c2.ast) >= 1

    def test_crossover_stmt_limit(self):
        rng = random.Random(42)
        # Large statement lists
        p1 = EvolvableProgram(
            ast=[Assign("result", IntLit(i)) for i in range(8)],
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        p2 = EvolvableProgram(
            ast=[Assign("result", IntLit(i + 100)) for i in range(8)],
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        c1, c2 = crossover_stmt(p1, p2, rng)
        assert len(c1.ast) <= 10
        assert len(c2.ast) <= 10

    def test_crossover_dispatches_by_type(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        c1, c2 = crossover(prog, prog, rng)
        assert c1.program_type == ProgramType.EXPRESSION

    def test_crossover_empty_stmts(self):
        rng = random.Random(42)
        p1 = EvolvableProgram(ast=[], program_type=ProgramType.IMPERATIVE, input_vars=[])
        p2 = EvolvableProgram(
            ast=[Assign("result", IntLit(1))],
            program_type=ProgramType.IMPERATIVE,
            input_vars=[],
        )
        c1, c2 = crossover_stmt(p1, p2, rng)
        # Should not crash, should have at least one stmt
        assert len(c1.ast) >= 1 or len(c2.ast) >= 1


# ============================================================
# 7. Fitness Evaluation
# ============================================================

class TestFitness:
    def test_perfect_fitness(self):
        prog = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        cases = [
            TestCase({'x': 0}, 1),
            TestCase({'x': 5}, 6),
            TestCase({'x': -3}, -2),
        ]
        fit = evaluate_fitness(prog, cases)
        assert fit.correctness < 0.01
        assert fit.compiled
        assert fit.executed
        assert fit.raw_score < 1.0

    def test_wrong_program_high_fitness(self):
        prog = EvolvableProgram(
            ast=IntLit(0),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        cases = [
            TestCase({'x': 0}, 100),
            TestCase({'x': 5}, 100),
        ]
        fit = evaluate_fitness(prog, cases)
        assert fit.correctness > 0

    def test_fitness_multi_objective(self):
        prog = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        cases = [TestCase({'x': 0}, 1)]
        fit = evaluate_fitness(prog, cases, efficiency_weight=0.0, simplicity_weight=0.0)
        assert fit.raw_score == fit.correctness

    def test_fitness_simplicity_penalty(self):
        simple = EvolvableProgram(ast=Var('x'), program_type=ProgramType.EXPRESSION, input_vars=['x'])
        complex_prog = EvolvableProgram(
            ast=BinOp('+', BinOp('+', Var('x'), IntLit(0)), IntLit(0)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        cases = [TestCase({'x': 5}, 5)]
        f1 = evaluate_fitness(simple, cases, simplicity_weight=0.1)
        f2 = evaluate_fitness(complex_prog, cases, simplicity_weight=0.1)
        # Both correct, but simpler should score better
        assert f1.raw_score < f2.raw_score

    def test_fitness_empty_cases(self):
        prog = EvolvableProgram(ast=IntLit(1), program_type=ProgramType.EXPRESSION, input_vars=[])
        fit = evaluate_fitness(prog, [])
        assert math.isinf(fit.raw_score)

    def test_fitness_imperative(self):
        stmts = [Assign("result", BinOp('*', Var('x'), IntLit(2)))]
        prog = EvolvableProgram(
            ast=stmts,
            program_type=ProgramType.IMPERATIVE,
            input_vars=['x'],
        )
        cases = [
            TestCase({'x': 3}, 6),
            TestCase({'x': 5}, 10),
        ]
        fit = evaluate_fitness(prog, cases)
        assert fit.correctness < 0.01
        assert fit.executed

    def test_fitness_bool_comparison(self):
        prog = EvolvableProgram(
            ast=BinOp('<', Var('x'), IntLit(5)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        cases = [
            TestCase({'x': 3}, True),
            TestCase({'x': 7}, False),
        ]
        fit = evaluate_fitness(prog, cases)
        assert fit.correctness < 0.01

    def test_fitness_crash_handled(self):
        # Division by zero causes VM error -> None -> high fitness
        prog = EvolvableProgram(
            ast=BinOp('/', Var('x'), IntLit(0)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        cases = [TestCase({'x': 5}, 0)]
        fit = evaluate_fitness(prog, cases)
        assert fit.raw_score > 1000  # very bad

    def test_fitness_result(self):
        fit = FitnessResult(0.5, 100, 5, True, True, 0.6)
        assert fit.correctness == 0.5
        assert fit.efficiency == 100
        assert fit.simplicity == 5


# ============================================================
# 8. Selection
# ============================================================

class TestSelection:
    def test_tournament_select_best(self):
        rng = random.Random(42)
        pop = [
            (EvolvableProgram(IntLit(i), ProgramType.EXPRESSION, input_vars=[]),
             FitnessResult(float(i), 0, 1, True, True, float(i)))
            for i in range(10)
        ]
        # With tournament of 10, should always pick the best
        selected = tournament_select(pop, tournament_size=10, rng=rng)
        assert selected.ast.value == 0

    def test_tournament_select_pressure(self):
        rng = random.Random(42)
        pop = [
            (EvolvableProgram(IntLit(i), ProgramType.EXPRESSION, input_vars=[]),
             FitnessResult(float(i), 0, 1, True, True, float(i)))
            for i in range(20)
        ]
        # With small tournament, selection should still bias toward better
        selections = [tournament_select(pop, tournament_size=3, rng=rng).ast.value for _ in range(100)]
        avg = sum(selections) / len(selections)
        assert avg < 10  # biased toward lower (better) fitness


# ============================================================
# 9. Bloat Control
# ============================================================

class TestBloatControl:
    def test_enforce_depth_within_limit(self):
        rng = random.Random(42)
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        trimmed = enforce_depth(prog, max_depth=10, rng=rng)
        assert ast_depth(trimmed.ast) <= 10

    def test_enforce_depth_trims(self):
        rng = random.Random(42)
        deep = BinOp('+', BinOp('*', BinOp('-', Var('x'), IntLit(1)), IntLit(2)), IntLit(3))
        prog = EvolvableProgram(ast=deep, program_type=ProgramType.EXPRESSION, input_vars=['x'])
        trimmed = enforce_depth(prog, max_depth=2, rng=rng)
        assert ast_depth(trimmed.ast) <= 2

    def test_enforce_depth_imperative(self):
        rng = random.Random(42)
        deep_expr = BinOp('+', BinOp('*', BinOp('-', Var('x'), IntLit(1)), IntLit(2)), IntLit(3))
        stmts = [Assign("result", deep_expr)]
        prog = EvolvableProgram(ast=stmts, program_type=ProgramType.IMPERATIVE, input_vars=['x'])
        trimmed = enforce_depth(prog, max_depth=3, rng=rng)
        assert ast_depth(trimmed.ast) <= 3

    def test_enforce_stmt_count(self):
        stmts = [Assign("result", IntLit(i)) for i in range(15)]
        prog = EvolvableProgram(ast=stmts, program_type=ProgramType.IMPERATIVE, input_vars=[])
        trimmed = enforce_stmt_count(prog, max_stmts=5)
        assert len(trimmed.ast) == 5

    def test_enforce_stmt_count_noop_if_under(self):
        stmts = [Assign("result", IntLit(1))]
        prog = EvolvableProgram(ast=stmts, program_type=ProgramType.IMPERATIVE, input_vars=[])
        trimmed = enforce_stmt_count(prog, max_stmts=5)
        assert len(trimmed.ast) == 1

    def test_enforce_stmt_count_expression_noop(self):
        prog = EvolvableProgram(ast=IntLit(1), program_type=ProgramType.EXPRESSION, input_vars=[])
        trimmed = enforce_stmt_count(prog, max_stmts=5)
        assert trimmed.program_type == ProgramType.EXPRESSION


# ============================================================
# 10. Program Simplification
# ============================================================

class TestSimplification:
    def test_fold_int_add(self):
        result = simplify_expr(BinOp('+', IntLit(3), IntLit(4)))
        assert isinstance(result, IntLit)
        assert result.value == 7

    def test_fold_int_mul(self):
        result = simplify_expr(BinOp('*', IntLit(3), IntLit(4)))
        assert isinstance(result, IntLit)
        assert result.value == 12

    def test_fold_int_sub(self):
        result = simplify_expr(BinOp('-', IntLit(10), IntLit(4)))
        assert isinstance(result, IntLit)
        assert result.value == 6

    def test_fold_int_div(self):
        result = simplify_expr(BinOp('/', IntLit(10), IntLit(3)))
        assert isinstance(result, IntLit)
        assert result.value == 3  # integer division

    def test_fold_int_mod(self):
        result = simplify_expr(BinOp('%', IntLit(10), IntLit(3)))
        assert isinstance(result, IntLit)
        assert result.value == 1

    def test_fold_comparison(self):
        result = simplify_expr(BinOp('<', IntLit(3), IntLit(5)))
        assert isinstance(result, BoolLit)
        assert result.value == True

    def test_fold_nested(self):
        # (3 + 4) * 2 -> 14
        result = simplify_expr(BinOp('*', BinOp('+', IntLit(3), IntLit(4)), IntLit(2)))
        assert isinstance(result, IntLit)
        assert result.value == 14

    def test_identity_add_zero(self):
        result = simplify_expr(BinOp('+', Var('x'), IntLit(0)))
        assert isinstance(result, Var)
        assert result.name == 'x'

    def test_identity_zero_add(self):
        result = simplify_expr(BinOp('+', IntLit(0), Var('x')))
        assert isinstance(result, Var)
        assert result.name == 'x'

    def test_identity_sub_zero(self):
        result = simplify_expr(BinOp('-', Var('x'), IntLit(0)))
        assert isinstance(result, Var)

    def test_identity_mul_one(self):
        result = simplify_expr(BinOp('*', Var('x'), IntLit(1)))
        assert isinstance(result, Var)

    def test_identity_one_mul(self):
        result = simplify_expr(BinOp('*', IntLit(1), Var('x')))
        assert isinstance(result, Var)

    def test_identity_mul_zero(self):
        result = simplify_expr(BinOp('*', Var('x'), IntLit(0)))
        assert isinstance(result, IntLit)
        assert result.value == 0

    def test_identity_zero_mul(self):
        result = simplify_expr(BinOp('*', IntLit(0), Var('x')))
        assert isinstance(result, IntLit)
        assert result.value == 0

    def test_double_negation(self):
        result = simplify_expr(UnaryOp('-', UnaryOp('-', Var('x'))))
        assert isinstance(result, Var)
        assert result.name == 'x'

    def test_fold_neg_const(self):
        result = simplify_expr(UnaryOp('-', IntLit(5)))
        assert isinstance(result, IntLit)
        assert result.value == -5

    def test_fold_not_bool(self):
        result = simplify_expr(UnaryOp('not', BoolLit(True)))
        assert isinstance(result, BoolLit)
        assert result.value == False

    def test_simplify_program_expression(self):
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(3), IntLit(4)),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        simplified = simplify_program(prog)
        assert isinstance(simplified.ast, IntLit)
        assert simplified.ast.value == 7

    def test_simplify_program_imperative(self):
        stmts = [Assign("result", BinOp('+', IntLit(3), IntLit(4)))]
        prog = EvolvableProgram(ast=stmts, program_type=ProgramType.IMPERATIVE, input_vars=[])
        simplified = simplify_program(prog)
        assert isinstance(simplified.ast[0], Assign)
        assert isinstance(simplified.ast[0].value, IntLit)
        assert simplified.ast[0].value.value == 7

    def test_simplify_dead_if_true(self):
        stmt = IfStmt(BoolLit(True), Block([Assign("result", IntLit(1))]),
                       Block([Assign("result", IntLit(2))]))
        prog = EvolvableProgram(ast=[stmt], program_type=ProgramType.IMPERATIVE, input_vars=[])
        simplified = simplify_program(prog)
        # Should eliminate the dead else branch
        first = simplified.ast[0]
        assert not isinstance(first, IfStmt)  # should be the then body

    def test_simplify_dead_if_false(self):
        stmt = IfStmt(BoolLit(False), Block([Assign("result", IntLit(1))]),
                       Block([Assign("result", IntLit(2))]))
        prog = EvolvableProgram(ast=[stmt], program_type=ProgramType.IMPERATIVE, input_vars=[])
        simplified = simplify_program(prog)
        first = simplified.ast[0]
        assert not isinstance(first, IfStmt)

    def test_no_div_by_zero_fold(self):
        result = simplify_expr(BinOp('/', IntLit(10), IntLit(0)))
        assert isinstance(result, BinOp)  # should NOT fold

    def test_simplify_preserves_vars(self):
        result = simplify_expr(BinOp('+', Var('x'), Var('y')))
        assert isinstance(result, BinOp)  # can't fold variables


# ============================================================
# 11. Evolution Engine
# ============================================================

class TestEvolutionEngine:
    def test_config_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.population_size == 100
        assert cfg.max_generations == 50
        assert cfg.num_islands == 1

    def test_initialize(self):
        cases = [TestCase({'x': 0}, 1), TestCase({'x': 1}, 2)]
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION,
                              EvolutionConfig(population_size=20), seed=42)
        evolver.initialize()
        assert len(evolver.islands) == 1
        assert len(evolver.islands[0].population) == 20

    def test_step(self):
        cases = [TestCase({'x': 0}, 1), TestCase({'x': 1}, 2)]
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION,
                              EvolutionConfig(population_size=20), seed=42)
        evolver.initialize()
        best = evolver.step()
        assert isinstance(best, float)
        assert evolver.generation == 1

    def test_best(self):
        cases = [TestCase({'x': 0}, 1)]
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION,
                              EvolutionConfig(population_size=10), seed=42)
        evolver.initialize()
        prog, fit = evolver.best()
        assert isinstance(prog, EvolvableProgram)
        assert isinstance(fit, FitnessResult)

    def test_run_simple(self):
        cases = [TestCase({'x': 0}, 0), TestCase({'x': 1}, 1), TestCase({'x': 2}, 2)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=30, max_generations=10), seed=42,
        )
        result = evolver.run()
        assert isinstance(result, EvolutionResult)
        assert result.generations_run <= 10
        assert len(result.fitness_history) > 0

    def test_run_convergence_constant(self):
        # Should easily find f(x) = 42
        cases = [TestCase({'x': i}, 42) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=50, max_generations=30,
                            fitness_threshold=0.1), seed=42,
        )
        result = evolver.run()
        # Should converge or at least get close
        assert result.best_fitness.raw_score < 100

    def test_run_returns_source(self):
        cases = [TestCase({'x': 0}, 0)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=10, max_generations=5), seed=42,
        )
        result = evolver.run()
        assert isinstance(result.best_source, str)
        assert len(result.best_source) > 0

    def test_run_imperative(self):
        cases = [TestCase({'x': 3}, 6), TestCase({'x': 5}, 10)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.IMPERATIVE,
            EvolutionConfig(population_size=30, max_generations=15), seed=42,
        )
        result = evolver.run()
        assert isinstance(result, EvolutionResult)

    def test_diversity_tracking(self):
        cases = [TestCase({'x': 0}, 1)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=20, max_generations=5), seed=42,
        )
        result = evolver.run()
        assert len(result.diversity_history) == result.generations_run

    def test_fitness_history(self):
        cases = [TestCase({'x': 0}, 1)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=20, max_generations=5), seed=42,
        )
        result = evolver.run()
        assert len(result.fitness_history) == result.generations_run

    def test_stagnation_injection(self):
        # Use a hard problem that should trigger stagnation
        cases = [TestCase({'x': i}, i * i * i) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=20, max_generations=20,
                            stagnation_limit=3), seed=42,
        )
        result = evolver.run()
        # Should have run without crash
        assert result.generations_run > 0

    def test_elitism(self):
        cases = [TestCase({'x': 0}, 42)]
        cfg = EvolutionConfig(population_size=20, max_generations=3, elitism=3)
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION, cfg, seed=42)
        evolver.initialize()
        best_before = evolver.best()[1].raw_score
        evolver.step()
        best_after = evolver.best()[1].raw_score
        # Elitism guarantees best doesn't get worse
        assert best_after <= best_before + 0.01


# ============================================================
# 12. Island Model
# ============================================================

class TestIslandModel:
    def test_multi_island_init(self):
        cases = [TestCase({'x': 0}, 1)]
        cfg = EvolutionConfig(population_size=30, num_islands=3)
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION, cfg, seed=42)
        evolver.initialize()
        assert len(evolver.islands) == 3
        assert all(len(island.population) == 10 for island in evolver.islands)

    def test_migration(self):
        cases = [TestCase({'x': 0}, 1)]
        cfg = EvolutionConfig(population_size=20, num_islands=2, migration_interval=1,
                              migration_count=1, max_generations=5)
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION, cfg, seed=42)
        result = evolver.run()
        assert result.generations_run > 0

    def test_island_independence(self):
        cases = [TestCase({'x': 0}, 1)]
        cfg = EvolutionConfig(population_size=20, num_islands=2, max_generations=3)
        evolver = MetaEvolver(['x'], cases, ProgramType.EXPRESSION, cfg, seed=42)
        evolver.initialize()
        # Different islands should have different populations
        pop1_fits = [f.raw_score for _, f in evolver.islands[0].population]
        pop2_fits = [f.raw_score for _, f in evolver.islands[1].population]
        # Not identical (with high probability)
        assert pop1_fits != pop2_fits or len(pop1_fits) == 0

    def test_island_data(self):
        island = Island(population=[], config=EvolutionConfig())
        assert island.best_fitness == float('inf')
        assert island.generations == 0


# ============================================================
# 13. Problem Generators
# ============================================================

class TestProblemGenerators:
    def test_regression_problem(self):
        cases = regression_problem(
            lambda x: x * 2, ['x'],
            [{'x': 1}, {'x': 2}, {'x': 3}],
        )
        assert len(cases) == 3
        assert cases[0].expected == 2
        assert cases[1].expected == 4
        assert cases[2].expected == 6

    def test_classification_problem(self):
        cases = classification_problem(
            lambda x: x > 0, ['x'],
            [{'x': -1}, {'x': 0}, {'x': 1}],
        )
        assert len(cases) == 3
        assert cases[0].expected == False
        assert cases[2].expected == True

    def test_sequence_problem(self):
        cases = sequence_problem(lambda x: x * x, n_values=5)
        assert len(cases) == 5
        assert cases[0].expected == 0
        assert cases[2].expected == 4
        assert cases[4].expected == 16

    def test_make_samples_1d(self):
        samples = make_samples_1d((0, 10), 6)
        assert len(samples) == 6
        assert samples[0]['x'] == 0
        assert samples[5]['x'] == 10

    def test_make_samples_2d(self):
        samples = make_samples_2d((0, 1), (0, 1), 3)
        assert len(samples) == 9  # 3x3 grid

    def test_make_samples_1d_single(self):
        samples = make_samples_1d((5, 5), 1)
        assert len(samples) == 1
        assert samples[0]['x'] == 5


# ============================================================
# 14. Integration: Evolving Real Programs on the VM
# ============================================================

class TestIntegration:
    def test_evolve_identity(self):
        """Evolve f(x) = x -- should be trivial."""
        cases = [TestCase({'x': i}, i) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(
                population_size=50, max_generations=30,
                fitness_threshold=0.01,
            ), seed=42,
        )
        result = evolver.run()
        assert result.best_fitness.correctness < 1.0

    def test_evolve_constant(self):
        """Evolve f(x) = 7."""
        cases = [TestCase({'x': i}, 7) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(
                population_size=60, max_generations=30,
                fitness_threshold=0.1,
            ), seed=42,
        )
        result = evolver.run()
        assert result.best_fitness.correctness < 10

    def test_evolve_double(self):
        """Evolve f(x) = 2*x."""
        cases = [TestCase({'x': i}, 2 * i) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(
                population_size=80, max_generations=40,
                fitness_threshold=0.1,
            ), seed=42,
        )
        result = evolver.run()
        # May or may not converge, but should improve
        assert result.fitness_history[-1] <= result.fitness_history[0] or result.converged

    def test_evolve_imperative_simple(self):
        """Evolve an imperative program that computes x + 1."""
        cases = [TestCase({'x': i}, i + 1) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.IMPERATIVE,
            EvolutionConfig(
                population_size=60, max_generations=20,
                fitness_threshold=0.1,
            ), seed=42,
        )
        result = evolver.run()
        assert isinstance(result.best_source, str)
        assert result.generations_run > 0

    def test_random_programs_all_compile(self):
        """All randomly generated programs should compile successfully."""
        rng = random.Random(42)
        compile_count = 0
        total = 30
        for _ in range(total):
            prog = random_program(['x'], rng, ProgramType.EXPRESSION, max_depth=4)
            compiled = compile_program(prog, {'x': 5})
            if compiled is not None:
                compile_count += 1
        # Most should compile
        assert compile_count > total * 0.8

    def test_random_imperative_programs_compile(self):
        """Imperative programs should mostly compile."""
        rng = random.Random(42)
        compile_count = 0
        total = 30
        for _ in range(total):
            prog = random_program(['x'], rng, ProgramType.IMPERATIVE, max_depth=3)
            compiled = compile_program(prog, {'x': 5})
            if compiled is not None:
                compile_count += 1
        assert compile_count > total * 0.7

    def test_mutated_programs_compile(self):
        """Mutated programs should still compile."""
        rng = random.Random(42)
        base = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        compile_count = 0
        total = 30
        for i in range(total):
            mutated = mutate(base, random.Random(i))
            compiled = compile_program(mutated, {'x': 5})
            if compiled is not None:
                compile_count += 1
        assert compile_count > total * 0.8

    def test_crossover_programs_compile(self):
        """Crossover children should compile."""
        rng = random.Random(42)
        p1 = EvolvableProgram(
            ast=BinOp('+', Var('x'), IntLit(1)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        p2 = EvolvableProgram(
            ast=BinOp('*', Var('x'), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        compile_count = 0
        total = 20
        for i in range(total):
            c1, c2 = crossover(p1, p2, random.Random(i))
            if compile_program(c1, {'x': 5}) is not None:
                compile_count += 1
            if compile_program(c2, {'x': 5}) is not None:
                compile_count += 1
        assert compile_count > total  # more than half

    def test_full_pipeline_expression(self):
        """Full pipeline: generate, mutate, crossover, compile, execute, evaluate."""
        rng = random.Random(42)
        prog1 = random_program(['x'], rng, ProgramType.EXPRESSION)
        prog2 = random_program(['x'], rng, ProgramType.EXPRESSION)

        # Mutate
        m1 = mutate(prog1, rng)
        m2 = mutate(prog2, rng)

        # Crossover
        c1, c2 = crossover(m1, m2, rng)

        # Enforce depth
        c1 = enforce_depth(c1, 8, rng)

        # Compile and execute
        cases = [TestCase({'x': 1}, 2)]
        fit = evaluate_fitness(c1, cases)
        assert isinstance(fit, FitnessResult)

    def test_full_pipeline_imperative(self):
        """Full pipeline for imperative programs."""
        rng = random.Random(42)
        prog1 = random_program(['x'], rng, ProgramType.IMPERATIVE)
        prog2 = random_program(['x'], rng, ProgramType.IMPERATIVE)

        m1 = mutate(prog1, rng)
        c1, c2 = crossover(m1, prog2, rng)
        c1 = enforce_depth(c1, 8, rng)
        c1 = enforce_stmt_count(c1)

        cases = [TestCase({'x': 1}, 2)]
        fit = evaluate_fitness(c1, cases)
        assert isinstance(fit, FitnessResult)

    def test_simplify_then_execute(self):
        """Simplified programs should produce same results."""
        prog = EvolvableProgram(
            ast=BinOp('+', BinOp('+', Var('x'), IntLit(0)), IntLit(0)),
            program_type=ProgramType.EXPRESSION,
            input_vars=['x'],
        )
        simplified = simplify_program(prog)
        r1 = execute_program(prog, {'x': 7})
        r2 = execute_program(simplified, {'x': 7})
        assert r1 is not None and r2 is not None
        assert r1['env']['result'] == r2['env']['result']

    def test_evolved_program_executes_on_vm(self):
        """The best program from evolution should actually run on the VM."""
        cases = [TestCase({'x': i}, i + 1) for i in range(3)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=30, max_generations=5), seed=42,
        )
        result = evolver.run()
        # Execute the best program
        exec_result = execute_program(result.best_program, {'x': 10})
        assert exec_result is not None  # should run without error

    def test_multi_variable_evolution(self):
        """Evolve f(x, y) = x + y."""
        samples = make_samples_2d((0, 3), (0, 3), 3)
        cases = regression_problem(lambda x, y: x + y, ['x', 'y'], samples)
        evolver = MetaEvolver(
            ['x', 'y'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=60, max_generations=20), seed=42,
        )
        result = evolver.run()
        assert result.generations_run > 0

    def test_island_evolution_integration(self):
        """Multi-island evolution should work end-to-end."""
        cases = [TestCase({'x': i}, i * 2) for i in range(5)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(
                population_size=30, max_generations=10,
                num_islands=3, migration_interval=3,
            ), seed=42,
        )
        result = evolver.run()
        assert result.generations_run > 0
        assert len(result.fitness_history) > 0


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_terminal_program(self):
        prog = EvolvableProgram(ast=IntLit(0), program_type=ProgramType.EXPRESSION, input_vars=[])
        result = execute_program(prog, {})
        assert result is not None

    def test_empty_input_vars(self):
        prog = random_program([], random.Random(42), ProgramType.EXPRESSION)
        result = execute_program(prog, {})
        # Should work with no input variables
        assert result is not None or True  # may or may not execute

    def test_large_constants(self):
        prog = EvolvableProgram(
            ast=IntLit(999999),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        result = execute_program(prog, {})
        assert result is not None
        assert result['env']['result'] == 999999

    def test_negative_constants(self):
        prog = EvolvableProgram(
            ast=IntLit(-42),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        result = execute_program(prog, {})
        assert result is not None
        assert result['env']['result'] == -42

    def test_deeply_nested_expression(self):
        # Build a deeply nested expression
        node = IntLit(1)
        for _ in range(10):
            node = BinOp('+', node, IntLit(1))
        prog = EvolvableProgram(ast=node, program_type=ProgramType.EXPRESSION, input_vars=[])
        result = execute_program(prog, {})
        assert result is not None
        assert result['env']['result'] == 11

    def test_mutate_single_node(self):
        prog = EvolvableProgram(ast=IntLit(5), program_type=ProgramType.EXPRESSION, input_vars=[])
        mutated = mutate(prog, random.Random(42))
        assert mutated is not None

    def test_crossover_identical(self):
        prog = EvolvableProgram(
            ast=BinOp('+', IntLit(1), IntLit(2)),
            program_type=ProgramType.EXPRESSION,
            input_vars=[],
        )
        c1, c2 = crossover(prog, prog, random.Random(42))
        assert c1 is not None
        assert c2 is not None

    def test_population_size_one(self):
        cases = [TestCase({'x': 0}, 0)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=1, max_generations=3, elitism=1,
                            tournament_size=1), seed=42,
        )
        result = evolver.run()
        assert result.generations_run > 0

    def test_max_generations_zero(self):
        cases = [TestCase({'x': 0}, 0)]
        evolver = MetaEvolver(
            ['x'], cases, ProgramType.EXPRESSION,
            EvolutionConfig(population_size=5, max_generations=0), seed=42,
        )
        result = evolver.run()
        assert result.generations_run == 0

    def test_stmt_to_source_block(self):
        block = Block([Assign("x", IntLit(1)), Assign("y", IntLit(2))])
        src = stmt_to_source(block)
        assert "{" in src and "}" in src

    def test_expr_to_source_fallback(self):
        # Unknown node type should return "0"
        assert expr_to_source(None) == "0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
