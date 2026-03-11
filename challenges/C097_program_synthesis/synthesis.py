"""
C097: Program Synthesis Engine
Synthesizes programs from input/output examples or logical specifications.

Architecture:
- Expression DSL: simple typed language (int arithmetic, booleans, conditionals)
- Enumerative synthesis: bottom-up enumeration with observational equivalence pruning
- Constraint-based synthesis: encode as SMT and solve (composes C037)
- CEGIS: Counterexample-Guided Inductive Synthesis loop
- Component-based synthesis: build from configurable component libraries
- Conditional synthesis: synthesize if-then-else programs
- Oracle synthesis: learn from black-box oracle functions

Key ideas:
- Programs are ASTs in a simple DSL
- Observational equivalence: two programs are equivalent if they produce
  the same outputs on all provided inputs -> prune search space
- CEGIS loop: synthesize candidate on examples, verify against spec,
  add counterexample if verification fails, repeat
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Callable
from enum import Enum
from collections import defaultdict
import itertools
import copy

# Import C037 SMT solver for constraint-based synthesis
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C037_smt_solver'))
from smt_solver import SMTSolver, SMTResult, Term, Op, INT, BOOL, Var
from smt_solver import IntConst as SMTIntConst


# ============================================================
# Expression DSL
# ============================================================

class Type(Enum):
    INT = "int"
    BOOL = "bool"


@dataclass(frozen=True)
class Expr:
    """Base class for DSL expressions."""
    pass


@dataclass(frozen=True)
class IntConst(Expr):
    value: int

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class BoolConst(Expr):
    value: bool

    def __repr__(self):
        return str(self.value).lower()


@dataclass(frozen=True)
class VarExpr(Expr):
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str  # 'neg', 'not', 'abs'
    arg: Expr

    def __repr__(self):
        if self.op == 'neg':
            return f"(-{self.arg})"
        if self.op == 'abs':
            return f"abs({self.arg})"
        return f"(!{self.arg})"


@dataclass(frozen=True)
class BinOp(Expr):
    op: str  # '+', '-', '*', '/', '%', 'max', 'min', '==', '!=', '<', '<=', '>', '>=', 'and', 'or'
    left: Expr
    right: Expr

    def __repr__(self):
        if self.op in ('max', 'min'):
            return f"{self.op}({self.left}, {self.right})"
        return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class IfExpr(Expr):
    cond: Expr
    then_expr: Expr
    else_expr: Expr

    def __repr__(self):
        return f"(if {self.cond} then {self.then_expr} else {self.else_expr})"


# ============================================================
# Evaluator
# ============================================================

class EvalError(Exception):
    pass


def evaluate(expr: Expr, env: dict):
    """Evaluate an expression in an environment. Returns int or bool."""
    if isinstance(expr, IntConst):
        return expr.value
    if isinstance(expr, BoolConst):
        return expr.value
    if isinstance(expr, VarExpr):
        if expr.name not in env:
            raise EvalError(f"Undefined variable: {expr.name}")
        return env[expr.name]
    if isinstance(expr, UnaryOp):
        a = evaluate(expr.arg, env)
        if expr.op == 'neg':
            return -a
        if expr.op == 'not':
            return not a
        if expr.op == 'abs':
            return abs(a)
        raise EvalError(f"Unknown unary op: {expr.op}")
    if isinstance(expr, BinOp):
        l = evaluate(expr.left, env)
        r = evaluate(expr.right, env)
        if expr.op == '+': return l + r
        if expr.op == '-': return l - r
        if expr.op == '*': return l * r
        if expr.op == '/':
            if r == 0:
                raise EvalError("Division by zero")
            # Integer division toward zero
            if (l < 0) != (r < 0) and l % r != 0:
                return -(abs(l) // abs(r))
            return l // r
        if expr.op == '%':
            if r == 0:
                raise EvalError("Modulo by zero")
            return l % r
        if expr.op == 'max': return max(l, r)
        if expr.op == 'min': return min(l, r)
        if expr.op == '==': return l == r
        if expr.op == '!=': return l != r
        if expr.op == '<': return l < r
        if expr.op == '<=': return l <= r
        if expr.op == '>': return l > r
        if expr.op == '>=': return l >= r
        if expr.op == 'and': return l and r
        if expr.op == 'or': return l or r
        raise EvalError(f"Unknown binary op: {expr.op}")
    if isinstance(expr, IfExpr):
        c = evaluate(expr.cond, env)
        if c:
            return evaluate(expr.then_expr, env)
        else:
            return evaluate(expr.else_expr, env)
    raise EvalError(f"Unknown expression type: {type(expr)}")


def expr_size(expr: Expr) -> int:
    """Count AST nodes."""
    if isinstance(expr, (IntConst, BoolConst, VarExpr)):
        return 1
    if isinstance(expr, UnaryOp):
        return 1 + expr_size(expr.arg)
    if isinstance(expr, BinOp):
        return 1 + expr_size(expr.left) + expr_size(expr.right)
    if isinstance(expr, IfExpr):
        return 1 + expr_size(expr.cond) + expr_size(expr.then_expr) + expr_size(expr.else_expr)
    return 1


def expr_depth(expr: Expr) -> int:
    """AST depth."""
    if isinstance(expr, (IntConst, BoolConst, VarExpr)):
        return 0
    if isinstance(expr, UnaryOp):
        return 1 + expr_depth(expr.arg)
    if isinstance(expr, BinOp):
        return 1 + max(expr_depth(expr.left), expr_depth(expr.right))
    if isinstance(expr, IfExpr):
        return 1 + max(expr_depth(expr.cond), expr_depth(expr.then_expr), expr_depth(expr.else_expr))
    return 0


def uses_variable(expr: Expr, name: str) -> bool:
    """Check if expression uses a given variable."""
    if isinstance(expr, VarExpr):
        return expr.name == name
    if isinstance(expr, (IntConst, BoolConst)):
        return False
    if isinstance(expr, UnaryOp):
        return uses_variable(expr.arg, name)
    if isinstance(expr, BinOp):
        return uses_variable(expr.left, name) or uses_variable(expr.right, name)
    if isinstance(expr, IfExpr):
        return (uses_variable(expr.cond, name) or
                uses_variable(expr.then_expr, name) or
                uses_variable(expr.else_expr, name))
    return False


def get_variables(expr: Expr) -> set:
    """Get all variables used in expression."""
    if isinstance(expr, VarExpr):
        return {expr.name}
    if isinstance(expr, (IntConst, BoolConst)):
        return set()
    if isinstance(expr, UnaryOp):
        return get_variables(expr.arg)
    if isinstance(expr, BinOp):
        return get_variables(expr.left) | get_variables(expr.right)
    if isinstance(expr, IfExpr):
        return (get_variables(expr.cond) |
                get_variables(expr.then_expr) |
                get_variables(expr.else_expr))
    return set()


# ============================================================
# Specification Types
# ============================================================

@dataclass
class IOExample:
    """An input/output example for synthesis."""
    inputs: dict   # variable name -> value
    output: object  # expected output


@dataclass
class SynthesisSpec:
    """Specification for synthesis."""
    examples: list           # list of IOExample
    input_vars: list         # list of variable names (ordered)
    output_type: Type = Type.INT
    constants: list = None   # optional constants to include
    components: list = None  # optional component operators

    def __post_init__(self):
        if self.constants is None:
            self.constants = [0, 1]
        if self.components is None:
            self.components = ['+', '-', '*']


@dataclass
class SynthesisResult:
    """Result of synthesis."""
    success: bool
    program: Optional[Expr] = None
    iterations: int = 0
    candidates_explored: int = 0
    method: str = ""


# ============================================================
# Observational Equivalence
# ============================================================

class ObservationalEquivalence:
    """
    Track observationally equivalent programs.
    Two programs are OE if they produce the same outputs on all examples.
    We keep only the smallest representative of each equivalence class.
    """

    def __init__(self, examples: list):
        self.examples = examples
        self._signatures = {}  # signature tuple -> (size, expr)

    def signature(self, expr: Expr) -> Optional[tuple]:
        """Compute output signature on all examples. None if any eval fails."""
        outputs = []
        for ex in self.examples:
            try:
                val = evaluate(expr, ex.inputs)
                outputs.append(val)
            except EvalError:
                return None
        return tuple(outputs)

    def is_new(self, expr: Expr) -> bool:
        """Check if this program has a new observational signature."""
        sig = self.signature(expr)
        if sig is None:
            return False
        size = expr_size(expr)
        if sig in self._signatures:
            existing_size, _ = self._signatures[sig]
            if size >= existing_size:
                return False
        self._signatures[sig] = (size, expr)
        return True

    def add(self, expr: Expr) -> bool:
        """Add expression, return True if it was novel."""
        return self.is_new(expr)

    def get_representative(self, sig: tuple) -> Optional[Expr]:
        """Get the smallest expression with this signature."""
        if sig in self._signatures:
            return self._signatures[sig][1]
        return None

    @property
    def num_classes(self) -> int:
        return len(self._signatures)


# ============================================================
# Enumerative Synthesis (Bottom-Up)
# ============================================================

class EnumerativeSynthesizer:
    """
    Bottom-up enumerative synthesis with observational equivalence pruning.
    Generates programs of increasing size/depth and checks against spec.
    """

    def __init__(self, spec: SynthesisSpec, max_size: int = 10, max_depth: int = 4):
        self.spec = spec
        self.max_size = max_size
        self.max_depth = max_depth
        self.oe = ObservationalEquivalence(spec.examples)
        self.candidates_explored = 0

        # Target signature
        self._target_sig = tuple(ex.output for ex in spec.examples)

    def synthesize(self) -> SynthesisResult:
        """Run bottom-up enumeration."""
        # Level 0: atoms (variables + constants)
        atoms = self._generate_atoms()
        bank = []  # all unique programs found so far

        for atom in atoms:
            self.candidates_explored += 1
            if self.oe.add(atom):
                bank.append(atom)
                if self._check(atom):
                    return SynthesisResult(True, atom, 0, self.candidates_explored, "enumerative")

        # Levels 1..max_depth: build larger programs from smaller ones
        for depth in range(1, self.max_depth + 1):
            new_programs = []

            # Unary operations
            for op in self._unary_ops():
                for prog in bank:
                    if expr_depth(prog) >= depth:
                        continue
                    candidate = UnaryOp(op, prog)
                    if expr_size(candidate) > self.max_size:
                        continue
                    self.candidates_explored += 1
                    if self.oe.add(candidate):
                        new_programs.append(candidate)
                        if self._check(candidate):
                            return SynthesisResult(True, candidate, depth, self.candidates_explored, "enumerative")

            # Binary operations
            for op in self.spec.components:
                for left in bank:
                    if expr_depth(left) >= depth:
                        continue
                    for right in bank:
                        if expr_depth(right) >= depth:
                            continue
                        if max(expr_depth(left), expr_depth(right)) + 1 > depth:
                            # At least one operand should be from the previous level
                            pass
                        candidate = BinOp(op, left, right)
                        if expr_size(candidate) > self.max_size:
                            continue
                        self.candidates_explored += 1
                        if self.oe.add(candidate):
                            new_programs.append(candidate)
                            if self._check(candidate):
                                return SynthesisResult(True, candidate, depth, self.candidates_explored, "enumerative")

            # If-then-else (with boolean conditions from comparisons)
            if depth >= 2:
                conds = [p for p in bank if self._is_bool_expr(p)]
                int_exprs = [p for p in bank if not self._is_bool_expr(p)]
                for cond in conds:
                    for then_e in int_exprs:
                        for else_e in int_exprs:
                            if then_e == else_e:
                                continue
                            candidate = IfExpr(cond, then_e, else_e)
                            if expr_size(candidate) > self.max_size:
                                continue
                            self.candidates_explored += 1
                            if self.oe.add(candidate):
                                new_programs.append(candidate)
                                if self._check(candidate):
                                    return SynthesisResult(True, candidate, depth, self.candidates_explored, "enumerative")

            bank.extend(new_programs)

        return SynthesisResult(False, None, self.max_depth, self.candidates_explored, "enumerative")

    def _generate_atoms(self) -> list:
        """Generate atomic expressions (variables and constants)."""
        atoms = []
        for name in self.spec.input_vars:
            atoms.append(VarExpr(name))
        for c in self.spec.constants:
            if isinstance(c, bool):
                atoms.append(BoolConst(c))
            else:
                atoms.append(IntConst(c))
        return atoms

    def _unary_ops(self) -> list:
        """Available unary operations."""
        ops = []
        if 'neg' in self.spec.components or '-' in self.spec.components:
            ops.append('neg')
        if 'abs' in self.spec.components:
            ops.append('abs')
        return ops

    def _is_bool_expr(self, expr: Expr) -> bool:
        """Check if expression evaluates to boolean on examples."""
        if isinstance(expr, BoolConst):
            return True
        if isinstance(expr, BinOp) and expr.op in ('==', '!=', '<', '<=', '>', '>=', 'and', 'or'):
            return True
        if isinstance(expr, UnaryOp) and expr.op == 'not':
            return True
        # Check by evaluating
        for ex in self.spec.examples:
            try:
                val = evaluate(expr, ex.inputs)
                return isinstance(val, bool)
            except EvalError:
                continue
        return False

    def _check(self, expr: Expr) -> bool:
        """Check if expression matches all examples."""
        sig = self.oe.signature(expr)
        return sig == self._target_sig


# ============================================================
# Constraint-Based Synthesis (SMT)
# ============================================================

class ConstraintSynthesizer:
    """
    Template-based synthesis with SMT constant solving.

    Approach: enumerate program templates (structures with unknown constants),
    then use SMT to solve for the constants. This avoids encoding the full
    synthesis problem as a single SMT formula.

    Templates: c, x, x+c, c*x, x+y, x-y, x*y, c*x+d, etc.
    """

    def __init__(self, spec: SynthesisSpec, max_nodes: int = 5):
        self.spec = spec
        self.max_nodes = max_nodes
        self.candidates_explored = 0

    def synthesize(self) -> SynthesisResult:
        """Try templates of increasing complexity."""
        vars = self.spec.input_vars
        examples = self.spec.examples

        # Generate and try templates
        templates = self._generate_templates()
        for template_fn, name in templates:
            self.candidates_explored += 1
            result = self._try_template(template_fn)
            if result is not None:
                return SynthesisResult(True, result, self.candidates_explored,
                                       self.candidates_explored, "constraint")

        return SynthesisResult(False, None, self.max_nodes,
                               self.candidates_explored, "constraint")

    def _generate_templates(self):
        """Generate program templates with SMT holes for constants."""
        vars = self.spec.input_vars
        templates = []

        # Size 1: just a constant
        templates.append((self._template_const, "c"))

        # Size 1: just a variable
        for v in vars:
            templates.append((lambda solver, v=v: (VarExpr(v), []), f"{v}"))

        # Size 3: x op c, c op x, x op y
        for v in vars:
            for op in self.spec.components:
                if op in ('+', '-', '*'):
                    templates.append((
                        lambda solver, v=v, op=op: self._template_var_op_const(solver, v, op),
                        f"{v} {op} c"
                    ))

        for i, v1 in enumerate(vars):
            for v2 in vars:
                for op in self.spec.components:
                    if op in ('+', '-', '*'):
                        templates.append((
                            lambda solver, v1=v1, v2=v2, op=op: (
                                BinOp(op, VarExpr(v1), VarExpr(v2)), []
                            ),
                            f"{v1} {op} {v2}"
                        ))

        # Size 5: c*x + d (linear)
        for v in vars:
            templates.append((
                lambda solver, v=v: self._template_linear(solver, v),
                f"c*{v}+d"
            ))

        # Size 5: (x op y) op c
        for v1 in vars:
            for v2 in vars:
                for op1 in self.spec.components:
                    for op2 in self.spec.components:
                        if op1 in ('+', '-', '*') and op2 in ('+', '-', '*'):
                            templates.append((
                                lambda solver, v1=v1, v2=v2, op1=op1, op2=op2:
                                    self._template_binop_op_const(solver, v1, v2, op1, op2),
                                f"({v1}{op1}{v2}){op2}c"
                            ))

        return templates

    def _template_const(self, solver):
        """Template: just a constant c."""
        c = solver.Int('c')
        return IntConst(0), [('c', c)]  # placeholder, will substitute

    def _template_var_op_const(self, solver, var, op):
        """Template: var op c."""
        c = solver.Int('c')
        return BinOp(op, VarExpr(var), IntConst(0)), [('c', c)]

    def _template_linear(self, solver, var):
        """Template: c * var + d."""
        c = solver.Int('c')
        d = solver.Int('d')
        return BinOp('+', BinOp('*', IntConst(0), VarExpr(var)), IntConst(0)), [('c', c), ('d', d)]

    def _template_binop_op_const(self, solver, v1, v2, op1, op2):
        """Template: (v1 op1 v2) op2 c."""
        c = solver.Int('c')
        return BinOp(op2, BinOp(op1, VarExpr(v1), VarExpr(v2)), IntConst(0)), [('c', c)]

    def _try_template(self, template_fn) -> Optional[Expr]:
        """Try to fill a template using SMT to solve for constants."""
        solver = SMTSolver()
        template_expr, const_vars = template_fn(solver)

        if not const_vars:
            # No constants to solve -- just check directly
            for ex in self.spec.examples:
                try:
                    if evaluate(template_expr, ex.inputs) != ex.output:
                        return None
                except EvalError:
                    return None
            return template_expr

        # Set up SMT constraints: for each example, the template with
        # constants must produce the expected output
        # We compute the template's value symbolically in terms of constants
        for ex in self.spec.examples:
            symbolic_val = self._symbolic_eval(template_expr, ex.inputs, const_vars, solver)
            if symbolic_val is None:
                return None
            # symbolic_val must equal expected output
            solver.add(symbolic_val - ex.output <= 0)
            solver.add(symbolic_val - ex.output >= 0)

        result = solver.check()
        if result == SMTResult.SAT:
            model = solver.model()
            # Substitute constants into template
            return self._substitute_constants(template_expr, const_vars, model)

        return None

    def _symbolic_eval(self, expr, inputs, const_vars, solver):
        """Evaluate template symbolically, returning SMT term."""
        const_names = {name for name, _ in const_vars}
        const_map = {name: var for name, var in const_vars}

        def eval_sym(e):
            if isinstance(e, IntConst):
                # Check if this is a placeholder for a constant
                # Placeholders are IntConst(0) at specific positions
                return SMTIntConst(e.value)
            if isinstance(e, VarExpr):
                if e.name in inputs:
                    return SMTIntConst(inputs[e.name])
                return None
            if isinstance(e, BinOp):
                l = eval_sym(e.left)
                r = eval_sym(e.right)
                if l is None or r is None:
                    return None
                if e.op == '+': return l + r
                if e.op == '-': return l - r
                if e.op == '*': return l * r
                return None
            return None

        # For templates with constants, we need to substitute const vars
        return self._symbolic_eval_with_consts(expr, inputs, const_vars, solver)

    def _symbolic_eval_with_consts(self, expr, inputs, const_vars, solver):
        """Evaluate template symbolically, using SMT vars for constants."""
        # Track which const slot we're filling
        const_iter = iter(const_vars)

        def eval_sym(e, const_positions):
            if isinstance(e, IntConst) and const_positions:
                # This IntConst(0) is a placeholder -- use next const var
                name, smt_var = const_positions.pop(0)
                return smt_var
            if isinstance(e, IntConst):
                return SMTIntConst(e.value)
            if isinstance(e, VarExpr):
                if e.name in inputs:
                    return SMTIntConst(inputs[e.name])
                return None
            if isinstance(e, BinOp):
                l = eval_sym(e.left, const_positions)
                r = eval_sym(e.right, const_positions)
                if l is None or r is None:
                    return None
                if e.op == '+': return l + r
                if e.op == '-': return l - r
                if e.op == '*': return l * r
                return None
            return None

        return eval_sym(expr, list(const_vars))

    def _substitute_constants(self, expr, const_vars, model) -> Expr:
        """Replace constant placeholders with solved values."""
        const_iter = iter(const_vars)

        def subst(e):
            if isinstance(e, IntConst):
                # Replace placeholder with solved value
                try:
                    name, _ = next(const_iter)
                    val = model.get(name, 0)
                    return IntConst(val)
                except StopIteration:
                    return e
            if isinstance(e, VarExpr):
                return e
            if isinstance(e, BinOp):
                return BinOp(e.op, subst(e.left), subst(e.right))
            if isinstance(e, UnaryOp):
                return UnaryOp(e.op, subst(e.arg))
            if isinstance(e, IfExpr):
                return IfExpr(subst(e.cond), subst(e.then_expr), subst(e.else_expr))
            return e

        result = subst(expr)
        return simplify(result)


# ============================================================
# CEGIS -- Counterexample-Guided Inductive Synthesis
# ============================================================

class CEGISSynthesizer:
    """
    CEGIS loop:
    1. Synthesize a candidate program from current examples
    2. Verify candidate against full specification
    3. If verification fails, add counterexample and repeat
    """

    def __init__(self, spec: SynthesisSpec,
                 oracle: Callable = None,
                 verifier: Callable = None,
                 max_iterations: int = 20,
                 max_size: int = 10,
                 max_depth: int = 4):
        self.spec = spec
        self.oracle = oracle      # input -> output (black-box function)
        self.verifier = verifier  # expr -> Optional[counterexample_inputs]
        self.max_iterations = max_iterations
        self.max_size = max_size
        self.max_depth = max_depth
        self.total_candidates = 0

    def synthesize(self) -> SynthesisResult:
        """Run CEGIS loop."""
        examples = list(self.spec.examples)

        for iteration in range(self.max_iterations):
            # Create spec with current examples
            current_spec = SynthesisSpec(
                examples=examples,
                input_vars=self.spec.input_vars,
                output_type=self.spec.output_type,
                constants=self.spec.constants,
                components=self.spec.components
            )

            # Synthesize candidate
            synth = EnumerativeSynthesizer(current_spec, self.max_size, self.max_depth)
            result = synth.synthesize()
            self.total_candidates += synth.candidates_explored

            if not result.success:
                return SynthesisResult(False, None, iteration + 1, self.total_candidates, "cegis")

            candidate = result.program

            # Verify
            counterexample = self._verify(candidate)
            if counterexample is None:
                return SynthesisResult(True, candidate, iteration + 1, self.total_candidates, "cegis")

            # Add counterexample
            examples.append(counterexample)

        return SynthesisResult(False, None, self.max_iterations, self.total_candidates, "cegis")

    def _verify(self, candidate: Expr) -> Optional[IOExample]:
        """Verify candidate against spec. Returns counterexample or None."""
        # Custom verifier
        if self.verifier:
            ce = self.verifier(candidate)
            if ce is not None:
                return ce

        # Oracle-based verification: try a set of test inputs
        if self.oracle:
            test_inputs = self._generate_test_inputs()
            for inputs in test_inputs:
                try:
                    actual = evaluate(candidate, inputs)
                    expected = self.oracle(inputs)
                    if actual != expected:
                        return IOExample(inputs, expected)
                except EvalError:
                    expected = self.oracle(inputs)
                    return IOExample(inputs, expected)

        return None

    def _generate_test_inputs(self) -> list:
        """Generate test inputs for verification."""
        inputs_list = []
        var_names = self.spec.input_vars

        # Test with small values
        test_values = list(range(-5, 6))

        if len(var_names) == 1:
            for v in test_values:
                inputs_list.append({var_names[0]: v})
        elif len(var_names) == 2:
            for v1 in test_values:
                for v2 in test_values:
                    inputs_list.append({var_names[0]: v1, var_names[1]: v2})
        else:
            # For 3+ variables, sample
            import random
            rng = random.Random(42)
            for _ in range(100):
                inputs_list.append({name: rng.randint(-5, 5) for name in var_names})

        return inputs_list


# ============================================================
# Component-Based Synthesis
# ============================================================

@dataclass
class Component:
    """A synthesis component (function/operator)."""
    name: str
    arity: int
    func: Callable
    input_types: list  # list of Type
    output_type: Type

    def apply(self, *args) -> Expr:
        """Create an expression applying this component."""
        if self.arity == 0:
            return self.func()
        elif self.arity == 1:
            return self.func(args[0])
        elif self.arity == 2:
            return self.func(args[0], args[1])
        else:
            return self.func(*args)


# Standard component libraries
ARITHMETIC_COMPONENTS = [
    Component('+', 2, lambda a, b: BinOp('+', a, b), [Type.INT, Type.INT], Type.INT),
    Component('-', 2, lambda a, b: BinOp('-', a, b), [Type.INT, Type.INT], Type.INT),
    Component('*', 2, lambda a, b: BinOp('*', a, b), [Type.INT, Type.INT], Type.INT),
    Component('neg', 1, lambda a: UnaryOp('neg', a), [Type.INT], Type.INT),
]

COMPARISON_COMPONENTS = [
    Component('==', 2, lambda a, b: BinOp('==', a, b), [Type.INT, Type.INT], Type.BOOL),
    Component('<', 2, lambda a, b: BinOp('<', a, b), [Type.INT, Type.INT], Type.BOOL),
    Component('<=', 2, lambda a, b: BinOp('<=', a, b), [Type.INT, Type.INT], Type.BOOL),
    Component('>', 2, lambda a, b: BinOp('>', a, b), [Type.INT, Type.INT], Type.BOOL),
    Component('>=', 2, lambda a, b: BinOp('>=', a, b), [Type.INT, Type.INT], Type.BOOL),
]

CONDITIONAL_COMPONENTS = [
    Component('ite', 3, lambda c, t, e: IfExpr(c, t, e), [Type.BOOL, Type.INT, Type.INT], Type.INT),
]

EXTENDED_COMPONENTS = [
    Component('abs', 1, lambda a: UnaryOp('abs', a), [Type.INT], Type.INT),
    Component('max', 2, lambda a, b: BinOp('max', a, b), [Type.INT, Type.INT], Type.INT),
    Component('min', 2, lambda a, b: BinOp('min', a, b), [Type.INT, Type.INT], Type.INT),
]


class ComponentSynthesizer:
    """
    Component-based synthesis: build programs from a library of components.
    Uses type-directed enumeration.
    """

    def __init__(self, spec: SynthesisSpec, components: list = None,
                 max_depth: int = 3, max_size: int = 9):
        self.spec = spec
        self.components = components or (ARITHMETIC_COMPONENTS + COMPARISON_COMPONENTS)
        self.max_depth = max_depth
        self.max_size = max_size
        self.oe = ObservationalEquivalence(spec.examples)
        self.candidates_explored = 0
        self._target_sig = tuple(ex.output for ex in spec.examples)

    def synthesize(self) -> SynthesisResult:
        """Type-directed bottom-up synthesis from components."""
        # Bank organized by type
        int_bank = []
        bool_bank = []

        # Level 0: atoms
        for name in self.spec.input_vars:
            expr = VarExpr(name)
            self.candidates_explored += 1
            if self.oe.add(expr):
                int_bank.append(expr)
                if self._check(expr):
                    return SynthesisResult(True, expr, 0, self.candidates_explored, "component")

        for c in self.spec.constants:
            if isinstance(c, bool):
                expr = BoolConst(c)
                self.candidates_explored += 1
                if self.oe.add(expr):
                    bool_bank.append(expr)
            else:
                expr = IntConst(c)
                self.candidates_explored += 1
                if self.oe.add(expr):
                    int_bank.append(expr)
                    if self._check(expr):
                        return SynthesisResult(True, expr, 0, self.candidates_explored, "component")

        for depth in range(1, self.max_depth + 1):
            new_int = []
            new_bool = []

            for comp in self.components:
                if comp.arity == 1:
                    source = int_bank if comp.input_types[0] == Type.INT else bool_bank
                    for arg in source:
                        candidate = comp.apply(arg)
                        if expr_size(candidate) > self.max_size:
                            continue
                        self.candidates_explored += 1
                        if self.oe.add(candidate):
                            if comp.output_type == Type.INT:
                                new_int.append(candidate)
                            else:
                                new_bool.append(candidate)
                            if self._check(candidate):
                                return SynthesisResult(True, candidate, depth, self.candidates_explored, "component")

                elif comp.arity == 2:
                    left_source = int_bank if comp.input_types[0] == Type.INT else bool_bank
                    right_source = int_bank if comp.input_types[1] == Type.INT else bool_bank
                    for left in left_source:
                        for right in right_source:
                            candidate = comp.apply(left, right)
                            if expr_size(candidate) > self.max_size:
                                continue
                            self.candidates_explored += 1
                            if self.oe.add(candidate):
                                if comp.output_type == Type.INT:
                                    new_int.append(candidate)
                                else:
                                    new_bool.append(candidate)
                                if self._check(candidate):
                                    return SynthesisResult(True, candidate, depth, self.candidates_explored, "component")

                elif comp.arity == 3 and comp.name == 'ite':
                    for cond in bool_bank:
                        for then_e in int_bank:
                            for else_e in int_bank:
                                if then_e == else_e:
                                    continue
                                candidate = comp.apply(cond, then_e, else_e)
                                if expr_size(candidate) > self.max_size:
                                    continue
                                self.candidates_explored += 1
                                if self.oe.add(candidate):
                                    new_int.append(candidate)
                                    if self._check(candidate):
                                        return SynthesisResult(True, candidate, depth, self.candidates_explored, "component")

            int_bank.extend(new_int)
            bool_bank.extend(new_bool)

        return SynthesisResult(False, None, self.max_depth, self.candidates_explored, "component")

    def _check(self, expr: Expr) -> bool:
        sig = self.oe.signature(expr)
        return sig == self._target_sig


# ============================================================
# Oracle-Guided Synthesis
# ============================================================

class OracleSynthesizer:
    """
    Learn a program from a black-box oracle function.
    Starts with no examples, queries oracle to build up a spec, then synthesizes.
    """

    def __init__(self, oracle: Callable, input_vars: list,
                 components: list = None, constants: list = None,
                 max_queries: int = 20, max_size: int = 10):
        self.oracle = oracle
        self.input_vars = input_vars
        self.components = components or ['+', '-', '*']
        self.constants = constants or [0, 1]
        self.max_queries = max_queries
        self.max_size = max_size

    def synthesize(self) -> SynthesisResult:
        """Learn program from oracle."""
        examples = []

        # Generate initial distinguishing inputs
        initial_inputs = self._initial_inputs()
        for inputs in initial_inputs[:5]:
            output = self.oracle(inputs)
            examples.append(IOExample(inputs, output))

        # CEGIS-style loop with oracle
        for iteration in range(self.max_queries - len(examples)):
            spec = SynthesisSpec(
                examples=examples,
                input_vars=self.input_vars,
                constants=self.constants,
                components=self.components
            )

            synth = EnumerativeSynthesizer(spec, self.max_size, 4)
            result = synth.synthesize()

            if not result.success:
                # Need more examples or larger programs
                new_inputs = self._generate_distinguishing_inputs(examples)
                if new_inputs:
                    output = self.oracle(new_inputs)
                    examples.append(IOExample(new_inputs, output))
                    continue
                return SynthesisResult(False, None, iteration, 0, "oracle")

            candidate = result.program

            # Verify against oracle on more inputs
            verification_inputs = self._verification_inputs()
            found_ce = False
            for inputs in verification_inputs:
                try:
                    actual = evaluate(candidate, inputs)
                except EvalError:
                    actual = None
                expected = self.oracle(inputs)
                if actual != expected:
                    examples.append(IOExample(inputs, expected))
                    found_ce = True
                    break

            if not found_ce:
                return SynthesisResult(True, candidate, iteration + 1, result.candidates_explored, "oracle")

        return SynthesisResult(False, None, self.max_queries, 0, "oracle")

    def _initial_inputs(self) -> list:
        """Generate initial diverse inputs."""
        inputs_list = []
        test_values = [-3, -2, -1, 0, 1, 2, 3, 5, 10]
        if len(self.input_vars) == 1:
            for v in test_values:
                inputs_list.append({self.input_vars[0]: v})
        elif len(self.input_vars) == 2:
            for v1 in test_values[:6]:
                for v2 in test_values[:6]:
                    inputs_list.append({self.input_vars[0]: v1, self.input_vars[1]: v2})
        else:
            import random
            rng = random.Random(42)
            for _ in range(50):
                inputs_list.append({name: rng.randint(-5, 5) for name in self.input_vars})
        return inputs_list

    def _generate_distinguishing_inputs(self, examples: list) -> Optional[dict]:
        """Generate inputs not yet covered."""
        import random
        rng = random.Random(len(examples))
        return {name: rng.randint(-10, 10) for name in self.input_vars}

    def _verification_inputs(self) -> list:
        """Inputs for verification phase."""
        inputs_list = []
        test_values = list(range(-10, 11))
        if len(self.input_vars) == 1:
            for v in test_values:
                inputs_list.append({self.input_vars[0]: v})
        elif len(self.input_vars) == 2:
            for v1 in range(-5, 6):
                for v2 in range(-5, 6):
                    inputs_list.append({self.input_vars[0]: v1, self.input_vars[1]: v2})
        else:
            import random
            rng = random.Random(99)
            for _ in range(100):
                inputs_list.append({name: rng.randint(-10, 10) for name in self.input_vars})
        return inputs_list


# ============================================================
# Conditional Synthesis
# ============================================================

class ConditionalSynthesizer:
    """
    Synthesize programs with if-then-else by:
    1. Partitioning examples into groups
    2. Synthesizing branch conditions and bodies separately
    3. Combining into conditional programs
    """

    def __init__(self, spec: SynthesisSpec, max_branches: int = 3,
                 max_size: int = 8, max_depth: int = 3):
        self.spec = spec
        self.max_branches = max_branches
        self.max_size = max_size
        self.max_depth = max_depth
        self.candidates_explored = 0

    def synthesize(self) -> SynthesisResult:
        """Synthesize conditional program."""
        # Separate arithmetic from comparison components
        comparison_ops = {'<', '<=', '>', '>=', '==', '!=', 'and', 'or'}
        body_components = [c for c in self.spec.components if c not in comparison_ops]
        if not body_components:
            body_components = ['+', '-', '*']

        # First try without conditionals (arithmetic only to avoid bool/int mixing)
        simple_spec = SynthesisSpec(
            examples=self.spec.examples,
            input_vars=self.spec.input_vars,
            constants=self.spec.constants,
            components=body_components
        )
        simple = EnumerativeSynthesizer(simple_spec, self.max_size, self.max_depth)
        result = simple.synthesize()
        self.candidates_explored += simple.candidates_explored
        if result.success:
            return SynthesisResult(True, result.program, 1, self.candidates_explored, "conditional")

        # Try 2-way split
        for partition in self._generate_partitions(2):
            result = self._try_partition(partition)
            if result is not None:
                return result

        # Try 3-way split if allowed
        if self.max_branches >= 3 and len(self.spec.examples) >= 3:
            for partition in self._generate_partitions(3):
                result = self._try_partition(partition)
                if result is not None:
                    return result

        return SynthesisResult(False, None, 0, self.candidates_explored, "conditional")

    def _generate_partitions(self, k: int):
        """Generate k-way partitions of examples."""
        n = len(self.spec.examples)
        if k == 2:
            # All ways to split into 2 non-empty groups
            for i in range(1, 2 ** n - 1):
                group1 = [self.spec.examples[j] for j in range(n) if i & (1 << j)]
                group2 = [self.spec.examples[j] for j in range(n) if not (i & (1 << j))]
                if group1 and group2:
                    yield [group1, group2]
        elif k == 3 and n <= 8:
            # Enumerate 3-way partitions (expensive, limit to small n)
            for assignment in itertools.product(range(3), repeat=n):
                groups = [[], [], []]
                for j, g in enumerate(assignment):
                    groups[g].append(self.spec.examples[j])
                if all(groups):
                    yield groups

    def _try_partition(self, groups: list) -> Optional[SynthesisResult]:
        """Try to synthesize a conditional program for this partition."""
        # Separate arithmetic from comparison components for body synthesis
        comparison_ops = {'<', '<=', '>', '>=', '==', '!=', 'and', 'or'}
        body_components = [c for c in self.spec.components if c not in comparison_ops]
        if not body_components:
            body_components = ['+', '-', '*']

        # Synthesize body for each group
        bodies = []
        for group in groups:
            spec = SynthesisSpec(
                examples=group,
                input_vars=self.spec.input_vars,
                constants=self.spec.constants,
                components=body_components
            )
            synth = EnumerativeSynthesizer(spec, self.max_size, self.max_depth)
            result = synth.synthesize()
            self.candidates_explored += synth.candidates_explored
            if not result.success:
                return None
            bodies.append(result.program)

        # Synthesize conditions to distinguish groups
        conditions = self._synthesize_conditions(groups)
        if conditions is None:
            return None

        # Build conditional program
        if len(groups) == 2:
            program = IfExpr(conditions[0], bodies[0], bodies[1])
        else:
            # Nested if-then-else
            program = IfExpr(conditions[1], bodies[1], bodies[2])
            program = IfExpr(conditions[0], bodies[0], program)

        # Verify against all examples
        for ex in self.spec.examples:
            try:
                val = evaluate(program, ex.inputs)
                if val != ex.output:
                    return None
            except EvalError:
                return None

        return SynthesisResult(True, program, len(groups), self.candidates_explored, "conditional")

    def _synthesize_conditions(self, groups: list) -> Optional[list]:
        """Synthesize boolean conditions that separate the groups."""
        conditions = []

        for i in range(len(groups) - 1):
            # Condition should be True for group[i], False for others
            cond_examples = []
            for ex in groups[i]:
                cond_examples.append(IOExample(ex.inputs, True))
            for j in range(len(groups)):
                if j != i:
                    for ex in groups[j]:
                        cond_examples.append(IOExample(ex.inputs, False))

            # Synthesize the boolean condition
            cond_spec = SynthesisSpec(
                examples=cond_examples,
                input_vars=self.spec.input_vars,
                output_type=Type.BOOL,
                constants=self.spec.constants,
                components=['<', '<=', '>', '>=', '==', '!=']
            )
            synth = EnumerativeSynthesizer(cond_spec, self.max_size, self.max_depth)
            result = synth.synthesize()
            self.candidates_explored += synth.candidates_explored
            if not result.success:
                return None
            conditions.append(result.program)

        return conditions


# ============================================================
# Program Equivalence (via testing)
# ============================================================

def programs_equivalent(expr1: Expr, expr2: Expr, input_vars: list,
                       test_range: range = range(-10, 11)) -> bool:
    """Check if two programs are equivalent on a range of inputs."""
    if len(input_vars) == 1:
        for v in test_range:
            env = {input_vars[0]: v}
            try:
                v1 = evaluate(expr1, env)
                v2 = evaluate(expr2, env)
                if v1 != v2:
                    return False
            except EvalError:
                continue
    elif len(input_vars) == 2:
        for v1 in test_range:
            for v2 in test_range:
                env = {input_vars[0]: v1, input_vars[1]: v2}
                try:
                    r1 = evaluate(expr1, env)
                    r2 = evaluate(expr2, env)
                    if r1 != r2:
                        return False
                except EvalError:
                    continue
    return True


# ============================================================
# Simplification / Normalization
# ============================================================

def simplify(expr: Expr) -> Expr:
    """Simplify an expression using algebraic rules."""
    if isinstance(expr, (IntConst, BoolConst, VarExpr)):
        return expr

    if isinstance(expr, UnaryOp):
        arg = simplify(expr.arg)
        if expr.op == 'neg':
            if isinstance(arg, IntConst):
                return IntConst(-arg.value)
            if isinstance(arg, UnaryOp) and arg.op == 'neg':
                return arg.arg
        if expr.op == 'not':
            if isinstance(arg, BoolConst):
                return BoolConst(not arg.value)
            if isinstance(arg, UnaryOp) and arg.op == 'not':
                return arg.arg
        if expr.op == 'abs':
            if isinstance(arg, IntConst):
                return IntConst(abs(arg.value))
        return UnaryOp(expr.op, arg)

    if isinstance(expr, BinOp):
        left = simplify(expr.left)
        right = simplify(expr.right)

        # Constant folding
        if isinstance(left, IntConst) and isinstance(right, IntConst):
            try:
                val = evaluate(BinOp(expr.op, left, right), {})
                if isinstance(val, bool):
                    return BoolConst(val)
                return IntConst(val)
            except EvalError:
                pass

        # Identity rules
        if expr.op == '+':
            if isinstance(right, IntConst) and right.value == 0:
                return left
            if isinstance(left, IntConst) and left.value == 0:
                return right
        if expr.op == '-':
            if isinstance(right, IntConst) and right.value == 0:
                return left
            if left == right:
                return IntConst(0)
        if expr.op == '*':
            if isinstance(right, IntConst) and right.value == 1:
                return left
            if isinstance(left, IntConst) and left.value == 1:
                return right
            if isinstance(right, IntConst) and right.value == 0:
                return IntConst(0)
            if isinstance(left, IntConst) and left.value == 0:
                return IntConst(0)

        return BinOp(expr.op, left, right)

    if isinstance(expr, IfExpr):
        cond = simplify(expr.cond)
        then_e = simplify(expr.then_expr)
        else_e = simplify(expr.else_expr)

        if isinstance(cond, BoolConst):
            return then_e if cond.value else else_e
        if then_e == else_e:
            return then_e

        return IfExpr(cond, then_e, else_e)

    return expr


# ============================================================
# Pretty Printer
# ============================================================

def pretty_print(expr: Expr, indent: int = 0) -> str:
    """Pretty print an expression as a multi-line tree."""
    prefix = "  " * indent
    if isinstance(expr, (IntConst, BoolConst, VarExpr)):
        return f"{prefix}{expr}"
    if isinstance(expr, UnaryOp):
        return f"{prefix}{expr.op}(\n{pretty_print(expr.arg, indent + 1)}\n{prefix})"
    if isinstance(expr, BinOp):
        return (f"{prefix}{expr.op}(\n"
                f"{pretty_print(expr.left, indent + 1)},\n"
                f"{pretty_print(expr.right, indent + 1)}\n"
                f"{prefix})")
    if isinstance(expr, IfExpr):
        return (f"{prefix}if(\n"
                f"{pretty_print(expr.cond, indent + 1)},\n"
                f"{pretty_print(expr.then_expr, indent + 1)},\n"
                f"{pretty_print(expr.else_expr, indent + 1)}\n"
                f"{prefix})")
    return f"{prefix}{expr}"


# ============================================================
# Convenience: synthesize()
# ============================================================

def synthesize(examples: list, input_vars: list, method: str = "enumerative",
               components: list = None, constants: list = None,
               max_size: int = 10, max_depth: int = 4, **kwargs) -> SynthesisResult:
    """
    High-level synthesis function.

    Args:
        examples: list of (inputs_dict, output) or IOExample
        input_vars: list of variable names
        method: "enumerative", "constraint", "cegis", "component", "conditional"
        components: operator list (default ['+', '-', '*'])
        constants: constant list (default [0, 1])
        max_size: max AST node count
        max_depth: max AST depth

    Returns:
        SynthesisResult
    """
    # Normalize examples
    io_examples = []
    for ex in examples:
        if isinstance(ex, IOExample):
            io_examples.append(ex)
        else:
            inputs, output = ex
            io_examples.append(IOExample(inputs, output))

    spec = SynthesisSpec(
        examples=io_examples,
        input_vars=input_vars,
        constants=constants if constants is not None else [0, 1],
        components=components if components is not None else ['+', '-', '*']
    )

    if method == "enumerative":
        synth = EnumerativeSynthesizer(spec, max_size, max_depth)
        return synth.synthesize()
    elif method == "constraint":
        synth = ConstraintSynthesizer(spec, kwargs.get('max_nodes', 5))
        return synth.synthesize()
    elif method == "cegis":
        synth = CEGISSynthesizer(spec,
                                  oracle=kwargs.get('oracle'),
                                  verifier=kwargs.get('verifier'),
                                  max_iterations=kwargs.get('max_iterations', 20),
                                  max_size=max_size,
                                  max_depth=max_depth)
        return synth.synthesize()
    elif method == "component":
        comp_list = kwargs.get('component_list', ARITHMETIC_COMPONENTS + COMPARISON_COMPONENTS)
        synth = ComponentSynthesizer(spec, comp_list, max_depth, max_size)
        return synth.synthesize()
    elif method == "conditional":
        synth = ConditionalSynthesizer(spec, kwargs.get('max_branches', 3),
                                        max_size, max_depth)
        return synth.synthesize()
    else:
        raise ValueError(f"Unknown synthesis method: {method}")
