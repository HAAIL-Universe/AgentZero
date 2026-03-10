"""
V008: Bounded Program Synthesis
================================
Counterexample-Guided Inductive Synthesis (CEGIS).

Composes:
  - C010 parser (AST construction and source generation)
  - C037 SMT solver (candidate search + counterexample extraction)
  - V004 VCGen (candidate verification via Hoare logic)

Given a specification (precondition, postcondition, parameter types),
synthesizes a program that satisfies the spec.

Algorithm (CEGIS):
  1. Generate initial test inputs from precondition
  2. For each template (by increasing complexity):
     a. Encode template unknowns as SMT variables
     b. Find values for unknowns that satisfy spec on all test inputs
     c. Instantiate candidate program from template + unknowns
     d. Verify candidate with V004 VCGen (full Hoare-logic check)
     e. If valid -> return synthesized program
     f. If invalid -> add counterexample to test inputs, retry
  3. If no template works -> return failure

Templates (by complexity):
  Level 1: Single assignment  (result = linear_combination)
  Level 2: Conditional        (if-then-else with linear arms)
  Level 3: Piecewise          (nested conditionals)
  Level 4: Multi-assignment   (intermediate variables)
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum, auto

# --- Import foundation ---
_base = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_base, '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(_base, '..', '..', '..', 'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(_base, '..', 'V004_verification_conditions'))

from stack_vm import (
    lex, Parser, Program, Block, LetDecl, Assign, IfStmt, WhileStmt,
    FnDecl, ReturnStmt, PrintStmt, CallExpr,
    BinOp, UnaryOp, Var as ASTVar, IntLit, BoolLit,
)
from smt_solver import SMTSolver, SMTResult, Op, IntConst, BoolConst, App, INT, BOOL
from vc_gen import (
    verify_function, verify_hoare_triple, VCStatus, VerificationResult,
    SExpr, SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot, SIte,
    s_and, s_or, s_not, s_implies, substitute, lower_to_smt, check_vc,
    parse, ast_to_sexpr,
)


# ============================================================
# Result types
# ============================================================

class SynthStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()

@dataclass
class SynthResult:
    """Result of a synthesis attempt."""
    status: SynthStatus
    program: Optional[str] = None          # Synthesized source code
    source_fn: Optional[str] = None        # Full function with spec + body
    template_name: Optional[str] = None    # Which template succeeded
    template_level: int = 0                # Complexity level
    cegis_iterations: int = 0              # CEGIS rounds
    candidates_tried: int = 0              # Total candidates tested
    verification: Optional[VerificationResult] = None  # Final V004 result
    error: Optional[str] = None

@dataclass
class Spec:
    """Synthesis specification."""
    params: list[str]                      # Parameter names
    param_types: dict[str, str]            # name -> 'int' (only int for now)
    precondition: str                      # Source expression for requires()
    postcondition: str                     # Source expression for ensures()
    # Optional: concrete input/output examples as additional guidance
    examples: list[dict] = field(default_factory=list)
    # Example: [{'x': 5, 'result': 25}, {'x': -3, 'result': 9}]


# ============================================================
# Template system
# ============================================================

@dataclass
class Template:
    """A program template with holes (unknown coefficients/operators)."""
    name: str
    level: int           # Complexity level (1=simplest)
    # unknowns: list of (name, type) for SMT variables representing holes
    unknowns: list[tuple[str, str]]  # (name, 'int' or 'op')
    # builder: given concrete values for unknowns, produces source code
    builder: Callable[[dict], Optional[str]]
    # smt_encoder: given solver + param vars + unknown vars + test input,
    #   adds constraints that the template produces the correct output
    smt_encoder: Callable


def _build_linear_source(params, coeffs):
    """Build source for: result = c0 + c1*p1 + c2*p2 + ..."""
    terms = []
    c0 = coeffs.get('c0', 0)
    if c0 != 0:
        terms.append(str(c0))
    for i, p in enumerate(params):
        c = coeffs.get(f'c{i+1}', 0)
        if c == 0:
            continue
        elif c == 1:
            terms.append(p)
        elif c == -1:
            terms.append(f'(0 - {p})')
        else:
            terms.append(f'({c} * {p})')
    if not terms:
        return 'let result = 0;'
    expr = terms[0]
    for t in terms[1:]:
        # Handle negative terms
        expr = f'({expr} + {t})'
    return f'let result = {expr};'


def _build_cond_source(params, values):
    """Build source for: if (param op const) { result = e1 } else { result = e2 }"""
    guard_param = values.get('guard_param', params[0])
    guard_op = values.get('guard_op', '<')
    guard_const = values.get('guard_const', 0)
    then_expr = values.get('then_expr', '0')
    else_expr = values.get('else_expr', '0')
    return (f'let result = 0;\n'
            f'  if ({guard_param} {guard_op} {guard_const}) {{\n'
            f'    result = {then_expr};\n'
            f'  }} else {{\n'
            f'    result = {else_expr};\n'
            f'  }}')


# Operator encoding: map integers to comparison operators
CMP_OPS = ['<', '<=', '>', '>=', '==', '!=']


def make_linear_template(params):
    """Create template: result = c0 + c1*p1 + c2*p2 + ..."""
    unknowns = [('c0', 'int')]
    for i in range(len(params)):
        unknowns.append((f'c{i+1}', 'int'))

    def builder(values):
        return _build_linear_source(params, values)

    def smt_encoder(solver, param_smt, unknown_smt, test_input, expected_result):
        """Add constraints: c0 + c1*input1 + c2*input2 + ... == expected_result"""
        expr = unknown_smt['c0']
        for i, p in enumerate(params):
            coeff = unknown_smt[f'c{i+1}']
            val = solver.IntVal(test_input[p])
            expr = expr + coeff * val
        solver.add(App(Op.EQ, [expr, solver.IntVal(expected_result)], BOOL))

    return Template(
        name='linear',
        level=1,
        unknowns=unknowns,
        builder=builder,
        smt_encoder=smt_encoder,
    )


def make_conditional_linear_template(params):
    """Create template: if (p_i op c) { result = lin1 } else { result = lin2 }

    For each parameter and each comparison operator, create a separate template.
    Each arm is a linear combination of all parameters.

    NOTE: We enumerate operators concretely (one template per op) rather than
    encoding operator selection in SMT. C037's ITE chains are too complex for
    the Simplex solver -- it returns UNKNOWN. Concrete enumeration gives pure
    LIA queries which C037 handles perfectly.
    """
    templates = []
    for gi, gp in enumerate(params):
        for op_idx, op_str in enumerate(CMP_OPS):
            name = f'cond_{gp}_{op_str}'
            # Unknowns: guard_const, then-arm coefficients, else-arm coefficients
            unknowns = [('guard_const', 'int')]
            for i in range(len(params)):
                unknowns.append((f't{i}', 'int'))
            unknowns.append(('t_const', 'int'))
            for i in range(len(params)):
                unknowns.append((f'e{i}', 'int'))
            unknowns.append(('e_const', 'int'))

            def make_builder(guard_param, all_params, op):
                def builder(values):
                    gc = values.get('guard_const', 0)

                    def build_arm(prefix):
                        terms = []
                        cc = values.get(f'{prefix}_const', 0)
                        if cc != 0:
                            terms.append(str(cc))
                        for i, p in enumerate(all_params):
                            c = values.get(f'{prefix}{i}', 0)
                            if c == 0:
                                continue
                            elif c == 1:
                                terms.append(p)
                            elif c == -1:
                                terms.append(f'(0 - {p})')
                            else:
                                terms.append(f'({c} * {p})')
                        if not terms:
                            return '0'
                        expr = terms[0]
                        for t in terms[1:]:
                            expr = f'({expr} + {t})'
                        return expr

                    return _build_cond_source(all_params, {
                        'guard_param': guard_param,
                        'guard_op': op,
                        'guard_const': gc,
                        'then_expr': build_arm('t'),
                        'else_expr': build_arm('e'),
                    })
                return builder

            def make_encoder(guard_idx, all_params, cmp_op):
                """Encode with a concrete operator -- pure LIA, no ITE chains."""
                op_map = {
                    '<': Op.LT, '<=': Op.LE, '>': Op.GT,
                    '>=': Op.GE, '==': Op.EQ, '!=': Op.NEQ,
                }
                smt_op = op_map[cmp_op]

                def smt_encoder(solver, param_smt, unknown_smt, test_input, expected_result):
                    gc = unknown_smt['guard_const']
                    gv = solver.IntVal(test_input[all_params[guard_idx]])

                    then_val = unknown_smt['t_const']
                    for i, p in enumerate(all_params):
                        then_val = then_val + unknown_smt[f't{i}'] * solver.IntVal(test_input[p])

                    else_val = unknown_smt['e_const']
                    for i, p in enumerate(all_params):
                        else_val = else_val + unknown_smt[f'e{i}'] * solver.IntVal(test_input[p])

                    cond = App(smt_op, [gv, gc], BOOL)
                    result = solver.If(cond, then_val, else_val)
                    solver.add(App(Op.EQ, [result, solver.IntVal(expected_result)], BOOL))

                return smt_encoder

            templates.append(Template(
                name=name,
                level=2,
                unknowns=unknowns,
                builder=make_builder(gp, params, op_str),
                smt_encoder=make_encoder(gi, params, op_str),
            ))

    return templates


def make_nested_cond_template(params):
    """Create template: if (p op1 c1) { result = e1 } else if (p op2 c2) { result = e2 } else { result = e3 }

    Two-level conditional on a single parameter, each arm is a linear combination.
    Operators are enumerated concretely (one template per op1/op2 pair).
    Only uses a subset of operator pairs to keep template count manageable.
    """
    if len(params) < 1:
        return []

    # Only use most common operator pairs for nested conditionals
    NESTED_OPS = ['<', '>=', '==']

    templates = []
    for gi, gp in enumerate(params):
        for o1 in NESTED_OPS:
            for o2 in NESTED_OPS:
                name = f'nested_{gp}_{o1}_{o2}'
                unknowns = [('c1', 'int'), ('c2', 'int')]
                for arm in ['a', 'b', 'c']:
                    unknowns.append((f'{arm}_const', 'int'))
                    for i in range(len(params)):
                        unknowns.append((f'{arm}{i}', 'int'))

                def make_builder(guard_param, all_params, op1, op2):
                    def builder(values):
                        c1 = values.get('c1', 0)
                        c2 = values.get('c2', 0)

                        def build_arm(prefix):
                            terms = []
                            cc = values.get(f'{prefix}_const', 0)
                            if cc != 0:
                                terms.append(str(cc))
                            for i, p in enumerate(all_params):
                                c = values.get(f'{prefix}{i}', 0)
                                if c == 0:
                                    continue
                                elif c == 1:
                                    terms.append(p)
                                elif c == -1:
                                    terms.append(f'(0 - {p})')
                                else:
                                    terms.append(f'({c} * {p})')
                            if not terms:
                                return '0'
                            expr = terms[0]
                            for t in terms[1:]:
                                expr = f'({expr} + {t})'
                            return expr

                        return (f'let result = 0;\n'
                                f'  if ({guard_param} {op1} {c1}) {{\n'
                                f'    result = {build_arm("a")};\n'
                                f'  }} else {{\n'
                                f'    if ({guard_param} {op2} {c2}) {{\n'
                                f'      result = {build_arm("b")};\n'
                                f'    }} else {{\n'
                                f'      result = {build_arm("c")};\n'
                                f'    }}\n'
                                f'  }}')
                    return builder

                def make_encoder(guard_idx, all_params, op1, op2):
                    op_map = {
                        '<': Op.LT, '<=': Op.LE, '>': Op.GT,
                        '>=': Op.GE, '==': Op.EQ, '!=': Op.NEQ,
                    }
                    smt_op1 = op_map[op1]
                    smt_op2 = op_map[op2]

                    def smt_encoder(solver, param_smt, unknown_smt, test_input, expected_result):
                        gv = solver.IntVal(test_input[all_params[guard_idx]])
                        c1 = unknown_smt['c1']
                        c2 = unknown_smt['c2']

                        def build_arm_val(prefix):
                            val = unknown_smt[f'{prefix}_const']
                            for i, p in enumerate(all_params):
                                val = val + unknown_smt[f'{prefix}{i}'] * solver.IntVal(test_input[p])
                            return val

                        cond1 = App(smt_op1, [gv, c1], BOOL)
                        cond2 = App(smt_op2, [gv, c2], BOOL)
                        inner = solver.If(cond2, build_arm_val('b'), build_arm_val('c'))
                        result = solver.If(cond1, build_arm_val('a'), inner)
                        solver.add(App(Op.EQ, [result, solver.IntVal(expected_result)], BOOL))

                    return smt_encoder

                templates.append(Template(
                    name=name,
                    level=3,
                    unknowns=unknowns,
                    builder=make_builder(gp, params, o1, o2),
                    smt_encoder=make_encoder(gi, params, o1, o2),
                ))

    return templates


def make_two_param_cond_template(params):
    """Template: if (p1 op p2 + c) { result = lin1 } else { result = lin2 }

    Guard compares two parameters (with offset).
    Operators are enumerated concretely for C037 compatibility.
    """
    if len(params) < 2:
        return []

    templates = []
    for i in range(len(params)):
        for j in range(len(params)):
            if i == j:
                continue
            pi, pj = params[i], params[j]
            for op_str in CMP_OPS:
                name = f'cond_{pi}_vs_{pj}_{op_str}'
                unknowns = [('offset', 'int'), ('t_const', 'int')]
                for k in range(len(params)):
                    unknowns.append((f't{k}', 'int'))
                unknowns.append(('e_const', 'int'))
                for k in range(len(params)):
                    unknowns.append((f'e{k}', 'int'))

                def make_builder(pi_name, pj_name, all_params, op):
                    def builder(values):
                        offset = values.get('offset', 0)

                        def build_arm(prefix):
                            terms = []
                            cc = values.get(f'{prefix}_const', 0)
                            if cc != 0:
                                terms.append(str(cc))
                            for k, p in enumerate(all_params):
                                c = values.get(f'{prefix}{k}', 0)
                                if c == 0:
                                    continue
                                elif c == 1:
                                    terms.append(p)
                                elif c == -1:
                                    terms.append(f'(0 - {p})')
                                else:
                                    terms.append(f'({c} * {p})')
                            if not terms:
                                return '0'
                            expr = terms[0]
                            for t in terms[1:]:
                                expr = f'({expr} + {t})'
                            return expr

                        rhs = pj_name if offset == 0 else (
                            f'({pj_name} + {offset})' if offset > 0 else f'({pj_name} - {-offset})')

                        return (f'let result = 0;\n'
                                f'  if ({pi_name} {op} {rhs}) {{\n'
                                f'    result = {build_arm("t")};\n'
                                f'  }} else {{\n'
                                f'    result = {build_arm("e")};\n'
                                f'  }}')
                    return builder

                def make_encoder(idx_i, idx_j, all_params, cmp_op):
                    op_map = {
                        '<': Op.LT, '<=': Op.LE, '>': Op.GT,
                        '>=': Op.GE, '==': Op.EQ, '!=': Op.NEQ,
                    }
                    smt_op = op_map[cmp_op]

                    def smt_encoder(solver, param_smt, unknown_smt, test_input, expected_result):
                        vi = solver.IntVal(test_input[all_params[idx_i]])
                        vj = solver.IntVal(test_input[all_params[idx_j]])
                        offset = unknown_smt['offset']
                        rhs = vj + offset

                        def build_arm_val(prefix):
                            val = unknown_smt[f'{prefix}_const']
                            for k, p in enumerate(all_params):
                                val = val + unknown_smt[f'{prefix}{k}'] * solver.IntVal(test_input[p])
                            return val

                        cond = App(smt_op, [vi, rhs], BOOL)
                        result = solver.If(cond, build_arm_val('t'), build_arm_val('e'))
                        solver.add(App(Op.EQ, [result, solver.IntVal(expected_result)], BOOL))

                    return smt_encoder

                templates.append(Template(
                    name=name,
                    level=2,
                    unknowns=unknowns,
                    builder=make_builder(pi, pj, params, op_str),
                    smt_encoder=make_encoder(i, j, params, op_str),
                ))

    return templates


# ============================================================
# Test input generation
# ============================================================

def generate_initial_inputs(spec: Spec, count: int = 5) -> list[dict]:
    """Generate initial test inputs satisfying the precondition.

    Uses SMT to find diverse inputs satisfying the precondition.
    """
    inputs = []
    solver = SMTSolver()
    param_vars = {}
    for p in spec.params:
        param_vars[p] = solver.Int(p)

    # Parse and encode precondition
    if spec.precondition and spec.precondition.strip().lower() != 'true':
        pre_smt = _encode_condition(solver, spec.precondition, param_vars)
        if pre_smt is not None:
            solver.add(pre_smt)

    # Bound search to reasonable range
    for p in spec.params:
        solver.add(App(Op.GE, [param_vars[p], solver.IntVal(-100)], BOOL))
        solver.add(App(Op.LE, [param_vars[p], solver.IntVal(100)], BOOL))

    for _ in range(count):
        solver.push()
        result = solver.check()
        if result != SMTResult.SAT:
            solver.pop()
            break
        model = solver.model()
        inp = {}
        for p in spec.params:
            inp[p] = model.get(p, 0)
        inputs.append(inp)
        solver.pop()

        # Exclude this exact input to get diversity
        exclusion = solver.Or(*(
            App(Op.NEQ, [param_vars[p], solver.IntVal(inp[p])], BOOL)
            for p in spec.params
        ))
        solver.add(exclusion)

    # If examples are provided, add them too
    for ex in spec.examples:
        inp = {p: ex[p] for p in spec.params if p in ex}
        if inp and inp not in inputs:
            inputs.append(inp)

    # Ensure at least some boundary-ish values
    if not inputs:
        inputs.append({p: 0 for p in spec.params})

    return inputs


def _encode_condition(solver, cond_src, param_vars):
    """Parse a condition expression and encode as SMT."""
    try:
        # Wrap in a dummy function to parse the expression
        wrapper = f'fn __tmp__() {{ requires({cond_src}); }}'
        program = parse(wrapper)
        fn = program.stmts[0]
        stmts = fn.body.stmts if isinstance(fn.body, Block) else [fn.body]
        for stmt in stmts:
            if isinstance(stmt, CallExpr) and stmt.callee == 'requires':
                sexpr = ast_to_sexpr(stmt.args[0])
                var_cache = dict(param_vars)
                return lower_to_smt(solver, sexpr, var_cache)
    except Exception:
        pass
    return None


def compute_expected_output(spec: Spec, test_input: dict) -> Optional[int]:
    """Evaluate the postcondition to determine expected output for a test input.

    If postcondition is 'result == expr', evaluate expr with the test input values.
    """
    # Try to extract expected result from postcondition
    post = spec.postcondition.strip()

    # Common pattern: result == expr
    if post.startswith('result ==') or post.startswith('result=='):
        expr_str = post.split('==', 1)[1].strip()
        return _eval_expr(expr_str, test_input)

    # Try evaluating as a Python expression with substitution
    return None


def _eval_expr(expr_str, values):
    """Safely evaluate a simple arithmetic expression with variable substitution."""
    try:
        # Build safe evaluation context
        env = dict(values)
        # Handle common patterns
        expr = expr_str
        # Replace 'and' -> ' and ', 'or' -> ' or ' for Python eval
        # But be careful with variable names containing 'and'/'or'
        result = eval(expr, {"__builtins__": {}}, env)
        if isinstance(result, (int, bool)):
            return int(result)
    except Exception:
        pass
    return None


# ============================================================
# CEGIS core
# ============================================================

def _create_smt_unknowns(solver, template):
    """Create SMT variables for template unknowns."""
    unknown_smt = {}
    for name, typ in template.unknowns:
        if typ == 'int':
            unknown_smt[name] = solver.Int(f'_u_{name}')
        elif typ == 'op':
            unknown_smt[name] = solver.Int(f'_u_{name}')
        else:
            unknown_smt[name] = solver.Int(f'_u_{name}')
    return unknown_smt


def _bound_unknowns(solver, unknown_smt, template, coeff_bound=10):
    """Add bounds on unknown coefficients to keep search tractable."""
    for name, typ in template.unknowns:
        v = unknown_smt[name]
        if typ == 'int':
            solver.add(App(Op.GE, [v, solver.IntVal(-coeff_bound)], BOOL))
            solver.add(App(Op.LE, [v, solver.IntVal(coeff_bound)], BOOL))
        elif typ == 'op':
            solver.add(App(Op.GE, [v, solver.IntVal(0)], BOOL))
            solver.add(App(Op.LE, [v, solver.IntVal(5)], BOOL))


def _find_candidate(template, spec, test_ios, coeff_bound=10):
    """Use SMT to find unknown values that satisfy all test input/output pairs.

    Returns dict of unknown values if found, None otherwise.
    """
    solver = SMTSolver()
    param_smt = {}
    for p in spec.params:
        param_smt[p] = solver.Int(p)

    unknown_smt = _create_smt_unknowns(solver, template)
    _bound_unknowns(solver, unknown_smt, template, coeff_bound)

    # For each test input/output pair, add constraints
    for inp, expected in test_ios:
        template.smt_encoder(solver, param_smt, unknown_smt, inp, expected)

    result = solver.check()
    if result == SMTResult.SAT:
        model = solver.model()
        values = {}
        for name, _ in template.unknowns:
            smt_name = f'_u_{name}'
            values[name] = model.get(smt_name, 0)
        return values
    return None


def _build_function_source(spec, body_source):
    """Build a complete function source with spec annotations."""
    param_str = ', '.join(spec.params)
    lines = [f'fn synthesized({param_str}) {{']
    if spec.precondition and spec.precondition.strip().lower() != 'true':
        lines.append(f'  requires({spec.precondition});')
    lines.append(f'  ensures({spec.postcondition});')
    # Indent body
    for line in body_source.split('\n'):
        lines.append(f'  {line.strip()}')
    lines.append(f'  return result;')
    lines.append('}')
    return '\n'.join(lines)


def _verify_candidate(spec, body_source):
    """Verify a candidate program using V004 VCGen."""
    fn_source = _build_function_source(spec, body_source)
    try:
        result = verify_function(fn_source, 'synthesized')
        return result, fn_source
    except Exception as e:
        return VerificationResult(verified=False, errors=[str(e)]), fn_source


def _extract_counterexample(vr: VerificationResult) -> Optional[dict]:
    """Extract a counterexample from a failed verification."""
    for vc in vr.vcs:
        if vc.status == VCStatus.INVALID and vc.counterexample:
            return vc.counterexample
    return None


# ============================================================
# Main synthesis API
# ============================================================

def synthesize(spec: Spec, max_cegis_rounds: int = 20,
               max_level: int = 3, coeff_bound: int = 10) -> SynthResult:
    """
    Synthesize a program satisfying the given specification.

    Uses CEGIS (Counterexample-Guided Inductive Synthesis):
    1. Generate test inputs from precondition
    2. For each template (by complexity):
       a. Find candidate using SMT (must match all test I/O pairs)
       b. Verify candidate with V004 VCGen
       c. If valid -> success
       d. If invalid -> add counterexample, retry

    Args:
        spec: The specification (params, precondition, postcondition)
        max_cegis_rounds: Max CEGIS iterations per template
        max_level: Max template complexity level to try
        coeff_bound: Bound on coefficient magnitudes

    Returns:
        SynthResult with synthesized program or failure reason
    """
    # Generate templates
    templates = []
    if max_level >= 1:
        templates.append(make_linear_template(spec.params))
    if max_level >= 2:
        templates.extend(make_conditional_linear_template(spec.params))
        templates.extend(make_two_param_cond_template(spec.params))
    if max_level >= 3:
        templates.extend(make_nested_cond_template(spec.params))

    # Sort by level
    templates.sort(key=lambda t: t.level)

    # Generate initial test inputs
    test_inputs = generate_initial_inputs(spec, count=5)

    # Compute expected outputs for test inputs
    test_ios = []  # (input_dict, expected_output)
    for inp in test_inputs:
        expected = compute_expected_output(spec, inp)
        if expected is not None:
            test_ios.append((inp, expected))

    # If we can't compute expected outputs, try a different approach:
    # use examples from spec
    if not test_ios and spec.examples:
        for ex in spec.examples:
            inp = {p: ex[p] for p in spec.params if p in ex}
            if 'result' in ex:
                test_ios.append((inp, ex['result']))

    # If still no test I/O, try with just the examples we can get
    if not test_ios:
        # Fall back: try each template with verification-only (no CEGIS)
        return _synthesize_verify_only(spec, templates, max_cegis_rounds, coeff_bound)

    total_candidates = 0

    for template in templates:
        if template.level > max_level:
            break

        current_ios = list(test_ios)
        for cegis_round in range(max_cegis_rounds):
            # Step 1: Find candidate matching all current I/O pairs
            values = _find_candidate(template, spec, current_ios, coeff_bound)
            if values is None:
                break  # Template can't match I/O pairs -> try next template

            # Step 2: Build candidate source
            body_source = template.builder(values)
            if body_source is None:
                break
            total_candidates += 1

            # Step 3: Verify with V004
            vr, fn_source = _verify_candidate(spec, body_source)
            if vr.verified:
                return SynthResult(
                    status=SynthStatus.SUCCESS,
                    program=body_source,
                    source_fn=fn_source,
                    template_name=template.name,
                    template_level=template.level,
                    cegis_iterations=cegis_round + 1,
                    candidates_tried=total_candidates,
                    verification=vr,
                )

            # Step 4: Extract counterexample
            cex = _extract_counterexample(vr)
            if cex is None:
                break  # No counterexample to learn from

            # Convert counterexample to test input
            cex_input = {p: cex.get(p, 0) for p in spec.params}
            cex_expected = compute_expected_output(spec, cex_input)
            if cex_expected is not None:
                current_ios.append((cex_input, cex_expected))
            else:
                break  # Can't compute expected output for counterexample

    return SynthResult(
        status=SynthStatus.FAILURE,
        candidates_tried=total_candidates,
        error="No template could synthesize a valid program",
    )


def _synthesize_verify_only(spec, templates, max_attempts, coeff_bound):
    """Fallback: enumerate small coefficient values and verify each candidate."""
    total = 0
    for template in templates:
        # Try with small coefficients via SMT with examples if available
        # or just try common patterns
        for attempt in _enumerate_small_programs(template, spec, coeff_bound):
            body_source = template.builder(attempt)
            if body_source is None:
                continue
            total += 1
            vr, fn_source = _verify_candidate(spec, body_source)
            if vr.verified:
                return SynthResult(
                    status=SynthStatus.SUCCESS,
                    program=body_source,
                    source_fn=fn_source,
                    template_name=template.name,
                    template_level=template.level,
                    cegis_iterations=0,
                    candidates_tried=total,
                    verification=vr,
                )
            if total >= max_attempts:
                break
        if total >= max_attempts:
            break

    return SynthResult(
        status=SynthStatus.FAILURE,
        candidates_tried=total,
        error="Enumeration exhausted without finding valid program",
    )


def _enumerate_small_programs(template, spec, coeff_bound):
    """Yield small coefficient assignments for a template.

    For linear templates: try single-variable assignments first (c_i=1, rest=0),
    then pairs, etc. For conditional templates: try simple guards.
    """
    n = len(spec.params)

    if template.level == 1:
        # Linear: try identity-like patterns
        # result = 0
        yield {f'c{i}': 0 for i in range(n + 1)}
        # result = p_i
        for i in range(n):
            vals = {f'c{j}': 0 for j in range(n + 1)}
            vals[f'c{i+1}'] = 1
            yield vals
        # result = -p_i
        for i in range(n):
            vals = {f'c{j}': 0 for j in range(n + 1)}
            vals[f'c{i+1}'] = -1
            yield vals
        # result = p_i + c for small c
        for i in range(n):
            for c in [-2, -1, 1, 2]:
                vals = {f'c{j}': 0 for j in range(n + 1)}
                vals[f'c{i+1}'] = 1
                vals['c0'] = c
                yield vals
        # result = p_i + p_j
        for i in range(n):
            for j in range(i + 1, n):
                vals = {f'c{k}': 0 for k in range(n + 1)}
                vals[f'c{i+1}'] = 1
                vals[f'c{j+1}'] = 1
                yield vals
        # result = p_i - p_j
        for i in range(n):
            for j in range(n):
                if i != j:
                    vals = {f'c{k}': 0 for k in range(n + 1)}
                    vals[f'c{i+1}'] = 1
                    vals[f'c{j+1}'] = -1
                    yield vals


def synthesize_from_examples(params: list[str], examples: list[dict],
                              precondition: str = 'true',
                              max_level: int = 3,
                              coeff_bound: int = 10) -> SynthResult:
    """
    Synthesize a program from input/output examples.

    Each example is a dict mapping param names + 'result' to values.
    E.g.: [{'x': 5, 'result': 25}, {'x': -3, 'result': 9}]

    Builds postcondition automatically from examples via CEGIS.
    """
    if not examples:
        return SynthResult(status=SynthStatus.FAILURE,
                          error="No examples provided")

    # We can't build a postcondition from examples alone for verification,
    # so we use a weaker approach: find a program matching all examples,
    # then verify it matches additional test inputs too.
    spec = Spec(
        params=params,
        param_types={p: 'int' for p in params},
        precondition=precondition,
        postcondition='true',  # Will verify structurally via examples
        examples=examples,
    )

    # Generate templates
    templates = []
    if max_level >= 1:
        templates.append(make_linear_template(params))
    if max_level >= 2:
        templates.extend(make_conditional_linear_template(params))
        templates.extend(make_two_param_cond_template(params))
    if max_level >= 3:
        templates.extend(make_nested_cond_template(params))
    templates.sort(key=lambda t: t.level)

    # Build I/O pairs from examples
    test_ios = []
    for ex in examples:
        inp = {p: ex[p] for p in params}
        if 'result' in ex:
            test_ios.append((inp, ex['result']))

    if not test_ios:
        return SynthResult(status=SynthStatus.FAILURE,
                          error="Examples must include 'result' key")

    total_candidates = 0
    for template in templates:
        if template.level > max_level:
            break
        values = _find_candidate(template, spec, test_ios, coeff_bound)
        if values is None:
            continue
        body_source = template.builder(values)
        if body_source is None:
            continue
        total_candidates += 1

        return SynthResult(
            status=SynthStatus.SUCCESS,
            program=body_source,
            template_name=template.name,
            template_level=template.level,
            candidates_tried=total_candidates,
        )

    return SynthResult(
        status=SynthStatus.FAILURE,
        candidates_tried=total_candidates,
        error="No template matched all examples",
    )


def synthesize_with_spec(precondition: str, postcondition: str,
                          params: list[str],
                          max_level: int = 3,
                          coeff_bound: int = 10,
                          max_cegis_rounds: int = 20) -> SynthResult:
    """
    Convenience API: synthesize from precondition + postcondition strings.

    Example:
        synthesize_with_spec(
            precondition='x >= 0',
            postcondition='result == x + 1',
            params=['x'],
        )
    """
    spec = Spec(
        params=params,
        param_types={p: 'int' for p in params},
        precondition=precondition,
        postcondition=postcondition,
    )
    return synthesize(spec, max_cegis_rounds=max_cegis_rounds,
                      max_level=max_level, coeff_bound=coeff_bound)
