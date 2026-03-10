"""
C036: Bounded Model Checker
Composes C035 (SAT Solver) + C010 (Stack VM parser)

Verifies safety properties of C010 programs by:
1. Parsing source to AST (reusing C010's parser with assert extension)
2. Encoding program execution as SAT formula via bit-blasting
3. Unrolling loops up to a bound k
4. Checking if assertions can be violated
5. Producing counterexample traces when violations found

Supports: integer arithmetic (bounded bit-width), comparisons,
if/else, while (bounded), assert, variable assignments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C035_sat_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from sat_solver import Solver, SolverResult, Literal
from stack_vm import (
    lex, Parser, TokenType, Token, ParseError,
    IntLit, BoolLit, Var, BinOp, UnaryOp, Assign, LetDecl,
    Block, IfStmt, WhileStmt, PrintStmt, Program, FnDecl, CallExpr, ReturnStmt
)

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ============================================================
# Extended AST node for assert
# ============================================================

@dataclass
class AssertStmt:
    cond: object
    message: str = ""
    line: int = 0


@dataclass
class AssumeStmt:
    cond: object
    line: int = 0


# ============================================================
# Extended Parser (adds assert/assume to C010 parser)
# ============================================================

class ModelParser(Parser):
    """Extends C010 Parser to support assert() and assume() statements."""

    def statement(self):
        if self.peek().type == TokenType.IDENT and self.peek().value == 'assert':
            return self.assert_stmt()
        if self.peek().type == TokenType.IDENT and self.peek().value == 'assume':
            return self.assume_stmt()
        return super().statement()

    def assert_stmt(self):
        tok = self.advance()  # assert
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        msg = ""
        if self.match(TokenType.COMMA):
            msg_tok = self.advance()
            msg = msg_tok.value
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return AssertStmt(cond, msg, tok.line)

    def assume_stmt(self):
        tok = self.advance()  # assume
        self.expect(TokenType.LPAREN)
        cond = self.expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)
        return AssumeStmt(cond, tok.line)


def parse(source):
    """Parse source with assert/assume support."""
    tokens = lex(source)
    parser = ModelParser(tokens)
    return parser.parse()


# ============================================================
# Bit-Vector: a fixed-width integer represented as SAT variables
# ============================================================

class BitVec:
    """A bit-vector represented as a list of SAT variable IDs.
    bit[0] is LSB. Values are signed (two's complement)."""

    def __init__(self, bits, width):
        self.bits = list(bits)  # list of SAT variable IDs (positive int)
        self.width = width
        assert len(self.bits) == width

    def __repr__(self):
        return f"BitVec({self.bits})"


# ============================================================
# SAT Encoder: translates operations to SAT clauses
# ============================================================

class Encoder:
    """Encodes bit-vector operations as SAT clauses."""

    def __init__(self, solver, bit_width=8):
        self.solver = solver
        self.bit_width = bit_width

    def new_bitvec(self):
        """Create a fresh bit-vector with new SAT variables."""
        bits = [self.solver.new_var() for _ in range(self.bit_width)]
        return BitVec(bits, self.bit_width)

    def new_bool_var(self):
        """Create a single boolean SAT variable."""
        return self.solver.new_var()

    def const_bitvec(self, value):
        """Create a bit-vector constrained to a constant value.
        Value is interpreted as signed two's complement."""
        bv = self.new_bitvec()
        # Handle negative values via two's complement
        if value < 0:
            value = value + (1 << self.bit_width)
        value = value & ((1 << self.bit_width) - 1)
        for i in range(self.bit_width):
            bit = (value >> i) & 1
            if bit:
                self.solver.add_clause([bv.bits[i]])
            else:
                self.solver.add_clause([-bv.bits[i]])
        return bv

    def const_bool(self, value):
        """Create a boolean variable constrained to a constant."""
        v = self.new_bool_var()
        if value:
            self.solver.add_clause([v])
        else:
            self.solver.add_clause([-v])
        return v

    # --- Boolean operations on single variables ---

    def bool_and(self, a, b):
        """r = a AND b"""
        r = self.new_bool_var()
        # r -> a, r -> b, a & b -> r
        self.solver.add_clause([-r, a])
        self.solver.add_clause([-r, b])
        self.solver.add_clause([r, -a, -b])
        return r

    def bool_or(self, a, b):
        """r = a OR b"""
        r = self.new_bool_var()
        # r -> a|b, a -> r, b -> r
        self.solver.add_clause([-r, a, b])
        self.solver.add_clause([r, -a])
        self.solver.add_clause([r, -b])
        return r

    def bool_not(self, a):
        """r = NOT a"""
        r = self.new_bool_var()
        self.solver.add_clause([-r, -a])
        self.solver.add_clause([r, a])
        return r

    def bool_xor(self, a, b):
        """r = a XOR b"""
        r = self.new_bool_var()
        self.solver.add_clause([-r, a, b])
        self.solver.add_clause([-r, -a, -b])
        self.solver.add_clause([r, -a, b])
        self.solver.add_clause([r, a, -b])
        return r

    def bool_ite(self, cond, then_v, else_v):
        """r = if cond then then_v else else_v (for single bool vars)"""
        r = self.new_bool_var()
        # cond & then_v -> r
        self.solver.add_clause([-cond, -then_v, r])
        # cond & ~then_v -> ~r
        self.solver.add_clause([-cond, then_v, -r])
        # ~cond & else_v -> r
        self.solver.add_clause([cond, -else_v, r])
        # ~cond & ~else_v -> ~r
        self.solver.add_clause([cond, else_v, -r])
        return r

    # --- Bit-vector arithmetic ---

    def bv_add(self, a, b):
        """Bit-vector addition (mod 2^width)."""
        result = self.new_bitvec()
        carry = self.const_bool(False)
        for i in range(self.bit_width):
            # sum = a[i] XOR b[i] XOR carry
            xor_ab = self.bool_xor(a.bits[i], b.bits[i])
            sum_bit = self.bool_xor(xor_ab, carry)
            # new_carry = (a[i] AND b[i]) OR (carry AND (a[i] XOR b[i]))
            and_ab = self.bool_and(a.bits[i], b.bits[i])
            and_cx = self.bool_and(carry, xor_ab)
            new_carry = self.bool_or(and_ab, and_cx)
            # Constrain result bit
            self.solver.add_clause([-result.bits[i], sum_bit])
            self.solver.add_clause([result.bits[i], -sum_bit])
            carry = new_carry
        return result

    def bv_neg(self, a):
        """Two's complement negation: -a = ~a + 1"""
        inverted = self.bv_not(a)
        one = self.const_bitvec(1)
        return self.bv_add(inverted, one)

    def bv_sub(self, a, b):
        """Bit-vector subtraction: a - b = a + (-b)"""
        neg_b = self.bv_neg(b)
        return self.bv_add(a, neg_b)

    def bv_not(self, a):
        """Bitwise NOT."""
        result = self.new_bitvec()
        for i in range(self.bit_width):
            not_bit = self.bool_not(a.bits[i])
            self.solver.add_clause([-result.bits[i], not_bit])
            self.solver.add_clause([result.bits[i], -not_bit])
        return result

    def bv_and(self, a, b):
        """Bitwise AND."""
        result = self.new_bitvec()
        for i in range(self.bit_width):
            and_bit = self.bool_and(a.bits[i], b.bits[i])
            self.solver.add_clause([-result.bits[i], and_bit])
            self.solver.add_clause([result.bits[i], -and_bit])
        return result

    def bv_mul(self, a, b):
        """Bit-vector multiplication via shift-and-add."""
        # Start with 0
        result = self.const_bitvec(0)
        for i in range(self.bit_width):
            # Shift a left by i: take bits [0..width-1-i] and pad with 0s
            shifted = self.new_bitvec()
            for j in range(self.bit_width):
                if j < i:
                    # Lower bits are 0
                    self.solver.add_clause([-shifted.bits[j]])
                else:
                    # shifted[j] = a[j-i]
                    src = a.bits[j - i]
                    self.solver.add_clause([-shifted.bits[j], src])
                    self.solver.add_clause([shifted.bits[j], -src])

            # Mask: if b[i] is 1, add shifted to result; else add 0
            masked = self.new_bitvec()
            for j in range(self.bit_width):
                m = self.bool_and(shifted.bits[j], b.bits[i])
                self.solver.add_clause([-masked.bits[j], m])
                self.solver.add_clause([masked.bits[j], -m])

            result = self.bv_add(result, masked)
        return result

    def bv_mod(self, a, b):
        """Bit-vector modulo: find q, r such that a = b*q + r, 0 <= r < |b|.
        Simplified: works for small bit widths. Returns remainder."""
        # We encode: there exists q such that a = b*q + r and 0 <= r < b
        q = self.new_bitvec()
        r = self.new_bitvec()
        # a = b*q + r
        bq = self.bv_mul(b, q)
        bq_plus_r = self.bv_add(bq, r)
        self.bv_eq_constraint(a, bq_plus_r)
        # 0 <= r (unsigned) -- r >= 0 is automatic for unsigned
        # r < b (unsigned)
        r_lt_b = self.bv_ult(r, b)
        self.solver.add_clause([r_lt_b])
        return r

    def bv_eq_constraint(self, a, b):
        """Assert that two bit-vectors are equal."""
        for i in range(self.bit_width):
            self.solver.add_clause([-a.bits[i], b.bits[i]])
            self.solver.add_clause([a.bits[i], -b.bits[i]])

    # --- Comparisons (return bool var) ---

    def bv_eq(self, a, b):
        """r = (a == b)"""
        # All bits must match
        bit_eqs = []
        for i in range(self.bit_width):
            eq_i = self.bool_not(self.bool_xor(a.bits[i], b.bits[i]))
            bit_eqs.append(eq_i)
        # AND all bit equalities
        result = bit_eqs[0]
        for i in range(1, len(bit_eqs)):
            result = self.bool_and(result, bit_eqs[i])
        return result

    def bv_ne(self, a, b):
        """r = (a != b)"""
        return self.bool_not(self.bv_eq(a, b))

    def bv_ult(self, a, b):
        """Unsigned less-than: a < b"""
        # Compare from MSB down
        # At the first differing bit, if a has 0 and b has 1, then a < b
        result = self.const_bool(False)
        eq_so_far = self.const_bool(True)
        for i in range(self.bit_width - 1, -1, -1):
            # a[i] < b[i] means ~a[i] & b[i]
            a_lt_b_here = self.bool_and(self.bool_not(a.bits[i]), b.bits[i])
            # If equal so far and a < b at this bit, then result is true
            contributes = self.bool_and(eq_so_far, a_lt_b_here)
            result = self.bool_or(result, contributes)
            # Update eq_so_far: still equal if bits match
            bits_equal = self.bool_not(self.bool_xor(a.bits[i], b.bits[i]))
            eq_so_far = self.bool_and(eq_so_far, bits_equal)
        return result

    def bv_slt(self, a, b):
        """Signed less-than (two's complement)."""
        # For signed: compare sign bits first, then magnitude
        msb = self.bit_width - 1
        # Case 1: a is negative, b is positive -> a < b
        a_neg_b_pos = self.bool_and(a.bits[msb], self.bool_not(b.bits[msb]))
        # Case 2: same sign -> use unsigned comparison on all bits
        same_sign = self.bool_not(self.bool_xor(a.bits[msb], b.bits[msb]))
        unsigned_lt = self.bv_ult(a, b)
        same_and_ult = self.bool_and(same_sign, unsigned_lt)
        return self.bool_or(a_neg_b_pos, same_and_ult)

    def bv_sle(self, a, b):
        """Signed less-than-or-equal."""
        return self.bool_or(self.bv_slt(a, b), self.bv_eq(a, b))

    def bv_sgt(self, a, b):
        """Signed greater-than."""
        return self.bv_slt(b, a)

    def bv_sge(self, a, b):
        """Signed greater-than-or-equal."""
        return self.bv_sle(b, a)

    # --- Conditional assignment ---

    def bv_ite(self, cond, then_bv, else_bv):
        """if cond then then_bv else else_bv (bit-vector)"""
        result = self.new_bitvec()
        for i in range(self.bit_width):
            r = self.bool_ite(cond, then_bv.bits[i], else_bv.bits[i])
            self.solver.add_clause([-result.bits[i], r])
            self.solver.add_clause([result.bits[i], -r])
        return result

    # --- Extract value from SAT model ---

    def extract_value(self, bv, model):
        """Extract integer value from a solved bitvec."""
        val = 0
        for i in range(self.bit_width):
            if model.get(bv.bits[i], False):
                val |= (1 << i)
        # Sign-extend
        if val >= (1 << (self.bit_width - 1)):
            val -= (1 << self.bit_width)
        return val

    def extract_bool(self, var, model):
        """Extract boolean from model."""
        return model.get(var, False)


# ============================================================
# Symbolic State
# ============================================================

class SymbolicState:
    """Tracks symbolic variable assignments using SSA-like naming."""

    def __init__(self, encoder, parent=None):
        self.encoder = encoder
        # var_name -> BitVec (current symbolic value)
        if parent is not None:
            self.vars = dict(parent.vars)
            self.bool_vars = dict(parent.bool_vars)
        else:
            self.vars = {}
            self.bool_vars = {}

    def get(self, name):
        """Get current symbolic bitvec for variable."""
        if name not in self.vars:
            raise ModelCheckError(f"Undefined variable: {name}")
        return self.vars[name]

    def set(self, name, bv):
        """Set symbolic value for variable."""
        self.vars[name] = bv

    def has(self, name):
        return name in self.vars

    def copy(self):
        return SymbolicState(self.encoder, parent=self)


# ============================================================
# Verification Result
# ============================================================

class VerifyResult(Enum):
    SAFE = "SAFE"           # Property holds up to bound
    UNSAFE = "UNSAFE"       # Property violation found
    UNKNOWN = "UNKNOWN"     # Could not determine


@dataclass
class Counterexample:
    """A concrete trace showing how an assertion can be violated."""
    variables: dict = field(default_factory=dict)  # name -> value at violation
    assertion_line: int = 0
    assertion_msg: str = ""
    path: list = field(default_factory=list)  # list of (line, description)


@dataclass
class CheckResult:
    """Result of model checking."""
    verdict: VerifyResult
    assertions_checked: int = 0
    counterexamples: list = field(default_factory=list)  # list of Counterexample
    bound: int = 0
    bit_width: int = 8
    sat_vars: int = 0
    sat_clauses: int = 0


class ModelCheckError(Exception):
    pass


# ============================================================
# Model Checker
# ============================================================

class ModelChecker:
    """Bounded model checker for C010 programs.

    Encodes program execution as SAT via bit-blasting:
    - Each integer variable becomes a bit-vector of SAT variables
    - Arithmetic encoded as circuit (adder, multiplier)
    - Control flow via path condition (ITE muxing)
    - Loops unrolled up to bound k
    - Assertions become SAT queries: is negation satisfiable?
    """

    def __init__(self, bit_width=8, loop_bound=10):
        self.bit_width = bit_width
        self.loop_bound = loop_bound

    def check(self, source):
        """Check all assertions in source code.
        Returns CheckResult."""
        ast = parse(source)
        return self._check_ast(ast)

    def check_property(self, source, prop_source):
        """Check a property (given as expression source) against a program.
        The property is appended as an assertion."""
        combined = source.rstrip()
        if not combined.endswith(';'):
            combined += ';'
        combined += f"\nassert({prop_source});"
        return self.check(combined)

    def _check_ast(self, program):
        """Check all assertions in the AST."""
        # Collect all assertions and their path conditions
        assertions = self._find_assertions(program.stmts)
        if not assertions:
            return CheckResult(
                verdict=VerifyResult.SAFE,
                assertions_checked=0,
                bound=self.loop_bound,
                bit_width=self.bit_width
            )

        result = CheckResult(
            verdict=VerifyResult.SAFE,
            assertions_checked=len(assertions),
            bound=self.loop_bound,
            bit_width=self.bit_width
        )

        # Check each assertion independently
        for assert_info in assertions:
            ce = self._check_single_assertion(program, assert_info)
            if ce is not None:
                result.verdict = VerifyResult.UNSAFE
                result.counterexamples.append(ce)

        return result

    def _find_assertions(self, stmts, depth=0):
        """Find all assertion nodes in the AST."""
        found = []
        for stmt in stmts:
            if isinstance(stmt, AssertStmt):
                found.append(stmt)
            elif isinstance(stmt, Block):
                found.extend(self._find_assertions(stmt.stmts, depth))
            elif isinstance(stmt, IfStmt):
                found.extend(self._find_assertions(
                    [stmt.then_body] if not isinstance(stmt.then_body, Block)
                    else stmt.then_body.stmts, depth))
                if stmt.else_body:
                    if isinstance(stmt.else_body, Block):
                        found.extend(self._find_assertions(stmt.else_body.stmts, depth))
                    elif isinstance(stmt.else_body, IfStmt):
                        found.extend(self._find_assertions([stmt.else_body], depth))
            elif isinstance(stmt, WhileStmt):
                if isinstance(stmt.body, Block):
                    found.extend(self._find_assertions(stmt.body.stmts, depth + 1))
                else:
                    found.extend(self._find_assertions([stmt.body], depth + 1))
        return found

    def _check_single_assertion(self, program, assert_stmt):
        """Check a single assertion. Returns Counterexample or None."""
        solver = Solver()
        encoder = Encoder(solver, self.bit_width)
        state = SymbolicState(encoder)

        # Path condition: conjunction of conditions that must hold for this path
        path_cond = encoder.const_bool(True)

        # Execute program symbolically up to and including the assertion
        path_cond, found = self._symbolic_exec_stmts(
            program.stmts, encoder, state, path_cond, assert_stmt
        )

        if not found:
            # Assertion not reachable on this path encoding
            return None

        # Encode the assertion condition
        cond_var = self._encode_expr(assert_stmt.cond, encoder, state)

        if isinstance(cond_var, BitVec):
            # Nonzero means true
            zero = encoder.const_bitvec(0)
            cond_bool = encoder.bv_ne(cond_var, zero)
        else:
            cond_bool = cond_var

        # We want to check: can (path_cond AND NOT assertion) be satisfied?
        neg_assert = encoder.bool_not(cond_bool)
        violation = encoder.bool_and(path_cond, neg_assert)
        solver.add_clause([violation])

        # Solve
        sat_result = solver.solve()

        if sat_result == SolverResult.SAT:
            model = solver.model()
            ce = Counterexample(
                assertion_line=assert_stmt.line,
                assertion_msg=assert_stmt.message
            )
            # Extract variable values
            for name, bv in state.vars.items():
                ce.variables[name] = encoder.extract_value(bv, model)
            return ce

        return None

    def _symbolic_exec_stmts(self, stmts, encoder, state, path_cond, target_assert):
        """Symbolically execute statements.
        Returns (path_cond, found) tuple."""
        for stmt in stmts:
            if stmt is target_assert:
                return (path_cond, True)
            if self._contains_assert(stmt, target_assert):
                pc, found = self._symbolic_exec_stmt(stmt, encoder, state, path_cond, target_assert)
                if found:
                    return (pc, True)
                path_cond = pc
            else:
                pc, _ = self._symbolic_exec_stmt(stmt, encoder, state, path_cond, target_assert)
                path_cond = pc
        return (path_cond, False)

    def _contains_assert(self, stmt, target):
        """Check if stmt contains the target assertion."""
        if stmt is target:
            return True
        if isinstance(stmt, Block):
            return any(self._contains_assert(s, target) for s in stmt.stmts)
        if isinstance(stmt, IfStmt):
            if self._contains_assert(stmt.then_body, target):
                return True
            if stmt.else_body and self._contains_assert(stmt.else_body, target):
                return True
        if isinstance(stmt, WhileStmt):
            return self._contains_assert(stmt.body, target)
        return False

    def _symbolic_exec_stmt(self, stmt, encoder, state, path_cond, target_assert):
        """Execute a single statement symbolically.
        Returns (path_cond, found) tuple."""
        if isinstance(stmt, LetDecl):
            val = self._encode_expr(stmt.value, encoder, state)
            state.set(stmt.name, val)
            return (path_cond, False)

        if isinstance(stmt, Assign):
            val = self._encode_expr(stmt.value, encoder, state)
            state.set(stmt.name, val)
            return (path_cond, False)

        if isinstance(stmt, Block):
            return self._symbolic_exec_stmts(stmt.stmts, encoder, state, path_cond, target_assert)

        if isinstance(stmt, IfStmt):
            return self._symbolic_exec_if(stmt, encoder, state, path_cond, target_assert)

        if isinstance(stmt, WhileStmt):
            return self._symbolic_exec_while(stmt, encoder, state, path_cond, target_assert)

        if isinstance(stmt, AssumeStmt):
            cond_var = self._encode_expr(stmt.cond, encoder, state)
            if isinstance(cond_var, BitVec):
                zero = encoder.const_bitvec(0)
                cond_bool = encoder.bv_ne(cond_var, zero)
            else:
                cond_bool = cond_var
            return (encoder.bool_and(path_cond, cond_bool), False)

        if isinstance(stmt, PrintStmt):
            return (path_cond, False)

        if isinstance(stmt, AssertStmt):
            # Not the target -- treat as assume (path condition)
            cond_var = self._encode_expr(stmt.cond, encoder, state)
            if isinstance(cond_var, BitVec):
                zero = encoder.const_bitvec(0)
                cond_bool = encoder.bv_ne(cond_var, zero)
            else:
                cond_bool = cond_var
            return (encoder.bool_and(path_cond, cond_bool), False)

        if isinstance(stmt, FnDecl):
            return (path_cond, False)

        # Expression statement (bare expression) -- skip
        if isinstance(stmt, (IntLit, BoolLit, Var, BinOp, UnaryOp, CallExpr)):
            return (path_cond, False)

        return (path_cond, False)

    def _symbolic_exec_if(self, stmt, encoder, state, path_cond, target_assert):
        """Symbolically execute if/else using ITE muxing.
        Returns (path_cond, found) tuple."""
        cond_var = self._encode_expr(stmt.cond, encoder, state)
        if isinstance(cond_var, BitVec):
            zero = encoder.const_bitvec(0)
            cond_bool = encoder.bv_ne(cond_var, zero)
        else:
            cond_bool = cond_var

        # Check if target is in then or else branch
        target_in_then = self._contains_assert(stmt.then_body, target_assert)
        target_in_else = stmt.else_body and self._contains_assert(stmt.else_body, target_assert)

        if target_in_then:
            then_pc = encoder.bool_and(path_cond, cond_bool)
            then_state = state.copy()
            result_pc, found = self._symbolic_exec_stmt(
                stmt.then_body, encoder, then_state, then_pc, target_assert
            )
            for name, bv in then_state.vars.items():
                state.vars[name] = bv
            return (result_pc, found)

        if target_in_else:
            then_state = state.copy()
            then_pc = encoder.bool_and(path_cond, cond_bool)
            self._symbolic_exec_stmt(stmt.then_body, encoder, then_state, then_pc, target_assert)

            else_pc = encoder.bool_and(path_cond, encoder.bool_not(cond_bool))
            else_state = state.copy()
            result_pc, found = self._symbolic_exec_stmt(
                stmt.else_body, encoder, else_state, else_pc, target_assert
            )
            for name, bv in else_state.vars.items():
                state.vars[name] = bv
            return (result_pc, found)

        # Target not in either branch -- execute both and merge
        then_state = state.copy()
        then_body = stmt.then_body
        if isinstance(then_body, Block):
            for s in then_body.stmts:
                self._symbolic_exec_stmt(s, encoder, then_state, path_cond, target_assert)
        else:
            self._symbolic_exec_stmt(then_body, encoder, then_state, path_cond, target_assert)

        else_state = state.copy()
        if stmt.else_body:
            else_body = stmt.else_body
            if isinstance(else_body, Block):
                for s in else_body.stmts:
                    self._symbolic_exec_stmt(s, encoder, else_state, path_cond, target_assert)
            else:
                self._symbolic_exec_stmt(else_body, encoder, else_state, path_cond, target_assert)

        # Merge: for each variable modified in either branch, use ITE
        all_names = set(then_state.vars.keys()) | set(else_state.vars.keys())
        for name in all_names:
            then_val = then_state.vars.get(name)
            else_val = else_state.vars.get(name)
            if then_val is None and else_val is not None:
                state.vars[name] = else_val
            elif else_val is None and then_val is not None:
                state.vars[name] = then_val
            elif then_val is not None and else_val is not None:
                if isinstance(then_val, BitVec) and isinstance(else_val, BitVec):
                    merged = encoder.bv_ite(cond_bool, then_val, else_val)
                    state.vars[name] = merged
                elif isinstance(then_val, int) and isinstance(else_val, int):
                    merged = encoder.bool_ite(cond_bool, then_val, else_val)
                    state.vars[name] = merged
                else:
                    state.vars[name] = then_val

        return (path_cond, False)

    def _symbolic_exec_while(self, stmt, encoder, state, path_cond, target_assert):
        """Unroll while loop up to bound k.
        Returns (path_cond, found) tuple."""
        target_in_body = self._contains_assert(stmt.body, target_assert)

        for iteration in range(self.loop_bound):
            cond_var = self._encode_expr(stmt.cond, encoder, state)
            if isinstance(cond_var, BitVec):
                zero = encoder.const_bitvec(0)
                cond_bool = encoder.bv_ne(cond_var, zero)
            else:
                cond_bool = cond_var

            if target_in_body:
                iter_state = state.copy()
                iter_pc = encoder.bool_and(path_cond, cond_bool)
                result_pc, found = self._symbolic_exec_stmt(
                    stmt.body, encoder, iter_state, iter_pc, target_assert
                )
                if found:
                    for name, bv in iter_state.vars.items():
                        state.vars[name] = bv
                    return (result_pc, True)

            # Execute loop body (for state updates)
            body_state = state.copy()
            if isinstance(stmt.body, Block):
                for s in stmt.body.stmts:
                    self._symbolic_exec_stmt(s, encoder, body_state, path_cond, target_assert)
            else:
                self._symbolic_exec_stmt(stmt.body, encoder, body_state, path_cond, target_assert)

            # Merge: if cond, use body_state; else keep state
            all_names = set(body_state.vars.keys()) | set(state.vars.keys())
            for name in all_names:
                body_val = body_state.vars.get(name)
                orig_val = state.vars.get(name)
                if body_val is not None and orig_val is not None:
                    if isinstance(body_val, BitVec) and isinstance(orig_val, BitVec):
                        merged = encoder.bv_ite(cond_bool, body_val, orig_val)
                        state.vars[name] = merged
                    elif isinstance(body_val, int) and isinstance(orig_val, int):
                        merged = encoder.bool_ite(cond_bool, body_val, orig_val)
                        state.vars[name] = merged
                    else:
                        state.vars[name] = body_val
                elif body_val is not None:
                    state.vars[name] = body_val

        return (path_cond, False)

    def _encode_expr(self, expr, encoder, state):
        """Encode an expression as a BitVec or bool var."""
        if isinstance(expr, IntLit):
            return encoder.const_bitvec(expr.value)

        if isinstance(expr, BoolLit):
            if expr.value:
                return encoder.const_bitvec(1)
            else:
                return encoder.const_bitvec(0)

        if isinstance(expr, Var):
            if not state.has(expr.name):
                # Unconstrained input variable
                bv = encoder.new_bitvec()
                state.set(expr.name, bv)
            return state.get(expr.name)

        if isinstance(expr, UnaryOp):
            operand = self._encode_expr(expr.operand, encoder, state)
            if expr.op == '-':
                if isinstance(operand, BitVec):
                    return encoder.bv_neg(operand)
            if expr.op == 'not':
                if isinstance(operand, BitVec):
                    zero = encoder.const_bitvec(0)
                    is_zero = encoder.bv_eq(operand, zero)
                    return encoder.bv_ite(is_zero, encoder.const_bitvec(1), encoder.const_bitvec(0))
                else:
                    not_val = encoder.bool_not(operand)
                    return encoder.bv_ite(not_val, encoder.const_bitvec(1), encoder.const_bitvec(0))
            raise ModelCheckError(f"Unsupported unary op: {expr.op}")

        if isinstance(expr, BinOp):
            return self._encode_binop(expr, encoder, state)

        if isinstance(expr, Assign):
            val = self._encode_expr(expr.value, encoder, state)
            state.set(expr.name, val)
            return val

        raise ModelCheckError(f"Unsupported expression type: {type(expr).__name__}")

    def _encode_binop(self, expr, encoder, state):
        """Encode a binary operation."""
        left = self._encode_expr(expr.left, encoder, state)
        right = self._encode_expr(expr.right, encoder, state)

        # Ensure both are BitVec for arithmetic
        if isinstance(left, int) and isinstance(right, int):
            # Both are raw bool vars -- convert to bitvec
            left = encoder.bv_ite(left, encoder.const_bitvec(1), encoder.const_bitvec(0))
            right = encoder.bv_ite(right, encoder.const_bitvec(1), encoder.const_bitvec(0))
        elif isinstance(left, int):
            left = encoder.bv_ite(left, encoder.const_bitvec(1), encoder.const_bitvec(0))
        elif isinstance(right, int):
            right = encoder.bv_ite(right, encoder.const_bitvec(1), encoder.const_bitvec(0))

        op = expr.op
        if op == '+':
            return encoder.bv_add(left, right)
        if op == '-':
            return encoder.bv_sub(left, right)
        if op == '*':
            return encoder.bv_mul(left, right)
        if op == '%':
            return encoder.bv_mod(left, right)

        # Comparisons return bitvec (0 or 1) for consistency
        if op == '==':
            eq = encoder.bv_eq(left, right)
            return encoder.bv_ite(eq, encoder.const_bitvec(1), encoder.const_bitvec(0))
        if op == '!=':
            ne = encoder.bv_ne(left, right)
            return encoder.bv_ite(ne, encoder.const_bitvec(1), encoder.const_bitvec(0))
        if op == '<':
            lt = encoder.bv_slt(left, right)
            return encoder.bv_ite(lt, encoder.const_bitvec(1), encoder.const_bitvec(0))
        if op == '>':
            gt = encoder.bv_sgt(left, right)
            return encoder.bv_ite(gt, encoder.const_bitvec(1), encoder.const_bitvec(0))
        if op == '<=':
            le = encoder.bv_sle(left, right)
            return encoder.bv_ite(le, encoder.const_bitvec(1), encoder.const_bitvec(0))
        if op == '>=':
            ge = encoder.bv_sge(left, right)
            return encoder.bv_ite(ge, encoder.const_bitvec(1), encoder.const_bitvec(0))

        # Logical
        if op == 'and':
            # a and b: both nonzero
            zero = encoder.const_bitvec(0)
            a_nz = encoder.bv_ne(left, zero)
            b_nz = encoder.bv_ne(right, zero)
            both = encoder.bool_and(a_nz, b_nz)
            return encoder.bv_ite(both, encoder.const_bitvec(1), encoder.const_bitvec(0))
        if op == 'or':
            zero = encoder.const_bitvec(0)
            a_nz = encoder.bv_ne(left, zero)
            b_nz = encoder.bv_ne(right, zero)
            either = encoder.bool_or(a_nz, b_nz)
            return encoder.bv_ite(either, encoder.const_bitvec(1), encoder.const_bitvec(0))

        raise ModelCheckError(f"Unsupported binary op: {op}")


# ============================================================
# Convenience functions
# ============================================================

def check(source, bit_width=8, loop_bound=10):
    """Check all assertions in source code."""
    mc = ModelChecker(bit_width=bit_width, loop_bound=loop_bound)
    return mc.check(source)


def check_property(source, prop, bit_width=8, loop_bound=10):
    """Check a property against source code."""
    mc = ModelChecker(bit_width=bit_width, loop_bound=loop_bound)
    return mc.check_property(source, prop)


def verify_safe(source, bit_width=8, loop_bound=10):
    """Returns True if all assertions hold up to bound."""
    result = check(source, bit_width=bit_width, loop_bound=loop_bound)
    return result.verdict == VerifyResult.SAFE


def find_bug(source, bit_width=8, loop_bound=10):
    """Returns first counterexample or None if safe."""
    result = check(source, bit_width=bit_width, loop_bound=loop_bound)
    if result.counterexamples:
        return result.counterexamples[0]
    return None
