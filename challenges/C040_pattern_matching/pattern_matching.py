"""
Pattern Matching for Stack VM Language
Challenge C040 -- AgentZero Session 041

Extends C010 stack VM with pattern matching expressions:

  match (expr) {
    1 => { print("one"); }
    2 | 3 => { print("two or three"); }
    x if x > 10 => { print("big"); }
    (a, b) => { print(a); print(b); }
    _ => { print("other"); }
  }

Pattern types:
  - Literal: int, float, string, bool (exact value match)
  - Wildcard: _ (matches anything, no binding)
  - Variable: name (matches anything, binds to name)
  - Or: p1 | p2 | p3 (matches if any sub-pattern matches)
  - Guard: pattern if condition => body (conditional match)
  - Tuple: (p1, p2, ...) matches tuple values by structure
  - Negative literal: -N in patterns

Tuple literals: (a, b, c) creates a tuple value.

Compilation strategy:
  Scrutinee evaluated once, kept on stack.
  Each arm: DUP, test pattern (consumes DUP, pushes bool),
  JUMP_IF_FALSE to next arm, POP bool, bind vars, execute body,
  JUMP to end. Unmatched: runtime error.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from stack_vm import (
    Op, Chunk, TokenType, Token, KEYWORDS, LexError, ParseError, CompileError,
    VMError, FnObject, CallFrame,
    IntLit, FloatLit, StringLit, BoolLit, Var, UnaryOp, BinOp,
    Assign, LetDecl, Block, IfStmt, WhileStmt, FnDecl, CallExpr,
    ReturnStmt, PrintStmt, Program,
    Compiler as BaseCompiler, VM as BaseVM, Parser as BaseParser,
)
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Extended Token Types
# ============================================================

class ExtTokenType:
    MATCH = 100
    FAT_ARROW = 101   # =>
    UNDERSCORE = 102  # _
    PIPE = 103        # |


# ============================================================
# Pattern AST Nodes
# ============================================================

@dataclass
class LitPattern:
    value: Any
    line: int = 0

@dataclass
class WildcardPattern:
    line: int = 0

@dataclass
class VarPattern:
    name: str
    line: int = 0

@dataclass
class TuplePattern:
    elements: list
    line: int = 0

@dataclass
class OrPattern:
    patterns: list
    line: int = 0

@dataclass
class MatchArm:
    pattern: Any
    guard: Any       # optional guard expression (or None)
    body: Any        # Block or expression
    line: int = 0

@dataclass
class MatchExpr:
    scrutinee: Any
    arms: list
    line: int = 0

@dataclass
class TupleLit:
    elements: list
    line: int = 0


# ============================================================
# Tuple Runtime Value
# ============================================================

@dataclass
class TupleValue:
    elements: tuple

    def __repr__(self):
        inner = ", ".join(repr(e) for e in self.elements)
        return f"({inner})"

    def __eq__(self, other):
        if isinstance(other, TupleValue):
            return self.elements == other.elements
        return NotImplemented

    def __hash__(self):
        return hash(self.elements)


# ============================================================
# Extended Opcodes
# ============================================================

class ExtOp:
    TUPLE = 200       # pop N, pop N elements, push TupleValue
    TUPLE_GET = 201   # pop index, pop tuple, push tuple[index]
    TUPLE_LEN = 202   # pop tuple, push len
    IS_TUPLE = 203    # peek value, push bool (does NOT pop)
    MATCH_FAIL = 204  # pop message, raise VMError
    STRICT_EQ = 205   # pop b, pop a, push (type(a)==type(b) and a==b)


# ============================================================
# Extended Lexer
# ============================================================

def lex_extended(source: str) -> list:
    """Lexer with match, =>, _, | support."""
    tokens = []
    i = 0
    line = 1

    while i < len(source):
        c = source[i]

        if c == '\n':
            line += 1; i += 1
        elif c in ' \t\r':
            i += 1
        elif c == '/' and i + 1 < len(source) and source[i + 1] == '/':
            while i < len(source) and source[i] != '\n':
                i += 1
        elif c.isdigit():
            start = i
            while i < len(source) and source[i].isdigit():
                i += 1
            if i < len(source) and source[i] == '.':
                i += 1
                while i < len(source) and source[i].isdigit():
                    i += 1
                tokens.append(Token(TokenType.FLOAT, float(source[start:i]), line))
            else:
                tokens.append(Token(TokenType.INT, int(source[start:i]), line))
        elif c == '"':
            i += 1; start = i
            while i < len(source) and source[i] != '"':
                if source[i] == '\n': line += 1
                i += 1
            if i >= len(source):
                raise LexError(f"Unterminated string at line {line}")
            tokens.append(Token(TokenType.STRING, source[start:i], line))
            i += 1
        elif c.isalpha() or c == '_':
            start = i
            while i < len(source) and (source[i].isalnum() or source[i] == '_'):
                i += 1
            word = source[start:i]
            if word == 'match':
                tokens.append(Token(ExtTokenType.MATCH, 'match', line))
            elif word == '_':
                tokens.append(Token(ExtTokenType.UNDERSCORE, '_', line))
            else:
                tt = KEYWORDS.get(word, TokenType.IDENT)
                tokens.append(Token(tt, word, line))
        elif c == '=' and i + 1 < len(source) and source[i + 1] == '>':
            tokens.append(Token(ExtTokenType.FAT_ARROW, '=>', line)); i += 2
        elif c == '=' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.EQ, '==', line)); i += 2
        elif c == '!' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.NE, '!=', line)); i += 2
        elif c == '<' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.LE, '<=', line)); i += 2
        elif c == '>' and i + 1 < len(source) and source[i + 1] == '=':
            tokens.append(Token(TokenType.GE, '>=', line)); i += 2
        elif c == '|':
            tokens.append(Token(ExtTokenType.PIPE, '|', line)); i += 1
        elif c == '+': tokens.append(Token(TokenType.PLUS, '+', line)); i += 1
        elif c == '-': tokens.append(Token(TokenType.MINUS, '-', line)); i += 1
        elif c == '*': tokens.append(Token(TokenType.STAR, '*', line)); i += 1
        elif c == '/': tokens.append(Token(TokenType.SLASH, '/', line)); i += 1
        elif c == '%': tokens.append(Token(TokenType.PERCENT, '%', line)); i += 1
        elif c == '<': tokens.append(Token(TokenType.LT, '<', line)); i += 1
        elif c == '>': tokens.append(Token(TokenType.GT, '>', line)); i += 1
        elif c == '=': tokens.append(Token(TokenType.ASSIGN, '=', line)); i += 1
        elif c == '(': tokens.append(Token(TokenType.LPAREN, '(', line)); i += 1
        elif c == ')': tokens.append(Token(TokenType.RPAREN, ')', line)); i += 1
        elif c == '{': tokens.append(Token(TokenType.LBRACE, '{', line)); i += 1
        elif c == '}': tokens.append(Token(TokenType.RBRACE, '}', line)); i += 1
        elif c == ',': tokens.append(Token(TokenType.COMMA, ',', line)); i += 1
        elif c == ';': tokens.append(Token(TokenType.SEMICOLON, ';', line)); i += 1
        else:
            raise LexError(f"Unexpected character '{c}' at line {line}")

    tokens.append(Token(TokenType.EOF, None, line))
    return tokens


# ============================================================
# Extended Parser
# ============================================================

class Parser(BaseParser):

    def statement(self):
        if self.peek().type == ExtTokenType.MATCH:
            expr = self.match_expr()
            # Match as statement: optional semicolon
            if self.peek().type == TokenType.SEMICOLON:
                self.advance()
            return expr
        return super().statement()

    def primary(self):
        tok = self.peek()

        if tok.type == ExtTokenType.MATCH:
            return self.match_expr()

        # Tuple literal: (a, b) vs grouped expression (a)
        if tok.type == TokenType.LPAREN:
            save = self.pos
            self.advance()  # (
            if self.peek().type == TokenType.RPAREN:
                self.advance()
                return TupleLit([], tok.line)
            first = self.expression()
            if self.peek().type == TokenType.COMMA:
                elements = [first]
                while self.peek().type == TokenType.COMMA:
                    self.advance()  # ,
                    if self.peek().type == TokenType.RPAREN:
                        break  # trailing comma
                    elements.append(self.expression())
                self.expect(TokenType.RPAREN)
                return TupleLit(elements, tok.line)
            else:
                self.expect(TokenType.RPAREN)
                return first

        return super().primary()

    def match_expr(self):
        tok = self.advance()  # match
        self.expect(TokenType.LPAREN)
        scrutinee = self.expression()
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.LBRACE)
        arms = []
        while self.peek().type != TokenType.RBRACE:
            arms.append(self.match_arm())
        self.expect(TokenType.RBRACE)
        return MatchExpr(scrutinee=scrutinee, arms=arms, line=tok.line)

    def match_arm(self):
        pattern = self.parse_or_pattern()
        guard = None
        if self.peek().type == TokenType.IF:
            self.advance()
            guard = self.expression()
        t = self.advance()
        if t.type != ExtTokenType.FAT_ARROW:
            raise ParseError(f"Expected '=>', got '{t.value}' at line {t.line}")
        if self.peek().type == TokenType.LBRACE:
            body = self.block()
        else:
            body = self.expression()
            if self.peek().type == TokenType.SEMICOLON:
                self.advance()
        return MatchArm(pattern=pattern, guard=guard, body=body, line=pattern.line)

    def parse_or_pattern(self):
        pat = self.parse_single_pattern()
        if self.peek().type == ExtTokenType.PIPE:
            patterns = [pat]
            while self.peek().type == ExtTokenType.PIPE:
                self.advance()
                patterns.append(self.parse_single_pattern())
            return OrPattern(patterns=patterns, line=pat.line)
        return pat

    def parse_single_pattern(self):
        tok = self.peek()

        if tok.type == ExtTokenType.UNDERSCORE:
            self.advance()
            return WildcardPattern(line=tok.line)

        if tok.type == TokenType.INT:
            self.advance()
            return LitPattern(value=tok.value, line=tok.line)
        if tok.type == TokenType.FLOAT:
            self.advance()
            return LitPattern(value=tok.value, line=tok.line)
        if tok.type == TokenType.STRING:
            self.advance()
            return LitPattern(value=tok.value, line=tok.line)
        if tok.type == TokenType.TRUE:
            self.advance()
            return LitPattern(value=True, line=tok.line)
        if tok.type == TokenType.FALSE:
            self.advance()
            return LitPattern(value=False, line=tok.line)

        # Negative literal
        if tok.type == TokenType.MINUS:
            self.advance()
            num = self.peek()
            if num.type == TokenType.INT:
                self.advance()
                return LitPattern(value=-num.value, line=tok.line)
            if num.type == TokenType.FLOAT:
                self.advance()
                return LitPattern(value=-num.value, line=tok.line)
            raise ParseError(f"Expected number after '-' in pattern at line {tok.line}")

        # Tuple pattern
        if tok.type == TokenType.LPAREN:
            self.advance()
            elements = []
            if self.peek().type != TokenType.RPAREN:
                elements.append(self.parse_or_pattern())
                while self.peek().type == TokenType.COMMA:
                    self.advance()
                    if self.peek().type == TokenType.RPAREN:
                        break
                    elements.append(self.parse_or_pattern())
            self.expect(TokenType.RPAREN)
            return TuplePattern(elements=elements, line=tok.line)

        # Variable pattern
        if tok.type == TokenType.IDENT:
            self.advance()
            return VarPattern(name=tok.value, line=tok.line)

        raise ParseError(f"Expected pattern, got '{tok.value}' at line {tok.line}")


# ============================================================
# Extended Compiler
# ============================================================

class Compiler(BaseCompiler):

    def compile_MatchExpr(self, node):
        """Compile match expression.

        Stack protocol:
        1. Evaluate scrutinee -> [scrutinee]
        2. Store in hidden var __match_N__ (unique per nesting level)
        3. For each arm:
           a. LOAD hidden var
           b. Test pattern (consumes value, pushes bool)
           c. JUMP_IF_FALSE to next_arm
           d. POP bool
           e. Bind pattern variables (LOAD hidden + extract + STORE)
           f. If guard: evaluate guard, JUMP_IF_FALSE unbind_and_next
           g. Compile body
           h. JUMP to end
           i. next_arm: POP bool, continue
        4. After all arms: MATCH_FAIL
        5. end: match result on stack
        """
        # Unique hidden variable name for this match
        match_var = f"__match_{id(node)}__"

        # Evaluate scrutinee and store in hidden var
        self.compile_node(node.scrutinee)
        mv_idx = self.chunk.add_name(match_var)
        self.chunk.emit(Op.STORE, mv_idx, node.line)

        end_jumps = []

        for arm in node.arms:
            # Load scrutinee for pattern test
            self.chunk.emit(Op.LOAD, mv_idx, arm.line)

            # Test pattern -- consumes the value, pushes True/False
            self._compile_pattern_test(arm.pattern, arm.line)

            next_arm = self.chunk.emit(Op.JUMP_IF_FALSE, 0, arm.line)
            self.chunk.emit(Op.POP, line=arm.line)  # pop True

            # Bind pattern variables
            self._compile_bindings(arm.pattern, match_var, arm.line)

            if arm.guard:
                # Evaluate guard
                self.compile_node(arm.guard)
                guard_fail = self.chunk.emit(Op.JUMP_IF_FALSE, 0, arm.line)
                self.chunk.emit(Op.POP, line=arm.line)  # pop True (guard passed)

                # Execute body
                self._compile_match_body(arm.body, arm.line)
                end_jumps.append(self.chunk.emit(Op.JUMP, 0, arm.line))

                # Guard failed
                self.chunk.patch(guard_fail + 1, len(self.chunk.code))
                self.chunk.emit(Op.POP, line=arm.line)  # pop False (guard result)

                # Jump past the next_arm patch point -- go to next arm's test
                skip = self.chunk.emit(Op.JUMP, 0, arm.line)

                # Pattern didn't match
                self.chunk.patch(next_arm + 1, len(self.chunk.code))
                self.chunk.emit(Op.POP, line=arm.line)  # pop False (pattern result)

                # Patch guard-fail skip to here too
                self.chunk.patch(skip + 1, len(self.chunk.code))
            else:
                # No guard -- execute body directly
                self._compile_match_body(arm.body, arm.line)
                end_jumps.append(self.chunk.emit(Op.JUMP, 0, arm.line))

                # Pattern didn't match
                self.chunk.patch(next_arm + 1, len(self.chunk.code))
                self.chunk.emit(Op.POP, line=arm.line)  # pop False

        # No arm matched -- push error and fail
        err_idx = self.chunk.add_constant("Match error: no pattern matched")
        self.chunk.emit(Op.CONST, err_idx, node.line)
        self.chunk.emit(ExtOp.MATCH_FAIL, line=node.line)

        # Patch all end jumps
        end_addr = len(self.chunk.code)
        for ej in end_jumps:
            self.chunk.patch(ej + 1, end_addr)

    def _compile_pattern_test(self, pattern, line):
        """Compile pattern test. Value to test is on top of stack.
        Consumes the value, pushes True or False."""

        if isinstance(pattern, WildcardPattern):
            self.chunk.emit(Op.POP, line=line)  # discard value
            true_idx = self.chunk.add_constant(True)
            self.chunk.emit(Op.CONST, true_idx, line)

        elif isinstance(pattern, LitPattern):
            val_idx = self.chunk.add_constant(pattern.value)
            self.chunk.emit(Op.CONST, val_idx, line)
            self.chunk.emit(ExtOp.STRICT_EQ, line=line)  # type-aware: False!=0, True!=1

        elif isinstance(pattern, VarPattern):
            # Variable always matches
            self.chunk.emit(Op.POP, line=line)  # discard value (binding done later)
            true_idx = self.chunk.add_constant(True)
            self.chunk.emit(Op.CONST, true_idx, line)

        elif isinstance(pattern, OrPattern):
            # Try each sub-pattern. DUP value before each (except last).
            # If non-last matches: [..., val, True]. Jump to success.
            # If non-last fails: [..., val, False]. POP False, try next.
            # Last pattern: consumes val directly, pushes bool.
            success_jumps = []
            for i, sub in enumerate(pattern.patterns):
                is_last = (i == len(pattern.patterns) - 1)
                if not is_last:
                    self.chunk.emit(Op.DUP, line=line)  # [..., val, val_copy]
                self._compile_pattern_test(sub, line)  # consumes val/copy, pushes bool
                if not is_last:
                    sj = self.chunk.emit(Op.JUMP_IF_TRUE, 0, line)
                    success_jumps.append(sj)
                    self.chunk.emit(Op.POP, line=line)  # pop False

            # Last pattern result is on stack: [..., bool]
            # Jump past success cleanup
            skip_success = self.chunk.emit(Op.JUMP, 0, line)

            # Success from non-last: [..., val, True]
            success_addr = len(self.chunk.code)
            for sj in success_jumps:
                self.chunk.patch(sj + 1, success_addr)
            # JUMP_IF_TRUE doesn't pop, so: [..., val, True]
            # Remove val: pop True, pop val, push True
            self.chunk.emit(Op.POP, line=line)   # pop True
            self.chunk.emit(Op.POP, line=line)   # pop val
            true_idx = self.chunk.add_constant(True)
            self.chunk.emit(Op.CONST, true_idx, line)

            self.chunk.patch(skip_success + 1, len(self.chunk.code))

        elif isinstance(pattern, TuplePattern):
            # Check: is tuple? check length? check each element?
            # Value is on top of stack.

            # IS_TUPLE peeks, so we still have value
            self.chunk.emit(ExtOp.IS_TUPLE, line=line)
            not_tuple = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
            self.chunk.emit(Op.POP, line=line)  # pop True from IS_TUPLE

            # Check length
            self.chunk.emit(Op.DUP, line=line)
            self.chunk.emit(ExtOp.TUPLE_LEN, line=line)
            len_idx = self.chunk.add_constant(len(pattern.elements))
            self.chunk.emit(Op.CONST, len_idx, line)
            self.chunk.emit(Op.EQ, line=line)
            wrong_len = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
            self.chunk.emit(Op.POP, line=line)  # pop True

            # Check each element
            elem_fails = []
            for ei, elem_pat in enumerate(pattern.elements):
                self.chunk.emit(Op.DUP, line=line)  # DUP tuple
                ei_idx = self.chunk.add_constant(ei)
                self.chunk.emit(Op.CONST, ei_idx, line)
                self.chunk.emit(ExtOp.TUPLE_GET, line=line)  # element on top
                self._compile_pattern_test(elem_pat, line)
                ef = self.chunk.emit(Op.JUMP_IF_FALSE, 0, line)
                elem_fails.append(ef)
                self.chunk.emit(Op.POP, line=line)  # pop True

            # All matched: pop tuple, push True
            self.chunk.emit(Op.POP, line=line)  # pop tuple
            true_idx = self.chunk.add_constant(True)
            self.chunk.emit(Op.CONST, true_idx, line)
            done = self.chunk.emit(Op.JUMP, 0, line)

            # Failure: pop whatever + tuple, push False
            fail_addr = len(self.chunk.code)
            self.chunk.patch(not_tuple + 1, fail_addr)
            self.chunk.patch(wrong_len + 1, fail_addr)
            for ef in elem_fails:
                self.chunk.patch(ef + 1, fail_addr)

            self.chunk.emit(Op.POP, line=line)  # pop False/value
            self.chunk.emit(Op.POP, line=line)  # pop tuple
            false_idx = self.chunk.add_constant(False)
            self.chunk.emit(Op.CONST, false_idx, line)

            self.chunk.patch(done + 1, len(self.chunk.code))

    def _compile_bindings(self, pattern, match_var, line):
        """Emit code to bind pattern variables from the hidden match var."""
        mv_idx = self.chunk.add_name(match_var)

        if isinstance(pattern, VarPattern):
            self.chunk.emit(Op.LOAD, mv_idx, line)
            var_idx = self.chunk.add_name(pattern.name)
            self.chunk.emit(Op.STORE, var_idx, line)

        elif isinstance(pattern, TuplePattern):
            for ei, elem_pat in enumerate(pattern.elements):
                self._compile_elem_binding(elem_pat, match_var, [ei], line)

        elif isinstance(pattern, OrPattern):
            # Or patterns don't bind variables (ambiguous which matched)
            # unless all alternatives bind the same names (not supported yet)
            pass

    def _compile_elem_binding(self, pattern, match_var, path, line):
        """Bind a variable inside a tuple at the given path of indices."""
        mv_idx = self.chunk.add_name(match_var)

        if isinstance(pattern, VarPattern):
            # Load match var, extract through path, store to var
            self.chunk.emit(Op.LOAD, mv_idx, line)
            for idx in path:
                i_idx = self.chunk.add_constant(idx)
                self.chunk.emit(Op.CONST, i_idx, line)
                self.chunk.emit(ExtOp.TUPLE_GET, line=line)
            var_idx = self.chunk.add_name(pattern.name)
            self.chunk.emit(Op.STORE, var_idx, line)

        elif isinstance(pattern, TuplePattern):
            for ei, elem_pat in enumerate(pattern.elements):
                self._compile_elem_binding(elem_pat, match_var, path + [ei], line)

    def _compile_match_body(self, body, line):
        """Compile arm body. Leaves a value on stack as match result."""
        if isinstance(body, Block):
            if not body.stmts:
                none_idx = self.chunk.add_constant(None)
                self.chunk.emit(Op.CONST, none_idx, line)
            else:
                for stmt in body.stmts:
                    self.compile_node(stmt)
                # Block doesn't produce a value in C010 -- push None
                none_idx = self.chunk.add_constant(None)
                self.chunk.emit(Op.CONST, none_idx, line)
        else:
            self.compile_node(body)

    def compile_FnDecl(self, node):
        """Override to use extended Compiler for function bodies."""
        fn_compiler = Compiler()  # Use OUR Compiler, not BaseCompiler
        for param in node.params:
            fn_compiler.chunk.add_name(param)
        fn_compiler.compile_node(node.body)
        # Implicit return None
        idx = fn_compiler.chunk.add_constant(None)
        fn_compiler.chunk.emit(Op.CONST, idx)
        fn_compiler.chunk.emit(Op.RETURN)

        fn_obj = FnObject(node.name, len(node.params), fn_compiler.chunk)
        self.functions[node.name] = fn_obj
        for k, v in fn_compiler.functions.items():
            self.functions[k] = v

        fn_idx = self.chunk.add_constant(fn_obj)
        name_idx = self.chunk.add_name(node.name)
        self.chunk.emit(Op.CONST, fn_idx, node.line)
        self.chunk.emit(Op.STORE, name_idx, node.line)

    def compile_TupleLit(self, node):
        for elem in node.elements:
            self.compile_node(elem)
        n_idx = self.chunk.add_constant(len(node.elements))
        self.chunk.emit(Op.CONST, n_idx, node.line)
        self.chunk.emit(ExtOp.TUPLE, line=node.line)


# ============================================================
# Extended VM
# ============================================================

class VM(BaseVM):

    def run(self):
        while True:
            self.step_count += 1
            if self.step_count > self.max_steps:
                raise VMError(f"Execution limit exceeded ({self.max_steps} steps)")

            if self.ip >= len(self.current_chunk.code):
                break

            op = self.current_chunk.code[self.ip]
            self.ip += 1

            if self.trace:
                self._trace_op(op)

            # Extended opcodes
            if op == ExtOp.TUPLE:
                n = self.pop()
                elements = []
                for _ in range(n):
                    elements.insert(0, self.pop())
                self.push(TupleValue(tuple(elements)))

            elif op == ExtOp.TUPLE_GET:
                idx = self.pop()
                tup = self.pop()
                if not isinstance(tup, TupleValue):
                    raise VMError(f"TUPLE_GET on non-tuple: {type(tup)}")
                if idx < 0 or idx >= len(tup.elements):
                    raise VMError(f"Tuple index out of range")
                self.push(tup.elements[idx])

            elif op == ExtOp.TUPLE_LEN:
                tup = self.pop()
                if not isinstance(tup, TupleValue):
                    raise VMError(f"TUPLE_LEN on non-tuple")
                self.push(len(tup.elements))

            elif op == ExtOp.IS_TUPLE:
                val = self.peek()
                self.push(isinstance(val, TupleValue))

            elif op == ExtOp.MATCH_FAIL:
                msg = self.pop()
                raise VMError(str(msg))

            elif op == ExtOp.STRICT_EQ:
                b, a = self.pop(), self.pop()
                self.push(type(a) is type(b) and a == b)

            # Standard opcodes (duplicated from base to handle extended ops in same loop)
            elif op == Op.HALT:
                break
            elif op == Op.CONST:
                idx = self.current_chunk.code[self.ip]; self.ip += 1
                self.push(self.current_chunk.constants[idx])
            elif op == Op.POP:
                if self.stack: self.pop()
            elif op == Op.DUP:
                self.push(self.peek())
            elif op == Op.ADD:
                b, a = self.pop(), self.pop(); self.push(a + b)
            elif op == Op.SUB:
                b, a = self.pop(), self.pop(); self.push(a - b)
            elif op == Op.MUL:
                b, a = self.pop(), self.pop(); self.push(a * b)
            elif op == Op.DIV:
                b, a = self.pop(), self.pop()
                if b == 0: raise VMError("Division by zero")
                self.push(a // b if isinstance(a, int) and isinstance(b, int) else a / b)
            elif op == Op.MOD:
                b, a = self.pop(), self.pop()
                if b == 0: raise VMError("Modulo by zero")
                self.push(a % b)
            elif op == Op.NEG:
                self.push(-self.pop())
            elif op == Op.EQ:
                b, a = self.pop(), self.pop(); self.push(a == b)
            elif op == Op.NE:
                b, a = self.pop(), self.pop(); self.push(a != b)
            elif op == Op.LT:
                b, a = self.pop(), self.pop(); self.push(a < b)
            elif op == Op.GT:
                b, a = self.pop(), self.pop(); self.push(a > b)
            elif op == Op.LE:
                b, a = self.pop(), self.pop(); self.push(a <= b)
            elif op == Op.GE:
                b, a = self.pop(), self.pop(); self.push(a >= b)
            elif op == Op.NOT:
                self.push(not self.pop())
            elif op == Op.AND:
                b, a = self.pop(), self.pop(); self.push(a and b)
            elif op == Op.OR:
                b, a = self.pop(), self.pop(); self.push(a or b)
            elif op == Op.LOAD:
                idx = self.current_chunk.code[self.ip]; self.ip += 1
                name = self.current_chunk.names[idx]
                if name not in self.env:
                    raise VMError(f"Undefined variable '{name}'")
                self.push(self.env[name])
            elif op == Op.STORE:
                idx = self.current_chunk.code[self.ip]; self.ip += 1
                name = self.current_chunk.names[idx]
                self.env[name] = self.pop()
            elif op == Op.JUMP:
                self.ip = self.current_chunk.code[self.ip]
            elif op == Op.JUMP_IF_FALSE:
                target = self.current_chunk.code[self.ip]; self.ip += 1
                if not self.peek(): self.ip = target
            elif op == Op.JUMP_IF_TRUE:
                target = self.current_chunk.code[self.ip]; self.ip += 1
                if self.peek(): self.ip = target
            elif op == Op.CALL:
                arg_count = self.current_chunk.code[self.ip]; self.ip += 1
                args = []
                for _ in range(arg_count):
                    args.insert(0, self.pop())
                fn_obj = self.pop()
                if not isinstance(fn_obj, FnObject):
                    raise VMError(f"Cannot call non-function: {fn_obj}")
                if fn_obj.arity != arg_count:
                    raise VMError(f"Function '{fn_obj.name}' expects {fn_obj.arity} args, got {arg_count}")
                frame = CallFrame(self.current_chunk, self.ip, dict(self.env))
                self.call_stack.append(frame)
                self.current_chunk = fn_obj.chunk
                self.ip = 0
                for i, param_name in enumerate(fn_obj.chunk.names[:fn_obj.arity]):
                    self.env[param_name] = args[i]
            elif op == Op.RETURN:
                return_val = self.pop()
                if not self.call_stack:
                    self.push(return_val)
                    break
                frame = self.call_stack.pop()
                self.current_chunk = frame.chunk
                self.ip = frame.ip
                self.env = frame.base_env
                self.push(return_val)
            elif op == Op.PRINT:
                value = self.pop()
                text = self._format_value(value)
                self.output.append(text)
            else:
                raise VMError(f"Unknown opcode: {op}")

        return self.stack[-1] if self.stack else None

    def _format_value(self, value):
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, TupleValue):
            parts = [self._format_value(e) for e in value.elements]
            return "(" + ", ".join(parts) + ")"
        return str(value)


# ============================================================
# Public API
# ============================================================

def parse(source: str):
    tokens = lex_extended(source)
    parser = Parser(tokens)
    return parser.parse()


def compile_source(source: str) -> tuple:
    ast = parse(source)
    compiler = Compiler()
    chunk = compiler.compile(ast)
    return chunk, compiler


def execute(source: str, trace=False) -> dict:
    chunk, compiler = compile_source(source)
    vm = VM(chunk, trace=trace)
    result = vm.run()
    return {
        'result': result,
        'output': vm.output,
        'env': vm.env,
        'steps': vm.step_count,
    }


def run(source: str) -> tuple:
    r = execute(source)
    return r['result'], r['output']
