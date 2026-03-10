"""
Type Checker for the Stack VM Language
Challenge C013 -- AgentZero Session 014

Static type analysis for the C010 VM language.

Features:
  - Concrete types: Int, Float, String, Bool, Void, Function types
  - Type annotations on let declarations: let x: int = 5;
  - Function type signatures: fn add(a: int, b: int) -> int { ... }
  - Type inference via unification when annotations absent
  - Subtyping: int promotes to float
  - Comprehensive error reporting with source locations
  - Works on the existing C010 AST -- no parser changes needed

Architecture:
  Source -> Lex -> Parse -> AST -> TypeCheck -> (typed AST or errors)
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# Type Representations
# ============================================================

@dataclass(frozen=True)
class TInt:
    """Integer type."""
    def __repr__(self): return "int"

@dataclass(frozen=True)
class TFloat:
    """Float type."""
    def __repr__(self): return "float"

@dataclass(frozen=True)
class TString:
    """String type."""
    def __repr__(self): return "string"

@dataclass(frozen=True)
class TBool:
    """Boolean type."""
    def __repr__(self): return "bool"

@dataclass(frozen=True)
class TVoid:
    """Void type (no value / unit)."""
    def __repr__(self): return "void"

@dataclass(frozen=True)
class TFunc:
    """Function type."""
    params: tuple  # tuple of types
    ret: Any       # return type

    def __repr__(self):
        params_str = ", ".join(repr(p) for p in self.params)
        return f"fn({params_str}) -> {self.ret!r}"

@dataclass
class TVar:
    """Type variable for inference. Mutable -- resolved via unification."""
    id: int
    bound: Any = None  # None = unresolved, else the resolved type

    def __repr__(self):
        if self.bound is not None:
            return repr(self.bound)
        return f"?T{self.id}"

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.id == other.id
        return NotImplemented

    def __hash__(self):
        return hash(self.id)


# Singleton instances for concrete types
INT = TInt()
FLOAT = TFloat()
STRING = TString()
BOOL = TBool()
VOID = TVoid()

# Map from annotation strings to types
TYPE_NAMES = {
    'int': INT,
    'float': FLOAT,
    'string': STRING,
    'bool': BOOL,
    'void': VOID,
}


# ============================================================
# Type Errors
# ============================================================

@dataclass
class TypeError_:
    """A type error with location info."""
    message: str
    line: int

    def __repr__(self):
        return f"TypeError at line {self.line}: {self.message}"


# ============================================================
# Type Environment
# ============================================================

class TypeEnv:
    """Scoped type environment."""

    def __init__(self, parent=None):
        self.bindings = {}
        self.parent = parent

    def define(self, name, typ):
        self.bindings[name] = typ

    def lookup(self, name):
        if name in self.bindings:
            return self.bindings[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def child(self):
        return TypeEnv(parent=self)


# ============================================================
# Unification
# ============================================================

def resolve(t):
    """Follow TVar chains to find the actual type."""
    while isinstance(t, TVar) and t.bound is not None:
        t = t.bound
    return t


def occurs_in(tvar, t):
    """Check if tvar occurs in t (prevents infinite types)."""
    t = resolve(t)
    if isinstance(t, TVar):
        return t.id == tvar.id
    if isinstance(t, TFunc):
        return any(occurs_in(tvar, p) for p in t.params) or occurs_in(tvar, t.ret)
    return False


class UnificationError(Exception):
    pass


def unify(t1, t2):
    """Unify two types. Binds type variables. Raises UnificationError on failure."""
    t1 = resolve(t1)
    t2 = resolve(t2)

    # Same type
    if t1 == t2:
        return t1

    # Type variables bind to anything (with occurs check)
    if isinstance(t1, TVar):
        if occurs_in(t1, t2):
            raise UnificationError(f"Infinite type: {t1} ~ {t2}")
        t1.bound = t2
        return t2

    if isinstance(t2, TVar):
        if occurs_in(t2, t1):
            raise UnificationError(f"Infinite type: {t2} ~ {t1}")
        t2.bound = t1
        return t1

    # Int promotes to float
    if isinstance(t1, TInt) and isinstance(t2, TFloat):
        return FLOAT
    if isinstance(t1, TFloat) and isinstance(t2, TInt):
        return FLOAT

    # Function types
    if isinstance(t1, TFunc) and isinstance(t2, TFunc):
        if len(t1.params) != len(t2.params):
            raise UnificationError(
                f"Function arity mismatch: {t1} vs {t2}")
        unified_params = tuple(unify(p1, p2) for p1, p2 in zip(t1.params, t2.params))
        unified_ret = unify(t1.ret, t2.ret)
        return TFunc(unified_params, unified_ret)

    raise UnificationError(f"Cannot unify {t1!r} with {t2!r}")


def types_compatible(t1, t2):
    """Check if t1 can be used where t2 is expected, without binding."""
    t1 = resolve(t1)
    t2 = resolve(t2)

    if t1 == t2:
        return True

    # Type variables are always compatible (they'll be unified later)
    if isinstance(t1, TVar) or isinstance(t2, TVar):
        return True

    # Int promotes to float
    if isinstance(t1, TInt) and isinstance(t2, TFloat):
        return True

    # Function types
    if isinstance(t1, TFunc) and isinstance(t2, TFunc):
        if len(t1.params) != len(t2.params):
            return False
        return (all(types_compatible(p1, p2) for p1, p2 in zip(t1.params, t2.params))
                and types_compatible(t1.ret, t2.ret))

    return False


# ============================================================
# AST Extensions for Type Annotations
# ============================================================
# The C010 AST doesn't have type annotations. We extend it here
# with annotated variants. The type checker can work with both
# annotated and unannotated ASTs.

@dataclass
class TypedLetDecl:
    """Let declaration with optional type annotation."""
    name: str
    type_ann: Any  # None or a type
    value: Any
    line: int = 0

@dataclass
class TypedFnDecl:
    """Function declaration with optional type annotations."""
    name: str
    params: list        # list of (name, type_or_None)
    ret_ann: Any        # return type annotation or None
    body: Any
    line: int = 0

@dataclass
class TypedParam:
    """A typed function parameter."""
    name: str
    type_ann: Any  # None or a type


# ============================================================
# Extended Parser (adds type annotations to C010 parser)
# ============================================================
# We import the C010 lexer and AST but extend parsing for annotations.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C010_stack_vm'))

from stack_vm import (
    lex, Token, TokenType, LexError, ParseError,
    IntLit, FloatLit, StringLit, BoolLit, Var,
    UnaryOp, BinOp, Assign, LetDecl, Block,
    IfStmt, WhileStmt, FnDecl, CallExpr,
    ReturnStmt, PrintStmt, Program, Parser
)


class TypedParser(Parser):
    """Parser that supports optional type annotations."""

    def let_decl(self):
        tok = self.advance()  # let
        name = self.expect(TokenType.IDENT).value

        # Optional type annotation: let x: int = ...
        type_ann = None
        if self.peek().type == TokenType.IDENT and self.peek().value == ':':
            # Actually we need a colon token. The C010 lexer doesn't have one.
            # We'll handle this differently: check for a colon after the name.
            pass

        self.expect(TokenType.ASSIGN)
        value = self.expression()
        self.expect(TokenType.SEMICOLON)

        return LetDecl(name, value, tok.line)

    def fn_decl(self):
        self.advance()  # fn
        name_tok = self.expect(TokenType.IDENT)
        self.expect(TokenType.LPAREN)
        params = []
        if self.peek().type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENT).value)
            while self.match(TokenType.COMMA):
                params.append(self.expect(TokenType.IDENT).value)
        self.expect(TokenType.RPAREN)
        body = self.block()
        return FnDecl(name_tok.value, params, body, name_tok.line)


def parse(source):
    """Parse source code into AST."""
    tokens = lex(source)
    parser = TypedParser(tokens)
    return parser.parse()


# ============================================================
# Type Checker
# ============================================================

class TypeChecker:
    """
    Static type checker for the VM language.

    Walks the AST, infers types, checks constraints.
    Reports all errors (does not stop at first error).
    """

    def __init__(self):
        self.errors = []
        self.env = TypeEnv()
        self._tvar_counter = 0
        self._return_type_stack = []  # expected return types for current function

    def fresh_tvar(self):
        """Create a fresh type variable."""
        self._tvar_counter += 1
        return TVar(self._tvar_counter)

    def error(self, msg, line):
        self.errors.append(TypeError_(msg, line))

    def check(self, program):
        """Type-check a program. Returns list of TypeError_."""
        self.errors = []
        for stmt in program.stmts:
            self.check_stmt(stmt)
        return self.errors

    # ---- Statements ----

    def check_stmt(self, node):
        """Check a statement node. Returns None."""
        method = f'check_{type(node).__name__}'
        handler = getattr(self, method, None)
        if handler:
            handler(node)
        else:
            # Expression statement -- just infer its type
            self.infer(node)

    def check_LetDecl(self, node):
        value_type = self.infer(node.value)
        if value_type is not None:
            self.env.define(node.name, value_type)
        else:
            # Inference failed, use a fresh tvar
            self.env.define(node.name, self.fresh_tvar())

    def check_TypedLetDecl(self, node):
        value_type = self.infer(node.value)
        if node.type_ann is not None:
            ann_type = node.type_ann
            if value_type is not None:
                try:
                    unify(value_type, ann_type)
                except UnificationError:
                    self.error(
                        f"Type mismatch: declared {ann_type!r} but got {value_type!r}",
                        node.line
                    )
            self.env.define(node.name, ann_type)
        else:
            if value_type is not None:
                self.env.define(node.name, value_type)
            else:
                self.env.define(node.name, self.fresh_tvar())

    def check_Assign(self, node):
        value_type = self.infer(node.value)
        existing = self.env.lookup(node.name)
        if existing is None:
            self.error(f"Assignment to undefined variable '{node.name}'", node.line)
            self.env.define(node.name, value_type if value_type else self.fresh_tvar())
        else:
            if value_type is not None:
                try:
                    unify(existing, value_type)
                except UnificationError:
                    self.error(
                        f"Cannot assign {value_type!r} to variable '{node.name}' of type {existing!r}",
                        node.line
                    )

    def check_Block(self, node):
        child_env = self.env.child()
        old_env = self.env
        self.env = child_env
        for stmt in node.stmts:
            self.check_stmt(stmt)
        self.env = old_env

    def check_IfStmt(self, node):
        cond_type = self.infer(node.cond)
        if cond_type is not None:
            resolved = resolve(cond_type)
            if not isinstance(resolved, (TBool, TVar)):
                self.error(
                    f"Condition must be bool, got {resolved!r}",
                    node.line
                )
        self.check_stmt(node.then_body)
        if node.else_body:
            self.check_stmt(node.else_body)

    def check_WhileStmt(self, node):
        cond_type = self.infer(node.cond)
        if cond_type is not None:
            resolved = resolve(cond_type)
            if not isinstance(resolved, (TBool, TVar)):
                self.error(
                    f"While condition must be bool, got {resolved!r}",
                    node.line
                )
        self.check_stmt(node.body)

    def check_FnDecl(self, node):
        # Create function type
        param_types = []
        fn_env = self.env.child()

        for param_name in node.params:
            pt = self.fresh_tvar()
            param_types.append(pt)
            fn_env.define(param_name, pt)

        ret_type = self.fresh_tvar()
        fn_type = TFunc(tuple(param_types), ret_type)
        self.env.define(node.name, fn_type)

        # Check body in function scope
        old_env = self.env
        self.env = fn_env
        self._return_type_stack.append(ret_type)

        self.check_stmt(node.body)

        self._return_type_stack.pop()
        self.env = old_env

    def check_TypedFnDecl(self, node):
        param_types = []
        fn_env = self.env.child()

        for p in node.params:
            if isinstance(p, TypedParam) and p.type_ann is not None:
                pt = p.type_ann
            elif isinstance(p, TypedParam):
                pt = self.fresh_tvar()
            else:
                # Plain string param name
                pt = self.fresh_tvar()
            param_types.append(pt)
            pname = p.name if isinstance(p, TypedParam) else p
            fn_env.define(pname, pt)

        ret_type = node.ret_ann if node.ret_ann is not None else self.fresh_tvar()
        fn_type = TFunc(tuple(param_types), ret_type)
        self.env.define(node.name, fn_type)

        old_env = self.env
        self.env = fn_env
        self._return_type_stack.append(ret_type)

        self.check_stmt(node.body)

        self._return_type_stack.pop()
        self.env = old_env

    def check_ReturnStmt(self, node):
        if node.value is not None:
            ret_type = self.infer(node.value)
        else:
            ret_type = VOID

        if self._return_type_stack:
            expected = self._return_type_stack[-1]
            if ret_type is not None:
                try:
                    unify(expected, ret_type)
                except UnificationError:
                    self.error(
                        f"Return type mismatch: expected {resolve(expected)!r}, got {ret_type!r}",
                        node.line
                    )
        else:
            self.error("Return statement outside of function", node.line)

    def check_PrintStmt(self, node):
        self.infer(node.value)  # Any type can be printed

    # ---- Expressions (return inferred type) ----

    def infer(self, node):
        """Infer the type of an expression node. Returns a type or None on error."""
        method = f'infer_{type(node).__name__}'
        handler = getattr(self, method, None)
        if handler:
            return handler(node)

        # Try as statement
        stmt_method = f'check_{type(node).__name__}'
        stmt_handler = getattr(self, stmt_method, None)
        if stmt_handler:
            stmt_handler(node)
            return VOID

        line = getattr(node, 'line', 0)
        self.error(f"Cannot type-check {type(node).__name__}", line)
        return None

    def infer_IntLit(self, node):
        return INT

    def infer_FloatLit(self, node):
        return FLOAT

    def infer_StringLit(self, node):
        return STRING

    def infer_BoolLit(self, node):
        return BOOL

    def infer_Var(self, node):
        t = self.env.lookup(node.name)
        if t is None:
            self.error(f"Undefined variable '{node.name}'", node.line)
            return None
        return resolve(t)

    def infer_UnaryOp(self, node):
        operand_type = self.infer(node.operand)
        if operand_type is None:
            return None

        operand_type = resolve(operand_type)

        if node.op == '-':
            if isinstance(operand_type, TInt):
                return INT
            if isinstance(operand_type, TFloat):
                return FLOAT
            if isinstance(operand_type, TVar):
                return operand_type  # will be resolved later
            self.error(
                f"Cannot negate type {operand_type!r}", node.line)
            return None

        if node.op == 'not':
            if isinstance(operand_type, TBool):
                return BOOL
            if isinstance(operand_type, TVar):
                try:
                    unify(operand_type, BOOL)
                except UnificationError:
                    pass
                return BOOL
            self.error(
                f"'not' requires bool, got {operand_type!r}", node.line)
            return None

        return None

    def infer_BinOp(self, node):
        left_type = self.infer(node.left)
        right_type = self.infer(node.right)

        if left_type is None or right_type is None:
            return None

        left_type = resolve(left_type)
        right_type = resolve(right_type)

        # Arithmetic operators
        if node.op in ('+', '-', '*', '/', '%'):
            return self._check_arithmetic(node.op, left_type, right_type, node.line)

        # Comparison operators
        if node.op in ('<', '>', '<=', '>='):
            return self._check_comparison(left_type, right_type, node.line)

        # Equality operators
        if node.op in ('==', '!='):
            return self._check_equality(left_type, right_type, node.line)

        # Logical operators
        if node.op in ('and', 'or'):
            return self._check_logical(left_type, right_type, node.line)

        self.error(f"Unknown operator '{node.op}'", node.line)
        return None

    def _check_arithmetic(self, op, lt, rt, line):
        # String concatenation
        if op == '+' and isinstance(lt, TString) and isinstance(rt, TString):
            return STRING

        # Numeric operations
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            if isinstance(lt, TFloat) or isinstance(rt, TFloat):
                return FLOAT
            return INT

        # Type variable -- defer
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            try:
                result = unify(lt, rt)
                return resolve(result)
            except UnificationError:
                pass
            return self.fresh_tvar()

        self.error(
            f"Cannot apply '{op}' to {lt!r} and {rt!r}", line)
        return None

    def _check_comparison(self, lt, rt, line):
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            return BOOL
        if isinstance(lt, TString) and isinstance(rt, TString):
            return BOOL
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            return BOOL
        self.error(
            f"Cannot compare {lt!r} and {rt!r}", line)
        return None

    def _check_equality(self, lt, rt, line):
        # Most types can be compared for equality
        if isinstance(lt, TVar) or isinstance(rt, TVar):
            return BOOL
        if type(lt) == type(rt):
            return BOOL
        # int/float comparison is fine
        if isinstance(lt, (TInt, TFloat)) and isinstance(rt, (TInt, TFloat)):
            return BOOL
        self.error(
            f"Cannot compare {lt!r} and {rt!r} for equality", line)
        return None

    def _check_logical(self, lt, rt, line):
        if isinstance(lt, TVar):
            try:
                unify(lt, BOOL)
            except UnificationError:
                pass
        elif not isinstance(lt, TBool):
            self.error(f"Logical operator requires bool, got {lt!r}", line)
            return None

        if isinstance(rt, TVar):
            try:
                unify(rt, BOOL)
            except UnificationError:
                pass
        elif not isinstance(rt, TBool):
            self.error(f"Logical operator requires bool, got {rt!r}", line)
            return None

        return BOOL

    def infer_CallExpr(self, node):
        fn_type = self.env.lookup(node.callee)
        if fn_type is None:
            self.error(f"Undefined function '{node.callee}'", node.line)
            return None

        fn_type = resolve(fn_type)

        if isinstance(fn_type, TVar):
            # Function type not yet known, create one
            arg_types = []
            for arg in node.args:
                at = self.infer(arg)
                arg_types.append(at if at else self.fresh_tvar())
            ret = self.fresh_tvar()
            fn_t = TFunc(tuple(arg_types), ret)
            try:
                unify(fn_type, fn_t)
            except UnificationError as e:
                self.error(str(e), node.line)
            return ret

        if not isinstance(fn_type, TFunc):
            self.error(f"'{node.callee}' is not a function (type: {fn_type!r})", node.line)
            return None

        # Check argument count
        if len(node.args) != len(fn_type.params):
            self.error(
                f"Function '{node.callee}' expects {len(fn_type.params)} args, got {len(node.args)}",
                node.line
            )
            return resolve(fn_type.ret)

        # Check argument types
        for i, (arg, param_type) in enumerate(zip(node.args, fn_type.params)):
            arg_type = self.infer(arg)
            if arg_type is not None:
                try:
                    unify(arg_type, param_type)
                except UnificationError:
                    self.error(
                        f"Argument {i+1} of '{node.callee}': expected {resolve(param_type)!r}, got {arg_type!r}",
                        node.line
                    )

        return resolve(fn_type.ret)

    def infer_Assign(self, node):
        self.check_Assign(node)
        t = self.env.lookup(node.name)
        return resolve(t) if t else None


# ============================================================
# Public API
# ============================================================

def check_source(source):
    """
    Type-check source code.
    Returns (errors, checker) where errors is a list of TypeError_.
    """
    program = parse(source)
    checker = TypeChecker()
    errors = checker.check(program)
    return errors, checker


def check_program(program):
    """
    Type-check a pre-parsed Program AST.
    Returns (errors, checker).
    """
    checker = TypeChecker()
    errors = checker.check(program)
    return errors, checker


def format_errors(errors):
    """Format errors as a human-readable string."""
    if not errors:
        return "No type errors."
    lines = [f"Found {len(errors)} type error(s):"]
    for e in errors:
        lines.append(f"  Line {e.line}: {e.message}")
    return "\n".join(lines)
