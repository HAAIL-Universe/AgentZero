"""
C095: Logic Programming Engine composing C094 (Constraint Solver).

Prolog-style logic programming with:
- Terms: atoms, numbers, variables, compound terms (functors)
- Unification with occurs check
- SLD resolution with depth-first search and backtracking
- Cut (!) for pruning search
- Negation as failure (\\+)
- Built-in predicates: arithmetic, comparison, I/O, list ops, meta
- Definite clause grammars (DCG) via term expansion
- Assert/retract for dynamic predicates
- Constraint Logic Programming (CLP) via C094 integration
- findall/bagof/setof for collecting solutions
- Operator definitions and arithmetic evaluation
"""

import sys
import os
from enum import Enum
from collections import OrderedDict
from copy import deepcopy

# Compose C094
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C094_constraint_solver'))
from constraint_solver import CSPSolver, Variable as CSPVariable, CSPResult


# ============================================================
# Terms
# ============================================================

class Term:
    """Base class for all terms."""
    pass


class Atom(Term):
    """An atom (constant symbol)."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name

    def __hash__(self):
        return hash(('atom', self.name))


class Number(Term):
    """A numeric constant."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        v = self.value
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)

    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value

    def __hash__(self):
        return hash(('num', self.value))


class Var(Term):
    """A logic variable."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(('var', self.name))


class Compound(Term):
    """A compound term: functor(arg1, arg2, ...)."""
    __slots__ = ('functor', 'args')

    def __init__(self, functor, args):
        self.functor = functor
        self.args = tuple(args)

    def __repr__(self):
        if self.functor == '.' and len(self.args) == 2:
            return self._list_repr()
        if len(self.args) == 0:
            return f"{self.functor}()"
        return f"{self.functor}({', '.join(repr(a) for a in self.args)})"

    def _list_repr(self):
        """Pretty-print list terms like [a, b, c | T]."""
        elems = []
        cur = self
        while isinstance(cur, Compound) and cur.functor == '.' and len(cur.args) == 2:
            elems.append(repr(cur.args[0]))
            cur = cur.args[1]
        if isinstance(cur, Atom) and cur.name == '[]':
            return f"[{', '.join(elems)}]"
        return f"[{', '.join(elems)} | {repr(cur)}]"

    @property
    def indicator(self):
        return f"{self.functor}/{len(self.args)}"

    def __eq__(self, other):
        return (isinstance(other, Compound) and self.functor == other.functor
                and self.args == other.args)

    def __hash__(self):
        return hash(('compound', self.functor, self.args))


# Convenience list constructors
NIL = Atom('[]')

def cons(head, tail):
    return Compound('.', [head, tail])

def make_list(elements, tail=None):
    result = tail if tail is not None else NIL
    for elem in reversed(elements):
        result = cons(elem, result)
    return result

def list_to_python(term):
    """Convert a Prolog list term to a Python list."""
    result = []
    cur = term
    while isinstance(cur, Compound) and cur.functor == '.' and len(cur.args) == 2:
        result.append(cur.args[0])
        cur = cur.args[1]
    if isinstance(cur, Atom) and cur.name == '[]':
        return result
    return None  # not a proper list


# ============================================================
# Substitution / Unification
# ============================================================

class Substitution:
    """A substitution mapping variables to terms."""

    def __init__(self, bindings=None):
        self.bindings = dict(bindings) if bindings else {}

    def lookup(self, var):
        """Walk the substitution chain to find the final binding."""
        if not isinstance(var, Var):
            return var
        seen = set()
        while isinstance(var, Var) and var.name in self.bindings:
            if var.name in seen:
                return var
            seen.add(var.name)
            var = self.bindings[var.name]
        return var

    def walk(self, term):
        """Fully dereference a term through the substitution."""
        if isinstance(term, Var):
            term = self.lookup(term)
            if isinstance(term, Var):
                return term
            return self.walk(term)
        if isinstance(term, Compound):
            return Compound(term.functor, [self.walk(a) for a in term.args])
        return term

    def bind(self, var_name, term):
        """Create a new substitution with an additional binding."""
        s = Substitution(self.bindings)
        s.bindings[var_name] = term
        return s

    def copy(self):
        return Substitution(self.bindings)

    def __repr__(self):
        return f"Subst({self.bindings})"


def occurs_check(var_name, term, subst):
    """Check if var_name occurs in term (prevents infinite terms)."""
    term = subst.lookup(term) if isinstance(term, Var) else term
    if isinstance(term, Var):
        return term.name == var_name
    if isinstance(term, Compound):
        return any(occurs_check(var_name, a, subst) for a in term.args)
    return False


def unify(t1, t2, subst=None, check_occurs=True):
    """Unify two terms under a substitution. Returns new Substitution or None."""
    if subst is None:
        subst = Substitution()

    t1 = subst.walk(t1)
    t2 = subst.walk(t2)

    if t1 == t2:
        return subst

    if isinstance(t1, Var):
        if check_occurs and occurs_check(t1.name, t2, subst):
            return None
        return subst.bind(t1.name, t2)

    if isinstance(t2, Var):
        if check_occurs and occurs_check(t2.name, t1, subst):
            return None
        return subst.bind(t2.name, t1)

    if isinstance(t1, Number) and isinstance(t2, Number):
        return subst if t1.value == t2.value else None

    if isinstance(t1, Atom) and isinstance(t2, Atom):
        return subst if t1.name == t2.name else None

    if isinstance(t1, Compound) and isinstance(t2, Compound):
        if t1.functor != t2.functor or len(t1.args) != len(t2.args):
            return None
        for a1, a2 in zip(t1.args, t2.args):
            subst = unify(a1, a2, subst, check_occurs)
            if subst is None:
                return None
        return subst

    return None


# ============================================================
# Clauses and Database
# ============================================================

class Clause:
    """A Horn clause: head :- body1, body2, ..."""

    def __init__(self, head, body=None):
        self.head = head
        self.body = body or []

    def __repr__(self):
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {', '.join(repr(b) for b in self.body)}."


class CutSignal(Exception):
    """Signal for cut (!) -- prunes choice points."""
    pass


class HaltSignal(Exception):
    """Signal to halt execution."""
    pass


# ============================================================
# Parser
# ============================================================

class Token:
    __slots__ = ('type', 'value', 'line')
    def __init__(self, type, value, line=0):
        self.type = type
        self.value = value
        self.line = line
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """Tokenize Prolog source."""

    KEYWORDS = {'is', 'not', 'mod'}

    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.line = 1

    def peek(self):
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def advance(self):
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
        return ch

    def skip_whitespace_and_comments(self):
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in ' \t\r\n':
                self.advance()
            elif ch == '%':
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self.advance()
            elif ch == '/' and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == '*':
                self.advance()
                self.advance()
                while self.pos + 1 < len(self.source):
                    if self.source[self.pos] == '*' and self.source[self.pos + 1] == '/':
                        self.advance()
                        self.advance()
                        break
                    self.advance()
            else:
                break

    def tokenize(self):
        tokens = []
        while True:
            self.skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                tokens.append(Token('EOF', None, self.line))
                break

            ch = self.source[self.pos]
            line = self.line

            # Numbers
            if ch.isdigit() or (ch == '-' and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                # Only treat '-' as negative sign if previous token is an operator or start
                if ch == '-':
                    if tokens and tokens[-1].type in ('ATOM', 'VAR', 'NUMBER', 'RPAREN', 'RBRACKET'):
                        tokens.append(Token('OP', '-', line))
                        self.advance()
                        continue
                start = self.pos
                if ch == '-':
                    self.advance()
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self.advance()
                if self.pos < len(self.source) and self.source[self.pos] == '.':
                    if self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit():
                        self.advance()
                        while self.pos < len(self.source) and self.source[self.pos].isdigit():
                            self.advance()
                        tokens.append(Token('NUMBER', float(self.source[start:self.pos]), line))
                        continue
                tokens.append(Token('NUMBER', int(self.source[start:self.pos]), line))
                continue

            # Variables (uppercase or _)
            if ch.isupper() or ch == '_':
                start = self.pos
                while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.advance()
                name = self.source[start:self.pos]
                if name == '_':
                    tokens.append(Token('ANON', '_', line))
                else:
                    tokens.append(Token('VAR', name, line))
                continue

            # Atoms (lowercase identifiers or quoted)
            if ch.islower():
                start = self.pos
                while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.advance()
                name = self.source[start:self.pos]
                tokens.append(Token('ATOM', name, line))
                continue

            # Quoted atoms
            if ch == "'":
                self.advance()
                start = self.pos
                while self.pos < len(self.source) and self.source[self.pos] != "'":
                    self.advance()
                name = self.source[start:self.pos]
                if self.pos < len(self.source):
                    self.advance()
                tokens.append(Token('ATOM', name, line))
                continue

            # Strings (double-quoted -> list of character codes)
            if ch == '"':
                self.advance()
                s = ''
                while self.pos < len(self.source) and self.source[self.pos] != '"':
                    if self.source[self.pos] == '\\' and self.pos + 1 < len(self.source):
                        self.advance()
                        esc = self.advance()
                        if esc == 'n':
                            s += '\n'
                        elif esc == 't':
                            s += '\t'
                        else:
                            s += esc
                    else:
                        s += self.advance()
                if self.pos < len(self.source):
                    self.advance()
                tokens.append(Token('STRING', s, line))
                continue

            # Multi-character operators
            two = self.source[self.pos:self.pos + 2]
            three = self.source[self.pos:self.pos + 3]

            if three == '\\==':
                tokens.append(Token('OP', '\\==', line))
                self.pos += 3
                continue
            if three == '=..':
                tokens.append(Token('OP', '=..', line))
                self.pos += 3
                continue
            if three == '-->':
                tokens.append(Token('OP', '-->', line))
                self.pos += 3
                continue

            if two == ':-':
                tokens.append(Token('NECK', ':-', line))
                self.pos += 2
                continue
            if two == '?-':
                tokens.append(Token('QUERY', '?-', line))
                self.pos += 2
                continue
            if two == '\\+':
                tokens.append(Token('OP', '\\+', line))
                self.pos += 2
                continue
            if three == '=:=':
                tokens.append(Token('OP', '=:=', line))
                self.pos += 3
                continue
            if three == '=\\=':
                tokens.append(Token('OP', '=\\=', line))
                self.pos += 3
                continue
            if two == '>=':
                tokens.append(Token('OP', '>=', line))
                self.pos += 2
                continue
            if two == '=<':
                tokens.append(Token('OP', '=<', line))
                self.pos += 2
                continue
            if two == '->':
                tokens.append(Token('OP', '->', line))
                self.pos += 2
                continue
            if two == '==':
                tokens.append(Token('OP', '==', line))
                self.pos += 2
                continue
            if two == '@<':
                tokens.append(Token('OP', '@<', line))
                self.pos += 2
                continue
            if two == '@>':
                tokens.append(Token('OP', '@>', line))
                self.pos += 2
                continue

            # Single-character tokens
            if ch == '(':
                tokens.append(Token('LPAREN', '(', line))
                self.advance()
                continue
            if ch == ')':
                tokens.append(Token('RPAREN', ')', line))
                self.advance()
                continue
            if ch == '[':
                tokens.append(Token('LBRACKET', '[', line))
                self.advance()
                continue
            if ch == ']':
                tokens.append(Token('RBRACKET', ']', line))
                self.advance()
                continue
            if ch == '.':
                tokens.append(Token('DOT', '.', line))
                self.advance()
                continue
            if ch == ',':
                tokens.append(Token('COMMA', ',', line))
                self.advance()
                continue
            if ch == ';':
                tokens.append(Token('SEMI', ';', line))
                self.advance()
                continue
            if ch == '|':
                tokens.append(Token('BAR', '|', line))
                self.advance()
                continue
            if ch == '!':
                tokens.append(Token('CUT', '!', line))
                self.advance()
                continue
            if ch in '+-*/<>=\\':
                tokens.append(Token('OP', ch, line))
                self.advance()
                continue

            # Skip unknown
            self.advance()

        return tokens


class Parser:
    """Parse Prolog terms and clauses from tokens."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self._anon_counter = 0

    def peek(self):
        if self.pos >= len(self.tokens):
            return Token('EOF', None)
        return self.tokens[self.pos]

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type, value=None):
        tok = self.advance()
        if tok.type != type or (value is not None and tok.value != value):
            raise SyntaxError(f"Expected {type}({value}) but got {tok}")
        return tok

    def fresh_anon(self):
        self._anon_counter += 1
        return Var(f"_Anon{self._anon_counter}")

    def parse_program(self):
        """Parse a full program (sequence of clauses and queries)."""
        clauses = []
        queries = []
        while self.peek().type != 'EOF':
            if self.peek().type == 'QUERY':
                self.advance()
                body = self.parse_body()
                self.expect('DOT')
                queries.append(body)
            else:
                clause = self.parse_clause()
                clauses.append(clause)
        return clauses, queries

    def parse_clause(self):
        """Parse a single clause: head. or head :- body."""
        head = self.parse_term()
        if self.peek().type == 'NECK':
            self.advance()
            body = self.parse_body()
            self.expect('DOT')
            return Clause(head, body)
        else:
            self.expect('DOT')
            return Clause(head, [])

    def parse_body(self):
        """Parse a clause body (conjunction of goals)."""
        goals = [self.parse_goal()]
        while self.peek().type == 'COMMA':
            self.advance()
            goals.append(self.parse_goal())
        return goals

    def parse_goal(self):
        """Parse a single goal (may include disjunction and if-then)."""
        return self.parse_disjunction()

    def parse_disjunction(self):
        """Parse disjunction (;) and if-then (->)."""
        left = self.parse_conjunction_goal()

        # Check for if-then first
        if self.peek().type == 'OP' and self.peek().value == '->':
            self.advance()
            then = self.parse_conjunction_goal()
            left = Compound('->', [left, then])

        if self.peek().type == 'SEMI':
            self.advance()
            right = self.parse_disjunction()
            # Check for if-then-else: (Cond -> Then ; Else)
            if (isinstance(left, Compound) and left.functor == '->'
                    and len(left.args) == 2):
                return Compound('if_then_else', [left.args[0], left.args[1], right])
            return Compound(';', [left, right])

        return left

    def parse_conjunction_goal(self):
        """Parse a single term within a goal context."""
        return self.parse_term()

    def parse_term(self):
        """Parse a term with operator precedence."""
        return self.parse_comparison()

    def parse_comparison(self):
        """Parse comparison operators."""
        left = self.parse_arith()
        tok = self.peek()
        if tok.type == 'OP' and tok.value in ('=:=', '=\\=', '<', '>', '>=', '=<',
                                                '==', '\\==', '=..', '@<', '@>',
                                                '\\+'):
            if tok.value == '\\+':
                return left
            op = self.advance().value
            right = self.parse_arith()
            return Compound(op, [left, right])
        if tok.type == 'ATOM' and tok.value == 'is':
            self.advance()
            right = self.parse_arith()
            return Compound('is', [left, right])
        # Unification
        if tok.type == 'OP' and tok.value == '=':
            self.advance()
            right = self.parse_arith()
            return Compound('=', [left, right])
        if tok.type == 'OP' and tok.value == '\\':
            # Check for \=
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'OP' and self.tokens[self.pos + 1].value == '=':
                self.advance()  # consume '\'
                self.advance()  # consume '='
                right = self.parse_arith()
                return Compound('\\=', [left, right])
            return left
        return left

    def parse_arith(self):
        """Parse arithmetic: + and -."""
        left = self.parse_factor()
        while self.peek().type == 'OP' and self.peek().value in ('+', '-'):
            op = self.advance().value
            right = self.parse_factor()
            left = Compound(op, [left, right])
        return left

    def parse_factor(self):
        """Parse multiplication, division, mod."""
        left = self.parse_unary()
        while True:
            tok = self.peek()
            if tok.type == 'OP' and tok.value in ('*', '/', '//', 'mod'):
                op = self.advance().value
                right = self.parse_unary()
                left = Compound(op, [left, right])
            elif tok.type == 'ATOM' and tok.value == 'mod':
                self.advance()
                right = self.parse_unary()
                left = Compound('mod', [left, right])
            else:
                break
        return left

    def parse_unary(self):
        """Parse unary operators."""
        tok = self.peek()
        if tok.type == 'OP' and tok.value == '\\+':
            self.advance()
            arg = self.parse_primary()
            return Compound('\\+', [arg])
        if tok.type == 'OP' and tok.value == '-':
            self.advance()
            arg = self.parse_primary()
            if isinstance(arg, Number):
                return Number(-arg.value)
            return Compound('-', [arg])
        return self.parse_primary()

    def parse_primary(self):
        """Parse primary terms: atoms, numbers, variables, compounds, lists, parens."""
        tok = self.peek()

        if tok.type == 'LPAREN':
            self.advance()
            term = self.parse_goal()
            # Handle conjunction (commas) inside parentheses
            if self.peek().type == 'COMMA':
                terms = [term]
                while self.peek().type == 'COMMA':
                    self.advance()
                    terms.append(self.parse_goal())
                term = terms[0]
                for t in terms[1:]:
                    term = Compound(',', [term, t])
            # Handle :- inside parentheses (for assert((Head :- Body)))
            if self.peek().type == 'NECK':
                self.advance()
                body = self.parse_goal()
                if self.peek().type == 'COMMA':
                    body_terms = [body]
                    while self.peek().type == 'COMMA':
                        self.advance()
                        body_terms.append(self.parse_goal())
                    body = body_terms[0]
                    for t in body_terms[1:]:
                        body = Compound(',', [body, t])
                term = Compound(':-', [term, body])
            self.expect('RPAREN')
            return term

        if tok.type == 'LBRACKET':
            return self.parse_list()

        if tok.type == 'NUMBER':
            self.advance()
            return Number(tok.value)

        if tok.type == 'VAR':
            self.advance()
            return Var(tok.value)

        if tok.type == 'ANON':
            self.advance()
            return self.fresh_anon()

        if tok.type == 'CUT':
            self.advance()
            return Atom('!')

        if tok.type == 'STRING':
            self.advance()
            # String as list of character codes
            return make_list([Number(ord(c)) for c in tok.value])

        if tok.type == 'ATOM':
            name = tok.value
            self.advance()
            if self.peek().type == 'LPAREN':
                self.advance()
                args = []
                if self.peek().type != 'RPAREN':
                    args.append(self.parse_goal())
                    while self.peek().type == 'COMMA':
                        self.advance()
                        args.append(self.parse_goal())
                self.expect('RPAREN')
                return Compound(name, args)
            return Atom(name)

        if tok.type == 'OP' and tok.value == '-':
            # Negative number
            self.advance()
            next_tok = self.peek()
            if next_tok.type == 'NUMBER':
                self.advance()
                return Number(-next_tok.value)
            return Atom('-')

        raise SyntaxError(f"Unexpected token: {tok}")

    def parse_list(self):
        """Parse a list: [a, b, c] or [H | T]."""
        self.expect('LBRACKET')
        if self.peek().type == 'RBRACKET':
            self.advance()
            return NIL

        elements = [self.parse_goal()]
        while self.peek().type == 'COMMA':
            self.advance()
            if self.peek().type == 'RBRACKET':
                break
            elements.append(self.parse_goal())

        tail = NIL
        if self.peek().type == 'BAR':
            self.advance()
            tail = self.parse_goal()

        self.expect('RBRACKET')
        return make_list(elements, tail)


def parse(source):
    """Parse a Prolog source string into clauses and queries."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse_program()


def parse_term(source):
    """Parse a single term from a string."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse_term()


def parse_query(source):
    """Parse a query (body) from a string."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse_body()


# ============================================================
# Arithmetic Evaluator
# ============================================================

def eval_arith(term, subst):
    """Evaluate an arithmetic expression under a substitution."""
    term = subst.walk(term)

    if isinstance(term, Number):
        return term.value

    if isinstance(term, Var):
        raise ValueError(f"Unbound variable in arithmetic: {term.name}")

    if isinstance(term, Compound):
        if len(term.args) == 2:
            left = eval_arith(term.args[0], subst)
            right = eval_arith(term.args[1], subst)
            if term.functor == '+':
                return left + right
            if term.functor == '-':
                return left - right
            if term.functor == '*':
                return left * right
            if term.functor == '/':
                return left / right
            if term.functor == '//':
                return int(left) // int(right)
            if term.functor == 'mod':
                return int(left) % int(right)
            if term.functor == '**':
                return left ** right
            if term.functor == 'min':
                return min(left, right)
            if term.functor == 'max':
                return max(left, right)
        if len(term.args) == 1:
            val = eval_arith(term.args[0], subst)
            if term.functor == '-':
                return -val
            if term.functor == 'abs':
                return abs(val)
            if term.functor == 'succ':
                return val + 1
        raise ValueError(f"Unknown arithmetic functor: {term.functor}/{len(term.args)}")

    raise ValueError(f"Cannot evaluate: {term}")


# ============================================================
# Interpreter
# ============================================================

class Interpreter:
    """Prolog interpreter with SLD resolution."""

    def __init__(self, max_depth=1000):
        self.database = OrderedDict()  # functor/arity -> [Clause]
        self.max_depth = max_depth
        self._var_counter = 0
        self.output = []
        self.trace = False
        # CLP(FD) state
        self._clp_vars = {}  # var_name -> CSPVariable
        self._clp_constraints = []

    def add_clause(self, clause):
        """Add a clause to the database."""
        if isinstance(clause.head, Atom):
            key = f"{clause.head.name}/0"
        elif isinstance(clause.head, Compound):
            key = clause.head.indicator
        else:
            raise ValueError(f"Invalid clause head: {clause.head}")
        if key not in self.database:
            self.database[key] = []
        self.database[key].append(clause)

    def add_clauses(self, clauses):
        """Add multiple clauses."""
        for c in clauses:
            self.add_clause(c)

    def consult(self, source):
        """Parse and load a Prolog source string."""
        clauses, queries = parse(source)
        self.add_clauses(clauses)
        results = []
        for query_goals in queries:
            for sol in self.query_all(query_goals):
                results.append(sol)
        return results

    def _rename_vars(self, clause):
        """Rename variables in a clause to fresh names (standardize apart)."""
        self._var_counter += 1
        suffix = f"_{self._var_counter}"
        mapping = {}

        def rename(term):
            if isinstance(term, Var):
                if term.name not in mapping:
                    mapping[term.name] = Var(term.name + suffix)
                return mapping[term.name]
            if isinstance(term, Compound):
                return Compound(term.functor, [rename(a) for a in term.args])
            return term

        new_head = rename(clause.head)
        new_body = [rename(g) for g in clause.body]
        return Clause(new_head, new_body)

    def query(self, goals, subst=None):
        """Yield substitutions that satisfy the goals (generator)."""
        if subst is None:
            subst = Substitution()
        yield from self._solve(goals, subst, 0)

    def query_all(self, goals, subst=None):
        """Return all solutions as a list of substitutions."""
        return list(self.query(goals, subst))

    def query_string(self, source):
        """Parse a query string and return all solutions."""
        goals = parse_query(source)
        return self.query_all(goals)

    def _solve(self, goals, subst, depth):
        """SLD resolution: solve a list of goals."""
        if depth > self.max_depth:
            return

        if not goals:
            yield subst
            return

        goal = subst.walk(goals[0])
        rest = goals[1:]

        if self.trace:
            self.output.append(f"CALL: {goal} depth={depth}")

        # Handle atom-based builtins (no args)
        if isinstance(goal, Atom):
            if goal.name == '!':
                def cut_gen():
                    for s in self._solve(rest, subst, depth + 1):
                        yield s
                    raise CutSignal()
                yield from cut_gen()
                return
            if goal.name == 'true':
                yield from self._solve(rest, subst, depth + 1)
                return
            if goal.name == 'fail':
                return
            if goal.name == 'halt':
                raise HaltSignal()
            if goal.name == 'nl':
                self.output.append('\n')
                yield from self._solve(rest, subst, depth + 1)
                return

        # Handle built-in goals
        builtin = self._try_builtin(goal, subst, rest, depth)
        if builtin is not None:
            yield from builtin
            return

        # User-defined predicates
        if isinstance(goal, Atom):
            key = f"{goal.name}/0"
        elif isinstance(goal, Compound):
            key = goal.indicator
        else:
            return

        clauses = self.database.get(key, [])

        for clause in clauses:
            renamed = self._rename_vars(clause)
            new_subst = unify(goal, renamed.head, subst)
            if new_subst is not None:
                try:
                    yield from self._solve(renamed.body + rest, new_subst, depth + 1)
                except CutSignal:
                    return

    def _try_builtin(self, goal, subst, rest, depth):
        """Try to handle goal as a built-in. Returns generator or None."""
        # Cut
        if isinstance(goal, Atom) and goal.name == '!':
            def cut_gen():
                for s in self._solve(rest, subst, depth + 1):
                    yield s
                raise CutSignal()
            return cut_gen()

        # true/fail
        if isinstance(goal, Atom) and goal.name == 'true':
            return self._solve(rest, subst, depth + 1)
        if isinstance(goal, Atom) and goal.name == 'fail':
            return iter([])

        # halt
        if isinstance(goal, Atom) and goal.name == 'halt':
            def halt_gen():
                raise HaltSignal()
                yield  # make it a generator
            return halt_gen()

        if not isinstance(goal, Compound):
            return None

        f, args = goal.functor, goal.args

        # Conjunction
        if f == ',' and len(args) == 2:
            return self._solve([args[0], args[1]] + rest, subst, depth + 1)

        # Disjunction
        if f == ';' and len(args) == 2:
            def disj_gen():
                yield from self._solve([args[0]] + rest, subst, depth + 1)
                yield from self._solve([args[1]] + rest, subst, depth + 1)
            return disj_gen()

        # If-then-else
        if f == 'if_then_else' and len(args) == 3:
            def ite_gen():
                found = False
                for s in self._solve([args[0]], subst, depth + 1):
                    found = True
                    yield from self._solve([args[1]] + rest, s, depth + 1)
                    break  # commit to first solution of condition
                if not found:
                    yield from self._solve([args[2]] + rest, subst, depth + 1)
            return ite_gen()

        # If-then (no else -> fail)
        if f == '->' and len(args) == 2:
            def ifthen_gen():
                for s in self._solve([args[0]], subst, depth + 1):
                    yield from self._solve([args[1]] + rest, s, depth + 1)
                    break
            return ifthen_gen()

        # Unification
        if f == '=' and len(args) == 2:
            new_subst = unify(args[0], args[1], subst)
            if new_subst is not None:
                return self._solve(rest, new_subst, depth + 1)
            return iter([])

        # Not-unify
        if f == '\\=' and len(args) == 2:
            new_subst = unify(args[0], args[1], subst)
            if new_subst is None:
                return self._solve(rest, subst, depth + 1)
            return iter([])

        # Structural equality
        if f == '==' and len(args) == 2:
            t1 = subst.walk(args[0])
            t2 = subst.walk(args[1])
            if t1 == t2:
                return self._solve(rest, subst, depth + 1)
            return iter([])

        # Structural inequality
        if f == '\\==' and len(args) == 2:
            t1 = subst.walk(args[0])
            t2 = subst.walk(args[1])
            if t1 != t2:
                return self._solve(rest, subst, depth + 1)
            return iter([])

        # Negation as failure
        if f == '\\+' and len(args) == 1:
            def naf_gen():
                for _ in self._solve([args[0]], subst, depth + 1):
                    return  # goal succeeded, so \+ fails
                yield from self._solve(rest, subst, depth + 1)
            return naf_gen()

        # Arithmetic is
        if f == 'is' and len(args) == 2:
            try:
                val = eval_arith(args[1], subst)
                if isinstance(val, float) and val == int(val):
                    val = int(val)
                new_subst = unify(args[0], Number(val), subst)
                if new_subst is not None:
                    return self._solve(rest, new_subst, depth + 1)
            except (ValueError, ZeroDivisionError):
                pass
            return iter([])

        # Arithmetic comparisons
        if f == '=:=' and len(args) == 2:
            try:
                if eval_arith(args[0], subst) == eval_arith(args[1], subst):
                    return self._solve(rest, subst, depth + 1)
            except ValueError:
                pass
            return iter([])

        if f == '=\\=' and len(args) == 2:
            try:
                if eval_arith(args[0], subst) != eval_arith(args[1], subst):
                    return self._solve(rest, subst, depth + 1)
            except ValueError:
                pass
            return iter([])

        if f == '<' and len(args) == 2:
            try:
                if eval_arith(args[0], subst) < eval_arith(args[1], subst):
                    return self._solve(rest, subst, depth + 1)
            except ValueError:
                pass
            return iter([])

        if f == '>' and len(args) == 2:
            try:
                if eval_arith(args[0], subst) > eval_arith(args[1], subst):
                    return self._solve(rest, subst, depth + 1)
            except ValueError:
                pass
            return iter([])

        if f == '>=' and len(args) == 2:
            try:
                if eval_arith(args[0], subst) >= eval_arith(args[1], subst):
                    return self._solve(rest, subst, depth + 1)
            except ValueError:
                pass
            return iter([])

        if f == '=<' and len(args) == 2:
            try:
                if eval_arith(args[0], subst) <= eval_arith(args[1], subst):
                    return self._solve(rest, subst, depth + 1)
            except ValueError:
                pass
            return iter([])

        # Type checking
        if f == 'atom' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Atom):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'number' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Number):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'integer' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Number) and isinstance(t.value, int):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'float' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Number) and isinstance(t.value, float):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'var' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Var):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'nonvar' and len(args) == 1:
            t = subst.walk(args[0])
            if not isinstance(t, Var):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'compound' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Compound):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'is_list' and len(args) == 1:
            t = subst.walk(args[0])
            if list_to_python(t) is not None:
                return self._solve(rest, subst, depth + 1)
            return iter([])

        if f == 'callable' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, (Atom, Compound)):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        # Functor / arg / =..
        if f == 'functor' and len(args) == 3:
            t = subst.walk(args[0])
            if isinstance(t, Atom):
                s1 = unify(args[1], t, subst)
                if s1:
                    s2 = unify(args[2], Number(0), s1)
                    if s2:
                        return self._solve(rest, s2, depth + 1)
            elif isinstance(t, Number):
                s1 = unify(args[1], t, subst)
                if s1:
                    s2 = unify(args[2], Number(0), s1)
                    if s2:
                        return self._solve(rest, s2, depth + 1)
            elif isinstance(t, Compound):
                s1 = unify(args[1], Atom(t.functor), subst)
                if s1:
                    s2 = unify(args[2], Number(len(t.args)), s1)
                    if s2:
                        return self._solve(rest, s2, depth + 1)
            elif isinstance(t, Var):
                # Construct: functor(Name, Arity, Result)
                name = subst.walk(args[1])
                arity = subst.walk(args[2])
                if isinstance(name, Atom) and isinstance(arity, Number):
                    n = int(arity.value)
                    if n == 0:
                        s = unify(args[0], name, subst)
                    else:
                        new_args = [Var(f"_FA{i}_{self._var_counter}") for i in range(n)]
                        self._var_counter += 1
                        s = unify(args[0], Compound(name.name, new_args), subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'arg' and len(args) == 3:
            n = subst.walk(args[0])
            t = subst.walk(args[1])
            if isinstance(n, Number) and isinstance(t, Compound):
                idx = int(n.value) - 1
                if 0 <= idx < len(t.args):
                    s = unify(args[2], t.args[idx], subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == '=..' and len(args) == 2:
            t = subst.walk(args[0])
            if isinstance(t, Compound):
                lst = make_list([Atom(t.functor)] + list(t.args))
                s = unify(args[1], lst, subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(t, Atom):
                lst = make_list([t])
                s = unify(args[1], lst, subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(t, Number):
                lst = make_list([t])
                s = unify(args[1], lst, subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(t, Var):
                # Reconstruct from list
                lst = subst.walk(args[1])
                elems = list_to_python(lst)
                if elems and len(elems) >= 1:
                    head = elems[0]
                    if isinstance(head, Atom) and len(elems) > 1:
                        s = unify(args[0], Compound(head.name, elems[1:]), subst)
                    else:
                        s = unify(args[0], head, subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        # Copy term
        if f == 'copy_term' and len(args) == 2:
            original = subst.walk(args[0])
            copied = self._copy_term(original)
            s = unify(args[1], copied, subst)
            if s:
                return self._solve(rest, s, depth + 1)
            return iter([])

        # List operations
        if f == 'append' and len(args) == 3:
            return self._builtin_append(args, subst, rest, depth)

        if f == 'length' and len(args) == 2:
            t = subst.walk(args[0])
            elems = list_to_python(t)
            if elems is not None:
                s = unify(args[1], Number(len(elems)), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'nth0' and len(args) == 3:
            n = subst.walk(args[0])
            lst = subst.walk(args[1])
            elems = list_to_python(lst)
            if isinstance(n, Number) and elems is not None:
                idx = int(n.value)
                if 0 <= idx < len(elems):
                    s = unify(args[2], elems[idx], subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'nth1' and len(args) == 3:
            n = subst.walk(args[0])
            lst = subst.walk(args[1])
            elems = list_to_python(lst)
            if isinstance(n, Number) and elems is not None:
                idx = int(n.value) - 1
                if 0 <= idx < len(elems):
                    s = unify(args[2], elems[idx], subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'last' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems and len(elems) > 0:
                s = unify(args[1], elems[-1], subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'reverse' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems is not None:
                s = unify(args[1], make_list(list(reversed(elems))), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'member' and len(args) == 2:
            def member_gen():
                lst = subst.walk(args[1])
                elems = list_to_python(lst)
                if elems is not None:
                    for elem in elems:
                        s = unify(args[0], elem, subst)
                        if s:
                            yield from self._solve(rest, s, depth + 1)
            return member_gen()

        if f == 'msort' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems is not None:
                sorted_elems = sorted(elems, key=self._term_sort_key)
                s = unify(args[1], make_list(sorted_elems), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'sort' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems is not None:
                # sort removes duplicates
                seen = []
                for e in sorted(elems, key=self._term_sort_key):
                    if not seen or seen[-1] != e:
                        seen.append(e)
                s = unify(args[1], make_list(seen), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'flatten' and len(args) == 2:
            lst = subst.walk(args[0])
            flat = self._flatten_list(lst)
            if flat is not None:
                s = unify(args[1], make_list(flat), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'permutation' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems is not None:
                def perm_gen():
                    from itertools import permutations
                    for p in permutations(elems):
                        s = unify(args[1], make_list(list(p)), subst)
                        if s:
                            yield from self._solve(rest, s, depth + 1)
                return perm_gen()
            return iter([])

        if f == 'select' and len(args) == 3:
            def select_gen():
                lst = subst.walk(args[1])
                elems = list_to_python(lst)
                if elems is not None:
                    for i, elem in enumerate(elems):
                        s = unify(args[0], elem, subst)
                        if s:
                            remaining = elems[:i] + elems[i+1:]
                            s2 = unify(args[2], make_list(remaining), s)
                            if s2:
                                yield from self._solve(rest, s2, depth + 1)
            return select_gen()

        if f == 'maplist' and len(args) >= 2:
            return self._builtin_maplist(args, subst, rest, depth)

        if f == 'include' and len(args) == 3:
            return self._builtin_include(args, subst, rest, depth)

        if f == 'foldl' and len(args) == 4:
            return self._builtin_foldl(args, subst, rest, depth)

        if f == 'numlist' and len(args) == 3:
            lo = subst.walk(args[0])
            hi = subst.walk(args[1])
            if isinstance(lo, Number) and isinstance(hi, Number):
                nums = [Number(i) for i in range(int(lo.value), int(hi.value) + 1)]
                s = unify(args[2], make_list(nums), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'msort' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems is not None:
                sorted_elems = sorted(elems, key=self._term_sort_key)
                s = unify(args[1], make_list(sorted_elems), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        # I/O
        if f == 'write' and len(args) == 1:
            t = subst.walk(args[0])
            self.output.append(self._format_term(t))
            return self._solve(rest, subst, depth + 1)

        if f == 'writeln' and len(args) == 1:
            t = subst.walk(args[0])
            self.output.append(self._format_term(t) + '\n')
            return self._solve(rest, subst, depth + 1)

        if f == 'nl' and len(args) == 0:
            self.output.append('\n')
            return self._solve(rest, subst, depth + 1)

        if f == 'write_canonical' and len(args) == 1:
            t = subst.walk(args[0])
            self.output.append(repr(t))
            return self._solve(rest, subst, depth + 1)

        if f == 'print' and len(args) == 1:
            t = subst.walk(args[0])
            self.output.append(self._format_term(t))
            return self._solve(rest, subst, depth + 1)

        # findall / bagof / setof
        if f == 'findall' and len(args) == 3:
            return self._builtin_findall(args, subst, rest, depth)

        if f == 'bagof' and len(args) == 3:
            return self._builtin_bagof(args, subst, rest, depth)

        if f == 'setof' and len(args) == 3:
            return self._builtin_setof(args, subst, rest, depth)

        if f == 'aggregate_all' and len(args) == 3:
            return self._builtin_findall(args, subst, rest, depth)

        # Assert / Retract
        if f == 'assert' and len(args) == 1:
            return self._builtin_assert(args[0], subst, rest, depth, at_end=True)

        if f == 'assertz' and len(args) == 1:
            return self._builtin_assert(args[0], subst, rest, depth, at_end=True)

        if f == 'asserta' and len(args) == 1:
            return self._builtin_assert(args[0], subst, rest, depth, at_end=False)

        if f == 'retract' and len(args) == 1:
            return self._builtin_retract(args[0], subst, rest, depth)

        if f == 'abolish' and len(args) == 1:
            t = subst.walk(args[0])
            if isinstance(t, Compound) and t.functor == '/' and len(t.args) == 2:
                name = t.args[0]
                arity = t.args[1]
                if isinstance(name, Atom) and isinstance(arity, Number):
                    key = f"{name.name}/{int(arity.value)}"
                    self.database.pop(key, None)
                    return self._solve(rest, subst, depth + 1)
            return iter([])

        # Call
        if f == 'call' and len(args) >= 1:
            callee = subst.walk(args[0])
            if len(args) > 1:
                # call/N: apply extra args
                extra = [subst.walk(a) for a in args[1:]]
                if isinstance(callee, Atom):
                    callee = Compound(callee.name, extra)
                elif isinstance(callee, Compound):
                    callee = Compound(callee.functor, list(callee.args) + extra)
            return self._solve([callee] + rest, subst, depth + 1)

        # Apply (for higher-order)
        if f == 'apply' and len(args) == 2:
            callee = subst.walk(args[0])
            arg_list = subst.walk(args[1])
            elems = list_to_python(arg_list)
            if elems is not None:
                if isinstance(callee, Atom):
                    callee = Compound(callee.name, elems)
                elif isinstance(callee, Compound):
                    callee = Compound(callee.functor, list(callee.args) + elems)
                return self._solve([callee] + rest, subst, depth + 1)
            return iter([])

        # between/3
        if f == 'between' and len(args) == 3:
            lo = subst.walk(args[0])
            hi = subst.walk(args[1])
            if isinstance(lo, Number) and isinstance(hi, Number):
                def between_gen():
                    for i in range(int(lo.value), int(hi.value) + 1):
                        s = unify(args[2], Number(i), subst)
                        if s:
                            yield from self._solve(rest, s, depth + 1)
                return between_gen()
            return iter([])

        # succ/2
        if f == 'succ' and len(args) == 2:
            t1 = subst.walk(args[0])
            t2 = subst.walk(args[1])
            if isinstance(t1, Number):
                s = unify(args[1], Number(int(t1.value) + 1), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(t2, Number):
                s = unify(args[0], Number(int(t2.value) - 1), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        # plus/3
        if f == 'plus' and len(args) == 3:
            a = subst.walk(args[0])
            b = subst.walk(args[1])
            c = subst.walk(args[2])
            if isinstance(a, Number) and isinstance(b, Number):
                s = unify(args[2], Number(a.value + b.value), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(a, Number) and isinstance(c, Number):
                s = unify(args[1], Number(c.value - a.value), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(b, Number) and isinstance(c, Number):
                s = unify(args[0], Number(c.value - b.value), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        # ground/1
        if f == 'ground' and len(args) == 1:
            t = subst.walk(args[0])
            if self._is_ground(t):
                return self._solve(rest, subst, depth + 1)
            return iter([])

        # number_chars / number_codes / atom_chars / atom_codes / atom_length / atom_concat / char_code
        if f == 'number_chars' and len(args) == 2:
            t = subst.walk(args[0])
            if isinstance(t, Number):
                chars = [Atom(c) for c in str(t.value)]
                s = unify(args[1], make_list(chars), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            else:
                lst = subst.walk(args[1])
                elems = list_to_python(lst)
                if elems:
                    num_str = ''.join(e.name for e in elems if isinstance(e, Atom))
                    try:
                        val = int(num_str) if '.' not in num_str else float(num_str)
                        s = unify(args[0], Number(val), subst)
                        if s:
                            return self._solve(rest, s, depth + 1)
                    except ValueError:
                        pass
            return iter([])

        if f == 'atom_chars' and len(args) == 2:
            t = subst.walk(args[0])
            if isinstance(t, Atom):
                chars = [Atom(c) for c in t.name]
                s = unify(args[1], make_list(chars), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            else:
                lst = subst.walk(args[1])
                elems = list_to_python(lst)
                if elems:
                    name = ''.join(e.name for e in elems if isinstance(e, Atom))
                    s = unify(args[0], Atom(name), subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'atom_length' and len(args) == 2:
            t = subst.walk(args[0])
            if isinstance(t, Atom):
                s = unify(args[1], Number(len(t.name)), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'atom_concat' and len(args) == 3:
            a = subst.walk(args[0])
            b = subst.walk(args[1])
            if isinstance(a, Atom) and isinstance(b, Atom):
                s = unify(args[2], Atom(a.name + b.name), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'char_code' and len(args) == 2:
            t = subst.walk(args[0])
            if isinstance(t, Atom) and len(t.name) == 1:
                s = unify(args[1], Number(ord(t.name)), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            else:
                n = subst.walk(args[1])
                if isinstance(n, Number):
                    s = unify(args[0], Atom(chr(int(n.value))), subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'sub_atom' and len(args) == 5:
            atom_t = subst.walk(args[0])
            if isinstance(atom_t, Atom):
                def sub_atom_gen():
                    name = atom_t.name
                    for before in range(len(name) + 1):
                        for length in range(len(name) - before + 1):
                            sub = name[before:before + length]
                            after = len(name) - before - length
                            s = unify(args[1], Number(before), subst)
                            if s:
                                s = unify(args[2], Number(length), s)
                                if s:
                                    s = unify(args[3], Number(after), s)
                                    if s:
                                        s = unify(args[4], Atom(sub), s)
                                        if s:
                                            yield from self._solve(rest, s, depth + 1)
                return sub_atom_gen()
            return iter([])

        # number_to_atom
        if f == 'atom_number' and len(args) == 2:
            t = subst.walk(args[0])
            if isinstance(t, Atom):
                try:
                    val = int(t.name)
                except ValueError:
                    try:
                        val = float(t.name)
                    except ValueError:
                        return iter([])
                s = unify(args[1], Number(val), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            elif isinstance(t, Var):
                n = subst.walk(args[1])
                if isinstance(n, Number):
                    s = unify(args[0], Atom(str(n.value)), subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        # Conversion: term_to_atom
        if f == 'term_to_atom' and len(args) == 2:
            t = subst.walk(args[0])
            if not isinstance(t, Var):
                s = unify(args[1], Atom(self._format_term(t)), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        # CLP(FD) integration
        if f == 'clp_in' and len(args) == 3:
            return self._clp_in(args, subst, rest, depth)

        if f == 'clp_constraint' and len(args) == 3:
            return self._clp_constraint(args, subst, rest, depth)

        if f == 'clp_all_different' and len(args) == 1:
            return self._clp_all_different(args, subst, rest, depth)

        if f == 'clp_solve' and len(args) == 1:
            return self._clp_solve(args, subst, rest, depth)

        if f == 'clp_label' and len(args) == 1:
            return self._clp_label(args, subst, rest, depth)

        # Catch / throw
        if f == 'catch' and len(args) == 3:
            return self._builtin_catch(args, subst, rest, depth)

        if f == 'throw' and len(args) == 1:
            def throw_gen():
                t = subst.walk(args[0])
                raise PrologError(t)
                yield  # make it a generator
            return throw_gen()

        # once/1
        if f == 'once' and len(args) == 1:
            def once_gen():
                for s in self._solve([args[0]], subst, depth + 1):
                    yield from self._solve(rest, s, depth + 1)
                    return
            return once_gen()

        # ignore/1
        if f == 'ignore' and len(args) == 1:
            def ignore_gen():
                for s in self._solve([args[0]], subst, depth + 1):
                    yield from self._solve(rest, s, depth + 1)
                    return
                yield from self._solve(rest, subst, depth + 1)
            return ignore_gen()

        # forall/2
        if f == 'forall' and len(args) == 2:
            def forall_gen():
                for s in self._solve([args[0]], subst, depth + 1):
                    found = False
                    for _ in self._solve([args[1]], s, depth + 1):
                        found = True
                        break
                    if not found:
                        return
                yield from self._solve(rest, subst, depth + 1)
            return forall_gen()

        # aggregate: sum_list, max_list, min_list
        if f == 'sum_list' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems is not None:
                total = sum(e.value for e in elems if isinstance(e, Number))
                s = unify(args[1], Number(total), subst)
                if s:
                    return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'max_list' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems:
                vals = [e.value for e in elems if isinstance(e, Number)]
                if vals:
                    s = unify(args[1], Number(max(vals)), subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        if f == 'min_list' and len(args) == 2:
            lst = subst.walk(args[0])
            elems = list_to_python(lst)
            if elems:
                vals = [e.value for e in elems if isinstance(e, Number)]
                if vals:
                    s = unify(args[1], Number(min(vals)), subst)
                    if s:
                        return self._solve(rest, s, depth + 1)
            return iter([])

        # msort already handled above

        # not a builtin
        return None

    # --- Builtin helpers ---

    def _builtin_append(self, args, subst, rest, depth):
        """append/3: append(Xs, Ys, Zs)."""
        # Try unification-based approach
        xs = subst.walk(args[0])
        if isinstance(xs, Atom) and xs.name == '[]':
            s = unify(args[1], args[2], subst)
            if s:
                return self._solve(rest, s, depth + 1)
            return iter([])

        if isinstance(xs, Compound) and xs.functor == '.' and len(xs.args) == 2:
            # append([H|T], Ys, [H|Zs]) :- append(T, Ys, Zs).
            h = xs.args[0]
            t = xs.args[1]
            self._var_counter += 1
            zs_tail = Var(f"_AppZs{self._var_counter}")
            s = unify(args[2], cons(h, zs_tail), subst)
            if s:
                return self._solve(
                    [Compound('append', [t, args[1], zs_tail])] + rest,
                    s, depth + 1
                )
            return iter([])

        # If Xs is a variable, enumerate by splitting Zs
        zs = subst.walk(args[2])
        if isinstance(zs, (Atom, Compound)):
            def append_enum():
                elems = list_to_python(zs)
                if elems is not None:
                    for i in range(len(elems) + 1):
                        xs_part = make_list(elems[:i])
                        ys_part = make_list(elems[i:])
                        s = unify(args[0], xs_part, subst)
                        if s:
                            s = unify(args[1], ys_part, s)
                            if s:
                                yield from self._solve(rest, s, depth + 1)
            return append_enum()

        return iter([])

    def _builtin_findall(self, args, subst, rest, depth):
        """findall/3: collect all solutions."""
        template = args[0]
        goal = args[1]
        results = []
        for s in self._solve([goal], subst, depth + 1):
            results.append(s.walk(template))
        new_subst = unify(args[2], make_list(results), subst)
        if new_subst:
            return self._solve(rest, new_subst, depth + 1)
        return iter([])

    def _builtin_bagof(self, args, subst, rest, depth):
        """bagof/3: like findall but fails if no solutions."""
        template = args[0]
        goal = args[1]
        results = []
        for s in self._solve([goal], subst, depth + 1):
            results.append(s.walk(template))
        if not results:
            return iter([])
        new_subst = unify(args[2], make_list(results), subst)
        if new_subst:
            return self._solve(rest, new_subst, depth + 1)
        return iter([])

    def _builtin_setof(self, args, subst, rest, depth):
        """setof/3: like bagof but sorted and deduplicated."""
        template = args[0]
        goal = args[1]
        results = []
        seen = set()
        for s in self._solve([goal], subst, depth + 1):
            t = s.walk(template)
            key = repr(t)
            if key not in seen:
                seen.add(key)
                results.append(t)
        if not results:
            return iter([])
        results.sort(key=self._term_sort_key)
        new_subst = unify(args[2], make_list(results), subst)
        if new_subst:
            return self._solve(rest, new_subst, depth + 1)
        return iter([])

    def _builtin_assert(self, term, subst, rest, depth, at_end=True):
        """assert/assertz/asserta: add clause to database."""
        t = subst.walk(term)
        if isinstance(t, Compound) and t.functor == ':-' and len(t.args) == 2:
            head = t.args[0]
            body_term = t.args[1]
            body = self._term_to_goals(body_term)
            clause = Clause(head, body)
        else:
            clause = Clause(t, [])

        if isinstance(clause.head, Atom):
            key = f"{clause.head.name}/0"
        elif isinstance(clause.head, Compound):
            key = clause.head.indicator
        else:
            return iter([])

        if key not in self.database:
            self.database[key] = []

        if at_end:
            self.database[key].append(clause)
        else:
            self.database[key].insert(0, clause)

        return self._solve(rest, subst, depth + 1)

    def _builtin_retract(self, term, subst, rest, depth):
        """retract/1: remove first matching clause."""
        t = subst.walk(term)
        if isinstance(t, Compound) and t.functor == ':-' and len(t.args) == 2:
            head_pat = t.args[0]
            body_pat = t.args[1]
        else:
            head_pat = t
            body_pat = None

        if isinstance(head_pat, Atom):
            key = f"{head_pat.name}/0"
        elif isinstance(head_pat, Compound):
            key = head_pat.indicator
        else:
            return iter([])

        def retract_gen():
            clauses = self.database.get(key, [])
            for i, clause in enumerate(clauses):
                renamed = self._rename_vars(clause)
                s = unify(head_pat, renamed.head, subst)
                if s is not None:
                    if body_pat is not None:
                        if renamed.body:
                            body_term = self._goals_to_term(renamed.body)
                            s = unify(body_pat, body_term, s)
                        else:
                            s = unify(body_pat, Atom('true'), s)
                    if s is not None:
                        clauses.pop(i)
                        yield from self._solve(rest, s, depth + 1)
                        return
        return retract_gen()

    def _builtin_catch(self, args, subst, rest, depth):
        """catch(Goal, Catcher, Recovery)."""
        def catch_gen():
            try:
                yield from self._solve([args[0]], subst, depth + 1)
            except PrologError as e:
                s = unify(args[1], e.term, subst)
                if s:
                    yield from self._solve([args[2]] + rest, s, depth + 1)
                else:
                    raise
        # Need to chain with rest on success
        def catch_and_rest():
            try:
                for s in self._solve([args[0]], subst, depth + 1):
                    yield from self._solve(rest, s, depth + 1)
            except PrologError as e:
                s = unify(args[1], e.term, subst)
                if s:
                    yield from self._solve([args[2]] + rest, s, depth + 1)
                else:
                    raise
        return catch_and_rest()

    def _builtin_maplist(self, args, subst, rest, depth):
        """maplist/2..5."""
        pred = subst.walk(args[0])
        lists = [subst.walk(a) for a in args[1:]]
        elems_lists = [list_to_python(l) for l in lists]

        # Find the first concrete list to determine length
        concrete_idx = None
        length = None
        for i, el in enumerate(elems_lists):
            if el is not None:
                concrete_idx = i
                length = len(el)
                break

        if concrete_idx is None:
            return iter([])

        if length == 0:
            # Base case: empty lists -- unify unbound lists with []
            new_subst = subst
            for i, el in enumerate(elems_lists):
                if el is None:
                    new_subst = unify(args[i + 1], NIL, new_subst)
                    if new_subst is None:
                        return iter([])
            return self._solve(rest, new_subst, depth + 1)

        # Generate fresh variables for unbound output lists
        for i, el in enumerate(elems_lists):
            if el is None:
                self._var_counter += 1
                fresh_vars = [Var(f"_Map{self._var_counter}_{j}") for j in range(length)]
                elems_lists[i] = fresh_vars
                subst = unify(args[i + 1], make_list(fresh_vars), subst)
                if subst is None:
                    return iter([])

        # Check all lists same length
        lengths = [len(e) for e in elems_lists]
        if len(set(lengths)) != 1:
            return iter([])

        def maplist_gen():
            goals = []
            for i in range(length):
                call_args = [elems_lists[j][i] for j in range(len(elems_lists))]
                if isinstance(pred, Atom):
                    goals.append(Compound(pred.name, call_args))
                elif isinstance(pred, Compound):
                    goals.append(Compound(pred.functor, list(pred.args) + call_args))
                else:
                    return
            yield from self._solve(goals + rest, subst, depth + 1)
        return maplist_gen()

    def _builtin_include(self, args, subst, rest, depth):
        """include(Goal, List, Included)."""
        pred = subst.walk(args[0])
        lst = subst.walk(args[1])
        elems = list_to_python(lst)
        if elems is None:
            return iter([])

        def include_gen():
            included = []
            for elem in elems:
                if isinstance(pred, Atom):
                    goal = Compound(pred.name, [elem])
                elif isinstance(pred, Compound):
                    goal = Compound(pred.functor, list(pred.args) + [elem])
                else:
                    return
                found = False
                for _ in self._solve([goal], subst, depth + 1):
                    found = True
                    break
                if found:
                    included.append(elem)
            s = unify(args[2], make_list(included), subst)
            if s:
                yield from self._solve(rest, s, depth + 1)
        return include_gen()

    def _builtin_foldl(self, args, subst, rest, depth):
        """foldl(Goal, List, V0, V)."""
        pred = subst.walk(args[0])
        lst = subst.walk(args[1])
        elems = list_to_python(lst)
        if elems is None:
            return iter([])

        def foldl_gen():
            acc = subst.walk(args[2])
            current_subst = subst
            for elem in elems:
                self._var_counter += 1
                next_acc = Var(f"_Fold{self._var_counter}")
                if isinstance(pred, Atom):
                    goal = Compound(pred.name, [elem, acc, next_acc])
                elif isinstance(pred, Compound):
                    goal = Compound(pred.functor, list(pred.args) + [elem, acc, next_acc])
                else:
                    return
                found = False
                for s in self._solve([goal], current_subst, depth + 1):
                    acc = s.walk(next_acc)
                    current_subst = s
                    found = True
                    break
                if not found:
                    return
            s = unify(args[3], acc, current_subst)
            if s:
                yield from self._solve(rest, s, depth + 1)
        return foldl_gen()

    # --- CLP(FD) ---

    def _clp_in(self, args, subst, rest, depth):
        """clp_in(Var, Low, High): declare FD variable."""
        var_term = subst.walk(args[0])
        lo = subst.walk(args[1])
        hi = subst.walk(args[2])
        if isinstance(lo, Number) and isinstance(hi, Number):
            var_name = repr(var_term)
            self._clp_vars[var_name] = {
                'term': var_term,
                'low': int(lo.value),
                'high': int(hi.value)
            }
            return self._solve(rest, subst, depth + 1)
        return iter([])

    def _clp_constraint(self, args, subst, rest, depth):
        """clp_constraint(X, Op, Y): add constraint X Op Y."""
        x = subst.walk(args[0])
        op = subst.walk(args[1])
        y = subst.walk(args[2])
        if isinstance(op, Atom):
            self._clp_constraints.append({
                'x': x, 'op': op.name, 'y': y
            })
        return self._solve(rest, subst, depth + 1)

    def _clp_all_different(self, args, subst, rest, depth):
        """clp_all_different(List): all variables must be different."""
        lst = subst.walk(args[0])
        elems = list_to_python(lst)
        if elems is not None:
            self._clp_constraints.append({
                'type': 'alldiff',
                'vars': elems
            })
        return self._solve(rest, subst, depth + 1)

    def _clp_solve(self, args, subst, rest, depth):
        """clp_solve(VarList): solve the CLP problem, bind variables."""
        return self._clp_label(args, subst, rest, depth)

    def _clp_label(self, args, subst, rest, depth):
        """clp_label(VarList): label (solve + bind) the CLP variables."""
        lst = subst.walk(args[0])
        var_terms = list_to_python(lst)
        if var_terms is None:
            return iter([])

        # Build CSP
        solver = CSPSolver()
        var_names = []
        term_to_csp = {}

        for vt in var_terms:
            name = repr(vt)
            info = self._clp_vars.get(name)
            if info:
                domain = range(info['low'], info['high'] + 1)
                solver.add_variable(name, list(domain))
                var_names.append(name)
                term_to_csp[name] = vt

        # Add constraints
        for c in self._clp_constraints:
            if c.get('type') == 'alldiff':
                csp_names = []
                for v in c['vars']:
                    n = repr(v)
                    if n in var_names:
                        csp_names.append(n)
                if len(csp_names) >= 2:
                    solver.add_alldiff(csp_names)
            elif 'op' in c:
                xn = repr(c['x'])
                yn = repr(c['y'])
                op = c['op']
                if xn in var_names and yn in var_names:
                    if op == '=':
                        solver.add_equality(xn, yn)
                    elif op == '\\=':
                        solver.add_inequality(xn, yn)
                    elif op == '<':
                        solver.add_comparison(xn, yn, '<')
                    elif op == '>':
                        solver.add_comparison(xn, yn, '>')
                    elif op == '>=':
                        solver.add_comparison(xn, yn, '>=')
                    elif op == '=<':
                        solver.add_comparison(xn, yn, '<=')
                elif xn in var_names and isinstance(c['y'], Number):
                    val = int(c['y'].value)
                    if op == '=':
                        solver.add_callback([xn], lambda a, v=val, n=xn: a.get(n) == v)
                    elif op == '\\=':
                        solver.add_callback([xn], lambda a, v=val, n=xn: a.get(n) != v)
                    elif op == '<':
                        solver.add_callback([xn], lambda a, v=val, n=xn: a.get(n, v) < v)
                    elif op == '>':
                        solver.add_callback([xn], lambda a, v=val, n=xn: a.get(n, v) > v)

        def label_gen():
            result, assignment = solver.solve()
            if result == CSPResult.SOLVED and assignment:
                new_subst = subst
                for name, val in assignment.items():
                    if name in term_to_csp:
                        vt = term_to_csp[name]
                        new_subst = unify(vt, Number(val), new_subst)
                        if new_subst is None:
                            return
                yield from self._solve(rest, new_subst, depth + 1)
        return label_gen()

    # --- Helpers ---

    def _copy_term(self, term):
        """Copy a term, replacing variables with fresh ones."""
        mapping = {}
        def copy(t):
            if isinstance(t, Var):
                if t.name not in mapping:
                    self._var_counter += 1
                    mapping[t.name] = Var(f"_Copy{self._var_counter}")
                return mapping[t.name]
            if isinstance(t, Compound):
                return Compound(t.functor, [copy(a) for a in t.args])
            return t
        return copy(term)

    def _is_ground(self, term):
        """Check if a term has no variables."""
        if isinstance(term, Var):
            return False
        if isinstance(term, Compound):
            return all(self._is_ground(a) for a in term.args)
        return True

    def _term_to_goals(self, term):
        """Convert a body term to a list of goals."""
        if isinstance(term, Compound) and term.functor == ',' and len(term.args) == 2:
            return self._term_to_goals(term.args[0]) + self._term_to_goals(term.args[1])
        return [term]

    def _goals_to_term(self, goals):
        """Convert a list of goals to a conjunction term."""
        if len(goals) == 1:
            return goals[0]
        return Compound(',', [goals[0], self._goals_to_term(goals[1:])])

    def _format_term(self, term):
        """Format a term for output."""
        if isinstance(term, Atom):
            return term.name
        if isinstance(term, Number):
            v = term.value
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            return str(v)
        if isinstance(term, Var):
            return f"_{term.name}"
        if isinstance(term, Compound):
            if term.functor == '.' and len(term.args) == 2:
                return term._list_repr()
            return repr(term)
        return str(term)

    def _term_sort_key(self, term):
        """Sort key for standard order of terms."""
        if isinstance(term, Number):
            return (0, term.value, '')
        if isinstance(term, Atom):
            return (1, 0, term.name)
        if isinstance(term, Var):
            return (2, 0, term.name)
        if isinstance(term, Compound):
            return (3, len(term.args), term.functor)
        return (4, 0, '')

    def _flatten_list(self, term):
        """Flatten a nested list."""
        if isinstance(term, Atom) and term.name == '[]':
            return []
        if isinstance(term, Compound) and term.functor == '.' and len(term.args) == 2:
            head = term.args[0]
            tail = term.args[1]
            head_elems = list_to_python(head)
            if head_elems is not None:
                flat_head = []
                for e in head_elems:
                    sub = self._flatten_list(e)
                    if sub is not None:
                        flat_head.extend(sub)
                    else:
                        flat_head.append(e)
            else:
                flat_head = [head]
            flat_tail = self._flatten_list(tail)
            if flat_tail is not None:
                return flat_head + flat_tail
        return None

    def reset_clp(self):
        """Reset CLP state."""
        self._clp_vars = {}
        self._clp_constraints = []


class PrologError(Exception):
    """Exception thrown by throw/1."""
    def __init__(self, term):
        self.term = term
        super().__init__(f"Prolog error: {term}")


# ============================================================
# Convenience API
# ============================================================

def run(source, query_str=None):
    """Create an interpreter, load source, optionally run query.
    Returns (interpreter, results) where results is list of dicts mapping var names to terms.
    """
    interp = Interpreter()
    interp.consult(source)

    if query_str is None:
        return interp, []

    goals = parse_query(query_str)
    results = []
    # Collect variable names from query
    query_vars = set()
    for g in goals:
        _collect_vars(g, query_vars)

    for subst in interp.query(goals):
        result = {}
        for v in query_vars:
            val = subst.walk(Var(v))
            result[v] = val
        results.append(result)

    return interp, results


def _collect_vars(term, vars_set):
    """Collect variable names from a term."""
    if isinstance(term, Var):
        if not term.name.startswith('_'):
            vars_set.add(term.name)
    elif isinstance(term, Compound):
        for a in term.args:
            _collect_vars(a, vars_set)
