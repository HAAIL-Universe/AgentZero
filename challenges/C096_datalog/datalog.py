"""
C096: Datalog Engine composing C095 (Logic Programming).

Datalog -- a restricted subset of Prolog with bottom-up evaluation:
- No function symbols (only atoms, numbers, variables as arguments)
- Range-restricted rules (head vars must appear in positive body)
- Bottom-up fixpoint evaluation (naive and semi-naive)
- Stratified negation (negation allowed if stratifiable)
- Aggregation (count, sum, min, max, group_by)
- Safety checking
- Magic sets optimization for goal-directed queries
- Incremental maintenance (add/retract facts, re-evaluate affected strata)
"""

import sys
import os
from collections import defaultdict, OrderedDict
from copy import deepcopy

# Compose C095
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C095_logic_programming'))
from logic_programming import (
    Term, Atom, Number, Var, Compound, Substitution,
    unify, NIL, cons, make_list, list_to_python,
)


# ============================================================
# Datalog Terms (reuse C095 + restrictions)
# ============================================================

def is_ground(term):
    """Check if a term has no variables."""
    if isinstance(term, Var):
        return False
    if isinstance(term, (Atom, Number)):
        return True
    if isinstance(term, Compound):
        return all(is_ground(a) for a in term.args)
    return True


def term_vars(term):
    """Collect all variable names in a term."""
    if isinstance(term, Var):
        return {term.name}
    if isinstance(term, Compound):
        result = set()
        for a in term.args:
            result |= term_vars(a)
        return result
    return set()


def apply_subst(term, subst):
    """Apply substitution to a term, returning ground term."""
    if isinstance(term, Var):
        bound = subst.lookup(term)
        if isinstance(bound, Var):
            return bound
        return bound
    if isinstance(term, (Atom, Number)):
        return term
    if isinstance(term, Compound):
        new_args = tuple(apply_subst(a, subst) for a in term.args)
        return Compound(term.functor, new_args)
    return term


def term_to_tuple(term):
    """Convert a ground term to a hashable tuple for indexing."""
    if isinstance(term, Atom):
        return ('a', term.name)
    if isinstance(term, Number):
        return ('n', term.value)
    if isinstance(term, Var):
        return ('v', term.name)
    if isinstance(term, Compound):
        return ('c', term.functor, tuple(term_to_tuple(a) for a in term.args))
    return ('?',)


def tuple_to_term(t):
    """Convert a tuple back to a term."""
    if t[0] == 'a':
        return Atom(t[1])
    if t[0] == 'n':
        return Number(t[1])
    if t[0] == 'v':
        return Var(t[1])
    if t[0] == 'c':
        return Compound(t[1], [tuple_to_term(a) for a in t[2]])
    return Atom('unknown')


# ============================================================
# Datalog Literals and Rules
# ============================================================

class Literal:
    """A Datalog literal: predicate(arg1, ..., argN) or not predicate(...)."""
    __slots__ = ('predicate', 'args', 'negated')

    def __init__(self, predicate, args, negated=False):
        self.predicate = predicate
        self.args = tuple(args)
        self.negated = negated

    @property
    def arity(self):
        return len(self.args)

    @property
    def indicator(self):
        return f"{self.predicate}/{self.arity}"

    def to_term(self):
        """Convert to C095 term for unification."""
        if self.arity == 0:
            return Atom(self.predicate)
        return Compound(self.predicate, list(self.args))

    def vars(self):
        result = set()
        for a in self.args:
            result |= term_vars(a)
        return result

    def is_ground(self):
        return all(is_ground(a) for a in self.args)

    def __repr__(self):
        prefix = "not " if self.negated else ""
        if self.arity == 0:
            return f"{prefix}{self.predicate}"
        args_str = ", ".join(str(a) for a in self.args)
        return f"{prefix}{self.predicate}({args_str})"

    def __eq__(self, other):
        return (isinstance(other, Literal) and
                self.predicate == other.predicate and
                self.args == other.args and
                self.negated == other.negated)

    def __hash__(self):
        return hash((self.predicate, self.args, self.negated))


class Comparison:
    """A built-in comparison: X op Y (=, !=, <, >, <=, >=)."""
    __slots__ = ('left', 'op', 'right')

    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def vars(self):
        return term_vars(self.left) | term_vars(self.right)

    def evaluate(self, subst):
        """Evaluate comparison under substitution."""
        l = apply_subst(self.left, subst)
        r = apply_subst(self.right, subst)
        lv = _term_value(l)
        rv = _term_value(r)
        if self.op == '=':
            return lv == rv
        elif self.op == '!=':
            return lv != rv
        elif self.op == '<':
            return lv < rv
        elif self.op == '>':
            return lv > rv
        elif self.op == '<=':
            return lv <= rv
        elif self.op == '>=':
            return lv >= rv
        return False

    def __repr__(self):
        return f"{self.left} {self.op} {self.right}"


class Assignment:
    """A built-in assignment: X = expr."""
    __slots__ = ('var', 'expr')

    def __init__(self, var, expr):
        self.var = var  # Var
        self.expr = expr  # arithmetic expression term

    def vars(self):
        return {self.var.name} | term_vars(self.expr)

    def __repr__(self):
        return f"{self.var} := {self.expr}"


class Aggregate:
    """An aggregate literal: X = agg_func(Y, ...) : body_literal."""
    __slots__ = ('result_var', 'func', 'agg_var', 'group_vars', 'body')

    def __init__(self, result_var, func, agg_var, group_vars, body):
        self.result_var = result_var  # Var to bind result
        self.func = func  # 'count', 'sum', 'min', 'max'
        self.agg_var = agg_var  # Var to aggregate over (or None for count)
        self.group_vars = group_vars  # list of Var for grouping
        self.body = body  # Literal to aggregate over

    def vars(self):
        result = {self.result_var.name}
        for gv in self.group_vars:
            result.add(gv.name)
        return result

    def __repr__(self):
        gv = ", ".join(str(v) for v in self.group_vars)
        return f"{self.result_var} = {self.func}({self.agg_var}) : {self.body}"


def _term_value(term):
    """Extract comparable value from a ground term."""
    if isinstance(term, Number):
        return term.value
    if isinstance(term, Atom):
        return term.name
    return str(term)


class Rule:
    """A Datalog rule: head :- body1, body2, ..., bodyN."""

    def __init__(self, head, body=None):
        self.head = head  # Literal
        self.body = body or []  # list of Literal | Comparison | Aggregate | Assignment

    @property
    def is_fact(self):
        return len(self.body) == 0

    def head_vars(self):
        return self.head.vars()

    def body_vars(self):
        result = set()
        for lit in self.body:
            result |= lit.vars()
        return result

    def positive_body_vars(self):
        """Variables appearing in positive (non-negated) body literals."""
        result = set()
        for lit in self.body:
            if isinstance(lit, Literal) and not lit.negated:
                result |= lit.vars()
            elif isinstance(lit, Comparison):
                result |= lit.vars()
            elif isinstance(lit, Assignment):
                result |= lit.vars()
            elif isinstance(lit, Aggregate):
                result |= lit.vars()
        return result

    def is_safe(self):
        """Check range restriction: all head vars must appear in positive body."""
        if self.is_fact:
            return self.head.is_ground()
        hv = self.head_vars()
        pbv = self.positive_body_vars()
        return hv.issubset(pbv)

    def __repr__(self):
        if self.is_fact:
            return f"{self.head}."
        body_str = ", ".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}."


# ============================================================
# Datalog Parser
# ============================================================

class DatalogLexer:
    """Lexer for Datalog programs."""

    KEYWORDS = {'not', 'count', 'sum', 'min', 'max'}

    def __init__(self, source):
        self.source = source
        self.pos = 0
        self.tokens = []

    def tokenize(self):
        while self.pos < len(self.source):
            self._skip_ws_comments()
            if self.pos >= len(self.source):
                break
            ch = self.source[self.pos]

            if ch == '.' and (self.pos + 1 >= len(self.source) or
                              not self.source[self.pos + 1].isdigit()):
                self.tokens.append(('DOT', '.'))
                self.pos += 1
            elif ch == ',' :
                self.tokens.append(('COMMA', ','))
                self.pos += 1
            elif ch == '(':
                self.tokens.append(('LPAREN', '('))
                self.pos += 1
            elif ch == ')':
                self.tokens.append(('RPAREN', ')'))
                self.pos += 1
            elif ch == ':' and self._peek(1) == '-':
                self.tokens.append(('NECK', ':-'))
                self.pos += 2
            elif ch == ':' and self._peek(1) == '=':
                self.tokens.append(('ASSIGN', ':='))
                self.pos += 2
            elif ch == ':' and self._peek(1) != '-':
                self.tokens.append(('COLON', ':'))
                self.pos += 1
            elif ch == '?' and self._peek(1) == '-':
                self.tokens.append(('QUERY', '?-'))
                self.pos += 2
            elif ch == '!' and self._peek(1) == '=':
                self.tokens.append(('OP', '!='))
                self.pos += 2
            elif ch == '<' and self._peek(1) == '=':
                self.tokens.append(('OP', '<='))
                self.pos += 2
            elif ch == '>' and self._peek(1) == '=':
                self.tokens.append(('OP', '>='))
                self.pos += 2
            elif ch == ':' and self._peek(1) == '=':
                self.tokens.append(('ASSIGN', ':='))
                self.pos += 2
            elif ch == '=':
                self.tokens.append(('OP', '='))
                self.pos += 1
            elif ch == '<':
                self.tokens.append(('OP', '<'))
                self.pos += 1
            elif ch == '>':
                self.tokens.append(('OP', '>'))
                self.pos += 1
            elif ch == '+':
                self.tokens.append(('OP', '+'))
                self.pos += 1
            elif ch == '-':
                # Check if negative number
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit():
                    self.pos += 1
                    num = self._read_number()
                    self.tokens.append(('NUMBER', -num))
                else:
                    self.tokens.append(('OP', '-'))
                    self.pos += 1
            elif ch == '*':
                self.tokens.append(('OP', '*'))
                self.pos += 1
            elif ch == '/':
                self.tokens.append(('OP', '/'))
                self.pos += 1
            elif ch == '"' or ch == "'":
                self.tokens.append(('ATOM', self._read_string(ch)))
            elif ch.isdigit():
                self.tokens.append(('NUMBER', self._read_number()))
            elif ch.isupper() or ch == '_':
                name = self._read_name()
                if name == '_':
                    self.tokens.append(('WILDCARD', '_'))
                else:
                    self.tokens.append(('VAR', name))
            elif ch.islower():
                name = self._read_name()
                if name in ('not',):
                    self.tokens.append(('NOT', name))
                elif name in ('count', 'sum', 'min', 'max'):
                    self.tokens.append(('AGG', name))
                else:
                    self.tokens.append(('ATOM', name))
            else:
                raise SyntaxError(f"Unexpected character: {ch!r} at position {self.pos}")

        self.tokens.append(('EOF', None))
        return self.tokens

    def _peek(self, offset=1):
        p = self.pos + offset
        return self.source[p] if p < len(self.source) else ''

    def _skip_ws_comments(self):
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in ' \t\r\n':
                self.pos += 1
            elif ch == '%':
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self.pos += 1
            elif ch == '/' and self._peek(1) == '*':
                self.pos += 2
                while self.pos + 1 < len(self.source):
                    if self.source[self.pos] == '*' and self.source[self.pos + 1] == '/':
                        self.pos += 2
                        break
                    self.pos += 1
            else:
                break

    def _read_name(self):
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self.pos += 1
        return self.source[start:self.pos]

    def _read_number(self):
        start = self.pos
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1
        if (self.pos < len(self.source) and self.source[self.pos] == '.' and
                self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
            self.pos += 1
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
            return float(self.source[start:self.pos])
        return int(self.source[start:self.pos])

    def _read_string(self, quote):
        self.pos += 1  # skip opening quote
        result = []
        while self.pos < len(self.source) and self.source[self.pos] != quote:
            if self.source[self.pos] == '\\':
                self.pos += 1
                if self.pos < len(self.source):
                    result.append(self.source[self.pos])
            else:
                result.append(self.source[self.pos])
            self.pos += 1
        self.pos += 1  # skip closing quote
        return ''.join(result)


class DatalogParser:
    """Parser for Datalog programs."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self._wildcard_counter = 0

    def current(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', None)

    def peek(self, offset=1):
        p = self.pos + offset
        return self.tokens[p] if p < len(self.tokens) else ('EOF', None)

    def consume(self, expected_type=None, expected_value=None):
        tok = self.current()
        if expected_type and tok[0] != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok[0]} ({tok[1]!r})")
        if expected_value is not None and tok[1] != expected_value:
            raise SyntaxError(f"Expected {expected_value!r}, got {tok[1]!r}")
        self.pos += 1
        return tok

    def parse_program(self):
        """Parse a Datalog program into rules and queries."""
        rules = []
        queries = []
        while self.current()[0] != 'EOF':
            if self.current()[0] == 'QUERY':
                self.consume('QUERY')
                q = self._parse_body()
                self.consume('DOT')
                queries.append(q)
            else:
                rules.append(self._parse_rule())
        return rules, queries

    def _parse_rule(self):
        """Parse a rule or fact."""
        head = self._parse_literal()
        if self.current()[0] == 'NECK':
            self.consume('NECK')
            body = self._parse_body()
            self.consume('DOT')
            return Rule(head, body)
        else:
            self.consume('DOT')
            return Rule(head, [])

    def _parse_body(self):
        """Parse a comma-separated list of body literals."""
        body = [self._parse_body_element()]
        while self.current()[0] == 'COMMA':
            self.consume('COMMA')
            body.append(self._parse_body_element())
        return body

    def _parse_body_element(self):
        """Parse a single body element: literal, negated literal, comparison, aggregate, or assignment."""
        # Negation
        if self.current()[0] == 'NOT':
            self.consume('NOT')
            lit = self._parse_literal()
            lit.negated = True
            return lit

        # Aggregate: Var = agg_func(Var) : literal
        if (self.current()[0] == 'VAR' and
            self.peek()[0] == 'OP' and self.peek()[1] == '=' and
            self.peek(2)[0] == 'AGG'):
            return self._parse_aggregate()

        # Assignment: Var := expr
        if (self.current()[0] == 'VAR' and self.peek()[0] == 'ASSIGN'):
            var_name = self.consume('VAR')[1]
            self.consume('ASSIGN')
            expr = self._parse_arith_expr()
            return Assignment(Var(var_name), expr)

        # Check for comparison: term op term
        # We need lookahead to distinguish comparison from literal
        saved_pos = self.pos
        try:
            term1 = self._parse_term()
            if self.current()[0] == 'OP' and self.current()[1] in ('=', '!=', '<', '>', '<=', '>='):
                op = self.consume('OP')[1]
                term2 = self._parse_term()
                return Comparison(term1, op, term2)
            # Not a comparison -- backtrack
            self.pos = saved_pos
        except (SyntaxError, IndexError):
            self.pos = saved_pos

        # Regular literal
        return self._parse_literal()

    def _parse_literal(self):
        """Parse a predicate literal: name or name(args)."""
        tok = self.current()
        if tok[0] == 'ATOM':
            name = self.consume('ATOM')[1]
        elif tok[0] == 'AGG':
            # Allow agg keywords as predicate names
            name = self.consume('AGG')[1]
        else:
            raise SyntaxError(f"Expected predicate name, got {tok[0]} ({tok[1]!r})")

        if self.current()[0] == 'LPAREN':
            self.consume('LPAREN')
            args = []
            if self.current()[0] != 'RPAREN':
                args.append(self._parse_term())
                while self.current()[0] == 'COMMA':
                    self.consume('COMMA')
                    args.append(self._parse_term())
            self.consume('RPAREN')
            return Literal(name, args)
        else:
            return Literal(name, [])

    def _parse_term(self):
        """Parse a single term: variable, atom, number, or string."""
        tok = self.current()
        if tok[0] == 'VAR':
            return Var(self.consume('VAR')[1])
        elif tok[0] == 'WILDCARD':
            self.consume('WILDCARD')
            self._wildcard_counter += 1
            return Var(f'_W{self._wildcard_counter}')
        elif tok[0] == 'NUMBER':
            return Number(self.consume('NUMBER')[1])
        elif tok[0] == 'ATOM':
            return Atom(self.consume('ATOM')[1])
        elif tok[0] == 'AGG':
            # Allow agg func names as atoms
            return Atom(self.consume('AGG')[1])
        else:
            raise SyntaxError(f"Expected term, got {tok[0]} ({tok[1]!r})")

    def _parse_arith_expr(self):
        """Parse an arithmetic expression."""
        left = self._parse_arith_factor()
        while self.current()[0] == 'OP' and self.current()[1] in ('+', '-'):
            op = self.consume('OP')[1]
            right = self._parse_arith_factor()
            left = Compound(op, [left, right])
        return left

    def _parse_arith_factor(self):
        """Parse arithmetic factor (multiplication/division)."""
        left = self._parse_arith_primary()
        while self.current()[0] == 'OP' and self.current()[1] in ('*', '/'):
            op = self.consume('OP')[1]
            right = self._parse_arith_primary()
            left = Compound(op, [left, right])
        return left

    def _parse_arith_primary(self):
        """Parse primary arithmetic value."""
        tok = self.current()
        if tok[0] == 'VAR':
            return Var(self.consume('VAR')[1])
        elif tok[0] == 'NUMBER':
            return Number(self.consume('NUMBER')[1])
        elif tok[0] == 'LPAREN':
            self.consume('LPAREN')
            expr = self._parse_arith_expr()
            self.consume('RPAREN')
            return expr
        else:
            raise SyntaxError(f"Expected arithmetic value, got {tok[0]} ({tok[1]!r})")

    def _parse_aggregate(self):
        """Parse aggregate: ResultVar = func(AggVar) : body_literal."""
        result_name = self.consume('VAR')[1]
        self.consume('OP', '=')
        func = self.consume('AGG')[1]
        self.consume('LPAREN')
        if self.current()[0] == 'RPAREN':
            # count() -- no specific variable
            agg_var = None
            self.consume('RPAREN')
        else:
            agg_var = Var(self.consume('VAR')[1])
            self.consume('RPAREN')
        self.consume('COLON')
        body = self._parse_literal()

        # Determine group vars: vars in body that are NOT the agg var
        group_vars = []
        for v_name in body.vars():
            if agg_var is None or v_name != agg_var.name:
                group_vars.append(Var(v_name))
        # Sort for deterministic grouping
        group_vars.sort(key=lambda v: v.name)

        return Aggregate(Var(result_name), func, agg_var, group_vars, body)


def parse_datalog(source):
    """Parse a Datalog source string into rules and queries."""
    lexer = DatalogLexer(source)
    tokens = lexer.tokenize()
    parser = DatalogParser(tokens)
    return parser.parse_program()


# ============================================================
# Relation (in-memory table of ground tuples)
# ============================================================

class Relation:
    """A named relation storing ground tuples."""

    def __init__(self, name, arity):
        self.name = name
        self.arity = arity
        self._tuples = set()  # set of tuple-of-tuples (hashable term representations)
        self._index = {}  # column_idx -> value -> set of tuples (for fast lookups)

    def add(self, tup):
        """Add a ground tuple. Returns True if new."""
        key = tuple(term_to_tuple(t) for t in tup)
        if key in self._tuples:
            return False
        self._tuples.add(key)
        # Update indexes
        for i, t in enumerate(tup):
            tk = term_to_tuple(t)
            if i not in self._index:
                self._index[i] = defaultdict(set)
            self._index[i][tk].add(key)
        return True

    def remove(self, tup):
        """Remove a ground tuple. Returns True if existed."""
        key = tuple(term_to_tuple(t) for t in tup)
        if key not in self._tuples:
            return False
        self._tuples.discard(key)
        for i, t in enumerate(tup):
            tk = term_to_tuple(t)
            if i in self._index and tk in self._index[i]:
                self._index[i][tk].discard(key)
        return True

    def contains(self, tup):
        """Check if a ground tuple exists."""
        key = tuple(term_to_tuple(t) for t in tup)
        return key in self._tuples

    def all_tuples(self):
        """Yield all tuples as lists of terms."""
        for key in self._tuples:
            yield [tuple_to_term(t) for t in key]

    def size(self):
        return len(self._tuples)

    def clear(self):
        self._tuples.clear()
        self._index.clear()

    def copy(self):
        r = Relation(self.name, self.arity)
        r._tuples = set(self._tuples)
        r._index = {i: defaultdict(set, {k: set(v) for k, v in idx.items()})
                     for i, idx in self._index.items()}
        return r

    def __repr__(self):
        return f"Relation({self.name}/{self.arity}, {self.size()} tuples)"


# ============================================================
# Stratification
# ============================================================

class StratificationError(Exception):
    """Raised when program cannot be stratified (negation cycle)."""
    pass


def compute_dependency_graph(rules):
    """Build predicate dependency graph from rules.
    Returns (edges, neg_edges) where each is {pred -> set of preds}.
    """
    edges = defaultdict(set)  # positive dependencies
    neg_edges = defaultdict(set)  # negative dependencies

    for rule in rules:
        head_pred = rule.head.indicator
        for lit in rule.body:
            if isinstance(lit, Literal):
                dep = lit.indicator
                if lit.negated:
                    neg_edges[head_pred].add(dep)
                else:
                    edges[head_pred].add(dep)
            elif isinstance(lit, Aggregate):
                dep = lit.body.indicator
                # Aggregates are treated like negation for stratification
                neg_edges[head_pred].add(dep)

    return edges, neg_edges


def compute_strata(rules):
    """Compute stratification of rules.
    Returns list of strata (each stratum is a set of predicate indicators).
    Raises StratificationError if not stratifiable.
    """
    edges, neg_edges = compute_dependency_graph(rules)

    # Collect all predicates
    all_preds = set()
    for rule in rules:
        all_preds.add(rule.head.indicator)
        for lit in rule.body:
            if isinstance(lit, Literal):
                all_preds.add(lit.indicator)
            elif isinstance(lit, Aggregate):
                all_preds.add(lit.body.indicator)

    # Assign strata: iterative fixpoint
    stratum = {p: 0 for p in all_preds}
    changed = True
    max_iter = len(all_preds) + 1

    for _ in range(max_iter):
        if not changed:
            break
        changed = False
        for p in all_preds:
            # Positive deps: same stratum or lower
            for dep in edges.get(p, set()):
                if stratum[dep] > stratum[p]:
                    stratum[p] = stratum[dep]
                    changed = True
            # Negative deps: strictly higher
            for dep in neg_edges.get(p, set()):
                if stratum[dep] >= stratum[p]:
                    stratum[p] = stratum[dep] + 1
                    changed = True
                    if stratum[p] > len(all_preds):
                        raise StratificationError(
                            f"Cannot stratify: negation cycle involving {p} and {dep}")
    else:
        if changed:
            raise StratificationError("Cannot stratify: possible negation cycle")

    # Group predicates by stratum
    strata_groups = defaultdict(set)
    for p, s in stratum.items():
        strata_groups[s].add(p)

    # Return ordered list of strata
    max_stratum = max(stratum.values()) if stratum else 0
    result = []
    for s in range(max_stratum + 1):
        if s in strata_groups:
            result.append(strata_groups[s])
    return result


# ============================================================
# Evaluation Engine
# ============================================================

class DatalogEngine:
    """Bottom-up Datalog evaluation engine with stratified negation."""

    def __init__(self):
        self.rules = []  # All rules (including facts)
        self.relations = {}  # predicate indicator -> Relation
        self.output = []  # Output from write predicates

    def add_rule(self, rule):
        """Add a rule to the program."""
        # Fix aggregate group_vars: group by vars that appear in head
        head_vars = rule.head.vars()
        for element in rule.body:
            if isinstance(element, Aggregate):
                # Group vars = vars in agg body that also appear in rule head
                body_vars = element.body.vars()
                agg_var_name = element.agg_var.name if element.agg_var else None
                element.group_vars = [
                    Var(v) for v in sorted(body_vars & head_vars)
                    if v != agg_var_name
                ]

        self.rules.append(rule)
        # Ensure relation exists
        ind = rule.head.indicator
        if ind not in self.relations:
            self.relations[ind] = Relation(rule.head.predicate, rule.head.arity)

    def add_fact(self, predicate, args):
        """Add a ground fact directly."""
        lit = Literal(predicate, args)
        rule = Rule(lit, [])
        self.add_rule(rule)

    def add_rules(self, rules):
        """Add multiple rules."""
        for r in rules:
            self.add_rule(r)

    def load(self, source):
        """Parse and load a Datalog program."""
        rules, queries = parse_datalog(source)
        self.add_rules(rules)
        return queries

    def get_relation(self, predicate, arity):
        """Get or create a relation."""
        ind = f"{predicate}/{arity}"
        if ind not in self.relations:
            self.relations[ind] = Relation(predicate, arity)
        return self.relations[ind]

    def fact_count(self):
        """Total number of derived facts across all relations."""
        return sum(r.size() for r in self.relations.values())

    def facts(self, predicate, arity=None):
        """Get all facts for a predicate. Returns list of tuples."""
        if arity is not None:
            ind = f"{predicate}/{arity}"
            rel = self.relations.get(ind)
            if rel:
                return [tuple(t) for t in rel.all_tuples()]
            return []
        # Search all arities
        result = []
        for ind, rel in self.relations.items():
            if ind.startswith(predicate + '/'):
                result.extend(tuple(t) for t in rel.all_tuples())
        return result

    def check_safety(self):
        """Check all rules are safe (range-restricted)."""
        unsafe = []
        for rule in self.rules:
            if not rule.is_safe():
                unsafe.append(rule)
        return unsafe

    def evaluate(self, mode='semi_naive'):
        """Evaluate all rules to fixpoint using stratified evaluation."""
        # Load base facts
        self._load_facts()

        # Compute strata
        strata = compute_strata(self.rules)

        # Evaluate each stratum
        for stratum_preds in strata:
            stratum_rules = [r for r in self.rules if not r.is_fact
                             and r.head.indicator in stratum_preds]
            if not stratum_rules:
                continue

            if mode == 'naive':
                self._evaluate_naive(stratum_rules)
            else:
                self._evaluate_semi_naive(stratum_rules)

    def _load_facts(self):
        """Load all ground facts into relations."""
        for rule in self.rules:
            if rule.is_fact:
                rel = self.get_relation(rule.head.predicate, rule.head.arity)
                rel.add(list(rule.head.args))

    def _evaluate_naive(self, rules):
        """Naive fixpoint evaluation: apply all rules until no new facts."""
        changed = True
        while changed:
            changed = False
            for rule in rules:
                new_facts = self._apply_rule(rule)
                for fact in new_facts:
                    rel = self.get_relation(rule.head.predicate, rule.head.arity)
                    if rel.add(fact):
                        changed = True

    def _evaluate_semi_naive(self, rules):
        """Semi-naive evaluation: only consider new facts from previous iteration."""
        # First pass: compute initial delta (all current facts are "new")
        delta = {}  # pred indicator -> Relation (new facts from last iteration)
        for rule in rules:
            ind = rule.head.indicator
            if ind not in delta:
                delta[ind] = Relation(rule.head.predicate, rule.head.arity)

        # Initial delta = results of first application
        for rule in rules:
            new_facts = self._apply_rule(rule)
            for fact in new_facts:
                rel = self.get_relation(rule.head.predicate, rule.head.arity)
                ind = rule.head.indicator
                if rel.add(fact):
                    delta[ind].add(fact)

        # Iterate until no new deltas
        while any(d.size() > 0 for d in delta.values()):
            new_delta = {}
            for rule in rules:
                ind = rule.head.indicator
                if ind not in new_delta:
                    new_delta[ind] = Relation(rule.head.predicate, rule.head.arity)

            for rule in rules:
                # Apply rule but only where at least one body literal
                # uses a delta relation
                new_facts = self._apply_rule_with_delta(rule, delta)
                for fact in new_facts:
                    rel = self.get_relation(rule.head.predicate, rule.head.arity)
                    ind = rule.head.indicator
                    if rel.add(fact):
                        new_delta[ind].add(fact)

            delta = new_delta

    def _apply_rule(self, rule):
        """Apply a rule and return list of new derived facts (as lists of terms)."""
        results = []
        for subst in self._match_body(rule.body, Substitution()):
            # Build head with substitution
            head_args = [apply_subst(a, subst) for a in rule.head.args]
            if all(is_ground(a) for a in head_args):
                results.append(head_args)
        return results

    def _apply_rule_with_delta(self, rule, delta):
        """Apply rule requiring at least one body literal to use delta facts."""
        results = []
        positive_lits = [(i, lit) for i, lit in enumerate(rule.body)
                         if isinstance(lit, Literal) and not lit.negated]

        if not positive_lits:
            return results

        # For each positive literal position, try matching with delta
        seen = set()
        for delta_idx, delta_lit in positive_lits:
            ind = delta_lit.indicator
            if ind not in delta or delta.get(ind).size() == 0:
                continue
            for subst in self._match_body_with_delta(rule.body, Substitution(),
                                                      delta, delta_idx):
                head_args = [apply_subst(a, subst) for a in rule.head.args]
                if all(is_ground(a) for a in head_args):
                    key = tuple(term_to_tuple(a) for a in head_args)
                    if key not in seen:
                        seen.add(key)
                        results.append(head_args)
        return results

    def _match_body(self, body, subst, idx=0):
        """Match body literals against current relations. Yields substitutions."""
        if idx >= len(body):
            yield subst
            return

        element = body[idx]

        if isinstance(element, Literal):
            if element.negated:
                # Negation as failure: succeed if no match
                has_match = False
                for _ in self._match_literal(element, subst):
                    has_match = True
                    break
                if not has_match:
                    yield from self._match_body(body, subst, idx + 1)
            else:
                for new_subst in self._match_literal(element, subst):
                    yield from self._match_body(body, new_subst, idx + 1)

        elif isinstance(element, Comparison):
            # All comparison vars should be bound at this point
            if element.evaluate(subst):
                yield from self._match_body(body, subst, idx + 1)

        elif isinstance(element, Assignment):
            val = self._eval_arith(element.expr, subst)
            if val is not None:
                new_subst = subst.bind(element.var.name, Number(val))
                yield from self._match_body(body, new_subst, idx + 1)

        elif isinstance(element, Aggregate):
            yield from self._match_aggregate(element, body, subst, idx)

    def _match_body_with_delta(self, body, subst, delta, delta_idx, idx=0):
        """Like _match_body but uses delta for the literal at delta_idx."""
        if idx >= len(body):
            yield subst
            return

        element = body[idx]

        if isinstance(element, Literal):
            if element.negated:
                has_match = False
                for _ in self._match_literal(element, subst):
                    has_match = True
                    break
                if not has_match:
                    yield from self._match_body_with_delta(body, subst, delta, delta_idx, idx + 1)
            elif idx == delta_idx:
                # Use delta relation instead of full relation
                for new_subst in self._match_literal_in_relation(element, subst, delta.get(element.indicator)):
                    yield from self._match_body_with_delta(body, new_subst, delta, delta_idx, idx + 1)
            else:
                for new_subst in self._match_literal(element, subst):
                    yield from self._match_body_with_delta(body, new_subst, delta, delta_idx, idx + 1)

        elif isinstance(element, Comparison):
            if element.evaluate(subst):
                yield from self._match_body_with_delta(body, subst, delta, delta_idx, idx + 1)

        elif isinstance(element, Assignment):
            val = self._eval_arith(element.expr, subst)
            if val is not None:
                new_subst = subst.bind(element.var.name, Number(val))
                yield from self._match_body_with_delta(body, new_subst, delta, delta_idx, idx + 1)

        elif isinstance(element, Aggregate):
            yield from self._match_aggregate(element, body, subst, idx,
                                              delta=delta, delta_idx=delta_idx)

    def _match_literal(self, literal, subst):
        """Match a literal against the current relation. Yields substitutions."""
        ind = literal.indicator
        rel = self.relations.get(ind)
        if rel is None:
            return
        yield from self._match_literal_in_relation(literal, subst, rel)

    def _match_literal_in_relation(self, literal, subst, rel):
        """Match a literal against a specific relation."""
        if rel is None:
            return
        applied_args = [apply_subst(a, subst) for a in literal.args]

        for tup in rel.all_tuples():
            new_subst = subst
            match = True
            for arg, val in zip(applied_args, tup):
                result = unify(arg, val, new_subst, check_occurs=False)
                if result is None:
                    match = False
                    break
                new_subst = result
            if match:
                yield new_subst

    def _match_aggregate(self, agg, body, subst, idx, delta=None, delta_idx=None):
        """Evaluate an aggregate and continue matching."""
        ind = agg.body.indicator
        rel = self.relations.get(ind)
        if rel is None:
            return

        # Determine which group vars are already bound vs unbound
        bound_group_vars = []
        unbound_group_vars = []
        for gv in agg.group_vars:
            val = apply_subst(gv, subst)
            if not isinstance(val, Var):
                bound_group_vars.append(gv)
            else:
                unbound_group_vars.append(gv)

        # All group vars (bound + unbound) form the group key
        all_group_vars = bound_group_vars + unbound_group_vars

        groups = defaultdict(list)
        for tup in rel.all_tuples():
            match_subst = subst
            match = True
            applied_args = [apply_subst(a, match_subst) for a in agg.body.args]
            for arg, val in zip(applied_args, tup):
                result = unify(arg, val, match_subst, check_occurs=False)
                if result is None:
                    match = False
                    break
                match_subst = result
            if not match:
                continue

            # Group key uses all group vars
            group_key = tuple(
                term_to_tuple(apply_subst(gv, match_subst))
                for gv in all_group_vars
            )

            if agg.agg_var is not None:
                agg_val = apply_subst(agg.agg_var, match_subst)
                groups[group_key].append(agg_val)
            else:
                groups[group_key].append(None)

        # If no matches, and we're counting, emit 0
        if not groups:
            if agg.func == 'count':
                new_subst = subst.bind(agg.result_var.name, Number(0))
                if delta is not None:
                    yield from self._match_body_with_delta(body, new_subst, delta, delta_idx, idx + 1)
                else:
                    yield from self._match_body(body, new_subst, idx + 1)
            return

        # Compute aggregate for each group
        for group_key, values in groups.items():
            new_subst = subst
            for gv, gk in zip(all_group_vars, group_key):
                new_subst = new_subst.bind(gv.name, tuple_to_term(gk))

            if agg.func == 'count':
                result_val = Number(len(values))
            elif agg.func == 'sum':
                total = sum(_term_value(v) for v in values if isinstance(v, Number))
                result_val = Number(total)
            elif agg.func == 'min':
                nums = [_term_value(v) for v in values if isinstance(v, Number)]
                if not nums:
                    continue
                result_val = Number(min(nums))
            elif agg.func == 'max':
                nums = [_term_value(v) for v in values if isinstance(v, Number)]
                if not nums:
                    continue
                result_val = Number(max(nums))
            else:
                continue

            new_subst = new_subst.bind(agg.result_var.name, result_val)
            if delta is not None:
                yield from self._match_body_with_delta(body, new_subst, delta, delta_idx, idx + 1)
            else:
                yield from self._match_body(body, new_subst, idx + 1)

    def _eval_arith(self, expr, subst):
        """Evaluate an arithmetic expression under substitution."""
        expr = apply_subst(expr, subst)
        if isinstance(expr, Number):
            return expr.value
        if isinstance(expr, Var):
            return None  # Unbound
        if isinstance(expr, Compound):
            left = self._eval_arith(Compound('dummy', [expr.args[0]]).args[0], subst) if len(expr.args) > 0 else None
            right = self._eval_arith(Compound('dummy', [expr.args[1]]).args[0], subst) if len(expr.args) > 1 else None
            # Actually recurse properly
            left_val = self._eval_arith(expr.args[0], subst)
            right_val = self._eval_arith(expr.args[1], subst) if len(expr.args) > 1 else None
            if left_val is None:
                return None
            if expr.functor == '+' and right_val is not None:
                return left_val + right_val
            if expr.functor == '-' and right_val is not None:
                return left_val - right_val
            if expr.functor == '*' and right_val is not None:
                return left_val * right_val
            if expr.functor == '/' and right_val is not None and right_val != 0:
                return left_val / right_val
        return None

    # ========================================================
    # Query interface
    # ========================================================

    def query(self, goals):
        """Query the evaluated database. Returns list of result dicts."""
        if not goals:
            return [{}]

        results = []
        # Collect variable names from goals
        query_vars = set()
        for g in goals:
            query_vars |= g.vars()

        for subst in self._match_body(goals, Substitution()):
            result = {}
            for vn in query_vars:
                val = subst.lookup(Var(vn))
                if not isinstance(val, Var):
                    result[vn] = val
            results.append(result)

        return results

    def query_string(self, query_str):
        """Parse and execute a query string."""
        # Parse as a body (comma-separated literals)
        lexer = DatalogLexer(query_str.rstrip('.'))
        tokens = lexer.tokenize()
        parser = DatalogParser(tokens)
        body = parser._parse_body()
        return self.query(body)

    # ========================================================
    # Incremental maintenance
    # ========================================================

    def add_fact_incremental(self, predicate, args):
        """Add a fact and incrementally re-evaluate affected rules."""
        rel = self.get_relation(predicate, len(args))
        if not rel.add(args):
            return False  # Already existed

        # Find rules that depend on this predicate
        affected = self._rules_depending_on(f"{predicate}/{len(args)}")
        if affected:
            # Re-evaluate affected strata
            self._evaluate_semi_naive(affected)
        return True

    def retract_fact(self, predicate, args):
        """Retract a fact and re-evaluate from scratch (simple approach)."""
        rel = self.relations.get(f"{predicate}/{len(args)}")
        if rel is None or not rel.remove(args):
            return False

        # Remove matching fact rule from self.rules
        args_tuple = tuple(term_to_tuple(a) for a in args)
        self.rules = [
            r for r in self.rules
            if not (r.is_fact and r.head.predicate == predicate and
                    r.head.arity == len(args) and
                    tuple(term_to_tuple(a) for a in r.head.args) == args_tuple)
        ]

        # Clear all relations and re-evaluate from scratch
        for r in self.relations.values():
            r.clear()
        self._load_facts()
        strata = compute_strata(self.rules)
        for stratum_preds in strata:
            stratum_rules = [r for r in self.rules if not r.is_fact
                             and r.head.indicator in stratum_preds]
            if stratum_rules:
                self._evaluate_semi_naive(stratum_rules)
        return True

    def _clear_derived(self):
        """Clear all derived (non-fact) relations."""
        fact_preds = set()
        for rule in self.rules:
            if rule.is_fact:
                fact_preds.add(rule.head.indicator)

        derived_preds = set()
        for rule in self.rules:
            if not rule.is_fact:
                derived_preds.add(rule.head.indicator)

        for pred in derived_preds:
            if pred in self.relations and pred not in fact_preds:
                self.relations[pred].clear()
            elif pred in self.relations:
                # Has both facts and derived -- clear derived by rebuilding from facts
                rel = self.relations[pred]
                rel.clear()

    def _rules_depending_on(self, indicator):
        """Find rules whose body references the given predicate."""
        result = []
        for rule in self.rules:
            if rule.is_fact:
                continue
            for lit in rule.body:
                if isinstance(lit, Literal) and lit.indicator == indicator:
                    result.append(rule)
                    break
                elif isinstance(lit, Aggregate) and lit.body.indicator == indicator:
                    result.append(rule)
                    break
        return result

    # ========================================================
    # Magic Sets Optimization
    # ========================================================

    def magic_sets_rewrite(self, query_pred, query_arity, bound_args=None):
        """Apply magic sets transformation for goal-directed evaluation.

        Rewrites rules so that only facts reachable from the query are derived.
        bound_args: list of booleans indicating which query args are bound.
        """
        if bound_args is None:
            bound_args = [False] * query_arity

        adornment = ''.join('b' if b else 'f' for b in bound_args)
        magic_pred = f"magic_{query_pred}_{adornment}"
        ind = f"{query_pred}/{query_arity}"

        new_rules = []
        # Keep all original facts
        for rule in self.rules:
            if rule.is_fact:
                new_rules.append(rule)

        # Find rules defining query_pred
        target_rules = [r for r in self.rules if r.head.indicator == ind and not r.is_fact]

        if not target_rules:
            return  # No rules to rewrite

        # Create magic seed fact
        seed_args = []
        for i, b in enumerate(bound_args):
            if b:
                seed_args.append(Var(f'_bound_{i}'))
            else:
                seed_args.append(Var(f'_free_{i}'))

        # Add magic rules
        for rule in target_rules:
            # Add magic predicate to rule body
            magic_args = [rule.head.args[i] for i, b in enumerate(bound_args) if b]
            if magic_args:
                magic_lit = Literal(magic_pred, magic_args)
                new_body = [magic_lit] + list(rule.body)
                new_rules.append(Rule(rule.head, new_body))
            else:
                new_rules.append(rule)

            # Generate magic rules for body predicates
            for lit in rule.body:
                if isinstance(lit, Literal) and not lit.negated:
                    lit_ind = lit.indicator
                    sub_rules = [r for r in self.rules if r.head.indicator == lit_ind and not r.is_fact]
                    if sub_rules:
                        # Create magic fact propagation rule
                        sub_magic = f"magic_{lit.predicate}_{'f' * lit.arity}"
                        propagation_body = [magic_lit] if magic_args else []
                        # Add preceding body literals
                        for prev_lit in rule.body:
                            if prev_lit is lit:
                                break
                            propagation_body.append(prev_lit)
                        if propagation_body:
                            prop_head = Literal(sub_magic, [])
                            new_rules.append(Rule(prop_head, propagation_body))

        # Add non-target rules unchanged
        for rule in self.rules:
            if not rule.is_fact and rule.head.indicator != ind:
                new_rules.append(rule)

        self.rules = new_rules

    # ========================================================
    # Convenience: consult and query
    # ========================================================

    def consult_and_query(self, source, query_str=None):
        """Load program, evaluate, and optionally query."""
        queries = self.load(source)
        unsafe = self.check_safety()
        if unsafe:
            raise ValueError(f"Unsafe rules detected: {unsafe}")
        self.evaluate()

        if query_str:
            return self.query_string(query_str)
        elif queries:
            # Execute embedded queries
            all_results = []
            for q_goals in queries:
                all_results.append(self.query(q_goals))
            return all_results
        return []

    def __repr__(self):
        n_rules = len(self.rules)
        n_facts = sum(1 for r in self.rules if r.is_fact)
        n_derived = self.fact_count()
        return f"DatalogEngine({n_rules} rules, {n_facts} base facts, {n_derived} total facts)"
