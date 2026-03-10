"""
C015: In-Memory Graph Database with Query Language
===================================================
An in-memory property graph database supporting:
- Nodes with labels and properties
- Edges with types and properties
- A query language for traversals, filtering, and pattern matching
- Indexes for fast property lookups
- Transaction-like snapshots (save/restore)

Query language (GQL - Graph Query Language):
  MATCH (n:Label {prop: value}) RETURN n
  MATCH (a)-[:EDGE_TYPE]->(b) RETURN a, b
  MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 30 RETURN a.name, b.name
  CREATE (n:Label {prop: value})
  CREATE (a)-[:TYPE {prop: value}]->(b)  -- a, b are node IDs
  DELETE node <id>
  DELETE edge <id>
  SET node <id> prop = value
  SET edge <id> prop = value
"""

from __future__ import annotations
import json
import copy
import re
from typing import Any


# ============================================================
# Core Data Model
# ============================================================

class Node:
    """A node in the property graph."""
    __slots__ = ('id', 'labels', 'props')

    def __init__(self, node_id: int, labels: set[str] = None, props: dict[str, Any] = None):
        self.id = node_id
        self.labels = labels or set()
        self.props = props or {}

    def __repr__(self):
        return f"Node({self.id}, {self.labels}, {self.props})"

    def to_dict(self):
        return {'id': self.id, 'labels': sorted(self.labels), 'props': dict(self.props)}


class Edge:
    """A directed edge in the property graph."""
    __slots__ = ('id', 'src', 'dst', 'edge_type', 'props')

    def __init__(self, edge_id: int, src: int, dst: int, edge_type: str, props: dict[str, Any] = None):
        self.id = edge_id
        self.src = src
        self.dst = dst
        self.edge_type = edge_type
        self.props = props or {}

    def __repr__(self):
        return f"Edge({self.id}, {self.src}-[:{self.edge_type}]->{self.dst}, {self.props})"

    def to_dict(self):
        return {'id': self.id, 'src': self.src, 'dst': self.dst,
                'type': self.edge_type, 'props': dict(self.props)}


# ============================================================
# Indexes
# ============================================================

class PropertyIndex:
    """Index on (label, property_name) -> {value: set of node_ids}."""

    def __init__(self):
        self._data: dict[tuple[str, str], dict[Any, set[int]]] = {}

    def add_index(self, label: str, prop: str):
        key = (label, prop)
        if key not in self._data:
            self._data[key] = {}

    def has_index(self, label: str, prop: str) -> bool:
        return (label, prop) in self._data

    def insert(self, node: Node):
        for label in node.labels:
            for prop, val in node.props.items():
                key = (label, prop)
                if key in self._data:
                    hashable = _make_hashable(val)
                    self._data[key].setdefault(hashable, set()).add(node.id)

    def remove(self, node: Node):
        for label in node.labels:
            for prop, val in node.props.items():
                key = (label, prop)
                if key in self._data:
                    hashable = _make_hashable(val)
                    if hashable in self._data[key]:
                        self._data[key][hashable].discard(node.id)
                        if not self._data[key][hashable]:
                            del self._data[key][hashable]

    def lookup(self, label: str, prop: str, val: Any) -> set[int]:
        key = (label, prop)
        if key not in self._data:
            return set()
        hashable = _make_hashable(val)
        return set(self._data[key].get(hashable, set()))

    def clear(self):
        self._data.clear()

    def rebuild(self, nodes: dict[int, Node]):
        """Rebuild all indexes from scratch."""
        for key in self._data:
            self._data[key] = {}
        for node in nodes.values():
            self.insert(node)


def _make_hashable(val):
    if isinstance(val, list):
        return tuple(val)
    if isinstance(val, dict):
        return tuple(sorted(val.items()))
    return val


# ============================================================
# Graph Database
# ============================================================

class GraphDB:
    """In-memory property graph database."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        self._next_node_id = 1
        self._next_edge_id = 1
        # Adjacency: node_id -> list of edge_ids (outgoing)
        self._out_edges: dict[int, list[int]] = {}
        # Adjacency: node_id -> list of edge_ids (incoming)
        self._in_edges: dict[int, list[int]] = {}
        # Indexes
        self._index = PropertyIndex()
        # Snapshots
        self._snapshots: list[dict] = []

    # --- Node operations ---

    def add_node(self, labels: set[str] | list[str] | str = None,
                 props: dict[str, Any] = None) -> Node:
        nid = self._next_node_id
        self._next_node_id += 1
        if isinstance(labels, str):
            labels = {labels}
        elif isinstance(labels, list):
            labels = set(labels)
        node = Node(nid, labels or set(), props or {})
        self.nodes[nid] = node
        self._out_edges[nid] = []
        self._in_edges[nid] = []
        self._index.insert(node)
        return node

    def get_node(self, node_id: int) -> Node | None:
        return self.nodes.get(node_id)

    def delete_node(self, node_id: int) -> bool:
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        # Remove all connected edges
        for eid in list(self._out_edges.get(node_id, [])):
            self._remove_edge_internal(eid)
        for eid in list(self._in_edges.get(node_id, [])):
            self._remove_edge_internal(eid)
        self._index.remove(node)
        del self.nodes[node_id]
        del self._out_edges[node_id]
        del self._in_edges[node_id]
        return True

    def set_node_prop(self, node_id: int, prop: str, value: Any) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        self._index.remove(node)
        node.props[prop] = value
        self._index.insert(node)
        return True

    def remove_node_prop(self, node_id: int, prop: str) -> bool:
        node = self.nodes.get(node_id)
        if not node or prop not in node.props:
            return False
        self._index.remove(node)
        del node.props[prop]
        self._index.insert(node)
        return True

    def add_label(self, node_id: int, label: str) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        self._index.remove(node)
        node.labels.add(label)
        self._index.insert(node)
        return True

    def remove_label(self, node_id: int, label: str) -> bool:
        node = self.nodes.get(node_id)
        if not node or label not in node.labels:
            return False
        self._index.remove(node)
        node.labels.discard(label)
        self._index.insert(node)
        return True

    # --- Edge operations ---

    def add_edge(self, src: int, dst: int, edge_type: str,
                 props: dict[str, Any] = None) -> Edge | None:
        if src not in self.nodes or dst not in self.nodes:
            return None
        eid = self._next_edge_id
        self._next_edge_id += 1
        edge = Edge(eid, src, dst, edge_type, props or {})
        self.edges[eid] = edge
        self._out_edges[src].append(eid)
        self._in_edges[dst].append(eid)
        return edge

    def get_edge(self, edge_id: int) -> Edge | None:
        return self.edges.get(edge_id)

    def delete_edge(self, edge_id: int) -> bool:
        return self._remove_edge_internal(edge_id)

    def _remove_edge_internal(self, edge_id: int) -> bool:
        edge = self.edges.get(edge_id)
        if not edge:
            return False
        if edge.src in self._out_edges:
            try:
                self._out_edges[edge.src].remove(edge_id)
            except ValueError:
                pass
        if edge.dst in self._in_edges:
            try:
                self._in_edges[edge.dst].remove(edge_id)
            except ValueError:
                pass
        del self.edges[edge_id]
        return True

    def set_edge_prop(self, edge_id: int, prop: str, value: Any) -> bool:
        edge = self.edges.get(edge_id)
        if not edge:
            return False
        edge.props[prop] = value
        return True

    # --- Traversal ---

    def neighbors(self, node_id: int, edge_type: str = None,
                  direction: str = 'out') -> list[Node]:
        """Get neighboring nodes. direction: 'out', 'in', or 'both'."""
        result = []
        if direction in ('out', 'both'):
            for eid in self._out_edges.get(node_id, []):
                edge = self.edges[eid]
                if edge_type is None or edge.edge_type == edge_type:
                    node = self.nodes.get(edge.dst)
                    if node:
                        result.append(node)
        if direction in ('in', 'both'):
            for eid in self._in_edges.get(node_id, []):
                edge = self.edges[eid]
                if edge_type is None or edge.edge_type == edge_type:
                    node = self.nodes.get(edge.src)
                    if node:
                        result.append(node)
        return result

    def edges_of(self, node_id: int, edge_type: str = None,
                 direction: str = 'out') -> list[Edge]:
        """Get edges connected to a node."""
        result = []
        if direction in ('out', 'both'):
            for eid in self._out_edges.get(node_id, []):
                edge = self.edges[eid]
                if edge_type is None or edge.edge_type == edge_type:
                    result.append(edge)
        if direction in ('in', 'both'):
            for eid in self._in_edges.get(node_id, []):
                edge = self.edges[eid]
                if edge_type is None or edge.edge_type == edge_type:
                    result.append(edge)
        return result

    def find_nodes(self, label: str = None, **props) -> list[Node]:
        """Find nodes matching label and/or properties."""
        candidates = None

        # Try index first
        if label and props:
            for prop, val in props.items():
                if self._index.has_index(label, prop):
                    indexed = self._index.lookup(label, prop, val)
                    if candidates is None:
                        candidates = indexed
                    else:
                        candidates &= indexed

        if candidates is not None:
            result = []
            for nid in candidates:
                node = self.nodes[nid]
                if self._node_matches(node, label, props):
                    result.append(node)
            return result

        # Full scan
        result = []
        for node in self.nodes.values():
            if self._node_matches(node, label, props):
                result.append(node)
        return result

    def _node_matches(self, node: Node, label: str = None, props: dict = None) -> bool:
        if label and label not in node.labels:
            return False
        if props:
            for k, v in props.items():
                if k not in node.props or node.props[k] != v:
                    return False
        return True

    # --- Path finding ---

    def shortest_path(self, src: int, dst: int, edge_type: str = None,
                      max_depth: int = 20) -> list[int] | None:
        """BFS shortest path. Returns list of node IDs or None."""
        if src not in self.nodes or dst not in self.nodes:
            return None
        if src == dst:
            return [src]

        visited = {src}
        queue = [(src, [src])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for neighbor in self.neighbors(current, edge_type, 'out'):
                if neighbor.id == dst:
                    return path + [dst]
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append((neighbor.id, path + [neighbor.id]))
        return None

    def all_paths(self, src: int, dst: int, edge_type: str = None,
                  max_depth: int = 10) -> list[list[int]]:
        """DFS all simple paths (no repeated nodes). Returns list of node ID paths."""
        if src not in self.nodes or dst not in self.nodes:
            return []

        results = []

        def dfs(current, path, visited):
            if current == dst:
                results.append(list(path))
                return
            if len(path) > max_depth:
                return
            for neighbor in self.neighbors(current, edge_type, 'out'):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    path.append(neighbor.id)
                    dfs(neighbor.id, path, visited)
                    path.pop()
                    visited.discard(neighbor.id)

        dfs(src, [src], {src})
        return results

    # --- Index management ---

    def create_index(self, label: str, prop: str):
        """Create an index on (label, property) for faster lookups."""
        self._index.add_index(label, prop)
        self._index.rebuild(self.nodes)

    # --- Snapshots ---

    def save_snapshot(self) -> int:
        """Save a snapshot and return its index."""
        snap = {
            'nodes': {nid: Node(n.id, set(n.labels), copy.deepcopy(n.props))
                      for nid, n in self.nodes.items()},
            'edges': {eid: Edge(e.id, e.src, e.dst, e.edge_type, copy.deepcopy(e.props))
                      for eid, e in self.edges.items()},
            'next_node_id': self._next_node_id,
            'next_edge_id': self._next_edge_id,
        }
        self._snapshots.append(snap)
        return len(self._snapshots) - 1

    def restore_snapshot(self, index: int = -1) -> bool:
        """Restore from a snapshot. Default: latest."""
        if not self._snapshots:
            return False
        if index < 0:
            index = len(self._snapshots) + index
        if index < 0 or index >= len(self._snapshots):
            return False

        snap = self._snapshots[index]
        self.nodes = {nid: Node(n.id, set(n.labels), copy.deepcopy(n.props))
                      for nid, n in snap['nodes'].items()}
        self.edges = {eid: Edge(e.id, e.src, e.dst, e.edge_type, copy.deepcopy(e.props))
                      for eid, e in snap['edges'].items()}
        self._next_node_id = snap['next_node_id']
        self._next_edge_id = snap['next_edge_id']

        # Rebuild adjacency
        self._out_edges = {nid: [] for nid in self.nodes}
        self._in_edges = {nid: [] for nid in self.nodes}
        for edge in self.edges.values():
            self._out_edges[edge.src].append(edge.id)
            self._in_edges[edge.dst].append(edge.id)

        # Rebuild indexes
        self._index.rebuild(self.nodes)
        return True

    # --- Statistics ---

    def stats(self) -> dict:
        return {
            'nodes': len(self.nodes),
            'edges': len(self.edges),
            'snapshots': len(self._snapshots),
        }


# ============================================================
# Query Language Parser & Executor
# ============================================================

class QueryError(Exception):
    pass


class Token:
    __slots__ = ('type', 'value')
    def __init__(self, type: str, value: str):
        self.type = type
        self.value = value
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


# Token types
T_KEYWORD = 'KEYWORD'
T_IDENT = 'IDENT'
T_NUMBER = 'NUMBER'
T_STRING = 'STRING'
T_LPAREN = 'LPAREN'
T_RPAREN = 'RPAREN'
T_LBRACKET = 'LBRACKET'
T_RBRACKET = 'RBRACKET'
T_LBRACE = 'LBRACE'
T_RBRACE = 'RBRACE'
T_COLON = 'COLON'
T_COMMA = 'COMMA'
T_DOT = 'DOT'
T_ARROW = 'ARROW'
T_DASH = 'DASH'
T_EQ = 'EQ'
T_NEQ = 'NEQ'
T_LT = 'LT'
T_GT = 'GT'
T_LTE = 'LTE'
T_GTE = 'GTE'
T_STAR = 'STAR'
T_EOF = 'EOF'

KEYWORDS = {'MATCH', 'WHERE', 'RETURN', 'CREATE', 'DELETE', 'SET', 'ORDER', 'BY',
            'LIMIT', 'AND', 'OR', 'NOT', 'TRUE', 'FALSE', 'NULL', 'AS',
            'ASC', 'DESC', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT',
            'node', 'edge', 'IN', 'CONTAINS', 'STARTS', 'ENDS', 'WITH'}


def tokenize(query: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(query):
        c = query[i]

        if c.isspace():
            i += 1
            continue

        if c == '-' and i + 1 < len(query) and query[i + 1] == '>':
            tokens.append(Token(T_ARROW, '->'))
            i += 2
            continue

        if c == '<' and i + 1 < len(query) and query[i + 1] == '-':
            tokens.append(Token(T_DASH, '<-'))
            i += 2
            continue

        if c == '<' and i + 1 < len(query) and query[i + 1] == '=':
            tokens.append(Token(T_LTE, '<='))
            i += 2
            continue

        if c == '>' and i + 1 < len(query) and query[i + 1] == '=':
            tokens.append(Token(T_GTE, '>='))
            i += 2
            continue

        if c == '!' and i + 1 < len(query) and query[i + 1] == '=':
            tokens.append(Token(T_NEQ, '!='))
            i += 2
            continue

        simple = {
            '(': T_LPAREN, ')': T_RPAREN,
            '[': T_LBRACKET, ']': T_RBRACKET,
            '{': T_LBRACE, '}': T_RBRACE,
            ':': T_COLON, ',': T_COMMA, '.': T_DOT,
            '-': T_DASH, '=': T_EQ, '<': T_LT, '>': T_GT,
            '*': T_STAR,
        }
        if c in simple:
            tokens.append(Token(simple[c], c))
            i += 1
            continue

        if c == "'" or c == '"':
            quote = c
            i += 1
            start = i
            while i < len(query) and query[i] != quote:
                if query[i] == '\\':
                    i += 1
                i += 1
            if i >= len(query):
                raise QueryError(f"Unterminated string")
            val = query[start:i].replace('\\' + quote, quote).replace('\\\\', '\\')
            tokens.append(Token(T_STRING, val))
            i += 1
            continue

        if c.isdigit() or (c == '-' and i + 1 < len(query) and query[i + 1].isdigit()):
            start = i
            if c == '-':
                i += 1
            while i < len(query) and (query[i].isdigit() or query[i] == '.'):
                i += 1
            tokens.append(Token(T_NUMBER, query[start:i]))
            continue

        if c.isalpha() or c == '_':
            start = i
            while i < len(query) and (query[i].isalnum() or query[i] == '_'):
                i += 1
            word = query[start:i]
            if word.upper() in KEYWORDS:
                tokens.append(Token(T_KEYWORD, word.upper()))
            else:
                tokens.append(Token(T_IDENT, word))
            continue

        raise QueryError(f"Unexpected character: {c!r} at position {i}")

    tokens.append(Token(T_EOF, ''))
    return tokens


class Parser:
    """Parse GQL queries into AST."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, ttype: str, value: str = None) -> Token:
        t = self.advance()
        if t.type != ttype or (value is not None and t.value != value):
            raise QueryError(f"Expected {ttype}({value}), got {t}")
        return t

    def match(self, ttype: str, value: str = None) -> Token | None:
        t = self.peek()
        if t.type == ttype and (value is None or t.value == value):
            return self.advance()
        return None

    def parse(self) -> dict:
        t = self.peek()
        if t.type == T_KEYWORD:
            if t.value == 'MATCH':
                return self.parse_match()
            elif t.value == 'CREATE':
                return self.parse_create()
            elif t.value == 'DELETE':
                return self.parse_delete()
            elif t.value == 'SET':
                return self.parse_set()
        raise QueryError(f"Unexpected token: {t}")

    def parse_match(self) -> dict:
        self.expect(T_KEYWORD, 'MATCH')
        pattern = self.parse_pattern()

        where = None
        if self.match(T_KEYWORD, 'WHERE'):
            where = self.parse_expression()

        ret = None
        if self.match(T_KEYWORD, 'RETURN'):
            ret = self.parse_return_list()

        order_by = None
        if self.match(T_KEYWORD, 'ORDER'):
            self.expect(T_KEYWORD, 'BY')
            order_by = self.parse_order_list()

        limit = None
        if self.match(T_KEYWORD, 'LIMIT'):
            limit = int(self.expect(T_NUMBER).value)

        return {'type': 'match', 'pattern': pattern, 'where': where,
                'return': ret, 'order_by': order_by, 'limit': limit}

    def parse_pattern(self) -> list:
        """Parse a pattern like (a:Person)-[:KNOWS]->(b:Person)"""
        elements = []
        elements.append(self.parse_node_pattern())

        while self.peek().type in (T_DASH, T_ARROW):
            # Edge pattern
            edge, direction = self.parse_edge_pattern()
            elements.append({'element': 'edge', **edge, 'direction': direction})
            elements.append(self.parse_node_pattern())

        return elements

    def parse_node_pattern(self) -> dict:
        self.expect(T_LPAREN)
        name = None
        labels = []
        props = {}

        if self.peek().type == T_IDENT:
            name = self.advance().value
        elif self.peek().type == T_NUMBER:
            name = self.advance().value

        while self.match(T_COLON):
            labels.append(self.expect(T_IDENT).value)

        if self.match(T_LBRACE):
            props = self.parse_prop_map()

        self.expect(T_RPAREN)
        return {'element': 'node', 'name': name, 'labels': labels, 'props': props}

    def parse_edge_pattern(self) -> tuple[dict, str]:
        """Parse edge pattern, return (edge_dict, direction)."""
        # Could be: -[...]-> or <-[...]-  or -[...]-
        left_arrow = False
        if self.peek().type == T_DASH and self.peek().value == '<-':
            left_arrow = True
            self.advance()
        else:
            self.expect(T_DASH)

        name = None
        edge_types = []
        props = {}
        min_hops = 1
        max_hops = 1

        if self.match(T_LBRACKET):
            if self.peek().type == T_IDENT:
                name = self.advance().value

            while self.match(T_COLON):
                edge_types.append(self.expect(T_IDENT).value)

            # Variable length: *1..3
            if self.match(T_STAR):
                min_hops, max_hops = self._parse_hop_range()

            if self.match(T_LBRACE):
                props = self.parse_prop_map()

            self.expect(T_RBRACKET)

        if left_arrow:
            if self.match(T_DASH):
                pass  # <-[...]-
            direction = 'in'
        elif self.match(T_ARROW):
            direction = 'out'
        elif self.match(T_DASH):
            direction = 'both'
        else:
            direction = 'out'

        return ({'name': name, 'types': edge_types, 'props': props,
                 'min_hops': min_hops, 'max_hops': max_hops}, direction)

    def _parse_hop_range(self) -> tuple[int, int]:
        if self.peek().type == T_NUMBER:
            n = int(self.advance().value)
            if self.match(T_DOT):
                self.expect(T_DOT)
                m = int(self.expect(T_NUMBER).value)
                return (n, m)
            return (n, n)
        return (1, 10)  # default variable length

    def parse_prop_map(self) -> dict:
        props = {}
        if self.peek().type != T_RBRACE:
            key = self.expect(T_IDENT).value
            self.expect(T_COLON)
            val = self.parse_value()
            props[key] = val
            while self.match(T_COMMA):
                key = self.expect(T_IDENT).value
                self.expect(T_COLON)
                val = self.parse_value()
                props[key] = val
        self.expect(T_RBRACE)
        return props

    def parse_value(self) -> Any:
        t = self.peek()
        if t.type == T_NUMBER:
            self.advance()
            return float(t.value) if '.' in t.value else int(t.value)
        if t.type == T_STRING:
            self.advance()
            return t.value
        if t.type == T_KEYWORD and t.value == 'TRUE':
            self.advance()
            return True
        if t.type == T_KEYWORD and t.value == 'FALSE':
            self.advance()
            return False
        if t.type == T_KEYWORD and t.value == 'NULL':
            self.advance()
            return None
        raise QueryError(f"Expected value, got {t}")

    def parse_expression(self) -> dict:
        return self.parse_or()

    def parse_or(self) -> dict:
        left = self.parse_and()
        while self.match(T_KEYWORD, 'OR'):
            right = self.parse_and()
            left = {'op': 'or', 'left': left, 'right': right}
        return left

    def parse_and(self) -> dict:
        left = self.parse_not()
        while self.match(T_KEYWORD, 'AND'):
            right = self.parse_not()
            left = {'op': 'and', 'left': left, 'right': right}
        return left

    def parse_not(self) -> dict:
        if self.match(T_KEYWORD, 'NOT'):
            expr = self.parse_comparison()
            return {'op': 'not', 'expr': expr}
        return self.parse_comparison()

    def parse_comparison(self) -> dict:
        left = self.parse_primary_expr()

        ops = {T_EQ: '==', T_NEQ: '!=', T_LT: '<', T_GT: '>',
               T_LTE: '<=', T_GTE: '>='}

        t = self.peek()
        if t.type in ops:
            self.advance()
            right = self.parse_primary_expr()
            return {'op': ops[t.type], 'left': left, 'right': right}

        # String ops: CONTAINS, STARTS WITH, ENDS WITH
        if t.type == T_KEYWORD and t.value == 'CONTAINS':
            self.advance()
            right = self.parse_primary_expr()
            return {'op': 'contains', 'left': left, 'right': right}

        if t.type == T_KEYWORD and t.value == 'STARTS':
            self.advance()
            self.expect(T_KEYWORD, 'WITH')
            right = self.parse_primary_expr()
            return {'op': 'starts_with', 'left': left, 'right': right}

        if t.type == T_KEYWORD and t.value == 'ENDS':
            self.advance()
            self.expect(T_KEYWORD, 'WITH')
            right = self.parse_primary_expr()
            return {'op': 'ends_with', 'left': left, 'right': right}

        if t.type == T_KEYWORD and t.value == 'IN':
            self.advance()
            right = self.parse_primary_expr()
            return {'op': 'in', 'left': left, 'right': right}

        return left

    def parse_primary_expr(self) -> dict:
        t = self.peek()

        if t.type == T_LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(T_RPAREN)
            return expr

        if t.type in (T_NUMBER, T_STRING):
            self.advance()
            if t.type == T_NUMBER:
                val = float(t.value) if '.' in t.value else int(t.value)
            else:
                val = t.value
            return {'op': 'literal', 'value': val}

        if t.type == T_KEYWORD and t.value in ('TRUE', 'FALSE', 'NULL'):
            self.advance()
            val = True if t.value == 'TRUE' else (False if t.value == 'FALSE' else None)
            return {'op': 'literal', 'value': val}

        # Aggregate functions
        if t.type == T_KEYWORD and t.value in ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX'):
            func = t.value.lower()
            self.advance()
            self.expect(T_LPAREN)
            if self.match(T_STAR):
                arg = {'op': 'literal', 'value': '*'}
            else:
                arg = self.parse_expression()
            self.expect(T_RPAREN)
            return {'op': 'aggregate', 'func': func, 'arg': arg}

        # Unary minus
        if t.type == T_DASH and t.value == '-':
            self.advance()
            inner = self.parse_primary_expr()
            if inner.get('op') == 'literal' and isinstance(inner.get('value'), (int, float)):
                return {'op': 'literal', 'value': -inner['value']}
            return {'op': 'negate', 'expr': inner}

        if t.type == T_IDENT:
            self.advance()
            parts = [t.value]
            while self.match(T_DOT):
                parts.append(self.expect(T_IDENT).value)
            if len(parts) == 1:
                return {'op': 'ref', 'name': parts[0]}
            return {'op': 'prop', 'name': parts[0], 'prop': '.'.join(parts[1:])}

        raise QueryError(f"Expected expression, got {t}")

    def parse_return_list(self) -> list:
        items = []
        distinct = False
        if self.match(T_KEYWORD, 'DISTINCT'):
            distinct = True

        items.append(self._parse_return_item())
        while self.match(T_COMMA):
            items.append(self._parse_return_item())

        return {'items': items, 'distinct': distinct}

    def _parse_return_item(self) -> dict:
        if self.match(T_STAR):
            return {'expr': {'op': 'literal', 'value': '*'}, 'alias': None}
        expr = self.parse_expression()
        alias = None
        if self.match(T_KEYWORD, 'AS'):
            # Accept both IDENT and KEYWORD as alias names
            t = self.peek()
            if t.type in (T_IDENT, T_KEYWORD):
                alias = self.advance().value.lower() if t.type == T_KEYWORD else self.advance().value
            else:
                alias = self.expect(T_IDENT).value
        return {'expr': expr, 'alias': alias}

    def parse_order_list(self) -> list:
        items = []
        expr = self.parse_expression()
        direction = 'asc'
        if self.match(T_KEYWORD, 'DESC'):
            direction = 'desc'
        elif self.match(T_KEYWORD, 'ASC'):
            direction = 'asc'
        items.append({'expr': expr, 'direction': direction})

        while self.match(T_COMMA):
            expr = self.parse_expression()
            direction = 'asc'
            if self.match(T_KEYWORD, 'DESC'):
                direction = 'desc'
            elif self.match(T_KEYWORD, 'ASC'):
                direction = 'asc'
            items.append({'expr': expr, 'direction': direction})

        return items

    def parse_create(self) -> dict:
        self.expect(T_KEYWORD, 'CREATE')

        # Could be node or edge
        # Node: CREATE (n:Label {props})
        # Edge: CREATE (a)-[:TYPE]->(b) where a, b are node refs
        node = self.parse_node_pattern()

        if self.peek().type in (T_DASH, T_ARROW):
            # It's an edge creation
            edge, direction = self.parse_edge_pattern()
            target = self.parse_node_pattern()
            return {'type': 'create_edge', 'src': node, 'edge': edge,
                    'dst': target, 'direction': direction}

        return {'type': 'create_node', 'node': node}

    def parse_delete(self) -> dict:
        self.expect(T_KEYWORD, 'DELETE')
        kind = self.advance()  # 'node' or 'edge'
        if kind.value not in ('node', 'edge'):
            raise QueryError(f"Expected 'node' or 'edge' after DELETE, got {kind.value}")
        id_tok = self.expect(T_NUMBER)
        return {'type': f'delete_{kind.value}', 'id': int(id_tok.value)}

    def parse_set(self) -> dict:
        self.expect(T_KEYWORD, 'SET')
        kind = self.advance()
        if kind.value not in ('node', 'edge'):
            raise QueryError(f"Expected 'node' or 'edge' after SET, got {kind.value}")
        id_tok = self.expect(T_NUMBER)
        prop = self.expect(T_IDENT).value
        self.expect(T_EQ)
        val = self.parse_value()
        return {'type': f'set_{kind.value}', 'id': int(id_tok.value),
                'prop': prop, 'value': val}


# ============================================================
# Query Executor
# ============================================================

class QueryExecutor:
    """Execute parsed GQL queries against a GraphDB."""

    def __init__(self, db: GraphDB):
        self.db = db

    def execute(self, query_str: str) -> list[dict]:
        tokens = tokenize(query_str)
        parser = Parser(tokens)
        ast = parser.parse()

        dispatch = {
            'match': self._exec_match,
            'create_node': self._exec_create_node,
            'create_edge': self._exec_create_edge,
            'delete_node': self._exec_delete_node,
            'delete_edge': self._exec_delete_edge,
            'set_node': self._exec_set_node,
            'set_edge': self._exec_set_edge,
        }

        handler = dispatch.get(ast['type'])
        if not handler:
            raise QueryError(f"Unknown query type: {ast['type']}")
        return handler(ast)

    def _exec_match(self, ast: dict) -> list[dict]:
        pattern = ast['pattern']
        bindings_list = self._match_pattern(pattern)

        # Apply WHERE
        if ast['where']:
            bindings_list = [b for b in bindings_list
                            if self._eval_expr(ast['where'], b)]

        # Apply RETURN
        if ast['return']:
            results = self._apply_return(bindings_list, ast['return'])
        else:
            results = [self._bindings_to_dict(b) for b in bindings_list]

        # ORDER BY
        if ast.get('order_by'):
            results = self._apply_order(results, ast['order_by'], bindings_list, ast['return'])

        # LIMIT
        if ast.get('limit') is not None:
            results = results[:ast['limit']]

        return results

    def _match_pattern(self, pattern: list) -> list[dict]:
        """Match pattern against graph. Returns list of binding dicts."""
        # Start with node patterns, then expand through edges
        if not pattern:
            return [{}]

        first = pattern[0]
        bindings = self._match_node(first)

        i = 1
        while i < len(pattern):
            edge_pat = pattern[i]
            node_pat = pattern[i + 1]
            new_bindings = []

            for binding in bindings:
                # Get the previous node
                prev_node_name = pattern[i - 1]['name']
                if prev_node_name and prev_node_name in binding:
                    prev_node = binding[prev_node_name]
                    expanded = self._expand_edge(prev_node, edge_pat, node_pat, binding)
                    new_bindings.extend(expanded)

            bindings = new_bindings
            i += 2

        return bindings

    def _match_node(self, node_pat: dict) -> list[dict]:
        """Find all nodes matching a node pattern."""
        bindings = []
        label = node_pat['labels'][0] if node_pat['labels'] else None

        candidates = self.db.find_nodes(label, **node_pat['props'])

        # Filter by additional labels
        for node in candidates:
            if all(l in node.labels for l in node_pat['labels']):
                binding = {}
                if node_pat['name']:
                    binding[node_pat['name']] = node
                bindings.append(binding)

        return bindings

    def _expand_edge(self, prev_node: Node, edge_pat: dict,
                     node_pat: dict, binding: dict) -> list[dict]:
        """Expand from a node through an edge pattern to find next nodes."""
        results = []
        direction = edge_pat['direction']
        edge_types = edge_pat['types']
        min_hops = edge_pat.get('min_hops', 1)
        max_hops = edge_pat.get('max_hops', 1)

        if min_hops == 1 and max_hops == 1:
            # Simple single-hop
            edges = self.db.edges_of(prev_node.id, direction=direction)
            for edge in edges:
                if edge_types and edge.edge_type not in edge_types:
                    continue
                if edge_pat['props']:
                    if not all(edge.props.get(k) == v for k, v in edge_pat['props'].items()):
                        continue

                next_id = edge.dst if direction in ('out', 'both') and edge.src == prev_node.id else edge.src
                next_node = self.db.get_node(next_id)
                if not next_node:
                    continue

                if not self._node_matches_pattern(next_node, node_pat):
                    continue

                new_binding = dict(binding)
                if edge_pat.get('name'):
                    new_binding[edge_pat['name']] = edge
                if node_pat['name']:
                    new_binding[node_pat['name']] = next_node
                results.append(new_binding)
        else:
            # Variable length path
            self._variable_expand(prev_node, edge_pat, node_pat, binding,
                                 results, min_hops, max_hops)

        return results

    def _variable_expand(self, start: Node, edge_pat: dict, node_pat: dict,
                         binding: dict, results: list, min_hops: int, max_hops: int):
        """BFS for variable-length paths."""
        direction = edge_pat['direction']
        edge_types = edge_pat['types']

        # BFS: (current_node, depth, visited)
        queue = [(start, 0, {start.id})]

        while queue:
            current, depth, visited = queue.pop(0)

            if depth >= min_hops:
                if self._node_matches_pattern(current, node_pat) and depth > 0:
                    new_binding = dict(binding)
                    if node_pat['name']:
                        new_binding[node_pat['name']] = current
                    results.append(new_binding)

            if depth >= max_hops:
                continue

            edges = self.db.edges_of(current.id, direction=direction)
            for edge in edges:
                if edge_types and edge.edge_type not in edge_types:
                    continue
                next_id = edge.dst if edge.src == current.id else edge.src
                if next_id in visited:
                    continue
                next_node = self.db.get_node(next_id)
                if next_node:
                    queue.append((next_node, depth + 1, visited | {next_id}))

    def _node_matches_pattern(self, node: Node, pat: dict) -> bool:
        for label in pat['labels']:
            if label not in node.labels:
                return False
        for k, v in pat['props'].items():
            if node.props.get(k) != v:
                return False
        return True

    def _eval_expr(self, expr: dict, binding: dict) -> Any:
        op = expr.get('op')

        if op == 'literal':
            return expr['value']

        if op == 'ref':
            name = expr['name']
            if name in binding:
                return binding[name]
            raise QueryError(f"Unknown reference: {name}")

        if op == 'prop':
            name = expr['name']
            prop = expr['prop']
            obj = binding.get(name)
            if obj is None:
                return None
            if isinstance(obj, (Node, Edge)):
                return obj.props.get(prop)
            return None

        if op == 'and':
            return self._eval_expr(expr['left'], binding) and self._eval_expr(expr['right'], binding)

        if op == 'or':
            return self._eval_expr(expr['left'], binding) or self._eval_expr(expr['right'], binding)

        if op == 'not':
            return not self._eval_expr(expr['expr'], binding)

        if op in ('==', '!=', '<', '>', '<=', '>='):
            left = self._eval_expr(expr['left'], binding)
            right = self._eval_expr(expr['right'], binding)
            if left is None or right is None:
                return op == '!=' if (left is None) != (right is None) else op == '=='
            ops = {
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '>': lambda a, b: a > b,
                '<=': lambda a, b: a <= b,
                '>=': lambda a, b: a >= b,
            }
            return ops[op](left, right)

        if op == 'contains':
            left = self._eval_expr(expr['left'], binding)
            right = self._eval_expr(expr['right'], binding)
            return right in left if left and right else False

        if op == 'starts_with':
            left = self._eval_expr(expr['left'], binding)
            right = self._eval_expr(expr['right'], binding)
            return left.startswith(right) if isinstance(left, str) and isinstance(right, str) else False

        if op == 'ends_with':
            left = self._eval_expr(expr['left'], binding)
            right = self._eval_expr(expr['right'], binding)
            return left.endswith(right) if isinstance(left, str) and isinstance(right, str) else False

        if op == 'in':
            left = self._eval_expr(expr['left'], binding)
            right = self._eval_expr(expr['right'], binding)
            return left in right if right else False

        if op == 'aggregate':
            # Aggregates are handled at the RETURN level, not here
            return expr

        raise QueryError(f"Unknown expression op: {op}")

    def _apply_return(self, bindings_list: list[dict], ret: dict) -> list[dict]:
        items = ret['items']
        distinct = ret['distinct']

        # Check for aggregates
        has_agg = any(self._has_aggregate(item['expr']) for item in items)

        if has_agg:
            return self._apply_aggregate_return(bindings_list, items)

        results = []
        seen = set() if distinct else None

        for binding in bindings_list:
            row = {}
            for item in items:
                expr = item['expr']
                alias = item['alias']

                if expr.get('op') == 'literal' and expr.get('value') == '*':
                    # Return all bindings
                    for k, v in binding.items():
                        if isinstance(v, Node):
                            row[k] = v.to_dict()
                        elif isinstance(v, Edge):
                            row[k] = v.to_dict()
                        else:
                            row[k] = v
                    continue

                val = self._eval_expr(expr, binding)
                key = alias or self._expr_key(expr)

                if isinstance(val, Node):
                    row[key] = val.to_dict()
                elif isinstance(val, Edge):
                    row[key] = val.to_dict()
                else:
                    row[key] = val

            if distinct:
                row_key = tuple(sorted((k, _make_hashable(v)) for k, v in row.items()))
                if row_key in seen:
                    continue
                seen.add(row_key)

            results.append(row)

        return results

    def _has_aggregate(self, expr: dict) -> bool:
        if not isinstance(expr, dict):
            return False
        if expr.get('op') == 'aggregate':
            return True
        for v in expr.values():
            if isinstance(v, dict) and self._has_aggregate(v):
                return True
        return False

    def _apply_aggregate_return(self, bindings_list: list, items: list) -> list[dict]:
        """Handle aggregate functions in RETURN."""
        row = {}
        for item in items:
            expr = item['expr']
            alias = item['alias']

            if expr.get('op') == 'aggregate':
                func = expr['func']
                arg = expr['arg']

                if func == 'count':
                    if arg.get('op') == 'literal' and arg.get('value') == '*':
                        row[alias or 'count(*)'] = len(bindings_list)
                    else:
                        vals = [self._eval_expr(arg, b) for b in bindings_list]
                        row[alias or f'count'] = len([v for v in vals if v is not None])
                elif func in ('sum', 'avg', 'min', 'max'):
                    vals = [self._eval_expr(arg, b) for b in bindings_list]
                    nums = [v for v in vals if isinstance(v, (int, float))]
                    key = alias or func
                    if not nums:
                        row[key] = None
                    elif func == 'sum':
                        row[key] = sum(nums)
                    elif func == 'avg':
                        row[key] = sum(nums) / len(nums)
                    elif func == 'min':
                        row[key] = min(nums)
                    elif func == 'max':
                        row[key] = max(nums)
            else:
                # Non-aggregate in an aggregate query: take first
                key = alias or self._expr_key(expr)
                if bindings_list:
                    val = self._eval_expr(expr, bindings_list[0])
                    if isinstance(val, Node):
                        row[key] = val.to_dict()
                    elif isinstance(val, Edge):
                        row[key] = val.to_dict()
                    else:
                        row[key] = val
                else:
                    row[key] = None

        return [row]

    def _apply_order(self, results: list[dict], order_by: list,
                     bindings_list: list, ret: dict) -> list[dict]:
        """Sort results by ORDER BY clauses."""
        def sort_key(row):
            keys = []
            for clause in order_by:
                key_name = self._expr_key(clause['expr'])
                val = row.get(key_name)
                # Handle None and mixed types
                if val is None:
                    keys.append((1, ''))
                elif isinstance(val, (int, float)):
                    keys.append((0, val if clause['direction'] == 'asc' else -val))
                else:
                    keys.append((0, str(val)))
            return keys

        reverse = any(c['direction'] == 'desc' for c in order_by)
        # Simple case: single order direction
        if len(order_by) == 1:
            key_name = self._expr_key(order_by[0]['expr'])
            reverse = order_by[0]['direction'] == 'desc'
            results.sort(key=lambda r: (r.get(key_name) is None, r.get(key_name) or 0),
                        reverse=reverse)
        else:
            results.sort(key=sort_key)

        return results

    def _expr_key(self, expr: dict) -> str:
        op = expr.get('op')
        if op == 'ref':
            return expr['name']
        if op == 'prop':
            return f"{expr['name']}.{expr['prop']}"
        if op == 'aggregate':
            return f"{expr['func']}({self._expr_key(expr['arg'])})"
        if op == 'literal':
            return str(expr['value'])
        return str(expr)

    def _bindings_to_dict(self, binding: dict) -> dict:
        result = {}
        for k, v in binding.items():
            if isinstance(v, Node):
                result[k] = v.to_dict()
            elif isinstance(v, Edge):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def _exec_create_node(self, ast: dict) -> list[dict]:
        node_pat = ast['node']
        labels = set(node_pat['labels'])
        props = node_pat['props']
        node = self.db.add_node(labels, props)
        return [node.to_dict()]

    def _exec_create_edge(self, ast: dict) -> list[dict]:
        src_pat = ast['src']
        dst_pat = ast['dst']
        edge_info = ast['edge']

        # Source and dest must have names that refer to node IDs
        src_name = src_pat['name']
        dst_name = dst_pat['name']

        if not src_name or not dst_name:
            raise QueryError("Edge creation requires named node references")

        # Try to interpret names as IDs (numeric)
        try:
            src_id = int(src_name)
            dst_id = int(dst_name)
        except ValueError:
            raise QueryError("Edge creation node references must be numeric IDs")

        if not edge_info['types']:
            raise QueryError("Edge creation requires a type")

        edge_type = edge_info['types'][0]
        edge = self.db.add_edge(src_id, dst_id, edge_type, edge_info.get('props', {}))
        if not edge:
            raise QueryError(f"Cannot create edge: source {src_id} or dest {dst_id} not found")
        return [edge.to_dict()]

    def _exec_delete_node(self, ast: dict) -> list[dict]:
        success = self.db.delete_node(ast['id'])
        return [{'deleted': success, 'type': 'node', 'id': ast['id']}]

    def _exec_delete_edge(self, ast: dict) -> list[dict]:
        success = self.db.delete_edge(ast['id'])
        return [{'deleted': success, 'type': 'edge', 'id': ast['id']}]

    def _exec_set_node(self, ast: dict) -> list[dict]:
        success = self.db.set_node_prop(ast['id'], ast['prop'], ast['value'])
        return [{'updated': success, 'type': 'node', 'id': ast['id']}]

    def _exec_set_edge(self, ast: dict) -> list[dict]:
        success = self.db.set_edge_prop(ast['id'], ast['prop'], ast['value'])
        return [{'updated': success, 'type': 'edge', 'id': ast['id']}]


def query(db: GraphDB, query_str: str) -> list[dict]:
    """Convenience function to execute a query."""
    executor = QueryExecutor(db)
    return executor.execute(query_str)
