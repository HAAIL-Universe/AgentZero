"""
Tests for C015: Graph Database
"""
import pytest
from graph_db import (
    GraphDB, Node, Edge, PropertyIndex, QueryError,
    QueryExecutor, query, tokenize, Parser
)


# ============================================================
# Core: Node operations
# ============================================================

class TestNodeOperations:
    def test_add_node_basic(self):
        db = GraphDB()
        n = db.add_node("Person", {"name": "Alice", "age": 30})
        assert n.id == 1
        assert "Person" in n.labels
        assert n.props["name"] == "Alice"

    def test_add_node_multiple_labels(self):
        db = GraphDB()
        n = db.add_node({"Person", "Employee"}, {"name": "Bob"})
        assert "Person" in n.labels
        assert "Employee" in n.labels

    def test_add_node_no_labels(self):
        db = GraphDB()
        n = db.add_node(props={"key": "value"})
        assert len(n.labels) == 0
        assert n.props["key"] == "value"

    def test_add_node_increments_id(self):
        db = GraphDB()
        n1 = db.add_node("A")
        n2 = db.add_node("B")
        n3 = db.add_node("C")
        assert n1.id == 1
        assert n2.id == 2
        assert n3.id == 3

    def test_get_node(self):
        db = GraphDB()
        n = db.add_node("X", {"val": 42})
        got = db.get_node(n.id)
        assert got is n

    def test_get_node_nonexistent(self):
        db = GraphDB()
        assert db.get_node(999) is None

    def test_delete_node(self):
        db = GraphDB()
        n = db.add_node("X")
        assert db.delete_node(n.id) is True
        assert db.get_node(n.id) is None

    def test_delete_node_nonexistent(self):
        db = GraphDB()
        assert db.delete_node(999) is False

    def test_delete_node_removes_edges(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        c = db.add_node("C")
        e1 = db.add_edge(a.id, b.id, "KNOWS")
        e2 = db.add_edge(b.id, c.id, "KNOWS")
        db.delete_node(b.id)
        assert db.get_edge(e1.id) is None
        assert db.get_edge(e2.id) is None

    def test_set_node_prop(self):
        db = GraphDB()
        n = db.add_node("X", {"a": 1})
        db.set_node_prop(n.id, "b", 2)
        assert n.props["b"] == 2

    def test_set_node_prop_overwrite(self):
        db = GraphDB()
        n = db.add_node("X", {"a": 1})
        db.set_node_prop(n.id, "a", 99)
        assert n.props["a"] == 99

    def test_remove_node_prop(self):
        db = GraphDB()
        n = db.add_node("X", {"a": 1, "b": 2})
        db.remove_node_prop(n.id, "a")
        assert "a" not in n.props
        assert "b" in n.props

    def test_add_label(self):
        db = GraphDB()
        n = db.add_node("A")
        db.add_label(n.id, "B")
        assert "A" in n.labels
        assert "B" in n.labels

    def test_remove_label(self):
        db = GraphDB()
        n = db.add_node({"A", "B"})
        db.remove_label(n.id, "A")
        assert "A" not in n.labels
        assert "B" in n.labels

    def test_node_to_dict(self):
        db = GraphDB()
        n = db.add_node({"Person"}, {"name": "Alice"})
        d = n.to_dict()
        assert d["id"] == 1
        assert d["labels"] == ["Person"]
        assert d["props"]["name"] == "Alice"


# ============================================================
# Core: Edge operations
# ============================================================

class TestEdgeOperations:
    def test_add_edge(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        e = db.add_edge(a.id, b.id, "KNOWS")
        assert e.id == 1
        assert e.src == a.id
        assert e.dst == b.id
        assert e.edge_type == "KNOWS"

    def test_add_edge_with_props(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        e = db.add_edge(a.id, b.id, "KNOWS", {"since": 2020})
        assert e.props["since"] == 2020

    def test_add_edge_invalid_nodes(self):
        db = GraphDB()
        a = db.add_node("A")
        assert db.add_edge(a.id, 999, "X") is None
        assert db.add_edge(999, a.id, "X") is None

    def test_delete_edge(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        e = db.add_edge(a.id, b.id, "X")
        assert db.delete_edge(e.id) is True
        assert db.get_edge(e.id) is None

    def test_set_edge_prop(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        e = db.add_edge(a.id, b.id, "X")
        db.set_edge_prop(e.id, "weight", 5)
        assert e.props["weight"] == 5

    def test_edge_to_dict(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        e = db.add_edge(a.id, b.id, "KNOWS", {"since": 2020})
        d = e.to_dict()
        assert d["src"] == a.id
        assert d["dst"] == b.id
        assert d["type"] == "KNOWS"


# ============================================================
# Traversal
# ============================================================

class TestTraversal:
    def _build_social_graph(self):
        db = GraphDB()
        alice = db.add_node("Person", {"name": "Alice", "age": 30})
        bob = db.add_node("Person", {"name": "Bob", "age": 25})
        carol = db.add_node("Person", {"name": "Carol", "age": 35})
        dave = db.add_node("Person", {"name": "Dave", "age": 28})
        db.add_edge(alice.id, bob.id, "KNOWS")
        db.add_edge(alice.id, carol.id, "KNOWS")
        db.add_edge(bob.id, dave.id, "KNOWS")
        db.add_edge(carol.id, dave.id, "KNOWS")
        db.add_edge(alice.id, bob.id, "LIKES")
        return db, alice, bob, carol, dave

    def test_neighbors_out(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        neighbors = db.neighbors(alice.id, "KNOWS", "out")
        names = {n.props["name"] for n in neighbors}
        assert names == {"Bob", "Carol"}

    def test_neighbors_in(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        neighbors = db.neighbors(dave.id, "KNOWS", "in")
        names = {n.props["name"] for n in neighbors}
        assert names == {"Bob", "Carol"}

    def test_neighbors_both(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        neighbors = db.neighbors(bob.id, "KNOWS", "both")
        names = {n.props["name"] for n in neighbors}
        assert names == {"Alice", "Dave"}

    def test_neighbors_by_type(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        likes_neighbors = db.neighbors(alice.id, "LIKES", "out")
        assert len(likes_neighbors) == 1
        assert likes_neighbors[0].props["name"] == "Bob"

    def test_neighbors_all_types(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        all_out = db.neighbors(alice.id, direction="out")
        # Alice->Bob (KNOWS), Alice->Carol (KNOWS), Alice->Bob (LIKES)
        assert len(all_out) == 3

    def test_edges_of(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        edges = db.edges_of(alice.id, direction="out")
        assert len(edges) == 3

    def test_edges_of_filtered(self):
        db, alice, bob, carol, dave = self._build_social_graph()
        edges = db.edges_of(alice.id, "LIKES", "out")
        assert len(edges) == 1
        assert edges[0].edge_type == "LIKES"

    def test_find_nodes_by_label(self):
        db, *_ = self._build_social_graph()
        persons = db.find_nodes("Person")
        assert len(persons) == 4

    def test_find_nodes_by_props(self):
        db, *_ = self._build_social_graph()
        results = db.find_nodes("Person", name="Alice")
        assert len(results) == 1
        assert results[0].props["name"] == "Alice"

    def test_find_nodes_no_match(self):
        db, *_ = self._build_social_graph()
        results = db.find_nodes("Robot")
        assert len(results) == 0


# ============================================================
# Path finding
# ============================================================

class TestPathFinding:
    def _build_chain(self):
        db = GraphDB()
        nodes = [db.add_node("N", {"i": i}) for i in range(5)]
        for i in range(4):
            db.add_edge(nodes[i].id, nodes[i+1].id, "NEXT")
        return db, nodes

    def test_shortest_path(self):
        db, nodes = self._build_chain()
        path = db.shortest_path(nodes[0].id, nodes[4].id)
        assert path == [1, 2, 3, 4, 5]

    def test_shortest_path_same_node(self):
        db, nodes = self._build_chain()
        path = db.shortest_path(nodes[0].id, nodes[0].id)
        assert path == [nodes[0].id]

    def test_shortest_path_no_route(self):
        db, nodes = self._build_chain()
        path = db.shortest_path(nodes[4].id, nodes[0].id)
        assert path is None

    def test_shortest_path_with_shortcut(self):
        db, nodes = self._build_chain()
        db.add_edge(nodes[0].id, nodes[3].id, "NEXT")
        path = db.shortest_path(nodes[0].id, nodes[4].id)
        assert path == [1, 4, 5]

    def test_all_paths(self):
        db, nodes = self._build_chain()
        db.add_edge(nodes[0].id, nodes[2].id, "NEXT")
        paths = db.all_paths(nodes[0].id, nodes[4].id)
        assert len(paths) == 2

    def test_all_paths_no_cycles(self):
        db = GraphDB()
        a = db.add_node("N")
        b = db.add_node("N")
        c = db.add_node("N")
        db.add_edge(a.id, b.id, "X")
        db.add_edge(b.id, c.id, "X")
        db.add_edge(b.id, a.id, "X")  # cycle
        paths = db.all_paths(a.id, c.id)
        assert len(paths) == 1
        assert paths[0] == [a.id, b.id, c.id]


# ============================================================
# Indexes
# ============================================================

class TestIndexes:
    def test_create_and_use_index(self):
        db = GraphDB()
        db.create_index("Person", "name")
        db.add_node("Person", {"name": "Alice"})
        db.add_node("Person", {"name": "Bob"})
        db.add_node("Person", {"name": "Alice"})
        results = db.find_nodes("Person", name="Alice")
        assert len(results) == 2

    def test_index_updates_on_delete(self):
        db = GraphDB()
        db.create_index("X", "val")
        n = db.add_node("X", {"val": 42})
        db.delete_node(n.id)
        results = db.find_nodes("X", val=42)
        assert len(results) == 0

    def test_index_updates_on_prop_change(self):
        db = GraphDB()
        db.create_index("X", "val")
        n = db.add_node("X", {"val": 1})
        db.set_node_prop(n.id, "val", 2)
        assert len(db.find_nodes("X", val=1)) == 0
        assert len(db.find_nodes("X", val=2)) == 1

    def test_index_rebuild_on_snapshot_restore(self):
        db = GraphDB()
        db.create_index("X", "val")
        db.add_node("X", {"val": 1})
        db.save_snapshot()
        db.add_node("X", {"val": 2})
        db.restore_snapshot()
        assert len(db.find_nodes("X", val=1)) == 1
        assert len(db.find_nodes("X", val=2)) == 0


# ============================================================
# Snapshots
# ============================================================

class TestSnapshots:
    def test_save_and_restore(self):
        db = GraphDB()
        db.add_node("A", {"x": 1})
        idx = db.save_snapshot()
        db.add_node("B", {"x": 2})
        assert len(db.nodes) == 2
        db.restore_snapshot(idx)
        assert len(db.nodes) == 1

    def test_restore_latest(self):
        db = GraphDB()
        db.add_node("A")
        db.save_snapshot()
        db.add_node("B")
        db.save_snapshot()
        db.add_node("C")
        db.restore_snapshot()  # restore latest (2 nodes)
        assert len(db.nodes) == 2

    def test_restore_no_snapshots(self):
        db = GraphDB()
        assert db.restore_snapshot() is False

    def test_snapshot_preserves_edges(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        db.add_edge(a.id, b.id, "X")
        db.save_snapshot()
        db.delete_edge(1)
        db.restore_snapshot()
        assert db.get_edge(1) is not None

    def test_snapshot_isolation(self):
        db = GraphDB()
        n = db.add_node("A", {"x": 1})
        db.save_snapshot()
        db.set_node_prop(n.id, "x", 999)
        db.restore_snapshot()
        assert db.get_node(1).props["x"] == 1


# ============================================================
# Stats
# ============================================================

class TestStats:
    def test_stats(self):
        db = GraphDB()
        db.add_node("A")
        db.add_node("B")
        db.add_edge(1, 2, "X")
        db.save_snapshot()
        s = db.stats()
        assert s["nodes"] == 2
        assert s["edges"] == 1
        assert s["snapshots"] == 1


# ============================================================
# Tokenizer
# ============================================================

class TestTokenizer:
    def test_basic_match(self):
        tokens = tokenize("MATCH (n:Person) RETURN n")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert 'KEYWORD' in types
        assert 'IDENT' in types

    def test_string_literal(self):
        tokens = tokenize("'hello world'")
        assert tokens[0].value == "hello world"

    def test_number(self):
        tokens = tokenize("42 3.14")
        assert tokens[0].value == "42"
        assert tokens[1].value == "3.14"

    def test_arrow(self):
        tokens = tokenize("->")
        assert tokens[0].type == 'ARROW'

    def test_comparison_ops(self):
        tokens = tokenize("<= >= != = < >")
        types = [t.type for t in tokens[:-1]]
        assert types == ['LTE', 'GTE', 'NEQ', 'EQ', 'LT', 'GT']

    def test_unterminated_string(self):
        with pytest.raises(QueryError):
            tokenize("'unterminated")


# ============================================================
# Query Language: MATCH
# ============================================================

class TestQueryMatch:
    def _social_db(self):
        db = GraphDB()
        alice = db.add_node("Person", {"name": "Alice", "age": 30})
        bob = db.add_node("Person", {"name": "Bob", "age": 25})
        carol = db.add_node("Person", {"name": "Carol", "age": 35})
        dave = db.add_node("Person", {"name": "Dave", "age": 28})
        eve = db.add_node("Person", {"name": "Eve", "age": 30})
        db.add_edge(alice.id, bob.id, "KNOWS", {"since": 2015})
        db.add_edge(alice.id, carol.id, "KNOWS", {"since": 2018})
        db.add_edge(bob.id, dave.id, "KNOWS", {"since": 2020})
        db.add_edge(carol.id, dave.id, "KNOWS")
        db.add_edge(dave.id, eve.id, "KNOWS")
        return db

    def test_match_all_nodes_by_label(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN n")
        assert len(results) == 5

    def test_match_with_property(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person {name: 'Alice'}) RETURN n")
        assert len(results) == 1
        assert results[0]["n"]["props"]["name"] == "Alice"

    def test_match_with_where_gt(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.age > 29 RETURN n.name")
        names = {r["n.name"] for r in results}
        assert names == {"Alice", "Carol", "Eve"}

    def test_match_with_where_eq(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.age = 30 RETURN n.name")
        names = {r["n.name"] for r in results}
        assert names == {"Alice", "Eve"}

    def test_match_edge_pattern(self):
        db = self._social_db()
        results = query(db, "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        assert len(results) == 5  # 5 KNOWS edges
        pairs = {(r["a.name"], r["b.name"]) for r in results}
        assert ("Alice", "Bob") in pairs
        assert ("Bob", "Dave") in pairs

    def test_match_edge_with_where(self):
        db = self._social_db()
        results = query(db, "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE b.age > 30 RETURN a.name, b.name")
        assert len(results) == 1
        assert results[0]["b.name"] == "Carol"

    def test_match_and_where(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.age > 25 AND n.age < 32 RETURN n.name")
        names = {r["n.name"] for r in results}
        assert names == {"Alice", "Dave", "Eve"}

    def test_match_or_where(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.age = 25 OR n.age = 35 RETURN n.name")
        names = {r["n.name"] for r in results}
        assert names == {"Bob", "Carol"}

    def test_match_return_star(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person {name: 'Alice'}) RETURN *")
        assert len(results) == 1
        assert "n" in results[0]

    def test_match_with_limit(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN n.name LIMIT 2")
        assert len(results) == 2

    def test_match_order_by_asc(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age")
        ages = [r["n.age"] for r in results]
        assert ages == sorted(ages)

    def test_match_order_by_desc(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age DESC")
        ages = [r["n.age"] for r in results]
        assert ages == sorted(ages, reverse=True)

    def test_match_contains(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.name CONTAINS 'li' RETURN n.name")
        names = {r["n.name"] for r in results}
        assert names == {"Alice"}

    def test_match_starts_with(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.name")
        assert len(results) == 1

    def test_match_ends_with(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) WHERE n.name ENDS WITH 'e' RETURN n.name")
        names = {r["n.name"] for r in results}
        assert names == {"Alice", "Dave", "Eve"}

    def test_match_distinct(self):
        db = self._social_db()
        results = query(db, "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN DISTINCT a.name")
        names = {r["a.name"] for r in results}
        # Alice, Bob, Carol, Dave all have outgoing KNOWS
        assert len(results) == len(names)  # no duplicates


# ============================================================
# Query Language: Aggregates
# ============================================================

class TestQueryAggregates:
    def _social_db(self):
        db = GraphDB()
        db.add_node("Person", {"name": "Alice", "age": 30})
        db.add_node("Person", {"name": "Bob", "age": 25})
        db.add_node("Person", {"name": "Carol", "age": 35})
        return db

    def test_count_star(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN COUNT(*) AS total")
        assert results[0]["total"] == 3

    def test_sum(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN SUM(n.age) AS total_age")
        assert results[0]["total_age"] == 90

    def test_avg(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN AVG(n.age) AS avg_age")
        assert results[0]["avg_age"] == 30.0

    def test_min_max(self):
        db = self._social_db()
        results = query(db, "MATCH (n:Person) RETURN MIN(n.age) AS youngest, MAX(n.age) AS oldest")
        assert results[0]["youngest"] == 25
        assert results[0]["oldest"] == 35


# ============================================================
# Query Language: CREATE
# ============================================================

class TestQueryCreate:
    def test_create_node(self):
        db = GraphDB()
        results = query(db, "CREATE (n:Person {name: 'Alice', age: 30})")
        assert results[0]["id"] == 1
        assert db.get_node(1).props["name"] == "Alice"

    def test_create_node_multiple_labels(self):
        db = GraphDB()
        results = query(db, "CREATE (n:Person:Employee {name: 'Bob'})")
        node = db.get_node(1)
        assert "Person" in node.labels
        assert "Employee" in node.labels

    def test_create_edge(self):
        db = GraphDB()
        db.add_node("Person", {"name": "Alice"})
        db.add_node("Person", {"name": "Bob"})
        results = query(db, "CREATE (1)-[:KNOWS {since: 2020}]->(2)")
        assert results[0]["type"] == "KNOWS"
        assert results[0]["props"]["since"] == 2020

    def test_create_edge_invalid_node(self):
        db = GraphDB()
        db.add_node("A")
        with pytest.raises(QueryError):
            query(db, "CREATE (1)-[:X]->(99)")


# ============================================================
# Query Language: DELETE
# ============================================================

class TestQueryDelete:
    def test_delete_node(self):
        db = GraphDB()
        db.add_node("A")
        results = query(db, "DELETE node 1")
        assert results[0]["deleted"] is True
        assert db.get_node(1) is None

    def test_delete_edge(self):
        db = GraphDB()
        db.add_node("A")
        db.add_node("B")
        db.add_edge(1, 2, "X")
        results = query(db, "DELETE edge 1")
        assert results[0]["deleted"] is True


# ============================================================
# Query Language: SET
# ============================================================

class TestQuerySet:
    def test_set_node_prop(self):
        db = GraphDB()
        db.add_node("A", {"x": 1})
        query(db, "SET node 1 x = 42")
        assert db.get_node(1).props["x"] == 42

    def test_set_edge_prop(self):
        db = GraphDB()
        db.add_node("A")
        db.add_node("B")
        db.add_edge(1, 2, "X")
        query(db, "SET edge 1 weight = 5")
        assert db.get_edge(1).props["weight"] == 5

    def test_set_string_value(self):
        db = GraphDB()
        db.add_node("A")
        query(db, "SET node 1 name = 'hello'")
        assert db.get_node(1).props["name"] == "hello"


# ============================================================
# Complex queries
# ============================================================

class TestComplexQueries:
    def _build_company(self):
        db = GraphDB()
        # People
        alice = db.add_node({"Person", "Manager"}, {"name": "Alice", "salary": 90000})
        bob = db.add_node("Person", {"name": "Bob", "salary": 60000})
        carol = db.add_node("Person", {"name": "Carol", "salary": 75000})
        dave = db.add_node("Person", {"name": "Dave", "salary": 55000})
        # Departments
        eng = db.add_node("Department", {"name": "Engineering"})
        sales = db.add_node("Department", {"name": "Sales"})
        # Relationships
        db.add_edge(alice.id, eng.id, "MANAGES")
        db.add_edge(bob.id, eng.id, "WORKS_IN")
        db.add_edge(carol.id, eng.id, "WORKS_IN")
        db.add_edge(dave.id, sales.id, "WORKS_IN")
        db.add_edge(alice.id, bob.id, "SUPERVISES")
        db.add_edge(alice.id, carol.id, "SUPERVISES")
        return db

    def test_two_hop_pattern(self):
        db = self._build_company()
        results = query(db,
            "MATCH (m:Manager)-[:SUPERVISES]->(p:Person)-[:WORKS_IN]->(d:Department) "
            "RETURN m.name, p.name, d.name")
        assert len(results) == 2
        names = {r["p.name"] for r in results}
        assert names == {"Bob", "Carol"}

    def test_where_with_and_or(self):
        db = self._build_company()
        results = query(db,
            "MATCH (p:Person) WHERE (p.salary > 70000 AND p.salary < 100000) OR p.name = 'Dave' "
            "RETURN p.name")
        names = {r["p.name"] for r in results}
        assert "Alice" in names
        assert "Carol" in names
        assert "Dave" in names

    def test_order_limit_combo(self):
        db = self._build_company()
        results = query(db,
            "MATCH (p:Person) RETURN p.name, p.salary ORDER BY p.salary DESC LIMIT 2")
        assert len(results) == 2
        assert results[0]["p.name"] == "Alice"
        assert results[1]["p.name"] == "Carol"

    def test_aggregate_on_filtered(self):
        db = self._build_company()
        results = query(db,
            "MATCH (p:Person) WHERE p.salary > 55000 RETURN AVG(p.salary) AS avg_sal")
        assert results[0]["avg_sal"] == 75000.0

    def test_count_relationships(self):
        db = self._build_company()
        results = query(db,
            "MATCH (a:Manager)-[:SUPERVISES]->(b:Person) RETURN COUNT(*) AS count")
        assert results[0]["count"] == 2

    def test_not_where(self):
        db = self._build_company()
        results = query(db,
            "MATCH (p:Person) WHERE NOT p.name = 'Dave' RETURN p.name")
        names = {r["p.name"] for r in results}
        assert "Dave" not in names
        assert len(names) == 3


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_db_match(self):
        db = GraphDB()
        results = query(db, "MATCH (n:Person) RETURN n")
        assert results == []

    def test_node_no_labels_no_props(self):
        db = GraphDB()
        n = db.add_node()
        assert n.id == 1
        assert len(n.labels) == 0
        assert len(n.props) == 0

    def test_self_loop(self):
        db = GraphDB()
        a = db.add_node("A")
        e = db.add_edge(a.id, a.id, "SELF")
        assert e is not None
        neighbors = db.neighbors(a.id, "SELF", "out")
        assert len(neighbors) == 1
        assert neighbors[0].id == a.id

    def test_multiple_edges_same_nodes(self):
        db = GraphDB()
        a = db.add_node("A")
        b = db.add_node("B")
        db.add_edge(a.id, b.id, "X")
        db.add_edge(a.id, b.id, "Y")
        db.add_edge(a.id, b.id, "X")
        edges = db.edges_of(a.id, direction="out")
        assert len(edges) == 3

    def test_boolean_values(self):
        db = GraphDB()
        results = query(db, "CREATE (n:Flag {active: true, deleted: false})")
        node = db.get_node(1)
        assert node.props["active"] is True
        assert node.props["deleted"] is False

    def test_null_value(self):
        db = GraphDB()
        query(db, "CREATE (n:X {val: null})")
        node = db.get_node(1)
        assert node.props["val"] is None

    def test_query_error_bad_syntax(self):
        db = GraphDB()
        with pytest.raises(QueryError):
            query(db, "INVALID QUERY")

    def test_labels_as_list(self):
        db = GraphDB()
        n = db.add_node(["A", "B", "C"])
        assert "A" in n.labels
        assert "B" in n.labels
        assert "C" in n.labels

    def test_large_graph_performance(self):
        """Test that 1000 nodes + 2000 edges work fine."""
        db = GraphDB()
        for i in range(1000):
            db.add_node("N", {"i": i})
        import random
        random.seed(42)
        for _ in range(2000):
            a = random.randint(1, 1000)
            b = random.randint(1, 1000)
            db.add_edge(a, b, "LINK")
        s = db.stats()
        assert s["nodes"] == 1000
        assert s["edges"] == 2000
        # Path finding should still work
        path = db.shortest_path(1, 500)
        # May or may not find a path, but shouldn't crash

    def test_float_property(self):
        db = GraphDB()
        n = db.add_node("X", {"val": 3.14})
        assert n.props["val"] == 3.14

    def test_negative_number_in_query(self):
        db = GraphDB()
        db.add_node("X", {"val": -5})
        # Negative numbers in where clause
        results = query(db, "MATCH (n:X) WHERE n.val = -5 RETURN n")
        assert len(results) == 1


# ============================================================
# Regression: ensure snapshot doesn't share references
# ============================================================

class TestSnapshotIsolation:
    def test_deep_isolation(self):
        db = GraphDB()
        n = db.add_node("A", {"items": [1, 2, 3]})
        db.save_snapshot()
        n.props["items"].append(4)
        db.restore_snapshot()
        restored = db.get_node(1)
        assert restored.props["items"] == [1, 2, 3]

    def test_id_counter_preserved(self):
        db = GraphDB()
        db.add_node("A")
        db.add_node("B")
        db.save_snapshot()
        db.add_node("C")  # id=3
        db.restore_snapshot()
        n = db.add_node("D")
        # Should get id 3 again since snapshot was from when next_id was 3
        assert n.id == 3
