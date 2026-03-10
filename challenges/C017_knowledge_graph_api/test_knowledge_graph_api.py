"""
Tests for C017: Knowledge Graph API
====================================
Tests the REST API layer over the GraphDB, exercising all endpoints
through synthetic HTTP requests routed through the Router.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C015_graph_db'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C016_http_server'))

from knowledge_graph_api import create_api, node_to_json, edge_to_json
from http_server import Request, Response


# ============================================================
# Test Helpers
# ============================================================

def make_request(router, method, path, body=None, query=None):
    """Build a synthetic request and route it. Returns parsed JSON response."""
    if query:
        qs = '&'.join(f'{k}={v}' for k, v in query.items())
        full_path = f'{path}?{qs}'
    else:
        full_path = path

    body_bytes = json.dumps(body).encode('utf-8') if body is not None else b''
    headers = []
    if body is not None:
        headers.append(('Content-Type', 'application/json'))
        headers.append(('Content-Length', str(len(body_bytes))))

    req = Request(method, full_path, 'HTTP/1.1', headers, body_bytes)
    resp = router.handle(req)

    result = {
        'status': resp.status,
        'body': None,
        'raw': resp.body,
    }
    if resp.body:
        try:
            result['body'] = json.loads(resp.body)
        except (json.JSONDecodeError, ValueError):
            pass
    return result


def fresh_api():
    """Create a fresh API with empty DB."""
    return create_api()


def seeded_api():
    """Create an API with some test data pre-loaded."""
    router, db = create_api()

    # Create some nodes
    alice = db.add_node(labels=['Person'], props={'name': 'Alice', 'age': 30})
    bob = db.add_node(labels=['Person'], props={'name': 'Bob', 'age': 25})
    charlie = db.add_node(labels=['Person'], props={'name': 'Charlie', 'age': 35})
    py = db.add_node(labels=['Language'], props={'name': 'Python', 'version': '3.12'})

    # Create some edges
    db.add_edge(alice.id, bob.id, 'KNOWS', {'since': 2020})
    db.add_edge(bob.id, charlie.id, 'KNOWS', {'since': 2021})
    db.add_edge(alice.id, charlie.id, 'KNOWS', {'since': 2019})
    db.add_edge(alice.id, py.id, 'USES', {'level': 'expert'})

    return router, db


# ============================================================
# Node CRUD Tests
# ============================================================

class TestNodeCRUD:

    def test_create_node_minimal(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes', body={})
        assert r['status'] == 201
        assert r['body']['id'] == 1
        assert r['body']['labels'] == []
        assert r['body']['properties'] == {}

    def test_create_node_with_labels_and_props(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes', body={
            'labels': ['Person', 'Employee'],
            'properties': {'name': 'Alice', 'age': 30}
        })
        assert r['status'] == 201
        assert sorted(r['body']['labels']) == ['Employee', 'Person']
        assert r['body']['properties']['name'] == 'Alice'
        assert r['body']['properties']['age'] == 30

    def test_create_node_string_label(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes', body={
            'labels': 'Person'
        })
        assert r['status'] == 201
        assert r['body']['labels'] == ['Person']

    def test_get_node(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/1')
        assert r['status'] == 200
        assert r['body']['id'] == 1
        assert r['body']['properties']['name'] == 'Alice'

    def test_get_node_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'GET', '/api/nodes/999')
        assert r['status'] == 404

    def test_get_node_invalid_id(self):
        router, db = fresh_api()
        r = make_request(router, 'GET', '/api/nodes/abc')
        assert r['status'] == 400

    def test_update_node_properties(self):
        router, db = seeded_api()
        r = make_request(router, 'PUT', '/api/nodes/1', body={
            'properties': {'age': 31, 'city': 'NYC'}
        })
        assert r['status'] == 200
        assert r['body']['properties']['age'] == 31
        assert r['body']['properties']['city'] == 'NYC'
        assert r['body']['properties']['name'] == 'Alice'  # unchanged

    def test_update_node_remove_properties(self):
        router, db = seeded_api()
        r = make_request(router, 'PUT', '/api/nodes/1', body={
            'remove_properties': ['age']
        })
        assert r['status'] == 200
        assert 'age' not in r['body']['properties']
        assert r['body']['properties']['name'] == 'Alice'

    def test_update_node_labels(self):
        router, db = seeded_api()
        r = make_request(router, 'PUT', '/api/nodes/1', body={
            'labels': ['Person', 'Developer']
        })
        assert r['status'] == 200
        assert sorted(r['body']['labels']) == ['Developer', 'Person']

    def test_update_node_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'PUT', '/api/nodes/999', body={
            'properties': {'x': 1}
        })
        assert r['status'] == 404

    def test_delete_node(self):
        router, db = seeded_api()
        r = make_request(router, 'DELETE', '/api/nodes/1')
        assert r['status'] == 200
        assert r['body']['deleted'] is True

        # Verify gone
        r = make_request(router, 'GET', '/api/nodes/1')
        assert r['status'] == 404

    def test_delete_node_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'DELETE', '/api/nodes/999')
        assert r['status'] == 404

    def test_delete_node_removes_edges(self):
        router, db = seeded_api()
        # Alice (1) has edges. Delete her.
        make_request(router, 'DELETE', '/api/nodes/1')
        # Her edges should be gone
        r = make_request(router, 'GET', '/api/edges')
        edges = r['body']['edges']
        for e in edges:
            assert e['src'] != 1
            assert e['dst'] != 1


# ============================================================
# Node Label Tests
# ============================================================

class TestNodeLabels:

    def test_add_label(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/nodes/1/labels', body={
            'label': 'Developer'
        })
        assert r['status'] == 200
        assert 'Developer' in r['body']['labels']
        assert 'Person' in r['body']['labels']

    def test_add_label_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes/999/labels', body={
            'label': 'Test'
        })
        assert r['status'] == 404

    def test_add_label_missing(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/nodes/1/labels', body={})
        assert r['status'] == 400

    def test_remove_label(self):
        router, db = seeded_api()
        r = make_request(router, 'DELETE', '/api/nodes/1/labels/Person')
        assert r['status'] == 200
        assert 'Person' not in r['body']['labels']

    def test_remove_label_not_found(self):
        router, db = seeded_api()
        r = make_request(router, 'DELETE', '/api/nodes/1/labels/Nonexistent')
        assert r['status'] == 404


# ============================================================
# Node Listing and Filtering Tests
# ============================================================

class TestNodeListing:

    def test_list_all_nodes(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes')
        assert r['status'] == 200
        assert r['body']['total'] == 4
        assert len(r['body']['nodes']) == 4

    def test_list_nodes_with_label_filter(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes', query={'label': 'Person'})
        assert r['status'] == 200
        assert r['body']['total'] == 3
        for n in r['body']['nodes']:
            assert 'Person' in n['labels']

    def test_list_nodes_with_prop_filter(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes', query={
            'label': 'Person', 'name': 'Alice'
        })
        assert r['status'] == 200
        assert r['body']['total'] == 1
        assert r['body']['nodes'][0]['properties']['name'] == 'Alice'

    def test_list_nodes_pagination(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes', query={
            'limit': '2', 'offset': '0'
        })
        assert r['status'] == 200
        assert len(r['body']['nodes']) == 2
        assert r['body']['total'] == 4

        r2 = make_request(router, 'GET', '/api/nodes', query={
            'limit': '2', 'offset': '2'
        })
        assert len(r2['body']['nodes']) == 2

    def test_list_nodes_empty(self):
        router, db = fresh_api()
        r = make_request(router, 'GET', '/api/nodes')
        assert r['status'] == 200
        assert r['body']['total'] == 0
        assert r['body']['nodes'] == []


# ============================================================
# Node Neighbors and Edges Tests
# ============================================================

class TestNodeRelationships:

    def test_get_neighbors_out(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/1/neighbors', query={
            'direction': 'out'
        })
        assert r['status'] == 200
        names = [n['properties']['name'] for n in r['body']['neighbors']]
        assert 'Bob' in names
        assert 'Charlie' in names
        assert 'Python' in names

    def test_get_neighbors_in(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/2/neighbors', query={
            'direction': 'in'
        })
        assert r['status'] == 200
        names = [n['properties']['name'] for n in r['body']['neighbors']]
        assert 'Alice' in names

    def test_get_neighbors_with_edge_type(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/1/neighbors', query={
            'direction': 'out', 'edge_type': 'KNOWS'
        })
        assert r['status'] == 200
        names = [n['properties']['name'] for n in r['body']['neighbors']]
        assert 'Bob' in names
        assert 'Charlie' in names
        assert 'Python' not in names

    def test_get_neighbors_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'GET', '/api/nodes/999/neighbors')
        assert r['status'] == 404

    def test_get_neighbors_invalid_direction(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/1/neighbors', query={
            'direction': 'sideways'
        })
        assert r['status'] == 400

    def test_get_node_edges(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/1/edges')
        assert r['status'] == 200
        assert r['body']['count'] == 3  # 3 outgoing edges from Alice

    def test_get_node_edges_with_type(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/nodes/1/edges', query={
            'edge_type': 'USES'
        })
        assert r['status'] == 200
        assert r['body']['count'] == 1
        assert r['body']['edges'][0]['type'] == 'USES'


# ============================================================
# Edge CRUD Tests
# ============================================================

class TestEdgeCRUD:

    def test_create_edge(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/edges', body={
            'src': 2, 'dst': 1, 'type': 'FOLLOWS',
            'properties': {'public': True}
        })
        assert r['status'] == 201
        assert r['body']['type'] == 'FOLLOWS'
        assert r['body']['src'] == 2
        assert r['body']['dst'] == 1
        assert r['body']['properties']['public'] is True

    def test_create_edge_missing_fields(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/edges', body={
            'src': 1
        })
        assert r['status'] == 400

    def test_create_edge_invalid_nodes(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/edges', body={
            'src': 999, 'dst': 1, 'type': 'FAKE'
        })
        assert r['status'] == 404

    def test_get_edge(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges/1')
        assert r['status'] == 200
        assert r['body']['type'] == 'KNOWS'
        assert r['body']['properties']['since'] == 2020

    def test_get_edge_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'GET', '/api/edges/999')
        assert r['status'] == 404

    def test_update_edge(self):
        router, db = seeded_api()
        r = make_request(router, 'PUT', '/api/edges/1', body={
            'properties': {'since': 2018, 'strength': 'close'}
        })
        assert r['status'] == 200
        assert r['body']['properties']['since'] == 2018
        assert r['body']['properties']['strength'] == 'close'

    def test_update_edge_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'PUT', '/api/edges/999', body={
            'properties': {'x': 1}
        })
        assert r['status'] == 404

    def test_delete_edge(self):
        router, db = seeded_api()
        r = make_request(router, 'DELETE', '/api/edges/1')
        assert r['status'] == 200
        assert r['body']['deleted'] is True

        r = make_request(router, 'GET', '/api/edges/1')
        assert r['status'] == 404

    def test_delete_edge_not_found(self):
        router, db = fresh_api()
        r = make_request(router, 'DELETE', '/api/edges/999')
        assert r['status'] == 404


# ============================================================
# Edge Listing Tests
# ============================================================

class TestEdgeListing:

    def test_list_all_edges(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges')
        assert r['status'] == 200
        assert r['body']['total'] == 4

    def test_list_edges_by_type(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges', query={'type': 'KNOWS'})
        assert r['status'] == 200
        assert r['body']['total'] == 3
        for e in r['body']['edges']:
            assert e['type'] == 'KNOWS'

    def test_list_edges_by_src(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges', query={'src': '1'})
        assert r['status'] == 200
        assert r['body']['total'] == 3  # Alice has 3 outgoing

    def test_list_edges_by_dst(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges', query={'dst': '3'})
        assert r['status'] == 200
        for e in r['body']['edges']:
            assert e['dst'] == 3

    def test_list_edges_pagination(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges', query={
            'limit': '2', 'offset': '0'
        })
        assert len(r['body']['edges']) == 2
        assert r['body']['total'] == 4

    def test_list_edges_combined_filters(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/edges', query={
            'type': 'KNOWS', 'src': '1'
        })
        assert r['status'] == 200
        assert r['body']['total'] == 2  # Alice KNOWS Bob and Charlie


# ============================================================
# GQL Query Tests
# ============================================================

class TestQueryEndpoint:

    def test_match_nodes_by_label(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'MATCH (n:Person) RETURN n'
        })
        assert r['status'] == 200
        assert r['body']['count'] == 3

    def test_match_with_property_filter(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'MATCH (n:Person {name: "Alice"}) RETURN n'
        })
        assert r['status'] == 200
        assert r['body']['count'] == 1

    def test_match_edges(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b'
        })
        assert r['status'] == 200
        assert r['body']['count'] == 3

    def test_create_via_query(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'CREATE (n:City {name: "NYC"})'
        })
        assert r['status'] == 200
        # Verify node was created
        nodes = db.find_nodes(label='City')
        assert len(nodes) == 1
        assert nodes[0].props['name'] == 'NYC'

    def test_query_missing_body(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/query', body={})
        assert r['status'] == 400

    def test_query_invalid_gql(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'INVALID GIBBERISH'
        })
        assert r['status'] == 400

    def test_query_with_where(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'MATCH (n:Person) WHERE n.age > 28 RETURN n'
        })
        assert r['status'] == 200
        # Alice (30) and Charlie (35) match
        assert r['body']['count'] == 2


# ============================================================
# Path Finding Tests
# ============================================================

class TestPathFinding:

    def test_shortest_path(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1', 'dst': '3'
        })
        assert r['status'] == 200
        assert r['body']['found'] is True
        assert r['body']['path'] == [1, 3]  # direct edge Alice->Charlie
        assert r['body']['length'] == 1

    def test_shortest_path_indirect(self):
        router, db = seeded_api()
        # Bob (2) -> Charlie (3) is direct
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '2', 'dst': '3'
        })
        assert r['status'] == 200
        assert r['body']['found'] is True
        assert r['body']['path'] == [2, 3]

    def test_shortest_path_not_found(self):
        router, db = seeded_api()
        # Charlie (3) has no outgoing KNOWS to Alice (1)
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '3', 'dst': '1'
        })
        assert r['status'] == 200
        assert r['body']['found'] is False
        assert r['body']['path'] is None

    def test_shortest_path_missing_params(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1'
        })
        assert r['status'] == 400

    def test_shortest_path_with_edge_type(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1', 'dst': '4', 'edge_type': 'KNOWS'
        })
        assert r['status'] == 200
        assert r['body']['found'] is False  # no KNOWS path to Python

    def test_shortest_path_with_nodes(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1', 'dst': '3'
        })
        assert r['body']['nodes'] is not None
        assert len(r['body']['nodes']) == len(r['body']['path'])

    def test_all_paths(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/all', query={
            'src': '1', 'dst': '3'
        })
        assert r['status'] == 200
        assert r['body']['count'] == 2  # direct and via Bob
        paths = r['body']['paths']
        assert [1, 3] in paths
        assert [1, 2, 3] in paths

    def test_all_paths_missing_params(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/all', query={})
        assert r['status'] == 400

    def test_all_paths_with_max_depth(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/all', query={
            'src': '1', 'dst': '3', 'max_depth': '1'
        })
        assert r['status'] == 200
        # Only direct path fits in depth 1
        assert r['body']['count'] == 1


# ============================================================
# Index Tests
# ============================================================

class TestIndexes:

    def test_create_index(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/indexes', body={
            'label': 'Person', 'property': 'name'
        })
        assert r['status'] == 201
        assert r['body']['created'] is True

    def test_create_index_missing_fields(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/indexes', body={
            'label': 'Person'
        })
        assert r['status'] == 400

    def test_list_indexes(self):
        router, db = seeded_api()
        db.create_index('Person', 'name')
        db.create_index('Person', 'age')
        r = make_request(router, 'GET', '/api/indexes')
        assert r['status'] == 200
        assert len(r['body']['indexes']) == 2

    def test_index_speeds_lookup(self):
        router, db = seeded_api()
        # Create index and verify lookup still works
        make_request(router, 'POST', '/api/indexes', body={
            'label': 'Person', 'property': 'name'
        })
        r = make_request(router, 'GET', '/api/nodes', query={
            'label': 'Person', 'name': 'Alice'
        })
        assert r['body']['total'] == 1


# ============================================================
# Snapshot Tests
# ============================================================

class TestSnapshots:

    def test_save_snapshot(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/snapshots')
        assert r['status'] == 201
        assert r['body']['snapshot_index'] == 0

    def test_list_snapshots(self):
        router, db = seeded_api()
        make_request(router, 'POST', '/api/snapshots')
        make_request(router, 'POST', '/api/snapshots')
        r = make_request(router, 'GET', '/api/snapshots')
        assert r['status'] == 200
        assert r['body']['count'] == 2

    def test_restore_snapshot(self):
        router, db = seeded_api()
        # Save state with 4 nodes
        make_request(router, 'POST', '/api/snapshots')

        # Delete a node
        make_request(router, 'DELETE', '/api/nodes/1')
        r = make_request(router, 'GET', '/api/stats')
        assert r['body']['nodes'] == 3

        # Restore
        r = make_request(router, 'POST', '/api/snapshots/restore', body={
            'index': 0
        })
        assert r['status'] == 200
        assert r['body']['restored'] is True

        # Verify restoration
        r = make_request(router, 'GET', '/api/stats')
        assert r['body']['nodes'] == 4

    def test_restore_default_latest(self):
        router, db = seeded_api()
        make_request(router, 'POST', '/api/snapshots')
        # Modify
        make_request(router, 'DELETE', '/api/nodes/1')
        # Restore latest (default index=-1)
        r = make_request(router, 'POST', '/api/snapshots/restore', body={})
        assert r['status'] == 200
        assert r['body']['restored'] is True

    def test_restore_no_snapshots(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/snapshots/restore', body={
            'index': 0
        })
        assert r['status'] == 404

    def test_restore_invalid_index(self):
        router, db = seeded_api()
        make_request(router, 'POST', '/api/snapshots')
        r = make_request(router, 'POST', '/api/snapshots/restore', body={
            'index': 99
        })
        assert r['status'] == 404


# ============================================================
# Stats Tests
# ============================================================

class TestStats:

    def test_basic_stats(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/stats')
        assert r['status'] == 200
        assert r['body']['nodes'] == 4
        assert r['body']['edges'] == 4
        assert r['body']['snapshots'] == 0
        assert r['body']['indexes'] == 0

    def test_stats_after_modifications(self):
        router, db = seeded_api()
        make_request(router, 'DELETE', '/api/nodes/1')
        r = make_request(router, 'GET', '/api/stats')
        assert r['body']['nodes'] == 3
        # Alice's edges should also be gone
        assert r['body']['edges'] < 4


# ============================================================
# Batch Operation Tests
# ============================================================

class TestBatchOperations:

    def test_batch_create_nodes(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': [
                {'method': 'POST', 'path': '/api/nodes',
                 'body': {'labels': ['A'], 'properties': {'n': 1}}},
                {'method': 'POST', 'path': '/api/nodes',
                 'body': {'labels': ['B'], 'properties': {'n': 2}}},
                {'method': 'POST', 'path': '/api/nodes',
                 'body': {'labels': ['C'], 'properties': {'n': 3}}},
            ]
        })
        assert r['status'] == 200
        assert r['body']['count'] == 3
        assert all(res['status'] == 201 for res in r['body']['results'])

        # Verify all 3 nodes exist
        r2 = make_request(router, 'GET', '/api/stats')
        assert r2['body']['nodes'] == 3

    def test_batch_mixed_operations(self):
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': [
                {'method': 'POST', 'path': '/api/nodes',
                 'body': {'labels': ['City'], 'properties': {'name': 'Paris'}}},
                {'method': 'PUT', 'path': '/api/nodes/1',
                 'body': {'properties': {'city': 'Paris'}}},
            ]
        })
        assert r['status'] == 200
        assert r['body']['count'] == 2

    def test_batch_rollback_on_failure(self):
        router, db = fresh_api()
        # Create a node first
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['Original'], 'properties': {'keep': True}
        })

        # Batch: create another then try to update nonexistent -> should rollback
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': [
                {'method': 'POST', 'path': '/api/nodes',
                 'body': {'labels': ['New'], 'properties': {'n': 2}}},
                {'method': 'PUT', 'path': '/api/nodes/999',
                 'body': {'properties': {'x': 1}}},
            ]
        })
        assert r['status'] == 400
        assert r['body']['failed_index'] == 1

        # The first operation should have been rolled back
        r2 = make_request(router, 'GET', '/api/stats')
        assert r2['body']['nodes'] == 1  # only original node

    def test_batch_empty(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': []
        })
        assert r['status'] == 200
        assert r['body']['count'] == 0

    def test_batch_too_many(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': [{'method': 'GET', 'path': '/api/stats'}] * 101
        })
        assert r['status'] == 400

    def test_batch_create_and_connect(self):
        router, db = fresh_api()
        # Create two nodes
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['A'], 'properties': {'name': 'a'}
        })
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['B'], 'properties': {'name': 'b'}
        })

        # Batch: create edge between them and update properties
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': [
                {'method': 'POST', 'path': '/api/edges',
                 'body': {'src': 1, 'dst': 2, 'type': 'LINKS'}},
                {'method': 'PUT', 'path': '/api/nodes/1',
                 'body': {'properties': {'linked': True}}},
            ]
        })
        assert r['status'] == 200
        assert r['body']['count'] == 2


# ============================================================
# Request Validation Tests
# ============================================================

class TestValidation:

    def test_invalid_json_body(self):
        router, db = fresh_api()
        req = Request('POST', '/api/nodes', 'HTTP/1.1',
                       [('Content-Type', 'application/json')],
                       b'not json')
        resp = router.handle(req)
        assert resp.status == 400

    def test_wrong_content_type(self):
        router, db = fresh_api()
        req = Request('POST', '/api/nodes', 'HTTP/1.1',
                       [('Content-Type', 'text/plain')],
                       b'{"labels": ["Test"]}')
        resp = router.handle(req)
        assert resp.status == 415

    def test_empty_body_for_post(self):
        router, db = fresh_api()
        req = Request('POST', '/api/nodes', 'HTTP/1.1', [], b'')
        resp = router.handle(req)
        assert resp.status == 400

    def test_method_not_allowed(self):
        router, db = fresh_api()
        r = make_request(router, 'PATCH', '/api/nodes')
        assert r['status'] == 405


# ============================================================
# CORS Tests
# ============================================================

class TestCORS:

    def test_cors_preflight(self):
        router, db = fresh_api()
        req = Request('OPTIONS', '/api/nodes', 'HTTP/1.1', [], b'')
        resp = router.handle(req)
        assert resp.status == 204
        headers_dict = dict(resp.headers.items())
        assert headers_dict.get('Access-Control-Allow-Origin') == '*'

    def test_cors_on_response(self):
        router, db = seeded_api()
        req = Request('GET', '/api/stats', 'HTTP/1.1', [], b'')
        resp = router.handle(req)
        headers_dict = dict(resp.headers.items())
        assert headers_dict.get('Access-Control-Allow-Origin') == '*'


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:

    def test_404_nonexistent_route(self):
        router, db = fresh_api()
        r = make_request(router, 'GET', '/api/nonexistent')
        assert r['status'] == 404

    def test_error_handler_catches_exceptions(self):
        """Verify the error handler catches exceptions in route handlers."""
        router, db = fresh_api()

        # Register a route that always throws
        def bad_handler(req):
            raise RuntimeError("kaboom")

        router.get('/api/explode', bad_handler)

        r = make_request(router, 'GET', '/api/explode')
        assert r['status'] == 500
        assert 'error' in r['body']


# ============================================================
# Integration Tests (multi-step workflows)
# ============================================================

class TestIntegration:

    def test_full_crud_workflow(self):
        """Create, read, update, delete a node."""
        router, db = fresh_api()

        # Create
        r = make_request(router, 'POST', '/api/nodes', body={
            'labels': ['Person'], 'properties': {'name': 'Test', 'score': 0}
        })
        assert r['status'] == 201
        nid = r['body']['id']

        # Read
        r = make_request(router, 'GET', f'/api/nodes/{nid}')
        assert r['body']['properties']['name'] == 'Test'

        # Update
        r = make_request(router, 'PUT', f'/api/nodes/{nid}', body={
            'properties': {'score': 100}
        })
        assert r['body']['properties']['score'] == 100

        # Delete
        r = make_request(router, 'DELETE', f'/api/nodes/{nid}')
        assert r['body']['deleted'] is True

        # Confirm gone
        r = make_request(router, 'GET', f'/api/nodes/{nid}')
        assert r['status'] == 404

    def test_build_graph_and_query(self):
        """Build a graph through the API and query it."""
        router, db = fresh_api()

        # Create nodes
        for name in ['Alice', 'Bob', 'Charlie']:
            make_request(router, 'POST', '/api/nodes', body={
                'labels': ['Person'], 'properties': {'name': name}
            })

        # Create edges
        make_request(router, 'POST', '/api/edges', body={
            'src': 1, 'dst': 2, 'type': 'KNOWS'
        })
        make_request(router, 'POST', '/api/edges', body={
            'src': 2, 'dst': 3, 'type': 'KNOWS'
        })

        # Query: find path from Alice to Charlie
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1', 'dst': '3'
        })
        assert r['body']['found'] is True
        assert r['body']['path'] == [1, 2, 3]
        assert r['body']['length'] == 2

        # Query via GQL
        r = make_request(router, 'POST', '/api/query', body={
            'query': 'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b'
        })
        assert r['body']['count'] == 2

    def test_snapshot_restore_workflow(self):
        """Save state, modify, restore."""
        router, db = fresh_api()

        # Build initial state
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['A'], 'properties': {'v': 1}
        })
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['B'], 'properties': {'v': 2}
        })

        # Snapshot
        make_request(router, 'POST', '/api/snapshots')

        # Modify heavily
        make_request(router, 'DELETE', '/api/nodes/1')
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['C'], 'properties': {'v': 3}
        })
        make_request(router, 'POST', '/api/nodes', body={
            'labels': ['D'], 'properties': {'v': 4}
        })

        r = make_request(router, 'GET', '/api/stats')
        assert r['body']['nodes'] == 3  # B, C, D

        # Restore
        make_request(router, 'POST', '/api/snapshots/restore', body={'index': 0})

        r = make_request(router, 'GET', '/api/stats')
        assert r['body']['nodes'] == 2  # A and B

        # Verify node 1 is back
        r = make_request(router, 'GET', '/api/nodes/1')
        assert r['status'] == 200
        assert r['body']['properties']['v'] == 1

    def test_index_and_filtered_query(self):
        """Create index and verify filtered queries still work."""
        router, db = fresh_api()

        # Build data
        for i in range(10):
            make_request(router, 'POST', '/api/nodes', body={
                'labels': ['Item'], 'properties': {'rank': i, 'name': f'item_{i}'}
            })

        # Create index
        make_request(router, 'POST', '/api/indexes', body={
            'label': 'Item', 'property': 'rank'
        })

        # Filter by indexed property
        r = make_request(router, 'GET', '/api/nodes', query={
            'label': 'Item', 'rank': '5'
        })
        assert r['body']['total'] == 1
        assert r['body']['nodes'][0]['properties']['name'] == 'item_5'

    def test_edge_operations_after_node_delete(self):
        """Verify edges are cleaned up when nodes are deleted."""
        router, db = fresh_api()

        # Create triangle
        for name in ['A', 'B', 'C']:
            make_request(router, 'POST', '/api/nodes', body={
                'labels': ['N'], 'properties': {'name': name}
            })

        make_request(router, 'POST', '/api/edges', body={
            'src': 1, 'dst': 2, 'type': 'E'
        })
        make_request(router, 'POST', '/api/edges', body={
            'src': 2, 'dst': 3, 'type': 'E'
        })
        make_request(router, 'POST', '/api/edges', body={
            'src': 1, 'dst': 3, 'type': 'E'
        })

        assert make_request(router, 'GET', '/api/stats')['body']['edges'] == 3

        # Delete node 1
        make_request(router, 'DELETE', '/api/nodes/1')

        # Only edge 2->3 should remain
        r = make_request(router, 'GET', '/api/edges')
        assert r['body']['total'] == 1
        assert r['body']['edges'][0]['src'] == 2
        assert r['body']['edges'][0]['dst'] == 3

    def test_batch_with_query(self):
        """Batch operation that includes GQL queries."""
        router, db = seeded_api()
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': [
                {'method': 'POST', 'path': '/api/query',
                 'body': {'query': 'MATCH (n:Person) RETURN n'}},
                {'method': 'GET', 'path': '/api/stats'},
            ]
        })
        assert r['status'] == 200
        assert r['body']['count'] == 2
        # First result should be query results
        assert r['body']['results'][0]['data']['count'] == 3

    def test_large_graph_operations(self):
        """Test with a moderately large graph."""
        router, db = fresh_api()

        # Create 50 nodes
        for i in range(50):
            make_request(router, 'POST', '/api/nodes', body={
                'labels': ['Node'], 'properties': {'idx': i}
            })

        # Create a chain of edges
        for i in range(1, 50):
            make_request(router, 'POST', '/api/edges', body={
                'src': i, 'dst': i + 1, 'type': 'NEXT'
            })

        # Stats
        r = make_request(router, 'GET', '/api/stats')
        assert r['body']['nodes'] == 50
        assert r['body']['edges'] == 49

        # Find path from first to last (chain is 49 hops, need max_depth > 20)
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1', 'dst': '50', 'max_depth': '50'
        })
        assert r['body']['found'] is True
        assert r['body']['length'] == 49

        # Pagination
        r = make_request(router, 'GET', '/api/nodes', query={
            'limit': '10', 'offset': '20'
        })
        assert len(r['body']['nodes']) == 10
        assert r['body']['total'] == 50


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:

    def test_self_loop_edge(self):
        router, db = fresh_api()
        make_request(router, 'POST', '/api/nodes', body={'labels': ['N']})
        r = make_request(router, 'POST', '/api/edges', body={
            'src': 1, 'dst': 1, 'type': 'SELF'
        })
        assert r['status'] == 201
        assert r['body']['src'] == 1
        assert r['body']['dst'] == 1

    def test_multiple_edges_same_pair(self):
        router, db = fresh_api()
        make_request(router, 'POST', '/api/nodes', body={'labels': ['A']})
        make_request(router, 'POST', '/api/nodes', body={'labels': ['B']})

        make_request(router, 'POST', '/api/edges', body={
            'src': 1, 'dst': 2, 'type': 'LINK1'
        })
        make_request(router, 'POST', '/api/edges', body={
            'src': 1, 'dst': 2, 'type': 'LINK2'
        })

        r = make_request(router, 'GET', '/api/nodes/1/edges', query={
            'direction': 'out'
        })
        assert r['body']['count'] == 2

    def test_node_with_no_labels(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes', body={
            'properties': {'x': 1}
        })
        assert r['status'] == 201
        assert r['body']['labels'] == []

    def test_node_with_complex_properties(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes', body={
            'labels': ['Complex'],
            'properties': {
                'name': 'test',
                'tags': ['a', 'b', 'c'],
                'nested': {'key': 'value'},
                'score': 3.14,
                'active': True,
            }
        })
        assert r['status'] == 201
        p = r['body']['properties']
        assert p['tags'] == ['a', 'b', 'c']
        assert p['nested'] == {'key': 'value'}
        assert p['score'] == 3.14
        assert p['active'] is True

    def test_unicode_properties(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/nodes', body={
            'labels': ['Test'],
            'properties': {'name': 'test data', 'note': 'plain text'}
        })
        assert r['status'] == 201

    def test_empty_operations_list(self):
        router, db = fresh_api()
        r = make_request(router, 'POST', '/api/batch', body={
            'operations': []
        })
        assert r['status'] == 200
        assert r['body']['count'] == 0

    def test_shortest_path_same_node(self):
        router, db = seeded_api()
        r = make_request(router, 'GET', '/api/paths/shortest', query={
            'src': '1', 'dst': '1'
        })
        assert r['body']['found'] is True
        assert r['body']['path'] == [1]
        assert r['body']['length'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
