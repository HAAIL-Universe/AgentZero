"""
C017: Knowledge Graph API -- Composing C015 (GraphDB) + C016 (HTTP Server)
==========================================================================
A RESTful API service over an in-memory property graph database.

Endpoints:
  Nodes:
    GET    /api/nodes              - List/filter nodes
    POST   /api/nodes              - Create node
    GET    /api/nodes/:id          - Get node by ID
    PUT    /api/nodes/:id          - Update node properties
    DELETE /api/nodes/:id          - Delete node
    POST   /api/nodes/:id/labels   - Add label
    DELETE /api/nodes/:id/labels/:label - Remove label
    GET    /api/nodes/:id/neighbors - Get neighbors
    GET    /api/nodes/:id/edges    - Get edges

  Edges:
    GET    /api/edges              - List edges (with optional filters)
    POST   /api/edges              - Create edge
    GET    /api/edges/:id          - Get edge by ID
    PUT    /api/edges/:id          - Update edge properties
    DELETE /api/edges/:id          - Delete edge

  Query:
    POST   /api/query              - Execute GQL query

  Paths:
    GET    /api/paths/shortest     - Shortest path (src, dst params)
    GET    /api/paths/all          - All paths (src, dst params)

  Indexes:
    POST   /api/indexes            - Create index
    GET    /api/indexes            - List indexes

  Snapshots:
    POST   /api/snapshots          - Save snapshot
    POST   /api/snapshots/restore  - Restore snapshot
    GET    /api/snapshots          - List snapshot count

  Stats:
    GET    /api/stats              - Database statistics

  Batch:
    POST   /api/batch              - Execute multiple operations atomically

Difficulty: 4
Domain: System Composition / API Design
Composes: C015 (GraphDB), C016 (HTTP Server)
"""

import sys
import os
import json
import traceback

# Import from sibling challenge directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C015_graph_db'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C016_http_server'))

from graph_db import GraphDB, Node, Edge, QueryExecutor, QueryError
from http_server import Router, Response, Request, HTTPServer


# ============================================================
# JSON Serialization Helpers
# ============================================================

def node_to_json(node):
    """Convert a Node to a JSON-serializable dict."""
    return {
        'id': node.id,
        'labels': sorted(node.labels),
        'properties': dict(node.props),
    }


def edge_to_json(edge):
    """Convert an Edge to a JSON-serializable dict."""
    return {
        'id': edge.id,
        'src': edge.src,
        'dst': edge.dst,
        'type': edge.edge_type,
        'properties': dict(edge.props),
    }


# ============================================================
# Request Validation
# ============================================================

def require_json(req):
    """Validate request has JSON body. Returns (data, error_response)."""
    ct = req.content_type
    if ct and 'application/json' not in ct:
        return None, Response.json_response(
            {'error': 'Content-Type must be application/json'}, 415)
    try:
        data = req.json()
        if data is None:
            return None, Response.json_response(
                {'error': 'Request body is required'}, 400)
        return data, None
    except (json.JSONDecodeError, ValueError) as e:
        return None, Response.json_response(
            {'error': f'Invalid JSON: {str(e)}'}, 400)


def parse_int_param(req, name):
    """Extract an integer path parameter. Returns (value, error_response)."""
    raw = req.params.get(name)
    if raw is None:
        return None, Response.json_response(
            {'error': f'Missing parameter: {name}'}, 400)
    try:
        return int(raw), None
    except (ValueError, TypeError):
        return None, Response.json_response(
            {'error': f'Parameter {name} must be an integer'}, 400)


def parse_int_query(req, name, default=None):
    """Extract an integer query parameter. Returns (value, error_response)."""
    raw = req.query.get(name)
    if raw is None:
        return default, None
    try:
        return int(raw), None
    except (ValueError, TypeError):
        return None, Response.json_response(
            {'error': f'Query parameter {name} must be an integer'}, 400)


# ============================================================
# API Builder
# ============================================================

def create_api(db=None):
    """Create a Router with all Knowledge Graph API endpoints wired to db.

    Args:
        db: GraphDB instance. If None, creates a new one.

    Returns:
        (router, db) tuple.
    """
    if db is None:
        db = GraphDB()

    router = Router()

    # --- Middleware: JSON error handling ---
    def error_handler(req, err):
        tb = traceback.format_exc()
        return Response.json_response(
            {'error': 'Internal server error', 'detail': str(err)}, 500)

    router.on_error(error_handler)

    # --- Middleware: CORS ---
    def cors_middleware(req, next_fn):
        if req.method == 'OPTIONS':
            r = Response(204)
            r.headers['Access-Control-Allow-Origin'] = '*'
            r.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, PATCH, OPTIONS'
            r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return r
        return None  # let routing proceed

    router.use(cors_middleware)

    # Wrap router.handle to add CORS headers to all responses
    _original_handle = router.handle

    def handle_with_cors(request):
        resp = _original_handle(request)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    router.handle = handle_with_cors

    # ================================================================
    # NODE ENDPOINTS
    # ================================================================

    def list_nodes(req):
        """GET /api/nodes -- List nodes with optional filtering."""
        label = req.query.get('label')
        limit, err = parse_int_query(req, 'limit', default=100)
        if err:
            return err
        offset, err = parse_int_query(req, 'offset', default=0)
        if err:
            return err

        # Collect filter props from query params (skip reserved names)
        reserved = {'label', 'limit', 'offset'}
        filter_props = {k: _coerce_value(v) for k, v in req.query.items()
                        if k not in reserved}

        if label or filter_props:
            nodes = db.find_nodes(label=label, **filter_props)
        else:
            nodes = list(db.nodes.values())

        # Sort by ID for deterministic output
        nodes.sort(key=lambda n: n.id)
        total = len(nodes)
        page = nodes[offset:offset + limit]

        return Response.json_response({
            'nodes': [node_to_json(n) for n in page],
            'total': total,
            'offset': offset,
            'limit': limit,
        })

    def create_node(req):
        """POST /api/nodes -- Create a new node."""
        data, err = require_json(req)
        if err:
            return err

        labels = data.get('labels', [])
        if isinstance(labels, str):
            labels = [labels]
        props = data.get('properties', {})

        if not isinstance(props, dict):
            return Response.json_response(
                {'error': 'properties must be an object'}, 400)

        node = db.add_node(labels=labels, props=props)
        return Response.json_response(node_to_json(node), 201)

    def get_node(req):
        """GET /api/nodes/:id -- Get a node by ID."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        node = db.get_node(nid)
        if not node:
            return Response.json_response({'error': 'Node not found'}, 404)

        return Response.json_response(node_to_json(node))

    def update_node(req):
        """PUT /api/nodes/:id -- Update node properties."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        data, err = require_json(req)
        if err:
            return err

        node = db.get_node(nid)
        if not node:
            return Response.json_response({'error': 'Node not found'}, 404)

        props = data.get('properties', {})
        remove = data.get('remove_properties', [])

        for key, val in props.items():
            db.set_node_prop(nid, key, val)

        for key in remove:
            db.remove_node_prop(nid, key)

        # Labels update
        if 'labels' in data:
            new_labels = set(data['labels']) if isinstance(data['labels'], list) else {data['labels']}
            current = set(node.labels)
            for l in new_labels - current:
                db.add_label(nid, l)
            for l in current - new_labels:
                db.remove_label(nid, l)

        updated = db.get_node(nid)
        return Response.json_response(node_to_json(updated))

    def delete_node(req):
        """DELETE /api/nodes/:id -- Delete a node."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        if not db.delete_node(nid):
            return Response.json_response({'error': 'Node not found'}, 404)

        return Response.json_response({'deleted': True, 'id': nid})

    def add_node_label(req):
        """POST /api/nodes/:id/labels -- Add a label to a node."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        data, err = require_json(req)
        if err:
            return err

        label = data.get('label')
        if not label or not isinstance(label, str):
            return Response.json_response(
                {'error': 'label (string) is required'}, 400)

        node = db.get_node(nid)
        if not node:
            return Response.json_response({'error': 'Node not found'}, 404)

        db.add_label(nid, label)
        updated = db.get_node(nid)
        return Response.json_response(node_to_json(updated))

    def remove_node_label(req):
        """DELETE /api/nodes/:id/labels/:label -- Remove label from node."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        label = req.params.get('label')
        node = db.get_node(nid)
        if not node:
            return Response.json_response({'error': 'Node not found'}, 404)

        if label not in node.labels:
            return Response.json_response({'error': 'Label not found on node'}, 404)

        db.remove_label(nid, label)
        updated = db.get_node(nid)
        return Response.json_response(node_to_json(updated))

    def get_neighbors(req):
        """GET /api/nodes/:id/neighbors -- Get neighboring nodes."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        node = db.get_node(nid)
        if not node:
            return Response.json_response({'error': 'Node not found'}, 404)

        edge_type = req.query.get('edge_type')
        direction = req.query.get('direction', 'out')
        if direction not in ('out', 'in', 'both'):
            return Response.json_response(
                {'error': 'direction must be out, in, or both'}, 400)

        neighbors = db.neighbors(nid, edge_type=edge_type, direction=direction)
        return Response.json_response({
            'neighbors': [node_to_json(n) for n in neighbors],
            'count': len(neighbors),
        })

    def get_node_edges(req):
        """GET /api/nodes/:id/edges -- Get edges connected to a node."""
        nid, err = parse_int_param(req, 'id')
        if err:
            return err

        node = db.get_node(nid)
        if not node:
            return Response.json_response({'error': 'Node not found'}, 404)

        edge_type = req.query.get('edge_type')
        direction = req.query.get('direction', 'both')
        if direction not in ('out', 'in', 'both'):
            return Response.json_response(
                {'error': 'direction must be out, in, or both'}, 400)

        edges = db.edges_of(nid, edge_type=edge_type, direction=direction)
        return Response.json_response({
            'edges': [edge_to_json(e) for e in edges],
            'count': len(edges),
        })

    # ================================================================
    # EDGE ENDPOINTS
    # ================================================================

    def list_edges(req):
        """GET /api/edges -- List edges with optional filtering."""
        edge_type = req.query.get('type')
        src, err = parse_int_query(req, 'src')
        if err:
            return err
        dst, err = parse_int_query(req, 'dst')
        if err:
            return err
        limit, err = parse_int_query(req, 'limit', default=100)
        if err:
            return err
        offset, err = parse_int_query(req, 'offset', default=0)
        if err:
            return err

        edges = list(db.edges.values())

        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        if src is not None:
            edges = [e for e in edges if e.src == src]
        if dst is not None:
            edges = [e for e in edges if e.dst == dst]

        edges.sort(key=lambda e: e.id)
        total = len(edges)
        page = edges[offset:offset + limit]

        return Response.json_response({
            'edges': [edge_to_json(e) for e in page],
            'total': total,
            'offset': offset,
            'limit': limit,
        })

    def create_edge(req):
        """POST /api/edges -- Create an edge."""
        data, err = require_json(req)
        if err:
            return err

        src = data.get('src')
        dst = data.get('dst')
        edge_type = data.get('type')
        props = data.get('properties', {})

        if src is None or dst is None or not edge_type:
            return Response.json_response(
                {'error': 'src, dst, and type are required'}, 400)

        try:
            src = int(src)
            dst = int(dst)
        except (ValueError, TypeError):
            return Response.json_response(
                {'error': 'src and dst must be integers'}, 400)

        if not isinstance(props, dict):
            return Response.json_response(
                {'error': 'properties must be an object'}, 400)

        edge = db.add_edge(src, dst, edge_type, props)
        if edge is None:
            return Response.json_response(
                {'error': 'Source or destination node not found'}, 404)

        return Response.json_response(edge_to_json(edge), 201)

    def get_edge(req):
        """GET /api/edges/:id -- Get edge by ID."""
        eid, err = parse_int_param(req, 'id')
        if err:
            return err

        edge = db.get_edge(eid)
        if not edge:
            return Response.json_response({'error': 'Edge not found'}, 404)

        return Response.json_response(edge_to_json(edge))

    def update_edge(req):
        """PUT /api/edges/:id -- Update edge properties."""
        eid, err = parse_int_param(req, 'id')
        if err:
            return err

        data, err = require_json(req)
        if err:
            return err

        edge = db.get_edge(eid)
        if not edge:
            return Response.json_response({'error': 'Edge not found'}, 404)

        props = data.get('properties', {})
        for key, val in props.items():
            db.set_edge_prop(eid, key, val)

        updated = db.get_edge(eid)
        return Response.json_response(edge_to_json(updated))

    def delete_edge(req):
        """DELETE /api/edges/:id -- Delete an edge."""
        eid, err = parse_int_param(req, 'id')
        if err:
            return err

        if not db.delete_edge(eid):
            return Response.json_response({'error': 'Edge not found'}, 404)

        return Response.json_response({'deleted': True, 'id': eid})

    # ================================================================
    # QUERY ENDPOINT
    # ================================================================

    def execute_query(req):
        """POST /api/query -- Execute a GQL query."""
        data, err = require_json(req)
        if err:
            return err

        query_str = data.get('query')
        if not query_str or not isinstance(query_str, str):
            return Response.json_response(
                {'error': 'query (string) is required'}, 400)

        try:
            executor = QueryExecutor(db)
            results = executor.execute(query_str)
            return Response.json_response({
                'results': results,
                'count': len(results),
            })
        except QueryError as e:
            return Response.json_response(
                {'error': f'Query error: {str(e)}'}, 400)

    # ================================================================
    # PATH ENDPOINTS
    # ================================================================

    def shortest_path(req):
        """GET /api/paths/shortest -- Find shortest path between nodes."""
        src, err = parse_int_query(req, 'src')
        if err:
            return err
        dst, err = parse_int_query(req, 'dst')
        if err:
            return err

        if src is None or dst is None:
            return Response.json_response(
                {'error': 'src and dst query parameters are required'}, 400)

        edge_type = req.query.get('edge_type')
        max_depth, err = parse_int_query(req, 'max_depth', default=20)
        if err:
            return err

        path = db.shortest_path(src, dst, edge_type=edge_type, max_depth=max_depth)
        if path is None:
            return Response.json_response({
                'path': None,
                'found': False,
            })

        # Resolve node details for the path
        path_nodes = []
        for nid in path:
            node = db.get_node(nid)
            if node:
                path_nodes.append(node_to_json(node))

        return Response.json_response({
            'path': path,
            'nodes': path_nodes,
            'length': len(path) - 1,
            'found': True,
        })

    def all_paths(req):
        """GET /api/paths/all -- Find all paths between nodes."""
        src, err = parse_int_query(req, 'src')
        if err:
            return err
        dst, err = parse_int_query(req, 'dst')
        if err:
            return err

        if src is None or dst is None:
            return Response.json_response(
                {'error': 'src and dst query parameters are required'}, 400)

        edge_type = req.query.get('edge_type')
        max_depth, err = parse_int_query(req, 'max_depth', default=10)
        if err:
            return err

        paths = db.all_paths(src, dst, edge_type=edge_type, max_depth=max_depth)
        return Response.json_response({
            'paths': paths,
            'count': len(paths),
        })

    # ================================================================
    # INDEX ENDPOINTS
    # ================================================================

    def create_index(req):
        """POST /api/indexes -- Create a property index."""
        data, err = require_json(req)
        if err:
            return err

        label = data.get('label')
        prop = data.get('property')

        if not label or not prop:
            return Response.json_response(
                {'error': 'label and property are required'}, 400)

        db.create_index(label, prop)
        return Response.json_response({
            'created': True,
            'label': label,
            'property': prop,
        }, 201)

    def list_indexes(req):
        """GET /api/indexes -- List all indexes."""
        indexes = []
        for (label, prop) in db._index._data.keys():
            indexes.append({'label': label, 'property': prop})
        return Response.json_response({'indexes': indexes})

    # ================================================================
    # SNAPSHOT ENDPOINTS
    # ================================================================

    def save_snapshot(req):
        """POST /api/snapshots -- Save a snapshot."""
        idx = db.save_snapshot()
        return Response.json_response({
            'snapshot_index': idx,
            'total_snapshots': len(db._snapshots),
        }, 201)

    def restore_snapshot(req):
        """POST /api/snapshots/restore -- Restore a snapshot."""
        data, err = require_json(req)
        if err:
            return err

        index = data.get('index', -1)
        try:
            index = int(index)
        except (ValueError, TypeError):
            return Response.json_response(
                {'error': 'index must be an integer'}, 400)

        if db.restore_snapshot(index):
            return Response.json_response({
                'restored': True,
                'snapshot_index': index,
            })
        else:
            return Response.json_response(
                {'error': 'Snapshot not found'}, 404)

    def list_snapshots(req):
        """GET /api/snapshots -- List snapshot count."""
        return Response.json_response({
            'count': len(db._snapshots),
        })

    # ================================================================
    # STATS ENDPOINT
    # ================================================================

    def get_stats(req):
        """GET /api/stats -- Database statistics."""
        stats = db.stats()
        stats['indexes'] = len(db._index._data)
        return Response.json_response(stats)

    # ================================================================
    # BATCH ENDPOINT
    # ================================================================

    def batch_operations(req):
        """POST /api/batch -- Execute multiple operations atomically.

        Body: {"operations": [
            {"method": "POST", "path": "/api/nodes", "body": {...}},
            {"method": "PUT", "path": "/api/edges/1", "body": {...}},
            ...
        ]}

        If any operation fails with 4xx/5xx, the entire batch is rolled back
        using snapshots.
        """
        data, err = require_json(req)
        if err:
            return err

        operations = data.get('operations')
        if not isinstance(operations, list):
            return Response.json_response(
                {'error': 'operations (array) is required'}, 400)

        if len(operations) > 100:
            return Response.json_response(
                {'error': 'Maximum 100 operations per batch'}, 400)

        # Save snapshot for rollback
        snap_idx = db.save_snapshot()
        results = []

        for i, op in enumerate(operations):
            method = op.get('method', 'GET').upper()
            path = op.get('path', '')
            body = op.get('body')

            # Build a synthetic request
            body_bytes = json.dumps(body).encode('utf-8') if body else b''
            headers = [('Content-Type', 'application/json')] if body else []
            synth_req = Request(method, path, 'HTTP/1.1', headers, body_bytes)

            # Route it
            resp = router.handle(synth_req)
            status = resp.status

            # Parse response body
            try:
                resp_data = json.loads(resp.body) if resp.body else None
            except (json.JSONDecodeError, ValueError):
                resp_data = None

            if status >= 400:
                # Rollback
                db.restore_snapshot(snap_idx)
                # Remove the snapshot we created
                if len(db._snapshots) > snap_idx:
                    db._snapshots = db._snapshots[:snap_idx]
                return Response.json_response({
                    'error': 'Batch operation failed',
                    'failed_index': i,
                    'failed_operation': op,
                    'detail': resp_data,
                    'results': results,
                }, 400)

            results.append({
                'index': i,
                'status': status,
                'data': resp_data,
            })

        # Clean up the rollback snapshot (batch succeeded)
        if len(db._snapshots) > snap_idx:
            db._snapshots = db._snapshots[:snap_idx]

        return Response.json_response({
            'results': results,
            'count': len(results),
        })

    # ================================================================
    # REGISTER ROUTES
    # ================================================================

    # Nodes
    router.get('/api/nodes', list_nodes)
    router.post('/api/nodes', create_node)
    router.get('/api/nodes/:id', get_node)
    router.put('/api/nodes/:id', update_node)
    router.delete('/api/nodes/:id', delete_node)
    router.post('/api/nodes/:id/labels', add_node_label)
    router.delete('/api/nodes/:id/labels/:label', remove_node_label)
    router.get('/api/nodes/:id/neighbors', get_neighbors)
    router.get('/api/nodes/:id/edges', get_node_edges)

    # Edges
    router.get('/api/edges', list_edges)
    router.post('/api/edges', create_edge)
    router.get('/api/edges/:id', get_edge)
    router.put('/api/edges/:id', update_edge)
    router.delete('/api/edges/:id', delete_edge)

    # Query
    router.post('/api/query', execute_query)

    # Paths
    router.get('/api/paths/shortest', shortest_path)
    router.get('/api/paths/all', all_paths)

    # Indexes
    router.post('/api/indexes', create_index)
    router.get('/api/indexes', list_indexes)

    # Snapshots
    router.post('/api/snapshots', save_snapshot)
    router.post('/api/snapshots/restore', restore_snapshot)
    router.get('/api/snapshots', list_snapshots)

    # Stats
    router.get('/api/stats', get_stats)

    # Batch
    router.post('/api/batch', batch_operations)

    return router, db


# ============================================================
# Value Coercion (query string values -> typed values)
# ============================================================

def _coerce_value(v):
    """Try to coerce a string value to int, float, or bool."""
    if isinstance(v, list):
        return [_coerce_value(x) for x in v]
    if not isinstance(v, str):
        return v
    # Booleans
    if v.lower() == 'true':
        return True
    if v.lower() == 'false':
        return False
    # Integers
    try:
        return int(v)
    except ValueError:
        pass
    # Floats
    try:
        return float(v)
    except ValueError:
        pass
    return v


# ============================================================
# Convenience: start a live server
# ============================================================

def serve(host='127.0.0.1', port=8080):
    """Start a live Knowledge Graph API server."""
    router, db = create_api()
    server = HTTPServer(router, host=host, port=port)
    server.start()
    print(f"Knowledge Graph API running at http://{host}:{server.port}")
    print("Press Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("\nStopped.")


if __name__ == '__main__':
    serve()
