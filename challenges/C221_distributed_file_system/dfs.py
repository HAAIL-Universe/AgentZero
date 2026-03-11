"""
C221: Distributed File System
Composes: C205 (Consistent Hashing), C204 (Vector Clocks)

A GFS/HDFS-inspired distributed file system with:
- MetadataServer: namespace tree, chunk locations, leases
- ChunkServer: stores fixed-size data chunks, heartbeats
- ChunkReplicator: places replicas via consistent hashing
- LeaseManager: write coordination via time-limited leases
- DFSClient: user-facing file operations
- DFSCluster: orchestrates the full system
"""

import sys, os, time, hashlib, math, threading
from collections import defaultdict
from enum import Enum, auto

# Compose C205 and C204
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C205_consistent_hashing'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C204_vector_clocks'))

from consistent_hashing import HashRing, ReplicatedHashRing
from vector_clocks import VectorClock


# ============================================================
# Constants
# ============================================================

DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB
DEFAULT_REPLICATION_FACTOR = 3
LEASE_DURATION = 60.0  # seconds
HEARTBEAT_INTERVAL = 3.0
HEARTBEAT_TIMEOUT = 10.0


# ============================================================
# Data Types
# ============================================================

class FileType(Enum):
    FILE = auto()
    DIRECTORY = auto()


class ChunkStatus(Enum):
    COMPLETE = auto()
    PARTIAL = auto()
    CORRUPTED = auto()


class ServerStatus(Enum):
    ALIVE = auto()
    DEAD = auto()
    DECOMMISSIONING = auto()


class OperationType(Enum):
    CREATE = auto()
    DELETE = auto()
    WRITE = auto()
    READ = auto()
    RENAME = auto()
    MKDIR = auto()
    LIST = auto()


class DFSError(Exception):
    pass


class FileNotFoundError_(DFSError):
    pass


class FileExistsError_(DFSError):
    pass


class NotADirectoryError_(DFSError):
    pass


class IsADirectoryError_(DFSError):
    pass


class PermissionError_(DFSError):
    pass


class NoAvailableServersError(DFSError):
    pass


class LeaseError(DFSError):
    pass


class ChunkError(DFSError):
    pass


class CorruptionError(ChunkError):
    pass


# ============================================================
# INode -- metadata for files and directories
# ============================================================

class INode:
    """Represents a file or directory in the namespace."""

    _next_id = 1

    def __init__(self, name, file_type, parent=None):
        self.inode_id = INode._next_id
        INode._next_id += 1
        self.name = name
        self.file_type = file_type
        self.parent = parent
        self.children = {}  # name -> INode (directories only)
        self.chunks = []  # list of chunk_ids (files only)
        self.size = 0
        self.created_at = time.time()
        self.modified_at = self.created_at
        self.version = VectorClock()
        self.permissions = 0o755 if file_type == FileType.DIRECTORY else 0o644
        self.owner = "root"
        self.metadata = {}  # user-defined key-value

    @property
    def path(self):
        parts = []
        node = self
        while node is not None:
            parts.append(node.name)
            node = node.parent
        parts.reverse()
        return "/" + "/".join(p for p in parts if p)

    def is_file(self):
        return self.file_type == FileType.FILE

    def is_directory(self):
        return self.file_type == FileType.DIRECTORY

    def touch(self, server_id="meta"):
        self.modified_at = time.time()
        self.version.increment(server_id)


# ============================================================
# Chunk -- a piece of file data
# ============================================================

class Chunk:
    """A fixed-size piece of file data."""

    def __init__(self, chunk_id, data=b"", checksum=None):
        self.chunk_id = chunk_id
        self.data = data
        self.size = len(data)
        self.checksum = checksum or self._compute_checksum()
        self.status = ChunkStatus.COMPLETE
        self.version = 0
        self.created_at = time.time()

    def _compute_checksum(self):
        return hashlib.md5(self.data).hexdigest()

    def verify(self):
        """Verify chunk integrity."""
        return self._compute_checksum() == self.checksum

    def update(self, data):
        self.data = data
        self.size = len(data)
        self.checksum = self._compute_checksum()
        self.version += 1


# ============================================================
# ChunkServer -- stores chunks
# ============================================================

class ChunkServer:
    """Stores data chunks and reports health."""

    def __init__(self, server_id, capacity=1024 * 1024 * 1024):
        self.server_id = server_id
        self.capacity = capacity
        self.used_space = 0
        self.chunks = {}  # chunk_id -> Chunk
        self.status = ServerStatus.ALIVE
        self.last_heartbeat = time.time()
        self.stats = {
            "reads": 0,
            "writes": 0,
            "deletes": 0,
            "bytes_read": 0,
            "bytes_written": 0,
        }

    @property
    def available_space(self):
        return self.capacity - self.used_space

    @property
    def utilization(self):
        return self.used_space / self.capacity if self.capacity > 0 else 0.0

    def store_chunk(self, chunk_id, data):
        """Store a chunk. Returns the Chunk object."""
        size = len(data)
        if size > self.available_space:
            raise DFSError(f"Server {self.server_id}: insufficient space")
        chunk = Chunk(chunk_id, data)
        if chunk_id in self.chunks:
            # Update existing -- reclaim old space
            self.used_space -= self.chunks[chunk_id].size
        self.chunks[chunk_id] = chunk
        self.used_space += size
        self.stats["writes"] += 1
        self.stats["bytes_written"] += size
        return chunk

    def read_chunk(self, chunk_id):
        """Read a chunk's data. Verifies integrity."""
        if chunk_id not in self.chunks:
            raise ChunkError(f"Chunk {chunk_id} not found on server {self.server_id}")
        chunk = self.chunks[chunk_id]
        if not chunk.verify():
            chunk.status = ChunkStatus.CORRUPTED
            raise CorruptionError(f"Chunk {chunk_id} corrupted on server {self.server_id}")
        self.stats["reads"] += 1
        self.stats["bytes_read"] += chunk.size
        return chunk.data

    def delete_chunk(self, chunk_id):
        """Delete a chunk."""
        if chunk_id not in self.chunks:
            return False
        chunk = self.chunks.pop(chunk_id)
        self.used_space -= chunk.size
        self.stats["deletes"] += 1
        return True

    def has_chunk(self, chunk_id):
        return chunk_id in self.chunks

    def get_chunk_ids(self):
        return list(self.chunks.keys())

    def heartbeat(self):
        """Send heartbeat -- returns chunk report."""
        self.last_heartbeat = time.time()
        return {
            "server_id": self.server_id,
            "status": self.status,
            "used_space": self.used_space,
            "capacity": self.capacity,
            "chunk_count": len(self.chunks),
            "chunk_ids": list(self.chunks.keys()),
        }

    def is_alive(self):
        return self.status == ServerStatus.ALIVE


# ============================================================
# LeaseManager -- write coordination
# ============================================================

class Lease:
    def __init__(self, chunk_id, holder, duration=LEASE_DURATION):
        self.chunk_id = chunk_id
        self.holder = holder
        self.granted_at = time.time()
        self.duration = duration
        self.renewed_at = self.granted_at

    @property
    def expires_at(self):
        return self.renewed_at + self.duration

    def is_expired(self, now=None):
        now = now or time.time()
        return now >= self.expires_at

    def renew(self):
        self.renewed_at = time.time()


class LeaseManager:
    """Manages write leases for chunk coordination."""

    def __init__(self, duration=LEASE_DURATION):
        self.leases = {}  # chunk_id -> Lease
        self.duration = duration
        self._now_override = None  # for testing

    def _now(self):
        return self._now_override if self._now_override is not None else time.time()

    def grant(self, chunk_id, holder):
        """Grant a lease. Fails if already held by another."""
        if chunk_id in self.leases:
            lease = self.leases[chunk_id]
            if not lease.is_expired(self._now()) and lease.holder != holder:
                raise LeaseError(
                    f"Chunk {chunk_id} lease held by {lease.holder}"
                )
        lease = Lease(chunk_id, holder, self.duration)
        lease.granted_at = self._now()
        lease.renewed_at = self._now()
        self.leases[chunk_id] = lease
        return lease

    def revoke(self, chunk_id):
        """Revoke a lease."""
        return self.leases.pop(chunk_id, None) is not None

    def renew(self, chunk_id, holder):
        """Renew an existing lease."""
        if chunk_id not in self.leases:
            raise LeaseError(f"No lease for chunk {chunk_id}")
        lease = self.leases[chunk_id]
        if lease.holder != holder:
            raise LeaseError(f"Lease held by {lease.holder}, not {holder}")
        if lease.is_expired(self._now()):
            raise LeaseError(f"Lease for {chunk_id} has expired")
        lease.renewed_at = self._now()
        return lease

    def check(self, chunk_id, holder):
        """Check if holder has a valid lease."""
        if chunk_id not in self.leases:
            return False
        lease = self.leases[chunk_id]
        return lease.holder == holder and not lease.is_expired(self._now())

    def cleanup_expired(self):
        """Remove expired leases."""
        now = self._now()
        expired = [cid for cid, l in self.leases.items() if l.is_expired(now)]
        for cid in expired:
            del self.leases[cid]
        return expired

    def get_holder(self, chunk_id):
        """Get current lease holder, or None."""
        if chunk_id in self.leases:
            lease = self.leases[chunk_id]
            if not lease.is_expired(self._now()):
                return lease.holder
        return None

    def list_leases(self):
        """List all active leases."""
        now = self._now()
        return {
            cid: {"holder": l.holder, "expires_at": l.expires_at}
            for cid, l in self.leases.items()
            if not l.is_expired(now)
        }


# ============================================================
# ChunkReplicator -- places replicas via consistent hashing
# ============================================================

class ChunkReplicator:
    """Manages chunk placement and replication using consistent hashing."""

    def __init__(self, replication_factor=DEFAULT_REPLICATION_FACTOR):
        self.replication_factor = replication_factor
        self.hash_ring = ReplicatedHashRing(
            replication_factor=replication_factor
        )
        self.chunk_locations = defaultdict(set)  # chunk_id -> set of server_ids

    def add_server(self, server_id):
        self.hash_ring.add_node(server_id)

    def remove_server(self, server_id):
        self.hash_ring.remove_node(server_id)

    def get_target_servers(self, chunk_id):
        """Get the servers where a chunk should be placed."""
        if not self.hash_ring.nodes:
            return []
        pref_list = self.hash_ring.get_preference_list(chunk_id)
        return pref_list[:self.replication_factor]

    def record_location(self, chunk_id, server_id):
        """Record that a chunk exists on a server."""
        self.chunk_locations[chunk_id].add(server_id)

    def remove_location(self, chunk_id, server_id):
        """Remove a chunk location record."""
        if chunk_id in self.chunk_locations:
            self.chunk_locations[chunk_id].discard(server_id)
            if not self.chunk_locations[chunk_id]:
                del self.chunk_locations[chunk_id]

    def get_locations(self, chunk_id):
        """Get all servers that have this chunk."""
        return list(self.chunk_locations.get(chunk_id, set()))

    def get_under_replicated(self):
        """Find chunks with fewer replicas than desired."""
        under = {}
        for chunk_id, servers in self.chunk_locations.items():
            if len(servers) < self.replication_factor:
                under[chunk_id] = {
                    "current": len(servers),
                    "target": self.replication_factor,
                    "servers": list(servers),
                }
        return under

    def get_over_replicated(self):
        """Find chunks with more replicas than desired."""
        over = {}
        for chunk_id, servers in self.chunk_locations.items():
            if len(servers) > self.replication_factor:
                over[chunk_id] = {
                    "current": len(servers),
                    "target": self.replication_factor,
                    "servers": list(servers),
                }
        return over

    def plan_rebalance(self, available_servers):
        """Plan chunk moves to achieve balanced replication."""
        moves = []  # (chunk_id, from_server, to_server)
        for chunk_id, servers in list(self.chunk_locations.items()):
            target_servers = set(self.get_target_servers(chunk_id))
            target_servers = target_servers & set(available_servers)
            current = servers & set(available_servers)

            # Need to add replicas
            for target in target_servers - current:
                if current:
                    source = next(iter(current))
                    moves.append((chunk_id, source, target))

            # Could remove excess replicas
            excess = current - target_servers
            if len(current) > self.replication_factor:
                for srv in list(excess)[:len(current) - self.replication_factor]:
                    moves.append((chunk_id, srv, None))  # None = delete

        return moves

    def remove_chunk(self, chunk_id):
        """Remove all location records for a chunk."""
        self.chunk_locations.pop(chunk_id, None)


# ============================================================
# MetadataServer -- namespace + chunk location tracking
# ============================================================

class MetadataServer:
    """Manages the file system namespace and chunk locations."""

    def __init__(self, server_id="meta-1", chunk_size=DEFAULT_CHUNK_SIZE):
        self.server_id = server_id
        self.chunk_size = chunk_size
        self.root = INode("", FileType.DIRECTORY)
        self.inodes = {self.root.inode_id: self.root}  # inode_id -> INode
        self.chunk_to_file = {}  # chunk_id -> inode_id
        self._next_chunk_id = 1
        self.op_log = []  # operation log for auditing
        self.version = VectorClock()

    def _resolve_path(self, path):
        """Resolve a path to its INode. Returns None if not found."""
        if path == "/":
            return self.root
        parts = [p for p in path.strip("/").split("/") if p]
        node = self.root
        for part in parts:
            if not node.is_directory():
                return None
            if part not in node.children:
                return None
            node = node.children[part]
        return node

    def _resolve_parent(self, path):
        """Resolve the parent directory and return (parent, name)."""
        parts = [p for p in path.strip("/").split("/") if p]
        if not parts:
            raise DFSError("Cannot operate on root")
        name = parts[-1]
        parent_path = "/" + "/".join(parts[:-1])
        parent = self._resolve_path(parent_path)
        if parent is None:
            raise FileNotFoundError_(f"Parent directory not found: {parent_path}")
        if not parent.is_directory():
            raise NotADirectoryError_(f"Not a directory: {parent_path}")
        return parent, name

    def _log_op(self, op_type, path, **kwargs):
        self.op_log.append({
            "type": op_type,
            "path": path,
            "time": time.time(),
            **kwargs,
        })
        self.version.increment(self.server_id)

    def _alloc_chunk_id(self):
        cid = f"chunk-{self._next_chunk_id:08d}"
        self._next_chunk_id += 1
        return cid

    # -- Namespace operations --

    def create_file(self, path, owner="root"):
        """Create a new empty file."""
        parent, name = self._resolve_parent(path)
        if name in parent.children:
            raise FileExistsError_(f"File already exists: {path}")
        inode = INode(name, FileType.FILE, parent)
        inode.owner = owner
        parent.children[name] = inode
        self.inodes[inode.inode_id] = inode
        parent.touch(self.server_id)
        self._log_op(OperationType.CREATE, path, inode_id=inode.inode_id)
        return inode

    def mkdir(self, path, parents=False, owner="root"):
        """Create a directory. If parents=True, create intermediate dirs."""
        if parents:
            parts = [p for p in path.strip("/").split("/") if p]
            current = self.root
            for part in parts:
                if part not in current.children:
                    child = INode(part, FileType.DIRECTORY, current)
                    child.owner = owner
                    current.children[part] = child
                    self.inodes[child.inode_id] = child
                    current.touch(self.server_id)
                else:
                    child = current.children[part]
                    if not child.is_directory():
                        raise NotADirectoryError_(f"Not a directory: {child.path}")
                current = child
            self._log_op(OperationType.MKDIR, path)
            return current

        parent, name = self._resolve_parent(path)
        if name in parent.children:
            raise FileExistsError_(f"Already exists: {path}")
        inode = INode(name, FileType.DIRECTORY, parent)
        inode.owner = owner
        parent.children[name] = inode
        self.inodes[inode.inode_id] = inode
        parent.touch(self.server_id)
        self._log_op(OperationType.MKDIR, path, inode_id=inode.inode_id)
        return inode

    def delete(self, path, recursive=False):
        """Delete a file or directory."""
        if path == "/":
            raise DFSError("Cannot delete root")
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        if node.is_directory():
            if node.children and not recursive:
                raise DFSError(f"Directory not empty: {path}")
            # Collect all chunks to delete
            chunks_to_delete = []
            self._collect_chunks(node, chunks_to_delete)
            for cid in chunks_to_delete:
                self.chunk_to_file.pop(cid, None)
            # Remove from parent
            self._remove_subtree(node)
        else:
            # File -- remove chunk mappings
            for cid in node.chunks:
                self.chunk_to_file.pop(cid, None)
            parent = node.parent
            del parent.children[node.name]
            del self.inodes[node.inode_id]
            parent.touch(self.server_id)
        self._log_op(OperationType.DELETE, path)
        return True

    def _collect_chunks(self, node, chunks):
        if node.is_file():
            chunks.extend(node.chunks)
        elif node.is_directory():
            for child in node.children.values():
                self._collect_chunks(child, chunks)

    def _remove_subtree(self, node):
        if node.is_directory():
            for child in list(node.children.values()):
                self._remove_subtree(child)
        if node.parent:
            node.parent.children.pop(node.name, None)
            node.parent.touch(self.server_id)
        self.inodes.pop(node.inode_id, None)

    def rename(self, old_path, new_path):
        """Rename/move a file or directory."""
        node = self._resolve_path(old_path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {old_path}")
        new_parent, new_name = self._resolve_parent(new_path)
        if new_name in new_parent.children:
            raise FileExistsError_(f"Already exists: {new_path}")
        # Remove from old parent
        old_parent = node.parent
        del old_parent.children[node.name]
        old_parent.touch(self.server_id)
        # Add to new parent
        node.name = new_name
        node.parent = new_parent
        new_parent.children[new_name] = node
        new_parent.touch(self.server_id)
        node.touch(self.server_id)
        self._log_op(OperationType.RENAME, old_path, new_path=new_path)
        return node

    def list_dir(self, path):
        """List directory contents."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        if not node.is_directory():
            raise NotADirectoryError_(f"Not a directory: {path}")
        result = []
        for name, child in sorted(node.children.items()):
            result.append({
                "name": name,
                "type": child.file_type,
                "size": child.size,
                "inode_id": child.inode_id,
                "modified_at": child.modified_at,
            })
        return result

    def stat(self, path):
        """Get file/directory metadata."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        return {
            "path": path,
            "type": node.file_type,
            "size": node.size,
            "inode_id": node.inode_id,
            "chunks": list(node.chunks),
            "created_at": node.created_at,
            "modified_at": node.modified_at,
            "owner": node.owner,
            "permissions": node.permissions,
            "version": dict(node.version.clock),
        }

    def exists(self, path):
        return self._resolve_path(path) is not None

    # -- Chunk management --

    def allocate_chunks(self, path, size):
        """Allocate chunk IDs for a file write of given size."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        if not node.is_file():
            raise IsADirectoryError_(f"Is a directory: {path}")
        num_chunks = max(1, math.ceil(size / self.chunk_size)) if size > 0 else 0
        chunk_ids = []
        for _ in range(num_chunks):
            cid = self._alloc_chunk_id()
            chunk_ids.append(cid)
            self.chunk_to_file[cid] = node.inode_id
        return chunk_ids

    def commit_chunks(self, path, chunk_ids, total_size):
        """Commit allocated chunks to a file after data is written."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        node.chunks = chunk_ids
        node.size = total_size
        node.touch(self.server_id)
        self._log_op(OperationType.WRITE, path, chunks=chunk_ids, size=total_size)

    def get_chunks(self, path):
        """Get chunk IDs for a file."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        if not node.is_file():
            raise IsADirectoryError_(f"Is a directory: {path}")
        return list(node.chunks)

    def get_file_size(self, path):
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        return node.size

    def set_metadata(self, path, key, value):
        """Set user-defined metadata on a file/dir."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        node.metadata[key] = value
        node.touch(self.server_id)

    def get_metadata(self, path, key=None):
        """Get user-defined metadata."""
        node = self._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        if key is not None:
            return node.metadata.get(key)
        return dict(node.metadata)

    def get_namespace_stats(self):
        """Get overall namespace statistics."""
        file_count = 0
        dir_count = 0
        total_size = 0
        total_chunks = 0
        for inode in self.inodes.values():
            if inode.is_file():
                file_count += 1
                total_size += inode.size
                total_chunks += len(inode.chunks)
            else:
                dir_count += 1
        return {
            "files": file_count,
            "directories": dir_count,
            "total_size": total_size,
            "total_chunks": total_chunks,
            "operations": len(self.op_log),
        }


# ============================================================
# DFSCluster -- orchestrates the full system
# ============================================================

class DFSCluster:
    """
    Orchestrates MetadataServer, ChunkServers, Replicator, and LeaseManager
    into a complete distributed file system.
    """

    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE,
                 replication_factor=DEFAULT_REPLICATION_FACTOR):
        self.chunk_size = chunk_size
        self.replication_factor = replication_factor
        self.metadata = MetadataServer(chunk_size=chunk_size)
        self.lease_mgr = LeaseManager()
        self.replicator = ChunkReplicator(replication_factor=replication_factor)
        self.chunk_servers = {}  # server_id -> ChunkServer
        self.dead_servers = set()

    # -- Server management --

    def add_server(self, server_id, capacity=1024 * 1024 * 1024):
        """Add a chunk server to the cluster."""
        server = ChunkServer(server_id, capacity)
        self.chunk_servers[server_id] = server
        self.replicator.add_server(server_id)
        self.dead_servers.discard(server_id)
        return server

    def remove_server(self, server_id):
        """Remove a chunk server (decommission)."""
        if server_id not in self.chunk_servers:
            raise DFSError(f"Server not found: {server_id}")
        server = self.chunk_servers[server_id]
        server.status = ServerStatus.DECOMMISSIONING
        # Replicate chunks elsewhere before removing
        for chunk_id in list(server.get_chunk_ids()):
            self.replicator.remove_location(chunk_id, server_id)
        self.replicator.remove_server(server_id)
        del self.chunk_servers[server_id]
        self.dead_servers.discard(server_id)
        return True

    def get_alive_servers(self):
        """Get list of alive server IDs."""
        return [
            sid for sid, srv in self.chunk_servers.items()
            if srv.status == ServerStatus.ALIVE and sid not in self.dead_servers
        ]

    def mark_server_dead(self, server_id):
        """Mark a server as dead (failed heartbeat)."""
        if server_id in self.chunk_servers:
            self.chunk_servers[server_id].status = ServerStatus.DEAD
            self.dead_servers.add(server_id)

    def mark_server_alive(self, server_id):
        """Revive a dead server."""
        if server_id in self.chunk_servers:
            self.chunk_servers[server_id].status = ServerStatus.ALIVE
            self.dead_servers.discard(server_id)

    # -- File operations --

    def create_file(self, path, owner="root"):
        """Create an empty file."""
        return self.metadata.create_file(path, owner)

    def mkdir(self, path, parents=False, owner="root"):
        """Create a directory."""
        return self.metadata.mkdir(path, parents, owner)

    def delete(self, path, recursive=False):
        """Delete a file or directory, cleaning up chunks."""
        node = self.metadata._resolve_path(path)
        if node is None:
            raise FileNotFoundError_(f"Not found: {path}")
        # Collect chunks to delete from chunk servers
        chunks_to_delete = []
        self.metadata._collect_chunks(node, chunks_to_delete)
        # Delete from chunk servers
        for chunk_id in chunks_to_delete:
            for server_id in self.replicator.get_locations(chunk_id):
                if server_id in self.chunk_servers:
                    self.chunk_servers[server_id].delete_chunk(chunk_id)
            self.replicator.remove_chunk(chunk_id)
            self.lease_mgr.revoke(chunk_id)
        # Delete from metadata
        return self.metadata.delete(path, recursive)

    def rename(self, old_path, new_path):
        return self.metadata.rename(old_path, new_path)

    def list_dir(self, path):
        return self.metadata.list_dir(path)

    def stat(self, path):
        return self.metadata.stat(path)

    def exists(self, path):
        return self.metadata.exists(path)

    # -- Read/Write operations --

    def write_file(self, path, data, client_id="client-1"):
        """Write data to a file. Creates if needed, overwrites existing content."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Ensure file exists
        if not self.metadata.exists(path):
            # Ensure parent dirs exist
            parts = path.strip("/").split("/")
            if len(parts) > 1:
                parent_path = "/" + "/".join(parts[:-1])
                if not self.metadata.exists(parent_path):
                    self.metadata.mkdir(parent_path, parents=True)
            self.metadata.create_file(path)

        # Clean up old chunks
        old_chunks = self.metadata.get_chunks(path)
        for chunk_id in old_chunks:
            for server_id in self.replicator.get_locations(chunk_id):
                if server_id in self.chunk_servers:
                    self.chunk_servers[server_id].delete_chunk(chunk_id)
            self.replicator.remove_chunk(chunk_id)

        alive = self.get_alive_servers()
        if not alive:
            raise NoAvailableServersError("No chunk servers available")

        # Allocate chunks
        chunk_ids = self.metadata.allocate_chunks(path, len(data))

        # Split data and write to chunk servers
        for i, chunk_id in enumerate(chunk_ids):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, len(data))
            chunk_data = data[start:end]

            # Get target servers from consistent hashing
            targets = self.replicator.get_target_servers(chunk_id)
            # Filter to alive servers
            targets = [t for t in targets if t in self.chunk_servers
                       and t not in self.dead_servers]

            if not targets:
                # Fallback: use any alive server
                targets = alive[:self.replication_factor]

            # Acquire lease
            self.lease_mgr.grant(chunk_id, client_id)

            # Write to all target servers
            for server_id in targets:
                self.chunk_servers[server_id].store_chunk(chunk_id, chunk_data)
                self.replicator.record_location(chunk_id, server_id)

            # Release lease
            self.lease_mgr.revoke(chunk_id)

        # Commit to metadata
        self.metadata.commit_chunks(path, chunk_ids, len(data))
        return len(data)

    def read_file(self, path):
        """Read a file's complete contents."""
        chunk_ids = self.metadata.get_chunks(path)
        if not chunk_ids:
            return b""

        data_parts = []
        for chunk_id in chunk_ids:
            locations = self.replicator.get_locations(chunk_id)
            # Filter to alive servers
            alive_locations = [
                s for s in locations
                if s in self.chunk_servers and s not in self.dead_servers
            ]
            if not alive_locations:
                raise ChunkError(
                    f"No alive server has chunk {chunk_id}"
                )
            # Try each server until success
            last_error = None
            for server_id in alive_locations:
                try:
                    data = self.chunk_servers[server_id].read_chunk(chunk_id)
                    data_parts.append(data)
                    last_error = None
                    break
                except (ChunkError, CorruptionError) as e:
                    last_error = e
            if last_error:
                raise last_error

        return b"".join(data_parts)

    def append_file(self, path, data, client_id="client-1"):
        """Append data to an existing file."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        existing = self.read_file(path)
        return self.write_file(path, existing + data, client_id)

    # -- Chunk health --

    def check_replication(self):
        """Check replication status of all chunks."""
        under = self.replicator.get_under_replicated()
        over = self.replicator.get_over_replicated()
        return {
            "under_replicated": under,
            "over_replicated": over,
            "healthy": len(self.replicator.chunk_locations) - len(under) - len(over),
            "total_chunks": len(self.replicator.chunk_locations),
        }

    def repair_replication(self):
        """Repair under-replicated chunks by copying to new servers."""
        under = self.replicator.get_under_replicated()
        alive = self.get_alive_servers()
        repaired = 0

        for chunk_id, info in under.items():
            current_servers = [s for s in info["servers"] if s in alive]
            if not current_servers:
                continue  # No source available

            needed = self.replication_factor - len(current_servers)
            available = [s for s in alive if s not in current_servers]

            for target in available[:needed]:
                source = current_servers[0]
                try:
                    data = self.chunk_servers[source].read_chunk(chunk_id)
                    self.chunk_servers[target].store_chunk(chunk_id, data)
                    self.replicator.record_location(chunk_id, target)
                    repaired += 1
                except (ChunkError, CorruptionError):
                    continue

        return repaired

    def rebalance(self):
        """Rebalance chunks across servers."""
        alive = self.get_alive_servers()
        moves = self.replicator.plan_rebalance(alive)
        executed = 0

        for chunk_id, source, target in moves:
            if target is None:
                # Delete excess replica
                if source in self.chunk_servers:
                    self.chunk_servers[source].delete_chunk(chunk_id)
                    self.replicator.remove_location(chunk_id, source)
                    executed += 1
            else:
                # Copy to new server
                try:
                    data = self.chunk_servers[source].read_chunk(chunk_id)
                    self.chunk_servers[target].store_chunk(chunk_id, data)
                    self.replicator.record_location(chunk_id, target)
                    executed += 1
                except (ChunkError, CorruptionError):
                    continue

        return executed

    # -- Metadata operations --

    def set_metadata(self, path, key, value):
        return self.metadata.set_metadata(path, key, value)

    def get_metadata(self, path, key=None):
        return self.metadata.get_metadata(path, key)

    # -- Cluster stats --

    def get_cluster_stats(self):
        """Get comprehensive cluster statistics."""
        ns = self.metadata.get_namespace_stats()
        repl = self.check_replication()
        server_stats = {}
        total_capacity = 0
        total_used = 0
        for sid, srv in self.chunk_servers.items():
            server_stats[sid] = {
                "status": srv.status.name,
                "utilization": srv.utilization,
                "chunks": len(srv.chunks),
                "capacity": srv.capacity,
                "used": srv.used_space,
            }
            total_capacity += srv.capacity
            total_used += srv.used_space

        return {
            "namespace": ns,
            "replication": repl,
            "servers": server_stats,
            "total_capacity": total_capacity,
            "total_used": total_used,
            "cluster_utilization": total_used / total_capacity if total_capacity > 0 else 0,
            "alive_servers": len(self.get_alive_servers()),
            "dead_servers": len(self.dead_servers),
        }


# ============================================================
# DFSClient -- user-facing API
# ============================================================

class DFSClient:
    """User-facing client for the distributed file system."""

    def __init__(self, cluster, client_id="client-1"):
        self.cluster = cluster
        self.client_id = client_id

    def write(self, path, data):
        return self.cluster.write_file(path, data, self.client_id)

    def read(self, path):
        return self.cluster.read_file(path)

    def append(self, path, data):
        return self.cluster.append_file(path, data, self.client_id)

    def delete(self, path, recursive=False):
        return self.cluster.delete(path, recursive)

    def mkdir(self, path, parents=False):
        return self.cluster.mkdir(path, parents)

    def ls(self, path="/"):
        return self.cluster.list_dir(path)

    def stat(self, path):
        return self.cluster.stat(path)

    def exists(self, path):
        return self.cluster.exists(path)

    def rename(self, old_path, new_path):
        return self.cluster.rename(old_path, new_path)

    def create(self, path):
        return self.cluster.create_file(path)

    def set_metadata(self, path, key, value):
        return self.cluster.set_metadata(path, key, value)

    def get_metadata(self, path, key=None):
        return self.cluster.get_metadata(path, key)

    def copy(self, src_path, dst_path):
        """Copy a file."""
        data = self.read(src_path)
        return self.write(dst_path, data)

    def move(self, src_path, dst_path):
        """Move a file (rename)."""
        return self.rename(src_path, dst_path)

    def tree(self, path="/", prefix=""):
        """Return a tree representation of the directory structure."""
        lines = []
        entries = self.ls(path)
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "`-- " if is_last else "|-- "
            child_path = path.rstrip("/") + "/" + entry["name"]
            type_suffix = "/" if entry["type"] == FileType.DIRECTORY else f" ({entry['size']}B)"
            lines.append(f"{prefix}{connector}{entry['name']}{type_suffix}")
            if entry["type"] == FileType.DIRECTORY:
                child_prefix = prefix + ("    " if is_last else "|   ")
                lines.extend(self.tree(child_path, child_prefix))
        return lines


# ============================================================
# SnapshotManager -- point-in-time snapshots
# ============================================================

class Snapshot:
    """A point-in-time snapshot of file metadata and data."""

    def __init__(self, snapshot_id, name=""):
        self.snapshot_id = snapshot_id
        self.name = name
        self.created_at = time.time()
        self.file_chunks = {}  # path -> list of chunk_ids
        self.file_sizes = {}  # path -> size
        self.file_data = {}  # path -> bytes (actual data copy)


class SnapshotManager:
    """Manages point-in-time snapshots of the file system."""

    def __init__(self, cluster):
        self.cluster = cluster
        self.snapshots = {}  # snapshot_id -> Snapshot
        self._next_id = 1

    def create_snapshot(self, name=""):
        """Create a snapshot of the current file system state."""
        snap_id = f"snap-{self._next_id:04d}"
        self._next_id += 1
        snap = Snapshot(snap_id, name)

        # Walk the namespace and record all file chunks
        self._walk_and_record(self.cluster.metadata.root, "/", snap)
        self.snapshots[snap_id] = snap
        return snap

    def _walk_and_record(self, node, path, snap):
        if node.is_file():
            snap.file_chunks[path] = list(node.chunks)
            snap.file_sizes[path] = node.size
            # Store actual data copy for restore
            try:
                snap.file_data[path] = self.cluster.read_file(path)
            except (ChunkError, FileNotFoundError_):
                snap.file_data[path] = b""
        elif node.is_directory():
            for name, child in node.children.items():
                child_path = path.rstrip("/") + "/" + name
                self._walk_and_record(child, child_path, snap)

    def list_snapshots(self):
        return [
            {"id": s.snapshot_id, "name": s.name, "created_at": s.created_at,
             "files": len(s.file_chunks)}
            for s in self.snapshots.values()
        ]

    def get_snapshot(self, snapshot_id):
        if snapshot_id not in self.snapshots:
            raise DFSError(f"Snapshot not found: {snapshot_id}")
        return self.snapshots[snapshot_id]

    def restore_file(self, snapshot_id, path):
        """Restore a single file from a snapshot."""
        snap = self.get_snapshot(snapshot_id)
        if path not in snap.file_chunks:
            raise FileNotFoundError_(f"File not in snapshot: {path}")

        # Use stored data copy
        full_data = snap.file_data.get(path, b"")
        self.cluster.write_file(path, full_data)
        return len(full_data)

    def delete_snapshot(self, snapshot_id):
        if snapshot_id not in self.snapshots:
            raise DFSError(f"Snapshot not found: {snapshot_id}")
        del self.snapshots[snapshot_id]
        return True

    def diff_snapshots(self, snap_id_a, snap_id_b):
        """Compare two snapshots. Returns added, removed, modified files."""
        snap_a = self.get_snapshot(snap_id_a)
        snap_b = self.get_snapshot(snap_id_b)
        paths_a = set(snap_a.file_chunks.keys())
        paths_b = set(snap_b.file_chunks.keys())
        added = paths_b - paths_a
        removed = paths_a - paths_b
        modified = set()
        for path in paths_a & paths_b:
            if snap_a.file_chunks[path] != snap_b.file_chunks[path]:
                modified.add(path)
        return {
            "added": sorted(added),
            "removed": sorted(removed),
            "modified": sorted(modified),
        }


# ============================================================
# DFSAnalyzer -- cluster health analysis
# ============================================================

class DFSAnalyzer:
    """Analyzes cluster health, balance, and provides recommendations."""

    def __init__(self, cluster):
        self.cluster = cluster

    def analyze_balance(self):
        """Analyze data distribution balance across servers."""
        servers = self.cluster.chunk_servers
        if not servers:
            return {"balanced": True, "variance": 0, "recommendations": []}

        utilizations = [s.utilization for s in servers.values() if s.is_alive()]
        if not utilizations:
            return {"balanced": True, "variance": 0, "recommendations": []}

        avg = sum(utilizations) / len(utilizations)
        variance = sum((u - avg) ** 2 for u in utilizations) / len(utilizations)
        std_dev = variance ** 0.5

        recommendations = []
        for sid, srv in servers.items():
            if srv.is_alive() and srv.utilization > avg + 2 * std_dev:
                recommendations.append(f"Server {sid} is hot ({srv.utilization:.1%})")
            elif srv.is_alive() and srv.utilization < avg - 2 * std_dev and avg > 0.01:
                recommendations.append(f"Server {sid} is cold ({srv.utilization:.1%})")

        return {
            "balanced": std_dev < 0.1,
            "avg_utilization": avg,
            "std_dev": std_dev,
            "variance": variance,
            "server_count": len(utilizations),
            "recommendations": recommendations,
        }

    def analyze_replication_health(self):
        """Analyze chunk replication health."""
        repl = self.cluster.check_replication()
        health = "healthy"
        if repl["under_replicated"]:
            health = "degraded"
            # Critical only if some chunk has zero replicas
            for info in repl["under_replicated"].values():
                if info["current"] == 0:
                    health = "critical"
                    break

        return {
            "health": health,
            "total_chunks": repl["total_chunks"],
            "healthy_chunks": repl["healthy"],
            "under_replicated": len(repl["under_replicated"]),
            "over_replicated": len(repl["over_replicated"]),
            "replication_factor": self.cluster.replication_factor,
        }

    def analyze_capacity(self):
        """Analyze cluster capacity and project when it will fill."""
        stats = self.cluster.get_cluster_stats()
        total = stats["total_capacity"]
        used = stats["total_used"]
        free = total - used

        return {
            "total_capacity": total,
            "used": used,
            "free": free,
            "utilization": used / total if total > 0 else 0,
            "can_survive_server_loss": len(self.cluster.get_alive_servers()) > self.cluster.replication_factor,
        }

    def full_report(self):
        """Generate a comprehensive health report."""
        return {
            "balance": self.analyze_balance(),
            "replication": self.analyze_replication_health(),
            "capacity": self.analyze_capacity(),
            "cluster_stats": self.cluster.get_cluster_stats(),
        }
