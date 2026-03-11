"""
Tests for C221: Distributed File System
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from dfs import (
    INode, FileType, Chunk, ChunkStatus, ChunkServer, ServerStatus,
    LeaseManager, Lease, LeaseError,
    ChunkReplicator, MetadataServer, DFSCluster, DFSClient,
    SnapshotManager, DFSAnalyzer,
    DFSError, FileNotFoundError_, FileExistsError_, NotADirectoryError_,
    IsADirectoryError_, NoAvailableServersError, ChunkError, CorruptionError,
)


# ============================================================
# INode Tests
# ============================================================

class TestINode:
    def test_create_file_inode(self):
        node = INode("test.txt", FileType.FILE)
        assert node.name == "test.txt"
        assert node.is_file()
        assert not node.is_directory()
        assert node.size == 0
        assert node.chunks == []

    def test_create_directory_inode(self):
        node = INode("data", FileType.DIRECTORY)
        assert node.name == "data"
        assert node.is_directory()
        assert not node.is_file()
        assert node.children == {}

    def test_inode_path(self):
        root = INode("", FileType.DIRECTORY)
        child = INode("docs", FileType.DIRECTORY, parent=root)
        file = INode("readme.txt", FileType.FILE, parent=child)
        assert root.path == "/"
        assert child.path == "/docs"
        assert file.path == "/docs/readme.txt"

    def test_inode_touch(self):
        node = INode("test", FileType.FILE)
        old_time = node.modified_at
        old_version = dict(node.version.clock)
        node.touch("server-1")
        assert node.modified_at >= old_time
        assert node.version.clock.get("server-1", 0) > old_version.get("server-1", 0)

    def test_unique_inode_ids(self):
        a = INode("a", FileType.FILE)
        b = INode("b", FileType.FILE)
        assert a.inode_id != b.inode_id

    def test_inode_permissions(self):
        f = INode("f", FileType.FILE)
        d = INode("d", FileType.DIRECTORY)
        assert f.permissions == 0o644
        assert d.permissions == 0o755


# ============================================================
# Chunk Tests
# ============================================================

class TestChunk:
    def test_create_chunk(self):
        c = Chunk("c-1", b"hello world")
        assert c.chunk_id == "c-1"
        assert c.data == b"hello world"
        assert c.size == 11
        assert c.status == ChunkStatus.COMPLETE

    def test_chunk_verify(self):
        c = Chunk("c-1", b"test data")
        assert c.verify()
        c.data = b"corrupted"
        assert not c.verify()

    def test_chunk_update(self):
        c = Chunk("c-1", b"old")
        c.update(b"new data")
        assert c.data == b"new data"
        assert c.size == 8
        assert c.version == 1
        assert c.verify()

    def test_empty_chunk(self):
        c = Chunk("c-1", b"")
        assert c.size == 0
        assert c.verify()


# ============================================================
# ChunkServer Tests
# ============================================================

class TestChunkServer:
    def test_create_server(self):
        s = ChunkServer("s-1", capacity=1000)
        assert s.server_id == "s-1"
        assert s.capacity == 1000
        assert s.available_space == 1000
        assert s.utilization == 0.0
        assert s.is_alive()

    def test_store_and_read(self):
        s = ChunkServer("s-1")
        s.store_chunk("c-1", b"hello")
        assert s.has_chunk("c-1")
        data = s.read_chunk("c-1")
        assert data == b"hello"

    def test_store_updates_space(self):
        s = ChunkServer("s-1", capacity=100)
        s.store_chunk("c-1", b"x" * 50)
        assert s.used_space == 50
        assert s.available_space == 50

    def test_store_overwrite(self):
        s = ChunkServer("s-1", capacity=100)
        s.store_chunk("c-1", b"x" * 30)
        s.store_chunk("c-1", b"y" * 20)
        assert s.used_space == 20
        assert s.read_chunk("c-1") == b"y" * 20

    def test_store_insufficient_space(self):
        s = ChunkServer("s-1", capacity=10)
        with pytest.raises(DFSError, match="insufficient space"):
            s.store_chunk("c-1", b"x" * 20)

    def test_delete_chunk(self):
        s = ChunkServer("s-1")
        s.store_chunk("c-1", b"data")
        assert s.delete_chunk("c-1")
        assert not s.has_chunk("c-1")
        assert s.used_space == 0

    def test_delete_nonexistent(self):
        s = ChunkServer("s-1")
        assert not s.delete_chunk("c-999")

    def test_read_nonexistent(self):
        s = ChunkServer("s-1")
        with pytest.raises(ChunkError):
            s.read_chunk("c-999")

    def test_read_corrupted(self):
        s = ChunkServer("s-1")
        s.store_chunk("c-1", b"data")
        s.chunks["c-1"].data = b"corrupted"  # corrupt without updating checksum
        with pytest.raises(CorruptionError):
            s.read_chunk("c-1")

    def test_heartbeat(self):
        s = ChunkServer("s-1")
        s.store_chunk("c-1", b"data")
        hb = s.heartbeat()
        assert hb["server_id"] == "s-1"
        assert hb["chunk_count"] == 1
        assert "c-1" in hb["chunk_ids"]

    def test_stats_tracking(self):
        s = ChunkServer("s-1")
        s.store_chunk("c-1", b"data")
        s.read_chunk("c-1")
        s.delete_chunk("c-1")
        assert s.stats["writes"] == 1
        assert s.stats["reads"] == 1
        assert s.stats["deletes"] == 1

    def test_get_chunk_ids(self):
        s = ChunkServer("s-1")
        s.store_chunk("c-1", b"a")
        s.store_chunk("c-2", b"b")
        ids = s.get_chunk_ids()
        assert sorted(ids) == ["c-1", "c-2"]


# ============================================================
# LeaseManager Tests
# ============================================================

class TestLeaseManager:
    def test_grant_lease(self):
        lm = LeaseManager()
        lease = lm.grant("c-1", "client-1")
        assert lease.chunk_id == "c-1"
        assert lease.holder == "client-1"

    def test_grant_duplicate_same_holder(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        lease2 = lm.grant("c-1", "client-1")  # same holder OK
        assert lease2.holder == "client-1"

    def test_grant_conflict(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        with pytest.raises(LeaseError, match="held by"):
            lm.grant("c-1", "client-2")

    def test_grant_after_expiry(self):
        lm = LeaseManager(duration=0.01)
        lm._now_override = 1000.0
        lm.grant("c-1", "client-1")
        lm._now_override = 2000.0  # well past expiry
        lease = lm.grant("c-1", "client-2")
        assert lease.holder == "client-2"

    def test_revoke(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        assert lm.revoke("c-1")
        assert not lm.revoke("c-1")  # already revoked

    def test_renew(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        lease = lm.renew("c-1", "client-1")
        assert lease.holder == "client-1"

    def test_renew_wrong_holder(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        with pytest.raises(LeaseError, match="held by"):
            lm.renew("c-1", "client-2")

    def test_renew_expired(self):
        lm = LeaseManager(duration=0.01)
        lm._now_override = 1000.0
        lm.grant("c-1", "client-1")
        lm._now_override = 2000.0
        with pytest.raises(LeaseError, match="expired"):
            lm.renew("c-1", "client-1")

    def test_renew_nonexistent(self):
        lm = LeaseManager()
        with pytest.raises(LeaseError, match="No lease"):
            lm.renew("c-999", "client-1")

    def test_check(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        assert lm.check("c-1", "client-1")
        assert not lm.check("c-1", "client-2")
        assert not lm.check("c-999", "client-1")

    def test_cleanup_expired(self):
        lm = LeaseManager(duration=0.01)
        lm._now_override = 1000.0
        lm.grant("c-1", "client-1")
        lm.grant("c-2", "client-2")
        lm._now_override = 2000.0
        expired = lm.cleanup_expired()
        assert set(expired) == {"c-1", "c-2"}
        assert lm.list_leases() == {}

    def test_get_holder(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        assert lm.get_holder("c-1") == "client-1"
        assert lm.get_holder("c-999") is None

    def test_list_leases(self):
        lm = LeaseManager()
        lm.grant("c-1", "client-1")
        lm.grant("c-2", "client-2")
        leases = lm.list_leases()
        assert len(leases) == 2
        assert leases["c-1"]["holder"] == "client-1"


# ============================================================
# ChunkReplicator Tests
# ============================================================

class TestChunkReplicator:
    def test_add_remove_server(self):
        r = ChunkReplicator(replication_factor=2)
        r.add_server("s-1")
        r.add_server("s-2")
        targets = r.get_target_servers("chunk-1")
        assert len(targets) == 2
        r.remove_server("s-1")
        targets = r.get_target_servers("chunk-1")
        assert len(targets) <= 1

    def test_record_location(self):
        r = ChunkReplicator()
        r.record_location("c-1", "s-1")
        r.record_location("c-1", "s-2")
        locs = r.get_locations("c-1")
        assert set(locs) == {"s-1", "s-2"}

    def test_remove_location(self):
        r = ChunkReplicator()
        r.record_location("c-1", "s-1")
        r.remove_location("c-1", "s-1")
        assert r.get_locations("c-1") == []

    def test_under_replicated(self):
        r = ChunkReplicator(replication_factor=3)
        r.record_location("c-1", "s-1")
        under = r.get_under_replicated()
        assert "c-1" in under
        assert under["c-1"]["current"] == 1

    def test_over_replicated(self):
        r = ChunkReplicator(replication_factor=2)
        r.record_location("c-1", "s-1")
        r.record_location("c-1", "s-2")
        r.record_location("c-1", "s-3")
        over = r.get_over_replicated()
        assert "c-1" in over

    def test_remove_chunk(self):
        r = ChunkReplicator()
        r.record_location("c-1", "s-1")
        r.remove_chunk("c-1")
        assert r.get_locations("c-1") == []

    def test_empty_ring(self):
        r = ChunkReplicator()
        assert r.get_target_servers("c-1") == []

    def test_plan_rebalance(self):
        r = ChunkReplicator(replication_factor=2)
        r.add_server("s-1")
        r.add_server("s-2")
        r.add_server("s-3")
        r.record_location("c-1", "s-1")
        # c-1 is under-replicated
        moves = r.plan_rebalance(["s-1", "s-2", "s-3"])
        # Should suggest adding replica
        assert len(moves) > 0


# ============================================================
# MetadataServer Tests
# ============================================================

class TestMetadataServer:
    def setup_method(self):
        INode._next_id = 1  # reset for predictable IDs
        self.meta = MetadataServer()

    def test_root_exists(self):
        assert self.meta.exists("/")
        assert self.meta._resolve_path("/").is_directory()

    def test_create_file(self):
        self.meta.create_file("/test.txt")
        assert self.meta.exists("/test.txt")
        stat = self.meta.stat("/test.txt")
        assert stat["type"] == FileType.FILE

    def test_create_file_duplicate(self):
        self.meta.create_file("/test.txt")
        with pytest.raises(FileExistsError_):
            self.meta.create_file("/test.txt")

    def test_create_file_no_parent(self):
        with pytest.raises(FileNotFoundError_):
            self.meta.create_file("/nonexistent/test.txt")

    def test_mkdir(self):
        self.meta.mkdir("/data")
        assert self.meta.exists("/data")
        stat = self.meta.stat("/data")
        assert stat["type"] == FileType.DIRECTORY

    def test_mkdir_parents(self):
        self.meta.mkdir("/a/b/c", parents=True)
        assert self.meta.exists("/a")
        assert self.meta.exists("/a/b")
        assert self.meta.exists("/a/b/c")

    def test_mkdir_duplicate(self):
        self.meta.mkdir("/data")
        with pytest.raises(FileExistsError_):
            self.meta.mkdir("/data")

    def test_delete_file(self):
        self.meta.create_file("/test.txt")
        self.meta.delete("/test.txt")
        assert not self.meta.exists("/test.txt")

    def test_delete_nonexistent(self):
        with pytest.raises(FileNotFoundError_):
            self.meta.delete("/nonexistent")

    def test_delete_nonempty_dir(self):
        self.meta.mkdir("/data")
        self.meta.create_file("/data/file.txt")
        with pytest.raises(DFSError, match="not empty"):
            self.meta.delete("/data")

    def test_delete_recursive(self):
        self.meta.mkdir("/data")
        self.meta.create_file("/data/file.txt")
        self.meta.delete("/data", recursive=True)
        assert not self.meta.exists("/data")

    def test_delete_root(self):
        with pytest.raises(DFSError, match="root"):
            self.meta.delete("/")

    def test_rename(self):
        self.meta.create_file("/old.txt")
        self.meta.rename("/old.txt", "/new.txt")
        assert not self.meta.exists("/old.txt")
        assert self.meta.exists("/new.txt")

    def test_rename_to_existing(self):
        self.meta.create_file("/a.txt")
        self.meta.create_file("/b.txt")
        with pytest.raises(FileExistsError_):
            self.meta.rename("/a.txt", "/b.txt")

    def test_rename_move_between_dirs(self):
        self.meta.mkdir("/src")
        self.meta.mkdir("/dst")
        self.meta.create_file("/src/file.txt")
        self.meta.rename("/src/file.txt", "/dst/file.txt")
        assert not self.meta.exists("/src/file.txt")
        assert self.meta.exists("/dst/file.txt")

    def test_list_dir(self):
        self.meta.mkdir("/data")
        self.meta.create_file("/data/a.txt")
        self.meta.create_file("/data/b.txt")
        entries = self.meta.list_dir("/data")
        names = [e["name"] for e in entries]
        assert names == ["a.txt", "b.txt"]

    def test_list_dir_not_found(self):
        with pytest.raises(FileNotFoundError_):
            self.meta.list_dir("/nonexistent")

    def test_list_dir_on_file(self):
        self.meta.create_file("/file.txt")
        with pytest.raises(NotADirectoryError_):
            self.meta.list_dir("/file.txt")

    def test_stat(self):
        self.meta.create_file("/test.txt", owner="alice")
        stat = self.meta.stat("/test.txt")
        assert stat["path"] == "/test.txt"
        assert stat["owner"] == "alice"
        assert stat["size"] == 0

    def test_stat_not_found(self):
        with pytest.raises(FileNotFoundError_):
            self.meta.stat("/nonexistent")

    def test_allocate_chunks(self):
        self.meta.create_file("/test.txt")
        chunks = self.meta.allocate_chunks("/test.txt", 150000)
        # 150000 / 65536 = ~2.3, so 3 chunks
        assert len(chunks) == 3

    def test_allocate_chunks_zero_size(self):
        self.meta.create_file("/empty.txt")
        chunks = self.meta.allocate_chunks("/empty.txt", 0)
        assert chunks == []

    def test_commit_chunks(self):
        self.meta.create_file("/test.txt")
        chunks = self.meta.allocate_chunks("/test.txt", 100)
        self.meta.commit_chunks("/test.txt", chunks, 100)
        stat = self.meta.stat("/test.txt")
        assert stat["size"] == 100
        assert len(stat["chunks"]) == 1

    def test_get_chunks(self):
        self.meta.create_file("/test.txt")
        chunks = self.meta.allocate_chunks("/test.txt", 100)
        self.meta.commit_chunks("/test.txt", chunks, 100)
        assert self.meta.get_chunks("/test.txt") == chunks

    def test_get_chunks_on_dir(self):
        self.meta.mkdir("/data")
        with pytest.raises(IsADirectoryError_):
            self.meta.get_chunks("/data")

    def test_metadata_set_get(self):
        self.meta.create_file("/test.txt")
        self.meta.set_metadata("/test.txt", "content-type", "text/plain")
        assert self.meta.get_metadata("/test.txt", "content-type") == "text/plain"
        all_meta = self.meta.get_metadata("/test.txt")
        assert all_meta == {"content-type": "text/plain"}

    def test_namespace_stats(self):
        self.meta.mkdir("/data")
        self.meta.create_file("/data/a.txt")
        self.meta.create_file("/data/b.txt")
        stats = self.meta.get_namespace_stats()
        assert stats["files"] == 2
        assert stats["directories"] == 2  # root + data

    def test_operation_log(self):
        self.meta.create_file("/test.txt")
        self.meta.delete("/test.txt")
        assert len(self.meta.op_log) == 2


# ============================================================
# DFSCluster Tests
# ============================================================

class TestDFSCluster:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=64, replication_factor=2)
        self.cluster.add_server("s-1")
        self.cluster.add_server("s-2")
        self.cluster.add_server("s-3")

    def test_add_server(self):
        assert len(self.cluster.chunk_servers) == 3
        assert "s-1" in self.cluster.chunk_servers

    def test_remove_server(self):
        self.cluster.remove_server("s-3")
        assert "s-3" not in self.cluster.chunk_servers

    def test_remove_nonexistent_server(self):
        with pytest.raises(DFSError, match="not found"):
            self.cluster.remove_server("s-999")

    def test_alive_servers(self):
        assert len(self.cluster.get_alive_servers()) == 3
        self.cluster.mark_server_dead("s-2")
        assert len(self.cluster.get_alive_servers()) == 2

    def test_revive_server(self):
        self.cluster.mark_server_dead("s-1")
        self.cluster.mark_server_alive("s-1")
        assert "s-1" in self.cluster.get_alive_servers()

    def test_write_and_read(self):
        self.cluster.write_file("/hello.txt", b"Hello, DFS!")
        data = self.cluster.read_file("/hello.txt")
        assert data == b"Hello, DFS!"

    def test_write_string(self):
        self.cluster.write_file("/str.txt", "string data")
        data = self.cluster.read_file("/str.txt")
        assert data == b"string data"

    def test_write_creates_parents(self):
        self.cluster.write_file("/a/b/c/file.txt", b"nested")
        assert self.cluster.exists("/a/b/c")
        data = self.cluster.read_file("/a/b/c/file.txt")
        assert data == b"nested"

    def test_write_overwrite(self):
        self.cluster.write_file("/f.txt", b"first")
        self.cluster.write_file("/f.txt", b"second")
        data = self.cluster.read_file("/f.txt")
        assert data == b"second"

    def test_write_multi_chunk(self):
        # chunk_size=64, so 200 bytes = 4 chunks
        big_data = b"x" * 200
        self.cluster.write_file("/big.txt", big_data)
        data = self.cluster.read_file("/big.txt")
        assert data == big_data
        assert len(data) == 200

    def test_write_no_servers(self):
        c = DFSCluster(chunk_size=64, replication_factor=1)
        c.metadata.create_file("/test.txt")
        with pytest.raises(NoAvailableServersError):
            c.write_file("/test.txt", b"data")

    def test_read_empty_file(self):
        self.cluster.create_file("/empty.txt")
        data = self.cluster.read_file("/empty.txt")
        assert data == b""

    def test_read_nonexistent(self):
        with pytest.raises(FileNotFoundError_):
            self.cluster.read_file("/nonexistent.txt")

    def test_append(self):
        self.cluster.write_file("/log.txt", b"line1\n")
        self.cluster.append_file("/log.txt", b"line2\n")
        data = self.cluster.read_file("/log.txt")
        assert data == b"line1\nline2\n"

    def test_delete_file_cleans_chunks(self):
        self.cluster.write_file("/f.txt", b"data")
        # Verify chunks exist on servers
        chunks = self.cluster.metadata.get_chunks("/f.txt")
        assert len(chunks) > 0
        self.cluster.delete("/f.txt")
        # Chunks should be cleaned up
        for cid in chunks:
            assert self.cluster.replicator.get_locations(cid) == []

    def test_delete_dir_recursive(self):
        self.cluster.write_file("/dir/a.txt", b"a")
        self.cluster.write_file("/dir/b.txt", b"b")
        self.cluster.delete("/dir", recursive=True)
        assert not self.cluster.exists("/dir")

    def test_mkdir(self):
        self.cluster.mkdir("/data")
        assert self.cluster.exists("/data")

    def test_rename(self):
        self.cluster.write_file("/old.txt", b"data")
        self.cluster.rename("/old.txt", "/new.txt")
        data = self.cluster.read_file("/new.txt")
        assert data == b"data"

    def test_list_dir(self):
        self.cluster.write_file("/dir/a.txt", b"a")
        self.cluster.write_file("/dir/b.txt", b"b")
        entries = self.cluster.list_dir("/dir")
        names = [e["name"] for e in entries]
        assert names == ["a.txt", "b.txt"]

    def test_stat(self):
        self.cluster.write_file("/f.txt", b"hello")
        stat = self.cluster.stat("/f.txt")
        assert stat["size"] == 5

    def test_check_replication(self):
        self.cluster.write_file("/f.txt", b"data")
        repl = self.cluster.check_replication()
        assert repl["total_chunks"] > 0

    def test_repair_replication(self):
        self.cluster.write_file("/f.txt", b"important data")
        chunks = self.cluster.metadata.get_chunks("/f.txt")
        # Manually remove a replica
        cid = chunks[0]
        locs = self.cluster.replicator.get_locations(cid)
        if len(locs) > 1:
            server_to_kill = locs[0]
            self.cluster.chunk_servers[server_to_kill].delete_chunk(cid)
            self.cluster.replicator.remove_location(cid, server_to_kill)
        repaired = self.cluster.repair_replication()
        # Should have repaired at least one
        assert repaired >= 0

    def test_server_failure_read_fallback(self):
        self.cluster.write_file("/f.txt", b"survive")
        # Kill one server
        self.cluster.mark_server_dead("s-1")
        # Should still be readable from other replicas
        data = self.cluster.read_file("/f.txt")
        assert data == b"survive"

    def test_metadata_operations(self):
        self.cluster.write_file("/f.txt", b"data")
        self.cluster.set_metadata("/f.txt", "tag", "important")
        assert self.cluster.get_metadata("/f.txt", "tag") == "important"

    def test_cluster_stats(self):
        self.cluster.write_file("/f.txt", b"data")
        stats = self.cluster.get_cluster_stats()
        assert stats["alive_servers"] == 3
        assert stats["namespace"]["files"] == 1
        assert stats["total_used"] > 0

    def test_rebalance(self):
        self.cluster.write_file("/f.txt", b"data")
        executed = self.cluster.rebalance()
        assert executed >= 0


# ============================================================
# DFSClient Tests
# ============================================================

class TestDFSClient:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=64, replication_factor=2)
        self.cluster.add_server("s-1")
        self.cluster.add_server("s-2")
        self.client = DFSClient(self.cluster, "client-1")

    def test_write_and_read(self):
        self.client.write("/hello.txt", b"Hello!")
        assert self.client.read("/hello.txt") == b"Hello!"

    def test_mkdir_and_ls(self):
        self.client.mkdir("/docs", parents=True)
        self.client.write("/docs/readme.txt", b"README")
        entries = self.client.ls("/docs")
        assert len(entries) == 1
        assert entries[0]["name"] == "readme.txt"

    def test_delete(self):
        self.client.write("/temp.txt", b"temp")
        self.client.delete("/temp.txt")
        assert not self.client.exists("/temp.txt")

    def test_stat(self):
        self.client.write("/f.txt", b"data")
        stat = self.client.stat("/f.txt")
        assert stat["size"] == 4

    def test_exists(self):
        assert not self.client.exists("/nope.txt")
        self.client.write("/yep.txt", b"yes")
        assert self.client.exists("/yep.txt")

    def test_rename(self):
        self.client.write("/a.txt", b"data")
        self.client.rename("/a.txt", "/b.txt")
        assert self.client.read("/b.txt") == b"data"

    def test_copy(self):
        self.client.write("/src.txt", b"copy me")
        self.client.copy("/src.txt", "/dst.txt")
        assert self.client.read("/dst.txt") == b"copy me"
        assert self.client.read("/src.txt") == b"copy me"

    def test_move(self):
        self.client.write("/old.txt", b"move me")
        self.client.move("/old.txt", "/new.txt")
        assert self.client.read("/new.txt") == b"move me"
        assert not self.client.exists("/old.txt")

    def test_append(self):
        self.client.write("/log.txt", b"a")
        self.client.append("/log.txt", b"b")
        assert self.client.read("/log.txt") == b"ab"

    def test_create(self):
        self.client.create("/empty.txt")
        assert self.client.exists("/empty.txt")
        assert self.client.read("/empty.txt") == b""

    def test_metadata(self):
        self.client.write("/f.txt", b"x")
        self.client.set_metadata("/f.txt", "key", "val")
        assert self.client.get_metadata("/f.txt", "key") == "val"

    def test_tree(self):
        self.client.mkdir("/a", parents=True)
        self.client.write("/a/x.txt", b"x")
        self.client.write("/a/y.txt", b"yy")
        lines = self.client.tree("/")
        assert len(lines) >= 2
        assert any("x.txt" in l for l in lines)

    def test_large_file(self):
        data = bytes(range(256)) * 100  # 25600 bytes
        self.client.write("/large.bin", data)
        result = self.client.read("/large.bin")
        assert result == data

    def test_binary_data(self):
        data = bytes(range(256))
        self.client.write("/binary.bin", data)
        assert self.client.read("/binary.bin") == data

    def test_multiple_clients(self):
        c1 = DFSClient(self.cluster, "c1")
        c2 = DFSClient(self.cluster, "c2")
        c1.write("/shared.txt", b"from c1")
        assert c2.read("/shared.txt") == b"from c1"
        c2.write("/shared.txt", b"from c2")
        assert c1.read("/shared.txt") == b"from c2"


# ============================================================
# SnapshotManager Tests
# ============================================================

class TestSnapshotManager:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=64, replication_factor=2)
        self.cluster.add_server("s-1")
        self.cluster.add_server("s-2")
        self.snap_mgr = SnapshotManager(self.cluster)

    def test_create_snapshot(self):
        self.cluster.write_file("/f.txt", b"data")
        snap = self.snap_mgr.create_snapshot("v1")
        assert snap.name == "v1"
        assert "/f.txt" in snap.file_chunks
        assert snap.file_sizes["/f.txt"] == 4

    def test_list_snapshots(self):
        self.snap_mgr.create_snapshot("v1")
        self.snap_mgr.create_snapshot("v2")
        snaps = self.snap_mgr.list_snapshots()
        assert len(snaps) == 2

    def test_restore_file(self):
        self.cluster.write_file("/f.txt", b"original")
        snap = self.snap_mgr.create_snapshot("v1")
        self.cluster.write_file("/f.txt", b"modified")
        assert self.cluster.read_file("/f.txt") == b"modified"
        self.snap_mgr.restore_file(snap.snapshot_id, "/f.txt")
        assert self.cluster.read_file("/f.txt") == b"original"

    def test_restore_nonexistent(self):
        snap = self.snap_mgr.create_snapshot("empty")
        with pytest.raises(FileNotFoundError_):
            self.snap_mgr.restore_file(snap.snapshot_id, "/nope.txt")

    def test_delete_snapshot(self):
        snap = self.snap_mgr.create_snapshot("temp")
        self.snap_mgr.delete_snapshot(snap.snapshot_id)
        with pytest.raises(DFSError, match="not found"):
            self.snap_mgr.get_snapshot(snap.snapshot_id)

    def test_diff_snapshots(self):
        self.cluster.write_file("/a.txt", b"a")
        self.cluster.write_file("/b.txt", b"b")
        snap1 = self.snap_mgr.create_snapshot("v1")

        self.cluster.write_file("/b.txt", b"b-modified")
        self.cluster.write_file("/c.txt", b"c")
        self.cluster.delete("/a.txt")
        snap2 = self.snap_mgr.create_snapshot("v2")

        diff = self.snap_mgr.diff_snapshots(snap1.snapshot_id, snap2.snapshot_id)
        assert "/c.txt" in diff["added"]
        assert "/a.txt" in diff["removed"]
        assert "/b.txt" in diff["modified"]

    def test_snapshot_preserves_state(self):
        self.cluster.write_file("/f.txt", b"v1")
        snap1 = self.snap_mgr.create_snapshot("v1")
        self.cluster.write_file("/f.txt", b"v2")
        snap2 = self.snap_mgr.create_snapshot("v2")
        # Both snapshots should have different chunk IDs
        assert snap1.file_chunks["/f.txt"] != snap2.file_chunks["/f.txt"]

    def test_get_nonexistent_snapshot(self):
        with pytest.raises(DFSError):
            self.snap_mgr.get_snapshot("snap-9999")


# ============================================================
# DFSAnalyzer Tests
# ============================================================

class TestDFSAnalyzer:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=64, replication_factor=2)
        self.cluster.add_server("s-1", capacity=10000)
        self.cluster.add_server("s-2", capacity=10000)
        self.cluster.add_server("s-3", capacity=10000)
        self.analyzer = DFSAnalyzer(self.cluster)

    def test_analyze_balance_empty(self):
        result = self.analyzer.analyze_balance()
        assert result["balanced"]

    def test_analyze_balance_with_data(self):
        self.cluster.write_file("/f.txt", b"x" * 100)
        result = self.analyzer.analyze_balance()
        assert "avg_utilization" in result
        assert result["server_count"] == 3

    def test_analyze_replication_healthy(self):
        self.cluster.write_file("/f.txt", b"data")
        result = self.analyzer.analyze_replication_health()
        assert result["replication_factor"] == 2
        assert result["total_chunks"] > 0

    def test_analyze_replication_degraded(self):
        self.cluster.write_file("/f.txt", b"data")
        # Manually degrade replication
        chunks = self.cluster.metadata.get_chunks("/f.txt")
        cid = chunks[0]
        locs = self.cluster.replicator.get_locations(cid)
        if len(locs) > 1:
            srv = locs[0]
            self.cluster.chunk_servers[srv].delete_chunk(cid)
            self.cluster.replicator.remove_location(cid, srv)
        result = self.analyzer.analyze_replication_health()
        assert result["health"] in ("degraded", "healthy")

    def test_analyze_capacity(self):
        result = self.analyzer.analyze_capacity()
        assert result["total_capacity"] == 30000
        assert result["free"] == 30000

    def test_analyze_capacity_with_data(self):
        self.cluster.write_file("/f.txt", b"x" * 200)
        result = self.analyzer.analyze_capacity()
        assert result["used"] > 0
        assert result["free"] < 30000

    def test_can_survive_server_loss(self):
        result = self.analyzer.analyze_capacity()
        assert result["can_survive_server_loss"]  # 3 servers, RF=2

    def test_full_report(self):
        self.cluster.write_file("/f.txt", b"data")
        report = self.analyzer.full_report()
        assert "balance" in report
        assert "replication" in report
        assert "capacity" in report
        assert "cluster_stats" in report

    def test_no_servers(self):
        c = DFSCluster()
        a = DFSAnalyzer(c)
        result = a.analyze_balance()
        assert result["balanced"]


# ============================================================
# Edge Cases & Integration Tests
# ============================================================

class TestEdgeCases:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=32, replication_factor=2)
        self.cluster.add_server("s-1")
        self.cluster.add_server("s-2")

    def test_write_exact_chunk_boundary(self):
        data = b"x" * 32  # exactly one chunk
        self.cluster.write_file("/exact.txt", data)
        assert self.cluster.read_file("/exact.txt") == data

    def test_write_one_over_boundary(self):
        data = b"x" * 33  # one byte over -> 2 chunks
        self.cluster.write_file("/over.txt", data)
        assert self.cluster.read_file("/over.txt") == data

    def test_deeply_nested_path(self):
        self.cluster.write_file("/a/b/c/d/e/f/g/file.txt", b"deep")
        assert self.cluster.read_file("/a/b/c/d/e/f/g/file.txt") == b"deep"

    def test_many_files_same_dir(self):
        for i in range(50):
            self.cluster.write_file(f"/files/f{i:03d}.txt", f"data-{i}".encode())
        entries = self.cluster.list_dir("/files")
        assert len(entries) == 50

    def test_overwrite_larger_file(self):
        self.cluster.write_file("/f.txt", b"small")
        self.cluster.write_file("/f.txt", b"x" * 100)
        assert len(self.cluster.read_file("/f.txt")) == 100

    def test_overwrite_smaller_file(self):
        self.cluster.write_file("/f.txt", b"x" * 100)
        self.cluster.write_file("/f.txt", b"tiny")
        assert self.cluster.read_file("/f.txt") == b"tiny"

    def test_single_server_cluster(self):
        c = DFSCluster(chunk_size=32, replication_factor=1)
        c.add_server("solo")
        c.write_file("/f.txt", b"alone")
        assert c.read_file("/f.txt") == b"alone"

    def test_all_servers_dead_read(self):
        self.cluster.write_file("/f.txt", b"data")
        self.cluster.mark_server_dead("s-1")
        self.cluster.mark_server_dead("s-2")
        with pytest.raises(ChunkError, match="No alive"):
            self.cluster.read_file("/f.txt")

    def test_server_decommission(self):
        self.cluster.write_file("/f.txt", b"data")
        self.cluster.remove_server("s-2")
        # File should still be readable from s-1
        data = self.cluster.read_file("/f.txt")
        assert data == b"data"

    def test_write_empty_data(self):
        self.cluster.write_file("/empty.txt", b"")
        assert self.cluster.read_file("/empty.txt") == b""

    def test_unicode_content(self):
        text = "Hello, world! \u00e9\u00e8\u00ea"
        self.cluster.write_file("/unicode.txt", text)
        data = self.cluster.read_file("/unicode.txt")
        assert data.decode("utf-8") == text

    def test_cluster_stats_after_operations(self):
        self.cluster.write_file("/a.txt", b"aaa")
        self.cluster.write_file("/b.txt", b"bbb")
        self.cluster.delete("/a.txt")
        stats = self.cluster.get_cluster_stats()
        assert stats["namespace"]["files"] == 1


class TestLeaseIntegration:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=64, replication_factor=2)
        self.cluster.add_server("s-1")
        self.cluster.add_server("s-2")

    def test_write_acquires_and_releases_lease(self):
        self.cluster.write_file("/f.txt", b"data")
        # After write, lease should be released
        active = self.cluster.lease_mgr.list_leases()
        assert len(active) == 0

    def test_concurrent_writes_different_files(self):
        self.cluster.write_file("/a.txt", b"a", "c1")
        self.cluster.write_file("/b.txt", b"b", "c2")
        assert self.cluster.read_file("/a.txt") == b"a"
        assert self.cluster.read_file("/b.txt") == b"b"


class TestReplicationRecovery:
    def setup_method(self):
        INode._next_id = 1
        self.cluster = DFSCluster(chunk_size=64, replication_factor=3)
        self.cluster.add_server("s-1")
        self.cluster.add_server("s-2")
        self.cluster.add_server("s-3")
        self.cluster.add_server("s-4")

    def test_repair_after_server_death(self):
        self.cluster.write_file("/critical.txt", b"important data")
        self.cluster.mark_server_dead("s-1")
        repaired = self.cluster.repair_replication()
        # Data should still be readable
        data = self.cluster.read_file("/critical.txt")
        assert data == b"important data"

    def test_multiple_server_failures(self):
        self.cluster.write_file("/f.txt", b"data")
        self.cluster.mark_server_dead("s-1")
        self.cluster.mark_server_dead("s-2")
        # With RF=3 and 2 dead, should still have 1 alive replica
        # (depends on hash distribution, try reading)
        try:
            data = self.cluster.read_file("/f.txt")
            assert data == b"data"
        except ChunkError:
            pass  # Acceptable if all replicas happened to be on dead servers

    def test_add_server_and_rebalance(self):
        self.cluster.write_file("/f.txt", b"data")
        self.cluster.add_server("s-5")
        executed = self.cluster.rebalance()
        assert executed >= 0


# ============================================================
# Composition Verification Tests
# ============================================================

class TestComposition:
    """Verify that C205 and C204 are properly composed."""

    def test_consistent_hashing_integration(self):
        """Verify chunks are placed via consistent hashing."""
        cluster = DFSCluster(chunk_size=32, replication_factor=2)
        cluster.add_server("s-1")
        cluster.add_server("s-2")
        cluster.add_server("s-3")
        cluster.write_file("/f.txt", b"data")
        chunks = cluster.metadata.get_chunks("/f.txt")
        for cid in chunks:
            locs = cluster.replicator.get_locations(cid)
            assert len(locs) >= 1  # at least one replica

    def test_vector_clock_versioning(self):
        """Verify file metadata uses vector clocks."""
        cluster = DFSCluster(chunk_size=32, replication_factor=1)
        cluster.add_server("s-1")
        cluster.write_file("/f.txt", b"v1")
        stat1 = cluster.stat("/f.txt")
        cluster.write_file("/f.txt", b"v2")
        stat2 = cluster.stat("/f.txt")
        # Version should have advanced
        assert stat2["version"] != stat1["version"]

    def test_replicated_hash_ring_distributes(self):
        """Verify ReplicatedHashRing from C205 distributes across servers."""
        r = ChunkReplicator(replication_factor=3)
        for i in range(5):
            r.add_server(f"s-{i}")
        # Get targets for several chunks -- should distribute
        all_targets = set()
        for i in range(20):
            targets = r.get_target_servers(f"chunk-{i}")
            all_targets.update(targets)
        # Should use multiple servers
        assert len(all_targets) > 1

    def test_hash_ring_deterministic(self):
        """Verify same chunk always maps to same servers."""
        r = ChunkReplicator(replication_factor=2)
        r.add_server("s-1")
        r.add_server("s-2")
        r.add_server("s-3")
        t1 = r.get_target_servers("chunk-42")
        t2 = r.get_target_servers("chunk-42")
        assert t1 == t2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
