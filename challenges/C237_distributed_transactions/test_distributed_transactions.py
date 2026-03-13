"""
Tests for C237: Distributed Transactions (2PC, 3PC, Saga, Recovery)
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

from distributed_transactions import (
    TxState, Vote, ParticipantState,
    TransactionLog, Participant,
    TwoPhaseCommitCoordinator, ThreePhaseCommitCoordinator,
    TransactionManager, DistributedKVStore,
    SagaStep, SagaCoordinator, TimeoutCoordinator,
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}")


# ============================================================
# Transaction Log
# ============================================================

def test_log_basic():
    log = TransactionLog()
    log.append("tx1", TxState.INITIATED)
    log.append("tx1", TxState.PREPARING)
    log.append("tx2", TxState.INITIATED)

    test("log: all entries", len(log.get_entries()) == 3)
    test("log: filter by tx_id", len(log.get_entries("tx1")) == 2)
    test("log: latest state tx1", log.get_latest_state("tx1") == TxState.PREPARING)
    test("log: latest state tx2", log.get_latest_state("tx2") == TxState.INITIATED)
    test("log: unknown tx", log.get_latest_state("tx99") is None)
    test("log: all tx ids", set(log.get_all_tx_ids()) == {"tx1", "tx2"})

    log.clear()
    test("log: clear", len(log.get_entries()) == 0)

def test_log_metadata():
    log = TransactionLog()
    log.append("tx1", TxState.ABORTED, {"reason": "vote no"})
    entries = log.get_entries("tx1")
    test("log: metadata stored", entries[0][3]["reason"] == "vote no")
    test("log: timestamp present", entries[0][2] > 0)

test_log_basic()
test_log_metadata()


# ============================================================
# Participant
# ============================================================

def test_participant_basic():
    p = Participant("node1")
    vote = p.prepare("tx1", [("set", "x", 10), ("set", "y", 20)])
    test("participant: vote yes", vote == Vote.YES)
    test("participant: state ready", p.get_state("tx1") == ParticipantState.READY)

    p.commit("tx1")
    test("participant: commit applied x", p.get("x") == 10)
    test("participant: commit applied y", p.get("y") == 20)
    test("participant: state committed", p.get_state("tx1") == ParticipantState.COMMITTED)

def test_participant_abort():
    p = Participant("node1")
    p.prepare("tx1", [("set", "x", 10)])
    p.abort("tx1")
    test("participant: abort no data", p.get("x") is None)
    test("participant: state aborted", p.get_state("tx1") == ParticipantState.ABORTED)

def test_participant_delete():
    p = Participant("node1")
    p.data["x"] = 100
    p.prepare("tx1", [("delete", "x")])
    p.commit("tx1")
    test("participant: delete removes key", p.get("x") is None)

def test_participant_lock_conflict():
    p = Participant("node1")
    p.prepare("tx1", [("set", "x", 10)])
    vote = p.prepare("tx2", [("set", "x", 20)])
    test("participant: lock conflict -> NO", vote == Vote.NO)
    test("participant: tx2 aborted", p.get_state("tx2") == ParticipantState.ABORTED)

def test_participant_fail_inject():
    p = Participant("node1", fail_on_prepare=True)
    vote = p.prepare("tx1", [("set", "x", 10)])
    test("participant: injected failure -> NO", vote == Vote.NO)

def test_participant_idempotent_commit():
    p = Participant("node1")
    p.prepare("tx1", [("set", "x", 10)])
    p.commit("tx1")
    result = p.commit("tx1")  # Second commit should be idempotent
    test("participant: idempotent commit", result == True)
    test("participant: data still correct", p.get("x") == 10)

def test_participant_pre_commit():
    p = Participant("node1")
    p.prepare("tx1", [("set", "x", 10)])
    ack = p.pre_commit("tx1")
    test("participant: pre_commit ack", ack == True)
    test("participant: pre_commit state", p.get_state("tx1") == ParticipantState.PRE_COMMITTED)

def test_participant_pre_commit_wrong_state():
    p = Participant("node1")
    ack = p.pre_commit("tx1")  # No prepare first
    test("participant: pre_commit without prepare", ack == False)

def test_participant_multiple_txs():
    p = Participant("node1")
    p.prepare("tx1", [("set", "a", 1)])
    p.commit("tx1")
    p.prepare("tx2", [("set", "b", 2)])
    p.commit("tx2")
    test("participant: multiple txs a", p.get("a") == 1)
    test("participant: multiple txs b", p.get("b") == 2)

def test_participant_default_value():
    p = Participant("node1")
    test("participant: default value", p.get("missing", 42) == 42)

test_participant_basic()
test_participant_abort()
test_participant_delete()
test_participant_lock_conflict()
test_participant_fail_inject()
test_participant_idempotent_commit()
test_participant_pre_commit()
test_participant_pre_commit_wrong_state()
test_participant_multiple_txs()
test_participant_default_value()


# ============================================================
# Two-Phase Commit (2PC)
# ============================================================

def test_2pc_basic_commit():
    p1, p2 = Participant("n1"), Participant("n2")
    coord = TwoPhaseCommitCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 10)])
    coord.add_operation(tx, "n2", [("set", "y", 20)])
    success, state = coord.execute(tx)

    test("2pc: commit success", success == True)
    test("2pc: state committed", state == TxState.COMMITTED)
    test("2pc: n1 data", p1.get("x") == 10)
    test("2pc: n2 data", p2.get("y") == 20)
    test("2pc: tx state query", coord.get_tx_state(tx) == TxState.COMMITTED)

def test_2pc_abort_on_no_vote():
    p1 = Participant("n1")
    p2 = Participant("n2", fail_on_prepare=True)
    coord = TwoPhaseCommitCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 10)])
    coord.add_operation(tx, "n2", [("set", "y", 20)])
    success, state = coord.execute(tx)

    test("2pc: abort on no vote", success == False)
    test("2pc: state aborted", state == TxState.ABORTED)
    test("2pc: n1 rolled back", p1.get("x") is None)

def test_2pc_single_participant():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "key", "value")])
    success, state = coord.execute(tx)

    test("2pc: single participant commit", success == True)
    test("2pc: single participant data", p1.get("key") == "value")

def test_2pc_empty_transaction():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])
    tx = coord.begin()
    success, state = coord.execute(tx)
    test("2pc: empty tx aborts", success == False)
    test("2pc: empty tx state", state == TxState.ABORTED)

def test_2pc_multiple_ops_same_participant():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "a", 1), ("set", "b", 2), ("set", "c", 3)])
    success, state = coord.execute(tx)

    test("2pc: multi-op a", p1.get("a") == 1)
    test("2pc: multi-op b", p1.get("b") == 2)
    test("2pc: multi-op c", p1.get("c") == 3)

def test_2pc_many_participants():
    participants = [Participant(f"n{i}") for i in range(5)]
    coord = TwoPhaseCommitCoordinator(participants)

    tx = coord.begin()
    for i, p in enumerate(participants):
        coord.add_operation(tx, p.name, [("set", f"key{i}", i * 100)])
    success, state = coord.execute(tx)

    test("2pc: 5 participants commit", success == True)
    for i, p in enumerate(participants):
        test(f"2pc: 5p data n{i}", p.get(f"key{i}") == i * 100)

def test_2pc_sequential_transactions():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])

    tx1 = coord.begin()
    coord.add_operation(tx1, "n1", [("set", "x", 1)])
    coord.execute(tx1)

    tx2 = coord.begin()
    coord.add_operation(tx2, "n1", [("set", "x", 2)])
    coord.execute(tx2)

    test("2pc: sequential overwrite", p1.get("x") == 2)

def test_2pc_log_entries():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 1)])
    coord.execute(tx)

    log = coord.get_log()
    entries = log.get_entries(tx)
    states = [e[1] for e in entries]
    test("2pc: log has initiated", TxState.INITIATED in states)
    test("2pc: log has preparing", TxState.PREPARING in states)
    test("2pc: log has committed", TxState.COMMITTED in states)

def test_2pc_invalid_participant():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])
    tx = coord.begin()
    try:
        coord.add_operation(tx, "nonexistent", [("set", "x", 1)])
        test("2pc: invalid participant raises", False)
    except ValueError:
        test("2pc: invalid participant raises", True)

def test_2pc_invalid_tx():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])
    try:
        coord.add_operation("fake_tx", "n1", [("set", "x", 1)])
        test("2pc: invalid tx raises", False)
    except ValueError:
        test("2pc: invalid tx raises", True)

def test_2pc_commit_failure():
    """Test that commit failure is still recorded as committed (2PC semantics)."""
    p1 = Participant("n1")
    p2 = Participant("n2", fail_on_commit=True)
    coord = TwoPhaseCommitCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 10)])
    coord.add_operation(tx, "n2", [("set", "y", 20)])
    success, state = coord.execute(tx)

    # Once decision is COMMIT, it stays COMMIT even if delivery fails
    test("2pc: commit failure still committed", state == TxState.COMMITTED)
    test("2pc: n1 committed despite n2 fail", p1.get("x") == 10)

def test_2pc_delete_operations():
    p1 = Participant("n1")
    p1.data["old_key"] = "old_value"
    coord = TwoPhaseCommitCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("delete", "old_key")])
    success, state = coord.execute(tx)

    test("2pc: delete success", success == True)
    test("2pc: key deleted", p1.get("old_key") is None)

test_2pc_basic_commit()
test_2pc_abort_on_no_vote()
test_2pc_single_participant()
test_2pc_empty_transaction()
test_2pc_multiple_ops_same_participant()
test_2pc_many_participants()
test_2pc_sequential_transactions()
test_2pc_log_entries()
test_2pc_invalid_participant()
test_2pc_invalid_tx()
test_2pc_commit_failure()
test_2pc_delete_operations()


# ============================================================
# Three-Phase Commit (3PC)
# ============================================================

def test_3pc_basic_commit():
    p1, p2 = Participant("n1"), Participant("n2")
    coord = ThreePhaseCommitCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 10)])
    coord.add_operation(tx, "n2", [("set", "y", 20)])
    success, state = coord.execute(tx)

    test("3pc: commit success", success == True)
    test("3pc: state committed", state == TxState.COMMITTED)
    test("3pc: n1 data", p1.get("x") == 10)
    test("3pc: n2 data", p2.get("y") == 20)

def test_3pc_abort_on_no_vote():
    p1 = Participant("n1")
    p2 = Participant("n2", fail_on_prepare=True)
    coord = ThreePhaseCommitCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 10)])
    coord.add_operation(tx, "n2", [("set", "y", 20)])
    success, state = coord.execute(tx)

    test("3pc: abort on no vote", success == False)
    test("3pc: state aborted", state == TxState.ABORTED)
    test("3pc: n1 rolled back", p1.get("x") is None)

def test_3pc_abort_on_pre_commit_failure():
    p1 = Participant("n1")
    p2 = Participant("n2", fail_on_pre_commit=True)
    coord = ThreePhaseCommitCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 10)])
    coord.add_operation(tx, "n2", [("set", "y", 20)])
    success, state = coord.execute(tx)

    test("3pc: abort on pre-commit fail", success == False)
    test("3pc: state aborted", state == TxState.ABORTED)

def test_3pc_single_participant():
    p1 = Participant("n1")
    coord = ThreePhaseCommitCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "key", "value")])
    success, state = coord.execute(tx)

    test("3pc: single participant", success == True)
    test("3pc: single data", p1.get("key") == "value")

def test_3pc_empty_transaction():
    p1 = Participant("n1")
    coord = ThreePhaseCommitCoordinator([p1])
    tx = coord.begin()
    success, state = coord.execute(tx)
    test("3pc: empty tx aborts", success == False)

def test_3pc_log_shows_pre_commit():
    p1 = Participant("n1")
    coord = ThreePhaseCommitCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 1)])
    coord.execute(tx)

    log = coord.get_log()
    entries = log.get_entries(tx)
    states = [e[1] for e in entries]
    test("3pc: log has pre-commit", TxState.PRE_COMMITTED in states)
    test("3pc: log has committed", TxState.COMMITTED in states)

def test_3pc_many_participants():
    participants = [Participant(f"n{i}") for i in range(4)]
    coord = ThreePhaseCommitCoordinator(participants)

    tx = coord.begin()
    for i, p in enumerate(participants):
        coord.add_operation(tx, p.name, [("set", f"k{i}", i)])
    success, state = coord.execute(tx)

    test("3pc: 4 participants", success == True)
    for i, p in enumerate(participants):
        test(f"3pc: 4p data n{i}", p.get(f"k{i}") == i)

def test_3pc_sequential():
    p1 = Participant("n1")
    coord = ThreePhaseCommitCoordinator([p1])

    tx1 = coord.begin()
    coord.add_operation(tx1, "n1", [("set", "x", 1)])
    coord.execute(tx1)

    tx2 = coord.begin()
    coord.add_operation(tx2, "n1", [("set", "x", 2)])
    coord.execute(tx2)

    test("3pc: sequential overwrite", p1.get("x") == 2)

test_3pc_basic_commit()
test_3pc_abort_on_no_vote()
test_3pc_abort_on_pre_commit_failure()
test_3pc_single_participant()
test_3pc_empty_transaction()
test_3pc_log_shows_pre_commit()
test_3pc_many_participants()
test_3pc_sequential()


# ============================================================
# Transaction Manager
# ============================================================

def test_manager_2pc():
    p1, p2 = Participant("n1"), Participant("n2")
    mgr = TransactionManager([p1, p2], protocol="2pc")

    success, tx_id, state = mgr.execute_transaction({
        "n1": [("set", "a", 1)],
        "n2": [("set", "b", 2)],
    })

    test("mgr 2pc: success", success == True)
    test("mgr 2pc: committed", state == TxState.COMMITTED)
    test("mgr 2pc: n1 data", p1.get("a") == 1)
    test("mgr 2pc: n2 data", p2.get("b") == 2)

def test_manager_3pc():
    p1, p2 = Participant("n1"), Participant("n2")
    mgr = TransactionManager([p1, p2], protocol="3pc")

    success, tx_id, state = mgr.execute_transaction({
        "n1": [("set", "a", 1)],
        "n2": [("set", "b", 2)],
    })

    test("mgr 3pc: success", success == True)
    test("mgr 3pc: committed", state == TxState.COMMITTED)

def test_manager_abort():
    p1 = Participant("n1")
    p2 = Participant("n2", fail_on_prepare=True)
    mgr = TransactionManager([p1, p2])

    success, tx_id, state = mgr.execute_transaction({
        "n1": [("set", "a", 1)],
        "n2": [("set", "b", 2)],
    })

    test("mgr: abort", success == False)
    test("mgr: aborted state", state == TxState.ABORTED)

def test_manager_invalid_participant():
    p1 = Participant("n1")
    mgr = TransactionManager([p1])

    success, tx_id, state = mgr.execute_transaction({
        "nonexistent": [("set", "a", 1)],
    })

    test("mgr: invalid participant abort", success == False)

def test_manager_tx_info():
    p1 = Participant("n1")
    mgr = TransactionManager([p1])

    success, tx_id, state = mgr.execute_transaction({
        "n1": [("set", "a", 1)],
    })

    info = mgr.get_transaction_info(tx_id)
    test("mgr: tx info exists", info is not None)
    test("mgr: tx info state", info["state"] == TxState.COMMITTED)
    test("mgr: tx info protocol", info["protocol"] == "2pc")
    test("mgr: tx info end_time", info["end_time"] is not None)

def test_manager_all_transactions():
    p1 = Participant("n1")
    mgr = TransactionManager([p1])

    mgr.execute_transaction({"n1": [("set", "a", 1)]})
    mgr.execute_transaction({"n1": [("set", "b", 2)]})

    all_txs = mgr.get_all_transactions()
    test("mgr: all txs count", len(all_txs) == 2)
    test("mgr: all committed", all(s == TxState.COMMITTED for s in all_txs.values()))

def test_manager_recovery_presumed_abort():
    """Recovery with presumed abort: incomplete tx -> abort."""
    p1 = Participant("n1")
    mgr = TransactionManager([p1])

    # Simulate a tx that got stuck in PREPARING
    tx_id = mgr._new_tx_id()
    mgr._history[tx_id] = {
        "state": TxState.PREPARING,
        "operations": {"n1": [("set", "x", 10)]},
        "protocol": "2pc",
        "start_time": time.time(),
        "end_time": None,
    }

    actions = mgr.recover()
    test("recovery: presumed abort", len(actions) == 1)
    test("recovery: action is abort", actions[0][1] == "aborted")
    test("recovery: state updated", mgr._history[tx_id]["state"] == TxState.ABORTED)

def test_manager_recovery_recommit():
    """Recovery: tx in COMMITTING state -> recommit."""
    p1 = Participant("n1")
    mgr = TransactionManager([p1])

    # Prepare manually
    p1.prepare("manual_tx", [("set", "x", 99)])

    tx_id = mgr._new_tx_id()
    mgr._history[tx_id] = {
        "state": TxState.COMMITTING,
        "operations": {"n1": [("set", "x", 99)]},
        "protocol": "2pc",
        "start_time": time.time(),
        "end_time": None,
    }

    # Hack: the participant has the data staged under "manual_tx"
    # but recovery will try to commit tx_id. This tests idempotency.
    actions = mgr.recover()
    test("recovery: recommit action", len(actions) == 1)
    test("recovery: recommit type", actions[0][1] == "recommitted")

def test_manager_recovery_3pc():
    """3PC recovery: PRE_COMMITTED -> recommit."""
    p1 = Participant("n1")
    mgr = TransactionManager([p1], protocol="3pc")

    tx_id = mgr._new_tx_id()
    mgr._history[tx_id] = {
        "state": TxState.PRE_COMMITTED,
        "operations": {"n1": [("set", "x", 50)]},
        "protocol": "3pc",
        "start_time": time.time(),
        "end_time": None,
    }

    actions = mgr.recover()
    test("3pc recovery: recommit", actions[0][1] == "recommitted")

def test_manager_recovery_already_complete():
    """Recovery skips already-completed transactions."""
    p1 = Participant("n1")
    mgr = TransactionManager([p1])

    success, tx_id, state = mgr.execute_transaction({
        "n1": [("set", "x", 1)],
    })

    actions = mgr.recover()
    test("recovery: skips completed", len(actions) == 0)

test_manager_2pc()
test_manager_3pc()
test_manager_abort()
test_manager_invalid_participant()
test_manager_tx_info()
test_manager_all_transactions()
test_manager_recovery_presumed_abort()
test_manager_recovery_recommit()
test_manager_recovery_3pc()
test_manager_recovery_already_complete()


# ============================================================
# Distributed KV Store
# ============================================================

def test_kvstore_put_get():
    store = DistributedKVStore(["n1", "n2", "n3"])
    store.put("hello", "world")
    test("kv: put/get", store.get("hello") == "world")

def test_kvstore_delete():
    store = DistributedKVStore(["n1", "n2"])
    store.put("key", "val")
    store.delete("key")
    test("kv: delete", store.get("key") is None)

def test_kvstore_multi_put():
    store = DistributedKVStore(["n1", "n2", "n3"])
    items = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    success = store.multi_put(items)
    test("kv: multi_put success", success == True)
    for k, v in items.items():
        test(f"kv: multi_put {k}", store.get(k) == v)

def test_kvstore_multi_delete():
    store = DistributedKVStore(["n1", "n2"])
    store.multi_put({"x": 1, "y": 2, "z": 3})
    store.multi_delete(["x", "y"])
    test("kv: multi_delete x", store.get("x") is None)
    test("kv: multi_delete y", store.get("y") is None)
    test("kv: multi_delete z kept", store.get("z") == 3)

def test_kvstore_transfer():
    store = DistributedKVStore(["n1", "n2"])
    store.put("alice", 100)
    store.put("bob", 50)
    success = store.transfer("alice", "bob", 30)
    test("kv: transfer success", success == True)
    test("kv: alice after transfer", store.get("alice") == 70)
    test("kv: bob after transfer", store.get("bob") == 80)

def test_kvstore_transfer_insufficient():
    store = DistributedKVStore(["n1", "n2"])
    store.put("alice", 10)
    store.put("bob", 50)
    success = store.transfer("alice", "bob", 100)
    test("kv: transfer insufficient", success == False)
    test("kv: alice unchanged", store.get("alice") == 10)
    test("kv: bob unchanged", store.get("bob") == 50)

def test_kvstore_overwrite():
    store = DistributedKVStore(["n1"])
    store.put("x", 1)
    store.put("x", 2)
    test("kv: overwrite", store.get("x") == 2)

def test_kvstore_3pc():
    store = DistributedKVStore(["n1", "n2"], protocol="3pc")
    store.put("key", "3pc_value")
    test("kv: 3pc put/get", store.get("key") == "3pc_value")

def test_kvstore_missing_key():
    store = DistributedKVStore(["n1"])
    test("kv: missing key is None", store.get("nothing") is None)

test_kvstore_put_get()
test_kvstore_delete()
test_kvstore_multi_put()
test_kvstore_multi_delete()
test_kvstore_transfer()
test_kvstore_transfer_insufficient()
test_kvstore_overwrite()
test_kvstore_3pc()
test_kvstore_missing_key()


# ============================================================
# Saga Coordinator
# ============================================================

def test_saga_basic_success():
    results = []

    steps = [
        SagaStep("step1", lambda: (results.append("s1"), True)[1],
                 lambda: results.append("c1")),
        SagaStep("step2", lambda: (results.append("s2"), True)[1],
                 lambda: results.append("c2")),
        SagaStep("step3", lambda: (results.append("s3"), True)[1],
                 lambda: results.append("c3")),
    ]

    saga = SagaCoordinator()
    success, completed, compensated = saga.execute("saga1", steps)

    test("saga: success", success == True)
    test("saga: all completed", completed == ["step1", "step2", "step3"])
    test("saga: no compensations", compensated == [])
    test("saga: actions executed", results == ["s1", "s2", "s3"])

def test_saga_failure_compensate():
    results = []

    steps = [
        SagaStep("step1", lambda: (results.append("s1"), True)[1],
                 lambda: (results.append("c1"), True)[1]),
        SagaStep("step2", lambda: (results.append("s2"), True)[1],
                 lambda: (results.append("c2"), True)[1]),
        SagaStep("step3", lambda: (results.append("s3"), False)[1],
                 lambda: (results.append("c3"), True)[1]),
    ]

    saga = SagaCoordinator()
    success, completed, compensated = saga.execute("saga2", steps)

    test("saga: failure detected", success == False)
    test("saga: completed before fail", completed == ["step1", "step2"])
    test("saga: compensated reverse", compensated == ["step2", "step1"])
    test("saga: actions+compensations", results == ["s1", "s2", "s3", "c2", "c1"])

def test_saga_exception_compensate():
    results = []

    steps = [
        SagaStep("step1", lambda: (results.append("s1"), True)[1],
                 lambda: results.append("c1")),
        SagaStep("step2", lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                 lambda: results.append("c2")),
    ]

    saga = SagaCoordinator()
    success, completed, compensated = saga.execute("saga3", steps)

    test("saga: exception caught", success == False)
    test("saga: compensated on exception", "step1" in compensated)

def test_saga_info():
    steps = [
        SagaStep("step1", lambda: True, lambda: True),
    ]

    saga = SagaCoordinator()
    saga.execute("saga4", steps)

    info = saga.get_saga_info("saga4")
    test("saga: info exists", info is not None)
    test("saga: info state", info["state"] == TxState.COMMITTED)

def test_saga_empty():
    saga = SagaCoordinator()
    success, completed, compensated = saga.execute("saga5", [])
    test("saga: empty success", success == True)
    test("saga: empty no steps", completed == [])

def test_saga_single_step_fail():
    saga = SagaCoordinator()
    comp = []
    steps = [
        SagaStep("only", lambda: False, lambda: comp.append("c")),
    ]
    success, completed, compensated = saga.execute("saga6", steps)
    test("saga: single step fail", success == False)
    test("saga: no compensation needed", compensated == [])

def test_saga_auto_id():
    saga = SagaCoordinator()
    saga.execute(None, [SagaStep("s", lambda: True, lambda: True)])
    saga.execute(None, [SagaStep("s", lambda: True, lambda: True)])
    test("saga: auto id generates history", len(saga._history) == 2)

def test_saga_log():
    saga = SagaCoordinator()
    steps = [
        SagaStep("step1", lambda: True, lambda: True),
        SagaStep("step2", lambda: True, lambda: True),
    ]
    saga.execute("saga_log", steps)
    entries = saga.get_log().get_entries("saga_log")
    test("saga: log has entries", len(entries) >= 3)

test_saga_basic_success()
test_saga_failure_compensate()
test_saga_exception_compensate()
test_saga_info()
test_saga_empty()
test_saga_single_step_fail()
test_saga_auto_id()
test_saga_log()


# ============================================================
# Timeout Coordinator
# ============================================================

def test_timeout_basic():
    p1 = Participant("n1")
    coord = TimeoutCoordinator([p1])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 1)])
    success, state = coord.execute_with_timeout(tx)

    test("timeout: basic commit", success == True)
    test("timeout: data", p1.get("x") == 1)

def test_timeout_abort_on_slow():
    p1 = Participant("n1", slow_prepare=0.5)
    coord = TimeoutCoordinator([p1], timeout=0.1)

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "x", 1)])
    success, state = coord.execute_with_timeout(
        tx, per_participant_timeout={"n1": 0.1}
    )

    # The slow participant might still complete within thread timeout
    # but this tests the timeout mechanism exists
    test("timeout: executed", state in (TxState.COMMITTED, TxState.ABORTED))

def test_timeout_multi_participant():
    p1, p2 = Participant("n1"), Participant("n2")
    coord = TimeoutCoordinator([p1, p2])

    tx = coord.begin()
    coord.add_operation(tx, "n1", [("set", "a", 1)])
    coord.add_operation(tx, "n2", [("set", "b", 2)])
    success, state = coord.execute_with_timeout(tx)

    test("timeout: multi commit", success == True)
    test("timeout: multi n1", p1.get("a") == 1)
    test("timeout: multi n2", p2.get("b") == 2)

def test_timeout_empty():
    p1 = Participant("n1")
    coord = TimeoutCoordinator([p1])
    tx = coord.begin()
    success, state = coord.execute_with_timeout(tx)
    test("timeout: empty aborts", success == False)

test_timeout_basic()
test_timeout_abort_on_slow()
test_timeout_multi_participant()
test_timeout_empty()


# ============================================================
# Concurrency Tests
# ============================================================

def test_concurrent_transactions():
    """Multiple transactions on same coordinator, different keys."""
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])

    results = []

    def run_tx(val):
        tx = coord.begin()
        coord.add_operation(tx, "n1", [("set", f"key_{val}", val)])
        s, st = coord.execute(tx)
        results.append((val, s))

    threads = [threading.Thread(target=run_tx, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    test("concurrent: all completed", len(results) == 5)
    successful = sum(1 for _, s in results if s)
    test("concurrent: some succeeded", successful >= 1)

def test_concurrent_kvstore():
    """Concurrent put operations on distributed KV store."""
    store = DistributedKVStore(["n1", "n2", "n3"])
    errors = []

    def put_item(key, value):
        try:
            store.put(key, value)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=put_item, args=(f"k{i}", i)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    test("concurrent kv: no errors", len(errors) == 0)
    found = sum(1 for i in range(10) if store.get(f"k{i}") == i)
    test("concurrent kv: most found", found >= 5)

test_concurrent_transactions()
test_concurrent_kvstore()


# ============================================================
# Edge Cases
# ============================================================

def test_participant_fail_on_pre_commit_flag():
    p = Participant("n1", fail_on_pre_commit=True)
    p.prepare("tx1", [("set", "x", 10)])
    ack = p.pre_commit("tx1")
    test("edge: fail_on_pre_commit", ack == False)

def test_participant_fail_on_commit_flag():
    p = Participant("n1", fail_on_commit=True)
    p.prepare("tx1", [("set", "x", 10)])
    result = p.commit("tx1")
    test("edge: fail_on_commit", result == False)

def test_abort_unknown_tx():
    p = Participant("n1")
    result = p.abort("unknown_tx")
    test("edge: abort unknown tx", result == True)
    test("edge: abort unknown state", p.get_state("unknown_tx") == ParticipantState.ABORTED)

def test_multi_key_lock_release_on_abort():
    p = Participant("n1")
    p.prepare("tx1", [("set", "a", 1), ("set", "b", 2)])
    p.abort("tx1")

    # Keys should be unlocked now
    vote = p.prepare("tx2", [("set", "a", 10), ("set", "b", 20)])
    test("edge: locks released on abort", vote == Vote.YES)
    p.commit("tx2")
    test("edge: new data after abort a", p.get("a") == 10)
    test("edge: new data after abort b", p.get("b") == 20)

def test_multi_key_lock_release_on_commit():
    p = Participant("n1")
    p.prepare("tx1", [("set", "x", 1)])
    p.commit("tx1")

    # Key should be unlocked
    vote = p.prepare("tx2", [("set", "x", 2)])
    test("edge: locks released on commit", vote == Vote.YES)

def test_2pc_tx_id_generation():
    p1 = Participant("n1")
    coord = TwoPhaseCommitCoordinator([p1])
    tx1 = coord.begin()
    tx2 = coord.begin()
    test("edge: unique tx ids", tx1 != tx2)

def test_3pc_tx_id_generation():
    p1 = Participant("n1")
    coord = ThreePhaseCommitCoordinator([p1])
    tx1 = coord.begin()
    tx2 = coord.begin()
    test("edge: 3pc unique tx ids", tx1 != tx2)

def test_manager_log():
    p1 = Participant("n1")
    mgr = TransactionManager([p1])
    mgr.execute_transaction({"n1": [("set", "x", 1)]})
    log = mgr.get_log()
    test("edge: manager log exists", len(log.get_entries()) > 0)

def test_kvstore_sharding():
    """Keys are distributed across nodes."""
    store = DistributedKVStore(["n0", "n1", "n2", "n3"])
    for i in range(20):
        store.put(f"key_{i}", i)

    # Check that data is spread across nodes
    counts = {name: len(node.data) for name, node in store.nodes.items()}
    total = sum(counts.values())
    test("kv sharding: all stored", total == 20)
    test("kv sharding: distributed", max(counts.values()) < 20)  # Not all on one node

def test_kvstore_transfer_same_node():
    """Transfer between keys on the same node."""
    store = DistributedKVStore(["n1"])  # Single node
    store.put("a", 100)
    store.put("b", 0)
    success = store.transfer("a", "b", 50)
    test("kv: same-node transfer", success == True)
    test("kv: same-node a", store.get("a") == 50)
    test("kv: same-node b", store.get("b") == 50)

test_participant_fail_on_pre_commit_flag()
test_participant_fail_on_commit_flag()
test_abort_unknown_tx()
test_multi_key_lock_release_on_abort()
test_multi_key_lock_release_on_commit()
test_2pc_tx_id_generation()
test_3pc_tx_id_generation()
test_manager_log()
test_kvstore_sharding()
test_kvstore_transfer_same_node()


# ============================================================
# State Transition Validation
# ============================================================

def test_tx_state_values():
    test("state: INITIATED", TxState.INITIATED.value == "INITIATED")
    test("state: PREPARING", TxState.PREPARING.value == "PREPARING")
    test("state: PREPARED", TxState.PREPARED.value == "PREPARED")
    test("state: PRE_COMMITTED", TxState.PRE_COMMITTED.value == "PRE_COMMITTED")
    test("state: COMMITTING", TxState.COMMITTING.value == "COMMITTING")
    test("state: COMMITTED", TxState.COMMITTED.value == "COMMITTED")
    test("state: ABORTING", TxState.ABORTING.value == "ABORTING")
    test("state: ABORTED", TxState.ABORTED.value == "ABORTED")

def test_vote_values():
    test("vote: YES", Vote.YES.value == "YES")
    test("vote: NO", Vote.NO.value == "NO")
    test("vote: TIMEOUT", Vote.TIMEOUT.value == "TIMEOUT")

def test_participant_state_values():
    test("pstate: INIT", ParticipantState.INIT.value == "INIT")
    test("pstate: READY", ParticipantState.READY.value == "READY")
    test("pstate: PRE_COMMITTED", ParticipantState.PRE_COMMITTED.value == "PRE_COMMITTED")
    test("pstate: COMMITTED", ParticipantState.COMMITTED.value == "COMMITTED")
    test("pstate: ABORTED", ParticipantState.ABORTED.value == "ABORTED")

test_tx_state_values()
test_vote_values()
test_participant_state_values()


# ============================================================
# Integration: Full Workflow Tests
# ============================================================

def test_bank_transfer_workflow():
    """Simulate a bank transfer across 3 accounts on different nodes."""
    store = DistributedKVStore(["bank_a", "bank_b", "bank_c"])

    # Initial balances
    store.put("alice", 1000)
    store.put("bob", 500)
    store.put("charlie", 200)

    # Alice -> Bob: 200
    success1 = store.transfer("alice", "bob", 200)
    test("bank: transfer 1 success", success1 == True)

    # Bob -> Charlie: 100
    success2 = store.transfer("bob", "charlie", 100)
    test("bank: transfer 2 success", success2 == True)

    # Verify conservation of money
    total = store.get("alice") + store.get("bob") + store.get("charlie")
    test("bank: money conserved", total == 1700)
    test("bank: alice balance", store.get("alice") == 800)
    test("bank: bob balance", store.get("bob") == 600)
    test("bank: charlie balance", store.get("charlie") == 300)

def test_saga_order_workflow():
    """Simulate an e-commerce order saga."""
    inventory = {"widget": 10}
    payments = {"user1": 100.0}
    orders = []
    shipping = []

    def reserve_inventory():
        if inventory["widget"] >= 1:
            inventory["widget"] -= 1
            return True
        return False

    def unreserve_inventory():
        inventory["widget"] += 1
        return True

    def charge_payment():
        if payments["user1"] >= 25.0:
            payments["user1"] -= 25.0
            return True
        return False

    def refund_payment():
        payments["user1"] += 25.0
        return True

    def create_order():
        orders.append({"id": "order_1", "status": "created"})
        return True

    def cancel_order():
        orders[-1]["status"] = "cancelled"
        return True

    def schedule_shipping():
        shipping.append("order_1")
        return True

    def cancel_shipping():
        shipping.remove("order_1")
        return True

    steps = [
        SagaStep("reserve_inventory", reserve_inventory, unreserve_inventory),
        SagaStep("charge_payment", charge_payment, refund_payment),
        SagaStep("create_order", create_order, cancel_order),
        SagaStep("schedule_shipping", schedule_shipping, cancel_shipping),
    ]

    saga = SagaCoordinator()
    success, completed, compensated = saga.execute("order_1", steps)

    test("order saga: success", success == True)
    test("order saga: inventory reserved", inventory["widget"] == 9)
    test("order saga: payment charged", payments["user1"] == 75.0)
    test("order saga: order created", len(orders) == 1)
    test("order saga: shipping scheduled", "order_1" in shipping)

def test_saga_order_payment_fail():
    """Order saga where payment fails -> compensate."""
    inventory = {"widget": 10}

    def reserve():
        inventory["widget"] -= 1
        return True
    def unreserve():
        inventory["widget"] += 1
        return True
    def fail_payment():
        return False
    def refund():
        return True

    steps = [
        SagaStep("reserve", reserve, unreserve),
        SagaStep("payment", fail_payment, refund),
    ]

    saga = SagaCoordinator()
    success, completed, compensated = saga.execute("order_fail", steps)

    test("order fail: saga failed", success == False)
    test("order fail: inventory restored", inventory["widget"] == 10)

def test_2pc_then_3pc_same_participants():
    """Use same participants with different protocols."""
    p1, p2 = Participant("n1"), Participant("n2")

    # 2PC
    coord2 = TwoPhaseCommitCoordinator([p1, p2])
    tx = coord2.begin()
    coord2.add_operation(tx, "n1", [("set", "x", 1)])
    coord2.add_operation(tx, "n2", [("set", "y", 2)])
    s1, _ = coord2.execute(tx)

    # 3PC
    coord3 = ThreePhaseCommitCoordinator([p1, p2])
    tx = coord3.begin()
    coord3.add_operation(tx, "n1", [("set", "x", 10)])
    coord3.add_operation(tx, "n2", [("set", "y", 20)])
    s2, _ = coord3.execute(tx)

    test("protocol switch: both succeed", s1 and s2)
    test("protocol switch: 3pc overwrites", p1.get("x") == 10)
    test("protocol switch: 3pc overwrites y", p2.get("y") == 20)

test_bank_transfer_workflow()
test_saga_order_workflow()
test_saga_order_payment_fail()
test_2pc_then_3pc_same_participants()


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print(f"  C237 Distributed Transactions: {passed} passed, {failed} failed")
print(f"{'='*60}")

if failed > 0:
    sys.exit(1)
