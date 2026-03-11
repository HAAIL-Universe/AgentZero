"""Tests for C171: Meta-Learning"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from meta_learning import (
    Task, TaskDistribution, MAML, Reptile, PrototypicalNetwork,
    MatchingNetwork, MetaTrainer, FewShotClassifier, TaskAugmenter,
    MetaScheduler, make_few_shot_data, _clone_model, _get_flat_params,
    _set_flat_params, _param_subtract, _param_add, _param_scale,
    _param_zeros_like, _inner_train_step, _euclidean_distance,
    _cosine_similarity
)
from neural_network import (
    Tensor, Dense, Activation, Sequential, CrossEntropyLoss, MSELoss,
    SGD, Adam, build_model, save_weights, load_weights
)


# ============================================================
# Task and TaskDistribution
# ============================================================

def test_task_creation():
    sx = Tensor([[1, 2], [3, 4]])
    sy = [0, 1]
    qx = Tensor([[5, 6]])
    qy = [0]
    t = Task(sx, sy, qx, qy, [10, 20])
    assert t.n_way == 2
    assert t.k_shot == 1
    assert t.classes == [10, 20]
    assert len(t.support_y) == 2
    assert len(t.query_y) == 1

def test_task_distribution_creation():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    assert td.num_classes == 5
    assert td.feat_dim == 4

def test_task_distribution_sample():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(n_way=3, k_shot=2, q_queries=1)
    assert task.n_way == 3
    assert task.k_shot == 2
    assert len(task.support_y) == 6  # 3 * 2
    assert len(task.query_y) == 3    # 3 * 1
    assert task.support_x.shape == (6, 4)
    assert task.query_x.shape == (3, 4)

def test_task_distribution_multiple_queries():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(n_way=2, k_shot=3, q_queries=2)
    assert len(task.support_y) == 6  # 2 * 3
    assert len(task.query_y) == 4    # 2 * 2

def test_task_distribution_sample_tasks():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    tasks = td.sample_tasks(n_tasks=5, n_way=2, k_shot=2)
    assert len(tasks) == 5
    for t in tasks:
        assert t.n_way == 2

def test_task_distribution_labels_remapped():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(n_way=3, k_shot=2, q_queries=1)
    # Labels should be 0, 1, 2 (remapped)
    assert set(task.support_y).issubset({0, 1, 2})
    assert set(task.query_y).issubset({0, 1, 2})

def test_task_distribution_n_way_exceeds_classes():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    try:
        td.sample_task(n_way=5, k_shot=2)
        assert False, "Should raise"
    except ValueError:
        pass

def test_task_distribution_list_input():
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    Y = [0, 0, 1, 1]
    td = TaskDistribution(X, Y)
    assert td.num_classes == 2
    task = td.sample_task(n_way=2, k_shot=1, q_queries=1)
    assert task.n_way == 2

def test_make_few_shot_data():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=6)
    assert X.shape == (75, 6)
    assert len(Y) == 75
    assert len(set(Y)) == 5


# ============================================================
# Parameter utilities
# ============================================================

def test_clone_model():
    model = build_model([4, 8, 3])
    clone = _clone_model(model)
    p1 = _get_flat_params(model)
    p2 = _get_flat_params(clone)
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert abs(a - b) < 1e-10

def test_get_set_flat_params():
    model = build_model([4, 8, 3])
    params = _get_flat_params(model)
    n = len(params)
    assert n > 0
    new_params = [0.5] * n
    _set_flat_params(model, new_params)
    retrieved = _get_flat_params(model)
    for a, b in zip(retrieved, new_params):
        assert abs(a - b) < 1e-10

def test_param_arithmetic():
    a = [1.0, 2.0, 3.0]
    b = [0.5, 1.0, 1.5]
    assert _param_subtract(a, b) == [0.5, 1.0, 1.5]
    assert _param_add(a, b) == [1.5, 3.0, 4.5]
    assert _param_scale(a, 2.0) == [2.0, 4.0, 6.0]
    assert _param_zeros_like(a) == [0.0, 0.0, 0.0]

def test_inner_train_step():
    model = build_model([4, 8, 3])
    X = Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    Y = [0, 1, 2]
    loss_fn = CrossEntropyLoss()
    params_before = _get_flat_params(model)
    loss = _inner_train_step(model, X, Y, loss_fn, 0.01)
    params_after = _get_flat_params(model)
    assert loss > 0
    # Params should have changed
    diffs = sum(1 for a, b in zip(params_before, params_after) if abs(a - b) > 1e-12)
    assert diffs > 0


# ============================================================
# Distance functions
# ============================================================

def test_euclidean_distance():
    a = Tensor([1, 0, 0])
    b = Tensor([0, 1, 0])
    d = _euclidean_distance(a, b)
    assert abs(d - 2.0) < 1e-10

def test_euclidean_distance_same():
    a = Tensor([1, 2, 3])
    d = _euclidean_distance(a, a)
    assert abs(d) < 1e-10

def test_cosine_similarity():
    a = Tensor([1, 0, 0])
    b = Tensor([1, 0, 0])
    assert abs(_cosine_similarity(a, b) - 1.0) < 1e-10

def test_cosine_similarity_orthogonal():
    a = Tensor([1, 0])
    b = Tensor([0, 1])
    assert abs(_cosine_similarity(a, b)) < 1e-10


# ============================================================
# MAML
# ============================================================

def test_maml_creation():
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1)
    assert maml.inner_lr == 0.01
    assert maml.outer_lr == 0.001
    assert maml.inner_steps == 1
    assert not maml.first_order

def test_maml_adapt():
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, inner_steps=2)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(n_way=3, k_shot=3, q_queries=2)
    adapted = maml.adapt(task)
    # Adapted model should differ from original
    p1 = _get_flat_params(model)
    p2 = _get_flat_params(adapted)
    diffs = sum(1 for a, b in zip(p1, p2) if abs(a - b) > 1e-12)
    assert diffs > 0

def test_maml_meta_train_step():
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1,
                first_order=True)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    tasks = td.sample_tasks(2, n_way=3, k_shot=2, q_queries=1)
    params_before = _get_flat_params(model)
    loss = maml.meta_train_step(tasks)
    params_after = _get_flat_params(model)
    assert loss > 0
    diffs = sum(1 for a, b in zip(params_before, params_after) if abs(a - b) > 1e-12)
    assert diffs > 0

def test_maml_fomaml_training():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=1)
    td = TaskDistribution(X, Y, seed=1)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=3,
                first_order=True)
    history = maml.meta_train(td, n_way=3, k_shot=3, q_queries=2,
                               meta_epochs=10, tasks_per_epoch=2)
    assert 'meta_loss' in history
    assert len(history['meta_loss']) == 10

def test_maml_second_order():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4, seed=2)
    td = TaskDistribution(X, Y, seed=2)
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=1,
                first_order=False)
    loss = maml.meta_train_step(td.sample_tasks(2, 3, 2, 1))
    assert loss > 0

def test_maml_evaluate():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=3)
    td = TaskDistribution(X, Y, seed=3)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=3,
                first_order=True)
    maml.meta_train(td, n_way=3, k_shot=3, q_queries=2,
                     meta_epochs=10, tasks_per_epoch=2)
    result = maml.evaluate(td, n_way=3, k_shot=3, q_queries=2, n_tasks=5)
    assert 'accuracy' in result
    assert 'loss' in result
    assert 'per_task_accuracy' in result
    assert 0.0 <= result['accuracy'] <= 1.0
    assert len(result['per_task_accuracy']) == 5

def test_maml_history():
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    maml.meta_train(td, n_way=3, k_shot=2, meta_epochs=5, tasks_per_epoch=2)
    assert len(maml._history['meta_loss']) == 5
    assert len(maml._history['meta_accuracy']) == 5

def test_maml_adapt_more_steps():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4, seed=4)
    td = TaskDistribution(X, Y, seed=4)
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, inner_steps=1)
    task = td.sample_task(3, 2, 1)
    adapted_1 = maml.adapt(task, steps=1)
    adapted_5 = maml.adapt(task, steps=5)
    p1 = _get_flat_params(adapted_1)
    p5 = _get_flat_params(adapted_5)
    # More steps -> more divergence from original
    diff1 = sum(abs(a - b) for a, b in zip(_get_flat_params(model), p1))
    diff5 = sum(abs(a - b) for a, b in zip(_get_flat_params(model), p5))
    assert diff5 > diff1


# ============================================================
# Reptile
# ============================================================

def test_reptile_creation():
    model = build_model([4, 8, 3])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=5)
    assert reptile.inner_lr == 0.01
    assert reptile.outer_lr == 0.1
    assert reptile.inner_steps == 5

def test_reptile_adapt():
    model = build_model([4, 8, 3])
    reptile = Reptile(model, inner_lr=0.01, inner_steps=3)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 2, 1)
    adapted = reptile.adapt(task)
    p1 = _get_flat_params(model)
    p2 = _get_flat_params(adapted)
    diffs = sum(1 for a, b in zip(p1, p2) if abs(a - b) > 1e-12)
    assert diffs > 0

def test_reptile_meta_train_step():
    model = build_model([4, 8, 3])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=3)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    tasks = td.sample_tasks(2, 3, 2, 1)
    params_before = _get_flat_params(model)
    loss = reptile.meta_train_step(tasks)
    params_after = _get_flat_params(model)
    assert loss > 0
    diffs = sum(1 for a, b in zip(params_before, params_after) if abs(a - b) > 1e-12)
    assert diffs > 0

def test_reptile_training():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=5)
    td = TaskDistribution(X, Y, seed=5)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=5)
    history = reptile.meta_train(td, n_way=3, k_shot=3, q_queries=2,
                                  meta_epochs=10, tasks_per_epoch=2)
    assert 'meta_loss' in history
    assert len(history['meta_loss']) == 10

def test_reptile_evaluate():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=6)
    td = TaskDistribution(X, Y, seed=6)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=5)
    reptile.meta_train(td, n_way=3, k_shot=3, meta_epochs=10, tasks_per_epoch=2)
    result = reptile.evaluate(td, n_way=3, k_shot=3, n_tasks=5)
    assert 0.0 <= result['accuracy'] <= 1.0
    assert len(result['per_task_accuracy']) == 5

def test_reptile_moves_toward_adapted():
    model = build_model([4, 8, 3])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=1.0, inner_steps=3)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4, seed=7)
    td = TaskDistribution(X, Y, seed=7)
    params_before = _get_flat_params(model)
    task = td.sample_task(3, 2, 1)
    adapted = reptile.adapt(task)
    adapted_params = _get_flat_params(adapted)
    # With outer_lr=1.0, meta update should move fully toward adapted
    reptile.meta_train_step([task])
    params_after = _get_flat_params(model)
    # Should be closer to adapted than before
    dist_before = sum((a - b)**2 for a, b in zip(params_before, adapted_params))
    dist_after = sum((a - b)**2 for a, b in zip(params_after, adapted_params))
    assert dist_after < dist_before


# ============================================================
# Prototypical Networks
# ============================================================

def test_protonet_creation():
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder, distance='euclidean', lr=0.001)
    assert pn.distance == 'euclidean'
    assert pn.lr == 0.001

def test_protonet_compute_prototypes():
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 1)
    protos = pn.compute_prototypes(task.support_x, task.support_y, 3)
    assert len(protos) == 3
    for p in protos:
        assert len(p.data) == 4  # embed_dim matches output

def test_protonet_classify():
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 2)
    protos = pn.compute_prototypes(task.support_x, task.support_y, 3)
    preds, log_probs = pn.classify(task.query_x, protos)
    assert len(preds) == 6  # 3 * 2
    assert all(p in [0, 1, 2] for p in preds)

def test_protonet_episode_loss():
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 2)
    loss, acc = pn.episode_loss(task)
    assert loss > 0
    assert 0.0 <= acc <= 1.0

def test_protonet_train_step():
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder, lr=0.01)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 2)
    params_before = _get_flat_params(encoder)
    loss = pn.train_step(task)
    params_after = _get_flat_params(encoder)
    assert loss > 0
    diffs = sum(1 for a, b in zip(params_before, params_after) if abs(a - b) > 1e-12)
    assert diffs > 0

def test_protonet_training():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=8)
    td = TaskDistribution(X, Y, seed=8)
    encoder = build_model([4, 16, 8])
    pn = PrototypicalNetwork(encoder, lr=0.005)
    history = pn.meta_train(td, n_way=3, k_shot=3, q_queries=2, episodes=20)
    assert 'loss' in history
    assert len(history['loss']) == 20

def test_protonet_evaluate():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=9)
    td = TaskDistribution(X, Y, seed=9)
    encoder = build_model([4, 16, 8])
    pn = PrototypicalNetwork(encoder, lr=0.005)
    pn.meta_train(td, n_way=3, k_shot=3, episodes=20)
    result = pn.evaluate(td, n_way=3, k_shot=3, n_tasks=5)
    assert 0.0 <= result['accuracy'] <= 1.0

def test_protonet_cosine_distance():
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder, distance='cosine')
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 2)
    loss, acc = pn.episode_loss(task)
    assert loss > 0


# ============================================================
# Matching Networks
# ============================================================

def test_matching_net_creation():
    encoder = build_model([4, 8, 4])
    mn = MatchingNetwork(encoder, lr=0.001)
    assert mn.lr == 0.001

def test_matching_net_classify():
    encoder = build_model([4, 8, 4])
    mn = MatchingNetwork(encoder)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 2)
    loss, acc = mn.episode_loss(task)
    assert loss > 0
    assert 0.0 <= acc <= 1.0

def test_matching_net_train_step():
    encoder = build_model([4, 8, 4])
    mn = MatchingNetwork(encoder, lr=0.01)
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 2)
    params_before = _get_flat_params(encoder)
    loss = mn.train_step(task)
    params_after = _get_flat_params(encoder)
    assert loss > 0
    diffs = sum(1 for a, b in zip(params_before, params_after) if abs(a - b) > 1e-12)
    assert diffs > 0

def test_matching_net_training():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=10)
    td = TaskDistribution(X, Y, seed=10)
    encoder = build_model([4, 16, 8])
    mn = MatchingNetwork(encoder, lr=0.005)
    history = mn.meta_train(td, n_way=3, k_shot=3, q_queries=2, episodes=20)
    assert 'loss' in history
    assert len(history['loss']) == 20

def test_matching_net_evaluate():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=11)
    td = TaskDistribution(X, Y, seed=11)
    encoder = build_model([4, 16, 8])
    mn = MatchingNetwork(encoder, lr=0.005)
    mn.meta_train(td, n_way=3, k_shot=3, episodes=20)
    result = mn.evaluate(td, n_way=3, k_shot=3, n_tasks=5)
    assert 0.0 <= result['accuracy'] <= 1.0
    assert len(result['per_task_accuracy']) == 5


# ============================================================
# MetaTrainer
# ============================================================

def test_meta_trainer_maml():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=12)
    td = TaskDistribution(X, Y, seed=12)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, first_order=True)
    trainer = MetaTrainer(maml, td)
    history = trainer.train(n_way=3, k_shot=3, epochs=5, tasks_per_epoch=2)
    assert 'meta_loss' in history

def test_meta_trainer_reptile():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=13)
    td = TaskDistribution(X, Y, seed=13)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1)
    trainer = MetaTrainer(reptile, td)
    history = trainer.train(n_way=3, k_shot=3, epochs=5, tasks_per_epoch=2)
    assert 'meta_loss' in history

def test_meta_trainer_protonet():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=14)
    td = TaskDistribution(X, Y, seed=14)
    encoder = build_model([4, 16, 8])
    pn = PrototypicalNetwork(encoder, lr=0.005)
    trainer = MetaTrainer(pn, td)
    history = trainer.train(n_way=3, k_shot=3, epochs=10)
    assert 'loss' in history

def test_meta_trainer_evaluate():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=15)
    td = TaskDistribution(X, Y, seed=15)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=3)
    trainer = MetaTrainer(reptile, td)
    trainer.train(n_way=3, k_shot=3, epochs=5, tasks_per_epoch=2)
    result = trainer.evaluate(n_way=3, k_shot=3, n_tasks=5)
    assert 'accuracy' in result

def test_meta_trainer_compare():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=16)
    td = TaskDistribution(X, Y, seed=16)
    m1 = build_model([4, 16, 5])
    m2 = build_model([4, 16, 5])
    algos = {
        'fomaml': MAML(m1, inner_lr=0.01, outer_lr=0.001, first_order=True),
        'reptile': Reptile(m2, inner_lr=0.01, outer_lr=0.1)
    }
    trainer = MetaTrainer(algos['fomaml'], td)
    results = trainer.compare_algorithms(algos, n_way=3, k_shot=3,
                                          train_epochs=5, eval_tasks=5)
    assert 'fomaml' in results
    assert 'reptile' in results
    assert 'accuracy' in results['fomaml']


# ============================================================
# FewShotClassifier
# ============================================================

def test_few_shot_classifier_maml():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=17)
    td = TaskDistribution(X, Y, seed=17)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, first_order=True)
    maml.meta_train(td, n_way=3, k_shot=3, meta_epochs=5, tasks_per_epoch=2)

    fsc = FewShotClassifier(maml)
    task = td.sample_task(3, 3, 2)
    fsc.fit(task.support_x, task.support_y)
    preds = fsc.predict(task.query_x)
    assert len(preds) == 6
    # Model has 5 outputs, so preds can be 0-4 (MAML uses full model)
    assert all(0 <= p < 5 for p in preds)

def test_few_shot_classifier_reptile():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=18)
    td = TaskDistribution(X, Y, seed=18)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=3)
    reptile.meta_train(td, n_way=3, k_shot=3, meta_epochs=5, tasks_per_epoch=2)

    fsc = FewShotClassifier(reptile)
    task = td.sample_task(3, 3, 2)
    fsc.fit(task.support_x, task.support_y)
    preds = fsc.predict(task.query_x)
    assert len(preds) == 6

def test_few_shot_classifier_protonet():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=19)
    td = TaskDistribution(X, Y, seed=19)
    encoder = build_model([4, 16, 8])
    pn = PrototypicalNetwork(encoder, lr=0.005)
    pn.meta_train(td, n_way=3, k_shot=3, episodes=10)

    fsc = FewShotClassifier(pn)
    task = td.sample_task(3, 3, 2)
    fsc.fit(task.support_x, task.support_y)
    preds = fsc.predict(task.query_x)
    assert len(preds) == 6

def test_few_shot_classifier_matching():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=20)
    td = TaskDistribution(X, Y, seed=20)
    encoder = build_model([4, 16, 8])
    mn = MatchingNetwork(encoder, lr=0.005)
    mn.meta_train(td, n_way=3, k_shot=3, episodes=10)

    fsc = FewShotClassifier(mn)
    task = td.sample_task(3, 3, 2)
    fsc.fit(task.support_x, task.support_y)
    preds = fsc.predict(task.query_x)
    assert len(preds) == 6

def test_few_shot_classifier_predict_proba_maml():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4, seed=21)
    td = TaskDistribution(X, Y, seed=21)
    model = build_model([4, 8, 3])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, first_order=True)
    maml.meta_train(td, n_way=3, k_shot=2, meta_epochs=5, tasks_per_epoch=2)

    fsc = FewShotClassifier(maml)
    task = td.sample_task(3, 2, 1)
    fsc.fit(task.support_x, task.support_y)
    probs = fsc.predict_proba(task.query_x)
    assert len(probs) == 3
    for p in probs:
        assert abs(sum(p) - 1.0) < 1e-6

def test_few_shot_classifier_predict_proba_protonet():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4, seed=22)
    td = TaskDistribution(X, Y, seed=22)
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder, lr=0.005)
    pn.meta_train(td, n_way=3, k_shot=2, episodes=10)

    fsc = FewShotClassifier(pn)
    task = td.sample_task(3, 2, 1)
    fsc.fit(task.support_x, task.support_y)
    probs = fsc.predict_proba(task.query_x)
    assert len(probs) == 3
    for p in probs:
        assert abs(sum(p) - 1.0) < 1e-6

def test_few_shot_classifier_predict_proba_matching():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4, seed=23)
    td = TaskDistribution(X, Y, seed=23)
    encoder = build_model([4, 8, 4])
    mn = MatchingNetwork(encoder, lr=0.005)
    mn.meta_train(td, n_way=3, k_shot=2, episodes=10)

    fsc = FewShotClassifier(mn)
    task = td.sample_task(3, 2, 1)
    fsc.fit(task.support_x, task.support_y)
    probs = fsc.predict_proba(task.query_x)
    assert len(probs) == 3
    for p in probs:
        assert abs(sum(p) - 1.0) < 1e-6

def test_few_shot_infer_n_way():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=24)
    td = TaskDistribution(X, Y, seed=24)
    encoder = build_model([4, 16, 8])
    pn = PrototypicalNetwork(encoder)
    fsc = FewShotClassifier(pn)
    task = td.sample_task(4, 2, 1)
    fsc.fit(task.support_x, task.support_y)  # n_way inferred from labels
    assert fsc._n_way == 4


# ============================================================
# TaskAugmenter
# ============================================================

def test_task_augmenter_noise():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 2, 1)
    aug = TaskAugmenter()
    augmented = aug.add_noise(task, std=0.1)
    assert len(augmented.support_y) == 12  # doubled
    assert augmented.support_x.shape[0] == 12
    # Query set unchanged
    assert len(augmented.query_y) == len(task.query_y)

def test_task_augmenter_mixup():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 3, 1)
    aug = TaskAugmenter()
    augmented = aug.mixup_support(task, alpha=0.2, n_extra=6)
    assert len(augmented.support_y) >= len(task.support_y)

def test_task_augmenter_scale():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 2, 1)
    aug = TaskAugmenter()
    scaled = aug.random_scale(task, low=0.9, high=1.1)
    assert scaled.support_x.shape == task.support_x.shape
    # Values should differ
    diffs = 0
    for i in range(len(task.support_y)):
        for j in range(4):
            if abs(scaled.support_x.data[i][j] - task.support_x.data[i][j]) > 1e-10:
                diffs += 1
    assert diffs > 0

def test_task_augmenter_preserves_query():
    X, Y = make_few_shot_data(n_classes=3, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    task = td.sample_task(3, 2, 1)
    aug = TaskAugmenter()
    augmented = aug.add_noise(task)
    # Query unchanged
    for i in range(len(task.query_y)):
        for j in range(4):
            assert abs(augmented.query_x.data[i][j] - task.query_x.data[i][j]) < 1e-10


# ============================================================
# MetaScheduler
# ============================================================

def test_meta_scheduler_cosine():
    model = build_model([4, 8, 3])
    maml = MAML(model, outer_lr=0.01)
    sched = MetaScheduler(maml, schedule='cosine', T_max=100, eta_min=0.0001)
    lrs = []
    for _ in range(50):
        lr = sched.step()
        lrs.append(lr)
    assert lrs[0] > lrs[-1]
    assert all(lr >= 0.0001 for lr in lrs)

def test_meta_scheduler_step():
    model = build_model([4, 8, 3])
    reptile = Reptile(model, outer_lr=0.1)
    sched = MetaScheduler(reptile, schedule='step', T_max=30)
    lr_start = sched.get_lr()
    for _ in range(15):
        sched.step()
    lr_mid = sched.get_lr()
    assert lr_mid < lr_start

def test_meta_scheduler_linear():
    model = build_model([4, 8, 3])
    maml = MAML(model, outer_lr=0.01)
    sched = MetaScheduler(maml, schedule='linear', T_max=100, eta_min=0.001)
    lrs = []
    for _ in range(100):
        lr = sched.step()
        lrs.append(lr)
    assert lrs[-1] >= 0.001
    assert lrs[0] > lrs[-1]

def test_meta_scheduler_get_lr():
    model = build_model([4, 8, 3])
    maml = MAML(model, outer_lr=0.01)
    sched = MetaScheduler(maml, schedule='cosine', T_max=50)
    assert sched.get_lr() == 0.01
    sched.step()
    assert sched.get_lr() < 0.01


# ============================================================
# Integration tests
# ============================================================

def test_maml_with_scheduler():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=25)
    td = TaskDistribution(X, Y, seed=25)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.01, first_order=True)
    sched = MetaScheduler(maml, schedule='cosine', T_max=10)
    for _ in range(10):
        tasks = td.sample_tasks(2, 3, 3, 1)
        maml.meta_train_step(tasks)
        sched.step()
    assert maml.outer_lr < 0.01

def test_augmented_meta_training():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=26)
    td = TaskDistribution(X, Y, seed=26)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=3)
    aug = TaskAugmenter()
    for _ in range(5):
        tasks = td.sample_tasks(2, 3, 3, 1)
        augmented_tasks = [aug.add_noise(t, std=0.05) for t in tasks]
        reptile.meta_train_step(augmented_tasks)
    # Should train without error
    result = reptile.evaluate(td, n_way=3, k_shot=3, n_tasks=3)
    assert 'accuracy' in result

def test_end_to_end_maml_pipeline():
    """Full pipeline: generate data, train MAML, adapt, classify."""
    X, Y = make_few_shot_data(n_classes=8, samples_per_class=20, feat_dim=6, seed=27)
    td = TaskDistribution(X, Y, seed=27)
    model = build_model([6, 32, 8])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=3,
                first_order=True)
    maml.meta_train(td, n_way=4, k_shot=3, q_queries=2,
                     meta_epochs=15, tasks_per_epoch=3)
    fsc = FewShotClassifier(maml)
    task = td.sample_task(4, 3, 2)
    fsc.fit(task.support_x, task.support_y, adaptation_steps=5)
    preds = fsc.predict(task.query_x)
    assert len(preds) == 8
    probs = fsc.predict_proba(task.query_x)
    assert len(probs) == 8
    for p in probs:
        assert abs(sum(p) - 1.0) < 1e-5

def test_end_to_end_protonet_pipeline():
    """Full pipeline: generate data, train ProtoNet, classify."""
    X, Y = make_few_shot_data(n_classes=8, samples_per_class=20, feat_dim=6, seed=28)
    td = TaskDistribution(X, Y, seed=28)
    encoder = build_model([6, 32, 16])
    pn = PrototypicalNetwork(encoder, distance='euclidean', lr=0.005)
    pn.meta_train(td, n_way=4, k_shot=3, q_queries=2, episodes=30)
    fsc = FewShotClassifier(pn)
    task = td.sample_task(4, 3, 2)
    fsc.fit(task.support_x, task.support_y)
    preds = fsc.predict(task.query_x)
    assert len(preds) == 8

def test_different_k_shots():
    """Test with 1-shot and 5-shot."""
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=20, feat_dim=4, seed=29)
    td = TaskDistribution(X, Y, seed=29)

    # 1-shot
    task1 = td.sample_task(3, 1, 1)
    assert len(task1.support_y) == 3
    assert task1.k_shot == 1

    # 5-shot
    task5 = td.sample_task(3, 5, 1)
    assert len(task5.support_y) == 15
    assert task5.k_shot == 5

def test_1_way_classification():
    """Edge case: 1-way (should always predict class 0)."""
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4, seed=30)
    td = TaskDistribution(X, Y, seed=30)
    encoder = build_model([4, 8, 4])
    pn = PrototypicalNetwork(encoder)
    task = td.sample_task(1, 3, 2)
    preds, _ = pn.classify(task.query_x, pn.compute_prototypes(
        task.support_x, task.support_y, 1))
    assert all(p == 0 for p in preds)

def test_large_n_way():
    """Test with many classes."""
    X, Y = make_few_shot_data(n_classes=10, samples_per_class=10, feat_dim=4, seed=31)
    td = TaskDistribution(X, Y, seed=31)
    task = td.sample_task(n_way=10, k_shot=2, q_queries=1)
    assert task.n_way == 10
    assert len(task.support_y) == 20
    assert len(task.query_y) == 10

def test_meta_trainer_results_tracking():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=32)
    td = TaskDistribution(X, Y, seed=32)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, first_order=True)
    trainer = MetaTrainer(maml, td)
    trainer.train(n_way=3, k_shot=3, epochs=5, tasks_per_epoch=2)
    assert len(trainer._results) == 1
    assert trainer._results[0]['algorithm'] == 'MAML'
    assert trainer._results[0]['n_way'] == 3


# ============================================================
# Edge cases and robustness
# ============================================================

def test_task_with_replacement():
    """When samples_per_class < k_shot + q_queries, sample with replacement."""
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    Y = [0, 0, 1, 1]
    td = TaskDistribution(X, Y)
    # Need 3+1=4 per class, but only 2 per class -> replacement
    task = td.sample_task(n_way=2, k_shot=3, q_queries=1)
    assert len(task.support_y) == 6

def test_clone_model_independence():
    """Cloned model should be independent."""
    model = build_model([4, 8, 3])
    clone = _clone_model(model)
    # Modify clone
    _set_flat_params(clone, [99.0] * len(_get_flat_params(clone)))
    # Original unchanged
    assert all(abs(p - 99.0) > 0.01 for p in _get_flat_params(model)
               if abs(p) > 0.01)

def test_matching_net_attention_sums_to_one():
    """Attention weights should sum to 1."""
    encoder = build_model([4, 8, 4])
    mn = MatchingNetwork(encoder)
    support_embeds = Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    query_embed = Tensor([0.5, 0.5, 0, 0])
    pred, probs = mn.attention_classify(query_embed, support_embeds, [0, 1, 2], 3)
    assert abs(sum(probs) - 1.0) < 1e-6

def test_few_shot_classifier_auto_nway():
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4, seed=33)
    td = TaskDistribution(X, Y, seed=33)
    model = build_model([4, 16, 5])
    reptile = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=3)
    fsc = FewShotClassifier(reptile)
    task = td.sample_task(3, 2, 1)
    fsc.fit(task.support_x, task.support_y)  # auto n_way
    assert fsc._n_way == 3

def test_multiple_training_rounds():
    """Can train, evaluate, train more."""
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=15, feat_dim=4, seed=34)
    td = TaskDistribution(X, Y, seed=34)
    model = build_model([4, 16, 5])
    maml = MAML(model, inner_lr=0.01, outer_lr=0.001, first_order=True)
    maml.meta_train(td, n_way=3, k_shot=3, meta_epochs=5, tasks_per_epoch=2)
    r1 = maml.evaluate(td, n_way=3, k_shot=3, n_tasks=3)
    maml.meta_train(td, n_way=3, k_shot=3, meta_epochs=5, tasks_per_epoch=2)
    r2 = maml.evaluate(td, n_way=3, k_shot=3, n_tasks=3)
    assert 'accuracy' in r1 and 'accuracy' in r2

def test_seed_reproducibility():
    """Same seed should give same tasks."""
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4, seed=35)
    td1 = TaskDistribution(X, Y, seed=100)
    td2 = TaskDistribution(X, Y, seed=100)
    t1 = td1.sample_task(3, 2, 1)
    t2 = td2.sample_task(3, 2, 1)
    assert t1.classes == t2.classes
    for i in range(len(t1.support_y)):
        assert t1.support_y[i] == t2.support_y[i]

def test_task_distribution_deterministic():
    """Multiple calls should give different tasks (seeded RNG)."""
    X, Y = make_few_shot_data(n_classes=5, samples_per_class=10, feat_dim=4)
    td = TaskDistribution(X, Y)
    t1 = td.sample_task(3, 2, 1)
    t2 = td.sample_task(3, 2, 1)
    # Very likely different classes
    # (could be same by chance, but 3-choose-from-5 twice same is rare)
    # Just check they're valid
    assert t1.n_way == 3
    assert t2.n_way == 3


# ============================================================
# Run all tests
# ============================================================

if __name__ == '__main__':
    import traceback
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    passed = 0
    failed = 0
    errors = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  FAIL: {name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print(f"\nFailed tests:")
        for name, e in errors:
            print(f"  {name}: {e}")
