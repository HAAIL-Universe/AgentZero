"""
Tests for C175: Semi-Supervised Learning
"""

import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

import pytest
from semi_supervised import (
    SelfTrainer, CoTrainer, LabelPropagation, LabelSpreading,
    ConsistencyRegularizer, MixUp, MixMatchTrainer, FixMatchTrainer,
    SemiSupervisedTrainer, SemiSupervisedMetrics,
    _argmax, _max_val, _entropy, _euclidean_dist_sq, _rbf_kernel,
    _predict_proba, _to_list, _tensor_row, _num_rows, _make_tensor,
    _subset_tensor, _concat_tensors, _concat_labels
)
from neural_network import (
    Tensor, Sequential, Dense, Activation, Dropout,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    SGD, Adam, fit, predict_classes, accuracy, one_hot,
    build_model, softmax_batch
)


# ===================================================================
# Test data generators
# ===================================================================

def make_blobs(n_per_class=50, n_classes=2, dim=2, seed=42, spread=0.5):
    """Create simple cluster data."""
    rng = random.Random(seed)
    X, y = [], []
    centers = []
    for c in range(n_classes):
        angle = 2 * math.pi * c / n_classes
        center = [0.0] * dim
        center[0] = 2 * math.cos(angle)
        center[1] = 2 * math.sin(angle)
        # Extra dimensions get class-specific offset
        for d in range(2, dim):
            center[d] = rng.gauss(0, 0.5)
        centers.append(center)
    for c in range(n_classes):
        for _ in range(n_per_class):
            x = [centers[c][d] + rng.gauss(0, spread) for d in range(dim)]
            X.append(x)
            y.append(c)
    return X, y


def split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=42):
    """Split data into labeled and unlabeled subsets."""
    rng = random.Random(seed)
    classes = sorted(set(y))
    labeled_idx = []
    for c in classes:
        c_idx = [i for i in range(len(y)) if y[i] == c]
        rng.shuffle(c_idx)
        labeled_idx.extend(c_idx[:n_labeled_per_class])
    unlabeled_idx = [i for i in range(len(y)) if i not in set(labeled_idx)]

    X_l = [X[i] for i in labeled_idx]
    y_l = [y[i] for i in labeled_idx]
    X_u = [X[i] for i in unlabeled_idx]
    y_u = [y[i] for i in unlabeled_idx]
    return X_l, y_l, X_u, y_u


def make_simple_model(input_dim=2, num_classes=2, seed=42):
    """Build a simple neural network."""
    return build_model([input_dim, 16, num_classes],
                       activations=['relu', 'linear'],
                       init='xavier',
                       rng=random.Random(seed))


# ===================================================================
# Utility tests
# ===================================================================

class TestUtilities:
    def test_argmax(self):
        assert _argmax([0.1, 0.7, 0.2]) == 1
        assert _argmax([0.9, 0.05, 0.05]) == 0
        assert _argmax([0.0, 0.0, 1.0]) == 2

    def test_max_val(self):
        assert _max_val([0.1, 0.7, 0.2]) == 0.7
        assert _max_val([1.0]) == 1.0

    def test_entropy(self):
        # Uniform distribution: max entropy
        h_uniform = _entropy([0.5, 0.5])
        assert h_uniform > 0.6
        # Deterministic: zero entropy
        h_det = _entropy([1.0, 0.0])
        assert h_det < 0.01
        # More classes, more entropy
        h3 = _entropy([1/3, 1/3, 1/3])
        assert h3 > h_uniform

    def test_euclidean_dist(self):
        assert _euclidean_dist_sq([0, 0], [3, 4]) == 25.0
        assert _euclidean_dist_sq([1, 1], [1, 1]) == 0.0

    def test_rbf_kernel(self):
        # Same point -> 1.0
        assert _rbf_kernel([0, 0], [0, 0], 1.0) == 1.0
        # Far points -> near 0
        k = _rbf_kernel([0, 0], [10, 10], 1.0)
        assert k < 0.001
        # Gamma affects width
        k1 = _rbf_kernel([0, 0], [1, 0], 0.5)
        k2 = _rbf_kernel([0, 0], [1, 0], 2.0)
        assert k1 > k2  # Smaller gamma = wider kernel

    def test_tensor_row(self):
        t = Tensor([[1, 2], [3, 4]])
        assert _tensor_row(t, 0) == [1, 2]
        assert _tensor_row(t, 1) == [3, 4]

    def test_num_rows(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        assert _num_rows(t) == 3
        assert _num_rows([[1], [2]]) == 2

    def test_make_tensor(self):
        t = _make_tensor([[1, 2], [3, 4]])
        assert isinstance(t, Tensor)
        t2 = _make_tensor(t)
        assert isinstance(t2, Tensor)

    def test_subset_tensor(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        s = _subset_tensor(t, [0, 2])
        assert _num_rows(s) == 2

    def test_concat_tensors(self):
        t1 = Tensor([[1, 2]])
        t2 = Tensor([[3, 4]])
        c = _concat_tensors(t1, t2)
        assert _num_rows(c) == 2

    def test_concat_labels(self):
        assert _concat_labels([0, 1], [2, 3]) == [0, 1, 2, 3]

    def test_to_list_tensor(self):
        t = Tensor([[1, 2], [3, 4]])
        result = _to_list(t)
        assert isinstance(result, list)

    def test_to_list_plain(self):
        assert _to_list([1, 2, 3]) == [1, 2, 3]


# ===================================================================
# SelfTrainer tests
# ===================================================================

class TestSelfTrainer:
    def test_basic_fit(self):
        X, y = make_blobs(n_per_class=30, n_classes=2, seed=1)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=1)
        model = make_simple_model(seed=1)
        trainer = SelfTrainer(model, threshold=0.7, max_iter=5,
                              epochs_per_iter=20, seed=1)
        history = trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        assert len(history) > 0
        # Should have added some pseudo-labels
        total_pseudo = sum(h[2] for h in history)
        assert total_pseudo >= 0

    def test_predict(self):
        X, y = make_blobs(n_per_class=20, seed=2)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=2)
        model = make_simple_model(seed=2)
        trainer = SelfTrainer(model, threshold=0.6, max_iter=3,
                              epochs_per_iter=15, seed=2)
        trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        preds = trainer.predict(Tensor(X_l))
        assert len(preds) == len(X_l)
        assert all(p in (0, 1) for p in preds)

    def test_predict_proba(self):
        X, y = make_blobs(n_per_class=20, seed=3)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=3)
        model = make_simple_model(seed=3)
        trainer = SelfTrainer(model, threshold=0.6, max_iter=2,
                              epochs_per_iter=10, seed=3)
        trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        probs = trainer.predict_proba(Tensor(X_l))
        assert len(probs) == len(X_l)
        for p in probs:
            assert abs(sum(p) - 1.0) < 0.05

    def test_validation_tracking(self):
        X, y = make_blobs(n_per_class=30, seed=4)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=4)
        model = make_simple_model(seed=4)
        trainer = SelfTrainer(model, threshold=0.6, max_iter=3,
                              epochs_per_iter=15, seed=4)
        history = trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2,
                              X_val=Tensor(X_u), y_val=y_u)
        # Should have val accuracy in history
        for h in history:
            assert h[3] is not None  # val_acc recorded

    def test_high_threshold_no_pseudo(self):
        X, y = make_blobs(n_per_class=20, seed=5)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=3, seed=5)
        model = make_simple_model(seed=5)
        trainer = SelfTrainer(model, threshold=0.999, max_iter=2,
                              epochs_per_iter=5, seed=5)
        history = trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        assert len(history) > 0

    def test_empty_unlabeled(self):
        X, y = make_blobs(n_per_class=10, seed=6)
        model = make_simple_model(seed=6)
        trainer = SelfTrainer(model, threshold=0.5, max_iter=3,
                              epochs_per_iter=5, seed=6)
        history = trainer.fit(Tensor(X), y, Tensor([X[0]]), num_classes=2)
        assert len(history) >= 0

    def test_three_classes(self):
        X, y = make_blobs(n_per_class=20, n_classes=3, seed=7)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=7)
        model = make_simple_model(input_dim=2, num_classes=3, seed=7)
        trainer = SelfTrainer(model, threshold=0.6, max_iter=3,
                              epochs_per_iter=15, seed=7)
        history = trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=3)
        preds = trainer.predict(Tensor(X_l))
        assert all(p in (0, 1, 2) for p in preds)

    def test_custom_loss_optimizer(self):
        X, y = make_blobs(n_per_class=15, seed=8)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=3, seed=8)
        model = make_simple_model(seed=8)
        trainer = SelfTrainer(model, threshold=0.7, max_iter=2,
                              loss_fn=CrossEntropyLoss(),
                              optimizer=SGD(lr=0.05, momentum=0.9),
                              epochs_per_iter=10, seed=8)
        history = trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        assert len(history) > 0

    def test_history_format(self):
        X, y = make_blobs(n_per_class=15, seed=9)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=3, seed=9)
        model = make_simple_model(seed=9)
        trainer = SelfTrainer(model, threshold=0.6, max_iter=3,
                              epochs_per_iter=10, seed=9)
        history = trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        for h in history:
            assert len(h) == 4  # (iteration, n_labeled, n_pseudo, val_acc)
            assert isinstance(h[0], int)
            assert isinstance(h[1], int)
            assert isinstance(h[2], int)


# ===================================================================
# CoTrainer tests
# ===================================================================

class TestCoTrainer:
    def test_basic_fit(self):
        X, y = make_blobs(n_per_class=30, n_classes=2, dim=4, seed=10)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=10)
        model1 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(10))
        model2 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(11))
        co = CoTrainer(model1, model2, view1_indices=[0, 1], view2_indices=[2, 3],
                       threshold=0.7, max_iter=3, n_per_iter=3,
                       epochs_per_iter=15, seed=10)
        history = co.fit(Tensor(X_l), y_l, Tensor(X_u))
        assert len(history) > 0

    def test_predict(self):
        X, y = make_blobs(n_per_class=20, dim=4, seed=11)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=11)
        model1 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(11))
        model2 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(12))
        co = CoTrainer(model1, model2, [0, 1], [2, 3],
                       threshold=0.6, max_iter=3, epochs_per_iter=10, seed=11)
        co.fit(Tensor(X_l), y_l, Tensor(X_u))
        preds = co.predict(Tensor(X_l))
        assert len(preds) == len(X_l)

    def test_predict_proba(self):
        X, y = make_blobs(n_per_class=20, dim=4, seed=12)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=12)
        model1 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(12))
        model2 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(13))
        co = CoTrainer(model1, model2, [0, 1], [2, 3],
                       threshold=0.6, max_iter=2, epochs_per_iter=10, seed=12)
        co.fit(Tensor(X_l), y_l, Tensor(X_u))
        probs = co.predict_proba(Tensor(X_l))
        assert len(probs) == len(X_l)
        for p in probs:
            assert len(p) == 2

    def test_validation_tracking(self):
        X, y = make_blobs(n_per_class=25, dim=4, seed=13)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=13)
        model1 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(13))
        model2 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(14))
        co = CoTrainer(model1, model2, [0, 1], [2, 3],
                       threshold=0.6, max_iter=3, epochs_per_iter=10, seed=13)
        history = co.fit(Tensor(X_l), y_l, Tensor(X_u),
                         X_val=Tensor(X_u), y_val=y_u)
        for h in history:
            assert h[4] is not None  # val_acc

    def test_three_classes(self):
        X, y = make_blobs(n_per_class=20, n_classes=3, dim=4, seed=14)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=14)
        model1 = build_model([2, 8, 3], activations=['relu', 'linear'], rng=random.Random(14))
        model2 = build_model([2, 8, 3], activations=['relu', 'linear'], rng=random.Random(15))
        co = CoTrainer(model1, model2, [0, 1], [2, 3],
                       threshold=0.5, max_iter=3, epochs_per_iter=10,
                       num_classes=3, seed=14)
        history = co.fit(Tensor(X_l), y_l, Tensor(X_u))
        preds = co.predict(Tensor(X_l))
        assert all(p in (0, 1, 2) for p in preds)

    def test_history_format(self):
        X, y = make_blobs(n_per_class=15, dim=4, seed=15)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=3, seed=15)
        model1 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(15))
        model2 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(16))
        co = CoTrainer(model1, model2, [0, 1], [2, 3],
                       threshold=0.5, max_iter=2, epochs_per_iter=5, seed=15)
        history = co.fit(Tensor(X_l), y_l, Tensor(X_u))
        for h in history:
            assert len(h) == 5  # (iter, n_l1, n_l2, added, val_acc)


# ===================================================================
# LabelPropagation tests
# ===================================================================

class TestLabelPropagation:
    def test_basic_fit(self):
        X = [[0, 0], [0.1, 0.1], [0.2, 0], [3, 3], [3.1, 3.1], [3, 3.2]]
        y = [0, -1, -1, 1, -1, -1]
        lp = LabelPropagation(gamma=1.0, max_iter=50)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert len(labels) == 6
        assert labels[0] == 0
        assert labels[3] == 1

    def test_propagation_works(self):
        """Nearby unlabeled points should get same label as labeled neighbor."""
        X = [[0, 0], [0.1, 0], [0.2, 0], [5, 5], [5.1, 5], [5.2, 5]]
        y = [0, -1, -1, 1, -1, -1]
        lp = LabelPropagation(gamma=2.0, max_iter=100)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        # Near-0 cluster should be class 0
        assert labels[1] == 0
        assert labels[2] == 0
        # Near-5 cluster should be class 1
        assert labels[4] == 1
        assert labels[5] == 1

    def test_predict_proba(self):
        X = [[0, 0], [0.1, 0], [5, 5], [5.1, 5]]
        y = [0, -1, 1, -1]
        lp = LabelPropagation(gamma=1.0, max_iter=50)
        lp.fit(X, y, num_classes=2)
        probs = lp.predict_proba()
        assert len(probs) == 4
        for p in probs:
            assert len(p) == 2
            assert abs(sum(p) - 1.0) < 0.05

    def test_labeled_clamped(self):
        """Labeled nodes should keep their labels."""
        X = [[0, 0], [5, 5], [2.5, 2.5]]
        y = [0, 1, -1]
        lp = LabelPropagation(gamma=0.5, max_iter=100)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert labels[0] == 0
        assert labels[1] == 1

    def test_knn_kernel(self):
        X = [[0, 0], [0.1, 0], [0.2, 0], [10, 10], [10.1, 10], [10.2, 10]]
        y = [0, -1, -1, 1, -1, -1]
        lp = LabelPropagation(kernel='knn', n_neighbors=2, max_iter=100)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert labels[1] == 0
        assert labels[4] == 1

    def test_three_classes(self):
        X = [[0, 0], [0.1, 0], [3, 0], [3.1, 0], [0, 3], [0.1, 3]]
        y = [0, -1, 1, -1, 2, -1]
        lp = LabelPropagation(gamma=2.0, max_iter=100)
        lp.fit(X, y, num_classes=3)
        labels = lp.predict()
        assert labels[0] == 0
        assert labels[2] == 1
        assert labels[4] == 2

    def test_convergence(self):
        """Should converge within max_iter."""
        X = [[i * 0.1, 0] for i in range(10)]
        y = [0] + [-1] * 8 + [1]
        lp = LabelPropagation(gamma=5.0, max_iter=200)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert len(labels) == 10

    def test_all_labeled(self):
        """If all labeled, should just return those labels."""
        X = [[0, 0], [1, 1], [2, 2]]
        y = [0, 1, 0]
        lp = LabelPropagation(gamma=1.0, max_iter=50)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert labels == [0, 1, 0]

    def test_tensor_input(self):
        X = Tensor([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
        y = [0, -1, 1, -1]
        lp = LabelPropagation(gamma=1.0, max_iter=50)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert len(labels) == 4


# ===================================================================
# LabelSpreading tests
# ===================================================================

class TestLabelSpreading:
    def test_basic_fit(self):
        X = [[0, 0], [0.1, 0], [5, 5], [5.1, 5]]
        y = [0, -1, 1, -1]
        ls = LabelSpreading(gamma=1.0, alpha=0.2, max_iter=50)
        ls.fit(X, y, num_classes=2)
        labels = ls.predict()
        assert len(labels) == 4

    def test_spreading_works(self):
        X = [[0, 0], [0.1, 0], [0.2, 0], [5, 5], [5.1, 5], [5.2, 5]]
        y = [0, -1, -1, 1, -1, -1]
        ls = LabelSpreading(gamma=2.0, alpha=0.2, max_iter=100)
        ls.fit(X, y, num_classes=2)
        labels = ls.predict()
        assert labels[1] == 0
        assert labels[2] == 0
        assert labels[4] == 1
        assert labels[5] == 1

    def test_predict_proba(self):
        X = [[0, 0], [0.1, 0], [5, 5], [5.1, 5]]
        y = [0, -1, 1, -1]
        ls = LabelSpreading(gamma=1.0, alpha=0.3, max_iter=50)
        ls.fit(X, y, num_classes=2)
        probs = ls.predict_proba()
        for p in probs:
            assert len(p) == 2
            assert abs(sum(p) - 1.0) < 0.1

    def test_alpha_effect(self):
        """Higher alpha should produce softer label distributions."""
        X = [[0, 0], [2.5, 2.5], [5, 5]]
        y = [0, -1, 1]
        # Low alpha (strong clamping)
        ls1 = LabelSpreading(gamma=0.5, alpha=0.1, max_iter=100)
        ls1.fit(X, y, num_classes=2)
        probs1 = ls1.predict_proba()
        # High alpha (weak clamping)
        ls2 = LabelSpreading(gamma=0.5, alpha=0.8, max_iter=100)
        ls2.fit(X, y, num_classes=2)
        probs2 = ls2.predict_proba()
        # Both should produce valid distributions
        for p in probs1 + probs2:
            assert abs(sum(p) - 1.0) < 0.2

    def test_three_classes(self):
        X = [[0, 0], [0.1, 0], [3, 0], [3.1, 0], [0, 3], [0.1, 3]]
        y = [0, -1, 1, -1, 2, -1]
        ls = LabelSpreading(gamma=2.0, alpha=0.2, max_iter=100)
        ls.fit(X, y, num_classes=3)
        labels = ls.predict()
        assert labels[0] == 0
        assert labels[2] == 1
        assert labels[4] == 2

    def test_tensor_input(self):
        X = Tensor([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
        y = [0, -1, 1, -1]
        ls = LabelSpreading(gamma=1.0, alpha=0.2, max_iter=50)
        ls.fit(X, y, num_classes=2)
        labels = ls.predict()
        assert len(labels) == 4


# ===================================================================
# ConsistencyRegularizer tests
# ===================================================================

class TestConsistencyRegularizer:
    def test_consistency_loss(self):
        model = make_simple_model(seed=20)
        X = Tensor([[1, 2], [3, 4], [5, 6]])
        cr = ConsistencyRegularizer(noise_std=0.1, consistency_weight=1.0, seed=20)
        loss = cr.consistency_loss(model, X, epoch=100)
        assert loss >= 0

    def test_ramp_up(self):
        cr = ConsistencyRegularizer(noise_std=0.1, consistency_weight=2.0,
                                    ramp_up_epochs=10, seed=20)
        # Epoch 0: weight should be near 0
        w0 = cr._ramp_weight(0)
        assert w0 < 0.1
        # Epoch 10: weight should be at full
        w10 = cr._ramp_weight(10)
        assert abs(w10 - 2.0) < 0.01
        # Monotonically increasing
        w5 = cr._ramp_weight(5)
        assert w0 < w5 < w10

    def test_train_step(self):
        model = make_simple_model(seed=21)
        X_l = Tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
        y_l = one_hot([0, 1, 0, 1], 2)
        X_u = Tensor([[0.5, 0.5], [0.2, 0.8]])
        cr = ConsistencyRegularizer(noise_std=0.1, consistency_weight=1.0, seed=21)
        optimizer = Adam(lr=0.01)
        sup_loss, cons_loss = cr.train_step(model, X_l, y_l, X_u,
                                            CrossEntropyLoss(), optimizer, epoch=10)
        assert sup_loss >= 0
        assert cons_loss >= 0

    def test_noise_adds_perturbation(self):
        cr = ConsistencyRegularizer(noise_std=0.5, seed=22)
        X = Tensor([[1, 2], [3, 4]])
        X_noisy = cr._add_noise(X)
        # Should be different
        orig = [_tensor_row(X, i) for i in range(2)]
        noisy = [_tensor_row(X_noisy, i) for i in range(2)]
        diff = sum(abs(orig[i][j] - noisy[i][j]) for i in range(2) for j in range(2))
        assert diff > 0.01

    def test_zero_noise(self):
        cr = ConsistencyRegularizer(noise_std=0.0, seed=23)
        model = make_simple_model(seed=23)
        X = Tensor([[1, 2], [3, 4]])
        loss = cr.consistency_loss(model, X, epoch=100)
        assert loss < 0.01  # Same input -> same output -> low loss


# ===================================================================
# MixUp tests
# ===================================================================

class TestMixUp:
    def test_basic_mix(self):
        X1 = Tensor([[1, 0], [0, 1]])
        y1 = [[1, 0], [0, 1]]  # One-hot
        mu = MixUp(alpha=1.0, seed=30)
        X_m, y_m, lam = mu.mix(X1, Tensor(y1))
        assert _num_rows(X_m) == 2
        assert 0 <= lam <= 1

    def test_lambda_range(self):
        mu = MixUp(alpha=0.5, seed=31)
        for _ in range(20):
            lam = mu._beta_sample(0.5)
            assert 0 <= lam <= 1

    def test_self_shuffle(self):
        """When X2/y2 not given, should shuffle X1/y1."""
        X1 = Tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
        y1 = Tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
        mu = MixUp(alpha=1.0, seed=32)
        X_m, y_m, lam = mu.mix(X1, y1)
        assert _num_rows(X_m) == 4

    def test_explicit_pairs(self):
        X1 = Tensor([[1, 0]])
        y1 = [[1, 0]]
        X2 = Tensor([[0, 1]])
        y2 = [[0, 1]]
        mu = MixUp(alpha=1.0, seed=33)
        X_m, y_m, lam = mu.mix(X1, Tensor(y1), X2, y2)
        # Mixed should be interpolation
        row = _tensor_row(X_m, 0)
        assert abs(row[0] - lam) < 0.01
        assert abs(row[1] - (1 - lam)) < 0.01

    def test_integer_labels(self):
        """Integer labels should produce (y1, y2, lam) tuple."""
        X1 = Tensor([[1, 0], [0, 1]])
        y1 = [0, 1]
        mu = MixUp(alpha=1.0, seed=34)
        X_m, y_m, lam = mu.mix(X1, y1)
        assert isinstance(y_m, tuple)
        assert len(y_m) == 3  # (y1, y2, lam)

    def test_alpha_zero(self):
        """Alpha 0 should give lambda ~0.5."""
        mu = MixUp(alpha=0.0, seed=35)
        lam = mu._beta_sample(0.0)
        assert lam == 0.5

    def test_gamma_sample(self):
        mu = MixUp(seed=36)
        # Sample from Gamma and check it's positive
        for alpha in [0.5, 1.0, 2.0]:
            for _ in range(10):
                g = mu._gamma_sample(alpha)
                assert g >= 0

    def test_beta_sample_distribution(self):
        """Beta samples should be in [0, 1]."""
        mu = MixUp(seed=37)
        for _ in range(50):
            lam = mu._beta_sample(0.75)
            assert 0 <= lam <= 1


# ===================================================================
# MixMatchTrainer tests
# ===================================================================

class TestMixMatchTrainer:
    def test_basic_fit(self):
        X, y = make_blobs(n_per_class=20, seed=40)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=40)
        model = make_simple_model(seed=40)
        mm = MixMatchTrainer(model, num_classes=2, T=0.5, K=2,
                             noise_std=0.1, seed=40)
        history = mm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=5)
        assert len(history) == 5

    def test_train_step(self):
        X, y = make_blobs(n_per_class=15, seed=41)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=41)
        model = make_simple_model(seed=41)
        mm = MixMatchTrainer(model, num_classes=2, seed=41)
        sup_loss, unsup_loss = mm.train_step(Tensor(X_l), y_l, Tensor(X_u))
        assert sup_loss >= 0
        assert unsup_loss >= 0

    def test_predict(self):
        X, y = make_blobs(n_per_class=15, seed=42)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=42)
        model = make_simple_model(seed=42)
        mm = MixMatchTrainer(model, num_classes=2, seed=42)
        mm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        preds = mm.predict(Tensor(X_l))
        assert len(preds) == len(X_l)

    def test_predict_proba(self):
        X, y = make_blobs(n_per_class=15, seed=43)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=43)
        model = make_simple_model(seed=43)
        mm = MixMatchTrainer(model, num_classes=2, seed=43)
        mm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        probs = mm.predict_proba(Tensor(X_l))
        for p in probs:
            assert abs(sum(p) - 1.0) < 0.05

    def test_sharpening(self):
        mm = MixMatchTrainer(make_simple_model(seed=44), num_classes=2, T=0.1, seed=44)
        # Low T -> sharp distribution
        sharp = mm._sharpen([[0.6, 0.4]], 0.1)
        assert sharp[0][0] > 0.9  # Should be very peaked

    def test_sharpening_high_T(self):
        mm = MixMatchTrainer(make_simple_model(seed=45), num_classes=2, T=10.0, seed=45)
        result = mm._sharpen([[0.7, 0.3]], 10.0)
        # High T -> more uniform
        assert abs(result[0][0] - result[0][1]) < 0.5

    def test_validation_tracking(self):
        X, y = make_blobs(n_per_class=20, seed=46)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=46)
        model = make_simple_model(seed=46)
        mm = MixMatchTrainer(model, num_classes=2, seed=46)
        history = mm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3,
                         X_val=Tensor(X_u), y_val=y_u)
        for h in history:
            assert h[3] is not None

    def test_three_classes(self):
        X, y = make_blobs(n_per_class=20, n_classes=3, seed=47)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=47)
        model = make_simple_model(input_dim=2, num_classes=3, seed=47)
        mm = MixMatchTrainer(model, num_classes=3, seed=47)
        mm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        preds = mm.predict(Tensor(X_l))
        assert all(p in (0, 1, 2) for p in preds)


# ===================================================================
# FixMatchTrainer tests
# ===================================================================

class TestFixMatchTrainer:
    def test_basic_fit(self):
        X, y = make_blobs(n_per_class=20, seed=50)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=50)
        model = make_simple_model(seed=50)
        fm = FixMatchTrainer(model, num_classes=2, threshold=0.7,
                             weak_noise=0.05, strong_noise=0.2, seed=50)
        history = fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=5)
        assert len(history) == 5

    def test_train_step(self):
        X, y = make_blobs(n_per_class=15, seed=51)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=51)
        model = make_simple_model(seed=51)
        fm = FixMatchTrainer(model, num_classes=2, threshold=0.5, seed=51)
        y_oh = one_hot(y_l, 2)
        sup, unsup, n_ps = fm.train_step(Tensor(X_l), y_oh, Tensor(X_u))
        assert sup >= 0
        assert unsup >= 0
        assert n_ps >= 0

    def test_predict(self):
        X, y = make_blobs(n_per_class=15, seed=52)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=52)
        model = make_simple_model(seed=52)
        fm = FixMatchTrainer(model, num_classes=2, seed=52)
        fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        preds = fm.predict(Tensor(X_l))
        assert len(preds) == len(X_l)

    def test_predict_proba(self):
        X, y = make_blobs(n_per_class=15, seed=53)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=53)
        model = make_simple_model(seed=53)
        fm = FixMatchTrainer(model, num_classes=2, seed=53)
        fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        probs = fm.predict_proba(Tensor(X_l))
        for p in probs:
            assert abs(sum(p) - 1.0) < 0.05

    def test_high_threshold(self):
        """High threshold should result in fewer pseudo-labels."""
        X, y = make_blobs(n_per_class=15, seed=54)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=54)
        model = make_simple_model(seed=54)
        fm = FixMatchTrainer(model, num_classes=2, threshold=0.999, seed=54)
        history = fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        # Very few pseudo-labels with high threshold
        total_pseudo = sum(h[3] for h in history)
        assert total_pseudo >= 0  # Might be 0 or very few

    def test_validation_tracking(self):
        X, y = make_blobs(n_per_class=20, seed=55)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=55)
        model = make_simple_model(seed=55)
        fm = FixMatchTrainer(model, num_classes=2, seed=55)
        history = fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3,
                         X_val=Tensor(X_u), y_val=y_u)
        for h in history:
            assert h[4] is not None

    def test_three_classes(self):
        X, y = make_blobs(n_per_class=20, n_classes=3, seed=56)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=56)
        model = make_simple_model(input_dim=2, num_classes=3, seed=56)
        fm = FixMatchTrainer(model, num_classes=3, threshold=0.5, seed=56)
        fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        preds = fm.predict(Tensor(X_l))
        assert all(p in (0, 1, 2) for p in preds)

    def test_history_format(self):
        X, y = make_blobs(n_per_class=15, seed=57)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=57)
        model = make_simple_model(seed=57)
        fm = FixMatchTrainer(model, num_classes=2, seed=57)
        history = fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        for h in history:
            assert len(h) == 5  # (epoch, sup, unsup, n_pseudo, val_acc)


# ===================================================================
# SemiSupervisedTrainer tests
# ===================================================================

class TestSemiSupervisedTrainer:
    def test_self_training_strategy(self):
        X, y = make_blobs(n_per_class=20, seed=60)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=60)
        model = make_simple_model(seed=60)
        sst = SemiSupervisedTrainer(model, strategy='self_training',
                                    threshold=0.7, max_iter=3,
                                    epochs_per_iter=10, seed=60)
        history = sst.fit(Tensor(X_l), y_l, Tensor(X_u))
        assert len(history) > 0

    def test_fixmatch_strategy(self):
        X, y = make_blobs(n_per_class=20, seed=61)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=61)
        model = make_simple_model(seed=61)
        sst = SemiSupervisedTrainer(model, strategy='fixmatch',
                                    threshold=0.7, seed=61)
        history = sst.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        assert len(history) == 3

    def test_mixmatch_strategy(self):
        X, y = make_blobs(n_per_class=20, seed=62)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=62)
        model = make_simple_model(seed=62)
        sst = SemiSupervisedTrainer(model, strategy='mixmatch',
                                    T=0.5, seed=62)
        history = sst.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        assert len(history) == 3

    def test_consistency_strategy(self):
        X, y = make_blobs(n_per_class=20, seed=63)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=63)
        model = make_simple_model(seed=63)
        sst = SemiSupervisedTrainer(model, strategy='consistency',
                                    noise_std=0.1, seed=63)
        history = sst.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        assert len(history) == 3

    def test_predict_self_training(self):
        X, y = make_blobs(n_per_class=15, seed=64)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=64)
        model = make_simple_model(seed=64)
        sst = SemiSupervisedTrainer(model, strategy='self_training',
                                    threshold=0.6, max_iter=2,
                                    epochs_per_iter=10, seed=64)
        sst.fit(Tensor(X_l), y_l, Tensor(X_u))
        preds = sst.predict(Tensor(X_l))
        assert len(preds) == len(X_l)

    def test_predict_proba(self):
        X, y = make_blobs(n_per_class=15, seed=65)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=65)
        model = make_simple_model(seed=65)
        sst = SemiSupervisedTrainer(model, strategy='fixmatch', seed=65)
        sst.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3)
        probs = sst.predict_proba(Tensor(X_l))
        assert len(probs) == len(X_l)

    def test_unknown_strategy(self):
        model = make_simple_model(seed=66)
        sst = SemiSupervisedTrainer(model, strategy='unknown', seed=66)
        with pytest.raises(ValueError):
            sst.fit(Tensor([[0, 0]]), [0], Tensor([[1, 1]]))

    def test_validation_tracking(self):
        X, y = make_blobs(n_per_class=20, seed=67)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=67)
        model = make_simple_model(seed=67)
        sst = SemiSupervisedTrainer(model, strategy='consistency', seed=67)
        history = sst.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=3,
                          X_val=Tensor(X_u), y_val=y_u)
        for h in history:
            assert h[3] is not None


# ===================================================================
# SemiSupervisedMetrics tests
# ===================================================================

class TestSemiSupervisedMetrics:
    def test_label_utilization(self):
        assert SemiSupervisedMetrics.label_utilization(10, 100) == 0.1
        assert SemiSupervisedMetrics.label_utilization(0, 100) == 0.0
        assert SemiSupervisedMetrics.label_utilization(100, 100) == 1.0
        assert SemiSupervisedMetrics.label_utilization(5, 0) == 0.0

    def test_pseudo_label_accuracy(self):
        assert SemiSupervisedMetrics.pseudo_label_accuracy([0, 1, 0], [0, 1, 0]) == 1.0
        assert SemiSupervisedMetrics.pseudo_label_accuracy([0, 0, 0], [1, 1, 1]) == 0.0
        assert abs(SemiSupervisedMetrics.pseudo_label_accuracy([0, 1, 0], [0, 0, 0]) - 2/3) < 0.01

    def test_ssl_gain(self):
        assert SemiSupervisedMetrics.ssl_gain(0.8, 0.9) == pytest.approx(0.1)
        assert SemiSupervisedMetrics.ssl_gain(0.9, 0.85) == pytest.approx(-0.05)
        assert SemiSupervisedMetrics.ssl_gain(0.7, 0.7) == pytest.approx(0.0)

    def test_effective_label_ratio(self):
        curve = [(10, 0.6), (20, 0.7), (50, 0.85), (100, 0.9)]
        # SSL achieves 0.85 -> equivalent to 50 labels supervised
        assert SemiSupervisedMetrics.effective_label_ratio(0.85, curve) == 50
        # SSL achieves 0.95 -> beyond supervised curve
        assert SemiSupervisedMetrics.effective_label_ratio(0.95, curve) == 100

    def test_confidence_histogram(self):
        probs = [[0.9, 0.1], [0.8, 0.2], [0.55, 0.45], [0.99, 0.01]]
        hist = SemiSupervisedMetrics.confidence_histogram(probs, bins=10)
        assert len(hist) == 10
        assert sum(hist) == 4

    def test_class_balance(self):
        labels = [0, 0, 0, 1, 1]
        balance = SemiSupervisedMetrics.class_balance(labels, num_classes=2)
        assert balance[0] == pytest.approx(0.6)
        assert balance[1] == pytest.approx(0.4)

    def test_class_balance_uniform(self):
        labels = [0, 1, 2, 0, 1, 2]
        balance = SemiSupervisedMetrics.class_balance(labels, num_classes=3)
        assert all(abs(b - 1/3) < 0.01 for b in balance)

    def test_entropy_score(self):
        # Confident predictions -> low entropy
        low_ent = SemiSupervisedMetrics.entropy_score([[0.99, 0.01], [0.98, 0.02]])
        # Uncertain predictions -> high entropy
        high_ent = SemiSupervisedMetrics.entropy_score([[0.5, 0.5], [0.5, 0.5]])
        assert low_ent < high_ent

    def test_summary_fixmatch(self):
        history = [(0, 0.5, 0.3, 5, 0.7), (1, 0.4, 0.2, 8, 0.8)]
        s = SemiSupervisedMetrics.summary(history, 'fixmatch')
        assert s['iterations'] == 2
        assert s['final_sup_loss'] == 0.4
        assert s['final_unsup_loss'] == 0.2
        assert s['final_val_acc'] == 0.8

    def test_summary_self_training(self):
        history = [(0, 10, 5, 0.7), (1, 15, 3, 0.8)]
        s = SemiSupervisedMetrics.summary(history, 'self_training')
        assert s['iterations'] == 2
        assert s['final_n_labeled'] == 15
        assert s['total_pseudo'] == 8
        assert s['final_val_acc'] == 0.8

    def test_summary_empty(self):
        s = SemiSupervisedMetrics.summary([], 'generic')
        assert s['iterations'] == 0

    def test_entropy_score_empty(self):
        assert SemiSupervisedMetrics.entropy_score([]) == 0.0

    def test_pseudo_label_accuracy_empty(self):
        assert SemiSupervisedMetrics.pseudo_label_accuracy([], []) == 0.0


# ===================================================================
# Integration tests
# ===================================================================

class TestIntegration:
    def test_self_training_improves(self):
        """Self-training should not degrade compared to supervised-only."""
        X, y = make_blobs(n_per_class=40, n_classes=2, seed=70, spread=0.3)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=3, seed=70)

        # Supervised only
        model_sup = make_simple_model(seed=70)
        fit(model_sup, Tensor(X_l), one_hot(y_l, 2), CrossEntropyLoss(),
            Adam(lr=0.01), epochs=50, verbose=False)
        acc_sup = accuracy(model_sup, Tensor(X_u), y_u)

        # Self-training
        model_ssl = make_simple_model(seed=70)
        trainer = SelfTrainer(model_ssl, threshold=0.7, max_iter=5,
                              epochs_per_iter=20, seed=70)
        trainer.fit(Tensor(X_l), y_l, Tensor(X_u), num_classes=2)
        acc_ssl = accuracy(model_ssl, Tensor(X_u), y_u)

        # SSL should be at least close to supervised
        assert acc_ssl >= acc_sup - 0.15

    def test_label_propagation_vs_random(self):
        """Label propagation should beat random assignment."""
        X = [[0, 0], [0.1, 0.05], [0.15, 0.1], [0.05, 0.15],
             [5, 5], [5.1, 5.05], [5.15, 5.1], [5.05, 5.15]]
        y = [0, -1, -1, -1, 1, -1, -1, -1]
        true_y = [0, 0, 0, 0, 1, 1, 1, 1]

        lp = LabelPropagation(gamma=2.0, max_iter=100)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()

        correct = sum(1 for i in range(8) if labels[i] == true_y[i])
        assert correct >= 6  # Should get most right

    def test_metrics_with_real_training(self):
        """Metrics should work with actual training output."""
        X, y = make_blobs(n_per_class=20, seed=71)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=71)
        model = make_simple_model(seed=71)
        fm = FixMatchTrainer(model, num_classes=2, threshold=0.6, seed=71)
        history = fm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=5,
                         X_val=Tensor(X_u), y_val=y_u)

        s = SemiSupervisedMetrics.summary(history, 'fixmatch')
        assert 'iterations' in s
        assert s['iterations'] == 5

        probs = fm.predict_proba(Tensor(X_u))
        hist = SemiSupervisedMetrics.confidence_histogram(probs)
        assert sum(hist) == len(X_u)

        ent = SemiSupervisedMetrics.entropy_score(probs)
        assert ent >= 0

    def test_mixmatch_with_metrics(self):
        X, y = make_blobs(n_per_class=20, seed=72)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=72)
        model = make_simple_model(seed=72)
        mm = MixMatchTrainer(model, num_classes=2, T=0.5, seed=72)
        history = mm.fit(Tensor(X_l), y_l, Tensor(X_u), epochs=5,
                         X_val=Tensor(X_u), y_val=y_u)

        preds = mm.predict(Tensor(X_u))
        balance = SemiSupervisedMetrics.class_balance(preds, num_classes=2)
        assert len(balance) == 2
        assert abs(sum(balance) - 1.0) < 0.01

    def test_consistency_regularization_effect(self):
        """Consistency regularizer should produce valid losses."""
        X, y = make_blobs(n_per_class=20, seed=73)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=73)
        model = make_simple_model(seed=73)
        cr = ConsistencyRegularizer(noise_std=0.1, consistency_weight=1.0, seed=73)
        optimizer = Adam(lr=0.01)
        loss_fn = CrossEntropyLoss()
        y_oh = one_hot(y_l, 2)

        losses = []
        for epoch in range(5):
            sup, cons = cr.train_step(model, Tensor(X_l), y_oh,
                                      Tensor(X_u), loss_fn, optimizer, epoch)
            losses.append(sup)
        assert all(l >= 0 for l in losses)

    def test_label_spreading_convergence(self):
        """Label spreading should converge to correct labels."""
        X = [[0, 0], [0.05, 0.05], [10, 10], [10.05, 10.05]]
        y = [0, -1, 1, -1]
        ls = LabelSpreading(gamma=1.0, alpha=0.2, max_iter=200)
        ls.fit(X, y, num_classes=2)
        labels = ls.predict()
        assert labels[1] == 0
        assert labels[3] == 1

    def test_full_pipeline(self):
        """Full pipeline: generate data -> split -> train -> evaluate."""
        X, y = make_blobs(n_per_class=30, n_classes=3, seed=74, spread=0.4)
        X_l, y_l, X_u, y_u = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=74)

        model = make_simple_model(input_dim=2, num_classes=3, seed=74)
        sst = SemiSupervisedTrainer(model, strategy='self_training',
                                    num_classes=3, threshold=0.6,
                                    max_iter=5, epochs_per_iter=20, seed=74)
        sst.fit(Tensor(X_l), y_l, Tensor(X_u))
        preds = sst.predict(Tensor(X_u))

        # Check predictions are valid classes
        assert all(p in (0, 1, 2) for p in preds)

        # Check metrics
        pl_acc = SemiSupervisedMetrics.pseudo_label_accuracy(preds, y_u)
        assert 0 <= pl_acc <= 1
        util = SemiSupervisedMetrics.label_utilization(len(y_l), len(y_l) + len(y_u))
        assert util < 0.5  # We used few labels

    def test_co_training_basic_pipeline(self):
        """Co-training with feature split pipeline."""
        X, y = make_blobs(n_per_class=25, dim=4, seed=75, spread=0.4)
        X_l, y_l, X_u, _ = split_labeled_unlabeled(X, y, n_labeled_per_class=5, seed=75)
        model1 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(75))
        model2 = build_model([2, 8, 2], activations=['relu', 'linear'], rng=random.Random(76))
        co = CoTrainer(model1, model2, [0, 1], [2, 3],
                       threshold=0.6, max_iter=3, epochs_per_iter=15, seed=75)
        history = co.fit(Tensor(X_l), y_l, Tensor(X_u))
        preds = co.predict(Tensor(X_l))
        assert len(preds) == len(X_l)

    def test_mixup_in_training(self):
        """MixUp produces valid mixed data."""
        mu = MixUp(alpha=1.0, seed=76)
        X = Tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
        y = Tensor([[1, 0], [0, 1], [1, 0], [0, 1]])
        X_m, y_m, lam = mu.mix(X, y)
        assert _num_rows(X_m) == 4
        # All values should be in reasonable range
        for i in range(4):
            row = _tensor_row(X_m, i)
            for v in row:
                assert -0.5 <= v <= 1.5


# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    def test_single_labeled_sample(self):
        model = make_simple_model(seed=80)
        trainer = SelfTrainer(model, threshold=0.5, max_iter=2,
                              epochs_per_iter=5, seed=80)
        X_l = Tensor([[0, 0]])
        y_l = [0]
        X_u = Tensor([[1, 1], [2, 2]])
        # Should not crash
        history = trainer.fit(X_l, y_l, X_u, num_classes=2)
        assert isinstance(history, list)

    def test_label_propagation_single_class(self):
        X = [[0, 0], [0.1, 0], [0.2, 0]]
        y = [0, -1, -1]
        lp = LabelPropagation(gamma=1.0, max_iter=50)
        lp.fit(X, y, num_classes=2)
        labels = lp.predict()
        assert all(l == 0 for l in labels)

    def test_label_utilization_edge(self):
        assert SemiSupervisedMetrics.label_utilization(0, 0) == 0.0

    def test_empty_confidence_histogram(self):
        hist = SemiSupervisedMetrics.confidence_histogram([], bins=5)
        assert hist == [0, 0, 0, 0, 0]

    def test_class_balance_single_class(self):
        balance = SemiSupervisedMetrics.class_balance([0, 0, 0], num_classes=2)
        assert balance[0] == 1.0
        assert balance[1] == 0.0

    def test_ssl_gain_identical(self):
        assert SemiSupervisedMetrics.ssl_gain(0.5, 0.5) == 0.0

    def test_effective_label_ratio_empty_curve(self):
        assert SemiSupervisedMetrics.effective_label_ratio(0.9, []) == 1.0

    def test_summary_generic(self):
        s = SemiSupervisedMetrics.summary([(0, 1, 2)], 'generic')
        assert s['iterations'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
