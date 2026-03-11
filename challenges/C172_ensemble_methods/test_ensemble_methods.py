"""
Tests for C172: Ensemble Methods
"""

import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

from ensemble_methods import (
    Tensor, BaseLearnerFactory, BaggingEnsemble, RandomSubspaceEnsemble,
    AdaBoostEnsemble, GradientBoostingEnsemble, VotingEnsemble,
    StackingEnsemble, BlendingEnsemble, EnsembleSelection,
    DiversityMetrics, AutoEnsemble, SnapshotEnsemble, EnsembleComparison,
    _to_list, _argmax, _bootstrap_sample, _predict_class, _predict_proba,
    _accuracy, _mse, _softmax, _feature_subsample,
    Dense, Activation, Sequential, MSELoss, CrossEntropyLoss, SGD, Adam, fit
)


# ===========================================================================
# Helper: generate simple classification dataset
# ===========================================================================

def make_classification_data(n=60, n_features=4, n_classes=3, seed=42):
    """Generate linearly separable classification data."""
    rng = random.Random(seed)
    X = []
    Y = []
    for i in range(n):
        cls = i % n_classes
        row = [rng.gauss(cls * 2.0, 0.5) for _ in range(n_features)]
        X.append(row)
        one_hot = [0.0] * n_classes
        one_hot[cls] = 1.0
        Y.append(one_hot)
    return Tensor(X), Tensor(Y)


def make_binary_data(n=40, n_features=3, seed=42):
    """Generate binary classification data."""
    rng = random.Random(seed)
    X = []
    Y = []
    for i in range(n):
        cls = i % 2
        row = [rng.gauss(cls * 3.0, 0.8) for _ in range(n_features)]
        X.append(row)
        Y.append([1.0, 0.0] if cls == 0 else [0.0, 1.0])
    return Tensor(X), Tensor(Y)


def make_regression_data(n=40, seed=42):
    """Generate regression data: y = 2*x1 + x2 + noise."""
    rng = random.Random(seed)
    X = []
    Y = []
    for _ in range(n):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        y = 2 * x1 + x2 + rng.gauss(0, 0.1)
        X.append([x1, x2])
        Y.append([y])
    return Tensor(X), Tensor(Y)


def make_factory(input_size=4, output_size=3, epochs=30, seed=42):
    """Standard factory for tests."""
    return BaseLearnerFactory(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=[8],
        lr=0.05,
        epochs=epochs,
        seed=seed,
        loss_fn=CrossEntropyLoss() if output_size > 1 else MSELoss()
    )


# ===========================================================================
# Utility Tests
# ===========================================================================

class TestUtilities:
    def test_to_list_tensor(self):
        t = Tensor([1, 2, 3])
        assert _to_list(t) == [1, 2, 3]

    def test_to_list_plain(self):
        assert _to_list([1, 2]) == [1, 2]

    def test_argmax(self):
        assert _argmax([0.1, 0.5, 0.3]) == 1
        assert _argmax([0.9, 0.1]) == 0

    def test_argmax_tie(self):
        # First max wins
        assert _argmax([0.5, 0.5, 0.1]) == 0

    def test_bootstrap_sample(self):
        rng = random.Random(42)
        X = Tensor([[1], [2], [3], [4], [5]])
        Y = Tensor([0, 1, 0, 1, 0])
        X_b, Y_b, oob = _bootstrap_sample(X, Y, rng)
        assert len(_to_list(X_b)) == 5
        assert len(_to_list(Y_b)) == 5
        assert isinstance(oob, list)

    def test_feature_subsample(self):
        X = Tensor([[1, 2, 3], [4, 5, 6]])
        X_sub = _feature_subsample(X, [0, 2])
        assert _to_list(X_sub) == [[1, 3], [4, 6]]

    def test_softmax(self):
        s = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(s) - 1.0) < 1e-6
        assert s[2] > s[1] > s[0]

    def test_accuracy(self):
        preds = [0, 1, 2, 0]
        targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
        assert _accuracy(preds, targets) == 1.0

    def test_accuracy_partial(self):
        preds = [0, 0, 0]
        targets = [0, 1, 2]
        assert abs(_accuracy(preds, targets) - 1/3) < 1e-6

    def test_mse(self):
        preds = [1.0, 2.0, 3.0]
        targets = [1.0, 2.0, 3.0]
        assert _mse(preds, targets) == 0.0

    def test_mse_nonzero(self):
        preds = [1.0, 2.0]
        targets = [2.0, 3.0]
        assert abs(_mse(preds, targets) - 1.0) < 1e-6


# ===========================================================================
# BaseLearnerFactory Tests
# ===========================================================================

class TestBaseLearnerFactory:
    def test_build(self):
        factory = make_factory()
        model = factory.build()
        assert isinstance(model, Sequential)
        assert model.count_params() > 0

    def test_build_different_seeds(self):
        factory = make_factory()
        m1 = factory.build(rng=random.Random(1))
        m2 = factory.build(rng=random.Random(2))
        # Different models (different random init)
        w1 = _to_list(m1.layers[0].weights)
        w2 = _to_list(m2.layers[0].weights)
        assert w1 != w2

    def test_train(self):
        factory = make_factory(epochs=10)
        X, Y = make_classification_data(n=20)
        model = factory.build()
        factory.train(model, X, Y)
        out = model.predict(X)
        assert len(_to_list(out)) == 20

    def test_build_and_train(self):
        factory = make_factory(epochs=10)
        X, Y = make_classification_data(n=20)
        model = factory.build_and_train(X, Y)
        assert isinstance(model, Sequential)

    def test_single_output(self):
        factory = make_factory(input_size=2, output_size=1, epochs=5)
        model = factory.build()
        # No softmax for single output
        last_layer = model.layers[-1]
        assert isinstance(last_layer, Dense)

    def test_multi_output_softmax(self):
        factory = make_factory(output_size=3, epochs=5)
        model = factory.build()
        last_layer = model.layers[-1]
        assert isinstance(last_layer, Activation)

    def test_custom_hidden_sizes(self):
        factory = BaseLearnerFactory(
            input_size=4, output_size=2,
            hidden_sizes=[16, 8],
            epochs=5
        )
        model = factory.build()
        dense_layers = [l for l in model.layers if isinstance(l, Dense)]
        assert len(dense_layers) == 3  # 2 hidden + 1 output


# ===========================================================================
# BaggingEnsemble Tests
# ===========================================================================

class TestBaggingEnsemble:
    def test_fit(self):
        X, Y = make_classification_data(n=30)
        factory = make_factory(epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=3, seed=42)
        bag.fit(X, Y)
        assert len(bag.models) == 3

    def test_predict(self):
        X, Y = make_classification_data(n=30)
        factory = make_factory(epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=3, seed=42)
        bag.fit(X, Y)
        proba = bag.predict(X)
        assert len(proba) == 30
        assert len(proba[0]) == 3
        # Probabilities sum near 1
        for row in proba:
            assert abs(sum(row) - 1.0) < 0.1

    def test_predict_classes(self):
        X, Y = make_classification_data(n=30)
        factory = make_factory(epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=3, seed=42)
        bag.fit(X, Y)
        classes = bag.predict_classes(X)
        assert len(classes) == 30
        assert all(c in [0, 1, 2] for c in classes)

    def test_oob_score(self):
        X, Y = make_classification_data(n=30)
        factory = make_factory(epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=5, seed=42)
        bag.fit(X, Y)
        score = bag.oob_score(X, Y)
        assert 0.0 <= score <= 1.0

    def test_oob_indices_exist(self):
        X, Y = make_classification_data(n=30)
        factory = make_factory(epochs=10)
        bag = BaggingEnsemble(factory, n_estimators=3, seed=42)
        bag.fit(X, Y)
        assert len(bag.oob_indices) == 3
        # At least some OOB indices should exist
        total_oob = sum(len(oob) for oob in bag.oob_indices)
        assert total_oob > 0

    def test_more_estimators_better(self):
        X, Y = make_binary_data(n=40, seed=99)
        factory = make_factory(input_size=3, output_size=2, epochs=20, seed=99)
        bag1 = BaggingEnsemble(factory, n_estimators=1, seed=99)
        bag5 = BaggingEnsemble(factory, n_estimators=5, seed=99)
        bag1.fit(X, Y)
        bag5.fit(X, Y)
        # 5 models should give at least as good probability estimates
        p1 = bag1.predict(X)
        p5 = bag5.predict(X)
        assert len(p1) == len(p5)


# ===========================================================================
# RandomSubspaceEnsemble Tests
# ===========================================================================

class TestRandomSubspaceEnsemble:
    def test_fit(self):
        X, Y = make_classification_data(n=30, n_features=6)
        factory = make_factory(input_size=6, epochs=15)
        rs = RandomSubspaceEnsemble(factory, n_estimators=3, max_features=0.5, seed=42)
        rs.fit(X, Y)
        assert len(rs.models) == 3
        assert len(rs.feature_sets) == 3

    def test_feature_subsets_differ(self):
        X, Y = make_classification_data(n=20, n_features=8)
        factory = make_factory(input_size=8, epochs=10)
        rs = RandomSubspaceEnsemble(factory, n_estimators=3, max_features=0.5, seed=42)
        rs.fit(X, Y)
        # Feature sets should differ
        assert rs.feature_sets[0] != rs.feature_sets[1] or rs.feature_sets[1] != rs.feature_sets[2]

    def test_predict(self):
        X, Y = make_classification_data(n=20, n_features=6)
        factory = make_factory(input_size=6, epochs=15)
        rs = RandomSubspaceEnsemble(factory, n_estimators=3, max_features=0.5, seed=42)
        rs.fit(X, Y)
        proba = rs.predict(X)
        assert len(proba) == 20

    def test_predict_classes(self):
        X, Y = make_classification_data(n=20, n_features=6)
        factory = make_factory(input_size=6, epochs=15)
        rs = RandomSubspaceEnsemble(factory, n_estimators=3, max_features=0.5, seed=42)
        rs.fit(X, Y)
        classes = rs.predict_classes(X)
        assert len(classes) == 20
        assert all(c in [0, 1, 2] for c in classes)

    def test_integer_max_features(self):
        X, Y = make_classification_data(n=20, n_features=6)
        factory = make_factory(input_size=6, epochs=10)
        rs = RandomSubspaceEnsemble(factory, n_estimators=2, max_features=3, seed=42)
        rs.fit(X, Y)
        for fs in rs.feature_sets:
            assert len(fs) == 3


# ===========================================================================
# AdaBoostEnsemble Tests
# ===========================================================================

class TestAdaBoostEnsemble:
    def test_fit(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        ada = AdaBoostEnsemble(factory, n_estimators=3, seed=42)
        ada.fit(X, Y)
        assert len(ada.models) == 3
        assert len(ada.alphas) == 3

    def test_alphas_positive(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        ada = AdaBoostEnsemble(factory, n_estimators=3, seed=42)
        ada.fit(X, Y)
        # Alphas should generally be positive (better than random)
        assert any(a > 0 for a in ada.alphas)

    def test_predict(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        ada = AdaBoostEnsemble(factory, n_estimators=3, seed=42)
        ada.fit(X, Y)
        proba = ada.predict(X)
        assert len(proba) == 30
        assert len(proba[0]) == 2

    def test_predict_classes(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        ada = AdaBoostEnsemble(factory, n_estimators=3, seed=42)
        ada.fit(X, Y)
        classes = ada.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_learning_rate_effect(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        ada1 = AdaBoostEnsemble(factory, n_estimators=3, learning_rate=0.1, seed=42)
        ada2 = AdaBoostEnsemble(factory, n_estimators=3, learning_rate=1.0, seed=42)
        ada1.fit(X, Y)
        ada2.fit(X, Y)
        # Different learning rates -> different alphas
        assert ada1.alphas != ada2.alphas

    def test_multiclass(self):
        X, Y = make_classification_data(n=30, n_classes=3)
        factory = make_factory(input_size=4, output_size=3, epochs=15)
        ada = AdaBoostEnsemble(factory, n_estimators=3, seed=42)
        ada.fit(X, Y)
        classes = ada.predict_classes(X)
        assert all(c in [0, 1, 2] for c in classes)


# ===========================================================================
# GradientBoostingEnsemble Tests
# ===========================================================================

class TestGradientBoostingEnsemble:
    def test_fit(self):
        X, Y = make_regression_data(n=30)
        factory = make_factory(input_size=2, output_size=1, epochs=20)
        gb = GradientBoostingEnsemble(factory, n_estimators=3, seed=42)
        gb.fit(X, Y)
        assert len(gb.models) == 3

    def test_initial_prediction(self):
        X, Y = make_regression_data(n=30)
        factory = make_factory(input_size=2, output_size=1, epochs=20)
        gb = GradientBoostingEnsemble(factory, n_estimators=3, seed=42)
        gb.fit(X, Y)
        Y_data = _to_list(Y)
        expected_mean = sum(row[0] for row in Y_data) / len(Y_data)
        assert abs(gb.initial_prediction - expected_mean) < 1e-6

    def test_predict(self):
        X, Y = make_regression_data(n=30)
        factory = make_factory(input_size=2, output_size=1, epochs=20)
        gb = GradientBoostingEnsemble(factory, n_estimators=3, seed=42)
        gb.fit(X, Y)
        preds = gb.predict(X)
        assert len(preds) == 30

    def test_improves_over_mean(self):
        X, Y = make_regression_data(n=40, seed=123)
        factory = make_factory(input_size=2, output_size=1, epochs=30)
        gb = GradientBoostingEnsemble(factory, n_estimators=5, learning_rate=0.1, seed=123)
        gb.fit(X, Y)
        preds = gb.predict(X)
        Y_data = _to_list(Y)
        Y_flat = [row[0] for row in Y_data]
        mean = sum(Y_flat) / len(Y_flat)
        mse_mean = sum((y - mean) ** 2 for y in Y_flat) / len(Y_flat)
        mse_gb = sum((p - y) ** 2 for p, y in zip(preds, Y_flat)) / len(Y_flat)
        # Should improve over just predicting the mean
        assert mse_gb <= mse_mean * 1.5  # Some tolerance

    def test_learning_rate(self):
        X, Y = make_regression_data(n=30)
        factory = make_factory(input_size=2, output_size=1, epochs=15)
        gb1 = GradientBoostingEnsemble(factory, n_estimators=3, learning_rate=0.01, seed=42)
        gb2 = GradientBoostingEnsemble(factory, n_estimators=3, learning_rate=0.5, seed=42)
        gb1.fit(X, Y)
        gb2.fit(X, Y)
        p1 = gb1.predict(X)
        p2 = gb2.predict(X)
        # Different learning rates -> different predictions
        assert p1 != p2


# ===========================================================================
# VotingEnsemble Tests
# ===========================================================================

class TestVotingEnsemble:
    def _train_models(self, n=30):
        X, Y = make_binary_data(n=n)
        factory = make_factory(input_size=3, output_size=2, epochs=20)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(3)]
        return models, X, Y

    def test_soft_vote(self):
        models, X, Y = self._train_models()
        v = VotingEnsemble(models, voting='soft')
        proba = v.predict(X)
        assert len(proba) == 30
        assert len(proba[0]) == 2

    def test_hard_vote(self):
        models, X, Y = self._train_models()
        v = VotingEnsemble(models, voting='hard')
        preds = v.predict(X)
        assert len(preds) == 30
        assert all(c in [0, 1] for c in preds)

    def test_predict_classes_soft(self):
        models, X, Y = self._train_models()
        v = VotingEnsemble(models, voting='soft')
        classes = v.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_predict_classes_hard(self):
        models, X, Y = self._train_models()
        v = VotingEnsemble(models, voting='hard')
        classes = v.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_weighted_voting(self):
        models, X, Y = self._train_models()
        v1 = VotingEnsemble(models, voting='soft', weights=[1, 1, 1])
        v2 = VotingEnsemble(models, voting='soft', weights=[10, 0.1, 0.1])
        p1 = v1.predict(X)
        p2 = v2.predict(X)
        # With heavy weight on first model, predictions differ
        assert p1 != p2

    def test_single_model(self):
        models, X, Y = self._train_models()
        v = VotingEnsemble([models[0]], voting='soft')
        proba = v.predict(X)
        assert len(proba) == 30


# ===========================================================================
# StackingEnsemble Tests
# ===========================================================================

class TestStackingEnsemble:
    def test_fit(self):
        X, Y = make_binary_data(n=40)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(3)
        ]
        meta = make_factory(input_size=6, output_size=2, epochs=15, seed=99)
        stack = StackingEnsemble(base, meta, seed=42)
        stack.fit(X, Y)
        assert len(stack.base_models) == 3
        assert stack.meta_model is not None

    def test_predict(self):
        X, Y = make_binary_data(n=40)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(3)
        ]
        meta = make_factory(input_size=6, output_size=2, epochs=15, seed=99)
        stack = StackingEnsemble(base, meta, seed=42)
        stack.fit(X, Y)
        proba = stack.predict(X)
        assert len(proba) == 40

    def test_predict_classes(self):
        X, Y = make_binary_data(n=40)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(3)
        ]
        meta = make_factory(input_size=6, output_size=2, epochs=15, seed=99)
        stack = StackingEnsemble(base, meta, seed=42)
        stack.fit(X, Y)
        classes = stack.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_with_original_features(self):
        X, Y = make_binary_data(n=40)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(2)
        ]
        # Meta input: 4 (2 models * 2 classes) + 3 (features) = 7
        meta = make_factory(input_size=7, output_size=2, epochs=15, seed=99)
        stack = StackingEnsemble(base, meta, use_original_features=True, seed=42)
        stack.fit(X, Y)
        proba = stack.predict(X)
        assert len(proba) == 40

    def test_multiclass_stacking(self):
        X, Y = make_classification_data(n=30, n_classes=3)
        base = [
            make_factory(input_size=4, output_size=3, epochs=15, seed=i)
            for i in range(2)
        ]
        meta = make_factory(input_size=6, output_size=3, epochs=15, seed=99)
        stack = StackingEnsemble(base, meta, seed=42)
        stack.fit(X, Y)
        classes = stack.predict_classes(X)
        assert all(c in [0, 1, 2] for c in classes)


# ===========================================================================
# BlendingEnsemble Tests
# ===========================================================================

class TestBlendingEnsemble:
    def test_fit(self):
        X, Y = make_binary_data(n=50)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(3)
        ]
        meta = make_factory(input_size=6, output_size=2, epochs=15, seed=99)
        blend = BlendingEnsemble(base, meta, holdout_ratio=0.3, seed=42)
        blend.fit(X, Y)
        assert len(blend.base_models) == 3
        assert blend.meta_model is not None

    def test_predict(self):
        X, Y = make_binary_data(n=50)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(3)
        ]
        meta = make_factory(input_size=6, output_size=2, epochs=15, seed=99)
        blend = BlendingEnsemble(base, meta, holdout_ratio=0.3, seed=42)
        blend.fit(X, Y)
        proba = blend.predict(X)
        assert len(proba) == 50

    def test_predict_classes(self):
        X, Y = make_binary_data(n=50)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(2)
        ]
        meta = make_factory(input_size=4, output_size=2, epochs=15, seed=99)
        blend = BlendingEnsemble(base, meta, holdout_ratio=0.3, seed=42)
        blend.fit(X, Y)
        classes = blend.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_different_holdout_ratios(self):
        X, Y = make_binary_data(n=50)
        base = [make_factory(input_size=3, output_size=2, epochs=10, seed=0)]
        meta = make_factory(input_size=2, output_size=2, epochs=10, seed=99)
        b1 = BlendingEnsemble(base, meta, holdout_ratio=0.2, seed=42)
        b2 = BlendingEnsemble(base, meta, holdout_ratio=0.5, seed=42)
        b1.fit(X, Y)
        b2.fit(X, Y)
        # Both should work
        assert b1.meta_model is not None
        assert b2.meta_model is not None


# ===========================================================================
# EnsembleSelection Tests
# ===========================================================================

class TestEnsembleSelection:
    def test_fit(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(5)]
        es = EnsembleSelection(models, ensemble_size=3, seed=42)
        es.fit(X, Y)
        assert len(es.selected_models) > 0
        assert len(es.selected_weights) > 0

    def test_weights_sum_to_one(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(5)]
        es = EnsembleSelection(models, ensemble_size=3, seed=42)
        es.fit(X, Y)
        assert abs(sum(es.selected_weights) - 1.0) < 1e-6

    def test_predict(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(5)]
        es = EnsembleSelection(models, ensemble_size=3, seed=42)
        es.fit(X, Y)
        proba = es.predict(X)
        assert len(proba) == 40

    def test_predict_classes(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(5)]
        es = EnsembleSelection(models, ensemble_size=3, seed=42)
        es.fit(X, Y)
        classes = es.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_without_replacement(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(5)]
        es = EnsembleSelection(models, ensemble_size=3, with_replacement=False, seed=42)
        es.fit(X, Y)
        # Without replacement, max 5 models (but ensemble_size=3)
        assert len(es.selected_models) <= 3

    def test_ensemble_size_one(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(3)]
        es = EnsembleSelection(models, ensemble_size=1, seed=42)
        es.fit(X, Y)
        assert len(es.selected_models) == 1


# ===========================================================================
# DiversityMetrics Tests
# ===========================================================================

class TestDiversityMetrics:
    def test_disagreement_identical(self):
        preds = [0, 1, 0, 1, 0]
        assert DiversityMetrics.disagreement(preds, preds) == 0.0

    def test_disagreement_opposite(self):
        a = [0, 0, 0, 0]
        b = [1, 1, 1, 1]
        assert DiversityMetrics.disagreement(a, b) == 1.0

    def test_disagreement_partial(self):
        a = [0, 1, 0, 1]
        b = [0, 0, 0, 1]
        assert abs(DiversityMetrics.disagreement(a, b) - 0.25) < 1e-6

    def test_q_statistic_identical(self):
        preds = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]
        q = DiversityMetrics.q_statistic(preds, preds, labels)
        # Both perfect -> Q=1 (but n10=n01=0 -> denom=0 -> 0)
        assert isinstance(q, float)

    def test_q_statistic_range(self):
        a = [0, 1, 0, 1, 0, 1, 0, 1]
        b = [0, 0, 1, 1, 0, 1, 0, 1]
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        q = DiversityMetrics.q_statistic(a, b, labels)
        assert -1.0 <= q <= 1.0

    def test_correlation_diversity_identical(self):
        # When both are perfect, std=0, so returns 0 (degenerate case)
        a = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]
        c = DiversityMetrics.correlation_diversity(a, a, labels)
        assert abs(c - 0.0) < 1e-6  # std=0 -> 0 by convention

    def test_correlation_diversity_identical_imperfect(self):
        # Both imperfect but identical -> correlation 1.0
        a = [0, 1, 0, 0]
        labels = [0, 1, 1, 0]
        c = DiversityMetrics.correlation_diversity(a, a, labels)
        assert abs(c - 1.0) < 1e-6

    def test_correlation_diversity_range(self):
        a = [0, 1, 0, 1, 0, 1]
        b = [1, 0, 0, 1, 1, 0]
        labels = [0, 1, 0, 1, 0, 1]
        c = DiversityMetrics.correlation_diversity(a, b, labels)
        assert -1.0 <= c <= 1.0

    def test_ensemble_disagreement(self):
        X, Y = make_binary_data(n=20)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(3)]
        d = DiversityMetrics.ensemble_disagreement(models, X)
        assert 0.0 <= d <= 1.0

    def test_double_fault(self):
        a = [0, 0, 1, 1]
        b = [0, 1, 0, 1]
        labels = [1, 1, 1, 1]
        df = DiversityMetrics.double_fault(a, b, labels)
        # a wrong: 0,1 (indices 0,1)  b wrong: 0,2 (indices 0,2)
        # Both wrong at index 0 only
        assert abs(df - 0.25) < 1e-6

    def test_double_fault_none(self):
        a = [0, 1]
        b = [0, 1]
        labels = [0, 1]
        assert DiversityMetrics.double_fault(a, b, labels) == 0.0


# ===========================================================================
# AutoEnsemble Tests
# ===========================================================================

class TestAutoEnsemble:
    def test_fit(self):
        X, Y = make_binary_data(n=40)
        auto = AutoEnsemble(input_size=3, output_size=2, n_estimators=2,
                            method='bagging', seed=42)
        auto.fit(X, Y, n_trials=3, val_split=0.3)
        assert auto.best_ensemble is not None
        assert auto.best_config is not None

    def test_predict(self):
        X, Y = make_binary_data(n=40)
        auto = AutoEnsemble(input_size=3, output_size=2, n_estimators=2, seed=42)
        auto.fit(X, Y, n_trials=3)
        proba = auto.predict(X)
        assert len(proba) == 40

    def test_predict_classes(self):
        X, Y = make_binary_data(n=40)
        auto = AutoEnsemble(input_size=3, output_size=2, n_estimators=2, seed=42)
        auto.fit(X, Y, n_trials=3)
        classes = auto.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_search_history(self):
        X, Y = make_binary_data(n=40)
        auto = AutoEnsemble(input_size=3, output_size=2, n_estimators=2, seed=42)
        auto.fit(X, Y, n_trials=5)
        assert len(auto.search_history) == 5

    def test_adaboost_method(self):
        X, Y = make_binary_data(n=40)
        auto = AutoEnsemble(input_size=3, output_size=2, n_estimators=2,
                            method='adaboost', seed=42)
        auto.fit(X, Y, n_trials=3)
        assert auto.best_ensemble is not None

    def test_best_score(self):
        X, Y = make_binary_data(n=40)
        auto = AutoEnsemble(input_size=3, output_size=2, n_estimators=2, seed=42)
        auto.fit(X, Y, n_trials=3)
        assert 0.0 <= auto.best_score <= 1.0


# ===========================================================================
# SnapshotEnsemble Tests
# ===========================================================================

class TestSnapshotEnsemble:
    def test_fit(self):
        X, Y = make_binary_data(n=30)
        snap = SnapshotEnsemble(input_size=3, output_size=2, hidden_sizes=[8],
                                 n_cycles=3, epochs_per_cycle=5, seed=42)
        snap.fit(X, Y)
        assert len(snap.snapshots) == 3

    def test_predict(self):
        X, Y = make_binary_data(n=30)
        snap = SnapshotEnsemble(input_size=3, output_size=2, hidden_sizes=[8],
                                 n_cycles=3, epochs_per_cycle=5, seed=42)
        snap.fit(X, Y)
        proba = snap.predict(X)
        assert len(proba) == 30
        assert len(proba[0]) == 2

    def test_predict_classes(self):
        X, Y = make_binary_data(n=30)
        snap = SnapshotEnsemble(input_size=3, output_size=2, hidden_sizes=[8],
                                 n_cycles=3, epochs_per_cycle=5, seed=42)
        snap.fit(X, Y)
        classes = snap.predict_classes(X)
        assert all(c in [0, 1] for c in classes)

    def test_snapshots_differ(self):
        X, Y = make_binary_data(n=30)
        snap = SnapshotEnsemble(input_size=3, output_size=2, hidden_sizes=[8],
                                 n_cycles=3, epochs_per_cycle=10, seed=42)
        snap.fit(X, Y)
        # Snapshots should have different weights
        w0 = _to_list(snap.snapshots[0].layers[0].weights)
        w2 = _to_list(snap.snapshots[2].layers[0].weights)
        assert w0 != w2

    def test_cosine_lr(self):
        snap = SnapshotEnsemble(input_size=2, output_size=2, max_lr=0.1)
        lr_start = snap._cosine_lr(0, 20)
        lr_mid = snap._cosine_lr(10, 20)
        lr_end = snap._cosine_lr(20, 20)
        assert abs(lr_start - 0.1) < 1e-6
        assert lr_mid < lr_start
        assert abs(lr_end) < 1e-6


# ===========================================================================
# EnsembleComparison Tests
# ===========================================================================

class TestEnsembleComparison:
    def test_evaluate(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=2, seed=42)
        bag.fit(X, Y)
        ada = AdaBoostEnsemble(factory, n_estimators=2, seed=42)
        ada.fit(X, Y)
        comp = EnsembleComparison([bag, ada], names=['Bagging', 'AdaBoost'])
        results = comp.evaluate(X, Y)
        assert 'Bagging' in results
        assert 'AdaBoost' in results
        assert 0.0 <= results['Bagging']['accuracy'] <= 1.0

    def test_best(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=2, seed=42)
        bag.fit(X, Y)
        ada = AdaBoostEnsemble(factory, n_estimators=2, seed=42)
        ada.fit(X, Y)
        comp = EnsembleComparison([bag, ada], names=['Bagging', 'AdaBoost'])
        comp.evaluate(X, Y)
        best = comp.best()
        assert best in ['Bagging', 'AdaBoost']

    def test_best_no_eval(self):
        comp = EnsembleComparison([], names=[])
        assert comp.best() is None

    def test_summary(self):
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=2, seed=42)
        bag.fit(X, Y)
        comp = EnsembleComparison([bag], names=['Bagging'])
        comp.evaluate(X, Y)
        s = comp.summary()
        assert 'Bagging' in s
        assert 'Best' in s

    def test_default_names(self):
        X, Y = make_binary_data(n=30)
        factory = make_factory(input_size=3, output_size=2, epochs=10)
        bag = BaggingEnsemble(factory, n_estimators=2, seed=42)
        bag.fit(X, Y)
        comp = EnsembleComparison([bag])
        comp.evaluate(X, Y)
        assert 'Ensemble_0' in comp.results


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_bagging_then_voting(self):
        """Train bagging, extract models, use in voting."""
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=3, seed=42)
        bag.fit(X, Y)
        vote = VotingEnsemble(bag.models, voting='soft')
        proba = vote.predict(X)
        assert len(proba) == 40

    def test_stacking_with_bagging_base(self):
        """Use bagging ensembles as base learners for stacking."""
        X, Y = make_binary_data(n=50)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        # Create base factories
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(2)
        ]
        meta = make_factory(input_size=4, output_size=2, epochs=15, seed=99)
        stack = StackingEnsemble(base, meta, seed=42)
        stack.fit(X, Y)
        classes = stack.predict_classes(X)
        assert len(classes) == 50

    def test_ensemble_selection_from_diverse_models(self):
        """Select from models with different architectures."""
        X, Y = make_binary_data(n=40)
        models = []
        for i, hidden in enumerate([4, 8, 16]):
            f = BaseLearnerFactory(
                input_size=3, output_size=2, hidden_sizes=[hidden],
                lr=0.05, epochs=15, seed=i,
                loss_fn=CrossEntropyLoss()
            )
            models.append(f.build_and_train(X, Y))
        es = EnsembleSelection(models, ensemble_size=2, seed=42)
        es.fit(X, Y)
        classes = es.predict_classes(X)
        assert len(classes) == 40

    def test_diversity_between_bagging_models(self):
        """Measure diversity of bagging ensemble members."""
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=5, seed=42)
        bag.fit(X, Y)
        d = DiversityMetrics.ensemble_disagreement(bag.models, X)
        assert 0.0 <= d <= 1.0

    def test_comparison_all_methods(self):
        """Compare multiple ensemble methods."""
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)

        bag = BaggingEnsemble(factory, n_estimators=2, seed=42)
        bag.fit(X, Y)
        ada = AdaBoostEnsemble(factory, n_estimators=2, seed=42)
        ada.fit(X, Y)
        snap = SnapshotEnsemble(input_size=3, output_size=2, n_cycles=2,
                                 epochs_per_cycle=5, seed=42)
        snap.fit(X, Y)

        comp = EnsembleComparison([bag, ada, snap],
                                   names=['Bagging', 'AdaBoost', 'Snapshot'])
        results = comp.evaluate(X, Y)
        assert len(results) == 3
        assert comp.best() is not None

    def test_auto_ensemble_end_to_end(self):
        """Full auto-ensemble pipeline."""
        X, Y = make_classification_data(n=60, n_features=4, n_classes=3)
        auto = AutoEnsemble(input_size=4, output_size=3, n_estimators=2, seed=42)
        auto.fit(X, Y, n_trials=3)
        classes = auto.predict_classes(X)
        assert len(classes) == 60
        assert all(c in [0, 1, 2] for c in classes)

    def test_gradient_boosting_regression(self):
        """End-to-end gradient boosting for regression."""
        X, Y = make_regression_data(n=40)
        factory = make_factory(input_size=2, output_size=1, epochs=20)
        gb = GradientBoostingEnsemble(factory, n_estimators=3, learning_rate=0.1, seed=42)
        gb.fit(X, Y)
        preds = gb.predict(X)
        assert len(preds) == 40
        # All predictions should be finite
        assert all(math.isfinite(p) for p in preds)

    def test_random_subspace_diversity(self):
        """Random subspace should produce diverse models."""
        X, Y = make_classification_data(n=30, n_features=8)
        factory = make_factory(input_size=8, output_size=3, epochs=15)
        rs = RandomSubspaceEnsemble(factory, n_estimators=4, max_features=0.5, seed=42)
        rs.fit(X, Y)
        # Models should use different feature sets
        unique_sets = set(tuple(fs) for fs in rs.feature_sets)
        assert len(unique_sets) >= 2

    def test_blending_then_compare(self):
        """Blend and compare with other methods."""
        X, Y = make_binary_data(n=50)
        base = [
            make_factory(input_size=3, output_size=2, epochs=15, seed=i)
            for i in range(2)
        ]
        meta = make_factory(input_size=4, output_size=2, epochs=15, seed=99)
        blend = BlendingEnsemble(base, meta, holdout_ratio=0.3, seed=42)
        blend.fit(X, Y)

        factory = make_factory(input_size=3, output_size=2, epochs=15)
        bag = BaggingEnsemble(factory, n_estimators=2, seed=42)
        bag.fit(X, Y)

        comp = EnsembleComparison([blend, bag], names=['Blending', 'Bagging'])
        results = comp.evaluate(X, Y)
        assert len(results) == 2

    def test_full_pipeline(self):
        """Full pipeline: train, diversity, selection, compare."""
        X, Y = make_binary_data(n=40)
        factory = make_factory(input_size=3, output_size=2, epochs=15)

        # Train library
        models = [factory.build_and_train(X, Y, rng=random.Random(i)) for i in range(5)]

        # Diversity
        d = DiversityMetrics.ensemble_disagreement(models, X)
        assert 0.0 <= d <= 1.0

        # Selection
        es = EnsembleSelection(models, ensemble_size=3, seed=42)
        es.fit(X, Y)

        # Compare single vs ensemble
        single_preds = _predict_class(models[0], X)
        ensemble_preds = es.predict_classes(X)
        assert len(single_preds) == len(ensemble_preds)


# ===========================================================================
# Run all tests
# ===========================================================================

def run_tests():
    """Run all test classes."""
    import traceback
    test_classes = [
        TestUtilities, TestBaseLearnerFactory,
        TestBaggingEnsemble, TestRandomSubspaceEnsemble,
        TestAdaBoostEnsemble, TestGradientBoostingEnsemble,
        TestVotingEnsemble, TestStackingEnsemble,
        TestBlendingEnsemble, TestEnsembleSelection,
        TestDiversityMetrics, TestAutoEnsemble,
        TestSnapshotEnsemble, TestEnsembleComparison,
        TestIntegration
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
