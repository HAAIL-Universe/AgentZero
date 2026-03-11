"""Tests for C170: Transfer Learning."""
import math
import random
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.dirname(__file__))

from neural_network import (
    Tensor, Sequential, Dense, Activation, Dropout, BatchNorm,
    MSELoss, CrossEntropyLoss, SGD, Adam, fit, evaluate,
    predict_classes, accuracy, save_weights, load_weights,
    build_model, one_hot, normalize, train_test_split,
    make_xor_data, make_spiral_data, make_regression_data
)
from transfer_learning import (
    freeze_layer, unfreeze_layer, is_frozen,
    count_frozen_params, count_trainable_params,
    PretrainedModel, FeatureExtractor, FineTuner,
    FrozenAwareOptimizer, DomainAdapter, KnowledgeDistiller,
    ModelRegistry, TransferTrainer, DataAugmenter,
    MultiTaskHead, EWC, ProgressiveNet
)


# ============================================================
# Helper: create and train a simple model
# ============================================================

def make_trained_model(input_size=2, hidden=8, output_size=2, seed=42):
    """Create and train a small model on XOR-like data."""
    rng = random.Random(seed)
    model = Sequential([
        Dense(input_size, hidden, rng=rng),
        Activation('relu'),
        Dense(hidden, hidden, rng=rng),
        Activation('relu'),
        Dense(hidden, output_size, rng=rng),
    ])
    X, Y = make_xor_data(n=50, seed=seed)
    loss_fn = CrossEntropyLoss()
    opt = Adam(lr=0.01)
    fit(model, X, Y, loss_fn, opt, epochs=30, batch_size=25)
    return model


def make_regression_model(input_size=1, hidden=8, output_size=1, seed=42):
    """Create and train a small regression model."""
    rng = random.Random(seed)
    model = Sequential([
        Dense(input_size, hidden, rng=rng),
        Activation('relu'),
        Dense(hidden, hidden, rng=rng),
        Activation('relu'),
        Dense(hidden, output_size, rng=rng),
    ])
    X, Y = make_regression_data(n=50, seed=seed)
    loss_fn = MSELoss()
    opt = Adam(lr=0.01)
    fit(model, X, Y, loss_fn, opt, epochs=30, batch_size=25)
    return model


# ============================================================
# Tests: Layer Freezing Utilities
# ============================================================

class TestLayerFreezing:
    def test_freeze_unfreeze(self):
        layer = Dense(4, 3)
        assert not is_frozen(layer)
        freeze_layer(layer)
        assert is_frozen(layer)
        unfreeze_layer(layer)
        assert not is_frozen(layer)

    def test_freeze_activation(self):
        layer = Activation('relu')
        freeze_layer(layer)
        assert is_frozen(layer)

    def test_count_frozen_params(self):
        model = Sequential([
            Dense(4, 3),
            Activation('relu'),
            Dense(3, 2),
        ])
        assert count_frozen_params(model) == 0
        freeze_layer(model.layers[0])
        assert count_frozen_params(model) == 4 * 3 + 3  # weights + bias

    def test_count_trainable_params(self):
        model = Sequential([
            Dense(4, 3),
            Activation('relu'),
            Dense(3, 2),
        ])
        total = model.count_params()
        freeze_layer(model.layers[0])
        trainable = count_trainable_params(model)
        frozen = count_frozen_params(model)
        assert trainable + frozen == total

    def test_all_frozen(self):
        model = Sequential([Dense(2, 3), Dense(3, 1)])
        for l in model.layers:
            freeze_layer(l)
        assert count_trainable_params(model) == 0


# ============================================================
# Tests: PretrainedModel
# ============================================================

class TestPretrainedModel:
    def test_from_sequential(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        assert 'backbone' in pm.groups
        assert 'head' in pm.groups
        assert len(pm.groups['backbone']) + len(pm.groups['head']) == len(model.layers)

    def test_freeze_group(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm.freeze_group('backbone')
        for idx in pm.groups['backbone']:
            assert is_frozen(model.layers[idx])
        for idx in pm.groups['head']:
            assert not is_frozen(model.layers[idx])

    def test_unfreeze_group(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm.freeze_all()
        pm.unfreeze_group('head')
        for idx in pm.groups['head']:
            assert not is_frozen(model.layers[idx])
        for idx in pm.groups['backbone']:
            assert is_frozen(model.layers[idx])

    def test_freeze_all_unfreeze_all(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm.freeze_all()
        assert count_trainable_params(model) == 0
        pm.unfreeze_all()
        assert count_frozen_params(model) == 0

    def test_get_group_layers(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        backbone = pm.get_group_layers('backbone')
        assert len(backbone) > 0

    def test_add_group(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm.add_group('custom', [0, 1])
        assert 'custom' in pm.groups

    def test_replace_head(self):
        model = make_trained_model(output_size=2)
        pm = PretrainedModel.from_sequential(model)
        new_head = [Dense(8, 5), Activation('relu'), Dense(5, 3)]
        pm2 = pm.replace_head(new_head)
        assert len(pm2.groups['head']) == 3
        # Forward should work
        x = Tensor([[0.5, 0.5]])
        out = pm2.forward(x)
        assert out.shape[1] == 3

    def test_forward(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        x = Tensor([[0.0, 1.0]])
        out = pm.forward(x)
        assert len(out.shape) == 2

    def test_predict(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        x = Tensor([[1.0, 0.0]])
        out = pm.predict(x)
        assert out is not None

    def test_summary(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        s = pm.summary()
        assert 'backbone' in s
        assert 'head' in s
        assert 'Trainable' in s

    def test_unknown_group_error(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        with pytest.raises(ValueError):
            pm.freeze_group('nonexistent')

    def test_metadata(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm._metadata['task'] = 'xor'
        assert pm._metadata['task'] == 'xor'

    def test_custom_groups(self):
        model = Sequential([
            Dense(4, 8), Activation('relu'),
            Dense(8, 8), Activation('relu'),
            Dense(8, 4), Activation('relu'),
            Dense(4, 2)
        ])
        groups = {
            'layer1': [0, 1],
            'layer2': [2, 3],
            'layer3': [4, 5],
            'output': [6]
        }
        pm = PretrainedModel(model, groups)
        pm.freeze_group('layer1')
        pm.freeze_group('layer2')
        assert is_frozen(model.layers[0])
        assert is_frozen(model.layers[2])
        assert not is_frozen(model.layers[4])


# ============================================================
# Tests: FeatureExtractor
# ============================================================

class TestFeatureExtractor:
    def test_extract_single(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        fe = FeatureExtractor(pm)
        x = Tensor([[1.0, 0.0]])
        features = fe.extract_single(x)
        assert isinstance(features, Tensor)

    def test_extract_multi(self):
        model = make_trained_model()
        fe = FeatureExtractor(model, extract_at=[1, 3])
        x = Tensor([[1.0, 0.0]])
        features = fe.extract(x)
        assert 1 in features
        assert 3 in features

    def test_extract_up_to(self):
        model = make_trained_model()
        fe = FeatureExtractor(model)
        x = Tensor([[1.0, 0.0]])
        out = fe.extract_up_to(x, 1)
        assert isinstance(out, Tensor)

    def test_extract_default_layer(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        fe = FeatureExtractor(pm)
        assert len(fe.extract_at) == 1

    def test_extract_from_plain_model(self):
        model = make_trained_model()
        fe = FeatureExtractor(model, extract_at=0)
        x = Tensor([[0.0, 1.0]])
        features = fe.extract(x)
        assert 0 in features

    def test_extract_batch(self):
        model = make_trained_model()
        fe = FeatureExtractor(model, extract_at=[0])
        x = Tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        features = fe.extract(x)
        assert features[0].shape[0] == 3

    def test_feature_consistency(self):
        """Same input should give same features."""
        model = make_trained_model()
        fe = FeatureExtractor(model, extract_at=[0])
        x = Tensor([[0.5, 0.5]])
        f1 = fe.extract_single(x)
        f2 = fe.extract_single(x)
        for a, b in zip(f1.data[0], f2.data[0]):
            assert abs(a - b) < 1e-10


# ============================================================
# Tests: FineTuner
# ============================================================

class TestFineTuner:
    def test_discriminative_lrs(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ft = FineTuner(pm, base_lr=0.001, lr_mult=0.1)
        lrs = ft.set_discriminative_lrs()
        # Head should have highest LR
        head_indices = pm.groups['head']
        backbone_indices = pm.groups['backbone']
        head_lr = max(lrs[i] for i in head_indices if i in lrs)
        if backbone_indices:
            bb_lrs = [lrs[i] for i in backbone_indices if i in lrs]
            if bb_lrs:
                assert head_lr >= max(bb_lrs)

    def test_gradual_unfreeze_setup(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ft = FineTuner(pm)
        schedule = ft.setup_gradual_unfreeze(epochs_per_layer=2)
        assert len(schedule) > 0
        # After setup, everything should be frozen
        assert count_trainable_params(model) == 0

    def test_step_unfreeze(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ft = FineTuner(pm)
        ft.setup_gradual_unfreeze(epochs_per_layer=1)
        # At epoch 0, head should be unfrozen
        ft.step_unfreeze(0)
        head_layers = pm.get_group_layers('head')
        has_unfrozen = any(not is_frozen(l) for l in head_layers)
        assert has_unfrozen

    def test_discriminative_step(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ft = FineTuner(pm, base_lr=0.1)
        ft.set_discriminative_lrs()
        pm.unfreeze_all()

        # Save head weight value before
        head_idx = pm.groups['head']
        head_dense = None
        for idx in head_idx:
            if isinstance(model.layers[idx], Dense):
                head_dense = model.layers[idx]
                break
        assert head_dense is not None
        w_before = head_dense.weights.data[0][0]

        # Do a forward/backward pass
        x = Tensor([[1.0, 0.0]])
        loss_fn = CrossEntropyLoss()
        output = model.forward(x)
        grad = loss_fn.backward(output, [1])
        model.backward(grad)

        ft.discriminative_step(None)

        w_after = head_dense.weights.data[0][0]
        # Head layer with base_lr=0.1 should visibly change
        assert abs(w_after - w_before) > 1e-10

    def test_lr_mult_effect(self):
        """Lower lr_mult should result in smaller backbone LRs."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ft1 = FineTuner(pm, base_lr=0.01, lr_mult=0.5)
        lrs1 = ft1.set_discriminative_lrs()
        ft2 = FineTuner(pm, base_lr=0.01, lr_mult=0.01)
        lrs2 = ft2.set_discriminative_lrs()
        # With smaller lr_mult, backbone LRs should be smaller
        bb = pm.groups['backbone']
        for idx in bb:
            if idx in lrs1 and idx in lrs2:
                assert lrs2[idx] <= lrs1[idx]


# ============================================================
# Tests: FrozenAwareOptimizer
# ============================================================

class TestFrozenAwareOptimizer:
    def test_skips_frozen(self):
        model = Sequential([
            Dense(2, 4, init='zeros'),
            Activation('relu'),
            Dense(4, 2, init='zeros'),
        ])
        freeze_layer(model.layers[0])
        opt = FrozenAwareOptimizer(Adam(lr=0.1))

        # Do a forward/backward
        x = Tensor([[1.0, 0.0]])
        output = model.forward(x)
        loss_fn = CrossEntropyLoss()
        grad = loss_fn.backward(output, [0])
        model.backward(grad)

        w0_before = model.layers[0].weights.data[0][0]
        opt.step(model.get_trainable_layers())
        w0_after = model.layers[0].weights.data[0][0]
        # Frozen layer should not change
        assert w0_before == w0_after

    def test_updates_unfrozen(self):
        rng = random.Random(42)
        model = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        freeze_layer(model.layers[0])
        opt = FrozenAwareOptimizer(Adam(lr=0.1))

        x = Tensor([[1.0, 0.0]])
        loss_fn = CrossEntropyLoss()
        output = model.forward(x)
        grad = loss_fn.backward(output, [0])
        model.backward(grad)

        w2_before = model.layers[2].weights.data[0][0]
        opt.step(model.get_trainable_layers())
        w2_after = model.layers[2].weights.data[0][0]
        # Unfrozen layer should change
        assert w2_before != w2_after


# ============================================================
# Tests: DomainAdapter
# ============================================================

class TestDomainAdapter:
    def test_mmd_same_distribution(self):
        da = DomainAdapter(method='mmd')
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        mmd = da.compute_mmd(x, x)
        assert mmd < 1e-10

    def test_mmd_different_distributions(self):
        da = DomainAdapter(method='mmd')
        s = Tensor([[0.0, 0.0], [1.0, 1.0]])
        t = Tensor([[10.0, 10.0], [11.0, 11.0]])
        mmd = da.compute_mmd(s, t)
        assert mmd > 0

    def test_mmd_1d(self):
        da = DomainAdapter(method='mmd')
        s = Tensor([1.0, 2.0, 3.0])
        t = Tensor([4.0, 5.0, 6.0])
        mmd = da.compute_mmd(s, t)
        assert mmd > 0

    def test_coral_same(self):
        da = DomainAdapter(method='coral')
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        coral = da.compute_coral(x, x)
        assert coral < 1e-10

    def test_coral_different(self):
        da = DomainAdapter(method='coral')
        s = Tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        t = Tensor([[10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        coral = da.compute_coral(s, t)
        assert coral > 0

    def test_adaptation_loss_mmd(self):
        da = DomainAdapter(method='mmd', lambda_adapt=2.0)
        s = Tensor([[0.0], [1.0]])
        t = Tensor([[5.0], [6.0]])
        loss = da.adaptation_loss(s, t)
        raw_mmd = da.compute_mmd(s, t)
        assert abs(loss - 2.0 * raw_mmd) < 1e-10

    def test_adaptation_loss_coral(self):
        da = DomainAdapter(method='coral', lambda_adapt=1.0)
        s = Tensor([[0.0, 1.0], [1.0, 0.0]])
        t = Tensor([[5.0, 6.0], [6.0, 5.0]])
        loss = da.adaptation_loss(s, t)
        assert loss >= 0

    def test_mmd_gradient(self):
        da = DomainAdapter(method='mmd', lambda_adapt=1.0)
        s = Tensor([[0.0, 0.0], [1.0, 1.0]])
        t = Tensor([[5.0, 5.0], [6.0, 6.0]])
        grad = da.compute_mmd_gradient(s, t)
        assert grad.shape == s.shape

    def test_mmd_gradient_1d(self):
        da = DomainAdapter(method='mmd', lambda_adapt=1.0)
        s = Tensor([1.0, 2.0])
        t = Tensor([5.0, 6.0])
        grad = da.compute_mmd_gradient(s, t)
        assert len(grad.data) == 2

    def test_unknown_method_error(self):
        da = DomainAdapter(method='unknown')
        s = Tensor([[1.0]])
        with pytest.raises(ValueError):
            da.adaptation_loss(s, s)

    def test_lambda_scales_loss(self):
        da1 = DomainAdapter(method='mmd', lambda_adapt=1.0)
        da2 = DomainAdapter(method='mmd', lambda_adapt=10.0)
        s = Tensor([[0.0], [1.0]])
        t = Tensor([[5.0], [6.0]])
        l1 = da1.adaptation_loss(s, t)
        l2 = da2.adaptation_loss(s, t)
        assert abs(l2 - 10.0 * l1) < 1e-10


# ============================================================
# Tests: KnowledgeDistiller
# ============================================================

class TestKnowledgeDistiller:
    def test_soft_softmax(self):
        teacher = make_trained_model()
        student = make_trained_model(seed=99)
        kd = KnowledgeDistiller(teacher, student, temperature=2.0)
        logits = Tensor([1.0, 2.0, 3.0])
        soft = kd.soft_softmax(logits, 2.0)
        assert abs(sum(soft.data) - 1.0) < 1e-6

    def test_soft_softmax_batch(self):
        teacher = make_trained_model()
        student = make_trained_model(seed=99)
        kd = KnowledgeDistiller(teacher, student, temperature=2.0)
        logits = Tensor([[1.0, 2.0], [3.0, 4.0]])
        soft = kd.soft_softmax(logits, 2.0)
        for row in soft.data:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_higher_temp_softer(self):
        teacher = make_trained_model()
        student = make_trained_model(seed=99)
        kd = KnowledgeDistiller(teacher, student)
        logits = Tensor([1.0, 5.0])
        soft1 = kd.soft_softmax(logits, 1.0)
        soft10 = kd.soft_softmax(logits, 10.0)
        # Higher temp -> more uniform -> lower max prob
        assert max(soft10.data) < max(soft1.data)

    def test_soft_cross_entropy(self):
        teacher = make_trained_model()
        student = make_trained_model(seed=99)
        kd = KnowledgeDistiller(teacher, student, temperature=2.0)
        s_logits = Tensor([1.0, 2.0])
        t_logits = Tensor([1.0, 2.0])
        loss = kd.soft_cross_entropy(s_logits, t_logits)
        # Same logits -> KL divergence should be near 0
        assert loss < 0.01

    def test_soft_cross_entropy_different(self):
        teacher = make_trained_model()
        student = make_trained_model(seed=99)
        kd = KnowledgeDistiller(teacher, student, temperature=2.0)
        s_logits = Tensor([1.0, 2.0])
        t_logits = Tensor([5.0, -5.0])
        loss = kd.soft_cross_entropy(s_logits, t_logits)
        assert loss > 0

    def test_soft_cross_entropy_gradient(self):
        teacher = make_trained_model()
        student = make_trained_model(seed=99)
        kd = KnowledgeDistiller(teacher, student, temperature=2.0)
        s_logits = Tensor([1.0, 2.0])
        t_logits = Tensor([1.0, 2.0])
        grad = kd.soft_cross_entropy_gradient(s_logits, t_logits)
        # Same logits -> gradient should be near 0
        assert all(abs(g) < 0.01 for g in grad.data)

    def test_distill_step(self):
        teacher = make_trained_model()
        rng = random.Random(99)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        kd = KnowledgeDistiller(teacher, student, temperature=3.0, alpha=0.5)
        x = Tensor([[1.0, 0.0], [0.0, 1.0]])
        y = [1, 0]
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        total, hard, soft = kd.distill_step(x, y, loss_fn, opt)
        assert total >= 0

    def test_distill_full(self):
        teacher = make_trained_model()
        rng = random.Random(99)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        kd = KnowledgeDistiller(teacher, student, temperature=3.0, alpha=0.5)
        X, Y = make_xor_data(n=30, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        history = kd.distill(X, Y, loss_fn, opt, epochs=10, batch_size=15)
        assert len(history['loss']) == 10
        assert len(history['hard_loss']) == 10
        assert len(history['soft_loss']) == 10

    def test_distill_improves(self):
        """Student should improve via distillation."""
        teacher = make_trained_model()
        rng = random.Random(99)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        kd = KnowledgeDistiller(teacher, student, temperature=3.0, alpha=0.3)
        X, Y = make_xor_data(n=50, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        history = kd.distill(X, Y, loss_fn, opt, epochs=30, batch_size=25)
        # Loss should decrease
        assert history['loss'][-1] < history['loss'][0]

    def test_distill_batch_2d(self):
        teacher = make_trained_model()
        rng = random.Random(99)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        kd = KnowledgeDistiller(teacher, student, temperature=2.0)
        x = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = [1, 0, 0]
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        total, hard, soft = kd.distill_step(x, y, loss_fn, opt)
        assert isinstance(total, float)


# ============================================================
# Tests: ModelRegistry
# ============================================================

class TestModelRegistry:
    def test_register_and_load(self):
        model = make_trained_model()
        reg = ModelRegistry()
        reg.register('xor_model', model, metadata={'task': 'xor'})
        loaded = reg.load('xor_model')
        assert isinstance(loaded, PretrainedModel)

    def test_load_preserves_weights(self):
        model = make_trained_model()
        w_orig = save_weights(model)
        reg = ModelRegistry()
        reg.register('test', model)
        loaded = reg.load('test')
        w_loaded = save_weights(loaded.model)
        for key in w_orig:
            if isinstance(w_orig[key][0], list):
                for i in range(len(w_orig[key])):
                    for j in range(len(w_orig[key][i])):
                        assert abs(w_orig[key][i][j] - w_loaded[key][i][j]) < 1e-10
            else:
                for i in range(len(w_orig[key])):
                    assert abs(w_orig[key][i] - w_loaded[key][i]) < 1e-10

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register('a', make_trained_model())
        reg.register('b', make_trained_model(seed=99))
        names = reg.list_models()
        assert 'a' in names
        assert 'b' in names

    def test_get_metadata(self):
        reg = ModelRegistry()
        reg.register('m', make_trained_model(), metadata={'version': '1.0'})
        meta = reg.get_metadata('m')
        assert meta['version'] == '1.0'

    def test_remove(self):
        reg = ModelRegistry()
        reg.register('m', make_trained_model())
        reg.remove('m')
        assert 'm' not in reg.list_models()

    def test_load_not_found(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.load('nonexistent')

    def test_register_pretrained(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        reg = ModelRegistry()
        reg.register('pm', pm)
        loaded = reg.load('pm')
        assert loaded.groups is not None

    def test_export_import_weights(self):
        model = make_trained_model()
        reg = ModelRegistry()
        reg.register('m', model)
        weights = reg.export_weights('m')
        assert isinstance(weights, dict)
        reg.import_weights('m', weights)

    def test_register_with_dropout(self):
        model = Sequential([
            Dense(2, 4),
            Activation('relu'),
            Dropout(0.5),
            Dense(4, 2),
        ])
        reg = ModelRegistry()
        reg.register('dropout_model', model)
        loaded = reg.load('dropout_model')
        assert len(loaded.model.layers) == 4

    def test_register_with_batchnorm(self):
        model = Sequential([
            Dense(2, 4),
            BatchNorm(4),
            Activation('relu'),
            Dense(4, 2),
        ])
        reg = ModelRegistry()
        reg.register('bn_model', model)
        loaded = reg.load('bn_model')
        assert isinstance(loaded.model.layers[1], BatchNorm)


# ============================================================
# Tests: TransferTrainer
# ============================================================

class TestTransferTrainer:
    def test_phase1_train_head(self):
        model = make_trained_model(output_size=2)
        pm = PretrainedModel.from_sequential(model)
        new_head = [Dense(8, 3)]
        trainer = TransferTrainer(pm, new_head_layers=new_head)
        X, Y = make_spiral_data(n_per_class=20, n_classes=3, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        history = trainer.phase1_train_head(X, Y, loss_fn, opt, epochs=10)
        assert len(history['loss']) == 10

    def test_phase1_freezes_backbone(self):
        model = make_trained_model(output_size=2)
        pm = PretrainedModel.from_sequential(model)
        new_head = [Dense(8, 3)]
        trainer = TransferTrainer(pm, new_head_layers=new_head)

        w_before = model.layers[0].weights.data[0][0]
        X, Y = make_spiral_data(n_per_class=10, n_classes=3, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        trainer.phase1_train_head(X, Y, loss_fn, opt, epochs=5)
        w_after = model.layers[0].weights.data[0][0]
        assert w_before == w_after  # Backbone should not change

    def test_phase2_fine_tune(self):
        model = make_trained_model(output_size=2)
        pm = PretrainedModel.from_sequential(model)
        new_head = [Dense(8, 3)]
        trainer = TransferTrainer(pm, new_head_layers=new_head)
        X, Y = make_spiral_data(n_per_class=20, n_classes=3, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        trainer.phase1_train_head(X, Y, loss_fn, opt, epochs=5)
        history = trainer.phase2_fine_tune(X, Y, loss_fn, base_lr=0.001, epochs=5)
        assert len(history['phase2_loss']) == 5

    def test_phase2_gradual(self):
        model = make_trained_model(output_size=2)
        pm = PretrainedModel.from_sequential(model)
        new_head = [Dense(8, 3)]
        trainer = TransferTrainer(pm, new_head_layers=new_head)
        X, Y = make_spiral_data(n_per_class=20, n_classes=3, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        trainer.phase1_train_head(X, Y, loss_fn, opt, epochs=5)
        history = trainer.phase2_fine_tune(
            X, Y, loss_fn, base_lr=0.001, epochs=10,
            gradual=True, epochs_per_unfreeze=3
        )
        assert len(history['phase2_loss']) == 10

    def test_predict(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        trainer = TransferTrainer(pm)
        x = Tensor([[1.0, 0.0]])
        out = trainer.predict(x)
        assert out is not None

    def test_evaluate(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        trainer = TransferTrainer(pm)
        x = Tensor([[1.0, 0.0], [0.0, 1.0]])
        y = [1, 0]
        loss_fn = CrossEntropyLoss()
        loss = trainer.evaluate(x, y, loss_fn)
        assert isinstance(loss, float)

    def test_no_new_head(self):
        """TransferTrainer works without replacing head."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        trainer = TransferTrainer(pm)
        X, Y = make_xor_data(n=30, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        history = trainer.phase1_train_head(X, Y, loss_fn, opt, epochs=5)
        assert len(history['loss']) == 5


# ============================================================
# Tests: DataAugmenter
# ============================================================

class TestDataAugmenter:
    def test_add_noise(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        noisy = aug.add_noise(x, std=0.1)
        assert noisy.shape == x.shape
        # Should be different from original
        diff = False
        for i in range(2):
            for j in range(2):
                if abs(noisy.data[i][j] - x.data[i][j]) > 1e-10:
                    diff = True
        assert diff

    def test_add_noise_1d(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([1.0, 2.0, 3.0])
        noisy = aug.add_noise(x, std=0.5)
        assert noisy.shape == x.shape

    def test_mixup(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        y = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        x_mix, y_mix = aug.mixup(x, y, alpha=0.2)
        assert x_mix.shape == x.shape
        assert y_mix.shape == y.shape

    def test_mixup_1d(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])
        x_mix, y_mix = aug.mixup(x, y)
        # 1D should return copies
        assert x_mix.shape == x.shape

    def test_cutout(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        cut = aug.cutout(x, n_features=1)
        assert cut.shape == x.shape
        # At least one zero should exist
        has_zero = False
        for row in cut.data:
            if 0.0 in row:
                has_zero = True
        assert has_zero

    def test_cutout_1d(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        cut = aug.cutout(x, n_features=2)
        zeros = sum(1 for v in cut.data if v == 0.0)
        assert zeros >= 1

    def test_random_scale(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        scaled = aug.random_scale(x, low=0.5, high=1.5)
        assert scaled.shape == x.shape

    def test_random_scale_1d(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([1.0, 2.0, 3.0])
        scaled = aug.random_scale(x)
        assert scaled.shape == x.shape

    def test_augment_batch(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = [0, 1]
        x_aug, y_aug = aug.augment_batch(x, y, methods=['noise'])
        # Should have original + augmented
        assert len(x_aug.data) == 4
        assert len(y_aug) == 4

    def test_augment_batch_multiple(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = Tensor([[1.0, 0.0], [0.0, 1.0]])
        x_aug, y_aug = aug.augment_batch(x, y, methods=['noise', 'scale'])
        assert x_aug.shape[0] == 6  # 2 original + 2 noise + 2 scale

    def test_augment_batch_cutout(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0, 3.0]])
        y = [0]
        x_aug, y_aug = aug.augment_batch(x, y, methods=['cutout'], n_cutout=1)
        assert len(x_aug.data) == 2

    def test_beta_sample(self):
        aug = DataAugmenter(seed=42)
        val = aug._beta_sample(2.0, 2.0)
        assert 0.0 <= val <= 1.0

    def test_gamma_sample(self):
        aug = DataAugmenter(seed=42)
        val = aug._gamma_sample(2.0)
        assert val >= 0

    def test_gamma_sample_small_shape(self):
        aug = DataAugmenter(seed=42)
        val = aug._gamma_sample(0.5)
        assert val >= 0


# ============================================================
# Tests: MultiTaskHead
# ============================================================

class TestMultiTaskHead:
    def test_add_task(self):
        backbone = Sequential([Dense(4, 8), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('cls', [Dense(8, 3)], CrossEntropyLoss())
        mt.add_task('reg', [Dense(8, 1)], MSELoss())
        assert 'cls' in mt.heads
        assert 'reg' in mt.heads

    def test_forward_single_task(self):
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('cls', [Dense(8, 3, rng=rng)], CrossEntropyLoss())
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = mt.forward(x, task_name='cls')
        assert out.shape[1] == 3

    def test_forward_all_tasks(self):
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('cls', [Dense(8, 3, rng=rng)], CrossEntropyLoss())
        mt.add_task('reg', [Dense(8, 1, rng=rng)], MSELoss())
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        outputs = mt.forward(x)
        assert 'cls' in outputs
        assert 'reg' in outputs

    def test_train_step(self):
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('cls', [Dense(8, 2, rng=rng)], CrossEntropyLoss())
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        targets = {'cls': [0, 1]}
        opt = Adam(lr=0.01)
        losses = mt.train_step(x, targets, opt)
        assert 'cls' in losses

    def test_multi_task_train(self):
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('a', [Dense(8, 2, rng=rng)], CrossEntropyLoss())
        mt.add_task('b', [Dense(8, 1, rng=rng)], MSELoss())
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        targets = {'a': [0, 1], 'b': Tensor([[1.0], [2.0]])}
        opt = Adam(lr=0.01)
        losses = mt.train_step(x, targets, opt)
        assert 'a' in losses
        assert 'b' in losses

    def test_predict(self):
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('cls', [Dense(8, 2, rng=rng)], CrossEntropyLoss())
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = mt.predict(x, 'cls')
        assert out.shape[1] == 2

    def test_task_weights(self):
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('cls', [Dense(8, 2, rng=rng)], CrossEntropyLoss(), weight=2.0)
        assert mt.task_weights['cls'] == 2.0

    def test_unknown_task_error(self):
        backbone = Sequential([Dense(4, 8), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        with pytest.raises(ValueError):
            mt.forward(x, task_name='nonexistent')


# ============================================================
# Tests: EWC (Elastic Weight Consolidation)
# ============================================================

class TestEWC:
    def test_compute_fisher(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ewc = EWC(pm, lambda_ewc=1000.0)
        X, Y = make_xor_data(n=20, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc.compute_fisher(X, Y, loss_fn, n_samples=10)
        assert ewc._computed

    def test_penalty_at_optimal(self):
        """Penalty should be 0 when params haven't moved."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ewc = EWC(pm, lambda_ewc=1000.0)
        X, Y = make_xor_data(n=20, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc.compute_fisher(X, Y, loss_fn, n_samples=10)
        penalty = ewc.penalty()
        assert abs(penalty) < 1e-10

    def test_penalty_after_change(self):
        """Penalty should increase when params move."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ewc = EWC(pm, lambda_ewc=1000.0)
        X, Y = make_xor_data(n=20, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc.compute_fisher(X, Y, loss_fn, n_samples=10)

        # Modify some weights
        model.layers[0].weights.data[0][0] += 1.0
        penalty = ewc.penalty()
        assert penalty > 0

    def test_penalty_gradient(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ewc = EWC(pm, lambda_ewc=100.0)
        X, Y = make_xor_data(n=20, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc.compute_fisher(X, Y, loss_fn, n_samples=10)

        model.layers[0].weights.data[0][0] += 0.5
        grads = ewc.penalty_gradient()
        assert len(grads) > 0

    def test_penalty_gradient_zero_at_optimal(self):
        """Gradient should be 0 when params at saved values."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ewc = EWC(pm, lambda_ewc=100.0)
        X, Y = make_xor_data(n=20, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc.compute_fisher(X, Y, loss_fn, n_samples=10)
        grads = ewc.penalty_gradient()
        for layer_idx, layer_grads in grads.items():
            for g in layer_grads:
                if len(g.shape) == 1:
                    assert all(abs(v) < 1e-10 for v in g.data)
                else:
                    for row in g.data:
                        assert all(abs(v) < 1e-10 for v in row)

    def test_penalty_not_computed(self):
        model = make_trained_model()
        ewc = EWC(model, lambda_ewc=100.0)
        assert ewc.penalty() == 0.0

    def test_lambda_scales_penalty(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)

        ewc1 = EWC(pm, lambda_ewc=100.0)
        X, Y = make_xor_data(n=20, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc1.compute_fisher(X, Y, loss_fn, n_samples=10)

        ewc2 = EWC(pm, lambda_ewc=200.0)
        ewc2.compute_fisher(X, Y, loss_fn, n_samples=10)

        model.layers[0].weights.data[0][0] += 1.0
        p1 = ewc1.penalty()
        p2 = ewc2.penalty()
        assert abs(p2 - 2.0 * p1) < 1e-6


# ============================================================
# Tests: ProgressiveNet
# ============================================================

class TestProgressiveNet:
    def test_add_column(self):
        pnet = ProgressiveNet()
        idx = pnet.add_column([4, 8, 2])
        assert idx == 0
        assert pnet.num_columns() == 1

    def test_add_multiple_columns(self):
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2])
        pnet.add_column([4, 8, 3])
        assert pnet.num_columns() == 2

    def test_forward_single_column(self):
        rng = random.Random(42)
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2], rng=rng)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = pnet.forward(x, column_idx=0)
        assert out.shape[1] == 2

    def test_forward_second_column(self):
        rng = random.Random(42)
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2], rng=rng)
        pnet.add_column([4, 8, 3], rng=rng)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = pnet.forward(x, column_idx=1)
        assert out.shape[1] == 3

    def test_forward_default_latest(self):
        rng = random.Random(42)
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2], rng=rng)
        pnet.add_column([4, 8, 5], rng=rng)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = pnet.forward(x)
        assert out.shape[1] == 5

    def test_previous_columns_frozen(self):
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2])
        pnet.add_column([4, 8, 3])
        # First column should be frozen
        for layer in pnet.columns[0].layers:
            assert is_frozen(layer)

    def test_get_column(self):
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2])
        col = pnet.get_column(0)
        assert isinstance(col, Sequential)

    def test_three_columns(self):
        rng = random.Random(42)
        pnet = ProgressiveNet()
        pnet.add_column([4, 8, 2], rng=rng)
        pnet.add_column([4, 8, 3], rng=rng)
        pnet.add_column([4, 8, 4], rng=rng)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        out = pnet.forward(x, column_idx=2)
        assert out.shape[1] == 4
        assert pnet.num_columns() == 3


# ============================================================
# Tests: Integration / End-to-End
# ============================================================

class TestIntegration:
    def test_full_transfer_workflow(self):
        """Test complete transfer learning workflow."""
        # Step 1: Train source model
        source_model = make_trained_model(hidden=8, output_size=2)

        # Step 2: Create pretrained model
        pm = PretrainedModel.from_sequential(source_model)

        # Step 3: Replace head for new task
        rng = random.Random(42)
        new_head = [Dense(8, 3, rng=rng)]
        pm2 = pm.replace_head(new_head)

        # Step 4: Train with transfer
        trainer = TransferTrainer(pm2)
        X, Y = make_spiral_data(n_per_class=15, n_classes=3, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)

        # Phase 1: Train head
        h1 = trainer.phase1_train_head(X, Y, loss_fn, opt, epochs=10)
        assert len(h1['loss']) == 10

        # Phase 2: Fine-tune
        h2 = trainer.phase2_fine_tune(X, Y, loss_fn, base_lr=0.001, epochs=5)
        assert len(h2['phase2_loss']) == 5

    def test_registry_transfer_workflow(self):
        """Test registry-based transfer workflow."""
        # Train and register
        model = make_trained_model()
        reg = ModelRegistry()
        reg.register('source', model, metadata={'task': 'xor'})

        # Load and transfer
        loaded = reg.load('source')
        rng = random.Random(42)
        new_head = [Dense(8, 3, rng=rng)]
        transferred = loaded.replace_head(new_head)

        X, Y = make_spiral_data(n_per_class=10, n_classes=3, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)

        transferred.unfreeze_all()
        fit(transferred.model, X, Y, loss_fn, opt, epochs=5)

    def test_feature_extraction_pipeline(self):
        """Test feature extraction -> new classifier."""
        # Train source model
        source = make_trained_model(hidden=8)
        fe = FeatureExtractor(source, extract_at=[2])

        # Extract features for new task
        X_new = Tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        features = fe.extract_single(X_new)
        assert features.shape[0] == 4

    def test_distillation_pipeline(self):
        """Test knowledge distillation pipeline."""
        teacher = make_trained_model(hidden=16)
        rng = random.Random(42)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])

        kd = KnowledgeDistiller(teacher, student, temperature=4.0, alpha=0.3)
        X, Y = make_xor_data(n=50, seed=42)
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        history = kd.distill(X, Y, loss_fn, opt, epochs=20, batch_size=25)

        # Student should learn something
        assert history['loss'][-1] < history['loss'][0]

    def test_domain_adaptation_pipeline(self):
        """Test domain adaptation with feature alignment."""
        model = make_trained_model()
        fe = FeatureExtractor(model, extract_at=[2])

        source = Tensor([[0.0, 1.0], [1.0, 0.0]])
        target = Tensor([[0.5, 0.5], [0.3, 0.7]])

        s_feat = fe.extract_single(source)
        t_feat = fe.extract_single(target)

        da = DomainAdapter(method='mmd')
        loss = da.adaptation_loss(s_feat, t_feat)
        assert isinstance(loss, float)

    def test_augmented_training(self):
        """Test training with data augmentation."""
        aug = DataAugmenter(seed=42)
        X, Y = make_xor_data(n=20, seed=42)
        X_aug, Y_aug = aug.augment_batch(X, Y, methods=['noise', 'scale'])

        rng = random.Random(42)
        model = Sequential([
            Dense(2, 8, rng=rng),
            Activation('relu'),
            Dense(8, 2, rng=rng),
        ])
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        history = fit(model, X_aug, Y_aug, loss_fn, opt, epochs=10)
        assert len(history['loss']) == 10

    def test_ewc_prevents_forgetting(self):
        """EWC penalty should resist parameter changes."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        ewc = EWC(pm, lambda_ewc=10000.0)
        X, Y = make_xor_data(n=30, seed=42)
        loss_fn = CrossEntropyLoss()
        ewc.compute_fisher(X, Y, loss_fn, n_samples=15)

        # Check penalty is 0 initially
        assert abs(ewc.penalty()) < 1e-10

        # Small weight change -> positive penalty
        model.layers[0].weights.data[0][0] += 0.01
        assert ewc.penalty() > 0

    def test_multi_task_learning(self):
        """Test multi-task learning with shared backbone."""
        rng = random.Random(42)
        backbone = Sequential([Dense(4, 8, rng=rng), Activation('relu')])
        mt = MultiTaskHead(backbone, 8)
        mt.add_task('classify', [Dense(8, 2, rng=rng)], CrossEntropyLoss())
        mt.add_task('regress', [Dense(8, 1, rng=rng)], MSELoss())

        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        targets = {
            'classify': [0, 1],
            'regress': Tensor([[1.0], [2.0]])
        }
        opt = Adam(lr=0.01)

        # Train for several steps
        for _ in range(5):
            losses = mt.train_step(x, targets, opt)
        assert 'classify' in losses
        assert 'regress' in losses

    def test_progressive_net_multi_task(self):
        """Test progressive network across multiple tasks."""
        rng = random.Random(42)
        pnet = ProgressiveNet()

        # Task 1
        pnet.add_column([2, 8, 2], rng=rng)
        x = Tensor([[1.0, 0.0], [0.0, 1.0]])
        out1 = pnet.forward(x, column_idx=0)
        assert out1.shape == (2, 2)

        # Task 2 (different output)
        pnet.add_column([2, 8, 3], rng=rng)
        out2 = pnet.forward(x, column_idx=1)
        assert out2.shape == (2, 3)

    def test_frozen_aware_with_transfer(self):
        """Test FrozenAwareOptimizer in transfer scenario."""
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm.freeze_group('backbone')

        frozen_before = count_frozen_params(model)
        trainable_before = count_trainable_params(model)
        assert frozen_before > 0
        assert trainable_before > 0

        opt = FrozenAwareOptimizer(Adam(lr=0.01))
        x = Tensor([[1.0, 0.0]])
        output = model.forward(x)
        loss_fn = CrossEntropyLoss()
        grad = loss_fn.backward(output, [0])
        model.backward(grad)
        opt.step(model.get_trainable_layers())

    def test_coral_then_transfer(self):
        """CORAL domain adaptation then transfer."""
        da = DomainAdapter(method='coral', lambda_adapt=0.5)
        source = Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        target = Tensor([[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]])
        coral_loss = da.adaptation_loss(source, target)
        assert coral_loss > 0

    def test_save_load_transfer(self):
        """Save trained model, load, transfer to new task."""
        model = make_trained_model()
        weights = save_weights(model)

        # Create new model with same architecture
        rng = random.Random(42)
        new_model = Sequential([
            Dense(2, 8, rng=rng),
            Activation('relu'),
            Dense(8, 8, rng=rng),
            Activation('relu'),
            Dense(8, 2, rng=rng),
        ])
        load_weights(new_model, weights)

        # Transfer to new task
        pm = PretrainedModel.from_sequential(new_model)
        new_head = [Dense(8, 3)]
        pm2 = pm.replace_head(new_head)
        x = Tensor([[1.0, 0.0]])
        out = pm2.forward(x)
        assert out.shape[1] == 3


# ============================================================
# Tests: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_groups(self):
        model = make_trained_model()
        pm = PretrainedModel(model, groups={})
        pm.freeze_all()
        pm.unfreeze_all()

    def test_single_layer_model(self):
        model = Sequential([Dense(2, 3)])
        pm = PretrainedModel.from_sequential(model)
        assert 'backbone' in pm.groups
        assert 'head' in pm.groups

    def test_freeze_no_params(self):
        layer = Activation('relu')
        freeze_layer(layer)
        assert is_frozen(layer)

    def test_extractor_empty_model(self):
        model = Sequential([Activation('relu')])
        fe = FeatureExtractor(model, extract_at=[0])
        x = Tensor([1.0, 2.0])
        features = fe.extract(x)
        assert 0 in features

    def test_distiller_alpha_zero(self):
        """Alpha=0 should use only soft loss."""
        teacher = make_trained_model()
        rng = random.Random(42)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        kd = KnowledgeDistiller(teacher, student, temperature=3.0, alpha=0.0)
        x = Tensor([[1.0, 0.0]])
        y = [0]
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        total, hard, soft = kd.distill_step(x, y, loss_fn, opt)
        # With alpha=0, total should be soft_loss
        assert abs(total - soft) < 1e-6

    def test_distiller_alpha_one(self):
        """Alpha=1 should use only hard loss."""
        teacher = make_trained_model()
        rng = random.Random(42)
        student = Sequential([
            Dense(2, 4, rng=rng),
            Activation('relu'),
            Dense(4, 2, rng=rng),
        ])
        kd = KnowledgeDistiller(teacher, student, temperature=3.0, alpha=1.0)
        x = Tensor([[1.0, 0.0]])
        y = [0]
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        total, hard, soft = kd.distill_step(x, y, loss_fn, opt)
        assert abs(total - hard) < 1e-6

    def test_mmd_zero_distance(self):
        da = DomainAdapter(method='mmd')
        x = Tensor([[5.0, 5.0]])
        assert da.compute_mmd(x, x) < 1e-10

    def test_coral_1d_returns_zero(self):
        da = DomainAdapter(method='coral')
        x = Tensor([1.0, 2.0])
        assert da.compute_coral(x, x) == 0.0

    def test_registry_overwrite(self):
        reg = ModelRegistry()
        m1 = make_trained_model(seed=1)
        m2 = make_trained_model(seed=2)
        reg.register('model', m1)
        reg.register('model', m2)
        # Should have latest
        assert len(reg.list_models()) == 1

    def test_progressive_net_single_column(self):
        rng = random.Random(42)
        pnet = ProgressiveNet()
        pnet.add_column([2, 4, 2], rng=rng)
        x = Tensor([[1.0, 0.0]])
        out = pnet.forward(x)
        assert out.shape[1] == 2

    def test_augmenter_zero_std(self):
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 2.0]])
        noisy = aug.add_noise(x, std=0.0)
        assert abs(noisy.data[0][0] - 1.0) < 1e-10

    def test_ewc_with_plain_model(self):
        """EWC works with plain Sequential (not PretrainedModel)."""
        model = make_trained_model()
        ewc = EWC(model, lambda_ewc=100.0)
        X, Y = make_xor_data(n=20, seed=42)
        ewc.compute_fisher(X, Y, CrossEntropyLoss(), n_samples=5)
        assert ewc.penalty() < 1e-10

    def test_replace_head_preserves_metadata(self):
        model = make_trained_model()
        pm = PretrainedModel.from_sequential(model)
        pm._metadata['source_task'] = 'xor'
        pm2 = pm.replace_head([Dense(8, 3)])
        assert pm2._metadata['source_task'] == 'xor'

    def test_fine_tuner_no_backbone_group(self):
        """FineTuner works without backbone group."""
        model = Sequential([Dense(2, 4), Activation('relu'), Dense(4, 2)])
        pm = PretrainedModel(model, groups={'all': [0, 1, 2]})
        ft = FineTuner(pm, base_lr=0.001)
        lrs = ft.set_discriminative_lrs()
        assert len(lrs) > 0

    def test_mixup_different_sizes(self):
        """Mixup with different batch sizes."""
        aug = DataAugmenter(seed=42)
        x = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        x_mix, y_mix = aug.mixup(x, y, alpha=0.5)
        assert x_mix.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
