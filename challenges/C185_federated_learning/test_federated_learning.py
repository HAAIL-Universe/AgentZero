"""
Tests for C185: Federated Learning
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from federated_learning import (
    DenseLayer, FederatedModel, FederatedClient, FederatedServer,
    PersonalizedClient, AsyncFederatedServer, SecureAggregator,
    FederatedDiagnostics,
    partition_iid, partition_label_skew, partition_dirichlet, partition_quantity_skew,
    aggregate_fedavg, aggregate_fedsgd, aggregate_trimmed_mean, aggregate_krum,
    aggregate_median,
    compress_topk, decompress_topk, compress_quantize, decompress_quantize,
    compute_client_drift,
    make_classification_data, make_regression_data,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def seed():
    np.random.seed(42)

@pytest.fixture
def classification_data(seed):
    return make_classification_data(300, 5, 3, seed=42)

@pytest.fixture
def simple_model(seed):
    return FederatedModel([5, 16, 3], ["relu", "none"])

@pytest.fixture
def binary_model(seed):
    return FederatedModel([5, 8, 2], ["relu", "none"])


# ============================================================
# DenseLayer
# ============================================================

class TestDenseLayer:
    def test_forward_shape(self, seed):
        layer = DenseLayer(5, 3, "relu")
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        assert out.shape == (10, 3)

    def test_relu_nonnegative(self, seed):
        layer = DenseLayer(5, 3, "relu")
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        assert np.all(out >= 0)

    def test_sigmoid_range(self, seed):
        layer = DenseLayer(5, 3, "sigmoid")
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_softmax_sums_to_one(self, seed):
        layer = DenseLayer(5, 3, "softmax")
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        np.testing.assert_allclose(np.sum(out, axis=1), np.ones(10), atol=1e-6)

    def test_none_activation(self, seed):
        layer = DenseLayer(5, 3, "none")
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        expected = x @ layer.W + layer.b
        np.testing.assert_allclose(out, expected, atol=1e-10)

    def test_backward_shapes(self, seed):
        layer = DenseLayer(5, 3, "relu")
        x = np.random.randn(10, 5)
        out = layer.forward(x)
        grad_out = np.random.randn(10, 3)
        grad_in, grad_W, grad_b = layer.backward(grad_out)
        assert grad_in.shape == (10, 5)
        assert grad_W.shape == (5, 3)
        assert grad_b.shape == (3,)


# ============================================================
# FederatedModel
# ============================================================

class TestFederatedModel:
    def test_forward(self, simple_model, seed):
        x = np.random.randn(10, 5)
        out = simple_model.forward(x)
        assert out.shape == (10, 3)

    def test_predict(self, simple_model, seed):
        x = np.random.randn(10, 5)
        preds = simple_model.predict(x)
        assert preds.shape == (10,)
        assert all(p in [0, 1, 2] for p in preds)

    def test_get_set_params(self, simple_model, seed):
        params = simple_model.get_params()
        assert len(params) == 4  # W1, b1, W2, b2
        assert params[0].shape == (5, 16)
        assert params[1].shape == (16,)
        # Modify and set back
        modified = [p * 2 for p in params]
        simple_model.set_params(modified)
        new_params = simple_model.get_params()
        for orig, new in zip(modified, new_params):
            np.testing.assert_allclose(orig, new)

    def test_copy_independence(self, simple_model, seed):
        copied = simple_model.copy()
        # Modify original
        params = simple_model.get_params()
        params[0] *= 0
        simple_model.set_params(params)
        # Copy should be unchanged
        copy_params = copied.get_params()
        assert np.any(copy_params[0] != 0)

    def test_compute_loss(self, simple_model, seed):
        x = np.random.randn(20, 5)
        y = np.array([0, 1, 2] * 6 + [0, 1])
        loss = simple_model.compute_loss(x, y)
        assert isinstance(loss, float)
        assert loss > 0

    def test_evaluate(self, simple_model, seed):
        x = np.random.randn(20, 5)
        y = np.array([0, 1, 2] * 6 + [0, 1])
        result = simple_model.evaluate(x, y)
        assert "accuracy" in result
        assert "loss" in result
        assert 0 <= result["accuracy"] <= 1

    def test_train_step_reduces_loss(self, seed):
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        loss1 = model.compute_loss(x, y)
        for _ in range(20):
            model.train_step(x, y, lr=0.01)
        loss2 = model.compute_loss(x, y)
        assert loss2 < loss1

    def test_train_step_with_proximal(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        global_params = model.get_params()
        x = np.random.randn(20, 5)
        y = np.random.randint(0, 3, 20)
        loss = model.train_step(x, y, lr=0.01, proximal_term=0.1,
                                global_params=global_params)
        assert isinstance(loss, float)

    def test_mse_loss(self, seed):
        model = FederatedModel([5, 8, 1], ["relu", "none"])
        x = np.random.randn(20, 5)
        y = np.random.randn(20)
        loss = model.compute_loss(x, y, loss_type="mse")
        assert isinstance(loss, float)
        assert loss >= 0


# ============================================================
# Data Partitioning
# ============================================================

class TestPartitioning:
    def test_iid_partition(self, classification_data):
        x, y = classification_data
        parts = partition_iid(x, y, 5)
        assert len(parts) == 5
        total = sum(len(p[0]) for p in parts)
        assert total == len(x)

    def test_label_skew(self, classification_data):
        x, y = classification_data
        parts = partition_label_skew(x, y, 5, labels_per_client=2)
        assert len(parts) == 5
        # Each client should have limited labels
        for px, py in parts:
            assert len(np.unique(py)) <= 3  # May get some overlap

    def test_dirichlet_partition(self, classification_data):
        x, y = classification_data
        parts = partition_dirichlet(x, y, 5, alpha=0.5)
        assert len(parts) == 5
        for px, py in parts:
            assert len(px) > 0

    def test_dirichlet_low_alpha_skewed(self, classification_data):
        x, y = classification_data
        parts = partition_dirichlet(x, y, 5, alpha=0.01)
        # Low alpha = more skewed, some clients may have very few classes
        assert len(parts) == 5

    def test_quantity_skew(self, classification_data):
        x, y = classification_data
        parts = partition_quantity_skew(x, y, 5, min_ratio=0.05)
        assert len(parts) == 5
        sizes = [len(p[0]) for p in parts]
        # Should have variation
        assert max(sizes) > min(sizes) or len(x) < 10


# ============================================================
# FederatedClient
# ============================================================

class TestFederatedClient:
    def test_create_client(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, simple_model)
        assert client.client_id == 0
        assert client.num_samples == 50

    def test_receive_global_model(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, simple_model)
        new_params = [np.zeros_like(p) for p in simple_model.get_params()]
        client.receive_global_model(new_params)
        # Client model should now be all zeros
        for p in client.model.get_params():
            np.testing.assert_allclose(p, 0)

    def test_train_local(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, simple_model, local_epochs=2)
        result = client.train_local()
        assert "delta" in result
        assert "num_samples" in result
        assert "avg_loss" in result
        assert result["num_samples"] == 50
        # Delta should be non-zero
        total_delta = sum(np.sum(np.abs(d)) for d in result["delta"])
        assert total_delta > 0

    def test_train_local_with_clipping(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, simple_model, local_epochs=2)
        result = client.train_local(max_grad_norm=1.0)
        total_norm = np.sqrt(sum(np.sum(d ** 2) for d in result["delta"]))
        assert total_norm <= 1.0 + 1e-6

    def test_train_local_with_dp_noise(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client1 = FederatedClient(0, x, y, simple_model, local_epochs=1)
        client2 = FederatedClient(1, x, y, simple_model, local_epochs=1)
        # Same data, but DP noise should cause different results
        np.random.seed(42)
        r1 = client1.train_local(dp_noise_scale=1.0)
        np.random.seed(43)
        r2 = client2.train_local(dp_noise_scale=1.0)
        # Deltas should differ due to noise
        diff = sum(np.sum(np.abs(d1 - d2)) for d1, d2 in zip(r1["delta"], r2["delta"]))
        assert diff > 0

    def test_evaluate_local(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, simple_model)
        result = client.evaluate_local()
        assert "accuracy" in result
        assert "loss" in result

    def test_train_local_with_fedprox(self, simple_model, seed):
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, simple_model, local_epochs=2)
        global_params = simple_model.get_params()
        result = client.train_local(proximal_term=0.1, global_params=global_params)
        assert result["avg_loss"] > 0


# ============================================================
# Aggregation Strategies
# ============================================================

class TestAggregation:
    def _make_updates(self, seed_val=42):
        np.random.seed(seed_val)
        global_params = [np.random.randn(5, 3), np.random.randn(3)]
        updates = []
        for i in range(5):
            delta = [np.random.randn(*p.shape) * 0.1 for p in global_params]
            updates.append({
                "client_id": i,
                "delta": delta,
                "num_samples": 100 + i * 10,
                "avg_loss": 1.0 - i * 0.1,
            })
        return updates, global_params

    def test_fedavg(self):
        updates, gp = self._make_updates()
        result = aggregate_fedavg(updates, gp)
        assert len(result) == len(gp)
        for r, g in zip(result, gp):
            assert r.shape == g.shape

    def test_fedavg_weighted(self):
        """Clients with more data should have more influence."""
        np.random.seed(42)
        gp = [np.zeros((3, 2)), np.zeros(2)]
        # Client 0: push positive, 10 samples
        # Client 1: push negative, 90 samples
        u0 = {"client_id": 0, "delta": [np.ones((3, 2)), np.ones(2)],
               "num_samples": 10, "avg_loss": 1.0}
        u1 = {"client_id": 1, "delta": [-np.ones((3, 2)), -np.ones(2)],
               "num_samples": 90, "avg_loss": 1.0}
        result = aggregate_fedavg([u0, u1], gp)
        # Should be closer to client 1's update (negative)
        assert np.mean(result[0]) < 0

    def test_fedsgd(self):
        updates, gp = self._make_updates()
        result = aggregate_fedsgd(updates, gp, server_lr=0.5)
        assert len(result) == len(gp)

    def test_trimmed_mean(self):
        updates, gp = self._make_updates()
        result = aggregate_trimmed_mean(updates, gp, trim_ratio=0.2)
        assert len(result) == len(gp)

    def test_trimmed_mean_robust_to_outlier(self):
        np.random.seed(42)
        gp = [np.zeros((3,))]
        updates = []
        for i in range(5):
            updates.append({"client_id": i, "delta": [np.ones(3) * 0.1],
                            "num_samples": 100, "avg_loss": 1.0})
        # Add outlier
        updates.append({"client_id": 5, "delta": [np.ones(3) * 100.0],
                        "num_samples": 100, "avg_loss": 1.0})
        updates.append({"client_id": 6, "delta": [np.ones(3) * -100.0],
                        "num_samples": 100, "avg_loss": 1.0})
        result = aggregate_trimmed_mean(updates, gp, trim_ratio=0.15)
        # Should be close to 0.1, not pulled by outliers
        assert np.all(np.abs(result[0] - 0.1) < 10)

    def test_krum(self):
        updates, gp = self._make_updates()
        result = aggregate_krum(updates, gp, num_byzantine=1)
        assert len(result) == len(gp)

    def test_krum_rejects_outlier(self):
        np.random.seed(42)
        gp = [np.zeros((3,))]
        updates = []
        for i in range(5):
            updates.append({"client_id": i, "delta": [np.ones(3) * 0.1],
                            "num_samples": 100, "avg_loss": 1.0})
        # Byzantine client with extreme update
        updates.append({"client_id": 5, "delta": [np.ones(3) * 1000.0],
                        "num_samples": 100, "avg_loss": 1.0})
        result = aggregate_krum(updates, gp, num_byzantine=1)
        # Krum should select a non-Byzantine client
        assert np.all(np.abs(result[0] - 0.1) < 1.0)

    def test_median(self):
        updates, gp = self._make_updates()
        result = aggregate_median(updates, gp)
        assert len(result) == len(gp)

    def test_median_robust(self):
        np.random.seed(42)
        gp = [np.zeros((3,))]
        updates = []
        for i in range(5):
            updates.append({"client_id": i, "delta": [np.ones(3) * 0.1],
                            "num_samples": 100, "avg_loss": 1.0})
        updates.append({"client_id": 5, "delta": [np.ones(3) * 999.0],
                        "num_samples": 100, "avg_loss": 1.0})
        result = aggregate_median(updates, gp)
        # Median should be 0.1 (majority)
        assert np.all(np.abs(result[0] - 0.1) < 0.5)


# ============================================================
# Compression
# ============================================================

class TestCompression:
    def test_topk_compress_decompress(self, seed):
        delta = [np.random.randn(10, 5), np.random.randn(5)]
        compressed = compress_topk(delta, k_ratio=0.5)
        decompressed = decompress_topk(compressed)
        assert len(decompressed) == 2
        assert decompressed[0].shape == (10, 5)
        assert decompressed[1].shape == (5,)

    def test_topk_sparsity(self, seed):
        delta = [np.random.randn(100)]
        compressed = compress_topk(delta, k_ratio=0.1)
        decompressed = decompress_topk(compressed)
        # 90% should be zero
        zeros = np.sum(decompressed[0] == 0)
        assert zeros >= 89

    def test_topk_preserves_largest(self, seed):
        d = np.zeros(100)
        d[50] = 10.0
        d[75] = -8.0
        delta = [d]
        compressed = compress_topk(delta, k_ratio=0.02)
        decompressed = decompress_topk(compressed)
        # Should keep the largest value
        assert decompressed[0][50] == 10.0 or decompressed[0][75] == -8.0

    def test_quantize_compress_decompress(self, seed):
        delta = [np.random.randn(10, 5), np.random.randn(5)]
        compressed = compress_quantize(delta, bits=8)
        decompressed = decompress_quantize(compressed, bits=8)
        assert len(decompressed) == 2
        # Should be close to original
        for orig, dec in zip(delta, decompressed):
            np.testing.assert_allclose(orig, dec, atol=0.05)

    def test_quantize_zero_range(self, seed):
        delta = [np.ones((5, 3)) * 3.0]
        compressed = compress_quantize(delta)
        decompressed = decompress_quantize(compressed)
        np.testing.assert_allclose(decompressed[0], 3.0, atol=1e-6)


# ============================================================
# Secure Aggregation
# ============================================================

class TestSecureAggregation:
    def test_masks_cancel(self, seed):
        sa = SecureAggregator(3)
        shapes = [(5, 3), (3,)]
        sa.generate_masks(shapes)
        # Sum of all masks should be zero
        for p in range(len(shapes)):
            total = sum(sa.masks[i][p] for i in range(3))
            np.testing.assert_allclose(total, 0, atol=1e-10)

    def test_masked_aggregation_equals_unmasked(self, seed):
        sa = SecureAggregator(3)
        deltas = [[np.random.randn(5, 3), np.random.randn(3)] for _ in range(3)]
        shapes = [(5, 3), (3,)]
        sa.generate_masks(shapes)

        masked = [sa.mask_update(i, deltas[i]) for i in range(3)]
        result = sa.aggregate_masked(masked)

        # Should equal simple average of unmasked deltas
        expected = [np.mean([d[p] for d in deltas], axis=0) for p in range(2)]
        for r, e in zip(result, expected):
            np.testing.assert_allclose(r, e, atol=1e-10)


# ============================================================
# FederatedServer
# ============================================================

class TestFederatedServer:
    def _setup_server(self, num_clients=5, aggregation="fedavg"):
        np.random.seed(42)
        x, y = make_classification_data(300, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model, aggregation=aggregation)
        parts = partition_iid(x, y, num_clients)
        for i, (px, py) in enumerate(parts):
            client = FederatedClient(i, px, py, model, lr=0.01, local_epochs=2)
            server.add_client(client)
        return server, x, y

    def test_add_clients(self):
        server, x, y = self._setup_server()
        assert len(server.clients) == 5

    def test_select_clients_full(self):
        server, x, y = self._setup_server()
        selected = server.select_clients()
        assert len(selected) == 5

    def test_select_clients_partial(self):
        np.random.seed(42)
        server, x, y = self._setup_server()
        server.client_fraction = 0.4
        selected = server.select_clients()
        assert len(selected) == 2  # 0.4 * 5 = 2

    def test_run_round(self):
        server, x, y = self._setup_server()
        result = server.run_round()
        assert "round" in result
        assert result["round"] == 1
        assert "avg_client_loss" in result

    def test_multiple_rounds_improve(self):
        server, x, y = self._setup_server()
        eval_before = server.evaluate_global(x, y)
        for _ in range(10):
            server.run_round()
        eval_after = server.evaluate_global(x, y)
        assert eval_after["loss"] < eval_before["loss"]

    def test_evaluate_global(self):
        server, x, y = self._setup_server()
        result = server.evaluate_global(x, y)
        assert "accuracy" in result
        assert "loss" in result

    def test_run_training(self):
        server, x, y = self._setup_server()
        results = server.run_training(5, test_x=x, test_y=y, eval_every=2)
        assert len(results) == 5
        # Rounds 2 and 4 should have accuracy
        assert "accuracy" in results[1]
        assert "accuracy" in results[3]

    def test_fedavg_aggregation(self):
        server, x, y = self._setup_server(aggregation="fedavg")
        server.run_round()
        assert server.current_round == 1

    def test_fedsgd_aggregation(self):
        server, x, y = self._setup_server(aggregation="fedsgd")
        server.run_round()
        assert server.current_round == 1

    def test_trimmed_mean_aggregation(self):
        server, x, y = self._setup_server(aggregation="trimmed_mean")
        server.run_round()
        assert server.current_round == 1

    def test_krum_aggregation(self):
        server, x, y = self._setup_server(aggregation="krum")
        server.run_round()
        assert server.current_round == 1

    def test_median_aggregation(self):
        server, x, y = self._setup_server(aggregation="median")
        server.run_round()
        assert server.current_round == 1

    def test_round_history(self):
        server, x, y = self._setup_server()
        server.run_round()
        server.run_round()
        assert len(server.round_history) == 2
        assert server.round_history[0]["round"] == 1
        assert server.round_history[1]["round"] == 2

    def test_compression_topk(self):
        server, x, y = self._setup_server()
        result = server.run_round(compression="topk", compression_ratio=0.5)
        assert result["round"] == 1

    def test_compression_quantize(self):
        server, x, y = self._setup_server()
        result = server.run_round(compression="quantize")
        assert result["round"] == 1

    def test_fedprox_round(self):
        server, x, y = self._setup_server()
        result = server.run_round(proximal_term=0.1)
        assert result["round"] == 1

    def test_dp_round(self):
        server, x, y = self._setup_server()
        result = server.run_round(dp_noise_scale=0.01)
        assert result["round"] == 1

    def test_grad_clip_round(self):
        server, x, y = self._setup_server()
        result = server.run_round(max_grad_norm=1.0)
        assert result["round"] == 1


# ============================================================
# Client Drift
# ============================================================

class TestClientDrift:
    def test_drift_zero_initially(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        x, y = make_classification_data(100, 5, 3, seed=42)
        parts = partition_iid(x, y, 3)
        clients = [FederatedClient(i, px, py, model) for i, (px, py) in enumerate(parts)]
        # Before any local training, all clients have the same params
        drift = compute_client_drift(clients, model.get_params())
        for d in drift["per_client"]:
            assert abs(d["drift"]) < 1e-6

    def test_drift_increases_after_training(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        x, y = make_classification_data(100, 5, 3, seed=42)
        parts = partition_iid(x, y, 3)
        clients = [FederatedClient(i, px, py, model, local_epochs=5)
                    for i, (px, py) in enumerate(parts)]
        global_params = model.get_params()
        for c in clients:
            c.train_local()
        drift = compute_client_drift(clients, global_params)
        assert drift["mean_drift"] > 0
        assert drift["max_drift"] > 0


# ============================================================
# Personalization
# ============================================================

class TestPersonalization:
    def test_personalized_client(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = PersonalizedClient(0, x, y, model, personalization_epochs=3)
        client.personalize()
        assert client.personalized_model is not None

    def test_personalized_evaluation(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = PersonalizedClient(0, x, y, model, personalization_epochs=3)
        result = client.evaluate_personalized()
        assert "accuracy" in result
        assert "loss" in result

    def test_personalized_differs_from_global(self, seed):
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        x = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        client = PersonalizedClient(0, x, y, model, personalization_epochs=10)
        client.personalize()
        global_params = client.model.get_params()
        personal_params = client.personalized_model.get_params()
        diff = sum(np.sum(np.abs(g - p)) for g, p in zip(global_params, personal_params))
        assert diff > 0


# ============================================================
# Async Federated Learning
# ============================================================

class TestAsyncFL:
    def test_async_server_create(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        server = AsyncFederatedServer(model, staleness_penalty=0.5)
        assert server.staleness_penalty == 0.5

    def test_async_update(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        server = AsyncFederatedServer(model, staleness_penalty=0.5)
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client = FederatedClient(0, x, y, model, local_epochs=1)
        update = client.train_local()
        result = server.apply_async_update(update)
        assert "staleness" in result
        assert "weight" in result
        assert result["weight"] == 1.0  # First update, no staleness

    def test_async_staleness_penalty(self, seed):
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        server = AsyncFederatedServer(model, staleness_penalty=1.0)
        x = np.random.randn(50, 5)
        y = np.random.randint(0, 3, 50)
        client0 = FederatedClient(0, x, y, model, local_epochs=1)
        client1 = FederatedClient(1, x, y, model, local_epochs=1)
        # Client 0 updates at round 0
        update0 = client0.train_local()
        r0 = server.apply_async_update(update0)
        # Client 1 updates -- round advances
        update1 = client1.train_local()
        r1 = server.apply_async_update(update1)
        # Client 0 updates again with staleness
        update0b = client0.train_local()
        r0b = server.apply_async_update(update0b)
        assert r0b["staleness"] > 0
        assert r0b["weight"] < 1.0


# ============================================================
# Diagnostics
# ============================================================

class TestDiagnostics:
    def test_record_round(self):
        diag = FederatedDiagnostics()
        diag.record_round({"round": 1, "avg_client_loss": 2.0, "client_ids": [0, 1]})
        diag.record_round({"round": 2, "avg_client_loss": 1.5, "client_ids": [0, 2]})
        assert len(diag.round_data) == 2

    def test_convergence_rate(self):
        diag = FederatedDiagnostics()
        for i in range(5):
            diag.record_round({"avg_client_loss": 2.0 * (0.8 ** i)})
        rate = diag.convergence_rate()
        assert rate is not None
        assert 0.7 < rate < 0.9

    def test_participation_stats(self):
        diag = FederatedDiagnostics()
        diag.record_round({"round": 1, "client_ids": [0, 1, 2]})
        diag.record_round({"round": 2, "client_ids": [0, 1, 3]})
        diag.record_round({"round": 3, "client_ids": [0, 2, 3]})
        stats = diag.participation_stats()
        assert stats["unique_clients"] == 4
        assert stats["total_rounds"] == 3

    def test_loss_summary(self):
        diag = FederatedDiagnostics()
        for i in range(5):
            diag.record_round({"avg_client_loss": 3.0 - i * 0.5})
        summary = diag.loss_summary()
        assert summary["initial_loss"] == 3.0
        assert summary["final_loss"] == 1.0
        assert summary["improvement"] == 2.0

    def test_accuracy_summary(self):
        diag = FederatedDiagnostics()
        for i in range(4):
            diag.record_round({"accuracy": 0.25 + i * 0.15})
        summary = diag.accuracy_summary()
        assert summary["initial_accuracy"] == 0.25
        assert summary["final_accuracy"] == 0.70
        assert abs(summary["max_accuracy"] - 0.70) < 1e-6

    def test_empty_diagnostics(self):
        diag = FederatedDiagnostics()
        assert diag.convergence_rate() is None
        assert diag.loss_summary() == {}
        assert diag.accuracy_summary() == {}


# ============================================================
# End-to-End Integration
# ============================================================

class TestEndToEnd:
    def test_full_federated_pipeline(self):
        """Complete FL pipeline: data -> partition -> clients -> server -> train -> evaluate."""
        np.random.seed(42)
        x, y = make_classification_data(300, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model, aggregation="fedavg")
        parts = partition_iid(x, y, 5)
        for i, (px, py) in enumerate(parts):
            client = FederatedClient(i, px, py, model, lr=0.01, local_epochs=3)
            server.add_client(client)

        eval_before = server.evaluate_global(x, y)
        results = server.run_training(15, test_x=x, test_y=y, eval_every=5)
        eval_after = server.evaluate_global(x, y)

        # Model should improve
        assert eval_after["loss"] < eval_before["loss"]
        assert eval_after["accuracy"] > eval_before["accuracy"]

    def test_noniid_training(self):
        """FL works even with non-IID data (may converge slower)."""
        np.random.seed(42)
        x, y = make_classification_data(300, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model, aggregation="fedavg")
        parts = partition_label_skew(x, y, 5, labels_per_client=2)
        for i, (px, py) in enumerate(parts):
            client = FederatedClient(i, px, py, model, lr=0.01, local_epochs=2)
            server.add_client(client)

        eval_before = server.evaluate_global(x, y)
        for _ in range(15):
            server.run_round()
        eval_after = server.evaluate_global(x, y)
        assert eval_after["loss"] < eval_before["loss"]

    def test_fedprox_pipeline(self):
        """FedProx (proximal term) should help with non-IID data."""
        np.random.seed(42)
        x, y = make_classification_data(200, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model, aggregation="fedavg")
        parts = partition_dirichlet(x, y, 4, alpha=0.3)
        for i, (px, py) in enumerate(parts):
            client = FederatedClient(i, px, py, model, lr=0.01, local_epochs=3)
            server.add_client(client)

        for _ in range(10):
            server.run_round(proximal_term=0.1)
        result = server.evaluate_global(x, y)
        assert result["accuracy"] > 0.33  # Better than random for 3 classes

    def test_with_secure_aggregation(self):
        """Secure aggregation produces same result as regular."""
        np.random.seed(42)
        model = FederatedModel([5, 8, 3], ["relu", "none"])
        global_params = model.get_params()
        shapes = [p.shape for p in global_params]

        sa = SecureAggregator(3)
        sa.generate_masks(shapes)

        deltas = [[np.random.randn(*s) * 0.1 for s in shapes] for _ in range(3)]
        masked = [sa.mask_update(i, deltas[i]) for i in range(3)]
        secure_result = sa.aggregate_masked(masked)
        regular_result = [np.mean([d[p] for d in deltas], axis=0) for p in range(len(shapes))]

        for sr, rr in zip(secure_result, regular_result):
            np.testing.assert_allclose(sr, rr, atol=1e-10)

    def test_diagnostics_integration(self):
        """Diagnostics track training properly."""
        np.random.seed(42)
        x, y = make_classification_data(200, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model)
        parts = partition_iid(x, y, 3)
        for i, (px, py) in enumerate(parts):
            server.add_client(FederatedClient(i, px, py, model, lr=0.01, local_epochs=2))

        diag = FederatedDiagnostics()
        results = server.run_training(8, test_x=x, test_y=y, eval_every=1)
        for r in results:
            diag.record_round(r)

        rate = diag.convergence_rate()
        assert rate is not None and rate < 1.0  # Loss should decrease
        stats = diag.participation_stats()
        assert stats["unique_clients"] == 3

    def test_compression_preserves_learning(self):
        """Training with compression still learns (slower but correct)."""
        np.random.seed(42)
        x, y = make_classification_data(200, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model)
        parts = partition_iid(x, y, 3)
        for i, (px, py) in enumerate(parts):
            server.add_client(FederatedClient(i, px, py, model, lr=0.01, local_epochs=2))

        eval_before = server.evaluate_global(x, y)
        for _ in range(10):
            server.run_round(compression="topk", compression_ratio=0.5)
        eval_after = server.evaluate_global(x, y)
        assert eval_after["loss"] < eval_before["loss"]

    def test_partial_client_participation(self):
        """Training works with partial client selection."""
        np.random.seed(42)
        x, y = make_classification_data(300, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = FederatedServer(model, client_fraction=0.5)
        parts = partition_iid(x, y, 6)
        for i, (px, py) in enumerate(parts):
            server.add_client(FederatedClient(i, px, py, model, lr=0.01, local_epochs=2))

        eval_before = server.evaluate_global(x, y)
        for _ in range(15):
            server.run_round()
        eval_after = server.evaluate_global(x, y)
        assert eval_after["loss"] < eval_before["loss"]

    def test_async_pipeline(self):
        """Async FL pipeline works."""
        np.random.seed(42)
        x, y = make_classification_data(200, 5, 3, seed=42)
        model = FederatedModel([5, 16, 3], ["relu", "none"])
        server = AsyncFederatedServer(model, staleness_penalty=0.5)
        parts = partition_iid(x, y, 3)
        clients = [FederatedClient(i, px, py, model, lr=0.01, local_epochs=2)
                    for i, (px, py) in enumerate(parts)]

        eval_before = server.evaluate_global(x, y)
        for round_idx in range(15):
            # Simulate async: random client reports
            c = clients[round_idx % 3]
            c.receive_global_model(server.global_model.get_params())
            update = c.train_local()
            server.apply_async_update(update)
        eval_after = server.evaluate_global(x, y)
        assert eval_after["loss"] < eval_before["loss"]


# ============================================================
# Data Generation
# ============================================================

class TestDataGeneration:
    def test_classification_data(self):
        x, y = make_classification_data(100, 5, 3, seed=42)
        assert x.shape == (100, 5)
        assert y.shape == (100,)
        assert set(y) == {0, 1, 2}

    def test_regression_data(self):
        x, y = make_regression_data(100, 5, seed=42)
        assert x.shape == (100, 5)
        assert y.shape == (100,)

    def test_classification_reproducible(self):
        x1, y1 = make_classification_data(50, 3, 2, seed=123)
        x2, y2 = make_classification_data(50, 3, 2, seed=123)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
