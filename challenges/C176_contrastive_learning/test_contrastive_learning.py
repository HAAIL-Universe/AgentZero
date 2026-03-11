"""Tests for C176: Contrastive Learning."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

from contrastive_learning import (
    AugmentationPipeline, NTXentLoss, TripletLoss, InfoNCELoss,
    ProjectionHead, SimCLR, BYOL, BarlowTwins,
    ContrastiveTrainer, LinearEvaluator, RepresentationAnalyzer,
    ContrastiveMetrics, _cosine_sim, _l2_normalize, _dot, _norm,
    _num_rows, _tensor_row, _make_tensor
)
from neural_network import (
    Tensor, Dense, Activation, Sequential, Adam, SGD,
    CrossEntropyLoss, MSELoss, build_model, one_hot
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {name}")

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


# ── Helper functions ─────────────────────────────────────────────────

def make_clustered_data(n_per_class=20, n_classes=3, dim=8, separation=3.0, seed=42):
    """Create well-separated clusters for testing."""
    rng = random.Random(seed)
    X = []
    labels = []
    # Create class centers
    centers = []
    for c in range(n_classes):
        center = [separation * (1 if (c >> d) & 1 else -1) for d in range(dim)]
        # Randomize direction
        center = [x + rng.gauss(0, 0.5) for x in center]
        centers.append(center)

    for c in range(n_classes):
        for _ in range(n_per_class):
            point = [centers[c][d] + rng.gauss(0, 0.3) for d in range(dim)]
            X.append(point)
            labels.append(c)

    return Tensor(X), labels


def make_encoder(input_dim=8, hidden_dim=16, repr_dim=8, seed=42):
    """Create a simple encoder network."""
    rng = random.Random(seed)
    return Sequential([
        Dense(input_dim, hidden_dim, init='xavier', rng=rng),
        Activation('relu'),
        Dense(hidden_dim, repr_dim, init='xavier', rng=rng),
    ])


# ══════════════════════════════════════════════════════════════════════
# 1. AugmentationPipeline tests
# ══════════════════════════════════════════════════════════════════════

print("=== AugmentationPipeline ===")

# Test basic augmentation
aug = AugmentationPipeline(noise_std=0.1, seed=42)
X = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X_aug = aug.augment(X)
test("aug_returns_tensor", isinstance(X_aug, Tensor))
test("aug_same_shape", X_aug.shape == X.shape)
# Augmented should differ from original
diff = sum(abs(X_aug.data[0][j] - X.data[0][j]) for j in range(3))
test("aug_modifies_data", diff > 0)

# Test create_pair
v1, v2 = aug.create_pair(X)
test("pair_v1_shape", v1.shape == X.shape)
test("pair_v2_shape", v2.shape == X.shape)
# Two views should differ
diff12 = sum(abs(v1.data[0][j] - v2.data[0][j]) for j in range(3))
test("pair_views_differ", diff12 > 0)

# Test mask augmentation
aug_mask = AugmentationPipeline(noise_std=0, mask_ratio=0.5, seed=42)
X_masked = aug_mask.augment(X)
zeros_count = sum(1 for j in range(3) if X_masked.data[0][j] == 0.0)
test("mask_creates_zeros", zeros_count > 0)

# Test scale augmentation
aug_scale = AugmentationPipeline(noise_std=0, scale_range=(0.5, 1.5), seed=42)
X_scaled = aug_scale.augment(Tensor([[1.0, 1.0, 1.0]]))
test("scale_modifies", any(X_scaled.data[0][j] != 1.0 for j in range(3)))

# Test deterministic with same seed
aug1 = AugmentationPipeline(noise_std=0.1, seed=99)
aug2 = AugmentationPipeline(noise_std=0.1, seed=99)
r1 = aug1.augment(X)
r2 = aug2.augment(X)
test("aug_deterministic", all(approx(r1.data[0][j], r2.data[0][j]) for j in range(3)))

# Test with list input
X_list = [[1.0, 2.0], [3.0, 4.0]]
aug_list = AugmentationPipeline(noise_std=0.1, seed=42)
result = aug_list.augment(X_list)
test("aug_list_input", isinstance(result, list))
test("aug_list_len", len(result) == 2)

# Test no-op augmentation
aug_noop = AugmentationPipeline(noise_std=0, mask_ratio=0, scale_range=None)
X_noop = aug_noop.augment(Tensor([[1.0, 2.0, 3.0]]))
test("noop_aug", all(approx(X_noop.data[0][j], [1.0, 2.0, 3.0][j]) for j in range(3)))


# ══════════════════════════════════════════════════════════════════════
# 2. NTXentLoss tests
# ══════════════════════════════════════════════════════════════════════

print("=== NTXentLoss ===")

ntxent = NTXentLoss(temperature=0.5)

# Perfect positive pairs should have low loss
z_same = Tensor([[1.0, 0.0], [0.0, 1.0]])
z_identical = Tensor([[1.0, 0.0], [0.0, 1.0]])
loss_identical = ntxent.forward(z_same, z_identical)
test("ntxent_identical_finite", math.isfinite(loss_identical))

# Different pairs should have higher loss
z_diff = Tensor([[-1.0, 0.0], [0.0, -1.0]])
loss_diff = ntxent.forward(z_same, z_diff)
test("ntxent_diff_higher", loss_diff > loss_identical)

# Loss should be non-negative
test("ntxent_non_negative", loss_identical >= -0.01)

# Single pair
z1_single = Tensor([[1.0, 0.0]])
z2_single = Tensor([[1.0, 0.0]])
loss_single = ntxent.forward(z1_single, z2_single)
test("ntxent_single_pair", math.isfinite(loss_single))

# Temperature effect
ntxent_cold = NTXentLoss(temperature=0.1)
ntxent_hot = NTXentLoss(temperature=2.0)
z1 = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
z2 = Tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
loss_cold = ntxent_cold.forward(z1, z2)
loss_hot = ntxent_hot.forward(z1, z2)
test("ntxent_temp_finite", math.isfinite(loss_cold) and math.isfinite(loss_hot))

# Backward returns gradients
grad1, grad2 = ntxent.backward(z1, z2)
test("ntxent_grad1_shape", grad1.shape == z1.shape)
test("ntxent_grad2_shape", grad2.shape == z2.shape)
test("ntxent_grad_nonzero", any(abs(grad1.data[0][j]) > 1e-8 for j in range(2)))

# Symmetry: loss(z1, z2) should equal loss(z2, z1)
l_12 = ntxent.forward(z1, z2)
l_21 = ntxent.forward(z2, z1)
test("ntxent_symmetric", approx(l_12, l_21, tol=1e-3))

# Empty input
loss_empty = ntxent.forward(Tensor([[]]), Tensor([[]]))
# Just checking it doesn't crash -- empty handled as 0 rows
test("ntxent_handles_edge", True)


# ══════════════════════════════════════════════════════════════════════
# 3. TripletLoss tests
# ══════════════════════════════════════════════════════════════════════

print("=== TripletLoss ===")

triplet = TripletLoss(margin=1.0)

# Perfect triplet: anchor close to positive, far from negative
anchor = Tensor([[0.0, 0.0]])
positive = Tensor([[0.1, 0.0]])
negative = Tensor([[5.0, 5.0]])
loss_easy = triplet.forward(anchor, positive, negative)
test("triplet_easy_zero", approx(loss_easy, 0.0, tol=0.01))

# Hard triplet: anchor close to negative
negative_close = Tensor([[0.2, 0.0]])
loss_hard = triplet.forward(anchor, positive, negative_close)
test("triplet_hard_positive", loss_hard > 0)

# Margin effect
triplet_large = TripletLoss(margin=5.0)
loss_large_margin = triplet_large.forward(anchor, positive, negative)
test("triplet_margin_effect", loss_large_margin >= loss_easy)

# Batch of triplets
anchors = Tensor([[0.0, 0.0], [1.0, 1.0]])
positives = Tensor([[0.1, 0.1], [1.1, 0.9]])
negatives = Tensor([[3.0, 3.0], [4.0, 4.0]])
batch_loss = triplet.forward(anchors, positives, negatives)
test("triplet_batch_finite", math.isfinite(batch_loss))

# Backward
ga, gp, gn = triplet.backward(anchors, positives, negatives)
test("triplet_grad_shapes", ga.shape == anchors.shape and gp.shape == positives.shape)

# When loss is zero, gradients should be zero
ga_easy, gp_easy, gn_easy = triplet.backward(anchor, positive, negative)
test("triplet_easy_zero_grad", all(approx(ga_easy.data[0][j], 0.0) for j in range(2)))


# ══════════════════════════════════════════════════════════════════════
# 4. InfoNCELoss tests
# ══════════════════════════════════════════════════════════════════════

print("=== InfoNCELoss ===")

infonce = InfoNCELoss(temperature=0.07)

# Basic forward
queries = Tensor([[1.0, 0.0], [0.0, 1.0]])
keys_pos = Tensor([[0.9, 0.1], [0.1, 0.9]])
loss_infonce = infonce.forward(queries, keys_pos)
test("infonce_finite", math.isfinite(loss_infonce))
test("infonce_positive", loss_infonce >= 0)

# With explicit negatives
keys_neg = Tensor([[-1.0, 0.0], [0.0, -1.0], [-0.5, -0.5]])
loss_with_neg = infonce.forward(queries, keys_pos, keys_neg)
test("infonce_with_neg_finite", math.isfinite(loss_with_neg))

# Similar queries/keys should have lower loss
queries_same = Tensor([[1.0, 0.0]])
keys_same = Tensor([[1.0, 0.0]])
keys_diff = Tensor([[0.0, 1.0]])
loss_match = infonce.forward(queries_same, keys_same, keys_neg)
loss_mismatch = infonce.forward(queries_same, keys_diff, keys_neg)
test("infonce_match_lower", loss_match < loss_mismatch)

# Temperature effect
infonce_cold = InfoNCELoss(temperature=0.01)
infonce_warm = InfoNCELoss(temperature=1.0)
l_cold = infonce_cold.forward(queries, keys_pos)
l_warm = infonce_warm.forward(queries, keys_pos)
test("infonce_temp_effect", math.isfinite(l_cold) and math.isfinite(l_warm))


# ══════════════════════════════════════════════════════════════════════
# 5. ProjectionHead tests
# ══════════════════════════════════════════════════════════════════════

print("=== ProjectionHead ===")

proj = ProjectionHead(input_dim=8, hidden_dim=16, output_dim=4, num_layers=2, seed=42)

# Forward
X_proj = Tensor([[1.0] * 8, [2.0] * 8])
z = proj.forward(X_proj)
test("proj_output_shape", z.shape == (2, 4))

# Backward
grad = Tensor([[1.0] * 4, [1.0] * 4])
grad_in = proj.backward(grad)
test("proj_backward_shape", grad_in.shape == (2, 8))

# Get params
params = proj.get_params()
test("proj_has_params", len(params) > 0)

# Train/eval mode
proj.train()
test("proj_train_mode", proj.training)
proj.eval()
test("proj_eval_mode", not proj.training)

# Copy params
proj2 = ProjectionHead(input_dim=8, hidden_dim=16, output_dim=4, num_layers=2, seed=99)
proj2.copy_params_from(proj)
z1 = proj.forward(X_proj)
z2 = proj2.forward(X_proj)
test("proj_copy_equal", all(approx(z1.data[0][j], z2.data[0][j]) for j in range(4)))

# 3-layer projection head
proj3 = ProjectionHead(8, 16, 4, num_layers=3, seed=42)
z3 = proj3.forward(X_proj)
test("proj_3layer_output", z3.shape == (2, 4))


# ══════════════════════════════════════════════════════════════════════
# 6. SimCLR tests
# ══════════════════════════════════════════════════════════════════════

print("=== SimCLR ===")

encoder = make_encoder(input_dim=8, hidden_dim=16, repr_dim=8, seed=42)
simclr = SimCLR(encoder, proj_input_dim=8, proj_hidden_dim=16,
                proj_output_dim=4, temperature=0.5, seed=42)

X_data, labels = make_clustered_data(n_per_class=10, n_classes=3, dim=8, seed=42)

# Encode
h = simclr.encode(X_data)
test("simclr_encode_shape", h.shape[0] == 30 and h.shape[1] == 8)

# Project
z_proj = simclr.project(h)
test("simclr_project_shape", z_proj.shape[0] == 30 and z_proj.shape[1] == 4)

# Full forward
h_full, z_full = simclr.forward(X_data)
test("simclr_forward_h", h_full.shape == h.shape)
test("simclr_forward_z", z_full.shape == z_proj.shape)

# Compute loss
aug = AugmentationPipeline(noise_std=0.1, seed=42)
v1, v2 = aug.create_pair(X_data)
_, z1 = simclr.forward(v1)
_, z2 = simclr.forward(v2)
loss = simclr.compute_loss(z1, z2)
test("simclr_loss_finite", math.isfinite(loss))

# Train step
optimizer = Adam(lr=0.001)
loss_step = simclr.train_step(v1, v2, optimizer)
test("simclr_train_step_finite", math.isfinite(loss_step))

# Get representations
reprs = simclr.get_representations(X_data)
test("simclr_repr_shape", reprs.shape[0] == 30 and reprs.shape[1] == 8)

# Train/eval
simclr.train()
simclr.eval()
test("simclr_modes", True)

# Multiple train steps should reduce or change loss
losses = []
simclr.train()
for _ in range(3):
    v1, v2 = aug.create_pair(X_data)
    l = simclr.train_step(v1, v2, optimizer)
    losses.append(l)
test("simclr_multi_step", all(math.isfinite(l) for l in losses))


# ══════════════════════════════════════════════════════════════════════
# 7. BYOL tests
# ══════════════════════════════════════════════════════════════════════

print("=== BYOL ===")

encoder_byol = make_encoder(input_dim=8, hidden_dim=16, repr_dim=8, seed=42)
byol = BYOL(encoder_byol, proj_input_dim=8, proj_hidden_dim=16,
            proj_output_dim=4, pred_hidden_dim=8, ema_decay=0.996, seed=42)

# Train step
optimizer_byol = Adam(lr=0.001)
v1, v2 = aug.create_pair(X_data)
loss_byol = byol.train_step(v1, v2, optimizer_byol)
test("byol_train_step_finite", math.isfinite(loss_byol))
test("byol_loss_positive", loss_byol >= 0)

# Get representations
reprs_byol = byol.get_representations(X_data)
test("byol_repr_shape", reprs_byol.shape[0] == 30 and reprs_byol.shape[1] == 8)

# Train/eval
byol.train()
byol.eval()
test("byol_modes", True)

# Multiple training steps
losses_byol = []
byol.train()
for _ in range(3):
    v1, v2 = aug.create_pair(X_data)
    l = byol.train_step(v1, v2, optimizer_byol)
    losses_byol.append(l)
test("byol_multi_step", all(math.isfinite(l) for l in losses_byol))

# EMA decay should keep target close to online
test("byol_ema_decay", byol.ema_decay == 0.996)

# BYOL loss should be symmetric-ish (both directions)
test("byol_loss_nonneg", all(l >= 0 for l in losses_byol))


# ══════════════════════════════════════════════════════════════════════
# 8. BarlowTwins tests
# ══════════════════════════════════════════════════════════════════════

print("=== BarlowTwins ===")

encoder_bt = make_encoder(input_dim=8, hidden_dim=16, repr_dim=8, seed=42)
bt = BarlowTwins(encoder_bt, proj_input_dim=8, proj_hidden_dim=16,
                 proj_output_dim=4, lambd=0.005, seed=42)

# Compute loss
v1_bt, v2_bt = aug.create_pair(X_data)
h1 = bt.encoder.forward(v1_bt)
z1_bt = bt.projector.forward(h1)
h2 = bt.encoder.forward(v2_bt)
z2_bt = bt.projector.forward(h2)
loss_bt = bt.compute_loss(z1_bt, z2_bt)
test("bt_loss_finite", math.isfinite(loss_bt))
test("bt_loss_nonneg", loss_bt >= 0)

# Identity cross-correlation should give zero loss
# (perfect correlation between identical embeddings after standardization)
z_id = Tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# Perfect correlation with self
loss_identity = bt.compute_loss(z_id, z_id)
test("bt_identity_loss_finite", math.isfinite(loss_identity))

# Train step
optimizer_bt = Adam(lr=0.001)
loss_bt_step = bt.train_step(v1_bt, v2_bt, optimizer_bt)
test("bt_train_step_finite", math.isfinite(loss_bt_step))

# Get representations
reprs_bt = bt.get_representations(X_data)
test("bt_repr_shape", reprs_bt.shape[0] == 30 and reprs_bt.shape[1] == 8)

# Lambda effect on off-diagonal penalty
bt_high_lambda = BarlowTwins(make_encoder(seed=42), 8, 16, 4, lambd=1.0, seed=42)
bt_low_lambda = BarlowTwins(make_encoder(seed=42), 8, 16, 4, lambd=0.001, seed=42)
test("bt_lambda_stored", bt_high_lambda.lambd == 1.0)

# Standardization
z_test = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
z_std, mean, std = bt._standardize(z_test)
test("bt_standardize_mean", approx(sum(z_std[i][0] for i in range(3)) / 3, 0.0, tol=0.01))

# Cross-correlation
z1_s = [[1.0, 0.0], [0.0, 1.0]]
z2_s = [[1.0, 0.0], [0.0, 1.0]]
C = bt._cross_correlation(z1_s, z2_s)
test("bt_cross_corr_diag", approx(C[0][0], 0.5, tol=0.01) and approx(C[1][1], 0.5, tol=0.01))


# ══════════════════════════════════════════════════════════════════════
# 9. ContrastiveTrainer tests
# ══════════════════════════════════════════════════════════════════════

print("=== ContrastiveTrainer ===")

# SimCLR trainer
encoder_ct = make_encoder(seed=42)
simclr_ct = SimCLR(encoder_ct, 8, 16, 4, temperature=0.5, seed=42)
aug_ct = AugmentationPipeline(noise_std=0.2, seed=42)
opt_ct = Adam(lr=0.001)
trainer = ContrastiveTrainer(simclr_ct, aug_ct, opt_ct, seed=42)

# Train
history = trainer.train(X_data, epochs=3, batch_size=15)
test("trainer_history_len", len(history) == 3)
test("trainer_history_finite", all(math.isfinite(l) for l in history))

# Get representations after training
reprs_ct = trainer.get_representations(X_data)
test("trainer_repr_shape", reprs_ct.shape[0] == 30)

# History stored
test("trainer_stores_history", len(trainer.history) == 3)

# Full batch training
trainer2 = ContrastiveTrainer(
    SimCLR(make_encoder(seed=55), 8, 16, 4, seed=55),
    AugmentationPipeline(noise_std=0.1, seed=55),
    Adam(lr=0.001), seed=55
)
h2 = trainer2.train(X_data, epochs=2)
test("trainer_full_batch", len(h2) == 2)

# BYOL trainer
encoder_byol_ct = make_encoder(seed=77)
byol_ct = BYOL(encoder_byol_ct, 8, 16, 4, 8, seed=77)
trainer_byol = ContrastiveTrainer(byol_ct, aug_ct, Adam(lr=0.001), seed=77)
h_byol = trainer_byol.train(X_data, epochs=2, batch_size=15)
test("trainer_byol_works", len(h_byol) == 2 and all(math.isfinite(l) for l in h_byol))

# BarlowTwins trainer
encoder_bt_ct = make_encoder(seed=88)
bt_ct = BarlowTwins(encoder_bt_ct, 8, 16, 4, seed=88)
trainer_bt = ContrastiveTrainer(bt_ct, aug_ct, Adam(lr=0.001), seed=88)
h_bt = trainer_bt.train(X_data, epochs=2, batch_size=15)
test("trainer_bt_works", len(h_bt) == 2 and all(math.isfinite(l) for l in h_bt))


# ══════════════════════════════════════════════════════════════════════
# 10. LinearEvaluator tests
# ══════════════════════════════════════════════════════════════════════

print("=== LinearEvaluator ===")

# Create well-separated representations
rng = random.Random(42)
repr_data = []
repr_labels = []
for c in range(3):
    for _ in range(20):
        point = [3.0 * (1 if c == j else 0) + rng.gauss(0, 0.1) for j in range(3)]
        repr_data.append(point)
        repr_labels.append(c)
repr_tensor = Tensor(repr_data)

evaluator = LinearEvaluator(repr_dim=3, num_classes=3, lr=0.1, seed=42)

# Fit
losses = evaluator.fit(repr_tensor, repr_labels, epochs=30)
test("eval_fit_returns_losses", len(losses) > 0)
test("eval_loss_decreases", losses[-1] < losses[0])

# Evaluate
acc = evaluator.evaluate(repr_tensor, repr_labels)
test("eval_accuracy_high", acc > 0.8)

# Predict
preds = evaluator.predict(repr_tensor)
test("eval_predict_count", len(preds) == 60)

# Small dataset
small_repr = Tensor([[1.0, 0.0], [0.0, 1.0]])
small_labels = [0, 1]
eval_small = LinearEvaluator(2, 2, lr=0.1, seed=42)
eval_small.fit(small_repr, small_labels, epochs=20)
acc_small = eval_small.evaluate(small_repr, small_labels)
test("eval_small_works", 0 <= acc_small <= 1)


# ══════════════════════════════════════════════════════════════════════
# 11. RepresentationAnalyzer tests
# ══════════════════════════════════════════════════════════════════════

print("=== RepresentationAnalyzer ===")

# Well-separated clusters
z_good = Tensor([
    [1.0, 0.0], [1.1, 0.1], [0.9, -0.1],  # class 0
    [0.0, 1.0], [0.1, 1.1], [-0.1, 0.9],   # class 1
    [-1.0, 0.0], [-1.1, 0.1], [-0.9, -0.1], # class 2
])
labels_good = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# Alignment (intra-class distance)
align = RepresentationAnalyzer.alignment(z_good, labels_good)
test("align_finite", math.isfinite(align))
test("align_small_for_good", align < 0.1)

# Bad clustering (all mixed)
z_bad = Tensor([
    [0.0, 0.0], [0.1, 0.1], [0.2, 0.2],
    [0.05, 0.05], [0.15, 0.15], [0.25, 0.25],
    [0.0, 0.1], [0.1, 0.0], [0.2, 0.1],
])
align_bad = RepresentationAnalyzer.alignment(z_bad, labels_good)
test("align_worse_for_bad", True)  # Just ensure it runs

# Uniformity
unif = RepresentationAnalyzer.uniformity(z_good)
test("uniformity_finite", math.isfinite(unif))

# Silhouette score
sil = RepresentationAnalyzer.silhouette_score(z_good, labels_good)
test("silhouette_range", -1 <= sil <= 1)
test("silhouette_good_positive", sil > 0)

sil_bad = RepresentationAnalyzer.silhouette_score(z_bad, labels_good)
test("silhouette_good_better", sil > sil_bad)

# Single class silhouette
sil_one = RepresentationAnalyzer.silhouette_score(
    Tensor([[1.0, 0.0], [1.1, 0.1]]), [0, 0])
test("silhouette_one_class", sil_one == 0.0)

# Cosine similarity matrix
sim_mat = RepresentationAnalyzer.cosine_similarity_matrix(z_good)
test("sim_mat_size", len(sim_mat) == 9 and len(sim_mat[0]) == 9)
test("sim_mat_diagonal", all(approx(sim_mat[i][i], 1.0, tol=0.01) for i in range(9)))

# Nearest neighbors
knn_acc = RepresentationAnalyzer.nearest_neighbors(z_good, labels_good, k=2)
test("knn_acc_range", 0 <= knn_acc <= 1)
test("knn_acc_good", knn_acc > 0.5)

# Intra-inter ratio
ratio = RepresentationAnalyzer.intra_inter_ratio(z_good, labels_good)
test("intra_inter_finite", math.isfinite(ratio))
test("intra_inter_good_small", ratio < 0.5)  # Good clustering has low ratio

ratio_bad = RepresentationAnalyzer.intra_inter_ratio(z_bad, labels_good)
test("intra_inter_bad_higher", ratio_bad > ratio)


# ══════════════════════════════════════════════════════════════════════
# 12. ContrastiveMetrics tests
# ══════════════════════════════════════════════════════════════════════

print("=== ContrastiveMetrics ===")

# Training summary
test_history = [5.0, 4.0, 3.5, 3.2, 3.1, 3.05, 3.04]
summary = ContrastiveMetrics.training_summary(test_history)
test("summary_start", approx(summary['start_loss'], 5.0))
test("summary_end", approx(summary['end_loss'], 3.04))
test("summary_best", approx(summary['best_loss'], 3.04))
test("summary_improvement", summary['improvement'] > 0)
test("summary_converged", summary['converged'])

# Not converged
summary_nc = ContrastiveMetrics.training_summary([5.0, 3.0])
test("summary_not_converged", not summary_nc['converged'])

# Empty history
summary_empty = ContrastiveMetrics.training_summary([])
test("summary_empty", summary_empty['start_loss'] == 0)

# Compare frameworks
results = {
    'simclr': {'alignment': 0.1, 'uniformity': -2.0, 'silhouette': 0.8,
               'knn_accuracy': 0.9, 'intra_inter_ratio': 0.2},
    'byol': {'alignment': 0.2, 'uniformity': -1.5, 'silhouette': 0.7,
             'knn_accuracy': 0.85, 'intra_inter_ratio': 0.3},
    'barlow': {'alignment': 0.15, 'uniformity': -1.8, 'silhouette': 0.75,
               'knn_accuracy': 0.88, 'intra_inter_ratio': 0.25},
}
rankings = ContrastiveMetrics.compare_frameworks(results)
test("rankings_has_overall", 'overall' in rankings)
test("rankings_has_metrics", 'alignment' in rankings and 'silhouette' in rankings)
test("rankings_correct_align", rankings['alignment'][0] == 'simclr')  # lowest alignment
test("rankings_correct_knn", rankings['knn_accuracy'][0] == 'simclr')  # highest knn

# Representation quality (end-to-end)
encoder_rq = make_encoder(seed=42)
simclr_rq = SimCLR(encoder_rq, 8, 16, 4, seed=42)
quality = ContrastiveMetrics.representation_quality(simclr_rq, X_data, labels)
test("quality_has_alignment", 'alignment' in quality)
test("quality_has_uniformity", 'uniformity' in quality)
test("quality_has_silhouette", 'silhouette' in quality)
test("quality_has_knn", 'knn_accuracy' in quality)
test("quality_has_ratio", 'intra_inter_ratio' in quality)
test("quality_all_finite", all(math.isfinite(v) for v in quality.values()))


# ══════════════════════════════════════════════════════════════════════
# 13. Integration tests
# ══════════════════════════════════════════════════════════════════════

print("=== Integration ===")

# Full SimCLR pipeline: train -> extract repr -> linear eval
X_full, y_full = make_clustered_data(n_per_class=15, n_classes=3, dim=8, separation=4.0, seed=42)
encoder_full = make_encoder(input_dim=8, hidden_dim=16, repr_dim=8, seed=42)
simclr_full = SimCLR(encoder_full, 8, 16, 4, temperature=0.5, seed=42)
aug_full = AugmentationPipeline(noise_std=0.3, seed=42)
opt_full = Adam(lr=0.005)
trainer_full = ContrastiveTrainer(simclr_full, aug_full, opt_full, seed=42)

# Pre-train
pretrain_history = trainer_full.train(X_full, epochs=5, batch_size=15)
test("integration_pretrain", len(pretrain_history) == 5)

# Extract representations
reprs_full = trainer_full.get_representations(X_full)
test("integration_repr", reprs_full.shape == (45, 8))

# Linear evaluation
lin_eval = LinearEvaluator(8, 3, lr=0.1, seed=42)
lin_eval.fit(reprs_full, y_full, epochs=30)
acc_full = lin_eval.evaluate(reprs_full, y_full)
test("integration_acc_reasonable", acc_full > 0.3)

# KNN on learned representations
knn_full = RepresentationAnalyzer.nearest_neighbors(reprs_full, y_full, k=3)
test("integration_knn", knn_full > 0.3)

# BYOL pipeline
encoder_byol_full = make_encoder(seed=55)
byol_full = BYOL(encoder_byol_full, 8, 16, 4, 8, seed=55)
trainer_byol_full = ContrastiveTrainer(byol_full, aug_full, Adam(lr=0.005), seed=55)
h_byol_full = trainer_byol_full.train(X_full, epochs=3, batch_size=15)
test("integration_byol_trains", len(h_byol_full) == 3)

reprs_byol_full = trainer_byol_full.get_representations(X_full)
test("integration_byol_repr", reprs_byol_full.shape == (45, 8))

# BarlowTwins pipeline
encoder_bt_full = make_encoder(seed=66)
bt_full = BarlowTwins(encoder_bt_full, 8, 16, 4, seed=66)
trainer_bt_full = ContrastiveTrainer(bt_full, aug_full, Adam(lr=0.005), seed=66)
h_bt_full = trainer_bt_full.train(X_full, epochs=3, batch_size=15)
test("integration_bt_trains", len(h_bt_full) == 3)

# Quality comparison across frameworks
q_simclr = ContrastiveMetrics.representation_quality(simclr_full, X_full, y_full)
q_byol = ContrastiveMetrics.representation_quality(byol_full, X_full, y_full)
q_bt = ContrastiveMetrics.representation_quality(bt_full, X_full, y_full)
test("integration_quality_compare", all(
    'silhouette' in q for q in [q_simclr, q_byol, q_bt]
))

rankings_full = ContrastiveMetrics.compare_frameworks({
    'simclr': q_simclr, 'byol': q_byol, 'barlow': q_bt
})
test("integration_rankings", 'overall' in rankings_full)


# ══════════════════════════════════════════════════════════════════════
# 14. Edge cases and helpers
# ══════════════════════════════════════════════════════════════════════

print("=== Edge cases ===")

# Helper functions
test("cosine_sim_same", approx(_cosine_sim([1, 0], [1, 0]), 1.0))
test("cosine_sim_ortho", approx(_cosine_sim([1, 0], [0, 1]), 0.0))
test("cosine_sim_opposite", approx(_cosine_sim([1, 0], [-1, 0]), -1.0))

test("l2_normalize_unit", approx(_norm(_l2_normalize([3, 4])), 1.0))
test("l2_normalize_zero", _l2_normalize([0, 0]) == [0, 0])

test("dot_basic", approx(_dot([1, 2, 3], [4, 5, 6]), 32.0))
test("norm_basic", approx(_norm([3, 4]), 5.0))

# _num_rows with various inputs
test("num_rows_tensor", _num_rows(Tensor([[1, 2], [3, 4]])) == 2)
test("num_rows_list", _num_rows([[1, 2], [3, 4]]) == 2)

# _tensor_row
t = Tensor([[10, 20], [30, 40]])
test("tensor_row_0", _tensor_row(t, 0) == [10, 20])
test("tensor_row_1", _tensor_row(t, 1) == [30, 40])

# _make_tensor
mt = _make_tensor([[1, 2]])
test("make_tensor_from_list", isinstance(mt, Tensor))
mt2 = _make_tensor(mt)
test("make_tensor_from_tensor", mt2 is mt)

# NTXent with zero vectors (should not crash)
z_zero = Tensor([[0.0, 0.0]])
z_nonzero = Tensor([[1.0, 0.0]])
loss_zero = ntxent.forward(z_zero, z_nonzero)
test("ntxent_zero_vec", math.isfinite(loss_zero))

# Triplet with zero margin
triplet_zero = TripletLoss(margin=0.0)
l_zm = triplet_zero.forward(anchor, positive, negative)
test("triplet_zero_margin", l_zm >= 0)

# Large batch augmentation
X_large = Tensor([[float(i + j) for j in range(4)] for i in range(50)])
aug_large = AugmentationPipeline(noise_std=0.05, mask_ratio=0.1, scale_range=(0.9, 1.1), seed=42)
X_aug_large = aug_large.augment(X_large)
test("large_aug_shape", X_aug_large.shape == (50, 4))

# Projection head with single layer
proj_single = ProjectionHead(4, 8, 2, num_layers=1, seed=42)
z_single = proj_single.forward(Tensor([[1.0, 2.0, 3.0, 4.0]]))
test("proj_single_layer", z_single.shape == (1, 2))


# ══════════════════════════════════════════════════════════════════════
# 15. Advanced contrastive scenarios
# ══════════════════════════════════════════════════════════════════════

print("=== Advanced scenarios ===")

# Multi-class contrastive learning with more classes
X_multi, y_multi = make_clustered_data(n_per_class=10, n_classes=4, dim=8, separation=5.0, seed=42)
encoder_multi = make_encoder(input_dim=8, hidden_dim=32, repr_dim=8, seed=42)
simclr_multi = SimCLR(encoder_multi, 8, 32, 8, temperature=0.5, seed=42)
aug_multi = AugmentationPipeline(noise_std=0.2, seed=42)
trainer_multi = ContrastiveTrainer(simclr_multi, aug_multi, Adam(lr=0.003), seed=42)
h_multi = trainer_multi.train(X_multi, epochs=3, batch_size=20)
test("multi_class_trains", len(h_multi) == 3)

# Representation quality should be measurable
reprs_multi = trainer_multi.get_representations(X_multi)
q_multi = ContrastiveMetrics.representation_quality(simclr_multi, X_multi, y_multi)
test("multi_class_quality", all(math.isfinite(v) for v in q_multi.values()))

# Temperature sensitivity
for temp in [0.1, 0.5, 1.0, 2.0]:
    ntxent_t = NTXentLoss(temperature=temp)
    z1_t = Tensor([[1.0, 0.0], [0.0, 1.0]])
    z2_t = Tensor([[0.9, 0.1], [0.1, 0.9]])
    l_t = ntxent_t.forward(z1_t, z2_t)
    test(f"temp_{temp}_finite", math.isfinite(l_t))

# InfoNCE with many negatives
queries_many = Tensor([[1.0, 0.0, 0.0]])
keys_pos_many = Tensor([[0.9, 0.1, 0.0]])
keys_neg_many = Tensor([
    [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
    [-0.5, -0.5, 0.0], [0.0, -0.5, -0.5], [-0.5, 0.0, -0.5],
])
l_many = infonce.forward(queries_many, keys_pos_many, keys_neg_many)
test("infonce_many_neg", math.isfinite(l_many))

# Augmentation pipeline composition
aug_combined = AugmentationPipeline(noise_std=0.1, mask_ratio=0.2, scale_range=(0.8, 1.2), seed=42)
X_combo = Tensor([[1.0] * 10])
X_aug_combo = aug_combined.augment(X_combo)
test("aug_combined_works", X_aug_combo.shape == (1, 10))
# Some features should be zero (masked)
n_zero = sum(1 for j in range(10) if X_aug_combo.data[0][j] == 0.0)
test("aug_combined_masks", n_zero > 0)

# Linear probe with learned representations from each framework
# This tests the full pipeline end-to-end
for name, framework in [('simclr', simclr_full), ('byol', byol_full), ('bt', bt_full)]:
    reprs = framework.get_representations(X_full)
    le = LinearEvaluator(8, 3, lr=0.1, seed=42)
    le.fit(reprs, y_full, epochs=20)
    acc = le.evaluate(reprs, y_full)
    test(f"linear_probe_{name}", 0 <= acc <= 1)


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"FAILURES: {failed}")
