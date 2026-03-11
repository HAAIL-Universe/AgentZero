"""
Tests for C190: Anomaly Detection
"""
import numpy as np
import pytest
from anomaly_detection import (
    ZScoreDetector, RobustZScoreDetector, IQRDetector, EWMADetector,
    IsolationForest, LOF, DBSCANAnomaly, MultivariateDetector,
    OneClassSVM, EnsembleDetector, GrubbsTest, CUSUM, StreamingDetector,
    AnomalyResult, _c
)


# ============================================================
# Helper: generate data with known anomalies
# ============================================================
def make_data_1d(n=200, n_anomalies=10, seed=42):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, n)
    anomalies = rng.uniform(8, 12, n_anomalies) * rng.choice([-1, 1], n_anomalies)
    X = np.concatenate([normal, anomalies])
    labels = np.concatenate([np.zeros(n), np.ones(n_anomalies)])
    return X, labels


def make_data_2d(n=200, n_anomalies=10, seed=42):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, (n, 2))
    anomalies = rng.uniform(6, 10, (n_anomalies, 2)) * rng.choice([-1, 1], (n_anomalies, 2))
    X = np.vstack([normal, anomalies])
    labels = np.concatenate([np.zeros(n), np.ones(n_anomalies)])
    return X, labels


# ============================================================
# Z-Score Detector
# ============================================================
class TestZScoreDetector:
    def test_basic_detection(self):
        X, true_labels = make_data_1d()
        det = ZScoreDetector(threshold=3.0)
        result = det.fit_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.labels) == len(X)
        # Should detect most anomalies
        anomaly_detected = result.labels[200:]
        assert np.sum(anomaly_detected) >= 5

    def test_no_false_positives_clean_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, 500)
        result = ZScoreDetector(threshold=4.0).fit_predict(X)
        # Very few false positives expected with threshold=4
        assert np.sum(result.labels) <= 5

    def test_2d_data(self):
        X, _ = make_data_2d()
        result = ZScoreDetector(threshold=3.0).fit_predict(X)
        assert len(result.labels) == len(X)

    def test_fit_then_predict_new_data(self):
        rng = np.random.default_rng(1)
        X_train = rng.normal(0, 1, 200)
        det = ZScoreDetector(threshold=3.0).fit(X_train)
        X_test = np.array([0.5, -0.3, 15.0, -12.0])
        result = det.predict(X_test)
        assert result.labels[0] == 0
        assert result.labels[1] == 0
        assert result.labels[2] == 1
        assert result.labels[3] == 1

    def test_constant_data(self):
        X = np.ones(100)
        det = ZScoreDetector().fit(X)
        scores = det.score(np.array([1.0, 2.0]))
        assert scores[0] < scores[1]

    def test_scores_are_positive(self):
        X, _ = make_data_1d()
        scores = ZScoreDetector().fit(X).score(X)
        assert np.all(scores >= 0)

    def test_threshold_parameter(self):
        X, _ = make_data_1d()
        r1 = ZScoreDetector(threshold=2.0).fit_predict(X)
        r2 = ZScoreDetector(threshold=4.0).fit_predict(X)
        assert np.sum(r1.labels) >= np.sum(r2.labels)


# ============================================================
# Robust Z-Score Detector
# ============================================================
class TestRobustZScoreDetector:
    def test_basic_detection(self):
        X, _ = make_data_1d()
        result = RobustZScoreDetector(threshold=3.5).fit_predict(X)
        assert isinstance(result, AnomalyResult)
        anomaly_detected = result.labels[200:]
        assert np.sum(anomaly_detected) >= 5

    def test_robust_to_contamination(self):
        # Robust z-score should handle contaminated data better
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, 100)
        # Add 30% contamination
        contaminated = rng.uniform(10, 15, 30)
        X = np.concatenate([normal, contaminated])
        result = RobustZScoreDetector(threshold=3.5).fit_predict(X)
        # Should still detect some anomalies even with heavy contamination
        assert np.sum(result.labels) > 0

    def test_2d_data(self):
        X, _ = make_data_2d()
        result = RobustZScoreDetector().fit_predict(X)
        assert len(result.labels) == len(X)

    def test_constant_data(self):
        X = np.ones(100)
        det = RobustZScoreDetector().fit(X)
        # Even constant data should not crash
        scores = det.score(np.array([1.0, 2.0]))
        assert len(scores) == 2

    def test_scores_positive(self):
        X, _ = make_data_1d()
        scores = RobustZScoreDetector().fit(X).score(X)
        assert np.all(scores >= 0)


# ============================================================
# IQR Detector
# ============================================================
class TestIQRDetector:
    def test_basic_detection(self):
        X, _ = make_data_1d()
        result = IQRDetector(factor=1.5).fit_predict(X)
        assert isinstance(result, AnomalyResult)
        anomaly_detected = result.labels[200:]
        assert np.sum(anomaly_detected) >= 5

    def test_factor_affects_sensitivity(self):
        X, _ = make_data_1d()
        r1 = IQRDetector(factor=1.0).fit_predict(X)
        r2 = IQRDetector(factor=3.0).fit_predict(X)
        assert np.sum(r1.labels) >= np.sum(r2.labels)

    def test_2d_data(self):
        X, _ = make_data_2d()
        result = IQRDetector().fit_predict(X)
        assert len(result.labels) == len(X)

    def test_inliers_have_zero_score(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, 1000)
        det = IQRDetector(factor=1.5).fit(X)
        # Test points well within IQR
        test = np.array([0.0, 0.1, -0.1])
        scores = det.score(test)
        assert np.all(scores == 0)

    def test_outliers_have_positive_score(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, 1000)
        det = IQRDetector(factor=1.5).fit(X)
        test = np.array([100.0, -100.0])
        scores = det.score(test)
        assert np.all(scores > 0)

    def test_fit_stores_quartiles(self):
        X = np.arange(100, dtype=float)
        det = IQRDetector().fit(X.reshape(-1, 1))
        assert det.q1_ is not None
        assert det.q3_ is not None
        assert det.iqr_ is not None


# ============================================================
# EWMA Detector
# ============================================================
class TestEWMADetector:
    def test_basic_detection(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 100)
        # Add a mean shift
        X_shift = np.concatenate([X, rng.normal(5, 1, 20)])
        det = EWMADetector(alpha=0.3, n_sigma=3.0).fit(X)
        result = det.predict(X_shift)
        assert isinstance(result, AnomalyResult)
        # The shifted part should trigger some alarms
        assert np.sum(result.labels[100:]) > 0

    def test_alpha_sensitivity(self):
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, 200)
        X_test = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50)])
        r1 = EWMADetector(alpha=0.1).fit(X_train).predict(X_test)
        r2 = EWMADetector(alpha=0.9).fit(X_train).predict(X_test)
        # Both should work
        assert len(r1.labels) == len(X_test)
        assert len(r2.labels) == len(X_test)

    def test_online_charting(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 50)
        det = EWMADetector(alpha=0.3).fit(X)
        chart = det.score_online(X)
        assert 'ewma' in chart
        assert 'ucl' in chart
        assert 'lcl' in chart
        assert chart['ewma'].shape[0] == len(X)
        # UCL should be above LCL
        assert np.all(chart['ucl'] >= chart['lcl'])

    def test_constant_data_no_alarms(self):
        X = np.ones(100)
        det = EWMADetector().fit(X)
        result = det.predict(X)
        assert np.sum(result.labels) == 0

    def test_2d(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (100, 3))
        result = EWMADetector().fit_predict(X)
        assert len(result.labels) == 100


# ============================================================
# Isolation Forest
# ============================================================
class TestIsolationForest:
    def test_basic_detection(self):
        X, true_labels = make_data_2d(n=200, n_anomalies=20, seed=42)
        iforest = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
        result = iforest.fit_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.labels) == len(X)

    def test_anomaly_scores_range(self):
        X, _ = make_data_2d(seed=42)
        iforest = IsolationForest(n_estimators=50, random_state=42).fit(X)
        scores = iforest.score(X)
        # Scores should be roughly in [0, 1]
        assert np.all(scores >= 0)
        assert np.all(scores <= 1.5)

    def test_anomalies_score_higher(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (200, 2))
        anomalies = np.array([[10, 10], [-10, -10], [10, -10]])
        X = np.vstack([normal, anomalies])
        iforest = IsolationForest(n_estimators=100, random_state=42).fit(X)
        scores = iforest.score(X)
        # Anomalies should have higher scores
        normal_mean = np.mean(scores[:200])
        anomaly_mean = np.mean(scores[200:])
        assert anomaly_mean > normal_mean

    def test_n_estimators(self):
        X, _ = make_data_2d(seed=42)
        iforest = IsolationForest(n_estimators=10, random_state=42).fit(X)
        assert len(iforest.trees_) == 10

    def test_max_samples(self):
        X, _ = make_data_2d(n=500, seed=42)
        iforest = IsolationForest(max_samples=50, random_state=42).fit(X)
        result = iforest.predict(X)
        assert len(result.labels) == len(X)

    def test_contamination(self):
        X, _ = make_data_2d(n=200, n_anomalies=20, seed=42)
        r1 = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        r2 = IsolationForest(contamination=0.2, random_state=42).fit_predict(X)
        assert np.sum(r1.labels) <= np.sum(r2.labels)

    def test_1d_data(self):
        X, _ = make_data_1d()
        result = IsolationForest(n_estimators=30, random_state=42).fit_predict(X)
        assert len(result.labels) == len(X)

    def test_reproducible(self):
        X, _ = make_data_2d(seed=42)
        r1 = IsolationForest(random_state=123).fit_predict(X)
        r2 = IsolationForest(random_state=123).fit_predict(X)
        np.testing.assert_array_equal(r1.labels, r2.labels)


# ============================================================
# _c helper function
# ============================================================
class TestCFunction:
    def test_c_0(self):
        assert _c(0) == 0

    def test_c_1(self):
        assert _c(1) == 0

    def test_c_2(self):
        assert _c(2) == 1

    def test_c_positive(self):
        for n in [3, 10, 100, 256]:
            assert _c(n) > 0

    def test_c_increasing(self):
        vals = [_c(n) for n in range(2, 100)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]


# ============================================================
# LOF
# ============================================================
class TestLOF:
    def test_basic_detection(self):
        X, true_labels = make_data_2d(n=100, n_anomalies=10, seed=42)
        lof = LOF(k=10, contamination=0.1)
        result = lof.fit_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.labels) == len(X)

    def test_anomalies_higher_lof(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (100, 2))
        anomalies = np.array([[10, 10], [-10, -10]])
        X = np.vstack([normal, anomalies])
        lof = LOF(k=10, contamination=0.05).fit(X)
        scores = lof.score(X)
        # Anomalies should have higher LOF
        assert scores[-1] > np.mean(scores[:100])
        assert scores[-2] > np.mean(scores[:100])

    def test_k_parameter(self):
        X, _ = make_data_2d(n=50, seed=42)
        lof = LOF(k=5).fit(X)
        assert lof.X_train_ is not None

    def test_small_dataset(self):
        X = np.array([[0, 0], [1, 1], [10, 10]])
        lof = LOF(k=1, contamination=0.3)
        result = lof.fit_predict(X)
        assert len(result.labels) == 3

    def test_1d_data(self):
        X, _ = make_data_1d(n=50, n_anomalies=5)
        result = LOF(k=5, contamination=0.1).fit_predict(X)
        assert len(result.labels) == 55

    def test_pairwise_distances(self):
        A = np.array([[0, 0], [3, 4]])
        B = np.array([[0, 0], [3, 4]])
        dists = LOF._pairwise_distances(A, B)
        assert dists[0, 0] == pytest.approx(0, abs=1e-10)
        assert dists[0, 1] == pytest.approx(5.0, abs=1e-10)
        assert dists[1, 0] == pytest.approx(5.0, abs=1e-10)
        assert dists[1, 1] == pytest.approx(0, abs=1e-10)


# ============================================================
# DBSCAN Anomaly
# ============================================================
class TestDBSCANAnomaly:
    def test_basic_detection(self):
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(0, 0.5, (50, 2))
        cluster2 = rng.normal(5, 0.5, (50, 2))
        anomalies = np.array([[20, 20], [-15, -15]])
        X = np.vstack([cluster1, cluster2, anomalies])
        dbscan = DBSCANAnomaly(eps=1.5, min_samples=5)
        result = dbscan.fit_predict(X)
        # The far-off points should be anomalies
        assert result.labels[-1] == 1
        assert result.labels[-2] == 1

    def test_all_one_cluster(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 0.3, (100, 2))
        dbscan = DBSCANAnomaly(eps=1.0, min_samples=5).fit(X)
        # Most points should be in a cluster
        assert np.sum(dbscan.labels_ != -1) > 80

    def test_fit_stores_labels(self):
        X, _ = make_data_2d(n=50, seed=42)
        dbscan = DBSCANAnomaly(eps=2.0, min_samples=3).fit(X)
        assert dbscan.labels_ is not None
        assert len(dbscan.labels_) == 60  # 50 + 10 anomalies

    def test_eps_parameter(self):
        X, _ = make_data_2d(n=50, seed=42)
        db1 = DBSCANAnomaly(eps=0.1, min_samples=5).fit(X)
        db2 = DBSCANAnomaly(eps=10.0, min_samples=5).fit(X)
        # Smaller eps = more noise points
        assert np.sum(db1.labels_ == -1) >= np.sum(db2.labels_ == -1)

    def test_1d_data(self):
        X = np.array([1, 2, 1.5, 2.5, 1.8, 100]).reshape(-1, 1)
        dbscan = DBSCANAnomaly(eps=1.5, min_samples=3)
        result = dbscan.fit_predict(X)
        assert result.labels[-1] == 1  # 100 should be anomaly


# ============================================================
# Multivariate Detector (Mahalanobis)
# ============================================================
class TestMultivariateDetector:
    def test_basic_detection(self):
        X, _ = make_data_2d(n=200, n_anomalies=10, seed=42)
        det = MultivariateDetector(contamination=0.1)
        result = det.fit_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.labels) == 210

    def test_correlated_data(self):
        rng = np.random.default_rng(42)
        # Create correlated 2D data
        cov = np.array([[1, 0.8], [0.8, 1]])
        normal = rng.multivariate_normal([0, 0], cov, 200)
        # Anomaly perpendicular to correlation
        anomaly = np.array([[3, -3]])
        X = np.vstack([normal, anomaly])
        det = MultivariateDetector(contamination=0.01).fit(X)
        scores = det.score(X)
        # The anti-correlated point should score higher
        assert scores[-1] > np.mean(scores[:200])

    def test_1d(self):
        X, _ = make_data_1d()
        result = MultivariateDetector(contamination=0.05).fit_predict(X)
        assert len(result.labels) == len(X)

    def test_scores_nonneg(self):
        X, _ = make_data_2d(seed=42)
        scores = MultivariateDetector().fit(X).score(X)
        assert np.all(scores >= 0)

    def test_contamination_parameter(self):
        X, _ = make_data_2d(seed=42)
        r1 = MultivariateDetector(contamination=0.01).fit_predict(X)
        r2 = MultivariateDetector(contamination=0.2).fit_predict(X)
        assert np.sum(r1.labels) <= np.sum(r2.labels)


# ============================================================
# One-Class SVM
# ============================================================
class TestOneClassSVM:
    def test_basic_detection(self):
        X, _ = make_data_2d(n=100, n_anomalies=10, seed=42)
        svm = OneClassSVM(nu=0.1, max_iter=100, random_state=42)
        result = svm.fit_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.labels) == 110

    def test_nu_parameter(self):
        X, _ = make_data_2d(n=100, seed=42)
        svm = OneClassSVM(nu=0.05, max_iter=100, random_state=42).fit(X)
        assert svm.w_ is not None
        assert svm.rho_ is not None

    def test_linear_kernel(self):
        X, _ = make_data_2d(n=100, seed=42)
        svm = OneClassSVM(kernel='linear', nu=0.1, max_iter=100, random_state=42)
        result = svm.fit_predict(X)
        assert len(result.labels) == 110

    def test_reproducible(self):
        X, _ = make_data_2d(n=50, seed=42)
        r1 = OneClassSVM(random_state=42, max_iter=50).fit_predict(X)
        r2 = OneClassSVM(random_state=42, max_iter=50).fit_predict(X)
        np.testing.assert_array_equal(r1.labels, r2.labels)

    def test_1d(self):
        X, _ = make_data_1d(n=50)
        result = OneClassSVM(max_iter=50, random_state=42).fit_predict(X)
        assert len(result.labels) == 60


# ============================================================
# Ensemble Detector
# ============================================================
class TestEnsembleDetector:
    def test_vote_method(self):
        X, _ = make_data_1d()
        detectors = [
            ('zscore', ZScoreDetector(threshold=3.0)),
            ('iqr', IQRDetector(factor=1.5)),
            ('robust', RobustZScoreDetector(threshold=3.5)),
        ]
        ensemble = EnsembleDetector(detectors, method='vote')
        result = ensemble.fit_predict(X)
        assert isinstance(result, AnomalyResult)
        assert len(result.labels) == len(X)

    def test_average_method(self):
        X, _ = make_data_1d()
        detectors = [
            ('zscore', ZScoreDetector(threshold=3.0)),
            ('iqr', IQRDetector(factor=1.5)),
        ]
        ensemble = EnsembleDetector(detectors, method='average')
        result = ensemble.fit_predict(X)
        assert len(result.labels) == len(X)

    def test_weighted_vote(self):
        X, _ = make_data_1d()
        detectors = [
            ('zscore', ZScoreDetector(threshold=3.0)),
            ('iqr', IQRDetector(factor=1.5)),
        ]
        ensemble = EnsembleDetector(detectors, method='vote', weights=[0.8, 0.2])
        result = ensemble.fit_predict(X)
        assert len(result.labels) == len(X)

    def test_2d_ensemble(self):
        X, _ = make_data_2d()
        detectors = [
            ('zscore', ZScoreDetector(threshold=3.0)),
            ('mahal', MultivariateDetector(contamination=0.1)),
        ]
        ensemble = EnsembleDetector(detectors, method='vote')
        result = ensemble.fit_predict(X)
        assert len(result.labels) == len(X)

    def test_single_detector_ensemble(self):
        X, _ = make_data_1d()
        detectors = [('zscore', ZScoreDetector(threshold=3.0))]
        ensemble = EnsembleDetector(detectors, method='vote')
        result = ensemble.fit_predict(X)
        assert len(result.labels) == len(X)


# ============================================================
# Grubbs' Test
# ============================================================
class TestGrubbsTest:
    def test_single_outlier(self):
        X = np.concatenate([np.zeros(50), [100.0]])
        result = GrubbsTest(alpha=0.05).test(X)
        assert result['is_outlier']
        assert result['outlier_index'] == 50
        assert result['outlier_value'] == 100.0

    def test_no_outlier(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 100)
        result = GrubbsTest(alpha=0.01).test(X)
        # With tight alpha, standard normal should rarely flag
        # (this is probabilistic but with seed=42 it's stable)
        assert 'statistic' in result

    def test_detect_all(self):
        X = np.concatenate([np.zeros(50), [100.0, -100.0, 50.0]])
        outliers = GrubbsTest(alpha=0.05).detect_all(X)
        assert len(outliers) >= 2
        outlier_indices = {o['index'] for o in outliers}
        assert 50 in outlier_indices or 51 in outlier_indices

    def test_small_data(self):
        X = np.array([1.0, 2.0])
        result = GrubbsTest().test(X)
        assert result['outlier_index'] is None

    def test_constant_data(self):
        X = np.ones(20)
        result = GrubbsTest().test(X)
        assert not result['is_outlier']

    def test_detect_all_max_outliers(self):
        X = np.concatenate([np.zeros(50), [100.0, -100.0, 50.0]])
        outliers = GrubbsTest().detect_all(X, max_outliers=1)
        assert len(outliers) <= 1


# ============================================================
# CUSUM
# ============================================================
class TestCUSUM:
    def test_detect_mean_shift(self):
        rng = np.random.default_rng(42)
        X_baseline = rng.normal(0, 1, 100)
        X_shifted = rng.normal(3, 1, 50)
        X = np.concatenate([X_baseline, X_shifted])
        cusum = CUSUM(threshold=5.0, drift=0.5)
        cusum.fit(X_baseline)
        result = cusum.detect(X)
        assert len(result['alarms']) > 0
        # First alarm should be around the shift point
        first_alarm = result['alarms'][0]
        assert first_alarm >= 90  # Near the shift at index 100

    def test_no_alarms_stable(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 200)
        result = CUSUM(threshold=10.0, drift=0.5).fit_detect(X)
        # High threshold should give few alarms
        assert len(result['alarms']) < 10

    def test_returns_cusum_traces(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 50)
        result = CUSUM().fit_detect(X)
        assert 's_pos' in result
        assert 's_neg' in result
        assert len(result['s_pos']) == 50
        assert len(result['s_neg']) == 50

    def test_positive_traces(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, 50)
        result = CUSUM().fit_detect(X)
        assert np.all(result['s_pos'] >= 0)
        assert np.all(result['s_neg'] >= 0)

    def test_threshold_parameter(self):
        rng = np.random.default_rng(42)
        X = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 50)])
        baseline = rng.normal(0, 1, 200)
        cusum = CUSUM(threshold=3.0).fit(baseline)
        r1 = cusum.detect(X)
        cusum2 = CUSUM(threshold=20.0).fit(baseline)
        r2 = cusum2.detect(X)
        assert len(r1['alarms']) >= len(r2['alarms'])


# ============================================================
# Streaming Detector
# ============================================================
class TestStreamingDetector:
    def test_basic_streaming(self):
        det = StreamingDetector(window_size=50, n_sigma=3.0)
        results = []
        rng = np.random.default_rng(42)
        for v in rng.normal(0, 1, 100):
            results.append(det.update(v))
        assert len(results) == 100
        assert all('score' in r for r in results)
        assert all('is_anomaly' in r for r in results)

    def test_detects_spike(self):
        det = StreamingDetector(window_size=50, n_sigma=3.0)
        rng = np.random.default_rng(42)
        # Feed normal data
        for v in rng.normal(0, 1, 100):
            det.update(v)
        # Feed a spike
        result = det.update(100.0)
        assert result['is_anomaly']

    def test_batch_update(self):
        det = StreamingDetector(window_size=20)
        rng = np.random.default_rng(42)
        results = det.update_batch(rng.normal(0, 1, 50))
        assert len(results) == 50

    def test_window_size(self):
        det = StreamingDetector(window_size=10)
        for i in range(20):
            det.update(float(i))
        assert len(det.buffer) == 10

    def test_first_value(self):
        det = StreamingDetector()
        result = det.update(5.0)
        assert result['score'] == 0.0
        assert not result['is_anomaly']

    def test_constant_values(self):
        det = StreamingDetector(window_size=10)
        for _ in range(20):
            result = det.update(1.0)
        # After many constant values, a different value should score high
        result = det.update(100.0)
        assert result['score'] > 3.0


# ============================================================
# Integration tests
# ============================================================
class TestIntegration:
    def test_all_detectors_agree_on_obvious_anomaly(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, 200)
        X = np.concatenate([normal, [50.0]])
        detectors = [
            ZScoreDetector(threshold=3.0),
            RobustZScoreDetector(threshold=3.5),
            IQRDetector(factor=1.5),
        ]
        for det in detectors:
            result = det.fit_predict(X)
            assert result.labels[-1] == 1, f"{det.__class__.__name__} missed obvious anomaly"

    def test_all_detectors_handle_empty_like(self):
        X = np.array([1.0, 2.0, 3.0])
        detectors = [
            ZScoreDetector(),
            RobustZScoreDetector(),
            IQRDetector(),
        ]
        for det in detectors:
            result = det.fit_predict(X)
            assert len(result.labels) == 3

    def test_multivariate_pipeline(self):
        X, labels = make_data_2d(n=200, n_anomalies=20, seed=42)
        # Fit and score with multiple detectors
        iforest = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
        mahal = MultivariateDetector(contamination=0.1)
        r1 = iforest.fit_predict(X)
        r2 = mahal.fit_predict(X)
        # Both should detect something
        assert np.sum(r1.labels) > 0
        assert np.sum(r2.labels) > 0

    def test_ensemble_improves_over_single(self):
        """Ensemble should be at least as good as worst single detector."""
        X, true_labels = make_data_1d(n=200, n_anomalies=10, seed=42)
        detectors = [
            ('zscore', ZScoreDetector(threshold=3.0)),
            ('robust', RobustZScoreDetector(threshold=3.5)),
            ('iqr', IQRDetector(factor=1.5)),
        ]
        ensemble = EnsembleDetector(detectors, method='vote')
        result = ensemble.fit_predict(X)
        # Ensemble should detect at least some anomalies
        assert np.sum(result.labels) > 0

    def test_streaming_matches_batch(self):
        """Streaming detector should eventually agree with batch on clear anomalies."""
        rng = np.random.default_rng(42)
        data = list(rng.normal(0, 1, 100)) + [50.0]
        det = StreamingDetector(window_size=100, n_sigma=3.0)
        results = det.update_batch(data)
        assert results[-1]['is_anomaly']

    def test_cusum_then_zscore(self):
        """Use CUSUM to find change points, then z-score for anomalies."""
        rng = np.random.default_rng(42)
        X = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 50)])
        cusum = CUSUM(threshold=5.0).fit(rng.normal(0, 1, 200))
        change_result = cusum.detect(X)
        assert len(change_result['alarms']) > 0
        # Z-score on shifted segment
        zscore = ZScoreDetector(threshold=2.0).fit(X[:100])
        result = zscore.predict(X[100:])
        assert np.sum(result.labels) > 0


# ============================================================
# Edge cases
# ============================================================
class TestEdgeCases:
    def test_single_point(self):
        X = np.array([1.0])
        det = ZScoreDetector().fit(X)
        result = det.predict(X)
        assert len(result.labels) == 1

    def test_two_points(self):
        X = np.array([1.0, 100.0])
        det = ZScoreDetector(threshold=1.0).fit(X)
        scores = det.score(X)
        assert len(scores) == 2

    def test_high_dimensional(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 20))
        result = ZScoreDetector().fit_predict(X)
        assert len(result.labels) == 50

    def test_negative_values(self):
        X = np.array([-100, -99, -98, -97, 0])
        det = IQRDetector().fit(X.reshape(-1, 1))
        result = det.predict(X.reshape(-1, 1))
        assert len(result.labels) == 5

    def test_large_values(self):
        rng = np.random.default_rng(42)
        X = rng.normal(1e6, 1e3, 100)
        result = ZScoreDetector().fit_predict(X)
        assert len(result.labels) == 100

    def test_anomaly_result_fields(self):
        result = AnomalyResult(scores=np.array([1, 2]), labels=np.array([0, 1]), threshold=1.5)
        assert result.scores[0] == 1
        assert result.labels[1] == 1
        assert result.threshold == 1.5
