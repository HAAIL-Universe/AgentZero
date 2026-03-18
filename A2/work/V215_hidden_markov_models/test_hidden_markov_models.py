"""Tests for V215: Hidden Markov Models."""

import pytest
from math import log, exp, isclose, inf
from hidden_markov_models import (
    HiddenMarkovModel, ProfileHMM, CoupledHMM, _logsumexp,
)


# ---------------------------------------------------------------------------
# Fixtures: classic weather HMM
# ---------------------------------------------------------------------------

@pytest.fixture
def weather_hmm():
    """Classic weather HMM: hidden={Rainy, Sunny}, obs={Walk, Shop, Clean}."""
    return HiddenMarkovModel(
        states=["Rainy", "Sunny"],
        observations=["Walk", "Shop", "Clean"],
        initial={"Rainy": 0.6, "Sunny": 0.4},
        transition={
            "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
            "Sunny": {"Rainy": 0.4, "Sunny": 0.6},
        },
        emission={
            "Rainy": {"Walk": 0.1, "Shop": 0.4, "Clean": 0.5},
            "Sunny": {"Walk": 0.6, "Shop": 0.3, "Clean": 0.1},
        },
    )


@pytest.fixture
def coin_hmm():
    """Two-coin HMM: Fair and Biased coins."""
    return HiddenMarkovModel(
        states=["Fair", "Biased"],
        observations=["H", "T"],
        initial={"Fair": 0.5, "Biased": 0.5},
        transition={
            "Fair": {"Fair": 0.9, "Biased": 0.1},
            "Biased": {"Fair": 0.2, "Biased": 0.8},
        },
        emission={
            "Fair": {"H": 0.5, "T": 0.5},
            "Biased": {"H": 0.9, "T": 0.1},
        },
    )


# ---------------------------------------------------------------------------
# Tests: logsumexp
# ---------------------------------------------------------------------------

class TestLogSumExp:
    def test_single_value(self):
        assert isclose(_logsumexp([3.0]), 3.0)

    def test_two_equal(self):
        assert isclose(_logsumexp([0.0, 0.0]), log(2.0))

    def test_large_diff(self):
        # Should not overflow/underflow
        result = _logsumexp([1000.0, 0.0])
        assert isclose(result, 1000.0, rel_tol=1e-10)

    def test_empty(self):
        assert _logsumexp([]) == -inf

    def test_neg_inf(self):
        assert _logsumexp([-inf, -inf]) == -inf

    def test_mixed(self):
        result = _logsumexp([log(0.3), log(0.7)])
        assert isclose(exp(result), 1.0, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Forward algorithm
# ---------------------------------------------------------------------------

class TestForward:
    def test_single_obs(self, weather_hmm):
        alpha, lp = weather_hmm.forward(["Walk"])
        p = exp(lp)
        # P(Walk) = P(Walk|Rainy)*P(Rainy) + P(Walk|Sunny)*P(Sunny)
        expected = 0.1 * 0.6 + 0.6 * 0.4
        assert isclose(p, expected, rel_tol=1e-10)

    def test_two_obs(self, weather_hmm):
        alpha, lp = weather_hmm.forward(["Walk", "Shop"])
        p = exp(lp)
        assert 0 < p < 1

    def test_three_obs(self, weather_hmm):
        _, lp = weather_hmm.forward(["Walk", "Shop", "Clean"])
        assert lp < 0  # log prob is negative

    def test_empty_obs(self, weather_hmm):
        alpha, lp = weather_hmm.forward([])
        assert alpha == []
        assert lp == 0.0

    def test_alpha_shape(self, weather_hmm):
        alpha, _ = weather_hmm.forward(["Walk", "Shop", "Clean"])
        assert len(alpha) == 3
        assert len(alpha[0]) == 2

    def test_alpha_sums(self, weather_hmm):
        """Alpha values at each t should sum (in prob space) to P(o1..ot)."""
        alpha, lp = weather_hmm.forward(["Walk", "Shop"])
        # Sum of alpha at t=1 should equal total probability
        total_t1 = _logsumexp(alpha[1])
        assert isclose(total_t1, lp, rel_tol=1e-10)

    def test_probability_method(self, weather_hmm):
        p = weather_hmm.probability(["Walk", "Shop", "Clean"])
        assert 0 < p < 1

    def test_log_probability_method(self, weather_hmm):
        lp = weather_hmm.log_probability(["Walk", "Shop", "Clean"])
        p = weather_hmm.probability(["Walk", "Shop", "Clean"])
        assert isclose(exp(lp), p, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Backward algorithm
# ---------------------------------------------------------------------------

class TestBackward:
    def test_single_obs(self, weather_hmm):
        beta = weather_hmm.backward(["Walk"])
        assert len(beta) == 1
        # beta[T-1] = log(1) = 0 for all states
        for i in range(2):
            assert isclose(beta[0][i], 0.0)

    def test_two_obs(self, weather_hmm):
        beta = weather_hmm.backward(["Walk", "Shop"])
        assert len(beta) == 2
        # beta[1] should be 0 for all states
        for i in range(2):
            assert isclose(beta[1][i], 0.0)

    def test_empty(self, weather_hmm):
        beta = weather_hmm.backward([])
        assert beta == []

    def test_forward_backward_consistency(self, weather_hmm):
        """P(O) computed via forward should match via backward + initial."""
        obs = ["Walk", "Shop", "Clean"]
        alpha, lp_fwd = weather_hmm.forward(obs)
        beta = weather_hmm.backward(obs)
        log_pi = weather_hmm._log_pi()
        log_B = weather_hmm._log_B()
        o0 = weather_hmm._o2i[obs[0]]

        # P(O) = sum_i pi[i] * B[i][o_0] * beta[0][i]
        lp_bwd = _logsumexp([
            log_pi[i] + log_B[i][o0] + beta[0][i]
            for i in range(weather_hmm.n_states)
        ])
        assert isclose(lp_fwd, lp_bwd, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Smoothing (gamma)
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_probabilities_sum_to_one(self, weather_hmm):
        gamma = weather_hmm.smooth(["Walk", "Shop", "Clean"])
        for t in range(3):
            total = sum(gamma[t])
            assert isclose(total, 1.0, rel_tol=1e-8)

    def test_single_obs(self, weather_hmm):
        gamma = weather_hmm.smooth(["Walk"])
        # P(Rainy|Walk) and P(Sunny|Walk)
        p_rainy = 0.1 * 0.6
        p_sunny = 0.6 * 0.4
        total = p_rainy + p_sunny
        assert isclose(gamma[0][0], p_rainy / total, rel_tol=1e-8)
        assert isclose(gamma[0][1], p_sunny / total, rel_tol=1e-8)

    def test_shape(self, weather_hmm):
        gamma = weather_hmm.smooth(["Walk", "Shop", "Clean"])
        assert len(gamma) == 3
        assert len(gamma[0]) == 2


# ---------------------------------------------------------------------------
# Tests: Viterbi
# ---------------------------------------------------------------------------

class TestViterbi:
    def test_single_obs(self, weather_hmm):
        path, lp = weather_hmm.viterbi(["Walk"])
        assert len(path) == 1
        # Sunny is more likely for Walk (0.6*0.4=0.24 vs 0.1*0.6=0.06)
        assert path[0] == "Sunny"

    def test_three_obs(self, weather_hmm):
        path, lp = weather_hmm.viterbi(["Walk", "Shop", "Clean"])
        assert len(path) == 3
        assert all(s in ["Rainy", "Sunny"] for s in path)
        assert lp < 0

    def test_empty_obs(self, weather_hmm):
        path, lp = weather_hmm.viterbi([])
        assert path == []
        assert lp == 0.0

    def test_all_same_obs(self, weather_hmm):
        path, _ = weather_hmm.viterbi(["Clean", "Clean", "Clean", "Clean"])
        # Clean strongly indicates Rainy
        assert all(s == "Rainy" for s in path)

    def test_viterbi_vs_forward_bound(self, weather_hmm):
        """Viterbi path prob should be <= total probability."""
        obs = ["Walk", "Shop", "Clean"]
        _, lp_viterbi = weather_hmm.viterbi(obs)
        lp_total = weather_hmm.log_probability(obs)
        assert lp_viterbi <= lp_total + 1e-10

    def test_coin_long_heads(self, coin_hmm):
        """Many heads in a row should indicate biased coin."""
        path, _ = coin_hmm.viterbi(["H"] * 10)
        # Should eventually settle into Biased
        assert path[-1] == "Biased"

    def test_coin_alternating(self, coin_hmm):
        """Alternating H/T should indicate fair coin."""
        path, _ = coin_hmm.viterbi(["H", "T"] * 5)
        # Fair coin is more likely for alternating
        assert path.count("Fair") >= path.count("Biased")


# ---------------------------------------------------------------------------
# Tests: Posterior decoding
# ---------------------------------------------------------------------------

class TestPosteriorDecode:
    def test_single_obs(self, weather_hmm):
        decoded = weather_hmm.posterior_decode(["Walk"])
        assert decoded[0] == "Sunny"  # Same as Viterbi for single obs

    def test_shape(self, weather_hmm):
        decoded = weather_hmm.posterior_decode(["Walk", "Shop", "Clean"])
        assert len(decoded) == 3

    def test_all_clean(self, weather_hmm):
        decoded = weather_hmm.posterior_decode(["Clean"] * 5)
        assert all(s == "Rainy" for s in decoded)


# ---------------------------------------------------------------------------
# Tests: Baum-Welch (EM)
# ---------------------------------------------------------------------------

class TestBaumWelch:
    def test_likelihood_increases(self, weather_hmm):
        """Log-likelihood should be monotonically non-decreasing."""
        sequences = [
            ["Walk", "Shop", "Clean", "Walk"],
            ["Clean", "Clean", "Shop"],
            ["Walk", "Walk", "Walk"],
        ]
        history = weather_hmm.baum_welch(sequences, max_iter=20)
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-10

    def test_convergence(self):
        """EM should converge on a simple model."""
        hmm = HiddenMarkovModel(
            states=["A", "B"],
            observations=["X", "Y"],
            initial={"A": 0.5, "B": 0.5},
            transition={
                "A": {"A": 0.5, "B": 0.5},
                "B": {"A": 0.5, "B": 0.5},
            },
            emission={
                "A": {"X": 0.5, "Y": 0.5},
                "B": {"X": 0.5, "Y": 0.5},
            },
        )
        # Train on sequences generated from a known model
        sequences = [["X", "X", "Y"], ["Y", "X", "X"], ["X", "Y", "Y"]]
        history = hmm.baum_welch(sequences, max_iter=50, tol=1e-8)
        assert len(history) >= 2

    def test_parameters_valid(self, weather_hmm):
        """After training, parameters should still be valid distributions."""
        sequences = [["Walk", "Shop"], ["Clean", "Walk"]]
        weather_hmm.baum_welch(sequences, max_iter=10)

        # Initial probs sum to 1
        assert isclose(sum(weather_hmm.pi), 1.0, rel_tol=1e-6)

        # Each transition row sums to 1
        for row in weather_hmm.A:
            assert isclose(sum(row), 1.0, rel_tol=1e-6)

        # Each emission row sums to 1
        for row in weather_hmm.B:
            assert isclose(sum(row), 1.0, rel_tol=1e-6)

    def test_learns_structure(self):
        """EM should learn to distinguish two clearly different states."""
        # State A always emits X, State B always emits Y
        # Generate obvious sequences
        import random
        rng = random.Random(42)
        sequences = []
        for _ in range(20):
            seq = []
            state = rng.choice(["A", "B"])
            for _ in range(10):
                if state == "A":
                    seq.append("X")
                else:
                    seq.append("Y")
                state = "A" if rng.random() < 0.8 else "B"
            sequences.append(seq)

        # Break symmetry with slightly asymmetric initial emission
        hmm = HiddenMarkovModel(
            states=["S1", "S2"],
            observations=["X", "Y"],
            initial={"S1": 0.6, "S2": 0.4},
            transition={
                "S1": {"S1": 0.5, "S2": 0.5},
                "S2": {"S1": 0.5, "S2": 0.5},
            },
            emission={
                "S1": {"X": 0.6, "Y": 0.4},
                "S2": {"X": 0.4, "Y": 0.6},
            },
        )
        hmm.baum_welch(sequences, max_iter=50)

        # One state should mostly emit X, the other Y
        # (we don't know which is which due to label symmetry)
        s1_x = hmm.B[0][0]  # S1 -> X
        s2_x = hmm.B[1][0]  # S2 -> X
        # They should be separated
        assert abs(s1_x - s2_x) > 0.3

    def test_empty_sequence_ignored(self, weather_hmm):
        sequences = [[], ["Walk", "Shop"]]
        history = weather_hmm.baum_welch(sequences, max_iter=5)
        assert len(history) >= 1


# ---------------------------------------------------------------------------
# Tests: Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def test_basic_sample(self, weather_hmm):
        import random
        states, obs = weather_hmm.sample(10, rng=random.Random(42))
        assert len(states) == 10
        assert len(obs) == 10
        assert all(s in ["Rainy", "Sunny"] for s in states)
        assert all(o in ["Walk", "Shop", "Clean"] for o in obs)

    def test_deterministic_with_seed(self, weather_hmm):
        import random
        s1, o1 = weather_hmm.sample(20, rng=random.Random(123))
        s2, o2 = weather_hmm.sample(20, rng=random.Random(123))
        assert s1 == s2
        assert o1 == o2

    def test_sample_length(self, coin_hmm):
        import random
        states, obs = coin_hmm.sample(100, rng=random.Random(0))
        assert len(states) == 100
        assert len(obs) == 100

    def test_sampled_sequences_have_valid_prob(self, weather_hmm):
        import random
        _, obs = weather_hmm.sample(5, rng=random.Random(42))
        p = weather_hmm.probability(obs)
        assert p > 0


# ---------------------------------------------------------------------------
# Tests: Stationary distribution
# ---------------------------------------------------------------------------

class TestStationary:
    def test_weather_stationary(self, weather_hmm):
        dist = weather_hmm.stationary_distribution()
        assert isclose(sum(dist.values()), 1.0, rel_tol=1e-8)
        # A=[0.7, 0.3; 0.4, 0.6] -> pi=[4/7, 3/7]
        assert isclose(dist["Rainy"], 4.0 / 7.0, rel_tol=1e-6)
        assert isclose(dist["Sunny"], 3.0 / 7.0, rel_tol=1e-6)

    def test_absorbing_state(self):
        """If one state is absorbing, stationary should concentrate there."""
        hmm = HiddenMarkovModel(
            states=["A", "B"],
            observations=["X"],
            initial={"A": 1.0, "B": 0.0},
            transition={"A": {"A": 0.0, "B": 1.0}, "B": {"A": 0.0, "B": 1.0}},
            emission={"A": {"X": 1.0}, "B": {"X": 1.0}},
        )
        dist = hmm.stationary_distribution()
        assert isclose(dist["B"], 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Score sequences
# ---------------------------------------------------------------------------

class TestScoreSequences:
    def test_score_single(self, weather_hmm):
        score = weather_hmm.score_sequences([["Walk", "Shop"]])
        lp = weather_hmm.log_probability(["Walk", "Shop"])
        assert isclose(score, lp, rel_tol=1e-10)

    def test_score_multiple(self, weather_hmm):
        seqs = [["Walk"], ["Shop"], ["Clean"]]
        score = weather_hmm.score_sequences(seqs)
        expected = sum(weather_hmm.log_probability(s) for s in seqs)
        assert isclose(score, expected, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Model repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self, weather_hmm):
        r = repr(weather_hmm)
        assert "HiddenMarkovModel" in r
        assert "Rainy" in r


# ---------------------------------------------------------------------------
# Tests: Profile HMM
# ---------------------------------------------------------------------------

class TestProfileHMM:
    def test_create(self):
        phmm = ProfileHMM(motif_length=3, alphabet=["A", "C", "G", "T"])
        assert phmm.k == 3
        assert phmm.n_symbols == 4

    def test_train(self):
        phmm = ProfileHMM(motif_length=3, alphabet=["A", "C", "G", "T"])
        sequences = [
            ["A", "C", "G"],
            ["A", "C", "T"],
            ["A", "G", "G"],
        ]
        phmm.train(sequences)
        # Position 0: all A -> high prob for A
        assert phmm.match_emit[0][0] > 0.5  # A is index 0

    def test_score(self):
        phmm = ProfileHMM(motif_length=3, alphabet=["A", "C", "G", "T"])
        phmm.train([["A", "C", "G"]] * 10)
        # Exact match should score higher
        score_match = phmm.score(["A", "C", "G"])
        score_diff = phmm.score(["T", "T", "T"])
        assert score_match > score_diff

    def test_score_longer_sequence(self):
        phmm = ProfileHMM(motif_length=2, alphabet=["A", "B", "C"])
        phmm.train([["A", "B"]] * 5)
        # Longer sequence uses insertions
        score = phmm.score(["A", "C", "B"])
        assert score > -inf

    def test_score_shorter_sequence(self):
        phmm = ProfileHMM(motif_length=3, alphabet=["A", "B"])
        phmm.train([["A", "B", "A"]] * 5)
        # Shorter sequence uses deletions
        score = phmm.score(["A", "B"])
        assert score > -inf

    def test_empty_training(self):
        phmm = ProfileHMM(motif_length=2, alphabet=["X", "Y"])
        phmm.train([])  # Should not crash
        score = phmm.score(["X", "Y"])
        assert score > -inf


# ---------------------------------------------------------------------------
# Tests: Coupled HMM
# ---------------------------------------------------------------------------

class TestCoupledHMM:
    @pytest.fixture
    def coupled(self):
        states1 = ["Up", "Down"]
        states2 = ["High", "Low"]
        obs1 = ["gain", "loss"]
        obs2 = ["active", "quiet"]

        # Joint transition: (s1, s2) -> (s1', s2')
        transition = {}
        for s1 in states1:
            for s2 in states2:
                row = {}
                for s1p in states1:
                    for s2p in states2:
                        # Simple: mostly stay
                        if s1 == s1p and s2 == s2p:
                            row[(s1p, s2p)] = 0.7
                        else:
                            row[(s1p, s2p)] = 0.1
                transition[(s1, s2)] = row

        return CoupledHMM(
            states1=states1,
            states2=states2,
            observations1=obs1,
            observations2=obs2,
            initial1={"Up": 0.5, "Down": 0.5},
            initial2={"High": 0.5, "Low": 0.5},
            transition=transition,
            emission1={
                "Up": {"gain": 0.8, "loss": 0.2},
                "Down": {"gain": 0.3, "loss": 0.7},
            },
            emission2={
                "High": {"active": 0.9, "quiet": 0.1},
                "Low": {"active": 0.2, "quiet": 0.8},
            },
        )

    def test_forward(self, coupled):
        alpha, lp = coupled.forward(
            ["gain", "gain", "loss"],
            ["active", "active", "quiet"],
        )
        assert len(alpha) == 3
        assert len(alpha[0]) == 4  # 2x2 joint states
        assert lp < 0

    def test_forward_empty(self, coupled):
        alpha, lp = coupled.forward([], [])
        assert alpha == []
        assert lp == 0.0

    def test_viterbi(self, coupled):
        path, lp = coupled.forward(
            ["gain", "gain", "loss"],
            ["active", "active", "quiet"],
        )
        v_path, v_lp = coupled.viterbi(
            ["gain", "gain", "loss"],
            ["active", "active", "quiet"],
        )
        assert len(v_path) == 3
        assert all(isinstance(p, tuple) and len(p) == 2 for p in v_path)
        assert v_lp <= lp + 1e-10  # Viterbi <= total

    def test_viterbi_empty(self, coupled):
        path, lp = coupled.viterbi([], [])
        assert path == []
        assert lp == 0.0

    def test_viterbi_consistent_gains(self, coupled):
        """Consistent gains+active should decode to Up+High."""
        path, _ = coupled.viterbi(
            ["gain"] * 5,
            ["active"] * 5,
        )
        # Should mostly be (Up, High)
        assert all(p == ("Up", "High") for p in path)

    def test_viterbi_consistent_losses(self, coupled):
        """Consistent losses+quiet should decode to Down+Low."""
        path, _ = coupled.viterbi(
            ["loss"] * 5,
            ["quiet"] * 5,
        )
        assert all(p == ("Down", "Low") for p in path)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state(self):
        hmm = HiddenMarkovModel(
            states=["Only"],
            observations=["A", "B"],
            initial={"Only": 1.0},
            transition={"Only": {"Only": 1.0}},
            emission={"Only": {"A": 0.7, "B": 0.3}},
        )
        path, lp = hmm.viterbi(["A", "B", "A"])
        assert path == ["Only", "Only", "Only"]
        p = hmm.probability(["A", "B", "A"])
        assert isclose(p, 0.7 * 0.3 * 0.7, rel_tol=1e-10)

    def test_deterministic_transitions(self):
        """Test with deterministic (0/1) transitions."""
        hmm = HiddenMarkovModel(
            states=["A", "B"],
            observations=["X", "Y"],
            initial={"A": 1.0, "B": 0.0},
            transition={"A": {"A": 0.0, "B": 1.0}, "B": {"A": 1.0, "B": 0.0}},
            emission={"A": {"X": 1.0, "Y": 0.0}, "B": {"X": 0.0, "Y": 1.0}},
        )
        path, _ = hmm.viterbi(["X", "Y", "X", "Y"])
        assert path == ["A", "B", "A", "B"]
        # Probability should be 1.0
        assert isclose(hmm.probability(["X", "Y", "X", "Y"]), 1.0)
        # Impossible sequence
        assert isclose(hmm.probability(["X", "X"]), 0.0, abs_tol=1e-300)

    def test_three_states(self):
        hmm = HiddenMarkovModel(
            states=["S1", "S2", "S3"],
            observations=["a", "b"],
            initial={"S1": 0.5, "S2": 0.3, "S3": 0.2},
            transition={
                "S1": {"S1": 0.5, "S2": 0.3, "S3": 0.2},
                "S2": {"S1": 0.2, "S2": 0.5, "S3": 0.3},
                "S3": {"S1": 0.3, "S2": 0.2, "S3": 0.5},
            },
            emission={
                "S1": {"a": 0.9, "b": 0.1},
                "S2": {"a": 0.5, "b": 0.5},
                "S3": {"a": 0.1, "b": 0.9},
            },
        )
        path, _ = hmm.viterbi(["a", "a", "b", "b"])
        assert len(path) == 4
        # First two should lean S1, last two S3
        assert path[0] == "S1"
        assert path[3] == "S3"

    def test_long_sequence(self, weather_hmm):
        """Ensure no numerical issues with longer sequences."""
        obs = ["Walk", "Shop", "Clean"] * 30  # 90 observations
        path, lp = weather_hmm.viterbi(obs)
        assert len(path) == 90
        assert lp < 0
        # Should not be -inf (log-space prevents underflow)
        assert lp > -inf

    def test_smooth_long_sequence(self, weather_hmm):
        obs = ["Walk", "Shop", "Clean"] * 10
        gamma = weather_hmm.smooth(obs)
        assert len(gamma) == 30
        for g in gamma:
            assert isclose(sum(g), 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Round-trip (sample then decode)
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_sample_then_viterbi(self, weather_hmm):
        """Viterbi on sampled data should produce valid states."""
        import random
        true_states, obs = weather_hmm.sample(20, rng=random.Random(42))
        decoded, _ = weather_hmm.viterbi(obs)
        assert len(decoded) == 20
        assert all(s in ["Rainy", "Sunny"] for s in decoded)

    def test_sample_then_smooth(self, weather_hmm):
        import random
        _, obs = weather_hmm.sample(15, rng=random.Random(99))
        gamma = weather_hmm.smooth(obs)
        assert len(gamma) == 15
        for g in gamma:
            assert isclose(sum(g), 1.0, rel_tol=1e-6)

    def test_baum_welch_on_sampled_data(self):
        """Training on data from a known model should recover structure."""
        import random
        rng = random.Random(42)

        # True model with clear structure
        true_hmm = HiddenMarkovModel(
            states=["A", "B"],
            observations=["X", "Y"],
            initial={"A": 0.8, "B": 0.2},
            transition={"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.2, "B": 0.8}},
            emission={"A": {"X": 0.9, "Y": 0.1}, "B": {"X": 0.1, "Y": 0.9}},
        )

        # Generate training data
        sequences = []
        for _ in range(30):
            _, obs = true_hmm.sample(20, rng=rng)
            sequences.append(obs)

        # Train with slightly asymmetric init to break symmetry
        learner = HiddenMarkovModel(
            states=["S1", "S2"],
            observations=["X", "Y"],
            initial={"S1": 0.6, "S2": 0.4},
            transition={"S1": {"S1": 0.6, "S2": 0.4}, "S2": {"S1": 0.4, "S2": 0.6}},
            emission={"S1": {"X": 0.6, "Y": 0.4}, "S2": {"X": 0.4, "Y": 0.6}},
        )
        history = learner.baum_welch(sequences, max_iter=50)

        # Emissions should separate: one state ~X, other ~Y
        s1_x = learner.B[0][0]
        s2_x = learner.B[1][0]
        assert abs(s1_x - s2_x) > 0.5  # Clear separation


# ---------------------------------------------------------------------------
# Tests: Model comparison
# ---------------------------------------------------------------------------

class TestModelComparison:
    def test_better_model_scores_higher(self):
        """A model matching the data should score higher than a random one."""
        good = HiddenMarkovModel(
            states=["A", "B"],
            observations=["X", "Y"],
            initial={"A": 1.0, "B": 0.0},
            transition={"A": {"A": 0.9, "B": 0.1}, "B": {"A": 0.1, "B": 0.9}},
            emission={"A": {"X": 0.95, "Y": 0.05}, "B": {"X": 0.05, "Y": 0.95}},
        )
        bad = HiddenMarkovModel(
            states=["A", "B"],
            observations=["X", "Y"],
            initial={"A": 0.5, "B": 0.5},
            transition={"A": {"A": 0.5, "B": 0.5}, "B": {"A": 0.5, "B": 0.5}},
            emission={"A": {"X": 0.5, "Y": 0.5}, "B": {"X": 0.5, "Y": 0.5}},
        )
        # Data from good model
        data = [["X", "X", "X", "Y", "Y", "Y"]] * 5
        assert good.score_sequences(data) > bad.score_sequences(data)
