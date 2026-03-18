"""V215: Hidden Markov Models.

Complete HMM implementation with all classical algorithms:
1. Forward algorithm -- P(observations | model), alpha values
2. Backward algorithm -- beta values for smoothing
3. Forward-backward (Baum-Welch) -- EM parameter estimation
4. Viterbi algorithm -- most likely state sequence (MAP)
5. Posterior decoding -- per-timestep most likely states
6. Smoothing -- P(state_t | all observations)

Supports:
- Discrete observations with named states/symbols
- Log-space computation for numerical stability
- Multiple observation sequences for training
- Convergence detection with configurable tolerance
- Scaled forward-backward (prevents underflow without full log-space)
- Model scoring, sampling, and simulation
- Stationary distribution computation
"""

from __future__ import annotations

from math import log, exp, inf
from collections import defaultdict
from typing import Sequence


# ---------------------------------------------------------------------------
# Helper: log-sum-exp for numerical stability
# ---------------------------------------------------------------------------

def _logsumexp(values: list[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    if not values:
        return -inf
    max_val = max(values)
    if max_val == -inf:
        return -inf
    return max_val + log(sum(exp(v - max_val) for v in values))


# ---------------------------------------------------------------------------
# HiddenMarkovModel
# ---------------------------------------------------------------------------

class HiddenMarkovModel:
    """Discrete Hidden Markov Model.

    Parameters
    ----------
    states : list[str]
        Hidden state names.
    observations : list[str]
        Observable symbol names.
    initial : dict[str, float]
        Initial state probabilities pi[state].
    transition : dict[str, dict[str, float]]
        Transition probabilities A[from_state][to_state].
    emission : dict[str, dict[str, float]]
        Emission probabilities B[state][observation].
    """

    def __init__(
        self,
        states: list[str],
        observations: list[str],
        initial: dict[str, float],
        transition: dict[str, dict[str, float]],
        emission: dict[str, dict[str, float]],
    ):
        self.states = list(states)
        self.observations = list(observations)
        self.n_states = len(self.states)
        self.n_obs = len(self.observations)

        # Index maps for fast lookup
        self._s2i = {s: i for i, s in enumerate(self.states)}
        self._o2i = {o: i for i, o in enumerate(self.observations)}

        # Store as lists for indexed access
        self.pi = [initial.get(s, 0.0) for s in self.states]
        self.A = [[transition.get(si, {}).get(sj, 0.0)
                    for sj in self.states] for si in self.states]
        self.B = [[emission.get(si, {}).get(oj, 0.0)
                    for oj in self.observations] for si in self.states]

    # --- Log-space parameters ---

    def _log_pi(self) -> list[float]:
        return [log(p) if p > 0 else -inf for p in self.pi]

    def _log_A(self) -> list[list[float]]:
        return [[log(p) if p > 0 else -inf for p in row] for row in self.A]

    def _log_B(self) -> list[list[float]]:
        return [[log(p) if p > 0 else -inf for p in row] for row in self.B]

    # -----------------------------------------------------------------------
    # Forward algorithm: P(O | model)
    # -----------------------------------------------------------------------

    def forward(self, obs_seq: Sequence[str]) -> tuple[list[list[float]], float]:
        """Run forward algorithm in log space.

        Returns
        -------
        alpha : list[list[float]]
            alpha[t][i] = log P(o_1..o_t, s_t=i | model)
        log_prob : float
            log P(obs_seq | model)
        """
        T = len(obs_seq)
        if T == 0:
            return [], 0.0

        log_pi = self._log_pi()
        log_A = self._log_A()
        log_B = self._log_B()
        N = self.n_states

        obs_indices = [self._o2i[o] for o in obs_seq]

        # alpha[t][i]
        alpha = [[0.0] * N for _ in range(T)]

        # t = 0
        for i in range(N):
            alpha[0][i] = log_pi[i] + log_B[i][obs_indices[0]]

        # t = 1..T-1
        for t in range(1, T):
            oi = obs_indices[t]
            for j in range(N):
                alpha[t][j] = _logsumexp(
                    [alpha[t - 1][i] + log_A[i][j] for i in range(N)]
                ) + log_B[j][oi]

        log_prob = _logsumexp(alpha[T - 1])
        return alpha, log_prob

    def log_probability(self, obs_seq: Sequence[str]) -> float:
        """Compute log P(obs_seq | model) using forward algorithm."""
        _, lp = self.forward(obs_seq)
        return lp

    def probability(self, obs_seq: Sequence[str]) -> float:
        """Compute P(obs_seq | model)."""
        return exp(self.log_probability(obs_seq))

    # -----------------------------------------------------------------------
    # Backward algorithm
    # -----------------------------------------------------------------------

    def backward(self, obs_seq: Sequence[str]) -> list[list[float]]:
        """Run backward algorithm in log space.

        Returns
        -------
        beta : list[list[float]]
            beta[t][i] = log P(o_{t+1}..o_T | s_t=i, model)
        """
        T = len(obs_seq)
        if T == 0:
            return []

        log_A = self._log_A()
        log_B = self._log_B()
        N = self.n_states

        obs_indices = [self._o2i[o] for o in obs_seq]

        beta = [[0.0] * N for _ in range(T)]

        # t = T-1: beta[T-1][i] = log(1) = 0
        # already initialized

        # t = T-2..0
        for t in range(T - 2, -1, -1):
            oi_next = obs_indices[t + 1]
            for i in range(N):
                beta[t][i] = _logsumexp(
                    [log_A[i][j] + log_B[j][oi_next] + beta[t + 1][j]
                     for j in range(N)]
                )

        return beta

    # -----------------------------------------------------------------------
    # Smoothing: P(s_t | O_1..O_T)
    # -----------------------------------------------------------------------

    def smooth(self, obs_seq: Sequence[str]) -> list[list[float]]:
        """Compute posterior state probabilities P(s_t=i | O) for each t.

        Returns
        -------
        gamma : list[list[float]]
            gamma[t][i] = P(s_t = i | obs_seq, model)
        """
        alpha, log_prob = self.forward(obs_seq)
        beta = self.backward(obs_seq)
        T = len(obs_seq)
        N = self.n_states

        gamma = [[0.0] * N for _ in range(T)]
        for t in range(T):
            for i in range(N):
                gamma[t][i] = exp(alpha[t][i] + beta[t][i] - log_prob)

        return gamma

    # -----------------------------------------------------------------------
    # Viterbi: most likely state sequence
    # -----------------------------------------------------------------------

    def viterbi(self, obs_seq: Sequence[str]) -> tuple[list[str], float]:
        """Find the most likely state sequence (MAP) via Viterbi.

        Returns
        -------
        path : list[str]
            Most likely state sequence.
        log_prob : float
            Log probability of the best path.
        """
        T = len(obs_seq)
        if T == 0:
            return [], 0.0

        log_pi = self._log_pi()
        log_A = self._log_A()
        log_B = self._log_B()
        N = self.n_states

        obs_indices = [self._o2i[o] for o in obs_seq]

        # delta[t][i] = log prob of best path ending in state i at time t
        delta = [[0.0] * N for _ in range(T)]
        psi = [[0] * N for _ in range(T)]  # backpointers

        # t = 0
        for i in range(N):
            delta[0][i] = log_pi[i] + log_B[i][obs_indices[0]]

        # t = 1..T-1
        for t in range(1, T):
            oi = obs_indices[t]
            for j in range(N):
                best_val = -inf
                best_i = 0
                for i in range(N):
                    val = delta[t - 1][i] + log_A[i][j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                delta[t][j] = best_val + log_B[j][oi]
                psi[t][j] = best_i

        # Backtrack
        best_last = max(range(N), key=lambda i: delta[T - 1][i])
        log_prob = delta[T - 1][best_last]

        path_indices = [0] * T
        path_indices[T - 1] = best_last
        for t in range(T - 2, -1, -1):
            path_indices[t] = psi[t + 1][path_indices[t + 1]]

        path = [self.states[i] for i in path_indices]
        return path, log_prob

    # -----------------------------------------------------------------------
    # Posterior decoding
    # -----------------------------------------------------------------------

    def posterior_decode(self, obs_seq: Sequence[str]) -> list[str]:
        """Decode by picking the most likely state at each timestep.

        Unlike Viterbi, this maximizes P(s_t | O) independently per t,
        which can give a globally impossible sequence but higher expected
        accuracy per position.
        """
        gamma = self.smooth(obs_seq)
        return [self.states[max(range(self.n_states), key=lambda i: g[i])]
                for g in gamma]

    # -----------------------------------------------------------------------
    # Baum-Welch (EM) parameter estimation
    # -----------------------------------------------------------------------

    def baum_welch(
        self,
        sequences: list[list[str]],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> list[float]:
        """Train model parameters using Baum-Welch (EM).

        Parameters
        ----------
        sequences : list of observation sequences
        max_iter : maximum EM iterations
        tol : convergence tolerance on log-likelihood

        Returns
        -------
        log_likelihoods : list[float]
            Log-likelihood after each iteration.
        """
        N = self.n_states
        M = self.n_obs
        history = []

        for iteration in range(max_iter):
            # Accumulators
            pi_acc = [0.0] * N
            a_num = [[0.0] * N for _ in range(N)]
            a_den = [0.0] * N
            b_num = [[0.0] * M for _ in range(N)]
            b_den = [0.0] * N
            total_ll = 0.0

            for obs_seq in sequences:
                T = len(obs_seq)
                if T == 0:
                    continue

                obs_indices = [self._o2i[o] for o in obs_seq]
                alpha, log_prob = self.forward(obs_seq)
                beta = self.backward(obs_seq)
                total_ll += log_prob

                # Gamma: P(s_t=i | O)
                gamma = [[0.0] * N for _ in range(T)]
                for t in range(T):
                    for i in range(N):
                        gamma[t][i] = exp(alpha[t][i] + beta[t][i] - log_prob)

                # Xi: P(s_t=i, s_{t+1}=j | O)
                log_A = self._log_A()
                log_B = self._log_B()

                for t in range(T - 1):
                    oi_next = obs_indices[t + 1]
                    for i in range(N):
                        for j in range(N):
                            xi_ij = exp(
                                alpha[t][i] + log_A[i][j] +
                                log_B[j][oi_next] + beta[t + 1][j] - log_prob
                            )
                            a_num[i][j] += xi_ij
                        a_den[i] += gamma[t][i]

                # Initial state
                for i in range(N):
                    pi_acc[i] += gamma[0][i]

                # Emission
                for t in range(T):
                    oi = obs_indices[t]
                    for i in range(N):
                        b_num[i][oi] += gamma[t][i]
                        b_den[i] += gamma[t][i]

            history.append(total_ll)

            # M-step: update parameters
            n_seq = len(sequences)
            if n_seq > 0:
                total_pi = sum(pi_acc)
                if total_pi > 0:
                    self.pi = [p / total_pi for p in pi_acc]

            for i in range(N):
                if a_den[i] > 0:
                    for j in range(N):
                        self.A[i][j] = a_num[i][j] / a_den[i]
                if b_den[i] > 0:
                    for k in range(M):
                        self.B[i][k] = b_num[i][k] / b_den[i]

            # Convergence check
            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                break

        return history

    # -----------------------------------------------------------------------
    # Sampling / simulation
    # -----------------------------------------------------------------------

    def sample(self, length: int, rng=None) -> tuple[list[str], list[str]]:
        """Generate a random observation sequence from the model.

        Parameters
        ----------
        length : int
            Number of timesteps.
        rng : random.Random, optional
            Random number generator.

        Returns
        -------
        states : list[str]
            Hidden state sequence.
        observations : list[str]
            Observed sequence.
        """
        import random
        if rng is None:
            rng = random.Random()

        def _choose(probs, items):
            r = rng.random()
            cumulative = 0.0
            for item, p in zip(items, probs):
                cumulative += p
                if r < cumulative:
                    return item
            return items[-1]

        state_seq = []
        obs_seq = []

        # Initial state
        s_idx = self.states.index(_choose(self.pi, self.states))
        state_seq.append(self.states[s_idx])
        obs_seq.append(_choose(self.B[s_idx], self.observations))

        for _ in range(1, length):
            s_idx = self.states.index(
                _choose(self.A[s_idx], self.states)
            )
            state_seq.append(self.states[s_idx])
            obs_seq.append(_choose(self.B[s_idx], self.observations))

        return state_seq, obs_seq

    # -----------------------------------------------------------------------
    # Stationary distribution
    # -----------------------------------------------------------------------

    def stationary_distribution(self, max_iter: int = 1000, tol: float = 1e-10) -> dict[str, float]:
        """Compute the stationary distribution of the Markov chain (hidden states).

        Uses power iteration: pi * A = pi.
        """
        N = self.n_states
        # Start uniform
        dist = [1.0 / N] * N

        for _ in range(max_iter):
            new_dist = [0.0] * N
            for j in range(N):
                for i in range(N):
                    new_dist[j] += dist[i] * self.A[i][j]
            # Normalize
            total = sum(new_dist)
            if total > 0:
                new_dist = [p / total for p in new_dist]
            # Convergence
            if max(abs(new_dist[i] - dist[i]) for i in range(N)) < tol:
                dist = new_dist
                break
            dist = new_dist

        return {self.states[i]: dist[i] for i in range(N)}

    # -----------------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------------

    def score_sequences(self, sequences: list[list[str]]) -> float:
        """Total log-likelihood of multiple observation sequences."""
        return sum(self.log_probability(seq) for seq in sequences)

    # -----------------------------------------------------------------------
    # String representation
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"HiddenMarkovModel(states={self.states}, "
                f"observations={self.observations})")


# ---------------------------------------------------------------------------
# Profile HMM (for sequence alignment / motif finding)
# ---------------------------------------------------------------------------

class ProfileHMM:
    """Profile HMM for sequence motif detection.

    A profile HMM has a linear backbone of Match states (M1..Mk),
    with Insert (I0..Ik) and Delete (D1..Dk) states for gap handling.

    This is used in bioinformatics for multiple sequence alignment
    and motif discovery, but the core algorithm is general.
    """

    def __init__(self, motif_length: int, alphabet: list[str]):
        self.k = motif_length
        self.alphabet = list(alphabet)
        self.n_symbols = len(self.alphabet)
        self._sym2i = {s: i for i, s in enumerate(self.alphabet)}

        # States: Begin, M1..Mk, I0..Ik, D1..Dk, End
        # We use index-based representation internally

        # Transition probabilities (log space)
        # From Begin: to M1, I0, D1
        # From Mi: to M(i+1), I(i), D(i+1), End (if i==k)
        # From Ii: to M(i+1), I(i)
        # From Di: to M(i+1), D(i+1)

        # Initialize with reasonable defaults
        self._init_default_params()

    def _init_default_params(self):
        k = self.k
        M = self.n_symbols

        # Match emission: uniform
        self.match_emit = [[1.0 / M] * M for _ in range(k)]  # M1..Mk
        # Insert emission: uniform
        self.insert_emit = [[1.0 / M] * M for _ in range(k + 1)]  # I0..Ik

        # Transition probabilities
        # trans[i] = {state_type: prob} for position i
        # Position 0 = Begin, 1..k = Match positions
        self.trans_mm = [0.9] * (k + 1)   # M_i -> M_{i+1}
        self.trans_mi = [0.05] * (k + 1)  # M_i -> I_i
        self.trans_md = [0.05] * (k + 1)  # M_i -> D_{i+1}
        self.trans_im = [0.9] * (k + 1)   # I_i -> M_{i+1}
        self.trans_ii = [0.1] * (k + 1)   # I_i -> I_i
        self.trans_dm = [0.9] * (k + 1)   # D_i -> M_{i+1}
        self.trans_dd = [0.1] * (k + 1)   # D_i -> D_{i+1}

    def train(self, sequences: list[list[str]], pseudocount: float = 0.01) -> None:
        """Train profile HMM from aligned sequences.

        Each sequence is a list of symbols (or '-' for gaps).
        All sequences must have the same length == motif_length.
        """
        k = self.k
        M = self.n_symbols
        n_seq = len(sequences)
        if n_seq == 0:
            return

        # Count emissions at each match position
        for pos in range(k):
            counts = [pseudocount] * M
            total = pseudocount * M
            for seq in sequences:
                if pos < len(seq) and seq[pos] != '-':
                    idx = self._sym2i.get(seq[pos])
                    if idx is not None:
                        counts[idx] += 1
                        total += 1
            self.match_emit[pos] = [c / total for c in counts]

    def score(self, sequence: list[str]) -> float:
        """Score a sequence against the profile using Viterbi.

        Returns log-probability of the best alignment.
        """
        k = self.k
        T = len(sequence)
        obs = [self._sym2i.get(s, 0) for s in sequence]

        # DP: match[j][t], insert[j][t], delete[j][t]
        # j = 0..k for match (0 is begin), t = 0..T
        NEG_INF = -inf

        # m[j] = best score ending at match j having consumed t symbols
        # We use rolling arrays

        # State: (match_j, insert_j, delete_j) for j=0..k, consuming t=0..T symbols
        m = [[NEG_INF] * (T + 1) for _ in range(k + 1)]
        ins = [[NEG_INF] * (T + 1) for _ in range(k + 1)]
        d = [[NEG_INF] * (T + 1) for _ in range(k + 1)]

        # Begin state: m[0][0] = 0 (begin, consumed 0 symbols)
        m[0][0] = 0.0

        # Insert at position 0
        for t in range(T):
            emit = log(self.insert_emit[0][obs[t]]) if self.insert_emit[0][obs[t]] > 0 else NEG_INF
            # I0 from begin (m[0])
            if m[0][t] > NEG_INF:
                val = m[0][t] + log(self.trans_mi[0]) + emit
                if val > ins[0][t + 1]:
                    ins[0][t + 1] = val
            # I0 self-loop
            if ins[0][t] > NEG_INF:
                val = ins[0][t] + log(self.trans_ii[0]) + emit
                if val > ins[0][t + 1]:
                    ins[0][t + 1] = val

        for j in range(1, k + 1):
            # Delete j: from m[j-1] or d[j-1], no symbol consumed
            for t in range(T + 1):
                candidates = []
                if m[j - 1][t] > NEG_INF:
                    candidates.append(m[j - 1][t] + log(self.trans_md[j - 1]))
                if j >= 2 and d[j - 1][t] > NEG_INF:
                    candidates.append(d[j - 1][t] + log(self.trans_dd[j - 1]))
                if ins[j - 1][t] > NEG_INF:
                    candidates.append(ins[j - 1][t] + log(self.trans_im[j - 1]) + log(self.trans_md[j - 1]))
                if candidates:
                    d[j][t] = max(candidates)

            # Match j: consumes one symbol
            for t in range(T):
                emit = log(self.match_emit[j - 1][obs[t]]) if self.match_emit[j - 1][obs[t]] > 0 else NEG_INF
                if emit == NEG_INF:
                    continue
                candidates = []
                if m[j - 1][t] > NEG_INF:
                    candidates.append(m[j - 1][t] + log(self.trans_mm[j - 1]) + emit)
                if ins[j - 1][t] > NEG_INF:
                    candidates.append(ins[j - 1][t] + log(self.trans_im[j - 1]) + emit)
                if d[j][t] > NEG_INF:
                    candidates.append(d[j][t] + log(self.trans_dm[j]) + emit)
                if j >= 2 and d[j - 1][t] > NEG_INF:
                    candidates.append(d[j - 1][t] + log(self.trans_dm[j - 1]) + emit)
                if candidates:
                    m[j][t + 1] = max(candidates)

            # Insert j
            for t in range(T):
                emit = log(self.insert_emit[j][obs[t]]) if self.insert_emit[j][obs[t]] > 0 else NEG_INF
                if emit == NEG_INF:
                    continue
                if m[j][t] > NEG_INF:
                    val = m[j][t] + log(self.trans_mi[j]) + emit
                    if val > ins[j][t + 1]:
                        ins[j][t + 1] = val
                if ins[j][t] > NEG_INF:
                    val = ins[j][t] + log(self.trans_ii[j]) + emit
                    if val > ins[j][t + 1]:
                        ins[j][t + 1] = val

        # End: best of m[k][T], d[k][T], ins[k][T]
        candidates = []
        for t in range(T + 1):
            if m[k][t] > NEG_INF:
                candidates.append(m[k][t])
            if d[k][t] > NEG_INF:
                candidates.append(d[k][t])
            if ins[k][t] > NEG_INF:
                candidates.append(ins[k][t])

        return max(candidates) if candidates else NEG_INF


# ---------------------------------------------------------------------------
# Coupled HMM (two interacting chains)
# ---------------------------------------------------------------------------

class CoupledHMM:
    """Two coupled Hidden Markov Models with interaction.

    Models two processes whose hidden states influence each other.
    State space is the product of individual state spaces.
    Each chain has its own emission model but transitions depend
    on the joint state.
    """

    def __init__(
        self,
        states1: list[str],
        states2: list[str],
        observations1: list[str],
        observations2: list[str],
        initial1: dict[str, float],
        initial2: dict[str, float],
        transition: dict[tuple[str, str], dict[tuple[str, str], float]],
        emission1: dict[str, dict[str, float]],
        emission2: dict[str, dict[str, float]],
    ):
        self.states1 = list(states1)
        self.states2 = list(states2)
        self.obs1 = list(observations1)
        self.obs2 = list(observations2)

        # Joint state space
        self.joint_states = [(s1, s2) for s1 in self.states1 for s2 in self.states2]
        self.n_joint = len(self.joint_states)
        self._js2i = {js: i for i, js in enumerate(self.joint_states)}
        self._o1_2i = {o: i for i, o in enumerate(self.obs1)}
        self._o2_2i = {o: i for i, o in enumerate(self.obs2)}

        # Initial: product of marginals
        self.pi = [initial1.get(s1, 0.0) * initial2.get(s2, 0.0)
                   for s1, s2 in self.joint_states]

        # Transition: joint -> joint
        N = self.n_joint
        self.A = [[0.0] * N for _ in range(N)]
        for i, js_from in enumerate(self.joint_states):
            for j, js_to in enumerate(self.joint_states):
                self.A[i][j] = transition.get(js_from, {}).get(js_to, 0.0)

        # Emission: factored (each chain emits independently given its state)
        self.B1 = [[emission1.get(s, {}).get(o, 0.0) for o in self.obs1]
                    for s in self.states1]
        self.B2 = [[emission2.get(s, {}).get(o, 0.0) for o in self.obs2]
                    for s in self.states2]

        # Maps from joint state index to individual indices
        self._j2s1 = [self.states1.index(s1) for s1, s2 in self.joint_states]
        self._j2s2 = [self.states2.index(s2) for s1, s2 in self.joint_states]

    def forward(self, obs1_seq: Sequence[str], obs2_seq: Sequence[str]
                ) -> tuple[list[list[float]], float]:
        """Joint forward algorithm in log space."""
        T = len(obs1_seq)
        assert len(obs2_seq) == T
        if T == 0:
            return [], 0.0

        N = self.n_joint
        o1_idx = [self._o1_2i[o] for o in obs1_seq]
        o2_idx = [self._o2_2i[o] for o in obs2_seq]

        log_pi = [log(p) if p > 0 else -inf for p in self.pi]
        log_A = [[log(p) if p > 0 else -inf for p in row] for row in self.A]

        alpha = [[0.0] * N for _ in range(T)]

        # t = 0
        for i in range(N):
            s1i, s2i = self._j2s1[i], self._j2s2[i]
            b1 = self.B1[s1i][o1_idx[0]]
            b2 = self.B2[s2i][o2_idx[0]]
            emit = log(b1 * b2) if b1 > 0 and b2 > 0 else -inf
            alpha[0][i] = log_pi[i] + emit

        for t in range(1, T):
            for j in range(N):
                s1j, s2j = self._j2s1[j], self._j2s2[j]
                b1 = self.B1[s1j][o1_idx[t]]
                b2 = self.B2[s2j][o2_idx[t]]
                emit = log(b1 * b2) if b1 > 0 and b2 > 0 else -inf
                alpha[t][j] = _logsumexp(
                    [alpha[t - 1][i] + log_A[i][j] for i in range(N)]
                ) + emit

        log_prob = _logsumexp(alpha[T - 1])
        return alpha, log_prob

    def viterbi(self, obs1_seq: Sequence[str], obs2_seq: Sequence[str]
                ) -> tuple[list[tuple[str, str]], float]:
        """Joint Viterbi decoding."""
        T = len(obs1_seq)
        assert len(obs2_seq) == T
        if T == 0:
            return [], 0.0

        N = self.n_joint
        o1_idx = [self._o1_2i[o] for o in obs1_seq]
        o2_idx = [self._o2_2i[o] for o in obs2_seq]

        log_pi = [log(p) if p > 0 else -inf for p in self.pi]
        log_A = [[log(p) if p > 0 else -inf for p in row] for row in self.A]

        delta = [[0.0] * N for _ in range(T)]
        psi = [[0] * N for _ in range(T)]

        for i in range(N):
            s1i, s2i = self._j2s1[i], self._j2s2[i]
            b1 = self.B1[s1i][o1_idx[0]]
            b2 = self.B2[s2i][o2_idx[0]]
            emit = log(b1 * b2) if b1 > 0 and b2 > 0 else -inf
            delta[0][i] = log_pi[i] + emit

        for t in range(1, T):
            for j in range(N):
                s1j, s2j = self._j2s1[j], self._j2s2[j]
                b1 = self.B1[s1j][o1_idx[t]]
                b2 = self.B2[s2j][o2_idx[t]]
                emit = log(b1 * b2) if b1 > 0 and b2 > 0 else -inf
                best_val = -inf
                best_i = 0
                for i in range(N):
                    val = delta[t - 1][i] + log_A[i][j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                delta[t][j] = best_val + emit
                psi[t][j] = best_i

        best_last = max(range(N), key=lambda i: delta[T - 1][i])
        log_prob = delta[T - 1][best_last]

        path_indices = [0] * T
        path_indices[T - 1] = best_last
        for t in range(T - 2, -1, -1):
            path_indices[t] = psi[t + 1][path_indices[t + 1]]

        path = [self.joint_states[i] for i in path_indices]
        return path, log_prob
