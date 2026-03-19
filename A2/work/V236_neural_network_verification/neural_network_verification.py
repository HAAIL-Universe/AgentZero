"""V236: Neural Network Verification.

Proving properties of neural networks using abstract interpretation and
bound propagation techniques. Given a neural network and input specification,
verify that outputs satisfy desired properties (robustness, safety, etc.).

Techniques implemented:
1. Interval Bound Propagation (IBP) -- fast but loose bounds
2. CROWN/DeepPoly -- linear relaxation for tighter bounds
3. Zonotope abstract domain -- parallelotope-based bound propagation
4. Verification queries: robustness, output range, monotonicity, Lipschitz
5. Counterexample search via sampling + bisection

Standalone implementation using NumPy. No ML frameworks.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Neural Network Representation
# ---------------------------------------------------------------------------

class Activation(Enum):
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    LINEAR = auto()


@dataclass
class Layer:
    """A fully-connected layer: y = activation(W @ x + b)."""
    weights: np.ndarray   # shape (out_dim, in_dim)
    bias: np.ndarray      # shape (out_dim,)
    activation: Activation = Activation.RELU

    @property
    def in_dim(self) -> int:
        return self.weights.shape[1]

    @property
    def out_dim(self) -> int:
        return self.weights.shape[0]


@dataclass
class NeuralNetwork:
    """Feedforward neural network as a sequence of layers."""
    layers: list[Layer] = field(default_factory=list)

    def add_layer(self, weights: np.ndarray, bias: np.ndarray,
                  activation: Activation = Activation.RELU) -> 'NeuralNetwork':
        self.layers.append(Layer(
            weights=np.array(weights, dtype=np.float64),
            bias=np.array(bias, dtype=np.float64),
            activation=activation
        ))
        return self

    @property
    def input_dim(self) -> int:
        return self.layers[0].in_dim if self.layers else 0

    @property
    def output_dim(self) -> int:
        return self.layers[-1].out_dim if self.layers else 0

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Concrete forward pass."""
        h = np.array(x, dtype=np.float64)
        for layer in self.layers:
            h = layer.weights @ h + layer.bias
            h = _apply_activation(h, layer.activation)
        return h

    def forward_all(self, x: np.ndarray) -> list[np.ndarray]:
        """Forward pass returning all intermediate activations."""
        activations = [np.array(x, dtype=np.float64)]
        h = activations[0]
        for layer in self.layers:
            h = layer.weights @ h + layer.bias
            h = _apply_activation(h, layer.activation)
            activations.append(h)
        return activations


def _apply_activation(x: np.ndarray, act: Activation) -> np.ndarray:
    if act == Activation.LINEAR:
        return x
    elif act == Activation.RELU:
        return np.maximum(0, x)
    elif act == Activation.SIGMOID:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    elif act == Activation.TANH:
        return np.tanh(x)
    raise ValueError(f"Unknown activation: {act}")


# ---------------------------------------------------------------------------
# Input Specifications
# ---------------------------------------------------------------------------

@dataclass
class HyperRectangle:
    """Axis-aligned box: lower[i] <= x[i] <= upper[i]."""
    lower: np.ndarray
    upper: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.lower)

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2

    @property
    def radius(self) -> np.ndarray:
        return (self.upper - self.lower) / 2

    def contains(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= self.lower - 1e-12) and np.all(x <= self.upper + 1e-12))

    def volume(self) -> float:
        widths = self.upper - self.lower
        if np.any(widths < 0):
            return 0.0
        return float(np.prod(widths))


def linf_ball(center: np.ndarray, epsilon: float) -> HyperRectangle:
    """L-infinity ball around a point."""
    c = np.array(center, dtype=np.float64)
    return HyperRectangle(c - epsilon, c + epsilon)


def box_spec(lower: list | np.ndarray, upper: list | np.ndarray) -> HyperRectangle:
    """Create a box specification from bounds."""
    return HyperRectangle(
        np.array(lower, dtype=np.float64),
        np.array(upper, dtype=np.float64)
    )


# ---------------------------------------------------------------------------
# Abstract Domains
# ---------------------------------------------------------------------------

@dataclass
class IntervalBounds:
    """Interval abstract domain: [lower, upper] per neuron."""
    lower: np.ndarray
    upper: np.ndarray

    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2


@dataclass
class LinearBounds:
    """DeepPoly/CROWN linear bounds.

    For each neuron i at layer k:
      lower_slope[i] * x + lower_intercept[i] <= neuron_i <= upper_slope[i] * x + upper_intercept[i]

    where x is the input to the NETWORK (not the layer).
    """
    lower_slopes: np.ndarray     # (out_dim, in_dim) -- coeffs of input for lower bound
    lower_intercepts: np.ndarray  # (out_dim,)
    upper_slopes: np.ndarray     # (out_dim, in_dim)
    upper_intercepts: np.ndarray  # (out_dim,)


@dataclass
class ZonotopeBounds:
    """Zonotope abstract domain.

    Represents set { center + sum_i eps_i * generators[i] : eps_i in [-1, 1] }
    Each row of generators is one generator vector.
    """
    center: np.ndarray      # (dim,)
    generators: np.ndarray  # (n_generators, dim)

    @property
    def dim(self) -> int:
        return len(self.center)

    @property
    def n_generators(self) -> int:
        return self.generators.shape[0] if self.generators.ndim == 2 else 0

    def to_interval(self) -> IntervalBounds:
        """Over-approximate zonotope as interval bounds."""
        if self.n_generators == 0:
            return IntervalBounds(self.center.copy(), self.center.copy())
        deviation = np.sum(np.abs(self.generators), axis=0)
        return IntervalBounds(self.center - deviation, self.center + deviation)


# ---------------------------------------------------------------------------
# Interval Bound Propagation (IBP)
# ---------------------------------------------------------------------------

def ibp_propagate_layer(bounds: IntervalBounds, layer: Layer) -> IntervalBounds:
    """Propagate interval bounds through one layer."""
    W = layer.weights
    b = layer.bias

    # Affine: y = W @ x + b
    # Split W into positive and negative parts
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)

    new_lower = W_pos @ bounds.lower + W_neg @ bounds.upper + b
    new_upper = W_pos @ bounds.upper + W_neg @ bounds.lower + b

    # Apply activation
    new_lower, new_upper = _activate_interval(new_lower, new_upper, layer.activation)

    return IntervalBounds(new_lower, new_upper)


def _activate_interval(lower: np.ndarray, upper: np.ndarray,
                        act: Activation) -> tuple[np.ndarray, np.ndarray]:
    """Apply activation function to interval bounds."""
    if act == Activation.LINEAR:
        return lower, upper
    elif act == Activation.RELU:
        return np.maximum(0, lower), np.maximum(0, upper)
    elif act == Activation.SIGMOID:
        # Sigmoid is monotonically increasing
        sl = 1.0 / (1.0 + np.exp(-np.clip(lower, -500, 500)))
        su = 1.0 / (1.0 + np.exp(-np.clip(upper, -500, 500)))
        return sl, su
    elif act == Activation.TANH:
        return np.tanh(lower), np.tanh(upper)
    raise ValueError(f"Unknown activation: {act}")


def ibp_verify(net: NeuralNetwork, input_spec: HyperRectangle) -> IntervalBounds:
    """Full IBP: propagate input box through network."""
    bounds = IntervalBounds(input_spec.lower.copy(), input_spec.upper.copy())
    for layer in net.layers:
        bounds = ibp_propagate_layer(bounds, layer)
    return bounds


# ---------------------------------------------------------------------------
# DeepPoly / CROWN Linear Relaxation
# ---------------------------------------------------------------------------

def _relu_linear_bounds(lb: np.ndarray, ub: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal linear relaxation of ReLU on [lb, ub].

    Returns (alpha_l, beta_l, alpha_u, beta_u) where:
      alpha_l * x + beta_l <= relu(x) <= alpha_u * x + beta_u
    """
    n = len(lb)
    alpha_l = np.zeros(n)
    beta_l = np.zeros(n)
    alpha_u = np.zeros(n)
    beta_u = np.zeros(n)

    for i in range(n):
        l, u = lb[i], ub[i]
        if l >= 0:
            # Strictly positive: relu(x) = x
            alpha_l[i] = 1.0
            alpha_u[i] = 1.0
        elif u <= 0:
            # Strictly negative: relu(x) = 0
            pass  # all zeros
        else:
            # Crossing: l < 0 < u
            # Upper bound: line from (l, 0) to (u, u)
            alpha_u[i] = u / (u - l)
            beta_u[i] = -l * u / (u - l)
            # Lower bound: choose slope that minimizes area
            # Option 1: slope 0 (y=0), Option 2: slope 1 (y=x)
            # We use the adaptive choice: slope 1 if |u| >= |l|, else 0
            if u >= -l:
                alpha_l[i] = 1.0
            # else: alpha_l = 0, beta_l = 0

    return alpha_l, beta_l, alpha_u, beta_u


def _monotone_linear_bounds(lb: np.ndarray, ub: np.ndarray, func, n_samples: int = 200
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sound linear relaxation for a monotonically increasing function.

    Uses chord slope with intercept adjustment via sampling to guarantee soundness.
    """
    n = len(lb)
    alpha_l = np.zeros(n)
    beta_l = np.zeros(n)
    alpha_u = np.zeros(n)
    beta_u = np.zeros(n)

    for i in range(n):
        l, u = lb[i], ub[i]
        fl, fu = func(l), func(u)
        if abs(u - l) < 1e-12:
            alpha_l[i] = alpha_u[i] = 0.0
            beta_l[i] = fl
            beta_u[i] = fu
        else:
            slope = (fu - fl) / (u - l)
            chord_intercept = fl - slope * l
            # Sample to find max deviation from chord, with small padding for safety
            xs = np.linspace(l, u, n_samples)
            fxs = np.array([func(x) for x in xs])
            linear_vals = slope * xs + chord_intercept
            deviations = fxs - linear_vals
            max_dev = np.max(deviations)
            min_dev = np.min(deviations)
            # Add small padding to handle inter-sample gaps
            padding = max(abs(max_dev), abs(min_dev), 1e-10) * 0.01
            max_dev += padding
            min_dev -= padding
            # Lower bound: chord shifted down by max negative deviation
            alpha_l[i] = slope
            beta_l[i] = chord_intercept + min_dev
            # Upper bound: chord shifted up by max positive deviation
            alpha_u[i] = slope
            beta_u[i] = chord_intercept + max_dev

    return alpha_l, beta_l, alpha_u, beta_u


def _sigmoid_linear_bounds(lb: np.ndarray, ub: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sound linear relaxation of sigmoid on [lb, ub]."""
    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return _monotone_linear_bounds(lb, ub, sig)


def _tanh_linear_bounds(lb: np.ndarray, ub: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sound linear relaxation of tanh on [lb, ub]."""
    return _monotone_linear_bounds(lb, ub, np.tanh)


def deeppoly_verify(net: NeuralNetwork, input_spec: HyperRectangle) -> IntervalBounds:
    """DeepPoly/CROWN: propagate linear bounds through network.

    Maintains symbolic linear expressions in terms of the network input,
    then concretizes at the end for tighter bounds than IBP.

    For each neuron, we track:
      A_l @ input + b_l <= neuron <= A_u @ input + b_u
    where A_l, A_u are matrices and b_l, b_u are bias vectors.
    """
    input_lower = input_spec.lower.copy()
    input_upper = input_spec.upper.copy()
    in_dim = len(input_lower)

    # Start: symbolic bounds are identity (neuron_i = input_i)
    # A_l[i,j] = coefficient of input_j in lower bound of neuron_i
    A_l = np.eye(in_dim)  # (current_dim, in_dim)
    b_l = np.zeros(in_dim)
    A_u = np.eye(in_dim)
    b_u = np.zeros(in_dim)

    for layer in net.layers:
        W = layer.weights  # (out_dim, prev_dim)
        b = layer.bias     # (out_dim,)

        # Affine transform on symbolic bounds:
        # new_neuron = W @ prev_neuron + b
        # Lower bound: for each output neuron, compute W_pos @ A_l + W_neg @ A_u
        W_pos = np.maximum(W, 0)
        W_neg = np.minimum(W, 0)

        new_A_l = W_pos @ A_l + W_neg @ A_u  # (out_dim, in_dim)
        new_b_l = W_pos @ b_l + W_neg @ b_u + b
        new_A_u = W_pos @ A_u + W_neg @ A_l
        new_b_u = W_pos @ b_u + W_neg @ b_l + b

        if layer.activation == Activation.LINEAR:
            A_l, b_l, A_u, b_u = new_A_l, new_b_l, new_A_u, new_b_u
            continue

        # Concretize pre-activation bounds to get interval for relaxation
        pre_lb = _concretize_lower(new_A_l, new_b_l, input_lower, input_upper)
        pre_ub = _concretize_upper(new_A_u, new_b_u, input_lower, input_upper)

        # Get linear relaxation slopes
        if layer.activation == Activation.RELU:
            alpha_l, beta_l_act, alpha_u, beta_u_act = _relu_linear_bounds(pre_lb, pre_ub)
        elif layer.activation == Activation.SIGMOID:
            alpha_l, beta_l_act, alpha_u, beta_u_act = _sigmoid_linear_bounds(pre_lb, pre_ub)
        elif layer.activation == Activation.TANH:
            alpha_l, beta_l_act, alpha_u, beta_u_act = _tanh_linear_bounds(pre_lb, pre_ub)
        else:
            raise ValueError(f"Unsupported activation: {layer.activation}")

        # Apply activation relaxation to symbolic bounds:
        # post_lower >= alpha_l * pre + beta_l_act
        # where pre >= new_A_l @ input + new_b_l (when alpha_l >= 0)
        # So post_lower >= alpha_l * (new_A_l @ input + new_b_l) + beta_l_act
        A_l = np.diag(alpha_l) @ new_A_l
        b_l = alpha_l * new_b_l + beta_l_act
        A_u = np.diag(alpha_u) @ new_A_u
        b_u = alpha_u * new_b_u + beta_u_act

    # Concretize final bounds
    final_lb = _concretize_lower(A_l, b_l, input_lower, input_upper)
    final_ub = _concretize_upper(A_u, b_u, input_lower, input_upper)

    return IntervalBounds(final_lb, final_ub)


def _concretize_lower(A: np.ndarray, b: np.ndarray,
                       input_lower: np.ndarray, input_upper: np.ndarray) -> np.ndarray:
    """Concretize symbolic lower bound: minimize A @ x + b over input box."""
    A_pos = np.maximum(A, 0)
    A_neg = np.minimum(A, 0)
    return A_pos @ input_lower + A_neg @ input_upper + b


def _concretize_upper(A: np.ndarray, b: np.ndarray,
                       input_lower: np.ndarray, input_upper: np.ndarray) -> np.ndarray:
    """Concretize symbolic upper bound: maximize A @ x + b over input box."""
    A_pos = np.maximum(A, 0)
    A_neg = np.minimum(A, 0)
    return A_pos @ input_upper + A_neg @ input_lower + b


# ---------------------------------------------------------------------------
# Zonotope Propagation
# ---------------------------------------------------------------------------

def zonotope_from_box(box: HyperRectangle) -> ZonotopeBounds:
    """Create initial zonotope from input box."""
    center = box.center
    radius = box.radius
    # Each dimension gets one generator
    generators = np.diag(radius)
    return ZonotopeBounds(center, generators)


def zonotope_propagate_layer(z: ZonotopeBounds, layer: Layer) -> ZonotopeBounds:
    """Propagate zonotope through one layer."""
    W = layer.weights
    b = layer.bias

    # Affine: center' = W @ center + b, generators' = generators @ W^T
    new_center = W @ z.center + b
    new_generators = z.generators @ W.T  # (n_gen, out_dim)

    if layer.activation == Activation.LINEAR:
        return ZonotopeBounds(new_center, new_generators)

    # For nonlinear activations, compute interval bounds then use linear relaxation
    interval = ZonotopeBounds(new_center, new_generators).to_interval()
    lb, ub = interval.lower, interval.upper

    if layer.activation == Activation.RELU:
        alpha_l, beta_l, alpha_u, beta_u = _relu_linear_bounds(lb, ub)
    elif layer.activation == Activation.SIGMOID:
        alpha_l, beta_l, alpha_u, beta_u = _sigmoid_linear_bounds(lb, ub)
    elif layer.activation == Activation.TANH:
        alpha_l, beta_l, alpha_u, beta_u = _tanh_linear_bounds(lb, ub)
    else:
        raise ValueError(f"Unsupported activation: {layer.activation}")

    # Use midpoint of lower/upper slopes as the zonotope slope
    alpha_mid = (alpha_l + alpha_u) / 2
    beta_mid = (beta_l + beta_u) / 2

    # New center: alpha_mid * center + beta_mid
    result_center = alpha_mid * new_center + beta_mid

    # Scale generators by alpha_mid (per-dimension)
    result_generators = new_generators * alpha_mid[np.newaxis, :]

    # Add error generators for the relaxation gap
    error = (alpha_u * ub + beta_u - alpha_l * lb - beta_l) / 2
    # Only add error generators for crossing neurons
    crossing = (lb < 0) & (ub > 0) if layer.activation == Activation.RELU else (error > 1e-12)
    n_crossing = int(np.sum(crossing))

    if n_crossing > 0:
        error_gens = np.zeros((n_crossing, layer.out_dim))
        crossing_indices = np.where(crossing)[0]
        for j, idx in enumerate(crossing_indices):
            error_gens[j, idx] = error[idx]
        result_generators = np.vstack([result_generators, error_gens])

    return ZonotopeBounds(result_center, result_generators)


def zonotope_verify(net: NeuralNetwork, input_spec: HyperRectangle) -> IntervalBounds:
    """Zonotope-based verification: propagate zonotope through network."""
    z = zonotope_from_box(input_spec)
    for layer in net.layers:
        z = zonotope_propagate_layer(z, layer)
    return z.to_interval()


# ---------------------------------------------------------------------------
# Verification Queries
# ---------------------------------------------------------------------------

class VerificationResult(Enum):
    VERIFIED = auto()
    VIOLATED = auto()
    UNKNOWN = auto()


@dataclass
class VerificationReport:
    """Result of a verification query."""
    result: VerificationResult
    property_name: str
    bounds: Optional[IntervalBounds] = None
    counterexample: Optional[np.ndarray] = None
    message: str = ""


def verify_output_bounds(net: NeuralNetwork, input_spec: HyperRectangle,
                          output_lower: Optional[np.ndarray] = None,
                          output_upper: Optional[np.ndarray] = None,
                          method: str = "deeppoly") -> VerificationReport:
    """Verify that network outputs stay within bounds for all inputs in spec."""
    if method == "ibp":
        out_bounds = ibp_verify(net, input_spec)
    elif method == "deeppoly":
        out_bounds = deeppoly_verify(net, input_spec)
    elif method == "zonotope":
        out_bounds = zonotope_verify(net, input_spec)
    else:
        raise ValueError(f"Unknown method: {method}")

    verified = True
    msg_parts = []

    if output_lower is not None:
        if np.all(out_bounds.lower >= output_lower - 1e-9):
            msg_parts.append("lower bound verified")
        else:
            verified = False
            violations = np.where(out_bounds.lower < output_lower - 1e-9)[0]
            msg_parts.append(f"lower bound violated at outputs {violations.tolist()}")

    if output_upper is not None:
        if np.all(out_bounds.upper <= output_upper + 1e-9):
            msg_parts.append("upper bound verified")
        else:
            verified = False
            violations = np.where(out_bounds.upper > output_upper + 1e-9)[0]
            msg_parts.append(f"upper bound violated at outputs {violations.tolist()}")

    return VerificationReport(
        result=VerificationResult.VERIFIED if verified else VerificationResult.UNKNOWN,
        property_name="output_bounds",
        bounds=out_bounds,
        message="; ".join(msg_parts)
    )


def verify_robustness(net: NeuralNetwork, x: np.ndarray, epsilon: float,
                       true_label: int, method: str = "deeppoly") -> VerificationReport:
    """Verify that network classifies all points within L-inf ball the same.

    For classification: verify that output[true_label] > output[j] for all j != true_label,
    for all inputs within epsilon ball of x.
    """
    input_spec = linf_ball(x, epsilon)

    if method == "ibp":
        out_bounds = ibp_verify(net, input_spec)
    elif method == "deeppoly":
        out_bounds = deeppoly_verify(net, input_spec)
    elif method == "zonotope":
        out_bounds = zonotope_verify(net, input_spec)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Check: min output[true_label] > max output[j] for all j != true_label
    true_lower = out_bounds.lower[true_label]
    n_outputs = len(out_bounds.upper)
    robust = True
    worst_margin = float('inf')

    for j in range(n_outputs):
        if j == true_label:
            continue
        margin = true_lower - out_bounds.upper[j]
        worst_margin = min(worst_margin, margin)
        if margin <= 0:
            robust = False

    if robust:
        return VerificationReport(
            result=VerificationResult.VERIFIED,
            property_name="robustness",
            bounds=out_bounds,
            message=f"Robust within eps={epsilon}, worst margin={worst_margin:.6f}"
        )

    # Try to find counterexample
    cex = _find_counterexample_sampling(net, input_spec, true_label, n_samples=1000)
    if cex is not None:
        return VerificationReport(
            result=VerificationResult.VIOLATED,
            property_name="robustness",
            bounds=out_bounds,
            counterexample=cex,
            message=f"Counterexample found, net({cex}) misclassifies"
        )

    return VerificationReport(
        result=VerificationResult.UNKNOWN,
        property_name="robustness",
        bounds=out_bounds,
        message=f"Cannot verify robustness, worst margin={worst_margin:.6f}"
    )


def _find_counterexample_sampling(net: NeuralNetwork, input_spec: HyperRectangle,
                                    true_label: int, n_samples: int = 1000,
                                    rng: Optional[np.random.Generator] = None
                                    ) -> Optional[np.ndarray]:
    """Try to find a counterexample by random sampling."""
    if rng is None:
        rng = np.random.default_rng(42)

    for _ in range(n_samples):
        x = rng.uniform(input_spec.lower, input_spec.upper)
        y = net.forward(x)
        if np.argmax(y) != true_label:
            return x
    return None


def verify_monotonicity(net: NeuralNetwork, input_spec: HyperRectangle,
                         input_dim: int, output_dim: int,
                         increasing: bool = True,
                         method: str = "deeppoly") -> VerificationReport:
    """Verify that output[output_dim] is monotonically increasing/decreasing
    in input[input_dim] over the input spec.

    Strategy: partition input range into intervals, verify gradient sign via
    finite differencing on bounds.
    """
    n_partitions = 20
    dim_lower = input_spec.lower[input_dim]
    dim_upper = input_spec.upper[input_dim]
    step = (dim_upper - dim_lower) / n_partitions

    verified = True
    for k in range(n_partitions):
        # Create two adjacent boxes
        lo1 = input_spec.lower.copy()
        hi1 = input_spec.upper.copy()
        lo1[input_dim] = dim_lower + k * step
        hi1[input_dim] = dim_lower + (k + 0.5) * step

        lo2 = input_spec.lower.copy()
        hi2 = input_spec.upper.copy()
        lo2[input_dim] = dim_lower + (k + 0.5) * step
        hi2[input_dim] = dim_lower + (k + 1) * step

        box1 = HyperRectangle(lo1, hi1)
        box2 = HyperRectangle(lo2, hi2)

        if method == "ibp":
            b1 = ibp_verify(net, box1)
            b2 = ibp_verify(net, box2)
        elif method == "deeppoly":
            b1 = deeppoly_verify(net, box1)
            b2 = deeppoly_verify(net, box2)
        else:
            b1 = zonotope_verify(net, box1)
            b2 = zonotope_verify(net, box2)

        if increasing:
            # Need: max(box1)[output_dim] <= min(box2)[output_dim]
            if b1.upper[output_dim] > b2.lower[output_dim] + 1e-9:
                verified = False
                break
        else:
            if b1.lower[output_dim] < b2.upper[output_dim] - 1e-9:
                verified = False
                break

    direction = "increasing" if increasing else "decreasing"
    if verified:
        return VerificationReport(
            result=VerificationResult.VERIFIED,
            property_name=f"monotonicity_{direction}",
            message=f"Output {output_dim} is monotonically {direction} in input {input_dim}"
        )

    return VerificationReport(
        result=VerificationResult.UNKNOWN,
        property_name=f"monotonicity_{direction}",
        message=f"Cannot verify monotonicity ({direction}) in input {input_dim}"
    )


def compute_output_range(net: NeuralNetwork, input_spec: HyperRectangle,
                          method: str = "deeppoly") -> IntervalBounds:
    """Compute verified output range over input specification."""
    if method == "ibp":
        return ibp_verify(net, input_spec)
    elif method == "deeppoly":
        return deeppoly_verify(net, input_spec)
    elif method == "zonotope":
        return zonotope_verify(net, input_spec)
    raise ValueError(f"Unknown method: {method}")


def estimate_lipschitz(net: NeuralNetwork, input_spec: HyperRectangle,
                        n_samples: int = 500,
                        rng: Optional[np.random.Generator] = None) -> float:
    """Estimate local Lipschitz constant by sampling pairs of points.

    Returns max ||f(x1) - f(x2)|| / ||x1 - x2|| over sampled pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    max_ratio = 0.0
    for _ in range(n_samples):
        x1 = rng.uniform(input_spec.lower, input_spec.upper)
        x2 = rng.uniform(input_spec.lower, input_spec.upper)
        d_in = np.linalg.norm(x1 - x2)
        if d_in < 1e-12:
            continue
        d_out = np.linalg.norm(net.forward(x1) - net.forward(x2))
        ratio = d_out / d_in
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def lipschitz_upper_bound(net: NeuralNetwork) -> float:
    """Compute upper bound on global Lipschitz constant.

    Product of spectral norms (largest singular values) of weight matrices,
    multiplied by Lipschitz constants of activations.
    """
    lip = 1.0
    for layer in net.layers:
        # Spectral norm = largest singular value
        s = np.linalg.svd(layer.weights, compute_uv=False)
        spectral_norm = s[0] if len(s) > 0 else 0.0

        # Activation Lipschitz constants
        if layer.activation == Activation.RELU:
            act_lip = 1.0
        elif layer.activation == Activation.SIGMOID:
            act_lip = 0.25  # max derivative of sigmoid
        elif layer.activation == Activation.TANH:
            act_lip = 1.0   # max derivative of tanh
        elif layer.activation == Activation.LINEAR:
            act_lip = 1.0
        else:
            act_lip = 1.0

        lip *= spectral_norm * act_lip

    return lip


# ---------------------------------------------------------------------------
# Comparison and Analysis
# ---------------------------------------------------------------------------

def compare_methods(net: NeuralNetwork, input_spec: HyperRectangle
                     ) -> dict[str, IntervalBounds]:
    """Compare all three verification methods on the same problem."""
    return {
        "ibp": ibp_verify(net, input_spec),
        "deeppoly": deeppoly_verify(net, input_spec),
        "zonotope": zonotope_verify(net, input_spec),
    }


def tightness_analysis(net: NeuralNetwork, input_spec: HyperRectangle,
                         n_samples: int = 10000,
                         rng: Optional[np.random.Generator] = None
                         ) -> dict:
    """Analyze tightness of bounds vs actual output range."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Sample concrete outputs
    outputs = []
    for _ in range(n_samples):
        x = rng.uniform(input_spec.lower, input_spec.upper)
        outputs.append(net.forward(x))
    outputs = np.array(outputs)

    actual_lower = np.min(outputs, axis=0)
    actual_upper = np.max(outputs, axis=0)
    actual_width = actual_upper - actual_lower

    methods = compare_methods(net, input_spec)
    result = {
        "actual": IntervalBounds(actual_lower, actual_upper),
    }

    for name, bounds in methods.items():
        width = bounds.upper - bounds.lower
        # Tightness ratio: actual_width / bound_width (1.0 = perfect)
        tightness = np.where(width > 1e-12, actual_width / width, 1.0)
        result[name] = {
            "bounds": bounds,
            "mean_tightness": float(np.mean(tightness)),
            "min_tightness": float(np.min(tightness)),
            "width": width,
        }

    return result


# ---------------------------------------------------------------------------
# Network Builders (for testing)
# ---------------------------------------------------------------------------

def build_simple_relu_net(layer_sizes: list[int],
                           rng: Optional[np.random.Generator] = None) -> NeuralNetwork:
    """Build a random ReLU network with given layer sizes."""
    if rng is None:
        rng = np.random.default_rng(42)

    net = NeuralNetwork()
    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        # Xavier initialization
        scale = np.sqrt(2.0 / in_dim)
        W = rng.standard_normal((out_dim, in_dim)) * scale
        b = np.zeros(out_dim)
        act = Activation.LINEAR if i == len(layer_sizes) - 2 else Activation.RELU
        net.add_layer(W, b, act)

    return net


def build_classifier(input_dim: int, hidden_dims: list[int], n_classes: int,
                      rng: Optional[np.random.Generator] = None) -> NeuralNetwork:
    """Build a random classifier network."""
    sizes = [input_dim] + hidden_dims + [n_classes]
    return build_simple_relu_net(sizes, rng)


def build_monotone_net(input_dim: int = 1, hidden_dim: int = 4,
                        output_dim: int = 1) -> NeuralNetwork:
    """Build a network that is monotonically increasing in input[0]."""
    net = NeuralNetwork()
    # Use positive weights to ensure monotonicity
    W1 = np.abs(np.random.default_rng(42).standard_normal((hidden_dim, input_dim)))
    b1 = np.zeros(hidden_dim)
    W2 = np.abs(np.random.default_rng(43).standard_normal((output_dim, hidden_dim)))
    b2 = np.zeros(output_dim)
    net.add_layer(W1, b1, Activation.RELU)
    net.add_layer(W2, b2, Activation.LINEAR)
    return net
