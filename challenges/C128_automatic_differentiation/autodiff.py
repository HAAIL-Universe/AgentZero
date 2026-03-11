"""
C128: Automatic Differentiation

A complete automatic differentiation library featuring:
1. Dual -- dual numbers for forward-mode AD
2. ForwardAD -- forward-mode AD (directional derivatives, JVP)
3. Var -- computation graph nodes for reverse-mode AD
4. ReverseAD -- reverse-mode AD (gradients, VJP, backpropagation)
5. jacobian / hessian -- higher-order derivative utilities
6. grad / value_and_grad -- convenient gradient functions
7. checkpoint -- memory-efficient reverse-mode via recomputation
8. ADOptimizer -- gradient-based optimization using AD (composes C127 concepts)

Composes: C127 (ConvexOptimization) VectorOps concepts, standalone implementation.
No numpy dependency -- all math from scratch.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any, Union
from enum import Enum
import math


# ---------------------------------------------------------------------------
# Forward Mode: Dual Numbers
# ---------------------------------------------------------------------------

class Dual:
    """Dual number for forward-mode automatic differentiation.

    A dual number a + b*epsilon where epsilon^2 = 0.
    The real part carries the value, the dual part carries the derivative.
    """
    __slots__ = ('val', 'der')

    def __init__(self, val: float, der: float = 0.0):
        self.val = float(val)
        self.der = float(der)

    def __repr__(self):
        return f"Dual({self.val}, {self.der})"

    def __float__(self):
        return self.val

    def __int__(self):
        return int(self.val)

    # Arithmetic
    def __add__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der)
        other = float(other)
        return Dual(self.val + other, self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.der - other.der)
        other = float(other)
        return Dual(self.val - other, self.der)

    def __rsub__(self, other):
        other = float(other)
        return Dual(other - self.val, -self.der)

    def __mul__(self, other):
        if isinstance(other, Dual):
            return Dual(self.val * other.val, self.val * other.der + self.der * other.val)
        other = float(other)
        return Dual(self.val * other, self.der * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Dual):
            return Dual(
                self.val / other.val,
                (self.der * other.val - self.val * other.der) / (other.val * other.val)
            )
        other = float(other)
        return Dual(self.val / other, self.der / other)

    def __rtruediv__(self, other):
        other = float(other)
        return Dual(
            other / self.val,
            -other * self.der / (self.val * self.val)
        )

    def __pow__(self, other):
        if isinstance(other, Dual):
            # f^g: d/dx = f^g * (g' * ln(f) + g * f'/f)
            if self.val <= 0:
                # Handle integer powers of negative numbers
                if other.der == 0 and other.val == int(other.val):
                    n = int(other.val)
                    result_val = self.val ** n
                    result_der = n * (self.val ** (n - 1)) * self.der
                    return Dual(result_val, result_der)
                raise ValueError("Cannot raise negative number to non-integer power in AD")
            ln_f = math.log(self.val)
            result_val = self.val ** other.val
            result_der = result_val * (other.der * ln_f + other.val * self.der / self.val)
            return Dual(result_val, result_der)
        n = float(other)
        if n == 0:
            return Dual(1.0, 0.0)
        if n == 1:
            return Dual(self.val, self.der)
        return Dual(self.val ** n, n * (self.val ** (n - 1)) * self.der)

    def __rpow__(self, other):
        # other^self where other is a constant
        other = float(other)
        if other <= 0:
            raise ValueError("Cannot raise non-positive base to variable power")
        result_val = other ** self.val
        result_der = result_val * math.log(other) * self.der
        return Dual(result_val, result_der)

    def __neg__(self):
        return Dual(-self.val, -self.der)

    def __abs__(self):
        if self.val > 0:
            return Dual(self.val, self.der)
        elif self.val < 0:
            return Dual(-self.val, -self.der)
        return Dual(0.0, 0.0)  # subgradient at 0

    # Comparison (by value only)
    def __eq__(self, other):
        if isinstance(other, Dual):
            return self.val == other.val
        return self.val == float(other)

    def __lt__(self, other):
        if isinstance(other, Dual):
            return self.val < other.val
        return self.val < float(other)

    def __le__(self, other):
        if isinstance(other, Dual):
            return self.val <= other.val
        return self.val <= float(other)

    def __gt__(self, other):
        if isinstance(other, Dual):
            return self.val > other.val
        return self.val > float(other)

    def __ge__(self, other):
        if isinstance(other, Dual):
            return self.val >= other.val
        return self.val >= float(other)

    def __hash__(self):
        return hash(self.val)


# ---------------------------------------------------------------------------
# Dual number math functions
# ---------------------------------------------------------------------------

def dual_sin(x):
    if isinstance(x, Dual):
        return Dual(math.sin(x.val), math.cos(x.val) * x.der)
    return math.sin(x)

def dual_cos(x):
    if isinstance(x, Dual):
        return Dual(math.cos(x.val), -math.sin(x.val) * x.der)
    return math.cos(x)

def dual_tan(x):
    if isinstance(x, Dual):
        c = math.cos(x.val)
        return Dual(math.tan(x.val), x.der / (c * c))
    return math.tan(x)

def dual_exp(x):
    if isinstance(x, Dual):
        e = math.exp(x.val)
        return Dual(e, e * x.der)
    return math.exp(x)

def dual_log(x):
    if isinstance(x, Dual):
        return Dual(math.log(x.val), x.der / x.val)
    return math.log(x)

def dual_sqrt(x):
    if isinstance(x, Dual):
        s = math.sqrt(x.val)
        return Dual(s, x.der / (2.0 * s))
    return math.sqrt(x)

def dual_abs(x):
    if isinstance(x, Dual):
        return abs(x)
    return abs(x)

def dual_tanh(x):
    if isinstance(x, Dual):
        t = math.tanh(x.val)
        return Dual(t, (1 - t * t) * x.der)
    return math.tanh(x)

def dual_sigmoid(x):
    if isinstance(x, Dual):
        s = 1.0 / (1.0 + math.exp(-x.val))
        return Dual(s, s * (1 - s) * x.der)
    s = 1.0 / (1.0 + math.exp(-x))
    return s

def dual_relu(x):
    if isinstance(x, Dual):
        if x.val > 0:
            return Dual(x.val, x.der)
        return Dual(0.0, 0.0)
    return max(0.0, x)

def dual_max(a, b):
    """Differentiable max (subgradient at tie goes to a)."""
    a_is_dual = isinstance(a, Dual)
    b_is_dual = isinstance(b, Dual)
    a_val = a.val if a_is_dual else float(a)
    b_val = b.val if b_is_dual else float(b)
    if a_val >= b_val:
        if a_is_dual:
            return Dual(a_val, a.der)
        return Dual(a_val, 0.0)
    else:
        if b_is_dual:
            return Dual(b_val, b.der)
        return Dual(b_val, 0.0)

def dual_min(a, b):
    """Differentiable min."""
    a_is_dual = isinstance(a, Dual)
    b_is_dual = isinstance(b, Dual)
    a_val = a.val if a_is_dual else float(a)
    b_val = b.val if b_is_dual else float(b)
    if a_val <= b_val:
        if a_is_dual:
            return Dual(a_val, a.der)
        return Dual(a_val, 0.0)
    else:
        if b_is_dual:
            return Dual(b_val, b.der)
        return Dual(b_val, 0.0)


# ---------------------------------------------------------------------------
# Forward Mode AD Engine
# ---------------------------------------------------------------------------

class ForwardAD:
    """Forward-mode automatic differentiation using dual numbers.

    Best for: f: R^n -> R^m where m >> n (many outputs, few inputs).
    One forward pass per input dimension gives one column of the Jacobian.
    """

    @staticmethod
    def derivative(f: Callable, x: float) -> Tuple[float, float]:
        """Compute f(x) and f'(x) for scalar function."""
        d = Dual(x, 1.0)
        result = f(d)
        if isinstance(result, Dual):
            return (result.val, result.der)
        return (float(result), 0.0)

    @staticmethod
    def partial(f: Callable, x: List[float], i: int) -> Tuple[float, float]:
        """Compute f(x) and df/dx_i at point x."""
        duals = [Dual(xi, 1.0 if j == i else 0.0) for j, xi in enumerate(x)]
        result = f(duals)
        if isinstance(result, Dual):
            return (result.val, result.der)
        return (float(result), 0.0)

    @staticmethod
    def gradient(f: Callable, x: List[float]) -> Tuple[float, List[float]]:
        """Compute f(x) and full gradient for f: R^n -> R."""
        n = len(x)
        grad = [0.0] * n
        val = None
        for i in range(n):
            v, d = ForwardAD.partial(f, x, i)
            grad[i] = d
            if val is None:
                val = v
        return (val, grad)

    @staticmethod
    def jvp(f: Callable, x: List[float], v: List[float]) -> Tuple[Any, Any]:
        """Jacobian-vector product: J(x) @ v.

        One forward pass gives J @ v for any vector v.
        """
        duals = [Dual(xi, vi) for xi, vi in zip(x, v)]
        result = f(duals)
        if isinstance(result, Dual):
            return (result.val, result.der)
        elif isinstance(result, (list, tuple)):
            vals = []
            ders = []
            for r in result:
                if isinstance(r, Dual):
                    vals.append(r.val)
                    ders.append(r.der)
                else:
                    vals.append(float(r))
                    ders.append(0.0)
            return (vals, ders)
        return (float(result), 0.0)

    @staticmethod
    def jacobian(f: Callable, x: List[float]) -> Tuple[List[float], List[List[float]]]:
        """Full Jacobian matrix for f: R^n -> R^m.

        Requires n forward passes (one per input dimension).
        """
        n = len(x)
        # First pass to determine output dimensionality
        e0 = [0.0] * n
        e0[0] = 1.0
        vals, col0 = ForwardAD.jvp(f, x, e0)

        if isinstance(vals, list):
            m = len(vals)
            jac = [[0.0] * n for _ in range(m)]
            for j in range(m):
                jac[j][0] = col0[j]
            for i in range(1, n):
                ei = [0.0] * n
                ei[i] = 1.0
                _, coli = ForwardAD.jvp(f, x, ei)
                for j in range(m):
                    jac[j][i] = coli[j]
            return (vals, jac)
        else:
            # Scalar output
            grad = [col0 if isinstance(col0, float) else col0]
            for i in range(1, n):
                ei = [0.0] * n
                ei[i] = 1.0
                _, di = ForwardAD.jvp(f, x, ei)
                grad.append(di)
            return (vals, [grad])

    @staticmethod
    def second_derivative(f: Callable, x: float, eps: float = 1e-5) -> Tuple[float, float, float]:
        """Compute f(x), f'(x), f''(x).

        Uses forward-mode for f' and symmetric finite differences on f' for f''.
        """
        v, fp = ForwardAD.derivative(f, x)
        _, fp_plus = ForwardAD.derivative(f, x + eps)
        _, fp_minus = ForwardAD.derivative(f, x - eps)
        fpp = (fp_plus - fp_minus) / (2 * eps)
        return (v, fp, fpp)


# ---------------------------------------------------------------------------
# Reverse Mode: Computation Graph
# ---------------------------------------------------------------------------

class Var:
    """Variable node in a computation graph for reverse-mode AD.

    Tracks the computation graph automatically. Call backward() on the
    output to compute gradients with respect to all inputs.
    """
    __slots__ = ('val', 'grad', '_children', '_backward', '_name')

    def __init__(self, val: float, _children=(), _backward=None, name=''):
        self.val = float(val)
        self.grad = 0.0
        self._children = _children
        self._backward = _backward
        self._name = name

    def __repr__(self):
        if self._name:
            return f"Var({self.val:.6f}, name='{self._name}')"
        return f"Var({self.val:.6f})"

    def __float__(self):
        return self.val

    def backward(self):
        """Compute gradients via reverse-mode AD (backpropagation)."""
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Reset gradients
        for v in topo:
            v.grad = 0.0

        # Backward pass
        self.grad = 1.0
        for v in reversed(topo):
            if v._backward is not None:
                v._backward()

    def zero_grad(self):
        """Zero out gradient."""
        self.grad = 0.0

    # Arithmetic operations
    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val + other.val, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val - other.val, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val * other.val, (self, other))

        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        out = Var(self.val / other.val, (self, other))

        def _backward():
            self.grad += out.grad / other.val
            other.grad -= out.grad * self.val / (other.val * other.val)
        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return other.__truediv__(self)

    def __pow__(self, other):
        if isinstance(other, Var):
            # f^g
            if self.val <= 0 and other.val != int(other.val):
                raise ValueError("Cannot raise non-positive to non-integer power")
            if self.val > 0:
                out = Var(self.val ** other.val, (self, other))
                def _backward():
                    self.grad += out.grad * other.val * (self.val ** (other.val - 1))
                    other.grad += out.grad * (self.val ** other.val) * math.log(self.val)
                out._backward = _backward
            else:
                n = int(other.val)
                out = Var(self.val ** n, (self, other))
                def _backward():
                    self.grad += out.grad * n * (self.val ** (n - 1))
                out._backward = _backward
            return out
        n = float(other)
        out = Var(self.val ** n, (self,))
        def _backward():
            if n == 0:
                return
            self.grad += out.grad * n * (self.val ** (n - 1))
        out._backward = _backward
        return out

    def __rpow__(self, other):
        # other^self
        other = other if isinstance(other, Var) else Var(other)
        return other.__pow__(self)

    def __neg__(self):
        out = Var(-self.val, (self,))
        def _backward():
            self.grad -= out.grad
        out._backward = _backward
        return out

    def __abs__(self):
        out = Var(abs(self.val), (self,))
        def _backward():
            if self.val > 0:
                self.grad += out.grad
            elif self.val < 0:
                self.grad -= out.grad
        out._backward = _backward
        return out

    # Comparison (by value only, no gradient)
    def __eq__(self, other):
        if isinstance(other, Var):
            return self.val == other.val
        return self.val == float(other)

    def __lt__(self, other):
        if isinstance(other, Var):
            return self.val < other.val
        return self.val < float(other)

    def __le__(self, other):
        if isinstance(other, Var):
            return self.val <= other.val
        return self.val <= float(other)

    def __gt__(self, other):
        if isinstance(other, Var):
            return self.val > other.val
        return self.val > float(other)

    def __ge__(self, other):
        if isinstance(other, Var):
            return self.val >= other.val
        return self.val >= float(other)

    def __hash__(self):
        return id(self)


# ---------------------------------------------------------------------------
# Var math functions (reverse mode)
# ---------------------------------------------------------------------------

def var_sin(x):
    if not isinstance(x, Var):
        return math.sin(x)
    out = Var(math.sin(x.val), (x,))
    def _backward():
        x.grad += out.grad * math.cos(x.val)
    out._backward = _backward
    return out

def var_cos(x):
    if not isinstance(x, Var):
        return math.cos(x)
    out = Var(math.cos(x.val), (x,))
    def _backward():
        x.grad -= out.grad * math.sin(x.val)
    out._backward = _backward
    return out

def var_tan(x):
    if not isinstance(x, Var):
        return math.tan(x)
    out = Var(math.tan(x.val), (x,))
    def _backward():
        c = math.cos(x.val)
        x.grad += out.grad / (c * c)
    out._backward = _backward
    return out

def var_exp(x):
    if not isinstance(x, Var):
        return math.exp(x)
    e = math.exp(x.val)
    out = Var(e, (x,))
    def _backward():
        x.grad += out.grad * e
    out._backward = _backward
    return out

def var_log(x):
    if not isinstance(x, Var):
        return math.log(x)
    out = Var(math.log(x.val), (x,))
    def _backward():
        x.grad += out.grad / x.val
    out._backward = _backward
    return out

def var_sqrt(x):
    if not isinstance(x, Var):
        return math.sqrt(x)
    s = math.sqrt(x.val)
    out = Var(s, (x,))
    def _backward():
        x.grad += out.grad / (2.0 * s)
    out._backward = _backward
    return out

def var_tanh(x):
    if not isinstance(x, Var):
        return math.tanh(x)
    t = math.tanh(x.val)
    out = Var(t, (x,))
    def _backward():
        x.grad += out.grad * (1 - t * t)
    out._backward = _backward
    return out

def var_sigmoid(x):
    if not isinstance(x, Var):
        return 1.0 / (1.0 + math.exp(-x))
    s = 1.0 / (1.0 + math.exp(-x.val))
    out = Var(s, (x,))
    def _backward():
        x.grad += out.grad * s * (1 - s)
    out._backward = _backward
    return out

def var_relu(x):
    if not isinstance(x, Var):
        return max(0.0, x)
    out = Var(max(0.0, x.val), (x,))
    def _backward():
        if x.val > 0:
            x.grad += out.grad
    out._backward = _backward
    return out

def var_max(a, b):
    a_is_var = isinstance(a, Var)
    b_is_var = isinstance(b, Var)
    a_val = a.val if a_is_var else float(a)
    b_val = b.val if b_is_var else float(b)

    if not a_is_var:
        a = Var(a_val)
    if not b_is_var:
        b = Var(b_val)

    if a_val >= b_val:
        out = Var(a_val, (a, b))
        def _backward():
            a.grad += out.grad
        out._backward = _backward
    else:
        out = Var(b_val, (a, b))
        def _backward():
            b.grad += out.grad
        out._backward = _backward
    return out

def var_min(a, b):
    a_is_var = isinstance(a, Var)
    b_is_var = isinstance(b, Var)
    a_val = a.val if a_is_var else float(a)
    b_val = b.val if b_is_var else float(b)

    if not a_is_var:
        a = Var(a_val)
    if not b_is_var:
        b = Var(b_val)

    if a_val <= b_val:
        out = Var(a_val, (a, b))
        def _backward():
            a.grad += out.grad
        out._backward = _backward
    else:
        out = Var(b_val, (a, b))
        def _backward():
            b.grad += out.grad
        out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Reverse Mode AD Engine
# ---------------------------------------------------------------------------

class ReverseAD:
    """Reverse-mode automatic differentiation using computation graphs.

    Best for: f: R^n -> R where n >> 1 (many inputs, scalar output).
    One backward pass gives the full gradient.
    """

    @staticmethod
    def gradient(f: Callable, x: List[float]) -> Tuple[float, List[float]]:
        """Compute f(x) and gradient for f: R^n -> R."""
        vars_list = [Var(xi, name=f'x{i}') for i, xi in enumerate(x)]
        result = f(vars_list)
        if not isinstance(result, Var):
            return (float(result), [0.0] * len(x))
        result.backward()
        return (result.val, [v.grad for v in vars_list])

    @staticmethod
    def vjp(f: Callable, x: List[float], v: Union[float, List[float]]) -> Tuple[Any, List[float]]:
        """Vector-Jacobian product: v^T @ J(x).

        One backward pass per output component weighted by v.
        For scalar output, v is a float (typically 1.0).
        """
        vars_list = [Var(xi) for xi in x]
        result = f(vars_list)

        if isinstance(result, Var):
            # Scalar output
            v_scalar = v if isinstance(v, (int, float)) else v[0]
            result.backward()
            return (result.val, [vi.grad * v_scalar for vi in vars_list])
        elif isinstance(result, (list, tuple)):
            # Vector output -- accumulate weighted gradients
            vals = []
            total_grads = [0.0] * len(x)

            if isinstance(v, (int, float)):
                v = [v] * len(result)

            for j, rj in enumerate(result):
                if isinstance(rj, Var):
                    vals.append(rj.val)
                else:
                    vals.append(float(rj))

            # Need to do separate backward passes per output component
            for j, rj in enumerate(result):
                if isinstance(rj, Var) and v[j] != 0:
                    # Re-evaluate to get fresh graph
                    fresh_vars = [Var(xi) for xi in x]
                    fresh_result = f(fresh_vars)
                    if isinstance(fresh_result, (list, tuple)):
                        fresh_rj = fresh_result[j]
                    else:
                        fresh_rj = fresh_result
                    if isinstance(fresh_rj, Var):
                        fresh_rj.backward()
                        for k, fv in enumerate(fresh_vars):
                            total_grads[k] += v[j] * fv.grad

            return (vals, total_grads)
        else:
            return (float(result), [0.0] * len(x))

    @staticmethod
    def jacobian(f: Callable, x: List[float]) -> Tuple[List[float], List[List[float]]]:
        """Full Jacobian for f: R^n -> R^m via reverse mode.

        Requires m backward passes (one per output dimension).
        """
        vars_list = [Var(xi) for xi in x]
        result = f(vars_list)
        n = len(x)

        if isinstance(result, Var):
            # Scalar output -> 1xn Jacobian
            result.backward()
            return (result.val, [[v.grad for v in vars_list]])
        elif isinstance(result, (list, tuple)):
            m = len(result)
            vals = [r.val if isinstance(r, Var) else float(r) for r in result]
            jac = [[0.0] * n for _ in range(m)]

            for j in range(m):
                fresh_vars = [Var(xi) for xi in x]
                fresh_result = f(fresh_vars)
                rj = fresh_result[j] if isinstance(fresh_result, (list, tuple)) else fresh_result
                if isinstance(rj, Var):
                    rj.backward()
                    for k in range(n):
                        jac[j][k] = fresh_vars[k].grad

            return (vals, jac)
        else:
            return (float(result), [[0.0] * n])


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def grad(f: Callable, mode: str = 'reverse') -> Callable:
    """Returns a function that computes the gradient of f.

    Args:
        f: Function taking a list of floats, returning a scalar.
        mode: 'forward' or 'reverse'.

    Returns:
        Function that takes x and returns gradient list.
    """
    if mode == 'reverse':
        def grad_fn(x):
            _, g = ReverseAD.gradient(f, x)
            return g
        return grad_fn
    else:
        def grad_fn(x):
            _, g = ForwardAD.gradient(f, x)
            return g
        return grad_fn


def value_and_grad(f: Callable, mode: str = 'reverse') -> Callable:
    """Returns a function that computes both value and gradient.

    Returns:
        Function that takes x and returns (value, gradient).
    """
    if mode == 'reverse':
        return lambda x: ReverseAD.gradient(f, x)
    else:
        return lambda x: ForwardAD.gradient(f, x)


def jacobian(f: Callable, x: List[float], mode: str = 'forward') -> Tuple[Any, List[List[float]]]:
    """Compute the Jacobian matrix of f at x.

    Args:
        f: Function R^n -> R^m.
        x: Point at which to evaluate.
        mode: 'forward' (default, better for n < m) or 'reverse' (better for m < n).
    """
    if mode == 'forward':
        return ForwardAD.jacobian(f, x)
    else:
        return ReverseAD.jacobian(f, x)


def hessian(f: Callable, x: List[float], eps: float = 1e-5) -> Tuple[float, List[List[float]]]:
    """Compute the Hessian matrix of f: R^n -> R at point x.

    Uses finite differences on the reverse-mode gradient for reliability.
    H[i][j] = (g_i(x + eps*e_j) - g_i(x - eps*e_j)) / (2*eps)
    """
    n = len(x)
    val, g0 = ReverseAD.gradient(f, x)

    H = [[0.0] * n for _ in range(n)]

    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += eps
        x_minus[j] -= eps
        _, g_plus = ReverseAD.gradient(f, x_plus)
        _, g_minus = ReverseAD.gradient(f, x_minus)
        for i in range(n):
            H[i][j] = (g_plus[i] - g_minus[i]) / (2 * eps)

    return (val, H)


def directional_derivative(f: Callable, x: List[float], v: List[float]) -> Tuple[float, float]:
    """Compute directional derivative of f at x in direction v.

    Returns (f(x), nabla_f(x) . v).
    """
    return ForwardAD.jvp(f, x, v)


# ---------------------------------------------------------------------------
# Gradient Checking
# ---------------------------------------------------------------------------

def check_grad(f: Callable, x: List[float], eps: float = 1e-7) -> Tuple[List[float], List[float], float]:
    """Compare AD gradient to finite differences for verification.

    Returns:
        (ad_grad, fd_grad, max_relative_error)
    """
    _, ad_grad = ReverseAD.gradient(f, x)

    fd_grad = []
    for i in range(len(x)):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += eps
        x_minus[i] -= eps
        # Evaluate with plain floats
        fp = f(x_plus)
        fm = f(x_minus)
        if isinstance(fp, Var):
            fp = fp.val
        if isinstance(fm, Var):
            fm = fm.val
        fd_grad.append((fp - fm) / (2 * eps))

    max_err = 0.0
    for ag, fg in zip(ad_grad, fd_grad):
        denom = max(abs(ag), abs(fg), 1e-10)
        err = abs(ag - fg) / denom
        if err > max_err:
            max_err = err

    return (ad_grad, fd_grad, max_err)


# ---------------------------------------------------------------------------
# Checkpoint (Memory-Efficient Reverse Mode)
# ---------------------------------------------------------------------------

def checkpoint(f: Callable, *args):
    """Memory-efficient reverse mode via recomputation.

    Instead of storing all intermediate values, stores only the inputs
    and recomputes forward pass during backward.
    """
    # Save input values
    saved_vals = []
    for a in args:
        if isinstance(a, Var):
            saved_vals.append(a.val)
        else:
            saved_vals.append(float(a))

    # Forward pass (discard intermediate graph)
    result_val = f(*[Var(v) for v in saved_vals])
    if isinstance(result_val, Var):
        result_val = result_val.val

    # Create output node connected to inputs
    input_vars = [a for a in args if isinstance(a, Var)]
    out = Var(result_val, tuple(input_vars))

    def _backward():
        # Recompute forward pass to get graph
        fresh = [Var(v) for v in saved_vals]
        fresh_result = f(*fresh)
        if isinstance(fresh_result, Var):
            fresh_result.grad = out.grad
            # Build topo order and backprop
            topo = []
            visited = set()
            def build_topo(v):
                if id(v) not in visited:
                    visited.add(id(v))
                    for c in v._children:
                        build_topo(c)
                    topo.append(v)
            build_topo(fresh_result)
            for v in topo:
                if v is not fresh_result:
                    v.grad = 0.0
            for v in reversed(topo):
                if v._backward is not None:
                    v._backward()
            # Transfer gradients to original input vars
            for orig_var, fresh_var in zip(input_vars, fresh):
                orig_var.grad += fresh_var.grad

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# AD-Powered Optimizer
# ---------------------------------------------------------------------------

class ADOptimizerStatus(Enum):
    OPTIMAL = "optimal"
    MAX_ITER = "max_iterations"
    DIVERGED = "diverged"


@dataclass
class ADOptResult:
    status: ADOptimizerStatus
    x: List[float]
    objective: float
    gradient_norm: float
    iterations: int
    history: List[float] = field(default_factory=list)


class ADOptimizer:
    """Gradient-based optimization using automatic differentiation.

    No need to supply gradient functions -- they're computed automatically.
    Composes C127 optimization concepts with AD.
    """

    @staticmethod
    def gradient_descent(
        f: Callable,
        x0: List[float],
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
        mode: str = 'reverse'
    ) -> ADOptResult:
        """Gradient descent with automatic gradients."""
        x = list(x0)
        history = []
        vg = value_and_grad(f, mode)

        for it in range(max_iter):
            val, g = vg(x)
            history.append(val)
            gnorm = math.sqrt(sum(gi * gi for gi in g))

            if gnorm < tol:
                return ADOptResult(
                    status=ADOptimizerStatus.OPTIMAL,
                    x=x, objective=val, gradient_norm=gnorm,
                    iterations=it + 1, history=history
                )

            if math.isinf(val) or math.isnan(val):
                return ADOptResult(
                    status=ADOptimizerStatus.DIVERGED,
                    x=x, objective=val, gradient_norm=gnorm,
                    iterations=it + 1, history=history
                )

            x = [xi - lr * gi for xi, gi in zip(x, g)]

        val, g = vg(x)
        gnorm = math.sqrt(sum(gi * gi for gi in g))
        history.append(val)
        return ADOptResult(
            status=ADOptimizerStatus.MAX_ITER,
            x=x, objective=val, gradient_norm=gnorm,
            iterations=max_iter, history=history
        )

    @staticmethod
    def adam(
        f: Callable,
        x0: List[float],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_iter: int = 1000,
        tol: float = 1e-8,
        mode: str = 'reverse'
    ) -> ADOptResult:
        """Adam optimizer with automatic gradients."""
        x = list(x0)
        n = len(x)
        m = [0.0] * n  # first moment
        v = [0.0] * n  # second moment
        history = []
        vg = value_and_grad(f, mode)

        for it in range(1, max_iter + 1):
            val, g = vg(x)
            history.append(val)
            gnorm = math.sqrt(sum(gi * gi for gi in g))

            if gnorm < tol:
                return ADOptResult(
                    status=ADOptimizerStatus.OPTIMAL,
                    x=x, objective=val, gradient_norm=gnorm,
                    iterations=it, history=history
                )

            if math.isinf(val) or math.isnan(val):
                return ADOptResult(
                    status=ADOptimizerStatus.DIVERGED,
                    x=x, objective=val, gradient_norm=gnorm,
                    iterations=it, history=history
                )

            for i in range(n):
                m[i] = beta1 * m[i] + (1 - beta1) * g[i]
                v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i]

            # Bias correction
            m_hat = [mi / (1 - beta1 ** it) for mi in m]
            v_hat = [vi / (1 - beta2 ** it) for vi in v]

            x = [xi - lr * mhi / (math.sqrt(vhi) + eps)
                 for xi, mhi, vhi in zip(x, m_hat, v_hat)]

        val, g = vg(x)
        gnorm = math.sqrt(sum(gi * gi for gi in g))
        history.append(val)
        return ADOptResult(
            status=ADOptimizerStatus.MAX_ITER,
            x=x, objective=val, gradient_norm=gnorm,
            iterations=max_iter, history=history
        )

    @staticmethod
    def lbfgs(
        f: Callable,
        x0: List[float],
        max_iter: int = 100,
        m: int = 10,
        tol: float = 1e-8,
        c1: float = 1e-4,
        c2: float = 0.9,
        mode: str = 'reverse'
    ) -> ADOptResult:
        """L-BFGS optimizer with automatic gradients.

        Limited-memory BFGS uses the last m gradient differences to
        approximate the inverse Hessian direction.
        """
        n = len(x0)
        x = list(x0)
        vg = value_and_grad(f, mode)

        val, g = vg(x)
        history = [val]

        s_history = []  # x_{k+1} - x_k
        y_history = []  # g_{k+1} - g_k
        rho_history = []  # 1 / (y^T s)

        for it in range(max_iter):
            gnorm = math.sqrt(sum(gi * gi for gi in g))
            if gnorm < tol:
                return ADOptResult(
                    status=ADOptimizerStatus.OPTIMAL,
                    x=x, objective=val, gradient_norm=gnorm,
                    iterations=it + 1, history=history
                )

            # L-BFGS two-loop recursion
            q = list(g)
            alphas = [0.0] * len(s_history)

            for i in range(len(s_history) - 1, -1, -1):
                alphas[i] = rho_history[i] * sum(
                    si * qi for si, qi in zip(s_history[i], q)
                )
                q = [qi - alphas[i] * yi for qi, yi in zip(q, y_history[i])]

            # Initial Hessian approximation
            if s_history:
                ys = sum(yi * si for yi, si in zip(y_history[-1], s_history[-1]))
                yy = sum(yi * yi for yi in y_history[-1])
                gamma = ys / yy if yy > 0 else 1.0
            else:
                gamma = 1.0 / max(gnorm, 1.0)

            r = [gamma * qi for qi in q]

            for i in range(len(s_history)):
                beta = rho_history[i] * sum(
                    yi * ri for yi, ri in zip(y_history[i], r)
                )
                r = [ri + (alphas[i] - beta) * si
                     for ri, si in zip(r, s_history[i])]

            # Direction
            d = [-ri for ri in r]

            # Backtracking line search
            alpha = 1.0
            dg = sum(di * gi for di, gi in zip(d, g))

            for _ in range(30):
                x_new = [xi + alpha * di for xi, di in zip(x, d)]
                val_new, g_new = vg(x_new)
                if val_new <= val + c1 * alpha * dg:
                    break
                alpha *= 0.5
            else:
                x_new = [xi + alpha * di for xi, di in zip(x, d)]
                val_new, g_new = vg(x_new)

            # Store curvature pair
            s = [xn - xi for xn, xi in zip(x_new, x)]
            y = [gn - gi for gn, gi in zip(g_new, g)]
            ys_val = sum(si * yi for si, yi in zip(s, y))

            if ys_val > 1e-10:
                if len(s_history) >= m:
                    s_history.pop(0)
                    y_history.pop(0)
                    rho_history.pop(0)
                s_history.append(s)
                y_history.append(y)
                rho_history.append(1.0 / ys_val)

            x = x_new
            val = val_new
            g = g_new
            history.append(val)

        gnorm = math.sqrt(sum(gi * gi for gi in g))
        return ADOptResult(
            status=ADOptimizerStatus.MAX_ITER,
            x=x, objective=val, gradient_norm=gnorm,
            iterations=max_iter, history=history
        )


# ---------------------------------------------------------------------------
# Neural Network Building Blocks (demonstrating AD composability)
# ---------------------------------------------------------------------------

class NeuralOps:
    """Neural network operations built on top of Var (reverse-mode AD).

    Demonstrates that the AD system can support ML-style computation.
    """

    @staticmethod
    def linear(x: List[Var], weights: List[List[Var]], bias: List[Var]) -> List[Var]:
        """Linear layer: y = Wx + b."""
        out = []
        for i in range(len(bias)):
            s = bias[i]
            for j in range(len(x)):
                s = s + weights[i][j] * x[j]
            out.append(s)
        return out

    @staticmethod
    def softmax(x: List[Var]) -> List[Var]:
        """Softmax activation."""
        max_val = max(xi.val for xi in x)
        exps = [var_exp(xi - max_val) for xi in x]
        total = exps[0]
        for e in exps[1:]:
            total = total + e
        return [e / total for e in exps]

    @staticmethod
    def cross_entropy(predicted: List[Var], target: List[float]) -> Var:
        """Cross-entropy loss."""
        loss = Var(0.0)
        for p, t in zip(predicted, target):
            if t > 0:
                loss = loss - t * var_log(p + 1e-10)
        return loss

    @staticmethod
    def mse_loss(predicted: List[Var], target: List[float]) -> Var:
        """Mean squared error loss."""
        n = len(predicted)
        loss = Var(0.0)
        for p, t in zip(predicted, target):
            diff = p - t
            loss = loss + diff * diff
        return loss / n

    @staticmethod
    def batch_norm(x: List[Var], eps: float = 1e-5) -> List[Var]:
        """Batch normalization (inference mode -- fixed mean/var)."""
        n = len(x)
        mean = x[0]
        for xi in x[1:]:
            mean = mean + xi
        mean = mean / n

        var = Var(0.0)
        for xi in x:
            diff = xi - mean
            var = var + diff * diff
        var = var / n

        return [(xi - mean) / var_sqrt(var + eps) for xi in x]


# ---------------------------------------------------------------------------
# Utility: Numerical Jacobian (for testing)
# ---------------------------------------------------------------------------

def numerical_jacobian(f: Callable, x: List[float], eps: float = 1e-7) -> List[List[float]]:
    """Compute Jacobian via finite differences (for testing)."""
    n = len(x)
    # Determine output size
    y0 = f(list(x))
    if isinstance(y0, (list, tuple)):
        m = len(y0)
        y0_vals = [float(yi) if not isinstance(yi, (Var, Dual)) else yi.val for yi in y0]
    else:
        m = 1
        y0_vals = [float(y0) if not isinstance(y0, (Var, Dual)) else y0.val]

    jac = [[0.0] * n for _ in range(m)]

    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += eps
        x_minus[j] -= eps

        yp = f(x_plus)
        ym = f(x_minus)

        if isinstance(yp, (list, tuple)):
            for i in range(m):
                vp = float(yp[i]) if not isinstance(yp[i], (Var, Dual)) else yp[i].val
                vm = float(ym[i]) if not isinstance(ym[i], (Var, Dual)) else ym[i].val
                jac[i][j] = (vp - vm) / (2 * eps)
        else:
            vp = float(yp) if not isinstance(yp, (Var, Dual)) else yp.val
            vm = float(ym) if not isinstance(ym, (Var, Dual)) else ym.val
            jac[0][j] = (vp - vm) / (2 * eps)

    return jac
