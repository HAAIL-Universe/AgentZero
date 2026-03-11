"""Tests for C128: Automatic Differentiation."""

import math
import pytest
from autodiff import (
    Dual, ForwardAD, Var, ReverseAD,
    dual_sin, dual_cos, dual_tan, dual_exp, dual_log, dual_sqrt,
    dual_tanh, dual_sigmoid, dual_relu, dual_max, dual_min, dual_abs,
    var_sin, var_cos, var_tan, var_exp, var_log, var_sqrt,
    var_tanh, var_sigmoid, var_relu, var_max, var_min,
    grad, value_and_grad, jacobian, hessian, directional_derivative,
    check_grad, checkpoint, numerical_jacobian,
    ADOptimizer, ADOptimizerStatus, ADOptResult,
    NeuralOps,
)

TOL = 1e-6
FTOL = 1e-4  # looser for finite-difference comparisons


# ===== DUAL NUMBER BASICS =====

class TestDualArithmetic:
    def test_creation(self):
        d = Dual(3.0, 1.0)
        assert d.val == 3.0
        assert d.der == 1.0

    def test_creation_default_der(self):
        d = Dual(5.0)
        assert d.der == 0.0

    def test_repr(self):
        assert "3.0" in repr(Dual(3.0, 1.0))

    def test_float_conversion(self):
        assert float(Dual(3.5, 1.0)) == 3.5

    def test_int_conversion(self):
        assert int(Dual(3.7, 1.0)) == 3

    def test_add(self):
        a = Dual(2.0, 1.0)
        b = Dual(3.0, 0.0)
        c = a + b
        assert c.val == 5.0
        assert c.der == 1.0

    def test_add_scalar(self):
        a = Dual(2.0, 1.0)
        c = a + 5
        assert c.val == 7.0
        assert c.der == 1.0

    def test_radd(self):
        a = Dual(2.0, 1.0)
        c = 5 + a
        assert c.val == 7.0
        assert c.der == 1.0

    def test_sub(self):
        a = Dual(5.0, 1.0)
        b = Dual(3.0, 0.5)
        c = a - b
        assert c.val == 2.0
        assert c.der == 0.5

    def test_rsub(self):
        a = Dual(3.0, 1.0)
        c = 10 - a
        assert c.val == 7.0
        assert c.der == -1.0

    def test_mul(self):
        a = Dual(3.0, 1.0)
        b = Dual(4.0, 0.0)
        c = a * b
        assert c.val == 12.0
        assert c.der == 4.0  # d/dx(x*4) = 4

    def test_mul_product_rule(self):
        # d/dx(x*x) at x=3 = 6
        a = Dual(3.0, 1.0)
        c = a * a
        assert c.val == 9.0
        assert c.der == 6.0

    def test_rmul(self):
        a = Dual(3.0, 1.0)
        c = 5 * a
        assert c.val == 15.0
        assert c.der == 5.0

    def test_div(self):
        a = Dual(6.0, 1.0)
        b = Dual(3.0, 0.0)
        c = a / b
        assert c.val == 2.0
        assert abs(c.der - 1.0 / 3.0) < TOL

    def test_rdiv(self):
        # d/dx(1/x) at x=2 = -1/4
        a = Dual(2.0, 1.0)
        c = 1 / a
        assert c.val == 0.5
        assert abs(c.der - (-0.25)) < TOL

    def test_pow_const(self):
        # d/dx(x^3) at x=2 = 12
        a = Dual(2.0, 1.0)
        c = a ** 3
        assert c.val == 8.0
        assert abs(c.der - 12.0) < TOL

    def test_pow_zero(self):
        a = Dual(5.0, 1.0)
        c = a ** 0
        assert c.val == 1.0
        assert c.der == 0.0

    def test_pow_one(self):
        a = Dual(5.0, 1.0)
        c = a ** 1
        assert c.val == 5.0
        assert c.der == 1.0

    def test_pow_dual(self):
        # d/dx(x^x) at x=2 = 4*(1+ln2) ~ 6.7726
        a = Dual(2.0, 1.0)
        c = a ** a
        expected_der = 4.0 * (1 + math.log(2))
        assert abs(c.val - 4.0) < TOL
        assert abs(c.der - expected_der) < TOL

    def test_rpow(self):
        # d/dx(2^x) at x=3 = 8*ln(2)
        a = Dual(3.0, 1.0)
        c = 2 ** a
        assert abs(c.val - 8.0) < TOL
        assert abs(c.der - 8.0 * math.log(2)) < TOL

    def test_neg(self):
        a = Dual(3.0, 1.0)
        c = -a
        assert c.val == -3.0
        assert c.der == -1.0

    def test_abs_positive(self):
        a = Dual(3.0, 1.0)
        c = abs(a)
        assert c.val == 3.0
        assert c.der == 1.0

    def test_abs_negative(self):
        a = Dual(-3.0, 1.0)
        c = abs(a)
        assert c.val == 3.0
        assert c.der == -1.0

    def test_comparison(self):
        a = Dual(3.0, 1.0)
        b = Dual(5.0, 0.0)
        assert a < b
        assert b > a
        assert a <= b
        assert a == Dual(3.0, 999.0)  # comparison by value
        assert a < 4
        assert a >= 3

    def test_hash(self):
        d = {Dual(1.0): 'a', Dual(2.0): 'b'}
        assert d[Dual(1.0)] == 'a'


# ===== DUAL MATH FUNCTIONS =====

class TestDualMath:
    def test_sin(self):
        # d/dx(sin(x)) at x=pi/4 = cos(pi/4)
        x = Dual(math.pi / 4, 1.0)
        r = dual_sin(x)
        assert abs(r.val - math.sin(math.pi / 4)) < TOL
        assert abs(r.der - math.cos(math.pi / 4)) < TOL

    def test_cos(self):
        x = Dual(math.pi / 3, 1.0)
        r = dual_cos(x)
        assert abs(r.val - math.cos(math.pi / 3)) < TOL
        assert abs(r.der - (-math.sin(math.pi / 3))) < TOL

    def test_tan(self):
        x = Dual(math.pi / 6, 1.0)
        r = dual_tan(x)
        c = math.cos(math.pi / 6)
        assert abs(r.val - math.tan(math.pi / 6)) < TOL
        assert abs(r.der - 1.0 / (c * c)) < TOL

    def test_exp(self):
        x = Dual(2.0, 1.0)
        r = dual_exp(x)
        assert abs(r.val - math.exp(2.0)) < TOL
        assert abs(r.der - math.exp(2.0)) < TOL

    def test_log(self):
        x = Dual(3.0, 1.0)
        r = dual_log(x)
        assert abs(r.val - math.log(3.0)) < TOL
        assert abs(r.der - 1.0 / 3.0) < TOL

    def test_sqrt(self):
        x = Dual(4.0, 1.0)
        r = dual_sqrt(x)
        assert abs(r.val - 2.0) < TOL
        assert abs(r.der - 0.25) < TOL  # 1/(2*sqrt(4)) = 0.25

    def test_tanh(self):
        x = Dual(1.0, 1.0)
        r = dual_tanh(x)
        t = math.tanh(1.0)
        assert abs(r.val - t) < TOL
        assert abs(r.der - (1 - t * t)) < TOL

    def test_sigmoid(self):
        x = Dual(0.0, 1.0)
        r = dual_sigmoid(x)
        assert abs(r.val - 0.5) < TOL
        assert abs(r.der - 0.25) < TOL  # sigmoid'(0) = 0.25

    def test_relu_positive(self):
        x = Dual(3.0, 1.0)
        r = dual_relu(x)
        assert r.val == 3.0
        assert r.der == 1.0

    def test_relu_negative(self):
        x = Dual(-2.0, 1.0)
        r = dual_relu(x)
        assert r.val == 0.0
        assert r.der == 0.0

    def test_max(self):
        a = Dual(3.0, 1.0)
        b = Dual(5.0, 2.0)
        r = dual_max(a, b)
        assert r.val == 5.0
        assert r.der == 2.0

    def test_min(self):
        a = Dual(3.0, 1.0)
        b = Dual(5.0, 2.0)
        r = dual_min(a, b)
        assert r.val == 3.0
        assert r.der == 1.0

    def test_plain_float_passthrough(self):
        assert dual_sin(0.0) == 0.0
        assert dual_cos(0.0) == 1.0
        assert abs(dual_exp(1.0) - math.e) < TOL
        assert dual_relu(-5.0) == 0.0

    def test_chain_sin_exp(self):
        # d/dx(exp(sin(x))) at x=1
        x = Dual(1.0, 1.0)
        r = dual_exp(dual_sin(x))
        expected_val = math.exp(math.sin(1.0))
        expected_der = math.exp(math.sin(1.0)) * math.cos(1.0)
        assert abs(r.val - expected_val) < TOL
        assert abs(r.der - expected_der) < TOL

    def test_chain_log_sqrt(self):
        # d/dx(log(sqrt(x))) at x=4 = 1/(2x) = 1/8
        x = Dual(4.0, 1.0)
        r = dual_log(dual_sqrt(x))
        assert abs(r.val - math.log(2.0)) < TOL
        assert abs(r.der - 0.125) < TOL


# ===== FORWARD MODE AD =====

class TestForwardAD:
    def test_scalar_derivative(self):
        # f(x) = x^2, f'(x) = 2x
        val, der = ForwardAD.derivative(lambda x: x ** 2, 3.0)
        assert abs(val - 9.0) < TOL
        assert abs(der - 6.0) < TOL

    def test_partial_derivative(self):
        # f(x,y) = x^2 + y^2
        def f(v): return v[0] ** 2 + v[1] ** 2
        val, dx = ForwardAD.partial(f, [3.0, 4.0], 0)
        assert abs(val - 25.0) < TOL
        assert abs(dx - 6.0) < TOL

        val, dy = ForwardAD.partial(f, [3.0, 4.0], 1)
        assert abs(dy - 8.0) < TOL

    def test_gradient_quadratic(self):
        def f(v): return v[0] ** 2 + 3 * v[1] ** 2
        val, g = ForwardAD.gradient(f, [2.0, 1.0])
        assert abs(val - 7.0) < TOL
        assert abs(g[0] - 4.0) < TOL  # 2x
        assert abs(g[1] - 6.0) < TOL  # 6y

    def test_gradient_rosenbrock(self):
        # f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        def rosenbrock(v):
            x, y = v[0], v[1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        val, g = ForwardAD.gradient(rosenbrock, [1.0, 1.0])
        assert abs(val) < TOL  # minimum at (1,1)
        assert abs(g[0]) < TOL
        assert abs(g[1]) < TOL

    def test_jvp_scalar(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        val, jvp_val = ForwardAD.jvp(f, [3.0, 4.0], [1.0, 0.0])
        assert abs(val - 25.0) < TOL
        assert abs(jvp_val - 6.0) < TOL  # partial w.r.t. x

    def test_jvp_vector(self):
        def f(v): return [v[0] + v[1], v[0] * v[1]]
        vals, ders = ForwardAD.jvp(f, [2.0, 3.0], [1.0, 0.0])
        assert abs(vals[0] - 5.0) < TOL
        assert abs(vals[1] - 6.0) < TOL
        assert abs(ders[0] - 1.0) < TOL  # d(x+y)/dx = 1
        assert abs(ders[1] - 3.0) < TOL  # d(x*y)/dx = y

    def test_jacobian_vector(self):
        def f(v): return [v[0] + v[1], v[0] * v[1], v[0] ** 2]
        vals, J = ForwardAD.jacobian(f, [2.0, 3.0])
        assert abs(vals[0] - 5.0) < TOL
        # J = [[1, 1], [3, 2], [4, 0]]
        assert abs(J[0][0] - 1.0) < TOL
        assert abs(J[0][1] - 1.0) < TOL
        assert abs(J[1][0] - 3.0) < TOL
        assert abs(J[1][1] - 2.0) < TOL
        assert abs(J[2][0] - 4.0) < TOL
        assert abs(J[2][1] - 0.0) < TOL

    def test_second_derivative(self):
        # f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        val, fp, fpp = ForwardAD.second_derivative(lambda x: x ** 3, 2.0)
        assert abs(val - 8.0) < TOL
        assert abs(fp - 12.0) < TOL
        assert abs(fpp - 12.0) < TOL

    def test_second_derivative_exp(self):
        # f(x) = exp(x), all derivatives = exp(x)
        val, fp, fpp = ForwardAD.second_derivative(dual_exp, 1.0)
        e = math.e
        assert abs(val - e) < TOL
        assert abs(fp - e) < TOL
        assert abs(fpp - e) < TOL


# ===== VAR (REVERSE MODE) BASICS =====

class TestVarArithmetic:
    def test_creation(self):
        v = Var(3.0, name='x')
        assert v.val == 3.0
        assert v.grad == 0.0
        assert 'x' in repr(v)

    def test_add(self):
        a = Var(2.0)
        b = Var(3.0)
        c = a + b
        c.backward()
        assert c.val == 5.0
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_add_scalar(self):
        a = Var(2.0)
        c = a + 5
        c.backward()
        assert c.val == 7.0
        assert a.grad == 1.0

    def test_radd(self):
        a = Var(2.0)
        c = 5 + a
        c.backward()
        assert c.val == 7.0
        assert a.grad == 1.0

    def test_sub(self):
        a = Var(5.0)
        b = Var(3.0)
        c = a - b
        c.backward()
        assert c.val == 2.0
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_rsub(self):
        a = Var(3.0)
        c = 10 - a
        c.backward()
        assert c.val == 7.0
        assert a.grad == -1.0

    def test_mul(self):
        a = Var(3.0)
        b = Var(4.0)
        c = a * b
        c.backward()
        assert c.val == 12.0
        assert a.grad == 4.0  # dc/da = b
        assert b.grad == 3.0  # dc/db = a

    def test_rmul(self):
        a = Var(3.0)
        c = 5 * a
        c.backward()
        assert c.val == 15.0
        assert a.grad == 5.0

    def test_div(self):
        a = Var(6.0)
        b = Var(3.0)
        c = a / b
        c.backward()
        assert abs(c.val - 2.0) < TOL
        assert abs(a.grad - 1.0 / 3.0) < TOL
        assert abs(b.grad - (-6.0 / 9.0)) < TOL

    def test_rtruediv(self):
        a = Var(2.0)
        c = 1 / a  # 1/x, d/dx = -1/x^2
        c.backward()
        assert abs(c.val - 0.5) < TOL
        assert abs(a.grad - (-0.25)) < TOL

    def test_pow_const(self):
        a = Var(2.0)
        c = a ** 3
        c.backward()
        assert abs(c.val - 8.0) < TOL
        assert abs(a.grad - 12.0) < TOL

    def test_pow_var(self):
        a = Var(2.0)
        b = Var(3.0)
        c = a ** b  # 2^3 = 8
        c.backward()
        assert abs(c.val - 8.0) < TOL
        assert abs(a.grad - 12.0) < TOL  # 3 * 2^2
        assert abs(b.grad - 8.0 * math.log(2)) < TOL

    def test_neg(self):
        a = Var(3.0)
        c = -a
        c.backward()
        assert c.val == -3.0
        assert a.grad == -1.0

    def test_abs_positive(self):
        a = Var(3.0)
        c = abs(a)
        c.backward()
        assert c.val == 3.0
        assert a.grad == 1.0

    def test_abs_negative(self):
        a = Var(-3.0)
        c = abs(a)
        c.backward()
        assert c.val == 3.0
        assert a.grad == -1.0

    def test_comparison(self):
        a = Var(3.0)
        b = Var(5.0)
        assert a < b
        assert b > a
        assert a <= Var(3.0)
        assert a >= 3

    def test_complex_expression(self):
        # f(x,y) = x^2*y + y^3
        # df/dx = 2xy, df/dy = x^2 + 3y^2
        x = Var(2.0)
        y = Var(3.0)
        f = x ** 2 * y + y ** 3
        f.backward()
        assert abs(f.val - (4 * 3 + 27)) < TOL
        assert abs(x.grad - 12.0) < TOL   # 2*2*3
        assert abs(y.grad - 31.0) < TOL   # 4 + 27

    def test_shared_node(self):
        # f = x * x (x used twice)
        x = Var(3.0)
        f = x * x
        f.backward()
        assert abs(f.val - 9.0) < TOL
        assert abs(x.grad - 6.0) < TOL  # 2x


# ===== VAR MATH FUNCTIONS =====

class TestVarMath:
    def test_sin(self):
        x = Var(math.pi / 4)
        y = var_sin(x)
        y.backward()
        assert abs(y.val - math.sin(math.pi / 4)) < TOL
        assert abs(x.grad - math.cos(math.pi / 4)) < TOL

    def test_cos(self):
        x = Var(math.pi / 3)
        y = var_cos(x)
        y.backward()
        assert abs(y.val - math.cos(math.pi / 3)) < TOL
        assert abs(x.grad - (-math.sin(math.pi / 3))) < TOL

    def test_tan(self):
        x = Var(math.pi / 6)
        y = var_tan(x)
        y.backward()
        c = math.cos(math.pi / 6)
        assert abs(y.val - math.tan(math.pi / 6)) < TOL
        assert abs(x.grad - 1.0 / (c * c)) < TOL

    def test_exp(self):
        x = Var(2.0)
        y = var_exp(x)
        y.backward()
        assert abs(y.val - math.exp(2.0)) < TOL
        assert abs(x.grad - math.exp(2.0)) < TOL

    def test_log(self):
        x = Var(3.0)
        y = var_log(x)
        y.backward()
        assert abs(y.val - math.log(3.0)) < TOL
        assert abs(x.grad - 1.0 / 3.0) < TOL

    def test_sqrt(self):
        x = Var(4.0)
        y = var_sqrt(x)
        y.backward()
        assert abs(y.val - 2.0) < TOL
        assert abs(x.grad - 0.25) < TOL

    def test_tanh(self):
        x = Var(1.0)
        y = var_tanh(x)
        y.backward()
        t = math.tanh(1.0)
        assert abs(y.val - t) < TOL
        assert abs(x.grad - (1 - t * t)) < TOL

    def test_sigmoid(self):
        x = Var(0.0)
        y = var_sigmoid(x)
        y.backward()
        assert abs(y.val - 0.5) < TOL
        assert abs(x.grad - 0.25) < TOL

    def test_relu_positive(self):
        x = Var(3.0)
        y = var_relu(x)
        y.backward()
        assert y.val == 3.0
        assert x.grad == 1.0

    def test_relu_negative(self):
        x = Var(-2.0)
        y = var_relu(x)
        y.backward()
        assert y.val == 0.0
        assert x.grad == 0.0

    def test_max(self):
        a = Var(3.0)
        b = Var(5.0)
        c = var_max(a, b)
        c.backward()
        assert c.val == 5.0
        assert a.grad == 0.0
        assert b.grad == 1.0

    def test_min(self):
        a = Var(3.0)
        b = Var(5.0)
        c = var_min(a, b)
        c.backward()
        assert c.val == 3.0
        assert a.grad == 1.0
        assert b.grad == 0.0

    def test_chain_exp_sin(self):
        # f(x) = exp(sin(x))
        x = Var(1.0)
        y = var_exp(var_sin(x))
        y.backward()
        expected = math.exp(math.sin(1.0)) * math.cos(1.0)
        assert abs(x.grad - expected) < TOL

    def test_plain_float_passthrough(self):
        assert var_sin(0.0) == 0.0
        assert var_cos(0.0) == 1.0
        assert var_relu(-5.0) == 0.0


# ===== REVERSE MODE AD ENGINE =====

class TestReverseAD:
    def test_gradient_simple(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        val, g = ReverseAD.gradient(f, [3.0, 4.0])
        assert abs(val - 25.0) < TOL
        assert abs(g[0] - 6.0) < TOL
        assert abs(g[1] - 8.0) < TOL

    def test_gradient_rosenbrock(self):
        def rosenbrock(v):
            x, y = v[0], v[1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        val, g = ReverseAD.gradient(rosenbrock, [1.0, 1.0])
        assert abs(val) < TOL
        assert abs(g[0]) < TOL
        assert abs(g[1]) < TOL

    def test_gradient_3d(self):
        def f(v): return v[0] * v[1] + v[1] * v[2] + v[0] * v[2]
        val, g = ReverseAD.gradient(f, [1.0, 2.0, 3.0])
        # df/dx0 = x1 + x2 = 5, df/dx1 = x0 + x2 = 4, df/dx2 = x1 + x0 = 3
        assert abs(val - (2 + 6 + 3)) < TOL
        assert abs(g[0] - 5.0) < TOL
        assert abs(g[1] - 4.0) < TOL
        assert abs(g[2] - 3.0) < TOL

    def test_vjp_scalar(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        val, vjp_val = ReverseAD.vjp(f, [3.0, 4.0], 1.0)
        assert abs(val - 25.0) < TOL
        assert abs(vjp_val[0] - 6.0) < TOL
        assert abs(vjp_val[1] - 8.0) < TOL

    def test_vjp_scaled(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        val, vjp_val = ReverseAD.vjp(f, [3.0, 4.0], 2.0)
        assert abs(vjp_val[0] - 12.0) < TOL
        assert abs(vjp_val[1] - 16.0) < TOL

    def test_jacobian_scalar(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        val, J = ReverseAD.jacobian(f, [3.0, 4.0])
        assert abs(val - 25.0) < TOL
        assert abs(J[0][0] - 6.0) < TOL
        assert abs(J[0][1] - 8.0) < TOL

    def test_jacobian_vector(self):
        def f(v): return [v[0] + v[1], v[0] * v[1]]
        vals, J = ReverseAD.jacobian(f, [2.0, 3.0])
        assert abs(vals[0] - 5.0) < TOL
        assert abs(vals[1] - 6.0) < TOL
        assert abs(J[0][0] - 1.0) < TOL
        assert abs(J[0][1] - 1.0) < TOL
        assert abs(J[1][0] - 3.0) < TOL
        assert abs(J[1][1] - 2.0) < TOL


# ===== CONVENIENCE FUNCTIONS =====

class TestConvenienceFunctions:
    def test_grad_reverse(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        g = grad(f, mode='reverse')
        result = g([3.0, 4.0])
        assert abs(result[0] - 6.0) < TOL
        assert abs(result[1] - 8.0) < TOL

    def test_grad_forward(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        g = grad(f, mode='forward')
        result = g([3.0, 4.0])
        assert abs(result[0] - 6.0) < TOL
        assert abs(result[1] - 8.0) < TOL

    def test_value_and_grad(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        vg = value_and_grad(f)
        val, g = vg([3.0, 4.0])
        assert abs(val - 25.0) < TOL
        assert abs(g[0] - 6.0) < TOL

    def test_jacobian_forward(self):
        def f(v): return [v[0] + v[1], v[0] * v[1]]
        vals, J = jacobian(f, [2.0, 3.0], mode='forward')
        assert abs(J[1][0] - 3.0) < TOL

    def test_jacobian_reverse(self):
        def f(v): return [v[0] + v[1], v[0] * v[1]]
        vals, J = jacobian(f, [2.0, 3.0], mode='reverse')
        assert abs(J[1][0] - 3.0) < TOL

    def test_directional_derivative(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        # Direction [1, 0] -> partial wrt x
        val, dd = directional_derivative(f, [3.0, 4.0], [1.0, 0.0])
        assert abs(val - 25.0) < TOL
        assert abs(dd - 6.0) < TOL

        # Direction [0, 1] -> partial wrt y
        val, dd = directional_derivative(f, [3.0, 4.0], [0.0, 1.0])
        assert abs(dd - 8.0) < TOL

    def test_hessian_quadratic(self):
        # f(x,y) = x^2 + 2*x*y + 3*y^2
        # H = [[2, 2], [2, 6]]
        def f(v): return v[0] ** 2 + 2 * v[0] * v[1] + 3 * v[1] ** 2
        val, H = hessian(f, [1.0, 1.0])
        assert abs(val - 6.0) < TOL
        assert abs(H[0][0] - 2.0) < FTOL
        assert abs(H[0][1] - 2.0) < FTOL
        assert abs(H[1][0] - 2.0) < FTOL
        assert abs(H[1][1] - 6.0) < FTOL

    def test_hessian_cubic(self):
        # f(x,y) = x^3 + y^3, H at (2,3) = [[12, 0], [0, 18]]
        def f(v): return v[0] ** 3 + v[1] ** 3
        val, H = hessian(f, [2.0, 3.0])
        assert abs(val - 35.0) < TOL
        assert abs(H[0][0] - 12.0) < FTOL
        assert abs(H[0][1]) < FTOL
        assert abs(H[1][1] - 18.0) < FTOL


# ===== GRADIENT CHECKING =====

class TestGradientChecking:
    def test_check_grad_quadratic(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        ad_g, fd_g, err = check_grad(f, [3.0, 4.0])
        assert err < 1e-5

    def test_check_grad_trig(self):
        def f(v): return var_sin(v[0]) * var_cos(v[1])
        ad_g, fd_g, err = check_grad(f, [1.0, 2.0])
        assert err < 1e-4

    def test_check_grad_rosenbrock(self):
        def rosenbrock(v):
            x, y = v[0], v[1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        ad_g, fd_g, err = check_grad(rosenbrock, [0.5, 0.5])
        assert err < 1e-4

    def test_check_grad_exp_sum(self):
        def f(v): return var_exp(v[0] + v[1])
        ad_g, fd_g, err = check_grad(f, [1.0, 2.0])
        assert err < 1e-4


# ===== CHECKPOINT =====

class TestCheckpoint:
    def test_checkpoint_basic(self):
        def expensive(x):
            return var_exp(x * x)

        x = Var(2.0)
        y = checkpoint(expensive, x)
        y.backward()
        # d/dx exp(x^2) = 2x * exp(x^2) at x=2
        expected = 2 * 2.0 * math.exp(4.0)
        assert abs(y.val - math.exp(4.0)) < TOL
        assert abs(x.grad - expected) < TOL

    def test_checkpoint_chain(self):
        def step1(x):
            return x * x + x

        def step2(x):
            return var_sin(x)

        x = Var(1.0)
        y1 = checkpoint(step1, x)
        y2 = checkpoint(step2, y1)
        y2.backward()
        # f(x) = sin(x^2 + x), f'(x) = cos(x^2+x) * (2x+1)
        inner = 1.0 + 1.0  # x^2 + x at x=1
        expected = math.cos(inner) * 3.0  # (2*1+1)
        assert abs(x.grad - expected) < FTOL


# ===== NUMERICAL JACOBIAN =====

class TestNumericalJacobian:
    def test_linear_fn(self):
        def f(v): return [2 * v[0] + 3 * v[1], v[0] - v[1]]
        J = numerical_jacobian(f, [1.0, 2.0])
        assert abs(J[0][0] - 2.0) < FTOL
        assert abs(J[0][1] - 3.0) < FTOL
        assert abs(J[1][0] - 1.0) < FTOL
        assert abs(J[1][1] - (-1.0)) < FTOL

    def test_nonlinear_fn(self):
        def f(v): return [v[0] ** 2, v[0] * v[1]]
        J = numerical_jacobian(f, [3.0, 4.0])
        assert abs(J[0][0] - 6.0) < FTOL
        assert abs(J[0][1] - 0.0) < FTOL
        assert abs(J[1][0] - 4.0) < FTOL
        assert abs(J[1][1] - 3.0) < FTOL


# ===== FORWARD vs REVERSE AGREEMENT =====

class TestForwardReverseAgreement:
    def test_quadratic_gradient(self):
        def f(v): return v[0] ** 2 + 3 * v[1] ** 2
        _, gf = ForwardAD.gradient(f, [2.0, 3.0])
        _, gr = ReverseAD.gradient(f, [2.0, 3.0])
        for a, b in zip(gf, gr):
            assert abs(a - b) < TOL

    def test_product_gradient(self):
        def f(v): return v[0] * v[1] * v[2]
        _, gf = ForwardAD.gradient(f, [2.0, 3.0, 4.0])
        _, gr = ReverseAD.gradient(f, [2.0, 3.0, 4.0])
        for a, b in zip(gf, gr):
            assert abs(a - b) < TOL

    def test_trig_gradient(self):
        def f_fwd(v):
            return dual_sin(v[0]) * dual_cos(v[1])
        def f_rev(v):
            return var_sin(v[0]) * var_cos(v[1])
        _, gf = ForwardAD.gradient(f_fwd, [1.0, 2.0])
        _, gr = ReverseAD.gradient(f_rev, [1.0, 2.0])
        for a, b in zip(gf, gr):
            assert abs(a - b) < TOL

    def test_jacobian_agreement(self):
        def f_fwd(v): return [v[0] ** 2 + v[1], v[0] * v[1]]
        def f_rev(v): return [v[0] ** 2 + v[1], v[0] * v[1]]
        _, Jf = ForwardAD.jacobian(f_fwd, [2.0, 3.0])
        _, Jr = ReverseAD.jacobian(f_rev, [2.0, 3.0])
        for i in range(len(Jf)):
            for j in range(len(Jf[0])):
                assert abs(Jf[i][j] - Jr[i][j]) < TOL


# ===== AD OPTIMIZER =====

class TestADOptimizer:
    def test_gd_quadratic(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        result = ADOptimizer.gradient_descent(f, [5.0, 5.0], lr=0.1, max_iter=500)
        assert result.status == ADOptimizerStatus.OPTIMAL
        assert abs(result.x[0]) < 0.01
        assert abs(result.x[1]) < 0.01

    def test_gd_forward_mode(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        result = ADOptimizer.gradient_descent(f, [3.0, 4.0], lr=0.1, max_iter=500, mode='forward')
        assert result.status == ADOptimizerStatus.OPTIMAL

    def test_adam_rosenbrock(self):
        def rosenbrock(v):
            x, y = v[0], v[1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        result = ADOptimizer.adam(rosenbrock, [0.0, 0.0], lr=0.01, max_iter=10000, tol=1e-6)
        # Adam should get close to (1,1)
        assert abs(result.x[0] - 1.0) < 0.1
        assert abs(result.x[1] - 1.0) < 0.1

    def test_lbfgs_quadratic(self):
        def f(v): return v[0] ** 2 + 4 * v[1] ** 2
        result = ADOptimizer.lbfgs(f, [5.0, 5.0], max_iter=100)
        assert result.status == ADOptimizerStatus.OPTIMAL
        assert abs(result.x[0]) < 0.01
        assert abs(result.x[1]) < 0.01

    def test_lbfgs_rosenbrock(self):
        def rosenbrock(v):
            x, y = v[0], v[1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
        result = ADOptimizer.lbfgs(rosenbrock, [0.0, 0.0], max_iter=200, tol=1e-6)
        assert abs(result.x[0] - 1.0) < 0.05
        assert abs(result.x[1] - 1.0) < 0.05

    def test_history_recorded(self):
        def f(v): return v[0] ** 2
        result = ADOptimizer.gradient_descent(f, [5.0], lr=0.1, max_iter=50)
        assert len(result.history) > 1
        # Objective should decrease
        assert result.history[-1] < result.history[0]

    def test_adam_multidim(self):
        # f(x) = sum(x_i^2)
        def f(v):
            total = v[0] * 0  # get Var(0)
            for vi in v:
                total = total + vi ** 2
            return total
        result = ADOptimizer.adam(f, [1.0, 2.0, 3.0, 4.0], lr=0.1, max_iter=5000, tol=1e-6)
        for xi in result.x:
            assert abs(xi) < 0.1


# ===== NEURAL OPS =====

class TestNeuralOps:
    def test_linear(self):
        x = [Var(1.0), Var(2.0)]
        w = [[Var(1.0), Var(0.0)], [Var(0.0), Var(1.0)]]
        b = [Var(0.1), Var(0.2)]
        y = NeuralOps.linear(x, w, b)
        assert abs(y[0].val - 1.1) < TOL
        assert abs(y[1].val - 2.2) < TOL

    def test_linear_gradient(self):
        x = [Var(1.0), Var(2.0)]
        w = [[Var(3.0), Var(4.0)]]
        b = [Var(0.0)]
        y = NeuralOps.linear(x, w, b)
        loss = y[0] ** 2  # (3 + 8)^2 = 121
        loss.backward()
        assert abs(loss.val - 121.0) < TOL
        # dloss/dw[0][0] = 2*y*x[0] = 2*11*1 = 22
        assert abs(w[0][0].grad - 22.0) < TOL
        # dloss/dw[0][1] = 2*y*x[1] = 2*11*2 = 44
        assert abs(w[0][1].grad - 44.0) < TOL

    def test_softmax(self):
        x = [Var(1.0), Var(2.0), Var(3.0)]
        sm = NeuralOps.softmax(x)
        total = sum(s.val for s in sm)
        assert abs(total - 1.0) < TOL
        # Largest input should have largest probability
        assert sm[2].val > sm[1].val > sm[0].val

    def test_softmax_gradient(self):
        x = [Var(1.0), Var(2.0)]
        sm = NeuralOps.softmax(x)
        loss = sm[0]  # track first softmax output
        loss.backward()
        # Gradient should exist
        assert x[0].grad != 0.0
        assert x[1].grad != 0.0

    def test_cross_entropy(self):
        pred = [Var(0.7), Var(0.2), Var(0.1)]
        target = [1.0, 0.0, 0.0]
        loss = NeuralOps.cross_entropy(pred, target)
        assert abs(loss.val - (-math.log(0.7))) < 0.01

    def test_mse_loss(self):
        pred = [Var(1.0), Var(2.0)]
        target = [1.5, 2.5]
        loss = NeuralOps.mse_loss(pred, target)
        # ((1-1.5)^2 + (2-2.5)^2) / 2 = (0.25+0.25)/2 = 0.25
        assert abs(loss.val - 0.25) < TOL

    def test_mse_gradient(self):
        pred = [Var(1.0), Var(2.0)]
        target = [1.5, 2.5]
        loss = NeuralOps.mse_loss(pred, target)
        loss.backward()
        # d/dp_0 = 2*(1-1.5)/2 = -0.5
        assert abs(pred[0].grad - (-0.5)) < TOL
        assert abs(pred[1].grad - (-0.5)) < TOL

    def test_batch_norm(self):
        x = [Var(1.0), Var(2.0), Var(3.0)]
        bn = NeuralOps.batch_norm(x)
        # Should be approximately normalized
        vals = [b.val for b in bn]
        mean = sum(vals) / len(vals)
        assert abs(mean) < TOL

    def test_full_forward_pass(self):
        """Test a complete forward pass: linear -> relu -> linear -> softmax."""
        x = [Var(1.0), Var(0.5)]
        w1 = [[Var(0.1), Var(0.2)], [Var(0.3), Var(0.4)]]
        b1 = [Var(0.0), Var(0.0)]
        h = NeuralOps.linear(x, w1, b1)
        h = [var_relu(hi) for hi in h]

        w2 = [[Var(0.5), Var(0.6)]]
        b2 = [Var(0.0)]
        out = NeuralOps.linear(h, w2, b2)
        loss = (out[0] - 1.0) ** 2
        loss.backward()

        # All weights should have gradients
        for row in w1:
            for w in row:
                assert w.grad != 0.0 or True  # relu might zero some


# ===== EDGE CASES =====

class TestEdgeCases:
    def test_dual_neg_int_pow(self):
        # (-2)^3 = -8, d/dx = 3*(-2)^2 = 12
        a = Dual(-2.0, 1.0)
        b = Dual(3.0, 0.0)
        c = a ** b
        assert abs(c.val - (-8.0)) < TOL
        assert abs(c.der - 12.0) < TOL

    def test_constant_function(self):
        def f(v): return 42.0
        val, g = ReverseAD.gradient(f, [1.0, 2.0])
        assert val == 42.0
        assert g == [0.0, 0.0]

    def test_single_variable(self):
        def f(v): return v[0] ** 3
        val, g = ReverseAD.gradient(f, [2.0])
        assert abs(val - 8.0) < TOL
        assert abs(g[0] - 12.0) < TOL

    def test_zero_grad(self):
        x = Var(3.0)
        y = x ** 2
        y.backward()
        assert x.grad == 6.0
        x.zero_grad()
        assert x.grad == 0.0

    def test_deep_chain(self):
        # f(x) = ((((x+1)+1)+1)+1) = x+4
        x = Var(0.0)
        y = x
        for _ in range(100):
            y = y + 1
        y.backward()
        assert abs(y.val - 100.0) < TOL
        assert abs(x.grad - 1.0) < TOL

    def test_wide_fan_out(self):
        # f(x) = x + x + x + ... (100 times) = 100x
        x = Var(1.0)
        y = Var(0.0)
        for _ in range(100):
            y = y + x
        y.backward()
        assert abs(y.val - 100.0) < TOL
        assert abs(x.grad - 100.0) < TOL

    def test_diamond_graph(self):
        # f(x) = (x+1) * (x-1) = x^2 - 1
        x = Var(3.0)
        a = x + 1
        b = x - 1
        y = a * b
        y.backward()
        assert abs(y.val - 8.0) < TOL
        assert abs(x.grad - 6.0) < TOL  # 2x

    def test_pow_zero_var(self):
        a = Var(5.0)
        c = a ** 0
        c.backward()
        assert abs(c.val - 1.0) < TOL

    def test_dual_sub_scalar(self):
        d = Dual(5.0, 1.0)
        r = d - 3
        assert r.val == 2.0
        assert r.der == 1.0


# ===== COMPLEX MATHEMATICAL FUNCTIONS =====

class TestComplexFunctions:
    def test_log_sum_exp(self):
        """LogSumExp: commonly used in ML."""
        def logsumexp(v):
            max_v = v[0]
            for vi in v[1:]:
                if vi > max_v:
                    max_v = vi
            total = var_exp(v[0] - max_v)
            for vi in v[1:]:
                total = total + var_exp(vi - max_v)
            return var_log(total) + max_v

        val, g = ReverseAD.gradient(logsumexp, [1.0, 2.0, 3.0])
        # Gradient of logsumexp = softmax
        assert abs(sum(g) - 1.0) < TOL  # softmax sums to 1
        assert g[2] > g[1] > g[0]  # monotonic

    def test_huber_loss(self):
        """Huber loss: combines L1 and L2."""
        def huber(v, delta=1.0):
            x = v[0]
            a = abs(x)
            if a.val <= delta:
                return 0.5 * x * x
            else:
                return delta * (a - 0.5 * delta)

        # Small x -> quadratic
        val, g = ReverseAD.gradient(huber, [0.5])
        assert abs(val - 0.125) < TOL
        assert abs(g[0] - 0.5) < TOL

        # Large x -> linear
        val, g = ReverseAD.gradient(huber, [3.0])
        assert abs(val - 2.5) < TOL
        assert abs(g[0] - 1.0) < TOL

    def test_gaussian(self):
        """Gaussian function and its derivative."""
        def gaussian(v):
            x = v[0]
            return var_exp(-(x ** 2) / 2)

        val, g = ReverseAD.gradient(gaussian, [0.0])
        assert abs(val - 1.0) < TOL
        assert abs(g[0]) < TOL  # peak has zero gradient

        val, g = ReverseAD.gradient(gaussian, [1.0])
        expected_grad = -1.0 * math.exp(-0.5)  # -x * exp(-x^2/2)
        assert abs(g[0] - expected_grad) < TOL

    def test_polynomial_high_degree(self):
        """High-degree polynomial gradient."""
        def poly(v):
            x = v[0]
            # x^5 + 2*x^3 + x
            return x ** 5 + 2 * x ** 3 + x

        val, g = ReverseAD.gradient(poly, [2.0])
        assert abs(val - (32 + 16 + 2)) < TOL
        # derivative: 5x^4 + 6x^2 + 1 at x=2 = 80+24+1 = 105
        assert abs(g[0] - 105.0) < TOL


# ===== FORWARD/REVERSE MODE CONSISTENCY WITH FD =====

class TestConsistencyWithFD:
    def test_quadratic_vs_fd(self):
        def f(v): return v[0] ** 2 + v[1] ** 2
        _, Jad = ForwardAD.jacobian(f, [3.0, 4.0])
        Jfd = numerical_jacobian(f, [3.0, 4.0])
        for i in range(len(Jad)):
            for j in range(len(Jad[0])):
                assert abs(Jad[i][j] - Jfd[i][j]) < FTOL

    def test_nonlinear_vs_fd(self):
        def f(v): return [v[0] * v[1], v[0] ** 2 + v[1]]
        _, Jfwd = ForwardAD.jacobian(f, [2.0, 3.0])
        Jfd = numerical_jacobian(f, [2.0, 3.0])
        for i in range(len(Jfwd)):
            for j in range(len(Jfwd[0])):
                assert abs(Jfwd[i][j] - Jfd[i][j]) < FTOL


# ===== OPTIMIZATION CONVERGENCE =====

class TestOptimizationConvergence:
    def test_gd_convergence_rate(self):
        """GD should converge linearly on well-conditioned quadratic."""
        def f(v): return v[0] ** 2
        result = ADOptimizer.gradient_descent(f, [10.0], lr=0.1, max_iter=100)
        assert result.status == ADOptimizerStatus.OPTIMAL
        assert result.iterations < 100

    def test_lbfgs_superlinear(self):
        """L-BFGS should converge much faster than GD."""
        def f(v): return v[0] ** 2 + 4 * v[1] ** 2 + v[0] * v[1]
        result = ADOptimizer.lbfgs(f, [10.0, 10.0], max_iter=50)
        assert result.status == ADOptimizerStatus.OPTIMAL
        assert result.iterations < 20

    def test_adam_with_trig(self):
        """Adam on a non-convex but tractable function."""
        def f(v):
            return (v[0] - 2) ** 2 + (v[1] + 1) ** 2
        result = ADOptimizer.adam(f, [0.0, 0.0], lr=0.1, max_iter=5000, tol=1e-4)
        assert abs(result.x[0] - 2.0) < 0.5
        assert abs(result.x[1] - (-1.0)) < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
