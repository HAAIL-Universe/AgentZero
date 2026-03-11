"""
C130: ODE Solvers -- Composing C127 (VectorOps) + C128 (Automatic Differentiation)

Ordinary Differential Equation solvers for y'(t) = f(t, y).

Components:
  - Euler method (1st order explicit)
  - Midpoint method (2nd order explicit)
  - RK4 (4th order explicit, classic)
  - Dormand-Prince RK45 (adaptive step-size, 4th/5th order embedded pair)
  - Implicit Euler (1st order, A-stable, for stiff systems)
  - Implicit Trapezoid (2nd order, A-stable)
  - BDF2 (2nd order backward differentiation, stiff systems)
  - Event detection (zero-crossing)
  - Sensitivity analysis via adjoint method (composes C128 AD)
  - ODE parameter fitting via AD optimization

Composition:
  - C127 VectorOps: vector arithmetic for state vectors
  - C128 autodiff: gradients for implicit solvers and sensitivity analysis
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C127_convex_optimization'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))

from convex_optimization import VectorOps as V
from autodiff import (
    ForwardAD, ReverseAD, Var, Dual,
    var_sin, var_cos, var_exp, var_log, var_sqrt,
    dual_sin, dual_cos, dual_exp, dual_log, dual_sqrt,
    grad, value_and_grad,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ODEResult:
    """Result of an ODE integration."""

    def __init__(self, t, y, success=True, message='', n_steps=0, n_f_evals=0,
                 events=None):
        self.t = t              # list of time points
        self.y = y              # list of state vectors (each a list of floats)
        self.success = success
        self.message = message
        self.n_steps = n_steps
        self.n_f_evals = n_f_evals
        self.events = events or []  # list of (t_event, y_event, event_index)

    def __repr__(self):
        n = len(self.t)
        dim = len(self.y[0]) if self.y else 0
        return (f"ODEResult(n_points={n}, dim={dim}, success={self.success}, "
                f"steps={self.n_steps}, f_evals={self.n_f_evals})")


# ---------------------------------------------------------------------------
# Fixed-step methods
# ---------------------------------------------------------------------------

class EulerMethod:
    """Forward Euler: y_{n+1} = y_n + h * f(t_n, y_n)."""

    def solve(self, f, t_span, y0, n_steps=100):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        n_evals = 0

        for _ in range(n_steps):
            dy = f(t, y)
            n_evals += 1
            y = V.add(y, V.scale(h, dy))
            t += h
            t_list.append(t)
            y_list.append(list(y))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)


class MidpointMethod:
    """Explicit midpoint (RK2): uses midpoint slope."""

    def solve(self, f, t_span, y0, n_steps=100):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        n_evals = 0

        for _ in range(n_steps):
            k1 = f(t, y)
            y_mid = V.add(y, V.scale(h / 2, k1))
            k2 = f(t + h / 2, y_mid)
            n_evals += 2
            y = V.add(y, V.scale(h, k2))
            t += h
            t_list.append(t)
            y_list.append(list(y))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)


class RK4Method:
    """Classic 4th-order Runge-Kutta."""

    def solve(self, f, t_span, y0, n_steps=100):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        n_evals = 0

        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, V.add(y, V.scale(h / 2, k1)))
            k3 = f(t + h / 2, V.add(y, V.scale(h / 2, k2)))
            k4 = f(t + h, V.add(y, V.scale(h, k3)))
            n_evals += 4

            # y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
            dy = V.add(V.add(k1, V.scale(2, k2)),
                       V.add(V.scale(2, k3), k4))
            y = V.add(y, V.scale(h / 6, dy))
            t += h
            t_list.append(t)
            y_list.append(list(y))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)


# ---------------------------------------------------------------------------
# Adaptive step-size: Dormand-Prince (RK45)
# ---------------------------------------------------------------------------

class DormandPrince45:
    """
    Dormand-Prince embedded RK4(5) with adaptive step control.
    Uses 4th order solution for stepping, 5th order for error estimation.
    """

    # Butcher tableau coefficients
    A = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84],
    ]
    # 4th order weights (for stepping)
    B4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    # 5th order weights (for error estimation)
    B5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    # Time fractions
    C = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]

    def solve(self, f, t_span, y0, rtol=1e-6, atol=1e-9, max_steps=10000,
              h_init=None, h_min=1e-12, h_max=None, dense_output=False,
              events=None):
        t0, tf = t_span
        dim = len(y0)
        if h_max is None:
            h_max = abs(tf - t0)
        if h_init is None:
            # Estimate initial step size
            dy0 = f(t0, list(y0))
            norm_dy = V.norm(dy0)
            norm_y = V.norm(list(y0))
            h_init = min(0.01 * max(norm_y, 1.0) / max(norm_dy, 1e-10),
                         h_max, abs(tf - t0))

        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        h = h_init
        n_steps = 0
        n_evals = 0
        n_rejected = 0
        detected_events = []

        while t < tf - 1e-14 * abs(tf):
            if n_steps >= max_steps:
                return ODEResult(t_list, y_list, success=False,
                                 message='Max steps exceeded',
                                 n_steps=n_steps, n_f_evals=n_evals)

            # Clamp step to not overshoot
            h = min(h, tf - t)
            h = max(h, h_min)

            # Compute stages
            k = [None] * 7
            k[0] = f(t, y)
            n_evals += 1

            for i in range(1, 7):
                ti = t + self.C[i] * h
                yi = list(y)
                for j in range(i):
                    if self.A[i][j] != 0:
                        yi = V.add(yi, V.scale(h * self.A[i][j], k[j]))
                k[i] = f(ti, yi)
                n_evals += 1

            # 4th order solution
            y4 = list(y)
            for i in range(7):
                if self.B4[i] != 0:
                    y4 = V.add(y4, V.scale(h * self.B4[i], k[i]))

            # 5th order solution
            y5 = list(y)
            for i in range(7):
                if self.B5[i] != 0:
                    y5 = V.add(y5, V.scale(h * self.B5[i], k[i]))

            # Error estimate
            err_vec = V.sub(y5, y4)
            # Scaled error: err_i / (atol + rtol * max(|y_i|, |y4_i|))
            err_norm = 0.0
            for i in range(dim):
                sc = atol + rtol * max(abs(y[i]), abs(y4[i]))
                err_norm += (err_vec[i] / sc) ** 2
            err_norm = math.sqrt(err_norm / dim)

            if err_norm <= 1.0:
                # Accept step
                # Event detection
                if events:
                    for ei, event_fn in enumerate(events):
                        g0 = event_fn(t, y)
                        g1 = event_fn(t + h, y4)
                        if g0 * g1 < 0:
                            # Zero crossing -- bisection
                            t_event, y_event = self._bisect_event(
                                f, event_fn, t, y, t + h, y4, g0, g1)
                            detected_events.append((t_event, y_event, ei))

                t += h
                y = y4
                t_list.append(t)
                y_list.append(list(y))
                n_steps += 1

                # Step size adjustment (PI controller)
                if err_norm > 0:
                    factor = 0.9 * err_norm ** (-0.2)
                    factor = max(0.2, min(factor, 5.0))
                    h *= factor
                else:
                    h *= 5.0
                h = min(h, h_max)
            else:
                # Reject step
                n_rejected += 1
                factor = 0.9 * err_norm ** (-0.2)
                factor = max(0.2, min(factor, 1.0))
                h *= factor
                h = max(h, h_min)

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals,
                         events=detected_events)

    def _bisect_event(self, f, event_fn, t0, y0, t1, y1, g0, g1, tol=1e-10):
        """Find zero crossing of event function via bisection."""
        for _ in range(50):
            tm = (t0 + t1) / 2
            if abs(t1 - t0) < tol:
                break
            # Quick interpolation: linear for y
            alpha = (tm - t0) / (t1 - t0) if t1 != t0 else 0.5
            ym = V.add(V.scale(1 - alpha, y0), V.scale(alpha, y1))
            gm = event_fn(tm, ym)
            if gm * g0 < 0:
                t1, y1, g1 = tm, ym, gm
            else:
                t0, y0, g0 = tm, ym, gm
        return tm, ym


# ---------------------------------------------------------------------------
# Implicit methods (for stiff systems)
# ---------------------------------------------------------------------------

class ImplicitEuler:
    """
    Backward Euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1}).
    Solves nonlinear system via Newton iteration using numerical Jacobian.
    A-stable, 1st order.
    """

    def solve(self, f, t_span, y0, n_steps=100, newton_tol=1e-10,
              newton_max_iter=50):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        dim = len(y0)
        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        n_evals = 0

        for _ in range(n_steps):
            t_next = t + h
            # Initial guess: explicit Euler
            dy = f(t, y)
            n_evals += 1
            y_guess = V.add(y, V.scale(h, dy))

            # Newton iteration: solve G(z) = z - y - h*f(t_next, z) = 0
            z = list(y_guess)
            for _ni in range(newton_max_iter):
                fz = f(t_next, z)
                n_evals += 1
                G = V.sub(z, V.add(y, V.scale(h, fz)))

                # Check convergence
                if V.norm(G) < newton_tol:
                    break

                # Numerical Jacobian of G
                J = self._jacobian_G(f, t_next, z, y, h, dim)
                n_evals += dim

                # Solve J * delta = -G
                neg_G = V.scale(-1, G)
                delta = V.solve_linear(J, neg_G)
                if delta is None:
                    # Fallback: explicit step
                    z = V.add(y, V.scale(h, fz))
                    break
                z = V.add(z, delta)

            y = z
            t = t_next
            t_list.append(t)
            y_list.append(list(y))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)

    def _jacobian_G(self, f, t, z, y, h, dim, eps=1e-7):
        """Numerical Jacobian of G(z) = z - y - h*f(t, z)."""
        J = []
        fz = f(t, z)
        for j in range(dim):
            z_pert = list(z)
            z_pert[j] += eps
            fz_pert = f(t, z_pert)
            row = []
            for i in range(dim):
                dGi_dzj = (1.0 if i == j else 0.0) - h * (fz_pert[i] - fz[i]) / eps
                row.append(dGi_dzj)
            J.append(row)
        # J is stored row-major: J[i][j] = dG_i/dz_j
        return J


class ImplicitTrapezoid:
    """
    Implicit trapezoidal (Crank-Nicolson):
    y_{n+1} = y_n + (h/2) * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
    A-stable, 2nd order.
    """

    def solve(self, f, t_span, y0, n_steps=100, newton_tol=1e-10,
              newton_max_iter=50):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        dim = len(y0)
        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        n_evals = 0

        for _ in range(n_steps):
            t_next = t + h
            fn = f(t, y)
            n_evals += 1

            # Initial guess: explicit Euler
            y_guess = V.add(y, V.scale(h, fn))

            # Newton: G(z) = z - y - (h/2)*(fn + f(t_next, z)) = 0
            z = list(y_guess)
            for _ni in range(newton_max_iter):
                fz = f(t_next, z)
                n_evals += 1
                rhs = V.add(y, V.scale(h / 2, V.add(fn, fz)))
                G = V.sub(z, rhs)

                if V.norm(G) < newton_tol:
                    break

                # Jacobian of G: I - (h/2) * df/dz
                J = self._jacobian_G(f, t_next, z, h, dim)
                n_evals += dim
                neg_G = V.scale(-1, G)
                delta = V.solve_linear(J, neg_G)
                if delta is None:
                    z = rhs
                    break
                z = V.add(z, delta)

            y = z
            t = t_next
            t_list.append(t)
            y_list.append(list(y))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)

    def _jacobian_G(self, f, t, z, h, dim, eps=1e-7):
        """Numerical Jacobian of G(z) = z - y - (h/2)*(fn + f(t, z))."""
        fz = f(t, z)
        J = []
        for j in range(dim):
            z_pert = list(z)
            z_pert[j] += eps
            fz_pert = f(t, z_pert)
            row = []
            for i in range(dim):
                dGi_dzj = (1.0 if i == j else 0.0) - (h / 2) * (fz_pert[i] - fz[i]) / eps
                row.append(dGi_dzj)
            J.append(row)
        return J


class BDF2:
    """
    2nd-order Backward Differentiation Formula.
    y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2/3)*h*f(t_{n+1}, y_{n+1})
    A-stable, good for stiff systems. Uses implicit Euler for the first step.
    """

    def solve(self, f, t_span, y0, n_steps=100, newton_tol=1e-10,
              newton_max_iter=50):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        dim = len(y0)
        t_list = [t0]
        y_list = [list(y0)]
        n_evals = 0

        # First step: implicit Euler
        ie = ImplicitEuler()
        res1 = ie.solve(f, (t0, t0 + h), y0, n_steps=1,
                        newton_tol=newton_tol, newton_max_iter=newton_max_iter)
        y_prev = list(y0)
        y_curr = list(res1.y[-1])
        t_list.append(t0 + h)
        y_list.append(list(y_curr))
        n_evals += res1.n_f_evals

        # BDF2 steps
        for step in range(1, n_steps):
            t_next = t0 + (step + 1) * h

            # Predictor
            pred = V.add(V.scale(4/3, y_curr), V.scale(-1/3, y_prev))

            # Newton: G(z) = z - (4/3)*y_n + (1/3)*y_{n-1} - (2/3)*h*f(t_{n+1}, z) = 0
            z = list(pred)
            for _ni in range(newton_max_iter):
                fz = f(t_next, z)
                n_evals += 1
                rhs = V.add(pred, V.scale(2 * h / 3, fz))
                G = V.sub(z, rhs)

                if V.norm(G) < newton_tol:
                    break

                # Jacobian: I - (2h/3) * df/dz
                J = []
                for j in range(dim):
                    z_pert = list(z)
                    z_pert[j] += 1e-7
                    fz_pert = f(t_next, z_pert)
                    n_evals += 1
                    row = []
                    for i in range(dim):
                        dGi_dzj = (1.0 if i == j else 0.0) - (2*h/3) * (fz_pert[i] - fz[i]) / 1e-7
                        row.append(dGi_dzj)
                    J.append(row)

                neg_G = V.scale(-1, G)
                delta = V.solve_linear(J, neg_G)
                if delta is None:
                    z = rhs
                    break
                z = V.add(z, delta)

            y_prev = y_curr
            y_curr = z
            t_list.append(t_next)
            y_list.append(list(y_curr))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)


# ---------------------------------------------------------------------------
# System builders (helper for common ODE systems)
# ---------------------------------------------------------------------------

def second_order_to_system(accel_fn):
    """
    Convert a 2nd-order ODE y'' = a(t, y, y') to a 1st-order system.
    Returns f(t, state) where state = [y, y'].
    accel_fn(t, y, v) -> acceleration
    Works for scalar y.
    """
    def system(t, state):
        y, v = state[0], state[1]
        a = accel_fn(t, y, v)
        return [v, a]
    return system


def harmonic_oscillator(omega=1.0, zeta=0.0):
    """Damped harmonic oscillator: y'' + 2*zeta*omega*y' + omega^2*y = 0."""
    def accel(t, y, v):
        return -2 * zeta * omega * v - omega * omega * y
    return second_order_to_system(accel)


def van_der_pol(mu=1.0):
    """Van der Pol oscillator: y'' - mu*(1-y^2)*y' + y = 0."""
    def accel(t, y, v):
        return mu * (1 - y * y) * v - y
    return second_order_to_system(accel)


def lorenz(sigma=10.0, rho=28.0, beta=8/3):
    """Lorenz system (chaotic attractor)."""
    def f(t, state):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
    return f


def lotka_volterra(alpha=1.0, beta=0.5, delta=0.5, gamma=2.0):
    """Predator-prey model."""
    def f(t, state):
        prey, pred = state
        return [
            alpha * prey - beta * prey * pred,
            delta * prey * pred - gamma * pred,
        ]
    return f


def sir_model(beta_param=0.3, gamma_param=0.1):
    """SIR epidemiological model."""
    def f(t, state):
        s, i, r = state
        return [
            -beta_param * s * i,
            beta_param * s * i - gamma_param * i,
            gamma_param * i,
        ]
    return f


# ---------------------------------------------------------------------------
# Sensitivity analysis via adjoint method (composes C128 AD)
# ---------------------------------------------------------------------------

class SensitivityAnalysis:
    """
    Compute dy_final/dy0 and dy_final/dp using forward sensitivity or
    finite-difference approaches.
    """

    @staticmethod
    def forward_sensitivity(f, t_span, y0, n_steps=100, solver=None):
        """
        Compute sensitivity matrix S(t) = dy(t)/dy0 via variational equation.
        Augments the ODE with the sensitivity equation:
          dS/dt = (df/dy) * S, S(0) = I

        Returns (result, S_final) where S_final[i][j] = dy_i(T)/dy0_j
        """
        if solver is None:
            solver = RK4Method()

        dim = len(y0)
        eps = 1e-7

        # Augmented state: [y (dim), S flattened (dim*dim)]
        S0_flat = []
        for i in range(dim):
            for j in range(dim):
                S0_flat.append(1.0 if i == j else 0.0)

        aug_y0 = list(y0) + S0_flat

        def aug_f(t, aug_state):
            y = aug_state[:dim]
            S_flat = aug_state[dim:]

            # dy/dt = f(t, y)
            dy = f(t, y)

            # Compute Jacobian df/dy numerically
            fy = f(t, y)
            J = []
            for j in range(dim):
                y_pert = list(y)
                y_pert[j] += eps
                fy_pert = f(t, y_pert)
                col = [(fy_pert[i] - fy[i]) / eps for i in range(dim)]
                J.append(col)
            # J[j][i] = df_i/dy_j, transpose to get J[i][j] = df_i/dy_j
            Jt = [[J[j][i] for j in range(dim)] for i in range(dim)]

            # dS/dt = J * S
            dS_flat = [0.0] * (dim * dim)
            for i in range(dim):
                for j in range(dim):
                    val = 0.0
                    for k in range(dim):
                        val += Jt[i][k] * S_flat[k * dim + j]
                    dS_flat[i * dim + j] = val

            return dy + dS_flat

        result = solver.solve(aug_f, t_span, aug_y0, n_steps=n_steps)

        # Extract final S
        final_aug = result.y[-1]
        y_final = final_aug[:dim]
        S_flat = final_aug[dim:]
        S_final = []
        for i in range(dim):
            row = S_flat[i * dim:(i + 1) * dim]
            S_final.append(row)

        # Build clean result with just y
        t_clean = result.t
        y_clean = [state[:dim] for state in result.y]
        clean_result = ODEResult(t_clean, y_clean, n_steps=result.n_steps,
                                 n_f_evals=result.n_f_evals)

        return clean_result, S_final

    @staticmethod
    def parameter_sensitivity(f_factory, params, t_span, y0, n_steps=100,
                              solver=None, eps=1e-6):
        """
        Compute dy_final/dp_k for each parameter p_k via finite differences.
        f_factory(params) -> f(t, y) ODE right-hand side.
        Returns (result, sensitivities) where sensitivities[k] is a list of
        dy_i(T)/dp_k.
        """
        if solver is None:
            solver = RK4Method()

        n_params = len(params)
        dim = len(y0)

        # Base solution
        f_base = f_factory(params)
        result = solver.solve(f_base, t_span, y0, n_steps=n_steps)
        y_base = result.y[-1]

        # Perturb each parameter
        sensitivities = []
        for k in range(n_params):
            params_pert = list(params)
            params_pert[k] += eps
            f_pert = f_factory(params_pert)
            result_pert = solver.solve(f_pert, t_span, y0, n_steps=n_steps)
            y_pert = result_pert.y[-1]
            sens = [(y_pert[i] - y_base[i]) / eps for i in range(dim)]
            sensitivities.append(sens)

        return result, sensitivities


# ---------------------------------------------------------------------------
# Parameter fitting via optimization (composes C128 AD)
# ---------------------------------------------------------------------------

class ODEParameterFitter:
    """
    Fit ODE parameters to observed data by minimizing residuals.
    Uses numerical gradients + simple gradient descent.
    """

    @staticmethod
    def fit(f_factory, t_data, y_data, params0, t_span=None,
            n_steps=200, lr=0.001, max_iter=500, tol=1e-8, solver=None):
        """
        f_factory(params) -> f(t, y)
        t_data: list of observation times (must be subset of integration points)
        y_data: list of observed states at t_data
        params0: initial parameter guess
        Returns (optimal_params, loss_history)
        """
        if solver is None:
            solver = RK4Method()
        if t_span is None:
            t_span = (t_data[0], t_data[-1])

        y0 = y_data[0]
        dim = len(y0)
        params = list(params0)
        n_params = len(params)
        loss_history = []
        eps = 1e-6

        for iteration in range(max_iter):
            # Compute loss
            loss, y_pred_at_data = ODEParameterFitter._compute_loss(
                f_factory, params, t_span, y0, t_data, y_data, n_steps, solver)
            loss_history.append(loss)

            if loss < tol:
                break

            # Numerical gradient
            grad_params = []
            for k in range(n_params):
                params_pert = list(params)
                params_pert[k] += eps
                loss_pert, _ = ODEParameterFitter._compute_loss(
                    f_factory, params_pert, t_span, y0, t_data, y_data,
                    n_steps, solver)
                grad_params.append((loss_pert - loss) / eps)

            # Gradient descent step
            for k in range(n_params):
                params[k] -= lr * grad_params[k]

            # Early stopping if gradient is tiny
            grad_norm = math.sqrt(sum(g * g for g in grad_params))
            if grad_norm < tol:
                break

        return params, loss_history

    @staticmethod
    def _compute_loss(f_factory, params, t_span, y0, t_data, y_data,
                      n_steps, solver):
        """Compute sum-of-squares loss between ODE solution and data."""
        f = f_factory(params)
        result = solver.solve(f, t_span, list(y0), n_steps=n_steps)

        # Map t_data to nearest integration points
        loss = 0.0
        y_pred_at_data = []
        for td, yd in zip(t_data, y_data):
            # Find closest t in result
            best_idx = 0
            best_dist = abs(result.t[0] - td)
            for idx, ti in enumerate(result.t):
                d = abs(ti - td)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            y_pred = result.y[best_idx]
            y_pred_at_data.append(y_pred)
            for i in range(len(yd)):
                loss += (y_pred[i] - yd[i]) ** 2

        return loss, y_pred_at_data


# ---------------------------------------------------------------------------
# Stiffness detection
# ---------------------------------------------------------------------------

def estimate_stiffness_ratio(f, t, y, eps=1e-7):
    """
    Estimate stiffness ratio |Re(lambda_max)/Re(lambda_min)| of the Jacobian
    via power iteration approximation. High ratio suggests stiff system.
    """
    dim = len(y)
    fy = f(t, y)

    # Build Jacobian
    J = []
    for j in range(dim):
        y_pert = list(y)
        y_pert[j] += eps
        fy_pert = f(t, y_pert)
        col = [(fy_pert[i] - fy[i]) / eps for i in range(dim)]
        J.append(col)
    # Transpose: J_mat[i][j] = df_i/dy_j
    J_mat = [[J[j][i] for j in range(dim)] for i in range(dim)]

    if dim == 1:
        return abs(J_mat[0][0]), [J_mat[0][0]]

    # Power iteration for dominant eigenvalue magnitude
    v = [1.0 / math.sqrt(dim)] * dim
    lam_max = 0.0
    for _ in range(50):
        w = V.mat_vec(J_mat, v)
        nw = V.norm(w)
        if nw < 1e-15:
            break
        lam_max = nw
        v = V.scale(1.0 / nw, w)

    # Inverse power iteration for smallest eigenvalue magnitude
    # Approximate by shifting
    lam_min = lam_max
    u = [1.0 / math.sqrt(dim)] * dim
    # Shift matrix: (J - sigma*I), with sigma = 0
    # We need to solve (J)x = u iteratively
    for _ in range(50):
        w = V.solve_linear(J_mat, u)
        if w is None:
            lam_min = 0.0
            break
        nw = V.norm(w)
        if nw < 1e-15:
            break
        lam_min = 1.0 / nw
        u = V.scale(1.0 / nw, w)

    ratio = lam_max / max(lam_min, 1e-15)
    return ratio, [lam_max, lam_min]


# ---------------------------------------------------------------------------
# Convenience: solve_ode dispatcher
# ---------------------------------------------------------------------------

def solve_ode(f, t_span, y0, method='rk4', **kwargs):
    """
    Unified ODE solver interface.

    Args:
        f: callable f(t, y) -> dy/dt
        t_span: (t0, tf)
        y0: initial state (list of floats)
        method: 'euler', 'midpoint', 'rk4', 'rk45', 'implicit_euler',
                'implicit_trapezoid', 'bdf2'
        **kwargs: solver-specific options

    Returns:
        ODEResult
    """
    methods = {
        'euler': EulerMethod,
        'midpoint': MidpointMethod,
        'rk4': RK4Method,
        'rk45': DormandPrince45,
        'implicit_euler': ImplicitEuler,
        'implicit_trapezoid': ImplicitTrapezoid,
        'bdf2': BDF2,
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Available: {list(methods.keys())}")

    solver = methods[method]()
    return solver.solve(f, t_span, y0, **kwargs)


# ---------------------------------------------------------------------------
# Multi-step Adams methods
# ---------------------------------------------------------------------------

class AdamsBashforth:
    """
    Adams-Bashforth explicit multi-step methods (orders 1-4).
    Uses stored previous slopes for higher-order extrapolation.
    """

    def solve(self, f, t_span, y0, n_steps=100, order=4):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t_list = [t0]
        y_list = [list(y0)]
        t = t0
        y = list(y0)
        n_evals = 0

        # Bootstrap with RK4 for first (order-1) steps
        slopes = []
        k0 = f(t, y)
        n_evals += 1
        slopes.append(k0)

        rk4 = RK4Method()
        for i in range(min(order - 1, n_steps)):
            res = rk4.solve(f, (t, t + h), y, n_steps=1)
            n_evals += 4
            t += h
            y = list(res.y[-1])
            t_list.append(t)
            y_list.append(list(y))
            k = f(t, y)
            n_evals += 1
            slopes.append(k)

        # Adams-Bashforth coefficients
        ab_coeffs = {
            1: [1],
            2: [3/2, -1/2],
            3: [23/12, -16/12, 5/12],
            4: [55/24, -59/24, 37/24, -9/24],
        }
        coeffs = ab_coeffs[min(order, 4)]

        steps_done = min(order - 1, n_steps)
        for step in range(steps_done, n_steps):
            # y_{n+1} = y_n + h * sum(c_j * f_{n-j})
            dy = V.zeros(len(y0))
            for j, c in enumerate(coeffs):
                idx = len(slopes) - 1 - j
                if idx >= 0:
                    dy = V.add(dy, V.scale(c, slopes[idx]))
            y = V.add(y, V.scale(h, dy))
            t += h
            t_list.append(t)
            y_list.append(list(y))

            k = f(t, y)
            n_evals += 1
            slopes.append(k)

            # Keep only needed history
            if len(slopes) > order:
                slopes = slopes[-order:]

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)


# ---------------------------------------------------------------------------
# Symplectic integrator for Hamiltonian systems
# ---------------------------------------------------------------------------

class StormerVerlet:
    """
    Stormer-Verlet (leapfrog) symplectic integrator for Hamiltonian systems.
    Preserves energy over long integrations.
    System: q' = p/m, p' = F(q)
    Takes force_fn(q) -> force vector.
    State is [q0, ..., q_{n-1}, p0, ..., p_{n-1}].
    """

    def solve(self, force_fn, t_span, q0, p0, n_steps=100, mass=None):
        t0, tf = t_span
        h = (tf - t0) / n_steps
        dim = len(q0)
        if mass is None:
            mass = [1.0] * dim

        t_list = [t0]
        state = list(q0) + list(p0)
        y_list = [list(state)]
        q = list(q0)
        p = list(p0)
        n_evals = 0

        for _ in range(n_steps):
            # Half-step momentum
            F = force_fn(q)
            n_evals += 1
            p = [p[i] + 0.5 * h * F[i] for i in range(dim)]

            # Full-step position
            q = [q[i] + h * p[i] / mass[i] for i in range(dim)]

            # Half-step momentum
            F = force_fn(q)
            n_evals += 1
            p = [p[i] + 0.5 * h * F[i] for i in range(dim)]

            t0 += h
            state = list(q) + list(p)
            t_list.append(t0)
            y_list.append(list(state))

        return ODEResult(t_list, y_list, n_steps=n_steps, n_f_evals=n_evals)


# ---------------------------------------------------------------------------
# Phase portrait analysis
# ---------------------------------------------------------------------------

class PhasePortrait:
    """Analyze fixed points and stability of 2D autonomous systems."""

    @staticmethod
    def find_fixed_points(f, search_region, grid_size=10, newton_tol=1e-10,
                          newton_max_iter=50):
        """
        Find fixed points of f(t, y) = 0 in 2D search region.
        search_region: ((x_min, x_max), (y_min, y_max))
        Returns list of (point, classification) tuples.
        """
        (x_min, x_max), (y_min, y_max) = search_region
        dx = (x_max - x_min) / grid_size
        dy = (y_max - y_min) / grid_size

        found = []
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                x0 = x_min + i * dx
                y0 = y_min + j * dy
                pt = PhasePortrait._newton_2d(f, [x0, y0], newton_tol,
                                              newton_max_iter)
                if pt is not None:
                    # Check if duplicate
                    is_dup = False
                    for prev_pt, _ in found:
                        if abs(pt[0] - prev_pt[0]) < 1e-6 and abs(pt[1] - prev_pt[1]) < 1e-6:
                            is_dup = True
                            break
                    if not is_dup:
                        cls = PhasePortrait.classify_fixed_point(f, pt)
                        found.append((pt, cls))

        return found

    @staticmethod
    def classify_fixed_point(f, point, eps=1e-7):
        """
        Classify a 2D fixed point using eigenvalue analysis of the Jacobian.
        Returns: 'stable_node', 'unstable_node', 'saddle', 'stable_spiral',
                 'unstable_spiral', 'center', 'degenerate'
        """
        x, y = point
        # Compute 2x2 Jacobian
        f0 = f(0, [x, y])
        fx = f(0, [x + eps, y])
        fy = f(0, [x, y + eps])

        a = (fx[0] - f0[0]) / eps  # df1/dx
        b = (fy[0] - f0[0]) / eps  # df1/dy
        c = (fx[1] - f0[1]) / eps  # df2/dx
        d = (fy[1] - f0[1]) / eps  # df2/dy

        # Eigenvalues of [[a,b],[c,d]]
        tr = a + d
        det = a * d - b * c
        disc = tr * tr - 4 * det

        if det < 0:
            return 'saddle'
        elif disc >= 0:
            # Real eigenvalues
            lam1 = (tr + math.sqrt(disc)) / 2
            lam2 = (tr - math.sqrt(disc)) / 2
            if lam1 < 0 and lam2 < 0:
                return 'stable_node'
            elif lam1 > 0 and lam2 > 0:
                return 'unstable_node'
            else:
                return 'saddle'
        else:
            # Complex eigenvalues
            if abs(tr) < 1e-10:
                return 'center'
            elif tr < 0:
                return 'stable_spiral'
            else:
                return 'unstable_spiral'

    @staticmethod
    def _newton_2d(f, y0, tol, max_iter):
        """Newton's method for finding f(0, y) = 0 in 2D."""
        y = list(y0)
        eps = 1e-7
        for _ in range(max_iter):
            fy = f(0, y)
            if V.norm(fy) < tol:
                return y

            # Jacobian
            fx0 = f(0, [y[0] + eps, y[1]])
            fy0 = f(0, [y[0], y[1] + eps])
            J = [
                [(fx0[0] - fy[0]) / eps, (fy0[0] - fy[0]) / eps],
                [(fx0[1] - fy[1]) / eps, (fy0[1] - fy[1]) / eps],
            ]
            det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
            if abs(det) < 1e-15:
                return None
            inv = [
                [J[1][1] / det, -J[0][1] / det],
                [-J[1][0] / det, J[0][0] / det],
            ]
            dy = [
                inv[0][0] * (-fy[0]) + inv[0][1] * (-fy[1]),
                inv[1][0] * (-fy[0]) + inv[1][1] * (-fy[1]),
            ]
            y = [y[0] + dy[0], y[1] + dy[1]]

        return None


# ---------------------------------------------------------------------------
# Energy analysis for Hamiltonian / conservative systems
# ---------------------------------------------------------------------------

def compute_energy_drift(result, kinetic_fn, potential_fn):
    """
    Compute energy over time for a Hamiltonian system.
    result: ODEResult with state = [q..., p...]
    kinetic_fn(p) -> T
    potential_fn(q) -> V
    Returns (energies, max_drift) where drift = max|E(t) - E(0)|/|E(0)|
    """
    dim = len(result.y[0]) // 2
    energies = []
    for state in result.y:
        q = state[:dim]
        p = state[dim:]
        E = kinetic_fn(p) + potential_fn(q)
        energies.append(E)

    E0 = energies[0]
    if abs(E0) < 1e-15:
        max_drift = max(abs(E - E0) for E in energies)
    else:
        max_drift = max(abs(E - E0) / abs(E0) for E in energies)

    return energies, max_drift


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def convergence_order(f, t_span, y0, method='rk4', reference_steps=10000,
                      step_counts=None):
    """
    Estimate convergence order by solving at different resolutions.
    Returns list of (n_steps, error, estimated_order).
    """
    if step_counts is None:
        step_counts = [50, 100, 200, 400, 800]

    # Reference solution
    ref = solve_ode(f, t_span, y0, method=method, n_steps=reference_steps)
    y_ref = ref.y[-1]

    results = []
    prev_err = None
    prev_h = None

    for n in step_counts:
        res = solve_ode(f, t_span, y0, method=method, n_steps=n)
        y_final = res.y[-1]
        err = V.norm(V.sub(y_final, y_ref))
        h = (t_span[1] - t_span[0]) / n

        order = None
        if prev_err is not None and err > 0 and prev_err > 0:
            order = math.log(prev_err / err) / math.log(prev_h / h)

        results.append((n, err, order))
        prev_err = err
        prev_h = h

    return results
