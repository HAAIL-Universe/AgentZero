"""
C132: Linear Algebra
LU, QR, SVD decompositions, eigenvalues, linear solvers, matrix operations.
Foundational numerical computing for the scientific stack.
"""

import math
import copy


# ============================================================
# Matrix class
# ============================================================

class Matrix:
    """Dense matrix with float entries."""

    def __init__(self, data):
        if isinstance(data, int):
            # Zero matrix of size data x data
            self.rows = data
            self.cols = data
            self.data = [[0.0] * data for _ in range(data)]
        elif isinstance(data, list):
            self.rows = len(data)
            self.cols = len(data[0]) if data else 0
            self.data = [[float(x) for x in row] for row in data]
        else:
            raise ValueError("Matrix data must be list of lists or int")

    @staticmethod
    def identity(n):
        m = Matrix(n)
        for i in range(n):
            m.data[i][i] = 1.0
        return m

    @staticmethod
    def zeros(rows, cols):
        m = Matrix(rows)
        m.rows = rows
        m.cols = cols
        m.data = [[0.0] * cols for _ in range(rows)]
        return m

    @staticmethod
    def from_vector(v):
        """Column vector from list."""
        return Matrix([[x] for x in v])

    def to_vector(self):
        """Extract column vector as list."""
        if self.cols != 1:
            raise ValueError("Not a column vector")
        return [self.data[i][0] for i in range(self.rows)]

    def copy(self):
        return Matrix([row[:] for row in self.data])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self.data[i][j]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            self.data[i][j] = float(value)
        else:
            self.data[key] = [float(x) for x in value]

    def __repr__(self):
        return f"Matrix({self.data})"

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return self.rows == other.rows and self.cols == other.cols and self.data == other.data

    def approx_eq(self, other, tol=1e-9):
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.data[i][j] - other.data[i][j]) > tol:
                    return False
        return True

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def T(self):
        """Transpose."""
        result = Matrix.zeros(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Dimension mismatch")
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Dimension mismatch")
        result = Matrix.zeros(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = Matrix.zeros(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result
        if isinstance(other, Matrix):
            return self.matmul(other)
        raise TypeError(f"Cannot multiply Matrix by {type(other)}")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        raise TypeError(f"Cannot multiply {type(other)} by Matrix")

    def __neg__(self):
        return self * (-1.0)

    def matmul(self, other):
        if self.cols != other.rows:
            raise ValueError(f"Cannot multiply {self.rows}x{self.cols} by {other.rows}x{other.cols}")
        result = Matrix.zeros(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self.data[i][k] * other.data[k][j]
                result.data[i][j] = s
        return result

    def col(self, j):
        """Extract column j as a list."""
        return [self.data[i][j] for i in range(self.rows)]

    def set_col(self, j, v):
        """Set column j from a list."""
        for i in range(self.rows):
            self.data[i][j] = float(v[i])

    def row(self, i):
        """Extract row i as a list."""
        return self.data[i][:]

    def submatrix(self, row_start, row_end, col_start, col_end):
        """Extract submatrix [row_start:row_end, col_start:col_end]."""
        result = Matrix.zeros(row_end - row_start, col_end - col_start)
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                result.data[i - row_start][j - col_start] = self.data[i][j]
        return result

    def trace(self):
        n = min(self.rows, self.cols)
        return sum(self.data[i][i] for i in range(n))

    def frobenius_norm(self):
        s = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                s += self.data[i][j] ** 2
        return math.sqrt(s)

    def is_square(self):
        return self.rows == self.cols

    def is_symmetric(self, tol=1e-10):
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if abs(self.data[i][j] - self.data[j][i]) > tol:
                    return False
        return True

    def is_upper_triangular(self, tol=1e-10):
        for i in range(self.rows):
            for j in range(min(i, self.cols)):
                if abs(self.data[i][j]) > tol:
                    return False
        return True

    def is_lower_triangular(self, tol=1e-10):
        for i in range(self.rows):
            for j in range(i + 1, self.cols):
                if abs(self.data[i][j]) > tol:
                    return False
        return True

    def is_orthogonal(self, tol=1e-9):
        if not self.is_square():
            return False
        product = self.T * self
        identity = Matrix.identity(self.rows)
        return product.approx_eq(identity, tol)

    def diag(self):
        """Extract diagonal as list."""
        n = min(self.rows, self.cols)
        return [self.data[i][i] for i in range(n)]

    @staticmethod
    def diag_matrix(values):
        """Create diagonal matrix from list."""
        n = len(values)
        m = Matrix(n)
        for i in range(n):
            m.data[i][i] = float(values[i])
        return m


# ============================================================
# Vector operations (as lists)
# ============================================================

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))

def vec_scale(v, s):
    return [x * s for x in v]

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def vec_normalize(v):
    n = vec_norm(v)
    if n < 1e-15:
        return v[:]
    return vec_scale(v, 1.0 / n)


# ============================================================
# LU Decomposition (partial pivoting)
# ============================================================

def lu_decompose(A):
    """
    PA = LU decomposition with partial pivoting.
    Returns (L, U, P) where P is a permutation matrix.
    """
    if not A.is_square():
        raise ValueError("LU decomposition requires square matrix")
    n = A.rows
    U = A.copy()
    L = Matrix.identity(n)
    perm = list(range(n))

    for k in range(n):
        # Partial pivoting: find max in column
        max_val = abs(U.data[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(U.data[i][k]) > max_val:
                max_val = abs(U.data[i][k])
                max_row = i

        if max_row != k:
            # Swap rows in U
            U.data[k], U.data[max_row] = U.data[max_row], U.data[k]
            perm[k], perm[max_row] = perm[max_row], perm[k]
            # Swap rows in L (only the part already computed)
            for j in range(k):
                L.data[k][j], L.data[max_row][j] = L.data[max_row][j], L.data[k][j]

        if abs(U.data[k][k]) < 1e-15:
            continue  # Singular, skip

        for i in range(k + 1, n):
            factor = U.data[i][k] / U.data[k][k]
            L.data[i][k] = factor
            for j in range(k, n):
                U.data[i][j] -= factor * U.data[k][j]

    # Build permutation matrix
    P = Matrix.zeros(n, n)
    for i in range(n):
        P.data[i][perm[i]] = 1.0

    return L, U, P


def lu_solve(A, b):
    """Solve Ax = b using LU decomposition. b is a list."""
    L, U, P = lu_decompose(A)
    n = A.rows

    # Apply permutation: Pb
    pb = [0.0] * n
    for i in range(n):
        for j in range(n):
            if P.data[i][j] == 1.0:
                pb[i] = b[j]
                break

    # Forward substitution: Ly = Pb
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i]
        for j in range(i):
            y[i] -= L.data[i][j] * y[j]

    # Back substitution: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U.data[i][j] * x[j]
        if abs(U.data[i][i]) < 1e-15:
            raise ValueError("Singular matrix")
        x[i] /= U.data[i][i]

    return x


def determinant(A):
    """Compute determinant via LU decomposition."""
    if not A.is_square():
        raise ValueError("Determinant requires square matrix")
    L, U, P = lu_decompose(A)
    # det(A) = det(P^-1) * det(L) * det(U)
    # det(L) = 1 (unit lower triangular)
    # det(U) = product of diagonal
    # det(P) = (-1)^(number of swaps)
    det_u = 1.0
    for i in range(A.rows):
        det_u *= U.data[i][i]

    # Count swaps from permutation
    perm = [0] * A.rows
    for i in range(A.rows):
        for j in range(A.rows):
            if P.data[i][j] == 1.0:
                perm[i] = j
                break

    # Count inversions
    swaps = 0
    visited = [False] * A.rows
    for i in range(A.rows):
        if visited[i] or perm[i] == i:
            visited[i] = True
            continue
        cycle_len = 0
        j = i
        while not visited[j]:
            visited[j] = True
            j = perm[j]
            cycle_len += 1
        swaps += cycle_len - 1

    sign = (-1) ** swaps
    return sign * det_u


def matrix_inverse(A):
    """Compute inverse via LU decomposition."""
    if not A.is_square():
        raise ValueError("Inverse requires square matrix")
    n = A.rows
    result = Matrix.zeros(n, n)
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        col = lu_solve(A, e)
        for i in range(n):
            result.data[i][j] = col[i]
    return result


# ============================================================
# QR Decomposition (Householder reflections)
# ============================================================

def _householder_vector(x):
    """Compute Householder vector v such that (I - 2vv^T)x = ||x||e1."""
    n = len(x)
    sigma = sum(x[i] ** 2 for i in range(1, n))
    v = x[:]
    if sigma < 1e-30 and x[0] >= 0:
        return v, 0.0
    if sigma < 1e-30 and x[0] < 0:
        v[0] = -x[0]
        return v, 2.0

    norm_x = math.sqrt(x[0] ** 2 + sigma)
    if x[0] <= 0:
        v[0] = x[0] - norm_x
    else:
        v[0] = -sigma / (x[0] + norm_x)

    tau = 2.0 * v[0] ** 2 / (sigma + v[0] ** 2)
    scale = v[0]
    if abs(scale) > 1e-15:
        v = [vi / scale for vi in v]
    return v, tau


def qr_decompose(A):
    """
    QR decomposition using Householder reflections.
    Returns (Q, R) where Q is orthogonal and R is upper triangular.
    Works for m x n matrices where m >= n.
    """
    m, n = A.rows, A.cols
    R = A.copy()
    Q = Matrix.identity(m)

    for k in range(min(m - 1, n)):
        # Extract column below diagonal
        x = [R.data[i][k] for i in range(k, m)]
        v, tau = _householder_vector(x)

        if tau == 0.0:
            continue

        # Apply reflection to R[k:, k:]
        for j in range(k, n):
            col_j = [R.data[i][j] for i in range(k, m)]
            d = dot(v, col_j)
            for i in range(m - k):
                R.data[i + k][j] -= tau * v[i] * d

        # Apply reflection to Q[:, k:]
        for j in range(m):
            row_j = [Q.data[j][i] for i in range(k, m)]
            d = dot(v, row_j)
            for i in range(m - k):
                Q.data[j][i + k] -= tau * v[i] * d

    return Q, R


def qr_solve(A, b):
    """Solve Ax = b using QR decomposition. b is a list."""
    Q, R = qr_decompose(A)
    m, n = A.rows, A.cols

    # Q^T b
    qtb = [0.0] * m
    for i in range(m):
        for j in range(m):
            qtb[i] += Q.data[j][i] * b[j]

    # Back substitution on R
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = qtb[i]
        for j in range(i + 1, n):
            x[i] -= R.data[i][j] * x[j]
        if abs(R.data[i][i]) < 1e-15:
            raise ValueError("Singular or rank-deficient matrix")
        x[i] /= R.data[i][i]

    return x


# ============================================================
# Cholesky Decomposition (for symmetric positive definite)
# ============================================================

def cholesky(A):
    """
    Cholesky decomposition: A = L L^T where L is lower triangular.
    A must be symmetric positive definite.
    """
    if not A.is_square():
        raise ValueError("Cholesky requires square matrix")
    if not A.is_symmetric():
        raise ValueError("Cholesky requires symmetric matrix")
    n = A.rows
    L = Matrix.zeros(n, n)

    for i in range(n):
        for j in range(i + 1):
            s = sum(L.data[i][k] * L.data[j][k] for k in range(j))
            if i == j:
                val = A.data[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L.data[i][j] = math.sqrt(val)
            else:
                L.data[i][j] = (A.data[i][j] - s) / L.data[j][j]

    return L


def cholesky_solve(A, b):
    """Solve Ax = b using Cholesky decomposition. A must be SPD."""
    L = cholesky(A)
    n = A.rows

    # Forward substitution: Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L.data[i][j] * y[j]
        y[i] /= L.data[i][i]

    # Back substitution: L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= L.data[j][i] * x[j]
        x[i] /= L.data[i][i]

    return x


# ============================================================
# SVD (via eigendecomposition of A^T A)
# ============================================================

def _givens_rotation(a, b):
    """Compute Givens rotation parameters c, s such that [c s; -s c]^T [a; b] = [r; 0]."""
    if abs(b) < 1e-15:
        return 1.0, 0.0
    if abs(a) < 1e-15:
        return 0.0, 1.0 if b > 0 else -1.0
    if abs(b) > abs(a):
        t = a / b
        s = 1.0 / math.sqrt(1 + t * t)
        c = s * t
    else:
        t = b / a
        c = 1.0 / math.sqrt(1 + t * t)
        s = c * t
    return c, s


def _symmetric_eigen(M, max_iter=300):
    """Eigendecomposition of symmetric matrix using Jacobi rotations.
    Returns (eigenvalues, eigenvectors_as_columns_matrix)."""
    n = M.rows
    A = M.copy()
    V = Matrix.identity(n)

    for _ in range(max_iter * n * n):
        # Find largest off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A.data[i][j]) > max_val:
                    max_val = abs(A.data[i][j])
                    p, q = i, j

        if max_val < 1e-12:
            break

        # Compute Jacobi rotation
        app = A.data[p][p]
        aqq = A.data[q][q]
        apq = A.data[p][q]

        if abs(app - aqq) < 1e-30:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2.0 * apq, app - aqq)

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply rotation to A: A' = G^T A G
        for i in range(n):
            if i == p or i == q:
                continue
            aip = A.data[i][p]
            aiq = A.data[i][q]
            A.data[i][p] = c * aip + s * aiq
            A.data[p][i] = A.data[i][p]
            A.data[i][q] = -s * aip + c * aiq
            A.data[q][i] = A.data[i][q]

        new_pp = c * c * app + 2 * s * c * apq + s * s * aqq
        new_qq = s * s * app - 2 * s * c * apq + c * c * aqq
        A.data[p][p] = new_pp
        A.data[q][q] = new_qq
        A.data[p][q] = 0.0
        A.data[q][p] = 0.0

        # Accumulate eigenvectors
        for i in range(n):
            vip = V.data[i][p]
            viq = V.data[i][q]
            V.data[i][p] = c * vip + s * viq
            V.data[i][q] = -s * vip + c * viq

    eigs = [A.data[i][i] for i in range(n)]
    return eigs, V


def svd(A, max_iter=200):
    """
    Compute SVD: A = U * S * V^T
    Returns (U, S, V) where S is diagonal matrix of singular values.
    Uses eigendecomposition of A^T A for V, then computes U = A V S^{-1}.
    """
    m, n = A.rows, A.cols

    if m < n:
        V2, S, U2 = svd(A.T, max_iter)
        return U2, S, V2

    # A^T A is n x n symmetric positive semi-definite
    AtA = A.T * A

    # Eigendecompose A^T A -> V D V^T
    eigs, V = _symmetric_eigen(AtA)

    # Sort by descending eigenvalue
    indices = sorted(range(n), key=lambda i: -eigs[i])
    singular_values = [math.sqrt(max(0, eigs[i])) for i in indices]

    V_sorted = Matrix.zeros(n, n)
    for new_idx, old_idx in enumerate(indices):
        for i in range(n):
            V_sorted.data[i][new_idx] = V.data[i][old_idx]

    # Compute U: u_i = (1/sigma_i) * A * v_i
    U = Matrix.zeros(m, n)
    for j in range(n):
        if singular_values[j] > 1e-12:
            v_col = V_sorted.col(j)
            # Av = A * v_col
            av = [0.0] * m
            for i in range(m):
                for k in range(n):
                    av[i] += A.data[i][k] * v_col[k]
            inv_s = 1.0 / singular_values[j]
            for i in range(m):
                U.data[i][j] = av[i] * inv_s

    # Thin SVD: U is m x n, S is n x n, V is n x n
    # So U * S * V^T = (m x n) * (n x n) * (n x n) = m x n = A
    S = Matrix.zeros(n, n)
    for i in range(n):
        S.data[i][i] = singular_values[i]

    return U, S, V_sorted


# ============================================================
# Eigenvalue decomposition (QR algorithm with shifts)
# ============================================================

def _hessenberg(A):
    """Reduce A to upper Hessenberg form: H = Q^T A Q."""
    if not A.is_square():
        raise ValueError("Hessenberg reduction requires square matrix")
    n = A.rows
    H = A.copy()
    Q = Matrix.identity(n)

    for k in range(n - 2):
        x = [H.data[i][k] for i in range(k + 1, n)]
        v, tau = _householder_vector(x)
        if tau == 0.0:
            continue

        # H[k+1:, k:] -= tau * v * (v^T * H[k+1:, k:])
        for j in range(k, n):
            col = [H.data[i][j] for i in range(k + 1, n)]
            d = dot(v, col)
            for i in range(n - k - 1):
                H.data[i + k + 1][j] -= tau * v[i] * d

        # H[:, k+1:] -= tau * (H[:, k+1:] * v) * v^T
        for i in range(n):
            row = [H.data[i][j] for j in range(k + 1, n)]
            d = dot(row, v)
            for j in range(n - k - 1):
                H.data[i][j + k + 1] -= tau * v[j] * d

        # Accumulate Q
        for j in range(n):
            row = [Q.data[j][i] for i in range(k + 1, n)]
            d = dot(v, row)
            for i in range(n - k - 1):
                Q.data[j][i + k + 1] -= tau * v[i] * d

    return H, Q


def eigenvalues(A, max_iter=300):
    """
    Compute eigenvalues using QR algorithm with implicit shifts.
    Returns list of eigenvalues (real parts only for now).
    For symmetric matrices, all eigenvalues are real.
    """
    if not A.is_square():
        raise ValueError("Eigenvalues require square matrix")
    n = A.rows
    H, _ = _hessenberg(A)

    # QR iteration with Wilkinson shift
    size = n
    eigs = []

    for _ in range(max_iter * n):
        if size <= 0:
            break
        if size == 1:
            eigs.append(H.data[0][0])
            break

        # Check for convergence of H[size-1, size-2]
        if abs(H.data[size - 1][size - 2]) <= 1e-12 * (abs(H.data[size - 1][size - 1]) + abs(H.data[size - 2][size - 2]) + 1e-30):
            eigs.append(H.data[size - 1][size - 1])
            size -= 1
            continue

        # Check for 2x2 block
        if size == 2:
            a, b = H.data[0][0], H.data[0][1]
            c, d = H.data[1][0], H.data[1][1]
            tr = a + d
            det_val = a * d - b * c
            disc = tr * tr - 4 * det_val
            if disc >= 0:
                eigs.append((tr + math.sqrt(disc)) / 2)
                eigs.append((tr - math.sqrt(disc)) / 2)
            else:
                # Complex eigenvalues -- return real parts
                eigs.append(tr / 2)
                eigs.append(tr / 2)
            break

        # Wilkinson shift
        a = H.data[size - 2][size - 2]
        b = H.data[size - 2][size - 1]
        c = H.data[size - 1][size - 2]
        d = H.data[size - 1][size - 1]
        tr = a + d
        det_val = a * d - b * c
        disc = tr * tr - 4 * det_val
        if disc >= 0:
            s1 = (tr + math.sqrt(disc)) / 2
            s2 = (tr - math.sqrt(disc)) / 2
            shift = s1 if abs(s1 - d) < abs(s2 - d) else s2
        else:
            shift = d  # Use corner element

        # Shifted QR step on H[0:size, 0:size]
        # Subtract shift
        for i in range(size):
            H.data[i][i] -= shift

        # QR decomposition via Givens rotations
        cs = []
        for i in range(size - 1):
            co, si = _givens_rotation(H.data[i][i], H.data[i + 1][i])
            cs.append((co, si))
            for j in range(size):
                t1 = H.data[i][j]
                t2 = H.data[i + 1][j]
                H.data[i][j] = co * t1 + si * t2
                H.data[i + 1][j] = -si * t1 + co * t2

        # RQ
        for i in range(size - 1):
            co, si = cs[i]
            for j in range(size):
                t1 = H.data[j][i]
                t2 = H.data[j][i + 1]
                H.data[j][i] = co * t1 + si * t2
                H.data[j][i + 1] = -si * t1 + co * t2

        # Add shift back
        for i in range(size):
            H.data[i][i] += shift

    return sorted(eigs, reverse=True)


def eigen_decompose(A, max_iter=300):
    """
    Eigenvalue decomposition for symmetric matrices: A = V * D * V^T
    Returns (eigenvalues, V) where V columns are eigenvectors.
    """
    if not A.is_symmetric():
        raise ValueError("eigen_decompose requires symmetric matrix")
    n = A.rows

    # Use QR algorithm accumulating transformations
    H, Q_hess = _hessenberg(A)

    # QR iteration accumulating Q
    Q_total = Q_hess.copy()
    size = n

    for _ in range(max_iter * n):
        if size <= 1:
            break

        # Check convergence
        if abs(H.data[size - 1][size - 2]) <= 1e-12 * (abs(H.data[size - 1][size - 1]) + abs(H.data[size - 2][size - 2]) + 1e-30):
            size -= 1
            continue

        # Wilkinson shift
        a = H.data[size - 2][size - 2]
        b = H.data[size - 2][size - 1]
        c = H.data[size - 1][size - 2]
        d = H.data[size - 1][size - 1]
        tr = a + d
        det_val = a * d - b * c
        disc = tr * tr - 4 * det_val
        if disc >= 0:
            s1 = (tr + math.sqrt(disc)) / 2
            s2 = (tr - math.sqrt(disc)) / 2
            shift = s1 if abs(s1 - d) < abs(s2 - d) else s2
        else:
            shift = d

        # Shift
        for i in range(n):
            H.data[i][i] -= shift

        # QR via Givens
        cs = []
        for i in range(size - 1):
            co, si = _givens_rotation(H.data[i][i], H.data[i + 1][i])
            cs.append((co, si))
            for j in range(n):
                t1 = H.data[i][j]
                t2 = H.data[i + 1][j]
                H.data[i][j] = co * t1 + si * t2
                H.data[i + 1][j] = -si * t1 + co * t2

        # RQ
        for i in range(size - 1):
            co, si = cs[i]
            for j in range(n):
                t1 = H.data[j][i]
                t2 = H.data[j][i + 1]
                H.data[j][i] = co * t1 + si * t2
                H.data[j][i + 1] = -si * t1 + co * t2

            # Accumulate in Q_total
            for j in range(n):
                t1 = Q_total.data[j][i]
                t2 = Q_total.data[j][i + 1]
                Q_total.data[j][i] = co * t1 + si * t2
                Q_total.data[j][i + 1] = -si * t1 + co * t2

        # Unshift
        for i in range(n):
            H.data[i][i] += shift

    eigs = [H.data[i][i] for i in range(n)]
    return eigs, Q_total


# ============================================================
# Least squares (via QR)
# ============================================================

def least_squares(A, b):
    """Solve min ||Ax - b||_2 via QR decomposition."""
    return qr_solve(A, b)


# ============================================================
# Matrix rank, null space, condition number
# ============================================================

def matrix_rank(A, tol=1e-6):
    """Compute rank via SVD."""
    _, S, _ = svd(A)
    p = min(A.rows, A.cols)
    rank = 0
    for i in range(p):
        if S.data[i][i] > tol:
            rank += 1
    return rank


def null_space(A, tol=1e-10):
    """Compute null space basis vectors via SVD."""
    _, S, V = svd(A)
    p = min(A.rows, A.cols)
    basis = []
    for j in range(p, A.cols):
        basis.append(V.col(j))
    for j in range(p):
        if S.data[j][j] <= tol:
            basis.append(V.col(j))
    return basis


def condition_number(A):
    """Compute 2-norm condition number via SVD."""
    _, S, _ = svd(A)
    p = min(A.rows, A.cols)
    s_max = S.data[0][0]
    s_min = S.data[p - 1][p - 1]
    if s_min < 1e-15:
        return float('inf')
    return s_max / s_min


# ============================================================
# Matrix norms
# ============================================================

def norm_1(A):
    """1-norm (max column sum)."""
    max_sum = 0.0
    for j in range(A.cols):
        col_sum = sum(abs(A.data[i][j]) for i in range(A.rows))
        max_sum = max(max_sum, col_sum)
    return max_sum


def norm_inf(A):
    """Infinity norm (max row sum)."""
    max_sum = 0.0
    for i in range(A.rows):
        row_sum = sum(abs(A.data[i][j]) for j in range(A.cols))
        max_sum = max(max_sum, row_sum)
    return max_sum


def norm_2(A):
    """2-norm (largest singular value)."""
    _, S, _ = svd(A)
    return S.data[0][0]


# ============================================================
# Matrix power and exponential
# ============================================================

def matrix_power(A, n):
    """Compute A^n via repeated squaring."""
    if not A.is_square():
        raise ValueError("Matrix power requires square matrix")
    if n == 0:
        return Matrix.identity(A.rows)
    if n < 0:
        A = matrix_inverse(A)
        n = -n

    result = Matrix.identity(A.rows)
    base = A.copy()
    while n > 0:
        if n % 2 == 1:
            result = result * base
        base = base * base
        n //= 2
    return result


def matrix_exp(A, terms=20):
    """Matrix exponential via Taylor series: exp(A) = sum(A^k / k!)."""
    if not A.is_square():
        raise ValueError("Matrix exponential requires square matrix")
    n = A.rows
    result = Matrix.identity(n)
    term = Matrix.identity(n)
    for k in range(1, terms + 1):
        term = term * A * (1.0 / k)
        result = result + term
        # Check convergence
        if term.frobenius_norm() < 1e-15:
            break
    return result


# ============================================================
# Pseudoinverse (Moore-Penrose via SVD)
# ============================================================

def pseudoinverse(A, tol=1e-10):
    """Compute Moore-Penrose pseudoinverse via SVD."""
    U, S, V = svd(A)
    m, n = A.rows, A.cols
    p = min(m, n)

    # S is p x p (thin SVD). S^+ inverts non-zero singular values.
    S_pinv = Matrix.zeros(p, p)
    for i in range(p):
        if S.data[i][i] > tol:
            S_pinv.data[i][i] = 1.0 / S.data[i][i]

    # A^+ = V * S^+ * U^T
    # V is n x p, S_pinv is p x p, U is m x p -> U^T is p x m
    # Result: (n x p) * (p x p) * (p x m) = n x m -- correct!
    return V * S_pinv * U.T


# ============================================================
# Gram-Schmidt orthogonalization
# ============================================================

def gram_schmidt(vectors):
    """
    Classical Gram-Schmidt orthogonalization.
    Input: list of vectors (lists).
    Output: list of orthonormal vectors.
    """
    result = []
    for v in vectors:
        u = v[:]
        for q in result:
            proj = dot(u, q)
            u = vec_sub(u, vec_scale(q, proj))
        n = vec_norm(u)
        if n > 1e-12:
            result.append(vec_scale(u, 1.0 / n))
    return result


def modified_gram_schmidt(vectors):
    """
    Modified Gram-Schmidt (more numerically stable).
    """
    vecs = [v[:] for v in vectors]
    result = []
    for i in range(len(vecs)):
        n = vec_norm(vecs[i])
        if n < 1e-12:
            continue
        q = vec_scale(vecs[i], 1.0 / n)
        result.append(q)
        for j in range(i + 1, len(vecs)):
            proj = dot(vecs[j], q)
            vecs[j] = vec_sub(vecs[j], vec_scale(q, proj))
    return result


# ============================================================
# Power iteration (dominant eigenvalue)
# ============================================================

def power_iteration(A, max_iter=1000, tol=1e-10):
    """Find dominant eigenvalue and eigenvector via power iteration."""
    if not A.is_square():
        raise ValueError("Power iteration requires square matrix")
    n = A.rows
    # Start with random-ish vector
    v = [1.0 / math.sqrt(n)] * n
    v[0] = 1.0
    v = vec_normalize(v)

    eigenvalue = 0.0
    for _ in range(max_iter):
        # w = A * v
        w = [0.0] * n
        for i in range(n):
            for j in range(n):
                w[i] += A.data[i][j] * v[j]

        # Rayleigh quotient
        new_eigenvalue = dot(w, v)

        # Normalize
        v = vec_normalize(w)

        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, v
        eigenvalue = new_eigenvalue

    return eigenvalue, v


# ============================================================
# Iterative solvers
# ============================================================

def conjugate_gradient(A, b, x0=None, max_iter=1000, tol=1e-10):
    """
    Conjugate gradient method for solving Ax = b.
    A must be symmetric positive definite.
    """
    n = A.rows
    x = x0[:] if x0 else [0.0] * n

    # r = b - Ax
    Ax = [sum(A.data[i][j] * x[j] for j in range(n)) for i in range(n)]
    r = vec_sub(b, Ax)
    p = r[:]
    rs_old = dot(r, r)

    if rs_old < tol * tol:
        return x

    for _ in range(max_iter):
        # Ap = A * p
        Ap = [sum(A.data[i][j] * p[j] for j in range(n)) for i in range(n)]
        pAp = dot(p, Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp

        x = vec_add(x, vec_scale(p, alpha))
        r = vec_sub(r, vec_scale(Ap, alpha))

        rs_new = dot(r, r)
        if math.sqrt(rs_new) < tol:
            break

        p = vec_add(r, vec_scale(p, rs_new / rs_old))
        rs_old = rs_new

    return x


# ============================================================
# Matrix factorization utilities
# ============================================================

def is_positive_definite(A):
    """Check if symmetric matrix is positive definite via Cholesky."""
    if not A.is_symmetric():
        return False
    try:
        cholesky(A)
        return True
    except ValueError:
        return False
